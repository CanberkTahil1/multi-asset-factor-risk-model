"""Generates ``reports/data_contracts.md`` from the local cache.

``python -m mafrm.data.contract_report``

The weekly contract suite asserts; this reports. The two are complementary: a
test can only say pass or fail, and several of the things worth knowing about
this data layer -- how much of a FRED series ALFRED can reconstruct, how far a
declared inception is from the vendor's first printable bar, how stale each AQR
workbook is -- are numbers to be read rather than thresholds to be met. Numbers
that have no validated threshold are reported here and gated nowhere, which is
CLAUDE.md invariant 9 applied to contracts rather than to model parameters.

Never fetches. Everything comes from what ``make data`` already cached.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from mafrm.config import load
from mafrm.data import aqr, cache, contracts, fred, french, prices

__all__ = ["contract_markdown"]

_MISSING = "_not cached_"


def _latest(source: str, name: str) -> pd.DataFrame | None:
    try:
        manifest = cache.Manifest.load()
        return cache.read(manifest.latest(source=source, name=name), manifest=manifest)
    except cache.CacheError:
        return None


def _etf_rows() -> list[str]:
    universe = load().universe
    rows: list[str] = []
    for ticker in universe.yfinance_tickers:
        asset = next(a for a in universe.assets if a.ticker == ticker)
        frame = _latest("yfinance", f"{ticker.lower()}_prices")
        actions = _latest("yfinance", f"{ticker.lower()}_actions")
        if frame is None or actions is None:
            rows.append(f"| {ticker} | {asset.history_start} | {_MISSING} | | | |")
            continue
        index = pd.DatetimeIndex(frame.index)
        first = index.min().date()
        lag = (first - asset.history_start).days
        try:
            splits = prices.check_splits_applied(frame, actions)
            split_note = (
                "none" if not splits else ", ".join(f"{c.ex_date} {c.ratio:g}:1 ok" for c in splits)
            )
        except prices.PricesError as exc:  # pragma: no cover - reported, not raised
            split_note = f"**{exc}**"
        rows.append(
            f"| {ticker} | {asset.history_start} | {first} | {lag:+d} | "
            f"{len(frame):,} | {split_note} |"
        )
    return rows


def _fred_rows() -> list[str]:
    rows: list[str] = []
    for series in load().model.data.fred.series:
        try:
            report = fred.load_coverage(series.name)
        except cache.CacheError:
            rows.append(f"| {series.id} | {series.classification} | {_MISSING} | | | | |")
            continue
        rows.append(
            f"| {series.id} | {series.classification} | "
            f"{report.observation_start}..{report.observation_end} | "
            f"{report.n_observations:,} | {report.n_vintages:,} | "
            f"{report.first_vintage} | {report.fraction:.1%} |"
        )
    return rows


def _aqr_rows() -> list[str]:
    config = load().model.data.aqr
    rows: list[str] = []
    for dataset in config.datasets:
        frame = _latest(aqr.SOURCE, dataset.name)
        if frame is None:
            rows.append(f"| {dataset.name} | {_MISSING} | | | | |")
            continue
        index = pd.DatetimeIndex(frame.index)
        gaps = len(contracts.missing_periods(frame, freq="M", label=dataset.name))
        if dataset.expected_to_update:
            drift = aqr.release_drift(dataset.name)
            state = "**DRIFT**" if drift.fired else ("ok" if drift.cadence_observable else "n/a")
            observed = f"{len(drift.releases)} release(s), cadence {state}"
        else:
            observed = "static, exempt"
        rows.append(
            f"| {dataset.name} | **{index.min().date()}..{index.max().date()}** | "
            f"{len(frame):,} | {len(frame.columns)} | {gaps} | {observed} |"
        )
    return rows


def _risk_free_paragraph() -> str:
    config = load().model.data
    table = _latest(french.SOURCE, french.CACHE_NAME)
    interim = _latest("fred", config.risk_free.interim_fred.fred_series.lower())
    if table is None or interim is None:
        return "The two risk-free legs cannot be compared: one of them is not cached."

    converted = french.as_annual_percent(
        table[config.risk_free.ken_french_daily_rf.rf_column],
        trading_days_per_year=config.trading_days_per_year,
    )
    other = interim["value"].dropna()
    common = converted.index.intersection(other.index)
    difference = (converted.loc[common] - other.loc[common]).abs()
    return (
        f"Over the {len(common):,} days both legs quote "
        f"({common.min().date()}..{common.max().date()}), the converted Ken French rate and "
        f"`{config.risk_free.interim_fred.fred_series}` differ by a median of "
        f"{float(difference.median()):.3f} and a mean of {float(difference.mean()):.3f} "
        "percentage points. That is the real spread between an overnight bill return and a "
        "one-month constant-maturity yield. A dropped units conversion would show here as a "
        f"gap of order {config.trading_days_per_year}x instead."
    )


def _cross_check_lines() -> str:
    from mafrm.data import crosschecks

    lines: list[str] = []
    try:
        lines.append(f"* {crosschecks.spy_versus_market().render()}")
    except (cache.CacheError, ValueError) as exc:  # pragma: no cover - reported, not raised
        lines.append(f"* SPY vs Mkt-RF: unavailable ({exc})")
    try:
        from mafrm.data.gsw_report import cross_check_tlt

        gate = load().model.data.tlt_cross_check
        check = cross_check_tlt()
        verdict = "pass" if gate.beta_min <= check.ladder_beta <= gate.beta_max else "FAIL"
        lines.append(
            f"* TLT vs GSW par-coupon ladder: n={check.observations:,} "
            f"{check.start.date()}..{check.end.date()}; beta {check.ladder_beta:.4f} "
            f"(band [{gate.beta_min}, {gate.beta_max}]) -- {verdict}. Mean gap "
            f"{check.ladder_gap_pct:+.4f}%/yr is W1-P3's open residual and is not gated here."
        )
    except Exception as exc:  # pragma: no cover - reported, not raised
        lines.append(f"* TLT vs GSW ladder: unavailable ({exc})")
    return "\n".join(lines)


def contract_markdown() -> str:
    """The committed report. Deterministic apart from the generation timestamp."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    return f"""# Data-layer contracts

Generated by `python -m mafrm.data.contract_report` on {stamp} from the local
cache. Nothing here is fetched; run `make data` first.

Companion to `tests/test_data_contracts.py`, which asserts. This reports. Several
quantities below have no validated threshold -- how far a vendor's first bar lags
a fund's inception, how much of a FRED series ALFRED can reconstruct -- and
inventing one would breach CLAUDE.md invariant 9, so they are numbers to read
rather than gates to pass.

## 1. ETF price series (yfinance)

`history_start` is what `config/universe.yaml` declares. `first bar` is what
Yahoo actually serves. **The gap is not a defect and is not gated**: a vendor's
first printable bar can lag a fund's inception by a few sessions, and a tolerance
invented to absorb that would be fitted to the data rather than derived from
anything. What *is* gated is that the sleeve's effective common start falls in
the month SPEC.md 3.1 names, and that every member shares a trading calendar from
that point on.

The split column is the check `mafrm.data.prices` rests on: Yahoo's
`auto_adjust=False` close is already split-adjusted backwards, so no split factor
is applied, and `ok` means the close was verified continuous across the ex-date
rather than stepping by the ratio.

| ticker | history_start | first bar | lag (days) | rows | splits |
|---|---|---|---|---|---|
{chr(10).join(_etf_rows())}

## 2. FRED / ALFRED vintage coverage

SPEC.md 3.5: "Check vintage coverage per series; drop or conservatively lag
anything without vintages." Coverage is partial -- FRED began recording real-time
periods when ALFRED adopted each series, not when the series began.

**Reclassified in W1-P5, and the reclassification is the finding.** Vintages exist
to defeat look-ahead through *revision*: a statistic that is estimated, published
and then restated as source data arrives. All six macro factors in this project
are **market-observed** -- equity index returns, GSW curve principal components,
the HY excess return, the AQR commodity index and `DTWEXBGS`. A price, a yield or
an index computed from the day's quotes is not an estimate of anything
unobserved; there is nothing to restate, and the value is final when printed.

So `DTWEXBGS`'s 36.6% coverage **owes no drop-or-lag decision**, and this
sentence is here so that a later session does not reopen one. The pre-2019 gap
records when ALFRED adopted the series, not a defect in it.

`mafrm.data.loaders.check_vintage_availability` raises only for a series
classified `revised_statistic` with zero vintages. That list is currently
**empty**, which is the correct state rather than an oversight -- the machinery
stays so the first CPI or payrolls series added meets the stricter contract
without anyone having to remember why.

| series | class | observations | n | vintages | first vintage | with a first release |
|---|---|---|---|---|---|---|
{chr(10).join(_fred_rows())}

## 3. AQR research workbooks

**These are validation comparands, not production inputs.** Nothing is priced off
their most recent month, so what they owe is coverage of the window they will be
compared over -- not freshness. The W1-P4 contract asserted a 90-day age limit,
failed on all three files, and was testing the wrong property; it was replaced in
W1-P5 by coverage plus a release-drift detector.

**The observed end date is in bold because it is load-bearing.** Every downstream
comparison against one of these series must state the window it actually ran
over. "Through 2025-05" and "to present" are different claims and only one of
them is true of `commodities_long_run`.

| dataset | observed window | rows | columns | interior gaps | release cadence |
|---|---|---|---|---|---|
{chr(10).join(_aqr_rows())}

Release drift fires when a re-pull returns an unchanged end date after longer
than the longest interval between releases **already observed** for that file.
The cadence is inferred from the manifest's own release log rather than assumed,
so with fewer than two observed releases it cannot fire -- shown as `n/a` above,
which is the honest state on a first pull and the reason this is not an age
limit under a new name.

`commodities_long_run` is exempt (`expected_to_update: false`): it is a static
companion to Levine, Ooi, Richardson & Sasseville (2018), AQR has never committed
to updating it, and an unchanging end date there is correct behaviour. It is also
the only one of the three on the production path, as the pre-2006 leg of the DBC
splice (`config/universe.yaml`, `commodity`) -- and that use needs its 1877-2006
history, which is complete, not its last month.

## 4. Risk-free leg

{_risk_free_paragraph()}

## 5. Independent cross-checks -- and the seven tickers that have none

Yahoo rewrites adjusted-close history backwards on every dividend, so a series
built from it can be wrong in a way that leaves **no trace inside Yahoo**. The
only non-circular defence is a comparand from a different vendor, with a
different dividend convention, reached by a different code path. Stooq was to
have been that comparand for the whole sleeve and is unreachable, so this is what
independence the project actually has:

{_cross_check_lines()}

`SPY` is checked against Ken French's `Mkt-RF`, the CRSP value-weighted US market
excess return: not Yahoo, not our dividend handling, no shared code. `TLT` is
checked against a par-coupon ladder priced off the Fed's published Svensson
parameters, on the regression **beta** -- a question about the price series,
distinct from W1-P3's mean-gap gate, which asks whether the ladder *construction*
is right and remains open at -0.192%/yr under a strict xfail.

Every tolerance is derived from what it must discriminate, not fitted to what was
measured. The mean-gap limit of 0.50%/yr is the clearest case: SPY's own dividend
yield is 1.862%/yr, so a dropped dividend stream cannot hide inside the limit,
while the largest plausible S&P-500-versus-total-market size effect (15% of cap
times even a 3%/yr small-cap premium) is about 0.45%/yr and sits just under it.

### The seven tickers with a single source

**IWM, EFA, EEM, LQD, HYG, DBC and GLD have Yahoo as their sole price source.**
Nothing in this repository corroborates them.

*The pathology.* Yahoo recomputes adjusted-close history retroactively whenever a
distribution is paid. Two pulls on different dates return different histories for
the same past day, so any statistic computed on an adjusted series carries a
forward-looking component, and a vendor-side error in a past bar is invisible
because there is nothing to compare it against.

*The mitigation, and its limits.* Raw unadjusted OHLCV and the corporate-actions
table are cached as two separately content-hashed artefacts, so a changed
dividend shows up as a changed actions file rather than as a silent rewrite of
prices; `mafrm.data.prices.total_return` applies the adjustment itself and never
looks forward, so appending tomorrow's dividend cannot alter yesterday's return;
`adjusted_close` is stamped with the `as_of` date whose dividend table produced
it; split continuity is verified on every load rather than assumed; and the
shared-trading-calendar contract in section 1 would catch a missing or spurious
session.

Those defend against *our* mishandling of Yahoo's data and against Yahoo changing
its history under us. **None of them detects a bar that was wrong in Yahoo when we
first cached it.** For the seven, that risk is open and unquantified. The two
corroborated tickers bound it indirectly -- if Yahoo's SPY and TLT bars are sound
over 33 and 24 years respectively, a systematic feed-wide defect is less likely --
but that is an argument, not a measurement.

*Stooq stays unsolved.* The site answers every plain client, `pandas_datareader`
included, with a JavaScript proof-of-work browser challenge: find a SHA-256
preimage, POST it to `/__verify`, receive a cookie. It is perhaps ten lines of
Python. It is also an access control the site operator put up deliberately, and
this repository will be made public, so working around it is not appropriate.
`mafrm.data.stooq.parse_csv` detects the challenge specifically and raises
`StooqUnavailable`, so the failure is distinguishable from a broken parser, and
the cross-check skips with a stated reason rather than silently passing. The
loader stays wired: if Stooq serves CSV to plain clients again, `make data` picks
it up with no code change.
"""


def _main(argv: Sequence[str]) -> int:
    if list(argv) not in ([], ["report"]):
        print("usage: python -m mafrm.data.contract_report", file=sys.stderr)
        return 2
    target = Path(__file__).resolve().parents[3] / "reports" / "data_contracts.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(contract_markdown(), encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
