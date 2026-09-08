"""Contracts on what the vendors actually served. Weekly, not on every push.

These read the local cache written by ``make data``. They are marked ``dataset``
and therefore deselected by ``make test`` and the fast CI workflow -- a clean
clone has no ``data/raw`` -- and run by ``.github/workflows/data-contracts.yml``
after a refresh.

**They are expected to fail sometimes and that is the point.** A vendor renaming
a column, FRED truncating a series as ICE did in April 2026, or yfinance
shipping another breaking patch release should surface here rather than silently
inside a backtest six months from now. Each assertion therefore reports the
measured number, not just a verdict.

Where a source did not refresh at all, ``conftest.cached`` skips with the reason
instead of failing, so one unavailable vendor cannot mask a real breach in
another.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from conftest import cached
from mafrm.config import load
from mafrm.data import aqr, contracts, crosschecks, fred, french, loaders, prices, stooq

pytestmark = pytest.mark.dataset

CONFIG = load()
UNIVERSE = CONFIG.universe
DATA = CONFIG.model.data


# ---------------------------------------------------------------------------
# ETF bars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ticker", UNIVERSE.yfinance_tickers)
def test_etf_prices_satisfy_the_shape_contract(ticker: str) -> None:
    frame = cached("yfinance", f"{ticker.lower()}_prices")
    contracts.check_frame(
        frame,
        label=f"yfinance/{ticker} prices",
        required_columns=DATA.etf.price_columns,
        min_rows=100,
    )


@pytest.mark.parametrize("ticker", UNIVERSE.yfinance_tickers)
def test_etf_actions_satisfy_the_shape_contract(ticker: str) -> None:
    frame = cached("yfinance", f"{ticker.lower()}_actions")
    # `Stock Splits` is legitimately all-zero for a fund that never split, and
    # zero is not NaN, so the all-NaN contract still applies unmodified.
    contracts.check_frame(
        frame,
        label=f"yfinance/{ticker} actions",
        required_columns=DATA.etf.action_columns,
        min_rows=100,
    )


@pytest.mark.parametrize("ticker", UNIVERSE.yfinance_tickers)
def test_adj_close_is_not_cached(ticker: str) -> None:
    """SPEC.md 3.5. Yahoo's adjusted close is rewritten backwards on every
    dividend, so caching it would put a forward-looking column inside a
    content-hashed provenance record."""
    frame = cached("yfinance", f"{ticker.lower()}_prices")
    assert "Adj Close" not in frame.columns


@pytest.mark.parametrize("ticker", UNIVERSE.yfinance_tickers)
def test_the_vendor_close_is_already_split_adjusted(ticker: str) -> None:
    """The assumption ``mafrm.data.prices`` rests on, tested rather than asserted.

    Four real splits exercise this: IWM 2:1 and EFA 3:1 on 2005-06-09, EEM 3:1
    on 2005-06-09 and again on 2008-07-24. If Yahoo ever stops pre-applying
    them, the close steps by the ratio and every return through that date is
    wrong by a factor of two or three.
    """
    checks = prices.check_splits_applied(
        cached("yfinance", f"{ticker.lower()}_prices"),
        cached("yfinance", f"{ticker.lower()}_actions"),
    )
    for check in checks:
        assert check.already_applied, check.render()


def _first_bars() -> dict[str, pd.Timestamp]:
    return {
        ticker: pd.DatetimeIndex(cached("yfinance", f"{ticker.lower()}_prices").index).min()
        for ticker in UNIVERSE.yfinance_tickers
    }


def test_the_headline_window_starts_in_the_month_the_universe_names() -> None:
    """SPEC.md 3.1: "Headline window: 2007-04 to present", driven by HYG.

    Deliberately *not* an equality check against each asset's ``history_start``
    in ``config/universe.yaml``. Yahoo's first printable bar can lag a fund's
    inception -- EFA is declared at 2001-08-17 and Yahoo's first bar is
    2001-08-27; DBC at 2006-02-03 against 2006-02-06 -- and inventing a
    tolerance to absorb that would be fitting a threshold to the data.

    Nor is it "every member covers ``common_start``", which is false by
    construction: ``common_start`` is 2007-04-01 and HYG, the fund that *sets*
    it, did not print until 2007-04-11. What SPEC.md actually claims is the
    month, so that is what is checked, and the effective start -- the latest
    first bar in the sleeve -- is reported alongside.
    """
    first_bars = _first_bars()
    effective = max(first_bars.values()).date()
    declared = UNIVERSE.common_start

    assert (effective.year, effective.month) == (declared.year, declared.month), (
        f"the sleeve's effective common start is {effective}, outside the month "
        f"{declared:%Y-%m} that config/universe.yaml names"
    )
    binding = [t for t, d in first_bars.items() if d.date() == effective]
    assert binding == ["HYG"], f"the binding inception is {binding}, not HYG as documented"


def test_the_sleeve_shares_a_trading_calendar_over_the_headline_window() -> None:
    """CLAUDE.md failure mode 1, the top source of "my Sharpe is 2.4" moments.

    US-listed ETFs trade the NYSE calendar, so from the point every member is
    listed they must quote on exactly the same days. A member missing a session
    silently drops it from every covariance estimate that intersects indices,
    and a member with an *extra* session is quoting on a day the exchange was
    shut -- which is what a stale-forward-fill looks like.

    Anchored on the effective common start rather than the configured one,
    because before HYG lists there is genuinely nothing to compare.
    """
    start = max(_first_bars().values())
    calendars = {}
    for ticker in UNIVERSE.yfinance_tickers:
        index = pd.DatetimeIndex(cached("yfinance", f"{ticker.lower()}_prices").index)
        calendars[ticker] = set(index[index >= start])

    reference = calendars["SPY"]
    for ticker, days in calendars.items():
        missing = reference - days
        extra = days - reference
        assert not missing, (
            f"{ticker} is missing {len(missing)} SPY session(s) since {start.date()}, "
            f"first {min(missing).date()}"
        )
        assert not extra, (
            f"{ticker} quotes {len(extra)} session(s) SPY does not since {start.date()}, "
            f"first {min(extra).date()}"
        )


# ---------------------------------------------------------------------------
# FRED / ALFRED
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("series", DATA.fred.series, ids=lambda s: s.id)
def test_fred_series_satisfy_the_shape_contract(series: object) -> None:
    frame = cached("fred", series.name)  # type: ignore[attr-defined]
    contracts.check_frame(
        frame,
        label=f"fred/{series.id}",  # type: ignore[attr-defined]
        required_columns=["value", "realtime_start", "realtime_end"],
        min_rows=100,
    )


@pytest.mark.parametrize("series", DATA.fred.series, ids=lambda s: s.id)
def test_the_still_current_sentinel_is_preserved(series: object) -> None:
    """9999-12-31 is outside pandas' Timestamp range; coercing it loses it."""
    frame = cached("fred", f"{series.name}{fred.FIRST_RELEASE_SUFFIX}")  # type: ignore[attr-defined]
    assert frame["realtime_end"].notna().all()
    assert (frame["realtime_end"] == "9999-12-31").any(), (
        "no observation is marked still-current, which means the sentinel was "
        "coerced away rather than carried"
    )


@pytest.mark.parametrize("series", DATA.fred.revised_statistics, ids=lambda s: s.id)
def test_a_revised_statistic_has_alfred_vintages(series: object) -> None:
    """SPEC.md 3.5: without vintages every value enters at its latest revision.

    Parametrised over an EMPTY list today, and that is the correct state: every
    macro factor in this project is market-observed, so nothing is subject to
    look-ahead through revision. The test exists so that the first CPI or
    payrolls series added to the config meets the contract without anyone having
    to remember it.
    """
    vintages = fred.load_vintage_dates(series.name)  # type: ignore[attr-defined]
    assert vintages, f"{series.id} has no ALFRED vintages"  # type: ignore[attr-defined]


def test_no_market_observed_series_is_required_to_have_vintages() -> None:
    """The reclassification, asserted rather than left as a comment.

    A price, a yield or an index computed from the day's quotes is not an
    estimate of anything unobserved, so there is nothing to restate and partial
    vintage coverage carries no drop-or-lag decision. DTWEXBGS sits at 36.6%
    coverage for that reason and it is not a defect.
    """
    assert DATA.fred.market_observed, "the config classifies nothing as market-observed"
    for series in DATA.fred.market_observed:
        assert not series.requires_vintages
    # And the loader agrees: check_vintage_availability raises only for the
    # stricter class, so this returns rather than raising on partial coverage.
    reports = loaders.check_vintage_availability()
    assert len(reports) == len(DATA.fred.series)


@pytest.mark.parametrize("series", DATA.fred.series, ids=lambda s: s.id)
def test_the_first_release_is_never_later_than_the_observation(series: object) -> None:
    """A value cannot be published before the day it describes.

    The one check that catches a mis-stitched chunk boundary: taking the wrong
    row per date would date a 2005 observation to a 2013 release, which is
    look-ahead in the flattering direction and would not otherwise raise.
    """
    frame = cached("fred", f"{series.name}{fred.FIRST_RELEASE_SUFFIX}")  # type: ignore[attr-defined]
    observed = pd.DatetimeIndex(frame.index)
    released = pd.DatetimeIndex(frame["realtime_start"])
    early = (released < observed).sum()
    assert early == 0, f"{early} observation(s) claim to have been released before their own date"


@pytest.mark.parametrize("series", DATA.fred.series, ids=lambda s: s.id)
def test_vintage_coverage_is_reported(series: object, capsys: pytest.CaptureFixture[str]) -> None:
    """Partial coverage is normal, and for a market-observed series it is inert.

    DTWEXBGS observations begin 2006 but ALFRED's first vintage is 2019, so 63%
    of its history has no first-release value. **That owes no decision.** The
    index is computed from the day's observed exchange rates; it is not an
    estimate that gets restated, so there is no revision to be ahead of.
    SPEC.md 3.5's "drop or conservatively lag" applies to revised statistics,
    of which this project currently has none. Recorded here so a later session
    does not reopen it.
    """
    report = fred.load_coverage(series.name)  # type: ignore[attr-defined]
    with capsys.disabled():
        print(f"\n  {report.render()}")
    assert report.has_vintages
    assert 0.0 < report.fraction <= 1.0


# ---------------------------------------------------------------------------
# Ken French
# ---------------------------------------------------------------------------


def test_the_ken_french_table_satisfies_the_shape_contract() -> None:
    frame = cached(french.SOURCE, french.CACHE_NAME)
    contracts.check_frame(
        frame,
        label="ken_french/daily_factors",
        required_columns=["Mkt-RF", "SMB", "HML", DATA.risk_free.ken_french_daily_rf.rf_column],
        min_rows=20_000,
    )


def test_the_risk_free_leg_spans_the_whole_curve_sample() -> None:
    """The reason this leg was selected over DGS1MO: no splice (SPEC.md 3.1).

    The GSW curve starts 1961-06-14. A short rate starting 2001 leaves 40 years
    of the sample with total returns and no excess returns.
    """
    frame = cached(french.SOURCE, french.CACHE_NAME)
    first = pd.DatetimeIndex(frame.index).min().date()
    assert first <= date(1961, 6, 14), f"the risk-free leg starts {first}, after the curve does"


def test_the_two_risk_free_legs_agree_where_they_overlap() -> None:
    """The units check, on real data.

    Ken French publishes percent per trading day and DGS1MO percent per annum --
    a factor of about 252. If the conversion were dropped the two would differ by
    that factor rather than by the small, real spread between an overnight bill
    return and a one-month constant-maturity yield.
    """
    kf = french.as_annual_percent(
        cached(french.SOURCE, french.CACHE_NAME)[DATA.risk_free.ken_french_daily_rf.rf_column],
        trading_days_per_year=DATA.trading_days_per_year,
    )
    dgs = cached("fred", "dgs1mo")["value"].dropna()

    common = kf.index.intersection(dgs.index)
    assert len(common) > 1_000
    difference = (kf.loc[common] - dgs.loc[common]).abs()

    assert float(difference.median()) < 1.0, (
        f"median |Ken French - DGS1MO| is {float(difference.median()):.3f} percentage points; "
        "a gap of this size means the percent-per-day to percent-per-annum conversion is wrong"
    )


# ---------------------------------------------------------------------------
# AQR
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", DATA.aqr.datasets, ids=lambda d: d.name)
def test_aqr_workbooks_satisfy_the_shape_contract(dataset: object) -> None:
    frame = cached(aqr.SOURCE, dataset.name)  # type: ignore[attr-defined]
    contracts.check_frame(
        frame,
        label=f"aqr/{dataset.name}",  # type: ignore[attr-defined]
        required_columns=dataset.required_columns,  # type: ignore[attr-defined]
        min_rows=100,
    )


@pytest.mark.parametrize("dataset", DATA.aqr.datasets, ids=lambda d: d.name)
def test_aqr_series_cover_the_in_sample_window(
    dataset: object, capsys: pytest.CaptureFixture[str]
) -> None:
    """Coverage, not freshness. Replaces the W1-P4 90-day age check.

    AQR data is a validation comparand for W2-P3 and the pre-2006 leg of one
    splice. Nothing is priced off its most recent month, so what it owes is not
    recency but that it spans the window it will be compared over, with no gap
    in the middle. The observed end date is printed and recorded in
    reports/data_contracts.md, so every downstream comparison can state the
    window it actually ran over rather than implying "to present".
    """
    frame = cached(aqr.SOURCE, dataset.name)  # type: ignore[attr-defined]
    index = pd.DatetimeIndex(frame.index)
    with capsys.disabled():
        print(f"\n  aqr/{dataset.name}: {index.min().date()}..{index.max().date()}")  # type: ignore[attr-defined]
    contracts.check_covers(
        frame,
        label=f"aqr/{dataset.name}",  # type: ignore[attr-defined]
        freq="M",
        start=CONFIG.model.sample.start,
        end=CONFIG.model.sample.holdout_start,
    )


def test_the_holdout_boundary_cannot_be_checked_until_it_is_pinned() -> None:
    """The upper half of the coverage contract, and why it is not enforced yet.

    ``check_covers`` takes ``end=sample.holdout_start`` and skips that assertion
    while it is ``None``. That is deliberate rather than lenient: SPEC.md 9 gives
    a range and not a date, the value cannot be revised once spent (CLAUDE.md
    invariant 5), and inventing one here to make a contract enforceable would be
    the most expensive guess in the repository. This test fails the day the
    holdout is pinned without the coverage contract being re-run.
    """
    if CONFIG.model.sample.holdout_start is None:
        pytest.skip(
            "sample.holdout_start is unset, so the end of the in-sample window is "
            "undefined. Coverage is enforced from sample.start and for interior gaps; "
            "the upper bound becomes enforceable when a human pins the date."
        )
    for dataset in DATA.aqr.datasets:
        frame = cached(aqr.SOURCE, dataset.name)
        last = pd.DatetimeIndex(frame.index).max().date()
        assert last >= CONFIG.model.sample.holdout_start, (
            f"aqr/{dataset.name} ends {last}, before the in-sample window closes"
        )


@pytest.mark.parametrize("dataset", DATA.aqr.maintained, ids=lambda d: d.name)
def test_maintained_aqr_files_have_not_stopped_publishing(
    dataset: object, capsys: pytest.CaptureFixture[str]
) -> None:
    """Release drift, judged against the file's own observed history.

    Fires when a re-pull returns an unchanged end date after longer than the
    longest interval between releases already observed. The cadence is inferred
    from the manifest's release log rather than assumed, so with fewer than two
    observed releases it cannot fire -- which is the honest state today, and the
    reason this replaces an age limit rather than renaming one.

    ``commodities_long_run`` is excluded by ``expected_to_update: false``: it is
    a static companion to a 2018 paper, and an unchanging end date there is the
    expected behaviour, not a signal.
    """
    drift = aqr.release_drift(dataset.name)  # type: ignore[attr-defined]
    with capsys.disabled():
        print(f"\n  {drift.render()}")
    assert not drift.fired, drift.render()


def test_the_static_dataset_is_exempt_from_drift() -> None:
    """Stated as an assertion so the exemption cannot be lost in a config edit."""
    exempt = [d.name for d in DATA.aqr.datasets if not d.expected_to_update]
    assert exempt == ["commodities_long_run"]


def test_the_commodity_splice_leg_reaches_back_to_1877() -> None:
    """The only AQR file on the production path (config/universe.yaml, `commodity`)."""
    frame = cached(aqr.SOURCE, "commodities_long_run")
    first = pd.DatetimeIndex(frame.index).min().date()
    assert first <= date(1877, 12, 31), f"the long-run commodity index starts {first}"


def test_the_commodity_splice_leg_covers_dbcs_inception() -> None:
    """A splice needs an overlap to be a splice rather than a concatenation."""
    frame = cached(aqr.SOURCE, "commodities_long_run")
    last = pd.DatetimeIndex(frame.index).max().date()
    assert last >= UNIVERSE.by_id("commodity").history_start


# ---------------------------------------------------------------------------
# Independent cross-checks -- the corroboration the sleeve actually has
# ---------------------------------------------------------------------------


def test_spy_agrees_with_the_crsp_market_return(capsys: pytest.CaptureFixture[str]) -> None:
    """Our SPY total return against Ken French's Mkt-RF. Fully independent.

    The point of this test is what it would catch. Our series is built from raw
    Yahoo bars by our own dividend handling; Ken French's comes from CRSP with
    CRSP's conventions, and the two share no vendor and no code. A retroactive
    back-adjustment error, a dropped dividend stream or a mishandled bar in the
    Yahoo leg cannot be present in the CRSP leg, so it shows up here.

    Every tolerance is derived from what it must discriminate rather than fitted
    to what was measured -- see ``data.cross_checks`` in config/model.yaml. The
    mean-gap limit in particular sits below a third of SPY's own dividend yield,
    so a dropped dividend stream cannot hide inside it.
    """
    limits = DATA.cross_checks.spy_versus_market
    check = crosschecks.spy_versus_market()
    with capsys.disabled():
        print(f"\n  {check.render()}")

    assert check.observations > 5_000
    assert check.correlation >= limits.min_correlation, check.render()
    assert abs(check.mean_gap_pct_per_year) <= limits.max_abs_mean_gap_pct_per_year, check.render()
    assert check.max_abs_daily_pct <= limits.max_abs_daily_difference_pct, check.render()


def test_the_spy_gap_is_far_smaller_than_a_dropped_dividend_stream() -> None:
    """The separation argument the mean-gap tolerance rests on.

    A tolerance is only worth having if the thing it must catch and the thing it
    must ignore are far apart. Here they are: a dropped dividend stream would
    show as SPY's dividend yield, and the measured gap is an order of magnitude
    below it. This asserts the separation itself rather than the limit, so it
    keeps its meaning if the limit is ever revisited.
    """
    check = crosschecks.spy_versus_market()
    assert abs(check.mean_gap_pct_per_year) < check.dividend_yield_pct_per_year / 3.0, (
        f"the measured gap {check.mean_gap_pct_per_year:+.4f}%/yr is within a third of "
        f"SPY's {check.dividend_yield_pct_per_year:.3f}%/yr dividend yield, so this "
        "cross-check can no longer distinguish index composition from a dividend defect"
    )


def test_tlt_moves_one_for_one_with_the_curve_derived_ladder(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """TLT's Yahoo closes against a ladder priced off the Fed's own parameters.

    A regression beta, which is a question about the PRICES: if the Yahoo TLT
    series developed a back-adjustment pathology, its sensitivity to a
    curve-derived comparand would move off one. This is a different question
    from W1-P3's mean-gap gate, which asks whether our ladder CONSTRUCTION is
    right and remains open at -0.192%/yr under a strict xfail in
    tests/test_ladder.py. Both bounds come from ``data.tlt_cross_check``.
    """
    from mafrm.data.gsw_report import cross_check_tlt

    gate = DATA.tlt_cross_check
    check = cross_check_tlt()
    with capsys.disabled():
        print(
            f"\n  {DATA.cross_checks.tlt_versus_ladder.ticker} vs GSW ladder: "
            f"n={check.observations:,} {check.start.date()}..{check.end.date()}; "
            f"beta {check.ladder_beta:.4f} (band [{gate.beta_min}, {gate.beta_max}]); "
            f"mean gap {check.ladder_gap_pct:+.4f}%/yr (W1-P3 open residual, not gated here)"
        )
    assert gate.beta_min <= check.ladder_beta <= gate.beta_max, (
        f"TLT's beta on the curve-derived ladder is {check.ladder_beta:.4f}, outside "
        f"[{gate.beta_min}, {gate.beta_max}]. That is a property of the price series, "
        "not of the ladder construction."
    )


def test_only_two_tickers_have_independent_corroboration() -> None:
    """Stated as an assertion so the limitation cannot quietly stop being true.

    Seven of the nine cached tickers have Yahoo as their sole source. If a future
    session adds a cross-check, this test fails and forces the README and
    reports/data_contracts.md to be updated with it.
    """
    corroborated = {
        DATA.cross_checks.spy_versus_market.ticker,
        DATA.cross_checks.tlt_versus_ladder.ticker,
    }
    all_tickers = {*UNIVERSE.yfinance_tickers, DATA.tlt_cross_check.ticker}
    assert corroborated == {"SPY", "TLT"}
    assert len(all_tickers - corroborated) == 7


# ---------------------------------------------------------------------------
# Stooq -- unavailable, and deliberately not worked around
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ticker", DATA.stooq.cross_check_tickers)
def test_stooq_agrees_with_yahoo_on_raw_closes(
    ticker: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reported, not gated.

    Two vendors disagreeing about a close is ordinary -- different consolidation
    feeds, different handling of a late print -- and no threshold for how much
    has been validated here, so inventing one would breach CLAUDE.md invariant 9.
    What this asserts is that the comparison is *possible*: overlapping dates and
    a finite distribution. The distribution itself is printed.

    SKIPS as of 2026-08-29: stooq.com serves a JavaScript proof-of-work browser
    challenge to every plain client. Satisfying it would mean circumventing an
    access control the site operator put up deliberately, which is not something
    a repository that will be made public should be doing. It stays unsolved.
    """
    secondary = cached(stooq.SOURCE, f"{ticker.lower()}_prices")
    primary = cached("yfinance", f"{ticker.lower()}_prices")

    difference = stooq.compare_closes(primary["Close"], secondary["Close"])

    with capsys.disabled():
        print(
            f"\n  {ticker} yfinance vs stooq close: n={len(difference):,} "
            f"median {float(difference.median()):+.6f} "
            f"p95 {float(difference.abs().quantile(0.95)):.6f} "
            f"max {float(difference.abs().max()):.6f}"
        )
    assert len(difference) > 100
    assert difference.notna().all()
