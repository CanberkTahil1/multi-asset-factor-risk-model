"""Generates ``reports/equity_market_cap.{md,png,csv}`` and ``reports/equity_cik_mapping.csv``.

SPEC.md 15.4.1 (W7-P2a). The point-in-time market-cap panel joined to the
estimation universe of SPEC.md 15.3: how many names have a cap on each
session and why the rest do not, how much of the sample the pre-XBRL
back-fill covers, the CIK mapping's coverage, the loader's per-quarter status
and the unexplained-jump listing. Scores experiments.md rows 269-272 against
the registrations in ``config/model.yaml`` and prints each verdict; row 273 is
a measurement and is labelled NOT PRE-REGISTERED.

Reads the cache through ``cache.read`` only (the holdout boundary applies) and
the committed reference tables; nothing here crosses the boundary. The panel
itself is :func:`mafrm.data.market_cap.build`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from mafrm import config as config_mod
from mafrm.data import cache, edgar, market_cap, sp500, sp500_reference
from mafrm.factors import equity_universe

__all__ = [
    "CapReport",
    "Verdict",
    "build",
    "coverage_counts",
    "in_band_jumps",
    "no_cap_dollar_volume_share",
    "per_session_series",
    "render",
    "residual_cells",
    "score",
    "score_rulings",
    "totals_table",
]

_TRILLION = 1e12
#: How many names a listing prints before saying "and n more". Display only.
_MAX_NAMES = 60
#: Report-only classification of the jump listing (row 273). A consecutive
#: ratio at or beyond this, or its reciprocal, is a UNIT-SCALE error -- a count
#: tagged in thousands or millions -- not a corporate action. Filters nothing.
_UNIT_SCALE_RATIO = 100.0
#: Report-only: an in-force count this many times off the name's own median
#: count (or below its reciprocal) is counted as "off scale" for the aggregate
#: comparison. Filters nothing; a production rule is the operator's to issue.
_OFF_MEDIAN_RATIO = 10.0
#: Row 275's ORIGINAL band on consecutive-count ratios, (10, 1000), kept for the record
#: after it was refuted and superseded by the measured median-relative band in config
#: (W7-P2b ruling 1). Report only.
_ORIGINAL_BAND = (10.0, 1000.0)


def _report_path(name: str) -> Path:
    return Path(__file__).resolve().parents[3] / "reports" / name


@dataclass(frozen=True)
class Verdict:
    row: int
    leg: str
    holds: bool
    detail: str


@dataclass(frozen=True)
class QuarterStatus:
    """The loader's per-quarter status for one artefact (ruling 6's contract)."""

    name: str
    ok: int
    empty: int
    failed: int
    failed_labels: tuple[str, ...]
    empty_labels: tuple[str, ...]


@dataclass(frozen=True)
class CapReport:
    panel: market_cap.MarketCapPanel
    screen: equity_universe.Screen
    #: Per session: ``universe``, ``with_cap``, ``no_cap``, ``no_cap_unmapped``,
    #: ``no_cap_ambiguous``, ``no_cap_no_facts``, ``approximated``, ``total_cap_usd_trn``.
    series: pd.DataFrame
    coverage: dict[str, int]
    approximated_share_pct: float
    quarter_status: tuple[QuarterStatus, ...]
    verdicts: tuple[Verdict, ...]
    #: Tickers sharing one CIK: ``cik -> tickers``.
    multi_class: dict[int, tuple[str, ...]]
    #: Row 273: the jump listing restricted to names ever in the estimation universe.
    jumps: pd.DataFrame
    #: Names without a cap on the first and last sample session, by reason.
    no_cap_first: dict[str, tuple[str, ...]]
    no_cap_last: dict[str, tuple[str, ...]]
    sample_start: pd.Timestamp
    holdout_start: pd.Timestamp
    #: Row 273's classification: jumps at or beyond ``_UNIT_SCALE_RATIO`` (or its
    #: reciprocal) and the rest; names in each class.
    unit_scale_jumps: int = 0
    unit_scale_names: tuple[str, ...] = ()
    other_jumps: int = 0
    other_names: tuple[str, ...] = ()
    #: (cells off scale, cells with a cap, names affected) within the universe.
    off_scale: tuple[int, int, int] = (0, 0, 0)
    #: Year-end total cap of the universe: ``raw`` and ``excluding_off_scale`` (USD trn).
    total_cap_check: pd.DataFrame | None = None
    # --- SPEC.md 15.4.3 (W7-P2): the three cap rulings -------------------------
    #: Ruling (i): the filings the screen dropped among names ever in the universe.
    dropped: pd.DataFrame | None = None
    #: Year ends: ``unscreened``, ``screened``, ``published_upper_bound`` (USD trn) and
    #: ``screened_below`` / ``unscreened_below`` the published value.
    totals: pd.DataFrame | None = None
    published: market_cap.PublishedTotals | None = None
    #: Ruling (i)'s band: UNSCREENED consecutive-count jumps among universe names whose
    #: ratio lies strictly inside ``empty_band`` or its reciprocal.
    in_band_jumps: pd.DataFrame | None = None
    #: The residual the screen leaves: post-screen jumps among universe names (splits
    #: the actions table lacks, class restructurings) and the cells they touch --
    #: (cells at or beyond 2x off the name's median, cells with a cap, names).
    residual_jumps: pd.DataFrame | None = None
    residual_cells: tuple[int, int, int] = (0, 0, 0)
    #: Ruling (iii): per session, the no-cap names' share of universe dollar volume (%)
    #: and their count; ``no_facts`` names on the last session.
    no_cap_dollar_volume: pd.DataFrame | None = None
    dual_class_last: tuple[str, ...] = ()


def _year_ends(sessions: pd.DatetimeIndex) -> list[pd.Timestamp]:
    return [pd.Timestamp(sessions[sessions.year == y][-1]) for y in sorted(set(sessions.year))]


def no_cap_dollar_volume_share(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    universe: pd.DataFrame,
    cap: pd.DataFrame,
) -> pd.DataFrame:
    """Per session: universe names WITHOUT a cap, and their share of the universe's dollar volume.

    SPEC.md 15.4.3 ruling (iii). The quantity missing for these names is the
    cap itself, so their cap share cannot be stated; ``close x volume`` over
    valid bars is the proxy the report states instead, labelled as one. Columns
    ``no_cap_names`` and ``no_cap_dollar_volume_pct``.
    """
    sessions = pd.DatetimeIndex(universe.index)
    tickers = [str(t) for t in universe.columns]
    in_u = universe.astype(bool).to_numpy()
    c = close.reindex(index=sessions, columns=tickers).to_numpy(dtype=float)
    v = volume.reindex(index=sessions, columns=tickers).to_numpy(dtype=float)
    has_cap = cap.reindex(index=sessions, columns=tickers).notna().to_numpy()
    dollar = np.where(in_u & np.isfinite(c) & np.isfinite(v) & (c > 0) & (v > 0), c * v, 0.0)
    total = dollar.sum(axis=1)
    missing = np.where(~has_cap, dollar, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(total > 0, 100.0 * missing / total, np.nan)
    out = pd.DataFrame(
        {"no_cap_names": (in_u & ~has_cap).sum(axis=1), "no_cap_dollar_volume_pct": pct},
        index=sessions,
    )
    out.index.name = "date"
    return out


def residual_cells(
    panel: market_cap.MarketCapPanel, universe: pd.DataFrame, *, ratio: float
) -> tuple[int, int, int]:
    """Universe cells whose in-force SCREENED count is ``ratio`` off the name's median, or beyond.

    An UPPER BOUND on the cells a split the actions table lacks (or a class
    restructuring) mis-states: a genuine unrecorded 2:1 leaves one side of the
    name's history 2x off its median. Returns (cells, cells with a cap, names).
    """
    tickers = [str(c) for c in universe.columns]
    sessions = pd.DatetimeIndex(universe.index)
    in_u = universe.reindex(index=sessions, columns=tickers).fillna(False).astype(bool)
    shares = panel.shares.reindex(index=sessions, columns=tickers)
    cap = panel.cap.reindex(index=sessions, columns=tickers)
    rel = shares.div(shares.median(axis=0), axis=1)
    off = ((rel >= ratio) | (rel <= 1.0 / ratio)) & in_u & cap.notna()
    with_cap = in_u & cap.notna()
    return int(off.to_numpy().sum()), int(with_cap.to_numpy().sum()), int(off.any(axis=0).sum())


def totals_table(
    unscreened: market_cap.MarketCapPanel,
    screened: market_cap.MarketCapPanel,
    universe: pd.DataFrame,
    published: market_cap.PublishedTotals,
) -> pd.DataFrame:
    """Year-end universe total cap without and with the screen, beside the published bound.

    All three columns in USD trn.
    """
    sessions = pd.DatetimeIndex(universe.index)
    tickers = [str(c) for c in universe.columns]
    in_u = universe.reindex(index=sessions, columns=tickers).fillna(False).astype(bool)
    year_ends = _year_ends(sessions)
    raw = unscreened.cap.reindex(index=sessions, columns=tickers).where(in_u).loc[year_ends]
    scr = screened.cap.reindex(index=sessions, columns=tickers).where(in_u).loc[year_ends]
    pub = published.values_usd
    pub_years = pd.DatetimeIndex(pub.index).year
    values: list[float] = []
    for d in year_ends:
        hit = pub[pub_years == d.year]
        values.append(float(hit.iloc[0]) if len(hit) else float("nan"))
    bound = pd.Series(values, index=pd.DatetimeIndex(year_ends))
    out = pd.DataFrame(
        {
            "unscreened": raw.sum(axis=1) / _TRILLION,
            "screened": scr.sum(axis=1) / _TRILLION,
            "published_upper_bound": bound / _TRILLION,
        }
    )
    out["screened_below"] = out["screened"] < out["published_upper_bound"]
    out["unscreened_below"] = out["unscreened"] < out["published_upper_bound"]
    out.index.name = "date"
    return out


def in_band_jumps(jumps: pd.DataFrame, *, band: tuple[float, float]) -> pd.DataFrame:
    """Jumps whose ratio lies strictly inside ``(low, high)`` or ``(1/high, 1/low)``."""
    low, high = band
    ratio = jumps["ratio"].astype(float)
    inside = ((ratio > low) & (ratio < high)) | ((ratio > 1.0 / high) & (ratio < 1.0 / low))
    return jumps[inside].copy()


# ---------------------------------------------------------------------------
# Pure pieces
# ---------------------------------------------------------------------------


def per_session_series(panel: market_cap.MarketCapPanel, universe: pd.DataFrame) -> pd.DataFrame:
    """Counts by cause and the total cap, on every session of ``universe``."""
    tickers = sorted(set(map(str, universe.columns)) | set(map(str, panel.cap.columns)))
    sessions = pd.DatetimeIndex(universe.index)
    in_universe = universe.reindex(index=sessions, columns=tickers).fillna(False).astype(bool)
    cap = panel.cap.reindex(index=sessions, columns=tickers)
    has_cap = cap.notna()
    approx = panel.approximated.reindex(index=sessions, columns=tickers).fillna(False).astype(bool)
    status = panel.mapping["status"].reindex(tickers).fillna("unmapped")
    out = pd.DataFrame(index=sessions)
    out.index.name = "date"
    out["universe"] = in_universe.sum(axis=1)
    out["with_cap"] = (in_universe & has_cap).sum(axis=1)
    out["no_cap"] = out["universe"] - out["with_cap"]
    for reason in market_cap.NO_CAP_REASONS:
        mask = (status == reason).to_numpy()
        out[f"no_cap_{reason}"] = (in_universe & ~has_cap).loc[:, mask].sum(axis=1)
    out["approximated"] = (in_universe & has_cap & approx).sum(axis=1)
    out["total_cap_usd_trn"] = cap.where(in_universe).sum(axis=1) / _TRILLION
    return out


def approximated_share(panel: market_cap.MarketCapPanel, universe: pd.DataFrame) -> float:
    """Per cent of (session, name) pairs with a cap in the universe that are back-filled."""
    tickers = [str(t) for t in universe.columns]
    sessions = pd.DatetimeIndex(universe.index)
    in_universe = universe.astype(bool).to_numpy()
    has_cap = panel.cap.reindex(index=sessions, columns=tickers).notna().to_numpy()
    approx = (
        panel.approximated.reindex(index=sessions, columns=tickers)
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    with_cap = in_universe & has_cap
    total = int(with_cap.sum())
    if total == 0:
        return float("nan")
    return 100.0 * float((with_cap & approx).sum()) / total


def coverage_counts(mapping: pd.DataFrame) -> dict[str, int]:
    """Mapping coverage by list (current / departed) and method, plus ``no_facts``."""
    current = mapping[mapping["current"].astype(bool)]
    departed = mapping[~mapping["current"].astype(bool)]
    out: dict[str, int] = {"current": len(current), "departed": len(departed)}
    for label, part in (("current", current), ("departed", departed)):
        for method in ("ticker", "name", "ambiguous", "unmapped"):
            out[f"{label}_{method}"] = int((part["method"] == method).sum())
        out[f"{label}_no_facts"] = int((part["status"] == "no_facts").sum())
        out[f"{label}_ok"] = int((part["status"] == "ok").sum())
    return out


def _names_without_cap(
    panel: market_cap.MarketCapPanel, universe: pd.DataFrame, session: pd.Timestamp
) -> dict[str, tuple[str, ...]]:
    row = universe.loc[session]
    members = [str(t) for t, flag in row.items() if bool(flag)]
    cap_row = panel.cap.reindex(columns=members).loc[session] if members else pd.Series(dtype=float)
    out: dict[str, list[str]] = {reason: [] for reason in market_cap.NO_CAP_REASONS}
    for ticker in members:
        if pd.notna(cap_row.get(ticker, np.nan)):
            continue
        status = str(panel.mapping["status"].get(ticker, "unmapped"))
        out.setdefault(status, []).append(ticker)
    return {k: tuple(sorted(v)) for k, v in out.items()}


def score(
    cfg: config_mod.Config,
    *,
    mapping: pd.DataFrame,
    coverage: dict[str, int],
    approximated_share_pct: float,
    series: pd.DataFrame,
    panel: market_cap.MarketCapPanel,
    forms: pd.Series,
) -> tuple[Verdict, ...]:
    """Rows 269-272 against ``equity_shares.registrations``. ``forms`` maps accn -> form."""
    reg = cfg.model.equity_shares.registrations
    verdicts: list[Verdict] = []

    # Row 269
    pct_ticker = 100.0 * coverage["current_ticker"] / max(coverage["current"], 1)
    verdicts.append(
        Verdict(
            269,
            "(a) current by ticker",
            pct_ticker >= reg.row_269_current_by_ticker_min_pct,
            f"{coverage['current_ticker']} of {coverage['current']} = {pct_ticker:.1f}% "
            f"against >= {reg.row_269_current_by_ticker_min_pct:g}%",
        )
    )
    pct_name = 100.0 * coverage["departed_name"] / max(coverage["departed"], 1)
    verdicts.append(
        Verdict(
            269,
            "(b) departed by name",
            pct_name >= reg.row_269_departed_by_name_min_pct,
            f"{coverage['departed_name']} of {coverage['departed']} = {pct_name:.1f}% "
            f"against >= {reg.row_269_departed_by_name_min_pct:g}%",
        )
    )

    # Row 270
    low, high = reg.row_270_approximated_share_band_pct
    verdicts.append(
        Verdict(
            270,
            "approximated share",
            bool(low <= approximated_share_pct <= high),
            f"{approximated_share_pct:.2f}% of (session, name) pairs with a cap against "
            f"[{low:g}, {high:g}]%",
        )
    )

    # Row 271
    first = int(series["no_cap"].iloc[0])
    last = int(series["no_cap"].iloc[-1])
    verdicts.append(
        Verdict(
            271,
            f"(a) no cap on {series.index[0].date()}",
            first > reg.row_271_no_cap_first_session_min,
            f"{first} of {int(series['universe'].iloc[0])} against > "
            f"{reg.row_271_no_cap_first_session_min}",
        )
    )
    verdicts.append(
        Verdict(
            271,
            f"(b) no cap on {series.index[-1].date()}",
            last < reg.row_271_no_cap_last_session_max,
            f"{last} of {int(series['universe'].iloc[-1])} against < "
            f"{reg.row_271_no_cap_last_session_max}",
        )
    )

    # Row 272
    r = reg.row_272
    table = panel.counts.get(r.ticker, pd.DataFrame())
    cik_ok = r.ticker in mapping.index and int(str(mapping.loc[r.ticker, "cik"])) == r.cik

    def in_force(day: pd.Timestamp) -> tuple[pd.Timestamp | None, float, str | None]:
        if table.empty:
            return None, float("nan"), None
        before = table[table.index <= day]
        if before.empty:
            first_row = table.iloc[0]
            return None, float(first_row["shares"]), str(first_row["accn"])
        row = before.iloc[-1]
        return pd.Timestamp(before.index[-1]), float(row["shares"]), str(row["accn"])

    filed_a, shares_a, accn_a = in_force(pd.Timestamp(r.in_force_on))
    form_a = str(forms.get(accn_a, "?")) if accn_a is not None else "?"
    holds_a = (
        cik_ok
        and filed_a == pd.Timestamp(r.expected_filed)
        and shares_a == float(r.expected_shares)
        and form_a == r.expected_form
    )
    verdicts.append(
        Verdict(
            272,
            f"(a) {r.ticker} in force on {r.in_force_on}",
            bool(holds_a),
            f"filed {filed_a.date() if filed_a is not None else None} ({form_a}), "
            f"{shares_a:,.0f} against filed {r.expected_filed} ({r.expected_form}), "
            f"{r.expected_shares:,}",
        )
    )
    ex = pd.Timestamp(r.split_ex_date)
    filed_b, shares_b, _ = in_force(ex)
    expected_b = float(r.split_ratio * r.pre_split_shares)
    holds_b = cik_ok and filed_b == pd.Timestamp(r.pre_split_filed) and shares_b == expected_b
    verdicts.append(
        Verdict(
            272,
            f"(b) {r.ticker} in force on the split ex-date {r.split_ex_date}",
            bool(holds_b),
            f"filed {filed_b.date() if filed_b is not None else None}, {shares_b:,.0f} against "
            f"filed {r.pre_split_filed}, {r.split_ratio} x {r.pre_split_shares:,} = "
            f"{expected_b:,.0f}",
        )
    )
    shares_col = (
        panel.shares[r.ticker] if r.ticker in panel.shares.columns else pd.Series(dtype=float)
    )
    sessions = pd.DatetimeIndex(shares_col.index)
    prior = sessions[sessions < ex]
    if len(prior) and ex in sessions:
        s_before, s_on = float(shares_col.loc[prior[-1]]), float(shares_col.loc[ex])
        cap_col = panel.cap[r.ticker].astype(float)
        cap_before = float(cap_col.loc[prior[-1]]) / _TRILLION
        cap_on = float(cap_col.loc[ex]) / _TRILLION
        holds_c = s_before == s_on
        detail_c = (
            f"{s_before:,.0f} on {prior[-1].date()} and {s_on:,.0f} on {ex.date()}; cap "
            f"{cap_before:.3f} -> {cap_on:.3f} USD trn"
        )
    else:
        holds_c, detail_c = False, "the split ex-date or the session before it is not in the panel"
    verdicts.append(Verdict(272, "(c) no jump at the split", bool(holds_c), detail_c))
    return tuple(verdicts)


def score_rulings(
    cfg: config_mod.Config,
    *,
    panel: market_cap.MarketCapPanel,
    dropped: pd.DataFrame,
    totals: pd.DataFrame,
    in_band: pd.DataFrame,
    dollar_volume: pd.DataFrame,
) -> tuple[Verdict, ...]:
    """Rows 274-277 (SPEC.md 15.4.3) against ``equity_shares.registrations``."""
    reg = cfg.model.equity_shares.registrations
    band = cfg.model.equity_shares.consistency_screen.empty_band
    verdicts: list[Verdict] = []
    below = bool(totals["screened_below"].all())
    raw_above = int((~totals["unscreened_below"]).sum())
    verdicts.append(
        Verdict(
            274,
            "(a) screened total below the published US total on every year end",
            below and raw_above > 0,
            f"screened below on {int(totals['screened_below'].sum())} of {len(totals)} year ends; "
            f"unscreened above on {raw_above}",
        )
    )
    verdicts.append(
        Verdict(
            274,
            "(b) universe filings dropped",
            len(dropped) <= reg.row_274_max_dropped_universe_filings,
            f"{len(dropped)} filings over {dropped['ticker'].nunique() if len(dropped) else 0} "
            f"names against <= {reg.row_274_max_dropped_universe_filings}",
        )
    )
    verdicts.append(
        Verdict(
            275,
            "as REGISTERED: consecutive jumps inside (10, 1000) or its reciprocal, unscreened",
            len(in_band) <= reg.row_275_jumps_inside_band_max,
            f"{len(in_band)} over {in_band['ticker'].nunique() if len(in_band) else 0} names "
            f"against <= {reg.row_275_jumps_inside_band_max}; superseded by the measured "
            f"median-relative band ({band[0]:g}, {band[1]:g}) around R (W7-P2b ruling 1)",
        )
    )
    r = reg.row_276
    m = panel.mapping.loc[r.ticker] if r.ticker in panel.mapping.index else None
    table = panel.counts.get(r.ticker, pd.DataFrame())
    before = table[table.index <= pd.Timestamp(r.in_force_on)] if len(table) else table
    filed = pd.Timestamp(before.index[-1]) if len(before) else None
    shares = float(before.iloc[-1]["shares"]) if len(before) else float("nan")
    route_ok = (
        m is not None
        and str(m["status"]) == "ok"
        and pd.notna(m["predecessor_cik"])
        and int(str(m["predecessor_cik"])) == r.predecessor_cik
        and filed == pd.Timestamp(r.expected_filed)
    )
    verdicts.append(
        Verdict(
            276,
            f"{r.ticker} via predecessor CIK {r.predecessor_cik}, against the cover page",
            bool(route_ok and shares == float(r.expected_shares)),
            f"status {m['status'] if m is not None else None}, filed "
            f"{filed.date() if filed is not None else None}, {shares:,.0f} against "
            f"{r.expected_shares:,} (the registration's 4,395,095,000 was a rounded-print "
            "transcription; experiments.md row 276)",
        )
    )
    low, high = reg.row_277_no_cap_dollar_volume_share_band_pct
    last_pct = float(dollar_volume["no_cap_dollar_volume_pct"].iloc[-1])
    verdicts.append(
        Verdict(
            277,
            f"no-cap names' dollar-volume share on {dollar_volume.index[-1].date()}",
            bool(low <= last_pct <= high),
            f"{last_pct:.2f}% over {int(dollar_volume['no_cap_names'].iloc[-1])} names against "
            f"[{low:g}, {high:g}]%",
        )
    )
    return tuple(verdicts)


def off_scale_check(
    panel: market_cap.MarketCapPanel, universe: pd.DataFrame
) -> tuple[int, int, int, pd.DataFrame]:
    """Cells whose in-force count is ``_OFF_MEDIAN_RATIO`` off the name's own median.

    A MEASUREMENT for the report, not a filter: returns the count of such
    (session, name) cells within the universe, the count of cells with a cap,
    the names affected, and a year-end table of the universe's total cap with
    and without those cells (USD trn). The rule that would remove them from the
    production panel is the operator's to issue (SPEC.md 15.4.2).
    """
    tickers = [str(c) for c in universe.columns]
    sessions = pd.DatetimeIndex(universe.index)
    in_universe = universe.reindex(index=sessions, columns=tickers).fillna(False).astype(bool)
    shares = panel.shares.reindex(index=sessions, columns=tickers)
    cap = panel.cap.reindex(index=sessions, columns=tickers)
    median = shares.median(axis=0)
    rel = shares.div(median, axis=1)
    off = (
        ((rel >= _OFF_MEDIAN_RATIO) | (rel <= 1.0 / _OFF_MEDIAN_RATIO)) & in_universe & cap.notna()
    )
    with_cap = in_universe & cap.notna()
    year_ends = [pd.Timestamp(sessions[sessions.year == y][-1]) for y in sorted(set(sessions.year))]
    raw = cap.where(in_universe).loc[year_ends].sum(axis=1) / _TRILLION
    excluded = cap.where(in_universe & ~off).loc[year_ends].sum(axis=1) / _TRILLION
    check = pd.DataFrame(
        {
            "raw": raw,
            "excluding_off_scale": excluded,
            "off_scale_names": off.loc[year_ends].sum(axis=1).astype(int),
        }
    )
    check.index.name = "date"
    return (
        int(off.to_numpy().sum()),
        int(with_cap.to_numpy().sum()),
        int(off.any(axis=0).sum()),
        check,
    )


def _quarter_status(entry: cache.ManifestEntry, name: str) -> QuarterStatus:
    status = dict(entry.status)
    failed = tuple(sorted(k for k, v in status.items() if v == "failed"))
    empty = tuple(sorted(k for k, v in status.items() if v == "empty"))
    return QuarterStatus(
        name=name,
        ok=sum(1 for v in status.values() if v == "ok"),
        empty=len(empty),
        failed=len(failed),
        failed_labels=failed,
        empty_labels=empty,
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(cfg: config_mod.Config | None = None) -> CapReport:
    cfg = cfg or config_mod.load()
    universe_cfg = cfg.model.equity_universe
    sample_start = pd.Timestamp(cfg.model.sample.start)
    holdout_start = pd.Timestamp(cfg.require_holdout_start())
    manifest = cache.Manifest.load()

    # The bars the committed matrix was built on, when the cache still holds
    # them (one-snapshot rule, W7-P1b); otherwise the latest pull.
    snapshot_entries = sp500_reference.snapshot_input_entries(cfg, manifest)
    prices_entry = actions_entry = None
    if snapshot_entries is not None:
        by_key = {(e.source, e.name): e for e in snapshot_entries}
        prices_entry = by_key.get(("yfinance", "sp500_prices"))
        actions_entry = by_key.get(("yfinance", "sp500_actions"))

    panel = market_cap.build(
        cfg, manifest=manifest, prices_entry=prices_entry, actions_entry=actions_entry
    )
    close, volume = equity_universe.load_bars(cfg, entry=prices_entry)
    membership = equity_universe.load_membership(cfg)
    screen = equity_universe.screen(
        close,
        volume,
        membership,
        min_history_days=universe_cfg.min_history_days,
        min_dollar_adv=universe_cfg.min_dollar_adv,
        start=sample_start,
        end=holdout_start,
        strict=snapshot_entries is not None,
    )
    universe = screen.universe
    series = per_session_series(panel, universe)
    share_pct = approximated_share(panel, universe)
    coverage = coverage_counts(panel.mapping)

    index_entry = next(e for e in panel.inputs if e.name == edgar.INDEX_NAME)
    frames_entry = next(e for e in panel.inputs if e.name == edgar.FRAMES_NAME)
    tickers_entry = next(e for e in panel.inputs if e.name == edgar.COMPANY_TICKERS_NAME)
    index = cache.read(index_entry, manifest=manifest)
    forms = pd.Series(index["form"].to_numpy(), index=index["accn"].astype(str).to_numpy())
    forms = forms[~forms.index.duplicated(keep="first")]

    # The UNSCREENED panel, without the override route: row 273's listing, row
    # 275's band and the "without the screen" column are all statements about
    # the source as filed, so they are computed on it rather than on the panel
    # the model reads.
    unscreened = market_cap.assemble(
        frames=cache.read(frames_entry, manifest=manifest),
        index=index,
        company_tickers=market_cap.load_company_tickers(manifest, entry=tickers_entry),
        close=market_cap.load_close(manifest, entry=prices_entry).reindex(panel.sessions),
        splits=market_cap.load_splits(manifest, entry=actions_entry),
        intervals=sp500.load_intervals(
            sp500_reference.reference_dir(cfg) / cfg.model.equity_universe.intervals_file
        ),
        sessions=panel.sessions,
    )

    verdicts = score(
        cfg,
        mapping=panel.mapping,
        coverage=coverage,
        approximated_share_pct=share_pct,
        series=series,
        panel=panel,
        forms=forms,
    )
    mapped = panel.mapping[panel.mapping["cik"].notna()]
    multi = {
        int(str(cik)): tuple(sorted(str(t) for t in group.index))
        for cik, group in mapped.groupby("cik")
        if len(group) > 1
    }
    ever = {str(t) for t in universe.columns[universe.any(axis=0)]}
    jumps = unscreened.jumps[unscreened.jumps["ticker"].isin(ever)].copy()
    ratio = jumps["ratio"].astype(float)
    is_unit = (ratio >= _UNIT_SCALE_RATIO) | (ratio <= 1.0 / _UNIT_SCALE_RATIO)
    off_cells, total_cells, off_names, check = off_scale_check(unscreened, universe)

    shares_cfg = cfg.model.equity_shares
    published = market_cap.load_published_totals(
        config_mod.config_dir() / shares_cfg.published_totals_file
    )
    dropped = panel.dropped[panel.dropped["ticker"].isin(ever)].copy()
    totals = totals_table(unscreened, panel, universe, published)
    in_band = in_band_jumps(jumps, band=_ORIGINAL_BAND)
    residual = panel.jumps[panel.jumps["ticker"].isin(ever)].copy()
    dollar_volume = no_cap_dollar_volume_share(close, volume, universe, panel.cap)
    verdicts = (
        *verdicts,
        *score_rulings(
            cfg,
            panel=panel,
            dropped=dropped,
            totals=totals,
            in_band=in_band,
            dollar_volume=dollar_volume,
        ),
    )
    no_cap_last = _names_without_cap(panel, universe, pd.Timestamp(universe.index[-1]))
    return CapReport(
        unit_scale_jumps=int(is_unit.sum()),
        unit_scale_names=tuple(sorted(jumps.loc[is_unit, "ticker"].unique())),
        other_jumps=int((~is_unit).sum()),
        other_names=tuple(sorted(jumps.loc[~is_unit, "ticker"].unique())),
        off_scale=(off_cells, total_cells, off_names),
        total_cap_check=check,
        dropped=dropped,
        totals=totals,
        published=published,
        in_band_jumps=in_band,
        residual_jumps=residual,
        residual_cells=residual_cells(panel, universe, ratio=edgar.JUMP_RATIO),
        no_cap_dollar_volume=dollar_volume,
        dual_class_last=no_cap_last.get("no_facts", ()),
        panel=panel,
        screen=screen,
        series=series,
        coverage=coverage,
        approximated_share_pct=share_pct,
        quarter_status=(
            _quarter_status(frames_entry, "frames"),
            _quarter_status(index_entry, "xbrl index"),
        ),
        verdicts=verdicts,
        multi_class=multi,
        jumps=jumps,
        no_cap_first=_names_without_cap(panel, universe, pd.Timestamp(universe.index[0])),
        no_cap_last=no_cap_last,
        sample_start=sample_start,
        holdout_start=holdout_start,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _list(names: tuple[str, ...]) -> str:
    if not names:
        return "none"
    shown = ", ".join(names[:_MAX_NAMES])
    more = len(names) - _MAX_NAMES
    return shown + (f", and {more} more" if more > 0 else "")


def render(report: CapReport, cfg: config_mod.Config) -> str:
    shares_cfg = cfg.model.equity_shares
    s = report.series
    cov = report.coverage
    first, last = s.index[0].date(), s.index[-1].date()
    lines: list[str] = []
    w = lines.append
    w(
        "# Equity market cap -- point-in-time shares outstanding from EDGAR "
        "(SPEC.md 15.4.1, W7-P2a; the three cap rulings of 15.4.3, W7-P2)"
    )
    w("")
    w(
        f"Generated by `mafrm.factors.cap_report`. Estimation sample {first} to {last} "
        f"(holdout from {report.holdout_start.date()}, nothing on or after it read). Source: "
        f"`{shares_cfg.taxonomy}:{shares_cfg.tag}` ({shares_cfg.unit}) from EDGAR's frames "
        f"endpoint, one request per calendar quarter from {shares_cfg.first_period}; filing "
        "dates from EDGAR's quarterly XBRL index, joined by accession; SEC's "
        "`company_tickers.json` for the current list. Operator ruling 1 and the two "
        "orientation findings are at SPEC.md 15.4.1; the consistency screen, the "
        "predecessor-CIK table and the dual-class ruling are at SPEC.md 15.4.3."
    )
    w("")
    screen_ratio = report.panel.consistency_ratio
    w(
        f"**The panel the model reads is SCREENED** at `R = {screen_ratio:g}` "
        "(`equity_shares.consistency_screen.ratio`; SPEC.md 15.4.3 ruling (i)): a filed count "
        f"more than {screen_ratio:g}x off the name's own median split-adjusted count, or below "
        f"1/{screen_ratio:g} of it, is dropped, the previous count stays in force, and the "
        "pre-XBRL back-fill reads the earliest SURVIVING count. `MarketCapPanel.is_model_input` "
        "is true only for a screened panel. The row-273 listing, the row-275 band and the "
        "'without the screen' column below are computed on the UNSCREENED counts, as filed."
        if screen_ratio is not None
        else "**The panel is UNSCREENED** and is a loader output, not a model input."
    )
    w("")
    w("## Provenance")
    w("")
    w("| Input | Path | Rows | Range |")
    w("|---|---|---|---|")
    for e in report.panel.inputs:
        w(f"| {e.source}/{e.name} | `{e.path}` | {e.rows:,} | {e.first_date}..{e.last_date} |")
    w("")
    w("## The status contract, per quarter")
    w("")
    w(
        "Same contract as the S&P 500 bars (SPEC.md 15.3.1 ruling 6): `ok`, `empty` (the "
        "server answered 404 -- no frame or index for that quarter) or `failed` (anything "
        "else, retried once). A quarter still `failed` is listed here by name and is never "
        'read as "no filers".'
    )
    w("")
    w("| Artefact | ok | empty | failed | empty quarters | failed quarters |")
    w("|---|---|---|---|---|---|")
    for q in report.quarter_status:
        w(
            f"| {q.name} | {q.ok} | {q.empty} | {q.failed} | {_list(q.empty_labels)} | "
            f"{_list(q.failed_labels)} |"
        )
    w("")
    w("## CIK mapping coverage (row 269)")
    w("")
    w(
        "A current-list ticker maps by ticker through SEC's map; a departed name maps by "
        "exact normalised company name only, because tickers are recycled. `ambiguous` = "
        "the name matches more than one filer; `unmapped` = it matches none; `no_facts` = "
        "mapped, but no usable count (no fact in any frame, or every fact's accession is "
        "in no cached index). Nothing is resolved by hand; the full table is "
        "`reports/equity_cik_mapping.csv`."
    )
    w("")
    w(
        "| List | Tickers | by ticker | by name | ambiguous | unmapped | mapped, no facts | "
        "with counts |"
    )
    w("|---|---|---|---|---|---|---|---|")
    for label in ("current", "departed"):
        w(
            f"| {label} | {cov[label]} | {cov[f'{label}_ticker']} | {cov[f'{label}_name']} | "
            f"{cov[f'{label}_ambiguous']} | {cov[f'{label}_unmapped']} | "
            f"{cov[f'{label}_no_facts']} | {cov[f'{label}_ok']} |"
        )
    w("")
    if report.multi_class:
        w(
            f"**Tickers sharing one CIK ({len(report.multi_class)} filers):** each class ticker "
            "receives the filer's non-dimensional total count times its own close, so a "
            "dual-listed company is counted once per listed class. Where the filer reports "
            "the fact per class with a dimension instead, the frames carry nothing and the "
            "name is `no_facts`."
        )
        w("")
        for cik, tickers in sorted(report.multi_class.items()):
            w(f"- CIK {cik}: {', '.join(tickers)}")
        w("")
    w("## Coverage of the estimation universe (row 271)")
    w("")
    w(
        "Per session, the estimation universe (SPEC.md 15.3's screen) and how many of its "
        f"names have a cap. The universe runs {int(s['universe'].min())}-"
        f"{int(s['universe'].max())} names; with a cap, {int(s['with_cap'].min())}-"
        f"{int(s['with_cap'].max())}. Full series in `reports/equity_market_cap.csv`; the "
        "figure is `reports/equity_market_cap.png`."
    )
    w("")
    w(
        "| Session | Universe | With cap | No cap | unmapped | ambiguous | no facts | "
        "back-filled | Total cap, SCREENED (USD trn) |"
    )
    w("|---|---|---|---|---|---|---|---|---|")
    idx = pd.DatetimeIndex(s.index)
    year_ends = [pd.Timestamp(idx[idx.year == y][-1]) for y in sorted(set(idx.year))]
    picks = [pd.Timestamp(idx[0]), *year_ends]
    seen: set[pd.Timestamp] = set()
    for d in picks:
        if d in seen:
            continue
        seen.add(d)
        vals = {str(k): v for k, v in s.loc[[d]].iloc[0].to_dict().items()}
        w(
            f"| {d.date()} | {int(vals['universe'])} | {int(vals['with_cap'])} | "
            f"{int(vals['no_cap'])} | "
            f"{int(vals['no_cap_unmapped'])} | {int(vals['no_cap_ambiguous'])} | "
            f"{int(vals['no_cap_no_facts'])} | {int(vals['approximated'])} | "
            f"{float(vals['total_cap_usd_trn']):.2f} |"
        )
    w("")
    w(f"**Names without a cap on {first}, by reason:**")
    w("")
    for reason, names in report.no_cap_first.items():
        w(f"- `{reason}` ({len(names)}): {_list(names)}")
    w("")
    w(f"**Names without a cap on {last}, by reason:**")
    w("")
    for reason, names in report.no_cap_last.items():
        w(f"- `{reason}` ({len(names)}): {_list(names)}")
    w("")
    w("## The pre-XBRL approximation (row 270)")
    w("")
    w(
        f"**{report.approximated_share_pct:.2f}%** of the (session, name) pairs with a cap in the "
        "estimation universe carry a BACK-FILLED count: the name's earliest filed count, "
        "split-adjusted, in force on sessions before that filing. Direction: buybacks between "
        "the session and the first filing mean the true count was HIGHER, so the back-fill "
        "UNDERSTATES those early caps; net issuance means it OVERSTATES them. For the largest "
        "filers the gap runs from the sample start to their first cover-page fact in 2009; "
        "smaller filers phase in through 2011. Every consumer of `cap` can read the mask "
        "(`MarketCapPanel.approximated`) and every equity report states this share."
    )
    w("")
    w("## Unexplained jumps (row 273, NOT PRE-REGISTERED)")
    w("")
    w(
        "Consecutive split-adjusted counts of one filer that move by more than 2x, or to "
        "less than half, with no split in the cache's actions table to explain it. A REPORT "
        "listing, not a filter: a placeholder value (a filer reporting `1`), a split the "
        "vendor's actions table lacks, or a class restructuring. Names ever in the estimation "
        f"universe only; {len(report.jumps)} jump(s) over "
        f"{report.jumps['ticker'].nunique() if len(report.jumps) else 0} name(s)."
    )
    w("")
    if len(report.jumps):
        w("| Ticker | Filed | Previous | New | Ratio |")
        w("|---|---|---|---|---|")
        ordered = report.jumps.assign(_k=np.abs(np.log(report.jumps["ratio"].astype(float))))
        for row in ordered.sort_values("_k", ascending=False).head(40).itertuples(index=False):
            w(
                f"| {row.ticker} | {pd.Timestamp(str(row.filed)).date()} | {row.previous:,.0f} | "
                f"{row.shares:,.0f} | {row.ratio:.4g} |"
            )
        w("")
    w("## FINDING (row 273): the source carries unit-scale errors, and the raw aggregate shows it")
    w("")
    off_cells, total_cells, off_names = report.off_scale
    w(
        f"Of the {len(report.jumps)} unexplained jumps, **{report.unit_scale_jumps} are at or "
        f"beyond {_UNIT_SCALE_RATIO:g}x (or below 1/{_UNIT_SCALE_RATIO:g})** over "
        f"{len(report.unit_scale_names)} names -- a cover-page count tagged in thousands or "
        f"millions in one filing and in shares in the next, not a corporate action -- and "
        f"{report.other_jumps} lie between, over {len(report.other_names)} names (a split the "
        "vendor's actions table lacks, a class restructuring, a shell filing of 1 or 100 "
        "shares after an acquisition or before a spin-off). In-force counts more than "
        f"{_OFF_MEDIAN_RATIO:g}x off the name's own median cover **{off_cells:,} of "
        f"{total_cells:,} (session, name) cells with a cap in the universe "
        f"({100.0 * off_cells / max(total_cells, 1):.2f}%) over {off_names} names**, and "
        "because the earliest filed count is back-filled, a name whose FIRST filing carries "
        "the error carries it over 2007-2009 too. This is the finding of W7-P2a (SPEC.md "
        "15.4.2), computed here on the UNSCREENED counts; the screen that answers it is "
        "ruling (i) below."
    )
    w("")
    if report.total_cap_check is not None:
        w(
            "| Year end | Total cap, UNSCREENED (USD trn) | Excluding off-scale cells | "
            "Off-scale names |"
        )
        w("|---|---|---|---|")
        check = report.total_cap_check
        for day, raw_v, exc_v, n_v in zip(
            [pd.Timestamp(str(x)).date() for x in check.index],
            check["raw"].astype(float).tolist(),
            check["excluding_off_scale"].astype(float).tolist(),
            check["off_scale_names"].astype(int).tolist(),
            strict=True,
        ):
            w(f"| {day} | {raw_v:,.2f} | {exc_v:,.2f} | {n_v} |")
        w("")
    w(f"**Unit-scale names ({len(report.unit_scale_names)}):** {_list(report.unit_scale_names)}")
    w("")
    w(f"**Other jump names ({len(report.other_names)}):** {_list(report.other_names)}")
    w("")
    w(
        "**A second coverage limit, found at scoring (W7-P2a).** SEC's ticker map points to the "
        "CURRENT registrant, so a company that re-incorporated under a new CIK (Exxon Mobil, "
        "CIK 2115436 succeeding 34088 in 2025) maps to a filer with no XBRL history before the "
        "boundary and would be `no_facts`. This is the by-ticker route's mirror image of the "
        "recycled-ticker hazard the by-name route avoids; ruling (ii) below is the remedy."
    )
    w("")
    _render_rulings(w, report, cfg)
    w("## Registered rows (experiments.md 269-272 and 274-277)")
    w("")
    w("| Row | Leg | Verdict | Detail |")
    w("|---|---|---|---|")
    for v in report.verdicts:
        w(f"| {v.row} | {v.leg} | {'HOLDS' if v.holds else 'REFUTED'} | {v.detail} |")
    w("")
    w("## What this is not")
    w("")
    w(
        "- Not float-adjusted: the fact is total common shares outstanding on the cover page, "
        "so caps are total, not free-float, and the cap weights of closely held names are "
        "larger than an index provider's."
    )
    w(
        "- Not per class: the frames carry only the non-dimensional total, so a multi-class "
        "filer either contributes its total to every listed class or, when it reports per "
        "class, nothing at all (`no_facts`)."
    )
    w(
        "- Not complete before 2009: the pre-XBRL back-fill above is an approximation and is "
        "labelled as one everywhere the panel is read."
    )
    w(
        f"- The survivorship finding of SPEC.md 15.3.2 stands underneath all of this: the "
        f"universe on {first} is what Yahoo still serves, and the names without a cap on that "
        "session are overwhelmingly departures that never filed the XBRL fact."
    )
    w("")
    return "\n".join(lines)


def _render_rulings(w: Callable[[str], None], report: CapReport, cfg: config_mod.Config) -> None:
    shares_cfg = cfg.model.equity_shares
    screen = shares_cfg.consistency_screen
    w("## Ruling (i): the within-name consistency screen (SPEC.md 15.4.3, rows 274-275)")
    w("")
    dropped = report.dropped if report.dropped is not None else pd.DataFrame()
    w(
        f"`R = {screen.ratio:g}`. A filed count more than R times the name's own median "
        "split-adjusted count, or below 1/R of it, is dropped; the previous count stays in "
        "force; the back-fill reads the earliest surviving count. R is not tuned. On the "
        "median-relative ratios the screen acts on (count / the name's median, folded to >= 1, "
        "every usable filing among universe names, unscreened), the 2026-09-05 pull leaves "
        f"({screen.empty_band[0]:g}, {screen.empty_band[1]:g}) empty and R sits inside it; "
        "`tests/test_market_cap_dataset.py::test_r_sits_inside_the_measured_empty_band` "
        "recomputes the band around R on every re-pull, prints its ends and asserts it still "
        "contains this one, and `test_no_dropped_count_is_a_corporate_action` asserts that no "
        "dropped ratio is one the actions table explains. A future filing inside the band "
        "fails the first test and RE-OPENS R with numbers. (W7-P2's first statement of the "
        "band, (10, 1000) on consecutive-count ratios, was a misreading; SPEC.md 15.4.3.)"
    )
    w("")
    w(
        f"**Dropped among universe names: {len(dropped)} filings over "
        f"{dropped['ticker'].nunique() if len(dropped) else 0} names** "
        f"({len(report.panel.dropped)} over every mapped name). Every drop, with the median it "
        "was compared against:"
    )
    w("")
    if len(dropped):
        w("| Ticker | Filed | As filed | Split basis | Name's median | Ratio to median |")
        w("|---|---|---|---|---|---|")
        for row in dropped.sort_values(["ticker", "filed"]).itertuples(index=False):
            w(
                f"| {row.ticker} | {pd.Timestamp(str(row.filed)).date()} | {row.val:,.0f} | "
                f"{row.shares:,.0f} | {row.median:,.0f} | {row.ratio:.3g} |"
            )
        w("")
    if report.totals is not None and report.published is not None:
        pub = report.published
        w(
            "**The universe's total cap, without and with the screen, beside a published total.** "
            f"The published column is *{pub.label}* -- read {pub.date_read} from "
            f"<{pub.source_url}> in {pub.unit}. It is NOT the index's own total: no free, citable "
            "history of the S&P 500's year-end total market value was found (S&P DJI's factsheet "
            "carries the current figure only), so the comparison is a one-sided bound -- a "
            "universe total ABOVE it is impossible for a correct panel, a total below it is "
            "consistent, not confirmed. The estimation universe is 330-488 survivors of the "
            "index, so its total should sit well below the whole market's."
        )
        w("")
        w(
            "| Year end | Unscreened (USD trn) | Screened (USD trn) | Published total US market "
            "(USD trn) | Screened below | Unscreened below |"
        )
        w("|---|---|---|---|---|---|")
        for row in report.totals.itertuples():
            w(
                f"| {pd.Timestamp(str(row.Index)).date()} | {row.unscreened:,.2f} | "
                f"{row.screened:,.2f} | {row.published_upper_bound:,.2f} | "
                f"{'yes' if row.screened_below else 'NO'} | "
                f"{'yes' if row.unscreened_below else 'NO'} |"
            )
        w("")
    band = report.in_band_jumps if report.in_band_jumps is not None else pd.DataFrame()
    w(
        "**Row 275's original band, for the record.** Unscreened CONSECUTIVE-count jumps among "
        "universe names with a ratio strictly inside (10, 1000) or its reciprocal -- the band "
        f"W7-P2 registered and refuted: **{len(band)}** over "
        f"{band['ticker'].nunique() if len(band) else 0} names. Nineteen are the x1000 family "
        "with buyback or issuance drift between the two filings (the screen catches every one "
        "of them on the median-relative ratio); AIG's 13.3x is the 2011 recapitalisation, kept, "
        "correctly, as a corporate action; BRK-B's 1/54 is a Class-A fact against a 50:1 B "
        "split, now a `treat_as: no_facts` row. The measured median-relative band above "
        "supersedes this one (W7-P2b ruling 1)."
    )
    w("")
    if len(band):
        w("| Ticker | Filed | Previous | New | Ratio | Reading |")
        w("|---|---|---|---|---|---|")
        for row in band.sort_values("ratio").itertuples(index=False):
            r = float(str(row.ratio))
            near_unit = (0.9 * screen.empty_band[1] < r < screen.empty_band[1]) or (
                1.0 / screen.empty_band[1] < r < 1.0 / (0.9 * screen.empty_band[1])
            )
            reading = (
                "unit-scale error with buyback/issuance drift between the two filings"
                if near_unit
                else "a corporate action or class change the actions table lacks; for the operator"
            )
            w(
                f"| {row.ticker} | {pd.Timestamp(str(row.filed)).date()} | {row.previous:,.0f} | "
                f"{row.shares:,.0f} | {r:.4g} | {reading} |"
            )
        w("")
    residual = report.residual_jumps if report.residual_jumps is not None else pd.DataFrame()
    cells, with_cap, names = report.residual_cells
    w(
        f"**The residual the screen leaves (2-10x).** After the screen, {len(residual)} "
        f"consecutive-count jumps over {residual['ticker'].nunique() if len(residual) else 0} "
        "universe names remain -- splits the actions table lacks, class restructurings, "
        "recapitalisations. Cells whose in-force screened count is 2x or more off the name's "
        f"median (an upper bound on the cells such a jump mis-states): **{cells:,} of {with_cap:,} "
        f"({100.0 * cells / max(with_cap, 1):.2f}%) over {names} names.** A known residual, "
        "not screened: the screen is for unit-scale errors, and a 2:1 the vendor lacks is a "
        "different defect with a different fix (the actions table)."
    )
    w("")
    if len(residual):
        w("| Ticker | Filed | Previous | New | Ratio |")
        w("|---|---|---|---|---|")
        ordered = residual.assign(_k=np.abs(np.log(residual["ratio"].astype(float))))
        for row in ordered.sort_values("_k", ascending=False).itertuples(index=False):
            w(
                f"| {row.ticker} | {pd.Timestamp(str(row.filed)).date()} | {row.previous:,.0f} | "
                f"{row.shares:,.0f} | {row.ratio:.4g} |"
            )
        w("")
    w("## Ruling (ii): successor CIKs from a hand table (row 276)")
    w("")
    overrides = report.panel.overrides
    w(
        f"`config/{shares_cfg.overrides_file}` names a predecessor CIK per re-incorporated "
        "registrant, with the EDGAR URL each figure was read from and the date read; the "
        "predecessor's cover-page facts are read beside the mapped CIK's. No automated second "
        f"pass by name. Predecessor rows applied on this pull: {len(overrides)} -- "
        + (
            ", ".join(f"{t} <- CIK {c}" for t, c in sorted(overrides.items()))
            if overrides
            else "none"
        )
        + f". `treat_as: no_facts` rows (W7-P2b ruling 3, a filer whose only non-dimensional "
        f"fact is one class's count): {len(report.panel.excluded)} -- "
        + (
            ", ".join(f"{t} ({v})" for t, v in sorted(report.panel.excluded.items()))
            if report.panel.excluded
            else "none"
        )
        + "."
    )
    w("")
    w("## Ruling (iii): dual-class filers stay `no_facts` (row 277)")
    w("")
    dv = report.no_cap_dollar_volume
    w(
        "No free per-class source exists, so a filer that reports the cover-page fact per class "
        "with a dimension the frames do not carry has no cap and is out of the standardization "
        "set and the cap-weighted market. The quantity missing for these names is the cap "
        "itself, so their cap share cannot be stated; the report states their share of the "
        "universe's DOLLAR VOLUME (close x volume, valid bars) as the proxy, and the equity "
        "descriptor report carries the same number beside the market return's SPY correlation "
        "(row 278). Those two numbers decide whether a per-class XBRL parse is a W7-P2c; "
        "nothing here prejudges it."
    )
    w("")
    last_day = report.holdout_start.date() if dv is None else dv.index[-1].date()
    w(
        f"**`no_facts` names in the universe on {last_day} ({len(report.dual_class_last)}):** "
        f"{_list(report.dual_class_last)}"
    )
    w("")
    if dv is not None:
        w("| Session | Universe names without a cap | Their share of universe dollar volume |")
        w("|---|---|---|")
        idx = pd.DatetimeIndex(dv.index)
        picks = [idx[0], *_year_ends(idx)]
        seen: set[pd.Timestamp] = set()
        for d in picks:
            if d in seen:
                continue
            seen.add(d)
            n_names = int(dv["no_cap_names"].loc[d])
            pct = float(dv["no_cap_dollar_volume_pct"].loc[d])
            w(f"| {d.date()} | {n_names} | {pct:.2f}% |")
        w("")


def _draw(report: CapReport) -> Figure:
    s = report.series
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    top.plot(s.index, s["universe"], color="0.3", lw=1.2, label="estimation universe")
    top.plot(s.index, s["with_cap"], color="tab:blue", lw=1.2, label="with a cap")
    top.fill_between(
        s.index,
        0,
        s["approximated"],
        color="tab:orange",
        alpha=0.35,
        label="cap back-filled (pre-XBRL)",
    )
    top.plot(s.index, s["no_cap"], color="tab:red", lw=1.0, label="no cap")
    top.set_ylabel("names")
    top.legend(loc="upper right", fontsize=8, frameon=False)
    top.set_title("SPEC.md 15.4.1: estimation-universe names with a point-in-time cap, by session")
    bottom.plot(
        s.index,
        s["total_cap_usd_trn"],
        color="tab:blue",
        lw=1.0,
        label=f"screened at R = {report.panel.consistency_ratio:g}"
        if report.panel.consistency_ratio is not None
        else "as filed",
    )
    if report.totals is not None:
        t = report.totals
        bottom.plot(
            t.index,
            t["unscreened"],
            color="tab:red",
            lw=0,
            marker="x",
            ms=5,
            label="year ends, unscreened (as filed)",
        )
        bottom.plot(
            t.index,
            t["published_upper_bound"],
            color="0.2",
            lw=1.0,
            ls="--",
            marker="s",
            ms=3,
            label="published total US equity market (upper bound on the index)",
        )
    bottom.set_yscale("log")
    bottom.set_ylabel("total cap, USD trn (log)")
    bottom.set_xlabel("session")
    bottom.set_title(
        "Total market cap of the estimation universe: as filed, screened (ruling (i)), and the "
        "published bound"
    )
    bottom.legend(loc="upper left", fontsize=8, frameon=False)
    for ax in (top, bottom):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> int:
    cfg = config_mod.load()
    report = build(cfg)
    for v in report.verdicts:
        print(f"row {v.row} {'HOLDS ' if v.holds else 'REFUTED'} {v.leg}: {v.detail}")
    print(
        f"row 273 (not pre-registered) {len(report.jumps)} unexplained jump(s) over "
        f"{report.jumps['ticker'].nunique() if len(report.jumps) else 0} name(s) in the universe"
    )
    figure = _draw(report)
    png = _report_path("equity_market_cap.png")
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    csv = _report_path("equity_market_cap.csv")
    report.series.to_csv(csv, lineterminator="\n", float_format="%.6f")
    mapping_csv = _report_path("equity_cik_mapping.csv")
    report.panel.mapping.to_csv(mapping_csv, lineterminator="\n")
    md = _report_path("equity_market_cap.md")
    md.write_text(render(report, cfg), encoding="utf-8")
    print(f"wrote {md}, {png}, {csv}, {mapping_csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
