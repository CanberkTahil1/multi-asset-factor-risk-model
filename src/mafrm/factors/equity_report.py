"""Generates ``reports/equity_descriptor_correlations.{md,png,csv}``.

SPEC.md 15.4 and 15.5 (W7-P2). The six price-only style exposures on the
estimation universe: the correlation matrix of the exposures (time-averaged
daily cross-sectional Pearson, equal- and cap-weighted) and its VIFs; what the
two orthogonalizations removed; the winsorization sensitivity of SPEC.md 15.4.1
ruling 2; the cap-weighted market return against SPY; the coverage the
descriptors have on every session; and the cap-weighted portfolio's exposure
to every factor, which SPEC.md 15.5's centring makes zero. Scores
experiments.md rows 278-281 against ``equity_descriptors.registrations``.

Reads the cache through ``cache.read`` only and the committed reference
tables; nothing here crosses the holdout boundary. The exposures themselves
are :func:`mafrm.factors.equity.build`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from mafrm import config as config_mod
from mafrm.data import sp500, sp500_reference
from mafrm.factors import cap_report, equity, equity_universe, macro
from mafrm.factors.cap_report import Verdict

__all__ = [
    "EquityReport",
    "build",
    "cross_sectional_correlations",
    "render",
    "score",
    "vif",
]

#: The three pairs the orthogonalizations act on.
PAIRS: tuple[tuple[str, str], ...] = (("NLSIZE", "SIZE"), ("RESVOL", "BETA"), ("RESVOL", "SIZE"))


def _report_path(name: str) -> Path:
    return Path(__file__).resolve().parents[3] / "reports" / name


# ---------------------------------------------------------------------------
# Pure pieces
# ---------------------------------------------------------------------------


def _row_corr(x: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    """Correlation matrix of the columns of ``x`` (names x K), equal- or cap-weighted."""
    if w is None:
        return np.asarray(np.corrcoef(x, rowvar=False), dtype=float)
    w = w / w.sum()
    mean = w @ x
    centred = x - mean
    cov = (centred * w[:, None]).T @ centred
    sd = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(cov / np.outer(sd, sd), dtype=float)


def cross_sectional_correlations(
    frames: dict[str, pd.DataFrame],
    *,
    weights: pd.DataFrame | None = None,
    pairs: tuple[tuple[str, str], ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-averaged daily cross-sectional correlation of the frames, plus daily series for pairs.

    On each session the names with EVERY frame present form the cross-section;
    ``weights`` (cap for cap-weighted, sqrt(cap) for the WLS metric of SPEC.md
    15.6) or ``None`` (equal-weighted). Each session is one
    cross-section, not a rolling window, so the daily series are not the
    overlapping-window statistic CLAUDE.md failure mode 9 warns about -- but
    consecutive cross-sections share almost all their names and most of each
    name's descriptor window, so the time average is a summary, not an
    estimate with a standard error.
    """
    names = list(frames)
    first = frames[names[0]]
    sessions = pd.DatetimeIndex(first.index)
    arrays = [frames[n].to_numpy(dtype=float) for n in names]
    caps = (
        weights.reindex(index=sessions, columns=first.columns).to_numpy(dtype=float)
        if weights is not None
        else None
    )
    k = len(names)
    total = np.zeros((k, k))
    count = 0
    pair_index = [(names.index(a), names.index(b)) for a, b in pairs]
    daily = np.full((len(sessions), len(pairs)), np.nan)
    for i in range(len(sessions)):
        x = np.column_stack([a[i] for a in arrays])
        valid = np.isfinite(x).all(axis=1)
        if caps is not None:
            valid &= np.isfinite(caps[i]) & (caps[i] > 0)
        if valid.sum() <= k + 1:
            continue
        corr = _row_corr(x[valid], caps[i, valid] if caps is not None else None)
        if not np.isfinite(corr).all():
            continue
        total += corr
        count += 1
        for j, (a, b) in enumerate(pair_index):
            daily[i, j] = corr[a, b]
    average = pd.DataFrame(total / max(count, 1), index=names, columns=names)
    series = pd.DataFrame(daily, index=sessions, columns=[f"{a}/{b}" for a, b in pairs])
    return average, series


def vif(corr: pd.DataFrame) -> pd.Series:
    """Variance inflation factors from a correlation matrix: ``diag(R^-1)``."""
    inv = np.linalg.inv(corr.to_numpy(dtype=float))
    return pd.Series(np.diag(inv), index=corr.index, name="vif")


@dataclass(frozen=True)
class MarketCheck:
    """The cap-weighted universe excess return against SPY's excess return (row 278)."""

    correlation: float
    sessions: int
    by_year: pd.DataFrame
    tracking_sd_daily: float


def market_against_spy(market: pd.Series, spy: pd.Series) -> MarketCheck:
    common = market.dropna().index.intersection(spy.dropna().index)
    m = market.loc[common].astype(float)
    s = spy.loc[common].astype(float)
    frame = pd.DataFrame({"market": m, "spy": s})
    by_year = (
        frame.groupby(pd.DatetimeIndex(frame.index).year)
        .apply(lambda g: pd.Series({"correlation": g["market"].corr(g["spy"]), "sessions": len(g)}))
        .astype({"sessions": int})
    )
    by_year.index.name = "year"
    return MarketCheck(
        correlation=float(m.corr(s)),
        sessions=len(common),
        by_year=by_year,
        tracking_sd_daily=float((m - s).std(ddof=1)),
    )


@dataclass(frozen=True)
class Sensitivity:
    bound: float
    corr_ew: pd.DataFrame
    vifs: pd.Series


@dataclass(frozen=True)
class EquityReport:
    exposures: equity.Exposures
    corr_ew: pd.DataFrame
    corr_cw: pd.DataFrame
    #: sqrt(cap)-weighted -- the metric of SPEC.md 15.6's WLS, so the collinearity
    #: W7-P3's regression will feel.
    corr_sw: pd.DataFrame
    vifs: pd.Series
    vifs_sw: pd.Series
    #: Daily equal-weighted correlations of the three pairs, before and after.
    pairs_before: pd.DataFrame
    pairs_after: pd.DataFrame
    #: Time-averaged: ``before_ew``, ``after_ew``, ``after_sw``, ``after_cw`` per pair.
    pair_summary: pd.DataFrame
    sensitivity: tuple[Sensitivity, ...]
    market: MarketCheck
    no_cap_dollar_volume: pd.DataFrame
    cap_weighted_exposure_max_abs: pd.Series
    burn_in_sessions: int
    frozen_universe_names: int
    #: Names the committed matrix marks MEMBER (state 1) on a session the cached bars
    #: have no valid bar for, with the cell count.
    ok_without_bars: tuple[str, ...]
    ok_without_bars_cells: int
    verdicts: tuple[Verdict, ...]
    sample_start: pd.Timestamp
    holdout_start: pd.Timestamp


def score(cfg: config_mod.Config, report_parts: dict[str, object]) -> tuple[Verdict, ...]:
    """Rows 278-281 against ``equity_descriptors.registrations``."""
    reg = cfg.model.equity_descriptors.registrations
    market = report_parts["market"]
    summary = report_parts["pair_summary"]
    vifs = report_parts["vifs"]
    sens = report_parts["sensitivity"]
    corr = report_parts["corr_ew"]
    assert isinstance(market, MarketCheck)
    assert isinstance(summary, pd.DataFrame) and isinstance(vifs, pd.Series)
    assert isinstance(corr, pd.DataFrame) and isinstance(sens, tuple)
    verdicts: list[Verdict] = []
    verdicts.append(
        Verdict(
            278,
            "cap-weighted universe vs SPY, daily excess-return correlation",
            market.correlation >= reg.row_278_spy_correlation_min,
            f"{market.correlation:.4f} over {market.sessions} sessions against >= "
            f"{reg.row_278_spy_correlation_min:g}; by year "
            f"{market.by_year['correlation'].min():.4f}-{market.by_year['correlation'].max():.4f}",
        )
    )
    before_nl = abs(float(summary["before_ew"].loc["NLSIZE/SIZE"]))
    before_rv = abs(float(summary["before_ew"].loc["RESVOL/BETA"]))
    verdicts.append(
        Verdict(
            279,
            "(a) before: |corr(NLSIZE_raw, SIZE)| and |corr(RESVOL_raw, BETA)|",
            before_nl > reg.row_279_raw_nlsize_size_min_abs
            and before_rv > reg.row_279_raw_resvol_beta_min_abs,
            f"{before_nl:.3f} against > {reg.row_279_raw_nlsize_size_min_abs:g}; "
            f"{before_rv:.3f} against > {reg.row_279_raw_resvol_beta_min_abs:g}",
        )
    )
    after = summary["after_ew"].abs()
    verdicts.append(
        Verdict(
            279,
            "(b) after: equal-weighted |corr| of the three pairs",
            bool((after < reg.row_279_orthogonalized_max_abs).all()),
            ", ".join(f"{k} {v:.3f}" for k, v in after.items())
            + f" against < {reg.row_279_orthogonalized_max_abs:g}",
        )
    )
    moves_corr: list[float] = []
    moves_vif: list[float] = []
    for s in sens:
        assert isinstance(s, Sensitivity)
        diff = (s.corr_ew - corr).to_numpy(dtype=float).copy()
        np.fill_diagonal(diff, 0.0)
        moves_corr.append(float(np.abs(diff).max()))
        moves_vif.append(float((s.vifs - vifs).abs().max()))
    verdicts.append(
        Verdict(
            280,
            "winsorization sensitivity: largest correlation move, largest VIF move",
            max(moves_corr) <= reg.row_280_correlation_move_max
            and max(moves_vif) <= reg.row_280_vif_move_max,
            f"corr {max(moves_corr):.4f} against <= {reg.row_280_correlation_move_max:g}; "
            f"VIF {max(moves_vif):.4f} against <= {reg.row_280_vif_move_max:g}",
        )
    )
    verdicts.append(
        Verdict(
            281,
            "largest VIF",
            float(vifs.max()) < reg.row_281_vif_max,
            f"{vifs.idxmax()} {float(vifs.max()):.3f} against < {reg.row_281_vif_max:g}",
        )
    )
    return tuple(verdicts)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _ok_without_bars(cfg: config_mod.Config, inputs: equity.Inputs) -> tuple[tuple[str, ...], int]:
    membership = equity_universe.load_membership(cfg)
    window = pd.DatetimeIndex(membership.index)
    window = window[(window >= inputs.sample_start) & (window < inputs.holdout_start)]
    states = membership.reindex(index=window)
    valid = sp500_reference.valid_bars(
        inputs.close.reindex(index=window), inputs.volume.reindex(index=window)
    )
    tickers = [str(t) for t in states.columns]
    valid = valid.reindex(columns=tickers).fillna(False).astype(bool)
    member_ok = states.fillna(sp500.NOT_MEMBER) == sp500.MEMBER
    missing = member_ok & ~valid
    names = tuple(sorted(str(t) for t in missing.columns[missing.any(axis=0)]))
    return names, int(missing.to_numpy().sum())


def build(cfg: config_mod.Config | None = None) -> EquityReport:
    cfg = cfg or config_mod.load()
    desc = cfg.model.equity_descriptors
    exposures, raw, inputs = equity.build(cfg)
    factors = {f: exposures.exposures[f] for f in equity.FACTORS}
    corr_ew, pairs_after = cross_sectional_correlations(factors, pairs=PAIRS)
    corr_cw, pairs_after_cw = cross_sectional_correlations(
        factors, weights=exposures.cap, pairs=PAIRS
    )
    corr_sw, pairs_after_sw = cross_sectional_correlations(
        factors, weights=exposures.cap.pow(0.5), pairs=PAIRS
    )
    before_frames = {
        "NLSIZE": exposures.pre_orthogonal["NLSIZE"],
        "RESVOL": exposures.pre_orthogonal["RESVOL"],
        "SIZE": exposures.exposures["SIZE"],
        "BETA": exposures.exposures["BETA"],
    }
    _, pairs_before = cross_sectional_correlations(before_frames, pairs=PAIRS)
    summary = pd.DataFrame(
        {
            "before_ew": pairs_before.mean(),
            "after_ew": pairs_after.mean(),
            "after_sw": pairs_after_sw.mean(),
            "after_cw": pairs_after_cw.mean(),
        }
    )
    summary.index.name = "pair"
    vifs = vif(corr_ew)

    sensitivity: list[Sensitivity] = []
    for bound in desc.winsorization.sensitivity_bounds_sd:
        alt = equity.exposures_from_raw(
            raw,
            cap=inputs.cap,
            universe=inputs.universe,
            descriptors=desc,
            winsor_bound=bound,
            sample_start=inputs.sample_start,
        )
        alt_corr, _ = cross_sectional_correlations(
            {f: alt.exposures[f] for f in equity.FACTORS}, pairs=()
        )
        sensitivity.append(Sensitivity(bound=bound, corr_ew=alt_corr, vifs=vif(alt_corr)))

    spy = macro.load_excess_return("SPY")
    market = market_against_spy(exposures.market_excess, spy)
    dollar_volume = cap_report.no_cap_dollar_volume_share(
        inputs.close.reindex(index=exposures.universe.index),
        inputs.volume.reindex(index=exposures.universe.index),
        inputs.screen.universe.reindex(
            index=exposures.universe.index, columns=exposures.cap.columns
        ).fillna(False),
        inputs.panel.cap,
    )
    cw_max = exposures.cap_weighted_exposure().abs().max()
    ok_names, ok_cells = _ok_without_bars(cfg, inputs)
    parts: dict[str, object] = {
        "market": market,
        "pair_summary": summary,
        "vifs": vifs,
        "sensitivity": tuple(sensitivity),
        "corr_ew": corr_ew,
    }
    return EquityReport(
        exposures=exposures,
        corr_ew=corr_ew,
        corr_cw=corr_cw,
        corr_sw=corr_sw,
        vifs=vifs,
        vifs_sw=vif(corr_sw),
        pairs_before=pairs_before,
        pairs_after=pairs_after,
        pair_summary=summary,
        sensitivity=tuple(sensitivity),
        market=market,
        no_cap_dollar_volume=dollar_volume,
        cap_weighted_exposure_max_abs=cw_max,
        burn_in_sessions=int(inputs.universe_frozen.sum()),
        frozen_universe_names=int(inputs.universe.loc[inputs.universe_frozen].iloc[0].sum())
        if inputs.universe_frozen.any()
        else 0,
        ok_without_bars=ok_names,
        ok_without_bars_cells=ok_cells,
        verdicts=score(cfg, parts),
        sample_start=inputs.sample_start,
        holdout_start=inputs.holdout_start,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _matrix(frame: pd.DataFrame, fmt: str = "{:+.3f}") -> list[str]:
    lines = ["| | " + " | ".join(frame.columns) + " |", "|---|" + "---|" * len(frame.columns)]
    for name, row in frame.iterrows():
        lines.append(f"| **{name}** | " + " | ".join(fmt.format(float(v)) for v in row) + " |")
    return lines


def render(report: EquityReport, cfg: config_mod.Config) -> str:
    desc = cfg.model.equity_descriptors
    exp = report.exposures
    counts = exp.counts
    first, last = counts.index[0].date(), counts.index[-1].date()
    lines: list[str] = []
    w = lines.append
    w(
        "# Equity descriptor correlations -- the six price-only CNE5 style exposures "
        "(SPEC.md 15.4, 15.5; W7-P2)"
    )
    w("")
    w(
        f"Generated by `mafrm.factors.equity_report`. Daily exposures {first} to {last} "
        f"(holdout from {report.holdout_start.date()}, nothing on or after it read). "
        "BETA, MOMENTUM (RSTR), SIZE (LNCAP), NON-LINEAR SIZE, RESIDUAL VOLATILITY (0.74 DASTD "
        "+ 0.16 CMRA + 0.10 HSIGMA) and LIQUIDITY (0.35 STOM + 0.35 STOQ + 0.30 STOA), every "
        "window and half-life from `equity_descriptors` in `config/model.yaml`. Standardized "
        "with SPEC.md 15.5's asymmetry -- CAP-WEIGHTED mean, EQUAL-WEIGHTED standard deviation "
        f"-- after winsorization at +/-{exp.winsor_bound:g} equal-weighted sd of the raw "
        "descriptor (a READING, SPEC.md 15.4.1 ruling 2). NLSIZE is the cube of standardized "
        "SIZE regressed on SIZE with the cross-sectional regression's sqrt(cap) weights "
        "(USE4's 'regression-weighted basis'; W7-P2b ruling 2), the residual winsorized then "
        "re-standardized; RESVOL is orthogonalized against BETA and SIZE with the same weights "
        "(USE4; ruling 5) and re-standardized. "
        "The market return for BETA is the cap-weighted excess return of the estimation "
        "universe with the previous session's screened caps as weights."
    )
    w("")
    w("## The standardization set, per session")
    w("")
    w(
        "The estimation universe is SPEC.md 15.3's screen (330-488 survivors, not 500; SPEC.md "
        "15.3.2) intersected with the names that have a SCREENED point-in-time cap (SPEC.md "
        "15.4.3 ruling (i)); a name without a cap cannot be cap-centred, cannot carry SIZE or "
        "LIQUIDITY and cannot enter W7-P3's sqrt-cap WLS, so it is out of the set. A descriptor "
        "needs a COMPLETE window (every session valid), so MOMENTUM's 525-session window and "
        "the 252-session windows leave recently listed names without an exposure; the counts "
        "say how many."
    )
    w("")
    w("| Session | Universe | With cap | " + " | ".join(equity.FACTORS) + " |")
    w("|---|---|---|" + "---|" * len(equity.FACTORS))
    idx = pd.DatetimeIndex(counts.index)
    year_ends = [idx[idx.year == y][-1] for y in sorted(set(idx.year))]
    for d in [idx[0], *year_ends]:
        cells = [int(counts[c].loc[d]) for c in ("universe", "with_cap", *equity.FACTORS)]
        w(f"| {d.date()} | " + " | ".join(str(c) for c in cells) + " |")
    w("")
    w(
        f"**Burn-in.** The rolling stage runs from {report.burn_in_sessions} sessions before "
        f"{report.sample_start.date()} so the longest window (RSTR's lag + window = "
        f"{desc.momentum.lag + desc.momentum.window}) is complete on the first sample session. "
        "On those pre-sample sessions the estimation universe is the FIRST sample session's "
        f"universe, frozen ({report.frozen_universe_names} names), used only to form the "
        "cap-weighted market return that BETA's and HSIGMA's first-year windows reach back "
        "into; nothing is standardized before the sample starts. The same survivorship that "
        "SPEC.md 15.3.2 states for the early sample applies to that frozen year."
    )
    w("")
    w(
        f"**Names the committed matrix marks MEMBER with no valid bar on this pull:** "
        f"{len(report.ok_without_bars)} names over {report.ok_without_bars_cells:,} "
        f"(session, name) cells"
        + (f" -- {', '.join(report.ok_without_bars)}." if report.ok_without_bars else ".")
        + " Listed, not silently dropped (W7-P1's discipline); such a cell has no return and "
        "no exposure."
    )
    w("")
    w("## The cap-weighted portfolio's exposure (SPEC.md 15.5)")
    w("")
    w(
        "Cap-weighted centring makes `sum_n w_n X_nk = 0` on every session for every factor, so "
        "the market has no style exposure and the country factor absorbs it (SPEC.md 15.6). "
        "Largest absolute value over all sessions, per factor: "
        + ", ".join(f"{k} {v:.1e}" for k, v in report.cap_weighted_exposure_max_abs.items())
        + ". A unit test pins the same property on a synthetic panel."
    )
    w("")
    w("## The exposure correlation matrix and VIFs (rows 279, 281)")
    w("")
    w(
        "Time average of the DAILY cross-sectional Pearson correlations over the names with "
        "all six exposures, equal-weighted, sqrt(cap)-weighted (the regression's metric, the "
        "one the orthogonalizations zero) and cap-weighted (the centring's metric). Each session "
        "is a separate cross-section, not a rolling window; consecutive sessions still share "
        "almost every name and most of every descriptor window, so the average is a summary "
        "and carries no standard error. VIF = diag(R^-1) of the equal-weighted matrix."
    )
    w("")
    w("**Equal-weighted:**")
    w("")
    lines.extend(_matrix(report.corr_ew))
    w("")
    w(
        "**sqrt(cap)-weighted** -- the metric of SPEC.md 15.6's WLS (`w_n` proportional to "
        "sqrt(cap)), so this is the collinearity W7-P3's regression feels:"
    )
    w("")
    lines.extend(_matrix(report.corr_sw))
    w("")
    w("**Cap-weighted:**")
    w("")
    lines.extend(_matrix(report.corr_cw))
    w("")
    w("| Factor | VIF (equal-weighted) | VIF (sqrt-cap-weighted) |")
    w("|---|---|---|")
    for name in report.vifs.index:
        w(
            f"| {name} | {float(report.vifs.loc[name]):.3f} | "
            f"{float(report.vifs_sw.loc[name]):.3f} |"
        )
    w("")
    w("## What the two orthogonalizations removed (row 279)")
    w("")
    w(
        "SPEC.md 15.4: Non-Linear Size must be orthogonalized against Size or it is ~collinear "
        "with it; Residual Volatility against Beta (and Size) or it is a beta proxy. Before = "
        "the standardized cube of SIZE and the standardized RESVOL composite; after = the "
        "shipped exposures. Time averages of the daily cross-sectional correlations; the "
        "figure shows the daily series."
    )
    w("")
    w(
        "| Pair | Before (equal-weighted) | After (equal-weighted) | After (sqrt-cap-weighted) | "
        "After (cap-weighted) |"
    )
    w("|---|---|---|---|---|")
    for pair in report.pair_summary.index:
        row = report.pair_summary.loc[pair]
        w(
            f"| {pair} | {float(row['before_ew']):+.3f} | {float(row['after_ew']):+.3f} | "
            f"{float(row['after_sw']):+.3f} | {float(row['after_cw']):+.1e} |"
        )
    w("")
    w(
        "Both orthogonalizations use the cross-sectional regression's weights, sqrt(cap) "
        "(USE4: 'orthogonalized ... on a regression-weighted basis'; W7-P2b ruling 2, "
        "superseding W7-P2's cap-weighted reading, whose sqrt-cap NLSIZE/SIZE correlation was "
        "+0.286), so the sqrt-cap column is the one they zero; the equal- and cap-weighted "
        "columns show what is left in the other two metrics. The sqrt-cap 'after' for "
        "NLSIZE/SIZE is not exactly zero because NLSIZE is winsorized AFTER its "
        "orthogonalization (SPEC.md 15.5's one exception) and the clipping re-introduces a "
        "trace; RESVOL is re-standardized without a second clip and its sqrt-cap correlations "
        "are zero to floating precision. Cap-weighted CENTRING is a different operation with a "
        "different purpose and is unchanged."
    )
    w("")
    w("## Winsorization sensitivity (row 280, SPEC.md 15.4.1 ruling 2)")
    w("")
    w(
        f"The bound is +/-{desc.winsorization.bound_sd:g} sd by convention (READING). The "
        "correlation matrix and VIFs re-run at the two sensitivity bounds; a `data-diagnostic` "
        "on the descriptor structure only -- no factor return, no Sharpe. Largest move in any "
        "off-diagonal correlation and in any VIF against the READING:"
    )
    w("")
    w("| Bound (sd) | Largest |corr| move | Largest VIF move | Largest VIF |")
    w("|---|---|---|---|")
    for s in report.sensitivity:
        diff = (s.corr_ew - report.corr_ew).to_numpy(dtype=float).copy()
        np.fill_diagonal(diff, 0.0)
        w(
            f"| {s.bound:g} | {np.abs(diff).max():.4f} | "
            f"{float((s.vifs - report.vifs).abs().max()):.4f} | {float(s.vifs.max()):.3f} |"
        )
    w("")
    w("## The market return against SPY (row 278), and what it is missing (ruling (iii))")
    w("")
    m = report.market
    w(
        f"Daily excess-return correlation between the cap-weighted estimation-universe return "
        f"and SPY's (both less the Ken French daily bill): **{m.correlation:.4f}** over "
        f"{m.sessions} sessions; daily tracking sd {m.tracking_sd_daily * 1e4:.1f} bp. By "
        "calendar year (each a separate, non-overlapping window):"
    )
    w("")
    w("| Year | Correlation | Sessions |")
    w("|---|---|---|")
    for year, row in m.by_year.iterrows():
        w(f"| {year} | {float(row['correlation']):.4f} | {int(row['sessions'])} |")
    w("")
    dv = report.no_cap_dollar_volume
    w(
        "The names the market return cannot weight are the universe names without a cap -- "
        "dual-class filers whose cover-page fact is per class (SPEC.md 15.4.3 ruling (iii)), "
        "plus the unmapped and ambiguous. Their CAP share cannot be stated (the cap is what is "
        "missing); their share of universe DOLLAR VOLUME is the proxy, per session:"
    )
    w("")
    w("| Session | Names without a cap | Share of universe dollar volume |")
    w("|---|---|---|")
    dv_idx = pd.DatetimeIndex(dv.index)
    for d in [dv_idx[0], *[dv_idx[dv_idx.year == y][-1] for y in sorted(set(dv_idx.year))]]:
        n_names = int(dv["no_cap_names"].loc[d])
        pct = float(dv["no_cap_dollar_volume_pct"].loc[d])
        w(f"| {d.date()} | {n_names} | {pct:.2f}% |")
    w("")
    w("## Registered rows (experiments.md 278-281)")
    w("")
    w("| Row | Leg | Verdict | Detail |")
    w("|---|---|---|---|")
    for v in report.verdicts:
        w(f"| {v.row} | {v.leg} | {'HOLDS' if v.holds else 'REFUTED'} | {v.detail} |")
    w("")
    w("## What this is not")
    w("")
    w(
        "- Six of USE4's twelve style factors (SPEC.md 15.8): Book-to-Price, Earnings Yield, "
        "Growth, Leverage and Dividend Yield need point-in-time fundamentals, and the "
        "analyst-estimate descriptors have no free substitute."
    )
    w(
        "- Total, not float-adjusted, caps; a 330-488-name survivor universe; a pre-XBRL "
        "back-filled count before 2009-2011; a screened but not corrected share count (the "
        "2-10x residual of `reports/equity_market_cap.md`)."
    )
    w("- No factor returns yet: the cross-sectional regression is W7-P3 (SPEC.md 15.6).")
    w("")
    return "\n".join(lines)


def _draw(report: EquityReport) -> Figure:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    counts = report.exposures.counts
    top.plot(counts.index, counts["universe"], color="0.3", lw=1.2, label="estimation universe")
    top.plot(
        counts.index, counts["with_cap"], color="tab:blue", lw=1.2, label="with a screened cap"
    )
    top.plot(
        counts.index,
        counts["MOMENTUM"],
        color="tab:orange",
        lw=1.0,
        label="with MOMENTUM (525-session window)",
    )
    top.plot(
        counts.index,
        counts["LIQUIDITY"],
        color="tab:green",
        lw=1.0,
        ls="--",
        label="with LIQUIDITY",
    )
    top.set_ylabel("names")
    top.legend(loc="lower right", fontsize=8, frameon=False)
    top.set_title(
        "SPEC.md 15.4: the standardization set and the names with an exposure, by session"
    )
    before, after = report.pairs_before, report.pairs_after
    for col, color in (("NLSIZE/SIZE", "tab:red"), ("RESVOL/BETA", "tab:purple")):
        bottom.plot(
            before.index,
            before[col],
            color=color,
            lw=0.8,
            alpha=0.5,
            label=f"{col} before orthogonalization",
        )
        bottom.plot(after.index, after[col], color=color, lw=1.0, label=f"{col} after")
    bottom.axhline(0.0, color="0.5", lw=0.6)
    bottom.set_ylabel("daily cross-sectional correlation (equal-weighted)")
    bottom.set_xlabel("session")
    bottom.set_title("What the two orthogonalizations remove (row 279)")
    bottom.legend(loc="lower right", fontsize=8, frameon=False, ncol=2)
    for ax in (top, bottom):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> int:
    cfg = config_mod.load()
    report = build(cfg)
    for v in report.verdicts:
        print(f"row {v.row} {'HOLDS ' if v.holds else 'REFUTED'} {v.leg}: {v.detail}")
    figure = _draw(report)
    png = _report_path("equity_descriptor_correlations.png")
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    csv = _report_path("equity_descriptor_correlations.csv")
    report.corr_ew.to_csv(csv, lineterminator="\n", float_format="%.6f")
    counts_csv = _report_path("equity_descriptor_counts.csv")
    report.exposures.counts.to_csv(counts_csv, lineterminator="\n")
    md = _report_path("equity_descriptor_correlations.md")
    md.write_text(render(report, cfg), encoding="utf-8")
    print(f"wrote {md}, {png}, {csv}, {counts_csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
