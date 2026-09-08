"""Generates ``reports/equity_regression.{md,png,csv}`` and ``reports/equity_industries.{md,csv}``.

SPEC.md 15.6 and 15.7 (W7-P3). The daily cross-sectional regression's
outputs on the estimation universe: the two constraint tests on every date,
the country factor against the cap-weighted market and the size of the gap
the sqrt-cap weights leave, the daily R^2 series against Connor's 39%, the
per-factor ``|t| > 2`` frequencies, the condition numbers, the thin-industry
counts, and -- for ruling 2 -- the current SIC against the first in-sample
10-K's, listed by name. Scores experiments.md rows 283-289 against the
registrations in ``config/model.yaml``.

Reads the cache through ``cache.read`` (and the three undated identifier
tables through :mod:`mafrm.data.sic`); nothing here crosses the holdout
boundary. The regression itself is :mod:`mafrm.factors.equity_regression`.
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
from mafrm.data import sic
from mafrm.factors import cap_report, equity, equity_regression
from mafrm.factors.cap_report import Verdict

__all__ = ["RegressionReport", "build", "render", "score", "t_frequencies"]

#: Trading days in the rolling mean drawn on the R^2 panel. A DISPLAY window,
#: not a parameter: consecutive readings share 62 of 63 observations.
_ROLLING_DAYS = 63


def _report_path(name: str) -> Path:
    return Path(__file__).resolve().parents[3] / "reports" / name


# ---------------------------------------------------------------------------
# Pure pieces
# ---------------------------------------------------------------------------


def t_frequencies(tstats: pd.DataFrame, returns: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    """Per factor: dates present, the share with ``|t| > threshold``, and the
    time-series t of the mean return.

    The frequency is over DAILY cross-sectional t-statistics -- one
    regression per date, no overlap -- so it is a count of independent
    readings. The time-series t (``mean / sd * sqrt(T)``) is information only.
    """
    rows = []
    for col in tstats.columns:
        t = tstats[col].dropna()
        f = returns[col].dropna()
        present = len(t)
        freq = float((t.abs() > threshold).mean() * 100.0) if present else np.nan
        ts_t = float(f.mean() / f.std(ddof=1) * np.sqrt(len(f))) if len(f) > 1 else np.nan
        rows.append(
            {
                "factor": col,
                "dates": present,
                "abs_t_gt_threshold_pct": freq,
                "mean_bp_per_day": float(f.mean() * 1e4) if present else np.nan,
                "sd_bp_per_day": float(f.std(ddof=1) * 1e4) if present > 1 else np.nan,
                "time_series_t": ts_t,
            }
        )
    return pd.DataFrame(rows).set_index("factor")


@dataclass(frozen=True)
class Coverage:
    """Row 283 and the regression set's provenance."""

    universe_with_cap: int
    with_industry: int
    other: int
    no_sic: int
    no_cik: int
    not_pulled: int
    other_names: tuple[str, ...]
    no_sic_names: tuple[str, ...]

    @property
    def industry_pct(self) -> float:
        return 100.0 * self.with_industry / self.universe_with_cap if self.universe_with_cap else 0

    @property
    def other_pct(self) -> float:
        return 100.0 * self.other / self.with_industry if self.with_industry else 0


@dataclass(frozen=True)
class Drift:
    """Row 284: the first in-sample 10-K's SIC against the current one."""

    compared: int
    sic_differs: int
    industry_differs: int
    no_10k: int
    no_sic_in_header: int
    not_pulled: int
    table: pd.DataFrame

    @property
    def sic_pct(self) -> float:
        return 100.0 * self.sic_differs / self.compared if self.compared else 0.0

    @property
    def industry_pct(self) -> float:
        return 100.0 * self.industry_differs / self.compared if self.compared else 0.0


@dataclass(frozen=True)
class RegressionReport:
    factors: equity_regression.FactorReturns
    coverage: Coverage
    drift: Drift
    frequencies: pd.DataFrame
    industry_table: pd.DataFrame
    by_year: pd.DataFrame
    max_industry_term: float
    max_closing: float
    country_vs_regression_set: float
    country_vs_market_excess: float
    gap_sd_ratio: float
    gap_sd_bp: float
    market_sd_bp: float
    no_cap_dollar_volume: pd.DataFrame
    verdicts: tuple[Verdict, ...]
    labels: dict[int, str]
    #: W7-P3b (SPEC.md 15.6.3): regression cells the point-in-time rule re-classified.
    reclassified_cells: int
    regression_cells: int
    history_filings: int
    history_status: dict[str, int]


def coverage_of(
    industries: pd.DataFrame, universe_names: pd.Index, cap_ok: pd.Index, *, unassigned: int
) -> Coverage:
    names = pd.Index([n for n in universe_names if n in cap_ok])
    sub = industries.reindex(names)
    with_ind = sub["industry"].notna()
    other = with_ind & (sub["industry"] == unassigned)
    status = sub["status"].astype(str)
    return Coverage(
        universe_with_cap=len(names),
        with_industry=int(with_ind.sum()),
        other=int(other.sum()),
        no_sic=int((status == "no_sic").sum()),
        no_cik=int((status == "no_cik").sum()),
        not_pulled=int((~with_ind & (status != "no_sic") & (status != "no_cik")).sum()),
        other_names=tuple(sorted(str(t) for t in names[other.to_numpy()])),
        no_sic_names=tuple(sorted(str(t) for t in names[(status == "no_sic").to_numpy()])),
    )


def drift_of(drift: pd.DataFrame, names: pd.Index) -> Drift:
    sub = drift.reindex(names)
    both = sub["sic_differs"].notna()
    status = sub["first_status"].astype(str)
    table = sub[both & sub["industry_differs"].fillna(False).astype(bool)].copy()
    return Drift(
        compared=int(both.sum()),
        sic_differs=int(sub["sic_differs"].fillna(False).astype(bool).sum()),
        industry_differs=int(sub["industry_differs"].fillna(False).astype(bool).sum()),
        no_10k=int((status == "no_10k").sum()),
        no_sic_in_header=int((status == "no_sic_in_header").sum()),
        not_pulled=int((~both & (status != "no_10k") & (status != "no_sic_in_header")).sum()),
        table=table,
    )


def score(cfg: config_mod.Config, report: dict[str, object]) -> tuple[Verdict, ...]:
    """Rows 283-289 against the registrations."""
    ind_reg = cfg.model.equity_industries.registrations
    reg = cfg.model.equity_regression
    r = reg.registrations
    cov = report["coverage"]
    drift = report["drift"]
    summary = report["summary"]
    freq = report["frequencies"]
    assert isinstance(cov, Coverage) and isinstance(drift, Drift)
    assert isinstance(summary, pd.DataFrame) and isinstance(freq, pd.DataFrame)
    out: list[Verdict] = []
    out.append(
        Verdict(
            283,
            "(a) universe names with a cap that have an industry",
            cov.industry_pct >= ind_reg.row_283_industry_coverage_min_pct,
            f"{cov.with_industry} of {cov.universe_with_cap} = {cov.industry_pct:.1f}% against >= "
            f"{ind_reg.row_283_industry_coverage_min_pct:g}% (no_sic {cov.no_sic}, no_cik "
            f"{cov.no_cik}, not pulled {cov.not_pulled})",
        )
    )
    out.append(
        Verdict(
            283,
            "(b) share of those in 49 Other",
            cov.other_pct <= ind_reg.row_283_other_max_pct,
            f"{cov.other} of {cov.with_industry} = {cov.other_pct:.1f}% against <= "
            f"{ind_reg.row_283_other_max_pct:g}%",
        )
    )
    out.append(
        Verdict(
            284,
            "(a) SIC code differs between first in-sample 10-K and now",
            drift.sic_pct < ind_reg.row_284_sic_differs_max_pct,
            f"{drift.sic_differs} of {drift.compared} = {drift.sic_pct:.1f}% against < "
            f"{ind_reg.row_284_sic_differs_max_pct:g}%",
        )
    )
    out.append(
        Verdict(
            284,
            "(b) FF49 industry differs",
            drift.industry_pct < ind_reg.row_284_industry_differs_max_pct,
            f"{drift.industry_differs} of {drift.compared} = {drift.industry_pct:.1f}% against < "
            f"{ind_reg.row_284_industry_differs_max_pct:g}%",
        )
    )
    max_term = float(report["max_industry_term"])  # type: ignore[arg-type]
    max_close = float(report["max_closing"])  # type: ignore[arg-type]
    out.append(
        Verdict(
            285,
            "(a) |cap-weighted industry sum| on every date; (b) identity closes on every date",
            max_term < reg.identity_tolerance and max_close < reg.identity_tolerance,
            f"max |sum w_i f_i| {max_term:.2e}, max |closing| {max_close:.2e} against < "
            f"{reg.identity_tolerance:g} over {int(summary['names'].notna().sum())} dates",
        )
    )
    corr = float(report["country_vs_regression_set"])  # type: ignore[arg-type]
    corr_mkt = float(report["country_vs_market_excess"])  # type: ignore[arg-type]
    ratio = float(report["gap_sd_ratio"])  # type: ignore[arg-type]
    gap_bp = float(report["gap_sd_bp"])  # type: ignore[arg-type]
    mkt_bp = float(report["market_sd_bp"])  # type: ignore[arg-type]
    out.append(
        Verdict(
            286,
            "(a) corr(country, cap-weighted return of the regression set)",
            corr >= r.row_286_country_market_correlation_min,
            f"{corr:.4f} against >= {r.row_286_country_market_correlation_min:g}; against the "
            f"estimation universe's market_excess {corr_mkt:.4f}",
        )
    )
    out.append(
        Verdict(
            286,
            "(b) sd(c'r - f_c) / sd(c'r)",
            ratio < r.row_286_gap_sd_ratio_max,
            f"{ratio:.3f} against < {r.row_286_gap_sd_ratio_max:g} "
            f"({gap_bp:.1f} bp/day against {mkt_bp:.1f})",
        )
    )
    mean_w = float(summary["r2_weighted"].mean())
    mean_e = float(summary["r2_equal"].mean())
    out.append(
        Verdict(
            287,
            "mean weighted daily cross-sectional R^2",
            mean_w >= reg.r_squared_target,
            f"{mean_w:.4f} against >= {reg.r_squared_target:g} (equal-weighted {mean_e:.4f}; "
            f"median weighted {float(summary['r2_weighted'].median()):.4f})",
        )
    )
    thin_median = float(summary["thin_industries"].median())
    style_max = float(summary["cond_style"].max())
    full_median = float(summary["cond_full"].median())
    out.append(
        Verdict(
            288,
            "(a) median count of industries with <= 2 members",
            thin_median <= r.row_288_thin_industries_median_max,
            f"{thin_median:g} against <= {r.row_288_thin_industries_median_max} (range "
            f"{int(summary['thin_industries'].min())}-{int(summary['thin_industries'].max())})",
        )
    )
    out.append(
        Verdict(
            288,
            "(b) largest style-block condition number",
            style_max < r.row_288_style_condition_max,
            f"{style_max:.3f} against < {r.row_288_style_condition_max:g} (median "
            f"{float(summary['cond_style'].median()):.3f})",
        )
    )
    out.append(
        Verdict(
            288,
            "(c) median full design-matrix condition number",
            full_median < r.row_288_full_condition_median_max,
            f"{full_median:.1f} against < {r.row_288_full_condition_median_max:g} (range "
            f"{float(summary['cond_full'].min()):.1f}-{float(summary['cond_full'].max()):.1f})",
        )
    )
    styles = [f for f in equity.FACTORS if f in freq.index]
    freqs = freq.loc[styles, "abs_t_gt_threshold_pct"]
    out.append(
        Verdict(
            289,
            f"every style factor's |t| > {reg.t_stat_threshold:g} frequency",
            bool((freqs > r.row_289_t_frequency_min_pct).all()),
            ", ".join(f"{k} {v:.1f}%" for k, v in freqs.items())
            + f" against > {r.row_289_t_frequency_min_pct:g}%",
        )
    )
    share = float(report["reclassified_pct"])  # type: ignore[arg-type]
    cells = report["reclassified_cells"]
    total = report["regression_cells"]
    assert isinstance(cells, int) and isinstance(total, int)
    low, high = ind_reg.row_290_reclassified_cells_band_pct
    out.append(
        Verdict(
            290,
            "(a) regression cells whose industry the point-in-time rule changed",
            low <= share <= high,
            f"{cells} of {total} = {share:.2f}% against [{low:g}, {high:g}]%",
        )
    )
    r2_move = mean_w - ind_reg.row_290_baseline_r2_weighted
    out.append(
        Verdict(
            290,
            "(b) move in the mean weighted R^2 from W7-P3",
            abs(r2_move) < ind_reg.row_290_r2_move_max,
            f"{mean_w:.4f} - {ind_reg.row_290_baseline_r2_weighted:.4f} = {r2_move:+.4f} against "
            f"|move| < {ind_reg.row_290_r2_move_max:g}",
        )
    )
    moves = {k: float(freqs[k]) - v for k, v in ind_reg.row_290_baseline_t_frequency_pct.items()}
    worst = max(moves, key=lambda k: abs(moves[k]))
    out.append(
        Verdict(
            290,
            "(c) largest move in a style factor's |t| > 2 frequency; NLSIZE still flagged",
            abs(moves[worst]) <= ind_reg.row_290_t_frequency_move_max_pp
            and float(freqs["NLSIZE"]) < r.row_289_t_frequency_min_pct,
            ", ".join(f"{k} {v:+.1f} pp" for k, v in moves.items())
            + f" against |move| <= {ind_reg.row_290_t_frequency_move_max_pp:g} pp; NLSIZE "
            f"{float(freqs['NLSIZE']):.1f}%",
        )
    )
    return tuple(out)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(cfg: config_mod.Config | None = None) -> RegressionReport:
    cfg = cfg or config_mod.load()
    reg = cfg.model.equity_regression
    unassigned = cfg.model.data.ken_french_siccodes49.unassigned_industry
    factors, exposures, inputs = equity_regression.build(cfg)
    summary = factors.summary
    ran = summary["names"].notna()
    s = summary[ran]

    universe = exposures.universe.loc[exposures.universe.index >= exposures.sample_start]
    ever = pd.Index([str(t) for t in universe.columns[universe.any(axis=0)]])
    cap_ok = pd.Index([str(t) for t in exposures.cap.columns[exposures.cap.notna().any(axis=0)]])
    coverage = coverage_of(factors.industries, ever, cap_ok, unassigned=unassigned)
    names_with_cap = pd.Index([n for n in ever if n in cap_ok])
    drift = drift_of(factors.drift, names_with_cap)

    closing = s["cap_weighted_return"] - (
        s["country"] + s["industry_term"] + s["style_term"] + s["cap_weighted_specific"]
    )
    corr_set = float(s["country"].corr(s["cap_weighted_return"]))
    corr_mkt = float(s["country"].corr(s["market_excess"]))
    gap_sd = float(s["identity_gap"].std(ddof=1))
    mkt_sd = float(s["cap_weighted_return"].std(ddof=1))

    freq = t_frequencies(factors.tstats, factors.returns, threshold=reg.t_stat_threshold)
    ind_cols = list(factors.industry_columns)
    members = factors.members.loc[s.index]
    industry_table = pd.DataFrame(
        {
            "dates_present": (members > 0).sum(axis=0),
            "mean_members": members.where(members > 0).mean(axis=0),
            "min_members": members.where(members > 0).min(axis=0),
            "abs_t_gt_threshold_pct": freq.loc[ind_cols, "abs_t_gt_threshold_pct"],
            "sd_bp_per_day": freq.loc[ind_cols, "sd_bp_per_day"],
        }
    )
    years = pd.DatetimeIndex(s.index).year
    by_year = pd.DataFrame(
        {
            "dates": s.groupby(years).size(),
            "r2_weighted": s["r2_weighted"].groupby(years).mean(),
            "r2_equal": s["r2_equal"].groupby(years).mean(),
            "names": s["names"].groupby(years).mean(),
            "industries_present": s["industries_present"].groupby(years).mean(),
            "thin_industries": s["thin_industries"].groupby(years).mean(),
            "cond_full_median": s["cond_full"].groupby(years).median(),
            "cond_style_median": s["cond_style"].groupby(years).median(),
        }
    )
    dollar_volume = cap_report.no_cap_dollar_volume_share(
        inputs.close.reindex(index=exposures.universe.index),
        inputs.volume.reindex(index=exposures.universe.index),
        inputs.screen.universe.reindex(
            index=exposures.universe.index, columns=exposures.cap.columns
        ).fillna(False),
        inputs.panel.cap,
    )
    history = sic.load_history()
    regression_cells = int(s["names"].sum())
    reclassified_cells = int(s["reclassified"].fillna(0).sum())
    parts: dict[str, object] = {
        "coverage": coverage,
        "drift": drift,
        "summary": s,
        "reclassified_cells": reclassified_cells,
        "regression_cells": regression_cells,
        "reclassified_pct": 100.0 * reclassified_cells / regression_cells
        if regression_cells
        else 0,
        "frequencies": freq,
        "max_industry_term": float(s["industry_term"].abs().max()),
        "max_closing": float(closing.abs().max()),
        "country_vs_regression_set": corr_set,
        "country_vs_market_excess": corr_mkt,
        "gap_sd_ratio": gap_sd / mkt_sd if mkt_sd > 0 else np.nan,
        "gap_sd_bp": gap_sd * 1e4,
        "market_sd_bp": mkt_sd * 1e4,
    }
    return RegressionReport(
        factors=factors,
        coverage=coverage,
        drift=drift,
        frequencies=freq,
        industry_table=industry_table,
        by_year=by_year,
        max_industry_term=float(parts["max_industry_term"]),  # type: ignore[arg-type]
        max_closing=float(parts["max_closing"]),  # type: ignore[arg-type]
        country_vs_regression_set=corr_set,
        country_vs_market_excess=corr_mkt,
        gap_sd_ratio=float(parts["gap_sd_ratio"]),  # type: ignore[arg-type]
        gap_sd_bp=gap_sd * 1e4,
        market_sd_bp=mkt_sd * 1e4,
        no_cap_dollar_volume=dollar_volume,
        verdicts=score(cfg, parts),
        labels=factors.labels,
        reclassified_cells=reclassified_cells,
        regression_cells=regression_cells,
        history_filings=len(history),
        history_status={str(k): int(v) for k, v in history["status"].value_counts().items()},
    )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def _table(frame: pd.DataFrame, fmt: dict[str, str] | None = None) -> list[str]:
    fmt = fmt or {}
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join([str(frame.index.name or ""), *cols]) + " |"]
    lines.append("|" + "---|" * (len(cols) + 1))
    for idx, row in frame.iterrows():
        cells = []
        for c in frame.columns:
            v = row[c]
            if pd.isna(v):
                cells.append("--")
            elif c in fmt:
                cells.append(fmt[c].format(v))
            elif isinstance(v, float):
                cells.append(f"{v:.4g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join([str(idx), *cells]) + " |")
    return lines


def render(report: RegressionReport, cfg: config_mod.Config) -> str:
    reg = cfg.model.equity_regression
    f = report.factors
    s = f.summary[f.summary["names"].notna()]
    cov = report.coverage
    dr = report.drift
    first, last = s.index.min().date(), s.index.max().date()
    dv = report.no_cap_dollar_volume
    dv_last = dv.index.max()
    lines: list[str] = []
    lines.append("# Equity cross-sectional regression (SPEC.md 15.6, W7-P3)")
    lines.append("")
    lines.append(
        f"Generated by `python -m mafrm.factors.equity_regression_report`. Daily, {first} to "
        f"{last} ({len(s)} regression dates; exposures, caps and industry shares at t-1 "
        f"against excess returns over t). Every window, weight, tolerance and target is read "
        f"from `config/model.yaml` (`equity_regression`, `equity_industries`, "
        f"`data.ken_french_siccodes49`). Nothing here reaches `sample.holdout_start` "
        f"({f.holdout_start.date()})."
    )
    lines.append("")
    lines.append("## Verdicts (experiments.md rows 283-290)")
    lines.append("")
    for v in report.verdicts:
        lines.append(
            f"- row {v.row} **{'HOLDS' if v.holds else 'REFUTED'}** -- {v.leg}: {v.detail}"
        )
    lines.append("")
    lines.append("## The regression")
    lines.append("")
    lines.append(
        f"USE4 Eq. 3.1: `r_n = f_c + sum_i X_ni f_i + sum_s X_ns f_s + u_n` by weighted least "
        f"squares with `w_n` proportional to `cap_n^{reg.weight_exponent:g}` normalised to one "
        f"-- MSCI's stated assumption that specific variance is inversely proportional to "
        f"sqrt(cap); not cap-weighted, not equal-weighted (CLAUDE.md failure mode 8) -- under "
        f"the CAP-weighted industry sum-to-zero constraint `sum_i w_i f_i = 0` (Eq. 3.3) "
        f"through a K x (K-1) matrix R built on the industries present that date, eliminating "
        f"the largest-cap one. Industries: **Fama-French 49 mapped from EDGAR SIC codes** "
        f"(`{cfg.model.data.ken_french_siccodes49.url}`). GICS is proprietary and is arguably "
        f"the single largest contributor to Barra's explanatory power, so this substitution "
        f"costs more than it appears to: FF49 has no notion of a company's business mix, "
        f"assigns one SIC-based industry per name, and leaves many of its 49 near-empty on a "
        f"~400-name universe (below). Thin-industry rule: **{reg.thin_industry_rule}** "
        f"(SPEC.md 15.6.1 ruling 3) -- an absent industry's return is NaN, a one-member "
        f"industry is estimated as-is."
    )
    lines.append("")
    lines.append(
        f"**Regression set.** The estimation universe minus the no-cap names (dual-class "
        f"filers and BRK-B, `no_facts`), whose share of universe DOLLAR VOLUME on "
        f"{dv_last.date()} is **{float(dv.loc[dv_last, 'no_cap_dollar_volume_pct']):.2f}%** "
        f"({int(dv.loc[dv_last, 'no_cap_names'])} names; a proxy, because the cap is what they "
        f"lack), minus names without a SIC (ruling 2; {cov.no_sic} in the universe: "
        f"{', '.join(cov.no_sic_names) or 'none'}), a complete set of exposures, or a return. "
        f"Per date: {int(s['names'].min())}-{int(s['names'].max())} names (mean "
        f"{float(s['names'].mean()):.0f}), of {int(s['with_cap'].min())}-"
        f"{int(s['with_cap'].max())} with a cap; excluded on the mean date -- no SIC "
        f"{float(s['excluded_no_sic'].mean()):.1f}, no exposure "
        f"{float(s['excluded_no_exposure'].mean()):.1f}, no return "
        f"{float(s['excluded_no_return'].mean()):.1f}. Two known residuals from W7-P2, stated "
        f"and not fixed here: 2.2% of universe cells carry splits the actions table lacks, and "
        f"every pre-first-filing count is a labelled approximation "
        f"(`reports/equity_market_cap.md`)."
    )
    lines.append("")
    lines.append("## The two constraint tests")
    lines.append("")
    lines.append(
        f"On every one of the {len(s)} dates: (a) the cap-weighted sum of the present "
        f"industries' factor returns is zero to **{report.max_industry_term:.2e}** at most "
        f"(tolerance {reg.identity_tolerance:g}); (b) the identity "
        f"`c'r = f_c + sum_i w_i f_i + sum_s (c'x_s) f_s + c'u` over the regression set closes "
        f"to **{report.max_closing:.2e}** at most. The country factor is the cap-weighted "
        f"market return of the regression set **minus the market's specific return** `c'u` "
        f"(plus a residual style term, since the regression set is a subset of the "
        f"standardization set): exact under cap weights, and under the sqrt-cap weights the "
        f"spec mandates exact only up to `c'u` (SPEC.md 15.6.1, derived before the run). "
        f"Measured: corr(f_c, c'r) = **{report.country_vs_regression_set:.4f}**; "
        f"corr(f_c, estimation-universe market_excess) = "
        f"**{report.country_vs_market_excess:.4f}**; the gap `c'r - f_c` has daily sd "
        f"**{report.gap_sd_bp:.1f} bp** against the market's {report.market_sd_bp:.1f} bp, "
        f"ratio **{report.gap_sd_ratio:.3f}**; its mean is "
        f"{float(s['identity_gap'].mean() * 1e4):+.2f} bp/day, of which the style term "
        f"contributes {float(s['style_term'].mean() * 1e4):+.3f} bp/day and the specific term "
        f"{float(s['cap_weighted_specific'].mean() * 1e4):+.2f} bp/day."
    )
    lines.append("")
    lines.append("## Explanatory power")
    lines.append("")
    lines.append(
        f"Mean daily cross-sectional R^2: **{float(s['r2_weighted'].mean()):.4f} weighted** "
        f"(the regression's own metric, scored against SPEC.md 15.7's "
        f"{reg.r_squared_target:g}), {float(s['r2_equal'].mean()):.4f} equal-weighted; medians "
        f"{float(s['r2_weighted'].median()):.4f} / {float(s['r2_equal'].median()):.4f}; "
        f"weighted range {float(s['r2_weighted'].min()):.3f}-"
        f"{float(s['r2_weighted'].max()):.3f}. The figure's rolling mean uses a "
        f"{_ROLLING_DAYS}-day window (consecutive readings share {_ROLLING_DAYS - 1} of "
        f"{_ROLLING_DAYS} observations; the by-year table below is non-overlapping). Connor "
        f"(1995): statistical models 39.0%, fundamental 42.6%, macro 10.9%."
    )
    lines.append("")
    lines.extend(
        _table(
            report.by_year,
            {
                "dates": "{:.0f}",
                "r2_weighted": "{:.4f}",
                "r2_equal": "{:.4f}",
                "names": "{:.0f}",
                "industries_present": "{:.1f}",
                "thin_industries": "{:.1f}",
                "cond_full_median": "{:.1f}",
                "cond_style_median": "{:.3f}",
            },
        )
    )
    lines.append("")
    lines.append("## Per-factor t-statistics (SPEC.md 15.7)")
    lines.append("")
    lines.append(
        f"`|t| > {reg.t_stat_threshold:g}` frequency over DAILY cross-sectional t-statistics "
        f"(one regression per date, no overlap; a noise factor would show about 5%). The "
        f"time-series t of the mean return (`mean / sd * sqrt(T)`) is information, not a test. "
        f"A style factor below the registered {reg.registrations.row_289_t_frequency_min_pct:g}% "
        f"is a pruning candidate flagged to the operator; nothing is pruned here."
    )
    lines.append("")
    style_rows = report.frequencies.loc[[equity_regression.COUNTRY, *f.style_factors]]
    lines.extend(
        _table(
            style_rows,
            {
                "dates": "{:.0f}",
                "abs_t_gt_threshold_pct": "{:.1f}",
                "mean_bp_per_day": "{:+.2f}",
                "sd_bp_per_day": "{:.1f}",
                "time_series_t": "{:+.2f}",
            },
        )
    )
    lines.append("")
    lines.append("## Conditioning and thinness")
    lines.append("")
    lines.append(
        f"2-norm condition number of the matrix the solve inverts, `W^(1/2) X R`: median "
        f"**{float(s['cond_full'].median()):.1f}**, range {float(s['cond_full'].min()):.1f}-"
        f"{float(s['cond_full'].max()):.1f} -- driven by the thinnest present industry's "
        f"column norm. Of the style block `W^(1/2) X_S` alone, the number LIQUIDITY's VIF of "
        f"2.50 is about: median **{float(s['cond_style'].median()):.3f}**, max "
        f"{float(s['cond_style'].max()):.3f}. Industries present per date: "
        f"{int(s['industries_present'].min())}-{int(s['industries_present'].max())} of "
        f"{len(f.industry_columns)}; with at most {reg.thin_industry_report_max_members} "
        f"members: median {float(s['thin_industries'].median()):g}, range "
        f"{int(s['thin_industries'].min())}-{int(s['thin_industries'].max())}, holding "
        f"{float(s['thin_names'].mean()):.1f} names on the mean date. The consequence for "
        f"W7-P4 (ruling 3): a name alone in its industry has a residual of ~0 by construction, "
        f"so its specific variance is understated; SPEC.md 5.5's Bayesian shrinkage (q = 0.1) "
        f"is the published mechanism and W7-P4 measures whether it is enough."
    )
    lines.append("")
    lines.extend(
        _table(
            report.industry_table.sort_values("mean_members", ascending=False),
            {
                "dates_present": "{:.0f}",
                "mean_members": "{:.1f}",
                "min_members": "{:.0f}",
                "abs_t_gt_threshold_pct": "{:.1f}",
                "sd_bp_per_day": "{:.1f}",
            },
        )
    )
    lines.append("")
    lines.append("## Industries: coverage and SIC drift (SPEC.md 15.6.1 rulings 1-2)")
    lines.append("")
    lines.append(
        f"Of the {cov.universe_with_cap} estimation-universe names with a screened cap, "
        f"**{cov.with_industry} ({cov.industry_pct:.1f}%) have a current SIC** through EDGAR's "
        f"submissions record and hence an FF49 industry; {cov.no_sic} have a CIK but no SIC "
        f"(excluded, never 'Other'), {cov.no_cik} no CIK, {cov.not_pulled} not delivered by the "
        f"pull. **{cov.other} ({cov.other_pct:.1f}%) fall in 49 'Other'**: "
        f"{', '.join(cov.other_names) or 'none'}. Drift: among the {dr.compared} names with "
        f"both a current SIC and a first in-sample 10-K header SIC, **{dr.sic_differs} "
        f"({dr.sic_pct:.1f}%) changed SIC code and {dr.industry_differs} ({dr.industry_pct:.1f}%) "
        f"changed FF49 industry**; {dr.no_10k} names have no 10-K on or after "
        f"`sample.start`, {dr.no_sic_in_header} a header without an ASSIGNED-SIC, "
        f"{dr.not_pulled} not delivered. The names whose industry differs are listed in "
        f"`reports/equity_industries.md`; the current SIC is applied backwards by ruling 2, "
        f"and whether the drift is 'small' is the operator's call on these numbers."
    )
    lines.append("")
    lines.append("## The point-in-time correction (SPEC.md 15.6.3, W7-P3b)")
    lines.append("")
    rc = f.reclassified
    lines.append(
        f"For the {dr.industry_differs} universe names whose FF49 industry differs between the "
        f"first in-sample 10-K and now (and every other mapped name the two-point check flags), "
        f"every in-sample 10-K header was pulled ({report.history_filings} in-sample filings: "
        f"{report.history_status}) and the SIC is KNOWN from its filing date, "
        f"forward-filled to the next filing and back-filled to `sample.start` with the first "
        f"filing's SIC (labelled). The other names keep the current code. The rule changed the "
        f"industry of **{report.reclassified_cells} of {report.regression_cells} regression "
        f"cells ({100.0 * report.reclassified_cells / max(report.regression_cells, 1):.2f}%)**, "
        f"for **{len(rc)} names**; the other drifted names changed industry before they entered "
        f"the estimation universe, so their early-sample code never reached a regression. The "
        f"residual the two-point check cannot see, named and not pulled: a change-and-revert "
        f"between two 10-Ks of an unchanged name."
    )
    lines.append("")
    if not rc.empty:
        show = rc.copy()
        show["first_session"] = pd.to_datetime(show["first_session"]).dt.date
        lines.extend(_table(show.sort_values("cells", ascending=False), {"cells": "{:.0f}"}))
        lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(
        "- `reports/equity_regression.csv` -- the per-date summary (names, R^2, condition "
        "numbers, the identity's terms, exclusions).\n"
        "- `reports/equity_factor_returns.csv` -- the daily factor returns, COUNTRY + industries "
        "+ styles (NaN where an industry is absent).\n"
        "- `reports/equity_industries.md` and `.csv` -- per name: CIK, current SIC and industry, "
        "the first in-sample 10-K and its SIC, the drift flags.\n"
        "- `reports/equity_regression.png` -- the R^2 series, the country factor against the "
        "cap-weighted market, the condition numbers and the thin-industry count."
    )
    lines.append("")
    return "\n".join(lines)


def render_industries(report: RegressionReport, cfg: config_mod.Config) -> str:
    cov = report.coverage
    dr = report.drift
    labels = report.labels
    lines = ["# Equity industries: SIC codes and the FF49 mapping (SPEC.md 15.6.1, W7-P3)", ""]
    lines.append(
        f"Generated by `python -m mafrm.factors.equity_regression_report`. Source: EDGAR's "
        f"submissions endpoint (current SIC, one request per CIK) and the SGML header of each "
        f"name's first in-sample 10-K (`{cfg.model.equity_industries.header_url}`); industry "
        f"definitions: `{cfg.model.data.ken_french_siccodes49.url}`. Rows 283-284 are scored "
        f"in `reports/equity_regression.md`."
    )
    lines.append("")
    lines.append(
        f"Universe names with a cap: {cov.universe_with_cap}; with an industry "
        f"{cov.with_industry} ({cov.industry_pct:.1f}%); in 49 Other {cov.other} "
        f"({cov.other_pct:.1f}%); no SIC {cov.no_sic}; no CIK {cov.no_cik}; not pulled "
        f"{cov.not_pulled}. Drift compared on {dr.compared}: SIC differs {dr.sic_differs} "
        f"({dr.sic_pct:.1f}%), industry differs {dr.industry_differs} ({dr.industry_pct:.1f}%); "
        f"no_10k {dr.no_10k}; no_sic_in_header {dr.no_sic_in_header}; not pulled "
        f"{dr.not_pulled}."
    )
    lines.append("")
    lines.append("## Names whose FF49 industry differs between the first in-sample 10-K and now")
    lines.append("")
    if dr.table.empty:
        lines.append("None.")
    else:
        t = dr.table.copy()
        t["industry_first"] = t["industry_first"].map(lambda i: labels.get(int(i), str(i)))
        t["industry_current"] = t["industry_current"].map(lambda i: labels.get(int(i), str(i)))
        t["first_10k_filed"] = pd.to_datetime(t["first_10k_filed"]).dt.date
        show = t[
            [
                "first_10k_filed",
                "sic_first",
                "industry_first",
                "sic_current",
                "industry_current",
            ]
        ]
        show.index.name = "ticker"
        lines.extend(_table(show))
    lines.append("")
    lines.append("## Names in 49 Other")
    lines.append("")
    lines.append(", ".join(cov.other_names) or "None.")
    lines.append("")
    lines.append("## Names with a CIK but no SIC (excluded from the regression)")
    lines.append("")
    lines.append(", ".join(cov.no_sic_names) or "None.")
    lines.append("")
    return "\n".join(lines)


def _draw(report: RegressionReport) -> Figure:
    f = report.factors
    s = f.summary[f.summary["names"].notna()]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    ax = axes[0, 0]
    ax.plot(s.index, s["r2_weighted"], color="0.8", lw=0.5, label="daily, weighted")
    ax.plot(
        s.index,
        s["r2_weighted"].rolling(_ROLLING_DAYS).mean(),
        color="C0",
        lw=1.4,
        label=f"{_ROLLING_DAYS}-day mean (overlap {_ROLLING_DAYS - 1}/{_ROLLING_DAYS})",
    )
    ax.axhline(0.39, color="C3", ls="--", lw=1, label="Connor 39%")
    ax.set_ylim(0, 1)
    ax.set_title("Daily cross-sectional R^2 (sqrt-cap weighted)")
    ax.legend(loc="upper right", fontsize=8)
    ax = axes[0, 1]
    ax.scatter(s["cap_weighted_return"] * 1e4, s["country"] * 1e4, s=3, alpha=0.4, color="C0")
    lim = float(np.nanmax(np.abs(s[["cap_weighted_return", "country"]].to_numpy()))) * 1e4
    ax.plot([-lim, lim], [-lim, lim], color="C3", lw=0.8)
    ax.set_xlabel("cap-weighted return of the regression set, bp")
    ax.set_ylabel("COUNTRY factor return, bp")
    ax.set_title(
        f"Country vs market: corr {report.country_vs_regression_set:.4f}, gap sd "
        f"{report.gap_sd_bp:.1f} bp"
    )
    ax = axes[1, 0]
    ax.plot(s.index, s["cond_full"], color="C1", lw=0.6, label="full W^1/2 X R")
    ax.plot(s.index, s["cond_style"], color="C2", lw=0.8, label="style block")
    ax.set_yscale("log")
    ax.set_title("Condition numbers")
    ax.legend(loc="upper right", fontsize=8)
    ax = axes[1, 1]
    ax.plot(s.index, s["industries_present"], color="C0", lw=0.8, label="industries present")
    ax.plot(
        s.index,
        s["thin_industries"],
        color="C3",
        lw=0.8,
        label=f"with <= {f.thin_max_members} members",
    )
    ax.set_title("FF49 on the regression set")
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("SPEC.md 15.6 cross-sectional regression (W7-P3)")
    fig.tight_layout()
    return fig


def main() -> int:
    cfg = config_mod.load()
    report = build(cfg)
    for v in report.verdicts:
        print(f"row {v.row} {'HOLDS ' if v.holds else 'REFUTED'} {v.leg}: {v.detail}")
    figure = _draw(report)
    png = _report_path("equity_regression.png")
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    report.factors.summary.to_csv(
        _report_path("equity_regression.csv"), lineterminator="\n", float_format="%.6g"
    )
    report.factors.returns.to_csv(
        _report_path("equity_factor_returns.csv"), lineterminator="\n", float_format="%.6e"
    )
    ind = report.factors.drift.join(report.factors.industries[["cik", "sic_description", "status"]])
    ind.to_csv(_report_path("equity_industries.csv"), lineterminator="\n")
    _report_path("equity_regression.md").write_text(render(report, cfg), encoding="utf-8")
    _report_path("equity_industries.md").write_text(
        render_industries(report, cfg), encoding="utf-8"
    )
    print(f"wrote {_report_path('equity_regression.md')} and companions")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
