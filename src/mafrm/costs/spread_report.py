"""Generates ``reports/spread_vs_vol.png``. SPEC.md 3.4.

The figure the spec calls "the single most persuasive figure in the repo": the
EDGE effective spread estimated on rolling 63-day windows, drawn against
realised volatility across 2008, 2020 and 2022, with the flat 5bp assumption
that an unestimated cost model would use drawn as a horizontal line. The
distance between the flat line and the estimated series during a crisis is the
argument for estimating spreads at all.

It carries one qualification, drawn on the figure rather than buried here.
:func:`mafrm.costs.spread.zero_spread_reference` simulates a market with **no
spread at all** and reports what the estimator returns on it; the band that
measurement produces is shaded on the figure, and the estimated series is never
corrected by it.

The first version of this figure drew that band an order of magnitude too wide.
It was built from ``bidask``'s default, which returns ``|estimate|``, so the
band showed ``E|X| = 0.798*sd(X)`` -- a positive number on any input, mistaken
for a finite-sample property of EDGE. With signed estimates the estimator's bias
on a zero-spread market is approximately zero, and what the band now shows is
the residual left by clipping negatives to zero, ``E[max(X,0)] = 0.399*sd(X)``,
about half the old width. See experiments.md rows 54-58.

One consequence of the correction is visible in how the top panel is drawn. A
fifth to a half of each asset's monthly estimates are negative before clipping,
and a log axis can render neither a negative nor the zero they become. Drawing
them anyway produces a picket fence of vertical strokes to the axis floor that
buries the series. They are therefore not connected: the line BREAKS at every
month the estimator could not resolve, and each such month gets a tick on the
rug along the bottom. Nothing is dropped, and the density of that rug per asset
is itself the reading -- it is the same quantity the committed table reports as
the negative share.

CLAUDE.md invariant 5: everything here stops strictly before
``sample.holdout_start``. The three episodes shown sit well inside the in-sample
window, but the series is truncated regardless, so that the figure cannot start
including holdout months the day the cache is refreshed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display in CI; must precede the pyplot import.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from mafrm import config
from mafrm.costs import spread as spread_mod
from mafrm.data import cache, calendar, prices

__all__ = ["SpreadFigureData", "build", "figure_path", "main", "render_table", "table_path"]

#: Assets highlighted by name. The rest of the sleeve is drawn thin and grey.
#: Chosen as a liquidity ladder -- the most liquid equity ETF in existence, an
#: investment-grade credit fund, a high-yield fund and emerging markets -- so the
#: reader can see the level order as well as the time variation.
_HIGHLIGHT: tuple[str, ...] = ("SPY", "LQD", "HYG", "EEM")

#: Ticks per day for the two ends of the indicative bias band. The bias falls as
#: the assumed intraday process gets finer; these bracket a plausible range for
#: an exchange-traded fund rather than claiming a single right answer.
_BIAS_TICKS: tuple[int, int] = (6000, 390)

#: Bottom of the "unresolved month" rug, in bp, and the multiplicative spacing
#: between each highlighted asset's row of it. The rug sits below the tightest
#: true ETF spread in the sleeve so it cannot be mistaken for an estimate; the
#: rows are separated so four assets' ticks do not overprint each other.
#: SPY's measured daily volatility and overnight variance share over 2023-2024,
#: used as the reference conditions for the recovery study. Measured from the
#: cached raw bars, in-sample; they are inputs to a diagnostic simulation and
#: touch no model, which is why they live here rather than in model.yaml.
_SPY_VOLATILITY = 0.0081
_SPY_GAP_SHARE = 0.39

#: True spreads the recovery table reports, and how many independent windows
#: each row averages. Chosen to bracket the sleeve: 1bp is tighter than any real
#: ETF, 50bp wider than any calm-period estimate in the panel.
_RECOVERY_SPREADS_BPS: tuple[float, ...] = (1.0, 5.0, 20.0, 50.0)
_RECOVERY_WINDOWS = 500

_RUG_BASE_BPS = 0.10
_RUG_SPACING = 1.6

_TRADING_DAYS = 252


def figure_path() -> Path:
    """Where the committed figure lives."""
    return Path(__file__).resolve().parents[3] / "reports" / "spread_vs_vol.png"


def table_path() -> Path:
    """Where the committed negative-incidence and crisis-multiple table lives."""
    return Path(__file__).resolve().parents[3] / "reports" / "spread_bias_correction.md"


@dataclass(frozen=True)
class SpreadFigureData:
    """Everything the figure draws, so it can be inspected and tested apart from it."""

    #: Monthly production spread in bp -- signed, negatives clipped to zero.
    spread_bps: pd.DataFrame
    #: The same estimates in bp BEFORE the clip. What any average must read.
    signed_bps: pd.DataFrame
    #: Annualised realised volatility in percent, same index and columns.
    realised_vol_pct: pd.DataFrame
    #: Indicative zero-spread bias in bp at each observed volatility, (low, high).
    #: Drawn from the CLIPPED mean, which is what the production series carries.
    bias_band: tuple[pd.Series, pd.Series]
    #: Per-asset incidence of negative estimates, counted before the clip.
    negatives: dict[str, spread_mod.NegativeEstimates]
    #: The holdout boundary the series was truncated at.
    truncated_at: pd.Timestamp


def _load(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = cache.Manifest.load()
    key = ticker.lower()
    bars = cache.read(manifest.latest(source="yfinance", name=f"{key}_prices"), manifest=manifest)
    actions = cache.read(
        manifest.latest(source="yfinance", name=f"{key}_actions"), manifest=manifest
    )
    return bars, actions


def build(cfg: config.Config | None = None) -> SpreadFigureData:
    """Estimate spreads and realised volatility for every cached sleeve ETF."""
    cfg = cfg or config.load()
    costs = cfg.model.costs
    edge = costs.edge_spread
    holdout = pd.Timestamp(cfg.require_holdout_start())

    spreads: dict[str, pd.Series] = {}
    vols: dict[str, pd.Series] = {}
    for ticker in cfg.universe.yfinance_tickers:
        try:
            bars, actions = _load(ticker)
        except cache.CacheError:
            continue

        daily_spread = spread_mod.effective_spread(
            bars,
            window=edge.window,
            min_periods=edge.min_periods,
            signed=edge.signed_estimates,
            label=ticker,
        )
        # Realised volatility comes from ADJUSTED total returns while the spread
        # comes from RAW bars. That asymmetry is the point of SPEC.md 3.4: a
        # spread is a property of traded prices, a return is not.
        returns = prices.total_return(bars, actions)
        window = costs.spread_vs_volatility.realised_volatility_window
        daily_vol = returns.rolling(window=window, min_periods=window).std() * np.sqrt(
            _TRADING_DAYS
        )

        # The one canonical month end (mafrm.data.calendar): the last date this
        # asset actually traded, never the calendar 31st and never a holiday.
        estimated = daily_spread.dropna()
        picks = calendar.month_end_dates(calendar.trading_dates(estimated, label=ticker))
        picks = picks[picks < holdout]
        if len(picks) == 0:
            continue
        spreads[ticker] = daily_spread.reindex(picks)
        vols[ticker] = daily_vol.reindex(picks)

    if not spreads:
        raise cache.CacheError(
            "no cached ETF bars found; run `make data` before regenerating this figure"
        )

    index = sorted({stamp for series in spreads.values() for stamp in series.index})
    signed_frame = pd.DataFrame({k: v for k, v in spreads.items()}, index=pd.DatetimeIndex(index))
    vol_frame = pd.DataFrame({k: v for k, v in vols.items()}, index=pd.DatetimeIndex(index))
    signed_frame.index.name = vol_frame.index.name = "date"

    # Negatives are counted on the SIGNED frame, before the clip. Counting after
    # it returns zero by construction -- the reading that let W1-P5 conclude the
    # estimator was well behaved on tight names when it had simply been handed
    # an absolute value.
    negatives = {
        ticker: spread_mod.count_negative(signed_frame[ticker]) for ticker in signed_frame.columns
    }
    production = signed_frame
    if edge.clip_negative_to_zero:
        clipped = spread_mod.clip_negative(signed_frame)
        assert isinstance(clipped, pd.DataFrame)
        production = clipped

    # The indicative zero-spread bias, as a TIME series so it can be read against
    # the estimates directly. It is a function of volatility only, so it is
    # evaluated once on a grid and interpolated onto each month's sleeve-median
    # realised volatility rather than simulated 380 times.
    median_daily_vol = vol_frame.median(axis=1) / np.sqrt(_TRADING_DAYS)
    grid = np.linspace(0.0, float(np.nanmax(median_daily_vol.to_numpy())), 16)
    bands = []
    for ticks in _BIAS_TICKS:
        curve = np.array(
            [
                spread_mod.zero_spread_reference(
                    daily_volatility=float(v),
                    window=edge.window,
                    seed=cfg.model.seed,
                    ticks_per_day=ticks,
                ).clipped_mean
                / 1e-4
                for v in grid
            ]
        )
        bands.append(
            pd.Series(
                np.interp(median_daily_vol.to_numpy(), grid, curve),
                index=median_daily_vol.index,
            )
        )

    return SpreadFigureData(
        spread_bps=production / 1e-4,
        signed_bps=signed_frame / 1e-4,
        realised_vol_pct=vol_frame * 100.0,
        bias_band=(bands[0], bands[1]),
        negatives=negatives,
        truncated_at=holdout,
    )


def _draw(data: SpreadFigureData, cfg: config.Config) -> Figure:
    costs = cfg.model.costs
    flat = costs.flat_spread_assumption_bps
    windows = costs.spread_vs_volatility.crisis_windows

    fig = plt.figure(figsize=(13.5, 9.0), dpi=160)
    grid = fig.add_gridspec(2, len(windows), height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.22)
    top = fig.add_subplot(grid[0, :])

    # Months the estimator could not resolve -- signed estimate at or below zero,
    # clipped to zero for production -- BREAK the line rather than drawing it down
    # to the axis. Connecting through them would paint a picket fence that hides
    # the series, and drawing them at zero on a log axis is impossible anyway.
    # They are not dropped: each one gets a tick on the rug at the bottom, so the
    # reader sees how much of each asset's history the estimator cannot resolve.
    def _resolved(ticker: str) -> pd.Series:
        series = data.spread_bps[ticker]
        return series.where(series > 0.0).dropna()

    def _unresolved(ticker: str) -> pd.DatetimeIndex:
        series = data.spread_bps[ticker].dropna()
        return pd.DatetimeIndex(series.index[series <= 0.0])

    for ticker in data.spread_bps.columns:
        if ticker in _HIGHLIGHT:
            continue
        series = _resolved(ticker)
        top.plot(series.index, series.to_numpy(), lw=0.7, color="0.78", zorder=1)
    palette = {"SPY": "#1f4e79", "LQD": "#c1666b", "HYG": "#e08e45", "EEM": "#4f7942"}
    for offset, ticker in enumerate(_HIGHLIGHT):
        if ticker not in data.spread_bps.columns:
            continue
        series = _resolved(ticker)
        top.plot(
            series.index, series.to_numpy(), lw=1.5, color=palette[ticker], label=ticker, zorder=3
        )
        unresolved = _unresolved(ticker)
        top.plot(
            unresolved,
            np.full(len(unresolved), _RUG_BASE_BPS * _RUG_SPACING**offset),
            marker="|",
            ls="none",
            ms=3.5,
            color=palette[ticker],
            alpha=0.75,
            zorder=3,
        )

    for window in windows:
        # matplotlib converts datetimes on the axis at draw time; its stubs type
        # `axvspan` for floats only, so the ignore is a stub gap, not a cast.
        top.axvspan(
            pd.Timestamp(window.start),  # type: ignore[arg-type]
            pd.Timestamp(window.end),  # type: ignore[arg-type]
            color="#d62728",
            alpha=0.08,
            zorder=0,
        )
        top.annotate(
            window.label,
            # Just above the rug and clear of the legend, which occupies the
            # upper right where two of the three windows fall.
            xy=(pd.Timestamp(window.start), 0.27),
            xycoords=("data", "axes fraction"),
            fontsize=8.5,
            color="#8b1a1a",
            ha="left",
            va="bottom",
        )

    low, high = data.bias_band
    top.fill_between(
        low.index,
        low.to_numpy(),
        high.to_numpy(),
        color="#6a5acd",
        alpha=0.16,
        lw=0,
        zorder=2,
        label="zero-spread bias after clipping (indicative)",
    )

    top.axhline(flat, color="black", ls="--", lw=1.4, zorder=4)
    top.annotate(
        f"the flat {flat:.0f}bp assumption",
        xy=(0.012, flat),
        xycoords=("axes fraction", "data"),
        fontsize=9,
        va="bottom",
        fontweight="bold",
    )
    top.set_yscale("log")
    top.set_ylim(bottom=_RUG_BASE_BPS / 1.6)
    top.set_ylabel("EDGE effective spread (bp, log scale)")
    top.set_title(
        "Estimated effective spreads are not flat, and they widen exactly when a risk model "
        "wants to trade\n"
        f"EDGE on rolling {cfg.model.costs.edge_spread.window}-day windows of RAW OHLC, "
        f"monthly step; in-sample only, truncated at {data.truncated_at.date()}",
        fontsize=11.5,
        loc="left",
    )
    # One proxy entry for the rug: without it the ticks read as data near zero
    # rather than as the months where there is no estimate to draw.
    handles, labels = top.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="|", ls="none", color="0.35", ms=6, label="unresolved"))
    labels.append("month clipped to zero (no resolvable spread)")
    top.legend(handles, labels, loc="upper right", ncols=3, fontsize=8.5, framealpha=0.92)
    top.grid(alpha=0.25, which="both", lw=0.5)

    for column, window in enumerate(windows):
        axis = fig.add_subplot(grid[1, column])
        mask = (data.spread_bps.index >= pd.Timestamp(window.start)) & (
            data.spread_bps.index <= pd.Timestamp(window.end)
        )
        # The sleeve median reads the PRODUCTION frame -- signed, then clipped.
        # That is the authors' own prescription for averaging studies, and it is
        # not the obvious choice: clipping is one-sided and adds back ~0.399*sd
        # (experiments.md row 55). Their stated reason for preferring it anyway
        # is that "more negative estimates are typically associated with larger
        # spreads empirically", so keeping the negatives biases an average DOWN
        # in a way that correlates with the quantity being measured. Between two
        # biases the published guidance wins; the cost is recorded, not hidden.
        spread_median = data.spread_bps[mask].median(axis=1)
        vol_median = data.realised_vol_pct[mask].median(axis=1)

        axis.plot(
            spread_median.index,
            spread_median.to_numpy(),
            color="#1f4e79",
            lw=1.9,
            label="sleeve median spread",
        )
        axis.axhline(flat, color="black", ls="--", lw=1.2)
        axis.set_ylabel("spread (bp)" if column == 0 else "")
        axis.set_title(f"{window.label}  ({window.start:%Y-%m} to {window.end:%Y-%m})", fontsize=10)
        axis.grid(alpha=0.25, lw=0.5)
        axis.tick_params(axis="x", labelrotation=30, labelsize=8)

        twin = axis.twinx()
        twin.plot(
            vol_median.index,
            vol_median.to_numpy(),
            color="#8b1a1a",
            lw=1.3,
            ls=":",
            label="realised vol (rhs)",
        )
        twin.set_ylabel("realised vol (%/yr)" if column == len(windows) - 1 else "", fontsize=9)
        if column == 0:
            handles = axis.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
            labels = axis.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
            axis.legend(handles, labels, fontsize=8, loc="upper left")

    negative_share = np.mean([report.fraction for report in data.negatives.values()])
    fig.text(
        0.008,
        0.012,
        "Spread from RAW unadjusted OHLC (adjusted closes contaminate the high-low range); "
        "realised volatility from dividend-adjusted total returns. Estimates are SIGNED "
        "(bidask sign=True);\n"
        f"{100 * negative_share:.0f}% of monthly estimates are negative across the sleeve before "
        "being clipped to zero, which is the authors' prescription for averaging.\n"
        "Clipping is one-sided and leaves about half the bias it replaces, so the LEVEL remains "
        "untrustworthy for the most liquid names and no cost-model calibration may use it until "
        "the issuer-disclosed\nmedian comparand of SPEC.md 3.4 is fitted -- see experiments.md "
        "rows 54-58. Ardia, Guidotti & Kroencke (2024), JFE 161:103916.",
        fontsize=7.4,
        color="0.35",
        va="bottom",
    )
    fig.subplots_adjust(left=0.062, right=0.945, top=0.925, bottom=0.135)
    return fig


def _crisis_multiple(frame: pd.DataFrame, window: config.CrisisWindow) -> tuple[float, float]:
    """Sleeve-median spread in the 12 months before a crisis, and its peak inside it.

    This is the statistic W1-P5 row 52 reported as "30.3 -> 131.9bp (4.3x)": a
    median over the preceding twelve months against the highest monthly sleeve
    median inside the window. Reproduced here rather than recomputed a second
    way, so the corrected numbers are comparable with the ones they replace.
    """
    start, end = pd.Timestamp(window.start), pd.Timestamp(window.end)
    median = frame.median(axis=1)
    before = median[(median.index >= start - pd.DateOffset(months=12)) & (median.index < start)]
    inside = median[(median.index >= start) & (median.index <= end)]
    return float(before.median()), float(inside.max())


def render_table(data: SpreadFigureData, cfg: config.Config) -> str:
    """The committed record of what the sign correction changed. experiments.md rows 54-58."""
    edge = cfg.model.costs.edge_spread
    baseline_floor = cfg.model.costs.spread_vs_volatility.minimum_baseline_for_ratio_bps
    lines = [
        "# EDGE spreads: the absolute-value correction",
        "",
        "Generated by `make report` from `mafrm.costs.spread_report`. Every number is",
        "in basis points, in-sample only, truncated strictly before",
        f"`sample.holdout_start` = {data.truncated_at.date()}.",
        "",
        f"`costs.edge_spread.signed_estimates` = {str(edge.signed_estimates).lower()}, ",
        f"`costs.edge_spread.clip_negative_to_zero` = {str(edge.clip_negative_to_zero).lower()}.",
        "",
        "## Negative estimates per asset, counted BEFORE the clip",
        "",
        "`bidask` returns `|estimate|` unless asked for signed estimates, so under the",
        "previous configuration this column was 0.0% for every asset by construction,",
        "and that was read as evidence the estimator was well behaved on tight names.",
        "A fraction near one half means the estimator is centred on zero and resolving",
        "nothing at that window length.",
        "",
        "| asset | months | negative | share | most negative | median \\|est\\| (old) "
        "| median signed | median production |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ticker in data.signed_bps.columns:
        signed = data.signed_bps[ticker].dropna()
        report = data.negatives[ticker]
        most = f"{report.most_negative / 1e-4:.1f}" if report.most_negative is not None else "--"
        lines.append(
            f"| {ticker} | {signed.size} | {report.negative} | {100 * report.fraction:.1f}% "
            f"| {most} | {signed.abs().median():.1f} | {signed.median():.1f} "
            f"| {data.spread_bps[ticker].dropna().median():.1f} |"
        )

    lines += [
        "",
        "## Crisis widening, on the definition W1-P5 row 52 used",
        "",
        "Sleeve median over the 12 months before the window, against the highest",
        "monthly sleeve median inside it. The `old` columns are the same statistic on",
        "`|estimate|`, which is what row 52 reported.",
        "",
        "**The widening survives the correction and is larger in basis points. The",
        "MULTIPLE mostly does not survive, and that is a property of the statistic,",
        "not of the spreads.** A ratio needs a denominator, and row 52's denominator",
        "was a median of absolute values, which cannot approach zero. With signed",
        "estimates the calm-period baseline falls to roughly zero for two of the three",
        "windows, so the ratio is reported only where that baseline exceeds",
        f"`minimum_baseline_for_ratio_bps` = {baseline_floor:.0f}bp and the widening in",
        "basis points -- which is always defined -- is given alongside it.",
        "",
        "| window | old pre | old peak | old multiple | new pre | new peak | new multiple "
        "| new widening |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    absolute = data.signed_bps.abs()
    for window in cfg.model.costs.spread_vs_volatility.crisis_windows:
        old_pre, old_peak = _crisis_multiple(absolute, window)
        new_pre, new_peak = _crisis_multiple(data.spread_bps, window)
        new_multiple = (
            f"{new_peak / new_pre:.2f}x"
            if new_pre >= baseline_floor
            else "not defined (baseline ~0)"
        )
        lines.append(
            f"| {window.label} | {old_pre:.1f} | {old_peak:.1f} | {old_peak / old_pre:.2f}x "
            f"| {new_pre:.1f} | {new_peak:.1f} | {new_multiple} "
            f"| +{new_peak - new_pre:.0f}bp |"
        )
    lines += [
        "",
        "## Recovery of KNOWN spreads at the same 63-day window",
        "",
        "The zero-spread control above tests the NULL. It says nothing about whether",
        "a spread can be resolved, which is the question that decides whether this",
        "estimator can deliver a level. Simulated markets at SPY's measured 2023-24",
        f"daily volatility ({100 * _SPY_VOLATILITY:.2f}%) and overnight variance share",
        f"({_SPY_GAP_SHARE:.2f}), {_RECOVERY_WINDOWS} INDEPENDENT windows per row.",
        "",
        "`sd` is the spread of a SINGLE window's estimate and does not shrink with",
        "more windows; `SNR` is the true spread over it. **Below 1, one window cannot",
        "tell the spread from zero.**",
        "",
        "| true | median est | mean est | per-window sd | SNR | negative |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for true_bps in _RECOVERY_SPREADS_BPS:
        rec = spread_mod.recovery_reference(
            true_spread=true_bps * 1e-4,
            daily_volatility=_SPY_VOLATILITY,
            window=edge.window,
            seed=cfg.model.seed,
            windows=_RECOVERY_WINDOWS,
            overnight_gap_share=_SPY_GAP_SHARE,
        )
        lines.append(
            f"| {true_bps:.0f}bp | {rec.median / 1e-4:.1f} | {rec.mean / 1e-4:.1f} "
            f"| {rec.standard_deviation / 1e-4:.1f} | {rec.signal_to_noise:.2f} "
            f"| {100 * rec.negative_fraction:.0f}% |"
        )
    lines += [
        "",
        "The estimator has **no positive resolution floor** -- at a true 1bp it returns",
        "a median near zero and goes negative half the time -- and it **recovers 50bp**",
        "cleanly. So neither the implementation nor the estimator is at fault. What",
        "binds is per-window noise: at 63 daily bars the standard deviation of one",
        "estimate is around 20bp regardless of the truth, so no spread below roughly",
        "20bp is resolvable from a single window. The authors say as much -- a few",
        'daily prices give "potentially large estimation uncertainty", and "the higher',
        'the frequency, the better" (Ardia, Guidotti & Kroencke, package FAQ).',
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """Regenerate the figure and the table. Wired to ``make report``."""
    cfg = config.load()
    data = build(cfg)

    figure = _draw(data, cfg)
    path = figure_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"wrote {path} ({data.spread_bps.shape[1]} assets, {len(data.spread_bps)} months)")

    table = table_path()
    table.write_text(render_table(data, cfg), encoding="utf-8")
    print(f"wrote {table}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
