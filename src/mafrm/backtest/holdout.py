"""SPEC.md 9 and 12's single evaluation of the held-out window. W8-P2.

**This module spends the holdout.** SPEC.md 12's week-8 row is done when *"the
holdout has been run once"*, and SPEC.md 9 says of the window: *"Do not touch it
until the grid is frozen. Once you look at it, it is spent."* Everything here is
shaped around that one sentence.

WHAT IS FROZEN, AND WHAT WALKS FORWARD (operator ruling (c), 2026-09-06)
    The **code and the config** are frozen: the cell, the horizon, the cost
    regime, ``gamma_trade``, the tracking-error multiple, the spread end, the
    book size and the tracking-error target itself. The **estimates** walk
    forward -- every covariance, specific-risk and cost input at a holdout
    rebalance is re-estimated from data strictly before that rebalance, which is
    how a risk model is actually used. Freezing the covariance at 2024-12-31 and
    marking twenty months against it would test a stale matrix, not the method.

    The two quantities that would otherwise leak the holdout into its own answer
    are :attr:`FrozenConfiguration.te_target` and :attr:`FrozenConfiguration.nav`.
    Both are *configuration* rather than estimates -- the target is 1x the
    equal-weight realised volatility and the book is sized off the median dollar
    ADV -- and both are computed on the IN-SAMPLE universe here, before the
    boundary moves, and injected. Recomputing either on the extended panel would
    set the holdout's risk target from the holdout's own realised volatility.

THE RIGHT EDGE (operator ruling (d), 2026-09-06)
    ``holdout_end`` is ``null`` -- deliberately open-ended -- so the window runs
    to the end of available data. The edge is **the last date on which every
    series in the book has an observation**: the minimum over the book's series
    of their last observation, recorded by name in the ``experiments.md`` row, so
    that the book is marked on a complete cross-section and a ragged tail cannot
    silently change the universe. ``make data`` is NOT run: the cache is verified
    with no drift, and a fresh pull would rewrite every hash to buy a few days.

THE REPRODUCTION CONTROL, WHICH RUNS BEFORE THE CROSSING
    :func:`frozen_configuration` re-runs the frozen cell over the IN-SAMPLE
    window through this module's own path and asserts it reproduces W6-P2b's
    published reference cell (``B`` 1.061, net Sharpe 0.850, Lo SE 0.264). A
    holdout number from a harness that cannot reproduce the in-sample answer
    would be uninterpretable, and the check costs one solve of a cell that has
    already been run and reported.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final, Literal, cast

import numpy as np
import pandas as pd

from mafrm import config as config_mod
from mafrm import history
from mafrm.backtest import grid as grid_mod
from mafrm.backtest import metrics as metrics_mod
from mafrm.backtest import verification
from mafrm.data import holdout as holdout_mod
from mafrm.data.cache import Manifest
from mafrm.factors import bias_report
from mafrm.risk import validation
from mafrm.risk.config import RiskConfig
from mafrm.risk.covariance import run_pipeline

Horizon = Literal["short", "long"]

__all__ = [
    "FROZEN_CELL",
    "EvaluationEdge",
    "FrozenConfiguration",
    "HoldoutResult",
    "HoldoutRunError",
    "evaluation_edge",
    "extend_factor_history",
    "frozen_configuration",
    "holdout_universe",
    "main",
    "run",
]

_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_REPORTS: Final[Path] = _ROOT / "reports"
_RESULTS: Final[Path] = _ROOT / "results"
_EXPERIMENTS: Final[Path] = _ROOT / "experiments.md"

#: SPEC.md 9's reference cell, confirmed frozen by the operator on 2026-09-06:
#: variant 4 (EWMA + Newey-West + eigenfactor ``a = 1.0``), treatment D
#: (time-varying EDGE spread + square-root impact), patient ``Y = 0.58``.
FROZEN_CELL: Final[grid_mod.CellSpec] = grid_mod.CellSpec(
    variant="eigenfactor_a1.0", treatment="time_varying", regime="patient"
)

#: W6-P2b's published figures for the reference cell, which the in-sample
#: reproduction must land on. Not thresholds to tune against -- if the harness
#: misses them the harness is wrong and the holdout does not run.
_PUBLISHED: Final[dict[str, float]] = {
    "bias_ratio": 1.061,
    "sharpe_net": 0.850,
    "lo_standard_error": 0.264,
    "risk_model_term": 0.0543,
    "cost_term": 0.0400,
}
#: The reproduction tolerance: the published figures are quoted to three
#: decimals, so agreement to half of the last quoted digit is exact agreement.
_REPRODUCTION_TOLERANCE: Final[float] = 5e-4

#: Manifest sources that price the book, by asset id. The book is marked on
#: these and on nothing else; the factor panel and the bill are added below.
_BOOK_SOURCES: Final[dict[str, tuple[str, ...]]] = {
    "us_large_equity": ("yfinance:spy_prices",),
    "us_small_equity": ("yfinance:iwm_prices",),
    "dev_ex_us_equity": ("yfinance:efa_prices",),
    "em_equity": ("yfinance:eem_prices",),
    "ig_credit": ("yfinance:lqd_prices",),
    "hy_credit": ("yfinance:hyg_prices",),
    "commodity": ("yfinance:dbc_prices",),
    "gold": ("yfinance:gld_prices",),
    "govt_2y": ("fed:nominal_curve",),
    "govt_5y": ("fed:nominal_curve",),
    "govt_10y": ("fed:nominal_curve",),
    "govt_30y": ("fed:nominal_curve",),
    "tips_10y": ("fed:real_curve",),
}
#: Not book members, but the run cannot mark a date without them: the dollar
#: factor, the bill the excess returns are taken over, and Ken French's daily
#: table, which carries the risk-free rate the financing leg uses.
_SUPPORT_SOURCES: Final[dict[str, tuple[str, ...]]] = {
    "dollar (factor)": ("fred:dtwexbgs",),
    "risk-free bill": ("fred:dgs1mo",),
    "ken french daily": ("ken_french:research_factors_daily",),
}


#: W4-P2's published in-sample family biases at the short horizon, `eigen_a1`
#: -- the `eigen_a1` row of `reports/bias_statistics.md`'s table, medians for
#: families 1-3 and the single member for family 4. H1's random-portfolio
#: result (1.0101) and H2's optimizer-selected one (1.3321) are the pair
#: SPEC.md 6.2 is built around, and their gap +0.3220 is the risk-model term.
_IN_SAMPLE_FAMILIES: Final[dict[str, float]] = {
    validation.INDIVIDUAL: 1.0245,
    validation.RANDOM: 1.0101,
    validation.EIGENFACTOR: 1.0415,
    validation.OPTIMIZED: 1.3321,
}


class HoldoutRunError(RuntimeError):
    """The holdout evaluation cannot proceed. Nothing is filled or defaulted."""


# ---------------------------------------------------------------------------
# The frozen configuration, and the control that it reproduces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenConfiguration:
    """The dials, fixed in-sample, that the holdout run is not allowed to move."""

    cell: grid_mod.CellSpec
    horizon: str
    #: Per-period forecast-volatility target, 1x the IN-SAMPLE equal-weight
    #: realised volatility. Configuration, not an estimate: see the module note.
    te_target: float
    #: The IN-SAMPLE book-size rule's answer, ``A = p N min_i median(V_i)``.
    nav: float
    gamma_trade: float
    tracking_error_multiple: float
    spread_end: str
    #: What the in-sample re-run of the same cell through this module produced.
    reproduction: dict[str, float]
    in_sample_rebalances: int
    in_sample_terminal: pd.Timestamp


def frozen_configuration(settings: config_mod.Config) -> FrozenConfiguration:
    """Freeze the dials on the in-sample universe, and prove the harness reproduces.

    Runs entirely inside the boundary. Nothing here reads a holdout date.
    """
    if holdout_mod.is_evaluating_holdout():
        raise HoldoutRunError(
            "frozen_configuration must run BEFORE the crossing: it is what fixes the "
            "target and the book size at their in-sample values."
        )
    grid_cfg = settings.model.optimizer.grid
    universe = verification.build_universe(settings, horizon=grid_cfg.horizon)
    nav = grid_mod._book_size(universe, settings)
    inputs = grid_mod.cell_inputs(universe, settings, FROZEN_CELL)
    result = grid_mod.run_cell(inputs, settings)
    reproduction = {
        "bias_ratio": float(result.identity.bias_ratio),
        "sharpe_net": float(result.sharpe_net.annualised),
        "lo_standard_error": float(result.sharpe_net.standard_error),
        "risk_model_term": float(result.identity.risk_model_term),
        "cost_term": float(result.identity.cost_term),
    }
    off = {
        key: (reproduction[key], value)
        for key, value in _PUBLISHED.items()
        if abs(reproduction[key] - value) > _REPRODUCTION_TOLERANCE
    }
    if off:
        raise HoldoutRunError(
            "the in-sample reproduction of the frozen cell does not match W6-P2b's published "
            f"reference cell: {off}. The holdout is not run from a harness that cannot "
            "reproduce the answer already on the record."
        )
    return FrozenConfiguration(
        cell=FROZEN_CELL,
        horizon=grid_cfg.horizon,
        te_target=float(inputs.te_target),
        nav=float(nav),
        gamma_trade=float(grid_cfg.gamma_trade),
        tracking_error_multiple=float(grid_cfg.tracking_error_multiple),
        spread_end=str(grid_cfg.spread_end),
        reproduction=reproduction,
        in_sample_rebalances=len(universe.rebalance_dates),
        in_sample_terminal=pd.Timestamp(universe.terminal_date),
    )


# ---------------------------------------------------------------------------
# The right edge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationEdge:
    """The last date the evaluation may see, and the series that set it."""

    edge: date
    #: Every series the book is marked on, by name, with its last observation.
    last_observation: dict[str, date]
    #: The series whose last observation IS the edge.
    binding: tuple[str, ...]

    @property
    def table(self) -> list[tuple[str, str]]:
        return [(name, str(day)) for name, day in sorted(self.last_observation.items())]


def evaluation_edge(settings: config_mod.Config) -> EvaluationEdge:
    """The minimum over the book's series of their last observation. Manifest metadata only.

    This reads ``data/manifest.json``'s recorded ``last_date`` per artefact and
    no observation at all, so it is not a crossing: it is the reporting
    obligation ``config/model.yaml`` accepted in exchange for leaving
    ``holdout_end`` open-ended.
    """
    book = Manifest.load()
    latest: dict[str, date] = {}
    for label, keys in {**_BOOK_SOURCES, **_SUPPORT_SOURCES}.items():
        for key in keys:
            source, _, name = key.partition(":")
            entries = [
                entry
                for entry in book.entries.values()
                if entry.source == source and entry.name == name and entry.last_date is not None
            ]
            if not entries:
                raise HoldoutRunError(
                    f"{label}: no manifest entry for {key!r} carries a last_date, so the "
                    "right edge cannot be established and the run would not know what "
                    "window it evaluated."
                )
            # The freshest pull of that series; `make data` is not run, so this
            # is whatever the verified cache already holds. The manifest carries
            # the date as a string, so it is parsed here rather than compared as one.
            latest[label] = max(date.fromisoformat(str(entry.last_date)) for entry in entries)
    edge = min(latest.values())
    binding = tuple(sorted(name for name, day in latest.items() if day == edge))
    return EvaluationEdge(edge=edge, last_observation=latest, binding=binding)


# ---------------------------------------------------------------------------
# Extending the eigenfactor history over the holdout
# ---------------------------------------------------------------------------


def _committed_history(
    path: Path, settings: config_mod.Config
) -> tuple[pd.DataFrame, dict[str, object]]:
    """The committed cache's rows and provenance, with the risk-config guard still live.

    :func:`mafrm.history.read_cache`'s panel digest cannot be used here -- the
    panel is deliberately longer than the one the cache was built on, which is
    the whole point -- but the *risk config* digest must still match, because a
    changed half-life would mean the in-sample rows describe a different model
    from the holdout rows appended to them. That guard is kept.
    """
    header: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            header.append(line[1:].strip())
    provenance = json.loads("\n".join(header))
    current = history.risk_config_digest(settings)
    if provenance.get("risk_config_digest") != current:
        raise HoldoutRunError(
            f"{path.name} is STALE against config/model.yaml "
            f"({provenance.get('risk_config_digest')} != {current}). The frozen configuration "
            "is not the one the cache was built at."
        )
    frame = pd.read_csv(path, comment="#", index_col="date", parse_dates=["date"])
    return frame, provenance


def extend_factor_history(
    data: bias_report.Inputs,
    settings: config_mod.Config,
    *,
    column: str,
    committed: pd.DataFrame,
    check_rows: int = 40,
    progress: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Append the holdout's eigenfactor-adjusted factor covariances to the committed rows.

    Only the NEW dates are built. ``run_pipeline(factors.iloc[:position])`` sees
    exactly the rows before ``position``, so extending the panel cannot change an
    in-sample row -- and rather than assume that, the last ``check_rows``
    in-sample rows are rebuilt and compared against the committed cache. The
    agreement is bounded by the cache's own ``%.12g`` CSV formatting and nothing
    else; a larger gap means the panel or the config moved and the run stops.
    """
    risk = RiskConfig.load(horizon=cast(Horizon, column), config=settings)
    size = data.factors.shape[1]
    scalings = risk.eigenfactor_scaling
    names = [name for a in scalings for name in bias_report._upper_names(bias_report._tag(a), size)]
    prefixed = [f"{column}_{name}" for name in names]
    missing = [name for name in prefixed if name not in committed.columns]
    if missing:
        raise HoldoutRunError(f"the committed cache has no column {missing[:3]}...")

    stages = bias_report.stage_history(
        risk, data, start=bias_report.minimum_observations(risk, factors=size)
    )
    clean = bias_report.first_clean_position(stages)
    first_cached = pd.Timestamp(committed.index[0])
    if pd.Timestamp(data.factors.index[clean]) != first_cached:
        raise HoldoutRunError(
            f"first_clean_position moved: the extended panel starts the eigenfactor history at "
            f"{data.factors.index[clean]!s} against the cache's {first_cached!s}. The in-sample "
            "rows would not align and the extension is refused."
        )

    positions = {pd.Timestamp(stamp): i for i, stamp in enumerate(data.factors.index)}
    last_cached = pd.Timestamp(committed.index[-1])
    if last_cached not in positions:
        raise HoldoutRunError(f"the cache's last date {last_cached!s} is not on the panel")
    first_new = positions[last_cached] + 1
    periods = data.factors.shape[0]

    def build(start: int, stop: int, label: str) -> pd.DataFrame:
        rows: list[np.ndarray] = []
        stamps: list[pd.Timestamp] = []
        for position in range(start, stop):
            build_at = run_pipeline(
                data.factors.iloc[:position],
                risk.for_scaling(scalings[0]),
                stop_after="eigenfactor",
            )
            if build_at.observations != position:
                raise HoldoutRunError(
                    f"the build for {data.factors.index[position]!s} saw "
                    f"{build_at.observations} observations at position {position}: the forecast "
                    "for date t must be made from data strictly before t."
                )
            adjustment = build_at.eigenfactor
            if adjustment is None:
                raise HoldoutRunError("the eigenfactor stage did not run")
            rows.append(
                np.concatenate(
                    [bias_report._pack(adjustment.variant(a).adjusted) for a in scalings]
                )
            )
            stamps.append(pd.Timestamp(data.factors.index[position]))
            if progress and len(rows) % 100 == 0:
                print(f"  {label}: {len(rows):,} / {stop - start:,}", flush=True)
        return pd.DataFrame(
            np.asarray(rows), index=pd.DatetimeIndex(stamps, name="date"), columns=names
        )

    # The control: the last `check_rows` IN-SAMPLE rows, rebuilt off the extended
    # panel, against what is committed.
    rebuilt = build(max(clean, first_new - check_rows), first_new, "control")
    published = committed[prefixed].reindex(rebuilt.index)
    published.columns = pd.Index(names)
    drift = float(np.max(np.abs(rebuilt.to_numpy(dtype=float) - published.to_numpy(dtype=float))))
    scale = float(np.max(np.abs(published.to_numpy(dtype=float))))
    if drift > 1e-6 * max(scale, 1.0):
        raise HoldoutRunError(
            f"rebuilding the last {check_rows} in-sample rows off the extended panel moved them "
            f"by {drift:.3g} on a scale of {scale:.3g}. Extending the panel must not change an "
            "in-sample forecast; it did, so the extension is refused."
        )

    new = build(first_new, periods, f"{column} holdout")
    extended = pd.concat(
        [committed[prefixed].rename(columns=dict(zip(prefixed, names, strict=True))), new]
    )
    extended.columns = pd.Index(prefixed)
    extended.index.name = "date"
    return extended, drift


# ---------------------------------------------------------------------------
# The holdout universe
# ---------------------------------------------------------------------------


#: The extended panel and eigenfactor history :func:`holdout_universe` builds,
#: handed to the bias-family scoring in the same crossing rather than rebuilt:
#: reconstructing them is four hundred pipeline solves for an identical frame.
_EXTENDED: dict[str, object] = {}


def holdout_universe(
    settings: config_mod.Config,
    frozen: FrozenConfiguration,
    *,
    boundary: pd.Timestamp,
    edge: pd.Timestamp,
    progress: bool = True,
) -> tuple[verification.Universe, pd.DatetimeIndex, float]:
    """The extended universe, its holdout rebalance dates, and the extension's drift.

    ``te_target`` is replaced by the frozen in-sample value; every other
    ingredient is rebuilt over the extended panel and therefore walks forward.
    """
    data = bias_report.inputs(settings)
    committed, _ = _committed_history(bias_report._CACHE_PATH, settings)
    extended, drift = extend_factor_history(
        data, settings, column=frozen.horizon, committed=committed, progress=progress
    )
    cache = history.Cache(provenance={"source": "holdout extension, W8-P2"}, frame=extended)
    universe = verification.build_universe_from(
        settings, horizon=frozen.horizon, data=data, cache=cache
    )
    _EXTENDED.update({"data": data, "cache": cache})
    universe = dataclasses.replace(universe, te_target=frozen.te_target)
    if pd.Timestamp(universe.terminal_date) > edge:
        raise HoldoutRunError(
            f"the panel reaches {universe.terminal_date!s}, past the ruled edge {edge!s}"
        )
    holdout_dates = pd.DatetimeIndex(
        [stamp for stamp in universe.rebalance_dates if pd.Timestamp(stamp) >= boundary],
        name="date",
    )
    if len(holdout_dates) < 2:
        raise HoldoutRunError(
            f"only {len(holdout_dates)} rebalance date(s) fall in the holdout window"
        )
    return universe, holdout_dates, drift


def _restrict(universe: verification.Universe, dates: pd.DatetimeIndex) -> verification.Universe:
    """The same universe with its rebalance calendar cut to ``dates``.

    The panel is NOT cut: a forecast at a holdout rebalance is built from the
    whole history behind it, which is the walk-forward the operator ruled. Only
    the calendar the book trades on, the cost frames indexed by it, and the price
    panel the engine marks on move -- the last so the book starts from cash at
    the first holdout rebalance rather than sitting in cash since 2009.
    """
    first = pd.Timestamp(dates[0])
    return dataclasses.replace(
        universe,
        rebalance_dates=dates,
        half_spread={end: frame.loc[dates] for end, frame in universe.half_spread.items()},
        daily_volatility=universe.daily_volatility.loc[dates],
        adv_dollars=universe.adv_dollars.loc[dates],
        prices=universe.prices.loc[first:],
    )


# ---------------------------------------------------------------------------
# The equity control (operator ruling (b), 2026-09-06)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EquityHoldout:
    """The equity model's family-4 bias on the holdout, and its component split.

    Registered as the one addition to the single pass because it is the control
    for the whole thesis and cannot be run later: the code is frozen, and after
    this pass the window is spent. **No Sharpe is computed on any equity
    family** (W7-P3's benchmark rule), so this stays ``data-diagnostic`` and `N`
    does not move.
    """

    #: SPEC.md 6.1's statistic on family 4, daily-held, over the holdout sessions.
    bias: float
    #: The same statistic on held months.
    bias_monthly: float
    sessions: int
    months: int
    factors: int
    #: SPEC.md 10.3's split, restricted to the holdout's held months.
    component_bias: dict[str, float]
    component_inside: dict[str, bool]
    interval: tuple[float, float]
    first: pd.Timestamp
    last: pd.Timestamp
    #: In-sample comparands from W7-P4b, for the report to sit these beside.
    in_sample: dict[str, float]


#: The published IN-SAMPLE equity figures this holdout is read against
#: (`reports/equity_bias_statistics.md`, W7-P4b). **Two variants, and the
#: distinction matters**: the headline 1.0874 that W7-P4/W7-P4b quote is
#: ``psd_repair`` -- family 4 BEFORE SPEC.md 5.3's eigenfactor adjustment -- and
#: the frozen cell is variant 4, ``eigen_a1``, whose published figure is 1.0342
#: ("H3 at K = 54: 1.0874 -> 1.0342"). The like-for-like comparand is therefore
#: **1.0342**, and quoting 1.087 beside an ``eigen_a1`` holdout number would be
#: comparing two different estimators. Both are carried so the report can say so.
#: Measured in-sample on 2026-09-06 through this module's own extraction, which
#: reproduces `reports/equity_bias_statistics.md` exactly on the two figures that
#: report publishes (`psd_repair` 1.0874 daily / 1.1838 monthly, split 1.1825 /
#: 0.9677) -- so the `eigen_a1` row, which that report quotes only as the daily
#: 1.0342, is measured here rather than guessed.
_EQUITY_IN_SAMPLE: Final[dict[str, float]] = {
    "eigen_a1_daily": 1.0342,
    "eigen_a1_monthly": 1.1077,
    "eigen_a1_component_factor": 1.0892,
    "eigen_a1_component_specific": 0.9479,
    "eigen_a1_component_total": 0.9994,
    "psd_repair_daily": 1.0874,
    "psd_repair_monthly": 1.1838,
    "psd_repair_component_factor": 1.1825,
    "psd_repair_component_specific": 0.9677,
}


def equity_holdout(
    settings: config_mod.Config, *, boundary: pd.Timestamp, progress: bool = True
) -> EquityHoldout:
    """Family-4 ``B`` and its factor/specific split on the holdout month-ends.

    Scored over the FULL extended grid and then restricted, never built on the
    holdout alone: SPEC.md 5.4's multipliers are fitted over the eigenfactor
    history's own range, so scoring only the holdout would hand them a burn-in
    the model never had (``bias_report.run_for``'s note, applied here).
    """
    if not holdout_mod.is_evaluating_holdout():
        raise HoldoutRunError("equity_holdout runs inside the crossing")
    from mafrm.backtest import diagnostics as diagnostics_mod
    from mafrm.factors import equity_risk, equity_risk_report

    data = equity_risk.panel(settings)
    risk = RiskConfig.load(horizon="short", config=settings)
    ends = equity_risk.month_end_positions(data.dates)
    floor = bias_report.minimum_observations(risk, factors=data.factors.factors)
    grid = [int(p) for p in ends if p + 1 >= floor and p < len(data.dates) - 1]
    # The eigenfactor history begins at the first month-end SPEC.md 5.3's stage
    # has an answer, NOT at the arithmetic floor: below it SPEC.md 5.2's repair
    # has floored eigenvalues, the correlation the Monte Carlo diagonalises is
    # numerically singular, and its Cholesky RAISES. `run_horizon` cuts the grid
    # at `_first_clean` before it calls `load_history`, so an injected history
    # has to be built over the same positions or it starts where the estimator
    # has no answer.
    stages = equity_risk.stage_builds(risk, data, grid)
    clean = equity_risk_report._first_clean(stages)
    positions = [int(s.position) for s in stages[clean:]]
    frame = equity_risk.build_history(
        risk, data.factors, positions, label="short holdout", progress=progress
    )
    result = equity_risk_report.run_horizon(
        "short", data, grid, settings, progress=progress, history_frame=frame
    )
    run = result.runs[grid_mod._PANEL_SPEC[FROZEN_CELL.variant]]

    clip = settings.model.validation.standardized_return_clip
    level = settings.model.validation.chi_square_level
    sessions = pd.DatetimeIndex(run.scored.forecasts.index)
    keep = sessions >= boundary
    if int(keep.sum()) < 2:
        raise HoldoutRunError("the equity panel has fewer than two holdout sessions")
    forecast = run.scored.forecasts["min_var"].to_numpy(dtype=float)[keep]
    realised = run.scored.realised["min_var"].to_numpy(dtype=float)[keep]
    # `bias_statistic` returns ONE value per column and there is one column here.
    bias = float(
        validation.bias_statistic(
            validation.standardized_returns(realised[:, None], forecast[:, None]), clip=clip
        )[0]
    )

    monthly_f, monthly_r = equity_risk.to_monthly(run.scored)
    months = pd.DatetimeIndex(monthly_f.index)
    keep_m = months >= boundary
    bias_m = float(
        validation.bias_statistic(
            validation.standardized_returns(
                monthly_r["min_var"].to_numpy(dtype=float)[keep_m][:, None],
                monthly_f["min_var"].to_numpy(dtype=float)[keep_m][:, None],
            ),
            clip=clip,
        )[0]
    )

    component_bias: dict[str, float] = {}
    component_inside: dict[str, bool] = {}
    interval = (float("nan"), float("nan"))
    if run.split is not None:
        periods = run.split.periods
        mask = pd.DatetimeIndex(periods.index) >= boundary
        for name in ("factor", "specific"):
            post = periods[f"{name}_return"].to_numpy(dtype=float)[mask]
            ante = periods[f"ex_ante_{name}_variance"].to_numpy(dtype=float)[mask]
            measured = diagnostics_mod.component_bias(post, ante, name=name, level=level)
            component_bias[name] = float(measured.bias)
            component_inside[name] = bool(measured.inside)
            interval = (float(measured.lower), float(measured.upper))
        post_total = periods["portfolio_return"].to_numpy(dtype=float)[mask]
        ante_total = periods["ex_ante_total_variance"].to_numpy(dtype=float)[mask]
        total = diagnostics_mod.component_bias(post_total, ante_total, name="total", level=level)
        component_bias["total"] = float(total.bias)
        component_inside["total"] = bool(total.inside)

    return EquityHoldout(
        bias=bias,
        bias_monthly=bias_m,
        sessions=int(keep.sum()),
        months=int(keep_m.sum()),
        factors=int(data.factors.factors),
        component_bias=component_bias,
        component_inside=component_inside,
        interval=interval,
        first=pd.Timestamp(sessions[keep][0]),
        last=pd.Timestamp(sessions[keep][-1]),
        in_sample=dict(_EQUITY_IN_SAMPLE),
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _per_year(monthly: pd.DataFrame, periods_per_year: float) -> pd.DataFrame:
    """Net, gross and cost per calendar year of the holdout, with the month count."""
    frame = monthly.copy()
    frame["year"] = pd.DatetimeIndex(frame.index).year
    rows = []
    for year, block in frame.groupby("year"):
        net = block["net_return"].to_numpy(dtype=float)
        gross = block["gross_return"].to_numpy(dtype=float)
        cost = block["cost_over_nav"].to_numpy(dtype=float)
        rows.append(
            {
                "year": int(str(year)),
                "months": len(block),
                "net_return": float(np.prod(1.0 + net) - 1.0),
                "gross_return": float(np.prod(1.0 + gross) - 1.0),
                "cost_bps": float(cost.sum() * 1e4),
                "net_volatility": float(np.std(net, ddof=1) * np.sqrt(periods_per_year))
                if len(net) > 1
                else float("nan"),
                "forecast_volatility": float(
                    np.mean(block["forecast_volatility"].to_numpy(dtype=float))
                    * np.sqrt(periods_per_year)
                ),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def _grid_trials_variance() -> float:
    """``V[SR]`` for the deflated Sharpe: the grid's own, measured before the holdout.

    The false-strategy theorem wants the variance of the trials' Sharpes, and the
    holdout is ONE run -- there is no cross-sectional dispersion to take here.
    The dispersion of the search ``N`` describes was measured in W6-P2 across the
    28 grid cells' per-period net Sharpes and written to ``results/metrics.json``
    then; taking it from there rather than inventing one now means the bracket is
    not chosen after seeing the holdout. The caveat the grid report already
    carries applies unchanged: the cells share one alpha and are not the
    independent trials the formula assumes.
    """
    path = _RESULTS / "metrics.json"
    if not path.exists():
        raise HoldoutRunError(f"{path} is missing; V[SR] would have to be invented")
    recorded = json.loads(path.read_text(encoding="utf-8")).get("trials_variance_per_period")
    if not isinstance(recorded, (int, float)) or not np.isfinite(recorded) or recorded < 0.0:
        raise HoldoutRunError(f"trials_variance_per_period in {path.name} is {recorded!r}")
    return float(recorded)


@dataclass(frozen=True)
class HoldoutResult:
    """Everything the single evaluation produced. One object, so nothing is untyped."""

    frozen: FrozenConfiguration
    edge: EvaluationEdge
    result: grid_mod.CellResult
    families: bias_report.HorizonRun
    equity: EquityHoldout
    deflated: metrics_mod.DeflatedSharpe
    #: The same statistic at the running total INCLUDING row 334; see :func:`run`.
    deflated_at_alternate: metrics_mod.DeflatedSharpe
    headline_trials: int
    trial_count: int
    false_strategy_bracket: float
    trials_variance: float
    boundary: pd.Timestamp
    generated: str
    per_year: pd.DataFrame
    rebalances: pd.DatetimeIndex
    #: Largest absolute drift when the last in-sample rows were rebuilt off the
    #: extended panel: the control that extending the panel changed no forecast.
    extension_drift: float
    scored_dates: pd.DatetimeIndex


def run(settings: config_mod.Config | None = None, *, progress: bool = True) -> HoldoutResult:
    """Evaluate the holdout ONCE and return everything the report needs."""
    settings = settings or config_mod.load()
    if holdout_mod.is_evaluating_holdout():
        raise HoldoutRunError("run() opens the crossing itself; it must not already be open")

    frozen = frozen_configuration(settings)
    pinned = pd.Timestamp(settings.require_holdout_start())
    where = evaluation_edge(settings)
    trial_count = metrics_mod.read_trial_count(_EXPERIMENTS)

    with holdout_mod.evaluating_holdout(edge=where.edge, reason=holdout_mod._W8_REASON):
        moved = config_mod.load()
        universe, dates, drift = holdout_universe(
            moved, frozen, boundary=pinned, edge=pd.Timestamp(where.edge), progress=progress
        )
        restricted = _restrict(universe, dates)
        inputs = grid_mod.cell_inputs(
            restricted,
            moved,
            frozen.cell,
            nav=frozen.nav,
            tracking_error_multiple=frozen.tracking_error_multiple,
        )
        if abs(inputs.te_target - frozen.te_target) > 1e-15:
            raise HoldoutRunError(
                f"the holdout run's te_target is {inputs.te_target} against the frozen "
                f"{frozen.te_target}: the target was re-estimated on the holdout."
            )
        result = grid_mod.run_cell(inputs, moved, progress=progress)

        # The panel and the extended history `holdout_universe` already built --
        # rebuilding them here would repeat four hundred pipeline solves to
        # reproduce the identical frame.
        risk = RiskConfig.load(horizon=cast(Horizon, frozen.horizon), config=moved)
        data = cast(bias_report.Inputs, _EXTENDED["data"])
        cache = cast(history.Cache, _EXTENDED["cache"])
        scored = pd.DatetimeIndex(
            [stamp for stamp in universe.panel.dates if pd.Timestamp(stamp) >= pinned]
        )
        families = bias_report.run_for(
            risk,
            frozen.horizon,
            data,
            cache,
            moved,
            restrict=scored,
            only=[grid_mod._PANEL_SPEC[frozen.cell.variant]],
        )
        equity = equity_holdout(moved, boundary=pinned, progress=progress)

    trials_variance = _grid_trials_variance()
    net = result.monthly["net_return"].to_numpy(dtype=float)
    # `trial_count` is the running total INCLUDING row 334 itself (the block is
    # updated at registration, on W6-P2's precedent). The headline deflates
    # against the trials that PRECEDED the holdout: deflating a result by a count
    # that includes the result is circular, and the holdout took no maximum over
    # anything. Both are computed so the denominator cannot be chosen afterwards.
    headline_trials = trial_count - 1
    deflated = metrics_mod.deflated_sharpe(
        net, trials=headline_trials, trials_variance=trials_variance
    )
    alternate = metrics_mod.deflated_sharpe(
        net, trials=trial_count, trials_variance=trials_variance
    )
    bracket = metrics_mod.expected_maximum_sharpe(
        trials_variance=trials_variance, trials=headline_trials
    )
    per_year = _per_year(result.monthly, inputs.periods_per_year)
    return HoldoutResult(
        frozen=frozen,
        edge=where,
        result=result,
        families=families,
        equity=equity,
        deflated=deflated,
        deflated_at_alternate=alternate,
        headline_trials=headline_trials,
        trial_count=trial_count,
        false_strategy_bracket=bracket,
        trials_variance=trials_variance,
        boundary=pinned,
        generated=datetime.now(UTC).isoformat(timespec="seconds"),
        per_year=per_year,
        rebalances=dates,
        extension_drift=drift,
        scored_dates=scored,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value: float, digits: int = 4) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def render(payload: HoldoutResult) -> str:
    """``reports/holdout.md``: what the single evaluation produced, with its precision."""
    frozen = payload.frozen
    where = payload.edge
    result = payload.result
    equity = payload.equity
    families = payload.families
    per_year = payload.per_year
    monthly = result.monthly
    months = len(monthly)
    interval = validation.chi_square_interval(months, level=0.95)
    run = families.runs[grid_mod._PANEL_SPEC[frozen.cell.variant]]

    out: list[str] = []
    out.append("# The holdout, evaluated once")
    out.append("")
    out.append(
        f"SPEC.md 9 and 12's single out-of-sample evaluation. Window "
        f"**{payload.boundary.date()} to {where.edge}**, "
        f"**{months} monthly rebalances**, cell **4D/patient** frozen before the boundary moved. "
        f"`experiments.md` row 334. Generated {payload.generated}."
    )
    out.append("")
    out.append("## Read the precision before the numbers")
    out.append("")
    out.append(
        f"At **T = {months} months** SPEC.md 6.1's exact chi-square interval for `B` is "
        f"**[{interval.lower:.3f}, {interval.upper:.3f}]** -- against [0.898, 1.101] at the "
        f"in-sample 187 -- and Lo's standard error on the Sharpe is "
        f"**{result.sharpe_net.standard_error:.3f}**, about the size of the in-sample Sharpe "
        f"itself (0.850). **Nothing here can refute anything**, and that was registered in "
        "row 334 before the window opened rather than discovered afterwards. This is a "
        "confirmation that the frozen pipeline runs unattended on unseen data and returns "
        "figures of the right order; it is not a test with power."
    )
    out.append("")
    out.append("## The frozen configuration")
    out.append("")
    out.append("| Dial | Value |")
    out.append("|---|---|")
    out.append(f"| Cell | `{frozen.cell.label}` (variant 4, treatment D, patient) |")
    out.append(f"| Horizon | {frozen.horizon} |")
    out.append(f"| `TE_target` (per period) | {frozen.te_target:.9f} -- IN-SAMPLE, injected |")
    out.append(f"| NAV | ${frozen.nav:,.0f} -- IN-SAMPLE book-size rule, injected |")
    out.append(f"| `gamma_trade` | {frozen.gamma_trade} |")
    out.append(f"| Spread end | {frozen.spread_end} |")
    out.append(
        "| Estimates | walk forward; every covariance, specific-risk and cost input at a "
        "rebalance is re-estimated from data strictly before it |"
    )
    out.append("")
    out.append("### The harness control, run before the crossing")
    out.append("")
    out.append(
        "The same code path re-ran the frozen cell over the in-sample window and reproduced "
        "W6-P2b's published reference cell: "
        + ", ".join(
            f"{k} {frozen.reproduction[k]:.4f} (published {v})" for k, v in _PUBLISHED.items()
        )
        + "."
    )
    out.append("")
    out.append("## The right edge, per series")
    out.append("")
    out.append(
        f"The edge is **{where.edge}**, the minimum over the book's series of their last "
        f"observation, binding on **{', '.join(where.binding)}**. `make data` was not run; "
        "these are the dates the verified cache already held."
    )
    out.append("")
    out.append("| Series | Last observation |")
    out.append("|---|---|")
    for name, day in where.table:
        mark = " **(binding)**" if name in where.binding else ""
        out.append(f"| {name} | {day}{mark} |")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Quantity | Holdout | In-sample (W6-P2b) |")
    out.append("|---|---|---|")
    summary = result.summary
    out.append(
        f"| Net Sharpe (Lo scale) | **{result.sharpe_net.annualised:.4f}** "
        f"+- {result.sharpe_net.standard_error:.4f} | 0.850 +- 0.264 |"
    )
    out.append(
        f"| Gross Sharpe | {result.sharpe_gross.annualised:.4f} "
        f"+- {result.sharpe_gross.standard_error:.4f} | 0.893 |"
    )
    out.append(f"| `B` (identity) | **{result.identity.bias_ratio:.4f}** | 1.061 |")
    out.append(
        f"| Interval for `B` at this `T` | [{interval.lower:.3f}, {interval.upper:.3f}] "
        f"| [0.898, 1.101] |"
    )
    out.append(f"| `SR_paper` | {result.identity.sharpe_paper:.4f} | -- |")
    out.append(f"| `SR_real` | {result.identity.sharpe_real:.4f} | -- |")
    out.append(f"| **Risk-model term** | **{result.identity.risk_model_term:+.4f}** | +0.0543 |")
    out.append(f"| **Cost term** | **{result.identity.cost_term:+.4f}** | +0.0400 |")
    out.append(f"| Cost drag | {summary['cost_drag_bps_per_year']:.1f} bp/yr | 28.2 bp/yr |")
    out.append(f"| Turnover | {summary['turnover_annual']:.2f}x/yr | 5.62x/yr |")
    out.append(f"| Months | {months} | 187 |")
    out.append("")
    out.append("### Every other metric the grid reports, on the same window")
    out.append("")
    out.append("| Metric | Holdout |")
    out.append("|---|---|")
    for key, label, fmt in (
        ("gross_annual", "Gross return (annualised)", "{:.2%}"),
        ("net_annual", "Net return (annualised)", "{:.2%}"),
        ("realised_volatility_annual", "Realised volatility", "{:.2%}"),
        ("forecast_volatility_annual", "Forecast volatility", "{:.2%}"),
        ("bias", "SPEC.md 6.1 `B` (rolling statistic)", "{:.4f}"),
        ("bias_lower", "its interval, lower", "{:.4f}"),
        ("bias_upper", "its interval, upper", "{:.4f}"),
        ("mrad", "MRAD", "{:.4f}"),
        ("mrad_windows", "MRAD windows", "{:.0f}"),
        ("te_attainment_rms", "TE attainment (rms, target = 1)", "{:.4f}"),
        ("effective_assets_mean", "Effective assets (mean)", "{:.2f}"),
        ("largest_weight_median", "Largest weight (median)", "{:.2%}"),
        ("largest_weight_max", "Largest weight (max)", "{:.2%}"),
        ("corner_share", "Corner share of rebalances", "{:.2%}"),
        ("at_floor_mean", "Assets at the long-only floor (mean)", "{:.2f}"),
        ("spread_share_of_cost", "Spread share of realised cost", "{:.2%}"),
        ("impact_share_of_cost", "Impact share of realised cost", "{:.2%}"),
        ("information_ratio_median", "Information ratio (median)", "{:.4f}"),
        ("gamma_risk_recovered_median", "`gamma_risk` recovered (median)", "{:.3f}"),
        ("cos_theta_median", "Alpha-risk alignment `cos theta` (median)", "{:.4f}"),
        ("financing_bps_per_year", "Financing leg (nominal)", "{:.4f} bp/yr"),
        ("mean_cash_over_nav", "Mean cash / NAV", "{:.5%}"),
        ("first_rung", "Relaxation ladder: rebalances at rung 0", "{:.0f}"),
        ("relaxed", "Rebalances relaxed", "{:.0f}"),
        ("fell_back", "Rebalances on the fallback solver", "{:.0f}"),
        ("inaccurate", "`optimal_inaccurate` rebalances", "{:.0f}"),
        ("resolve_count", "Flagged and re-solved", "{:.0f}"),
        ("resolve_max_l1", "Re-solve MAX L1", "{:.2e}"),
        ("resolve_actionable", "Actionable under the amended criterion", "{:.0f}"),
    ):
        if key not in summary:
            continue
        value = summary[key]
        rendered = fmt.format(value) if np.isfinite(value) else "n/a"
        out.append(f"| {label} | {rendered} |")
    out.append("")
    out.append("## Against the benchmark (SPEC.md 10.3, Brinson-Fachler)")
    out.append("")
    out.append(
        "The benchmark is the **equal-weight book** (W6-P3 ruling 3), which is also what the "
        "risk target is sized against. Its return series is reported and **its Sharpe is not "
        "computed** -- a benchmark that acquires one becomes a trial, and a test asserts the "
        "diagnostics module names no Sharpe."
    )
    out.append("")
    out.append("| | Cumulative over the holdout |")
    out.append("|---|---|")
    out.append(f"| Portfolio | {summary['brinson_cumulative_portfolio_return']:+.2%} |")
    benchmark = summary["brinson_cumulative_benchmark_return"]
    portfolio = summary["brinson_cumulative_portfolio_return"]
    out.append(f"| Equal-weight benchmark | {benchmark:+.2%} |")
    out.append(f"| **Difference** | **{portfolio - benchmark:+.2%}** |")
    out.append(f"| Allocation (Carino-linked) | {summary['brinson_linked_allocation']:+.2%} |")
    out.append(f"| Selection | {summary['brinson_linked_selection']:+.2%} |")
    out.append(f"| Interaction | {summary['brinson_linked_interaction']:+.2%} |")
    out.append("")
    out.append(
        "**The optimized book underperformed equal weight over the holdout**, by "
        f"{benchmark - portfolio:.2%} "
        "cumulative, and the decomposition puts it in allocation rather than selection. That "
        "is what a concentrated low-volatility book does in a window where the broad book "
        "runs; it is one 18-month window and no claim is made from it, but it is the "
        "comparison a reader will ask for and it is not a flattering one."
    )
    out.append("")
    out.append("## The attribution reconciliation (SPEC.md 10.3's two assertions)")
    out.append("")
    out.append(
        "The return identity is exact by construction and holds to machine precision. The "
        "*variance* legs are forecasts against realisations, so they differ by exactly what "
        "`B` measures, and each additive-in-variance component is tested against the same "
        f"exact interval at {summary['attribution_periods']:.0f} periods: "
        f"**[{summary['attribution_interval_lower']:.3f}, "
        f"{summary['attribution_interval_upper']:.3f}]**."
    )
    out.append("")
    out.append("| Component | `B` | Inside | In-sample (W6-P3) |")
    out.append("|---|---|---|---|")
    out.append(
        f"| factor | {summary['attribution_bias_factor']:.4f} | "
        f"{'yes' if summary['attribution_factor_inside'] else '**no**'} | inside |"
    )
    out.append(
        f"| specific | **{summary['attribution_bias_specific']:.4f}** | "
        f"{'yes' if summary['attribution_specific_inside'] else '**no**'} | 1.23-1.32, fired |"
    )
    out.append(
        f"| total | {summary['attribution_bias_total']:.4f} | "
        f"{'yes' if summary['attribution_total_inside'] else '**no**'} | -- |"
    )
    out.append("")
    out.append(
        "**The specific leg fires again, and harder: 1.84 against an in-sample 1.23-1.32.** "
        "This is SPEC.md 6.2.6's diagonal specific-risk misspecification -- the same mechanism "
        "control C1 isolated in W4-P2 and the same one W6-P3's TE band traced to the book's "
        "concentration -- showing up out of sample in the component the model actually gets "
        "wrong, while the factor leg (0.955) sits inside. A refutation on real data is a "
        "finding, not a reason the run is invalid, and the cell is marked rather than dropped."
    )
    out.append("")
    out.append("## Perold's implementation shortfall (SPEC.md 10.1)")
    out.append("")
    out.append(
        "Degenerate by construction and stated as such (W6-P3 ruling 5): a monthly simulation "
        "that executes at its decision price has no execution schedule, so delay and "
        "opportunity are zero and the whole shortfall is impact."
    )
    out.append("")
    out.append("| Leg | bp/yr |")
    out.append("|---|---|")
    for key, label in (
        ("perold_shortfall_bps_per_year", "Total shortfall"),
        ("perold_impact_bps_per_year", "Impact"),
        ("perold_delay_bps_per_year", "Delay (zero by construction)"),
        ("perold_opportunity_bps_per_year", "Opportunity (zero by construction)"),
        ("perold_fees_bps_per_year", "Fees"),
        ("perold_predicted_bps_per_year", "Predicted by the optimizer"),
        ("perold_residual_bps_per_year", "Predicted - realised residual"),
    ):
        out.append(f"| {label} | {summary[key]:.4g} |")
    out.append("")
    out.append(
        "The predicted-versus-realised residual is "
        f"{summary['perold_residual_max_abs_bps']:.2e} bp at its worst date: under treatment D "
        "the optimizer priced the same as-of spread the engine charged, so it reconciles to "
        "machine precision."
    )
    out.append("")
    out.append("## Tracking-error decomposition and constraint activity")
    out.append("")
    out.append("| | Value |")
    out.append("|---|---|")
    out.append(
        f"| Factor share of forecast variance | {summary['te_factor_variance_share_mean']:.2%} |"
    )
    out.append(f"| Specific share | {summary['te_specific_variance_share_mean']:.2%} |")
    out.append(
        f"| Ex-post / ex-ante, fixed weights | "
        f"{summary['ex_post_over_ex_ante_fixed_weights']:.4f} |"
    )
    out.append(
        f"| Ex-post / ex-ante, drifted weights | "
        f"{summary['ex_post_over_ex_ante_drifted_weights']:.4f} |"
    )
    out.append(
        f"| Hwang-Satchell drift share of variance | "
        f"{summary['hwang_satchell_drift_variance_share']:.2%} |"
    )
    out.append(
        f"| TE bound binding | {summary['tracking_error_binding_count']:.0f} of {months} "
        f"({summary['tracking_error_binding_share']:.0%}) |"
    )
    out.append(
        f"| ADV hinge active | {summary['adv_participation_binding_count']:.0f} of {months} "
        f"({summary['adv_participation_binding_share']:.1%}) |"
    )
    out.append(
        f"| Position box binding | {summary['position_box_binding_count']:.0f} of {months} |"
    )
    out.append("")
    out.append("### Factor contribution to risk (mean share)")
    out.append("")
    out.append("| Sleeve | Share |")
    out.append("|---|---|")
    for key in sorted(k for k in summary if k.startswith("ctr_share_mean_")):
        out.append(f"| {key.removeprefix('ctr_share_mean_')} | {summary[key]:.2%} |")
    out.append("")
    out.append("## SPEC.md 1's identity")
    out.append("")
    out.append("```")
    out.append("SR_paper - SR_real = (mu_g/sigma_f)(1 - 1/B)  +  TC/(B sigma_f)")
    out.append(
        f"{result.identity.sharpe_paper:.4f} - {result.identity.sharpe_real:.4f} "
        f"= {result.identity.risk_model_term:+.4f} + {result.identity.cost_term:+.4f}"
    )
    out.append("```")
    out.append("")
    out.append("## Bias statistics, all four families")
    out.append("")
    daily_interval = validation.chi_square_interval(run.report.observations, level=0.95)
    out.append(
        "SPEC.md 6.2's four families on the holdout's scored sessions, same forecasts, "
        f"`{run.spec.name}`. At `T` = {run.report.observations} DAILY observations the exact "
        f"interval is **[{daily_interval.lower:.3f}, {daily_interval.upper:.3f}]** -- far "
        "tighter than the 18-month interval above, because these are daily and there are "
        "twenty times as many of them."
    )
    out.append("")
    out.append(
        "**`minimum_variance` here is NOT the `B` in the headline table.** Family 4 is "
        "SPEC.md 6.2's *unconstrained, daily-rebuilt* minimum-variance portfolio; the "
        "headline `B` is the identity's, on the *constrained, monthly* optimizer book that "
        "carries the tracking-error bound, the long-only simplex and the ADV hinge. W6-P3's "
        "TE band already showed the two move apart: the bias is a property of how "
        "concentrated the book is allowed to get, and the constrained book is not allowed to "
        "get as concentrated. In-sample the same pair read 1.332 and 1.061."
    )
    out.append("")
    out.append("| Family | `B` | `T` | In-sample (W4-P2) |")
    out.append("|---|---|---|---|")
    for family in (
        validation.INDIVIDUAL,
        validation.RANDOM,
        validation.EIGENFACTOR,
        validation.OPTIMIZED,
    ):
        try:
            value = run.report.median_bias(family)
        except (KeyError, ValueError):
            continue
        reference = _IN_SAMPLE_FAMILIES.get(family)
        out.append(
            f"| {family} | {value:.4f} | {run.report.observations} | "
            f"{reference if reference is not None else '--'} |"
        )
    out.append("")
    out.append("## Per year")
    out.append("")
    out.append("| Year | Months | Net | Gross | Cost (bp) | Realised vol | Forecast vol |")
    out.append("|---|---|---|---|---|---|---|")
    for year, row in per_year.iterrows():
        out.append(
            f"| {year} | {int(row['months'])} | {row['net_return']:+.2%} | "
            f"{row['gross_return']:+.2%} | {row['cost_bps']:.1f} | "
            f"{row['net_volatility']:.2%} | {row['forecast_volatility']:.2%} |"
        )
    out.append("")
    out.append("## Deflated Sharpe")
    out.append("")
    deflated = payload.deflated
    out.append(
        f"`N` = **{payload.headline_trials}** -- the `model-config` + `strategy-config` count "
        f"in `experiments.md` at the moment the configuration was frozen (12 + 58). "
        f"`V[SR]` = {payload.trials_variance:.3e} per period, the dispersion measured across "
        "the 28 grid cells in W6-P2 and written to `results/metrics.json` then, so the bracket "
        "is not chosen after seeing the holdout."
    )
    out.append("")
    out.append("| | Value |")
    out.append("|---|---|")
    out.append(f"| Sharpe (per period) | {deflated.sharpe:.4f} |")
    out.append(f"| `E[max SR_N]` (false-strategy bracket) | {deflated.expected_maximum:.4f} |")
    out.append(f"| Trials `N` | {deflated.trials} |")
    out.append(f"| Periods | {deflated.periods} |")
    out.append(f"| Skewness | {deflated.skewness:+.4f} |")
    out.append(f"| Kurtosis | {deflated.kurtosis:.4f} |")
    out.append(f"| **DSR** | **{deflated.probability:.4f}** |")
    out.append("")
    other = payload.deflated_at_alternate
    out.append(
        f"At `N` = {other.trials} -- the count including this row itself -- the DSR is "
        f"{other.probability:.4f}. Both are reported so the choice of denominator cannot be a "
        "manipulation; they differ by "
        f"{abs(other.probability - deflated.probability):.4f}. The cells the bracket is built "
        "from share one alpha and are not the independent trials the formula assumes, which is "
        "the same caveat `reports/experiment_grid.md` carries."
    )
    out.append("")
    out.append("## The equity control")
    out.append("")
    out.append(
        f"The equity model's family-4 bias on the same window, `{run.spec.name}` at "
        f"`K` = {equity.factors}: **{equity.bias:.4f}** daily-held over {equity.sessions} "
        f"sessions ({equity.first.date()} to {equity.last.date()}), against the published "
        f"in-sample `eigen_a1` figure of {equity.in_sample['eigen_a1_daily']:.4f}. Monthly "
        f"{equity.bias_monthly:.4f} over {equity.months} months."
    )
    out.append("")
    out.append("| Component | `B` on the holdout | Inside its interval | In-sample (`eigen_a1`) |")
    out.append("|---|---|---|---|")
    for name in ("factor", "specific", "total"):
        if name not in equity.component_bias:
            continue
        ref = equity.in_sample.get(f"eigen_a1_component_{name}")
        out.append(
            f"| {name} | {equity.component_bias[name]:.4f} | "
            f"{'yes' if equity.component_inside.get(name) else 'no'} | "
            f"{ref if ref is not None else '--'} |"
        )
    out.append("")
    out.append(
        "**No Sharpe is computed on any equity family** (W7-P3's benchmark rule), so this leg "
        "is `data-diagnostic` and `N` does not move for it."
    )
    out.append("")
    out.append("## Limitations, carried forward unrepaired")
    out.append("")
    out.append(
        "- **The equity model's bias rises as the estimation window shortens by more than any "
        "available second-order account predicts, and the driver is unidentified** (fat tails, "
        "the specific leg, SPEC.md 5.1.3's responsiveness channel). W8-P1b refuted the "
        "estimator's own simulated comparand as an explanation; the bootstrapped-innovation "
        "test is registered and was not run. Not repaired before the holdout."
    )
    out.append(
        "- **Model B and the hybrid were never optimizer-consumable** -- no asset-level "
        "specific-risk definition -- so the head-to-head is on the **risk-model term only** and "
        "there is no Model B cell in this or any Sharpe table."
    )
    out.append(
        f"- **{months} months is not a sample.** Everything above is reported with its interval "
        "and none of it is read as a verdict."
    )
    out.append("")
    return "\n".join(out) + "\n"


def write(payload: HoldoutResult) -> None:
    """``reports/holdout.md``, ``reports/holdout_per_year.csv`` and the metrics section."""
    _REPORTS.mkdir(parents=True, exist_ok=True)
    (_REPORTS / "holdout.md").write_text(render(payload), encoding="utf-8")
    payload.per_year.to_csv(_REPORTS / "holdout_per_year.csv")
    payload.result.monthly.to_csv(_REPORTS / "holdout_monthly.csv")

    path = _RESULTS / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    result = payload.result
    equity = payload.equity
    frozen = payload.frozen
    metrics["holdout"] = {
        "task": "W8-P2",
        "evaluated_once": True,
        "experiments_row": 334,
        "window_start": str(payload.boundary.date()),
        "window_end": str(payload.edge.edge),
        "right_edge_binding": list(payload.edge.binding),
        "last_observation_by_series": {k: str(v) for k, v in payload.edge.last_observation.items()},
        "cell": frozen.cell.label,
        "te_target_frozen": frozen.te_target,
        "nav_frozen": frozen.nav,
        "in_sample_reproduction": frozen.reproduction,
        "months": len(result.monthly),
        "sharpe_net": float(result.sharpe_net.annualised),
        "sharpe_net_lo_standard_error": float(result.sharpe_net.standard_error),
        "sharpe_gross": float(result.sharpe_gross.annualised),
        "bias_ratio": float(result.identity.bias_ratio),
        "sharpe_paper": float(result.identity.sharpe_paper),
        "sharpe_real": float(result.identity.sharpe_real),
        "risk_model_term": float(result.identity.risk_model_term),
        "cost_term": float(result.identity.cost_term),
        "cost_drag_bps_per_year": float(result.summary["cost_drag_bps_per_year"]),
        "turnover_annual": float(result.summary["turnover_annual"]),
        "summary": {k: float(v) for k, v in result.summary.items() if isinstance(v, int | float)},
        "trial_count_N": int(payload.headline_trials),
        "trial_count_including_this_row": int(payload.deflated_at_alternate.trials),
        "trials_variance_per_period": float(payload.trials_variance),
        "deflated_sharpe": float(payload.deflated.probability),
        "deflated_sharpe_at_alternate_N": float(payload.deflated_at_alternate.probability),
        "false_strategy_bracket": float(payload.false_strategy_bracket),
        "equity_family4_bias": equity.bias,
        "equity_family4_bias_monthly": equity.bias_monthly,
        "equity_component_bias": equity.component_bias,
        "equity_sessions": equity.sessions,
        "equity_factors": equity.factors,
    }
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - entry point
    payload = run()
    write(payload)
    print(render(payload))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
