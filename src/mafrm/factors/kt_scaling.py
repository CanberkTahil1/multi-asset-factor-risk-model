"""SPEC.md 15.1's ``K/T`` scaling result: the points, their comparands, the tracking test. W8-P1.

WHY THIS LIVES IN ``factors/``
------------------------------

:mod:`mafrm.risk.shepard` turns a parameter count and an effective sample size
into a multiplier and knows nothing else. *This* module knows which panel a
point comes from, which half-life it was built at, that an equity month-end
build carries a point-in-time ``K_d``, and which registration in
``experiments.md`` each number is compared against. None of that may cross
into ``risk/`` (CLAUDE.md invariant 10).

THE RULINGS THIS IMPLEMENTS (SPEC.md 15.1.1, operator, 2026-09-06)
------------------------------------------------------------------

1. **"Tracks"** means: the curve's predicted ``B`` lies inside the point's own
   exact chi-square interval at the number of MONTHS the point's ``B`` pools
   over, in the unit ``B`` is measured in. ``B`` is a ratio of standard
   deviations, so the curve is ``(1 - K/T)^-1`` in that unit and
   ``(1 - K/T)^-2`` in variance; both are drawn, the test is in sd units.
   The direction is registered PER MODEL (``validation.kt_scaling.expected_tracking``)
   and the reversing result is one point at a time.
2. **The x-axis** is ``K_d / T_eff`` with ``T_eff`` the realised Kish size
   averaged over the scored dates each ``B`` pools over; the asymptotic
   ``2 tau / ln 2`` is an annotation and the per-date range is printed beside
   each point.
3. **The equity sweep** moves the factor-volatility half-life alone over the
   macro sweep's grid, pre-eigenfactor, on the month-end grid.
4. **The comparand's ``T`` is the formula's own** (task item iii): the
   correlation window for a factor-model point, because that is the window the
   estimator re-estimates ``rho`` on (SPEC.md 6.4.2); the volatility window for
   the one-window naive comparand. Eq. 32 at the volatility window is carried
   beside every factor point as SPEC.md 6.4.3's UPPER BOUND, never as the
   comparand.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from mafrm import config, history
from mafrm.backtest import diagnostics
from mafrm.factors import bias_report, equity_risk
from mafrm.numerics import effective_sample_size, ewma_weights, realised_effective_sample_size
from mafrm.risk import second_order, shepard, validation
from mafrm.risk.checks import assert_psd
from mafrm.risk.config import RiskConfig
from mafrm.risk.covariance import (
    correlation_from_covariance,
    ewma_second_moment,
    separated_ewma_covariance,
)

__all__ = [
    "EquityControl",
    "EquityPoint",
    "EquitySweep",
    "KtScalingError",
    "MacroPoint",
    "Point",
    "SimulatedComparand",
    "SubperiodSplit",
    "correlation_kish",
    "equity_control",
    "equity_second_order",
    "equity_sweep",
    "held_months",
    "macro_points",
    "make_point",
    "predicted_bias",
    "subperiod_split",
    "tracking_verdict",
]

#: The two families the chart reads; the rest of the ladder is not scored here.
_FAMILIES: Final[tuple[str, ...]] = (validation.RANDOM, validation.OPTIMIZED)


class KtScalingError(ValueError):
    """A point could not be built as the rulings require. Nothing is filled or defaulted."""


# ---------------------------------------------------------------------------
# The arithmetic: Kish at the correlation window, and the comparand
# ---------------------------------------------------------------------------


def correlation_kish(rows: int, halflife: float) -> float:
    """Kish's effective size of an EWMA at ``halflife`` fed ``rows`` observations.

    The correlation window's ``T`` on a real, finite window -- what ruling (iii)'s
    comparand is evaluated at. Equal to ``2 tau / ln 2`` once ``rows`` is long
    relative to the half-life, and smaller before that.
    """
    if rows < 1:
        raise KtScalingError(f"correlation_kish: rows must be positive, got {rows}")
    return realised_effective_sample_size(ewma_weights(rows, halflife))


def predicted_bias(parameters: int, effective_observations: float) -> shepard.SecondOrderRisk:
    """Shepard's multiplier at ``K`` (or ``N``) and a realised ``T``; both units live on it."""
    return shepard.second_order_risk(parameters, effective_observations=effective_observations)


# ---------------------------------------------------------------------------
# A point on the chart
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Point:
    """One ``(K/T_eff, B)`` point with its comparand, its interval and its verdict."""

    model: str
    label: str
    halflife: int
    #: ``K`` (a factor model; the mean ``K_d`` over the scored dates for equity) or ``N``.
    parameters: float
    parameters_min: int
    parameters_max: int
    #: Realised Kish sizes averaged over the scored dates, both windows.
    kish_volatility: float
    kish_correlation: float
    #: Ruling 2: ``K / mean Kish`` at the volatility window, and the per-date range.
    k_over_t: float
    k_over_t_min: float
    k_over_t_max: float
    k_over_t_asymptotic: float
    #: Ruling 1: the monthly ``B`` and the point's OWN exact interval at that many months --
    #: the values of the true bias the measurement is consistent with, ``[B / u_T, B / l_T]``
    #: where ``[l_T, u_T]`` is the chi-square interval under the null at ``T`` months.
    bias_monthly: float
    months: int
    interval_lower: float
    interval_upper: float
    bias_daily: float
    days: int
    #: Ruling (iii): which window the comparand's ``T`` is, and the comparand.
    comparand_window: str
    comparand_t: float
    predicted_bias: float
    predicted_variance_multiplier: float
    #: Eq. 32 / Eq. 13 at the VOLATILITY window -- SPEC.md 6.4.3's upper bound.
    upper_bound_bias: float
    #: ``"track"``, ``"not_track"`` or ``""`` (no registration for this cluster).
    expected: str
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def tracks(self) -> bool:
        """The comparand lies inside the point's own exact interval.

        ``(T - 1) B^2 / B_true^2 ~ chi^2_{T-1}`` (SPEC.md 6.1.1), so the true bias
        consistent with a measured ``B`` at ``T`` months is ``B / u_T <= B_true <= B / l_T``;
        the comparand tracks when it lies in that range. Equivalently, the measured
        ``B`` lies inside the null interval scaled by the comparand.
        """
        return bool(self.interval_lower <= self.predicted_bias <= self.interval_upper)

    @property
    def verdict(self) -> str:
        return tracking_verdict(self)


def tracking_verdict(point: Point) -> str:
    """``HOLDS`` / ``REFUTED`` against the cluster's registered direction, or ``reported``."""
    if point.expected == "track":
        return "HOLDS" if point.tracks else "REFUTED"
    if point.expected == "not_track":
        return "HOLDS" if not point.tracks else "REFUTED"
    return "reported"


def make_point(
    *,
    model: str,
    label: str,
    halflife: int,
    parameters_per_date: np.ndarray,
    kish_volatility_per_date: np.ndarray,
    kish_correlation_per_date: np.ndarray,
    bias_monthly: float,
    months: int,
    bias_daily: float,
    days: int,
    comparand_window: str,
    expected: str,
    level: float,
    extra: dict[str, float] | None = None,
) -> Point:
    """Assemble a point under rulings 1, 2 and (iii) from its per-date ingredients."""
    parameters = np.asarray(parameters_per_date, dtype=float)
    kish_v = np.asarray(kish_volatility_per_date, dtype=float)
    kish_c = np.asarray(kish_correlation_per_date, dtype=float)
    if not (parameters.shape == kish_v.shape == kish_c.shape) or parameters.size == 0:
        raise KtScalingError(f"make_point[{label}]: per-date ingredients disagree in length")
    if comparand_window not in ("correlation", "volatility"):
        raise KtScalingError(f"make_point[{label}]: unknown comparand window {comparand_window!r}")
    per_date = parameters / kish_v
    mean_k = float(parameters.mean())
    mean_kish_v = float(kish_v.mean())
    mean_kish_c = float(kish_c.mean())
    comparand_t = mean_kish_c if comparand_window == "correlation" else mean_kish_v
    comparand = predicted_bias(round(mean_k), comparand_t)
    bound = predicted_bias(round(mean_k), mean_kish_v)
    null = validation.chi_square_interval(months, level=level)
    return Point(
        model=model,
        label=label,
        halflife=halflife,
        parameters=mean_k,
        parameters_min=int(parameters.min()),
        parameters_max=int(parameters.max()),
        kish_volatility=mean_kish_v,
        kish_correlation=mean_kish_c,
        k_over_t=mean_k / mean_kish_v,
        k_over_t_min=float(per_date.min()),
        k_over_t_max=float(per_date.max()),
        k_over_t_asymptotic=mean_k / effective_sample_size(float(halflife)),
        bias_monthly=float(bias_monthly),
        months=int(months),
        interval_lower=float(bias_monthly) / null.upper,
        interval_upper=float(bias_monthly) / null.lower,
        bias_daily=float(bias_daily),
        days=int(days),
        comparand_window=comparand_window,
        comparand_t=comparand_t,
        predicted_bias=comparand.volatility_multiplier,
        predicted_variance_multiplier=comparand.variance_multiplier,
        upper_bound_bias=bound.volatility_multiplier,
        expected=expected,
        extra=dict(extra or {}),
    )


# ---------------------------------------------------------------------------
# The macro model: family 4 and the naive comparand at every grid point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MacroPoint:
    """One grid point of the macro sweep, both the factor model and the naive comparand."""

    halflife: int
    factor: Point
    naive: Point
    #: Family 4's daily forecast and realised series, for the sub-period split.
    dates: pd.DatetimeIndex
    forecast: np.ndarray
    realised: np.ndarray
    k_over_t_per_date: np.ndarray


def macro_points(
    data: bias_report.Inputs,
    cache: history.Cache,
    settings: config.Config,
    *,
    restrict: pd.DatetimeIndex,
    halflives: Sequence[int],
) -> tuple[MacroPoint, ...]:
    """Score the chart's two variants at every grid point off the half-life cache."""
    kt = settings.model.validation.kt_scaling
    level = float(config.resolve(settings.model, kt.interval_level_from))
    short = RiskConfig.load(horizon="short", config=settings)
    k = len(data.factor_names)
    n = len(data.assets)
    out: list[MacroPoint] = []
    for halflife in halflives:
        risk = dataclasses.replace(short, volatility_halflife=int(halflife))
        run = bias_report.run_for(
            risk,
            f"hl{halflife}",
            data,
            cache,
            settings,
            restrict=restrict,
            only=(kt.family4_variant, "sample"),
        )
        positions = data.dates.get_indexer(run.dates)
        if np.any(positions < 0):
            raise KtScalingError("macro_points: a scored date is not on the panel")
        # A macro forecast for the date at panel position p is built from rows [:p].
        kish_c = np.asarray(
            [correlation_kish(int(p), float(risk.correlation_halflife)) for p in positions]
        )
        kt_v = np.asarray(run.k_over_realised_t_eff, dtype=float)
        kish_v = k / kt_v
        factor_run = run.runs[kt.family4_variant]
        naive_run = run.runs["sample"]
        common = {
            "days": factor_run.report.observations,
            "months": factor_run.monthly.observations,
        }
        factor = make_point(
            model="macro",
            label=f"macro K={k}, tau={halflife}",
            halflife=int(halflife),
            parameters_per_date=np.full(kish_v.size, float(k)),
            kish_volatility_per_date=kish_v,
            kish_correlation_per_date=kish_c,
            bias_monthly=factor_run.monthly.median_bias(validation.OPTIMIZED),
            months=common["months"],
            bias_daily=factor_run.report.median_bias(validation.OPTIMIZED),
            days=common["days"],
            comparand_window=kt.factor_comparand_window,
            expected=kt.expected_tracking["macro"],
            level=level,
            extra={
                "family2_monthly": factor_run.monthly.median_bias(validation.RANDOM),
                "family2_daily": factor_run.report.median_bias(validation.RANDOM),
                "min_var_realised_volatility": factor_run.min_var_realised_volatility,
            },
        )
        naive = make_point(
            model="macro_naive",
            label=f"naive N={n}, tau={halflife}",
            halflife=int(halflife),
            parameters_per_date=np.full(kish_v.size, float(n)),
            kish_volatility_per_date=kish_v,
            kish_correlation_per_date=kish_c,
            bias_monthly=naive_run.monthly.median_bias(validation.OPTIMIZED),
            months=naive_run.monthly.observations,
            bias_daily=naive_run.report.median_bias(validation.OPTIMIZED),
            days=naive_run.report.observations,
            comparand_window=kt.naive_comparand_window,
            expected="",
            level=level,
            extra={
                "family2_monthly": naive_run.monthly.median_bias(validation.RANDOM),
                "min_var_realised_volatility": naive_run.min_var_realised_volatility,
            },
        )
        out.append(
            MacroPoint(
                halflife=int(halflife),
                factor=factor,
                naive=naive,
                dates=pd.DatetimeIndex(run.dates),
                forecast=factor_run.member_forecasts["min_var"].to_numpy(dtype=float),
                realised=factor_run.member_returns["min_var"].to_numpy(dtype=float),
                k_over_t_per_date=kt_v,
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# The equity model: the pre-eigenfactor family 4 across the same grid (ruling 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EquityPoint:
    halflife: int
    factor: Point
    naive: Point | None
    #: Why the naive comparand is absent, when it is.
    naive_note: str
    dates: pd.DatetimeIndex
    forecast: np.ndarray
    realised: np.ndarray
    #: Per SESSION: the realised ``K_d / T_eff`` of the forecast standing on it.
    k_over_t_per_date: np.ndarray
    scored: equity_risk.Scored


@dataclass(frozen=True)
class EquitySweep:
    points: tuple[EquityPoint, ...]
    #: Grid points with NO scored month-end of their own -- SPEC.md 5.2's repair fired on
    #: a build at or after the last one the other conditions allow -- and why.
    no_answer: dict[int, str]
    #: Per grid point: how many month-end builds the PSD repair fired on, and the last date.
    repair_firings: dict[int, int]
    last_firing: dict[int, pd.Timestamp | None]
    #: The common scored month-ends (forecast dates), and each point's own count before the
    #: intersection.
    scored_dates: pd.DatetimeIndex
    own_counts: dict[int, int]
    dropped_by_repair: dict[int, int]
    #: The grid positions (every month-end at or after the arithmetic floor).
    grid: np.ndarray
    subset_size: int

    def by_halflife(self, halflife: int) -> EquityPoint:
        for point in self.points:
            if point.halflife == halflife:
                return point
        raise KtScalingError(f"no equity grid point at tau = {halflife}")


def _first_clean(stages: Sequence[equity_risk.StageBuild]) -> int:
    fired = [i for i, s in enumerate(stages) if s.repair_fired]
    return (fired[-1] + 1) if fired else 0


def _families_report(
    forecasts: pd.DataFrame,
    realised: pd.DataFrame,
    portfolios: validation.PortfolioSet,
    *,
    clip: float,
) -> validation.ValidationReport:
    """SPEC.md 6.1 on families 2 and 4 only -- what the chart reads."""
    families = {name: members for name, members in portfolios.families.items() if name in _FAMILIES}
    kept = validation.PortfolioSet(families=families, rebalance=portfolios.rebalance)
    members = [m for names in families.values() for m in names]
    return validation.validate(forecasts[members], realised[members], kept, clip=clip)


def equity_sweep(
    data: equity_risk.Panel,
    settings: config.Config,
    *,
    halflives: Sequence[int],
    progress: bool = True,
) -> EquitySweep:
    """Ruling 3: family 4 pre-eigenfactor at every grid point, everything else at the short values.

    Stages 1-3 only (no Monte Carlo), month-end grid, SPEC.md 5.5's specific
    risk at the short horizon's half-life computed ONCE because it does not read
    the factor-volatility half-life. The scored window is the intersection across
    grid points of each point's own window under the equity report's three
    conditions (after the last PSD repair, ``T > N_subset`` for the naive
    comparand, one earlier forecast, not the last month-end).
    """
    kt = settings.model.validation.kt_scaling
    battery = settings.model.validation
    level = float(config.resolve(settings.model, kt.interval_level_from))
    short = RiskConfig.load(horizon=kt.equity_sweep_horizon, config=settings)  # type: ignore[arg-type]
    spec = next(s for s in bias_report.variants(settings) if s.name == kt.family4_variant)
    sample_spec = next(s for s in bias_report.variants(settings) if s.name == "sample")
    ends = equity_risk.month_end_positions(data.dates)
    floor = bias_report.minimum_observations(short, factors=data.factors.factors)
    grid = np.asarray([p for p in ends if p + 1 >= floor and p < len(data.dates) - 1], dtype=int)
    buckets = settings.model.equity_risk.size_buckets
    structural = settings.model.equity_risk.singleton_specific_risk == "structural"
    residuals = data.residuals.to_numpy(dtype=float)
    designs = {int(p): equity_risk.design_at(data, int(p), buckets=buckets) for p in grid}
    specifics = {
        p: equity_risk.specific_at(short, residuals, d, singleton_structural=structural)
        for p, d in designs.items()
    }
    n_subset = len(data.always_present)

    builds: dict[int, list[equity_risk.StageBuild]] = {}
    windows: dict[int, np.ndarray] = {}
    own_counts: dict[int, int] = {}
    dropped: dict[int, int] = {}
    no_answer: dict[int, str] = {}
    firings: dict[int, int] = {}
    last_firing: dict[int, pd.Timestamp | None] = {}
    for halflife in halflives:
        risk = dataclasses.replace(short, volatility_halflife=int(halflife))
        stages = equity_risk.stage_builds(risk, data, [int(p) for p in grid])
        clean = _first_clean(stages)
        fired = [s for s in stages if s.repair_fired]
        firings[int(halflife)] = len(fired)
        last_firing[int(halflife)] = fired[-1].date if fired else None
        tail = stages[clean:]
        positions = np.asarray([s.position for s in tail], dtype=int)
        rank_ok = positions + 1 > n_subset
        regime_ok = np.arange(positions.size) >= 1
        last_ok = np.arange(positions.size) < positions.size - 1
        keep = rank_ok & regime_ok & last_ok
        builds[int(halflife)] = tail
        windows[int(halflife)] = positions[keep]
        own_counts[int(halflife)] = int(keep.sum())
        dropped[int(halflife)] = clean
        if int(keep.sum()) == 0:
            # SPEC.md 5.2.2 / 6.2.4: a repaired matrix is not a covariance matrix and the
            # estimator has no answer on it; a point whose repair fires through the sample
            # has no scored window and is REPORTED as such rather than scored on a
            # manufactured one.
            no_answer[int(halflife)] = (
                f"the PSD repair fired on {len(fired)} of {len(stages)} month-end builds, the "
                f"last on {fired[-1].date.date() if fired else 'n/a'} at realised K_d/T_eff "
                f"{fired[-1].k_over_t:.3f}; no month-end survives the scored-window rule"
                if fired
                else "no month-end survives the scored-window rule"
            )
        if progress:
            print(
                f"  equity tau {halflife}: {keep.sum()} scored month-ends, repair fired on "
                f"{len(fired)} of {len(stages)} builds",
                flush=True,
            )
    answered = [int(h) for h in halflives if int(h) not in no_answer]
    if not answered:
        raise KtScalingError("equity_sweep: no grid point has a scored month-end")
    common: set[int] | None = None
    for halflife in answered:
        chosen = set(windows[halflife].tolist())
        common = chosen if common is None else common & chosen
    if not common:
        raise KtScalingError("equity_sweep: the answered grid points share no scored month-end")
    scored_positions = np.asarray(sorted(common), dtype=int)
    scored_dates = pd.DatetimeIndex(data.dates[scored_positions])

    points: list[EquityPoint] = []
    for halflife in answered:
        tail = builds[int(halflife)]
        position_list = [s.position for s in tail]
        index_of = {p: i for i, p in enumerate(position_list)}
        scored_idx = [index_of[int(p)] for p in scored_positions]
        design_list = [designs[p] for p in position_list]
        specific_list = [specifics[p] for p in position_list]
        stage_list = [s.matrices for s in tail]
        empty_eigen: list[dict[str, np.ndarray]] = [{} for _ in position_list]
        lambda_s = np.full(len(position_list), np.nan)
        scored = equity_risk.score_variant(
            spec,
            data=data,
            positions=position_list,
            scored=scored_idx,
            designs=design_list,
            stages=stage_list,
            eigen=empty_eigen,
            specifics=specific_list,
            lambda_f={},
            lambda_s=lambda_s,
            subset=data.always_present,
            random_portfolios=battery.random_portfolios,
            seed=settings.model.seed,
            volatility_halflife=float(halflife),
        )
        daily = _families_report(
            scored.forecasts,
            scored.realised,
            scored.portfolios,
            clip=battery.standardized_return_clip,
        )
        monthly_f, monthly_r = equity_risk.to_monthly(scored)
        monthly = _families_report(
            monthly_f, monthly_r, scored.portfolios, clip=battery.standardized_return_clip
        )
        split = equity_risk.component_split(
            data=data,
            positions=position_list,
            scored=scored_idx,
            designs=design_list,
            weights=scored.min_var_weights,
            factor=[m[spec.stage] for m in stage_list],
            specific_variance=[s.variance for s in specific_list],
        )
        components = {
            name: diagnostics.component_bias(*split.component(name), name=name, level=level)
            for name in ("factor", "specific")
        }
        # Per scored build: K_d, the volatility-window Kish size and the correlation-window one.
        k_d = np.asarray([tail[i].factors for i in scored_idx], dtype=float)
        kt_v = np.asarray([tail[i].k_over_t for i in scored_idx], dtype=float)
        kish_v = k_d / kt_v
        # An equity forecast at panel position p is built from rows [: p + 1].
        kish_c = np.asarray(
            [
                correlation_kish(int(tail[i].observations), float(short.correlation_halflife))
                for i in scored_idx
            ]
        )
        factor = make_point(
            model="equity",
            label=f"equity K_d={int(k_d.min())}-{int(k_d.max())}, tau={halflife}",
            halflife=int(halflife),
            parameters_per_date=k_d,
            kish_volatility_per_date=kish_v,
            kish_correlation_per_date=kish_c,
            bias_monthly=monthly.median_bias(validation.OPTIMIZED),
            months=monthly.observations,
            bias_daily=daily.median_bias(validation.OPTIMIZED),
            days=daily.observations,
            comparand_window=kt.factor_comparand_window,
            expected=kt.expected_tracking["equity"],
            level=level,
            extra={
                "family2_monthly": monthly.median_bias(validation.RANDOM),
                "family2_daily": daily.median_bias(validation.RANDOM),
                "b_factor": components["factor"].bias,
                "b_factor_lower": components["factor"].lower,
                "b_factor_upper": components["factor"].upper,
                "b_specific": components["specific"].bias,
                "b_specific_lower": components["specific"].lower,
                "b_specific_upper": components["specific"].upper,
                "min_var_realised_volatility": float(
                    np.std(scored.realised["min_var"].to_numpy(dtype=float), ddof=1)
                ),
                "liquidated_cells": float(scored.liquidated_cells),
            },
        )
        naive: Point | None = None
        note = ""
        try:
            naive_scored = equity_risk.score_variant(
                sample_spec,
                data=data,
                positions=position_list,
                scored=scored_idx,
                designs=design_list,
                stages=stage_list,
                eigen=empty_eigen,
                specifics=specific_list,
                lambda_f={},
                lambda_s=lambda_s,
                subset=data.always_present,
                random_portfolios=battery.random_portfolios,
                seed=settings.model.seed,
                volatility_halflife=float(halflife),
            )
        except (validation.ValidationError, np.linalg.LinAlgError) as exc:
            note = f"the naive comparand has no answer at tau = {halflife}: {exc}"
        else:
            naive_daily = _families_report(
                naive_scored.forecasts,
                naive_scored.realised,
                naive_scored.portfolios,
                clip=battery.standardized_return_clip,
            )
            nf, nr = equity_risk.to_monthly(naive_scored)
            naive_monthly = _families_report(
                nf, nr, naive_scored.portfolios, clip=battery.standardized_return_clip
            )
            naive = make_point(
                model="equity_naive",
                label=f"naive N={n_subset}, tau={halflife}",
                halflife=int(halflife),
                parameters_per_date=np.full(kish_v.size, float(n_subset)),
                kish_volatility_per_date=kish_v,
                kish_correlation_per_date=kish_c,
                bias_monthly=naive_monthly.median_bias(validation.OPTIMIZED),
                months=naive_monthly.observations,
                bias_daily=naive_daily.median_bias(validation.OPTIMIZED),
                days=naive_daily.observations,
                comparand_window=kt.naive_comparand_window,
                expected="",
                level=level,
                extra={
                    "family2_monthly": naive_monthly.median_bias(validation.RANDOM),
                    "min_var_realised_volatility": float(
                        np.std(naive_scored.realised["min_var"].to_numpy(dtype=float), ddof=1)
                    ),
                },
            )
        # month_of_session holds the index into `positions` of the forecast standing
        # on each session; map it to the scored-build order.
        order = {idx: j for j, idx in enumerate(scored_idx)}
        per_session = np.asarray([kt_v[order[int(m)]] for m in scored.month_of_session])
        points.append(
            EquityPoint(
                halflife=int(halflife),
                factor=factor,
                naive=naive,
                naive_note=note,
                dates=pd.DatetimeIndex(scored.forecasts.index),
                forecast=scored.forecasts["min_var"].to_numpy(dtype=float),
                realised=scored.realised["min_var"].to_numpy(dtype=float),
                k_over_t_per_date=per_session,
                scored=scored,
            )
        )
        if progress:
            print(
                f"  equity tau {halflife}: family 4 B {factor.bias_daily:.4f} daily / "
                f"{factor.bias_monthly:.4f} monthly, K/T {factor.k_over_t:.4f}",
                flush=True,
            )
    return EquitySweep(
        points=tuple(points),
        no_answer=no_answer,
        repair_firings=firings,
        last_firing=last_firing,
        scored_dates=scored_dates,
        own_counts=own_counts,
        dropped_by_repair=dropped,
        grid=grid,
        subset_size=n_subset,
    )


# ---------------------------------------------------------------------------
# Rows 327-333: the SIMULATED second-order comparand (W8-P1b)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulatedComparand:
    """The split-window estimator's own second-order ``B`` at one volatility half-life.

    **Why this exists.** Shepard's closed form is derived for a covariance
    estimated on ONE window and SPEC.md 5.1's estimator uses two, so neither
    single-``T`` evaluation of it is the estimator's own theory: at the
    correlation window it under-predicts the short-window equity points, at the
    volatility window it over-predicts by 4.1x down to 0.94x along the grid
    (SPEC.md 6.4.5). :func:`mafrm.risk.second_order.simulate` IS the estimator's
    second-order theory, computed on the estimator rather than on a formula.

    **The mapping to a ``B`` is the closed form's own.** Shepard turns a
    variance multiplier into a bias statistic by one identity,
    ``B = sqrt(realised variance / forecast variance)``; ``simulate`` computes
    exactly that ratio for the minimum-variance ``w`` of each drawn ``F-hat``
    and :attr:`~mafrm.risk.second_order.ConditionResult.bias` is its square
    root. Nothing new is invented here.

    **What it is not.** The simulated book is minimum-variance in FACTOR space
    while the chart's ``B`` is the asset-space book of ``Sigma = X F X' +
    Delta`` -- the same relation Eq. 32 has to the book its multiplier is
    applied to. And the truth is Gaussian with the panel's own ``F``, so this is
    an estimation-error figure containing no specific-risk error, no fat tails
    and no specification error.
    """

    halflife: int
    seed: int
    result: second_order.Decomposition

    @property
    def predicted_bias(self) -> float:
        """``B`` under the estimator's own second-order theory."""
        return self.result.joint.bias

    @property
    def standard_error(self) -> float:
        """Monte Carlo s.e. of :attr:`predicted_bias`, by the delta method on the square root."""
        return self.result.joint.standard_error / (2.0 * self.predicted_bias)

    @property
    def correlation_leg(self) -> float:
        return self.result.correlation_only.excess

    @property
    def volatility_leg(self) -> float:
        return self.result.volatility_only.excess


def equity_second_order(
    data: equity_risk.Panel,
    settings: config.Config,
    *,
    halflives: Sequence[int],
    progress: bool = True,
) -> tuple[SimulatedComparand, ...]:
    """Rows 327-333: ``simulate`` on the equity panel's own ``F`` at each answered half-life.

    ONE truth -- the panel's separated-EWMA ``F`` at the shipped short-horizon
    half-lives -- so that the only thing moving along the grid is the
    estimator's volatility window, exactly as rows 318-326 are built on the
    macro panel. ``tau_rho`` is held at the shipped correlation half-life
    throughout, and the seed varies across grid points because two runs sharing
    one seed are one measurement (rows 177/183).
    """
    second = settings.model.validation.second_order
    trials = int(config.resolve(settings.model, second.monte_carlo_trials_from))
    risk = RiskConfig.load(
        horizon=settings.model.validation.kt_scaling.equity_sweep_horizon,  # type: ignore[arg-type]
        config=settings,
    )
    frame = data.factors.frame.to_numpy(dtype=float)
    truth = separated_ewma_covariance(
        frame,
        volatility_halflife=float(risk.volatility_halflife),
        correlation_halflife=float(risk.correlation_halflife),
    )
    # CLAUDE.md invariant 4 at the one matrix this construction builds.
    assert_psd(truth, "kt_scaling.equity_second_order.truth", config=settings)
    out: list[SimulatedComparand] = []
    for index, halflife in enumerate(halflives):
        seed = int(settings.model.seed) + index
        result = second_order.simulate(
            truth,
            volatility_halflife=float(halflife),
            correlation_halflife=float(risk.correlation_halflife),
            observations=int(frame.shape[0]),
            trials=trials,
            seed=seed,
        )
        point = SimulatedComparand(halflife=int(halflife), seed=seed, result=result)
        out.append(point)
        if progress:
            print(
                f"  simulated tau {halflife}: B {point.predicted_bias:.4f} "
                f"+- {point.standard_error:.4f}, joint {100 * result.joint.excess:.2f}% "
                f"of variance",
                flush=True,
            )
    return tuple(out)


# ---------------------------------------------------------------------------
# Row 164: control C1 on the equity residual panel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EquityControl:
    """Control C1 (SPEC.md 6.2.5, rows 162-163) re-run on the equity panel: row 164.

    The same construction as :class:`mafrm.factors.bias_report.DiagonalControl`:
    ``diag(delta^2)`` becomes ``D_delta R_u D_delta`` with ``R_u`` the EWMA
    residual correlation at the specific-risk half-life and ``D_delta`` SPEC.md
    5.5's own shrunk deviations, unchanged. On the always-present subset
    (``row_164_residual_subset``), because a residual correlation needs a
    complete history and the naive comparand already lives there.
    """

    base_variant: str
    months: int
    days: int
    diagonal_family_4: float
    correlated_family_4: float
    diagonal_family_2: float
    correlated_family_2: float
    eigen_family_4: float
    naive_family_4: float
    half_width: float
    #: Realised min-var volatility, bps/day: diagonal, correlated.
    realised: tuple[float, float]
    #: Smallest eigenvalue of ``R_u`` across the scored builds -- how singular the control is.
    min_residual_eigenvalue: float
    #: Always-present names with a complete residual history (the control's subset), and the
    #: always-present count they were drawn from.
    subset_size: int
    always_present: int

    @property
    def excess(self) -> float:
        return self.diagonal_family_4 - 1.0

    @property
    def c1_share_of_excess(self) -> float:
        """The share of ``B_diag - 1`` that restoring the residual correlations removes."""
        return (self.diagonal_family_4 - self.correlated_family_4) / self.excess

    @property
    def eigen_share_of_excess(self) -> float:
        """The share of ``B_diag - 1`` that SPEC.md 5.3's eigenfactor stage removes."""
        return (self.diagonal_family_4 - self.eigen_family_4) / self.excess

    @property
    def half_width_share(self) -> float:
        return self.half_width / self.excess


def _subset_books(
    covariances: Sequence[np.ndarray],
    *,
    data: equity_risk.Panel,
    held: Sequence[tuple[int, int]],
    columns: np.ndarray,
    random_portfolios: int,
    seed: int,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, validation.PortfolioSet, np.ndarray]:
    """Families 2 and 4 on one covariance per scored month-end, held over ``held[j]``'s sessions.

    ``held[j] = (start, stop)`` are panel rows, half-open, of the month the
    ``j``-th forecast is held through; a missing return inside it is zero
    (SPEC.md 15.7.1 construction 3). Returns the session-level forecasts and
    realisations, the two families, and the month index of every session.
    """
    if len(covariances) != len(held):
        raise KtScalingError("_subset_books: one covariance per held month")
    all_returns = data.returns.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    random_names = [f"random_{i:03d}" for i in range(random_portfolios)]
    members = [*random_names, "min_var"]
    rows_f: list[np.ndarray] = []
    rows_r: list[np.ndarray] = []
    dates: list[pd.Timestamp] = []
    month_of: list[int] = []
    for j, (covariance, (start, stop)) in enumerate(zip(covariances, held, strict=True)):
        forecast = validation.RiskForecast(covariance=covariance)
        validation.assert_forecast_psd(forecast, f"{label} {data.dates[start - 1]!s}")
        draw = validation.random_dollar_neutral(random_portfolios, columns.size, rng)
        min_var = validation.minimum_variance_weights(covariance)
        weights = np.vstack([draw, min_var[None, :]])
        sigma = np.sqrt(np.einsum("pi,ij,pj->p", weights, covariance, weights))
        raw = all_returns[start:stop][:, columns]
        returns = np.where(np.isfinite(raw), raw, 0.0) @ weights.T
        for offset in range(stop - start):
            rows_f.append(sigma)
            rows_r.append(returns[offset])
            dates.append(pd.Timestamp(data.dates[start + offset]))
            month_of.append(j)
    index = pd.DatetimeIndex(dates)
    portfolios = validation.PortfolioSet(
        families={validation.RANDOM: tuple(random_names), validation.OPTIMIZED: ("min_var",)},
        rebalance="held between month-ends",
    )
    return (
        pd.DataFrame(np.asarray(rows_f), index=index, columns=members),
        pd.DataFrame(np.asarray(rows_r), index=index, columns=members),
        portfolios,
        np.asarray(month_of, dtype=int),
    )


def _monthly(
    forecasts: pd.DataFrame, realised: pd.DataFrame, months: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``R_m = sum_t R_t``, ``sigma_m = sigma sqrt(days)`` per held month -- ``to_monthly``'s "
    "rule."""
    keys = np.unique(months)
    f = forecasts.to_numpy(dtype=float)
    r = realised.to_numpy(dtype=float)
    rows_f, rows_r, dates = [], [], []
    for key in keys:
        mask = months == key
        rows_f.append(f[mask][0] * np.sqrt(int(mask.sum())))
        rows_r.append(r[mask].sum(axis=0))
        dates.append(pd.Timestamp(forecasts.index[np.flatnonzero(mask)[-1]]))
    index = pd.DatetimeIndex(dates)
    return (
        pd.DataFrame(np.asarray(rows_f), index=index, columns=forecasts.columns),
        pd.DataFrame(np.asarray(rows_r), index=index, columns=realised.columns),
    )


def held_months(grid: np.ndarray, scored_positions: Sequence[int]) -> list[tuple[int, int]]:
    """``(start, stop)`` panel rows of the month held after each scored month-end.

    The held month runs from the session after the forecast's close to the next
    month-end on the grid inclusive -- :func:`equity_risk.score_variant`'s own
    rule, restated on the month-end grid.
    """
    out: list[tuple[int, int]] = []
    for p in scored_positions:
        where = int(np.searchsorted(grid, p))
        if where >= grid.size or grid[where] != p or where + 1 >= grid.size:
            raise KtScalingError(
                f"held_months: position {p} has no following month-end on the grid"
            )
        out.append((int(p) + 1, int(grid[where + 1]) + 1))
    return out


def equity_control(
    data: equity_risk.Panel,
    settings: config.Config,
    sweep: EquitySweep,
    *,
    eigen: Sequence[np.ndarray],
    eigen_positions: Sequence[int],
    progress: bool = True,
) -> EquityControl:
    """Row 164: C1 on the equity panel at the shipped short half-life, on the always-present subset.

    ``eigen[i]`` is the eigenfactor-adjusted factor covariance (``a = 1.0``)
    built at month-end ``eigen_positions[i]``, from the equity report's
    committed partial, so the share SPEC.md 5.3 closes is measured on the same
    books and the same subset as C1's share. The diagonal base is
    ``family4_variant`` (pre-eigenfactor), the equity chart's own variant.
    """
    kt = settings.model.validation.kt_scaling
    battery = settings.model.validation
    level = float(config.resolve(settings.model, kt.interval_level_from))
    short = RiskConfig.load(horizon=kt.equity_sweep_horizon, config=settings)  # type: ignore[arg-type]
    point = sweep.by_halflife(short.volatility_halflife)
    # A residual correlation needs a complete history. An always-present name has a RETURN on
    # every regression date; it lacks a residual on the dates it was outside the regression set
    # (no screened cap, no exposure -- SPEC.md 15.7.1 construction 2), so the control's subset is
    # the always-present names with a residual on every regression date, and the count is stated.
    name_index = {name: i for i, name in enumerate(data.names)}
    complete = data.residuals.notna().all(axis=0)
    subset = [name for name in data.always_present if bool(complete[name])]
    if len(subset) < 2:
        raise KtScalingError(
            f"equity_control: {len(subset)} always-present name(s) carry a complete residual "
            "history"
        )
    subset_set = set(subset)
    subset_columns = np.asarray([name_index[name] for name in subset], dtype=int)
    residuals = data.residuals.to_numpy(dtype=float)[:, subset_columns]
    if not np.all(np.isfinite(residuals)):  # pragma: no cover - the filter above guarantees it
        raise KtScalingError("equity_control: the residual subset has a missing residual")
    full_residuals = data.residuals.to_numpy(dtype=float)
    eigen_at = {int(p): m for p, m in zip(eigen_positions, eigen, strict=True)}
    buckets = settings.model.equity_risk.size_buckets
    structural = settings.model.equity_risk.singleton_specific_risk == "structural"
    scored_positions = [int(p) for p in data.dates.get_indexer(sweep.scored_dates)]
    if any(p < 0 for p in scored_positions):
        raise KtScalingError("equity_control: a scored date is not on the panel")
    missing = [p for p in scored_positions if p not in eigen_at]
    if missing:
        raise KtScalingError(
            f"equity_control: no eigenfactor matrix for {len(missing)} scored month-end(s); the "
            "equity partial and the sweep's window disagree"
        )
    held = held_months(sweep.grid, scored_positions)
    stage_at = {
        int(s.position): s.matrices for s in equity_risk.stage_builds(short, data, scored_positions)
    }
    diagonal_cov: list[np.ndarray] = []
    correlated_cov: list[np.ndarray] = []
    eigen_cov: list[np.ndarray] = []
    min_eig = np.inf
    columns_used: np.ndarray | None = None
    for count, p in enumerate(scored_positions, start=1):
        design = equity_risk.design_at(data, p, buckets=buckets)
        spec_at = equity_risk.specific_at(
            short, full_residuals, design, singleton_structural=structural
        )
        in_subset = np.asarray([name in subset_set for name in design.names], dtype=bool)
        order = np.asarray([subset.index(name) for name in design.names if name in subset_set])
        x = design.exposures[in_subset]
        deviation = spec_at.deviation[in_subset]
        common = x @ stage_at[p][kt.family4_variant] @ x.T
        moment = ewma_second_moment(
            residuals[: p + 1][:, order], halflife=float(short.specific_volatility_halflife)
        )
        r_u = correlation_from_covariance(moment)
        min_eig = min(min_eig, float(np.linalg.eigvalsh(r_u).min()))
        diagonal = common + np.diag(np.square(deviation))
        correlated = common + r_u * np.outer(deviation, deviation)
        with_eigen = x @ eigen_at[p] @ x.T + np.diag(np.square(deviation))
        diagonal_cov.append(0.5 * (diagonal + diagonal.T))
        correlated_cov.append(0.5 * (correlated + correlated.T))
        eigen_cov.append(0.5 * (with_eigen + with_eigen.T))
        cols = design.columns[in_subset]
        if columns_used is None:
            columns_used = cols
        elif not np.array_equal(cols, columns_used):
            raise KtScalingError("equity_control: the subset's column order moved between dates")
        if progress and count % 50 == 0:
            print(f"  C1 equity: {count} / {len(scored_positions)} month-ends", flush=True)
    assert columns_used is not None
    reports: dict[str, tuple[validation.ValidationReport, validation.ValidationReport, float]] = {}
    for name, covs in (
        ("diagonal", diagonal_cov),
        ("correlated", correlated_cov),
        ("eigen", eigen_cov),
    ):
        f, r, portfolios, month_of = _subset_books(
            covs,
            data=data,
            held=held,
            columns=columns_used,
            random_portfolios=battery.random_portfolios,
            seed=settings.model.seed,
            label=f"C1 equity {name}",
        )
        daily = validation.validate(f, r, portfolios, clip=battery.standardized_return_clip)
        mf, mr = _monthly(f, r, month_of)
        monthly = validation.validate(mf, mr, portfolios, clip=battery.standardized_return_clip)
        reports[name] = (daily, monthly, float(np.std(r["min_var"].to_numpy(), ddof=1)))
    naive = point.naive.bias_monthly if point.naive is not None else float("nan")
    months = reports["diagonal"][1].observations
    return EquityControl(
        base_variant=kt.family4_variant,
        months=months,
        days=reports["diagonal"][0].observations,
        diagonal_family_4=reports["diagonal"][1].median_bias(validation.OPTIMIZED),
        correlated_family_4=reports["correlated"][1].median_bias(validation.OPTIMIZED),
        diagonal_family_2=reports["diagonal"][1].median_bias(validation.RANDOM),
        correlated_family_2=reports["correlated"][1].median_bias(validation.RANDOM),
        eigen_family_4=reports["eigen"][1].median_bias(validation.OPTIMIZED),
        naive_family_4=naive,
        half_width=validation.chi_square_interval(months, level=level).upper - 1.0,
        realised=(reports["diagonal"][2], reports["correlated"][2]),
        min_residual_eigenvalue=min_eig,
        subset_size=len(subset),
        always_present=len(data.always_present),
    )


# ---------------------------------------------------------------------------
# Row 164's second falsifier: family 4 by NON-overlapping sub-period against K/T_eff
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubperiodSplit:
    """Family 4's ``B`` on consecutive non-overlapping blocks, beside the block's ``K/T_eff``."""

    model: str
    block_months: int
    table: pd.DataFrame
    #: Spearman rank correlation between the block ``B`` and the block ``K/T_eff``.
    rank_correlation: float
    blocks: int

    @property
    def overlap(self) -> int:
        """Zero, by construction -- stated because CLAUDE.md failure mode 9 requires it."""
        return 0


def subperiod_split(
    *,
    model: str,
    dates: pd.DatetimeIndex,
    forecast: np.ndarray,
    realised: np.ndarray,
    k_over_t: np.ndarray,
    block_months: int,
    clip: float,
    level: float,
) -> SubperiodSplit:
    """Non-overlapping blocks of ``block_months`` calendar months from the first scored date."""
    if not (len(dates) == forecast.size == realised.size == k_over_t.size):
        raise KtScalingError("subperiod_split: the series disagree in length")
    if block_months < 1:
        raise KtScalingError("subperiod_split: block_months must be positive")
    periods = pd.DatetimeIndex(dates).to_period("M")
    months_since = (periods.year - periods[0].year) * 12 + (periods.month - periods[0].month)
    block = np.asarray(months_since // block_months, dtype=int)
    standardized = validation.standardized_returns(realised[:, None], forecast[:, None])[:, 0]
    rows = []
    for key in np.unique(block):
        mask = block == key
        if mask.sum() < 2:
            continue
        b = validation.bias_statistic(standardized[mask][:, None], clip=clip)[0]
        interval = validation.chi_square_interval(int(mask.sum()), level=level)
        rows.append(
            {
                "block": int(key),
                "start": pd.Timestamp(dates[mask][0]),
                "end": pd.Timestamp(dates[mask][-1]),
                "T": int(mask.sum()),
                "B": float(b),
                "lower": interval.lower,
                "upper": interval.upper,
                "k_over_t_mean": float(k_over_t[mask].mean()),
                "k_over_t_min": float(k_over_t[mask].min()),
                "k_over_t_max": float(k_over_t[mask].max()),
            }
        )
    table = pd.DataFrame(rows).set_index("block")
    if len(table) >= 3:
        rank_b = table["B"].rank().to_numpy(dtype=float)
        rank_kt = table["k_over_t_mean"].rank().to_numpy(dtype=float)
        corr = float(np.corrcoef(rank_b, rank_kt)[0, 1])
    else:
        corr = float("nan")
    return SubperiodSplit(
        model=model,
        block_months=block_months,
        table=table,
        rank_correlation=corr,
        blocks=len(table),
    )
