"""SPEC.md 9's experiment grid: seven covariance treatments by four cost treatments. W6-P2.

Twenty-eight cells, Model A only, every cell a ``strategy-config`` trial, under
the eight operator rulings of 2026-09-04 recorded at SPEC.md 9.1 and in
``config/model.yaml`` under ``optimizer.grid``. Nothing else is crossed: every
other dial sits at the W6-P1 verification point and is a W6-P3 band.

WHAT A CELL IS
    A :class:`CellSpec` names a covariance variant and a cost treatment (and,
    for treatment D, the ``Y`` regime -- both regimes run inside the one cell as
    a band). :func:`cell_inputs` turns it into a :class:`CellInputs`: the
    forecasts at every rebalance in the optimizer's units, the alpha, the cost
    model the OPTIMIZER prices and the cost model the ENGINE charges, and the
    NAV. :func:`run_cell` consumes a :class:`CellInputs` and nothing else, so a
    W6-P3 band is the same runner with one field changed, and a synthetic
    :class:`CellInputs` drives the golden-weights fixture offline.

THE TWO AXES
    Variants 2-6 read SPEC.md 5's stages off the committed forecast panel
    (:func:`mafrm.factors.bias_report.forecast_panel`), exactly the matrices
    W4-P2's families were scored on. Variants 1 and 7 are DENSE asset-level
    estimators on the ``T_eff``-day window the parser derives from the EWMA
    half-life (SPEC.md 9.1 rulings 2 and 3): the equal-weighted second moment
    about zero, and constant-correlation Ledoit-Wolf on the same rows. A dense
    forecast enters the optimizer as ``X = I, F = Sigma, Delta = 0``, so the same
    objective prices it and SPEC.md 8.3's decomposition puts all of alpha in the
    span -- a full-rank model sees every direction.

    Treatments differ in what the optimizer SEES; what the engine CHARGES is
    the true SPEC.md 7.1 model in every cell but A. The identity's ``TC`` is
    therefore the realised cost of the book each treatment produces.

SPEC.md 8.4's SECOND STEP (ruling 1)
    In ``tracking_error_constraint`` mode the ``gamma_risk`` term leaves the
    objective and ``||M w||_2 <= TE_target`` is imposed hard; the multiplier is
    recovered and the ``gamma_risk`` it implies, ``mu / (2 TE_target)``, is
    reported beside SPEC.md 8.4's initialisation. SPEC.md 8.3's penalty is
    priced at the recovered multiplier in a second pass (the CONSTRUCTION in the
    config). The ADV cap is a hinge at the p80 multiplier W6-P1 observed.

HOLDOUT
    CLAUDE.md invariant 5 is enforced HERE, in code: :func:`assert_in_sample`
    refuses any rebalance date, and the close of the last holding period, on or
    after ``sample.holdout_start``. It runs before the first solve of every cell.

W6-P3 (operator rulings of 2026-09-04, SPEC.md 9.3)
    :class:`CellInputs` carries OVERRIDES -- ``gamma_trade``, ``risk_aversion``,
    ``position_box``, a hard ADV cap and its ladder -- so a band is this runner
    with one field changed (:mod:`mafrm.backtest.bands`), and the capacity
    re-optimisation is the same runner at another NAV
    (:mod:`mafrm.backtest.capacity_curve`). Every rebalance the primary solver
    returned ``optimal_inaccurate`` on is RE-SOLVED with the fallback solver
    and the L1 distance recorded; the cell's MAX distance against
    ``grid.resolve.l1_threshold`` decides whether it re-runs as new rows
    (ruling 6). The financing leg (``backtest.financing``, ruling 1) is
    reported per cell in nominal bp/yr: the price panel is the excess-return
    index, so the engine carries cash at zero by identity and
    :func:`mafrm.backtest.engine.financing_leg` prices the balance at RF for
    the report. SPEC.md 10.1-10.3's diagnostics
    (:mod:`mafrm.backtest.diagnostics`) run on every cell after the engine.

Everything else -- the incremental engine re-run for ``NAV_pre``, the monthly
scoring, the bias statistic -- follows :mod:`mafrm.backtest.verification`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mafrm import config as config_mod
from mafrm.backtest import alpha as alpha_mod
from mafrm.backtest import constraints as con
from mafrm.backtest import diagnostics as diagnostics_mod
from mafrm.backtest import engine, verification
from mafrm.backtest import metrics as metrics_mod
from mafrm.backtest import optimizer as opt
from mafrm.factors import bias_report, statistical
from mafrm.risk import validation

__all__ = [
    "CellInputs",
    "CellResult",
    "CellSpec",
    "ChartData",
    "Forecast",
    "GridError",
    "GridResult",
    "HoldoutError",
    "assert_in_sample",
    "cell_inputs",
    "cells",
    "dense_covariance",
    "golden_payload",
    "main",
    "monthly_returns",
    "render",
    "run_cell",
    "run_grid",
    "synthetic_inputs",
    "write_metrics",
]

FloatArray = NDArray[np.float64]

_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_REPORTS: Final[Path] = _ROOT / "reports"
_RESULTS: Final[Path] = _ROOT / "results"
_MARKDOWN_PATH: Final[Path] = _REPORTS / "experiment_grid.md"
_DIAGNOSTICS_PATH: Final[Path] = _REPORTS / "diagnostics.md"
_DIAGNOSTICS_CSV_PATH: Final[Path] = _REPORTS / "diagnostics.csv"
_CSV_PATH: Final[Path] = _REPORTS / "experiment_grid.csv"
_CHART_PATH: Final[Path] = _REPORTS / "experiment_grid.png"
_METRICS_PATH: Final[Path] = _RESULTS / "metrics.json"
_EXPERIMENTS_PATH: Final[Path] = _ROOT / "experiments.md"

_TREATMENT_LETTER: Final[Mapping[str, str]] = {
    "none": "A",
    "gross_then_net": "B",
    "flat_spread": "C",
    "time_varying": "D",
}
_VARIANT_NUMBER: Final[Mapping[str, int]] = {
    "sample_equal_weight": 1,
    "ewma": 2,
    "newey_west": 3,
    "eigenfactor_a1.0": 4,
    "eigenfactor_a1.4": 5,
    "volatility_regime": 6,
    "ledoit_wolf": 7,
}
_DENSE: Final[frozenset[str]] = frozenset({"sample_equal_weight", "ledoit_wolf"})
#: Grid variant -> the bias report's ladder name (SPEC.md 5's stage it reads).
_PANEL_SPEC: Final[Mapping[str, str]] = {
    "ewma": "ewma",
    "newey_west": "psd_repair",
    "eigenfactor_a1.0": "eigen_a1",
    "eigenfactor_a1.4": "eigen_a1.4",
    "volatility_regime": "specified_a1",
}
_TE_LIMIT: Final[str] = "tracking_error"
_ADV_LIMIT: Final[str] = "adv_participation"
_BOX_LIMIT: Final[str] = "position_box"
_INACCURATE: Final[str] = "optimal_inaccurate"
#: Re-solve verdicts under the AMENDED criterion (SPEC.md 9.3 ruling 6, W6-P3b).
_VERDICT_FALLBACK_INFEASIBLE: Final[str] = "fallback infeasible"
_VERDICT_PRIMARY_BETTER: Final[str] = "primary better"
_VERDICT_FALLBACK_BETTER: Final[str] = "fallback better (feasible)"
#: Sections other modules write into ``results/metrics.json``; ``main`` preserves them.
_PRESERVED_SECTIONS: Final[tuple[str, ...]] = ("bands", "capacity_reoptimised")
#: The share of rebalances read as the "worst 5% of volatility" windows (SPEC.md 1, H5).
_H5_TAIL: Final[float] = 0.05


class GridError(ValueError):
    """The grid is mis-specified. Nothing is filled or defaulted."""


class HoldoutError(GridError):
    """CLAUDE.md invariant 5: something on or after ``sample.holdout_start`` was asked for."""


# ---------------------------------------------------------------------------
# The holdout guard
# ---------------------------------------------------------------------------


def assert_in_sample(
    rebalance_dates: pd.DatetimeIndex, terminal_date: pd.Timestamp, holdout_start: pd.Timestamp
) -> None:
    """Refuse any rebalance, and the close of the last holding period, at or past the boundary."""
    boundary = pd.Timestamp(holdout_start)
    if len(rebalance_dates) == 0:
        raise GridError("assert_in_sample: no rebalance dates")
    last = pd.Timestamp(rebalance_dates.max())
    if last >= boundary:
        raise HoldoutError(
            f"a rebalance on {last.date()} is on or after HOLDOUT_START {boundary.date()} "
            "(CLAUDE.md invariant 5). The grid does not evaluate the holdout."
        )
    if pd.Timestamp(terminal_date) >= boundary:
        raise HoldoutError(
            f"the last holding period closes on {pd.Timestamp(terminal_date).date()}, on or after "
            f"HOLDOUT_START {boundary.date()} (CLAUDE.md invariant 5). The grid does not "
            "evaluate the holdout."
        )
    if pd.Timestamp(terminal_date) < last:
        raise GridError("assert_in_sample: the terminal date precedes the last rebalance")


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellSpec:
    """One cell of SPEC.md 9's grid, plus the ``Y`` regime for treatment D's band."""

    variant: str
    treatment: str
    regime: str = "patient"
    #: W6-P3: the band this cell belongs to (``"spread_high"``, ``"te_0.5x"``, ...); empty
    #: for SPEC.md 9's 28 cells.
    tag: str = ""

    def __post_init__(self) -> None:
        if self.variant not in _VARIANT_NUMBER:
            raise GridError(f"CellSpec: unknown variant {self.variant!r}")
        if self.treatment not in _TREATMENT_LETTER:
            raise GridError(f"CellSpec: unknown treatment {self.treatment!r}")
        if self.regime not in ("patient", "urgent"):
            raise GridError(f"CellSpec: regime is 'patient' or 'urgent', got {self.regime!r}")

    @property
    def cell_id(self) -> str:
        """The 28-cell key: variant number and treatment letter, ``4D``."""
        return f"{_VARIANT_NUMBER[self.variant]}{_TREATMENT_LETTER[self.treatment]}"

    @property
    def label(self) -> str:
        base = f"{self.variant}/{self.treatment}/{self.regime}"
        return f"{base}/{self.tag}" if self.tag else base

    @property
    def key(self) -> str:
        """``cell_id`` plus the band tag: ``4D:spread_high``."""
        return f"{self.cell_id}:{self.tag}" if self.tag else self.cell_id

    @property
    def dense(self) -> bool:
        return self.variant in _DENSE

    @property
    def optimizer_sees_cost(self) -> bool:
        return self.treatment in ("flat_spread", "time_varying")

    @property
    def charged(self) -> bool:
        return self.treatment != "none"


def cells(settings: config_mod.Config) -> tuple[CellSpec, ...]:
    """SPEC.md 9's 28 cells in row-major order; treatment D carries both regimes."""
    grid = settings.model.optimizer.grid
    out: list[CellSpec] = []
    for variant in grid.covariance_variants:
        for treatment in grid.cost_treatments:
            if treatment == "time_varying":
                out.extend(
                    CellSpec(variant, treatment, regime) for regime in grid.time_varying_regimes
                )
            else:
                out.append(CellSpec(variant, treatment, "patient"))
    return tuple(out)


# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Forecast:
    """One rebalance's risk forecast in the optimizer's units (decimal^2 per period)."""

    exposures: FloatArray
    factor_covariance: FloatArray
    specific_variance: FloatArray
    #: Ledoit-Wolf's shrinkage intensity; ``NaN`` for every other variant.
    intensity: float = float("nan")

    def risk(self) -> opt.FactorRisk:
        return opt.FactorRisk(
            exposures=self.exposures,
            factor_covariance=self.factor_covariance,
            specific_variance=self.specific_variance,
        )


def dense_covariance(returns: FloatArray, *, variant: str) -> tuple[FloatArray, float]:
    """A dense ``N x N`` covariance from a ``T x N`` block, moments about ZERO.

    ``sample_equal_weight`` is ``R'R / T`` (SPEC.md 9.1 ruling 2). ``ledoit_wolf``
    is :func:`mafrm.factors.statistical.ledoit_wolf_constant_correlation` on the
    same rows with ``centre=False`` (ruling 3); its intensity is returned beside
    the matrix. Both are symmetrised and PSD is asserted at the point of use.
    """
    block = np.asarray(returns, dtype=float)
    if block.ndim != 2 or block.shape[0] < 2:
        raise GridError(f"dense_covariance: expected a T x N block with T >= 2, got {block.shape}")
    if not np.all(np.isfinite(block)):
        raise GridError("dense_covariance: the return block has a non-finite entry")
    if variant == "sample_equal_weight":
        sigma = block.T @ block / block.shape[0]
        return 0.5 * (sigma + sigma.T), float("nan")
    if variant == "ledoit_wolf":
        estimate = statistical.ledoit_wolf_constant_correlation(block, centre=False)
        assert estimate.covariance is not None
        return np.asarray(estimate.covariance, dtype=float), float(estimate.intensity)
    raise GridError(f"dense_covariance: {variant!r} is not a dense variant")


def _panel_spec(settings: config_mod.Config, variant: str) -> bias_report.VariantSpec:
    name = _PANEL_SPEC[variant]
    for spec in bias_report.variants(settings):
        if spec.name == name:
            return spec
    raise GridError(f"no ladder variant named {name!r} in bias_report.variants")


def _forecasts(
    universe: verification.Universe, settings: config_mod.Config, cell: CellSpec
) -> tuple[Forecast, ...]:
    """The forecast standing at each rebalance, built from everything through that close."""
    positions = universe.panel.dates.get_indexer(universe.rebalance_dates)
    if np.any(positions < 0):
        raise GridError("a rebalance date is not a scored date")
    out: list[Forecast] = []
    if cell.dense:
        n = len(universe.assets)
        for stamp in universe.rebalance_dates:
            # The rebalance date's own close is known when the forecast is made.
            sigma, intensity = _dense_forecast(universe, settings, cell.variant, through=stamp)
            out.append(
                Forecast(
                    exposures=np.eye(n),
                    factor_covariance=sigma,
                    specific_variance=np.zeros(n),
                    intensity=intensity,
                )
            )
        return tuple(out)
    spec = _panel_spec(settings, cell.variant)
    for position in positions:
        exposures, factor, specific = universe.panel.ingredients(spec, int(position) + 1)
        out.append(
            Forecast(
                exposures=np.asarray(exposures, dtype=float),
                factor_covariance=np.asarray(factor, dtype=float) * universe.scale,
                specific_variance=np.asarray(specific, dtype=float) * universe.scale,
            )
        )
    return tuple(out)


def _dense_forecast(
    universe: verification.Universe,
    settings: config_mod.Config,
    variant: str,
    *,
    through: pd.Timestamp | None = None,
    before: pd.Timestamp | None = None,
) -> tuple[FloatArray, float]:
    """A dense forecast in decimal^2 per PERIOD from the last ``dense_window`` complete-case days.

    THE DENSE VARIANTS SEE THE TRUE EXCESS RETURNS, NOT THE REGRESSAND
        The residual panel every pipeline variant forecasts starts in April 2009,
        when the last rolling beta is first fitted, so it holds fifteen rows
        before the first rebalance and cannot fill a 242-day window. The dense
        estimators therefore read ``Universe.excess_history`` -- the assets'
        true daily excess returns from the 2007 sample start, complete cases --
        which is also the series the book is marked on. For the four government
        zeros the pipeline's regressand is the duration leg (SPEC.md 4.3) and
        differs from the true return by the carry-and-roll term; the dense
        variants carry no such gap. Stated in the report rather than hidden.

    ``through`` includes that date's row (a forecast made at its close);
    ``before`` excludes it (a forecast FOR that date). Exactly one is given.
    """
    if (through is None) == (before is None):
        raise GridError("_dense_forecast: give exactly one of through= and before=")
    window = settings.model.optimizer.grid.dense_window
    history = universe.excess_history[list(universe.assets)]
    if through is not None:
        block = history.loc[:through]
    else:
        assert before is not None
        block = history.loc[history.index < before]
    if len(block) < window:
        stamp = through if through is not None else before
        raise GridError(
            f"the dense window of {window} complete-case days is not full at {stamp!s}: "
            f"{len(block)} available"
        )
    rows = block.to_numpy(dtype=float)[-window:]
    sigma, intensity = dense_covariance(rows, variant=variant)
    return sigma * universe.horizon_days, intensity


def min_var_realised_volatility(
    universe: verification.Universe, settings: config_mod.Config, variant: str
) -> float:
    """SPEC.md 6.5's realised min-var volatility, daily, in bps/day, on the scored window.

    The same construction as :func:`mafrm.factors.bias_report.run_for`'s: the
    unconstrained minimum-variance weights of each scored date's forecast applied
    to that date's realised return, ``ddof = 1``. For the five pipeline variants
    this reproduces the bias report's figure; for the two dense variants it is
    the first time it is computed. The realised return is the panel's regressand
    for every variant alike; the dense variants' forecasts come from the true
    excess-return history (see :func:`_dense_forecast`).
    """
    panel = universe.panel
    rows = universe.asset_returns[list(universe.assets)].to_numpy(dtype=float)
    realised = rows[panel.positions]
    if variant in _DENSE:
        series = np.empty(len(panel.dates))
        for index, stamp in enumerate(panel.dates):
            sigma, _ = _dense_forecast(universe, settings, variant, before=pd.Timestamp(stamp))
            series[index] = float(validation.minimum_variance_weights(sigma) @ realised[index])
        return float(np.std(series, ddof=1))
    forecasts = panel.forecasts(_panel_spec(settings, variant))
    series = np.array(
        [
            float(validation.minimum_variance_weights(forecast.covariance) @ realised[index])
            for index, forecast in enumerate(forecasts)
        ]
    )
    return float(np.std(series, ddof=1))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellInputs:
    """Everything :func:`run_cell` consumes. Assets are positions in a vector."""

    cell: CellSpec
    assets: tuple[str, ...]
    rebalance_dates: pd.DatetimeIndex
    terminal_date: pd.Timestamp
    holdout_start: pd.Timestamp
    #: Daily price panel through ``terminal_date``; the engine's.
    prices: pd.DataFrame
    forecasts: tuple[Forecast, ...]
    #: ``R x N`` alpha per rebalance, centred, per-period return units.
    alphas: FloatArray
    nav: float
    #: Per-period forecast volatility the constraint (or SPEC.md 8.4) targets.
    te_target: float
    periods_per_year: float
    #: What the OPTIMIZER prices; ``None`` for a cost-free objective.
    optimizer_cost: engine.CostModel | None
    #: What the ENGINE charges.
    realised_cost: engine.CostModel
    #: The identity assets, for the ``IR`` diagnostic; may be empty.
    excluded: tuple[str, ...] = ()
    #: Trailing realised volatility of the equal-weight book at each rebalance,
    #: for H5's "worst 5% of volatility" windows; ``None`` when not available.
    stress: pd.Series | None = None
    # -- W6-P3 overrides; ``None`` means the grid's reference value ----------------
    gamma_trade: float | None = None
    #: ``"tracking_error_constraint"`` or ``"spec_initialisation"``.
    risk_aversion: str | None = None
    #: A hard per-asset upper bound replacing ``constraints.position_box.upper``.
    position_box: float | None = None
    #: Impose the ADV cap HARD (SPEC.md 10.4's re-optimisation) instead of as a hinge.
    adv_participation_hard: bool = False
    relaxation_ladder: tuple[str, ...] | None = None
    # -- W6-P3 context for the financing leg and SPEC.md 10's diagnostics ------------
    #: Daily decimal risk-free rate on the price calendar; ``None`` -> no financing line.
    risk_free: pd.Series | None = None
    #: Daily DECIMAL factor returns on the price calendar, ``K`` columns in the
    #: forecasts' factor order (a dense variant's are its assets); ``None`` -> no attribution.
    factor_returns: pd.DataFrame | None = None
    #: Asset -> sleeve, for the Brinson-Fachler check; ``None`` -> one sleeve.
    sleeves: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        n = len(self.assets)
        r = len(self.rebalance_dates)
        if len(self.forecasts) != r:
            raise GridError(f"CellInputs: {len(self.forecasts)} forecasts against {r} rebalances")
        if self.alphas.shape != (r, n):
            raise GridError(f"CellInputs: alphas {self.alphas.shape} against ({r}, {n})")
        if not (np.isfinite(self.nav) and self.nav > 0.0):
            raise GridError(f"CellInputs: nav must be positive, got {self.nav}")
        if not (np.isfinite(self.te_target) and self.te_target > 0.0):
            raise GridError(f"CellInputs: te_target must be positive, got {self.te_target}")
        if list(self.prices.columns) != list(self.assets):
            raise GridError("CellInputs: prices must carry exactly the assets, in order")
        if self.risk_aversion is not None and self.risk_aversion not in (
            "tracking_error_constraint",
            "spec_initialisation",
        ):
            raise GridError(f"CellInputs: unknown risk_aversion {self.risk_aversion!r}")
        if self.position_box is not None and not (0.0 < self.position_box <= 1.0):
            raise GridError(f"CellInputs: position_box must be in (0, 1], got {self.position_box}")
        if self.gamma_trade is not None and not self.gamma_trade > 0.0:
            raise GridError(f"CellInputs: gamma_trade must be positive, got {self.gamma_trade}")
        if self.risk_free is not None:
            missing = self.prices.index.difference(self.risk_free.index)
            if len(missing) > 0:
                raise GridError("CellInputs: risk_free must cover every price date")
        if self.factor_returns is not None:
            missing = self.prices.index.difference(self.factor_returns.index)
            if len(missing) > 0:
                raise GridError("CellInputs: factor_returns must cover every price date")
            k = self.forecasts[0].factor_covariance.shape[0] if self.forecasts else 0
            if self.factor_returns.shape[1] != k:
                raise GridError(
                    f"CellInputs: factor_returns has {self.factor_returns.shape[1]} columns "
                    f"against {k} factors in the forecast"
                )


def _book_size(universe: verification.Universe, settings: config_mod.Config) -> float:
    """``A = p N min_i median(V_i)`` at the ruled participation, or the ruled fixed NAV."""
    book = settings.model.optimizer.grid.book_size
    if book.rule == "fixed":
        assert book.nav_dollars is not None
        return float(book.nav_dollars)
    assert book.participation_of_adv is not None
    return float(book.participation_of_adv * len(universe.assets) * universe.adv_median.min())


def _cost_frames(
    universe: verification.Universe,
    settings: config_mod.Config,
    *,
    regime: str,
    flat: bool,
    spread_end: str | None = None,
) -> engine.CostModel:
    costs = settings.model.costs
    grid = settings.model.optimizer.grid
    assets = list(universe.assets)
    end = spread_end if spread_end is not None else grid.spread_end
    if flat:
        half = pd.DataFrame(grid.flat_half_spread, index=universe.rebalance_dates, columns=assets)
    else:
        half = universe.half_spread[end][assets]
    return engine.CostModel(
        half_spread=half,
        daily_volatility=universe.daily_volatility[assets],
        adv_dollars=universe.adv_dollars[assets],
        prefactor=float(getattr(costs.square_root_prefactor, regime)),
        exponent=costs.total_cost_exponent,
        asymmetry=costs.buy_sell_asymmetry,
    )


def cell_inputs(
    universe: verification.Universe,
    settings: config_mod.Config,
    cell: CellSpec,
    *,
    spread_end: str | None = None,
    tracking_error_multiple: float | None = None,
    nav: float | None = None,
    gamma_trade: float | None = None,
    risk_aversion: str | None = None,
    position_box: float | None = None,
    adv_participation_hard: bool = False,
    relaxation_ladder: tuple[str, ...] | None = None,
) -> CellInputs:
    """The real inputs for one cell: forecasts, alpha, both cost models, the NAV.

    The keyword overrides are W6-P3's bands and the capacity re-optimisation;
    ``None`` is the grid's reference value in every case.
    """
    grid = settings.model.optimizer.grid
    costs = settings.model.costs
    assets = list(universe.assets)
    holdout = pd.Timestamp(settings.require_holdout_start())
    assert_in_sample(universe.rebalance_dates, universe.terminal_date, holdout)
    end = spread_end if spread_end is not None else grid.spread_end
    if end not in ("low", "high"):
        raise GridError(f"cell_inputs: spread_end is 'low' or 'high', got {end!r}")
    multiple = (
        tracking_error_multiple
        if tracking_error_multiple is not None
        else grid.tracking_error_multiple
    )

    rates = universe.alpha_rate.loc[universe.rebalance_dates, assets]
    centred = alpha_mod.demean(rates)
    assert isinstance(centred, pd.DataFrame)
    scaled = alpha_mod.to_horizon(centred, days=universe.horizon_days)
    alphas = np.asarray(scaled.to_numpy(dtype=float), dtype=float)
    if not np.all(np.isfinite(alphas)):
        raise GridError("cell_inputs: RSTR is missing at a rebalance date")

    free = engine.CostModel.free(
        index=universe.rebalance_dates, columns=pd.Index(assets), exponent=costs.total_cost_exponent
    )
    true_cost = _cost_frames(universe, settings, regime=cell.regime, flat=False, spread_end=end)
    optimizer_cost: engine.CostModel | None
    if cell.treatment == "flat_spread":
        optimizer_cost = _cost_frames(
            universe, settings, regime=cell.regime, flat=True, spread_end=end
        )
    elif cell.treatment == "time_varying":
        optimizer_cost = true_cost
    else:
        optimizer_cost = None
    realised_cost = true_cost if cell.charged else free

    equal_weight = universe.excess[assets].mean(axis=1)
    stress = (
        equal_weight.rolling(universe.horizon_days, min_periods=universe.horizon_days)
        .std(ddof=1)
        .reindex(universe.rebalance_dates)
    )
    prices = universe.prices[assets]
    # SPEC.md 10.3's realised factor returns: the panel's for a factor model; a
    # dense variant enters as X = I, so its "factors" are the assets themselves.
    factor_returns: pd.DataFrame | None
    if cell.dense:
        factor_returns = universe.excess[assets].reindex(prices.index)
    elif universe.factor_returns is not None:
        factor_returns = universe.factor_returns.reindex(prices.index)
    else:
        factor_returns = None
    return CellInputs(
        cell=cell,
        assets=tuple(assets),
        rebalance_dates=universe.rebalance_dates,
        terminal_date=universe.terminal_date,
        holdout_start=holdout,
        prices=prices,
        forecasts=_forecasts(universe, settings, cell),
        alphas=alphas,
        nav=nav if nav is not None else _book_size(universe, settings),
        te_target=universe.te_target * multiple,
        periods_per_year=settings.model.data.trading_days_per_year / universe.horizon_days,
        optimizer_cost=optimizer_cost,
        realised_cost=realised_cost,
        excluded=tuple(universe.excluded),
        stress=stress,
        gamma_trade=gamma_trade,
        risk_aversion=risk_aversion,
        position_box=position_box,
        adv_participation_hard=adv_participation_hard,
        relaxation_ladder=relaxation_ladder,
        risk_free=universe.risk_free.reindex(prices.index)
        if universe.risk_free is not None
        else None,
        factor_returns=factor_returns,
        sleeves=dict(universe.sleeves),
    )


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellResult:
    inputs: CellInputs
    #: One row per rebalance.
    rows: pd.DataFrame
    targets: pd.DataFrame
    backtest: engine.BacktestResult
    readings: tuple[tuple[con.Reading, ...], ...]
    monthly: pd.DataFrame
    identity: metrics_mod.IdentityTerms
    sharpe_net: metrics_mod.SharpeEstimate
    sharpe_gross: metrics_mod.SharpeEstimate
    summary: dict[str, float] = field(default_factory=dict)
    #: Filled by :func:`run_grid` once every cell's Sharpe exists.
    deflated: metrics_mod.DeflatedSharpe | None = None
    #: SPEC.md 10.1-10.3 on this cell (W6-P3).
    diagnostics: diagnostics_mod.CellDiagnostics | None = None

    @property
    def cell(self) -> CellSpec:
        return self.inputs.cell


def _risk_aversion(inputs: CellInputs, grid: config_mod.OptimizerGridConfig) -> str:
    return inputs.risk_aversion if inputs.risk_aversion is not None else grid.risk_aversion


def _limits(
    inputs: CellInputs, opt_cfg: config_mod.OptimizerConfig
) -> tuple[tuple[con.Limit, ...], tuple[str, ...]]:
    """The cell's limits: hinges from ``grid.hinge_priorities``, the TE bound in constraint mode.

    W6-P3's overrides: ``position_box`` replaces the box's bound (hard), a hard
    ADV cap drops the hinge priority, and ``relaxation_ladder`` replaces the
    grid's ladder; only HARD limits can sit on a ladder.
    """
    grid = opt_cfg.grid
    limits: list[con.Limit] = []
    for limit in con.limits_from_config(opt_cfg.constraints):
        if (
            limit.name == _ADV_LIMIT
            and inputs.optimizer_cost is None
            and grid.cost_free_drops_adv_cap
        ):
            continue
        priority = grid.hinge_priorities.get(limit.name, limit.priority)
        bound = limit.bound
        if limit.name == _ADV_LIMIT and inputs.adv_participation_hard:
            priority = None
        if limit.name == _BOX_LIMIT and inputs.position_box is not None:
            bound, priority = inputs.position_box, None
        limits.append(con.Limit(name=limit.name, bound=bound, priority=priority))
    names = {limit.name for limit in limits}
    if _risk_aversion(inputs, grid) == "tracking_error_constraint":
        if _TE_LIMIT in names:
            raise GridError(
                "constraint mode imposes the tracking-error bound itself; "
                "optimizer.constraints.tracking_error.bound must be null"
            )
        limits.append(con.Limit(name=_TE_LIMIT, bound=inputs.te_target, priority=None))
        names.add(_TE_LIMIT)
    order = (
        inputs.relaxation_ladder if inputs.relaxation_ladder is not None else grid.relaxation_ladder
    )
    hard = {limit.name for limit in limits if limit.hard}
    ladder = tuple(name for name in order if name in hard)
    return tuple(limits), ladder


def _trading_cost(
    model: engine.CostModel, stamp: pd.Timestamp, assets: list[str], nav_pre: float
) -> opt.TradingCost:
    return opt.TradingCost(
        half_spread=model.half_spread.loc[stamp, assets].to_numpy(dtype=float),
        daily_volatility=model.daily_volatility.loc[stamp, assets].to_numpy(dtype=float),
        adv_over_nav=model.adv_dollars.loc[stamp, assets].to_numpy(dtype=float) / nav_pre,
        prefactor=model.prefactor,
        exponent=model.exponent,
        asymmetry=model.asymmetry,
    )


def _objective_value(result: opt.RebalanceResult, limits: Sequence[con.Limit]) -> float:
    """SPEC.md 8.1's objective at the result's point: the terms less the hinge penalties."""
    terms = result.terms
    value = terms["alpha"] - terms["risk"] - terms["misalignment"] - terms["trading_cost"]
    priorities = {limit.name: limit.priority for limit in limits if not limit.hard}
    for reading in result.readings:
        priority = priorities.get(reading.name)
        if priority is not None and np.isfinite(reading.violation):
            value -= priority * reading.violation
    return float(value)


def _resolve_verdict(
    primary: opt.RebalanceResult,
    fallback: opt.RebalanceResult,
    limits: Sequence[con.Limit],
    *,
    te_target: float,
    constraint_mode: bool,
    tolerance: float,
) -> tuple[str, float]:
    """The amended criterion (SPEC.md 9.3 ruling 6, W6-P3b).

    Two points are compared only when BOTH are feasible to the solver's
    tolerance; the feasible point with the better objective -- higher, since
    SPEC.md 8.1's problem is a maximisation -- is the solution. The L1 to an
    infeasible or worse-objective fallback is the fallback's error, reported
    and not acted on. Returns the verdict and the relative objective gap
    ``(fallback - primary) / |primary|``.
    """
    gap = float("nan")
    primary_value = _objective_value(primary, limits)
    fallback_value = _objective_value(fallback, limits)
    if primary_value != 0.0:
        gap = (fallback_value - primary_value) / abs(primary_value)
    if constraint_mode:
        # The same reading as optimizer._readings: slack within tol * max(1, bound).
        allowed = 1.0 + tolerance * max(1.0, te_target) / te_target
        if fallback.forecast_volatility / te_target > allowed:
            return _VERDICT_FALLBACK_INFEASIBLE, gap
    if fallback_value > primary_value:
        return _VERDICT_FALLBACK_BETTER, gap
    return _VERDICT_PRIMARY_BETTER, gap


def _te_multiplier(result: opt.RebalanceResult) -> float:
    for reading in result.readings:
        if reading.name == _TE_LIMIT and not reading.relaxed:
            return float(reading.multiplier) if np.isfinite(reading.multiplier) else 0.0
    return float("nan")


@dataclass(frozen=True)
class _Rebalance:
    """One rebalance's fixed ingredients, so the solve helpers close over nothing."""

    alpha: FloatArray
    risk: opt.FactorRisk
    prior: FloatArray
    cost: opt.TradingCost | None
    limits: tuple[con.Limit, ...]
    ladder: tuple[str, ...]
    sigma: FloatArray
    decomposition: alpha_mod.Misalignment


def _solve(
    rebalance: _Rebalance,
    opt_cfg: config_mod.OptimizerConfig,
    *,
    gamma_risk: float,
    penalty: alpha_mod.MisalignmentPenalty,
    gamma_trade: float | None = None,
    swap_solvers: bool = False,
) -> opt.RebalanceResult:
    """One solve. ``swap_solvers`` puts the fallback first -- the W6-P3 re-solve."""
    grid = opt_cfg.grid
    primary, fallback = opt_cfg.solver.primary, opt_cfg.solver.fallback
    if swap_solvers:
        primary, fallback = fallback, primary
    problem = opt.RebalanceProblem(
        alpha=rebalance.alpha,
        risk=rebalance.risk,
        prior_weights=rebalance.prior,
        penalties=opt.Penalties(
            gamma_risk=gamma_risk,
            gamma_trade=gamma_trade if gamma_trade is not None else grid.gamma_trade,
            gamma_hold=opt_cfg.gamma_hold,
            psi_unit=penalty.psi_unit,
            misalignment_direction=penalty.direction if penalty.active else None,
            rho=opt_cfg.robustification.return_forecast_rho,
            varrho=opt_cfg.robustification.covariance_varrho,
        ),
        cost=rebalance.cost,
        limits=rebalance.limits,
        long_only=opt_cfg.constraints.long_only,
        fully_invested=opt_cfg.constraints.fully_invested,
        relaxation_order=rebalance.ladder,
        primary_solver=primary,
        fallback_solver=fallback,
        feasibility_tolerance=opt_cfg.solver.feasibility_tolerance,
    )
    return opt.solve_rebalance(problem)


def _penalty_at(
    rebalance: _Rebalance, opt_cfg: config_mod.OptimizerConfig, gamma_risk: float
) -> alpha_mod.MisalignmentPenalty:
    n = rebalance.alpha.size
    if opt_cfg.misalignment_penalty != "msci" or not np.isfinite(gamma_risk):
        return alpha_mod.MisalignmentPenalty(psi_unit=0.0, direction=np.zeros(n))
    return alpha_mod.misalignment_penalty(
        rebalance.decomposition, rebalance.sigma, gamma_risk=gamma_risk
    )


def run_cell(
    inputs: CellInputs, settings: config_mod.Config, *, progress: bool = False
) -> CellResult:
    """Solve every rebalance, feed the engine, score the cell. The holdout guard runs first."""
    assert_in_sample(inputs.rebalance_dates, inputs.terminal_date, inputs.holdout_start)
    opt_cfg = settings.model.optimizer
    grid = opt_cfg.grid
    n = len(inputs.assets)
    assets = list(inputs.assets)
    limits, ladder = _limits(inputs, opt_cfg)
    constraint_mode = _risk_aversion(inputs, grid) == "tracking_error_constraint"
    two_pass = constraint_mode and grid.misalignment_pricing == "recovered_multiplier_two_pass"
    gamma_trade = inputs.gamma_trade

    targets = pd.DataFrame(np.nan, index=inputs.rebalance_dates, columns=assets)
    rows: list[dict[str, object]] = []
    readings: list[tuple[con.Reading, ...]] = []
    #: Every flagged rebalance's problem, for the re-solve after the loop.
    flagged: list[tuple[int, _Rebalance, float, alpha_mod.MisalignmentPenalty, FloatArray]] = []
    primary_results: dict[int, opt.RebalanceResult] = {}
    for i, t in enumerate(inputs.rebalance_dates):
        stamp = pd.Timestamp(t)
        if i == 0:
            nav_pre = inputs.nav
            prior = np.zeros(n)
        else:
            partial = engine.run_backtest(
                inputs.prices.loc[:stamp],
                targets.iloc[:i],
                cost_model=inputs.realised_cost,
                initial_nav=inputs.nav,
            )
            nav_pre = float(partial.nav.loc[stamp])
            prior = np.asarray(partial.weights.loc[stamp].to_numpy(dtype=float), dtype=float)

        forecast = inputs.forecasts[i]
        risk = forecast.risk()
        sigma = risk.covariance()
        alpha = inputs.alphas[i]
        ir = opt.information_ratio(alpha, sigma)
        keep = [k for k, name in enumerate(assets) if name not in inputs.excluded]
        ir_without_identity = (
            opt.information_ratio(alpha[keep] - alpha[keep].mean(), sigma[np.ix_(keep, keep)])
            if len(keep) < n and len(keep) > 1
            else ir
        )
        spec_lambda = (
            opt.gamma_risk_from_spec(alpha, sigma, te_target=inputs.te_target)
            if np.any(alpha)
            else float("nan")
        )
        decomposition = alpha_mod.decompose(alpha, forecast.exposures)

        cost: opt.TradingCost | None = None
        if inputs.optimizer_cost is not None:
            cost = _trading_cost(inputs.optimizer_cost, stamp, assets, nav_pre)
        rebalance = _Rebalance(
            alpha=alpha,
            risk=risk,
            prior=prior,
            cost=cost,
            limits=limits,
            ladder=ladder,
            sigma=sigma,
            decomposition=decomposition,
        )

        no_penalty = alpha_mod.MisalignmentPenalty(psi_unit=0.0, direction=np.zeros(n))
        if constraint_mode:
            first = _solve(
                rebalance, opt_cfg, gamma_risk=0.0, penalty=no_penalty, gamma_trade=gamma_trade
            )
            mu_first = _te_multiplier(first)
            implied = mu_first / (2.0 * inputs.te_target) if np.isfinite(mu_first) else 0.0
            if two_pass:
                penalty = _penalty_at(rebalance, opt_cfg, implied)
                result = (
                    _solve(
                        rebalance,
                        opt_cfg,
                        gamma_risk=0.0,
                        penalty=penalty,
                        gamma_trade=gamma_trade,
                    )
                    if penalty.active
                    else first
                )
            else:
                penalty = no_penalty
                result = first
            gamma_risk = implied
            solved_at = 0.0
            mu_final = _te_multiplier(result)
            implied_final = mu_final / (2.0 * inputs.te_target) if np.isfinite(mu_final) else 0.0
        else:
            gamma_risk = spec_lambda if np.isfinite(spec_lambda) else 0.0
            penalty = _penalty_at(rebalance, opt_cfg, gamma_risk)
            result = _solve(
                rebalance, opt_cfg, gamma_risk=gamma_risk, penalty=penalty, gamma_trade=gamma_trade
            )
            solved_at = gamma_risk
            mu_first = mu_final = float("nan")
            implied_final = gamma_risk
        if result.status == _INACCURATE:
            flagged.append((i, rebalance, solved_at, penalty, result.weights.copy()))
            primary_results[i] = result

        targets.loc[stamp] = result.weights
        readings.append(result.readings)
        row: dict[str, object] = {
            "cell": inputs.cell.cell_id,
            "configuration": inputs.cell.label,
            "date": stamp,
            "nav_pre": nav_pre,
            "gamma_risk_spec": spec_lambda,
            "gamma_risk_used": float(gamma_risk),
            "gamma_risk_recovered": float(implied_final),
            "te_multiplier_first_pass": mu_first,
            "te_multiplier": mu_final,
            "information_ratio": ir,
            "information_ratio_without_identity": ir_without_identity,
            "cos_theta": decomposition.cos_theta,
            "psi_unit": penalty.psi_unit,
            "ledoit_wolf_intensity": forecast.intensity,
            "forecast_volatility": result.forecast_volatility,
            "te_attainment": result.forecast_volatility / inputs.te_target,
            "turnover": result.turnover,
            "trading_cost_priced": result.trading_cost,
            "status": result.status,
            "rung": result.attempt.rung,
            "attempt": result.attempt.label,
            "relaxed": ",".join(result.relaxed),
            "fell_back": result.fell_back,
            "at_lower_bound": result.at_lower_bound,
            "largest_weight": result.largest_weight,
            "effective_assets": result.effective_assets,
            "term_alpha": result.terms["alpha"],
            "term_misalignment": result.terms["misalignment"],
            "term_trading_cost": result.terms["trading_cost"],
        }
        for reading in result.readings:
            row[f"{reading.name}_binding"] = reading.binding
            row[f"{reading.name}_multiplier"] = reading.multiplier
            row[f"{reading.name}_value"] = reading.value
            row[f"{reading.name}_violation"] = reading.violation
        if inputs.stress is not None:
            row["stress_volatility"] = float(inputs.stress.loc[stamp])
        for name, weight in zip(assets, result.weights, strict=True):
            row[f"w_{name}"] = float(weight)
        rows.append(row)
        if progress and (i + 1) % 50 == 0:
            print(f"  {inputs.cell.label}: {i + 1:,} / {len(inputs.rebalance_dates):,}", flush=True)

    backtest = engine.run_backtest(
        inputs.prices, targets, cost_model=inputs.realised_cost, initial_nav=inputs.nav
    )
    frame = pd.DataFrame(rows).set_index("date")
    frame["resolve_l1"] = np.nan
    frame["resolve_status"] = ""
    frame["resolve_te_ratio"] = np.nan
    frame["resolve_objective_gap"] = np.nan
    frame["resolve_verdict"] = ""
    # THE RE-SOLVE (W6-P3, SPEC.md 9.2's registered residual; ruling 6). The
    # same problem, the fallback solver first; the weights the cell USED stay
    # the primary's, and the L1 distance is recorded per flagged rebalance.
    for i, rebalance, solved_at, penalty, used in flagged:
        again = _solve(
            rebalance,
            opt_cfg,
            gamma_risk=solved_at,
            penalty=penalty,
            gamma_trade=gamma_trade,
            swap_solvers=True,
        )
        stamp = inputs.rebalance_dates[i]
        frame.loc[stamp, "resolve_l1"] = float(np.abs(again.weights - used).sum())
        frame.loc[stamp, "resolve_status"] = f"{again.attempt.solver}:{again.status}"
        # Feasibility of the RE-SOLVED point against the TE bound (W6-P3 found the
        # fallback's point 2.7% outside it on the worst date): its forecast
        # volatility over TE_target, so the report can say which point was wrong.
        frame.loc[stamp, "resolve_te_ratio"] = (
            again.forecast_volatility / inputs.te_target if constraint_mode else np.nan
        )
        verdict, gap = _resolve_verdict(
            primary_results[i],
            again,
            limits,
            te_target=inputs.te_target,
            constraint_mode=constraint_mode,
            tolerance=opt_cfg.solver.feasibility_tolerance,
        )
        frame.loc[stamp, "resolve_objective_gap"] = gap
        frame.loc[stamp, "resolve_verdict"] = verdict
    monthly = monthly_returns(inputs.rebalance_dates, inputs.terminal_date, frame, backtest)
    gross = monthly["gross_return"].to_numpy(dtype=float)
    net = monthly["net_return"].to_numpy(dtype=float)
    forecast_vol = monthly["forecast_volatility"].to_numpy(dtype=float)
    q = grid.sharpe_aggregation_periods
    sharpe_net = metrics_mod.annualised_sharpe(net, q=q)
    sharpe_gross = metrics_mod.annualised_sharpe(gross, q=q)
    # The identity on Lo's scale: volatilities annualised by q / eta of the NET
    # series, the series SR_real is the Sharpe of (W6-P2b, operator). On a
    # cost-free cell net = gross, so gross Sharpe x B = SR_paper EXACTLY, and
    # that is asserted rather than eyeballed.
    identity = metrics_mod.identity_terms(
        gross,
        net,
        forecast_vol,
        periods_per_year=inputs.periods_per_year,
        aggregation=sharpe_net.eta,
    )
    if not inputs.cell.charged:
        product = sharpe_gross.annualised * identity.bias_ratio
        if abs(product - identity.sharpe_paper) > 1e-12 * max(1.0, abs(identity.sharpe_paper)):
            raise GridError(
                f"{inputs.cell.label}: gross Sharpe x B = {product} but SR_paper = "
                f"{identity.sharpe_paper}; the identity and the Sharpe columns are on different "
                "scales"
            )
    summary = _summary(inputs, settings, frame, backtest, monthly, readings, identity)
    diagnostics = diagnostics_mod.cell_diagnostics(
        prices=inputs.prices,
        targets=targets,
        terminal_date=inputs.terminal_date,
        forecasts=inputs.forecasts,
        nav_before_costs=backtest.nav_before_costs,
        spread_cost=backtest.spread_cost,
        impact_cost=backtest.impact_cost,
        asymmetry_cost=backtest.asymmetry_cost,
        weights_daily=backtest.weights,
        turnover=backtest.turnover,
        predicted_cost=frame["trading_cost_priced"].astype(float),
        factor_returns=inputs.factor_returns,
        sleeves=inputs.sleeves,
        periods_per_year=inputs.periods_per_year,
        level=settings.model.validation.chi_square_level,
        commission_rate=settings.model.costs.commission,
        per_factor_summary=not inputs.cell.dense,
    )
    summary.update(diagnostics.summary)
    return CellResult(
        inputs=inputs,
        rows=frame,
        targets=targets,
        backtest=backtest,
        readings=tuple(readings),
        monthly=monthly,
        identity=identity,
        sharpe_net=sharpe_net,
        sharpe_gross=sharpe_gross,
        summary=summary,
        diagnostics=diagnostics,
    )


def monthly_returns(
    rebalance_dates: pd.DatetimeIndex,
    terminal_date: pd.Timestamp,
    rows: pd.DataFrame,
    backtest: engine.BacktestResult,
) -> pd.DataFrame:
    """Net and gross returns per holding period, non-overlapping, with the standing forecast."""
    marks = rebalance_dates.append(pd.DatetimeIndex([terminal_date]))
    nav = backtest.nav.reindex(marks).to_numpy(dtype=float)
    cost = backtest.total_cost.reindex(marks[1:]).fillna(0.0).to_numpy(dtype=float)
    net = nav[1:] / nav[:-1] - 1.0
    gross = (nav[1:] + cost) / nav[:-1] - 1.0
    return pd.DataFrame(
        {
            "start": marks[:-1],
            "net_return": net,
            "gross_return": gross,
            "cost_over_nav": cost / nav[:-1],
            "forecast_volatility": rows["forecast_volatility"].to_numpy(dtype=float),
        },
        index=pd.DatetimeIndex(marks[1:], name="end"),
    )


def _financing_bps_per_year(
    inputs: CellInputs, backtest: engine.BacktestResult, monthly: pd.DataFrame
) -> float:
    """The financing leg at RF, nominal, annualised on the initial NAV; ``NaN`` without a rate.

    Operator ruling 1 (W6-P3): the book is marked on the excess-return index,
    where cash at RF accrues zero by identity, so this is the SIZE of the leg
    the identity nets out -- printed so that "negligible" is computed.
    """
    if inputs.risk_free is None:
        return float("nan")
    leg = engine.financing_leg(backtest, inputs.risk_free)
    # Over the holding periods only: before the first rebalance the book IS cash
    # and nothing is scored there (the monthly series starts at that close).
    held = leg.loc[leg.index > inputs.rebalance_dates[0]]
    years = len(monthly) / inputs.periods_per_year
    return float(held.sum() / backtest.initial_nav / years * 1e4)


def _summary(
    inputs: CellInputs,
    settings: config_mod.Config,
    rows: pd.DataFrame,
    backtest: engine.BacktestResult,
    monthly: pd.DataFrame,
    readings: Sequence[Sequence[con.Reading]],
    identity: metrics_mod.IdentityTerms,
) -> dict[str, float]:
    opt_cfg = settings.model.optimizer
    battery = settings.model.validation
    n = len(inputs.assets)
    ppy = inputs.periods_per_year
    net = monthly["net_return"].to_numpy(dtype=float)
    forecast = monthly["forecast_volatility"].to_numpy(dtype=float)
    standardized = validation.standardized_returns(net[:, None], forecast[:, None])
    bias = float(validation.bias_statistic(standardized, clip=battery.standardized_return_clip)[0])
    interval = validation.chi_square_interval(len(net), level=battery.chi_square_level)
    window = battery.rolling_window_months
    mrad = float("nan")
    windows = 0
    if len(net) >= window:
        rolling = validation.rolling_bias_statistic(
            standardized,
            pd.DatetimeIndex(monthly.index),
            window=window,
            clip=battery.standardized_return_clip,
            step=window,
        )
        mrad = validation.mean_absolute_deviation(rolling.values[:, 0])
        windows = int(rolling.values.shape[0])
    spread = float(backtest.spread_cost.to_numpy(dtype=float).sum())
    impact = float(backtest.impact_cost.to_numpy(dtype=float).sum())
    total_cost = float(backtest.total_cost.to_numpy(dtype=float).sum())
    at_floor = rows["at_lower_bound"].to_numpy(dtype=int)
    summary: dict[str, float] = {
        "rebalances": float(len(rows)),
        "first_rung": float((rows["rung"] == 0).sum()),
        "solver_rungs": float((rows["rung"] <= 1).sum()),
        "inaccurate": float((rows["status"] == "optimal_inaccurate").sum()),
        "relaxed": float((rows["relaxed"] != "").sum()),
        "fell_back": float(rows["fell_back"].sum()),
        "cos_theta_median": float(rows["cos_theta"].median()),
        "cos_theta_min": float(rows["cos_theta"].min()),
        "gamma_risk_spec_median": float(rows["gamma_risk_spec"].median()),
        "gamma_risk_recovered_median": float(rows["gamma_risk_recovered"].median()),
        "information_ratio_median": float(rows["information_ratio"].median()),
        "ledoit_wolf_intensity_median": float(rows["ledoit_wolf_intensity"].median()),
        "ledoit_wolf_intensity_max": float(rows["ledoit_wolf_intensity"].max()),
        "effective_assets_mean": float(rows["effective_assets"].mean()),
        "largest_weight_median": float(rows["largest_weight"].median()),
        "largest_weight_max": float(rows["largest_weight"].max()),
        "at_floor_mean": float(at_floor.mean()),
        "corner_share": float((at_floor >= n - 2).mean()),
        "te_attainment_rms": float(np.sqrt(np.mean(forecast**2)) / inputs.te_target),
        "turnover_annual": float(rows["turnover"].mean() * ppy),
        "cost_drag_bps_per_year": identity.cost_drag * 1e4,
        "spread_share_of_cost": spread / total_cost if total_cost > 0.0 else float("nan"),
        "impact_share_of_cost": impact / total_cost if total_cost > 0.0 else float("nan"),
        "gross_annual": identity.gross_return,
        "net_annual": identity.gross_return - identity.cost_drag,
        "realised_volatility_annual": identity.realised_volatility,
        "forecast_volatility_annual": identity.forecast_volatility,
        "bias_ratio": identity.bias_ratio,
        "identity_aggregation_eta": identity.aggregation,
        "bias": bias,
        "bias_lower": interval.lower,
        "bias_upper": interval.upper,
        "mrad": mrad,
        "mrad_windows": float(windows),
        "months": float(len(net)),
        "sharpe_paper": identity.sharpe_paper,
        "sharpe_real": identity.sharpe_real,
        "risk_model_term": identity.risk_model_term,
        "cost_term": identity.cost_term,
        "final_cash_over_nav": float(backtest.cash.iloc[-1] / backtest.nav.iloc[-1]),
        "mean_cash_over_nav": float(
            (backtest.cash / backtest.nav)
            .loc[backtest.nav.index >= inputs.rebalance_dates[0]]
            .mean()
        ),
        "financing_bps_per_year": _financing_bps_per_year(inputs, backtest, monthly),
        "resolve_count": float(rows["resolve_l1"].notna().sum()),
        "resolve_max_l1": (
            float(rows["resolve_l1"].max()) if rows["resolve_l1"].notna().any() else 0.0
        ),
        "resolve_exceeds": float(
            rows["resolve_l1"].notna().any()
            and float(rows["resolve_l1"].max()) > opt_cfg.grid.resolve.l1_threshold
        ),
        # The AMENDED criterion (SPEC.md 9.3 ruling 6, W6-P3b): a re-run is owed only
        # where the L1 is above the threshold AND the fallback's point is feasible AND
        # its objective is better; every other flag is the fallback's error, reported.
        "resolve_actionable": float(
            (
                (rows["resolve_l1"] > opt_cfg.grid.resolve.l1_threshold)
                & (rows["resolve_verdict"] == _VERDICT_FALLBACK_BETTER)
            ).sum()
        ),
        "resolve_worst_te_ratio": (
            float(rows["resolve_te_ratio"].max())
            if rows["resolve_te_ratio"].notna().any()
            else float("nan")
        ),
    }
    table = con.summarise(readings, percentile=opt_cfg.constraints.lagrange_percentile)
    for name in (_TE_LIMIT, _ADV_LIMIT, _BOX_LIMIT):
        if name in table.index:
            entry = table.to_dict(orient="index")[name]
            summary[f"{name}_binding_share"] = float(entry["binding_share"])
            summary[f"{name}_binding_count"] = float(entry["binding"])
            summary[f"{name}_multiplier_median"] = float(entry["multiplier_median"])
            summary[f"{name}_violation_mean"] = float(entry["violation_mean"])
    return summary


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GridResult:
    results: tuple[CellResult, ...]
    #: ``N`` as read from ``experiments.md`` at run time.
    trials: int
    #: Variance of the per-period net Sharpe ratios across the cells.
    trials_variance: float
    min_var_realised_volatility: Mapping[str, float]
    nav: float
    te_target: float
    dense_window: int
    checks: tuple[dict[str, object], ...]
    hypotheses: dict[str, dict[str, object]]

    def by_id(self, cell_id: str, regime: str = "patient") -> CellResult:
        for result in self.results:
            if result.cell.cell_id == cell_id and result.cell.regime == regime:
                return result
        raise GridError(f"no cell {cell_id}/{regime}")


def run_grid(
    universe: verification.Universe,
    settings: config_mod.Config,
    *,
    only: Sequence[CellSpec] | None = None,
    progress: bool = True,
) -> GridResult:
    """Every cell, then the grid-level statistics that need all of them."""
    holdout = pd.Timestamp(settings.require_holdout_start())
    assert_in_sample(universe.rebalance_dates, universe.terminal_date, holdout)
    opt_cfg = settings.model.optimizer
    grid = opt_cfg.grid
    specs = tuple(only) if only is not None else cells(settings)
    results: list[CellResult] = []
    for spec in specs:
        if progress:
            print(f"solving {spec.cell_id} {spec.label}", flush=True)
        results.append(run_cell(cell_inputs(universe, settings, spec), settings, progress=progress))

    trials = metrics_mod.read_trial_count(_EXPERIMENTS_PATH)
    reference = [r for r in results if r.cell.regime == "patient"]
    if trials < len(reference):
        raise GridError(
            f"experiments.md carries N = {trials} but {len(reference)} cells are being evaluated: "
            "the rows must be registered BEFORE the run (experiments.md rule 1)"
        )
    per_period = np.array([r.sharpe_net.per_period for r in reference])
    trials_variance = float(np.var(per_period, ddof=1)) if per_period.size > 1 else 0.0
    deflated = [
        replace_deflated(
            r,
            metrics_mod.deflated_sharpe(
                r.monthly["net_return"].to_numpy(dtype=float),
                trials_variance=trials_variance,
                trials=trials,
            ),
        )
        for r in results
    ]
    variants_run = sorted({r.cell.variant for r in results}, key=lambda v: _VARIANT_NUMBER[v])
    min_var = {v: min_var_realised_volatility(universe, settings, v) for v in variants_run}
    out = GridResult(
        results=tuple(deflated),
        trials=trials,
        trials_variance=trials_variance,
        min_var_realised_volatility=min_var,
        nav=deflated[0].inputs.nav,
        te_target=deflated[0].inputs.te_target,
        dense_window=grid.dense_window,
        checks=(),
        hypotheses={},
    )
    checks = registered_checks(out)
    # The headline clauses read leg (e)'s verdict, so score them on a result
    # that already carries the checks.
    with_checks = GridResult(
        results=out.results,
        trials=out.trials,
        trials_variance=out.trials_variance,
        min_var_realised_volatility=out.min_var_realised_volatility,
        nav=out.nav,
        te_target=out.te_target,
        dense_window=out.dense_window,
        checks=checks,
        hypotheses={},
    )
    hypotheses = score_hypotheses(with_checks, settings)
    return GridResult(
        results=out.results,
        trials=out.trials,
        trials_variance=out.trials_variance,
        min_var_realised_volatility=out.min_var_realised_volatility,
        nav=out.nav,
        te_target=out.te_target,
        dense_window=out.dense_window,
        checks=checks,
        hypotheses=hypotheses,
    )


def replace_deflated(result: CellResult, deflated: metrics_mod.DeflatedSharpe) -> CellResult:
    return CellResult(
        inputs=result.inputs,
        rows=result.rows,
        targets=result.targets,
        backtest=result.backtest,
        readings=result.readings,
        monthly=result.monthly,
        identity=result.identity,
        sharpe_net=result.sharpe_net,
        sharpe_gross=result.sharpe_gross,
        summary=result.summary,
        deflated=deflated,
        diagnostics=result.diagnostics,
    )


# ---------------------------------------------------------------------------
# Registered checks (experiments.md rows 211-238) and H5 / H6
# ---------------------------------------------------------------------------


def _cell(
    grid: GridResult, variant: int, treatment: str, regime: str = "patient"
) -> CellResult | None:
    try:
        return grid.by_id(f"{variant}{treatment}", regime)
    except GridError:
        return None


def registered_checks(grid: GridResult) -> tuple[dict[str, object], ...]:
    """The legs registered in experiments.md before the run, each with its verdict."""
    checks: list[dict[str, object]] = []
    patient = [r for r in grid.results if r.cell.regime == "patient"]
    factor_cells = [r for r in patient if not r.cell.dense]
    charged = [r for r in patient if r.cell.charged]

    def add(leg: str, claim: str, holds: bool, observed: str) -> None:
        checks.append({"leg": leg, "claim": claim, "holds": bool(holds), "observed": observed})

    # (a) solves at a solver rung everywhere
    bad = [
        r.cell.label for r in grid.results if r.summary["relaxed"] > 0 or r.summary["fell_back"] > 0
    ]
    add(
        "a",
        "every cell solves at a solver rung on every rebalance: nothing relaxed, no fall-back",
        not bad,
        f"{len(bad)} cell(s) relaxed or fell back" + (f": {bad}" if bad else ""),
    )
    # (b) TE constraint binds on >= 90% of rebalances, attainment >= 0.9
    shares = {
        r.cell.label: r.summary.get(f"{_TE_LIMIT}_binding_share", float("nan"))
        for r in grid.results
    }
    attain = {r.cell.label: r.summary["te_attainment_rms"] for r in grid.results}
    worst_share = min(shares.values()) if shares else float("nan")
    worst_attain = min(attain.values()) if attain else float("nan")
    add(
        "b",
        "the TE constraint binds on >= 90% of rebalances and rms attainment is >= 0.9 in every cell",
        bool(worst_share >= 0.9 and worst_attain >= 0.9),
        f"lowest binding share {worst_share:.1%}, lowest rms attainment {worst_attain:.3f}",
    )
    # (c) factor-model cells: B_identity > 1 and B_bias above the interval
    fails = [
        r.cell.label
        for r in factor_cells
        if not (r.identity.bias_ratio > 1.0 and r.summary["bias"] > r.summary["bias_upper"])
    ]
    add(
        "c",
        "every factor-model cell (variants 2-6) has B > 1 with SPEC.md 6.1's B above the chi-square interval",
        not fails,
        f"{len(fails)} of {len(factor_cells)} factor-model cells fail"
        + (f": {fails}" if fails else ""),
    )
    # (d) dense variant 1 below variant 4 at every treatment
    pairs = [(t, _cell(grid, 1, t), _cell(grid, 4, t)) for t in "ABCD"]
    fails = [
        t
        for t, one, four in pairs
        if one and four and not one.identity.bias_ratio < four.identity.bias_ratio
    ]
    obs = ", ".join(
        f"{t}: {one.identity.bias_ratio:.3f} vs {four.identity.bias_ratio:.3f}"
        for t, one, four in pairs
        if one and four
    )
    add(
        "d",
        "variant 1's B is below variant 4's at every treatment (SPEC.md 6.2.6 on the trading book)",
        not fails,
        obs,
    )
    # (e) eigenfactor moves B by < 0.02 against variant 3
    moves = []
    for t in "ABCD":
        three = _cell(grid, 3, t)
        for v in (4, 5):
            cell = _cell(grid, v, t)
            if three and cell:
                moves.append((f"{v}{t}", cell.identity.bias_ratio - three.identity.bias_ratio))
    worst = max((abs(m) for _, m in moves), default=float("nan"))
    add(
        "e",
        "the eigenfactor adjustment moves B by less than 0.02 against variant 3 at both a (H3's refutation persists)",
        bool(worst < 0.02),
        "largest |dB| "
        + f"{worst:.4f}"
        + " ("
        + ", ".join(f"{k} {m:+.4f}" for k, m in moves)
        + ")",
    )
    # (f) VRA moves B toward one against variant 4
    fails = []
    obs_parts = []
    for t in "ABCD":
        four, six = _cell(grid, 4, t), _cell(grid, 6, t)
        if four and six:
            obs_parts.append(
                f"{t}: {six.identity.bias_ratio:.3f} vs {four.identity.bias_ratio:.3f}"
            )
            if not abs(six.identity.bias_ratio - 1.0) < abs(four.identity.bias_ratio - 1.0):
                fails.append(t)
    add(
        "f",
        "variant 6 (VRA) has B closer to one than variant 4 at every treatment",
        not fails,
        ", ".join(obs_parts),
    )
    # (g) LW near sample
    fails = []
    obs_parts = []
    for t in "ABCD":
        one, seven = _cell(grid, 1, t), _cell(grid, 7, t)
        if one and seven:
            diff = seven.identity.bias_ratio - one.identity.bias_ratio
            obs_parts.append(f"{t}: dB {diff:+.4f}")
            if abs(diff) >= 0.05:
                fails.append(t)
    mv1 = grid.min_var_realised_volatility.get("sample_equal_weight", float("nan"))
    mv7 = grid.min_var_realised_volatility.get("ledoit_wolf", float("nan"))
    ratio = mv7 / mv1 if mv1 else float("nan")
    within = bool(abs(ratio - 1.0) < 0.05)
    add(
        "g",
        "Ledoit-Wolf sits near the equal-weighted sample: |dB| < 0.05 at every treatment and realised min-var volatility within 5%",
        not fails and within,
        ", ".join(obs_parts) + f"; min-var vol {mv7:.3f} vs {mv1:.3f} bps/day (ratio {ratio:.4f})",
    )
    # (h1) TC(B) > TC(D); (h2) turnover C < D < B; (h3) net SR D > B
    h1, h2, h3, obs_h = [], [], [], []
    for v in range(1, 8):
        b, c, d = _cell(grid, v, "B"), _cell(grid, v, "C"), _cell(grid, v, "D")
        if not (b and c and d):
            continue
        obs_h.append(
            f"v{v}: TC {b.identity.cost_drag * 1e4:.1f}/{c.identity.cost_drag * 1e4:.1f}/{d.identity.cost_drag * 1e4:.1f}bp, "
            f"turnover {b.summary['turnover_annual']:.2f}/{c.summary['turnover_annual']:.2f}/{d.summary['turnover_annual']:.2f}, "
            f"net SR {b.sharpe_net.annualised:.3f}/{c.sharpe_net.annualised:.3f}/{d.sharpe_net.annualised:.3f}"
        )
        if not b.identity.cost_drag > d.identity.cost_drag:
            h1.append(v)
        if not (
            c.summary["turnover_annual"]
            < d.summary["turnover_annual"]
            < b.summary["turnover_annual"]
        ):
            h2.append(v)
        if not d.sharpe_net.annualised > b.sharpe_net.annualised:
            h3.append(v)
    add(
        "h1",
        "realised cost drag under B exceeds D at every variant",
        not h1,
        "B/C/D per variant: " + "; ".join(obs_h),
    )
    add(
        "h2",
        "annual turnover orders C < D < B at every variant",
        not h2,
        f"fails at variants {h2}" if h2 else "holds everywhere",
    )
    add(
        "h3",
        "net Sharpe under D exceeds B at every variant",
        not h3,
        f"fails at variants {h3}" if h3 else "holds everywhere",
    )
    # (i) risk term > cost term in every charged cell
    fails = [r.cell.label for r in charged if not r.identity.risk_model_term > r.identity.cost_term]
    add(
        "i",
        "the risk-model term exceeds the cost term in every cell where cost is charged (SPEC.md 9's expected headline REFUTED)",
        not fails,
        f"{len(fails)} of {len(charged)} charged cells have cost term >= risk term"
        + (f": {fails}" if fails else ""),
    )
    # (l) ADV hinge active on < 5% of rebalances
    active = {
        r.cell.label: r.summary.get(f"{_ADV_LIMIT}_binding_share", 0.0)
        for r in grid.results
        if r.inputs.optimizer_cost is not None
    }
    worst_active = max(active.values()) if active else 0.0
    add(
        "l",
        "the ADV hinge is active on fewer than 5% of rebalances in every costed cell at the reference NAV",
        bool(worst_active < 0.05),
        f"largest active share {worst_active:.1%}",
    )
    return tuple(checks)


def score_hypotheses(grid: GridResult, settings: config_mod.Config) -> dict[str, dict[str, object]]:
    """SPEC.md 1's H5 and H6 on the reference variant, with their registered legs."""
    out: dict[str, dict[str, object]] = {}
    ref = 4
    b, d = _cell(grid, ref, "B"), _cell(grid, ref, "D")
    c = _cell(grid, ref, "C")
    if d is not None and b is not None and d.inputs.stress is not None:
        rows = d.rows
        stress = rows["stress_volatility"].to_numpy(dtype=float)
        count = int(np.ceil(_H5_TAIL * len(stress)))
        order = np.argsort(-stress)
        stressed = np.zeros(len(stress), dtype=bool)
        stressed[order[:count]] = True
        # Leg 1: realised SPREAD cost of D's trades against the flat assumption on the same trades.
        spread_dollars = d.backtest.spread_cost.sum(axis=1).to_numpy(dtype=float)
        nav_pre = rows["nav_pre"].to_numpy(dtype=float)
        turnover = rows["turnover"].to_numpy(dtype=float)
        flat_half_spread = settings.model.optimizer.grid.flat_half_spread
        flat_dollars = flat_half_spread * turnover * nav_pre
        realised_bps = spread_dollars / nav_pre * 1e4
        flat_bps = flat_dollars / nav_pre * 1e4
        ratio_stressed = (
            float(realised_bps[stressed].sum() / flat_bps[stressed].sum())
            if flat_bps[stressed].sum() > 0
            else float("nan")
        )
        ratio_calm = (
            float(realised_bps[~stressed].sum() / flat_bps[~stressed].sum())
            if flat_bps[~stressed].sum() > 0
            else float("nan")
        )
        # Leg 2: what costs inside the objective recovered (D - B, net, per period) in stressed vs calm months.
        diff = d.monthly["net_return"].to_numpy(dtype=float) - b.monthly["net_return"].to_numpy(
            dtype=float
        )
        recovered_stressed = float(diff[stressed].mean())
        recovered_calm = float(diff[~stressed].mean())
        out["H5"] = {
            "stressed_rebalances": count,
            "of": len(stress),
            "ranking": "trailing 21-session realised volatility of the equal-weight book at the rebalance date",
            "leg1_realised_spread_over_flat_stressed": ratio_stressed,
            "leg1_realised_spread_over_flat_calm": ratio_calm,
            "leg1_registered": "ratio in [2, 4] in the stressed windows",
            "leg1_holds": bool(2.0 <= ratio_stressed <= 4.0),
            "leg2_d_minus_b_net_per_period_stressed": recovered_stressed,
            "leg2_d_minus_b_net_per_period_calm": recovered_calm,
            "leg2_registered": "D - B larger in the stressed windows than in calm ones; uncertainty NOT quantified at ~10 stressed months",
            "leg2_holds": bool(recovered_stressed > recovered_calm),
            "variant": ref,
        }
    # SPEC.md 9's expected headline, clause by clause (W6-P2b, operator). The
    # "cost term larger" clause is stated UNDER A TIME-VARYING SPREAD MODEL, which
    # is treatment D and nothing else.
    inside: dict[str, dict[str, float]] = {}
    under_d: dict[str, dict[str, float]] = {}
    for r in grid.results:
        if r.cell.regime != "patient":
            continue
        v = r.cell.variant
        if r.cell.treatment == "gross_then_net":
            inside.setdefault(v, {})["B"] = r.identity.cost_term
        if r.cell.treatment == "time_varying":
            inside.setdefault(v, {})["D"] = r.identity.cost_term
            under_d[v] = {
                "risk_model_term": r.identity.risk_model_term,
                "cost_term": r.identity.cost_term,
            }
    factor_variants = [v for v in under_d if v not in _DENSE]
    dense_variants = [v for v in under_d if v in _DENSE]
    cost_ahead = [v for v in under_d if under_d[v]["cost_term"] > under_d[v]["risk_model_term"]]
    eig = [c for c in grid.checks if c["leg"] == "e"]
    out["SPEC9_headline"] = {
        "clause_inside_the_objective": {
            "claim": "the cost term responds to putting costs inside the objective (D) rather than outside (B)",
            "cost_term_B_vs_D": inside,
            "holds": all(x["D"] < x["B"] for x in inside.values() if "B" in x and "D" in x),
        },
        "clause_eigenfactor_removes_most_of_it": {
            "claim": "the eigenfactor adjustment removes most of the risk-model term",
            "holds": False,
            "observed": eig[0]["observed"] if eig else "leg (e) not scored",
        },
        "clause_cost_larger_under_time_varying_spread": {
            "claim": "under a time-varying spread model (treatment D, nothing else) the cost term exceeds the risk-model term",
            "terms_under_D": under_d,
            "variants_with_cost_ahead": cost_ahead,
            "holds_at_factor_model_cells": bool(factor_variants)
            and all(v in cost_ahead for v in factor_variants),
            "holds_at_dense_cells": bool(dense_variants)
            and all(v in cost_ahead for v in dense_variants),
            "confound": "the dense variants forecast the assets' true excess returns while the pipeline variants' regressand differs for the four government zeros by carry-and-roll, so the dense-versus-factor gap in B is not the diagonal alone; and that gap is inside the per-cell chi-square interval",
        },
    }
    if b is not None and d is not None:
        d_cells = [
            r
            for r in grid.results
            if r.cell.treatment == "time_varying" and r.cell.regime == "patient"
        ]
        srs = {r.cell.variant: r.sharpe_net.annualised for r in d_cells}
        cov_range = max(srs.values()) - min(srs.values()) if len(srs) > 1 else float("nan")
        cost_gain = d.sharpe_net.annualised - b.sharpe_net.annualised
        out["H6"] = {
            "variant": ref,
            "net_sharpe_B": b.sharpe_net.annualised,
            "net_sharpe_C": c.sharpe_net.annualised if c else float("nan"),
            "net_sharpe_D": d.sharpe_net.annualised,
            "delta_cost_placement_D_minus_B": cost_gain,
            "delta_covariance_range_under_D": cov_range,
            "covariance_net_sharpes_under_D": srs,
            "registered": "costs inside beat gross-then-net by MORE than the range across covariance variants; predicted REFUTED",
            "holds": bool(cost_gain > cov_range),
        }
    return out


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _num(value: float, digits: int = 3) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def _pct(value: float, digits: int = 1) -> str:
    return "n/a" if not np.isfinite(value) else f"{100.0 * value:.{digits}f}%"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def _variant_label(variant: str, settings: config_mod.Config) -> str:
    grid = settings.model.optimizer.grid
    labels = {
        "sample_equal_weight": f"1 sample covariance, equal-weighted {grid.dense_window}d",
        "ewma": "2 EWMA, separated half-lives",
        "newey_west": "3 + Newey-West (with PSD repair)",
        "eigenfactor_a1.0": "4 + eigenfactor a = 1.0",
        "eigenfactor_a1.4": "5 + eigenfactor a = 1.4",
        "volatility_regime": "6 + volatility regime adjustment",
        "ledoit_wolf": f"7 Ledoit-Wolf constant-correlation, {grid.dense_window}d",
    }
    return labels[variant]


def render(grid: GridResult, settings: config_mod.Config) -> str:
    """``reports/experiment_grid.md``."""
    cfg = settings.model.optimizer.grid
    first = grid.results[0]
    dates = first.inputs.rebalance_dates
    lines: list[str] = [
        "# SPEC.md 9's experiment grid: seven covariance treatments by four cost treatments",
        "",
        "Generated by `python -m mafrm.backtest.grid`. W6-P2, under the rulings recorded at SPEC.md 9.1. Every cell is a `strategy-config` trial (experiments.md rows 211-238); `N` at run time was "
        f"**{grid.trials}**.",
        "",
        "## The reference point",
        "",
        "| Ingredient | Value |",
        "|---|---|",
        f"| Calendar | {len(dates)} month-end rebalances, {dates[0].date()} to {dates[-1].date()}, held through {first.inputs.terminal_date.date()}; nothing on or after `HOLDOUT_START` = {first.inputs.holdout_start.date()} (asserted in code) |",
        f"| Book size | ${grid.nav:,.0f}: the AUM at which the equal-weight trade in the thinnest proxy is {cfg.book_size.participation_of_adv:.0%} of its in-sample median ADV -- the cost model's lower calibration anchor (ruling, SPEC.md 9.1) |",
        f"| `TE_target` | {grid.te_target:.4%} per period ({cfg.tracking_error_multiple:g}x the equal-weight book's realised in-sample volatility) |",
        f"| Risk aversion | `{cfg.risk_aversion}`: SPEC.md 8.4's second step -- the `gamma_risk` term replaced by a hard forecast-volatility bound at `TE_target`, the multiplier recovered; `psi_mis` priced at the recovered multiplier in a second pass |",
        f"| Costs | `gamma_trade` = {cfg.gamma_trade:g}; treatment D at both `Y` regimes; issuer half-spread at the `{cfg.spread_end}` end of its interval -- **the interval FLOOR, which `config/spread_levels.yaml` itself says is not a value: SPY and IWM trade at a spread of exactly zero in every cell (SSGA's own document puts SPY near 0.6bp), so every cost figure in this grid is a LOWER BOUND on the spread leg**; the other end is W6-P3's band; ADV cap a hinge at priority {cfg.hinge_priorities.get('adv_participation', float('nan')):.4f} (W6-P1's p80) |",
        "| Objective normalisation | W6-P3 (SPEC.md 9.3 ruling 6, resolved): in constraint mode the objective is left in its own units (scale 1) rather than divided by the largest `\\|alpha\\|`; the affected cells are experiments.md rows 254-260 |",
        f"| Dense window | {grid.dense_window} days = `2 tau / ln 2` at the {cfg.horizon}-horizon factor volatility half-life; moments about zero |",
        f"| Sharpe | Lo (2002): `eta(q)` at q = {cfg.sharpe_aggregation_periods}, GMM standard error with a Bartlett kernel at q - 1 lags, `eta` held fixed; deflated Sharpe against `N` = {grid.trials} with `V[SR]` = {grid.trials_variance:.3e} across the 28 cells' per-period net Sharpes (the cells share one alpha and are not independent trials; the formula assumes they are) |",
        "",
        "## The answer table",
        "",
        "SPEC.md 1: `SR_paper - SR_real = (mu_g / sigma_f)(1 - 1/B) + TC / (B sigma_f)`, with `B = sigma_r / sigma_f` the ratio of annualised realised (net) to forecast (rms) volatility. **Every Sharpe-type column is on ONE scale, Lo's:** the gross and net Sharpes are `eta(12)` times their per-period values with the GMM standard error following, and the identity's volatilities are annualised by `12 / eta` of the net series so that `SR_paper` and `SR_real` sit on the same scale (on a cost-free cell `gross SR x B = SR_paper` exactly, asserted in code). `B` is scale-free. `B (6.1)` is SPEC.md 6.1's statistic on the same non-overlapping months with its exact chi-square interval; MRAD is the mean |B - 1| over NON-overlapping 12-month windows (overlap 0 of 12). DSR is the deflated Sharpe's probability.",
        "",
    ]
    header = [
        "cell",
        "covariance",
        "cost treatment",
        "Y",
        "SR_paper",
        "SR_real",
        "risk-model term",
        "cost term",
        "B",
        "B (6.1)",
        "interval",
        "MRAD",
        "min-var vol bps/d",
        "turnover /yr",
        "cost bp/yr",
        "gross SR (Lo SE)",
        "net SR (Lo SE)",
        "DSR",
    ]
    rows: list[list[str]] = []
    for r in grid.results:
        s = r.summary
        ident = r.identity
        rows.append(
            [
                r.cell.cell_id,
                _variant_label(r.cell.variant, settings),
                f"{_TREATMENT_LETTER[r.cell.treatment]} {r.cell.treatment}",
                r.cell.regime if r.cell.charged else "-",
                _num(ident.sharpe_paper),
                _num(ident.sharpe_real),
                _num(ident.risk_model_term, 4),
                _num(ident.cost_term, 4),
                _num(ident.bias_ratio),
                _num(s["bias"]),
                f"[{s['bias_lower']:.3f}, {s['bias_upper']:.3f}]",
                _num(s["mrad"]),
                _num(grid.min_var_realised_volatility.get(r.cell.variant, float("nan"))),
                _num(s["turnover_annual"], 2),
                _num(s["cost_drag_bps_per_year"], 1),
                f"{r.sharpe_gross.annualised:.3f} ({r.sharpe_gross.standard_error:.3f})",
                f"{r.sharpe_net.annualised:.3f} ({r.sharpe_net.standard_error:.3f})",
                _num(r.deflated.probability if r.deflated else float("nan")),
            ]
        )
    lines += _table(header, rows)
    lines += [
        "",
        "## What the optimizer did in each cell",
        "",
        "`TE bound` is the share of rebalances on which the hard forecast-volatility constraint bound; where it is slack the long-only simplex cannot reach `TE_target` at any risk aversion. `gamma_risk` is the value the recovered multiplier implies, `mu / (2 TE_target)`, beside SPEC.md 8.4's initialisation on the same date. `ADV hinge` is the share of rebalances on which the soft cap was exceeded.",
        "",
    ]
    header2 = [
        "cell",
        "Y",
        "solver rungs",
        "inaccurate",
        "relaxed",
        "fell back",
        "TE bound",
        "rms attainment",
        "gamma_risk recovered (median)",
        "gamma_risk SPEC 8.4 (median)",
        "IR median",
        "cos theta median",
        "eff. assets",
        "largest w median",
        "corner share",
        "ADV hinge",
        "spread share",
        "LW intensity median",
    ]
    rows2: list[list[str]] = []
    for r in grid.results:
        s = r.summary
        rows2.append(
            [
                r.cell.cell_id,
                r.cell.regime if r.cell.charged else "-",
                f"{int(s['solver_rungs'])}/{int(s['rebalances'])}",
                str(int(s["inaccurate"])),
                str(int(s["relaxed"])),
                str(int(s["fell_back"])),
                _pct(s.get(f"{_TE_LIMIT}_binding_share", float("nan"))),
                _num(s["te_attainment_rms"]),
                _num(s["gamma_risk_recovered_median"], 2),
                _num(s["gamma_risk_spec_median"], 2),
                _num(s["information_ratio_median"], 2),
                _num(s["cos_theta_median"]),
                _num(s["effective_assets_mean"], 2),
                _pct(s["largest_weight_median"]),
                _pct(s["corner_share"]),
                _pct(s.get(f"{_ADV_LIMIT}_binding_share", float("nan"))),
                _pct(s["spread_share_of_cost"]),
                _num(s["ledoit_wolf_intensity_median"]),
            ]
        )
    lines += _table(header2, rows2)
    lines += [
        "",
        "## The financing leg and the `optimal_inaccurate` re-solve (W6-P3, SPEC.md 9.3)",
        "",
        "**Financing (ruling 1):** cash earns and pays the Ken French daily bill rate both ways, no spread, in every cell. The book is marked on the excess-return index, in which frame cash at RF accrues `rf - rf = 0` by identity, so the engine carries cash at zero and the column is the NOMINAL size of the leg the identity nets out -- the cash path priced at RF, in bp of the initial NAV per year -- printed so that 'negligible' is a computed line. Under A the cash is zero. **Re-solve (ruling 6):** every rebalance the primary solver returned `optimal_inaccurate` on was re-solved with the fallback solver on the same problem; `max L1` is the LARGEST distance between the two weight vectors over the flagged dates, against the threshold "
        f"{cfg.resolve.l1_threshold:g} registered before any distance was measured. **Amended criterion (W6-P3b, SPEC.md 9.3):** two points are compared only when both are feasible to the solver's tolerance, and the feasible point with the better objective (higher: SPEC.md 8.1's problem is a maximisation) is the solution; the L1 to an infeasible or worse-objective fallback is the fallback's error, reported and not acted on. A cell owes a re-run only where the L1 is above the threshold AND the fallback's point is feasible AND better.",
        "",
    ]
    rows3: list[list[str]] = []
    for r in grid.results:
        s = r.summary
        rows3.append(
            [
                r.cell.cell_id,
                r.cell.regime if r.cell.charged else "-",
                _num(s["financing_bps_per_year"], 3),
                _pct(s["mean_cash_over_nav"], 4),
                _pct(s["final_cash_over_nav"], 4),
                str(int(s["inaccurate"])),
                str(int(s["resolve_count"])),
                f"{s['resolve_max_l1']:.2e}" if s["resolve_count"] > 0 else "-",
                _num(s["resolve_worst_te_ratio"], 4) if s["resolve_count"] > 0 else "-",
                (
                    "; ".join(
                        f"{v}: {int(c)}"
                        for v, c in r.rows["resolve_verdict"].value_counts().items()
                        if v
                    )
                    if s["resolve_count"] > 0
                    else "-"
                ),
                (
                    "**RE-RUN OWED**"
                    if s["resolve_actionable"] > 0
                    else (
                        "fallback's error, reported"
                        if s["resolve_exceeds"] > 0
                        else ("accepted" if s["resolve_count"] > 0 else "-")
                    )
                ),
            ]
        )
    lines += _table(
        [
            "cell",
            "Y",
            "financing leg at RF, bp/yr",
            "mean cash / NAV",
            "final cash / NAV",
            "inaccurate",
            "re-solved",
            "max L1",
            "re-solved point's worst sigma_f / TE_target",
            "amended criterion, per flagged date",
            "verdict",
        ],
        rows3,
    )
    lines += [
        "",
        "SPEC.md 10.1-10.3's diagnostics on every cell -- Perold at the parent level, the variance-additive tracking-error decomposition with x-sigma-rho contributions, and the attribution reconciled against the risk model -- are in `reports/diagnostics.md`.",
    ]
    lines += ["", "## Registered legs (experiments.md rows 211-238), scored", ""]
    lines += _table(
        ["leg", "registered", "verdict", "observed"],
        [
            [
                str(c["leg"]),
                str(c["claim"]),
                "HOLDS" if c["holds"] else "**REFUTED**",
                str(c["observed"]),
            ]
            for c in grid.checks
        ],
    )
    headline = grid.hypotheses.get("SPEC9_headline")
    if headline:
        lines += [
            "",
            "## SPEC.md 9's expected headline, clause by clause",
            "",
            "The headline's cost clause is stated *under a time-varying spread model*, which is treatment D and nothing else; treatments B and C are the two cost treatments SPEC.md 9 deprecates (B trades as if trading were free, C over-prices the spread against the issuer levels). All terms on Lo's scale.",
            "",
        ]
        rows_h: list[list[str]] = []
        for key, clause in headline.items():
            assert isinstance(clause, dict)
            if "holds_at_factor_model_cells" in clause:
                verdict = (
                    f"factor-model cells: {'HOLDS' if clause['holds_at_factor_model_cells'] else '**REFUTED**'}; "
                    f"dense cells: {'HOLDS' if clause['holds_at_dense_cells'] else '**REFUTED**'}"
                )
                observed = "cost ahead at " + (
                    ", ".join(clause["variants_with_cost_ahead"]) or "no variant"
                )
            else:
                verdict = "HOLDS" if clause["holds"] else "**REFUTED**"
                observed = str(clause.get("observed", ""))
                if "cost_term_B_vs_D" in clause:
                    pairs = clause["cost_term_B_vs_D"]
                    assert isinstance(pairs, dict)
                    observed = "; ".join(
                        f"{v} {x['B']:.4f} -> {x['D']:.4f}"
                        for v, x in pairs.items()
                        if "B" in x and "D" in x
                    )
            rows_h.append([key, str(clause["claim"]), verdict, observed])
        lines += _table(["clause", "claim", "verdict", "observed"], rows_h)
        terms = headline["clause_cost_larger_under_time_varying_spread"]
        assert isinstance(terms, dict)
        under_d = terms["terms_under_D"]
        assert isinstance(under_d, dict)
        lines += ["", "Under treatment D (patient `Y`), risk-model term against cost term:", ""]
        lines += _table(
            ["variant", "risk-model term", "cost term", "ahead"],
            [
                [
                    v,
                    f"{x['risk_model_term']:.4f}",
                    f"{x['cost_term']:.4f}",
                    "risk model" if x["risk_model_term"] > x["cost_term"] else "cost",
                ]
                for v, x in under_d.items()
            ],
        )
        lines += ["", f"Confound, named: {terms['confound']}", ""]
    lines += ["", "## H5 and H6 (SPEC.md 1), on the reference variant 4", ""]
    for name, payload in grid.hypotheses.items():
        if name == "SPEC9_headline":
            continue
        lines.append(f"### {name}")
        lines.append("")
        for key, value in payload.items():
            shown = f"{value:.5g}" if isinstance(value, float) else str(value)
            lines.append(f"- `{key}`: {shown}")
        lines.append("")
    lines += [
        "## Reading notes",
        "",
        "- Treatments differ in what the optimizer SEES; every cell but A is CHARGED the true SPEC.md 7.1 cost on realisation, so the identity's `TC` is the realised cost of the book each treatment produced. Treatment C keeps the square-root impact term and swaps the spread model, so C against D isolates the spread model (H5's question).",
        "- The two dense variants enter the optimizer as `X = I`, so SPEC.md 8.3's `cos theta` is 1 by construction and the misalignment penalty is zero: a full-rank model sees every direction.",
        "- The two dense variants are estimated on the assets' TRUE daily excess returns (complete cases from the 2007 sample start), because the residual panel the pipeline variants forecast starts in April 2009 and holds fifteen rows before the first rebalance. They therefore forecast exactly the series the book is marked on, while the pipeline variants' regressand differs from it for the four government zeros by the carry-and-roll leg (SPEC.md 4.3). The realised return in the min-var volatility column is the panel's regressand for every variant.",
        "- The four government zeros are marked on their true excess returns while their forecast was fitted to the duration leg (SPEC.md 4.3), a level difference of a few basis points a month carried in every cell alike.",
        "- Failure mode 9: the 12-month MRAD uses NON-overlapping windows; nothing here quotes a count of rolling windows as an `n`.",
        "",
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class ChartData:
    """What the figure needs, buildable from a :class:`GridResult` or from ``results/metrics.json``."""

    #: ``(variant, treatment) -> (risk-model term, cost term)`` at the patient regime.
    terms: Mapping[tuple[str, str], tuple[float, float]]
    rebalances: int
    nav: float
    trials: int

    @classmethod
    def from_grid(cls, grid: GridResult) -> ChartData:
        return cls(
            terms={
                (r.cell.variant, r.cell.treatment): (
                    r.identity.risk_model_term,
                    r.identity.cost_term,
                )
                for r in grid.results
                if r.cell.regime == "patient"
            },
            rebalances=len(grid.results[0].inputs.rebalance_dates),
            nav=grid.nav,
            trials=grid.trials,
        )

    @classmethod
    def from_metrics(cls, path: Path) -> ChartData:
        payload = json.loads(path.read_text(encoding="utf-8"))
        terms = {
            (str(c["variant"]), str(c["treatment"])): (
                float(c["identity"]["risk_model_term"]),
                float(c["identity"]["cost_term"]),
            )
            for c in payload["cells"]
            if c["regime"] in (None, "patient")
        }
        return cls(
            terms=terms,
            rebalances=int(payload["reference"]["rebalances"]),
            nav=float(payload["reference"]["nav_dollars"]),
            trials=int(payload["trial_count_N"]),
        )


def chart(data: ChartData, settings: config_mod.Config, path: Path) -> None:
    """The two identity terms by covariance variant, one bar per cost treatment."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink_2, gridline = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
    # The validated categorical order (dataviz palette, light mode), one slot per treatment.
    slots = {
        "none": "#2a78d6",
        "gross_then_net": "#eb6834",
        "flat_spread": "#1baf7a",
        "time_varying": "#eda100",
    }
    cfg = settings.model.optimizer.grid
    variants = list(cfg.covariance_variants)
    treatments = list(cfg.cost_treatments)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
    fig.patch.set_facecolor(surface)
    y = np.arange(len(variants))
    height = 0.8 / len(treatments)
    for ax, column, title in zip(
        axes,
        (0, 1),
        ("risk-model term  (mu_g / sigma_f)(1 - 1/B)", "cost term  TC / (B sigma_f)"),
        strict=True,
    ):
        for j, treatment in enumerate(treatments):
            values = [
                data.terms[(v, treatment)][column] if (v, treatment) in data.terms else np.nan
                for v in variants
            ]
            ax.barh(
                y - 0.4 + height * (j + 0.5),
                values,
                height=height * 0.9,
                color=slots[treatment],
                edgecolor=surface,
                linewidth=1.0,
                label=f"{_TREATMENT_LETTER[treatment]} {treatment}",
            )
        ax.set_title(title, color=ink, fontsize=11, loc="left", pad=28)
        ax.set_facecolor(surface)
        ax.grid(axis="x", color=gridline, linewidth=1.0)
        ax.axvline(0.0, color=ink_2, linewidth=1.0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(gridline)
        ax.tick_params(colors=ink_2, labelsize=9)
        ax.set_xlabel("annualised Sharpe units", color=ink_2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [_variant_label(v, settings) for v in variants], color=ink_2, fontsize=9
    )
    axes[0].invert_yaxis()
    axes[0].legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=4,
        frameon=False,
        fontsize=9,
        labelcolor=ink_2,
    )
    fig.suptitle(
        "SPEC.md 1's identity across SPEC.md 9's grid: the paper-versus-real Sharpe gap, split",
        color=ink,
        fontsize=12,
        x=0.02,
        y=0.985,
        ha="left",
    )
    fig.text(
        0.02,
        0.945,
        f"W6-P2. Model A, {data.rebalances} monthly rebalances in-sample, TE-constrained long-only book at ${data.nav / 1e6:.1f}M; treatment D at patient Y. N = {data.trials}.",
        color=ink_2,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)


def metrics_payload(grid: GridResult, settings: config_mod.Config) -> dict[str, object]:
    """``results/metrics.json``: every headline number, so a change shows in a diff."""
    cfg = settings.model.optimizer.grid
    first = grid.results[0]

    def clean(value: object) -> object:
        if isinstance(value, float):
            return None if not np.isfinite(value) else float(f"{value:.10g}")
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [clean(v) for v in value]
        if isinstance(value, np.bool_ | bool):
            return bool(value)
        if isinstance(value, np.integer | int):
            return int(value)
        return value

    cells_out = []
    for r in grid.results:
        ident = r.identity
        entry: dict[str, object] = {
            "cell": r.cell.cell_id,
            "variant": r.cell.variant,
            "treatment": r.cell.treatment,
            "regime": r.cell.regime if r.cell.charged else None,
            "identity": {
                "gross_return_annual": ident.gross_return,
                "cost_drag_annual": ident.cost_drag,
                "forecast_volatility_annual": ident.forecast_volatility,
                "realised_volatility_annual": ident.realised_volatility,
                "bias_ratio": ident.bias_ratio,
                "aggregation_eta": ident.aggregation,
                "sharpe_paper": ident.sharpe_paper,
                "sharpe_real": ident.sharpe_real,
                "risk_model_term": ident.risk_model_term,
                "cost_term": ident.cost_term,
            },
            "sharpe_net": {
                "per_period": r.sharpe_net.per_period,
                "annualised": r.sharpe_net.annualised,
                "standard_error": r.sharpe_net.standard_error,
                "eta": r.sharpe_net.eta,
            },
            "sharpe_gross": {
                "per_period": r.sharpe_gross.per_period,
                "annualised": r.sharpe_gross.annualised,
                "standard_error": r.sharpe_gross.standard_error,
                "eta": r.sharpe_gross.eta,
            },
            "deflated_sharpe": (
                {
                    "probability": r.deflated.probability,
                    "expected_maximum_per_period": r.deflated.expected_maximum,
                    "skewness": r.deflated.skewness,
                    "kurtosis": r.deflated.kurtosis,
                }
                if r.deflated
                else None
            ),
            "min_var_realised_volatility_bps_per_day": grid.min_var_realised_volatility.get(
                r.cell.variant
            ),
            "summary": dict(r.summary),
            "financing": {
                "convention": settings.model.backtest.financing.convention,
                "leg_bps_per_year_nominal": r.summary["financing_bps_per_year"],
                "mean_cash_over_nav": r.summary["mean_cash_over_nav"],
                "final_cash_over_nav": r.summary["final_cash_over_nav"],
            },
            "resolve": {
                "flagged": int(r.summary["resolve_count"]),
                "max_l1": r.summary["resolve_max_l1"],
                "resolved_point_worst_te_ratio": r.summary["resolve_worst_te_ratio"],
                "threshold": cfg.resolve.l1_threshold,
                "exceeds": bool(r.summary["resolve_exceeds"] > 0),
                "actionable_under_amended_criterion": int(r.summary["resolve_actionable"]),
                "verdicts": {
                    str(v): int(c) for v, c in r.rows["resolve_verdict"].value_counts().items() if v
                },
            },
            "diagnostics": (
                {
                    "reconciliation": {
                        k: {
                            "bias": c.bias,
                            "lower": c.lower,
                            "upper": c.upper,
                            "inside": c.inside,
                            "defined": c.defined,
                        }
                        for k, c in r.diagnostics.reconciliation.items()
                    },
                    "reconciliation_flag": r.diagnostics.reconciliation_flag,
                }
                if r.diagnostics is not None
                else None
            ),
        }
        cells_out.append(entry)
    payload: dict[str, object] = {
        "task": "W6-P3",
        "spec": "SPEC.md 9 (re-run under 9.3's rulings) and SPEC.md 10.1-10.3",
        "holdout_start": str(first.inputs.holdout_start.date()),
        "trial_count_N": grid.trials,
        "cells_evaluated": len([r for r in grid.results if r.cell.regime == "patient"]),
        "runs": len(grid.results),
        "trials_variance_per_period": grid.trials_variance,
        "reference": {
            "nav_dollars": grid.nav,
            "te_target_per_period": grid.te_target,
            "rebalances": len(first.inputs.rebalance_dates),
            "first_rebalance": str(first.inputs.rebalance_dates[0].date()),
            "last_rebalance": str(first.inputs.rebalance_dates[-1].date()),
            "terminal_date": str(first.inputs.terminal_date.date()),
            "dense_window": grid.dense_window,
            "spread_end": cfg.spread_end,
            "gamma_trade": cfg.gamma_trade,
            "risk_aversion": cfg.risk_aversion,
            "hinge_priorities": dict(cfg.hinge_priorities),
            "flat_half_spread": cfg.flat_half_spread,
        },
        "registered_checks": list(grid.checks),
        "hypotheses": grid.hypotheses,
        "resolve": {
            "criterion": cfg.resolve.criterion,
            "threshold": cfg.resolve.l1_threshold,
            "cells_flagged": [
                r.cell.key + "/" + r.cell.regime
                for r in grid.results
                if r.summary["resolve_count"] > 0
            ],
            "max_l1_over_grid": max(r.summary["resolve_max_l1"] for r in grid.results),
            "cells_above_threshold": [
                r.cell.key + "/" + r.cell.regime
                for r in grid.results
                if r.summary["resolve_exceeds"] > 0
            ],
            "cells_owing_a_rerun_amended_criterion": [
                r.cell.key + "/" + r.cell.regime
                for r in grid.results
                if r.summary["resolve_actionable"] > 0
            ],
        },
        "financing": {
            "convention": settings.model.backtest.financing.convention,
            "largest_abs_leg_bps_per_year": max(
                abs(r.summary["financing_bps_per_year"]) for r in grid.results
            ),
        },
        "attribution_reconciliation": {
            "cells_marked": [
                r.cell.key + "/" + r.cell.regime
                for r in grid.results
                if r.diagnostics is not None and r.diagnostics.reconciliation_flag
            ],
        },
        "cells": cells_out,
    }
    return clean(payload)  # type: ignore[return-value]


def diagnostics_entries(grid: GridResult) -> tuple[diagnostics_mod.DiagnosticsEntry, ...]:
    return tuple(
        diagnostics_mod.DiagnosticsEntry(
            cell_id=r.cell.key,
            label=r.cell.label,
            treatment=r.cell.treatment,
            regime=r.cell.regime if r.cell.charged else "-",
            diagnostics=r.diagnostics,
        )
        for r in grid.results
        if r.diagnostics is not None
    )


def write_metrics(payload: dict[str, object], path: Path = _METRICS_PATH) -> None:
    """Write ``results/metrics.json``, carrying over the sections other modules own."""
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in _PRESERVED_SECTIONS:
            if key in existing and key not in payload:
                payload[key] = existing[key]
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# A synthetic cell, for the golden-weights fixture and the offline tests
# ---------------------------------------------------------------------------


def synthetic_inputs(
    settings: config_mod.Config,
    *,
    cell: CellSpec,
    seed: int,
    assets: int = 3,
    rebalances: int = 14,
) -> CellInputs:
    """A small self-contained :class:`CellInputs`: seeded prices, one factor, flat costs.

    Everything :func:`run_cell` needs and nothing from the cache, so the fixture
    ``tests/fixtures/golden_weights.json`` pins the runner's output offline:
    ``python -m mafrm.backtest.grid --golden > tests/fixtures/golden_weights.json``
    regenerates it, deliberately. The dates sit in 2015, well inside the sample;
    fourteen months, so Lo's ``q - 1 = 11`` lags fit inside the series.
    """
    rng = np.random.default_rng(seed)
    names = tuple(f"asset_{i}" for i in range(assets))
    horizon = settings.model.optimizer.alpha.horizon_days
    days = pd.bdate_range("2015-01-02", periods=(rebalances + 1) * horizon, name="date")
    daily = rng.standard_normal((len(days), assets)) * 0.01 + 0.0002
    prices = pd.DataFrame(np.cumprod(1.0 + daily, axis=0), index=days, columns=list(names))
    month_ends = pd.DatetimeIndex(
        pd.Series(days, index=days).groupby(days.to_period("M")).last().to_numpy()
    )
    rebalance_dates = pd.DatetimeIndex(month_ends[:rebalances], name="date")
    terminal = pd.Timestamp(days[-1])
    # Exposures 0.5 .. 1.5 on one factor: asset volatilities from ~1.4% to ~3.2% per period
    # around a 2% target, so the TE bound binds on some rebalances and is slack on others.
    exposures = np.linspace(0.5, 1.5, assets)[:, None]
    factor = np.array([[4.0e-4]])
    specific = np.full(assets, 1.0e-4)
    forecasts = tuple(
        Forecast(
            exposures=exposures,
            factor_covariance=factor * (1.0 + 0.1 * np.sin(i)),
            specific_variance=specific,
        )
        for i in range(rebalances)
    )
    alphas = rng.standard_normal((rebalances, assets)) * 0.005
    alphas = alphas - alphas.mean(axis=1, keepdims=True)
    costs = settings.model.costs
    columns = pd.Index(names)
    model = engine.CostModel(
        half_spread=pd.DataFrame(5.0e-4, index=rebalance_dates, columns=columns),
        daily_volatility=pd.DataFrame(0.01, index=rebalance_dates, columns=columns),
        adv_dollars=pd.DataFrame(1.0e7, index=rebalance_dates, columns=columns),
        prefactor=float(getattr(costs.square_root_prefactor, cell.regime)),
        exponent=costs.total_cost_exponent,
        asymmetry=costs.buy_sell_asymmetry,
    )
    free = engine.CostModel.free(index=rebalance_dates, columns=columns, exponent=model.exponent)
    return CellInputs(
        cell=cell,
        assets=names,
        rebalance_dates=rebalance_dates,
        terminal_date=terminal,
        holdout_start=pd.Timestamp(settings.require_holdout_start()),
        prices=prices,
        forecasts=forecasts,
        alphas=alphas,
        nav=1.0e6,
        te_target=0.02,
        periods_per_year=settings.model.data.trading_days_per_year / horizon,
        optimizer_cost=model if cell.optimizer_sees_cost else None,
        realised_cost=model if cell.charged else free,
    )


def golden_payload(settings: config_mod.Config, *, seed: int) -> dict[str, object]:
    """The fixture: weights and forecast volatilities of two synthetic cells."""
    out: dict[str, object] = {"seed": seed, "cells": {}}
    cells_out: dict[str, object] = {}
    for spec in (CellSpec("eigenfactor_a1.0", "time_varying"), CellSpec("ewma", "none")):
        inputs = synthetic_inputs(settings, cell=spec, seed=seed)
        result = run_cell(inputs, settings)
        cells_out[spec.label] = {
            "weights": [[float(f"{w:.12g}") for w in row] for row in result.targets.to_numpy()],
            "forecast_volatility": [
                float(f"{v:.12g}") for v in result.rows["forecast_volatility"].to_numpy(dtype=float)
            ],
            "te_target": inputs.te_target,
        }
    out["cells"] = cells_out
    return out


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - entry point
    parser = argparse.ArgumentParser(description="SPEC.md 9's experiment grid (W6-P2)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--chart-only",
        action="store_true",
        help="redraw reports/experiment_grid.png from the committed results/metrics.json; no solve",
    )
    parser.add_argument(
        "--golden",
        action="store_true",
        help="print the synthetic golden-weights fixture as JSON and exit; reads no data",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    settings = config_mod.load()
    if args.golden:
        print(json.dumps(golden_payload(settings, seed=settings.model.seed), indent=2))
        return 0
    if args.chart_only:
        chart(ChartData.from_metrics(_METRICS_PATH), settings, _CHART_PATH)
        print(f"redrew {_CHART_PATH} from {_METRICS_PATH}")
        return 0
    universe = verification.build_universe(settings)
    grid = run_grid(universe, settings, progress=not args.quiet)
    _RESULTS.mkdir(parents=True, exist_ok=True)
    pd.concat([r.rows for r in grid.results]).to_csv(_CSV_PATH, float_format="%.10g")
    _MARKDOWN_PATH.write_text(render(grid, settings), encoding="utf-8")
    write_metrics(metrics_payload(grid, settings))
    entries = diagnostics_entries(grid)
    _DIAGNOSTICS_PATH.write_text(
        diagnostics_mod.render(
            entries, level=settings.model.validation.chi_square_level, trials=grid.trials
        ),
        encoding="utf-8",
    )
    reference = (
        grid.by_id("4D") if any(r.cell.cell_id == "4D" for r in grid.results) else grid.results[0]
    )
    if reference.diagnostics is not None and reference.diagnostics.attribution is not None:
        per_period = reference.diagnostics.attribution.periods.join(
            reference.diagnostics.shortfall.reindex(
                reference.diagnostics.attribution.periods["start"]
            ).set_index(reference.diagnostics.attribution.periods.index)[
                ["impact", "fees", "shortfall", "predicted", "realised", "residual"]
            ]
        )
        per_period.to_csv(_DIAGNOSTICS_CSV_PATH, float_format="%.10g")
    chart(ChartData.from_grid(grid), settings, _CHART_PATH)
    for r in grid.results:
        s = r.summary
        print(
            f"  {r.cell.cell_id} {r.cell.label}: rung<=1 {int(s['solver_rungs'])}/{int(s['rebalances'])}, "
            f"TE bound {s.get('tracking_error_binding_share', float('nan')):.0%}, B {r.identity.bias_ratio:.3f}, "
            f"risk {r.identity.risk_model_term:+.4f}, cost {r.identity.cost_term:+.4f}, "
            f"net SR {r.sharpe_net.annualised:.3f}, turnover {s['turnover_annual']:.2f}, "
            f"cost {s['cost_drag_bps_per_year']:.1f} bp/yr"
        )
    for c in grid.checks:
        print(f"  leg {c['leg']}: {'HOLDS' if c['holds'] else 'REFUTED'} -- {c['observed']}")
    for r in grid.results:
        s = r.summary
        if s["resolve_count"] > 0 or not np.isnan(s["financing_bps_per_year"]):
            print(
                f"  W6-P3 {r.cell.key}/{r.cell.regime}: financing {s['financing_bps_per_year']:+.4f} bp/yr, "
                f"inaccurate {int(s['inaccurate'])}, re-solved {int(s['resolve_count'])}, "
                f"max L1 {s['resolve_max_l1']:.2e}, "
                f"{'RE-RUN OWED' if s['resolve_actionable'] > 0 else ('fallback error' if s['resolve_exceeds'] > 0 else 'accepted')}; "
                f"attribution B factor {s.get('attribution_bias_factor', float('nan')):.3f} "
                f"specific {s.get('attribution_bias_specific', float('nan')):.3f} "
                f"total {s.get('attribution_bias_total', float('nan')):.3f}"
                f"{' MARKED' if s.get('attribution_reconciliation_flag', 0.0) > 0 else ''}"
            )
    for r in grid.results:
        if not r.cell.charged:
            print(
                f"  scale check {r.cell.cell_id}: gross SR {r.sharpe_gross.annualised:.6f} x B "
                f"{r.identity.bias_ratio:.6f} = {r.sharpe_gross.annualised * r.identity.bias_ratio:.6f}; "
                f"SR_paper {r.identity.sharpe_paper:.6f}"
            )
    print(f"N = {grid.trials}; wrote {_METRICS_PATH}, {_MARKDOWN_PATH}, {_CSV_PATH}, {_CHART_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
