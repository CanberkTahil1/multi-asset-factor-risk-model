"""``reports/bias_statistics.md`` and ``reports/bias_statistics.png``. SPEC.md 6.1, 6.2.

WHY THIS LIVES IN ``factors/`` AND NOT IN ``risk/``
---------------------------------------------------

The same reason as :mod:`mafrm.factors.vra_report` and
:mod:`mafrm.factors.specific_report`. The statistic and the four portfolio
families live in :mod:`mafrm.risk.validation`, which knows nothing about what its
columns are; *this* module reads Model A's residual panel, knows which two assets
are exact linear combinations of the factor set, and knows that a rolling beta at
date ``t`` was fitted on a window ending at ``t``. None of that may cross into
``risk/`` (CLAUDE.md invariant 10).

WHAT IS CACHED, AND WHY IT IS ONLY THAT
----------------------------------------

One thing in the whole battery costs real time: SPEC.md 5.3's Monte Carlo, at
2,000 trials per date. Everything else -- the EWMA, Newey-West and repair stages,
SPEC.md 5.5's specific risk, the portfolio construction, the statistic itself --
runs over the full panel in under a minute. So the committed cache holds the
**eigenfactor-adjusted factor covariance at both published ``a``**, at every
date and horizon, and nothing else; the three cheap stages are rebuilt on every
render from the same expanding-window pipeline. Rebuilding the cache is
``python -m mafrm.factors.bias_report --rebuild`` and takes about a quarter of an
hour fanned out across the two horizons.

Both published ``a`` come out of **one** simulation per date:
``eigenfactor_adjustment`` computes the bias curve once and derives every
``a`` from it, and :attr:`~mafrm.risk.covariance.CovarianceBuild.eigenfactor`
hands that object out. Reading both variants off it is the production accessor,
not a second application of the stage.

THE THREE PLACES A LOOK-AHEAD COULD ENTER, AND WHERE EACH IS CLOSED
-------------------------------------------------------------------

1. **The factor covariance.** Built from ``factors.iloc[:p]`` for the date at
   position ``p`` -- strictly before. :func:`mafrm.risk.regime.forecast_history`
   makes the same guarantee and asserts it; this module runs its own loop
   because it needs every stage's matrix rather than one, so it asserts the
   same thing itself.
2. **The specific variances.** ``specific_forecast_history`` slices the same way.
3. **The exposures, which is the one that is easy to miss.** ``betas`` fits a
   rolling regression on a window *ending at* ``t``, so ``X_t`` has already seen
   the return at ``t``. Using it would standardise a return by a forecast built
   partly from that return. **This module lags the exposure panel by one row**
   and the lag is the reason ``exposures[p - 1]`` appears rather than
   ``exposures[p]`` at every construction site.

``tests/test_validation.py`` perturbs one realised return and requires that no
forecast anywhere moves.

WHY THE HEADLINE IS THE PRE-VRA NUMBER
---------------------------------------

SPEC.md 6.2.2, ruled in this session. SPEC.md 5.4 sets
``lambda_F^2 = sum_t w_t (B_t^F)^2`` -- it is fitted to realised data expressly to
drive a bias statistic to one -- so a post-VRA ``B ~ 1`` is substantially what the
stage was built to produce rather than evidence the risk model is calibrated. It
is not perfectly circular, because ``lambda_F^2`` is a lagged EWMA and the
out-of-sample ``B`` is not identically one; it is close enough that the project's
central claim cannot rest on it. Both are reported, and the report says which
question each answers.
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from mafrm import config, history
from mafrm.factors import hybrid
from mafrm.risk import shepard, specific, validation
from mafrm.risk.config import HORIZONS, Horizon, RiskConfig
from mafrm.risk.covariance import (
    CovarianceBuild,
    correlation_from_covariance,
    ewma_second_moment,
    run_pipeline,
)
from mafrm.risk.regime import cross_sectional_bias, rolling_multiplier

__all__ = [
    "DiagonalControl",
    "ForecastPanel",
    "HorizonColumn",
    "Inputs",
    "VariantSpec",
    "build_history",
    "build_partial",
    "forecast_panel",
    "inputs",
    "main",
    "read_cache",
    "render",
    "residual_correlation_control",
    "restrict_window",
    "run",
    "run_for",
    "scaling_tag",
    "shepard_for",
    "shepard_table",
    "variants",
]

_REPORTS: Final[Path] = Path(__file__).resolve().parents[3] / "reports"
_CACHE_PATH: Final[Path] = _REPORTS / "bias_forecast_history.csv"
#: Per-horizon partials. Gitignored (``data/processed/``) because they are build
#: intermediates, exactly as ``vra_report.PARTIALS`` is: everything a reader
#: needs is in the committed cache, which is reproducible from them and from
#: ``model.seed``.
_PARTIALS: Final[Path] = Path(__file__).resolve().parents[3] / "data" / "processed" / "bias"
_MARKDOWN_PATH: Final[Path] = _REPORTS / "bias_statistics.md"
_CHART_PATH: Final[Path] = _REPORTS / "bias_statistics.png"

_REBUILD_HINT: Final[str] = "python -m mafrm.factors.bias_report --rebuild"


#: The two published eigenfactor scalings, as they appear in cached column
#: names. Derived from ``config`` at use; this is only the formatting.
def scaling_tag(scaling: float) -> str:
    """``a1`` / ``a1.4``: the cached column tag for a published eigenfactor ``a``.

    Public since W7-P4 so that :mod:`mafrm.factors.equity_risk` names its cache
    columns with the same function rather than a second f-string.
    """
    return f"a{scaling:g}"


_tag = scaling_tag


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Inputs:
    """Model A's panel, dressed for SPEC.md 6.2's families."""

    panel: hybrid.ResidualPanel
    #: ``T x N`` realised asset excess returns, basis points per day.
    returns: pd.DataFrame
    #: ``T x K`` realised factor returns, in the pipeline's regression units.
    factors: pd.DataFrame
    #: ``T x N x K``. Row ``p`` is the design fitted on the window ending at
    #: ``dates[p]`` -- so a forecast for ``dates[p]`` must use row ``p - 1``.
    exposures: np.ndarray
    buckets: pd.Series
    #: The two identity assets. The W4-P2 forward constraint as amended in
    #: W4-P1b; see :func:`inputs`.
    excluded: tuple[str, ...]

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.returns.index)

    @property
    def assets(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.returns.columns)

    @property
    def factor_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.factors.columns)


def inputs(settings: config.Config | None = None) -> Inputs:
    """Model A's residuals, realised returns, factors and the ``(date, asset)`` design.

    THE PANEL IS THE HYBRID'S, WHICH IS THE PANEL W4-P1 BUILT SPECIFIC RISK ON
        ``hybrid.residual_panel`` puts the four government zeros on their
        duration-effect leg (W2-P3's control) and every other asset is regressed
        exactly as Model A regresses it. Using it here is not a preference: the
        exposures, the residuals, the specific volatilities and the realised
        returns must all describe the **same** regressand, and W4-P1's specific
        risk is already fitted to this one. Scoring a forecast built from these
        residuals against a differently-constructed return would be a
        mismatch no assertion in the pipeline could see.

    THE EXCLUSION IS THE W4-P2 FORWARD CONSTRAINT, AT ITS AMENDED WIDTH
        ``hy_credit`` and ``commodity`` are exact linear combinations of the
        factor set (``R^2 = 1.000``), so their specific volatility is a
        construction artefact rather than an estimate. W4-P1b widened the
        constraint from "exclude them from bias aggregation" to "exclude them
        from every statistic that pools across assets", and both halves are
        applied here: they are out of SPEC.md 5.5(d)'s shrinkage target, out of
        the specific VRA's cross-section, and out of SPEC.md 6.1's family-1
        aggregate. **They stay in the model** -- they are forecast, they are in
        the asset covariance every family-2 and family-4 portfolio is built from,
        and they are reported individually and labelled.
    """
    loaded = settings or config.load()
    panel = hybrid.residual_panel(config=loaded)
    dates = pd.DatetimeIndex(panel.residuals.index)
    assets = [str(name) for name in panel.residuals.columns]
    factor_names = list(panel.factor_names)
    block = panel.exposures[factor_names]
    design = (
        block.reset_index().pivot(index="date", columns="asset", values=factor_names).reindex(dates)
    )
    stacked = np.empty((len(dates), len(assets), len(factor_names)), dtype=float)
    for column, name in enumerate(factor_names):
        stacked[:, :, column] = design[name][assets].to_numpy(dtype=float)
    if not np.all(np.isfinite(stacked)):
        raise ValueError(
            "bias_report.inputs: the exposure panel has holes on the residual panel's own "
            "index. The residual panel is complete-cases by construction, so this is a "
            "defect rather than a raggedness to fill."
        )
    sleeves = {asset.id: asset.sleeve for asset in loaded.universe.assets}
    buckets = pd.Series([sleeves[name] for name in assets], index=assets)
    scheme = loaded.model.factors.macro
    spans = {scheme.credit.asset_id, scheme.commodity.asset_id}
    return Inputs(
        panel=panel,
        returns=panel.returns,
        factors=panel.factors,
        exposures=stacked,
        buckets=buckets,
        excluded=tuple(sorted(name for name in assets if name in spans)),
    )


def minimum_observations(risk: RiskConfig, *, factors: int) -> int:
    """``max(K + 1, lags + 1)``. SPEC.md 5.4.1 ruling 3's ARITHMETIC floor.

    Not a burn-in. Below ``lags + 1`` SPEC.md 5.2's Bartlett sum has no pairs at
    its longest lag and below ``K + 1`` SPEC.md 5.3's Monte Carlo re-estimates a
    singular correlation; both are points where a stage has no answer at all
    rather than a poor one. The burn-in proper is :class:`ScoredWindow`, which is
    registered separately and is a bound on ``K/T_eff`` rather than a day count.
    """
    return max(factors + 1, risk.volatility_newey_west_lags + 1)


# ---------------------------------------------------------------------------
# The cached half -- SPEC.md 5.3's Monte Carlo, and nothing else
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HorizonColumn:
    """One cached partial: one horizon, carrying both published ``a``.

    The fan-out unit is the **horizon** and not the ``(horizon, a)`` pair,
    because one simulation per date produces every ``a``
    (:func:`mafrm.risk.eigenfactor.eigenfactor_adjustment` derives them from a
    single bias curve). Fanning out per ``a`` would run the Monte Carlo twice to
    get numbers the first run already computed.
    """

    horizon: Horizon

    @property
    def name(self) -> str:
        return self.horizon


def _upper_names(prefix: str, size: int) -> list[str]:
    return [f"{prefix}_{row}{column}" for row in range(size) for column in range(row, size)]


def _pack(matrix: np.ndarray) -> np.ndarray:
    size = matrix.shape[0]
    return np.asarray(
        [matrix[row, column] for row in range(size) for column in range(row, size)], dtype=float
    )


def _unpack(values: np.ndarray, size: int) -> np.ndarray:
    matrix = np.empty((size, size), dtype=float)
    position = 0
    for row in range(size):
        for column in range(row, size):
            matrix[row, column] = matrix[column, row] = values[position]
            position += 1
    return matrix


def build_partial(
    column: HorizonColumn, data: Inputs, settings: config.Config, *, progress: bool = True
) -> pd.DataFrame:
    """One horizon's history, at the shipped half-lives. Delegates to :func:`build_history`."""
    return build_history(
        RiskConfig.load(horizon=column.horizon, config=settings),
        data,
        label=column.name,
        progress=progress,
    )


def build_history(
    risk: RiskConfig, data: Inputs, *, label: str, progress: bool = True
) -> pd.DataFrame:
    """Run the full pre-VRA pipeline at every date and keep stage 4's output.

    TAKES A ``RiskConfig`` RATHER THAN A HORIZON, WHICH IS WHY IT IS SEPARATE
        W4-P2b's half-life sweep (SPEC.md 5.1.3) needs this exact loop at nine
        volatility half-lives that are not either horizon's shipped value, and
        :func:`build_partial` needs it at the two that are. Extracted at the
        second caller rather than copied, because the two must produce
        byte-identical histories wherever their configs agree -- SPEC.md 5.1.3's
        reproduction check of W4-P2's published ``B`` is exactly that assertion,
        and it would be checking nothing if the two loops were separate code.

    One :func:`~mafrm.risk.covariance.run_pipeline` call per date, stopping after
    the eigenfactor stage, with **both** published ``a`` read off the single
    :class:`~mafrm.risk.eigenfactor.EigenfactorAdjustment` it produces.

    The lead-lag is asserted here rather than assumed, in the same shape and for
    the same reason as :func:`mafrm.risk.regime.forecast_history`: the build for
    the date at position ``p`` must report exactly ``p`` observations.

    IT STARTS AT :func:`first_clean_position` AND NOT AT THE ARITHMETIC FLOOR
        Below that point SPEC.md 5.2's repair has floored eigenvalues to
        ``psd_eigenvalue_floor``, the correlation SPEC.md 5.3 diagonalises is
        numerically singular, and the Monte Carlo's Cholesky of it **raises**.
        That is the stage having no answer rather than a poor one, so the history
        begins where it has one. A cheap pass over the whole panel establishes
        the boundary first, because "where did the repair last fire" is not
        knowable from inside a forward loop.
    """
    scalings = risk.eigenfactor_scaling
    factors = data.factors
    size = factors.shape[1]
    periods = factors.shape[0]
    stages = stage_history(risk, data, start=minimum_observations(risk, factors=size))
    start = first_clean_position(stages)

    names = [name for scaling in scalings for name in _upper_names(_tag(scaling), size)]
    rows: list[np.ndarray] = []
    dates: list[pd.Timestamp] = []
    for position in range(start, periods):
        build = run_pipeline(
            factors.iloc[:position], risk.for_scaling(scalings[0]), stop_after="eigenfactor"
        )
        if build.observations != position:
            raise ValueError(
                f"bias_report.build_history: the build for {factors.index[position]!s} saw "
                f"{build.observations} observations at position {position}. The forecast for "
                "date t must be made from data strictly before t; this is the look-ahead "
                "check and it has failed."
            )
        adjustment = build.eigenfactor
        if adjustment is None:  # pragma: no cover - stop_after guarantees it ran
            raise ValueError("bias_report.build_history: the eigenfactor stage did not run")
        rows.append(np.concatenate([_pack(adjustment.variant(a).adjusted) for a in scalings]))
        dates.append(pd.Timestamp(factors.index[position]))
        if progress and len(rows) % 250 == 0:
            print(f"  {label}: {len(rows):,} / {periods - start:,}", flush=True)
    return pd.DataFrame(np.asarray(rows), index=pd.DatetimeIndex(dates, name="date"), columns=names)


def build_cache(data: Inputs, settings: config.Config, *, only: str | None = None) -> None:
    """Build the partial for one horizon, or assemble the committed cache from both."""
    columns = [HorizonColumn(horizon) for horizon in HORIZONS]
    digest = {
        "risk_config_digest": history.risk_config_digest(settings),
        "panel_digest": history.panel_digest(data.factors),
    }
    if only is not None:
        chosen = [column for column in columns if column.name == only]
        if not chosen:
            names = [column.name for column in columns]
            raise SystemExit(f"unknown horizon {only!r}; expected one of {names}")
        history.load_or_build(
            history.partial_path(_PARTIALS, chosen[0]),
            digest,
            lambda: build_partial(chosen[0], data, settings),
        )
        return
    if history.fan_out("mafrm.factors.bias_report", columns, flag="--column"):
        raise SystemExit(1)
    pieces = []
    order: list[str] = []
    for column in columns:
        piece = history.read_partial(history.partial_path(_PARTIALS, column))
        piece = piece.rename(columns={name: f"{column.name}_{name}" for name in piece.columns})
        order.extend(piece.columns)
        pieces.append(piece)
    cache = history.assemble(
        pieces,
        order=order,
        what=(
            "SPEC.md 5.3's eigenfactor-adjusted factor covariance at both published a, one "
            "row per forecast date and horizon, upper triangle row-major. Built by "
            f"`{_REBUILD_HINT}`. It is the ONLY expensive quantity in SPEC.md 6's battery -- "
            "every other stage is rebuilt on each render. tests/test_validation.py fails if "
            "risk_config_digest stops matching config/model.yaml."
        ),
        settings=settings,
        panel_digests=history.panel_digest(data.factors),
        variants=columns,
        extra={"factors": int(data.factors.shape[1]), "assets": int(data.returns.shape[1])},
    )
    history.write_cache(cache, _CACHE_PATH)
    print(f"wrote {_CACHE_PATH} ({len(cache.frame):,} rows, {len(cache.frame.columns)} columns)")


# ---------------------------------------------------------------------------
# The variants scored by the battery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantSpec:
    """One risk-model variant, and what it says about the pipeline.

    The ladder is a **one-at-a-time attribution**, not a grid: the factor rows
    vary a single SPEC.md 5 stage with the specific leg held at its pre-VRA
    value, and the one specific row varies the specific leg with the factor leg
    held at ``a = 1.0``. Nothing here is selected on the basis of the ``B`` it
    produces -- the pipeline stays exactly as SPEC.md 5 specifies and this is the
    accounting of what each published stage contributed to the statistic.
    """

    name: str
    #: Which SPEC.md 5 stage the factor covariance is taken after.
    stage: str
    #: The published eigenfactor ``a``, or ``None`` before stage 4.
    scaling: float | None
    #: Whether SPEC.md 5.4's factor multiplier is applied.
    factor_regime: bool
    #: Whether SPEC.md 5.4's specific multiplier is applied.
    specific_regime: bool
    label: str


def variants(settings: config.Config) -> tuple[VariantSpec, ...]:
    """The attribution ladder. Nine rows, and the last is the pipeline as specified."""
    low, high = settings.model.eigenfactor.scaling_a[0], settings.model.eigenfactor.scaling_a[-1]
    return (
        VariantSpec(
            name="sample",
            stage="sample",
            scaling=None,
            factor_regime=False,
            specific_regime=False,
            label="naive asset-level EWMA sample covariance (no factor structure)",
        ),
        VariantSpec(
            name="ewma",
            stage="ewma",
            scaling=None,
            factor_regime=False,
            specific_regime=False,
            label="SPEC.md 5.1 only -- separated EWMA, moments about zero",
        ),
        VariantSpec(
            name="newey_west",
            stage="newey_west",
            scaling=None,
            factor_regime=False,
            specific_regime=False,
            label="+ SPEC.md 5.2 Newey-West",
        ),
        VariantSpec(
            name="psd_repair",
            stage="psd_repair",
            scaling=None,
            factor_regime=False,
            specific_regime=False,
            label="+ SPEC.md 5.2 PSD repair (the pre-eigenfactor forecast)",
        ),
        VariantSpec(
            name=f"eigen_{_tag(low)}",
            stage="eigenfactor",
            scaling=low,
            factor_regime=False,
            specific_regime=False,
            label=f"+ SPEC.md 5.3 eigenfactor, a = {low:g} (attribution-facing, USE4 production)",
        ),
        VariantSpec(
            name=f"eigen_{_tag(high)}",
            stage="eigenfactor",
            scaling=high,
            factor_regime=False,
            specific_regime=False,
            label=f"+ SPEC.md 5.3 eigenfactor, a = {high:g} (optimizer-facing)",
        ),
        VariantSpec(
            name=f"vra_{_tag(low)}",
            stage="eigenfactor",
            scaling=low,
            factor_regime=True,
            specific_regime=False,
            label=f"+ SPEC.md 5.4 factor VRA, a = {low:g}",
        ),
        VariantSpec(
            name=f"vra_{_tag(high)}",
            stage="eigenfactor",
            scaling=high,
            factor_regime=True,
            specific_regime=False,
            label=f"+ SPEC.md 5.4 factor VRA, a = {high:g}",
        ),
        VariantSpec(
            name=f"specified_{_tag(low)}",
            stage="eigenfactor",
            scaling=low,
            factor_regime=True,
            specific_regime=True,
            label=f"+ SPEC.md 5.4 SPECIFIC VRA, a = {low:g} -- the pipeline exactly as specified",
        ),
    )


# ---------------------------------------------------------------------------
# The cheap half -- rebuilt on every render
# ---------------------------------------------------------------------------


def _lagged_rolling_multiplier(bias: np.ndarray, *, halflife: int) -> np.ndarray:
    """``lambda^2`` as it stood at ``t-1``, for the forecast applied at ``t``.

    :func:`mafrm.risk.regime.rolling_multiplier` returns the multiplier including
    the current date's own ``B_t``, which is computed from the return at ``t``.
    Applying that to the forecast for ``t`` would put the realisation inside its
    own forecast -- a look-ahead of exactly the kind SPEC.md 6.1 is measuring the
    absence of. The series is therefore shifted by one and the first date has no
    multiplier at all rather than a substituted one.
    """
    rolled = rolling_multiplier(bias, halflife=halflife)
    lagged = np.empty_like(rolled)
    lagged[0] = np.nan
    lagged[1:] = rolled[:-1]
    return lagged


@dataclass(frozen=True)
class StageHistory:
    """The cheap stages, rebuilt on every render, plus what the repair did."""

    #: Stage name -> ``n x K x K`` factor covariances, one per forecast date.
    matrices: dict[str, np.ndarray]
    #: ``n x N x N`` naive asset-level EWMA sample covariances.
    sample: np.ndarray
    #: Whether SPEC.md 5.2's PSD repair fired at each date.
    repair_fired: np.ndarray
    dates: pd.DatetimeIndex
    #: ``K / T_eff`` realised, at each forecast date.
    k_over_realised_t_eff: np.ndarray
    #: How many panel rows fed each forecast. Strictly increasing by one, and the
    #: quantity :class:`ScoredWindow`'s rank condition is expressed in.
    window_rows: np.ndarray
    #: Position in the panel of each forecast date. ``exposures[position - 1]``
    #: is the design known when the forecast was made.
    positions: np.ndarray


def stage_history(risk: RiskConfig, data: Inputs, *, start: int) -> StageHistory:
    """Stages 1-3 at every date, plus the naive asset-level comparand.

    One expanding-window pass gives all three factor stages, because
    :attr:`~mafrm.risk.covariance.CovarianceBuild.stages` carries every stage's
    matrix and stopping after ``psd_repair`` runs all three. The naive sample
    covariance is the **same** EWMA second moment
    (:func:`~mafrm.risk.covariance.ewma_second_moment`) applied to asset returns
    instead of factor returns at the same volatility half-life -- one estimator,
    two panels, which is what makes the comparison a statement about factor
    structure rather than about two different estimators.

    THE NAIVE COMPARAND IS EWMA AND NOT AN EQUAL-WEIGHT WINDOW, AND THAT IS A CHOICE
        SPEC.md 6.2 asks for "the naive sample estimator" and SPEC.md 6.3's
        published comparison uses equal-weight windows of ``T = 60 ... 500``.
        Written here as the EWMA second moment at this model's own volatility
        half-life, because the question SPEC.md 6.4 poses is what **factor
        structure** buys at a fixed effective sample size -- its table puts the
        ``N = 15`` sample covariance and the 6-factor model at the same
        ``T_eff = 242``. An equal-weight comparand would change the estimator and
        the sample size at once and could not answer it. No parameter is invented:
        the half-life is ``covariance.factor_volatility_halflife``, unchanged.
    """
    factors = data.factors
    asset_values = data.returns.to_numpy(dtype=float)
    periods = factors.shape[0]
    stages = ("ewma", "newey_west", "psd_repair")
    collected: dict[str, list[np.ndarray]] = {name: [] for name in stages}
    sample: list[np.ndarray] = []
    fired: list[bool] = []
    ratios: list[float] = []
    dates: list[pd.Timestamp] = []
    positions: list[int] = []
    for position in range(start, periods):
        build: CovarianceBuild = run_pipeline(
            factors.iloc[:position], risk, stop_after="psd_repair"
        )
        if build.observations != position:  # pragma: no cover - a construction check
            raise ValueError(
                f"bias_report.stage_history: the build for {factors.index[position]!s} saw "
                f"{build.observations} observations at position {position}. The forecast for "
                "date t must be made from data strictly before t."
            )
        by_name = {result.name: result.matrix for result in build.stages}
        for name in stages:
            collected[name].append(by_name[name])
        sample.append(
            ewma_second_moment(asset_values[:position], halflife=float(risk.volatility_halflife))
        )
        fired.append(build.repair_fired)
        ratios.append(build.k_over_realised_t_eff)
        dates.append(pd.Timestamp(factors.index[position]))
        positions.append(position)
    return StageHistory(
        matrices={name: np.asarray(values) for name, values in collected.items()},
        sample=np.asarray(sample),
        repair_fired=np.asarray(fired, dtype=bool),
        dates=pd.DatetimeIndex(dates),
        k_over_realised_t_eff=np.asarray(ratios, dtype=float),
        window_rows=np.asarray(positions, dtype=int),
        positions=np.asarray(positions, dtype=int),
    )


# ---------------------------------------------------------------------------
# The burn-in, registered before any bias statistic was computed
# ---------------------------------------------------------------------------


def first_clean_position(stages: StageHistory) -> int:
    """One past the LAST date SPEC.md 5.2's PSD repair fired.

    Not "the first date it did not fire". The firings on this panel are
    **interleaved** with non-firing dates rather than forming the single
    contiguous episode ``reports/psd_repairs.md`` found on the macro factor
    panel, and a history that skipped the firing dates in the middle would have
    interior gaps. SPEC.md 5.4's multiplier is an exponentially weighted sum over
    consecutive forecast dates and ``history.Cache.bias`` refuses interior gaps
    outright, for the same reason: a hole inside an expanding-window history is a
    defect, not an alignment artefact. So the history is the **contiguous tail**
    on which the estimator is well posed throughout.
    """
    fired = np.flatnonzero(stages.repair_fired)
    if fired.size == 0:
        return int(stages.positions[0])
    return int(stages.positions[fired[-1]]) + 1


@dataclass(frozen=True)
class ScoredWindow:
    """Which forecast dates the reported statistics run on, and why.

    REGISTERED BEFORE ANY BIAS STATISTIC WAS COMPUTED (W4-P2), because SPEC.md
    5.4.1 ruling 3 says the W4 harness owes a criterion written before the
    numbers are looked at, and SPEC.md 5.1.2 fixes its shape: a bound on
    ``K/T_eff``, never a day count.

    **A date is scored when every variant has a forecast that exists.** Three
    conditions, each of them "the estimator has no answer" rather than "the
    estimator's answer is poor", and not one of them a threshold anyone chose:

    1. **It is at or after** :func:`first_clean_position`. Before that, SPEC.md
       5.2's repair has written ``psd_eigenvalue_floor`` into the spectrum, the
       matrix was not a covariance matrix, and SPEC.md 5.3's Monte Carlo raises
       on the Cholesky of the correlation it produces. The boundary is measured
       rather than chosen -- ``RepairReport`` records it -- and it is a
       ``K/T_eff`` boundary in SPEC.md 5.1.2's required shape: every firing on
       this panel sits at ``K/T_eff >= 0.46`` and the boundary lands at the same
       place at both horizons though their volatility half-lives differ by a
       factor of three. Excluding these dates removes manufactured numbers and
       substitutes no estimate -- the same class as ``experiments.md`` row 119's.
    2. **The naive asset-level comparand is of full rank**, i.e. ``T > N``. Below
       it the sample covariance is singular by construction and
       :func:`~mafrm.risk.validation.minimum_variance_weights` has nothing to
       solve. Arithmetic, in the same sense as SPEC.md 5.4.1's
       ``max(K + 1, lags + 1)``.
    3. **SPEC.md 5.4's multiplier exists**, which needs at least one earlier
       forecast date: ``lambda^2`` at ``t`` is an exponentially weighted mean of
       ``B_s^2`` for ``s < t``, so the first date of any history has none. Costs
       exactly one date, and substituting a value for it would be inventing a
       multiplier rather than computing one.

    **The window is common to every variant**, which is the half that would be
    easy to get wrong. A comparison of families across variants scored on
    different date sets is not a comparison, and letting each variant start where
    its own estimator became well behaved would let the worst-behaved one look
    best by being scored on the calmest sample.

    **What this criterion is NOT.** It is not a claim that the retained early
    dates are well estimated. ``K/T_eff`` runs above 0.4 at the start of the
    scored window and Shepard's ``[1 - K/T_eff]^-2`` is enormous there. Those
    dates are **labelled rather than suppressed** -- SPEC.md 5.4.1 ruling 3's
    treatment -- and the report carries the ``K/T_eff`` profile beside the chart
    so a reader can see which part of the series is estimated at what.
    """

    mask: np.ndarray
    dates: pd.DatetimeIndex
    dropped_by_repair: int
    dropped_by_rank: int
    dropped_by_regime: int
    assets: int
    #: Dates this variant could have scored that a COMMON window removed. Zero
    #: for SPEC.md 6.2's own runs, where the window is already common to every
    #: variant; non-zero for W4-P2b's sweep, where nine half-lives have nine
    #: different repair boundaries and SPEC.md 5.1.3 scores their intersection.
    dropped_by_common: int = 0

    @property
    def observations(self) -> int:
        return int(self.mask.sum())

    def render(self) -> str:
        return (
            f"scored window: {self.observations:,} of {self.mask.size:,} forecast dates, "
            f"{self.dates[0].date()} to {self.dates[-1].date()}; dropped "
            f"{self.dropped_by_repair} before the last PSD-repair firing, "
            f"{self.dropped_by_rank} where T <= N = {self.assets} left the naive comparand "
            f"singular, and {self.dropped_by_regime} with no SPEC.md 5.4 multiplier yet"
            + (
                f", and {self.dropped_by_common} outside the window common to every grid point"
                if self.dropped_by_common
                else ""
            )
        )


def scored_window(stages: StageHistory, *, assets: int, clean: int) -> ScoredWindow:
    """Apply :class:`ScoredWindow`'s three registered conditions. Common to every variant."""
    before_clean = stages.positions < clean
    rank_deficient = stages.window_rows <= assets
    no_regime = stages.positions == clean
    mask = ~(before_clean | rank_deficient | no_regime)
    if not mask.any():
        raise ValueError(
            "bias_report.scored_window: no date survives the registered criterion. That is a "
            "defect in the panel or the pipeline, not a reason to widen the criterion."
        )
    return ScoredWindow(
        mask=mask,
        dates=stages.dates[mask],
        dropped_by_repair=int(before_clean.sum()),
        dropped_by_rank=int((rank_deficient & ~before_clean).sum()),
        dropped_by_regime=int((no_regime & ~before_clean & ~rank_deficient).sum()),
        assets=int(assets),
    )


def restrict_window(
    window: ScoredWindow, dates: pd.DatetimeIndex, keep: pd.DatetimeIndex
) -> ScoredWindow:
    """Narrow a scored window to dates a COMMON window also scores. SPEC.md 5.1.3.

    SPEC.md 6.2.4's *"the window is common to every variant"* clause, extended
    one step for W4-P2b. Nine volatility half-lives have nine PSD-repair
    boundaries, so a curve of ``B`` against the half-life scored on nine
    different date sets would be partly a curve of ``B`` against **sample** --
    and the reason is SPEC.md 6.2.4's own: letting each point start where its own
    estimator became well behaved would let the worst-behaved one look best by
    being scored on the calmest sample.

    Each variant's own window is a contiguous tail, so an intersection of them is
    one too. That is asserted rather than assumed: an interior hole would mean a
    forecast date went missing inside an unbroken expanding loop, which is the
    defect ``history.Cache.bias`` refuses for the same reason.
    """
    inside = np.asarray(pd.DatetimeIndex(dates).isin(keep))
    mask = window.mask & inside
    if not mask.any():
        raise ValueError(
            "bias_report.restrict_window: no date survives the common window. The grid points "
            "share no scored dates, which is a defect in the panel or the grid rather than a "
            "reason to score them on different samples."
        )
    positions = np.flatnonzero(mask)
    if positions[-1] - positions[0] + 1 != positions.size:
        raise ValueError(
            "bias_report.restrict_window: the common window has an interior gap. Every "
            "variant's own window is a contiguous tail, so an intersection of them must be "
            "one; a hole means a forecast date went missing inside an expanding loop."
        )
    return dataclasses.replace(
        window,
        mask=mask,
        dates=pd.DatetimeIndex(dates)[mask],
        dropped_by_common=int((window.mask & ~inside).sum()),
    )


# ---------------------------------------------------------------------------
# One horizon, end to end
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantRun:
    """One variant's forecasts and the report SPEC.md 15.2's ``validate`` produced."""

    spec: VariantSpec
    report: validation.ValidationReport
    rolling: dict[str, validation.RollingBias]
    monthly: validation.ValidationReport
    #: Median realised volatility of the min-var portfolio, in bps/day. SPEC.md
    #: 6.5's headline model-comparison metric: Menchero & Ji argue realised
    #: min-var volatility converges far faster than a realised information ratio.
    min_var_realised_volatility: float
    #: Daily forecast volatility and realised return for family 1's members and
    #: family 4, on the scored window -- what SPEC.md 6.5's battery runs on
    #: (W4-P3). Family 2 is not kept: nothing downstream reads it.
    member_forecasts: pd.DataFrame
    member_returns: pd.DataFrame


@dataclass(frozen=True)
class HorizonRun:
    """Everything one horizon produced."""

    horizon: Horizon
    risk: RiskConfig
    window: ScoredWindow
    runs: dict[str, VariantRun]
    k_over_realised_t_eff: np.ndarray
    dates: pd.DatetimeIndex
    #: SPEC.md 5.4's factor multiplier as it stood at each scored date, lagged.
    factor_lambda: dict[str, np.ndarray]
    #: SPEC.md 5.4's specific multiplier, lagged.
    specific_lambda: np.ndarray


@dataclass(frozen=True)
class ForecastPanel:
    """One horizon's forecast ingredients on its scored window, indexed by the date FORECAST.

    EXTRACTED FROM :func:`run_for` IN W6-P1, AT THE SECOND CALLER
        SPEC.md 8's optimizer must see exactly the matrices SPEC.md 6.2's family
        4 was scored on, or the ``alpha = 0`` control of SPEC.md 8.5.1 compares
        two different models. One code path is what makes that a check rather
        than a coincidence -- the same reasoning that made :func:`build_history`
        and :func:`run_for` shared with W4-P2b's sweep.

    Row ``i`` is the forecast FOR ``dates[i]``, built from data strictly before
    it: ``exposures[i]`` is the design fitted through the previous panel row,
    ``factor[tag][i]`` the eigenfactor-adjusted factor covariance from the
    committed cache (pre-VRA), ``specific_variance[i]`` the pre-VRA specific
    variances. **An optimizer deciding at the close of ``dates[i]`` therefore
    uses row ``i + 1``**, which is the forecast made from everything through
    ``dates[i]``. Units are the panel's: basis points per day.
    """

    risk: RiskConfig
    column: str
    dates: pd.DatetimeIndex
    #: Panel row of each scored date.
    positions: np.ndarray
    #: ``n x N x K``, already LAGGED by one panel row.
    exposures: np.ndarray
    #: Eigenfactor tag -> ``n x K x K``, pre-VRA.
    factor: dict[str, np.ndarray]
    #: Stage name -> ``n x K x K`` for the cheap stages.
    stage_matrices: dict[str, np.ndarray]
    #: ``n x N x N`` naive asset-level comparand.
    sample: np.ndarray
    #: ``n x N`` pre-VRA specific variances (deviations squared).
    specific_variance: np.ndarray
    #: SPEC.md 5.4's lagged multipliers on the scored window.
    factor_lambda: dict[str, np.ndarray]
    specific_lambda: np.ndarray
    window: ScoredWindow
    k_over_realised_t_eff: np.ndarray

    def __len__(self) -> int:
        return len(self.dates)

    def ingredients(
        self, spec: VariantSpec, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(X, F, delta^2)`` for row ``index`` under ``spec``. The sample variant has none."""
        if spec.stage == "sample":
            raise ValueError(
                "ForecastPanel.ingredients: the sample variant has no factor structure"
            )
        if spec.stage == "eigenfactor":
            assert spec.scaling is not None
            factor = self.factor[_tag(spec.scaling)][index]
        else:
            factor = self.stage_matrices[spec.stage][index]
        if spec.factor_regime:
            assert spec.scaling is not None
            factor = factor * self.factor_lambda[_tag(spec.scaling)][index]
        variances = self.specific_variance[index]
        if spec.specific_regime:
            variances = variances * self.specific_lambda[index]
        return self.exposures[index], np.asarray(factor), np.asarray(variances)

    def forecasts(self, spec: VariantSpec) -> list[validation.RiskForecast]:
        """One variant's ``RiskForecast`` for every scored date -- what :func:`run_for` scores."""
        if spec.stage == "sample":
            return [validation.RiskForecast(covariance=matrix) for matrix in self.sample]
        if spec.stage == "eigenfactor":
            assert spec.scaling is not None
            factor = self.factor[_tag(spec.scaling)]
        else:
            factor = self.stage_matrices[spec.stage]
        if spec.factor_regime:
            assert spec.scaling is not None
            factor = factor * self.factor_lambda[_tag(spec.scaling)][:, None, None]
        variances = self.specific_variance
        if spec.specific_regime:
            variances = variances * self.specific_lambda[:, None]
        return [
            validation.RiskForecast.from_factor_model(
                self.exposures[index], factor[index], variances[index]
            )
            for index in range(factor.shape[0])
        ]


def read_cache(data: Inputs, settings: config.Config) -> history.Cache:
    """The committed eigenfactor history, refused if stale against the config or the panel."""
    return history.read_cache(
        _CACHE_PATH,
        rebuild_hint=_REBUILD_HINT,
        settings=settings,
        panel_digests={"macro": history.panel_digest(data.factors)},
    )


def forecast_panel(
    risk: RiskConfig,
    column: str,
    data: Inputs,
    cache: history.Cache,
    *,
    restrict: pd.DatetimeIndex | None = None,
) -> ForecastPanel:
    """Assemble every per-date forecast ingredient for one ``RiskConfig`` on its scored window.

    This is :func:`run_for`'s preamble, unchanged in arithmetic. The lag is
    applied HERE, once, at the one place that can see the panel positions:
    ``exposures[i] = data.exposures[positions[i] - 1]``.
    """
    factor_names = list(data.factor_names)
    assets = list(data.assets)
    size = len(factor_names)
    start = minimum_observations(risk, factors=size)

    stages = stage_history(risk, data, start=start)
    # The cached history begins at the first date SPEC.md 5.3's stage has an
    # answer (see first_clean_position); the cheap pass begins at the arithmetic
    # floor. Everything below is indexed on the cheap pass's dates, with the
    # eigenfactor block NaN before `clean` -- and the scored window excludes
    # exactly those dates, so no NaN can reach a statistic.
    clean = first_clean_position(stages)
    clean_mask = stages.positions >= clean
    cached = {
        _tag(scaling): np.asarray(
            [
                _unpack(row, size)
                for row in cache.frame[
                    [f"{column}_{name}" for name in _upper_names(_tag(scaling), size)]
                ]
                .reindex(stages.dates)
                .to_numpy(dtype=float)
            ]
        )
        for scaling in risk.eigenfactor_scaling
    }
    for tag, block in cached.items():
        if not np.all(np.isfinite(block[clean_mask])):
            raise ValueError(
                f"bias_report.forecast_panel: the committed cache has no row for every forecast "
                f"date at or after {stages.dates[clean_mask][0].date()} at {column}/{tag}. "
                f"Rebuild: {_REBUILD_HINT}"
            )

    include = [name for name in assets if name not in set(data.excluded)]
    specific_history = specific.specific_forecast_history(
        data.panel.residuals,
        data.panel.exposure_matrix(pd.Timestamp(data.panel.residuals.index[-1])),
        data.buckets,
        risk,
        minimum_observations=start,
        include=include,
        shrinkage_target=include,
    )
    deviations = specific_history.deviations.reindex(stages.dates)[assets].to_numpy(dtype=float)
    if not np.all(np.isfinite(deviations)):
        raise ValueError(
            "bias_report.forecast_panel: the specific-risk history does not cover every forecast "
            "date. Both histories start at the same arithmetic floor, so this is an alignment "
            "defect."
        )
    specific_lambda = _lagged_rolling_multiplier(
        specific_history.bias, halflife=risk.volatility_regime_halflife
    )

    # SPEC.md 5.4's multiplier runs over the eigenfactor history's OWN range,
    # which starts at `clean` -- not over the scored window. A multiplier fitted
    # only on scored dates would silently give itself a burn-in the model never
    # had.
    factor_returns = data.factors.to_numpy(dtype=float)[stages.positions]
    factor_lambda = {}
    for scaling in risk.eigenfactor_scaling:
        block = cached[_tag(scaling)][clean_mask]
        bias = cross_sectional_bias(
            factor_returns[clean_mask], np.sqrt(np.einsum("tkk->tk", block))
        )
        lagged = np.full(len(stages.dates), np.nan)
        lagged[clean_mask] = _lagged_rolling_multiplier(
            bias, halflife=risk.volatility_regime_halflife
        )
        factor_lambda[_tag(scaling)] = lagged

    window = scored_window(stages, assets=len(assets), clean=clean)
    if restrict is not None:
        window = restrict_window(window, stages.dates, restrict)
    mask = window.mask
    return ForecastPanel(
        risk=risk,
        column=column,
        dates=stages.dates[mask],
        positions=stages.positions[mask],
        # THE LAG. `positions[i]` is the panel row of scored date i, so the design
        # known when its forecast was made is the row before it.
        exposures=data.exposures[stages.positions[mask] - 1],
        factor={tag: block[mask] for tag, block in cached.items()},
        stage_matrices={name: block[mask] for name, block in stages.matrices.items()},
        sample=stages.sample[mask],
        specific_variance=np.square(deviations[mask]),
        factor_lambda={tag: values[mask] for tag, values in factor_lambda.items()},
        specific_lambda=specific_lambda[mask],
        window=window,
        k_over_realised_t_eff=stages.k_over_realised_t_eff[mask],
    )


def _month_starts(dates: pd.DatetimeIndex) -> np.ndarray:
    """True on the first scored date of each calendar month.

    The monthly leg holds each portfolio for the month it was built at the start
    of, because a month's return is only the return *of a portfolio* if the
    portfolio was not rebuilt inside it.
    """
    period = dates.to_period("M")
    mask = np.ones(len(dates), dtype=bool)
    mask[1:] = period[1:] != period[:-1]
    return mask


def _to_monthly(
    forecasts: pd.DataFrame, realised: pd.DataFrame, starts: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate a daily portfolio series to one observation per month.

    ``R_m = sum_{t in m} R_t`` for a portfolio whose weights were fixed at the
    start of the month, and ``sigma_m = sigma_{first day of m} * sqrt(days in m)``
    -- the square-root-of-time rule applied to the forecast that was standing
    when the month opened, which is the forecast a monthly-horizon user would
    have had. The day count is the month's own number of scored dates rather
    than ``data.trading_days_per_month``, because a short month really does
    contain less variance and substituting the nominal 21 would put a known
    error into the denominator.
    """
    period = pd.DatetimeIndex(forecasts.index).to_period("M")
    counts = pd.Series(1, index=forecasts.index).groupby(period).sum()
    opening = forecasts[starts]
    opening.index = pd.PeriodIndex(pd.DatetimeIndex(opening.index).to_period("M"))
    scaled = opening.mul(np.sqrt(counts.reindex(opening.index).to_numpy(dtype=float)), axis=0)
    summed = realised.groupby(period).sum()
    return scaled.reindex(summed.index), summed


def run(
    horizon: Horizon,
    data: Inputs,
    cache: history.Cache,
    settings: config.Config,
) -> HorizonRun:
    """Score every variant at one horizon on all four of SPEC.md 6.2's families."""
    return run_for(
        RiskConfig.load(horizon=horizon, config=settings), horizon, data, cache, settings
    )


def run_for(
    risk: RiskConfig,
    column: str,
    data: Inputs,
    cache: history.Cache,
    settings: config.Config,
    *,
    restrict: pd.DatetimeIndex | None = None,
    only: Sequence[str] | None = None,
) -> HorizonRun:
    """Score every variant for one ``RiskConfig``, reading ``column``'s block of the cache.

    SEPARATE FROM :func:`run` FOR THE SAME REASON :func:`build_history` IS
        W4-P2b's sweep (SPEC.md 5.1.3) scores nine volatility half-lives that are
        not either horizon's shipped value, out of its own cache, and the two
        shipped points must come out of this function identically to the way
        W4-P2 published them. One code path is what makes that a check rather
        than a coincidence.

    ``column`` is the cache's column prefix, which for :func:`run` is the horizon
    and for the sweep is the grid point. ``risk.horizon`` still says which
    horizon's *shape* -- specific-risk half-life and VRA half-life -- is in force.

    ``restrict`` narrows the scored window to dates some wider comparison also
    scores; see :func:`restrict_window`. SPEC.md 5.4's multipliers are still
    fitted over the eigenfactor history's own range, which starts at
    :func:`first_clean_position` and not at the scored window, so restricting the
    scored set cannot give a multiplier a burn-in the model never had.

    ``only`` restricts the ladder to the named variants (W8-P1's K/T chart
    scores two of nine at each grid point); ``None`` scores the whole ladder.
    The arithmetic of a scored variant does not depend on which others ran.
    """
    battery = settings.model.validation
    factor_names = list(data.factor_names)
    assets = list(data.assets)
    panel = forecast_panel(risk, column, data, cache, restrict=restrict)
    chosen = variants(settings)
    if only is not None:
        wanted = set(only)
        missing = wanted - {spec.name for spec in chosen}
        if missing:
            raise ValueError(f"bias_report.run_for: unknown variant(s) {sorted(missing)}")
        chosen = tuple(spec for spec in chosen if spec.name in wanted)
    dates = panel.dates
    asset_returns = data.returns.to_numpy(dtype=float)[panel.positions]
    realised_factors = data.factors.to_numpy(dtype=float)[panel.positions]
    starts = _month_starts(dates)
    window_days = battery.rolling_window_months * settings.model.data.trading_days_per_month

    runs: dict[str, VariantRun] = {}
    for spec in chosen:
        forecasts = panel.forecasts(spec)
        for index, forecast in enumerate(forecasts):
            validation.assert_forecast_psd(forecast, f"{spec.name} [{column}] {dates[index]!s}")
        vols, returns, portfolios = validation.portfolio_families(
            forecasts,
            dates=dates,
            asset_returns=asset_returns,
            factor_returns=None if spec.stage == "sample" else realised_factors,
            asset_names=assets,
            factor_names=[] if spec.stage == "sample" else factor_names,
            random_portfolios=battery.random_portfolios,
            seed=settings.model.seed,
            rebalance=None,
        )
        scored = _score(vols, returns, portfolios, data, battery)
        held_vols, held_returns, held_portfolios = validation.portfolio_families(
            forecasts,
            dates=dates,
            asset_returns=asset_returns,
            factor_returns=None if spec.stage == "sample" else realised_factors,
            asset_names=assets,
            factor_names=[] if spec.stage == "sample" else factor_names,
            random_portfolios=battery.random_portfolios,
            seed=settings.model.seed,
            rebalance=starts,
        )
        monthly_vols, monthly_returns = _to_monthly(held_vols, held_returns, starts)
        monthly = _score(monthly_vols, monthly_returns, held_portfolios, data, battery)
        rolling = _rolling_by_family(vols, returns, portfolios, data, battery, window_days)
        runs[spec.name] = VariantRun(
            spec=spec,
            report=scored,
            rolling=rolling,
            monthly=monthly,
            min_var_realised_volatility=float(np.std(returns["min_var"].to_numpy(), ddof=1)),
            member_forecasts=vols[[*assets, "min_var"]],
            member_returns=returns[[*assets, "min_var"]],
        )

    return HorizonRun(
        horizon=risk.horizon,
        risk=risk,
        window=panel.window,
        runs=runs,
        k_over_realised_t_eff=panel.k_over_realised_t_eff,
        dates=dates,
        factor_lambda=panel.factor_lambda,
        specific_lambda=panel.specific_lambda,
    )


def _drop_identity(
    portfolios: validation.PortfolioSet, excluded: Sequence[str]
) -> validation.PortfolioSet:
    """The W4-P2 forward constraint, applied to family 1 and nowhere else.

    ``hy_credit`` and ``commodity`` regress against the factor set at ``R^2 =
    1.000``, so their ``b_nt`` is pinned at a ratio that cannot understate
    anything, and averaging them into a 13-asset MRAD pulls it toward zero. They
    are removed from **family 1's membership**, which is what SPEC.md 6.1
    aggregates over. They stay in the asset covariance, so every family-2 and
    family-4 portfolio still holds them -- excluding them from the model would
    change the model, and the constraint is about aggregation.
    """
    dropped = set(excluded)
    families = dict(portfolios.families)
    families[validation.INDIVIDUAL] = tuple(
        name for name in families[validation.INDIVIDUAL] if name not in dropped
    )
    return validation.PortfolioSet(families=families, rebalance=portfolios.rebalance)


def _score(
    vols: pd.DataFrame,
    returns: pd.DataFrame,
    portfolios: validation.PortfolioSet,
    data: Inputs,
    battery: config.ValidationBatteryConfig,
) -> validation.ValidationReport:
    kept = _drop_identity(portfolios, data.excluded)
    members = [name for names in kept.families.values() for name in names]
    return validation.validate(
        vols[members], returns[members], kept, clip=battery.standardized_return_clip
    )


def _rolling_by_family(
    vols: pd.DataFrame,
    returns: pd.DataFrame,
    portfolios: validation.PortfolioSet,
    data: Inputs,
    battery: config.ValidationBatteryConfig,
    window: int,
) -> dict[str, validation.RollingBias]:
    """The chart's series: one rolling ``B`` per family, averaged across members.

    The family aggregate is the **median** across members at each window, not the
    mean: family 2 has a hundred members and one of them drawn with a tiny
    forecast would move a mean. The median is a location statistic on a
    cross-section of genuinely different portfolios, which is the one place in
    SPEC.md 6.1 where averaging across members is legitimate -- averaging across
    *windows* would not be (CLAUDE.md failure mode 9).
    """
    kept = _drop_identity(portfolios, data.excluded)
    index = pd.DatetimeIndex(vols.index)
    out: dict[str, validation.RollingBias] = {}
    for family, names in kept.families.items():
        if not names:
            continue
        standardized = validation.standardized_returns(
            returns[list(names)].to_numpy(dtype=float), vols[list(names)].to_numpy(dtype=float)
        )
        rolling = validation.rolling_bias_statistic(
            standardized, index, window=window, clip=battery.standardized_return_clip
        )
        out[family] = validation.RollingBias(
            values=np.median(rolling.values, axis=1, keepdims=True),
            index=rolling.index,
            window=rolling.window,
            overlap=rolling.overlap,
        )
    return out


# ---------------------------------------------------------------------------
# Control C1 -- pre-registered in experiments.md before it was run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagonalControl:
    """Control C1: is family 4's bias driven by SPEC.md 5.5's DIAGONAL `Delta`?

    ``experiments.md`` rows 162-163, registered with both falsifiers before the
    measurement existed. The suspect and the mechanism are at the row; the short
    version is that ``Sigma = X F X' + diag(delta^2)`` throws away every residual
    correlation, W4-P1 measured ``SPY/IWM`` at **-0.5488**, and ``w = Sigma^-1 1``
    loads hardest on exactly the directions a wrong residual correlation
    mis-prices.

    **Off-diagonal only.** ``D_delta R_u D_delta`` keeps SPEC.md 5.5's own shrunk
    deviations on the diagonal to the last bit, so the control isolates the
    diagonality assumption and changes nothing else. Anything that moved the
    diagonal too would confound "the residuals are correlated" with "the
    specific volatilities are wrong", which are different findings.
    """

    horizon: str
    diagonal_family_4: float
    correlated_family_4: float
    diagonal_family_2: float
    correlated_family_2: float
    sample_family_4: float
    half_width: float
    #: Realised min-var volatility, bps/day, for the three.
    realised: tuple[float, float, float]

    @property
    def moved_toward_one(self) -> float:
        return abs(self.diagonal_family_4 - 1.0) - abs(self.correlated_family_4 - 1.0)

    @property
    def holds(self) -> bool:
        """Both registered falsifiers survived."""
        return (
            self.moved_toward_one > self.half_width
            and abs(self.correlated_family_2 - self.diagonal_family_2) <= self.half_width
        )

    @property
    def gap_closed(self) -> float:
        """Fraction of the factor-model-to-naive gap in family 4 that closes."""
        gap = self.diagonal_family_4 - self.sample_family_4
        if abs(gap) < 1e-12:  # pragma: no cover - the gap is the reason C1 exists
            return float("nan")
        return (self.diagonal_family_4 - self.correlated_family_4) / gap


def residual_correlation_control(
    horizon: Horizon,
    data: Inputs,
    cache: history.Cache,
    settings: config.Config,
    base: HorizonRun,
) -> DiagonalControl:
    """Run control C1 at one horizon. See :class:`DiagonalControl`."""
    risk = RiskConfig.load(horizon=horizon, config=settings)
    battery = settings.model.validation
    factor_names = list(data.factor_names)
    assets = list(data.assets)
    size = len(factor_names)
    start = minimum_observations(risk, factors=size)
    stages = stage_history(risk, data, start=start)
    clean = first_clean_position(stages)
    clean_mask = stages.positions >= clean
    low = settings.model.eigenfactor.scaling_a[0]
    block = np.asarray(
        [
            _unpack(row, size)
            for row in cache.frame[[f"{horizon}_{name}" for name in _upper_names(_tag(low), size)]]
            .reindex(stages.dates)
            .to_numpy(dtype=float)
        ]
    )
    include = [name for name in assets if name not in set(data.excluded)]
    specific_history = specific.specific_forecast_history(
        data.panel.residuals,
        data.panel.exposure_matrix(pd.Timestamp(data.panel.residuals.index[-1])),
        data.buckets,
        risk,
        minimum_observations=start,
        include=include,
        shrinkage_target=include,
    )
    deviations = specific_history.deviations.reindex(stages.dates)[assets].to_numpy(dtype=float)
    del clean_mask

    window = scored_window(stages, assets=len(assets), clean=clean)
    mask = window.mask
    dates = stages.dates[mask]
    positions = stages.positions[mask]
    lagged_exposures = data.exposures[positions - 1]
    asset_returns = data.returns.to_numpy(dtype=float)[positions]
    realised_factors = data.factors.to_numpy(dtype=float)[positions]
    residuals = data.panel.residuals[assets].to_numpy(dtype=float)

    forecasts: list[validation.RiskForecast] = []
    for index, position in enumerate(positions):
        # The SAME EWMA second moment the pipeline uses, on the residual panel,
        # sliced strictly before the date it forecasts. Correlation only: the
        # diagonal comes from SPEC.md 5.5 untouched.
        moment = ewma_second_moment(
            residuals[:position], halflife=float(risk.specific_volatility_halflife)
        )
        correlation = correlation_from_covariance(moment)
        deviation = deviations[mask][index]
        specific_covariance = correlation * np.outer(deviation, deviation)
        exposures = lagged_exposures[index]
        matrix = exposures @ block[mask][index] @ exposures.T + specific_covariance
        forecasts.append(
            validation.RiskForecast(
                covariance=0.5 * (matrix + matrix.T), factor_covariance=block[mask][index]
            )
        )
    for index, forecast in enumerate(forecasts):
        validation.assert_forecast_psd(forecast, f"C1 [{horizon}] {dates[index]!s}")

    vols, returns, portfolios = validation.portfolio_families(
        forecasts,
        dates=dates,
        asset_returns=asset_returns,
        factor_returns=realised_factors,
        asset_names=assets,
        factor_names=factor_names,
        random_portfolios=battery.random_portfolios,
        seed=settings.model.seed,
        rebalance=None,
    )
    scored = _score(vols, returns, portfolios, data, battery)
    diagonal = base.runs[f"eigen_{_tag(low)}"]
    return DiagonalControl(
        horizon=horizon,
        diagonal_family_4=diagonal.report.median_bias(validation.OPTIMIZED),
        correlated_family_4=scored.median_bias(validation.OPTIMIZED),
        diagonal_family_2=diagonal.report.median_bias(validation.RANDOM),
        correlated_family_2=scored.median_bias(validation.RANDOM),
        sample_family_4=base.runs["sample"].report.median_bias(validation.OPTIMIZED),
        half_width=validation.chi_square_interval(
            window.observations, level=battery.chi_square_level
        ).upper
        - 1.0,
        realised=(
            base.runs["sample"].min_var_realised_volatility,
            diagonal.min_var_realised_volatility,
            float(np.std(returns["min_var"].to_numpy(), ddof=1)),
        ),
    )


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _interval_line(observations: int, battery: config.ValidationBatteryConfig) -> str:
    normal = validation.normal_band(observations, z=battery.normal_band_z)
    exact = validation.chi_square_interval(observations, level=battery.chi_square_level)
    return (
        f"T = {observations:,}: chi-square {exact.lower:.4f}-{exact.upper:.4f} "
        f"(**exact, and the one that decides**), normal band {normal.lower:.4f}-{normal.upper:.4f} "
        f"(display convention)"
    )


def shepard_for(spec: VariantSpec, data: Inputs, risk: RiskConfig) -> shepard.SecondOrderRisk:
    """The closed form that applies to one variant: Eq. 13 at ``N`` or Eq. 32 at ``K``.

    Both at the VOLATILITY half-life's ``T_eff``, which is exact for the
    single-window ``sample`` variant and an upper bound for every split-window
    factor variant (see :func:`shepard_table`). One function so the bias report
    and SPEC.md 6.5's battery report cannot disagree about which form a variant
    gets.
    """
    parameters = len(data.assets) if spec.stage == "sample" else len(data.factor_names)
    return shepard.second_order_risk(parameters, halflife=risk.volatility_halflife)


def shepard_table(
    runs: dict[Horizon, HorizonRun], data: Inputs, settings: config.Config
) -> list[str]:
    """SPEC.md 6.4: the Shepard-corrected family-4 forecast beside the raw one. W4-P3.

    One closed form, applied where it is derived to apply. The naive ``sample``
    variant IS a whole covariance on one window, so Eq. 13 at ``N`` and the
    volatility half-life is its correction outright. Every factor variant is
    ``F = D rho D`` on two windows, so Eq. 32 at the volatility window is the
    whole-covariance figure for an estimator this pipeline does not run: it is
    printed as SPEC.md 6.4 asks, and labelled an UPPER BOUND, because after
    SPEC.md 5.3 the ``rho`` leg is already corrected and ``rho`` sits on a window
    six times longer. ``reports/second_order_risk.md`` carries the measured
    split-window figure and the three-way accounting; this table carries the
    closed form with its unit attached and the corrected ``B`` it implies.
    """
    factors = len(data.factor_names)
    assets = len(data.assets)
    lines = [
        "| Horizon | Variant | closed form | `p/T_eff` | multiplier (variance) "
        "| multiplier (volatility) | family 4 forecast, median bps/day: raw | corrected "
        "| family 4 `B`: raw | corrected |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for horizon in HORIZONS:
        run_at = runs[horizon]
        for spec in variants(settings):
            run = run_at.runs[spec.name]
            risk = shepard_for(spec, data, run_at.risk)
            if spec.stage == "sample":
                label = f"Eq. 13, `N = {assets}`, exact for a single-window covariance"
            else:
                label = (
                    f"Eq. 32, `K = {factors}`, whole-covariance UPPER BOUND for a split-window "
                    "estimator"
                )
            raw_forecast = float(np.median(run.member_forecasts["min_var"].to_numpy()))
            raw_bias = run.report.median_bias(validation.OPTIMIZED)
            multiplier = risk.multiplier(shepard.Unit.VOLATILITY)
            lines.append(
                f"| {horizon} | `{spec.name}` | {label} | {risk.ratio:.4f} "
                f"| {risk.multiplier(shepard.Unit.VARIANCE):.4f} | {multiplier:.4f} "
                f"| {raw_forecast:.3f} | **{raw_forecast * multiplier:.3f}** "
                f"| {raw_bias:.4f} | **{raw_bias / multiplier:.4f}** |"
            )
    return lines


def _headline_table(runs: dict[Horizon, HorizonRun], settings: config.Config) -> list[str]:
    """Families 2 and 4 on the same matrix, variant by variant. SPEC.md 6.2."""
    lines = [
        "| Horizon | Variant | family 1 median `B` | family 2 median `B` | family 3 median `B` "
        "| family 4 `B` | **gap 4 - 2** | MRAD (family 1) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for horizon in HORIZONS:
        run_at = runs[horizon]
        for spec in variants(settings):
            report = run_at.runs[spec.name].report
            families = report.portfolios.families
            individual = report.median_bias(validation.INDIVIDUAL)
            random = report.median_bias(validation.RANDOM)
            optimized = report.median_bias(validation.OPTIMIZED)
            eigen = (
                f"{report.median_bias(validation.EIGENFACTOR):.4f}"
                if validation.EIGENFACTOR in families
                else "n/a"
            )
            lines.append(
                f"| {horizon} | `{spec.name}` | {individual:.4f} | {random:.4f} | {eigen} "
                f"| **{optimized:.4f}** | **{optimized - random:+.4f}** "
                f"| {report.mrad(validation.INDIVIDUAL):.4f} |"
            )
    return lines


def render(
    runs: dict[Horizon, HorizonRun],
    controls: dict[Horizon, DiagonalControl],
    data: Inputs,
    settings: config.Config,
) -> str:
    """``reports/bias_statistics.md``."""
    battery = settings.model.validation
    days = battery.rolling_window_months * settings.model.data.trading_days_per_month
    short = runs["short"]
    first = short.runs
    pre_vra = f"eigen_{_tag(settings.model.eigenfactor.scaling_a[0])}"

    out: list[str] = []
    add = out.append
    add("# Bias statistics: the test that matters is family 4")
    add("")
    add("Generated by `python -m mafrm.factors.bias_report`. SPEC.md 6.1 and 6.2, W4-P2.")
    add("")
    add(
        'SPEC.md 6.2 states the claim this whole report exists to test: *"validating only on '
        'random or benchmark portfolios will tell you your model is fine when it is not"*. '
        "The test is families 2 and 4 evaluated on the **same covariance matrix** -- random "
        "dollar-neutral portfolios whose weights cannot see the estimation error, against "
        "minimum-variance portfolios whose weights are a function of it."
    )
    add("")

    add("## What was scored")
    add("")
    add(
        f"- Panel: Model A's residual panel, **{len(data.dates):,} dates**, "
        f"{data.dates[0].date()} to {data.dates[-1].date()}, N = {len(data.assets)} assets and "
        f"K = {len(data.factor_names)} factors. Everything ends strictly before "
        f"`sample.holdout_start` = {settings.model.sample.holdout_start}."
    )
    add(f"- {short.window.render()}")
    add(
        f"- Families: {len(data.assets)} individual assets (two excluded from the aggregate, "
        f"below), {battery.random_portfolios} random dollar-neutral portfolios redrawn each "
        f"date, {len(data.factor_names)} factor eigen-portfolios ranked by eigenvalue, and one "
        "minimum-variance fully-invested portfolio rebuilt each date."
    )
    add(
        "- **Family 4's alpha-constrained sibling is not here.** SPEC.md 6.2 also names "
        "minimum variance subject to `alpha'w = 1`; SPEC.md 6.2.1 defers it to W6 because "
        "`alpha` is unspecified and with `alpha = 1` it degenerates to the fully-invested "
        "portfolio already reported."
    )
    add("")

    add("## The headline")
    add("")
    add(
        "`B > 1` is under-forecasting. The column that matters is **gap 4 - 2**: family 2 is "
        "the control that should sit near 1 for every estimator, so whatever family 4 does "
        "above it is the risk model failing on the portfolios an optimizer actually picks."
    )
    add("")
    out.extend(_headline_table(runs, settings))
    add("")
    add(f"Full-sample interval, {_interval_line(short.window.observations, battery)}.")
    add("")
    add("### SPEC.md 6.4: the Shepard-corrected forecast beside the raw one")
    add("")
    add(
        "`[1 - p/T_eff]^-2` multiplies a VARIANCE; the volatility multiplier is its square root, "
        "and the corrected `B` is the raw `B` divided by the volatility multiplier -- the part "
        "of family 4's excess that Shepard's sampling-error argument can account for. For the "
        "`sample` variant the closed form applies exactly (one window, whole covariance). For "
        "every factor variant it is an **upper bound** on the estimation-error correction: `rho` "
        "is estimated on a window six times longer than `sigma` and SPEC.md 5.3 has already "
        "corrected its leg, so applying Eq. 32 at the volatility window on top would double-count "
        "(SPEC.md 5.4.2). The measured split-window figure and the three-way accounting are in "
        "`reports/second_order_risk.md`. What the corrected `B` shows is the finding of SPEC.md "
        "6.2.6 from the other side: the factor variants' family-4 `B` barely moves, because the "
        "excess was never sampling error."
    )
    add("")
    out.extend(shepard_table(runs, data, settings))
    add("")
    low = settings.model.eigenfactor.scaling_a[0]
    high = settings.model.eigenfactor.scaling_a[-1]
    base = short.runs[pre_vra].report
    every_random = [
        runs[horizon].runs[spec.name].report.median_bias(validation.RANDOM)
        for horizon in HORIZONS
        for spec in variants(settings)
    ]
    pre_eigen = short.runs["psd_repair"].report.median_bias(validation.OPTIMIZED)
    at_low = base.median_bias(validation.OPTIMIZED)
    at_high = short.runs[f"eigen_{_tag(high)}"].report.median_bias(validation.OPTIMIZED)
    # `+ 0.0` normalises a negative zero: a change of -1e-6 formats as "-0.0000"
    # at four decimals, which reads as a signed quantity when it is a zero.
    low_change = round(at_low - pre_eigen, 4) + 0.0
    high_change = round(at_high - pre_eigen, 4) + 0.0
    add("### What the battery found, in four lines")
    add("")
    add(
        f"1. **H1 HOLDS.** Family 2 runs {min(every_random):.4f}-{max(every_random):.4f} "
        "across every variant and both horizons, the naive sample covariance included. Random "
        "portfolios say all nine of these models are fine."
    )
    add(
        f"2. **H2 HOLDS**, mid-range rather than at the low end. Family 4 is "
        f"**{base.median_bias(validation.OPTIMIZED):.4f}** short against a registered 1.2-1.5, "
        f"and **the gap between families 2 and 4 on the same covariance matrix is "
        f"{base.median_bias(validation.OPTIMIZED) - base.median_bias(validation.RANDOM):+.4f}**. "
        "That number is the risk-model term of SPEC.md 1's identity, measured."
    )
    add(
        f"3. **H3's first half is REFUTED.** SPEC.md 5.3's eigenfactor adjustment closes **none** "
        f"of that gap -- family 4 moves {low_change:+.4f} at `a = {low:g}` and "
        f"{high_change:+.4f} at `a = {high:g}`. "
        "Its second half holds: family 2 moves by less than 0.0005. **The "
        "stage is not broken** -- family 3, the eigen-portfolios it is actually about, moves the "
        "right way."
    )
    add(
        "4. **Control C1 says why, and it was pre-registered.** The gap is SPEC.md 5.5's "
        "**diagonal** specific-risk assumption, not factor-covariance estimation error. "
        "Restoring the residual correlations off-diagonal only takes family 4 to "
        f"**{controls['short'].correlated_family_4:.4f}**, "
        f"{controls['short'].gap_closed:.0%} of the gap to the naive comparand, while family 2 "
        "moves 0.002. SPEC.md 5.3 corrects the factor covariance, and at "
        f"`K/T_eff = {short.k_over_realised_t_eff[-1]:.4f}` there is barely any of that bias to "
        "correct. **That is a statement about `K/T`, and W7 is where the technique is tested.**"
    )
    add("")

    add("## The reframing: this is SPECIFICATION error, not estimation error")
    add("")
    add(
        "**This is the finding, and it changes what the risk-model term of SPEC.md 1's identity "
        "is made of.** `K/T`, Shepard's second-order correction and SPEC.md 5.3's eigenfactor "
        "adjustment all describe **estimation error** -- the covariance matrix is right in form "
        "and wrong in its numbers because it was fitted on a finite sample. What this panel has "
        "instead is **specification error**: the false diagonal in `Sigma = X F X' + Delta`. The "
        "matrix is wrong in its form, and no amount of data fixes a form."
    )
    add("")
    closed_form = shepard.second_order_risk(
        len(data.factor_names),
        effective_observations=len(data.factor_names) / short.k_over_realised_t_eff[-1],
    )
    shepard_pct = 100.0 * closed_form.understatement(shepard.Unit.VARIANCE)
    shepard_vol_pct = 100.0 * closed_form.understatement(shepard.Unit.VOLATILITY)
    observed_pct = 100.0 * (
        base.median_bias(validation.OPTIMIZED) - base.median_bias(validation.RANDOM)
    )
    add("**Two independent measurements say so, and they were taken different ways.**")
    add("")
    add(
        f"- **C1, empirically.** {controls['short'].gap_closed:.0%} of the family-4 gap is "
        "attributable to the residual correlations, against family 2 moving "
        f"{abs(controls['short'].correlated_family_2 - controls['short'].diagonal_family_2):.3f}. "
        "The effect is direction-selective, which is what the second registered falsifier "
        "established."
    )
    add(
        f"- **Shepard, arithmetically.** At `K/T_eff = {short.k_over_realised_t_eff[-1]:.4f}` the "
        f"closed form puts the understatement at **{shepard_pct:.1f}% in variance** (the figure "
        f"SPEC.md 6.4's table quotes) against an **observed {observed_pct:.0f}%**. So sampling "
        "error accounts "
        "for **at most a sixth of the gap even in principle** -- and that is the generous "
        "reading: taking `[1 - K/T_eff]^-2` as the variance multiplier SPEC.md 6.4's Eq. 32 "
        f"writes puts it at {shepard_vol_pct:.1f}% on volatility and the share at a thirteenth. "
        "At the long horizon `K/T_eff` is "
        f"{runs['long'].k_over_realised_t_eff[-1]:.4f} and the share is smaller again."
    )
    add("")
    add(
        "**That is why the eigenfactor adjustment closes none of the gap, and why H3's first "
        "half being refuted is not a disappointment.** It is the same measurement seen from the "
        "other side. A stage that corrects estimation error cannot move a bias that is not "
        "estimation error, and both C1 and Shepard say this one is not."
    )
    add("")
    add(
        "**Stated as the general finding: at small `K` the industry-standard corrections address "
        "a term that is negligible, while the term that actually matters is one the model "
        "assumes away.** That belongs beside the five published techniques this project has "
        "already measured out of regime -- Marchenko-Pastur denoising, SPEC.md 5.5(c)'s blend, "
        "SPEC.md 5.5(d) at two-member buckets, the Bartlett fallback firing at `K = 3`, and now "
        "SPEC.md 5.3 at `K/T_eff` of a few hundredths. It is the sixth, and it is the one with "
        "the largest consequence, because it is about the technique the project was built around."
    )
    add("")
    add(
        "**It also explains the naive sample covariance winning at `N = 13`.** It imposes no "
        "diagonal assumption, so it has none to be wrong about. Its advantage is the absence of "
        "a misspecification rather than better estimation -- it estimates 91 free covariance "
        "parameters where the factor model estimates 21 in `F` and 13 in `Delta`, so it is the "
        "noisier of the two, and C1 is what separates the two explanations: give the factor "
        "model the residual correlations and it matches the naive comparand at the short horizon "
        "and beats it at the long one. SPEC.md 15.2's table is why the naive advantage does not "
        "survive the equity module: at `N = 500` and `T_eff = 727` the same estimator sits at "
        "`N/T = 0.688` and `[1 - N/T]^-2 = 10.25`, effectively singular, where estimation error "
        "is the whole story."
    )
    add("")
    add(
        "**Why the diagonal is not repaired, since a reader will ask.** Repairing it would mean "
        "not implementing the specified model, and **the specification's failure is the "
        "deliverable**. That is W4-P1's ruling holding rather than a new one: SPEC.md 5.5.1 "
        "settled five undefined terms on structural arguments with the alternatives deliberately "
        "not built, SPEC.md 5.5.5 kept stage (d) after proving it data-independent at two-member "
        "buckets, and in every case the alternative -- drop the stage because it does nothing "
        "here -- was declined. A project that repairs each published technique wherever this "
        "panel embarrasses it ends up reporting a model nobody specified, and can no longer say "
        "which published method failed where. C1 is a control and not a candidate; SPEC.md 5.5 "
        "still specifies a diagonal; `config/model.yaml` is untouched by it."
    )
    add("")
    add("## Which number is the headline, and why it is the pre-VRA one")
    add("")
    add(
        "**The headline is `eigen_*`, not `specified_*`.** SPEC.md 5.4 sets "
        "`lambda_F^2 = sum_t w_t (B_t^F)^2` -- the volatility regime adjustment is calibrated "
        "on realised data precisely to drive the bias statistic to one. A post-VRA `B` near 1 "
        "is therefore substantially what that stage was built to produce, not evidence that "
        "the risk model is calibrated. It is not perfectly circular: `lambda_F^2` is a lagged "
        "EWMA, so the out-of-sample `B` is not identically one. It is close enough that the "
        "project's central claim cannot rest on it."
    )
    add("")
    add(
        "So the two answer different questions. **Pre-VRA `B` measures whether the risk model "
        "understates risk**, which is the project's thesis and the `(1 - 1/B)` term of "
        "SPEC.md 1's identity. **Post-VRA `B` measures whether the regime adjustment "
        "generalises out of sample**, which is a different and lesser question."
    )
    add("")
    add(
        "W4-P1 made the same point about the specific leg from the other side: the specific "
        "VRA absorbs 82-90% of SPEC.md 5.5(a)'s Newey-West correction and reverses the sign of "
        "`lambda_S`. Both legs of `specified_*` are therefore close to circular, and that is "
        "why it is reported last rather than first."
    )
    add("")

    add("## The rolling chart")
    add("")
    add("![rolling bias statistic](bias_statistics.png)")
    add("")
    add(
        f"One panel per family, one line per covariance variant, rolling window "
        f"**{battery.rolling_window_months} months = {days} trading days** stepped one day."
    )
    add("")
    add(
        f"> **CLAUDE.md failure mode 9.** Consecutive readings share "
        f"**{days - 1} of {days}** observations. The series is enormously autocorrelated and "
        f"the number of readings is not an `n`. Every interval on the chart is computed at "
        f"`T = {days}` -- the observations *inside one window* -- and never at the number of "
        f"windows."
    )
    add("")
    add(f"Band at the chart's own window, {_interval_line(days, battery)}.")
    add("")
    add(
        "**A single rolling reading of 1.4 is not evidence of miscalibration.** SPEC.md 6.1 "
        "says so and the arithmetic says why: at `T = 12` monthly observations the normal band "
        f"is +-{validation.normal_band(12, z=battery.normal_band_z).upper - 1:.2f} and the "
        f"exact interval is "
        f"{validation.chi_square_interval(12, level=battery.chi_square_level).lower:.2f}-"
        f"{validation.chi_square_interval(12, level=battery.chi_square_level).upper:.2f}. "
        "Only a persistent level shift is evidence. **The claim survives under both intervals "
        "and the margin does not**: 1.4 sits comfortably inside the normal band and a hair "
        "inside the exact one, and a reading of 1.45 would be inside the band and outside the "
        "exact interval. That is why the exact interval is the one every claim here rests on "
        "-- the normal band's constant is arguable (the sampling standard deviation of a "
        "standard deviation is `sigma/sqrt(2(T-1))`, so a `sqrt(2T)` denominator is equally "
        "defensible), while `(T-1)B^2 ~ chi^2_{T-1}` under the null is not."
    )
    add("")

    add("## Daily and monthly, and why daily is primary")
    add("")
    add(
        "Daily is the primary frequency and it is where the power is: the scored window gives "
        f"`T = {short.window.observations:,}` daily observations against "
        f"`T = {short.runs[pre_vra].monthly.observations:,}` monthly ones, and a 12-observation "
        "window cannot separate `B = 1.0` from `B = 1.5` at all. The monthly leg is reported "
        "alongside because it is what MSCI plots and because a monthly-horizon user is a real "
        "consumer of this model."
    )
    add("")
    add(
        "The monthly construction: portfolio weights are fixed on the first scored date of each "
        "month and held, `R_m` is the sum of that portfolio's daily returns, and `sigma_m` is "
        "the forecast standing when the month opened scaled by the square root of the month's "
        "own number of scored dates. Weights are **not** rebuilt inside a month -- a month's "
        "return is only the return of a portfolio if the portfolio existed for the whole month."
    )
    add("")
    lines = [
        "| Horizon | Variant | family 2 `B` daily | family 2 `B` monthly | family 4 `B` daily "
        "| family 4 `B` monthly |",
        "|---|---|---|---|---|---|",
    ]
    for horizon in HORIZONS:
        for spec in variants(settings):
            item = runs[horizon].runs[spec.name]
            lines.append(
                f"| {horizon} | `{spec.name}` "
                f"| {item.report.median_bias(validation.RANDOM):.4f} "
                f"| {item.monthly.median_bias(validation.RANDOM):.4f} "
                f"| {item.report.median_bias(validation.OPTIMIZED):.4f} "
                f"| {item.monthly.median_bias(validation.OPTIMIZED):.4f} |"
            )
    out.extend(lines)
    add("")
    add(f"Monthly interval, {_interval_line(short.runs[pre_vra].monthly.observations, battery)}.")
    add("")

    add("## Stage attribution -- what each published correction did to `B`")
    add("")
    add(
        "**`B` is reported for the pipeline as specified, and alongside it what each stage "
        "contributed.** This is attribution and not selection: the pipeline stays exactly as "
        "SPEC.md 5 specifies, nothing is chosen on the basis of what `B` comes out as, and the "
        "sign of every stage's effect was put on file before any bias statistic existed."
    )
    add("")
    add(
        "The reason it is needed: W3-P2 measured SPEC.md 5.2's Newey-West correction as **net "
        "negative** on this panel -- median volatility ratio 0.9653 short and 0.9828 long, with "
        "`rates_slope` down to 0.79. A lower forecast raises `B`. So stage 2 pushes the bias "
        "statistic toward the project's own headline claim that risk models understate risk, by "
        "roughly 30x more than W3-P1's zero-mean convention pushes it the other way. That is "
        "not a defect -- it is published methodology applied as specified -- but it means a raw "
        '`B` of 1.15 cannot be read as "the risk model understates by 15%" without knowing '
        "how much of it is a 3.5% volatility haircut from a correction that happened to point "
        "that way."
    )
    add("")
    add(
        "W3-P4 measured the other side. `lambda_F` is 1.0162 short and **0.9204 long**: the "
        "long-horizon model over-forecasts by 8%, which is what a 252-day volatility half-life "
        "does to a panel containing 2008 and 2020 -- slow to come down after a crisis, carrying "
        "crisis variance through the calm that follows. On the long horizon the VRA therefore "
        "scales the forecast **down** by 8% and Newey-West already scaled it down by 1.7%, so "
        "roughly **10% of downward forecast pressure sits in the pipeline before any realised "
        "return is read**, all of it pushing `B` up toward the project's own claim."
    )
    add("")
    for horizon in HORIZONS:
        run_at = runs[horizon]
        add(f"### `{horizon}` horizon")
        add("")
        add(
            "| Stage added | family 4 `B` | change | family 2 `B` | change | clip bound "
            "(family 4) |"
        )
        add("|---|---|---|---|---|---|")
        previous_optimized: float | None = None
        previous_random: float | None = None
        for spec in variants(settings):
            report = run_at.runs[spec.name].report
            optimized = report.median_bias(validation.OPTIMIZED)
            random = report.median_bias(validation.RANDOM)
            optimized_change = (
                "--" if previous_optimized is None else f"{optimized - previous_optimized:+.4f}"
            )
            random_change = "--" if previous_random is None else f"{random - previous_random:+.4f}"
            add(
                f"| {spec.label} | {optimized:.4f} | {optimized_change} "
                f"| {random:.4f} | {random_change} "
                f"| {report.clipped_fraction[validation.OPTIMIZED]:.4%} |"
            )
            if spec.name != "sample":
                previous_optimized, previous_random = optimized, random
        add("")
        add(
            "`lambda_F` (lagged, at the last scored date) "
            + ", ".join(
                f"`a = {tag[1:]}`: {np.sqrt(values[-1]):.4f}"
                for tag, values in run_at.factor_lambda.items()
            )
            + f"; `lambda_S`: {np.sqrt(run_at.specific_lambda[-1]):.4f}."
        )
        add("")

    add("## The identity assets, and what is excluded from what")
    add("")
    add(
        f"`{'` and `'.join(data.excluded)}` regress against the macro factor set at "
        "`R^2 = 1.000` because the proxy **is** the factor (SPEC.md 4.1.1). Their `b_nt` is "
        "pinned at a ratio that cannot understate anything, so averaging them into a "
        f"{len(data.assets)}-asset family-1 aggregate pulls it toward 1 by roughly "
        f"{len(data.excluded)}/{len(data.assets)} of the distance between the true mean bias "
        "and unity."
    )
    add("")
    add(
        "**They are excluded from family 1's aggregate and from nothing else.** They remain in "
        "the asset covariance, so every family-2 and family-4 portfolio still holds them; they "
        "are reported individually below; and, per the W4-P1b amendment, they are also out of "
        "SPEC.md 5.5(d)'s shrinkage target and the specific VRA's cross-section. Removing them "
        "from the model would change the model -- the constraint is about aggregation."
    )
    add("")
    add("| Asset | `B` (`" + pre_vra + "`, short) | in family-1 aggregate |")
    add("|---|---|---|")
    frame = first[pre_vra].report.frame()
    scored_assets = set(frame.index)
    for asset in data.assets:
        if asset in scored_assets:
            add(f"| `{asset}` | {frame.loc[asset, 'B']:.4f} | yes |")
        else:
            add(f"| `{asset}` | excluded -- identity asset | **no** |")
    add("")

    add("## `K/T_eff` across the scored window")
    add("")
    add(
        "SPEC.md 5.1.2's diagnostic, carried rather than used as a filter. The burn-in "
        "criterion above removes only dates where an estimator has *no* answer; the early "
        "scored dates are estimated at a `K/T_eff` that makes them poor, and they are labelled "
        "rather than suppressed."
    )
    add("")
    add("| Horizon | first scored date | `K/T_eff` there | median | last | `K/T_eff` there |")
    add("|---|---|---|---|---|---|")
    for horizon in HORIZONS:
        run_at = runs[horizon]
        ratios = run_at.k_over_realised_t_eff
        add(
            f"| {horizon} | {run_at.dates[0].date()} | {ratios[0]:.4f} "
            f"| {np.median(ratios):.4f} | {run_at.dates[-1].date()} | {ratios[-1]:.4f} |"
        )
    add("")

    add("## Control C1: is family 4's bias the DIAGONAL specific-risk assumption?")
    add("")
    add(
        "Pre-registered in `experiments.md` rows 162-163, with both falsifiers written before "
        "the measurement existed. **The suspect:** the naive sample covariance beats the factor "
        "model on family 4, which is the opposite of the direction SPEC.md 6.4's argument for "
        "factor structure predicts, and the candidate mechanism is that "
        "`Sigma = X F X' + diag(delta^2)` throws away every residual correlation. W4-P1 "
        "measured all 78 residual pairs and found `SPY/IWM` at **-0.5488**, confirming "
        "SPEC.md 5.5's own closing prediction with the opposite sign. A diagonal `Delta` "
        "understates the variance of long-short residual combinations, and `w = Sigma^-1 1` "
        "loads hardest on exactly the directions the model calls lowest-variance."
    )
    add("")
    add(
        "**The test changes the off-diagonal and nothing else**: `diag(delta^2)` becomes "
        "`D_delta R_u D_delta` with `R_u` the EWMA residual correlation at the specific-risk "
        "half-life and `D_delta` SPEC.md 5.5's own shrunk deviations, unchanged to the last "
        "bit. Anything that moved the diagonal too would confound *the residuals are "
        "correlated* with *the specific volatilities are wrong*."
    )
    add("")
    add(
        "| Horizon | family 4 `B` diagonal | family 4 `B` correlated | moved toward 1 "
        "| family 2 `B` diagonal | family 2 `B` correlated | naive `B` | gap closed | verdict |"
    )
    add("|---|---|---|---|---|---|---|---|---|")
    for horizon in HORIZONS:
        control = controls[horizon]
        add(
            f"| {horizon} | {control.diagonal_family_4:.4f} "
            f"| **{control.correlated_family_4:.4f}** "
            f"| {control.moved_toward_one:+.4f} "
            f"| {control.diagonal_family_2:.4f} | {control.correlated_family_2:.4f} "
            f"| {control.sample_family_4:.4f} | {control.gap_closed:.1%} "
            f"| {'**HOLDS**' if control.holds else '**REFUTED**'} |"
        )
    add("")
    add(
        f"Falsifier 1: family 4 moves toward 1 by less than the exact interval's half-width "
        f"at the scored `T` ({controls['short'].half_width:.4f}). Falsifier 2: family 2 moves "
        "by more than the same half-width, which would make the reading about the level of the "
        "forecast rather than about its direction-selectivity."
    )
    add("")
    add("| Horizon | realised min-var vol, naive | diagonal | correlated |")
    add("|---|---|---|---|")
    for horizon in HORIZONS:
        naive, diagonal, correlated = controls[horizon].realised
        add(f"| {horizon} | {naive:.3f} | {diagonal:.3f} | {correlated:.3f} |")
    add("")
    add(
        "**What C1 is not.** It is not a candidate. SPEC.md 5.5 specifies a diagonal `Delta`, "
        "USE4 does, and SPEC.md 15.2's `N ~ 500` equity module makes a full residual covariance "
        "from `T_eff = 727` precisely the singular regime SPEC.md 15.2 uses to argue that a "
        "sample covariance is not a thing that works. The production path is unchanged. The row "
        "is counted `model-config` anyway, on row 137's test and W3-P2's precedent -- the "
        "argument for lowering it is at the row and is declined there."
    )
    add("")
    add("## Realised volatility of the minimum-variance portfolio")
    add("")
    add(
        "SPEC.md 6.5's headline model-comparison metric, and Menchero & Ji's methodological "
        "argument for it: do **not** compare risk models by realised information ratio, it is "
        "far too noisy. Compare by the realised volatility of the minimum-variance portfolios "
        "built from each. Lower is better, and unlike `B` this number cannot be moved by "
        "rescaling the matrix -- a pure level correction changes no min-var weight at all."
    )
    add("")
    add("| Horizon | " + " | ".join(f"`{spec.name}`" for spec in variants(settings)) + " |")
    add("|---" * (len(variants(settings)) + 1) + "|")
    for horizon in HORIZONS:
        add(
            f"| {horizon} | "
            + " | ".join(
                f"{runs[horizon].runs[spec.name].min_var_realised_volatility:.3f}"
                for spec in variants(settings)
            )
            + " |"
        )
    add("")
    add("Basis points per day, realised standard deviation over the scored window.")
    add("")
    return "\n".join(out) + "\n"


def chart(runs: dict[Horizon, HorizonRun], settings: config.Config, path: Path) -> None:
    """``reports/bias_statistics.png`` -- one panel per family, one line per variant."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    battery = settings.model.validation
    horizon: Horizon = "short"
    run_at = runs[horizon]
    # The union over variants, in SPEC.md 6.2's numbering order. Taking the first
    # variant's families would silently drop family 3: `sample` has no factor
    # structure and therefore no eigenfactors.
    present = {name for item in run_at.runs.values() for name in item.rolling}
    families = [name for name in validation.FAMILIES if name in present]
    shown = [spec for spec in variants(settings) if spec.name != "sample"]
    figure, axes = plt.subplots(len(families), 1, figsize=(11, 3.1 * len(families)), sharex=True)
    window = battery.rolling_window_months * settings.model.data.trading_days_per_month
    normal = validation.normal_band(window, z=battery.normal_band_z)
    exact = validation.chi_square_interval(window, level=battery.chi_square_level)

    for axis, family in zip(np.atleast_1d(axes), families, strict=True):
        for spec in shown:
            rolling = run_at.runs[spec.name].rolling.get(family)
            if rolling is None:
                continue
            axis.plot(rolling.index, rolling.values[:, 0], linewidth=0.9, label=spec.name)
        sample_rolling = run_at.runs["sample"].rolling.get(family)
        if sample_rolling is not None:
            axis.plot(
                sample_rolling.index,
                sample_rolling.values[:, 0],
                linewidth=1.1,
                linestyle=":",
                color="black",
                label="sample (naive)",
            )
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.axhspan(normal.lower, normal.upper, color="grey", alpha=0.13, linewidth=0)
        axis.axhline(exact.lower, color="crimson", linewidth=0.7, linestyle="--")
        axis.axhline(exact.upper, color="crimson", linewidth=0.7, linestyle="--")
        axis.set_ylabel("B")
        axis.set_title(
            f"{family} -- rolling {battery.rolling_window_months}m "
            f"(T = {window}, {window - 1} of {window} overlap)",
            fontsize=9,
        )
        axis.set_ylim(0.0, 2.2)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False)
    figure.suptitle(
        f"SPEC.md 6.2 bias statistics, {horizon} horizon -- grey: normal band "
        f"1 +- {battery.normal_band_z}/sqrt(T) (display); red dashes: exact chi-square "
        f"{battery.chi_square_level:.0%} interval (decides)",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.97))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - entry point
    parser = argparse.ArgumentParser(description="SPEC.md 6.1 and 6.2's validation battery")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the committed cache")
    parser.add_argument("--column", default=None, help="build one horizon's partial (fan-out)")
    args = parser.parse_args(list(argv) if argv is not None else None)
    settings = config.load()
    data = inputs(settings)
    if args.column is not None:
        build_cache(data, settings, only=args.column)
        return 0
    if args.rebuild:
        build_cache(data, settings)
        return 0
    cache = read_cache(data, settings)
    runs = {horizon: run(horizon, data, cache, settings) for horizon in HORIZONS}
    controls = {
        horizon: residual_correlation_control(horizon, data, cache, settings, runs[horizon])
        for horizon in HORIZONS
    }
    _MARKDOWN_PATH.write_text(render(runs, controls, data, settings), encoding="utf-8")
    chart(runs, settings, _CHART_PATH)
    print(f"wrote {_MARKDOWN_PATH}")
    print(f"wrote {_CHART_PATH}")
    for horizon in HORIZONS:
        print(f"  {runs[horizon].window.render()}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
