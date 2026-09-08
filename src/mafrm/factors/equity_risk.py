"""The equity ``(X, f, u)`` through the UNCHANGED risk pipeline. SPEC.md 15.2, 15.7; W7-P4.

WHY THIS LIVES IN ``factors/`` AND NOT IN ``risk/``
---------------------------------------------------

The same reason as :mod:`mafrm.factors.bias_report`. Every estimator here is
:mod:`mafrm.risk`'s, called through the signatures SPEC.md 15.2 fixes; *this*
module knows that a column is an industry that was absent on some dates, that a
name was alone in its industry, that a ticker had no bar after it left the
index, and that a bucket is a size decile. None of that may cross into
``risk/`` (CLAUDE.md invariant 10), and the session's acceptance is the empty
diff under ``src/mafrm/risk/``.

THE FIVE RULINGS (SPEC.md 15.7.1)
----------------------------------

1. **The T x K frame.** Never-present industries are dropped from ``K``;
   intermittent ones stay and their absent dates are zero-filled, with the
   understatement ``1 - f_k`` reported (:func:`factor_frame`).
2. **Size deciles.** Ten equal-count buckets by screened cap at ``t-1``,
   recomputed each date (:func:`design_at`).
3. **Per-name residual histories, gaps removed**, through per-name calls into
   :func:`mafrm.risk.specific.time_series_estimate` (:func:`specific_at`).
4. **The month-end grid**, weights held through the following month
   (:func:`month_end_positions`, :func:`score_variant`).
5. **Scope**: no residual PCs, no MP comparison here.

THE LEAD-LAG, ASSERTED
----------------------

The forecast made at the close of month-end ``d`` (panel position ``p``) is
built from panel rows ``0..p`` inclusive -- the return over ``d`` is known at
the close of ``d`` -- and scored on rows ``p+1..p_next``. Every construction
site below indexes ``[: p + 1]`` for the forecast and ``p + 1 : p_next + 1`` for
the realisation, and ``tests/test_equity_risk.py`` perturbs one realised return
and requires that no forecast moves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from mafrm import config, history
from mafrm.factors import equity, equity_regression
from mafrm.factors.bias_report import VariantSpec, scaling_tag
from mafrm.risk import specific, validation
from mafrm.risk.config import Horizon, RiskConfig
from mafrm.risk.covariance import ewma_second_moment, run_pipeline
from mafrm.risk.regime import rolling_multiplier

__all__ = [
    "Design",
    "FactorFrame",
    "HorizonColumn",
    "Panel",
    "Scored",
    "SpecificAt",
    "StageBuild",
    "build_history",
    "component_split",
    "design_at",
    "embed",
    "factor_frame",
    "load_history",
    "month_end_positions",
    "panel",
    "present_block",
    "regime_multipliers",
    "score_variant",
    "specific_at",
    "stage_builds",
    "to_monthly",
    "unpack",
    "variant_ingredients",
]

#: Per-horizon partials of SPEC.md 5.3's Monte Carlo on the equity frame.
#: Gitignored (``data/processed/``): about nine minutes a horizon to rebuild,
#: digest-guarded, and everything a reader needs is in the committed reports.
PARTIALS: Final[Path] = Path(__file__).resolve().parents[3] / "data" / "processed" / "equity_bias"

#: The cheap stages, in ``covariance.stages`` order, that the ladder reads.
CHEAP_STAGES: Final[tuple[str, ...]] = ("ewma", "newey_west", "psd_repair")

#: Family-4 on the always-present subset (SPEC.md 15.7.1 construction 7).
MIN_VAR_SUBSET: Final[str] = "min_var_subset"
#: The extra family key the subset leg is reported under.
OPTIMIZED_SUBSET: Final[str] = "minimum_variance_subset"


class EquityRiskError(ValueError):
    """The equity panel could not be dressed for SPEC.md 15.2's signatures."""


# ---------------------------------------------------------------------------
# Ruling 1 -- the T x K frame
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorFrame:
    """``f`` as ``risk/`` receives it: ``T x K``, complete, in basis points per day."""

    frame: pd.DataFrame
    #: Per KEPT industry: the fraction of dates it was present. 1.0 means always.
    presence: pd.Series
    #: Industries dropped because they were never present (ruling 1).
    dropped: tuple[str, ...]
    industry_columns: tuple[str, ...]
    style_columns: tuple[str, ...]
    country_column: str
    #: Per KEPT industry: the panel position of its first present date.
    first_present: pd.Series

    @property
    def factors(self) -> int:
        """``K`` at the end of the sample: every industry that ever appeared."""
        return int(self.frame.shape[1])

    def present_columns(self, position: int) -> np.ndarray:
        """The factors on the universe AS KNOWN at ``position`` (ruling 1, point in time).

        The country, the styles, and every kept industry that has appeared at
        least once by ``position``. An industry that has not yet appeared has an
        all-zero history, which ``risk/`` refuses as a zero variance -- correctly:
        there is no observation to estimate from. Reading the end-of-sample
        industry list at an earlier date would also be a look-ahead in the
        design. ``K_d`` therefore grows through the sample and the W8 ``K/T``
        points read it per date, as ruling 1's last sentence says.
        """
        first = self.first_present
        keep = [
            i
            for i, name in enumerate(self.frame.columns)
            if str(name) not in first.index or int(first[str(name)]) <= position
        ]
        return np.asarray(keep, dtype=int)

    def factors_at(self, position: int) -> int:
        return int(self.present_columns(position).size)

    @property
    def intermittent(self) -> tuple[str, ...]:
        """Kept industries absent on at least one date -- the zero-filled ones."""
        return tuple(str(name) for name, value in self.presence.items() if value < 1.0)

    @property
    def understatement(self) -> pd.Series:
        """``1 - f_k``: the share of an intermittent industry's EWMA variance the zero-fill "
        "removes."""
        return (1.0 - self.presence[list(self.intermittent)]).rename("understatement")

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.frame.index)


def factor_frame(
    returns: pd.DataFrame,
    *,
    industry_columns: Sequence[str],
    style_columns: Sequence[str],
    country_column: str,
    scale: float,
) -> FactorFrame:
    """Ruling 1. Drop never-present industries, zero-fill the intermittent ones, scale.

    The zero-fill is exact in one sense and biased in one: on an absent date no
    name is exposed to the factor, so a zero return is consistent with every
    realised asset return, and the EWMA variance of the factor is understated by
    the weight the absent dates carry -- ``1 - f_k`` in the equal-weight limit,
    which is what :attr:`FactorFrame.understatement` reports.
    """
    columns = [country_column, *industry_columns, *style_columns]
    missing = [name for name in columns if name not in returns.columns]
    if missing:
        raise EquityRiskError(f"factor_frame: {missing} are not factor columns")
    if returns.empty:
        raise EquityRiskError("factor_frame: an empty factor panel")
    named = returns[[country_column, *style_columns]]
    if not np.all(np.isfinite(named.to_numpy(dtype=float))):
        raise EquityRiskError(
            "factor_frame: the country or a style factor has a missing return. Only an "
            "industry can be absent on a date (SPEC.md 15.6.1 ruling 3); anything else is a "
            "defect upstream."
        )
    industries = returns[list(industry_columns)]
    present = industries.notna()
    presence = present.mean(axis=0)
    dropped = tuple(str(name) for name, value in presence.items() if value == 0.0)
    kept = tuple(str(name) for name in industry_columns if name not in set(dropped))
    frame = pd.concat(
        [
            returns[[country_column]],
            industries[list(kept)].fillna(0.0),
            returns[list(style_columns)],
        ],
        axis=1,
    ).astype(float) * float(scale)
    frame.index = pd.DatetimeIndex(frame.index)
    first = pd.Series(
        {name: int(np.argmax(present[name].to_numpy())) for name in kept}, dtype=int
    ).rename("first_present")
    return FactorFrame(
        frame=frame,
        presence=presence[list(kept)].astype(float).rename("presence"),
        dropped=dropped,
        industry_columns=kept,
        style_columns=tuple(str(name) for name in style_columns),
        country_column=str(country_column),
        first_present=first,
    )


# ---------------------------------------------------------------------------
# The panel -- (X, f, u) plus what the families need
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Panel:
    """The equity ``(X, f, u)`` on the regression dates, in basis points per day.

    Row ``t`` of :attr:`returns` and :attr:`residuals` is the realisation ON
    ``dates[t]``; row ``t`` of :attr:`cap`, :attr:`styles` and :attr:`industry`
    is what was known at the CLOSE of ``dates[t]`` -- the design the next
    session's regression reads. A forecast made at position ``p`` therefore
    takes the design at ``p`` and factor rows ``[: p + 1]``.
    """

    factors: FactorFrame
    #: ``T x N`` realised excess returns, NaN where the name has no bar.
    returns: pd.DataFrame
    #: ``T x N`` regression residuals, NaN outside the regression set.
    residuals: pd.DataFrame
    #: ``T x N`` screened cap at the close, NaN without one.
    cap: pd.DataFrame
    #: Style -> ``T x N`` standardized exposure at the close.
    styles: dict[str, pd.DataFrame]
    #: ``T x N`` point-in-time FF49 industry id at the close, ``-1`` without one.
    industry: pd.DataFrame
    #: Industry id -> column label, for the ids in :attr:`FactorFrame.industry_columns`.
    industry_label: dict[int, str]
    #: The regression's per-date summary (``cond_style`` is read beside every stage).
    summary: pd.DataFrame
    #: Names with a return on every regression date (construction 7).
    always_present: tuple[str, ...]
    sample_start: pd.Timestamp
    holdout_start: pd.Timestamp

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.returns.index)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.returns.columns)

    def alone(self) -> pd.DataFrame:
        """``T x N`` boolean: the name is the only member of its industry at that close."""
        ids = self.industry.to_numpy(dtype=int)
        out = np.zeros(ids.shape, dtype=bool)
        for row in range(ids.shape[0]):
            present = ids[row] >= 0
            values, counts = np.unique(ids[row][present], return_counts=True)
            singles = set(values[counts == 1].tolist())
            out[row] = present & np.isin(ids[row], list(singles))
        return pd.DataFrame(out, index=self.industry.index, columns=self.industry.columns)


def panel(
    cfg: config.Config | None = None,
    *,
    built: tuple[equity_regression.FactorReturns, equity.Exposures, equity.Inputs] | None = None,
) -> Panel:
    """Dress W7-P3b's regression output for SPEC.md 15.2. Reads the cache only."""
    cfg = cfg or config.load()
    settings = cfg.model.equity_risk
    factors, exposures, inputs = built if built is not None else equity_regression.build(cfg)
    labels = {int(k): str(v) for k, v in factors.labels.items()}
    frame = factor_frame(
        factors.returns,
        industry_columns=factors.industry_columns,
        style_columns=factors.style_factors,
        country_column=equity_regression.COUNTRY,
        scale=settings.basis_points_per_unit,
    )
    dates = frame.dates
    with_cap = exposures.cap.notna().any(axis=0)
    names = pd.Index([str(t) for t in exposures.cap.columns[with_cap.to_numpy()]])
    scale = settings.basis_points_per_unit
    returns = inputs.excess.reindex(index=dates, columns=names).astype(float) * scale
    residuals = factors.residuals.reindex(index=dates, columns=names).astype(float) * scale
    cap = exposures.cap.reindex(index=dates, columns=names).astype(float)
    styles = {
        str(name): exposures.exposures[name].reindex(index=dates, columns=names).astype(float)
        for name in factors.style_factors
    }
    pit = factors.industries_pit.reindex(index=dates, columns=names)
    industry = pd.DataFrame(
        np.where(pit.notna().to_numpy(), pit.fillna(-1).to_numpy(dtype=int), -1),
        index=dates,
        columns=names,
    )
    kept_ids = {i for i, label in labels.items() if label in set(frame.industry_columns)}
    industry = industry.where(industry.isin(list(kept_ids)) | (industry < 0), -1)
    always = tuple(str(n) for n in names[np.isfinite(returns.to_numpy(dtype=float)).all(axis=0)])
    return Panel(
        factors=frame,
        returns=returns,
        residuals=residuals,
        cap=cap,
        styles=styles,
        industry=industry,
        industry_label={i: labels[i] for i in kept_ids},
        summary=factors.summary.reindex(dates),
        always_present=always,
        sample_start=factors.sample_start,
        holdout_start=factors.holdout_start,
    )


# ---------------------------------------------------------------------------
# Ruling 4 -- the month-end grid
# ---------------------------------------------------------------------------


def month_end_positions(dates: pd.DatetimeIndex) -> np.ndarray:
    """Positions of the LAST date in each calendar month. The forecast grid."""
    if len(dates) == 0:
        raise EquityRiskError("month_end_positions: no dates")
    period = dates.to_period("M")
    last = np.ones(len(dates), dtype=bool)
    last[:-1] = period[:-1] != period[1:]
    return np.flatnonzero(last)


# ---------------------------------------------------------------------------
# Ruling 2 -- the design at a date, with its size deciles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Design:
    """``X`` at one close, and everything SPEC.md 5.5 and 6.2 need beside it."""

    date: pd.Timestamp
    position: int
    names: tuple[str, ...]
    #: ``N x K`` in the factor frame's column order.
    exposures: np.ndarray
    cap: np.ndarray
    industry_id: np.ndarray
    #: Alone in its industry at this close (construction 5).
    singleton: np.ndarray
    #: Opaque decile labels, one per name (ruling 2).
    buckets: tuple[str, ...]
    #: Positions of ``names`` in the panel's column order.
    columns: np.ndarray
    #: Names whose industry has not yet entered ``K_d`` at this close (its first
    #: regression is the next session); their industry exposure meets a zero
    #: factor variance for that one month. Counted, reported.
    unentered: np.ndarray

    @property
    def assets(self) -> int:
        return len(self.names)


def size_deciles(cap: np.ndarray, *, buckets: int) -> tuple[str, ...]:
    """Ruling 2: equal-count buckets by cap. Rank ``r`` of ``N`` goes to ``floor(r * B / N)``."""
    if buckets < 1:
        raise EquityRiskError("size_deciles: at least one bucket")
    if np.any(~np.isfinite(cap)) or np.any(cap <= 0.0):
        raise EquityRiskError("size_deciles: every cap must be positive and finite")
    order = np.argsort(np.argsort(cap, kind="stable"), kind="stable")
    labels = (order * buckets) // cap.size
    return tuple(f"size_{int(label)}" for label in labels)


def design_at(data: Panel, position: int, *, buckets: int) -> Design:
    """The names known at the close of ``dates[position]`` and their design.

    Construction 2: a screened cap, an industry in ``K`` and all six exposures
    at the close; membership does not condition on the next session's return.
    """
    cap = data.cap.to_numpy(dtype=float)[position]
    ids = data.industry.to_numpy(dtype=int)[position]
    styles = np.stack(
        [data.styles[name].to_numpy(dtype=float)[position] for name in data.factors.style_columns],
        axis=1,
    )
    use = np.isfinite(cap) & (cap > 0.0) & (ids >= 0) & np.isfinite(styles).all(axis=1)
    columns = np.flatnonzero(use)
    if columns.size < 2:
        raise EquityRiskError(f"design_at: {columns.size} name(s) at {data.dates[position]!s}")
    industry_index = {label: i for i, label in enumerate(data.factors.industry_columns)}
    k = data.factors.factors
    n_industries = len(data.factors.industry_columns)
    x = np.zeros((columns.size, k))
    x[:, 0] = 1.0
    for row, column in enumerate(columns):
        x[row, 1 + industry_index[data.industry_label[int(ids[column])]]] = 1.0
    x[:, 1 + n_industries :] = styles[columns]
    chosen_ids = ids[columns]
    values, counts = np.unique(chosen_ids, return_counts=True)
    singles = values[counts == 1]
    all_names = data.names
    names = tuple(all_names[int(c)] for c in columns)
    first = data.factors.first_present
    unentered = np.asarray(
        [int(first[data.industry_label[int(i)]]) > position for i in chosen_ids], dtype=bool
    )
    return Design(
        date=pd.Timestamp(data.dates[position]),
        position=int(position),
        names=names,
        exposures=x,
        cap=cap[columns],
        industry_id=chosen_ids,
        singleton=np.isin(chosen_ids, singles),
        buckets=size_deciles(cap[columns], buckets=buckets),
        columns=columns,
        unentered=unentered,
    )


# ---------------------------------------------------------------------------
# The cheap stages, per month-end -- rebuilt on every render
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageBuild:
    """SPEC.md 5.1-5.2 at one month-end, with every PSD check beside the style block's condition."""

    position: int
    date: pd.Timestamp
    observations: int
    matrices: dict[str, np.ndarray]
    minimum_eigenvalues: dict[str, float]
    repair_fired: bool
    repair_floored: int
    bartlett_fired: bool
    k_over_t: float
    #: The regression's style-block condition number on this date (row 288(b)).
    style_condition: float
    #: ``K_d``: the factors present on the universe as known at this close.
    factors: int


def stage_builds(risk: RiskConfig, data: Panel, positions: Sequence[int]) -> list[StageBuild]:
    """One :func:`run_pipeline` per month-end, stopped after the repair. Seconds in total."""
    frame = data.factors.frame
    cond = data.summary["cond_style"].to_numpy(dtype=float)
    out: list[StageBuild] = []
    size = data.factors.factors
    for position in positions:
        columns = data.factors.present_columns(int(position))
        build = run_pipeline(frame.iloc[: position + 1, columns], risk, stop_after="psd_repair")
        if build.observations != position + 1:  # pragma: no cover - a construction check
            raise EquityRiskError("stage_builds: the build saw the wrong number of rows")
        repairs = build.repairs
        out.append(
            StageBuild(
                position=int(position),
                date=pd.Timestamp(frame.index[position]),
                observations=build.observations,
                matrices={stage.name: embed(stage.matrix, columns, size) for stage in build.stages},
                minimum_eigenvalues={
                    stage.name: float(stage.minimum_eigenvalue) for stage in build.stages
                },
                repair_fired=build.repair_fired,
                repair_floored=int(sum(r.floored for r in repairs)),
                bartlett_fired=build.bartlett_fired,
                k_over_t=float(build.k_over_realised_t_eff),
                style_condition=float(cond[position]),
                factors=int(columns.size),
            )
        )
    return out


# ---------------------------------------------------------------------------
# SPEC.md 5.3 -- the one expensive stage, cached per horizon
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HorizonColumn:
    horizon: Horizon

    @property
    def name(self) -> str:
        return self.horizon


def _upper_names(prefix: str, size: int) -> list[str]:
    return [f"{prefix}_{row}_{column}" for row in range(size) for column in range(row, size)]


def pack(matrix: np.ndarray) -> np.ndarray:
    size = matrix.shape[0]
    rows, cols = np.triu_indices(size)
    return np.asarray(matrix[rows, cols], dtype=float)


def unpack(values: np.ndarray, size: int) -> np.ndarray:
    rows, cols = np.triu_indices(size)
    matrix = np.zeros((size, size))
    matrix[rows, cols] = values
    matrix[cols, rows] = values
    return matrix


def embed(matrix: np.ndarray, columns: np.ndarray, size: int) -> np.ndarray:
    """A ``K_d x K_d`` matrix in the full ``K x K`` layout, zero outside its factors.

    Exact for everything downstream: a not-yet-present industry has no name
    exposed to it, so ``X F X'`` is unchanged by the zero rows and columns, and
    family 3 reads the present block back through the zero diagonal.
    """
    out = np.zeros((size, size))
    out[np.ix_(columns, columns)] = matrix
    return out


def present_block(matrix: np.ndarray) -> np.ndarray:
    """The factors a full-layout matrix actually carries: a positive diagonal."""
    return np.flatnonzero(np.diag(matrix) > 0.0)


def _padded(values: np.ndarray, size: int) -> np.ndarray:
    out = np.full(size, np.nan)
    out[: values.size] = values
    return out


def history_columns(size: int, scalings: Sequence[float]) -> list[str]:
    names = [name for a in scalings for name in _upper_names(scaling_tag(a), size)]
    names += [f"lambda_raw_{k}" for k in range(size)]
    names += [f"lambda_fit_{k}" for k in range(size)]
    return [*names, "amplitude", "k_over_t", "factors"]


def build_history(
    risk: RiskConfig,
    factors: FactorFrame,
    positions: Sequence[int],
    *,
    label: str,
    progress: bool = True,
) -> pd.DataFrame:
    """SPEC.md 5.3 at every month-end: both published ``a`` from ONE simulation per date.

    Also keeps the raw and fitted ``lambda(k)`` curves, because rows 293(a)-(b)
    read them and the adjustment object is not otherwise retained.
    """
    scalings = risk.eigenfactor_scaling
    frame = factors.frame
    size = factors.factors
    rows: list[np.ndarray] = []
    dates: list[pd.Timestamp] = []
    for count, position in enumerate(positions, start=1):
        columns = factors.present_columns(int(position))
        build = run_pipeline(
            frame.iloc[: position + 1, columns],
            risk.for_scaling(scalings[0]),
            stop_after="eigenfactor",
        )
        if build.observations != position + 1:
            raise EquityRiskError(
                f"build_history: the build for {frame.index[position]!s} saw {build.observations} "
                f"rows at position {position}; the forecast made at the close of d must be made "
                "from rows through d and nothing after -- the look-ahead check has failed."
            )
        adjustment = build.eigenfactor
        if adjustment is None:  # pragma: no cover - stop_after guarantees it ran
            raise EquityRiskError("build_history: the eigenfactor stage did not run")
        packed = [pack(embed(adjustment.variant(a).adjusted, columns, size)) for a in scalings]
        rows.append(
            np.concatenate(
                [
                    *packed,
                    _padded(adjustment.bias, size),
                    _padded(adjustment.fit.fitted, size),
                    [adjustment.amplitude, adjustment.k_over_t, float(columns.size)],
                ]
            )
        )
        dates.append(pd.Timestamp(frame.index[position]))
        if progress and count % 25 == 0:
            print(f"  {label}: {count:,} / {len(positions):,}", flush=True)
    return pd.DataFrame(
        np.asarray(rows),
        index=pd.DatetimeIndex(dates, name="date"),
        columns=history_columns(size, scalings),
    )


def history_digests(
    risk: RiskConfig, factors: FactorFrame, settings: config.Config
) -> dict[str, str]:
    return {
        "risk_config_digest": history.risk_config_digest(settings),
        "panel_digest": history.panel_digest(factors.frame),
        "horizon": risk.horizon,
        "layout": "point_in_time_k_v1",
    }


def load_history(
    risk: RiskConfig,
    factors: FactorFrame,
    positions: Sequence[int],
    settings: config.Config,
    *,
    progress: bool = True,
) -> pd.DataFrame:
    """The partial for one horizon: reused when its digests match, rebuilt otherwise."""
    column = HorizonColumn(risk.horizon)
    return history.load_or_build(
        history.partial_path(PARTIALS, column),
        history_digests(risk, factors, settings),
        lambda: build_history(risk, factors, positions, label=column.name, progress=progress),
    )


# ---------------------------------------------------------------------------
# Ruling 3 -- SPEC.md 5.5 per month-end, per-name histories with gaps removed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecificAt:
    """SPEC.md 5.5's four legs at one close, pre-VRA, plus the registered diagnostics."""

    design: Design
    #: ``sigma_SH`` per name, basis points per day, before the specific VRA.
    deviation: np.ndarray
    time_series: np.ndarray
    structural: np.ndarray
    #: Observed residuals per name at this close.
    observations: np.ndarray
    #: Names with no leg (a) estimate (construction 4), forecast by leg (b).
    short_history: np.ndarray
    #: Names whose every observed residual is exactly zero (no time-series information).
    exact_zero: np.ndarray
    #: Cross-sectional percentile rank (0-100) of ``deviation``.
    percentile: np.ndarray
    #: ``Z_n`` of SPEC.md 5.5(c), reported, nothing routed on it.
    z: np.ndarray
    structural_r_squared: float
    structural_degrees_of_freedom: int
    bartlett_fallbacks: int
    #: Singletons whose ``sigma_TS`` is the cross-sectional minimum (row 294(a)).
    singletons_at_floor: int
    #: How far holding singletons out moved their buckets' targets (largest relative move).
    target_shift: float

    @property
    def variance(self) -> np.ndarray:
        # ``np.square`` is typed as returning ``Any`` under the 3.11 stub
        # resolution, so the annotation is carried explicitly rather than
        # inferred -- mypy's ``no-any-return`` fires on 3.11 otherwise while
        # staying silent on 3.12, which is how this reached CI unnoticed.
        squared: np.ndarray = np.square(self.deviation)
        return squared


def specific_at(
    risk: RiskConfig,
    residuals: np.ndarray,
    design: Design,
    *,
    singleton_structural: bool = False,
) -> SpecificAt:
    """Ruling 3 and constructions 4-5, through ``risk/specific``'s public legs.

    ``singleton_structural`` is SPEC.md 15.7.3's rule (W7-P4b): a name alone in
    its industry has a residual that is ~0 by the regression's identification --
    not an observation of specific risk -- so it has no leg (a) estimate and
    takes leg (b)'s, exactly as a name below the arithmetic floor does. Its
    ``sigma_TS`` is still computed and carried for the report.

    ``residuals`` is the panel's ``T x N_all`` array; only rows ``[: position + 1]``
    and the design's columns are read. Leg (a) runs per name on that name's
    observed residuals with the gaps removed -- the estimator ``risk/`` exposes,
    on the series the ruling names -- and legs (b)-(d) run on the cross-section
    exactly as :func:`mafrm.risk.specific.build_specific_risk` composes them.
    """
    floor = risk.specific_newey_west_lags + 1
    window = residuals[: design.position + 1][:, design.columns]
    assets = design.assets
    sigma_ts = np.full(assets, np.nan)
    z = np.full(assets, np.nan)
    counts = np.zeros(assets, dtype=int)
    exact_zero = np.zeros(assets, dtype=bool)
    fallbacks = 0
    for n in range(assets):
        column = window[:, n]
        observed = column[np.isfinite(column)]
        counts[n] = observed.size
        if observed.size < floor:
            continue
        if not np.any(observed != 0.0):
            exact_zero[n] = True
            continue
        estimate = specific.time_series_estimate(
            observed[:, None],
            halflife=float(risk.specific_volatility_halflife),
            autocorrelation_halflife=float(risk.specific_autocorrelation_halflife),
            lags=risk.specific_newey_west_lags,
        )
        sigma_ts[n] = float(estimate.deviation[0])
        fallbacks += int(estimate.fallback.fired)
        z[n] = float(
            specific.blend_weight(
                observed[:, None],
                estimate.deviation,
                halflife=float(risk.specific_volatility_halflife),
            ).z[0]
        )
    has_ts = np.isfinite(sigma_ts) & (sigma_ts > 0.0)
    if singleton_structural:
        has_ts &= ~design.singleton
    if int(has_ts.sum()) <= design.exposures.shape[1] + 1:
        raise EquityRiskError(
            f"specific_at: {int(has_ts.sum())} name(s) with a time-series estimate at "
            f"{design.date.date()} cannot identify SPEC.md 5.5(b)'s "
            f"{design.exposures.shape[1] + 1} coefficients"
        )
    structural = specific.structural_fit(
        np.where(has_ts, sigma_ts, 1.0), design.exposures, fit_set=has_ts
    )
    blended = np.where(has_ts, sigma_ts, structural.deviation)
    target = ~design.singleton
    shrinkage = specific.bayesian_shrinkage(
        blended, design.buckets, q=risk.bayesian_shrinkage_q, target_set=target
    )
    unrestricted = specific.bucket_statistics(blended, design.buckets)
    shift = float(np.max(np.abs(shrinkage.buckets.mean / unrestricted.mean - 1.0)))
    deviation = shrinkage.deviation
    order = np.argsort(np.argsort(deviation, kind="stable"), kind="stable")
    percentile = 100.0 * order / max(assets - 1, 1)
    at_floor = int(np.count_nonzero(design.singleton & has_ts & (sigma_ts == np.nanmin(sigma_ts))))
    return SpecificAt(
        design=design,
        deviation=np.asarray(deviation, dtype=float),
        time_series=sigma_ts,
        structural=np.asarray(structural.deviation, dtype=float),
        observations=counts,
        short_history=~has_ts,
        exact_zero=exact_zero,
        percentile=percentile,
        z=z,
        structural_r_squared=float(structural.r_squared),
        structural_degrees_of_freedom=int(structural.residual_degrees_of_freedom),
        bartlett_fallbacks=fallbacks,
        singletons_at_floor=at_floor,
        target_shift=shift,
    )


# ---------------------------------------------------------------------------
# SPEC.md 5.4 on the held forecast -- daily B, published half-lives in days
# ---------------------------------------------------------------------------


def _held_days(positions: Sequence[int], index: int) -> tuple[int, int]:
    """Panel rows ``(start, stop)`` the forecast at ``positions[index]`` is held over."""
    return int(positions[index]) + 1, int(positions[index + 1]) + 1


def regime_multipliers(
    risk: RiskConfig,
    data: Panel,
    positions: Sequence[int],
    factor: Sequence[np.ndarray],
    specifics: Sequence[SpecificAt],
) -> tuple[np.ndarray, np.ndarray]:
    """``lambda_F^2`` and ``lambda_S^2`` standing at each month-end (construction 6).

    ``positions[i]`` is the ``i``-th forecast date; ``factor[i]`` its ``K x K``
    factor covariance and ``specifics[i]`` its specific build. The daily
    ``B_t`` run over the sessions each forecast is held, the EWMA runs over
    that daily history at the published half-life in DAYS, and element ``i``
    is the multiplier available at the close of ``positions[i]`` -- NaN for the
    first forecast, which has no history behind it. Lagged by construction:
    nothing scored under forecast ``i`` entered multiplier ``i``.
    """
    if len(factor) != len(positions) or len(specifics) != len(positions):
        raise EquityRiskError("regime_multipliers: one forecast per position")
    returns = data.factors.frame.to_numpy(dtype=float)
    residuals = data.residuals.to_numpy(dtype=float)
    daily_factor: list[np.ndarray] = []
    daily_specific: list[np.ndarray] = []
    for i in range(len(positions) - 1):
        start, stop = _held_days(positions, i)
        present = present_block(factor[i])
        if present.size == 0:
            raise EquityRiskError("regime_multipliers: a factor forecast with no factor in it")
        sigma_k = np.sqrt(np.diag(factor[i])[present])
        daily_factor.append(
            np.sqrt(np.mean(np.square(returns[start:stop][:, present] / sigma_k), axis=1))
        )
        build = specifics[i]
        include = ~build.design.singleton
        columns = build.design.columns[include]
        sigma_n = build.deviation[include]
        block = residuals[start:stop][:, columns] / sigma_n
        with np.errstate(invalid="ignore"):
            daily_specific.append(np.sqrt(np.nanmean(np.square(block), axis=1)))
    if not daily_factor:
        return np.full(len(positions), np.nan), np.full(len(positions), np.nan)
    factor_bias = np.concatenate(daily_factor)
    specific_bias = np.concatenate(daily_specific)
    if not np.all(np.isfinite(specific_bias)):
        raise EquityRiskError("regime_multipliers: a session with no residual in the cross-section")
    lambda_f = rolling_multiplier(factor_bias, halflife=risk.volatility_regime_halflife)
    lambda_s = rolling_multiplier(specific_bias, halflife=risk.volatility_regime_halflife)
    ends = np.cumsum(
        [
            stop - start
            for start, stop in (_held_days(positions, i) for i in range(len(positions) - 1))
        ]
    )
    out_f = np.full(len(positions), np.nan)
    out_s = np.full(len(positions), np.nan)
    out_f[1:] = lambda_f[ends - 1]
    out_s[1:] = lambda_s[ends - 1]
    return out_f, out_s


# ---------------------------------------------------------------------------
# SPEC.md 6.2's families, daily-held on the month-end grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scored:
    """One variant's daily-held forecasts and realisations on the scored window."""

    spec: VariantSpec
    #: Sessions x members: the forecast daily volatility standing on each session.
    forecasts: pd.DataFrame
    #: Sessions x members: the realised return on each session.
    realised: pd.DataFrame
    portfolios: validation.PortfolioSet
    #: Per scored forecast: family-4 weights (full universe) and their names.
    min_var_weights: tuple[np.ndarray, ...]
    #: Per scored forecast: family-2's draws, kept only when ``keep_random`` was asked.
    random_weights: tuple[np.ndarray, ...]
    #: ``(name, session)`` cells whose return was missing and taken as zero.
    liquidated_cells: int
    #: The month each session belongs to, as the index of the forecast held.
    month_of_session: np.ndarray


def variant_ingredients(
    spec: VariantSpec,
    *,
    stage: dict[str, np.ndarray],
    eigen: dict[str, np.ndarray],
    specific_variance: np.ndarray,
    lambda_f: dict[str, float],
    lambda_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``(F, delta^2)`` under one ladder variant -- the same accounting as the macro ladder."""
    if spec.stage == "sample":
        raise EquityRiskError("variant_ingredients: the sample variant has no factor structure")
    if spec.stage == "eigenfactor":
        assert spec.scaling is not None
        matrix = eigen[scaling_tag(spec.scaling)]
    else:
        matrix = stage[spec.stage]
    if spec.factor_regime:
        assert spec.scaling is not None
        matrix = matrix * lambda_f[scaling_tag(spec.scaling)]
    variance = specific_variance * lambda_s if spec.specific_regime else specific_variance
    return np.asarray(matrix, dtype=float), np.asarray(variance, dtype=float)


def _forecast_matrix(
    spec: VariantSpec,
    *,
    design: Design,
    stage: dict[str, np.ndarray],
    eigen: dict[str, np.ndarray],
    specific_variance: np.ndarray,
    lambda_f: dict[str, float],
    lambda_s: float,
) -> validation.RiskForecast:
    matrix, variance = variant_ingredients(
        spec,
        stage=stage,
        eigen=eigen,
        specific_variance=specific_variance,
        lambda_f=lambda_f,
        lambda_s=lambda_s,
    )
    return validation.RiskForecast.from_factor_model(design.exposures, matrix, variance)


def score_variant(
    spec: VariantSpec,
    *,
    data: Panel,
    positions: Sequence[int],
    scored: Sequence[int],
    designs: Sequence[Design],
    stages: Sequence[dict[str, np.ndarray]],
    eigen: Sequence[dict[str, np.ndarray]],
    specifics: Sequence[SpecificAt],
    lambda_f: dict[str, np.ndarray],
    lambda_s: np.ndarray,
    subset: Sequence[str],
    random_portfolios: int,
    seed: int,
    volatility_halflife: float,
    keep_random: bool = False,
) -> Scored:
    """SPEC.md 6.2's four families under one variant, weights held across each month.

    ``positions`` are the forecast dates with a forecast, ``scored`` the indices
    into them that the common window keeps. The ``sample`` variant lives on the
    always-present ``subset`` (construction 7); every factor variant carries an
    extra family-4 member on that subset for the like-for-like comparison.
    """
    frame_dates = data.dates
    all_returns = data.returns.to_numpy(dtype=float)
    factor_returns = data.factors.frame.to_numpy(dtype=float)
    factor_names = list(data.factors.frame.columns)
    names_all = list(data.names)
    column_of = {name: i for i, name in enumerate(names_all)}
    subset_columns = np.asarray([column_of[str(n)] for n in subset], dtype=int)
    is_sample = spec.stage == "sample"
    individual = list(subset) if is_sample else names_all
    random_names = [f"random_{i:03d}" for i in range(random_portfolios)]
    eigen_names = [] if is_sample else [f"eigen_{k + 1:02d}" for k in range(len(factor_names))]
    members = [*individual, *random_names, "min_var", MIN_VAR_SUBSET, *eigen_names]
    member_of = {name: i for i, name in enumerate(members)}
    session_rows = [np.arange(*_held_days(positions, i)) for i in scored]
    sessions = np.concatenate(session_rows)
    forecasts = np.full((sessions.size, len(members)), np.nan)
    realised = np.full((sessions.size, len(members)), np.nan)
    month_of = np.concatenate(
        [np.full(rows.size, i) for i, rows in zip(scored, session_rows, strict=True)]
    )
    rng = np.random.default_rng(seed)
    weights_kept: list[np.ndarray] = []
    random_kept: list[np.ndarray] = []
    liquidated = 0
    offset = 0
    for i, rows in zip(scored, session_rows, strict=True):
        design = designs[i]
        if is_sample:
            asset_columns = subset_columns
            covariance = ewma_second_moment(
                all_returns[: positions[i] + 1][:, asset_columns], halflife=volatility_halflife
            )
            forecast = validation.RiskForecast(covariance=covariance)
            asset_names = list(subset)
        else:
            asset_columns = design.columns
            forecast = _forecast_matrix(
                spec,
                design=design,
                stage=stages[i],
                eigen=eigen[i],
                specific_variance=specifics[i].variance,
                lambda_f={tag: float(values[i]) for tag, values in lambda_f.items()},
                lambda_s=float(lambda_s[i]),
            )
            asset_names = list(design.names)
        validation.assert_forecast_psd(forecast, f"{spec.name} {design.date.date()}")
        assets = len(asset_names)
        draw = validation.random_dollar_neutral(random_portfolios, assets, rng)
        min_var = validation.minimum_variance_weights(forecast.covariance)
        weights_kept.append(min_var)
        if keep_random:
            random_kept.append(draw)
        block = [np.eye(assets), draw, min_var[None, :]]
        block_members = [*asset_names, *random_names, "min_var"]
        if is_sample:
            block.append(min_var[None, :])
            block_members.append(MIN_VAR_SUBSET)
        else:
            inside = np.asarray([n in set(subset) for n in asset_names], dtype=bool)
            sub = validation.minimum_variance_weights(forecast.covariance[np.ix_(inside, inside)])
            full = np.zeros(assets)
            full[inside] = sub
            block.append(full[None, :])
            block_members.append(MIN_VAR_SUBSET)
        weights = np.vstack(block)
        sigma = np.sqrt(np.einsum("pi,ij,pj->p", weights, forecast.covariance, weights))
        raw = all_returns[rows][:, asset_columns]
        missing = ~np.isfinite(raw)
        liquidated += int(missing.sum())
        returns = np.where(missing, 0.0, raw) @ weights.T
        targets = np.asarray([member_of[name] for name in block_members], dtype=int)
        span = slice(offset, offset + rows.size)
        forecasts[span, targets] = sigma
        realised[span, targets] = returns
        if not is_sample:
            factor_matrix = forecast.factor_covariance
            assert factor_matrix is not None
            present = present_block(factor_matrix)
            vectors, values = validation.eigen_portfolios(factor_matrix[np.ix_(present, present)])
            # Ranks from the top (eigen_01 = the largest eigenvalue); ranks beyond
            # K_d stay NaN and the report restricts family 3 to the ranks every
            # scored month-end carries.
            eigen_targets = np.asarray(
                [member_of[name] for name in eigen_names[: present.size]], dtype=int
            )
            forecasts[span, eigen_targets] = np.sqrt(values)
            realised[span, eigen_targets] = factor_returns[rows][:, present] @ vectors.T
        offset += rows.size
    families: dict[str, tuple[str, ...]] = {
        validation.INDIVIDUAL: tuple(individual),
        validation.RANDOM: tuple(random_names),
        validation.OPTIMIZED: ("min_var",),
        OPTIMIZED_SUBSET: (MIN_VAR_SUBSET,),
    }
    if eigen_names:
        families[validation.EIGENFACTOR] = tuple(eigen_names)
    index = pd.DatetimeIndex(frame_dates[sessions])
    return Scored(
        spec=spec,
        forecasts=pd.DataFrame(forecasts, index=index, columns=members),
        realised=pd.DataFrame(realised, index=index, columns=members),
        portfolios=validation.PortfolioSet(families=families, rebalance="held between month-ends"),
        min_var_weights=tuple(weights_kept),
        random_weights=tuple(random_kept),
        liquidated_cells=liquidated,
        month_of_session=month_of,
    )


def to_monthly(scored: Scored) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``R_m = sum_t R_t``, ``sigma_m = sigma * sqrt(days)`` per held month (construction 1)."""
    groups = scored.month_of_session
    keys = np.unique(groups)
    forecasts = scored.forecasts.to_numpy(dtype=float)
    realised = scored.realised.to_numpy(dtype=float)
    rows_f: list[np.ndarray] = []
    rows_r: list[np.ndarray] = []
    dates: list[pd.Timestamp] = []
    for key in keys:
        mask = groups == key
        days = int(mask.sum())
        rows_f.append(forecasts[mask][0] * np.sqrt(days))
        rows_r.append(realised[mask].sum(axis=0))
        dates.append(pd.Timestamp(scored.forecasts.index[np.flatnonzero(mask)[-1]]))
    index = pd.DatetimeIndex(dates)
    return (
        pd.DataFrame(np.asarray(rows_f), index=index, columns=scored.forecasts.columns),
        pd.DataFrame(np.asarray(rows_r), index=index, columns=scored.realised.columns),
    )


# ---------------------------------------------------------------------------
# SPEC.md 10.3's component split, per held month
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentSplit:
    """Per held month: ex-post factor / specific / total returns and the ex-ante variances."""

    periods: pd.DataFrame

    def component(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.periods[f"{name}_return"].to_numpy(dtype=float),
            self.periods[f"ex_ante_{name}_variance"].to_numpy(dtype=float),
        )


def component_split(
    *,
    data: Panel,
    positions: Sequence[int],
    scored: Sequence[int],
    designs: Sequence[Design],
    weights: Sequence[np.ndarray],
    factor: Sequence[np.ndarray],
    specific_variance: Sequence[np.ndarray],
) -> ComponentSplit:
    """Construction 9: the return identity per month, specific as the remainder.

    ``weights[j]`` is the book held under the ``j``-th scored forecast on
    ``designs[scored[j]]``'s names. ``w'u`` from the regression's own residuals
    is carried beside the remainder; their gap is exposure drift inside the
    month.
    """
    all_returns = data.returns.to_numpy(dtype=float)
    residuals = data.residuals.to_numpy(dtype=float)
    factor_returns = data.factors.frame.to_numpy(dtype=float)
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for j, i in enumerate(scored):
        design = designs[i]
        w = np.asarray(weights[j], dtype=float)
        start, stop = _held_days(positions, i)
        raw = all_returns[start:stop][:, design.columns]
        raw = np.where(np.isfinite(raw), raw, 0.0)
        u = residuals[start:stop][:, design.columns]
        u = np.where(np.isfinite(u), u, 0.0)
        exposure = design.exposures.T @ w
        factor_daily = factor_returns[start:stop] @ exposure
        total_daily = raw @ w
        days = stop - start
        rows.append(
            {
                "end": pd.Timestamp(data.dates[stop - 1]),
                "days": days,
                "portfolio_return": float(total_daily.sum()),
                "factor_return": float(factor_daily.sum()),
                "specific_return": float((total_daily - factor_daily).sum()),
                "regression_specific_return": float((u @ w).sum()),
                "ex_ante_factor_variance": float(days * exposure @ factor[i] @ exposure),
                "ex_ante_specific_variance": float(
                    days * np.sum(np.square(w) * specific_variance[i])
                ),
            }
        )
    periods = pd.DataFrame(rows).set_index("end")
    periods["ex_ante_total_variance"] = (
        periods["ex_ante_factor_variance"] + periods["ex_ante_specific_variance"]
    )
    return ComponentSplit(periods=periods)
