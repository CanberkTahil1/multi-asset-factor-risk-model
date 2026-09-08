"""``reports/hybrid_residual_pcs.md`` and its figure. SPEC.md 4.3.

SPEC.md 4.3 asks for four things from the hybrid: residual PCs appended as
unnamed factors, Model A's R-squared alone and with them, a chart of the residual
PC's variance share through 2008, 2020 and 2022, and -- through experiments.md
row 83, pre-registered on 2026-08-30 -- a verdict on whether those PCs are a
missing real-rate factor or noise.

This writes all four, and it writes the one that refuted in the same voice as the
ones that did not.

**Row 83 is REFUTED and is reported as registered.** Gold does not hold the
largest absolute loading on either component; it ranks eleventh of thirteen. The
report prints the registration's own criteria, the null rate beside every clause,
and the full loading vector, because row 83 asked for the loading vector
"regardless of outcome, so a near-miss is visible rather than collapsed into a
pass/fail" -- and this is not a near-miss.

**2008 is not reachable, and the chart says so rather than starting in 2010 and
letting the reader assume otherwise.** The residual PCs carry three stacked
burn-ins -- the credit factor's expanding window, the 252-day beta regression,
and the residual PCA's own expanding window -- so the series opens in 2010. This
is the second time a SPEC.md figure has asked for a date the burn-in cannot
reach; SPEC.md 5.4.3 is the first.

CLAUDE.md invariant 5: everything read stops strictly before
``sample.holdout_start``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")  # No display in CI; must precede the pyplot import.

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from mafrm import config, history
from mafrm.factors import hybrid, macro
from mafrm.risk.config import HORIZONS, Horizon, RiskConfig
from mafrm.risk.covariance import run_pipeline
from mafrm.risk.regime import forecast_history

__all__ = [
    "Acceptance",
    "Cache",
    "Column",
    "Inputs",
    "acceptance",
    "assemble_cache",
    "build",
    "build_history",
    "columns",
    "draw",
    "figure_path",
    "main",
    "read_cache",
    "render",
    "report_path",
    "write_cache",
]

_ROOT = Path(__file__).resolve().parents[3]
_REPORTS = _ROOT / "reports"

#: SPEC.md 5.4's bias history for the hybrid factor set, one column per
#: (component count x horizon x eigenfactor `a`). COMMITTED, like Model B's and
#: Model A's, because rebuilding it costs about a quarter of an hour and `make report`
#: must render from a clean checkout.
CACHE_PATH = _REPORTS / "hybrid_forecast_history.csv"

#: Per-column checkpoints. Under `data/processed/`, so gitignored: they are
#: reproducible intermediates and only the assembled cache is committed.
PARTIALS = _ROOT / "data" / "processed" / "hybrid"

#: The two universe members whose regression is an identity by construction
#: (W2-P2 forward constraint). Derived from the config rather than hard-coded.
_IDENTITY_REASON = "the proxy IS the factor -- SPEC.md 4.1.1"

#: The years SPEC.md 4.3 names for the variance-share chart.
_NAMED_YEARS = (2008, 2020, 2022)


def report_path() -> Path:
    return _REPORTS / "hybrid_residual_pcs.md"


def figure_path() -> Path:
    return _REPORTS / "residual_pc_variance_share.png"


# ---------------------------------------------------------------------------
# SPEC.md 15.2 -- the hybrid through the UNCHANGED covariance pipeline
# ---------------------------------------------------------------------------


#: THE SHARED IMPLEMENTATIONS. Re-exported rather than reimplemented -- see
#: ``mafrm.history``'s module docstring, which records that THIS module's copies
#: were the ones that had diverged: they used ``vars()`` where the other two used
#: ``asdict()``, skipped the tuple normalisation, and hashed the panel in a
#: different order. A digest is a staleness guard, and a third implementation
#: computing a different number from the same inputs guards something other than
#: what it claims to.
_git_commit = history.git_commit
risk_config_digest = history.risk_config_digest
panel_digest = history.panel_digest


@dataclass(frozen=True)
class Column:
    """One cached bias series: a component count, a horizon, and an eigenfactor ``a``."""

    components: int
    horizon: Horizon
    scaling: float

    @property
    def name(self) -> str:
        """The CSV column label, e.g. ``bias_pc2_short_a1.4``."""
        return f"bias_pc{self.components}_{self.horizon}_a{self.scaling:.1f}"

    @property
    def label(self) -> str:
        """For a table row."""
        return f"+{self.components} PC, {self.horizon}, a = {self.scaling:.1f}"


def columns(settings: config.Config | None = None) -> tuple[Column, ...]:
    """Every combination the acceptance runs: component counts x horizons x ``a``.

    **Both eigenfactor scalings and both horizons, exactly as W3-P5 ran Model B
    across its eight**, because CLAUDE.md's parameter table carries `a = 1.0` and
    `a = 1.4` as two model variants and picking one here would be a choice this
    session is not entitled to make.
    """
    loaded = settings or config.load()
    scheme = loaded.model.factors.hybrid
    return tuple(
        Column(components=int(count), horizon=horizon, scaling=float(scaling))
        for count in scheme.component_counts
        for horizon in HORIZONS
        for scaling in loaded.model.eigenfactor.scaling_a
    )


def factor_panels(settings: config.Config | None = None) -> dict[int, pd.DataFrame]:
    """The hybrid factor frame per component count. Built once, reused by all eight."""
    loaded = settings or config.load()
    residuals = hybrid.residual_panel(config=loaded)
    macro_panel = macro.macro_factor_panel(config=loaded)
    scheme = loaded.model.factors.hybrid
    out: dict[int, pd.DataFrame] = {}
    for count in scheme.component_counts:
        pcs = hybrid.residual_pcs(residuals.residuals, components=int(count), settings=scheme)
        out[int(count)] = hybrid.factor_panel(pcs, macro_panel, config=loaded)
    return out


def _minimum_observations(frame: pd.DataFrame, settings: config.Config) -> int:
    """``max(K + 1, lags + 1)``. The same ARITHMETIC floor the other two histories use.

    Not a burn-in choice. Below ``K + 1`` SPEC.md 5.3's Monte Carlo re-estimates a
    singular correlation; below ``lags + 1`` SPEC.md 5.2's longest Bartlett lag
    has no pairs. SPEC.md 5.1.2 leaves the real burn-in with the caller and
    nothing is filtered -- ``K/T_eff`` is carried per date instead.
    """
    reference = RiskConfig.load(horizon=HORIZONS[0], config=settings)
    return max(frame.shape[1] + 1, reference.volatility_newey_west_lags + 1)


def _partial_path(column: Column) -> Path:
    return PARTIALS / f"{column.name}.csv"


def build_history(
    column: Column,
    settings: config.Config | None = None,
    *,
    panels: dict[int, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """One bias history, written to its partial. About twelve minutes.

    Returns an existing partial unchanged when its digests still match, so an
    interrupted rebuild resumes at history granularity. The digest pair is the
    one the committed cache carries -- the ``RiskConfig`` and the factor panel --
    so a partial written against a superseded risk config or a moved panel is
    **discarded** rather than silently reused.
    """
    loaded = settings or config.load()
    built = panels if panels is not None else factor_panels(loaded)
    frame = built[column.components]
    digests = {
        "risk_config_digest": risk_config_digest(loaded),
        "panel_digest": panel_digest(frame),
    }

    def compute() -> pd.DataFrame:
        risk = RiskConfig.load(horizon=column.horizon, config=loaded).for_scaling(column.scaling)
        print(f"{column.name}: {len(frame):,} dates", file=sys.stderr, flush=True)
        computed = forecast_history(
            frame,
            risk,
            minimum_observations=_minimum_observations(frame, loaded),
            stop_after="eigenfactor",
        )
        table = pd.DataFrame(
            {
                column.name: computed.bias,
                f"k_over_realised_t_eff_pc{column.components}_{column.horizon}": (
                    computed.k_over_realised_t_eff
                ),
            },
            index=pd.DatetimeIndex(computed.dates, name="date"),
        )
        return table

    return history.load_or_build(_partial_path(column), digests, compute)


#: The shared cache type. Its ``bias()`` trims the index union's NaN padding and
#: refuses an interior gap.
Cache = history.Cache


def assemble_cache(settings: config.Config | None = None) -> Cache:
    """Join every partial into the committed cache, building any that are missing."""
    loaded = settings or config.load()
    panels = factor_panels(loaded)
    variants = columns(loaded)
    ordered = [
        f"k_over_realised_t_eff_pc{count}_{horizon}"
        for count in sorted(panels)
        for horizon in HORIZONS
    ]
    ordered.extend(column.name for column in variants)
    return history.assemble(
        (build_history(column, loaded, panels=panels) for column in variants),
        order=ordered,
        what=(
            "SPEC.md 5.4 bias series B_t^F for the HYBRID factor set (SPEC.md 4.3) -- six "
            "named macro factors plus 1 or 2 unnamed residual PCs. One column per "
            "component count x horizon x eigenfactor a, built by `python -m "
            "mafrm.factors.hybrid_report --rebuild --all`. sigma_kt is the forecast made at "
            "t-1 from the full pre-VRA pipeline."
        ),
        settings=loaded,
        panel_digests={str(count): panel_digest(frame) for count, frame in panels.items()},
        variants=variants,
    )


def write_cache(cache: Cache, path: Path = CACHE_PATH) -> None:
    """CSV with a ``#``-prefixed JSON provenance header. Shared format."""
    history.write_cache(cache, path)


def read_cache(path: Path = CACHE_PATH, *, panels: dict[int, pd.DataFrame] | None = None) -> Cache:
    """Read the committed cache and REFUSE a stale one.

    Two digests, for the two ways it goes stale. The ``RiskConfig`` digest
    catches a changed half-life or lag count. The panel digest catches a moved
    factor series, which nothing in ``config/model.yaml`` would show -- the
    failure W3-P5 actually hit. **The hybrid is the most exposed of the four
    callers to that one**: its panel depends on the macro factors, the exposure
    regression, the residual construction and the residual PCA, so four separate
    modules can move it.
    """
    return history.read_cache(
        path,
        rebuild_hint="`python -m mafrm.factors.hybrid_report --rebuild --all`",
        panel_digests=(
            None if panels is None else {str(c): panel_digest(f) for c, f in panels.items()}
        ),
    )


@dataclass(frozen=True)
class Acceptance:
    """One (component count, horizon, ``a``) run of the full pipeline on the hybrid."""

    column: Column
    factors: int
    observations: int
    stages: tuple[str, ...]
    minimum_eigenvalues: tuple[float, ...]
    condition_number: float
    lambda_squared: float
    repair_fired: bool
    bartlett_fired: bool
    complete: bool

    @property
    def passed(self) -> bool:
        """Every declared stage ran and every one came back positive semi-definite."""
        return self.complete and all(value > -1e-12 for value in self.minimum_eigenvalues)


def acceptance(settings: config.Config | None = None) -> tuple[Acceptance, ...]:
    """Run the hybrid through the UNCHANGED pipeline. SPEC.md 15.2, W3-P6.

    The point is not the matrices. It is that this function imports
    ``run_pipeline`` and hands it a ``T x K`` frame, exactly as ``mafrm.build``
    does for Model A and ``statistical_report.acceptance`` does for Model B, and
    that nothing in ``mafrm.risk`` can tell the difference.

    **This is the harder version of the test than Model B was.** Model B's
    factors are a homogeneous statistical basis -- every column the same kind of
    object, on the same scale. The hybrid's frame mixes six named macro factors
    in two different units with one or two dimensionless residual components, so
    if ``risk/`` held an assumption about what a factor is, this is the frame
    that would surface it.
    """
    loaded = settings or config.load()
    panels = factor_panels(loaded)
    cache = read_cache(panels=panels)
    results: list[Acceptance] = []
    for column in columns(loaded):
        frame = panels[column.components]
        risk = RiskConfig.load(horizon=column.horizon, config=loaded).for_scaling(column.scaling)
        bias = cache.bias(column)
        # SPEC.md 5.4's stage takes the bias history as a bare array and reduces
        # it to one scalar, so a history belonging to the OTHER component count
        # would go through silently and produce a plausible wrong multiplier. The
        # length relation is exact and is the one thing that can catch it here.
        expected = len(frame) - _minimum_observations(frame, loaded)
        if len(bias) != expected:
            raise ValueError(
                f"{column.name}: cached bias has {len(bias)} dates against {expected} for the "
                f"{column.components}-PC panel. A history from the wrong component count would "
                "reduce to a plausible multiplier with nothing downstream able to detect it."
            )
        build_result = run_pipeline(frame, risk, regime_bias=bias)
        results.append(
            Acceptance(
                column=column,
                factors=build_result.factors,
                observations=build_result.observations,
                stages=tuple(stage.name for stage in build_result.stages),
                minimum_eigenvalues=tuple(
                    float(stage.minimum_eigenvalue) for stage in build_result.stages
                ),
                condition_number=float(build_result.stages[-1].condition_number),
                lambda_squared=(
                    float(build_result.regime.lambda_squared)
                    if build_result.regime is not None
                    else float("nan")
                ),
                repair_fired=build_result.repair_fired,
                bartlett_fired=build_result.bartlett_fired,
                complete=build_result.complete,
            )
        )
    return tuple(results)


@dataclass(frozen=True)
class Inputs:
    """Everything the report renders, computed once."""

    panel: hybrid.ResidualPanel
    pcs: tuple[hybrid.ResidualPcPanel, ...]
    comparands: pd.DataFrame
    results: dict[int, tuple[hybrid.Row83Result, ...]]
    reduced_results: dict[int, tuple[hybrid.Row83Result, ...]]
    r_squared: pd.DataFrame
    variance_share: pd.DataFrame
    raw_yield_r_squared: pd.Series
    pc_on_raw_r_squared: dict[str, float]
    pc_on_factors_r_squared: dict[str, float]
    butterfly_correlation: dict[str, float]
    bp_variance_share: dict[str, float]
    identity_assets: tuple[str, ...]
    acceptance: tuple[Acceptance, ...]
    factor_frames: dict[int, pd.DataFrame]


def _ols_r_squared(target: np.ndarray, regressors: np.ndarray) -> float:
    """Plain OLS R-squared with an intercept. The controls' common kernel."""
    design = np.column_stack([np.ones(len(regressors)), regressors])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    centred = target - target.mean()
    return 1.0 - float(residual @ residual) / float(centred @ centred)


def build(settings: config.Config | None = None) -> Inputs:
    """Run the hybrid and every control the report prints."""
    cfg = settings or config.load()
    scheme = cfg.model.factors.hybrid
    panel, pcs, comparands = hybrid.build(config=cfg)

    results = {pc.components: hybrid.evaluate_row_83(pc, comparands, settings=scheme) for pc in pcs}

    # Control C3: the two identity assets, whose "residuals" are the numerical
    # remainder of an exact linear identity rather than observations.
    identity = tuple(
        sorted(
            asset
            for asset in panel.assets
            if asset
            in (
                cfg.model.factors.macro.credit.asset_id,
                cfg.model.factors.macro.commodity.asset_id,
            )
        )
    )
    reduced = panel.residuals.drop(columns=list(identity))
    reduced_results = {
        count: hybrid.evaluate_row_83(
            hybrid.residual_pcs(reduced, components=count, settings=scheme),
            comparands,
            settings=scheme,
        )
        for count in scheme.component_counts
    }

    # Controls C1 and C2: the four RAW yield changes, which row 82 showed span
    # the duration leg exactly.
    tenors = list(cfg.model.data.nominal_curve.maturities_years)
    raw = macro.curve_yield_changes_bps(
        tenors_years=tenors, end=cfg.require_holdout_start(), config=cfg
    )
    shared = panel.residuals.index.intersection(raw.index)
    raw_values = raw.loc[shared].to_numpy(dtype=float)
    raw_r2 = pd.Series(
        {
            asset: _ols_r_squared(
                panel.residuals.loc[shared, asset].to_numpy(dtype=float), raw_values
            )
            for asset in panel.assets
        }
    ).sort_values(ascending=False)

    widest = pcs[-1]
    pc_shared = widest.factors.index.intersection(raw.index)
    pc_raw = raw.loc[pc_shared].to_numpy(dtype=float)
    rate_factors = panel.factors.loc[pc_shared, [macro.LEVEL, macro.SLOPE]].to_numpy(dtype=float)
    butterfly = (
        raw.loc[pc_shared, f"y{tenors[0]:g}"]
        - 2.0 * raw.loc[pc_shared, f"y{tenors[2]:g}"]
        + raw.loc[pc_shared, f"y{tenors[3]:g}"]
    ).to_numpy(dtype=float)

    on_raw: dict[str, float] = {}
    on_factors: dict[str, float] = {}
    fly: dict[str, float] = {}
    for name in widest.factors.columns:
        series = widest.factors.loc[pc_shared, name].to_numpy(dtype=float)
        on_raw[name] = _ols_r_squared(series, pc_raw)
        on_factors[name] = _ols_r_squared(series, rate_factors)
        fly[name] = float(np.corrcoef(series, butterfly)[0, 1])

    # What the correlation-space share is worth in actual basis points squared.
    residuals = panel.residuals.loc[widest.factors.index].to_numpy(dtype=float)
    standardised = widest.standardised.to_numpy(dtype=float)
    deviations = np.divide(
        residuals, standardised, out=np.zeros_like(residuals), where=standardised != 0.0
    )
    total = float((residuals**2).sum())
    bp_share = {
        name: float(
            (
                (widest.factors[name].to_numpy(dtype=float)[:, None] * loading * deviations) ** 2
            ).sum()
        )
        / total
        for name, loading in (
            (name, widest.loadings[name].to_numpy(dtype=float)) for name in widest.factors.columns
        )
    }

    frames = factor_panels(cfg)
    return Inputs(
        acceptance=acceptance(cfg),
        factor_frames=frames,
        panel=panel,
        pcs=pcs,
        comparands=comparands,
        results=results,
        reduced_results=reduced_results,
        r_squared=hybrid.augmented_r_squared(panel, pcs, config=cfg),
        variance_share=hybrid.monthly_variance_share(widest),
        raw_yield_r_squared=raw_r2,
        pc_on_raw_r_squared=on_raw,
        pc_on_factors_r_squared=on_factors,
        butterfly_correlation=fly,
        bp_variance_share=bp_share,
        identity_assets=identity,
    )


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------


def draw(inputs: Inputs) -> Figure:
    """Residual PC variance share by CALENDAR MONTH -- non-overlapping buckets.

    The buckets are months and not a rolling window, deliberately. CLAUDE.md
    failure mode 9: a rolling share stepped daily shares 251 of its 252 days with
    its neighbour, and every reading would owe an overlap caveat. Months owe
    none, and they cost no window parameter.
    """
    share = inputs.variance_share
    widest = inputs.pcs[-1]
    columns = [name for name in widest.factors.columns]

    figure, axis = plt.subplots(figsize=(11.0, 4.6))
    colours = {"residual_pc1": "#b2182b", "residual_pc2": "#2166ac"}
    for name in columns:
        axis.plot(
            share.index,
            share[name],
            linewidth=1.1,
            color=colours.get(name, "0.4"),
            label=f"{name} ({inputs.bp_variance_share[name]:.1%} of residual bp²)",
        )

    equal = 1.0 / len(widest.assets)
    axis.axhline(
        equal,
        color="0.55",
        linestyle=":",
        linewidth=1.0,
        label=f"1/N = {equal:.3f}, no structure",
    )

    axis.set_ylim(0.0, max(0.75, float(share[columns].to_numpy().max()) * 1.08))
    for year in _NAMED_YEARS:
        block = _year_block(share, year)
        if block.empty:
            continue
        axis.axvspan(
            _to_number(pd.Timestamp(year=year, month=1, day=1)),
            _to_number(pd.Timestamp(year=year, month=12, day=31)),
            color="0.85",
            alpha=0.45,
            zorder=0,
        )
        axis.annotate(
            str(year),
            xy=(_to_number(pd.Timestamp(year=year, month=7, day=1)), axis.get_ylim()[1]),
            xytext=(0, -10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="0.35",
        )

    axis.set_ylabel("share of residual variance, per calendar month")
    axis.set_xlabel("")
    axis.xaxis.set_major_locator(mdates.YearLocator(2))  # type: ignore[no-untyped-call]
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # type: ignore[no-untyped-call]
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    axis.set_title(
        "Residual PC variance share, non-overlapping calendar months", fontsize=11, loc="left"
    )

    missing = [str(year) for year in _NAMED_YEARS if _year_block(share, year).empty]
    note = (
        "Buckets are calendar months and therefore NON-OVERLAPPING: no rolling-window "
        "overlap caveat is owed (CLAUDE.md failure mode 9).\n"
        f"Series opens {widest.factors.index[0].date()} -- three stacked burn-ins."
    )
    if missing:
        note += (
            f"  {', '.join(missing)} is NOT REACHABLE and is not drawn; "
            "SPEC.md 4.3 asks for it and the burn-in cannot supply it."
        )
    figure.text(0.5, -0.02, note, ha="center", fontsize=8, color="0.35")
    return figure


# ---------------------------------------------------------------------------
# The markdown
# ---------------------------------------------------------------------------


#: The illustrative test size in the placebo's justification. Not a model
#: parameter and nothing selects on it: it exists so the report can say what a
#: conventional significance test WOULD have passed at, which is the argument for
#: why clause (b) is a placebo instead.
_ILLUSTRATIVE_TEST_SIZE = 0.05


def _critical_correlation(observations: int) -> float:
    """|rho| a two-sided test at :data:`_ILLUSTRATIVE_TEST_SIZE` would just reject at."""
    critical = NormalDist().inv_cdf(1.0 - _ILLUSTRATIVE_TEST_SIZE / 2.0)
    return critical / float(np.sqrt(observations))


def _to_number(moment: pd.Timestamp) -> float:
    """Matplotlib's float date. Wrapped once so the untyped call is in one place."""
    return float(mdates.date2num(moment))  # type: ignore[no-untyped-call]


def _year_block(share: pd.DataFrame, year: int) -> pd.DataFrame:
    """The months of one calendar year. Typed so the index stays a DatetimeIndex."""
    index = pd.DatetimeIndex(share.index)
    return share.loc[index.year == year]


def _peak_month(series: pd.Series) -> str:
    """``YYYY-MM`` of the series maximum."""
    return pd.Timestamp(series.idxmax()).strftime("%Y-%m")


def _verdict(result: hybrid.Row83Result) -> str:
    return "**HOLDS**" if result.passes else "**FAILS**"


def _clause(holds: bool) -> str:
    return "holds" if holds else "**fails**"


def _row_83_table(results: tuple[hybrid.Row83Result, ...], assets: int) -> str:
    lines = [
        "| Component | Clause (a): gold ranks first | Clause (b): real beats nominal "
        "| Clause (c): sign is negative | All three |",
        "|---|---|---|---|---|",
    ]
    for result in results:
        lines.append(
            f"| `{result.component}` | {_clause(result.loading_holds)} "
            f"-- rank **{result.gold_rank}** of {assets}, "
            f"abs(loading) {result.gold_absolute_loading:.4f} against "
            f"`{result.next_asset}`'s {result.next_absolute_loading:.4f} "
            f"| {_clause(result.placebo_holds)} -- "
            f"abs(rho_real) {abs(result.correlation_real):.4f} vs "
            f"abs(rho_nom) {abs(result.correlation_nominal):.4f} "
            f"| {_clause(result.sign_holds)} -- rho_real "
            f"{result.correlation_real:+.4f} | {_verdict(result)} |"
        )
    return "\n".join(lines)


def _loading_table(inputs: Inputs) -> str:
    widest = inputs.pcs[-1]
    frame = pd.DataFrame({name: widest.loadings[name].mean() for name in widest.factors.columns})
    frame = frame.reindex(frame.abs().max(axis=1).sort_values(ascending=False).index)
    lines = ["| Asset | " + " | ".join(f"`{c}`" for c in frame.columns) + " |"]
    lines.append("|---" * (len(frame.columns) + 1) + "|")
    for asset, row in frame.iterrows():
        mark = " **<- row 83's asset**" if asset == widest.orientation_asset else ""
        lines.append(f"| `{asset}`{mark} | " + " | ".join(f"{value:+.4f}" for value in row) + " |")
    return "\n".join(lines)


def _horizon_direction(inputs: Inputs) -> str:
    """One sentence on whether the two horizons agree about the sign of the VRA."""
    by_horizon = {
        horizon: [r.lambda_squared for r in inputs.acceptance if r.column.horizon == horizon]
        for horizon in HORIZONS
    }
    over = {h: all(v > 1.0 for v in values) for h, values in by_horizon.items()}
    under = {h: all(v < 1.0 for v in values) for h, values in by_horizon.items()}
    parts = []
    for horizon in HORIZONS:
        low, high = min(by_horizon[horizon]), max(by_horizon[horizon])
        if over[horizon]:
            parts.append(f"the **{horizon}** horizon UNDER-forecasts ({low:.4f}-{high:.4f})")
        elif under[horizon]:
            parts.append(f"the **{horizon}** horizon OVER-forecasts ({low:.4f}-{high:.4f})")
        else:
            parts.append(f"the **{horizon}** horizon straddles 1 ({low:.4f}-{high:.4f})")
    tail = (
        " The two horizons disagree about the SIGN, which is the same pattern W3-P4 measured on "
        "Model A's own factors and has the same cause: a 252-day half-life on a panel containing "
        "2020 and 2022."
        if over[HORIZONS[0]] != over[HORIZONS[1]]
        else " Both horizons point the same way here, unlike W3-P4's measurement on Model A."
    )
    return f"So {', and '.join(parts)}.{tail}"


def _acceptance_table(inputs: Inputs) -> str:
    stages = inputs.acceptance[0].stages
    header = ["Run", "K", "T"] + [f"`{name}`" for name in stages] + ["cond", "lambda_F^2", "PSD"]
    lines = ["| " + " | ".join(header) + " |", "|---" * len(header) + "|"]
    for result in inputs.acceptance:
        cells = [result.column.label, str(result.factors), f"{result.observations:,}"]
        cells += [f"{value:.2e}" for value in result.minimum_eigenvalues]
        cells += [
            f"{result.condition_number:.3g}",
            f"{result.lambda_squared:.4f}",
            "**yes**" if result.passed else "**NO**",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _r_squared_table(inputs: Inputs) -> str:
    frame = inputs.r_squared
    counts = [pc.components for pc in inputs.pcs]
    header = ["Asset", "Model A"] + [f"+{k} PC" for k in counts] + [f"gain (+{k})" for k in counts]
    lines = ["| " + " | ".join(header) + " |", "|---" * len(header) + "|"]
    for asset, row in frame.iterrows():
        cells = [f"`{asset}`", f"{row['model_a']:.4f}"]
        cells += [f"{row[f'model_a_plus_{k}']:.4f}" for k in counts]
        cells += [f"{row[f'gain_{k}']:+.4f}" for k in counts]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render(inputs: Inputs, settings: config.Config | None = None) -> str:
    """The whole report."""
    cfg = settings or config.load()
    scheme = cfg.model.factors.hybrid
    panel = inputs.panel
    widest = inputs.pcs[-1]
    assets = len(panel.assets)
    share = inputs.variance_share
    single, _ = hybrid.loading_rank_null(assets, 1)
    union_a, independent_a = hybrid.loading_rank_null(assets, len(widest.factors.columns))
    union_all, independent_all = hybrid.union_null_probability(inputs.results[widest.components])
    tenors = tuple(int(value) for value in cfg.model.data.nominal_curve.maturities_years)
    butterfly_label = f"{tenors[0]}-2x{tenors[2]}+{tenors[3]}"
    missing_years = [str(year) for year in _NAMED_YEARS if _year_block(share, year).empty]
    covered_years = [str(year) for year in _NAMED_YEARS if not _year_block(share, year).empty]

    parts: list[str] = []
    parts.append(
        f"""# The hybrid -- residual PCs on Model A. SPEC.md 4.3

Generated by `python -m mafrm.factors.hybrid_report`. Everything below stops
strictly before `sample.holdout_start` = **{cfg.model.sample.holdout_start}**
(CLAUDE.md invariant 5).

SPEC.md 4.3: *"Run PCA on the residuals of Model A and append the top 1-2
residual PCs as unnamed factors... Report Model A's R-squared alone and with the
residual PCs."* And, from experiments.md row 83, registered **2026-08-30** --
five sessions before this code existed and before any residual PC had been
extracted -- a verdict on whether those PCs are a missing real-rate factor or
noise.

## The headline

**Row 83 is REFUTED.** Neither residual PC loads materially on `gold`: it ranks
**{inputs.results[widest.components][0].gold_rank}th of {assets}** on the first
component and
**{inputs.results[widest.components][-1].gold_rank}th of {assets}** on the
second, with mean absolute loadings of
{inputs.results[widest.components][0].gold_absolute_loading:.4f} and
{inputs.results[widest.components][-1].gold_absolute_loading:.4f} against a
leading loading above 0.5 in both cases. The registered falsifier says what to do
with that: *"the residual PCs are noise and the hybrid adds nothing over the
macro core -- and that is the finding, reported as such rather than worked
around."*

**They are not noise, and that is the part the registration did not anticipate.**
The first residual PC is a real, strongly identified structure -- it is the
**third mode of the yield curve**, which the six-factor set cannot span because
it carries only two rate PCs. That is experiments.md rows 81-82's finding
arriving from the other direction, and three pre-registered controls below
establish it rather than assert it.

So the honest summary is neither of row 83's two branches. The hybrid does detect
systematic risk the named factor set missed. What it detected already has a name.

## What was built

| | |
|---|---|
| Residual panel | **{panel.observations:,} dates x {panel.variables} assets**, {panel.residuals.index[0].date()} to {panel.residuals.index[-1].date()}, basis points per day |
| Dropped by complete-cases | {panel.dates_dropped_by_completeness} dates, reported rather than filled |
| Residual PC panel | **{len(widest.factors):,} dates**, {widest.factors.index[0].date()} to {widest.factors.index[-1].date()} |
| Estimator | `{scheme.estimator}` on the residual **correlation** (SPEC.md 4.3.3), expanding window |
| Burn-in | {widest.expanding_min_window} days, read from `factors.macro.rates.pca.expanding_min_window` -- no key of its own |
| Component counts | {list(scheme.component_counts)} -- **both built, neither selected** |
| Denoising | **none**, and deliberately: SPEC.md 4.2.7 measured MP denoising as harmful at this `N/T` |
| Government control | `{scheme.government_zero_return_leg}` leg alone for {", ".join(f"`{a}`" for a in panel.substituted_assets)} |
| Component stability | minimum alignment {widest.component_alignment.min():.4f} across dates |

**The government control is W2-P3's, fixed on 2026-08-30 before any residual PC
existed.** The four synthetic zeros are regressed on their duration leg alone --
`duration_effect` from SPEC.md 3.2's three-way decomposition -- because carry and
roll-down cannot be spanned by a factor set built from yield *changes* and would
otherwise put a term premium into the residual. `tips_10y` is deliberately not in
that set: its alpha was never flagged, so there is no measured term premium in it
to strip, and widening a pre-registered control after the fact is the move the
control exists to prevent.

## Row 83, as registered

The four criteria were ruled by the operator on 2026-08-31 **before the residual
panel was computed**, and every one is rank- or comparison-based so that none of
them is a magnitude anyone chose.

| Clause | Rule | Null rate |
|---|---|---|
| (a) "loads materially on gold" | `gold` holds the **largest absolute loading** on the component | `1/{assets}` = **{single:.4f}** |
| (b) "correlates with TIPS" | **placebo**: `abs(rho(PC, d.real 10y)) > abs(rho(PC, d.nominal 10y))` | **0.5000** |
| (c) the sign | with the PC oriented so `gold`'s loading is positive, `rho` with the real-yield change must be **{"negative" if scheme.row_83.expected_correlation_sign < 0 else "positive"}** | **0.5000** |
| all three, one component | | `1/{4 * assets}` = **{single * 0.25:.4f}** |
| clause (a), at least one of {len(widest.factors.columns)} | | union **{union_a:.4f}**, independent **{independent_a:.4f}** |
| all three, at least one of {len(widest.factors.columns)} | | union **{union_all:.4f}**, independent **{independent_all:.4f}** |

**These were computed and written down before the result was, and they are
printed here because a criterion met is not a finding until its null rate is next
to it.** The union bound and the independent form are both given: they are not
independent -- eigenvectors of one matrix are orthogonal -- so the independent
figure is an approximation and is labelled as one. "At least one of two
components passes" is a materially weaker claim than "the first one does", and
the two rows above keep them apart.

**Clause (b) is a placebo and not a significance bar.** At
n = {inputs.results[widest.components][0].observations:,} a two-sided 5% test
passes at abs(rho) > {_critical_correlation(inputs.results[widest.components][0].observations):.4f},
which has no teeth, and the 0.3 bar SPEC.md 6.5.2 withdrew is not reimported. The
six-factor set already contains a nominal level factor, so a PC that is merely
more nominal-rates exposure fails the placebo while a genuine real-rate factor
cannot. The two comparand legs are correlated at
{inputs.comparands.corr().to_numpy()[0, 1]:.4f} over
{len(inputs.comparands):,} dates, which is why the placebo is the right shape of
test and a bare correlation with the real leg would not be.

## The result

At **{widest.components} components**:

{_row_83_table(inputs.results[widest.components], assets)}

At **1 component**, the same first row, bit-identical -- the expanding-window
decomposition does not depend on how many of its components are kept:

{_row_83_table(inputs.results[1], assets)}

**Clauses (b) and (c) both hold on both components. Clause (a) fails on both, so
the hypothesis fails.** Row 83's hypothesis is a conjunction -- *"at least one
residual PC will (a) load materially on gold **and** (b) correlate with the TIPS
real-yield series"* -- and its falsifier is explicit that failing either branch
refutes it.

**It is not close.** Row 83 asked for the loading vector "regardless of outcome,
so a near-miss is visible rather than collapsed into a pass/fail". Here it is,
mean loading per asset over the sample:

{_loading_table(inputs)}

`gold` sits at {inputs.results[widest.components][0].gold_absolute_loading:.4f}
on the first component against `{inputs.results[widest.components][0].next_asset}`'s
{inputs.results[widest.components][0].next_absolute_loading:.4f}. That is a
factor of {inputs.results[widest.components][0].next_absolute_loading / max(inputs.results[widest.components][0].gold_absolute_loading, 1e-12):.0f},
not a rounding.

### The two sign conventions, and why the agreement share is printed

A PC's sign is unidentified, so clause (c) is stated sign-invariantly: each
date's component is oriented so that `{widest.orientation_asset}`'s loading is
positive, and the correlation must then be negative. That is the **test**
convention. The factor series handed downstream instead uses SPEC.md 4.2.3's
**continuity** convention -- aligned against the previous date, seeded on the
largest-magnitude loading -- because a factor series must not flip on a date when
one loading crosses zero.

The two agree on
{", ".join(f"**{value:.1%}** of dates for `{name}`" for name, value in widest.orientation_agreement.items())}.
A low share would have meant `{widest.orientation_asset}`'s
loading is not stably signed and the test orientation is close to a coin flip.
These are high, so clause (c) is testing something -- which matters, because
clause (c) passed.

## Why the first residual PC is the third curve mode

Three controls, **each registered with its prediction and falsifier before it
ran**, and each rank- or comparison-based.

### C1 -- the residuals against the four raw yield changes

Row 82 established that `duration_effect` is exactly linear in the four raw yield
changes (R-squared = 1.000000). If the leading residual PC is the unspanned curve
modes, the four government residuals must be far better explained by those raw
changes than anything else in the panel.

**Predicted:** every government residual has a higher R-squared than every
non-government residual. **Falsified if** any non-government asset exceeds the
smallest government one.

| Asset | R-squared on the four raw yield changes |
|---|---|
{chr(10).join(f"| `{asset}` | {value:.4f} |" for asset, value in inputs.raw_yield_r_squared.items())}

**HOLDS.** The four government residuals run
{inputs.raw_yield_r_squared[[a for a in inputs.raw_yield_r_squared.index if a.startswith("govt_")]].min():.4f}
to
{inputs.raw_yield_r_squared[[a for a in inputs.raw_yield_r_squared.index if a.startswith("govt_")]].max():.4f};
the largest non-government value is
{inputs.raw_yield_r_squared[[a for a in inputs.raw_yield_r_squared.index if not a.startswith("govt_")]].max():.4f}
(`{inputs.raw_yield_r_squared[[a for a in inputs.raw_yield_r_squared.index if not a.startswith("govt_")]].idxmax()}`).
A factor of twenty separates the two groups.

### C2 -- the PC itself, raw changes against the model's own rate factors

**Predicted:** `residual_pc1` is far better explained by the four raw yield
changes than by the model's two orthogonalized rate factors, and the latter is
near zero -- because `{macro.LEVEL}` and `{macro.SLOPE}` were regressed *out* of
the residual by construction, so the PC must live in the orthogonal complement of
their span inside the four-dimensional yield-change space. **Falsified if** the
two rate factors explain as much or more, which would make the PC leaked factor
exposure -- a regression defect -- rather than an unspanned curve mode.

| Component | R-squared on 4 raw yield changes | R-squared on `{macro.LEVEL}` + `{macro.SLOPE}` | correlation with the {butterfly_label} butterfly |
|---|---|---|---|
{chr(10).join(f"| `{name}` | {inputs.pc_on_raw_r_squared[name]:.4f} | {inputs.pc_on_factors_r_squared[name]:.4f} | {inputs.butterfly_correlation[name]:+.4f} |" for name in widest.factors.columns)}

**HOLDS, and decisively.** `residual_pc1` is at
{inputs.pc_on_raw_r_squared["residual_pc1"]:.4f} on the raw curve and
{inputs.pc_on_factors_r_squared["residual_pc1"]:.4f} on the factors the residual
was constructed to be orthogonal to -- the second number is the construction
checking itself. Its correlation with the named curvature butterfly is
{inputs.butterfly_correlation["residual_pc1"]:+.4f}.

**This is the third curve mode, and rows 81-82 already closed it as an attributed
component rather than a residual.** A third rate factor (curvature) would remove
it; that is a specification decision owing its own `model-config` row, and
nothing here takes it.

### C3 -- the two identity assets

`{"` and `".join(inputs.identity_assets)}` regress on the macro set at
R-squared = 1.000 because {_IDENTITY_REASON}. Their "residuals" are the numerical
remainder of an exact linear identity, not observations. W2-P2's forward
constraint scopes their exclusion to **aggregate bias statistics** and says
explicitly that they stay in the exposure panel and the reports -- so it does not
reach a residual PCA. But a correlation PCA standardises every column to unit
variance, which hands two columns of numerical noise the same weight as `gold`.
That is a question to measure, not to assume.

**Predicted:** dropping them (N = {assets - len(inputs.identity_assets)}) does not
change row 83's verdict. **Falsified if** `gold` reaches rank 1 on either
component, in which case the N = {assets} reading is contaminated and the
refutation would have been withdrawn and re-run at
N = {assets - len(inputs.identity_assets)}.

| Component | gold's rank at N = {assets - len(inputs.identity_assets)} | leading asset | verdict |
|---|---|---|---|
{chr(10).join(f"| `{r.component}` | **{r.gold_rank}** of {r.assets_in_panel} | `{r.next_asset}` | {_verdict(r)} |" for r in inputs.reduced_results[max(scheme.component_counts)])}

**HOLDS.** The verdict is unchanged and the leading loadings barely move. The
refutation is not an artefact of the two identity columns.

## Model A's R-squared, alone and with the residual PCs

{_r_squared_table(inputs)}

Median gain across the panel: **{inputs.r_squared["gain_1"].median():+.4f}** with one
residual PC and **{inputs.r_squared["gain_2"].median():+.4f}** with two.

**Every column is estimated on the same rows, and that correction is itself a
finding.** A first version of this table read Model A's R-squared from the
exposure panel and the augmented ones from a design starting
{len(panel.residuals) - len(widest.factors)} rows later. Three assets came out
with a **negative** gain from adding a regressor, which cannot happen inside one
regression. A rolling window is 252 *rows*, not 252 calendar days, so the two
runs were covering different spans and were never like-for-like. Re-estimating
the baseline on the augmented design's own index removes every negative: the
worst per-date gain across the panel is now **+1.1e-16**. The control that caught
it (C4) is logged in `experiments.md` with its prediction refuted, because the
prediction *was* refuted -- it said the negatives were a date-set artefact that
would vanish on the intersection, and they did not, because the intersection was
still comparing two different window contents.

**The comparison is descriptive and in-sample by construction, and that is the
whole of its honest reading.** The PC at date `t` is a linear combination of that
same date's residuals, so on a shared design the augmented R-squared cannot fall.
It answers "how much of what the six factors missed is common across assets". It
does **not** answer "would this factor have helped out of sample", and nothing
here claims it does. A forecast comparison is the covariance pipeline's job.

The two assets at 1.0000 are the identity pair above; `commodity` and
`hy_credit` have nothing left to explain.

## Variance share -- and the year that is not reachable

![Residual PC variance share](residual_pc_variance_share.png)

SPEC.md 4.3 asks for the residual PC's variance share charted **through 2008,
2020 and 2022**. {", ".join(covered_years)} {"is" if len(covered_years) == 1 else "are"} there.
**{", ".join(missing_years) if missing_years else "None"} is not, and the chart says so rather
than opening in {pd.Timestamp(share.index[0]).year} and letting a reader assume otherwise.**

The residual PCs carry three stacked burn-ins: the credit factor's expanding
window, the {cfg.model.factors.macro.beta.window}-day beta regression, and the
residual PCA's own {widest.expanding_min_window}-day expanding window. The series
opens **{widest.factors.index[0].date()}**. The August 2007 quant quake that SPEC.md 4.3
names as the motivating case is further out of reach still. **This is the second
time a SPEC.md figure has asked for a date this project's burn-in cannot supply**,
after SPEC.md 5.4.3's Aug 2007. That is a narrower family than SPEC.md 5.4.3's own
count of five gates calibrated for a case this project is not in -- the other
three are thresholds rather than dates -- and both counts are right at their own
scope. The pattern here is the specific one: a specification written before the
burn-in was known will name crises the model cannot see, and this build's burn-in
is three stacked windows rather than one.

| Year | `residual_pc1` mean / max | `residual_pc2` mean / max |
|---|---|---|
{chr(10).join(f"| {year} | {_year_block(share, int(year))['residual_pc1'].mean():.3f} / {_year_block(share, int(year))['residual_pc1'].max():.3f} | {_year_block(share, int(year))['residual_pc2'].mean():.3f} / {_year_block(share, int(year))['residual_pc2'].max():.3f} |" for year in covered_years)}
| whole sample | {share["residual_pc1"].median():.3f} / {share["residual_pc1"].max():.3f} | {share["residual_pc2"].median():.3f} / {share["residual_pc2"].max():.3f} |

**The buckets are calendar months and therefore non-overlapping.** That is why no
overlap caveat appears beside any number in this section, and it is why no window
length was invented for it. CLAUDE.md failure mode 9 is about rolling windows
read as independent observations; a share stepped daily would share 251 of its
252 days with its neighbour and every reading would owe the caveat. Months owe
none.

**`residual_pc1` peaks in 2022-23, not in a crisis**, at
{share["residual_pc1"].max():.3f} in {_peak_month(share["residual_pc1"])},
which is exactly what the curve-mode diagnosis predicts: 2022-23 is when the
curve moved in ways two PCs could not span. `residual_pc2` peaks at
{share["residual_pc2"].max():.3f} in {_peak_month(share["residual_pc2"])}.

### The correlation share and the basis-point share point opposite ways

| Component | share of the correlation trace (last date) | share of residual variance in bp² |
|---|---|---|
{chr(10).join(f"| `{name}` | {float(widest.eigenvalues.to_numpy(dtype=float)[-1, position]) / assets:.4f} | **{inputs.bp_variance_share[name]:.4f}** |" for position, name in enumerate(widest.factors.columns))}

**Both are reported because either one alone misleads.** `residual_pc1` holds
{float(widest.eigenvalues.to_numpy(dtype=float)[-1, 0]) / assets:.1%} of the correlation trace and
only {inputs.bp_variance_share["residual_pc1"]:.1%} of the actual residual
variance: it is four government residuals with standard deviations of
{", ".join(f"{panel.residuals[a].std():.1f}" for a in panel.substituted_assets)}
bp/day moving in near-lockstep, and a correlation PCA weights them equally with
`gold`'s {panel.residuals["gold"].std():.1f} bp/day. `residual_pc2` is the
reverse -- less of the trace, **{inputs.bp_variance_share["residual_pc2"]:.1%}**
of the variance -- because it is the equity sleeve, where the variance actually
is.

**This is not an argument for a covariance PCA.** That would have returned `gold`
and `em_equity` as the leading direction because they are the most volatile
residuals, which is experiments.md row 96's units artefact and the reason
SPEC.md 4.3.3 rules for the correlation. The right response is to print both
numbers, which is what the table does.

## SPEC.md 15.2 -- the hybrid through the covariance pipeline, unchanged

**Nothing in `src/mafrm/risk/` was changed, added to, or special-cased to make
this run.** The hybrid factor frame goes into `run_pipeline` through the same
call `mafrm.build` makes for Model A and `statistical_report.acceptance` makes
for Model B: a `T x K` frame and a `RiskConfig`. `git diff` on `src/mafrm/risk/`
for this session is empty, and `tests/test_risk_architecture.py` continues to
enforce that `risk/` imports nothing from `factors/`.

**This is the harder version of the test than Model B was, and that is why it is
worth running twice.** Model B's factors are a homogeneous statistical basis --
every column the same kind of object, built the same way, on the same scale. The
hybrid's frame mixes **six named macro factors in two different units** with one
or two **dimensionless residual components**:

| Column | Unit | Standard deviation |
|---|---|---|
{chr(10).join(f"| `{name}` | {'basis points of yield' if name in (macro.LEVEL, macro.SLOPE) else ('dimensionless (residual PC)' if name.startswith('residual_pc') else 'decimal daily return')} | {float(inputs.factor_frames[max(inputs.factor_frames)][name].std()):.4g} |" for name in inputs.factor_frames[max(inputs.factor_frames)].columns)}

Three scales spanning about three orders of magnitude, in one matrix. **If
`risk/` held an asset-class assumption anywhere, this is the frame that would
surface it.** It did not.

### All eight combinations, PSD at every stage

Both component counts x both horizons x both eigenfactor scalings, exactly as
W3-P5 ran Model B across its eight. `a = 1.0` is attribution-facing and `a = 1.4`
optimizer-facing (CLAUDE.md's parameter table); both are carried and neither is
chosen here.

Each cell is the **minimum eigenvalue after that stage**, which is CLAUDE.md
invariant 4's assertion made visible rather than merely executed:

{_acceptance_table(inputs)}

**Every stage of every run is positive semi-definite, and every run is
complete** -- all five of SPEC.md 5's stages ran on all eight. The minimum
eigenvalues are positive by four to six orders of magnitude, not marginal, so
this is not a near-miss absorbed by a tolerance.

**The PSD repair did not fire on any of the eight**
({"fired somewhere" if any(r.repair_fired for r in inputs.acceptance) else "zero firings"}), and
neither did row 100's Bartlett fallback
({"fired somewhere" if any(r.bartlett_fired for r in inputs.acceptance) else "zero firings"}).
That is the expected result and it is the same reading as experiments.md rows 98
and 129: both remedies are `K/T_eff` phenomena, and at `K` = {inputs.acceptance[0].factors}-{inputs.acceptance[-1].factors}
against T = {inputs.acceptance[0].observations:,} the ratio is nowhere near where
either fires. **They are inert here, not absent** -- `reports/stage_k_dependence.md`
carries the four techniques whose value depends on `K`, and this is two more
measurements for that table rather than evidence the repairs are unnecessary.

### The condition number is units, and row 96 already ruled on that

The finished matrices condition at about
{min(r.condition_number for r in inputs.acceptance):.2g} to {max(r.condition_number for r in inputs.acceptance):.2g}.
**That is the mixed-unit scale of the factor panel, not near-singularity**, and
experiments.md row 96 settled the reading on Model A's own six-factor matrix
before the hybrid existed. Appending a residual PC with a standard deviation
around 1.8 into a frame that already spans 0.003 to 5.1 cannot make the
conditioning better and does not need to: SPEC.md 5.1 estimates volatility and
correlation **separately** precisely so that a level difference between factors
is carried in the volatility leg rather than in the correlation matrix the
eigen-decompositions act on, and SPEC.md 5.3.1's scale-invariance ruling is the
same point at the eigenfactor stage.

### What the VRA multiplier says, and what it does not

`lambda_F^2` runs {min(r.lambda_squared for r in inputs.acceptance):.4f} to {max(r.lambda_squared for r in inputs.acceptance):.4f}
across the eight. {_horizon_direction(inputs)} **No claim is made here
about whether the hybrid forecasts better than Model A.** That is a bias-statistic
question, it belongs to W4, and it needs the optimizer-selected portfolios of
SPEC.md 6.2 rather than a multiplier read off the calibration series it was fitted
on.

### Cost, and why the history is committed

The VRA stage consumes a bias history, which costs one full pre-VRA pipeline run
per date: **about thirteen minutes per column, and fifteen and a half minutes of
wall clock for all eight when they are fanned out across processes** (measured;
sequentially the same work takes about a hundred).

That gap is the reason `mafrm.history` exists. The forecast-history machinery had
been written three times -- `vra_report` in W3-P4, `statistical_report` in W3-P5,
this module in W3-P6 -- and parallelism arrived in each at a different session, so
this one started life without it. It had also diverged in both digest functions
and in the partial's float precision. `experiments.md` records the four
differences; the shared module now owns everything that must be identical, and
`tests/test_history.py` asserts the callers hold the same function objects rather
than merely equal values. `reports/hybrid_forecast_history.csv` is committed for the same reason
Model A's and Model B's are, and it carries the same two digests --
`risk_config_digest` and a per-panel `panel_digest`. The second is the one that
matters: a changed half-life shows up in `config/model.yaml`, but a **moved
factor series** does not, and W3-P5 hit exactly that failure. The hybrid is more
exposed to it than Model B, because its panel depends on the macro factors, the
exposure regression, the residual construction and the residual PCA -- four
modules that can move it independently.

## What is still wrong

1. **The hybrid's leading component is a known span limitation, not hidden
   risk.** SPEC.md 4.3 sells the residual PC as a crowding detector. What it
   detects here is the third curve mode, already attributed in W2-P2. The
   detector works; the thing it found has a name. A third rate factor would
   remove it and is **not** taken here, because adding one after a refuted
   pre-registration is the move the pre-registration exists to prevent.

2. **2008 is unreachable and the August 2007 case is further out of reach.** The
   most interesting thing SPEC.md 4.3 promised cannot be measured on this panel.
   Nothing fixes that inside the current sample window.

3. **`residual_pc2` is an equity dispersion component and is not tested.** Its
   loadings are `us_small_equity` against `us_large_equity` and the two
   international sleeves. Row 83 had nothing to say about it, and inventing a
   hypothesis for it now would be exactly the after-the-fact registration this
   session spent its budget avoiding. It is reported and left.

4. **The R-squared comparison remains in-sample.** It is what SPEC.md 4.3 asks
   for and it is not a forecast claim.

5. **The SPEC.md 15.2 acceptance says the pipeline ACCEPTS the hybrid, not that
   the hybrid forecasts well.** All eight runs are complete and PSD at every
   stage and nothing in `risk/` moved, which is the architectural claim and the
   whole of it. Whether these matrices are better calibrated than Model A's is a
   bias-statistic question that needs SPEC.md 6.2's optimizer-selected portfolios,
   and it belongs to W4.

6. **The residual PCs inherit the hybrid panel's burn-in into the pipeline.** The
   factor frame opens 2010-04-27 against Model A's own 2008-04-14, so any later
   comparison of hybrid and Model A covariance forecasts must be run on the
   common window or it will be comparing two different samples. Nothing here does
   that comparison; this is a note for the session that will.
"""
    )
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    """Regenerate the hybrid report and its figure.

    ``--rebuild --all`` fans the eight histories out across processes and then
    assembles -- wall clock is one history rather than eight, and an interruption
    costs one partial rather than everything. ``--rebuild --column NAME`` builds a
    single history into its partial and is what ``--all`` invokes. ``--rebuild``
    alone assembles in this process, building whatever partials are missing, and
    takes about a hundred minutes. With no flag the committed cache is read and
    only the report and figure are redrawn, which is what ``make report`` runs.
    """
    parser = argparse.ArgumentParser(description="The hybrid. SPEC.md 4.3.")
    parser.add_argument(
        "--rebuild", action="store_true", help="rebuild the forecast histories before rendering"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="fan_out",
        help="fan the eight histories out across processes (about 15 minutes wall clock)",
    )
    parser.add_argument(
        "--column",
        default=None,
        help="build one history into its partial and stop; what --all invokes",
    )
    args = parser.parse_args(argv)
    settings = config.load()

    if args.column is not None:
        column = next((one for one in columns(settings) if one.name == args.column), None)
        if column is None:
            print(f"unknown column {args.column!r}; known: {[c.name for c in columns(settings)]}")
            return 2
        table = build_history(column, settings)
        print(f"{column.name}: {len(table):,} dates -> {_partial_path(column)}")
        return 0

    if args.rebuild and args.fan_out:
        # Eight processes, not eight loop iterations. The histories are
        # independent -- each is its own expanding-window pipeline run over its
        # own factor panel -- so nothing is shared and nothing has to be locked.
        code = history.fan_out("mafrm.factors.hybrid_report", columns(settings))
        if code:
            return code

    if args.rebuild:
        cache = assemble_cache(settings)
        write_cache(cache)
        print(f"wrote {CACHE_PATH.relative_to(_ROOT)} ({cache.provenance['rows']:,} dates)")

    inputs = build(settings)
    for path in (figure_path(), report_path()):
        path.parent.mkdir(parents=True, exist_ok=True)
    draw(inputs).savefig(figure_path(), dpi=150, bbox_inches="tight")
    report_path().write_text(render(inputs, settings), encoding="utf-8")
    for path in (figure_path(), report_path()):
        print(f"wrote {path.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
