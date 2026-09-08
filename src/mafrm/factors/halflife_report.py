"""``reports/halflife_sensitivity.md`` and ``.png``. SPEC.md 5.1's sweep, W4-P2b.

SPEC.md 5.1: *"Half-life selection is the main dial and there is no right answer
... ship a sensitivity chart of bias statistic and realised min-var portfolio
volatility across half-lives from 21 to 504 days."*

WHAT THIS IS, WHICH IS NOT WHAT IT LOOKS LIKE
---------------------------------------------

It looks like a robustness appendix. It is a ``K/T`` experiment. ``T_eff =
2*tau/ln 2``, so the nine-point grid sweeps ``K/T_eff`` over **24x** -- 0.0041 to
0.0990 -- **on one panel, with K, N, the asset class, the data source, the factor
construction and the scored window all held fixed.** Shepard's predicted
volatility understatement runs 0.8% to 23.2% across it.

SPEC.md 12's week-8 deliverable is measured optimizer-portfolio bias against
``K/T`` with Shepard's ``[1 - K/T]^-2`` overlaid, and until this session the
second end of that range depended entirely on W7's equity module arriving. It no
longer does. A within-panel range is also arguably the *cleaner* evidence: a
macro-versus-equity comparison moves ``K``, ``N``, the asset class and the
estimator's input data at once, while this moves one number.

THE CLASSIFICATION, WHICH IS THE PART THAT COULD BE ABUSED
-----------------------------------------------------------

``data-diagnostic``. ``N`` is unchanged. The shipped half-lives are the published
USE4 constants pinned in CLAUDE.md's parameter table and fixed long before this
grid existed, so **no result here may change which value ships** -- there is no
reversing result, and W3-P3's criterion makes it a diagnostic rather than a
trial. ``experiments.md``'s category table names half-lives as ``model-config``
*by example*, because in the general case a half-life sweep is a selection. It is
not one here.

The enforcement is written at ``experiments.md`` rows 165-173 and in SPEC.md
5.1.3, before the run: **the moment anyone proposes shipping a swept value, all
nine rows become ``model-config`` retroactively** -- the temptation is the
trigger, not the act -- and the grid is reported in full across its range rather
than only near the shipped points.

:func:`assert_shipped_on_grid` is that rule in code. It refuses to render if
either shipped half-life has left the grid, because the whole classification
rests on the shipped values being points *on* the curve rather than a separate
run beside it.

NINE HISTORIES AND NOT EIGHTEEN
--------------------------------

The expensive quantity is SPEC.md 5.3's Monte Carlo, and it depends on ``tau``
**alone** -- not on which horizon's shape is in force. Of the four
:class:`~mafrm.risk.config.RiskConfig` fields that differ between the horizons,
three cannot reach the factor covariance (``horizon`` is a label,
``specific_volatility_halflife`` enters SPEC.md 5.5 which is rebuilt at render,
``volatility_regime_halflife`` is post-VRA while this chart is pre-VRA) and the
fourth is the swept dial itself. So both horizons render from the same nine
files. :data:`COVARIANCE_FIELDS` and :func:`assert_horizon_independent` hold that
claim as an assertion rather than a comment, and SPEC.md 5.1.3 records the
measurement that established it: ``tau = 252`` built under the *short* shape
reproduces the committed ``long`` column to 4.5e-12 relative, the committed
cache's own 12-significant-digit write precision.

THE CACHE KEY IS THREE-PART
----------------------------

``history.risk_config_digest`` hashes both horizons' full ``RiskConfig``, so it
cannot separate grid points -- every point would present the same key while
describing a different model, which is a staleness guard protecting the wrong
thing (the defect ``history`` was extracted to fix). The partials therefore key on
``(shipped risk_config_digest, tau, panel_digest)``, carried as three separate
header fields so that a reader can see which one moved. That strengthens the
guard: a partial is reused only when the panel, the shipped configuration and the
grid point all match.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pandas as pd

from mafrm import config, history
from mafrm.factors import bias_report
from mafrm.numerics import effective_sample_size
from mafrm.risk import validation
from mafrm.risk.config import HORIZONS, RiskConfig
from mafrm.risk.shepard import second_order_risk

__all__ = [
    "COVARIANCE_FIELDS",
    "GridPoint",
    "PointRun",
    "assert_horizon_independent",
    "assert_shipped_on_grid",
    "build_cache",
    "chart",
    "common_dates",
    "effective_sample_size",
    "grid_points",
    "main",
    "partial_digests",
    "point_windows",
    "render",
    "run_grid",
    "shepard_understatement",
]

_REPORTS: Final[Path] = Path(__file__).resolve().parents[3] / "reports"
_CACHE_PATH: Final[Path] = _REPORTS / "halflife_forecast_history.csv"
_PARTIALS: Final[Path] = Path(__file__).resolve().parents[3] / "data" / "processed" / "halflife"
_MARKDOWN_PATH: Final[Path] = _REPORTS / "halflife_sensitivity.md"
_CHART_PATH: Final[Path] = _REPORTS / "halflife_sensitivity.png"

_REBUILD_HINT: Final[str] = "python -m mafrm.factors.halflife_report --rebuild"

#: Every ``RiskConfig`` field the factor-covariance pipeline reads. The nine
#: cached histories are a function of these and nothing else, which is what makes
#: one history per grid point sufficient for both horizons.
#:
#: Listed explicitly rather than derived by exclusion: a field added to
#: ``RiskConfig`` later is caught by :func:`assert_horizon_independent`, which
#: compares the two horizons on the fields NOT listed here and requires that the
#: only disagreements are ones the render stage applies. Getting this list wrong
#: in the unsafe direction -- omitting a field the pipeline does read -- would
#: mean silently rendering one horizon from the other's matrices.
COVARIANCE_FIELDS: Final[tuple[str, ...]] = (
    "volatility_halflife",
    "correlation_halflife",
    "mean_convention",
    "volatility_newey_west_lags",
    "correlation_newey_west_lags",
    "eigenfactor_monte_carlo_trials",
    "eigenfactor_scaling",
    "stages",
    "psd_eigenvalue_floor",
    "psd_minimum_eigenvalue",
    "symmetry_absolute_tolerance",
    "seed",
    "eigenfactor_variant",
)

#: The fields that legitimately differ between horizons and are applied when the
#: report is RENDERED rather than when the history is built: SPEC.md 5.5's
#: specific-risk half-life, SPEC.md 5.4's multiplier half-life, and the label.
_RENDER_FIELDS: Final[tuple[str, ...]] = (
    "horizon",
    "specific_volatility_halflife",
    "volatility_regime_halflife",
)


@dataclass(frozen=True)
class GridPoint:
    """One point of SPEC.md 5.1's sweep: a factor volatility half-life in days."""

    halflife: int

    @property
    def name(self) -> str:
        return f"hl{self.halflife}"


def grid_points(settings: config.Config) -> tuple[GridPoint, ...]:
    """``validation.halflife_sensitivity.grid``, as variants."""
    return tuple(
        GridPoint(halflife) for halflife in settings.model.validation.halflife_sensitivity.grid
    )


def assert_horizon_independent(settings: config.Config) -> None:
    """The nine-not-eighteen claim, as an assertion that runs on every build.

    Two halves. The horizons must agree on every field in
    :data:`COVARIANCE_FIELDS` once the swept dial is held equal -- otherwise a
    history built under one shape does not describe the other. And every field
    they disagree on must be one :data:`_RENDER_FIELDS` names, so that a field
    added to ``RiskConfig`` in a later session cannot quietly acquire a
    horizon-dependence that this module would render straight past.
    """
    short, long_ = (
        dataclasses.asdict(RiskConfig.load(horizon=horizon, config=settings))
        for horizon in HORIZONS
    )
    swept = "volatility_halflife"
    disagree = {key for key in short if short[key] != long_[key]}
    covariance_disagreements = sorted(disagree.intersection(COVARIANCE_FIELDS) - {swept})
    if covariance_disagreements:
        raise ValueError(
            f"halflife_report: the horizons disagree on {covariance_disagreements}, which the "
            "factor-covariance pipeline reads. SPEC.md 5.1.3's nine-histories-not-eighteen "
            "argument assumed the cached history is a function of the volatility half-life "
            "alone, and it no longer is. Either sweep per horizon or restore the agreement."
        )
    unaccounted = sorted(disagree - set(COVARIANCE_FIELDS) - set(_RENDER_FIELDS))
    if unaccounted:
        raise ValueError(
            f"halflife_report: {unaccounted} differ between the horizons and this module has "
            "no ruling on them. They are neither read by the covariance pipeline "
            "(COVARIANCE_FIELDS) nor applied at render (_RENDER_FIELDS). Classify them before "
            "rendering rather than letting the sweep pick one horizon's value silently."
        )


def assert_shipped_on_grid(settings: config.Config) -> None:
    """Both shipped half-lives must be POINTS ON the grid. SPEC.md 5.1.3.

    Not a tidiness check. The sweep is classified ``data-diagnostic`` because the
    shipped values were fixed before it ran and it reports them rather than
    selecting them -- and *"reports them"* means the published constants are
    points on the same curve, computed by the same code, on the same scored
    window. A grid that missed them would leave the report comparing a sweep
    against a separately-run pair, which is a different and much weaker claim.

    Also checks SPEC.md 5.1's endpoints and the grid's stated construction
    principle -- integer trading months -- because those are the justification
    ``config/model.yaml`` gives and an unchecked justification is a comment.
    """
    grid = settings.model.validation.halflife_sensitivity.grid
    shipped = {
        horizon: RiskConfig.load(horizon=horizon, config=settings).volatility_halflife
        for horizon in HORIZONS
    }
    missing = {horizon: value for horizon, value in shipped.items() if value not in grid}
    if missing:
        raise ValueError(
            f"halflife_report: the shipped volatility half-life is off the grid for {missing}. "
            "SPEC.md 5.1.3 classifies this sweep data-diagnostic on the ground that it REPORTS "
            "the published values rather than selecting among candidates, and that requires "
            f"them to be points on the curve. Grid: {list(grid)}."
        )
    month = settings.model.data.trading_days_per_month
    ragged = [value for value in grid if value % month]
    if ragged:
        raise ValueError(
            f"halflife_report: {ragged} are not integer trading months of {month} days. "
            "config/model.yaml justifies the grid by that principle and SPEC.md 5.1.3 records "
            "why the alternative -- justifying the nine points individually -- was refused."
        )


def partial_digests(
    point: GridPoint, data: bias_report.Inputs, settings: config.Config
) -> dict[str, str]:
    """SPEC.md 5.1.3's three-part cache key, carried as three visible fields.

    ``history.risk_config_digest`` alone cannot separate grid points: it hashes
    both horizons' shipped ``RiskConfig`` and every point of this sweep presents
    the same one. A guard that returns the same key for two different models is
    the defect ``mafrm.history`` was extracted to fix, so the half-life is
    carried beside it rather than folded into it -- when a partial is rejected,
    the header says which of the three moved.
    """
    return {
        "risk_config_digest": history.risk_config_digest(settings),
        "panel_digest": history.panel_digest(data.factors),
        "volatility_halflife": str(point.halflife),
    }


def risk_for(point: GridPoint, settings: config.Config) -> RiskConfig:
    """The shipped short-horizon config with the volatility half-life replaced.

    The *short* shape is an arbitrary choice and it is safe only because
    :func:`assert_horizon_independent` has just proved the cached history does
    not depend on it. Nothing else about the short horizon survives into the
    cache.
    """
    base = RiskConfig.load(horizon="short", config=settings)
    return dataclasses.replace(base, volatility_halflife=point.halflife)


def build_cache(
    data: bias_report.Inputs, settings: config.Config, *, only: str | None = None
) -> None:
    """Build one grid point's partial, or fan out across the grid and assemble.

    The fan-out is one process per grid point, uncapped, as
    ``history.fan_out`` has always been. Nine points is under the machine's
    physical core count; a materially larger grid would need a concurrency cap
    that does not exist yet, and would straggle on efficiency cores rather than
    fail, which is the failure mode worth naming before someone widens the grid.
    """
    assert_horizon_independent(settings)
    assert_shipped_on_grid(settings)
    points = grid_points(settings)
    if only is not None:
        chosen = [point for point in points if point.name == only]
        if not chosen:
            raise SystemExit(
                f"unknown grid point {only!r}; expected one of {[p.name for p in points]}"
            )
        point = chosen[0]
        history.load_or_build(
            history.partial_path(_PARTIALS, point),
            partial_digests(point, data, settings),
            lambda: bias_report.build_history(risk_for(point, settings), data, label=point.name),
        )
        return
    if history.fan_out("mafrm.factors.halflife_report", points, flag="--point"):
        raise SystemExit(1)
    pieces = []
    order: list[str] = []
    for point in points:
        piece = history.read_partial(history.partial_path(_PARTIALS, point))
        piece = piece.rename(columns={name: f"{point.name}_{name}" for name in piece.columns})
        order.extend(piece.columns)
        pieces.append(piece)
    cache = history.assemble(
        pieces,
        order=order,
        what=(
            "SPEC.md 5.1's half-life sweep (W4-P2b): SPEC.md 5.3's eigenfactor-adjusted factor "
            "covariance at both published a, one column block per volatility half-life on "
            "validation.halflife_sensitivity.grid, upper triangle row-major. Built by "
            f"`{_REBUILD_HINT}`. The history depends on the volatility half-life ALONE -- see "
            "halflife_report.COVARIANCE_FIELDS -- so BOTH horizons render from these nine "
            "blocks, differing only in the specific-risk half-life applied at render."
        ),
        settings=settings,
        panel_digests=history.panel_digest(data.factors),
        variants=points,
        extra={
            "grid": [point.halflife for point in points],
            "factors": int(data.factors.shape[1]),
            "assets": int(data.returns.shape[1]),
            "swept": "covariance.factor_volatility_halflife",
            "held": "covariance.factor_correlation_halflife (504, both horizons, pinned)",
        },
    )
    history.write_cache(cache, _CACHE_PATH)
    print(f"wrote {_CACHE_PATH} ({len(cache.frame):,} rows, {len(cache.frame.columns)} columns)")


# ---------------------------------------------------------------------------
# The scored window, common to every grid point
# ---------------------------------------------------------------------------


def point_windows(
    data: bias_report.Inputs, settings: config.Config
) -> dict[str, bias_report.ScoredWindow]:
    """Each grid point's OWN scored window, before any intersection.

    Cheap -- SPEC.md 5.1/5.2's stages run at well under a millisecond per date --
    and it has to be done per point because SPEC.md 5.2's repair boundary is a
    function of the half-life. SPEC.md 6.2.4 recorded the boundary landing in the
    same place at both shipped horizons despite a 3x half-life difference; across
    this grid's 24x that is measured rather than assumed.
    """
    windows: dict[str, bias_report.ScoredWindow] = {}
    for point in grid_points(settings):
        risk = risk_for(point, settings)
        stages = bias_report.stage_history(
            risk,
            data,
            start=bias_report.minimum_observations(risk, factors=data.factors.shape[1]),
        )
        windows[point.name] = bias_report.scored_window(
            stages,
            assets=data.returns.shape[1],
            clean=bias_report.first_clean_position(stages),
        )
    return windows


def common_dates(windows: dict[str, bias_report.ScoredWindow]) -> pd.DatetimeIndex:
    """The intersection of every grid point's scored window. SPEC.md 5.1.3."""
    shared: pd.DatetimeIndex | None = None
    for window in windows.values():
        shared = window.dates if shared is None else shared.intersection(window.dates)
    if shared is None or shared.empty:
        raise ValueError(
            "halflife_report.common_dates: the grid points share no scored dates. That is a "
            "defect in the panel or the grid, not a reason to score them on different samples."
        )
    return pd.DatetimeIndex(shared)


# ---------------------------------------------------------------------------
# Shepard, which is what the grid is actually sweeping
# ---------------------------------------------------------------------------


def cache_path() -> Path:
    """The sweep's committed cache, for the callers that read it through the shipped guard."""
    return _CACHE_PATH


def shepard_understatement(parameters: int, halflife: int) -> float:
    """``[1 - K/T_eff]^-2 - 1`` in SPEC.md 6.4's TABLE CONVENTION: variance multiplier minus one.

    ``parameters`` is ``K`` for the factor model (Eq. 32) and ``N`` for the
    asset-level sample comparand (Eq. 13) -- one formula, two ranks, which is the
    comparison SPEC.md 6.4's table is built on.

    W4-P3 (:mod:`mafrm.risk.shepard`): SPEC.md 6.4's table applies this number to
    volatility, and W4-P2b published the sweep in that convention. It is kept
    here so those figures reproduce, every label that prints it now says
    "variance", and the volatility understatement is the square root's -- half
    of this to first order. ``effective_sample_size`` is :mod:`mafrm.numerics`'s;
    the private copy this module carried is gone.
    """
    return second_order_risk(parameters, halflife=halflife).table_convention


@dataclass(frozen=True)
class PointRun:
    """One grid point scored under one horizon's shape, on the common window."""

    point: GridPoint
    horizon: str
    run: bias_report.HorizonRun

    @property
    def halflife(self) -> int:
        return self.point.halflife


def run_grid(
    data: bias_report.Inputs,
    cache: history.Cache,
    settings: config.Config,
    *,
    restrict: pd.DatetimeIndex,
) -> tuple[PointRun, ...]:
    """Score every grid point under both horizon shapes, all on ``restrict``.

    Eighteen runs from nine cached histories: the cached factor covariance is a
    function of the half-life alone (:data:`COVARIANCE_FIELDS`), and the horizon
    contributes only SPEC.md 5.5's specific-risk half-life, which is rebuilt here.
    """
    runs: list[PointRun] = []
    for point in grid_points(settings):
        for horizon in HORIZONS:
            base = RiskConfig.load(horizon=horizon, config=settings)
            risk = dataclasses.replace(base, volatility_halflife=point.halflife)
            runs.append(
                PointRun(
                    point=point,
                    horizon=horizon,
                    run=bias_report.run_for(
                        risk, point.name, data, cache, settings, restrict=restrict
                    ),
                )
            )
    return tuple(runs)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

#: The variant the chart is drawn on. PRE-VRA, ``a = 1.0`` -- SPEC.md 6.2.2's
#: headline, and the same one W4-P2 published 1.3321/1.3135 for. A post-VRA
#: sweep would show SPEC.md 5.4 absorbing each half-life's error rather than the
#: half-life's effect on calibration, and would read as flat for the wrong reason.
HEADLINE: Final[str] = "eigen_a1"
#: SPEC.md 6.2's naive comparand, carried because SPEC.md 6.4's Eq. 13 leg of the
#: K/T panel is a statement about it.
COMPARAND: Final[str] = "sample"


def _median(report: validation.ValidationReport, family: str) -> float:
    """One family's median ``B``, or NaN where the family has no members.

    SPEC.md 6.2's family 3 is the **eigenfactors** of the factor covariance, so
    the naive asset-level comparand has none: it has no factor structure to
    diagonalise, and ``bias_report`` builds its families with
    ``factor_returns=None`` for exactly that reason. Reporting NaN there is the
    honest cell -- the alternative would be to quote family 3 for a model that
    does not have one.
    """
    if not report.frame()["family"].eq(family).any():
        return float("nan")
    return report.median_bias(family)


def _row(entry: PointRun, variant: str) -> dict[str, float]:
    report = entry.run.runs[variant].report
    optimized = _median(report, validation.OPTIMIZED)
    random = _median(report, validation.RANDOM)
    return {
        "individual": _median(report, validation.INDIVIDUAL),
        "random": random,
        "eigenfactors": _median(report, validation.EIGENFACTOR),
        "optimized": optimized,
        "gap": optimized - random,
        "min_var_volatility": entry.run.runs[variant].min_var_realised_volatility,
        "mrad": report.mrad(validation.INDIVIDUAL),
    }


def render(
    runs: tuple[PointRun, ...],
    windows: dict[str, bias_report.ScoredWindow],
    dates: pd.DatetimeIndex,
    data: bias_report.Inputs,
    settings: config.Config,
) -> str:
    """The whole of ``reports/halflife_sensitivity.md``.

    ``K`` and ``N`` are read off the panel rather than written here: SPEC.md 6.4's
    two Shepard columns are Eq. 32 at the factor count and Eq. 13 at the asset
    count, and hard-coding either would be a magic number in a report whose entire
    subject is how those counts divide into ``T_eff``.
    """
    parameters = int(data.factors.shape[1])
    assets = int(data.returns.shape[1])
    battery = settings.model.validation
    shipped = {
        horizon: RiskConfig.load(horizon=horizon, config=settings).volatility_halflife
        for horizon in HORIZONS
    }
    grid = [point.halflife for point in grid_points(settings)]
    observations = len(dates)
    interval = validation.chi_square_interval(observations, level=battery.chi_square_level)
    normal = validation.normal_band(observations, z=battery.normal_band_z)

    lines = [
        "# Half-life sensitivity, and what it says about `K/T`",
        "",
        f"Generated {datetime.now(UTC).date().isoformat()} by "
        "`python -m mafrm.factors.halflife_report`. SPEC.md 5.1, registered in SPEC.md 5.1.3.",
        "",
        'SPEC.md 5.1: *"Half-life selection is the main dial and there is no right answer ...',
        "ship a sensitivity chart of bias statistic and realised min-var portfolio volatility",
        'across half-lives from 21 to 504 days."* This is that chart, and it turned out to be',
        "more than that.",
        "",
        "## This is a `K/T` experiment",
        "",
        "`T_eff = 2*tau/ln 2`, so sweeping the volatility half-life over 24x sweeps `K/T_eff`",
        f"over 24x -- {parameters / effective_sample_size(grid[-1]):.4f} to "
        f"{parameters / effective_sample_size(grid[0]):.4f} -- **on one panel, with `K`, `N`, the",
        "asset class, the data source, the factor construction and the scored window all held",
        "fixed.**",
        "Shepard's predicted understatement -- variance multiplier minus one, SPEC.md 6.4's table"
        " convention; the volatility figure is about half -- runs "
        f"{shepard_understatement(parameters, grid[-1]):.1%} to "
        f"{shepard_understatement(parameters, grid[0]):.1%}",
        "across it.",
        "",
        "SPEC.md 12's week-8 deliverable plots measured optimizer-portfolio bias against `K/T`",
        "with Shepard's `[1 - K/T]^-2` overlaid. Until this session the second end of that range",
        "depended entirely on W7's equity module arriving. It no longer does -- and a",
        "within-panel range is arguably the cleaner evidence, because a macro-versus-equity",
        "comparison moves `K`, `N`, the asset class and the input data at once while this moves",
        "one number. W7 remains the wider range and the harder test; it is no longer the only",
        "one.",
        "",
        "## What is swept, and what is NOT being selected",
        "",
        "The **factor volatility** half-life, alone. The correlation half-life is held at 504 at",
        "both horizons, because CLAUDE.md's parameter table pins it identically across horizons",
        "and marks that deliberate: it is not a dial in this model.",
        "",
        f"**The shipped values are unchanged: {shipped['short']}d at the short horizon and "
        f"{shipped['long']}d at the long one**, the published USE4 constants, fixed before this",
        "grid existed. They are points **on** the curve below, computed by the same code on the",
        "same scored window, not a separate run beside it -- `assert_shipped_on_grid` refuses to",
        "render otherwise. No result here may change which value ships; the moment one does,",
        "`experiments.md` rows 165-173 become `model-config` and count toward the deflated",
        "Sharpe trial count retroactively and in full. The sweep is reported across its whole",
        "range rather than near the shipped points, so nobody can claim the published values",
        "were confirmed by a curve that only examined its own neighbourhood.",
        "",
        "## The scored window, common to all nine points",
        "",
        'SPEC.md 6.2.4\'s *"the window is common to every variant"* extended one step. A curve of',
        "`B` against `tau` scored on nine different date sets would be partly a curve of `B`",
        "against **sample**.",
        "",
        f"- **Common window: {observations:,} dates, {dates[0].date()} to {dates[-1].date()}.**",
        f"- Exact chi-square {battery.chi_square_level:.0%} interval at `T = {observations:,}`: "
        f"{interval.render()}. Normal band (display convention, SPEC.md 6.1.1): {normal.render()}.",
        "",
        "| `tau` | own scored dates | own window starts | dropped to reach the common window |",
        "|---|---|---|---|",
    ]
    for point in grid_points(settings):
        window = windows[point.name]
        lost = window.observations - observations
        lines.append(
            f"| {point.halflife} | {window.observations:,} | {window.dates[0].date()} | {lost:,} |"
        )
    lines.extend(
        [
            "",
            "**CLAUDE.md failure mode 9.** Every `B` below is a full-sample statistic over the",
            f"{observations:,} common dates with its exact interval at that `T`; none is a rolling",
            "window and no count of windows is quoted as an `n`. The nine histories are expanding",
            "windows and consecutive dates share all but one observation, which is why the",
            "uncertainty quoted is the chi-square interval on the scored sample and not a spread",
            "across the grid.",
            "",
        ]
    )

    for horizon in HORIZONS:
        entries = [entry for entry in runs if entry.horizon == horizon]
        lines.extend(
            [
                f"## `{horizon}` horizon shape (SPEC.md 5.5 specific half-life "
                f"{entries[0].run.risk.specific_volatility_halflife}d)",
                "",
                "Pre-VRA, `a = 1.0` (SPEC.md 6.2.2). `B` is the median across each family's",
                "members; min-var volatility is in bps/day and is SPEC.md 6.5's headline",
                "model-comparison metric.",
                "",
                "| `tau` | `K/T_eff` | Shepard `K` (var) | Shepard `N` (var) | fam1 | fam2 | fam3 "
                "| **fam4** "
                "| **fam4-fam2** | min-var vol | naive vol |",
                "|---|---|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for entry in entries:
            row = _row(entry, HEADLINE)
            naive = _row(entry, COMPARAND)
            tau = entry.halflife
            mark = " **(shipped)**" if tau == shipped[horizon] else ""
            lines.append(
                f"| {tau}{mark} | {parameters / effective_sample_size(tau):.4f} "
                f"| {shepard_understatement(parameters, tau):.1%} "
                f"| {shepard_understatement(assets, tau):.1%} "
                f"| {row['individual']:.4f} | {row['random']:.4f} | {row['eigenfactors']:.4f} "
                f"| **{row['optimized']:.4f}** | **{row['gap']:+.4f}** "
                f"| {row['min_var_volatility']:.3f} | {naive['min_var_volatility']:.3f} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def chart(
    runs: tuple[PointRun, ...],
    dates: pd.DatetimeIndex,
    data: bias_report.Inputs,
    settings: config.Config,
    path: Path,
) -> None:
    """``reports/halflife_sensitivity.png``. Six panels, and the bottom two are the point.

    Rows 1-2 are what SPEC.md 5.1 asked for -- the bias statistic and the realised
    min-var volatility against the half-life. Row 3 is what the sweep turned out
    to be: `K/T_eff` moves 24x across this grid with everything else held fixed,
    so the same nine builds are a within-panel test of SPEC.md 6.4's Shepard
    correction, which SPEC.md 12 had planned to get from W7 alone.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parameters = int(data.factors.shape[1])
    assets = int(data.returns.shape[1])
    battery = settings.model.validation
    interval = validation.chi_square_interval(len(dates), level=battery.chi_square_level)
    grid = [point.halflife for point in grid_points(settings)]
    shipped = {
        horizon: RiskConfig.load(horizon=horizon, config=settings).volatility_halflife
        for horizon in HORIZONS
    }
    by_horizon = {
        horizon: [entry for entry in runs if entry.horizon == horizon] for horizon in HORIZONS
    }
    families = (
        (validation.INDIVIDUAL, "family 1 individual assets", "tab:grey"),
        (validation.RANDOM, "family 2 random portfolios", "tab:blue"),
        (validation.EIGENFACTOR, "family 3 eigenfactors", "tab:green"),
        (validation.OPTIMIZED, "family 4 min-var (optimizer-selected)", "tab:red"),
    )

    figure, axes = plt.subplots(3, 2, figsize=(13, 13))

    for column, horizon in enumerate(HORIZONS):
        axis = axes[0][column]
        entries = by_horizon[horizon]
        rows = [_row(entry, HEADLINE) for entry in entries]
        for key, label, colour in families:
            name = {
                validation.INDIVIDUAL: "individual",
                validation.RANDOM: "random",
                validation.EIGENFACTOR: "eigenfactors",
                validation.OPTIMIZED: "optimized",
            }[key]
            axis.plot(grid, [row[name] for row in rows], marker="o", color=colour, label=label)
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.axhspan(interval.lower, interval.upper, color="red", alpha=0.10)
        axis.axvline(shipped[horizon], color="black", linestyle=":", linewidth=1.2)
        axis.set_xscale("log")
        axis.set_xticks(grid)
        axis.set_xticklabels([str(value) for value in grid], fontsize=7)
        axis.set_xlabel("factor volatility half-life (trading days, log scale)")
        axis.set_ylabel("bias statistic B")
        axis.set_title(
            f"{horizon} shape: B against the half-life (pre-VRA, a = 1.0)\n"
            f"dotted line = shipped {shipped[horizon]}d; shaded = exact chi-square "
            f"{battery.chi_square_level:.0%} interval at T = {len(dates):,}",
            fontsize=9,
        )
        axis.legend(fontsize=7)
        axis.grid(alpha=0.3)

    axis = axes[1][0]
    for horizon, style in zip(HORIZONS, ("-", "--"), strict=True):
        entries = by_horizon[horizon]
        axis.plot(
            grid,
            [_row(entry, HEADLINE)["min_var_volatility"] for entry in entries],
            marker="o",
            linestyle=style,
            color="tab:red",
            label=f"{horizon}: factor model",
        )
        axis.plot(
            grid,
            [_row(entry, COMPARAND)["min_var_volatility"] for entry in entries],
            marker="s",
            linestyle=style,
            color="tab:purple",
            label=f"{horizon}: naive sample covariance",
        )
    axis.set_xscale("log")
    axis.set_xticks(grid)
    axis.set_xticklabels([str(value) for value in grid], fontsize=7)
    axis.set_xlabel("factor volatility half-life (trading days, log scale)")
    axis.set_ylabel("realised volatility, bps/day")
    axis.set_title(
        "SPEC.md 6.5's headline comparison: realised min-var portfolio volatility\n"
        "(Menchero & Ji: converges far faster than a realised information ratio)",
        fontsize=9,
    )
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)

    axis = axes[1][1]
    for horizon, colour in zip(HORIZONS, ("tab:red", "tab:orange"), strict=True):
        entries = by_horizon[horizon]
        gaps = [_row(entry, HEADLINE)["gap"] for entry in entries]
        axis.plot(grid, gaps, marker="o", color=colour, label=f"{horizon} shape")
        reference = gaps[grid.index(shipped[horizon])]
        axis.axhline(
            reference,
            color=colour,
            linestyle=":",
            linewidth=1.0,
            label=f"{horizon} at the shipped {shipped[horizon]}d = {reference:+.3f}",
        )
    axis.set_xscale("log")
    axis.set_xticks(grid)
    axis.set_xticklabels([str(value) for value in grid], fontsize=7)
    axis.set_xlabel("factor volatility half-life (trading days, log scale)")
    axis.set_ylabel("family 4 - family 2 bias gap")
    axis.set_title(
        "LEG (b) OF THE REGISTERED PREDICTION (SPEC.md 5.1.3)\n"
        "Delta's misspecification is constant in tau, so this should be flat.\n"
        "Falsified if it moves materially -- that puts the error back in F",
        fontsize=9,
    )
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)

    axis = axes[2][0]
    axis.plot(
        grid,
        [100 * shepard_understatement(parameters, tau) for tau in grid],
        marker="o",
        color="tab:blue",
        label=f"Shepard Eq. 32, K = {parameters} (the factor model)",
    )
    axis.plot(
        grid,
        [100 * shepard_understatement(assets, tau) for tau in grid],
        marker="s",
        color="tab:purple",
        label=f"Shepard Eq. 13, N = {assets} (the naive comparand)",
    )
    for horizon in HORIZONS:
        axis.axvline(shipped[horizon], color="black", linestyle=":", linewidth=1.0)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(grid)
    axis.set_xticklabels([str(value) for value in grid], fontsize=7)
    axis.set_xlabel("factor volatility half-life (trading days, log scale)")
    axis.set_ylabel("predicted understatement, % (variance multiplier - 1)")
    axis.set_title(
        f"WHY THIS IS A K/T EXPERIMENT: K/T_eff runs "
        f"{parameters / effective_sample_size(grid[-1]):.4f} to "
        f"{parameters / effective_sample_size(grid[0]):.4f} across the grid\n"
        "(24x, on one panel, with K, N, asset class and scored window all fixed)",
        fontsize=9,
    )
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3, which="both")

    axis = axes[2][1]
    for horizon, marker in zip(HORIZONS, ("o", "s"), strict=True):
        entries = by_horizon[horizon]
        predicted = [100 * shepard_understatement(parameters, entry.halflife) for entry in entries]
        for key, label, colour in families[:2]:
            name = "individual" if key == validation.INDIVIDUAL else "random"
            axis.plot(
                predicted,
                [100 * (_row(entry, HEADLINE)[name] - 1.0) for entry in entries],
                marker=marker,
                linestyle="-",
                color=colour,
                alpha=0.85,
                label=f"{horizon}: {label}",
            )
    limit = 100 * shepard_understatement(parameters, grid[0])
    axis.plot([0, limit], [0, limit], color="black", linewidth=0.9, label="Shepard exactly")
    axis.axhline(0.0, color="black", linewidth=0.6)
    axis.set_xlabel("Shepard's predicted understatement at this half-life, % (variance)")
    axis.set_ylabel("observed B - 1, %")
    axis.set_title(
        "LEG (a): the factor-covariance channel, against Shepard's closed form\n"
        "families 1 and 2 barely touch Delta, so this is where K/T should show",
        fontsize=9,
    )
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)

    figure.suptitle(
        "SPEC.md 5.1 half-life sensitivity -- pre-VRA bias statistic (SPEC.md 6.2.2), "
        f"common scored window of {len(dates):,} dates, "
        f"{dates[0].date()} to {dates[-1].date()}. "
        "data-diagnostic: the shipped half-lives are unchanged (SPEC.md 5.1.3).",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SPEC.md 5.1's half-life sensitivity sweep")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the forecast-history cache")
    parser.add_argument("--point", default=None, help="build one grid point's partial (internal)")
    args = parser.parse_args(argv)
    settings = config.load()
    if args.point is not None:
        build_cache(bias_report.inputs(settings), settings, only=args.point)
        return 0
    data = bias_report.inputs(settings)
    if args.rebuild:
        build_cache(data, settings)
        return 0
    assert_horizon_independent(settings)
    assert_shipped_on_grid(settings)
    cache = history.read_cache(
        _CACHE_PATH,
        rebuild_hint=_REBUILD_HINT,
        settings=settings,
        panel_digests={"macro": history.panel_digest(data.factors)},
    )
    windows = point_windows(data, settings)
    dates = common_dates(windows)
    runs = run_grid(data, cache, settings, restrict=dates)
    _MARKDOWN_PATH.write_text(render(runs, windows, dates, data, settings), encoding="utf-8")
    chart(runs, dates, data, settings, _CHART_PATH)
    print(f"wrote {_MARKDOWN_PATH}")
    print(f"wrote {_CHART_PATH}")
    print(f"  common scored window: {len(dates):,} dates, {dates[0].date()} to {dates[-1].date()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
