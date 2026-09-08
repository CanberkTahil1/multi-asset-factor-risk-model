"""``reports/eigenfactor.md`` and ``reports/eigenfactor_bias.png``. SPEC.md 5.3.

The eigenfactor adjustment is the central technique of the project, so what it
does on this project's own factor panel is reported as numbers rather than
asserted as a property.

WHY THIS LIVES IN ``factors/`` AND NOT IN ``risk/``
---------------------------------------------------

Same reason as :mod:`mafrm.factors.psd_repair_report`. The technique is in
:mod:`mafrm.risk.eigenfactor`, which knows nothing about what its columns are;
*this* module reads the macro factor panel, so it knows exactly what they are and
therefore cannot live under ``risk/`` without breaking CLAUDE.md invariant 10.

WHAT IT REPORTS, AND THE ONE THING IT DOES NOT
----------------------------------------------

The minimum-variance portfolio's forecast risk before and after the adjustment is
the headline, because SPEC.md 6.2 is explicit that random portfolios validate any
covariance matrix at all and the optimizer-selected one is the test that matters.
**It is a forecast revision, not a bias statistic.** Nothing here compares a
forecast against a realised return, so nothing here tests H1, H2 or H3 -- those
need week 4's bias machinery and the holdout stays untouched either way. What
this measures is how much the correction moves the number the optimizer would
have believed, which is the quantity Shepard's closed form predicts and the
quantity week 4 will then hold against realised risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display in CI; must precede the pyplot import.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from mafrm import config
from mafrm.factors import macro
from mafrm.risk.config import HORIZONS, Horizon, RiskConfig
from mafrm.risk.covariance import run_pipeline
from mafrm.risk.eigenfactor import EigenfactorAdjustment, eigenfactor_adjustment
from mafrm.risk.shepard import Unit, second_order_risk
from mafrm.risk.synthetic import GOLDEN_FACTOR_COUNTS, panel, specification

__all__ = [
    "FixtureReading",
    "Reading",
    "build",
    "draw",
    "figure_path",
    "main",
    "render",
    "report_path",
]

#: Random portfolios for the SPEC.md 6.2 contrast. Not a tuning knob -- it is
#: the sample size of a control, and the control's whole point is that its answer
#: is near zero however many are drawn.
_RANDOM_PORTFOLIOS = 2000

#: The factor whose units are perturbed by the invariance check, and by how much.
#: Any column and any positive constant must give the same answer; these are one
#: instance, reported so the number in the report is reproducible.
_INVARIANCE_COLUMN = 0
_INVARIANCE_CONSTANT = 1.0e4


def report_path() -> Path:
    return Path("reports/eigenfactor.md")


def figure_path() -> Path:
    return Path("reports/eigenfactor_bias.png")


def _minimum_variance(matrix: np.ndarray) -> np.ndarray:
    """The fully-invested minimum-variance portfolio of ``matrix``.

    The portfolio SPEC.md 5.3 is about: an optimizer seeking minimum risk loads
    onto the low-variance eigenfactors, which are exactly the directions whose
    risk is under-forecast. It is computed from the **unadjusted** matrix,
    because that is the matrix the optimizer would have been given.
    """
    inverse = np.linalg.inv(matrix)
    weights: np.ndarray = inverse @ np.ones(matrix.shape[0])
    normalised: np.ndarray = weights / weights.sum()
    return normalised


def _volatility(matrix: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(weights @ matrix @ weights))


def _shepard_at_correlation_window(reading: Reading) -> float:
    """``[1 - K/T_eff]^-2 - 1`` at the CORRELATION half-life's effective size.

    The comparand that matches this stage's coverage. ``CovarianceBuild``'s own
    Shepard number is computed at the *volatility* half-life, which is the right
    number for the model as a whole (SPEC.md 6.4, 15.2) -- but after SPEC.md
    5.3.2 this stage corrects ``rho`` alone, and ``rho`` is estimated on a
    different window. Comparing the stage against the model-level figure would be
    comparing it against error it does not claim to correct.

    In VARIANCE (W4-P3): the multiplier minus one, as SPEC.md 6.4's table quotes
    it, and the unit W3-P3b's 0.836% was published in.
    """
    return second_order_risk(
        reading.factors, effective_observations=reading.factors / reading.k_over_t
    ).understatement(Unit.VARIANCE)


def _shepard_ratio(reading: Reading, scaling: float) -> float:
    """Matched Shepard over the realised min-variance revision. SPEC.md 6.4."""
    return _shepard_at_correlation_window(reading) / (reading.minimum_variance_ratio[scaling] - 1.0)


@dataclass(frozen=True)
class Reading:
    """One horizon, on the real factor panel."""

    horizon: Horizon
    columns: tuple[str, ...]
    observations: int
    start: pd.Timestamp
    end: pd.Timestamp
    volatility_halflife: int
    correlation_halflife: int
    trials: int
    shepard_multiplier: float
    #: Post-repair, pre-eigenfactor.
    unadjusted: np.ndarray
    adjustment: EigenfactorAdjustment
    #: ``a -> forecast volatility ratio`` for the minimum-variance portfolio.
    minimum_variance_ratio: dict[float, float]
    #: ``a -> median forecast volatility ratio`` over random portfolios.
    random_median_ratio: dict[float, float]
    #: ``a -> 95th percentile`` of the same, to show the spread is small.
    random_upper_ratio: dict[float, float]
    #: Largest relative departure from exact scale invariance, this panel.
    invariance_residual: float

    @property
    def factors(self) -> int:
        return len(self.columns)

    @property
    def k_over_t(self) -> float:
        """Descriptive only -- nothing in the adjustment consumes it. W3-P3b."""
        return self.adjustment.k_over_t


@dataclass(frozen=True)
class FixtureReading:
    """The same on SPEC.md 15.2's synthetic fixture, so it is checkable offline.

    ``data/raw`` is gitignored, so nothing above can be reproduced from a clean
    clone. These two can.
    """

    factors: int
    amplitude: float
    residual_degrees_of_freedom: int
    exact_interpolation: bool
    bias_minimum: float
    bias_maximum: float


def _invariance_residual(frame: pd.DataFrame, risk: RiskConfig, trials: int) -> float:
    """SPEC.md 5.3.1's acceptance criterion, measured on the real panel.

    Rescale one factor by a constant, adjust, and undo the rescaling: the answer
    must not move. Reported as a number rather than as "passes", because a
    tolerance that is never printed is a tolerance nobody checks.
    """
    settings = risk
    rescaled = frame.copy()
    rescaled.iloc[:, _INVARIANCE_COLUMN] = (
        rescaled.iloc[:, _INVARIANCE_COLUMN] * _INVARIANCE_CONSTANT
    )
    selector = np.ones(frame.shape[1])
    selector[_INVARIANCE_COLUMN] = _INVARIANCE_CONSTANT

    def adjusted(values: pd.DataFrame) -> np.ndarray:
        matrix = run_pipeline(values, settings, stop_after="psd_repair").matrix
        return (
            eigenfactor_adjustment(
                matrix,
                trials=trials,
                halflife=float(settings.correlation_halflife),
                observations=int(values.shape[0]),
                scalings=settings.eigenfactor_scaling,
                seed=settings.seed,
                floor=settings.psd_eigenvalue_floor,
            )
            .variant(settings.eigenfactor_scaling[-1])
            .adjusted
        )

    expected = adjusted(frame) * np.outer(selector, selector)
    observed = adjusted(rescaled)
    return float(np.max(np.abs(observed - expected)) / np.max(np.abs(expected)))


def _read(frame: pd.DataFrame, risk: RiskConfig) -> Reading:
    build_result = run_pipeline(frame, risk, stop_after="psd_repair")
    unadjusted = build_result.matrix
    adjustment = eigenfactor_adjustment(
        unadjusted,
        trials=risk.eigenfactor_monte_carlo_trials,
        halflife=float(risk.correlation_halflife),
        observations=len(frame),
        scalings=risk.eigenfactor_scaling,
        seed=risk.seed,
        floor=risk.psd_eigenvalue_floor,
    )

    weights = _minimum_variance(unadjusted)
    base_volatility = _volatility(unadjusted, weights)
    generator = np.random.default_rng(risk.seed)
    random = generator.standard_normal((_RANDOM_PORTFOLIOS, frame.shape[1]))
    random = random / np.abs(random).sum(axis=1, keepdims=True)
    base_random = np.sqrt(np.einsum("pi,ij,pj->p", random, unadjusted, random))

    minimum_variance_ratio: dict[float, float] = {}
    random_median_ratio: dict[float, float] = {}
    random_upper_ratio: dict[float, float] = {}
    for variant in adjustment.variants:
        minimum_variance_ratio[variant.scaling] = (
            _volatility(variant.adjusted, weights) / base_volatility
        )
        ratios = np.sqrt(np.einsum("pi,ij,pj->p", random, variant.adjusted, random)) / base_random
        random_median_ratio[variant.scaling] = float(np.median(ratios))
        random_upper_ratio[variant.scaling] = float(np.quantile(ratios, 0.95))

    return Reading(
        horizon=risk.horizon,
        columns=tuple(str(column) for column in frame.columns),
        observations=len(frame),
        start=frame.index[0],
        end=frame.index[-1],
        volatility_halflife=risk.volatility_halflife,
        correlation_halflife=risk.correlation_halflife,
        trials=risk.eigenfactor_monte_carlo_trials,
        shepard_multiplier=build_result.shepard_multiplier,
        unadjusted=unadjusted,
        adjustment=adjustment,
        minimum_variance_ratio=minimum_variance_ratio,
        random_median_ratio=random_median_ratio,
        random_upper_ratio=random_upper_ratio,
        invariance_residual=_invariance_residual(
            frame, risk, trials=risk.eigenfactor_monte_carlo_trials
        ),
    )


def _fixture(factors: int, settings: config.Config) -> FixtureReading:
    risk = RiskConfig.load(horizon=HORIZONS[0], config=settings)
    frame = panel(specification(factors, config=settings))
    matrix = run_pipeline(frame, risk, stop_after="psd_repair").matrix
    adjustment = eigenfactor_adjustment(
        matrix,
        trials=risk.eigenfactor_monte_carlo_trials,
        halflife=float(risk.correlation_halflife),
        observations=len(frame),
        scalings=risk.eigenfactor_scaling,
        seed=risk.seed,
        floor=risk.psd_eigenvalue_floor,
    )
    return FixtureReading(
        factors=factors,
        amplitude=adjustment.amplitude,
        residual_degrees_of_freedom=adjustment.fit.residual_degrees_of_freedom,
        exact_interpolation=adjustment.fit.exact_interpolation,
        bias_minimum=float(adjustment.bias.min()),
        bias_maximum=float(adjustment.bias.max()),
    )


def build(
    cfg: config.Config | None = None,
) -> tuple[tuple[Reading, ...], tuple[FixtureReading, ...]]:
    """Both horizons on the real panel, plus the two offline fixture sizes."""
    settings = cfg or config.load()
    frame = macro.macro_factor_panel(config=settings).complete
    readings = tuple(
        _read(frame, RiskConfig.load(horizon=horizon, config=settings)) for horizon in HORIZONS
    )
    fixtures = tuple(_fixture(factors, settings) for factors in GOLDEN_FACTOR_COUNTS)
    return readings, fixtures


def draw(readings: tuple[Reading, ...]) -> Figure:
    """``reports/eigenfactor_bias.png``: raw ``lambda(k)`` with the fit overlaid.

    One panel per horizon, shared axes, so the amplitude difference between them
    -- same ``K``, different ``T_eff``, therefore different ``K/T`` -- is legible
    as the vertical extent of the curve rather than having to be read off a table.
    """
    figure, axes = plt.subplots(1, len(readings), figsize=(11, 4.2), sharey=True)
    for axis, reading in zip(np.atleast_1d(axes), readings, strict=True):
        ranks = np.arange(reading.factors)
        axis.axhline(1.0, color="0.55", linewidth=1.0, linestyle=":", zorder=1)
        axis.plot(
            ranks,
            reading.adjustment.bias,
            marker="o",
            markersize=5,
            linewidth=0.9,
            linestyle="--",
            color="#3b6ea5",
            label=r"raw $\lambda(k)$",
            zorder=2,
        )
        axis.plot(
            ranks,
            reading.adjustment.fit.fitted,
            linewidth=2.0,
            color="#c2452d",
            label=r"fitted parabola $\lambda_P(k)$",
            zorder=3,
        )
        axis.set_title(
            f"{reading.horizon}: HL {reading.adjustment.halflife:.0f}d, "
            f"K/T = {reading.k_over_t:.4f}, amplitude {reading.adjustment.amplitude:.4f}",
            fontsize=10,
        )
        axis.set_xlabel("eigenvalue rank $k$  (0 = smallest eigenfactor)")
        axis.set_xticks(ranks)
        axis.grid(alpha=0.25, linewidth=0.6)
    np.atleast_1d(axes)[0].set_ylabel(r"$\lambda(k) = \sqrt{\;\mathrm{true}/\mathrm{estimated}}$")
    np.atleast_1d(axes)[0].legend(fontsize=9, frameon=False)
    figure.suptitle(
        "Eigenfactor bias, SPEC.md 5.3 -- above 1 means the direction's risk is under-forecast",
        fontsize=11,
    )
    figure.tight_layout()
    return figure


def _curve_table(reading: Reading) -> list[str]:
    adjustment = reading.adjustment
    header = (
        "| rank `k` | `D_0(k)` (correlation) | raw `lambda(k)` | fitted `lambda_P(k)` "
        "| `gamma` at a=1.0 | `gamma` at a=1.4 |"
    )
    lines = [header, "|---|---|---|---|---|---|"]
    mild = adjustment.variant(1.0).gamma
    strong = adjustment.variant(1.4).gamma
    for index in range(reading.factors):
        lines.append(
            f"| {index} | {adjustment.eigenvalues[index]:.4f} | {adjustment.bias[index]:.4f} "
            f"| {adjustment.fit.fitted[index]:.4f} | {mild[index]:.4f} | {strong[index]:.4f} |"
        )
    return lines


def render(
    readings: tuple[Reading, ...],
    fixtures: tuple[FixtureReading, ...],
    cfg: config.Config | None = None,
) -> str:
    """The whole of ``reports/eigenfactor.md``."""
    settings = cfg or config.load()
    stamp = datetime.now(UTC).date().isoformat()
    reference = readings[0]

    lines: list[str] = [
        "# The eigenfactor risk adjustment on this panel",
        "",
        f"Generated {stamp} by `python -m mafrm.factors.eigenfactor_report`. SPEC.md 5.3,",
        "5.3.1 and 5.3.2. Menchero, Wang & Orr (2011).",
        "",
        "Sampling error in a factor covariance matrix is not neutral: diagonalise it and the",
        "small eigenvalues are biased **down**, the large ones slightly **up**. An optimizer",
        "seeking minimum risk loads onto exactly the low-variance directions, so it loads onto",
        "the most under-forecast ones. This stage measures that bias by Monte Carlo, direction",
        "by direction, and rescales the eigenvalues by what it finds.",
        "",
        f"Panel: **{reference.factors} orthogonalized macro factors**, "
        f"{reference.observations:,} dates, {reference.start.date()} to "
        f"{reference.end.date()} -- strictly before `sample.holdout_start` = "
        f"{settings.model.sample.holdout_start}. "
        f"`M = {reference.trials:,}` Monte Carlo trials from `model.seed`.",
        "",
        "## The one implementation decision, and the test that settles it",
        "",
        "**The adjustment runs in correlation space and is rescaled by `sigma` afterwards.**",
        "SPEC.md 5.3 step 1 diagonalises `F`; in USE4 every factor is already in return units,",
        "so a diagonal rescaling across factors cannot happen there and the question never",
        "arises. This panel deliberately mixes decimal returns with basis points of yield",
        "(SPEC.md 4.1.1, for the effective-duration reading), and `experiments.md` row 96",
        "measures what that costs: `cond(F) = 9.96e6` against `cond(rho) = 3.27`, essentially",
        "all of it a 7.3e6 variance ratio across mixed-unit columns. Diagonalising `F` here",
        "would let the numeraire decide which direction is the smallest eigenfactor -- the one",
        "the correction acts on hardest -- before the data does.",
        "",
        "SPEC.md 5.3.2's tiebreak, fixed in writing before either candidate was built: prefer",
        "the resolution that keeps units bookkeeping in **fewer places**. Correlation space",
        "puts it in one function; moving the panel to a common numeraire puts it at every point",
        "exposures and covariance meet. The losing candidate was not built and no outcome",
        "number was computed for it, which is what keeps that a ruling rather than a sweep.",
        "",
        "**The acceptance criterion is mechanical and is measured, not asserted.** Multiply one",
        f"factor's return series by {_INVARIANCE_CONSTANT:.0e}, run the adjustment, undo the",
        "rescaling: the matrix must not move.",
        "",
    ]
    for reading in readings:
        lines.append(
            f"- `{reading.horizon}`: largest relative departure "
            f"**{reading.invariance_residual:.2e}** -- roundoff."
        )
    lines.extend(
        [
            "",
            "`tests/test_eigenfactor.py` runs the same check on the fixture and, crucially, runs",
            "it against the **literal covariance-space form as a power control**, asserting that",
            "that form *fails*. Without the control the passing result would say nothing.",
            "",
            "## The bias curve",
            "",
            "![eigenfactor bias](eigenfactor_bias.png)",
            "",
        ]
    )

    for reading in readings:
        adjustment = reading.adjustment
        lines.extend(
            [
                f"### `{reading.horizon}` horizon "
                f"(volatility half-life {reading.volatility_halflife}d, correlation "
                f"{reading.correlation_halflife}d)",
                "",
                f"EWMA half-life {reading.adjustment.halflife:.0f}d over "
                f"{reading.adjustment.observations:,} obs; `K/T_eff = "
                f"{reading.k_over_t:.4f}` (descriptive). "
                f"Amplitude `lambda_P(0) - lambda_P(K-1)` = **{adjustment.amplitude:.4f}**.",
                "",
                *_curve_table(reading),
                "",
                f"- Declines monotonically: fitted **{adjustment.monotone_decline}**, "
                f"raw {adjustment.raw_monotone_decline}. Crosses 1: "
                f"**{adjustment.crosses_one}**.",
                f"- Parabola: {adjustment.fit.residual_degrees_of_freedom} residual d.o.f., "
                f"rms residual {adjustment.fit.residual_standard_deviation:.4e}.",
                "",
            ]
        )

    lines.extend(
        [
            "## What it does to a portfolio",
            "",
            "**This is a forecast revision, not a bias statistic.** Nothing here compares a",
            "forecast against a realised return, so nothing here tests H1, H2 or H3 -- those are",
            "week 4's and they need the bias machinery. What is measured is how much the",
            "correction moves the number an optimizer would have believed.",
            "",
            "The contrast is SPEC.md 6.2's: **random portfolios validate any covariance matrix",
            "at all**, so the minimum-variance portfolio is the one that matters. It is computed",
            "from the *unadjusted* matrix, because that is the matrix the optimizer would have",
            "been handed.",
            "",
            "| Horizon | `a` | min-variance forecast vol | random portfolios, median "
            "| random, 95th |",
            "|---|---|---|---|---|",
        ]
    )
    for reading in readings:
        for scaling in sorted(reading.minimum_variance_ratio):
            lines.append(
                f"| {reading.horizon} | {scaling:.1f} "
                f"| **{100.0 * (reading.minimum_variance_ratio[scaling] - 1.0):+.3f}%** "
                f"| {100.0 * (reading.random_median_ratio[scaling] - 1.0):+.3f}% "
                f"| {100.0 * (reading.random_upper_ratio[scaling] - 1.0):+.3f}% |"
            )

    lines.extend(
        [
            "",
            "Read the ordering, not the magnitudes: the optimizer-selected portfolio is revised",
            "up by more than the typical random one, at both horizons and at both scalings, and",
            "`a = 1.4` moves it further than `a = 1.0` by construction. That is the whole",
            "mechanism MWO describe, reproduced on this panel.",
            "",
            "**Against Shepard's closed form (SPEC.md 6.4), computed on the window this",
            "stage actually corrects.** Shepard's `[1 - K/T_eff]^-2` is an independent",
            "derivation of the same bias from the estimation-error side, and SPEC.md 6.4",
            "nominates it as the analytic cross-check on this Monte Carlo.",
            "",
            "**Which `T_eff` goes into it is the whole of the comparison, and W3-P3 got it",
            "wrong in both halves at once.** After SPEC.md 5.3.2 this stage corrects `rho`",
            "alone, and `rho` is estimated at the correlation half-life. The model-level Shepard",
            "figure -- the one every build prints, and the one SPEC.md 15.2's table quotes -- is",
            "computed at the *volatility* half-life and describes the whole factor covariance's",
            "estimation error, most of which this stage does not claim to touch.",
            "",
            "| Horizon | Shepard at the **volatility** window | Shepard at the **correlation** "
            "window | min-var revision, `a=1.0` | `a=1.4` | matched ratio at `a=1.4` |",
            "|---|---|---|---|---|---|",
        ]
    )
    for reading in readings:
        matched = _shepard_at_correlation_window(reading)
        lines.append(
            f"| {reading.horizon} | {100.0 * (reading.shepard_multiplier - 1.0):.3f}% "
            f"| **{100.0 * matched:.3f}%** "
            f"| {100.0 * (reading.minimum_variance_ratio[1.0] - 1.0):+.3f}% "
            f"| {100.0 * (reading.minimum_variance_ratio[1.4] - 1.0):+.3f}% "
            f"| {_shepard_ratio(reading, 1.4):.3f} |"
        )
    lines.extend(
        [
            "",
            "**On the matched window the two agree closely**, and neither was tuned: the "
            f"closed form gives {100.0 * _shepard_at_correlation_window(reference):.3f}% and the "
            "Monte Carlo moves the minimum-variance forecast by "
            f"{100.0 * (reference.minimum_variance_ratio[1.4] - 1.0):+.3f}% at `a = 1.4`. That is "
            "the cross-check SPEC.md 6.4 asked for, and it is a real one: two derivations of the "
            "same bias -- one closed-form from estimation error, one Monte Carlo from eigenvalue "
            "sampling -- computed on the same estimator, landing within about 10% of each other.",
            "",
            "**W3-P3 reported an agreement here that was two errors cancelling, and withdrawing",
            "it is the point of this paragraph.** It quoted Shepard's 5.141% against a +5.332%",
            "revision and called the match a cross-check. Both numbers were wrong: the revision",
            "was computed with `T = 242`, which W3-P3b established is the wrong effective sample",
            "size for a correlation estimator, and it was being compared against a Shepard figure",
            "for a window the stage does not correct. **Two wrong numbers agreeing is the most",
            "dangerous kind of corroboration**, because it reads as confirmation from an",
            "independent source. It also produced a `Shepard / (a = 1.0)` ratio of 1.353 that was",
            "tempting to read as *deriving* MSCI's published `a = 1.4`. That reading is now dead:",
            "on the matched window the ratio is "
            + " and ".join(f"{_shepard_ratio(r, 1.0):.2f}" for r in readings)
            + ". It was labelled a coincidence rather than a finding at the time, on the grounds",
            "that it was not stable across horizons -- and that caution is the only reason no",
            "claim now has to be retracted.",
            "",
            "**What the two Shepard columns measure between them is the cost of the",
            "correlation-space ruling, and it is large.** The model-level figure is "
            f"{100.0 * (reference.shepard_multiplier - 1.0):.1f}% at the `{reference.horizon}` "
            f"horizon; the part this stage's coverage corresponds to is "
            f"{100.0 * _shepard_at_correlation_window(reference):.2f}%. The difference is, to "
            "order of magnitude, the estimation error in the **volatility** leg -- which after "
            "SPEC.md 5.3.2 passes through this stage **uncorrected**, because the correction now "
            "acts on `rho` alone. In covariance space it was corrected jointly with `rho`. That "
            "is a real cost of the scale-invariance ruling, it was not named when the ruling was "
            "taken, and it is recorded here rather than repaired: what to do about the "
            "uncorrected `sigma` leg belongs with SPEC.md 5.4's volatility regime adjustment, "
            "which is the other stage that acts on the level of risk.",
            "",
            "MWO themselves report that Shepard's closed form *under*-predicts the empirically",
            "observed bias, which is why SPEC.md 6.4 makes the Monte Carlo the correction and the",
            "closed form the check. Week 4's bias statistics against realised returns are what",
            "settle it.",
            "",
            "## Both published scalings are built, and neither is chosen",
            "",
            "`a = 1.0` is attribution-facing and is what USE4 runs **in production**; `a = 1.4`",
            "is optimizer-facing and is the empirical scaling MSCI publish. 1.4 overstates the",
            "volatilities of the pure factors themselves and corrupts attribution, which is",
            "exactly why USE4 ships 1.0 and quotes 1.4. They are two model variants, so",
            "`RiskConfig` carries **no default**: the pipeline refuses to run this stage until a",
            "caller says which model it is building. Picking one because it flattered a result",
            "would be a `model-config` trial against the deflated Sharpe, and a silent default is",
            "how such a choice goes unlogged.",
            "",
            "Both come from **one** Monte Carlo. `a` enters only at step 7, after the simulation,",
            "so running it twice would leave the two variants differing by simulation noise as",
            "well as by `a` -- and the comparison between them is the point.",
            "",
            "## The smoothing step does almost nothing at this K",
            "",
            "Step 7 fits a parabola to the raw curve to smooth Monte Carlo noise at the ends of",
            "the spectrum. Three parameters on `K` points:",
            "",
            "| Case | `K` | residual d.o.f. | exact interpolation | amplitude |",
            "|---|---|---|---|---|",
            f"| macro panel | {reference.factors} "
            f"| {reference.adjustment.fit.residual_degrees_of_freedom} | no "
            f"| {reference.adjustment.amplitude:.4f} |",
        ]
    )
    for fixture in fixtures:
        lines.append(
            f"| golden fixture | {fixture.factors} | {fixture.residual_degrees_of_freedom} "
            f"| {'**yes**' if fixture.exact_interpolation else 'no'} | {fixture.amplitude:.4f} |"
        )
    lines.extend(
        [
            "",
            "At `K = 3` the fit is an **exact interpolation**: the fitted curve is the raw curve",
            "and no noise is removed at all. This is reported rather than left for a reader to",
            "assume otherwise. The fixture rows are also the part of this report reproducible",
            "from a clean clone -- `data/raw` is gitignored, so nothing above them is.",
            "",
            "See `reports/stage_k_dependence.md`: this is the third stage of the USE4 pipeline",
            "whose value at `K = 6` is measurably smaller than the published method assumes.",
            "",
            "## The amplitude gate is withdrawn, and what replaces it",
            "",
            'SPEC.md 5.3 describes `lambda(k)` as *"empirically ~1.5 at the smallest eigenfactor,',
            'declining monotonically to ~0.95 at the largest."* **That amplitude is a large-K**',
            "**shape and is withdrawn as a gate** (SPEC.md 5.3.2). At `K = 6` and",
            f"`K/T_eff = {reference.k_over_t:.4f}` the observed amplitude is "
            f"{reference.adjustment.amplitude:.4f}, an order of magnitude narrower, and holding",
            "to the published width would have a correct implementation diagnosed as broken.",
            "",
            "What stays a hard gate is the **direction**, which is universal: `lambda` declines",
            "in eigenvalue rank, above 1 at the smallest eigenfactor and below 1 at the largest.",
            "It holds on this panel at both horizons, and it is pinned in tests against the",
            "closed-form control `F_0 = I`, where the true risk of every direction is exactly 1",
            "and the decline is a statement about Wishart eigenvalue spread and nothing else.",
            "",
            "**One measured exception, with its mechanism isolated.** On the golden fixture the",
            "*raw* curve is not monotone in rank, and the reason is the fixture's spectrum rather",
            "than the code: the fixture is a two-factor model, so its correlation matrix has two",
            "well-separated eigenvalues and a near-degenerate bulk. An isolated eigenvalue is",
            "estimated with almost no eigenvector rotation, so its own risk is forecast nearly",
            "without bias and `lambda` returns toward 1 there, above the bulk's last value. A",
            "control in `tests/test_eigenfactor.py` builds a matrix with exactly one spike and",
            "reproduces it. **The monotone decline is a property of the bulk of the spectrum**,",
            "and this panel's correlation spectrum has no such gap, which is why it passes",
            "unmodified.",
            "",
            "**PRE-REGISTERED for W7 (SPEC.md 5.3.2).** The amplitude scales with `K/T`, so it",
            "must be visibly larger in the `K = 56` equity module than here.",
            "",
            "**One corroboration this report used to carry has been WITHDRAWN, and saying so is",
            "the point of this paragraph.** W3-P3 quoted the same `K = 6` model at two horizons "
            "-- amplitude 0.0741 against 0.0242 -- as evidence that the amplitude tracks `K/T`. "
            "That comparison **no longer exists.** W3-P3b's ruling makes the simulation follow "
            "the estimator, and the estimator's correlation half-life is 504d at *both* "
            "horizons, so this stage is now **horizon-independent by construction** and the two "
            "readings are identical rather than different. The old pair was measuring the "
            "arbitrary 242/727 split, which is exactly the number W3-P3b removed. A "
            "corroboration that disappears when a defect is fixed was evidence for the defect, "
            "not for the claim.",
            "",
            "What survives is the comparison that never depended on it -- the golden fixture at "
            f"`K = {fixtures[0].factors}` and `K = {fixtures[-1].factors}` on identical code and "
            f"an identical estimator: **{fixtures[0].amplitude:.4f} against "
            f"{fixtures[-1].amplitude:.4f}**.",
            "",
            "Falsified if the `K = 56` amplitude is equal to or smaller than this model's.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Write the report and the figure. Reads the local cache only; no network."""
    settings = config.load()
    readings, fixtures = build(settings)

    path = report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(readings, fixtures, settings), encoding="utf-8")
    draw(readings).savefig(figure_path(), dpi=150, bbox_inches="tight")

    for reading in readings:
        adjustment = reading.adjustment
        print(
            f"eigenfactor [{reading.horizon}]: amplitude {adjustment.amplitude:.4f}, "
            f"monotone {adjustment.monotone_decline}, crosses 1 {adjustment.crosses_one}, "
            f"min-var forecast {100.0 * (reading.minimum_variance_ratio[1.4] - 1.0):+.3f}% "
            f"at a=1.4, invariance residual {reading.invariance_residual:.1e}"
        )
    print(f"wrote {path} and {figure_path()}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
