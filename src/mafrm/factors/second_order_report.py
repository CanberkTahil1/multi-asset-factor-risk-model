"""``reports/second_order_risk.md`` and ``.png``. SPEC.md 6.4, 5.3.2, 5.4.2. W4-P3.

WHY THIS LIVES IN ``factors/``
------------------------------

:mod:`mafrm.risk.second_order` takes a covariance matrix and returns numbers.
*This* module knows which panel to take the matrix from (Model A's residual
panel, the one every W4 statistic is built on), which half-lives are shipped,
and which registrations in ``experiments.md`` the numbers are compared against.
None of that may cross into ``risk/`` (CLAUDE.md invariant 10).

WHAT IT ANSWERS
---------------

Three stages scale the level of risk and the question is whether they
double-count when composed (SPEC.md 5.4.2: "three-way, not two-way"). The
eigenfactor-VRA overlap was measured in W3-P4 (0.017-0.033% of volatility,
exactly quadratic in ``a``). This report measures the other two edges on the
pipeline's own estimators:

* **Shepard against the eigenfactor stage.** Shepard's closed form is for a
  whole covariance on one window; the eigenfactor stage corrects the ``rho`` leg
  alone, on a window six times longer than ``sigma``'s. Conditions (B), (C), (D)
  decompose the split-window bias and the interaction term says whether the
  legs add. If they do, the ``rho`` leg is the eigenfactor stage's (with Shepard
  at the correlation window as its analytic cross-check, as W3-P3b already
  reported) and the ``sigma`` leg is what remains uncorrected -- and the two may
  be quoted side by side without double counting.
* **Shepard against the VRA.** Second-order risk is a property of
  optimiser-selected directions; SPEC.md 5.4's ``B_t^F`` is fitted on the ``K``
  fixed factor axes. Condition (E) measures what a per-factor statistic sees of
  the estimation error: the Jensen term ``1 + 2/T_eff``, Shepard at ``p = 1``.
  That is the estimation-error side of the overlap, and it is the whole of it.

Every comparison below was registered with a numeric falsifier before the run
(``experiments.md`` rows 175-186); the thresholds are read from
``validation.second_order`` and the verdicts are printed against them.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from mafrm import config
from mafrm.factors import bias_report
from mafrm.numerics import effective_sample_size
from mafrm.risk import second_order
from mafrm.risk.checks import assert_psd
from mafrm.risk.config import HORIZONS, Horizon, RiskConfig
from mafrm.risk.covariance import separated_ewma_covariance
from mafrm.risk.shepard import SecondOrderRisk, Unit, second_order_risk

__all__ = [
    "Comparison",
    "HorizonDecomposition",
    "TauGridPoint",
    "compare",
    "compare_tau_grid",
    "main",
    "render",
    "render_tau_grid",
    "run",
    "run_tau_grid",
    "tau_grid_chart",
]

_REPORTS: Final[Path] = Path(__file__).resolve().parents[3] / "reports"
_MARKDOWN_PATH: Final[Path] = _REPORTS / "second_order_risk.md"
_CHART_PATH: Final[Path] = _REPORTS / "second_order_risk.png"
_TAU_CHART_PATH: Final[Path] = _REPORTS / "second_order_tau_grid.png"
#: experiments.md rows 318-326 (W8-P1), one per grid point in grid order.
_TAU_ROWS: Final[tuple[int, ...]] = (318, 319, 320, 321, 322, 323, 324, 325, 326)

#: SPEC.md 6.4's worked table was written at the universe's nominal asset count.
#: The panel carries 13; both are shown so the table's 13.6% is reproducible.
_SPEC_NOMINAL_ASSETS: Final[int] = 15


@dataclass(frozen=True)
class HorizonDecomposition:
    """One horizon's simulation on the panel's own ``F``, plus its ``rho = I`` control."""

    horizon: Horizon
    risk: RiskConfig
    panel: second_order.Decomposition
    identity_control: second_order.Decomposition
    #: The ``rho = I`` closed form at the control truth's own min-var weights.
    control_closed_form: float


@dataclass(frozen=True)
class Comparison:
    """One registered row, adjudicated against the number written before the run."""

    row: int
    label: str
    measured: float
    standard_error: float
    lower: float
    upper: float
    verdict: str

    @property
    def holds(self) -> bool:
        return self.verdict == "HOLDS"


def _within(measured: float, se: float, lower: float, upper: float) -> str:
    """Inside ``[lower, upper]`` widened by two Monte Carlo standard errors."""
    return "HOLDS" if lower - 2.0 * se <= measured <= upper + 2.0 * se else "REFUTED"


def run(data: bias_report.Inputs, settings: config.Config) -> dict[Horizon, HorizonDecomposition]:
    """Simulate both horizons from the panel's own separated-EWMA covariance."""
    second = settings.model.validation.second_order
    trials = int(config.resolve(settings.model, second.monte_carlo_trials_from))
    factors = data.factors.to_numpy(dtype=float)
    out: dict[Horizon, HorizonDecomposition] = {}
    for horizon in HORIZONS:
        risk = RiskConfig.load(horizon=horizon, config=settings)
        truth = separated_ewma_covariance(
            factors,
            volatility_halflife=float(risk.volatility_halflife),
            correlation_halflife=float(risk.correlation_halflife),
        )
        # CLAUDE.md invariant 4 at the one matrix this report constructs.
        assert_psd(truth, f"second_order.truth[{horizon}]", config=settings)
        panel = second_order.simulate(
            truth,
            volatility_halflife=float(risk.volatility_halflife),
            correlation_halflife=float(risk.correlation_halflife),
            observations=int(factors.shape[0]),
            trials=trials,
            seed=settings.model.seed,
        )
        control_truth = np.diag(np.diag(truth))
        control = second_order.simulate(
            control_truth,
            volatility_halflife=float(risk.volatility_halflife),
            correlation_halflife=float(risk.correlation_halflife),
            observations=int(factors.shape[0]),
            trials=trials,
            seed=settings.model.seed,
        )
        out[horizon] = HorizonDecomposition(
            horizon=horizon,
            risk=risk,
            panel=panel,
            identity_control=control,
            control_closed_form=control.volatility_leg_closed_form,
        )
    return out


def compare(
    result: HorizonDecomposition, settings: config.Config, *, rows: tuple[int, ...]
) -> tuple[Comparison, ...]:
    """Rows 175-180 (short) or 181-186 (long), in registration order."""
    second = settings.model.validation.second_order
    band = second.closed_form_relative_band
    panel = result.panel
    control = result.identity_control

    whole = panel.shepard_volatility_window.understatement(Unit.VARIANCE)
    single = Comparison(
        row=rows[0],
        label="(A) single-window joint against Shepard Eq. 32 at the volatility window",
        measured=panel.single_window.excess,
        standard_error=panel.single_window.standard_error,
        lower=whole * (1.0 - band),
        upper=whole * (1.0 + band),
        verdict=_within(
            panel.single_window.excess,
            panel.single_window.standard_error,
            whole * (1.0 - band),
            whole * (1.0 + band),
        ),
    )
    control_form = result.control_closed_form - 1.0
    instrument = Comparison(
        row=rows[1],
        label="rho = I control: (D) sigma-only against the derived closed form",
        measured=control.volatility_only.excess,
        standard_error=control.volatility_only.standard_error,
        lower=control_form * (1.0 - band),
        upper=control_form * (1.0 + band),
        verdict=_within(
            control.volatility_only.excess,
            control.volatility_only.standard_error,
            control_form * (1.0 - band),
            control_form * (1.0 + band),
        ),
    )
    at_asymptotic = panel.shepard_correlation_window.understatement(Unit.VARIANCE)
    at_equivalent = second_order_risk(
        panel.parameters, effective_observations=float(second.row_109_equivalent_correlation_t)
    ).understatement(Unit.VARIANCE)
    correlation_leg = Comparison(
        row=rows[2],
        label="(C) rho-only, between Shepard at row 109's T = 1890 and at T_eff = 1454",
        measured=panel.correlation_only.excess,
        standard_error=panel.correlation_only.standard_error,
        lower=at_equivalent,
        upper=at_asymptotic,
        verdict=_within(
            panel.correlation_only.excess,
            panel.correlation_only.standard_error,
            at_equivalent,
            at_asymptotic,
        ),
    )
    floor = panel.volatility_leg_closed_form - 1.0
    volatility_leg = Comparison(
        row=rows[3],
        label="(D) sigma-only, at least the rho = I closed form and below Shepard's whole",
        measured=panel.volatility_only.excess,
        standard_error=panel.volatility_only.standard_error,
        lower=floor,
        upper=whole,
        verdict=(
            "HOLDS"
            if panel.volatility_only.excess >= floor - 2.0 * panel.volatility_only.standard_error
            and panel.volatility_only.excess < whole
            else "REFUTED"
        ),
    )
    share = second.interaction_share_falsifier
    disjoint = Comparison(
        row=rows[4],
        label="(B) THE DISJOINTNESS TEST: |interaction| as a share of the joint excess",
        measured=abs(panel.interaction_share),
        standard_error=panel.interaction_standard_error / abs(panel.joint.excess),
        lower=0.0,
        upper=share,
        verdict="HOLDS" if abs(panel.interaction_share) <= share else "REFUTED",
    )
    jensen = panel.jensen.understatement(Unit.VARIANCE)
    per_factor = Comparison(
        row=rows[5],
        label="(E) per-factor B^2 - 1 against the Jensen term 1 + 2/T_eff (Shepard at p = 1)",
        measured=panel.per_factor.excess,
        standard_error=panel.per_factor.standard_error,
        lower=jensen,
        upper=jensen,
        verdict=_within(panel.per_factor.excess, panel.per_factor.standard_error, jensen, jensen),
    )
    return (single, instrument, correlation_leg, volatility_leg, disjoint, per_factor)


@dataclass(frozen=True)
class TauGridPoint:
    """The split-window simulation at one grid half-life, its own seed, the shipped truth."""

    halflife: int
    seed: int
    result: second_order.Decomposition

    @property
    def k_over_t(self) -> float:
        return self.result.parameters / effective_sample_size(float(self.halflife))


def run_tau_grid(data: bias_report.Inputs, settings: config.Config) -> tuple[TauGridPoint, ...]:
    """W4-P3's forward registration, run: ``simulate`` at every grid half-life, seed varied.

    The truth is the panel's own separated-EWMA covariance at the SHORT
    horizon's shipped half-lives -- rows 175-180's truth -- so that the only
    thing moving along the grid is the estimator's volatility window;
    ``tau_rho`` stays at the shipped correlation half-life. The seed is
    ``model.seed + grid index``, per rows 177/183's lesson that two runs sharing
    one seed are one measurement.
    """
    second = settings.model.validation.second_order
    grid = [int(h) for h in config.resolve(settings.model, second.tau_grid.grid_from)]
    trials = int(config.resolve(settings.model, second.monte_carlo_trials_from))
    factors = data.factors.to_numpy(dtype=float)
    risk = RiskConfig.load(horizon="short", config=settings)
    truth = separated_ewma_covariance(
        factors,
        volatility_halflife=float(risk.volatility_halflife),
        correlation_halflife=float(risk.correlation_halflife),
    )
    assert_psd(truth, "second_order.truth[tau_grid]", config=settings)
    out: list[TauGridPoint] = []
    for index, halflife in enumerate(grid):
        seed = int(settings.model.seed) + index
        out.append(
            TauGridPoint(
                halflife=halflife,
                seed=seed,
                result=second_order.simulate(
                    truth,
                    volatility_halflife=float(halflife),
                    correlation_halflife=float(risk.correlation_halflife),
                    observations=int(factors.shape[0]),
                    trials=trials,
                    seed=seed,
                ),
            )
        )
    return tuple(out)


def compare_tau_grid(
    points: Sequence[TauGridPoint], settings: config.Config, *, rows: Sequence[int] = _TAU_ROWS
) -> tuple[Comparison, ...]:
    """Rows 318-326: the rho leg at every point, and the joint at the shortest point (row 318).

    The first comparison is the joint at the shortest half-life against the
    registered band; the rest are the ``rho`` leg at each grid point against
    row 177's measured value within two Monte Carlo standard errors.
    """
    tau = settings.model.validation.second_order.tau_grid
    if len(rows) != len(points):
        raise ValueError(f"compare_tau_grid: {len(points)} points against {len(rows)} rows")
    shortest = min(points, key=lambda p: p.halflife)
    joint = Comparison(
        row=rows[0],
        label=(
            f"the split-window JOINT at tau = {shortest.halflife} against the registered "
            f"{100 * tau.joint_at_shortest_hypothesis:.1f}% (band "
            f"[{100 * tau.joint_at_shortest_lower:.1f}%, {100 * tau.joint_at_shortest_upper:.1f}%])"
        ),
        measured=shortest.result.joint.excess,
        standard_error=shortest.result.joint.standard_error,
        lower=tau.joint_at_shortest_lower,
        upper=tau.joint_at_shortest_upper,
        verdict=_within(
            shortest.result.joint.excess,
            shortest.result.joint.standard_error,
            tau.joint_at_shortest_lower,
            tau.joint_at_shortest_upper,
        ),
    )
    legs = [
        Comparison(
            row=row,
            label=(
                f"(C) rho-only at tau_sigma = {point.halflife} against row 177's "
                f"{100 * tau.correlation_leg_registered:.3f}% (tau_sigma-independence)"
            ),
            measured=point.result.correlation_only.excess,
            standard_error=point.result.correlation_only.standard_error,
            lower=tau.correlation_leg_registered,
            upper=tau.correlation_leg_registered,
            verdict=_within(
                point.result.correlation_only.excess,
                point.result.correlation_only.standard_error,
                tau.correlation_leg_registered,
                tau.correlation_leg_registered,
            ),
        )
        for row, point in zip(rows, points, strict=True)
    ]
    return (joint, *legs)


def render_tau_grid(points: Sequence[TauGridPoint], settings: config.Config) -> list[str]:
    """The ``tau`` grid section of ``reports/second_order_risk.md`` (W8-P1; SPEC.md 6.4.5)."""
    tau = settings.model.validation.second_order.tau_grid
    comparisons = compare_tau_grid(points, settings)
    joint, legs = comparisons[0], comparisons[1:]
    shortest = min(points, key=lambda p: p.halflife)
    whole_shortest = shortest.result.shepard_volatility_window.understatement(Unit.VARIANCE)
    out: list[str] = []
    add = out.append
    add("## The `tau` grid (W8-P1; experiments.md rows 318-326, registered W4-P3 after row 191)")
    add("")
    add(
        "The forward registration written at W4-P3, run: the same simulation at every point of "
        "SPEC.md 5.1.3's grid, the truth held at the short horizon's shipped `F`, `tau_rho` at "
        f"{points[0].result.correlation_halflife:.0f} throughout, `M = "
        f"{points[0].result.trials:,}`, "
        "and the SEED VARIED across grid points (`model.seed` + grid index) so that nine points "
        "are "
        "nine measurements. Every figure is in VARIANCE (`B^2 - 1`); `B` on volatility is beside "
        "it."
    )
    add("")
    add(
        f"**Row {joint.row}, the joint at `tau = {shortest.halflife}`: {joint.verdict}** -- "
        f"measured {100 * joint.measured:.2f}% +- {100 * joint.standard_error:.2f}% against the "
        f"registered {100 * tau.joint_at_shortest_hypothesis:.1f}% "
        f"[{100 * joint.lower:.1f}%, {100 * joint.upper:.1f}%]. Shepard's whole-covariance closed "
        "form "
        f"at that volatility window says "
        f"{100 * shortest.result.shepard_volatility_window.understatement(Unit.VARIANCE):.1f}%, "
        f"{whole_shortest / joint.measured:.1f}x "
        "the measured split-window bias."
    )
    add("")
    holding = sum(leg.holds for leg in legs)
    add(
        f"**The `rho` leg is `tau_sigma`-independent at {holding} of {len(legs)} grid points** "
        f"(within 2 MC s.e. of row 177's {100 * tau.correlation_leg_registered:.3f}%)"
        + (
            "; refuted at `tau` = "
            + ", ".join(
                str(p.halflife) for p, leg in zip(points, legs, strict=True) if not leg.holds
            )
            + "."
            if holding < len(legs)
            else "."
        )
    )
    add("")
    add(
        "| row | `tau_sigma` | `T_eff` asym. (Kish) | `K/T_eff` | seed | joint `B^2-1` (s.e.) | "
        "joint `B` | "
        "`rho` leg | `sigma` leg | interaction / joint | Shepard whole @ vol window | whole / "
        "joint | "
        "`rho` leg verdict |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for point, leg in zip(points, legs, strict=True):
        r = point.result
        whole = r.shepard_volatility_window.understatement(Unit.VARIANCE)
        add(
            f"| {leg.row} | {point.halflife} | {effective_sample_size(float(point.halflife)):.0f} "
            f"({r.realised_volatility_t_eff:.0f}) | {point.k_over_t:.4f} | {point.seed} | "
            f"{100 * r.joint.excess:.2f}% ({100 * r.joint.standard_error:.2f}%) | "
            f"{r.joint.bias:.4f} | "
            f"{100 * r.correlation_only.excess:.3f}% "
            f"({100 * r.correlation_only.standard_error:.3f}%) | "
            f"{100 * r.volatility_only.excess:.2f}% "
            f"({100 * r.volatility_only.standard_error:.2f}%) | "
            f"{100 * r.interaction_share:+.1f}% | {100 * whole:.1f}% | "
            f"{whole / r.joint.excess:.2f}x | "
            f"{leg.verdict} |"
        )
    add("")
    add(
        "**What the grid adds to SPEC.md 6.4.2.** The `rho` leg does not move with `tau_sigma` -- "
        "it "
        "cannot, since `tau_rho` is pinned -- so the whole of the split-window estimator's `tau` "
        "dependence is the `sigma` leg, which scales roughly as `1/T_eff` of the volatility "
        "window. "
        "The whole-covariance closed form at the volatility window over-predicts the estimator's "
        "bias by a factor that GROWS as the window shortens, because it charges `rho`'s "
        "estimation error at a window it is not estimated on. That ratio is the number the K/T "
        "chart (`reports/kt_scaling.md`) needs to read Shepard's curve honestly: on this estimator "
        "the curve is a comparand at the correlation window, and an upper bound at the volatility "
        "one."
    )
    add("")
    return out


def tau_grid_chart(points: Sequence[TauGridPoint], settings: config.Config, path: Path) -> None:
    """``reports/second_order_tau_grid.png``: the legs against ``K/T_eff`` at the volatility "
    "window."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tau = settings.model.validation.second_order.tau_grid
    x = np.array([p.k_over_t for p in points])
    fig, ax = plt.subplots(figsize=(8, 5))
    for attr, label, colour in (
        ("joint", "(B) split-window joint -- the pipeline", "tab:blue"),
        ("correlation_only", "(C) rho leg -- the eigenfactor stage's", "tab:green"),
        ("volatility_only", "(D) sigma leg -- uncorrected", "tab:orange"),
    ):
        y = np.array([100 * getattr(p.result, attr).excess for p in points])
        e = np.array([200 * getattr(p.result, attr).standard_error for p in points])
        ax.errorbar(x, y, yerr=e, marker="o", ms=4, capsize=2, label=label, color=colour)
    whole = np.array(
        [100 * p.result.shepard_volatility_window.understatement(Unit.VARIANCE) for p in points]
    )
    ax.plot(x, whole, "k--", lw=1, label="Shepard whole-covariance at the volatility window")
    ax.axhline(100 * tau.correlation_leg_registered, color="tab:green", lw=0.6, ls=":")
    ax.axhspan(
        100 * tau.joint_at_shortest_lower,
        100 * tau.joint_at_shortest_upper,
        color="tab:blue",
        alpha=0.08,
        label="registered band for the joint at the shortest tau",
    )
    for p in points:
        ax.annotate(
            f"tau={p.halflife}",
            (p.k_over_t, 100 * p.result.joint.excess),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.set_xlabel("K / T_eff at the volatility window (asymptotic)")
    ax.set_ylabel("B^2 - 1, % of variance (2 MC s.e.)")
    ax.set_title("The split-window estimator's second-order bias across the tau grid (W8-P1)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


_ROWS: Final[dict[str, tuple[int, ...]]] = {
    "short": (175, 176, 177, 178, 179, 180),
    "long": (181, 182, 183, 184, 185, 186),
}


def _worked_table(factors: int, assets: int, settings: config.Config) -> list[str]:
    """SPEC.md 6.4's table, computed from the config, in BOTH units and at both ``N``."""
    lines = [
        "| Setting | `p` | half-life | `T_eff` | `p/T_eff` | `[1 - p/T]^-2` "
        "| understatement (variance) | understatement (volatility) "
        "| SPEC.md 6.4 table convention |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for horizon in HORIZONS:
        halflife = RiskConfig.load(horizon=horizon, config=settings).volatility_halflife
        rows: list[tuple[str, SecondOrderRisk]] = [
            (
                f"Asset-level sample covariance, panel `N = {assets}`",
                second_order_risk(assets, halflife=halflife),
            ),
            (
                "Asset-level sample covariance, SPEC.md 6.4's nominal "
                f"`N = {_SPEC_NOMINAL_ASSETS}`",
                second_order_risk(_SPEC_NOMINAL_ASSETS, halflife=halflife),
            ),
            (f"{factors}-factor model", second_order_risk(factors, halflife=halflife)),
        ]
        for label, risk in rows:
            lines.append(
                f"| {label} | {risk.parameters} | {halflife}d | {risk.effective_observations:.1f} "
                f"| {risk.ratio:.4f} | {risk.variance_multiplier:.4f} "
                f"| **{risk.render(Unit.VARIANCE)}** | **{risk.render(Unit.VOLATILITY)}** "
                f"| {100 * risk.table_convention:.1f}% |"
            )
    return lines


def _condition_table(result: HorizonDecomposition) -> list[str]:
    panel = result.panel
    lines = [
        "| Condition | estimator | `B^2 - 1` (variance) | MC s.e. | `B` (volatility) "
        "| analytic comparand |",
        "|---|---|---|---|---|---|",
    ]
    comparands = {
        "single_window": (
            "Shepard Eq. 32 at the volatility window: "
            f"{panel.shepard_volatility_window.render(Unit.VARIANCE, decimals=2)}"
        ),
        "split_window_joint": "none in closed form -- the sum of (C) and (D) if disjoint",
        "correlation_only": (
            "Shepard at the correlation window: "
            f"{panel.shepard_correlation_window.render(Unit.VARIANCE, decimals=2)}; "
            "the eigenfactor stage's leg"
        ),
        "volatility_only": (
            f"`rho = I` closed form at the panel's min-var Herfindahl: "
            f"{100 * (panel.volatility_leg_closed_form - 1):.2f}% (variance), a floor"
        ),
        "per_factor": (
            f"Jensen `1 + 2/T_eff`: {panel.jensen.render(Unit.VARIANCE, decimals=2)}; exact "
            f"`E[1/X] - 1` for these EWMA weights: {100 * (panel.jensen_exact - 1):.3f}%"
        ),
    }
    descriptions = {
        "single_window": (
            f"(A) both legs at `tau = {panel.volatility_halflife:.0f}` -- Shepard's case"
        ),
        "split_window_joint": (
            f"(B) `sigma` at {panel.volatility_halflife:.0f}, `rho` at "
            f"{panel.correlation_halflife:.0f} -- SPEC.md 5.1's estimator"
        ),
        "correlation_only": "(C) `rho` estimated, `sigma` exact",
        "volatility_only": "(D) `sigma` estimated, `rho` exact",
        "per_factor": "(E) fixed factor axes under (B) -- what a per-factor `B_t^F` sees",
    }
    for condition in (
        panel.single_window,
        panel.joint,
        panel.correlation_only,
        panel.volatility_only,
        panel.per_factor,
    ):
        lines.append(
            f"| {descriptions[condition.name]} | `{condition.name}` "
            f"| **{100 * condition.excess:.3f}%** | {100 * condition.standard_error:.3f}% "
            f"| {condition.bias:.4f} | {comparands[condition.name]} |"
        )
    return lines


def _comparison_table(comparisons: Sequence[Comparison]) -> list[str]:
    lines = [
        "| Row | Registered comparison | measured | MC s.e. | registered band | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for item in comparisons:
        band = (
            f"[{100 * item.lower:.3f}%, {100 * item.upper:.3f}%] +- 2 s.e."
            if item.lower != item.upper
            else f"{100 * item.lower:.3f}% +- 2 s.e."
        )
        if item.row in (179, 185):
            band = f"<= {100 * item.upper:.0f}% of the joint excess"
        lines.append(
            f"| {item.row} | {item.label} | {100 * item.measured:.3f}% "
            f"| {100 * item.standard_error:.3f}% | {band} | **{item.verdict}** |"
        )
    return lines


def render(
    results: dict[Horizon, HorizonDecomposition],
    data: bias_report.Inputs,
    settings: config.Config,
    *,
    tau_grid: Sequence[TauGridPoint] | None = None,
) -> str:
    """The whole of ``reports/second_order_risk.md``; the ``tau`` grid section when it was run."""
    factors = len(data.factor_names)
    assets = len(data.assets)
    second = settings.model.validation.second_order
    short = results["short"].panel
    comparisons = {
        horizon: compare(results[horizon], settings, rows=_ROWS[horizon]) for horizon in HORIZONS
    }
    disjoint = all(comparisons[h][4].holds for h in HORIZONS)

    out: list[str] = []
    add = out.append
    add("# Second-order risk: Shepard's closed form, and whether three corrections double-count")
    add("")
    add(
        "Generated by `python -m mafrm.factors.second_order_report`. SPEC.md 6.4, 5.3.2, 5.4.2. "
        "W4-P3."
    )
    add("")
    add(
        "Shepard (2009) derives `[1 - K/T]^-2` for a covariance estimated on **one** window. "
        "This pipeline estimates `sigma` and `rho` on two, and SPEC.md 5.3 corrects `rho`'s "
        "sampling error alone. So the question a reader will ask -- may the eigenfactor stage's "
        "0.84% and Shepard's 5.14% be added? -- is a question about whether two corrections "
        "overlap, and CLAUDE.md failure mode 7 says not to answer it by assertion. This report "
        "answers it by simulation on the pipeline's own estimators, against comparisons "
        "registered in `experiments.md` rows 175-186 before any of it ran."
    )
    add("")
    add("## The unit, first")
    add("")
    add(
        "`[1 - K/T]^-2` multiplies a **variance** (Shepard Eq. 13 and 32 as written). MWO's "
        "cross-check in SPEC.md 6.4 -- Shepard 'predicts 2.0' at `T = 100, N = 50` against an "
        "observed bias statistic of 2.25 -- fixes the reading, because `(1 - 0.5)^-1 = 2.0` on "
        "volatility while `(1 - 0.5)^-2 = 4.0`. SPEC.md 6.4's worked table then applies the "
        "variance multiplier to volatility and calls the result a 'vol understatement'; that "
        "convention ran through every earlier report and is **kept and labelled** rather than "
        "silently corrected (W4-P3 ruling 6). No figure below is printed without its unit, and "
        "`mafrm.risk.shepard.SecondOrderRisk` refuses to format without one."
    )
    add("")
    add("## SPEC.md 6.4's worked table, computed from the config")
    add("")
    short_halflife = results["short"].risk.volatility_halflife
    factor_form = second_order_risk(factors, halflife=short_halflife).understatement(Unit.VARIANCE)
    panel_ratio = (
        second_order_risk(assets, halflife=short_halflife).understatement(Unit.VARIANCE)
        / factor_form
    )
    nominal_ratio = (
        second_order_risk(_SPEC_NOMINAL_ASSETS, halflife=short_halflife).understatement(
            Unit.VARIANCE
        )
        / factor_form
    )
    add(
        f"`T_eff = 2 tau / ln 2` at the shipped volatility half-lives; `K = {factors}` and "
        f"`N = {assets}` are read off the panel, and SPEC.md 6.4's nominal `N = "
        f"{_SPEC_NOMINAL_ASSETS}` is shown so its 13.6% reproduces. **H4's arithmetic**: at the "
        f"short horizon the factor structure divides the second-order risk by {panel_ratio:.2f}x "
        f"at `N = {assets}` in variance and by {nominal_ratio:.2f}x at `N = "
        f"{_SPEC_NOMINAL_ASSETS}` -- SPEC.md 6.4's 2.6x is the latter, to rounding. The "
        "empirical side of H4 is the simulation below and `experiments.md` rows 189-190, not "
        "this table."
    )
    add("")
    out.extend(_worked_table(factors, assets, settings))
    add("")
    add("## The decomposition, on the panel's own `F`")
    add("")
    add(
        f"Truth: `separated_ewma_covariance` of the {len(data.dates):,}-date factor panel at the "
        "shipped half-lives (PSD asserted). Each trial draws that many Gaussian rows and runs "
        "the pipeline's `ewma_second_moment` and `correlation_from_covariance`; the statistic is "
        "`E[w'Fw / w'F_hat w]` at the minimum-variance `w` of `F_hat`, the population `B^2` of "
        f"SPEC.md 6.2's family 4. `M = {short.trials:,}` trials from `model.seed`, read through "
        f"`{second.monte_carlo_trials_from}`. Realised Kish sizes on this window: "
        f"{short.realised_volatility_t_eff:.1f} at the volatility half-life against the asymptotic "
        f"{effective_sample_size(short.volatility_halflife):.1f}, and "
        f"{short.realised_correlation_t_eff:.1f} against "
        f"{effective_sample_size(short.correlation_halflife):.1f} for correlation (short horizon)."
    )
    add("")
    for horizon in HORIZONS:
        result = results[horizon]
        add(
            f"### `{horizon}` horizon: `tau_sigma = {result.risk.volatility_halflife}`, "
            f"`tau_rho = {result.risk.correlation_halflife}`"
        )
        add("")
        out.extend(_condition_table(result))
        add("")
        panel = result.panel
        add(
            f"Interaction `(B) - (C) - (D)` = **{100 * panel.interaction:+.4f}%** of variance "
            f"(MC s.e. {100 * panel.interaction_standard_error:.4f}%), i.e. "
            f"**{100 * panel.interaction_share:+.1f}%** of the joint excess. `rho = I` control at "
            "the panel's own variances: (D) measured "
            f"{100 * result.identity_control.volatility_only.excess:.3f}% "
            f"against the closed form {100 * (result.control_closed_form - 1):.3f}%."
        )
        add("")
        add("Registered comparisons, adjudicated:")
        add("")
        out.extend(_comparison_table(comparisons[horizon]))
        add("")

    add("## What this settles")
    add("")
    if disjoint:
        add(
            "**The legs add.** At both horizons the interaction between the `rho` leg and the "
            "`sigma` leg is inside the registered share of the joint excess, so the split-window "
            "estimator's second-order bias is, to Monte Carlo precision, the `rho`-only bias plus "
            "the `sigma`-only bias. That is the disjointness demonstration SPEC.md 5.4.2 asked "
            "for, by simulation, with the mechanism stated: for Gaussian data the sample "
            "correlation is independent of the sample variances, and the bias is a quadratic "
            "form in the estimation error."
        )
    else:
        add(
            "**The legs do NOT add within the registered share.** The interaction exceeds the "
            "falsifier at at least one horizon, so the eigenfactor stage and a `sigma`-leg figure "
            "may not be quoted side by side as a sum; only one is reported as a correction and "
            "the reason is this table."
        )
    add("")
    add("**The three-way accounting, one edge at a time.**")
    add("")
    add(
        "1. **Eigenfactor against the VRA** -- measured in W3-P4 (`reports/volatility_regime.md`, "
        "SPEC.md 5.4.2): the VRA absorbs 0.017-0.033% of volatility of the eigenfactor "
        "correction, exactly quadratic in `a`. Third-order, and rescalable rather than "
        "re-measured."
    )
    add(
        f"2. **Shepard against the eigenfactor stage** -- condition (C) is the leg SPEC.md 5.3 "
        f"corrects, and Shepard at the correlation window is its analytic cross-check "
        f"({short.shepard_correlation_window.render(Unit.VARIANCE, decimals=2)}, the 0.836% "
        "W3-P3b published). Condition (D) is what the correlation-space ruling left uncorrected. "
        "Because (C) and (D) add, quoting the eigenfactor stage's correction beside the `sigma` "
        "leg does not double-count. **What does double-count is Shepard's whole-covariance figure "
        "at the volatility window** -- it is not this estimator's bias at all: the split-window "
        f"joint (B) is {100 * short.joint.excess:.2f}% of variance against Eq. 32's "
        f"{short.shepard_volatility_window.render(Unit.VARIANCE, decimals=2)} at the short "
        "horizon, because `rho` sits on a window six times longer. **0.84% + 5.14% is therefore "
        "not a sum anyone should form**: the first term is inside the second, and the second is "
        "not the pipeline's."
    )
    exact = short.jensen_exact - 1.0
    add(
        f"3. **Shepard against the VRA** -- condition (E). What a fixed factor axis sees of the "
        "estimation error is the inverse-variance Jensen term, Shepard at `p = 1`, and it can be "
        f"computed exactly: `E[1/X] - 1` = **{100 * exact:.3f}%** of variance for the short "
        f"window's EWMA weights ({100 * (results['long'].panel.jensen_exact - 1):.3f}% long), "
        "against the first-order `1 + 2/T_eff` = "
        f"{short.jensen.render(Unit.VARIANCE, decimals=3)}. "
        f"The simulation's (E) is {100 * short.per_factor.excess:.2f}% +- "
        f"{100 * short.per_factor.standard_error:.2f}% -- see the verdict column and "
        "`experiments.md` row 191 for why the exact figure is the one to use. That exact term is "
        "the entire estimation-error content of SPEC.md 5.4's fitted multiplier, and so the size "
        f"of the Shepard-VRA overlap: **{exact / short.volatility_only.excess:.0%} of the `sigma` "
        f"leg and {exact / short.joint.excess:.0%} of the split-window joint**. The rest of the "
        "`sigma` leg is optimiser-specific and invisible to a statistic fitted on fixed axes."
    )
    add("")
    add("## What this does to W4-P2b's queued observation at `tau = 21`")
    add("")
    grid = settings.model.validation.halflife_sensitivity.grid
    shortest = grid[0]
    whole_21 = second_order_risk(factors, halflife=shortest)
    ratio_21 = short.correlation_halflife / shortest
    ratio_84 = short.correlation_halflife / short.volatility_halflife
    scaled_sigma = short.volatility_only.excess * (
        effective_sample_size(short.volatility_halflife) / effective_sample_size(shortest)
    )
    predicted_joint = short.correlation_only.excess + scaled_sigma
    overstatement = (
        short.shepard_volatility_window.understatement(Unit.VARIANCE) / short.joint.excess
    )
    overstatement_21 = whole_21.understatement(Unit.VARIANCE) / predicted_joint
    add(
        f"W4-P2b held one observation open for this session (`experiments.md` after row 173): at "
        f"`tau = {shortest}` Shepard's closed form predicts **{whole_21.render(Unit.VARIANCE)}** "
        f"understatement while family 2's observed `B - 1` was about 3% of volatility, roughly an "
        "eighth. Three things in that comparison are now known to be different objects, and they "
        "are separated here rather than left as one ratio."
    )
    add("")
    add(
        f"1. **The unit.** {whole_21.render(Unit.VARIANCE)} is {whole_21.render(Unit.VOLATILITY)}. "
        "Half the eighth is the factor of two between the two readings (6.4.1)."
    )
    add(
        "2. **The comparand.** Family 2 is estimation-independent: Shepard's formula describes the "
        "optimiser-selected portfolio, and what a fixed random direction sees of the estimation "
        "error is the Jensen term of condition (E) -- to first order `2/T_eff`, i.e. "
        f"{100 * 2 / effective_sample_size(shortest):.1f}% of variance "
        f"({100 * (np.sqrt(1 + 2 / effective_sample_size(shortest)) - 1):.1f}% of volatility) at "
        f"`tau = {shortest}`. That is arithmetic on a formula already in this report, not a "
        "measurement: it says about half of family 2's rise at the short end is the inverse-"
        "variance bias every fixed direction carries, and the rest is the real panel's departure "
        "from Gaussian. The like-for-like comparand, family 4, cannot be read at all on the real "
        "panel because its `B` is pinned near 1.33 by SPEC.md 6.2.6's specification error at every "
        "`tau` (rows 165-173)."
    )
    add(
        "3. **The window mismatch.** At the shipped short horizon the whole-covariance closed "
        f"form overstates the split-window estimator's bias by {overstatement:.1f}x "
        f"with `rho` on a window {ratio_84:.0f}x longer than `sigma`'s. At `tau = {shortest}` the "
        f"window ratio is {ratio_21:.0f}x, so the mismatch is larger, not smaller. **The share is "
        "not in hand and is not invented here.** The simulation has not been run at `tau = "
        f"{shortest}`, and it is registered as a W8-P1 question rather than opened now: scaling "
        "the "
        f"measured `sigma` leg by `T_eff` alone puts it at {100 * scaled_sigma:.1f}% and the "
        f"split-window joint at about **{100 * predicted_joint:.1f}% of variance** against the "
        f"closed form's {whole_21.render(Unit.VARIANCE)} -- a {overstatement_21:.1f}x "
        "overstatement -- but the `sigma` leg is not linear in `1/T_eff` at `K/T_eff` near 0.1, "
        "which is exactly what the run has to measure. Registered with that figure as the "
        "prediction and +-30% as the band, `experiments.md`, W8-P1 forward registration."
    )
    add("")
    add("**What is reported as the corrected forecast, and why.**")
    add("")
    add(
        "`reports/bias_statistics.md` prints the closed form beside every family-4 forecast: Eq. "
        "13 for the naive `sample` variant, whose estimator is exactly the one-window whole "
        "covariance the formula is derived for, and Eq. 32 at the volatility window for every "
        "factor variant, **labelled an upper bound** for the reason in point 2. The measured "
        "split-window bias above is the honest estimation-error figure for SPEC.md 5.1's "
        f"estimator: `B = {short.joint.bias:.4f}` short / `{results['long'].panel.joint.bias:.4f}` "
        "long, against a family-4 `B` near 1.33 on the real panel. The difference is SPEC.md "
        "6.2.6's specification error, and no closed form on `K/T` reaches it."
    )
    add("")
    add(
        "**Which `T_eff` case this is.** Every `sigma`-leg quantity is a second moment, where "
        "W3-P3b measured `2 tau / ln 2` transferring at 1.007x; row "
        f"{_ROWS['short'][0]} checks it here on Shepard's own case. The `rho` leg is the "
        "normalised "
        "case, where it does not, which is why row "
        f"{_ROWS['short'][2]} predicted (C) below the closed form at 1454 and the eigenfactor "
        "stage "
        "simulates its own estimator rather than reading a `T` off the formula."
    )
    add("")
    total = sum(len(items) for items in comparisons.values())
    holding = sum(item.holds for items in comparisons.values() for item in items)
    refuted = [str(item.row) for items in comparisons.values() for item in items if not item.holds]
    add(
        f"**{holding} of {total} registered comparisons hold"
        + (
            f"; rows {', '.join(refuted)} are refuted at the registration and explained at "
            "`experiments.md` row 191.**"
            if refuted
            else ".**"
        )
        + " Nothing here is a `model-config` row: no stage changed, nothing was selected, and "
        "`config/model.yaml`'s risk keys are untouched."
    )
    add("")
    if tau_grid is not None:
        out.extend(render_tau_grid(tau_grid, settings))
    return "\n".join(out) + "\n"


def chart(results: dict[Horizon, HorizonDecomposition], path: Path) -> None:
    """``reports/second_order_risk.png``: the five conditions per horizon with their comparands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for axis, horizon in zip(axes, HORIZONS, strict=True):
        panel = results[horizon].panel
        conditions = [
            panel.single_window,
            panel.joint,
            panel.correlation_only,
            panel.volatility_only,
            panel.per_factor,
        ]
        labels = [
            "(A) single\nwindow",
            "(B) split\njoint",
            "(C) rho\nonly",
            "(D) sigma\nonly",
            "(E) per\nfactor",
        ]
        heights = [100 * item.excess for item in conditions]
        errors = [200 * item.standard_error for item in conditions]
        axis.bar(labels, heights, yerr=errors, color="tab:blue", alpha=0.75, capsize=4)
        comparands = [
            panel.shepard_volatility_window.understatement(Unit.VARIANCE),
            panel.correlation_only.excess + panel.volatility_only.excess,
            panel.shepard_correlation_window.understatement(Unit.VARIANCE),
            panel.volatility_leg_closed_form - 1.0,
            panel.jensen.understatement(Unit.VARIANCE),
        ]
        axis.scatter(
            range(5),
            [100 * value for value in comparands],
            marker="_",
            s=600,
            color="black",
            zorder=3,
            label="comparand: Shepard / (C)+(D) / Shepard at corr. window / rho=I form / Jensen",
        )
        axis.set_title(
            f"{horizon}: tau_sigma = {panel.volatility_halflife:.0f}, "
            f"tau_rho = {panel.correlation_halflife:.0f}, M = {panel.trials:,}",
            fontsize=9,
        )
        axis.set_ylabel("B^2 - 1, % of variance (error bars: 2 MC s.e.)")
        axis.grid(alpha=0.3, axis="y")
        axis.legend(fontsize=7, loc="upper right")
    figure.suptitle(
        "Second-order risk decomposed: the rho leg and the sigma leg add, and the whole-covariance "
        "closed form (A) is not the split estimator's bias (B)",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - entry point
    parser = argparse.ArgumentParser(description="SPEC.md 6.4's second-order risk accounting")
    parser.parse_args(list(argv) if argv is not None else None)
    settings = config.load()
    data = bias_report.inputs(settings)
    results = run(data, settings)
    grid = run_tau_grid(data, settings)
    _MARKDOWN_PATH.write_text(render(results, data, settings, tau_grid=grid), encoding="utf-8")
    chart(results, _CHART_PATH)
    tau_grid_chart(grid, settings, _TAU_CHART_PATH)
    print(f"wrote {_MARKDOWN_PATH}")
    print(f"wrote {_CHART_PATH}")
    print(f"wrote {_TAU_CHART_PATH}")
    for horizon in HORIZONS:
        for item in compare(results[horizon], settings, rows=_ROWS[horizon]):
            print(f"  row {item.row}: {item.verdict:8s} {100 * item.measured:.3f}%  {item.label}")
    for item in compare_tau_grid(grid, settings):
        print(f"  row {item.row}: {item.verdict:8s} {100 * item.measured:.3f}%  {item.label}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
