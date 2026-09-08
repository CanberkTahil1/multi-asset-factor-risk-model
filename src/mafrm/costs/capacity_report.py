"""Generates ``reports/capacity.md`` and ``reports/capacity.png``. SPEC.md 10.4, W5-P2.

**Normalised units. No project inputs. No number is published.** Every input
SPEC.md 10.4's curve needs -- ``alpha_g``, ``tau``, the structure that fixes
``kappa`` -- comes from a trade series that does not exist until W6's
optimizer produces one, and inventing ``alpha_g`` is the one thing CLAUDE.md
most explicitly forbids. So the figure is drawn in the units the closed form is
scale-free in: AUM as a multiple of the patient regime's break-even AUM, net
alpha as a fraction of gross. In those units the curve is ``1 - sqrt(x)``, the
marked points are ``x = 1`` and ``x = 4/9``, and the only project values that
enter are the two calibrated ``Y`` regimes and the cost exponent -- which is
exactly what fixes the width of the sensitivity band. W6 redraws the same
figure in dollars from the trade series; this one shows the shape and the band
so that the machinery is demonstrated rather than described.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mafrm import config as config_mod
from mafrm.costs import capacity

__all__ = ["Normalised", "chart", "figure_path", "main", "normalised", "render", "report_path"]

#: Right edge of the normalised grid, in multiples of the patient break-even AUM.
#: Presentation only: 1.2 shows both curves crossing zero with margin.
_X_MAX = 1.2

#: The two regimes, in the order the report lists them. Both, always (SPEC.md 7.2.1).
_REGIMES = ("patient", "urgent")


def report_path() -> Path:
    return config_mod._REPO_ROOT / "reports" / "capacity.md"


def figure_path() -> Path:
    return config_mod._REPO_ROOT / "reports" / "capacity.png"


@dataclass(frozen=True)
class Normalised:
    """The normalised curve and every ratio the report states, so a test can pin them."""

    exponent: float
    y_patient: float
    y_urgent: float
    #: ``A_eff / A_BE`` at the configured exponent -- 4/9 at 3/2.
    effective_ratio: float
    #: ``A_BE(urgent) / A_BE(patient) = (Y_p/Y_u)^(1/(e-1))``: the sensitivity band.
    band_ratio: float
    #: Factor on capacity when ``kappa`` halves -- 4 at 3/2.
    halving_multiplier: float
    curve: capacity.CapacityCurve
    minimum_net_alpha: float


def normalised(cfg: config_mod.Config | None = None) -> Normalised:
    cfg = cfg or config_mod.load()
    costs = cfg.model.costs
    e = costs.total_cost_exponent
    y_p = costs.square_root_prefactor.patient
    y_u = costs.square_root_prefactor.urgent
    # kappa is linear in Y. Normalise the patient regime so that A_BE(patient)
    # = 1 with alpha_g = tau = 1; the urgent regime's kappa is then Y_u/Y_p.
    kappa = {"patient": 1.0, "urgent": y_u / y_p}
    grid = np.linspace(0.0, _X_MAX, costs.capacity.aum_grid.points)
    curve = capacity.capacity_curve(
        grid,
        gross_alpha=1.0,
        turnover=1.0,
        kappa_by_regime=kappa,
        exponent=e,
        minimum_net_alpha=costs.capacity.minimum_net_alpha,
    )
    return Normalised(
        exponent=e,
        y_patient=y_p,
        y_urgent=y_u,
        effective_ratio=capacity.effective_to_break_even_ratio(e),
        band_ratio=capacity.capacity_multiplier(y_u / y_p, exponent=e),
        halving_multiplier=capacity.capacity_multiplier(0.5, exponent=e),
        curve=curve,
        minimum_net_alpha=costs.capacity.minimum_net_alpha,
    )


def _marked(points: pd.DataFrame) -> dict[str, dict[str, float]]:
    """The per-regime marked points as plain floats, so the template and the chart read one thing."""
    return {
        str(regime): {str(column): float(value) for column, value in row.items()}
        for regime, row in points.to_dict(orient="index").items()
    }


def chart(norm: Normalised, path: Path) -> None:
    """``reports/capacity.png``: both regimes, the band between them, both points marked on each."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = norm.curve.aum
    net = norm.curve.net_alpha
    marked = _marked(norm.curve.points)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.fill_between(
        x,
        net["urgent"].to_numpy(),
        net["patient"].to_numpy(),
        color="0.85",
        label="cost-model sensitivity band (patient to urgent regime)",
    )
    styles = {"patient": ("tab:blue", "-"), "urgent": ("tab:red", "--")}
    for regime in _REGIMES:
        colour, dash = styles[regime]
        y_value = norm.y_patient if regime == "patient" else norm.y_urgent
        axis.plot(
            x,
            net[regime].to_numpy(),
            color=colour,
            linestyle=dash,
            label=f"{regime} regime, Y = {y_value:.2f}",
        )
        a_be = marked[regime]["break_even_aum"]
        a_eff = marked[regime]["effective_aum"]
        axis.plot([a_be], [0.0], marker="o", color=colour, markersize=8)
        axis.plot(
            [a_eff],
            [1.0 - a_eff ** (norm.exponent - 1.0) / a_be ** (norm.exponent - 1.0)],
            marker="s",
            color=colour,
            markersize=8,
        )
        axis.axvline(a_be, color=colour, linestyle=":", linewidth=0.8)
        axis.axvline(a_eff, color=colour, linestyle=":", linewidth=0.8)
        axis.annotate(
            f"A_BE ({regime}) = {a_be:.3f}",
            (a_be, 0.0),
            textcoords="offset points",
            xytext=(6, 8),
            color=colour,
            fontsize=8,
        )
        axis.annotate(
            f"A_eff = {norm.effective_ratio:.4f} x A_BE",
            (a_eff, 1.0 / 3.0),
            textcoords="offset points",
            xytext=(6, 8),
            color=colour,
            fontsize=8,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlim(0.0, _X_MAX)
    axis.set_ylim(-0.25, 1.05)
    axis.set_xlabel("AUM, as a multiple of the patient regime's break-even AUM")
    axis.set_ylabel("net alpha / gross alpha")
    axis.set_title(
        "SPEC.md 10.4 capacity curve, NORMALISED -- no project inputs (W5-P2)\n"
        "closed form = the rescaling SPEC.md 10.4 warns overstates cost; "
        "circles A_BE, squares A_eff = 4/9 A_BE",
        fontsize=10,
    )
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def render(cfg: config_mod.Config | None = None) -> str:
    cfg = cfg or config_mod.load()
    norm = normalised(cfg)
    rule = cfg.model.costs.capacity.aum_grid
    e = norm.exponent
    marked = _marked(norm.curve.points)
    lines: list[str] = []
    w = lines.append

    w("# Capacity (SPEC.md 10.4) -- the machinery, in normalised units")
    w("")
    w("Generated by `python -m mafrm.costs.capacity_report`. Figure: `reports/capacity.png`.")
    w("")
    w(
        "**NO PROJECT NUMBER IS PUBLISHED HERE.** Every input SPEC.md 10.4's curve needs -- the gross alpha"
    )
    w(
        "`alpha_g`, the annual two-way turnover `tau`, and the portfolio structure that fixes the impact"
    )
    w(
        "coefficient `kappa` -- comes from a trade series, and none exists until W6's optimizer produces"
    )
    w(
        "one (operator ruling, 2026-09-03, W5-P2; SPEC.md 10.4.1). Producing a number now would require"
    )
    w(
        "inventing `alpha_g`, which CLAUDE.md invariant 9 forbids. This report demonstrates the machinery"
    )
    w("in the units the closed form is scale-free in, and W6 redraws it in dollars.")
    w("")
    w("## 1. What the closed form is, and what it is not")
    w("")
    w(f"At `costs.total_cost_exponent` = {e:g}, with the portfolio structure fixed:")
    w("")
    w("```")
    w("TC(A)      = tau * kappa * A^(e-1)                 sqrt(A) at e = 3/2")
    w("alpha_n(A) = alpha_g - TC(A)")
    w("A_BE       = (alpha_g / (tau*kappa))^(1/(e-1))     break-even")
    w("A_eff      = ((alpha_g - alpha_min) / (e*tau*kappa))^(1/(e-1))")
    w(f"           = {norm.effective_ratio:.6f} * A_BE            at alpha_min = 0")
    w("```")
    w("")
    w("**This is the RESCALING SPEC.md 10.4 warns overstates cost badly.** It holds the small-AUM")
    w("portfolio's structure fixed and scales the impact function with AUM. The right capacity")
    w("experiment re-optimises at each candidate AUM with the ADV constraints binding and lets the")
    w("optimizer choose *different* holdings; that curve needs the optimizer and is registered for")
    w(
        "W6-P2/W6-P3 with a test that the holdings actually differ across AUM levels. What this closed"
    )
    w(
        "form gives is the upper bound on cost -- the lower bound on capacity -- the re-optimised curve"
    )
    w("is compared against.")
    w("")
    w(
        "**The L1 spread term is AUM-invariant and the spec's formula omits it.** `mafrm.costs.capacity`"
    )
    w(
        "carries it exactly as `aum_invariant_drag`, subtracted from `alpha_g` wherever `alpha_g` appears."
    )
    w(
        "Default zero reproduces the formulas above verbatim; W6 passes the structure's annual spread cost."
    )
    w("")
    w("## 2. The normalised curve and both marked points")
    w("")
    w(
        "Units: AUM as a multiple of the patient regime's break-even AUM; net alpha as a fraction of gross."
    )
    w(
        "`kappa` is linear in `Y`, so the patient regime is normalised to `kappa = 1` and the urgent regime's"
    )
    w(
        f"`kappa` is `Y_urgent/Y_patient` = {norm.y_urgent:.2f}/{norm.y_patient:.2f} = {norm.y_urgent / norm.y_patient:.4f}."
    )
    w("Both points are closed-form and read off no grid.")
    w("")
    w(
        "| Regime | `Y` | `kappa` (normalised) | `A_BE` | `A_eff` | `A_eff / A_BE` | net alpha at `A_eff` |"
    )
    w("|---|---|---|---|---|---|---|")
    for regime in _REGIMES:
        y_value = norm.y_patient if regime == "patient" else norm.y_urgent
        a_be = marked[regime]["break_even_aum"]
        a_eff = marked[regime]["effective_aum"]
        net_at_eff = 1.0 - (a_eff / a_be) ** (e - 1.0)
        w(
            f"| {regime} | {y_value:.2f} | {marked[regime]['kappa']:.4f} | **{a_be:.4f}** | **{a_eff:.4f}** | {a_eff / a_be:.6f} | {net_at_eff:.4f} |"
        )
    w("")
    w(
        f"`A_eff / A_BE` = `e^(-1/(e-1))` = **{norm.effective_ratio:.6f}** = 4/9 at 3/2 -- SPEC.md 10.4's hard-coded"
    )
    w(
        "relationship, pinned by `tests/test_capacity.py`. At the marginal point the net alpha is exactly one"
    )
    w("third of gross: `1 - sqrt(4/9) = 1/3`.")
    w("")
    w("## 3. The cost-model sensitivity band -- required beside every capacity number")
    w("")
    w(
        "You have no execution data, so any impact coefficient here is **borrowed rather than estimated**:"
    )
    w(
        "both `Y` regimes are backed out of published desk figures (`reports/cost_calibration.md` section"
    )
    w(
        "1), not fitted to this project's fills. A sensitivity band is therefore more honest and more"
    )
    w(
        "informative than a point estimate, and the band -- not a midpoint -- is what accompanies every"
    )
    w("capacity figure this project will publish.")
    w("")
    w(
        f"- Capacity is **quadratic in `kappa`** at `e` = {e:g}: multiplying `kappa` by `m` multiplies both `A_BE` and"
    )
    w(
        f"  `A_eff` by `m^(-1/(e-1))`. Halving the cost coefficient multiplies capacity by **{norm.halving_multiplier:.1f}x**."
    )
    w("  This is why AQR's estimates come out ~17x larger than Korajczyk-Sadka's (SPEC.md 10.4).")
    w(
        f"- The two calibrated regimes span `kappa` by {norm.y_urgent / norm.y_patient:.2f}x, so the band on capacity is"
    )
    w(
        f"  `(Y_patient/Y_urgent)^(1/(e-1))` = **{norm.band_ratio:.4f}**: the urgent regime's capacity is"
    )
    w(
        f"  {norm.band_ratio:.4f} of the patient regime's, i.e. patient capacity is **{1.0 / norm.band_ratio:.2f}x** urgent."
    )
    w(
        "- The band is evaluated at the two regimes and nowhere between them. SPEC.md 7.2.1 rules `Y` to be"
    )
    w(
        "  two calibrated regimes with no interior points -- they measure different things, and a sweep"
    )
    w(
        "  between them would manufacture a parameter from two numbers. The machinery accepts any `kappa`"
    )
    w("  (`capacity_multiplier`), so a sourced third regime slots in; none is invented.")
    w("")
    w("## 4. The rules that are written, and the inputs that are not")
    w("")
    w(
        f"- **`alpha_min`** (`costs.capacity.minimum_net_alpha` = {norm.minimum_net_alpha:g}): parameterised; only the spec's"
    )
    w(
        "  `alpha_min = 0` special case is implemented, and both the config parser and the module refuse any"
    )
    w("  other value until one is sourced.")
    w(
        "- **`alpha_g`** (`costs.capacity.gross_alpha`, null): the highest-stakes input in the project and"
    )
    w(
        "  published nowhere. It enters SPEC.md 1's risk term and the break-even condition. **W6-P1 must not"
    )
    w(
        "  pick it.** Ruled in advance: swept and reported as a band, with the principle for the sweep range"
    )
    w("  written before W6-P1 opens. The parser refuses a point value.")
    w(
        f"- **The AUM grid is a rule**: `{rule.spacing}`-spaced between the AUM at which the largest position's trade is"
    )
    w(
        f"  {rule.low_participation_of_adv:.1%} of its proxy's ADV and the AUM at which it is {rule.high_participation_of_adv:.0%}"
    )
    w(
        f"  (`aum_grid_bounds`). Both ends derive from the ADV data once a portfolio exists; {rule.points} points is"
    )
    w("  display resolution and reaches no number.")
    w(
        "- **`kappa` from a structure** (`structure_kappa`): `sum_i Y*sigma_i*|z_i|^e / V_i^(e-1)` over"
    )
    w(
        "  `sum_i |z_i|`, pinned against the SPEC.md 7.1 cost engine across a grid of AUMs rather than trusted."
    )
    w("")
    w("## 5. Handoff to W6")
    w("")
    w("Registered in `experiments.md` (W5-P2 forward registrations) and SPEC.md 10.4.1:")
    w("")
    w(
        "1. **W6-P1**: `alpha_g` swept, never a point; the sweep principle written before the session opens."
    )
    w(
        "2. **W6-P2/W6-P3**: the re-optimised capacity curve -- portfolio construction re-run at each grid AUM"
    )
    w(
        "   with ADV participation constraints binding -- with a test that the holdings differ across AUM, and"
    )
    w(
        "   this closed form drawn beside it as the rescaling comparand. Both `Y` regimes, both points marked,"
    )
    w(
        "   the band shown, in dollars, with the 30y-through-TLT and 2026-disclosure caveats carried from"
    )
    w("   `reports/cost_calibration.md`.")
    w(
        "3. The three SPEC.md 7.3 scaling laws checked empirically on the backtest's own cost output, charted."
    )
    w("")
    return "\n".join(lines) + "\n"


def main() -> int:
    cfg = config_mod.load()
    path = report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(cfg), encoding="utf-8")
    chart(normalised(cfg), figure_path())
    print(f"wrote {path} and {figure_path()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
