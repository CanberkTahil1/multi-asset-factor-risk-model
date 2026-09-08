"""SPEC.md 10.4's capacity RE-OPTIMISATION -- the real deliverable W5-P2 published no number for. W6-P3.

SPEC.md 10.4: *"The right capacity experiment is a re-optimization, not a
rescaling. Re-run portfolio construction at each candidate AUM with the ADV
constraints binding and let the optimizer choose different holdings. Simply
scaling the small-AUM portfolio and applying an impact function overstates
costs badly."* This module is that experiment, on the reference cell of
SPEC.md 9's grid (variant 4, treatment D), at every AUM on the RULED grid
(``costs.capacity.aum_grid``: log-spaced from the AUM at which the equal-weight
trade in the thinnest proxy is 0.1% of its median ADV to the AUM at which it is
10%), with the ADV participation cap imposed HARD (SPEC.md 10.4's "with the ADV
constraints binding") and the relaxation ladder dropping the cap before the TE
bound so that any relaxation is logged as the cap's. Both ``Y`` regimes, as
SPEC.md 7.2.1 rules and SPEC.md 10.4 requires ("any capacity number you publish
must be accompanied by its cost-model sensitivity"; capacity is quadratic in
``kappa``).

OPERATOR RULING 4 (2026-09-04, SPEC.md 9.3): the curve is ``data-diagnostic``
on ONE condition -- it reports ``TC(A)``, turnover and net-return drag per AUM
point and NO SHARPE per point. The Sharpe-bearing points on the curve are
``A_low`` and ``A_high``, already ``strategy-config`` rows in the book-size band
(:mod:`mafrm.backtest.bands`). Reversing result, named in advance: if any AUM
point acquires a Sharpe, every point becomes ``strategy-config`` and ``N`` goes
to ~150. :func:`metrics_section` asserts that no key it writes names a Sharpe,
and ``config`` refuses ``report_sharpe: true``.

THE COMPARAND. The closed form W5-P2 built (:mod:`mafrm.costs.capacity`) is
the RESCALING SPEC.md 10.4 warns about: it holds the small-AUM structure fixed
and scales the impact function. It is drawn beside the re-optimised curve from
the reference cell's OWN trade structure -- ``kappa`` and ``tau`` from its
realised trades, spreads, volatilities and ADVs through
:func:`mafrm.costs.capacity.structure_kappa` -- and its MEASURED gross alpha
(SPEC.md 8.5.1 ruling 2: what the RSTR signal did in sample, not what it will
do), labelled as the bound that overstates cost. ``alpha_min`` stays 0.

REGISTERED before the run (experiments.md, W6-P3): (a) the holdings differ
across AUM levels -- asserted in code, as SPEC.md 10.4.1 requires; (b) the
re-optimised ``TC(A)`` is at or below the rescaled ``TC(A)`` at every AUM at or
above the reference NAV, in both regimes; (c) the re-optimised break-even AUM
is at or beyond the closed form's ``A_BE`` in both regimes; (d) the hard ADV
cap binds on at least one rebalance at the top of the grid.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mafrm import config as config_mod
from mafrm.backtest import grid as grid_mod
from mafrm.backtest import verification
from mafrm.costs import capacity

__all__ = [
    "CapacityError",
    "CapacityResult",
    "CurvePoint",
    "aum_points",
    "main",
    "metrics_section",
    "registered_checks",
    "render",
    "rescaling_comparand",
    "run_curve",
    "run_point",
]

FloatArray = NDArray[np.float64]

_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_REPORTS: Final[Path] = _ROOT / "reports"
_RESULTS: Final[Path] = _ROOT / "results"
_MARKDOWN_PATH: Final[Path] = _REPORTS / "capacity_reoptimised.md"
_CSV_PATH: Final[Path] = _REPORTS / "capacity_reoptimised.csv"
_CHART_PATH: Final[Path] = _REPORTS / "capacity_reoptimised.png"
_METRICS_PATH: Final[Path] = _RESULTS / "metrics.json"
_ADV_LIMIT: Final[str] = "adv_participation"
#: Two AUM points whose target weights differ by less than this on EVERY
#: rebalance are the same book (the optimizer's feasibility tolerance is 1e-8;
#: this is a weight distance, not a solver tolerance).
_SAME_BOOK_L1: Final[float] = 1e-6


class CapacityError(ValueError):
    """The curve cannot be built as asked. Nothing is filled or defaulted."""


@dataclass(frozen=True)
class CurvePoint:
    """One AUM in one regime: what the re-optimised book cost and did. NO Sharpe."""

    aum: float
    regime: str
    cost_drag_annual: float
    spread_drag_annual: float
    impact_drag_annual: float
    turnover_annual: float
    gross_annual: float
    net_annual: float
    adv_binding_share: float
    relaxed: int
    fell_back: int
    effective_assets_mean: float
    largest_weight_median: float
    inaccurate: int
    resolve_max_l1: float
    targets: pd.DataFrame

    @property
    def impact_share(self) -> float:
        return self.impact_drag_annual / self.cost_drag_annual if self.cost_drag_annual > 0 else 0.0


@dataclass(frozen=True)
class CapacityResult:
    aum: FloatArray
    bounds: tuple[float, float]
    points: tuple[CurvePoint, ...]
    reference: grid_mod.CellResult
    #: Regime -> the reference cell's fixed structure under that regime's ``Y``.
    structures: dict[str, capacity.FixedStructure]
    closed_form: capacity.CapacityCurve
    gross_alpha: float
    checks: tuple[dict[str, object], ...] = ()

    def regime(self, regime: str) -> tuple[CurvePoint, ...]:
        return tuple(p for p in self.points if p.regime == regime)


def aum_points(
    universe: verification.Universe, settings: config_mod.Config
) -> tuple[tuple[float, float], FloatArray]:
    """The ruled grid in dollars: ``A = p N min_i median(V_i)`` at the rule's two participations."""
    rule = settings.model.costs.capacity.aum_grid
    n = len(universe.assets)
    thinnest = float(universe.adv_median.min())
    bounds = capacity.aum_grid_bounds(
        largest_trade_fraction_of_nav=1.0 / n, proxy_adv_dollars=thinnest, rule=rule
    )
    return bounds, capacity.aum_grid(bounds, rule=rule)


def run_point(
    inputs: grid_mod.CellInputs, settings: config_mod.Config, *, aum: float, progress: bool = False
) -> CurvePoint:
    """Re-optimise one cell at ``aum`` with the ADV cap HARD; return the point, never a Sharpe."""
    cap = settings.model.optimizer.grid.capacity
    if cap.report_sharpe:
        raise CapacityError("report_sharpe must be false (operator ruling 4, W6-P3)")
    cell = dataclasses.replace(inputs.cell, tag=f"aum_{aum:.0f}")
    point_inputs = dataclasses.replace(
        inputs,
        cell=cell,
        nav=float(aum),
        adv_participation_hard=True,
        relaxation_ladder=tuple(cap.relaxation_ladder),
    )
    result = grid_mod.run_cell(point_inputs, settings, progress=progress)
    s = result.summary
    ident = result.identity
    drag = recurring_drag(result)
    return CurvePoint(
        aum=float(aum),
        regime=cell.regime,
        cost_drag_annual=drag["total"],
        spread_drag_annual=drag["spread"],
        impact_drag_annual=drag["impact"],
        turnover_annual=drag["turnover"],
        gross_annual=ident.gross_return,
        net_annual=ident.gross_return - drag["total"],
        adv_binding_share=float(s.get(f"{_ADV_LIMIT}_binding_share", 0.0)),
        relaxed=int(s["relaxed"]),
        fell_back=int(s["fell_back"]),
        effective_assets_mean=s["effective_assets_mean"],
        largest_weight_median=s["largest_weight_median"],
        inaccurate=int(s["inaccurate"]),
        resolve_max_l1=s["resolve_max_l1"],
        targets=result.targets,
    )


def recurring_drag(result: grid_mod.CellResult) -> dict[str, float]:
    """The RECURRING annual drag: per-rebalance cost over ``NAV_pre``, the build from cash excluded.

    The first rebalance moves the whole book out of cash and is a one-off, not
    a drag; SPEC.md 10.4's ``TC(A)`` is the recurring cost of running the
    strategy, so the curve and its closed-form comparand both average over the
    rebalances after it, on the same ``NAV_pre`` normalisation
    :func:`mafrm.costs.capacity.structure_kappa` uses. At the reference NAV the
    closed form on the reference cell's own structure then reproduces this
    number exactly, which a test pins.
    """
    bt = result.backtest
    ppy = result.inputs.periods_per_year
    nav = bt.nav_before_costs.to_numpy(dtype=float)[1:]
    if nav.size == 0:
        raise CapacityError("recurring_drag: a curve point needs at least two rebalances")
    spread = bt.spread_cost.to_numpy(dtype=float)[1:].sum(axis=1) / nav
    impact = bt.impact_cost.to_numpy(dtype=float)[1:].sum(axis=1) / nav
    asym = bt.asymmetry_cost.to_numpy(dtype=float)[1:].sum(axis=1) / nav
    turnover = bt.trades.to_numpy(dtype=float)[1:]
    return {
        "spread": float(spread.mean() * ppy),
        "impact": float(impact.mean() * ppy),
        "asymmetry": float(asym.mean() * ppy),
        "total": float((spread + impact + asym).mean() * ppy),
        "turnover": float(np.abs(turnover).sum(axis=1).mean() * ppy),
    }


def rescaling_comparand(
    reference: grid_mod.CellResult, settings: config_mod.Config, aum: FloatArray
) -> tuple[dict[str, capacity.FixedStructure], capacity.CapacityCurve]:
    """W5-P2's closed form on the reference cell's own structure and measured gross alpha."""
    costs = settings.model.costs
    cap = settings.model.optimizer.grid.capacity
    # The recurring structure: every rebalance after the build from cash.
    dates = reference.inputs.rebalance_dates[1:]
    assets = list(reference.inputs.assets)
    model = reference.inputs.realised_cost
    trades = reference.backtest.trades.loc[dates, assets].to_numpy(dtype=float)
    sigma = model.daily_volatility.loc[dates, assets].to_numpy(dtype=float)
    adv = model.adv_dollars.loc[dates, assets].to_numpy(dtype=float)
    half = model.half_spread.loc[dates, assets].to_numpy(dtype=float)
    structures = {
        regime: capacity.structure_kappa(
            trades,
            daily_volatility=sigma,
            adv_dollars=adv,
            prefactor=float(getattr(costs.square_root_prefactor, regime)),
            exponent=costs.total_cost_exponent,
            rebalances_per_year=reference.inputs.periods_per_year,
            half_spread=half,
        )
        for regime in cap.regimes
    }
    first = structures[cap.regimes[0]]
    curve = capacity.capacity_curve(
        aum,
        gross_alpha=reference.identity.gross_return,
        turnover=first.turnover,
        kappa_by_regime={regime: fs.kappa for regime, fs in structures.items()},
        exponent=costs.total_cost_exponent,
        minimum_net_alpha=costs.capacity.minimum_net_alpha,
        aum_invariant_drag=first.aum_invariant_drag,
    )
    return structures, curve


def run_curve(
    universe: verification.Universe,
    settings: config_mod.Config,
    *,
    aum: FloatArray | None = None,
    progress: bool = True,
) -> CapacityResult:
    """The reference cell at every AUM in both regimes, and the closed form beside it."""
    cap = settings.model.optimizer.grid.capacity
    bands = settings.model.optimizer.grid.bands
    bounds, grid_aum = aum_points(universe, settings)
    aum_values = np.asarray(aum, dtype=float) if aum is not None else grid_aum
    reference_cell = grid_mod.CellSpec(
        bands.reference_variant, bands.reference_treatment, bands.reference_regime
    )
    if progress:
        print(f"solving the reference cell {reference_cell.label} at the reference NAV", flush=True)
    reference = grid_mod.run_cell(
        grid_mod.cell_inputs(universe, settings, reference_cell), settings, progress=progress
    )
    points: list[CurvePoint] = []
    for regime in cap.regimes:
        base = grid_mod.cell_inputs(
            universe,
            settings,
            dataclasses.replace(reference_cell, regime=regime),
            adv_participation_hard=True,
            relaxation_ladder=tuple(cap.relaxation_ladder),
        )
        for i, value in enumerate(aum_values):
            if progress:
                print(
                    f"  capacity {regime} {i + 1}/{len(aum_values)}: A = ${value:,.0f}", flush=True
                )
            points.append(run_point(base, settings, aum=float(value)))
    structures, closed = rescaling_comparand(reference, settings, aum_values)
    out = CapacityResult(
        aum=aum_values,
        bounds=bounds,
        points=tuple(points),
        reference=reference,
        structures=structures,
        closed_form=closed,
        gross_alpha=reference.identity.gross_return,
    )
    return dataclasses.replace(out, checks=registered_checks(out, settings))


# ---------------------------------------------------------------------------
# Derived points and the registered legs
# ---------------------------------------------------------------------------


def _closed_form_point(out: CapacityResult, regime: str, column: str) -> float:
    """One marked point of the closed form, as a float."""
    row = out.closed_form.points.to_dict(orient="index")[regime]
    return float(row[column])


def holdings_distance(a: CurvePoint, b: CurvePoint) -> float:
    """The largest per-rebalance L1 distance between two points' target weights."""
    if not a.targets.index.equals(b.targets.index):
        raise CapacityError("holdings_distance: the two points have different rebalance dates")
    return float(np.abs(a.targets.to_numpy() - b.targets.to_numpy()).sum(axis=1).max())


def break_even_interpolated(points: Sequence[CurvePoint]) -> float:
    """The AUM at which the re-optimised net return crosses zero, linear in log AUM; ``inf`` if never."""
    aum = np.array([p.aum for p in points])
    net = np.array([p.net_annual for p in points])
    order = np.argsort(aum)
    aum, net = aum[order], net[order]
    below = np.flatnonzero(net <= 0.0)
    if below.size == 0:
        return float("inf")
    j = int(below[0])
    if j == 0:
        return float(aum[0])
    x0, x1 = np.log(aum[j - 1]), np.log(aum[j])
    y0, y1 = net[j - 1], net[j]
    return float(np.exp(x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)))


def effective_argmax(points: Sequence[CurvePoint]) -> float:
    """The grid AUM maximising DOLLAR net alpha ``A alpha_n(A)`` -- the marginal condition at ``alpha_min = 0``."""
    aum = np.array([p.aum for p in points])
    dollars = aum * np.array([p.net_annual for p in points])
    return float(aum[int(np.argmax(dollars))])


def registered_checks(
    out: CapacityResult, settings: config_mod.Config
) -> tuple[dict[str, object], ...]:
    cap = settings.model.optimizer.grid.capacity
    checks: list[dict[str, object]] = []

    def add(leg: str, claim: str, holds: bool, observed: str) -> None:
        checks.append({"leg": leg, "claim": claim, "holds": bool(holds), "observed": observed})

    reference_nav = out.reference.inputs.nav
    for regime in cap.regimes:
        pts = out.regime(regime)
        if len(pts) < 2:
            continue
        # (a) holdings differ across AUM
        distance = holdings_distance(pts[0], pts[-1])
        add(
            f"a/{regime}",
            "the re-optimised holdings differ across AUM levels (SPEC.md 10.4.1)",
            distance > _SAME_BOOK_L1,
            f"largest per-rebalance L1 distance between A = ${pts[0].aum:,.0f} and "
            f"${pts[-1].aum:,.0f}: {distance:.4f}",
        )
        # (b) re-optimised TC(A) <= rescaled TC(A) at or above the reference NAV
        fs = out.structures[regime]
        rescaled = (
            fs.cost_drag(
                np.array([p.aum for p in pts]), exponent=settings.model.costs.total_cost_exponent
            )
            + fs.aum_invariant_drag
        )
        above = [(p, r) for p, r in zip(pts, rescaled, strict=True) if p.aum >= reference_nav]
        fails = [p.aum for p, r in above if p.cost_drag_annual > r * (1.0 + 1e-9)]
        worst = max(((p.cost_drag_annual - r) * 1e4 for p, r in above), default=float("nan"))
        add(
            f"b/{regime}",
            "the re-optimised TC(A) is at or below the rescaled TC(A) at every AUM at or above the reference NAV",
            not fails,
            f"{len(fails)} of {len(above)} points above; largest excess {worst:+.2f} bp/yr",
        )
        # (c) re-optimised break-even >= closed-form A_BE
        be_closed = _closed_form_point(out, regime, "break_even_aum")
        be_re = break_even_interpolated(pts)
        add(
            f"c/{regime}",
            "the re-optimised break-even AUM is at or beyond the closed form's A_BE",
            be_re >= be_closed,
            f"re-optimised {'beyond the grid' if not np.isfinite(be_re) else f'${be_re:,.0f}'} "
            f"vs closed form ${be_closed:,.0f} (grid top ${pts[-1].aum:,.0f})",
        )
        # (d) the hard cap binds at the top of the grid
        add(
            f"d/{regime}",
            "the hard ADV cap binds on at least one rebalance at the top of the grid",
            pts[-1].adv_binding_share > 0.0,
            f"binding share at A = ${pts[-1].aum:,.0f}: {pts[-1].adv_binding_share:.1%}; "
            f"relaxed on {pts[-1].relaxed} rebalances",
        )
    return tuple(checks)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _dollars(value: float) -> str:
    if not np.isfinite(value):
        return "beyond the grid"
    return f"${value / 1e6:,.2f}M" if value >= 1e6 else f"${value:,.0f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def render(out: CapacityResult, settings: config_mod.Config) -> str:
    """``reports/capacity_reoptimised.md``."""
    costs = settings.model.costs
    cap = settings.model.optimizer.grid.capacity
    e = costs.total_cost_exponent
    ref = out.reference
    lines = [
        "# SPEC.md 10.4's capacity curve, RE-OPTIMISED -- replacing W5-P2's closed-form placeholder",
        "",
        "Generated by `python -m mafrm.backtest.capacity_curve`. W6-P3, operator ruling 4 (SPEC.md 9.3): `data-diagnostic` -- the curve reports `TC(A)`, turnover and net-return drag per AUM point and NO Sharpe per point; the Sharpe-bearing points on it are `A_low` and `A_high`, already `strategy-config` in the book-size band (`reports/bands.md`).",
        "",
        "## What was run",
        "",
        f"The reference cell (variant 4, treatment D) re-optimised at each of {len(out.aum)} AUMs on the RULED grid -- log-spaced from {_dollars(out.bounds[0])} (the equal-weight trade in the thinnest proxy at {costs.capacity.aum_grid.low_participation_of_adv:.1%} of its median ADV) to {_dollars(out.bounds[1])} ({costs.capacity.aum_grid.high_participation_of_adv:.0%}) -- with the {settings.model.optimizer.constraints.adv_participation.bound:.0%} ADV participation cap imposed HARD and the ladder dropping it before the TE bound, in both `Y` regimes ({', '.join(f'{r} {getattr(costs.square_root_prefactor, r):.2f}' for r in cap.regimes)}). The optimizer chooses DIFFERENT holdings at each AUM; leg (a) below asserts it.",
        "",
        f"**The comparand is the rescaling.** W5-P2's closed form `TC(A) = tau kappa A^(e-1)` at `e = {e:g}` on the reference cell's OWN trade structure -- `tau`, `kappa` and the AUM-invariant spread drag from its {int(ref.summary['rebalances'])} realised rebalances through `mafrm.costs.capacity.structure_kappa` -- and its MEASURED gross alpha, {out.gross_alpha:.2%} a year (SPEC.md 8.5.1 ruling 2: what the RSTR signal did in sample, not what it will do). SPEC.md 10.4: the rescaling holds the small-AUM structure fixed and *overstates costs badly*; it is the upper bound on cost the re-optimised curve is drawn beside. `alpha_min = 0` (only that case is implemented, W5-P2).",
        "",
        "## The marked points, both regimes (the cost-model sensitivity SPEC.md 10.4 requires beside every capacity number)",
        "",
    ]
    rows = []
    for regime in cap.regimes:
        fs = out.structures[regime]
        pts = out.regime(regime)
        be_closed = _closed_form_point(out, regime, "break_even_aum")
        eff_closed = _closed_form_point(out, regime, "effective_aum")
        rows.append(
            [
                regime,
                f"{getattr(costs.square_root_prefactor, regime):.2f}",
                f"{fs.turnover:.2f}",
                f"{fs.kappa:.3e}",
                f"{fs.aum_invariant_drag * 1e4:.1f}",
                _dollars(be_closed),
                _dollars(eff_closed),
                _dollars(break_even_interpolated(pts)),
                _dollars(effective_argmax(pts)),
                f"{pts[-1].adv_binding_share:.1%}",
            ]
        )
    lines += _table(
        [
            "regime",
            "Y",
            "tau (turnover /yr)",
            "kappa ($^-(e-1))",
            "spread drag bp/yr (AUM-invariant)",
            "closed-form A_BE",
            "closed-form A_eff = 4/9 A_BE",
            "re-optimised break-even (interpolated on the grid)",
            "re-optimised argmax of dollar net alpha (grid point)",
            "ADV cap binding share at the grid top",
        ],
        rows,
    )
    lines += [
        "",
        f"Closed form: patient capacity is `(Y_u/Y_p)^(1/(e-1))` = {capacity.capacity_multiplier(costs.square_root_prefactor.urgent / costs.square_root_prefactor.patient, exponent=e):.4f}^-1 = {1 / capacity.capacity_multiplier(costs.square_root_prefactor.urgent / costs.square_root_prefactor.patient, exponent=e):.2f}x urgent; halving `kappa` multiplies capacity by {capacity.capacity_multiplier(0.5, exponent=e):.0f}. The re-optimised break-even is read off the grid by linear interpolation in log AUM and is `beyond the grid` where the net return never crosses zero inside it. With the 30y-through-TLT and 2026-disclosure caveats of SPEC.md 7.1.1-7.1.2 carried beside every figure: five of thirteen assets are curve points traded through maturity-matched ETF proxies, and every spread level is a 2026 disclosure applied to the whole history.",
        "",
        "## Registered legs, scored",
        "",
    ]
    lines += _table(
        ["leg", "registered", "verdict", "observed"],
        [
            [
                str(c["leg"]),
                str(c["claim"]),
                "HOLDS" if c["holds"] else "**REFUTED**",
                str(c["observed"]),
            ]
            for c in out.checks
        ],
    )
    lines += ["", "## The curve, per AUM point (no Sharpe)", ""]
    header = [
        "regime",
        "AUM",
        "TC(A) re-optimised bp/yr",
        "TC(A) rescaled bp/yr",
        "impact share",
        "turnover /yr",
        "gross %/yr",
        "net %/yr",
        "dollar net alpha $/yr",
        "ADV cap binding",
        "relaxed",
        "eff. assets",
        "largest w median",
        "inaccurate / max L1",
    ]
    rows = []
    for regime in cap.regimes:
        fs = out.structures[regime]
        for p in out.regime(regime):
            rescaled = float(fs.cost_drag(np.array([p.aum]), exponent=e)[0] + fs.aum_invariant_drag)
            rows.append(
                [
                    regime,
                    _dollars(p.aum),
                    f"{p.cost_drag_annual * 1e4:.1f}",
                    f"{rescaled * 1e4:.1f}",
                    f"{p.impact_share:.0%}",
                    f"{p.turnover_annual:.2f}",
                    f"{p.gross_annual * 100:.2f}",
                    f"{p.net_annual * 100:.2f}",
                    f"{p.aum * p.net_annual:,.0f}",
                    f"{p.adv_binding_share:.1%}",
                    str(p.relaxed),
                    f"{p.effective_assets_mean:.2f}",
                    f"{p.largest_weight_median:.1%}",
                    f"{p.inaccurate} / {p.resolve_max_l1:.1e}",
                ]
            )
    lines += _table(header, rows)
    lines += [
        "",
        "## Reading notes",
        "",
        "- Every point is the full 187-rebalance backtest re-solved at that AUM; the gross return moves with AUM because the cap and the impact term change the book, which is the whole point of re-optimising rather than rescaling.",
        "- `TC(A)` is the RECURRING drag: per-rebalance cost over `NAV_pre`, annualised, with the first rebalance (the build from cash, a one-off) excluded -- the same normalisation the closed form's structure uses, so at the reference NAV the rescaled figure reproduces the reference cell's own recurring cost exactly (a test pins it). The answer table's `cost bp/yr` is the identity's `TC`, on the holding periods; the two differ by the normalisation and by that one trade.",
        "- `TC(A) rescaled` is the closed form on the reference cell's structure at that AUM: spread drag (AUM-invariant) plus `tau kappa A^(e-1)`.",
        "- The dollar net alpha `A alpha_n(A)` is what a manager maximises; its grid argmax is the re-optimised analogue of SPEC.md 10.4's `A_eff` and is read off the grid, so it is display-resolution coarse. The closed form's marked points are exact.",
        "- No Sharpe appears for any AUM point, by ruling; `results/metrics.json`'s `capacity_reoptimised` section is asserted in code to carry no key naming one.",
        "",
    ]
    return "\n".join(lines)


def chart(out: CapacityResult, settings: config_mod.Config, path: Path) -> None:
    """Net return and cost drag against AUM, re-optimised (solid) beside the rescaling (dashed)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    costs = settings.model.costs
    cap = settings.model.optimizer.grid.capacity
    e = costs.total_cost_exponent
    surface, ink, ink_2, gridline = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
    colours = {"patient": "#2a78d6", "urgent": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor(surface)
    for regime in cap.regimes:
        pts = out.regime(regime)
        aum = np.array([p.aum for p in pts]) / 1e6
        colour = colours.get(regime, ink_2)
        fs = out.structures[regime]
        rescaled_tc = fs.cost_drag(aum * 1e6, exponent=e) + fs.aum_invariant_drag
        axes[0].plot(
            aum, [p.net_annual * 100 for p in pts], color=colour, label=f"{regime}, re-optimised"
        )
        axes[0].plot(
            aum,
            (out.gross_alpha - rescaled_tc) * 100,
            color=colour,
            linestyle="--",
            label=f"{regime}, rescaled (closed form)",
        )
        be = _closed_form_point(out, regime, "break_even_aum") / 1e6
        eff = _closed_form_point(out, regime, "effective_aum") / 1e6
        for x, marker in ((be, "o"), (eff, "s")):
            if aum.min() <= x <= aum.max():
                axes[0].plot(
                    [x],
                    [0.0 if marker == "o" else out.gross_alpha * 100 / 3],
                    marker=marker,
                    color=colour,
                    markersize=7,
                )
        axes[1].plot(
            aum,
            [p.cost_drag_annual * 1e4 for p in pts],
            color=colour,
            label=f"{regime}, re-optimised",
        )
        axes[1].plot(
            aum, rescaled_tc * 1e4, color=colour, linestyle="--", label=f"{regime}, rescaled"
        )
    axes[0].axhline(0.0, color=ink_2, linewidth=1.0)
    axes[0].axvline(out.reference.inputs.nav / 1e6, color=ink_2, linewidth=0.8, linestyle=":")
    axes[0].set_ylabel("net return, % per year", color=ink_2)
    axes[0].set_title(
        "net alpha against AUM: re-optimised vs rescaled", color=ink, fontsize=10, loc="left"
    )
    axes[1].set_ylabel("cost drag TC(A), bp per year", color=ink_2)
    axes[1].set_title("cost drag against AUM", color=ink, fontsize=10, loc="left")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("AUM, $M (log)", color=ink_2)
        ax.set_facecolor(surface)
        ax.grid(color=gridline, linewidth=1.0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=ink_2, labelsize=9)
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "SPEC.md 10.4's capacity curve, RE-OPTIMISED with the ADV cap binding, both Y regimes (W6-P3)",
        color=ink,
        fontsize=12,
        x=0.02,
        y=0.985,
        ha="left",
    )
    fig.text(
        0.02,
        0.94,
        f"Reference cell 4D; measured gross alpha {out.gross_alpha:.2%}/yr; dotted line the reference NAV; circles closed-form A_BE, squares A_eff = 4/9 A_BE. No Sharpe per point (ruling 4).",
        color=ink_2,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)


def metrics_section(out: CapacityResult, settings: config_mod.Config) -> dict[str, object]:
    """The ``capacity_reoptimised`` section of ``results/metrics.json``; carries no Sharpe."""
    cap = settings.model.optimizer.grid.capacity
    costs = settings.model.costs
    e = costs.total_cost_exponent

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

    regimes: dict[str, object] = {}
    for regime in cap.regimes:
        fs = out.structures[regime]
        pts = out.regime(regime)
        regimes[regime] = {
            "prefactor_Y": float(getattr(costs.square_root_prefactor, regime)),
            "structure": {
                "turnover_annual": fs.turnover,
                "kappa": fs.kappa,
                "spread_drag_annual": fs.aum_invariant_drag,
            },
            "closed_form": {
                "break_even_aum": _closed_form_point(out, regime, "break_even_aum"),
                "effective_aum": _closed_form_point(out, regime, "effective_aum"),
            },
            "reoptimised": {
                "break_even_aum_interpolated": break_even_interpolated(pts),
                "dollar_net_alpha_argmax_aum": effective_argmax(pts),
            },
            "points": [
                {
                    "aum": p.aum,
                    "cost_drag_annual": p.cost_drag_annual,
                    "cost_drag_rescaled_annual": float(
                        fs.cost_drag(np.array([p.aum]), exponent=e)[0] + fs.aum_invariant_drag
                    ),
                    "impact_share": p.impact_share,
                    "turnover_annual": p.turnover_annual,
                    "gross_annual": p.gross_annual,
                    "net_annual": p.net_annual,
                    "adv_binding_share": p.adv_binding_share,
                    "relaxed": p.relaxed,
                    "fell_back": p.fell_back,
                    "effective_assets_mean": p.effective_assets_mean,
                    "inaccurate": p.inaccurate,
                    "resolve_max_l1": p.resolve_max_l1,
                }
                for p in pts
            ],
        }
    payload: dict[str, object] = {
        "task": "W6-P3",
        "spec": "SPEC.md 10.4 (re-optimisation), 10.4.1",
        "category": "data-diagnostic",
        "report_sharpe": cap.report_sharpe,
        "gross_alpha_measured_annual": out.gross_alpha,
        "reference_nav_dollars": out.reference.inputs.nav,
        "aum_grid_bounds": list(out.bounds),
        "aum_points": len(out.aum),
        "regimes": regimes,
        "registered_checks": list(out.checks),
    }
    _assert_no_sharpe(payload)
    return clean(payload)  # type: ignore[return-value]


def _assert_no_sharpe(payload: object, path: str = "") -> None:
    """Operator ruling 4: a Sharpe per AUM point would make every point a trial."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            # The one permitted mention is the switch that says there is none.
            if "sharpe" in str(key).lower() and str(key) != "report_sharpe":
                raise CapacityError(f"capacity section carries a Sharpe at {path}/{key}")
            _assert_no_sharpe(value, f"{path}/{key}")
    elif isinstance(payload, list | tuple):
        for i, value in enumerate(payload):
            _assert_no_sharpe(value, f"{path}[{i}]")


def write_metrics_section(section: dict[str, object], path: Path = _METRICS_PATH) -> None:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload["capacity_reoptimised"] = section
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - entry point
    parser = argparse.ArgumentParser(description="SPEC.md 10.4's re-optimised capacity curve")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    settings = config_mod.load()
    universe = verification.build_universe(settings)
    out = run_curve(universe, settings, progress=not args.quiet)
    _RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in out.points:
        rows.append(
            {
                "regime": p.regime,
                "aum": p.aum,
                "cost_drag_annual": p.cost_drag_annual,
                "spread_drag_annual": p.spread_drag_annual,
                "impact_drag_annual": p.impact_drag_annual,
                "turnover_annual": p.turnover_annual,
                "gross_annual": p.gross_annual,
                "net_annual": p.net_annual,
                "adv_binding_share": p.adv_binding_share,
                "relaxed": p.relaxed,
                "effective_assets_mean": p.effective_assets_mean,
                "largest_weight_median": p.largest_weight_median,
                "inaccurate": p.inaccurate,
                "resolve_max_l1": p.resolve_max_l1,
            }
        )
    pd.DataFrame(rows).to_csv(_CSV_PATH, index=False, float_format="%.10g")
    _MARKDOWN_PATH.write_text(render(out, settings), encoding="utf-8")
    write_metrics_section(metrics_section(out, settings))
    chart(out, settings, _CHART_PATH)
    for c in out.checks:
        print(f"  leg {c['leg']}: {'HOLDS' if c['holds'] else 'REFUTED'} -- {c['observed']}")
    for regime in settings.model.optimizer.grid.capacity.regimes:
        pts = out.regime(regime)
        print(
            f"  {regime}: closed-form A_BE ${_closed_form_point(out, regime, 'break_even_aum'):,.0f}, "
            f"re-optimised break-even {break_even_interpolated(pts):,.0f}, "
            f"dollar-net-alpha argmax ${effective_argmax(pts):,.0f}"
        )
    print(f"wrote {_MARKDOWN_PATH}, {_CSV_PATH}, {_CHART_PATH}, {_METRICS_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
