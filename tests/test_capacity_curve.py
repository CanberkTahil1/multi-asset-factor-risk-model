"""SPEC.md 10.4's re-optimisation: holdings differ across AUM, no Sharpe per point. W6-P3."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mafrm import config
from mafrm.backtest import capacity_curve as cc
from mafrm.backtest import grid
from mafrm.costs import capacity, impact

CFG = config.load()
CAP = CFG.model.optimizer.grid.capacity


def _inputs(regime: str = "patient", seed: int = 5) -> grid.CellInputs:
    return grid.synthetic_inputs(
        CFG, cell=grid.CellSpec("eigenfactor_a1.0", "time_varying", regime), seed=seed
    )


@pytest.fixture(scope="module")
def two_points() -> tuple[cc.CurvePoint, cc.CurvePoint]:
    base = _inputs()
    # The synthetic ADV is $10M per asset; at $1M the 6% cap is 60% of NAV per trade
    # and rarely binds, at $50M it is 1.2% of NAV and binds on every rebalance.
    small = cc.run_point(base, CFG, aum=1.0e6)
    large = cc.run_point(base, CFG, aum=5.0e7)
    return small, large


def test_the_holdings_differ_across_aum_levels_and_the_cap_binds_at_the_top(
    two_points: tuple[cc.CurvePoint, cc.CurvePoint],
) -> None:
    small, large = two_points
    assert cc.holdings_distance(small, large) > 1e-3
    assert large.adv_binding_share > small.adv_binding_share
    assert large.adv_binding_share > 0.0
    # Impact is a larger share of cost at the larger book, spread drag scale-free.
    assert large.impact_share > small.impact_share
    # A point is a cost, a turnover and a drag -- never a Sharpe.
    assert not any("sharpe" in f.name.lower() for f in dataclasses.fields(cc.CurvePoint))
    assert small.net_annual == pytest.approx(small.gross_annual - small.cost_drag_annual)


def test_a_point_asks_for_the_hard_cap_and_the_capacity_ladder() -> None:
    base = _inputs()
    captured: dict[str, Any] = {}
    real = grid.run_cell

    def spy(inputs: grid.CellInputs, settings: config.Config, *, progress: bool = False) -> Any:
        captured["inputs"] = inputs
        return real(inputs, settings, progress=progress)

    cc.grid_mod.run_cell = spy  # type: ignore[assignment]
    try:
        cc.run_point(base, CFG, aum=2.0e6)
    finally:
        cc.grid_mod.run_cell = real  # type: ignore[assignment]
    inputs = captured["inputs"]
    assert inputs.adv_participation_hard is True
    assert (
        inputs.relaxation_ladder == CAP.relaxation_ladder == ("adv_participation", "tracking_error")
    )
    assert inputs.nav == 2.0e6 and inputs.cell.tag == "aum_2000000"


def test_break_even_interpolation_and_the_dollar_argmax_by_hand() -> None:
    def point(aum: float, net: float) -> cc.CurvePoint:
        return cc.CurvePoint(
            aum=aum,
            regime="patient",
            cost_drag_annual=0.0,
            spread_drag_annual=0.0,
            impact_drag_annual=0.0,
            turnover_annual=0.0,
            gross_annual=net,
            net_annual=net,
            adv_binding_share=0.0,
            relaxed=0,
            fell_back=0,
            effective_assets_mean=0.0,
            largest_weight_median=0.0,
            inaccurate=0,
            resolve_max_l1=0.0,
            targets=pd.DataFrame(),
        )

    pts = [point(1e6, 0.03), point(1e7, 0.01), point(1e8, -0.01)]
    # Linear in log AUM between 1e7 (0.01) and 1e8 (-0.01): the zero is at 10^7.5.
    assert cc.break_even_interpolated(pts) == pytest.approx(10**7.5, rel=1e-9)
    assert cc.break_even_interpolated(pts[:2]) == float("inf")
    # Dollar net alpha: 3e4, 1e5, -1e6 -> the argmax is 1e7.
    assert cc.effective_argmax(pts) == 1e7


def test_the_rescaling_comparand_reproduces_the_reference_cells_own_cost_at_its_nav() -> None:
    base = _inputs()
    reference = grid.run_cell(base, CFG)
    aum = np.array([base.nav / 4, base.nav, base.nav * 4])
    structures, curve = cc.rescaling_comparand(reference, CFG, aum)
    assert set(structures) == {"patient", "urgent"}
    fs = structures["patient"]
    e = CFG.model.costs.total_cost_exponent
    # At the reference NAV the closed form on the reference's own structure IS its
    # recurring annual cost (the build from cash excluded): spread drag plus
    # tau kappa A^(e-1).
    rescaled = float(fs.cost_drag(np.array([base.nav]), exponent=e)[0] + fs.aum_invariant_drag)
    drag = cc.recurring_drag(reference)
    # EXACT against the cost engine with every rebalance's impact re-priced at the
    # FIXED AUM the closed form holds (the engine priced each at its own NAV_pre).
    bt = reference.backtest
    dates = base.rebalance_dates[1:]
    model = base.realised_cost
    y = CFG.model.costs.square_root_prefactor
    per_order = []
    for stamp in dates:
        terms = impact.trade_cost(
            bt.trades.loc[stamp].to_numpy(dtype=float),
            half_spread=model.half_spread.loc[stamp].to_numpy(dtype=float),
            daily_volatility=model.daily_volatility.loc[stamp].to_numpy(dtype=float),
            adv_over_nav=model.adv_dollars.loc[stamp].to_numpy(dtype=float) / base.nav,
            prefactor=y.patient,
            exponent=e,
        )
        per_order.append(float(terms.total.sum()))
    assert rescaled == pytest.approx(np.mean(per_order) * base.periods_per_year, rel=1e-9)
    # Against the engine's own charges it differs only by the NAV drift between
    # rebalances (each was priced at its NAV_pre, the closed form at one A).
    assert rescaled == pytest.approx(drag["total"], rel=0.01)
    assert fs.turnover == pytest.approx(drag["turnover"], rel=1e-9)
    # By hand from the engine: the recurring drag is the per-rebalance cost over NAV_pre
    # after the first trade, times periods per year.
    charged = (bt.total_cost / bt.nav_before_costs).to_numpy(dtype=float)[1:]
    assert drag["total"] == pytest.approx(charged.mean() * base.periods_per_year, rel=1e-12)
    # The identity's TC is on the holding periods and differs by the normalisation only.
    assert drag["total"] == pytest.approx(reference.identity.cost_drag, rel=0.25)
    assert curve.gross_alpha == reference.identity.gross_return
    assert list(curve.net_alpha.columns) == ["patient", "urgent"]
    # Two regimes, quadratic in kappa: the urgent kappa is Y_u/Y_p times the patient one.
    y = CFG.model.costs.square_root_prefactor
    assert structures["urgent"].kappa == pytest.approx(fs.kappa * y.urgent / y.patient, rel=1e-12)
    assert capacity.capacity_multiplier(y.urgent / y.patient, exponent=e) == pytest.approx(
        (y.patient / y.urgent) ** 2
    )


def test_the_metrics_section_refuses_a_sharpe_and_the_checks_score(
    two_points: tuple[cc.CurvePoint, cc.CurvePoint],
) -> None:
    small, large = two_points
    base = _inputs()
    reference = grid.run_cell(base, CFG)
    aum = np.array([small.aum, large.aum])
    urgent_small = dataclasses.replace(small, regime="urgent")
    urgent_large = dataclasses.replace(large, regime="urgent")
    structures, curve = cc.rescaling_comparand(reference, CFG, aum)
    out = cc.CapacityResult(
        aum=aum,
        bounds=(small.aum, large.aum),
        points=(small, large, urgent_small, urgent_large),
        reference=reference,
        structures=structures,
        closed_form=curve,
        gross_alpha=reference.identity.gross_return,
    )
    out = dataclasses.replace(out, checks=cc.registered_checks(out, CFG))
    legs = {str(c["leg"]) for c in out.checks}
    assert {"a/patient", "b/patient", "c/patient", "d/patient", "a/urgent"} <= legs
    holds = {str(c["leg"]): c["holds"] for c in out.checks}
    assert holds["a/patient"] is True and holds["d/patient"] is True
    section = cc.metrics_section(out, CFG)
    assert section["category"] == "data-diagnostic" and section["report_sharpe"] is False
    # The one permitted mention is the switch that says there is none.
    text = json.dumps(section).lower().replace('"report_sharpe": false', "")
    assert "sharpe" not in text
    with pytest.raises(cc.CapacityError, match="Sharpe"):
        cc._assert_no_sharpe({"regimes": {"patient": {"points": [{"sharpe_net": 1.0}]}}})
    report = cc.render(out, CFG)
    assert "NO Sharpe per point" in report and "re-optimised" in report


def test_the_grid_points_are_the_ruled_rule_in_dollars() -> None:
    from types import SimpleNamespace

    names = [f"a{i}" for i in range(13)]
    universe = SimpleNamespace(
        assets=tuple(names), adv_median=pd.Series([31.2e6 + i for i in range(13)], index=names)
    )
    bounds, aum = cc.aum_points(universe, CFG)  # type: ignore[arg-type]
    rule = CFG.model.costs.capacity.aum_grid
    assert bounds[0] == pytest.approx(rule.low_participation_of_adv * 13 * 31.2e6, rel=1e-12)
    assert bounds[1] == pytest.approx(rule.high_participation_of_adv * 13 * 31.2e6, rel=1e-12)
    assert (
        len(aum) == rule.points
        and aum[0] == pytest.approx(bounds[0])
        and aum[-1] == pytest.approx(bounds[1])
    )
    assert np.allclose(np.diff(np.log(aum)), np.diff(np.log(aum))[0])
