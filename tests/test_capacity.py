"""SPEC.md 10.4's capacity arithmetic on SYNTHETIC inputs with hand-computed values (W5-P2).

No project input reaches this file: alpha_g, tau and the structure are chosen so
that every power is exact and every expected value can be checked by hand.

The synthetic structure used throughout:

    Y = 0.5, e = 3/2, R = 12 rebalances a year
    z     = [0.04, 0.01]        fraction of NAV per rebalance
    sigma = [0.02, 0.01]        daily
    V     = [1e8,  4e8]         dollars of ADV

    |z|^1.5  = [0.008, 0.001];  sqrt(V) = [1e4, 2e4]
    numerator = 0.5*0.02*0.008/1e4 + 0.5*0.01*0.001/2e4 = 8e-9 + 2.5e-10 = 8.25e-9
    sum|z|    = 0.05
    kappa     = 8.25e-9 / 0.05 = 1.65e-7   ($^-1/2)
    tau       = 12 * 0.05     = 0.6

    with alpha_g = 0.0198 / year:
    A_BE  = (0.0198 / (0.6 * 1.65e-7))^2 = (0.0198 / 9.9e-8)^2 = (2e5)^2 = 4e10
    A_eff = (0.0198 / (1.5 * 9.9e-8))^2  = (2e5 / 1.5)^2 = 4e10 * 4/9 = 1.7777...e10
    TC(1e8) = 0.6 * 1.65e-7 * sqrt(1e8) = 0.6 * 1.65e-7 * 1e4 = 9.9e-4
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mafrm.config import AumGridRule, load
from mafrm.costs import capacity, impact

Y = 0.5
E = 1.5
R = 12.0
Z = [0.04, 0.01]
SIGMA = [0.02, 0.01]
V = [1e8, 4e8]
KAPPA = 1.65e-7
TAU = 0.6
ALPHA_G = 0.0198
A_BE = 4e10
EXACT = {"rel": 1e-12, "abs": 0.0}


# ---------------------------------------------------------------------------
# The closed form, term by term
# ---------------------------------------------------------------------------


def test_cost_drag_hand_computed() -> None:
    drag = capacity.cost_drag(1e8, turnover=TAU, kappa=KAPPA, exponent=E)
    assert float(drag) == pytest.approx(9.9e-4, **EXACT)
    # A vector of AUMs comes back with the same shape, and zero AUM is zero drag.
    grid = capacity.cost_drag([0.0, 1e8, 4e8], turnover=TAU, kappa=KAPPA, exponent=E)
    assert grid.tolist() == pytest.approx([0.0, 9.9e-4, 1.98e-3], **EXACT)


def test_net_alpha_is_gross_minus_drag() -> None:
    net = capacity.net_alpha(1e8, gross_alpha=ALPHA_G, turnover=TAU, kappa=KAPPA, exponent=E)
    assert float(net) == pytest.approx(0.0198 - 9.9e-4, **EXACT)


def test_break_even_aum_hand_computed() -> None:
    a_be = capacity.break_even_aum(gross_alpha=ALPHA_G, turnover=TAU, kappa=KAPPA, exponent=E)
    assert a_be == pytest.approx(A_BE, **EXACT)
    # And the net alpha there is zero, to rounding.
    net = capacity.net_alpha(a_be, gross_alpha=ALPHA_G, turnover=TAU, kappa=KAPPA, exponent=E)
    assert float(net) == pytest.approx(0.0, abs=1e-15)


def test_effective_aum_is_four_ninths_of_break_even_and_net_alpha_there_is_a_third() -> None:
    a_eff = capacity.effective_aum(
        gross_alpha=ALPHA_G, turnover=TAU, kappa=KAPPA, exponent=E, minimum_net_alpha=0.0
    )
    assert a_eff == pytest.approx((2e5 / 1.5) ** 2, **EXACT)
    assert a_eff / A_BE == pytest.approx(4.0 / 9.0, **EXACT)
    assert a_eff / A_BE == pytest.approx(capacity.EFFECTIVE_TO_BREAK_EVEN_RATIO, **EXACT)
    # 1 - sqrt(4/9) = 1/3 of gross alpha survives at the marginal point.
    net = capacity.net_alpha(a_eff, gross_alpha=ALPHA_G, turnover=TAU, kappa=KAPPA, exponent=E)
    assert float(net) == pytest.approx(ALPHA_G / 3.0, rel=1e-12)


def test_effective_to_break_even_ratio_general_form_reduces_to_four_ninths_at_three_halves() -> (
    None
):
    """e^(-1/(e-1)): (3/2)^-2 = 4/9; at e = 2 it is 1/2; at e = 3 it is 1/sqrt(3)."""
    assert capacity.effective_to_break_even_ratio(1.5) == pytest.approx(4.0 / 9.0, **EXACT)
    assert capacity.effective_to_break_even_ratio(2.0) == pytest.approx(0.5, **EXACT)
    assert capacity.effective_to_break_even_ratio(3.0) == pytest.approx(1 / math.sqrt(3.0), **EXACT)


def test_the_configured_exponent_makes_the_hard_coded_ratio_true() -> None:
    """SPEC.md 10.4 says to hard-code 4/9. That is only true at e = 3/2; the config must agree."""
    e = load().model.costs.total_cost_exponent
    assert capacity.effective_to_break_even_ratio(e) == pytest.approx(
        capacity.EFFECTIVE_TO_BREAK_EVEN_RATIO, **EXACT
    )


def test_minimum_net_alpha_other_than_zero_is_refused_until_sourced() -> None:
    for bad in (0.01, -0.01, 1e-9):
        with pytest.raises(capacity.CapacityError, match="sourced"):
            capacity.effective_aum(
                gross_alpha=ALPHA_G,
                turnover=TAU,
                kappa=KAPPA,
                exponent=E,
                minimum_net_alpha=bad,
            )


# ---------------------------------------------------------------------------
# Quadratic sensitivity to kappa -- the band that must accompany every number
# ---------------------------------------------------------------------------


def test_halving_kappa_quadruples_capacity_exactly() -> None:
    a_be = capacity.break_even_aum(gross_alpha=ALPHA_G, turnover=TAU, kappa=KAPPA, exponent=E)
    half = capacity.break_even_aum(gross_alpha=ALPHA_G, turnover=TAU, kappa=KAPPA / 2, exponent=E)
    assert half / a_be == pytest.approx(4.0, **EXACT)
    assert capacity.capacity_multiplier(0.5, exponent=E) == pytest.approx(4.0, **EXACT)
    assert capacity.capacity_multiplier(2.0, exponent=E) == pytest.approx(0.25, **EXACT)
    # Doubling turnover is the same as doubling kappa: both quarter capacity.
    double_tau = capacity.break_even_aum(
        gross_alpha=ALPHA_G, turnover=2 * TAU, kappa=KAPPA, exponent=E
    )
    assert double_tau / a_be == pytest.approx(0.25, **EXACT)


def test_two_regime_band_is_the_squared_prefactor_ratio() -> None:
    """kappa is linear in Y, so A_BE(urgent)/A_BE(patient) = (0.58/1.40)^2 = 0.171633."""
    cfg = load().model.costs
    y_p, y_u = cfg.square_root_prefactor.patient, cfg.square_root_prefactor.urgent
    table = capacity.kappa_sensitivity(
        gross_alpha=ALPHA_G,
        turnover=TAU,
        kappa_by_regime={"patient": KAPPA, "urgent": KAPPA * y_u / y_p},
        exponent=cfg.total_cost_exponent,
        minimum_net_alpha=cfg.capacity.minimum_net_alpha,
    )
    ratio = table.loc["urgent", "break_even_aum"] / table.loc["patient", "break_even_aum"]
    assert ratio == pytest.approx((0.58 / 1.40) ** 2, rel=1e-12)
    assert ratio == pytest.approx(0.171633, abs=5e-7)
    assert 1.0 / ratio == pytest.approx(5.8264, abs=5e-5)
    # Both regimes are rows; the 4/9 holds on each.
    assert list(table.index) == ["patient", "urgent"]
    for regime in table.index:
        assert table.loc[regime, "effective_aum"] / table.loc[
            regime, "break_even_aum"
        ] == pytest.approx(4.0 / 9.0, **EXACT)


# ---------------------------------------------------------------------------
# kappa from a structure, pinned against the SPEC.md 7.1 cost engine
# ---------------------------------------------------------------------------


def test_structure_kappa_hand_computed() -> None:
    fs = capacity.structure_kappa(
        Z, daily_volatility=SIGMA, adv_dollars=V, prefactor=Y, exponent=E, rebalances_per_year=R
    )
    assert fs.kappa == pytest.approx(KAPPA, **EXACT)
    assert fs.turnover == pytest.approx(TAU, **EXACT)
    assert fs.aum_invariant_drag == 0.0
    assert fs.rebalances_in_sample == 1
    # Sells count like buys: the sign of the trade is irrelevant to tau and kappa.
    flipped = capacity.structure_kappa(
        [-0.04, 0.01],
        daily_volatility=SIGMA,
        adv_dollars=V,
        prefactor=Y,
        exponent=E,
        rebalances_per_year=R,
    )
    assert flipped.kappa == fs.kappa and flipped.turnover == fs.turnover


def test_structure_spread_drag_hand_computed() -> None:
    """a = [5bp, 1bp]: per rebalance 0.0005*0.04 + 0.0001*0.01 = 2.1e-5; x12 = 2.52e-4 a year."""
    fs = capacity.structure_kappa(
        Z,
        daily_volatility=SIGMA,
        adv_dollars=V,
        prefactor=Y,
        exponent=E,
        rebalances_per_year=R,
        half_spread=[0.0005, 0.0001],
    )
    assert fs.aum_invariant_drag == pytest.approx(2.52e-4, **EXACT)
    assert fs.kappa == pytest.approx(KAPPA, **EXACT)  # the spread does not touch kappa


@pytest.mark.parametrize("aum", [1e7, 1e8, 1e9, 4e10, 1.7777777777777778e10])
def test_structure_kappa_reproduces_the_cost_engine_at_every_aum(aum: float) -> None:
    """experiments.md row 198: R * engine impact at AUM A == tau * kappa * sqrt(A), to 1e-12.

    The engine sees ``adv_over_nav = V/A``; the closed form sees ``kappa``. If
    the two disagree anywhere on the grid the kappa derivation is wrong.
    """
    a_i = [0.0005, 0.0001]
    fs = capacity.structure_kappa(
        Z,
        daily_volatility=SIGMA,
        adv_dollars=V,
        prefactor=Y,
        exponent=E,
        rebalances_per_year=R,
        half_spread=a_i,
    )
    trades = pd.DataFrame([Z], index=pd.to_datetime(["2020-01-31"]), columns=["a", "b"])
    panel = impact.rebalance_costs(
        trades,
        half_spread=pd.DataFrame([a_i], index=trades.index, columns=trades.columns),
        daily_volatility=pd.DataFrame([SIGMA], index=trades.index, columns=trades.columns),
        adv_over_nav=pd.DataFrame(
            [[v / aum for v in V]], index=trades.index, columns=trades.columns
        ),
        prefactor=Y,
        exponent=E,
    )
    engine_impact_per_year = R * float(panel.impact.sum(axis=1).iloc[0])
    engine_spread_per_year = R * float(panel.spread.sum(axis=1).iloc[0])
    assert engine_impact_per_year == pytest.approx(float(fs.cost_drag(aum, exponent=E)), rel=1e-12)
    assert engine_spread_per_year == pytest.approx(fs.aum_invariant_drag, rel=1e-12)


def test_structure_kappa_over_a_sample_of_rebalances_averages_them() -> None:
    """Two rebalances, the second trading only asset b. tau = 12 * (0.05 + 0.02)/2 = 0.42."""
    z = [[0.04, 0.01], [0.0, 0.02]]
    fs = capacity.structure_kappa(
        z, daily_volatility=SIGMA, adv_dollars=V, prefactor=Y, exponent=E, rebalances_per_year=R
    )
    assert fs.turnover == pytest.approx(0.42, **EXACT)
    assert fs.rebalances_in_sample == 2
    # Engine identity on the sample mean at A = 1e9.
    aum = 1e9
    dates = pd.to_datetime(["2020-01-31", "2020-02-28"])
    trades = pd.DataFrame(z, index=dates, columns=["a", "b"])
    like = pd.DataFrame(1.0, index=dates, columns=trades.columns)
    panel = impact.rebalance_costs(
        trades,
        half_spread=like * 0.0,
        daily_volatility=like * np.array(SIGMA),
        adv_over_nav=like * (np.array(V) / aum),
        prefactor=Y,
        exponent=E,
    )
    engine = R * float(panel.impact.sum(axis=1).mean())
    assert engine == pytest.approx(float(fs.cost_drag(aum, exponent=E)), rel=1e-12)


def test_structure_kappa_refuses_bad_inputs() -> None:
    with pytest.raises(capacity.CapacityError, match="never trades"):
        capacity.structure_kappa(
            [0.0, 0.0],
            daily_volatility=SIGMA,
            adv_dollars=V,
            prefactor=Y,
            exponent=E,
            rebalances_per_year=R,
        )
    with pytest.raises(capacity.CapacityError, match="adv_dollars"):
        capacity.structure_kappa(
            Z,
            daily_volatility=SIGMA,
            adv_dollars=[1e8, 0.0],
            prefactor=Y,
            exponent=E,
            rebalances_per_year=R,
        )
    with pytest.raises(capacity.CapacityError, match="NaN"):
        capacity.structure_kappa(
            [0.04, float("nan")],
            daily_volatility=SIGMA,
            adv_dollars=V,
            prefactor=Y,
            exponent=E,
            rebalances_per_year=R,
        )
    # A zero trade against a bad input is fine: the input is never read there.
    fs = capacity.structure_kappa(
        [0.04, 0.0],
        daily_volatility=SIGMA,
        adv_dollars=[1e8, float("nan")],
        prefactor=Y,
        exponent=E,
        rebalances_per_year=R,
    )
    assert fs.kappa == pytest.approx(8e-9 / 0.04, **EXACT)


# ---------------------------------------------------------------------------
# The L1 term enters exactly, as a reduction of gross alpha
# ---------------------------------------------------------------------------


def test_aum_invariant_drag_shifts_break_even_exactly_like_a_lower_gross_alpha() -> None:
    d = 2.52e-4
    with_drag = capacity.break_even_aum(
        gross_alpha=ALPHA_G, turnover=TAU, kappa=KAPPA, exponent=E, aum_invariant_drag=d
    )
    lower_alpha = capacity.break_even_aum(
        gross_alpha=ALPHA_G - d, turnover=TAU, kappa=KAPPA, exponent=E
    )
    assert with_drag == pytest.approx(lower_alpha, **EXACT)
    assert with_drag < A_BE
    # The net alpha at that AUM is zero once the drag is charged.
    net = capacity.net_alpha(
        with_drag, gross_alpha=ALPHA_G, turnover=TAU, kappa=KAPPA, exponent=E, aum_invariant_drag=d
    )
    assert float(net) == pytest.approx(0.0, abs=1e-15)


def test_gross_alpha_that_does_not_cover_the_spread_has_zero_capacity() -> None:
    a_be = capacity.break_even_aum(
        gross_alpha=1e-4, turnover=TAU, kappa=KAPPA, exponent=E, aum_invariant_drag=2.52e-4
    )
    assert a_be == 0.0
    a_eff = capacity.effective_aum(
        gross_alpha=1e-4,
        turnover=TAU,
        kappa=KAPPA,
        exponent=E,
        minimum_net_alpha=0.0,
        aum_invariant_drag=2.52e-4,
    )
    assert a_eff == 0.0


# ---------------------------------------------------------------------------
# The AUM grid is a rule; the marked points are not read off it
# ---------------------------------------------------------------------------


def test_aum_grid_bounds_hand_computed() -> None:
    """z_max = 5% of NAV, proxy ADV $1e9: 0.1% of ADV at A = 0.001*1e9/0.05 = $20M; 10% at $2B."""
    rule = load().model.costs.capacity.aum_grid
    low, high = capacity.aum_grid_bounds(
        largest_trade_fraction_of_nav=0.05, proxy_adv_dollars=1e9, rule=rule
    )
    assert low == pytest.approx(2e7, **EXACT)
    assert high == pytest.approx(2e9, **EXACT)
    grid = capacity.aum_grid((low, high), rule=rule)
    assert grid.shape == (rule.points,)
    assert grid[0] == pytest.approx(low, **EXACT) and grid[-1] == pytest.approx(high, **EXACT)
    # Log spacing: consecutive ratios are constant.
    ratios = grid[1:] / grid[:-1]
    assert np.allclose(ratios, ratios[0], rtol=1e-12)
    assert ratios[0] == pytest.approx(100.0 ** (1.0 / (rule.points - 1)), rel=1e-12)


def test_marked_points_are_closed_form_not_grid_points() -> None:
    rule = AumGridRule(
        spacing="log", low_participation_of_adv=0.001, high_participation_of_adv=0.10, points=50
    )
    grid = capacity.aum_grid((2e7, 2e11), rule=rule)
    curve = capacity.capacity_curve(
        grid,
        gross_alpha=ALPHA_G,
        turnover=TAU,
        kappa_by_regime={"patient": KAPPA},
        exponent=E,
        minimum_net_alpha=0.0,
    )
    a_be = float(curve.points.loc["patient", "break_even_aum"])
    a_eff = float(curve.points.loc["patient", "effective_aum"])
    assert a_be == pytest.approx(A_BE, **EXACT)
    assert a_eff == pytest.approx(A_BE * 4.0 / 9.0, **EXACT)
    assert not np.isclose(grid, a_be, rtol=1e-6).any()
    assert not np.isclose(grid, a_eff, rtol=1e-6).any()
    # Changing the display resolution moves no marked point.
    coarse = capacity.capacity_curve(
        capacity.aum_grid((2e7, 2e11), rule=AumGridRule("log", 0.001, 0.10, 7)),
        gross_alpha=ALPHA_G,
        turnover=TAU,
        kappa_by_regime={"patient": KAPPA},
        exponent=E,
        minimum_net_alpha=0.0,
    )
    assert coarse.points.equals(curve.points)


def test_aum_grid_refuses_a_rule_that_is_not_the_ruled_one() -> None:
    with pytest.raises(capacity.CapacityError, match="log"):
        capacity.aum_grid((1.0, 2.0), rule=AumGridRule("linear", 0.001, 0.10, 5))
    with pytest.raises(capacity.CapacityError, match="0 < low < high"):
        capacity.aum_grid((2.0, 1.0), rule=AumGridRule("log", 0.001, 0.10, 5))


def test_capacity_curve_frame_shape_and_units() -> None:
    grid = np.array([0.0, 1e10, A_BE, 4 * A_BE])
    curve = capacity.capacity_curve(
        grid,
        gross_alpha=ALPHA_G,
        turnover=TAU,
        kappa_by_regime={"patient": KAPPA, "urgent": 2 * KAPPA},
        exponent=E,
        minimum_net_alpha=0.0,
    )
    assert list(curve.net_alpha.columns) == ["patient", "urgent"]
    assert curve.net_alpha.index.name == "aum"
    # At zero AUM the net alpha is the gross alpha; at A_BE it is zero; at 4 A_BE it is -alpha_g.
    patient = curve.net_alpha["patient"]
    assert patient.iloc[0] == pytest.approx(ALPHA_G, **EXACT)
    assert patient.iloc[2] == pytest.approx(0.0, abs=1e-15)
    assert patient.iloc[3] == pytest.approx(-ALPHA_G, rel=1e-12)
    # Twice the kappa: break-even at a quarter of the AUM.
    assert curve.points.loc["urgent", "break_even_aum"] == pytest.approx(A_BE / 4.0, **EXACT)


def test_the_closed_form_says_what_it_is() -> None:
    """SPEC.md 10.4: the rescaling overstates cost. The module must say so, not just do it."""
    doc = (capacity.__doc__ or "").lower()
    assert "rescaling" in doc and "overstates cost" in doc
    assert "re-optimis" in doc
