"""SPEC.md 6.4's closed form, with its unit attached. W4-P3.

The thing pinned here is not the arithmetic -- ``(1 - x)^-2`` is one line -- but
the unit discipline W4-P3 ruling 6 asked for: the same number means two things
a factor of two apart, and nothing may emit it without saying which.
"""

from __future__ import annotations

import math

import pytest

from mafrm.numerics import effective_sample_size
from mafrm.risk.shepard import SecondOrderRisk, ShepardError, Unit, jensen_term, second_order_risk


def test_spec_6_4s_worked_row_by_hand() -> None:
    """K = 6 at T_eff = 242.37: ratio 0.02476, variance multiplier 1.0514, volatility 1.0254."""
    risk = second_order_risk(6, halflife=84)
    assert risk.effective_observations == pytest.approx(2 * 84 / math.log(2), rel=1e-12)
    assert risk.ratio == pytest.approx(6 / 242.3728, abs=1e-6)
    # (1 - 0.0247553)^-2 = 1 / 0.9752447^2 = 1 / 0.951102 = 1.051412
    assert risk.variance_multiplier == pytest.approx(1.05141, abs=1e-5)
    assert risk.volatility_multiplier == pytest.approx(1.025384, abs=1e-5)
    # SPEC.md 6.4's table quotes 5.1%: that is the VARIANCE multiplier minus one.
    assert risk.table_convention == pytest.approx(0.0514, abs=5e-4)
    assert risk.understatement(Unit.VARIANCE) == pytest.approx(0.0514, abs=5e-4)
    assert risk.understatement(Unit.VOLATILITY) == pytest.approx(0.0254, abs=5e-4)


def test_the_two_readings_differ_by_a_factor_of_two_to_first_order() -> None:
    risk = second_order_risk(13, halflife=84)
    variance = risk.understatement(Unit.VARIANCE)
    volatility = risk.understatement(Unit.VOLATILITY)
    assert variance / volatility == pytest.approx(2.0, abs=0.06)
    assert risk.volatility_multiplier == pytest.approx(math.sqrt(risk.variance_multiplier))


def test_mwos_published_cross_check_fixes_the_reading() -> None:
    """SPEC.md 6.4: at T = 100, N = 50 Shepard 'predicts 2.0' against an observed B of 2.25.

    Only the volatility reading gives 2.0; the variance reading gives 4.0.
    """
    risk = second_order_risk(50, effective_observations=100)
    assert risk.multiplier(Unit.VOLATILITY) == pytest.approx(2.0)
    assert risk.multiplier(Unit.VARIANCE) == pytest.approx(4.0)


def test_render_carries_the_unit_and_format_refuses_without_one() -> None:
    risk = second_order_risk(6, halflife=84)
    assert risk.render(Unit.VARIANCE) == "5.1% (variance)"
    assert risk.render(Unit.VOLATILITY) == "2.5% (volatility)"
    with pytest.raises(ShepardError, match="unit"):
        f"{risk}"
    with pytest.raises(ShepardError, match="unit"):
        str(risk)


def test_saturation_is_inf_in_both_units() -> None:
    risk = second_order_risk(300, effective_observations=242.0)
    assert math.isinf(risk.variance_multiplier)
    assert math.isinf(risk.volatility_multiplier)
    assert "inf" in risk.render(Unit.VARIANCE)


def test_exactly_one_of_halflife_or_effective_observations() -> None:
    with pytest.raises(ShepardError):
        second_order_risk(6)
    with pytest.raises(ShepardError):
        second_order_risk(6, halflife=84, effective_observations=242.0)
    with pytest.raises(ShepardError):
        SecondOrderRisk(parameters=0, effective_observations=10.0)
    with pytest.raises(ShepardError):
        SecondOrderRisk(parameters=3, effective_observations=0.0)


def test_jensen_term_is_shepard_at_one_parameter() -> None:
    """(1 - 1/T)^-2 = 1 + 2/T + 3/T^2 + ...: the inverse-variance bias to first order."""
    term = jensen_term(effective_observations=242.0)
    assert term.parameters == 1
    assert term.understatement(Unit.VARIANCE) == pytest.approx(2 / 242, rel=0.01)
    assert term.understatement(Unit.VARIANCE) > 2 / 242


def test_halflife_route_matches_the_shared_effective_sample_size() -> None:
    """One T_eff formula in the project (mafrm.numerics), not a private copy here."""
    for halflife in (84, 252, 504):
        via_halflife = second_order_risk(6, halflife=halflife)
        via_t = second_order_risk(6, effective_observations=effective_sample_size(float(halflife)))
        assert via_halflife == via_t
