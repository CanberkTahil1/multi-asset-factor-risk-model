"""SPEC.md 7.1's cost function and 7.4's required tests, every value by hand.

The hand arithmetic uses inputs chosen so that the powers come out exact:
``V/v = 0.25`` gives ``sqrt(V/v) = 0.5``, and the trade sizes 0.01, 0.04 and
0.09 have ``|z|^{3/2}`` = 0.001, 0.008 and 0.027. Nothing here was produced by
the code and then pasted back.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mafrm import config
from mafrm.costs import impact

# The fixture inputs. Half-spread 5bp, sigma 2%/day, ADV a quarter of NAV.
A = 0.0005
SIGMA = 0.02
DEPTH = 0.25
Y_PATIENT = 0.58
Y_URGENT = 1.40
E = 1.5

EXACT = dict(rel=1e-12, abs=0.0)


# ---------------------------------------------------------------------------
# Each term at three trade sizes, both regimes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("z", "spread", "impact_patient", "impact_urgent"),
    [
        # spread = a*|z|; impact = Y*sigma*|z|^1.5 / sqrt(V/v) = Y*0.02*|z|^1.5/0.5
        #   patient: 0.58*0.02/0.5 = 0.0232 per |z|^1.5; urgent: 1.40*0.02/0.5 = 0.056
        # z = 0.01: |z|^1.5 = 0.001 -> spread 5e-6, impact 2.32e-5 / 5.6e-5
        (0.01, 5.0e-6, 2.32e-5, 5.6e-5),
        # z = 0.04: |z|^1.5 = 0.008 -> spread 2e-5, impact 1.856e-4 / 4.48e-4
        (0.04, 2.0e-5, 1.856e-4, 4.48e-4),
        # z = 0.09: |z|^1.5 = 0.027 -> spread 4.5e-5, impact 6.264e-4 / 1.512e-3
        (0.09, 4.5e-5, 6.264e-4, 1.512e-3),
    ],
)
def test_each_term_at_three_trade_sizes_hand_computed(
    z: float, spread: float, impact_patient: float, impact_urgent: float
) -> None:
    for y, expected_impact in ((Y_PATIENT, impact_patient), (Y_URGENT, impact_urgent)):
        terms = impact.trade_cost(
            z, half_spread=A, daily_volatility=SIGMA, adv_over_nav=DEPTH, prefactor=y, exponent=E
        )
        assert float(terms.spread) == pytest.approx(spread, **EXACT)
        assert float(terms.impact) == pytest.approx(expected_impact, **EXACT)
        assert float(terms.asymmetry) == 0.0
        assert float(terms.total) == pytest.approx(spread + expected_impact, **EXACT)


def test_total_cost_is_size_times_per_unit_impact_plus_spread() -> None:
    """SPEC.md 7.1's 'why 3/2': x * Y*sigma*sqrt(x/V) = Y*sigma*x^1.5/sqrt(V)."""
    z = 0.04
    per_unit = impact.impact_per_unit_traded(
        z / DEPTH, daily_volatility=SIGMA, prefactor=Y_PATIENT, exponent=E
    )
    # Q/V = 0.04/0.25 = 0.16, sqrt = 0.4 -> per unit = 0.58*0.02*0.4 = 0.00464
    assert float(per_unit) == pytest.approx(0.00464, **EXACT)
    terms = impact.trade_cost(
        z,
        half_spread=A,
        daily_volatility=SIGMA,
        adv_over_nav=DEPTH,
        prefactor=Y_PATIENT,
        exponent=E,
    )
    assert float(terms.impact) == pytest.approx(z * float(per_unit), **EXACT)
    assert float(terms.impact) == pytest.approx(1.856e-4, **EXACT)


def test_asymmetry_term_carries_the_sign_of_the_trade() -> None:
    c = 0.0003
    buy = impact.trade_cost(
        0.05,
        half_spread=A,
        daily_volatility=SIGMA,
        adv_over_nav=DEPTH,
        prefactor=Y_PATIENT,
        exponent=E,
        asymmetry=c,
    )
    sell = impact.trade_cost(
        -0.05,
        half_spread=A,
        daily_volatility=SIGMA,
        adv_over_nav=DEPTH,
        prefactor=Y_PATIENT,
        exponent=E,
        asymmetry=c,
    )
    assert float(buy.asymmetry) == pytest.approx(1.5e-5, **EXACT)
    assert float(sell.asymmetry) == pytest.approx(-1.5e-5, **EXACT)


# ---------------------------------------------------------------------------
# Zero trade -> exactly zero cost: the division-by-zero class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("half_spread", "sigma", "depth"),
    [
        (A, SIGMA, 0.0),  # no depth at all: 0/0 in a naive implementation
        (A, SIGMA, np.nan),  # ADV window not yet full
        (np.nan, SIGMA, DEPTH),  # spread window not yet full
        (A, np.nan, DEPTH),  # volatility window not yet full
        (np.nan, np.nan, np.nan),
        (A, 0.0, DEPTH),
    ],
)
def test_zero_trade_is_exactly_zero_whatever_the_other_inputs(
    half_spread: float, sigma: float, depth: float
) -> None:
    terms = impact.trade_cost(
        0.0,
        half_spread=half_spread,
        daily_volatility=sigma,
        adv_over_nav=depth,
        prefactor=Y_PATIENT,
        exponent=E,
    )
    assert float(terms.spread) == 0.0
    assert float(terms.impact) == 0.0
    assert float(terms.asymmetry) == 0.0
    assert float(terms.total) == 0.0
    assert not np.isnan(terms.total).any()


def test_zero_trade_in_a_vector_does_not_poison_its_neighbours() -> None:
    terms = impact.trade_cost(
        [0.0, 0.01],
        half_spread=[np.nan, A],
        daily_volatility=[np.nan, SIGMA],
        adv_over_nav=[0.0, DEPTH],
        prefactor=Y_PATIENT,
        exponent=E,
    )
    assert terms.total[0] == 0.0
    assert terms.total[1] == pytest.approx(5.0e-6 + 2.32e-5, **EXACT)


@pytest.mark.parametrize(
    ("half_spread", "sigma", "depth", "fragment"),
    [
        (A, SIGMA, 0.0, "adv_over_nav"),
        (A, SIGMA, -1.0, "adv_over_nav"),
        (A, SIGMA, np.nan, "adv_over_nav"),
        (np.nan, SIGMA, DEPTH, "half_spread"),
        (-A, SIGMA, DEPTH, "half_spread"),
        (A, np.nan, DEPTH, "daily_volatility"),
        (A, np.inf, DEPTH, "daily_volatility"),
    ],
)
def test_a_nonzero_trade_with_a_bad_input_raises_rather_than_returning_nan_or_inf(
    half_spread: float, sigma: float, depth: float, fragment: str
) -> None:
    with pytest.raises(impact.ImpactError, match=fragment):
        impact.trade_cost(
            0.01,
            half_spread=half_spread,
            daily_volatility=sigma,
            adv_over_nav=depth,
            prefactor=Y_PATIENT,
            exponent=E,
        )


def test_nan_trade_is_refused() -> None:
    with pytest.raises(impact.ImpactError, match="non-finite"):
        impact.trade_cost(
            np.nan,
            half_spread=A,
            daily_volatility=SIGMA,
            adv_over_nav=DEPTH,
            prefactor=Y_PATIENT,
            exponent=E,
        )


def test_exponent_must_be_convex() -> None:
    with pytest.raises(impact.ImpactError, match="convex"):
        impact.trade_cost(
            0.01,
            half_spread=A,
            daily_volatility=SIGMA,
            adv_over_nav=DEPTH,
            prefactor=Y_PATIENT,
            exponent=1.0,
        )


# ---------------------------------------------------------------------------
# Round trip and both sides
# ---------------------------------------------------------------------------


def test_round_trip_is_exactly_twice_the_one_way_cost() -> None:
    """Buy z then sell z. With c = 0 the two legs are identical in every term."""
    z = 0.04
    kwargs = dict(
        half_spread=A, daily_volatility=SIGMA, adv_over_nav=DEPTH, prefactor=Y_PATIENT, exponent=E
    )
    buy = impact.trade_cost(z, **kwargs)
    sell = impact.trade_cost(-z, **kwargs)
    one_way = 2.0e-5 + 1.856e-4
    assert float(buy.total) == pytest.approx(one_way, **EXACT)
    assert float(sell.total) == float(buy.total)  # bit-identical, not approximately
    assert float(buy.total) + float(sell.total) == 2.0 * float(buy.total)


def test_round_trip_with_asymmetry_nets_the_asymmetry_to_zero_exactly() -> None:
    """SPEC.md 7.4: twice the one-way cost plus impact asymmetry -- and c*z cancels."""
    z = 0.04
    c = 0.0003
    kwargs = dict(
        half_spread=A,
        daily_volatility=SIGMA,
        adv_over_nav=DEPTH,
        prefactor=Y_PATIENT,
        exponent=E,
        asymmetry=c,
    )
    buy = impact.trade_cost(z, **kwargs)
    sell = impact.trade_cost(-z, **kwargs)
    # legs differ by exactly 2*c*z = 2.4e-5 ...
    assert float(buy.total) - float(sell.total) == pytest.approx(2.4e-5, **EXACT)
    # ... and sum to twice the symmetric cost, the asymmetry gone
    assert float(buy.total) + float(sell.total) == pytest.approx(2.0 * (2.0e-5 + 1.856e-4), **EXACT)
    assert float(buy.asymmetry) + float(sell.asymmetry) == 0.0


def test_a_sell_costs_the_same_as_a_buy_of_the_same_size() -> None:
    kwargs = dict(
        half_spread=A, daily_volatility=SIGMA, adv_over_nav=DEPTH, prefactor=Y_URGENT, exponent=E
    )
    assert float(impact.trade_cost(-0.09, **kwargs).total) == float(
        impact.trade_cost(0.09, **kwargs).total
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibration_check_one_percent_of_adv_at_two_percent_vol_is_11_6bp() -> None:
    """SPEC.md 7.2's cross-check: 0.58 * 0.02 * sqrt(0.01) = 0.58 * 0.02 * 0.1 = 0.00116."""
    per_unit = impact.impact_per_unit_traded(
        0.01, daily_volatility=0.02, prefactor=0.58, exponent=1.5
    )
    assert float(per_unit) == pytest.approx(0.00116, **EXACT)
    assert float(per_unit) / 1e-4 == pytest.approx(11.6, **EXACT)


@pytest.mark.parametrize(
    ("cost_bps", "participation", "expected"),
    [
        # 17bp at 2%: 0.0017 / (0.02 * sqrt(0.02)) = 0.0017 / 0.00282843 = 0.6010
        (17.0, 0.02, 0.601),
        # 28bp at 6%: 0.0028 / (0.02 * sqrt(0.06)) = 0.0028 / 0.00489898 = 0.5715
        (28.0, 0.06, 0.572),
        # 40bp at 2%: 0.0040 / 0.00282843 = 1.4142
        (40.0, 0.02, 1.414),
    ],
)
def test_implied_prefactor_reproduces_spec_7_2_table(
    cost_bps: float, participation: float, expected: float
) -> None:
    y = impact.implied_prefactor(
        cost_of_trade=cost_bps * 1e-4,
        participation=participation,
        daily_volatility=0.02,
        exponent=1.5,
    )
    assert y == pytest.approx(expected, abs=5e-4)


def test_implied_prefactor_inverts_per_unit_impact() -> None:
    y = 0.73
    q = 0.035
    cost = float(impact.impact_per_unit_traded(q, daily_volatility=SIGMA, prefactor=y, exponent=E))
    assert impact.implied_prefactor(
        cost_of_trade=cost, participation=q, daily_volatility=SIGMA, exponent=E
    ) == pytest.approx(y, **EXACT)


def test_config_prefactors_are_what_the_anchors_imply() -> None:
    """costs.square_root_prefactor rounds the mean of the implied values, both regimes."""
    costs = config.load().model.costs
    anchors = costs.calibration_anchors
    patient = [
        impact.implied_prefactor(
            cost_of_trade=p.cost_bps * 1e-4,
            participation=p.participation_of_adv,
            daily_volatility=anchors.daily_volatility,
            exponent=costs.total_cost_exponent,
        )
        for p in anchors.patient.points
    ]
    urgent = [
        impact.implied_prefactor(
            cost_of_trade=p.cost_bps * 1e-4,
            participation=p.participation_of_adv,
            daily_volatility=anchors.daily_volatility,
            exponent=costs.total_cost_exponent,
        )
        for p in anchors.urgent.points
    ]
    assert costs.square_root_prefactor.patient == pytest.approx(
        sum(patient) / len(patient), abs=0.01
    )
    assert costs.square_root_prefactor.urgent == pytest.approx(sum(urgent) / len(urgent), abs=0.02)
    assert (
        costs.square_root_prefactor.urgent / costs.square_root_prefactor.patient
        == pytest.approx(2.41, abs=0.01)
    )


# ---------------------------------------------------------------------------
# SPEC.md 7.3 scaling laws, each pinned against the cost function itself
# ---------------------------------------------------------------------------


def _impact_fraction_of_nav(z: float, *, depth: float, sigma: float = SIGMA) -> float:
    return float(
        impact.trade_cost(
            z,
            half_spread=0.0,
            daily_volatility=sigma,
            adv_over_nav=depth,
            prefactor=Y_PATIENT,
            exponent=E,
        ).impact
    )


def test_doubling_aum_raises_bps_drag_by_root_two_not_two() -> None:
    """Fixed structure: z unchanged, V/v halves. Drag (fraction of NAV) x sqrt(2)."""
    base = _impact_fraction_of_nav(0.04, depth=DEPTH)
    doubled = _impact_fraction_of_nav(0.04, depth=DEPTH / 2.0)
    assert doubled / base == pytest.approx(math.sqrt(2.0), **EXACT)
    assert impact.ScalingLaws(E).drag_multiplier(aum_ratio=2.0) == pytest.approx(
        math.sqrt(2.0), **EXACT
    )


def test_doubling_aum_raises_dollar_cost_by_two_to_the_three_halves() -> None:
    base_dollars = _impact_fraction_of_nav(0.04, depth=DEPTH) * 1.0
    doubled_dollars = _impact_fraction_of_nav(0.04, depth=DEPTH / 2.0) * 2.0
    assert doubled_dollars / base_dollars == pytest.approx(2.0**1.5, **EXACT)
    assert impact.ScalingLaws(E).aum_exponent_dollars == 1.5


def test_doubling_turnover_doubles_the_drag() -> None:
    """Twice as many trades of the same size: the sum doubles. Turnover is linear."""
    one = _impact_fraction_of_nav(0.04, depth=DEPTH)
    two = float(
        np.sum(
            impact.trade_cost(
                [0.04, 0.04],
                half_spread=0.0,
                daily_volatility=SIGMA,
                adv_over_nav=DEPTH,
                prefactor=Y_PATIENT,
                exponent=E,
            ).impact
        )
    )
    assert two == pytest.approx(2.0 * one, **EXACT)
    assert impact.ScalingLaws(E).drag_multiplier(aum_ratio=1.0, turnover_ratio=2.0) == 2.0


def test_cost_is_linear_in_volatility() -> None:
    assert _impact_fraction_of_nav(0.04, depth=DEPTH, sigma=2 * SIGMA) == pytest.approx(
        2.0 * _impact_fraction_of_nav(0.04, depth=DEPTH), **EXACT
    )


def test_cost_falls_with_root_adv() -> None:
    assert _impact_fraction_of_nav(0.04, depth=4 * DEPTH) == pytest.approx(
        _impact_fraction_of_nav(0.04, depth=DEPTH) / 2.0, **EXACT
    )
    assert impact.ScalingLaws(E).adv_exponent == -0.5


def test_scaling_laws_reject_a_non_convex_exponent() -> None:
    with pytest.raises(impact.ImpactError):
        impact.ScalingLaws(1.0)


# ---------------------------------------------------------------------------
# Costs on actual turnover, both sides, at the correct time
# ---------------------------------------------------------------------------


def _panel(values: dict[str, list[float]], dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(values, index=pd.to_datetime(dates))


DATES = ["2020-01-31", "2020-02-28"]


def test_rebalance_costs_both_sides_at_each_dates_own_inputs_hand_computed() -> None:
    """Date 1 switches A -> B; date 2 buys A back into a widened spread.

    Date 1, A sells 0.04: a = 5bp, sigma = 1%, V/v = 0.04 so Q/V = 1 and per-unit
    impact = Y*sigma = 0.0058; spread 0.0005*0.04 = 2e-5; impact 0.04*0.0058 =
    2.32e-4.  B buys 0.04: a = 10bp, sigma = 2%, V/v = 0.16 so Q/V = 0.25,
    sqrt 0.5, per-unit 0.58*0.02*0.5 = 0.0058; spread 4e-5; impact 2.32e-4.
    Date-1 total = 2e-5 + 2.32e-4 + 4e-5 + 2.32e-4 = 5.24e-4.

    Date 2, A buys 0.04 with its spread now 20bp: spread 8e-5, impact 2.32e-4,
    total 3.12e-4.  B does not trade and its date-2 inputs are NaN -- which must
    cost exactly zero, not poison the row.
    """
    trades = _panel({"A": [-0.04, 0.04], "B": [0.04, 0.0]}, DATES)
    half_spread = _panel({"A": [0.0005, 0.0020], "B": [0.0010, np.nan]}, DATES)
    sigma = _panel({"A": [0.01, 0.01], "B": [0.02, np.nan]}, DATES)
    depth = _panel({"A": [0.04, 0.04], "B": [0.16, np.nan]}, DATES)

    panel = impact.rebalance_costs(
        trades,
        half_spread=half_spread,
        daily_volatility=sigma,
        adv_over_nav=depth,
        prefactor=Y_PATIENT,
        exponent=E,
    )
    assert panel.spread.loc[DATES[0], "A"] == pytest.approx(2.0e-5, **EXACT)
    assert panel.impact.loc[DATES[0], "A"] == pytest.approx(2.32e-4, **EXACT)
    assert panel.spread.loc[DATES[0], "B"] == pytest.approx(4.0e-5, **EXACT)
    assert panel.impact.loc[DATES[0], "B"] == pytest.approx(2.32e-4, **EXACT)
    assert panel.by_date.loc[DATES[0]] == pytest.approx(5.24e-4, **EXACT)
    # the sell leg was charged: A's date-1 cost is not zero
    assert panel.by_asset.loc[DATES[0], "A"] == pytest.approx(2.52e-4, **EXACT)
    # date 2 reads date 2's spread, not date 1's
    assert panel.spread.loc[DATES[1], "A"] == pytest.approx(8.0e-5, **EXACT)
    assert panel.by_date.loc[DATES[1]] == pytest.approx(3.12e-4, **EXACT)
    assert panel.by_asset.loc[DATES[1], "B"] == 0.0


def test_rebalance_costs_never_fill_a_missing_input_from_another_date() -> None:
    """A spread missing at a rebalance where the asset traded is a stop, not a carry."""
    trades = _panel({"A": [0.04, 0.04]}, DATES)
    half_spread = pd.DataFrame({"A": [0.0005]}, index=pd.to_datetime(DATES[:1]))  # date 2 absent
    sigma = _panel({"A": [0.01, 0.01]}, DATES)
    depth = _panel({"A": [0.04, 0.04]}, DATES)
    with pytest.raises(impact.ImpactError, match="half_spread"):
        impact.rebalance_costs(
            trades,
            half_spread=half_spread,
            daily_volatility=sigma,
            adv_over_nav=depth,
            prefactor=Y_PATIENT,
            exponent=E,
        )


def test_rebalance_costs_refuse_nan_trades_and_missing_columns() -> None:
    trades = _panel({"A": [np.nan, 0.04]}, DATES)
    ok = _panel({"A": [0.01, 0.01]}, DATES)
    with pytest.raises(impact.ImpactError, match="NaN is not a trade"):
        impact.rebalance_costs(
            trades,
            half_spread=ok,
            daily_volatility=ok,
            adv_over_nav=ok,
            prefactor=Y_PATIENT,
            exponent=E,
        )
    trades = _panel({"A": [0.04, 0.04], "B": [0.0, 0.0]}, DATES)
    with pytest.raises(impact.ImpactError, match="no column for \\['B'\\]"):
        impact.rebalance_costs(
            trades,
            half_spread=ok,
            daily_volatility=ok,
            adv_over_nav=ok,
            prefactor=Y_PATIENT,
            exponent=E,
        )


def test_there_is_no_separate_no_trade_band() -> None:
    """SPEC.md 7.1: the L1 term is the no-trade region. Nothing else may add one."""
    import inspect

    source = inspect.getsource(impact)
    for phrase in ("no_trade_band", "trade_band", "min_trade", "threshold"):
        assert phrase not in source.replace("no-trade band", ""), phrase
