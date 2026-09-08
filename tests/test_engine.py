"""The engine's accounting, every value by hand. SPEC.md 7.4, W5-P3.

The two-asset fixture has prices chosen so that every unit count, trade, cost
and NAV below is exact decimal arithmetic done on paper, not pasted back from
the code. The impact fixture reuses ``tests/test_impact.py``'s hand values, so
the engine is checked against numbers that were already checked by hand.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from mafrm.backtest import engine
from mafrm.costs import impact

EXACT = dict(rel=1e-12, abs=0.0)
E = 1.5


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-04", periods=n, name="date")


def two_asset_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = _dates(5)
    prices = pd.DataFrame(
        {"a": [100.0, 105.0, 110.0, 120.0, 120.0], "b": [50.0, 50.0, 55.0, 50.0, 60.0]},
        index=dates,
    )
    targets = pd.DataFrame({"a": [0.6, 0.6], "b": [0.4, 0.4]}, index=dates[[0, 3]])
    return prices, targets


def proportional(targets: pd.DataFrame, prices: pd.DataFrame, a: float) -> engine.CostModel:
    return engine.CostModel.proportional(a, index=targets.index, columns=prices.columns, exponent=E)


# ---------------------------------------------------------------------------
# The hand-computed path
# ---------------------------------------------------------------------------


def test_hand_computed_path_with_a_flat_half_spread() -> None:
    prices, targets = two_asset_fixture()
    res = engine.run_backtest(
        prices, targets, cost_model=proportional(targets, prices, 0.0005), initial_nav=1000.0
    )
    # Day 1, first rebalance on NAV 1000: buy 600 of a at 100 -> 6 units, 400 of b at
    # 50 -> 8 units. Spread cost 0.0005 * (600 + 400) = 0.5, debited from cash.
    assert res.units.iloc[0].tolist() == [6.0, 8.0]
    assert res.cash.iloc[0] == pytest.approx(-0.5, **EXACT)
    assert res.nav_before_costs.iloc[0] == 1000.0
    assert res.nav.iloc[0] == pytest.approx(999.5, **EXACT)
    # The target DOLLAR positions on NAV_pre are hit exactly.
    assert (res.units.iloc[0] * prices.iloc[0]).tolist() == [600.0, 400.0]
    # Drift, no trading: 6*105 + 8*50 - 0.5 ; 6*110 + 8*55 - 0.5.
    assert res.nav.iloc[1] == pytest.approx(1029.5, **EXACT)
    assert res.nav.iloc[2] == pytest.approx(1099.5, **EXACT)
    # Day 4, second rebalance. NAV_pre = 6*120 + 8*50 - 0.5 = 1119.5. Targets 671.7 / 447.8.
    # Trades -48.3 (sell a) and +47.8 (buy b); both sides cost: 0.0005 * 96.1 = 0.04805.
    assert res.nav_before_costs.iloc[1] == pytest.approx(1119.5, **EXACT)
    assert res.trade_dollars.iloc[1].tolist() == pytest.approx([-48.3, 47.8], **EXACT)
    assert res.trades.iloc[1].tolist() == pytest.approx([-48.3 / 1119.5, 47.8 / 1119.5], **EXACT)
    assert res.spread_cost.iloc[1].tolist() == pytest.approx([0.02415, 0.0239], **EXACT)
    assert res.total_cost.iloc[1] == pytest.approx(0.04805, **EXACT)
    assert res.turnover.iloc[1] == pytest.approx(96.1 / 1119.5, **EXACT)
    # Units: 6 - 48.3/120 = 5.5975 ; 8 + 47.8/50 = 8.956. Cash: -0.5 + 0.5 - 0.04805.
    assert res.units.iloc[3].tolist() == pytest.approx([5.5975, 8.956], **EXACT)
    # Cash is a difference of numbers of order 1000, so it carries their rounding:
    # 1e-11 absolute is 1e-14 of the quantities it was formed from.
    assert res.cash.iloc[3] == pytest.approx(-0.04805, abs=1e-11)
    assert res.nav.iloc[3] == pytest.approx(1119.45195, **EXACT)
    # Day 5: 5.5975*120 + 8.956*60 - 0.04805 = 671.7 + 537.36 - 0.04805.
    assert res.nav.iloc[4] == pytest.approx(1209.01195, **EXACT)
    assert res.final_nav == pytest.approx(1209.01195, **EXACT)
    assert res.total_return == pytest.approx(0.20901195, **EXACT)
    # Weights drift between rebalances and are units*P/NAV.
    assert res.weights.iloc[2]["a"] == pytest.approx(660.0 / 1099.5, **EXACT)
    assert res.impact_cost.to_numpy().tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_the_cost_is_charged_on_the_rebalance_date_and_no_other() -> None:
    """SPEC.md 7.4: at the correct time. NAV drops by the cost on the day the trade happens."""
    prices, targets = two_asset_fixture()
    free = engine.run_backtest(
        prices, targets, cost_model=proportional(targets, prices, 0.0), initial_nav=1000.0
    )
    charged = engine.run_backtest(
        prices, targets, cost_model=proportional(targets, prices, 0.0005), initial_nav=1000.0
    )
    gap = free.nav - charged.nav
    # Day 1 carries exactly the first cost; days 2-3 carry the same 0.5 (no new trade,
    # and cash earns nothing); day 4 adds the second cost.
    assert gap.iloc[0] == pytest.approx(0.5, **EXACT)
    assert gap.iloc[1] == pytest.approx(0.5, **EXACT)
    assert gap.iloc[2] == pytest.approx(0.5, **EXACT)
    # Day 4: the free book rebalances on a larger NAV (1120 vs 1119.5), so the two paths
    # differ by the cumulative cost plus the cost's own (zero, here: flat prices on day 5
    # for a) return. Checked as the cost itself: cost is 0.5 * 1120/1119.5 scaled trades.
    assert charged.total_cost.iloc[1] == pytest.approx(0.04805, **EXACT)
    assert free.total_cost.sum() == 0.0


def test_impact_term_reproduces_the_hand_values_of_test_impact() -> None:
    """Weights 4% and 1% on NAV 1000, a = 5bp, sigma = 2%/day, ADV = 250 dollars (V/v = 0.25)."""
    dates = _dates(2)
    prices = pd.DataFrame({"a": [100.0, 100.0], "b": [50.0, 50.0]}, index=dates)
    targets = pd.DataFrame({"a": [0.04], "b": [0.01]}, index=dates[[0]])
    for y, impact_a, impact_b in ((0.58, 1.856e-4, 2.32e-5), (1.40, 4.48e-4, 5.6e-5)):
        model = engine.CostModel(
            half_spread=pd.DataFrame(0.0005, index=targets.index, columns=prices.columns),
            daily_volatility=pd.DataFrame(0.02, index=targets.index, columns=prices.columns),
            adv_dollars=pd.DataFrame(250.0, index=targets.index, columns=prices.columns),
            prefactor=y,
            exponent=E,
        )
        res = engine.run_backtest(prices, targets, cost_model=model, initial_nav=1000.0)
        # Spread: 0.0005 * 40 = 0.02 and 0.0005 * 10 = 0.005 dollars.
        assert res.spread_cost.iloc[0].tolist() == pytest.approx([0.02, 0.005], **EXACT)
        # Impact: the test_impact proportions times NAV 1000.
        assert res.impact_cost.iloc[0].tolist() == pytest.approx(
            [impact_a * 1000.0, impact_b * 1000.0], **EXACT
        )
        assert res.asymmetry_cost.to_numpy().sum() == 0.0
        assert res.nav.iloc[0] == pytest.approx(
            1000.0 - 0.025 - (impact_a + impact_b) * 1000.0, **EXACT
        )
        # 4% and 1% invested, the rest is cash net of the cost.
        assert res.cash.iloc[0] == pytest.approx(950.0 - res.total_cost.iloc[0], **EXACT)


# ---------------------------------------------------------------------------
# SPEC.md 7.4's required properties
# ---------------------------------------------------------------------------


def test_zero_trade_costs_exactly_zero_and_reads_no_input() -> None:
    dates = _dates(3)
    prices = pd.DataFrame({"a": [100.0, 100.0, 100.0], "b": [50.0, 50.0, 50.0]}, index=dates)
    targets = pd.DataFrame({"a": [0.6, 0.6], "b": [0.4, 0.4]}, index=dates[[0, 2]])
    model = engine.CostModel(
        # NaN inputs on the second date: nothing trades there, so nothing may be read.
        half_spread=pd.DataFrame(
            [[0.0, 0.0], [np.nan, np.nan]], index=targets.index, columns=prices.columns
        ),
        daily_volatility=pd.DataFrame(
            [[0.0, 0.0], [np.nan, np.nan]], index=targets.index, columns=prices.columns
        ),
        adv_dollars=pd.DataFrame(
            [[1.0, 1.0], [np.nan, np.nan]], index=targets.index, columns=prices.columns
        ),
        prefactor=0.58,
        exponent=E,
    )
    res = engine.run_backtest(prices, targets, cost_model=model, initial_nav=1000.0)
    assert res.trades.iloc[1].tolist() == [0.0, 0.0]
    assert res.total_cost.iloc[1] == 0.0
    assert res.units.iloc[2].tolist() == [6.0, 8.0]
    assert res.nav.iloc[2] == 1000.0


def test_a_non_zero_trade_against_a_nan_input_raises_rather_than_returning_nan() -> None:
    prices, targets = two_asset_fixture()
    model = engine.CostModel(
        half_spread=pd.DataFrame(np.nan, index=targets.index, columns=prices.columns),
        daily_volatility=pd.DataFrame(0.0, index=targets.index, columns=prices.columns),
        adv_dollars=pd.DataFrame(1.0, index=targets.index, columns=prices.columns),
        prefactor=0.0,
        exponent=E,
    )
    with pytest.raises(impact.ImpactError, match="half_spread"):
        engine.run_backtest(prices, targets, cost_model=model, initial_nav=1000.0)


def test_round_trip_costs_twice_the_one_way_cost() -> None:
    """Buy everything, then sell everything at the same prices: 2 x a x notional, exactly."""
    dates = _dates(2)
    prices = pd.DataFrame({"a": [100.0, 100.0], "b": [50.0, 50.0]}, index=dates)
    targets = pd.DataFrame({"a": [0.6, 0.0], "b": [0.4, 0.0]}, index=dates)
    res = engine.run_backtest(
        prices, targets, cost_model=proportional(targets, prices, 0.0005), initial_nav=1000.0
    )
    # One way: 0.0005 * 1000 = 0.5. The sell leg unwinds exactly 1000 of notional.
    assert res.total_cost.iloc[0] == pytest.approx(0.5, **EXACT)
    assert res.total_cost.iloc[1] == pytest.approx(0.5, **EXACT)
    assert res.total_cost.sum() == pytest.approx(2 * res.total_cost.iloc[0], **EXACT)
    assert res.units.iloc[1].abs().max() < 1e-12
    assert res.nav.iloc[1] == pytest.approx(999.0, **EXACT)
    assert res.asymmetry_cost.to_numpy().sum() == 0.0


def test_both_sides_of_a_switch_are_charged() -> None:
    dates = _dates(2)
    prices = pd.DataFrame({"a": [100.0, 100.0], "b": [50.0, 50.0]}, index=dates)
    targets = pd.DataFrame({"a": [1.0, 0.0], "b": [0.0, 1.0]}, index=dates)
    res = engine.run_backtest(
        prices, targets, cost_model=proportional(targets, prices, 0.0005), initial_nav=1000.0
    )
    # Day 2: the a position is worth 1000 (10 units at 100) and cash is -0.5, so NAV_pre is
    # 999.5. Sell all 1000 of a and buy 999.5 of b: cost 0.0005 * 1999.5 = 0.99975, and
    # turnover 1999.5 / 999.5 -- both legs charged.
    assert res.trade_dollars.iloc[1].tolist() == pytest.approx([-1000.0, 999.5], **EXACT)
    assert res.total_cost.iloc[1] == pytest.approx(0.99975, **EXACT)
    assert res.turnover.iloc[1] == pytest.approx(1999.5 / 999.5, **EXACT)
    assert res.nav.iloc[1] == pytest.approx(999.5 - 0.99975, **EXACT)


def test_proportional_model_switches_impact_off_through_the_same_code_path() -> None:
    prices, targets = two_asset_fixture()
    model = proportional(targets, prices, 0.0005)
    assert model.prefactor == 0.0
    res = engine.run_backtest(prices, targets, cost_model=model, initial_nav=1000.0)
    assert (res.impact_cost.to_numpy() == 0.0).all()
    free = engine.CostModel.free(index=targets.index, columns=prices.columns, exponent=E)
    assert (free.half_spread.to_numpy() == 0.0).all()


def test_there_is_no_separate_no_trade_band() -> None:
    """SPEC.md 7.1: the L1 term is the no-trade region. The engine adds nothing on top."""
    source = inspect.getsource(engine).replace("no-trade band", "")
    for phrase in ("no_trade_band", "trade_band", "min_trade", "threshold"):
        assert phrase not in source, phrase


# ---------------------------------------------------------------------------
# Refusals -- nothing is filled, defaulted or silently reordered
# ---------------------------------------------------------------------------


def test_nan_price_is_refused_naming_the_date() -> None:
    prices, targets = two_asset_fixture()
    prices.iloc[2, 0] = np.nan
    with pytest.raises(engine.EngineError, match="2010-01-06"):
        engine.run_backtest(
            prices, targets, cost_model=proportional(targets, prices, 0.0), initial_nav=1.0
        )


def test_a_rebalance_on_a_non_trading_date_is_refused() -> None:
    prices, _ = two_asset_fixture()
    saturday = pd.DatetimeIndex([pd.Timestamp("2010-01-09")])
    targets = pd.DataFrame({"a": [0.6], "b": [0.4]}, index=saturday)
    with pytest.raises(engine.EngineError, match="not a trading date"):
        engine.run_backtest(
            prices, targets, cost_model=proportional(targets, prices, 0.0), initial_nav=1.0
        )


def test_nan_weights_and_mismatched_columns_are_refused() -> None:
    prices, targets = two_asset_fixture()
    bad = targets.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(engine.EngineError, match="NaN is not a weight"):
        engine.run_backtest(prices, bad, cost_model=proportional(bad, prices, 0.0), initial_nav=1.0)
    subset = targets[["a"]]
    with pytest.raises(engine.EngineError, match="exactly the price panel's assets"):
        engine.run_backtest(
            prices, subset, cost_model=proportional(targets, prices, 0.0), initial_nav=1.0
        )


def test_a_cost_input_missing_the_rebalance_date_is_refused_not_filled() -> None:
    prices, targets = two_asset_fixture()
    model = proportional(targets.iloc[[0]], prices, 0.0005)
    with pytest.raises(engine.EngineError, match="no row for the rebalance on 2010-01-07"):
        engine.run_backtest(prices, targets, cost_model=model, initial_nav=1000.0)


@pytest.mark.parametrize("nav", [0.0, -1.0, float("nan"), float("inf")])
def test_initial_nav_must_be_positive(nav: float) -> None:
    prices, targets = two_asset_fixture()
    with pytest.raises(engine.EngineError, match="initial_nav"):
        engine.run_backtest(
            prices, targets, cost_model=proportional(targets, prices, 0.0), initial_nav=nav
        )


def test_targets_are_reordered_to_the_price_columns_and_sorted_by_date() -> None:
    prices, targets = two_asset_fixture()
    shuffled = targets[["b", "a"]].iloc[::-1]
    res = engine.run_backtest(
        prices, shuffled, cost_model=proportional(shuffled, prices, 0.0005), initial_nav=1000.0
    )
    assert list(res.trades.columns) == ["a", "b"]
    assert res.nav.iloc[-1] == pytest.approx(1209.01195, **EXACT)


# ---------------------------------------------------------------------------
# The financing leg (operator ruling 1, W6-P3, SPEC.md 9.3)
# ---------------------------------------------------------------------------


def test_cash_accrues_at_the_daily_rate_before_the_days_rebalance_by_hand() -> None:
    prices, targets = two_asset_fixture()
    rate = pd.Series(0.01, index=prices.index)
    res = engine.run_backtest(
        prices,
        targets,
        cost_model=proportional(targets, prices, 0.0005),
        initial_nav=1000.0,
        cash_rate=rate,
    )
    # Day 0: no carried balance, so nothing accrues; the trade leaves cash at -0.5.
    assert res.cash.iloc[0] == pytest.approx(-0.5, **EXACT)
    # Day 1: -0.5 accrues one session at 1% -> -0.505; NAV = 6*105 + 8*50 - 0.505.
    assert res.cash.iloc[1] == pytest.approx(-0.5 * 1.01, **EXACT)
    assert res.nav.iloc[1] == pytest.approx(6 * 105 + 8 * 50 - 0.5 * 1.01, **EXACT)
    # Day 2: -0.505 -> -0.51005; NAV = 660 + 440 - 0.51005.
    cash2 = -0.5 * 1.01 * 1.01
    assert res.cash.iloc[2] == pytest.approx(cash2, **EXACT)
    assert res.nav.iloc[2] == pytest.approx(6 * 110 + 8 * 55 + cash2, **EXACT)
    # Day 3: the balance accrues FIRST (-0.5151505), then the rebalance is decided on
    # NAV_pre = 720 + 400 + cash3, spends exactly the cash on the net trade and
    # debits the cost, so cash after = -cost.
    cash3 = cash2 * 1.01
    pre = 6 * 120 + 8 * 50 + cash3
    assert res.nav_before_costs.iloc[1] == pytest.approx(pre, **EXACT)
    u_a = 0.6 * pre - 6 * 120
    u_b = 0.4 * pre - 8 * 50
    cost = 0.0005 * (abs(u_a) + abs(u_b))
    # Cash after a rebalance is the difference of two ~1000-dollar amounts landing
    # near 0.05, so it carries the engine's accounting-rounding bound
    # (c * eps * NAV ~ 4e-12 here; SPEC.md 7.4.2) rather than a relative 1e-12.
    assert res.cash.iloc[3] == pytest.approx(-cost, abs=1e-12)
    assert res.nav.iloc[3] == pytest.approx(pre - cost, **EXACT)
    # Day 4: -cost accrues once more; positions revalued at 120 / 60.
    units_a, units_b = 6 + u_a / 120, 8 + u_b / 50
    assert res.nav.iloc[4] == pytest.approx(units_a * 120 + units_b * 60 - cost * 1.01, **EXACT)


def test_a_zero_rate_reproduces_the_zero_interest_path_and_a_rate_only_touches_cash() -> None:
    prices, targets = two_asset_fixture()
    model = proportional(targets, prices, 0.0005)
    base = engine.run_backtest(prices, targets, cost_model=model, initial_nav=1000.0)
    zero = engine.run_backtest(
        prices,
        targets,
        cost_model=model,
        initial_nav=1000.0,
        cash_rate=pd.Series(0.0, index=prices.index),
    )
    assert zero.nav.tolist() == base.nav.tolist()
    assert zero.cash.tolist() == base.cash.tolist()
    # A cost-free, fully invested book holds NO cash, so the rate cannot reach it.
    free = engine.CostModel.free(index=targets.index, columns=prices.columns, exponent=E)
    a = engine.run_backtest(prices, targets, cost_model=free, initial_nav=1000.0)
    b = engine.run_backtest(
        prices,
        targets,
        cost_model=free,
        initial_nav=1000.0,
        cash_rate=pd.Series(0.05, index=prices.index),
    )
    assert (a.cash == 0.0).all()
    assert b.nav.tolist() == a.nav.tolist()


def test_the_excess_frame_at_zero_rate_is_the_nominal_frame_at_rf_deflated_by_the_bill() -> None:
    """Ruling 1's identity: a book financed at RF has excess return sum_i w_i (r_i - rf)."""
    prices, targets = two_asset_fixture()
    rf = pd.Series([0.0, 0.0004, 0.0003, 0.0005, 0.0002], index=prices.index)
    bill = (1.0 + rf).cumprod()
    nominal = prices.mul(bill, axis=0)  # nominal price = excess index x the bill index
    model = proportional(targets, prices, 0.0005)
    excess_frame = engine.run_backtest(prices, targets, cost_model=model, initial_nav=1000.0)
    nominal_frame = engine.run_backtest(
        nominal, targets, cost_model=model, initial_nav=1000.0, cash_rate=rf
    )
    deflated = nominal_frame.nav / bill
    assert deflated.to_numpy() == pytest.approx(excess_frame.nav.to_numpy(), rel=1e-12)
    assert (nominal_frame.cash / bill).to_numpy() == pytest.approx(
        excess_frame.cash.to_numpy(), rel=1e-9, abs=1e-12
    )
    # Not the same book at all when cash is NOT financed at RF: the negative cash
    # then costs nothing, so the nominal NAV is higher by the foregone accrual.
    unfinanced = engine.run_backtest(nominal, targets, cost_model=model, initial_nav=1000.0)
    assert unfinanced.nav.iloc[-1] > nominal_frame.nav.iloc[-1]


def test_financing_leg_is_the_carried_balance_times_the_rate_by_hand() -> None:
    prices, targets = two_asset_fixture()
    res = engine.run_backtest(
        prices, targets, cost_model=proportional(targets, prices, 0.0005), initial_nav=1000.0
    )
    leg = engine.financing_leg(res, pd.Series(0.01, index=prices.index))
    # Zero-interest cash path: -0.5 on days 0-2; the day-3 rebalance leaves -cost with
    # cost = 0.0005 * (|0.6*1119.5 - 720| + |0.4*1119.5 - 400|) = 0.0005 * 96.1.
    cost = 0.0005 * (abs(0.6 * 1119.5 - 720) + abs(0.4 * 1119.5 - 400))
    assert res.cash.iloc[3] == pytest.approx(-cost, abs=1e-12)  # the rounding bound, as above
    assert leg.iloc[0] == 0.0
    assert leg.iloc[1] == pytest.approx(-0.5 * 0.01, **EXACT)
    assert leg.iloc[2] == pytest.approx(-0.5 * 0.01, **EXACT)
    assert leg.iloc[3] == pytest.approx(-0.5 * 0.01, **EXACT)
    assert leg.iloc[4] == pytest.approx(-cost * 0.01, abs=1e-14)
    assert leg.name == "financing"


def test_a_cash_rate_missing_a_price_date_is_refused_not_filled() -> None:
    prices, targets = two_asset_fixture()
    short = pd.Series(0.01, index=prices.index[1:])
    with pytest.raises(engine.EngineError, match="cash_rate: no rate on 2010-01-04"):
        engine.run_backtest(
            prices,
            targets,
            cost_model=proportional(targets, prices, 0.0),
            initial_nav=1000.0,
            cash_rate=short,
        )
    res = engine.run_backtest(
        prices, targets, cost_model=proportional(targets, prices, 0.0), initial_nav=1000.0
    )
    with pytest.raises(engine.EngineError, match="cash_rate"):
        engine.financing_leg(res, short)
