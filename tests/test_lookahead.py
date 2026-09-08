"""SPEC.md 11's mechanical no-look-ahead test, on three states at t. W5-P3, experiments.md row 204.

The harness is generic (operator ruling 4). Here it is pointed at the engine's
NAV, the covariance forecast through SPEC.md 5.2's repair, and the estimated
cost inputs -- each at >= ``backtest.no_lookahead.minimum_evaluation_dates``
dates on synthetic panels -- plus two controls that prove the harness can fail.

The level components of ``a_i(t)`` -- the issuer level and the in-sample median
baseline of SPEC.md 7.1.2 -- are full-sample constants by ruling and are NOT
tested here; experiments.md row 204 records the exclusion and its reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mafrm import config
from mafrm.backtest import engine, lookahead
from mafrm.costs import adv, spread
from mafrm.data import calendar
from mafrm.data import prices as prices_mod
from mafrm.risk import synthetic
from mafrm.risk.config import RiskConfig
from mafrm.risk.covariance import run_pipeline

CFG = config.load()
MINIMUM = CFG.model.backtest.no_lookahead.minimum_evaluation_dates


def _rng(offset: int) -> np.random.Generator:
    return np.random.default_rng(CFG.seed + offset)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2012-01-02", periods=n, name="date")


def _price_panel(n_dates: int, n_assets: int, rng: np.random.Generator) -> pd.DataFrame:
    shocks = rng.standard_normal((n_dates, n_assets)) * 0.01
    levels = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    return pd.DataFrame(levels, index=_dates(n_dates), columns=[f"x{i}" for i in range(n_assets)])


# ---------------------------------------------------------------------------
# The three states
# ---------------------------------------------------------------------------


def test_engine_nav_at_t_is_causal() -> None:
    prices = _price_panel(420, 3, _rng(1))
    all_targets = pd.DataFrame(
        1.0 / 3.0,
        index=calendar.month_end_dates(pd.DatetimeIndex(prices.index)),
        columns=prices.columns,
    )
    exponent = CFG.model.costs.total_cost_exponent

    def nav_at(data: pd.DataFrame, t: pd.Timestamp) -> float:
        through = data.loc[:t]
        targets = all_targets.loc[:t]
        model = engine.CostModel.proportional(
            2.5e-4, index=targets.index, columns=through.columns, exponent=exponent
        )
        return engine.run_backtest(through, targets, cost_model=model, initial_nav=1.0e6).nav.loc[t]

    rng = _rng(2)
    times = lookahead.evaluation_dates(
        pd.DatetimeIndex(prices.index), count=MINIMUM, rng=rng, minimum_prefix=25
    )
    report = lookahead.assert_no_lookahead(
        nav_at,
        prices,
        times=times,
        minimum_evaluations=MINIMUM,
        rng=rng,
        perturb=lookahead.perturb_multiplicative,
    )
    assert report.count >= MINIMUM and report.sensitive


def test_covariance_forecast_at_t_is_causal() -> None:
    """SPEC.md 5.1-5.2 on data through t, bitwise invariant to anything after t."""
    spec = synthetic.specification(3, observations=500, config=CFG)
    returns = synthetic.panel(spec)
    returns.index = _dates(len(returns))
    risk = RiskConfig.load(horizon="short", config=CFG)

    def forecast_at(data: pd.DataFrame, t: pd.Timestamp) -> np.ndarray:
        return run_pipeline(data.loc[:t], risk, stop_after="psd_repair").matrix

    rng = _rng(3)
    times = lookahead.evaluation_dates(
        pd.DatetimeIndex(returns.index), count=MINIMUM, rng=rng, minimum_prefix=60
    )
    report = lookahead.assert_no_lookahead(
        forecast_at,
        returns,
        times=times,
        minimum_evaluations=MINIMUM,
        rng=rng,
        perturb=lookahead.perturb_additive,
    )
    assert report.count >= MINIMUM and report.sensitive


def _bars(n_dates: int, rng: np.random.Generator) -> pd.DataFrame:
    close = 50.0 * np.exp(np.cumsum(rng.standard_normal(n_dates) * 0.01))
    spread_half = 0.001
    high = close * (1.0 + np.abs(rng.standard_normal(n_dates)) * 0.005 + spread_half)
    low = close * (1.0 - np.abs(rng.standard_normal(n_dates)) * 0.005 - spread_half)
    open_ = np.clip(close * (1.0 + rng.standard_normal(n_dates) * 0.003), low, high)
    volume = rng.integers(500_000, 2_000_000, size=n_dates).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=_dates(n_dates),
    )


def test_estimated_cost_inputs_at_t_are_causal() -> None:
    """sigma_t, V_t and the EDGE window ending at t, each on data through t."""
    costs = CFG.model.costs
    bars = _bars(costs.volatility_window + 120, _rng(4))
    actions = pd.DataFrame(index=bars.index)

    def inputs_at(data: pd.DataFrame, t: pd.Timestamp) -> np.ndarray:
        through = data.loc[:t]
        returns = prices_mod.total_return(through, actions.loc[:t])
        sigma = returns.rolling(costs.volatility_window, min_periods=costs.volatility_window).std()
        depth = adv.average_daily_volume(through, window=costs.adv.window)
        edge = spread.effective_spread(
            through,
            window=costs.edge_spread.window,
            min_periods=costs.edge_spread.min_periods,
            signed=costs.edge_spread.signed_estimates,
        )
        return np.array([sigma.loc[t], depth.loc[t], edge.loc[t]])

    rng = _rng(5)
    times = lookahead.evaluation_dates(
        pd.DatetimeIndex(bars.index),
        count=MINIMUM,
        rng=rng,
        minimum_prefix=costs.volatility_window + 2,
    )

    def perturb_bars(rows: pd.DataFrame, gen: np.random.Generator) -> pd.DataFrame:
        # Positive, and High >= Close >= Low kept, so the EDGE estimator sees valid bars.
        factors = np.exp(gen.standard_normal(len(rows)) * 0.05)
        out = rows.copy()
        for column in ("Open", "High", "Low", "Close"):
            out[column] = rows[column].to_numpy() * factors
        out["Volume"] = rows["Volume"].to_numpy() * np.exp(gen.standard_normal(len(rows)))
        return out

    report = lookahead.assert_no_lookahead(
        inputs_at,
        bars,
        times=times,
        minimum_evaluations=MINIMUM,
        rng=rng,
        perturb=perturb_bars,
    )
    assert report.count >= MINIMUM and report.sensitive
    # And the states were finite, i.e. the windows were full at every evaluated t.
    assert np.isfinite(inputs_at(bars, times[0])).all()


# ---------------------------------------------------------------------------
# Controls: the harness must be able to fail
# ---------------------------------------------------------------------------


def test_a_state_that_reads_the_next_row_is_caught() -> None:
    prices = _price_panel(200, 2, _rng(6))

    def leaks(data: pd.DataFrame, t: pd.Timestamp) -> float:
        position = data.index.get_loc(t)
        return float(data.iloc[position + 1, 0])

    rng = _rng(7)
    times = lookahead.evaluation_dates(
        pd.DatetimeIndex(prices.index), count=MINIMUM, rng=rng, minimum_prefix=5
    )
    with pytest.raises(lookahead.LookaheadError, match="LOOK-AHEAD"):
        lookahead.assert_no_lookahead(
            leaks,
            prices,
            times=times,
            minimum_evaluations=MINIMUM,
            rng=rng,
            perturb=lookahead.perturb_multiplicative,
        )


def test_a_constant_state_is_reported_vacuous_not_passed() -> None:
    """Fixed target weights are exactly this case; the harness refuses to call it a pass."""
    prices = _price_panel(200, 2, _rng(8))
    weights = np.array([0.6, 0.4])

    def fixed_weights(data: pd.DataFrame, t: pd.Timestamp) -> np.ndarray:
        return weights

    rng = _rng(9)
    times = lookahead.evaluation_dates(
        pd.DatetimeIndex(prices.index), count=MINIMUM, rng=rng, minimum_prefix=5
    )
    with pytest.raises(lookahead.LookaheadError, match="VACUOUS"):
        lookahead.assert_no_lookahead(
            fixed_weights,
            prices,
            times=times,
            minimum_evaluations=MINIMUM,
            rng=rng,
            perturb=lookahead.perturb_multiplicative,
        )
    # With the power check switched off it passes -- which is exactly why it is on by default.
    report = lookahead.assert_no_lookahead(
        fixed_weights,
        prices,
        times=times,
        minimum_evaluations=MINIMUM,
        rng=rng,
        perturb=lookahead.perturb_multiplicative,
        require_sensitivity=False,
    )
    assert report.count >= MINIMUM


def test_fewer_dates_than_the_spec_minimum_is_refused() -> None:
    prices = _price_panel(200, 2, _rng(10))
    rng = _rng(11)
    times = lookahead.evaluation_dates(
        pd.DatetimeIndex(prices.index), count=MINIMUM - 1, rng=rng, minimum_prefix=5
    )
    with pytest.raises(lookahead.LookaheadError, match=r"SPEC\.md 11"):
        lookahead.assert_no_lookahead(
            lambda d, t: d.loc[t].to_numpy(),
            prices,
            times=times,
            minimum_evaluations=MINIMUM,
            rng=rng,
            perturb=lookahead.perturb_multiplicative,
        )


def test_evaluation_dates_keep_the_stated_prefix_and_suffix() -> None:
    index = _dates(100)
    rng = _rng(12)
    times = lookahead.evaluation_dates(
        index, count=10, rng=rng, minimum_prefix=30, minimum_suffix=20
    )
    assert len(set(times)) == 10
    assert all(index[30] <= t <= index[79] for t in times)
    with pytest.raises(lookahead.LookaheadError, match="interior positions"):
        lookahead.evaluation_dates(index, count=60, rng=rng, minimum_prefix=30, minimum_suffix=20)


def test_perturbations_move_every_value_and_keep_prices_positive() -> None:
    frame = _price_panel(50, 3, _rng(13))
    rng = _rng(14)
    mult = lookahead.perturb_multiplicative(frame, rng)
    add = lookahead.perturb_additive(frame, rng)
    assert (mult.to_numpy() > 0.0).all()
    assert (mult.to_numpy() != frame.to_numpy()).all()
    assert (add.to_numpy() != frame.to_numpy()).all()
