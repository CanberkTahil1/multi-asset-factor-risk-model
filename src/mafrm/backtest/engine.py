"""The monthly-rebalance backtest engine. SPEC.md 7.4, 11; the accounting half of 10.1.

**No framework.** For thirteen assets rebalancing monthly with an explicit cost
model, a framework is negative leverage, and the cost model is the point of the
project: it has to be first-class and unit-testable, not a callback (operator
ruling, W5-P3). Yin et al. (2026, alphaXiv 2603.20319) ran identical strategy
logic through mainstream backtest engines and found divergence of up to 3.71%
for high-turnover rotation strategies, purely from implementation differences,
including a silent division error in a production cost model. That is why every
line here has a hand-computed test (``tests/test_engine.py``) and why the
engine is reconciled against ``bt`` (:mod:`mafrm.backtest.reconciliation`).

ACCOUNTING CONVENTIONS, stated once and tested
    - Positions are held in **units** and cash in dollars; ``NAV_t = cash_t +
      sum_i units_i P_it``. Valuation is recomputed from units every day, so a
      daily step carries no accumulated rounding -- only a rebalance changes
      state.
    - ``prices`` is a (date x asset) panel of closes or total-return indices.
      ``NaN`` is refused outright: a position cannot be valued on a date the
      asset did not trade, and filling would invent a session (CLAUDE.md
      failure mode 1; SPEC.md 11's calendar test).
    - A rebalance happens **at the close** of each target date. The trade is
      decided on ``NAV_pre = cash + units . P_t``, executed at ``P_t``, the
      target weights are hit exactly, and the cost is debited from cash on the
      same date (SPEC.md 7.4: costs on actual turnover, on both sides, at the
      correct time). Cash may therefore go negative by the cost -- or by
      leverage, if the targets sum past one. THE FINANCING LEG (operator
      ruling 1, W6-P3, SPEC.md 9.3; ``config/model.yaml`` ``backtest.financing``):
      cash earns and pays the risk-free rate both ways with no spread. The grid
      marks the book on the EXCESS-return index ``(1 + excess).cumprod()``, in
      which frame the accrual on cash is ``rf - rf = 0`` by identity -- a fully
      invested book financed at RF has excess return exactly ``sum_i w_i (r_i -
      rf)``, the definition the Sharpe already assumes -- so the grid passes no
      ``cash_rate`` and the engine carries cash at zero; ``tests/test_engine.py``
      pins that a NOMINAL run at ``cash_rate = rf``, deflated by the bill,
      reproduces the excess-frame run exactly. A caller marking a book on
      NOMINAL prices passes ``cash_rate``, a daily decimal series on the price
      calendar, and the balance carried from the previous close accrues ``cash
      * rate_t`` before that day's rebalance. :func:`financing_leg` reports,
      for any run, the dollars the cash path would have accrued at a given
      rate, so that "negligible" is a computed line in the report and not an
      assertion. This is a different convention from ``bt``'s, which absorbs
      the fee into the trade so that cash stays flat; the reconciliation
      measures that difference and names it rather than adopting either
      silently.
    - The cost is SPEC.md 7.1 through :func:`mafrm.costs.impact.trade_cost` on
      ``z = u / NAV_pre``, at **that date's** inputs, and a dollar cost is the
      proportion times ``NAV_pre``. There is no separate no-trade band (SPEC.md
      7.1): an asset whose target equals its drifted weight trades zero and
      costs exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mafrm.costs import impact

__all__ = ["BacktestResult", "CostModel", "EngineError", "financing_leg", "run_backtest"]

FloatArray = NDArray[np.float64]


class EngineError(ValueError):
    """The inputs cannot be backtested as given. Nothing is filled or defaulted."""


@dataclass(frozen=True)
class CostModel:
    """SPEC.md 7.1's per-date inputs on the rebalance calendar, plus one regime.

    Frames are (date x asset) in proportional units, except ``adv_dollars``,
    which is in dollars because the engine supplies NAV -- ``adv_over_nav =
    V_i / NAV_pre`` at each rebalance -- the same split as
    :class:`mafrm.costs.inputs.CostInputs`. A frame may carry more dates than the
    targets; it must carry every rebalance date and every asset. A ``NaN`` where
    a trade is non-zero raises through :func:`mafrm.costs.impact.trade_cost`
    naming the position; where the trade is zero nothing is read.
    """

    half_spread: pd.DataFrame
    daily_volatility: pd.DataFrame
    adv_dollars: pd.DataFrame
    #: SPEC.md 7.2's ``Y``. Zero switches the impact term off through the same code path.
    prefactor: float
    exponent: float
    #: SPEC.md 7.1's ``c_i``.
    asymmetry: float = 0.0

    @classmethod
    def proportional(
        cls,
        half_spread: float,
        *,
        index: pd.Index,
        columns: pd.Index,
        exponent: float,
        asymmetry: float = 0.0,
    ) -> CostModel:
        """A flat L1 cost only: ``a |z|`` with the impact prefactor ZERO.

        Same code path as the full model. With ``Y = 0`` the impact term is
        ``0 x finite`` and is exactly ``0.0``; volatility is set to zero and depth
        to one dollar so both are finite and the term never reads a number. This
        is the model the ``bt`` reconciliation runs, because ``bt`` models a
        proportional commission and nothing else.
        """
        if not np.isfinite(half_spread) or half_spread < 0.0:
            raise EngineError(f"proportional cost: half_spread must be >= 0, got {half_spread}")
        return cls(
            half_spread=pd.DataFrame(float(half_spread), index=index, columns=columns),
            daily_volatility=pd.DataFrame(0.0, index=index, columns=columns),
            adv_dollars=pd.DataFrame(1.0, index=index, columns=columns),
            prefactor=0.0,
            exponent=exponent,
            asymmetry=asymmetry,
        )

    @classmethod
    def free(cls, *, index: pd.Index, columns: pd.Index, exponent: float) -> CostModel:
        """No cost at all -- the paper portfolio (SPEC.md 9, cost variant A)."""
        return cls.proportional(0.0, index=index, columns=columns, exponent=exponent)


@dataclass(frozen=True)
class BacktestResult:
    """Everything the run produced, at daily and at rebalance resolution.

    Daily: ``nav``, ``cash``, ``units``, ``weights`` (``units * P / NAV``, so
    they drift between rebalances). Per rebalance: ``nav_before_costs``, the
    signed ``trades`` as ``z = u / NAV_pre``, ``trade_dollars``, and the three
    SPEC.md 7.1 terms in **dollars**, each (rebalance date x asset).
    """

    nav: pd.Series
    cash: pd.Series
    units: pd.DataFrame
    weights: pd.DataFrame
    nav_before_costs: pd.Series
    trades: pd.DataFrame
    trade_dollars: pd.DataFrame
    spread_cost: pd.DataFrame
    impact_cost: pd.DataFrame
    asymmetry_cost: pd.DataFrame
    initial_nav: float

    @property
    def rebalance_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.trades.index)

    @property
    def total_cost(self) -> pd.Series:
        """Dollar cost per rebalance, all three terms, all assets."""
        return (self.spread_cost + self.impact_cost + self.asymmetry_cost).sum(axis=1)

    @property
    def turnover(self) -> pd.Series:
        """Two-way turnover per rebalance, ``sum_i |z_i|``, as a fraction of ``NAV_pre``."""
        return self.trades.abs().sum(axis=1)

    @property
    def final_nav(self) -> float:
        return float(self.nav.iloc[-1])

    @property
    def total_return(self) -> float:
        return self.final_nav / self.initial_nav - 1.0


def _check_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise EngineError("prices: expected a non-empty (date x asset) DataFrame")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise EngineError("prices: the index must be a DatetimeIndex of trading dates")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise EngineError("prices: dates must be strictly increasing and unique")
    if prices.columns.duplicated().any():
        raise EngineError(f"prices: duplicate asset columns {list(prices.columns)}")
    values = prices.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        bad = prices.index[~np.isfinite(values).all(axis=1)][0]
        raise EngineError(
            f"prices: NaN or non-finite on {bad.date()}. A position cannot be valued on a date "
            "the asset did not trade, and filling would invent a session (CLAUDE.md failure "
            "mode 1). Align the panel on its intersection calendar first."
        )
    if (values <= 0.0).any():
        raise EngineError("prices: every price must be positive")
    return prices.astype("float64")


def _check_targets(targets: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(targets, pd.DataFrame) or targets.empty:
        raise EngineError("targets: expected a non-empty (rebalance date x asset) DataFrame")
    if not targets.index.is_unique:
        raise EngineError("targets: one row per rebalance date; the index has duplicates")
    missing_dates = targets.index.difference(prices.index)
    if len(missing_dates) > 0:
        first = pd.Timestamp(missing_dates[0])
        raise EngineError(
            f"targets: {first.date()} is not a trading date in the price panel. A rebalance "
            "happens at a close that exists; use mafrm.data.calendar.month_end_dates."
        )
    if set(targets.columns) != set(prices.columns):
        raise EngineError(
            "targets: columns must be exactly the price panel's assets; write 0.0 for an "
            f"asset not held. Got {sorted(targets.columns)} against {sorted(prices.columns)}"
        )
    ordered = targets.reindex(columns=prices.columns).sort_index().astype("float64")
    if not np.isfinite(ordered.to_numpy()).all():
        raise EngineError("targets: NaN is not a weight; write 0.0 for an asset not held")
    return ordered


def _on_calendar(frame: pd.DataFrame, targets: pd.DataFrame, *, name: str) -> FloatArray:
    """The cost input on the rebalance calendar, as an array. Missing labels raise."""
    if not isinstance(frame, pd.DataFrame):
        raise EngineError(f"cost_model.{name}: expected a (date x asset) DataFrame")
    absent_dates = targets.index.difference(frame.index)
    if len(absent_dates) > 0:
        first = pd.Timestamp(absent_dates[0])
        raise EngineError(
            f"cost_model.{name}: no row for the rebalance on {first.date()}. Every rebalance "
            "reads that date's own inputs; nothing is filled from another date (SPEC.md 7.4)."
        )
    absent_assets = targets.columns.difference(frame.columns)
    if len(absent_assets) > 0:
        raise EngineError(f"cost_model.{name}: no column for {list(absent_assets)}")
    return frame.reindex(index=targets.index, columns=targets.columns).to_numpy(dtype="float64")


def _check_cash_rate(cash_rate: pd.Series | None, dates: pd.Index) -> FloatArray:
    """The daily decimal rate on the price calendar; zero when none is given."""
    if cash_rate is None:
        return np.zeros(len(dates), dtype="float64")
    if not isinstance(cash_rate, pd.Series):
        raise EngineError("cash_rate: expected a Series of daily decimal rates on the price dates")
    missing = pd.DatetimeIndex(dates).difference(pd.DatetimeIndex(cash_rate.index))
    if len(missing) > 0:
        raise EngineError(
            f"cash_rate: no rate on {pd.Timestamp(missing[0]).date()}; every price date needs "
            "one and nothing is filled"
        )
    values = cash_rate.reindex(dates).to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise EngineError("cash_rate: a rate is not finite")
    return values


def run_backtest(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    cost_model: CostModel,
    initial_nav: float,
    cash_rate: pd.Series | None = None,
) -> BacktestResult:
    """Run the book through ``prices``, rebalancing to ``targets`` on their dates.

    ``targets`` is (rebalance date x asset) of weights; every date must be a
    trading date in ``prices``. Weights need not sum to one -- the remainder is
    cash -- and are applied exactly. ``cash_rate``, when given, is the daily
    decimal rate the cash balance carried from the previous close accrues at
    (both ways, no spread); see the module docstring for when it is zero by
    identity. See :class:`CostModel` for what is charged.
    """
    price_frame = _check_prices(prices)
    weight_frame = _check_targets(targets, price_frame)
    if not np.isfinite(initial_nav) or initial_nav <= 0.0:
        raise EngineError(f"initial_nav must be a positive number, got {initial_nav}")
    rate = _check_cash_rate(cash_rate, price_frame.index)

    a = _on_calendar(cost_model.half_spread, weight_frame, name="half_spread")
    sigma = _on_calendar(cost_model.daily_volatility, weight_frame, name="daily_volatility")
    adv = _on_calendar(cost_model.adv_dollars, weight_frame, name="adv_dollars")

    dates = price_frame.index
    assets = price_frame.columns
    prices_arr = price_frame.to_numpy(dtype="float64")
    weights_arr = weight_frame.to_numpy(dtype="float64")
    n_dates, n_assets = prices_arr.shape
    n_rebalances = len(weight_frame)

    rebalance_at = dates.get_indexer(weight_frame.index)
    is_rebalance = np.zeros(n_dates, dtype=bool)
    is_rebalance[rebalance_at] = True

    units = np.zeros(n_assets, dtype="float64")
    cash = float(initial_nav)
    nav_path = np.empty(n_dates, dtype="float64")
    cash_path = np.empty(n_dates, dtype="float64")
    units_path = np.empty((n_dates, n_assets), dtype="float64")
    nav_pre = np.empty(n_rebalances, dtype="float64")
    trades = np.zeros((n_rebalances, n_assets), dtype="float64")
    trade_dollars = np.zeros((n_rebalances, n_assets), dtype="float64")
    spread = np.zeros((n_rebalances, n_assets), dtype="float64")
    impact_cost = np.zeros((n_rebalances, n_assets), dtype="float64")
    asymmetry = np.zeros((n_rebalances, n_assets), dtype="float64")

    r = 0
    for k in range(n_dates):
        p = prices_arr[k]
        if k > 0:
            # The balance carried from the previous close accrues one session at
            # the cash rate, before anything is traded at this close.
            cash = cash * (1.0 + rate[k])
        if is_rebalance[k]:
            pre = cash + float(units @ p)
            if not pre > 0.0:
                raise EngineError(
                    f"NAV is {pre:.6g} at {dates[k].date()}: there is nothing to rebalance"
                )
            # The trade: to the target dollars, from the drifted position, decided
            # on NAV_pre and executed at this close.
            u = weights_arr[r] * pre - units * p
            z = u / pre
            terms = impact.trade_cost(
                z,
                half_spread=a[r],
                daily_volatility=sigma[r],
                adv_over_nav=adv[r] / pre,
                prefactor=cost_model.prefactor,
                exponent=cost_model.exponent,
                asymmetry=cost_model.asymmetry,
            )
            units = units + u / p
            cash = cash - float(u.sum()) - float(terms.total.sum()) * pre

            nav_pre[r] = pre
            trades[r] = z
            trade_dollars[r] = u
            spread[r] = terms.spread * pre
            impact_cost[r] = terms.impact * pre
            asymmetry[r] = terms.asymmetry * pre
            r += 1

        nav_path[k] = cash + float(units @ p)
        cash_path[k] = cash
        units_path[k] = units

    if (nav_path <= 0.0).any():
        first = dates[int(np.argmax(nav_path <= 0.0))]
        raise EngineError(f"NAV is not positive on {first.date()}; the book is bankrupt")

    def per_rebalance(values: FloatArray) -> pd.DataFrame:
        return pd.DataFrame(values, index=weight_frame.index, columns=assets)

    daily_units = pd.DataFrame(units_path, index=dates, columns=assets)
    nav = pd.Series(nav_path, index=dates, name="nav")
    return BacktestResult(
        nav=nav,
        cash=pd.Series(cash_path, index=dates, name="cash"),
        units=daily_units,
        weights=daily_units.mul(price_frame).div(nav, axis=0),
        nav_before_costs=pd.Series(nav_pre, index=weight_frame.index, name="nav_before_costs"),
        trades=per_rebalance(trades),
        trade_dollars=per_rebalance(trade_dollars),
        spread_cost=per_rebalance(spread),
        impact_cost=per_rebalance(impact_cost),
        asymmetry_cost=per_rebalance(asymmetry),
        initial_nav=float(initial_nav),
    )


def financing_leg(result: BacktestResult, daily_rate: pd.Series) -> pd.Series:
    """The dollars the cash path WOULD accrue at ``daily_rate``: ``cash_{t-1} * rate_t``.

    Operator ruling 1 (W6-P3, SPEC.md 9.3). For a run whose price panel is the
    excess-return index the engine carried cash at zero because the accrual is
    ``rf - rf`` in that frame; this reports the nominal size of the leg so the
    report can print it per cell beside the cost drag. Positive when cash is
    long and the rate positive; negative when the book is short cash (the cost
    paid out of NAV). Zero on the first date, which has no carried balance.
    """
    rate = _check_cash_rate(daily_rate, pd.DatetimeIndex(result.cash.index))
    carried = np.concatenate([[0.0], result.cash.to_numpy(dtype="float64")[:-1]])
    return pd.Series(carried * rate, index=result.cash.index, name="financing")
