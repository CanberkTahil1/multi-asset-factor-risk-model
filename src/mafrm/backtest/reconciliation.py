"""SPEC.md 7.4's engine cross-check against ``bt``. W5-P3.

    *"Reconcile your engine against `bt` on a trivial strategy and commit the
    comparison table."*

WHAT IS RECONCILED, AND WHAT IS NOT -- read this before the table
    ``bt`` models a proportional commission and nothing else, so it can
    reconcile the **accounting** (units, cash, valuation, drift, rebalancing
    on given dates) and the **spread term** of SPEC.md 7.1 as a proportional
    cost. It cannot reconcile the **impact term**. The cross-check is complete
    in two halves: impact was reconciled against the SPEC.md 7.1 formula by
    W5-P1's hand-computed tests (``tests/test_impact.py``), and ``bt`` covers
    what it models (operator ruling 3). "Reconciled against bt" therefore never
    means the whole cost function, and ``reports/engine_reconciliation.md``
    says so in its first section.

THE THREE CONVENTIONS HELD FIXED, so that accounting is what gets tested
    - **Dates.** Both engines rebalance on :func:`mafrm.data.calendar.month_end_dates`
      of the same panel; ``bt`` is FED those dates (``RunOnDate`` + ``WeighTarget``)
      and never resamples its own month ends (operator ruling 2; CLAUDE.md
      failure mode 1).
    - **Prices.** Both engines see the same total-return index, built once by
      :func:`mafrm.data.prices.total_return`. Dividend accounting is therefore
      the project's (tested in W1) and is not what ``bt`` checks.
    - **Fee financing.** The engine hits the target weights and debits the cost
      from cash. ``bt``'s native commission is absorbed into the trade, so the
      target is missed by the fee and cash stays flat. Both are run: the
      cash-financed row gives ``bt`` the engine's convention through a two-line
      algo that debits the fee after ``Rebalance`` -- the same kind of alignment
      as feeding it the dates -- and the native row keeps the convention gap
      visible and ATTRIBUTES it, by replaying ``bt``'s fee-absorbing fixed point
      on the engine's own arithmetic (:func:`shadow_trade_financed`) and
      requiring that replay to agree with ``bt`` to the bound.

THE BOUND
    ``c * eps * T`` relative on NAV, ``c = 2 * (assets + accounting_roundings)``,
    from ``config/model.yaml`` ``backtest.reconciliation`` (operator ruling 1).
    A discrepancy above it is attributed to a named cause or it is a bug in one
    of the two engines. It is never widened.

Yin et al. (2026, alphaXiv 2603.20319) is the reason this file exists: identical
strategy logic through mainstream engines diverged by up to 3.71% for
high-turnover rotation strategies, from implementation differences alone.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from mafrm import config as config_mod
from mafrm.backtest import engine
from mafrm.data import cache, calendar
from mafrm.data import prices as prices_mod

__all__ = [
    "Comparison",
    "Reconciliation",
    "constant_targets",
    "real_panel",
    "reconcile_panel",
    "reconciliation",
    "run_bt",
    "shadow_trade_financed",
    "synthetic_panel",
]

FeeFinancing = Literal["cash", "trade"]

#: The synthetic fixture's shape. Two assets, three years of business days from
#: a date that overlaps nothing in the project, daily volatilities of 1% and
#: 0.5% and a small positive drift so the book both drifts and compounds. These
#: describe a FIXTURE, not a model -- the same footing as
#: :mod:`mafrm.risk.synthetic`'s loading constants -- and the reconciliation's
#: verdict is invariant to them by construction (it is a bound in ulps).
_SYNTHETIC_START = "2000-01-03"
_SYNTHETIC_YEARS = 3
_SYNTHETIC_DAILY_VOLATILITY = (0.01, 0.005)
_SYNTHETIC_DAILY_DRIFT = 0.0002
_SYNTHETIC_NAMES = ("synthetic_a", "synthetic_b")
_SYNTHETIC_BASE = 100.0

_BPS = 1e-4


def _bt() -> Any:
    """Import the comparand lazily: it is a dev extra and the engine never needs it."""
    try:
        return importlib.import_module("bt")
    except ImportError as exc:  # pragma: no cover - environment, not logic
        raise RuntimeError(
            "bt is not installed. It is the SPEC.md 7.4 reconciliation comparand and a dev "
            "extra: `uv sync --extra dev`."
        ) from exc


# ---------------------------------------------------------------------------
# Panels and targets
# ---------------------------------------------------------------------------


def synthetic_panel(cfg: config_mod.Config | None = None) -> pd.DataFrame:
    """Two seeded geometric random walks on business days. Byte-reproducible from ``seed``."""
    cfg = cfg or config_mod.load()
    days = _SYNTHETIC_YEARS * cfg.model.data.trading_days_per_year
    index = pd.bdate_range(_SYNTHETIC_START, periods=days, name="date")
    rng = np.random.default_rng(cfg.seed)
    shocks = rng.standard_normal((days, len(_SYNTHETIC_NAMES)))
    log_returns = _SYNTHETIC_DAILY_DRIFT + shocks * np.asarray(_SYNTHETIC_DAILY_VOLATILITY)
    levels = _SYNTHETIC_BASE * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(levels, index=index, columns=list(_SYNTHETIC_NAMES))


def _cost_assets(cfg: config_mod.Config) -> dict[str, str]:
    out: dict[str, str] = {}
    for asset in cfg.universe.assets:
        try:
            out[asset.id] = cfg.cost_ticker(asset.id)
        except config_mod.ConfigError:
            continue
    return out


def real_panel(cfg: config_mod.Config | None = None) -> pd.DataFrame:
    """Total-return indices for every cost-bearing asset through its cost ticker.

    Intersection calendar from ``sample.start``, cut at the first month end so
    that day one is a rebalance, and stopping strictly before the holdout: every
    read goes through :func:`mafrm.data.cache.read`, which truncates there, and
    :func:`mafrm.data.calendar.align` is passed the boundary again.
    """
    cfg = cfg or config_mod.load()
    holdout = cfg.require_holdout_start()
    manifest = cache.Manifest.load()
    returns: dict[str, pd.Series] = {}
    for asset_id, ticker in _cost_assets(cfg).items():
        key = ticker.lower()
        bars = cache.read(
            manifest.latest(source="yfinance", name=f"{key}_prices"), manifest=manifest
        )
        actions = cache.read(
            manifest.latest(source="yfinance", name=f"{key}_actions"), manifest=manifest
        )
        returns[asset_id] = prices_mod.total_return(bars, actions)
    aligned = calendar.align(returns, how="intersection", start=cfg.model.sample.start, end=holdout)
    index = _SYNTHETIC_BASE * (1.0 + aligned).cumprod()
    first_month_end = calendar.month_end_dates(pd.DatetimeIndex(index.index))[0]
    return index.loc[first_month_end:]


def constant_targets(
    prices: pd.DataFrame, weights: Sequence[float] | Literal["equal_weight"]
) -> pd.DataFrame:
    """The trivial strategy: the same weights on every month end of ``prices``."""
    dates = calendar.month_end_dates(pd.DatetimeIndex(prices.index))
    n = len(prices.columns)
    if weights == "equal_weight":
        row = np.full(n, 1.0 / n)
    else:
        row = np.asarray(list(weights), dtype="float64")
        if row.shape != (n,):
            raise engine.EngineError(
                f"constant_targets: {len(row)} weights for {n} assets {list(prices.columns)}"
            )
    return pd.DataFrame(np.tile(row, (len(dates), 1)), index=dates, columns=prices.columns)


# ---------------------------------------------------------------------------
# The three books: engine, bt, and the attribution replay
# ---------------------------------------------------------------------------


def run_engine(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    half_spread: float,
    initial_nav: float,
    exponent: float,
) -> engine.BacktestResult:
    model = engine.CostModel.proportional(
        half_spread, index=targets.index, columns=targets.columns, exponent=exponent
    )
    return engine.run_backtest(prices, targets, cost_model=model, initial_nav=initial_nav)


def run_bt(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    half_spread: float,
    initial_nav: float,
    fee_financing: FeeFinancing,
) -> pd.Series:
    """``bt``'s NAV on ``prices.index``, fed the same dates and the same weights.

    ``fee_financing="trade"`` is ``bt``'s native commission, ``|q| * p * a``,
    absorbed into each trade by ``SecurityBase.allocate``'s fixed point.
    ``"cash"`` gives ``bt`` the engine's convention: zero commission inside the
    allocation, then an algo that debits ``a * |outlay|`` from the strategy's
    cash on the same date (``adjust(..., flow=False)``, so it counts as a fee in
    ``bt``'s own return index too). ``integer_positions`` is off: the engine
    trades fractional units, and so must the comparand.
    """
    bt = _bt()
    rate = float(half_spread)
    weights = targets.astype("float64")
    dates = [pd.Timestamp(d) for d in weights.index]

    class DebitFeeFromCash(bt.Algo):  # type: ignore[misc, name-defined]
        """Charge ``rate * |outlay|`` against cash after the rebalance has executed."""

        def __call__(self, target: Any) -> bool:
            now = target.now
            notional = 0.0
            for child in target.children.values():
                outlays = child.outlays
                if now in outlays.index:
                    notional += abs(float(outlays.loc[now]))
            fee = rate * notional
            if fee != 0.0:
                target.adjust(-fee, update=True, flow=False, fee=fee)
            return True

    algos = [
        bt.algos.RunOnDate(*dates),
        bt.algos.WeighTarget(weights),
        bt.algos.Rebalance(),
    ]
    if fee_financing == "cash":
        algos.append(DebitFeeFromCash())
        commissions = None
    elif fee_financing == "trade":

        def commissions(q: float, p: float) -> float:
            return abs(q) * p * rate

    else:
        raise ValueError(f"fee_financing must be 'cash' or 'trade', got {fee_financing!r}")

    strategy = bt.Strategy("reconciliation", algos)
    backtest = bt.Backtest(
        strategy,
        prices.astype("float64"),
        initial_capital=float(initial_nav),
        commissions=commissions,
        integer_positions=False,
        progress_bar=False,
    )
    backtest.run()
    nav = backtest.strategy.values
    # bt prepends a virtual row one calendar day before the first date; drop it.
    out: pd.Series = nav.reindex(prices.index).astype("float64")
    out.name = "bt_nav"
    if out.isna().any():
        raise RuntimeError("bt returned no value on some panel dates; the comparison is void")
    return out


def shadow_trade_financed(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    half_spread: float,
    initial_nav: float,
    absolute_stop: float,
) -> pd.Series:
    """The ATTRIBUTION device: ``bt``'s fee-absorbing convention on the engine's arithmetic.

    Units and cash exactly as :func:`mafrm.backtest.engine.run_backtest` keeps
    them, but at each rebalance the trade is sized the way ``bt`` sizes it --
    ``amount = (w - v/NAV) * NAV``, then units reduced (buys) or increased
    (sells) until ``units*p + fee`` meets ``amount`` to ``bt``'s own stopping
    rule, ``numpy.isclose`` with ``absolute_stop`` -- so the fee comes out of
    the trade and cash stays flat. This is NOT a production path and it is not
    the engine: it exists so that the native-commission row's discrepancy can
    be shown to be entirely this convention, to the bound, rather than argued
    to be. Named cause: fee financing, and commission on the shares actually
    traded rather than on the intended notional.
    """
    rate = float(half_spread)
    p_all = prices.to_numpy(dtype="float64")
    w_all = targets.reindex(columns=prices.columns).to_numpy(dtype="float64")
    rebalance_at = set(prices.index.get_indexer(targets.index).tolist())
    n_assets = p_all.shape[1]
    units = np.zeros(n_assets)
    cash = float(initial_nav)
    nav = np.empty(len(prices))
    r = 0
    for k in range(len(prices)):
        p = p_all[k]
        if k in rebalance_at:
            base = cash + float(units @ p)
            for i in range(n_assets):
                value = units[i] * p[i]
                amount = (w_all[r, i] - value / base) * base
                if amount == 0.0:
                    continue
                if amount + value == 0.0:
                    q = -units[i]
                else:
                    q = amount / p[i]
                    full = q * p[i] + rate * abs(q) * p[i]
                    while not np.isclose(full, amount, rtol=0.0, atol=absolute_stop) and q != 0.0:
                        q = q - (full - amount) / p[i]
                        full = q * p[i] + rate * abs(q) * p[i]
                fee = rate * abs(q) * p[i]
                units[i] += q
                cash -= q * p[i] + fee
            r += 1
        nav[k] = cash + float(units @ p)
    return pd.Series(nav, index=prices.index, name="shadow_nav")


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    """One row of the table: two NAV paths on one calendar, and the bound."""

    label: str
    panel: str
    fee_financing: str
    half_spread: float
    reference: pd.Series
    comparand: pd.Series
    assets: int
    asset_names: tuple[str, ...]
    tolerance: float
    trading_days_per_year: int
    #: What the row is expected to do. ``"within"`` rows must pass; an
    #: ``"exceeds"`` row is the named convention gap and is expected to breach.
    expectation: Literal["within", "exceeds"]

    @property
    def steps(self) -> int:
        return len(self.reference)

    @property
    def gap(self) -> pd.Series:
        """``reference / comparand - 1`` at every date."""
        out: pd.Series = self.reference / self.comparand - 1.0
        return out

    @property
    def max_abs_gap(self) -> float:
        return float(self.gap.abs().max())

    @property
    def gap_over_tolerance(self) -> float:
        return self.max_abs_gap / self.tolerance

    @property
    def within_tolerance(self) -> bool:
        return self.max_abs_gap <= self.tolerance

    @property
    def as_expected(self) -> bool:
        return self.within_tolerance if self.expectation == "within" else not self.within_tolerance

    def _annualised(self, nav: pd.Series) -> float:
        years = (len(nav) - 1) / self.trading_days_per_year
        return float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0)

    @property
    def annualised_gap_bps(self) -> float:
        """Reference minus comparand annualised return, in basis points per year."""
        return (self._annualised(self.reference) - self._annualised(self.comparand)) / _BPS


@dataclass(frozen=True)
class Reconciliation:
    rows: tuple[Comparison, ...]
    initial_nav: float
    accounting_roundings: int

    def by_label(self, label: str) -> Comparison:
        for row in self.rows:
            if row.label == label:
                return row
        raise KeyError(label)

    @property
    def all_as_expected(self) -> bool:
        return all(row.as_expected for row in self.rows)


def reconcile_panel(
    prices: pd.DataFrame,
    weights: Sequence[float] | Literal["equal_weight"],
    *,
    panel: str,
    cfg: config_mod.Config | None = None,
) -> tuple[Comparison, ...]:
    """The rows for one panel: zero cost, cash-financed spread, native spread + shadow."""
    cfg = cfg or config_mod.load()
    rules = cfg.model.backtest.reconciliation
    costs = cfg.model.costs
    nav0 = rules.initial_nav
    half_spread = costs.flat_spread_assumption_bps * _BPS / 2.0
    exponent = costs.total_cost_exponent
    per_year = cfg.model.data.trading_days_per_year
    targets = constant_targets(prices, weights)
    n = len(prices.columns)
    tol = rules.tolerance(assets=n, steps=len(prices))

    def row(
        label: str,
        reference: pd.Series,
        comparand: pd.Series,
        *,
        financing: str,
        a: float,
        expectation: Literal["within", "exceeds"],
    ) -> Comparison:
        return Comparison(
            label=label,
            panel=panel,
            fee_financing=financing,
            half_spread=a,
            reference=reference,
            comparand=comparand,
            assets=n,
            asset_names=tuple(str(c) for c in prices.columns),
            tolerance=tol,
            trading_days_per_year=per_year,
            expectation=expectation,
        )

    free = run_engine(prices, targets, half_spread=0.0, initial_nav=nav0, exponent=exponent)
    bt_free = run_bt(prices, targets, half_spread=0.0, initial_nav=nav0, fee_financing="trade")
    charged = run_engine(
        prices, targets, half_spread=half_spread, initial_nav=nav0, exponent=exponent
    )
    bt_cash = run_bt(
        prices, targets, half_spread=half_spread, initial_nav=nav0, fee_financing="cash"
    )
    bt_native = run_bt(
        prices, targets, half_spread=half_spread, initial_nav=nav0, fee_financing="trade"
    )
    shadow = shadow_trade_financed(
        prices,
        targets,
        half_spread=half_spread,
        initial_nav=nav0,
        absolute_stop=_bt_absolute_stop(),
    )
    return (
        row(
            f"{panel}: zero cost, engine vs bt",
            free.nav,
            bt_free,
            financing="none",
            a=0.0,
            expectation="within",
        ),
        row(
            f"{panel}: spread, cash-financed, engine vs bt",
            charged.nav,
            bt_cash,
            financing="cash",
            a=half_spread,
            expectation="within",
        ),
        row(
            f"{panel}: spread, bt native commission vs engine",
            charged.nav,
            bt_native,
            financing="trade (bt) vs cash (engine)",
            a=half_spread,
            expectation="exceeds",
        ),
        row(
            f"{panel}: spread, shadow replay vs bt native",
            shadow,
            bt_native,
            financing="trade",
            a=half_spread,
            expectation="within",
        ),
    )


def _bt_absolute_stop() -> float:
    """``numpy.isclose``'s default ``atol`` -- the stopping rule inside ``bt``'s allocate.

    Read from numpy rather than written here, so it cannot drift from what bt
    actually does: ``SecurityBase.allocate`` calls ``np.isclose(full_outlay,
    amount, rtol=TOL)`` and leaves ``atol`` at its default.
    """
    import inspect

    default = inspect.signature(np.isclose).parameters["atol"].default
    return float(default)


def reconciliation(
    cfg: config_mod.Config | None = None, *, include_real: bool = True
) -> Reconciliation:
    """Synthetic first, then the real cache (operator ruling 2)."""
    cfg = cfg or config_mod.load()
    rules = cfg.model.backtest.reconciliation
    rows: list[Comparison] = list(
        reconcile_panel(
            synthetic_panel(cfg), rules.synthetic_weights, panel="synthetic 60/40", cfg=cfg
        )
    )
    if include_real:
        # The parser accepts exactly one real strategy (operator ruling 2); the
        # check here is so a widened parser cannot pass an unknown name through.
        if rules.real_strategy != "equal_weight":
            raise ValueError(f"unknown real strategy {rules.real_strategy!r}")
        rows.extend(
            reconcile_panel(real_panel(cfg), "equal_weight", panel="real equal-weight 13", cfg=cfg)
        )
    return Reconciliation(
        rows=tuple(rows),
        initial_nav=rules.initial_nav,
        accounting_roundings=rules.accounting_roundings,
    )
