"""SPEC.md 7.1's per-asset trading cost, and SPEC.md 7.3's scaling laws.

Per asset, on the normalised trade ``z_i = u_i / v`` (dollar trade over NAV):

    phi_i(z_i) = a_i*|z_i| + Y*sigma_i*|z_i|^e / (V_i/v)^(e-1) + c_i*z_i

with ``e = 3/2`` from ``costs.total_cost_exponent``, at which the middle term is
exactly SPEC.md 7.1's ``b*sigma_i*|z_i|^{3/2} / (V_i/v)^{1/2}``. The three terms:

- ``a_i`` is the half-spread plus commission -- built by
  :mod:`mafrm.costs.spread_level`, never taken from EDGE directly (SPEC.md 3.4.1).
- ``Y`` is the square-root prefactor, ``b`` in SPEC.md 7.1's notation. TWO
  calibrated regimes, ``costs.square_root_prefactor.patient`` and ``.urgent``,
  both always run and both always reported (SPEC.md 7.2: "Model both. Do not
  calibrate a single number."). This module takes one at a time and it is the
  caller's job to call it twice.
- ``sigma_i`` is the trailing one-year daily volatility, ``V_i`` the trailing
  one-year median dollar volume (SPEC.md 7.1; windows in ``config/model.yaml``).
- ``c_i`` is the buy/sell asymmetry, zero in this project.

**Why the exponent is 3/2.** Total cost is size times per-unit impact, and the
square-root law says per-unit impact is ``Y*sigma*sqrt(x/V)``; so the total is
``x * Y*sigma*sqrt(x/V) = Y*sigma*x^{3/2}/V^{1/2}``. The square-root law and the
3/2 cost model are the same statement, and ``3/2 > 1`` makes the term convex,
which is what makes the optimisation tractable. :func:`impact_per_unit_traded`
is the per-unit law and :func:`trade_cost` multiplies it by size; a test pins
the two against each other so the identity is checked rather than asserted.

**The no-trade region comes from the L1 term and from nowhere else.** ``a|z|``
is non-differentiable at zero, so an asset is left alone unless its marginal
alpha exceeds its round-trip spread cost. SPEC.md 7.1: do not add a separate
no-trade band on top -- it double-counts. There is no band in this module and
none is to be added downstream.

**Zero trade is exactly zero cost.** SPEC.md 7.4 names the division-by-zero
class: a cost model evaluated at ``z = 0`` with ``V = 0`` computes ``0/0``, and
the March 2026 implementation-risk paper (Yin et al.) found exactly that kind
of silent error in a production engine. :func:`trade_cost` therefore never
evaluates the impact term where the trade is zero, and returns ``0.0`` there
whatever the other inputs are -- ``NaN`` and zero included. Where the trade is
NOT zero, a missing or non-positive depth raises rather than returning
``inf`` or ``NaN``, because a cost that is not a number silently drops out of
every sum it enters.

**Units.** Everything is a PROPORTION -- of NAV for ``z``, ``V/v`` and the
result; of price for ``sigma``, ``a`` and ``c`` -- so the result is the cost as
a fraction of NAV. The cost as a fraction of the TRADE is ``phi / |z|``, which
is what published anchors quote and what :func:`implied_prefactor` inverts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "CostPanel",
    "CostTerms",
    "ImpactError",
    "ScalingLaws",
    "impact_per_unit_traded",
    "implied_prefactor",
    "rebalance_costs",
    "trade_cost",
]

FloatArray = NDArray[np.float64]


class ImpactError(ValueError):
    """The inputs cannot support a cost estimate for a non-zero trade."""


@dataclass(frozen=True)
class CostTerms:
    """SPEC.md 7.1's three terms, separately, so each can be tested by hand.

    All three are proportions of NAV with the trade's shape. ``spread`` and
    ``impact`` are non-negative; ``asymmetry`` carries the trade's sign.
    """

    spread: FloatArray
    impact: FloatArray
    asymmetry: FloatArray

    @property
    def total(self) -> FloatArray:
        out: FloatArray = self.spread + self.impact + self.asymmetry
        return out


def _as_array(value: ArrayLike, shape: tuple[int, ...], *, name: str) -> FloatArray:
    array = np.asarray(value, dtype="float64")
    try:
        return np.broadcast_to(array, shape).astype("float64", copy=True)
    except ValueError as exc:
        raise ImpactError(f"{name}: shape {array.shape} does not broadcast to {shape}") from exc


def _check_scalar(value: float, *, name: str, positive: bool = False) -> float:
    if not np.isfinite(value):
        raise ImpactError(f"{name}: must be finite, got {value}")
    if positive and value <= 0.0:
        raise ImpactError(f"{name}: must be positive, got {value}")
    if not positive and value < 0.0:
        raise ImpactError(f"{name}: must be non-negative, got {value}")
    return float(value)


def _check_exponent(exponent: float) -> float:
    exponent = _check_scalar(exponent, name="exponent", positive=True)
    if exponent <= 1.0:
        raise ImpactError(
            f"exponent must exceed 1 for the impact term to be convex (SPEC.md 7.1), got {exponent}"
        )
    return exponent


def impact_per_unit_traded(
    participation: ArrayLike,
    *,
    daily_volatility: ArrayLike,
    prefactor: float,
    exponent: float,
) -> FloatArray:
    """The square-root law: cost per unit traded, as a proportion of the trade.

    ``Y * sigma * participation^(e-1)``, which at ``e = 3/2`` is
    ``Y*sigma*sqrt(Q/V)`` -- the form SPEC.md 7.2's anchors are quoted in
    (17bp at 2% of ADV, and so on). ``participation`` is ``Q/V``, the trade as
    a fraction of ADV, and must be non-negative.
    """
    prefactor = _check_scalar(prefactor, name="prefactor")
    exponent = _check_exponent(exponent)
    q = np.asarray(participation, dtype="float64")
    sigma = np.asarray(daily_volatility, dtype="float64")
    if (q < 0.0).any():
        raise ImpactError("participation must be non-negative -- pass |Q|/V")
    out: FloatArray = prefactor * sigma * np.power(q, exponent - 1.0)
    return out


def implied_prefactor(
    *,
    cost_of_trade: float,
    participation: float,
    daily_volatility: float,
    exponent: float,
) -> float:
    """Back ``Y`` out of a published (participation, cost) pair. SPEC.md 7.2.

    ``Y = cost / (sigma * participation^(e-1))``, the inverse of
    :func:`impact_per_unit_traded`. ``cost_of_trade`` is a proportion of the
    trade, so 17bp is ``0.0017``. This is how both regimes in
    ``costs.square_root_prefactor`` were obtained and
    ``reports/cost_calibration.md`` shows the arithmetic.
    """
    cost_of_trade = _check_scalar(cost_of_trade, name="cost_of_trade", positive=True)
    participation = _check_scalar(participation, name="participation", positive=True)
    daily_volatility = _check_scalar(daily_volatility, name="daily_volatility", positive=True)
    exponent = _check_exponent(exponent)
    return float(cost_of_trade / (daily_volatility * participation ** (exponent - 1.0)))


def trade_cost(
    trade: ArrayLike,
    *,
    half_spread: ArrayLike,
    daily_volatility: ArrayLike,
    adv_over_nav: ArrayLike,
    prefactor: float,
    exponent: float,
    asymmetry: ArrayLike = 0.0,
) -> CostTerms:
    """SPEC.md 7.1, term by term, for one regime's ``prefactor``.

    ``trade`` is ``z = u/v``, signed; ``half_spread`` is ``a_i``;
    ``adv_over_nav`` is ``V_i/v``; ``asymmetry`` is ``c_i``. Inputs broadcast to
    the trade's shape, so a scalar spread against a vector of trades is fine.

    Where ``trade == 0`` every term is exactly ``0.0`` and NO other input is
    read -- see the module docstring on the division-by-zero class. Where it is
    not, a non-finite input or a non-positive depth raises :class:`ImpactError`
    naming the offending position, rather than returning ``inf`` or ``NaN``.
    """
    prefactor = _check_scalar(prefactor, name="prefactor")
    exponent = _check_exponent(exponent)

    z = np.asarray(trade, dtype="float64")
    shape = z.shape
    if not np.isfinite(z).all():
        raise ImpactError("trade contains a non-finite value; a trade is either a number or 0")

    a = _as_array(half_spread, shape, name="half_spread")
    sigma = _as_array(daily_volatility, shape, name="daily_volatility")
    depth = _as_array(adv_over_nav, shape, name="adv_over_nav")
    c = _as_array(asymmetry, shape, name="asymmetry")

    active = z != 0.0
    for name, values, positive in (
        ("half_spread", a, False),
        ("daily_volatility", sigma, False),
        ("adv_over_nav", depth, True),
        ("asymmetry", c, None),
    ):
        bad = active & ~np.isfinite(values)
        if positive is True:
            bad |= active & (values <= 0.0)
        elif positive is False:
            bad |= active & (values < 0.0)
        if bad.any():
            where = np.argwhere(bad)[0].tolist()
            kind = (
                "non-positive or non-finite"
                if positive is True
                else "negative or non-finite"
                if positive is False
                else "non-finite"
            )
            raise ImpactError(
                f"{name} is {kind} at position {where} where the trade is "
                f"{z[tuple(where)]:+.6g}; a cost cannot be computed for that trade"
            )

    size = np.abs(z)
    spread = np.zeros(shape, dtype="float64")
    impact = np.zeros(shape, dtype="float64")
    asym = np.zeros(shape, dtype="float64")
    if active.any():
        # Evaluated only where the trade is non-zero. `depth` is positive there,
        # so the division is safe; the zero-trade positions never reach it.
        participation = size[active] / depth[active]
        per_unit = impact_per_unit_traded(
            participation,
            daily_volatility=sigma[active],
            prefactor=prefactor,
            exponent=exponent,
        )
        spread[active] = a[active] * size[active]
        impact[active] = size[active] * per_unit
        asym[active] = c[active] * z[active]
    return CostTerms(spread=spread, impact=impact, asymmetry=asym)


@dataclass(frozen=True)
class ScalingLaws:
    """SPEC.md 7.3: how cost scales under the square-root law, as exponents.

    With a fixed portfolio structure (every ``z_i`` fixed) and AUM ``A``, the
    depth ``V/v`` scales as ``1/A``, so from the cost function directly:

    ======================================  ==================
    quantity                                scales as
    ======================================  ==================
    cost per unit traded                    ``A^(e-1)``  -- sqrt(A) at 3/2
    annual drag, bps of AUM                 ``tau * A^(e-1)``
    annual cost in dollars                  ``tau * A^e``  -- A^{3/2} at 3/2
    cost vs volatility                      linear
    cost vs ADV                             ``V^-(e-1)``  -- 1/sqrt(V) at 3/2
    ======================================  ==================

    Consequences SPEC.md 7.3 says to state: doubling AUM raises the bps drag
    by ``2^(e-1) = 1.41``, not 2; doubling turnover doubles it -- turnover is
    the linear lever, size the sublinear one; and a cost model without a vol
    term under-charges exactly when the optimizer most wants to trade. Each
    row is pinned against :func:`trade_cost` in the tests, not just asserted.
    """

    exponent: float

    def __post_init__(self) -> None:
        _check_exponent(self.exponent)

    @property
    def aum_exponent_per_unit(self) -> float:
        return self.exponent - 1.0

    @property
    def aum_exponent_drag_bps(self) -> float:
        return self.exponent - 1.0

    @property
    def aum_exponent_dollars(self) -> float:
        return self.exponent

    @property
    def turnover_exponent(self) -> float:
        return 1.0

    @property
    def volatility_exponent(self) -> float:
        return 1.0

    @property
    def adv_exponent(self) -> float:
        return -(self.exponent - 1.0)

    def drag_multiplier(self, *, aum_ratio: float, turnover_ratio: float = 1.0) -> float:
        """Factor on the impact drag in bps of AUM when AUM and turnover scale."""
        if aum_ratio <= 0.0 or turnover_ratio <= 0.0:
            raise ImpactError("ratios must be positive")
        return float(turnover_ratio * aum_ratio**self.aum_exponent_drag_bps)


@dataclass(frozen=True)
class CostPanel:
    """Per-date, per-asset cost terms, and the per-date total. Proportions of NAV."""

    spread: pd.DataFrame
    impact: pd.DataFrame
    asymmetry: pd.DataFrame

    @property
    def by_asset(self) -> pd.DataFrame:
        return self.spread + self.impact + self.asymmetry

    @property
    def by_date(self) -> pd.Series:
        total = self.by_asset.sum(axis=1)
        total.name = "trading_cost"
        return total


def _aligned(frame: pd.DataFrame, like: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """``frame`` on ``like``'s exact index and columns -- reindexed, NEVER filled.

    An input missing at a rebalance date comes back ``NaN`` and, where the trade
    there is non-zero, :func:`trade_cost` raises. Forward-filling would charge a
    stale spread; back-filling would charge tomorrow's. Neither is "the correct
    time" of SPEC.md 7.4.
    """
    missing_cols = [c for c in like.columns if c not in frame.columns]
    if missing_cols:
        raise ImpactError(f"{name}: no column for {missing_cols}")
    return frame.reindex(index=like.index, columns=like.columns)


def rebalance_costs(
    trades: pd.DataFrame,
    *,
    half_spread: pd.DataFrame,
    daily_volatility: pd.DataFrame,
    adv_over_nav: pd.DataFrame,
    prefactor: float,
    exponent: float,
    asymmetry: pd.DataFrame | float = 0.0,
) -> CostPanel:
    """SPEC.md 7.4: costs on ACTUAL turnover, on BOTH sides, at the CORRECT time.

    ``trades`` is a (date x asset) frame of ``z = u/v``: what was actually
    traded at each rebalance, signed, as a fraction of NAV -- not the target
    weight, not the weight change gross of drift. Sells cost as much as buys of
    the same size (``|z|`` enters both non-signed terms), so a switch from A to
    B is charged on both legs. Each date's cost reads that date's inputs and
    only that date's: the trailing-window spread, volatility and depth standing
    at the rebalance, which is when the trade is decided and executed.

    ``adv_over_nav`` is ``V_i(t)/v(t)`` and the caller supplies NAV -- this
    module does not know the fund size, and the capacity curve varies it.
    """
    if trades.empty:
        raise ImpactError("trades: an empty frame has no rebalance to cost")
    if trades.isna().any().any():
        raise ImpactError("trades: NaN is not a trade; write 0.0 for an asset that did not trade")

    a = _aligned(half_spread, trades, name="half_spread")
    sigma = _aligned(daily_volatility, trades, name="daily_volatility")
    depth = _aligned(adv_over_nav, trades, name="adv_over_nav")
    c = (
        _aligned(asymmetry, trades, name="asymmetry")
        if isinstance(asymmetry, pd.DataFrame)
        else pd.DataFrame(float(asymmetry), index=trades.index, columns=trades.columns)
    )

    terms = trade_cost(
        trades.to_numpy(dtype="float64"),
        half_spread=a.to_numpy(dtype="float64"),
        daily_volatility=sigma.to_numpy(dtype="float64"),
        adv_over_nav=depth.to_numpy(dtype="float64"),
        prefactor=prefactor,
        exponent=exponent,
        asymmetry=c.to_numpy(dtype="float64"),
    )

    def frame(values: FloatArray) -> pd.DataFrame:
        return pd.DataFrame(values, index=trades.index, columns=trades.columns)

    return CostPanel(
        spread=frame(terms.spread), impact=frame(terms.impact), asymmetry=frame(terms.asymmetry)
    )
