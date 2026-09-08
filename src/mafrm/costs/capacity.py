"""SPEC.md 10.4's capacity arithmetic under the square-root law -- the closed form.

With ``A`` = AUM, ``tau`` = annual two-way turnover and ``kappa`` the impact
coefficient a fixed portfolio structure pins (below), at the configured cost
exponent ``e`` (``costs.total_cost_exponent``, 3/2):

    TC(A)      = tau * kappa * A^(e-1)                      sqrt(A) at e = 3/2
    alpha_n(A) = alpha_g - TC(A)
    A_BE       = (alpha_g / (tau*kappa))^(1/(e-1))          break-even
    A_eff      = ((alpha_g - alpha_min) / (e*tau*kappa))^(1/(e-1))
               = (4/9) * A_BE                               at e = 3/2, alpha_min = 0

``A_eff`` is where the MARGINAL dollar of AUM earns ``alpha_min``: dollar net
alpha is ``A*alpha_n(A) = alpha_g*A - tau*kappa*A^e``, whose derivative is
``alpha_g - e*tau*kappa*A^(e-1)``; set it to ``alpha_min`` and solve. The ratio
``A_eff/A_BE = e^(-1/(e-1))`` is ``(2/3)^2 = 4/9`` at 3/2, which SPEC.md 10.4
says to hard-code and mark on the curve; :data:`EFFECTIVE_TO_BREAK_EVEN_RATIO`
is that constant and a test pins the general formula to it.

**THIS CLOSED FORM IS THE RESCALING SPEC.md 10.4 WARNS OVERSTATES COST.** It
holds the portfolio structure fixed and scales the impact function with AUM.
The right capacity experiment re-optimises at each candidate AUM with the ADV
constraints binding and lets the optimizer choose *different* holdings; that
curve needs the optimizer and is W6-P2/W6-P3 work (SPEC.md 10.4.1). What this
module produces is the upper bound on cost -- and therefore the LOWER bound on
capacity -- that the re-optimised curve is compared against.

**No input to the curve exists yet, and none is invented here.** ``alpha_g``,
``tau`` and the structure that fixes ``kappa`` all come from a trade series,
and there is none until W6's optimizer produces one. Every function takes them
as arguments; nothing reads them from data. ``alpha_min`` is parameterised and
ONLY the spec's ``alpha_min = 0`` special case is implemented -- any other
value raises until one is sourced (``costs.capacity.minimum_net_alpha``).

**Where ``kappa`` comes from.** For a fixed structure -- per-rebalance trades
``z_i`` as fractions of NAV, ``R`` rebalances a year, each asset with daily
volatility ``sigma_i`` and dollar ADV ``V_i`` -- SPEC.md 7.1's impact term at
AUM ``A`` is ``Y*sigma_i*|z_i|^e / (V_i/A)^(e-1)`` per rebalance, so

    annual impact drag = A^(e-1) * R * sum_i Y*sigma_i*|z_i|^e / V_i^(e-1)
    tau                = R * sum_i |z_i|
    kappa              = sum_i Y*sigma_i*|z_i|^e / V_i^(e-1)  /  sum_i |z_i|

Since ``kappa`` is linear in ``Y``, the two calibrated regimes give two
``kappa`` values and the capacity band between them is
``(Y_patient/Y_urgent)^(1/(e-1))`` -- ``(0.58/1.40)^2 = 0.172`` at 3/2. That
band IS the cost-model sensitivity SPEC.md 10.4 requires beside every capacity
number: capacity is quadratic in ``kappa`` (halving it quadruples capacity),
and the two regimes are the only two points the sensitivity is evaluated at,
because SPEC.md 7.2.1 rules ``Y`` to be two regimes with no interior points.

**The L1 term.** SPEC.md 7.1's spread term ``a_i*|z_i|`` is AUM-invariant, so
SPEC.md 10.4's impact-only form silently drops it. It enters here exactly, as
``aum_invariant_drag`` (annual, fraction of NAV) subtracted from ``alpha_g``
wherever ``alpha_g`` appears -- an algebraic fact, not a parameter. Default
zero reproduces the spec's formulas verbatim.

**Units.** ``alpha_g``, ``TC``, ``alpha_n`` and ``aum_invariant_drag`` are
annual fractions of NAV. ``tau`` is a fraction of NAV per year. ``A`` and
``V_i`` are dollars, so ``kappa`` carries ``$^-(e-1)``. Nothing here is in
basis points; the report converts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from mafrm.config import AumGridRule
from mafrm.costs import impact

__all__ = [
    "EFFECTIVE_TO_BREAK_EVEN_RATIO",
    "CapacityCurve",
    "CapacityError",
    "FixedStructure",
    "aum_grid",
    "aum_grid_bounds",
    "break_even_aum",
    "capacity_curve",
    "capacity_multiplier",
    "cost_drag",
    "effective_aum",
    "effective_to_break_even_ratio",
    "kappa_sensitivity",
    "net_alpha",
    "structure_kappa",
]

FloatArray = NDArray[np.float64]

#: SPEC.md 10.4: "Effective capacity under square-root impact is 4/9 of
#: break-even capacity. Hard-code that relationship and mark both points on the
#: curve." :func:`effective_to_break_even_ratio` is the general form and a test
#: pins it to this constant at the configured exponent.
EFFECTIVE_TO_BREAK_EVEN_RATIO: Final[float] = 4.0 / 9.0


class CapacityError(ValueError):
    """The inputs cannot support a capacity statement."""


def _exponent(exponent: float) -> float:
    try:
        impact.ScalingLaws(exponent)
    except impact.ImpactError as exc:
        raise CapacityError(str(exc)) from exc
    return float(exponent)


def _finite(value: float, *, name: str, positive: bool = False) -> float:
    if not np.isfinite(value):
        raise CapacityError(f"{name}: must be finite, got {value}")
    if positive and value <= 0.0:
        raise CapacityError(f"{name}: must be positive, got {value}")
    if not positive and value < 0.0:
        raise CapacityError(f"{name}: must be non-negative, got {value}")
    return float(value)


def _minimum_net_alpha(value: float) -> float:
    """Only SPEC.md 10.4's ``alpha_min = 0`` is implemented (operator ruling, W5-P2)."""
    if not np.isfinite(value) or value != 0.0:
        raise CapacityError(
            f"minimum_net_alpha: only the SPEC.md 10.4 special case alpha_min = 0 is implemented "
            f"and no source supplies another value; got {value}. Refused until one is sourced "
            "(costs.capacity.minimum_net_alpha)."
        )
    return float(value)


def _aum(aum: ArrayLike) -> FloatArray:
    a = np.asarray(aum, dtype="float64")
    if not np.isfinite(a).all():
        raise CapacityError("aum: must be finite")
    if (a < 0.0).any():
        raise CapacityError("aum: must be non-negative")
    return a


def effective_to_break_even_ratio(exponent: float) -> float:
    """``A_eff / A_BE = e^(-1/(e-1))`` at ``alpha_min = 0``; ``4/9`` at ``e = 3/2``.

    Independent of ``alpha_g``, ``tau`` and ``kappa``: the marginal condition
    ``alpha_g = e*tau*kappa*A^(e-1)`` is the break-even condition with
    ``tau*kappa`` scaled by ``e``.
    """
    e = _exponent(exponent)
    return float(e ** (-1.0 / (e - 1.0)))


def cost_drag(aum: ArrayLike, *, turnover: float, kappa: float, exponent: float) -> FloatArray:
    """``TC(A) = tau * kappa * A^(e-1)``, annual, fraction of NAV. SPEC.md 10.4."""
    a = _aum(aum)
    tau = _finite(turnover, name="turnover")
    k = _finite(kappa, name="kappa")
    e = _exponent(exponent)
    out: FloatArray = tau * k * np.power(a, e - 1.0)
    return out


def net_alpha(
    aum: ArrayLike,
    *,
    gross_alpha: float,
    turnover: float,
    kappa: float,
    exponent: float,
    aum_invariant_drag: float = 0.0,
) -> FloatArray:
    """``alpha_n(A) = alpha_g - d - TC(A)``, with ``d`` the AUM-invariant (spread) drag."""
    alpha_g = _finite(gross_alpha, name="gross_alpha")
    d = _finite(aum_invariant_drag, name="aum_invariant_drag")
    out: FloatArray = (
        alpha_g - d - cost_drag(aum, turnover=turnover, kappa=kappa, exponent=exponent)
    )
    return out


def _capacity(
    *,
    numerator: float,
    turnover: float,
    kappa: float,
    exponent: float,
    scale: float,
) -> float:
    """``(numerator / (scale*tau*kappa))^(1/(e-1))``; zero when the numerator is not positive.

    A strategy whose gross alpha does not cover its AUM-invariant drag has a
    non-positive net alpha at every AUM, so its capacity is exactly zero rather
    than an error.
    """
    tau = _finite(turnover, name="turnover", positive=True)
    k = _finite(kappa, name="kappa", positive=True)
    e = _exponent(exponent)
    if numerator <= 0.0:
        return 0.0
    return float((numerator / (scale * tau * k)) ** (1.0 / (e - 1.0)))


def break_even_aum(
    *,
    gross_alpha: float,
    turnover: float,
    kappa: float,
    exponent: float,
    aum_invariant_drag: float = 0.0,
) -> float:
    """``A_BE = ((alpha_g - d) / (tau*kappa))^(1/(e-1))``, where ``alpha_n = 0``. SPEC.md 10.4."""
    alpha_g = _finite(gross_alpha, name="gross_alpha")
    d = _finite(aum_invariant_drag, name="aum_invariant_drag")
    return _capacity(
        numerator=alpha_g - d, turnover=turnover, kappa=kappa, exponent=exponent, scale=1.0
    )


def effective_aum(
    *,
    gross_alpha: float,
    turnover: float,
    kappa: float,
    exponent: float,
    minimum_net_alpha: float,
    aum_invariant_drag: float = 0.0,
) -> float:
    """``A_eff = ((alpha_g - d - alpha_min) / (e*tau*kappa))^(1/(e-1))``. SPEC.md 10.4.

    Where the marginal dollar earns ``alpha_min``. Only ``alpha_min = 0`` is
    accepted, at which ``A_eff = (4/9) A_BE`` for ``e = 3/2``.
    """
    alpha_g = _finite(gross_alpha, name="gross_alpha")
    d = _finite(aum_invariant_drag, name="aum_invariant_drag")
    alpha_min = _minimum_net_alpha(minimum_net_alpha)
    e = _exponent(exponent)
    return _capacity(
        numerator=alpha_g - d - alpha_min, turnover=turnover, kappa=kappa, exponent=e, scale=e
    )


def capacity_multiplier(kappa_ratio: float, *, exponent: float) -> float:
    """Factor on ``A_BE`` (and ``A_eff``) when ``kappa`` is multiplied by ``kappa_ratio``.

    ``kappa_ratio^(-1/(e-1))``: quadratic at 3/2, so halving the cost
    coefficient quadruples capacity. SPEC.md 10.4: "any capacity number you
    publish must be accompanied by its cost-model sensitivity."
    """
    r = _finite(kappa_ratio, name="kappa_ratio", positive=True)
    e = _exponent(exponent)
    return float(r ** (-1.0 / (e - 1.0)))


# ---------------------------------------------------------------------------
# kappa from a fixed portfolio structure, pinned against the cost engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedStructure:
    """What a fixed structure implies for SPEC.md 10.4: ``tau``, ``kappa`` and the spread drag.

    ``turnover`` is annual two-way, fraction of NAV. ``kappa`` carries
    ``$^-(e-1)``. ``aum_invariant_drag`` is the annual L1 spread cost, fraction
    of NAV, which does not scale with AUM. ``rebalances_per_year`` and the
    number of rebalances the structure was averaged over are kept so the
    engine cross-check can reproduce the arithmetic.
    """

    turnover: float
    kappa: float
    aum_invariant_drag: float
    rebalances_per_year: float
    rebalances_in_sample: int

    def cost_drag(self, aum: ArrayLike, *, exponent: float) -> FloatArray:
        return cost_drag(aum, turnover=self.turnover, kappa=self.kappa, exponent=exponent)


def structure_kappa(
    trades: ArrayLike,
    *,
    daily_volatility: ArrayLike,
    adv_dollars: ArrayLike,
    prefactor: float,
    exponent: float,
    rebalances_per_year: float,
    half_spread: ArrayLike = 0.0,
) -> FixedStructure:
    """``kappa`` and ``tau`` for a fixed structure, from SPEC.md 7.1's cost function.

    ``trades`` is signed ``z = u/v``, either one rebalance (1-D, assets) or a
    sample of rebalances (2-D, rebalances x assets) whose average is the
    structure. ``daily_volatility`` and ``adv_dollars`` broadcast to it;
    ``adv_dollars`` is ``V_i`` in dollars, NOT ``V_i/v``, because the whole
    point is to vary the NAV. The result satisfies, for any AUM ``A``,

        rebalances_per_year * mean over rebalances of
            sum_i impact(z_i; sigma_i, V_i/A)   ==   tau * kappa * A^(e-1)

    with the impact taken from :func:`mafrm.costs.impact.trade_cost`; the test
    asserts that identity across a grid of ``A`` rather than trusting the
    algebra.
    """
    y = _finite(prefactor, name="prefactor")
    e = _exponent(exponent)
    r = _finite(rebalances_per_year, name="rebalances_per_year", positive=True)

    z = np.asarray(trades, dtype="float64")
    if z.ndim not in (1, 2):
        raise CapacityError(
            f"trades: expected 1-D (assets) or 2-D (rebalances x assets), got {z.ndim}-D"
        )
    if not np.isfinite(z).all():
        raise CapacityError("trades: NaN is not a trade; write 0.0 for an asset that did not trade")
    shape = z.shape
    sigma = np.broadcast_to(np.asarray(daily_volatility, dtype="float64"), shape)
    v = np.broadcast_to(np.asarray(adv_dollars, dtype="float64"), shape)
    a = np.broadcast_to(np.asarray(half_spread, dtype="float64"), shape)

    size = np.abs(z)
    active = size > 0.0
    for name, values, positive in (
        ("daily_volatility", sigma, False),
        ("adv_dollars", v, True),
        ("half_spread", a, False),
    ):
        bad = active & ~np.isfinite(values)
        bad |= active & ((values <= 0.0) if positive else (values < 0.0))
        if bad.any():
            where = np.argwhere(bad)[0].tolist()
            raise CapacityError(
                f"{name} is unusable at position {where} where the trade is non-zero"
            )

    total_size = float(size.sum())
    if total_size <= 0.0:
        raise CapacityError("trades: a structure that never trades has no turnover and no kappa")

    n_rebalances = int(shape[0]) if z.ndim == 2 else 1
    impact_numerator = np.zeros(shape, dtype="float64")
    impact_numerator[active] = (
        y * sigma[active] * np.power(size[active], e) / np.power(v[active], e - 1.0)
    )
    spread_per_rebalance = float((a * size).sum()) / n_rebalances

    return FixedStructure(
        turnover=r * total_size / n_rebalances,
        kappa=float(impact_numerator.sum()) / total_size,
        aum_invariant_drag=r * spread_per_rebalance,
        rebalances_per_year=r,
        rebalances_in_sample=n_rebalances,
    )


# ---------------------------------------------------------------------------
# The AUM grid rule and the curve
# ---------------------------------------------------------------------------


def aum_grid_bounds(
    *,
    largest_trade_fraction_of_nav: float,
    proxy_adv_dollars: float,
    rule: AumGridRule,
) -> tuple[float, float]:
    """The grid's two ends from the ADV data, per ``costs.capacity.aum_grid``.

    The largest position's trade is ``z_max * A`` dollars; it is participation
    ``p`` of its proxy's ADV ``V`` when ``A = p*V/z_max``. Both ends follow from
    the rule's two participations and nothing is chosen here.
    """
    z_max = _finite(
        largest_trade_fraction_of_nav, name="largest_trade_fraction_of_nav", positive=True
    )
    v = _finite(proxy_adv_dollars, name="proxy_adv_dollars", positive=True)
    low = rule.low_participation_of_adv * v / z_max
    high = rule.high_participation_of_adv * v / z_max
    return float(low), float(high)


def aum_grid(bounds: tuple[float, float], *, rule: AumGridRule) -> FloatArray:
    """Log-spaced candidate AUMs between the bounds; ``rule.points`` is display resolution."""
    low, high = bounds
    if not 0.0 < low < high:
        raise CapacityError(f"aum grid bounds must satisfy 0 < low < high, got {bounds}")
    if rule.spacing != "log":
        raise CapacityError(
            f"aum grid spacing: only 'log' is ruled (SPEC.md 10.4.1), got {rule.spacing!r}"
        )
    out: FloatArray = np.geomspace(low, high, num=rule.points, dtype="float64")
    return out


def kappa_sensitivity(
    *,
    gross_alpha: float,
    turnover: float,
    kappa_by_regime: Mapping[str, float],
    exponent: float,
    minimum_net_alpha: float,
    aum_invariant_drag: float = 0.0,
) -> pd.DataFrame:
    """``A_BE`` and ``A_eff`` per cost regime -- the sensitivity band SPEC.md 10.4 requires.

    One row per regime, columns ``kappa``, ``break_even_aum``, ``effective_aum``.
    Nothing is selected between the rows: both regimes are reported, always.
    """
    if not kappa_by_regime:
        raise CapacityError("kappa_by_regime: at least one regime is needed")
    rows = {}
    for regime, k in kappa_by_regime.items():
        rows[regime] = {
            "kappa": float(k),
            "break_even_aum": break_even_aum(
                gross_alpha=gross_alpha,
                turnover=turnover,
                kappa=k,
                exponent=exponent,
                aum_invariant_drag=aum_invariant_drag,
            ),
            "effective_aum": effective_aum(
                gross_alpha=gross_alpha,
                turnover=turnover,
                kappa=k,
                exponent=exponent,
                minimum_net_alpha=minimum_net_alpha,
                aum_invariant_drag=aum_invariant_drag,
            ),
        }
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "regime"
    return frame


@dataclass(frozen=True)
class CapacityCurve:
    """SPEC.md 10.4's curve at every regime, with both marked points per regime.

    ``net_alpha`` is (AUM x regime), annual fraction of NAV. ``points`` is the
    :func:`kappa_sensitivity` frame -- the two marked AUMs per regime -- and
    is computed in closed form, never read off the grid.
    """

    aum: FloatArray
    net_alpha: pd.DataFrame
    points: pd.DataFrame
    gross_alpha: float
    turnover: float
    exponent: float
    aum_invariant_drag: float


def capacity_curve(
    aum: ArrayLike,
    *,
    gross_alpha: float,
    turnover: float,
    kappa_by_regime: Mapping[str, float],
    exponent: float,
    minimum_net_alpha: float,
    aum_invariant_drag: float = 0.0,
) -> CapacityCurve:
    """The closed-form (rescaling) curve. See the module docstring for what it is not."""
    grid = _aum(aum)
    if grid.ndim != 1 or grid.size < 2:
        raise CapacityError("aum: need a 1-D grid of at least two AUMs")
    columns = {
        regime: net_alpha(
            grid,
            gross_alpha=gross_alpha,
            turnover=turnover,
            kappa=k,
            exponent=exponent,
            aum_invariant_drag=aum_invariant_drag,
        )
        for regime, k in kappa_by_regime.items()
    }
    frame = pd.DataFrame(columns, index=pd.Index(grid, name="aum"))
    points = kappa_sensitivity(
        gross_alpha=gross_alpha,
        turnover=turnover,
        kappa_by_regime=kappa_by_regime,
        exponent=exponent,
        minimum_net_alpha=minimum_net_alpha,
        aum_invariant_drag=aum_invariant_drag,
    )
    return CapacityCurve(
        aum=grid,
        net_alpha=frame,
        points=points,
        gross_alpha=float(gross_alpha),
        turnover=float(turnover),
        exponent=float(exponent),
        aum_invariant_drag=float(aum_invariant_drag),
    )
