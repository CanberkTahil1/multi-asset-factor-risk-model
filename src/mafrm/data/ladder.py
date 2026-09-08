"""Constant-maturity par-coupon Treasury ladder. **VALIDATION ONLY.**

This module exists to give the TLT cross-check (SPEC.md 3.2) a comparand with an
ETF's cashflow structure. TLT holds every outstanding Treasury with at least 20
years remaining -- coupon bonds spanning 20-30y, with a regression-implied
modified duration near 15.4 years. Scoring it against a single 20y *zero* was
comparing two different instruments: the zero earns +0.42%/yr of roll-down where
the ladder earns +0.21%, and the zero's duration is its full 20 years.

**Nothing here may enter the production series.** The government sleeve's factor
inputs are the constant-maturity zeros in :mod:`mafrm.data.gsw`, and they stay
that way. A coupon ladder is the right thing to validate an ETF against and the
wrong thing to build factors from: it blends maturities, so its loading on the
level/slope/curvature structure of the curve is a mixture rather than a clean
key rate, which is exactly the property the macro factor model (SPEC.md 4.1)
needs its inputs *not* to have. ``tests/test_ladder.py`` asserts that no module
outside the validation path imports this one.

Every bond is repriced to par at each date, so the ladder is a constant-maturity
construct in the same sense as the zeros: the coupon is reset daily to whatever
makes the bond worth 100 on that day's curve, and the return is what that bond
earns over the following day as it ages by ``Delta`` and the curve moves.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from mafrm.config import load
from mafrm.data.gsw import published_mask, svensson_yield

__all__ = [
    "LadderError",
    "cashflow_times",
    "discount_factors",
    "ladder_returns",
    "par_bond_returns",
    "par_coupon_rate",
    "seasoned_bond_returns",
    "seasoned_ladder_returns",
]


class LadderError(RuntimeError):
    """A ladder was asked for something the par-bond construction cannot give."""


def cashflow_times(
    maturity: float, frequency: int, *, first_coupon: float | None = None
) -> npt.NDArray[np.float64]:
    """Coupon dates in years, from the first payment to redemption at ``maturity``.

    A 2y bond paying semiannually returns ``[0.5, 1.0, 1.5, 2.0]``.

    ``first_coupon`` places the next coupon somewhere inside the current period
    instead of a full period away, which is the real situation for every bond on
    every day that is not a coupon date. Redemption stays at ``maturity``, so the
    last two cashflows fall closer together than ``1/f``. Used to bound how much
    the ladder's assumption of a full first period is worth; the default is the
    full period.
    """
    if maturity <= 0:
        raise LadderError(f"maturity must be positive, got {maturity}")
    if frequency <= 0:
        raise LadderError(f"coupon frequency must be positive, got {frequency}")
    period = 1.0 / frequency
    if first_coupon is None:
        periods = round(maturity * frequency)
        if abs(periods / frequency - maturity) > 1e-9:
            raise LadderError(
                f"maturity {maturity} is not a whole number of {frequency}-per-year periods"
            )
        return np.arange(1, periods + 1, dtype=float) * period
    if not 0.0 < first_coupon <= period:
        raise LadderError(f"first_coupon must lie in (0, {period}], got {first_coupon}")
    times = np.arange(first_coupon, maturity - 1e-9, period, dtype=float)
    return np.append(times, maturity)


def discount_factors(
    params: pd.DataFrame, times: npt.NDArray[np.float64], *, small_n: float | None = None
) -> npt.NDArray[np.float64]:
    """``exp(-y(t).t/100)`` for every date in ``params`` and every time in ``times``.

    Shape ``(len(params), len(times))``. The yields come from the same Svensson
    evaluation the production zeros use, so a curve error shows up in both.
    """
    grid = np.asarray(times, dtype=float)[None, :]

    def column(name: str) -> npt.NDArray[np.float64]:
        return np.asarray(params[name].to_numpy(), dtype=float)[:, None]

    yields = svensson_yield(
        column("BETA0"),
        column("BETA1"),
        column("BETA2"),
        column("BETA3"),
        column("TAU1"),
        column("TAU2"),
        grid,
        small_n=small_n,
    )
    factors: npt.NDArray[np.float64] = np.exp(-yields / 100.0 * grid)
    return factors


def par_coupon_rate(discounts: npt.NDArray[np.float64], frequency: int) -> npt.NDArray[np.float64]:
    """Annual coupon, in percent of face, that prices the bond at exactly par.

    From ``(c/f).sum(D) + 100.D(m) = 100``::

        c = f . (1 - D(m)) / sum(D) . 100

    ``discounts`` is one row per date, one column per cashflow, redemption last.
    The factor of ``f`` is not decoration: the coupon is quoted per annum but
    paid ``f`` times a year, and dropping it halves every coupon, so the bond no
    longer prices at par and the return series is nonsense. :func:`par_bond_returns`
    asserts the par identity for exactly this reason.
    """
    if discounts.ndim != 2 or discounts.shape[1] < 1:
        raise LadderError("discount factors must be a (dates x cashflows) matrix")
    if frequency <= 0:
        raise LadderError(f"coupon frequency must be positive, got {frequency}")
    rate: npt.NDArray[np.float64] = (
        frequency * (1.0 - discounts[:, -1]) / discounts.sum(axis=1) * 100.0
    )
    return rate


def par_bond_returns(
    params: pd.DataFrame,
    maturity: float,
    *,
    frequency: int,
    delta: float,
    small_n: float | None = None,
    first_coupon: float | None = None,
    yield_prefix: str | None = None,
    mask_unpublished: bool = True,
) -> pd.DataFrame:
    """Daily total return of a constant-maturity par-coupon bond.

    The bond is struck at par on day ``t`` -- its coupon is the day-``t`` par
    rate for maturity ``m`` -- and repriced on the day-``t+1`` curve with every
    cashflow one day closer. Same roll-down convention as the zeros: the ageing
    is ``Delta``, and the day-``t+1`` curve is read at the aged times, never at
    the original ones.

    Returns ``total_return``, plus the log decomposition ``carry`` (the bond's
    continuously-compounded yield over ``Delta``), ``roll_down`` (what ageing
    along the *unchanged* day-``t`` curve is worth beyond carry) and
    ``duration_effect`` (the day's curve move), which sum to ``log(1+r)``.
    """
    times = cashflow_times(maturity, frequency, first_coupon=first_coupon)
    aged = times - delta
    if aged[0] <= 0:
        raise LadderError(
            f"the first coupon of a {maturity}y bond falls within one roll step; "
            "this construction assumes no cashflow inside (t, t+Delta]"
        )

    discounts = discount_factors(params, times, small_n=small_n)
    aged_same_curve = discount_factors(params, aged, small_n=small_n)
    coupon = par_coupon_rate(discounts, frequency)

    def value(
        disc: npt.NDArray[np.float64], cpn: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        per_period = cpn / frequency
        priced: npt.NDArray[np.float64] = (per_period[:, None] * disc[:, :-1]).sum(axis=1) + (
            per_period + 100.0
        ) * disc[:, -1]
        return priced

    # The defining identity: struck at par, the bond is worth exactly 100 on its
    # own curve. Cheap, and it catches a mis-specified coupon immediately --
    # which is how the missing frequency factor above was found.
    at_par = value(discounts, coupon)
    worst = float(np.abs(at_par - 100.0).max())
    if worst > 1e-9:
        raise LadderError(
            f"par bond at {maturity}y does not price at 100 (worst {worst:.3e} off): "
            "the coupon or the discount factors are mis-specified"
        )

    # Row t, day-t curve, aged cashflows: isolates carry + roll, no yield change.
    static = value(aged_same_curve, coupon) / 100.0

    # Row t+1 is the realised return on the bond STRUCK AT t: the day-(t+1)
    # curve read at the aged cashflow times, valuing yesterday's coupon. Only
    # the coupon is lagged -- lagging the discount matrix too would price the
    # bond on the wrong day's curve, which is a silent and total corruption.
    coupon_struck_yesterday = pd.Series(coupon, index=params.index).shift(1).to_numpy()
    realised = value(aged_same_curve, coupon_struck_yesterday) / 100.0

    # A par bond's yield to maturity is its coupon; convert to continuous.
    ytm_continuous = frequency * np.log1p(coupon / (100.0 * frequency))
    carry = ytm_continuous * delta
    roll_down = np.log(static) - carry

    log_total = pd.Series(np.log(realised), index=params.index)
    frame = pd.DataFrame(
        {
            "total_return": np.expm1(log_total),
            "carry": pd.Series(carry, index=params.index).shift(1),
            "roll_down": pd.Series(roll_down, index=params.index).shift(1),
            "par_coupon_pct": pd.Series(coupon, index=params.index).shift(1),
        }
    )
    frame["duration_effect"] = log_total - frame["carry"] - frame["roll_down"]

    if mask_unpublished:
        # Same rule as the zeros: a maturity exists only on the dates the Fed
        # publishes a fitted yield covering it. Without this the ladder prices a
        # 30y bond on days with no observed 30y point.
        if yield_prefix is None:
            yield_prefix = load().model.data.nominal_curve.yield_prefix
        published = published_mask(params, maturity, yield_prefix=yield_prefix)
        frame = frame.where(published & published.shift(1))

    return frame.iloc[1:]


def ladder_returns(
    params: pd.DataFrame,
    *,
    weights: tuple[float, ...] | None = None,
    maturities: tuple[float, ...] | None = None,
    frequency: int | None = None,
    delta: float | None = None,
    small_n: float | None = None,
    yield_prefix: str | None = None,
    mask_unpublished: bool = True,
) -> pd.DataFrame:
    """Weighted ladder of constant-maturity par bonds. VALIDATION ONLY.

    Defaults come from ``model.data.validation_ladder``: maturities 20-30y with
    a 3:1 tilt toward 20-25y, which reproduces TLT's regression-implied 15.4y
    modified duration.
    """
    config = load().model.data
    ladder = config.validation_ladder
    if maturities is None:
        maturities = ladder.maturities_years
    if weights is None:
        weights = ladder.normalised_weights if len(maturities) == len(ladder.weights) else None
    if weights is None:
        raise LadderError("weights must be supplied when maturities differ from the configured set")
    if len(weights) != len(maturities):
        raise LadderError("weights and maturities must be the same length")
    if frequency is None:
        frequency = ladder.coupon_frequency
    if delta is None:
        delta = config.roll_down_step

    total = sum(weights)
    if total <= 0:
        raise LadderError("ladder weights must sum to something positive")
    normalised = [w / total for w in weights]

    columns = ("total_return", "carry", "roll_down", "duration_effect")
    blended: pd.DataFrame | None = None
    for weight, maturity in zip(normalised, maturities, strict=True):
        leg = par_bond_returns(params, maturity, frequency=frequency, delta=delta, small_n=small_n)[
            list(columns)
        ]
        blended = leg * weight if blended is None else blended.add(leg * weight, fill_value=0.0)
    assert blended is not None  # maturities is non-empty by config validation
    blended.attrs["maturities_years"] = tuple(maturities)
    blended.attrs["weights"] = tuple(normalised)
    return blended


# ---------------------------------------------------------------------------
# Seasoned bonds -- bounding the par idealisation
# ---------------------------------------------------------------------------
#
# A par ladder strikes a fresh bond at par every day. A real index holds bonds
# issued years ago, carrying whatever coupon was par then, and therefore trading
# at a premium or a discount today. Same maturity, same curve, different coupon
# -- and a different present-value weighting across the cashflows, hence a
# different effective duration. These functions measure that difference so it
# can be quoted as a number rather than waved at.


def seasoned_bond_returns(
    params: pd.DataFrame,
    remaining: float,
    seasoning: float,
    *,
    frequency: int,
    delta: float,
    small_n: float | None = None,
    yield_prefix: str | None = None,
    mask_unpublished: bool = True,
) -> pd.DataFrame:
    """Daily total return of a bond with ``remaining`` years left, struck ``seasoning`` years ago.

    The bond was issued at par ``seasoning`` years before each date, with original
    maturity ``remaining + seasoning``, so it carries the par coupon that
    prevailed then. Today it is priced off today's curve at whatever that coupon
    is worth -- a premium if rates have fallen since issue, a discount if they
    have risen -- and the return is measured against that price, not against par.

    ``seasoning=0`` reproduces :func:`par_bond_returns` exactly, which is the
    test that the two constructions differ only in the coupon.

    A bond with ``remaining`` years left cannot have been issued more than 30
    years ago, so the caller is responsible for capping ``seasoning`` at
    ``30 - remaining``; nothing here enforces a market convention.
    """
    if seasoning < 0:
        raise LadderError(f"seasoning must be non-negative, got {seasoning}")
    if yield_prefix is None:
        yield_prefix = load().model.data.nominal_curve.yield_prefix

    original = remaining + seasoning
    times = cashflow_times(remaining, frequency)
    aged = times - delta
    if aged[0] <= 0:
        raise LadderError(f"the first coupon of a {remaining}y bond falls within one roll step")

    # The coupon this bond carries: the par rate for `original` years, as of the
    # last curve date on or before `seasoning` years ago.
    at_issue = par_coupon_rate(
        discount_factors(params, cashflow_times(original, frequency), small_n=small_n), frequency
    )
    if mask_unpublished:
        at_issue = np.where(
            published_mask(params, original, yield_prefix=yield_prefix).to_numpy(),
            at_issue,
            np.nan,
        )
    index = pd.DatetimeIndex(params.index)
    struck_on = index - pd.DateOffset(years=int(seasoning)) if seasoning else index
    position = index.searchsorted(struck_on, side="right") - 1
    coupon = np.where(position >= 0, at_issue[np.maximum(position, 0)], np.nan)

    discounts = discount_factors(params, times, small_n=small_n)
    discounts_aged = discount_factors(params, aged, small_n=small_n)

    def value(
        disc: npt.NDArray[np.float64], cpn: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        per_period = cpn / frequency
        priced: npt.NDArray[np.float64] = (per_period[:, None] * disc[:, :-1]).sum(axis=1) + (
            per_period + 100.0
        ) * disc[:, -1]
        return priced

    coupon_yesterday = pd.Series(coupon, index=index).shift(1).to_numpy()
    # Denominator is yesterday's price of the SAME bond; numerator is today's
    # price of it, one day older. Neither is 100 unless it happens to be at par.
    price = pd.Series(value(discounts, coupon), index=index)
    price_yesterday = price.shift(1)
    value_today = pd.Series(value(discounts_aged, coupon_yesterday), index=index)

    frame = pd.DataFrame(
        {
            "total_return": value_today / price_yesterday - 1.0,
            "coupon_pct": pd.Series(coupon_yesterday, index=index),
            "price_pct": price_yesterday,
        }
    )
    if mask_unpublished:
        published = published_mask(params, remaining, yield_prefix=yield_prefix)
        frame = frame.where(published & published.shift(1))
    return frame.iloc[1:]


def seasoned_ladder_returns(
    params: pd.DataFrame,
    seasoning: float,
    *,
    maturities: tuple[float, ...] | None = None,
    weights: tuple[float, ...] | None = None,
    frequency: int | None = None,
    delta: float | None = None,
    small_n: float | None = None,
    max_original_maturity: float = 30.0,
    mask_unpublished: bool = True,
) -> pd.Series:
    """The configured ladder, but holding bonds issued ``seasoning`` years ago.

    Seasoning is capped per leg at ``max_original_maturity - maturity``: a bond
    with 30 years left must be newly issued, one with 20 years left can be a
    30y bond ten years old. At ``seasoning=0`` this is the par ladder.
    """
    config = load().model.data
    ladder = config.validation_ladder
    if maturities is None:
        maturities = ladder.maturities_years
    if weights is None:
        weights = ladder.normalised_weights
    if len(weights) != len(maturities):
        raise LadderError("weights and maturities must be the same length")
    if frequency is None:
        frequency = ladder.coupon_frequency
    if delta is None:
        delta = config.roll_down_step

    total = sum(weights)
    blended: pd.Series | None = None
    for weight, maturity in zip(weights, maturities, strict=True):
        capped = min(seasoning, max(max_original_maturity - maturity, 0.0))
        leg = seasoned_bond_returns(
            params,
            maturity,
            capped,
            frequency=frequency,
            delta=delta,
            small_n=small_n,
            mask_unpublished=mask_unpublished,
        )["total_return"] * (weight / total)
        blended = leg if blended is None else blended.add(leg, fill_value=0.0)
    assert blended is not None
    blended.name = f"seasoned_{seasoning:g}y"
    return blended
