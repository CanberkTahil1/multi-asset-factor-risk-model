"""Average daily dollar volume. SPEC.md 3.4.

ADV is the depth term of the cost model: impact in SPEC.md 7.1 is a function of
trade size *as a fraction of ADV*, so an ADV that is wrong by a factor of two
moves every impact estimate by the same factor.

**A median, never a mean.** SPEC.md 3.4 is explicit and the reason is mechanical:
index-rebalance days, quarterly triple witching and the occasional block print
put single sessions one to two orders of magnitude above the local level. A
63-day mean carries one such day at ~1.6% weight against a level it may exceed
fifty-fold, which is a ~80% overstatement of depth from a single session -- and
in the flattering direction, because a deeper book means a cheaper trade. The
median is unmoved by any minority of such days.

This is the one statistic in the module that is deliberately NOT configurable.
``costs.adv.window`` is in ``config/model.yaml``; the choice of median is not,
because a mean here is a defect rather than an alternative, and a config key
would present it as a setting someone may reasonably flip.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["AdvError", "average_daily_volume", "dollar_volume"]

#: Raw-bar columns this module needs. ``Close`` is the RAW close, matching
#: ``Volume``: dollar volume is what actually changed hands on the day, so a
#: back-adjusted close would restate historical turnover by every subsequent
#: dividend. See SPEC.md 3.5 and mafrm.data.prices.
_VOLUME = "Volume"
_CLOSE = "Close"


class AdvError(ValueError):
    """Bars cannot support a dollar-volume estimate."""


def dollar_volume(prices: pd.DataFrame, *, label: str = "prices") -> pd.Series:
    """Daily dollar volume, ``Volume x Close``, from RAW unadjusted bars."""
    missing = [column for column in (_VOLUME, _CLOSE) if column not in prices.columns]
    if missing:
        raise AdvError(f"{label}: raw bars have no {missing} column(s); got {list(prices.columns)}")

    volume = pd.to_numeric(prices[_VOLUME], errors="coerce")
    close = pd.to_numeric(prices[_CLOSE], errors="coerce")
    if (volume.dropna() < 0).any():
        raise AdvError(f"{label}: negative share volume, which is a parse defect")
    if (close.dropna() <= 0).any():
        raise AdvError(f"{label}: non-positive close, which is a parse defect")

    turnover = volume * close
    turnover.name = "dollar_volume"
    return turnover


def average_daily_volume(
    prices: pd.DataFrame,
    *,
    window: int,
    label: str = "prices",
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling **median** dollar volume over ``window`` trading days.

    ``window`` comes from ``config.model.costs.adv.window``: 252 sessions, SPEC.md
    7.1's "trailing 1-year median dollar volume". It WAS 63, "deliberately equal
    to the EDGE spread window"; that reasoning was withdrawn in W5-P1 (SPEC.md
    7.1.1) -- EDGE's window governs spread-estimation noise, this one governs
    volume stability, and the two have no reason to share a value.

    Windows are counted in **rows, not calendar days**, so a market holiday does
    not shorten the sample. That is only true because the frame is a trading
    calendar to begin with; see :mod:`mafrm.data.calendar`.
    """
    if window < 1:
        raise AdvError(f"{label}: adv window must be positive, got {window}")

    turnover = dollar_volume(prices, label=label)
    required = window if min_periods is None else min_periods
    adv = turnover.rolling(window=window, min_periods=required).median()
    adv.name = "adv"
    return adv
