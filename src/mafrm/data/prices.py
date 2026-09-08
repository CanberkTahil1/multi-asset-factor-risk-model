"""Deterministic total returns and adjusted closes from raw vendor prices.

SPEC.md 3.5, first trap: Yahoo rewrites history backwards every time a dividend
is paid, so two pulls on different dates give different adjusted histories and
every statistic computed on them carries a subtle forward-looking component. The
mitigation is to cache raw prices and a separate actions table, both hashed, and
to apply the adjustment here, deterministically, from data that was knowable on
the day.

Two functions, and the difference between them is the whole point:

:func:`total_return`
    ``r_t = (P_t + D_t) / P_{t-1} - 1``, with the dividend on its ex-date and
    nothing back-adjusted. Appending tomorrow's dividend cannot change
    yesterday's number. **This is what the model consumes.**

:func:`adjusted_close`
    The back-adjusted price series, stamped with the ``as_of`` date whose
    dividend table produced it. Back-adjustment *does* rewrite history -- that is
    what it is for -- so the series is only meaningful next to the date it was
    computed as of, and this function refuses to produce one without it. Use it
    to reconcile against a vendor's ``Adj Close``, not to compute returns.

**Splits.** Yahoo's ``auto_adjust=False`` close is already split-adjusted
backwards, and the dividends in the actions table are quoted on the same basis,
so no split factor belongs here -- applying one would double-count. That is a
claim about a vendor's behaviour rather than a law, so :func:`check_splits_applied`
tests it against the data instead of assuming it: at a genuine 2-for-1 the close
either steps down by half (not adjusted) or does not (adjusted), and the two are
separated by ``log 2`` against a typical daily move of about ``0.01``. Three
funds in the frozen universe exercise this -- IWM 2:1 and EFA 3:1 on 2005-06-09,
EEM 3:1 on 2005-06-09 and again on 2008-07-24.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final

import numpy as np
import pandas as pd

__all__ = [
    "AdjustedCloses",
    "PricesError",
    "SplitCheck",
    "adjusted_close",
    "adjustment_factors",
    "check_splits_applied",
    "total_return",
]

#: Decision boundary between "the vendor already applied the split" and "it did
#: not", as a fraction of ``|log(ratio)|``. NOT a tunable and deliberately not in
#: config/model.yaml: it does not change any result, it chooses between two
#: mutually exclusive vendor behaviours whose signatures are ``0`` and
#: ``log(ratio) >= log 2 = 0.69`` apart, against a daily move of order 0.01. Any
#: boundary in (0.05, 0.95) classifies every split in the frozen universe
#: identically. It is pinned here for the same reason as the parquet constants in
#: ``cache.py`` -- see the W1-P2 note in experiments.md.
_SPLIT_APPLIED_FRACTION: Final[float] = 0.5


class PricesError(RuntimeError):
    """Raw prices and actions do not support a defensible total return."""


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitCheck:
    """One split, and what the close did across it."""

    ex_date: date
    ratio: float
    log_return: float

    @property
    def expected_log_return_if_unadjusted(self) -> float:
        return -float(np.log(self.ratio))

    @property
    def already_applied(self) -> bool:
        """True when the close did *not* step by the split ratio."""
        threshold = _SPLIT_APPLIED_FRACTION * abs(self.expected_log_return_if_unadjusted)
        return abs(self.log_return) < threshold

    def render(self) -> str:
        verdict = "already applied" if self.already_applied else "NOT APPLIED"
        return (
            f"{self.ex_date.isoformat()} {self.ratio:g}-for-1: close log-return "
            f"{self.log_return:+.4f} against {self.expected_log_return_if_unadjusted:+.4f} "
            f"if unadjusted -- {verdict}"
        )


def _split_events(actions: pd.DataFrame) -> pd.Series:
    splits = actions.get("Stock Splits")
    if splits is None:
        return pd.Series(dtype=float)
    numeric = pd.to_numeric(splits, errors="coerce")
    return numeric[(numeric.notna()) & (numeric > 0.0) & (numeric != 1.0)]


def check_splits_applied(
    prices: pd.DataFrame, actions: pd.DataFrame, *, column: str = "Close"
) -> tuple[SplitCheck, ...]:
    """Verify the vendor's close is already split-adjusted, or raise.

    Returns one :class:`SplitCheck` per split so a caller can report them. A
    series with no splits returns an empty tuple and is trivially consistent.
    """
    events = _split_events(actions)
    if events.empty:
        return ()

    closes = prices[column].astype(float).sort_index()
    index = pd.DatetimeIndex(closes.index)
    log_close = np.log(closes.to_numpy(dtype=float))
    checks: list[SplitCheck] = []
    for stamp, ratio in zip(
        pd.DatetimeIndex(events.index), events.to_numpy(dtype=float), strict=True
    ):
        moment = pd.Timestamp(stamp)
        if moment not in index:
            raise PricesError(
                f"split on {moment.date()} has no matching row in the raw prices; the "
                "actions table and the price table came from different pulls"
            )
        position = int(index.get_indexer(pd.DatetimeIndex([moment]))[0])
        if position == 0:
            # A split on the first cached bar has no predecessor to compare
            # against; there is nothing to double-count either, so skip it.
            continue
        checks.append(
            SplitCheck(
                ex_date=moment.date(),
                ratio=float(ratio),
                log_return=float(log_close[position] - log_close[position - 1]),
            )
        )

    unapplied = [check for check in checks if not check.already_applied]
    if unapplied:
        detail = "\n  ".join(check.render() for check in unapplied)
        raise PricesError(
            f"{len(unapplied)} split(s) appear NOT to be reflected in the {column!r} series:\n  "
            f"{detail}\n"
            "This module assumes the vendor's raw close is split-adjusted backwards and its "
            "dividends are quoted on the same basis, so it applies no split factor. If the "
            "vendor changed, the adjustment has to be applied here rather than assumed away."
        )
    return tuple(checks)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def total_return(
    prices: pd.DataFrame, actions: pd.DataFrame, *, column: str = "Close"
) -> pd.Series:
    """Daily total return from raw closes and an ex-date dividend table.

    ``r_t = (P_t + D_t) / P_{t-1} - 1`` with ``D_t`` the cash dividend on its
    ex-date. Nothing is back-adjusted, so appending tomorrow's dividend cannot
    change yesterday's number -- which is the property SPEC.md 3.5 is after.

    Splits are checked rather than assumed: see :func:`check_splits_applied`.
    """
    if column not in prices.columns:
        raise PricesError(f"raw prices have no {column!r} column; got {list(prices.columns)}")

    closes = prices[column].astype(float).sort_index()
    check_splits_applied(prices, actions, column=column)

    dividends = _dividends(actions, closes.index)
    returns = (closes + dividends) / closes.shift(1) - 1.0
    returns.name = "total_return"
    return returns.iloc[1:]


def _dividends(actions: pd.DataFrame, index: pd.Index) -> pd.Series:
    if "Dividends" not in actions.columns:
        return pd.Series(0.0, index=index)
    cash = pd.to_numeric(actions["Dividends"], errors="coerce")
    return cash.reindex(index).fillna(0.0).astype(float)


# ---------------------------------------------------------------------------
# Back-adjustment, as of a stated date
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdjustedCloses:
    """A back-adjusted price series and the date whose dividend table made it.

    The ``as_of`` stamp is not decoration. Back-adjustment folds every dividend
    after date ``t`` into the price at ``t``, so the series at ``t`` depends on
    the future and the only way to say what it depends on is to name the cut-off.
    Two ``AdjustedCloses`` with different ``as_of`` values are different series
    even when built from identical raw prices.
    """

    as_of: date
    close: pd.Series
    factor: pd.Series
    n_dividends: int

    def render(self) -> str:
        return (
            f"adjusted close as of {self.as_of.isoformat()}: {len(self.close):,} bars, "
            f"{self.n_dividends} dividend(s) folded in, "
            f"factor {float(self.factor.iloc[0]):.6f}..{float(self.factor.iloc[-1]):.6f}"
        )


def adjustment_factors(
    prices: pd.DataFrame, actions: pd.DataFrame, *, as_of: date, column: str = "Close"
) -> pd.Series:
    """Backwards dividend-adjustment factors using only ex-dates on or before ``as_of``.

    ``f_t = prod over ex-dates d > t, d <= as_of of (1 - D_d / P_{d-1})``, the
    standard CRSP-style back-adjustment. ``f`` is 1.0 at the last bar and decays
    going backwards, so today's price is unchanged and history is scaled -- which
    is exactly the rewrite SPEC.md 3.5 warns about, made explicit and dated
    rather than inherited from a vendor.
    """
    if column not in prices.columns:
        raise PricesError(f"raw prices have no {column!r} column; got {list(prices.columns)}")

    closes = prices[column].astype(float).sort_index()
    check_splits_applied(prices, actions, column=column)

    cutoff = pd.Timestamp(as_of)
    if cutoff < closes.index[0]:
        raise PricesError(
            f"as_of {as_of.isoformat()} precedes the first bar {closes.index[0].date()}"
        )

    cash = _dividends(actions, closes.index)
    cash = cash.where(cash.index <= cutoff, 0.0)
    # Ratio of the ex-date drop: 1 - D_d / P_{d-1}. The previous close is the
    # cum-dividend price, so this is the fraction of value that stayed in the
    # share rather than leaving as cash.
    ratio = 1.0 - (cash / closes.shift(1)).fillna(0.0)
    # f_t is the product of the ratios STRICTLY AFTER t, so shift up before the
    # reverse cumulative product; the last bar has nothing after it and gets 1.0.
    forward = ratio.shift(-1).fillna(1.0)
    factor = forward[::-1].cumprod()[::-1]
    factor.name = "adjustment_factor"
    return factor


def adjusted_close(
    prices: pd.DataFrame, actions: pd.DataFrame, *, as_of: date, column: str = "Close"
) -> AdjustedCloses:
    """Dividend-back-adjusted closes, stamped with the ``as_of`` date.

    For reconciliation against a vendor's ``Adj Close`` -- not for computing
    returns. Use :func:`total_return` for that: it needs no cut-off date because
    it never looks forward.
    """
    closes = prices[column].astype(float).sort_index()
    factor = adjustment_factors(prices, actions, as_of=as_of, column=column)
    cash = _dividends(actions, closes.index)
    counted = cash[(cash > 0.0) & (cash.index <= pd.Timestamp(as_of))]
    series = (closes * factor).rename("adjusted_close")
    return AdjustedCloses(
        as_of=as_of, close=series, factor=factor, n_dividends=int(counted.count())
    )
