"""One canonical trading calendar, and the only sanctioned way to align onto it.

CLAUDE.md failure mode 1 is the top source of "my Sharpe is 2.4" moments, and it
is not one bug but a family:

* ``.resample("ME").last()`` stamps an observation on the calendar month end --
  the 31st -- whether or not anything traded that day, so two series with
  different holidays are silently declared contemporaneous.
* ``.resample("BME").last()`` stamps on the last *business* day, which is a
  market holiday often enough to matter (Good Friday is the canonical case, and
  W1-P3 found the Fed's curve files carrying an all-NA row for exactly that day).
* ``reindex(...).ffill()`` invents a return on a date an asset did not trade,
  which is the same error dressed as convenience: the carried price contributes
  a zero return to a day the market was shut and a real return to the day after,
  compressing measured volatility and inflating measured Sharpe.

The repair is to stop treating "the end of the month" as a date and treat it as
**the last date on which the asset actually traded in that month**. That is what
:func:`month_end_dates` returns and what every monthly series in this project is
stamped on. Where a calendar month end *was* a trading day the two agree, and
:func:`monthly` is then identical to ``resample("ME").last()``; where it was not,
they differ and this module follows the trading date. A test pins both halves of
that statement, because the day the two silently diverge is the day a downstream
statistic starts measuring the calendar instead of the market.

**Alignment never fills.** :func:`align` puts several assets on one index and
leaves ``NaN`` wherever an asset did not trade. Filling is a modelling decision
that belongs to whatever consumes the panel, stated there explicitly, not a
default buried in the loader.

This module is pure: it reads no files, holds no configuration and imports
nothing from the rest of the project. It sits in ``data/`` because alignment is a
property of the data layer, but it makes no network call and asks no question
about what asset class it is looking at.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Literal

import pandas as pd

__all__ = [
    "CalendarError",
    "align",
    "month_end_dates",
    "monthly",
    "trading_dates",
]

#: How the union/intersection of several assets' calendars is taken.
How = Literal["union", "intersection"]


class CalendarError(ValueError):
    """A frame cannot be placed on a trading calendar."""


def _index_of(obj: pd.Series | pd.DataFrame, *, label: str) -> pd.DatetimeIndex:
    """The object's index as a clean, sorted, unique ``DatetimeIndex``.

    Every failure here is a real defect rather than something to normalise away
    quietly: a duplicated date means two bars for one session, and an unsorted
    index means a parser emitted rows in file order rather than date order.
    """
    index = obj.index
    if not isinstance(index, pd.DatetimeIndex):
        try:
            index = pd.DatetimeIndex(index)
        except (TypeError, ValueError) as exc:
            raise CalendarError(f"{label}: index is not dates ({exc})") from exc
    if index.hasnans:
        raise CalendarError(f"{label}: index contains NaT, so the calendar is undefined")
    if index.tz is not None:
        raise CalendarError(
            f"{label}: index is timezone-aware ({index.tz}). Bars are dated observations, "
            "not instants; a tz turns a date comparison into a clock comparison and shifts "
            "a session across a day boundary."
        )
    normalised = index.normalize()
    if normalised.has_duplicates:
        dupes = normalised[normalised.duplicated()].unique()
        raise CalendarError(
            f"{label}: {len(dupes)} duplicated date(s), first {dupes[0].date()}. Two bars for "
            "one session is a parse defect, not something to average."
        )
    if not normalised.is_monotonic_increasing:
        raise CalendarError(
            f"{label}: index is not sorted ascending. A parser emitted rows in file order; "
            "sorting here would hide it."
        )
    return normalised


def trading_dates(obj: pd.Series | pd.DataFrame, *, label: str = "series") -> pd.DatetimeIndex:
    """The dates this asset actually traded.

    A row is a session. Rows that are entirely missing are not sessions -- the
    Fed's curve files carry a row for every weekday including market holidays,
    with every field NA, and W1-P3 had to drop them before the roll-down carry
    was honest. Anything all-NA is dropped here for the same reason.
    """
    index = _index_of(obj, label=label)
    frame = obj.to_frame() if isinstance(obj, pd.Series) else obj
    populated = frame.notna().any(axis=1).to_numpy()
    return index[populated]


def month_end_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last traded date within each calendar month present in ``dates``.

    This is the project's definition of "month end" and the reason the rest of
    the codebase never calls ``resample`` directly. The result is a subset of
    ``dates``, so every monthly observation is stamped on a date the market was
    genuinely open.
    """
    if len(dates) == 0:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    frame = pd.DataFrame({"date": dates}, index=dates)
    last = frame.groupby(dates.to_period("M"), sort=True)["date"].max()
    return pd.DatetimeIndex(last.to_numpy(), name=dates.name)


def monthly(
    obj: pd.Series | pd.DataFrame,
    *,
    label: str = "series",
    stamp: Literal["observation", "period_end"] = "observation",
) -> pd.Series | pd.DataFrame:
    """Sample ``obj`` at each month's last actually-traded date.

    ``stamp="observation"`` (the default) keeps the trading date on the index,
    which is what makes the result joinable against daily data without inventing
    a session. ``stamp="period_end"`` relabels to the calendar month end, which
    is what a reader expects on a chart axis; it is a *presentation* choice and
    is never used to join two series, because that is precisely the collision
    this module exists to prevent.
    """
    dates = trading_dates(obj, label=label)
    picks = month_end_dates(dates)
    sampled = obj.reindex(picks)
    if stamp == "period_end":
        sampled.index = pd.DatetimeIndex(picks).to_period("M").to_timestamp(how="end").normalize()
    return sampled


def align(
    series: Mapping[str, pd.Series],
    *,
    how: How = "union",
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Put several assets on one calendar, **without filling**.

    ``how="union"`` keeps every date any asset traded and leaves ``NaN`` where a
    given asset did not; ``how="intersection"`` keeps only dates every asset
    traded, which is what a covariance estimate needs and what silently discards
    the most data, so it is never the default.

    ``end`` is where CLAUDE.md invariant 5 is enforced in practice: callers pass
    ``config.require_holdout_start()`` and the panel cannot reach past it.
    """
    if not series:
        raise CalendarError("align: no series given, so there is no calendar to build")

    calendars = {name: trading_dates(item, label=name) for name, item in series.items()}
    empty = [name for name, dates in calendars.items() if len(dates) == 0]
    if empty:
        raise CalendarError(f"align: {', '.join(sorted(empty))} contributed no traded dates")

    index: pd.DatetimeIndex | None = None
    for dates in calendars.values():
        if index is None:
            index = dates
        elif how == "union":
            index = index.union(dates)
        else:
            index = index.intersection(dates)
    assert index is not None  # non-empty mapping, checked above
    if how == "intersection" and len(index) == 0:
        raise CalendarError(
            "align: the assets share no common trading date. Check that one of them is not "
            "quoted on a different exchange calendar."
        )

    if start is not None:
        index = index[index >= pd.Timestamp(start)]
    if end is not None:
        # Half-open on purpose: `end` is the holdout boundary and the holdout
        # begins ON it (CLAUDE.md invariant 5), so the boundary date is excluded.
        index = index[index < pd.Timestamp(end)]

    out = pd.DataFrame(index=index)
    for name, item in series.items():
        # `reindex` and not `ffill`: a date this asset did not trade stays NaN.
        out[name] = item.reindex(index)
    out.index.name = "date"
    return out
