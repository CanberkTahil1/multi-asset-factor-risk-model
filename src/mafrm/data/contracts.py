"""Shape assertions every cached table must satisfy, and the daily-bar guard.

SPEC.md 3.5 makes each data trap a test. Three of them are properties of a table
rather than of a source, so they live here and are applied identically to a Fed
curve, a FRED vintage panel, an AQR workbook and a yfinance pull:

* the expected columns are present (the ``Adj Close`` trap: yfinance flipped
  ``auto_adjust`` to ``True`` in a patch release and dropped the column, which
  silently changed the meaning of existing code);
* no column is entirely NaN (a renamed upstream field parses to an empty column
  rather than to an error);
* no date appears twice (the calendar trap: a duplicated index silently doubles
  a day's weight in every EWMA downstream).

Two further contracts are properties of a source rather than of a table:
:func:`check_covers`, which asks whether a series spans the window it will be
used over, and :class:`ReleaseDrift`, which asks whether a maintained file has
stopped publishing -- judged against its own observed release history rather
than against an assumed cadence.

:func:`require_daily_interval` is CLAUDE.md invariant 2 made mechanical. Yahoo
mis-applies dividend adjustments to weekly and monthly bars (yfinance issue
#1273) and inflates returns silently, so every loader that takes an interval
routes it through here and daily is the only value that survives.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final

import pandas as pd

__all__ = [
    "DAILY_INTERVAL",
    "ContractError",
    "IntervalError",
    "Release",
    "ReleaseDrift",
    "check_columns",
    "check_covers",
    "check_frame",
    "check_no_all_nan_columns",
    "check_no_duplicate_dates",
    "collapse_releases",
    "missing_periods",
    "require_daily_interval",
]

#: The only bar size this project ever requests. CLAUDE.md invariant 2.
DAILY_INTERVAL: Final[str] = "1d"


class ContractError(AssertionError):
    """A cached or freshly parsed table violates its data contract."""


class IntervalError(ValueError):
    """A loader was asked for a bar size other than daily. CLAUDE.md invariant 2."""


def require_daily_interval(interval: str) -> str:
    """Return ``interval`` if it is daily, else raise :class:`IntervalError`.

    CLAUDE.md invariant 2 and SPEC.md 3.5. ``1wk`` and ``1mo`` are named in the
    message because they are the two that silently inflate returns rather than
    failing; anything else is refused on the same principle -- this project
    aggregates from daily bars itself so that the aggregation is ours and
    testable.
    """
    if interval != DAILY_INTERVAL:
        raise IntervalError(
            f"interval={interval!r} is refused. CLAUDE.md invariant 2: only "
            f"{DAILY_INTERVAL!r} may be requested. Yahoo mis-applies dividend "
            "adjustments to '1wk' and '1mo' bars (yfinance issue #1273), inflating "
            "returns with no error. Pull daily and aggregate yourself."
        )
    return interval


# ---------------------------------------------------------------------------
# Table shape
# ---------------------------------------------------------------------------


def check_columns(frame: pd.DataFrame, required: Iterable[str], *, label: str) -> None:
    """Every name in ``required`` must be a column. Extra columns are allowed.

    Extras are allowed on purpose: a vendor adding a field is not a breakage,
    a vendor removing one is.
    """
    present = set(map(str, frame.columns))
    missing = [name for name in required if name not in present]
    if missing:
        raise ContractError(
            f"{label}: missing column(s) {missing}; got {sorted(present)}. "
            "The upstream layout changed -- fix the parser, do not relax the contract."
        )


def check_no_all_nan_columns(frame: pd.DataFrame, *, label: str, allow: Sequence[str] = ()) -> None:
    """No column may be entirely NaN.

    An all-NaN column is the signature of a renamed upstream field: the parser
    finds nothing under the old name, pandas fills with NaN, and every statistic
    downstream is computed on an empty series without anything raising.
    """
    allowed = set(allow)
    empty = [str(c) for c in frame.columns if str(c) not in allowed and frame[c].isna().all()]
    if empty:
        raise ContractError(
            f"{label}: column(s) {empty} are entirely NaN over {len(frame):,} rows. "
            "That is what a renamed upstream field looks like."
        )


def check_no_duplicate_dates(frame: pd.DataFrame, *, label: str) -> None:
    """The date axis must be unique.

    Index if it is a ``DatetimeIndex``, else a column called ``date``. A frame
    with neither is not checked -- the caller has already been told which tables
    carry a date axis by :func:`check_frame`'s ``require_dates``.
    """
    axis = _date_axis(frame)
    if axis is None:
        return
    duplicated = axis[axis.duplicated(keep=False)]
    if len(duplicated) == 0:
        return
    sample = sorted({pd.Timestamp(v).date().isoformat() for v in duplicated})[:5]
    raise ContractError(
        f"{label}: {len(duplicated):,} row(s) share a date, first {sample}. "
        "A duplicated date double-weights that day in every EWMA downstream."
    )


def _date_axis(frame: pd.DataFrame) -> pd.Index[Any] | None:
    index = frame.index
    if isinstance(index, pd.DatetimeIndex):
        return pd.Index(index)
    for column in frame.columns:
        if str(column).lower() == "date":
            values: pd.Index[Any] = pd.Index(frame[column])
            return values
    return None


def check_frame(
    frame: pd.DataFrame,
    *,
    label: str,
    required_columns: Iterable[str] = (),
    allow_all_nan: Sequence[str] = (),
    require_dates: bool = True,
    min_rows: int = 1,
) -> None:
    """Run every shape contract at once. The entry point the tests call."""
    if len(frame) < min_rows:
        raise ContractError(f"{label}: {len(frame):,} row(s), expected at least {min_rows:,}")
    check_columns(frame, required_columns, label=label)
    check_no_all_nan_columns(frame, label=label, allow=allow_all_nan)
    if require_dates and _date_axis(frame) is None:
        raise ContractError(
            f"{label}: no date axis -- expected a DatetimeIndex or a 'date' column, "
            f"got index {type(frame.index).__name__} and columns {list(frame.columns)}"
        )
    check_no_duplicate_dates(frame, label=label)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def missing_periods(frame: pd.DataFrame, *, freq: str, label: str) -> pd.PeriodIndex:
    """Periods absent between a series' own first and last observation.

    Interior gaps only -- a series that simply has not started yet, or has
    stopped, is not gappy. What this catches is a parse that dropped rows out of
    the middle, which is invisible in a row count and fatal to anything that
    reindexes.
    """
    axis = _date_axis(frame)
    if axis is None or len(axis) == 0:
        raise ContractError(f"{label}: no dates, so coverage is undefined")
    observed = pd.PeriodIndex(pd.to_datetime(pd.Series(axis)).dropna(), freq=freq)
    if len(observed) == 0:
        raise ContractError(f"{label}: no parseable dates, so coverage is undefined")
    expected = pd.period_range(observed.min(), observed.max(), freq=freq)
    return expected.difference(observed)


def check_covers(
    frame: pd.DataFrame,
    *,
    label: str,
    freq: str,
    start: date,
    end: date | None = None,
) -> None:
    """The series must span ``[start, end]`` with no interior gap at ``freq``.

    ``end`` is optional because the boundary this project actually cares about
    is ``sample.holdout_start``, which is deliberately unset until a human pins
    it (CLAUDE.md invariant 5). The caller passes ``None`` until then, and the
    start-and-gaps half of the contract is enforced regardless.
    """
    axis = _date_axis(frame)
    if axis is None:
        raise ContractError(f"{label}: no date axis, so coverage is undefined")
    stamps = pd.to_datetime(pd.Series(axis), errors="coerce").dropna()
    first = pd.Timestamp(stamps.min()).date()
    last = pd.Timestamp(stamps.max()).date()

    # Compared at the series' own frequency, not by raw date. A monthly series
    # is stamped on the last BUSINESS day of each month, so a file whose first
    # row is 2007-04-30 covers April 2007 in full even though that date is a
    # month after `sample.start` of 2007-04-01. Comparing the dates directly
    # would reject it, and the natural repair -- a few weeks' slack -- would be a
    # tolerance invented to paper over a units mismatch.
    first_period = pd.Period(first, freq=freq)
    last_period = pd.Period(last, freq=freq)

    if first_period > pd.Period(start, freq=freq):
        raise ContractError(
            f"{label}: starts {first} ({first_period}), after the in-sample window opens "
            f"on {start}; the first {pd.Period(start, freq=freq)} of the window is uncovered"
        )
    if end is not None and last_period < pd.Period(end, freq=freq):
        raise ContractError(
            f"{label}: ends {last} ({last_period}), before the in-sample window closes on "
            f"{end}; the window is uncovered from {last_period + 1} onwards"
        )
    gaps = missing_periods(frame, freq=freq, label=label)
    if len(gaps):
        raise ContractError(
            f"{label}: {len(gaps)} interior {freq}-period(s) missing between {first} and "
            f"{last}, first {gaps[0]}. A gap in the middle of a series is a dropped parse, "
            "not a publication schedule."
        )


# ---------------------------------------------------------------------------
# Release drift
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Release:
    """One distinct version of an upstream file, as the manifest recorded it."""

    #: When this content was first cached. The cache is content-addressed, so a
    #: re-pull of unchanged bytes keeps the original stamp and no new release
    #: appears -- which is what makes the manifest a release log for free.
    first_seen: date
    #: Last observation in that version of the file.
    last_observation: date | None


def collapse_releases(releases: Iterable[Release]) -> tuple[Release, ...]:
    """Collapse cache entries that carry the same end date into one release.

    A new dated file appears in the cache whenever the serialized bytes differ,
    and that is not the same thing as an upstream release: upgrading pandas or
    pyarrow rewrites every file under a new digest while the data is untouched.
    Left uncollapsed those rewrites read as releases, the inferred cadence
    shortens, and the detector starts firing on this project's own dependency
    upgrades rather than on the vendor.

    A release is therefore keyed on the **last observation date**, which is a
    property of the data, and the earliest sighting of each is kept. Entries with
    no recorded end date are dropped: they cannot be attributed to a release.
    """
    earliest: dict[date, date] = {}
    for release in releases:
        if release.last_observation is None:
            continue
        seen = earliest.get(release.last_observation)
        if seen is None or release.first_seen < seen:
            earliest[release.last_observation] = release.first_seen
    return tuple(
        Release(first_seen=first_seen, last_observation=observation)
        for observation, first_seen in sorted(earliest.items(), key=lambda kv: kv[1])
    )


@dataclass(frozen=True)
class ReleaseDrift:
    """Has an upstream file stopped publishing, judged against its own history.

    **The cadence is inferred, never assumed.** The W1-P4 contract this replaces
    asserted a 90-day age limit written before anything was known about how often
    AQR republishes, and failed on all three files for that reason. This asks a
    question the data can answer instead: has the file now gone longer without a
    new release than it ever has before, across the releases we have actually
    observed?

    With fewer than two observed releases there is no interval to compare
    against and the detector **cannot fire**. That is the honest state on a first
    pull, and it is why this is not an age check wearing a different name.

    One property worth stating: ``first_seen`` is when *we* noticed a release,
    not when the vendor published it, so the observed intervals include this
    project's own polling lag. That makes the inferred interval an over-estimate
    and the detector conservative about firing, which is the right direction for
    something that will run unattended every week.
    """

    label: str
    releases: tuple[Release, ...]
    today: date

    @property
    def observed_intervals(self) -> tuple[int, ...]:
        """Days between successive releases, as observed."""
        seen = sorted(r.first_seen for r in self.releases)
        return tuple((b - a).days for a, b in itertools.pairwise(seen))

    @property
    def cadence_observable(self) -> bool:
        return len(self.observed_intervals) > 0

    @property
    def longest_observed_interval(self) -> int | None:
        return max(self.observed_intervals) if self.cadence_observable else None

    @property
    def days_since_last_release(self) -> int | None:
        if not self.releases:
            return None
        return (self.today - max(r.first_seen for r in self.releases)).days

    @property
    def last_observation(self) -> date | None:
        if not self.releases:
            return None
        newest = max(self.releases, key=lambda r: r.first_seen)
        return newest.last_observation

    @property
    def fired(self) -> bool:
        longest = self.longest_observed_interval
        elapsed = self.days_since_last_release
        if longest is None or elapsed is None:
            return False
        return elapsed > longest

    def render(self) -> str:
        if not self.releases:
            return f"{self.label}: never cached"
        if not self.cadence_observable:
            return (
                f"{self.label}: 1 release observed (end {self.last_observation}), "
                f"{self.days_since_last_release} day(s) ago. Cadence not yet observable, "
                "so drift cannot be judged -- correct on a first pull."
            )
        verdict = "DRIFT" if self.fired else "ok"
        return (
            f"{self.label}: {len(self.releases)} releases observed, end "
            f"{self.last_observation}, {self.days_since_last_release} day(s) since the last; "
            f"longest previously observed interval {self.longest_observed_interval} day(s) "
            f"-- {verdict}"
        )
