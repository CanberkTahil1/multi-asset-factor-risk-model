"""FRED and ALFRED: parsing, vintage coverage, and cache access.

SPEC.md 3.5, the FRED-revisions trap: "Macro series get revised. Using current
values as factors is look-ahead. Use **ALFRED** vintages
(``realtime_start``/``realtime_end`` in the FRED API). Check vintage coverage per
series; drop or conservatively lag anything without vintages."

Two series are therefore cached per FRED id, and they are not interchangeable:

``<id>``
    The **current** series -- every observation at its latest revision. Correct
    for describing history, look-ahead for trading it.

``<id>_first_release``
    The **initial release** of each observation (``output_type=4``), carrying the
    ``realtime_start`` on which that value first became public. This is the
    look-ahead-free series and the only one a factor input may read.

The API, not ``fredgraph.csv``. The keyless CSV endpoint serves only the current
vintage, so a project that mixes the two ends up with two different numbers for
the same date and no way to tell which is in any given result. One endpoint, one
number.

**Vintage coverage is partial and that is the point of measuring it.** FRED began
recording real-time periods for a series when ALFRED took it on, not when the
series began: ``DTWEXBGS`` runs from 2006 but its first vintage is 2019, so more
than half its history has no first-release value at all. :func:`coverage` reports
that fraction so the decision SPEC.md asks for -- drop, or lag conservatively --
is made against a number instead of an assumption.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final, Literal

import pandas as pd

from mafrm.data import cache

__all__ = [
    "FIRST_RELEASE_SUFFIX",
    "FredError",
    "NoVintagesError",
    "Release",
    "VintageCoverage",
    "cache_name",
    "coverage",
    "first_releases",
    "load_coverage",
    "load_series",
    "load_vintage_dates",
    "parse_observations",
    "parse_vintage_dates",
]

#: FRED writes a missing observation as a single period, not as an empty field.
_MISSING: Final[str] = "."

#: Cache-name suffix for the initial-release series.
FIRST_RELEASE_SUFFIX: Final[str] = "_first_release"

#: Cache-name suffix for the vintage-date list.
_VINTAGE_SUFFIX: Final[str] = "_vintage_dates"

Release = Literal["current", "first_release"]


class FredError(RuntimeError):
    """A FRED payload is not the shape the API documents."""


class NoVintagesError(FredError):
    """A series has no ALFRED vintages, so it cannot be used look-ahead-free.

    Raised at load time on purpose. A series with no vintage record is not a
    smaller problem than a missing series -- it is a series whose every value is
    the current revision, which reaches a backtest as free information about the
    future and produces a result that is wrong in the flattering direction.
    """


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _payload(raw: bytes | str | dict[str, Any], *, what: str) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FredError(f"{what}: response is not JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise FredError(f"{what}: expected a JSON object, got {type(parsed).__name__}")
    if "error_message" in parsed:
        raise FredError(f"{what}: FRED returned an error: {parsed['error_message']}")
    return parsed


def parse_observations(raw: bytes | str | dict[str, Any], series_id: str) -> pd.DataFrame:
    """``fred/series/observations`` -> a dated frame.

    Columns are ``value`` (float, ``.`` becomes NaN), ``realtime_start``
    (datetime) and ``realtime_end`` (**string**). They are kept for both releases
    even though they are constant for the current series: dropping them would
    make the two cached tables differently shaped for no gain, and on the
    first-release table ``realtime_start`` is the whole point -- it is the date
    the value became knowable.

    ``realtime_end`` stays a string because FRED writes ``9999-12-31`` there to
    mean "still current", and that is outside pandas' nanosecond ``Timestamp``
    range. Parsing it with ``errors="coerce"`` turns the sentinel into ``NaT``,
    which is silent, indistinguishable from a genuinely missing value, and hit
    1,151 of DGS1MO's 5,524 initial releases -- exactly the class of loss
    SPEC.md 3.5 is about. Verbatim text keeps the sentinel legible.
    """
    payload = _payload(raw, what=f"{series_id} observations")
    rows = payload.get("observations")
    if not isinstance(rows, list):
        raise FredError(f"{series_id} observations: no 'observations' array in the response")
    if not rows:
        raise FredError(f"{series_id} observations: the response is empty")

    dates: list[Any] = []
    values: list[float] = []
    realtime_start: list[Any] = []
    realtime_end: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise FredError(f"{series_id} observations[{index}]: expected an object")
        for key in ("date", "value", "realtime_start", "realtime_end"):
            if key not in row:
                raise FredError(f"{series_id} observations[{index}]: missing {key!r}")
        dates.append(row["date"])
        text = str(row["value"]).strip()
        values.append(float("nan") if text == _MISSING or not text else float(text))
        realtime_start.append(row["realtime_start"])
        realtime_end.append(str(row["realtime_end"]))

    frame = pd.DataFrame(
        {
            "value": values,
            "realtime_start": pd.to_datetime(realtime_start, errors="raise"),
            "realtime_end": realtime_end,
        },
        index=pd.DatetimeIndex(pd.to_datetime(dates, errors="raise"), name="date"),
    )
    return frame.sort_index()


def parse_vintage_dates(raw: bytes | str | dict[str, Any], series_id: str) -> tuple[date, ...]:
    """``fred/series/vintagedates`` -> the dates on which the series was revised."""
    payload = _payload(raw, what=f"{series_id} vintage dates")
    dates = payload.get("vintage_dates")
    if not isinstance(dates, list):
        raise FredError(f"{series_id} vintage dates: no 'vintage_dates' array in the response")
    return tuple(date.fromisoformat(str(item)) for item in dates)


def first_releases(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Stitch chunked ``output_type=4`` responses into one initial-release table.

    FRED caps one initial-release request at a fixed number of vintage dates, so
    a long daily series arrives as several windows. Within a window, "initial
    release" means initial *relative to that window*: an observation first
    published before the window opens is reported again at the window's start.
    Keeping the row with the **earliest** ``realtime_start`` per observation date
    therefore recovers the true first release, and taking anything else -- the
    last row, or a plain concatenation -- would silently date part of the series
    to when it was re-reported rather than to when it was knowable.
    """
    if not frames:
        raise FredError("no observation windows to stitch")
    combined = pd.concat(frames)
    combined = combined.sort_values("realtime_start", kind="stable")
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined.sort_index()


# ---------------------------------------------------------------------------
# Vintage coverage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VintageCoverage:
    """How much of a series ALFRED can reconstruct without look-ahead."""

    series_id: str
    n_vintages: int
    first_vintage: date | None
    last_vintage: date | None
    observation_start: date | None
    observation_end: date | None
    n_observations: int
    n_first_release: int

    @property
    def fraction(self) -> float:
        """Share of observations that have a recorded initial release."""
        if self.n_observations == 0:
            return 0.0
        return self.n_first_release / self.n_observations

    @property
    def has_vintages(self) -> bool:
        return self.n_vintages > 0

    @property
    def is_complete(self) -> bool:
        """Every observation has a first release, so no lag decision is owed."""
        return self.has_vintages and self.n_first_release >= self.n_observations

    def render(self) -> str:
        if not self.has_vintages:
            return f"{self.series_id}: NO VINTAGES -- current revisions only, look-ahead"
        return (
            f"{self.series_id}: {self.n_vintages:,} vintages "
            f"{_iso(self.first_vintage)}..{_iso(self.last_vintage)}; "
            f"{self.n_first_release:,}/{self.n_observations:,} observations have a first "
            f"release ({self.fraction:.1%}); observations "
            f"{_iso(self.observation_start)}..{_iso(self.observation_end)}"
        )

    def require_vintages(self) -> VintageCoverage:
        """Raise :class:`NoVintagesError` if ALFRED has no real-time record."""
        if not self.has_vintages:
            raise NoVintagesError(
                f"{self.series_id} has no ALFRED vintages, so every value would enter the "
                "model at its latest revision -- look-ahead (SPEC.md 3.5). Drop the series, "
                "or lag it conservatively and say so in the config."
            )
        return self


def _iso(value: date | None) -> str:
    return value.isoformat() if value is not None else "?"


def coverage(
    *,
    series_id: str,
    vintage_dates: Sequence[date],
    current: pd.DataFrame,
    first_release: pd.DataFrame | None,
) -> VintageCoverage:
    """Assemble a :class:`VintageCoverage` from the three cached artefacts."""
    observed = pd.DatetimeIndex(current.index)
    released = (
        pd.DatetimeIndex([]) if first_release is None else pd.DatetimeIndex(first_release.index)
    )
    return VintageCoverage(
        series_id=series_id,
        n_vintages=len(vintage_dates),
        first_vintage=min(vintage_dates) if vintage_dates else None,
        last_vintage=max(vintage_dates) if vintage_dates else None,
        observation_start=observed.min().date() if len(observed) else None,
        observation_end=observed.max().date() if len(observed) else None,
        n_observations=len(observed),
        n_first_release=len(released.intersection(observed)),
    )


# ---------------------------------------------------------------------------
# Cache access. Never fetches -- CLAUDE.md invariant 1.
# ---------------------------------------------------------------------------


def cache_name(series_id: str, release: Release = "current") -> str:
    """Manifest ``name`` for one FRED series and release."""
    base = series_id.lower()
    return base if release == "current" else f"{base}{FIRST_RELEASE_SUFFIX}"


def load_series(series_id: str, release: Release = "current") -> pd.DataFrame:
    """Read a cached FRED series. ``mafrm.data.loaders`` is what fetches it."""
    manifest = cache.Manifest.load()
    entry = manifest.latest(source="fred", name=cache_name(series_id, release))
    return cache.read(entry, manifest=manifest)


def load_vintage_dates(series_id: str) -> tuple[date, ...]:
    """Read the cached ALFRED vintage-date list for one series."""
    manifest = cache.Manifest.load()
    entry = manifest.latest(source="fred", name=f"{series_id.lower()}{_VINTAGE_SUFFIX}")
    frame = cache.read(entry, manifest=manifest)
    return tuple(pd.Timestamp(value).date() for value in frame["vintage_date"])


def load_coverage(series_id: str) -> VintageCoverage:
    """Vintage coverage for one series, entirely from the cache."""
    try:
        first_release: pd.DataFrame | None = load_series(series_id, "first_release")
    except cache.CacheError:
        first_release = None
    return coverage(
        series_id=series_id,
        vintage_dates=load_vintage_dates(series_id),
        current=load_series(series_id, "current"),
        first_release=first_release,
    )
