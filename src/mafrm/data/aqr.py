"""AQR's public research workbooks: three files, three different layouts.

SPEC.md 3.1 uses *Commodities for the Long Run* for the pre-2006 commodity
splice; *Century of Factor Premia* and the *TSMOM Factors* file are reference
series for the long-history sensitivity run. All three are ``.xlsx`` with a
banner of copyright and description rows above the table, and **the banner is a
different height in each file**:

===============================  ================================  ======  ===============
File                             Sheet                             Header  Date column
===============================  ================================  ======  ===============
Century of Factor Premia         ``Century of Factor Premia``       row 18  ``Date``
Time Series Momentum Factors     ``TSMOM Factors``                  row 17  *(blank)*
Commodities for the Long Run     ``Commodities for the Long Run``   row 10  *(blank)*
===============================  ================================  ======  ===============

So each file gets an explicit :class:`WorkbookSpec` from ``config/model.yaml``
rather than a heuristic that searches for the first row that looks like a
header. A heuristic here fails silently: AQR's banner text contains dates, and a
sniffer that guesses one row too high produces a table whose first data row is
prose, which parses to NaN and disappears into an all-NaN column. Naming the row
means a layout change raises instead.

The date column is a further inconsistency worth stating rather than smoothing
over. *Century* labels it ``Date`` and writes ``MM/DD/YYYY`` strings; the other
two leave the header cell empty, so pandas names it ``Unnamed: 0``, and their
values arrive as a mix of ``Timestamp`` and ISO strings within a single column.
:func:`parse_workbook` therefore takes the date column by position when the
config names none, and coerces explicitly.

**Licensing.** AQR publishes these for research use and states no redistribution
licence, so the files are cached to a gitignored directory and never committed
(CLAUDE.md invariant 8). Only ``data/manifest.json`` records them.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Final

import pandas as pd

from mafrm.data import cache, contracts

__all__ = [
    "SOURCE",
    "AqrError",
    "WorkbookSpec",
    "load_dataset",
    "parse_workbook",
    "release_drift",
]

#: Manifest ``source`` for every AQR file.
SOURCE: Final[str] = "aqr"


class AqrError(RuntimeError):
    """An AQR workbook is not the shape ``config/model.yaml`` describes."""


@dataclass(frozen=True)
class WorkbookSpec:
    """Where the table lives inside one workbook. Mirrors ``data.aqr.datasets``."""

    name: str
    url: str
    sheet: str
    #: Zero-based row index of the header, i.e. what ``pd.read_excel(header=)`` takes.
    header_row: int
    #: Header text of the date column, or ``None`` when AQR left the cell blank.
    date_column: str | None
    required_columns: tuple[str, ...]
    description: str


def parse_workbook(payload: bytes, spec: WorkbookSpec) -> pd.DataFrame:
    """One AQR ``.xlsx`` -> a dated frame, or raise.

    The result is indexed by ``date`` with the date column removed, every
    remaining column coerced to float, and rows whose date did not parse
    dropped -- AQR pads the bottom of some sheets with footnote rows, and a
    footnote is not an observation.
    """
    try:
        raw = pd.read_excel(
            io.BytesIO(payload), sheet_name=spec.sheet, header=spec.header_row, engine="openpyxl"
        )
    except ValueError as exc:  # pandas raises ValueError for a missing sheet
        raise AqrError(f"{spec.name}: cannot read sheet {spec.sheet!r}: {exc}") from exc
    if raw.empty:
        raise AqrError(f"{spec.name}: sheet {spec.sheet!r} is empty below row {spec.header_row}")

    if spec.date_column is None:
        # AQR left the header cell blank, so pandas invented `Unnamed: 0`. Taking
        # it by position rather than by that generated name means a future file
        # that *does* label the column still parses.
        date_column = str(raw.columns[0])
    else:
        date_column = spec.date_column
        if date_column not in raw.columns:
            raise AqrError(
                f"{spec.name}: no {date_column!r} column at header row {spec.header_row}; "
                f"got {list(raw.columns)[:6]}... The banner height changed."
            )

    dates = pd.to_datetime(raw[date_column], errors="coerce", format="mixed")
    frame = raw.drop(columns=[date_column])
    frame.index = pd.DatetimeIndex(dates, name="date")
    frame = frame.loc[frame.index.notna()]
    if frame.empty:
        raise AqrError(
            f"{spec.name}: no row under {date_column!r} parsed as a date. The header row "
            f"({spec.header_row}) is probably wrong, so the table is prose."
        )

    frame = _coerce_numeric_where_possible(frame)
    # Drop trailing footnote rows: a row with no value anywhere is not an
    # observation. Interior gaps are kept -- a factor that has not started yet is
    # genuinely missing and must stay missing.
    frame = frame.loc[frame.notna().any(axis=1)]
    frame = frame.sort_index()

    contracts.check_frame(
        frame,
        label=f"aqr/{spec.name}",
        required_columns=spec.required_columns,
        min_rows=2,
    )
    return frame


def _coerce_numeric_where_possible(frame: pd.DataFrame) -> pd.DataFrame:
    """Numeric columns to float; leave genuinely categorical columns alone.

    Not every AQR column is a number. *Commodities for the Long Run* carries
    ``State of backwardation/contango`` and ``State of inflation``, which are
    regime labels, and a blanket ``to_numeric(errors="coerce")`` turns all 1,780
    rows of each into NaN -- which then trips the all-NaN contract and looks
    exactly like a renamed upstream field. So a conversion that would destroy
    every value in a column is rejected as evidence that the column was never
    numeric, and the original is kept.
    """
    columns: dict[str, pd.Series] = {}
    for name in frame.columns:
        original = frame[name]
        converted = pd.to_numeric(original, errors="coerce")
        had_values = bool(original.notna().any())
        lost_everything = bool(converted.isna().all())
        columns[str(name)] = original if (had_values and lost_everything) else converted
    return pd.DataFrame(columns, index=frame.index)


# ---------------------------------------------------------------------------
# Cache access. Never fetches -- CLAUDE.md invariant 1.
# ---------------------------------------------------------------------------


def load_dataset(name: str) -> pd.DataFrame:
    """Read one cached AQR table by its config key."""
    manifest = cache.Manifest.load()
    entry = manifest.latest(source=SOURCE, name=name)
    return cache.read(entry, manifest=manifest)


def release_drift(name: str, *, today: date | None = None) -> contracts.ReleaseDrift:
    """Release history for one AQR file, read straight out of the manifest.

    The cache is content-addressed and dates a raw file by its pull date, so a
    new manifest entry appears only when the cached bytes actually changed. That
    makes the manifest a release log at no extra cost, and means the detector
    needs no state file of its own to maintain or to keep in sync.

    Entries are collapsed by end date first -- see
    :func:`mafrm.data.contracts.collapse_releases`. A serializer upgrade rewrites
    every file under a new digest, and counting that as a release would make the
    detector fire on this project's own dependency bumps.
    """
    manifest = cache.Manifest.load()
    entries = tuple(
        contracts.Release(
            first_seen=datetime.fromisoformat(entry.downloaded_at.replace("Z", "+00:00")).date(),
            last_observation=(date.fromisoformat(entry.last_date) if entry.last_date else None),
        )
        for entry in manifest.find(stage="raw", source=SOURCE, name=name)
    )
    # Collapsed by end date: a pandas or pyarrow upgrade rewrites every cached
    # file under a new digest and a new pull date, which is not an upstream
    # release and must not shorten the inferred cadence.
    releases = contracts.collapse_releases(entries)
    # UTC, because the manifest's `downloaded_at` stamps are UTC. Comparing them
    # against a local `date.today()` west of Greenwich yields a negative elapsed
    # time on the day of a pull, which is harmless but reads as a bug.
    return contracts.ReleaseDrift(
        label=f"aqr/{name}", releases=releases, today=today or datetime.now(UTC).date()
    )
