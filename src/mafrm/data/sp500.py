"""Approximate point-in-time S&P 500 membership. SPEC.md 15.3, rulings at 15.3.1.

Two Wikipedia tables go in -- the current constituent list with its
``Date added`` column, and the *Historical components* changes table -- and a
membership **interval** table comes out, with a **daily** membership matrix on
trading sessions derived from it. Both are committed under ``data/reference/``
as a licensed derived table (CC BY-SA 4.0, SPEC.md 15.3.1 ruling 3), never
under ``data/raw/``.

**This is a genuine attempt at survivorship control, not a solution.** The
changes table is incomplete: it carries a handful of rows a year before 2007
against 16-30 a year from 2011, so every addition or removal Wikipedia did not
record is inherited by the walk. Nothing here selects a universe by a
procedure this project did not run; the gaps are counted per year and
reported, not patched.

**The walk.** Start from the current list and go backwards through the changes
table: before each row's effective date, its added tickers were not members
and its removed tickers were. An added ticker that is not in the running set,
or a removed ticker that already is, cannot both be true of a complete table
-- it is the signature of a missing later row or of a ticker rename -- and is
recorded as an :class:`Inconsistency`, per year, rather than resolved.

**One reconciliation, deterministic and counted.** A current member whose
``Date added`` has no addition row under its own ticker is linked to the
addition row on that exact date if, and only if, exactly one such member and
exactly one such row share the date (Facebook was added as ``FB`` and is
listed today as ``META``). Anything ambiguous stays a gap.

**Three states, per session** (ruling 5): ``0`` not a member, ``1`` member
with a valid bar in the cache on that session, ``2`` member with no valid bar
on that session -- which covers a ticker Yahoo has dropped entirely, a ticker
Yahoo has reassigned to a later company, and a hole in a history it does
carry. The count of ``2`` is the survivorship control's measured coverage and
is reported as a number, not a caveat. The matrix is daily because the changes
table gives dates; the pipeline reads month ends off it and invents no
calendar (ruling 4).

CLAUDE.md invariant 1: nothing here fetches. The HTML comes from
:func:`mafrm.data.loaders.fetch_sp500_wikipedia` through the cache; this module
parses bytes it is handed and the tables the cache returns.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, Tag

__all__ = [
    "CHANGES_NAME",
    "CONSTITUENTS_NAME",
    "MEMBER",
    "MEMBER_NO_DATA",
    "NOT_MEMBER",
    "SOURCE",
    "GapCounts",
    "Inconsistency",
    "Provenance",
    "Reconstruction",
    "SP500Error",
    "gap_table",
    "load_intervals",
    "load_membership",
    "member_on",
    "membership_matrix",
    "parse_changes",
    "parse_constituents",
    "reconstruct",
    "render_intervals_csv",
    "render_membership_csv",
    "to_vendor_symbol",
    "universe_tickers",
]

#: Manifest ``source`` for both Wikipedia pulls.
SOURCE: Final[str] = "wikipedia"
#: Manifest ``name`` of the parsed current-constituents table.
CONSTITUENTS_NAME: Final[str] = "sp500_constituents"
#: Manifest ``name`` of the parsed changes table.
CHANGES_NAME: Final[str] = "sp500_changes"

#: Matrix states (SPEC.md 15.3.1 ruling 5). An encoding, stated in the
#: committed file's header; not a parameter.
NOT_MEMBER: Final[int] = 0
MEMBER: Final[int] = 1
MEMBER_NO_DATA: Final[int] = 2

#: Wikipedia writes share classes with a dot (``BRK.B``); Yahoo writes them
#: with a dash (``BRK-B``). A vendor symbol convention, not a parameter --
#: logged under "constants held in code" in experiments.md (W7-P1).
_WIKIPEDIA_CLASS_SEPARATOR: Final[str] = "."
_YAHOO_CLASS_SEPARATOR: Final[str] = "-"

#: The changes table's date format on the page, e.g. ``August 18, 2026``, and
#: the constituents table's ISO ``Date added``.
_CHANGES_DATE_FORMAT: Final[str] = "%B %d, %Y"
_ISO_DATE_FORMAT: Final[str] = "%Y-%m-%d"

#: HTML table ``id`` attributes on the two pages, asserted on parse so a page
#: reshuffle fails loudly rather than parsing the wrong table.
_CONSTITUENTS_TABLE_ID: Final[str] = "constituents"
_CHANGES_TABLE_ID: Final[str] = "changes"


class SP500Error(RuntimeError):
    """A page or a table did not have the shape this module was written against."""


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------


#: What a symbol cell may hold once markup debris is removed. The 2026-09-04
#: snapshot carried ``ALLE |``, ``JCP |`` and ``ITT |`` -- a stray table pipe
#: inside the cell -- so the first whitespace-delimited token is taken and the
#: result is validated; anything else raises rather than reaching Yahoo.
_SYMBOL_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z0-9]+(?:[.][A-Z0-9]+)?$")


def to_vendor_symbol(wikipedia_ticker: str) -> str:
    """``BRK.B`` -> ``BRK-B``. Wikipedia's class separator to Yahoo's.

    Takes the first whitespace-delimited token of the cell (dropping markup
    debris such as a trailing ``|``) and refuses anything that is not a
    symbol, so a malformed cell fails the parse instead of becoming a ticker.
    """
    tokens = wikipedia_ticker.strip().upper().split()
    if not tokens or not _SYMBOL_RE.match(tokens[0]):
        raise SP500Error(f"not a ticker symbol: {wikipedia_ticker!r}")
    return tokens[0].replace(_WIKIPEDIA_CLASS_SEPARATOR, _YAHOO_CLASS_SEPARATOR)


def _blank(value: object) -> bool:
    """A blank side of a changes row.

    ``None`` when freshly parsed, ``NaN`` after a parquet round trip through
    the cache (pandas 3 reads the column back as a string dtype with a ``NaN``
    missing value). Both mean "no ticker here".
    """
    return value is None or (isinstance(value, float) and value != value)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _table_by_id(html: bytes | str, table_id: str, *, label: str) -> str:
    text = html.decode("utf-8") if isinstance(html, bytes) else html
    soup = BeautifulSoup(text, "lxml")
    table = soup.find("table", id=table_id)
    if not isinstance(table, Tag):
        found = [t.get("id") for t in soup.find_all("table") if isinstance(t, Tag)]
        raise SP500Error(
            f"{label}: no <table id={table_id!r}> on the page (tables found: {found}). "
            "Wikipedia has reshaped the page; check the URL in config and this parser."
        )
    return str(table)


def _clean(values: pd.Series) -> pd.Series:
    """Cell text stripped; blank, NaN and the literal ``nan`` become ``None``.

    Built as an object-dtype series explicitly: under pandas 3's default
    string dtype a ``map`` that returns ``None`` hands back ``NaN`` floats,
    which is how the first run of this parser handed ``nan`` to the symbol map.
    """
    cleaned: list[str | None] = []
    for value in values.tolist():
        text = "" if value is None or pd.isna(value) else str(value).strip()
        cleaned.append(None if text in ("", "nan") else text)
    return pd.Series(cleaned, index=values.index, dtype="object")


def parse_constituents(html: bytes | str) -> pd.DataFrame:
    """The current list: one row per ticker.

    Columns ``ticker`` (Yahoo symbol), ``wikipedia_ticker``, ``security``,
    ``date_added``. ``date_added`` is Wikipedia's ``Date added`` parsed as a
    date; a blank or unparseable cell becomes ``NaT`` and is reported as such
    rather than guessed.
    """
    table = _table_by_id(html, _CONSTITUENTS_TABLE_ID, label="constituents")
    raw = pd.read_html(io.StringIO(table))[0]
    columns = {str(c) for c in raw.columns}
    missing = {"Symbol", "Security", "Date added"} - columns
    if missing:
        raise SP500Error(f"constituents table lacks columns {sorted(missing)}")
    wikipedia = _clean(raw["Symbol"])
    if wikipedia.isna().any():
        raise SP500Error("constituents table has a blank Symbol cell")
    out = pd.DataFrame(
        {
            "ticker": wikipedia.map(to_vendor_symbol),
            "wikipedia_ticker": wikipedia,
            "security": _clean(raw["Security"]),
            "date_added": pd.to_datetime(
                _clean(raw["Date added"]), format=_ISO_DATE_FORMAT, errors="coerce"
            ),
        }
    )
    if out.empty:
        raise SP500Error("constituents table parsed to zero rows")
    if out["ticker"].duplicated().any():
        dupes = sorted(out.loc[out["ticker"].duplicated(), "ticker"])
        raise SP500Error(f"constituents table repeats tickers {dupes}")
    return out.reset_index(drop=True)


def parse_changes(html: bytes | str) -> pd.DataFrame:
    """The changes table, one row per line as Wikipedia has it.

    Columns ``effective_date``, ``added``, ``added_security``, ``removed``,
    ``removed_security``, ``reason``. Tickers are Yahoo symbols; a blank side
    is ``None``. A date that does not parse raises, because a silently dropped
    row is exactly a gap this module would then fail to count.
    """
    table = _table_by_id(html, _CHANGES_TABLE_ID, label="changes")
    raw = pd.read_html(io.StringIO(table))[0]
    columns: list[object] = list(raw.columns)
    flat = [" ".join(str(x) for x in c) if isinstance(c, tuple) else str(c) for c in columns]
    raw.columns = pd.Index(flat)

    def col(prefix: str, name: str) -> str:
        matches = [c for c in flat if c.startswith(prefix) and c.endswith(name)]
        if len(matches) != 1:
            raise SP500Error(
                f"changes table: expected one column {prefix!r}/{name!r}, got {matches}"
            )
        return matches[0]

    date_col = col("Effective Date", "Effective Date")
    dates = pd.to_datetime(_clean(raw[date_col]), format=_CHANGES_DATE_FORMAT, errors="coerce")
    if dates.isna().any():
        bad = raw.loc[dates.isna(), date_col].astype(str).tolist()[:5]
        raise SP500Error(f"changes table: {int(dates.isna().sum())} unparseable dates, e.g. {bad}")

    def symbols(prefix: str) -> pd.Series:
        cleaned = _clean(raw[col(prefix, "Ticker")])
        out = [None if v is None else to_vendor_symbol(str(v)) for v in cleaned.tolist()]
        return pd.Series(out, index=cleaned.index, dtype="object")

    out = pd.DataFrame(
        {
            "effective_date": dates,
            "added": symbols("Added"),
            "added_security": _clean(raw[col("Added", "Security")]),
            "removed": symbols("Removed"),
            "removed_security": _clean(raw[col("Removed", "Security")]),
            "reason": _clean(raw[col("Reason", "Reason")]),
        }
    )
    if out.empty:
        raise SP500Error("changes table parsed to zero rows")
    if (out["added"].isna() & out["removed"].isna()).any():
        raise SP500Error("changes table has a row with neither an added nor a removed ticker")
    return out.reset_index(drop=True)


def universe_tickers(constituents: pd.DataFrame, changes: pd.DataFrame) -> tuple[str, ...]:
    """Every Yahoo symbol the two tables name, sorted and deduplicated.

    This is the set the bars loader requests: current members and every
    ticker that was ever added or removed, so that a departed name is fetched
    rather than assumed absent.
    """
    names: set[str] = set(constituents["ticker"].astype(str))
    for column in ("added", "removed"):
        names.update(str(v) for v in changes[column].dropna())
    return tuple(sorted(names))


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


InconsistencyKind = Literal["added_not_in_set", "removed_already_in_set"]


@dataclass(frozen=True)
class Inconsistency:
    """A changes-table row the backward walk could not apply. Counted, not fixed."""

    effective_date: pd.Timestamp
    ticker: str
    kind: InconsistencyKind


@dataclass(frozen=True)
class GapCounts:
    """Known gaps by year. Every figure is a count of rows or names, never a rate."""

    #: Changes-table rows per year (a row with both sides counts once).
    rows: dict[int, int]
    #: Current members whose ``Date added`` falls in the year and has no
    #: addition row under their ticker (after rename linking).
    additions_without_row: dict[int, int]
    #: Rows the walk could not apply, by year of the row.
    inconsistencies: dict[int, int]
    #: Rename links applied, by year of the addition row.
    rename_links: dict[int, int]


@dataclass(frozen=True)
class Reconstruction:
    """The reconstructed intervals plus everything the report owes about them."""

    #: One row per membership spell: ``ticker``, ``start`` (``NaT`` when the
    #: spell reaches the table's earliest row), ``end`` (exclusive; ``NaT`` =
    #: still a member), ``start_known``, ``security``.
    intervals: pd.DataFrame
    #: Earliest effective date in the changes table -- the reach.
    reach: pd.Timestamp
    #: The date the current list was read.
    snapshot: pd.Timestamp
    inconsistencies: tuple[Inconsistency, ...]
    #: ``(current ticker, addition-row ticker, date)`` for each link applied.
    rename_links: tuple[tuple[str, str, pd.Timestamp], ...] = field(default_factory=tuple)
    #: Current members with a ``Date added`` on or after the reach and no
    #: addition row: ``(ticker, date_added)``.
    additions_without_row: tuple[tuple[str, pd.Timestamp], ...] = field(default_factory=tuple)
    #: Current members whose ``Date added`` did not parse.
    undated_members: tuple[str, ...] = field(default_factory=tuple)


def _link_renames(
    constituents: pd.DataFrame, changes: pd.DataFrame
) -> tuple[
    dict[str, str],
    tuple[tuple[str, str, pd.Timestamp], ...],
    tuple[tuple[str, pd.Timestamp], ...],
]:
    """Map addition-row tickers to today's tickers where the date settles it.

    Returns ``(alias -> current ticker, links applied, current members still
    without an addition row on their Date added)``.
    """
    added_on: dict[pd.Timestamp, list[str]] = {}
    for when_raw, added in zip(changes["effective_date"], changes["added"], strict=True):
        if not _blank(added):
            added_on.setdefault(pd.Timestamp(when_raw), []).append(str(added))
    reach = pd.Timestamp(changes["effective_date"].min())

    unmatched: list[tuple[str, pd.Timestamp]] = []
    for ticker_raw, date_added in zip(
        constituents["ticker"], constituents["date_added"], strict=True
    ):
        if pd.isna(date_added):
            continue
        when = pd.Timestamp(date_added)
        if when < reach:
            continue
        if str(ticker_raw) not in added_on.get(when, []):
            unmatched.append((str(ticker_raw), when))

    current = set(constituents["ticker"].astype(str))
    by_date: dict[pd.Timestamp, list[str]] = {}
    for ticker, when in unmatched:
        by_date.setdefault(when, []).append(ticker)

    alias: dict[str, str] = {}
    links: list[tuple[str, str, pd.Timestamp]] = []
    still: list[tuple[str, pd.Timestamp]] = []
    for when, members in sorted(by_date.items()):
        candidates = [t for t in added_on.get(when, []) if t not in current]
        if len(members) == 1 and len(candidates) == 1:
            alias[candidates[0]] = members[0]
            links.append((members[0], candidates[0], when))
        else:
            still.extend((m, when) for m in members)
    return alias, tuple(links), tuple(sorted(still))


def reconstruct(
    constituents: pd.DataFrame, changes: pd.DataFrame, *, snapshot: date | pd.Timestamp
) -> Reconstruction:
    """Walk the changes table backwards from the current list. SPEC.md 15.3.1.

    ``snapshot`` is the date the current list was read. Wikipedia lists
    announced changes under their future effective date, so rows dated after
    the snapshot describe a list the snapshot does not yet reflect; they are
    left out of the walk and show only in the row count.
    """
    snap = pd.Timestamp(snapshot)
    if changes.empty or constituents.empty:
        raise SP500Error("reconstruct: empty input")
    live = changes[changes["effective_date"] <= snap].copy()
    if live.empty:
        raise SP500Error("reconstruct: every change row is after the snapshot date")
    reach = pd.Timestamp(live["effective_date"].min())

    alias, links, additions_without_row = _link_renames(constituents, live)
    security: dict[str, str | None] = {
        str(t): (None if _blank(s) else str(s))
        for t, s in zip(constituents["ticker"], constituents["security"], strict=True)
    }
    undated = tuple(
        sorted(str(t) for t in constituents.loc[constituents["date_added"].isna(), "ticker"])
    )

    # Open spells: ticker -> (end exclusive or None, security name).
    members: dict[str, tuple[pd.Timestamp | None, str | None]] = {
        t: (None, security.get(t)) for t in security
    }
    closed: list[dict[str, object]] = []
    inconsistencies: list[Inconsistency] = []

    # Descending by date. Within one date, removals are applied before
    # additions so that a same-day swap of one ticker for itself (a
    # reincorporation that keeps its symbol) nets out rather than colliding.
    for when in sorted(live["effective_date"].unique(), reverse=True):
        stamp = pd.Timestamp(when)
        block = live[live["effective_date"] == when]
        for removed, removed_security in zip(
            block["removed"], block["removed_security"], strict=True
        ):
            if _blank(removed):
                continue
            ticker = str(removed)
            if ticker in members:
                inconsistencies.append(Inconsistency(stamp, ticker, "removed_already_in_set"))
                continue
            members[ticker] = (stamp, None if _blank(removed_security) else str(removed_security))
        for added, added_security in zip(block["added"], block["added_security"], strict=True):
            if _blank(added):
                continue
            ticker = alias.get(str(added), str(added))
            if ticker not in members:
                inconsistencies.append(Inconsistency(stamp, str(added), "added_not_in_set"))
                continue
            end, name = members.pop(ticker)
            closed.append(
                {
                    "ticker": ticker,
                    "start": stamp,
                    "end": pd.NaT if end is None else end,
                    "start_known": True,
                    "security": name
                    if name is not None
                    else (None if _blank(added_security) else str(added_security)),
                }
            )
    for ticker, (end, name) in members.items():
        closed.append(
            {
                "ticker": ticker,
                "start": pd.NaT,
                "end": pd.NaT if end is None else end,
                "start_known": False,
                "security": name,
            }
        )
    intervals = pd.DataFrame(closed, columns=["ticker", "start", "end", "start_known", "security"])
    intervals["start"] = pd.to_datetime(intervals["start"])
    intervals["end"] = pd.to_datetime(intervals["end"])
    intervals["start_known"] = intervals["start_known"].astype(bool)
    intervals = intervals.sort_values(
        ["ticker", "start"], na_position="first", kind="mergesort"
    ).reset_index(drop=True)
    return Reconstruction(
        intervals=intervals,
        reach=reach,
        snapshot=snap,
        inconsistencies=tuple(inconsistencies),
        rename_links=links,
        additions_without_row=additions_without_row,
        undated_members=undated,
    )


def gap_table(rec: Reconstruction, changes: pd.DataFrame) -> GapCounts:
    """Per-year counts. SPEC.md 15.3.1 ruling 4."""
    live = changes[changes["effective_date"] <= rec.snapshot]
    rows = Counter(int(y) for y in live["effective_date"].dt.year)
    without = Counter(int(when.year) for _, when in rec.additions_without_row)
    incons = Counter(int(i.effective_date.year) for i in rec.inconsistencies)
    links = Counter(int(when.year) for _, _, when in rec.rename_links)
    years = range(int(rec.reach.year), int(rec.snapshot.year) + 1)
    return GapCounts(
        rows={y: rows.get(y, 0) for y in years},
        additions_without_row={y: without.get(y, 0) for y in years},
        inconsistencies={y: incons.get(y, 0) for y in years},
        rename_links={y: links.get(y, 0) for y in years},
    )


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


def member_on(intervals: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean ``dates x tickers`` from the interval table.

    ``start`` ``NaT`` means from the reach; ``end`` ``NaT`` means open; ``end``
    is exclusive, so a name removed on a date is not a member on that date.
    """
    tickers = sorted(str(t) for t in intervals["ticker"].unique())
    values = np.zeros((len(dates), len(tickers)), dtype=bool)
    position = {ticker: i for i, ticker in enumerate(tickers)}
    for ticker, start, end in zip(
        intervals["ticker"], intervals["start"], intervals["end"], strict=True
    ):
        mask = np.ones(len(dates), dtype=bool)
        if not pd.isna(start):
            mask &= np.asarray(dates >= pd.Timestamp(start), dtype=bool)
        if not pd.isna(end):
            mask &= np.asarray(dates < pd.Timestamp(end), dtype=bool)
        values[:, position[str(ticker)]] |= mask
    return pd.DataFrame(values, index=dates, columns=pd.Index(tickers, name="ticker"))


def membership_matrix(
    rec: Reconstruction,
    *,
    sessions: pd.DatetimeIndex,
    bars_present: pd.DataFrame | None,
) -> pd.DataFrame:
    """The daily three-state matrix on ``sessions``. SPEC.md 15.3.1 rulings 4 and 5.

    ``sessions`` is the trading calendar -- the union of dates any cached bar
    exists for, which is what the pipeline aligns on -- and is restricted here
    to ``[reach, snapshot]``. ``bars_present`` is a boolean frame on the same
    sessions with one column per ticker: whether the cache holds a valid bar
    for the ticker on that session. ``None`` means no bars were consulted, and
    every member is marked ``MEMBER_NO_DATA`` -- the state is then honest
    rather than absent.
    """
    if not isinstance(sessions, pd.DatetimeIndex) or len(sessions) == 0:
        raise SP500Error("membership_matrix: sessions must be a non-empty DatetimeIndex")
    if not sessions.is_monotonic_increasing or sessions.has_duplicates:
        raise SP500Error("membership_matrix: sessions must be sorted and unique")
    dates = pd.DatetimeIndex(sessions[(sessions >= rec.reach) & (sessions <= rec.snapshot)])
    dates.name = "date"
    if len(dates) == 0:
        raise SP500Error("membership_matrix: no session between the reach and the snapshot")
    member = member_on(rec.intervals, dates)
    matrix = pd.DataFrame(NOT_MEMBER, index=dates, columns=member.columns, dtype="int8")
    if bars_present is None:
        matrix[member] = MEMBER_NO_DATA
        return matrix
    present = bars_present.reindex(index=dates, columns=member.columns).fillna(False).astype(bool)
    matrix[member & present] = MEMBER
    matrix[member & ~present] = MEMBER_NO_DATA
    return matrix


# ---------------------------------------------------------------------------
# The committed files
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """What the licence header of a committed reference file states."""

    constituents_url: str
    changes_url: str
    licence: str
    licence_url: str
    snapshot: pd.Timestamp
    #: ``downloaded_at`` of the constituents pull, verbatim from the manifest.
    pulled_at: str
    #: ``downloaded_at`` of the bars pull the data state was read from, or
    #: ``None`` for the interval file, which carries no data state.
    bars_pull: str | None
    generated_at: datetime
    #: Loader status counts of the bars pull (ruling 6), e.g. ``ok``/``empty``/
    #: ``failed``; empty for the interval file.
    status_counts: tuple[tuple[str, int], ...] = ()
    #: Tickers still ``failed`` after the loader's retry, by name (ruling 6).
    failed_tickers: tuple[str, ...] = ()

    def header(self, what: str) -> str:
        lines = [
            f"# {what}",
            "# Source: Wikipedia, two articles, read by mafrm.data.loaders.fetch_sp500_wikipedia",
            f"#   constituents: {self.constituents_url}",
            f"#   changes:      {self.changes_url}",
            f"# Snapshot date {self.snapshot.date().isoformat()}, pulled {self.pulled_at}; "
            f"this file generated {self.generated_at.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            f"# Licence of THIS FILE: {self.licence} ({self.licence_url}), as a derived work of "
            "Wikipedia text,",
            "#   attributed to Wikipedia contributors. Separate from the repository's code "
            "licence (LICENSE).",
            "# Reconstruction: mafrm.data.sp500 -- a backward walk through the changes table "
            "from the current list.",
            "#   The changes table is incomplete; known gaps are counted per year in "
            "reports/universe_screen.md.",
            "#   An approximate point-in-time universe (SPEC.md 15.3), not a selection "
            "procedure this project ran.",
            "# ONE SNAPSHOT: this file is committed once and is NOT re-snapshotted during the "
            "build (SPEC.md 15.3.1 ruling 3, W7-P1b).",
            "#   `make data` keeps an existing committed file; rewriting it is a deliberate "
            "`python -m mafrm.data.sp500_reference --resnapshot`,",
            "#   committed with its reason. Do not refresh it out of tidiness.",
        ]
        if self.bars_pull is not None:
            counts = ", ".join(f"{name} {n}" for name, n in self.status_counts) or "none"
            failed = ", ".join(self.failed_tickers) if self.failed_tickers else "none"
            lines.append(
                "# Data state, per trading session: 1 = member with a valid Yahoo bar on the "
                "session, 2 = member with no valid bar on the session"
            )
            lines.append(
                f"#   (bars pull {self.bars_pull}; loader status per ticker: {counts}; "
                f"still failed after the retry: {failed})"
            )
        return "\n".join(lines) + "\n"


def render_intervals_csv(rec: Reconstruction, provenance: Provenance) -> str:
    """The canonical interval table as CSV text with the licence header."""
    body = rec.intervals.copy()
    body["start"] = body["start"].dt.strftime(_ISO_DATE_FORMAT)
    body["end"] = body["end"].dt.strftime(_ISO_DATE_FORMAT)
    what = (
        "S&P 500 membership intervals: end exclusive; blank start = member at the "
        "table's reach with no addition row; blank end = current member"
    )
    return provenance.header(what) + body.to_csv(index=False, lineterminator="\n")


def render_membership_csv(matrix: pd.DataFrame, provenance: Provenance) -> str:
    """The daily matrix as CSV text with the licence header."""
    body = matrix.copy()
    body.index = pd.Index(pd.DatetimeIndex(matrix.index).strftime(_ISO_DATE_FORMAT), name="date")
    what = (
        "S&P 500 daily membership on trading sessions: 0 not a member, "
        "1 member with data, 2 member, no data"
    )
    return provenance.header(what) + body.to_csv(lineterminator="\n")


def load_intervals(path: Path) -> pd.DataFrame:
    """Read a committed interval file, skipping its ``#`` header."""
    frame = pd.read_csv(path, comment="#")
    frame["start"] = pd.to_datetime(frame["start"], format=_ISO_DATE_FORMAT)
    frame["end"] = pd.to_datetime(frame["end"], format=_ISO_DATE_FORMAT)
    frame["start_known"] = frame["start_known"].astype(bool)
    frame["ticker"] = frame["ticker"].astype(str)
    return frame


def load_membership(path: Path, *, end: date | pd.Timestamp | None = None) -> pd.DataFrame:
    """Read a committed daily matrix, skipping its ``#`` header.

    ``end`` is an exclusive upper bound: model-facing readers pass the holdout
    boundary (CLAUDE.md invariant 5), because the committed file is a source
    snapshot and reaches the snapshot date by construction.
    """
    frame = pd.read_csv(path, comment="#")
    frame["date"] = pd.to_datetime(frame["date"], format=_ISO_DATE_FORMAT)
    out = frame.set_index("date").astype("int8")
    out.columns = pd.Index([str(c) for c in out.columns], name="ticker")
    if end is not None:
        out = out[out.index < pd.Timestamp(end)]
    return out
