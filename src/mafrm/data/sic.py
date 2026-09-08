"""Industries for the equity cross-section: SIC codes and the Fama-French 49 mapping.

SPEC.md 15.6 and its rulings at 15.6.1 (W7-P3). Three cached tables, all
undated identifier tables, and the pure functions that read them:

* ``ken_french/siccodes49`` -- Ken French's own ``Siccodes49`` file, parsed to
  one row per SIC range with its industry index (ruling 1). A SIC code in
  none of the listed ranges is industry 49, "Other": French's rule.
* ``edgar/sic_current`` -- the CURRENT SIC of every mapped CIK from EDGAR's
  submissions endpoint (ruling 2), one request per CIK.
* ``edgar/sic_first_10k`` -- the SIC the SGML header of each name's FIRST
  in-sample 10-K carried at that date, so the current-applied-backwards
  approximation is measured rather than claimed (ruling 2).

Nothing here fetches (CLAUDE.md invariant 1): :mod:`mafrm.data.loaders` owns
the requests and calls the parsers below, which are exercised on fixture text
in ``tests/test_sic.py``. The three tables have no date axis, so like SEC's
ticker map they are read through :func:`mafrm.data.cache.read_unrestricted`
with the reason stated; ``tests/test_holdout_guard.py`` lists this module.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final

import numpy as np
import pandas as pd

from mafrm.data import cache, edgar, french, holdout

__all__ = [
    "CURRENT_NAME",
    "FIRST_10K_NAME",
    "HISTORY_NAME",
    "NO_10K",
    "NO_SIC",
    "NO_SIC_IN_HEADER",
    "REASON_UNDATED",
    "SICCODES_NAME",
    "Header",
    "SicError",
    "Submissions",
    "drift_table",
    "drifted_tickers",
    "first_filing",
    "industries_for",
    "industry_of",
    "load_current",
    "load_first_10k",
    "load_history",
    "load_siccodes",
    "map_sic_to_industry",
    "older_pages_needed",
    "parse_filings_page",
    "parse_header",
    "parse_siccodes",
    "parse_submissions",
    "point_in_time_industries",
    "submissions_name",
]

#: Cache names. The Siccodes table lives under Ken French's source, the two
#: SIC tables under EDGAR's.
SICCODES_NAME: Final[str] = "siccodes49"
CURRENT_NAME: Final[str] = "sic_current"
FIRST_10K_NAME: Final[str] = "sic_first_10k"
#: Every in-sample 10-K header of the names whose industry drifted (W7-P3b); indexed by filing date.
HISTORY_NAME: Final[str] = "sic_history"

#: Per-ticker statuses of the first-10-K table beyond the loader's ok/empty/failed.
NO_10K: Final[str] = "no_10k"
NO_SIC_IN_HEADER: Final[str] = "no_sic_in_header"
#: Per-CIK status when the submissions record carries no SIC (ruling 2: excluded, never "Other").
NO_SIC: Final[str] = "no_sic"

REASON_UNDATED: Final[str] = (
    "SPEC.md 15.6.1: the industry definitions and the per-CIK SIC tables are undated "
    "identifier tables (cache.read refuses undated frames by design); the SIC applied is the "
    "CURRENT one by ruling 2 and its drift against the first in-sample 10-K is reported"
)


REASON_HISTORY: Final[str] = (
    "SPEC.md 15.6.3: the per-filing header table is cached with filing_date as a column; it is "
    "re-indexed by filing date and truncated at the holdout boundary by holdout.truncate "
    "immediately after this read, so no filing on or after the boundary sets an industry"
)


class SicError(ValueError):
    """A source shape this module does not recognise, or an inconsistent table."""


# ---------------------------------------------------------------------------
# Ken French's Siccodes49
# ---------------------------------------------------------------------------

_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"^\s*(\d{1,2})\s+(\S+)\s+(.*?)\s*$")
_RANGE_RE: Final[re.Pattern[str]] = re.compile(r"^\s+(\d{4})-(\d{4})\s*(.*?)\s*$")


def parse_siccodes(text: bytes | str, *, industries: int, unassigned: int) -> pd.DataFrame:
    """Ken French's ``Siccodes49.txt`` as one row per SIC range.

    Columns ``industry`` (1-based index), ``abbrev``, ``name``, ``sic_low``,
    ``sic_high``, ``description``. Asserts the file names exactly
    ``industries`` industries numbered 1..industries, that every range is
    ordered, and that ranges do not overlap across industries -- a format
    change fails loudly rather than mapping silently.
    """
    raw = text.decode("latin-1") if isinstance(text, bytes) else text
    rows: list[tuple[int, str, str, int, int, str]] = []
    current: tuple[int, str, str] | None = None
    seen: dict[int, str] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        m_range = _RANGE_RE.match(line)
        if m_range is not None and current is not None and line[:1].isspace():
            low, high = int(m_range.group(1)), int(m_range.group(2))
            if low > high:
                raise SicError(f"Siccodes49: inverted range {low}-{high} under {current[1]}")
            rows.append((*current, low, high, m_range.group(3)))
            continue
        m_head = _HEADER_RE.match(line)
        if m_head is None:
            raise SicError(f"Siccodes49: unrecognised line {line!r}")
        index = int(m_head.group(1))
        if index in seen:
            raise SicError(f"Siccodes49: industry {index} listed twice")
        seen[index] = m_head.group(2)
        current = (index, m_head.group(2), m_head.group(3))
    if sorted(seen) != list(range(1, industries + 1)):
        raise SicError(f"Siccodes49: expected industries 1..{industries}, found {sorted(seen)}")
    if unassigned not in seen:
        raise SicError(f"Siccodes49: the unassigned industry {unassigned} is not in the file")
    table = pd.DataFrame(
        rows, columns=["industry", "abbrev", "name", "sic_low", "sic_high", "description"]
    )
    ordered = table.sort_values(["sic_low", "sic_high"]).reset_index(drop=True)
    overlap = ordered["sic_low"].to_numpy()[1:] <= ordered["sic_high"].to_numpy()[:-1]
    if overlap.any():
        i = int(np.argmax(overlap))
        raise SicError(
            f"Siccodes49: ranges overlap -- {ordered.iloc[i].to_dict()} and "
            f"{ordered.iloc[i + 1].to_dict()}"
        )
    return table.reset_index(drop=True)


def industry_of(sic: int, table: pd.DataFrame, *, unassigned: int) -> int:
    """The FF49 industry of one SIC code; ``unassigned`` when no listed range holds it."""
    hit = table[(table["sic_low"] <= sic) & (sic <= table["sic_high"])]
    if len(hit) > 1:
        raise SicError(f"SIC {sic} falls in {len(hit)} ranges")
    return int(hit["industry"].iloc[0]) if len(hit) == 1 else int(unassigned)


def map_sic_to_industry(sics: pd.Series, table: pd.DataFrame, *, unassigned: int) -> pd.Series:
    """Vectorised :func:`industry_of`; a missing SIC stays missing (never "Other")."""
    values = [
        industry_of(int(v), table, unassigned=unassigned) if pd.notna(v) else pd.NA
        for v in sics.tolist()
    ]
    return pd.Series(pd.array(values, dtype="Int64"), index=sics.index)


# ---------------------------------------------------------------------------
# EDGAR's submissions endpoint
# ---------------------------------------------------------------------------


def submissions_name(cik: int) -> str:
    """The record's file name under the submissions template: ``CIK0000034088.json``."""
    return f"CIK{int(cik):010d}.json"


@dataclass(frozen=True)
class Submissions:
    """One CIK's submissions record: its current SIC and its filing list."""

    cik: int
    name: str
    sic: int | None
    sic_description: str
    #: Columns ``accession``, ``form``, ``filing_date`` (datetime), ``report_date``.
    filings: pd.DataFrame
    #: Older pages: ``(file name, filing_from, filing_to)``.
    older_pages: tuple[tuple[str, date, date], ...]


_FILING_KEYS: Final[tuple[str, ...]] = ("accessionNumber", "form", "filingDate", "reportDate")


def parse_filings_page(payload: bytes | str | dict[str, Any]) -> pd.DataFrame:
    """The parallel filing arrays of a submissions record (``filings.recent``) or an older page."""
    node = json.loads(payload) if isinstance(payload, bytes | str) else payload
    missing = [k for k in _FILING_KEYS if k not in node]
    if missing:
        raise SicError(f"submissions filings: missing keys {missing}")
    lengths = {k: len(node[k]) for k in _FILING_KEYS}
    if len(set(lengths.values())) != 1:
        raise SicError(f"submissions filings: ragged arrays {lengths}")
    frame = pd.DataFrame(
        {
            "accession": [str(a) for a in node["accessionNumber"]],
            "form": [str(f) for f in node["form"]],
            "filing_date": pd.to_datetime(node["filingDate"], errors="coerce"),
            "report_date": pd.to_datetime(node["reportDate"], errors="coerce"),
        }
    )
    return frame


def parse_submissions(payload: bytes | str) -> Submissions:
    """A submissions record. ``sic`` is ``None`` when the record carries none."""
    node = json.loads(payload)
    for key in ("cik", "name", "sic", "sicDescription", "filings"):
        if key not in node:
            raise SicError(f"submissions: missing key {key!r}")
    sic_raw = str(node["sic"]).strip()
    sic = int(sic_raw) if sic_raw.isdigit() else None
    filings = node["filings"]
    if "recent" not in filings:
        raise SicError("submissions: filings.recent is missing")
    pages: list[tuple[str, date, date]] = []
    for item in filings.get("files", []) or []:
        pages.append(
            (
                str(item["name"]),
                pd.Timestamp(item["filingFrom"]).date(),
                pd.Timestamp(item["filingTo"]).date(),
            )
        )
    return Submissions(
        cik=int(node["cik"]),
        name=str(node["name"]),
        sic=sic,
        sic_description=str(node["sicDescription"]),
        filings=parse_filings_page(filings["recent"]),
        older_pages=tuple(pages),
    )


def older_pages_needed(record: Submissions, *, start: date) -> tuple[str, ...]:
    """The older pages whose span reaches ``start`` -- they may hold an in-sample filing."""
    return tuple(name for name, _from, to in record.older_pages if to >= start)


def first_filing(filings: pd.DataFrame, *, forms: Sequence[str], start: date) -> pd.Series | None:
    """The earliest filing of one of ``forms`` filed on or after ``start``, or ``None``.

    Ties on the filing date (two 10-Ks the same day, which a re-filing can
    produce) resolve to the lower accession number so the choice is
    deterministic.
    """
    if filings.empty:
        return None
    eligible = filings[
        filings["form"].isin(list(forms)) & (filings["filing_date"] >= pd.Timestamp(start))
    ]
    if eligible.empty:
        return None
    ordered = eligible.sort_values(["filing_date", "accession"], kind="mergesort")
    return ordered.iloc[0]


# ---------------------------------------------------------------------------
# The SGML filing header
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Header:
    """What one ``.hdr.sgml`` says about the filing and the filer it was searched under."""

    accession: str
    form: str
    filing_date: date | None
    period: date | None
    #: ``ASSIGNED-SIC`` of the ``<FILER>`` block whose ``<CIK>`` matched; ``None`` if absent.
    sic: int | None
    filer_found: bool


_TAG_RE: Final[re.Pattern[str]] = re.compile(r"^<([A-Z0-9-]+)>(.*)$")


def _tag_value(lines: Iterable[str], tag: str) -> str | None:
    for line in lines:
        m = _TAG_RE.match(line.strip())
        if m is not None and m.group(1) == tag:
            return m.group(2).strip()
    return None


def _date_or_none(value: str | None) -> date | None:
    if value is None or not value.isdigit() or len(value) != 8:
        return None
    return date(int(value[:4]), int(value[4:6]), int(value[6:8]))


def parse_header(payload: bytes | str, *, cik: int) -> Header:
    """Read form type, dates and the matching filer's ``ASSIGNED-SIC`` from an SGML header.

    A filing can carry several ``<FILER>`` blocks (a co-registrant, a
    subsidiary guarantor); the SIC is taken from the block whose ``<CIK>``
    is ``cik``, never from the first block.
    """
    raw = payload.decode("latin-1") if isinstance(payload, bytes) else payload
    if "<SEC-HEADER>" not in raw:
        raise SicError("header: no <SEC-HEADER> element")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    accession = _tag_value(lines, "ACCESSION-NUMBER") or ""
    form = _tag_value(lines, "TYPE") or ""
    filing_date = _date_or_none(_tag_value(lines, "FILING-DATE"))
    period = _date_or_none(_tag_value(lines, "PERIOD"))
    sic: int | None = None
    found = False
    blocks = raw.split("<FILER>")[1:]
    for block in blocks:
        block_lines = [ln for ln in block.split("</FILER>")[0].splitlines() if ln.strip()]
        block_cik = _tag_value(block_lines, "CIK")
        if block_cik is None or not block_cik.strip().isdigit():
            continue
        if int(block_cik) != int(cik):
            continue
        found = True
        value = _tag_value(block_lines, "ASSIGNED-SIC")
        if value is not None and value.isdigit():
            sic = int(value)
        break
    return Header(
        accession=accession,
        form=form,
        filing_date=filing_date,
        period=period,
        sic=sic,
        filer_found=found,
    )


# ---------------------------------------------------------------------------
# Cache access. Never fetches -- CLAUDE.md invariant 1.
# ---------------------------------------------------------------------------


def load_siccodes(manifest: cache.Manifest | None = None) -> pd.DataFrame:
    book = manifest if manifest is not None else cache.Manifest.load()
    entry = book.latest(source=french.SOURCE, name=SICCODES_NAME)
    return cache.read_unrestricted(entry, manifest=book, reason=REASON_UNDATED)


def load_current(manifest: cache.Manifest | None = None) -> pd.DataFrame:
    """One row per CIK (the index): ``sic`` (nullable), ``sic_description``, ``name``,
    ``status``.
    """
    book = manifest if manifest is not None else cache.Manifest.load()
    entry = book.latest(source=edgar.SOURCE, name=CURRENT_NAME)
    frame = cache.read_unrestricted(entry, manifest=book, reason=REASON_UNDATED)
    if frame.index.name != "cik":
        frame = frame.set_index("cik")
    return frame


def load_first_10k(manifest: cache.Manifest | None = None) -> pd.DataFrame:
    """One row per ticker (the index): the CIK searched, the filing chosen, its header's
    SIC, and status.
    """
    book = manifest if manifest is not None else cache.Manifest.load()
    entry = book.latest(source=edgar.SOURCE, name=FIRST_10K_NAME)
    frame = cache.read_unrestricted(entry, manifest=book, reason=REASON_UNDATED)
    if frame.index.name != "ticker":
        frame = frame.set_index("ticker")
    return frame


def load_history(manifest: cache.Manifest | None = None) -> pd.DataFrame:
    """The per-filing headers of the drifted names, DATED by filing date and cut at the boundary.

    Columns ``ticker``, ``cik``, ``accession``, ``form``, ``period``, ``sic``
    (nullable), ``status``; the index is the filing date, so ``cache.read``'s
    holdout truncation applies and a 10-K filed on or after the boundary
    never sets an in-sample industry.
    """
    book = manifest if manifest is not None else cache.Manifest.load()
    entry = book.latest(source=edgar.SOURCE, name=HISTORY_NAME)
    # Cached as a plain table with filing_date as a column (the manifest's
    # date range is taken from it); re-indexed by filing date here and cut at
    # the boundary by the same rule cache.read applies to a dated artefact.
    frame = cache.read_unrestricted(entry, manifest=book, reason=REASON_HISTORY)
    frame = frame.set_index(pd.DatetimeIndex(pd.to_datetime(frame["filing_date"]))).drop(
        columns=["filing_date"]
    )
    frame.index.name = "filing_date"
    return holdout.truncate(frame, label=f"{edgar.SOURCE}:{HISTORY_NAME}")


# ---------------------------------------------------------------------------
# Per-ticker industries
# ---------------------------------------------------------------------------


def industries_for(
    mapping: pd.DataFrame,
    current: pd.DataFrame,
    siccodes: pd.DataFrame,
    *,
    unassigned: int,
) -> pd.DataFrame:
    """The current SIC and FF49 industry of every ticker in ``mapping``.

    ``mapping`` is :attr:`mafrm.data.market_cap.MarketCapPanel.mapping` (one
    row per ticker, nullable ``cik``). Returns, indexed by ticker: ``cik``,
    ``sic`` (nullable Int64), ``sic_description``, ``industry`` (nullable
    Int64) and ``status`` -- ``ok``, ``no_cik`` (no CIK to look up), ``no_sic``
    (the record carries none: EXCLUDED, never "Other"), or the loader's
    ``empty`` / ``failed`` for a CIK the pull did not deliver.
    """
    tickers = [str(t) for t in mapping.index]
    ciks = pd.to_numeric(mapping["cik"], errors="coerce")
    sic = pd.Series(pd.NA, index=pd.Index(tickers, name="ticker"), dtype="Int64")
    desc = pd.Series("", index=sic.index, dtype=object)
    status = pd.Series("no_cik", index=sic.index, dtype=object)
    cur_sic = pd.to_numeric(current["sic"], errors="coerce") if "sic" in current else pd.Series()
    for ticker, cik in zip(tickers, ciks.tolist(), strict=True):
        if pd.isna(cik):
            continue
        key = int(cik)
        if key not in current.index:
            status[ticker] = "empty"
            continue
        record_status = str(current.loc[key, "status"])
        if record_status != "ok":
            status[ticker] = record_status
            continue
        value = cur_sic.loc[key]
        if pd.isna(value):
            status[ticker] = NO_SIC
            continue
        sic[ticker] = int(value)
        desc[ticker] = str(current.loc[key, "sic_description"])
        status[ticker] = "ok"
    industry = map_sic_to_industry(sic, siccodes, unassigned=unassigned)
    return pd.DataFrame(
        {
            "cik": ciks.astype("Int64").to_numpy(),
            "sic": sic.to_numpy(),
            "sic_description": desc.to_numpy(),
            "industry": industry.to_numpy(),
            "status": status.to_numpy(),
        },
        index=sic.index,
    )


def drift_table(
    industries: pd.DataFrame,
    first_10k: pd.DataFrame,
    siccodes: pd.DataFrame,
    *,
    unassigned: int,
) -> pd.DataFrame:
    """Ruling 2's check, per ticker: the current SIC and industry beside the first 10-K's.

    Columns ``sic_current``, ``industry_current``, ``first_10k_accession``,
    ``first_10k_filed``, ``first_10k_cik``, ``sic_first``, ``industry_first``,
    ``first_status``, ``sic_differs``, ``industry_differs``. The two
    ``*_differs`` flags are ``<NA>`` where either side is missing.
    """
    first = first_10k.reindex(industries.index)
    sic_first = pd.to_numeric(first["sic"], errors="coerce").astype("Int64")
    industry_first = map_sic_to_industry(sic_first, siccodes, unassigned=unassigned)
    sic_current = industries["sic"].astype("Int64")
    industry_current = industries["industry"].astype("Int64")
    both = sic_current.notna() & sic_first.notna()
    sic_differs = pd.Series(pd.NA, index=industries.index, dtype="boolean")
    industry_differs = pd.Series(pd.NA, index=industries.index, dtype="boolean")
    sic_differs[both] = (sic_current[both] != sic_first[both]).astype("boolean")
    industry_differs[both] = (industry_current[both] != industry_first[both]).astype("boolean")
    return pd.DataFrame(
        {
            "sic_current": sic_current,
            "industry_current": industry_current,
            "first_10k_accession": first.get("accession", pd.NA),
            "first_10k_filed": first.get("filing_date", pd.NaT),
            "first_10k_cik": first.get("cik", pd.NA),
            "sic_first": sic_first,
            "industry_first": industry_first,
            "first_status": first.get("status", pd.NA),
            "sic_differs": sic_differs,
            "industry_differs": industry_differs,
        },
        index=industries.index,
    )


def drifted_tickers(drift: pd.DataFrame) -> tuple[str, ...]:
    """The names whose FF49 industry differs between the first in-sample 10-K and now."""
    flag = drift["industry_differs"].fillna(False).astype(bool)
    return tuple(sorted(str(t) for t in drift.index[flag.to_numpy()]))


def point_in_time_industries(
    industries: pd.DataFrame,
    history: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    siccodes: pd.DataFrame,
    *,
    unassigned: int,
    tickers: Sequence[str] | None = None,
) -> pd.DataFrame:
    """SPEC.md 15.6.3's rule: sessions x tickers industry ids, nullable Int64.

    Every ticker starts from its CURRENT industry (``industries``) on every
    session. For each ticker in ``history`` -- the drifted names -- a filing's
    SIC is KNOWN from its filing date (in force from the first session on or
    after it) and forward-filled to the next filing; sessions before the
    first filing carry the first filing's SIC, the labelled back-fill.
    Filings without a SIC (``status`` other than ``ok``) set nothing.
    """
    cols = pd.Index([str(t) for t in (tickers if tickers is not None else industries.index)])
    current = industries["industry"].reindex(cols)
    out = pd.DataFrame(
        np.broadcast_to(current.to_numpy(dtype=object), (len(sessions), len(cols))).copy(),
        index=sessions,
        columns=cols,
    ).astype("Int64")
    if history.empty:
        return out
    usable = history[(history["status"] == "ok") & history["sic"].notna()]
    for ticker, rows in usable.groupby("ticker"):
        ticker = str(ticker)
        if ticker not in cols:
            continue
        filed = rows.sort_index(kind="mergesort")
        dates = pd.DatetimeIndex(filed.index)
        ind = map_sic_to_industry(
            filed["sic"].astype("Int64"), siccodes, unassigned=unassigned
        ).to_numpy()
        positions = np.searchsorted(dates.to_numpy(), sessions.to_numpy(), side="right") - 1
        positions = np.clip(positions, 0, len(ind) - 1)  # before the first filing: its SIC
        out[ticker] = pd.array(ind[positions], dtype="Int64")
    return out
