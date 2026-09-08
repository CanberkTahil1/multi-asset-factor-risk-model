"""SEC EDGAR XBRL share counts: parsing, CIK mapping and the point-in-time rule.

SPEC.md 15.4.1 (W7-P2a, operator ruling 1). The only free POINT-IN-TIME source
of shares outstanding is the XBRL cover-page fact
``dei:EntityCommonStockSharesOutstanding``, served by EDGAR's *frames*
endpoint one calendar quarter at a time for every filer, keyed by CIK -- and a
CIK outlives a ticker, so delisted filers are covered for as long as they
filed. Nothing in this module touches the network: :mod:`mafrm.data.loaders`
fetches the bytes and this module parses them, maps the universe's tickers to
CIKs, and turns filed counts into a sessions x tickers panel under one rule.

**The rule (ruling 1, as read in ``config/model.yaml`` ``equity_shares``):** a
count is *known* from the date the filing carrying it was FILED and is
forward-filled to the next filing. The frames records carry the accession
number, the CIK, the as-of date and the value but **no filing date** (found at
orientation), so the filing date comes from EDGAR's quarterly XBRL index,
joined by accession number. A fact whose accession is in no cached index has
no known-from date and is dropped, counted, never guessed.

**Same basis as the cached close.** Yahoo's raw close is split-adjusted
backwards (``mafrm.data.prices.check_splits_applied`` asserts it), so a filed
count is multiplied by every split ratio with an ex-date AFTER the count's
as-of date. That product times the cached close equals the actual count times
the actual price on every session -- including sessions before the first
filing, where the earliest adjusted count is back-filled and LABELLED an
approximation.

Everything in :mod:`mafrm.data.market_cap` and the descriptors reads the
panel this module builds; nothing else re-derives the rule.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Final, Literal

import numpy as np
import pandas as pd

__all__ = [
    "COMPANY_TICKERS_NAME",
    "FRAMES_NAME",
    "INDEX_NAME",
    "JUMP_RATIO",
    "SOURCE",
    "AdjustedCounts",
    "EdgarError",
    "MappingMethod",
    "ScreenedCounts",
    "adjusted_counts",
    "consistency_screen",
    "filing_dates",
    "frame_period",
    "map_ciks",
    "normalize_name",
    "parse_company_tickers",
    "parse_frame",
    "parse_xbrl_index",
    "period_label",
    "periods",
    "point_in_time_panel",
    "quarter_of",
    "unexplained_jumps",
]

SOURCE: Final[str] = "edgar"
#: Long table of frame records, one row per (period, cik); indexed by the as-of date.
FRAMES_NAME: Final[str] = "shares_frames"
#: Long table of XBRL filings from the quarterly index; indexed by the filing date.
INDEX_NAME: Final[str] = "xbrl_index"
#: SEC's current ticker -> CIK map; undated.
COMPANY_TICKERS_NAME: Final[str] = "company_tickers"

#: The index file's column header, asserted on parse so a format change fails loudly.
_INDEX_HEADER: Final[tuple[str, ...]] = (
    "CIK",
    "Company Name",
    "Form Type",
    "Date Filed",
    "Filename",
)
_INDEX_SEPARATOR: Final[str] = "|"
#: An accession number: ten digits, two, six.
_ACCN_RE: Final[re.Pattern[str]] = re.compile(r"(\d{10}-\d{2}-\d{6})")
_PERIOD_RE: Final[re.Pattern[str]] = re.compile(r"^(\d{4})Q([1-4])$")

#: Legal-form tokens dropped when comparing company names. A vendor-convention
#: normalisation, not a tunable: "Alcoa Inc." and "ALCOA INC" are one company.
#: Deliberately modest -- HOLDINGS, GROUP and the like are kept, because
#: dropping them merges distinct issuers. Logged under "constants held in code".
_LEGAL_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "INC",
        "INCORPORATED",
        "CORP",
        "CORPORATION",
        "CO",
        "COMPANY",
        "LTD",
        "LIMITED",
        "PLC",
        "LLC",
        "LP",
        "NV",
        "SA",
        "AG",
        "SE",
        "THE",
    }
)

#: Report-only display threshold: consecutive adjusted counts whose ratio lies
#: outside [1/JUMP, JUMP] are LISTED as unexplained by the actions table. It
#: filters nothing. Logged under "constants held in code" (W7-P2a).
JUMP_RATIO: Final[float] = 2.0

MappingMethod = Literal["ticker", "name", "ambiguous", "unmapped"]


class EdgarError(ValueError):
    """A payload this module will not accept, named."""


# ---------------------------------------------------------------------------
# Calendar quarters
# ---------------------------------------------------------------------------


def quarter_of(day: date) -> tuple[int, int]:
    return day.year, (day.month - 1) // 3 + 1


def period_label(year: int, quarter: int) -> str:
    """``2009Q2`` -- the config's and the manifest status's spelling."""
    return f"{year}Q{quarter}"


def frame_period(year: int, quarter: int) -> str:
    """The frames API's INSTANT period for a calendar quarter, ``CY2009Q2I``."""
    return f"CY{year}Q{quarter}I"


def periods(first: str, last_day: date) -> tuple[tuple[int, int], ...]:
    """Every calendar quarter from ``first`` (``YYYYQq``) to the one holding ``last_day``."""
    match = _PERIOD_RE.match(first)
    if match is None:
        raise EdgarError(f"first period must look like 2009Q2, got {first!r}")
    year, quarter = int(match.group(1)), int(match.group(2))
    last = quarter_of(last_day)
    if (year, quarter) > last:
        raise EdgarError(f"first period {first} is after the pull date {last_day}")
    out: list[tuple[int, int]] = []
    while (year, quarter) <= last:
        out.append((year, quarter))
        quarter += 1
        if quarter == 5:
            year, quarter = year + 1, 1
    return tuple(out)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_frame(
    payload: bytes | str, *, period: str, taxonomy: str, tag: str, unit: str
) -> pd.DataFrame:
    """One frames response as a table indexed by the fact's as-of date.

    Columns ``period``, ``cik``, ``entity_name``, ``accn``, ``val``. The
    response's own taxonomy, tag and unit are asserted against what was asked
    for, so a redirected or mistyped URL cannot cache the wrong fact.
    """
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    try:
        body = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EdgarError(f"frame {period}: not JSON ({exc})") from exc
    if not isinstance(body, dict) or "data" not in body:
        raise EdgarError(f"frame {period}: no 'data' array in the response")
    for key, expected in (("taxonomy", taxonomy), ("tag", tag), ("uom", unit), ("ccp", period)):
        got = body.get(key)
        if got != expected:
            raise EdgarError(f"frame {period}: response {key}={got!r}, requested {expected!r}")
    rows: list[dict[str, object]] = []
    for i, record in enumerate(body["data"]):
        missing = {"accn", "cik", "end", "val"} - set(record)
        if missing:
            raise EdgarError(f"frame {period}: record {i} lacks {sorted(missing)}")
        rows.append(
            {
                "end": pd.Timestamp(str(record["end"])),
                "period": period,
                "cik": int(record["cik"]),
                "entity_name": str(record.get("entityName", "")),
                "accn": str(record["accn"]),
                "val": float(record["val"]),
            }
        )
    frame = pd.DataFrame(
        rows, columns=["end", "period", "cik", "entity_name", "accn", "val"]
    ).set_index("end")
    frame.index = pd.DatetimeIndex(frame.index, name="end")
    return frame.sort_index(kind="mergesort")


def parse_xbrl_index(text: bytes | str, *, year: int, quarter: int) -> pd.DataFrame:
    """One quarterly ``xbrl.idx`` as a table indexed by the filing date.

    Columns ``period``, ``cik``, ``company_name``, ``form``, ``accn``. The
    file has a free-text banner, then the pipe-separated header, a dashed
    rule, then one filing per line; the header is asserted verbatim.
    """
    body = text.decode("latin-1") if isinstance(text, bytes) else text
    lines = body.splitlines()
    header_at = next(
        (
            i
            for i, line in enumerate(lines)
            if tuple(part.strip() for part in line.split(_INDEX_SEPARATOR)) == _INDEX_HEADER
        ),
        None,
    )
    if header_at is None:
        raise EdgarError(
            f"xbrl index {year}Q{quarter}: header "
            f"{_INDEX_SEPARATOR.join(_INDEX_HEADER)!r} not found"
        )
    data_lines = [
        line for line in lines[header_at + 1 :] if line.strip() and not set(line.strip()) <= {"-"}
    ]
    rows: list[dict[str, object]] = []
    for line in data_lines:
        parts = line.split(_INDEX_SEPARATOR)
        if len(parts) != len(_INDEX_HEADER):
            raise EdgarError(f"xbrl index {year}Q{quarter}: malformed line {line!r}")
        cik, name, form, filed, filename = (part.strip() for part in parts)
        match = _ACCN_RE.search(filename)
        if match is None:
            raise EdgarError(f"xbrl index {year}Q{quarter}: no accession in {filename!r}")
        rows.append(
            {
                "filed": pd.Timestamp(filed),
                "period": period_label(year, quarter),
                "cik": int(cik),
                "company_name": name,
                "form": form,
                "accn": match.group(1),
            }
        )
    frame = pd.DataFrame(
        rows, columns=["filed", "period", "cik", "company_name", "form", "accn"]
    ).set_index("filed")
    frame.index = pd.DatetimeIndex(frame.index, name="filed")
    return frame.sort_index(kind="mergesort")


def parse_company_tickers(payload: bytes | str) -> pd.DataFrame:
    """SEC's ``company_tickers.json``: columns ``cik``, ``ticker``, ``title``. Undated."""
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    try:
        body = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EdgarError(f"company_tickers: not JSON ({exc})") from exc
    if not isinstance(body, dict) or not body:
        raise EdgarError("company_tickers: expected a non-empty object")
    rows = []
    for key, record in body.items():
        missing = {"cik_str", "ticker", "title"} - set(record)
        if missing:
            raise EdgarError(f"company_tickers: entry {key} lacks {sorted(missing)}")
        rows.append(
            {
                "cik": int(record["cik_str"]),
                "ticker": str(record["ticker"]).upper(),
                "title": str(record["title"]),
            }
        )
    return pd.DataFrame(rows, columns=["cik", "ticker", "title"])


# ---------------------------------------------------------------------------
# CIK mapping
# ---------------------------------------------------------------------------


def normalize_name(name: str) -> str:
    """A comparison key for a company name.

    Upper-case; parentheticals dropped (``(Class A)``); EDGAR's trailing
    ``/DE/`` incorporation suffix dropped; ``&`` read as ``AND``; punctuation
    to spaces; legal-form tokens removed. Exact after that -- no fuzzy
    matching, so a failure is an honest ``unmapped`` rather than a guess.
    """
    s = name.upper()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"/[A-Z0-9 .]*/?\s*$", " ", s)
    s = s.replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    tokens = [t for t in s.split() if t not in _LEGAL_TOKENS]
    return " ".join(tokens)


def _name_candidates(names: Iterable[tuple[int, str]]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for cik, name in names:
        key = normalize_name(name)
        if key:
            out.setdefault(key, set()).add(int(cik))
    return out


def map_ciks(
    securities: Mapping[str, str],
    current: Iterable[str],
    company_tickers: pd.DataFrame,
    filer_names: Iterable[tuple[int, str]],
) -> pd.DataFrame:
    """Map every universe ticker to a CIK, or say why not.

    ``securities`` is ticker -> Wikipedia company name for every ticker the
    membership tables name; ``current`` the tickers on the current list.
    ``filer_names`` are ``(cik, name)`` pairs from the frames' entity names
    and the index's company names -- the names filers actually filed under.

    A CURRENT ticker maps by ticker through SEC's map (a current list's tickers
    are current), falling back to its name. A DEPARTED ticker maps by NAME
    only: tickers are recycled, and the company holding ``T`` today is not the
    one the index dropped. A name matching no filer is ``unmapped``; a name
    matching more than one CIK is ``ambiguous``; neither is resolved by hand.
    Returns one row per ticker: ``security``, ``current``, ``cik`` (nullable
    Int64), ``method``, ``matched_name``.
    """
    current_set = set(current)
    by_ticker: dict[str, tuple[int, str]] = {}
    sec_rows: list[tuple[int, str, str]] = [
        (int(c), str(tk).upper(), str(ti))
        for c, tk, ti in zip(
            company_tickers["cik"].astype(int).tolist(),
            company_tickers["ticker"].astype(str).tolist(),
            company_tickers["title"].astype(str).tolist(),
            strict=True,
        )
    ]
    for cik_value, sec_ticker, title in sec_rows:
        by_ticker.setdefault(sec_ticker, (cik_value, title))
    candidates = _name_candidates([*filer_names, *((c, ti) for c, _, ti in sec_rows)])

    def by_name(security: str) -> tuple[int | None, MappingMethod, str | None]:
        key = normalize_name(security)
        ciks = candidates.get(key, set())
        if len(ciks) == 1:
            return next(iter(ciks)), "name", key
        if len(ciks) > 1:
            return None, "ambiguous", key
        return None, "unmapped", key

    rows: list[dict[str, object]] = []
    for ticker in sorted(securities):
        security = securities[ticker]
        is_current = ticker in current_set
        cik: int | None
        if is_current and ticker.upper() in by_ticker:
            cik, title = by_ticker[ticker.upper()]
            method: MappingMethod = "ticker"
            matched: str | None = title
        else:
            cik, method, matched = by_name(security)
        rows.append(
            {
                "ticker": ticker,
                "security": security,
                "current": is_current,
                "cik": cik,
                "method": method,
                "matched_name": matched,
            }
        )
    out = pd.DataFrame(
        rows, columns=["ticker", "security", "current", "cik", "method", "matched_name"]
    )
    out["cik"] = out["cik"].astype("Int64")
    return out.set_index("ticker")


# ---------------------------------------------------------------------------
# The point-in-time rule
# ---------------------------------------------------------------------------


def filing_dates(index: pd.DataFrame) -> pd.Series:
    """``accn -> filing date`` from the cached index tables. One date per accession."""
    if not isinstance(index.index, pd.DatetimeIndex):
        raise EdgarError("xbrl index must be indexed by filing date")
    series = pd.Series(index.index, index=index["accn"].astype(str).to_numpy(), name="filed")
    return series[~series.index.duplicated(keep="first")]


@dataclass(frozen=True)
class AdjustedCounts:
    """One filer's usable counts, on the cache's split basis, keyed by filing date."""

    #: Indexed by ``filed`` (one row per filing date); columns ``end``, ``val``,
    #: ``accn``, ``shares``. ``shares`` is ``val`` x every split ratio after ``end``.
    table: pd.DataFrame
    #: Frame records dropped because their accession is in no cached index.
    no_filing_date: int
    #: Frame records dropped because the value is not positive.
    nonpositive: int


def adjusted_counts(
    facts: pd.DataFrame, filed: pd.Series, splits: pd.Series | None
) -> AdjustedCounts:
    """Apply ruling 1's two clauses to one filer's frame records.

    ``facts`` are frame rows for one CIK indexed by as-of date ``end`` with
    ``val`` and ``accn``; ``filed`` maps accession to filing date; ``splits``
    is the ticker's split ratios indexed by ex-date (``None`` = no splits).
    Duplicate filing dates keep the latest as-of date. Records are dropped only
    for the two named reasons and each drop is counted.
    """
    if facts.empty:
        return AdjustedCounts(_empty_counts(), 0, 0)
    work = facts.reset_index()[["end", "val", "accn"]].copy()
    work["filed"] = work["accn"].astype(str).map(filed)
    no_date = int(work["filed"].isna().sum())
    work = work[work["filed"].notna()]
    nonpositive = int((work["val"] <= 0).sum() + work["val"].isna().sum())
    work = work[work["val"] > 0]
    if work.empty:
        return AdjustedCounts(_empty_counts(), no_date, nonpositive)
    ratios = np.ones(len(work))
    if splits is not None and len(splits) > 0:
        ex_dates = pd.DatetimeIndex(splits.index)
        values = splits.to_numpy(dtype=float)
        for i, end in enumerate(pd.DatetimeIndex(work["end"])):
            after = values[ex_dates > end]
            if after.size:
                ratios[i] = float(np.prod(after))
    work["shares"] = work["val"].to_numpy(dtype=float) * ratios
    work = work.sort_values(["filed", "end"], kind="mergesort")
    work = work.drop_duplicates(subset="filed", keep="last")
    table = work.set_index(pd.DatetimeIndex(work["filed"], name="filed"))[
        ["end", "val", "accn", "shares"]
    ]
    return AdjustedCounts(table, no_date, nonpositive)


def _empty_counts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "end": pd.Series(dtype="datetime64[ns]"),
            "val": pd.Series(dtype=float),
            "accn": pd.Series(dtype=str),
            "shares": pd.Series(dtype=float),
        },
        index=pd.DatetimeIndex([], name="filed"),
    )


@dataclass(frozen=True)
class ScreenedCounts:
    """One filer's counts after SPEC.md 15.4.3 ruling (i), and the record of what it dropped."""

    #: The surviving rows of :attr:`AdjustedCounts.table`, same columns and index.
    table: pd.DataFrame
    #: The dropped rows, with ``median`` (the name's median adjusted count the
    #: screen compared against) and ``ratio`` (``shares / median``) appended.
    dropped: pd.DataFrame
    #: The median the screen used; NaN for a filer with no usable count.
    median: float


def consistency_screen(table: pd.DataFrame, *, ratio: float) -> ScreenedCounts:
    """Within-name consistency screen. SPEC.md 15.4.3 ruling (i) (operator, 2026-09-05).

    A filed count more than ``ratio`` times the name's own MEDIAN split-adjusted
    count, or below ``1/ratio`` of it, is a unit-scale error (a count tagged in
    thousands or millions) or a shell placeholder (``1``, ``100``) and is
    dropped. One pass against the median of ALL the name's usable counts, so
    the median is computed once and the screen does not chase itself. The
    previous count then stays in force through the dropped filing's date
    (:func:`point_in_time_panel` forward-fills what survives) and the pre-XBRL
    back-fill reads the earliest SURVIVING count. Every drop is returned, never
    silently discarded. A corporate action the actions table lacks (2-10x)
    passes: the ratio is the geometric midpoint of a band the pull found empty,
    and ``tests/test_market_cap_dataset.py`` asserts that band stays empty.
    """
    if not ratio > 1.0:
        raise EdgarError(f"consistency_screen: ratio must exceed 1, got {ratio}")
    empty_dropped = table.iloc[0:0].assign(
        median=pd.Series(dtype=float), ratio=pd.Series(dtype=float)
    )
    if table.empty:
        return ScreenedCounts(table, empty_dropped, float("nan"))
    shares = table["shares"].to_numpy(dtype=float)
    median = float(np.median(shares))
    rel = shares / median
    off = (rel > ratio) | (rel < 1.0 / ratio)
    dropped = table[off].assign(median=median, ratio=rel[off])
    return ScreenedCounts(table[~off], dropped, median)


def point_in_time_panel(
    counts: Mapping[str, pd.DataFrame], sessions: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Known-from-filing-date shares on every session, and the approximation mask.

    For each ticker the adjusted count filed on ``d`` is in force from the
    first session on or after ``d`` until the next filing. Sessions before the
    first filing carry the first count and are ``True`` in the returned mask;
    a ticker with no usable count is all-NaN and all-False.
    """
    if not sessions.is_monotonic_increasing or sessions.has_duplicates:
        raise EdgarError("sessions must be a strictly increasing DatetimeIndex")
    tickers = sorted(counts)
    shares = pd.DataFrame(np.nan, index=sessions, columns=pd.Index(tickers, name="ticker"))
    approx = pd.DataFrame(False, index=sessions, columns=pd.Index(tickers, name="ticker"))
    for ticker in tickers:
        table = counts[ticker]
        if table.empty:
            continue
        series = table["shares"].astype(float)
        series = series[~series.index.duplicated(keep="last")].sort_index()
        union = series.index.union(sessions)
        in_force = series.reindex(union).ffill().reindex(sessions)
        first = series.index[0]
        before = sessions < first
        in_force[before] = float(series.iloc[0])
        shares[ticker] = in_force
        approx[ticker] = before
    return shares, approx


def unexplained_jumps(
    counts: Mapping[str, pd.DataFrame], *, ratio: float = JUMP_RATIO
) -> pd.DataFrame:
    """Consecutive adjusted counts whose ratio the actions table does not explain.

    A REPORT listing, not a filter: after split adjustment a genuine count
    should move by buybacks and issuance, which are small; a ratio outside
    ``[1/ratio, ratio]`` is a placeholder value (a filer reporting ``1``), a
    split the vendor's actions table lacks, or a class restructuring, and each
    is a name for the operator to rule on. Columns ``ticker``, ``filed``,
    ``previous``, ``shares``, ``ratio``.
    """
    rows: list[dict[str, object]] = []
    for ticker in sorted(counts):
        table = counts[ticker]
        if len(table) < 2:
            continue
        values = table["shares"].to_numpy(dtype=float)
        for i in range(1, len(values)):
            r = values[i] / values[i - 1]
            if r > ratio or r < 1.0 / ratio:
                rows.append(
                    {
                        "ticker": ticker,
                        "filed": table.index[i],
                        "previous": values[i - 1],
                        "shares": values[i],
                        "ratio": r,
                    }
                )
    return pd.DataFrame(rows, columns=["ticker", "filed", "previous", "shares", "ratio"])
