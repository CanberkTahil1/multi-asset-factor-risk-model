"""The point-in-time market-cap panel, from the cache. SPEC.md 15.4.1 (W7-P2a).

Joins four cached artefacts under :mod:`mafrm.data.edgar`'s rule: the EDGAR
frames (share counts by as-of date), the quarterly XBRL index (filing dates by
accession), SEC's ticker map, and the S&P 500 bars and actions the committed
membership matrix was built on. The output is ``shares`` and ``cap`` on every
session of the estimation sample for every ticker the membership tables name,
an ``approximated`` mask for sessions before a name's first filing, and the
mapping table that says, per ticker, why a cap is or is not there.

Two reads cross the holdout boundary and both are questions about a SOURCE,
not about the model (``tests/test_holdout_guard.py`` pins this module):

* SEC's ticker -> CIK map is an undated identifier table -- it carries no
  observation, and ``cache.read`` refuses an undated frame by design;
* the actions table is read in full because Yahoo's split back-adjustment is
  set by the PULL date: a split after the boundary rescales every cached close
  before it, so putting a filed count on the close's basis needs every split
  the table holds. Only the ``Stock Splits`` column is taken from it.

The share counts and filing dates themselves go through ``cache.read`` and stop
at the boundary, so nothing filed on or after ``holdout_start`` is in force on
any in-sample session.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import yaml

from mafrm import config as config_mod
from mafrm.data import cache, edgar, loaders, sp500, sp500_reference

__all__ = [
    "REASON_TICKER_MAP",
    "TREAT_AS_NO_FACTS",
    "CikOverride",
    "MarketCapPanel",
    "PublishedTotals",
    "build",
    "load_cik_overrides",
    "load_close",
    "load_company_tickers",
    "load_published_totals",
    "load_splits",
]

REASON_TICKER_MAP: Final[str] = (
    "SEC's ticker->CIK map is an undated identifier table: it maps names to filers and "
    "carries no observation, and cache.read refuses an undated frame by design"
)
_REASON_SPLITS: Final[str] = (
    "the vendor's split back-adjustment basis is set by the pull date, not by the sample: a "
    "split after the holdout boundary rescales every cached close before it, so putting a "
    "filed share count on the close's basis needs every split the actions table holds; only "
    "the Stock Splits column is read"
)

#: Why a ticker has no cap. ``ok`` means at least one usable count exists.
CapStatus = str
NO_CAP_REASONS: Final[tuple[str, ...]] = ("unmapped", "ambiguous", "no_facts")


@dataclass(frozen=True)
class MarketCapPanel:
    """Shares and cap on the cache's split basis, with the record of how they got there."""

    #: sessions x tickers, split-adjusted to the cached close's basis.
    shares: pd.DataFrame
    #: ``shares x close``; NaN where either is missing.
    cap: pd.DataFrame
    #: True on sessions before the ticker's first filing (the back-filled approximation).
    approximated: pd.DataFrame
    #: One row per ticker: ``security``, ``current``, ``cik``, ``method``,
    #: ``matched_name``, ``facts``, ``used``, ``no_filing_date``, ``nonpositive``,
    #: ``first_filed``, ``last_filed``, ``status``.
    mapping: pd.DataFrame
    #: Consecutive adjusted counts the actions table does not explain (report listing).
    jumps: pd.DataFrame
    sessions: pd.DatetimeIndex
    inputs: tuple[cache.ManifestEntry, ...]
    #: Per ticker, the usable filings (``end``, ``val``, ``accn``, ``shares``) keyed
    #: by filing date -- the record of which filing is in force on a session.
    #: AFTER the consistency screen when ``consistency_ratio`` is set.
    counts: dict[str, pd.DataFrame] = field(default_factory=dict)
    #: SPEC.md 15.4.3 ruling (i): the filings the screen dropped, every one --
    #: ``ticker``, ``filed``, ``end``, ``val``, ``accn``, ``shares``, ``median``,
    #: ``ratio``. Empty when no screen was applied.
    dropped: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["ticker", "filed", "end", "val", "accn", "shares", "median", "ratio"]
        )
    )
    #: The screen's ratio, or ``None`` when the panel is UNSCREENED -- a loader
    #: output that no descriptor may read (SPEC.md 15.4.2).
    consistency_ratio: float | None = None
    #: Ruling (ii): ``ticker -> predecessor CIK`` read beside the mapped CIK.
    overrides: Mapping[str, int] = field(default_factory=dict)
    #: W7-P2b ruling 3: ``ticker -> treat_as`` for names removed by the hand table.
    excluded: Mapping[str, str] = field(default_factory=dict)

    @property
    def is_model_input(self) -> bool:
        """``cap`` may be read by a descriptor only once the screen has been applied."""
        return self.consistency_ratio is not None

    def require_model_input(self) -> None:
        if not self.is_model_input:
            raise ValueError(
                "MarketCapPanel is unscreened: SPEC.md 15.4.2 makes cap a loader output until "
                "the within-name consistency screen (equity_shares.consistency_screen.ratio) "
                "has been applied; build the panel through market_cap.build"
            )

    def __post_init__(self) -> None:
        if not (
            self.shares.index.equals(self.cap.index) and self.shares.index.equals(self.sessions)
        ):
            raise ValueError("shares, cap and sessions disagree on the calendar")
        if not self.shares.columns.equals(self.cap.columns):
            raise ValueError("shares and cap disagree on tickers")
        bad = self.cap.to_numpy(dtype=float)
        if np.any(bad[np.isfinite(bad)] <= 0):
            raise ValueError("a market cap is not positive")

    def has_cap(self) -> pd.DataFrame:
        return self.cap.notna()


def load_close(
    manifest: cache.Manifest | None = None, *, entry: cache.ManifestEntry | None = None
) -> pd.DataFrame:
    """Wide cached close (sessions x tickers), truncated at the holdout boundary."""
    book = manifest if manifest is not None else cache.Manifest.load()
    prices_entry = entry or book.latest(source="yfinance", name=loaders.SP500_PRICES_NAME)
    long = cache.read(prices_entry, manifest=book)
    if not isinstance(long.index, pd.DatetimeIndex):
        raise ValueError("sp500_prices: expected a date index")
    close = long.pivot(columns="ticker", values="Close").sort_index()
    close.columns = pd.Index([str(c) for c in close.columns], name="ticker")
    return close


def load_company_tickers(
    manifest: cache.Manifest | None = None, *, entry: cache.ManifestEntry | None = None
) -> pd.DataFrame:
    """SEC's ticker -> CIK map. UNDATED, so read unrestricted for the stated source reason."""
    book = manifest if manifest is not None else cache.Manifest.load()
    tickers_entry = entry or book.latest(source=edgar.SOURCE, name=edgar.COMPANY_TICKERS_NAME)
    return cache.read_unrestricted(tickers_entry, manifest=book, reason=REASON_TICKER_MAP)


def load_splits(
    manifest: cache.Manifest | None = None, *, entry: cache.ManifestEntry | None = None
) -> dict[str, pd.Series]:
    """Every split ratio the actions table holds, by ticker, indexed by ex-date. FULL history."""
    book = manifest if manifest is not None else cache.Manifest.load()
    actions_entry = entry or book.latest(source="yfinance", name=loaders.SP500_ACTIONS_NAME)
    long = cache.read_unrestricted(actions_entry, manifest=book, reason=_REASON_SPLITS)
    return splits_by_ticker(long)


def splits_by_ticker(actions: pd.DataFrame) -> dict[str, pd.Series]:
    """``ticker -> ratios indexed by ex-date`` from a long actions table. Pure."""
    if "Stock Splits" not in actions.columns or "ticker" not in actions.columns:
        raise ValueError("actions table lacks 'ticker' or 'Stock Splits'")
    ratios = pd.to_numeric(actions["Stock Splits"], errors="coerce")
    keep = actions[(ratios.notna()) & (ratios > 0.0) & (ratios != 1.0)]
    out: dict[str, pd.Series] = {}
    for ticker, group in keep.groupby("ticker", sort=True):
        series = group["Stock Splits"].astype(float).sort_index()
        series.index = pd.DatetimeIndex(series.index)
        out[str(ticker)] = series
    return out


# ---------------------------------------------------------------------------
# The two hand tables (SPEC.md 15.4.3 rulings (i) and (ii))
# ---------------------------------------------------------------------------


class HandTableError(ValueError):
    """A hand table is malformed or disagrees with the pull it is applied to."""


#: The one ``treat_as`` value the hand table accepts (W7-P2b ruling 3).
TREAT_AS_NO_FACTS: Final[str] = "no_facts"


@dataclass(frozen=True)
class CikOverride:
    """One row of ``config/cik_overrides.yaml``, with its provenance.

    Either a predecessor row (``predecessor_cik`` set, ruling (ii)) or a
    ``treat_as: no_facts`` row (ruling 3 of W7-P2b), never both.
    """

    ticker: str
    current_cik: int
    current_source_url: str
    date_read: date
    predecessor_cik: int | None = None
    predecessor_source_url: str | None = None
    treat_as: str | None = None

    @property
    def excludes(self) -> bool:
        return self.treat_as is not None


@dataclass(frozen=True)
class PublishedTotals:
    """``config/published_market_totals.yaml``: dated year-end totals with their provenance."""

    label: str
    source_url: str
    date_read: date
    unit: str
    #: Indexed by date, in USD (converted from ``unit`` on load).
    values_usd: pd.Series


def _hand_table(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HandTableError(f"hand table not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise HandTableError(f"{path}: expected a mapping at the top level")
    if raw.get("schema_version") != 1:
        raise HandTableError(f"{path}: schema_version must be 1, got {raw.get('schema_version')!r}")
    return raw


def _hand_str(node: Mapping[str, Any], key: str, where: str) -> str:
    value = node.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HandTableError(f"{where}.{key}: expected a non-empty string, got {value!r}")
    return value.strip()


def _hand_int(node: Mapping[str, Any], key: str, where: str) -> int:
    value = node.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HandTableError(f"{where}.{key}: expected a positive integer, got {value!r}")
    return value


def _hand_date(node: Mapping[str, Any], key: str, where: str) -> date:
    value = node.get(key)
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise HandTableError(f"{where}.{key}: expected an ISO date, got {value!r}") from exc
    raise HandTableError(f"{where}.{key}: expected an ISO date, got {value!r}")


def load_cik_overrides(path: Path) -> tuple[CikOverride, ...]:
    """Parse the predecessor-CIK hand table. Provenance is required on every row."""
    raw = _hand_table(path)
    entries = raw.get("entries")
    if not isinstance(entries, list):
        raise HandTableError(f"{path}: entries must be a list")
    out: list[CikOverride] = []
    seen: set[str] = set()
    for i, item in enumerate(entries):
        where = f"{path}: entries[{i}]"
        if not isinstance(item, Mapping):
            raise HandTableError(f"{where}: expected a mapping")
        treat_as = item.get("treat_as")
        has_predecessor = "predecessor_cik" in item
        if (treat_as is None) == (not has_predecessor):
            raise HandTableError(
                f"{where}: give predecessor_cik OR treat_as: {TREAT_AS_NO_FACTS}, "
                "not both or neither"
            )
        if treat_as is not None and treat_as != TREAT_AS_NO_FACTS:
            raise HandTableError(
                f"{where}.treat_as must be {TREAT_AS_NO_FACTS!r}, got {treat_as!r}"
            )
        row = CikOverride(
            ticker=_hand_str(item, "ticker", where).upper(),
            current_cik=_hand_int(item, "current_cik", where),
            current_source_url=_hand_str(item, "current_source_url", where),
            date_read=_hand_date(item, "date_read", where),
            predecessor_cik=_hand_int(item, "predecessor_cik", where) if has_predecessor else None,
            predecessor_source_url=(
                _hand_str(item, "predecessor_source_url", where) if has_predecessor else None
            ),
            treat_as=str(treat_as) if treat_as is not None else None,
        )
        if row.predecessor_cik is not None and row.current_cik == row.predecessor_cik:
            raise HandTableError(f"{where}: predecessor_cik equals current_cik")
        if row.ticker in seen:
            raise HandTableError(f"{where}: duplicate row for {row.ticker}")
        seen.add(row.ticker)
        out.append(row)
    return tuple(out)


_UNIT_MULTIPLIERS: Final[dict[str, float]] = {
    "USD": 1.0,
    "USD millions": 1e6,
    "USD billions": 1e9,
    "USD trillions": 1e12,
}


def load_published_totals(path: Path) -> PublishedTotals:
    """Parse the published-totals hand table into a USD series indexed by date."""
    raw = _hand_table(path)
    series = raw.get("series")
    if not isinstance(series, Mapping):
        raise HandTableError(f"{path}: series must be a mapping")
    where = f"{path}: series"
    unit = _hand_str(series, "unit", where)
    if unit not in _UNIT_MULTIPLIERS:
        raise HandTableError(
            f"{where}.unit must be one of {sorted(_UNIT_MULTIPLIERS)}, got {unit!r}"
        )
    values = series.get("values")
    if not isinstance(values, Mapping) or not values:
        raise HandTableError(f"{where}.values must be a non-empty mapping of date -> value")
    dates: list[pd.Timestamp] = []
    amounts: list[float] = []
    for key, value in values.items():
        try:
            day = pd.Timestamp(date.fromisoformat(str(key)))
        except ValueError as exc:
            raise HandTableError(f"{where}.values: {key!r} is not an ISO date") from exc
        if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
            raise HandTableError(
                f"{where}.values[{key}]: expected a positive number, got {value!r}"
            )
        dates.append(day)
        amounts.append(float(value) * _UNIT_MULTIPLIERS[unit])
    out = pd.Series(amounts, index=pd.DatetimeIndex(dates, name="date"), name="published_usd")
    if out.index.has_duplicates:
        raise HandTableError(f"{where}.values: duplicate dates")
    return PublishedTotals(
        label=_hand_str(series, "label", where),
        source_url=_hand_str(series, "source_url", where),
        date_read=_hand_date(series, "date_read", where),
        unit=unit,
        values_usd=out.sort_index(),
    )


def _securities(intervals: pd.DataFrame) -> tuple[dict[str, str], set[str]]:
    securities: dict[str, str] = {}
    current: set[str] = set()
    still_member = pd.to_datetime(intervals["end"]).isna().tolist()
    for ticker_value, security, is_current in zip(
        intervals["ticker"].astype(str).tolist(),
        intervals["security"].astype(str).tolist(),
        still_member,
        strict=True,
    ):
        securities.setdefault(ticker_value, security)
        if is_current:
            current.add(ticker_value)
    return securities, current


def build(
    cfg: config_mod.Config | None = None,
    *,
    manifest: cache.Manifest | None = None,
    prices_entry: cache.ManifestEntry | None = None,
    actions_entry: cache.ManifestEntry | None = None,
    start: date | None = None,
) -> MarketCapPanel:
    """The panel on every session from ``sample.start`` to the holdout boundary.

    ``start`` moves the first session EARLIER than ``sample.start`` for a
    descriptor burn-in (:mod:`mafrm.factors.equity`); it can never move the
    end, which is always the holdout boundary.
    """
    cfg = cfg or config_mod.load()
    book = manifest if manifest is not None else cache.Manifest.load()
    frames_entry = book.latest(source=edgar.SOURCE, name=edgar.FRAMES_NAME)
    index_entry = book.latest(source=edgar.SOURCE, name=edgar.INDEX_NAME)
    tickers_entry = book.latest(source=edgar.SOURCE, name=edgar.COMPANY_TICKERS_NAME)
    prices_entry = prices_entry or book.latest(source="yfinance", name=loaders.SP500_PRICES_NAME)
    actions_entry = actions_entry or book.latest(source="yfinance", name=loaders.SP500_ACTIONS_NAME)

    frames = cache.read(frames_entry, manifest=book)
    index = cache.read(index_entry, manifest=book)
    company_tickers = load_company_tickers(book, entry=tickers_entry)
    close = load_close(book, entry=prices_entry)
    splits = load_splits(book, entry=actions_entry)
    intervals = sp500.load_intervals(
        sp500_reference.reference_dir(cfg) / cfg.model.equity_universe.intervals_file
    )

    shares_cfg = cfg.model.equity_shares
    overrides = load_cik_overrides(config_mod.config_dir() / shares_cfg.overrides_file)

    first = pd.Timestamp(cfg.model.sample.start if start is None else start)
    if first > pd.Timestamp(cfg.model.sample.start):
        raise ValueError("market_cap.build: start may only move the first session earlier")
    end = pd.Timestamp(cfg.require_holdout_start())
    sessions = pd.DatetimeIndex(close.index)
    sessions = sessions[(sessions >= first) & (sessions < end)]
    panel = assemble(
        frames=frames,
        index=index,
        company_tickers=company_tickers,
        close=close.reindex(sessions),
        splits=splits,
        intervals=intervals,
        sessions=sessions,
        inputs=(frames_entry, index_entry, tickers_entry, prices_entry, actions_entry),
        consistency_ratio=shares_cfg.consistency_screen.ratio,
        overrides=overrides,
    )
    return panel


def _override_map(
    overrides: tuple[CikOverride, ...], mapping: pd.DataFrame, frames: pd.DataFrame
) -> tuple[dict[str, int], dict[str, str]]:
    """Check every hand row against the pull it is applied to, and refuse a stale one by name.

    Returns ``(predecessors, exclusions)``: ``ticker -> predecessor CIK`` and
    ``ticker -> treat_as`` for the rows that remove a name by hand.
    """
    ciks_with_facts = {int(c) for c in frames["cik"].unique()}
    out: dict[str, int] = {}
    excluded: dict[str, str] = {}
    for row in overrides:
        if row.ticker not in mapping.index:
            raise HandTableError(
                f"cik_overrides: {row.ticker} is not a ticker the membership tables name"
            )
        mapped = mapping.loc[row.ticker, "cik"]
        if pd.isna(mapped) or int(str(mapped)) != row.current_cik:
            raise HandTableError(
                f"cik_overrides: {row.ticker} maps to CIK {mapped} on this pull, not the "
                f"current_cik {row.current_cik} the hand table records; re-read the EDGAR pages "
                "and update the row"
            )
        if row.excludes:
            excluded[row.ticker] = str(row.treat_as)
            continue
        if row.predecessor_cik not in ciks_with_facts:
            raise HandTableError(
                f"cik_overrides: predecessor CIK {row.predecessor_cik} for {row.ticker} has no "
                "fact in the cached frames"
            )
        out[row.ticker] = row.predecessor_cik
    return out, excluded


def assemble(
    *,
    frames: pd.DataFrame,
    index: pd.DataFrame,
    company_tickers: pd.DataFrame,
    close: pd.DataFrame,
    splits: dict[str, pd.Series],
    intervals: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    inputs: tuple[cache.ManifestEntry, ...] = (),
    consistency_ratio: float | None = None,
    overrides: tuple[CikOverride, ...] = (),
) -> MarketCapPanel:
    """The pure join. Every input is a frame; tests build them by hand.

    ``consistency_ratio`` applies SPEC.md 15.4.3 ruling (i) to every filer's
    counts before the point-in-time rule; ``None`` leaves the panel UNSCREENED
    and :attr:`MarketCapPanel.is_model_input` false. ``overrides`` (ruling
    (ii)) read a predecessor CIK's facts beside the mapped CIK's for the named
    tickers; each row is checked against this pull's mapping and frames.
    """
    securities, current = _securities(intervals)
    filer_names = [
        *(
            (int(c), str(n))
            for c, n in frames[["cik", "entity_name"]].drop_duplicates().itertuples(index=False)
        ),
        *(
            (int(c), str(n))
            for c, n in index[["cik", "company_name"]].drop_duplicates().itertuples(index=False)
        ),
    ]
    mapping = edgar.map_ciks(securities, current, company_tickers, filer_names)
    filed = edgar.filing_dates(index)
    override_map, excluded = _override_map(overrides, mapping, frames)

    counts: dict[str, pd.DataFrame] = {}
    extra: dict[str, dict[str, object]] = {}
    dropped_parts: list[pd.DataFrame] = []
    for ticker, row in mapping.iterrows():
        ticker = str(ticker)
        if pd.isna(row["cik"]):
            counts[ticker] = edgar.adjusted_counts(frames.iloc[0:0], filed, None).table
            extra[ticker] = {
                "predecessor_cik": None,
                "facts": 0,
                "used": 0,
                "dropped": 0,
                "no_filing_date": 0,
                "nonpositive": 0,
                "first_filed": pd.NaT,
                "last_filed": pd.NaT,
                "status": str(row["method"]),
                "treat_as": None,
            }
            continue
        ciks = [int(row["cik"])]
        predecessor = override_map.get(ticker)
        if predecessor is not None:
            ciks.append(predecessor)
        facts = frames[frames["cik"].isin(ciks)]
        n_facts = len(facts)
        if ticker in excluded:
            # W7-P2b ruling 3: the filer's non-dimensional fact is one class's
            # count; the name joins the dual-class group by hand, and its facts
            # are counted but never used.
            facts = facts.iloc[0:0]
        adjusted = edgar.adjusted_counts(facts, filed, splits.get(ticker))
        table = adjusted.table
        n_dropped = 0
        if consistency_ratio is not None:
            screened = edgar.consistency_screen(table, ratio=consistency_ratio)
            table = screened.table
            n_dropped = len(screened.dropped)
            if n_dropped:
                dropped_parts.append(screened.dropped.reset_index().assign(ticker=ticker))
        counts[ticker] = table
        used = len(table)
        extra[ticker] = {
            "predecessor_cik": predecessor,
            "facts": n_facts,
            "used": used,
            "dropped": n_dropped,
            "no_filing_date": adjusted.no_filing_date,
            "nonpositive": adjusted.nonpositive,
            "first_filed": table.index.min() if used else pd.NaT,
            "last_filed": table.index.max() if used else pd.NaT,
            "status": "ok" if used else "no_facts",
            "treat_as": excluded.get(ticker),
        }
    dropped_columns = ["ticker", "filed", "end", "val", "accn", "shares", "median", "ratio"]
    if dropped_parts:
        dropped = pd.concat(dropped_parts, ignore_index=True)[dropped_columns]
    else:
        dropped = pd.DataFrame(columns=dropped_columns)
    shares, approximated = edgar.point_in_time_panel(counts, sessions)
    tickers = shares.columns
    close = close.reindex(index=sessions, columns=tickers).astype(float)
    close = close.where(close > 0)
    cap = shares * close
    extra_frame = pd.DataFrame.from_dict(extra, orient="index")
    extra_frame.index.name = "ticker"
    extra_frame["predecessor_cik"] = extra_frame["predecessor_cik"].astype("Int64")
    mapping = mapping.join(extra_frame)
    return MarketCapPanel(
        shares=shares,
        cap=cap,
        approximated=approximated,
        mapping=mapping,
        jumps=edgar.unexplained_jumps(counts),
        sessions=sessions,
        inputs=inputs,
        counts=counts,
        dropped=dropped,
        consistency_ratio=consistency_ratio,
        overrides=override_map,
        excluded=excluded,
    )
