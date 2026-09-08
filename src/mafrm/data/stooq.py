"""Stooq daily bars: a second opinion on the raw prices Yahoo serves.

Yahoo is the primary price source and it is the one with the known pathologies
(SPEC.md 3.5: backwards-rewritten adjusted closes, a patch release that dropped
``Adj Close``, mis-adjusted weekly bars). A cross-check against an independent
vendor is what turns "we assume the closes are right" into a number, so Stooq is
cached alongside and the two raw close series are compared.

It is a **cross-check only**. Nothing in the production path reads it, and the
comparison reports rather than gates -- see :func:`compare_closes`. There is no
threshold in ``config/model.yaml`` for how far the two vendors may drift,
because no such threshold has been validated against real Stooq data (see
below), and CLAUDE.md invariant 9 forbids inventing one.

**Status as of 2026-08-29: Stooq is not machine-readable.** Every request to
``stooq.com/q/d/l/`` -- including through ``pandas_datareader``, which builds the
same URL -- returns a JavaScript proof-of-work browser challenge instead of CSV.
The page asks the client to find a SHA-256 preimage and POST it to ``/__verify``
to earn a cookie. That is an access control, and this project does not defeat
access controls, so :func:`parse_csv` detects the challenge and raises a
:class:`StooqUnavailable` naming it. The loader stays wired up: if Stooq serves
CSV again, ``make data`` picks it up with no code change, and the contract test
skips with a stated reason in the meantime rather than silently passing.
"""

from __future__ import annotations

import io
from typing import Final

import pandas as pd

from mafrm.data import cache

__all__ = [
    "COLUMNS",
    "SOURCE",
    "StooqError",
    "StooqUnavailable",
    "compare_closes",
    "load_prices",
    "parse_csv",
    "symbol_for",
]

#: Manifest ``source`` for Stooq pulls.
SOURCE: Final[str] = "stooq"

#: The columns Stooq's daily CSV export carries. ``Date`` becomes the index.
COLUMNS: Final[tuple[str, ...]] = ("Open", "High", "Low", "Close", "Volume")

#: Markers of the interstitial the site serves in place of CSV.
_CHALLENGE_MARKERS: Final[tuple[str, ...]] = ("__verify", "requires JavaScript", "crypto.subtle")


class StooqError(RuntimeError):
    """A Stooq response is not a daily CSV export."""


class StooqUnavailable(StooqError):
    """Stooq served a browser challenge rather than data.

    Not a bug in this project and not something to work around: the site is
    asking for proof that the client is a browser, and satisfying it would mean
    circumventing an access control the operator put up deliberately.
    """


def symbol_for(ticker: str, *, template: str) -> str:
    """Vendor symbol for a US ticker, e.g. ``SPY`` -> ``spy.us``."""
    return template.format(ticker=ticker.lower())


def parse_csv(text: str, *, symbol: str) -> pd.DataFrame:
    """Stooq's daily CSV export -> a dated OHLCV frame.

    Raises :class:`StooqUnavailable` when the body is the browser challenge, and
    :class:`StooqError` when it is anything else that is not the documented CSV.
    Distinguishing the two matters: the first is the site's decision and the
    second is ours to fix.
    """
    head = text[:2000]
    if any(marker in head for marker in _CHALLENGE_MARKERS):
        raise StooqUnavailable(
            f"stooq returned a JavaScript proof-of-work browser challenge for {symbol!r} "
            "instead of CSV. Solving it would mean circumventing an access control, which "
            "this project does not do. The Stooq cross-check is unavailable until the site "
            "serves CSV to plain clients again."
        )
    if "<html" in head.lower() or "<!doctype" in head.lower():
        raise StooqError(f"stooq returned HTML for {symbol!r}, not CSV: {head[:200]!r}")

    frame = pd.read_csv(io.StringIO(text))
    if "Date" not in frame.columns:
        raise StooqError(
            f"stooq export for {symbol!r} has no 'Date' column; got {list(frame.columns)}"
        )
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise StooqError(f"stooq export for {symbol!r} is missing {missing}")

    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame = frame.set_index("Date").sort_index()
    frame.index.name = "date"
    return frame[list(COLUMNS)].astype(float)


def compare_closes(
    primary: pd.Series, secondary: pd.Series, *, label: str = "yfinance vs stooq"
) -> pd.Series:
    """Relative close differences on the dates both vendors quote.

    Reported, never gated. Two vendors disagreeing about a close is ordinary --
    different consolidation feeds, different handling of a late print -- and the
    useful artefact is the distribution of the disagreement, not a pass/fail on a
    number nobody has calibrated.
    """
    common = primary.index.intersection(secondary.index)
    if len(common) == 0:
        raise StooqError(f"{label}: the two series share no dates")
    difference = (primary.loc[common].astype(float) / secondary.loc[common].astype(float)) - 1.0
    difference.name = "relative_close_difference"
    return difference.dropna()


def load_prices(ticker: str) -> pd.DataFrame:
    """Read a cached Stooq pull. ``mafrm.data.loaders`` is what fetches it."""
    manifest = cache.Manifest.load()
    entry = manifest.latest(source=SOURCE, name=f"{ticker.lower()}_prices")
    return cache.read(entry, manifest=manifest)
