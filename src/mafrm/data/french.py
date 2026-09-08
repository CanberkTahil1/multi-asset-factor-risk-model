"""Ken French's daily research factors, and the risk-free leg taken from them.

The selected short rate for this project (``config/model.yaml``,
``data.risk_free.selected``) is Ken French's daily one-month Treasury bill,
because it is daily from 1926-07 and therefore spans the whole 1961- GSW sample
**with no splice**. A spliced short rate would put a construction seam inside the
excess-return series at exactly the point where the long history is the
differentiator (SPEC.md 3.1).

The zipped multi-table CSVs behind the Data Library are parsed by
``pandas_datareader.famafrench``, which knows where each embedded table starts
and ends, rather than by hand here. ``mafrm.data.loaders`` makes the call;
everything in this module is a pure function of what came back.

**Units, which are the only real trap in this file.** Ken French publishes ``RF``
as *percent per trading day* -- the simple daily rate that compounds over the
month's trading days to the one-month bill rate. FRED's ``DGS1MO``, which this
replaces, is *percent per annum*. They differ by a factor of about 252 and both
are plausible-looking small numbers, so nothing downstream would notice the
substitution. :func:`as_annual_percent` does the conversion explicitly, against
``data.trading_days_per_year``, and is pinned by a hand-computed test.

The conversion is a units change and not an economic claim: multiplying a daily
simple rate by the trading-day count reproduces exactly the daily accrual
``rf_annual * delta / 100`` that :func:`mafrm.data.gsw.constant_maturity_return`
applies with ``delta = 1/trading_days_per_year``. It is not an annualised yield
and must not be reported as one.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from mafrm.config import load
from mafrm.data import cache

__all__ = [
    "CACHE_NAME",
    "FrenchError",
    "as_annual_percent",
    "load_daily_factors",
    "load_daily_factors_unrestricted",
    "load_daily_rf",
    "load_risk_free_annual_percent",
    "normalise_daily_factors",
]

#: Manifest ``name`` for the cached daily factor table.
CACHE_NAME: Final[str] = "research_factors_daily"

#: Manifest ``source`` for everything from the Data Library.
SOURCE: Final[str] = "ken_french"


class FrenchError(RuntimeError):
    """The Data Library table is not the shape this project expects."""


def normalise_daily_factors(frame: pd.DataFrame, *, rf_column: str) -> pd.DataFrame:
    """Tidy one ``famafrench`` table into something cacheable.

    ``pandas_datareader`` returns a ``DatetimeIndex`` already; this sorts it,
    names it ``date``, checks the risk-free column is present and refuses a
    duplicated date. Values are left in the publisher's units -- percent per
    trading day -- because a cache that silently rescales its source is a cache
    whose numbers cannot be checked against the publisher's own file.
    """
    if rf_column not in frame.columns:
        raise FrenchError(
            f"Ken French daily table has no {rf_column!r} column; got {list(frame.columns)}"
        )
    # pandas_datareader returns a DatetimeIndex at 0.10 and a PeriodIndex at
    # 0.11 for the same daily table. Both are accepted and normalised to
    # timestamps; anything else raises rather than being coerced, because a
    # silently reinterpreted index is how a daily series becomes a monthly one.
    #
    # This was found by locking the dependency (W1-P5) rather than by a test
    # failing in production: `uv lock` resolved 0.11.1 where the working
    # environment had 0.10.0, and the type guard fired on the first fetch.
    # For a daily period, `to_timestamp()` returns the period's own start, so
    # the dates are unchanged and the cache does not churn.
    index = frame.index
    if isinstance(index, pd.PeriodIndex):
        index = index.to_timestamp()
    elif not isinstance(index, pd.DatetimeIndex):
        raise FrenchError(
            "expected a DatetimeIndex or PeriodIndex from pandas_datareader, got "
            f"{type(frame.index).__name__}"
        )

    tidy = frame.copy()
    # Pinned to nanosecond resolution. `PeriodIndex.to_timestamp()` yields
    # microseconds on pandas_datareader 0.11, where 0.10 gave nanoseconds, and
    # parquet writes the two as different column types -- so the same data cached
    # under two library versions hashes differently for a reason that has nothing
    # to do with the data, and `make verify` would report drift that is not drift.
    # Every other source in this project produces `datetime64[ns]`; a contract
    # test asserts that none of them stops.
    tidy.index = pd.DatetimeIndex(index).normalize().astype("datetime64[ns]")
    tidy.index.name = "date"
    tidy = tidy.sort_index()
    duplicated = tidy.index[tidy.index.duplicated(keep=False)]
    if len(duplicated):
        raise FrenchError(
            f"Ken French daily table repeats {len(duplicated)} date(s), "
            f"first {duplicated[0].date()}"
        )
    return tidy.astype(float)


def as_annual_percent(daily_percent: pd.Series, *, trading_days_per_year: int) -> pd.Series:
    """Percent per trading day -> percent per annum, on this project's convention.

    ``rf_annual = rf_daily * trading_days_per_year``, chosen so that the daily
    accrual ``rf_annual * (1 / trading_days_per_year) / 100`` used by
    :func:`mafrm.data.gsw.constant_maturity_return` returns exactly
    ``rf_daily / 100`` -- the fraction Ken French actually published for that
    day. No compounding is applied and none should be: the round trip has to be
    the identity or the substitution for ``DGS1MO`` changes the excess returns
    by more than the choice of series does.
    """
    if trading_days_per_year <= 0:
        raise FrenchError(f"trading_days_per_year must be positive, got {trading_days_per_year}")
    annual = daily_percent.astype(float) * float(trading_days_per_year)
    annual.name = f"{daily_percent.name}_annual_pct" if daily_percent.name else "rf_annual_pct"
    return annual


# ---------------------------------------------------------------------------
# Cache access. Never fetches -- CLAUDE.md invariant 1.
# ---------------------------------------------------------------------------


def load_daily_factors() -> pd.DataFrame:
    """Read the cached daily research-factor table, in published units."""
    manifest = cache.Manifest.load()
    entry = manifest.latest(source=SOURCE, name=CACHE_NAME)
    return cache.read(entry, manifest=manifest)


def load_daily_factors_unrestricted(*, reason: str) -> pd.DataFrame:
    """The Ken French daily table over its full published history.

    For the data-layer cross-checks, which ask whether a vendor series agrees
    with CRSP over everything cached rather than over the model window. Ordinary
    model use calls :func:`load_daily_factors`, which stops at the holdout
    boundary (:mod:`mafrm.data.holdout`).
    """
    manifest = cache.Manifest.load()
    entry = manifest.latest(source=SOURCE, name=CACHE_NAME)
    return cache.read_unrestricted(entry, manifest=manifest, reason=reason)


def load_daily_rf() -> pd.Series:
    """The risk-free column as published: percent per trading day."""
    config = load().model.data.risk_free.ken_french_daily_rf
    frame = load_daily_factors()
    if config.rf_column not in frame.columns:
        raise FrenchError(
            f"cached Ken French table has no {config.rf_column!r} column; got {list(frame.columns)}"
        )
    series = frame[config.rf_column].astype(float)
    series.name = config.rf_column
    return series


def load_risk_free_annual_percent() -> pd.Series:
    """The risk-free leg in ``DGS1MO``'s units, so the two are interchangeable."""
    config = load().model.data
    return as_annual_percent(load_daily_rf(), trading_days_per_year=config.trading_days_per_year)
