"""The only module in this project permitted to touch the network.

CLAUDE.md invariant 1. Everything else -- ``factors``, ``risk``, ``costs``,
``backtest``, and the other modules in ``data`` -- reads from the cache. Each
loader here is a thin pair: a fetcher closure that produces bytes, and a parser
that lives elsewhere and is testable without a socket. The cache decides whether
anything is written (``mafrm.data.cache``), so re-running ``make data`` on an
unchanged source is free and leaves the manifest byte-identical.

Nothing here is imported by a test. ``tests/conftest.py`` fails any unmarked
test that opens a socket, and the parsers are exercised against fixture text.

**The holdout boundary is enforced on the way out, not here.** CLAUDE.md
invariant 5 is applied by :func:`mafrm.data.cache.read`, which truncates every
cached artefact at ``model.sample.holdout_start`` by default; seeing past it
requires :func:`mafrm.data.cache.read_unrestricted` and a stated reason. The
policy lives in :mod:`mafrm.data.holdout` rather than in this module for one
mechanical reason -- this module imports ``cache`` and every per-source reader,
so any of them importing it back would be a cycle at import time. W2-P3 moved
that guard from discipline to the loader after an ad-hoc scoping read crossed
the boundary; ``tests/test_holdout_guard.py`` pins the sanctioned crossings.

Sources, and the one rule each carries:

======================  ====================================================
Fed (GSW curves)        Sends a browser-ish agent; the default urllib one 403s.
yfinance (ETF bars)     Daily only, ``auto_adjust=False``, raw OHLCV and the
                        actions table cached as **two** artefacts.
FRED / ALFRED           The API with a key, never ``fredgraph.csv``; a factor
                        input must have vintages or the load raises.
Ken French              Through ``pandas_datareader.famafrench``, not by hand.
AQR                     Three workbooks, three banner heights, named explicitly.
Stooq                   Cross-check only; currently behind a browser challenge.
Wikipedia (S&P 500)     Two pages, parsed by ``mafrm.data.sp500``; CC BY-SA 4.0.
EDGAR (XBRL frames)     One request per calendar quarter for every filer, plus
                        the quarterly XBRL index for filing dates; the
                        User-Agent EDGAR requires comes from the environment.
yfinance (S&P 500 bars) Every ticker the two pages name, with a per-ticker
                        status -- ``ok`` / ``empty`` / ``failed`` -- recorded
                        in the manifest, and one retry of ``failed``.
Ken French (Siccodes49) The FF49 industry definitions, one zipped text file,
                        parsed by ``mafrm.data.sic`` (SPEC.md 15.6.1).
EDGAR (submissions)     One request per mapped CIK for the CURRENT SIC, the
                        older filing pages that reach ``sample.start``, and
                        one SGML header per ticker for the first in-sample
                        10-K's SIC; the same status contract as the frames.
======================  ====================================================
"""

from __future__ import annotations

import dataclasses
import functools
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import Final, Literal

import pandas as pd

from mafrm import secrets
from mafrm.config import AqrDataset, EtfConfig, FredSeries, load
from mafrm.data import (
    aqr,
    cache,
    contracts,
    edgar,
    fred,
    french,
    gsw,
    market_cap,
    sic,
    sp500,
    stooq,
)

__all__ = [
    "EMPTY",
    "FAILED",
    "OK",
    "SP500_ACTIONS_NAME",
    "SP500_COVERAGE_NAME",
    "SP500_PRICES_NAME",
    "LoaderError",
    "TickerPull",
    "TickerStatus",
    "check_status_identity",
    "check_vintage_availability",
    "classify_history",
    "fetch_aqr",
    "fetch_edgar_shares",
    "fetch_edgar_sic",
    "fetch_edgar_sic_history",
    "fetch_etf",
    "fetch_fed_curve",
    "fetch_fred_series",
    "fetch_ken_french",
    "fetch_ken_french_siccodes",
    "fetch_risk_free",
    "fetch_sp500_bars",
    "fetch_sp500_wikipedia",
    "fetch_stooq",
    "numeric_actions",
    "pull_histories",
    "refresh_all",
    "sp500_requested_tickers",
]

#: Seconds. The Fed's curve files are ~16 MB and served slowly.
_TIMEOUT: Final[int] = 180

#: The Fed rejects the default urllib agent with 403.
_USER_AGENT: Final[str] = "mafrm/0.1 (research; contact via repository)"

#: AQR's CDN and Stooq both refuse a non-browser agent outright. Sent to be
#: served the file at all, not to disguise what this is -- everything fetched is
#: a public research file and every pull is recorded in data/manifest.json.
_BROWSER_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class LoaderError(RuntimeError):
    """A source returned something this project will not cache."""


def _download(url: str, *, user_agent: str = _USER_AGENT) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        payload: bytes = response.read()
    if not payload:
        raise LoaderError(f"{url} returned an empty body")
    return payload


# ---------------------------------------------------------------------------
# Fed fitted yield curves (SPEC.md 3.2)
# ---------------------------------------------------------------------------


def fetch_fed_curve(kind: gsw.CurveKind) -> cache.CacheResult:
    """Download and cache one Fed fitted zero curve. SPEC.md 3.2."""
    config = load().model.data
    curve = config.nominal_curve if kind == "nominal" else config.real_curve
    name = "nominal_curve" if kind == "nominal" else "real_curve"

    def fetch() -> cache.Fetched:
        payload = _download(curve.url)
        frame = gsw.parse_fed_curve_csv(
            payload.decode("utf-8", errors="replace"), yield_prefix=curve.yield_prefix
        )
        return cache.Fetched(frame=frame, source_bytes=payload)

    return cache.fetch_raw(source="fed", name=name, url=curve.url, fetcher=fetch)


# ---------------------------------------------------------------------------
# FRED and ALFRED (SPEC.md 3.5, the revisions trap)
# ---------------------------------------------------------------------------


def _fred_url(path: str, **params: str) -> str:
    """Build one FRED API URL. The key is read from the environment, never stored."""
    config = load().model.data.fred
    key = secrets.require(
        config.api_key_env_var,
        hint="Free, request one at https://fredaccount.stlouisfed.org/apikeys",
    )
    query = dict(params, api_key=key, file_type="json")
    return f"{config.api_base_url}/{path}?{urllib.parse.urlencode(query)}"


def _redact(url: str) -> str:
    """The manifest records the URL, so the key must not be in it."""
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    scrubbed = [(k, "REDACTED" if k == "api_key" else v) for k, v in query]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(scrubbed)))


def _fetch_fred_json(path: str, **params: str) -> tuple[bytes, str]:
    """Return the raw payload and the key-redacted URL to record against it."""
    url = _fred_url(path, **params)
    return _download(url), _redact(url)


def _vintage_frame(vintage_dates: Sequence[date]) -> pd.DataFrame:
    """Vintage dates as a cacheable table.

    The dates are carried as a *column* as well as the index. A frame with an
    index and no columns is ``empty`` as far as pandas is concerned, so the cache
    refuses it as a silently broken fetch -- correctly, since it cannot tell that
    case apart from a parser that returned nothing.
    """
    index = pd.DatetimeIndex(pd.to_datetime(list(vintage_dates)), name="date")
    return pd.DataFrame({"vintage_date": index}, index=index)


def fetch_fred_series(series: FredSeries) -> list[cache.CacheResult]:
    """Cache one FRED series three ways: current, initial-release, vintage dates.

    SPEC.md 3.5. The **current** table is every observation at its latest
    revision -- right for describing history, look-ahead for trading it. The
    **initial-release** table (``output_type=4``) carries, per observation, the
    ``realtime_start`` on which it first became public, and is the only one a
    factor input may read. The **vintage dates** are cached so that
    :func:`check_vintage_availability` can answer offline.

    A ``factor_input`` series with no vintages raises
    :class:`mafrm.data.fred.NoVintagesError` here, at fetch time, rather than
    silently supplying current revisions to a backtest.
    """
    config = load().model.data.fred
    results: list[cache.CacheResult] = []

    # 1. Vintage dates first: they decide whether the rest is usable at all.
    # No `limit`: FRED caps this endpoint at 10,000 and rejects a larger value
    # with a bare HTTP 400 that names neither the parameter nor the ceiling. The
    # default returns every vintage for both configured series (DGS1MO, the
    # longer of the two, has about 5,100).
    vintage_payload, vintage_url = _fetch_fred_json("series/vintagedates", series_id=series.id)
    vintage_dates = fred.parse_vintage_dates(vintage_payload, series.id)
    if series.requires_vintages and not vintage_dates:
        raise fred.NoVintagesError(
            f"{series.id} is classified revised_statistic in config/model.yaml but ALFRED "
            "has no vintages for it, so every value would enter the model at its latest "
            "revision -- look-ahead through revision (SPEC.md 3.5). Either it is really "
            "market-observed, in which case reclassify it and say why, or it must be "
            "lagged conservatively or dropped."
        )

    results.append(
        cache.fetch_raw(
            source="fred",
            name=f"{series.name}_vintage_dates",
            url=vintage_url,
            fetcher=lambda: cache.Fetched(
                frame=_vintage_frame(vintage_dates), source_bytes=vintage_payload
            ),
            allow_empty=not vintage_dates,
        )
    )

    # 2. The current series.
    def fetch_current() -> cache.Fetched:
        payload, _ = _fetch_fred_json("series/observations", series_id=series.id)
        return cache.Fetched(
            frame=fred.parse_observations(payload, series.id), source_bytes=payload
        )

    results.append(
        cache.fetch_raw(
            source="fred",
            name=series.name,
            url=_redact(_fred_url("series/observations", series_id=series.id)),
            fetcher=fetch_current,
        )
    )

    # 3. The initial-release series. Two API constraints shape this, and both
    #    surface as a bare HTTP 400 rather than as anything self-describing:
    #      * output_type=4 REJECTS the default real-time period ("No vintage
    #        dates exist for the specified real-time period"), so FRED's
    #        documented full span has to be named explicitly; and
    #      * one request may span at most `vintage_chunk_size` vintages, so a
    #        long-running daily series has to be fetched in windows.
    if vintage_dates:

        def fetch_first_release() -> cache.Fetched:
            frames: list[pd.DataFrame] = []
            payloads: list[bytes] = []
            for start, end in _vintage_windows(vintage_dates, config.vintage_chunk_size):
                payload, _ = _fetch_fred_json(
                    "series/observations",
                    series_id=series.id,
                    output_type="4",
                    realtime_start=start,
                    realtime_end=end,
                )
                frames.append(fred.parse_observations(payload, series.id))
                payloads.append(payload)
            return cache.Fetched(frame=fred.first_releases(frames), source_bytes=b"".join(payloads))

        results.append(
            cache.fetch_raw(
                source="fred",
                name=f"{series.name}{fred.FIRST_RELEASE_SUFFIX}",
                url=_redact(_fred_url("series/observations", series_id=series.id, output_type="4")),
                fetcher=fetch_first_release,
            )
        )
    return results


def _vintage_windows(vintage_dates: Sequence[date], chunk_size: int) -> list[tuple[str, str]]:
    """Split the vintage list into real-time windows FRED will accept.

    Windows are half-open at the top -- each starts on the first vintage of its
    chunk and ends the day before the next chunk begins -- so no vintage is
    counted twice and none falls between two windows. The final window runs to
    FRED's documented upper sentinel so that observations released after the last
    recorded vintage are still returned.
    """
    config = load().model.data.fred
    ordered = sorted(vintage_dates)
    if not ordered:
        return []
    windows: list[tuple[str, str]] = []
    for position in range(0, len(ordered), chunk_size):
        chunk = ordered[position : position + chunk_size]
        following = ordered[position + chunk_size : position + chunk_size + 1]
        end = (
            (following[0] - timedelta(days=1)).isoformat()
            if following
            else config.vintage_realtime_end
        )
        windows.append((chunk[0].isoformat(), end))
    # The earliest window must reach back past the first vintage so that an
    # observation dated before ALFRED took the series on is still attributed to
    # its first *recorded* release rather than dropped.
    windows[0] = (config.vintage_realtime_start.isoformat(), windows[0][1])
    return windows


def check_vintage_availability(series_id: str | None = None) -> tuple[fred.VintageCoverage, ...]:
    """Report ALFRED coverage per configured series, raising if a factor input has none.

    SPEC.md 3.5: "Check vintage coverage per series; drop or conservatively lag
    anything without vintages." This is that check, made a function so it runs at
    load time from the cache rather than being rediscovered in a backtest.

    **Only a ``revised_statistic`` is required to have vintages.** A
    market-observed series -- a price, a yield, an index computed from the day's
    quotes -- has no estimate to restate, so partial or absent coverage is a
    fact about when ALFRED adopted it rather than a defect. ``DTWEXBGS`` runs
    from 2006 with vintages only from 2019, and that gap owes no drop-or-lag
    decision for exactly that reason. Vintages are still fetched wherever they
    exist: they cost one request and they are the only way to answer "was this
    revised after all?" if the classification is ever doubted.

    Partial coverage never raises. Zero coverage raises only for a revised
    statistic, where it is fatal.
    """
    config = load().model.data.fred
    wanted = config.series if series_id is None else (config.by_id(series_id),)
    reports: list[fred.VintageCoverage] = []
    for series in wanted:
        cached = fred.load_coverage(series.name)
        # `load_coverage` keys the cache by config name; relabel with the FRED id
        # so the report reads as the series a human would look up.
        report = dataclasses.replace(cached, series_id=series.id)
        if series.requires_vintages:
            report.require_vintages()
        reports.append(report)
    return tuple(reports)


def fetch_risk_free() -> cache.CacheResult:
    """Cache the interim FRED short rate. SUPERSEDED as the leg by Ken French.

    Kept because it is the comparand for that substitution: the two series
    should agree closely from ``DGS1MO``'s 2001-07-31 start, and a divergence
    means the percent-per-day to percent-per-annum conversion in
    :mod:`mafrm.data.french` is wrong. Now pulled through the API like every
    other FRED series rather than through the keyless CSV endpoint.
    """
    config = load().model.data.fred
    series = config.by_id("DGS1MO")
    results = fetch_fred_series(series)
    for result in results:
        if result.entry.name == series.name:
            return result
    raise LoaderError(f"{series.id}: the current series was not cached")


# ---------------------------------------------------------------------------
# ETF bars (SPEC.md 3.5, the adjusted-close trap)
# ---------------------------------------------------------------------------


def fetch_etf(ticker: str, *, interval: str | None = None) -> tuple[cache.CacheResult, ...]:
    """Cache RAW unadjusted OHLCV and the corporate-action table, separately.

    SPEC.md 3.5: ``auto_adjust=False`` and two artefacts, because the spread
    estimator needs actually-traded prices and the return series needs an
    adjustment we apply ourselves, deterministically, as of a stated date
    (:func:`mafrm.data.prices.adjusted_close`).

    ``interval`` exists only so that it can be refused. CLAUDE.md invariant 2:
    Yahoo mis-applies a mid-week dividend to the whole week's close (yfinance
    issue #1273) and inflates returns with no error, so ``'1wk'`` and ``'1mo'``
    raise :class:`mafrm.data.contracts.IntervalError` before any request is made.
    """
    import yfinance  # imported here so the module is importable without a network stack

    config = load().model.data.etf
    contracts.require_daily_interval(interval if interval is not None else config.interval)

    url = f"https://finance.yahoo.com/quote/{ticker}/history"
    pulled: dict[str, pd.DataFrame] = {}

    def pull() -> pd.DataFrame:
        if "frame" in pulled:
            return pulled["frame"]
        history = yfinance.Ticker(ticker).history(
            period=config.period,
            interval=config.interval,
            auto_adjust=config.auto_adjust,
            actions=True,
        )
        if history.empty:
            raise LoaderError(f"yfinance returned no rows for {ticker}")
        history.index = pd.DatetimeIndex(history.index).tz_localize(None)
        history.index.name = "date"
        frame = pd.DataFrame(history)
        # SPEC.md 3.5, the `Adj Close` trap: assert on the RESPONSE column set.
        # yfinance 0.2.51 flipped auto_adjust to True in a patch release and
        # dropped the column, which changed the meaning of existing code without
        # raising anywhere. If it is missing now, `Close` is no longer raw.
        contracts.check_columns(
            frame,
            config.required_response_columns,
            label=f"yfinance/{ticker} response",
        )
        pulled["frame"] = frame
        return frame

    def fetch_prices() -> cache.Fetched:
        history = pull()
        return cache.Fetched(frame=history[list(config.price_columns)])

    def fetch_actions() -> cache.Fetched:
        history = pull()
        return cache.Fetched(frame=history[list(config.action_columns)])

    prices = cache.fetch_raw(
        source="yfinance", name=f"{ticker.lower()}_prices", url=url, fetcher=fetch_prices
    )
    actions = cache.fetch_raw(
        source="yfinance", name=f"{ticker.lower()}_actions", url=url, fetcher=fetch_actions
    )
    return prices, actions


# ---------------------------------------------------------------------------
# S&P 500 membership -- two Wikipedia pages, and the bars behind them (SPEC.md 15.3)
# ---------------------------------------------------------------------------

#: Manifest names of the three artefacts behind the membership matrix.
SP500_PRICES_NAME: Final[str] = "sp500_prices"
SP500_ACTIONS_NAME: Final[str] = "sp500_actions"
SP500_COVERAGE_NAME: Final[str] = "sp500_coverage"

#: Yahoo's chart endpoint -- what yfinance's history call hits per ticker, and
#: what the delisting probe asks directly.
_YAHOO_CHART_URL: Final[str] = "https://query2.finance.yahoo.com/v8/finance/chart/"

#: Concurrent history requests. A request-batching size, not a model
#: parameter: it changes how long a pull takes and nothing about what is
#: cached. Logged under "constants held in code" in experiments.md (W7-P1).
_YFINANCE_WORKERS: Final[int] = 8

#: SPEC.md 15.3.1 ruling 6, a CONTRACT rather than a parameter: a ticker whose
#: request failed is retried exactly once, and only ``empty`` -- the provider
#: answered and has no history -- becomes "member, no data".
_YFINANCE_RETRIES: Final[int] = 1

#: Seconds for the delisting probe; one small JSON answer per ticker.
_PROBE_TIMEOUT: Final[int] = 30

TickerStatus = Literal["ok", "empty", "failed"]
OK: Final[TickerStatus] = "ok"
EMPTY: Final[TickerStatus] = "empty"
FAILED: Final[TickerStatus] = "failed"

#: yfinance's exception class names, matched by name on the exception's MRO so
#: that this module's classification is testable without importing yfinance.
_PRICES_MISSING: Final[str] = "YFPricesMissingError"
_TZ_MISSING: Final[str] = "YFTzMissingError"


@dataclasses.dataclass(frozen=True)
class TickerPull:
    """One ticker's outcome after the pull and its retry (ruling 6)."""

    ticker: str
    status: TickerStatus
    #: 1, or 2 when the first attempt failed and the retry ran.
    attempts: int
    #: The raw response frame for ``ok``; ``None`` otherwise.
    frame: pd.DataFrame | None
    #: ``type: message`` of the last exception, or the probe's verdict.
    error: str | None


def classify_history(
    ticker: str,
    outcome: pd.DataFrame | BaseException,
    *,
    answered_no_data: Callable[[str], bool],
) -> tuple[TickerStatus, str | None]:
    """Map one history attempt's outcome to a status. SPEC.md 15.3.1 ruling 6.

    ``ok``     -- a non-empty frame came back.
    ``empty``  -- the provider ANSWERED and has no history: yfinance raised its
                  prices-missing error (Yahoo returned a chart with no prices),
                  or it raised its timezone-missing error or returned an empty
                  frame AND the chart endpoint, asked directly, answered with
                  its explicit "no data found" error payload.
    ``failed`` -- anything else: a rate limit, an HTTP or transport error, a
                  timeout, or a timezone-missing error the probe could not
                  confirm as a delisting.

    The probe exists because yfinance's timezone lookup swallows transport
    errors and reports them as "no timezone found" -- the same words it uses
    for a delisted symbol. That is exactly the indistinguishability the lost
    W7-P1 attempt found in what it had built, and this is the contract that
    closes it.
    """
    if isinstance(outcome, pd.DataFrame):
        if not outcome.empty:
            return OK, None
        return _probe_verdict(ticker, "empty frame", answered_no_data)
    names = {cls.__name__ for cls in type(outcome).__mro__}
    detail = f"{type(outcome).__name__}: {outcome}"
    if _PRICES_MISSING in names:
        return EMPTY, detail
    if _TZ_MISSING in names:
        return _probe_verdict(ticker, detail, answered_no_data)
    return FAILED, detail


def _probe_verdict(
    ticker: str, detail: str, answered_no_data: Callable[[str], bool]
) -> tuple[TickerStatus, str | None]:
    if answered_no_data(ticker):
        return EMPTY, f"{detail}; chart endpoint answered: no data found"
    return FAILED, f"{detail}; chart endpoint did not confirm a delisting"


#: The probe is serialised: one request at a time, through one session, so a
#: burst of delisted names does not read as a scrape. The first W7-P1 pull
#: probed from eight threads with a plain urllib client and was answered
#: ``429 Too Many Requests`` for 112 of 120 candidates -- an honest ``failed``
#: under the contract, and a defect in the probe's transport, not in Yahoo.
_PROBE_LOCK: Final[threading.Lock] = threading.Lock()


def yahoo_answered_no_data(ticker: str) -> bool:
    """Ask the chart endpoint directly whether Yahoo has no data for ``ticker``.

    ``True`` only when Yahoo ANSWERS with its explicit error payload
    (``{"chart": {"result": null, "error": {...}}}``, served with HTTP 404).
    A transport failure, a timeout, a rate limit, a non-JSON body or a
    successful answer all return ``False``: none of them is evidence of a
    delisting.

    Goes through the same browser-impersonating HTTP client yfinance itself
    uses (``curl_cffi``), because Yahoo rate-limits a plain client with a
    browser user-agent string where it answers the impersonated one.
    """
    from curl_cffi import requests  # yfinance's own HTTP client; imported lazily like yfinance

    url = f"{_YAHOO_CHART_URL}{urllib.parse.quote(ticker)}"
    with _PROBE_LOCK:
        try:
            response = requests.get(
                url,
                params={"range": "1d", "interval": "1d"},
                impersonate="chrome",
                timeout=_PROBE_TIMEOUT,
            )
            body = json.loads(response.content)
        except Exception:
            return False
    chart = body.get("chart") if isinstance(body, dict) else None
    return isinstance(chart, dict) and chart.get("error") is not None


def pull_histories(
    tickers: Sequence[str],
    *,
    history: Callable[[str], pd.DataFrame],
    answered_no_data: Callable[[str], bool],
    retries: int = _YFINANCE_RETRIES,
    workers: int = _YFINANCE_WORKERS,
) -> tuple[TickerPull, ...]:
    """Fetch every ticker once, retry the ``failed`` ones, classify each.

    Pure orchestration over two injected callables so that the retry and the
    classification are testable without a socket: ``history`` performs one
    request and either returns a frame or raises; ``answered_no_data`` is the
    delisting probe.
    """
    if retries < 0:
        raise LoaderError("retries must be non-negative")

    def attempt(ticker: str) -> pd.DataFrame | BaseException:
        try:
            return history(ticker)
        except Exception as exc:  # classified, never swallowed: see classify_history
            return exc

    results: dict[str, TickerPull] = {}
    pending = list(dict.fromkeys(tickers))
    attempts = 0
    while pending and attempts <= retries:
        attempts += 1
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            outcomes = list(pool.map(attempt, pending))
        still: list[str] = []
        for ticker, outcome in zip(pending, outcomes, strict=True):
            status, error = classify_history(ticker, outcome, answered_no_data=answered_no_data)
            frame = outcome if isinstance(outcome, pd.DataFrame) and status == OK else None
            results[ticker] = TickerPull(ticker, status, attempts, frame, error)
            if status == FAILED:
                still.append(ticker)
        pending = still
    return tuple(results[t] for t in dict.fromkeys(tickers))


def check_status_identity(tickers: Sequence[str], pulls: Sequence[TickerPull]) -> dict[str, int]:
    """Assert ``requested = ok + empty + failed`` and return the counts. W7-P1b.

    The lost attempt's batched download dropped tickers with no error anywhere;
    a printed table cannot catch that on the next pull and this does: every
    requested ticker must come back with exactly one status, and the three
    statuses must sum to the request.
    """
    requested = list(dict.fromkeys(tickers))
    statuses = {p.ticker: p.status for p in pulls}
    counts: dict[str, int] = {OK: 0, EMPTY: 0, FAILED: 0}
    for pull in pulls:
        counts[pull.status] = counts.get(pull.status, 0) + 1
    missing = [t for t in requested if t not in statuses]
    extra = [t for t in statuses if t not in set(requested)]
    duplicated = len(pulls) - len(statuses)
    total = counts[OK] + counts[EMPTY] + counts[FAILED]
    if missing or extra or duplicated or total != len(requested) or len(pulls) != len(requested):
        raise LoaderError(
            "status identity violated: requested "
            f"{len(requested)} != ok {counts[OK]} + empty {counts[EMPTY]} + failed "
            f"{counts[FAILED]} = {total}; missing {missing[:10]}, extra {extra[:10]}, "
            f"duplicated {duplicated}. A ticker was dropped without a status (SPEC.md 15.3.1 "
            "ruling 6)."
        )
    return {"requested": len(requested), **counts}


def fetch_sp500_wikipedia() -> tuple[cache.CacheResult, cache.CacheResult]:
    """Cache the two Wikipedia tables the membership walk reads. SPEC.md 15.3.1.

    The current-constituents table (with its ``Date added`` column) and the
    *Historical components* changes table, each parsed by
    :mod:`mafrm.data.sp500` and cached raw with the page bytes' hash. Wikipedia
    text is CC BY-SA 4.0; the parsed tables are a derived work and the
    committed reference files carry that licence in their header (ruling 3).
    """
    config = load().model.equity_universe

    def fetch_constituents() -> cache.Fetched:
        payload = _download(config.constituents_url)
        return cache.Fetched(frame=sp500.parse_constituents(payload), source_bytes=payload)

    def fetch_changes() -> cache.Fetched:
        payload = _download(config.changes_url)
        return cache.Fetched(frame=sp500.parse_changes(payload), source_bytes=payload)

    constituents = cache.fetch_raw(
        source=sp500.SOURCE,
        name=sp500.CONSTITUENTS_NAME,
        url=config.constituents_url,
        fetcher=fetch_constituents,
        date_column="date_added",
    )
    changes = cache.fetch_raw(
        source=sp500.SOURCE,
        name=sp500.CHANGES_NAME,
        url=config.changes_url,
        fetcher=fetch_changes,
        date_column="effective_date",
    )
    return constituents, changes


def sp500_requested_tickers() -> tuple[str, ...]:
    """Every ticker the cached Wikipedia tables name -- what the bars pull asks for."""
    manifest = cache.Manifest.load()
    reason = (
        "the membership tables are a SOURCE snapshot dated by column, not by index: a "
        "current constituent list is by construction read after the holdout boundary, "
        "and the model-facing reads of the bars behind it go through cache.read"
    )
    constituents = cache.read_unrestricted(
        manifest.latest(source=sp500.SOURCE, name=sp500.CONSTITUENTS_NAME),
        manifest=manifest,
        reason=reason,
    )
    changes = cache.read_unrestricted(
        manifest.latest(source=sp500.SOURCE, name=sp500.CHANGES_NAME),
        manifest=manifest,
        reason=reason,
    )
    return sp500.universe_tickers(constituents, changes)


def _yfinance_history(ticker: str, config: EtfConfig) -> pd.DataFrame:
    """One ticker's full daily history, raising rather than logging on failure."""
    import yfinance  # imported here so the module is importable without a network stack

    # yfinance 0.2.66 assigns the integer 0 into a ``Dividends`` column that
    # pandas 3 has inferred as its string dtype when every value is missing
    # (scrapers/history.py:415), which raises TypeError for a ticker whose
    # actions carry no dividend -- eight of the 876 in the first W7-P1 pull.
    # Restoring object inference for the duration of the vendor's own parsing
    # is a compatibility shim on the call, not a change to what is cached: the
    # price columns are numeric either way and the action columns are cast
    # to float in _assemble_panel.
    with pd.option_context("future.infer_string", False):
        frame: pd.DataFrame = yfinance.Ticker(ticker).history(
            period=config.period,
            interval=config.interval,
            auto_adjust=config.auto_adjust,
            actions=True,
            raise_errors=True,
        )
    if frame.empty:
        return frame
    for column in config.action_columns:
        if column in frame.columns:
            frame[column] = numeric_actions(frame[column], label=f"{ticker}/{column}")
    return frame


def numeric_actions(values: pd.Series, *, label: str) -> pd.Series:
    """An action column as float, accepting Yahoo's occasional ``"0.01 USD"`` string.

    The second W7-P1 pull found a dividend delivered as text with a currency
    token; the amount is in the same units as the numeric ones, so the first
    token is taken. Anything that does not parse raises, so the ticker is
    recorded ``failed`` by name (ruling 6) rather than the panel being lost.
    """
    out: list[float] = []
    for value in values.tolist():
        if value is None or (isinstance(value, float) and value != value):
            out.append(float("nan"))
            continue
        if isinstance(value, str):
            token = value.strip().split()
            try:
                out.append(float(token[0]))
            except (IndexError, ValueError) as exc:
                raise LoaderError(f"{label}: unparseable action value {value!r}") from exc
            continue
        try:
            out.append(float(value))
        except (TypeError, ValueError) as exc:
            raise LoaderError(f"{label}: unparseable action value {value!r}") from exc
    return pd.Series(out, index=values.index, dtype="float64", name=values.name)


def _assemble_panel(
    pulls: Sequence[TickerPull], config: EtfConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, str]]:
    """Long prices, long actions, the coverage table and the manifest status."""
    prices: list[pd.DataFrame] = []
    actions: list[pd.DataFrame] = []
    coverage: list[dict[str, object]] = []
    status: dict[str, str] = {}
    for pull in pulls:
        status[pull.ticker] = pull.status
        row: dict[str, object] = {
            "ticker": pull.ticker,
            "status": pull.status,
            "attempts": pull.attempts,
            "rows": 0,
            "first_date": pd.NaT,
            "last_date": pd.NaT,
            "error": pull.error,
        }
        if pull.status == OK and pull.frame is not None:
            frame = pull.frame
            contracts.check_columns(
                frame, config.required_response_columns, label=f"yfinance/{pull.ticker} response"
            )
            frame = frame.copy()
            frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
            frame.index.name = "date"
            bars = frame[list(config.price_columns)].copy()
            bars.insert(0, "ticker", pull.ticker)
            prices.append(bars)
            acts = frame[list(config.action_columns)].astype(float)
            acts = acts[(acts.fillna(0.0) != 0.0).any(axis=1)].copy()
            acts.insert(0, "ticker", pull.ticker)
            actions.append(acts)
            row.update(rows=len(bars), first_date=bars.index.min(), last_date=bars.index.max())
        coverage.append(row)
    if not prices:
        raise LoaderError("yfinance returned no bars for any S&P 500 ticker")
    long_prices = pd.concat(prices).sort_index(kind="mergesort")
    long_actions = pd.concat(actions).sort_index(kind="mergesort")
    table = pd.DataFrame(coverage).set_index("ticker")
    table["first_date"] = pd.to_datetime(table["first_date"])
    table["last_date"] = pd.to_datetime(table["last_date"])
    return long_prices, long_actions, table, status


def fetch_sp500_bars(*, interval: str | None = None) -> tuple[cache.CacheResult, ...]:
    """Cache RAW daily OHLCV, actions and coverage for every ticker the tables name.

    Three long-format artefacts -- prices, actions (only rows carrying a
    dividend or split) and a coverage table with one row per requested ticker
    carrying its status, attempt count and date range -- so that a departed
    name Yahoo has no history for is recorded as ``empty`` and a request that
    failed as ``failed``, never as one absence (SPEC.md 15.3.1 rulings 5 and
    6). The per-ticker status is also written on the coverage entry in
    ``data/manifest.json``. Same contract as :func:`fetch_etf` otherwise:
    ``auto_adjust=False``, ``Adj Close`` asserted on the response and then
    discarded, daily bars only (CLAUDE.md invariant 2).
    """
    config = load().model.data.etf
    contracts.require_daily_interval(interval if interval is not None else config.interval)
    tickers = sp500_requested_tickers()
    pulled: dict[str, object] = {}

    def pull() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, str]]:
        if "panel" not in pulled:
            pulls = pull_histories(
                tickers,
                history=functools.partial(_yfinance_history, config=config),
                answered_no_data=yahoo_answered_no_data,
            )
            counts = check_status_identity(tickers, pulls)
            print(
                f"sp500 bars: requested {counts['requested']} = ok {counts[OK]} + empty "
                f"{counts[EMPTY]} + failed {counts[FAILED]}"
            )
            pulled["panel"] = _assemble_panel(pulls, config)
        panel = pulled["panel"]
        assert isinstance(panel, tuple)
        return panel

    def fetch_prices() -> cache.Fetched:
        return cache.Fetched(frame=pull()[0])

    def fetch_actions() -> cache.Fetched:
        return cache.Fetched(frame=pull()[1])

    def fetch_coverage() -> cache.Fetched:
        _, _, table, status = pull()
        return cache.Fetched(frame=table, status=status)

    url = (
        f"{_YAHOO_CHART_URL}<ticker> for the {len(tickers)} tickers the cached Wikipedia "
        "tables name"
    )
    prices_result = cache.fetch_raw(
        source="yfinance", name=SP500_PRICES_NAME, url=url, fetcher=fetch_prices
    )
    actions_result = cache.fetch_raw(
        source="yfinance", name=SP500_ACTIONS_NAME, url=url, fetcher=fetch_actions
    )
    coverage_result = cache.fetch_raw(
        source="yfinance",
        name=SP500_COVERAGE_NAME,
        url=url,
        fetcher=fetch_coverage,
        date_column="last_date",
    )
    return prices_result, actions_result, coverage_result


# ---------------------------------------------------------------------------
# SEC EDGAR -- point-in-time shares outstanding (SPEC.md 15.4.1, W7-P2a)
# ---------------------------------------------------------------------------

#: Seconds between EDGAR requests. SEC's fair-access policy caps a client at
#: ten requests a second; this runs well under it, one request at a time. A
#: transport setting, not a parameter (constants held in code, W7-P2a).
_EDGAR_PAUSE: Final[float] = 0.15

#: Pause before the single retry of a `failed` quarter -- long enough for a
#: burst limit to clear. Same status contract as the S&P 500 bars (ruling 6):
#: `ok`, `empty` (the server answered 404: no frame or index for that
#: quarter) or `failed` (anything else), retried once.
_EDGAR_RETRY_PAUSE: Final[float] = 10.0


def _edgar_get(url: str, *, user_agent: str) -> bytes | None:
    """One EDGAR request. ``None`` on 404 -- the server ANSWERED: nothing there."""
    request = urllib.request.Request(
        url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            payload: bytes = response.read()
            if response.headers.get("Content-Encoding") == "gzip":
                import gzip

                payload = gzip.decompress(payload)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    if not payload:
        raise LoaderError(f"{url} returned an empty body")
    return payload


def _edgar_quarters(
    labels: Sequence[str],
    urls: Sequence[str],
    *,
    parse: Callable[[bytes, int], pd.DataFrame],
    user_agent: str,
    get: Callable[[str], bytes | None] | None = None,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """Fetch every item once, retry the failed ones once, classify each.

    Written for the quarterly frames and index; the per-CIK submissions
    records and per-filing headers of SPEC.md 15.6.1 have the same shape (a
    label, a URL, a parser) and reuse it unchanged.
    Returns the concatenated table, ``label -> status`` and ``label -> error``.
    Pure orchestration over an injected ``get`` so the retry and the status
    identity are testable without a socket.
    """
    fetch = get or functools.partial(_edgar_get, user_agent=user_agent)
    frames: dict[str, pd.DataFrame] = {}
    status: dict[str, str] = {}
    errors: dict[str, str] = {}
    pending = list(range(len(labels)))
    for attempt in range(1 + _YFINANCE_RETRIES):
        if not pending:
            break
        if attempt:
            time.sleep(_EDGAR_RETRY_PAUSE)
        still: list[int] = []
        for i in pending:
            label = labels[i]
            try:
                payload = fetch(urls[i])
                if payload is None:
                    status[label] = EMPTY
                    errors[label] = "HTTP 404: the server has no such quarter"
                else:
                    frames[label] = parse(payload, i)
                    status[label] = OK
                    errors.pop(label, None)
            except Exception as exc:  # classified as failed, never swallowed
                status[label] = FAILED
                errors[label] = f"{type(exc).__name__}: {exc}"
                still.append(i)
            time.sleep(_EDGAR_PAUSE)
        pending = still
    counts: dict[str, int] = {OK: 0, EMPTY: 0, FAILED: 0}
    for value in status.values():
        counts[value] += 1
    if sum(counts.values()) != len(labels) or set(status) != set(labels):
        raise LoaderError("EDGAR status identity violated: a quarter has no status")
    table = (
        pd.concat([frames[label] for label in labels if label in frames]).sort_index(
            kind="mergesort"
        )
        if frames
        else pd.DataFrame()
    )
    return table, status, errors


def fetch_edgar_shares(
    *, today: date | None = None, get: Callable[[str], bytes | None] | None = None
) -> tuple[cache.CacheResult, cache.CacheResult, cache.CacheResult]:
    """Cache the frames, the quarterly XBRL index and SEC's ticker map. SPEC.md 15.4.1.

    Three raw artefacts under source ``edgar``. The frames and the index carry
    a per-quarter status on their manifest entry -- ``ok`` / ``empty`` /
    ``failed`` -- with one retry of ``failed``; a quarter still ``failed``
    is listed by name by the report and never read as "no filers". The
    User-Agent is read from the environment variable the config names and is
    required BEFORE any request is made.
    """
    config = load().model.equity_shares
    user_agent = secrets.require(
        config.user_agent_env,
        hint=(
            "SEC EDGAR's fair-access policy requires a User-Agent naming the requester and a "
            "contact address, e.g. 'Name contact@example.com'; see .env.example."
        ),
    )
    quarters = edgar.periods(config.first_period, today or date.today())
    labels = [edgar.period_label(y, q) for y, q in quarters]
    frame_urls = [config.frame_url(edgar.frame_period(y, q)) for y, q in quarters]
    index_urls = [config.quarter_index_url(y, q) for y, q in quarters]

    def parse_frame(payload: bytes, i: int) -> pd.DataFrame:
        year, quarter = quarters[i]
        return edgar.parse_frame(
            payload,
            period=edgar.frame_period(year, quarter),
            taxonomy=config.taxonomy,
            tag=config.tag,
            unit=config.unit,
        )

    def parse_index(payload: bytes, i: int) -> pd.DataFrame:
        year, quarter = quarters[i]
        return edgar.parse_xbrl_index(payload, year=year, quarter=quarter)

    pulled: dict[str, object] = {}

    def frames() -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
        if "frames" not in pulled:
            pulled["frames"] = _edgar_quarters(
                labels, frame_urls, parse=parse_frame, user_agent=user_agent, get=get
            )
        out = pulled["frames"]
        assert isinstance(out, tuple)
        return out

    def index() -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
        if "index" not in pulled:
            pulled["index"] = _edgar_quarters(
                labels, index_urls, parse=parse_index, user_agent=user_agent, get=get
            )
        out = pulled["index"]
        assert isinstance(out, tuple)
        return out

    def report(kind: str, status: Mapping[str, str], errors: Mapping[str, str]) -> None:
        counts = {s: sum(1 for v in status.values() if v == s) for s in (OK, EMPTY, FAILED)}
        print(
            f"edgar {kind}: quarters {len(status)} = ok {counts[OK]} + empty {counts[EMPTY]} + "
            f"failed {counts[FAILED]}"
        )
        for label, err in sorted(errors.items()):
            if status[label] == FAILED:
                print(f"  failed {label}: {err}")

    def fetch_frames() -> cache.Fetched:
        table, status, errors = frames()
        report("frames", status, errors)
        if table.empty:
            raise LoaderError("EDGAR returned no frame records for any quarter")
        return cache.Fetched(frame=table, status=status)

    def fetch_index() -> cache.Fetched:
        table, status, errors = index()
        report("xbrl index", status, errors)
        if table.empty:
            raise LoaderError("EDGAR returned no XBRL index rows for any quarter")
        return cache.Fetched(frame=table, status=status)

    def fetch_tickers() -> cache.Fetched:
        fetch = get or functools.partial(_edgar_get, user_agent=user_agent)
        payload = fetch(config.company_tickers_url)
        if payload is None:
            raise LoaderError(f"{config.company_tickers_url} answered 404")
        return cache.Fetched(frame=edgar.parse_company_tickers(payload), source_bytes=payload)

    frames_result = cache.fetch_raw(
        source=edgar.SOURCE,
        name=edgar.FRAMES_NAME,
        url=config.frame_url("CY<year>Q<q>I") + f" for {labels[0]}..{labels[-1]}",
        fetcher=fetch_frames,
    )
    index_result = cache.fetch_raw(
        source=edgar.SOURCE,
        name=edgar.INDEX_NAME,
        url=config.quarter_index_url(0, 0).replace("0/QTR0", "<year>/QTR<q>")
        + f" for {labels[0]}..{labels[-1]}",
        fetcher=fetch_index,
    )
    tickers_result = cache.fetch_raw(
        source=edgar.SOURCE,
        name=edgar.COMPANY_TICKERS_NAME,
        url=config.company_tickers_url,
        fetcher=fetch_tickers,
    )
    return frames_result, index_result, tickers_result


# ---------------------------------------------------------------------------
# Ken French Data Library
# ---------------------------------------------------------------------------


def fetch_ken_french() -> cache.CacheResult:
    """Cache the daily research-factor table, including the risk-free leg.

    Through ``pandas_datareader.famafrench``, which knows where each embedded
    table starts and ends inside the zipped multi-table CSV. Values are cached in
    the publisher's units -- percent per trading day -- so that the cache can be
    checked against Ken French's own file; :func:`mafrm.data.french.as_annual_percent`
    does the conversion at read time.
    """
    from pandas_datareader import famafrench  # local: keeps the import off the fast path

    config = load().model.data.risk_free.ken_french_daily_rf

    def fetch() -> cache.Fetched:
        reader = famafrench.FamaFrenchReader(
            config.dataset, start=config.history_start, end=date.today()
        )
        try:
            tables = reader.read()
        finally:
            reader.close()
        if 0 not in tables:
            raise LoaderError(
                f"{config.dataset}: pandas_datareader returned no table 0; "
                f"got keys {sorted(k for k in tables if k != 'DESCR')}"
            )
        return cache.Fetched(
            frame=french.normalise_daily_factors(tables[0], rf_column=config.rf_column)
        )

    return cache.fetch_raw(
        source=french.SOURCE, name=french.CACHE_NAME, url=config.url, fetcher=fetch
    )


def fetch_ken_french_siccodes() -> cache.CacheResult:
    """Cache Ken French's Fama-French 49 industry definitions. SPEC.md 15.6.1 ruling 1.

    The zip's one text file, parsed by :func:`mafrm.data.sic.parse_siccodes`
    into one row per SIC range. Served with the browser-ish agent the Data
    Library's CDN requires, as the daily-factor pull is through
    ``pandas_datareader``.
    """
    import io
    import zipfile

    config = load().model.data.ken_french_siccodes49

    def fetch() -> cache.Fetched:
        payload = _download(config.url, user_agent=_BROWSER_USER_AGENT)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if config.member not in archive.namelist():
                raise LoaderError(
                    f"{config.url}: no member {config.member!r}; got {archive.namelist()}"
                )
            text = archive.read(config.member)
        table = sic.parse_siccodes(
            text, industries=config.industries, unassigned=config.unassigned_industry
        )
        return cache.Fetched(frame=table, source_bytes=payload)

    return cache.fetch_raw(
        source=french.SOURCE, name=sic.SICCODES_NAME, url=config.url, fetcher=fetch
    )


def _sic_targets(cfg: object | None = None) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """The CIKs to look up and, per ticker, the CIKs to search for its first 10-K.

    Every CIK the W7-P2a mapping assigns to any ticker the membership tables
    name, plus each override row's predecessor -- not only the estimation
    universe's names, so the SIC table's provenance does not depend on a
    screen parameter (SPEC.md 15.6.1, constructions).
    """
    panel = market_cap.build(load() if cfg is None else cfg)  # type: ignore[arg-type]
    mapping = panel.mapping
    per_ticker: dict[str, list[int]] = {}
    for ticker, row in mapping.iterrows():
        if pd.isna(row["cik"]):
            continue
        ciks = [int(row["cik"])]
        predecessor = row.get("predecessor_cik")
        if predecessor is not None and not pd.isna(predecessor):
            ciks.append(int(predecessor))
        per_ticker[str(ticker)] = ciks
    return mapping, per_ticker


def fetch_edgar_sic(
    *, get: Callable[[str], bytes | None] | None = None
) -> tuple[cache.CacheResult, cache.CacheResult]:
    """Cache the current SIC per CIK and the first in-sample 10-K's header SIC per ticker.

    SPEC.md 15.6.1 ruling 2. Three passes, each under the ok / empty / failed
    contract with one retry: the submissions record of every mapped CIK; the
    older filing pages whose span reaches ``sample.start``; then one SGML
    header per ticker, for the earliest ``first_filing_forms`` filing on or
    after ``sample.start`` across the ticker's CIK and its predecessor. A
    ticker with no such filing is ``no_10k``; a header without an
    ``ASSIGNED-SIC`` is ``no_sic_in_header``. Nothing is guessed.
    """
    cfg = load()
    industries_cfg = cfg.model.equity_industries
    shares_cfg = cfg.model.equity_shares
    user_agent = secrets.require(
        shares_cfg.user_agent_env,
        hint=(
            "SEC EDGAR's fair-access policy requires a User-Agent naming the requester and a "
            "contact address, e.g. 'Name contact@example.com'; see .env.example."
        ),
    )
    start = cfg.model.sample.start
    _, per_ticker = _sic_targets(cfg)
    ciks = sorted({c for cs in per_ticker.values() for c in cs})
    labels = [str(c) for c in ciks]
    urls = [industries_cfg.submissions_url_for(sic.submissions_name(c)) for c in ciks]
    records: dict[int, sic.Submissions] = {}

    def parse_record(payload: bytes, i: int) -> pd.DataFrame:
        record = sic.parse_submissions(payload)
        records[ciks[i]] = record
        return pd.DataFrame(
            {
                "cik": [record.cik],
                "sic": pd.array([record.sic], dtype="Int64"),
                "sic_description": [record.sic_description],
                "name": [record.name],
            }
        )

    table, status, errors = _edgar_quarters(
        labels, urls, parse=parse_record, user_agent=user_agent, get=get
    )
    _report_statuses("submissions", status, errors)
    if table.empty:
        raise LoaderError("EDGAR returned no submissions record for any CIK")
    current = table.set_index("cik")
    current["status"] = [
        status[str(c)] if pd.notna(current.loc[c, "sic"]) else sic.NO_SIC for c in current.index
    ]
    for c in ciks:
        if c not in current.index:
            current.loc[c] = {
                "sic": pd.NA,
                "sic_description": "",
                "name": "",
                "status": status[str(c)],
            }
    current = current.sort_index()

    # Older pages whose span reaches sample.start, so the first in-sample 10-K
    # is chosen over a name's whole history rather than its newest 1,000 filings.
    page_labels: list[str] = []
    page_urls: list[str] = []
    page_owner: list[int] = []
    for c, record in records.items():
        for name in sic.older_pages_needed(record, start=start):
            page_labels.append(f"{c}:{name}")
            page_urls.append(industries_cfg.submissions_url_for(name))
            page_owner.append(c)
    pages: dict[int, list[pd.DataFrame]] = {}

    def parse_page(payload: bytes, i: int) -> pd.DataFrame:
        frame = sic.parse_filings_page(payload)
        pages.setdefault(page_owner[i], []).append(frame)
        return frame.assign(cik=page_owner[i])

    if page_labels:
        _, page_status, page_errors = _edgar_quarters(
            page_labels, page_urls, parse=parse_page, user_agent=user_agent, get=get
        )
        _report_statuses("older filing pages", page_status, page_errors)
    else:
        page_status = {}

    # The first in-sample 10-K per ticker, searched across its CIKs.
    chosen: dict[str, tuple[int, pd.Series]] = {}
    first_status: dict[str, str] = {}
    for ticker, candidates in per_ticker.items():
        best: tuple[int, pd.Series] | None = None
        incomplete = False
        for c in candidates:
            rec = records.get(c)
            if rec is None:
                incomplete = True
                continue
            frames = [rec.filings, *pages.get(c, [])]
            needed = sic.older_pages_needed(rec, start=start)
            if any(page_status.get(f"{c}:{n}") == FAILED for n in needed):
                incomplete = True
            filings = pd.concat(frames, ignore_index=True)
            hit = sic.first_filing(filings, forms=industries_cfg.first_filing_forms, start=start)
            if hit is not None and (best is None or hit["filing_date"] < best[1]["filing_date"]):
                best = (c, hit)
        if best is None:
            first_status[ticker] = FAILED if incomplete else sic.NO_10K
        else:
            chosen[ticker] = best

    header_tickers = sorted(chosen)
    header_urls = [
        industries_cfg.header_url_for(chosen[t][0], str(chosen[t][1]["accession"]))
        for t in header_tickers
    ]
    headers: dict[str, sic.Header] = {}

    def parse_hdr(payload: bytes, i: int) -> pd.DataFrame:
        ticker = header_tickers[i]
        header = sic.parse_header(payload, cik=chosen[ticker][0])
        headers[ticker] = header
        return pd.DataFrame({"ticker": [ticker]})

    if header_tickers:
        _, hdr_status, hdr_errors = _edgar_quarters(
            header_tickers, header_urls, parse=parse_hdr, user_agent=user_agent, get=get
        )
        _report_statuses("filing headers", hdr_status, hdr_errors)
    else:
        hdr_status = {}

    rows: list[dict[str, object]] = []
    for ticker in sorted(per_ticker):
        if ticker in chosen:
            c, hit = chosen[ticker]
            header = headers.get(ticker)
            if header is None:
                state = hdr_status.get(ticker, FAILED)
            elif header.sic is None:
                state = sic.NO_SIC_IN_HEADER
            else:
                state = OK
            rows.append(
                {
                    "ticker": ticker,
                    "cik": c,
                    "accession": str(hit["accession"]),
                    "form": str(hit["form"]),
                    "filing_date": pd.Timestamp(hit["filing_date"]),
                    "period": pd.Timestamp(header.period) if header and header.period else pd.NaT,
                    "sic": header.sic if header is not None else None,
                    "status": state,
                }
            )
        else:
            rows.append(
                {
                    "ticker": ticker,
                    "cik": per_ticker[ticker][0],
                    "accession": "",
                    "form": "",
                    "filing_date": pd.NaT,
                    "period": pd.NaT,
                    "sic": None,
                    "status": first_status[ticker],
                }
            )
    first_10k = pd.DataFrame(rows).set_index("ticker")
    first_10k["sic"] = first_10k["sic"].astype("Int64")
    counts = first_10k["status"].value_counts().to_dict()
    print(f"edgar first-10-K headers: tickers {len(first_10k)} = {counts}")

    current_result = cache.fetch_raw(
        source=edgar.SOURCE,
        name=sic.CURRENT_NAME,
        url=industries_cfg.submissions_url_for("CIK<cik>.json") + f" for {len(ciks)} CIKs",
        fetcher=lambda: cache.Fetched(frame=current.reset_index(), status=status),
    )
    first_result = cache.fetch_raw(
        source=edgar.SOURCE,
        name=sic.FIRST_10K_NAME,
        url=industries_cfg.header_url_for(0, "<accession>").replace("/0/", "/<cik>/")
        + f" for {len(header_tickers)} first in-sample 10-Ks",
        fetcher=lambda: cache.Fetched(
            frame=first_10k.reset_index(),
            status={str(t): str(v) for t, v in first_10k["status"].items()},
        ),
    )
    return current_result, first_result


def fetch_edgar_sic_history(
    *, get: Callable[[str], bytes | None] | None = None
) -> cache.CacheResult:
    """Cache every in-sample 10-K header of the names whose industry drifted. SPEC.md 15.6.3.

    W7-P3b's targeted pull: the drifted names are computed from the two cached
    SIC tables (the first in-sample 10-K's industry differs from the current
    one), their submissions records and older pages are re-fetched, and one
    SGML header is pulled per in-sample ``first_filing_forms`` filing across
    the name's CIK and its predecessor. Rows are indexed by FILING DATE so
    ``cache.read`` truncates them at the holdout boundary.
    """
    cfg = load()
    industries_cfg = cfg.model.equity_industries
    shares_cfg = cfg.model.equity_shares
    user_agent = secrets.require(
        shares_cfg.user_agent_env,
        hint="SEC EDGAR's fair-access policy requires a User-Agent; see .env.example.",
    )
    start = cfg.model.sample.start
    unassigned = cfg.model.data.ken_french_siccodes49.unassigned_industry
    mapping, per_ticker = _sic_targets(cfg)
    siccodes = sic.load_siccodes()
    industries = sic.industries_for(mapping, sic.load_current(), siccodes, unassigned=unassigned)
    drift = sic.drift_table(industries, sic.load_first_10k(), siccodes, unassigned=unassigned)
    targets = [t for t in sic.drifted_tickers(drift) if t in per_ticker]
    print(f"edgar sic history: {len(targets)} drifted names: {', '.join(targets)}")
    if not targets:
        raise LoaderError("no drifted names: nothing to pull (row 284 found none?)")

    ciks = sorted({c for t in targets for c in per_ticker[t]})
    labels = [str(c) for c in ciks]
    urls = [industries_cfg.submissions_url_for(sic.submissions_name(c)) for c in ciks]
    records: dict[int, sic.Submissions] = {}

    def parse_record(payload: bytes, i: int) -> pd.DataFrame:
        records[ciks[i]] = sic.parse_submissions(payload)
        return pd.DataFrame({"cik": [ciks[i]]})

    _, status, errors = _edgar_quarters(
        labels, urls, parse=parse_record, user_agent=user_agent, get=get
    )
    _report_statuses("submissions (drifted)", status, errors)

    page_labels: list[str] = []
    page_urls: list[str] = []
    page_owner: list[int] = []
    for c, record in records.items():
        for name in sic.older_pages_needed(record, start=start):
            page_labels.append(f"{c}:{name}")
            page_urls.append(industries_cfg.submissions_url_for(name))
            page_owner.append(c)
    pages: dict[int, list[pd.DataFrame]] = {}

    def parse_page(payload: bytes, i: int) -> pd.DataFrame:
        frame = sic.parse_filings_page(payload)
        pages.setdefault(page_owner[i], []).append(frame)
        return frame.assign(cik=page_owner[i])

    if page_labels:
        _, page_status, page_errors = _edgar_quarters(
            page_labels, page_urls, parse=parse_page, user_agent=user_agent, get=get
        )
        _report_statuses("older filing pages (drifted)", page_status, page_errors)

    # Every in-sample 10-K per ticker, across its CIKs.
    filings: list[tuple[str, int, str, pd.Timestamp, str]] = []
    for ticker in targets:
        for c in per_ticker[ticker]:
            rec = records.get(c)
            if rec is None:
                continue
            table = pd.concat([rec.filings, *pages.get(c, [])], ignore_index=True)
            hits = table[
                table["form"].isin(list(industries_cfg.first_filing_forms))
                & (table["filing_date"] >= pd.Timestamp(start))
            ]
            for acc, filed_on, form in zip(
                hits["accession"].tolist(),
                hits["filing_date"].tolist(),
                hits["form"].tolist(),
                strict=True,
            ):
                filings.append((ticker, c, str(acc), pd.Timestamp(filed_on), str(form)))
    filings.sort(key=lambda x: (x[0], x[3], x[2]))
    hdr_labels = [f"{t}:{a}" for t, _c, a, _d, _f in filings]
    hdr_urls = [industries_cfg.header_url_for(c, a) for _t, c, a, _d, _f in filings]
    headers: dict[str, sic.Header] = {}

    def parse_hdr(payload: bytes, i: int) -> pd.DataFrame:
        _t, c, a, _d, _f = filings[i]
        headers[hdr_labels[i]] = sic.parse_header(payload, cik=c)
        return pd.DataFrame({"accession": [a]})

    _, hdr_status, hdr_errors = _edgar_quarters(
        hdr_labels, hdr_urls, parse=parse_hdr, user_agent=user_agent, get=get
    )
    _report_statuses("filing headers (drifted)", hdr_status, hdr_errors)

    rows: list[dict[str, object]] = []
    for label, (ticker, c, accession, filed, form) in zip(hdr_labels, filings, strict=True):
        header = headers.get(label)
        if header is None:
            state = hdr_status.get(label, FAILED)
        elif header.sic is None:
            state = sic.NO_SIC_IN_HEADER
        else:
            state = OK
        rows.append(
            {
                "filing_date": filed,
                "ticker": ticker,
                "cik": c,
                "accession": accession,
                "form": form,
                "period": pd.Timestamp(header.period) if header and header.period else pd.NaT,
                "sic": header.sic if header is not None else None,
                "status": state,
            }
        )
    history = pd.DataFrame(rows).set_index("filing_date").sort_index(kind="mergesort")
    history["sic"] = history["sic"].astype("Int64")
    print(
        f"edgar sic history: filings {len(history)} = {history['status'].value_counts().to_dict()}"
    )
    return cache.fetch_raw(
        source=edgar.SOURCE,
        name=sic.HISTORY_NAME,
        url=industries_cfg.header_url_for(0, "<accession>").replace("/0/", "/<cik>/")
        + f" for every in-sample 10-K of {len(targets)} drifted names",
        fetcher=lambda: cache.Fetched(
            frame=history.reset_index(),
            status={
                str(k): str(v) for k, v in zip(hdr_labels, [r["status"] for r in rows], strict=True)
            },
        ),
        date_column="filing_date",
    )


def _report_statuses(kind: str, status: Mapping[str, str], errors: Mapping[str, str]) -> None:
    counts = {s: sum(1 for v in status.values() if v == s) for s in (OK, EMPTY, FAILED)}
    print(
        f"edgar {kind}: items {len(status)} = ok {counts[OK]} + empty {counts[EMPTY]} + "
        f"failed {counts[FAILED]}"
    )
    for label, err in sorted(errors.items()):
        if status[label] == FAILED:
            print(f"  failed {label}: {err}")


# ---------------------------------------------------------------------------
# AQR research workbooks
# ---------------------------------------------------------------------------


def fetch_aqr(dataset: AqrDataset) -> cache.CacheResult:
    """Download and cache one AQR workbook. Never committed -- invariant 8."""
    config = load().model.data.aqr
    agent = _BROWSER_USER_AGENT if config.user_agent_is_browser else _USER_AGENT
    spec = aqr.WorkbookSpec(
        name=dataset.name,
        url=dataset.url,
        sheet=dataset.sheet,
        header_row=dataset.header_row,
        date_column=dataset.date_column,
        required_columns=dataset.required_columns,
        description=dataset.description,
    )

    def fetch() -> cache.Fetched:
        payload = _download(dataset.url, user_agent=agent)
        return cache.Fetched(frame=aqr.parse_workbook(payload, spec), source_bytes=payload)

    return cache.fetch_raw(source=aqr.SOURCE, name=dataset.name, url=dataset.url, fetcher=fetch)


# ---------------------------------------------------------------------------
# Stooq -- cross-check only
# ---------------------------------------------------------------------------


def fetch_stooq(ticker: str, *, interval: str | None = None) -> cache.CacheResult:
    """Cache Stooq daily bars for one ticker, as an independent second opinion.

    Same interval guard as :func:`fetch_etf`: Stooq will happily serve weekly
    bars, and a cross-check computed on a different bar size than the series it
    checks is not a cross-check.

    Raises :class:`mafrm.data.stooq.StooqUnavailable` when the site serves its
    JavaScript proof-of-work browser challenge instead of CSV, which is what it
    does as of 2026-08-29. That is an access control and this project does not
    defeat it.
    """
    config = load().model.data
    contracts.require_daily_interval(interval if interval is not None else config.etf.interval)

    symbol = stooq.symbol_for(ticker, template=config.stooq.symbol_template)
    url = config.stooq.url_template.format(symbol=symbol)

    def fetch() -> cache.Fetched:
        payload = _download(url, user_agent=_BROWSER_USER_AGENT)
        frame = stooq.parse_csv(payload.decode("utf-8", errors="replace"), symbol=symbol)
        return cache.Fetched(frame=frame, source_bytes=payload)

    return cache.fetch_raw(
        source=stooq.SOURCE, name=f"{ticker.lower()}_prices", url=url, fetcher=fetch
    )


# ---------------------------------------------------------------------------
# `make data`
# ---------------------------------------------------------------------------


def refresh_all(*, strict: bool = False) -> tuple[list[cache.CacheResult], list[str]]:
    """Everything the data layer knows how to fetch. ``make data``.

    Returns what was cached and a list of human-readable failures. A source
    being down must not stop the other eleven from refreshing, so each is
    attempted independently; pass ``strict=True`` to re-raise the first failure
    instead, which is what a contract test wants.
    """
    config = load()
    results: list[cache.CacheResult] = []
    failures: list[str] = []

    def attempt(label: str, work: Callable[[], Sequence[cache.CacheResult]]) -> None:
        try:
            results.extend(work())
        except Exception as exc:  # one bad source must not stop the other eleven
            if strict:
                raise
            failures.append(f"{label}: {type(exc).__name__}: {exc}")

    def one(
        fn: Callable[..., cache.CacheResult], *args: object
    ) -> Callable[[], list[cache.CacheResult]]:
        """Adapt a single-result fetcher to the sequence ``attempt`` expects."""
        return lambda: [fn(*args)]

    attempt("fed/nominal_curve", one(fetch_fed_curve, "nominal"))
    attempt("fed/real_curve", one(fetch_fed_curve, "real"))

    for series in config.model.data.fred.series:
        attempt(f"fred/{series.id}", functools.partial(fetch_fred_series, series))

    attempt("ken_french/daily_factors", one(fetch_ken_french))

    # Universe ETFs, the TLT cross-check, and the tradable proxies of SPEC.md 7's
    # five curve points (costs.tradable_proxies) -- deduplicated, since TLT is
    # both the cross-check and the 30y proxy.
    tickers: list[str] = []
    for ticker in (
        *config.universe.yfinance_tickers,
        config.model.data.tlt_cross_check.ticker,
        *(proxy.ticker for proxy in config.model.costs.tradable_proxies),
    ):
        if ticker not in tickers:
            tickers.append(ticker)
    for ticker in tickers:
        attempt(f"yfinance/{ticker}", functools.partial(fetch_etf, ticker))

    for dataset in config.model.data.aqr.datasets:
        attempt(f"aqr/{dataset.name}", one(fetch_aqr, dataset))

    # A source retired in data/manifest.json is skipped with its reason printed,
    # never silently: the retirement is a note in the provenance record (W7-P1b,
    # Stooq -- unreachable since W1-P5, read by nothing, and the refresh
    # target's non-zero-on-any-failure contract is right and stays).
    retired = cache.Manifest.load().retired_sources
    if stooq.SOURCE in retired:
        print(f"retired   {stooq.SOURCE}: {retired[stooq.SOURCE]}")
    else:
        for ticker in config.model.data.stooq.cross_check_tickers:
            attempt(f"stooq/{ticker}", one(fetch_stooq, ticker))

    # SPEC.md 15.3: the two Wikipedia membership tables, then the bars for
    # every ticker they name. The bars pull reads the cached tables, so it
    # follows them and is skipped with a recorded failure if they did not land.
    attempt("wikipedia/sp500", lambda: list(fetch_sp500_wikipedia()))
    attempt("yfinance/sp500_bars", lambda: list(fetch_sp500_bars()))
    # SPEC.md 15.4.1: point-in-time shares outstanding from EDGAR, one request
    # per quarter. Needs SEC_EDGAR_USER_AGENT (config: equity_shares.user_agent_env)
    # and is recorded as a failure, by name, when it is unset.
    attempt("edgar/shares", lambda: list(fetch_edgar_shares()))
    # SPEC.md 15.6.1: Ken French's FF49 definitions, then the current SIC per
    # mapped CIK and the first in-sample 10-K's header SIC per ticker. The SIC
    # pull reads the cached frames and bars through the cap panel's mapping,
    # so it follows the EDGAR shares pull.
    attempt("ken_french/siccodes49", one(fetch_ken_french_siccodes))
    attempt("edgar/sic", lambda: list(fetch_edgar_sic()))
    # SPEC.md 15.6.3 (W7-P3b): every in-sample 10-K header of the names whose
    # industry drifted, so their exposure is point-in-time. Reads the two SIC
    # tables above, so it follows them.
    attempt("edgar/sic_history", one(fetch_edgar_sic_history))

    return results, failures


def _main(argv: Sequence[str]) -> int:
    if list(argv) not in ([], ["refresh"]):
        print("usage: python -m mafrm.data.loaders [refresh]", file=sys.stderr)
        return 2
    results, failures = refresh_all()
    for result in results:
        entry = result.entry
        print(
            f"{result.action:<10} {entry.path:<52} {entry.rows:>7,} rows  "
            f"{entry.first_date}..{entry.last_date}"
        )
    print(f"manifest: {cache.manifest_path()}")
    if failures:
        print(f"\n{len(failures)} source(s) did not refresh:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
