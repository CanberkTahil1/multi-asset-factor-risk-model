"""The loader layer, tested without a socket.

Nothing here fetches. ``tests/conftest.py`` fails any unmarked test that opens
one, which is what makes CLAUDE.md invariant 1 mechanical: these tests exercise
the guards and the URL/window arithmetic that run *before* a request is made.

The interval tests are the W1-P4 acceptance criterion -- requesting ``1wk`` or
``1mo`` from any loader must raise.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
import yaml

from mafrm.config import ConfigError, _parse_data, load
from mafrm.data import contracts, loaders

FORBIDDEN = ["1wk", "1mo"]


# ---------------------------------------------------------------------------
# CLAUDE.md invariant 2 -- the acceptance criterion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interval", FORBIDDEN)
def test_fetch_etf_refuses_weekly_and_monthly_bars(interval: str) -> None:
    """Raises before yfinance is called, so no wrong number can be cached."""
    with pytest.raises(contracts.IntervalError):
        loaders.fetch_etf("SPY", interval=interval)


@pytest.mark.parametrize("interval", FORBIDDEN)
def test_fetch_stooq_refuses_weekly_and_monthly_bars(interval: str) -> None:
    with pytest.raises(contracts.IntervalError):
        loaders.fetch_stooq("SPY", interval=interval)


@pytest.mark.parametrize("interval", FORBIDDEN)
def test_fetch_sp500_bars_refuses_weekly_and_monthly_bars(interval: str) -> None:
    """W7-P1: the S&P 500 bars pull sits behind the same guard as fetch_etf."""
    with pytest.raises(contracts.IntervalError):
        loaders.fetch_sp500_bars(interval=interval)


# ---------------------------------------------------------------------------
# W7-P1, SPEC.md 15.3.1 ruling 6: ok / empty / failed, and exactly one retry
# ---------------------------------------------------------------------------


class YFTzMissingError(Exception):
    """Same class NAME as yfinance's; classification matches on the name."""


class YFPricesMissingError(Exception):
    """Same class NAME as yfinance's."""


def _bars(rows: int) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=rows, freq="B")
    return pd.DataFrame({"Close": 1.0, "Volume": 10}, index=index)


def test_classify_history_separates_a_provider_answer_from_a_transport_failure() -> None:
    answered = {"DEAD": True, "LIVE": False}

    def probe(ticker: str) -> bool:
        return answered[ticker]

    assert loaders.classify_history("X", _bars(3), answered_no_data=probe) == ("ok", None)
    status, detail = loaders.classify_history(
        "X", YFPricesMissingError("no price data found"), answered_no_data=probe
    )
    assert status == "empty" and detail is not None and "YFPricesMissingError" in detail
    # The timezone-missing error is what yfinance raises for BOTH a delisting
    # and a swallowed transport error, so the probe decides.
    assert (
        loaders.classify_history("DEAD", YFTzMissingError("tz"), answered_no_data=probe)[0]
        == "empty"
    )
    assert (
        loaders.classify_history("LIVE", YFTzMissingError("tz"), answered_no_data=probe)[0]
        == "failed"
    )
    assert loaders.classify_history("DEAD", _bars(0), answered_no_data=probe)[0] == "empty"
    assert loaders.classify_history("LIVE", _bars(0), answered_no_data=probe)[0] == "failed"
    status, detail = loaders.classify_history("X", TimeoutError("slow"), answered_no_data=probe)
    assert status == "failed" and detail == "TimeoutError: slow"


def test_pull_histories_retries_failed_exactly_once_and_keeps_the_verdict() -> None:
    calls: dict[str, int] = {}

    def history(ticker: str) -> pd.DataFrame:
        calls[ticker] = calls.get(ticker, 0) + 1
        if ticker == "FLAKY" and calls[ticker] == 1:
            raise ConnectionError("first attempt drops")
        if ticker == "DOWN":
            raise ConnectionError("always drops")
        if ticker == "GONE":
            raise YFPricesMissingError("no price data found")
        if ticker == "NOTZ":
            raise YFTzMissingError("no timezone found")
        return _bars(5)

    def probe(ticker: str) -> bool:
        return ticker == "NOTZ"

    pulls = loaders.pull_histories(
        ["OK", "FLAKY", "DOWN", "GONE", "NOTZ", "OK"],
        history=history,
        answered_no_data=probe,
        workers=2,
    )
    by = {p.ticker: p for p in pulls}
    assert [p.ticker for p in pulls] == ["OK", "FLAKY", "DOWN", "GONE", "NOTZ"]  # deduplicated
    assert (by["OK"].status, by["OK"].attempts) == ("ok", 1)
    assert (by["FLAKY"].status, by["FLAKY"].attempts) == ("ok", 2)
    assert (by["DOWN"].status, by["DOWN"].attempts) == ("failed", 2)
    assert (by["GONE"].status, by["GONE"].attempts) == ("empty", 1)
    assert (by["NOTZ"].status, by["NOTZ"].attempts) == ("empty", 1)
    assert calls == {"OK": 1, "FLAKY": 2, "DOWN": 2, "GONE": 1, "NOTZ": 1}
    assert by["FLAKY"].frame is not None and len(by["FLAKY"].frame) == 5
    assert by["DOWN"].frame is None and by["DOWN"].error == "ConnectionError: always drops"
    assert by["NOTZ"].error is not None and "chart endpoint answered" in by["NOTZ"].error


def test_status_identity_catches_a_ticker_dropped_without_a_status() -> None:
    """W7-P1b: requested = ok + empty + failed, asserted, not printed."""

    def pull(ticker: str, status: str) -> loaders.TickerPull:
        return loaders.TickerPull(ticker, status, 1, None, None)  # type: ignore[arg-type]

    full = [pull("A", "ok"), pull("B", "empty"), pull("C", "failed")]
    assert loaders.check_status_identity(["A", "B", "C", "A"], full) == {
        "requested": 3,
        "ok": 1,
        "empty": 1,
        "failed": 1,
    }
    with pytest.raises(loaders.LoaderError, match="missing \\['C'\\]"):
        loaders.check_status_identity(["A", "B", "C"], full[:2])
    with pytest.raises(loaders.LoaderError, match="duplicated 1"):
        loaders.check_status_identity(["A", "B", "C"], [*full, pull("A", "ok")])
    with pytest.raises(loaders.LoaderError, match="extra \\['Z'\\]"):
        loaders.check_status_identity(["A", "B"], [full[0], full[1], pull("Z", "ok")])


def test_numeric_actions_accepts_a_currency_token_and_refuses_garbage() -> None:
    """The second W7-P1 pull met a dividend delivered as the string '0.01 USD'."""
    values = pd.Series([0.0, "0.01 USD", None, float("nan"), 2], dtype="object")
    out = loaders.numeric_actions(values, label="T/Dividends")
    assert out.dtype == "float64"
    assert out.iloc[0] == 0.0 and out.iloc[1] == 0.01 and out.iloc[4] == 2.0
    assert pd.isna(out.iloc[2]) and pd.isna(out.iloc[3])
    with pytest.raises(loaders.LoaderError, match="unparseable action value"):
        loaders.numeric_actions(pd.Series(["n/a"], dtype="object"), label="T/Dividends")


@pytest.mark.parametrize("interval", FORBIDDEN)
def test_the_config_itself_refuses_a_non_daily_interval(interval: str) -> None:
    """The other door into the same mistake: writing it into model.yaml.

    ``fetch_etf`` defaults its interval from the config, so a guard only at the
    call site would be bypassed by editing the YAML.
    """
    document = _model_document()
    document["data"]["etf"]["interval"] = interval
    with pytest.raises(ConfigError, match="invariant 2"):
        _parse_data(document)


def test_the_config_refuses_auto_adjust() -> None:
    """SPEC.md 3.5: adjusted closes are rewritten backwards, so we adjust ourselves."""
    document = _model_document()
    document["data"]["etf"]["auto_adjust"] = True
    with pytest.raises(ConfigError, match="auto_adjust must be false"):
        _parse_data(document)


def _model_document() -> dict[str, object]:
    from mafrm.config import config_dir

    parsed = yaml.safe_load((config_dir() / "model.yaml").read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def test_the_shipped_config_requests_daily_bars() -> None:
    assert load().model.data.etf.interval == contracts.DAILY_INTERVAL
    assert load().model.data.etf.auto_adjust is False


# ---------------------------------------------------------------------------
# The API key must never reach the manifest
# ---------------------------------------------------------------------------


def test_the_api_key_is_redacted_from_a_recorded_url() -> None:
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        "?series_id=DGS1MO&api_key=abcdef0123456789abcdef0123456789&file_type=json"
    )
    redacted = loaders._redact(url)
    assert "abcdef0123456789" not in redacted
    assert "api_key=REDACTED" in redacted
    assert "series_id=DGS1MO" in redacted


def test_redaction_leaves_a_url_without_a_key_alone() -> None:
    url = "https://example.invalid/a.csv?x=1"
    assert loaders._redact(url) == url


def test_no_manifest_entry_carries_a_key() -> None:
    """The manifest is committed, so a leaked key would be a leaked secret."""
    from mafrm.data import cache

    for entry in cache.Manifest.load():
        assert "api_key=" not in entry.url or "api_key=REDACTED" in entry.url


# ---------------------------------------------------------------------------
# Vintage windows -- the arithmetic behind the chunked ALFRED fetch
# ---------------------------------------------------------------------------


def test_a_short_vintage_list_needs_one_window() -> None:
    dates = [date(2019, 2, 4), date(2019, 2, 11), date(2019, 2, 18)]
    windows = loaders._vintage_windows(dates, chunk_size=10)
    assert windows == [("1776-07-04", "9999-12-31")]


def test_windows_are_split_at_the_chunk_boundary_and_do_not_overlap() -> None:
    """Hand-computed: five vintages at chunk size 2 gives three windows.

    Chunks are [d0,d1], [d2,d3], [d4]. Each window ends the day *before* the next
    chunk's first vintage, so no vintage is counted twice and none is skipped.
    """
    dates = [
        date(2020, 1, 6),
        date(2020, 1, 7),
        date(2020, 1, 8),
        date(2020, 1, 9),
        date(2020, 1, 10),
    ]
    windows = loaders._vintage_windows(dates, chunk_size=2)
    assert windows == [
        ("1776-07-04", "2020-01-07"),
        ("2020-01-08", "2020-01-09"),
        ("2020-01-10", "9999-12-31"),
    ]


def test_the_first_window_reaches_back_before_the_first_vintage() -> None:
    """So observations predating ALFRED's adoption still get their first release."""
    windows = loaders._vintage_windows([date(2019, 2, 4), date(2019, 2, 11)], chunk_size=1)
    assert windows[0][0] == "1776-07-04"


def test_the_last_window_runs_to_the_sentinel() -> None:
    windows = loaders._vintage_windows([date(2019, 2, 4), date(2019, 2, 11)], chunk_size=1)
    assert windows[-1][1] == "9999-12-31"


def test_no_vintages_means_no_windows() -> None:
    assert loaders._vintage_windows([], chunk_size=100) == []


def test_every_vintage_falls_inside_exactly_one_window() -> None:
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(11)]
    windows = loaders._vintage_windows(dates, chunk_size=3)
    for stamp in dates:
        containing = [
            (lo, hi)
            for lo, hi in windows
            if date.fromisoformat(lo) <= stamp
            and (hi == "9999-12-31" or stamp <= date.fromisoformat(hi))
        ]
        assert len(containing) == 1, f"{stamp} landed in {len(containing)} windows"


# ---------------------------------------------------------------------------
# What `make data` promises to cover
# ---------------------------------------------------------------------------


def test_every_yfinance_universe_member_is_a_price_source() -> None:
    universe = load().universe
    assert universe.yfinance_tickers == ("SPY", "IWM", "EFA", "EEM", "LQD", "HYG", "DBC", "GLD")
    assert "DTWEXBGS" not in universe.yfinance_tickers, "the dollar index is a FRED series"


def test_the_credit_sleeve_is_the_etf_headline_not_the_reconstruction() -> None:
    """SPEC.md 3.3 option (c): accept LQD from 2002 and HYG from 2007."""
    universe = load().universe
    assert universe.by_id("ig_credit").history_start == date(2002, 7, 30)
    assert universe.by_id("hy_credit").history_start == date(2007, 4, 11)


# ---------------------------------------------------------------------------
# W7-P2a -- SEC EDGAR shares outstanding (SPEC.md 15.4.1)
# ---------------------------------------------------------------------------


def test_fetch_edgar_shares_requires_the_user_agent_before_any_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """CLAUDE.md: secrets from the environment, never a literal. Raises by name."""
    from mafrm import secrets

    name = load().model.equity_shares.user_agent_env
    monkeypatch.setenv("MAFRM_ENV_FILE", str(tmp_path) + "/absent")  # type: ignore[operator]
    monkeypatch.delenv(name, raising=False)
    calls: list[str] = []

    def never(url: str) -> bytes | None:  # pragma: no cover - must not run
        calls.append(url)
        return None

    with pytest.raises(secrets.MissingSecretError, match=name):
        loaders.fetch_edgar_shares(get=never)
    assert calls == []


def test_edgar_quarters_status_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """404 is `empty` (the server answered), an exception is `failed` and retried once."""
    monkeypatch.setattr(loaders, "_EDGAR_PAUSE", 0.0)
    monkeypatch.setattr(loaders, "_EDGAR_RETRY_PAUSE", 0.0)
    attempts: dict[str, int] = {}

    def get(url: str) -> bytes | None:
        attempts[url] = attempts.get(url, 0) + 1
        if url.endswith("Q1"):
            return b"ok"
        if url.endswith("Q2"):
            return None
        if url.endswith("Q3"):
            if attempts[url] == 1:
                raise OSError("burst limit")
            return b"ok"
        raise OSError("down")

    def parse(payload: bytes, i: int) -> pd.DataFrame:
        return pd.DataFrame({"i": [i]}, index=pd.DatetimeIndex([pd.Timestamp("2020-01-01")]))

    labels = ["2020Q1", "2020Q2", "2020Q3", "2020Q4"]
    table, status, errors = loaders._edgar_quarters(
        labels, [f"u/{lab}" for lab in labels], parse=parse, user_agent="t", get=get
    )
    assert status == {"2020Q1": "ok", "2020Q2": "empty", "2020Q3": "ok", "2020Q4": "failed"}
    assert attempts == {"u/2020Q1": 1, "u/2020Q2": 1, "u/2020Q3": 2, "u/2020Q4": 2}
    assert "404" in errors["2020Q2"] and "down" in errors["2020Q4"]
    assert sorted(table["i"].tolist()) == [0, 2]
