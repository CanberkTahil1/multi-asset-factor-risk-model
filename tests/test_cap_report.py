"""The cap report's pure pieces, on the same hand-built panel as tests/test_edgar.py."""

from __future__ import annotations

import pandas as pd

from mafrm.data import market_cap
from mafrm.factors import cap_report


def _panel() -> tuple[market_cap.MarketCapPanel, pd.DataFrame]:
    sessions = pd.bdate_range("2020-01-02", periods=6, name="date")
    tickers = pd.Index(["AAA", "BBB", "CCC", "DDD"], name="ticker")
    shares = pd.DataFrame(
        {"AAA": 100.0, "BBB": 50.0, "CCC": float("nan"), "DDD": float("nan")},
        index=sessions,
        columns=tickers,
    )
    close = pd.DataFrame(10.0, index=sessions, columns=tickers)
    cap = shares * close
    approx = pd.DataFrame(False, index=sessions, columns=tickers)
    approx.loc[sessions[:2], "AAA"] = True  # AAA back-filled on the first two sessions
    mapping = pd.DataFrame(
        {
            "security": ["A", "B", "C", "D"],
            "current": [True, True, False, False],
            "cik": pd.array([1, 2, None, 3], dtype="Int64"),
            "method": ["ticker", "name", "unmapped", "name"],
            "status": ["ok", "ok", "unmapped", "no_facts"],
        },
        index=tickers,
    )
    panel = market_cap.MarketCapPanel(
        shares=shares,
        cap=cap,
        approximated=approx,
        mapping=mapping,
        jumps=pd.DataFrame(columns=["ticker", "filed", "previous", "shares", "ratio"]),
        sessions=sessions,
        inputs=(),
        counts={},
    )
    # Universe: AAA every session; BBB from the third; CCC and DDD every session.
    universe = pd.DataFrame(True, index=sessions, columns=tickers)
    universe.loc[sessions[:2], "BBB"] = False
    return panel, universe


def test_per_session_series_partitions_the_universe_by_cause() -> None:
    panel, universe = _panel()
    s = cap_report.per_session_series(panel, universe)
    assert s["universe"].tolist() == [3, 3, 4, 4, 4, 4]
    assert s["with_cap"].tolist() == [1, 1, 2, 2, 2, 2]
    assert s["no_cap"].tolist() == [2, 2, 2, 2, 2, 2]
    assert s["no_cap_unmapped"].tolist() == [1] * 6 and s["no_cap_no_facts"].tolist() == [1] * 6
    assert s["no_cap_ambiguous"].tolist() == [0] * 6
    assert s["approximated"].tolist() == [1, 1, 0, 0, 0, 0]
    # total cap: AAA 1000 alone, then AAA + BBB = 1500, in USD trn
    assert s["total_cap_usd_trn"].round(15).tolist() == [1e-9, 1e-9, 1.5e-9, 1.5e-9, 1.5e-9, 1.5e-9]
    assert (s["with_cap"] + s["no_cap"]).equals(s["universe"])


def test_approximated_share_counts_pairs_with_a_cap_only() -> None:
    panel, universe = _panel()
    # pairs with a cap: AAA x 6 + BBB x 4 = 10; back-filled: AAA on 2 -> 20%
    assert cap_report.approximated_share(panel, universe) == 20.0


def test_coverage_counts_by_list_and_method() -> None:
    panel, _ = _panel()
    c = cap_report.coverage_counts(panel.mapping)
    assert c["current"] == 2 and c["current_ticker"] == 1 and c["current_name"] == 1
    assert c["departed"] == 2 and c["departed_unmapped"] == 1 and c["departed_name"] == 1
    assert c["departed_no_facts"] == 1 and c["current_ok"] == 2 and c["departed_ok"] == 0
