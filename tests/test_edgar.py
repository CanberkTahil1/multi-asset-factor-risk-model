"""SPEC.md 15.4.1's point-in-time share counts, on hand-built fixtures.

Every expected value below was worked by hand from the fixtures before the
code ran (CLAUDE.md, definition of done). Nothing here fetches: the JSON and
index fixtures stand in for the cached EDGAR pulls.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mafrm.data import edgar, market_cap

FIXTURES = Path(__file__).parent / "fixtures"
FRAME_KW = dict(
    period="CY2020Q1I", taxonomy="dei", tag="EntityCommonStockSharesOutstanding", unit="shares"
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return edgar.parse_frame((FIXTURES / "edgar_frame_CY2020Q1I.json").read_bytes(), **FRAME_KW)


@pytest.fixture(scope="module")
def index() -> pd.DataFrame:
    return edgar.parse_xbrl_index(
        (FIXTURES / "edgar_xbrl_2020_QTR1.idx").read_bytes(), year=2020, quarter=1
    )


@pytest.fixture(scope="module")
def company_tickers() -> pd.DataFrame:
    return edgar.parse_company_tickers((FIXTURES / "edgar_company_tickers.json").read_bytes())


# ---------------------------------------------------------------------------
# Calendar quarters
# ---------------------------------------------------------------------------


def test_periods_run_from_the_first_quarter_to_the_pull_dates_quarter() -> None:
    assert edgar.periods("2009Q3", date(2010, 2, 1)) == ((2009, 3), (2009, 4), (2010, 1))
    assert edgar.frame_period(2009, 3) == "CY2009Q3I"
    assert edgar.period_label(2009, 3) == "2009Q3"
    assert edgar.quarter_of(date(2020, 12, 31)) == (2020, 4)


def test_periods_reject_a_bad_label_and_a_first_quarter_after_the_pull() -> None:
    with pytest.raises(edgar.EdgarError, match="look like 2009Q2"):
        edgar.periods("2009-Q3", date(2010, 1, 1))
    with pytest.raises(edgar.EdgarError, match="after the pull date"):
        edgar.periods("2011Q1", date(2010, 1, 1))


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def test_frame_parses_to_a_table_indexed_by_as_of_date(frame: pd.DataFrame) -> None:
    """Indexed by ``end`` -- that is what lets cache.read apply the holdout boundary."""
    assert isinstance(frame.index, pd.DatetimeIndex) and frame.index.name == "end"
    assert list(frame.columns) == ["period", "cik", "entity_name", "accn", "val"]
    assert len(frame) == 4
    alpha = frame[frame["cik"] == 1001].iloc[0]
    assert alpha["accn"] == "0000000001-20-000001" and alpha["val"] == 100.0
    assert alpha.name == pd.Timestamp("2020-01-20") and alpha["period"] == "CY2020Q1I"
    assert "filed" not in frame.columns  # the frames carry NO filing date (W7-P2a orientation)


def test_frame_refuses_a_response_for_a_different_fact() -> None:
    payload = (FIXTURES / "edgar_frame_CY2020Q1I.json").read_bytes()
    with pytest.raises(edgar.EdgarError, match="tag="):
        edgar.parse_frame(payload, **{**FRAME_KW, "tag": "EntityPublicFloat"})
    with pytest.raises(edgar.EdgarError, match="ccp="):
        edgar.parse_frame(payload, **{**FRAME_KW, "period": "CY2020Q2I"})
    with pytest.raises(edgar.EdgarError, match="not JSON"):
        edgar.parse_frame(b"<html>rate limited</html>", **FRAME_KW)


def test_xbrl_index_parses_filing_dates_and_accessions(index: pd.DataFrame) -> None:
    assert isinstance(index.index, pd.DatetimeIndex) and index.index.name == "filed"
    assert list(index.columns) == ["period", "cik", "company_name", "form", "accn"]
    assert len(index) == 5
    bravo = index[index["cik"] == 1002].iloc[0]
    assert bravo.name == pd.Timestamp("2020-02-14")
    assert bravo["accn"] == "0000000002-20-000002" and bravo["form"] == "10-K"
    assert bravo["period"] == "2020Q1"


def test_xbrl_index_without_its_header_raises_by_name() -> None:
    with pytest.raises(edgar.EdgarError, match="header"):
        edgar.parse_xbrl_index("Description: nothing\n\n1|2|3\n", year=2020, quarter=1)


def test_company_tickers_parse_upper_case(company_tickers: pd.DataFrame) -> None:
    assert list(company_tickers.columns) == ["cik", "ticker", "title"]
    assert company_tickers.set_index("ticker").loc["ALP", "cik"] == 1001


# ---------------------------------------------------------------------------
# Name normalisation and CIK mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "key"),
    [
        ("Alcoa Inc.", "ALCOA"),
        ("ALCOA INC", "ALCOA"),
        ("Berkshire Hathaway Inc. (Class B)", "BERKSHIRE HATHAWAY"),
        ("WALT DISNEY CO/", "WALT DISNEY"),
        ("The Walt Disney Company", "WALT DISNEY"),
        ("Procter & Gamble", "PROCTER AND GAMBLE"),
        ("PROCTER & GAMBLE CO", "PROCTER AND GAMBLE"),
        ("WATERS CORP /DE/", "WATERS"),
        ("Fox Corporation (Class A)", "FOX"),
        ("Alpha Holdings", "ALPHA HOLDINGS"),  # HOLDINGS is kept: distinct issuers stay distinct
    ],
)
def test_normalize_name(raw: str, key: str) -> None:
    assert edgar.normalize_name(raw) == key


def test_map_ciks_current_by_ticker_departed_by_name_and_the_two_failure_modes(
    frame: pd.DataFrame, index: pd.DataFrame, company_tickers: pd.DataFrame
) -> None:
    securities = {
        "ALP": "Alpha Inc.",  # current, in SEC's map by ticker
        "WYE": "Wyeth",  # departed: name matches one filer (cik 1003)
        "CHL": "Charlie Corp.",  # departed: two CIKs filed under this name -> ambiguous
        "ZZZ": "Zulu Enterprises",  # departed: no filer -> unmapped
        "RCY": "Old Recycled Co",  # departed: the ticker exists in SEC's map for ANOTHER
        # company, and the name matches nothing -> unmapped, never by ticker
        "BRV": "Bravo Holdings",  # current, ticker in SEC's map -> by ticker
    }
    names = [
        *((int(c), str(n)) for c, n in frame[["cik", "entity_name"]].itertuples(index=False)),
        *((int(c), str(n)) for c, n in index[["cik", "company_name"]].itertuples(index=False)),
    ]
    mapping = edgar.map_ciks(securities, {"ALP", "BRV"}, company_tickers, names)
    assert list(mapping.index) == sorted(securities)
    assert mapping.loc["ALP", "method"] == "ticker" and mapping.loc["ALP", "cik"] == 1001
    assert mapping.loc["BRV", "method"] == "ticker" and mapping.loc["BRV", "cik"] == 1002
    assert mapping.loc["WYE", "method"] == "name" and mapping.loc["WYE", "cik"] == 1003
    assert mapping.loc["CHL", "method"] == "ambiguous" and pd.isna(mapping.loc["CHL", "cik"])
    assert mapping.loc["ZZZ", "method"] == "unmapped" and pd.isna(mapping.loc["ZZZ", "cik"])
    assert mapping.loc["RCY", "method"] == "unmapped"
    assert bool(mapping.loc["ALP", "current"]) and not bool(mapping.loc["WYE", "current"])


# ---------------------------------------------------------------------------
# The point-in-time rule, by hand
# ---------------------------------------------------------------------------


def _facts(rows: list[tuple[str, float, str]]) -> pd.DataFrame:
    """Frame rows for one filer: (as-of date, value, accession)."""
    table = pd.DataFrame(
        {
            "period": "CY2020Q1I",
            "cik": 1001,
            "entity_name": "ALPHA INC",
            "accn": [r[2] for r in rows],
            "val": [r[1] for r in rows],
        },
        index=pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows], name="end"),
    )
    return table


def _filed(pairs: dict[str, str]) -> pd.Series:
    return pd.Series(
        pd.DatetimeIndex([pd.Timestamp(v) for v in pairs.values()]), index=list(pairs), name="filed"
    )


def test_adjusted_counts_apply_every_split_after_the_as_of_date() -> None:
    """A 2-for-1 on 2020-02-10 sits between the first count's as-of date and the second's.

    First count 100 as of 2020-01-20 -> 200 on the cache's basis; second count 120
    as of 2020-02-20, after the split -> 120 unchanged.
    """
    facts = _facts([("2020-01-20", 100.0, "A1"), ("2020-02-20", 120.0, "A2")])
    filed = _filed({"A1": "2020-01-31", "A2": "2020-03-02"})
    splits = pd.Series([2.0], index=pd.DatetimeIndex(["2020-02-10"]))
    out = edgar.adjusted_counts(facts, filed, splits)
    assert out.no_filing_date == 0 and out.nonpositive == 0
    assert list(out.table.index) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-03-02")]
    assert out.table["shares"].tolist() == [200.0, 120.0]
    assert out.table["val"].tolist() == [100.0, 120.0]


def test_adjusted_counts_drop_and_count_the_two_named_reasons() -> None:
    facts = _facts(
        [("2020-01-20", 100.0, "A1"), ("2020-02-20", 0.0, "A2"), ("2020-03-20", 130.0, "A3")]
    )
    filed = _filed({"A1": "2020-01-31", "A2": "2020-03-02"})  # A3 is in no index
    out = edgar.adjusted_counts(facts, filed, None)
    assert out.no_filing_date == 1 and out.nonpositive == 1
    assert out.table["shares"].tolist() == [100.0]


def test_adjusted_counts_keep_the_latest_as_of_date_on_one_filing_date() -> None:
    facts = _facts([("2020-01-10", 90.0, "A0"), ("2020-01-20", 100.0, "A1")])
    filed = _filed({"A0": "2020-01-31", "A1": "2020-01-31"})
    out = edgar.adjusted_counts(facts, filed, None)
    assert len(out.table) == 1 and out.table["shares"].iloc[0] == 100.0


def test_point_in_time_panel_is_known_from_the_filing_date_and_backfilled_before_it() -> None:
    """Sessions 2020-01-02..2020-03-31 (business days). Filings: Fri 2020-01-31 (200),
    Mon 2020-03-02 (120), Sat 2020-03-14 (130, in force from Mon 2020-03-16)."""
    sessions = pd.bdate_range("2020-01-02", "2020-03-31", name="date")
    table = pd.DataFrame(
        {
            "end": pd.to_datetime(["2020-01-20", "2020-02-20", "2020-03-10"]),
            "val": [100.0, 120.0, 130.0],
            "accn": ["A1", "A2", "A3"],
            "shares": [200.0, 120.0, 130.0],
        },
        index=pd.DatetimeIndex(["2020-01-31", "2020-03-02", "2020-03-14"], name="filed"),
    )
    empty = table.iloc[0:0]
    shares, approx = edgar.point_in_time_panel({"AAA": table, "NONE": empty}, sessions)
    s = shares["AAA"]
    assert s.loc["2020-01-02"] == 200.0 and bool(approx.loc["2020-01-02", "AAA"])
    assert s.loc["2020-01-30"] == 200.0 and bool(approx.loc["2020-01-30", "AAA"])
    assert s.loc["2020-01-31"] == 200.0 and not bool(approx.loc["2020-01-31", "AAA"])
    assert s.loc["2020-02-28"] == 200.0
    assert s.loc["2020-03-02"] == 120.0
    assert s.loc["2020-03-13"] == 120.0  # Friday before the Saturday filing
    assert s.loc["2020-03-16"] == 130.0  # first session on or after it
    assert s.loc["2020-03-31"] == 130.0
    assert int(approx["AAA"].sum()) == len(sessions[sessions < "2020-01-31"]) == 21
    assert shares["NONE"].isna().all() and not approx["NONE"].any()


def test_unexplained_jumps_list_ratios_outside_the_display_band() -> None:
    table = pd.DataFrame(
        {
            "end": pd.to_datetime(["2020-01-20", "2020-02-20", "2020-03-20"]),
            "val": [100.0, 1.0, 100.0],
            "accn": ["A1", "A2", "A3"],
            "shares": [100.0, 1.0, 100.0],
        },
        index=pd.DatetimeIndex(["2020-01-31", "2020-03-02", "2020-04-01"], name="filed"),
    )
    jumps = edgar.unexplained_jumps({"AAA": table})
    assert len(jumps) == 2
    assert jumps["ratio"].tolist() == [0.01, 100.0]
    assert edgar.unexplained_jumps({"AAA": table.iloc[[0, 2]]}).empty


# ---------------------------------------------------------------------------
# The assembled panel
# ---------------------------------------------------------------------------


def test_splits_by_ticker_reads_only_real_splits() -> None:
    actions = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "AAA"],
            "Dividends": [0.5, 0.0, 0.0, 0.0],
            "Stock Splits": [0.0, 2.0, 1.0, 4.0],
        },
        index=pd.DatetimeIndex(["2020-01-05", "2020-02-10", "2020-02-11", "2020-06-01"]),
    )
    out = market_cap.splits_by_ticker(actions)
    assert list(out) == ["AAA"]
    assert out["AAA"].tolist() == [2.0, 4.0]


def test_assembled_panel_by_hand(
    frame: pd.DataFrame, index: pd.DataFrame, company_tickers: pd.DataFrame
) -> None:
    """Six tickers: ALP (current, split), BRV (current), WYE (departed, val 1),
    CHL (ambiguous), ZZZ (unmapped), ORP (mapped by name, but its only fact's
    accession is in no index -> no_facts)."""
    sessions = pd.bdate_range("2020-01-02", "2020-03-31", name="date")
    close = pd.DataFrame(
        10.0, index=sessions, columns=pd.Index(["ALP", "BRV", "WYE", "CHL", "ZZZ", "ORP"])
    )
    close["BRV"] = 20.0
    intervals = pd.DataFrame(
        {
            "ticker": ["ALP", "BRV", "WYE", "CHL", "ZZZ", "ORP"],
            "start": [pd.NaT] * 6,
            "end": [pd.NaT, pd.NaT, "2019-06-01", "2018-01-01", "2017-01-01", "2016-01-01"],
            "start_known": [False] * 6,
            "security": [
                "Alpha Inc.",
                "Bravo Holdings",
                "Wyeth",
                "Charlie Corp.",
                "Zulu Enterprises",
                "Orphan Filer",
            ],
        }
    )
    intervals["end"] = pd.to_datetime(intervals["end"])
    splits = {"ALP": pd.Series([2.0], index=pd.DatetimeIndex(["2020-01-25"]))}
    panel = market_cap.assemble(
        frames=frame,
        index=index,
        company_tickers=company_tickers,
        close=close,
        splits=splits,
        intervals=intervals,
        sessions=sessions,
    )
    m = panel.mapping
    assert m.loc["ALP", "status"] == "ok" and m.loc["ALP", "method"] == "ticker"
    assert m.loc["BRV", "status"] == "ok"
    assert m.loc["WYE", "status"] == "ok" and m.loc["WYE", "method"] == "name"
    assert m.loc["CHL", "status"] == "ambiguous"
    assert m.loc["ZZZ", "status"] == "unmapped"
    assert m.loc["ORP", "status"] == "no_facts" and m.loc["ORP", "no_filing_date"] == 1
    # ALP: 100 as of 2020-01-20, split 2-for-1 on 2020-01-25 (after the as-of date),
    # filed 2020-01-31 -> 200 on the cache's basis, in force from 2020-01-31 and
    # back-filled before it.
    assert panel.shares.loc["2020-01-02", "ALP"] == 200.0 and bool(
        panel.approximated.loc["2020-01-02", "ALP"]
    )
    assert panel.shares.loc["2020-03-31", "ALP"] == 200.0 and not bool(
        panel.approximated.loc["2020-03-31", "ALP"]
    )
    assert panel.cap.loc["2020-03-31", "ALP"] == 2000.0
    # BRV: 50 as of 2020-01-31, filed 2020-02-14, no split -> 50; cap 50 x 20 = 1000.
    assert panel.shares.loc["2020-02-13", "BRV"] == 50.0 and bool(
        panel.approximated.loc["2020-02-13", "BRV"]
    )
    assert panel.cap.loc["2020-02-14", "BRV"] == 1000.0 and not bool(
        panel.approximated.loc["2020-02-14", "BRV"]
    )
    # WYE: the placeholder value 1 is used as filed -- listed, not filtered.
    assert (
        panel.shares.loc["2020-03-31", "WYE"] == 1.0 and panel.cap.loc["2020-03-31", "WYE"] == 10.0
    )
    for ticker in ("CHL", "ZZZ", "ORP"):
        assert panel.cap[ticker].isna().all() and not panel.approximated[ticker].any()
    assert panel.has_cap().sum(axis=1).tolist() == [3] * len(sessions)
    assert panel.jumps.empty  # one filing each: nothing to compare
    assert panel.sessions.equals(sessions)


def test_assembled_panel_refuses_a_calendar_mismatch() -> None:
    sessions = pd.bdate_range("2020-01-02", periods=5)
    with pytest.raises(edgar.EdgarError, match="strictly increasing"):
        edgar.point_in_time_panel({}, sessions[::-1])


def test_panel_invariant_rejects_a_nonpositive_cap() -> None:
    sessions = pd.bdate_range("2020-01-02", periods=2)
    shares = pd.DataFrame({"A": [1.0, 1.0]}, index=sessions)
    cap = pd.DataFrame({"A": [1.0, -1.0]}, index=sessions)
    with pytest.raises(ValueError, match="not positive"):
        market_cap.MarketCapPanel(
            shares=shares,
            cap=cap,
            approximated=shares.astype(bool),
            mapping=pd.DataFrame(),
            jumps=pd.DataFrame(),
            sessions=sessions,
            inputs=(),
        )
    assert np.isfinite(cap.to_numpy()).all()


# ---------------------------------------------------------------------------
# SPEC.md 15.4.3 ruling (i): the within-name consistency screen
# ---------------------------------------------------------------------------


def _table(rows: list[tuple[str, float]]) -> pd.DataFrame:
    """Adjusted counts keyed by filing date: (filed, shares), val = shares, no splits."""
    return pd.DataFrame(
        {
            "end": pd.to_datetime([r[0] for r in rows]) - pd.Timedelta(days=10),
            "val": [r[1] for r in rows],
            "accn": [f"A{i}" for i in range(len(rows))],
            "shares": [r[1] for r in rows],
        },
        index=pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows], name="filed"),
    )


def test_consistency_screen_drops_unit_scale_errors_and_shells_and_keeps_missing_splits() -> None:
    """Hand-worked: counts 100, 100e6 (tagged in shares once), 110, 1, 55, 105.

    Median of (100, 1e8, 110, 1, 55, 105) = (100 + 105) / 2 = 102.5. Ratios:
    0.976, 975,610 (drop), 1.073, 0.00976 (drop, a shell), 0.537 (a 2:1 the
    actions table lacks -- kept at R = 100), 1.024.
    """
    table = _table(
        [
            ("2020-01-31", 100.0),
            ("2020-04-30", 100e6),
            ("2020-07-31", 110.0),
            ("2020-10-30", 1.0),
            ("2021-01-29", 55.0),
            ("2021-04-30", 105.0),
        ]
    )
    out = edgar.consistency_screen(table, ratio=100.0)
    assert out.median == 102.5
    assert out.table["shares"].tolist() == [100.0, 110.0, 55.0, 105.0]
    assert out.dropped["shares"].tolist() == [100e6, 1.0]
    assert out.dropped["median"].tolist() == [102.5, 102.5]
    assert out.dropped["ratio"].tolist() == [100e6 / 102.5, 1.0 / 102.5]
    assert list(out.dropped.columns) == ["end", "val", "accn", "shares", "median", "ratio"]


def test_consistency_screen_the_previous_count_stays_in_force_and_the_backfill_reads_a_survivor() -> (
    None
):
    """First filing is the error: 1,000,000 (thousands) then 1,020, 1,030.

    Median 1,020; the first is 980x off and drops. The back-fill before the
    first SURVIVING filing (2020-05-01) reads 1,020, and the sessions between
    the dropped filing and the survivor carry 1,020 too -- there is no earlier
    survivor to stay in force, so the ruling's "previous count stays in force"
    resolves to the back-fill. With the error in the MIDDLE instead, the count
    before it stays in force through its date.
    """
    sessions = pd.bdate_range("2020-01-02", "2020-09-30", name="date")
    first_is_error = _table(
        [("2020-01-31", 1_000_000.0), ("2020-05-01", 1020.0), ("2020-08-03", 1030.0)]
    )
    screened = edgar.consistency_screen(first_is_error, ratio=100.0)
    shares, approx = edgar.point_in_time_panel({"X": screened.table}, sessions)
    assert shares.loc["2020-01-02", "X"] == 1020.0 and bool(approx.loc["2020-01-02", "X"])
    assert shares.loc["2020-03-02", "X"] == 1020.0 and bool(approx.loc["2020-03-02", "X"])
    assert shares.loc["2020-05-01", "X"] == 1020.0 and not bool(approx.loc["2020-05-01", "X"])
    assert shares.loc["2020-09-30", "X"] == 1030.0

    middle_is_error = _table(
        [("2020-01-31", 1000.0), ("2020-05-01", 1_020_000.0), ("2020-08-03", 1030.0)]
    )
    screened = edgar.consistency_screen(middle_is_error, ratio=100.0)
    shares, _ = edgar.point_in_time_panel({"X": screened.table}, sessions)
    assert shares.loc["2020-05-01", "X"] == 1000.0  # the previous count stays in force
    assert shares.loc["2020-07-31", "X"] == 1000.0
    assert shares.loc["2020-08-03", "X"] == 1030.0


def test_consistency_screen_on_two_filings_drops_the_far_one_only() -> None:
    """Two counts 1 and 1e9: median 5e8 + 0.5; the shell is ~2e-9 x, dropped; the real one ~2x, kept."""
    out = edgar.consistency_screen(_table([("2020-01-31", 1.0), ("2020-05-01", 1e9)]), ratio=100.0)
    assert out.table["shares"].tolist() == [1e9] and out.dropped["shares"].tolist() == [1.0]


def test_consistency_screen_leaves_a_single_or_empty_table_alone() -> None:
    one = _table([("2020-01-31", 7.0)])
    assert edgar.consistency_screen(one, ratio=100.0).table.equals(one)
    empty = edgar.consistency_screen(one.iloc[0:0], ratio=100.0)
    assert empty.table.empty and empty.dropped.empty and np.isnan(empty.median)
    with pytest.raises(edgar.EdgarError, match="ratio must exceed 1"):
        edgar.consistency_screen(one, ratio=1.0)


def test_assembled_panel_applies_the_screen_and_records_every_drop(
    frame: pd.DataFrame, index: pd.DataFrame, company_tickers: pd.DataFrame
) -> None:
    """WYE's only fact is the placeholder 1: with one filing the median IS the
    count, so nothing drops -- a shell with a single filing survives the screen,
    and the report's residual listing is where it shows. Add a second WYE fact
    at 1,000,000 and the 1 drops (median 500,000.5; 1 is 2e-6 x)."""
    sessions = pd.bdate_range("2020-01-02", "2020-03-31", name="date")
    tickers = ["ALP", "BRV", "WYE"]
    close = pd.DataFrame(10.0, index=sessions, columns=pd.Index(tickers))
    intervals = pd.DataFrame(
        {
            "ticker": tickers,
            "start": [pd.NaT] * 3,
            "end": pd.to_datetime([pd.NaT, pd.NaT, "2019-06-01"]),
            "start_known": [False] * 3,
            "security": ["Alpha Inc.", "Bravo Holdings", "Wyeth"],
        }
    )
    second = frame[frame["cik"] == 1003].copy()
    second.index = pd.DatetimeIndex([pd.Timestamp("2020-03-05")], name="end")
    second["accn"] = "0000000003-20-000004"
    second["val"] = 1_000_000.0
    index2 = pd.concat(
        [
            index,
            pd.DataFrame(
                {
                    "period": ["2020Q1"],
                    "cik": [1003],
                    "company_name": ["WYETH"],
                    "form": ["10-Q"],
                    "accn": ["0000000003-20-000004"],
                },
                index=pd.DatetimeIndex([pd.Timestamp("2020-03-10")], name="filed"),
            ),
        ]
    )
    common = dict(
        company_tickers=company_tickers,
        close=close,
        splits={},
        intervals=intervals,
        sessions=sessions,
    )
    unscreened = market_cap.assemble(frames=pd.concat([frame, second]), index=index2, **common)
    assert not unscreened.is_model_input and unscreened.dropped.empty
    assert unscreened.shares.loc["2020-03-09", "WYE"] == 1.0
    with pytest.raises(ValueError, match="unscreened"):
        unscreened.require_model_input()

    screened = market_cap.assemble(
        frames=pd.concat([frame, second]), index=index2, consistency_ratio=100.0, **common
    )
    assert screened.is_model_input and screened.consistency_ratio == 100.0
    assert screened.dropped["ticker"].tolist() == ["WYE"]
    assert screened.dropped["shares"].tolist() == [1.0]
    assert screened.dropped["median"].tolist() == [500_000.5]
    assert screened.mapping.loc["WYE", "dropped"] == 1 and screened.mapping.loc["WYE", "used"] == 1
    # the survivor is back-filled over the sessions the placeholder used to cover
    assert screened.shares.loc["2020-02-14", "WYE"] == 1_000_000.0
    assert bool(screened.approximated.loc["2020-02-14", "WYE"])
    assert screened.shares.loc["2020-03-10", "WYE"] == 1_000_000.0
    assert not bool(screened.approximated.loc["2020-03-10", "WYE"])
    assert screened.mapping.loc["ALP", "dropped"] == 0


# ---------------------------------------------------------------------------
# SPEC.md 15.4.3 ruling (ii): the predecessor-CIK hand table
# ---------------------------------------------------------------------------


def _override(ticker: str, current: int, predecessor: int) -> market_cap.CikOverride:
    return market_cap.CikOverride(
        ticker=ticker,
        current_cik=current,
        predecessor_cik=predecessor,
        current_source_url="https://www.sec.gov/cgi-bin/browse-edgar?CIK=current",
        predecessor_source_url="https://www.sec.gov/cgi-bin/browse-edgar?CIK=predecessor",
        date_read=date(2026, 9, 5),
    )


def _exclusion(ticker: str, current: int) -> market_cap.CikOverride:
    return market_cap.CikOverride(
        ticker=ticker,
        current_cik=current,
        current_source_url="https://www.sec.gov/cgi-bin/browse-edgar?CIK=current",
        date_read=date(2026, 9, 5),
        treat_as=market_cap.TREAT_AS_NO_FACTS,
    )


def test_a_treat_as_no_facts_row_removes_the_name_by_hand_and_counts_its_facts(
    frame: pd.DataFrame, index: pd.DataFrame, company_tickers: pd.DataFrame
) -> None:
    """W7-P2b ruling 3 on ALP: one fact, mapped by ticker, forced to no_facts; nothing fabricated."""
    sessions = pd.bdate_range("2020-01-02", "2020-03-31", name="date")
    close = pd.DataFrame(10.0, index=sessions, columns=pd.Index(["ALP", "BRV"]))
    intervals = pd.DataFrame(
        {
            "ticker": ["ALP", "BRV"],
            "start": [pd.NaT, pd.NaT],
            "end": pd.to_datetime([pd.NaT, pd.NaT]),
            "start_known": [False, False],
            "security": ["Alpha Inc.", "Bravo Holdings"],
        }
    )
    panel = market_cap.assemble(
        frames=frame,
        index=index,
        company_tickers=company_tickers,
        close=close,
        splits={},
        intervals=intervals,
        sessions=sessions,
        consistency_ratio=100.0,
        overrides=(_exclusion("ALP", 1001),),
    )
    m = panel.mapping.loc["ALP"]
    assert m["status"] == "no_facts" and m["treat_as"] == "no_facts"
    assert int(m["facts"]) == 1 and int(m["used"]) == 0 and int(m["cik"]) == 1001
    assert panel.cap["ALP"].isna().all() and not panel.approximated["ALP"].any()
    assert panel.excluded == {"ALP": "no_facts"} and panel.overrides == {}
    assert panel.mapping.loc["BRV", "status"] == "ok" and pd.isna(
        panel.mapping.loc["BRV", "treat_as"]
    )
    with pytest.raises(market_cap.HandTableError, match="ALP maps to CIK 1001"):
        market_cap.assemble(
            frames=frame,
            index=index,
            company_tickers=company_tickers,
            close=close,
            splits={},
            intervals=intervals,
            sessions=sessions,
            overrides=(_exclusion("ALP", 4242),),
        )


def test_override_reads_the_predecessors_facts_beside_the_mapped_cik(
    frame: pd.DataFrame, index: pd.DataFrame, company_tickers: pd.DataFrame
) -> None:
    """RCY maps by ticker to CIK 1010 (no facts). With predecessor 1001 (ALPHA's
    single fact, 100 as of 2020-01-20, filed 2020-01-31) it gets a cap."""
    sessions = pd.bdate_range("2020-01-02", "2020-03-31", name="date")
    close = pd.DataFrame(10.0, index=sessions, columns=pd.Index(["ALP", "RCY"]))
    intervals = pd.DataFrame(
        {
            "ticker": ["ALP", "RCY"],
            "start": [pd.NaT, pd.NaT],
            "end": pd.to_datetime([pd.NaT, pd.NaT]),
            "start_known": [False, False],
            "security": ["Alpha Inc.", "Recycled Ticker Co"],
        }
    )
    common = dict(
        frames=frame,
        index=index,
        company_tickers=company_tickers,
        close=close,
        splits={},
        intervals=intervals,
        sessions=sessions,
        consistency_ratio=100.0,
    )
    without = market_cap.assemble(**common)
    assert without.mapping.loc["RCY", "status"] == "no_facts"
    assert without.cap["RCY"].isna().all()

    with_override = market_cap.assemble(overrides=(_override("RCY", 1010, 1001),), **common)
    m = with_override.mapping
    assert m.loc["RCY", "status"] == "ok" and int(m.loc["RCY", "predecessor_cik"]) == 1001
    assert int(m.loc["RCY", "cik"]) == 1010 and m.loc["RCY", "method"] == "ticker"
    assert with_override.shares.loc["2020-03-31", "RCY"] == 100.0
    assert with_override.cap.loc["2020-03-31", "RCY"] == 1000.0
    assert with_override.overrides == {"RCY": 1001}
    assert m["predecessor_cik"].isna().sum() == 1  # ALP has none

    # A stale row -- the ticker maps elsewhere on this pull -- is refused by name.
    with pytest.raises(market_cap.HandTableError, match="RCY maps to CIK 1010"):
        market_cap.assemble(overrides=(_override("RCY", 9999, 1001),), **common)
    with pytest.raises(market_cap.HandTableError, match="has no fact"):
        market_cap.assemble(overrides=(_override("RCY", 1010, 4242),), **common)
    with pytest.raises(market_cap.HandTableError, match="not a ticker"):
        market_cap.assemble(overrides=(_override("NOPE", 1, 2),), **common)


def test_hand_tables_parse_and_refuse_a_row_without_provenance(tmp_path: Path) -> None:
    good = tmp_path / "cik_overrides.yaml"
    good.write_text(
        "schema_version: 1\nentries:\n  - ticker: xom\n    current_cik: 2115436\n"
        "    predecessor_cik: 34088\n    current_source_url: https://a\n"
        "    predecessor_source_url: https://b\n    date_read: 2026-09-05\n"
    )
    rows = market_cap.load_cik_overrides(good)
    assert len(rows) == 1 and rows[0].ticker == "XOM" and rows[0].date_read == date(2026, 9, 5)
    assert not rows[0].excludes and rows[0].predecessor_cik == 34088
    mixed = tmp_path / "mixed.yaml"
    mixed.write_text(
        "schema_version: 1\nentries:\n  - ticker: brk-b\n    current_cik: 1067983\n"
        "    current_source_url: https://a\n    treat_as: no_facts\n    date_read: 2026-09-05\n"
    )
    excl = market_cap.load_cik_overrides(mixed)
    assert excl[0].excludes and excl[0].ticker == "BRK-B" and excl[0].predecessor_cik is None
    both = tmp_path / "both.yaml"
    both.write_text(
        "schema_version: 1\nentries:\n  - ticker: X\n    current_cik: 1\n    predecessor_cik: 2\n"
        "    current_source_url: https://a\n    predecessor_source_url: https://b\n"
        "    treat_as: no_facts\n    date_read: 2026-09-05\n"
    )
    with pytest.raises(market_cap.HandTableError, match="not both or neither"):
        market_cap.load_cik_overrides(both)
    wrong = tmp_path / "wrong.yaml"
    wrong.write_text(
        "schema_version: 1\nentries:\n  - ticker: X\n    current_cik: 1\n"
        "    current_source_url: https://a\n    treat_as: delete\n    date_read: 2026-09-05\n"
    )
    with pytest.raises(market_cap.HandTableError, match="treat_as must be"):
        market_cap.load_cik_overrides(wrong)
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "schema_version: 1\nentries:\n  - ticker: XOM\n    current_cik: 1\n    predecessor_cik: 2\n"
    )
    with pytest.raises(market_cap.HandTableError, match="current_source_url"):
        market_cap.load_cik_overrides(bad)
    same = tmp_path / "same.yaml"
    same.write_text(
        "schema_version: 1\nentries:\n  - ticker: XOM\n    current_cik: 5\n    predecessor_cik: 5\n"
        "    current_source_url: https://a\n    predecessor_source_url: https://b\n    date_read: 2026-09-05\n"
    )
    with pytest.raises(market_cap.HandTableError, match="equals current_cik"):
        market_cap.load_cik_overrides(same)

    totals = tmp_path / "totals.yaml"
    totals.write_text(
        "schema_version: 1\nseries:\n  label: L\n  source_url: https://c\n  date_read: 2026-09-05\n"
        '  unit: "USD millions"\n  values:\n    "2007-12-31": 19670052.9\n    "2008-12-31": 11461287.6\n'
    )
    published = market_cap.load_published_totals(totals)
    assert published.unit == "USD millions"
    assert published.values_usd.loc["2007-12-31"] == 19670052.9e6
    assert list(published.values_usd.index) == [
        pd.Timestamp("2007-12-31"),
        pd.Timestamp("2008-12-31"),
    ]
    totals.write_text(totals.read_text().replace("USD millions", "USD squillions"))
    with pytest.raises(market_cap.HandTableError, match="unit must be one of"):
        market_cap.load_published_totals(totals)


def test_the_committed_hand_tables_parse() -> None:
    from mafrm import config as config_mod

    cfg = config_mod.load()
    rows = market_cap.load_cik_overrides(
        config_mod.config_dir() / cfg.model.equity_shares.overrides_file
    )
    assert [r.ticker for r in rows] == ["XOM", "BRK-B"]
    assert rows[0].current_cik == 2115436 and rows[0].predecessor_cik == 34088
    assert rows[1].excludes and rows[1].current_cik == 1067983 and rows[1].predecessor_cik is None
    published = market_cap.load_published_totals(
        config_mod.config_dir() / cfg.model.equity_shares.published_totals_file
    )
    assert len(published.values_usd) == 18
    assert published.values_usd.index.min() == pd.Timestamp("2007-12-31")
    assert published.values_usd.index.max() == pd.Timestamp("2024-12-31")
    assert "upper bound" in published.label
