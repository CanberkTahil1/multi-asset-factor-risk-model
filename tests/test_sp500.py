"""SPEC.md 15.3's membership reconstruction, on a hand-built fixture.

Every expected value below was worked by hand from the two HTML fixtures
before the code ran (CLAUDE.md, definition of done). Nothing here fetches:
the fixtures stand in for the two cached Wikipedia pages.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from mafrm import config
from mafrm.data import sp500

FIXTURES = Path(__file__).parent / "fixtures"
SNAPSHOT = pd.Timestamp("2024-06-30")


@pytest.fixture(scope="module")
def constituents() -> pd.DataFrame:
    return sp500.parse_constituents((FIXTURES / "sp500_constituents.html").read_bytes())


@pytest.fixture(scope="module")
def changes() -> pd.DataFrame:
    return sp500.parse_changes((FIXTURES / "sp500_changes.html").read_bytes())


@pytest.fixture(scope="module")
def rec(constituents: pd.DataFrame, changes: pd.DataFrame) -> sp500.Reconstruction:
    return sp500.reconstruct(constituents, changes, snapshot=SNAPSHOT)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_constituents_parse_to_vendor_symbols_and_dates(constituents: pd.DataFrame) -> None:
    assert list(constituents["ticker"]) == ["AAA", "BBB", "META2", "DDD", "BRK-B", "EEE"]
    assert constituents.loc[4, "wikipedia_ticker"] == "BRK.B"
    assert constituents.loc[1, "date_added"] == pd.Timestamp("2010-03-01")
    assert pd.isna(constituents.loc[5, "date_added"])  # blank cell -> NaT, not a guess


def test_changes_parse_every_row_with_blank_sides_as_none(changes: pd.DataFrame) -> None:
    assert len(changes) == 6
    assert changes.loc[0, "effective_date"] == pd.Timestamp("2025-03-03")
    assert changes.loc[3, "removed"] is None and changes.loc[3, "added"] == "BBB"
    assert changes.loc[4, "added"] is None and changes.loc[4, "removed"] == "OLD2"


def test_a_page_without_the_table_raises_by_name() -> None:
    with pytest.raises(sp500.SP500Error, match="no <table id='changes'>"):
        sp500.parse_changes((FIXTURES / "sp500_constituents.html").read_bytes())


def test_universe_tickers_is_the_union_of_both_tables(
    constituents: pd.DataFrame, changes: pd.DataFrame
) -> None:
    assert sp500.universe_tickers(constituents, changes) == (
        "AAA",
        "BBB",
        "BRK-B",
        "DDD",
        "EEE",
        "FB2",
        "FUT",
        "META2",
        "NEW1",
        "OLD1",
        "OLD2",
        "OLD3",
    )


def test_vendor_symbol_map() -> None:
    assert sp500.to_vendor_symbol("BRK.B") == "BRK-B"
    assert sp500.to_vendor_symbol(" bf.b ") == "BF-B"
    # The 2026-09-04 snapshot carried a stray table pipe inside three cells.
    assert sp500.to_vendor_symbol("ALLE |") == "ALLE"
    with pytest.raises(sp500.SP500Error, match="not a ticker symbol"):
        sp500.to_vendor_symbol("|")
    with pytest.raises(sp500.SP500Error, match="not a ticker symbol"):
        sp500.to_vendor_symbol("A/B")


def test_reconstruction_is_identical_after_a_parquet_round_trip(
    constituents: pd.DataFrame, changes: pd.DataFrame
) -> None:
    """The cache reads blank sides back as NaN under a string dtype, not None.

    The lost attempt's first real run walked `nan` as a ticker for that reason;
    this pins that a table read through the cache reconstructs exactly as a
    freshly parsed one.
    """
    from mafrm.data import cache

    direct = sp500.reconstruct(constituents, changes, snapshot=SNAPSHOT)
    via_cache = sp500.reconstruct(
        cache._deserialize(cache._serialize(constituents)),
        cache._deserialize(cache._serialize(changes)),
        snapshot=SNAPSHOT,
    )
    pd.testing.assert_frame_equal(direct.intervals, via_cache.intervals)
    assert direct.inconsistencies == via_cache.inconsistencies
    assert direct.rename_links == via_cache.rename_links
    assert "nan" not in set(via_cache.intervals["ticker"])


# ---------------------------------------------------------------------------
# The walk, by hand
# ---------------------------------------------------------------------------


def _spell(rec: sp500.Reconstruction, ticker: str) -> pd.Series:
    rows = rec.intervals[rec.intervals["ticker"] == ticker]
    assert len(rows) == 1, ticker
    return rows.iloc[0]


def test_reach_excludes_the_row_after_the_snapshot(rec: sp500.Reconstruction) -> None:
    assert rec.reach == pd.Timestamp("2005-06-15")
    # The 2025 row is after the snapshot: AAA is still open, FUT never appears.
    assert pd.isna(_spell(rec, "AAA")["end"])
    assert "FUT" not in set(rec.intervals["ticker"])


def test_current_members_without_a_row_reach_back_with_start_unknown(
    rec: sp500.Reconstruction,
) -> None:
    for ticker in ("AAA", "BRK-B", "EEE", "DDD"):
        spell = _spell(rec, ticker)
        assert pd.isna(spell["start"]) and pd.isna(spell["end"])
        assert not spell["start_known"]


def test_an_addition_row_opens_the_spell(rec: sp500.Reconstruction) -> None:
    spell = _spell(rec, "BBB")
    assert spell["start"] == pd.Timestamp("2010-03-01") and pd.isna(spell["end"])
    assert spell["start_known"]


def test_removals_close_spells_exclusively(rec: sp500.Reconstruction) -> None:
    assert _spell(rec, "OLD1")["end"] == pd.Timestamp("2023-01-10")
    assert _spell(rec, "OLD2")["end"] == pd.Timestamp("2015-05-04")
    assert _spell(rec, "OLD3")["end"] == pd.Timestamp("2005-06-15")
    assert _spell(rec, "OLD2")["security"] == "Old Two"


def test_the_rename_link_is_applied_only_when_the_date_settles_it(
    rec: sp500.Reconstruction,
) -> None:
    # META2's Date added (2015-05-04) has one addition row, FB2, not a current
    # member: linked. DDD's (2020-09-21) has no row at all: a gap.
    assert rec.rename_links == (("META2", "FB2", pd.Timestamp("2015-05-04")),)
    assert _spell(rec, "META2")["start"] == pd.Timestamp("2015-05-04")
    assert "FB2" not in set(rec.intervals["ticker"])
    assert rec.additions_without_row == (("DDD", pd.Timestamp("2020-09-21")),)
    assert rec.undated_members == ("EEE",)


def test_inconsistencies_are_counted_not_resolved(rec: sp500.Reconstruction) -> None:
    assert rec.inconsistencies == (
        sp500.Inconsistency(pd.Timestamp("2023-01-10"), "NEW1", "added_not_in_set"),
        sp500.Inconsistency(pd.Timestamp("2008-02-02"), "OLD2", "removed_already_in_set"),
    )


def test_gap_table_by_year(rec: sp500.Reconstruction, changes: pd.DataFrame) -> None:
    gaps = sp500.gap_table(rec, changes)
    assert {y: n for y, n in gaps.rows.items() if n} == {
        2005: 1,
        2008: 1,
        2010: 1,
        2015: 1,
        2023: 1,
    }
    assert {y: n for y, n in gaps.additions_without_row.items() if n} == {2020: 1}
    assert {y: n for y, n in gaps.inconsistencies.items() if n} == {2008: 1, 2023: 1}
    assert {y: n for y, n in gaps.rename_links.items() if n} == {2015: 1}
    assert min(gaps.rows) == 2005 and max(gaps.rows) == 2024


# ---------------------------------------------------------------------------
# The daily matrix
# ---------------------------------------------------------------------------


def test_member_on_is_half_open(rec: sp500.Reconstruction) -> None:
    dates = pd.DatetimeIndex(["2005-06-14", "2005-06-15", "2010-02-28", "2010-03-01"])
    member = sp500.member_on(rec.intervals, dates)
    assert member["OLD3"].tolist() == [True, False, False, False]
    assert member["BBB"].tolist() == [False, False, False, True]


def _sessions() -> pd.DatetimeIndex:
    """Business days 2005-06-01..2024-07-15, so the reach and snapshot are interior."""
    return pd.bdate_range("2005-06-01", "2024-07-15", name="date")


def test_matrix_is_daily_on_sessions_between_reach_and_snapshot(
    rec: sp500.Reconstruction,
) -> None:
    matrix = sp500.membership_matrix(rec, sessions=_sessions(), bars_present=None)
    # 2005-06-15 is a Wednesday and the reach; 2024-06-28 is the last business
    # day on or before the Sunday snapshot.
    assert matrix.index[0] == pd.Timestamp("2005-06-15")
    assert matrix.index[-1] == pd.Timestamp("2024-06-28")
    assert len(matrix) == len(pd.bdate_range("2005-06-15", "2024-06-28"))
    members = (matrix != sp500.NOT_MEMBER).sum(axis=1)
    # 2005-06-15: AAA BRK-B DDD EEE OLD1 OLD2 (OLD3 left that day, end exclusive) = 6
    assert members.loc["2005-06-15"] == 6
    # 2010-02-26 (Friday before BBB's addition): still 6; 2010-03-01: 7
    assert members.loc["2010-02-26"] == 6
    assert members.loc["2010-03-01"] == 7
    # 2015-05-04: OLD2 out, META2 (via FB2) in: 7; 2023-01-10: OLD1 out, NEW1 not
    # applied (inconsistency): 6
    assert members.loc["2015-05-04"] == 7
    assert members.loc["2023-01-10"] == 6
    assert members.loc["2024-06-28"] == 6
    # No bars consulted: every member is "member, no data", never "member".
    assert set(matrix.to_numpy().ravel().tolist()) == {sp500.NOT_MEMBER, sp500.MEMBER_NO_DATA}


def test_matrix_third_state_is_per_session(rec: sp500.Reconstruction) -> None:
    sessions = _sessions()
    present = pd.DataFrame(False, index=sessions, columns=pd.Index(["AAA", "BBB"]))
    present.loc["2012-12-31", "AAA"] = True
    matrix = sp500.membership_matrix(rec, sessions=sessions, bars_present=present)
    assert matrix.loc["2012-12-31", "AAA"] == sp500.MEMBER
    assert matrix.loc["2012-12-28", "AAA"] == sp500.MEMBER_NO_DATA  # the session before
    assert matrix.loc["2012-12-31", "BBB"] == sp500.MEMBER_NO_DATA
    assert matrix.loc["2005-06-15", "BBB"] == sp500.NOT_MEMBER


def test_matrix_refuses_an_unsorted_or_empty_calendar(rec: sp500.Reconstruction) -> None:
    with pytest.raises(sp500.SP500Error, match="sorted and unique"):
        sp500.membership_matrix(
            rec, sessions=pd.DatetimeIndex(["2010-01-05", "2010-01-04"]), bars_present=None
        )
    with pytest.raises(sp500.SP500Error, match="no session between"):
        sp500.membership_matrix(
            rec, sessions=pd.DatetimeIndex(["1999-01-04", "1999-01-05"]), bars_present=None
        )


# ---------------------------------------------------------------------------
# The committed files round-trip with their licence header
# ---------------------------------------------------------------------------


def _provenance(bars_pull: str | None) -> sp500.Provenance:
    return sp500.Provenance(
        constituents_url="https://example/constituents",
        changes_url="https://example/changes",
        licence="CC BY-SA 4.0",
        licence_url="https://creativecommons.org/licenses/by-sa/4.0/",
        snapshot=SNAPSHOT,
        pulled_at="2024-06-30T12:00:00Z",
        bars_pull=bars_pull,
        generated_at=datetime(2024, 7, 1, tzinfo=UTC),
        status_counts=(("empty", 2), ("failed", 1), ("ok", 9)) if bars_pull else (),
        failed_tickers=("ZZZ",) if bars_pull else (),
    )


def test_intervals_csv_round_trips(rec: sp500.Reconstruction, tmp_path: Path) -> None:
    text = sp500.render_intervals_csv(rec, _provenance(None))
    assert text.startswith("# S&P 500 membership intervals")
    assert "CC BY-SA 4.0" in text and "https://example/changes" in text
    assert "Data state" not in text
    path = tmp_path / "intervals.csv"
    path.write_text(text, encoding="utf-8")
    back = sp500.load_intervals(path)
    pd.testing.assert_frame_equal(
        back.reset_index(drop=True), rec.intervals.reset_index(drop=True), check_dtype=False
    )


def test_membership_csv_round_trips_and_cuts_at_end(
    rec: sp500.Reconstruction, tmp_path: Path
) -> None:
    matrix = sp500.membership_matrix(rec, sessions=_sessions(), bars_present=None)
    text = sp500.render_membership_csv(matrix, _provenance("2024-06-30T13:00:00Z"))
    assert "bars pull 2024-06-30T13:00:00Z" in text
    assert "empty 2, failed 1, ok 9" in text and "still failed after the retry: ZZZ" in text
    path = tmp_path / "membership.csv"
    path.write_text(text, encoding="utf-8")
    back = sp500.load_membership(path)
    pd.testing.assert_frame_equal(back, matrix, check_dtype=False, check_freq=False)
    cut = sp500.load_membership(path, end=pd.Timestamp("2010-03-01"))
    assert cut.index.max() == pd.Timestamp("2010-02-26")


# ---------------------------------------------------------------------------
# The committed reference table itself -- SPEC.md 15.3's acceptance
# ---------------------------------------------------------------------------


def _reference_path(name: str) -> Path:
    cfg = config.load()
    root = Path(__file__).resolve().parents[1]
    return root / cfg.model.equity_universe.reference_dir / name


def test_committed_membership_count_stays_inside_the_band_on_every_session() -> None:
    """Member count within ``membership_count_band`` on every session of the sample."""
    cfg = config.load()
    universe = cfg.model.equity_universe
    path = _reference_path(universe.membership_file)
    assert path.is_file(), f"{path} is the committed reference table and must exist"
    matrix = sp500.load_membership(path, end=cfg.require_holdout_start())
    start = pd.Timestamp(cfg.model.sample.start)
    window = matrix[matrix.index >= start]
    assert len(window) >= 250 * 17, "the sample window should hold at least 17 years of sessions"
    members = (window != sp500.NOT_MEMBER).sum(axis=1)
    low, high = universe.membership_count_band
    assert members.min() >= low, f"min {members.min()} at {members.idxmin().date()}"
    assert members.max() <= high, f"max {members.max()} at {members.idxmax().date()}"


def test_committed_files_carry_the_licence_header_and_the_disclaimer() -> None:
    cfg = config.load()
    universe = cfg.model.equity_universe
    for name in (universe.intervals_file, universe.membership_file):
        head = _reference_path(name).read_text(encoding="utf-8")[:3000]
        assert universe.licence in head and universe.licence_url in head
        assert universe.constituents_url in head and universe.changes_url in head
        assert "not a selection procedure this project ran" in head
        assert "Separate from the repository's code licence" in head


def test_committed_matrix_is_daily_and_its_intervals_reproduce_it() -> None:
    """The month-end reading is derived from the daily file, never stored beside it."""
    cfg = config.load()
    universe = cfg.model.equity_universe
    matrix = sp500.load_membership(_reference_path(universe.membership_file))
    intervals = sp500.load_intervals(_reference_path(universe.intervals_file))
    gaps = pd.Series(matrix.index).diff().dt.days.dropna()
    assert gaps.median() == 1.0, "a daily matrix has consecutive sessions a day apart"
    member = sp500.member_on(intervals, pd.DatetimeIndex(matrix.index))
    pd.testing.assert_frame_equal(
        (matrix != sp500.NOT_MEMBER), member.reindex(columns=matrix.columns), check_names=False
    )
