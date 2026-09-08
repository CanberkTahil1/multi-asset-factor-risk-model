"""FRED/ALFRED parsing, chunk stitching and vintage coverage. SPEC.md 3.5.

Offline: every payload here is a literal shaped like a real API response.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pandas as pd
import pytest

from mafrm.data import fred


def observations(*rows: tuple[str, str, str, str]) -> dict[str, Any]:
    return {
        "observations": [
            {"date": d, "value": v, "realtime_start": rs, "realtime_end": re}
            for d, v, rs, re in rows
        ]
    }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_observations_parse_to_a_dated_frame() -> None:
    payload = observations(
        ("2019-02-04", "112.3238", "2019-02-11", "2019-02-18"),
        ("2019-02-05", "112.4000", "2019-02-11", "2019-02-18"),
    )
    frame = fred.parse_observations(json.dumps(payload), "DTWEXBGS")
    assert list(frame.columns) == ["value", "realtime_start", "realtime_end"]
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.name == "date"
    assert float(frame["value"].iloc[0]) == pytest.approx(112.3238)
    assert frame["realtime_start"].iloc[0] == pd.Timestamp("2019-02-11")
    assert frame["realtime_end"].iloc[0] == "2019-02-18"


def test_the_still_current_sentinel_survives_as_text() -> None:
    """FRED writes 9999-12-31 for "still current", outside pandas' Timestamp range.

    Coercing it to NaT loses the distinction between "still current" and
    "missing" on 1,151 of DGS1MO's 5,524 initial releases, with nothing raised.
    """
    payload = observations(("2026-08-27", "3.81", "2026-08-28", "9999-12-31"))
    frame = fred.parse_observations(json.dumps(payload), "DGS1MO")
    assert frame["realtime_end"].iloc[0] == "9999-12-31"
    assert frame["realtime_end"].notna().all()


def test_a_missing_observation_is_a_period_and_becomes_nan() -> None:
    payload = observations(("2020-01-01", ".", "2020-01-02", "9999-12-31"))
    frame = fred.parse_observations(json.dumps(payload), "DGS1MO")
    assert pd.isna(frame["value"].iloc[0])


def test_observations_are_sorted_by_date() -> None:
    payload = observations(
        ("2020-01-03", "2.0", "2020-01-04", "9999-12-31"),
        ("2020-01-02", "1.0", "2020-01-03", "9999-12-31"),
    )
    frame = fred.parse_observations(json.dumps(payload), "X")
    assert list(frame["value"]) == [1.0, 2.0]


def test_a_fred_error_payload_raises_rather_than_parsing_to_nothing() -> None:
    payload = {"error_code": 400, "error_message": "Bad Request. Something."}
    with pytest.raises(fred.FredError, match="Bad Request"):
        fred.parse_observations(json.dumps(payload), "X")


def test_a_row_missing_a_documented_key_raises() -> None:
    payload = {"observations": [{"date": "2020-01-02", "value": "1.0"}]}
    with pytest.raises(fred.FredError, match="realtime_start"):
        fred.parse_observations(json.dumps(payload), "X")


def test_an_empty_observation_array_raises() -> None:
    with pytest.raises(fred.FredError, match="empty"):
        fred.parse_observations(json.dumps({"observations": []}), "X")


def test_vintage_dates_parse() -> None:
    payload = {"vintage_dates": ["2019-02-04", "2019-02-11"]}
    assert fred.parse_vintage_dates(json.dumps(payload), "X") == (
        date(2019, 2, 4),
        date(2019, 2, 11),
    )


# ---------------------------------------------------------------------------
# Chunk stitching -- the part that would silently mis-date the series
# ---------------------------------------------------------------------------


def frame_of(rows: list[tuple[str, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": [v for _, v, _ in rows],
            "realtime_start": pd.to_datetime([rs for _, _, rs in rows]),
            "realtime_end": ["9999-12-31"] * len(rows),
        },
        index=pd.DatetimeIndex(pd.to_datetime([d for d, _, _ in rows]), name="date"),
    )


def test_stitching_keeps_the_earliest_release_of_a_repeated_observation() -> None:
    """The whole reason chunking needs a stitcher rather than a concat.

    ``output_type=4`` reports "initial release" relative to the *requested*
    window, so an observation first published in window 1 is reported again at
    the start of window 2. Keeping the later row would date a 2005 value to 2013.
    """
    window_one = frame_of([("2005-01-03", 2.20, "2005-01-04")])
    window_two = frame_of([("2005-01-03", 2.25, "2013-06-01"), ("2013-06-03", 0.02, "2013-06-04")])

    stitched = fred.first_releases([window_one, window_two])

    assert len(stitched) == 2
    assert stitched.loc[pd.Timestamp("2005-01-03"), "realtime_start"] == pd.Timestamp("2005-01-04")
    assert float(stitched.loc[pd.Timestamp("2005-01-03"), "value"]) == pytest.approx(2.20)


def test_stitching_returns_a_sorted_unique_index() -> None:
    stitched = fred.first_releases(
        [
            frame_of([("2020-01-05", 1.0, "2020-01-06")]),
            frame_of([("2020-01-02", 2.0, "2020-01-03")]),
        ]
    )
    assert list(stitched.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-05")]
    assert not stitched.index.has_duplicates


def test_stitching_nothing_raises() -> None:
    with pytest.raises(fred.FredError, match="no observation windows"):
        fred.first_releases([])


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_coverage_counts_only_observations_that_have_a_first_release() -> None:
    current = frame_of(
        [
            ("2006-01-02", 1.0, "2026-08-29"),
            ("2019-02-04", 2.0, "2026-08-29"),
            ("2019-02-05", 3.0, "2026-08-29"),
        ]
    )
    first_release = frame_of([("2019-02-04", 2.0, "2019-02-11"), ("2019-02-05", 3.0, "2019-02-11")])
    report = fred.coverage(
        series_id="DTWEXBGS",
        vintage_dates=[date(2019, 2, 4), date(2019, 2, 11)],
        current=current,
        first_release=first_release,
    )
    assert report.n_observations == 3
    assert report.n_first_release == 2
    assert report.fraction == pytest.approx(2 / 3)
    assert report.has_vintages
    assert not report.is_complete, "partial coverage owes the conservative-lag decision"
    assert report.observation_start == date(2006, 1, 2)


def test_full_coverage_is_complete() -> None:
    current = frame_of([("2020-01-02", 1.0, "2026-08-29")])
    report = fred.coverage(
        series_id="X",
        vintage_dates=[date(2020, 1, 3)],
        current=current,
        first_release=frame_of([("2020-01-02", 1.0, "2020-01-03")]),
    )
    assert report.is_complete
    assert report.fraction == pytest.approx(1.0)


def test_a_series_with_no_vintages_raises_on_demand() -> None:
    report = fred.coverage(
        series_id="X",
        vintage_dates=[],
        current=frame_of([("2020-01-02", 1.0, "2026-08-29")]),
        first_release=None,
    )
    assert not report.has_vintages
    assert "NO VINTAGES" in report.render()
    with pytest.raises(fred.NoVintagesError, match="look-ahead"):
        report.require_vintages()


def test_cache_names_distinguish_the_two_releases() -> None:
    assert fred.cache_name("DTWEXBGS") == "dtwexbgs"
    assert fred.cache_name("DTWEXBGS", "first_release") == "dtwexbgs_first_release"
