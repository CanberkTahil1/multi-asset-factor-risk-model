"""Data-contract primitives and the daily-bar guard. SPEC.md 3.5.

Offline. Every fixture is built in the test, so these run on a clean clone with
no cache and no network.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from mafrm import secrets
from mafrm.data import contracts


def dated(values: dict[str, list[object]], dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(values, index=pd.DatetimeIndex(pd.to_datetime(dates), name="date"))


# ---------------------------------------------------------------------------
# CLAUDE.md invariant 2 -- the acceptance criterion for W1-P4
# ---------------------------------------------------------------------------


def test_daily_is_the_only_accepted_interval() -> None:
    assert contracts.require_daily_interval("1d") == "1d"


@pytest.mark.parametrize("interval", ["1wk", "1mo", "5d", "3mo", "1h", "", "1D"])
def test_non_daily_intervals_raise(interval: str) -> None:
    with pytest.raises(contracts.IntervalError):
        contracts.require_daily_interval(interval)


def test_the_interval_error_names_the_reason_not_just_the_rule() -> None:
    with pytest.raises(contracts.IntervalError) as caught:
        contracts.require_daily_interval("1wk")
    message = str(caught.value)
    assert "1273" in message, "the yfinance issue number is the citation for the rule"
    assert "aggregate yourself" in message


# ---------------------------------------------------------------------------
# Column set (the `Adj Close` trap)
# ---------------------------------------------------------------------------


def test_missing_column_raises_and_names_what_is_missing() -> None:
    frame = dated({"Open": [1.0], "Close": [2.0]}, ["2020-01-02"])
    with pytest.raises(contracts.ContractError) as caught:
        contracts.check_columns(frame, ["Open", "Adj Close", "Close"], label="yf/SPY")
    assert "Adj Close" in str(caught.value)
    assert "Open" not in str(caught.value).split("got")[0]


def test_extra_columns_are_allowed() -> None:
    frame = dated({"Open": [1.0], "Close": [2.0], "Capital Gains": [0.0]}, ["2020-01-02"])
    contracts.check_columns(frame, ["Open", "Close"], label="yf/SPY")


# ---------------------------------------------------------------------------
# All-NaN columns (the renamed-field trap)
# ---------------------------------------------------------------------------


def test_all_nan_column_raises() -> None:
    frame = dated({"a": [1.0, 2.0], "b": [None, None]}, ["2020-01-02", "2020-01-03"])
    with pytest.raises(contracts.ContractError, match="entirely NaN"):
        contracts.check_no_all_nan_columns(frame, label="src")


def test_a_partially_nan_column_is_fine() -> None:
    frame = dated({"a": [1.0, None]}, ["2020-01-02", "2020-01-03"])
    contracts.check_no_all_nan_columns(frame, label="src")


def test_an_expected_empty_column_can_be_allowed_explicitly() -> None:
    frame = dated({"a": [1.0], "b": [None]}, ["2020-01-02"])
    contracts.check_no_all_nan_columns(frame, label="src", allow=["b"])


# ---------------------------------------------------------------------------
# Duplicate dates (the calendar trap)
# ---------------------------------------------------------------------------


def test_duplicate_index_dates_raise() -> None:
    frame = dated({"a": [1.0, 2.0]}, ["2020-01-02", "2020-01-02"])
    with pytest.raises(contracts.ContractError, match="share a date"):
        contracts.check_no_duplicate_dates(frame, label="src")


def test_duplicate_dates_are_found_in_a_date_column_too() -> None:
    frame = pd.DataFrame({"date": ["2020-01-02", "2020-01-02"], "a": [1.0, 2.0]})
    with pytest.raises(contracts.ContractError, match="share a date"):
        contracts.check_no_duplicate_dates(frame, label="src")


def test_a_frame_with_no_date_axis_fails_check_frame() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(contracts.ContractError, match="no date axis"):
        contracts.check_frame(frame, label="src")


def test_check_frame_accepts_a_well_formed_table() -> None:
    frame = dated({"a": [1.0, 2.0]}, ["2020-01-02", "2020-01-03"])
    contracts.check_frame(frame, label="src", required_columns=["a"], min_rows=2)


def test_check_frame_enforces_a_minimum_row_count() -> None:
    frame = dated({"a": [1.0]}, ["2020-01-02"])
    with pytest.raises(contracts.ContractError, match="expected at least"):
        contracts.check_frame(frame, label="src", min_rows=2)


# ---------------------------------------------------------------------------
# Coverage -- replaces the W1-P4 recency contract
# ---------------------------------------------------------------------------


def monthly(periods: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"a": range(len(periods))},
        index=pd.DatetimeIndex(pd.to_datetime(periods), name="date"),
    )


def test_a_complete_monthly_series_has_no_missing_periods() -> None:
    frame = monthly(["2020-01-31", "2020-02-29", "2020-03-31"])
    assert len(contracts.missing_periods(frame, freq="M", label="x")) == 0


def test_an_interior_gap_is_found() -> None:
    """The failure a row count cannot see: a month dropped out of the middle."""
    frame = monthly(["2020-01-31", "2020-03-31", "2020-04-30"])
    gaps = contracts.missing_periods(frame, freq="M", label="x")
    assert len(gaps) == 1
    assert str(gaps[0]) == "2020-02"


def test_month_end_dates_that_are_not_the_last_calendar_day_still_count() -> None:
    """AQR stamps its rows on the last BUSINESS day, which varies by month."""
    frame = monthly(["1877-02-28", "1877-03-30", "1877-04-30"])
    assert len(contracts.missing_periods(frame, freq="M", label="x")) == 0


def test_coverage_requires_the_series_to_start_before_the_window() -> None:
    frame = monthly(["2020-01-31", "2020-02-29"])
    with pytest.raises(contracts.ContractError, match="after the in-sample window opens"):
        contracts.check_covers(frame, label="x", freq="M", start=date(2019, 1, 1))


def test_a_month_end_stamp_covers_the_month_the_window_opens_in() -> None:
    """The trap a raw date comparison would fall into.

    AQR stamps monthly rows on the last business day. A file whose first row is
    2007-04-30 covers April 2007 in full, and `sample.start` is 2007-04-01;
    comparing the dates directly would reject it, and the obvious repair -- a few
    weeks of slack -- would be a tolerance invented to hide a units mismatch.
    """
    frame = monthly(["2007-04-30", "2007-05-31"])
    contracts.check_covers(frame, label="x", freq="M", start=date(2007, 4, 1))


def test_coverage_requires_the_series_to_reach_the_end_of_the_window() -> None:
    frame = monthly(["2020-01-31", "2020-02-29"])
    with pytest.raises(contracts.ContractError, match="before the in-sample window closes"):
        contracts.check_covers(
            frame, label="x", freq="M", start=date(2020, 1, 1), end=date(2021, 1, 1)
        )


def test_an_unset_end_leaves_the_upper_bound_unenforced() -> None:
    """`sample.holdout_start` is null until a human pins it (invariant 5).

    The start-and-gaps half of the contract still applies; only the upper bound
    waits. Passing `None` must not silently pass the whole contract.
    """
    complete = monthly(["2020-01-31", "2020-02-29"])
    contracts.check_covers(complete, label="x", freq="M", start=date(2020, 1, 1), end=None)

    gappy = monthly(["2020-01-31", "2020-03-31"])
    with pytest.raises(contracts.ContractError, match="interior"):
        contracts.check_covers(gappy, label="x", freq="M", start=date(2020, 1, 1), end=None)


def test_a_gappy_series_fails_coverage_even_when_the_bounds_are_fine() -> None:
    frame = monthly(["2020-01-31", "2020-03-31", "2020-04-30"])
    with pytest.raises(contracts.ContractError, match="interior"):
        contracts.check_covers(
            frame, label="x", freq="M", start=date(2020, 1, 1), end=date(2020, 4, 1)
        )


# ---------------------------------------------------------------------------
# Release drift -- cadence inferred, never assumed
# ---------------------------------------------------------------------------


def drift(seen: list[str], today: str) -> contracts.ReleaseDrift:
    return contracts.ReleaseDrift(
        label="src/x",
        releases=tuple(
            contracts.Release(first_seen=date.fromisoformat(s), last_observation=None) for s in seen
        ),
        today=date.fromisoformat(today),
    )


def test_a_serializer_rewrite_is_not_a_release() -> None:
    """The detector must not fire on this project's own dependency upgrades.

    Upgrading pandas or pyarrow rewrites every cached file under a new digest and
    a new pull date while the data is untouched. Counting those as releases would
    shorten the inferred cadence until the detector fired on a library bump --
    a false positive manufactured entirely by us.
    """
    entries = [
        contracts.Release(first_seen=date(2026, 1, 1), last_observation=date(2025, 12, 31)),
        # Same end date, re-serialized two months later under a new pyarrow.
        contracts.Release(first_seen=date(2026, 3, 1), last_observation=date(2025, 12, 31)),
    ]
    collapsed = contracts.collapse_releases(entries)
    assert len(collapsed) == 1
    assert collapsed[0].first_seen == date(2026, 1, 1), "the earliest sighting is the release"


def test_collapsing_keeps_genuinely_distinct_releases() -> None:
    entries = [
        contracts.Release(first_seen=date(2026, 1, 1), last_observation=date(2025, 12, 31)),
        contracts.Release(first_seen=date(2026, 2, 1), last_observation=date(2026, 1, 31)),
    ]
    assert len(contracts.collapse_releases(entries)) == 2


def test_an_entry_with_no_end_date_is_not_a_release() -> None:
    entries = [contracts.Release(first_seen=date(2026, 1, 1), last_observation=None)]
    assert contracts.collapse_releases(entries) == ()


def test_one_release_cannot_fire_however_old_it_is() -> None:
    """The property that makes this an inference and not an age limit.

    With a single observed release there is no interval to compare against, so
    the detector stays silent no matter how long ago it was. That is the honest
    state on a first pull, and it is exactly what the W1-P4 age check got wrong.
    """
    report = drift(["2020-01-01"], today="2030-01-01")
    assert not report.cadence_observable
    assert not report.fired
    assert "not yet observable" in report.render()


def test_zero_releases_cannot_fire() -> None:
    report = contracts.ReleaseDrift(label="src/x", releases=(), today=date(2026, 1, 1))
    assert not report.fired
    assert "never cached" in report.render()


def test_a_gap_within_the_observed_cadence_is_quiet() -> None:
    """Hand-computed: releases 30 and 20 days apart, 25 days since the last."""
    report = drift(["2026-01-01", "2026-01-31", "2026-02-20"], today="2026-03-17")
    assert report.observed_intervals == (30, 20)
    assert report.longest_observed_interval == 30
    assert report.days_since_last_release == 25
    assert not report.fired


def test_a_gap_longer_than_anything_observed_fires() -> None:
    """31 days since the last release, against a longest observed gap of 30."""
    report = drift(["2026-01-01", "2026-01-31", "2026-02-20"], today="2026-03-23")
    assert report.days_since_last_release == 31
    assert report.longest_observed_interval == 30
    assert report.fired
    assert "DRIFT" in report.render()


def test_the_boundary_is_strict_rather_than_inclusive() -> None:
    """Exactly as long as the longest observed gap is not yet evidence."""
    report = drift(["2026-01-01", "2026-01-31"], today="2026-03-02")
    assert report.days_since_last_release == 30
    assert report.longest_observed_interval == 30
    assert not report.fired


def test_the_last_observation_comes_from_the_newest_release() -> None:
    report = contracts.ReleaseDrift(
        label="src/x",
        releases=(
            contracts.Release(first_seen=date(2026, 1, 1), last_observation=date(2025, 12, 31)),
            contracts.Release(first_seen=date(2026, 2, 1), last_observation=date(2026, 1, 31)),
        ),
        today=date(2026, 2, 15),
    )
    assert report.last_observation == date(2026, 1, 31)


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


def test_env_file_parsing_skips_comments_and_strips_quotes() -> None:
    text = "\n".join(
        [
            "# a comment",
            "",
            "PLAIN=value",
            "  SPACED  =  padded  ",
            'QUOTED="has spaces"',
            "SINGLE='tick'",
            "EMPTY=",
            "not a pair",
        ]
    )
    values = secrets.parse_env_file(text)
    assert values == {
        "PLAIN": "value",
        "SPACED": "padded",
        "QUOTED": "has spaces",
        "SINGLE": "tick",
        "EMPTY": "",
    }


def test_the_environment_beats_the_env_file(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / "env"  # type: ignore[operator]
    env_file.write_text("MAFRM_TEST_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("MAFRM_ENV_FILE", str(env_file))
    monkeypatch.setenv("MAFRM_TEST_KEY", "from_environment")
    assert secrets.require("MAFRM_TEST_KEY") == "from_environment"


def test_the_env_file_is_the_fallback(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "env"  # type: ignore[operator]
    env_file.write_text("MAFRM_TEST_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("MAFRM_ENV_FILE", str(env_file))
    monkeypatch.delenv("MAFRM_TEST_KEY", raising=False)
    assert secrets.require("MAFRM_TEST_KEY") == "from_file"


def test_a_missing_secret_raises_and_says_where_to_put_it(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAFRM_ENV_FILE", str(tmp_path / "absent"))  # type: ignore[operator]
    monkeypatch.delenv("MAFRM_ABSENT_KEY", raising=False)
    with pytest.raises(secrets.MissingSecretError) as caught:
        secrets.require("MAFRM_ABSENT_KEY", hint="get one at example.com")
    message = str(caught.value)
    assert "MAFRM_ABSENT_KEY" in message
    assert "get one at example.com" in message


def test_an_empty_secret_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAFRM_ENV_FILE", "/nonexistent/env")
    monkeypatch.setenv("MAFRM_BLANK_KEY", "   ")
    with pytest.raises(secrets.MissingSecretError):
        secrets.require("MAFRM_BLANK_KEY")
