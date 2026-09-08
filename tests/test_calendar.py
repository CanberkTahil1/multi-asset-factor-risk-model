"""Calendar alignment. CLAUDE.md failure mode 1, the top source of fake Sharpes.

W1-P3 already found two calendar traps in the Fed's curve files -- an all-NA row
for every market holiday, and a Good Friday pairing that moved the ladder gate.
These tests assume there are more, so they pin the behaviour rather than the
implementation: what a month end IS, what alignment must never do, and exactly
where `.last()` and business-month-end are allowed to agree.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from mafrm.data import calendar

# Good Friday 2024 fell on 29 March, which was ALSO the last business day of the
# month. Business-month-end therefore points at a day the NYSE was shut -- the
# exact shape of the Good Friday trap W1-P3 hit in the Fed's curve files, where
# every field on that row was NA.
GOOD_FRIDAY_2024 = pd.Timestamp("2024-03-29")


def _sessions(start: str, end: str, drop: list[str] | None = None) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, end)
    if drop:
        days = days.drop([pd.Timestamp(d) for d in drop])
    return days


def _series(index: pd.DatetimeIndex, value: float = 1.0) -> pd.Series:
    return pd.Series(value, index=index, name="x")


# ---------------------------------------------------------------------------
# Month ends: hand-computed
# ---------------------------------------------------------------------------


def test_month_end_dates_hand_computed_for_a_known_quarter() -> None:
    """Q1 2021 business days, with Good Friday removed as a real market holiday.

    Hand-computed: January 2021's last business day is Friday the 29th, February's
    is Friday the 26th, March's is Wednesday the 31st. Three months, three dates.
    """
    sessions = _sessions("2021-01-01", "2021-03-31", drop=["2021-01-01"])
    picks = calendar.month_end_dates(sessions)

    assert list(picks) == [
        pd.Timestamp("2021-01-29"),
        pd.Timestamp("2021-02-26"),
        pd.Timestamp("2021-03-31"),
    ]


def test_monthly_produces_one_observation_per_calendar_month() -> None:
    """A known range: 2020-01-01..2020-12-31 holds exactly 12 month ends."""
    sessions = _sessions("2020-01-01", "2020-12-31")
    monthly = calendar.monthly(_series(sessions))

    assert len(monthly) == 12
    assert list(monthly.index.month) == list(range(1, 13))
    # Every stamp is a date that actually appears in the input calendar.
    assert set(monthly.index).issubset(set(sessions))


def test_month_end_count_matches_the_number_of_months_spanned() -> None:
    sessions = _sessions("2007-04-01", "2024-12-31")
    picks = calendar.month_end_dates(sessions)

    # 2007-04 through 2024-12 inclusive: 17 full years and 9 months.
    assert len(picks) == 17 * 12 + 9


# ---------------------------------------------------------------------------
# `.last()` vs business-month-end: where they agree, and where they must not
# ---------------------------------------------------------------------------


def test_month_end_agrees_with_resample_last_when_the_month_end_traded() -> None:
    """March 2021 ended on a Wednesday the market was open, so all three agree."""
    sessions = _sessions("2021-03-01", "2021-03-31")
    values = pd.Series(np.arange(len(sessions), dtype=float), index=sessions)

    ours = calendar.monthly(values)
    resampled = values.resample("ME").last()

    assert ours.iloc[-1] == resampled.iloc[-1]
    assert ours.index[-1] == pd.Timestamp("2021-03-31")


def test_month_end_follows_the_market_when_business_month_end_was_a_holiday() -> None:
    """The trap: `BME` for March 2024 is Good Friday, on which nothing traded.

    Business-month-end says 2024-03-29 and the market says 2024-03-28. Taking the
    calendar's word for it stamps the month on a day with no prices, and either
    carries the previous close forward or drops the month entirely, depending on
    which convenience the caller reached for. This asserts the divergence is real
    AND that we take the market's side.
    """
    sessions = _sessions("2024-03-01", "2024-03-31", drop=[str(GOOD_FRIDAY_2024.date())])
    values = pd.Series(np.arange(len(sessions), dtype=float), index=sessions)

    business_month_end = pd.bdate_range("2024-03-01", "2024-03-31", freq="BME")
    assert GOOD_FRIDAY_2024 in business_month_end  # the trap is real

    ours = calendar.month_end_dates(sessions)
    assert GOOD_FRIDAY_2024 not in ours
    assert ours[-1] == pd.Timestamp("2024-03-28")

    # `resample("ME").last()` survives here only because it ignores the label and
    # takes the last VALUE in the bucket; its stamp is still the calendar 31st, a
    # Sunday. Ours is stamped on a real session, which is what makes it joinable.
    resampled = values.resample("ME").last()
    assert resampled.index[-1] == pd.Timestamp("2024-03-31")
    assert values.loc[ours[-1]] == resampled.iloc[-1]


def test_period_end_stamping_is_presentation_only() -> None:
    sessions = _sessions("2021-02-01", "2021-02-28")
    values = _series(sessions)

    observation = calendar.monthly(values, stamp="observation")
    period_end = calendar.monthly(values, stamp="period_end")

    assert observation.index[-1] == pd.Timestamp("2021-02-26")  # a Friday
    assert period_end.index[-1] == pd.Timestamp("2021-02-28")  # the calendar 28th
    assert observation.to_numpy() == pytest.approx(period_end.to_numpy())


# ---------------------------------------------------------------------------
# Alignment must never invent a session
# ---------------------------------------------------------------------------


def test_align_never_contributes_a_value_on_a_date_the_asset_did_not_trade() -> None:
    """The single most important property in this module."""
    shared = _sessions("2021-06-01", "2021-06-30")
    closed = pd.Timestamp("2021-06-15")

    a = _series(shared, 1.0)
    b = _series(shared.drop([closed]), 2.0)

    panel = calendar.align({"a": a, "b": b}, how="union")

    assert closed in panel.index  # a traded, so the date survives the union
    assert panel.loc[closed, "a"] == 1.0
    assert np.isnan(panel.loc[closed, "b"])  # b did NOT trade: NaN, never 2.0


def test_align_intersection_keeps_only_common_sessions() -> None:
    shared = _sessions("2021-06-01", "2021-06-30")
    closed = pd.Timestamp("2021-06-15")

    panel = calendar.align(
        {"a": _series(shared), "b": _series(shared.drop([closed]))}, how="intersection"
    )

    assert closed not in panel.index
    assert len(panel) == len(shared) - 1
    assert not panel.isna().to_numpy().any()


def test_align_truncates_strictly_before_the_holdout() -> None:
    """CLAUDE.md invariant 5: the holdout begins ON holdout_start."""
    sessions = _sessions("2024-12-20", "2025-01-10")
    panel = calendar.align({"a": _series(sessions)}, end=date(2025, 1, 1))

    assert panel.index.max() < pd.Timestamp("2025-01-01")
    assert pd.Timestamp("2025-01-02") not in panel.index


def test_align_respects_start() -> None:
    sessions = _sessions("2020-01-01", "2020-12-31")
    panel = calendar.align({"a": _series(sessions)}, start=date(2020, 7, 1))
    assert panel.index.min() >= pd.Timestamp("2020-07-01")


# ---------------------------------------------------------------------------
# All-NA rows are not sessions -- the W1-P3 Fed-file trap
# ---------------------------------------------------------------------------


def test_all_na_rows_are_not_trading_days() -> None:
    """The Fed's curve files carry a row per weekday with every field NA."""
    sessions = _sessions("2021-03-01", "2021-03-31")
    frame = pd.DataFrame({"y2": 1.0, "y10": 2.0}, index=sessions)
    frame.loc[pd.Timestamp("2021-03-31")] = [np.nan, np.nan]

    dates = calendar.trading_dates(frame)

    assert pd.Timestamp("2021-03-31") not in dates
    # ...and so the month end moves back to the last day that really traded.
    assert calendar.month_end_dates(dates)[-1] == pd.Timestamp("2021-03-30")


def test_a_partially_populated_row_is_still_a_session() -> None:
    sessions = _sessions("2021-03-01", "2021-03-31")
    frame = pd.DataFrame({"y2": 1.0, "y10": 2.0}, index=sessions)
    frame.loc[pd.Timestamp("2021-03-31"), "y10"] = np.nan

    assert pd.Timestamp("2021-03-31") in calendar.trading_dates(frame)


# ---------------------------------------------------------------------------
# Loud failure on the defects that would otherwise be normalised away
# ---------------------------------------------------------------------------


def test_duplicate_dates_are_rejected() -> None:
    index = pd.DatetimeIndex(["2021-01-04", "2021-01-05", "2021-01-05"])
    with pytest.raises(calendar.CalendarError, match="duplicated"):
        calendar.trading_dates(_series(index))


def test_unsorted_index_is_rejected_rather_than_sorted() -> None:
    index = pd.DatetimeIndex(["2021-01-05", "2021-01-04"])
    with pytest.raises(calendar.CalendarError, match="not sorted"):
        calendar.trading_dates(_series(index))


def test_timezone_aware_index_is_rejected() -> None:
    index = pd.DatetimeIndex(["2021-01-04", "2021-01-05"], tz="America/New_York")
    with pytest.raises(calendar.CalendarError, match="timezone-aware"):
        calendar.trading_dates(_series(index))


def test_nat_in_the_index_is_rejected() -> None:
    index = pd.DatetimeIndex(["2021-01-04", None])
    with pytest.raises(calendar.CalendarError, match="NaT"):
        calendar.trading_dates(_series(index))


def test_align_rejects_an_empty_mapping() -> None:
    with pytest.raises(calendar.CalendarError, match="no series"):
        calendar.align({})


def test_align_rejects_a_series_with_no_traded_dates() -> None:
    sessions = _sessions("2021-01-04", "2021-01-08")
    empty = pd.Series(np.nan, index=sessions)
    with pytest.raises(calendar.CalendarError, match="no traded dates"):
        calendar.align({"a": _series(sessions), "b": empty})


def test_month_end_dates_of_an_empty_index_is_empty() -> None:
    assert len(calendar.month_end_dates(pd.DatetimeIndex([]))) == 0
