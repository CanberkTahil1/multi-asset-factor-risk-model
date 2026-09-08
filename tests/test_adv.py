"""Average daily dollar volume. SPEC.md 3.4 -- a median, and why it must be."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mafrm.costs import adv


def _bars(volume: list[float], close: list[float]) -> pd.DataFrame:
    index = pd.bdate_range("2021-01-04", periods=len(volume))
    return pd.DataFrame({"Volume": volume, "Close": close}, index=index)


def test_dollar_volume_is_volume_times_close_hand_computed() -> None:
    bars = _bars([100.0, 200.0, 50.0], [10.0, 11.0, 12.0])
    turnover = adv.dollar_volume(bars)

    assert list(turnover) == [1000.0, 2200.0, 600.0]


def test_rolling_median_hand_computed() -> None:
    """Five days, window 3. Every window is computed by hand below.

    dollar volume: 100, 300, 200, 900, 400
      window 1 (100, 300, 200) -> sorted 100, 200, 300 -> median 200
      window 2 (300, 200, 900) -> sorted 200, 300, 900 -> median 300
      window 3 (200, 900, 400) -> sorted 200, 400, 900 -> median 400
    """
    bars = _bars([100.0, 300.0, 200.0, 900.0, 400.0], [1.0] * 5)
    result = adv.average_daily_volume(bars, window=3)

    assert np.isnan(result.iloc[0]) and np.isnan(result.iloc[1])
    assert list(result.iloc[2:]) == [200.0, 300.0, 400.0]


def test_the_median_ignores_a_rebalance_day_and_the_mean_does_not() -> None:
    """SPEC.md 3.4's stated reason for the median, as an executable claim.

    Sixty-two ordinary days at 100 and one index-rebalance print at 5,000 -- a
    fiftyfold day, which is not unusual around a reconstitution. The median is
    unmoved at exactly 100. The mean is dragged to 177.8, a 78% overstatement of
    depth, and it overstates in the flattering direction: a deeper book makes
    every impact estimate cheaper.
    """
    volume = [100.0] * 62 + [5000.0]
    bars = _bars(volume, [1.0] * 63)

    median = adv.average_daily_volume(bars, window=63).iloc[-1]
    mean = adv.dollar_volume(bars).rolling(63).mean().iloc[-1]

    assert median == 100.0
    assert mean == pytest.approx((62 * 100.0 + 5000.0) / 63)
    assert mean == pytest.approx(177.78, abs=0.01)
    assert mean / median > 1.75


def test_window_is_counted_in_rows_not_calendar_days() -> None:
    """A holiday must not shorten the sample -- rows are sessions."""
    index = pd.DatetimeIndex(["2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28", "2024-04-01"])
    bars = pd.DataFrame({"Volume": [1.0, 2.0, 3.0, 4.0, 5.0], "Close": [1.0] * 5}, index=index)

    result = adv.average_daily_volume(bars, window=3)

    # The 4-day Easter gap between 03-28 and 04-01 is invisible: the last window
    # is still the last three ROWS (3, 4, 5), median 4.
    assert result.iloc[-1] == 4.0


def test_missing_columns_are_rejected() -> None:
    frame = pd.DataFrame({"Close": [1.0]}, index=pd.bdate_range("2021-01-04", periods=1))
    with pytest.raises(adv.AdvError, match="Volume"):
        adv.dollar_volume(frame)


def test_negative_volume_is_rejected_as_a_parse_defect() -> None:
    bars = _bars([100.0, -5.0], [1.0, 1.0])
    with pytest.raises(adv.AdvError, match="negative share volume"):
        adv.dollar_volume(bars)


def test_non_positive_close_is_rejected() -> None:
    bars = _bars([100.0, 100.0], [1.0, 0.0])
    with pytest.raises(adv.AdvError, match="non-positive close"):
        adv.dollar_volume(bars)


def test_non_positive_window_is_rejected() -> None:
    bars = _bars([1.0], [1.0])
    with pytest.raises(adv.AdvError, match="window must be positive"):
        adv.average_daily_volume(bars, window=0)


def test_partial_windows_produce_nothing_by_default() -> None:
    bars = _bars([1.0] * 10, [1.0] * 10)
    result = adv.average_daily_volume(bars, window=63)
    assert result.isna().all()
