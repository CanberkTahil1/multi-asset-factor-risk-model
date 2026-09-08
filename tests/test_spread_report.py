"""The crisis-widening statistic behind reports/spread_vs_vol.png. SPEC.md 3.4.

`build()` reads the cache and is exercised by `make report`, not here. What is
worth pinning is the statistic the report table publishes, because it is the one
W1-P5 row 52 reported and the one the absolute-value correction broke.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from mafrm import config
from mafrm.costs import spread_report


def _window(start: str, end: str, label: str = "TEST") -> config.CrisisWindow:
    return config.CrisisWindow(
        label=label, start=date.fromisoformat(start), end=date.fromisoformat(end)
    )


def test_crisis_multiple_is_hand_computable() -> None:
    """Median of the 12 months before, against the peak monthly median inside.

    Two assets, so the cross-sectional median of each month is the mean of the
    pair. The six pre-window months have medians 10, 10, 20, 20, 30, 30, whose
    own median is 20. Inside the window the medians are 40, 100, 60, so the peak
    is 100. The statistic is therefore (20, 100), a 5x widening.
    """
    index = pd.to_datetime(
        [
            "2007-07-31",
            "2007-08-31",
            "2007-09-28",
            "2007-10-31",
            "2007-11-30",
            "2007-12-31",
            "2008-01-31",
            "2008-02-29",
            "2008-03-31",
        ]
    )
    frame = pd.DataFrame(
        {
            "A": [5.0, 5.0, 15.0, 15.0, 25.0, 25.0, 35.0, 95.0, 55.0],
            "B": [15.0, 15.0, 25.0, 25.0, 35.0, 35.0, 45.0, 105.0, 65.0],
        },
        index=index,
    )

    before, peak = spread_report._crisis_multiple(frame, _window("2008-01-01", "2008-06-30"))

    assert before == pytest.approx(20.0)
    assert peak == pytest.approx(100.0)
    assert peak / before == pytest.approx(5.0)


def test_a_negative_baseline_does_not_produce_a_negative_multiple() -> None:
    """The 2022 window after the correction: the baseline goes below zero.

    A ratio against it is not a statement about spreads -- it reports -26x on
    the real sleeve. The table must refuse to print one, and the guard is the
    configured floor rather than a sign test, because a baseline of +0.8bp is
    just as unusable as one of -1.6bp.
    """
    cfg = config.load()
    floor = cfg.model.costs.spread_vs_volatility.minimum_baseline_for_ratio_bps
    assert floor > 0.0

    index = pd.to_datetime(["2021-06-30", "2021-12-31", "2022-03-31", "2022-06-30"])
    frame = pd.DataFrame({"A": [-2.0, -1.0, 40.0, 30.0]}, index=index)

    before, peak = spread_report._crisis_multiple(frame, _window("2022-01-01", "2022-12-31"))
    assert before < 0.0
    assert before < floor
    assert peak == pytest.approx(40.0)
    # The widening is still perfectly well defined in basis points.
    assert peak - before == pytest.approx(41.5)
