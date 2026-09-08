"""Total returns, split verification and dated back-adjustment. SPEC.md 3.5.

The adjustment tests carry hand-computed expected values, because the whole
claim of this module is that the adjustment is ours and checkable rather than
the vendor's and opaque.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from mafrm.data import prices

DATES = pd.DatetimeIndex(
    pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]), name="date"
)


def price_frame(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": closes, "Volume": [1e6] * len(closes)}, index=DATES[: len(closes)]
    )


def action_frame(dividends: list[float], splits: list[float] | None = None) -> pd.DataFrame:
    n = len(dividends)
    return pd.DataFrame(
        {"Dividends": dividends, "Stock Splits": splits if splits is not None else [0.0] * n},
        index=DATES[:n],
    )


# ---------------------------------------------------------------------------
# Total return
# ---------------------------------------------------------------------------


def test_total_return_adds_the_dividend_on_its_ex_date() -> None:
    prices_frame = price_frame([100.0, 110.0, 108.0, 112.0])
    actions = action_frame([0.0, 0.0, 2.0, 0.0])

    returns = prices.total_return(prices_frame, actions)

    # Hand-computed: (110+0)/100-1, (108+2)/110-1, (112+0)/108-1.
    assert list(returns.index) == list(DATES[1:])
    assert float(returns.iloc[0]) == pytest.approx(0.10)
    assert float(returns.iloc[1]) == pytest.approx(0.0)
    assert float(returns.iloc[2]) == pytest.approx(112.0 / 108.0 - 1.0)


def test_total_return_does_not_look_forward() -> None:
    """Appending tomorrow's dividend must not change yesterday's number.

    This is the property the whole raw-cache design exists to buy (SPEC.md 3.5).
    """
    short_prices = price_frame([100.0, 110.0, 108.0])
    short_actions = action_frame([0.0, 0.0, 0.0])
    long_prices = price_frame([100.0, 110.0, 108.0, 112.0])
    long_actions = action_frame([0.0, 0.0, 0.0, 5.0])

    early = prices.total_return(short_prices, short_actions)
    late = prices.total_return(long_prices, long_actions)

    assert early.to_numpy() == pytest.approx(late.iloc[: len(early)].to_numpy(), abs=0.0)


def test_a_missing_close_column_raises() -> None:
    frame = pd.DataFrame({"Open": [1.0]}, index=DATES[:1])
    with pytest.raises(prices.PricesError, match="no 'Close' column"):
        prices.total_return(frame, action_frame([0.0]))


# ---------------------------------------------------------------------------
# Splits -- verified, not assumed
# ---------------------------------------------------------------------------


def test_a_split_already_reflected_in_the_close_is_accepted() -> None:
    """IWM's real 2005-06-09 2:1: the close went 61.715 -> 62.320, continuous."""
    prices_frame = price_frame([61.715, 62.320])
    actions = action_frame([0.0, 0.0], splits=[0.0, 2.0])

    checks = prices.check_splits_applied(prices_frame, actions)

    assert len(checks) == 1
    assert checks[0].already_applied
    assert checks[0].ratio == 2.0
    # log(62.320/61.715) = 0.00976, against -log(2) = -0.6931 if unadjusted.
    assert checks[0].log_return == pytest.approx(math.log(62.320 / 61.715), rel=1e-12)
    assert checks[0].expected_log_return_if_unadjusted == pytest.approx(-math.log(2.0))


def test_a_split_NOT_reflected_in_the_close_raises() -> None:
    """The failure mode the check exists for: applying a factor twice."""
    prices_frame = price_frame([100.0, 50.5])
    actions = action_frame([0.0, 0.0], splits=[0.0, 2.0])

    with pytest.raises(prices.PricesError) as caught:
        prices.check_splits_applied(prices_frame, actions)
    assert "NOT APPLIED" in str(caught.value)
    assert "2020-01-03" in str(caught.value)


def test_no_splits_is_trivially_consistent() -> None:
    assert prices.check_splits_applied(price_frame([1.0, 2.0]), action_frame([0.0, 0.0])) == ()


def test_total_return_runs_the_split_check() -> None:
    with pytest.raises(prices.PricesError):
        prices.total_return(price_frame([100.0, 50.5]), action_frame([0.0, 0.0], splits=[0.0, 2.0]))


def test_a_split_on_the_first_bar_has_no_predecessor_and_is_skipped() -> None:
    checks = prices.check_splits_applied(
        price_frame([100.0, 101.0]), action_frame([0.0, 0.0], splits=[3.0, 0.0])
    )
    assert checks == ()


# ---------------------------------------------------------------------------
# Back-adjustment, as of a stated date
# ---------------------------------------------------------------------------


def test_adjustment_factors_are_hand_computable() -> None:
    """closes 100, 110, 108, 112 with a 2.00 dividend ex on the third bar.

    ratio at the ex-date is ``1 - 2/110`` -- the fraction of the cum-dividend
    price that stayed in the share. ``f_t`` is the product of the ratios strictly
    AFTER ``t``, so the last two bars are unadjusted at 1.0 and the first two
    carry ``108/110``.
    """
    prices_frame = price_frame([100.0, 110.0, 108.0, 112.0])
    actions = action_frame([0.0, 0.0, 2.0, 0.0])

    factor = prices.adjustment_factors(prices_frame, actions, as_of=date(2020, 1, 7))

    expected = 108.0 / 110.0
    assert factor.to_numpy() == pytest.approx([expected, expected, 1.0, 1.0], abs=1e-15)


def test_adjusted_close_at_the_ex_date_equals_the_price_less_the_dividend() -> None:
    prices_frame = price_frame([100.0, 110.0, 108.0, 112.0])
    actions = action_frame([0.0, 0.0, 2.0, 0.0])

    adjusted = prices.adjusted_close(prices_frame, actions, as_of=date(2020, 1, 7))

    # 110 * (1 - 2/110) = 110 - 2 = 108, exactly.
    assert float(adjusted.close.iloc[1]) == pytest.approx(108.0, abs=1e-12)
    assert float(adjusted.close.iloc[0]) == pytest.approx(100.0 * 108.0 / 110.0, abs=1e-12)
    assert float(adjusted.close.iloc[-1]) == pytest.approx(112.0), "today's price is untouched"
    assert adjusted.as_of == date(2020, 1, 7)
    assert adjusted.n_dividends == 1


def test_the_as_of_date_bounds_which_dividends_are_folded_in() -> None:
    """A dividend after ``as_of`` is invisible, which is what makes it dated."""
    prices_frame = price_frame([100.0, 110.0, 108.0, 112.0])
    actions = action_frame([0.0, 0.0, 2.0, 0.0])

    before = prices.adjusted_close(prices_frame, actions, as_of=date(2020, 1, 3))
    after = prices.adjusted_close(prices_frame, actions, as_of=date(2020, 1, 7))

    assert before.n_dividends == 0
    assert before.factor.to_numpy() == pytest.approx(np.ones(4))
    assert before.close.to_numpy() == pytest.approx(prices_frame["Close"].to_numpy())
    assert float(after.close.iloc[0]) != pytest.approx(float(before.close.iloc[0]))


def test_adjusted_and_raw_returns_agree_across_the_ex_date() -> None:
    """The consistency the two paths owe each other.

    Back-adjusted price returns and raw total returns are different objects, but
    on a fully-adjusted series they must produce the same daily return -- if they
    do not, one of the two treatments of the dividend is wrong.
    """
    prices_frame = price_frame([100.0, 110.0, 108.0, 112.0])
    actions = action_frame([0.0, 0.0, 2.0, 0.0])

    raw = prices.total_return(prices_frame, actions)
    adjusted = prices.adjusted_close(prices_frame, actions, as_of=date(2020, 1, 7)).close
    from_adjusted = (adjusted / adjusted.shift(1) - 1.0).iloc[1:]

    assert raw.to_numpy() == pytest.approx(from_adjusted.to_numpy(), rel=1e-12)


def test_an_as_of_before_the_first_bar_raises() -> None:
    with pytest.raises(prices.PricesError, match="precedes the first bar"):
        prices.adjustment_factors(
            price_frame([100.0, 110.0]), action_frame([0.0, 0.0]), as_of=date(2019, 1, 1)
        )
