"""EDGE effective spreads. SPEC.md 3.4.

The estimator itself is the authors' (``bidask``, MIT), so these tests do not
re-derive it. They pin the three things that are ours and are easy to get wrong:
the window semantics, the refusal to accept adjusted prices, and the units.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from bidask import edge

from mafrm.costs import spread

WINDOW = 63


def _simulate(
    *,
    days: int = 400,
    ticks: int = 390,
    true_spread: float = 0.0,
    daily_vol: float = 0.01,
    seed: int = 20260830,
    start: str = "2019-01-01",
) -> pd.DataFrame:
    """Raw-looking OHLCV bars with a KNOWN effective spread.

    A driftless random walk observed ``ticks`` times a day, with each trade
    placed half a spread either side of the mid at random. Open, high, low and
    close are formed from the traded prices exactly as an exchange would, so the
    high-low range carries the bid-ask bounce the estimator is built to remove.
    """
    rng = np.random.default_rng(seed)
    step = daily_vol / np.sqrt(ticks)
    log_mid = np.cumsum(rng.normal(0.0, step, size=days * ticks)) + np.log(100.0)
    mid = np.exp(log_mid).reshape(days, ticks)
    signs = rng.choice([-1.0, 1.0], size=(days, ticks))
    trades = mid * (1.0 + signs * true_spread / 2.0)
    index = pd.bdate_range(start, periods=days)
    return pd.DataFrame(
        {
            "Open": trades[:, 0],
            "High": trades.max(axis=1),
            "Low": trades.min(axis=1),
            "Close": trades[:, -1],
            "Volume": rng.integers(1_000, 10_000, size=days).astype(float),
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# Units -- hand computed
# ---------------------------------------------------------------------------


def test_to_basis_points_is_a_pure_unit_change() -> None:
    """EDGE returns a proportion where 0.01 is 1%. 0.0005 is therefore 5bp."""
    series = pd.Series([0.0005, 0.0001, 0.01, 0.0])
    assert list(spread.to_basis_points(series)) == [5.0, 1.0, 100.0, 0.0]


# ---------------------------------------------------------------------------
# Window semantics: the monthly step is a SAMPLE of a rolling estimator
# ---------------------------------------------------------------------------


def test_monthly_estimate_equals_edge_on_exactly_the_trailing_window() -> None:
    """The claim in the module docstring, checked rather than asserted.

    Computing EDGE daily and sampling at month ends must give the identical
    number to calling the authors' `edge` on exactly the trailing 63 rows. If
    this drifts, the "rolling 63-day window, monthly step" of SPEC.md 3.4 has
    quietly become something else.
    """
    bars = _simulate(true_spread=0.002)
    monthly = spread.monthly_effective_spread(bars, window=WINDOW, min_periods=WINDOW, signed=True)

    stamp = monthly.dropna().index[3]
    position = bars.index.get_loc(stamp)
    assert isinstance(position, int)
    window = bars.iloc[position - WINDOW + 1 : position + 1]
    assert len(window) == WINDOW

    direct = edge(window["Open"], window["High"], window["Low"], window["Close"])
    assert monthly.loc[stamp] == pytest.approx(direct, rel=1e-12)


def test_monthly_stamps_land_on_the_last_traded_day_of_each_month() -> None:
    bars = _simulate(days=200)
    monthly = spread.monthly_effective_spread(bars, window=WINDOW, min_periods=WINDOW, signed=True)

    assert set(monthly.index).issubset(set(bars.index))
    assert monthly.index.is_unique
    assert list(monthly.index) == sorted(monthly.index)
    # One stamp per month present, no more.
    assert len(monthly) == len({(d.year, d.month) for d in monthly.index})


def test_series_begins_exactly_one_full_window_in() -> None:
    """`require_full_window` is a policy, and this is what it buys."""
    bars = _simulate(days=200)
    daily = spread.effective_spread(bars, window=WINDOW, min_periods=WINDOW, signed=True)

    assert daily.iloc[: WINDOW - 1].isna().all()
    assert not np.isnan(daily.iloc[WINDOW - 1])


# ---------------------------------------------------------------------------
# The estimator recovers a known spread when it has enough data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("true_bp", [20, 50, 100])
def test_edge_recovers_a_known_spread_on_a_long_sample(true_bp: int) -> None:
    """A synthetic market whose spread we set, so the answer is known.

    Tolerance is deliberately loose and one-directional information: this is a
    check that the wiring feeds O/H/L/C in the right order and the right units,
    not a re-test of the published estimator. See the small-window bias test
    below for why it is stated on a LONG sample.
    """
    bars = _simulate(days=3000, true_spread=true_bp * 1e-4)
    estimate = edge(bars["Open"], bars["High"], bars["Low"], bars["Close"])

    assert estimate * 1e4 == pytest.approx(true_bp, rel=0.10)


def test_the_positive_bias_on_a_zero_spread_market_is_the_absolute_value() -> None:
    """The W1-P5 correction, pinned so the misdiagnosis cannot come back.

    W1-P5 measured a large positive EDGE estimate on a market with no spread at
    all, found that it scaled with volatility and that a longer window did not
    escape it, and concluded that EDGE carries a finite-sample bias. The cause
    was `bidask`'s `sign=False` default, which returns |estimate|. The estimator
    itself is very nearly unbiased here; its absolute value cannot be, at any
    window and any volatility, because |X| >= 0 by construction.

    This test states both halves, because the first without the second is the
    claim that was wrong.
    """
    quiet = spread.zero_spread_reference(
        daily_volatility=0.01, window=WINDOW, seed=20260830, days=800
    )
    loud = spread.zero_spread_reference(
        daily_volatility=0.03, window=WINDOW, seed=20260830, days=800
    )

    # The estimator is centred: signed bias is tiny beside the absolute value it
    # was mistaken for, and it goes negative about half the time.
    for bias in (quiet, loud):
        assert abs(bias.signed_mean) < 0.15 * bias.absolute_mean
        assert 0.35 < bias.negative_fraction < 0.65

    # The absolute value reproduces the refuted finding exactly: positive, and
    # scaling roughly one-for-one with volatility.
    assert quiet.absolute_mean > 0.0
    assert loud.absolute_mean > 2.0 * quiet.absolute_mean
    assert 2e-4 < quiet.absolute_mean < 2e-3


def test_clipping_leaves_exactly_half_the_absolute_value_bias() -> None:
    """Hand-computed. Why the fix narrows the defect rather than removing it.

    For a zero-centred X, E|X| = 2*E[max(X,0)] -- the negative half of the
    distribution contributes its magnitude to the first and zero to the second,
    and by symmetry the two halves are equal. So resetting negatives to zero
    must leave exactly half of the absolute-value bias, whatever the shape of
    the distribution, as long as it is centred. Under normality the two are
    sd*sqrt(2/pi) = 0.7979*sd and sd/sqrt(2*pi) = 0.3989*sd.

    Checked first on a series whose answer is arithmetic, then on the simulation.
    """
    # E|X| = (3 + 1 + 1 + 3)/4 = 2.0; E[max(X,0)] = (3 + 1 + 0 + 0)/4 = 1.0.
    symmetric = pd.Series([3.0, 1.0, -1.0, -3.0])
    clipped = spread.clip_negative(symmetric)
    assert symmetric.abs().mean() == pytest.approx(2.0)
    assert clipped.mean() == pytest.approx(1.0)
    assert symmetric.abs().mean() == pytest.approx(2.0 * clipped.mean())

    bias = spread.zero_spread_reference(
        daily_volatility=0.02, window=WINDOW, seed=20260830, days=800
    )
    assert bias.clipped_mean == pytest.approx(bias.absolute_mean / 2.0, rel=0.12)
    assert bias.clipped_mean > 0.0


def test_an_overnight_gap_inflates_the_estimators_noise_but_not_its_centre() -> None:
    """A hypothesis about the residual level, pinned as REFUTED. experiments.md row 58.

    EDGE's price process has no close-to-open jump; real index ETFs do, and the
    obvious guess is that a gap the estimator cannot attribute to the session is
    read as spread. It is not. On a market whose true spread is exactly zero, a
    gap worth 40% of daily variance roughly triples the estimator's DISPERSION
    and leaves its centre at zero.

    That is worth pinning for two reasons. It explains the SIZE of the old
    absolute-value bias, which was 0.798*sd and so grew with anything that grew
    the dispersion. And it removes the leading candidate for the level that
    survives the sign fix on the most liquid names, which is therefore still
    unexplained -- see experiments.md rows 57-58.

    Stated on a long sample deliberately. The windows overlap 62 days in 63, so
    at 1,500 days the signed mean carries several basis points of Monte Carlo
    error and briefly looks like a real positive bias. It is not; it converges.
    """
    kwargs = {"daily_volatility": 0.008, "window": WINDOW, "seed": 20260830, "days": 6000}
    continuous = spread.zero_spread_reference(**kwargs)  # type: ignore[arg-type]
    gapped = spread.zero_spread_reference(overnight_gap_share=0.4, **kwargs)  # type: ignore[arg-type]

    # The centre does not move: both are within a basis point or two of zero.
    assert abs(continuous.signed_mean) < 2e-4
    assert abs(gapped.signed_mean) < 2e-4

    # The dispersion does, and the absolute value tracks it one-for-one. This is
    # the whole mechanism of the defect in one assertion.
    assert gapped.absolute_mean > 2.5 * continuous.absolute_mean
    assert 0.4 < gapped.negative_fraction < 0.6


# ---------------------------------------------------------------------------
# Recovery: the null passing says nothing about whether a spread can be RESOLVED
# ---------------------------------------------------------------------------


SPY_VOL = 0.0081  # SPY's measured daily volatility, 2023-2024.
SPY_GAP = 0.39  # SPY's measured overnight share of daily variance, same window.


def test_edge_has_no_positive_resolution_floor_at_a_63_day_window() -> None:
    """The decisive negative result. experiments.md rows 59-60.

    The worry the recovery test was built to settle: that at 63 days EDGE
    returns some fixed positive number -- around 20bp -- whatever the truth is,
    which would make it the wrong instrument outright. It does not. On a market
    whose true spread is 1bp it returns a median near ZERO and goes negative
    about half the time, which is what an estimator with no floor looks like.
    """
    tight = spread.recovery_reference(
        true_spread=1e-4,
        daily_volatility=SPY_VOL,
        window=WINDOW,
        seed=20260830,
        windows=300,
        overnight_gap_share=SPY_GAP,
    )

    assert abs(tight.median) < 5e-4
    assert 0.4 < tight.negative_fraction < 0.6


def test_edge_recovers_a_wide_spread_at_the_same_window() -> None:
    """The other half. A floor-free estimator that recovered nothing would be
    just as useless, and this is what rules out our own implementation."""
    wide = spread.recovery_reference(
        true_spread=50e-4,
        daily_volatility=SPY_VOL,
        window=WINDOW,
        seed=20260830,
        windows=300,
        overnight_gap_share=SPY_GAP,
    )

    assert wide.median == pytest.approx(50e-4, rel=0.12)
    assert wide.mean == pytest.approx(50e-4, rel=0.15)
    assert wide.negative_fraction < 0.05
    assert wide.signal_to_noise > 1.0


def test_the_binding_constraint_is_per_window_noise_not_bias() -> None:
    """What actually stops EDGE delivering a level for a liquid ETF.

    The median is near-unbiased at every true spread; the per-window standard
    deviation is 20bp+ under SPY-like conditions and does not shrink with more
    windows, because it is a property of one estimate rather than of the mean.
    A signal-to-noise ratio below 1 means a single 63-day window cannot tell
    this spread from zero, and that is the whole finding.
    """
    kwargs = {
        "daily_volatility": SPY_VOL,
        "window": WINDOW,
        "seed": 20260830,
        "windows": 300,
        "overnight_gap_share": SPY_GAP,
    }
    tight = spread.recovery_reference(true_spread=1e-4, **kwargs)  # type: ignore[arg-type]
    middling = spread.recovery_reference(true_spread=20e-4, **kwargs)  # type: ignore[arg-type]

    # The median tracks the truth at 20bp even though a single window cannot.
    assert middling.median == pytest.approx(20e-4, rel=0.25)
    assert middling.signal_to_noise < 1.5
    assert tight.signal_to_noise < 0.2
    # Noise, not bias: the dispersion barely moves between the two.
    assert middling.standard_deviation == pytest.approx(tight.standard_deviation, rel=0.35)


def test_recovery_reference_rejects_bad_inputs() -> None:
    kwargs = {"daily_volatility": 0.01, "window": WINDOW, "seed": 1, "windows": 10}
    with pytest.raises(spread.SpreadError, match="non-negative"):
        spread.recovery_reference(true_spread=-1e-4, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(spread.SpreadError, match="at least 3"):
        spread.recovery_reference(true_spread=1e-4, daily_volatility=0.01, window=2, seed=1)
    with pytest.raises(spread.SpreadError, match="at least 2 windows"):
        spread.recovery_reference(
            true_spread=1e-4, daily_volatility=0.01, window=WINDOW, seed=1, windows=1
        )


def test_an_out_of_range_gap_share_is_rejected() -> None:
    with pytest.raises(spread.SpreadError, match="share of daily variance"):
        spread.zero_spread_reference(
            daily_volatility=0.01, window=WINDOW, seed=1, days=400, overnight_gap_share=1.0
        )


def test_zero_spread_reference_is_reproducible_from_its_seed() -> None:
    kwargs = {"daily_volatility": 0.02, "window": WINDOW, "days": 400}
    assert spread.zero_spread_reference(seed=1, **kwargs) == spread.zero_spread_reference(
        seed=1, **kwargs
    )
    assert spread.zero_spread_reference(seed=1, **kwargs) != spread.zero_spread_reference(
        seed=2, **kwargs
    )


# ---------------------------------------------------------------------------
# Raw bars only -- the adjusted-close trap of SPEC.md 3.5
# ---------------------------------------------------------------------------


def test_close_outside_the_high_low_range_is_rejected() -> None:
    """The signature of adjusted and unadjusted columns mixed in one frame."""
    bars = _simulate(days=100)
    bars.iloc[10, bars.columns.get_loc("Close")] = bars["High"].iloc[10] * 1.5

    with pytest.raises(spread.SpreadError, match="outside the high-low range"):
        spread.effective_spread(bars, window=WINDOW, min_periods=WINDOW, signed=True)


def test_high_below_low_is_rejected() -> None:
    bars = _simulate(days=100)
    high = bars.columns.get_loc("High")
    low = bars.columns.get_loc("Low")
    bars.iloc[5, high], bars.iloc[5, low] = bars.iloc[5, low], bars.iloc[5, high]

    with pytest.raises(spread.SpreadError, match="High < Low"):
        spread.effective_spread(bars, window=WINDOW, min_periods=WINDOW, signed=True)


def test_non_positive_price_is_rejected_because_edge_takes_logs() -> None:
    bars = _simulate(days=100)
    bars.iloc[7, bars.columns.get_loc("Low")] = 0.0

    with pytest.raises(spread.SpreadError, match="non-positive price"):
        spread.effective_spread(bars, window=WINDOW, min_periods=WINDOW, signed=True)


def test_missing_ohlc_columns_are_rejected() -> None:
    bars = _simulate(days=100).drop(columns=["Open"])
    with pytest.raises(spread.SpreadError, match="missing"):
        spread.effective_spread(bars, window=WINDOW, min_periods=WINDOW, signed=True)


def test_min_periods_below_the_published_minimum_is_rejected() -> None:
    bars = _simulate(days=100)
    with pytest.raises(spread.SpreadError, match="undefined below 3"):
        spread.effective_spread(bars, window=WINDOW, min_periods=2, signed=True)


def test_min_periods_above_the_window_is_rejected() -> None:
    bars = _simulate(days=100)
    with pytest.raises(spread.SpreadError, match="exceeds window"):
        spread.effective_spread(bars, window=10, min_periods=20, signed=True)


# ---------------------------------------------------------------------------
# Negative estimates are counted BEFORE the clip, and clipped exactly once
# ---------------------------------------------------------------------------


def test_negative_estimates_are_reported_not_silently_floored() -> None:
    series = pd.Series([0.001, -0.0002, 0.0005, np.nan, -0.0001])
    report = spread.count_negative(series)

    assert report.total == 4
    assert report.negative == 2
    assert report.most_negative == pytest.approx(-0.0002)
    assert report.fraction == pytest.approx(0.5)
    assert "50.00%" in report.render()


def test_count_negative_on_an_all_missing_series() -> None:
    report = spread.count_negative(pd.Series([np.nan, np.nan]))
    assert report.total == 0
    assert report.fraction == 0.0
    assert report.render() == "no estimates"


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------


def _panel(**kwargs: object) -> spread.SpreadPanel:
    long_asset = _simulate(days=400, start="2023-06-01", seed=1)
    short_asset = _simulate(days=150, start="2024-04-01", seed=2)
    defaults: dict[str, object] = {
        "window": WINDOW,
        "min_periods": WINDOW,
        "signed": True,
        "clip": True,
        "end": date(2025, 1, 1),
    }
    defaults.update(kwargs)
    return spread.monthly_spread_panel(
        {"LONG": long_asset, "SHORT": short_asset},
        **defaults,  # type: ignore[arg-type]
    )


def test_panel_aligns_assets_without_filling_and_stops_before_the_holdout() -> None:
    panel = _panel()

    assert list(panel.production.columns) == ["LONG", "SHORT"]
    assert panel.production.index.max() < pd.Timestamp("2025-01-01")
    # SHORT has no estimate before its own history plus a full window: NaN, not
    # a value carried backwards from nowhere.
    assert panel.production["SHORT"].isna().any()
    assert panel.production["LONG"].notna().sum() > panel.production["SHORT"].notna().sum()


def test_panel_clips_the_production_series_and_leaves_the_signed_one_alone() -> None:
    """The two frames are the same estimates under two different policies.

    They must agree everywhere the estimate was non-negative and differ only
    where it was, and the negative count must describe the SIGNED frame -- the
    clipped one has no negatives to count by construction.
    """
    panel = _panel()

    unchanged = panel.signed >= 0
    assert panel.production.where(unchanged).equals(panel.signed.where(unchanged))
    was_negative = (panel.signed < 0).to_numpy()
    assert (panel.production.to_numpy()[was_negative] == 0.0).all()
    assert spread.count_negative(panel.production.stack()).negative == 0

    for asset, report in panel.negatives.items():
        assert report.negative == int((panel.signed[asset].dropna() < 0).sum())
    # The simulation has no spread at all, so the estimator should be going
    # negative often. If this is ever zero, sign=True stopped being passed.
    assert sum(r.negative for r in panel.negatives.values()) > 0


def test_panel_can_skip_the_clip_for_an_averaging_diagnostic() -> None:
    panel = _panel(clip=False)
    assert panel.production.equals(panel.signed)
    assert (panel.signed.min() < 0).any()


def test_panel_rejects_an_empty_mapping() -> None:
    with pytest.raises(spread.SpreadError, match="no assets"):
        spread.monthly_spread_panel({}, window=WINDOW, min_periods=WINDOW, signed=True, clip=True)


def test_unsigned_estimates_can_never_be_negative() -> None:
    """The defect itself, pinned. `sign=False` returns |estimate|.

    This is what made `count_negative` report 0.0% on all eight sleeve assets in
    W1-P5 and read as evidence that the estimator was well behaved. On a market
    with no spread at all the signed series must go negative and the unsigned
    one must not.
    """
    bars = _simulate(days=400, true_spread=0.0)
    unsigned = spread.effective_spread(bars, window=WINDOW, min_periods=WINDOW, signed=False)
    signed = spread.effective_spread(bars, window=WINDOW, min_periods=WINDOW, signed=True)

    assert spread.count_negative(unsigned).negative == 0
    assert spread.count_negative(signed).negative > 0
    assert unsigned.dropna().equals(signed.dropna().abs())
