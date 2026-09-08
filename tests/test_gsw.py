"""Constant-maturity Treasury return tests. SPEC.md 3.2.

Every expected value here is hand-computed or analytically derived from the
Svensson form, never read back from the implementation. The four that matter are
the n -> 0 limit, the flat-curve identity, the sign and ordering of the roll-down,
and finiteness across the real 1961- sample.

No network: the parser runs against a fixture string, and the sample-wide tests
read the local cache and skip when it is empty.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mafrm.config import load
from mafrm.data import cache, gsw
from mafrm.data.gsw import (
    GswError,
    breakeven_inflation,
    constant_maturity_return,
    curve_yields,
    parse_fed_curve_csv,
    parse_fred_csv,
    published_mask,
    svensson_yield,
)

DELTA = 1.0 / 252.0
SMALL_N = 1e-6

# A file in the Fed's shape: multi-row preamble of a length you must not assume,
# a Nelson-Siegel row with the TAU2 sentinel, a market holiday with every field
# blank, and NA blocks where the published tenor does not yet exist.
FIXTURE_CSV = """"Note: This is not an official Federal Reserve statistical release."

Series,Compounding Convention,Mnemonic(s)
Zero-coupon yield,Continuously Compounded,SVENYXX
Parameters,N/A,BETA0 to TAU2

Date,BETA0,BETA1,BETA2,BETA3,SVENY02,SVENY10,SVENY30,TAU1,TAU2
1979-12-28,9.0,-1.0,0.5,0,10.10,10.50,NA,2.5,-999.99
1979-12-31,9.1,-1.1,0.4,0,10.20,10.60,NA,2.5,-999.99
1980-01-01,NA,NA,NA,NA,NA,NA,NA,NA,NA
1980-01-02,9.2,-1.2,0.3,0.2,10.30,10.70,10.90,2.5,7.5
1980-01-03,9.3,-1.3,0.2,0.1,10.40,10.80,11.00,2.5,7.5
"""


def flat_params(level: float = 5.0, dates: int = 3) -> pd.DataFrame:
    """A perfectly flat curve: y(n) = level for every n."""
    index = pd.to_datetime([f"2020-01-{day + 1:02d}" for day in range(dates)])
    index.name = "date"
    return pd.DataFrame(
        {
            "BETA0": level,
            "BETA1": 0.0,
            "BETA2": 0.0,
            "BETA3": 0.0,
            "TAU1": 2.0,
            "TAU2": 5.0,
            "SVENY02": level,
            "SVENY10": level,
            "SVENY30": level,
            "TIPSY02": level,
            "TIPSY10": level,
        },
        index=index,
    )


def sloped_params(beta0: float, beta1: float, tau1: float, dates: int = 3) -> pd.DataFrame:
    """A static Nelson-Siegel curve, unchanged from day to day.

    ``y(n) = beta0 + beta1.(1-exp(-n/tau1))/(n/tau1)``, which rises with ``n``
    whenever ``beta1 < 0``. Holding it fixed across dates isolates the roll:
    the duration term is identically zero, so total return is carry plus roll.
    """
    index = pd.to_datetime([f"2020-01-{day + 1:02d}" for day in range(dates)])
    index.name = "date"
    return pd.DataFrame(
        {
            "BETA0": beta0,
            "BETA1": beta1,
            "BETA2": 0.0,
            "BETA3": 0.0,
            "TAU1": tau1,
            "TAU2": np.nan,
            "SVENY02": 1.0,
            "SVENY30": 1.0,
        },
        index=index,
    )


def analytic_yield(beta0: float, beta1: float, tau1: float, n: float, beta2: float = 0.0) -> float:
    """The Nelson-Siegel yield, written out independently of the module."""
    u = n / tau1
    slope = (1.0 - math.exp(-u)) / u
    return beta0 + beta1 * slope + beta2 * (slope - math.exp(-u))


# ---------------------------------------------------------------------------
# 1. The n -> 0 limit
# ---------------------------------------------------------------------------


def test_svensson_yield_at_tiny_maturity_is_beta0_plus_beta1() -> None:
    # As n -> 0 both (1-exp(-n/tau))/(n/tau) and exp(-n/tau) tend to 1, so the
    # beta2 and beta3 terms vanish and y(0) = beta0 + beta1 exactly.
    y = svensson_yield(3.0, -1.5, 2.0, 0.5, 2.0, 10.0, 1e-9, small_n=SMALL_N)
    assert float(y) == pytest.approx(3.0 - 1.5, abs=1e-9)


def test_svensson_yield_is_continuous_across_the_small_n_threshold() -> None:
    # Just above the cutoff the true expression is evaluated; it must agree with
    # the limit, or the guard would introduce a discontinuity of its own.
    below = float(svensson_yield(3.0, -1.5, 2.0, 0.5, 2.0, 10.0, 1e-7, small_n=SMALL_N))
    above = float(svensson_yield(3.0, -1.5, 2.0, 0.5, 2.0, 10.0, 1e-5, small_n=SMALL_N))
    assert below == pytest.approx(above, abs=1e-5)
    assert np.isfinite(above)


def test_svensson_yield_matches_a_hand_written_nelson_siegel() -> None:
    # beta2 = beta3 = 0 reduces Svensson to Nelson-Siegel, computed here in
    # plain Python from the published formula.
    got = float(svensson_yield(5.0, -3.0, 0.0, 0.0, 3.0, 7.0, 10.0, small_n=SMALL_N))
    assert got == pytest.approx(analytic_yield(5.0, -3.0, 3.0, 10.0), rel=1e-12)


def test_pre_1980_sentinel_drops_the_fourth_term_instead_of_poisoning_it() -> None:
    # TAU2 = -999.99 with BETA3 = 0 is the Nelson-Siegel regime. exp(-n/-999.99)
    # overflows to inf and 0*inf is nan, which would delete 27% of the sample.
    sentinel = float(svensson_yield(9.0, -1.0, 0.5, 0.0, 2.5, -999.99, 10.0, small_n=SMALL_N))
    dropped = float(svensson_yield(9.0, -1.0, 0.5, 0.0, 2.5, np.nan, 10.0, small_n=SMALL_N))
    assert np.isfinite(sentinel)
    assert sentinel == pytest.approx(dropped, rel=1e-12)
    assert sentinel == pytest.approx(analytic_yield(9.0, -1.0, 2.5, 10.0, beta2=0.5), rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Flat curve: pure accrual, zero roll
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2.0, 30.0])
def test_flat_curve_returns_exactly_the_yield_accrual_with_zero_roll(n: float) -> None:
    level = 5.0
    frame = constant_maturity_return(
        n, params=flat_params(level), delta=DELTA, mask_unpublished=False
    )

    # log(1+r) = [n.y - (n-D).y]/100 = D.y/100 for any n, since y(n) = y(n-D).
    expected_log = DELTA * level / 100.0
    assert expected_log == pytest.approx(5.0 / 252.0 / 100.0, rel=1e-15)

    assert frame["roll_down"].abs().max() == 0.0
    assert frame["duration_effect"].abs().max() == pytest.approx(0.0, abs=1e-18)
    assert frame["carry"].to_numpy() == pytest.approx(expected_log, rel=1e-15)
    assert frame["total_return"].to_numpy() == pytest.approx(math.expm1(expected_log), rel=1e-15)
    # Hand value: x = 5/(100*252) = 1.98412698e-4; expm1(x) = x + x^2/2 + ...
    #           = 1.98412698e-4 + 1.96838e-8 = 1.98432382e-4
    assert float(frame["total_return"].iloc[0]) == pytest.approx(1.98432382e-4, rel=1e-8)


def test_components_reconstruct_the_total_return_exactly() -> None:
    # carry + roll + duration == log(1+r) by construction. If the n-Delta
    # evaluation is ever dropped this identity is the first thing to break.
    frame = constant_maturity_return(
        10.0, params=sloped_params(6.0, -4.0, 40.0), delta=DELTA, mask_unpublished=False
    )
    rebuilt = frame[["carry", "roll_down", "duration_effect"]].sum(axis=1)
    assert rebuilt.to_numpy() == pytest.approx(
        np.log1p(frame["total_return"].to_numpy()), rel=1e-14
    )


# ---------------------------------------------------------------------------
# 3. Roll-down: strictly positive, and the maturity ordering
# ---------------------------------------------------------------------------


def _static_roll(beta0: float, beta1: float, tau1: float, n: float) -> float:
    """Analytic roll-down on a static curve: (n-D).[y(n) - y(n-D)]/100."""
    return (
        (n - DELTA)
        * (analytic_yield(beta0, beta1, tau1, n) - analytic_yield(beta0, beta1, tau1, n - DELTA))
        / 100.0
    )


def test_upward_sloping_static_curve_rolls_down_positively_and_more_at_30y() -> None:
    # A near-linear upward slope: tau1 = 40 keeps dy/dn from collapsing before
    # 30y, so the (n-D) multiplier dominates and the long point rolls harder.
    beta0, beta1, tau1 = 6.0, -4.0, 40.0
    params = sloped_params(beta0, beta1, tau1)
    assert analytic_yield(beta0, beta1, tau1, 30.0) > analytic_yield(beta0, beta1, tau1, 2.0)

    two = constant_maturity_return(2.0, params=params, delta=DELTA, mask_unpublished=False)
    thirty = constant_maturity_return(30.0, params=params, delta=DELTA, mask_unpublished=False)

    assert (two["roll_down"] > 0).all()
    assert (thirty["roll_down"] > 0).all()
    assert float(thirty["roll_down"].iloc[0]) > float(two["roll_down"].iloc[0])

    # Hand-checked against the analytic expression, not against the module.
    assert float(two["roll_down"].iloc[0]) == pytest.approx(
        _static_roll(beta0, beta1, tau1, 2.0), rel=1e-12
    )
    assert float(thirty["roll_down"].iloc[0]) == pytest.approx(
        _static_roll(beta0, beta1, tau1, 30.0), rel=1e-12
    )

    # The static curve leaves no duration term, so the whole return is carry+roll.
    assert thirty["duration_effect"].abs().max() == pytest.approx(0.0, abs=1e-15)


def test_the_30y_over_2y_roll_ordering_is_a_curve_property_not_a_law() -> None:
    """Documents the counter-case, so nobody later treats the ordering as an invariant.

    With a short tau1 the slope is concentrated in the front end: dy/dn at 2y can
    exceed dy/dn at 30y by more than the 15x maturity ratio, and the 2y point
    then rolls harder. Both curves are upward-sloping and both rolls are positive.
    """
    beta0, beta1, tau1 = 5.0, -3.0, 3.0
    params = sloped_params(beta0, beta1, tau1)
    two = constant_maturity_return(2.0, params=params, delta=DELTA, mask_unpublished=False)
    thirty = constant_maturity_return(30.0, params=params, delta=DELTA, mask_unpublished=False)

    assert analytic_yield(beta0, beta1, tau1, 30.0) > analytic_yield(beta0, beta1, tau1, 2.0)
    assert float(two["roll_down"].iloc[0]) > 0
    assert float(thirty["roll_down"].iloc[0]) > 0
    assert float(two["roll_down"].iloc[0]) > float(thirty["roll_down"].iloc[0])


def test_evaluating_the_next_day_at_n_instead_of_n_minus_delta_deletes_the_roll() -> None:
    """The classic error of SPEC.md 3.2, point 1, reproduced and measured.

    On a static upward-sloping curve the correct return is carry + roll; reading
    day t+1 at n leaves carry alone. The gap is the roll, and at 30y it is
    basis points a day -- about 0.9% a year here, all of it lost.
    """
    beta0, beta1, tau1 = 6.0, -4.0, 40.0
    params = sloped_params(beta0, beta1, tau1)
    correct = constant_maturity_return(30.0, params=params, delta=DELTA, mask_unpublished=False)

    # The error: price the aged bond at (n - Delta) but read the yield at n.
    y30 = curve_yields(params, 30.0)
    wrong_log = (30.0 * y30.shift(1) - (30.0 - DELTA) * y30) / 100.0
    wrong = np.expm1(wrong_log.iloc[1:])

    # Exact in log space: the two differ by (n-Delta).[y(n) - y(n-Delta)]/100,
    # which is the roll term itself. In simple returns the gap carries a
    # second-order expm1 term, so it matches only to ~1e-8.
    lost_log = float(np.log1p(correct["total_return"].iloc[0])) - float(wrong_log.iloc[1])
    assert lost_log == pytest.approx(float(correct["roll_down"].iloc[0]), rel=1e-13)

    lost = float(correct["total_return"].iloc[0]) - float(wrong.iloc[0])
    assert lost == pytest.approx(float(correct["roll_down"].iloc[0]), abs=1e-8)
    assert lost * 252 > 0.008  # more than 80bp a year of return, silently deleted


# ---------------------------------------------------------------------------
# Excess returns, masking, monthly
# ---------------------------------------------------------------------------


def test_excess_return_subtracts_the_short_rate_known_at_the_start_of_the_day() -> None:
    params = flat_params(5.0, dates=3)
    rf = pd.Series([2.0, 99.0, 99.0], index=params.index)  # only the first is ever used
    frame = constant_maturity_return(
        10.0, params=params, risk_free_pct=rf, delta=DELTA, mask_unpublished=False
    )
    expected = math.expm1(DELTA * 5.0 / 100.0) - 2.0 * DELTA / 100.0
    assert float(frame["excess_return"].iloc[0]) == pytest.approx(expected, rel=1e-14)


def test_excess_return_is_missing_where_the_short_rate_is() -> None:
    params = flat_params(5.0, dates=3)
    rf = pd.Series([np.nan, 2.0, 2.0], index=params.index)
    frame = constant_maturity_return(
        10.0, params=params, risk_free_pct=rf, delta=DELTA, mask_unpublished=False
    )
    assert np.isnan(frame["excess_return"].iloc[0])
    assert np.isfinite(frame["total_return"].iloc[0])


def test_maturity_must_exceed_the_roll_step() -> None:
    with pytest.raises(GswError, match="must exceed the roll-down step"):
        constant_maturity_return(0.001, params=flat_params(), delta=DELTA, mask_unpublished=False)


def test_unknown_frequency_is_rejected() -> None:
    with pytest.raises(GswError, match="unknown freq"):
        constant_maturity_return(
            10.0,
            "weekly",  # type: ignore[arg-type]
            params=flat_params(),
            delta=DELTA,
            mask_unpublished=False,
        )


def test_monthly_compounds_the_daily_series_rather_than_resampling_a_level() -> None:
    params = flat_params(5.0, dates=25)
    daily = constant_maturity_return(10.0, params=params, delta=DELTA, mask_unpublished=False)
    monthly = constant_maturity_return(
        10.0, "monthly", params=params, delta=DELTA, mask_unpublished=False
    )
    expected = float((1.0 + daily["total_return"]).prod() - 1.0)
    assert len(monthly) == 1
    assert float(monthly["total_return"].iloc[0]) == pytest.approx(expected, rel=1e-14)
    assert float(monthly["carry"].iloc[0]) == pytest.approx(float(daily["carry"].sum()), rel=1e-14)


# ---------------------------------------------------------------------------
# Parsing -- the data contract
# ---------------------------------------------------------------------------


def test_parse_finds_the_header_without_guessing_skiprows() -> None:
    frame = parse_fed_curve_csv(FIXTURE_CSV, yield_prefix="SVENY")
    assert list(frame.columns[:4]) == ["BETA0", "BETA1", "BETA2", "BETA3"]
    assert {"TAU1", "TAU2", "SVENY02", "SVENY10", "SVENY30"} <= set(frame.columns)
    assert frame.index.name == "date"
    assert frame.index.is_monotonic_increasing


def test_parse_drops_market_holidays() -> None:
    # 1980-01-01 is present in the file with every field blank. Left in, it would
    # accrue a day of carry and roll that never happened.
    frame = parse_fed_curve_csv(FIXTURE_CSV, yield_prefix="SVENY")
    assert pd.Timestamp("1980-01-01") not in frame.index
    assert len(frame) == 4
    assert frame["BETA0"].notna().all()


def test_parse_rejects_a_file_whose_header_moved() -> None:
    with pytest.raises(GswError, match="do not guess skiprows"):
        parse_fed_curve_csv("no header here\n1,2,3\n", yield_prefix="SVENY")


def test_parse_rejects_a_missing_parameter_column() -> None:
    broken = (
        "preamble line\n\n"
        "Date,BETA0,BETA1,BETA2,BETA3,SVENY02,TAU1\n"
        "1980-01-02,9.2,-1.2,0.3,0.2,10.30,2.5\n"
    )
    with pytest.raises(GswError, match="missing required column"):
        parse_fed_curve_csv(broken, yield_prefix="SVENY")


def test_parse_rejects_a_file_with_no_published_yields() -> None:
    with pytest.raises(GswError, match="no published yield columns"):
        parse_fed_curve_csv(FIXTURE_CSV, yield_prefix="TIPSY")


def test_published_mask_bounds_history_to_what_the_fed_actually_publishes() -> None:
    frame = parse_fed_curve_csv(FIXTURE_CSV, yield_prefix="SVENY")
    mask = published_mask(frame, 30.0, yield_prefix="SVENY")
    assert not bool(mask.loc["1979-12-28"])  # SVENY30 is NA before the 30y existed
    assert bool(mask.loc["1980-01-02"])
    # n - Delta rounds up to the same published tenor.
    assert published_mask(frame, 30.0 - DELTA, yield_prefix="SVENY").equals(mask)


def test_published_mask_refuses_to_extrapolate_past_the_published_curve() -> None:
    frame = parse_fed_curve_csv(FIXTURE_CSV, yield_prefix="SVENY")
    with pytest.raises(GswError, match="to bound its history"):
        published_mask(frame, 45.0, yield_prefix="SVENY")


def test_masking_removes_returns_the_fed_does_not_publish_a_curve_for() -> None:
    frame = parse_fed_curve_csv(FIXTURE_CSV, yield_prefix="SVENY")
    masked = constant_maturity_return(
        30.0, params=frame, delta=DELTA, yield_prefix="SVENY", mask_unpublished=True
    )
    unmasked = constant_maturity_return(
        30.0, params=frame, delta=DELTA, yield_prefix="SVENY", mask_unpublished=False
    )
    assert masked["total_return"].notna().sum() == 1  # only 1980-01-03 has both ends published
    assert unmasked["total_return"].notna().sum() == 3


def test_parse_fred_csv_reads_a_percent_series() -> None:
    text = "observation_date,DGS1MO\n2001-07-31,3.67\n2001-08-01,.\n2001-08-02,3.65\n"
    series = parse_fred_csv(text, "DGS1MO")
    assert list(series.index) == [pd.Timestamp("2001-07-31"), pd.Timestamp("2001-08-02")]
    assert float(series.iloc[0]) == 3.67
    assert series.name == "DGS1MO"


def test_parse_fred_csv_rejects_the_wrong_series() -> None:
    with pytest.raises(GswError, match="has no column"):
        parse_fred_csv("observation_date,DGS1MO\n2001-07-31,3.67\n", "TB3MS")


# ---------------------------------------------------------------------------
# Breakeven
# ---------------------------------------------------------------------------


def test_breakeven_is_nominal_minus_real_on_the_same_day() -> None:
    nominal = flat_params(4.0, dates=3)
    real = flat_params(1.5, dates=3)
    series = breakeven_inflation(nominal=nominal, real=real, n=2.0)
    assert series.to_numpy() == pytest.approx(2.5, rel=1e-14)
    assert series.name == "breakeven_2y_pct"


# ---------------------------------------------------------------------------
# 4. The real 1961- sample
# ---------------------------------------------------------------------------


def _cached_or_skip(kind: gsw.CurveKind) -> pd.DataFrame:
    try:
        return gsw.load_curve(kind)
    except cache.CacheError as exc:  # pragma: no cover - depends on the local cache
        pytest.skip(f"no cached {kind} curve: {exc}")


@pytest.mark.dataset
@pytest.mark.parametrize("n", [2.0, 5.0, 10.0, 30.0])
def test_real_sample_returns_are_finite_and_plausible(n: float) -> None:
    """Bounds are scaled by maturity, because a zero's duration IS its maturity.

    A 30y zero moves 30x the yield change: the +18.13% of 1987-12-01 is a 55bp
    rally, not a defect. So the bound is stated as a yield move rather than as a
    return -- no single day has ever moved the fitted long yield 100bp -- and the
    annualised volatility is checked against the same duration relationship.
    Both are loose enough to pass on real history and tight enough to fail on a
    dropped roll, an unhandled TAU2 sentinel, or a mis-scaled percent.
    """
    params = _cached_or_skip("nominal")
    frame = constant_maturity_return(n, params=params, delta=DELTA)
    returns = frame["total_return"].dropna()

    assert len(returns) > 5_000
    assert np.isfinite(returns.to_numpy()).all()
    # |r| ~ n . |dy|; 100bp in one day is beyond anything in the record.
    assert returns.abs().max() < 0.01 * n
    # Measured: 1.03, 1.02, 1.03 and 0.94 percent of annualised vol per year of
    # maturity at 2y, 5y, 10y and 30y. The band is 0.5x to 2x that.
    annual_vol = float(returns.std() * math.sqrt(252))
    assert 0.005 * n < annual_vol < 0.02 * n
    assert float(returns.mean()) == pytest.approx(0.0, abs=0.001)


@pytest.mark.dataset
def test_real_sample_volatility_increases_with_maturity() -> None:
    params = _cached_or_skip("nominal")
    vols = {
        n: float(constant_maturity_return(n, params=params, delta=DELTA)["total_return"].std())
        for n in (2.0, 5.0, 10.0, 30.0)
    }
    assert vols[2.0] < vols[5.0] < vols[10.0] < vols[30.0]


@pytest.mark.dataset
def test_real_sample_survives_the_pre_1980_nelson_siegel_regime() -> None:
    # The TAU2 sentinel guard is what keeps this non-empty.
    params = _cached_or_skip("nominal")
    pre_1980 = constant_maturity_return(5.0, params=params, delta=DELTA).loc[:"1979-12-31"]
    # dropna on the column, not the frame: excess_return is empty before DGS1MO
    # begins in 2001, which would otherwise take the whole period with it.
    assert len(pre_1980["total_return"].dropna()) > 4_000
    assert np.isfinite(pre_1980["total_return"].dropna().to_numpy()).all()


@pytest.mark.dataset
def test_real_sample_component_identity_holds_everywhere() -> None:
    params = _cached_or_skip("nominal")
    frame = constant_maturity_return(30.0, params=params, delta=DELTA).dropna()
    rebuilt = frame[["carry", "roll_down", "duration_effect"]].sum(axis=1)
    assert rebuilt.to_numpy() == pytest.approx(
        np.log1p(frame["total_return"].to_numpy()), abs=1e-15
    )


@pytest.mark.dataset
def test_real_breakeven_is_economically_plausible() -> None:
    nominal = _cached_or_skip("nominal")
    real = _cached_or_skip("real")
    series = breakeven_inflation(nominal=nominal, real=real, n=10.0)
    assert len(series) > 5_000
    # 10y breakeven inflation has ranged roughly 0-3% since 1999, with a brief
    # negative print in the 2008 liquidity collapse.
    assert -2.0 < float(series.min()) < 1.0
    assert 1.5 < float(series.max()) < 5.0


@pytest.mark.dataset
def test_the_published_maturity_masks_match_the_documented_start_dates() -> None:
    params = _cached_or_skip("nominal")
    prefix = load().model.data.nominal_curve.yield_prefix
    starts = {
        n: published_mask(params, n, yield_prefix=prefix).idxmax() for n in (2.0, 5.0, 10.0, 30.0)
    }
    assert starts[2.0] == pd.Timestamp("1961-06-14")
    assert starts[5.0] == pd.Timestamp("1961-06-14")
    assert starts[10.0] == pd.Timestamp("1971-08-16")
    assert starts[30.0] == pd.Timestamp("1985-11-25")
