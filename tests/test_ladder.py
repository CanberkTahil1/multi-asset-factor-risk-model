"""Par-coupon ladder tests. SPEC.md 3.2, validation comparand.

Every expected value is derived analytically from the par-bond definition and
written out independently of the module. The flat-curve case is exact in closed
form, which is what makes it a real check rather than a regression snapshot:

    D(t_k) = exp(-y.t_k),  t_k = k/f,  k = 1..N     (flat, continuously compounded)
    sum(D) = r(1-r^N)/(1-r)  with r = exp(-y/f),    D(m) = r^N
    c = f.(1-D(m))/sum(D).100 = 100.f.(exp(y/f) - 1)

and, because ageing a flat curve by Delta multiplies every discount factor by
exactly exp(y.Delta), the static return is exp(y.Delta) - 1 and the roll-down is
identically zero.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mafrm.data import cache, gsw
from mafrm.data.ladder import (
    LadderError,
    cashflow_times,
    discount_factors,
    ladder_returns,
    par_bond_returns,
    par_coupon_rate,
    seasoned_bond_returns,
)

DELTA = 1.0 / 252.0
FREQ = 2


def flat_params(level_pct: float, dates: int = 3) -> pd.DataFrame:
    """A perfectly flat continuously-compounded curve at ``level_pct`` percent."""
    index = pd.to_datetime([f"2020-01-{day + 1:02d}" for day in range(dates)])
    index.name = "date"
    columns = {
        "BETA0": level_pct,
        "BETA1": 0.0,
        "BETA2": 0.0,
        "BETA3": 0.0,
        "TAU1": 2.0,
        "TAU2": 5.0,
    }
    frame = pd.DataFrame(columns, index=index)
    for tenor in range(1, 31):
        frame[f"SVENY{tenor:02d}"] = level_pct
    return frame


def sloped_params(beta0: float, beta1: float, tau1: float, dates: int = 3) -> pd.DataFrame:
    """A static upward-sloping Nelson-Siegel curve (beta1 < 0), unchanged daily."""
    frame = flat_params(beta0, dates)
    frame["BETA1"] = beta1
    frame["TAU1"] = tau1
    frame["TAU2"] = np.nan
    return frame


def analytic_par_coupon(y_pct: float, frequency: int) -> float:
    """``100.f.(exp(y/f) - 1)`` -- the par coupon on a flat cont-comp curve."""
    return 100.0 * frequency * (math.exp(y_pct / 100.0 / frequency) - 1.0)


# ---------------------------------------------------------------------------
# Cashflow schedule
# ---------------------------------------------------------------------------


def test_cashflow_times_are_the_coupon_dates() -> None:
    assert list(cashflow_times(2.0, 2)) == [0.5, 1.0, 1.5, 2.0]
    assert list(cashflow_times(1.0, 4)) == [0.25, 0.5, 0.75, 1.0]
    assert len(cashflow_times(30.0, 2)) == 60


def test_cashflow_times_reject_a_partial_period() -> None:
    with pytest.raises(LadderError, match="whole number"):
        cashflow_times(2.3, 2)
    with pytest.raises(LadderError, match="maturity must be positive"):
        cashflow_times(0.0, 2)


# ---------------------------------------------------------------------------
# Par coupon -- hand-computed
# ---------------------------------------------------------------------------


def test_par_coupon_on_a_flat_five_percent_curve() -> None:
    """c = 200.(e^0.025 - 1) = 200 x 0.0253151205244289 = 5.06302410488578%."""
    params = flat_params(5.0)
    discounts = discount_factors(params, cashflow_times(2.0, FREQ))
    coupon = par_coupon_rate(discounts, FREQ)

    expected = 100.0 * 2 * (math.exp(0.025) - 1.0)
    assert expected == pytest.approx(5.06302410488578, rel=1e-13)
    assert coupon == pytest.approx(expected, rel=1e-13)
    # Same answer at every maturity: a flat curve prices every par bond alike.
    for maturity in (5.0, 10.0, 30.0):
        other = par_coupon_rate(discount_factors(params, cashflow_times(maturity, FREQ)), FREQ)
        assert other == pytest.approx(expected, rel=1e-13)


def test_par_coupon_is_zero_on_a_zero_curve() -> None:
    # Every discount factor is 1, so the bond is worth its face with no coupon.
    discounts = discount_factors(flat_params(0.0), cashflow_times(10.0, FREQ))
    assert discounts == pytest.approx(1.0, abs=1e-15)
    assert par_coupon_rate(discounts, FREQ) == pytest.approx(0.0, abs=1e-15)


def test_par_coupon_carries_the_frequency_factor() -> None:
    """Dropping the ``f`` halves the coupon; the bond then prices near 50, not 100.

    This is the defect the par identity in ``par_bond_returns`` exists to catch.
    """
    params = flat_params(5.0)
    times = cashflow_times(10.0, FREQ)
    discounts = discount_factors(params, times)
    with_factor = par_coupon_rate(discounts, FREQ)
    without_factor = with_factor / FREQ

    def price(coupon: float) -> float:
        per_period = coupon / FREQ
        return float(per_period * discounts[0, :-1].sum() + (per_period + 100.0) * discounts[0, -1])

    assert price(float(with_factor[0])) == pytest.approx(100.0, abs=1e-12)
    assert price(float(without_factor[0])) < 90.0


def test_par_bond_prices_at_exactly_par_on_a_sloped_curve() -> None:
    params = sloped_params(6.0, -4.0, 40.0)
    times = cashflow_times(20.0, FREQ)
    discounts = discount_factors(params, times)
    coupon = par_coupon_rate(discounts, FREQ)
    per_period = coupon / FREQ
    price = (per_period[:, None] * discounts[:, :-1]).sum(axis=1) + (
        per_period + 100.0
    ) * discounts[:, -1]
    assert price == pytest.approx(100.0, abs=1e-11)


# ---------------------------------------------------------------------------
# Returns -- exact on a flat curve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("maturity", [2.0, 20.0, 30.0])
def test_flat_curve_gives_pure_accrual_and_exactly_zero_roll(maturity: float) -> None:
    """Ageing a flat curve by Delta scales every discount factor by exp(y.Delta).

    So the static return is exp(y.Delta) - 1 for any maturity and any coupon,
    and the roll-down is identically zero -- the same identity the zero-coupon
    series satisfies, which is what makes the two comparable.
    """
    level = 5.0
    frame = par_bond_returns(
        flat_params(level, dates=4), maturity, frequency=FREQ, delta=DELTA, mask_unpublished=False
    )

    expected_log = 0.05 * DELTA
    assert expected_log == pytest.approx(5.0 / 100.0 / 252.0, rel=1e-15)
    assert frame["roll_down"].abs().max() == pytest.approx(0.0, abs=1e-14)
    assert frame["carry"].to_numpy() == pytest.approx(expected_log, rel=1e-13)
    assert frame["total_return"].to_numpy() == pytest.approx(math.expm1(expected_log), rel=1e-12)
    # Hand value: x = 0.05/252 = 1.98412698e-4; expm1(x) = x + x^2/2 = 1.98432382e-4
    assert float(frame["total_return"].iloc[0]) == pytest.approx(1.98432382e-4, rel=1e-8)
    assert float(frame["par_coupon_pct"].iloc[0]) == pytest.approx(
        analytic_par_coupon(level, FREQ), rel=1e-13
    )


def test_zero_curve_gives_exactly_zero_return() -> None:
    frame = par_bond_returns(
        flat_params(0.0, dates=4), 10.0, frequency=FREQ, delta=DELTA, mask_unpublished=False
    )
    assert frame["total_return"].abs().max() == pytest.approx(0.0, abs=1e-15)
    assert frame["carry"].abs().max() == pytest.approx(0.0, abs=1e-15)
    assert frame["roll_down"].abs().max() == pytest.approx(0.0, abs=1e-14)


def test_components_reconstruct_the_total_return() -> None:
    frame = par_bond_returns(
        sloped_params(6.0, -4.0, 40.0),
        20.0,
        frequency=FREQ,
        delta=DELTA,
        mask_unpublished=False,
    )
    rebuilt = frame[["carry", "roll_down", "duration_effect"]].sum(axis=1)
    assert rebuilt.to_numpy() == pytest.approx(
        np.log1p(frame["total_return"].to_numpy()), rel=1e-12
    )


def test_upward_sloping_curve_rolls_down_positively_but_less_than_the_zero() -> None:
    """The economic point that made the ladder the right comparand.

    At the same maturity a coupon bond rolls less than a zero, because only its
    principal sits at ``m`` -- every coupon sits further down a curve that is
    flatter there. This is why scoring TLT against a 20y zero over-credited the
    synthetic with roll it could not have earned.
    """
    params = sloped_params(6.0, -4.0, 40.0)
    coupon_roll = float(
        par_bond_returns(params, 20.0, frequency=FREQ, delta=DELTA, mask_unpublished=False)[
            "roll_down"
        ].iloc[0]
    )
    zero_roll = float(
        gsw.constant_maturity_return(20.0, params=params, delta=DELTA, mask_unpublished=False)[
            "roll_down"
        ].iloc[0]
    )
    assert coupon_roll > 0.0
    assert coupon_roll < zero_roll


def test_a_first_coupon_inside_the_roll_step_is_refused() -> None:
    with pytest.raises(LadderError, match="within one roll step"):
        par_bond_returns(flat_params(5.0), 20.0, frequency=FREQ, delta=0.6, mask_unpublished=False)


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def test_ladder_is_the_weighted_average_of_its_legs() -> None:
    params = sloped_params(6.0, -4.0, 40.0, dates=4)
    maturities = (20.0, 30.0)
    weights = (3.0, 1.0)
    blended = ladder_returns(
        params, maturities=maturities, weights=weights, mask_unpublished=False, delta=DELTA
    )
    legs = [
        par_bond_returns(params, m, frequency=FREQ, delta=DELTA, mask_unpublished=False)
        for m in maturities
    ]
    expected = 0.75 * legs[0]["total_return"] + 0.25 * legs[1]["total_return"]
    assert blended["total_return"].to_numpy() == pytest.approx(expected.to_numpy(), rel=1e-13)


def test_ladder_weights_are_normalised_not_taken_literally() -> None:
    params = sloped_params(6.0, -4.0, 40.0, dates=4)
    a = ladder_returns(params, maturities=(20.0, 30.0), weights=(3.0, 1.0), mask_unpublished=False)
    b = ladder_returns(
        params, maturities=(20.0, 30.0), weights=(30.0, 10.0), mask_unpublished=False
    )
    assert a["total_return"].to_numpy() == pytest.approx(b["total_return"].to_numpy(), rel=1e-15)


def test_ladder_rejects_mismatched_weights() -> None:
    params = sloped_params(6.0, -4.0, 40.0)
    with pytest.raises(LadderError, match="same length"):
        ladder_returns(params, maturities=(20.0, 30.0), weights=(1.0,), mask_unpublished=False)


# ---------------------------------------------------------------------------
# The architectural boundary -- the ladder must not reach production
# ---------------------------------------------------------------------------


def test_no_production_module_imports_the_ladder() -> None:
    """CLAUDE.md invariant 10 in spirit, and SPEC.md 3.2 in letter.

    The ladder is a validation comparand. The government sleeve's factor inputs
    are the constant-maturity zeros, because a blended ladder's loading on the
    level/slope/curvature structure is a mixture rather than a clean key rate.
    Only the validation path may import it.
    """
    root = Path(__file__).resolve().parents[1] / "src" / "mafrm"
    allowed = {"ladder.py", "gsw_report.py"}
    importers = re.compile(
        r"^\s*(?:from\s+mafrm\.data\.ladder\b|import\s+mafrm\.data\.ladder\b|from\s+mafrm\.data\s+import\s+[^\n]*\bladder\b)",
        re.M,
    )
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if path.name not in allowed and importers.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"the ladder leaked into the production path: {offenders}"

    # And the production module genuinely does not carry it as an attribute.
    assert not hasattr(gsw, "ladder_returns")
    assert not hasattr(gsw, "par_bond_returns")


# ---------------------------------------------------------------------------
# The gate, on real data
# ---------------------------------------------------------------------------


#: These are W1-P3 data-layer cross-check tests: they compare a VENDOR price
#: series (TLT) against our curve construction over everything cached, which is a
#: question about the data rather than about the model, and their pinned numbers
#: come from that full window. They must read the same window `gsw_report` does,
#: or the comparand and the comparison would be on different calendars.
_REASON = "data-layer vendor cross-check: TLT vs the GSW ladder over the full cached history"


def _cached_or_skip() -> pd.DataFrame:
    try:
        return gsw.load_curve_unrestricted("nominal", reason=_REASON)
    except cache.CacheError as exc:  # pragma: no cover - depends on the local cache
        pytest.skip(f"no cached nominal curve: {exc}")


@pytest.mark.dataset
def test_real_ladder_beta_against_tlt_is_within_the_gate_band() -> None:
    from mafrm.config import load
    from mafrm.data.gsw_report import cross_check_tlt

    _cached_or_skip()
    gate = load().model.data.tlt_cross_check
    check = cross_check_tlt()
    assert gate.beta_min <= check.ladder_beta <= gate.beta_max


@pytest.mark.dataset
@pytest.mark.xfail(
    strict=True,
    reason=(
        "UNRESOLVED, deliberately. SPEC.md 3.2 sets the tolerance at TLT's 0.15%/yr "
        "expense ratio; the measured gap is -0.192%/yr. Chaining the fund's published "
        "tracking difference (-0.09%/yr vs the ICE 20+ index, iShares fact sheet) gives "
        "ladder - index = +0.102%/yr, and the fund beats its own fee, so no ETF "
        "implementation drag is available to absorb it. Three structural corrections "
        "were measured independently and pre-registered: par-versus-seasoned bonds "
        "(-0.005%/yr, prediction of +0.06%/yr REFUTED), coupon-date phase (-0.026%/yr) "
        "and Delta-versus-calendar accrual (+0.033%/yr). Net they WIDEN the gap to "
        "+0.114%/yr. The leading remaining suspect is that GSW fit a smooth curve to a "
        "filtered bond set excluding on-the-run issues, while the index prices the "
        "actual stock -- untestable without bond-level data. The tolerance is NOT "
        "widened to absorb a residual it is supposed to test. Remove this marker when "
        "the residual is explained or the comparand is refined; see "
        "reports/gsw_validation.md section 6."
    ),
)
def test_real_ladder_gap_against_tlt_is_within_the_gate() -> None:
    from mafrm.config import load
    from mafrm.data.gsw_report import cross_check_tlt

    _cached_or_skip()
    gate = load().model.data.tlt_cross_check
    check = cross_check_tlt()
    assert abs(check.ladder_gap_pct) <= gate.max_abs_gap_pct_per_year


@pytest.mark.dataset
def test_the_gate_has_power_against_a_flipped_roll() -> None:
    """Whatever the tolerance ends up being, the statistic must move when the roll does."""
    from mafrm.config import load
    from mafrm.data.gsw_report import cross_check_tlt

    _cached_or_skip()
    gate = load().model.data.tlt_cross_check
    check = cross_check_tlt()
    moved = abs(check.ladder_gap_pct_roll_flipped - check.ladder_gap_pct)
    assert moved > 2 * gate.max_abs_gap_pct_per_year


# ---------------------------------------------------------------------------
# Seasoned bonds -- bounding the par idealisation
# ---------------------------------------------------------------------------


def test_seasoning_zero_reproduces_the_par_bond_exactly() -> None:
    """The two constructions must differ only in the coupon they carry.

    At zero seasoning the bond is struck today at today's par rate, so it IS the
    par bond and its price base is 100. Anything else means the seasoned path
    has picked up a second difference.
    """
    params = sloped_params(6.0, -4.0, 40.0, dates=6)
    fresh = par_bond_returns(params, 20.0, frequency=FREQ, delta=DELTA, mask_unpublished=False)[
        "total_return"
    ]
    seasoned = seasoned_bond_returns(
        params, 20.0, 0.0, frequency=FREQ, delta=DELTA, mask_unpublished=False
    )
    assert seasoned["price_pct"].dropna().to_numpy() == pytest.approx(100.0, abs=1e-11)
    assert seasoned["total_return"].to_numpy() == pytest.approx(fresh.to_numpy(), abs=1e-14)


def long_static_params(beta0: float, beta1: float, tau1: float) -> pd.DataFrame:
    """The same static curve, but spanning enough calendar years to season into."""
    index = pd.bdate_range("2010-01-04", periods=1800, name="date")
    frame = pd.DataFrame(
        {
            "BETA0": beta0,
            "BETA1": beta1,
            "BETA2": 0.0,
            "BETA3": 0.0,
            "TAU1": tau1,
            "TAU2": np.nan,
        },
        index=index,
    )
    for tenor in range(1, 31):
        frame[f"SVENY{tenor:02d}"] = beta0
    return frame


def test_a_seasoned_bond_on_a_static_curve_is_priced_off_par() -> None:
    """Struck at a longer maturity on an upward-sloping curve it carries a higher
    coupon than today's par rate, so on an unchanged curve it trades at a premium.

    This is the mechanism the idealisation bound measures: same maturity, same
    curve, a different coupon and therefore a different present-value weighting.
    """
    params = long_static_params(6.0, -4.0, 40.0)
    seasoned = seasoned_bond_returns(
        params, 20.0, 5.0, frequency=FREQ, delta=DELTA, mask_unpublished=False
    )
    price = seasoned["price_pct"].dropna()
    assert not price.empty
    assert (price > 100.0).all()

    # And the fresh par bond of the same maturity is worth exactly 100.
    fresh = par_bond_returns(params, 20.0, frequency=FREQ, delta=DELTA, mask_unpublished=False)
    assert float(fresh["par_coupon_pct"].iloc[0]) < float(seasoned["coupon_pct"].dropna().iloc[0])


def test_seasoning_rejects_a_negative_age() -> None:
    with pytest.raises(LadderError, match="non-negative"):
        seasoned_bond_returns(
            flat_params(5.0), 20.0, -1.0, frequency=FREQ, delta=DELTA, mask_unpublished=False
        )


def test_coupon_phase_shifts_the_schedule_but_keeps_redemption() -> None:
    times = cashflow_times(2.0, 2, first_coupon=0.2)
    assert times[0] == pytest.approx(0.2)
    assert times[-1] == pytest.approx(2.0)
    assert np.all(np.diff(times) > 0)
    # A full period reproduces the default schedule exactly.
    assert list(cashflow_times(2.0, 2, first_coupon=0.5)) == list(cashflow_times(2.0, 2))


def test_coupon_phase_outside_the_period_is_refused() -> None:
    with pytest.raises(LadderError, match="first_coupon must lie"):
        cashflow_times(2.0, 2, first_coupon=0.7)


@pytest.mark.dataset
def test_the_measured_idealisation_does_not_explain_the_gap() -> None:
    """Pins the refutation, so a later session cannot quietly re-open it.

    The pre-registered prediction was +0.06%/yr rising in seasoning. The measured
    effect at plausible seasonings is an order of magnitude smaller and not
    reliably signed. If this ever starts passing at +0.10%/yr, the reconciliation
    in reports/gsw_validation.md section 6 has to be rewritten, not patched.
    """
    from mafrm.data.gsw_report import seasoning_sweep

    params = _cached_or_skip()
    from mafrm.data.gsw_report import _tlt_total_return

    sweep = seasoning_sweep(params, _tlt_total_return())
    realistic = sweep.loc[["3", "4", "5", "6", "7"], "par_minus_seasoned_pct"]
    assert abs(float(realistic.mean())) < 0.03
    assert float(sweep["par_minus_seasoned_pct"].min()) > -0.20
    assert float(sweep["par_minus_seasoned_pct"].max()) < 0.10
