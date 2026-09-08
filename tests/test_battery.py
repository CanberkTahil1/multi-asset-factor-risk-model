"""SPEC.md 6.5's battery, each test against a value worked by hand. W4-P3.

Every statistic here has at least one case whose answer is arithmetic on a
short series written out in the test, and one case where the answer is exactly
zero by construction -- a calibrated breach rate for Kupiec, equal transition
probabilities for Christoffersen, a perfectly linear pair for Mincer-Zarnowitz.
The zero cases matter more than the numeric ones: they pin the sign convention.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from mafrm.risk import battery
from mafrm.risk.battery import BatteryError

# ---------------------------------------------------------------------------
# Newey-West on a mean
# ---------------------------------------------------------------------------


def test_newey_west_variance_of_mean_at_zero_lags_is_the_plain_formula() -> None:
    # [1, 2, 3, 4]: deviations +-1.5, +-0.5 -> sum of squares 5 -> gamma_0 = 5/4
    # variance of the mean = gamma_0 / T = 1.25 / 4 = 0.3125
    assert battery.newey_west_variance_of_mean(np.array([1.0, 2, 3, 4]), lags=0) == (
        pytest.approx(0.3125)
    )


def test_newey_west_variance_of_mean_one_lag_by_hand() -> None:
    # gamma_1 = (d2 d1 + d3 d2 + d4 d3) / T = (0.75 - 0.25 + 0.75) / 4 = 0.3125
    # Bartlett weight at L = 1: 1 - 1/2 = 0.5 -> S = 1.25 + 2 * 0.5 * 0.3125 = 1.5625
    assert battery.newey_west_variance_of_mean(np.array([1.0, 2, 3, 4]), lags=1) == (
        pytest.approx(1.5625 / 4)
    )


# ---------------------------------------------------------------------------
# Mincer-Zarnowitz and QLIKE
# ---------------------------------------------------------------------------


def test_qlike_by_hand() -> None:
    # h = [1, 2], r^2 = [1, 4]: (log 1 + 1) and (log 2 + 2) -> mean = (1 + 2.693147) / 2
    assert battery.qlike_loss(np.array([1.0, 4.0]), np.array([1.0, 2.0])) == (
        pytest.approx(1.8465736, abs=1e-6)
    )


def test_qlike_rejects_a_non_positive_forecast() -> None:
    with pytest.raises(BatteryError):
        battery.qlike_loss(np.array([1.0]), np.array([0.0]))


def test_mincer_zarnowitz_recovers_an_exact_line() -> None:
    forecast = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    realised = 0.5 + 2.0 * forecast
    fit = battery.mincer_zarnowitz(realised, forecast, newey_west_lags=2)
    assert fit.intercept == pytest.approx(0.5)
    assert fit.slope == pytest.approx(2.0)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.observations == 6


def test_mincer_zarnowitz_is_calibrated_on_a_calibrated_forecast() -> None:
    rng = np.random.default_rng(1)
    forecast = np.exp(rng.standard_normal(4000) * 0.3)
    realised = forecast * rng.standard_normal(4000) ** 2
    fit = battery.mincer_zarnowitz(realised, forecast, newey_west_lags=5)
    assert fit.intercept == pytest.approx(0.0, abs=3 * fit.intercept_standard_error)
    assert fit.slope == pytest.approx(1.0, abs=3 * fit.slope_standard_error)
    assert fit.joint_p_value > 0.01


def test_mincer_zarnowitz_sees_a_level_understatement() -> None:
    """A forecast that is 1.33^2 too small in variance: slope well above one."""
    rng = np.random.default_rng(2)
    forecast = np.exp(rng.standard_normal(4000) * 0.3)
    realised = forecast * 1.33**2 * rng.standard_normal(4000) ** 2
    fit = battery.mincer_zarnowitz(realised, forecast, newey_west_lags=5)
    assert fit.slope > 1.4
    assert fit.joint_p_value < 1e-6


# ---------------------------------------------------------------------------
# Ljung-Box
# ---------------------------------------------------------------------------


def test_ljung_box_by_hand() -> None:
    # x = [1, 2, 3, 4], n = 4: r_1 = 0.25, r_2 = -0.30
    # Q(2) = 4 * 6 * (0.25^2 / 3 + 0.30^2 / 2) = 24 * 0.0658333 = 1.58
    result = battery.ljung_box(np.array([1.0, 2, 3, 4]), lags=2)
    assert result.statistic == pytest.approx(1.58, abs=1e-9)
    assert result.lags == 2
    assert result.observations == 4


def test_ljung_box_does_not_reject_white_noise_and_does_reject_clustering() -> None:
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(3000) ** 2
    assert battery.ljung_box(noise, lags=21).p_value > 0.01
    # GARCH-like clustering: variance follows a slow AR(1) in log space.
    log_variance = np.zeros(3000)
    shocks = rng.standard_normal(3000)
    for index in range(1, 3000):
        log_variance[index] = 0.97 * log_variance[index - 1] + 0.3 * shocks[index]
    clustered = np.exp(log_variance) * rng.standard_normal(3000) ** 2
    assert battery.ljung_box(clustered, lags=21).p_value < 1e-6


def test_ljung_box_rejects_a_bad_lag_count() -> None:
    with pytest.raises(BatteryError):
        battery.ljung_box(np.arange(5.0), lags=5)


# ---------------------------------------------------------------------------
# Kupiec, Christoffersen, Basel
# ---------------------------------------------------------------------------


def test_kupiec_is_zero_at_exact_coverage_and_by_hand_otherwise() -> None:
    exact = battery.kupiec(25, 500, level=0.95)
    assert exact.statistic == pytest.approx(0.0, abs=1e-12)
    assert exact.p_value == pytest.approx(1.0)
    # T = 250, x = 5, p = 0.01:
    #   2 [ 5 ln(5 / 2.5) + 245 ln(245 / 247.5) ] = 2 [ 3.465736 - 2.487346 ] = 1.956780
    result = battery.kupiec(5, 250, level=0.99)
    assert result.statistic == pytest.approx(1.95678, abs=1e-4)
    assert result.expected == pytest.approx(2.5)
    assert result.p_value > 0.05


def test_breach_indicator_is_the_normal_var_at_the_level() -> None:
    series = np.array([-3.0, -1.7, -1.6, 0.0, 2.0])
    at_95 = battery.breach_indicator(series, level=0.95)  # z = 1.645
    at_99 = battery.breach_indicator(series, level=0.99)  # z = 2.326
    assert at_95.tolist() == [True, True, False, False, False]
    assert at_99.tolist() == [True, False, False, False, False]


def test_christoffersen_independence_is_zero_when_transitions_are_equal() -> None:
    # 0 0 1 1 0 0 0 1 0 0 -> n00 = 4, n01 = 2, n10 = 2, n11 = 1: pi_01 = pi_11 = pi = 1/3
    flags = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0], dtype=bool)
    result = battery.christoffersen(flags, level=0.95)
    assert result.transitions == (4, 2, 2, 1)
    assert result.pi_01 == pytest.approx(1 / 3)
    assert result.pi_11 == pytest.approx(1 / 3)
    assert result.independence_statistic == pytest.approx(0.0, abs=1e-12)
    assert result.conditional_coverage_statistic == pytest.approx(result.unconditional.statistic)


def test_christoffersen_independence_by_hand_on_a_run() -> None:
    # 1 1 1 0 0 0 0 0 0 0 -> n00 = 6, n01 = 0, n10 = 1, n11 = 2; pi_01 = 0, pi_11 = 2/3, pi = 2/9
    #   restricted   = 7 ln(7/9) + 2 ln(2/9)           = -1.759201 - 3.008155 = -4.767356
    #   unrestricted = 6 ln 1 + 0 + 1 ln(1/3) + 2 ln(2/3) = -1.098612 - 0.810930 = -1.909543
    #   LR_ind = -2 (restricted - unrestricted) = 5.715626
    flags = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    result = battery.christoffersen(flags, level=0.95)
    assert result.transitions == (6, 0, 1, 2)
    assert result.independence_statistic == pytest.approx(5.715626, abs=1e-5)
    assert result.independence_p_value < 0.05


def test_basel_cumulative_probabilities_match_the_published_table() -> None:
    """Basel Committee (1996), 250 days at 99%: 4 exceptions 89.22%, 5 exceptions 95.88%."""
    assert battery.binomial_cumulative(4, window=250, probability=0.01) == (
        pytest.approx(0.8922, abs=5e-5)
    )
    assert battery.binomial_cumulative(5, window=250, probability=0.01) == (
        pytest.approx(0.9588, abs=5e-5)
    )


def test_basel_zones_on_non_overlapping_blocks() -> None:
    flags = np.zeros(750, dtype=bool)
    flags[:4] = True  # block 1: 4 -> green
    flags[250:259] = True  # block 2: 9 -> yellow
    flags[500:510] = True  # block 3: 10 -> red
    result = battery.basel_traffic_light(
        np.concatenate([flags, np.zeros(37, dtype=bool)]),
        window=250,
        level=0.99,
        green_max=4,
        yellow_max=9,
    )
    assert [block.zone for block in result.blocks] == ["green", "yellow", "red"]
    assert result.count("green") == 1
    assert result.unscored_tail == 37
    assert result.overlap == 0


def test_basel_refuses_a_series_shorter_than_one_block() -> None:
    with pytest.raises(BatteryError):
        battery.basel_traffic_light(
            np.zeros(100, dtype=bool), window=250, level=0.99, green_max=4, yellow_max=9
        )


# ---------------------------------------------------------------------------
# Acerbi-Szekely
# ---------------------------------------------------------------------------


def test_acerbi_szekely_by_hand_at_z_equal_one() -> None:
    """level = Phi(1), so z = 1, alpha = 0.158655, phi(1) = 0.241971, ES = phi/alpha = 1.52514.

    b = [-3, 0.5, -1.5, 2]: breaches where b < -1 -> [1, 0, 1, 0], sum(b I) = -4.5.
      Z_2 = -4.5 / (4 * 0.241971) + 1 = -4.64932 + 1 = -3.64932
      u   = 1 + max(-b - 1, 0)/alpha = [1 + 2/alpha, 1, 1 + 0.5/alpha, 1]
          = [13.60600, 1, 4.15150, 1] -> mean 4.93938 -> / ES - 1 = 2.23864
    """
    level = float(norm.cdf(1.0))
    series = np.array([-3.0, 0.5, -1.5, 2.0])
    result = battery.acerbi_szekely(series, level=level, trials=200, seed=0)
    assert result.breaches == 2
    assert result.z_2 == pytest.approx(-3.64932, abs=1e-4)
    assert result.minimally_biased == pytest.approx(2.23864, abs=1e-4)


def test_acerbi_szekely_is_centred_under_the_null_and_negative_when_es_is_understated() -> None:
    rng = np.random.default_rng(5)
    calibrated = rng.standard_normal(3000)
    null = battery.acerbi_szekely(calibrated, level=0.99, trials=1000, seed=1)
    assert null.z_2 == pytest.approx(0.0, abs=0.5)
    assert null.z_2_p_value > 0.01
    understated = battery.acerbi_szekely(1.5 * calibrated, level=0.99, trials=1000, seed=1)
    assert understated.z_2 < -0.5
    assert understated.z_2_p_value < 0.01
    assert understated.minimally_biased > 0.2
    assert understated.minimally_biased_p_value < 0.01


# ---------------------------------------------------------------------------
# The tail diagnostic
# ---------------------------------------------------------------------------


def test_empirical_quantile_against_the_normal() -> None:
    rng = np.random.default_rng(6)
    fat = rng.standard_t(3, size=20000)
    quantile = battery.empirical_quantile(fat, level=0.99)
    assert quantile.normal == pytest.approx(-2.3263, abs=1e-3)
    assert quantile.ratio > 1.5  # t(3) has a much fatter 1% tail than the normal
    thin = battery.empirical_quantile(rng.standard_normal(20000), level=0.99)
    assert thin.ratio == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------
# Rank ordering
# ---------------------------------------------------------------------------


def test_rank_correlation_by_hand_on_one_block() -> None:
    """Forecast ranks members 1 < 2 < 3; realised risk ranks them 1 < 3 < 2 -> Spearman 0.5."""
    forecast = np.tile(np.array([1.0, 2.0, 3.0]), (4, 1))
    realised = np.array(
        [
            [1.0, 4.0, 2.0],
            [-1.0, -4.0, -2.0],
            [1.0, 4.0, 2.0],
            [-1.0, -4.0, -2.0],
        ]
    )
    result = battery.cross_sectional_rank_correlation(forecast, realised, block=4)
    assert result.blocks == 1
    assert result.mean == pytest.approx(0.5)
    assert result.overlap == 0


def test_rank_correlation_is_high_when_the_forecast_orders_risk() -> None:
    rng = np.random.default_rng(8)
    scale = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    realised = rng.standard_normal((420, 5)) * scale
    forecast = np.tile(scale, (420, 1))
    result = battery.cross_sectional_rank_correlation(forecast, realised, block=21)
    assert result.blocks == 20
    assert result.mean > 0.8
    assert result.fraction_positive == 1.0


# ---------------------------------------------------------------------------
# Factor t-statistics
# ---------------------------------------------------------------------------


def test_factor_t_statistic_by_hand() -> None:
    # mean 2.5, NW variance at zero lags 0.3125 -> se 0.559017 -> t = 4.472136
    (result,) = battery.factor_t_statistics(
        np.array([[1.0], [2.0], [3.0], [4.0]]),
        ["only"],
        newey_west_lags=0,
        window=2,
        threshold=2.0,
    )
    assert result.t_statistic == pytest.approx(4.472136, abs=1e-5)
    # Rolling t over windows of 2 stepped by 1: sd = 0.7071, se = 0.5, means 1.5 / 2.5 / 3.5 -> 3, 5, 7
    assert result.rolling.tolist() == pytest.approx([3.0, 5.0, 7.0])
    assert result.overlap == 1
    # Non-overlapping: [1,2] and [3,4] -> t = 3 and 7
    assert result.non_overlapping.tolist() == pytest.approx([3.0, 7.0])
    assert result.rolling_exceedance == 1.0
    assert result.non_overlapping_exceedance == 1.0


def test_factor_t_statistics_on_noise_are_not_significant() -> None:
    rng = np.random.default_rng(9)
    returns = rng.standard_normal((3000, 3))
    results = battery.factor_t_statistics(
        returns, ["a", "b", "c"], newey_west_lags=5, window=252, threshold=2.0
    )
    assert all(abs(result.t_statistic) < 3.0 for result in results)
    # Under the null ~5% of independent windows exceed |t| > 2.
    assert all(result.non_overlapping_exceedance < 0.4 for result in results)
