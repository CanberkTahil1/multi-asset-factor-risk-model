"""SPEC.md 5.5's four legs, and the two claims its prose makes about itself.

The two that matter and would not be caught by a shape assertion:

* stage (d) shrinks an estimate **less** the further it sits from its bucket
  mean, which is backwards from naive shrinkage;
* the deviation in it is **linear absolute**, not squared. Both forms are
  monotone, both land in ``[0, 1]`` and both look right in a chart, so the
  distinction is pinned numerically here rather than trusted to a comment.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mafrm.risk import specific
from mafrm.risk.config import HORIZONS, RiskConfig
from mafrm.risk.covariance import ewma_autocovariance, ewma_second_moment

SEED = 20260901


def _panel(periods: int = 400, assets: int = 5, *, seed: int = SEED) -> np.ndarray:
    """A panel unrelated to either factor model. Never a global seed."""
    rng = np.random.default_rng(seed)
    scale = np.array([1.0, 4.0, 0.5, 9.0, 2.0])[:assets]
    return rng.standard_normal((periods, assets)) * scale


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        values,
        index=pd.bdate_range("2010-01-04", periods=values.shape[0]),
        columns=[f"s{i}" for i in range(values.shape[1])],
    )


# ---------------------------------------------------------------------------
# Leg (a) -- and the assertion that there is only one estimator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("halflife", [21.0, 84.0, 252.0])
def test_the_diagonal_path_reproduces_the_matrix_path(halflife: float) -> None:
    """``O(N T)`` and ``O(N^2 T)`` evaluations of the same estimator agree.

    The reason this test exists is in the note above
    ``mafrm.risk.covariance.bartlett_weight``: SPEC.md 5.5 needs only the
    diagonal, forming an ``N x N`` matrix to throw it away does not transfer to
    the ``N`` SPEC.md 15.2 points this code at, and a second evaluation that is
    never checked against the first is a second estimator.

    The tolerance is DERIVED rather than chosen: the two paths sum the same ``T``
    products in different orders, so they may differ by accumulated rounding of
    order ``T * eps`` relative to the scale of the result.
    """
    values = _panel()
    diagonal = specific.ewma_variance_diagonal(values, halflife=halflife)
    matrix = np.diag(ewma_second_moment(values, halflife=halflife))
    bound = np.finfo(float).eps * values.shape[0] * float(np.max(matrix))
    assert np.max(np.abs(diagonal - matrix)) <= bound


@pytest.mark.parametrize("lag", [1, 2, 5])
def test_the_lagged_diagonal_path_reproduces_the_matrix_path(lag: int) -> None:
    """Same claim for the autocovariance terms, including the pair convention."""
    values = _panel()
    diagonal = specific.lagged_autocovariance_diagonal(values, halflife=84.0, lag=lag)
    matrix = np.diag(ewma_autocovariance(values, halflife=84.0, lag=lag))
    bound = np.finfo(float).eps * values.shape[0] * float(np.max(np.abs(matrix)))
    assert np.max(np.abs(diagonal - matrix)) <= max(bound, 1e-12)


def test_the_lag_zero_autocovariance_is_the_variance_call_itself() -> None:
    """Not "agrees with" -- IS. One estimator, so the centring cannot drift."""
    values = _panel()
    assert np.array_equal(
        specific.lagged_autocovariance_diagonal(values, halflife=84.0, lag=0),
        specific.ewma_variance_diagonal(values, halflife=84.0),
    )


def test_time_series_estimate_hand_computed_at_one_lag() -> None:
    """A four-observation, one-asset panel with the whole sum written out.

    ``T = 4`` at half-life 1 gives decay ``1/2`` and unnormalised weights
    ``(1/8, 1/4, 1/2, 1)`` on the rows oldest-first, so ``w = (1, 2, 4, 8) / 15``.
    With ``L = 1`` the Bartlett kernel is ``1 - 1/2 = 1/2`` and the lag-1 pairs
    carry the weight of their more recent member, renormalised over the three
    surviving pairs: ``(2, 4, 8) / 14``.
    """
    residuals = np.array([[1.0], [-2.0], [3.0], [-4.0]])
    estimate = specific.time_series_estimate(
        residuals, halflife=1.0, autocorrelation_halflife=1.0, lags=1
    )

    weights = np.array([1.0, 2.0, 4.0, 8.0]) / 15.0
    c0 = float(weights @ np.array([1.0, 4.0, 9.0, 16.0]))
    pair_weights = np.array([2.0, 4.0, 8.0]) / 14.0
    products = np.array([-2.0 * 1.0, 3.0 * -2.0, -4.0 * 3.0])
    c1 = float(pair_weights @ products)
    expected = c0 + 0.5 * 2.0 * c1

    assert c0 == pytest.approx(173.0 / 15.0)  # 1 + 8 + 36 + 128 over 15
    assert c1 == pytest.approx((2.0 * -2.0 + 4.0 * -6.0 + 8.0 * -12.0) / 14.0)
    assert float(estimate.variance[0]) == pytest.approx(expected)
    assert float(estimate.deviation[0]) == pytest.approx(math.sqrt(expected))
    assert float(estimate.uncorrected_deviation[0]) == pytest.approx(math.sqrt(c0))


def test_the_two_half_lives_are_both_used() -> None:
    """SPEC.md 5.5(a) names two and the reading here uses both.

    A single-half-life implementation would return the same number whichever
    autocorrelation half-life it was handed, which is exactly what this refuses.
    """
    values = _panel()
    fast = specific.time_series_estimate(
        values, halflife=84.0, autocorrelation_halflife=21.0, lags=5
    )
    slow = specific.time_series_estimate(
        values, halflife=84.0, autocorrelation_halflife=252.0, lags=5
    )
    assert not np.allclose(fast.deviation, slow.deviation)
    # ...and the C_0 term is governed by the OTHER half-life, so it does not move.
    assert np.allclose(fast.uncorrected_deviation, slow.uncorrected_deviation)


def test_zero_lags_is_the_plain_ewma() -> None:
    values = _panel()
    estimate = specific.time_series_estimate(
        values, halflife=84.0, autocorrelation_halflife=252.0, lags=0
    )
    assert np.allclose(estimate.deviation, estimate.uncorrected_deviation)
    assert not estimate.fallback.fired


def test_a_non_positive_corrected_variance_falls_back_to_the_ewma() -> None:
    """experiments.md row 100's remedy, on the specific leg.

    A strictly alternating column has a large negative lag-1 autocovariance, so
    at enough lags the Bartlett sum drives the variance below zero. The fallback
    substitutes the uncorrected EWMA variance, reports it, and leaves the other
    columns untouched.
    """
    periods = 40
    alternating = np.array([(-1.0) ** t for t in range(periods)])
    values = np.column_stack([alternating, _panel(periods, 1, seed=SEED + 1)[:, 0]])
    estimate = specific.time_series_estimate(
        values, halflife=10.0, autocorrelation_halflife=10.0, lags=5
    )
    assert estimate.fallback.fired
    assert estimate.fallback.columns == (0,)
    assert estimate.fallback.corrected[0] <= 0.0
    assert estimate.fallback.substituted[0] > 0.0
    assert estimate.deviation[0] == pytest.approx(estimate.uncorrected_deviation[0])
    assert estimate.deviation[1] != pytest.approx(estimate.uncorrected_deviation[1])


# ---------------------------------------------------------------------------
# Leg (b)
# ---------------------------------------------------------------------------


def test_structural_fit_recovers_a_planted_log_linear_relationship() -> None:
    """When ``ln sigma`` really is linear in the exposures, the fit is exact.

    ``E_0`` is then 1 to rounding, because there is no log-transform bias to
    remove when the residuals are identically zero -- which is the check that
    ``E_0`` is a bias correction rather than a free level parameter.
    """
    rng = np.random.default_rng(SEED)
    exposures = rng.standard_normal((12, 3))
    truth = np.array([0.5, -0.25, 0.75])
    deviation = np.exp(-1.0 + exposures @ truth)
    fit = specific.structural_fit(deviation, exposures)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.coefficients[0] == pytest.approx(-1.0)
    assert np.allclose(fit.coefficients[1:], truth)
    assert fit.e_zero == pytest.approx(1.0)
    assert fit.e_zero_lognormal == pytest.approx(1.0)
    assert np.allclose(fit.deviation, deviation)
    assert fit.residual_degrees_of_freedom == 12 - 4


def test_e_zero_removes_the_jensen_bias_it_is_there_for() -> None:
    """``exp(mean(ln sigma)) < mean(sigma)``, and ``E_0`` closes exactly that gap."""
    rng = np.random.default_rng(SEED + 2)
    exposures = rng.standard_normal((20, 2))
    deviation = np.exp(rng.standard_normal(20))
    fit = specific.structural_fit(deviation, exposures)
    assert fit.e_zero > 1.0
    raw = fit.deviation / fit.e_zero
    assert float(raw.mean()) < float(deviation.mean())
    assert float(fit.deviation.mean()) == pytest.approx(float(deviation.mean()))


def test_the_structural_fit_is_applied_to_assets_it_was_not_fitted_on() -> None:
    """SPEC.md 5.5(b)'s whole point: a short history still gets a forecast."""
    rng = np.random.default_rng(SEED + 3)
    exposures = rng.standard_normal((12, 3))
    deviation = np.exp(exposures @ np.array([0.5, -0.25, 0.75]))
    mask = np.ones(12, dtype=bool)
    mask[-2:] = False
    fit = specific.structural_fit(deviation, exposures, fit_set=mask)
    assert not fit.fitted_on_every_asset
    assert fit.deviation.size == 12
    assert np.all(np.isfinite(fit.deviation[-2:]))
    assert fit.residual_degrees_of_freedom == 10 - 4


def test_a_fit_with_no_residual_freedom_is_refused() -> None:
    """At ``N = K + 1`` the fit interpolates and ``E_0`` is 1 by construction."""
    rng = np.random.default_rng(SEED + 4)
    with pytest.raises(specific.SpecificRiskError, match="residual degree"):
        specific.structural_fit(np.exp(rng.standard_normal(4)), rng.standard_normal((4, 3)))


# ---------------------------------------------------------------------------
# Leg (c)
# ---------------------------------------------------------------------------


def test_weighted_quantile_reduces_to_the_ordinary_one_at_equal_weights() -> None:
    values = _panel(101, 3)
    weights = np.full(values.shape[0], 1.0 / values.shape[0])
    for quantile in (0.25, 0.5, 0.75):
        assert np.allclose(
            specific.weighted_quantile(values, weights, quantile),
            np.quantile(values, quantile, axis=0),
        )


def test_the_robust_deviation_agrees_with_the_ewma_on_gaussian_data() -> None:
    """The IQR-to-sigma factor is derived, so the two estimators must agree.

    On a normal sample they estimate the same quantity, which is the check that
    the derived constant is the right one -- a transcribed 1.35 would land here
    within about half a percent and a wrong constant would not.
    """
    rng = np.random.default_rng(SEED + 5)
    values = rng.standard_normal((20_000, 2)) * np.array([1.0, 7.0])
    robust = specific.robust_deviation(values, halflife=1e6)
    raw = np.sqrt(specific.ewma_variance_diagonal(values, halflife=1e6))
    assert np.allclose(robust, raw, rtol=0.05)


def test_the_blend_weight_is_pinned_and_says_so() -> None:
    """Leg (c) is a RULING, not a computed weight, and the object carries that."""
    values = _panel()
    estimate = specific.time_series_estimate(
        values, halflife=84.0, autocorrelation_halflife=252.0, lags=5
    )
    blend = specific.blend_weight(values, estimate.deviation, halflife=84.0)
    assert blend.pinned
    assert np.array_equal(blend.gamma, np.ones(values.shape[1]))
    assert blend.observation_term_saturates
    assert blend.minimum_observations == values.shape[0]
    assert np.all(blend.z >= 0.0)


def test_the_fatness_term_is_measured_rather_than_assumed_saturated() -> None:
    """``Z`` responds to fat tails, which is the diagnostic's whole content.

    A normal panel sits near zero; the same panel with a handful of large
    excursions does not. If ``Z`` did not separate these two, reporting it as a
    reliability flag would be reporting nothing.
    """
    rng = np.random.default_rng(SEED + 6)
    thin = rng.standard_normal((2_000, 1))
    fat = thin.copy()
    fat[::200] *= 30.0
    deviations = [
        specific.blend_weight(
            panel,
            specific.time_series_estimate(
                panel, halflife=84.0, autocorrelation_halflife=252.0, lags=5
            ).deviation,
            halflife=84.0,
        ).maximum_z
        for panel in (thin, fat)
    ]
    assert deviations[0] < deviations[1]
    # ...and `fatness_term_saturates` answers about the bound it is HANDED,
    # because no bound is published (CLAUDE.md invariant 9).
    assert specific.blend_weight(
        thin,
        specific.time_series_estimate(
            thin, halflife=84.0, autocorrelation_halflife=252.0, lags=5
        ).deviation,
        halflife=84.0,
    ).fatness_term_saturates(1.0)


# ---------------------------------------------------------------------------
# Leg (d) -- the two claims SPEC.md 5.5 makes in prose
# ---------------------------------------------------------------------------


def test_shrinkage_is_monotone_further_from_the_bucket_mean_is_shrunk_less() -> None:
    """**The acceptance test.** SPEC.md 5.5(d), read the way it is written.

    Five assets in one bucket at increasing distance from its mean. ``v_n`` must
    increase with that distance -- the further out, the LESS the estimate is
    pulled back -- which is backwards from naive shrinkage and is the whole
    reason SPEC.md 5.5 spends a paragraph on it.
    """
    deviation = np.array([10.0, 11.0, 12.0, 16.0, 26.0])
    result = specific.bayesian_shrinkage(deviation, ["b"] * 5, q=0.1)
    distance = np.abs(deviation - deviation.mean())

    order = np.argsort(distance)
    ranked = result.v[order]
    assert np.all(np.diff(ranked) > 0.0), f"v is not monotone in distance: {result.v}"

    # The same claim stated in the units a reader cares about: the SHRINKAGE --
    # how far the estimate is actually moved, as a share of its distance from the
    # mean -- must FALL as the distance grows. That share is exactly ``1 - v``,
    # so this is the monotonicity above read the other way round rather than a
    # second fact, and it is written out because ``1 - v`` is the quantity a
    # report quotes.
    moved = np.abs(result.deviation - deviation) / distance
    assert moved == pytest.approx(1.0 - result.v)
    assert np.all(np.diff(moved[order]) < 0.0)


def test_shrinkage_hand_computed_on_a_three_asset_bucket() -> None:
    """Every number written out, so a formula change cannot pass quietly.

    ``sigma = (1, 2, 6)`` in one bucket: mean 3, distances ``(2, 1, 3)``, and a
    population dispersion of ``sqrt((4 + 1 + 9)/3) = sqrt(14/3)``.
    """
    deviation = np.array([1.0, 2.0, 6.0])
    result = specific.bayesian_shrinkage(deviation, ["b", "b", "b"], q=0.1)

    dispersion = math.sqrt(14.0 / 3.0)
    assert result.buckets.mean[0] == pytest.approx(3.0)
    assert result.buckets.dispersion[0] == pytest.approx(dispersion)

    expected_v = [
        2.0 / (2.0 + 0.1 * dispersion),
        1.0 / (1.0 + 0.1 * dispersion),
        3.0 / (3.0 + 0.1 * dispersion),
    ]
    assert result.v == pytest.approx(expected_v)
    # Pinned to a decimal as well, so that a change to the DERIVATION above and a
    # change to the code cannot cancel each other out.
    assert result.v[0] == pytest.approx(0.9025170, abs=1e-7)
    assert result.deviation == pytest.approx(
        [v * s + (1.0 - v) * 3.0 for v, s in zip(expected_v, deviation, strict=True)]
    )


def test_the_deviation_is_linear_absolute_and_not_the_squared_rendering() -> None:
    """SPEC.md 5.5: *"some PDF renderings garble this into a squared form"*.

    Both forms are monotone and both land in ``[0, 1]``, so this is pinned by
    computing the garbled one explicitly and requiring the implementation to
    differ from it.
    """
    deviation = np.array([1.0, 2.0, 6.0])
    result = specific.bayesian_shrinkage(deviation, ["b", "b", "b"], q=0.1)

    distance = np.abs(deviation - 3.0)
    dispersion = math.sqrt(14.0 / 3.0)
    garbled = distance**2 / (distance**2 + 0.1 * dispersion**2)

    assert not np.allclose(result.v, garbled)
    # And in the direction that matters: the garbled form shrinks the CLOSEST
    # estimate far harder, which is where the two would be told apart on data.
    assert garbled[1] < result.v[1]


def test_a_singleton_bucket_is_the_identity_whatever_v_is_filled_in() -> None:
    """``0/0`` is removable here: the map is the identity when the bucket is one."""
    deviation = np.array([5.0, 1.0, 3.0])
    result = specific.bayesian_shrinkage(deviation, ["alone", "pair", "pair"], q=0.1)
    assert result.buckets.singletons == ("alone",)
    assert result.identity_positions == (0,)
    assert result.deviation[0] == pytest.approx(5.0)
    assert result.buckets.dispersion[0] == pytest.approx(0.0)


def test_a_two_member_bucket_pins_v_at_one_over_one_plus_q_whatever_the_data() -> None:
    """The degeneracy at ``N_bucket = 2``, which is exact rather than empirical.

    In a two-member bucket each member's distance from the mean is half the gap
    and the population dispersion is *also* half the gap, so ``d == sigma_delta``
    identically and ``v = 1/(1+q)`` for **every** pair of values. Stage (d) at two
    members carries no information about the data at all: it is a fixed
    proportional pull toward the bucket mean.
    """
    for pair in ([1.0, 2.0], [0.2, 68.5], [7.0, 7.25]):
        result = specific.bayesian_shrinkage(np.array(pair), ["b", "b"], q=0.1)
        assert result.v == pytest.approx([1.0 / 1.1, 1.0 / 1.1])


def test_a_held_out_asset_does_not_enter_the_target_it_is_shrunk_toward() -> None:
    """W4-P1b. The exclusion removes a value from `sigma_bar` and `sigma_delta`.

    A bucket of three where one member is wildly out of line: holding it out of
    the target must leave the other two shrunk toward a mean computed from
    themselves alone, and must leave the held-out asset unshrunk (its target is
    then a one-contributor set, so `sigma_bar` differs from it and `sigma_delta`
    is zero -- `v = 1`).
    """
    deviation = np.array([10.0, 12.0, 900.0])
    mask = np.array([True, True, False])
    result = specific.bayesian_shrinkage(deviation, ["b"] * 3, q=0.1, target_set=mask)

    assert result.buckets.mean == pytest.approx([11.0, 11.0, 11.0])
    assert result.buckets.dispersion == pytest.approx([1.0, 1.0, 1.0])
    assert result.buckets.target_counts() == {"b": 2}
    assert result.buckets.counts() == {"b": 3}
    # The held-out asset is still bucketed, still shrunk, still returned -- it is
    # simply no longer part of the target it is measured against.
    assert result.deviation.size == 3
    assert result.deviation[2] < 900.0


def test_holding_an_asset_out_of_the_target_equals_deleting_it_for_everyone_else() -> None:
    """**The property W4-P1b rests on**, and the reason the fix is not a patch.

    A target computed from a set is the same number whether the non-contributors
    are present or absent. So every RETAINED asset must get bit-identical
    treatment under "held out of the target" and under "removed from the universe
    entirely" -- which is what makes the exclusion a removal of manufactured
    numbers rather than a re-weighting that happens to look better.
    """
    deviation = np.array([10.0, 12.0, 900.0, 5.0])
    labels = ["b", "b", "b", "other"]
    mask = np.array([True, True, False, True])

    held_out = specific.bayesian_shrinkage(deviation, labels, q=0.1, target_set=mask)
    deleted = specific.bayesian_shrinkage(deviation[mask], [labels[i] for i in (0, 1, 3)], q=0.1)

    assert held_out.deviation[mask] == pytest.approx(deleted.deviation, abs=0.0, rel=0.0)
    assert held_out.v[mask] == pytest.approx(deleted.v, abs=0.0, rel=0.0)


def test_a_two_member_bucket_with_one_held_out_becomes_a_no_op() -> None:
    """The consequence on this project's universe, asserted rather than observed.

    `credit` and `commodity` are two-member buckets whose second member is an
    identity asset. Holding it out leaves a one-contributor target, and stage (d)
    is then the identity for BOTH members -- the contributor because it is its own
    target, the held-out asset because `sigma_delta` is zero.
    """
    deviation = np.array([68.5, 0.22])
    result = specific.bayesian_shrinkage(
        deviation, ["b", "b"], q=0.1, target_set=np.array([True, False])
    )
    assert result.buckets.singletons == ("b",)
    assert result.deviation == pytest.approx(deviation)
    assert result.v == pytest.approx([1.0, 1.0])


def test_a_bucket_with_no_contributor_left_is_the_identity_and_says_so() -> None:
    """Handled rather than refused: a caller may exclude every member."""
    deviation = np.array([3.0, 7.0, 5.0])
    result = specific.bayesian_shrinkage(
        deviation, ["gone", "gone", "kept"], q=0.1, target_set=np.array([False, False, True])
    )
    assert result.buckets.targetless == ("gone",)
    assert result.deviation == pytest.approx(deviation)


def test_the_shrinkage_target_reaches_the_build_and_is_validated() -> None:
    """`shrinkage_target` names columns, and an unknown name is refused."""
    residuals, exposures, buckets = _inputs(120)
    config = RiskConfig.load(horizon="short")
    full = specific.build_specific_risk(residuals, exposures, buckets, config)
    held = specific.build_specific_risk(
        residuals, exposures, buckets, config, shrinkage_target=["s0", "s2", "s3", "s4"]
    )
    assert full.shrinkage.buckets.target_counts() != held.shrinkage.buckets.target_counts()
    assert not np.allclose(full.shrinkage.deviation, held.shrinkage.deviation)
    with pytest.raises(specific.SpecificRiskError, match="shrinkage_target"):
        specific.build_specific_risk(
            residuals, exposures, buckets, config, shrinkage_target=["nope"]
        )


def test_shrinkage_groups_by_label_equality_and_reads_nothing_else() -> None:
    """CLAUDE.md invariant 10: the labels are opaque and renaming cannot matter."""
    deviation = np.array([1.0, 2.0, 6.0, 9.0])
    first = specific.bayesian_shrinkage(deviation, ["x", "x", "y", "y"], q=0.1)
    second = specific.bayesian_shrinkage(deviation, ["zzz", "zzz", "aaa", "aaa"], q=0.1)
    assert np.allclose(first.deviation, second.deviation)


# ---------------------------------------------------------------------------
# The pipeline, the refusal, and the lead-lag
# ---------------------------------------------------------------------------


def _inputs(periods: int = 300) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    residuals = _frame(_panel(periods, 5))
    rng = np.random.default_rng(SEED + 7)
    exposures = pd.DataFrame(
        rng.standard_normal((5, 2)), index=residuals.columns, columns=["f0", "f1"]
    )
    buckets = pd.Series(["a", "a", "b", "b", "c"], index=residuals.columns)
    return residuals, exposures, buckets


@pytest.mark.parametrize("horizon", HORIZONS)
def test_the_build_refuses_to_publish_a_forecast_without_the_regime_stage(
    horizon: str,
) -> None:
    """The same mechanical separation ``CovarianceBuild.forecast`` makes.

    An un-adjusted specific volatility is positive, finite and correctly shaped;
    it is simply too small, which nothing downstream could detect.
    """
    residuals, exposures, buckets = _inputs()
    config = RiskConfig.load(horizon=horizon)  # type: ignore[arg-type]
    build = specific.build_specific_risk(residuals, exposures, buckets, config)
    assert build.regime is None
    assert np.all(build.shrinkage.deviation > 0.0)
    with pytest.raises(specific.SpecificRiskError, match="regime"):
        _ = build.forecast


def test_the_finished_forecast_is_the_shrunk_estimate_times_lambda() -> None:
    residuals, exposures, buckets = _inputs()
    config = RiskConfig.load(horizon="short")
    history = specific.specific_forecast_history(
        residuals, exposures, buckets, config, minimum_observations=6
    )
    build = specific.build_specific_risk(
        residuals, exposures, buckets, config, regime_bias=history.bias
    )
    assert build.regime is not None
    assert np.allclose(build.forecast, build.regime.lambda_ * build.shrinkage.deviation)
    assert list(build.series.index) == list(residuals.columns)


def test_specific_risk_matches_the_spec_15_2_signature() -> None:
    """Four positional arguments, ``pd.Series`` out. SPEC.md 15.2, verbatim."""
    residuals, exposures, buckets = _inputs()
    config = RiskConfig.load(horizon="short")
    history = specific.specific_forecast_history(
        residuals, exposures, buckets, config, minimum_observations=6
    )
    result = specific.specific_risk(residuals, exposures, buckets, config, regime_bias=history.bias)
    assert isinstance(result, pd.Series)
    assert list(result.index) == list(residuals.columns)
    assert (result > 0.0).all()


def test_the_forecast_history_cannot_see_its_own_realisation() -> None:
    """Perturb the return on date ``t``; every forecast must be unchanged.

    The leak this refuses would flatter ``B_t^S`` toward one, shrink
    ``lambda_S^2``, and silently under-scale every specific risk forecast. Same
    test, same reason, as the factor leg's in ``tests/test_regime.py``.
    """
    residuals, exposures, buckets = _inputs(120)
    config = RiskConfig.load(horizon="short")
    base = specific.specific_forecast_history(
        residuals, exposures, buckets, config, minimum_observations=6
    )

    perturbed = residuals.copy()
    position = 80
    perturbed.iloc[position, 0] += 500.0
    after = specific.specific_forecast_history(
        perturbed, exposures, buckets, config, minimum_observations=6
    )

    date = residuals.index[position]
    assert base.deviations.loc[date].equals(after.deviations.loc[date])
    assert base.bias[: base.index.get_loc(date)] == pytest.approx(
        after.bias[: after.index.get_loc(date)]
    )
    # ...and the perturbation IS visible afterwards, or the test would pass on a
    # function that ignored its input.
    assert not np.allclose(
        base.bias[base.index.get_loc(date) :], after.bias[base.index.get_loc(date) :]
    )


def test_the_excluded_assets_are_forecast_but_not_scored() -> None:
    """``include`` narrows the cross-section without narrowing the model."""
    residuals, exposures, buckets = _inputs(120)
    config = RiskConfig.load(horizon="short")
    history = specific.specific_forecast_history(
        residuals, exposures, buckets, config, minimum_observations=6, include=["s0", "s1", "s2"]
    )
    assert history.included == ("s0", "s1", "s2")
    assert history.excluded == ("s3", "s4")
    assert list(history.deviations.columns) == list(residuals.columns)


def test_a_misaligned_exposure_or_bucket_index_is_refused() -> None:
    residuals, exposures, buckets = _inputs(60)
    config = RiskConfig.load(horizon="short")
    with pytest.raises(specific.SpecificRiskError, match="exposure rows"):
        specific.build_specific_risk(residuals, exposures.iloc[::-1], buckets, config)
    with pytest.raises(specific.SpecificRiskError, match="bucket labels"):
        specific.build_specific_risk(residuals, exposures, buckets.iloc[::-1], config)


def test_a_non_finite_residual_is_refused_rather_than_filled() -> None:
    residuals, exposures, buckets = _inputs(60)
    residuals.iloc[3, 2] = np.nan
    config = RiskConfig.load(horizon="short")
    with pytest.raises(specific.SpecificRiskError, match="non-finite"):
        specific.build_specific_risk(residuals, exposures, buckets, config)


# ---------------------------------------------------------------------------
# The diagonal assumption
# ---------------------------------------------------------------------------


def test_the_residual_correlation_is_psd_and_ranks_its_own_violations() -> None:
    """SPEC.md 5.5's closing note: report where the diagonal assumption fails."""
    rng = np.random.default_rng(SEED + 8)
    common = rng.standard_normal((800, 1))
    values = np.column_stack(
        [
            common[:, 0] + 0.1 * rng.standard_normal(800),
            common[:, 0] + 0.1 * rng.standard_normal(800),
            rng.standard_normal(800),
        ]
    )
    frame = _frame(values)
    config = RiskConfig.load(horizon="short")
    correlation = specific.residual_correlation(frame, config=config)

    assert correlation.pair_count == 3
    assert np.allclose(np.diag(correlation.matrix), 1.0)
    top = correlation.pairs()[0]
    assert {top[0], top[1]} == {"s0", "s1"}
    assert top[2] > 0.9
    assert correlation.violations(0.5) == [top]
    assert correlation.violations(0.99) == []
