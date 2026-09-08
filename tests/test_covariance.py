"""The covariance pipeline, stage 1. SPEC.md 5.1, interface from SPEC.md 15.2.

Three things are pinned here and each is a decision that could otherwise drift:

* the arithmetic, against fractions computed by hand at a one-period half-life;
* the ZERO-MEAN convention (SPEC.md 5.1.1), by asserting what the estimator
  converges to and what it does NOT converge to;
* the SPEC.md 15.2 genericity test at K=3 and K=40, which is the early warning
  that ``mafrm.risk`` has stopped being asset-class agnostic.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mafrm.config import load
from mafrm.numerics import effective_sample_size
from mafrm.risk.checks import PsdError, eigenvalue_floor
from mafrm.risk.config import HORIZONS, RiskConfig
from mafrm.risk.covariance import (
    CovarianceError,
    assert_zero_mean_consistency,
    bartlett_newey_west,
    build_covariance,
    correlation_from_covariance,
    ewma_autocovariance,
    ewma_second_moment,
    implemented_stages,
    measured_not_asserted_stages,
    missing_stages,
    newey_west_covariance,
    psd_repair,
    run_pipeline,
    separated_ewma_covariance,
)
from mafrm.risk.synthetic import GOLDEN_FACTOR_COUNTS, panel, specification

# Three observations, two factors, oldest row first.
#
#   f = [[1, 0],
#        [0, 2],
#        [3, 1]]
#
# At halflife = 1 the normalised weights are [1/7, 2/7, 4/7] (tests/test_numerics.py).
_HAND = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, columns=[f"f{i}" for i in range(values.shape[1])])


#: SPEC.md 5.4's stage consumes a history of forecasts the CALLER builds, which
#: means running the whole pipeline once per date. Tests that are about stages 1
#: to 4 therefore stop after ``eigenfactor`` -- which is what ``stop_after`` is
#: for -- and the handful that genuinely exercise all five hand it this synthetic
#: bias series instead of paying for a real one. A perfectly calibrated model
#: gives B_t^F = 1 at every date, so this is the null history: it makes
#: ``lambda_F^2`` exactly 1 and leaves the matrix untouched, which is the right
#: neutral input for a test asking about some other stage's output.
_CALIBRATED_BIAS = np.ones(500)


# ---------------------------------------------------------------------------
# The arithmetic, by hand
# ---------------------------------------------------------------------------


def test_ewma_second_moment_hand_computed() -> None:
    """HAND-COMPUTED, moments about zero, weights [1/7, 2/7, 4/7].

    M_00 = 1/7*1 + 2/7*0 + 4/7*9 = (1 + 36)/7 = 37/7
    M_11 = 1/7*0 + 2/7*4 + 4/7*1 = (8 + 4)/7  = 12/7
    M_01 = 1/7*0 + 2/7*0 + 4/7*3 = 12/7
    """
    moment = ewma_second_moment(_HAND, halflife=1.0)
    np.testing.assert_allclose(moment, [[37 / 7, 12 / 7], [12 / 7, 12 / 7]], rtol=1e-14, atol=0)


def test_correlation_from_covariance_hand_computed() -> None:
    """HAND-COMPUTED. rho = (12/7) / sqrt(37/7 * 12/7) = sqrt(12/37) = 0.5694948..."""
    correlation = correlation_from_covariance(ewma_second_moment(_HAND, halflife=1.0))
    expected = np.sqrt(12.0 / 37.0)
    assert correlation[0, 1] == pytest.approx(expected, rel=1e-14)
    assert correlation[0, 1] == pytest.approx(0.5694947974, abs=1e-10)
    np.testing.assert_array_equal(np.diag(correlation), [1.0, 1.0])


def test_separated_covariance_hand_computed_with_two_different_halflives() -> None:
    """HAND-COMPUTED. F0_ij = rho_ij * sigma_i * sigma_j, SPEC.md 5.1.

    Volatility half-life 0.5 -> weights [1/21, 4/21, 16/21]:
        sigma_0^2 = (1*1 + 16*9)/21 = 145/21
        sigma_1^2 = (4*4 + 16*1)/21 =  32/21
    Correlation half-life 1 -> rho_01 = sqrt(12/37), from the test above.

    The variances therefore come from one half-life and the correlation from
    another, which is the entire content of SPEC.md 5.1.
    """
    covariance = separated_ewma_covariance(_HAND, volatility_halflife=0.5, correlation_halflife=1.0)
    var0, var1 = 145.0 / 21.0, 32.0 / 21.0
    expected_offdiagonal = np.sqrt(12.0 / 37.0) * np.sqrt(var0) * np.sqrt(var1)
    np.testing.assert_allclose(
        covariance,
        [[var0, expected_offdiagonal], [expected_offdiagonal, var1]],
        rtol=1e-14,
        atol=0,
    )


def test_equal_halflives_reduce_to_a_single_ewma() -> None:
    """The separation must be a generalisation, not a different estimator."""
    separated = separated_ewma_covariance(_HAND, volatility_halflife=3.0, correlation_halflife=3.0)
    np.testing.assert_allclose(
        separated, ewma_second_moment(_HAND, halflife=3.0), rtol=1e-13, atol=0
    )


def test_the_second_moment_is_exactly_symmetric() -> None:
    """Formed as a Gram matrix, so symmetry holds to the last bit, not to roundoff."""
    rng = np.random.default_rng(load().model.seed)
    values = rng.standard_normal((500, 12))
    moment = ewma_second_moment(values, halflife=84.0)
    assert np.max(np.abs(moment - moment.T)) == 0.0


# ---------------------------------------------------------------------------
# The zero-mean convention. SPEC.md 5.1.1.
# ---------------------------------------------------------------------------


def test_a_long_halflife_recovers_the_uncentred_sample_second_moment() -> None:
    """Moments about zero converge to X'X / T, and that is what is asserted."""
    rng = np.random.default_rng(load().model.seed)
    values = rng.standard_normal((400, 5)) + 0.5  # deliberately NOT mean zero
    moment = ewma_second_moment(values, halflife=1.0e12)
    np.testing.assert_allclose(moment, values.T @ values / len(values), rtol=1e-8, atol=0)


def test_it_does_not_converge_to_numpy_cov_and_the_gap_is_the_mean_outer_product() -> None:
    """The decision made visible: SPEC.md 5.1.1 subtracts no mean.

    ``np.cov`` demeans and divides by T-1. The exact identity between the two,
    which is what fails the moment somebody adds a demeaning step, is

        X'X / T  =  cov(X) * (T-1)/T  +  mean mean'
    """
    rng = np.random.default_rng(load().model.seed)
    values = rng.standard_normal((400, 5)) + 0.5
    periods = len(values)
    moment = ewma_second_moment(values, halflife=1.0e12)
    mean = values.mean(axis=0)
    centred = np.cov(values, rowvar=False) * (periods - 1) / periods + np.outer(mean, mean)
    np.testing.assert_allclose(moment, centred, rtol=1e-8, atol=0)

    # And the two are genuinely different at this mean, so the test has power.
    assert not np.allclose(moment, np.cov(values, rowvar=False), rtol=1e-3)


def test_there_is_no_reliability_correction_in_the_denominator() -> None:
    """A zero-mean estimator spends no degree of freedom, so weights sum to one.

    A ``1 - w'w`` denominator would inflate every variance by that factor. At a
    one-period half-life on three rows, w'w = 21/49, so the inflation would be
    a visible 75% rather than a rounding difference.
    """
    moment = ewma_second_moment(_HAND, halflife=1.0)
    inflation = 1.0 / (1.0 - 21.0 / 49.0)
    assert inflation == pytest.approx(1.75, rel=1e-12)
    assert moment[0, 0] == pytest.approx(37 / 7, rel=1e-14)
    assert moment[0, 0] != pytest.approx((37 / 7) * inflation, rel=1e-6)


def test_the_config_pins_the_convention_and_admits_no_other() -> None:
    from mafrm.config import ConfigError, _parse_covariance

    assert load().model.covariance.mean_convention == "zero"
    with pytest.raises(ConfigError, match="coherence requirement"):
        _parse_covariance({"covariance": {"mean_convention": "sample"}})


# ---------------------------------------------------------------------------
# SPEC.md 15.2 -- the genericity early warning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scaling", [1.0, 1.4])
@pytest.mark.parametrize("factors", GOLDEN_FACTOR_COUNTS)
@pytest.mark.parametrize("horizon", HORIZONS)
def test_the_pipeline_runs_at_k3_and_k40_and_returns_psd_of_the_right_shape(
    factors: int, horizon: str, scaling: float
) -> None:
    """SPEC.md 15.2's named test. A failure here is an ARCHITECTURE problem.

    The instruction is explicit: if this test fails, the module has started
    knowing what asset class it is looking at, and the fix is not a numerical
    one. K=3 stands in for the small model, K=40 for the large one.
    """
    spec = specification(factors)
    frame = panel(spec)
    config = RiskConfig.load(horizon=horizon).for_scaling(scaling)  # type: ignore[arg-type]

    matrix = build_covariance(frame, config, regime_bias=_CALIBRATED_BIAS)
    assert matrix.shape == (factors, factors)
    spectrum = np.linalg.eigvalsh(matrix)
    assert spectrum[0] >= eigenvalue_floor(
        len(spectrum), spectrum[-1], numerics=load().model.numerics
    )
    assert np.max(np.abs(matrix - matrix.T)) <= load().model.numerics.symmetry_absolute_tolerance


@pytest.mark.parametrize("factors", GOLDEN_FACTOR_COUNTS)
def test_the_fixture_recovers_the_known_covariance_within_its_own_sampling_error(
    factors: int,
) -> None:
    """The tolerance is DERIVED, not chosen.

    For T iid Gaussian rows with true covariance S, the uncentred second moment
    has Var(S_ij) = (S_ii S_jj + S_ij^2)/T, so the expected squared Frobenius
    error is the sum of those. Comparing the realised error against four times
    that RMS invents no tolerance: it is a distributional bound, and a real bug
    in either the volatility or the correlation half would blow past it.
    """
    spec = specification(factors)
    frame = panel(spec)
    truth = spec.covariance

    moment = ewma_second_moment(frame.to_numpy(), halflife=1.0e12)
    variances = np.diag(truth)
    expected_square = float(np.sum((np.outer(variances, variances) + truth**2) / spec.observations))
    assert np.linalg.norm(moment - truth, ord="fro") < 4.0 * np.sqrt(expected_square)


@pytest.mark.parametrize("factors", GOLDEN_FACTOR_COUNTS)
def test_the_fixture_has_a_non_diagonal_well_conditioned_truth(factors: int) -> None:
    """Guards the fixture itself: an iid generator could not detect a correlation bug."""
    spec = specification(factors)
    correlation = spec.correlation
    off = correlation[~np.eye(factors, dtype=bool)]
    assert np.max(np.abs(off)) > 0.2, "the fixture's true correlations are near zero"
    assert np.ptp(np.diag(spec.covariance)) > 0.5, "the fixture's true variances are near equal"
    # lambda_min(B B' + diag(psi)) >= min(psi) = 0.20 by construction, with
    # equality in the directions orthogonal to B -- so this is an equality test
    # at K=40 and eigvalsh returns 0.19999999999999992 for it.
    eigenvalues = np.linalg.eigvalsh(spec.covariance)
    assert eigenvalues[0] >= 0.2 - 1e-9
    assert eigenvalues[-1] / eigenvalues[0] < 1000.0


def test_the_fixture_window_is_long_relative_to_the_longest_halflife() -> None:
    """SPEC.md 15.2's second fixture requirement, asserted rather than asserted-to."""
    spec = specification(3)
    longest = max(
        load().model.covariance.factor_correlation_halflife.short,
        load().model.covariance.factor_correlation_halflife.long,
        load().model.covariance.factor_volatility_halflife.long,
    )
    assert spec.observations > 3.0 * effective_sample_size(float(longest))
    assert spec.observations >= 100 * max(GOLDEN_FACTOR_COUNTS)


# ---------------------------------------------------------------------------
# K/T_eff, emitted from this stage onward
# ---------------------------------------------------------------------------


def test_k_over_t_eff_is_emitted_and_matches_the_spec_table() -> None:
    """SPEC.md 15.2's table: K=6 at 252d -> T_eff 727, K/T 0.008, multiplier 1.017."""
    rng = np.random.default_rng(load().model.seed)
    frame = _frame(rng.standard_normal((3000, 6)))
    build = run_pipeline(frame, RiskConfig.load(horizon="long"), stop_after="ewma")

    assert build.factors == 6
    assert build.effective_sample_size == pytest.approx(727.1, abs=0.1)
    assert build.k_over_t_eff == pytest.approx(0.008, abs=0.0005)
    assert build.shepard_multiplier == pytest.approx(1.017, abs=0.001)
    assert any("K/T_eff" in line for line in build.render())
    assert any("Shepard" in line for line in build.render())


def test_shepard_multiplier_hand_computed() -> None:
    """HAND-COMPUTED. K/T_eff = 0.25 -> (1 - 0.25)^-2 = 16/9 = 1.7777..."""
    rng = np.random.default_rng(load().model.seed)
    factors = round(0.25 * effective_sample_size(84.0))
    frame = _frame(rng.standard_normal((2000, factors)))
    build = run_pipeline(frame, RiskConfig.load(horizon="short"), stop_after="ewma")
    assert build.k_over_t_eff == pytest.approx(0.25, abs=0.002)
    assert build.shepard_multiplier == pytest.approx(16.0 / 9.0, rel=0.01)


def test_realised_effective_sample_size_is_reported_separately_on_a_short_window() -> None:
    """A short window realises less than 2*tau/ln2 and the build says so."""
    rng = np.random.default_rng(load().model.seed)
    frame = _frame(rng.standard_normal((120, 4)))
    build = run_pipeline(frame, RiskConfig.load(horizon="long"), stop_after="ewma")
    assert build.realised_effective_sample_size < build.effective_sample_size
    assert build.k_over_realised_t_eff > build.k_over_t_eff


# ---------------------------------------------------------------------------
# Stage declaration
# ---------------------------------------------------------------------------


def test_declared_stages_are_spec_5s_order() -> None:
    assert load().model.covariance.stages == (
        "ewma",
        "newey_west",
        "psd_repair",
        "eigenfactor",
        "volatility_regime",
    )


def test_every_implemented_stage_is_declared() -> None:
    """The other direction is allowed to be non-empty; this one never is."""
    declared = set(load().model.covariance.stages)
    assert set(implemented_stages()) <= declared


def test_the_build_knows_it_is_incomplete() -> None:
    """A partial pipeline is not a finished forecast and the object must not pretend."""
    config = RiskConfig.load(horizon="short").for_scaling(1.0)
    rng = np.random.default_rng(load().model.seed)
    values = _frame(rng.standard_normal((600, 4)))
    build = run_pipeline(values, config, regime_bias=_CALIBRATED_BIAS)
    assert set(build.stages_not_implemented) == set(missing_stages(config))
    assert build.complete == (not build.stages_not_run)
    if build.stages_not_implemented:
        assert "NOT IMPLEMENTED" in "\n".join(build.render())

    # A build a diagnostic stopped short is incomplete for the same reason and
    # says so differently, so a truncated matrix cannot pass as a forecast.
    stopped = run_pipeline(values, config, stop_after="ewma")
    assert not stopped.complete
    assert stopped.stages_not_run == (
        "newey_west",
        "psd_repair",
        "eigenfactor",
        "volatility_regime",
    )
    assert "STOPPED AFTER ewma" in "\n".join(stopped.render())


def test_an_incomplete_build_refuses_to_hand_out_a_forecast() -> None:
    """GUARD (W3-P3b). A published number cannot come from a partial pipeline.

    ``.matrix`` is the last stage that ran and is legitimately partial -- that is
    what a stage diagnostic needs. ``.forecast`` is what anything PUBLISHING a
    number must use, and it raises unless every declared stage ran. The two are
    separated mechanically rather than by a docstring warning, because the failure
    is silent: a matrix missing SPEC.md 5.4's regime adjustment is still PSD,
    symmetric and identically conditioned. It is simply too small.

    It raises today at every horizon, which is correct: ``make model`` is red for
    the same reason, and no report in this repository quotes a risk forecast.
    """
    config = RiskConfig.load(horizon="short").for_scaling(1.0)
    rng = np.random.default_rng(load().model.seed)
    values = _frame(rng.standard_normal((600, 4)))

    build = run_pipeline(values, config, stop_after="eigenfactor")
    assert build.matrix.shape == (4, 4)
    with pytest.raises(CovarianceError, match="INCOMPLETE"):
        _ = build.forecast

    # And a build a diagnostic stopped short says WHERE it stopped, so the two
    # kinds of incompleteness are distinguishable in the message.
    stopped = run_pipeline(values, config, stop_after="ewma")
    with pytest.raises(CovarianceError, match="stopped after 'ewma'"):
        _ = stopped.forecast


def test_psd_is_observed_after_every_stage_that_ran() -> None:
    """Invariant 4: one observation per stage, each naming its stage.

    "Observed", not "asserted", since W3-P2: ``newey_west`` reports its spectrum
    and the assertion runs after ``psd_repair``. Which stages get which is pinned
    separately by :func:`test_exactly_one_stage_is_measured_without_being_asserted`.
    """
    rng = np.random.default_rng(load().model.seed)
    build = run_pipeline(
        _frame(rng.standard_normal((600, 4))),
        RiskConfig.load(horizon="short").for_scaling(1.0),
        regime_bias=_CALIBRATED_BIAS,
    )
    observed = [stage.check or stage.spectrum for stage in build.stages]
    assert len(observed) == len(build.stages)
    assert all(
        item is not None and item.stage.startswith(name)
        for item, name in zip(observed, [stage.name for stage in build.stages], strict=True)
    )
    assert all(item is not None and item.size == 4 for item in observed)


# ---------------------------------------------------------------------------
# Input contracts
# ---------------------------------------------------------------------------


def test_a_constant_column_is_refused_rather_than_divided_by_zero() -> None:
    values = np.column_stack([np.zeros(300), np.random.default_rng(0).standard_normal(300)])
    with pytest.raises(CovarianceError, match="non-positive estimated variance"):
        separated_ewma_covariance(values, volatility_halflife=84.0, correlation_halflife=504.0)


def test_missing_observations_are_refused_not_guessed() -> None:
    values = np.array([[1.0, 2.0], [np.nan, 1.0], [0.5, 0.5]])
    with pytest.raises(CovarianceError, match="non-finite"):
        ewma_second_moment(values, halflife=2.0)


def test_fewer_observations_than_factors_is_refused() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(CovarianceError, match="singular by construction"):
        run_pipeline(_frame(rng.standard_normal((3, 5))), RiskConfig.load(horizon="short"))


def test_duplicate_factor_columns_are_refused() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.standard_normal((300, 2)), columns=["a", "a"])
    with pytest.raises(CovarianceError, match="duplicate factor column"):
        run_pipeline(frame, RiskConfig.load(horizon="short"))


def test_the_frame_view_carries_the_input_labels() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.standard_normal((300, 3)), columns=["alpha", "beta", "gamma"])
    build = run_pipeline(
        frame, RiskConfig.load(horizon="short").for_scaling(1.0), regime_bias=_CALIBRATED_BIAS
    )
    assert list(build.frame.columns) == ["alpha", "beta", "gamma"]
    assert list(build.frame.index) == ["alpha", "beta", "gamma"]


def test_an_asymmetric_matrix_would_be_caught_by_the_stage_assertion() -> None:
    """Belt and braces: the assertion the pipeline calls does reject one."""
    from mafrm.risk.checks import assert_psd

    with pytest.raises(PsdError, match="not symmetric"):
        assert_psd(np.array([[1.0, 0.2], [0.9, 1.0]]), "ewma [short]")


# ---------------------------------------------------------------------------
# Stage 2 -- SPEC.md 5.2, Newey-West. The arithmetic, by hand.
# ---------------------------------------------------------------------------


def test_ewma_autocovariance_at_lag_zero_is_the_stage_one_estimator() -> None:
    """Not "agrees with" -- IS. The same object, so the two cannot drift apart.

    SPEC.md 5.1.1's zero-mean convention has to carry into the lag terms, and the
    cheapest way to guarantee that for the lag-0 term is to have one
    implementation of it rather than two that are written the same way.
    """
    stage_one = ewma_second_moment(_HAND, halflife=1.0)
    np.testing.assert_array_equal(ewma_autocovariance(_HAND, halflife=1.0, lag=0), stage_one)


def test_ewma_autocovariance_hand_computed_at_lag_one() -> None:
    """HAND-COMPUTED, moments about zero, T = 3 at halflife 1.

    Weights over the whole window are [1/7, 2/7, 4/7]. At lag 1 the pairs are
    (f_1, f_0) and (f_2, f_1); each takes the weight of its MORE RECENT member,
    so the surviving weights are [2/7, 4/7], renormalised to [1/3, 2/3].

        C_1[0,0] = 1/3 * 0*1 + 2/3 * 3*0 = 0
        C_1[0,1] = 1/3 * 0*0 + 2/3 * 3*2 = 4
        C_1[1,0] = 1/3 * 2*1 + 2/3 * 1*0 = 2/3
        C_1[1,1] = 1/3 * 2*0 + 2/3 * 1*2 = 4/3
    """
    cross = ewma_autocovariance(_HAND, halflife=1.0, lag=1)
    np.testing.assert_allclose(cross, [[0.0, 4.0], [2 / 3, 4 / 3]], rtol=1e-14, atol=1e-15)


def test_the_autocovariance_is_directional_and_the_bartlett_sum_symmetrises_it() -> None:
    """``C_delta`` is not symmetric; cross-serial-correlation has a direction."""
    cross = ewma_autocovariance(_HAND, halflife=1.0, lag=1)
    assert not np.allclose(cross, cross.T)
    total = bartlett_newey_west(_HAND, halflife=1.0, lags=1)
    np.testing.assert_array_equal(total, total.T)


def test_bartlett_newey_west_hand_computed() -> None:
    """HAND-COMPUTED. F_NW = C_0 + (1 - 1/2)(C_1 + C_1') at L = 1.

        C_0        = [[37/7, 12/7], [12/7, 12/7]]      (the stage-1 test above)
        C_1 + C_1' = [[0, 14/3], [14/3, 8/3]]
        F_NW       = [[37/7, 12/7 + 7/3], [12/7 + 7/3, 12/7 + 4/3]]
                   = [[37/7, 85/21], [85/21, 64/21]]

    Note what this little matrix is: ``det = (37/7)(64/21) - (85/21)^2 =
    -121/441 < 0``. **The hand-computed example is itself non-PSD**, which is the
    cheapest possible demonstration that SPEC.md 5.2's repair is not decorative.
    """
    total = bartlett_newey_west(_HAND, halflife=1.0, lags=1)
    np.testing.assert_allclose(
        total, [[37 / 7, 85 / 21], [85 / 21, 64 / 21]], rtol=1e-14, atol=1e-15
    )
    assert np.linalg.det(total) == pytest.approx(-121 / 441, rel=1e-12)
    assert np.linalg.eigvalsh(total)[0] < 0.0


def test_zero_lags_is_exactly_the_ewma_second_moment() -> None:
    """L = 0 deletes the sum, leaving C_0. The Bartlett kernel's boundary case."""
    np.testing.assert_array_equal(
        bartlett_newey_west(_HAND, halflife=1.0, lags=0),
        ewma_second_moment(_HAND, halflife=1.0),
    )


def test_the_bartlett_kernel_declines_linearly_to_zero_at_l_plus_one() -> None:
    """HAND-COMPUTED weights. At L = 5 they are 5/6, 4/6, 3/6, 2/6, 1/6.

    Recovered from the estimator rather than read off the source: build the sum
    at each L and difference it, which isolates one lag's contribution.
    """
    rng = np.random.default_rng(load().model.seed)
    values = rng.standard_normal((400, 2))
    contributions = []
    for lags in range(6):
        contributions.append(bartlett_newey_west(values, halflife=400.0, lags=lags))
    # The lag-1 term appears in every sum from L = 1 on, with a kernel that moves,
    # so compare L = 1 (kernel 1/2) against a hand-built one-lag sum.
    cross = ewma_autocovariance(values, halflife=400.0, lag=1)
    expected = contributions[0] + 0.5 * (cross + cross.T)
    np.testing.assert_allclose(contributions[1], expected, rtol=1e-13)


# ---------------------------------------------------------------------------
# The zero-mean convention carries into the lag terms. SPEC.md 5.1.1.
# ---------------------------------------------------------------------------


def test_the_lag_path_and_stage_one_agree_at_lag_zero_on_real_scale_input() -> None:
    """The consistency check passes, and by a margin far inside its bound."""
    rng = np.random.default_rng(load().model.seed)
    values = rng.standard_normal((2000, 4)) * np.array([1.0, 0.01, 100.0, 5.0])
    discrepancy = assert_zero_mean_consistency(values, halflife=84.0)
    assert discrepancy >= 0.0
    assert discrepancy < 1e-9


def test_the_consistency_check_catches_a_demeaned_lag_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POWER TEST. A check that cannot detect its own failure mode is not a check.

    CLAUDE.md failure mode 7 is corrections that each pass their own sanity check
    and interact badly. If a later session demeans the lag path while stage 1
    stays zero-mean, the pipeline estimates two different quantities and only
    ``B`` would show it -- weeks later. This asserts the tripwire fires.
    """
    import mafrm.risk.covariance as module

    original = module._weighted_lagged_moment

    def demeaned(returns: np.ndarray, weights: np.ndarray, lag: int) -> np.ndarray:
        centred = returns - (weights @ returns)
        return original(centred, weights, lag)

    monkeypatch.setattr(module, "_weighted_lagged_moment", demeaned)
    rng = np.random.default_rng(load().model.seed)
    drifting = rng.standard_normal((500, 3)) + 0.5
    with pytest.raises(CovarianceError, match="about ZERO"):
        module.assert_zero_mean_consistency(drifting, halflife=84.0)


def test_the_stage_refuses_a_convention_other_than_zero() -> None:
    """The stage will not silently inherit a convention it was not written for."""
    risk = replace(RiskConfig.load(horizon=HORIZONS[0]), mean_convention="sample")  # type: ignore[arg-type]
    frame = _frame(np.random.default_rng(load().model.seed).standard_normal((200, 3)))
    with pytest.raises(CovarianceError, match="mean_convention"):
        run_pipeline(frame, risk)


# ---------------------------------------------------------------------------
# What Newey-West does to serial correlation, in the right direction.
# ---------------------------------------------------------------------------


def _ar1(phi: float, periods: int, factors: int) -> np.ndarray:
    """A seeded AR(1) panel. ``model.seed`` through ``default_rng``, never global."""
    innovations = np.random.default_rng(load().model.seed).standard_normal((periods, factors))
    series = np.zeros((periods, factors))
    for step in range(1, periods):
        series[step] = phi * series[step - 1] + innovations[step]
    return series


#: Three sigma on the Bartlett sum of sample autocovariances, DERIVED not chosen.
#: Each ``Gamma_d`` has standard error ~ ``1/sqrt(T_eff)``; the sum weights them by
#: ``2 * kernel_d``, so the sum's standard deviation is
#: ``sqrt(4 * sum(kernel_d^2) / T_eff)`` = ``sqrt(4 * 1.5278 / 5040)`` = 0.035.
#: The bound below is three of those. It is a rounding-and-sampling bound, not a
#: tolerance widened until a result fitted inside it.
_SAMPLING_BOUND = 0.105


def test_serially_uncorrelated_input_leaves_the_estimate_approximately_unchanged() -> None:
    """The identity case. With no serial correlation the lag terms are noise."""
    values = np.random.default_rng(load().model.seed).standard_normal((5040, 4))
    plain = ewma_second_moment(values, halflife=5040.0)
    corrected = bartlett_newey_west(values, halflife=5040.0, lags=5)
    ratio = np.diag(corrected) / np.diag(plain)
    assert np.all(np.abs(ratio - 1.0) < _SAMPLING_BOUND), ratio
    assert np.max(np.abs(corrected - plain)) / np.max(np.abs(plain)) < _SAMPLING_BOUND


@pytest.mark.parametrize("phi", [0.5, 0.3, -0.5])
def test_an_ar1_panel_moves_the_variance_to_its_closed_form_bartlett_value(phi: float) -> None:
    """HAND-COMPUTABLE. For an AR(1), ``Gamma_d / Gamma_0 = phi^d`` exactly, so

        F_NW / F_0 = 1 + 2 * sum_{d=1..5} (1 - d/6) * phi^d

    which is 2.3438 at phi = 0.5, 1.6532 at 0.3 and 0.4062 at -0.5. The direction
    is the point: positive serial correlation INCREASES the variance estimate and
    negative serial correlation decreases it, which is the whole reason SPEC.md
    5.2 exists. Asserting the level rather than only the sign is what makes this
    a test of the Bartlett weights and not just of the sign of a sum.
    """
    expected = 1.0 + 2.0 * sum((1.0 - lag / 6.0) * phi**lag for lag in range(1, 6))
    values = _ar1(phi, periods=5040, factors=2)
    plain = ewma_second_moment(values, halflife=5040.0)
    corrected = bartlett_newey_west(values, halflife=5040.0, lags=5)
    ratio = np.diag(corrected) / np.diag(plain)
    np.testing.assert_allclose(ratio, expected, rtol=0.10)
    if phi > 0:
        assert np.all(ratio > 1.0)
    else:
        assert np.all(ratio < 1.0)


# ---------------------------------------------------------------------------
# SEPARATED, not assembled. SPEC.md 5.2's ruling.
# ---------------------------------------------------------------------------


def test_newey_west_is_applied_separately_to_the_volatility_and_correlation_terms() -> None:
    """The structure, pinned. Two corrections at two half-lives and two lag counts.

    If the correction were applied once to an assembled ``F_0`` there would be one
    lag number in USE4 Table 4.1, not two. This asserts the shipped estimator is
    exactly the recombination of two independently corrected terms.
    """
    values = np.random.default_rng(load().model.seed).standard_normal((600, 3))
    volatility = bartlett_newey_west(values, halflife=84.0, lags=5)
    correlation = correlation_from_covariance(bartlett_newey_west(values, halflife=504.0, lags=2))
    deviation = np.sqrt(np.diag(volatility))
    expected = correlation * np.outer(deviation, deviation)

    actual, fallback = newey_west_covariance(
        values,
        volatility_halflife=84.0,
        volatility_lags=5,
        correlation_halflife=504.0,
        correlation_lags=2,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-14)
    assert not fallback.fired, "row 100's remedy must not fire on well-behaved returns"


def test_the_separated_form_is_not_the_same_as_one_correction_on_an_assembled_matrix() -> None:
    """The two readings genuinely differ, so the ruling is not moot.

    Not a configuration comparison and not an evaluation: no criterion is applied
    to either result. It asserts only that the distinction the ruling draws is a
    distinction, which a test that could not tell them apart would not establish.
    """
    values = np.random.default_rng(load().model.seed).standard_normal((600, 3))
    separated, _ = newey_west_covariance(
        values,
        volatility_halflife=84.0,
        volatility_lags=5,
        correlation_halflife=504.0,
        correlation_lags=2,
    )
    assembled_at_one_halflife = bartlett_newey_west(values, halflife=84.0, lags=5)
    relative = np.max(np.abs(separated - assembled_at_one_halflife)) / np.max(
        np.abs(assembled_at_one_halflife)
    )
    assert relative > 0.01


def test_the_correlation_diagonal_is_renormalised_so_no_variance_is_counted_twice() -> None:
    """THE DOUBLE-COUNTING TRAP. SPEC.md 5.2's ruling.

    Newey-West does not preserve a unit diagonal. Two assertions:

    * correcting an already-normalised ``rho`` leaves a diagonal that is NOT one,
      which is the trap -- demonstrated here rather than described;
    * the shipped covariance's diagonal is EXACTLY the corrected volatility
      variance, so the correlation leg contributes no second copy of it.
    """
    values = np.random.default_rng(load().model.seed).standard_normal((600, 3))

    naive = correlation_from_covariance(ewma_second_moment(values, halflife=504.0))
    corrected_correlation = bartlett_newey_west(naive, halflife=504.0, lags=2)
    assert not np.allclose(np.diag(corrected_correlation), 1.0, atol=1e-6)

    shipped, _ = newey_west_covariance(
        values,
        volatility_halflife=84.0,
        volatility_lags=5,
        correlation_halflife=504.0,
        correlation_lags=2,
    )
    volatility = bartlett_newey_west(values, halflife=84.0, lags=5)
    np.testing.assert_allclose(np.diag(shipped), np.diag(volatility), rtol=1e-14)


#: Seed and window that provoke SPEC.md 5.2's non-positive variance. Found by an
#: exhaustive scan over seeds rather than chosen, and pinned so the fixture is
#: reproducible; ``test_a_negative_variance_falls_back...`` asserts it still
#: provokes the failure rather than trusting this comment.
_BARTLETT_FAILURE_SEED = 5
_BARTLETT_FAILURE_WINDOW = 7


def _panel_that_breaks_the_bartlett_correction() -> np.ndarray:
    """A synthetic panel the Bartlett correction drives non-positive. No model attached.

    **It is plain iid standard normal noise**, seven observations against five
    lags. Nothing exotic is required and that is itself informative: the failure
    is a property of the window length against the lag count, not of any special
    serial-correlation structure in the data. Seven observations give the
    longest lag two overlapping pairs, and two pairs is not an estimate -- which
    is the same reading as W3-P5's measurement that the failure fires at
    ``T = 7..10`` and never again (``K/T`` around 0.43, the shape of SPEC.md
    5.2's PSD repair).

    **It is not Model A's factors, not Model B's, and not the golden fixture** --
    it is a ``numpy`` construction with no asset class, no calendar and no
    economics in it, which is the whole point of the agnosticism test below
    (``experiments.md`` row 129).
    """
    generator = np.random.default_rng(_BARTLETT_FAILURE_SEED)
    return np.asarray(generator.standard_normal((_BARTLETT_FAILURE_WINDOW, 3)))


def test_a_negative_variance_falls_back_to_the_uncorrected_ewma_variance() -> None:
    """``experiments.md`` row 100's registered remedy, as applied in W3-P5.

    Registered in W3-P2 for ``K = 56`` and amended in W3-P5 when it fired at
    ``K = 3`` instead: the remedy attaches to the MECHANISM -- exponential weights
    break the quadratic form Newey & West's non-negativity guarantee rests on --
    and not to a panel size. **A floor is still refused**; what happens instead is
    a fall back to the stage-1 EWMA variance, which invents no constant because
    the pipeline already computes it.
    """
    panel = _panel_that_breaks_the_bartlett_correction()
    raw = np.diag(bartlett_newey_west(panel, halflife=84.0, lags=5))
    assert np.any(raw <= 0.0), "the fixture no longer provokes the failure it exists to provoke"

    matrix, fallback = newey_west_covariance(
        panel,
        volatility_halflife=84.0,
        volatility_lags=5,
        correlation_halflife=504.0,
        correlation_lags=2,
    )
    assert fallback.fired
    assert fallback.columns == tuple(int(i) for i in np.flatnonzero(raw <= 0.0))
    assert all(value <= 0.0 for value in fallback.corrected)
    assert all(value > 0.0 for value in fallback.substituted)

    # The substituted value is stage 1's, exactly -- not a floor, not a rescale.
    stage_one = np.diag(ewma_second_moment(panel, halflife=84.0))
    for column, substituted in zip(fallback.columns, fallback.substituted, strict=True):
        assert substituted == pytest.approx(stage_one[column], rel=1e-15)
    np.testing.assert_allclose(
        np.diag(matrix)[list(fallback.columns)], stage_one[list(fallback.columns)], rtol=1e-14
    )
    assert np.all(np.diag(matrix) > 0.0)


def test_the_fallback_leaves_every_other_variance_bit_identical() -> None:
    """It substitutes for the failing factors only, and changes nothing else.

    This is what made the W3-P5 change safe to apply to a half-finished rebuild:
    histories whose dates never fired are unaffected, so they did not have to be
    recomputed. A remedy that perturbed the untouched columns would have
    invalidated every partial on disk.
    """
    panel = _panel_that_breaks_the_bartlett_correction()
    raw = np.diag(bartlett_newey_west(panel, halflife=84.0, lags=5))
    matrix, fallback = newey_west_covariance(
        panel,
        volatility_halflife=84.0,
        volatility_lags=5,
        correlation_halflife=504.0,
        correlation_lags=2,
    )
    untouched = [index for index in range(panel.shape[1]) if index not in fallback.columns]
    assert untouched, "the fixture must leave at least one factor for this test to be about"
    assert fallback.fired, "the fixture no longer provokes the failure it exists to provoke"
    np.testing.assert_allclose(np.diag(matrix)[untouched], raw[untouched], rtol=1e-14)


def test_the_fallback_is_asset_class_agnostic_and_that_is_DEMONSTRATED() -> None:
    """SPEC.md 15.2's falsifier fired in W3-P5. This is the sharpened gate.

    Row 125 registered that **any** change under ``src/mafrm/risk/`` to
    accommodate a statistical factor set is the SPEC.md 15.2 early warning
    firing. Row 100's remedy is such a change. The trigger was a **true
    positive**; the inference -- "therefore an architecture problem" -- was a
    **false positive**, and the operator's ruling was that this be settled by
    test rather than by argument, because a gate waived once on a good argument
    is waived later on a worse one.

    So: the fallback is exercised on a panel unrelated to either model, and is
    required to behave identically -- same firing columns, same substituted
    values, same untouched columns. ``mafrm.risk`` cannot condition on what its
    columns are, and this demonstrates that where the change was made rather than
    asserting it. ``tests/test_risk_architecture.py`` continues to assert the
    complementary half: no file under ``risk/`` names an asset class or imports
    from ``mafrm.factors``.
    """
    panel = _panel_that_breaks_the_bartlett_correction()

    # The same numbers relabelled and rescaled per column. Nothing about the
    # estimator may depend on a column's identity, so the firing pattern must be
    # invariant to which column is which.
    permutation = [2, 0, 1]
    permuted = panel[:, permutation]

    _, original = newey_west_covariance(
        panel,
        volatility_halflife=84.0,
        volatility_lags=5,
        correlation_halflife=504.0,
        correlation_lags=2,
    )
    _, relabelled = newey_west_covariance(
        permuted,
        volatility_halflife=84.0,
        volatility_lags=5,
        correlation_halflife=504.0,
        correlation_lags=2,
    )
    assert original.fired and relabelled.fired
    expected = sorted(permutation.index(column) for column in original.columns)
    assert sorted(relabelled.columns) == expected
    np.testing.assert_allclose(
        sorted(relabelled.substituted), sorted(original.substituted), rtol=1e-12
    )


# ---------------------------------------------------------------------------
# Stage 3 -- SPEC.md 5.2's mandatory PSD repair.
# ---------------------------------------------------------------------------


def test_psd_repair_hand_computed() -> None:
    """HAND-COMPUTED. [[1, 2], [2, 1]] has eigenvalues 3 and -1.

    Eigenvectors are ``(1, 1)/sqrt(2)`` and ``(1, -1)/sqrt(2)``. Flooring -1 to
    ``e = 1e-14`` and rebuilding gives

        3 * (1/2)[[1, 1], [1, 1]]  +  e * (1/2)[[1, -1], [-1, 1]]
        = [[1.5 + e/2, 1.5 - e/2], [1.5 - e/2, 1.5 + e/2]]

    i.e. [[1.5, 1.5], [1.5, 1.5]] to within 5e-15.
    """
    floor = load().model.numerics.psd_eigenvalue_floor
    repaired, report = psd_repair(np.array([[1.0, 2.0], [2.0, 1.0]]), floor=floor)

    np.testing.assert_allclose(repaired, [[1.5, 1.5], [1.5, 1.5]], atol=1e-14)
    np.testing.assert_allclose(np.linalg.eigvalsh(repaired), [floor, 3.0], atol=1e-14)
    assert report.fired
    assert report.floored == 1
    assert report.minimum_eigenvalue_before == pytest.approx(-1.0)
    assert report.maximum_eigenvalue_before == pytest.approx(3.0)
    assert report.floor == floor


def test_the_repair_is_a_true_no_op_when_it_does_not_fire() -> None:
    """Not round-tripped through its own eigendecomposition for nothing.

    A no-op that perturbs the last few bits would make the golden fixture measure
    LAPACK's reconstruction error rather than the estimator.
    """
    matrix = np.array([[4.0, 1.0], [1.0, 9.0]])
    repaired, report = psd_repair(matrix, floor=load().model.numerics.psd_eigenvalue_floor)
    assert repaired is matrix
    assert not report.fired
    assert report.floored == 0


def test_the_repair_floor_comes_from_the_config_and_is_invariant_4s_value() -> None:
    """CLAUDE.md invariant 4 and invariant 6. The floor is read, not written.

    Behavioural rather than a grep for the literal: the repair is run at a floor
    the config does not contain, and the output has to use it. A hard-coded 1e-14
    fails this; a docstring mentioning 1e-14 does not, which is the right way
    round.
    """
    assert load().model.numerics.psd_eigenvalue_floor == 1e-14
    repaired, report = psd_repair(np.array([[1.0, 2.0], [2.0, 1.0]]), floor=0.25)
    assert report.floor == 0.25
    np.testing.assert_allclose(np.linalg.eigvalsh(repaired), [0.25, 3.0], atol=1e-12)


def test_the_repair_reports_the_floor_against_the_spectrum_not_only_in_absolute_terms() -> None:
    """The floor is ABSOLUTE and therefore scale-dependent. SPEC.md 5.3.1's problem.

    1e-14 is safe on this project's matrices by eight orders of magnitude, but
    that is an accident of the unit convention rather than a property of the
    constant. The ratio to ``lambda_max`` is what makes it readable, and
    W3-P3's numeraire decision changes what it means.
    """
    floor = load().model.numerics.psd_eigenvalue_floor
    _, report = psd_repair(np.array([[1.0, 2.0], [2.0, 1.0]]), floor=floor)
    assert report.floor_relative_to_maximum == pytest.approx(floor / 3.0)
    assert "of lambda_max" in report.render()


#: K = 6 factors from T = 12 observations, drawn from ``model.seed``. This is the
#: regime in which the repair actually fires on this project's own data --
#: ``reports/psd_repairs.md`` finds every firing at T <= 10 against K = 6 -- and it
#: is reproduced synthetically here so the test does not depend on ``data/raw``,
#: which is gitignored. The margin is ~1e-3 of ``lambda_max``, far above roundoff,
#: so the test does not turn on which BLAS is installed.
_FORCES_REPAIR = (12, 6)


@pytest.mark.parametrize("horizon", HORIZONS)
def test_the_pipeline_repairs_a_case_that_forces_a_non_psd_newey_west(horizon: str) -> None:
    """END TO END. The repair stage is live, and the assertion after it passes.

    Three things at once: ``newey_west`` really can emit a non-PSD matrix,
    ``psd_repair`` really does floor it, and the invariant-4 assertion that runs
    after the repair really does pass on the result.
    """
    periods, factors = _FORCES_REPAIR
    values = np.random.default_rng(load().model.seed).standard_normal((periods, factors))
    build = run_pipeline(
        _frame(values),
        RiskConfig.load(horizon=horizon),  # type: ignore[arg-type]
        stop_after="psd_repair",
    )

    stages = {stage.name: stage for stage in build.stages}
    assert stages["newey_west"].minimum_eigenvalue < 0.0
    assert stages["newey_west"].spectrum is not None
    assert stages["newey_west"].spectrum.negative_eigenvalues >= 1

    assert build.repair_fired
    report = build.repairs[0]
    assert report.floored >= 1
    assert report.minimum_eigenvalue_before < 0.0
    # The margin is real, not roundoff: 1e-3 of lambda_max is 1e13 * eps.
    assert abs(report.minimum_eigenvalue_before) > 1e-6 * report.maximum_eigenvalue_before

    assert stages["psd_repair"].check is not None
    repaired = np.linalg.eigvalsh(build.matrix)
    assert repaired[0] >= eigenvalue_floor(
        len(repaired), repaired[-1], numerics=load().model.numerics
    )


def test_the_pipeline_takes_the_floor_from_the_config_object_it_was_given() -> None:
    """The stage reads ``RiskConfig.psd_eigenvalue_floor`` and nothing else."""
    periods, factors = _FORCES_REPAIR
    values = np.random.default_rng(load().model.seed).standard_normal((periods, factors))
    raised = replace(RiskConfig.load(horizon=HORIZONS[0]), psd_eigenvalue_floor=1e-6)
    build = run_pipeline(_frame(values), raised, stop_after="psd_repair")
    assert build.repairs[0].floor == 1e-6
    assert np.linalg.eigvalsh(build.matrix)[0] == pytest.approx(1e-6, rel=1e-6)


# ---------------------------------------------------------------------------
# CLAUDE.md invariant 4, as read in W3-P2.
# ---------------------------------------------------------------------------


def test_exactly_one_stage_is_measured_without_being_asserted() -> None:
    """A READING of invariant 4, not a relaxation of it, and it may not spread.

    ``newey_west`` is the one stage in SPEC.md 5 whose non-PSD output is declared
    in advance and whose successor exists to repair it, so asserting between the
    two would convert an expected, handled outcome into a crash. Every other stage
    asserts. This test is what stops the exemption growing a second member
    quietly, and what stops a later session reading the missing assertion as an
    oversight and "fixing" it.
    """
    assert measured_not_asserted_stages() == frozenset({"newey_west"})
    declared = RiskConfig.load(horizon=HORIZONS[0]).stages
    assert "psd_repair" in declared
    assert declared.index("newey_west") + 1 == declared.index("psd_repair")


@pytest.mark.parametrize("horizon", HORIZONS)
def test_every_stage_is_either_asserted_or_measured_and_never_both(horizon: str) -> None:
    values = np.random.default_rng(load().model.seed).standard_normal((600, 4))
    build = run_pipeline(
        _frame(values),
        RiskConfig.load(horizon=horizon).for_scaling(1.0),  # type: ignore[arg-type]
        regime_bias=_CALIBRATED_BIAS,
    )
    for stage in build.stages:
        assert (stage.check is None) != (stage.spectrum is None)
        assert stage.asserted == (stage.name not in measured_not_asserted_stages())
        assert np.isfinite(stage.minimum_eigenvalue)
        assert np.isfinite(stage.maximum_eigenvalue)


def test_the_repair_report_is_attached_to_the_stage_that_repairs_and_no_other() -> None:
    values = np.random.default_rng(load().model.seed).standard_normal((600, 4))
    build = run_pipeline(
        _frame(values),
        RiskConfig.load(horizon=HORIZONS[0]).for_scaling(1.0),
        regime_bias=_CALIBRATED_BIAS,
    )
    with_repair = [stage.name for stage in build.stages if stage.repair is not None]
    assert with_repair == ["psd_repair"]
