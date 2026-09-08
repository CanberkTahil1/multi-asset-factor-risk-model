"""The disjointness simulation against its analytic control. W4-P3.

What is pinned: that the instrument recovers the closed form derived before the
run (``experiments.md`` rows 175-186) on an equal-weight window where the
formula is exact to second order; that Shepard's own single-window case comes
out at Shepard's number; that the per-factor bias is the Jensen term and not
the optimiser's; and that the interaction term is zero to Monte Carlo precision
where independence says it must be.
"""

from __future__ import annotations

import numpy as np
import pytest

from mafrm.risk.second_order import (
    SecondOrderError,
    inverse_second_moment_expectation,
    simulate,
    volatility_leg_closed_form,
)
from mafrm.risk.shepard import Unit
from mafrm.risk.validation import minimum_variance_weights

#: A half-life so long that every weight on a 300-row window is equal to within
#: 1e-4 -- the flat-weight control of experiments.md row 110, where T_eff is the
#: row count and the closed forms are exact to second order.
_FLAT = 1.0e6


def test_the_closed_form_by_hand() -> None:
    # K = 2 equal weights, T = 100: 1 + (2/100)(2 - 0.5) = 1.03
    assert volatility_leg_closed_form(np.array([0.5, 0.5]), effective_observations=100.0) == (
        pytest.approx(1.03)
    )
    # K = 1: Shepard's own 1 + 2/T
    assert volatility_leg_closed_form(np.array([1.0]), effective_observations=50.0) == (
        pytest.approx(1.04)
    )
    # Unequal weights: Herfindahl 0.5^2 + 0.3^2 + 0.2^2 = 0.38 -> 1 + 0.02 * 1.62
    assert volatility_leg_closed_form(
        np.array([0.5, 0.3, 0.2]), effective_observations=100.0
    ) == pytest.approx(1.0324)


def test_the_closed_form_rejects_bad_input() -> None:
    with pytest.raises(SecondOrderError):
        volatility_leg_closed_form(np.ones((2, 2)), effective_observations=10.0)
    with pytest.raises(SecondOrderError):
        volatility_leg_closed_form(np.ones(2), effective_observations=0.0)


@pytest.fixture(scope="module")
def flat_identity_run() -> tuple[np.ndarray, int, object]:
    variances = np.array([1.0, 4.0, 0.25, 2.25, 9.0, 0.64])
    truth = np.diag(variances)
    observations = 300
    result = simulate(
        truth,
        volatility_halflife=_FLAT,
        correlation_halflife=_FLAT,
        observations=observations,
        trials=4000,
        seed=20260825,
    )
    return truth, observations, result


def test_sigma_leg_recovers_the_derived_closed_form(
    flat_identity_run: tuple[np.ndarray, int, object],
) -> None:
    """Row 176's instrument check: rho = I, sigma only, against 1 + (2/T)(2 - sum w^2)."""
    truth, observations, result = flat_identity_run
    expected = volatility_leg_closed_form(
        minimum_variance_weights(truth), effective_observations=float(observations)
    )
    measured = result.volatility_only  # type: ignore[attr-defined]
    assert measured.standard_error < 0.1 * (expected - 1.0)
    assert measured.mean_ratio == pytest.approx(expected, abs=0.1 * (expected - 1.0))


def test_single_window_recovers_shepard(
    flat_identity_run: tuple[np.ndarray, int, object],
) -> None:
    """Row 175's shape on the flat control: (1 - 6/300)^-2 - 1 = 0.0412 in variance."""
    _, observations, result = flat_identity_run
    shepard = (1.0 - 6 / observations) ** -2.0 - 1.0
    single = result.single_window  # type: ignore[attr-defined]
    assert single.excess == pytest.approx(shepard, abs=0.1 * shepard)
    # And in volatility the bias statistic is the square root.
    assert single.bias == pytest.approx(np.sqrt(single.mean_ratio))


def test_per_factor_bias_is_the_jensen_term_not_the_optimisers(
    flat_identity_run: tuple[np.ndarray, int, object],
) -> None:
    """Row 180's shape: mean_k F_kk / F_hat_kk -> 1 + 2/T, well below the min-var bias."""
    _, observations, result = flat_identity_run
    per_factor = result.per_factor  # type: ignore[attr-defined]
    jensen = 2.0 / observations
    assert per_factor.excess == pytest.approx(
        jensen, abs=3 * per_factor.standard_error + 0.1 * jensen
    )
    assert per_factor.excess < result.single_window.excess  # type: ignore[attr-defined]


def test_interaction_is_zero_where_independence_holds(
    flat_identity_run: tuple[np.ndarray, int, object],
) -> None:
    """Row 179's statistic on the control: |I| within 3 MC standard errors of zero."""
    _, _, result = flat_identity_run
    assert abs(result.interaction) < 3 * result.interaction_standard_error  # type: ignore[attr-defined]
    assert abs(result.interaction_share) < 0.10  # type: ignore[attr-defined]


def test_split_window_legs_add_on_a_correlated_truth() -> None:
    """The disjointness claim on a truth with structure, at the pipeline's own half-lives."""
    rng = np.random.default_rng(7)
    loadings = rng.standard_normal((6, 2))
    truth = loadings @ loadings.T + np.diag(np.linspace(0.5, 2.0, 6))
    result = simulate(
        truth,
        volatility_halflife=84,
        correlation_halflife=504,
        observations=1500,
        trials=3000,
        seed=3,
    )
    assert result.joint.excess > 0.0
    assert result.correlation_only.excess > 0.0
    assert result.volatility_only.excess > 0.0
    assert abs(result.interaction_share) < 0.10
    # The whole-covariance closed form at the volatility window is NOT the split
    # estimator's bias: rho on a 6x longer window makes the joint smaller.
    assert result.joint.excess < result.shepard_volatility_window.understatement(Unit.VARIANCE)
    assert result.realised_volatility_t_eff < result.realised_correlation_t_eff


def test_simulation_is_reproducible_from_the_seed() -> None:
    truth = np.diag([1.0, 2.0, 3.0])
    first = simulate(
        truth, volatility_halflife=20, correlation_halflife=40, observations=80, trials=50, seed=1
    )
    second = simulate(
        truth, volatility_halflife=20, correlation_halflife=40, observations=80, trials=50, seed=1
    )
    assert np.array_equal(first.joint.ratios, second.joint.ratios)


def test_simulation_rejects_what_it_cannot_run_on() -> None:
    with pytest.raises(SecondOrderError):
        simulate(
            np.ones((3, 3)),  # rank one: not positive definite
            volatility_halflife=10,
            correlation_halflife=20,
            observations=50,
            trials=10,
            seed=0,
        )
    with pytest.raises(SecondOrderError):
        simulate(
            np.eye(3),
            volatility_halflife=10,
            correlation_halflife=20,
            observations=3,
            trials=10,
            seed=0,
        )
    with pytest.raises(SecondOrderError):
        simulate(
            np.eye(3),
            volatility_halflife=10,
            correlation_halflife=20,
            observations=50,
            trials=1,
            seed=0,
        )


def test_inverse_second_moment_expectation_is_exact_for_equal_weights() -> None:
    """E[T / chi^2_T] = T / (T - 2): T = 10 gives 1.25, T = 300 gives 1.0067114."""
    assert inverse_second_moment_expectation(np.full(10, 0.1)) == pytest.approx(1.25, rel=1e-6)
    assert inverse_second_moment_expectation(np.full(300, 1 / 300)) == pytest.approx(
        300 / 298, rel=1e-6
    )


def test_inverse_second_moment_expectation_matches_the_jensen_term_for_ewma_weights() -> None:
    from mafrm.numerics import ewma_weights, realised_effective_sample_size

    weights = ewma_weights(3889, 84.0)
    exact = inverse_second_moment_expectation(weights) - 1.0
    approximate = 2.0 / realised_effective_sample_size(weights)
    assert exact == pytest.approx(approximate, rel=0.01)
    assert exact > approximate  # the higher-order terms are positive


def test_inverse_second_moment_expectation_rejects_bad_weights() -> None:
    with pytest.raises(SecondOrderError):
        inverse_second_moment_expectation(np.array([-1.0, 2.0]))
    with pytest.raises(SecondOrderError):
        inverse_second_moment_expectation(np.zeros(3))


# ---------------------------------------------------------------------------
# W8-P1: the tau grid (experiments.md rows 318-326)
# ---------------------------------------------------------------------------


def test_the_tau_grid_varies_the_seed_and_adjudicates_each_point() -> None:
    """Two points from one truth: different seeds, the registered comparisons in row order."""
    import dataclasses

    from mafrm import config as config_mod
    from mafrm.factors import second_order_report as report
    from mafrm.risk import second_order

    settings = config_mod.load()
    truth = np.array([[4.0, 1.0], [1.0, 2.0]])
    points = tuple(
        report.TauGridPoint(
            halflife=h,
            seed=settings.model.seed + i,
            result=second_order.simulate(
                truth,
                volatility_halflife=float(h),
                correlation_halflife=40.0,
                observations=120,
                trials=40,
                seed=settings.model.seed + i,
            ),
        )
        for i, h in enumerate((10, 20))
    )
    assert points[0].seed != points[1].seed
    assert points[0].k_over_t == pytest.approx(2 / (2 * 10 / np.log(2)))
    comparisons = report.compare_tau_grid(points, settings, rows=(318, 319))
    assert [c.row for c in comparisons] == [318, 318, 319]
    joint, first, second = comparisons
    assert joint.measured == points[0].result.joint.excess
    assert (joint.lower, joint.upper) == (0.045, 0.085)
    assert first.measured == points[0].result.correlation_only.excess
    assert second.measured == points[1].result.correlation_only.excess
    # The rho-leg verdict is "within 2 MC s.e. of the registered value", nothing else.
    leg = points[1].result.correlation_only
    registered = settings.model.validation.second_order.tau_grid.correlation_leg_registered
    expected = abs(leg.excess - registered) <= 2.0 * leg.standard_error
    assert second.holds == expected
    # A row count that disagrees with the grid is refused.
    with pytest.raises(ValueError, match="against"):
        report.compare_tau_grid(points, settings, rows=(318,))
    # The registered band is transcribed as written at W4-P3 (experiments.md after row 191).
    tau = settings.model.validation.second_order.tau_grid
    assert dataclasses.astuple(tau)[1:4] == (0.065, 0.045, 0.085)
    assert tau.correlation_leg_registered == pytest.approx(0.00647)
