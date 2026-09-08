"""Model B, the statistical factor model. SPEC.md 4.2 and the W3-P5 rulings.

The hand-computed values are in :func:`test_marchenko_pastur_edges_are_hand_computable`,
:func:`test_gaussian_kde_at_a_single_point_is_the_normal_density` and
:func:`test_marchenko_pastur_density_at_the_centre_is_hand_computable`. Everything
else is either an identity the estimator must satisfy exactly (trace, unit
diagonal, PSD) or a control that would fire on a plausible wrong implementation.

The two tests that matter most are the ones about what the code must NOT do:
:func:`test_nothing_after_date_t_reaches_date_t` and
:func:`test_the_factor_is_a_linear_combination_of_raw_returns`. The first is the
look-ahead guard; the second is what keeps SPEC.md 5.1.1's zero-mean convention
coherent once these factors reach the covariance pipeline.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mafrm.config import ConfigError, _parse_statistical, load
from mafrm.factors import statistical as st

SETTINGS = load()
SCHEME = SETTINGS.model.factors.statistical


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    """Seeded explicitly. CLAUDE.md: never a global ``np.random.seed``."""
    return np.random.default_rng(SETTINGS.model.seed)


@pytest.fixture(scope="module")
def synthetic(rng: np.random.Generator) -> np.ndarray:
    """A ``T x N`` panel with three planted factors. No cache, no network."""
    n_vars, periods, factors = 12, 900, 3
    loadings = rng.standard_normal((n_vars, factors)) * 0.6
    return np.asarray(
        rng.standard_normal((periods, factors)) @ loadings.T
        + rng.standard_normal((periods, n_vars))
    )


# ---------------------------------------------------------------------------
# Hand-computed values
# ---------------------------------------------------------------------------


def test_marchenko_pastur_edges_are_hand_computable() -> None:
    """``sigma^2 = 1``, ``N/T = 0.25``: ``sqrt(0.25) = 0.5``, so the band is [0.25, 2.25]."""
    lower, upper = st.marchenko_pastur_edges(noise_variance=1.0, ratio=0.25)
    assert lower == pytest.approx(0.25, abs=1e-15)
    assert upper == pytest.approx(2.25, abs=1e-15)


def test_the_edge_scales_linearly_with_the_noise_variance() -> None:
    """``lambda_+`` is ``sigma^2`` times a function of ``N/T`` alone."""
    base = st.marchenko_pastur_edges(noise_variance=1.0, ratio=0.25)[1]
    scaled = st.marchenko_pastur_edges(noise_variance=0.36, ratio=0.25)[1]
    assert scaled == pytest.approx(0.36 * base, rel=1e-14)


def test_gaussian_kde_at_a_single_point_is_the_normal_density() -> None:
    """One sample at 0, bandwidth 1, evaluated at 0: ``1/sqrt(2 pi) = 0.3989422804``.

    The whole reason this function exists rather than a ``scipy.stats.gaussian_kde``
    call is that scipy's ``bw_method`` is a MULTIPLIER on the sample standard
    deviation. This asserts the ABSOLUTE reading, which is the one SPEC.md 4.2's
    "bandwidth 0.15" means.
    """
    value = st.gaussian_kde_pdf(np.array([0.0]), np.array([0.0]), bandwidth=1.0)
    assert value[0] == pytest.approx(1.0 / math.sqrt(2.0 * math.pi), abs=1e-15)


def test_the_kde_bandwidth_is_absolute_not_a_multiplier_of_the_spread() -> None:
    """The control for the test above, stated as the scaling identity it turns on.

    A density in absolute units satisfies ``f_{c*h}(c*x; c*sample) = f_h(x; sample)/c``
    -- scale the data, the grid AND the bandwidth together and only the Jacobian
    moves. Under scipy's relative convention the bandwidth would scale itself, so
    the identity would hold with ``h`` left alone instead. Both halves are asserted,
    because only the pair distinguishes the two conventions: a wrong implementation
    passes the single-point test and fails the second half here.
    """
    sample = np.array([-1.0, 0.0, 0.4, 1.0])
    grid = np.array([-0.2, 0.3, 0.9])
    base = st.gaussian_kde_pdf(sample, grid, bandwidth=0.15)

    scaled_with_bandwidth = st.gaussian_kde_pdf(10.0 * sample, 10.0 * grid, bandwidth=1.5)
    assert np.allclose(scaled_with_bandwidth, base / 10.0, rtol=1e-12)

    scaled_without_bandwidth = st.gaussian_kde_pdf(10.0 * sample, 10.0 * grid, bandwidth=0.15)
    assert not np.allclose(scaled_without_bandwidth, base / 10.0, rtol=1e-3)


def test_marchenko_pastur_density_at_the_centre_is_hand_computable() -> None:
    """``sigma^2 = 1``, ``N/T = 0.25``, ``x = 1``.

    ``sqrt((2.25 - 1)(1 - 0.25)) / (2 pi * 1 * 0.25 * 1)``
    ``= 0.9682458365... / 1.5707963267... = 0.6164044440...``
    """
    value = st.marchenko_pastur_pdf(np.array([1.0]), noise_variance=1.0, ratio=0.25)
    expected = math.sqrt(1.25 * 0.75) / (2.0 * math.pi * 0.25)
    assert value[0] == pytest.approx(expected, rel=1e-14)
    assert value[0] == pytest.approx(0.6164044, rel=1e-6)


def test_the_density_is_zero_outside_the_band() -> None:
    grid = np.array([0.1, 0.25, 2.25, 3.0])
    values = st.marchenko_pastur_pdf(grid, noise_variance=1.0, ratio=0.25)
    assert np.all(values == 0.0)


@pytest.mark.parametrize("ratio", [0.05, 0.25, 0.5])
def test_the_density_integrates_to_one(ratio: float) -> None:
    """It is a probability density. A missing ``2 pi`` or ``q`` would fail here."""
    lower, upper = st.marchenko_pastur_edges(noise_variance=1.0, ratio=ratio)
    grid = np.linspace(lower, upper, 400_001)
    mass = np.trapezoid(st.marchenko_pastur_pdf(grid, noise_variance=1.0, ratio=ratio), grid)
    assert mass == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# The sigma^2 fit
# ---------------------------------------------------------------------------


def test_the_fit_recovers_the_noise_variance_on_pure_noise(rng: np.random.Generator) -> None:
    """No planted factor: ``sigma^2`` is 1 on a correlation matrix and nothing survives."""
    n_vars, periods = 50, 500
    sample = rng.standard_normal((periods, n_vars))
    eigenvalues = np.linalg.eigvalsh(np.corrcoef(sample.T))[::-1]
    fit = st.fit_noise_variance(eigenvalues, ratio=n_vars / periods, settings=SCHEME)
    assert fit.noise_variance == pytest.approx(1.0, abs=1e-3)
    assert int(np.sum(eigenvalues > fit.lambda_plus)) == 0


def test_the_fit_finds_less_noise_when_a_factor_is_planted(synthetic: np.ndarray) -> None:
    """Signal takes variance from the noise, so ``sigma^2`` must fall below 1."""
    periods, n_vars = synthetic.shape
    eigenvalues = np.linalg.eigvalsh(np.corrcoef(synthetic.T))[::-1]
    fit = st.fit_noise_variance(eigenvalues, ratio=n_vars / periods, settings=SCHEME)
    assert 0.0 < fit.noise_variance < 1.0
    assert int(np.sum(eigenvalues > fit.lambda_plus)) == 3


def test_the_grid_resolution_is_a_discretisation_not_a_dial(synthetic: np.ndarray) -> None:
    """``experiments.md`` row 121. Moving ``kde_grid_points`` must not move the answer.

    If it did, a number that arrived in ``config/model.yaml`` described as a
    discretisation of a cited procedure would in fact be a tunable, and would owe
    a ``model-config`` row.
    """
    periods, n_vars = synthetic.shape
    eigenvalues = np.linalg.eigvalsh(np.corrcoef(synthetic.T))[::-1]
    fitted = [
        st.fit_noise_variance(
            eigenvalues,
            ratio=n_vars / periods,
            settings=dataclasses.replace(SCHEME, kde_grid_points=points),
        ).noise_variance
        for points in (500, 1000, 2000)
    ]
    assert max(fitted) - min(fitted) < 1e-4


def test_the_band_half_width_is_the_square_root_of_the_ratio() -> None:
    fit = st.fit_noise_variance(np.array([3.0, 1.0, 0.5, 0.4, 0.1]), ratio=0.25, settings=SCHEME)
    assert fit.band_half_width == pytest.approx(0.5, abs=1e-15)


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------


def test_denoising_preserves_the_trace_exactly(synthetic: np.ndarray) -> None:
    """SPEC.md 4.2's stated constraint. A correlation matrix has trace ``N``."""
    periods, n_vars = synthetic.shape
    correlation = np.corrcoef(synthetic.T)
    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
    fit = st.fit_noise_variance(eigenvalues, ratio=n_vars / periods, settings=SCHEME)
    result = st.denoise_correlation(correlation, fit=fit)
    assert np.trace(result.correlation) == pytest.approx(float(n_vars), abs=1e-10)


def test_denoising_replaces_the_noise_eigenvalues_with_one_constant() -> None:
    """The 'constant average' of SPEC.md 4.2, checked on a hand-built spectrum.

    Eigenvalues 4, 3, 0.4, 0.3, 0.3 sum to 8. With ``lambda_+`` between 0.4 and 3
    the three noise values average ``(0.4 + 0.3 + 0.3)/3 = 1/3``, and their sum is
    unchanged at 1.0.
    """
    spectrum = np.array([4.0, 3.0, 0.4, 0.3, 0.3])
    vectors = np.linalg.qr(np.eye(5)[::-1] + np.eye(5))[0]
    matrix = vectors @ np.diag(spectrum) @ vectors.T
    fit = st.NoiseFit(
        noise_variance=1.0,
        lambda_plus=1.0,
        lambda_minus=0.5,
        ratio=0.25,
        sum_squared_error=0.0,
        converged=True,
    )
    result = st.denoise_correlation(matrix, fit=fit)
    assert result.survivors == 2
    noise = result.denoised_eigenvalues[2:]
    assert np.allclose(noise, 1.0 / 3.0, atol=1e-12)
    assert noise.sum() == pytest.approx(1.0, abs=1e-12)


def test_the_denoised_matrix_is_a_correlation_matrix(synthetic: np.ndarray) -> None:
    """Unit diagonal, symmetric, positive semi-definite. All three, not two."""
    periods, n_vars = synthetic.shape
    correlation = np.corrcoef(synthetic.T)
    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
    fit = st.fit_noise_variance(eigenvalues, ratio=n_vars / periods, settings=SCHEME)
    result = st.denoise_correlation(correlation, fit=fit)
    assert np.allclose(np.diag(result.correlation), 1.0, atol=1e-12)
    assert np.allclose(result.correlation, result.correlation.T, atol=1e-12)
    assert np.linalg.eigvalsh(result.correlation).min() > -1e-12


def test_denoising_improves_the_conditioning(synthetic: np.ndarray) -> None:
    """The reason the stage exists: the noise floor is lifted off zero."""
    periods, n_vars = synthetic.shape
    correlation = np.corrcoef(synthetic.T)
    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
    fit = st.fit_noise_variance(eigenvalues, ratio=n_vars / periods, settings=SCHEME)
    result = st.denoise_correlation(correlation, fit=fit)
    assert np.linalg.cond(result.correlation) < np.linalg.cond(correlation)


def test_the_eigenvectors_are_the_inputs_and_denoising_rotates_nothing(
    synthetic: np.ndarray,
) -> None:
    """Constant-average replacement changes ``Lambda`` and leaves ``V`` alone.

    This is what "denoising before extraction" amounts to in the form SPEC.md 4.2
    specifies: a rule for how many components to keep, not for what they are. The
    factors are therefore extracted from the input correlation's own eigenvectors,
    and this pins that -- if the implementation ever started taking them from the
    RESCALED matrix, the count and the directions would come from two different
    spectra and this fails.
    """
    periods, n_vars = synthetic.shape
    correlation = np.corrcoef(synthetic.T)
    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
    fit = st.fit_noise_variance(eigenvalues, ratio=n_vars / periods, settings=SCHEME)
    result = st.denoise_correlation(correlation, fit=fit)

    expected = np.linalg.eigh(correlation)[1][:, ::-1]
    for column in range(result.survivors):
        assert abs(float(expected[:, column] @ result.eigenvectors[:, column])) == pytest.approx(
            1.0, abs=1e-10
        )
    assert np.allclose(result.eigenvalues, eigenvalues, atol=1e-12)


def test_the_rescale_moves_the_spectrum_and_that_is_recorded(
    synthetic: np.ndarray,
) -> None:
    """The reason the two survivor counts are kept apart rather than assumed equal.

    On this project's real panel the pre- and post-rescale counts disagree on
    about 6% of dates, which is why ``survivors_after_rescale`` exists. Here the
    assertion is only that the field is populated and consistent in type -- the
    disagreement itself is data-dependent and is reported, not required.
    """
    periods, n_vars = synthetic.shape
    correlation = np.corrcoef(synthetic.T)
    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
    fit = st.fit_noise_variance(eigenvalues, ratio=n_vars / periods, settings=SCHEME)
    result = st.denoise_correlation(correlation, fit=fit)
    assert isinstance(result.survivors_after_rescale, int)
    assert 0 <= result.survivors_after_rescale <= n_vars
    assert np.trace(result.correlation) == pytest.approx(float(n_vars), abs=1e-10)


def test_the_diagonal_deviation_is_reported_and_is_not_negligible(
    synthetic: np.ndarray,
) -> None:
    """The unit-diagonal rescale is a real step, not a rounding tidy-up.

    SPEC.md 4.2 does not mention it, so the size of what it does is asserted
    rather than assumed small -- if this ever became negligible the report's
    claim that the step matters would have gone stale.
    """
    periods, n_vars = synthetic.shape
    correlation = np.corrcoef(synthetic.T)
    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
    fit = st.fit_noise_variance(eigenvalues, ratio=n_vars / periods, settings=SCHEME)
    result = st.denoise_correlation(correlation, fit=fit)
    assert result.diagonal_deviation > 1e-3


# ---------------------------------------------------------------------------
# Detoning
# ---------------------------------------------------------------------------


def test_detoning_removes_exactly_the_leading_component(synthetic: np.ndarray) -> None:
    """One eigenvalue goes to zero and the matrix loses exactly one rank."""
    correlation = np.corrcoef(synthetic.T)
    detoned = st.detone_correlation(correlation, components=1)
    eigenvalues = np.linalg.eigvalsh(detoned)[::-1]
    assert abs(eigenvalues[-1]) < 1e-10
    assert eigenvalues[-2] > 1e-6
    assert np.allclose(np.diag(detoned), 1.0, atol=1e-12)


def test_detoning_is_refused_when_it_would_leave_nothing() -> None:
    with pytest.raises(st.StatisticalFactorError, match="leaves nothing"):
        st.detone_correlation(np.eye(3), components=3)


def test_denoising_after_detoning_removes_the_singularity(synthetic: np.ndarray) -> None:
    """SPEC.md 4.2's order: detone, THEN denoise. The zero is absorbed into the bulk.

    This is the step that makes the spec's ordering work at all -- a rank-deficient
    correlation would have no minimum-variance solution and no invertible
    covariance downstream.
    """
    periods, n_vars = synthetic.shape
    detoned = st.detone_correlation(np.corrcoef(synthetic.T), components=1)
    eigenvalues = np.linalg.eigvalsh(detoned)[::-1]
    fit = st.fit_noise_variance(eigenvalues, ratio=n_vars / periods, settings=SCHEME)
    result = st.denoise_correlation(detoned, fit=fit)
    assert np.linalg.eigvalsh(result.correlation).min() > 1e-6


# ---------------------------------------------------------------------------
# The comparands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "estimator", [st.ledoit_wolf_constant_correlation, st.oracle_approximating_shrinkage]
)
def test_a_comparand_returns_a_correlation_matrix(estimator: object, synthetic: np.ndarray) -> None:
    result = estimator(synthetic)  # type: ignore[operator]
    assert np.allclose(np.diag(result.correlation), 1.0, atol=1e-12)
    assert np.allclose(result.correlation, result.correlation.T, atol=1e-12)
    assert np.linalg.eigvalsh(result.correlation).min() > -1e-12
    assert 0.0 <= result.intensity <= 1.0


@pytest.mark.parametrize(
    "estimator", [st.ledoit_wolf_constant_correlation, st.oracle_approximating_shrinkage]
)
def test_shrinkage_intensity_falls_as_the_sample_grows(
    estimator: object, rng: np.random.Generator
) -> None:
    """The defining property of a shrinkage estimator: it shrinks less with more data.

    An implementation that dropped the ``1/T`` would pass every shape check above
    and fail here.
    """
    n_vars, factors = 20, 3
    loadings = rng.standard_normal((n_vars, factors)) * 0.5
    draws = rng.standard_normal((4000, factors)) @ loadings.T + rng.standard_normal((4000, n_vars))
    intensities = [
        estimator(draws[:periods]).intensity  # type: ignore[operator]
        for periods in (60, 250, 1000, 4000)
    ]
    assert intensities == sorted(intensities, reverse=True)
    assert intensities[0] > 2.0 * intensities[-1]


def test_ledoit_wolf_shrinks_toward_constant_correlation_not_identity(
    synthetic: np.ndarray,
) -> None:
    """SPEC.md 4.2's named trap, asserted rather than commented.

    ``sklearn.covariance.LedoitWolf`` uses the identity target. The
    constant-correlation target keeps the average off-diagonal correlation, so
    the shrunk matrix must stay CLOSER to the sample correlation than to the
    identity -- which is what distinguishes the two estimators.
    """
    result = st.ledoit_wolf_constant_correlation(synthetic)
    sample = np.corrcoef(synthetic.T)
    identity = np.eye(sample.shape[0])
    to_sample = np.linalg.norm(result.correlation - sample, ord="fro")
    to_identity = np.linalg.norm(result.correlation - identity, ord="fro")
    assert to_sample < to_identity
    assert "identity" in result.target


def test_a_comparand_refuses_a_sample_of_one() -> None:
    for estimator in (st.ledoit_wolf_constant_correlation, st.oracle_approximating_shrinkage):
        with pytest.raises(st.StatisticalFactorError, match="not a sample"):
            estimator(np.ones((1, 4)))


# ---------------------------------------------------------------------------
# The expanding window
# ---------------------------------------------------------------------------


def test_the_expanding_recursion_matches_numpy(synthetic: np.ndarray) -> None:
    """Running sums are an optimisation, so they are checked against the obvious form."""
    correlations, deviations, defined = st.expanding_correlations(synthetic, min_window=60)
    for index in (59, 100, 400, synthetic.shape[0] - 1):
        assert defined[index]
        expected = np.corrcoef(synthetic[: index + 1].T)
        assert np.allclose(correlations[index], expected, atol=1e-10)
        assert np.allclose(
            deviations[index], synthetic[: index + 1].std(axis=0, ddof=1), atol=1e-12
        )
    assert not defined[:59].any()


def test_the_expanding_window_refuses_a_degenerate_minimum(synthetic: np.ndarray) -> None:
    with pytest.raises(st.StatisticalFactorError, match="at least 2"):
        st.expanding_correlations(synthetic, min_window=1)


# ---------------------------------------------------------------------------
# The panel -- the two tests that matter
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_frame(synthetic: np.ndarray) -> pd.DataFrame:
    dates = pd.bdate_range("2010-01-04", periods=synthetic.shape[0])
    names = [f"asset_{position}" for position in range(synthetic.shape[1])]
    return pd.DataFrame(synthetic, index=dates, columns=names)


def test_nothing_after_date_t_reaches_date_t(small_frame: pd.DataFrame) -> None:
    """The look-ahead guard, asserted by perturbation rather than by inspection.

    Every value after a cut date is replaced with something wildly different. The
    factor returns and the diagnostics on and before the cut must be BIT-identical
    -- not close. Anything that reached forward, including a full-sample
    eigendecomposition or a standard deviation taken over the whole panel, moves
    them.
    """
    cut = 500
    perturbed = small_frame.copy()
    perturbed.iloc[cut + 1 :] = perturbed.iloc[cut + 1 :] * 7.0 + 3.0

    base = st.statistical_factor_panel(small_frame, config=SETTINGS)
    other = st.statistical_factor_panel(perturbed, config=SETTINGS)
    date = small_frame.index[cut]

    shared = min(base.factors.shape[1], other.factors.shape[1])
    left = base.factors.loc[:date].iloc[:, :shared]
    right = other.factors.loc[:date].iloc[:, :shared]
    assert left.shape == right.shape
    np.testing.assert_array_equal(left.to_numpy(), right.to_numpy())
    for column in ("lambda_plus", "noise_variance", "survivors", "n_over_t"):
        np.testing.assert_array_equal(
            base.diagnostics.loc[:date, column].to_numpy(),
            other.diagnostics.loc[:date, column].to_numpy(),
        )


def test_the_frame_width_depends_on_the_whole_window_and_the_values_do_not(
    small_frame: pd.DataFrame,
) -> None:
    """The one thing about this panel that DOES look forward, isolated and named.

    ``factors`` is as wide as the largest survivor count the window ever reached,
    so perturbing the far end of the panel can add a column. No factor VALUE moves
    -- :func:`test_nothing_after_date_t_reaches_date_t` asserts that on the same
    perturbation -- but a date whose own ``K`` is short of the maximum carries NaN
    in the surplus column and is therefore dropped from ``complete``.

    That makes the SET of dates handed to the covariance pipeline depend on the
    whole window, in the same way the asset panel's complete-case rule does. It
    changes no number and it is not hidden: ``dates_dropped_by_completeness``
    reports the cost, and the report prints it.
    """
    cut = 500
    perturbed = small_frame.copy()
    perturbed.iloc[cut + 1 :] = perturbed.iloc[cut + 1 :] * 7.0 + 3.0
    base = st.statistical_factor_panel(small_frame, config=SETTINGS)
    other = st.statistical_factor_panel(perturbed, config=SETTINGS)
    assert base.factors.shape[1] != other.factors.shape[1]
    assert base.dates_dropped_by_completeness == len(base.factors) - len(base.complete)


def test_the_factor_is_a_linear_combination_of_raw_returns(small_frame: pd.DataFrame) -> None:
    """SPEC.md 5.1.1's coherence requirement, carried into Model B.

    The covariance pipeline forecasts variance about ZERO because SPEC.md 6.1's
    bias statistic has a raw return in its numerator. That holds for a factor only
    if the factor is a fixed portfolio of RAW returns on each date. The
    correlation estimator demeans -- it is the sample correlation -- but the
    projection must not, and this reproduces the projection from the loadings and
    the raw row to prove it does not.
    """
    panel = st.statistical_factor_panel(small_frame, config=SETTINGS)
    position = 300
    date = panel.factors.index[position]
    row = small_frame.loc[date].to_numpy(dtype=float)
    deviations = small_frame.loc[:date].std(axis=0, ddof=1).to_numpy(dtype=float)
    for name, loadings in panel.loadings.items():
        vector = loadings.loc[date].to_numpy(dtype=float)
        if np.isnan(vector).any():
            continue
        expected = float(row @ (vector / deviations))
        assert panel.factors.loc[date, name] == pytest.approx(expected, rel=1e-12)


def test_the_sign_convention_makes_consecutive_loadings_agree(
    small_frame: pd.DataFrame,
) -> None:
    """SPEC.md 4.2.3. Without it, half the dates would carry a sign-flipped factor.

    The assertion is on the ALIGNMENT, not the flip count: the flip count is about
    half by construction and says nothing about stability.
    """
    panel = st.statistical_factor_panel(small_frame, config=SETTINGS)
    alignment = panel.diagnostics["component_alignment"]
    assert alignment.median() > 0.99
    assert (alignment > 0.9).mean() > 0.95


def test_the_orientation_flips_an_anti_aligned_vector_and_counts_it() -> None:
    """The sign convention tested at the mechanism, not through a fixture.

    An earlier version of this control looked for sign disagreements inside a
    synthetic panel and found none -- ``numpy.linalg.eigh``'s sign is
    deterministic for a given matrix, so on a smoothly growing window it often
    never disagrees, and the control was passing on the fixture rather than on the
    code. A test that can only fire when the data happens to be adversarial cannot
    detect its own failure mode.

    So the adversarial input is constructed instead: hand ``orient_eigenvectors`` a set of
    vectors that are exactly the negatives of the previous date's and assert it
    flips every one of them back and says so.
    """
    previous = np.linalg.qr(np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]))[0]
    oriented, flips, alignment = st.orient_eigenvectors(-previous, previous, survivors=3)
    assert flips == 3
    np.testing.assert_allclose(oriented, previous, atol=1e-12)
    assert alignment == pytest.approx(1.0, abs=1e-12)

    kept, no_flips, _ = st.orient_eigenvectors(previous, previous, survivors=3)
    assert no_flips == 0
    np.testing.assert_allclose(kept, previous, atol=1e-12)


def test_the_first_date_is_oriented_by_its_largest_loading() -> None:
    """The seed convention, and why it is the largest element rather than the sum.

    A "positive sum" convention is a coin flip for a component whose loadings
    cancel -- which is exactly what the higher components of a correlation matrix
    look like. The largest absolute element of a unit vector is at least
    ``1/sqrt(N)`` and cannot be that.
    """
    vector = np.array([[0.6], [-0.8]])
    oriented, flips, _ = st.orient_eigenvectors(vector, None, survivors=1)
    assert flips == 1
    np.testing.assert_allclose(oriented[:, 0], np.array([-0.6, 0.8]), atol=1e-12)
    assert abs(oriented[np.argmax(np.abs(oriented[:, 0])), 0]) == pytest.approx(0.8)
    assert oriented[np.argmax(np.abs(oriented[:, 0])), 0] > 0.0


def test_the_alignment_ignores_the_noise_block() -> None:
    """Restricted to survivors on purpose: the noise directions are near-arbitrary.

    The noise block's eigenvalues are equal after denoising, so its eigenvectors
    are only defined up to a rotation and swap constantly. Reporting their
    alignment would show a permanent instability in a subspace no factor is taken
    from.
    """
    previous = np.eye(3)
    swapped = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    _, _, alignment_survivor_only = st.orient_eigenvectors(swapped, previous, survivors=1)
    _, _, alignment_all = st.orient_eigenvectors(swapped, previous, survivors=3)
    assert alignment_survivor_only == pytest.approx(1.0, abs=1e-12)
    assert alignment_all == pytest.approx(0.0, abs=1e-12)


def test_an_unknown_variant_is_refused(small_frame: pd.DataFrame) -> None:
    with pytest.raises(st.StatisticalFactorError, match="is not one of"):
        st.statistical_factor_panel(small_frame, variant="whitened", config=SETTINGS)


def test_a_panel_shorter_than_the_burn_in_is_refused(small_frame: pd.DataFrame) -> None:
    with pytest.raises(st.StatisticalFactorError, match="expanding-window minimum"):
        st.statistical_factor_panel(small_frame.iloc[:100], config=SETTINGS)


def test_both_variants_build_and_detoning_changes_the_factors(
    small_frame: pd.DataFrame,
) -> None:
    """Detoning must not be a no-op -- if it were, carrying two variants is theatre."""
    denoised = st.statistical_factor_panel(small_frame, variant="denoised", config=SETTINGS)
    detoned = st.statistical_factor_panel(small_frame, variant="detoned", config=SETTINGS)
    shared = denoised.factors.index.intersection(detoned.factors.index)
    left = denoised.factors.loc[shared, "pc1"].to_numpy()
    right = detoned.factors.loc[shared, "pc1"].to_numpy()
    assert not np.allclose(left, right)


# ---------------------------------------------------------------------------
# The minimum-variance comparand metric
# ---------------------------------------------------------------------------


def test_minimum_variance_weights_sum_to_one_and_beat_equal_weight(
    synthetic: np.ndarray,
) -> None:
    """A sanity check on the metric itself before it is used to compare estimators.

    The minimum-variance portfolio of the TRUE covariance must realise a lower
    volatility than equal weights. If the metric could not show that, it could not
    show anything about an estimator either.
    """
    covariance = np.cov(synthetic.T)
    positions = [(position, covariance) for position in range(200, 800, 21)]
    minimum = st.minimum_variance_realised_volatility(synthetic, positions, holding_days=21)
    ones = np.ones(synthetic.shape[1])
    equal = st.minimum_variance_realised_volatility(
        synthetic,
        [(position, np.diag(ones)) for position in range(200, 800, 21)],
        holding_days=21,
    )
    assert minimum < equal


def test_the_metric_refuses_a_singular_covariance(synthetic: np.ndarray) -> None:
    singular = np.zeros((synthetic.shape[1], synthetic.shape[1]))
    with pytest.raises((st.StatisticalFactorError, np.linalg.LinAlgError)):
        st.minimum_variance_realised_volatility(synthetic, [(200, singular)], holding_days=21)


# ---------------------------------------------------------------------------
# The config contract -- the W3-P5 burn-in ruling
# ---------------------------------------------------------------------------


def test_the_burn_in_is_the_rates_pca_number_not_a_second_one() -> None:
    """The W3-P5 ruling: reuse within ``factors/``, one number in the file.

    SPEC.md 5.1.2 forbids importing a day count into ``risk/``. It does not forbid
    Model B reading the burn-in that already exists in the ``factors`` namespace
    for the same concept. What it does forbid is TWO numbers for one concept, so
    the config layer copies rather than re-declares.
    """
    settings = load()
    assert (
        settings.model.factors.statistical.expanding_min_window
        == settings.model.factors.macro.rates.pca.expanding_min_window
    )


def test_a_second_burn_in_key_is_rejected_rather_than_ignored() -> None:
    """An ignored key is exactly how the two would silently drift apart."""
    raw = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "model.yaml").read_text()
    )
    node = copy.deepcopy(raw["factors"])
    node["statistical"]["expanding_min_window"] = 99
    with pytest.raises(ConfigError, match="must not be set"):
        _parse_statistical(node, "model.factors", rates_pca=load().model.factors.macro.rates.pca)


def test_the_ewma_correlation_is_refused_as_an_input() -> None:
    """SPEC.md 4.2.1's ruling, enforced at load time rather than by a comment.

    Marchenko-Pastur is derived for equally-weighted iid observations. The EWMA
    correlation is not an available input here, and the config refuses it rather
    than supporting it -- the same shape as ``covariance.mean_convention``.
    """
    raw = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "model.yaml").read_text()
    )
    node = copy.deepcopy(raw["factors"])
    node["statistical"]["estimator"] = "ewma_correlation"
    with pytest.raises(ConfigError, match="not implemented"):
        _parse_statistical(node, "model.factors", rates_pca=load().model.factors.macro.rates.pca)


def test_both_variants_are_always_declared() -> None:
    """Dropping one would be choosing between them, which owes a ``model-config`` row."""
    assert set(SCHEME.variants) == {"denoised", "detoned"}


def test_the_noise_variance_bound_is_the_correlation_matrix_bound() -> None:
    """The upper bound is DERIVED, not chosen: unit variance caps the noise at 1."""
    assert SCHEME.noise_variance_bounds[1] == 1.0
    assert 0.0 < SCHEME.noise_variance_bounds[0] < 1.0


def test_the_kde_bandwidth_is_the_published_value() -> None:
    """SPEC.md 4.2 quotes Lopez de Prado ch. 2's 0.15. It is not adjusted here.

    ``experiments.md`` row 120 measured that the value is not load-bearing for
    Model B's factor count. It is still not moved, because a constant quoted from
    a source is moved by a ruling with a reversing result named in advance, not
    because a sweep found something.
    """
    assert SCHEME.kde_bandwidth == 0.15
