"""SPEC.md 5.3's eigenfactor risk adjustment, and 5.3.1's scale-invariance gate.

**The scale-invariance tests in the first section were written before the
implementation**, which is the whole point of them: SPEC.md 5.3.1 fixed a
mechanical acceptance criterion precisely so that the property would be *known*
rather than believed, and a test written afterwards tests what the code does.

They come with a POWER CONTROL. A test that passes because the property is easy
to satisfy is worth nothing, so ``test_the_covariance_space_alternative_fails_
the_same_test`` runs SPEC.md 5.3's algorithm the literal published way -- on the
covariance rather than on the correlation -- and asserts that it FAILS. That is
the difference between an invariance test and a decoration.

The direction gate on ``lambda(k)`` is here too, and the AMPLITUDE gate is not:
SPEC.md 5.3.2 withdrew it as a large-K shape this model is not in. See that
section for the pre-registered W7 prediction that replaces it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mafrm.config import load
from mafrm.numerics import effective_sample_size
from mafrm.risk.checks import eigenvalue_floor
from mafrm.risk.config import HORIZONS, RiskConfig
from mafrm.risk.covariance import (
    CovarianceError,
    correlation_from_covariance,
    run_pipeline,
)
from mafrm.risk.eigenfactor import (
    EigenfactorError,
    eigenfactor_adjustment,
    equivalent_sample_size,
    ewma_eigenvalue_bias,
    fit_parabola,
    simulated_eigenvalue_bias,
)
from mafrm.risk.synthetic import GOLDEN_FACTOR_COUNTS, panel, specification

#: Small enough to keep the suite fast, large enough that the curve is not
#: noise. The production value is ``eigenfactor.monte_carlo_trials`` and the
#: tests that care about the shipped number say so.
_TRIALS = 250

#: Every scale-invariance assertion is relative and this is the tolerance.
#: DERIVED, not chosen: the two paths multiply the same products in a different
#: order -- ``(c*x)*(c*y)`` against ``c**2 * (x*y)`` -- so they agree to a few
#: ulp per accumulated operation, and ``T`` of them is 5040 here. 1e-11 is four
#: orders of magnitude above that and ten below any real defect.
_RELATIVE = 1.0e-11


#: Every helper here stops after ``eigenfactor``. SPEC.md 5.4's stage is a pure
#: positive scalar multiplication of the whole matrix, so it can neither create
#: nor destroy any property this file tests -- scale invariance, the shape of the
#: curve, the diagonal inflation, PSD -- and it would cost a forecast history per
#: call to say so. Stopping is what ``stop_after`` is for, and the truncation is
#: visible in the build rather than silent.
_THROUGH_EIGENFACTOR = "eigenfactor"


def _covariance(factors: int, horizon: str = "short") -> np.ndarray:
    """The post-repair covariance the eigenfactor stage receives, at ``K``."""
    frame = panel(specification(factors))
    return run_pipeline(
        frame,
        RiskConfig.load(horizon=horizon).for_scaling(1.0),  # type: ignore[arg-type]
        stop_after=_THROUGH_EIGENFACTOR,
    ).matrix


def _rescaled_covariance(factors: int, column: int, constant: float) -> np.ndarray:
    """The same, with one factor's return series multiplied by ``constant``."""
    frame = panel(specification(factors))
    frame.iloc[:, column] = frame.iloc[:, column] * constant
    return run_pipeline(
        frame,
        RiskConfig.load(horizon="short").for_scaling(1.0),
        stop_after=_THROUGH_EIGENFACTOR,
    ).matrix


#: The simulation follows the ESTIMATOR (W3-P3b), so these are its parameters
#: rather than a sample size. 84d over 1,000 observations puts the Kish effective
#: size at ~242, the regime SPEC.md 5.3's table describes, and keeps the suite fast.
_HALFLIFE = 84.0
_OBSERVATIONS = 1000


def _adjust(
    matrix: np.ndarray,
    *,
    trials: int = _TRIALS,
    halflife: float = _HALFLIFE,
    observations: int = _OBSERVATIONS,
) -> object:
    settings = load().model
    return eigenfactor_adjustment(
        matrix,
        trials=trials,
        halflife=halflife,
        observations=observations,
        scalings=settings.eigenfactor.scaling_a,
        seed=settings.seed,
        floor=settings.numerics.psd_eigenvalue_floor,
    )


# ---------------------------------------------------------------------------
# SPEC.md 5.3.1's acceptance criterion -- written before the implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("constant", [1.0e-6, 0.5, 3.0, 1.0e4])
@pytest.mark.parametrize("column", [0, 2])
def test_rescaling_one_factor_leaves_the_adjusted_correlation_unchanged(
    column: int, constant: float
) -> None:
    """The criterion, in its most direct form. SPEC.md 5.3.1.

    Multiply one factor's return series by an arbitrary positive constant. The
    adjusted matrix *in correlation space* -- what the Monte Carlo actually acts
    on -- must not move at all, because nothing about estimation error depends
    on the units a factor is quoted in.
    """
    base = _adjust(_covariance(6))
    rescaled = _adjust(_rescaled_covariance(6, column, constant))
    for scaling in load().model.eigenfactor.scaling_a:
        np.testing.assert_allclose(
            rescaled.variant(scaling).adjusted_correlation,  # type: ignore[attr-defined]
            base.variant(scaling).adjusted_correlation,  # type: ignore[attr-defined]
            rtol=_RELATIVE,
            atol=0.0,
        )


@pytest.mark.parametrize("constant", [1.0e-6, 7.5])
def test_the_adjusted_covariance_transforms_exactly_as_the_rescaling_does(
    constant: float,
) -> None:
    """``F*(rescaled) = S F* S`` for ``S = diag(1, .., c, .., 1)``, any ``c != 0``.

    The adjustment commutes with a diagonal rescaling of the factors. For a
    POSITIVE constant this is exact: ``rho`` is literally unchanged, so the
    simulation draws the identical sample and the answer moves only by the
    outer-product factor. The negative case is separate -- see below.
    """
    column = 1
    selector = np.ones(6)
    selector[column] = constant
    base = _adjust(_covariance(6))
    rescaled = _adjust(_rescaled_covariance(6, column, constant))
    for scaling in load().model.eigenfactor.scaling_a:
        expected = base.variant(scaling).adjusted * np.outer(selector, selector)  # type: ignore[attr-defined]
        np.testing.assert_allclose(
            rescaled.variant(scaling).adjusted,  # type: ignore[attr-defined]
            expected,
            rtol=_RELATIVE,
            atol=0.0,
        )


@pytest.mark.parametrize("trials", [250, 1000])
def test_a_sign_flip_is_invariant_in_distribution_and_the_gap_is_monte_carlo(
    trials: int,
) -> None:
    """A NEGATIVE constant is invariant in distribution, not bit-exactly, and the
    difference is Monte Carlo noise rather than a defect in the invariance.

    Flipping a factor's sign takes ``rho`` to ``S rho S``, a different matrix, so
    the simulation draws a different sample from the same RNG stream. The
    estimator is exactly equivariant; the *Monte Carlo estimate of its bias* is
    equivariant only in distribution. This is a consequence of W3-P3b's ruling
    that the simulation follows the estimator, and it did not arise under the
    equal-weight surrogate.

    It is pinned as ``O(1/sqrt(M))`` rather than as a fixed tolerance, which is
    the claim that distinguishes noise from a defect: quadrupling the trials must
    halve the gap. Measured 3.1e-3 at M=250, 1.4e-3 at M=1000, 5.9e-4 at M=4000.
    All are two orders of magnitude below the 1.75e-2 at which the rejected
    covariance-space form fails, so the power control still discriminates.
    """
    column, constant = 1, -2.0
    selector = np.ones(6)
    selector[column] = constant
    base = _adjust(_covariance(6), trials=trials)
    rescaled = _adjust(_rescaled_covariance(6, column, constant), trials=trials)

    expected = base.variant(1.4).adjusted * np.outer(selector, selector)  # type: ignore[attr-defined]
    observed = rescaled.variant(1.4).adjusted  # type: ignore[attr-defined]
    relative = float(np.max(np.abs(observed - expected)) / np.max(np.abs(expected)))
    assert relative < 6.0 / np.sqrt(trials) * 0.05


@pytest.mark.parametrize("constant", [1.0e-6, 250.0])
def test_every_portfolio_risk_forecast_is_unchanged_by_the_rescaling(constant: float) -> None:
    """The consequence a user of the matrix actually sees. SPEC.md 5.3.1.

    A portfolio holding the same economic position has weights ``w_i / c`` on a
    factor rescaled by ``c``. Its forecast variance must be identical. Tested on
    random portfolios AND on the minimum-variance portfolio of the base matrix,
    because SPEC.md 6.2 is explicit that random portfolios validate any
    covariance matrix at all -- the optimizer-selected one is the test that
    matters, and it is the portfolio the eigenfactor adjustment exists for.
    """
    column = 4
    base = _adjust(_covariance(6))
    rescaled = _adjust(_rescaled_covariance(6, column, constant))
    generator = np.random.default_rng(load().model.seed)
    portfolios = list(generator.standard_normal((32, 6)))

    for scaling in load().model.eigenfactor.scaling_a:
        adjusted = base.variant(scaling).adjusted  # type: ignore[attr-defined]
        inverse = np.linalg.inv(adjusted)
        minimum_variance = inverse @ np.ones(6)
        portfolios.append(minimum_variance / minimum_variance.sum())

        for weights in portfolios:
            equivalent = weights.copy()
            equivalent[column] = equivalent[column] / constant
            np.testing.assert_allclose(
                equivalent @ rescaled.variant(scaling).adjusted @ equivalent,  # type: ignore[attr-defined]
                weights @ adjusted @ weights,
                rtol=_RELATIVE,
                atol=0.0,
            )


def test_the_covariance_space_alternative_fails_the_same_test() -> None:
    """POWER CONTROL. The literal published step does NOT pass this test.

    SPEC.md 5.3 step 1 diagonalises ``F``. In USE4 every factor is already in
    return units so that is exactly right; on a panel that mixes units it makes
    *which direction is the smallest eigenfactor* -- the direction corrected
    hardest -- a function of the numeraire. This test runs the literal form and
    asserts it moves, so that the passing tests above are evidence rather than a
    property that was free.

    Deliberately implemented here rather than shipped: it is the alternative
    that was ruled out, and having it in ``src/`` would invite its use.
    """
    column, constant = 1, 100.0
    settings = load().model
    sample = round(effective_sample_size(84.0))

    def literal(matrix: np.ndarray) -> np.ndarray:
        """SPEC.md 5.3's pseudocode, applied to the covariance as written."""
        eigenvalues, vectors = np.linalg.eigh(matrix)
        eigenvalues = np.where(
            eigenvalues <= 0.0, settings.numerics.psd_eigenvalue_floor, eigenvalues
        )
        generator = np.random.default_rng(settings.seed)
        total = np.zeros(matrix.shape[0])
        for _ in range(_TRIALS):
            draws = generator.standard_normal((matrix.shape[0], sample))
            simulated = vectors @ (np.sqrt(eigenvalues)[:, None] * draws)
            sample_covariance = simulated @ simulated.T / (sample - 1)
            values, directions = np.linalg.eigh(sample_covariance)
            values = np.where(values <= 0.0, settings.numerics.psd_eigenvalue_floor, values)
            total += np.diag(directions.T @ matrix @ directions) / values
        bias = np.sqrt(total / _TRIALS)
        fitted = np.polyval(
            np.polyfit(np.arange(matrix.shape[0]), bias, 2), np.arange(matrix.shape[0])
        )
        gamma = 1.4 * (fitted - 1.0) + 1.0
        return vectors @ np.diag(gamma**2 * eigenvalues) @ vectors.T

    selector = np.ones(6)
    selector[column] = constant
    base = literal(_covariance(6))
    rescaled = literal(_rescaled_covariance(6, column, constant))
    expected = base * np.outer(selector, selector)

    relative = np.max(np.abs(rescaled - expected)) / np.max(np.abs(expected))
    assert relative > 1.0e-3, (
        "the covariance-space form passed the invariance test, which means the test has no "
        f"power on this panel (relative move {relative:.3e}). Fix the control, not the gate."
    )


# ---------------------------------------------------------------------------
# The eight steps of SPEC.md 5.3
# ---------------------------------------------------------------------------


def test_gamma_is_linear_in_lambda_minus_one_and_not_a_power_law() -> None:
    """SPEC.md 5.3 step 7, and the garbled PDF rendering it warns about.

    ``gamma = a (lambda_P - 1) + 1``. One rendering of MSCI's Eq. A8 garbles it
    into ``lambda^a (lambda - 1) + 1``; the open-source implementations confirm
    the linear form. This asserts the linear one exactly and asserts the two are
    distinguishable on this data, so the test would catch the substitution.
    """
    adjustment = _adjust(_covariance(6))
    fitted = adjustment.fit.fitted  # type: ignore[attr-defined]
    for scaling in load().model.eigenfactor.scaling_a:
        gamma = adjustment.variant(scaling).gamma  # type: ignore[attr-defined]
        np.testing.assert_allclose(gamma, scaling * (fitted - 1.0) + 1.0, rtol=0.0, atol=0.0)

    garbled = fitted**1.4 * (fitted - 1.0) + 1.0
    linear = 1.4 * (fitted - 1.0) + 1.0
    assert np.max(np.abs(garbled - linear)) > 1.0e-6, "the two forms are indistinguishable here"


def test_a_equals_one_leaves_gamma_equal_to_the_fitted_curve() -> None:
    """``a = 1`` is the identity on ``(lambda_P - 1)``. USE4's production setting."""
    adjustment = _adjust(_covariance(6))
    np.testing.assert_allclose(
        adjustment.variant(1.0).gamma,  # type: ignore[attr-defined]
        adjustment.fit.fitted,  # type: ignore[attr-defined]
        rtol=0.0,
        atol=0.0,
    )


def test_both_published_scalings_come_from_one_monte_carlo() -> None:
    """Not an optimisation -- it is what makes the two variants COMPARABLE.

    ``a`` enters only at step 7, after the simulation. Running the Monte Carlo
    twice would leave the two variants differing by simulation noise as well as
    by ``a``, and the README paragraph SPEC.md 5.3 asks for compares them.
    """
    adjustment = _adjust(_covariance(6))
    assert tuple(variant.scaling for variant in adjustment.variants) == (  # type: ignore[attr-defined]
        1.0,
        1.4,
    )
    difference = adjustment.variant(1.4).gamma - 1.0  # type: ignore[attr-defined]
    reference = adjustment.variant(1.0).gamma - 1.0  # type: ignore[attr-defined]
    # Absolute, not relative, and DERIVED: gamma is formed as ``a(x) + 1``, so
    # subtracting the 1 back off leaves a residue of a few ulp OF ONE however
    # small ``x`` is. 8 * eps is that residue; a relative tolerance would be
    # asking for precision the representation never had.
    np.testing.assert_allclose(
        difference, 1.4 * reference, rtol=0.0, atol=8.0 * np.finfo(float).eps
    )


def test_the_simulation_is_reproducible_from_the_seed() -> None:
    """CLAUDE.md code conventions: ``default_rng(seed)``, never a global seed."""
    matrix = _covariance(6)
    first = _adjust(matrix)
    second = _adjust(matrix)
    np.testing.assert_array_equal(first.bias, second.bias)  # type: ignore[attr-defined]

    settings = load().model
    other = eigenfactor_adjustment(
        matrix,
        trials=_TRIALS,
        halflife=_HALFLIFE,
        observations=_OBSERVATIONS,
        scalings=settings.eigenfactor.scaling_a,
        seed=settings.seed + 1,
        floor=settings.numerics.psd_eigenvalue_floor,
    )
    assert not np.array_equal(other.bias, first.bias)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# The DIRECTION gate. SPEC.md 5.3.2 -- the amplitude gate is withdrawn
# ---------------------------------------------------------------------------


def _isolated(size: int, spike: float, seed: int) -> np.ndarray:
    """A correlation matrix with a flat bulk and one well-separated eigenvalue."""
    rotation, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((size, size)))
    spectrum = np.ones(size)
    spectrum[-1] = spike
    matrix = rotation @ np.diag(spectrum) @ rotation.T
    return matrix / np.sqrt(np.outer(np.diag(matrix), np.diag(matrix)))


@pytest.mark.parametrize("factors", [6, 40])
def test_lambda_declines_across_the_bulk_on_the_closed_form_control(factors: int) -> None:
    """The HARD gate, on the one matrix where the answer is known in closed form.

    ``F_0 = I`` has **no isolated eigenvalues** -- the whole spectrum is bulk --
    so here the amended bulk gate and the original rank gate coincide, which is
    exactly why this is the matrix the gate is pinned against.

    For ``F_0 = I`` the true risk of every direction is exactly 1, so step 5's
    ``D~`` is identically one and ``lambda(k) = sqrt(mean_m 1/d^m_k)``. Sorting
    ``d`` ascending makes the decline structural: it is a statement about the
    sampling spread of Wishart eigenvalues and nothing else. Both the raw curve
    and the fit must decline at every step and cross 1.

    This is the gate SPEC.md 5.3.2 keeps -- the *direction* of MWO's result,
    which is universal -- with the published *amplitude* withdrawn as a large-K
    shape. A flat or non-monotone curve here means the implementation is wrong.
    """
    adjustment = _adjust(np.eye(factors), trials=1000)
    assert np.all(np.diff(adjustment.bias) < 0.0), f"raw curve not declining: {adjustment.bias}"  # type: ignore[attr-defined]
    assert np.all(np.diff(adjustment.fit.fitted) < 0.0)  # type: ignore[attr-defined]
    assert adjustment.crosses_one  # type: ignore[attr-defined]
    assert adjustment.monotone_decline and adjustment.raw_monotone_decline  # type: ignore[attr-defined]


def test_the_closed_form_control_lands_inside_the_marchenko_pastur_edges() -> None:
    """The MAGNITUDE, against a DERIVED bound rather than a chosen tolerance.

    At ``F_0 = I`` the simulated eigenvalues are Wishart, so for large ``K`` and
    ``T`` at fixed ratio they fall in ``[(1-sqrt(K/T))^2, (1+sqrt(K/T))^2]``.
    Since ``lambda(k) = sqrt(mean 1/d_k)``, the ends of the curve are bounded by
    the reciprocal square roots of those edges. The bound is one-sided in each
    direction -- a finite sample does not reach the asymptotic edge -- and it is
    tight to about 3% here, so a mis-scaled ``T`` or a missing ``(T-1)`` would
    break it immediately.
    """
    factors, sample = 40, 242
    ratio = factors / sample
    lower = (1.0 - np.sqrt(ratio)) ** 2
    upper = (1.0 + np.sqrt(ratio)) ** 2
    # SPEC.md 5.3's EQUAL-WEIGHT simulation, called directly. Marchenko-Pastur
    # describes a Wishart sample; W3-P3b established that an EWMA *correlation*
    # estimator's equivalent sample size is not the Kish figure, so this bound
    # belongs to the published algorithm and not to the estimator the stage runs.
    bias, _, _ = simulated_eigenvalue_bias(
        np.eye(factors), trials=1000, sample_size=sample, seed=load().model.seed, floor=1e-14
    )

    assert bias[0] <= 1.0 / np.sqrt(lower)
    assert bias[-1] >= 1.0 / np.sqrt(upper)
    assert bias[0] > 0.9 / np.sqrt(lower)


@pytest.mark.parametrize("spike", [3.0, 10.0])
def test_an_isolated_eigenvalue_returns_lambda_toward_one(spike: float) -> None:
    """WHY the gate is a statement about the BULK, measured rather than assumed.

    A well-separated eigenvalue is estimated with very little eigenvector
    rotation, so its own risk is forecast almost without bias and ``lambda``
    comes back to ~1 there -- above the bulk's last value, which breaks
    monotonicity **in rank** at exactly that one point. The bulk itself still
    declines.

    This is not a tolerance widened to fit a result. It is the mechanism behind
    ``test_the_golden_fixture_breaks_monotonicity_only_at_its_own_two_spikes``
    below, isolated on a matrix built to contain exactly one spike, and it is
    the reason the production panel -- whose correlation spectrum has no such
    gap -- passes the gate unmodified.
    """
    adjustment = _adjust(_isolated(10, spike, seed=3), trials=1000)
    bulk = adjustment.bias[:-1]  # type: ignore[attr-defined]

    # NET decline across the bulk, not step-by-step. A large spike absorbs
    # enough variance to spread the remaining eigenvalues, so the "bulk" is no
    # longer flat and its curve carries real noise at K = 10. The claim being
    # tested is about the SPIKE, and it is the next two lines.
    assert bulk[0] > 1.0 > bulk[-1]
    assert adjustment.bias[-1] > bulk[-1]  # type: ignore[attr-defined]
    assert abs(adjustment.bias[-1] - 1.0) < abs(bulk[-1] - 1.0)  # type: ignore[attr-defined]


def test_the_golden_fixture_declines_across_its_bulk_and_returns_to_one_at_spikes() -> None:
    """The gate AS AMENDED (W3-P3b): monotone across the bulk, not in rank.

    The fixture is a 2-factor model, so it has exactly 2 isolated eigenvalues.

    ``SyntheticSpec.loadings`` is ``K x 2`` by construction, so the number of
    spikes is a property of the generator rather than a threshold chosen here.
    Across the remaining 38 bulk eigenvalues the raw curve declines at every
    step; the two spikes sit above it, for the reason the control above
    isolates. Pinned so that a genuine regression in the bulk is still caught.
    """
    spec = specification(40)
    spikes = spec.loadings.shape[1]
    adjustment = _adjust(_covariance(40))
    bulk = adjustment.bias[: 40 - spikes]  # type: ignore[attr-defined]

    assert np.all(np.diff(bulk) < 0.0), f"the fixture's bulk is not declining: {bulk}"
    assert bulk[0] > 1.0 > bulk[-1]
    assert np.all(adjustment.bias[40 - spikes :] > bulk[-1])  # type: ignore[attr-defined]


def test_the_amplitude_scales_with_k_over_t_which_is_W7s_prediction() -> None:
    """SPEC.md 5.3.2 pre-registers the amplitude as a W7 prediction, not a gate.

    Here is the mechanism it rests on, on the closed-form control where nothing
    else varies: ``K/T`` up by 6.7x takes the amplitude up several-fold. The W7
    reading is the same comparison at ``K = 56`` against this model's ``K = 6``.
    """
    narrow = _adjust(np.eye(6), trials=1000).amplitude  # type: ignore[attr-defined]
    wide = _adjust(np.eye(40), trials=1000).amplitude  # type: ignore[attr-defined]
    assert wide > 2.0 * narrow


def test_the_amplitude_is_reported_and_is_small_at_this_k_over_t() -> None:
    """Not a gate -- a recorded reading, so W7's prediction has a comparand.

    Asserted only to be positive and far below the published large-K width,
    which is the direction SPEC.md 5.3.2 predicts rather than a tolerance: if
    ``K = 6`` at ``T_eff = 242`` produced MSCI's 0.55 spread, the K-dependence
    story would be wrong and that is worth knowing.
    """
    adjustment = _adjust(_covariance(6))
    assert 0.0 < adjustment.amplitude < 0.55  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# The parabola, and what it does not do at this K
# ---------------------------------------------------------------------------


def test_the_parabola_recovers_an_exact_quadratic_by_hand() -> None:
    """HAND-COMPUTED. Degree-2 OLS on data that IS a parabola is interpolation.

    ``lambda(k) = 1.2 - 0.05 k + 0.002 k^2`` at k = 0..5 gives, by hand,
    ``lambda(0) = 1.2``, ``lambda(1) = 1.152``, ``lambda(2) = 1.108``,
    ``lambda(3) = 1.068``, ``lambda(4) = 1.032``, ``lambda(5) = 1.0``.
    The fit must return those coefficients and reproduce those six values.
    """
    values = np.array([1.2, 1.152, 1.108, 1.068, 1.032, 1.0])
    fit = fit_parabola(values)
    np.testing.assert_allclose(fit.fitted, values, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(fit.coefficients, [1.2, -0.05, 0.002], rtol=0.0, atol=1.0e-12)
    assert fit.residual_degrees_of_freedom == 3
    assert fit.residual_standard_deviation == pytest.approx(0.0, abs=1.0e-12)
    assert not fit.exact_interpolation


def test_at_k3_the_parabola_is_an_exact_interpolation_and_smooths_nothing() -> None:
    """SPEC.md 5.3.2 asks for this to be REPORTED rather than assumed away.

    Three parameters on three points. The fitted curve is the raw curve, the
    residual is zero by construction, and there are zero residual degrees of
    freedom -- so a reader who assumes Monte Carlo noise was removed at ``K = 3``
    is wrong, and the object says so.
    """
    fit = fit_parabola(np.array([1.10, 1.00, 0.93]))
    assert fit.residual_degrees_of_freedom == 0
    assert fit.exact_interpolation
    np.testing.assert_allclose(fit.fitted, [1.10, 1.00, 0.93], rtol=0.0, atol=1.0e-12)


def test_at_k6_the_smoothing_has_three_residual_degrees_of_freedom() -> None:
    """The number SPEC.md 5.3.2 reports for this model. Weak, not absent."""
    adjustment = _adjust(_covariance(6))
    assert adjustment.fit.residual_degrees_of_freedom == 3  # type: ignore[attr-defined]
    assert not adjustment.fit.exact_interpolation  # type: ignore[attr-defined]
    assert adjustment.fit.residual_standard_deviation > 0.0  # type: ignore[attr-defined]


def test_the_fit_smooths_a_spike_rather_than_reproducing_it() -> None:
    """The fit is doing its job at all: one displaced point does not survive it."""
    clean = np.array([1.10, 1.06, 1.03, 1.00, 0.98, 0.96])
    spiked = clean.copy()
    spiked[2] += 0.05
    assert abs(fit_parabola(spiked).fitted[2] - clean[2]) < 0.05


def test_a_parabola_needs_three_points() -> None:
    with pytest.raises(EigenfactorError, match="at least 3"):
        fit_parabola(np.array([1.1, 0.9]))


# ---------------------------------------------------------------------------
# What the adjustment does to the matrix
# ---------------------------------------------------------------------------


def test_the_adjusted_correlation_diagonal_is_inflated_and_NOT_renormalised() -> None:
    """SPEC.md 5.3.2's obligation, and the trap that mirrors SPEC.md 5.2.1's.

    Newey-West's correlation leg MUST be renormalised to a unit diagonal or
    every variance is counted twice. The eigenfactor adjustment MUST NOT be, or
    the correction is deleted: the risk increase the whole stage exists to
    produce arrives as exactly this diagonal inflation. The two rules are three
    stages apart in one file and a later session that makes them consistent
    breaks one of them.
    """
    adjustment = _adjust(_covariance(6))
    for scaling in load().model.eigenfactor.scaling_a:
        diagonal = np.diag(adjustment.variant(scaling).adjusted_correlation)  # type: ignore[attr-defined]
        assert not np.allclose(diagonal, 1.0, atol=1.0e-6), (
            f"the correction was renormalised away: {diagonal}"
        )
        # Upward ON NET, not factor by factor. gamma < 1 on the largest
        # eigenfactors, so a factor sitting mostly in one of them legitimately
        # has its own variance revised DOWN; what the stage guarantees is that
        # the total is revised up, because the low-variance directions -- the
        # ones an optimizer loads onto -- dominate the correction.
        assert diagonal.sum() > adjustment.factors  # type: ignore[attr-defined]


def test_the_adjustment_raises_risk_and_a_equals_1_4_raises_it_more() -> None:
    """The direction the stage exists for, and the ordering of the two variants.

    MWO: the optimizer loads onto the most under-forecast directions, so the
    correction is upward on net. ``a = 1.4`` scales the same ``(lambda - 1)``
    harder than ``a = 1.0``, so it must sit further from the input in the same
    direction -- that ORDERING is the claim, not any magnitude.
    """
    matrix = _covariance(6)
    adjustment = _adjust(matrix)
    mild = adjustment.variant(1.0).adjusted  # type: ignore[attr-defined]
    strong = adjustment.variant(1.4).adjusted  # type: ignore[attr-defined]

    assert np.trace(mild) > np.trace(matrix)
    assert np.trace(strong) > np.trace(mild)
    assert np.linalg.norm(strong - matrix) > np.linalg.norm(mild - matrix)


@pytest.mark.parametrize("factors", GOLDEN_FACTOR_COUNTS)
def test_the_adjusted_matrix_is_psd_and_symmetric_at_both_fixture_sizes(factors: int) -> None:
    """``gamma**2 D`` is non-negative whatever ``gamma`` is, so PSD is structural."""
    numerics = load().model.numerics
    adjustment = _adjust(_covariance(factors))
    for scaling in load().model.eigenfactor.scaling_a:
        adjusted = adjustment.variant(scaling).adjusted  # type: ignore[attr-defined]
        spectrum = np.linalg.eigvalsh(adjusted)
        assert spectrum[0] >= eigenvalue_floor(len(spectrum), spectrum[-1], numerics=numerics)
        assert np.max(np.abs(adjusted - adjusted.T)) <= numerics.symmetry_absolute_tolerance


def test_the_correlation_split_is_the_pipelines_own(factors: int = 6) -> None:
    """The stage splits with ``correlation_from_covariance``, not a second copy."""
    matrix = _covariance(factors)
    adjustment = _adjust(matrix)
    np.testing.assert_allclose(
        adjustment.correlation,  # type: ignore[attr-defined]
        correlation_from_covariance(matrix),
        rtol=0.0,
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# The stage inside the pipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", HORIZONS)
def test_the_pipeline_refuses_to_pick_a_scaling_for_you(horizon: str) -> None:
    """Both are published; choosing one is a MODELLING decision, not a default.

    CLAUDE.md's table says run both. A default here would let a later session
    inherit ``a = 1.4`` without noticing, and picking one because it flattered a
    result is a ``model-config`` trial that would then go unlogged.
    """
    frame = panel(specification(6))
    with pytest.raises(CovarianceError, match="eigenfactor_variant"):
        run_pipeline(
            frame,
            RiskConfig.load(horizon=horizon),  # type: ignore[arg-type]
            stop_after=_THROUGH_EIGENFACTOR,
        )


def test_an_unpublished_scaling_is_refused() -> None:
    """CLAUDE.md invariant 9. 1.0 and 1.4 are published; 1.2 is not."""
    with pytest.raises(ValueError, match=r"1\.2"):
        RiskConfig.load(horizon="short").for_scaling(1.2)


def test_the_stage_takes_no_sample_size_and_simulates_the_estimator() -> None:
    """W3-P3b. There is no ``T`` to choose, so there is none to get wrong.

    SPEC.md 5.3 names ``T = T_eff`` and simulates an equal-weight sample. W3-P3b
    measured what an EWMA *correlation* estimator's equivalent equal-weight size
    actually is and found the excess is not a constant -- 1.00 to 1.50 across
    ``K`` and half-life. So the simulation follows the estimator, which removes
    the parameter rather than replacing one guess with another.
    """
    assert not hasattr(RiskConfig.load(horizon="short"), "eigenfactor_sample_size")
    adjustment = _adjust(np.eye(6))
    assert adjustment.halflife == _HALFLIFE  # type: ignore[attr-defined]
    assert adjustment.observations == _OBSERVATIONS  # type: ignore[attr-defined]


@pytest.mark.parametrize("scaling", [1.0, 1.4])
def test_the_stage_is_horizon_independent_by_construction(scaling: float) -> None:
    """A CONSEQUENCE of W3-P3b, pinned so it is not later read as a bug.

    The matrix being diagonalised is ``rho_hat``, estimated at the correlation
    half-life -- and SPEC.md 5.1 puts that at **504d at both horizons**,
    deliberately. So the eigenfactor correction is now identical short and long.
    Under the superseded ``T = 242/727`` the two differed, and that difference was
    an artefact of the wrong sample size rather than a property of the model.
    """
    frame = panel(specification(6))
    short = run_pipeline(
        frame,
        RiskConfig.load(horizon="short").for_scaling(scaling),
        stop_after=_THROUGH_EIGENFACTOR,
    )
    long = run_pipeline(
        frame,
        RiskConfig.load(horizon="long").for_scaling(scaling),
        stop_after=_THROUGH_EIGENFACTOR,
    )
    assert short.eigenfactor is not None and long.eigenfactor is not None
    # Not bit-identical: rho is recovered from ``rho * outer(sigma, sigma)`` with a
    # different sigma at each horizon, so the two differ at ~1e-17 and the
    # simulation inherits it. Anything above roundoff means the stage is still
    # reading a horizon-dependent input.
    np.testing.assert_allclose(short.eigenfactor.bias, long.eigenfactor.bias, rtol=1e-12)
    for mild, strong in zip(short.eigenfactor.variants, long.eigenfactor.variants, strict=True):
        np.testing.assert_allclose(mild.gamma, strong.gamma, rtol=1e-12)


def test_the_equivalent_sample_size_instrument_recovers_a_known_answer() -> None:
    """The W3-P3b instrument, validated against a case whose answer is known.

    With a half-life so long the weights are flat, the equivalent equal-weight
    ``T`` must be the window length itself. If the instrument cannot recover that,
    its reading on the real panel means nothing. It recovers it **exactly** --
    but only un-normalised, which is the finding, not a caveat.
    """
    size, observations = 6, 300
    rotation, _ = np.linalg.qr(np.random.default_rng(4).standard_normal((size, size)))
    matrix = rotation @ np.diag(np.linspace(0.6, 1.6, size)) @ rotation.T
    matrix = matrix / np.sqrt(np.outer(np.diag(matrix), np.diag(matrix)))
    shared = {
        "halflife": 1.0e6,
        "observations": observations,
        "grid": (150, 200, 250, 300, 350, 450),
        "trials": 1500,
        "seed": load().model.seed,
        "floor": 1e-14,
    }

    exact = equivalent_sample_size(matrix, normalise=False, **shared)  # type: ignore[arg-type]
    assert exact.equivalent == observations, (
        "the instrument cannot recover a known answer, so its reading on the real panel "
        f"means nothing: got {exact.equivalent} for a flat-weight window of {observations}"
    )
    normalised = equivalent_sample_size(matrix, normalise=True, **shared)  # type: ignore[arg-type]
    assert normalised.equivalent > observations


def test_the_normalisation_is_the_dominant_effect_not_a_second_order_one() -> None:
    """W3-P3b row 111, which REFUTED the hypothesis it was registered with.

    The prediction was that normalising to a unit diagonal would be second-order
    beside the exponential weighting. It is the reverse: un-normalised the EWMA
    estimator's equivalent sample size is the Kish figure to within 1%, and
    **all** of the excess comes from the normalisation. Removing the variance
    error from the diagonal removes part of the eigenvalue dispersion, so a
    correlation estimator behaves like a longer sample than its variance-matched
    size implies.

    Pinned in the direction the measurement found, so a later session cannot
    quietly restore the intuition the control refuted.
    """
    size = 6
    rotation, _ = np.linalg.qr(np.random.default_rng(9).standard_normal((size, size)))
    matrix = rotation @ np.diag(np.linspace(0.5, 1.7, size)) @ rotation.T
    matrix = matrix / np.sqrt(np.outer(np.diag(matrix), np.diag(matrix)))
    shared = {"halflife": 84.0, "observations": 1000, "trials": 1500, "seed": 3, "floor": 1e-14}

    normalised = ewma_eigenvalue_bias(matrix, normalise=True, **shared)  # type: ignore[arg-type]
    raw = ewma_eigenvalue_bias(matrix, normalise=False, **shared)  # type: ignore[arg-type]

    assert (normalised[0] - normalised[-1]) < (raw[0] - raw[-1])
    assert np.max(np.abs(normalised - raw)) > 0.1 * float(raw[0] - raw[-1])


@pytest.mark.parametrize("scaling", [1.0, 1.4])
def test_the_stage_runs_inside_the_pipeline_and_changes_the_matrix(scaling: float) -> None:
    """End to end: all five stages, PSD asserted after each, and stage 4 bites.

    W3-P4 added SPEC.md 5.4, so ``stages_not_implemented`` is empty and this test
    runs the whole declared pipeline. The regime stage is handed a perfectly
    calibrated history -- ``B_t^F = 1`` everywhere, so ``lambda_F^2 = 1`` -- which
    keeps the trace comparison a statement about the eigenfactor stage alone.
    """
    frame = panel(specification(6))
    config = RiskConfig.load(horizon="short").for_scaling(scaling)
    build = run_pipeline(frame, config, regime_bias=np.ones(400))

    names = [stage.name for stage in build.stages]
    assert names == ["ewma", "newey_west", "psd_repair", "eigenfactor", "volatility_regime"]
    repaired = next(stage for stage in build.stages if stage.name == "psd_repair").matrix
    assert np.trace(build.matrix) > np.trace(repaired)
    assert build.eigenfactor is not None
    assert build.stages_not_implemented == ()
    assert build.complete


def test_the_build_reports_the_curve_and_the_fit() -> None:
    """``make model`` prints the K-dependence numbers, not just a green line."""
    frame = panel(specification(6))
    build = run_pipeline(
        frame,
        RiskConfig.load(horizon="short").for_scaling(1.4),
        stop_after=_THROUGH_EIGENFACTOR,
    )
    rendered = "\n".join(build.render())
    assert "lambda" in rendered
    assert "residual d.o.f." in rendered
