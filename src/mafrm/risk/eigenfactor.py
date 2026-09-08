"""SPEC.md 5.3's eigenfactor risk adjustment. Menchero, Wang & Orr (2011).

**The central technique of the project.** It is what makes the risk-model term
in SPEC.md 1's identity move.

THE PROBLEM
-----------

Sampling error in ``F_hat`` is not neutral. Diagonalise it and the eigenfactors'
variances are systematically mis-estimated: small eigenvalues biased **down**,
large ones slightly **up**. An optimizer seeking minimum risk deliberately loads
onto exactly the low-variance eigenfactors -- that is what minimising risk
*means* -- so it loads onto the most under-forecast directions. In MSCI's words,
*"portfolios with the lowest forecast risk are those whose risk is underestimated
the most."* Hence optimized portfolios' risk is under-predicted even where random
portfolios' risk is not, which is why SPEC.md 6.2 says the test that matters is
run on optimizer-selected portfolios and not on random ones.

The correction is a Monte Carlo measurement of that bias, direction by direction,
and a deterministic rescaling of the eigenvalues by what it finds.

THIS RUNS IN CORRELATION SPACE, AND THAT IS A RULING
----------------------------------------------------

SPEC.md 5.3 step 1 diagonalises ``F``. **This implementation diagonalises the
correlation matrix instead and rescales by ``sigma`` afterwards.** SPEC.md 5.3.1
and 5.3.2 carry the full argument; the short form is:

**USE4 is not wrong and nothing here should be read as saying it is.** Every
factor in USE4 is already in return units, so a diagonal rescaling across factors
is not something that can happen there and the question never arises.
Diagonalising ``F`` is exactly right in that setting.

What would be wrong is following the published step **literally into a case it
does not cover.** Eigenvectors of a covariance matrix are not equivariant under a
diagonal rescaling of the underlying variables, so on a panel that deliberately
mixes decimal returns with basis points of yield -- which SPEC.md 4.1.1 chose, in
order to make a level-factor coefficient read as an effective duration directly
-- *which direction is the smallest eigenfactor*, the direction this correction
acts on hardest, can be decided by the numeraire before it is decided by the
data. ``experiments.md`` row 96 measures that on this project's own panel:
``cond(F) = 9.96e6`` against ``cond(rho) = 3.27``, essentially all of the gap
being a 7.3e6 variance ratio across the mixed-unit columns.

A risk correction whose output changes when a factor is quoted in decimals rather
than in basis points is not a risk correction. So the correction is computed on
the one form of the matrix that has no units in it, and the units are put back
afterwards by the same ``sigma`` they were taken out with -- three lines apart in
this file, which is the whole reason this resolution was chosen over moving the
panel to a common numeraire (SPEC.md 5.3.2's tiebreak: fewest places where units
bookkeeping happens). ``tests/test_eigenfactor.py`` asserts the invariance
mechanically, and asserts that the literal covariance-space form *fails* the same
test, so the property is known rather than believed.

THE DIAGONAL IS INFLATED AND MUST NOT BE RENORMALISED
-----------------------------------------------------

``gamma**2`` scales the eigenvalues of ``rho``, which moves its trace, so the
adjusted matrix in correlation space is **not** a correlation matrix. That is the
correction. Renormalising the diagonal back to one would delete it: the risk
increase this stage exists to produce arrives as exactly that diagonal inflation.

This is the mirror image of the trap in :func:`~mafrm.risk.covariance.
newey_west_covariance`, where the correlation leg **must** be renormalised or
every variance is counted twice -- there the scale is already carried by
``sigma``, here it has nowhere else to live. **The two rules are opposite, they
are three stages apart in the same pipeline, and a later session that makes them
consistent with each other breaks one of them.** Both are pinned by tests.

THERE IS NO ``T``: THE SIMULATION USES THE ESTIMATOR'S OWN WEIGHTS
------------------------------------------------------------------

SPEC.md 5.3's steps 2-3 simulate an **equally weighted** sample of ``T`` rows and
form ``f f' / (T-1)``, and the section names ``T`` as the EWMA effective sample
size ``2*tau/ln 2`` -- 242 days at the 84-day half-life, 727 at the 252-day.
**This implementation does not use an equal-weight surrogate at all.** It draws
``observations`` rows and runs them through
:func:`~mafrm.risk.covariance.ewma_second_moment` at the correlation half-life and
then :func:`~mafrm.risk.covariance.correlation_from_covariance` -- the production
estimator, not a model of it. Steps 1 and 4-8 are unchanged.

**Why the deviation, in the order the reasoning actually went (W3-P3b, SPEC.md
5.3.2).**

W3-P3 first kept the published 242/727 and defended it three ways, and all three
fail. The spec states those numbers **for covariance-space diagonalisation**,
which SPEC.md 5.3.2 replaced -- so the appeal to the spec is an appeal to a
superseded procedure. Comparability with Shepard's cross-check cuts both ways: two
quantities evaluated at a ``K/T`` that describes neither estimator are comparably
wrong, not jointly right. And "the direction is conservative" is a tiebreak, not a
justification.

The principle that replaces them is that ``T`` must follow **the matrix being
diagonalised**, which after SPEC.md 5.3.2 is ``rho_hat`` at the *correlation*
half-life. That argues for ``T_eff(504) = 1454``. Rather than substitute one
formula-derived number for another, W3-P3b **measured** what the EWMA correlation
estimator's equivalent equal-weight sample size actually is --
:func:`equivalent_sample_size` is that measurement and it is still here.

**The measurement refuted the hypothesis it registered, and its control found
out why.** Three results, in the order they settle the question:

1. **``2*tau/ln 2`` transfers exactly -- for the SECOND-MOMENT estimator.**
   Un-normalised, the EWMA estimator's equivalent equal-weight ``T`` came out at
   **1.007x** the Kish figure on this project's own panel, and a flat-weight control (a
   half-life so long the weights are uniform, where the answer must be the window
   length) recovers **300 of 300 exactly**. Exponential weighting is not the
   problem and the formula is not wrong about what it describes.
2. **Normalising to a correlation matrix is what breaks it, and that was the
   registered control rather than a lucky guess.** The same flat-weight case
   normalised returns **400** instead of 300; that panel normalised returns
   **1890** instead of 1454. Removing the variance error from the diagonal removes
   part of the eigenvalue dispersion, so a *correlation* estimator behaves like a
   longer sample than its variance-matched size implies. ``experiments.md`` row
   111 predicted this would be second-order; **it is the whole of the effect.**
3. **The correction factor is not universal, so it cannot be a constant.** Across
   ``K`` in {3, 6, 40} and half-lives 84d and 504d it runs from **1.00 to 1.50**
   -- stable at 1.30 for ``K = 6`` on two different matrices, but 1.10 and 1.50
   elsewhere. It depends on the spectrum, which is exactly what a configuration
   constant cannot.

So there is no constant to put in ``config/model.yaml`` and no formula to apply
to the estimator this stage actually diagonalises.
Simulating the estimator itself removes the parameter instead of replacing it,
and it is **asset-class agnostic by construction** (CLAUDE.md invariant 10): the
simulation adapts to whatever panel and half-life it is handed, which a calibrated
``T`` could not do without being re-measured for every panel.

``K/T_eff`` is still **reported** -- it is the project's headline axis and a
reader needs it to place a build -- but nothing here consumes it, so there is no
longer a number for this stage to get wrong.

WHAT THE CORRELATION-SPACE RULING COSTS, STATED PLAINLY
-------------------------------------------------------

In covariance space the adjustment corrected the estimation error of ``sigma`` and
``rho`` **jointly**, because both live inside ``F``. In correlation space it
corrects ``rho``'s error alone: **``sigma``'s estimation error is untouched and
passes through this stage uncorrected.** The stage's coverage changed with the
numeraire, and SPEC.md 5.3.2's ruling did not name that when it was taken.

It is recorded rather than repaired. What to do about the uncorrected ``sigma``
leg is open and belongs with SPEC.md 5.4's volatility regime adjustment, which is
the other stage that acts on the level of risk -- see ``experiments.md`` under
W3-P3b for the forward note.

WHAT THIS MODULE MAY KNOW
-------------------------

A square matrix and some numbers. CLAUDE.md invariant 10: no branch here on what
the columns are, how many there are, or where they came from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mafrm.numerics import ewma_weights, realised_effective_sample_size
from mafrm.risk.covariance import correlation_from_covariance, ewma_second_moment

__all__ = [
    "EigenfactorAdjustment",
    "EigenfactorError",
    "EigenfactorVariant",
    "EquivalentSampleSize",
    "ParabolaFit",
    "eigenfactor_adjustment",
    "equivalent_sample_size",
    "ewma_eigenvalue_bias",
    "fit_parabola",
    "simulated_eigenvalue_bias",
]

#: SPEC.md 5.3 step 7 fits a *parabola*. Degree is part of the published method,
#: not a tunable, so it is named here rather than added to ``config/model.yaml``
#: -- CLAUDE.md invariant 6 governs numbers that could have been chosen
#: differently, and this one could not.
_PARABOLA_DEGREE: int = 2


class EigenfactorError(ValueError):
    """The eigenfactor adjustment was handed something it cannot correct."""


@dataclass(frozen=True)
class ParabolaFit:
    """SPEC.md 5.3 step 7's smoothing, and an honest account of how much it did.

    Plain degree-2 OLS in the eigenvalue **index**, which is what MWO do. The
    fit exists to smooth Monte Carlo noise at the ends of the spectrum, where
    ``lambda`` is least well determined.

    **At small ``K`` it barely smooths, and at ``K = 3`` it does not smooth at
    all.** Three parameters on ``K`` points leaves ``K - 3`` residual degrees of
    freedom: 3 at this project's ``K = 6``, and **zero** on SPEC.md 15.2's
    ``K = 3`` fixture, where the fit is an exact interpolation and the fitted
    curve *is* the raw curve. :attr:`exact_interpolation` says so out loud
    rather than leaving a reader to assume noise was removed when none was.
    SPEC.md 5.3.2 asks for this to be reported; it is the third instance of the
    pattern ``reports/stage_k_dependence.md`` tracks.
    """

    #: Ascending powers of the index: ``c0 + c1 k + c2 k^2``.
    coefficients: np.ndarray
    #: ``lambda_P(k)``, the smoothed curve step 7 consumes.
    fitted: np.ndarray
    #: ``K - 3``. Zero means the fit interpolates and removes nothing.
    residual_degrees_of_freedom: int
    #: Root-mean-square residual, on ``K`` points rather than on the d.o.f.,
    #: because at zero d.o.f. the latter is undefined and this one is exactly 0.
    residual_standard_deviation: float

    @property
    def exact_interpolation(self) -> bool:
        """True when there are no residual degrees of freedom left to smooth with."""
        return self.residual_degrees_of_freedom == 0

    def render(self) -> str:
        """One line, and it leads with the number a reader is most likely to assume."""
        if self.exact_interpolation:
            return (
                "parabola: EXACT INTERPOLATION -- 3 parameters on 3 points, 0 residual d.o.f., "
                "no Monte Carlo noise is removed at this K"
            )
        return (
            f"parabola: {self.residual_degrees_of_freedom} residual d.o.f., "
            f"rms residual {self.residual_standard_deviation:.4e}"
        )


def fit_parabola(values: np.ndarray) -> ParabolaFit:
    """Degree-2 OLS of ``values`` on the index ``0..K-1``. SPEC.md 5.3 step 7.

    The abscissa is the **eigenvalue index**, ascending with the eigenvalue, not
    the eigenvalue itself. SPEC.md 5.3 says only "a parabola fitted to lambda(k)";
    MWO fit against the index and so does this.
    """
    if values.ndim != 1:
        raise EigenfactorError(f"fit_parabola: expected a 1-D curve, got {values.shape}")
    size = values.size
    if size < _PARABOLA_DEGREE + 1:
        raise EigenfactorError(
            f"fit_parabola: a parabola needs at least 3 points, got {size}. Below that the "
            "fit is underdetermined -- there is no smoothing decision to make, and inventing "
            "one would be inventing a method."
        )
    index = np.arange(size, dtype=float)
    # ``polyfit`` returns highest power first; the stored order is ascending so
    # that ``coefficients[2]`` is the curvature a reader goes looking for.
    descending = np.polyfit(index, values, _PARABOLA_DEGREE)
    fitted: np.ndarray = np.polyval(descending, index)
    residual = values - fitted
    return ParabolaFit(
        coefficients=np.asarray(descending[::-1], dtype=float),
        fitted=fitted,
        residual_degrees_of_freedom=int(size - (_PARABOLA_DEGREE + 1)),
        residual_standard_deviation=float(np.sqrt(np.mean(np.square(residual)))),
    )


def simulated_eigenvalue_bias(
    matrix: np.ndarray,
    *,
    trials: int,
    sample_size: int,
    seed: int,
    floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SPEC.md 5.3 steps 1-6. Returns ``(lambda, D_0, U_0)``, eigenvalues ascending.

    Written to follow the eight published steps in the published order rather
    than in the algebraically equivalent cheaper form. ``U_0 b_m`` could be
    skipped -- the eigenvalues of ``f_m f_m' / (T-1)`` are those of
    ``b_m b_m' / (T-1)`` and the rotation cancels in step 5 -- but at ``K`` in
    the tens the saving is nothing and a reader checking this against MWO should
    find MWO.

    Step by step, with the published equation numbers in SPEC.md 5.3:

    1. ``F_0 = U_0 D_0 U_0'``, with non-positive eigenvalues floored as the
       pseudocode writes it.
    2. Trial ``m``: draw ``b^m``, ``K x T``, row ``k`` iid ``N(0, D_0(k))``, and
       set ``f^m = U_0 b^m``. These are simulated factor returns whose *true*
       covariance is exactly ``F_0``.
    3. ``F^m = f^m (f^m)' / (T - 1)`` -- the estimate a modeller would have made
       from that sample.
    4. ``F^m = U^m D^m (U^m)'``.
    5. ``D~^m = diag((U^m)' F_0 U^m)`` -- the **true** risk of the portfolio the
       simulation thought was eigenfactor ``k``.
    6. ``lambda(k) = sqrt(mean_m D~^m(k) / D^m(k))``.

    ``lambda`` above 1 means the direction's risk is under-estimated. The result
    declines monotonically in ``k`` and crosses 1: small eigenvalues are biased
    down, large ones slightly up. That *shape* is universal. Its **amplitude is
    not** -- it scales with ``K/T`` -- which is why SPEC.md 5.3.2 withdrew
    SPEC.md 5.3's published "1.5 down to 0.95" as a gate and pre-registered the
    amplitude as a week-7 prediction instead.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise EigenfactorError(
            f"simulated_eigenvalue_bias: expected a square matrix, got {matrix.shape}"
        )
    size = matrix.shape[0]
    if trials < 1:
        raise EigenfactorError(f"simulated_eigenvalue_bias: trials must be positive, got {trials}")
    if sample_size <= size:
        raise EigenfactorError(
            f"simulated_eigenvalue_bias: T = {sample_size} at K = {size}. Below T > K the "
            "simulated sample covariance is singular by construction and the bias this "
            "measures is not defined -- it is not a small-sample regime, it is no regime."
        )

    eigenvalues, vectors = np.linalg.eigh(matrix)
    eigenvalues = np.where(eigenvalues <= 0.0, floor, eigenvalues)
    deviations = np.sqrt(eigenvalues)

    generator = np.random.default_rng(seed)
    total = np.zeros(size)
    for _ in range(trials):
        simulated = vectors @ (deviations[:, None] * generator.standard_normal((size, sample_size)))
        estimate = simulated @ simulated.T / (sample_size - 1)
        estimated, directions = np.linalg.eigh(estimate)
        estimated = np.where(estimated <= 0.0, floor, estimated)
        true_risk = np.einsum("ij,jk,ki->i", directions.T, matrix, directions)
        total += true_risk / estimated
    bias: np.ndarray = np.sqrt(total / trials)
    return bias, eigenvalues, vectors


@dataclass(frozen=True)
class EigenfactorVariant:
    """One value of ``a``, and what it produced. SPEC.md 5.3's two published settings.

    ``a = 1.0`` is attribution-facing and is what USE4 runs **in production**;
    ``a = 1.4`` is optimizer-facing and is the empirical scaling MSCI publish.
    They are two model variants, not alternatives to choose between: 1.4
    overstates the volatilities of the pure factors themselves and corrupts
    attribution, which is exactly why USE4 ships 1.0 and quotes 1.4.
    """

    scaling: float
    #: ``gamma(k) = a (lambda_P(k) - 1) + 1``.
    gamma: np.ndarray
    #: ``U_0 diag(gamma^2 D_0) U_0'``. Diagonal deliberately NOT unit.
    adjusted_correlation: np.ndarray
    #: The same, rescaled back by ``sigma``. This is the stage's output.
    adjusted: np.ndarray

    def render(self) -> str:
        """One line per variant, with the range of the multiplier it applied."""
        return (
            f"a = {self.scaling:.1f}: gamma in [{self.gamma.min():.4f}, {self.gamma.max():.4f}], "
            f"variance multiplier in [{self.gamma.min() ** 2:.4f}, {self.gamma.max() ** 2:.4f}]"
        )


@dataclass(frozen=True)
class EigenfactorAdjustment:
    """Everything one Monte Carlo produced, for every published ``a``.

    **One simulation, both variants**, and that is not an optimisation. ``a``
    enters only at step 7, after the simulation has finished; running the Monte
    Carlo twice would leave the two variants differing by simulation noise as
    well as by ``a``, and the comparison SPEC.md 5.3 asks the README to make is
    between the two values of ``a`` alone.
    """

    #: ``rho``, the matrix the simulation actually ran on.
    correlation: np.ndarray
    #: ``sigma``, split off before the correction and put back after it.
    deviations: np.ndarray
    #: ``D_0``, ascending.
    eigenvalues: np.ndarray
    #: ``U_0``, columns matching :attr:`eigenvalues`.
    eigenvectors: np.ndarray
    #: ``lambda(k)``, raw from the simulation and unsmoothed.
    bias: np.ndarray
    fit: ParabolaFit
    trials: int
    #: The correlation half-life the estimator -- and therefore the simulation --
    #: runs at. SPEC.md 5.1 puts it at 504d at BOTH horizons.
    halflife: float
    #: The window length the estimator saw. The simulation uses the same one.
    observations: int
    variants: tuple[EigenfactorVariant, ...]

    @property
    def factors(self) -> int:
        """``K``."""
        return int(self.correlation.shape[0])

    @property
    def effective_sample_size(self) -> float:
        """Kish's ``(sum w)^2 / sum w^2`` for the window the estimator saw.

        **Descriptive, not an input.** Nothing in this adjustment consumes it:
        the simulation uses the estimator's own weights over its own window, so
        there is no effective sample size for it to get wrong. It is reported
        because ``K/T`` is the project's headline parameter and a reader needs
        the number to place a build on that axis.
        """
        return realised_effective_sample_size(ewma_weights(self.observations, self.halflife))

    @property
    def k_over_t(self) -> float:
        """``K / T_eff``, descriptive. The parameter the amplitude scales with."""
        return self.factors / self.effective_sample_size

    @property
    def amplitude(self) -> float:
        """``lambda_P(0) - lambda_P(K-1)``: how wide the correction is.

        **Reported, never gated.** SPEC.md 5.3.2 withdrew the published
        1.5-to-0.95 amplitude as a large-K shape and pre-registered this
        quantity as a week-7 prediction: it must be visibly larger at ``K = 56``
        than at ``K = 6`` on the same code.
        """
        return float(self.fit.fitted[0] - self.fit.fitted[-1])

    @property
    def monotone_decline(self) -> bool:
        """The hard gate, **as amended in W3-P3b: it is a claim about the BULK.**

        ``lambda`` declines in eigenvalue rank *across the bulk of the spectrum*,
        and returns toward 1 at any **isolated** eigenvalue. The mechanism is
        measured, not assumed (``experiments.md`` row 107): the bias is driven by
        eigenvector rotation, and a well-separated eigenvalue suffers almost none,
        so its own risk is forecast nearly without bias.

        **The original wording -- monotone in rank -- is wrong for any spiked
        spectrum**, and a later session must not read a return-to-1 at a dominant
        factor as a defect. This property reports monotonicity across the whole
        rank ordering, which is the right gate only when the spectrum has no
        isolated eigenvalue; :meth:`bulk_monotone_decline` is the amended form.

        The gate is on the fit rather than the raw curve because the raw curve
        carries Monte Carlo noise of order ``1/sqrt(M)`` at every point, and step
        7 exists to remove it. :attr:`raw_monotone_decline` reports the unsmoothed
        answer beside it rather than hiding the distinction.
        """
        return bool(np.all(np.diff(self.fit.fitted) < 0.0))

    def bulk_monotone_decline(self, spikes: int) -> bool:
        """The AMENDED gate: monotone across the ``K - spikes`` bulk eigenvalues.

        ``spikes`` is the number of isolated eigenvalues at the top of the
        spectrum and is a property of the matrix the caller knows and this module
        does not -- the golden fixture has exactly two because its loading matrix
        has two columns, and a cross-sectional panel typically has a dominant
        first factor plus whatever further structure separates from the bulk.
        Passing 0 recovers the un-amended rank gate.
        """
        if spikes < 0 or spikes >= self.factors:
            raise EigenfactorError(
                f"bulk_monotone_decline: spikes must be in [0, {self.factors}), got {spikes}"
            )
        bulk = self.fit.fitted[: self.factors - spikes]
        return bool(np.all(np.diff(bulk) < 0.0))

    @property
    def raw_monotone_decline(self) -> bool:
        """Whether the unsmoothed ``lambda(k)`` also declines at every step."""
        return bool(np.all(np.diff(self.bias) < 0.0))

    @property
    def crosses_one(self) -> bool:
        """Above 1 at the smallest eigenfactor, below 1 at the largest."""
        return bool(self.fit.fitted[0] > 1.0 > self.fit.fitted[-1])

    def variant(self, scaling: float) -> EigenfactorVariant:
        """The variant for one published ``a``. Raises rather than guessing."""
        for candidate in self.variants:
            if candidate.scaling == scaling:
                return candidate
        available = [candidate.scaling for candidate in self.variants]
        raise EigenfactorError(
            f"eigenfactor: no variant at a = {scaling}; this adjustment ran {available}"
        )

    def render(self) -> tuple[str, ...]:
        """Lines for ``make model``. Leads with the shape gate, not the matrix."""
        lines = [
            f"eigenfactor: M = {self.trials:,} trials, EWMA half-life {self.halflife:.0f}d "
            f"over {self.observations:,} obs (K/T_eff = {self.k_over_t:.4f}), correlation space",
            f"  lambda raw [{self.bias.min():.4f}, {self.bias.max():.4f}], "
            f"fitted [{self.fit.fitted.min():.4f}, {self.fit.fitted.max():.4f}], "
            f"amplitude {self.amplitude:.4f}",
            f"  declines monotonically: fitted {self.monotone_decline}, "
            f"raw {self.raw_monotone_decline}; crosses 1: {self.crosses_one}",
            f"  {self.fit.render()}",
        ]
        lines.extend(f"  {variant.render()}" for variant in self.variants)
        return tuple(lines)


def eigenfactor_adjustment(
    matrix: np.ndarray,
    *,
    trials: int,
    halflife: float,
    observations: int,
    scalings: tuple[float, ...],
    seed: int,
    floor: float,
) -> EigenfactorAdjustment:
    """SPEC.md 5.3, run in correlation space. See the module docstring for why.

    ``matrix`` is the post-repair factor covariance in whatever units the caller
    supplied. The correction is computed on its correlation matrix and rescaled
    by the same ``sigma`` it was split from, so the output carries the input's
    units and the adjustment does not depend on them.

    Step 8 is ``D_0* = gamma^2 D_0``, ``F_0* = U_0 D_0* U_0'`` -- once per value
    of ``a``, all from the one simulation.

    **``gamma = a (lambda_P - 1) + 1`` is LINEAR in ``(lambda - 1)``.** One PDF
    rendering of MSCI's Eq. A8 garbles it into ``lambda^a (lambda - 1) + 1``;
    the published text and the open-source implementations agree against that
    rendering, and a test asserts the linear form and that the two are
    distinguishable on this data.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise EigenfactorError(
            f"eigenfactor_adjustment: expected a square matrix, got {matrix.shape}"
        )
    if not scalings:
        raise EigenfactorError(
            "eigenfactor_adjustment: no scalings. Both 1.0 and 1.4 are published and "
            "CLAUDE.md's parameter table says run both; there is no default to fall back to."
        )
    variances = np.diag(matrix)
    if np.any(variances <= 0.0):
        bad = int(np.count_nonzero(variances <= 0.0))
        raise EigenfactorError(
            f"eigenfactor_adjustment: {bad} factor(s) with non-positive variance. The "
            "correlation the correction runs on is not defined."
        )
    deviations = np.sqrt(variances)
    # The pipeline's own splitter, not a second copy of it. Stage 2 forms a
    # correlation the same way and the two must not be able to drift apart --
    # the same structural argument as ``ewma_autocovariance(lag=0)`` being
    # ``ewma_second_moment`` rather than a re-derivation of it.
    correlation = correlation_from_covariance(matrix)

    # SPEC.md 5.3 steps 2-3 SIMULATE THE ESTIMATOR THE PIPELINE ACTUALLY USES,
    # not an equal-weight surrogate for it. See the module docstring's "T" section
    # and W3-P3b: the surrogate's effective sample size is not a panel-independent
    # quantity, so there is no single T that could have been right here.
    bias = ewma_eigenvalue_bias(
        correlation,
        halflife=halflife,
        observations=observations,
        trials=trials,
        seed=seed,
        floor=floor,
    )
    eigenvalues, vectors = np.linalg.eigh(correlation)
    eigenvalues = np.where(eigenvalues <= 0.0, floor, eigenvalues)
    fit = fit_parabola(bias)

    variants = []
    for scaling in scalings:
        gamma = scaling * (fit.fitted - 1.0) + 1.0
        adjusted_correlation = vectors @ np.diag(gamma**2 * eigenvalues) @ vectors.T
        adjusted_correlation = (adjusted_correlation + adjusted_correlation.T) / 2.0
        # The diagonal is NOT renormalised. See the module docstring: the risk
        # increase this stage exists to produce IS the diagonal inflation, and
        # renormalising would delete the correction while leaving a matrix that
        # passes every check the pipeline runs.
        adjusted = adjusted_correlation * np.outer(deviations, deviations)
        variants.append(
            EigenfactorVariant(
                scaling=float(scaling),
                gamma=gamma,
                adjusted_correlation=adjusted_correlation,
                adjusted=(adjusted + adjusted.T) / 2.0,
            )
        )

    return EigenfactorAdjustment(
        correlation=correlation,
        deviations=deviations,
        eigenvalues=eigenvalues,
        eigenvectors=vectors,
        bias=bias,
        fit=fit,
        trials=int(trials),
        halflife=float(halflife),
        observations=int(observations),
        variants=tuple(variants),
    )


# ---------------------------------------------------------------------------
# W3-P3b: what IS the effective sample size of an EWMA correlation estimator?
# ---------------------------------------------------------------------------


def ewma_eigenvalue_bias(
    truth: np.ndarray,
    *,
    halflife: float,
    observations: int,
    trials: int,
    seed: int,
    floor: float,
    normalise: bool = True,
) -> np.ndarray:
    """``lambda(k)`` for the EWMA estimator itself, measured rather than assumed.

    :func:`simulated_eigenvalue_bias` simulates an **equally weighted** sample of
    ``T`` rows, which is SPEC.md 5.3's algorithm and is what the adjustment runs.
    This simulates what the pipeline actually estimates: ``observations`` iid rows
    from ``truth``, run through :func:`~mafrm.risk.covariance.ewma_second_moment`
    at ``halflife`` and then normalised to a correlation -- the production path,
    not a model of it.

    The two are compared by :func:`equivalent_sample_size`, which asks which
    equal-weight ``T`` reproduces this curve. That question is the whole of
    W3-P3b: ``T_eff = 2*tau/ln 2`` is derived for an equally weighted window's
    effective size, and whether it transfers to the eigenvalue bias of a
    *normalised* EWMA estimator is empirical. Two things could break it -- the
    exponential weighting, and the unit-diagonal normalisation, which is
    non-linear, applied after the weighting, and has no counterpart in the
    formula.

    ``normalise=False`` is the control that separates the two: it stops at the
    second moment and skips the correlation step.
    """
    if truth.ndim != 2 or truth.shape[0] != truth.shape[1]:
        raise EigenfactorError(f"ewma_eigenvalue_bias: expected a square matrix, got {truth.shape}")
    size = truth.shape[0]
    if observations <= size:
        raise EigenfactorError(
            f"ewma_eigenvalue_bias: {observations} observations for {size} factors is singular "
            "by construction"
        )
    reference = correlation_from_covariance(truth) if normalise else truth
    factorisation = np.linalg.cholesky(truth)
    generator = np.random.default_rng(seed)
    total = np.zeros(size)
    for _ in range(trials):
        sample = generator.standard_normal((observations, size)) @ factorisation.T
        moment = ewma_second_moment(sample, halflife=halflife)
        estimate = correlation_from_covariance(moment) if normalise else moment
        estimated, directions = np.linalg.eigh(estimate)
        estimated = np.where(estimated <= 0.0, floor, estimated)
        total += np.einsum("ij,jk,ki->i", directions.T, reference, directions) / estimated
    bias: np.ndarray = np.sqrt(total / trials)
    return bias


@dataclass(frozen=True)
class EquivalentSampleSize:
    """What equal-weight ``T`` reproduces an EWMA estimator's eigenvalue bias.

    The answer to W3-P3b's question, with everything needed to judge it: the
    measured EWMA curve, the best-matching equal-weight curve, the grid that was
    searched, and the two formula comparands it is being tested against.
    """

    halflife: float
    observations: int
    trials: int
    #: The measured EWMA curve.
    ewma_bias: np.ndarray
    #: The best-matching equal-weight curve.
    matched_bias: np.ndarray
    #: The ``T`` that produced it.
    equivalent: int
    #: RMS distance between the two **mean-centred** curves at ``equivalent``.
    residual: float
    #: ``2*tau/ln 2``, the asymptotic formula.
    asymptotic: float
    #: Kish's ``(sum w)^2 / sum w^2`` for the actual window.
    realised: float
    grid: tuple[int, ...]
    distances: tuple[float, ...]

    @property
    def relative_to_asymptotic(self) -> float:
        """``equivalent / (2*tau/ln 2) - 1``. The quantity the falsifier is on."""
        return self.equivalent / self.asymptotic - 1.0

    @property
    def relative_to_realised(self) -> float:
        """The same against the window's realised Kish size, the honest comparand."""
        return self.equivalent / self.realised - 1.0

    def render(self) -> str:
        return (
            f"equivalent T = {self.equivalent:,} at half-life {self.halflife:.0f} over "
            f"{self.observations:,} obs (rms {self.residual:.2e}); "
            f"asymptotic 2*tau/ln2 = {self.asymptotic:.1f} "
            f"({self.relative_to_asymptotic:+.1%}), realised Kish = {self.realised:.1f} "
            f"({self.relative_to_realised:+.1%})"
        )


def equivalent_sample_size(
    truth: np.ndarray,
    *,
    halflife: float,
    observations: int,
    grid: tuple[int, ...],
    trials: int,
    seed: int,
    floor: float,
    normalise: bool = True,
) -> EquivalentSampleSize:
    """Grid-search the equal-weight ``T`` that reproduces the EWMA bias curve.

    The criterion is RMS distance over the **whole curve, each curve centred on
    its own mean**, and both halves of that matter.

    *Whole curve* rather than amplitude alone: matching two numbers would be
    satisfiable by a wrong ``T`` with a compensating shape error, and the shape is
    what the adjustment acts on.

    *Centred* because the two estimators differ by a normalisation convention that
    is not part of the question. SPEC.md 5.3's equal-weight simulation forms
    ``f f' / (T - 1)`` while the EWMA estimator divides by ``sum(w) = 1`` over
    ``T`` weighted terms, so the equal-weight curve sits about ``1/(2T)`` below the
    EWMA one **uniformly, at every ``T``**. Left in, that offset biases the search
    upward -- measurably: without centring, the instrument's own recovery control
    (a half-life so long the weights are flat, where the answer must be the window
    length) returns one grid step too high. Centring removes it, and the control
    then recovers the known answer exactly. A level offset and an effective sample
    size are different quantities and this criterion is about the second.

    Both simulations are driven from ``seed``, so the comparison is not
    contaminated by one of them having drawn a luckier sample than the other.
    """
    measured = ewma_eigenvalue_bias(
        truth,
        halflife=halflife,
        observations=observations,
        trials=trials,
        seed=seed,
        floor=floor,
        normalise=normalise,
    )
    reference = correlation_from_covariance(truth) if normalise else truth

    centred = measured - measured.mean()
    distances: list[float] = []
    curves: list[np.ndarray] = []
    for candidate in grid:
        curve, _, _ = simulated_eigenvalue_bias(
            reference, trials=trials, sample_size=candidate, seed=seed, floor=floor
        )
        curves.append(curve)
        residual = (curve - curve.mean()) - centred
        distances.append(float(np.sqrt(np.mean(np.square(residual)))))
    best = int(np.argmin(distances))

    return EquivalentSampleSize(
        halflife=float(halflife),
        observations=int(observations),
        trials=int(trials),
        ewma_bias=measured,
        matched_bias=curves[best],
        equivalent=int(grid[best]),
        residual=distances[best],
        asymptotic=2.0 * halflife / float(np.log(2.0)),
        realised=realised_effective_sample_size(ewma_weights(observations, halflife)),
        grid=tuple(grid),
        distances=tuple(distances),
    )
