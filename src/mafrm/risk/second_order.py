"""The disjointness demonstration. SPEC.md 5.3.2, 5.4.2 and 6.4, W4-P3.

WHAT IS AT STAKE
----------------

Three things in this pipeline scale the level of risk: SPEC.md 5.3's eigenfactor
adjustment, SPEC.md 5.4's volatility regime adjustment, and SPEC.md 6.4's Shepard
closed form. Each passes its own sanity check. Applied together they double-count
unless what each corrects is shown not to overlap -- CLAUDE.md failure mode 7 in
its purest form, and the arithmetic a reader will do (the eigenfactor stage's
0.84% plus Shepard's 5.14%) is valid only if the overlap is zero.

Shepard's ``[1 - K/T]^-2`` is derived for the **whole** covariance estimated on
**one** window -- a Wishart at ``T``. SPEC.md 5.1's estimator is
``F = D rho D`` with ``D`` from an EWMA at the volatility half-life and ``rho``
from a separate EWMA at the correlation half-life, and after SPEC.md 5.3.2 the
eigenfactor stage corrects the sampling error of ``rho`` alone. So the whole
covariance closed form at the volatility window is not this estimator's bias, the
``rho`` leg is already spoken for, and the ``sigma`` leg has no closed form under a
split window. Rather than argue, this module **measures**: it feeds Gaussian rows
from a known ``F`` through the pipeline's own estimators
(:func:`~mafrm.risk.covariance.ewma_second_moment`,
:func:`~mafrm.risk.covariance.correlation_from_covariance`) and decomposes the
minimum-variance portfolio's second-order bias into a ``rho``-only leg, a
``sigma``-only leg, and their joint. The disjointness statistic is the
interaction: joint excess minus the sum of the two legs' excesses. To second
order it is zero, because for Gaussian data the sample correlation matrix is
independent of the sample variances and the bias is a quadratic form in the
estimation error; what the simulation adds is the size of the remainder.

THE FIVE CONDITIONS
-------------------

(A) single-window joint at the volatility half-life -- Shepard's own case, and
    the check that ``T_eff = 2 tau / ln 2`` transfers for a second-moment
    estimator (W3-P3b measured 1.007x);
(B) split-window joint -- SPEC.md 5.1's estimator exactly;
(C) ``rho`` estimated at the correlation half-life, ``sigma`` exact;
(D) ``sigma`` estimated at the volatility half-life, ``rho`` exact;
(E) the per-factor bias ``mean_k F_kk / F_hat_kk`` under (B) -- what SPEC.md
    5.4's cross-sectional ``B_t^F`` can see of the estimation error, which is the
    estimation-error side of the Shepard-VRA overlap.

The statistic in (A)-(D) is ``E[w' F w / w' F_hat w]`` for the minimum-variance
``w`` of ``F_hat``, which is the population ``B^2`` of SPEC.md 6.2's family 4:
the next period's return is independent of the estimation sample, so the
expected squared standardised return is exactly this ratio.

THE ANALYTIC CONTROL
--------------------

For ``rho = I`` the ``sigma``-only bias has a closed form to second order,
:func:`volatility_leg_closed_form`: ``1 + (2/T_eff)(2 - sum_k w_k^2)`` at the
true minimum-variance weights ``w``. Its ``K = 1`` limit is Shepard's own
``1 + 2/T``; for equal variances it is ``1 + (2/T)(2 - 1/K)``. Derived before the
run and checked on synthetic iid data (``experiments.md`` rows 175-186), it is
the known answer :func:`simulate` is tested against, in the same role row 110's
flat-weight control played for row 109.

Nothing here knows what a factor is. Matrices in, numbers out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from mafrm.numerics import effective_sample_size, ewma_weights, realised_effective_sample_size
from mafrm.risk.covariance import correlation_from_covariance, ewma_second_moment
from mafrm.risk.shepard import SecondOrderRisk, jensen_term, second_order_risk
from mafrm.risk.validation import minimum_variance_weights

__all__ = [
    "ConditionResult",
    "Decomposition",
    "SecondOrderError",
    "inverse_second_moment_expectation",
    "simulate",
    "volatility_leg_closed_form",
]


class SecondOrderError(ValueError):
    """The simulation was handed something it cannot run on."""


@dataclass(frozen=True)
class ConditionResult:
    """One estimator condition's ``B^2``, with the Monte Carlo error beside it."""

    name: str
    #: Per-trial ratios, kept so interactions can be computed trial by trial.
    ratios: np.ndarray

    @property
    def trials(self) -> int:
        return int(self.ratios.size)

    @property
    def mean_ratio(self) -> float:
        """``E[realised variance / forecast variance]`` -- the population ``B^2``."""
        return float(self.ratios.mean())

    @property
    def excess(self) -> float:
        """``B^2 - 1``, in VARIANCE. The unit Shepard's Eq. 13/32 are written in."""
        return self.mean_ratio - 1.0

    @property
    def bias(self) -> float:
        """``B = sqrt(B^2)``, in VOLATILITY. The unit SPEC.md 6.1's statistic is in."""
        return float(np.sqrt(self.mean_ratio))

    @property
    def standard_error(self) -> float:
        """Monte Carlo standard error of :attr:`mean_ratio`."""
        if self.trials < 2:
            return float("nan")
        return float(self.ratios.std(ddof=1) / np.sqrt(self.trials))


@dataclass(frozen=True)
class Decomposition:
    """Everything one run of :func:`simulate` produced."""

    single_window: ConditionResult
    joint: ConditionResult
    correlation_only: ConditionResult
    volatility_only: ConditionResult
    per_factor: ConditionResult
    #: Shepard Eq. 32 at the VOLATILITY window -- the whole-covariance closed form.
    shepard_volatility_window: SecondOrderRisk
    #: Shepard Eq. 32 at the CORRELATION window -- the eigenfactor stage's comparand.
    shepard_correlation_window: SecondOrderRisk
    #: Shepard at ``p = 1`` at the volatility window -- the Jensen term (E) tests.
    jensen: SecondOrderRisk
    #: The SAME expectation computed exactly for the window's actual EWMA weights,
    #: :func:`inverse_second_moment_expectation`. What (E) is converging to.
    jensen_exact: float
    #: :func:`volatility_leg_closed_form` at the truth's own min-var weights.
    volatility_leg_closed_form: float
    parameters: int
    observations: int
    trials: int
    volatility_halflife: float
    correlation_halflife: float
    #: Kish sizes of the two EWMA windows on ``observations`` rows, beside the
    #: asymptotic figures the closed forms use.
    realised_volatility_t_eff: float
    realised_correlation_t_eff: float
    seed: int

    @property
    def interaction(self) -> float:
        """``(B_joint^2 - 1) - (B_rho^2 - 1) - (B_sigma^2 - 1)``. Zero means disjoint."""
        return float(self._interaction_draws().mean())

    @property
    def interaction_standard_error(self) -> float:
        draws = self._interaction_draws()
        return float(draws.std(ddof=1) / np.sqrt(draws.size))

    @property
    def interaction_share(self) -> float:
        """The interaction as a share of the joint excess -- the registered statistic."""
        return self.interaction / self.joint.excess

    def _interaction_draws(self) -> np.ndarray:
        # Trial by trial, on the SAME draws, so the correlation between the three
        # conditions is inside the standard error rather than assumed away.
        return np.asarray(
            self.joint.ratios - self.correlation_only.ratios - self.volatility_only.ratios + 1.0
        )


def volatility_leg_closed_form(weights: np.ndarray, *, effective_observations: float) -> float:
    """``1 + (2/T_eff)(2 - sum w^2)``: the ``sigma``-only second-order bias when ``rho = I``.

    Second-order expansion of ``E[w_hat' F w_hat / w_hat' F_hat w_hat]`` for
    ``F_hat = diag(sigma_hat^2)`` with ``sigma_hat^2 = sigma^2 (1 + eps)``,
    ``Var eps = 2/T_eff`` (Gaussian; the second-moment case where W3-P3b found
    ``2 tau / ln 2`` transfers). ``weights`` are the TRUE minimum-variance weights,
    whose Herfindahl ``sum w^2`` replaces ``1/K`` when the variances are unequal;
    ``K = 1`` gives ``1 + 2/T``, Shepard's own formula at ``p = 1``.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise SecondOrderError(
            f"volatility_leg_closed_form: expected a 1-D weight vector, got {weights.shape}"
        )
    if not effective_observations > 0.0:
        raise SecondOrderError("volatility_leg_closed_form: T_eff must be positive")
    herfindahl = float(np.sum(np.square(weights)))
    return 1.0 + (2.0 / effective_observations) * (2.0 - herfindahl)


def inverse_second_moment_expectation(weights: np.ndarray) -> float:
    """``E[1 / sum_i w_i z_i^2]`` for iid standard normal ``z``, exact.

    ``E[1/X] = int_0^inf E[e^{-sX}] ds = int_0^inf prod_i (1 + 2 s w_i)^{-1/2} ds``,
    the Laplace-transform identity for a positive variable applied to a weighted
    chi-square. For equal weights over ``T`` observations it is ``T/(T-2)``, and
    for EWMA weights it is what the Jensen term ``1 + 2/T_eff`` approximates to
    first order. This is the exact expectation of a per-axis ``sigma^2 /
    sigma_hat^2`` under estimation error, i.e. what condition (E) of
    :func:`simulate` converges to, and therefore the exact size of the
    estimation-error term a per-factor cross-sectional bias statistic can see.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size == 0 or np.any(weights < 0.0):
        raise SecondOrderError("inverse_second_moment_expectation: non-negative 1-D weights")
    if weights.sum() <= 0.0:
        raise SecondOrderError("inverse_second_moment_expectation: weights sum to zero")

    def transform(s: float) -> float:
        return float(np.exp(-0.5 * np.sum(np.log1p(2.0 * s * weights))))

    value, _ = quad(transform, 0.0, np.inf, limit=500)
    return float(value)


def _ratio(truth: np.ndarray, estimate: np.ndarray) -> float:
    """``w' F w / w' F_hat w`` at the minimum-variance ``w`` of ``F_hat``."""
    weights = minimum_variance_weights(estimate)
    return float((weights @ truth @ weights) / (weights @ estimate @ weights))


def simulate(
    truth: np.ndarray,
    *,
    volatility_halflife: float,
    correlation_halflife: float,
    observations: int,
    trials: int,
    seed: int,
) -> Decomposition:
    """Run the five conditions on ``trials`` Gaussian samples from ``truth``.

    ``truth`` is the ``K x K`` covariance the rows are drawn from and must be
    positive definite -- the Cholesky factor is the draw, and a semi-definite
    truth would make the minimum-variance problem ill-posed. ``observations``
    rows per trial, so the EWMA windows have the length they have on the real
    panel; their realised Kish sizes are reported beside the asymptotic ones.

    The generator is ``np.random.default_rng(seed)``, passed in, never global.
    """
    truth = np.asarray(truth, dtype=float)
    if truth.ndim != 2 or truth.shape[0] != truth.shape[1]:
        raise SecondOrderError(f"simulate: truth must be square, got {truth.shape}")
    size = truth.shape[0]
    if observations <= size:
        raise SecondOrderError(
            f"simulate: {observations} observations for {size} parameters is singular by "
            "construction"
        )
    if trials < 2:
        raise SecondOrderError(f"simulate: at least 2 trials are needed, got {trials}")
    if not np.allclose(truth, truth.T):
        raise SecondOrderError("simulate: truth is not symmetric")
    try:
        factor = np.linalg.cholesky(truth)
    except np.linalg.LinAlgError as exc:
        raise SecondOrderError("simulate: truth is not positive definite") from exc

    deviation = np.sqrt(np.diag(truth))
    correlation = correlation_from_covariance(truth)
    generator = np.random.default_rng(seed)

    single = np.empty(trials)
    joint = np.empty(trials)
    correlation_only = np.empty(trials)
    volatility_only = np.empty(trials)
    per_factor = np.empty(trials)
    true_variances = np.diag(truth)
    for trial in range(trials):
        sample = generator.standard_normal((observations, size)) @ factor.T
        volatility_moment = ewma_second_moment(sample, halflife=volatility_halflife)
        correlation_moment = ewma_second_moment(sample, halflife=correlation_halflife)
        estimated_deviation = np.sqrt(np.diag(volatility_moment))
        estimated_correlation = correlation_from_covariance(correlation_moment)
        # (A) one window, both legs: the Wishart-like case Shepard derives for.
        single[trial] = _ratio(truth, volatility_moment)
        # (B) SPEC.md 5.1's estimator: rho at its window, sigma at its own.
        split = estimated_correlation * np.outer(estimated_deviation, estimated_deviation)
        joint[trial] = _ratio(truth, split)
        # (C) only rho carries error.
        correlation_only[trial] = _ratio(
            truth, estimated_correlation * np.outer(deviation, deviation)
        )
        # (D) only sigma carries error.
        volatility_only[trial] = _ratio(
            truth, correlation * np.outer(estimated_deviation, estimated_deviation)
        )
        # (E) the fixed factor axes, which is all a per-factor statistic sees.
        per_factor[trial] = float(np.mean(true_variances / np.square(estimated_deviation)))

    return Decomposition(
        single_window=ConditionResult("single_window", single),
        joint=ConditionResult("split_window_joint", joint),
        correlation_only=ConditionResult("correlation_only", correlation_only),
        volatility_only=ConditionResult("volatility_only", volatility_only),
        per_factor=ConditionResult("per_factor", per_factor),
        shepard_volatility_window=second_order_risk(size, halflife=volatility_halflife),
        shepard_correlation_window=second_order_risk(size, halflife=correlation_halflife),
        jensen=jensen_term(halflife=volatility_halflife),
        jensen_exact=inverse_second_moment_expectation(
            ewma_weights(observations, volatility_halflife)
        ),
        volatility_leg_closed_form=volatility_leg_closed_form(
            minimum_variance_weights(truth),
            effective_observations=effective_sample_size(volatility_halflife),
        ),
        parameters=size,
        observations=int(observations),
        trials=int(trials),
        volatility_halflife=float(volatility_halflife),
        correlation_halflife=float(correlation_halflife),
        realised_volatility_t_eff=realised_effective_sample_size(
            ewma_weights(observations, volatility_halflife)
        ),
        realised_correlation_t_eff=realised_effective_sample_size(
            ewma_weights(observations, correlation_halflife)
        ),
        seed=int(seed),
    )
