"""The factor covariance pipeline. SPEC.md 5, built against SPEC.md 15.2.

SPEC.md 5 fixes the order and it is load-bearing::

    EWMA -> Newey-West -> PSD repair -> eigenfactor -> volatility regime

The regime adjustment is last because it is a pure level scaling that has to be
calibrated against the final matrix; the PSD repair sits where it does because
Newey-West is not PSD-guaranteed and everything after it assumes a valid
covariance. The order is declared in ``covariance.stages`` in
``config/model.yaml`` rather than implied by the order of function calls here,
so that :func:`missing_stages` can compare what is declared against what exists
and ``make model`` can fail on the difference. **As of W3-P4 all five stages
exist** and that comparison is empty; ``make model`` stays red on a second audit
(:data:`mafrm.build._UNBUILT`) covering SPEC.md 5.5, Model B and the hybrid, so
that finishing the last covariance stage did not turn the target green over three
missing components.

WHAT THIS MODULE MAY KNOW
-------------------------

A ``T x K`` frame of factor returns and a :class:`~mafrm.risk.config.RiskConfig`.
That is the whole of it. CLAUDE.md invariant 10 and SPEC.md 15.2: this package
must not know what asset class it is looking at, so that the week-7
cross-sectional module is a new file under ``factors/`` and **nothing here
changes**. There is no branch in this file on the number of columns, on their
names, on a calendar, or on anything reachable from ``RiskConfig``.

STAGE 1 -- SPEC.md 5.1
----------------------

Volatilities and correlations are estimated **separately, at different
half-lives**, and recombined::

    F0_ij = rho_ij * sigma_i * sigma_j

with ``sigma`` from an EWMA at ``volatility_halflife`` (84d short, 252d long)
and ``rho`` from an EWMA at ``correlation_halflife`` (504d at *both* horizons).
The separation is the point of the USE4 pipeline: volatilities mean-revert fast
and need responsiveness, correlations are noisier and need stability.

The recombination is PSD by construction -- ``rho`` is a correlation matrix
derived from a PSD second-moment matrix, and ``D rho D`` is PSD for any
non-negative diagonal ``D`` -- so the assertion after this stage should never
fire. It runs anyway, because the stages that follow have no such guarantee and
a pipeline whose assertions are only where failure is expected cannot tell you
that an earlier stage was fine.

MOMENTS ARE TAKEN ABOUT ZERO
----------------------------

No trailing mean is subtracted, at either half-life. This is a coherence
requirement against SPEC.md 6.1, not a convention preference, and the full
argument is at ``covariance.mean_convention`` in ``config/model.yaml`` and in
SPEC.md 5.1.1. In one line: the bias statistic is ``b_nt = R_nt / sigma_nt``
with a **raw** return in the numerator, so the denominator must forecast
variance about zero or the two sides measure different things and ``B`` -- the
one statistic this project exists to decompose -- is biased by the mismatch
alone.

Two consequences follow and both are asserted in ``tests/test_covariance.py``,
because they are how the decision stays visible:

1. As the half-life grows, this estimator converges to ``X'X / T``, the
   **uncentred** second-moment matrix -- *not* to ``numpy.cov``, which
   subtracts a mean and divides by ``T-1``.
2. There is no ``1 - w'w`` reliability correction in the denominator. That
   correction exists to restore the degree of freedom a *demeaned* estimator
   spends estimating the mean. A zero-mean estimator spends none, so the
   denominator is just ``sum(w) = 1``, and adding the correction anyway would
   inflate every variance by roughly ``1/(1 - w'w)``.

SCALE INVARIANCE -- A CONSTRAINT ON THE STAGES NOT YET WRITTEN
--------------------------------------------------------------

Ruled in W3-P1b, recorded here because this is the file the stage will be added
to. SPEC.md 5.3.1.

**The eigenfactor adjustment must be invariant to the units each factor is
expressed in.** A risk correction whose output changes when a factor is quoted in
decimals rather than in basis points is not a risk correction: it is a correction
to an arbitrary choice of numeraire, and nothing about estimation error depends
on that choice. The acceptance criterion is mechanical -- rescale any factor by
any positive constant, run the adjustment, undo the rescaling, and the resulting
matrix must be unchanged to numerical tolerance.

The step that does not satisfy this for free is the diagonalisation. Eigenvectors
of a covariance matrix are not equivariant under a diagonal rescaling of the
underlying variables, so *which direction is the smallest eigenfactor* -- the
direction the adjustment corrects hardest -- can be decided by the units before
it is decided by the data. On this project's own panel that is not hypothetical:
`experiments.md` row 96 measures a covariance condition number of 9.96e6 against
a correlation condition number of 3.27, with essentially all of the gap coming
from a 7.3e6 variance ratio across a factor set whose columns are deliberately in
mixed units.

**USE4 is not wrong here, and nothing in this file should read as though it
were.** Every factor in USE4 is already in return units, so a diagonal rescaling
across factors is not something that can happen there and the question never
arises; diagonalising ``F`` is exactly right in that setting. What would be wrong
is following the published step literally into a case it does not cover.

The choice of resolution, and the invariance test that settles it, belong to the
session that writes the stage. Nothing is implemented here.

K/T_eff
-------

Every build emits ``K / T_eff`` and Shepard's ``[1 - K/T_eff]^-2``. It is free
to compute and it is SPEC.md 15.2's headline result -- the understatement the
optimizer suffers, swept across a 10x range of ``K/T`` by building a small model
and a large one on the same code. Emitting it from the very first stage means
week 7's table is assembled from a quantity that has been reported all along,
rather than computed specially at the end to make a point.

Both denominators are reported. ``T_eff = 2*tau/ln 2`` is the asymptotic figure
SPEC.md 15.2's table quotes; the realised Kish figure is what the actual window
delivers, and it is smaller whenever the window is short relative to the
half-life. **A future burn-in rule belongs on this ratio and not on a day
count**: the bound then depends only on ``K`` and the half-lives, which makes it
asset-class agnostic by construction, where a minimum number of days would have
to be re-chosen for every panel and would say nothing about how many parameters
were being estimated. There is no burn-in parameter in this session because the
SPEC.md 15.2 signature takes its window from the caller and returns one matrix,
so there is genuinely no such decision to make yet.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mafrm.numerics import effective_sample_size, ewma_weights, realised_effective_sample_size
from mafrm.risk.checks import PsdCheck, Spectrum, assert_psd, measure_spectrum
from mafrm.risk.config import RiskConfig
from mafrm.risk.shepard import SecondOrderRisk, Unit, second_order_risk

if TYPE_CHECKING:  # pragma: no cover - the runtime import is local, see below
    from mafrm.risk.eigenfactor import EigenfactorAdjustment
    from mafrm.risk.regime import RegimeMultiplier

__all__ = [
    "BartlettFallback",
    "CovarianceBuild",
    "CovarianceError",
    "RepairReport",
    "StageInputs",
    "StageOutput",
    "StageResult",
    "assert_zero_mean_consistency",
    "bartlett_newey_west",
    "bartlett_weight",
    "build_covariance",
    "correlation_from_covariance",
    "declared_stages",
    "ewma_autocovariance",
    "ewma_second_moment",
    "implemented_stages",
    "lagged_pair_weights",
    "measured_not_asserted_stages",
    "missing_stages",
    "newey_west_covariance",
    "psd_repair",
    "run_pipeline",
    "separated_ewma_covariance",
]


class CovarianceError(ValueError):
    """The covariance pipeline was handed something it cannot estimate from."""


@dataclass(frozen=True)
class StageOutput:
    """What one stage returns: a matrix, and any diagnostic only it can produce.

    The repair field exists because SPEC.md 5.2 asks for a number that is
    destroyed by the operation that produces it -- how negative the spectrum was
    *before* the floor -- so the stage has to hand it out rather than let a
    caller re-derive it from an already-repaired matrix.
    """

    matrix: np.ndarray
    repair: RepairReport | None = None
    #: Set only by ``newey_west``: which variances reverted to the uncorrected
    #: EWMA estimate under experiments.md row 100's registered remedy. Same
    #: reason as ``repair`` -- the substitution destroys the number it replaced.
    bartlett: BartlettFallback | None = None
    #: Set only by ``eigenfactor``: the simulated curve, the fit and every
    #: published variant. Like ``repair``, it is a quantity the stage destroys
    #: on its way to a matrix, so it has to be handed out rather than re-derived.
    eigenfactor: EigenfactorAdjustment | None = None
    #: Set only by ``volatility_regime``: SPEC.md 5.4's ``lambda_F^2`` and the
    #: bias series it was computed from. Same reason as the two above -- the
    #: stage collapses a whole forecast history into one scalar multiplication,
    #: and the history cannot be recovered from the scaled matrix.
    regime: RegimeMultiplier | None = None


@dataclass(frozen=True)
class StageInputs:
    """Everything the stages read besides the running matrix and the config.

    ``returns`` is the ``T x K`` window every stage from SPEC.md 5.1 to 5.3
    estimates from. ``regime_bias`` is SPEC.md 5.4's ``B_t^F`` series and is
    **supplied by the caller**, because it is a history of forecasts rather than
    a window of returns and building it means running this pipeline once per
    date. SPEC.md 5.1.2 leaves the burn-in that governs such a history with the
    caller; :func:`mafrm.risk.regime.forecast_history` is the builder.

    It is ``None`` for every caller that does not need a finished forecast, and
    the regime stage **refuses to run** rather than defaulting to one. A default
    here would be a silently un-adjusted matrix that is PSD, symmetric,
    identically conditioned and simply too small -- the failure
    :attr:`CovarianceBuild.forecast` exists to make impossible.
    """

    returns: np.ndarray
    regime_bias: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Stage 1 -- SPEC.md 5.1
# ---------------------------------------------------------------------------


def ewma_second_moment(returns: np.ndarray, *, halflife: float) -> np.ndarray:
    """Exponentially weighted second moment about zero. SPEC.md 5.1.

    ``returns`` is ``T x K``, oldest row first. Returns the ``K x K`` matrix

    ``M_ij = sum_t w_t * f_it * f_jt``,   ``w`` normalised to sum to one,

    with ``w`` decaying at ``halflife`` toward the oldest row. **No mean is
    subtracted** -- see the module docstring and SPEC.md 5.1.1.

    Symmetric and positive semi-definite by construction: it is
    ``(sqrt(w) * F)' (sqrt(w) * F)``, a Gram matrix, and the implementation
    forms it that way rather than as a weighted product so that symmetry holds
    to the last bit rather than to roundoff.
    """
    if returns.ndim != 2:
        raise CovarianceError(
            f"ewma_second_moment: expected a 2-D T x K array, got {returns.shape}"
        )
    periods, factors = returns.shape
    if periods < 1 or factors < 1:
        raise CovarianceError(f"ewma_second_moment: empty input, shape {returns.shape}")
    if not np.all(np.isfinite(returns)):
        count = int(np.count_nonzero(~np.isfinite(returns)))
        raise CovarianceError(
            f"ewma_second_moment: {count} non-finite observation(s). The caller decides how a "
            "gap is filled or dropped -- this module will not guess, because the right answer "
            "depends on what the column is and it is not allowed to know."
        )
    weights = ewma_weights(periods, halflife)
    scaled = returns * np.sqrt(weights)[:, None]
    moment: np.ndarray = scaled.T @ scaled
    symmetric: np.ndarray = (moment + moment.T) / 2.0
    return symmetric


def correlation_from_covariance(matrix: np.ndarray) -> np.ndarray:
    """``rho_ij = F_ij / sqrt(F_ii F_jj)``, with an exact unit diagonal.

    The diagonal is written rather than divided into place: ``F_ii / F_ii`` is
    1.0 only up to roundoff, and a correlation matrix whose diagonal is
    ``1 - 2e-16`` recombines into a covariance whose variances are subtly wrong.
    """
    variances = np.diag(matrix)
    if np.any(variances <= 0.0):
        bad = int(np.count_nonzero(variances <= 0.0))
        raise CovarianceError(
            f"correlation_from_covariance: {bad} factor(s) with non-positive estimated variance. "
            "A constant or all-zero column cannot be correlated with anything; drop it upstream."
        )
    deviation = np.sqrt(variances)
    correlation: np.ndarray = matrix / np.outer(deviation, deviation)
    np.fill_diagonal(correlation, 1.0)
    symmetric: np.ndarray = (correlation + correlation.T) / 2.0
    return symmetric


def separated_ewma_covariance(
    returns: np.ndarray,
    *,
    volatility_halflife: float,
    correlation_halflife: float,
) -> np.ndarray:
    """``F0_ij = rho_ij * sigma_i * sigma_j``. SPEC.md 5.1, USE4 Table 4.1.

    ``sigma`` comes from an EWMA at ``volatility_halflife``; ``rho`` from a
    *separate* EWMA at ``correlation_halflife``. When the two half-lives are
    equal this reduces exactly to :func:`ewma_second_moment` at that half-life,
    which ``tests/test_covariance.py`` pins.
    """
    volatility_moment = ewma_second_moment(returns, halflife=volatility_halflife)
    correlation_moment = ewma_second_moment(returns, halflife=correlation_halflife)
    variances = np.diag(volatility_moment)
    if np.any(variances <= 0.0):
        bad = int(np.count_nonzero(variances <= 0.0))
        raise CovarianceError(
            f"separated_ewma_covariance: {bad} factor(s) with non-positive estimated variance"
        )
    deviation = np.sqrt(variances)
    correlation = correlation_from_covariance(correlation_moment)
    covariance: np.ndarray = correlation * np.outer(deviation, deviation)
    symmetric: np.ndarray = (covariance + covariance.T) / 2.0
    return symmetric


def _stage_ewma(matrix: np.ndarray, inputs: StageInputs, config: RiskConfig) -> StageOutput:
    """SPEC.md 5.1. Ignores ``matrix``: this is the stage that creates it."""
    del matrix
    return StageOutput(
        matrix=separated_ewma_covariance(
            inputs.returns,
            volatility_halflife=float(config.volatility_halflife),
            correlation_halflife=float(config.correlation_halflife),
        )
    )


# ---------------------------------------------------------------------------
# Stage 2 -- SPEC.md 5.2, Newey-West
# ---------------------------------------------------------------------------
#
# The two functions immediately below are the Newey-West CONVENTIONS, extracted
# so that there is one of each rather than one per consumer. SPEC.md 5.5's
# time-series estimate runs the same Bartlett sum over the same lag pairing, but
# needs only the DIAGONAL of it -- an ``N x N`` matrix formed to be thrown away
# is ``O(N^2 T)`` where the diagonal is ``O(N T)``, and at the ``N`` SPEC.md 15.2
# points this code at that is the difference between a module that transfers and
# one that does not. So there are two evaluation paths, and exactly one
# definition of what they evaluate.
#
# ``tests/test_specific_risk.py`` asserts the diagonal path reproduces
# ``np.diag`` of the matrix path to a derived rounding bound, for the same reason
# :func:`assert_zero_mean_consistency` exists: a second evaluation of an
# estimator that is never checked against the first is a second estimator.


def bartlett_weight(lag: int, lags: int) -> float:
    """``1 - d/(L+1)``, the Bartlett kernel. Newey & West (1987).

    One line, and it has its own function because it has two consumers and an
    off-by-one here (``L`` rather than ``L + 1`` in the denominator) is a change
    to a published estimator that no assertion downstream would catch: both forms
    are positive, declining and reach a plausible-looking place.
    """
    if lags < 0:
        raise CovarianceError(f"bartlett_weight: lags must be non-negative, got {lags}")
    return 1.0 - lag / (lags + 1.0)


def lagged_pair_weights(weights: np.ndarray, lag: int, *, periods: int) -> np.ndarray:
    """Normalised weights on the ``T - lag`` pairs at ``lag``. The W3-P2 ruling.

    A pair ``(f_t, f_{t-lag})`` takes **the weight of its more recent
    observation** and the survivors are renormalised to sum to one. The full
    argument, and the two alternatives deliberately not run, are at
    :func:`_weighted_lagged_moment`; this function is where the convention
    actually lives, so that a second consumer inherits it rather than re-deciding
    it.
    """
    if lag < 0:
        raise CovarianceError(f"lagged_pair_weights: lag must be non-negative, got {lag}")
    if lag >= periods:
        raise CovarianceError(
            f"lagged_pair_weights: lag {lag} needs at least {lag + 1} observations, got {periods}"
        )
    pair_weights = weights[lag:]
    total = float(pair_weights.sum())
    if total <= 0.0:
        raise CovarianceError(
            f"lagged_pair_weights: the {periods - lag} pairs at lag {lag} carry no weight"
        )
    normalised: np.ndarray = pair_weights / total
    return normalised


def _weighted_lagged_moment(returns: np.ndarray, weights: np.ndarray, lag: int) -> np.ndarray:
    """``sum_t w_t * f_t * f_{t-lag}'`` over the ``T - lag`` available pairs.

    **NO MEAN IS SUBTRACTED.** This is the same convention as SPEC.md 5.1.1's
    zero-mean second moment, and it has to be: see
    :func:`assert_zero_mean_consistency`.

    Not symmetric for ``lag > 0`` -- ``C_delta[i, j]`` leads factor ``i`` and lags
    factor ``j``, and cross-serial-correlation is directional. The Bartlett sum
    symmetrises it with ``C + C.T``, which is where symmetry is restored.

    WEIGHT ALIGNMENT (a choice SPEC.md 5.2 does not make; W3-P2)
        A pair ``(f_t, f_{t-lag})`` takes **the weight of its more recent
        observation**, ``w_t``, and the surviving weights are renormalised to sum
        to one. The pair carries information dated ``t``, and an estimator whose
        purpose is to weight recent information more heavily should date it by
        when it was complete. Renormalisation makes ``C_delta`` a proper weighted
        average rather than one biased toward zero by the mass of the ``lag``
        oldest observations, whose partners do not exist. On a long window the
        renormalisation is invisible -- the dropped weights are the smallest
        ones -- and it is the short-window case it is there for.

        The alternatives (date the pair by ``t - lag``, or by the geometric mean
        of the two weights) were **not run as a comparison**; this is a decision
        on a stated principle, logged in ``experiments.md`` under the W3-P2
        parameter choices, not a sweep.
    """
    periods = returns.shape[0]
    pair_weights = lagged_pair_weights(weights, lag, periods=periods)
    recent = returns[lag:] * pair_weights[:, None]
    lagged = returns[: periods - lag]
    moment: np.ndarray = recent.T @ lagged
    return moment


def ewma_autocovariance(returns: np.ndarray, *, halflife: float, lag: int) -> np.ndarray:
    """``C_delta``, the EWMA lag-``delta`` autocovariance about zero. SPEC.md 5.2.

    At ``lag = 0`` this **is** :func:`ewma_second_moment` -- the same call, not a
    second implementation of it. That is deliberate and structural: stage 2's
    ``C_0`` term and stage 1's estimator cannot drift apart into two different
    centring conventions if there is only one of them.
    """
    if lag == 0:
        return ewma_second_moment(returns, halflife=halflife)
    if returns.ndim != 2:
        raise CovarianceError(
            f"ewma_autocovariance: expected a 2-D T x K array, got {returns.shape}"
        )
    if not np.all(np.isfinite(returns)):
        count = int(np.count_nonzero(~np.isfinite(returns)))
        raise CovarianceError(f"ewma_autocovariance: {count} non-finite observation(s)")
    weights = ewma_weights(returns.shape[0], halflife)
    return _weighted_lagged_moment(returns, weights, lag)


def assert_zero_mean_consistency(returns: np.ndarray, *, halflife: float) -> float:
    """Stage 2 is centred the same way as stage 1. Raises if it is not.

    SPEC.md 5.1.1 rules that the EWMA moments are taken about **zero**. The
    Newey-West lag terms ``Gamma_j = sum w * f_t * f_{t-j}'`` must therefore also
    be computed without demeaning. If stage 1 is zero-mean and stage 2 demeans,
    the pipeline estimates two different quantities and the inconsistency shows
    up **nowhere except in B** -- CLAUDE.md failure mode 7 exactly: two
    corrections that each pass their own sanity check.

    W3-P1 measured what the convention is worth, and the two are not
    interchangeable at the extreme even though they agree at the median: a 10.2 bp
    median shift in a factor volatility at the short horizon, against a 323 bp
    maximum on ``dollar`` at 2015-01-26 (``reports/mean_convention.md``,
    ``experiments.md`` rows 93-95).

    So the agreement is **asserted rather than relied on having been written the
    same way twice**. The check evaluates the lag-0 term through
    :func:`_weighted_lagged_moment` -- the general lagged path that also serves
    every lag the Bartlett sum uses -- and compares it against
    :func:`ewma_second_moment`. A later session that adds demeaning to the lagged
    path breaks this immediately, at the stage that did it, rather than three
    weeks later in a bias statistic.

    The tolerance is **derived, not chosen** (CLAUDE.md invariants 6 and 9). The
    two paths sum the same ``T`` products in different orders -- a Gram form
    ``(sqrt(w) F)' (sqrt(w) F)`` against a weighted product ``F' diag(w) F`` --
    so they may differ by accumulated rounding of order ``T * eps`` relative to
    the scale of the matrix. The largest diagonal entry is that scale, and it is
    a sum of non-negative terms, so it carries no cancellation. Anything above
    that bound is a change of convention, not arithmetic.

    Returns the observed discrepancy, so a caller can report the margin.
    """
    weights = ewma_weights(returns.shape[0], halflife)
    through_lag_path = _weighted_lagged_moment(returns, weights, 0)
    stage_one = ewma_second_moment(returns, halflife=halflife)
    discrepancy = float(np.max(np.abs(through_lag_path - stage_one)))
    scale = float(np.max(np.diag(stage_one)))
    tolerance = float(np.finfo(float).eps) * returns.shape[0] * max(scale, 1.0)
    if discrepancy > tolerance:
        raise CovarianceError(
            "the Newey-West lag path and the stage-1 EWMA disagree at lag 0: "
            f"max|difference| = {discrepancy:.3e} against a rounding bound of {tolerance:.3e}. "
            "SPEC.md 5.1.1 takes every moment about ZERO; if one of these two now subtracts a "
            "mean, the pipeline is estimating two different quantities and only B would show "
            "it. Fix the estimator, not this check."
        )
    return discrepancy


def bartlett_newey_west(returns: np.ndarray, *, halflife: float, lags: int) -> np.ndarray:
    """``C_0 + sum_{d=1..L} (1 - d/(L+1)) (C_d + C_d')``. SPEC.md 5.2.

    The Bartlett kernel: a linearly declining weight on each lag, reaching zero
    at ``L + 1``. Newey & West (1987); USE4 Table 4.1 supplies ``L``.

    **NOT scaled by any horizon.** SPEC.md 5.2 writes *"then scaled by the horizon
    (x21 for monthly)"* and that multiplier is deliberately absent here: the whole
    pipeline stays in daily units and :func:`mafrm.risk.horizon.scale_to_horizon`
    applies it once, downstream, at the point of consumption. See the W3-P2
    ruling at ``data.trading_days_per_month`` in ``config/model.yaml``.

    **Not PSD-guaranteed, and the reason is worth stating precisely** -- a later
    reader who knows the Bartlett kernel from the HAC literature will expect a
    guarantee here and should know why there is not one.

    Newey & West's non-negativity result is a property of the **equally-weighted**
    estimator. It rests on a quadratic form: with constant weights the truncated
    sum ``sum_{|d| <= L} (1 - |d|/(L+1)) Gamma_d`` can be rewritten as a sum of
    outer products of overlapping block sums of the series, and a sum of outer
    products is positive semi-definite whatever the data. **Exponential weighting
    breaks that rewriting** -- the weights differ across the observations inside
    each block, so the terms no longer collect into squares -- and truncating at
    ``L`` on a finite sample breaks what survives. USE4 specifies both an EWMA and
    a Bartlett correction, so this project meets a case the textbook result does
    not cover. That is not a defect in either source; it is what happens where
    they compose.

    That is why SPEC.md 5.2 calls the eigenvalue repair REQUIRED and why it is a
    declared stage of its own rather than a line inside this function.
    """
    if lags < 0:
        raise CovarianceError(f"bartlett_newey_west: lags must be non-negative, got {lags}")
    periods = returns.shape[0]
    if lags >= periods:
        raise CovarianceError(
            f"bartlett_newey_west: {lags} lag(s) need more than {lags} observations, got "
            f"{periods}. The longest lag would have no pairs at all."
        )
    weights = ewma_weights(periods, halflife)
    total: np.ndarray = ewma_second_moment(returns, halflife=halflife)
    for lag in range(1, lags + 1):
        kernel = bartlett_weight(lag, lags)
        cross = _weighted_lagged_moment(returns, weights, lag)
        total = total + kernel * (cross + cross.T)
    symmetric: np.ndarray = (total + total.T) / 2.0
    return symmetric


@dataclass(frozen=True)
class BartlettFallback:
    """Which variances reverted to the uncorrected EWMA estimate, and by how much.

    ``experiments.md`` row 100's registered remedy produces a number that the
    substitution itself destroys -- the non-positive variance the Bartlett sum
    returned -- so, like :class:`RepairReport`, it has to be handed out rather
    than re-derived from a matrix that no longer contains it.

    ``fired`` is False on the overwhelming majority of matrices, and an instance
    is still returned, because a diagnostic that only exists when something went
    wrong cannot be used to say that nothing did.
    """

    #: Positions whose variance was substituted. Empty when nothing fired.
    columns: tuple[int, ...]
    #: What the Bartlett sum returned at those positions -- all non-positive.
    corrected: tuple[float, ...]
    #: The uncorrected EWMA variances substituted in. All strictly positive.
    substituted: tuple[float, ...]
    #: How many columns the matrix has, so a count can be read as a share.
    size: int

    @property
    def fired(self) -> bool:
        return bool(self.columns)

    def render(self) -> str:
        if not self.fired:
            return f"bartlett fallback: not fired ({self.size} factor(s))"
        pairs = ", ".join(
            f"[{index}] {bad:.3e} -> {good:.3e}"
            for index, bad, good in zip(self.columns, self.corrected, self.substituted, strict=True)
        )
        return (
            f"bartlett fallback: FIRED on {len(self.columns)} of {self.size} factor(s) -- {pairs}"
        )


def newey_west_covariance(
    returns: np.ndarray,
    *,
    volatility_halflife: float,
    volatility_lags: int,
    correlation_halflife: float,
    correlation_lags: int,
) -> tuple[np.ndarray, BartlettFallback]:
    """SPEC.md 5.2 applied the way SPEC.md 5.1 separates the estimate.

    **Newey-West is applied twice, not once** -- to the volatility term at
    ``volatility_halflife`` with ``volatility_lags`` (5), and separately to the
    correlation term at ``correlation_halflife`` with ``correlation_lags`` (2) --
    and the two are recombined as ``F_ij = rho_ij * sigma_i * sigma_j``.

    RULING (W3-P2, 2026-08-31): SEPARATED, NOT ASSEMBLED
        SPEC.md 5.2's pseudocode shows a single ``cov_ewa(F, half_life, i)`` loop,
        which reads as one Newey-West correction applied to an already-assembled
        ``F_0``. **The two lag counts are the proof that it is not.** If the
        correction were applied once to the assembled matrix there would be one
        lag number in USE4 Table 4.1, not two; serial correlation in volatility
        and serial correlation in correlation are different phenomena with
        different persistence, which is the same reason SPEC.md 5.1 gives them
        different half-lives in the first place. The pseudocode is exposition, not
        specification -- the same class of artefact as the ``lambda^a (lambda - 1)
        + 1`` power-law garble SPEC.md 5.3 flags in one PDF rendering of the MSCI
        paper, where the published text and the open-source implementations agree
        against the rendering.

        The assembled variant was **not run as a comparison**. This is a ruling on
        a structural argument, not a sweep, and running the alternative would make
        it one; ``experiments.md`` records that under W3-P2.

    THE DOUBLE-COUNTING TRAP
        Newey-West does not preserve a unit diagonal. Correcting ``rho`` directly
        and recombining would count each factor's variance twice -- once through
        ``sigma``, and again through the inflated diagonal left in ``rho``. The
        correction is therefore applied to the **second moment** at the correlation
        half-life and the correlation is formed from the corrected matrix, which
        renormalises the diagonal to exactly one by construction. That the diagonal
        really is unit is asserted below rather than left to the reader.

    A NON-POSITIVE VARIANCE FALLS BACK TO STAGE 1, ON A PRE-REGISTERED REMEDY
        The Bartlett sum can drive a variance below zero -- see
        :func:`bartlett_newey_west` for why the textbook non-negativity guarantee
        does not reach an exponentially weighted estimator. Newey & West's result
        is a property of the **equally weighted** HAC estimator, where the
        truncated sum rewrites as a sum of outer products of block sums and is
        therefore PSD whatever the data; exponential weights break that
        rewriting, because the weights differ across the observations inside each
        block and the terms no longer collect into squares.

        **The remedy was pre-registered in W3-P2** (``experiments.md`` row 100),
        four sessions before it fired, precisely so that it would not be chosen by
        the session that hit the failure: fall back to the **uncorrected EWMA
        variance** for that factor, count the substitution, report it. It invents
        no constant (CLAUDE.md invariant 9) -- the fallback value is one stage 1
        already computes -- it degrades gracefully rather than failing the whole
        matrix, and it is visible, where a silent floor would be neither. A floor
        was and remains refused: it would produce a correlation with an arbitrary
        denominator and fail silently.

        **It fired in W3-P5, at K = 3 rather than the K = 56 row 100 predicted**,
        on Model B's detoned panel at ``T = 7..10`` -- ``K/T`` around 0.43. Row
        100's amendment records why the prediction being wrong is informative:
        this is a ``K/T`` phenomenon and not a ``K`` phenomenon, the same shape as
        SPEC.md 5.2's PSD repair, and the remedy attaches to the **mechanism**
        above rather than to any panel size.

        **This function knows nothing about which factor set it is looking at**,
        and that is asserted rather than argued: ``tests/test_covariance.py``
        exercises the fallback on a synthetic panel unrelated to either model and
        requires it to behave identically. See ``experiments.md`` row 129.
    """
    volatility = bartlett_newey_west(returns, halflife=volatility_halflife, lags=volatility_lags)
    variances = np.diag(volatility).copy()
    uncorrected = np.diag(ewma_second_moment(returns, halflife=volatility_halflife))
    failed = np.flatnonzero(variances <= 0.0)
    fallback = BartlettFallback(
        columns=tuple(int(index) for index in failed),
        corrected=tuple(float(variances[index]) for index in failed),
        substituted=tuple(float(uncorrected[index]) for index in failed),
        size=int(variances.size),
    )
    if failed.size:
        if np.any(uncorrected[failed] <= 0.0):
            raise CovarianceError(
                f"newey_west_covariance: the Bartlett correction drove factor(s) "
                f"{fallback.columns} non-positive AND the stage-1 EWMA variance the "
                "experiments.md row 100 fallback substitutes is itself non-positive "
                f"({uncorrected[failed]}). The remedy has nothing to fall back TO, which is "
                "outside what row 100 registered and is refused rather than patched: a "
                "second-moment estimate about zero cannot be negative unless the window is "
                "degenerate, so this is a caller handing in a window that is empty or "
                "constant, not an estimator failure to be repaired here."
            )
        variances[failed] = uncorrected[failed]
    correlation_moment = bartlett_newey_west(
        returns, halflife=correlation_halflife, lags=correlation_lags
    )
    correlation = correlation_from_covariance(correlation_moment)
    diagonal = np.diag(correlation)
    if not np.array_equal(diagonal, np.ones_like(diagonal)):
        raise CovarianceError(
            "newey_west_covariance: the correlation diagonal is not exactly 1 after the "
            f"Bartlett correction ({diagonal}). Recombining with it would count every "
            "variance twice."
        )
    deviation = np.sqrt(variances)
    covariance: np.ndarray = correlation * np.outer(deviation, deviation)
    symmetric: np.ndarray = (covariance + covariance.T) / 2.0
    return symmetric, fallback


def _stage_newey_west(matrix: np.ndarray, inputs: StageInputs, config: RiskConfig) -> StageOutput:
    """SPEC.md 5.2. Discards ``matrix``: the correction is on the lag structure.

    Stage 1's output is not an input here. Newey-West re-estimates from the
    returns because the correction is a statement about serial correlation in
    ``f``, which a ``K x K`` contemporaneous matrix has already integrated away.
    Both half-lives are re-run at their own lag counts, so the stage consumes the
    same panel stage 1 did and nothing else.
    """
    del matrix
    if config.mean_convention != "zero":
        raise CovarianceError(
            f"_stage_newey_west: mean_convention is {config.mean_convention!r}. The lag terms "
            "here are computed about zero (SPEC.md 5.1.1); a demeaned convention would need "
            "them demeaned too, and that is a change to make deliberately, not to inherit."
        )
    assert_zero_mean_consistency(inputs.returns, halflife=float(config.volatility_halflife))
    assert_zero_mean_consistency(inputs.returns, halflife=float(config.correlation_halflife))
    matrix, fallback = newey_west_covariance(
        inputs.returns,
        volatility_halflife=float(config.volatility_halflife),
        volatility_lags=config.volatility_newey_west_lags,
        correlation_halflife=float(config.correlation_halflife),
        correlation_lags=config.correlation_newey_west_lags,
    )
    return StageOutput(matrix=matrix, bartlett=fallback)


# ---------------------------------------------------------------------------
# Stage 3 -- SPEC.md 5.2's mandatory PSD repair
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepairReport:
    """What the PSD repair did, whether or not it did anything. SPEC.md 5.2.

    *"Log how often it fires and by how much -- that number is itself a
    diagnostic."* This is that log for one matrix; ``reports/psd_repairs.md``
    aggregates it over the panel.
    """

    stage: str
    size: int
    fired: bool
    #: How many eigenvalues were raised to the floor.
    floored: int
    minimum_eigenvalue_before: float
    maximum_eigenvalue_before: float
    #: ``model.numerics.psd_eigenvalue_floor``, carried so a report is self-contained.
    floor: float

    @property
    def floor_relative_to_maximum(self) -> float:
        """The floor as a fraction of ``lambda_max``.

        The floor is an **absolute** constant and is therefore scale-dependent:
        the same 1e-14 is a different statement about a matrix in decimal returns
        and one in basis points. This ratio is what makes it readable, and it is
        why the report quotes the floor against the spectrum rather than alone.
        """
        if self.maximum_eigenvalue_before <= 0.0:
            return float("inf")
        return self.floor / self.maximum_eigenvalue_before

    def render(self) -> str:
        """One line. Says nothing happened when nothing happened.

        The stage name is deliberately NOT repeated here: this line is printed
        underneath the stage's own line, and a report that wants the label has
        :attr:`stage`.
        """
        if not self.fired:
            return (
                f"repair: not fired, least eigenvalue {self.minimum_eigenvalue_before:+.6e} "
                f"above the {self.floor:.0e} floor"
            )
        return (
            f"repair: FIRED on {self.floored} of {self.size} eigenvalue(s), most negative "
            f"{self.minimum_eigenvalue_before:+.6e}, floored to {self.floor:.0e} "
            f"({self.floor_relative_to_maximum:.2e} of lambda_max "
            f"{self.maximum_eigenvalue_before:.6e})"
        )


def psd_repair(
    matrix: np.ndarray, *, floor: float, stage: str = "psd_repair"
) -> tuple[np.ndarray, RepairReport]:
    """Floor non-positive eigenvalues at ``floor`` and rebuild. SPEC.md 5.2.

    ``D[D <= 0] = floor``, exactly as SPEC.md 5.2's pseudocode writes it, then
    ``U diag(D) U'``. CLAUDE.md invariant 4 fixes ``floor`` at 1e-14 and it is
    read from ``model.numerics.psd_eigenvalue_floor``, never written here.

    **A no-op is a true no-op.** When no eigenvalue needs flooring the input is
    returned unchanged rather than round-tripped through its own
    eigendecomposition, which would perturb it in the last few bits for nothing.
    That keeps the pipeline bit-identical on the (expected, at K = 6) path where
    the repair never fires, so the golden fixture measures the estimator rather
    than LAPACK's reconstruction error.
    """
    eigenvalues, vectors = np.linalg.eigh(matrix)
    breaches = eigenvalues <= 0.0
    report = RepairReport(
        stage=stage,
        size=int(matrix.shape[0]),
        fired=bool(np.any(breaches)),
        floored=int(np.count_nonzero(breaches)),
        minimum_eigenvalue_before=float(eigenvalues[0]),
        maximum_eigenvalue_before=float(eigenvalues[-1]),
        floor=float(floor),
    )
    if not report.fired:
        return matrix, report
    repaired_eigenvalues = np.where(breaches, floor, eigenvalues)
    rebuilt: np.ndarray = vectors @ np.diag(repaired_eigenvalues) @ vectors.T
    symmetric: np.ndarray = (rebuilt + rebuilt.T) / 2.0
    return symmetric, report


def _stage_psd_repair(matrix: np.ndarray, inputs: StageInputs, config: RiskConfig) -> StageOutput:
    """SPEC.md 5.2's REQUIRED repair, as a declared stage of its own."""
    del inputs
    repaired, report = psd_repair(
        matrix,
        floor=config.psd_eigenvalue_floor,
        stage=f"psd_repair [{config.horizon}]",
    )
    return StageOutput(matrix=repaired, repair=report)


# ---------------------------------------------------------------------------
# Stage 4 -- SPEC.md 5.3, the eigenfactor risk adjustment
# ---------------------------------------------------------------------------


def _stage_eigenfactor(matrix: np.ndarray, inputs: StageInputs, config: RiskConfig) -> StageOutput:
    """SPEC.md 5.3, run in correlation space per SPEC.md 5.3.2.

    The whole technique lives in :mod:`mafrm.risk.eigenfactor`, including the
    argument for why it diagonalises ``rho`` rather than ``F``. This function is
    the wiring: it refuses to choose ``a`` for the caller, and it hands the
    adjustment object out so the build can report the curve.

    ``inputs.returns`` supplies only its **length**. The correction is a statement about
    the sampling distribution of the estimator, so the simulation needs the window
    the estimator saw and the half-life it ran at, and nothing else about the
    panel. No value in it is read. W3-P3b: the simulation follows the estimator
    rather than an equal-weight surrogate for it, so there is no ``T`` here to
    choose and none to get wrong.
    """
    # Local, and the direction is deliberate. ``eigenfactor`` needs this
    # module's ``correlation_from_covariance`` -- one splitter, not two that can
    # drift -- so the import graph has to be broken somewhere, and it is broken
    # here rather than there: the module a reader studies for the technique
    # keeps its imports at the top, and the wiring carries the cost.
    from mafrm.risk.eigenfactor import eigenfactor_adjustment

    if config.eigenfactor_variant is None:
        raise CovarianceError(
            "eigenfactor: no eigenfactor_variant on this RiskConfig. SPEC.md 5.3 publishes "
            f"{list(config.eigenfactor_scaling)} and CLAUDE.md's parameter table says RUN "
            "BOTH -- a = 1.0 is attribution-facing and what USE4 runs in production, a = 1.4 "
            "is optimizer-facing. They are two model variants, so this stage will not pick "
            "one for you: call RiskConfig.for_scaling(a) and say which model you are "
            "building. A default here is how one gets inherited without being logged."
        )
    adjustment = eigenfactor_adjustment(
        matrix,
        trials=config.eigenfactor_monte_carlo_trials,
        # The CORRELATION half-life, because the matrix being diagonalised is
        # rho_hat (SPEC.md 5.3.2). 504d at both horizons, so this stage is
        # horizon-independent by construction -- see the W3-P3b note in
        # experiments.md, where that is recorded as a consequence and not a bug.
        halflife=float(config.correlation_halflife),
        observations=int(inputs.returns.shape[0]),
        scalings=config.eigenfactor_scaling,
        seed=config.seed,
        floor=config.psd_eigenvalue_floor,
    )
    return StageOutput(
        matrix=adjustment.variant(config.eigenfactor_variant).adjusted,
        eigenfactor=adjustment,
    )


# ---------------------------------------------------------------------------
# Stage 5 -- SPEC.md 5.4, the volatility regime adjustment
# ---------------------------------------------------------------------------


def _stage_volatility_regime(
    matrix: np.ndarray, inputs: StageInputs, config: RiskConfig
) -> StageOutput:
    """SPEC.md 5.4. **Last, and the order is load-bearing.**

    A pure level scaling: ``F <- lambda_F^2 F``. It has to be calibrated against
    the final matrix, so it runs after the eigenfactor adjustment rather than
    before it, and ``config/model.yaml`` declares that order rather than leaving
    it to the order of function calls in this file.

    Two properties follow from ``lambda_F^2`` being one positive scalar and are
    asserted in ``tests/test_regime.py`` rather than left to a reader:

    1. **PSD is preserved exactly.** Scaling by a positive constant scales every
       eigenvalue by it. The invariant-4 assertion after this stage therefore
       cannot fire on anything this stage did -- which is the reason to run it
       anyway, since a pipeline whose assertions sit only where failure is
       expected cannot tell you the stages before it were fine.
    2. **The correlation matrix is unchanged.** ``corr(cF) = corr(F)`` for
       ``c > 0``. SPEC.md 5.4 says this stage changes *the level of risk, not the
       correlation structure*, and that is a testable claim rather than a
       description.

    The bias history comes from the caller and there is no default. See
    :class:`StageInputs`.
    """
    # Local, to break the cycle: ``regime`` runs this pipeline once per date.
    from mafrm.risk.regime import regime_multiplier

    if inputs.regime_bias is None:
        raise CovarianceError(
            "volatility_regime: no regime_bias on this run. SPEC.md 5.4 standardises each "
            "factor return by THE FORECAST MADE AT t-1, so this stage consumes a history of "
            "forecasts rather than a window of returns, and building that history means "
            "running this pipeline once per date. SPEC.md 5.1.2 leaves the burn-in that "
            "governs the history with the caller, so the caller supplies it: build one with "
            "mafrm.risk.regime.forecast_history and pass run_pipeline(..., regime_bias=...). "
            "There is deliberately no default -- an un-adjusted matrix is PSD, symmetric, "
            "identically conditioned and simply too small, which is a failure nothing "
            "downstream could detect."
        )
    multiplier = regime_multiplier(inputs.regime_bias, halflife=config.volatility_regime_halflife)
    return StageOutput(
        matrix=multiplier.lambda_squared * matrix,
        regime=multiplier,
    )


#: Stage name -> implementation, keyed by the names declared in
#: ``covariance.stages``. A name declared in the config and absent here is what
#: :func:`missing_stages` reports and what ``make model`` fails on.
_STAGES: Mapping[str, Callable[[np.ndarray, StageInputs, RiskConfig], StageOutput]] = {
    "ewma": _stage_ewma,
    "newey_west": _stage_newey_west,
    "psd_repair": _stage_psd_repair,
    "eigenfactor": _stage_eigenfactor,
    "volatility_regime": _stage_volatility_regime,
}

#: Stages whose output is MEASURED and not asserted PSD. Exactly one, and it is
#: a reading of CLAUDE.md invariant 4 rather than a relaxation of it -- see
#: :mod:`mafrm.risk.checks` for the full argument.
#:
#: In one line: invariant 4 exists to catch *silent* non-PSD, and ``newey_west``
#: is the one stage in SPEC.md 5 whose non-PSD output is declared in advance and
#: whose successor in ``covariance.stages`` exists to repair it. Asserting between
#: the two would detect no defect; it would convert an expected and handled
#: outcome into a crash, and the only way to keep the pipeline running would then
#: be to widen ``psd_minimum_eigenvalue``, which is the move the residual stopping
#: rule forbids.
#:
#: **A later session reading a missing assertion as an oversight and "fixing" it
#: into a crash is the failure this comment exists to prevent.** The list is
#: pinned by ``tests/test_covariance.py``, so it cannot grow a second member
#: without a test changing.
_MEASURED_NOT_ASSERTED: frozenset[str] = frozenset({"newey_west"})


def measured_not_asserted_stages() -> frozenset[str]:
    """Stages that report their spectrum instead of asserting PSD. See above."""
    return _MEASURED_NOT_ASSERTED


def implemented_stages() -> tuple[str, ...]:
    """Stage names this module can actually run."""
    return tuple(_STAGES)


def declared_stages(config: RiskConfig) -> tuple[str, ...]:
    """Stage names ``config/model.yaml`` requires, in application order."""
    return config.stages


def missing_stages(config: RiskConfig) -> tuple[str, ...]:
    """Declared stages with no implementation, in declared order.

    Non-empty means the pipeline is incomplete and any matrix it produces is
    partial. ``make model`` fails on this rather than printing a green line for
    a build that skipped the stages SPEC.md 5 declares but nobody wrote.
    """
    return tuple(stage for stage in config.stages if stage not in _STAGES)


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageResult:
    """One stage's output and what was observed about it. Invariant 4.

    Exactly one of :attr:`check` and :attr:`spectrum` is set. ``check`` means the
    stage was **asserted** PSD; ``spectrum`` means it was **measured** and not
    asserted, which is true of ``newey_west`` alone and for the reason recorded
    at :data:`_MEASURED_NOT_ASSERTED`.
    """

    name: str
    matrix: np.ndarray
    check: PsdCheck | None
    spectrum: Spectrum | None
    #: Set only by ``psd_repair``, which is the only stage that repairs anything.
    repair: RepairReport | None = None
    #: Set only by ``newey_west``. experiments.md row 100's registered fallback.
    bartlett: BartlettFallback | None = None
    #: Set only by ``eigenfactor``. SPEC.md 5.3's curve, fit and both variants.
    eigenfactor: EigenfactorAdjustment | None = None
    #: Set only by ``volatility_regime``. SPEC.md 5.4's ``lambda_F^2``.
    regime: RegimeMultiplier | None = None

    def __post_init__(self) -> None:
        if (self.check is None) == (self.spectrum is None):
            raise CovarianceError(
                f"StageResult[{self.name}]: exactly one of check and spectrum must be set. A "
                "stage is either asserted PSD or measured without being asserted, and which "
                "one it was is not allowed to be ambiguous."
            )

    @property
    def asserted(self) -> bool:
        """True when CLAUDE.md invariant 4's assertion ran on this stage."""
        return self.check is not None

    @property
    def minimum_eigenvalue(self) -> float:
        """``lambda_min``, however it was observed."""
        return (self.check or self.spectrum).minimum_eigenvalue  # type: ignore[union-attr]

    @property
    def maximum_eigenvalue(self) -> float:
        """``lambda_max``, however it was observed."""
        return (self.check or self.spectrum).maximum_eigenvalue  # type: ignore[union-attr]

    @property
    def condition_number(self) -> float:
        """``lambda_max / lambda_min``; ``inf`` at or below zero."""
        return (self.check or self.spectrum).condition_number  # type: ignore[union-attr]

    def render(self) -> str:
        """One line, with the stage's own diagnostic appended where it has one."""
        observed = (self.check or self.spectrum).render()  # type: ignore[union-attr]
        if self.repair is not None:
            return f"{observed}\n  {self.repair.render()}"
        # Rendered only when it fired. An un-fired fallback on every date of a
        # 4,000-date scan would bury the lines that matter, and `bartlett_fired`
        # on the build is what a caller counts with.
        if self.bartlett is not None and self.bartlett.fired:
            return f"{observed}\n  {self.bartlett.render()}"
        if self.eigenfactor is not None:
            return "\n".join((observed, *(f"  {line}" for line in self.eigenfactor.render())))
        if self.regime is not None:
            return f"{observed}\n  {self.regime.render()}"
        return observed


@dataclass(frozen=True)
class CovarianceBuild:
    """Everything one pipeline run produced, including what it could not run.

    The unimplemented stages are carried rather than dropped so that a caller
    holding this object can tell whether it is holding a finished covariance
    matrix or the first stage of one.
    """

    columns: tuple[str, ...]
    observations: int
    stages: tuple[StageResult, ...]
    stages_not_implemented: tuple[str, ...]
    #: Every stage ``covariance.stages`` declares, in order, whatever ran.
    declared: tuple[str, ...]
    #: Set when a caller asked to stop early. Diagnostics only; see
    #: :func:`run_pipeline`.
    stopped_after: str | None
    volatility_halflife: int
    correlation_halflife: int
    horizon: str
    #: Kish effective sample size of the volatility weights over this window.
    realised_effective_sample_size: float

    @property
    def factors(self) -> int:
        """``K``."""
        return len(self.columns)

    @property
    def matrix(self) -> np.ndarray:
        """The last stage's output. **Partial while ``complete`` is False.**"""
        if not self.stages:
            raise CovarianceError("CovarianceBuild: no stage ran")
        return self.stages[-1].matrix

    @property
    def repairs(self) -> tuple[RepairReport, ...]:
        """Every repair report this build produced, fired or not. SPEC.md 5.2."""
        return tuple(stage.repair for stage in self.stages if stage.repair is not None)

    @property
    def bartlett_fallbacks(self) -> tuple[BartlettFallback, ...]:
        """Every ``newey_west`` stage's row-100 fallback report, fired or not."""
        return tuple(stage.bartlett for stage in self.stages if stage.bartlett is not None)

    @property
    def bartlett_fired(self) -> bool:
        """True when row 100's registered remedy substituted at least one variance."""
        return any(report.fired for report in self.bartlett_fallbacks)

    @property
    def repair_fired(self) -> bool:
        """Whether SPEC.md 5.2's PSD repair floored anything in this build."""
        return any(report.fired for report in self.repairs)

    @property
    def eigenfactor(self) -> EigenfactorAdjustment | None:
        """SPEC.md 5.3's adjustment, or ``None`` if that stage did not run."""
        for stage in self.stages:
            if stage.eigenfactor is not None:
                return stage.eigenfactor
        return None

    @property
    def regime(self) -> RegimeMultiplier | None:
        """SPEC.md 5.4's multiplier, or ``None`` if that stage did not run."""
        for stage in self.stages:
            if stage.regime is not None:
                return stage.regime
        return None

    @property
    def frame(self) -> pd.DataFrame:
        """:attr:`matrix` with the input's column labels on both axes."""
        return pd.DataFrame(self.matrix, index=list(self.columns), columns=list(self.columns))

    @property
    def forecast(self) -> np.ndarray:
        """The finished covariance forecast. **Raises unless every stage ran.**

        This is the accessor anything that PUBLISHES a number must use -- the
        optimizer, the validation battery, a report quoting a risk forecast.
        :attr:`matrix` is the last stage's output and is partial whenever the
        pipeline is incomplete or a diagnostic stopped it early; that is useful
        for a stage diagnostic and is exactly wrong for a forecast.

        The two are separated **mechanically** rather than by a warning, because
        the failure this prevents is silent: a matrix missing SPEC.md 5.4's
        regime adjustment is still PSD, symmetric and identically conditioned,
        and passes every check the pipeline runs. It is simply too small.

        Since W3-P4 it stops raising once a forecast history is supplied, which
        is the first time in the project this accessor has returned anything. It
        still raises on a build that ran without one, and on any build a
        diagnostic stopped short.
        """
        if not self.complete:
            raise CovarianceError(
                "CovarianceBuild.forecast: this build is INCOMPLETE and its matrix must not be "
                f"published as a risk forecast. Declared {list(self.declared)}; did not run "
                f"{list(self.stages_not_run)}"
                + (
                    f"; stopped after {self.stopped_after!r} at the caller's request"
                    if self.stopped_after
                    else ""
                )
                + ". Use .matrix if you want a stage's partial output for a diagnostic, and say "
                "in the caller why a partial matrix is the right input there."
            )
        return self.matrix

    @property
    def stages_not_run(self) -> tuple[str, ...]:
        """Declared stages with no result, whether unimplemented or stopped short.

        A superset of :attr:`stages_not_implemented`: a diagnostic that asked
        :func:`run_pipeline` to stop after an intermediate stage lands here too,
        so a build that was deliberately truncated can never report itself
        complete.
        """
        produced = {stage.name for stage in self.stages}
        return tuple(name for name in self.declared if name not in produced)

    @property
    def complete(self) -> bool:
        """True only when every stage SPEC.md 5 declares actually ran."""
        return not self.stages_not_run

    @property
    def effective_sample_size(self) -> float:
        """``T_eff = 2*tau/ln 2`` at the volatility half-life. SPEC.md 15.2's table."""
        return effective_sample_size(float(self.volatility_halflife))

    @property
    def k_over_t_eff(self) -> float:
        """``K / T_eff``, asymptotic. The x-axis of SPEC.md 15.2's headline plot."""
        return self.factors / self.effective_sample_size

    @property
    def k_over_realised_t_eff(self) -> float:
        """``K / T_eff`` using the window's realised Kish sample size.

        Equal to :attr:`k_over_t_eff` for a window long relative to the
        half-life, and larger for a short one -- which is the honest reading,
        since a short window really does estimate ``K`` parameters from less.
        """
        return self.factors / self.realised_effective_sample_size

    @property
    def shepard_multiplier(self) -> float:
        """``[1 - K/T_eff]^-2``. Shepard's second-order correction, SPEC.md 6.4.

        The factor by which an optimizer's forecast VARIANCE is understated from
        estimation error alone -- the volatility multiplier is its square root
        (:mod:`mafrm.risk.shepard`, W4-P3). ``inf`` once ``K >= T_eff``, which is the
        regime SPEC.md 15.2 uses to say a sample covariance at ``N=500`` is not
        a thing that works.
        """
        return self._second_order.variance_multiplier

    @property
    def _second_order(self) -> SecondOrderRisk:
        return second_order_risk(self.factors, effective_observations=self.effective_sample_size)

    def render(self) -> tuple[str, ...]:
        """Lines for ``make model``. Reports K/T_eff whether or not asked."""
        lines = [
            f"covariance [{self.horizon}] K={self.factors}, T={self.observations:,} obs, "
            f"vol half-life {self.volatility_halflife}d, corr half-life "
            f"{self.correlation_halflife}d, moments about zero",
            f"  K/T_eff = {self.factors}/{self.effective_sample_size:.0f} = "
            f"{self.k_over_t_eff:.4f} asymptotic; "
            f"{self.factors}/{self.realised_effective_sample_size:.0f} = "
            f"{self.k_over_realised_t_eff:.4f} realised over this window",
            f"  Shepard [1 - K/T_eff]^-2 = {self.shepard_multiplier:.4f} "
            f"({100.0 * (self.shepard_multiplier - 1.0):+.1f}% understatement in variance, "
            f"{self._second_order.render(Unit.VOLATILITY)})",
        ]
        lines.extend(f"  {line}" for stage in self.stages for line in stage.render().splitlines())
        if self.stopped_after is not None:
            lines.append(
                f"  STOPPED AFTER {self.stopped_after} at the caller's request -- the matrix "
                "above is PARTIAL and is a stage diagnostic, not a forecast"
            )
        if self.stages_not_implemented:
            lines.append(
                "  NOT IMPLEMENTED: " + ", ".join(self.stages_not_implemented) + " -- the matrix "
                "above is PARTIAL"
            )
        return tuple(lines)


def _validate(factor_returns: pd.DataFrame) -> np.ndarray:
    if factor_returns.ndim != 2:
        raise CovarianceError(f"expected a T x K frame, got {factor_returns.ndim} dimension(s)")
    periods, factors = factor_returns.shape
    if factors < 1:
        raise CovarianceError("expected at least one factor column")
    if periods < factors:
        raise CovarianceError(
            f"{periods} observation(s) for {factors} factor(s): the second moment is singular by "
            "construction below T = K. This is not a burn-in rule -- it is the point at which the "
            "arithmetic has no answer at all."
        )
    if factor_returns.columns.has_duplicates:
        duplicated = factor_returns.columns[factor_returns.columns.duplicated()].tolist()
        raise CovarianceError(f"duplicate factor column(s): {duplicated}")
    return factor_returns.to_numpy(dtype=float)


def run_pipeline(
    factor_returns: pd.DataFrame,
    config: RiskConfig,
    *,
    stop_after: str | None = None,
    regime_bias: np.ndarray | None = None,
) -> CovarianceBuild:
    """Run every implemented stage of SPEC.md 5, in the order the config declares.

    ``stop_after`` is for DIAGNOSTICS THAT MEASURE AN INTERMEDIATE STAGE and is
    not a way to skip one. ``reports/psd_repairs.md`` rebuilds the pipeline on an
    expanding window ending at every date in the panel -- 4,141 builds per
    horizon -- to log how often SPEC.md 5.2's repair fires; that report is about
    a stage two steps from the end, and running SPEC.md 5.3's 2,000-trial Monte
    Carlo 4,141 times to produce a number it then discards would cost hours and
    tell it nothing.

    The stop is **visible in the result**: the stages that did not run are listed
    in :attr:`CovarianceBuild.stages_not_run`, :attr:`CovarianceBuild.complete`
    is False, and :meth:`CovarianceBuild.render` says where it stopped. A
    truncated build can therefore never be mistaken for a finished forecast,
    which is the only reason this parameter is safe to have.

    Asserts PSD after **each** stage (CLAUDE.md invariant 4), not once at the
    end: individually-safe corrections interact badly, and a single assertion at
    the end cannot say which stage broke.

    Returns the full :class:`CovarianceBuild` including the K/T_eff diagnostics
    and the list of declared stages that do not yet exist.
    ``:func:`build_covariance`` is the SPEC.md 15.2 entry point and returns only
    the matrix.
    """
    if stop_after is not None and stop_after not in config.stages:
        raise CovarianceError(
            f"run_pipeline: stop_after={stop_after!r} is not a declared stage; "
            f"config/model.yaml declares {list(config.stages)}"
        )
    values = _validate(factor_returns)
    periods = values.shape[0]
    inputs = StageInputs(returns=values, regime_bias=regime_bias)

    matrix = np.empty((0, 0))
    results: list[StageResult] = []
    for name in config.stages:
        implementation = _STAGES.get(name)
        if implementation is None:
            continue
        output = implementation(matrix, inputs, config)
        matrix = output.matrix
        label = f"{name} [{config.horizon}]"
        check: PsdCheck | None = None
        spectrum: Spectrum | None = None
        if name in _MEASURED_NOT_ASSERTED:
            spectrum = measure_spectrum(matrix, label, expected_size=values.shape[1])
        else:
            check = assert_psd(matrix, label, expected_size=values.shape[1])
        results.append(
            StageResult(
                name=name,
                matrix=matrix,
                check=check,
                spectrum=spectrum,
                repair=output.repair,
                bartlett=output.bartlett,
                eigenfactor=output.eigenfactor,
                regime=output.regime,
            )
        )
        if name == stop_after:
            break

    if not results:
        raise CovarianceError(
            f"no declared stage is implemented: declared {list(config.stages)}, "
            f"implemented {list(_STAGES)}"
        )

    return CovarianceBuild(
        columns=tuple(str(column) for column in factor_returns.columns),
        observations=periods,
        stages=tuple(results),
        stages_not_implemented=missing_stages(config),
        declared=tuple(config.stages),
        stopped_after=stop_after,
        volatility_halflife=config.volatility_halflife,
        correlation_halflife=config.correlation_halflife,
        horizon=config.horizon,
        realised_effective_sample_size=realised_effective_sample_size(
            ewma_weights(periods, float(config.volatility_halflife))
        ),
    )


def build_covariance(
    factor_returns: pd.DataFrame,
    config: RiskConfig,
    *,
    regime_bias: np.ndarray | None = None,
) -> np.ndarray:
    """The SPEC.md 15.2 entry point. ``T x K`` frame in, ``K x K`` matrix out.

    Knows nothing about what the columns are. Runs the stages declared in
    ``covariance.stages`` that are implemented, asserting PSD after each.

    **As of W3-P4 that is all five stages**, so the matrix returned is a finished
    forecast -- provided ``regime_bias`` is supplied. It is not optional in
    practice: SPEC.md 5.4's stage refuses to run without a forecast history and
    this function does not build one, because building it means running this
    pipeline once per date (see :class:`StageInputs`). A caller that wants the
    pre-VRA matrix should use :func:`run_pipeline` with
    ``stop_after="eigenfactor"``, where the truncation is visible in the result,
    rather than a ``build_covariance`` call that silently omits a stage.
    """
    return run_pipeline(factor_returns, config, regime_bias=regime_bias).matrix
