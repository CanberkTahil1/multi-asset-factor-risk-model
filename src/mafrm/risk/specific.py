"""SPEC.md 5.5's specific risk. USE4 sections 4.5 and 4.4, built against SPEC.md 15.2.

Four legs, in the order SPEC.md 5.5 fixes them::

    (a) time series  ->  (b) structural  ->  (c) blend  ->  (d) Bayesian shrinkage

then the specific volatility regime multiplier, which is SPEC.md 5.4's second
paragraph and shares its half-life with the factor leg.

WHAT THIS MODULE MAY KNOW
-------------------------

A ``T x N`` frame of residuals, an ``N x K`` frame of exposures, a length-``N``
vector of bucket labels, and a :class:`~mafrm.risk.config.RiskConfig`. CLAUDE.md
invariant 10: nothing here may branch on what the columns are. The buckets arrive
as **opaque labels** -- this module groups by equality and never interprets one,
which is why SPEC.md 5.5's own bucket names appear nowhere in this file. The
caller that knows what a bucket means is the caller that chose the labels.

THE HEADLINE, BECAUSE IT GOVERNS HOW EVERY NUMBER BELOW SHOULD BE READ
-----------------------------------------------------------------------

**SPEC.md 5.5 is a large-``N`` cross-sectional construction and most of it
degenerates at small ``N``.** Leg (a) works at any ``N``. Leg (b) regresses a
length-``N`` vector on ``K + 1`` columns and is barely identified when ``N`` is
close to ``K``. Leg (c) exists because large universes carry thousands of ragged
histories. Leg (d) needs populated buckets.

All four are implemented as specified and the degeneration is **measured and
reported** rather than designed around. Every dataclass below therefore carries
the diagnostic that says how far into its own regime it is:
:attr:`StructuralFit.residual_degrees_of_freedom`,
:attr:`BlendWeight.maximum_z`, :attr:`BucketStatistics.singletons`.

LEG (a) -- TWO HALF-LIVES, AND WHY IT IS NOT ONE
-------------------------------------------------

SPEC.md 5.5(a): *"EWMA of squared specific returns, half-life 84d (short) / 252d
(long), plus a Newey-West term over 5 lags with autocorrelation half-life 252d."*
Two half-lives are named for one estimator, so the reading that uses both is the
one taken here: the ``C_0`` term runs at ``specific_volatility_halflife`` and the
Bartlett lag terms run at ``specific_autocorrelation_halflife``. Any other
reading leaves one of the two published numbers with nothing to do. At the long
horizon they coincide at 252d, so the distinction is visible only at the short
one -- which is also why it would be easy to write the single-half-life version
and never notice.

Both the pair-weighting convention and the kernel come from
:mod:`mafrm.risk.covariance` rather than being written again here. See the note
above :func:`~mafrm.risk.covariance.bartlett_weight` for why the diagonal is
evaluated on its own ``O(N T)`` path and what asserts the two paths agree.

A NON-POSITIVE SPECIFIC VARIANCE FALLS BACK TO THE UNCORRECTED EWMA
--------------------------------------------------------------------

``experiments.md`` row 100's registered remedy, applied unchanged to this leg.
The Bartlett sum can drive a variance below zero here for exactly the reason it
can in SPEC.md 5.2 -- Newey & West's non-negativity result is a property of the
*equally weighted* estimator and exponential weights break the rewriting it rests
on -- so the same remedy applies: substitute the uncorrected EWMA variance for
that asset, count it, report it. It invents no constant, and
:class:`~mafrm.risk.covariance.BartlettFallback` is reused rather than
reimplemented so that the two legs cannot report the same event differently.

LEG (c) -- gamma IS PINNED AT 1 ON THIS PANEL, BY RULING, AND THE REASON IS AN
INVERSION RATHER THAN A REGIME
------------------------------------------------------------------------------

This is the one place in SPEC.md 5.5 where a published constant was needed and
was **not available** (CLAUDE.md invariant 9). SPEC.md 5.5(c) specifies
``gamma_n`` qualitatively -- *"a data-quality weight driven by missing
observations and by return-distribution fatness (compare a robust vol estimate to
the raw one)"* -- and names no formula and no constants; ``config/model.yaml`` has
none either.

The route out was to show the weight **saturates**, so that no constant would be
needed. It half worked, and the half that failed is the finding:

* **The missing-observation term saturates**, whenever the panel handed in is
  complete. Every asset then has every observation on every date, and any ramp
  ceiling at or below the window length returns 1 whatever its value.
  :attr:`BlendWeight.observation_term_saturates` computes that live rather than
  assuming it.
* **The fatness term does not.** On this project's panel it reaches ``Z = 7.61``
  at the short horizon. The premise that complete histories saturate it conflates
  two different terms: completeness is a statement about *how many* observations
  there are, fatness about *what they look like*, and a complete history of a
  fat-tailed series is still fat-tailed.

So the constants could not be dissolved, and the ruling that resolves it is
**not** that the term is out of regime. It is that the term is **inverted** here.
``gamma_n < 1`` routes weight off leg (a) and onto leg (b), which is worth doing
when the structural estimate is the better-identified one. At the ``N`` this
model runs it is the worse one -- ``R^2 = 0.833`` on 6 residual degrees of
freedom, the structural estimate missing the time-series estimate by 0.31x to
4.17x -- so the mechanism moves weight the wrong way *even implemented exactly as
published*. The clinching detail is where it fires: March-April 2020, which is
precisely where leg (a) is fat-tailed and therefore precisely where leg (b) is
fitted on the same stressed data. Constants would not have rescued it.

**The code path is kept and both diagnostics are computed live**, with
``gamma_n`` pinned rather than the leg deleted. Same treatment as the two
published eigenfactor scalings: the mechanism stays exercised so that a
larger-``N`` module can turn it on where it belongs, and the prediction that it
*will* belong there is registered in ``experiments.md`` before the panel it
concerns exists.

WHAT ``Z`` IS FOR, GIVEN THAT NOTHING IS ROUTED ON IT
------------------------------------------------------

It is reported, not discarded. ``Z_n`` correctly identifies where the time-series
estimate is unreliable, and that is real information about a panel whether or not
a weight is computed from it. :class:`BlendWeight` carries it and the reports
table it.

LEG (d) -- READ THE FORMULA BACKWARDS FROM THE NAIVE ONE
---------------------------------------------------------

``v_n = |sigma_n - sigma_bar| / (|sigma_n - sigma_bar| + q * sigma_delta)``

The further an estimate sits from its bucket mean **relative to that bucket's own
dispersion**, the **larger** ``v_n`` is and therefore the **less** it is shrunk.
That is backwards from naive shrinkage and it is deliberate: the target is
estimation error rather than genuine heterogeneity, and extreme observed specific
volatilities are disproportionately noise.

**The linear absolute deviation is the correct form and the squared one is a PDF
rendering artefact** (SPEC.md 5.5). ``tests/test_specific_risk.py`` pins the
distinction numerically rather than trusting this paragraph, because the two
forms are both monotone, both land in ``[0, 1]``, and both look right in a chart.

THE DIAGONAL ASSUMPTION IS FALSE AND IS REPORTED RATHER THAN ASSUMED
---------------------------------------------------------------------

SPEC.md 5.5 closes on it: specific returns are taken to be cross-sectionally
uncorrelated, so ``Delta`` is diagonal, and that is false for linked instruments.
:func:`residual_correlation` computes the full matrix and ranks the off-diagonal
pairs so that a report can name the violations rather than assert there are none.
It is the one part of SPEC.md 5.5 that works properly at small ``N``: 13 series
give 78 distinct pairs, which is a real table rather than a degenerate one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Final

import numpy as np
import pandas as pd

from mafrm.numerics import ewma_weights, realised_effective_sample_size
from mafrm.risk.checks import assert_psd
from mafrm.risk.config import RiskConfig
from mafrm.risk.covariance import BartlettFallback, bartlett_weight, lagged_pair_weights
from mafrm.risk.regime import RegimeMultiplier, cross_sectional_bias, regime_multiplier

__all__ = [
    "BlendWeight",
    "BucketStatistics",
    "ResidualCorrelation",
    "ShrinkageResult",
    "SpecificForecastHistory",
    "SpecificRiskBuild",
    "SpecificRiskError",
    "StructuralFit",
    "TimeSeriesEstimate",
    "bayesian_shrinkage",
    "blend_weight",
    "bucket_statistics",
    "build_specific_risk",
    "ewma_variance_diagonal",
    "lagged_autocovariance_diagonal",
    "residual_correlation",
    "robust_deviation",
    "specific_forecast_history",
    "specific_risk",
    "structural_fit",
    "time_series_estimate",
    "weighted_quantile",
]


class SpecificRiskError(ValueError):
    """SPEC.md 5.5's pipeline was handed something it cannot estimate from."""


#: ``sigma = IQR / (2 * Phi^-1(0.75))`` makes the interquartile range a consistent
#: estimator of the standard deviation under normality.
#:
#: **DERIVED, not a parameter** (CLAUDE.md invariants 6 and 9). It is computed
#: here from the normal quantile rather than written down as 1.35, so that it
#: cannot be a transcription of a half-remembered constant -- which is the same
#: hazard the whole of leg (c) exists to avoid. ``statistics.NormalDist`` is in
#: the standard library, so this costs no dependency.
_IQR_TO_DEVIATION: Final[float] = 1.0 / (2.0 * NormalDist().inv_cdf(0.75))

#: ``gamma_n``'s pinned value. See the module docstring: a RULING for this panel,
#: not a computed weight and not a published constant. It lives here as a named
#: constant so that a reader grepping for where the blend is decided finds one
#: place, and so that the value cannot be typed inline twice and drift.
_PINNED_BLEND_WEIGHT: Final[float] = 1.0


# ---------------------------------------------------------------------------
# Leg (a) -- the time-series estimate
# ---------------------------------------------------------------------------


def _check_panel(residuals: np.ndarray, name: str) -> tuple[int, int]:
    """Shape and finiteness. Returns ``(T, N)``."""
    if residuals.ndim != 2:
        raise SpecificRiskError(f"{name}: expected a 2-D T x N array, got {residuals.shape}")
    periods, assets = residuals.shape
    if periods < 1 or assets < 1:
        raise SpecificRiskError(f"{name}: empty input, shape {residuals.shape}")
    if not np.all(np.isfinite(residuals)):
        count = int(np.count_nonzero(~np.isfinite(residuals)))
        raise SpecificRiskError(
            f"{name}: {count} non-finite observation(s). The caller decides how a gap is "
            "filled or dropped -- this module will not guess, because the right answer "
            "depends on what the column is and it is not allowed to know."
        )
    return periods, assets


def ewma_variance_diagonal(residuals: np.ndarray, *, halflife: float) -> np.ndarray:
    """``sum_t w_t * u_nt^2`` per column. The diagonal of the EWMA second moment.

    Exactly ``np.diag(ewma_second_moment(residuals, halflife=halflife))``,
    evaluated in ``O(N T)`` instead of ``O(N^2 T)``, and
    ``tests/test_specific_risk.py`` asserts the two agree to a derived rounding
    bound. **No mean is subtracted**: SPEC.md 5.1.1's zero-mean convention is a
    coherence requirement against SPEC.md 6.1's bias statistic and applies to
    every moment in the project, not only to the factor legs.
    """
    periods, _ = _check_panel(residuals, "ewma_variance_diagonal")
    weights = ewma_weights(periods, halflife)
    moment: np.ndarray = weights @ np.square(residuals)
    return moment


def lagged_autocovariance_diagonal(
    residuals: np.ndarray, *, halflife: float, lag: int
) -> np.ndarray:
    """``sum_t w_t * u_nt * u_n,t-lag`` per column, on the shared pair convention.

    The diagonal of :func:`~mafrm.risk.covariance.ewma_autocovariance`. At
    ``lag = 0`` it **is** :func:`ewma_variance_diagonal` -- the same call, not a
    second path to the same number, for the same structural reason the matrix
    version delegates there.
    """
    if lag == 0:
        return ewma_variance_diagonal(residuals, halflife=halflife)
    periods, _ = _check_panel(residuals, "lagged_autocovariance_diagonal")
    weights = ewma_weights(periods, halflife)
    pair_weights = lagged_pair_weights(weights, lag, periods=periods)
    products = residuals[lag:] * residuals[: periods - lag]
    moment: np.ndarray = pair_weights @ products
    return moment


@dataclass(frozen=True)
class TimeSeriesEstimate:
    """SPEC.md 5.5(a): the EWMA-plus-Newey-West specific volatility, per asset."""

    #: ``sigma_n^TS``, one per asset, in the units of the residuals.
    deviation: np.ndarray
    #: The corrected variance the deviation is the root of.
    variance: np.ndarray
    #: The stage's own ``C_0`` deviation, before the Bartlett terms. Carried
    #: because the correction's size is a reported diagnostic and the operation
    #: that applies it destroys the number.
    uncorrected_deviation: np.ndarray
    #: Row 100's registered remedy, fired or not.
    fallback: BartlettFallback
    volatility_halflife: float
    autocorrelation_halflife: float
    lags: int
    observations: int
    #: Kish effective sample size of the volatility weights over this window.
    realised_effective_sample_size: float

    @property
    def correction_ratio(self) -> np.ndarray:
        """``sigma^TS / sigma^EWMA``. Below one wherever the lag terms are negative."""
        ratio: np.ndarray = self.deviation / self.uncorrected_deviation
        return ratio

    def render(self) -> str:
        ratio = self.correction_ratio
        return (
            f"specific time series: N={self.deviation.size}, T={self.observations:,}, "
            f"vol half-life {self.volatility_halflife:.0f}d, autocorrelation half-life "
            f"{self.autocorrelation_halflife:.0f}d, {self.lags} lag(s), "
            f"Kish T_eff {self.realised_effective_sample_size:.0f}; Newey-West ratio "
            f"{ratio.min():.4f}..{ratio.max():.4f} (median {float(np.median(ratio)):.4f})"
        )


def time_series_estimate(
    residuals: np.ndarray,
    *,
    halflife: float,
    autocorrelation_halflife: float,
    lags: int,
) -> TimeSeriesEstimate:
    """SPEC.md 5.5(a). ``C_0`` at ``halflife``, the lag terms at the AC half-life.

    On the diagonal the Bartlett symmetrisation ``C_d + C_d'`` collapses to
    ``2 * C_d``, which is what is written below; the matrix path forms both
    triangles because it needs the off-diagonals.
    """
    if lags < 0:
        raise SpecificRiskError(f"time_series_estimate: lags must be non-negative, got {lags}")
    periods, _ = _check_panel(residuals, "time_series_estimate")
    if lags >= periods:
        raise SpecificRiskError(
            f"time_series_estimate: {lags} lag(s) need more than {lags} observations, got "
            f"{periods}. The longest lag would have no pairs at all."
        )
    uncorrected = ewma_variance_diagonal(residuals, halflife=halflife)
    corrected = uncorrected.copy()
    for lag in range(1, lags + 1):
        cross = lagged_autocovariance_diagonal(
            residuals, halflife=autocorrelation_halflife, lag=lag
        )
        corrected = corrected + bartlett_weight(lag, lags) * 2.0 * cross

    failed = np.flatnonzero(corrected <= 0.0)
    fallback = BartlettFallback(
        columns=tuple(int(index) for index in failed),
        corrected=tuple(float(corrected[index]) for index in failed),
        substituted=tuple(float(uncorrected[index]) for index in failed),
        size=int(corrected.size),
    )
    if failed.size:
        if np.any(uncorrected[failed] <= 0.0):
            raise SpecificRiskError(
                "time_series_estimate: the Bartlett correction drove asset(s) "
                f"{fallback.columns} non-positive AND the uncorrected EWMA variance the "
                "experiments.md row 100 fallback substitutes is itself non-positive. The "
                "remedy has nothing to fall back TO, which is outside what row 100 "
                "registered: a second moment about zero cannot be non-positive unless the "
                "window is empty or the column is identically zero, so this is a degenerate "
                "input rather than an estimator failure to be repaired here."
            )
        corrected = corrected.copy()
        corrected[failed] = uncorrected[failed]

    weights = ewma_weights(periods, halflife)
    return TimeSeriesEstimate(
        deviation=np.sqrt(corrected),
        variance=corrected,
        uncorrected_deviation=np.sqrt(uncorrected),
        fallback=fallback,
        volatility_halflife=float(halflife),
        autocorrelation_halflife=float(autocorrelation_halflife),
        lags=int(lags),
        observations=int(periods),
        realised_effective_sample_size=realised_effective_sample_size(weights),
    )


# ---------------------------------------------------------------------------
# Leg (b) -- the structural estimate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuralFit:
    """SPEC.md 5.5(b): ``sigma_n^STR = E_0 * exp(sum_k X_nk b_k)``.

    Fitted on the assets the caller marks complete and **applied to all of them**,
    which is the mechanism that gives a forecast to a series too short for leg
    (a). Whether that mechanism has anything to do here is
    :attr:`fitted_on_every_asset`.
    """

    #: ``sigma_n^STR`` for every asset, including any excluded from the fit.
    deviation: np.ndarray
    #: ``b``, with the intercept first. See :attr:`intercept` for why it is there.
    coefficients: np.ndarray
    #: ``E_0``, the multiplicative log-transform bias correction actually applied.
    e_zero: float
    #: ``exp(s^2/2)``, the log-normal closed form. Reported as a cross-check on
    #: :attr:`e_zero` and NOT applied -- it assumes normal log-residuals, which a
    #: handful of points cannot support. See :func:`structural_fit`.
    e_zero_lognormal: float
    #: ``R^2`` of the log-space regression, on the fit set.
    r_squared: float
    #: ``N_fit - (K + 1)``. The number this leg's credibility rests on.
    residual_degrees_of_freedom: int
    #: Boolean mask, one per asset: did this asset enter the fit.
    fit_set: np.ndarray
    #: Always True. Carried as a field rather than assumed, so that the ruling is
    #: visible on the object a report prints.
    intercept: bool

    @property
    def fitted_on_every_asset(self) -> bool:
        """True when nobody was excluded -- i.e. the fit-set threshold is idle."""
        return bool(np.all(self.fit_set))

    def render(self) -> str:
        return (
            f"specific structural: R^2 = {self.r_squared:.4f} on "
            f"{int(np.count_nonzero(self.fit_set))} of {self.fit_set.size} asset(s), "
            f"{self.residual_degrees_of_freedom} residual d.o.f., E_0 = {self.e_zero:.4f} "
            f"(log-normal cross-check {self.e_zero_lognormal:.4f}, not applied)"
        )


def structural_fit(
    time_series_deviation: np.ndarray,
    exposures: np.ndarray,
    *,
    fit_set: np.ndarray | None = None,
) -> StructuralFit:
    """SPEC.md 5.5(b). Regress ``ln sigma^TS`` on the exposures; exponentiate back.

    THE DESIGN MATRIX CARRIES AN INTERCEPT, AND THAT IS A RULING
        SPEC.md 5.5(b) writes ``ln sigma_n = sum_k X_nk b_k`` with no constant
        term. In a Barra model that is complete, because the exposure matrix
        already carries a unit column through its market or country factor. A
        design matrix without one cannot represent a **common level** of specific
        volatility, so the fit is forced through the origin in log space and is
        misspecified regardless of how it scores.

        An intercept is a **design-matrix column, not a parameter** (CLAUDE.md
        invariant 9): it introduces no number anyone chose, and no ``R^2`` could
        reverse it, which is the W3-P1 criterion for this being a ruling rather
        than a sweep. The no-intercept fit is reported as a diagnostic in
        ``reports/specific_risk.md`` rather than carried as a candidate.

    ``E_0`` IS A RATIO OF MEANS, NOT THE LOG-NORMAL CLOSED FORM
        ``E_0 = mean(sigma^TS) / mean(exp(X b))`` over the fit set. It removes the
        Jensen bias by construction -- ``exp(E[ln sigma]) < E[sigma]`` -- and it
        assumes nothing about the shape of the log residuals. The closed form
        ``exp(s^2/2)`` is exact under log-normality and is **computed and
        reported beside it** rather than applied, because at these sample sizes
        log-normality is an assumption the data cannot carry. Both are emitted so
        the gap between them is visible instead of implicit.

        The mean is unweighted. SPEC.md 5.5 asks for a notional weighting and no
        notionals exist before an optimizer does; equal weighting is the absence
        of a weighting rather than a chosen one, and every number derived from it
        says so where it is reported.
    """
    deviation = np.asarray(time_series_deviation, dtype=float)
    design_exposures = np.asarray(exposures, dtype=float)
    if deviation.ndim != 1:
        raise SpecificRiskError(f"structural_fit: expected a 1-D vector, got {deviation.shape}")
    if design_exposures.ndim != 2 or design_exposures.shape[0] != deviation.size:
        raise SpecificRiskError(
            f"structural_fit: exposures must be N x K with N = {deviation.size}, got "
            f"{design_exposures.shape}"
        )
    if np.any(deviation <= 0.0):
        bad = int(np.count_nonzero(deviation <= 0.0))
        raise SpecificRiskError(
            f"structural_fit: {bad} non-positive time-series deviation(s). The regressand is "
            "ln(sigma) and a non-positive volatility has no logarithm; leg (a)'s row 100 "
            "fallback exists to make this unreachable, so arriving here means it was bypassed."
        )
    if not np.all(np.isfinite(design_exposures)):
        raise SpecificRiskError("structural_fit: non-finite exposure(s) in the design matrix")

    mask = (
        np.ones(deviation.size, dtype=bool) if fit_set is None else np.asarray(fit_set, dtype=bool)
    )
    if mask.shape != deviation.shape:
        raise SpecificRiskError(
            f"structural_fit: fit_set has shape {mask.shape}, expected {deviation.shape}"
        )

    design = np.column_stack([np.ones(deviation.size), design_exposures])
    parameters = design.shape[1]
    fitted_count = int(np.count_nonzero(mask))
    dof = fitted_count - parameters
    if dof < 1:
        raise SpecificRiskError(
            f"structural_fit: {fitted_count} asset(s) against {parameters} parameters leaves "
            f"{dof} residual degree(s) of freedom. Below one the fit interpolates, E_0 is "
            "identically 1 by construction rather than by estimation, and the leg reports a "
            "precision it does not have. SPEC.md 5.5(b) is a cross-sectional regression and "
            "it needs a cross-section."
        )

    response = np.log(deviation[mask])
    coefficients, *_ = np.linalg.lstsq(design[mask], response, rcond=None)
    fitted_log = design @ coefficients
    residual = response - design[mask] @ coefficients
    centred = response - response.mean()
    total = float(centred @ centred)
    r_squared = 1.0 - float(residual @ residual) / total if total > 0.0 else float("nan")

    raw = np.exp(fitted_log)
    denominator = float(raw[mask].mean())
    if denominator <= 0.0:  # pragma: no cover - exp() is strictly positive
        raise SpecificRiskError("structural_fit: the fitted level averaged to zero")
    e_zero = float(deviation[mask].mean()) / denominator

    return StructuralFit(
        deviation=e_zero * raw,
        coefficients=coefficients,
        e_zero=e_zero,
        e_zero_lognormal=float(np.exp(0.5 * float(residual @ residual) / residual.size)),
        r_squared=r_squared,
        residual_degrees_of_freedom=int(dof),
        fit_set=mask,
        intercept=True,
    )


# ---------------------------------------------------------------------------
# Leg (c) -- the data-quality blend
# ---------------------------------------------------------------------------


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> np.ndarray:
    """Column-wise weighted quantile that REDUCES EXACTLY to the ordinary one.

    The plotting position of the ``i``-th sorted value is
    ``(C_i - w_i) / (C_last - w_i)`` with ``C`` the cumulative weight, and the
    quantile is linearly interpolated between them.

    **That form is chosen because it is the one that generalises rather than
    approximates.** At equal weights it collapses to ``i / (n - 1)``, which is
    numpy's own default position, so this function *is* ``np.quantile`` wherever
    the weights are flat -- pinned in ``tests/test_specific_risk.py`` against
    ``np.quantile`` itself rather than against a transcription of it. The
    midpoint alternative ``(C_i - w_i/2) / C_last`` is equally defensible as a
    definition and does **not** have that property: it lands at
    ``(i + 1/2) / n``, so a reader comparing this estimator against an ordinary
    quantile on a flat window would find a discrepancy that is a convention and
    looks like a bug.

    **No parameter is introduced by either form.** The choice is between two
    definitions, and the one that agrees with the unweighted estimator is the one
    that can be checked against something.
    """
    if not 0.0 <= quantile <= 1.0:
        raise SpecificRiskError(f"weighted_quantile: quantile must be in [0, 1], got {quantile}")
    periods, _ = _check_panel(values, "weighted_quantile")
    if weights.shape != (periods,):
        raise SpecificRiskError(
            f"weighted_quantile: weights must have shape ({periods},), got {weights.shape}"
        )
    order = np.argsort(values, axis=0, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=0)
    sorted_weights = np.take_along_axis(
        np.broadcast_to(weights[:, None], values.shape), order, axis=0
    )
    cumulative = np.cumsum(sorted_weights, axis=0)
    total = cumulative[-1]
    if np.any(total <= 0.0):
        raise SpecificRiskError("weighted_quantile: the weights carry no mass")
    span = total - sorted_weights
    if np.any(span <= 0.0):
        raise SpecificRiskError(
            "weighted_quantile: one observation carries the entire weight, so there is no "
            "interval between order statistics to interpolate a quantile across."
        )
    positions = (cumulative - sorted_weights) / span
    return np.array(
        [
            float(np.interp(quantile, positions[:, column], sorted_values[:, column]))
            for column in range(values.shape[1])
        ]
    )


def robust_deviation(residuals: np.ndarray, *, halflife: float) -> np.ndarray:
    """``IQR / (2 Phi^-1(0.75))`` on the same exponential weights as leg (a).

    The robust half of SPEC.md 5.5(c)'s *"compare a robust vol estimate to the raw
    one"*. Both halves run on the **same window with the same weights**, so the
    comparison measures distribution shape alone rather than a window mismatch --
    which it would if a robust estimate on a trailing box window were set against
    an exponentially weighted second moment.
    """
    periods, _ = _check_panel(residuals, "robust_deviation")
    weights = ewma_weights(periods, halflife)
    upper = weighted_quantile(residuals, weights, 0.75)
    lower = weighted_quantile(residuals, weights, 0.25)
    spread: np.ndarray = (upper - lower) * _IQR_TO_DEVIATION
    if np.any(spread <= 0.0):
        bad = int(np.count_nonzero(spread <= 0.0))
        raise SpecificRiskError(
            f"robust_deviation: {bad} asset(s) with a zero interquartile range. More than half "
            "the weighted observations are identical, which is a degenerate column rather "
            "than a fat-tailed one."
        )
    return spread


@dataclass(frozen=True)
class BlendWeight:
    """SPEC.md 5.5(c)'s ``gamma_n``, and the two diagnostics that were meant to fix it.

    ``gamma`` is **pinned**, and :attr:`pinned` says so on the object rather than
    only in a docstring, so that a report cannot present it as a computed weight.
    The module docstring carries the ruling and its reasoning.
    """

    #: ``gamma_n``, one per asset. Every entry is :data:`_PINNED_BLEND_WEIGHT`.
    gamma: np.ndarray
    #: ``h_n``: how many observations each asset actually has in the window.
    observation_counts: np.ndarray
    #: The window length. ``h_n == periods`` for every asset means no gaps.
    periods: int
    #: ``sigma^TS / sigma^robust``, per asset.
    robustness_ratio: np.ndarray
    #: ``Z_n = |ratio - 1|``. Reported as a flag; nothing is routed on it.
    z: np.ndarray
    pinned: bool

    @property
    def minimum_observations(self) -> int:
        """``min_n h_n``. Any ramp ceiling at or below this saturates term one."""
        return int(self.observation_counts.min())

    @property
    def observation_term_saturates(self) -> bool:
        """True when every asset has every observation, so term one returns 1.

        This is the half of the saturation demonstration that **worked**. It is
        computed rather than assumed because a later panel with a ragged edge
        would make it false, and the ruling pinning ``gamma`` is scoped to a
        panel where it is true.
        """
        return bool(np.all(self.observation_counts == self.periods))

    @property
    def maximum_z(self) -> float:
        """The largest fatness statistic. The half of the demonstration that failed."""
        return float(self.z.max())

    def fatness_term_saturates(self, bound: float) -> bool:
        """Would term two return 1 for a kernel that saturates at ``Z <= bound``?

        A method taking the bound rather than a property against a fixed one,
        because **there is no published bound** and inventing one to test against
        would be the invariant-9 violation this whole leg was routed around. A
        caller states the bound it is interested in and gets an answer about that
        bound.
        """
        return bool(np.all(self.z <= bound))

    def render(self) -> str:
        return (
            f"specific blend: gamma PINNED at {float(self.gamma[0]):.1f} for all "
            f"{self.gamma.size} asset(s) (SPEC.md 5.5 ruling, not a computed weight); "
            f"observation term saturates: {self.observation_term_saturates} "
            f"(min h_n = {self.minimum_observations} of {self.periods}); "
            f"max Z = {self.maximum_z:.4f}"
        )


def blend_weight(
    residuals: np.ndarray, time_series_deviation: np.ndarray, *, halflife: float
) -> BlendWeight:
    """SPEC.md 5.5(c). Computes both diagnostics live and pins ``gamma`` at 1.

    See the module docstring for the ruling. What matters at the call site is
    that this function is **not** a stub: it does the measurement SPEC.md 5.5(c)
    describes, and it is the measurement rather than the weight that this project
    can honestly produce.
    """
    periods, assets = _check_panel(residuals, "blend_weight")
    deviation = np.asarray(time_series_deviation, dtype=float)
    if deviation.shape != (assets,):
        raise SpecificRiskError(
            f"blend_weight: expected {assets} time-series deviation(s), got {deviation.shape}"
        )
    robust = robust_deviation(residuals, halflife=halflife)
    ratio = deviation / robust
    return BlendWeight(
        gamma=np.full(assets, _PINNED_BLEND_WEIGHT),
        # Counted rather than assumed: a caller handing in a masked or padded
        # panel would be caught here, and _check_panel has already refused NaN.
        observation_counts=np.full(assets, periods, dtype=int),
        periods=int(periods),
        robustness_ratio=ratio,
        z=np.abs(ratio - 1.0),
        pinned=True,
    )


# ---------------------------------------------------------------------------
# Leg (d) -- Bayesian shrinkage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BucketStatistics:
    """``sigma_bar(s_n)`` and ``sigma_delta(s_n)``, broadcast back to the assets."""

    #: The label each asset carries, in asset order.
    labels: tuple[str, ...]
    #: Distinct labels, sorted, so a report iterates deterministically.
    distinct: tuple[str, ...]
    #: ``sigma_bar`` for each asset's own bucket.
    mean: np.ndarray
    #: ``sigma_delta`` for each asset's own bucket.
    dispersion: np.ndarray
    #: How many assets share each asset's bucket.
    size: np.ndarray
    #: How many of those CONTRIBUTED to the target. Equal to :attr:`size` unless
    #: the caller passed a ``target_set``. See :func:`bucket_statistics`.
    target_size: np.ndarray
    #: Boolean mask, one per asset: did this asset contribute to its bucket's
    #: target. Carried rather than recomputed so a report can say which values
    #: were kept out of a mean they are still shrunk toward.
    target_set: np.ndarray

    @property
    def singletons(self) -> tuple[str, ...]:
        """Buckets whose TARGET has one contributor. Stage (d) is the identity here.

        Keyed on the target rather than on membership, because that is what
        governs the arithmetic: a bucket of two whose target is computed from one
        of them behaves exactly like a bucket of one, and reporting it as a
        two-member bucket would hide a no-op.
        """
        labels = np.asarray(self.labels)
        return tuple(
            label
            for label in self.distinct
            if int(np.count_nonzero((labels == label) & self.target_set)) == 1
        )

    @property
    def targetless(self) -> tuple[str, ...]:
        """Buckets with NO contributor left. Stage (d) is the identity there too."""
        labels = np.asarray(self.labels)
        return tuple(
            label for label in self.distinct if not np.any((labels == label) & self.target_set)
        )

    def counts(self) -> dict[str, int]:
        """Members per bucket, for a report table."""
        labels = np.asarray(self.labels)
        return {label: int(np.count_nonzero(labels == label)) for label in self.distinct}

    def target_counts(self) -> dict[str, int]:
        """Target contributors per bucket. Differs from :meth:`counts` under exclusion."""
        labels = np.asarray(self.labels)
        return {
            label: int(np.count_nonzero((labels == label) & self.target_set))
            for label in self.distinct
        }


def bucket_statistics(
    deviation: np.ndarray, buckets: Sequence[str], *, target_set: np.ndarray | None = None
) -> BucketStatistics:
    """Group by label equality. The labels are opaque; nothing here reads them.

    ``sigma_delta`` IS THE POPULATION DEVIATION, ddof = 0
        A bucket is the whole of itself rather than a sample drawn from a larger
        one, so there is no degree of freedom spent estimating its mean from
        outside it. At two members the two conventions differ by ``sqrt(2)``,
        which is large enough that the choice has to be stated: this is a ruling
        on what the object is, and the ``ddof = 1`` alternative was not built.

    ``target_set`` -- WHICH VALUES MAY BUILD THE TARGET, W4-P1b
        Every asset is *assigned* a bucket and every asset receives a shrunk
        value. ``target_set`` decides, separately, which values are allowed to
        **contribute** to ``sigma_bar`` and ``sigma_delta``. Default: all of them.

        The distinction exists because a shrinkage target is an estimate of where
        a bucket's true specific volatilities sit, and a value that is a
        **construction artefact rather than an estimate** has no business in it.
        Excluding such a value from a later average does nothing about the fact
        that it has already propagated into every other member's shrunk estimate
        through the mean -- excluding a corrupted value from an average does not
        uncorrupt the values it corrupted.

        This module cannot know which values are artefacts, so the caller says
        (CLAUDE.md invariant 10). What it does guarantee is that an excluded
        asset is still bucketed, still shrunk and still reported: the exclusion
        removes a manufactured number from a target and substitutes no estimate
        for it.

        A bucket left with **no** contributor is handled rather than refused: its
        members take their own value as the target and ``sigma_delta`` is zero, so
        stage (d) is the identity for them. :attr:`BucketStatistics.targetless`
        names those buckets so a report can say so.
    """
    values = np.asarray(deviation, dtype=float)
    labels = tuple(str(label) for label in buckets)
    if values.ndim != 1 or values.size != len(labels):
        raise SpecificRiskError(
            f"bucket_statistics: {values.size} deviation(s) against {len(labels)} bucket label(s)"
        )
    if values.size == 0:
        raise SpecificRiskError("bucket_statistics: empty input")
    mask = np.ones(values.size, dtype=bool) if target_set is None else np.asarray(target_set, bool)
    if mask.shape != values.shape:
        raise SpecificRiskError(
            f"bucket_statistics: target_set has shape {mask.shape}, expected {values.shape}"
        )
    array = np.asarray(labels)
    mean = np.empty(values.size)
    dispersion = np.empty(values.size)
    size = np.empty(values.size, dtype=int)
    target_size = np.empty(values.size, dtype=int)
    for label in set(labels):
        members = array == label
        contributors = members & mask
        size[members] = int(np.count_nonzero(members))
        target_size[members] = int(np.count_nonzero(contributors))
        if not np.any(contributors):
            # No target survives. Each member is its own target, so stage (d) is
            # the identity -- see the docstring. Not an error: a caller may
            # legitimately exclude every member of a bucket.
            mean[members] = values[members]
            dispersion[members] = 0.0
            continue
        block = values[contributors]
        mean[members] = block.mean()
        dispersion[members] = block.std(ddof=0)
    return BucketStatistics(
        labels=labels,
        distinct=tuple(sorted(set(labels))),
        mean=mean,
        dispersion=dispersion,
        target_set=mask,
        size=size,
        target_size=target_size,
    )


@dataclass(frozen=True)
class ShrinkageResult:
    """SPEC.md 5.5(d): ``sigma^SH = v sigma_hat + (1 - v) sigma_bar``."""

    #: ``sigma_n^SH``, the shrunk deviation.
    deviation: np.ndarray
    #: ``v_n``. Larger means LESS shrinkage -- see the module docstring.
    v: np.ndarray
    buckets: BucketStatistics
    q: float
    #: Assets whose ``v`` was undetermined because their bucket's target has one
    #: contributor, where the map is the identity for every ``v`` and the value
    #: filled in is arbitrary.
    identity_positions: tuple[int, ...]

    def render(self) -> str:
        singletons = self.buckets.singletons
        excluded = int(np.count_nonzero(~self.buckets.target_set))
        return (
            f"specific shrinkage: q = {self.q}, v in "
            f"[{self.v.min():.4f}, {self.v.max():.4f}] over "
            f"{len(self.buckets.distinct)} bucket(s)"
            + (f", {excluded} asset(s) held out of the target" if excluded else "")
            + (
                f"; {len(singletons)} bucket(s) {list(singletons)} have a "
                "one-contributor target and are a NO-OP"
                if singletons
                else ""
            )
        )


def bayesian_shrinkage(
    deviation: np.ndarray,
    buckets: Sequence[str],
    *,
    q: float,
    target_set: np.ndarray | None = None,
) -> ShrinkageResult:
    """SPEC.md 5.5(d). Linear absolute deviation -- NOT the squared rendering.

    ``v_n = |sigma_n - sigma_bar| / (|sigma_n - sigma_bar| + q * sigma_delta)``

    ``target_set`` restricts which values may build ``sigma_bar`` and
    ``sigma_delta``; every asset is still shrunk and still returned. See
    :func:`bucket_statistics` for what it is for and why the caller owns it.

    A ONE-CONTRIBUTOR TARGET IS A NO-OP BY CONSTRUCTION, NOT BY A SPECIAL CASE
        With one contributor ``sigma_bar = sigma_hat`` exactly, so the numerator and
        ``sigma_delta`` are both zero and ``v`` is ``0/0``. The indeterminacy is
        **removable**: ``v sigma_hat + (1 - v) sigma_bar = sigma_hat`` for every
        ``v`` when the two agree, so the shrunk value is the unshrunk one whatever
        is filled in. ``v = 1`` is written -- the value that says "not shrunk",
        which is what happened -- the positions are recorded on
        :attr:`ShrinkageResult.identity_positions` so a report can flag them, and
        ``tests/test_specific_risk.py`` asserts the identity rather than the fill.

        This is not a degenerate case to be tolerated. It is what SPEC.md 5.5(d)
        *does* to a bucket with nothing to shrink toward, and a model whose
        universe has such a bucket should say so in its report.
    """
    values = np.asarray(deviation, dtype=float)
    if np.any(values <= 0.0):
        bad = int(np.count_nonzero(values <= 0.0))
        raise SpecificRiskError(f"bayesian_shrinkage: {bad} non-positive deviation(s)")
    if q < 0.0:
        raise SpecificRiskError(f"bayesian_shrinkage: q must be non-negative, got {q}")
    statistics = bucket_statistics(values, buckets, target_set=target_set)
    distance = np.abs(values - statistics.mean)
    denominator = distance + q * statistics.dispersion
    degenerate = denominator <= 0.0
    v = np.where(degenerate, 1.0, distance / np.where(degenerate, 1.0, denominator))
    return ShrinkageResult(
        deviation=v * values + (1.0 - v) * statistics.mean,
        v=v,
        buckets=statistics,
        q=float(q),
        identity_positions=tuple(int(index) for index in np.flatnonzero(degenerate)),
    )


# ---------------------------------------------------------------------------
# The diagonal assumption, reported rather than asserted
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualCorrelation:
    """The cross-asset correlation of specific returns. SPEC.md 5.5's closing note.

    ``Delta`` is diagonal by assumption and this object is the evidence about
    whether it should be. It is computed **about zero**, matching the convention
    every other moment in the project is taken on (SPEC.md 5.1.1), because the
    quantity ``F = X F X' + Delta`` assumes away is the second moment of the
    specific returns rather than their covariance about a drifting sample mean.
    """

    matrix: np.ndarray
    columns: tuple[str, ...]
    #: Number of distinct off-diagonal pairs, ``N (N - 1) / 2``.
    pair_count: int
    #: Largest ``|rho|`` off the diagonal.
    maximum_absolute: float
    #: The largest single difference between this matrix and the demeaned Pearson
    #: correlation of the same panel. Reported so the convention's cost is a
    #: number rather than a claim.
    mean_convention_difference: float

    def pairs(self, *, limit: int | None = None) -> list[tuple[str, str, float]]:
        """Every distinct off-diagonal pair, largest ``|rho|`` first."""
        size = len(self.columns)
        out = [
            (self.columns[i], self.columns[j], float(self.matrix[i, j]))
            for i in range(size)
            for j in range(i + 1, size)
        ]
        out.sort(key=lambda item: abs(item[2]), reverse=True)
        return out if limit is None else out[:limit]

    def violations(self, threshold: float) -> list[tuple[str, str, float]]:
        """Pairs whose ``|rho|`` exceeds ``threshold``. The caller states it."""
        return [pair for pair in self.pairs() if abs(pair[2]) > threshold]


def residual_correlation(residuals: pd.DataFrame, *, config: RiskConfig) -> ResidualCorrelation:
    """The specific-return correlation matrix, asserted PSD. SPEC.md 5.5.

    Equally weighted over the whole window handed in: this is a description of
    the panel rather than a forecast, so it carries no half-life and no decay.
    """
    if config.mean_convention != "zero":  # pragma: no cover - the Literal has one member
        raise SpecificRiskError(
            f"residual_correlation: mean_convention is {config.mean_convention!r}. SPEC.md "
            "5.1.1 takes every moment about zero and this matrix has to match, or it "
            "describes a different object from the one the diagonal assumption is about."
        )
    values = residuals.to_numpy(dtype=float)
    _check_panel(values, "residual_correlation")
    moment = values.T @ values / values.shape[0]
    scale = np.sqrt(np.diag(moment))
    if np.any(scale <= 0.0):
        raise SpecificRiskError("residual_correlation: an identically zero residual column")
    matrix = moment / np.outer(scale, scale)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)
    assert_psd(matrix, "specific.residual_correlation", expected_size=values.shape[1])
    pearson = residuals.corr().to_numpy(dtype=float)
    size = values.shape[1]
    return ResidualCorrelation(
        matrix=matrix,
        columns=tuple(str(column) for column in residuals.columns),
        pair_count=size * (size - 1) // 2,
        maximum_absolute=float(np.max(np.abs(matrix - np.eye(size)))),
        mean_convention_difference=float(np.max(np.abs(matrix - pearson))),
    )


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecificRiskBuild:
    """Everything one SPEC.md 5.5 run produced, at one horizon.

    :attr:`forecast` refuses until the regime multiplier has been applied, for
    the same reason :attr:`~mafrm.risk.covariance.CovarianceBuild.forecast` does:
    an un-adjusted specific volatility is positive, finite, correctly shaped and
    simply too small, which is a failure nothing downstream could detect.
    """

    columns: tuple[str, ...]
    horizon: str
    time_series: TimeSeriesEstimate
    structural: StructuralFit
    blend: BlendWeight
    shrinkage: ShrinkageResult
    #: ``gamma sigma^TS + (1 - gamma) sigma^STR``, before shrinkage.
    blended: np.ndarray
    #: SPEC.md 5.4's specific leg, or ``None`` when no history was supplied.
    regime: RegimeMultiplier | None

    @property
    def assets(self) -> int:
        return len(self.columns)

    @property
    def forecast(self) -> np.ndarray:
        """The finished specific volatilities. **Raises unless the VRA ran.**"""
        if self.regime is None:
            raise SpecificRiskError(
                "SpecificRiskBuild.forecast: SPEC.md 5.4's specific volatility regime "
                "multiplier has not been applied, so these deviations must not be published "
                "as a forecast. The stage standardises each specific return by the forecast "
                "made at t-1, so it consumes a history rather than a window: build one with "
                "specific_forecast_history and pass regime_bias= to build_specific_risk. Use "
                ".shrinkage.deviation for the pre-VRA estimate and say at the call site why "
                "an unadjusted number is the right one there."
            )
        return self.regime.lambda_ * self.shrinkage.deviation

    @property
    def series(self) -> pd.Series:
        """:attr:`forecast` with the input's labels. Raises on the same condition."""
        return pd.Series(self.forecast, index=list(self.columns))

    def _regime_line(self) -> str:
        """SPEC.md 5.4's multiplier, relabelled for the SPECIFIC leg.

        :meth:`~mafrm.risk.regime.RegimeMultiplier.render` writes ``lambda_F`` and
        ``B_t^F`` because it was written for the factor leg, and the same class is
        reused here rather than copied. Reusing the estimator and relabelling the
        line is the right way round: two multipliers that print differently but
        compute identically is the situation SPEC.md 5.4 wants, and two that
        compute differently because someone forked the class to fix a label is
        the one it does not.
        """
        regime = self.regime
        if regime is None:  # pragma: no cover - guarded by the caller
            raise SpecificRiskError("SpecificRiskBuild._regime_line: no multiplier")
        return (
            f"specific regime: lambda_S^2 = {regime.lambda_squared:.6f} "
            f"(lambda_S = {regime.lambda_:.6f}, {100.0 * (regime.lambda_ - 1.0):+.2f}% on vol) "
            f"from {regime.observations:,} forecast dates at half-life {regime.halflife}d, "
            f"Kish T_eff {regime.realised_effective_sample_size:.0f}, "
            f"max B_t^S {regime.maximum_bias:.3f}, no clipping, EQUAL-weighted"
        )

    def render(self) -> tuple[str, ...]:
        lines = [
            f"specific risk [{self.horizon}] N={self.assets}, "
            f"T={self.time_series.observations:,} obs",
            f"  {self.time_series.render()}",
            f"  {self.structural.render()}",
            f"  {self.blend.render()}",
            f"  {self.shrinkage.render()}",
        ]
        if self.time_series.fallback.fired:
            lines.append(f"  {self.time_series.fallback.render()}")
        if self.regime is None:
            lines.append(
                "  volatility regime: NOT APPLIED -- this build is PRE-VRA and "
                ".forecast will refuse"
            )
        else:
            lines.append(f"  {self._regime_line()}")
        return tuple(lines)


def build_specific_risk(
    residuals: pd.DataFrame,
    exposures: pd.DataFrame,
    buckets: pd.Series,
    config: RiskConfig,
    *,
    regime_bias: np.ndarray | None = None,
    shrinkage_target: Sequence[str] | None = None,
) -> SpecificRiskBuild:
    """SPEC.md 5.5's four legs plus SPEC.md 5.4's specific multiplier.

    ``residuals`` is ``T x N`` oldest row first, ``exposures`` is ``N x K``
    aligned to the residual columns, ``buckets`` is one opaque label per asset.
    ``regime_bias`` is SPEC.md 5.4's ``B_t^S`` history and is **supplied by the
    caller**, exactly as it is for the factor leg: it is a history of forecasts
    rather than a window of returns, so building it means running this function
    once per date.

    ``shrinkage_target`` names the columns whose values may build stage (d)'s
    bucket mean and dispersion. Default: all of them. Every asset is shrunk and
    returned either way; see :func:`bucket_statistics` for what the restriction
    is for and why this module cannot decide it.
    """
    columns = tuple(str(column) for column in residuals.columns)
    if list(exposures.index) != list(residuals.columns):
        raise SpecificRiskError(
            "build_specific_risk: the exposure rows are not the residual columns. "
            f"residuals {list(residuals.columns)}, exposures {list(exposures.index)}"
        )
    if list(buckets.index) != list(residuals.columns):
        raise SpecificRiskError(
            "build_specific_risk: the bucket labels are not indexed by the residual columns. "
            f"residuals {list(residuals.columns)}, buckets {list(buckets.index)}"
        )
    values = residuals.to_numpy(dtype=float)

    series = time_series_estimate(
        values,
        halflife=float(config.specific_volatility_halflife),
        autocorrelation_halflife=float(config.specific_autocorrelation_halflife),
        lags=config.specific_newey_west_lags,
    )
    structural = structural_fit(series.deviation, exposures.to_numpy(dtype=float))
    blend = blend_weight(
        values, series.deviation, halflife=float(config.specific_volatility_halflife)
    )
    blended = blend.gamma * series.deviation + (1.0 - blend.gamma) * structural.deviation
    if shrinkage_target is None:
        target_mask = None
    else:
        chosen = {str(name) for name in shrinkage_target}
        unknown = sorted(chosen - set(columns))
        if unknown:
            raise SpecificRiskError(
                f"build_specific_risk: shrinkage_target names {unknown}, which are not "
                f"residual columns; got {list(columns)}"
            )
        target_mask = np.array([column in chosen for column in columns], dtype=bool)
    shrinkage = bayesian_shrinkage(
        blended,
        [str(label) for label in buckets],
        q=config.bayesian_shrinkage_q,
        target_set=target_mask,
    )
    regime = (
        None
        if regime_bias is None
        else regime_multiplier(regime_bias, halflife=config.volatility_regime_halflife)
    )
    return SpecificRiskBuild(
        columns=columns,
        horizon=config.horizon,
        time_series=series,
        structural=structural,
        blend=blend,
        shrinkage=shrinkage,
        blended=blended,
        regime=regime,
    )


def specific_risk(
    residuals: pd.DataFrame,
    exposures: pd.DataFrame,
    buckets: pd.Series,
    config: RiskConfig,
    *,
    regime_bias: np.ndarray | None = None,
    shrinkage_target: Sequence[str] | None = None,
) -> pd.Series:
    """SPEC.md 15.2's signature, verbatim. Returns ``sigma_n`` labelled by asset.

    With ``regime_bias`` this is the finished forecast; without it, this function
    raises rather than returning the pre-VRA estimate -- see
    :attr:`SpecificRiskBuild.forecast`. A caller that genuinely wants the
    unadjusted number calls :func:`build_specific_risk` and says so.
    """
    return build_specific_risk(
        residuals,
        exposures,
        buckets,
        config,
        regime_bias=regime_bias,
        shrinkage_target=shrinkage_target,
    ).series


# ---------------------------------------------------------------------------
# The forecast history -- the caller's half of SPEC.md 5.4's specific leg
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecificForecastHistory:
    """``B_t^S``, and the forecasts and realisations it was built from.

    THE CROSS-SECTIONAL MEAN IS EQUALLY WEIGHTED, AND THAT IS A RULING
        SPEC.md 5.4's second paragraph asks for a *notional*-weighted
        cross-sectional bias statistic. **No notionals exist**: no portfolio has
        been constructed at this point in the project and
        ``config/universe.yaml`` is right not to invent any. Equal weighting is
        the absence of a weighting rather than a chosen one, every number derived
        from it is labelled equal-weighted where it is reported, and a later
        session with real holdings that revisits this owes a ``model-config`` row
        at the moment of switching rather than at the end of the project.
    """

    #: ``B_t^S``, oldest first.
    bias: np.ndarray
    #: The dates ``bias`` is indexed by -- each is the date of the REALISED
    #: return, standardised by the forecast made the trading day before.
    index: pd.DatetimeIndex
    #: Which assets entered the cross-section. A caller may exclude some; see
    #: :func:`specific_forecast_history`.
    included: tuple[str, ...]
    #: Which assets were excluded, and therefore forecast but not scored.
    excluded: tuple[str, ...]
    #: The per-asset forecasts, one row per date in :attr:`index`.
    deviations: pd.DataFrame
    #: The window each forecast was made on had at least this many observations.
    minimum_observations: int

    @property
    def observations(self) -> int:
        return int(self.bias.size)

    def multiplier(self, *, halflife: int) -> RegimeMultiplier:
        """``lambda_S^2`` at ``tau_VRA``. Same estimator as the factor leg."""
        return regime_multiplier(self.bias, halflife=halflife)

    @property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"bias": self.bias}, index=self.index)


def specific_forecast_history(
    residuals: pd.DataFrame,
    exposures: pd.DataFrame,
    buckets: pd.Series,
    config: RiskConfig,
    *,
    minimum_observations: int,
    include: Sequence[str] | None = None,
    shrinkage_target: Sequence[str] | None = None,
) -> SpecificForecastHistory:
    """``B_t^S = sqrt(mean_n (u_nt / sigma_nt)^2)``, ``sigma_nt`` made at ``t-1``.

    THE LEAD-LAG, ASSERTED RATHER THAN DOCUMENTED
        The forecast for the return on row ``t`` is built from rows ``0..t-1``
        inclusive and **row t is never in its own window**. That is enforced by
        the slice below rather than by care, and pinned by a leak test in
        ``tests/test_specific_risk.py`` that perturbs one realisation and
        requires every forecast to be unchanged.

    ``include`` NAMES THE CROSS-SECTION, AND THE CALLER OWNS THE CHOICE
        A bias statistic is ``sd(realised / forecast)``, and an asset whose
        residual is zero by construction contributes a ratio pinned at one --
        pulling the aggregate toward one and making the model look better
        calibrated than it is. Which assets those are is a fact about how the
        factor set was built, which this module is not allowed to know
        (CLAUDE.md invariant 10), so the caller passes the cross-section it
        wants scored and the excluded assets are still **forecast and reported**,
        just not averaged.

    ``minimum_observations`` IS THE CALLER'S BURN-IN
        SPEC.md 5.1.2 leaves it there, and there is no published value; it is
        recorded on the returned object so that every number derived from the
        history carries the window it started at.
    """
    if minimum_observations < 1:
        raise SpecificRiskError(
            f"specific_forecast_history: minimum_observations must be at least 1, got "
            f"{minimum_observations}"
        )
    columns = [str(column) for column in residuals.columns]
    chosen = columns if include is None else [str(name) for name in include]
    unknown = [name for name in chosen if name not in columns]
    if unknown:
        raise SpecificRiskError(
            f"specific_forecast_history: {unknown} are not residual columns; got {columns}"
        )
    if not chosen:
        raise SpecificRiskError(
            "specific_forecast_history: an empty cross-section. SPEC.md 5.4's statistic is a "
            "mean over assets and there is nothing to average."
        )
    periods = residuals.shape[0]
    if periods <= minimum_observations:
        raise SpecificRiskError(
            f"specific_forecast_history: {periods} observation(s) against a burn-in of "
            f"{minimum_observations} leaves no date to forecast"
        )

    rows: list[np.ndarray] = []
    dates: list[pd.Timestamp] = []
    for position in range(minimum_observations, periods):
        window = residuals.iloc[:position]
        build = build_specific_risk(
            window, exposures, buckets, config, shrinkage_target=shrinkage_target
        )
        rows.append(build.shrinkage.deviation)
        dates.append(pd.Timestamp(residuals.index[position]))

    index = pd.DatetimeIndex(dates)
    deviations = pd.DataFrame(np.vstack(rows), index=index, columns=columns)
    realised = residuals.loc[index, chosen].to_numpy(dtype=float)
    bias = cross_sectional_bias(realised, deviations[chosen].to_numpy(dtype=float))
    return SpecificForecastHistory(
        bias=bias,
        index=index,
        included=tuple(chosen),
        excluded=tuple(name for name in columns if name not in set(chosen)),
        deviations=deviations,
        minimum_observations=int(minimum_observations),
    )
