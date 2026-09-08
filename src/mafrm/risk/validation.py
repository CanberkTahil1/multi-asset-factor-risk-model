"""SPEC.md 6.1's bias statistics and SPEC.md 6.2's four portfolio families.

SPEC.md 6 calls itself the centrepiece and SPEC.md 6.2 the intellectual core.
The claim both rest on is one sentence: **validating only on random or benchmark
portfolios will tell you your model is fine when it is not.** Everything here
exists to put a number on that sentence, on this project's own data, and the
number is the gap between family 2 and family 4 evaluated on the *same*
covariance matrix.

WHAT THE STATISTIC IS
---------------------

::

    b_nt = R_nt / sigma_nt      sigma_nt made at t-1 -- out of sample BY CONSTRUCTION
    B_n  = sd( clip(b_nt, -4, +4), ddof=1 )
    MRAD = mean_n | B_n - 1 |

``B > 1`` is under-forecasting, ``B < 1`` over-forecasting. The clip is SPEC.md
6.1's and belongs to *this* statistic only: SPEC.md 5.4's cross-sectional
``B_t^F`` is deliberately unclipped (SPEC.md 5.4.1 ruling 2), because it is an
exponentially weighted mean of ``B^2`` over hundreds of effective observations
rather than a standard deviation over twelve.

THE LEAD-LAG IS THE WHOLE THING, SO IT IS ASSERTED AND NOT DOCUMENTED
---------------------------------------------------------------------

If a contemporaneous forecast leaks in, ``b_nt`` is standardised by a number
that saw its own realisation, ``B`` is pulled toward one, and **every result
downstream is wrong in the direction of the project's own headline claim** --
which is the direction no reader can detect from the output. There are three
places it could leak here and all three are closed by construction rather than
by care:

* the **factor covariance** comes from an expanding window ending strictly
  before ``t`` (:func:`mafrm.risk.regime.forecast_history` asserts its own
  position);
* the **specific variances** likewise
  (:func:`mafrm.risk.specific.specific_forecast_history`);
* the **exposures**, which is the one that is easy to miss. A rolling beta at
  date ``t`` is estimated on a window *ending at* ``t``, so it has seen the
  return it would be used to forecast. :class:`RiskForecast` therefore takes the
  exposure row the caller says was known at ``t-1``, and
  :func:`portfolio_families` refuses to run unless the caller has lagged it.

``tests/test_validation.py`` perturbs the realised return on one date and
asserts that not one forecast anywhere moves.

WHICH INTERVAL DECIDES -- SEE SPEC.md 6.1.1
--------------------------------------------

SPEC.md 6.1 writes the band as ``1 +- 1.96/sqrt(T)``. That is a normal
approximation, and the constant is arguable: the sampling standard deviation of
a standard deviation is ``sigma/sqrt(2(T-1))``, so a ``sqrt(2T)`` denominator is
equally defensible and the two differ by a factor of 1.41. Under the null,
``(T-1) B^2 / 1`` is **exactly** ``chi^2_{T-1}``, so :func:`chi_square_interval`
is exact and it is the interval every claim rests on.
:func:`normal_band` is drawn on the chart because SPEC.md 6.1 asks for it, and
it is labelled a display convention there.

CLAUDE.md FAILURE MODE 9 IS EVERYWHERE IN THIS FILE
----------------------------------------------------

The rolling statistic is the deliverable and it is built from overlapping
windows: at a 252-day window stepped one day, consecutive readings share 251 of
252 observations. So :class:`RollingBias` carries its own ``overlap`` and window
length, no function here returns a window count that could be mistaken for an
``n``, and :class:`BiasInterval` is only ever constructed from the number of
observations *inside one window*.

WHAT THIS MODULE MAY KNOW
--------------------------

Arrays, a :class:`~mafrm.risk.config.RiskConfig`, and labels it never
interprets. CLAUDE.md invariant 10: nothing here branches on what an asset is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy.stats import chi2

from mafrm.risk.checks import assert_psd

__all__ = [
    "BiasInterval",
    "MemberBias",
    "PortfolioSet",
    "RiskForecast",
    "RollingBias",
    "ValidationError",
    "ValidationReport",
    "assert_forecast_psd",
    "asset_covariance",
    "bias_statistic",
    "chi_square_interval",
    "eigen_portfolios",
    "mean_absolute_deviation",
    "minimum_variance_weights",
    "normal_band",
    "portfolio_families",
    "random_dollar_neutral",
    "rolling_bias_statistic",
    "standardized_returns",
    "validate",
]

#: SPEC.md 6.2's four families, in the order SPEC.md numbers them. The names are
#: keys into :class:`PortfolioSet` and into every table this module produces, so
#: they are fixed here rather than spelled at each call site.
INDIVIDUAL: Final[str] = "individual_assets"
RANDOM: Final[str] = "random_portfolios"
EIGENFACTOR: Final[str] = "eigenfactors"
OPTIMIZED: Final[str] = "minimum_variance"

FAMILIES: Final[tuple[str, ...]] = (INDIVIDUAL, RANDOM, EIGENFACTOR, OPTIMIZED)


class ValidationError(ValueError):
    """The validation battery was handed inputs it cannot score."""


# ---------------------------------------------------------------------------
# SPEC.md 6.1 -- the bias statistic
# ---------------------------------------------------------------------------


def standardized_returns(realised: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """``b_nt = R_nt / sigma_nt``. SPEC.md 6.1.

    Both arrays are ``T x P`` and aligned so that row ``t`` of ``forecast`` is
    the volatility forecast **made at t-1** for the return in row ``t`` of
    ``realised``. This function cannot check that -- it sees two arrays and no
    time -- so the lead-lag is established in :func:`portfolio_families`, which
    can, and pinned by the leak test in ``tests/test_validation.py``.

    A non-positive forecast is refused rather than floored. Standardising by one
    gives an infinite or sign-flipped ``b``, which then enters a standard
    deviation where it is invisible; and there is no published floor to use
    (CLAUDE.md invariant 9). The same refusal, for the same reason, as
    :func:`mafrm.risk.regime.cross_sectional_bias`.
    """
    if realised.shape != forecast.shape:
        raise ValidationError(
            f"standardized_returns: realised {realised.shape} and forecast {forecast.shape} "
            "must be the same T x P shape"
        )
    if realised.ndim != 2:
        raise ValidationError(f"standardized_returns: expected a T x P array, got {realised.shape}")
    if not np.all(np.isfinite(forecast)):
        raise ValidationError(
            "standardized_returns: a forecast volatility is not finite. This is an upstream "
            "estimator failure and is refused here rather than dropped -- a dropped date would "
            "silently shorten T and widen every interval computed from it."
        )
    if np.any(forecast <= 0.0):
        raise ValidationError(
            "standardized_returns: a forecast volatility is zero or negative, so b_nt would be "
            "infinite or sign-flipped and would then disappear into a standard deviation. There "
            "is no published floor for it (CLAUDE.md invariant 9), so it is refused."
        )
    return np.asarray(realised / forecast, dtype=float)


def bias_statistic(standardized: np.ndarray, *, clip: float) -> np.ndarray:
    """``B_n = sd(clip(b_nt, -clip, +clip), ddof=1)``, one value per column.

    SPEC.md 6.1 writes it out as ``sqrt( (1/(T-1)) sum_t (b_nt - bbar_n)^2 )``,
    so the sample mean **is** subtracted and this is a plain ``ddof=1`` standard
    deviation. Written that way rather than about zero for two reasons, and the
    second is the one that would be easy to get wrong: it is what SPEC.md 6.1
    says, and it is the form for which :func:`chi_square_interval` is exact --
    ``(T-1)s^2/sigma^2 ~ chi^2_{T-1}`` is the mean-subtracted result, and the
    zero-mean sum would be ``chi^2_T`` instead. Deviating here would silently
    move the interval by one degree of freedom.

    Note that this is a **different** convention from SPEC.md 5.1.1's zero-mean
    EWMA, and the two do not conflict: 5.1.1 governs how the *forecast* is
    centred, 6.1 how the standardized returns' dispersion is measured. At a daily
    frequency ``bbar_n`` is parts in a hundred of ``B`` in any case.

    ``ddof=1`` matters at ``T = 12``: the difference between dividing by 11 and
    by 12 is 4% of ``B``, a third of the way to the edge of the exact interval at
    that window length.
    """
    if standardized.ndim != 2:
        raise ValidationError(f"bias_statistic: expected a T x P array, got {standardized.shape}")
    periods = standardized.shape[0]
    if periods < 2:
        raise ValidationError(
            f"bias_statistic: ddof=1 needs at least two observations, got {periods}"
        )
    if not clip > 0.0:
        raise ValidationError(f"bias_statistic: clip must be positive, got {clip}")
    clipped = np.clip(standardized, -clip, clip)
    return np.asarray(np.std(clipped, axis=0, ddof=1), dtype=float)


@dataclass(frozen=True)
class RollingBias:
    """``B_n`` over a rolling window, carrying what CLAUDE.md failure mode 9 needs.

    The chart's series. Every consumer of :attr:`values` also has :attr:`overlap`
    and :attr:`window` in hand, because a rolling reading quoted without them is
    the mistake this project has now made three separate times.
    """

    #: ``(T - window + 1) x P``. Row ``i`` is the window ending at
    #: ``index[i + window - 1]``.
    values: np.ndarray
    #: The date each window ENDS at. Aligned to :attr:`values`' rows.
    index: pd.DatetimeIndex
    #: Observations inside one window. This -- never the number of windows -- is
    #: the ``T`` any interval is computed at.
    window: int
    #: Observations two consecutive windows share, as ``"251 of 252"``.
    overlap: str

    @property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.values, index=self.index)


def rolling_bias_statistic(
    standardized: np.ndarray,
    index: pd.DatetimeIndex,
    *,
    window: int,
    clip: float,
    step: int = 1,
) -> RollingBias:
    """``B_n`` over a rolling window of ``window`` observations. SPEC.md 6.1.

    ``step`` is the stride between windows and is **reported, not hidden**: at
    ``step = 1`` consecutive readings share ``window - 1`` observations, and at
    ``step = window`` they share none. The default is 1 because that is what
    MSCI plots and what SPEC.md 6.1 asks for; the overlap it implies is carried
    on the result so that no caller can quote the number of readings as an ``n``.
    """
    if standardized.ndim != 2:
        raise ValidationError(
            f"rolling_bias_statistic: expected a T x P array, got {standardized.shape}"
        )
    periods = standardized.shape[0]
    if window < 2:
        raise ValidationError(f"rolling_bias_statistic: window must be at least 2, got {window}")
    if step < 1:
        raise ValidationError(f"rolling_bias_statistic: step must be at least 1, got {step}")
    if len(index) != periods:
        raise ValidationError(
            f"rolling_bias_statistic: index carries {len(index)} dates against {periods} rows"
        )
    if periods < window:
        raise ValidationError(
            f"rolling_bias_statistic: {periods} observation(s) cannot fill a {window}-wide "
            "window. The caller owns the burn-in and this function will not shorten the "
            "window to make one fit."
        )
    starts = range(0, periods - window + 1, step)
    rows = [bias_statistic(standardized[start : start + window], clip=clip) for start in starts]
    ends = [index[start + window - 1] for start in starts]
    shared = max(window - step, 0)
    return RollingBias(
        values=np.asarray(rows, dtype=float),
        index=pd.DatetimeIndex(ends),
        window=int(window),
        overlap=f"{shared} of {window}",
    )


def mean_absolute_deviation(bias: np.ndarray) -> float:
    """``MRAD = (1/N) sum_n |B_n - 1|``. SPEC.md 6.1's scalar model summary.

    The mean runs over **members** -- assets, random portfolios, eigenfactors --
    and not over time. A MRAD computed across rolling windows would be an
    average of near-duplicates (failure mode 9); this one averages across a
    cross-section, where the members are genuinely different portfolios.
    """
    values = np.asarray(bias, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValidationError(
            f"mean_absolute_deviation: expected a 1-D bias vector, got {values.shape}"
        )
    return float(np.mean(np.abs(values - 1.0)))


@dataclass(frozen=True)
class BiasInterval:
    """An interval on ``B`` under the null that the model is calibrated.

    Constructed only from the observations **inside one window**, never from a
    count of windows.
    """

    lower: float
    upper: float
    #: ``T``. Stated on every chart and in every table, so that the +-0.57 of a
    #: 12-observation window and the +-0.12 of a 250-observation one can never be
    #: confused for one another.
    observations: int
    #: ``"normal"`` or ``"chi2"``.
    kind: str

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def render(self) -> str:
        return f"[{self.lower:.4f}, {self.upper:.4f}] ({self.kind}, T={self.observations})"


def normal_band(observations: int, *, z: float) -> BiasInterval:
    """``1 +- z/sqrt(T)``. SPEC.md 6.1's band -- **a display convention**.

    Reproduces SPEC.md 6.1's table exactly: ``T = 12`` gives +-0.57, ``T = 60``
    +-0.25, ``T = 120`` +-0.18, ``T = 250`` +-0.12, ``T = 504`` +-0.09.

    The W4-P2 ruling recorded at SPEC.md 6.1.1: this is a normal approximation
    whose constant is arguable -- the sampling standard deviation of a standard
    deviation is ``sigma/sqrt(2(T-1))``, so a ``sqrt(2T)`` denominator is equally
    defensible and would halve the band. **No claim in this project rests on it.**
    :func:`chi_square_interval` is exact and it is the one that decides.
    """
    if observations < 2:
        raise ValidationError(f"normal_band: T must be at least 2, got {observations}")
    if not z > 0.0:
        raise ValidationError(f"normal_band: z must be positive, got {z}")
    half = z / np.sqrt(float(observations))
    return BiasInterval(
        lower=float(1.0 - half),
        upper=float(1.0 + half),
        observations=int(observations),
        kind="normal",
    )


def chi_square_interval(observations: int, *, level: float) -> BiasInterval:
    """The **exact** interval on ``B``. ``(T-1)B^2 ~ chi^2_{T-1}`` under the null.

    ``b_nt`` is standard normal under a calibrated model, so ``sum_t b_nt^2`` is
    ``chi^2_T`` and the ddof=1 form ``(T-1)B^2 = sum b^2`` used by
    :func:`bias_statistic` puts ``(T-1)B^2 ~ chi^2_{T-1}``. Inverting the
    two-sided ``level`` interval gives::

        B in [ sqrt(chi2.ppf(alpha/2, T-1)/(T-1)), sqrt(chi2.ppf(1-alpha/2, T-1)/(T-1)) ]

    **It is asymmetric, and that asymmetry is the point.** At ``T = 12`` the
    exact interval is ``[0.589, 1.412]`` against the normal band's
    ``[0.434, 1.566]``: the normal approximation is far too wide at both edges,
    and most so below one. A reading of 1.45 is unremarkable under the band and
    outside the exact interval.

    SPEC.md 6.1's own caveat still stands and is stated at every use: **returns
    are fat-tailed**, so the normal null this inverts is itself a floor on the
    uncertainty rather than the truth. What the exactness buys is that the
    interval no longer depends on which of two defensible constants was written
    into a specification by hand.
    """
    if observations < 2:
        raise ValidationError(f"chi_square_interval: T must be at least 2, got {observations}")
    if not 0.0 < level < 1.0:
        raise ValidationError(f"chi_square_interval: level must be in (0, 1), got {level}")
    degrees = observations - 1
    alpha = 1.0 - level
    lower = float(np.sqrt(chi2.ppf(alpha / 2.0, degrees) / degrees))
    upper = float(np.sqrt(chi2.ppf(1.0 - alpha / 2.0, degrees) / degrees))
    return BiasInterval(lower=lower, upper=upper, observations=int(observations), kind="chi2")


# ---------------------------------------------------------------------------
# SPEC.md 6.2 -- the four portfolio families
# ---------------------------------------------------------------------------


def asset_covariance(
    exposures: np.ndarray, factor_covariance: np.ndarray, specific_variance: np.ndarray
) -> np.ndarray:
    """``Sigma = X F X' + diag(delta^2)``. The factor model's asset covariance.

    PSD by construction when ``F`` is PSD and ``delta^2 >= 0``, and asserted
    anyway (CLAUDE.md invariant 4): this is a covariance transformation stage
    like any other, and the assertion is what catches a mis-shaped ``X`` or a
    specific variance that arrived as a volatility.
    """
    if exposures.ndim != 2:
        raise ValidationError(f"asset_covariance: expected an N x K design, got {exposures.shape}")
    assets, factors = exposures.shape
    if factor_covariance.shape != (factors, factors):
        raise ValidationError(
            f"asset_covariance: X is N x {factors} but F is {factor_covariance.shape}"
        )
    if specific_variance.shape != (assets,):
        raise ValidationError(
            f"asset_covariance: X has {assets} rows but the specific variance has "
            f"{specific_variance.shape}"
        )
    if np.any(specific_variance < 0.0):
        raise ValidationError("asset_covariance: a specific variance is negative")
    matrix = exposures @ factor_covariance @ exposures.T + np.diag(specific_variance)
    return np.asarray(0.5 * (matrix + matrix.T), dtype=float)


def random_dollar_neutral(count: int, assets: int, rng: np.random.Generator) -> np.ndarray:
    """SPEC.md 6.2 family 2: ``count`` dollar-neutral N(0,1) weight vectors.

    Dollar-neutral is imposed by subtracting the cross-sectional mean of each
    draw, which is the projection of an N(0,1) vector onto ``1'w = 0`` and
    leaves the weights exchangeable across assets. The draws are **independent
    of the covariance matrix**, which is the whole content of family 2: SPEC.md
    6.2 expects ``B ~ 1`` here for *every* estimator including the naive sample
    covariance, precisely because the weights cannot be a function of the
    estimation error.

    ``rng`` is passed in, never created here, and never seeded globally
    (CLAUDE.md code conventions).
    """
    if count < 1:
        raise ValidationError(f"random_dollar_neutral: count must be at least 1, got {count}")
    if assets < 2:
        raise ValidationError(
            f"random_dollar_neutral: dollar-neutrality needs at least two assets, got {assets}"
        )
    draws = rng.standard_normal((count, assets))
    return np.asarray(draws - draws.mean(axis=1, keepdims=True), dtype=float)


def minimum_variance_weights(covariance: np.ndarray) -> np.ndarray:
    """SPEC.md 6.2 family 4: ``w = Sigma^-1 1 / (1' Sigma^-1 1)``, fully invested.

    **Why this portfolio is the test that matters.** ``Sigma^-1`` weights the
    smallest eigen-directions of the forecast most heavily, and those are exactly
    the directions in which sampling error makes eigenvalues too *small*. So the
    optimizer's weights are a function of the estimation error, in the direction
    that maximises it -- which is what family 2's weights, drawn independently of
    ``Sigma``, cannot be. The gap between the two families is the risk-model term
    in SPEC.md 1's identity.

    Solved rather than inverted (``np.linalg.solve``, not ``inv``): a Cholesky
    solve on a PSD matrix is both more accurate and, at the condition numbers
    SPEC.md 5.3.1 records for this panel, the difference is not academic.

    The alpha-constrained sibling SPEC.md 6.2 also names -- minimum variance
    subject to ``alpha'w = 1`` -- is **deferred to W6** by the W4-P2 ruling at
    SPEC.md 6.2.1, because ``alpha`` is unspecified and CLAUDE.md admits expected
    returns only as a fixed, documented optimizer input that does not exist yet.
    """
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValidationError(
            f"minimum_variance_weights: expected a square matrix, got {covariance.shape}"
        )
    ones = np.ones(covariance.shape[0], dtype=float)
    solved = np.linalg.solve(covariance, ones)
    denominator = float(ones @ solved)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValidationError(
            f"minimum_variance_weights: 1' Sigma^-1 1 = {denominator}, which is not positive. "
            "For a PSD Sigma it must be; a non-positive value means the matrix is singular or "
            "has been corrupted upstream, and normalising by it would produce weights whose "
            "sign is meaningless."
        )
    return np.asarray(solved / denominator, dtype=float)


def eigen_portfolios(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SPEC.md 6.2 family 3: the eigenvectors of ``F``, ranked by eigenvalue.

    Returns ``(vectors, eigenvalues)`` with ``vectors[k]`` the ``k``-th
    eigenvector and rank 0 the **largest** eigenvalue. SPEC.md 6.2 expects the
    realised bias to run "1.5 at the smallest declining to 0.95 at the largest
    before adjustment, flat ~1.0 after", so the ranking is load-bearing and is
    fixed here rather than left to ``eigh``'s ascending convention.

    These live in **factor space**: eigenfactor ``k``'s realised return is
    ``u_k' f_t`` and its forecast variance is its own eigenvalue. They are the
    portfolios SPEC.md 5.3's adjustment is *about*, which is why family 3 is
    defined on ``F`` rather than on the asset-level ``Sigma`` -- and why a
    variant with no factor structure (the naive sample covariance) has no
    family 3 rather than a substituted one.

    The eigenvector sign is arbitrary and is left arbitrary: ``B`` is a standard
    deviation about zero, so ``u`` and ``-u`` give the identical statistic. Not
    a convention worth imposing, and SPEC.md 4.2.3 already ruled on what a sign
    flip is and is not.
    """
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValidationError(f"eigen_portfolios: expected a square matrix, got {covariance.shape}")
    eigenvalues, vectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return np.asarray(vectors[:, order].T, dtype=float), np.asarray(eigenvalues[order], dtype=float)


# ---------------------------------------------------------------------------
# One date's forecast, and the walk over dates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskForecast:
    """One date's finished risk forecast, in the two spaces SPEC.md 6.2 needs.

    :attr:`covariance` is the ``N x N`` asset-level matrix every asset-space
    family is evaluated on. :attr:`factor_covariance` is the ``K x K`` matrix
    family 3 lives in and is ``None`` for an estimator with no factor structure
    -- the naive sample covariance SPEC.md 6.2 asks to include as a comparand.
    """

    #: ``N x N``, made from data strictly before the date it forecasts.
    covariance: np.ndarray
    #: ``K x K``, or ``None`` when the estimator has no factors.
    factor_covariance: np.ndarray | None = None

    @classmethod
    def from_factor_model(
        cls,
        exposures: np.ndarray,
        factor_covariance: np.ndarray,
        specific_variance: np.ndarray,
    ) -> RiskForecast:
        """``Sigma = X F X' + diag(delta^2)``, keeping ``F`` for family 3.

        ``exposures`` must be the design **known at t-1**. A rolling beta at
        date ``t`` is fitted on a window ending at ``t``, so passing that row
        would standardise the return at ``t`` by a forecast that had already
        seen it. :func:`portfolio_families` enforces the lag at the only place
        that can see the dates.
        """
        return cls(
            covariance=asset_covariance(exposures, factor_covariance, specific_variance),
            factor_covariance=np.asarray(factor_covariance, dtype=float),
        )


@dataclass(frozen=True)
class PortfolioSet:
    """Which members belong to which of SPEC.md 6.2's families.

    Deliberately just the membership map. :func:`validate` takes the forecasts
    and realisations as frames and this as the partition, which is SPEC.md 15.2's
    signature verbatim and keeps ``validate`` unable to see how a portfolio was
    constructed -- so it cannot condition on it.
    """

    families: dict[str, tuple[str, ...]]
    #: How often the weights were rebuilt, for the label on every table.
    rebalance: str

    def family_of(self, member: str) -> str:
        for family, names in self.families.items():
            if member in names:
                return family
        raise ValidationError(f"PortfolioSet: {member!r} is in no family")


def _weights_for_date(
    forecast: RiskForecast, random_weights: np.ndarray, assets: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Asset-space weights for families 1, 2 and 4, and factor-space for family 3."""
    asset_space = [
        np.eye(assets),
        random_weights,
        minimum_variance_weights(forecast.covariance)[None, :],
    ]
    factor_space = None
    if forecast.factor_covariance is not None:
        factor_space = eigen_portfolios(forecast.factor_covariance)[0]
    return np.vstack(asset_space), factor_space


def portfolio_families(
    forecasts: Sequence[RiskForecast],
    *,
    dates: pd.DatetimeIndex,
    asset_returns: np.ndarray,
    factor_returns: np.ndarray | None,
    asset_names: Sequence[str],
    factor_names: Sequence[str],
    random_portfolios: int,
    seed: int,
    rebalance: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, PortfolioSet]:
    """Build SPEC.md 6.2's families and evaluate them date by date.

    ``forecasts[i]`` is the risk forecast **for** ``dates[i]``, built from data
    strictly before it; ``asset_returns[i]`` and ``factor_returns[i]`` are the
    realisations **on** ``dates[i]``. The caller owns that alignment, and it is
    the single thing this whole module depends on being right -- see the module
    docstring and the leak test.

    ``rebalance`` is a boolean mask over ``dates`` saying where the weights are
    rebuilt; ``None`` means every date, which is SPEC.md 6.2's "redrawn each
    period" and "rebuilt each period from the then-current forecast". A monthly
    mask holds each period's weights across the month, which is what the monthly
    leg of the report needs: a month's return is only the return *of a portfolio*
    if the portfolio was not rebuilt inside it.

    Returns ``(forecast_volatilities, realised_returns, portfolios)`` -- the
    three arguments of :func:`validate`, and SPEC.md 15.2's signature.
    """
    periods = len(dates)
    assets = len(asset_names)
    if len(forecasts) != periods:
        raise ValidationError(
            f"portfolio_families: {len(forecasts)} forecast(s) against {periods} date(s)"
        )
    if asset_returns.shape != (periods, assets):
        raise ValidationError(
            f"portfolio_families: asset_returns is {asset_returns.shape}, expected "
            f"({periods}, {assets})"
        )
    has_factors = factor_returns is not None
    if has_factors:
        assert factor_returns is not None  # narrowed for mypy
        if factor_returns.shape != (periods, len(factor_names)):
            raise ValidationError(
                f"portfolio_families: factor_returns is {factor_returns.shape}, expected "
                f"({periods}, {len(factor_names)})"
            )
    if rebalance is None:
        mask = np.ones(periods, dtype=bool)
    else:
        mask = np.asarray(rebalance, dtype=bool)
        if mask.shape != (periods,):
            raise ValidationError(
                f"portfolio_families: rebalance mask is {mask.shape}, expected ({periods},)"
            )
    if not mask[0]:
        raise ValidationError(
            "portfolio_families: the rebalance mask must be True on the first date -- there are "
            "no weights to hold before the first rebuild."
        )

    rng = np.random.default_rng(seed)
    members_asset = [str(name) for name in asset_names]
    members_asset += [f"random_{index:03d}" for index in range(random_portfolios)]
    members_asset += ["min_var"]
    members_factor = (
        [f"eigen_{rank + 1:02d}" for rank in range(len(factor_names))] if has_factors else []
    )

    forecast_rows: list[np.ndarray] = []
    realised_rows: list[np.ndarray] = []
    held_asset: np.ndarray | None = None
    held_factor: np.ndarray | None = None
    for position in range(periods):
        forecast = forecasts[position]
        if mask[position]:
            # The draw advances the generator on rebalance dates only, so the
            # monthly leg is not a subsample of the daily leg's draws -- it is
            # its own sequence. Stated because the alternative (drawing every
            # date and discarding) would make the two legs share weights and
            # look more alike than two independent draws would.
            draw = random_dollar_neutral(random_portfolios, assets, rng)
            held_asset, held_factor = _weights_for_date(forecast, draw, assets)
        if held_asset is None:  # pragma: no cover - guarded by the mask[0] check
            raise ValidationError("portfolio_families: no weights held")
        variances = np.einsum("pi,ij,pj->p", held_asset, forecast.covariance, held_asset)
        returns = held_asset @ asset_returns[position]
        if has_factors:
            assert factor_returns is not None and held_factor is not None
            factor_matrix = forecast.factor_covariance
            assert factor_matrix is not None
            variances = np.concatenate(
                [variances, np.einsum("pi,ij,pj->p", held_factor, factor_matrix, held_factor)]
            )
            returns = np.concatenate([returns, held_factor @ factor_returns[position]])
        forecast_rows.append(np.sqrt(variances))
        realised_rows.append(returns)

    columns = members_asset + members_factor
    families = {
        INDIVIDUAL: tuple(str(name) for name in asset_names),
        RANDOM: tuple(f"random_{index:03d}" for index in range(random_portfolios)),
        OPTIMIZED: ("min_var",),
    }
    if has_factors:
        families[EIGENFACTOR] = tuple(members_factor)
    return (
        pd.DataFrame(np.asarray(forecast_rows), index=dates, columns=columns),
        pd.DataFrame(np.asarray(realised_rows), index=dates, columns=columns),
        PortfolioSet(
            families=families,
            rebalance="every date" if rebalance is None else "held between rebuilds",
        ),
    )


# ---------------------------------------------------------------------------
# SPEC.md 15.2's entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemberBias:
    """One portfolio's ``B``, with the ``T`` it was computed at."""

    member: str
    family: str
    bias: float
    observations: int

    def within(self, interval: BiasInterval) -> bool:
        return interval.contains(self.bias)


@dataclass(frozen=True)
class ValidationReport:
    """What :func:`validate` produces. SPEC.md 15.2's return type.

    The headline of SPEC.md 6.2 is :meth:`family_bias` on ``random_portfolios``
    against ``minimum_variance``: the same covariance matrix, two families, and
    the gap between them is the risk-model term of SPEC.md 1's identity.
    """

    members: tuple[MemberBias, ...]
    portfolios: PortfolioSet
    #: ``T`` for the full-sample statistic. Stated everywhere it is used.
    observations: int
    clip: float
    #: Fraction of ``b_nt`` values the clip actually bound, per family. Reported
    #: because a clip that binds often is doing more than removing one 2020 day.
    clipped_fraction: dict[str, float]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "family": [row.family for row in self.members],
                "B": [row.bias for row in self.members],
                "T": [row.observations for row in self.members],
            },
            index=[row.member for row in self.members],
        )

    def family_bias(self, family: str) -> np.ndarray:
        """Every member's ``B`` in one family, in membership order."""
        values = [row.bias for row in self.members if row.family == family]
        if not values:
            raise ValidationError(f"ValidationReport: no member in family {family!r}")
        return np.asarray(values, dtype=float)

    def mrad(self, family: str) -> float:
        """SPEC.md 6.1's ``MRAD`` for one family."""
        return mean_absolute_deviation(self.family_bias(family))

    def median_bias(self, family: str) -> float:
        return float(np.median(self.family_bias(family)))


def validate(
    forecasts: pd.DataFrame, realised: pd.DataFrame, portfolios: PortfolioSet, *, clip: float
) -> ValidationReport:
    """SPEC.md 15.2's ``validate``. Bias statistics for every member and family.

    ``forecasts`` and ``realised`` are date x member frames on the same index and
    columns; ``portfolios`` says which family each member belongs to. Nothing
    here can see how a portfolio was built, which is what stops this function
    from ever treating family 4 differently from family 2 -- the comparison
    SPEC.md 6.2 turns on is only worth anything if both sides went through the
    identical code path.
    """
    if list(forecasts.columns) != list(realised.columns):
        raise ValidationError("validate: forecasts and realised carry different members")
    if not forecasts.index.equals(realised.index):
        raise ValidationError("validate: forecasts and realised carry different dates")
    standardized = standardized_returns(
        realised.to_numpy(dtype=float), forecasts.to_numpy(dtype=float)
    )
    bias = bias_statistic(standardized, clip=clip)
    members = tuple(
        MemberBias(
            member=str(column),
            family=portfolios.family_of(str(column)),
            bias=float(value),
            observations=int(standardized.shape[0]),
        )
        for column, value in zip(forecasts.columns, bias, strict=True)
    )
    bound = np.abs(standardized) > clip
    position_of = {str(name): index for index, name in enumerate(forecasts.columns)}
    clipped: dict[str, float] = {}
    for family, names in portfolios.families.items():
        positions = [position_of[name] for name in names]
        clipped[family] = float(bound[:, positions].mean())
    return ValidationReport(
        members=members,
        portfolios=portfolios,
        observations=int(standardized.shape[0]),
        clip=float(clip),
        clipped_fraction=clipped,
    )


def assert_forecast_psd(forecast: RiskForecast, label: str) -> None:
    """CLAUDE.md invariant 4 at the last transformation before a number is published.

    ``Sigma = X F X' + diag(delta^2)`` is a covariance transformation like any
    other and it is the one the optimizer inverts, so it gets the same assertion
    every stage of SPEC.md 5 gets rather than being trusted because its inputs
    were checked.
    """
    assert_psd(forecast.covariance, label, expected_size=forecast.covariance.shape[0])
