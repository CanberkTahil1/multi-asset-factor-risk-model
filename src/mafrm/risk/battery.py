"""SPEC.md 6.5's battery, as arithmetic on standardised returns. W4-P3.

Everything here takes arrays and returns numbers. The one series every test
runs on is ``b_t = R_t / sigma_t`` with ``sigma_t`` the forecast made at ``t-1``
(SPEC.md 6.1), and the report module says whose ``b_t`` it is; this module does
not know and may not (CLAUDE.md invariant 10).

WHAT THE DISTRIBUTIONAL TESTS ASSUME, AND WHY IT IS NOT A CHOICE
-----------------------------------------------------------------

A variance forecast says nothing about a quantile until a distribution is named.
The tests below use the **normal**: ``VaR_t = z_alpha sigma_t`` and
``ES_t = sigma_t phi(z_alpha) / alpha``. That is the distribution the forecast
implicitly assumes and the one Basel-style backtesting assumes, so it invents
nothing. A Student-t would need a ``nu`` -- invented or fitted, and fitting it
would be a ``model-config`` trial for a distributional claim this project never
made. What the normal *costs* is reported beside every test rather than fed into
one: :func:`empirical_quantile` gives the standardised returns' own tail quantile
against ``-z_alpha``, as a diagnostic of how fat the tail is. It is never used to
set the VaR being tested, because calibrating the quantile on the sample being
tested is circular.

Two levels, 95% and 99%, reported side by side and neither selected (W4-P3
ruling 1). At ``T ~ 3,900`` the 99% level expects ~39 breaches and the 95% ~195,
and Christoffersen's independence test needs breaches dense enough to cluster
detectably, so the pair is also a statement about the tests' own power.

CLAUDE.md FAILURE MODE 9
------------------------

:func:`basel_traffic_light` and :func:`cross_sectional_rank_correlation` use
**non-overlapping** blocks and say so in their results; :func:`factor_t_statistics`
reports a rolling series with its overlap stated and a non-overlapping series
beside it, and only the latter is given an ``n``.

CITATIONS
---------

Mincer & Zarnowitz (1969); Patton (2011) for QLIKE; Ljung & Box (1978); Kupiec
(1995); Christoffersen (1998); Basel Committee (1996) for the traffic light;
Acerbi & Szekely (2014) for ``Z_2`` and Acerbi & Szekely (2019) for the
minimally biased statistic, whose form is stated at the function because this
repository does not hold the paper and states its own normalisation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom, chi2, norm, spearmanr

__all__ = [
    "AcerbiSzekely",
    "BaselBlock",
    "BaselTrafficLight",
    "BatteryError",
    "Christoffersen",
    "FactorTStatistic",
    "Kupiec",
    "LjungBox",
    "MincerZarnowitz",
    "RankCorrelation",
    "TailQuantile",
    "acerbi_szekely",
    "basel_traffic_light",
    "binomial_cumulative",
    "breach_indicator",
    "christoffersen",
    "cross_sectional_rank_correlation",
    "empirical_quantile",
    "factor_t_statistics",
    "kupiec",
    "ljung_box",
    "mincer_zarnowitz",
    "newey_west_variance_of_mean",
    "qlike_loss",
]


class BatteryError(ValueError):
    """A test was handed inputs it cannot score."""


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise BatteryError(f"{name}: expected a non-empty 1-D array, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise BatteryError(f"{name}: non-finite values")
    return array


def _xlogy(count: float, probability: float) -> float:
    """``count * log(probability)`` with ``0 * log 0 = 0``, the likelihood convention."""
    if count == 0.0:
        return 0.0
    if probability <= 0.0:
        return -np.inf
    return float(count * np.log(probability))


# ---------------------------------------------------------------------------
# Newey-West on a mean
# ---------------------------------------------------------------------------


def newey_west_variance_of_mean(values: np.ndarray, *, lags: int) -> float:
    """HAC variance of a sample mean. Newey & West (1987), Bartlett kernel.

    ``S = gamma_0 + 2 sum_{l=1..L} (1 - l/(L+1)) gamma_l`` with autocovariances at
    ``1/T``, and the variance of the mean is ``S / T``. ``lags = 0`` is the plain
    ``sigma^2 / T``. Daily returns are serially correlated enough that a plain
    standard error on their mean is too small, which is why every mean
    t-statistic here goes through this.
    """
    series = _vector(values, "newey_west_variance_of_mean")
    if lags < 0:
        raise BatteryError(f"newey_west_variance_of_mean: lags must be >= 0, got {lags}")
    count = series.size
    centred = series - series.mean()
    long_run = float(centred @ centred) / count
    for lag in range(1, min(lags, count - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        long_run += 2.0 * weight * float(centred[lag:] @ centred[:-lag]) / count
    return long_run / count


# ---------------------------------------------------------------------------
# Mincer-Zarnowitz with QLIKE
# ---------------------------------------------------------------------------


def qlike_loss(realised_variance: np.ndarray, forecast_variance: np.ndarray) -> float:
    """Mean QLIKE, ``log h_t + r_t^2 / h_t``. Patton (2011).

    Patton's QLIKE is ``r^2/h - log(r^2/h) - 1``; this drops the
    ``-log r^2 - 1`` terms, which do not depend on the forecast, so the two rank
    forecasts identically and this one stays finite at ``r_t = 0``. It is the
    loss under which the conditional variance is the optimal forecast even when
    the realised proxy is noisy, and unlike MSE it does not let a handful of
    crisis days decide the ranking.
    """
    realised = _vector(realised_variance, "qlike_loss.realised")
    forecast = _vector(forecast_variance, "qlike_loss.forecast")
    if realised.shape != forecast.shape:
        raise BatteryError("qlike_loss: realised and forecast differ in length")
    if np.any(forecast <= 0.0):
        raise BatteryError("qlike_loss: a forecast variance is not positive")
    if np.any(realised < 0.0):
        raise BatteryError("qlike_loss: a realised variance is negative")
    return float(np.mean(np.log(forecast) + realised / forecast))


@dataclass(frozen=True)
class MincerZarnowitz:
    """``r_t^2 = a + b h_t + e_t``. Calibration is ``a = 0, b = 1``."""

    intercept: float
    slope: float
    intercept_standard_error: float
    slope_standard_error: float
    #: Wald statistic for ``(a, b) = (0, 1)`` jointly, HAC covariance, ``chi^2_2``.
    joint_wald: float
    joint_p_value: float
    r_squared: float
    qlike: float
    observations: int
    newey_west_lags: int

    @property
    def slope_t_against_one(self) -> float:
        return (self.slope - 1.0) / self.slope_standard_error


def mincer_zarnowitz(
    realised_variance: np.ndarray, forecast_variance: np.ndarray, *, newey_west_lags: int
) -> MincerZarnowitz:
    """OLS of realised on forecast variance, HAC standard errors, joint Wald test.

    Catches *conditional* miscalibration -- right on average and wrong at high
    volatility -- which a bias statistic, being one number, cannot see. Squared
    daily returns are heteroskedastic and autocorrelated, so the standard errors
    are Newey-West with the lag count the caller reads from
    ``covariance.volatility_newey_west_lags``.
    """
    realised = _vector(realised_variance, "mincer_zarnowitz.realised")
    forecast = _vector(forecast_variance, "mincer_zarnowitz.forecast")
    if realised.shape != forecast.shape:
        raise BatteryError("mincer_zarnowitz: realised and forecast differ in length")
    count = realised.size
    if count < 3:
        raise BatteryError(f"mincer_zarnowitz: {count} observations cannot fit two coefficients")
    design = np.column_stack([np.ones(count), forecast])
    bread = np.linalg.pinv(design.T @ design)
    coefficients = bread @ design.T @ realised
    residual = realised - design @ coefficients
    scores = design * residual[:, None]
    meat = scores.T @ scores
    for lag in range(1, min(newey_west_lags, count - 1) + 1):
        block = scores[lag:].T @ scores[:-lag]
        meat += (1.0 - lag / (newey_west_lags + 1.0)) * (block + block.T)
    covariance = bread @ meat @ bread
    total = float(np.sum(np.square(realised - realised.mean())))
    r_squared = 1.0 - float(residual @ residual) / total if total > 0.0 else float("nan")
    departure = coefficients - np.array([0.0, 1.0])
    wald = float(departure @ np.linalg.pinv(covariance) @ departure)
    return MincerZarnowitz(
        intercept=float(coefficients[0]),
        slope=float(coefficients[1]),
        intercept_standard_error=float(np.sqrt(max(covariance[0, 0], 0.0))),
        slope_standard_error=float(np.sqrt(max(covariance[1, 1], 0.0))),
        joint_wald=wald,
        joint_p_value=float(chi2.sf(wald, df=2)),
        r_squared=r_squared,
        qlike=qlike_loss(realised, forecast),
        observations=count,
        newey_west_lags=int(newey_west_lags),
    )


# ---------------------------------------------------------------------------
# Ljung-Box
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LjungBox:
    statistic: float
    p_value: float
    lags: int
    observations: int


def ljung_box(values: np.ndarray, *, lags: int) -> LjungBox:
    """``Q = n(n+2) sum_{k=1..h} r_k^2 / (n-k)``, ``chi^2_h``. Ljung & Box (1978).

    Run on ``b_t^2`` it asks whether volatility clustering survived the forecast:
    a model that tracks the level of variance leaves squared standardised
    returns without autocorrelation. Which ``h`` matters, and there is no
    published one; W4-P3 runs a month, a quarter and a year -- the timescales the
    model's own half-lives live at -- and reports all three.
    """
    series = _vector(values, "ljung_box")
    count = series.size
    if lags < 1 or lags >= count:
        raise BatteryError(f"ljung_box: lags must be in [1, n-1] = [1, {count - 1}], got {lags}")
    centred = series - series.mean()
    denominator = float(centred @ centred)
    if denominator <= 0.0:
        raise BatteryError("ljung_box: the series is constant")
    statistic = 0.0
    for lag in range(1, lags + 1):
        autocorrelation = float(centred[lag:] @ centred[:-lag]) / denominator
        statistic += autocorrelation**2 / (count - lag)
    statistic *= count * (count + 2.0)
    return LjungBox(
        statistic=float(statistic),
        p_value=float(chi2.sf(statistic, df=lags)),
        lags=int(lags),
        observations=int(count),
    )


# ---------------------------------------------------------------------------
# VaR breaches: Kupiec, Christoffersen, Basel
# ---------------------------------------------------------------------------


def breach_indicator(standardized: np.ndarray, *, level: float) -> np.ndarray:
    """``1{ b_t < -z_level }``: the loss exceeded the normal VaR at ``level``."""
    series = _vector(standardized, "breach_indicator")
    if not 0.5 < level < 1.0:
        raise BatteryError(f"breach_indicator: level must be in (0.5, 1), got {level}")
    threshold = -float(norm.ppf(level))
    return np.asarray(series < threshold, dtype=bool)


@dataclass(frozen=True)
class Kupiec:
    """Unconditional coverage. Kupiec (1995)."""

    breaches: int
    observations: int
    level: float
    statistic: float
    p_value: float

    @property
    def expected(self) -> float:
        return self.observations * (1.0 - self.level)

    @property
    def observed_rate(self) -> float:
        return self.breaches / self.observations


def kupiec(breaches: int, observations: int, *, level: float) -> Kupiec:
    """``LR_uc = 2[ x ln(x/(Tp)) + (T-x) ln((T-x)/(T(1-p))) ]``, ``chi^2_1``.

    ``p = 1 - level`` is the breach probability the VaR claims. Zero exactly when
    the observed breach rate equals ``p``.
    """
    if observations < 1:
        raise BatteryError("kupiec: no observations")
    if not 0 <= breaches <= observations:
        raise BatteryError(f"kupiec: {breaches} breaches out of {observations}")
    if not 0.5 < level < 1.0:
        raise BatteryError(f"kupiec: level must be in (0.5, 1), got {level}")
    probability = 1.0 - level
    misses = observations - breaches
    statistic = 2.0 * (
        _xlogy(breaches, breaches / observations)
        - _xlogy(breaches, probability)
        + _xlogy(misses, misses / observations)
        - _xlogy(misses, 1.0 - probability)
    )
    return Kupiec(
        breaches=int(breaches),
        observations=int(observations),
        level=float(level),
        statistic=float(statistic),
        p_value=float(chi2.sf(statistic, df=1)),
    )


@dataclass(frozen=True)
class Christoffersen:
    """Independence and conditional coverage of breaches. Christoffersen (1998)."""

    transitions: tuple[int, int, int, int]
    #: ``P(breach | no breach yesterday)`` and ``P(breach | breach yesterday)``.
    pi_01: float
    pi_11: float
    independence_statistic: float
    independence_p_value: float
    conditional_coverage_statistic: float
    conditional_coverage_p_value: float
    unconditional: Kupiec


def christoffersen(indicator: np.ndarray, *, level: float) -> Christoffersen:
    """``LR_ind`` on the first-order Markov chain of breaches; ``LR_cc = LR_uc + LR_ind``.

    Kupiec sees the breach count and is blind to when the breaches fell. A model
    that misses every crisis gets its breaches in runs, and ``LR_ind`` is the
    test that sees the run: it compares the breach probability after a breach to
    the breach probability after a quiet day. Zero exactly when the two are
    equal. ``LR_cc`` adds Kupiec's statistic and is ``chi^2_2``; it is computed on
    all ``T`` observations for the coverage part and ``T-1`` transitions for the
    independence part, the standard convention.
    """
    flags = np.asarray(indicator, dtype=bool)
    if flags.ndim != 1 or flags.size < 2:
        raise BatteryError("christoffersen: need at least two observations")
    previous, current = flags[:-1], flags[1:]
    n00 = int(np.sum(~previous & ~current))
    n01 = int(np.sum(~previous & current))
    n10 = int(np.sum(previous & ~current))
    n11 = int(np.sum(previous & current))
    pi_01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi_11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pooled = (n01 + n11) / (n00 + n01 + n10 + n11)
    restricted = _xlogy(n00 + n10, 1.0 - pooled) + _xlogy(n01 + n11, pooled)
    unrestricted = (
        _xlogy(n00, 1.0 - pi_01)
        + _xlogy(n01, pi_01)
        + _xlogy(n10, 1.0 - pi_11)
        + _xlogy(n11, pi_11)
    )
    independence = max(-2.0 * (restricted - unrestricted), 0.0)
    coverage = kupiec(int(flags.sum()), int(flags.size), level=level)
    conditional = coverage.statistic + independence
    return Christoffersen(
        transitions=(n00, n01, n10, n11),
        pi_01=float(pi_01),
        pi_11=float(pi_11),
        independence_statistic=float(independence),
        independence_p_value=float(chi2.sf(independence, df=1)),
        conditional_coverage_statistic=float(conditional),
        conditional_coverage_p_value=float(chi2.sf(conditional, df=2)),
        unconditional=coverage,
    )


def binomial_cumulative(breaches: int, *, window: int, probability: float) -> float:
    """``P(X <= breaches)`` for ``X ~ Binomial(window, probability)``. The Basel column."""
    return float(binom.cdf(breaches, window, probability))


@dataclass(frozen=True)
class BaselBlock:
    start: int
    breaches: int
    cumulative_probability: float
    zone: str


@dataclass(frozen=True)
class BaselTrafficLight:
    """Zone per NON-OVERLAPPING block of ``window`` days. Basel Committee (1996)."""

    blocks: tuple[BaselBlock, ...]
    window: int
    level: float
    green_max: int
    yellow_max: int
    #: Trailing observations that did not fill a block and were not scored.
    unscored_tail: int

    def count(self, zone: str) -> int:
        return sum(1 for block in self.blocks if block.zone == zone)

    @property
    def overlap(self) -> int:
        """Zero. Stated because CLAUDE.md failure mode 9 says to state it."""
        return 0


def basel_traffic_light(
    indicator: np.ndarray, *, window: int, level: float, green_max: int, yellow_max: int
) -> BaselTrafficLight:
    """Green ``<= green_max``, yellow to ``yellow_max``, red above, per block.

    The zones are Basel's own for a 99% VaR over 250 days (4 and 9), and the
    binomial cumulative probability is reported beside each count so the zone
    can be read as the probability statement it is. Blocks are consecutive and
    non-overlapping; a rolling 250-day count would share 249 of 250 days with
    its neighbour and its zone series would be one zone seen 250 times.
    """
    flags = np.asarray(indicator, dtype=bool)
    if flags.ndim != 1:
        raise BatteryError("basel_traffic_light: expected a 1-D indicator")
    if window < 1 or not 0 <= green_max < yellow_max:
        raise BatteryError("basel_traffic_light: window and zone bounds are inconsistent")
    if flags.size < window:
        raise BatteryError(
            f"basel_traffic_light: {flags.size} observations do not fill one {window}-day block"
        )
    probability = 1.0 - level
    blocks: list[BaselBlock] = []
    for start in range(0, flags.size - window + 1, window):
        breaches = int(flags[start : start + window].sum())
        zone = "green" if breaches <= green_max else "yellow" if breaches <= yellow_max else "red"
        blocks.append(
            BaselBlock(
                start=start,
                breaches=breaches,
                cumulative_probability=binomial_cumulative(
                    breaches, window=window, probability=probability
                ),
                zone=zone,
            )
        )
    return BaselTrafficLight(
        blocks=tuple(blocks),
        window=int(window),
        level=float(level),
        green_max=int(green_max),
        yellow_max=int(yellow_max),
        unscored_tail=int(flags.size - len(blocks) * window),
    )


# ---------------------------------------------------------------------------
# Expected shortfall: Acerbi-Szekely
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcerbiSzekely:
    """``Z_2`` and the minimally biased statistic, with simulated null p-values."""

    level: float
    z_2: float
    z_2_p_value: float
    minimally_biased: float
    minimally_biased_p_value: float
    breaches: int
    observations: int
    trials: int


def _acerbi_statistics(standardized: np.ndarray, *, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Both statistics, vectorised over columns of ``standardized`` (``T x M``)."""
    alpha = 1.0 - level
    quantile = float(norm.ppf(level))
    density = float(norm.pdf(quantile))
    expected_shortfall = density / alpha
    breaches = standardized < -quantile
    count = standardized.shape[0]
    z_2 = np.sum(standardized * breaches, axis=0) / (count * density) + 1.0
    ridge = quantile + np.maximum(-standardized - quantile, 0.0) / alpha
    minimally_biased = ridge.mean(axis=0) / expected_shortfall - 1.0
    return np.asarray(z_2, dtype=float), np.asarray(minimally_biased, dtype=float)


def acerbi_szekely(
    standardized: np.ndarray, *, level: float, trials: int, seed: int
) -> AcerbiSzekely:
    """Expected-shortfall backtests in standardised units under the normal.

    **Z_2** (Acerbi & Szekely 2014, test 2):
    ``Z_2 = sum_t X_t I_t / (T alpha ES_t) + 1``, zero in expectation when the
    ES forecast is right and negative when tail losses exceed it. With
    ``X_t = b_t sigma_t``, ``ES_t = sigma_t phi(z)/alpha`` and ``I_t = 1{b_t < -z}``
    this is ``sum_t b_t I_t / (T phi(z)) + 1`` -- ``sigma_t`` cancels, which is why
    the test can run on the standardised series alone.

    **The minimally biased statistic** (Acerbi & Szekely 2019). Stated here in
    the form this project implements, because the paper is not held: the
    Rockafellar-Uryasev representation ``ES = min_v [ v + E(-X - v)^+ / alpha ]``
    is attained at ``v = VaR``, so the realised ridge term
    ``u_t = VaR_t + (-X_t - VaR_t)^+ / alpha`` has expectation ``ES_t`` under a
    correct model and its bias is second-order in a VaR error. The statistic is
    ``mean_t u_t / ES - 1`` in standardised units -- zero when calibrated,
    **positive** when ES is understated. If the published normalisation differs
    it does so by a positive constant, which moves neither the sign nor the
    p-value.

    ES is not elicitable, so neither statistic has a closed-form null; both
    p-values come from ``trials`` simulated standard-normal series of the same
    length, one-sided in the direction of understatement. ``trials`` is read by
    the caller from ``eigenfactor.monte_carlo_trials`` (W4-P3 ruling 4).
    """
    series = _vector(standardized, "acerbi_szekely")
    if not 0.5 < level < 1.0:
        raise BatteryError(f"acerbi_szekely: level must be in (0.5, 1), got {level}")
    if trials < 2:
        raise BatteryError(f"acerbi_szekely: at least 2 trials, got {trials}")
    z_2, minimally_biased = _acerbi_statistics(series[:, None], level=level)
    generator = np.random.default_rng(seed)
    null = generator.standard_normal((series.size, trials))
    null_z_2, null_minimally_biased = _acerbi_statistics(null, level=level)
    return AcerbiSzekely(
        level=float(level),
        z_2=float(z_2[0]),
        z_2_p_value=float(np.mean(null_z_2 <= z_2[0])),
        minimally_biased=float(minimally_biased[0]),
        minimally_biased_p_value=float(np.mean(null_minimally_biased >= minimally_biased[0])),
        breaches=int(np.sum(series < -float(norm.ppf(level)))),
        observations=int(series.size),
        trials=int(trials),
    )


# ---------------------------------------------------------------------------
# The tail-fatness diagnostic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TailQuantile:
    """The standardised returns' own lower quantile beside the normal's."""

    level: float
    empirical: float
    normal: float

    @property
    def ratio(self) -> float:
        """``empirical / normal``: above one, the tail is fatter than the forecast assumes."""
        return self.empirical / self.normal


def empirical_quantile(standardized: np.ndarray, *, level: float) -> TailQuantile:
    """``quantile_{1-level}(b_t)`` against ``-z_level``. Diagnostic only, never a VaR."""
    series = _vector(standardized, "empirical_quantile")
    if not 0.5 < level < 1.0:
        raise BatteryError(f"empirical_quantile: level must be in (0.5, 1), got {level}")
    return TailQuantile(
        level=float(level),
        empirical=float(np.quantile(series, 1.0 - level)),
        normal=-float(norm.ppf(level)),
    )


# ---------------------------------------------------------------------------
# Rank ordering of risk across members
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankCorrelation:
    """Cross-sectional Spearman between forecast and realised risk, per block."""

    correlations: np.ndarray
    block: int
    members: int

    @property
    def blocks(self) -> int:
        return int(self.correlations.size)

    @property
    def mean(self) -> float:
        return float(self.correlations.mean())

    @property
    def standard_error(self) -> float:
        """Across NON-OVERLAPPING blocks, so ``blocks`` is an honest ``n``."""
        return float(self.correlations.std(ddof=1) / np.sqrt(self.blocks))

    @property
    def fraction_positive(self) -> float:
        return float(np.mean(self.correlations > 0.0))

    @property
    def overlap(self) -> int:
        return 0


def cross_sectional_rank_correlation(
    forecast_volatility: np.ndarray, realised_returns: np.ndarray, *, block: int
) -> RankCorrelation:
    """Does the model rank a risky member above a safe one? This project's own construction.

    NOT Menchero & Ji's Q-statistic. SPEC.md 6.5 names it in one sentence and
    the repository does not hold *JPM* 50(3) 2024, so a statistic built from that
    sentence would claim a provenance it does not have (W4-P3 ruling 5). This is
    the plain version of the same question: in each non-overlapping block of
    ``block`` periods, the Spearman rank correlation across members between the
    forecast volatility standing when the block opened and the realised
    standard deviation of returns inside it. A model with ``B = 1`` that cannot
    tell a risky member from a safe one scores zero here.
    """
    forecast = np.asarray(forecast_volatility, dtype=float)
    realised = np.asarray(realised_returns, dtype=float)
    if forecast.ndim != 2 or forecast.shape != realised.shape:
        raise BatteryError(
            "cross_sectional_rank_correlation: forecast and realised must be T x N of the same "
            f"shape, got {forecast.shape} and {realised.shape}"
        )
    periods, members = forecast.shape
    if members < 3:
        raise BatteryError("cross_sectional_rank_correlation: a rank needs at least 3 members")
    if block < 2 or block > periods:
        raise BatteryError(
            f"cross_sectional_rank_correlation: block {block} does not fit {periods} periods"
        )
    values: list[float] = []
    for start in range(0, periods - block + 1, block):
        realised_risk = realised[start : start + block].std(axis=0, ddof=1)
        correlation = spearmanr(forecast[start], realised_risk).statistic
        values.append(float(correlation))
    return RankCorrelation(correlations=np.asarray(values), block=int(block), members=int(members))


# ---------------------------------------------------------------------------
# Factor-return t-statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorTStatistic:
    """One column's mean-return significance, full sample and rolling."""

    name: str
    mean: float
    newey_west_standard_error: float
    observations: int
    #: Rolling ``t`` over ``window`` periods stepped one period: overlap ``window - 1``.
    rolling: np.ndarray
    window: int
    threshold: float
    #: The same on NON-OVERLAPPING windows -- the only series given an ``n``.
    non_overlapping: np.ndarray

    @property
    def t_statistic(self) -> float:
        return self.mean / self.newey_west_standard_error

    @property
    def overlap(self) -> int:
        return self.window - 1

    @property
    def rolling_exceedance(self) -> float:
        """Fraction of ROLLING windows with ``|t| > threshold``. Not an ``n``."""
        return float(np.mean(np.abs(self.rolling) > self.threshold))

    @property
    def non_overlapping_exceedance(self) -> float:
        return float(np.mean(np.abs(self.non_overlapping) > self.threshold))


def _window_t(values: np.ndarray, *, window: int, step: int) -> np.ndarray:
    count = values.size
    out: list[float] = []
    for start in range(0, count - window + 1, step):
        chunk = values[start : start + window]
        deviation = float(chunk.std(ddof=1))
        out.append(float(chunk.mean() / (deviation / np.sqrt(window))) if deviation > 0 else np.nan)
    return np.asarray(out, dtype=float)


def factor_t_statistics(
    returns: np.ndarray,
    names: Sequence[str],
    *,
    newey_west_lags: int,
    window: int,
    threshold: float,
) -> tuple[FactorTStatistic, ...]:
    """Per column: full-sample mean ``t`` with a Newey-West error, plus rolling ``t``.

    SPEC.md 6.5: "a factor whose returns are indistinguishable from noise is
    adding estimation error to the covariance matrix for nothing." Reported, and
    **nothing is pruned on it** -- choosing risk factors by their mean return
    would be alpha research (CLAUDE.md). The rolling series is plain ``t`` inside
    each window; the exceedance frequency is given for the rolling series with
    its overlap and for non-overlapping windows with their count.
    """
    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise BatteryError(
            f"factor_t_statistics: returns is {matrix.shape} against {len(names)} name(s)"
        )
    if window < 2 or window > matrix.shape[0]:
        raise BatteryError(f"factor_t_statistics: window {window} does not fit {matrix.shape[0]}")
    if not threshold > 0.0:
        raise BatteryError("factor_t_statistics: threshold must be positive")
    results = []
    for column, name in enumerate(names):
        series = _vector(matrix[:, column], f"factor_t_statistics[{name}]")
        results.append(
            FactorTStatistic(
                name=str(name),
                mean=float(series.mean()),
                newey_west_standard_error=float(
                    np.sqrt(newey_west_variance_of_mean(series, lags=newey_west_lags))
                ),
                observations=int(series.size),
                rolling=_window_t(series, window=window, step=1),
                window=int(window),
                threshold=float(threshold),
                non_overlapping=_window_t(series, window=window, step=window),
            )
        )
    return tuple(results)
