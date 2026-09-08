"""SPEC.md 1's identity, the Sharpe statistics of SPEC.md 6.6, and the trial count. W6-P2.

Every function here is arithmetic on a series of NON-OVERLAPPING holding-period
returns; nothing reads data, nothing knows what the assets are.

THE IDENTITY (SPEC.md 1)

    SR_paper = mu_g / sigma_f
    SR_real  = (mu_g - TC) / sigma_r = (mu_g - TC) / (B sigma_f),   B = sigma_r / sigma_f
    SR_paper - SR_real = (mu_g / sigma_f) (1 - 1/B)  +  TC / (B sigma_f)
                         ---- risk-model term ----      -- cost term --

It is exact, and :func:`identity_terms` asserts it to rounding rather than
trusting the algebra. **The identity is put on Lo's scale** (W6-P2b, operator):
returns are annualised by ``periods_per_year`` and volatilities by
``periods_per_year / eta`` with ``eta`` the Lo aggregation factor of the cell's
net series, so that ``SR_paper`` and ``SR_real`` are ``eta`` times their
per-period values -- the same scale the gross and net Sharpes and their standard
errors are on. On a cost-free cell ``gross Sharpe x B = SR_paper`` exactly,
which :mod:`mafrm.backtest.grid` asserts. ``B`` is the ratio of two annualised
volatilities and is scale-free -- the identity's own definition -- and is
reported beside SPEC.md 6.1's bias statistic (the standard deviation of clipped
standardised returns), which is a different estimator of the same quantity; the
two are close and neither is substituted for the other.

SHARPE RATIOS, LO (2002) -- "The Statistics of Sharpe Ratios", *FAJ* 58(4)

    Per period, ``SR = mu / sigma``. Time aggregation to ``q`` periods under
    autocorrelation is Lo's eq. (20), ``SR(q) = eta(q) SR`` with
    ``eta(q) = q / sqrt(q + 2 sum_{k=1}^{q-1} (q - k) rho_k)``, which is
    ``sqrt(q)`` when every ``rho_k`` is zero. The standard error is Lo's GMM
    delta method (his Section "Non-IID Returns"): with moment conditions
    ``g_t = (r_t - mu, (r_t - mu)^2 - sigma^2)``, ``SR = f(mu, sigma^2)``,
    ``Var(SR) = D' S D / T`` where ``D = (1/sigma, -mu / (2 sigma^3))`` and ``S``
    is the Bartlett long-run covariance of ``g_t`` at ``q - 1`` lags -- the same
    lags ``eta(q)`` reads, so the two corrections see the same autocorrelations.
    At zero lags and normal returns it collapses to Lo's iid closed form
    ``sqrt((1 + SR^2 / 2) / T)``, which a test pins. The aggregated standard
    error is ``eta(q)`` times the per-period one, holding ``eta`` fixed; the
    sampling error in ``eta`` is not propagated and the report says so.

DEFLATED SHARPE RATIO -- Bailey & Lopez de Prado (2014), *J. Portfolio Mgmt* 40(5)

    ``DSR = Phi[ (SR - SR_0) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1) SR^2 / 4) ]``
    with ``SR`` per period, ``g3`` the skewness and ``g4`` the (non-excess)
    kurtosis of the returns, and the false-strategy bracket
    ``SR_0 = sqrt(V[SR_n]) [ (1 - g) Z(1 - 1/N) + g Z(1 - 1/(N e)) ]``,
    ``g`` the Euler-Mascheroni constant, ``N`` the honest trial count from
    ``experiments.md`` and ``V[SR_n]`` the variance of the trials' Sharpe
    ratios. The null is ZERO skill (ruling 6, W6-P2). The formula treats the
    trials as independent; a grid whose cells share one alpha is not that, and
    the report states it rather than adjusting ``N`` downward.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis, norm, skew

__all__ = [
    "EULER_MASCHERONI",
    "DeflatedSharpe",
    "IdentityTerms",
    "MetricsError",
    "SharpeEstimate",
    "aggregation_factor",
    "annualised_sharpe",
    "autocorrelations",
    "deflated_sharpe",
    "identity_terms",
    "lo_standard_error",
    "read_trial_count",
    "sharpe_ratio",
]

FloatArray = NDArray[np.float64]

#: Bailey & Lopez de Prado's ``gamma``.
EULER_MASCHERONI: Final[float] = 0.5772156649015329

#: The running-totals line of ``experiments.md`` that carries ``N``.
_TRIAL_COUNT_LINE: Final[re.Pattern[str]] = re.compile(
    r"^\|\s*\*\*Total evaluated\*\*\s*\|\s*\*\*(?P<total>\d+)\*\*\s*\|"
    r"\s*\*\*`N`\s*=\s*(?P<n>\d+)\*\*\s*\|"
)


class MetricsError(ValueError):
    """The inputs cannot carry the statistic asked of them. Nothing is filled or defaulted."""


def _series(values: FloatArray | list[float], *, name: str, minimum: int = 2) -> FloatArray:
    out = np.asarray(values, dtype=float)
    if out.ndim != 1 or out.size < minimum:
        raise MetricsError(f"{name}: expected a vector of at least {minimum}, got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise MetricsError(f"{name}: every return must be finite")
    return out


# ---------------------------------------------------------------------------
# SPEC.md 1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentityTerms:
    """SPEC.md 1's identity for one cell, annualised."""

    #: Gross annualised excess return ``mu_g``.
    gross_return: float
    #: Annual cost drag ``TC`` (a return, not basis points).
    cost_drag: float
    #: Forecast volatility ``sigma_f``: the rms of the per-period forecasts, annualised.
    forecast_volatility: float
    #: Realised volatility ``sigma_r`` of the NET series, annualised.
    realised_volatility: float
    #: ``B = sigma_r / sigma_f``.
    bias_ratio: float
    #: The aggregation factor the volatilities were annualised with (``eta``).
    aggregation: float
    sharpe_paper: float
    sharpe_real: float
    risk_model_term: float
    cost_term: float

    @property
    def gap(self) -> float:
        return self.sharpe_paper - self.sharpe_real


def identity_terms(
    gross: FloatArray,
    net: FloatArray,
    forecast_volatility: FloatArray,
    *,
    periods_per_year: float,
    aggregation: float,
) -> IdentityTerms:
    """The two terms, from per-period gross and net returns and the standing forecast.

    ``gross - net`` is the per-period cost as a fraction of NAV, so ``TC`` is its
    annualised mean and ``mu_g - TC`` is the net mean exactly. Volatilities are
    annualised by ``periods_per_year / aggregation``: with ``aggregation =
    sqrt(periods_per_year)`` that is the naive ``sqrt(q)`` rule, with Lo's
    ``eta(q)`` it is Lo's scale. The identity is asserted at the end, to
    ``1e-12`` of the gap's scale.
    """
    g = _series(gross, name="identity_terms: gross")
    n = _series(net, name="identity_terms: net")
    f = _series(forecast_volatility, name="identity_terms: forecast_volatility")
    if not (g.shape == n.shape == f.shape):
        raise MetricsError("identity_terms: gross, net and forecast_volatility must align")
    if np.any(f <= 0.0):
        raise MetricsError("identity_terms: every forecast volatility must be positive")
    if not periods_per_year > 0.0:
        raise MetricsError(
            f"identity_terms: periods_per_year must be positive, got {periods_per_year}"
        )
    if not (math.isfinite(aggregation) and aggregation > 0.0):
        raise MetricsError(f"identity_terms: aggregation must be positive, got {aggregation}")
    root = periods_per_year / aggregation
    mu_g = float(g.mean() * periods_per_year)
    tc = float((g - n).mean() * periods_per_year)
    sigma_f = float(np.sqrt(np.mean(f**2)) * root)
    sigma_r = float(np.std(n, ddof=1) * root)
    if sigma_r <= 0.0:
        raise MetricsError("identity_terms: the net series has zero realised volatility")
    bias = sigma_r / sigma_f
    sr_paper = mu_g / sigma_f
    sr_real = (mu_g - tc) / sigma_r
    risk_term = sr_paper * (1.0 - 1.0 / bias)
    cost_term = tc / (bias * sigma_f)
    gap = sr_paper - sr_real
    scale = max(abs(sr_paper), abs(sr_real), 1.0)
    if abs(risk_term + cost_term - gap) > 1e-12 * scale:
        raise MetricsError(  # pragma: no cover - the identity is exact
            f"identity_terms: {risk_term} + {cost_term} != {gap}; the identity is exact and this "
            "is a bug"
        )
    return IdentityTerms(
        gross_return=mu_g,
        cost_drag=tc,
        forecast_volatility=sigma_f,
        realised_volatility=sigma_r,
        bias_ratio=bias,
        aggregation=aggregation,
        sharpe_paper=sr_paper,
        sharpe_real=sr_real,
        risk_model_term=risk_term,
        cost_term=cost_term,
    )


# ---------------------------------------------------------------------------
# Lo (2002)
# ---------------------------------------------------------------------------


def sharpe_ratio(returns: FloatArray) -> float:
    """Per-period ``mu / sigma`` with ``ddof = 1``; returns are already excess of cash."""
    r = _series(returns, name="sharpe_ratio")
    sigma = float(np.std(r, ddof=1))
    if sigma <= 0.0:
        raise MetricsError("sharpe_ratio: zero volatility")
    return float(r.mean() / sigma)


def autocorrelations(returns: FloatArray, *, lags: int) -> FloatArray:
    """Sample autocorrelations ``rho_1 .. rho_lags``, the standard biased estimator."""
    r = _series(returns, name="autocorrelations")
    if lags < 0:
        raise MetricsError(f"autocorrelations: lags must be >= 0, got {lags}")
    if lags >= r.size:
        raise MetricsError(f"autocorrelations: {lags} lags need more than {r.size} observations")
    d = r - r.mean()
    denominator = float(d @ d)
    if denominator <= 0.0:
        raise MetricsError("autocorrelations: constant series")
    out = np.array([float(d[k:] @ d[:-k]) / denominator for k in range(1, lags + 1)])
    return out


def aggregation_factor(returns: FloatArray, *, q: int) -> float:
    """Lo (2002) eq. (20): ``eta(q) = q / sqrt(q + 2 sum_{k<q} (q - k) rho_k)``."""
    if q < 1:
        raise MetricsError(f"aggregation_factor: q must be >= 1, got {q}")
    rho = autocorrelations(returns, lags=q - 1)
    weights = np.arange(q - 1, 0, -1, dtype=float)
    inside = q + 2.0 * float(weights @ rho)
    if inside <= 0.0:
        raise MetricsError(
            f"aggregation_factor: q + 2 sum (q-k) rho_k = {inside} <= 0; the sample "
            "autocorrelations imply a negative q-period variance"
        )
    return float(q / math.sqrt(inside))


def lo_standard_error(returns: FloatArray, *, lags: int) -> float:
    """Lo's GMM standard error of the per-period Sharpe ratio, Bartlett kernel at ``lags``.

    ``D' S D / T`` with ``D = (1/sigma, -mu/(2 sigma^3))`` and ``S`` the long-run
    covariance of ``(r - mu, (r - mu)^2 - sigma^2)``. At ``lags = 0`` and normal
    returns this is ``(1 + SR^2/2) / T`` -- Lo's iid result -- because the fourth
    central moment is then ``3 sigma^4`` and ``S = diag(sigma^2, 2 sigma^4)``.
    """
    r = _series(returns, name="lo_standard_error", minimum=3)
    if lags < 0 or lags >= r.size:
        raise MetricsError(f"lo_standard_error: lags {lags} against {r.size} observations")
    t = r.size
    mu = float(r.mean())
    var = float(np.mean((r - mu) ** 2))
    if var <= 0.0:
        raise MetricsError("lo_standard_error: zero volatility")
    sigma = math.sqrt(var)
    g = np.column_stack([r - mu, (r - mu) ** 2 - var])
    s = g.T @ g / t
    for k in range(1, lags + 1):
        gamma_k = g[k:].T @ g[:-k] / t
        weight = 1.0 - k / (lags + 1.0)
        s = s + weight * (gamma_k + gamma_k.T)
    d = np.array([1.0 / sigma, -mu / (2.0 * sigma**3)])
    variance = float(d @ s @ d)
    if variance < 0.0:
        raise MetricsError(f"lo_standard_error: negative variance {variance}; the kernel failed")
    return math.sqrt(variance / t)


@dataclass(frozen=True)
class SharpeEstimate:
    """A per-period Sharpe ratio, its Lo aggregation and both standard errors."""

    per_period: float
    standard_error_per_period: float
    q: int
    eta: float
    lags: int

    @property
    def annualised(self) -> float:
        return self.per_period * self.eta

    @property
    def standard_error(self) -> float:
        """``eta`` times the per-period error, ``eta`` held fixed."""
        return self.standard_error_per_period * self.eta

    @property
    def iid_annualised(self) -> float:
        """The naive ``sqrt(q)`` aggregation, for the report to show what ``eta`` moved."""
        return self.per_period * math.sqrt(self.q)


def annualised_sharpe(returns: FloatArray, *, q: int) -> SharpeEstimate:
    """Lo (2002) on one series: ``SR(q) = eta(q) SR`` with the GMM error at ``q - 1`` lags."""
    r = _series(returns, name="annualised_sharpe", minimum=3)
    return SharpeEstimate(
        per_period=sharpe_ratio(r),
        standard_error_per_period=lo_standard_error(r, lags=q - 1),
        q=q,
        eta=aggregation_factor(r, q=q),
        lags=q - 1,
    )


# ---------------------------------------------------------------------------
# Bailey & Lopez de Prado (2014)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeflatedSharpe:
    """One cell's deflated Sharpe ratio and the bracket it was deflated against."""

    #: Per-period Sharpe ratio of the cell.
    sharpe: float
    #: ``E[max SR_N]`` under zero skill -- the false-strategy bracket, per period.
    expected_maximum: float
    trials: int
    periods: int
    skewness: float
    kurtosis: float
    #: ``Phi[...]``: the probability the cell's Sharpe exceeds the bracket.
    probability: float


def expected_maximum_sharpe(*, trials_variance: float, trials: int) -> float:
    """``sqrt(V) [(1 - g) Z(1 - 1/N) + g Z(1 - 1/(N e))]``, the false-strategy theorem."""
    if trials < 2:
        raise MetricsError(f"expected_maximum_sharpe: N must be >= 2, got {trials}")
    if not (math.isfinite(trials_variance) and trials_variance >= 0.0):
        raise MetricsError(f"expected_maximum_sharpe: V[SR] must be >= 0, got {trials_variance}")
    g = EULER_MASCHERONI
    z1 = float(norm.ppf(1.0 - 1.0 / trials))
    z2 = float(norm.ppf(1.0 - 1.0 / (trials * math.e)))
    return math.sqrt(trials_variance) * ((1.0 - g) * z1 + g * z2)


def deflated_sharpe(returns: FloatArray, *, trials_variance: float, trials: int) -> DeflatedSharpe:
    """Bailey & Lopez de Prado's DSR for one series against ``N`` trials of variance ``V``."""
    r = _series(returns, name="deflated_sharpe", minimum=4)
    sr = sharpe_ratio(r)
    sr0 = expected_maximum_sharpe(trials_variance=trials_variance, trials=trials)
    skewness = float(skew(r, bias=False))
    kurt = float(kurtosis(r, fisher=False, bias=False))
    t = r.size
    inside = 1.0 - skewness * sr + (kurt - 1.0) / 4.0 * sr**2
    if inside <= 0.0:
        raise MetricsError(f"deflated_sharpe: 1 - g3 SR + (g4 - 1) SR^2 / 4 = {inside} <= 0")
    statistic = (sr - sr0) * math.sqrt(t - 1.0) / math.sqrt(inside)
    return DeflatedSharpe(
        sharpe=sr,
        expected_maximum=sr0,
        trials=trials,
        periods=t,
        skewness=skewness,
        kurtosis=kurt,
        probability=float(norm.cdf(statistic)),
    )


# ---------------------------------------------------------------------------
# The trial count
# ---------------------------------------------------------------------------


def read_trial_count(path: Path) -> int:
    """``N`` from ``experiments.md``'s consolidated running-totals line, or a refusal.

    The line is ``| **Total evaluated** | **<total>** | **`N` = <n>** |`` and there
    must be exactly ONE of them: the first block's stale copy was struck through
    on 2026-08-26 and the consolidated total at the foot of the file is the only
    one maintained. Two matches would mean the count had been duplicated, which
    is the drift the ruling exists to prevent.
    """
    if not path.exists():
        raise MetricsError(f"read_trial_count: {path} does not exist")
    matches = [
        m
        for line in path.read_text(encoding="utf-8").splitlines()
        if (m := _TRIAL_COUNT_LINE.match(line))
    ]
    if len(matches) != 1:
        raise MetricsError(
            f"read_trial_count: expected exactly one '**Total evaluated**' line in {path.name}, "
            f"found {len(matches)}"
        )
    n = int(matches[0].group("n"))
    if n < 1:
        raise MetricsError(f"read_trial_count: N = {n} is not a trial count")
    return n
