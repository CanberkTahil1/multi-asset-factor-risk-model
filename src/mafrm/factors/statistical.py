"""Model B, the statistical factor model. SPEC.md 4.2.

PCA on the sample correlation matrix of daily excess returns, with
Marchenko-Pastur denoising before extraction, on an expanding window. The
surviving components -- those whose eigenvalue clears the noise band's upper
edge -- are Model B's factors, and they feed the same ``mafrm.risk`` pipeline
Model A does, unchanged. That last clause is the point of the exercise: SPEC.md
15.2's architectural instruction says a second factor set must cost nothing in
``risk/``, and Model B is the first thing in the project that tests it.

FIVE THINGS ARE LOAD-BEARING.

**The correlation is the equally-weighted SAMPLE correlation, not the EWMA one**
(SPEC.md 4.2.1). Marchenko-Pastur is derived for ``T`` equally-weighted iid
observations and ``lambda_+ = sigma^2 (1 + sqrt(N/T))^2`` takes that ``T``
directly. Under EWMA weights the edge would need an effective sample size, and
W3-P3b measured that ``2*tau/ln2`` does not transfer to a normalised correlation
estimator -- off by 1.31x, and not by a constant. Using MP on an EWMA
correlation would be applying a formula outside its derivation with a
known-wrong ``T``.

**The window is expanding, and the burn-in is not a new number.** A full-sample
eigendecomposition uses the whole history to decide what the factor was on its
first day, which is look-ahead; SPEC.md 4.1.3 already ruled that out for the
rates PCA. The minimum is read from ``factors.macro.rates.pca`` -- the same
concept, in the same namespace. SPEC.md 5.1.2's prohibition is on importing a
day count into ``risk/``, and that direction stays closed.

**N is 13.** SPEC.md 4.2 says "15x15" and "expect 3-5 to survive"; the frozen
universe has 14 lines of which ``dollar`` is not investable. Both numbers were
written for a sleeve this project is not in and are void rather than adjustable
(SPEC.md 4.2.2). ``N/T`` is emitted on every date beside the fitted
``lambda_+``, because it is the Marchenko-Pastur parameter and it is the same
ratio the project's headline is about.

**Eigenvector signs are arbitrary and are fixed deliberately** (SPEC.md 4.2.3).
An eigendecomposition run once per date on a window that grew by one row can
return ``-v`` where it returned ``+v`` yesterday, which would flip the factor's
sign for a day and put a spurious ``-2f`` into its return series. Each date's
loadings are oriented against the previous date's; the first is oriented so its
largest-magnitude loading is positive. The flip count is reported, not hidden.

**The factor is a linear combination of RAW returns.** ``f_kt = (v_k / s_t)'
r_t``, where ``s_t`` are the expanding-window standard deviations that
standardise the panel for the correlation. The correlation estimator demeans, as
the sample correlation does; the projection does not. That is deliberate and it
is what keeps SPEC.md 5.1.1's zero-mean convention coherent downstream -- the
covariance pipeline forecasts variance about zero because SPEC.md 6.1's bias
statistic has a raw return in its numerator, and a factor built as a demeaned
combination would reintroduce exactly the mismatch that ruling closed.

This module reads the cache only. CLAUDE.md invariant 1: no network call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from mafrm.config import Config, StatisticalFactorConfig, load
from mafrm.factors import betas

__all__ = [
    "ComparandEstimate",
    "DenoisedCorrelation",
    "NoiseFit",
    "StatisticalFactorError",
    "StatisticalFactorPanel",
    "asset_panel",
    "denoise_correlation",
    "detone_correlation",
    "expanding_correlations",
    "fit_noise_variance",
    "gaussian_kde_pdf",
    "ledoit_wolf_constant_correlation",
    "marchenko_pastur_edges",
    "marchenko_pastur_pdf",
    "oracle_approximating_shrinkage",
    "orient_eigenvectors",
    "statistical_factor_panel",
]


class StatisticalFactorError(ValueError):
    """Model B cannot be estimated from what it was handed."""


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------


def asset_panel(
    *, start: date | None = None, end: date | None = None, config: Config | None = None
) -> pd.DataFrame:
    """Daily excess returns for the investable universe, complete cases only.

    ``start`` defaults to ``sample.start`` and ``end`` to ``holdout_start``;
    nothing on or after the holdout is ever read (CLAUDE.md invariant 5, enforced
    beneath this in ``mafrm.data.cache``).

    **Complete cases, not a filled panel.** A sample correlation over a ragged
    panel is either pairwise -- which is not guaranteed positive semi-definite
    and would hand a non-PSD matrix to an eigendecomposition that has no way to
    report it -- or it is filled, which invents returns. The universe's members
    have different holiday calendars (CLAUDE.md failure mode 1): the four
    synthetic curve points follow the Fed's business calendar and the ETFs follow
    the exchange's, so a handful of dates are dropped rather than reconciled.
    How many is reported, not swallowed.
    """
    settings = config or load()
    boundary = end if end is not None else settings.require_holdout_start()
    window_start = start if start is not None else settings.model.sample.start

    frame = pd.DataFrame(betas.asset_excess_returns(config=settings))
    frame = frame.loc[
        (frame.index >= pd.Timestamp(window_start)) & (frame.index < pd.Timestamp(boundary))
    ]
    complete = frame.dropna(how="any")
    if complete.empty:
        raise StatisticalFactorError(
            f"no date between {window_start} and {boundary} has all {frame.shape[1]} assets"
        )
    return complete


# ---------------------------------------------------------------------------
# Marchenko-Pastur
# ---------------------------------------------------------------------------


def marchenko_pastur_edges(*, noise_variance: float, ratio: float) -> tuple[float, float]:
    """``(lambda_-, lambda_+) = sigma^2 (1 -+ sqrt(N/T))^2``. SPEC.md 4.2.

    ``ratio`` is ``N/T`` -- variables over observations -- which is the form
    SPEC.md 4.2 writes the edge in and the same ratio the project's headline
    result is stated in.
    """
    if noise_variance <= 0.0:
        raise StatisticalFactorError(
            f"marchenko_pastur_edges: sigma^2 must be positive, got {noise_variance}"
        )
    if ratio <= 0.0:
        raise StatisticalFactorError(f"marchenko_pastur_edges: N/T must be positive, got {ratio}")
    root = float(np.sqrt(ratio))
    return noise_variance * (1.0 - root) ** 2, noise_variance * (1.0 + root) ** 2


def marchenko_pastur_pdf(grid: np.ndarray, *, noise_variance: float, ratio: float) -> np.ndarray:
    """The Marchenko-Pastur density on ``grid``. Zero outside ``[lambda_-, lambda_+]``.

    ``f(x) = sqrt((lambda_+ - x)(x - lambda_-)) / (2 pi sigma^2 (N/T) x)``.

    Lopez de Prado, *Machine Learning for Asset Managers*, ch. 2, writes this with
    ``q = T/N`` in the numerator; ``q = 1/(N/T)`` makes the two identical, and the
    ``N/T`` form is used here so the same ratio appears everywhere in this
    project.
    """
    lower, upper = marchenko_pastur_edges(noise_variance=noise_variance, ratio=ratio)
    values = np.zeros_like(grid, dtype=float)
    inside = (grid > lower) & (grid < upper) & (grid > 0.0)
    x = grid[inside]
    values[inside] = np.sqrt((upper - x) * (x - lower)) / (2.0 * np.pi * noise_variance * ratio * x)
    return values


def gaussian_kde_pdf(sample: np.ndarray, grid: np.ndarray, *, bandwidth: float) -> np.ndarray:
    """Gaussian kernel density of ``sample``, evaluated on ``grid``.

    **The bandwidth is ABSOLUTE**, in the units of ``sample``. That is the
    convention of ``sklearn.neighbors.KernelDensity``, which is what SPEC.md
    4.2's cited source (Lopez de Prado ch. 2) uses when it says "KDE bandwidth
    0.15", and it is why this is fifteen lines here rather than a call to
    ``scipy.stats.gaussian_kde``: scipy's ``bw_method`` is a MULTIPLIER on the
    sample standard deviation, so passing 0.15 to it would silently mean a
    different width -- and a different one at every date, since the eigenvalue
    spectrum's spread changes as the window grows.
    """
    if bandwidth <= 0.0:
        raise StatisticalFactorError(
            f"gaussian_kde_pdf: bandwidth must be positive, got {bandwidth}"
        )
    if sample.size == 0:
        raise StatisticalFactorError("gaussian_kde_pdf: an empty sample has no density")
    standardized = (grid[:, None] - sample[None, :]) / bandwidth
    kernel = np.exp(-0.5 * np.square(standardized)) / np.sqrt(2.0 * np.pi)
    density: np.ndarray = kernel.sum(axis=1) / (sample.size * bandwidth)
    return density


@dataclass(frozen=True)
class NoiseFit:
    """What the ``sigma^2`` fit found, and enough to say whether to believe it."""

    #: The fitted noise variance. NOT assumed to be 1 -- SPEC.md 4.2 is explicit.
    noise_variance: float
    #: ``lambda_+``, the noise band's upper edge. Everything below it is noise.
    lambda_plus: float
    lambda_minus: float
    #: ``N/T``. The Marchenko-Pastur parameter.
    ratio: float
    #: Squared error between the empirical KDE and the fitted MP density.
    sum_squared_error: float
    #: Whether the optimizer converged. A failure is reported, never defaulted.
    converged: bool

    @property
    def band_half_width(self) -> float:
        """``sqrt(N/T)`` -- how wide the noise band is, as a fraction of ``sigma^2``.

        The number that makes SPEC.md 4.2's "expect 3-5 to survive" void rather
        than merely wrong: at a small ``N/T`` the band is a narrow collar around
        ``sigma^2`` and almost any real structure clears it.
        """
        return float(np.sqrt(self.ratio))


def fit_noise_variance(
    eigenvalues: np.ndarray, *, ratio: float, settings: StatisticalFactorConfig
) -> NoiseFit:
    """Fit ``sigma^2`` to the empirical eigenvalue spectrum. SPEC.md 4.2.

    Lopez de Prado, *Machine Learning for Asset Managers*, ch. 2. Minimises the
    squared error between a Gaussian KDE of the observed eigenvalues and the
    Marchenko-Pastur density, both evaluated on a grid spanning the candidate
    band. **``sigma^2`` is fitted rather than assumed to be 1**, which SPEC.md
    4.2 calls out specifically: assuming 1 puts the whole of any real factor
    variance into the noise band's scale and shifts ``lambda_+`` with it.

    The search is bounded by ``noise_variance_bounds``, and that bound is
    derived rather than chosen: the estimator is a CORRELATION matrix, so every
    variable has unit variance and a noise component cannot carry more than all
    of it.
    """
    if eigenvalues.ndim != 1:
        raise StatisticalFactorError(
            f"fit_noise_variance: expected a 1-D spectrum, got {eigenvalues.shape}"
        )
    low, high = settings.noise_variance_bounds

    def objective(noise_variance: float) -> float:
        lower, upper = marchenko_pastur_edges(noise_variance=noise_variance, ratio=ratio)
        grid = np.linspace(lower, upper, settings.kde_grid_points)
        theoretical = marchenko_pastur_pdf(grid, noise_variance=noise_variance, ratio=ratio)
        empirical = gaussian_kde_pdf(eigenvalues, grid, bandwidth=settings.kde_bandwidth)
        return float(np.sum(np.square(empirical - theoretical)))

    result = minimize_scalar(objective, bounds=(low, high), method="bounded")
    noise_variance = float(result.x)
    lower, upper = marchenko_pastur_edges(noise_variance=noise_variance, ratio=ratio)
    return NoiseFit(
        noise_variance=noise_variance,
        lambda_plus=upper,
        lambda_minus=lower,
        ratio=ratio,
        sum_squared_error=float(result.fun),
        converged=bool(result.success),
    )


# ---------------------------------------------------------------------------
# Denoising and detoning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DenoisedCorrelation:
    """A denoised correlation matrix and what the denoising did to it."""

    #: The denoised correlation. Unit diagonal, symmetric, PSD.
    correlation: np.ndarray
    #: Eigenvalues before replacement, descending.
    eigenvalues: np.ndarray
    #: Eigenvalues after replacement, descending. The noise ones are equal.
    #: BEFORE the unit-diagonal rescale, which is the spectrum ``survivors``
    #: counts against.
    denoised_eigenvalues: np.ndarray
    #: Eigenvectors of the INPUT correlation, descending, sign-oriented by the
    #: caller. Constant-average replacement does not rotate anything, so these
    #: are also the denoised matrix's own eigenvectors -- see
    #: :func:`denoise_correlation` for why they are NOT taken from
    #: :attr:`correlation`.
    eigenvectors: np.ndarray
    #: How many eigenvalues cleared ``lambda_+``. Model B's ``K``.
    survivors: int
    fit: NoiseFit
    #: ``max |diag - 1|`` BEFORE the unit-diagonal rescale. Reported because the
    #: rescale is the one step SPEC.md 4.2 does not spell out; see the note in
    #: :func:`denoise_correlation`.
    diagonal_deviation: float
    #: How many eigenvalues clear ``lambda_+`` in :attr:`correlation`, i.e. AFTER
    #: the rescale. Carried because it is **not always** :attr:`survivors`: the
    #: rescale moves the spectrum, and on this project's panel the two agree
    #: everywhere for the denoised variant and disagree on 12.7% of dates for the
    #: detoned one. Reported so the divergence is visible rather than being a
    #: silent difference between the count and the directions.
    survivors_after_rescale: int

    @property
    def variance_share(self) -> np.ndarray:
        """Each surviving component's share of total variance, i.e. of ``N``."""
        return self.eigenvalues[: self.survivors] / float(self.eigenvalues.size)


def _oriented_eigendecomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Descending eigenvalues and their eigenvectors, from a symmetric matrix."""
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def denoise_correlation(correlation: np.ndarray, *, fit: NoiseFit) -> DenoisedCorrelation:
    """Replace every eigenvalue below ``lambda_+`` with their constant average.

    SPEC.md 4.2: *"Replace all eigenvalues below ``lambda_+`` with their constant
    average, preserving trace. Keep the components above ``lambda_+`` as
    factors."* The average is over the replaced eigenvalues only, so their sum --
    and therefore the trace -- is unchanged by construction.

    **The unit-diagonal rescale is a step SPEC.md 4.2 does not mention, and it is
    made visible rather than folded in.** Replacing eigenvalues rotates nothing
    but does move the diagonal off 1, so the result is not a correlation matrix
    until it is rescaled -- and the object has to remain one, because downstream
    it is recombined with volatilities. The rescale preserves the trace too (a
    unit diagonal on ``N`` variables has trace ``N``, which is what the original
    correlation had), so SPEC.md 4.2's stated constraint holds either way.
    ``diagonal_deviation`` records how far the rescale had to move anything, so a
    reader can see the size of a step the spec left implicit instead of taking
    it on trust. Lopez de Prado ch. 2 applies the same rescale.

    **THE EIGENVECTORS RETURNED ARE THE INPUT'S, NOT THE RESCALED MATRIX'S, AND
    THAT IS THE WHOLE OF WHAT "DENOISING BEFORE EXTRACTION" MEANS HERE.**
    Replacing eigenvalues with their constant average changes ``Lambda`` and
    leaves ``V`` alone, so **constant-average denoising rotates nothing** -- it is
    purely a rule for how many components to keep, not for what they are. SPEC.md
    4.2's phrasing suggests otherwise and it is worth being explicit that it does
    not.

    The rescale is the step that *would* rotate them, and it is measurably not
    small: on this project's panel it turns the leading directions by up to
    ``1 - |cos| = 0.10``, and it moves the count across ``lambda_+`` on **0 of
    4,170 dates for the denoised variant and 528 of 4,170 (12.7%) for the
    detoned one**, where deflating first leaves a zero eigenvalue for the
    rescale to act on. So the two objects are kept apart deliberately. :attr:`survivors` and
    :attr:`eigenvectors` both come from the **pre-rescale** spectrum, which makes
    the factor extraction self-consistent -- a count taken on one spectrum and
    directions taken on another would be neither. :attr:`correlation` is the
    rescaled matrix, which is what a caller needs when it wants a *correlation
    matrix* rather than a set of components, and
    :attr:`survivors_after_rescale` reports where the two diverge.
    """
    values, vectors = _oriented_eigendecomposition(correlation)
    survivors = int(np.sum(values > fit.lambda_plus))
    denoised = values.copy()
    if survivors < values.size:
        noise = denoised[survivors:]
        denoised[survivors:] = noise.sum() / float(noise.size)

    rebuilt = vectors @ np.diag(denoised) @ vectors.T
    rebuilt = 0.5 * (rebuilt + rebuilt.T)
    deviation = float(np.max(np.abs(np.diag(rebuilt) - 1.0)))
    scale = np.sqrt(np.diag(rebuilt))
    rescaled = rebuilt / np.outer(scale, scale)
    np.fill_diagonal(rescaled, 1.0)
    rescaled = 0.5 * (rescaled + rescaled.T)

    return DenoisedCorrelation(
        correlation=rescaled,
        eigenvalues=values,
        denoised_eigenvalues=denoised,
        eigenvectors=vectors,
        survivors=survivors,
        fit=fit,
        diagonal_deviation=deviation,
        survivors_after_rescale=int(np.sum(np.linalg.eigvalsh(rescaled) > fit.lambda_plus)),
    )


def detone_correlation(correlation: np.ndarray, *, components: int) -> np.ndarray:
    """Remove the leading ``components`` eigenvectors, then restore a unit diagonal.

    SPEC.md 4.2's detoned variant: *"remove the first eigenvector -- the 'market'
    -- before denoising"*.

    **The order differs from Lopez de Prado's and SPEC.md's is followed.** ch. 2
    detones the already-denoised correlation; SPEC.md 4.2 detones first, so the
    ``sigma^2`` fit sees the deflated spectrum rather than one still dominated by
    a market eigenvalue that has already been declared signal. The divergence is
    recorded at SPEC.md 4.2.4 rather than left to be rediscovered as a
    discrepancy against the book.

    The deflated matrix is singular by construction -- ``components`` of its
    eigenvalues are zero -- and stays that way here. Denoising is what removes
    the singularity afterwards: a zero eigenvalue is below ``lambda_+`` and is
    absorbed into the constant average with the rest of the noise.
    """
    values, vectors = _oriented_eigendecomposition(correlation)
    if components >= values.size:
        raise StatisticalFactorError(
            f"detone_correlation: removing {components} of {values.size} components leaves "
            "nothing to denoise"
        )
    lead = vectors[:, :components]
    deflated = correlation - lead @ np.diag(values[:components]) @ lead.T
    deflated = 0.5 * (deflated + deflated.T)
    scale = np.sqrt(np.clip(np.diag(deflated), a_min=1e-300, a_max=None))
    deflated = deflated / np.outer(scale, scale)
    np.fill_diagonal(deflated, 1.0)
    return cast("np.ndarray", 0.5 * (deflated + deflated.T))


# ---------------------------------------------------------------------------
# The comparands -- SPEC.md 4.2, "the standard benchmark to beat"
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparandEstimate:
    """A shrinkage estimator's correlation matrix and its shrinkage intensity."""

    name: str
    correlation: np.ndarray
    #: The shrinkage weight on the target, in ``[0, 1]``.
    intensity: float
    #: What it shrinks toward, in words. The whole point of the LW note below.
    target: str
    #: The shrunk COVARIANCE in the input's units, where the estimator produces
    #: one (Ledoit-Wolf); ``None`` for a correlation-only comparand.
    covariance: np.ndarray | None = None


def ledoit_wolf_constant_correlation(
    returns: np.ndarray, *, centre: bool = True
) -> ComparandEstimate:
    """Ledoit-Wolf (2004) shrinkage toward the CONSTANT-CORRELATION target.

    Ledoit & Wolf, "Honey, I Shrunk the Sample Covariance Matrix", *Journal of
    Portfolio Management* 30(4), 2004.

    **``sklearn.covariance.LedoitWolf`` implements the IDENTITY target, not this
    one.** They are different estimators with different targets and different
    optimal intensities, and the identity-target version is the one most replicas
    reach for because it is the one that ships. SPEC.md 4.2 calls this out
    specifically. This function implements the constant-correlation target of the
    2004 JPM paper: every off-diagonal correlation replaced by the average
    off-diagonal correlation, diagonal variances left alone. It is written here
    rather than imported because the version that would be imported is the wrong
    one -- and because ``sklearn`` is not a dependency of this project.

    Returns a CORRELATION matrix, so that it is comparable with the denoised
    correlation of SPEC.md 4.2 rather than with a covariance in different units;
    the shrunk COVARIANCE is carried beside it (W6-P2), because the constant-
    correlation target leaves the diagonal at the sample variances, so the
    shrunk covariance is exactly ``D corr D`` with ``D`` the sample deviations.

    ``centre=False`` takes the moments about ZERO instead of the sample mean --
    SPEC.md 5.1.1's convention for every covariance the optimizer consumes
    (W6-P2's grid variant 7). The default is unchanged for the SPEC.md 4.2
    comparison this function was written for.
    """
    periods, variables = returns.shape
    if periods < 2:
        raise StatisticalFactorError(
            f"ledoit_wolf_constant_correlation: {periods} observation(s) is not a sample"
        )
    centred = returns - returns.mean(axis=0) if centre else np.asarray(returns, dtype=float)
    sample = centred.T @ centred / periods
    deviations = np.sqrt(np.diag(sample))
    correlation = sample / np.outer(deviations, deviations)

    off_diagonal = ~np.eye(variables, dtype=bool)
    mean_correlation = float(correlation[off_diagonal].mean())
    target = mean_correlation * np.outer(deviations, deviations)
    np.fill_diagonal(target, np.diag(sample))

    # pi: sum of asymptotic variances of the sample covariance entries.
    squared = np.square(centred)
    pi_matrix = squared.T @ squared / periods - np.square(sample)
    pi_hat = float(pi_matrix.sum())

    # rho: pi's diagonal, plus the covariance between each sample entry and the
    # target it is shrunk toward. Eq. (5) of the 2004 paper's appendix.
    third = (centred**3).T @ centred / periods - sample * np.diag(sample)[:, None]
    ratio = deviations[None, :] / deviations[:, None]
    rho_off = 0.5 * mean_correlation * (ratio * third + ratio.T * third.T)
    rho_hat = float(np.trace(pi_matrix)) + float(rho_off[off_diagonal].sum())

    gamma_hat = float(np.square(target - sample).sum())
    if gamma_hat <= 0.0:
        intensity = 0.0
    else:
        intensity = max(0.0, min(1.0, (pi_hat - rho_hat) / gamma_hat / periods))

    shrunk = intensity * target + (1.0 - intensity) * sample
    shrunk_deviations = np.sqrt(np.diag(shrunk))
    shrunk_correlation = shrunk / np.outer(shrunk_deviations, shrunk_deviations)
    np.fill_diagonal(shrunk_correlation, 1.0)
    return ComparandEstimate(
        name="ledoit_wolf",
        correlation=0.5 * (shrunk_correlation + shrunk_correlation.T),
        intensity=intensity,
        target="constant correlation (Ledoit-Wolf 2004 JPM), NOT sklearn's identity target",
        covariance=0.5 * (shrunk + shrunk.T),
    )


def oracle_approximating_shrinkage(returns: np.ndarray) -> ComparandEstimate:
    """Oracle Approximating Shrinkage toward a scaled identity. Chen et al. (2010).

    Chen, Wiesel, Eldar & Hero, "Shrinkage Algorithms for MMSE Covariance
    Estimation", *IEEE Transactions on Signal Processing* 58(10), 2010, Eq. (23).

    The third point of comparison SPEC.md 4.2 asks for. Its target is the scaled
    identity ``(tr(S)/N) I`` -- the same family as ``sklearn``'s Ledoit-Wolf and
    a different one from :func:`ledoit_wolf_constant_correlation`, which is why
    both are here.
    """
    periods, variables = returns.shape
    if periods < 2:
        raise StatisticalFactorError(
            f"oracle_approximating_shrinkage: {periods} observation(s) is not a sample"
        )
    centred = returns - returns.mean(axis=0)
    sample = centred.T @ centred / periods
    mu = float(np.trace(sample)) / variables
    target = mu * np.eye(variables)

    squared_norm = float(np.square(sample).sum())
    trace_squared = float(np.trace(sample)) ** 2
    numerator = (1.0 - 2.0 / variables) * squared_norm + trace_squared
    denominator = (periods + 1.0 - 2.0 / variables) * (squared_norm - trace_squared / variables)
    intensity = 1.0 if denominator <= 0.0 else max(0.0, min(1.0, numerator / denominator))

    shrunk = intensity * target + (1.0 - intensity) * sample
    deviations = np.sqrt(np.diag(shrunk))
    correlation = shrunk / np.outer(deviations, deviations)
    np.fill_diagonal(correlation, 1.0)
    return ComparandEstimate(
        name="oas",
        correlation=0.5 * (correlation + correlation.T),
        intensity=intensity,
        target="scaled identity (Chen et al. 2010, Eq. 23)",
    )


# ---------------------------------------------------------------------------
# The expanding window
# ---------------------------------------------------------------------------


def expanding_correlations(
    values: np.ndarray, *, min_window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample correlation and standard deviations of ``values[:i+1]``, per date.

    Returns ``(correlations, deviations, defined)``. Running sums rather than a
    fresh ``np.corrcoef`` per date, for the same reason
    ``mafrm.factors.macro._expanding_covariances`` uses them: the latter is
    quadratic in the number of dates and this runs over about 4,400 of them. A
    test checks the recursion against ``np.corrcoef`` at several dates.

    The covariance is DEMEANED and divided by ``T - 1``, which is the sample
    correlation SPEC.md 4.2 asks for. That is not in tension with SPEC.md 5.1.1's
    zero-mean ruling: that ruling governs the FORECAST variance the bias
    statistic divides by, and this is an eigenvector-direction estimator whose
    output is a set of portfolio weights. What must stay coherent with 5.1.1 is
    the factor series itself, and :func:`statistical_factor_panel` builds that
    from raw returns.
    """
    n_obs, n_cols = values.shape
    if min_window < 2:
        raise StatisticalFactorError(
            f"expanding_correlations: min_window must be at least 2, got {min_window}"
        )
    sum_x = np.zeros(n_cols)
    sum_xx = np.zeros((n_cols, n_cols))
    correlations = np.full((n_obs, n_cols, n_cols), np.nan)
    deviations = np.full((n_obs, n_cols), np.nan)
    for index in range(n_obs):
        row = values[index]
        sum_x += row
        sum_xx += np.outer(row, row)
        count = index + 1
        if count < min_window:
            continue
        mean = sum_x / count
        covariance = (sum_xx - count * np.outer(mean, mean)) / (count - 1)
        sigma = np.sqrt(np.diag(covariance))
        if np.any(sigma <= 0.0):
            raise StatisticalFactorError(
                f"expanding_correlations: a zero-variance column at position {index}; a "
                "correlation is undefined there and is refused rather than floored"
            )
        deviations[index] = sigma
        correlations[index] = covariance / np.outer(sigma, sigma)
    defined = np.arange(n_obs) >= min_window - 1
    return correlations, deviations, defined


def orient_eigenvectors(
    vectors: np.ndarray, previous: np.ndarray | None, *, survivors: int
) -> tuple[np.ndarray, int, float]:
    """Fix each eigenvector's arbitrary sign. SPEC.md 4.2.3.

    Against the previous date's loadings where there are any -- the sign that
    makes the two agree -- and otherwise so that the largest-magnitude loading is
    positive. The seed convention is chosen for having no near-zero denominator:
    the largest absolute element of a unit vector in ``R^N`` is at least
    ``1/sqrt(N)``, so it cannot be the coin-flip that a "positive sum" convention
    would be for a component whose loadings cancel.

    Returns ``(oriented, flips, alignment)``, and the two statistics measure
    different things:

    * ``flips`` counts how often ``numpy.linalg.eigh``'s output disagreed in
      sign with the **accumulated** orientation. It is **path-dependent and not
      interpretable on its own**: ``eigh``'s sign is deterministic for a given
      matrix but is not continuous in it, so one genuine discontinuity leaves the
      convention negated relative to LAPACK's for every date after it, and the
      count then measures how long the two have been out of step rather than how
      often anything went wrong. It is carried as provenance. **Never read it as
      a per-date instability rate** -- the next statistic is the one for that.
    * ``alignment`` is the smallest ``|v_prev . v|`` across the **surviving**
      components after orientation, and it is the statistic that can actually go
      wrong. It sits near 1 while each component keeps its identity from one date
      to the next, and falls toward 0 where two eigenvalues cross and the
      components swap -- which a sign convention cannot fix and must not hide.
      ``1.0`` on the first date, which has nothing to align against.

      It is restricted to the survivors on purpose. Every eigenvector is
      oriented, because the accumulated convention has to stay consistent for all
      of them, but the noise block's eigenvalues are nearly equal, so its
      directions are close to arbitrary and swap constantly. Including them would
      report a permanent instability in a subspace no factor is taken from.
    """
    oriented = vectors.copy()
    flips = 0
    alignment = 1.0
    for column in range(vectors.shape[1]):
        vector = vectors[:, column]
        if previous is not None and column < previous.shape[1]:
            statistic = float(previous[:, column] @ vector)
            if column < survivors:
                alignment = min(alignment, abs(statistic))
        else:
            statistic = float(vector[int(np.argmax(np.abs(vector)))])
        if statistic < 0.0:
            oriented[:, column] = -vector
            flips += 1
    return oriented, flips, alignment


@dataclass(frozen=True)
class StatisticalFactorPanel:
    """Model B: its factor returns, its loadings, and what it was estimated from.

    ``factors`` is the ``T x K`` frame the covariance pipeline consumes, where
    ``K`` is the largest survivor count the window ever reached; columns beyond a
    date's own survivor count are NaN on that date rather than zero, so a date's
    burn-in is visible instead of being filled with a fake observation.

    ``diagnostics`` carries, per date, what SPEC.md 4.2's report has to state:
    ``N``, ``T``, ``N/T``, the fitted ``sigma^2``, ``lambda_+`` and the survivor
    count. ``N/T`` is there because it is the Marchenko-Pastur parameter and
    therefore the same governing ratio as the project's headline -- appearing
    this time inside a published formula rather than inferred from one.
    """

    variant: str
    factors: pd.DataFrame
    diagnostics: pd.DataFrame
    #: ``(date, asset, component)`` loadings, standardised units.
    loadings: dict[str, pd.DataFrame]
    #: The full ``T x N`` eigenvalue spectrum, descending per row. Kept because
    #: the report's figure is about where ``lambda_+`` falls INSIDE the spectrum,
    #: and a chart of the edge alone cannot show that.
    spectrum: pd.DataFrame
    assets: tuple[str, ...]
    #: How often ``eigh``'s sign disagreed with the accumulated orientation.
    #: Path-dependent and not a defect rate; see :func:`orient_eigenvectors`.
    sign_flips: int

    @property
    def variables(self) -> int:
        """``N``. 13, not SPEC.md 4.2's 15 -- see SPEC.md 4.2.2."""
        return len(self.assets)

    @property
    def complete(self) -> pd.DataFrame:
        """Dates on which every factor column exists. What ``risk/`` is handed.

        **This is the one place the panel looks forward, and it is named rather
        than buried.** ``factors`` is as wide as the largest survivor count the
        window ever reached, so a date whose own ``K`` fell short of that maximum
        carries NaN in the surplus columns and is dropped here. No factor VALUE
        depends on anything after its own date -- ``tests/test_statistical.py``
        asserts that by perturbation -- but the SET of dates does, in exactly the
        way :func:`asset_panel`'s complete-case rule does. What it costs is
        :attr:`dates_dropped_by_completeness`, which the report prints.
        """
        return self.factors.dropna(how="any")

    @property
    def dates_dropped_by_completeness(self) -> int:
        """How many dates :attr:`complete` removes because their ``K`` was short."""
        return len(self.factors) - len(self.complete)

    @property
    def survivor_counts(self) -> pd.Series:
        return self.diagnostics["survivors"]

    def render(self) -> tuple[str, ...]:
        counts = self.survivor_counts
        return (
            f"Model B [{self.variant}]: N={self.variables} assets, "
            f"{len(self.diagnostics):,} dates {self.diagnostics.index[0].date()}.."
            f"{self.diagnostics.index[-1].date()}",
            f"  N/T from {self.diagnostics['n_over_t'].max():.4f} down to "
            f"{self.diagnostics['n_over_t'].min():.4f}; "
            f"sigma^2 {self.diagnostics['noise_variance'].min():.4f}.."
            f"{self.diagnostics['noise_variance'].max():.4f}; "
            f"lambda_+ {self.diagnostics['lambda_plus'].min():.4f}.."
            f"{self.diagnostics['lambda_plus'].max():.4f}",
            f"  survivors K: min {counts.min()}, median {counts.median():.0f}, max {counts.max()}",
            f"  sign convention: {self.sign_flips:,} disagreements with eigh "
            f"(path-dependent, not a defect rate); component alignment "
            f"min {self.diagnostics['component_alignment'].min():.4f}, "
            f"median {self.diagnostics['component_alignment'].median():.4f}; "
            f"{self.dates_dropped_by_completeness} date(s) dropped by completeness",
        )


def statistical_factor_panel(
    panel: pd.DataFrame | None = None,
    *,
    variant: str = "denoised",
    config: Config | None = None,
) -> StatisticalFactorPanel:
    """Build Model B on an expanding window. SPEC.md 4.2.

    At each date ``t``, in this order: the sample correlation of the panel
    through ``t`` inclusive is formed; detoned first when
    ``variant == "detoned"``; ``sigma^2`` is fitted to its spectrum and
    ``lambda_+`` follows; the eigenvalues below ``lambda_+`` are replaced by
    their constant average; and the components above it are kept as factors. The
    factor value at ``t`` is that date's return vector projected onto those
    directions. Nothing after ``t`` is used, and
    ``tests/test_statistical.py`` asserts that by perturbation rather than by
    inspection.

    **The directions come from the pre-rescale eigenvectors**, which are the
    input correlation's own -- see :func:`denoise_correlation` for why that is
    both the faithful reading of SPEC.md 4.2 and the only self-consistent choice
    once the unit-diagonal rescale is known to move the spectrum.

    **The projection is of RAW returns.** ``f_kt = (v_k / s_t)' r_t``: the
    eigenvector divided elementwise by the expanding-window standard deviations
    that the correlation was taken with respect to, applied to the raw return
    vector. So a factor is a fixed portfolio of raw excess returns on each date,
    which is what SPEC.md 5.1.1's zero-mean convention requires of anything the
    covariance pipeline is handed.
    """
    settings = config or load()
    scheme = settings.model.factors.statistical
    if variant not in scheme.variants:
        raise StatisticalFactorError(
            f"statistical_factor_panel: variant {variant!r} is not one of {list(scheme.variants)}"
        )
    frame = panel if panel is not None else asset_panel(config=settings)
    values = frame.to_numpy(dtype=float)
    n_obs, n_vars = values.shape
    if n_obs < scheme.expanding_min_window:
        raise StatisticalFactorError(
            f"statistical_factor_panel: {n_obs} observation(s) is below the "
            f"{scheme.expanding_min_window}-day expanding-window minimum"
        )

    correlations, deviations, defined = expanding_correlations(
        values, min_window=scheme.expanding_min_window
    )

    rows: list[dict[str, float]] = []
    dates: list[pd.Timestamp] = []
    projections: list[np.ndarray] = []
    loading_rows: list[np.ndarray] = []
    spectra: list[np.ndarray] = []
    previous: np.ndarray | None = None
    flips = 0
    maximum_survivors = 0

    for index in range(n_obs):
        if not defined[index]:
            continue
        correlation = correlations[index]
        if variant == "detoned":
            correlation = detone_correlation(correlation, components=scheme.detoned_components)
        observations = index + 1
        ratio = n_vars / observations
        eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
        fit = fit_noise_variance(eigenvalues, ratio=ratio, settings=scheme)
        result = denoise_correlation(correlation, fit=fit)
        oriented, flipped, alignment = orient_eigenvectors(
            result.eigenvectors, previous, survivors=result.survivors
        )
        previous = oriented
        flips += flipped
        maximum_survivors = max(maximum_survivors, result.survivors)

        weights = oriented[:, : result.survivors] / deviations[index][:, None]
        projections.append(values[index] @ weights)
        loading_rows.append(oriented[:, : result.survivors])
        spectra.append(result.eigenvalues)
        dates.append(pd.Timestamp(cast("date", frame.index[index])))
        rows.append(
            {
                "n_variables": float(n_vars),
                "observations": float(observations),
                "n_over_t": ratio,
                "noise_variance": fit.noise_variance,
                "lambda_plus": fit.lambda_plus,
                "lambda_minus": fit.lambda_minus,
                "band_half_width": fit.band_half_width,
                "survivors": float(result.survivors),
                "largest_eigenvalue": float(result.eigenvalues[0]),
                "diagonal_deviation": result.diagonal_deviation,
                "fit_sum_squared_error": fit.sum_squared_error,
                "fit_converged": float(fit.converged),
                "component_alignment": alignment,
                "survivors_after_rescale": float(result.survivors_after_rescale),
            }
        )

    index_dates = pd.DatetimeIndex(dates)
    columns = [f"pc{position + 1}" for position in range(maximum_survivors)]
    factors = pd.DataFrame(np.nan, index=index_dates, columns=columns)
    for position, projection in enumerate(projections):
        factors.iloc[position, : projection.size] = projection

    loadings: dict[str, pd.DataFrame] = {}
    for component in range(maximum_survivors):
        block = np.full((len(dates), n_vars), np.nan)
        for position, matrix in enumerate(loading_rows):
            if component < matrix.shape[1]:
                block[position] = matrix[:, component]
        loadings[columns[component]] = pd.DataFrame(
            block, index=index_dates, columns=list(frame.columns)
        )

    spectrum = pd.DataFrame(
        np.vstack(spectra),
        index=index_dates,
        columns=[f"lambda{position + 1}" for position in range(n_vars)],
    )
    diagnostics = pd.DataFrame(rows, index=index_dates)
    diagnostics["survivors"] = diagnostics["survivors"].astype(int)
    return StatisticalFactorPanel(
        variant=variant,
        factors=factors,
        diagnostics=diagnostics,
        loadings=loadings,
        spectrum=spectrum,
        assets=tuple(str(column) for column in frame.columns),
        sign_flips=flips,
    )


# ---------------------------------------------------------------------------
# The comparand metric -- SPEC.md 6.2's shape, without a strategy attached
# ---------------------------------------------------------------------------


def minimum_variance_realised_volatility(
    values: np.ndarray,
    estimates: Sequence[tuple[int, np.ndarray]],
    *,
    holding_days: int,
) -> float:
    """Annualised realised volatility of a minimum-variance portfolio.

    ``estimates`` is ``(position, covariance)`` pairs, oldest first; the weights
    implied by each are held for ``holding_days`` rows of ``values`` starting at
    ``position + 1``, so nothing is ever held on the date it was estimated from.

    **Weights are analytic** -- ``Sigma^-1 1 / 1' Sigma^-1 1`` -- with no
    optimizer, no expected return, no constraints and no costs. That is what
    makes this a risk-model diagnostic in the sense of SPEC.md 6.2 rather than a
    strategy: nothing here has a Sharpe, and the only quantity compared is a
    realised volatility against a forecast of it.
    """
    if holding_days < 1:
        raise StatisticalFactorError(
            f"minimum_variance_realised_volatility: holding_days must be positive, "
            f"got {holding_days}"
        )
    returns: list[float] = []
    for position, covariance in estimates:
        ones = np.ones(covariance.shape[0])
        try:
            solved = np.linalg.solve(covariance, ones)
        except np.linalg.LinAlgError as error:  # pragma: no cover - singular is refused
            raise StatisticalFactorError(
                "minimum_variance_realised_volatility: a singular covariance. The estimator "
                "produced a matrix with no minimum-variance solution and that is reported "
                "rather than pseudo-inverted."
            ) from error
        weights = solved / float(ones @ solved)
        window = values[position + 1 : position + 1 + holding_days]
        returns.extend((window @ weights).tolist())
    if not returns:
        raise StatisticalFactorError("minimum_variance_realised_volatility: no held periods")
    return float(np.std(returns, ddof=1) * np.sqrt(252.0))
