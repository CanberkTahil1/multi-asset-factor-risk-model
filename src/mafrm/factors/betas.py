"""Model A's asset exposures. SPEC.md 4.1 -- the rolling half of it.

W2-P1 built the six factor series. This builds what SPEC.md 4.1's equation is
actually about::

    r_{i,t} = alpha_i + sum_k beta_{i,k} f_{k,t} + u_{i,t}

estimated as an EWMA-weighted time-series regression over a trailing window --
252 days at a 63-day half-life, matching Barra's BETA descriptor -- one
regression per asset per date. The output is a ``(date, asset, factor)``
exposure panel plus the two things the regression produces alongside it and that
nothing here is allowed to throw away: the intercept and the R-squared.

Four things are load-bearing.

**Everything is regressed in basis points, and that is what makes an exposure
readable.** The two rate factors arrive in basis points of yield already; the
four return factors and every asset return are multiplied by 1e4 before the
regression sees them. The result is that every coefficient comes out in its
natural unit with no further conversion: the four return-factor betas are
dimensionless, because both sides are in the same units, and the two rate betas
are in **years**, reading directly as minus an effective duration. A ten-year
zero returns about -10 on the level factor. :func:`duration_recovery` is the
test of exactly that, against predictions registered in ``experiments.md``
before any of this existed.

**The intercept is a deliverable, not a nuisance.** SPEC.md 4.1 writes
``alpha_i`` into the model and the config fits it. A persistently non-zero alpha
means the six factors miss something systematic about that asset, which is
precisely what W2-P3's validation and SPEC.md 4.3's hybrid residual-PC model
exist to detect, so it is reported per asset, flagged when it is persistent, and
never discarded.

**The left edge is ragged and the ragged edge is correct.** A six-factor
regression needs all six factors, ``credit`` carries a 252-day burn-in against
equity and level, and after the W2-P2 window-start ruling ``rates_slope`` and
``commodity`` carry one too. Nothing is filled and nothing is truncated to the
latest start: :class:`ExposurePanel` reports the effective first date per factor
and per asset, and any table that spans the burn-in says it is partial rather
than like-for-like.

**Weighted, so the sample size is effective rather than nominal.** A 252-day
window at a 63-day half-life is not 252 observations. Kish's effective sample
size ``(sum w)^2 / sum w^2`` is about 160 here, and it is what the standard
errors and the R-squared degrees of freedom are computed against -- using 252
would overstate the precision of every alpha in the report by about a quarter.

This module reads the cache only. CLAUDE.md invariant 1: no network call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import pandas as pd

from mafrm.config import Config, MacroBetaConfig, load
from mafrm.data import gsw
from mafrm.factors import macro
from mafrm.numerics import ewma_weights, realised_effective_sample_size

__all__ = [
    "AssetDiagnostics",
    "BetaError",
    "DurationRecovery",
    "ExposurePanel",
    "FullSampleFit",
    "asset_excess_returns",
    "build_exposures",
    "diagnostics",
    "duration_recovery",
    "exposure_panel",
    "full_sample_fit",
    "regression_units",
    "rolling_exposures",
    "univariate_exposure",
]


class BetaError(ValueError):
    """An exposure cannot be estimated from what is cached."""


#: Universe ``construction`` values this module knows how to price, mapped to the
#: leg it reads. The synthetic entries go to the Fed's fitted curves; the ETF
#: entries go to raw closes plus the ex-date dividend table, never to an adjusted
#: close. The two spliced members (``ig_credit`` pre-2002, ``commodity``
#: pre-2006) are read on their ETF leg alone: the model window opens 2007-04, so
#: neither splice is reachable from here and neither is silently invoked.
_NOMINAL_ZERO = "synthetic_gsw_zero_curve"
_REAL_ZERO = "synthetic_tips_real_curve"
_ETF_CONSTRUCTIONS = frozenset(
    {
        "etf_total_return",
        "etf_total_return_with_moodys_extension",
        "etf_total_return_with_aqr_clr_splice",
    }
)

#: Universe members that are factor-construction inputs rather than holdings.
#: SPEC.md 4.1 and config/universe.yaml: the broad dollar index is a level, not
#: a tradable total return, and the universe excludes it from the investable set.
_NOT_INVESTABLE = frozenset({"fred_index_level"})


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def regression_units(frame: pd.DataFrame, units: Mapping[str, str]) -> pd.DataFrame:
    """Put every column in basis points, leaving the ones already there alone.

    The mixed-unit factor panel is what makes a level-factor coefficient a
    duration, but a regression cannot be run on mixed units and still hand back
    coefficients that mean anything. Scaling the decimal-return columns by 1e4
    and leaving the basis-point columns untouched puts both sides of every
    coefficient in the same unit: the return-factor betas come out dimensionless
    and the rate betas come out in years.
    """
    missing = [name for name in frame.columns if str(name) not in units]
    if missing:
        raise BetaError(f"regression_units: no unit recorded for {missing}")
    scaled = frame.copy()
    for name in frame.columns:
        if "basis points" not in units[str(name)]:
            scaled[name] = scaled[name] * macro._BPS_PER_UNIT
    return scaled


# ---------------------------------------------------------------------------
# What the assets are
# ---------------------------------------------------------------------------


def asset_excess_returns(
    *, config: Config | None = None, extra_tickers: Sequence[str] = ()
) -> dict[str, pd.Series]:
    """Daily excess return for every investable member of the frozen universe.

    Synthetic government and TIPS points come from the Fed's fitted curve with
    the roll-down (SPEC.md 3.2); ETF members come from raw unadjusted closes plus
    the ex-date dividend table, so appending tomorrow's dividend cannot change
    yesterday's number (CLAUDE.md failure mode 5).

    ``extra_tickers`` adds instruments that are NOT universe members -- TLT is
    the only caller, and it is there because W1-P3 measured its duration
    independently, which is what makes it a test rather than a restatement.
    """
    settings = config or load()
    rate = macro._daily_risk_free()
    risk_free_pct = gsw.load_risk_free()

    out: dict[str, pd.Series] = {}
    for asset in settings.universe.assets:
        if asset.construction in _NOT_INVESTABLE:
            continue
        if asset.construction in (_NOMINAL_ZERO, _REAL_ZERO):
            if asset.maturity_years is None:  # pragma: no cover - parser enforces it
                raise BetaError(f"{asset.id}: a synthetic curve point needs a maturity")
            kind: gsw.CurveKind = "nominal" if asset.construction == _NOMINAL_ZERO else "real"
            frame = gsw.constant_maturity_return(
                asset.maturity_years, kind=kind, risk_free_pct=risk_free_pct
            )
            series = frame["excess_return"].dropna()
        elif asset.construction in _ETF_CONSTRUCTIONS:
            if asset.ticker is None:  # pragma: no cover - parser enforces it
                raise BetaError(f"{asset.id}: an ETF member needs a ticker")
            series = macro.load_excess_return(asset.ticker, risk_free=rate)
        else:
            raise BetaError(
                f"{asset.id}: construction {asset.construction!r} is not priced by this module. "
                "Add it here deliberately rather than letting an asset drop out silently."
            )
        out[asset.id] = series.rename(asset.id)

    for ticker in extra_tickers:
        key = ticker.lower()
        out[key] = macro.load_excess_return(ticker, risk_free=rate).rename(key)
    return out


# ---------------------------------------------------------------------------
# The weighted regression
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FullSampleFit:
    """One asset's exposures estimated once over the whole window.

    The comparand for the rolling panel, and what :func:`duration_recovery`
    gates on. Full-sample OLS is what a normalization check wants -- one number
    per instrument at maximum precision -- while the rolling panel is what the
    covariance pipeline will consume.
    """

    id: str
    observations: int
    start: pd.Timestamp
    end: pd.Timestamp
    factors: tuple[str, ...]
    beta: pd.Series
    standard_error: pd.Series
    alpha: float
    alpha_standard_error: float
    alpha_newey_west_standard_error: float
    r_squared: float

    @property
    def alpha_t_statistic(self) -> float:
        """Newey-West t on the intercept. Daily returns are serially correlated."""
        if self.alpha_newey_west_standard_error <= 0.0:
            return float("nan")
        return self.alpha / self.alpha_newey_west_standard_error


def _newey_west_variance(residuals: np.ndarray, design: np.ndarray, lags: int) -> np.ndarray:
    """HAC sandwich covariance of an OLS coefficient vector. Newey-West (1987).

    ``S = Gamma_0 + sum_{l=1..L} (1 - l/(L+1)) (Gamma_l + Gamma_l')`` with the
    Bartlett kernel, sandwiched between ``(X'X)^-1``. Used only for the alpha
    t-statistic: the intercept of a daily return regression is a mean daily
    return, and a plain OLS standard error on one understates its uncertainty.
    """
    n_obs = residuals.size
    scores = design * residuals[:, None]
    bread = np.linalg.pinv(design.T @ design)
    meat = scores.T @ scores
    for lag in range(1, min(lags, n_obs - 1) + 1):
        block = scores[lag:].T @ scores[:-lag]
        meat += (1.0 - lag / (lags + 1.0)) * (block + block.T)
    return np.asarray(bread @ meat @ bread, dtype=float)


def full_sample_fit(
    returns: pd.Series,
    factors: pd.DataFrame,
    *,
    fit_intercept: bool,
    newey_west_lags: int,
    label: str | None = None,
) -> FullSampleFit:
    """OLS of one asset on the whole factor set, over their common dates.

    Both sides arrive already in basis points, so the rate-factor coefficients
    are in years and read as minus an effective duration.
    """
    name = label if label is not None else str(returns.name)
    # `join`, not `concat`: an inner join on two DatetimeIndexes is what
    # `dropna(how="any")` would produce anyway, and it does not go through the
    # deprecated default-sort path that concat takes on unequal calendars.
    frame = factors.join(returns.rename("asset"), how="inner").dropna(how="any")
    columns = tuple(str(item) for item in factors.columns)
    if len(frame) <= len(columns) + 1:
        raise BetaError(
            f"{name}: {len(frame)} observation(s) shared with the factor set, which cannot "
            f"identify {len(columns)} coefficient(s)"
        )

    target = frame["asset"].to_numpy(dtype=float)
    regressors = frame[list(columns)].to_numpy(dtype=float)
    design = (
        np.column_stack([np.ones(target.size), regressors]) if fit_intercept else regressors.copy()
    )
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ beta
    dof = target.size - design.shape[1]
    variance = float(residual @ residual) / dof
    covariance = variance * np.linalg.pinv(design.T @ design)
    errors = np.sqrt(np.diag(covariance))
    hac = _newey_west_variance(residual, design, newey_west_lags)

    centred = target - target.mean() if fit_intercept else target
    r_squared = 1.0 - float(residual @ residual) / float(centred @ centred)
    offset = 1 if fit_intercept else 0
    return FullSampleFit(
        id=name,
        observations=len(frame),
        start=pd.Timestamp(frame.index[0]),
        end=pd.Timestamp(frame.index[-1]),
        factors=columns,
        beta=pd.Series(beta[offset:], index=list(columns), name=name),
        standard_error=pd.Series(errors[offset:], index=list(columns), name=name),
        alpha=float(beta[0]) if fit_intercept else 0.0,
        alpha_standard_error=float(errors[0]) if fit_intercept else float("nan"),
        alpha_newey_west_standard_error=(
            float(np.sqrt(hac[0, 0])) if fit_intercept else float("nan")
        ),
        r_squared=r_squared,
    )


def rolling_exposures(
    returns: pd.Series,
    factors: pd.DataFrame,
    *,
    window: int,
    halflife: int,
    fit_intercept: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """EWMA-weighted rolling regression of one asset on the factor set.

    SPEC.md 4.1: "EWMA-weighted, 252-day window, half-life 63 days, matching
    Barra's BETA descriptor." At each date the trailing ``window`` observations
    are weighted so that a point ``halflife`` days older counts half as much, and
    a weighted least-squares regression is solved on them; nothing after the date
    enters, so the exposures are usable as forecasts rather than as description.

    Returns ``(exposures, alpha, alpha_standard_error, r_squared)``, all indexed
    by the dates on which the asset and all six factors exist -- ``NaN`` before
    the window is full. Nothing is filled.

    The R-squared is the **weighted** one, computed against a weighted mean, so
    it answers "how much of the variation this regression was actually fitted to
    does the factor set explain" rather than mixing a weighted fit with an
    unweighted benchmark.
    """
    frame = factors.join(returns.rename("asset"), how="inner").dropna(how="any")
    columns = [str(item) for item in factors.columns]
    n_terms = len(columns) + (1 if fit_intercept else 0)
    if window <= n_terms:
        raise BetaError(
            f"{returns.name}: a {window}-day window cannot identify {n_terms} coefficient(s)"
        )

    index = frame.index
    n_obs = len(frame)
    exposures = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    alpha = pd.Series(np.nan, index=index, dtype=float)
    alpha_error = pd.Series(np.nan, index=index, dtype=float)
    r_squared = pd.Series(np.nan, index=index, dtype=float)
    if n_obs < window:
        return exposures, alpha, alpha_error, r_squared

    target = frame["asset"].to_numpy(dtype=float)
    regressors = frame[columns].to_numpy(dtype=float)
    design = np.column_stack([np.ones(n_obs), regressors]) if fit_intercept else regressors.copy()

    weights = ewma_weights(window, float(halflife))
    effective = realised_effective_sample_size(weights)
    dof = effective - n_terms
    if dof <= 0.0:
        raise BetaError(
            f"{returns.name}: a {window}-day window at a {halflife}-day half-life carries "
            f"{effective:.1f} effective observations, which cannot support {n_terms} terms"
        )
    root = np.sqrt(weights)
    offset = 1 if fit_intercept else 0

    beta_out = np.full((n_obs, len(columns)), np.nan)
    alpha_out = np.full(n_obs, np.nan)
    alpha_error_out = np.full(n_obs, np.nan)
    r2_out = np.full(n_obs, np.nan)

    for position in range(window - 1, n_obs):
        low = position - window + 1
        block = design[low : position + 1] * root[:, None]
        response = target[low : position + 1] * root
        gram = block.T @ block
        moment = block.T @ response
        try:
            coefficients = np.linalg.solve(gram, moment)
        except np.linalg.LinAlgError:  # pragma: no cover - a singular 252x7 design
            coefficients = np.linalg.pinv(gram) @ moment
        residual = response - block @ coefficients
        weighted_rss = float(residual @ residual)

        beta_out[position] = coefficients[offset:]
        if fit_intercept:
            # Sum(w)=1, so the weighted cross-product is an average rather than a
            # total: the sample size has to be put back in explicitly, and the
            # honest one is Kish's effective count, not the window length.
            scale = weighted_rss / dof
            inverse = np.linalg.pinv(gram)
            alpha_out[position] = coefficients[0]
            alpha_error_out[position] = float(np.sqrt(max(scale * inverse[0, 0], 0.0)))

        block_target = target[low : position + 1]
        mean = float(weights @ block_target) if fit_intercept else 0.0
        deviation = block_target - mean
        total = float(weights @ (deviation * deviation))
        r2_out[position] = 1.0 - weighted_rss / total if total > 0.0 else np.nan

    exposures = pd.DataFrame(beta_out, index=index, columns=columns)
    alpha = pd.Series(alpha_out, index=index, dtype=float)
    alpha_error = pd.Series(alpha_error_out, index=index, dtype=float)
    r_squared = pd.Series(r2_out, index=index, dtype=float)
    return exposures, alpha, alpha_error, r_squared


def univariate_exposure(
    returns: pd.Series,
    factors: pd.DataFrame,
    factor: str,
    *,
    units: Mapping[str, str],
    settings: MacroBetaConfig,
) -> pd.Series:
    """One asset's rolling exposure to ONE factor, with the others left out.

    Not a competitor to :class:`ExposurePanel` -- a different question. The
    panel's exposures are **partial** coefficients: they answer "what is left of
    this asset's sensitivity to that factor once the other five are held
    fixed". A univariate beta answers "how does this asset move with that factor",
    full stop.

    The distinction is not pedantic here and SPEC.md 4.1's headline chart is why.
    ``us_large_equity`` is SPY and the equity factor is Ken French ``Mkt-RF``;
    they are all but the same series, so the equity factor absorbs SPY's variance
    and SPY's *partial* rates exposure is ~0 on every date of the sample. The
    stock-bond beta that flips sign around 2021 is the **univariate** one, and it
    is only visible with the equity factor out of the regression.
    """
    if factor not in factors.columns:
        raise BetaError(f"{factor!r} is not a factor; got {list(factors.columns)}")
    scaled = regression_units(factors.dropna(how="any"), units)[[factor]]
    exposures, _, _, _ = rolling_exposures(
        (returns.dropna() * macro._BPS_PER_UNIT).rename(str(returns.name)),
        scaled,
        window=settings.window,
        halflife=settings.halflife,
        fit_intercept=settings.fit_intercept,
    )
    return exposures[factor].dropna().rename(f"{returns.name}/{factor} (univariate)")


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExposurePanel:
    """Model A's ``(date, asset, factor)`` exposures, with alpha and R-squared.

    ``exposures`` is indexed by a ``(date, asset)`` MultiIndex with one column
    per factor; ``alpha``, ``alpha_standard_error`` and ``r_squared`` are
    ``date x asset`` frames. All four share the same ragged left edge, and it is
    reported rather than filled: :attr:`asset_first_valid` and
    :attr:`factor_first_valid` say where each series actually begins.

    Units, in one place: rate exposures are **years** (minus an effective
    duration), return-factor exposures are dimensionless, alpha is **basis
    points per day**.
    """

    exposures: pd.DataFrame
    alpha: pd.DataFrame
    alpha_standard_error: pd.DataFrame
    r_squared: pd.DataFrame
    factors: tuple[str, ...]
    assets: tuple[str, ...]
    factor_first_valid: Mapping[str, pd.Timestamp]
    window: int
    halflife: int
    fit_intercept: bool
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def asset_first_valid(self) -> dict[str, pd.Timestamp]:
        """First date each asset's exposures are defined. The burn-in, per asset."""
        out: dict[str, pd.Timestamp] = {}
        for asset in self.assets:
            first = self.r_squared[asset].first_valid_index()
            if first is not None:
                out[asset] = pd.Timestamp(cast("date", first))
        return out

    @property
    def effective_sample_size(self) -> float:
        """Kish's effective observation count behind every single regression.

        A formula, not a parameter. A 252-day window at a 63-day half-life
        carries about 160 effective observations, not 252, and using the nominal
        count would overstate the precision of every coefficient by roughly a
        quarter. Shared with the covariance pipeline through
        :mod:`mafrm.numerics` rather than reimplemented here.
        """
        weights = ewma_weights(self.window, float(self.halflife))
        return realised_effective_sample_size(weights)

    def exposure(self, asset: str, factor: str) -> pd.Series:
        """One asset's exposure to one factor, through time, NaNs dropped."""
        if asset not in self.assets:
            raise BetaError(f"{asset!r} is not in the exposure panel; got {list(self.assets)}")
        if factor not in self.factors:
            raise BetaError(f"{factor!r} is not a factor; got {list(self.factors)}")
        series = self.exposures.xs(asset, level="asset")[factor].dropna()
        return series.rename(f"{asset}/{factor}")


def build_exposures(
    returns: Mapping[str, pd.Series],
    factors: pd.DataFrame,
    *,
    units: Mapping[str, str],
    settings: MacroBetaConfig,
) -> ExposurePanel:
    """Roll every asset through :func:`rolling_exposures` and stack the result.

    ``factors`` arrives in the panel's mixed units and is converted here, once,
    so that no caller can regress on a half-converted design.
    """
    if not returns:
        raise BetaError("build_exposures: no asset returns given")
    scaled_factors = regression_units(factors.dropna(how="any"), units)
    if scaled_factors.empty:
        raise BetaError("build_exposures: no date has every factor defined")
    columns = [str(item) for item in scaled_factors.columns]

    frames: dict[str, pd.DataFrame] = {}
    alphas: dict[str, pd.Series] = {}
    errors: dict[str, pd.Series] = {}
    fits: dict[str, pd.Series] = {}
    for asset in sorted(returns):
        series = returns[asset].dropna()
        scaled = series * macro._BPS_PER_UNIT
        exposures, alpha, error, r_squared = rolling_exposures(
            scaled.rename(asset),
            scaled_factors,
            window=settings.window,
            halflife=settings.halflife,
            fit_intercept=settings.fit_intercept,
        )
        frames[asset] = exposures
        alphas[asset] = alpha
        errors[asset] = error
        fits[asset] = r_squared

    stacked = pd.concat(frames, names=["asset", "date"]).reorder_levels(["date", "asset"])
    stacked = stacked.sort_index()
    alpha_frame = pd.DataFrame(alphas).sort_index()
    error_frame = pd.DataFrame(errors).sort_index()
    r_squared_frame = pd.DataFrame(fits).sort_index()
    return ExposurePanel(
        exposures=stacked,
        alpha=alpha_frame,
        alpha_standard_error=error_frame,
        r_squared=r_squared_frame,
        factors=tuple(columns),
        assets=tuple(sorted(returns)),
        factor_first_valid={
            str(name): pd.Timestamp(cast("date", factors[name].first_valid_index()))
            for name in factors.columns
            if factors[name].first_valid_index() is not None
        },
        window=settings.window,
        halflife=settings.halflife,
        fit_intercept=settings.fit_intercept,
        start=pd.Timestamp(r_squared_frame.index[0]),
        end=pd.Timestamp(r_squared_frame.index[-1]),
    )


def exposure_panel(
    panel: macro.FactorPanel | None = None, *, config: Config | None = None
) -> ExposurePanel:
    """Build Model A's exposure panel over the frozen universe. SPEC.md 4.1."""
    settings = config or load()
    factors = panel if panel is not None else macro.macro_factor_panel(config=settings)
    returns = asset_excess_returns(config=settings)
    boundary = settings.require_holdout_start()
    trimmed = {
        name: series.loc[(series.index >= factors.start) & (series.index < pd.Timestamp(boundary))]
        for name, series in returns.items()
    }
    return build_exposures(
        trimmed,
        factors.orthogonal,
        units=factors.units,
        settings=settings.model.factors.macro.beta,
    )


# ---------------------------------------------------------------------------
# Flags: what SPEC.md 4.1 asks the report to say out loud
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetDiagnostics:
    """One asset's R-squared and alpha record, and whether either is flagged.

    Both flags are REPORTING flags. SPEC.md 4.1 asks for "a candidate for
    removal from the sleeve or for an additional factor", and a candidate list is
    what these produce; nothing here removes an asset from anything.
    """

    id: str
    #: True when this asset is the instrument a factor is BUILT FROM -- `credit`
    #: is HYG over cash and `commodity` is DBC over cash, so `hy_credit` and
    #: `commodity` are exact linear combinations of the factor set by
    #: construction. Their R-squared of 1.000 is an identity, not a measurement,
    #: and the report has to say so rather than presenting it as a fit.
    spans_a_factor: bool
    observations: int
    start: pd.Timestamp
    end: pd.Timestamp
    r_squared_median: float
    r_squared_min: float
    r_squared_max: float
    r_squared_below_fraction: float
    r_squared_limit: float
    alpha_mean_bps_per_day: float
    alpha_annualized_pct: float
    alpha_significant_fraction: float
    alpha_t_limit: float
    full_sample_alpha_annualized_pct: float
    full_sample_alpha_t: float
    full_sample_r_squared: float
    persistence_fraction: float

    @property
    def r_squared_flagged(self) -> bool:
        """Persistently below SPEC.md 4.1's 0.5 line."""
        return self.r_squared_below_fraction >= self.persistence_fraction

    @property
    def alpha_flagged(self) -> bool:
        """Persistently distinguishable from zero inside its own window."""
        return self.alpha_significant_fraction >= self.persistence_fraction


def diagnostics(
    panel: ExposurePanel,
    returns: Mapping[str, pd.Series],
    factors: pd.DataFrame,
    *,
    units: Mapping[str, str],
    config: Config | None = None,
) -> tuple[AssetDiagnostics, ...]:
    """Per-asset R-squared and alpha records, with both SPEC.md 4.1 flags.

    The rolling statistics are what "persistently" is judged on; the full-sample
    alpha is reported beside them because a within-window t-statistic on 160
    effective observations has little power against a small persistent drift,
    and saying so is cheaper than letting a reader assume the rolling test is the
    stronger one.
    """
    settings = config or load()
    macro_config = settings.model.factors.macro
    lags = settings.model.covariance.volatility_newey_west_lags.short
    share = macro_config.persistence_fraction
    limit = macro_config.min_r_squared_flag
    t_limit = macro_config.alpha_flag.abs_t_statistic
    sources = {macro_config.credit.asset_id, macro_config.commodity.asset_id}
    scaled_factors = regression_units(factors.dropna(how="any"), units)
    per_year = settings.model.data.trading_days_per_year

    out: list[AssetDiagnostics] = []
    for asset in panel.assets:
        r_squared = panel.r_squared[asset].dropna()
        alpha = panel.alpha[asset].dropna()
        error = panel.alpha_standard_error[asset].reindex(alpha.index)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_statistic = (alpha / error).replace([np.inf, -np.inf], np.nan).dropna()
        fit = full_sample_fit(
            (returns[asset].dropna() * macro._BPS_PER_UNIT).rename(asset),
            scaled_factors,
            fit_intercept=macro_config.beta.fit_intercept,
            newey_west_lags=lags,
            label=asset,
        )
        # bps/day -> %/yr: /100 to reach percent, x trading days to annualize.
        annualize = per_year / 100.0
        out.append(
            AssetDiagnostics(
                id=asset,
                spans_a_factor=asset in sources,
                observations=len(r_squared),
                start=pd.Timestamp(r_squared.index[0]),
                end=pd.Timestamp(r_squared.index[-1]),
                r_squared_median=float(r_squared.median()),
                r_squared_min=float(r_squared.min()),
                r_squared_max=float(r_squared.max()),
                r_squared_below_fraction=float((r_squared < limit).mean()),
                r_squared_limit=limit,
                alpha_mean_bps_per_day=float(alpha.mean()),
                alpha_annualized_pct=float(alpha.mean()) * annualize,
                alpha_significant_fraction=(
                    float((t_statistic.abs() > t_limit).mean())
                    if len(t_statistic)
                    else float("nan")
                ),
                alpha_t_limit=t_limit,
                full_sample_alpha_annualized_pct=fit.alpha * annualize,
                full_sample_alpha_t=fit.alpha_t_statistic,
                full_sample_r_squared=fit.r_squared,
                persistence_fraction=share,
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Duration recovery: the session's primary validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DurationRecovery:
    """One instrument's level exposure against the duration it should reproduce.

    ``predicted`` is negative -- it is a coefficient on the normalized level
    factor, which reads as minus the effective duration in years.

    ``basis`` says where the prediction comes from and the distinction matters:
    ``registered`` means ``experiments.md`` rows 64-67, pre-registered on
    2026-08-30 before the factor set existed; ``identity`` means the maturity of
    a zero-coupon bond, whose duration equals its maturity by definition and is
    therefore not a prediction anyone had to make. Neither number was chosen
    after seeing a measurement.
    """

    id: str
    basis: str
    predicted: float
    band: tuple[float, float]
    #: W2-P1's own number, from :func:`mafrm.factors.macro.duration_check`: level
    #: only, on the dates the instrument shares with the RAW level factor, which
    #: is a longer sample than the panel's complete cases. ``None`` for the two
    #: instruments W2-P1 never measured.
    w2p1_univariate: float | None
    #: Level only, but on the panel's own complete-case dates. Comparable with
    #: `multivariate` because it is the same sample; NOT the same number as
    #: `w2p1_univariate`, and conflating the two would attribute a change of
    #: sample to a change of specification.
    univariate: float
    multivariate: float
    multivariate_standard_error: float
    rolling_mean: float
    rolling_min: float
    rolling_max: float
    observations: int
    start: pd.Timestamp
    end: pd.Timestamp
    r_squared: float

    @property
    def passed(self) -> bool:
        """Judged on the multivariate exposure -- the panel's own specification."""
        low, high = self.band
        return low <= self.multivariate <= high

    @property
    def difference(self) -> float:
        return self.multivariate - self.predicted

    @property
    def implied_duration(self) -> float:
        return -self.multivariate

    def render(self) -> str:
        low, high = self.band
        spot = "--" if self.w2p1_univariate is None else f"{self.w2p1_univariate:+.2f}"
        return (
            f"{self.id}: level exposure {self.multivariate:+.2f} "
            f"(predicted {self.predicted:+.2f} [{self.basis}], band [{low:+.2f}, {high:+.2f}], "
            f"difference {self.difference:+.2f}) -- {'pass' if self.passed else 'FAIL'}; "
            f"W2-P1 spot check {spot}, univariate here {self.univariate:+.2f}, "
            f"rolling mean {self.rolling_mean:+.2f}, n={self.observations:,}"
        )


def duration_recovery(
    factors: macro.FactorPanel,
    panel: ExposurePanel,
    *,
    config: Config | None = None,
) -> tuple[DurationRecovery, ...]:
    """Does the exposure panel reproduce durations derived independently?

    The session's primary validation. Every instrument here has a duration this
    project already knows from somewhere the factor set cannot see: the four
    synthetic zeros from their own maturity (a zero-coupon bond's duration is its
    maturity, by definition), TLT from W1-P3's regression against the GSW
    par-coupon ladder, HYG from its registered analytical range.

    Three measurements are reported per instrument because they answer different
    questions. ``univariate`` is W2-P1's spot check, reproduced so that the panel
    can be checked against it rather than merely replacing it. ``multivariate``
    is the panel's own specification -- all six factors, full sample -- and is
    what the band is judged on: it is the better estimator here, because a
    univariate regression on level alone omits the slope factor and a long bond's
    return loads on both. ``rolling_mean`` is the mean of the panel's own EWMA
    exposures, which is the number the covariance pipeline will actually consume.
    """
    settings = config or load()
    check = settings.model.factors.macro.duration_check
    macro_config = settings.model.factors.macro
    lags = settings.model.covariance.volatility_newey_west_lags.short
    tolerance = check.tolerance_fraction
    registered = {item.id: item for item in check.instruments}

    scaled = regression_units(factors.orthogonal.dropna(how="any"), factors.units)
    level_only = scaled[[macro.LEVEL]]
    boundary = pd.Timestamp(settings.require_holdout_start())

    # The four synthetic zeros carry an IDENTITY rather than a registered
    # prediction: a zero-coupon bond's duration is its maturity, by definition.
    # Read off the frozen universe, so the number is not written down twice and
    # cannot be quietly adjusted to fit a measurement.
    # Ordered by MATURITY, not by name: a string sort puts govt_10y before
    # govt_2y and every table and sentence downstream then reads out of order.
    identities = {
        asset.id: -float(asset.maturity_years)
        for asset in sorted(
            settings.universe.assets,
            key=lambda item: item.maturity_years if item.maturity_years is not None else 0.0,
        )
        if asset.construction == _NOMINAL_ZERO and asset.maturity_years is not None
    }
    # W2-P1's own measurements, reproduced rather than restated. They are on a
    # LONGER sample -- level only, every date the instrument shares with the raw
    # level factor -- so they are carried in their own column and never mixed
    # with the panel's complete-case figures.
    spot_checks = {item.id: item.slope for item in macro.duration_check(factors, config=settings)}
    universe_ids = {asset.id for asset in settings.universe.assets}
    outside = [
        item.ticker
        for item in check.instruments
        if item.ticker is not None and item.id not in universe_ids
    ]
    returns = asset_excess_returns(config=settings, extra_tickers=outside)

    order = [
        *identities,
        *[item.id for item in check.instruments if item.id not in identities],
    ]

    out: list[DurationRecovery] = []
    for name in order:
        instrument = registered.get(name)
        # Prefer the universe id: `hy_credit` and the registered `HYG` are the
        # same series, and reading it under one name keeps the duration table and
        # the exposure panel demonstrably on the same data.
        key = name
        if key not in returns and instrument is not None and instrument.ticker is not None:
            key = instrument.ticker.lower()
        if key not in returns:
            raise BetaError(f"duration_recovery: no excess-return series for {name!r}")
        series = returns[key].dropna()
        series = series.loc[(series.index >= factors.start) & (series.index < boundary)]
        target = (series * macro._BPS_PER_UNIT).rename(name)

        if instrument is not None:
            predicted = instrument.predicted_duration
            band = instrument.band(tolerance)
            basis = "registered"
        else:
            predicted = identities[name]
            half = abs(predicted) * tolerance
            band = (predicted - half, predicted + half)
            basis = "identity"

        univariate = full_sample_fit(
            target, level_only, fit_intercept=True, newey_west_lags=lags, label=name
        )
        multivariate = full_sample_fit(
            target, scaled, fit_intercept=True, newey_west_lags=lags, label=name
        )
        if name in panel.assets:
            rolling = panel.exposure(name, macro.LEVEL)
        else:
            exposures, _, _, _ = rolling_exposures(
                target,
                scaled,
                window=macro_config.beta.window,
                halflife=macro_config.beta.halflife,
                fit_intercept=macro_config.beta.fit_intercept,
            )
            rolling = exposures[macro.LEVEL].dropna()

        out.append(
            DurationRecovery(
                id=name,
                basis=basis,
                predicted=predicted,
                band=band,
                w2p1_univariate=spot_checks.get(name),
                univariate=float(univariate.beta[macro.LEVEL]),
                multivariate=float(multivariate.beta[macro.LEVEL]),
                multivariate_standard_error=float(multivariate.standard_error[macro.LEVEL]),
                rolling_mean=float(rolling.mean()),
                rolling_min=float(rolling.min()),
                rolling_max=float(rolling.max()),
                observations=multivariate.observations,
                start=multivariate.start,
                end=multivariate.end,
                r_squared=multivariate.r_squared,
            )
        )
    return tuple(out)


def _main() -> int:
    """``make model``: build the exposure panel and report what it did."""
    settings = load()
    factors = macro.macro_factor_panel(config=settings)
    panel = exposure_panel(factors, config=settings)
    returns = asset_excess_returns(config=settings)

    print(
        f"exposure panel {panel.start.date()}..{panel.end.date()}, "
        f"{len(panel.assets)} assets x {len(panel.factors)} factors, "
        f"{panel.window}d window at a {panel.halflife}d half-life "
        f"({panel.effective_sample_size:.0f} effective observations)"
    )
    print("  factor burn-in:")
    for name, first in panel.factor_first_valid.items():
        print(f"    {name}: first defined {first.date()}")
    print("  asset burn-in:")
    for name, first in sorted(panel.asset_first_valid.items()):
        print(f"    {name}: first exposure {first.date()}")

    print("  duration recovery (SPEC.md 4.1, experiments.md rows 64-67 and 73-78):")
    for estimate in duration_recovery(factors, panel, config=settings):
        print(f"    {estimate.render()}")

    print("  per-asset R^2 and alpha:")
    for record in diagnostics(
        panel, returns, factors.orthogonal, units=factors.units, config=settings
    ):
        flags = [
            label
            for label, fired in (("R2", record.r_squared_flagged), ("alpha", record.alpha_flagged))
            if fired
        ]
        print(
            f"    {record.id}: R^2 median {record.r_squared_median:.3f} "
            f"[{record.r_squared_min:.3f}, {record.r_squared_max:.3f}], "
            f"below {record.r_squared_limit} on {100 * record.r_squared_below_fraction:.0f}% "
            f"of dates; alpha {record.alpha_annualized_pct:+.2f}%/yr rolling "
            f"(|t| > {record.alpha_t_limit:g} on "
            f"{100 * record.alpha_significant_fraction:.0f}% of dates), "
            f"{record.full_sample_alpha_annualized_pct:+.2f}%/yr full-sample "
            f"(NW t {record.full_sample_alpha_t:+.2f})"
            + ("  [IDENTITY: this asset builds a factor]" if record.spans_a_factor else "")
            + (f" -- FLAGGED: {', '.join(flags)}" if flags else "")
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(_main())
