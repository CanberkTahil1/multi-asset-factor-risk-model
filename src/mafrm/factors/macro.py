"""Model A -- the six-factor macro set. SPEC.md 4.1, as amended by SPEC.md 4.1.1.

Six observed factor return series over the multi-asset sleeve: equity, rates
level, rates slope, credit, commodity, dollar. Model A is a **time-series**
model -- with 15 assets and 6 factors you regress each asset's excess return on
observed factor returns, rather than fitting exposures cross-sectionally -- so
the factors have to be real, observable series before any beta exists. This
module builds them and nothing else.

Three things here are not obvious and each is load-bearing.

**Units are mixed, deliberately.** The four return factors are decimal daily
returns; the two rate factors are **basis points of yield**. Correlation and
Gram-Schmidt are both scale-covariant, so the mixture costs nothing where the
factors are used together, and it buys the property the normalization exists
for: a regression of an asset's excess return **in basis points** on the level
factor returns that asset's **negative effective duration**, directly. A ten-year
zero returns about -10. :class:`FactorPanel` carries the unit of every column
alongside the column so that a caller cannot forget which is which, and
:func:`to_basis_points` is the only sanctioned conversion.

**The PC normalization is also the sign fix.** An eigenvector's scale and sign
are both arbitrary, and LAPACK's orientation is not stable across slightly
different matrices. SPEC.md 4.1 asks for "sign-fixed so positive = yields up"
and SPEC.md 4.1.1 adds the scale: the level PC is rescaled so its **mean loading
across the tenors is one basis point**, the slope PC so its **long-minus-short
loading difference is one basis point**. Both rescalings are quadratic in the
eigenvector's sign -- flip the eigenvector and both the loading vector and the
factor score flip, and their product does not -- so the normalization fixes the
sign as a by-product rather than needing a separate convention that could
disagree with it. :class:`SignFixDiagnostics` still counts how often the raw
eigenvector came back pointing the wrong way, because a high count is a finding
about the PC's stability and not a nuisance to suppress.

**Everything expanding-window is expanding-window for one reason.** A full-sample
eigendecomposition uses the whole history's covariance to decide what the factor
was on its first day; a full-sample orthogonalization uses the whole history's
covariance to decide what the hedge ratio was on its first day. Both are
look-ahead, both would look completely normal in a plot, and both would flatter
every downstream statistic. The PCs and the Gram-Schmidt are therefore estimated
on data through date ``t`` only, with a minimum window before which the factor is
``NaN``, and ``tests/test_macro_factors.py`` asserts the expanding and
full-sample constructions actually differ rather than trusting this paragraph.

The credit factor is HYG excess over **cash**, not over a duration-matched
Treasury. SPEC.md 4.1 specified the hedge twice -- once as a matched Treasury leg
and again as the orthogonalization against the rates level factor -- and running
both hedges the rates exposure twice. SPEC.md 4.1.1 records the amendment; the
practical consequence is that this module invents no duration constant for HYG,
and the hedge ratio is estimated and time-varying instead of assumed and fixed.

This module reads the cache only. CLAUDE.md invariant 1: no network call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import pandas as pd

from mafrm.config import (
    Config,
    DurationCheckInstrument,
    OrthogonalizationConfig,
    RatePcaConfig,
    load,
)
from mafrm.data import cache, calendar, french, gsw, prices
from mafrm.risk.covariance import correlation_from_covariance, ewma_second_moment

__all__ = [
    "Conditioning",
    "DurationEstimate",
    "FactorPanel",
    "GateResult",
    "MacroFactorError",
    "RatePcaResult",
    "SignFixDiagnostics",
    "condition_numbers",
    "conditioning",
    "correlation_matrices",
    "curve_yield_changes_bps",
    "duration_check",
    "load_excess_return",
    "macro_factor_panel",
    "orthogonalize",
    "rate_factors",
    "raw_macro_factors",
    "to_basis_points",
]

#: Basis points in one unit of decimal return or yield. A unit conversion, not a
#: tunable -- the same status as ``_BPS`` in :mod:`mafrm.costs.spread`.
_BPS_PER_UNIT = 1e4

#: Percent-per-annum to basis-points-per-annum. The GSW loader returns fitted
#: yields in percent, which is the file's own unit.
_BPS_PER_PERCENT = 100.0

#: Factor names. These are the keys used in ``model.factors.macro.orthogonalization``
#: and the column names of every frame this module returns.
LEVEL = "rates_level"
SLOPE = "rates_slope"
EQUITY = "equity"
CREDIT = "credit"
COMMODITY = "commodity"
DOLLAR = "dollar"

#: Unit of each factor column. Carried with the panel rather than documented,
#: because the duration reading is only correct if the caller knows which is which.
_UNITS: Mapping[str, str] = {
    EQUITY: "decimal daily excess return",
    LEVEL: "basis points of yield",
    SLOPE: "basis points of yield",
    CREDIT: "decimal daily excess return",
    COMMODITY: "decimal daily excess return",
    DOLLAR: "decimal daily index return",
}


class MacroFactorError(ValueError):
    """A macro factor cannot be built from what is cached."""


def to_basis_points(values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Decimal returns to basis points. ``0.001 -> 10.0``.

    The conversion that makes a level-factor regression coefficient read as a
    duration: with both sides in basis points, the slope is dimensionless and
    equals minus the effective duration in years.
    """
    return values * _BPS_PER_UNIT


# ---------------------------------------------------------------------------
# The curve: yield changes at the four key rates
# ---------------------------------------------------------------------------


def curve_yield_changes_bps(
    *,
    tenors_years: Sequence[float],
    params: pd.DataFrame | None = None,
    end: date | None = None,
    config: Config | None = None,
    kind: gsw.CurveKind = "nominal",
) -> pd.DataFrame:
    """Daily changes in fitted zero yields at each tenor, in basis points.

    Each tenor is masked to the dates its published column exists
    (:func:`mafrm.data.gsw.published_mask`), because beyond the longest bond
    outstanding the Svensson form is extrapolation rather than observation. Rows
    where any tenor is unpublished are dropped **before** differencing, so a
    change always runs between two dates on which every tenor was genuinely
    published -- including across a market holiday, which is a real multi-day
    change and is left as one observation rather than being scaled to a day.

    ``end`` is the holdout boundary and is half-open (CLAUDE.md invariant 5).

    ``kind`` selects the Fed curve. The default is the nominal one, which is
    what SPEC.md 4.1's two rate factors are built from. W3-P6 added the
    ``"real"`` leg for row 83's placebo, which needs the 10-year REAL yield
    change and the 10-year NOMINAL yield change on the same footing --
    identically masked, identically differenced -- so that the comparison is
    between two curves and not between two constructions.
    """
    settings = config or load()
    curve = (
        settings.model.data.nominal_curve if kind == "nominal" else settings.model.data.real_curve
    )
    if params is None:
        params = gsw.load_curve(kind)

    levels = pd.DataFrame(
        {
            f"y{tenor:g}": gsw.curve_yields(params, tenor).where(
                gsw.published_mask(params, tenor, yield_prefix=curve.yield_prefix)
            )
            for tenor in tenors_years
        }
    )
    published = levels.dropna(how="any")
    if end is not None:
        published = published[published.index < pd.Timestamp(end)]
    if len(published) < 2:
        raise MacroFactorError(
            f"only {len(published)} date(s) have all of {list(tenors_years)} published; the "
            "curve cannot supply a yield change. Run `make data`."
        )
    changes = published.diff().iloc[1:] * _BPS_PER_PERCENT
    changes.index.name = "date"
    return changes


# ---------------------------------------------------------------------------
# Expanding-window PCA with the 1bp normalization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignFixDiagnostics:
    """How the arbitrary parts of an eigendecomposition actually behaved.

    ``sign_fix_fired`` counts the dates on which the raw eigenvector came back
    pointing the wrong way and had to be flipped. A *high* count is not itself a
    problem -- LAPACK's orientation is arbitrary and this is exactly the
    convention the fix exists to impose -- but a count that is neither near zero
    nor near half is worth looking at, because it means the orientation is
    tracking something.

    ``orientation_reversals`` is the diagnostic that matters. It counts dates on
    which the **already-fixed** loading vector points away from the previous
    date's -- a negative dot product after the convention has been applied. That
    cannot happen to a stable PC, because one day of data cannot rotate a
    covariance eigenvector through 90 degrees. A nonzero count means the two PCs
    are close to degenerate and are swapping places, which is a finding about the
    curve and not a nuisance to suppress.

    ``shape_violations`` counts dates where the PC does not have the shape its
    name claims: for the level PC, loadings that are not all the same sign; for
    the slope PC, ends that do not have opposite signs.
    """

    factor: str
    observations: int
    sign_fix_fired: int
    orientation_reversals: int
    shape_violations: int

    @property
    def sign_fix_fraction(self) -> float:
        return self.sign_fix_fired / self.observations if self.observations else 0.0

    @property
    def reversal_fraction(self) -> float:
        return self.orientation_reversals / self.observations if self.observations else 0.0

    def render(self) -> str:
        return (
            f"{self.factor}: {self.observations:,} estimation dates; sign fix fired "
            f"{self.sign_fix_fired:,} ({100 * self.sign_fix_fraction:.1f}%); "
            f"orientation reversals {self.orientation_reversals:,} "
            f"({100 * self.reversal_fraction:.2f}%); shape violations "
            f"{self.shape_violations:,}"
        )


@dataclass(frozen=True)
class RatePcaResult:
    """The two rate factors, their loadings, and what the eigendecomposition did.

    ``factors`` are in basis points of yield. ``loadings`` holds, per date, the
    basis points each tenor moves when the factor moves by one unit -- normalized
    so that the level PC's mean loading is 1bp and the slope PC's long-minus-short
    difference is 1bp. ``variance_share`` is each PC's share of the total variance
    of yield changes in the expanding window, which is the honest read on whether
    "level" and "slope" are still the right names.
    """

    factors: pd.DataFrame
    loadings: Mapping[str, pd.DataFrame]
    variance_share: pd.DataFrame
    diagnostics: Mapping[str, SignFixDiagnostics]


def _expanding_covariances(values: np.ndarray, min_window: int) -> tuple[np.ndarray, np.ndarray]:
    """Covariance of ``values[:i+1]`` for every ``i >= min_window - 1``.

    Running sums rather than a fresh ``np.cov`` per date: the latter is O(T^2) in
    the number of dates and this runs over ~10,000 of them. The arithmetic is
    exact up to floating-point accumulation, and a test checks it against
    ``np.cov`` at several dates.
    """
    n_obs, n_cols = values.shape
    sum_x = np.zeros(n_cols)
    sum_xx = np.zeros((n_cols, n_cols))
    covariances = np.full((n_obs, n_cols, n_cols), np.nan)
    for index in range(n_obs):
        row = values[index]
        sum_x += row
        sum_xx += np.outer(row, row)
        count = index + 1
        if count < min_window:
            continue
        mean = sum_x / count
        centred = sum_xx - count * np.outer(mean, mean)
        covariances[index] = centred / (count - 1)
    defined = np.arange(n_obs) >= min_window - 1
    return covariances, defined


def _normalize_component(
    vector: np.ndarray, *, weights: np.ndarray, target: float, floor: float, label: str
) -> tuple[np.ndarray, bool]:
    """Rescale one eigenvector so ``weights . loadings == target``, and fix its sign.

    ``weights`` picks out the normalization statistic: an equal-weight average
    for the level PC, and ``(-1, 0, ..., 0, +1)`` -- long minus short -- for the
    slope PC. Returns the normalized **loading** vector and whether the sign fix
    fired.

    The sign fix and the scaling are the same operation seen twice: dividing by a
    negative statistic already flips the vector. It is written as two steps only
    so that the flip can be counted.
    """
    statistic = float(weights @ vector)
    if abs(statistic) < floor:
        raise MacroFactorError(
            f"{label}: the normalizing statistic is {statistic:.3e}, below the "
            f"{floor:.1e} floor. The eigenvector is degenerate for this normalization and "
            "rescaling it would emit a factor multiplied by an arbitrarily large number."
        )
    flipped = statistic < 0.0
    oriented = -vector if flipped else vector
    scale = target / float(weights @ oriented)
    return oriented * scale, flipped


def rate_factors(
    changes: pd.DataFrame,
    *,
    settings: RatePcaConfig,
    tenors_years: Sequence[float],
) -> RatePcaResult:
    """Expanding-window level and slope factors from daily yield changes.

    At each date ``t`` the covariance of yield changes through ``t`` inclusive is
    eigendecomposed and the top two eigenvectors are normalized to their 1bp
    targets; the factor value at ``t`` is that date's yield change projected onto
    the resulting direction. Nothing after ``t`` is used, which is what makes the
    series usable as a factor rather than as a description.

    The projection weight is the *inverse* of the loading vector's scale: if
    ``l`` are the loadings, so that ``dy ~ l . f``, then ``f = (v . dy) / k``
    where ``v`` is the unit eigenvector and ``k`` the scale applied to it. Getting
    this backwards is the failure mode the duration check in
    ``experiments.md`` rows 64-67 was registered to catch.
    """
    if changes.empty:
        raise MacroFactorError("rate_factors: no yield changes given")
    if changes.shape[1] != len(tenors_years):
        raise MacroFactorError(
            f"rate_factors: {changes.shape[1]} change column(s) against {len(tenors_years)} "
            "configured tenors"
        )
    if changes.isna().to_numpy().any():
        raise MacroFactorError(
            "rate_factors: yield changes contain NaN. Unpublished tenors must be dropped "
            "before differencing, or a change silently spans a gap in one tenor only."
        )

    values = changes.to_numpy(dtype=float)
    n_obs, n_cols = values.shape
    covariances, defined = _expanding_covariances(values, settings.expanding_min_window)

    equal = np.full(n_cols, 1.0 / n_cols)
    difference = np.zeros(n_cols)
    difference[-1], difference[0] = 1.0, -1.0
    specs = (
        (LEVEL, 0, equal, settings.level_mean_loading_bps),
        (SLOPE, 1, difference, settings.slope_loading_difference_bps),
    )

    factors = {name: np.full(n_obs, np.nan) for name, *_ in specs}
    loadings = {name: np.full((n_obs, n_cols), np.nan) for name, *_ in specs}
    shares = np.full((n_obs, len(specs)), np.nan)
    flips = dict.fromkeys(factors, 0)
    reversals = dict.fromkeys(factors, 0)
    violations = dict.fromkeys(factors, 0)
    previous: dict[str, np.ndarray | None] = dict.fromkeys(factors, None)
    counted = 0

    for index in range(n_obs):
        if not defined[index]:
            continue
        counted += 1
        eigenvalues, eigenvectors = np.linalg.eigh(covariances[index])
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        total = float(eigenvalues.sum())

        for position, (name, rank, weights, target) in enumerate(specs):
            vector = eigenvectors[:, rank]
            loading, flipped = _normalize_component(
                vector, weights=weights, target=target, floor=settings.min_normalizer, label=name
            )
            # dy ~ loading * factor, so the factor is the raw score divided by the
            # scale that was applied to the unit eigenvector.
            scale = float(loading @ vector)
            factors[name][index] = float(vector @ values[index]) / scale
            loadings[name][index] = loading
            flips[name] += int(flipped)
            shares[index, position] = eigenvalues[rank] / total if total > 0.0 else np.nan

            earlier = previous[name]
            if earlier is not None and float(earlier @ loading) < 0.0:
                reversals[name] += 1
            previous[name] = loading

            if name == LEVEL:
                same_sign = bool(np.all(loading > 0.0) or np.all(loading < 0.0))
                violations[name] += int(not same_sign)
            else:
                violations[name] += int(loading[0] * loading[-1] >= 0.0)

    dates = changes.index
    return RatePcaResult(
        factors=pd.DataFrame({name: factors[name] for name, *_ in specs}, index=dates),
        loadings={
            name: pd.DataFrame(
                {
                    str(column): loadings[name][:, position]
                    for position, column in enumerate(changes.columns)
                },
                index=dates,
            )
            for name, *_ in specs
        },
        variance_share=pd.DataFrame(
            {name: shares[:, position] for position, (name, *_) in enumerate(specs)},
            index=dates,
        ),
        diagnostics={
            name: SignFixDiagnostics(
                factor=name,
                observations=counted,
                sign_fix_fired=flips[name],
                orientation_reversals=reversals[name],
                shape_violations=violations[name],
            )
            for name, *_ in specs
        },
    )


# ---------------------------------------------------------------------------
# Expanding-window sequential Gram-Schmidt
# ---------------------------------------------------------------------------


def _expanding_residual(
    target: np.ndarray,
    regressors: np.ndarray,
    *,
    min_window: int,
    fit_intercept: bool,
    subtract_intercept: bool,
) -> np.ndarray:
    """Residual at each date from an OLS fitted on data through that date.

    Running cross-product sums for the same reason as the covariance above. The
    normal equations are solved directly and fall back to the pseudo-inverse if
    the design is singular, which can happen early in a window where two
    regressors have not yet separated.
    """
    n_obs = target.size
    design = np.column_stack([np.ones(n_obs), regressors]) if fit_intercept else regressors.copy()
    n_terms = design.shape[1]
    gram = np.zeros((n_terms, n_terms))
    moment = np.zeros(n_terms)
    residuals = np.full(n_obs, np.nan)

    for index in range(n_obs):
        row = design[index]
        gram += np.outer(row, row)
        moment += row * target[index]
        if index + 1 < max(min_window, n_terms + 1):
            continue
        try:
            beta = np.linalg.solve(gram, moment)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram) @ moment
        fitted = float(row @ beta)
        if fit_intercept and not subtract_intercept:
            # Remove only the slope terms: the intercept is the factor's own mean
            # return, a risk premium, and an orthogonalization has no business
            # taking it away.
            fitted -= float(beta[0])
        residuals[index] = target[index] - fitted
    return residuals


def orthogonalize(factors: pd.DataFrame, *, settings: OrthogonalizationConfig) -> pd.DataFrame:
    """Sequential expanding-window Gram-Schmidt on the factor series. SPEC.md 4.1.

    Processes factors in ``settings.order``. A factor with no ``against`` entry is
    passed through untouched; a factor with one is replaced by the residual of an
    expanding-window regression on the named factors, using the already-processed
    version of each. The config parser guarantees every regressor is either
    earlier in the order or never modified, so "the already-processed version" is
    never ambiguous.

    Rows where the target or any of its regressors is missing come back ``NaN``.
    They are not filled: a hedge ratio cannot be estimated against a factor that
    does not exist yet, and inventing one is the look-ahead this whole
    construction is arranged to avoid.
    """
    missing = [name for name in settings.order if name not in factors.columns]
    if missing:
        raise MacroFactorError(
            f"orthogonalize: {missing} named in the order but absent from the panel; got "
            f"{list(factors.columns)}"
        )

    out = factors.copy()
    for name in settings.order:
        regressors = settings.regressors_for(name)
        if not regressors:
            continue
        frame = pd.concat([factors[name], out[list(regressors)]], axis=1).dropna(how="any")
        if len(frame) < settings.expanding_min_window:
            raise MacroFactorError(
                f"orthogonalize: {name} shares only {len(frame)} complete observation(s) with "
                f"{list(regressors)}, below the {settings.expanding_min_window}-day minimum"
            )
        residuals = _expanding_residual(
            frame.iloc[:, 0].to_numpy(dtype=float),
            frame.iloc[:, 1:].to_numpy(dtype=float),
            min_window=settings.expanding_min_window,
            fit_intercept=settings.fit_intercept,
            subtract_intercept=settings.subtract_intercept,
        )
        out[name] = pd.Series(residuals, index=frame.index).reindex(out.index)
    return out


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------


def _cached(source: str, name: str) -> pd.DataFrame:
    manifest = cache.Manifest.load()
    return cache.read(manifest.latest(source=source, name=name), manifest=manifest)


def _daily_risk_free() -> pd.Series:
    """Ken French's daily bill rate as a decimal daily return.

    Published in percent per trading day, which is already the accrual over one
    session -- the same quantity the ETF total return covers -- so the conversion
    is a division by 100 and nothing else. No annualisation, no compounding.
    """
    return french.load_daily_rf() / 100.0


def load_excess_return(ticker: str, *, risk_free: pd.Series | None = None) -> pd.Series:
    """One ETF's daily total return less the cash rate.

    Total return comes from raw unadjusted closes plus the ex-date dividend table
    (:func:`mafrm.data.prices.total_return`), never from an adjusted close, so
    appending tomorrow's dividend cannot change yesterday's number.
    """
    key = ticker.lower()
    total = prices.total_return(
        _cached("yfinance", f"{key}_prices"), _cached("yfinance", f"{key}_actions")
    )
    rate = _daily_risk_free() if risk_free is None else risk_free
    common = total.index.intersection(rate.index)
    if len(common) == 0:
        raise MacroFactorError(f"{ticker} and the risk-free series share no dates")
    excess = total.loc[common] - rate.loc[common]
    excess.name = f"{key}_excess"
    return excess


def raw_macro_factors(
    *, end: date | None = None, config: Config | None = None
) -> tuple[pd.DataFrame, RatePcaResult]:
    """The six factor series before orthogonalization, on one calendar.

    Built over each source's **full** available history rather than from
    ``sample.start``. That is not a widening of the sample: every expanding window
    -- the PCA and the Gram-Schmidt alike -- needs a burn-in before it emits
    anything, and taking that burn-in out of history that precedes the model
    window is strictly better than taking it out of the model window itself. The
    reporting slice is applied afterwards, by the caller.

    ``end`` is the holdout boundary and is applied to every leg.
    """
    settings = config or load()
    macro = settings.model.factors.macro
    boundary = end if end is not None else settings.require_holdout_start()

    if macro.equity.source != "ken_french":
        raise MacroFactorError(
            f"equity source {macro.equity.source!r} is not implemented; SPEC.md 4.1.1 settled "
            "this on Ken French Mkt-RF"
        )
    table = french.load_daily_factors()
    if macro.equity.column not in table.columns:
        raise MacroFactorError(
            f"Ken French daily table has no {macro.equity.column!r} column; got "
            f"{list(table.columns)}"
        )
    # Published in percent per trading day, and it is already an EXCESS return.
    equity = table[macro.equity.column] / 100.0

    rate = _daily_risk_free()
    for label, spec in ((CREDIT, macro.credit), (COMMODITY, macro.commodity)):
        if spec.excess_over != "cash":
            raise MacroFactorError(
                f"{label}: excess_over={spec.excess_over!r} is not implemented. SPEC.md 4.1.1 "
                "settled both on cash, with the rates hedge done by the orthogonalization."
            )
    tickers = {
        asset.id: asset.ticker for asset in settings.universe.assets if asset.ticker is not None
    }
    for label, spec in ((CREDIT, macro.credit), (COMMODITY, macro.commodity)):
        if spec.asset_id not in tickers:
            raise MacroFactorError(
                f"{label}: {spec.asset_id!r} is not a ticker-bearing member of the frozen "
                f"universe; got {sorted(tickers)}"
            )
    credit = load_excess_return(str(tickers[macro.credit.asset_id]), risk_free=rate)
    commodity = load_excess_return(str(tickers[macro.commodity.asset_id]), risk_free=rate)

    series_id = macro.dollar.fred_series
    index_levels = _cached("fred", series_id.lower())
    column = "value" if "value" in index_levels.columns else series_id
    if column not in index_levels.columns:
        raise MacroFactorError(
            f"cached {series_id} has no value column; got {list(index_levels.columns)}"
        )
    levels = pd.to_numeric(index_levels[column], errors="coerce").dropna()
    dollar = levels.pct_change().dropna() * macro.dollar.sign

    pca = rate_factors(
        curve_yield_changes_bps(
            tenors_years=macro.rates.tenors_years, end=boundary, config=settings
        ),
        settings=macro.rates.pca,
        tenors_years=macro.rates.tenors_years,
    )

    panel = calendar.align(
        {
            EQUITY: equity,
            LEVEL: pca.factors[LEVEL],
            SLOPE: pca.factors[SLOPE],
            CREDIT: credit,
            COMMODITY: commodity,
            DOLLAR: dollar,
        },
        how="union",
        end=boundary,
    )
    return panel[list(_UNITS)], pca


@dataclass(frozen=True)
class FactorPanel:
    """The six macro factors, before and after orthogonalization.

    ``raw`` and ``orthogonal`` are the same shape and the same calendar, so the
    two correlation matrices SPEC.md 4.1 asks for side by side are computed on
    like against like. ``units`` names the unit of every column: four decimal
    returns and two in basis points of yield, which is what makes a level-factor
    regression coefficient a duration.
    """

    raw: pd.DataFrame
    orthogonal: pd.DataFrame
    units: Mapping[str, str]
    pca: RatePcaResult
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def first_valid(self) -> dict[str, pd.Timestamp]:
        """First date each orthogonalized factor is defined. The burn-in, per factor."""
        return {
            str(name): pd.Timestamp(cast("date", self.orthogonal[name].first_valid_index()))
            for name in self.orthogonal.columns
        }

    @property
    def complete(self) -> pd.DataFrame:
        """Dates on which every orthogonalized factor exists."""
        return self.orthogonal.dropna(how="any")


def macro_factor_panel(
    *, start: date | None = None, end: date | None = None, config: Config | None = None
) -> FactorPanel:
    """Build Model A's factor set. SPEC.md 4.1.

    ``start`` defaults to ``sample.start`` and ``end`` to ``holdout_start``.

    **The two expanding windows now burn in in different places, and that is the
    W2-P2 decision rather than an inconsistency.** The PCA still burns in on
    curve history that precedes the model window, because taking its burn-in out
    of history is free. The Gram-Schmidt does not: when
    ``orthogonalization.expanding_window_start`` is ``"sample.start"`` the raw
    panel is sliced to the model window *before* it is orthogonalized, so no
    hedge ratio is ever estimated on data the model window does not contain.

    The reason is a measurement W2-P1 already had in hand. The two normalized
    rate PCs correlate about -0.196 over the pre-2007 burn-in and about +0.124
    over the model window; a hedge ratio fitted mostly on the former has the
    wrong sign for the latter and *adds* level exposure to the slope factor
    instead of removing it. Estimating a hedge where the relationship carries
    the opposite sign is wrong whatever correlation it produces, which is what
    makes this a design decision rather than a choice between outcomes.
    ``experiments.md`` row 69 records that it was taken before the resulting
    correlation was known.

    It costs ``rates_slope`` and ``commodity`` their first ``expanding_min_window``
    days of the model window. The PCA's window is deliberately NOT moved with it:
    that would push level and slope to the same edge and then push everything
    hedged against them a further year out, turning two ragged edges into three.
    """
    settings = config or load()
    boundary = end if end is not None else settings.require_holdout_start()
    window_start = start if start is not None else settings.model.sample.start
    scheme = settings.model.factors.macro.orthogonalization

    raw, pca = raw_macro_factors(end=boundary, config=settings)
    if scheme.burns_in_inside_the_sample:
        raw = raw.loc[raw.index >= pd.Timestamp(settings.model.sample.start)]
    orthogonal = orthogonalize(raw, settings=scheme)

    mask = raw.index >= pd.Timestamp(window_start)
    raw, orthogonal = raw[mask], orthogonal[mask]
    if raw.empty:
        raise MacroFactorError(f"no factor observations between {window_start} and {boundary}")
    return FactorPanel(
        raw=raw,
        orthogonal=orthogonal,
        units=dict(_UNITS),
        pca=pca,
        start=pd.Timestamp(raw.index[0]),
        end=pd.Timestamp(raw.index[-1]),
    )


def correlation_matrices(panel: FactorPanel) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Pre- and post-orthogonalization correlations, on **identical** rows.

    Both matrices are computed on the dates where every orthogonalized factor
    exists. Using each matrix's own complete cases would let the "before" matrix
    include the burn-in window that the "after" matrix cannot have, and the
    difference between them would then be partly a difference of sample -- which
    is precisely the comparison SPEC.md 4.1 asks the report to make honestly.
    """
    rows = panel.complete.index
    return panel.raw.loc[rows].corr(), panel.orthogonal.loc[rows].corr(), len(rows)


# ---------------------------------------------------------------------------
# Conditioning: the number week 3 actually cares about
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    """One targeted pair, judged on the criterion SPEC.md 4.1.2 leaves standing.

    Only pairs the orthogonalization scheme actually acts on appear here. The
    all-pairs version of this bar was withdrawn as invalid -- a pair the scheme
    does not touch cannot be a gate on the scheme, and forcing every pair down
    would drive the factor correlation matrix toward diagonal and leave the
    eigenfactor adjustment nothing to correct.
    """

    left: str
    right: str
    before: float
    after: float
    limit: float
    exempt: bool
    exempt_reason: str | None

    @property
    def meets_criterion(self) -> bool:
        return abs(self.after) <= self.limit

    @property
    def passed(self) -> bool:
        return self.meets_criterion or self.exempt

    @property
    def fall(self) -> float:
        """Reduction in absolute correlation, as a fraction of where it started."""
        return 1.0 - abs(self.after) / abs(self.before) if self.before else float("nan")


def gate_results(panel: FactorPanel, *, config: Config | None = None) -> tuple[GateResult, ...]:
    """The four targeted pairs, before and after. SPEC.md 4.1.2."""
    settings = (config or load()).model.factors.macro.orthogonalization
    before, after, _ = correlation_matrices(panel)
    results: list[GateResult] = []
    for target, regressors in settings.against.items():
        for regressor in regressors:
            pair = frozenset((target, regressor))
            exemption = next((item for item in settings.exempt_pairs if item.pair == pair), None)
            results.append(
                GateResult(
                    left=target,
                    right=regressor,
                    before=float(
                        before.to_numpy()[before.index.get_loc(target)][
                            before.columns.get_loc(regressor)
                        ]
                    ),
                    after=float(
                        after.to_numpy()[after.index.get_loc(target)][
                            after.columns.get_loc(regressor)
                        ]
                    ),
                    limit=settings.targeted_pair_max_abs_correlation,
                    exempt=exemption is not None,
                    exempt_reason=exemption.reason if exemption else None,
                )
            )
    return tuple(results)


def condition_numbers(factors: pd.DataFrame, *, halflife: int, min_observations: int) -> pd.Series:
    """Condition number of the EWMA factor correlation matrix, through time.

    The diagnostic SPEC.md 4.1.2 puts in place of the withdrawn pairwise bar, and
    the direct measurement of CLAUDE.md failure mode 2 -- the covariance matrix
    going near-singular in exactly the crises that matter. At each date the
    correlation matrix is estimated on every observation through that date with
    exponentially decaying weights at ``halflife``, and its condition number
    (largest over smallest eigenvalue) is recorded.

    **This calls the covariance pipeline's own estimators rather than
    reimplementing them** -- :func:`~mafrm.risk.covariance.ewma_second_moment`
    and :func:`~mafrm.risk.covariance.correlation_from_covariance` -- so the
    matrix diagnosed here is bit-identical to the one SPEC.md 5.1 builds at the
    same half-life. It was not always: through W3-P1 this function demeaned and
    carried the ``1 - w'w`` reliability correction, and the pipeline took its
    moments about zero (SPEC.md 5.1.1). **That was re-based in W3-P1b as a
    consistency fix, not a moved goalpost.** Two code paths computing the same
    statistic two ways is a trap: a future session comparing them would find a
    discrepancy with no explanation in either file, and the cost of removing it
    was a re-published diagnostic against a divergence `experiments.md` row 95
    had already bounded at 0.005 of a correlation.

    Only the demeaning actually moved anything. The ``1 - w'w`` correction was a
    no-op for this statistic all along, because it scales the whole covariance
    matrix by a scalar and a correlation matrix is invariant to that -- worth
    saying, because it means the change here is the one change SPEC.md 5.1.1
    ruled on and not two changes at once.

    Row 68's published figures stand in the log as published. The live numbers in
    ``reports/factor_correlations.md`` are the re-based ones.

    **Reported, never gated.** A condition number is a property of the market in
    that window -- factors genuinely do co-move in a crisis -- not a property of
    the implementation, so there is no value of it that would mean the code is
    wrong. What it is for is stating the size of the problem SPEC.md 5.3's
    eigenfactor adjustment exists to correct: when the smallest eigenvalues of
    ``F`` are small they are also the most poorly estimated, an optimizer will
    load onto exactly those directions, and the resulting risk understatement is
    what the Monte Carlo adjustment corrects.

    ``min_observations`` is a diagnostic minimum and deliberately shorter than
    the EWMA effective sample size at this half-life; early estimates therefore
    carry near-flat weights and are closer to a sample correlation than to a
    converged EWMA. The report says so rather than hiding it.
    """
    complete = factors.dropna(how="any")
    if len(complete) < min_observations:
        raise MacroFactorError(
            f"condition_numbers: {len(complete)} complete observation(s), below the "
            f"{min_observations}-day minimum"
        )
    values = complete.to_numpy(dtype=float)
    out = np.full(len(complete), np.nan)
    for index in range(min_observations - 1, len(complete)):
        window = values[: index + 1]
        moment = ewma_second_moment(window, halflife=float(halflife))
        correlation = correlation_from_covariance(moment)
        eigenvalues = np.linalg.eigvalsh(correlation)
        smallest = float(eigenvalues.min())
        out[index] = float(eigenvalues.max()) / smallest if smallest > 0.0 else np.inf
    series = pd.Series(out, index=complete.index, name="condition_number")
    return series.dropna()


@dataclass(frozen=True)
class Conditioning:
    """Condition number of the factor correlation matrix, raw against orthogonalized.

    Both are reported because the orthogonalization's real justification is that
    it improves conditioning, and that is a claim worth measuring rather than
    asserting.
    """

    raw: pd.Series
    orthogonal: pd.Series
    halflife: int
    min_observations: int

    def in_window(self, start: date, end: date) -> tuple[pd.Series, pd.Series]:
        """Both series restricted to one episode, ``end`` inclusive."""
        low, high = pd.Timestamp(start), pd.Timestamp(end)
        return (
            self.raw[(self.raw.index >= low) & (self.raw.index <= high)],
            self.orthogonal[(self.orthogonal.index >= low) & (self.orthogonal.index <= high)],
        )


def conditioning(panel: FactorPanel, *, config: Config | None = None) -> Conditioning:
    """Condition numbers for the raw and orthogonalized factor sets. SPEC.md 4.1.2.

    The half-life is read from ``covariance.factor_correlation_halflife.short``,
    not from a key of its own, so this diagnostic sees the same matrix the
    covariance pipeline will estimate rather than a differently-smoothed
    lookalike.
    """
    settings = config or load()
    spec = settings.model.factors.macro.condition_number
    halflife = settings.model.covariance.factor_correlation_halflife.short
    return Conditioning(
        raw=condition_numbers(panel.raw, halflife=halflife, min_observations=spec.min_observations),
        orthogonal=condition_numbers(
            panel.orthogonal, halflife=halflife, min_observations=spec.min_observations
        ),
        halflife=halflife,
        min_observations=spec.min_observations,
    )


# ---------------------------------------------------------------------------
# The duration check: does the normalization mean what it claims?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DurationEstimate:
    """One instrument's measured duration against its pre-registered prediction.

    ``slope`` is the OLS coefficient of the asset's excess return **in basis
    points** on the level factor **in basis points**, which reads as minus the
    effective duration in years. ``slope_with_slope_factor`` adds the slope factor
    as a second regressor: reported as a diagnostic, never as the headline,
    because experiments.md rows 64-67 registered the univariate number.
    """

    id: str
    observations: int
    start: pd.Timestamp
    end: pd.Timestamp
    slope: float
    standard_error: float
    r_squared: float
    slope_with_slope_factor: float
    predicted: float
    band: tuple[float, float]

    @property
    def passed(self) -> bool:
        low, high = self.band
        return low <= self.slope <= high

    @property
    def implied_duration(self) -> float:
        return -self.slope

    def render(self) -> str:
        low, high = self.band
        return (
            f"{self.id}: slope {self.slope:+.2f} (predicted {self.predicted:+.2f}, band "
            f"[{low:+.2f}, {high:+.2f}]) -- {'pass' if self.passed else 'FAIL'}; "
            f"implied duration {self.implied_duration:.2f}y, R^2 {self.r_squared:.3f}, "
            f"n={self.observations:,}"
        )


def _instrument_excess_return(
    instrument: DurationCheckInstrument, *, risk_free: pd.Series, config: Config
) -> pd.Series:
    """The excess-return series one duration-check instrument is measured on."""
    if instrument.kind == "synthetic_zero":
        assert instrument.maturity_years is not None  # enforced by the config parser
        frame = gsw.constant_maturity_return(
            instrument.maturity_years, risk_free_pct=gsw.load_risk_free()
        )
        return frame["excess_return"].dropna()
    assert instrument.ticker is not None  # enforced by the config parser
    return load_excess_return(instrument.ticker, risk_free=risk_free)


def _ols(target: np.ndarray, regressors: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Full-sample OLS with an intercept. Returns coefficients, standard errors, R^2."""
    design = np.column_stack([np.ones(target.size), regressors])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ beta
    dof = target.size - design.shape[1]
    variance = float(residual @ residual) / dof
    covariance = variance * np.linalg.pinv(design.T @ design)
    centred = target - target.mean()
    r_squared = 1.0 - float(residual @ residual) / float(centred @ centred)
    return beta, np.sqrt(np.diag(covariance)), r_squared


def duration_check(
    panel: FactorPanel, *, config: Config | None = None
) -> tuple[DurationEstimate, ...]:
    """Regress each pre-registered instrument on the normalized level factor.

    The test of the normalization, registered in ``experiments.md`` rows 64-67
    before this module existed. Both sides are in basis points, so the slope is
    minus the effective duration in years and a ten-year zero should return about
    -10. The regression is full-sample OLS over the panel's own window: one
    number per instrument at maximum precision, which is what a normalization
    check wants. The rolling EWMA betas of SPEC.md 4.1 are W2-P2's deliverable
    and a different question.

    The level factor used is the **raw** one. Level is first in the
    orthogonalization order and is never itself orthogonalized, so raw and
    orthogonal are the same series; reading it from ``raw`` says so explicitly.
    """
    settings = config or load()
    check = settings.model.factors.macro.duration_check
    rate = _daily_risk_free()
    level = panel.raw[LEVEL].dropna()
    slope_factor = panel.raw[SLOPE]

    estimates: list[DurationEstimate] = []
    for instrument in check.instruments:
        returns = _instrument_excess_return(instrument, risk_free=rate, config=settings)
        asset_bps = to_basis_points(returns)
        assert isinstance(asset_bps, pd.Series)
        frame = calendar.align(
            {"asset": asset_bps, LEVEL: level, SLOPE: slope_factor}, how="union"
        ).dropna(how="any")
        frame = frame.loc[(frame.index >= panel.start) & (frame.index <= panel.end)]
        if len(frame) < 2:
            raise MacroFactorError(
                f"{instrument.id}: {len(frame)} overlapping observation(s) with the level "
                "factor; nothing to regress"
            )
        target = frame["asset"].to_numpy(dtype=float)
        beta, errors, r_squared = _ols(target, frame[[LEVEL]].to_numpy(dtype=float))
        bivariate, _, _ = _ols(target, frame[[LEVEL, SLOPE]].to_numpy(dtype=float))
        estimates.append(
            DurationEstimate(
                id=instrument.id,
                observations=len(frame),
                start=pd.Timestamp(frame.index[0]),
                end=pd.Timestamp(frame.index[-1]),
                slope=float(beta[1]),
                standard_error=float(errors[1]),
                r_squared=r_squared,
                slope_with_slope_factor=float(bivariate[1]),
                predicted=instrument.predicted_duration,
                band=instrument.band(check.tolerance_fraction),
            )
        )
    return tuple(estimates)


def _main() -> int:
    """``make model``: build the factor set and report what it did."""
    panel = macro_factor_panel()
    _, _, rows = correlation_matrices(panel)

    print(f"macro factor panel {panel.start.date()}..{panel.end.date()}, {len(panel.raw):,} dates")
    for diagnostics in panel.pca.diagnostics.values():
        print(f"  {diagnostics.render()}")
    print(f"  complete rows after orthogonalization: {rows:,}")
    for name, first in panel.first_valid.items():
        print(f"    {name}: first defined {first.date()}")

    print("  gate -- the four targeted pairs (SPEC.md 4.1.2):")
    for gate in gate_results(panel):
        verdict = (
            "pass" if gate.meets_criterion else ("exempt by ruling" if gate.exempt else "FAIL")
        )
        print(
            f"    {gate.left}/{gate.right}: {gate.before:+.3f} -> {gate.after:+.3f} "
            f"({100 * gate.fall:+.0f}%) -- {verdict}"
        )

    numbers = conditioning(panel)
    print(f"  conditioning (reported, never gated; EWMA half-life {numbers.halflife}d):")
    print(
        f"    raw          median {numbers.raw.median():.2f}, max {numbers.raw.max():.2f} "
        f"({numbers.raw.index[0].date()}..)"
    )
    print(
        f"    orthogonal   median {numbers.orthogonal.median():.2f}, "
        f"max {numbers.orthogonal.max():.2f} ({numbers.orthogonal.index[0].date()}..)"
    )

    for estimate in duration_check(panel):
        print(f"  {estimate.render()}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(_main())
