"""The hybrid. SPEC.md 4.3 -- PCA on Model A's residuals.

Model A leaves a residual ``u_it`` per asset per date. SPEC.md 4.3 asks for a
PCA on those residuals, with the top 1-2 components appended to the six named
macro factors as unnamed ones, and for Model A's R-squared reported alone and
with them. The point is not the R-squared: it is that a residual PC which loads
heavily and whose variance spikes is the crowding/deleveraging signal that no
pre-specified factor set contains -- the August 2007 case.

FIVE THINGS ARE LOAD-BEARING.

**The PCA is on the residual CORRELATION, not the covariance** (SPEC.md 4.3.3).
The residuals are in basis points per day and their cross-asset volatilities
span about an order of magnitude. A covariance PCA on that panel returns
whichever asset is most volatile as its leading eigenvector and calls it a
factor. That is experiments.md row 96's units artefact arriving a second time,
and it is the same reasoning that made the eigenfactor stage scale-invariant
(SPEC.md 5.3.1). There is no switch.

**Nothing is denoised.** SPEC.md 4.3 asks for a PCA and does not ask for
Marchenko-Pastur, and SPEC.md 4.2.7 already measured what MP denoising does at
this panel's ``N/T``: it loses to the sample correlation by 57% on out-of-sample
minimum-variance volatility, because averaging eigenvalues that were never noise
destroys real structure. Importing it here would import a technique this project
has measured as harmful in this regime.

**The window is expanding and the burn-in is not a new number.** Read from
``factors.macro.rates.pca.expanding_min_window``, as Model B reads it. A
full-sample eigendecomposition decides what the factor was on its first day
using the whole history; SPEC.md 4.1.3 ruled that out for the rates PCA and
SPEC.md 4.2.1 for Model B.

**The four synthetic government zeros are regressed on their DURATION LEG
ALONE.** experiments.md rows 79-82: a constant-maturity zero's carry and
roll-down cannot be spanned by a factor set built from yield *changes*, so they
land in the intercept -- +0.71, +2.00, +3.29, +4.71 %/yr at 2/5/10/30. Left in,
a residual PC would load on the government sleeve for a reason with nothing to
do with real rates, and row 83 would be uninformative by its own registered
criterion. W2-P3 fixed this control on 2026-08-30, before any residual PC
existed. ``tips_10y`` is deliberately not in the set: its alpha was never
flagged, so there is no measured term premium in it to strip.

**Two sign conventions, for two different jobs, and both are reported.** The
factor series handed downstream uses SPEC.md 4.2.3's continuity convention --
each date aligned against the previous, seeded on the largest-magnitude loading
-- because a factor series must not flip sign on a date when one loading passes
through zero. Row 83's sign clause instead orients each date so that gold's
loading is positive, which is the only way to state "the correlation must be
negative" about an object whose sign is unidentified. The share of dates on
which the two agree is reported: a low share would mean gold's loading is not
stably signed and the test orientation is close to a coin flip.

This module reads the cache only. CLAUDE.md invariant 1: no network call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import pandas as pd

from mafrm.config import Config, HybridConfig, load
from mafrm.data import gsw
from mafrm.factors import betas, macro, statistical
from mafrm.risk.checks import assert_psd

__all__ = [
    "HybridError",
    "ResidualPanel",
    "ResidualPcPanel",
    "Row83Result",
    "augmented_r_squared",
    "build",
    "comparand_yield_changes",
    "evaluate_row_83",
    "factor_panel",
    "hybrid_regressands",
    "loading_rank_null",
    "monthly_variance_share",
    "residual_panel",
    "residual_pcs",
    "union_null_probability",
]

#: The universe ``construction`` whose regressand is the duration leg alone.
#: Mirrors :data:`mafrm.config._GOVERNMENT_ZERO_CONSTRUCTION`; the string is the
#: universe's own vocabulary and appears here because this is where the
#: substitution happens.
_GOVERNMENT_ZERO = "synthetic_gsw_zero_curve"


class HybridError(ValueError):
    """The hybrid cannot be built from what is cached."""


# ---------------------------------------------------------------------------
# What Model A is regressed on, once the W2-P3 control is applied
# ---------------------------------------------------------------------------


def hybrid_regressands(
    *, config: Config | None = None
) -> tuple[dict[str, pd.Series], tuple[str, ...]]:
    """Per-asset regressands for the hybrid, with the government control applied.

    Identical to :func:`mafrm.factors.betas.asset_excess_returns` except that
    every universe member built as ``synthetic_gsw_zero_curve`` is replaced by
    its ``duration_effect`` leg -- the excess return with carry and roll-down
    taken out (SPEC.md 3.2's exact three-way decomposition).

    Returns ``(regressands, substituted)`` so that the substitution is visible
    to the caller and to the report rather than being applied silently.
    """
    settings = config or load()
    scheme = settings.model.factors.hybrid
    if scheme.government_zero_return_leg not in ("duration_effect",):  # pragma: no cover
        raise HybridError(f"unsupported leg {scheme.government_zero_return_leg!r}")

    returns = betas.asset_excess_returns(config=settings)
    risk_free_pct = gsw.load_risk_free()
    substituted: list[str] = []
    for asset in settings.universe.assets:
        if asset.construction != _GOVERNMENT_ZERO or asset.id not in returns:
            continue
        if asset.maturity_years is None:  # pragma: no cover - parser enforces it
            raise HybridError(f"{asset.id}: a synthetic curve point needs a maturity")
        frame = gsw.constant_maturity_return(
            asset.maturity_years, kind="nominal", risk_free_pct=risk_free_pct
        )
        leg = frame[scheme.government_zero_return_leg].dropna()
        if leg.empty:  # pragma: no cover - the curve would have to be empty
            raise HybridError(f"{asset.id}: the {scheme.government_zero_return_leg} leg is empty")
        returns[asset.id] = leg.rename(asset.id)
        substituted.append(asset.id)
    if not substituted:  # pragma: no cover - the frozen universe has four
        raise HybridError(
            "hybrid_regressands: no universe member is built as "
            f"{_GOVERNMENT_ZERO!r}, so W2-P3's control could not be applied. That control was "
            "registered before any residual PC existed and is not optional."
        )
    return returns, tuple(sorted(substituted))


# ---------------------------------------------------------------------------
# The residuals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualPanel:
    """Model A's residuals ``u_it``, in basis points per day.

    ``residuals`` is a complete-cases ``date x asset`` frame: a correlation over
    a ragged panel is either pairwise -- not guaranteed PSD -- or filled, which
    invents returns. How many dates that costs is reported, not swallowed
    (:attr:`dates_dropped_by_completeness`).

    The residual at ``t`` is taken from the regression estimated on the window
    *ending at* ``t``, so it is the fitted equation's residual at the last point
    of its own window. Nothing after ``t`` enters it.
    """

    residuals: pd.DataFrame
    r_squared: pd.DataFrame
    factors: pd.DataFrame
    assets: tuple[str, ...]
    factor_names: tuple[str, ...]
    substituted_assets: tuple[str, ...]
    dates_dropped_by_completeness: int
    #: The ``(date, asset)`` exposure panel these residuals were taken from.
    #: Carried rather than left to a caller to rebuild: SPEC.md 5.5(b) regresses
    #: on exactly this design, and a second `build_exposures` call at a second
    #: call site is a second chance for the two to disagree about the W2-P3
    #: duration-leg substitution.
    exposures: pd.DataFrame
    #: The REGRESSANDS these residuals were taken from, in basis points per day,
    #: on the same index and columns as :attr:`residuals`. Carried for the same
    #: reason :attr:`exposures` is, and it matters more here: SPEC.md 6.1's bias
    #: statistic standardises a RAW realised return, so W4-P2 needs the exact
    #: series the exposures were fitted to -- including :attr:`substituted_assets`'
    #: duration-effect legs. A caller re-deriving it from `asset_excess_returns`
    #: would silently score the four government zeros against a different return
    #: than the one their exposures and residuals describe.
    returns: pd.DataFrame

    @property
    def observations(self) -> int:
        return int(self.residuals.shape[0])

    def exposure_matrix(self, date: pd.Timestamp) -> pd.DataFrame:
        """The ``N x K`` design at one date, rows in residual-column order.

        SPEC.md 15.2's ``specific_risk`` takes ``exposures`` as ``N x K``; this
        is where the ``(date, asset)`` panel becomes one.
        """
        block = cast("pd.DataFrame", self.exposures.xs(date, level="date"))
        return block[list(self.factor_names)].reindex(list(self.residuals.columns))

    @property
    def variables(self) -> int:
        return int(self.residuals.shape[1])


def residual_panel(
    panel: macro.FactorPanel | None = None, *, config: Config | None = None
) -> ResidualPanel:
    """Model A's residuals over the frozen universe. SPEC.md 4.3.

    The exposures are re-estimated here rather than read from
    :func:`mafrm.factors.betas.exposure_panel`, because the regressand differs:
    W2-P3's control puts the four government zeros on their duration leg. Every
    other asset is regressed exactly as Model A regresses it, and
    ``tests/test_hybrid.py`` asserts that the untouched assets reproduce
    ``exposure_panel``'s R-squared to floating point.
    """
    settings = config or load()
    factors = panel if panel is not None else macro.macro_factor_panel(config=settings)
    regressands, substituted = hybrid_regressands(config=settings)
    boundary = settings.require_holdout_start()
    trimmed = {
        name: series.loc[(series.index >= factors.start) & (series.index < pd.Timestamp(boundary))]
        for name, series in regressands.items()
    }
    exposures = betas.build_exposures(
        trimmed,
        factors.complete,
        units=factors.units,
        settings=settings.model.factors.macro.beta,
    )

    scaled = betas.regression_units(factors.complete.dropna(how="any"), factors.units)
    columns = list(exposures.factors)
    residuals: dict[str, pd.Series] = {}
    for asset in exposures.assets:
        block = exposures.exposures.xs(asset, level="asset")[columns]
        design = scaled.reindex(block.index)[columns]
        fitted = (block * design).sum(axis=1, skipna=False)
        realised = trimmed[asset].reindex(block.index) * macro._BPS_PER_UNIT
        residuals[asset] = realised - exposures.alpha[asset].reindex(block.index) - fitted

    frame = pd.DataFrame(residuals).sort_index()
    complete = frame.dropna(how="any")
    if complete.empty:
        raise HybridError(
            "residual_panel: no date has a residual for every asset. The exposure panel's left "
            "edge is ragged by construction and is reported rather than filled."
        )
    dropped = int(frame.shape[0] - complete.shape[0])
    return ResidualPanel(
        residuals=complete,
        r_squared=exposures.r_squared.reindex(complete.index),
        factors=scaled.reindex(complete.index),
        assets=tuple(exposures.assets),
        factor_names=tuple(columns),
        substituted_assets=substituted,
        dates_dropped_by_completeness=dropped,
        exposures=exposures.exposures.loc[
            exposures.exposures.index.get_level_values("date").isin(complete.index)
        ],
        returns=pd.DataFrame(
            {
                asset: trimmed[asset].reindex(complete.index) * macro._BPS_PER_UNIT
                for asset in complete.columns
            },
            index=complete.index,
        ),
    )


# ---------------------------------------------------------------------------
# The residual PCs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualPcPanel:
    """The residual PCs, their loadings, and the two sign conventions.

    ``factors`` is ``date x pc1..pcK`` under the CONTINUITY convention; the
    factor value at ``t`` is ``f_kt = v_k' z_t`` where ``z_t = u_t / s_t`` is
    that date's residual standardised by the expanding-window deviations the
    correlation was taken with respect to. So each PC is a fixed portfolio of
    standardised residuals on each date.

    :attr:`test_factors` is the same series re-oriented per date so that
    :attr:`orientation_asset`'s loading is positive -- row 83's sign convention,
    and only that. :attr:`orientation_agreement` is the share of dates on which
    the two conventions coincide, which is the statistic that says whether the
    test orientation is stable or a coin flip.

    **Both are DERIVED from** ``factors`` **and** ``loadings`` **rather than
    stored**, and that is what makes row 83's sign clause testable. LAPACK's
    eigenvector sign is arbitrary, so a different build could hand back ``-v``
    and ``-f`` together; negating both must leave every clause's verdict
    untouched. Storing the test series would let a stale copy survive that
    negation and the invariance would be asserted against a constant.
    """

    variant: str
    factors: pd.DataFrame
    loadings: dict[str, pd.DataFrame]
    eigenvalues: pd.DataFrame
    standardised: pd.DataFrame
    assets: tuple[str, ...]
    components: int
    expanding_min_window: int
    orientation_asset: str
    sign_flips: int
    component_alignment: pd.Series

    @property
    def orientation_signs(self) -> pd.DataFrame:
        """``+1`` where the continuity convention already makes gold positive.

        A zero loading is mapped to ``+1`` rather than to ``0``: it is a
        measure-zero case in floating point, and mapping it to zero would
        silently drop that date from the correlation instead of orienting it.
        """
        return pd.DataFrame(
            {
                name: np.where(frame[self.orientation_asset].to_numpy(dtype=float) < 0.0, -1.0, 1.0)
                for name, frame in self.loadings.items()
            },
            index=self.factors.index,
        )

    @property
    def test_factors(self) -> pd.DataFrame:
        """Row 83's orientation: each date flipped so gold's loading is positive."""
        return self.factors * self.orientation_signs

    @property
    def orientation_agreement(self) -> dict[str, float]:
        """Share of dates on which the test and continuity conventions coincide."""
        signs = self.orientation_signs
        return {str(name): float((signs[name] > 0.0).mean()) for name in signs.columns}

    @property
    def variance_share(self) -> pd.DataFrame:
        """Each component's share of the trace, per date. The trace is ``N``."""
        total = float(len(self.assets))
        return self.eigenvalues.iloc[:, : self.components] / total


def residual_pcs(
    residuals: pd.DataFrame,
    *,
    components: int,
    settings: HybridConfig,
) -> ResidualPcPanel:
    """Expanding-window PCA of the residual correlation. SPEC.md 4.3.

    At each date ``t`` the sample correlation of the residuals through ``t``
    inclusive is formed, eigendecomposed, and its leading ``components``
    directions are kept. Nothing after ``t`` is used.

    PSD is asserted on every date's correlation before it is decomposed. A
    sample correlation is positive semi-definite by construction, so the check
    is not expected to fire -- which is exactly why it is cheap to keep, and
    CLAUDE.md invariant 4 asks for the assertion at every stage rather than at
    the ones that seem likely to fail.
    """
    if components < 1:
        raise HybridError(f"residual_pcs: components must be at least 1, got {components}")
    if components > residuals.shape[1]:
        raise HybridError(
            f"residual_pcs: {components} component(s) from a {residuals.shape[1]}-asset panel"
        )
    values = residuals.to_numpy(dtype=float)
    n_obs, n_vars = values.shape
    minimum = settings.expanding_min_window
    if n_obs < minimum:
        raise HybridError(
            f"residual_pcs: {n_obs} observation(s) is below the {minimum}-day expanding-window "
            "minimum, which is read from factors.macro.rates.pca.expanding_min_window"
        )

    correlations, deviations, defined = statistical.expanding_correlations(
        values, min_window=minimum
    )

    dates: list[pd.Timestamp] = []
    projections: list[np.ndarray] = []
    loading_rows: list[np.ndarray] = []
    spectra: list[np.ndarray] = []
    standardised_rows: list[np.ndarray] = []
    alignments: list[float] = []
    previous: np.ndarray | None = None
    flips = 0

    for index in range(n_obs):
        if not defined[index]:
            continue
        correlation = correlations[index]
        assert_psd(correlation, "hybrid.residual_correlation", expected_size=n_vars)
        eigenvalues, vectors = np.linalg.eigh(correlation)
        eigenvalues = eigenvalues[::-1]
        vectors = vectors[:, ::-1]
        oriented, flipped, alignment = statistical.orient_eigenvectors(
            vectors, previous, survivors=components
        )
        previous = oriented
        flips += flipped

        standardised = values[index] / deviations[index]
        projections.append(standardised @ oriented[:, :components])
        loading_rows.append(oriented[:, :components])
        spectra.append(eigenvalues)
        standardised_rows.append(standardised)
        alignments.append(alignment)
        dates.append(pd.Timestamp(cast("date", residuals.index[index])))

    index_dates = pd.DatetimeIndex(dates)
    columns = [f"residual_pc{position + 1}" for position in range(components)]
    factors = pd.DataFrame(np.vstack(projections), index=index_dates, columns=columns)
    loadings = {
        columns[component]: pd.DataFrame(
            np.vstack([row[:, component] for row in loading_rows]),
            index=index_dates,
            columns=list(residuals.columns),
        )
        for component in range(components)
    }

    asset = settings.row_83.orientation_asset
    if asset not in residuals.columns:
        raise HybridError(
            f"residual_pcs: {asset!r} is not in the residual panel; row 83's orientation asset "
            f"must be one of {list(residuals.columns)}"
        )

    return ResidualPcPanel(
        variant=settings.estimator,
        factors=factors,
        loadings=loadings,
        eigenvalues=pd.DataFrame(
            np.vstack(spectra),
            index=index_dates,
            columns=[f"lambda{position + 1}" for position in range(n_vars)],
        ),
        standardised=pd.DataFrame(
            np.vstack(standardised_rows), index=index_dates, columns=list(residuals.columns)
        ),
        assets=tuple(str(column) for column in residuals.columns),
        components=components,
        expanding_min_window=minimum,
        orientation_asset=asset,
        sign_flips=flips,
        component_alignment=pd.Series(alignments, index=index_dates, name="component_alignment"),
    )


# ---------------------------------------------------------------------------
# What SPEC.md 4.3 asks to be reported
# ---------------------------------------------------------------------------


def augmented_r_squared(
    panel: ResidualPanel,
    pcs: Sequence[ResidualPcPanel],
    *,
    config: Config | None = None,
) -> pd.DataFrame:
    """Model A's R-squared alone and with the residual PCs appended. SPEC.md 4.3.

    **Every column is estimated on the SAME rows, and that is not a detail.**
    The residual PCs carry their own expanding-window burn-in, so their panel
    starts about 251 trading days after Model A's and the complete-cases filter
    has dropped dates in between. A rolling window is 252 ROWS, not 252 calendar
    days, so a window ending on a given date covers a different span in the two
    panels -- and the baseline is therefore re-estimated here on the augmented
    design's own index rather than read from
    :attr:`ResidualPanel.r_squared`. Reading it across cost three assets an
    apparently NEGATIVE gain from adding a regressor, which is impossible within
    one regression and was the signal that the two runs were not like-for-like
    (experiments.md, W3-P6 control C4).

    **The comparison is DESCRIPTIVE AND IN-SAMPLE BY CONSTRUCTION, and saying so
    is the whole of its honest reading.** The PC at date ``t`` is a linear
    combination of that same date's residuals, so on a shared design the
    augmented weighted R-squared cannot fall. The number answers "how much of
    what the six factors missed is common across assets", which is worth
    reporting, and it does NOT answer "would this factor have helped out of
    sample", which nothing here claims. A forecast comparison is the covariance
    pipeline's job in week 4.

    One row per asset: the median rolling R-squared under Model A, then under
    Model A plus each of the ``component_counts``.
    """
    settings = config or load()
    beta = settings.model.factors.macro.beta
    regressands, _ = hybrid_regressands(config=settings)

    common = panel.factors.index
    for pc in pcs:
        common = common.intersection(pc.factors.index)
    base_design = panel.factors.loc[common]

    designs: list[tuple[str, pd.DataFrame]] = [("model_a", base_design)]
    designs += [
        (f"model_a_plus_{pc.components}", base_design.join(pc.factors, how="inner")) for pc in pcs
    ]

    frames: dict[str, pd.Series] = {}
    for label, design in designs:
        medians: dict[str, float] = {}
        for asset in panel.assets:
            series = regressands[asset].reindex(design.index).dropna()
            _, _, _, r_squared = betas.rolling_exposures(
                (series * macro._BPS_PER_UNIT).rename(asset),
                design,
                window=beta.window,
                halflife=beta.halflife,
                fit_intercept=beta.fit_intercept,
            )
            medians[asset] = float(r_squared.median())
        frames[label] = pd.Series(medians)

    out = pd.DataFrame(frames)
    for pc in pcs:
        column = f"model_a_plus_{pc.components}"
        out[f"gain_{pc.components}"] = out[column] - out["model_a"]
    return out.sort_index()


def monthly_variance_share(pcs: ResidualPcPanel) -> pd.DataFrame:
    """Each residual PC's realised share of residual variance, by calendar month.

    SPEC.md 4.3 wants the variance share charted through 2008, 2020 and 2022,
    which needs a time-LOCAL measure: the expanding-window eigenvalue share is a
    whole-history quantity and is nearly flat by the end of the sample, so a
    spike would be invisible in it.

    **The buckets are calendar months and therefore NON-OVERLAPPING**, which is
    deliberate and is the reason no window length appears here. CLAUDE.md
    failure mode 9 is about rolling windows read as independent observations;
    a rolling variance share stepped daily would share 251 of its 252 days with
    its neighbour and would need an overlap caveat printed beside every reading.
    Calendar months need none, and they cost no new parameter.

    The share is ``sum_t f_kt^2 / sum_t sum_i z_it^2`` within the month, where
    ``z`` is the standardised residual. The denominator is the realised trace,
    so a share equals the eigenvalue share in expectation over the window the
    eigenvector was estimated on, and departs from it exactly when the month is
    unusual -- which is the signal.
    """
    months = pd.DatetimeIndex(pcs.factors.index).to_period("M")
    numerators = (pcs.factors**2).groupby(months).sum()
    denominator = (pcs.standardised**2).sum(axis=1).groupby(months).sum()
    shares = numerators.div(denominator, axis=0)
    shares["observations"] = pcs.factors.groupby(months).size()
    shares.index = pd.PeriodIndex(shares.index).to_timestamp()
    shares.index.name = "month"
    return shares


# ---------------------------------------------------------------------------
# Row 83
# ---------------------------------------------------------------------------


def comparand_yield_changes(
    *, config: Config | None = None, end: date | None = None
) -> pd.DataFrame:
    """Daily 10-year REAL and NOMINAL yield changes in basis points. Row 83's placebo.

    Both legs come from :func:`mafrm.factors.macro.curve_yield_changes_bps` at
    the same tenor, so they are masked and differenced identically and the
    comparison is between two curves rather than between two constructions. The
    tenor is read from ``data.real_curve.maturities_years``, which SPEC.md 3.1
    puts exactly one entry in.

    First differences, not levels: correlating a daily return series against a
    yield LEVEL is a spurious-regression setup, which is why the ruling names
    the change.
    """
    settings = config or load()
    tenor = settings.model.factors.hybrid.row_83.tenor_years
    boundary = end if end is not None else settings.require_holdout_start()
    real = macro.curve_yield_changes_bps(
        tenors_years=[tenor], end=boundary, config=settings, kind="real"
    )
    nominal = macro.curve_yield_changes_bps(
        tenors_years=[tenor], end=boundary, config=settings, kind="nominal"
    )
    frame = pd.DataFrame(
        {
            "real": real[f"y{tenor:g}"],
            "nominal": nominal[f"y{tenor:g}"],
        }
    ).dropna(how="any")
    if frame.empty:  # pragma: no cover - both curves would have to be empty
        raise HybridError("comparand_yield_changes: no date has both curves published")
    return frame


@dataclass(frozen=True)
class Row83Result:
    """One residual PC read against experiments.md row 83, as registered.

    Every field is reported whatever the outcome. Row 83's registration asks for
    the full loading vector and gold's rank "regardless of outcome, so a
    near-miss is visible rather than collapsed into a pass/fail", and the
    :attr:`gold_rank` / :attr:`loadings` pair is that.
    """

    component: str
    components_in_model: int
    #: ``N``. The denominator of clause (a)'s null rate, taken from the panel.
    assets_in_panel: int
    #: Mean loading per asset over the sample, under the CONTINUITY convention.
    loadings: pd.Series
    #: Where the orientation asset sits when assets are ranked by mean absolute
    #: loading. ``1`` is the registered criterion.
    gold_rank: int
    gold_absolute_loading: float
    next_asset: str
    next_absolute_loading: float
    #: Correlations of the TEST-oriented PC with the two yield-change legs.
    correlation_real: float
    correlation_nominal: float
    observations: int
    orientation_agreement: float
    loading_holds: bool
    placebo_holds: bool
    sign_holds: bool

    @property
    def passes(self) -> bool:
        """All three clauses. Row 83's falsifier is that any of them fails."""
        return self.loading_holds and self.placebo_holds and self.sign_holds

    @property
    def null_probability(self) -> float:
        """The chance this component passes all three clauses under the null.

        ``(1/N) x (1/2) x (1/2)``: gold holding the largest absolute loading is
        ``1/N`` under exchangeable assets; the placebo is a coin flip because a
        PC unrelated to either curve is as likely to correlate more with one as
        with the other; and the sign clause is a coin flip once the orientation
        is fixed by a loading whose sign is itself arbitrary under the null.

        The three are treated as independent, which is an approximation and is
        stated as one. It is the number that has to be printed beside a pass --
        a criterion met is not a finding until its null rate is next to it.
        """
        return (1.0 / float(self.assets_in_panel)) * 0.5 * 0.5


def evaluate_row_83(
    pcs: ResidualPcPanel,
    comparands: pd.DataFrame,
    *,
    settings: HybridConfig,
) -> tuple[Row83Result, ...]:
    """Read every component of ``pcs`` against row 83's three registered clauses.

    Clause (a), the loading: the orientation asset must hold the LARGEST
    ABSOLUTE mean loading on the component. A rank, so it is scale-free and its
    null is ``1/N``.

    Clause (b), the comparand: ``|rho(PC, d.real)| > |rho(PC, d.nominal)|``. A
    placebo, not a significance bar. At ``n`` in the thousands a two-sided 5%
    test passes at ``|rho| > 0.03``, which has no teeth, and the 0.3 bar
    SPEC.md 6.5.2 withdrew is not reimported.

    Clause (c), the sign: with each date's PC oriented so that the orientation
    asset's loading is positive, ``rho(PC, d.real)`` must carry
    ``expected_correlation_sign``. Gold rises when real yields fall.
    """
    rule = settings.row_83
    if rule.material_loading_rule != "largest_absolute":  # pragma: no cover - parser gates it
        raise HybridError(f"unsupported loading rule {rule.material_loading_rule!r}")
    if rule.comparand_rule != "real_beats_nominal":  # pragma: no cover - parser gates it
        raise HybridError(f"unsupported comparand rule {rule.comparand_rule!r}")

    results: list[Row83Result] = []
    for name in pcs.factors.columns:
        loadings = pcs.loadings[name].mean()
        absolute = loadings.abs().sort_values(ascending=False)
        order = [str(item) for item in absolute.index]
        rank = order.index(pcs.orientation_asset) + 1
        runner_up = order[1] if rank == 1 else order[0]

        joined = pd.concat([pcs.test_factors[name], comparands], axis=1, join="inner").dropna()
        correlation = joined.corr().to_numpy(dtype=float)
        real = float(correlation[0, 1])
        nominal = float(correlation[0, 2])

        results.append(
            Row83Result(
                component=name,
                components_in_model=pcs.components,
                assets_in_panel=len(pcs.assets),
                loadings=loadings,
                gold_rank=rank,
                gold_absolute_loading=float(absolute.loc[pcs.orientation_asset]),
                next_asset=str(runner_up),
                next_absolute_loading=float(absolute.loc[runner_up]),
                correlation_real=real,
                correlation_nominal=nominal,
                observations=int(joined.shape[0]),
                orientation_agreement=pcs.orientation_agreement[name],
                loading_holds=rank == 1,
                placebo_holds=abs(real) > abs(nominal),
                sign_holds=float(np.sign(real)) == float(rule.expected_correlation_sign),
            )
        )
    return tuple(results)


def union_null_probability(results: Sequence[Row83Result]) -> tuple[float, float]:
    """``(union bound, independent approximation)`` for "at least one component passes".

    Both are printed because they say different things and the gap between them
    is small enough that quoting one alone would look like precision. The union
    bound ``sum p_k`` is exact as a bound and is what the claim is safe against;
    the independent form ``1 - prod(1 - p_k)`` is what it would be if the
    components were independent, which they are not -- eigenvectors of one
    matrix are orthogonal, so a second component cannot load on gold in exactly
    the way the first does.

    The point of reporting this at all: "at least one of two components passes"
    is a materially weaker claim than "the first one does", and it must not be
    presented as the same result.
    """
    probabilities = [result.null_probability for result in results]
    union = float(sum(probabilities))
    independent = float(1.0 - np.prod([1.0 - value for value in probabilities]))
    return union, independent


def loading_rank_null(assets: int, components: int) -> tuple[float, float]:
    """``(union bound, independent approximation)`` for clause (a) alone.

    Gold holding the largest absolute loading on a given component is ``1/N``
    under exchangeable assets. Across ``components`` components the union bound
    is ``k/N`` and the independent form is ``1 - (1 - 1/N)^k``. At ``N = 13``
    and ``k = 2`` these are ``2/13 = 0.1538`` and ``25/169 = 0.1479``.
    """
    single = 1.0 / float(assets)
    return float(components * single), float(1.0 - (1.0 - single) ** components)


def factor_panel(
    pcs: ResidualPcPanel,
    panel: macro.FactorPanel | None = None,
    *,
    config: Config | None = None,
) -> pd.DataFrame:
    """The hybrid factor set: six named macro factors plus ``K`` unnamed residual PCs.

    SPEC.md 4.3: *"append the top 1-2 residual PCs as unnamed factors"*. This is
    that append, and the frame it returns is what
    :func:`mafrm.risk.covariance.run_pipeline` consumes -- **the same shape, in
    the same units, from the same accessor** that ``mafrm.build`` hands it for
    Model A alone. SPEC.md 15.2's whole claim is that a second factor set costs
    nothing in ``risk/``, and the hybrid is the harder test of it than Model B
    was: Model B's factors are a homogeneous statistical basis, while this frame
    mixes named macro factors with unnamed residual components, which is exactly
    where an asset-class assumption would surface if ``risk/`` held one.

    **The macro columns arrive in the panel's own mixed units** -- four decimal
    returns and two in basis points of yield -- because that is what Model A's
    own pipeline run consumes and changing it here would make the hybrid's
    matrices incomparable with Model A's. The residual PCs are dimensionless
    combinations of standardised residuals, with a standard deviation near
    ``sqrt(lambda_k)``, so the frame spans three scales. That shows up in the
    condition number and nowhere else: experiments.md row 96 established that the
    1e7 conditioning of this project's factor matrices is **units, not
    near-singularity**, and SPEC.md 5.1 estimates volatility and correlation
    separately for exactly this reason.

    Inner join on dates, then complete cases. The residual PCs start about 251
    trading days after the macro panel does, and nothing is filled.
    """
    settings = config or load()
    factors = panel if panel is not None else macro.macro_factor_panel(config=settings)
    joined = factors.complete.join(pcs.factors, how="inner").dropna(how="any")
    if joined.empty:
        raise HybridError(
            "factor_panel: the macro panel and the residual PCs share no complete date"
        )
    return joined


def build(
    *, config: Config | None = None
) -> tuple[ResidualPanel, tuple[ResidualPcPanel, ...], pd.DataFrame]:
    """The whole hybrid: residuals, both PC counts, and the comparand legs.

    Both ``component_counts`` are built and neither is selected -- the third
    application of the precedent that carries ``a = 1.0`` alongside ``a = 1.4``
    and ``denoised`` alongside ``detoned``.
    """
    settings = config or load()
    scheme = settings.model.factors.hybrid
    panel = residual_panel(config=settings)
    pcs = tuple(
        residual_pcs(panel.residuals, components=count, settings=scheme)
        for count in scheme.component_counts
    )
    comparands = comparand_yield_changes(config=settings)
    return panel, pcs, comparands
