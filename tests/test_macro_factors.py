"""Model A's factor construction. SPEC.md 4.1 and 4.1.1.

Three properties carry this module and each gets a test that could actually
fail:

1. **The 1bp normalization means what it claims.** A synthetic curve built from a
   known level component and a known slope component must return those exact
   components, with loadings of exactly ``(1, 1, 1, 1)`` and exactly ``w/2``.
   Every number in ``test_normalization_recovers_a_hand_built_curve`` is
   derived on paper below, not read off a run.
2. **The sign convention is a property of the loadings, not of the arithmetic.**
   PCA sign is arbitrary and LAPACK's orientation is not stable, so the fix is
   tested directly: an eigenvector and its negation must produce the *same*
   loading vector, with the flip counted exactly once.
3. **Nothing looks ahead.** Truncating the input must leave every retained value
   bit-identical, for the PCA and the Gram-Schmidt alike -- and the expanding
   construction must actually differ from the full-sample one, or the test is
   passing for the wrong reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mafrm.config import ConfigError, OrthogonalizationConfig, RatePcaConfig, load
from mafrm.factors import macro
from mafrm.numerics import ewma_weights

# ---------------------------------------------------------------------------
# A hand-built curve with a known level and a known slope
# ---------------------------------------------------------------------------
#
# Yield changes are constructed as an exact two-component process:
#
#     dy_t = a_t * u + b_t * w,    u = (1, 1, 1, 1),  w = (-1, -1/2, 1/2, 1)
#
# with `a` and `b` INTERLEAVED sign patterns, each nonzero exactly where the
# other is zero:
#
#     a = (+1, -1,  0,  0, +1, -1,  0,  0, ...)
#     b = ( 0,  0, +1, -1,  0,  0, +1, -1, ...)
#
# That interleaving is what makes the fixture exact at EVERY window length
# rather than only at whole multiples of the pattern. `a_t * b_t = 0` on every
# single row, so the raw cross-moment is zero for every prefix; and `a` can have
# a nonzero prefix mean only when the prefix ends mid-a-pair (n = 1 mod 4) while
# `b` can only when it ends mid-b-pair (n = 3 mod 4), which cannot both happen.
# So the centred sample covariance is exactly
#
#     Var(a).u u'  +  Var(b).w w'
#
# after 8 rows and after 157, with no approximation to absorb an error into.
#
# u . w = -1 - 1/2 + 1/2 + 1 = 0, so u and w are orthogonal and are therefore
# exactly the eigenvectors, with eigenvalues Var(a)|u|^2 = 4.Var(a) and
# Var(b)|w|^2 = 2.5.Var(b).  Var(a) = Var(b) on complete blocks, so the level
# eigenvalue is the larger and PC1 is the level.
#
# Working the normalization through by hand:
#
#   LEVEL.  |u| = 2, so v1 = u/2 = (1/2, 1/2, 1/2, 1/2) and mean(v1) = 1/2.
#           Scaling to a mean loading of 1bp multiplies by 1/(1/2) = 2, giving
#           loadings exactly (1, 1, 1, 1) = u.  The factor is the raw score
#           divided by that scale: f1 = (v1 . dy) / 2 = (u . dy) / 4 = a,
#           because u . dy = a|u|^2 = 4a.  So the level factor IS `a`.
#
#   SLOPE.  |w|^2 = 1 + 1/4 + 1/4 + 1 = 5/2.  v2 = w/|w|, and the normalizing
#           statistic is v2[-1] - v2[0] = 2/|w|, so the scale is |w|/2 and the
#           loadings are exactly w/2 = (-1/2, -1/4, 1/4, 1/2), whose last minus
#           first is exactly 1.  The factor is
#           f2 = (v2 . dy)/(|w|/2) = 2(w . dy)/|w|^2 = 2b.
#
# Both statements are exact, not approximate, which is what makes them worth
# asserting to 1e-12 rather than to a tolerance chosen to pass.

_U = np.array([1.0, 1.0, 1.0, 1.0])
_W = np.array([-1.0, -0.5, 0.5, 1.0])
_TENORS = (2.0, 5.0, 10.0, 30.0)
_PERIODS = 40


def _hand_built_changes(periods: int = _PERIODS) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """The synthetic curve above, plus the exact level and slope components."""
    a = np.tile([1.0, -1.0, 0.0, 0.0], periods)
    b = np.tile([0.0, 0.0, 1.0, -1.0], periods)
    values = np.outer(a, _U) + np.outer(b, _W)
    index = pd.bdate_range("2000-01-03", periods=values.shape[0], name="date")
    frame = pd.DataFrame(values, index=index, columns=[f"y{t:g}" for t in _TENORS])
    return frame, a, b


def _pca_settings(min_window: int = 8) -> RatePcaConfig:
    """The shipped normalization targets on a short window, for a short fixture."""
    shipped = load().model.factors.macro.rates.pca
    return RatePcaConfig(
        estimator=shipped.estimator,
        expanding_min_window=min_window,
        level_mean_loading_bps=shipped.level_mean_loading_bps,
        slope_loading_difference_bps=shipped.slope_loading_difference_bps,
        min_normalizer=shipped.min_normalizer,
    )


def test_normalization_recovers_a_hand_built_curve() -> None:
    """Every number here is derived on paper in the module docstring above."""
    changes, a, b = _hand_built_changes()
    result = macro.rate_factors(changes, settings=_pca_settings(), tenors_years=_TENORS)

    defined = result.factors[macro.LEVEL].notna().to_numpy()
    assert defined.sum() == len(changes) - 7  # min_window = 8

    rows = int(defined.sum())
    np.testing.assert_allclose(
        result.loadings[macro.LEVEL].to_numpy()[defined], np.tile(_U, (rows, 1)), atol=1e-12
    )
    np.testing.assert_allclose(
        result.loadings[macro.SLOPE].to_numpy()[defined], np.tile(_W / 2.0, (rows, 1)), atol=1e-12
    )
    np.testing.assert_allclose(
        result.factors[macro.LEVEL].to_numpy()[defined], a[defined], atol=1e-12
    )
    np.testing.assert_allclose(
        result.factors[macro.SLOPE].to_numpy()[defined], 2.0 * b[defined], atol=1e-12
    )


def test_normalization_targets_hold_on_every_date() -> None:
    """Mean level loading is 1bp and the slope's long-minus-short is 1bp, always.

    Unlike the exact recovery above this holds on EVERY defined date, whatever
    the window length, because it is a property of the normalization rather than
    of the fixture's periodicity.
    """
    changes, _, _ = _hand_built_changes()
    settings = _pca_settings()
    result = macro.rate_factors(changes, settings=settings, tenors_years=_TENORS)

    level = result.loadings[macro.LEVEL].dropna()
    slope = result.loadings[macro.SLOPE].dropna()
    np.testing.assert_allclose(level.mean(axis=1), settings.level_mean_loading_bps, atol=1e-12)
    np.testing.assert_allclose(
        slope.iloc[:, -1] - slope.iloc[:, 0], settings.slope_loading_difference_bps, atol=1e-12
    )


def test_the_loadings_reconstruct_the_curve() -> None:
    """``dy = l_level.f_level + l_slope.f_slope`` exactly, for a rank-2 process.

    The identity the normalization is defined by. If the factor were scaled by
    the loading rather than by its inverse -- the easiest possible error, and the
    one experiments.md rows 64-67 were registered to catch -- this fails by the
    square of the scale.
    """
    changes, _, _ = _hand_built_changes()
    result = macro.rate_factors(changes, settings=_pca_settings(), tenors_years=_TENORS)
    defined = result.factors[macro.LEVEL].notna()

    rebuilt = result.loadings[macro.LEVEL][defined].mul(
        result.factors.loc[defined, macro.LEVEL], axis=0
    ) + result.loadings[macro.SLOPE][defined].mul(result.factors.loc[defined, macro.SLOPE], axis=0)
    np.testing.assert_allclose(rebuilt.to_numpy(), changes[defined].to_numpy(), atol=1e-12)


# ---------------------------------------------------------------------------
# The sign fix
# ---------------------------------------------------------------------------


def test_the_sign_fix_makes_an_eigenvector_and_its_negation_identical() -> None:
    """PCA sign is arbitrary; the loading vector must not be.

    This is the property that stops the level factor silently meaning "yields
    down" on some runs and "yields up" on others. Asserted on the normalizer
    itself so it cannot pass by luck of the fixture's orientation.
    """
    vector = np.array([0.5, 0.5, 0.5, 0.5])
    weights = np.full(4, 0.25)
    forward, flipped_forward = macro._normalize_component(
        vector, weights=weights, target=1.0, floor=1e-6, label="level"
    )
    backward, flipped_backward = macro._normalize_component(
        -vector, weights=weights, target=1.0, floor=1e-6, label="level"
    )
    np.testing.assert_allclose(forward, backward, atol=1e-15)
    assert flipped_forward is False
    assert flipped_backward is True


def test_a_negated_curve_negates_the_factor_and_leaves_the_loadings_alone() -> None:
    """Global sign of the data must not change what "level" means.

    ``cov(-X) == cov(X)``, so the loadings are identical and only the projection
    flips. A construction that fixed the sign from the data rather than from the
    loadings would fail this.
    """
    changes, _, _ = _hand_built_changes()
    settings = _pca_settings()
    forward = macro.rate_factors(changes, settings=settings, tenors_years=_TENORS)
    backward = macro.rate_factors(-changes, settings=settings, tenors_years=_TENORS)

    for name in (macro.LEVEL, macro.SLOPE):
        np.testing.assert_allclose(
            forward.loadings[name].dropna().to_numpy(),
            backward.loadings[name].dropna().to_numpy(),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            forward.factors[name].dropna().to_numpy(),
            -backward.factors[name].dropna().to_numpy(),
            atol=1e-12,
        )


def test_a_parallel_rise_is_a_positive_level_and_a_steepening_is_a_positive_slope() -> None:
    """SPEC.md 4.1's stated conventions, asserted on the factor's own sign."""
    changes, _, _ = _hand_built_changes()
    shock = changes.copy()
    shock.iloc[-2] = [1.0, 1.0, 1.0, 1.0]  # parallel +1bp
    shock.iloc[-1] = [-1.0, -0.5, 0.5, 1.0]  # steepening

    result = macro.rate_factors(shock, settings=_pca_settings(), tenors_years=_TENORS)
    assert result.factors[macro.LEVEL].iloc[-2] > 0.0
    assert result.factors[macro.SLOPE].iloc[-1] > 0.0


def test_a_degenerate_eigenvector_raises_rather_than_scaling_by_infinity() -> None:
    """The ``min_normalizer`` guard. A vanishing denominator is not a factor."""
    with pytest.raises(macro.MacroFactorError, match="below the"):
        macro._normalize_component(
            np.array([1.0, -1.0, 1.0, -1.0]),
            weights=np.full(4, 0.25),
            target=1.0,
            floor=1e-6,
            label="level",
        )


# ---------------------------------------------------------------------------
# No look-ahead
# ---------------------------------------------------------------------------


def test_the_pca_does_not_look_ahead() -> None:
    """Truncating the input leaves every retained factor value bit-identical.

    The definition of "estimated on data through t". A full-sample
    eigendecomposition fails this on every row.
    """
    changes, _, _ = _hand_built_changes()
    settings = _pca_settings()
    full = macro.rate_factors(changes, settings=settings, tenors_years=_TENORS)
    truncated = macro.rate_factors(changes.iloc[:60], settings=settings, tenors_years=_TENORS)

    for name in (macro.LEVEL, macro.SLOPE):
        pd.testing.assert_series_equal(
            full.factors[name].iloc[:60], truncated.factors[name], check_names=False
        )


def test_the_expanding_pca_differs_from_a_full_sample_one() -> None:
    """Guards the test above from passing vacuously.

    On the hand-built fixture the two agree exactly, because its covariance is
    stationary by construction -- which is why this uses a curve whose second
    half moves differently from its first.
    """
    changes, _, _ = _hand_built_changes()
    changes.iloc[len(changes) // 2 :] *= np.array([3.0, 1.0, 1.0, 0.2])
    settings = _pca_settings()

    expanding = macro.rate_factors(changes, settings=settings, tenors_years=_TENORS)
    tail = changes.index[-1]
    # A "full sample" PC is what you get if every date is allowed to see the
    # whole history: the last date's estimate, applied throughout.
    last_loading = expanding.loadings[macro.LEVEL].loc[tail]
    early_loading = expanding.loadings[macro.LEVEL].iloc[settings.expanding_min_window]
    assert not np.allclose(last_loading.to_numpy(), early_loading.to_numpy(), atol=1e-6)


def _orthogonalization_settings(min_window: int = 8) -> OrthogonalizationConfig:
    shipped = load().model.factors.macro.orthogonalization
    return OrthogonalizationConfig(
        order=("x", "y"),
        against={"y": ("x",)},
        expanding_min_window=min_window,
        expanding_window_start=shipped.expanding_window_start,
        fit_intercept=shipped.fit_intercept,
        subtract_intercept=shipped.subtract_intercept,
        targeted_pair_max_abs_correlation=shipped.targeted_pair_max_abs_correlation,
        exempt_pairs=(),
    )


def _linear_panel(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(load().seed)
    x = rng.normal(size=n)
    y = 0.25 + 2.0 * x + rng.normal(scale=0.1, size=n)
    return pd.DataFrame(
        {"x": x, "y": y}, index=pd.bdate_range("2010-01-04", periods=n, name="date")
    )


def test_the_orthogonalization_does_not_look_ahead() -> None:
    """Same truncation argument, on the Gram-Schmidt."""
    panel = _linear_panel()
    settings = _orthogonalization_settings()
    full = macro.orthogonalize(panel, settings=settings)
    truncated = macro.orthogonalize(panel.iloc[:600], settings=settings)
    pd.testing.assert_series_equal(full["y"].iloc[:600], truncated["y"])


def test_the_expanding_orthogonalization_differs_from_a_full_sample_one() -> None:
    """Guards the test above. If these agreed, the look-ahead test proves nothing."""
    panel = _linear_panel()
    expanding = macro.orthogonalize(panel, settings=_orthogonalization_settings())["y"].dropna()

    design = np.column_stack([np.ones(len(panel)), panel["x"].to_numpy()])
    beta, *_ = np.linalg.lstsq(design, panel["y"].to_numpy(), rcond=None)
    full_sample = panel["y"] - panel["x"] * beta[1]

    assert not np.allclose(expanding.to_numpy(), full_sample.loc[expanding.index].to_numpy())
    # The full-sample residual is orthogonal to x by construction; the honest
    # expanding one is only approximately so, and that gap is the price of not
    # looking ahead.
    assert abs(float(np.corrcoef(full_sample, panel["x"])[0, 1])) < 1e-12
    residual_correlation = abs(float(expanding.corr(panel["x"].loc[expanding.index])))
    assert 0.0 < residual_correlation < 0.05


def test_the_orthogonalization_keeps_the_factor_mean_and_removes_the_slope() -> None:
    """``subtract_intercept: false``. The mean return is a premium, not a nuisance.

    ``y = 0.25 + 2x + e``. Removing only the slope leaves a residual whose mean is
    the intercept, 0.25 -- not zero. A build that subtracted the intercept would
    silently delete every factor's risk premium.
    """
    panel = _linear_panel()
    residual = macro.orthogonalize(panel, settings=_orthogonalization_settings())["y"].dropna()
    assert residual.mean() == pytest.approx(0.25, abs=0.03)
    assert residual.std() == pytest.approx(0.1, abs=0.02)


def test_a_factor_with_no_regressors_is_passed_through_untouched() -> None:
    panel = _linear_panel()
    out = macro.orthogonalize(panel, settings=_orthogonalization_settings())
    pd.testing.assert_series_equal(out["x"], panel["x"])


def test_missing_regressor_dates_come_back_missing_rather_than_filled() -> None:
    """A hedge ratio cannot be estimated against a factor that does not exist yet."""
    panel = _linear_panel()
    panel.loc[panel.index[:50], "x"] = np.nan
    out = macro.orthogonalize(panel, settings=_orthogonalization_settings())
    assert out["y"].iloc[:50].isna().all()
    assert out["y"].iloc[50:57].isna().all()  # the 8-day minimum, restarted
    assert out["y"].iloc[-1] == pytest.approx(out["y"].dropna().iloc[-1])


# ---------------------------------------------------------------------------
# The duration identity, by hand
# ---------------------------------------------------------------------------


def test_a_bond_return_regressed_on_the_level_factor_returns_minus_its_duration() -> None:
    """The whole point of the 1bp normalization, checked with no data at all.

    A zero-coupon bond's return in basis points is ``-D`` times the change in
    its own yield. On the hand-built curve the level loading is exactly 1bp at
    every tenor, so regressing a 5-year zero's return on the level factor must
    return exactly ``-5``. The bond return is formed from the YIELD column, not
    from the factor, so this is not the identity restated -- a reciprocal scale
    or a 100x units error in the normalization moves the answer and nothing else
    in the fixture absorbs it.

    ``R^2`` is 0.8 rather than 1 and that is the point: the 5y yield also carries
    the slope component ``w_5 = -1/2``, which the univariate regression leaves in
    the residual. Var contributions are ``(5*1)^2 = 25`` against
    ``(5*1/2)^2 = 6.25``, so ``R^2 = 25/31.25 = 0.8`` exactly.
    """
    changes, _, _ = _hand_built_changes()
    result = macro.rate_factors(changes, settings=_pca_settings(), tenors_years=_TENORS)

    # A contiguous block of whole 4-row cycles, so `a` and `b` are exactly
    # mean-zero and exactly uncorrelated over the regression sample as well as
    # over every estimation window inside it.
    rows = slice(8, len(changes))
    assert (len(changes) - 8) % 4 == 0

    duration = 5.0
    bond_return_bps = -duration * changes["y5"].to_numpy()[rows]
    factor = result.factors[macro.LEVEL].to_numpy()[rows]

    beta, _, r_squared = macro._ols(bond_return_bps, factor[:, None])
    assert beta[1] == pytest.approx(-duration, abs=1e-9)
    assert r_squared == pytest.approx(0.8, abs=1e-9)


# ---------------------------------------------------------------------------
# Config: the promises the YAML comments make
# ---------------------------------------------------------------------------


def test_the_macro_beta_window_matches_the_equity_descriptor() -> None:
    """``config/model.yaml`` says a test asserts these cannot drift apart."""
    model = load().model
    assert model.factors.macro.beta.window_halflife == model.equity_descriptors.beta


def test_the_shipped_parameters_are_the_ones_spec_names() -> None:
    """SPEC.md 4.1 and 4.1.1, transcribed once and checked here."""
    macro_config = load().model.factors.macro
    assert macro_config.equity.source == "ken_french"
    assert macro_config.equity.column == "Mkt-RF"
    assert macro_config.rates.tenors_years == (2.0, 5.0, 10.0, 30.0)
    assert macro_config.rates.pca.expanding_min_window == 252
    assert macro_config.rates.pca.level_mean_loading_bps == 1.0
    assert macro_config.rates.pca.slope_loading_difference_bps == 1.0
    assert macro_config.orthogonalization.expanding_min_window == 252
    assert macro_config.orthogonalization.against == {
        "rates_slope": ("rates_level",),
        "credit": ("equity", "rates_level"),
        "commodity": ("dollar",),
    }
    assert macro_config.credit.excess_over == "cash"
    assert macro_config.commodity.excess_over == "cash"
    assert macro_config.dollar.sign == 1.0
    assert macro_config.beta.window == 252
    assert macro_config.beta.halflife == 63


def test_the_registered_predictions_are_the_ones_experiments_md_carries() -> None:
    """experiments.md rows 64-67, pre-registered 2026-08-30 and unrevisable.

    A test rather than a comment because the failure mode this guards against is
    a later session quietly widening a band to fit a measurement -- which is the
    one thing the experiment log exists to make impossible.
    """
    check = load().model.factors.macro.duration_check
    assert check.tolerance_fraction == 0.25
    registered = {
        instrument.id: (instrument.predicted_duration, instrument.band(check.tolerance_fraction))
        for instrument in check.instruments
    }
    assert registered == {
        "govt_10y": (-10.0, (-12.5, -7.5)),
        "govt_2y": (-2.0, (-2.5, -1.5)),
        "tlt": (-15.4, (-19.25, -11.55)),
        "hy_credit": (-3.5, (-4.0, -3.0)),
    }


def test_an_orthogonalization_order_that_leaves_a_regressor_undefined_is_rejected(
    tmp_path: object,
) -> None:
    """The parser's own guard, exercised rather than trusted.

    ``commodity`` is regressed on the later-ordered ``dollar``, which is safe
    only because dollar is never itself orthogonalized. Give dollar an ``against``
    entry without moving it, and which version to use becomes undefined.
    """
    from mafrm import config as config_module

    node = {
        "order": ["a", "b", "c"],
        "against": {"b": ["c"], "c": ["a"]},
        "expanding_min_window": 252,
        "expanding_window_start": "sample.start",
        "fit_intercept": True,
        "subtract_intercept": False,
        "targeted_pair_max_abs_correlation": 0.15,
        "exempt_pairs": [],
    }
    with pytest.raises(ConfigError, match="does not precede"):
        config_module._parse_orthogonalization({"orthogonalization": node}, "model.factors.macro")


def test_half_a_prediction_band_is_rejected() -> None:
    """``predicted_min`` without ``predicted_max`` is not a band."""
    from mafrm import config as config_module

    node = {
        "tolerance_fraction": 0.25,
        "instruments": [
            {
                "id": "x",
                "kind": "etf_excess",
                "ticker": "SPY",
                "predicted_duration": -1.0,
                "predicted_min": -2.0,
            }
        ],
    }
    with pytest.raises(ConfigError, match="half a band"):
        config_module._parse_duration_check({"duration_check": node}, "model.factors.macro")


# ---------------------------------------------------------------------------
# Running-sum arithmetic
# ---------------------------------------------------------------------------


def test_the_running_covariance_matches_numpy() -> None:
    """The O(T) shortcut must equal the O(T^2) definition it replaces."""
    rng = np.random.default_rng(load().seed)
    values = rng.normal(size=(300, 4))
    covariances, defined = macro._expanding_covariances(values, 20)
    for index in (19, 100, 299):
        assert defined[index]
        np.testing.assert_allclose(
            covariances[index], np.cov(values[: index + 1], rowvar=False), atol=1e-10
        )
    assert not defined[:19].any()


# ---------------------------------------------------------------------------
# The real panel. `dataset`-marked: reads data/raw, absent on a clean clone.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panel() -> macro.FactorPanel:
    try:
        return macro.macro_factor_panel()
    except Exception as exc:  # pragma: no cover - depends on what is cached
        pytest.skip(f"macro panel unavailable: {exc}")


@pytest.mark.dataset
def test_the_panel_stops_before_the_holdout(panel: macro.FactorPanel) -> None:
    """CLAUDE.md invariant 5, enforced rather than remembered."""
    assert panel.end < pd.Timestamp(load().require_holdout_start())
    assert panel.start >= pd.Timestamp(load().model.sample.start)


@pytest.mark.dataset
def test_the_orthogonalization_removes_the_credit_equity_correlation(
    panel: macro.FactorPanel,
) -> None:
    """SPEC.md 4.1's own claim: ~0.7 before, near zero after.

    The one pair the spec makes a numerical prediction about. It is asserted at
    the value the spec states, not at whatever came out.
    """
    before, after, _ = macro.correlation_matrices(panel)
    assert before.loc[macro.CREDIT, macro.EQUITY] == pytest.approx(0.7, abs=0.1)
    assert abs(after.loc[macro.CREDIT, macro.EQUITY]) < 0.15


@pytest.mark.dataset
def test_the_level_factor_is_the_dominant_curve_mode(panel: macro.FactorPanel) -> None:
    """ "Level" and "slope" have to still be the right names for PC1 and PC2."""
    shares = panel.pca.variance_share.dropna()
    assert shares[macro.LEVEL].iloc[-1] > 0.7
    assert shares[macro.SLOPE].iloc[-1] > 0.05
    assert (shares[macro.LEVEL] > shares[macro.SLOPE]).all()


@pytest.mark.dataset
def test_the_pcs_never_reverse_orientation(panel: macro.FactorPanel) -> None:
    """One day of data cannot rotate a stable eigenvector through 90 degrees.

    A nonzero count means PC1 and PC2 are close to degenerate and swapping, which
    would make the level and slope factors mean different things on different
    dates. The sign fix firing is fine and expected; a reversal after the fix is
    not.
    """
    for diagnostics in panel.pca.diagnostics.values():
        assert diagnostics.orientation_reversals == 0
        assert diagnostics.shape_violations == 0


@pytest.mark.dataset
def test_every_expanding_curve_covariance_is_psd(panel: macro.FactorPanel) -> None:
    """CLAUDE.md invariant 4, at the one place this module forms a covariance.

    The eigendecomposition is the only covariance stage in the macro factor set,
    and it is a 4x4 sample covariance rather than a Newey-West or shrinkage
    transform, so it is PSD by construction and no flooring is expected to fire.
    Asserted rather than assumed, because the running-sum accumulation could in
    principle drift negative on a near-singular window and the failure would show
    up downstream as a nonsensical variance share rather than as an error.
    """
    settings = load().model.factors.macro
    changes = macro.curve_yield_changes_bps(
        tenors_years=settings.rates.tenors_years, end=load().require_holdout_start()
    )
    covariances, defined = macro._expanding_covariances(
        changes.to_numpy(dtype=float), settings.rates.pca.expanding_min_window
    )
    eigenvalues = np.linalg.eigvalsh(covariances[defined])
    assert eigenvalues.min() > load().model.numerics.psd_eigenvalue_floor
    assert panel.pca.variance_share.dropna().to_numpy().min() > 0.0


# ---------------------------------------------------------------------------
# The gate, as rescoped by SPEC.md 4.1.2
# ---------------------------------------------------------------------------


def _orthogonalization_node(**overrides: object) -> dict[str, object]:
    node: dict[str, object] = {
        "order": ["a", "b"],
        "against": {"b": ["a"]},
        "expanding_min_window": 252,
        "expanding_window_start": "sample.start",
        "fit_intercept": True,
        "subtract_intercept": False,
        "targeted_pair_max_abs_correlation": 0.15,
        "exempt_pairs": [],
    }
    node.update(overrides)
    return node


def test_the_gate_covers_exactly_the_pairs_the_scheme_targets() -> None:
    """SPEC.md 4.1.2: a pair the scheme does not touch cannot be a gate on it.

    The all-pairs version of this bar was withdrawn as invalid -- it was
    unmeetable by any correct implementation, and meeting it would have driven
    the factor correlation matrix toward diagonal and left the eigenfactor
    adjustment nothing to operate on. This asserts the surviving gate's scope.
    """
    settings = load().model.factors.macro.orthogonalization
    assert set(settings.targeted_pairs) == {
        frozenset(("rates_slope", "rates_level")),
        frozenset(("credit", "equity")),
        frozenset(("credit", "rates_level")),
        frozenset(("commodity", "dollar")),
    }
    assert not hasattr(settings, "max_abs_correlation")
    assert settings.targeted_pair_max_abs_correlation == 0.15


def test_the_exemption_list_is_empty_because_the_one_exemption_was_DISCHARGED() -> None:
    """W2-P1 exempted `rates_slope`/`rates_level`; W2-P2 removed the need for it.

    Two assertions, and the second is the one that matters. Emptying the list is
    only legitimate if the pair now meets the criterion **on its own** -- an
    empty list plus a failing pair would be the silent removal of a gate, which
    is worse than the exemption it replaced. So this pins both.
    """
    settings = load().model.factors.macro.orthogonalization
    assert settings.exempt_pairs == ()
    assert not settings.is_exempt(frozenset(("rates_slope", "rates_level")))
    # And the reason it may be empty: the window-start ruling of SPEC.md 4.1.3.
    assert settings.burns_in_inside_the_sample


def test_exempting_a_pair_the_scheme_does_not_target_is_rejected() -> None:
    """Only targeted pairs are gated, so only targeted pairs can be exempted.

    Exempting an untargeted pair would be claiming the gate covers ground
    SPEC.md 4.1.2 explicitly took away from it.
    """
    from mafrm import config as config_module

    node = _orthogonalization_node(
        exempt_pairs=[{"target": "a", "regressor": "b", "reason": "because", "owed": "nothing"}]
    )
    with pytest.raises(ConfigError, match="nothing to exempt"):
        config_module._parse_orthogonalization({"orthogonalization": node}, "model.factors.macro")


def test_an_exemption_without_a_reason_is_rejected() -> None:
    """ "An unexplained exemption is a lowered threshold wearing a different name."""
    from mafrm import config as config_module

    node = _orthogonalization_node(
        exempt_pairs=[{"target": "b", "regressor": "a", "reason": "  ", "owed": "W2-P2"}]
    )
    with pytest.raises(ConfigError, match="must carry a reason"):
        config_module._parse_orthogonalization({"orthogonalization": node}, "model.factors.macro")


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------


def test_the_ewma_weights_halve_every_half_life() -> None:
    """The defining property, hand-checked. Weights are oldest-first.

    The helper moved to :mod:`mafrm.numerics` in W3-P1 so that the covariance
    pipeline and this module share one copy rather than two that can drift; the
    property being checked is unchanged. ``normalize=False`` gives the raw decay,
    which is what makes the three ages readable as 1, 1/2, 1/4.
    """
    weights = ewma_weights(11, 5.0, normalize=False)
    assert weights[-1] == pytest.approx(1.0)  # age 0
    assert weights[-6] == pytest.approx(0.5)  # age 5 = one half-life
    assert weights[-11] == pytest.approx(0.25)  # age 10 = two half-lives


def test_the_condition_number_of_a_known_correlation_matrix() -> None:
    """Hand-computed. Two columns correlated at exactly 0.6 give exactly 4.0.

    A 2x2 correlation matrix has eigenvalues ``1 + r`` and ``1 - r``, so its
    condition number is ``(1 + r) / (1 - r)``. At ``r = 0.6`` that is
    ``1.6 / 0.4 = 4`` exactly.

    ``x`` and ``z`` are the period-4 sign patterns used above: exactly mean-zero,
    exactly equal in variance and exactly uncorrelated over whole periods, so
    ``y = r.x + sqrt(1 - r^2).z`` has sample correlation exactly ``r`` with ``x``.
    The half-life is set enormous so the weights are flat and the EWMA collapses
    to the sample correlation, which is what makes the answer exact.
    """
    r = 0.6
    x = np.tile([1.0, -1.0, 1.0, -1.0], 25)
    z = np.tile([1.0, 1.0, -1.0, -1.0], 25)
    y = r * x + np.sqrt(1.0 - r**2) * z
    frame = pd.DataFrame(
        {"x": x, "y": y}, index=pd.bdate_range("2005-01-03", periods=x.size, name="date")
    )

    numbers = macro.condition_numbers(frame, halflife=10**9, min_observations=len(frame))
    assert len(numbers) == 1
    assert float(numbers.iloc[0]) == pytest.approx((1.0 + r) / (1.0 - r), abs=1e-9)


def test_uncorrelated_columns_are_perfectly_conditioned() -> None:
    """The floor of the statistic: an identity correlation matrix has κ = 1."""
    x = np.tile([1.0, -1.0, 1.0, -1.0], 25)
    z = np.tile([1.0, 1.0, -1.0, -1.0], 25)
    frame = pd.DataFrame(
        {"x": x, "z": z}, index=pd.bdate_range("2005-01-03", periods=x.size, name="date")
    )
    numbers = macro.condition_numbers(frame, halflife=10**9, min_observations=len(frame))
    assert float(numbers.iloc[0]) == pytest.approx(1.0, abs=1e-9)


def test_the_conditioning_diagnostic_reads_week_threes_half_life() -> None:
    """Not a key of its own. The coupling is the point, so it is asserted."""
    model = load().model
    assert (
        model.factors.macro.condition_number.halflife_from
        == "covariance.factor_correlation_halflife.short"
    )
    assert model.factors.macro.condition_number.min_observations == 252
    # Shorter than the effective sample size at this half-life, deliberately:
    # requiring 1,454 days would start the series in 2014 and miss every crisis.
    effective = model.numerics.effective_sample_size.reference[
        model.covariance.factor_correlation_halflife.short
    ]
    assert model.factors.macro.condition_number.min_observations < effective


@pytest.mark.dataset
def test_the_gate_holds_on_the_real_panel(panel: macro.FactorPanel) -> None:
    """All four targeted pairs meet the criterion, and none is exempt.

    Asserted as SPEC.md 4.1.2 and 4.1.3 record it, so that a regression in the
    orthogonalization shows up here rather than in a report nobody diffs.

    **This changed in W2-P2 and the change is the point.** W2-P1 could assert
    only three of four, with `rates_slope`/`rates_level` exempted by name; after
    the window-start ruling the pair meets the criterion unaided and the
    exemption was discharged. The assertion is now the stronger one, and a later
    session that reintroduced the old window start would fail here.
    """
    gates = {f"{gate.left}|{gate.right}": gate for gate in macro.gate_results(panel)}
    assert gates["credit|equity"].before == pytest.approx(0.680, abs=0.01)
    assert all(gate.meets_criterion for gate in gates.values())
    assert not any(gate.exempt for gate in gates.values())
    # The pair the exemption was granted for, pinned specifically: it must both
    # FALL and land inside the criterion, which is what W2-P1 could not say.
    slope = gates["rates_slope|rates_level"]
    assert abs(slope.after) < abs(slope.before)
    assert abs(slope.after) <= slope.limit


@pytest.mark.dataset
def test_the_orthogonalization_improves_conditioning_in_every_crisis(
    panel: macro.FactorPanel,
) -> None:
    """The scheme's real justification, asserted rather than only reported.

    CLAUDE.md's second-ranked failure mode is the covariance matrix going
    near-singular in exactly the crises that matter. If the orthogonalization
    ever stopped improving the condition number through those windows, the case
    for running it at all would be gone -- and the report would still print a
    pretty chart.
    """
    cfg = load()
    numbers = macro.conditioning(panel, config=cfg)
    assert numbers.orthogonal.max() < numbers.raw.max()
    assert numbers.orthogonal.median() < numbers.raw.median()

    for window in cfg.model.costs.spread_vs_volatility.crisis_windows:
        raw_window, orthogonal_window = numbers.in_window(window.start, window.end)
        if raw_window.empty or orthogonal_window.empty:  # pragma: no cover - coverage varies
            continue
        assert orthogonal_window.max() < raw_window.max(), window.label
