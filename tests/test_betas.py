"""Model A's asset exposures. SPEC.md 4.1, and the W2-P2 window-start ruling of 4.1.3.

Four properties carry this module and each gets a test that could actually fail:

1. **A weighted regression on a hand-built series returns the coefficients that
   were put into it**, to 1e-12, with the weights doing what the half-life says.
   Every number in ``test_a_hand_built_series_returns_its_own_coefficients`` is
   derived on paper, not read off a run.
2. **A level-factor coefficient is minus an effective duration.** A synthetic
   bond return built as ``-n * dy`` must return exactly ``-n``, which is the
   property the whole basis-point unit convention exists for.
3. **Nothing looks ahead.** Truncating the input leaves every retained exposure
   bit-identical, and the rolling estimator uses no observation after its date.
4. **Row 69 actually changed something.** The two orthogonalization window
   starts must produce different factors, or the ruling is a comment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mafrm.config import ConfigError, MacroBetaConfig, load
from mafrm.factors import betas, macro
from mafrm.numerics import ewma_weights, realised_effective_sample_size

# ---------------------------------------------------------------------------
# A hand-built regression with known coefficients
# ---------------------------------------------------------------------------
#
# One factor, one asset, no noise:
#
#     f_t = the factor,  y_t = a + b * f_t
#
# A weighted least-squares fit of y on f with an intercept must return exactly
# (a, b) for ANY set of positive weights, because the fit is exact -- the
# residual is identically zero, so every weighted residual sum is zero and the
# minimiser is unique as long as f is not constant. That makes the expected
# values hand-derivable without knowing the EWMA weights at all, which is
# precisely what makes it a test of the regression rather than of the weights.
#
# The weights are then tested separately, on their own, against the definition
# `w(age) = 2^(-age/halflife)`.

_ALPHA = 3.0
_BETA = -7.5


def _linear_frame(periods: int = 400) -> tuple[pd.Series, pd.DataFrame]:
    dates = pd.bdate_range("2010-01-01", periods=periods)
    rng = np.random.default_rng(load().seed)
    factor = pd.Series(rng.normal(scale=5.0, size=periods), index=dates, name="f")
    asset = (_ALPHA + _BETA * factor).rename("asset")
    return asset, factor.to_frame()


def _settings(window: int = 60, halflife: int = 20) -> MacroBetaConfig:
    return MacroBetaConfig(window=window, halflife=halflife, fit_intercept=True)


def test_a_hand_built_series_returns_its_own_coefficients() -> None:
    """y = 3 - 7.5f must return beta = -7.5 and alpha = 3 on every date."""
    asset, factors = _linear_frame()
    exposures, alpha, error, r_squared = betas.rolling_exposures(
        asset, factors, window=60, halflife=20, fit_intercept=True
    )
    defined = exposures["f"].dropna()
    assert len(defined) == 400 - 59
    np.testing.assert_allclose(defined.to_numpy(), _BETA, atol=1e-12)
    np.testing.assert_allclose(alpha.dropna().to_numpy(), _ALPHA, atol=1e-12)
    # An exact fit: zero residual, so R^2 is exactly 1 and the intercept has no
    # sampling error left to report.
    np.testing.assert_allclose(r_squared.dropna().to_numpy(), 1.0, atol=1e-12)
    np.testing.assert_allclose(error.dropna().to_numpy(), 0.0, atol=1e-12)


def test_the_weights_halve_every_half_life() -> None:
    """The EWMA weights are the definition, hand-checked at three ages."""
    weights = ewma_weights(253, 63.0, normalize=False)
    # Ordered oldest first, so the LAST entry is age 0.
    assert weights[-1] == pytest.approx(1.0)
    assert weights[-64] == pytest.approx(0.5)
    assert weights[-127] == pytest.approx(0.25)
    assert weights[-253] == pytest.approx(2.0**-4)


def test_the_effective_sample_size_of_flat_weights_is_the_count() -> None:
    """Kish's formula must reduce to n when nothing is discounted."""
    flat = np.full(120, 1.0 / 120.0)
    assert realised_effective_sample_size(flat) == pytest.approx(120.0)


def test_the_shipped_window_carries_the_effective_size_the_report_quotes() -> None:
    """252d at a 63d half-life is about 160 observations, not 252.

    Hand-derived: with r = 2^(-1/63), sum(r^k) for k = 0..251 is
    (1 - r^252)/(1 - r) = (1 - 2^-4)/(1 - r), and sum(r^2k) is
    (1 - 2^-8)/(1 - r^2). Kish's ratio of those is the number below.
    """
    settings = load().model.factors.macro.beta
    weights = ewma_weights(settings.window, float(settings.halflife))
    ratio = 2.0 ** (-1.0 / settings.halflife)
    total = (1.0 - 2.0**-4) / (1.0 - ratio)
    squares = (1.0 - 2.0**-8) / (1.0 - ratio * ratio)
    expected = total * total / squares
    assert realised_effective_sample_size(weights) == pytest.approx(expected)
    # Well short of the nominal window, which is the whole point.
    assert 150.0 < expected < 170.0


def test_a_bond_return_regressed_on_the_level_factor_returns_minus_its_duration() -> None:
    """The unit convention, tested end to end at n = 12.

    A zero-coupon bond returning ``-n * dy`` in basis points, regressed on a
    level factor in basis points, must return exactly ``-12``. This is the
    property that makes every rates exposure in the report readable, and it fails
    loudly if a 1e4 conversion is applied to one side and not the other.
    """
    maturity = 12.0
    dates = pd.bdate_range("2010-01-01", periods=300)
    rng = np.random.default_rng(load().seed)
    level = pd.Series(rng.normal(scale=4.0, size=300), index=dates, name=macro.LEVEL)
    # The bond return, in DECIMAL units: -n * dy(bp) / 1e4.
    bond = (-maturity * level / macro._BPS_PER_UNIT).rename("bond")

    units = {macro.LEVEL: "basis points of yield"}
    exposure = betas.univariate_exposure(
        bond, level.to_frame(), macro.LEVEL, units=units, settings=_settings()
    )
    np.testing.assert_allclose(exposure.to_numpy(), -maturity, atol=1e-10)


def test_regression_units_scales_returns_and_leaves_basis_points_alone() -> None:
    """The one sanctioned conversion, checked on a mixed frame."""
    frame = pd.DataFrame(
        {macro.EQUITY: [0.001, -0.002], macro.LEVEL: [3.0, -1.5]},
        index=pd.bdate_range("2010-01-01", periods=2),
    )
    units = {
        macro.EQUITY: "decimal daily excess return",
        macro.LEVEL: "basis points of yield",
    }
    scaled = betas.regression_units(frame, units)
    np.testing.assert_allclose(scaled[macro.EQUITY].to_numpy(), [10.0, -20.0])
    np.testing.assert_allclose(scaled[macro.LEVEL].to_numpy(), [3.0, -1.5])


def test_a_column_with_no_recorded_unit_is_rejected() -> None:
    """Guessing a unit is how a duration silently becomes 10,000 durations."""
    frame = pd.DataFrame({"mystery": [1.0]}, index=pd.bdate_range("2010-01-01", periods=1))
    with pytest.raises(betas.BetaError, match="no unit recorded"):
        betas.regression_units(frame, {macro.LEVEL: "basis points of yield"})


# ---------------------------------------------------------------------------
# Look-ahead
# ---------------------------------------------------------------------------


def test_the_rolling_regression_does_not_look_ahead() -> None:
    """Truncating the input must leave every retained exposure bit-identical."""
    rng = np.random.default_rng(load().seed)
    dates = pd.bdate_range("2010-01-01", periods=300)
    factors = pd.DataFrame(rng.normal(size=(300, 2)), index=dates, columns=["a", "b"])
    asset = pd.Series(rng.normal(size=300), index=dates, name="asset")

    full, alpha_full, _, r2_full = betas.rolling_exposures(
        asset, factors, window=60, halflife=20, fit_intercept=True
    )
    cut = 220
    short, alpha_short, _, r2_short = betas.rolling_exposures(
        asset.iloc[:cut], factors.iloc[:cut], window=60, halflife=20, fit_intercept=True
    )
    pd.testing.assert_frame_equal(full.iloc[:cut], short)
    pd.testing.assert_series_equal(alpha_full.iloc[:cut], alpha_short)
    pd.testing.assert_series_equal(r2_full.iloc[:cut], r2_short)


def test_an_exposure_uses_only_its_own_window() -> None:
    """Changing a value one day AFTER a date cannot move that date's exposure.

    The sharper version of the truncation test: it also catches an off-by-one
    that reads ``position + 1`` into the window.
    """
    rng = np.random.default_rng(load().seed)
    dates = pd.bdate_range("2010-01-01", periods=200)
    factors = pd.DataFrame(rng.normal(size=(200, 1)), index=dates, columns=["a"])
    asset = pd.Series(rng.normal(size=200), index=dates, name="asset")

    base, _, _, _ = betas.rolling_exposures(
        asset, factors, window=50, halflife=15, fit_intercept=True
    )
    tampered = asset.copy()
    tampered.iloc[120] = 1_000.0
    moved, _, _, _ = betas.rolling_exposures(
        tampered, factors, window=50, halflife=15, fit_intercept=True
    )
    pd.testing.assert_series_equal(base["a"].iloc[:120], moved["a"].iloc[:120])
    assert base["a"].iloc[120] != moved["a"].iloc[120]


def test_a_window_that_cannot_identify_the_coefficients_is_rejected() -> None:
    """Three terms need more than three observations, and saying so beats a pinv."""
    asset, factors = _linear_frame(periods=50)
    with pytest.raises(betas.BetaError, match="cannot identify"):
        betas.rolling_exposures(asset, factors, window=2, halflife=1, fit_intercept=True)


# ---------------------------------------------------------------------------
# Config: the promises the YAML comments make
# ---------------------------------------------------------------------------


def test_the_beta_regression_fits_an_intercept_because_spec_4_1_writes_one() -> None:
    """SPEC.md 4.1's equation carries alpha_i, so this is specified, not chosen."""
    assert load().model.factors.macro.beta.fit_intercept is True


def test_the_macro_beta_window_still_matches_the_equity_descriptor() -> None:
    """The W2-P1 drift guard, preserved after `beta` grew a third key."""
    model = load().model
    assert model.factors.macro.beta.window_halflife == model.equity_descriptors.beta


def test_the_orthogonalization_burns_in_inside_the_sample() -> None:
    """SPEC.md 4.1.3 / experiments.md row 69, the W2-P2 ruling."""
    scheme = load().model.factors.macro.orthogonalization
    assert scheme.expanding_window_start == "sample.start"
    assert scheme.burns_in_inside_the_sample is True


def test_an_unrecognised_window_start_is_rejected() -> None:
    """It is a pointer to a closed set of two, not a free-text field."""
    from mafrm import config as config_module

    node = {
        "order": ["a", "b"],
        "against": {"b": ["a"]},
        "expanding_min_window": 252,
        "expanding_window_start": "whenever",
        "fit_intercept": True,
        "subtract_intercept": False,
        "targeted_pair_max_abs_correlation": 0.15,
        "exempt_pairs": [],
    }
    with pytest.raises(ConfigError, match="expanding_window_start"):
        config_module._parse_orthogonalization({"orthogonalization": node}, "model.factors.macro")


def test_the_two_window_starts_actually_produce_different_factors() -> None:
    """Row 69 must be a change, not a comment.

    Slicing before the Gram-Schmidt against slicing after it has to move the
    residuals, or the ruling did nothing and the correlation improvement it is
    credited with came from somewhere else.
    """
    rng = np.random.default_rng(load().seed)
    dates = pd.bdate_range("2000-01-03", periods=1200)
    scheme = load().model.factors.macro.orthogonalization
    # Two regimes with OPPOSITE hedge signs, which is the situation row 69 is
    # about: an early era where b loads +2 on a, and a later one where it loads -2.
    a = pd.Series(rng.normal(size=1200), index=dates, name="rates_level")
    sign = np.where(np.arange(1200) < 600, 2.0, -2.0)
    b = pd.Series(sign * a.to_numpy() + rng.normal(scale=0.1, size=1200), index=dates)
    frame = pd.DataFrame({"rates_level": a, "rates_slope": b.rename("rates_slope")})
    frame["equity"] = rng.normal(size=1200)
    frame["credit"] = rng.normal(size=1200)
    frame["commodity"] = rng.normal(size=1200)
    frame["dollar"] = rng.normal(size=1200)

    cut = dates[600]
    from_history = macro.orthogonalize(frame, settings=scheme).loc[cut:]
    from_sample = macro.orthogonalize(frame.loc[cut:], settings=scheme)
    common = from_history.index.intersection(from_sample.index)
    left = from_history.loc[common, "rates_slope"].dropna()
    right = from_sample.loc[common, "rates_slope"].dropna()
    shared = left.index.intersection(right.index)
    assert len(shared) > 100
    assert not np.allclose(left.loc[shared].to_numpy(), right.loc[shared].to_numpy())


# ---------------------------------------------------------------------------
# The real panel
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def factor_panel() -> macro.FactorPanel:
    return macro.macro_factor_panel()


@pytest.fixture(scope="module")
def panel(factor_panel: macro.FactorPanel) -> betas.ExposurePanel:
    return betas.exposure_panel(factor_panel)


@pytest.mark.dataset
def test_the_exposure_panel_stops_before_the_holdout(panel: betas.ExposurePanel) -> None:
    """CLAUDE.md invariant 5, enforced rather than trusted."""
    boundary = pd.Timestamp(load().require_holdout_start())
    assert panel.end < boundary
    assert panel.exposures.index.get_level_values("date").max() < boundary
    assert panel.alpha.index.max() < boundary


@pytest.mark.dataset
def test_every_investable_universe_member_has_exposures(panel: betas.ExposurePanel) -> None:
    """An asset that silently dropped out would never be missed downstream."""
    universe = load().universe
    expected = {asset.id for asset in universe.assets if asset.construction != "fred_index_level"}
    assert set(panel.assets) == expected
    assert len(panel.assets) == len(universe.assets) - 1


@pytest.mark.dataset
def test_the_ragged_edge_is_reported_and_not_filled(panel: betas.ExposurePanel) -> None:
    """The burn-in is real: exposures must be missing before it, not carried back."""
    first = panel.asset_first_valid
    assert set(first) == set(panel.assets)
    for asset in panel.assets:
        series = panel.r_squared[asset]
        before = series.loc[series.index < first[asset]]
        assert before.isna().all(), f"{asset} has an exposure before its own burn-in"
    # And the binding constraint is the factor set, not any asset's history.
    assert min(first.values()) >= panel.factor_first_valid[macro.CREDIT]


@pytest.mark.dataset
def test_the_synthetic_zeros_recover_their_own_maturities(
    factor_panel: macro.FactorPanel, panel: betas.ExposurePanel
) -> None:
    """The session's primary validation, as an assertion rather than a report row.

    A zero-coupon bond's duration is its maturity by definition, so this compares
    the panel against an identity read off `config/universe.yaml` -- not against
    a tolerance anyone chose after seeing the answer.
    """
    estimates = {item.id: item for item in betas.duration_recovery(factor_panel, panel)}
    for asset in load().universe.assets:
        if asset.construction != "synthetic_gsw_zero_curve":
            continue
        estimate = estimates[asset.id]
        assert estimate.passed, estimate.render()
        # Sign, not just magnitude: a positive exposure would mean a bond that
        # rises when yields rise.
        assert estimate.multivariate < 0.0
        assert estimate.rolling_mean < 0.0


@pytest.mark.dataset
def test_the_duration_bands_are_the_registered_ones(
    factor_panel: macro.FactorPanel, panel: betas.ExposurePanel
) -> None:
    """experiments.md rows 64-67 and 73: a band moved to fit a measurement is not a band."""
    estimates = {item.id: item for item in betas.duration_recovery(factor_panel, panel)}
    assert estimates["govt_10y"].band == (-12.5, -7.5)
    assert estimates["govt_2y"].band == (-2.5, -1.5)
    assert estimates["tlt"].band == (-19.25, -11.55)
    assert estimates["hy_credit"].band == (-4.0, -3.0)
    # The two identity rows: +/-25% of the maturity itself.
    assert estimates["govt_5y"].band == (-6.25, -3.75)
    assert estimates["govt_30y"].band == (-37.5, -22.5)
    assert estimates["hy_credit"].basis == "registered"
    assert estimates["govt_5y"].basis == "identity"


@pytest.mark.dataset
def test_hy_credit_stays_refuted(
    factor_panel: macro.FactorPanel, panel: betas.ExposurePanel
) -> None:
    """W2-P1 refuted row 67 and W2-P2 does not rescue it.

    The test exists so that a later session cannot quietly widen the band; it
    asserts the refutation, not the measurement.
    """
    estimates = {item.id: item for item in betas.duration_recovery(factor_panel, panel)}
    assert not estimates["hy_credit"].passed


@pytest.mark.dataset
def test_the_assets_that_build_a_factor_are_marked_as_identities(
    factor_panel: macro.FactorPanel, panel: betas.ExposurePanel
) -> None:
    """`credit` is HYG and `commodity` is DBC, so those two R-squareds are not fits."""
    settings = load()
    records = {
        item.id: item
        for item in betas.diagnostics(
            panel,
            betas.asset_excess_returns(config=settings),
            factor_panel.orthogonal,
            units=factor_panel.units,
            config=settings,
        )
    }
    macro_config = settings.model.factors.macro
    for name in (macro_config.credit.asset_id, macro_config.commodity.asset_id):
        assert records[name].spans_a_factor
        assert records[name].r_squared_median > 0.99
    assert not records["us_small_equity"].spans_a_factor


@pytest.mark.dataset
def test_the_genuine_fits_reach_the_r_squared_spec_4_1_expects(
    factor_panel: macro.FactorPanel, panel: betas.ExposurePanel
) -> None:
    """SPEC.md 4.1: 0.7-0.95 for asset-class-level series, flag below 0.5.

    Asserted as a floor on the median rather than as a band, because the upper
    end is not a failure -- three of the sleeve are nearly spanned by the factor
    set and that is a property of the sleeve, not a defect.
    """
    settings = load()
    records = betas.diagnostics(
        panel,
        betas.asset_excess_returns(config=settings),
        factor_panel.orthogonal,
        units=factor_panel.units,
        config=settings,
    )
    genuine = [item for item in records if not item.spans_a_factor]
    flagged = [item.id for item in genuine if item.r_squared_flagged]
    # Exactly one asset is expected to flag, and the universe predicted which:
    # gold's loadings are real-rate and dollar, and there is no real-rate factor.
    assert flagged == ["gold"]
    unflagged = [item for item in genuine if not item.r_squared_flagged]
    assert all(item.r_squared_median >= 0.5 for item in unflagged)


# ---------------------------------------------------------------------------
# Multiplicity, and the handle W4-P2 will filter on
# ---------------------------------------------------------------------------


def test_the_two_sided_normal_tail_is_the_exact_one() -> None:
    """Hand-computed: P(|Z| > 2) = erfc(2/sqrt(2)) = 0.0455002638963584.

    Asserted against the literal value rather than against `scipy`, because the
    point of the number in the report is that the shipped threshold of 2.0 is
    NOT a 5% test -- it is a rounding of 1.96 -- and a reader is entitled to see
    the difference.
    """
    from mafrm.factors import exposure_report

    assert exposure_report.two_sided_normal_tail(2.0) == pytest.approx(
        0.0455002638963584, abs=1e-15
    )
    # And the 5% critical value it was rounded from returns 5%.
    assert exposure_report.two_sided_normal_tail(1.959963984540054) == pytest.approx(
        0.05, abs=1e-12
    )


def test_the_null_expectation_of_the_alpha_flag_family() -> None:
    """13 intercepts at |t| > 2 is 0.59 expected by chance, not 3.9.

    The 78-test figure belongs to the whole exposure panel (13 assets x 6
    factors); the alpha flag runs on the intercepts only. The two are asserted
    separately here so that a later edit cannot quietly swap one for the other
    in the report, which would turn "4 observed against 0.59 expected" into
    "4 against 3.55" -- the difference between a finding and a coincidence.
    """
    from mafrm.factors import exposure_report

    settings = load()
    per_test = exposure_report.two_sided_normal_tail(
        settings.model.factors.macro.alpha_flag.abs_t_statistic
    )
    investable = [
        asset for asset in settings.universe.assets if asset.construction != "fred_index_level"
    ]
    assert len(investable) == 13
    assert per_test * len(investable) == pytest.approx(0.59, abs=0.005)
    assert per_test * len(investable) * 6 == pytest.approx(3.55, abs=0.005)


@pytest.mark.dataset
def test_w4_can_filter_the_identity_assets_off_the_panel(
    factor_panel: macro.FactorPanel, panel: betas.ExposurePanel
) -> None:
    """The W4-P2 forward constraint needs a handle, and this is it.

    `hy_credit` and `commodity` fit at R^2 = 1.000 because the proxy IS the
    factor, so a bias statistic computed on them is pinned at 1 and would pull
    a 13-asset aggregate toward "well calibrated". `spans_a_factor` is derived
    from the config rather than hard-coded, so W4 can filter on it; this asserts
    the filter selects exactly two assets and leaves eleven genuine fits.
    """
    settings = load()
    records = betas.diagnostics(
        panel,
        betas.asset_excess_returns(config=settings),
        factor_panel.orthogonal,
        units=factor_panel.units,
        config=settings,
    )
    excluded = [item.id for item in records if item.spans_a_factor]
    assert sorted(excluded) == ["commodity", "hy_credit"]
    assert len([item for item in records if not item.spans_a_factor]) == 11
