"""The hybrid. SPEC.md 4.3 and the W3-P6 rulings, plus experiments.md row 83.

The hand-computed values are in
:func:`test_the_residual_pc_of_a_two_asset_panel_is_hand_computable` and
:func:`test_row_83_clauses_on_a_hand_built_panel`. Both work on panels small
enough to solve on paper, which is what makes them tests of the code rather than
restatements of it.

The tests that matter most are the ones about what the code must NOT do:

* :func:`test_nothing_after_date_t_reaches_date_t` -- the look-ahead guard, by
  perturbation rather than by inspection;
* :func:`test_the_government_control_is_the_duration_leg_and_not_the_excess_return`
  -- W2-P3's mechanical control, which was fixed before any residual PC existed
  and is the reason row 83 is readable at all;
* :func:`test_a_covariance_pca_would_pick_the_most_volatile_asset` -- the control
  behind SPEC.md 4.3.3's correlation ruling. It asserts that the rejected
  alternative really does fail, so the ruling is a measurement and not a
  preference;
* :func:`test_the_row_83_thresholds_cannot_be_widened` -- the config parser
  refuses a magnitude bar, so no later session can turn the rank criterion into a
  number that fits a result.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mafrm.config import ConfigError, _curve_tenors, _parse_hybrid, load
from mafrm.factors import betas, hybrid, hybrid_report, macro

SETTINGS = load()
SCHEME = SETTINGS.model.factors.hybrid

#: The frozen universe's investable count. Row 83's null rate is 1/N.
_UNIVERSE_ASSETS = 13


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    """Seeded explicitly. CLAUDE.md: never a global ``np.random.seed``."""
    return np.random.default_rng(SETTINGS.model.seed)


@pytest.fixture(scope="module")
def scheme() -> hybrid.HybridConfig:
    """A hybrid config with a burn-in small enough for a synthetic panel."""
    return dataclasses.replace(SCHEME, expanding_min_window=20)


@pytest.fixture(scope="module")
def synthetic(rng: np.random.Generator) -> pd.DataFrame:
    """A residual panel with one planted common direction. No cache, no network.

    Four assets load on a shared driver and two do not, so the leading component
    has a known shape and the test can assert something about it rather than
    only that it exists.
    """
    periods = 400
    index = pd.bdate_range("2015-01-01", periods=periods, name="date")
    driver = rng.standard_normal(periods)
    columns = {
        "a_loud": 40.0 * driver + 8.0 * rng.standard_normal(periods),
        "b_loud": 35.0 * driver + 8.0 * rng.standard_normal(periods),
        "c_quiet": 2.0 * driver + 0.4 * rng.standard_normal(periods),
        "d_quiet": 1.8 * driver + 0.4 * rng.standard_normal(periods),
        "gold": 30.0 * rng.standard_normal(periods),
        "e_alone": 12.0 * rng.standard_normal(periods),
    }
    return pd.DataFrame(columns, index=index)


# ---------------------------------------------------------------------------
# Hand-computed
# ---------------------------------------------------------------------------


def test_the_residual_pc_of_a_two_asset_panel_is_hand_computable(
    scheme: hybrid.HybridConfig,
) -> None:
    """A 2x2 correlation has eigenvectors ``(1, 1)/sqrt(2)`` and ``(1, -1)/sqrt(2)``.

    Whatever the correlation is, the eigenvectors of a 2x2 correlation matrix are
    fixed by symmetry alone, and the eigenvalues are ``1 +- rho``. So a two-asset
    panel gives an exact expected loading with no linear algebra library in the
    expectation: ``1/sqrt(2) = 0.70710678``.

    The factor value then has a closed form too. With ``z`` the standardised
    residuals, ``f_1t = (z_1t + z_2t)/sqrt(2)``.
    """
    periods = 120
    index = pd.bdate_range("2020-01-01", periods=periods, name="date")
    generator = np.random.default_rng(SETTINGS.model.seed)
    first = generator.standard_normal(periods)
    frame = pd.DataFrame(
        {"gold": first, "other": 0.5 * first + generator.standard_normal(periods)}, index=index
    )

    panel = hybrid.residual_pcs(frame, components=1, settings=scheme)
    loadings = panel.loadings["residual_pc1"]

    root_half = 1.0 / math.sqrt(2.0)
    assert np.allclose(loadings.abs().to_numpy(dtype=float), root_half, atol=1e-12)

    # The eigenvalues are 1 +- rho, so they sum to the trace, which is N = 2.
    assert np.allclose(panel.eigenvalues.sum(axis=1).to_numpy(dtype=float), 2.0, atol=1e-12)

    # And the factor is the stated combination of standardised residuals.
    signs = np.sign(loadings.to_numpy(dtype=float))
    expected = (panel.standardised.to_numpy(dtype=float) * signs).sum(axis=1) * root_half
    assert np.allclose(panel.factors["residual_pc1"].to_numpy(dtype=float), expected, atol=1e-12)


def test_the_eigenvalues_sum_to_n_on_every_date(
    synthetic: pd.DataFrame, scheme: hybrid.HybridConfig
) -> None:
    """A correlation matrix has unit diagonal, so its trace is exactly ``N``."""
    panel = hybrid.residual_pcs(synthetic, components=2, settings=scheme)
    total = panel.eigenvalues.sum(axis=1).to_numpy(dtype=float)
    assert np.allclose(total, float(synthetic.shape[1]), atol=1e-10)


def test_the_variance_share_is_a_share(
    synthetic: pd.DataFrame, scheme: hybrid.HybridConfig
) -> None:
    """Monthly shares are non-negative and the components cannot together exceed 1."""
    panel = hybrid.residual_pcs(synthetic, components=2, settings=scheme)
    share = hybrid.monthly_variance_share(panel)
    columns = ["residual_pc1", "residual_pc2"]
    values = share[columns].to_numpy(dtype=float)
    assert (values >= 0.0).all()
    assert (values.sum(axis=1) <= 1.0 + 1e-12).all()
    assert (share["observations"] > 0).all()


def test_the_monthly_buckets_are_non_overlapping(
    synthetic: pd.DataFrame, scheme: hybrid.HybridConfig
) -> None:
    """CLAUDE.md failure mode 9: the observation counts must partition the sample.

    A rolling window would double-count; calendar months cannot. If the counts
    ever sum to more than the number of dates, the buckets overlap and every
    reading in the report would owe an overlap caveat it does not carry.
    """
    panel = hybrid.residual_pcs(synthetic, components=1, settings=scheme)
    share = hybrid.monthly_variance_share(panel)
    assert int(share["observations"].sum()) == len(panel.factors)


# ---------------------------------------------------------------------------
# What the code must not do
# ---------------------------------------------------------------------------


def test_nothing_after_date_t_reaches_date_t(
    synthetic: pd.DataFrame, scheme: hybrid.HybridConfig
) -> None:
    """The expanding window is expanding. By perturbation, not by inspection.

    Rewriting the last quarter of the panel must leave every earlier loading,
    factor value and eigenvalue bit-identical. A full-sample eigendecomposition
    -- the thing SPEC.md 4.1.3 and 4.2.1 ruled out -- would fail this on the
    first date.
    """
    cut = int(len(synthetic) * 0.75)
    perturbed = synthetic.copy()
    perturbed.iloc[cut:] = perturbed.iloc[cut:] * 7.0 + 3.0

    base = hybrid.residual_pcs(synthetic, components=2, settings=scheme)
    after = hybrid.residual_pcs(perturbed, components=2, settings=scheme)

    kept = base.factors.index[base.factors.index < synthetic.index[cut]]
    assert len(kept) > 10
    for name in base.factors.columns:
        assert np.array_equal(
            base.factors.loc[kept, name].to_numpy(dtype=float),
            after.factors.loc[kept, name].to_numpy(dtype=float),
        )
        assert np.array_equal(
            base.loadings[name].loc[kept].to_numpy(dtype=float),
            after.loadings[name].loc[kept].to_numpy(dtype=float),
        )
    assert np.array_equal(
        base.eigenvalues.loc[kept].to_numpy(dtype=float),
        after.eigenvalues.loc[kept].to_numpy(dtype=float),
    )


def test_a_covariance_pca_would_pick_the_most_volatile_asset(
    synthetic: pd.DataFrame, scheme: hybrid.HybridConfig
) -> None:
    """SPEC.md 4.3.3's ruling, asserted as a MEASUREMENT of the rejected option.

    The synthetic panel plants one common driver in four assets: two loud ones at
    volatility ~40 and two quiet ones at ~2. The common structure is identical in
    all four -- only the scale differs.

    On the CORRELATION, the leading component must see all four. On the
    COVARIANCE, it collapses onto the loud pair and the quiet pair effectively
    vanishes, which is experiments.md row 96's units artefact. Asserting both
    directions is what makes the ruling a measurement rather than a preference.
    """
    panel = hybrid.residual_pcs(synthetic, components=1, settings=scheme)
    correlation_loadings = panel.loadings["residual_pc1"].iloc[-1].abs()

    values = synthetic.to_numpy(dtype=float)
    covariance = np.cov(values, rowvar=False)
    _, vectors = np.linalg.eigh(covariance)
    covariance_loadings = pd.Series(np.abs(vectors[:, -1]), index=synthetic.columns)

    quiet = ["c_quiet", "d_quiet"]
    loud = ["a_loud", "b_loud"]

    # Correlation space: the quiet pair carries real weight in the same direction.
    assert correlation_loadings[quiet].min() > 0.3
    # Covariance space: it does not. The gap is what row 96 is about.
    assert covariance_loadings[quiet].max() < 0.1
    assert covariance_loadings[loud].min() > 0.5


def test_the_row_83_thresholds_cannot_be_widened(tmp_path: Path) -> None:
    """A magnitude bar is REFUSED by the parser, not merely undocumented.

    Row 83 was registered on 2026-08-30 and its criteria were ruled on
    2026-08-31 before the residual panel was computed. The whole value of that
    sequence is destroyed if a later session can swap the rank rule for a number
    chosen once the loadings are visible, so the parser rejects the substitution
    rather than accepting it silently.
    """
    raw = yaml.safe_load(Path("config/model.yaml").read_text(encoding="utf-8"))
    node = copy.deepcopy(raw["factors"])
    node["hybrid"]["row_83"]["material_loading_rule"] = "absolute_above_0_3"
    with pytest.raises(ConfigError, match="magnitude bar"):
        _parse_hybrid(
            node,
            "model.factors",
            rates_pca=SETTINGS.model.factors.macro.rates.pca,
            curves=_curve_tenors(raw),
        )


def test_the_component_count_range_cannot_be_narrowed(tmp_path: Path) -> None:
    """Both 1 and 2 are always built. Dropping one is choosing, and owes a row."""
    raw = yaml.safe_load(Path("config/model.yaml").read_text(encoding="utf-8"))
    node = copy.deepcopy(raw["factors"])
    node["hybrid"]["component_counts"] = [1]
    with pytest.raises(ConfigError, match="model-config"):
        _parse_hybrid(
            node,
            "model.factors",
            rates_pca=SETTINGS.model.factors.macro.rates.pca,
            curves=_curve_tenors(raw),
        )


def test_the_burn_in_has_no_key_of_its_own() -> None:
    """One number in the file says how long an expanding-window PCA must run."""
    raw = yaml.safe_load(Path("config/model.yaml").read_text(encoding="utf-8"))
    assert "expanding_min_window" not in raw["factors"]["hybrid"]
    assert (
        SCHEME.expanding_min_window == SETTINGS.model.factors.macro.rates.pca.expanding_min_window
    )

    node = copy.deepcopy(raw["factors"])
    node["hybrid"]["expanding_min_window"] = 100
    with pytest.raises(ConfigError, match="drift apart"):
        _parse_hybrid(
            node,
            "model.factors",
            rates_pca=SETTINGS.model.factors.macro.rates.pca,
            curves=_curve_tenors(raw),
        )


def test_the_placebo_tenor_is_read_from_the_universe_not_from_a_key() -> None:
    """Row 83's comparand tenor has no key. It is the real curve's one maturity."""
    raw = yaml.safe_load(Path("config/model.yaml").read_text(encoding="utf-8"))
    assert "tenor_years" not in raw["factors"]["hybrid"]["row_83"]
    assert SCHEME.row_83.tenor_years == SETTINGS.model.data.real_curve.maturities_years[0]
    assert SCHEME.row_83.tenor_years in SETTINGS.model.data.nominal_curve.maturities_years


# ---------------------------------------------------------------------------
# Row 83's clauses, on a panel built so the answer is known
# ---------------------------------------------------------------------------


def test_row_83_clauses_on_a_hand_built_panel(scheme: hybrid.HybridConfig) -> None:
    """All three clauses, on a panel constructed so each has a known answer.

    Two assets. ``gold`` is built to be the larger loading and to move against a
    synthetic real-yield change; the comparand's nominal leg is pure noise. So
    clause (a) must hold, clause (b) must hold, and clause (c) must hold with the
    negative sign the ruling registered -- and the PASS is the point, because
    every other test here is on a case that fails.
    """
    periods = 300
    index = pd.bdate_range("2019-01-02", periods=periods, name="date")
    generator = np.random.default_rng(SETTINGS.model.seed + 1)
    real = generator.standard_normal(periods) * 5.0
    nominal = generator.standard_normal(periods) * 5.0

    # gold moves against the real yield with a large loading; the partner asset
    # carries the same driver at a quarter of the weight plus its own noise.
    frame = pd.DataFrame(
        {
            "gold": -4.0 * real + 0.20 * generator.standard_normal(periods),
            "partner": -1.0 * real + 3.0 * generator.standard_normal(periods),
        },
        index=index,
    )
    comparands = pd.DataFrame({"real": real, "nominal": nominal}, index=index)

    panel = hybrid.residual_pcs(frame, components=1, settings=scheme)
    (result,) = hybrid.evaluate_row_83(panel, comparands, settings=scheme)

    assert result.loading_holds, "gold was built to hold the largest loading"
    assert result.placebo_holds, "the nominal leg is noise; the real leg is the driver"
    assert result.sign_holds, "gold was built to move against the real yield"
    assert result.passes
    assert result.gold_rank == 1
    assert result.correlation_real < 0.0
    assert result.assets_in_panel == 2
    assert result.null_probability == pytest.approx(0.5 * 0.5 * 0.5)


def test_the_sign_clause_is_invariant_to_the_eigenvector_sign(
    scheme: hybrid.HybridConfig,
) -> None:
    """Negating the eigenvector must not change any clause's verdict.

    LAPACK's eigenvector sign is arbitrary and not continuous in the matrix, so a
    different build could return ``-v`` where this one returns ``+v``, taking the
    factor series ``-f`` with it. Row 83's clause (c) is stated about a sign, so
    it is only meaningful if that pair of negations cancels exactly. This is the
    test of the ruling, not of the data.
    """
    periods = 260
    index = pd.bdate_range("2018-01-01", periods=periods, name="date")
    generator = np.random.default_rng(SETTINGS.model.seed + 2)
    real = generator.standard_normal(periods) * 4.0
    frame = pd.DataFrame(
        {
            "gold": -3.0 * real + 0.3 * generator.standard_normal(periods),
            "partner": -0.5 * real + 2.0 * generator.standard_normal(periods),
        },
        index=index,
    )
    comparands = pd.DataFrame(
        {"real": real, "nominal": generator.standard_normal(periods) * 4.0}, index=index
    )

    panel = hybrid.residual_pcs(frame, components=1, settings=scheme)
    negated = dataclasses.replace(
        panel,
        factors=-panel.factors,
        loadings={name: -frame_ for name, frame_ in panel.loadings.items()},
    )

    (straight,) = hybrid.evaluate_row_83(panel, comparands, settings=scheme)
    (flipped,) = hybrid.evaluate_row_83(negated, comparands, settings=scheme)

    assert straight.loading_holds == flipped.loading_holds
    assert straight.placebo_holds == flipped.placebo_holds
    assert straight.sign_holds == flipped.sign_holds
    assert straight.correlation_real == pytest.approx(flipped.correlation_real, abs=1e-12)
    assert straight.correlation_nominal == pytest.approx(flipped.correlation_nominal, abs=1e-12)


def test_the_sign_clause_is_NOT_invariant_to_negating_the_data(
    scheme: hybrid.HybridConfig,
) -> None:
    """The control that keeps the previous test from being vacuous.

    Negating every residual is an ECONOMIC change, not a convention: gold now
    rises when real yields rise. Clause (c) must notice. A criterion invariant to
    both the eigenvector sign and the data's sign would be measuring nothing, and
    this asserts that the second invariance does not hold.
    """
    periods = 260
    index = pd.bdate_range("2018-01-01", periods=periods, name="date")
    generator = np.random.default_rng(SETTINGS.model.seed + 2)
    real = generator.standard_normal(periods) * 4.0
    frame = pd.DataFrame(
        {
            "gold": -3.0 * real + 0.3 * generator.standard_normal(periods),
            "partner": -0.5 * real + 2.0 * generator.standard_normal(periods),
        },
        index=index,
    )
    comparands = pd.DataFrame(
        {"real": real, "nominal": generator.standard_normal(periods) * 4.0}, index=index
    )

    (straight,) = hybrid.evaluate_row_83(
        hybrid.residual_pcs(frame, components=1, settings=scheme), comparands, settings=scheme
    )
    (inverted,) = hybrid.evaluate_row_83(
        hybrid.residual_pcs(-frame, components=1, settings=scheme), comparands, settings=scheme
    )

    assert straight.sign_holds
    assert not inverted.sign_holds
    assert straight.correlation_real == pytest.approx(-inverted.correlation_real, abs=1e-12)
    # The loading rank is a magnitude question and is untouched by either flip.
    assert straight.gold_rank == inverted.gold_rank


def test_the_placebo_fails_a_pc_that_is_only_nominal_rates(
    scheme: hybrid.HybridConfig,
) -> None:
    """Clause (b) is the whole reason a bare correlation would not do.

    A component driven by the NOMINAL yield change still correlates with the real
    one -- the two legs run about 0.77 correlated in the real data -- so a
    significance test would pass it. The placebo must not.
    """
    periods = 300
    index = pd.bdate_range("2017-01-02", periods=periods, name="date")
    generator = np.random.default_rng(SETTINGS.model.seed + 3)
    nominal = generator.standard_normal(periods) * 5.0
    real = 0.8 * nominal + 0.6 * generator.standard_normal(periods) * 5.0

    frame = pd.DataFrame(
        {
            "gold": -4.0 * nominal + 0.2 * generator.standard_normal(periods),
            "partner": -1.0 * nominal + 3.0 * generator.standard_normal(periods),
        },
        index=index,
    )
    comparands = pd.DataFrame({"real": real, "nominal": nominal}, index=index)
    panel = hybrid.residual_pcs(frame, components=1, settings=scheme)
    (result,) = hybrid.evaluate_row_83(panel, comparands, settings=scheme)

    assert result.loading_holds, "the panel is built so gold leads"
    assert abs(result.correlation_real) > 0.3, "a significance test would pass this easily"
    assert not result.placebo_holds, "but it is nominal-rates exposure and must fail the placebo"
    assert not result.passes


def test_the_null_rates_are_the_registered_arithmetic() -> None:
    """1/13, 1/2, 1/2 and their products. The numbers printed beside the result."""
    single, independent_single = hybrid.loading_rank_null(_UNIVERSE_ASSETS, 1)
    assert single == pytest.approx(1.0 / 13.0)
    assert independent_single == pytest.approx(1.0 / 13.0)

    union, independent = hybrid.loading_rank_null(_UNIVERSE_ASSETS, 2)
    assert union == pytest.approx(2.0 / 13.0)
    assert independent == pytest.approx(25.0 / 169.0)
    assert independent < union, "the union bound must be the conservative one"


# ---------------------------------------------------------------------------
# Against the real cache
# ---------------------------------------------------------------------------


@pytest.mark.dataset
def test_the_government_control_is_the_duration_leg_and_not_the_excess_return() -> None:
    """W2-P3's mechanical control, fixed before any residual PC existed.

    The four synthetic government zeros must be regressed on ``duration_effect``;
    every other asset must be untouched. This is the assertion that keeps row 83
    readable by its own registered criterion.
    """
    plain = betas.asset_excess_returns(config=SETTINGS)
    swapped, substituted = hybrid.hybrid_regressands(config=SETTINGS)

    assert substituted == ("govt_10y", "govt_2y", "govt_30y", "govt_5y")
    assert "tips_10y" not in substituted, "the registered control names the GOVERNMENT zeros"

    for asset, series in plain.items():
        if asset in substituted:
            shared = series.index.intersection(swapped[asset].index)
            assert not np.allclose(
                series.loc[shared].to_numpy(dtype=float),
                swapped[asset].loc[shared].to_numpy(dtype=float),
            )
        else:
            pd.testing.assert_series_equal(series, swapped[asset])


@pytest.mark.dataset
def test_the_residual_is_the_regression_equation_it_claims_to_be() -> None:
    """``u = r - alpha - beta.f`` reproduced independently from the panel's own parts."""
    panel = hybrid.residual_panel(config=SETTINGS)
    regressands, _ = hybrid.hybrid_regressands(config=SETTINGS)

    factor_panel = macro.macro_factor_panel(config=SETTINGS)
    exposures = betas.build_exposures(
        {
            name: series.loc[
                (series.index >= factor_panel.start)
                & (series.index < pd.Timestamp(SETTINGS.require_holdout_start()))
            ]
            for name, series in regressands.items()
        },
        factor_panel.complete,
        units=factor_panel.units,
        settings=SETTINGS.model.factors.macro.beta,
    )
    scaled = betas.regression_units(factor_panel.complete.dropna(how="any"), factor_panel.units)

    asset = "gold"
    dates = panel.residuals.index
    block = exposures.exposures.xs(asset, level="asset").loc[dates, list(exposures.factors)]
    fitted = (block * scaled.loc[dates, list(exposures.factors)]).sum(axis=1)
    realised = regressands[asset].loc[dates] * macro._BPS_PER_UNIT
    expected = realised - exposures.alpha[asset].loc[dates] - fitted

    assert np.allclose(
        panel.residuals[asset].to_numpy(dtype=float), expected.to_numpy(dtype=float), atol=1e-10
    )


@pytest.mark.dataset
def test_the_panel_has_the_thirteen_investable_assets() -> None:
    """``N = 13``, which is row 83's null denominator."""
    panel = hybrid.residual_panel(config=SETTINGS)
    assert panel.variables == _UNIVERSE_ASSETS
    assert "gold" in panel.assets
    assert panel.residuals.index[-1] < pd.Timestamp(SETTINGS.require_holdout_start())


@pytest.mark.dataset
def test_row_83_is_refuted_on_the_real_panel() -> None:
    """The registered verdict, pinned so a later change cannot quietly flip it.

    Row 83 refuted in W3-P6: `gold` ranks 11th of 13 on both components. This
    test does NOT assert the rank is 11 -- that would pin a measurement. It
    asserts the VERDICT, because the verdict is what was registered, and a
    session that changes the model enough to reverse it owes a `model-config`
    row and an amendment to row 83 rather than a silently green test.
    """
    _, pcs, comparands = hybrid.build(config=SETTINGS)
    for pc in pcs:
        for result in hybrid.evaluate_row_83(pc, comparands, settings=SCHEME):
            assert not result.loading_holds, f"{result.component}: clause (a) was refuted in W3-P6"
            assert not result.passes


@pytest.mark.dataset
def test_appending_a_pc_never_lowers_the_r_squared_on_a_shared_design() -> None:
    """Control C4, after the fix it forced.

    The first version of ``augmented_r_squared`` compared a baseline estimated on
    a different date set, and three assets showed a negative gain from adding a
    regressor. On a shared design that is impossible, and this asserts it.
    """
    panel = hybrid.residual_panel(config=SETTINGS)
    pcs = tuple(
        hybrid.residual_pcs(panel.residuals, components=count, settings=SCHEME)
        for count in SCHEME.component_counts
    )
    table = hybrid.augmented_r_squared(panel, pcs, config=SETTINGS)
    for count in SCHEME.component_counts:
        assert (table[f"gain_{count}"] >= -1e-12).all()
    assert (table["gain_2"] >= table["gain_1"] - 1e-12).all()


@pytest.mark.dataset
def test_the_comparand_legs_are_the_same_tenor_and_are_differences() -> None:
    """Row 83's placebo needs two curves at one tenor, first-differenced."""
    frame = hybrid.comparand_yield_changes(config=SETTINGS)
    assert list(frame.columns) == ["real", "nominal"]
    assert frame.index[-1] < pd.Timestamp(SETTINGS.require_holdout_start())
    # A yield CHANGE is centred near zero; a yield LEVEL is not. This is the
    # cheapest available check that the ruling's first-difference was applied.
    for column in frame.columns:
        assert abs(float(frame[column].mean())) < float(frame[column].std())


# ---------------------------------------------------------------------------
# SPEC.md 15.2 -- the pipeline must not know what it is looking at
# ---------------------------------------------------------------------------


def test_the_risk_package_imports_nothing_from_factors() -> None:
    """CLAUDE.md invariant 10's corollary, restated at the hybrid.

    ``tests/test_risk_architecture.py`` already enforces this globally. It is
    asserted again here because the hybrid is the frame that would break it: a
    pipeline that needed to know a column was "a macro factor" or "a residual PC"
    would have to reach into ``factors/`` to find out.
    """
    root = Path("src/mafrm/risk")
    offenders = [
        path.name
        for path in sorted(root.glob("*.py"))
        if "from mafrm.factors" in path.read_text(encoding="utf-8")
        or "import mafrm.factors" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


@dataclasses.dataclass(frozen=True)
class _StubPanel:
    """The one attribute :func:`hybrid.factor_panel` reads off a ``FactorPanel``."""

    raw: pd.DataFrame

    @property
    def complete(self) -> pd.DataFrame:
        return self.raw.dropna(how="any")


def test_the_hybrid_frame_is_the_macro_panel_plus_the_pcs(
    scheme: hybrid.HybridConfig,
) -> None:
    """``factor_panel`` APPENDS and does not transform. Synthetic, no cache.

    The columns must arrive in the pipeline exactly as their own modules built
    them. A rescale here -- putting the macro columns into regression units, say,
    to "match" the PCs -- would make the hybrid's matrices incomparable with
    Model A's own pipeline run, which consumes the mixed-unit panel.
    """
    periods = 120
    index = pd.bdate_range("2020-01-01", periods=periods, name="date")
    generator = np.random.default_rng(SETTINGS.model.seed + 7)
    residuals = pd.DataFrame(
        {
            "gold": generator.standard_normal(periods) * 40.0,
            "other": generator.standard_normal(periods) * 5.0,
            "third": generator.standard_normal(periods) * 12.0,
        },
        index=index,
    )
    pcs = hybrid.residual_pcs(residuals, components=2, settings=scheme)
    named = pd.DataFrame({"equity": generator.standard_normal(periods) * 0.01}, index=index)

    frame = hybrid.factor_panel(pcs, _StubPanel(named))  # type: ignore[arg-type]

    assert list(frame.columns) == ["equity", "residual_pc1", "residual_pc2"]
    pd.testing.assert_series_equal(
        frame["equity"], named["equity"].reindex(frame.index), check_names=False
    )
    for name in ("residual_pc1", "residual_pc2"):
        pd.testing.assert_series_equal(
            frame[name], pcs.factors[name].reindex(frame.index), check_names=False
        )


@pytest.mark.dataset
def test_the_hybrid_runs_the_unchanged_pipeline_psd_at_every_stage() -> None:
    """SPEC.md 15.2's acceptance, all eight combinations. W3-P6.

    Both component counts x both horizons x both eigenfactor scalings, every
    declared stage run, every stage positive semi-definite. This is the harder
    version of the test than Model B was: the hybrid's frame mixes named macro
    factors in two units with dimensionless residual components, which is where
    an asset-class assumption in ``risk/`` would surface if one existed.
    """
    results = hybrid_report.acceptance(SETTINGS)
    expected = (
        len(SCHEME.component_counts)
        * len(hybrid_report.HORIZONS)
        * len(SETTINGS.model.eigenfactor.scaling_a)
    )
    assert len(results) == expected == 8

    declared = hybrid_report.RiskConfig.load(horizon=hybrid_report.HORIZONS[0], config=SETTINGS)
    for result in results:
        assert result.complete, f"{result.column.name}: a stage did not run"
        assert len(result.stages) == len(declared.stages)
        assert result.passed, f"{result.column.name}: {result.minimum_eigenvalues}"
        for stage, value in zip(result.stages, result.minimum_eigenvalues, strict=True):
            assert value > 0.0, f"{result.column.name}/{stage} minimum eigenvalue {value:.3e}"
        assert result.factors == 6 + result.column.components
        assert np.isfinite(result.lambda_squared)


@pytest.mark.dataset
def test_the_committed_forecast_history_matches_the_panel_it_claims() -> None:
    """The digest guard, exercised. A moved factor series must be refused.

    W3-P5 hit this failure for real: an extraction fix moved Model B's factor
    series while ``config/model.yaml`` stayed byte-identical, so the risk-config
    digest alone would have accepted a cache describing a model that no longer
    existed. The hybrid has four modules that can move its panel.
    """
    panels = hybrid_report.factor_panels(SETTINGS)
    cache = hybrid_report.read_cache(panels=panels)
    assert cache.provenance["holdout_start"] == str(SETTINGS.model.sample.holdout_start)
    assert set(cache.provenance["panel_digest"]) == {str(c) for c in SCHEME.component_counts}

    moved = {count: frame * 1.0000001 for count, frame in panels.items()}
    with pytest.raises(ValueError, match="factor series itself moved"):
        hybrid_report.read_cache(panels=moved)


@pytest.mark.dataset
def test_the_bias_history_length_pins_it_to_its_own_component_count() -> None:
    """A history from the wrong panel would reduce to a plausible multiplier."""
    panels = hybrid_report.factor_panels(SETTINGS)
    cache = hybrid_report.read_cache(panels=panels)
    for column in hybrid_report.columns(SETTINGS):
        frame = panels[column.components]
        expected = len(frame) - hybrid_report._minimum_observations(frame, SETTINGS)
        assert len(cache.bias(column)) == expected
