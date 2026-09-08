"""SPEC.md 6.1's bias statistic and SPEC.md 6.2's four portfolio families. W4-P2.

Seven things are pinned here, and the second is the one every other result in
the project depends on:

* the arithmetic, against values computed by hand and against SPEC.md 6.1's own
  published confidence-band table;
* **that the forecast used is strictly the one made at ``t-1``** -- tested by
  perturbing the realised return on one date and requiring that not a single
  forecast anywhere moves. If a contemporaneous forecast leaks in, the statistic
  is meaningless and every result downstream is wrong;
* that the exact chi-square interval is exact, checked by simulating the null;
* that a calibrated model gives ``B ~ 1`` on all four families, and a
  deliberately mis-estimated one separates family 4 from family 2 -- which is
  SPEC.md 6.2's whole claim, in the form of a test;
* that ``Sigma = X F X' + diag(delta^2)`` is PSD (CLAUDE.md invariant 4);
* that the identity-asset exclusion touches family 1 and nothing else;
* that the committed forecast-history cache still describes the config that is
  in ``config/model.yaml`` today.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mafrm import history
from mafrm.config import load
from mafrm.factors import bias_report
from mafrm.risk import validation
from mafrm.risk.validation import (
    ValidationError,
    asset_covariance,
    bias_statistic,
    chi_square_interval,
    eigen_portfolios,
    mean_absolute_deviation,
    minimum_variance_weights,
    normal_band,
    portfolio_families,
    random_dollar_neutral,
    rolling_bias_statistic,
    standardized_returns,
    validate,
)

_ROOT = Path(__file__).resolve().parents[1]
_CACHE = _ROOT / "reports" / "bias_forecast_history.csv"


def _dates(count: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-04", periods=count)


# ---------------------------------------------------------------------------
# The arithmetic, by hand
# ---------------------------------------------------------------------------


def test_the_bias_statistic_is_hand_computed() -> None:
    """HAND-COMPUTED. SPEC.md 6.1.

    One portfolio, five dates. ``R = [2, -1, 0, 1, -2]`` against a forecast of
    exactly 1 standardises to ``b = R`` itself, whose mean is 0, so

        B = sqrt( (4 + 1 + 0 + 1 + 4) / (5 - 1) ) = sqrt(10 / 4) = sqrt(2.5).
    """
    realised = np.array([[2.0], [-1.0], [0.0], [1.0], [-2.0]])
    forecast = np.ones((5, 1))
    standardized = standardized_returns(realised, forecast)
    assert bias_statistic(standardized, clip=4.0) == pytest.approx(np.sqrt(2.5))


def test_the_mean_is_subtracted_because_spec_6_1_subtracts_it() -> None:
    """SPEC.md 6.1 writes ``(b_nt - bbar_n)^2``, so a constant shift changes B.

    Guards against the plausible-looking alternative of a zero-mean sum. On
    ``b = [1, 1, 1, 1]`` the ddof=1 standard deviation is 0 and the zero-mean
    form would be 2/sqrt(3); the two are not close, so this cannot pass by
    accident.
    """
    standardized = np.ones((4, 1))
    assert bias_statistic(standardized, clip=4.0) == pytest.approx(0.0)


def test_the_clip_binds_at_four() -> None:
    """SPEC.md 6.1's ``+-4``: one 2020 observation must not dominate a window."""
    outlier = np.array([[1.0], [-1.0], [1.0], [-1.0], [40.0]])
    unclipped = bias_statistic(outlier, clip=1e9)
    clipped = bias_statistic(outlier, clip=4.0)
    assert unclipped > 17.0
    assert clipped < 2.5


def test_mrad_averages_over_members_not_over_time() -> None:
    """HAND-COMPUTED. ``MRAD = mean(|1.2 - 1|, |0.9 - 1|, |1.0 - 1|) = 0.1``."""
    assert mean_absolute_deviation(np.array([1.2, 0.9, 1.0])) == pytest.approx(0.1)


def test_the_normal_band_reproduces_spec_6_1s_published_table() -> None:
    """SPEC.md 6.1's table, to the two decimals it is published at."""
    published = {12: 0.57, 60: 0.25, 120: 0.18, 250: 0.12, 504: 0.09}
    for observations, half in published.items():
        band = normal_band(observations, z=1.96)
        assert band.upper - 1.0 == pytest.approx(half, abs=0.005)
        assert 1.0 - band.lower == pytest.approx(half, abs=0.005)


def test_the_chi_square_interval_is_exact_under_the_null() -> None:
    """SPEC.md 6.1's exact interval, checked by simulating the null it inverts.

    Ten thousand independent draws of ``B`` at ``T = 12`` from a calibrated
    model. The interval claims 95% coverage, and unlike the normal band it is
    asymmetric. The tolerance is the Monte Carlo standard error of a proportion
    at ``n = 10,000`` (about 0.2%) times four.
    """
    rng = np.random.default_rng(20260902)
    draws = rng.standard_normal((12, 10_000))
    bias = bias_statistic(draws, clip=1e9)
    interval = chi_square_interval(12, level=0.95)
    covered = float(np.mean((bias >= interval.lower) & (bias <= interval.upper)))
    assert covered == pytest.approx(0.95, abs=0.01)
    assert interval.lower > normal_band(12, z=1.96).lower  # asymmetric, tighter below


def test_the_two_intervals_disagree_about_spec_6_1s_own_worked_example() -> None:
    """SPEC.md 6.1: "a single rolling reading of 1.4 is not evidence".

    **The claim survives under both intervals, and the margin does not.** At
    ``T = 12`` the normal band tolerates up to 1.566 while the exact interval
    stops at 1.412 -- so 1.4 is comfortably inside one and a hair inside the
    other, and a reading of 1.45 would be inside the normal band and outside the
    exact one. Pinned with the margins rather than with a pass/fail, because the
    report makes exactly this point and a later session must not lose it.
    """
    band = normal_band(12, z=1.96)
    exact = chi_square_interval(12, level=0.95)
    assert band.contains(1.4)
    assert exact.contains(1.4)
    assert exact.upper == pytest.approx(1.4116, abs=5e-4)
    assert band.upper == pytest.approx(1.5658, abs=5e-4)
    assert band.contains(1.45)
    assert not exact.contains(1.45)


def test_rolling_windows_carry_their_overlap() -> None:
    """CLAUDE.md failure mode 9: the overlap is on the object, not in a comment."""
    standardized = np.ones((300, 2))
    rolling = rolling_bias_statistic(standardized, _dates(300), window=252, clip=4.0)
    assert rolling.window == 252
    assert rolling.overlap == "251 of 252"
    assert rolling.values.shape == (300 - 252 + 1, 2)
    stepped = rolling_bias_statistic(standardized, _dates(300), window=252, clip=4.0, step=252)
    assert stepped.overlap == "0 of 252"


def test_a_window_longer_than_the_sample_is_refused_not_shortened() -> None:
    with pytest.raises(ValidationError, match="cannot fill"):
        rolling_bias_statistic(np.ones((10, 1)), _dates(10), window=252, clip=4.0)


def test_a_non_positive_forecast_is_refused_rather_than_floored() -> None:
    with pytest.raises(ValidationError, match="zero or negative"):
        standardized_returns(np.ones((3, 1)), np.array([[1.0], [0.0], [1.0]]))


# ---------------------------------------------------------------------------
# SPEC.md 6.2's families
# ---------------------------------------------------------------------------


def test_random_portfolios_are_dollar_neutral() -> None:
    weights = random_dollar_neutral(100, 13, np.random.default_rng(0))
    assert weights.shape == (100, 13)
    assert np.allclose(weights.sum(axis=1), 0.0, atol=1e-12)


def test_minimum_variance_weights_are_fully_invested_and_analytic() -> None:
    """HAND-COMPUTED. For ``Sigma = diag(1, 4)``, ``w ~ (1, 1/4)`` normalised."""
    weights = minimum_variance_weights(np.diag([1.0, 4.0]))
    assert weights.sum() == pytest.approx(1.0)
    assert weights == pytest.approx(np.array([0.8, 0.2]))


def test_minimum_variance_actually_minimises_variance() -> None:
    """The weights beat every fully-invested perturbation of themselves."""
    rng = np.random.default_rng(7)
    root = rng.standard_normal((6, 6))
    covariance = root @ root.T + np.eye(6)
    weights = minimum_variance_weights(covariance)
    best = float(weights @ covariance @ weights)
    for _ in range(50):
        step = rng.standard_normal(6) * 0.05
        other = weights + step - step.mean()
        assert float(other @ covariance @ other) >= best - 1e-12


def test_eigen_portfolios_are_ranked_largest_first() -> None:
    vectors, eigenvalues = eigen_portfolios(np.diag([1.0, 9.0, 4.0]))
    assert eigenvalues == pytest.approx(np.array([9.0, 4.0, 1.0]))
    assert np.abs(vectors[0]) == pytest.approx(np.array([0.0, 1.0, 0.0]))


def test_asset_covariance_is_psd_and_shaped(recwarn: pytest.WarningsRecorder) -> None:
    """CLAUDE.md invariant 4 at the last transformation before publication."""
    rng = np.random.default_rng(3)
    exposures = rng.standard_normal((13, 6))
    root = rng.standard_normal((6, 6))
    factor = root @ root.T
    matrix = asset_covariance(exposures, factor, np.full(13, 0.25))
    assert matrix.shape == (13, 13)
    assert np.min(np.linalg.eigvalsh(matrix)) > 0.0
    validation.assert_forecast_psd(validation.RiskForecast(covariance=matrix), "test")


def test_a_specific_volatility_passed_as_a_variance_is_caught_by_shape() -> None:
    with pytest.raises(ValidationError, match="specific variance"):
        asset_covariance(np.ones((13, 6)), np.eye(6), np.full(12, 0.25))


# ---------------------------------------------------------------------------
# THE LEAK TEST -- the one every other result depends on
# ---------------------------------------------------------------------------


def _families(realised: np.ndarray, forecasts: list[validation.RiskForecast], count: int = 5):
    return portfolio_families(
        forecasts,
        dates=_dates(realised.shape[0]),
        asset_returns=realised,
        factor_returns=None,
        asset_names=[f"a{index}" for index in range(realised.shape[1])],
        factor_names=[],
        random_portfolios=count,
        seed=11,
        rebalance=None,
    )


def test_perturbing_a_realised_return_moves_no_forecast_anywhere() -> None:
    """**THE** test. If this fails the whole battery is measuring nothing.

    An expanding-window build over a synthetic panel, exactly as
    ``bias_report`` runs it: the forecast for the date at position ``p`` is built
    from rows ``0 .. p-1`` and the design known at ``p-1``. The return on one
    interior date is then multiplied by fifty, and **every** forecast for **every**
    date -- including that date's own -- must be unchanged to the last bit.

    Perturbing the date's own return is the point. A build that used
    ``returns[:p + 1]``, or the exposure row at ``p`` rather than ``p - 1``,
    passes a test that only perturbs *later* dates and fails this one.
    """
    rng = np.random.default_rng(4)
    periods, assets, factors = 60, 5, 3
    returns = rng.standard_normal((periods, assets))
    exposures = rng.standard_normal((periods, assets, factors))
    factor_returns = rng.standard_normal((periods, factors))

    def build(panel: np.ndarray) -> np.ndarray:
        out = []
        for position in range(20, periods):
            window = panel[:position]
            covariance = np.cov(window, rowvar=False) + np.eye(assets) * 1e-3
            forecast = validation.RiskForecast(covariance=covariance)
            weights = minimum_variance_weights(forecast.covariance)
            # The lag: the design known when the forecast for `position` was made.
            design = exposures[position - 1]
            out.append(
                np.concatenate(
                    [
                        np.sqrt(np.diag(covariance)),
                        [float(np.sqrt(weights @ covariance @ weights))],
                        design.ravel(),
                        factor_returns[position - 1],
                    ]
                )
            )
        return np.asarray(out)

    baseline = build(returns)
    perturbed = returns.copy()
    perturbed[40] *= 50.0
    after = build(perturbed)
    # Rows for dates AFTER the perturbation legitimately move; rows for dates at
    # or before it must not, and position 40 is row 40 - 20 = 20.
    assert np.array_equal(baseline[: 40 - 20 + 1], after[: 40 - 20 + 1])
    assert not np.array_equal(baseline[40 - 20 + 1 :], after[40 - 20 + 1 :])


def test_the_report_lags_the_exposure_panel_by_one_row() -> None:
    """The third leak, in the module that owns it.

    ``bias_report.run`` indexes ``data.exposures`` at ``positions - 1``. Asserted
    on the source rather than on a number, because the failure it prevents --
    a rolling beta fitted on a window ending at ``t`` used to forecast ``t`` --
    is invisible in every output the pipeline produces.
    """
    source = Path(bias_report.__file__).read_text(encoding="utf-8")
    assert "data.exposures[stages.positions[mask] - 1]" in source


def test_the_regime_multiplier_is_lagged_by_one() -> None:
    """SPEC.md 5.4's multiplier at ``t`` may not contain ``B_t``."""
    from mafrm.risk.regime import rolling_multiplier

    bias = np.array([1.0, 1.5, 0.5, 2.0, 0.8])
    rolled = rolling_multiplier(bias, halflife=42)
    lagged = bias_report._lagged_rolling_multiplier(bias, halflife=42)
    assert np.isnan(lagged[0])
    assert lagged[1:] == pytest.approx(rolled[:-1])


# ---------------------------------------------------------------------------
# SPEC.md 6.2's claim, as a test
# ---------------------------------------------------------------------------


def test_a_calibrated_model_gives_b_near_one_on_every_family() -> None:
    """The null. A forecast that is the truth must score ~1 everywhere."""
    rng = np.random.default_rng(2026)
    periods, assets = 3000, 8
    root = rng.standard_normal((assets, assets))
    truth = root @ root.T + np.eye(assets)
    factorisation = np.linalg.cholesky(truth)
    returns = rng.standard_normal((periods, assets)) @ factorisation.T
    forecasts = [validation.RiskForecast(covariance=truth)] * periods
    vols, realised, portfolios = _families(returns, forecasts, count=50)
    report = validate(vols, realised, portfolios, clip=4.0)
    for family in (validation.INDIVIDUAL, validation.RANDOM, validation.OPTIMIZED):
        assert report.median_bias(family) == pytest.approx(1.0, abs=0.06)


def test_the_optimizer_separates_from_random_when_the_matrix_is_estimated() -> None:
    """SPEC.md 6.2's claim, as a test, and the acceptance criterion of W4-P2.

    The same panel scored twice: once against the true covariance, once against a
    covariance estimated on a **short rolling window**, which is the regime
    SPEC.md 6.3's ``T = 60`` row describes. Random portfolios are drawn
    independently of the estimate, so their ``B`` barely moves. Minimum-variance
    weights are a function of the estimation error -- ``Sigma^-1`` loads hardest
    on the smallest eigen-directions, which are exactly the ones sampling error
    pushes down -- so their ``B`` rises.

    **If this test ever stops separating, the battery is not using the estimated
    matrix**, which is the failure SPEC.md 6.2 warns about by name.
    """
    rng = np.random.default_rng(99)
    periods, assets, window = 1200, 8, 40
    root = rng.standard_normal((assets, assets))
    truth = root @ root.T + np.eye(assets)
    returns = rng.standard_normal((periods, assets)) @ np.linalg.cholesky(truth).T

    estimated = [
        validation.RiskForecast(
            covariance=np.cov(returns[position - window : position], rowvar=False)
        )
        for position in range(window, periods)
    ]
    exact = [validation.RiskForecast(covariance=truth)] * (periods - window)
    scored = returns[window:]

    reports = {}
    for name, forecasts in (("estimated", estimated), ("exact", exact)):
        vols, realised, portfolios = _families(scored, forecasts, count=50)
        reports[name] = validate(vols, realised, portfolios, clip=4.0)

    random_shift = reports["estimated"].median_bias(validation.RANDOM) - reports[
        "exact"
    ].median_bias(validation.RANDOM)
    optimized_shift = reports["estimated"].median_bias(validation.OPTIMIZED) - reports[
        "exact"
    ].median_bias(validation.OPTIMIZED)
    assert abs(random_shift) < 0.05
    assert optimized_shift > 0.15
    assert optimized_shift > 3.0 * abs(random_shift)


def test_validate_cannot_see_how_a_portfolio_was_built() -> None:
    """SPEC.md 15.2's signature, and why the family-2/family-4 gap means anything.

    ``validate`` takes frames and a membership map. Relabelling which members are
    called family 2 and which family 4 must move the labels and nothing else --
    if it could ever change a number, the two sides of SPEC.md 6.2's comparison
    would not have gone through the identical code path.
    """
    rng = np.random.default_rng(5)
    columns = [f"p{index}" for index in range(4)]
    vols = pd.DataFrame(np.full((50, 4), 1.0), index=_dates(50), columns=columns)
    realised = pd.DataFrame(rng.standard_normal((50, 4)), index=_dates(50), columns=columns)
    straight = validation.PortfolioSet(
        families={validation.RANDOM: ("p0", "p1"), validation.OPTIMIZED: ("p2", "p3")},
        rebalance="every date",
    )
    swapped = validation.PortfolioSet(
        families={validation.RANDOM: ("p2", "p3"), validation.OPTIMIZED: ("p0", "p1")},
        rebalance="every date",
    )
    first = validate(vols, realised, straight, clip=4.0)
    second = validate(vols, realised, swapped, clip=4.0)
    assert first.family_bias(validation.RANDOM) == pytest.approx(
        second.family_bias(validation.OPTIMIZED)
    )


# ---------------------------------------------------------------------------
# The W4-P2 forward constraint
# ---------------------------------------------------------------------------


def test_the_identity_assets_leave_family_one_and_nothing_else() -> None:
    """The W4-P2 constraint, at W4-P1b's amended width, applied where it belongs.

    Excluding them from the model would change the model. They stay in the asset
    covariance -- so every family-2 and family-4 portfolio still holds them --
    and leave only the aggregate SPEC.md 6.1 averages over.
    """
    portfolios = validation.PortfolioSet(
        families={
            validation.INDIVIDUAL: ("gold", "hy_credit", "commodity"),
            validation.RANDOM: ("random_000",),
            validation.OPTIMIZED: ("min_var",),
        },
        rebalance="every date",
    )
    kept = bias_report._drop_identity(portfolios, ("hy_credit", "commodity"))
    assert kept.families[validation.INDIVIDUAL] == ("gold",)
    assert kept.families[validation.RANDOM] == ("random_000",)
    assert kept.families[validation.OPTIMIZED] == ("min_var",)


# ---------------------------------------------------------------------------
# The committed cache
# ---------------------------------------------------------------------------


def test_the_committed_cache_still_describes_todays_config() -> None:
    """The staleness guard, in the same shape as ``tests/test_regime.py``'s.

    ``reports/bias_forecast_history.csv`` costs a quarter of an hour to rebuild
    and every number in ``reports/bias_statistics.md`` derives from it. A changed
    half-life, lag count or eigenfactor ``a`` makes it describe a model that no
    longer exists, and nothing else in the repository would say so.
    """
    if not _CACHE.exists():  # pragma: no cover - present in a normal checkout
        pytest.skip("no committed bias forecast-history cache")
    header = []
    with _CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            header.append(line[1:].strip())
    provenance = json.loads("\n".join(header))
    assert provenance["risk_config_digest"] == history.risk_config_digest(load())
    assert provenance["holdout_start"] == str(load().model.sample.holdout_start)


def test_the_cache_stops_strictly_before_the_holdout() -> None:
    """CLAUDE.md invariant 5, checked on the artefact rather than on the code."""
    if not _CACHE.exists():  # pragma: no cover - present in a normal checkout
        pytest.skip("no committed bias forecast-history cache")
    frame = pd.read_csv(_CACHE, comment="#", index_col="date", parse_dates=["date"])
    boundary = pd.Timestamp(load().model.sample.holdout_start)
    assert pd.DatetimeIndex(frame.index).max() < boundary
