"""SPEC.md 10.1-10.3's diagnostics, every identity by hand. W6-P3."""

from __future__ import annotations

import ast
import inspect
import math

import numpy as np
import pandas as pd
import pytest

from mafrm.backtest import diagnostics as diag
from mafrm.risk import validation

EXACT = dict(rel=1e-12, abs=1e-14)


# ---------------------------------------------------------------------------
# 10.1 Perold
# ---------------------------------------------------------------------------


def test_perold_four_legs_by_hand_and_the_residual_is_realised_minus_predicted() -> None:
    dates = pd.bdate_range("2015-01-30", periods=2)
    nav = pd.Series([1000.0, 2000.0], index=dates)
    spread = pd.DataFrame({"a": [1.0, 2.0], "b": [0.5, 0.0]}, index=dates)
    impact = pd.DataFrame({"a": [0.2, 0.4], "b": [0.3, 0.6]}, index=dates)
    asym = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]}, index=dates)
    turnover = pd.Series([0.5, 0.25], index=dates)
    predicted = pd.Series([0.0018, 0.0010], index=dates)
    out = diag.implementation_shortfall(
        nav_before_costs=nav,
        spread_cost=spread,
        impact_cost=impact,
        asymmetry_cost=asym,
        turnover=turnover,
        predicted=predicted,
    )
    # Day 1: spread 1.5, impact 0.5 on NAV 1000 -> 0.0015 + 0.0005 = 0.0020.
    assert out.loc[dates[0], "impact_spread"] == pytest.approx(0.0015, **EXACT)
    assert out.loc[dates[0], "impact_market"] == pytest.approx(0.0005, **EXACT)
    assert out.loc[dates[0], "impact"] == pytest.approx(0.0020, **EXACT)
    assert out.loc[dates[0], "delay"] == 0.0 and out.loc[dates[0], "opportunity"] == 0.0
    assert out.loc[dates[0], "fees"] == 0.0
    assert out.loc[dates[0], "shortfall"] == pytest.approx(0.0020, **EXACT)
    assert out.loc[dates[0], "residual"] == pytest.approx(0.0020 - 0.0018, **EXACT)
    # Day 2: spread 2.0, impact 1.0 on NAV 2000 -> 0.0010 + 0.0005 = 0.0015; residual 0.0005.
    assert out.loc[dates[1], "shortfall"] == pytest.approx(0.0015, **EXACT)
    assert out.loc[dates[1], "residual"] == pytest.approx(0.0005, **EXACT)
    # A commission enters the fees leg on traded notional and nowhere else.
    with_fee = diag.implementation_shortfall(
        nav_before_costs=nav,
        spread_cost=spread,
        impact_cost=impact,
        asymmetry_cost=asym,
        turnover=turnover,
        predicted=predicted,
        commission_rate=0.0001,
    )
    assert with_fee.loc[dates[0], "fees"] == pytest.approx(0.5 * 0.0001, **EXACT)
    assert with_fee.loc[dates[0], "shortfall"] == pytest.approx(0.0020 + 0.00005, **EXACT)
    assert with_fee.loc[dates[0], "impact"] == pytest.approx(0.0020, **EXACT)


# ---------------------------------------------------------------------------
# 10.2 Tracking error and x-sigma-rho
# ---------------------------------------------------------------------------


def _two_factor_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w = np.array([0.5, 0.3, 0.2])
    x = np.array([[1.0, 0.0], [0.5, 1.0], [0.0, -1.0]])
    f = np.array([[0.04, 0.01], [0.01, 0.09]])
    delta = np.array([0.01, 0.02, 0.03])
    return w, x, f, delta


def test_tracking_error_decomposition_by_hand() -> None:
    w, x, f, delta = _two_factor_case()
    d = diag.decompose_tracking_error(w, x, f, delta)
    # x_a = X' w = (0.5 + 0.15, 0.3 - 0.2) = (0.65, 0.10).
    assert d.active_exposures.tolist() == pytest.approx([0.65, 0.10], **EXACT)
    # x' F x = 0.04*0.4225 + 2*0.01*0.065 + 0.09*0.01 = 0.0169 + 0.0013 + 0.0009 = 0.0191.
    assert d.factor_variance == pytest.approx(0.0191, **EXACT)
    # h' Delta h = 0.01*0.25 + 0.02*0.09 + 0.03*0.04 = 0.0025 + 0.0018 + 0.0012 = 0.0055.
    assert d.specific_variance == pytest.approx(0.0055, **EXACT)
    assert d.total_variance == pytest.approx(0.0246, **EXACT)
    assert d.factor_share == pytest.approx(0.0191 / 0.0246, **EXACT)
    # x-sigma-rho: (F x) = (0.04*0.65 + 0.01*0.10, 0.01*0.65 + 0.09*0.10) = (0.027, 0.0155).
    sigma_a = math.sqrt(0.0246)
    assert d.factor_volatilities.tolist() == pytest.approx([0.2, 0.3], **EXACT)
    assert d.factor_correlations.tolist() == pytest.approx(
        [0.027 / (0.2 * sigma_a), 0.0155 / (0.3 * sigma_a)], **EXACT
    )
    ctr = [0.65 * 0.027 / sigma_a, 0.10 * 0.0155 / sigma_a]
    assert d.factor_contributions.tolist() == pytest.approx(ctr, **EXACT)
    assert d.specific_contribution == pytest.approx(0.0055 / sigma_a, **EXACT)
    # Euler: the K + 1 contributions add to sigma_a exactly.
    assert sum(ctr) + 0.0055 / sigma_a == pytest.approx(sigma_a, **EXACT)


def test_the_volatilities_do_not_add_and_a_pair_that_adds_is_refused() -> None:
    w, x, f, delta = _two_factor_case()
    d = diag.decompose_tracking_error(w, x, f, delta)
    # Variances add; volatilities overshoot the total strictly.
    assert d.factor_variance + d.specific_variance == pytest.approx(d.total_variance, **EXACT)
    assert d.factor_volatility + d.specific_volatility > d.total_volatility
    # A "factor TE" and "specific TE" that sum to the total TE are not this decomposition:
    # variances of (0.3^2, 0.4^2) against a total of 0.7^2 do not add.
    with pytest.raises(diag.DiagnosticsError, match="additive in VARIANCE"):
        diag.TrackingErrorDecomposition(
            total_variance=0.49,
            factor_variance=0.09,
            specific_variance=0.16,
            active_exposures=np.zeros(1),
            factor_volatilities=np.zeros(1),
            factor_correlations=np.zeros(1),
            factor_contributions=np.zeros(1),
            specific_contribution=0.0,
        )


def test_a_diversifying_exposure_has_a_negative_contribution() -> None:
    w = np.array([0.5, 0.5])
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    f = np.array([[0.04, -0.018], [-0.018, 0.01]])  # rho = -0.9 between the factors
    delta = np.zeros(2)
    d = diag.decompose_tracking_error(w, x, f, delta)
    # (F x) = (0.02 - 0.009, -0.009 + 0.005) = (0.011, -0.004): factor 2 diversifies.
    assert d.factor_contributions[1] < 0.0
    assert d.contribution_shares[1] < 0.0
    assert d.factor_contributions.sum() == pytest.approx(d.total_volatility, **EXACT)


# ---------------------------------------------------------------------------
# 10.3 Attribution
# ---------------------------------------------------------------------------


class _Forecast:
    def __init__(self, x: np.ndarray, f: np.ndarray, d: np.ndarray) -> None:
        self.exposures = x
        self.factor_covariance = f
        self.specific_variance = d


def _attribution_case() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    dates = pd.bdate_range("2015-01-05", periods=7)
    prices = pd.DataFrame(
        {"a": [100.0, 101.0, 102.0, 100.0, 103.0, 104.0, 102.0], "b": [50.0] * 7}, index=dates
    )
    prices["b"] = [50.0, 50.5, 50.0, 51.0, 51.5, 51.0, 52.0]
    targets = pd.DataFrame({"a": [0.6, 0.4], "b": [0.4, 0.6]}, index=dates[[0, 3]])
    # Drifted daily weights: hold the targets fixed in this fixture so that the
    # fixed-weight and drifted returns coincide and the identity is checkable.
    weights = pd.DataFrame(
        {"a": [0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4], "b": [0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6]},
        index=dates,
    )
    factor = pd.DataFrame({"m": [0.0, 0.01, -0.01, 0.005, 0.0, 0.002, -0.004]}, index=dates)
    x = np.array([[1.0], [0.5]])
    forecasts = [
        _Forecast(x, np.array([[4e-4]]), np.array([1e-4, 2e-4])),
        _Forecast(x, np.array([[4e-4]]), np.array([1e-4, 2e-4])),
    ]
    return prices, targets, weights, factor, forecasts


def test_the_return_identity_holds_daily_and_the_period_sums_by_hand() -> None:
    prices, targets, weights, factor, forecasts = _attribution_case()
    out = diag.attribute(
        prices=prices,
        weights_daily=weights,
        targets=targets,
        terminal_date=pd.Timestamp(prices.index[-1]),
        forecasts=forecasts,
        factor_returns=factor,
        sleeves={"a": "eq", "b": "eq"},
    )
    p = out.periods
    assert len(p) == 2 and p["days"].tolist() == [3, 3]
    # Period 1, days 2-4 at w = (0.6, 0.4):
    ra = np.array([101 / 100 - 1, 102 / 101 - 1, 100 / 102 - 1])
    rb = np.array([50.5 / 50 - 1, 50 / 50.5 - 1, 51 / 50 - 1])
    daily = 0.6 * ra + 0.4 * rb
    f = np.array([0.01, -0.01, 0.005])
    x_a = 0.6 * 1.0 + 0.4 * 0.5  # 0.8
    assert p["portfolio_return"].iloc[0] == pytest.approx(daily.sum(), **EXACT)
    assert p["factor_return"].iloc[0] == pytest.approx((x_a * f).sum(), **EXACT)
    assert p["specific_return"].iloc[0] == pytest.approx((daily - x_a * f).sum(), **EXACT)
    assert p["contrib_m"].iloc[0] == pytest.approx((x_a * f).sum(), **EXACT)
    assert p["fixed_weight_return"].iloc[0] == pytest.approx(daily.sum(), **EXACT)
    # Buy-and-hold on the targets over the period's price change; the EW benchmark is the mean.
    period = np.array([100 / 100 - 1, 51 / 50 - 1])
    assert p["buy_and_hold_return"].iloc[0] == pytest.approx(0.6 * period[0] + 0.4 * period[1])
    assert p["benchmark_return"].iloc[0] == pytest.approx(period.mean(), **EXACT)
    # Ex ante: x' F x = 0.64 * 4e-4 = 2.56e-4; h' Delta h = 0.36e-4 + 0.32e-4 = 0.68e-4.
    assert p["ex_ante_factor_variance"].iloc[0] == pytest.approx(2.56e-4, **EXACT)
    assert p["ex_ante_specific_variance"].iloc[0] == pytest.approx(0.68e-4, **EXACT)
    assert p["ex_ante_total_variance"].iloc[0] == pytest.approx(3.24e-4, **EXACT)
    # Identity on the period sums.
    assert (p["factor_return"] + p["specific_return"]).to_numpy() == pytest.approx(
        p["portfolio_return"].to_numpy(), **EXACT
    )


def test_component_bias_fires_on_the_specs_own_example_and_passes_a_calibrated_one() -> None:
    # SPEC.md 10.3: ex ante says 60% of variance is one factor, ex post it explains 5%.
    rng = np.random.default_rng(7)
    t = 187
    total = 1.0
    ex_post_factor = rng.standard_normal(t) * math.sqrt(0.05 * total)
    ex_ante_factor = np.full(t, 0.60 * total)
    ex_post_factor = ex_post_factor / ex_post_factor.std(ddof=1) * math.sqrt(0.05 * total)
    comp = diag.component_bias(ex_post_factor, ex_ante_factor, name="factor", level=0.95)
    assert comp.bias == pytest.approx(math.sqrt(0.05 / 0.60), rel=1e-9)
    assert comp.bias == pytest.approx(0.2887, abs=5e-4)
    interval = validation.chi_square_interval(t, level=0.95)
    assert (comp.lower, comp.upper) == (interval.lower, interval.upper)
    assert comp.lower == pytest.approx(0.898, abs=5e-4) and comp.upper == pytest.approx(
        1.101, abs=5e-4
    )
    assert not comp.inside
    # A calibrated component sits inside: ex-post variance equal to the forecast.
    calibrated = ex_post_factor / ex_post_factor.std(ddof=1) * math.sqrt(0.60)
    assert diag.component_bias(calibrated, ex_ante_factor, name="f", level=0.95).inside
    # A component absent ex ante is undefined, not a failure.
    absent = diag.component_bias(np.zeros(t), np.zeros(t), name="s", level=0.95)
    assert not absent.defined and absent.inside


def test_reconcile_raises_on_corrupted_input_in_strict_mode_and_marks_otherwise() -> None:
    rng = np.random.default_rng(11)
    t = 60
    periods = pd.DataFrame(
        {
            "factor_return": rng.standard_normal(t) * 0.02,
            "specific_return": rng.standard_normal(t) * 0.01,
            "ex_ante_factor_variance": np.full(t, 0.02**2),
            "ex_ante_specific_variance": np.full(t, 0.01**2),
        }
    )
    periods["portfolio_return"] = periods["factor_return"] + periods["specific_return"]
    periods["ex_ante_total_variance"] = (
        periods["ex_ante_factor_variance"] + periods["ex_ante_specific_variance"]
    )
    good = diag.Attribution(periods=periods, factor_names=("m",))
    out = diag.reconcile(good, level=0.95, strict=True)
    assert set(out) == {"factor", "specific", "total"}
    # Corrupt the ex-ante factor variance by 10x: the factor component's B is ~1/sqrt(10).
    corrupted = periods.copy()
    corrupted["ex_ante_factor_variance"] *= 10.0
    corrupted["ex_ante_total_variance"] = (
        corrupted["ex_ante_factor_variance"] + corrupted["ex_ante_specific_variance"]
    )
    bad = diag.Attribution(periods=corrupted, factor_names=("m",))
    with pytest.raises(diag.DiagnosticsError, match="does not reconcile"):
        diag.reconcile(bad, level=0.95, strict=True)
    marked = diag.reconcile(bad, level=0.95, strict=False)
    assert not marked["factor"].inside
    assert marked["factor"].bias == pytest.approx(out["factor"].bias / math.sqrt(10.0), rel=1e-9)


def test_brinson_fachler_by_hand_and_the_identity() -> None:
    w = np.array([0.5, 0.3, 0.2])
    wb = np.array([1 / 3, 1 / 3, 1 / 3])
    r = np.array([0.10, 0.02, -0.04])
    sleeves = ["eq", "eq", "bond"]
    out = diag.brinson_fachler(w, wb, r, sleeves)
    b = float(wb @ r)  # 0.08/3
    # Equity sleeve: w = 0.8, W = 2/3, b_i = 0.06, b_i^p = (0.05 + 0.006)/0.8 = 0.07.
    assert out.loc["eq", "allocation"] == pytest.approx((0.8 - 2 / 3) * (0.06 - b), **EXACT)
    assert out.loc["eq", "selection"] == pytest.approx((2 / 3) * (0.07 - 0.06), **EXACT)
    assert out.loc["eq", "interaction"] == pytest.approx((0.8 - 2 / 3) * (0.07 - 0.06), **EXACT)
    # Bond sleeve: one member, so no selection and no interaction.
    assert out.loc["bond", "selection"] == 0.0 and out.loc["bond", "interaction"] == 0.0
    assert out.loc["bond", "allocation"] == pytest.approx((0.2 - 1 / 3) * (-0.04 - b), **EXACT)
    assert out["total"].sum() == pytest.approx(float(w @ r) - b, **EXACT)
    # The "- b" of Brinson-Fachler: an overweight to a sleeve that beat the benchmark is positive.
    assert out.loc["eq", "allocation"] > 0.0


def test_carino_linking_by_hand_and_order_independence() -> None:
    rp = np.array([0.10, -0.05])
    rb = np.array([0.04, 0.01])
    effects = pd.DataFrame({"alloc": [0.04, -0.02], "select": [0.02, -0.04]})
    linked = diag.carino_link(effects, rp, rb)
    cum_p, cum_b = 1.1 * 0.95 - 1, 1.04 * 1.01 - 1
    k = (math.log1p(cum_p) - math.log1p(cum_b)) / (cum_p - cum_b)
    k1 = (math.log1p(0.10) - math.log1p(0.04)) / 0.06
    k2 = (math.log1p(-0.05) - math.log1p(0.01)) / (-0.06)
    assert linked["alloc"] == pytest.approx((k1 * 0.04 + k2 * -0.02) / k, **EXACT)
    assert linked["select"] == pytest.approx((k1 * 0.02 + k2 * -0.04) / k, **EXACT)
    assert linked.sum() == pytest.approx(cum_p - cum_b, **EXACT)
    reversed_ = diag.carino_link(effects.iloc[::-1].reset_index(drop=True), rp[::-1], rb[::-1])
    assert reversed_.tolist() == pytest.approx(linked.tolist(), **EXACT)
    with pytest.raises(diag.DiagnosticsError, match="do not add"):
        diag.carino_link(effects.assign(alloc=[0.05, -0.02]), rp, rb)


def test_the_benchmark_never_acquires_a_sharpe() -> None:
    """Operator ruling 3: a benchmark that acquires a Sharpe becomes a trial.

    Structural: no identifier in the module's code names a Sharpe, and the
    metrics module (where every Sharpe lives) is not imported. Prose in the
    report template may say the word; code may not compute the thing.
    """
    tree = ast.parse(inspect.getsource(diag))
    names = {
        node.id if isinstance(node, ast.Name) else node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Name | ast.Attribute)
    }
    assert not {n for n in names if "sharpe" in n.lower()}
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not {m for m in imported if "metrics" in m}
