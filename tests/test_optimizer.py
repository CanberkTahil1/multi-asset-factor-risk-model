"""SPEC.md 8's optimizer, alpha input and constraint machinery, every value by hand. W6-P1.

Three tests the task named -- the two-asset closed form, the ladder on a
deliberately infeasible problem, turnover monotone in ``gamma_trade`` -- plus
the hand-computed RSTR with its lag, SPEC.md 8.3's decomposition, SPEC.md 8.4's
``lambda`` on the spec's own worked example, the ``alpha = 0`` reproduction of
SPEC.md 6.2's family 4, and SPEC.md 11's no-look-ahead harness pointed at the
optimizer's weights (the version W5-P3 left for this session).
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from mafrm import config
from mafrm.backtest import alpha as alpha_mod
from mafrm.backtest import constraints as con
from mafrm.backtest import lookahead
from mafrm.backtest import optimizer as opt
from mafrm.risk import validation

CFG = config.load()
OPT = CFG.model.optimizer
EXACT = dict(rel=1e-9, abs=1e-12)


def _dates(n: int, start: str = "2015-01-05") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n, name="date")


# ---------------------------------------------------------------------------
# RSTR -- SPEC.md 15.4, by hand
# ---------------------------------------------------------------------------


def test_rstr_weights_are_normalised_half_life_decays() -> None:
    weights = alpha_mod.rstr_weights(3, 1)
    # raw 1, 1/2, 1/4 -> sum 7/4 -> 4/7, 2/7, 1/7
    assert weights.tolist() == pytest.approx([4 / 7, 2 / 7, 1 / 7], **EXACT)
    assert weights.sum() == pytest.approx(1.0, **EXACT)


def test_rstr_is_the_lagged_weighted_mean_by_hand() -> None:
    frame = pd.DataFrame({"a": [0.01, 0.02, 0.03, 0.04]}, index=_dates(4))
    out = alpha_mod.rstr(frame, window=2, halflife=1, lag=1)
    # t = 2 reads rows 0..1 (lag skips row 2): (2/3)*0.02 + (1/3)*0.01
    assert np.isnan(out["a"].iloc[0]) and np.isnan(out["a"].iloc[1])
    assert out["a"].iloc[2] == pytest.approx((2 / 3) * 0.02 + (1 / 3) * 0.01, **EXACT)
    assert out["a"].iloc[3] == pytest.approx((2 / 3) * 0.03 + (1 / 3) * 0.02, **EXACT)


def test_rstr_ignores_the_lagged_month_and_everything_after_t() -> None:
    rng = np.random.default_rng(CFG.seed)
    frame = pd.DataFrame(rng.standard_normal((60, 2)) * 0.01, index=_dates(60), columns=["a", "b"])
    base = alpha_mod.rstr(frame, window=20, halflife=5, lag=3)
    moved = frame.copy()
    moved.iloc[40:, :] = 9.0  # rows t-lag+1 .. end for t = 42
    again = alpha_mod.rstr(moved, window=20, halflife=5, lag=3)
    assert again.iloc[42].tolist() == base.iloc[42].tolist()
    assert not np.allclose(again.iloc[45].to_numpy(), base.iloc[45].to_numpy())


def test_a_nan_inside_the_window_gives_nan_not_a_shorter_window() -> None:
    frame = pd.DataFrame({"a": [0.01, np.nan, 0.03, 0.04, 0.05]}, index=_dates(5))
    out = alpha_mod.rstr(frame, window=2, halflife=1, lag=0)
    assert np.isnan(out["a"].iloc[1]) and np.isnan(out["a"].iloc[2])
    assert out["a"].iloc[3] == pytest.approx((2 / 3) * 0.04 + (1 / 3) * 0.03, **EXACT)


def test_log_excess_return_is_the_cne5_summand() -> None:
    total = pd.DataFrame({"a": [0.01]}, index=_dates(1))
    rf = pd.Series([0.0001], index=_dates(1))
    out = alpha_mod.log_excess_returns(total, rf)
    assert out["a"].iloc[0] == pytest.approx(np.log(1.01) - np.log(1.0001), **EXACT)


def test_demean_centres_and_leaves_incomplete_dates_nan() -> None:
    frame = pd.DataFrame(
        [[1.0, 2.0, 3.0], [1.0, np.nan, 3.0]], index=_dates(2), columns=list("abc")
    )
    out = alpha_mod.demean(frame)
    assert isinstance(out, pd.DataFrame)
    assert out.iloc[0].tolist() == [-1.0, 0.0, 1.0]
    assert out.iloc[1].isna().all()


def test_the_configured_rstr_is_cne5s_and_points_at_one_copy() -> None:
    a = OPT.alpha
    momentum = CFG.model.equity_descriptors.momentum
    assert (a.window, a.halflife, a.lag) == (504, 126, 21)
    assert (a.window, a.halflife, a.lag) == (momentum.window, momentum.halflife, momentum.lag)
    assert a.weights == "normalised" and a.cross_sectional == "demean"
    assert a.horizon_days == CFG.model.data.trading_days_per_month == 21


# ---------------------------------------------------------------------------
# SPEC.md 8.3 -- misalignment
# ---------------------------------------------------------------------------


def test_decomposition_by_hand_and_cos_theta() -> None:
    x = np.array([[1.0], [0.0], [0.0]])
    a = np.array([3.0, 4.0, 0.0])
    mis = alpha_mod.decompose(a, x)
    assert mis.spanned.tolist() == pytest.approx([3.0, 0.0, 0.0], **EXACT)
    assert mis.orthogonal.tolist() == pytest.approx([0.0, 4.0, 0.0], **EXACT)
    assert mis.cos_theta == pytest.approx(0.6, **EXACT)
    assert float(mis.spanned @ mis.orthogonal) == pytest.approx(0.0, abs=1e-12)


def test_alpha_in_the_factor_span_has_cos_theta_one_and_no_penalty() -> None:
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    a = x @ np.array([0.2, -0.1])
    mis = alpha_mod.decompose(a, x)
    assert mis.cos_theta == pytest.approx(1.0, abs=1e-12)
    penalty = alpha_mod.misalignment_penalty(mis, np.eye(3), gamma_risk=3.0)
    assert not penalty.active and penalty.psi_unit == 0.0 and not penalty.direction.any()


def test_misalignment_penalty_by_hand() -> None:
    mis = alpha_mod.decompose(np.array([3.0, 4.0, 0.0]), np.array([[1.0], [0.0], [0.0]]))
    # u = (0, 1, 0), u' I u = 1, lambda = 2 -> psi_unit = 2; ||alpha_perp||^2 = 16 -> psi_mis = 1/8
    penalty = alpha_mod.misalignment_penalty(mis, np.eye(3), gamma_risk=2.0)
    assert penalty.psi_unit == pytest.approx(2.0, **EXACT)
    assert penalty.direction.tolist() == pytest.approx([0.0, 1.0, 0.0], **EXACT)
    assert penalty.psi_mis == pytest.approx(0.125, **EXACT)


# ---------------------------------------------------------------------------
# SPEC.md 8.4 -- gamma_risk
# ---------------------------------------------------------------------------


def test_information_ratio_and_lambda_by_hand() -> None:
    sigma = np.diag([0.04, 0.09])
    alpha = np.array([0.02, 0.03])
    # alpha' Sigma^-1 alpha = 0.0004/0.04 + 0.0009/0.09 = 0.02
    assert opt.information_ratio(alpha, sigma) == pytest.approx(np.sqrt(0.02), **EXACT)
    assert opt.gamma_risk_from_spec(alpha, sigma, te_target=0.04) == pytest.approx(
        np.sqrt(0.02) / 0.08, **EXACT
    )


def test_spec_8_4s_worked_example_gives_6_25() -> None:
    # IR = 0.5 with Sigma = I and alpha = (0.3, 0.4); TE_target = 4% -> 6.25
    assert opt.gamma_risk_from_spec(
        np.array([0.3, 0.4]), np.eye(2), te_target=0.04
    ) == pytest.approx(6.25, **EXACT)


def test_lambda_makes_the_unconstrained_optimum_carry_te_target_whatever_alphas_scale() -> None:
    sigma = np.array([[0.04, 0.006], [0.006, 0.09]])
    for scale in (1.0, 17.0, 1e-4):
        alpha = scale * np.array([0.02, 0.05])
        lam = opt.gamma_risk_from_spec(alpha, sigma, te_target=0.03)
        h = np.linalg.solve(sigma, alpha) / (2 * lam)
        assert np.sqrt(h @ sigma @ h) == pytest.approx(0.03, rel=1e-9)


def test_alpha_zero_refuses_spec_8_4() -> None:
    with pytest.raises(opt.OptimizerError, match="lambda = 0"):
        opt.gamma_risk_from_spec(np.zeros(2), np.eye(2), te_target=0.03)


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------


def _two_asset() -> tuple[np.ndarray, np.ndarray, opt.FactorRisk]:
    sigma = np.array([[0.04, 0.006], [0.006, 0.09]])
    alpha = np.array([0.02, 0.05])
    risk = opt.FactorRisk(
        exposures=np.eye(2), factor_covariance=sigma, specific_variance=np.zeros(2)
    )
    return alpha, sigma, risk


def test_two_assets_no_costs_no_constraints_recovers_the_closed_form() -> None:
    alpha, sigma, risk = _two_asset()
    lam = 2.5
    result = opt.solve_rebalance(
        opt.RebalanceProblem(
            alpha=alpha,
            risk=risk,
            prior_weights=np.zeros(2),
            penalties=opt.Penalties(gamma_risk=lam, gamma_trade=0.0),
            cost=None,
            long_only=False,
            fully_invested=False,
        )
    )
    closed = np.linalg.solve(sigma, alpha) / (2 * lam)
    assert result.status == "optimal" and result.attempt.rung == 0
    assert result.weights.tolist() == pytest.approx(closed.tolist(), abs=1e-7)


def test_factor_structure_risk_equals_the_dense_quadratic_form() -> None:
    rng = np.random.default_rng(CFG.seed)
    x = rng.standard_normal((5, 2))
    f = np.array([[2.0, 0.3], [0.3, 1.0]])
    d = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    risk = opt.FactorRisk(exposures=x, factor_covariance=f, specific_variance=d)
    w = rng.standard_normal(5)
    assert float(np.sum((risk.risk_rows() @ w) ** 2)) == pytest.approx(
        float(w @ risk.covariance() @ w), rel=1e-12
    )


def test_a_non_psd_factor_covariance_is_refused_not_repaired() -> None:
    with pytest.raises(opt.OptimizerError, match="not PSD"):
        opt.FactorRisk(
            exposures=np.eye(2),
            factor_covariance=np.array([[1.0, 2.0], [2.0, 1.0]]),
            specific_variance=np.zeros(2),
        )


def _three_asset_risk() -> opt.FactorRisk:
    rng = np.random.default_rng(CFG.seed)
    x = rng.standard_normal((3, 2))
    f = np.array([[1e-4, 2e-5], [2e-5, 5e-5]])
    return opt.FactorRisk(exposures=x, factor_covariance=f, specific_variance=np.full(3, 2e-5))


def test_alpha_zero_cost_free_fully_invested_reproduces_family_4() -> None:
    """SPEC.md 8.5.1 ruling 1's control: with shorts allowed it IS family 4's min-var."""
    risk = _three_asset_risk()
    result = opt.solve_rebalance(
        opt.RebalanceProblem(
            alpha=np.zeros(3),
            risk=risk,
            prior_weights=np.zeros(3),
            penalties=opt.Penalties(gamma_risk=1.0, gamma_trade=0.0),
            cost=None,
            long_only=False,
            fully_invested=True,
        )
    )
    expected = validation.minimum_variance_weights(risk.covariance())
    assert result.weights.tolist() == pytest.approx(expected.tolist(), abs=1e-6)


def test_alpha_zero_cost_free_is_invariant_to_gamma_risk() -> None:
    risk = _three_asset_risk()

    def solve(gamma: float) -> np.ndarray:
        return opt.solve_rebalance(
            opt.RebalanceProblem(
                alpha=np.zeros(3),
                risk=risk,
                prior_weights=np.zeros(3),
                penalties=opt.Penalties(gamma_risk=gamma, gamma_trade=0.0),
                cost=None,
            )
        ).weights

    assert solve(1.0).tolist() == pytest.approx(solve(7.0).tolist(), abs=1e-6)
    assert solve(1.0).min() >= -1e-9 and solve(1.0).sum() == pytest.approx(1.0, abs=1e-9)


def _cost(adv_over_nav: np.ndarray) -> opt.TradingCost:
    return opt.TradingCost(
        half_spread=np.full(3, 1e-4),
        daily_volatility=np.full(3, 0.01),
        adv_over_nav=adv_over_nav,
        prefactor=CFG.model.costs.square_root_prefactor.patient,
        exponent=CFG.model.costs.total_cost_exponent,
    )


def test_the_ladder_relaxes_an_infeasible_cap_and_logs_which_constraint_bound() -> None:
    """From cash, sum(cap * V/NAV) = 0.18 < 1: the simplex and the cap cannot both hold."""
    result = opt.solve_rebalance(
        opt.RebalanceProblem(
            alpha=np.array([0.004, -0.001, 0.003]),
            risk=_three_asset_risk(),
            prior_weights=np.zeros(3),
            penalties=opt.Penalties(gamma_risk=5.0, gamma_trade=1.0),
            cost=_cost(np.ones(3)),
            limits=(con.Limit("adv_participation", 0.06, None),),
            relaxation_order=("adv_participation",),
        )
    )
    labels = [(attempt.label, status) for attempt, status in result.attempts]
    assert labels[0] == ("CLARABEL", "infeasible") and labels[1] == ("SCS", "infeasible")
    assert result.attempt.label == "CLARABEL without adv_participation"
    assert result.relaxed == ("adv_participation",) and not result.fell_back
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-8) and result.weights.min() >= -1e-9
    reading = result.readings[0]
    assert reading.name == "adv_participation" and reading.relaxed and not reading.binding


def test_every_solver_failing_falls_back_to_prior_weights_never_zeros() -> None:
    prior = np.array([0.2, 0.3, 0.5])
    result = opt.solve_rebalance(
        opt.RebalanceProblem(
            alpha=np.array([0.004, -0.001, 0.003]),
            risk=_three_asset_risk(),
            prior_weights=prior,
            penalties=opt.Penalties(gamma_risk=5.0, gamma_trade=1.0),
            cost=_cost(np.ones(3)),
            limits=(con.Limit("adv_participation", 0.06, None),),
            relaxation_order=("adv_participation",),
            primary_solver="NO_SUCH_SOLVER",
            fallback_solver="ALSO_NO_SOLVER",
        )
    )
    assert result.fell_back and result.attempt.label == con.PRIOR_WEIGHTS
    assert result.weights.tolist() == prior.tolist()
    assert len(result.attempts) == 5 and all(
        "exception" in s or s == con.PRIOR_WEIGHTS for _, s in result.attempts
    )


def test_the_same_cap_as_a_hinge_is_feasible_at_the_first_rung_and_reports_its_violation() -> None:
    result = opt.solve_rebalance(
        opt.RebalanceProblem(
            alpha=np.array([0.004, -0.001, 0.003]),
            risk=_three_asset_risk(),
            prior_weights=np.zeros(3),
            penalties=opt.Penalties(gamma_risk=5.0, gamma_trade=1.0),
            cost=_cost(np.ones(3)),
            limits=(con.Limit("adv_participation", 0.06, 1.0),),
        )
    )
    assert result.attempt.rung == 0 and result.status == "optimal"
    reading = result.readings[0]
    assert not reading.hard and reading.binding and reading.violation > 0.0
    # The three trades sum to one and the caps sum to 0.18, so at least 0.82 is violated.
    assert reading.violation == pytest.approx(1.0 - 0.18, abs=1e-6)


def test_a_hard_cap_that_binds_carries_a_positive_multiplier() -> None:
    # Caps sum to 1.2 from cash: feasible, and the cheapest-to-trade asset wants more than its cap.
    result = opt.solve_rebalance(
        opt.RebalanceProblem(
            alpha=np.array([0.02, 0.0, 0.0]),
            risk=_three_asset_risk(),
            prior_weights=np.zeros(3),
            penalties=opt.Penalties(gamma_risk=1.0, gamma_trade=1.0),
            cost=_cost(np.array([5.0, 10.0, 5.0])),
            limits=(con.Limit("adv_participation", 0.06, None),),
        )
    )
    reading = result.readings[0]
    assert reading.hard and reading.binding and reading.multiplier > 0.0
    assert result.weights[0] == pytest.approx(0.30, abs=1e-6)


def test_turnover_is_monotone_non_increasing_in_gamma_trade() -> None:
    grid = CFG.model.costs.gamma_trade
    gammas = np.arange(grid.sweep_min, grid.sweep_max + 1e-12, grid.sweep_step)
    prior = np.array([0.6, 0.3, 0.1])
    turnovers = []
    for gamma_trade in gammas:
        result = opt.solve_rebalance(
            opt.RebalanceProblem(
                alpha=np.array([0.001, 0.004, 0.003]),
                risk=_three_asset_risk(),
                prior_weights=prior,
                penalties=opt.Penalties(gamma_risk=5.0, gamma_trade=float(gamma_trade)),
                cost=_cost(np.array([2.0, 2.0, 2.0])),
            )
        )
        turnovers.append(result.turnover)
    assert len(turnovers) == 5 and turnovers[0] > 0.0
    assert all(later <= earlier + 1e-7 for earlier, later in itertools.pairwise(turnovers))
    assert turnovers[-1] < turnovers[0]


def test_trading_cost_in_the_objective_matches_the_evaluated_cost() -> None:
    cost = _cost(np.array([2.0, 2.0, 2.0]))
    z = np.array([0.1, -0.05, 0.0])
    coefficient = 0.58 * 0.01 / np.sqrt(2.0)
    expected = 1e-4 * 0.15 + coefficient * (0.1**1.5 + 0.05**1.5)
    assert cost.evaluate(z) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# The ladder and the summary, as values
# ---------------------------------------------------------------------------


def test_the_ladder_is_written_down_before_any_solve() -> None:
    limits = (con.Limit("adv_participation", 0.06, None), con.Limit("position_box", 1.0, None))
    attempts = con.ladder(limits, order=("adv_participation",), primary="A", fallback="B")
    assert [a.label for a in attempts] == [
        "A",
        "B",
        "A without adv_participation",
        "B without adv_participation",
        con.PRIOR_WEIGHTS,
    ]
    with pytest.raises(con.ConstraintError, match="hinge"):
        con.ladder(
            (con.Limit("turnover", 2.0, 1.0),), order=("turnover",), primary="A", fallback="B"
        )


def test_summarise_counts_binding_and_takes_the_percentile_over_binding_dates_only() -> None:
    def reading(binding: bool, multiplier: float, relaxed: bool = False) -> con.Reading:
        return con.Reading("adv_participation", True, 0.06, 0.0, binding, multiplier, 0.0, relaxed)

    rows = [
        (reading(True, 1.0),),
        (reading(True, 3.0),),
        (reading(False, 0.0),),
        (reading(False, 0.0, relaxed=True),),
    ]
    table = con.summarise(rows, percentile=0.5)
    row = table.loc["adv_participation"]
    assert (row["rebalances"], row["imposed"], row["binding"], row["relaxed"]) == (4, 3, 2, 1)
    assert row["binding_share"] == pytest.approx(2 / 3)
    assert row["multiplier_p50"] == pytest.approx(2.0)


def test_the_config_limits_are_the_ruled_ones() -> None:
    limits = con.limits_from_config(OPT.constraints)
    assert [limit.name for limit in limits] == ["adv_participation", "position_box", "turnover"]
    assert all(limit.hard for limit in limits)
    by_name = {limit.name: limit for limit in limits}
    assert by_name["adv_participation"].bound == 0.06
    assert by_name["position_box"].bound == 1.0 and by_name["turnover"].bound == 2.0
    assert OPT.relaxation_ladder == ("adv_participation",)
    assert OPT.constraints.lagrange_percentile == 0.80


# ---------------------------------------------------------------------------
# SPEC.md 11 -- the weights version of the no-look-ahead test
# ---------------------------------------------------------------------------


def test_optimizer_weights_at_t_are_causal() -> None:
    """The version W5-P3 left for W6-P1: RSTR through t-lag, covariance through t, weights at t."""
    rng = np.random.default_rng(CFG.seed + 7)
    n_dates, n_assets = 420, 3
    returns = pd.DataFrame(
        rng.standard_normal((n_dates, n_assets)) * 0.01,
        index=_dates(n_dates),
        columns=[f"x{i}" for i in range(n_assets)],
    )
    window, halflife, lag, lookback = 60, 20, 5, 60
    rf = pd.Series(0.0, index=returns.index)

    def weights_at(data: pd.DataFrame, t: pd.Timestamp) -> np.ndarray:
        through = data.loc[:t]
        rate = alpha_mod.rstr(
            alpha_mod.log_excess_returns(through, rf), window=window, halflife=halflife, lag=lag
        ).loc[t]
        alpha = np.asarray(alpha_mod.demean(rate), dtype=float) * 21.0
        recent = through.iloc[-lookback:].to_numpy(dtype=float)
        sigma = 21.0 * (recent.T @ recent) / lookback
        risk = opt.FactorRisk(
            exposures=np.eye(n_assets),
            factor_covariance=sigma,
            specific_variance=np.zeros(n_assets),
        )
        gamma = opt.gamma_risk_from_spec(alpha, sigma, te_target=0.05)
        result = opt.solve_rebalance(
            opt.RebalanceProblem(
                alpha=alpha,
                risk=risk,
                prior_weights=np.full(n_assets, 1.0 / n_assets),
                penalties=opt.Penalties(gamma_risk=gamma, gamma_trade=1.0),
                cost=_cost(np.full(3, 4.0)),
            )
        )
        assert not result.fell_back
        return result.weights

    minimum = CFG.model.backtest.no_lookahead.minimum_evaluation_dates
    harness_rng = np.random.default_rng(CFG.seed + 8)
    times = lookahead.evaluation_dates(
        pd.DatetimeIndex(returns.index),
        count=minimum,
        rng=harness_rng,
        minimum_prefix=window + lag + 1,
    )
    report = lookahead.assert_no_lookahead(
        weights_at,
        returns,
        times=times,
        minimum_evaluations=minimum,
        rng=harness_rng,
        perturb=lookahead.perturb_additive,
    )
    assert report.count >= minimum and report.sensitive


# ---------------------------------------------------------------------------
# The config block: the rulings pinned, the parser loud
# ---------------------------------------------------------------------------


def test_the_rulings_are_in_the_config() -> None:
    assert OPT.gamma_risk is None
    assert OPT.tracking_error_target.anchor == "equal_weight_realised_volatility"
    assert OPT.tracking_error_target.multiples == (0.5, 1.0, 2.0)
    assert OPT.tracking_error_target.verification_multiple == 1.0
    assert OPT.robustification.return_forecast_rho == 0.0
    assert OPT.robustification.covariance_varrho == 0.0
    assert OPT.gamma_hold == 0.0 and OPT.misalignment_penalty == "msci" and OPT.benchmark == "zero"
    assert OPT.constraints.long_only and OPT.constraints.fully_invested
    assert OPT.constraints.tracking_error.bound is None
    ver = OPT.verification
    assert (ver.alpha, ver.cost_regime, ver.eigenfactor_scaling, ver.horizon) == (
        "rstr",
        "patient",
        1.0,
        "short",
    )
    assert (
        ver.risk_variant == "eigen"
        and ver.gamma_trade == 1.0
        and ver.tracking_error_multiple == 1.0
    )
    assert (
        ver.book_size.rule == "aum_grid_endpoints_equal_weight"
        and ver.book_size.nav_dollars is None
    )
    assert ver.spread_ends == ("low", "high") and ver.control_cost_free


def test_the_adv_cap_is_the_upper_calibration_anchor() -> None:
    anchors = CFG.model.costs.calibration_anchors
    largest = max(
        point.participation_of_adv
        for regime in (anchors.patient, anchors.urgent)
        for point in regime.points
    )
    assert OPT.constraints.adv_participation.bound == largest == 0.06


def _raw_model() -> dict[str, Any]:
    with (Path(__file__).resolve().parents[1] / "config" / "model.yaml").open() as handle:
        return yaml.safe_load(handle)  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda o: o["alpha"].update({"construction": "reversal"}), "RSTR by ruling"),
        (lambda o: o["alpha"].update({"weights": "raw"}), "RATE"),
        (lambda o: o["alpha"].update({"cross_sectional": "standardise"}), "unit variance"),
        (lambda o: o["tracking_error_target"].update({"anchor": "0.04"}), "MEASURED"),
        (lambda o: o["tracking_error_target"].update({"verification_multiple": 1.5}), "band"),
        (lambda o: o.update({"gamma_risk": 0.0}), "positive pin"),
        (lambda o: o.update({"benchmark": "equal_weight"}), "ABSOLUTE"),
        (lambda o: o.update({"misalignment_penalty": "axioma"}), "COMPUTED"),
        (lambda o: o["constraints"]["adv_participation"].update({"cap": 0.05}), "LARGEST anchored"),
        (lambda o: o["constraints"]["position_box"].update({"upper": 1.5}), "mathematical maximum"),
        (lambda o: o["constraints"]["turnover"].update({"priority": 0.0}), "positive number"),
        (lambda o: o["constraints"].update({"lagrange_percentile": 1.0}), "probability"),
        (lambda o: o.update({"relaxation_ladder": ["turnover", "turnover"]}), "duplicate"),
        (lambda o: o.update({"relaxation_ladder": ["leverage"]}), "not a constraint"),
        (lambda o: o["verification"].update({"gamma_trade": 1.25}), "grid"),
        (lambda o: o["verification"].update({"eigenfactor_scaling": 1.2}), "published"),
        (
            lambda o: o["verification"]["book_size"].update({"nav_dollars": 1.0e7}),
            "nav_dollars must",
        ),
        (
            lambda o: o["verification"]["book_size"].update({"rule": "fixed"}),
            "positive nav_dollars",
        ),
        (lambda o: o["verification"].update({"spread_ends": ["mid"]}), "subset"),
        (lambda o: o["solver"].update({"fallback": "CLARABEL"}), "must differ"),
    ],
)
def test_optimizer_parser_fails_loudly(mutate: Any, fragment: str) -> None:
    raw = copy.deepcopy(_raw_model())
    mutate(raw["optimizer"])
    with pytest.raises(config.ConfigError, match=fragment):
        config._parse_model(raw)


def test_relaxation_ladder_refuses_a_hinge() -> None:
    raw = copy.deepcopy(_raw_model())
    raw["optimizer"]["constraints"]["adv_participation"]["priority"] = 2.0
    with pytest.raises(config.ConfigError, match="HARD"):
        config._parse_model(raw)


def test_the_objective_is_left_in_its_own_units_when_there_is_no_risk_term() -> None:
    """W6-P3: with gamma_risk = 0 the scale is 1, not the largest |alpha| (SPEC.md 9.3 ruling 6)."""
    from mafrm.backtest import optimizer as opt_mod

    risk = opt_mod.FactorRisk(
        exposures=np.array([[1.0], [0.5]]),
        factor_covariance=np.array([[4e-4]]),
        specific_variance=np.array([1e-4, 1e-4]),
    )
    alpha = np.array([0.03, -0.03])
    penalties = opt_mod.Penalties(gamma_risk=0.0, gamma_trade=1.0, gamma_hold=0.0, psi_unit=0.0)
    problem = opt_mod.RebalanceProblem(
        alpha=alpha, risk=risk, prior_weights=np.array([0.5, 0.5]), penalties=penalties, cost=None
    )
    assert opt_mod._objective_scale(problem) == 1.0
    with_risk = opt_mod.RebalanceProblem(
        alpha=alpha,
        risk=risk,
        prior_weights=np.array([0.5, 0.5]),
        penalties=opt_mod.Penalties(gamma_risk=2.0, gamma_trade=1.0, gamma_hold=0.0, psi_unit=0.0),
        cost=None,
    )
    variance = float(np.mean(np.diag(risk.covariance())))
    assert opt_mod._objective_scale(with_risk) == pytest.approx(2.0 * variance, rel=1e-12)
