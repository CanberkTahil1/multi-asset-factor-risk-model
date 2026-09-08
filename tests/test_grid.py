"""SPEC.md 9's grid runner: the holdout guard, the cells, the dense variants, the golden fixture. W6-P2."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from mafrm import config, numerics
from mafrm.backtest import grid
from mafrm.backtest import optimizer as opt
from mafrm.factors import statistical

CFG = config.load()
GRID = CFG.model.optimizer.grid
EXACT = dict(rel=1e-9, abs=1e-12)
_GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden_weights.json"


# ---------------------------------------------------------------------------
# CLAUDE.md invariant 5, enforced in code
# ---------------------------------------------------------------------------


def test_assert_in_sample_fires_on_a_rebalance_on_or_after_the_boundary() -> None:
    boundary = pd.Timestamp("2025-01-01")
    before = pd.DatetimeIndex(["2024-10-31", "2024-11-29"])
    grid.assert_in_sample(before, pd.Timestamp("2024-12-31"), boundary)
    with pytest.raises(grid.HoldoutError, match="rebalance on 2025-01-31"):
        grid.assert_in_sample(
            before.append(pd.DatetimeIndex(["2025-01-31"])), pd.Timestamp("2025-02-28"), boundary
        )
    # Equality is a breach: "on or after".
    with pytest.raises(grid.HoldoutError, match="rebalance on 2025-01-01"):
        grid.assert_in_sample(
            pd.DatetimeIndex(["2024-11-29", "2025-01-01"]), pd.Timestamp("2025-01-31"), boundary
        )


def test_assert_in_sample_fires_when_the_last_holding_period_closes_in_the_holdout() -> None:
    boundary = pd.Timestamp("2025-01-01")
    dates = pd.DatetimeIndex(["2024-10-31", "2024-11-29"])
    with pytest.raises(grid.HoldoutError, match="holding period closes on 2025-01-02"):
        grid.assert_in_sample(dates, pd.Timestamp("2025-01-02"), boundary)
    with pytest.raises(grid.HoldoutError, match="closes on 2025-01-01"):
        grid.assert_in_sample(dates, pd.Timestamp("2025-01-01"), boundary)


def test_run_cell_refuses_a_boundary_inside_its_window_before_any_solve() -> None:
    inputs = grid.synthetic_inputs(CFG, cell=grid.CellSpec("ewma", "none"), seed=1)
    inside = dataclasses.replace(inputs, holdout_start=pd.Timestamp(inputs.rebalance_dates[3]))
    with pytest.raises(grid.HoldoutError, match=r"CLAUDE\.md invariant 5"):
        grid.run_cell(inside, CFG)
    # And the boundary the repository actually carries is the pinned one.
    assert inputs.holdout_start == pd.Timestamp("2025-01-01")


# ---------------------------------------------------------------------------
# The cells
# ---------------------------------------------------------------------------


def test_the_grid_is_28_cells_with_treatment_d_carrying_both_regimes() -> None:
    specs = grid.cells(CFG)
    ids = {spec.cell_id for spec in specs}
    assert len(ids) == 28
    assert len(specs) == 35  # 7 variants x (3 + 2 regimes for D)
    assert [s.regime for s in specs if s.cell_id == "4D"] == ["patient", "urgent"]
    assert specs[0].cell_id == "1A" and specs[-1].cell_id == "7D"
    assert grid.CellSpec("ledoit_wolf", "flat_spread").dense
    assert not grid.CellSpec("ewma", "none").charged
    assert not grid.CellSpec("ewma", "gross_then_net").optimizer_sees_cost
    assert grid.CellSpec("ewma", "gross_then_net").charged
    with pytest.raises(grid.GridError, match="unknown variant"):
        grid.CellSpec("oas", "none")


# ---------------------------------------------------------------------------
# The dense variants
# ---------------------------------------------------------------------------


def test_equal_weighted_sample_covariance_is_the_second_moment_about_zero() -> None:
    block = np.array([[1.0, 2.0], [3.0, 4.0]])
    sigma, intensity = grid.dense_covariance(block, variant="sample_equal_weight")
    # R'R / T = [[1+9, 2+12], [2+12, 4+16]] / 2
    np.testing.assert_allclose(sigma, [[5.0, 7.0], [7.0, 10.0]], rtol=1e-12, atol=0.0)
    assert np.isnan(intensity)


def test_ledoit_wolf_keeps_the_sample_variances_and_shrinks_the_off_diagonal() -> None:
    rng = np.random.default_rng(CFG.seed)
    block = rng.standard_normal((60, 4)) @ np.array(
        [[1.0, 0.5, 0.2, 0.0], [0.0, 1.0, 0.3, 0.1], [0.0, 0.0, 1.0, 0.4], [0.0, 0.0, 0.0, 1.0]]
    )
    shrunk, intensity = grid.dense_covariance(block, variant="ledoit_wolf")
    sample, _ = grid.dense_covariance(block, variant="sample_equal_weight")
    assert 0.0 <= intensity <= 1.0
    assert np.diag(shrunk).tolist() == pytest.approx(np.diag(sample).tolist(), rel=1e-10)
    # About ZERO: the estimate must agree with the uncentred call, not the centred default.
    uncentred = statistical.ledoit_wolf_constant_correlation(block, centre=False)
    assert uncentred.covariance is not None
    np.testing.assert_allclose(shrunk, uncentred.covariance, rtol=1e-12, atol=0.0)
    centred = statistical.ledoit_wolf_constant_correlation(block)
    assert centred.covariance is not None
    assert not np.allclose(centred.covariance, shrunk)
    # PSD, and consumable by the optimizer as X = I.
    opt.FactorRisk(exposures=np.eye(4), factor_covariance=shrunk, specific_variance=np.zeros(4))


def test_dense_covariance_refuses_a_non_dense_variant_and_bad_blocks() -> None:
    with pytest.raises(grid.GridError, match="not a dense"):
        grid.dense_covariance(np.eye(2), variant="ewma")
    with pytest.raises(grid.GridError, match="T >= 2"):
        grid.dense_covariance(np.ones((1, 2)), variant="sample_equal_weight")


# ---------------------------------------------------------------------------
# The config: derived, not chosen
# ---------------------------------------------------------------------------


def test_the_grid_config_is_derived_from_the_published_constants() -> None:
    assert GRID.dense_window == 242
    assert GRID.dense_window == round(
        numerics.effective_sample_size(CFG.model.covariance.factor_volatility_halflife.short)
    )
    assert GRID.flat_full_spread_bps == CFG.model.costs.flat_spread_assumption_bps == 5.0
    assert GRID.flat_half_spread == pytest.approx(2.5e-4, **EXACT)
    anchors = CFG.model.costs.calibration_anchors
    assert (
        GRID.book_size.participation_of_adv
        == min(p.participation_of_adv for p in anchors.patient.points)
        == 0.02
    )
    assert GRID.sharpe_aggregation_periods == 12
    assert GRID.hinge_priorities == {"adv_participation": 0.0048}
    assert GRID.relaxation_ladder == ("tracking_error",)
    assert GRID.risk_aversion == "tracking_error_constraint"
    assert GRID.spread_end == "low"
    assert tuple(GRID.covariance_variants) == (
        "sample_equal_weight",
        "ewma",
        "newey_west",
        "eigenfactor_a1.0",
        "eigenfactor_a1.4",
        "volatility_regime",
        "ledoit_wolf",
    )
    assert tuple(GRID.cost_treatments) == ("none", "gross_then_net", "flat_spread", "time_varying")


def _raw_model() -> dict[str, Any]:
    with (Path(__file__).resolve().parents[1] / "config" / "model.yaml").open() as handle:
        return yaml.safe_load(handle)  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda g: g.update({"alpha": "reversal"}), "ruled alpha-hat"),
        (lambda g: g.update({"eigenfactor_scaling": 1.2}), "published"),
        (lambda g: g.update({"gamma_trade": 1.25}), "grid"),
        (lambda g: g.update({"tracking_error_multiple": 1.5}), "band"),
        (lambda g: g.update({"spread_end": "mid"}), "'low' or 'high'"),
        (lambda g: g.update({"risk_aversion": "cyclic_search"}), "risk_aversion"),
        (lambda g: g.update({"risk_aversion": "spec_initialisation"}), "no recovered multiplier"),
        (lambda g: g["book_size"].update({"participation_of_adv": 0.06}), "SMALLEST anchored"),
        (lambda g: g["book_size"].update({"nav_dollars": 8.0e6}), "nav_dollars"),
        (lambda g: g["book_size"].update({"rule": "fixed"}), "positive nav_dollars"),
        (lambda g: g.update({"covariance_variants": ["ewma"]}), "seven"),
        (lambda g: g.update({"dense_window_rule": "days_252"}), "DERIVED"),
        (lambda g: g.update({"cost_treatments": ["none"]}), "four"),
        (lambda g: g.update({"time_varying_regimes": ["patient"]}), "BOTH regimes"),
        (lambda g: g.update({"hinge_priorities": {"leverage": 1.0}}), "not a constraint"),
        (lambda g: g.update({"hinge_priorities": {"adv_participation": 0.0}}), "positive"),
        (lambda g: g.update({"relaxation_ladder": ["adv_participation"]}), "hinge here"),
        (lambda g: g.update({"relaxation_ladder": ["position_box"]}), "must be on the ladder"),
        (lambda g: g.update({"trial_count_source": "results/metrics.json"}), "experiments.md"),
    ],
)
def test_grid_parser_fails_loudly(mutate: Any, fragment: str) -> None:
    raw = copy.deepcopy(_raw_model())
    mutate(raw["optimizer"]["grid"])
    with pytest.raises(config.ConfigError, match=fragment):
        config._parse_model(raw)


# ---------------------------------------------------------------------------
# The runner on a synthetic cell
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_d() -> grid.CellResult:
    inputs = grid.synthetic_inputs(
        CFG, cell=grid.CellSpec("eigenfactor_a1.0", "time_varying"), seed=CFG.seed
    )
    return grid.run_cell(inputs, CFG)


def test_the_te_constraint_holds_and_binds_and_the_recovered_gamma_is_the_kkt_ratio(
    synthetic_d: grid.CellResult,
) -> None:
    rows = synthetic_d.rows
    te = synthetic_d.inputs.te_target
    tol = 1e-4 * te
    assert (rows["forecast_volatility"] <= te + tol).all()
    binding = rows["tracking_error_binding"].to_numpy(dtype=bool)
    assert binding.any()
    assert rows.loc[binding, "forecast_volatility"].to_numpy() == pytest.approx(te, rel=1e-4)
    # gamma_risk_recovered = mu / (2 TE_target), and it is what the report calls gamma_risk.
    mu = rows["te_multiplier"].to_numpy(dtype=float)
    assert rows["gamma_risk_recovered"].to_numpy() == pytest.approx(
        (mu / (2 * te)).tolist(), rel=1e-12
    )
    assert (rows.loc[binding, "gamma_risk_recovered"] > 0).all()
    assert (rows["rung"] <= 1).all() and not rows["fell_back"].any()


def test_the_identity_is_exact_on_the_cell_and_costs_are_charged(
    synthetic_d: grid.CellResult,
) -> None:
    ident = synthetic_d.identity
    assert ident.risk_model_term + ident.cost_term == pytest.approx(ident.gap, abs=1e-12)
    assert ident.cost_drag > 0.0
    assert len(synthetic_d.monthly) == len(synthetic_d.inputs.rebalance_dates)
    # Net is gross less the charged cost, period by period.
    m = synthetic_d.monthly
    assert (m["gross_return"] - m["net_return"]).to_numpy() == pytest.approx(
        m["cost_over_nav"].to_numpy(), abs=1e-14
    )
    assert synthetic_d.summary["cost_drag_bps_per_year"] == pytest.approx(
        ident.cost_drag * 1e4, **EXACT
    )


def test_a_cost_free_cell_charges_nothing_and_has_no_adv_limit() -> None:
    inputs = grid.synthetic_inputs(CFG, cell=grid.CellSpec("ewma", "none"), seed=CFG.seed)
    result = grid.run_cell(inputs, CFG)
    assert result.identity.cost_drag == 0.0 and result.identity.cost_term == 0.0
    assert "adv_participation_binding" not in result.rows.columns
    assert "tracking_error_binding" in result.rows.columns


def test_gross_then_net_solves_the_same_problem_as_the_paper_cell_and_is_charged() -> None:
    paper = grid.run_cell(
        grid.synthetic_inputs(CFG, cell=grid.CellSpec("ewma", "none"), seed=3), CFG
    )
    charged = grid.run_cell(
        grid.synthetic_inputs(CFG, cell=grid.CellSpec("ewma", "gross_then_net"), seed=3), CFG
    )
    # The first rebalance starts from cash in both, so the weights coincide exactly;
    # afterwards the charged book's drift differs and so may the prior.
    assert charged.targets.iloc[0].tolist() == pytest.approx(
        paper.targets.iloc[0].tolist(), abs=1e-7
    )
    assert charged.identity.cost_drag > 0.0 and paper.identity.cost_drag == 0.0
    assert charged.identity.gross_return == pytest.approx(paper.identity.gross_return, rel=0.05)


def test_the_golden_weights_reproduce() -> None:
    assert _GOLDEN.exists(), f"the golden fixture is not committed: {_GOLDEN}"
    golden: dict[str, Any] = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    fresh = grid.golden_payload(CFG, seed=int(golden["seed"]))
    assert set(fresh["cells"]) == set(golden["cells"])  # type: ignore[arg-type]
    for label, expected in golden["cells"].items():
        got = fresh["cells"][label]  # type: ignore[index]
        assert np.asarray(got["weights"]) == pytest.approx(
            np.asarray(expected["weights"]), abs=1e-6
        )
        assert got["forecast_volatility"] == pytest.approx(
            expected["forecast_volatility"], abs=1e-8
        )
        assert got["te_target"] == expected["te_target"]


def test_monthly_returns_by_hand() -> None:
    dates = pd.bdate_range("2015-01-02", periods=5)
    rebalances = pd.DatetimeIndex([dates[0], dates[2]])
    nav = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates)
    cost = pd.Series([0.5, 0.25], index=rebalances)

    class _Backtest:
        def __init__(self) -> None:
            self.nav = nav
            self.total_cost = cost

    rows = pd.DataFrame({"forecast_volatility": [0.02, 0.03]}, index=rebalances)
    out = grid.monthly_returns(rebalances, pd.Timestamp(dates[-1]), rows, _Backtest())  # type: ignore[arg-type]
    assert out["net_return"].tolist() == pytest.approx([102 / 100 - 1, 104 / 102 - 1], **EXACT)
    # Gross adds back the cost charged AT THE END mark of each period.
    assert out["gross_return"].tolist() == pytest.approx(
        [(102 + 0.25) / 100 - 1, 104 / 102 - 1], **EXACT
    )
    assert out["forecast_volatility"].tolist() == [0.02, 0.03]


def test_the_chart_data_reads_the_committed_metrics_and_the_identity_is_exact_in_them() -> None:
    metrics_path = Path(__file__).resolve().parents[1] / "results" / "metrics.json"
    data = grid.ChartData.from_metrics(metrics_path)
    assert set(data.terms) == {
        (v, t) for v in GRID.covariance_variants for t in GRID.cost_treatments
    }
    assert data.trials >= 38 and data.rebalances == 187
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["cells_evaluated"] == 28 and payload["holdout_start"] == "2025-01-01"
    for cell in payload["cells"]:
        ident = cell["identity"]
        gap = ident["sharpe_paper"] - ident["sharpe_real"]
        assert ident["risk_model_term"] + ident["cost_term"] == pytest.approx(gap, abs=1e-8)
        assert ident["bias_ratio"] == pytest.approx(
            ident["realised_volatility_annual"] / ident["forecast_volatility_annual"], rel=1e-8
        )


# ---------------------------------------------------------------------------
# W6-P3: the overrides, the re-solve, the financing line, the diagnostics hook
# ---------------------------------------------------------------------------


def _d_inputs(seed: int = 5, **overrides: Any) -> grid.CellInputs:
    base = grid.synthetic_inputs(
        CFG, cell=grid.CellSpec("eigenfactor_a1.0", "time_varying"), seed=seed
    )
    return dataclasses.replace(base, **overrides)


def test_a_position_box_override_is_imposed_hard_and_binds() -> None:
    n = 3
    bounded = grid.run_cell(_d_inputs(position_box=2.0 / n), CFG)
    free = grid.run_cell(_d_inputs(), CFG)
    tol = 1e-6
    assert (bounded.targets.to_numpy() <= 2.0 / n + tol).all()
    assert (free.targets.to_numpy().max(axis=1) > 2.0 / n + tol).any()
    assert "position_box_binding" in bounded.rows.columns
    assert bounded.rows["position_box_binding"].any()
    # Effective assets can never fall below 1 / max weight = n / 2 under the bound.
    assert (bounded.rows["effective_assets"] >= n / 2.0 - 1e-6).all()
    assert bounded.summary["position_box_binding_share"] > 0.0


def test_gamma_trade_override_lowers_turnover_and_the_grid_value_is_the_default() -> None:
    reference = grid.run_cell(_d_inputs(), CFG)
    explicit = grid.run_cell(_d_inputs(gamma_trade=GRID.gamma_trade), CFG)
    assert explicit.targets.to_numpy() == pytest.approx(reference.targets.to_numpy(), abs=1e-7)
    heavier = grid.run_cell(_d_inputs(gamma_trade=3.0), CFG)
    assert heavier.summary["turnover_annual"] < reference.summary["turnover_annual"]


def test_spec_initialisation_override_solves_at_the_spec_lambda_without_a_te_bound() -> None:
    result = grid.run_cell(_d_inputs(risk_aversion="spec_initialisation"), CFG)
    rows = result.rows
    assert "tracking_error_binding" not in rows.columns
    assert rows["gamma_risk_used"].to_numpy() == pytest.approx(
        rows["gamma_risk_spec"].to_numpy(), rel=1e-12
    )
    assert np.isnan(rows["te_multiplier"].to_numpy()).all()


def test_a_hard_adv_cap_replaces_the_hinge_and_its_ladder_can_drop_it() -> None:
    hard = grid.run_cell(
        _d_inputs(
            adv_participation_hard=True, relaxation_ladder=("adv_participation", "tracking_error")
        ),
        CFG,
    )
    assert "adv_participation_multiplier" in hard.rows.columns
    # A hard limit reports a multiplier (finite) rather than a hinge violation.
    assert np.isfinite(hard.rows["adv_participation_multiplier"].to_numpy()).all()
    assert (hard.rows["adv_participation_violation"] == 0.0).all()


def test_the_resolve_columns_exist_and_are_empty_when_nothing_was_flagged() -> None:
    result = grid.run_cell(_d_inputs(), CFG)
    if int(result.summary["resolve_count"]) == 0:
        assert result.rows["resolve_l1"].isna().all()
        assert result.summary["resolve_max_l1"] == 0.0
        assert result.summary["resolve_exceeds"] == 0.0
    else:  # pragma: no cover - depends on the solver's status on the synthetic cell
        flagged = result.rows["resolve_l1"].notna()
        assert (result.rows.loc[flagged, "status"] == "optimal_inaccurate").all()
        assert result.summary["resolve_max_l1"] == result.rows["resolve_l1"].max()


def test_the_resolve_is_the_same_problem_on_the_fallback_solver(monkeypatch: Any) -> None:
    # Force the flag on every rebalance by relabelling the primary's status, so
    # the re-solve runs on a cell where it can be checked against a direct call.
    real = grid.opt.solve_rebalance

    def relabel(problem: opt.RebalanceProblem) -> opt.RebalanceResult:
        out = real(problem)
        if problem.primary_solver == CFG.model.optimizer.solver.primary:
            return dataclasses.replace(out, status="optimal_inaccurate")
        return out

    monkeypatch.setattr(grid.opt, "solve_rebalance", relabel)
    result = grid.run_cell(_d_inputs(), CFG)
    assert result.summary["resolve_count"] == len(result.rows)
    l1 = result.rows["resolve_l1"].to_numpy(dtype=float)
    assert np.isfinite(l1).all() and (l1 >= 0.0).all()
    assert result.summary["resolve_max_l1"] == pytest.approx(l1.max(), **EXACT)
    # The re-solve puts the fallback FIRST; where it fails the ladder walks on to the
    # primary, so every flagged date carries a status and most start with the fallback.
    statuses = result.rows["resolve_status"]
    assert (statuses != "").all()
    assert statuses.str.startswith(CFG.model.optimizer.solver.fallback).sum() >= len(statuses) // 2
    assert np.isfinite(result.rows["resolve_te_ratio"].to_numpy(dtype=float)).all()
    assert result.summary["resolve_worst_te_ratio"] == pytest.approx(
        result.rows["resolve_te_ratio"].max(), **EXACT
    )
    # The AMENDED criterion (W6-P3b): every flagged date carries a verdict and an
    # objective gap; a re-run is owed only where the fallback is feasible AND better
    # AND the L1 is above the threshold.
    verdicts = result.rows["resolve_verdict"]
    assert set(verdicts) <= {
        "fallback infeasible",
        "primary better",
        "fallback better (feasible)",
    }
    gaps = result.rows["resolve_objective_gap"].to_numpy(dtype=float)
    assert np.isfinite(gaps).all()
    better = verdicts == "fallback better (feasible)"
    assert (gaps[better.to_numpy()] > 0).all()
    assert (gaps[(verdicts == "primary better").to_numpy()] <= 0).all()
    owed = (result.rows["resolve_l1"] > GRID.resolve.l1_threshold) & better
    assert result.summary["resolve_actionable"] == float(owed.sum())
    assert result.summary["resolve_exceeds"] == float(l1.max() > GRID.resolve.l1_threshold)
    # The MAX decides (ruling 6), not the median.
    assert result.summary["resolve_max_l1"] >= float(np.median(l1))


def test_the_resolve_verdict_by_hand() -> None:
    """The amended criterion on hand-built results: infeasible, worse, better."""
    limits = (
        grid.con.Limit(name="adv_participation", bound=0.06, priority=0.0048),
        grid.con.Limit(name="tracking_error", bound=0.02, priority=None),
    )

    def made(alpha: float, forecast_vol: float, violation: float = 0.0) -> Any:
        attempt = grid.con.Attempt(rung=0, solver="X", relaxed=())
        readings = (
            grid.con.Reading(
                name="adv_participation",
                hard=False,
                bound=0.06,
                value=0.0,
                binding=violation > 0,
                multiplier=float("nan"),
                violation=violation,
                relaxed=False,
            ),
        )
        return opt.RebalanceResult(
            weights=np.array([0.5, 0.5]),
            trade=np.zeros(2),
            attempt=attempt,
            attempts=((attempt, "optimal"),),
            status="optimal",
            readings=readings,
            forecast_variance=forecast_vol**2,
            trading_cost=0.0,
            at_lower_bound=0,
            terms={"alpha": alpha, "risk": 0.0, "misalignment": 0.0, "trading_cost": 0.0},
        )

    primary = made(0.010, 0.02)
    common = dict(te_target=0.02, constraint_mode=True, tolerance=1e-8)
    # Fallback 0.07% outside the bound: infeasible, whatever its objective.
    verdict, gap = grid._resolve_verdict(primary, made(0.011, 0.02 * 1.0007), limits, **common)
    assert verdict == "fallback infeasible" and gap == pytest.approx(0.1, **EXACT)
    # Feasible but worse (a hinge violation of 1.0 at priority 0.0048 costs it 0.0048).
    verdict, gap = grid._resolve_verdict(
        primary, made(0.011, 0.019, violation=1.0), limits, **common
    )
    assert verdict == "primary better" and gap == pytest.approx((0.011 - 0.0048 - 0.010) / 0.010)
    # Feasible and better.
    verdict, gap = grid._resolve_verdict(primary, made(0.0105, 0.019), limits, **common)
    assert verdict == "fallback better (feasible)" and gap == pytest.approx(0.05, **EXACT)


def test_the_financing_line_is_the_cash_path_at_the_rate_and_nan_without_one() -> None:
    without = grid.run_cell(_d_inputs(), CFG)
    assert np.isnan(without.summary["financing_bps_per_year"])
    base = _d_inputs()
    rate = pd.Series(0.0001, index=base.prices.index)  # 1bp per session, constant
    with_rate = grid.run_cell(dataclasses.replace(base, risk_free=rate), CFG)
    leg = grid.engine.financing_leg(with_rate.backtest, rate)
    # Over the holding periods only: before the first rebalance the book is cash
    # and nothing is scored there.
    held = leg.loc[leg.index > base.rebalance_dates[0]]
    years = len(with_rate.monthly) / base.periods_per_year
    expected = held.sum() / base.nav / years * 1e4
    assert with_rate.summary["financing_bps_per_year"] == pytest.approx(expected, **EXACT)
    # Long-only, fully invested: after every rebalance cash is minus that trade's cost,
    # so the leg is NEGATIVE and small next to the cost drag.
    assert expected < 0.0
    assert abs(expected) < 0.1 * with_rate.summary["cost_drag_bps_per_year"]
    assert with_rate.summary["mean_cash_over_nav"] < 0.0
    # The weights are untouched by the reporting line.
    assert with_rate.targets.to_numpy() == pytest.approx(without.targets.to_numpy(), abs=1e-12)


def test_the_diagnostics_run_on_every_cell_and_perold_is_degenerate_by_construction() -> None:
    base = _d_inputs()
    rng = np.random.default_rng(3)
    factor = pd.DataFrame(
        rng.standard_normal((len(base.prices), 1)) * 0.005, index=base.prices.index, columns=["f"]
    )
    result = grid.run_cell(
        dataclasses.replace(base, factor_returns=factor, sleeves={a: "one" for a in base.assets}),
        CFG,
    )
    d = result.diagnostics
    assert d is not None
    s = result.summary
    assert s["perold_delay_bps_per_year"] == 0.0 and s["perold_opportunity_bps_per_year"] == 0.0
    assert s["perold_shortfall_bps_per_year"] == pytest.approx(
        s["perold_impact_bps_per_year"] + s["perold_fees_bps_per_year"], **EXACT
    )
    # IS is the cost charged on each parent order over NAV_pre -- the paper book pays
    # nothing, the real one the cost -- annualised over the rebalances.
    bt = result.backtest
    per_order = (bt.total_cost / bt.nav_before_costs).to_numpy(dtype=float)
    assert s["perold_shortfall_bps_per_year"] == pytest.approx(
        per_order.mean() * base.periods_per_year * 1e4, rel=1e-9
    )
    # Treatment D prices the same model it is charged, so the residual is solver-precision zero.
    assert abs(s["perold_residual_abs_mean_bps"]) < 1e-3
    assert "attribution_bias_total" in s and "attribution_interval_lower" in s
    assert d.brinson_linked is not None and "total" in d.brinson_linked.columns
    # Variances add and the volatilities do not, on every rebalance.
    te = d.tracking_error
    assert (te["factor_variance"] + te["specific_variance"]).to_numpy() == pytest.approx(
        te["total_variance"].to_numpy(), rel=1e-12
    )
    assert ((te["factor_volatility"] + te["specific_volatility"]) > te["total_volatility"]).all()


def test_treatment_b_prices_nothing_so_its_perold_residual_is_the_realised_cost() -> None:
    inputs = grid.synthetic_inputs(CFG, cell=grid.CellSpec("ewma", "gross_then_net"), seed=5)
    result = grid.run_cell(inputs, CFG)
    s = result.summary
    assert s["perold_predicted_bps_per_year"] == 0.0
    assert s["perold_residual_bps_per_year"] == pytest.approx(
        s["perold_shortfall_bps_per_year"], **EXACT
    )


def test_cell_inputs_refuses_a_bad_override() -> None:
    with pytest.raises(grid.GridError, match="risk_aversion"):
        _d_inputs(risk_aversion="cyclic")
    with pytest.raises(grid.GridError, match="position_box"):
        _d_inputs(position_box=1.5)
    with pytest.raises(grid.GridError, match="gamma_trade"):
        _d_inputs(gamma_trade=0.0)


def test_the_w6p3_config_blocks_are_derived_from_the_rules() -> None:
    bands = GRID.bands
    assert bands.reference_variant == "eigenfactor_a1.0"
    assert bands.reference_treatment == "time_varying" and bands.reference_regime == "patient"
    assert bands.spread_ends == ("high",)
    assert bands.tracking_error_multiples == (0.5, 2.0)
    assert bands.gamma_trades == (1.5, 2.0, 2.5, 3.0)
    assert bands.position_box_equal_weight_multiple == 2
    assert bands.position_box(13) == pytest.approx(2 / 13, **EXACT)
    assert bands.horizons == ("long",) and bands.risk_aversions == ("spec_initialisation",)
    assert GRID.resolve.criterion == "max" and GRID.resolve.l1_threshold == 1e-3
    assert GRID.capacity.report_sharpe is False
    assert GRID.capacity.relaxation_ladder == ("adv_participation", "tracking_error")
    assert GRID.diagnostics.brinson_benchmark == "equal_weight"
    assert GRID.diagnostics.decision_price == "rebalance_close"
    assert CFG.model.backtest.financing.convention == "risk_free_both_ways"


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda g: g["bands"].update({"spread_ends": ["low"]}), "reference cell's own end"),
        (lambda g: g["bands"].update({"position_box_equal_weight_multiple": 1}), "equal weight"),
        (lambda g: g["bands"].update({"horizons": ["short"]}), "own horizon"),
        (lambda g: g["bands"].update({"book_size_rule": "fixed"}), "A_low and A_high"),
        (lambda g: g["resolve"].update({"criterion": "median"}), "MAX"),
        (lambda g: g["resolve"].update({"l1_threshold": 0.0}), "positive"),
        (lambda g: g["capacity"].update({"report_sharpe": True}), "N ~150"),
        (lambda g: g["capacity"].update({"adv_participation": "hinge"}), "hard"),
        (lambda g: g["capacity"].update({"relaxation_ladder": ["tracking_error"]}), "ladder"),
        (
            lambda g: g["diagnostics"]["attribution"].update({"brinson_benchmark": "cap_weight"}),
            "equal_weight",
        ),
        (
            lambda g: g["diagnostics"]["implementation_shortfall"].update(
                {"decision_price": "next_open"}
            ),
            "rebalance_close",
        ),
    ],
)
def test_w6p3_parser_fails_loudly(mutate: Any, fragment: str) -> None:
    raw = copy.deepcopy(_raw_model())
    mutate(raw["optimizer"]["grid"])
    with pytest.raises(config.ConfigError, match=fragment):
        config._parse_model(raw)


def test_financing_parser_refuses_a_spread_or_another_convention() -> None:
    raw = copy.deepcopy(_raw_model())
    raw["backtest"]["financing"]["borrowing_spread"] = 0.001
    with pytest.raises(config.ConfigError, match="invented number"):
        config._parse_model(raw)
    raw = copy.deepcopy(_raw_model())
    raw["backtest"]["financing"]["convention"] = "zero_interest"
    with pytest.raises(config.ConfigError, match="risk_free_both_ways"):
        config._parse_model(raw)


def test_cell_spec_key_and_label_carry_the_band_tag() -> None:
    plain = grid.CellSpec("eigenfactor_a1.0", "time_varying")
    tagged = grid.CellSpec("eigenfactor_a1.0", "time_varying", "patient", tag="spread_high")
    assert plain.key == "4D" and tagged.key == "4D:spread_high"
    assert tagged.label.endswith("/spread_high") and plain.label.count("/") == 2
