"""Config loader tests.

Two jobs: prove the loader fails loudly rather than defaulting, and pin the
published constants of the CLAUDE.md parameter table so that a silent edit to
config/model.yaml breaks the build instead of the results.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from mafrm.config import SEED, ConfigError, _parse_data, load, load_model, load_universe

# ---------------------------------------------------------------------------
# It loads at all
# ---------------------------------------------------------------------------


def test_load_returns_both_configs() -> None:
    config = load()
    assert config.model.schema_version == 1
    assert config.universe.schema_version == 1
    assert config.seed == SEED


def test_sample_start_is_the_headline_window() -> None:
    # SPEC.md 3.1: headline window 2007-04 to present, driven by HYG.
    assert load_model().sample.start == date(2007, 4, 1)


# ---------------------------------------------------------------------------
# CLAUDE.md parameter table -- USE4 Table 4.1
# ---------------------------------------------------------------------------


def test_covariance_halflives_and_lags() -> None:
    cov = load_model().covariance
    assert (cov.factor_volatility_halflife.short, cov.factor_volatility_halflife.long) == (84, 252)
    # Identical across horizons -- deliberate, not a copy-paste error.
    assert (cov.factor_correlation_halflife.short, cov.factor_correlation_halflife.long) == (
        504,
        504,
    )
    assert (cov.volatility_newey_west_lags.short, cov.volatility_newey_west_lags.long) == (5, 5)
    assert (cov.correlation_newey_west_lags.short, cov.correlation_newey_west_lags.long) == (2, 2)


def test_vra_halflife_is_shared_by_factor_and_specific() -> None:
    # USE4 Table 4.1 gives one half-life, applied to both. Its own section, so
    # a second copy cannot appear under specific_risk and drift.
    vra = load_model().volatility_regime_adjustment.halflife
    assert (vra.short, vra.long) == (42, 168)


def test_specific_risk_parameters() -> None:
    spec = load_model().specific_risk
    assert (spec.ewma_halflife.short, spec.ewma_halflife.long) == (84, 252)
    assert (spec.newey_west_lags.short, spec.newey_west_lags.long) == (5, 5)
    assert (spec.autocorrelation_halflife.short, spec.autocorrelation_halflife.long) == (252, 252)
    assert (spec.bayesian_shrinkage_q.short, spec.bayesian_shrinkage_q.long) == (0.1, 0.1)


def test_eigenfactor_runs_both_scalings() -> None:
    # CLAUDE.md: run BOTH. a=1.0 is attribution-facing (USE4 production),
    # a=1.4 is optimizer-facing. They are not alternatives.
    eig = load_model().eigenfactor
    assert eig.scaling_a == (1.0, 1.4)
    assert 1000 <= eig.monte_carlo_trials <= 3000


# ---------------------------------------------------------------------------
# EWMA effective sample size -- hand-computed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("halflife", "expected"),
    [
        # T_eff = 2*tau/ln(2), computed by hand:
        #   2*84  / 0.6931472 = 242.37 -> 242
        #   2*252 / 0.6931472 = 727.12 -> 727
        #   2*504 / 0.6931472 = 1454.24 -> 1454
        (84, 242),
        (252, 727),
        (504, 1454),
    ],
)
def test_effective_sample_size_matches_closed_form(halflife: int, expected: int) -> None:
    reference = load_model().numerics.effective_sample_size.reference
    assert reference[halflife] == expected
    assert int(2 * halflife / math.log(2)) == expected


def test_psd_floor_matches_invariant_4() -> None:
    assert load_model().numerics.psd_eigenvalue_floor == 1e-14


# ---------------------------------------------------------------------------
# CNE5 descriptors -- SPEC.md 15.4
# ---------------------------------------------------------------------------


def test_equity_descriptor_windows() -> None:
    desc = load_model().equity_descriptors
    assert (desc.beta.window, desc.beta.halflife) == (252, 63)
    assert (desc.dastd.window, desc.dastd.halflife) == (252, 42)
    assert (desc.hsigma.window, desc.hsigma.halflife) == (252, 63)
    # RSTR is LAGGED: the sum runs t = 21 ... 525, i.e. 504 observations
    # ending 21 trading days before the estimation date.
    assert (desc.momentum.window, desc.momentum.halflife, desc.momentum.lag) == (504, 126, 21)
    assert desc.momentum.lag + desc.momentum.window == 525


def test_composite_weights() -> None:
    desc = load_model().equity_descriptors
    resvol = desc.resvol_composite
    assert (resvol.dastd, resvol.cmra, resvol.hsigma) == (0.74, 0.16, 0.10)
    liquidity = desc.liquidity_composite
    assert (liquidity.stom, liquidity.stoq, liquidity.stoa) == (0.35, 0.35, 0.30)


# ---------------------------------------------------------------------------
# Costs -- SPEC.md 7.2
# ---------------------------------------------------------------------------


def test_cost_parameters() -> None:
    costs = load_model().costs
    assert costs.square_root_prefactor.patient == 0.58
    assert costs.square_root_prefactor.urgent == 1.40
    assert costs.total_cost_exponent == 1.5
    assert (costs.gamma_trade.sweep_min, costs.gamma_trade.sweep_max) == (1.0, 3.0)
    # Must be time-varying: 63-day rolling window, monthly step.
    assert costs.edge_spread.window == 63
    assert costs.edge_spread.step == "monthly"


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------


def test_universe_is_the_frozen_fourteen() -> None:
    universe = load_universe()
    assert len(universe.assets) == 14
    assert universe.frozen_date == date(2026, 8, 25)
    # SPEC.md 3.1: three splices maximum; two are spent.
    assert sum(a.splice for a in universe.assets) == 2


def test_synthetic_series_have_no_ticker() -> None:
    universe = load_universe()
    synthetic = [a for a in universe.assets if a.ticker is None]
    assert {a.id for a in synthetic} == {
        "govt_2y",
        "govt_5y",
        "govt_10y",
        "govt_30y",
        "tips_10y",
    }
    assert all(a.maturity_years is not None for a in synthetic)
    assert "SPY" in universe.tickers
    assert len(universe.tickers) == 9


def test_by_id_raises_for_unknown_asset() -> None:
    with pytest.raises(KeyError):
        load_universe().by_id("nonexistent")


# ---------------------------------------------------------------------------
# Invariant 5 -- the holdout must fail loudly while unset
# ---------------------------------------------------------------------------


def test_require_holdout_start_raises_while_unset() -> None:
    config = load()
    if config.model.sample.holdout_start is None:
        with pytest.raises(ConfigError, match="holdout_start"):
            config.require_holdout_start()
    else:
        assert config.require_holdout_start() > config.model.sample.start


# ---------------------------------------------------------------------------
# It fails loudly rather than defaulting
# ---------------------------------------------------------------------------


def _model_dict() -> dict[str, object]:
    with (Path(__file__).resolve().parents[1] / "config" / "model.yaml").open() as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_missing_required_key_raises_with_its_path(tmp_path: Path) -> None:
    payload = _model_dict()
    covariance = payload["covariance"]
    assert isinstance(covariance, dict)
    del covariance["factor_volatility_halflife"]
    with pytest.raises(ConfigError, match=r"model\.covariance\.factor_volatility_halflife"):
        load_model(_write(tmp_path, payload))


def test_wrong_type_raises_rather_than_coercing(tmp_path: Path) -> None:
    payload = _model_dict()
    covariance = payload["covariance"]
    assert isinstance(covariance, dict)
    covariance["volatility_newey_west_lags"] = {"short": "five", "long": 5}
    with pytest.raises(ConfigError, match="expected an int"):
        load_model(_write(tmp_path, payload))


def test_composite_weights_must_sum_to_one(tmp_path: Path) -> None:
    payload = _model_dict()
    descriptors = payload["equity_descriptors"]
    assert isinstance(descriptors, dict)
    descriptors["resvol_composite"] = {"dastd": 0.74, "cmra": 0.16, "hsigma": 0.20}
    with pytest.raises(ConfigError, match=r"must sum to 1\.0"):
        load_model(_write(tmp_path, payload))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_model(tmp_path / "absent.yaml")


def test_network_is_blocked_in_unmarked_tests() -> None:
    """The autouse guard in conftest is the enforcement of invariant 1."""
    import socket

    with pytest.raises(RuntimeError, match="invariant 1"):
        socket.socket()


# ---------------------------------------------------------------------------
# Data layer (SPEC.md 3.2, amended 2026-08-26)
# ---------------------------------------------------------------------------


def test_roll_down_step_is_one_over_252() -> None:
    # SPEC.md 3.2: Delta = 1/252. Held as the denominator so one definition of
    # the year is used everywhere.
    data = load_model().data
    assert data.trading_days_per_year == 252
    assert data.roll_down_step == 1.0 / 252.0


def test_svensson_guard_constants() -> None:
    data = load_model().data
    # SPEC.md 3.2 point 2: the n -> 0 limit is used below this maturity.
    assert data.svensson_small_n == 1e-6
    # Observed property of feds200628.csv, not a published constant.
    assert data.curve_tau2_missing_sentinel == -999.99


def test_curve_maturities_match_the_frozen_universe() -> None:
    data = load_model().data
    universe = load_universe()
    synthetic = tuple(
        asset.maturity_years
        for asset in universe.assets
        if asset.construction == "synthetic_gsw_zero_curve"
    )
    assert data.nominal_curve.maturities_years == synthetic
    assert data.real_curve.maturities_years == (10.0,)


def test_per_tenor_history_starts_are_the_published_ones() -> None:
    # The Fed publishes a fitted yield only out to the longest bond outstanding.
    # tests/test_gsw.py pins these against the file itself.
    starts = {a.id: a.history_start for a in load_universe().assets if a.id.startswith("govt_")}
    assert starts["govt_2y"] == date(1961, 6, 14)
    assert starts["govt_5y"] == date(1961, 6, 14)
    assert starts["govt_10y"] == date(1971, 8, 16)
    assert starts["govt_30y"] == date(1985, 11, 25)


def test_risk_free_leg_is_ken_french_and_its_loader_has_landed() -> None:
    """W1-P4 flipped `implemented`; excess returns now begin 1961, not 2001."""
    rf = load_model().data.risk_free
    # Chosen because it covers the whole 1961- GSW sample with no splice.
    assert rf.selected == "ken_french_daily_rf"
    assert rf.ken_french_daily_rf.history_start == date(1926, 7, 1)
    assert rf.ken_french_daily_rf.implemented is True
    assert rf.ken_french_daily_rf.dataset == "F-F_Research_Data_Factors_daily"
    assert rf.ken_french_daily_rf.rf_column == "RF"
    # The units field exists because the substitution is otherwise silent:
    # Ken French publishes percent per trading day, DGS1MO percent per annum.
    assert rf.ken_french_daily_rf.units == "percent_per_trading_day"
    # Retained as the cross-check comparand for that substitution.
    assert rf.interim_fred.implemented is True
    assert rf.interim_fred.fred_series == "DGS1MO"
    assert rf.interim_fred.history_start == date(2001, 7, 31)


def test_an_unrecognised_risk_free_unit_is_refused() -> None:
    """A different unit needs a different conversion, not a different string."""
    document = _model_document()
    document["data"]["risk_free"]["ken_french_daily_rf"]["units"] = "percent_per_annum"
    with pytest.raises(ConfigError, match="percent_per_trading_day"):
        _parse_data(document)


def _model_document() -> dict[str, Any]:
    from mafrm.config import config_dir

    parsed = yaml.safe_load((config_dir() / "model.yaml").read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def test_tlt_gate_is_the_amended_one() -> None:
    gate = load_model().data.tlt_cross_check
    # SPEC.md 3.2 amended: the gate is the ladder gap at TLT's expense ratio,
    # with beta indistinguishable from one.
    assert gate.max_abs_gap_pct_per_year == 0.15
    assert (gate.beta_min, gate.beta_max) == (0.95, 1.05)
    # Retained as a reported diagnostic only -- a no-roll control moved it by 1e-7.
    assert gate.min_correlation == 0.98


def test_validation_ladder_spans_tlt_and_normalises() -> None:
    ladder = load_model().data.validation_ladder
    # TLT tracks the ICE US Treasury 20+ Year index.
    assert ladder.maturities_years == tuple(float(m) for m in range(20, 31))
    assert ladder.coupon_frequency == 2
    assert len(ladder.weights) == len(ladder.maturities_years)
    assert math.isclose(sum(ladder.normalised_weights), 1.0, abs_tol=1e-12)


def test_spread_structure_controls_are_not_universe_members() -> None:
    """IVV and VOO exist only to isolate the SPY spread residual (rows 61-63).

    They must never reach a factor model, a covariance matrix or the optimizer,
    and the frozen dated universe must not have quietly acquired two members.
    """
    cfg = load()
    controls = cfg.model.data.spread_structure_control

    assert controls == ("IVV", "VOO")
    assert not set(controls) & set(cfg.universe.yfinance_tickers)


# ---------------------------------------------------------------------------
# SPEC.md 7 -- the cost model's keys (W5-P1)
# ---------------------------------------------------------------------------


def test_cost_windows_are_one_year_by_the_projects_own_year() -> None:
    """SPEC.md 7.1: sigma_i and V_i are TRAILING ONE-YEAR quantities.

    One year is data.trading_days_per_year. Pinned as a test rather than a
    parser assertion so a sensitivity run can move the cost windows without
    redefining the calendar -- such a run is a `strategy-config` row.
    """
    model = load_model()
    assert model.costs.volatility_window == model.data.trading_days_per_year == 252
    assert model.costs.adv.window == model.data.trading_days_per_year == 252


def test_cost_constants_that_are_absences_not_values() -> None:
    costs = load_model().costs
    assert costs.commission == 0.0  # no published figure; enters additively
    assert costs.buy_sell_asymmetry == 0.0  # SPEC.md 7.1: 0 unless short-sale asymmetry
    assert costs.total_cost_exponent == 1.5
    assert (costs.square_root_prefactor.patient, costs.square_root_prefactor.urgent) == (0.58, 1.40)


def test_gamma_trade_grid_is_five_half_steps() -> None:
    assert load_model().costs.gamma_trade.points == (1.0, 1.5, 2.0, 2.5, 3.0)


def test_tradable_proxies_cover_exactly_the_five_curve_points() -> None:
    cfg = load()
    proxies = {p.asset: p.ticker for p in cfg.model.costs.tradable_proxies}
    assert proxies == {
        "govt_2y": "SHY",
        "govt_5y": "IEI",
        "govt_10y": "IEF",
        "govt_30y": "TLT",
        "tips_10y": "TIP",
    }
    for proxy in cfg.model.costs.tradable_proxies:
        assert cfg.universe.by_id(proxy.asset).ticker is None
        assert "cost of trading" in proxy.caveat
    assert "HALF" in cfg.model.costs.proxy_for("govt_30y").caveat  # type: ignore[union-attr]


def test_cost_ticker_resolves_etfs_and_proxies_and_refuses_the_dollar_index() -> None:
    cfg = load()
    assert cfg.cost_ticker("us_large_equity") == "SPY"
    assert cfg.cost_ticker("govt_10y") == "IEF"
    with pytest.raises(ConfigError, match="no tradable instrument"):
        cfg.cost_ticker("dollar")
    assert len(cfg.cost_tickers) == 13
    assert cfg.cost_tickers.count("TLT") == 1


def test_capacity_block_holds_rules_and_no_inputs() -> None:
    """SPEC.md 10.4 (W5-P2): alpha_min = 0, alpha_g null, the AUM grid a 0.1%..10% log rule."""
    cap = load_model().costs.capacity
    assert cap.minimum_net_alpha == 0.0
    assert cap.gross_alpha is None
    assert cap.aum_grid.spacing == "log"
    assert cap.aum_grid.low_participation_of_adv == 0.001
    assert cap.aum_grid.high_participation_of_adv == 0.10
    assert cap.aum_grid.points >= 2


def _costs_node() -> dict[str, Any]:
    with (Path(__file__).resolve().parents[1] / "config" / "model.yaml").open() as handle:
        return yaml.safe_load(handle)  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda c: c.update({"total_cost_exponent": 1.0}), "convex"),
        (lambda c: c.update({"commission": -0.0001}), "non-negative"),
        (lambda c: c.update({"volatility_window": 1}), "at least 2"),
        (lambda c: c["gamma_trade"].update({"sweep_step": 0.3}), "whole steps"),
        (lambda c: c["gamma_trade"].update({"sweep_step": 0.0}), "positive"),
        (lambda c: c["spread_level"].update({"shape_baseline": "disclosure_window"}), "holdout"),
        (
            lambda c: c["tradable_proxies"].append(
                {"asset": "govt_2y", "ticker": "SHY", "caveat": "x"}
            ),
            "duplicate",
        ),
        (lambda c: c["tradable_proxies"][0].update({"caveat": "   "}), "caveat"),
        # SPEC.md 10.4's rules (W5-P2): alpha_min only at the spec's special case,
        # alpha_g never a point, the grid a log rule with sane participations.
        (lambda c: c["capacity"].update({"minimum_net_alpha": 0.01}), "sourced"),
        (lambda c: c["capacity"].update({"gross_alpha": 0.03}), "NEVER a point"),
        (lambda c: c["capacity"]["aum_grid"].update({"spacing": "linear"}), "log"),
        (lambda c: c["capacity"]["aum_grid"].update({"points": 1}), "at least 2"),
        (lambda c: c["capacity"]["aum_grid"].update({"low_participation_of_adv": 0.5}), "0 < low"),
        (lambda c: c["capacity"]["aum_grid"].update({"high_participation_of_adv": 1.5}), "ADV"),
    ],
)
def test_cost_parser_fails_loudly(mutate: Any, fragment: str) -> None:
    from mafrm.config import _parse_costs

    root = _costs_node()
    mutate(root["costs"])
    with pytest.raises(ConfigError, match=fragment):
        _parse_costs(root)


# ---------------------------------------------------------------------------
# Backtest (SPEC.md 7.4, 11) -- W5-P3
# ---------------------------------------------------------------------------


def _backtest_node() -> dict[str, Any]:
    import yaml

    from mafrm.config import config_dir

    with (config_dir() / "model.yaml").open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return {"backtest": raw["backtest"]}


def test_backtest_block_holds_the_spec_constant_and_the_comparands_default() -> None:
    backtest = load_model().backtest
    # SPEC.md 11: ">= 50 values of t".
    assert backtest.no_lookahead.minimum_evaluation_dates == 50
    rules = backtest.reconciliation
    # bt.Backtest's own default initial_capital, a normalisation.
    assert rules.initial_nav == 1.0e6
    assert rules.accounting_roundings == 8
    assert rules.synthetic_weights == (0.6, 0.4)
    assert rules.real_strategy == "equal_weight"
    # c = 2 * (assets + roundings): 20 for the two-asset fixture, 42 for the thirteen.
    assert rules.coefficient(2) == 20
    assert rules.coefficient(13) == 42
    import sys

    assert rules.tolerance(assets=13, steps=100) == 42 * sys.float_info.epsilon * 100


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda b: b["no_lookahead"].update({"minimum_evaluation_dates": 49}), "at least 50"),
        (lambda b: b["reconciliation"].update({"initial_nav": 0.0}), "positive NAV"),
        (lambda b: b["reconciliation"].update({"accounting_roundings": 0}), "positive integer"),
        (lambda b: b["reconciliation"].update({"synthetic_weights": [0.7, 0.4]}), "at most 1"),
        (lambda b: b["reconciliation"].update({"synthetic_weights": [1.0]}), "at least two"),
        (lambda b: b["reconciliation"].update({"real_strategy": "cap_weight"}), "equal_weight"),
    ],
)
def test_backtest_parser_fails_loudly(mutate: Any, fragment: str) -> None:
    from mafrm.config import _parse_backtest

    root = _backtest_node()
    mutate(root["backtest"])
    with pytest.raises(ConfigError, match=fragment):
        _parse_backtest(root)
