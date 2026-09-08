"""SPEC.md 15.1's K/T scaling result (W8-P1): the arithmetic, the rulings, and the harness on toys.

Every expected value here is hand-computed or derived; nothing reads the cache.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from mafrm import config
from mafrm.config import ConfigError, load
from mafrm.factors import bias_report, equity_risk, kt_scaling
from mafrm.risk import validation
from mafrm.risk.config import RiskConfig
from mafrm.risk.covariance import separated_ewma_covariance

CFG = load()
KT = CFG.model.validation.kt_scaling

# ---------------------------------------------------------------------------
# The config block transcribes the rulings
# ---------------------------------------------------------------------------


def test_the_rulings_are_in_the_config_and_point_at_existing_keys() -> None:
    assert KT.family4_variant == "psd_repair"
    assert KT.expected_tracking == {"equity": "track", "macro": "not_track"}
    assert KT.factor_comparand_window == "correlation"
    assert KT.naive_comparand_window == "volatility"
    assert (
        config.resolve(CFG.model, KT.interval_level_from) == CFG.model.validation.chi_square_level
    )
    assert config.resolve(CFG.model, KT.subperiod_block_months_from) == 12
    assert tuple(config.resolve(CFG.model, KT.equity_sweep_grid_from)) == (
        21,
        42,
        63,
        84,
        126,
        168,
        252,
        336,
        504,
    )
    reg = KT.registrations
    # (1.3321 - 1.0481) / 0.3321, by hand from reports/bias_statistics.md's C1 table.
    assert reg.row_164_macro_c1_share_of_excess == pytest.approx(
        (1.3321 - 1.0481) / 0.3321, abs=5e-4
    )
    assert reg.row_164_macro_eigen_share_of_excess == 0.0
    tau = CFG.model.validation.second_order.tau_grid
    assert (
        tau.joint_at_shortest_lower,
        tau.joint_at_shortest_hypothesis,
        tau.joint_at_shortest_upper,
    ) == (
        0.045,
        0.065,
        0.085,
    )


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda n: n["expected_tracking"].__setitem__("macro", "maybe"), "expected_tracking.macro"),
        (lambda n: n.__setitem__("factor_comparand_window", "both"), "factor_comparand_window"),
        (lambda n: n.__setitem__("x_axis", "asymptotic"), "x_axis"),
        (
            lambda n: n.__setitem__("interval_level_from", "validation.no_such_key"),
            "interval_level_from",
        ),
        (
            lambda n: n["registrations"].__setitem__("row_164_residual_subset", "all"),
            "row_164_residual_subset",
        ),
    ],
)
def test_the_kt_parser_fails_loudly(mutate: Any, fragment: str) -> None:
    raw = copy.deepcopy(_raw_model())
    mutate(raw["validation"]["kt_scaling"])
    with pytest.raises(ConfigError, match=fragment):
        config._parse_model(raw)


def _raw_model() -> dict[str, Any]:
    with (Path(__file__).resolve().parents[1] / "config" / "model.yaml").open() as handle:
        return yaml.safe_load(handle)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# The arithmetic, by hand
# ---------------------------------------------------------------------------


def test_correlation_kish_by_hand() -> None:
    """Three rows at half-life 1: weights 1/4, 1/2, 1 -> (1.75)^2 / 1.3125 = 2.3333."""
    assert kt_scaling.correlation_kish(3, 1.0) == pytest.approx(1.75**2 / 1.3125)
    # Long window: the asymptotic 2 tau / ln 2.
    assert kt_scaling.correlation_kish(20_000, 504.0) == pytest.approx(
        2 * 504 / math.log(2), rel=1e-6
    )
    with pytest.raises(kt_scaling.KtScalingError):
        kt_scaling.correlation_kish(0, 504.0)


def test_predicted_bias_is_shepard_in_both_units() -> None:
    """K = 6, T = 242: (1 - 0.02479)^-1 = 1.02542 on volatility, 1.05149 on variance (SPEC.md 6.4.1)."""
    risk = kt_scaling.predicted_bias(6, 242.0)
    x = 6 / 242.0
    assert risk.volatility_multiplier == pytest.approx(1.0 / (1.0 - x))
    assert risk.variance_multiplier == pytest.approx(1.0 / (1.0 - x) ** 2)
    assert risk.volatility_multiplier == pytest.approx(1.0254, abs=5e-5)


def _point(
    bias: float, months: int, *, expected: str, comparand_window: str = "correlation"
) -> kt_scaling.Point:
    return kt_scaling.make_point(
        model="toy",
        label="toy",
        halflife=84,
        parameters_per_date=np.full(3, 6.0),
        kish_volatility_per_date=np.array([200.0, 240.0, 242.0]),
        kish_correlation_per_date=np.array([1000.0, 1400.0, 1454.0]),
        bias_monthly=bias,
        months=months,
        bias_daily=bias,
        days=months * 21,
        comparand_window=comparand_window,
        expected=expected,
        level=0.95,
    )


def test_make_point_takes_the_mean_kish_and_the_formulas_own_window() -> None:
    p = _point(1.02, 187, expected="track")
    assert p.kish_volatility == pytest.approx((200 + 240 + 242) / 3)
    assert p.kish_correlation == pytest.approx((1000 + 1400 + 1454) / 3)
    assert p.k_over_t == pytest.approx(6 / p.kish_volatility)
    assert (p.k_over_t_min, p.k_over_t_max) == pytest.approx((6 / 242, 6 / 200))
    assert p.k_over_t_asymptotic == pytest.approx(6 / (2 * 84 / math.log(2)))
    # The comparand is at the CORRELATION window; the upper bound at the volatility one.
    assert p.comparand_t == p.kish_correlation
    assert p.predicted_bias == pytest.approx(1.0 / (1.0 - 6 / p.kish_correlation))
    assert p.upper_bound_bias == pytest.approx(1.0 / (1.0 - 6 / p.kish_volatility))
    assert p.upper_bound_bias > p.predicted_bias
    # The point's own interval: the true bias consistent with B = 1.02 at 187 months,
    # [B / u_T, B / l_T] from (T - 1) B^2 / B_true^2 ~ chi^2_{T-1}.
    null = validation.chi_square_interval(187, level=0.95)
    assert (p.interval_lower, p.interval_upper) == (1.02 / null.upper, 1.02 / null.lower)
    assert p.interval_lower < 1.02 < p.interval_upper
    naive = _point(1.02, 187, expected="", comparand_window="volatility")
    assert naive.comparand_t == naive.kish_volatility
    assert naive.predicted_bias == naive.upper_bound_bias


def test_tracking_is_the_interval_test_and_the_verdict_follows_the_registered_direction() -> None:
    """At 187 months B = 1.02 is consistent with [0.93, 1.14]; a comparand of 1.0047 is inside."""
    inside = _point(1.02, 187, expected="track")
    assert inside.tracks and inside.verdict == "HOLDS"
    assert dataclasses.replace(inside, expected="not_track").verdict == "REFUTED"
    assert dataclasses.replace(inside, expected="").verdict == "reported"
    # B = 1.33 at 187 months: the interval [1.21, 1.48] excludes 1.0047.
    outside = _point(1.33, 187, expected="not_track")
    assert not outside.tracks and outside.verdict == "HOLDS"
    assert dataclasses.replace(outside, expected="track").verdict == "REFUTED"
    with pytest.raises(kt_scaling.KtScalingError):
        _point(1.0, 12, expected="track", comparand_window="both")


# ---------------------------------------------------------------------------
# The sub-period split: non-overlapping, and the overlap is stated
# ---------------------------------------------------------------------------


def test_subperiod_split_uses_non_overlapping_blocks_and_recovers_a_planted_bias() -> None:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2015-01-02", periods=24 * 21)
    forecast = np.full(len(dates), 2.0)
    realised = rng.standard_normal(len(dates)) * 2.0
    # Plant B = 1.5 on the second year.
    realised[252:] *= 1.5
    kt = np.linspace(0.05, 0.01, len(dates))
    split = kt_scaling.subperiod_split(
        model="toy",
        dates=dates,
        forecast=forecast,
        realised=realised,
        k_over_t=kt,
        block_months=12,
        clip=4.0,
        level=0.95,
    )
    assert split.overlap == 0
    assert split.blocks == 2
    table = split.table
    assert table["T"].sum() == len(dates)
    assert table.loc[0, "end"] < table.loc[1, "start"]
    b0, b1 = table["B"].to_numpy()
    assert 0.85 < b0 < 1.15 and 1.3 < b1 < 1.7
    assert table.loc[0, "k_over_t_mean"] > table.loc[1, "k_over_t_mean"]
    assert math.isnan(split.rank_correlation)  # fewer than three blocks: no correlation quoted
    with pytest.raises(kt_scaling.KtScalingError):
        kt_scaling.subperiod_split(
            model="toy",
            dates=dates,
            forecast=forecast[:-1],
            realised=realised,
            k_over_t=kt,
            block_months=12,
            clip=4.0,
            level=0.95,
        )


# ---------------------------------------------------------------------------
# The equity sweep on a toy panel
# ---------------------------------------------------------------------------

_DATES = pd.bdate_range("2019-01-01", periods=420)
_STYLES = ("BETA", "MOMENTUM")


def _toy_panel(seed: int = 5, *, names: int = 12) -> equity_risk.Panel:
    rng = np.random.default_rng(seed)
    dates = _DATES
    t = len(dates)
    tickers = [f"N{i:02d}" for i in range(names)]
    industry_ids = np.array([1 + (i % 2) for i in range(names)])
    labels = {1: "Ind1", 2: "Ind2"}
    industry_columns = ["Ind1", "Ind2"]
    f = pd.DataFrame(rng.standard_normal((t, 1 + 2 + len(_STYLES))) * 0.01, index=dates)
    f.columns = ["COUNTRY", *industry_columns, *_STYLES]
    frame = equity_risk.factor_frame(
        f,
        industry_columns=industry_columns,
        style_columns=_STYLES,
        country_column="COUNTRY",
        scale=1e4,
    )
    cap = pd.DataFrame(np.exp(rng.standard_normal((t, names))) * 1e9, index=dates, columns=tickers)
    styles = {
        s: pd.DataFrame(rng.standard_normal((t, names)), index=dates, columns=tickers)
        for s in _STYLES
    }
    residuals = pd.DataFrame(rng.standard_normal((t, names)) * 80.0, index=dates, columns=tickers)
    returns = residuals + rng.standard_normal((t, names)) * 50.0
    industry = pd.DataFrame(np.tile(industry_ids, (t, 1)), index=dates, columns=tickers)
    summary = pd.DataFrame({"cond_style": np.full(t, 2.5)}, index=dates)
    return equity_risk.Panel(
        factors=frame,
        returns=returns,
        residuals=residuals,
        cap=cap,
        styles=styles,
        industry=industry,
        industry_label=labels,
        summary=summary,
        always_present=tuple(tickers),
        sample_start=dates[0],
        holdout_start=dates[-1] + pd.Timedelta(days=1),
    )


def test_equity_sweep_moves_only_the_volatility_halflife_and_reproduces_score_variant() -> None:
    data = _toy_panel()
    sweep = kt_scaling.equity_sweep(data, CFG, halflives=(21, 84), progress=False)
    assert [p.halflife for p in sweep.points] == [21, 84]
    p21, p84 = sweep.points
    # Same scored window for both points, and every point pools the same months.
    assert p21.factor.months == p84.factor.months == len(sweep.scored_dates)
    assert p21.dates.equals(p84.dates)
    # The comparand sits at the (pinned) correlation window: identical across the two points.
    assert p21.factor.kish_correlation == pytest.approx(p84.factor.kish_correlation)
    assert p21.factor.predicted_bias == pytest.approx(p84.factor.predicted_bias)
    # The volatility-window Kish is smaller at the shorter half-life, so K/T is larger.
    assert p21.factor.k_over_t > p84.factor.k_over_t
    assert p21.factor.upper_bound_bias > p84.factor.upper_bound_bias
    # The forecast at tau = 84 is score_variant's, hand-rebuilt at the first scored month-end.
    short = RiskConfig.load(horizon="short", config=CFG)
    first = int(data.dates.get_indexer([sweep.scored_dates[0]])[0])
    columns = data.factors.present_columns(first)
    from mafrm.risk.covariance import run_pipeline

    build = run_pipeline(
        data.factors.frame.iloc[: first + 1, columns], short, stop_after="psd_repair"
    )
    design = equity_risk.design_at(data, first, buckets=10)
    specific = equity_risk.specific_at(short, data.residuals.to_numpy(dtype=float), design)
    f_matrix = equity_risk.embed(build.matrix, columns, data.factors.factors)
    sigma = design.exposures @ f_matrix @ design.exposures.T + np.diag(specific.variance)
    w = np.linalg.solve(sigma, np.ones(design.assets))
    w /= w.sum()
    assert p84.scored.forecasts["min_var"].iloc[0] == pytest.approx(math.sqrt(w @ sigma @ w))
    np.testing.assert_allclose(p84.scored.min_var_weights[0], w, rtol=1e-10)
    # K_d and the Kish size per session line up with the sessions.
    assert p84.k_over_t_per_date.size == len(p84.dates)
    assert p84.factor.parameters == pytest.approx(data.factors.factors)


def test_held_months_are_the_grid_gaps_and_refuse_a_last_position() -> None:
    grid = np.array([10, 30, 51, 72])
    assert kt_scaling.held_months(grid, [10, 30]) == [(11, 31), (31, 52)]
    with pytest.raises(kt_scaling.KtScalingError):
        kt_scaling.held_months(grid, [72])
    with pytest.raises(kt_scaling.KtScalingError):
        kt_scaling.held_months(grid, [11])


def test_equity_control_keeps_the_diagonal_and_moves_only_the_off_diagonal() -> None:
    """C1's covariance is D R_u D + X F X': its diagonal equals the diagonal variant's to the bit."""
    data = _toy_panel()
    sweep = kt_scaling.equity_sweep(data, CFG, halflives=(84,), progress=False)
    scored_positions = [int(p) for p in data.dates.get_indexer(sweep.scored_dates)]
    k = data.factors.factors
    eigen = [np.eye(k) * 4.0 for _ in scored_positions]
    control = kt_scaling.equity_control(
        data, CFG, sweep, eigen=eigen, eigen_positions=scored_positions, progress=False
    )
    assert control.months == len(scored_positions)
    assert control.subset_size == len(data.always_present)
    assert control.half_width > 0.0
    # With every name always present and a complete residual panel, R_u is full rank.
    assert control.min_residual_eigenvalue > 0.0
    # The share arithmetic, by hand.
    expected = (control.diagonal_family_4 - control.correlated_family_4) / (
        control.diagonal_family_4 - 1.0
    )
    assert control.c1_share_of_excess == pytest.approx(expected)
    assert control.half_width_share == pytest.approx(
        control.half_width / (control.diagonal_family_4 - 1.0)
    )
    with pytest.raises(kt_scaling.KtScalingError):
        kt_scaling.equity_control(
            data,
            CFG,
            sweep,
            eigen=eigen[:-1],
            eigen_positions=scored_positions[:-1],
            progress=False,
        )


def test_the_psd_repair_variant_is_the_charts_variant() -> None:
    names = [s.name for s in bias_report.variants(CFG)]
    assert KT.family4_variant in names
    assert names.index(KT.family4_variant) < names.index("eigen_a1")  # pre-eigenfactor


# ---------------------------------------------------------------------------
# Rows 327-333: the estimator's own second-order comparand (W8-P1b)
# ---------------------------------------------------------------------------


def test_the_simulated_comparand_is_the_closed_forms_own_identity() -> None:
    """`B = sqrt(E[realised variance / forecast variance])` -- the square root, hand-checked.

    The whole justification for overlaying a simulated curve is that it reaches
    a `B` by the SAME identity Shepard's multiplier does. If that stopped being
    true the third curve would be a different quantity plotted on one axis.
    """
    data = _toy_panel()
    points = kt_scaling.equity_second_order(data, CFG, halflives=(84, 252), progress=False)
    assert [p.halflife for p in points] == [84, 252]
    # The seed VARIES across grid points: two points sharing one seed are one measurement.
    assert points[0].seed != points[1].seed
    assert points[0].seed == CFG.model.seed
    for point in points:
        joint = point.result.joint
        assert point.predicted_bias == pytest.approx(math.sqrt(joint.mean_ratio))
        assert point.predicted_bias == pytest.approx(math.sqrt(1.0 + joint.excess))
        # Delta method on the square root: se(B) = se(m) / (2 sqrt(m)).
        assert point.standard_error == pytest.approx(
            joint.standard_error / (2.0 * math.sqrt(joint.mean_ratio))
        )
        assert point.correlation_leg == joint_leg(point, "correlation_only")
        assert point.volatility_leg == joint_leg(point, "volatility_only")


def joint_leg(point: kt_scaling.SimulatedComparand, name: str) -> float:
    return float(getattr(point.result, name).excess)


def test_the_simulated_comparand_holds_the_correlation_window_and_moves_the_volatility_one() -> (
    None
):
    """Only the estimator's volatility window sweeps; `tau_rho` is pinned, so the rho leg is flat."""
    data = _toy_panel()
    short = RiskConfig.load(horizon="short", config=CFG)
    points = kt_scaling.equity_second_order(data, CFG, halflives=(21, 504), progress=False)
    for point in points:
        assert point.result.correlation_halflife == float(short.correlation_halflife)
        assert point.result.volatility_halflife == float(point.halflife)
    # The sigma leg falls with a longer window; the rho leg is tau_sigma-independent by
    # construction and agrees between the two points to within their Monte Carlo error.
    short_tau, long_tau = points
    assert short_tau.volatility_leg > long_tau.volatility_leg
    gap = abs(short_tau.correlation_leg - long_tau.correlation_leg)
    error = 2.0 * (
        short_tau.result.correlation_only.standard_error
        + long_tau.result.correlation_only.standard_error
    )
    assert gap <= error


def test_the_simulated_truth_is_one_matrix_at_the_shipped_half_lives() -> None:
    """One population, swept estimator -- the rows 318-326 design, asserted rather than assumed."""
    data = _toy_panel()
    short = RiskConfig.load(horizon="short", config=CFG)
    frame = data.factors.frame.to_numpy(dtype=float)
    expected = separated_ewma_covariance(
        frame,
        volatility_halflife=float(short.volatility_halflife),
        correlation_halflife=float(short.correlation_halflife),
    )
    # Both grid points draw from the SAME truth: their `parameters` and the truth's rank agree,
    # and a point built at tau = 21 sees the same K as one built at 504.
    points = kt_scaling.equity_second_order(data, CFG, halflives=(21, 504), progress=False)
    assert {p.result.parameters for p in points} == {expected.shape[0]}
    assert {p.result.observations for p in points} == {frame.shape[0]}
