"""SPEC.md 15.2 and 15.7.1 on synthetic panels: the wrapper in ``factors/`` and the unchanged pipeline.

Every expected value here is hand-computed or derived; nothing reads the cache.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from mafrm.config import ConfigError, load
from mafrm.factors import bias_report, equity_risk
from mafrm.risk import specific, validation
from mafrm.risk.config import RiskConfig
from mafrm.risk.covariance import ewma_second_moment, run_pipeline
from mafrm.risk.regime import regime_multiplier

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DATES = pd.bdate_range("2020-01-01", periods=130)
_STYLES = ("BETA", "MOMENTUM")


def _toy_panel(
    seed: int = 3, *, names: int = 14, gaps: bool = True, singleton: bool = False
) -> equity_risk.Panel:
    """A small ragged panel with two industries, two styles, and optional gaps."""
    rng = np.random.default_rng(seed)
    dates = _DATES
    t = len(dates)
    tickers = [f"N{i:02d}" for i in range(names)]
    industry_ids = np.array([1 + (i % 2) for i in range(names)])
    if singleton:
        industry_ids[-1] = 3  # one name alone in industry 3
    labels = {1: "Ind1", 2: "Ind2", 3: "Ind3"}
    industry_columns = ["Ind1", "Ind2", "Ind3", "Never"]
    f = pd.DataFrame(rng.standard_normal((t, 1 + 4 + len(_STYLES))) * 0.01, index=dates)
    f.columns = ["COUNTRY", *industry_columns, *_STYLES]
    f["Never"] = np.nan
    if not singleton:
        f["Ind3"] = np.nan
    else:
        f.loc[dates[:30], "Ind3"] = np.nan  # Ind3 enters K at date 30
    f.loc[dates[:10], "Ind2"] = np.nan  # intermittent
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
    if gaps:
        residuals.iloc[20:40, 0] = np.nan  # a gap inside N00's history
        returns.iloc[t - 5 :, 1] = np.nan  # N01 has no bar over the last week
        cap.iloc[: t // 2, 2] = np.nan  # N02 joins half-way
    summary = pd.DataFrame({"cond_style": np.full(t, 2.5)}, index=dates)
    always = tuple(n for n in tickers if np.isfinite(returns[n].to_numpy()).all())
    return equity_risk.Panel(
        factors=frame,
        returns=returns,
        residuals=residuals,
        cap=cap,
        styles=styles,
        industry=industry,
        industry_label={i: labels[i] for i in labels if labels[i] in frame.industry_columns},
        summary=summary,
        always_present=always,
        sample_start=dates[0],
        holdout_start=dates[-1] + pd.Timedelta(days=1),
    )


@pytest.fixture(scope="module")
def risk() -> RiskConfig:
    return RiskConfig.load(horizon="short")


# ---------------------------------------------------------------------------
# Ruling 1 -- the frame
# ---------------------------------------------------------------------------


def test_factor_frame_drops_never_present_zero_fills_intermittent_and_scales() -> None:
    dates = pd.bdate_range("2021-01-01", periods=8)
    raw = pd.DataFrame(
        {
            "COUNTRY": 0.01,
            "A": [0.02, np.nan, 0.02, np.nan, 0.02, 0.02, 0.02, 0.02],
            "B": np.nan,
            "S": 0.005,
        },
        index=dates,
    )
    frame = equity_risk.factor_frame(
        raw, industry_columns=["A", "B"], style_columns=["S"], country_column="COUNTRY", scale=1e4
    )
    assert frame.dropped == ("B",)
    assert frame.factors == 3
    assert list(frame.frame.columns) == ["COUNTRY", "A", "S"]
    assert frame.presence["A"] == pytest.approx(6 / 8)
    assert frame.understatement["A"] == pytest.approx(2 / 8)
    assert frame.intermittent == ("A",)
    assert frame.frame.loc[dates[1], "A"] == 0.0
    assert frame.frame.loc[dates[0], "A"] == pytest.approx(200.0)
    assert frame.frame.loc[dates[0], "COUNTRY"] == pytest.approx(100.0)
    assert int(frame.first_present["A"]) == 0
    assert list(frame.present_columns(0)) == [0, 1, 2]


def test_an_industry_enters_k_at_its_first_appearance_point_in_time() -> None:
    dates = pd.bdate_range("2021-01-01", periods=8)
    raw = pd.DataFrame(
        {
            "COUNTRY": 0.01,
            "A": 0.02,
            "L": [np.nan, np.nan, np.nan, 0.03, np.nan, 0.03, 0.03, 0.03],
            "S": 0.005,
        },
        index=dates,
    )
    frame = equity_risk.factor_frame(
        raw, industry_columns=["A", "L"], style_columns=["S"], country_column="COUNTRY", scale=1.0
    )
    assert frame.factors == 4
    assert int(frame.first_present["L"]) == 3
    assert list(frame.present_columns(2)) == [0, 1, 3]
    assert list(frame.present_columns(3)) == [0, 1, 2, 3]
    assert frame.factors_at(2) == 3 and frame.factors_at(7) == 4


def test_embed_and_present_block_round_trip() -> None:
    block = np.array([[2.0, 0.5], [0.5, 3.0]])
    full = equity_risk.embed(block, np.array([0, 2]), 3)
    assert full[1].sum() == 0.0 and full[:, 1].sum() == 0.0
    assert list(equity_risk.present_block(full)) == [0, 2]
    np.testing.assert_array_equal(full[np.ix_([0, 2], [0, 2])], block)


def test_factor_frame_refuses_a_missing_style_return() -> None:
    dates = pd.bdate_range("2021-01-01", periods=4)
    raw = pd.DataFrame({"COUNTRY": 0.01, "A": 0.02, "S": [0.1, np.nan, 0.1, 0.1]}, index=dates)
    with pytest.raises(equity_risk.EquityRiskError, match="style"):
        equity_risk.factor_frame(
            raw, industry_columns=["A"], style_columns=["S"], country_column="COUNTRY", scale=1.0
        )


def test_the_zero_fill_understatement_is_exactly_one_minus_f_in_the_equal_weight_limit() -> None:
    """``1 - f_k`` is the reported bias: with equal weights the zero-filled second moment is ``f``
    times the moment on the present rows, so the variance is understated by exactly ``1 - f``."""
    rng = np.random.default_rng(0)
    values = rng.standard_normal(400)
    present = np.ones(400, dtype=bool)
    present[::4] = False
    filled = np.where(present, values, 0.0)[:, None]
    observed = values[present][:, None]
    zero_filled = ewma_second_moment(filled, halflife=1e12)[0, 0]
    on_present = ewma_second_moment(observed, halflife=1e12)[0, 0]
    f = present.mean()
    assert zero_filled / on_present == pytest.approx(f, rel=1e-9)
    assert 1.0 - zero_filled / on_present == pytest.approx(1.0 - f, rel=1e-9)


# ---------------------------------------------------------------------------
# Rulings 2 and 4 -- deciles and the grid
# ---------------------------------------------------------------------------


def test_month_end_positions_pick_the_last_session_of_each_month() -> None:
    dates = pd.bdate_range("2021-01-25", "2021-03-05")
    positions = equity_risk.month_end_positions(dates)
    assert [dates[p].date().isoformat() for p in positions] == [
        "2021-01-29",
        "2021-02-26",
        "2021-03-05",
    ]


def test_size_deciles_are_equal_count_and_monotone_in_cap() -> None:
    cap = np.arange(1, 21, dtype=float)[::-1]  # 20 names, descending cap
    labels = equity_risk.size_deciles(cap, buckets=10)
    counts = pd.Series(labels).value_counts()
    assert set(counts) == {2}
    assert labels[-1] == "size_0" and labels[0] == "size_9"  # smallest cap -> bucket 0


def test_design_at_builds_the_country_industry_style_design_and_flags_singletons() -> None:
    data = _toy_panel(singleton=True, gaps=False)
    design = equity_risk.design_at(data, 5, buckets=10)
    assert design.assets == 14
    assert design.exposures.shape == (14, data.factors.factors)
    assert np.all(design.exposures[:, 0] == 1.0)
    industries = design.exposures[:, 1 : 1 + len(data.factors.industry_columns)]
    assert np.all(industries.sum(axis=1) == 1.0)
    assert design.singleton.sum() == 1 and design.singleton[-1]
    assert len(set(design.buckets)) == 10
    # At position 5 neither Ind2 (enters at 10; six names) nor Ind3 (enters at 30; the
    # singleton) is a factor yet: seven names carry an unentered industry.
    assert design.unentered.sum() == 7
    middle = equity_risk.design_at(data, 20, buckets=10)
    assert middle.unentered.sum() == 1 and middle.unentered[-1]
    later = equity_risk.design_at(data, 40, buckets=10)
    assert later.unentered.sum() == 0


def test_design_at_excludes_a_name_without_a_cap() -> None:
    data = _toy_panel()
    early = equity_risk.design_at(data, 5, buckets=10)
    late = equity_risk.design_at(data, 120, buckets=10)
    assert "N02" not in early.names and "N02" in late.names


# ---------------------------------------------------------------------------
# Ruling 3 -- per-name specific risk
# ---------------------------------------------------------------------------


def test_specific_at_reproduces_build_specific_risk_on_a_complete_panel(risk: RiskConfig) -> None:
    """With no gaps, no singletons and no short history, the per-name composition must equal
    ``risk/``'s own -- the same four legs, the same numbers."""
    data = _toy_panel(gaps=False)
    position = 100
    design = equity_risk.design_at(data, position, buckets=10)
    ours = equity_risk.specific_at(risk, data.residuals.to_numpy(dtype=float), design)
    window = data.residuals.iloc[: position + 1][list(design.names)]
    exposures = pd.DataFrame(design.exposures, index=list(design.names))
    buckets = pd.Series(list(design.buckets), index=list(design.names))
    theirs = specific.build_specific_risk(window, exposures, buckets, risk)
    np.testing.assert_allclose(ours.time_series, theirs.time_series.deviation, rtol=1e-12)
    np.testing.assert_allclose(ours.deviation, theirs.shrinkage.deviation, rtol=1e-12)
    assert not ours.short_history.any()


def test_specific_at_removes_gaps_rather_than_filling_them(risk: RiskConfig) -> None:
    data = _toy_panel(gaps=True)
    position = 100
    design = equity_risk.design_at(data, position, buckets=10)
    ours = equity_risk.specific_at(risk, data.residuals.to_numpy(dtype=float), design)
    column = data.residuals["N00"].to_numpy(dtype=float)[: position + 1]
    observed = column[np.isfinite(column)]
    assert observed.size == position + 1 - 20
    expected = specific.time_series_estimate(
        observed[:, None],
        halflife=float(risk.specific_volatility_halflife),
        autocorrelation_halflife=float(risk.specific_autocorrelation_halflife),
        lags=risk.specific_newey_west_lags,
    ).deviation[0]
    n = design.names.index("N00")
    assert ours.time_series[n] == pytest.approx(expected, rel=1e-12)
    assert ours.observations[n] == observed.size


def test_a_name_below_the_arithmetic_floor_takes_the_structural_estimate(risk: RiskConfig) -> None:
    data = _toy_panel(gaps=False)
    residuals = data.residuals.to_numpy(dtype=float).copy()
    residuals[:-3, 4] = np.nan  # N04 has three observed residuals: below lags + 1
    design = equity_risk.design_at(data, 120, buckets=10)
    ours = equity_risk.specific_at(risk, residuals, design)
    n = design.names.index("N04")
    assert ours.short_history[n]
    assert np.isnan(ours.time_series[n])
    assert ours.deviation[n] > 0.0
    # Leg (b) is fitted on the other names and applied to this one; shrinkage then acts.
    assert ours.structural[n] > 0.0


def test_singletons_are_held_out_of_the_target_but_still_shrunk(risk: RiskConfig) -> None:
    data = _toy_panel(gaps=False, singleton=True)
    residuals = data.residuals.to_numpy(dtype=float).copy()
    residuals[:, -1] = 0.0  # the singleton's residual is zero by construction
    # Two buckets, so the singleton has bucket-mates whose mean is the target it is shrunk toward.
    design = equity_risk.design_at(data, 120, buckets=2)
    ours = equity_risk.specific_at(risk, residuals, design)
    assert design.singleton[-1]
    assert ours.exact_zero[-1] and ours.short_history[-1]
    # Its value did not build its bucket's target: the target equals the mean of the others.
    others = ~design.singleton
    stats = specific.bucket_statistics(
        np.where(ours.short_history, ours.structural, ours.time_series),
        design.buckets,
        target_set=others,
    )
    label = design.buckets[-1]
    mates = [i for i in range(design.assets) if design.buckets[i] == label and others[i]]
    expected = np.mean(np.where(ours.short_history, ours.structural, ours.time_series)[mates])
    assert stats.mean[-1] == pytest.approx(expected)


def test_a_singleton_takes_the_structural_estimate_under_the_w7_p4b_rule(
    risk: RiskConfig,
) -> None:
    """SPEC.md 15.7.3: no information, so leg (b) -- and its own sigma_TS is still carried."""
    data = _toy_panel(gaps=False, singleton=True)
    residuals = data.residuals.to_numpy(dtype=float)
    design = equity_risk.design_at(data, 120, buckets=2)
    before = equity_risk.specific_at(risk, residuals, design)
    after = equity_risk.specific_at(risk, residuals, design, singleton_structural=True)
    n = design.assets - 1
    assert design.singleton[n]
    assert not before.short_history[n] and after.short_history[n]
    assert np.isfinite(after.time_series[n]) and after.time_series[n] == before.time_series[n]
    # Leg (b) is fitted without the singleton and applied to it; nobody else changes leg.
    assert after.structural[n] > 0.0
    assert np.array_equal(after.short_history[:n], before.short_history[:n])


# ---------------------------------------------------------------------------
# The cache columns and the packing
# ---------------------------------------------------------------------------


def test_pack_unpack_round_trip_and_column_count() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal((5, 5))
    a = a @ a.T
    assert np.array_equal(equity_risk.unpack(equity_risk.pack(a), 5), a)
    assert len(equity_risk.history_columns(5, (1.0, 1.4))) == 2 * 15 + 5 + 5 + 3


# ---------------------------------------------------------------------------
# Construction 6 -- the multipliers, lagged by construction
# ---------------------------------------------------------------------------


def test_regime_multipliers_are_the_ewma_of_daily_b_and_lag_by_one_forecast(
    risk: RiskConfig,
) -> None:
    data = _toy_panel(gaps=False)
    positions = [60, 80, 100]
    designs = [equity_risk.design_at(data, p, buckets=10) for p in positions]
    residuals = data.residuals.to_numpy(dtype=float)
    specifics = [equity_risk.specific_at(risk, residuals, d) for d in designs]
    k = data.factors.factors
    factor = [np.eye(k) * 4.0 for _ in positions]  # sigma_k = 2 everywhere, every factor present
    lambda_f, lambda_s = equity_risk.regime_multipliers(risk, data, positions, factor, specifics)
    assert np.isnan(lambda_f[0]) and np.isnan(lambda_s[0])
    f = data.factors.frame.to_numpy(dtype=float)
    daily = np.sqrt(np.mean(np.square(f[61:81] / 2.0), axis=1))
    expected = regime_multiplier(daily, halflife=risk.volatility_regime_halflife).lambda_squared
    assert lambda_f[1] == pytest.approx(expected)
    daily_2 = np.sqrt(np.mean(np.square(f[61:101] / 2.0), axis=1))
    assert lambda_f[2] == pytest.approx(
        regime_multiplier(daily_2, halflife=risk.volatility_regime_halflife).lambda_squared
    )


# ---------------------------------------------------------------------------
# The families, daily-held, and the leak test
# ---------------------------------------------------------------------------


def _parts(data: equity_risk.Panel, risk: RiskConfig, positions: list[int]) -> dict[str, object]:
    designs = [equity_risk.design_at(data, p, buckets=10) for p in positions]
    residuals = data.residuals.to_numpy(dtype=float)
    specifics = [equity_risk.specific_at(risk, residuals, d) for d in designs]
    k = data.factors.factors
    stages = []
    eigen = []
    for p in positions:
        columns = data.factors.present_columns(p)
        build = run_pipeline(
            data.factors.frame.iloc[: p + 1, columns], risk, stop_after="psd_repair"
        )
        stages.append({s.name: equity_risk.embed(s.matrix, columns, k) for s in build.stages})
        eigen.append(
            {
                "a1": equity_risk.embed(build.matrix * 1.1, columns, k),
                "a1.4": equity_risk.embed(build.matrix * 1.2, columns, k),
            }
        )
    lambda_f = {"a1": np.array([np.nan, 1.05, 1.05]), "a1.4": np.array([np.nan, 1.02, 1.02])}
    lambda_s = np.array([np.nan, 0.9, 0.9])
    return {
        "designs": designs,
        "specifics": specifics,
        "stages": stages,
        "eigen": eigen,
        "lambda_f": lambda_f,
        "lambda_s": lambda_s,
        "k": k,
    }


def _psd_repair_spec() -> bias_report.VariantSpec:
    return next(s for s in bias_report.variants(load()) if s.name == "psd_repair")


def test_score_variant_matches_a_hand_built_min_var_book(risk: RiskConfig) -> None:
    data = _toy_panel(gaps=False)
    positions = [60, 80, 100]
    parts = _parts(data, risk, positions)
    scored = equity_risk.score_variant(
        _psd_repair_spec(),
        data=data,
        positions=positions,
        scored=[1],
        designs=parts["designs"],  # type: ignore[arg-type]
        stages=parts["stages"],  # type: ignore[arg-type]
        eigen=parts["eigen"],  # type: ignore[arg-type]
        specifics=parts["specifics"],  # type: ignore[arg-type]
        lambda_f=parts["lambda_f"],  # type: ignore[arg-type]
        lambda_s=parts["lambda_s"],  # type: ignore[arg-type]
        subset=data.always_present,
        random_portfolios=5,
        seed=7,
        volatility_halflife=float(risk.volatility_halflife),
    )
    design = parts["designs"][1]  # type: ignore[index]
    f = parts["stages"][1]["psd_repair"]  # type: ignore[index]
    delta = parts["specifics"][1].variance  # type: ignore[index]
    sigma = design.exposures @ f @ design.exposures.T + np.diag(delta)
    w = np.linalg.solve(sigma, np.ones(design.assets))
    w /= w.sum()
    np.testing.assert_allclose(scored.min_var_weights[0], w, rtol=1e-10)
    forecast = np.sqrt(w @ sigma @ w)
    held = data.returns.to_numpy(dtype=float)[81:101][:, design.columns]
    assert scored.forecasts["min_var"].to_numpy()[0] == pytest.approx(forecast)
    np.testing.assert_allclose(scored.realised["min_var"].to_numpy(), held @ w, rtol=1e-12)
    assert len(scored.forecasts) == 20
    assert scored.liquidated_cells == 0
    # Family 1: a name's own forecast is sqrt(Sigma_nn) and its realised return is its own.
    n = 3
    assert scored.forecasts[design.names[n]].iloc[0] == pytest.approx(np.sqrt(sigma[n, n]))
    assert scored.realised[design.names[n]].iloc[4] == pytest.approx(held[4, n])


def test_a_missing_return_in_the_held_month_is_zero_and_counted(risk: RiskConfig) -> None:
    data = _toy_panel(gaps=True)
    positions = [100, 115, 129]
    parts = _parts(data, risk, positions)
    scored = equity_risk.score_variant(
        _psd_repair_spec(),
        data=data,
        positions=positions,
        scored=[1],
        designs=parts["designs"],  # type: ignore[arg-type]
        stages=parts["stages"],  # type: ignore[arg-type]
        eigen=parts["eigen"],  # type: ignore[arg-type]
        specifics=parts["specifics"],  # type: ignore[arg-type]
        lambda_f=parts["lambda_f"],  # type: ignore[arg-type]
        lambda_s=parts["lambda_s"],  # type: ignore[arg-type]
        subset=data.always_present,
        random_portfolios=5,
        seed=7,
        volatility_halflife=float(risk.volatility_halflife),
    )
    assert scored.liquidated_cells == 5
    assert scored.realised["N01"].iloc[-1] == 0.0


def test_no_realised_return_in_the_held_month_reaches_its_own_forecast(risk: RiskConfig) -> None:
    """The leak test. Perturb a return inside the held month; no forecast may move."""
    base = _toy_panel(gaps=False)
    perturbed = dataclasses.replace(base, returns=base.returns.copy())
    perturbed.returns.iloc[90, 5] += 500.0
    positions = [60, 80, 100]
    out = []
    for data in (base, perturbed):
        parts = _parts(data, risk, positions)
        out.append(
            equity_risk.score_variant(
                _psd_repair_spec(),
                data=data,
                positions=positions,
                scored=[1],
                designs=parts["designs"],  # type: ignore[arg-type]
                stages=parts["stages"],  # type: ignore[arg-type]
                eigen=parts["eigen"],  # type: ignore[arg-type]
                specifics=parts["specifics"],  # type: ignore[arg-type]
                lambda_f=parts["lambda_f"],  # type: ignore[arg-type]
                lambda_s=parts["lambda_s"],  # type: ignore[arg-type]
                subset=data.always_present,
                random_portfolios=5,
                seed=7,
                volatility_halflife=float(risk.volatility_halflife),
            )
        )
    pd.testing.assert_frame_equal(out[0].forecasts, out[1].forecasts)
    assert not out[0].realised.equals(out[1].realised)


def test_to_monthly_sums_returns_and_scales_the_forecast_by_root_days(risk: RiskConfig) -> None:
    data = _toy_panel(gaps=False)
    positions = [60, 80, 100]
    parts = _parts(data, risk, positions)
    scored = equity_risk.score_variant(
        _psd_repair_spec(),
        data=data,
        positions=positions,
        scored=[1],
        designs=parts["designs"],  # type: ignore[arg-type]
        stages=parts["stages"],  # type: ignore[arg-type]
        eigen=parts["eigen"],  # type: ignore[arg-type]
        specifics=parts["specifics"],  # type: ignore[arg-type]
        lambda_f=parts["lambda_f"],  # type: ignore[arg-type]
        lambda_s=parts["lambda_s"],  # type: ignore[arg-type]
        subset=data.always_present,
        random_portfolios=5,
        seed=7,
        volatility_halflife=float(risk.volatility_halflife),
    )
    f, r = equity_risk.to_monthly(scored)
    assert len(f) == 1
    assert f["min_var"].iloc[0] == pytest.approx(scored.forecasts["min_var"].iloc[0] * np.sqrt(20))
    assert r["min_var"].iloc[0] == pytest.approx(scored.realised["min_var"].sum())


def test_the_component_split_closes_the_return_identity(risk: RiskConfig) -> None:
    data = _toy_panel(gaps=False)
    positions = [60, 80, 100]
    parts = _parts(data, risk, positions)
    designs = parts["designs"]
    w = np.full(designs[1].assets, 1.0 / designs[1].assets)  # type: ignore[index]
    split = equity_risk.component_split(
        data=data,
        positions=positions,
        scored=[1],
        designs=designs,  # type: ignore[arg-type]
        weights=[w],
        factor=[np.eye(parts["k"]) for _ in positions],  # type: ignore[arg-type]
        specific_variance=[s.variance for s in parts["specifics"]],  # type: ignore[union-attr]
    )
    row = split.periods.iloc[0]
    assert row["factor_return"] + row["specific_return"] == pytest.approx(row["portfolio_return"])
    assert row["days"] == 20
    assert row["ex_ante_specific_variance"] == pytest.approx(
        20 * float(np.sum(w**2 * parts["specifics"][1].variance))  # type: ignore[index]
    )


# ---------------------------------------------------------------------------
# SPEC.md 15.2 at K = 53 on the unchanged pipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scaling", [1.0, 1.4])
def test_the_unchanged_pipeline_runs_at_k53_and_is_psd_at_every_stage(scaling: float) -> None:
    """The equity K, on synthetic data, through every pre-VRA stage. An architecture test."""
    rng = np.random.default_rng(load().model.seed)
    k, t = 53, 600
    loadings = rng.standard_normal((k, 3))
    frame = pd.DataFrame(
        rng.standard_normal((t, 3)) @ loadings.T * 30 + rng.standard_normal((t, k)) * 20,
        index=pd.bdate_range("2010-01-01", periods=t),
        columns=[f"f{i}" for i in range(k)],
    )
    risk = dataclasses.replace(
        RiskConfig.load(horizon="short").for_scaling(scaling), eigenfactor_monte_carlo_trials=60
    )
    build = run_pipeline(frame, risk, stop_after="eigenfactor")
    assert [s.name for s in build.stages] == ["ewma", "newey_west", "psd_repair", "eigenfactor"]
    for stage in build.stages:
        assert stage.matrix.shape == (k, k)
        if stage.name != "newey_west":
            assert stage.minimum_eigenvalue > -1e-12
    assert build.eigenfactor is not None and build.eigenfactor.factors == k


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_the_equity_risk_readings_and_registrations_are_in_the_config() -> None:
    settings = load().model.equity_risk
    assert settings.size_buckets == 10
    assert settings.forecast_grid == "month_end"
    assert settings.registrations.row_291_factors == 53
    assert (
        settings.registrations.row_291_family4_pre_eigen_min
        < settings.registrations.row_291_family4_pre_eigen_max
    )


def test_an_unknown_reading_is_refused(tmp_path: object) -> None:
    from mafrm import config as config_mod

    raw = config_mod._read_yaml(config_mod.config_dir() / "model.yaml")  # type: ignore[attr-defined]
    raw["equity_risk"]["absent_industry_return"] = "pairwise"
    with pytest.raises(ConfigError, match="absent_industry_return"):
        config_mod._parse_equity_risk(raw)  # type: ignore[attr-defined]


def test_scaling_tag_is_the_cached_column_name() -> None:
    assert bias_report.scaling_tag(1.0) == "a1" and bias_report.scaling_tag(1.4) == "a1.4"


def test_validate_sees_the_subset_family_as_an_ordinary_family() -> None:
    portfolios = validation.PortfolioSet(
        families={
            validation.OPTIMIZED: ("min_var",),
            equity_risk.OPTIMIZED_SUBSET: ("min_var_subset",),
        },
        rebalance="held",
    )
    assert portfolios.family_of("min_var_subset") == equity_risk.OPTIMIZED_SUBSET
