"""``reports/equity_bias_statistics.{md,png,csv}``, ``reports/equity_validation_battery.md``,
``reports/equity_risk_stages.csv``, ``reports/equity_lambda_curve.png``. SPEC.md 15.2, 15.7; W7-P4.

The equity ``(X, f, u)`` of W7-P3b through the UNCHANGED ``risk/`` pipeline
and SPEC.md 6's battery, on the month-end grid of SPEC.md 15.7.1 ruling 4.
The machinery is :mod:`mafrm.factors.equity_risk`; this module wires it to the
ladder of :func:`mafrm.factors.bias_report.variants`, scores experiments.md
rows 291-297 against the registrations in ``config/model.yaml`` and renders.

Reads the cache only (through :func:`mafrm.factors.equity_regression.build`);
nothing here crosses the holdout boundary. SPEC.md 5.3's Monte Carlo per
horizon is the one expensive step and is kept as a digest-guarded partial under
``data/processed/equity_bias/``; ``--rebuild`` discards it and ``--column
<horizon>`` builds one horizon's partial (the fan-out child).
"""

from __future__ import annotations

import argparse
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mafrm import config, history
from mafrm.backtest import diagnostics
from mafrm.factors import battery_report, bias_report, equity_risk
from mafrm.factors.cap_report import Verdict
from mafrm.risk import battery, checks, shepard, validation
from mafrm.risk.config import HORIZONS, Horizon, RiskConfig

__all__ = ["EquityRiskReport", "HorizonResult", "VariantRun", "build", "main", "render"]

_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_REPORTS: Final[Path] = _ROOT / "reports"
_MODULE: Final[str] = "mafrm.factors.equity_risk_report"


def _report_path(name: str) -> Path:
    return _REPORTS / name


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantRun:
    """One ladder variant at one horizon, scored daily-held and monthly."""

    scored: equity_risk.Scored
    #: SPEC.md 6.1 on the complete columns (family 1 restricted to the always-present names).
    daily: validation.ValidationReport
    monthly: validation.ValidationReport
    rolling: dict[str, validation.RollingBias]
    #: Family 1 pooled over every ``(name, session)`` cell, with the cell count.
    pooled_individual: float
    pooled_cells: int
    #: Per name on its own sessions: ``B`` and ``T``.
    per_name: pd.DataFrame
    min_var_realised_volatility: float
    #: SPEC.md 10.3's split for family 4, and the random control's share inside.
    split: equity_risk.ComponentSplit | None
    components: dict[str, diagnostics.ComponentBias]
    random_specific_inside: float
    random_specific_median: float
    battery: battery_report.VariantBattery


@dataclass(frozen=True)
class HorizonResult:
    horizon: Horizon
    risk: RiskConfig
    #: Every month-end at or after the arithmetic floor, with a cheap build.
    stages: list[equity_risk.StageBuild]
    #: Month-ends with an eigenfactor forecast (the contiguous tail after the last repair).
    positions: np.ndarray
    #: Indices into ``positions`` the common window keeps.
    scored: np.ndarray
    designs: list[equity_risk.Design]
    eigen: list[dict[str, np.ndarray]]
    specifics: list[equity_risk.SpecificAt]
    history: pd.DataFrame
    lambda_f: dict[str, np.ndarray]
    lambda_s: np.ndarray
    runs: dict[str, VariantRun]
    dropped_by_rank: int
    dropped_by_regime: int
    dropped_by_repair: int
    factor_t: tuple[battery.FactorTStatistic, ...]
    seconds: float

    @property
    def scored_positions(self) -> np.ndarray:
        return np.asarray(self.positions[self.scored], dtype=int)

    @property
    def scored_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([self.designs[i].date for i in self.scored])

    def window(self, assets: int) -> bias_report.ScoredWindow:
        mask = np.zeros(len(self.positions), dtype=bool)
        mask[self.scored] = True
        return bias_report.ScoredWindow(
            mask=mask,
            dates=self.scored_dates,
            dropped_by_repair=self.dropped_by_repair,
            dropped_by_rank=self.dropped_by_rank,
            dropped_by_regime=self.dropped_by_regime,
            assets=assets,
        )


@dataclass(frozen=True)
class EquityRiskReport:
    panel: equity_risk.Panel
    grid: np.ndarray
    results: dict[Horizon, HorizonResult]
    verdicts: tuple[Verdict, ...]
    #: Row 292(c): the two groups' per-name ``B``.
    membership: pd.DataFrame
    #: Row 292(d): family 4's mean absolute weight share on intermittent-industry names.
    intermittent_weight_share: float
    #: Row 294: family 4's mean absolute weight share on singleton names (`psd_repair`, short).
    singleton_weight_share: float
    #: Row 294: the singleton cells (short horizon).
    singletons: pd.DataFrame
    risk_diff: str
    generated: str


# ---------------------------------------------------------------------------
# One horizon
# ---------------------------------------------------------------------------


def _first_clean(stages: Sequence[equity_risk.StageBuild]) -> int:
    fired = [i for i, s in enumerate(stages) if s.repair_fired]
    return (fired[-1] + 1) if fired else 0


def _rolling(
    scored: equity_risk.Scored, report: validation.ValidationReport, *, window: int, clip: float
) -> dict[str, validation.RollingBias]:
    """One rolling ``B`` per family, the median across members, as the macro chart draws it."""
    out: dict[str, validation.RollingBias] = {}
    index = pd.DatetimeIndex(scored.forecasts.index)
    for family, names in report.portfolios.families.items():
        columns = list(names)
        forecasts = scored.forecasts[columns].to_numpy(dtype=float)
        realised = scored.realised[columns].to_numpy(dtype=float)
        if forecasts.shape[0] < window:
            continue
        standardized = validation.standardized_returns(realised, forecasts)
        rolling = validation.rolling_bias_statistic(standardized, index, window=window, clip=clip)
        out[family] = validation.RollingBias(
            values=np.median(rolling.values, axis=1, keepdims=True),
            index=rolling.index,
            window=rolling.window,
            overlap=rolling.overlap,
        )
    return out


def _per_name(scored: equity_risk.Scored, names: Sequence[str], *, clip: float) -> pd.DataFrame:
    rows = []
    for name in names:
        f = scored.forecasts[name].to_numpy(dtype=float)
        r = scored.realised[name].to_numpy(dtype=float)
        mask = np.isfinite(f) & np.isfinite(r)
        if mask.sum() < 2:
            rows.append((name, np.nan, int(mask.sum())))
            continue
        b = validation.bias_statistic(
            validation.standardized_returns(r[mask][:, None], f[mask][:, None]), clip=clip
        )
        rows.append((name, float(b[0]), int(mask.sum())))
    return pd.DataFrame(rows, columns=["name", "B", "T"]).set_index("name")


@dataclass(frozen=True)
class Forecasts:
    """Every per-month-end ingredient one horizon produced, before any family is built."""

    positions: list[int]
    scored: list[int]
    designs: list[equity_risk.Design]
    stages: list[dict[str, np.ndarray]]
    eigen: list[dict[str, np.ndarray]]
    specifics: list[equity_risk.SpecificAt]
    lambda_f: dict[str, np.ndarray]
    lambda_s: np.ndarray


def _score_run(
    spec: bias_report.VariantSpec,
    parts: Forecasts,
    *,
    data: equity_risk.Panel,
    settings: config.Config,
    risk: RiskConfig,
    inputs: battery_report.BatteryInputs,
) -> VariantRun:
    battery_cfg = settings.model.validation
    clip = battery_cfg.standardized_return_clip
    scored = equity_risk.score_variant(
        spec,
        data=data,
        positions=parts.positions,
        scored=parts.scored,
        designs=parts.designs,
        stages=parts.stages,
        eigen=parts.eigen,
        specifics=parts.specifics,
        lambda_f=parts.lambda_f,
        lambda_s=parts.lambda_s,
        subset=data.always_present,
        random_portfolios=battery_cfg.random_portfolios,
        seed=settings.model.seed,
        volatility_halflife=float(risk.volatility_halflife),
        keep_random=spec.stage != "sample",
    )
    # The always-present names have a bar on every session; a few still lack a
    # screened cap or an exposure at some month-end, so `validate`'s complete
    # columns are the subset names in EVERY scored design.
    subset = [name for name in data.always_present if bool(scored.forecasts[name].notna().all())]
    families = dict(scored.portfolios.families)
    families[validation.INDIVIDUAL] = tuple(subset)
    if validation.EIGENFACTOR in families:
        # Ranks from the top that every scored month-end carries (K_d grows).
        families[validation.EIGENFACTOR] = tuple(
            name
            for name in families[validation.EIGENFACTOR]
            if bool(scored.forecasts[name].notna().all())
        )
    complete = validation.PortfolioSet(families=families, rebalance=scored.portfolios.rebalance)
    members = [name for names in complete.families.values() for name in names]
    daily = validation.validate(
        scored.forecasts[members], scored.realised[members], complete, clip=clip
    )
    monthly_f, monthly_r = equity_risk.to_monthly(scored)
    monthly = validation.validate(monthly_f[members], monthly_r[members], complete, clip=clip)
    window = battery_cfg.rolling_window_months * settings.model.data.trading_days_per_month
    rolling = _rolling(scored, daily, window=window, clip=clip)
    individual = list(scored.portfolios.families[validation.INDIVIDUAL])
    f_all = scored.forecasts[individual].to_numpy(dtype=float)
    r_all = scored.realised[individual].to_numpy(dtype=float)
    cells = np.isfinite(f_all) & np.isfinite(r_all)
    pooled = validation.bias_statistic(
        validation.standardized_returns(r_all[cells][:, None], f_all[cells][:, None]), clip=clip
    )
    per_name = _per_name(scored, individual, clip=clip)
    realised_min_var = scored.realised["min_var"].to_numpy(dtype=float)
    forecast_min_var = scored.forecasts["min_var"].to_numpy(dtype=float)

    split: equity_risk.ComponentSplit | None = None
    components: dict[str, diagnostics.ComponentBias] = {}
    random_inside = float("nan")
    random_median = float("nan")
    level = battery_cfg.chi_square_level
    if spec.stage != "sample":
        factor_list: list[np.ndarray] = [np.empty((0, 0)) for _ in parts.positions]
        variance_list: list[np.ndarray] = [np.empty(0) for _ in parts.positions]
        for i in parts.scored:
            factor_list[i], variance_list[i] = equity_risk.variant_ingredients(
                spec,
                stage=parts.stages[i],
                eigen=parts.eigen[i],
                specific_variance=parts.specifics[i].variance,
                lambda_f={tag: float(v[i]) for tag, v in parts.lambda_f.items()},
                lambda_s=float(parts.lambda_s[i]),
            )
        split = equity_risk.component_split(
            data=data,
            positions=parts.positions,
            scored=parts.scored,
            designs=parts.designs,
            weights=scored.min_var_weights,
            factor=factor_list,
            specific_variance=variance_list,
        )
        for name in ("factor", "specific"):
            post, ante = split.component(name)
            components[name] = diagnostics.component_bias(post, ante, name=name, level=level)
        post_total = split.periods["portfolio_return"].to_numpy(dtype=float)
        ante_total = split.periods["ex_ante_total_variance"].to_numpy(dtype=float)
        components["total"] = diagnostics.component_bias(
            post_total, ante_total, name="total", level=level
        )
        inside = []
        random_biases = []
        count = scored.random_weights[0].shape[0]
        for r in range(count):
            draws = [w[r] for w in scored.random_weights]
            control = equity_risk.component_split(
                data=data,
                positions=parts.positions,
                scored=parts.scored,
                designs=parts.designs,
                weights=draws,
                factor=factor_list,
                specific_variance=variance_list,
            )
            post, ante = control.component("specific")
            control_bias = diagnostics.component_bias(post, ante, name="specific", level=level)
            inside.append(control_bias.inside)
            random_biases.append(control_bias.bias)
        random_inside = float(np.mean(inside))
        random_median = float(np.median(random_biases))

    parameters = len(data.always_present) if spec.stage == "sample" else data.factors.factors
    closed_form = shepard.second_order_risk(parameters, halflife=risk.volatility_halflife)
    label = (
        f"Eq. 13, N = {len(data.always_present)} (always-present subset)"
        if spec.stage == "sample"
        else f"Eq. 32, K = {data.factors.factors} (upper bound)"
    )
    scored_battery = battery_report.score_series(
        spec,
        risk.horizon,
        forecast=forecast_min_var,
        realised=realised_min_var,
        member_forecasts=scored.forecasts[subset].to_numpy(dtype=float),
        member_returns=scored.realised[subset].to_numpy(dtype=float),
        bias=daily.median_bias(validation.OPTIMIZED),
        min_var_realised_volatility=float(np.std(realised_min_var, ddof=1)),
        closed_form=closed_form,
        shepard_label=label,
        inputs=inputs,
    )
    return VariantRun(
        scored=scored,
        daily=daily,
        monthly=monthly,
        rolling=rolling,
        pooled_individual=float(pooled[0]),
        pooled_cells=int(cells.sum()),
        per_name=per_name,
        min_var_realised_volatility=float(np.std(realised_min_var, ddof=1)),
        split=split,
        components=components,
        random_specific_inside=random_inside,
        random_specific_median=random_median,
        battery=scored_battery,
    )


def run_horizon(
    horizon: Horizon,
    data: equity_risk.Panel,
    grid: Sequence[int],
    settings: config.Config,
    *,
    progress: bool = True,
    history_frame: pd.DataFrame | None = None,
) -> HorizonResult:
    """Everything one horizon produces. The eigenfactor partial is loaded or built here.

    ``history_frame`` injects an already-built eigenfactor history instead of
    going through :func:`equity_risk.load_history`. W8-P2's holdout evaluation is
    the only caller that passes it, and it must: its panel is longer than the
    committed one, so ``load_history`` would rebuild -- and, worse, WRITE -- a
    holdout-contaminated partial over the in-sample cache that every later
    in-sample run reads. Injection keeps that build in memory.
    """
    started = time.perf_counter()
    risk = RiskConfig.load(horizon=horizon, config=settings)
    frame = data.factors.frame
    stages = equity_risk.stage_builds(risk, data, [int(p) for p in grid])
    clean = _first_clean(stages)
    positions = np.asarray([s.position for s in stages[clean:]], dtype=int)
    position_list = [int(p) for p in positions]
    if positions.size < 3:
        raise equity_risk.EquityRiskError(
            f"run_horizon[{horizon}]: only {positions.size} month-end(s) after the last PSD "
            "repair; the estimator has no answer on this panel"
        )
    hist = (
        history_frame
        if history_frame is not None
        else equity_risk.load_history(
            risk, data.factors, position_list, settings, progress=progress
        )
    )
    expected = pd.DatetimeIndex(frame.index[positions])
    if not pd.DatetimeIndex(hist.index).equals(expected):
        raise equity_risk.EquityRiskError(
            f"run_horizon[{horizon}]: the cached eigenfactor history covers "
            f"{len(hist)} month-ends against {len(expected)} expected; rebuild with "
            f"`python -m {_MODULE} --rebuild`"
        )
    k = data.factors.factors
    scalings = risk.eigenfactor_scaling
    eigen = [
        {
            bias_report.scaling_tag(a): equity_risk.unpack(
                hist[
                    equity_risk.history_columns(k, scalings)[
                        idx * (k * (k + 1) // 2) : (idx + 1) * (k * (k + 1) // 2)
                    ]
                ]
                .iloc[row]
                .to_numpy(dtype=float),
                k,
            )
            for idx, a in enumerate(scalings)
        }
        for row in range(len(hist))
    ]
    buckets = settings.model.equity_risk.size_buckets
    designs = [equity_risk.design_at(data, int(p), buckets=buckets) for p in positions]
    residuals = data.residuals.to_numpy(dtype=float)
    structural = settings.model.equity_risk.singleton_specific_risk == "structural"
    specifics = [
        equity_risk.specific_at(risk, residuals, d, singleton_structural=structural)
        for d in designs
    ]
    lambda_f: dict[str, np.ndarray] = {}
    lambda_s = np.full(len(positions), np.nan)
    for a in scalings:
        tag = bias_report.scaling_tag(a)
        lf, ls = equity_risk.regime_multipliers(
            risk, data, position_list, [e[tag] for e in eigen], specifics
        )
        lambda_f[tag] = lf
        lambda_s = ls
    n_subset = len(data.always_present)
    rank_ok = positions + 1 > n_subset
    regime_ok = np.arange(len(positions)) >= 1
    last_ok = np.arange(len(positions)) < len(positions) - 1
    keep = rank_ok & regime_ok & last_ok
    scored = np.flatnonzero(keep)
    if scored.size == 0:
        raise equity_risk.EquityRiskError(f"run_horizon[{horizon}]: no month-end survives")
    stage_matrices = [s.matrices for s in stages[clean:]]
    parts = Forecasts(
        positions=position_list,
        scored=[int(i) for i in scored],
        designs=designs,
        stages=stage_matrices,
        eigen=eigen,
        specifics=specifics,
        lambda_f=lambda_f,
        lambda_s=lambda_s,
    )
    inputs = battery_report.BatteryInputs.from_config(settings, horizon)
    runs: dict[str, VariantRun] = {}
    for spec in bias_report.variants(settings):
        runs[spec.name] = _score_run(
            spec, parts, data=data, settings=settings, risk=risk, inputs=inputs
        )
        if progress:
            print(f"  {horizon}: scored {spec.name}", flush=True)
    sessions = pd.DatetimeIndex(runs["psd_repair"].scored.forecasts.index)
    factor_t = battery.factor_t_statistics(
        frame.reindex(sessions).to_numpy(dtype=float),
        [str(c) for c in frame.columns],
        newey_west_lags=inputs.newey_west_lags,
        window=inputs.t_window,
        threshold=inputs.t_threshold,
    )
    return HorizonResult(
        horizon=horizon,
        risk=risk,
        stages=stages,
        positions=positions,
        scored=scored,
        designs=designs,
        eigen=eigen,
        specifics=specifics,
        history=hist,
        lambda_f=lambda_f,
        lambda_s=lambda_s,
        runs=runs,
        dropped_by_rank=int((~rank_ok & last_ok).sum()),
        dropped_by_regime=int((rank_ok & ~regime_ok & last_ok).sum()),
        dropped_by_repair=int(clean),
        factor_t=factor_t,
        seconds=time.perf_counter() - started,
    )


# ---------------------------------------------------------------------------
# The rows
# ---------------------------------------------------------------------------


def _curve_ends(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row of NaN-padded ``lambda(k)``: the top (largest eigenvalue), bulk minimum, bottom."""
    top = np.empty(raw.shape[0])
    bulk = np.empty(raw.shape[0])
    bottom = np.empty(raw.shape[0])
    for i, row in enumerate(raw):
        finite = row[np.isfinite(row)]
        top[i] = finite[-1]
        bulk[i] = finite[:-1].min()
        bottom[i] = finite[0]
    return top, bulk, bottom


def _half_width(observations: int, level: float) -> float:
    interval = validation.chi_square_interval(observations, level=level)
    return float(interval.upper - interval.lower) / 2.0


def _membership(data: equity_risk.Panel, run: VariantRun, result: HorizonResult) -> pd.DataFrame:
    """Row 292(c): each name's own-session ``B`` and whether its industry is intermittent."""
    intermittent = set(data.factors.intermittent)
    last_industry: dict[str, str] = {}
    for design in (result.designs[i] for i in result.scored):
        for name, industry_id in zip(design.names, design.industry_id, strict=True):
            last_industry[name] = data.industry_label[int(industry_id)]
    frame = run.per_name.copy()
    frame["industry"] = [last_industry.get(str(n), "") for n in frame.index]
    frame["intermittent"] = frame["industry"].isin(intermittent)
    return frame


def _weight_share(data: equity_risk.Panel, run: VariantRun, result: HorizonResult) -> float:
    intermittent = set(data.factors.intermittent)
    shares = []
    for j, i in enumerate(result.scored):
        design = result.designs[i]
        w = np.abs(run.scored.min_var_weights[j])
        mask = np.asarray(
            [data.industry_label[int(x)] in intermittent for x in design.industry_id], dtype=bool
        )
        shares.append(float(w[mask].sum() / w.sum()))
    return float(np.mean(shares))


def _singleton_weight_share(run: VariantRun, result: HorizonResult) -> float:
    shares = []
    for j, i in enumerate(result.scored):
        w = np.abs(run.scored.min_var_weights[j])
        shares.append(float(w[result.designs[i].singleton].sum() / w.sum()))
    return float(np.mean(shares))


def _singletons(data: equity_risk.Panel, result: HorizonResult) -> pd.DataFrame:
    alone = data.alone().to_numpy()
    finite = np.isfinite(data.residuals.to_numpy(dtype=float))
    rows = []
    for i in result.scored:
        design = result.designs[i]
        build = result.specifics[i]
        for n in np.flatnonzero(design.singleton):
            column = int(design.columns[n])
            observed = finite[: design.position + 1, column]
            pure = bool(alone[: design.position + 1, column][observed].all())
            rows.append(
                {
                    "date": design.date,
                    "name": design.names[n],
                    "industry": data.industry_label[int(design.industry_id[n])],
                    "percentile": float(build.percentile[n]),
                    "sigma_ts": float(build.time_series[n]),
                    "sigma_sh": float(build.deviation[n]),
                    "observations": int(build.observations[n]),
                    "pure": pure,
                    "short_history": bool(build.short_history[n]),
                }
            )
    return pd.DataFrame(rows)


def _verdicts(
    data: equity_risk.Panel,
    results: dict[Horizon, HorizonResult],
    settings: config.Config,
    *,
    membership: pd.DataFrame,
    weight_share: float,
    singletons: pd.DataFrame,
    singleton_share: float,
    risk_diff: str,
) -> tuple[Verdict, ...]:
    reg = settings.model.equity_risk.registrations
    level = settings.model.validation.chi_square_level
    short = results["short"]
    low_tag = bias_report.scaling_tag(settings.model.eigenfactor.scaling_a[0])
    pre = short.runs["psd_repair"]
    eig = short.runs[f"eigen_{low_tag}"]
    out: list[Verdict] = []

    # Row 291. run_pipeline asserted PSD after every asserted stage on every
    # build (it raises otherwise); the cached eigenfactor matrices are re-checked
    # here against the pipeline's OWN floor (checks.eigenvalue_floor), because a
    # literal threshold on a bps^2-scaled matrix would be a second criterion.
    numerics = settings.model.numerics
    eigen_min = float("inf")
    eigen_ok = True
    for r in results.values():
        for e in r.eigen:
            for m in e.values():
                present = equity_risk.present_block(m)
                spectrum = np.linalg.eigvalsh(m[np.ix_(present, present)])
                eigen_min = min(eigen_min, float(spectrum[0]))
                floor = checks.eigenvalue_floor(
                    present.size, float(spectrum[-1]), numerics=numerics
                )
                eigen_ok = eigen_ok and bool(spectrum[0] >= floor)
    stage_min = {
        name: min(
            s.minimum_eigenvalues[name]
            for r in results.values()
            for s in r.stages[r.dropped_by_repair :]
        )
        for name in equity_risk.CHEAP_STAGES
    }
    out.append(
        Verdict(
            291,
            "(a) SPEC.md 15.2: every pre-VRA stage PSD on every month-end build at both "
            "horizons, with zero changes under src/mafrm/risk/",
            eigen_ok and risk_diff == "",
            "every asserted stage passed the pipeline's own PSD check on every build; smallest "
            "eigenvalue after "
            + ", ".join(f"`{k}` {v:.3e}" for k, v in stage_min.items())
            + f", eigenfactor {eigen_min:.3e} against the pipeline's floor (bps^2 units; "
            "newey_west is measured, not asserted); "
            + (
                "`git diff -- src/mafrm/risk/` is empty"
                if risk_diff == ""
                else f"risk/ DIFF: {risk_diff}"
            ),
        )
    )
    b4 = pre.daily.median_bias(validation.OPTIMIZED)
    t_daily = pre.daily.observations
    hw = _half_width(t_daily, level)
    out.append(
        Verdict(
            291,
            "(b) SPEC.md 15.7: family-4 B pre-eigen-adjustment (`psd_repair`, short, daily-held) "
            f"in [{reg.row_291_family4_pre_eigen_min}, {reg.row_291_family4_pre_eigen_max}]",
            reg.row_291_family4_pre_eigen_min <= b4 <= reg.row_291_family4_pre_eigen_max,
            f"B = {b4:.4f} at T = {t_daily:,} sessions (exact half-width {hw:.4f}); monthly leg "
            f"{pre.monthly.median_bias(validation.OPTIMIZED):.4f} at T = "
            f"{pre.monthly.observations}",
        )
    )
    out.append(
        Verdict(
            291,
            "(c) the W8 contrast: that B exceeds the macro model's "
            f"{reg.row_291_macro_family4_pre_eigen} "
            "by more than the half-width",
            b4 - reg.row_291_macro_family4_pre_eigen > hw,
            f"{b4:.4f} - {reg.row_291_macro_family4_pre_eigen} = "
            f"{b4 - reg.row_291_macro_family4_pre_eigen:+.4f} against {hw:.4f}",
        )
    )
    b4_eig = eig.daily.median_bias(validation.OPTIMIZED)
    out.append(
        Verdict(
            291,
            f"(d) H3 at K = {data.factors.factors}: `eigen_{low_tag}` moves family 4 toward 1 by "
            "more than the half-width",
            (abs(b4 - 1.0) - abs(b4_eig - 1.0)) > hw,
            f"{b4:.4f} -> {b4_eig:.4f} ({b4_eig - b4:+.4f}) against {hw:.4f}",
        )
    )
    mrad_pre = pre.daily.mrad(validation.EIGENFACTOR)
    mrad_eig = eig.daily.mrad(validation.EIGENFACTOR)
    out.append(
        Verdict(
            291,
            "(e) family 3's MRAD falls from `psd_repair` to the eigenfactor stage",
            mrad_eig < mrad_pre,
            f"MRAD {mrad_pre:.4f} -> {mrad_eig:.4f}",
        )
    )
    # Row 292
    dropped = data.factors.dropped
    out.append(
        Verdict(
            292,
            "(a) exactly the registered never-present industries are dropped and K = "
            f"{reg.row_291_factors}",
            set(dropped) == set(reg.row_292_never_present)
            and data.factors.factors == reg.row_291_factors,
            f"dropped {list(dropped)}; K = {data.factors.factors}",
        )
    )
    groups = membership.dropna(subset=["B"])
    inter = groups[groups["intermittent"]]
    always = groups[~groups["intermittent"]]
    t_small = (
        int(min(inter["T"].median(), always["T"].median())) if len(inter) and len(always) else 2
    )
    hw_group = _half_width(max(t_small, 2), level)
    gap = (
        float(inter["B"].median() - always["B"].median())
        if len(inter) and len(always)
        else float("nan")
    )
    out.append(
        Verdict(
            292,
            "(c) names in intermittent industries do not carry a higher B than names in "
            "always-present "
            "industries by more than the half-width at the smaller group's median T",
            bool(np.isfinite(gap)) and gap < hw_group,
            f"median B intermittent {inter['B'].median():.4f} ({len(inter)} names, median T "
            f"{inter['T'].median():.0f}) "
            f"vs always-present {always['B'].median():.4f} ({len(always)} names, median T "
            f"{always['T'].median():.0f}); "
            f"gap {gap:+.4f} against {hw_group:.4f}; family-4 |w| share on intermittent names "
            f"{weight_share:.2%}",
        )
    )
    # Row 293
    k = data.factors.factors
    hist = short.history.iloc[short.scored]
    amplitude = float(hist["amplitude"].median())
    raw = hist[[f"lambda_raw_{i}" for i in range(k)]].to_numpy(dtype=float)
    top, bulk_min, bottom = _curve_ends(raw)
    returns_to_one = top > bulk_min
    share = float(returns_to_one.mean())
    out.append(
        Verdict(
            293,
            f"(a) parabola amplitude at K = {k} above the macro model's "
            f"{reg.row_293_macro_amplitude}",
            amplitude > reg.row_293_macro_amplitude,
            f"median over scored month-ends {amplitude:.4f} (range "
            f"{hist['amplitude'].min():.4f}-{hist['amplitude'].max():.4f}); "
            f"K/T_eff {hist['k_over_t'].min():.4f}-{hist['k_over_t'].max():.4f}",
        )
    )
    out.append(
        Verdict(
            293,
            "(b) raw lambda at the dominant eigenvalue above the bulk minimum on at least "
            f"{reg.row_293_return_to_one_min_share:.0%} of scored month-ends",
            share >= reg.row_293_return_to_one_min_share,
            f"{share:.1%}; median raw lambda at the top {np.median(top):.4f}, bulk minimum "
            f"{np.median(bulk_min):.4f}, bottom {np.median(bottom):.4f}; K_d "
            f"{int(hist['factors'].min())}-{int(hist['factors'].max())}",
        )
    )
    fired = [(s.date, s.k_over_t) for r in results.values() for s in r.stages if s.repair_fired]
    quiet = [s.k_over_t for r in results.values() for s in r.stages if not s.repair_fired]
    firing_min = min(kt for _, kt in fired) if fired else float("nan")
    out.append(
        Verdict(
            293,
            "(c) the PSD repair fires on some month-end build and only at realised K/T_eff >= "
            f"{reg.row_293_firing_k_over_t_min}",
            bool(fired) and firing_min >= reg.row_293_firing_k_over_t_min,
            f"{len(fired)} firing(s) over both horizons at K/T_eff {firing_min:.3f} and above; "
            f"largest K/T_eff without a firing {max(quiet):.3f}; last firing "
            + (f"{max(d for d, _ in fired).date()}" if fired else "none"),
        )
    )
    scored_fired = [
        s.date for r in results.values() for s in r.stages[r.dropped_by_repair :] if s.repair_fired
    ]
    out.append(
        Verdict(
            293,
            "(d) row 118: no firing reaches a scored forecast",
            not scored_fired,
            "none" if not scored_fired else f"{len(scored_fired)} on scored forecasts",
        )
    )
    bart = [(s.date, s.k_over_t) for r in results.values() for s in r.stages if s.bartlett_fired]
    out.append(
        Verdict(
            293,
            "(e) row 100's Bartlett fallback fires only at K/T_eff >= "
            f"{reg.row_293_firing_k_over_t_min}, if at all",
            all(kt >= reg.row_293_firing_k_over_t_min for _, kt in bart),
            f"{len(bart)} firing(s)"
            + (f", smallest K/T_eff {min(kt for _, kt in bart):.3f}" if bart else ""),
        )
    )
    # Row 294
    median_pct = float(singletons["percentile"].median()) if len(singletons) else float("nan")
    pure = singletons[singletons["pure"]]
    mixed = singletons[~singletons["pure"]]
    out.append(
        Verdict(
            294,
            f"(b) singleton names' shrunk sigma sits above the {reg.row_294_floor_percentile:g}th "
            "cross-sectional percentile (median rank)",
            bool(np.isfinite(median_pct)) and median_pct > reg.row_294_floor_percentile,
            f"median percentile {median_pct:.1f} over {len(singletons)} singleton cells "
            f"({singletons['name'].nunique() if len(singletons) else 0} names); pure singletons "
            f"{pure['percentile'].median() if len(pure) else float('nan'):.1f} ({len(pure)} "
            "cells), "
            f"intermittent {mixed['percentile'].median() if len(mixed) else float('nan'):.1f} "
            f"({len(mixed)} cells); "
            "at the pre-shrinkage floor "
            f"{sum(s.singletons_at_floor for s in (short.specifics[i] for i in short.scored))} "
            "cells",
        )
    )
    # Row 295
    pre_spec = pre.components["specific"]
    pre_fac = pre.components["factor"]
    eig_spec = eig.components["specific"]
    eig_fac = eig.components["factor"]
    out.append(
        Verdict(
            295,
            "(a) family-4 specific B inside the exact interval at T months, pre-eigen "
            "(`psd_repair`) and post (`eigen_a1`)",
            pre_spec.inside and eig_spec.inside,
            f"B_specific {pre_spec.bias:.4f} -> {eig_spec.bias:.4f} in "
            f"[{pre_spec.lower:.3f}, {pre_spec.upper:.3f}] at T = {pre_spec.periods} months",
        )
    )
    out.append(
        Verdict(
            295,
            "(b) the excess is carried by the FACTOR component pre-eigen: B_factor above its "
            "interval",
            (not pre_fac.inside) and pre_fac.bias > 1.0,
            f"B_factor {pre_fac.bias:.4f} pre-eigen -> {eig_fac.bias:.4f} after the adjustment "
            f"({'inside' if eig_fac.inside else 'outside'}); B_total "
            f"{pre.components['total'].bias:.4f} -> {eig.components['total'].bias:.4f}",
        )
    )
    out.append(
        Verdict(
            295,
            f"(c) control: at least {reg.row_295_random_inside_min_share:.0%} of the random "
            "books' specific B inside (`psd_repair`)",
            pre.random_specific_inside >= reg.row_295_random_inside_min_share,
            f"{pre.random_specific_inside:.0%} inside; median random B_specific "
            f"{pre.random_specific_median:.4f}",
        )
    )
    # Row 296
    outside = []
    for name, run in short.runs.items():
        b2 = run.daily.median_bias(validation.RANDOM)
        interval = validation.chi_square_interval(run.daily.observations, level=level)
        if not interval.contains(b2):
            outside.append(f"`{name}` {b2:.4f}")
    out.append(
        Verdict(
            296,
            "(a) H1: family 2's median B inside the exact interval for every variant including "
            "the naive comparand",
            not outside,
            "all inside" if not outside else "outside: " + ", ".join(outside),
        )
    )
    naive = short.runs["sample"].daily.median_bias(validation.OPTIMIZED)
    factor_sub = pre.daily.median_bias(equity_risk.OPTIMIZED_SUBSET)
    out.append(
        Verdict(
            296,
            "(b) the naive comparand's family-4 B exceeds the factor model's on the same "
            "always-present subset",
            naive > factor_sub,
            f"naive {naive:.4f} vs `psd_repair` on the subset {factor_sub:.4f} (N = "
            f"{len(data.always_present)})",
        )
    )
    # Row 297
    alpha = short.runs["psd_repair"].battery.observations and settings.model.validation.battery
    inputs = battery_report.BatteryInputs.from_config(settings, "short")
    levels = inputs.levels
    sig = inputs.significance
    high = max(levels)
    pre_vra = [r for r in short.runs.values() if r.battery.pre_vra_factor]
    factor_variants = [r for r in short.runs.values() if r.scored.spec.stage != "sample"]

    def names_of(items: Sequence[VariantRun]) -> str:
        return ", ".join(f"`{i.scored.spec.name}`" for i in items) if items else "none"

    kupiec_fail = [r for r in pre_vra if any(r.battery.kupiec[lv].p_value >= sig for lv in levels)]
    tail_fail = [
        r
        for r in short.runs.values()
        if not (r.battery.tails[high].empirical / r.battery.bias) / r.battery.tails[high].normal
        > 1.0
    ]
    ljung_fail = [r for r in pre_vra if any(t.p_value >= sig for t in r.battery.ljung_box.values())]
    slope_fail = [r for r in factor_variants if r.battery.mincer_zarnowitz.slope <= 1.0]
    basel_fail = [
        r for r in pre_vra if r.battery.basel.count("red") <= len(r.battery.basel.blocks) / 2
    ]
    del alpha
    for leg, claim, fails in (
        (
            "(a)",
            f"Kupiec rejects at {sig:.0%} at both levels for every pre-VRA factor variant",
            kupiec_fail,
        ),
        (
            "(b)",
            f"empirical {1 - high:.0%} quantile of `b/B` below the normal's for every variant",
            tail_fail,
        ),
        (
            "(c)",
            "Ljung-Box on `b^2` rejects at all lags for every pre-VRA factor variant",
            ljung_fail,
        ),
        ("(d)", "Mincer-Zarnowitz slope `b > 1` for every factor variant", slope_fail),
        ("(e)", "Basel: a majority of blocks red for every pre-VRA factor variant", basel_fail),
    ):
        out.append(Verdict(297, f"{leg} {claim}", not fails, f"fails on {names_of(fails)}"))
    # Row 298 (W7-P4b, SPEC.md 15.7.3): scored only when the structural reading is in force.
    if settings.model.equity_risk.singleton_specific_risk == "structural":
        out.append(
            Verdict(
                298,
                f"(i) the median singleton cell's shrunk sigma at or above the "
                f"{reg.row_298_floor_percentile:g}th cross-sectional percentile",
                bool(np.isfinite(median_pct)) and median_pct >= reg.row_298_floor_percentile,
                f"median percentile {median_pct:.1f} (from "
                f"{reg.row_298_baseline_singleton_percentile}) over {len(singletons)} cells; pure "
                f"{pure['percentile'].median() if len(pure) else float('nan'):.1f}, intermittent "
                f"{mixed['percentile'].median() if len(mixed) else float('nan'):.1f}",
            )
        )
        n_names = int(singletons["name"].nunique()) if len(singletons) else 0
        out.append(
            Verdict(
                298,
                "(ii) family 4's absolute weight share on the singleton names falls",
                singleton_share < reg.row_298_baseline_singleton_weight_share,
                f"{singleton_share:.2%} from {reg.row_298_baseline_singleton_weight_share:.2%}; "
                f"count share {n_names} of {len(data.always_present)} always-present names = "
                f"{n_names / len(data.always_present):.1%}, of the mean regression set "
                f"{n_names / np.mean([d.assets for d in short.designs]):.1%}",
            )
        )
        b4m = pre.monthly.median_bias(validation.OPTIMIZED)
        out.append(
            Verdict(
                298,
                "(iii) family-4 B and the component split re-scored on the fixed panel "
                "(reported, not gated)",
                True,
                f"B daily-held {b4:.4f} (from {reg.row_298_baseline_family4_daily}), monthly "
                f"{b4m:.4f} (from {reg.row_298_baseline_family4_monthly}); B_specific "
                f"{pre_spec.bias:.4f} (from {reg.row_298_baseline_b_specific}), B_factor "
                f"{pre_fac.bias:.4f} (from {reg.row_298_baseline_b_factor}); eigen_a1 family 4 "
                f"{b4_eig:.4f}",
            )
        )
        spec_fam2 = short.runs["specified_a1"].daily.median_bias(validation.RANDOM)
        out.append(
            Verdict(
                298,
                "re-read once: the fully specified variant's family 2 and the random control",
                True,
                f"specified_a1 family 2 {spec_fam2:.4f} (from "
                f"{reg.row_298_baseline_specified_family2}); random books inside "
                f"{pre.random_specific_inside:.0%} (from "
                f"{reg.row_298_baseline_random_inside:.0%}), "
                f"median random B_specific {pre.random_specific_median:.4f}",
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _risk_diff() -> str:
    try:
        result = subprocess.run(
            ["git", "diff", "--stat", "HEAD", "--", "src/mafrm/risk/"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_ROOT,
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - not a checkout
        return "unknown (not a git checkout)"
    return result.stdout.strip()


def build(cfg: config.Config | None = None, *, progress: bool = True) -> EquityRiskReport:
    cfg = cfg or config.load()
    data = equity_risk.panel(cfg)
    ends = equity_risk.month_end_positions(data.dates)
    floor = bias_report.minimum_observations(
        RiskConfig.load(horizon=HORIZONS[0], config=cfg), factors=data.factors.factors
    )
    grid = np.asarray([p for p in ends if p + 1 >= floor and p < len(data.dates) - 1], dtype=int)
    results = {
        h: run_horizon(h, data, [int(p) for p in grid], cfg, progress=progress) for h in HORIZONS
    }
    short = results["short"]
    membership = _membership(data, short.runs["psd_repair"], short)
    share = _weight_share(data, short.runs["psd_repair"], short)
    singletons = _singletons(data, short)
    singleton_share = _singleton_weight_share(short.runs["psd_repair"], short)
    diff = _risk_diff()
    verdicts = _verdicts(
        data,
        results,
        cfg,
        membership=membership,
        weight_share=share,
        singletons=singletons,
        singleton_share=singleton_share,
        risk_diff=diff,
    )
    from datetime import UTC, datetime

    return EquityRiskReport(
        panel=data,
        grid=grid,
        results=results,
        verdicts=verdicts,
        membership=membership,
        intermittent_weight_share=share,
        singleton_weight_share=singleton_share,
        singletons=singletons,
        risk_diff=diff,
        generated=datetime.now(UTC).isoformat(timespec="seconds"),
    )


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def ladder_table(report: EquityRiskReport) -> pd.DataFrame:
    rows = []
    for horizon, result in report.results.items():
        for name, run in result.runs.items():
            d = run.daily
            m = run.monthly
            has3 = validation.EIGENFACTOR in d.portfolios.families
            rows.append(
                {
                    "horizon": horizon,
                    "variant": name,
                    "fam1_subset_median": d.median_bias(validation.INDIVIDUAL),
                    "fam1_pooled": run.pooled_individual,
                    "fam1_pooled_cells": run.pooled_cells,
                    "fam2_median": d.median_bias(validation.RANDOM),
                    "fam3_median": d.median_bias(validation.EIGENFACTOR) if has3 else np.nan,
                    "fam3_mrad": d.mrad(validation.EIGENFACTOR) if has3 else np.nan,
                    "fam4": d.median_bias(validation.OPTIMIZED),
                    "fam4_subset": d.median_bias(equity_risk.OPTIMIZED_SUBSET),
                    "gap_fam4_fam2": d.median_bias(validation.OPTIMIZED)
                    - d.median_bias(validation.RANDOM),
                    "T_sessions": d.observations,
                    "monthly_fam2": m.median_bias(validation.RANDOM),
                    "monthly_fam4": m.median_bias(validation.OPTIMIZED),
                    "T_months": m.observations,
                    "min_var_realised_vol_bp": run.min_var_realised_volatility,
                    "B_factor": run.components["factor"].bias if run.components else np.nan,
                    "B_specific": run.components["specific"].bias if run.components else np.nan,
                    "B_total_monthly": run.components["total"].bias if run.components else np.nan,
                    "random_specific_inside": run.random_specific_inside,
                    "random_specific_median": run.random_specific_median,
                    "clipped_fam4": d.clipped_fraction[validation.OPTIMIZED],
                }
            )
    return pd.DataFrame(rows)


def stages_table(report: EquityRiskReport) -> pd.DataFrame:
    rows = []
    k = report.panel.factors.factors
    for horizon, result in report.results.items():
        scored_positions = set(int(p) for p in result.scored_positions)
        eigen_by_position = {int(p): e for p, e in zip(result.positions, result.eigen, strict=True)}
        spec_by_position = {
            int(p): s for p, s in zip(result.positions, result.specifics, strict=True)
        }
        hist = result.history
        for s in result.stages:
            row: dict[str, object] = {
                "horizon": horizon,
                "date": s.date,
                "observations": s.observations,
                "k_over_t_eff": s.k_over_t,
                "cond_style": s.style_condition,
                "repair_fired": s.repair_fired,
                "repair_floored": s.repair_floored,
                "bartlett_fired": s.bartlett_fired,
                "scored": s.position in scored_positions,
            }
            for name in equity_risk.CHEAP_STAGES:
                row[f"min_eig_{name}"] = s.minimum_eigenvalues[name]
            e = eigen_by_position.get(s.position)
            if e is not None:
                for tag, m in e.items():
                    row[f"min_eig_eigen_{tag}"] = float(np.linalg.eigvalsh(m)[0])
                values = hist.loc[[s.date]].to_numpy(dtype=float)[0]
                at = {name: i for i, name in enumerate(hist.columns)}
                row["amplitude"] = float(values[at["amplitude"]])
                curve = values[[at[f"lambda_raw_{i}"] for i in range(k)]]
                top, bulk_min, bottom = _curve_ends(curve[None, :])
                row["lambda_raw_top"] = float(top[0])
                row["lambda_raw_bottom"] = float(bottom[0])
                row["lambda_raw_bulk_min"] = float(bulk_min[0])
                row["factors_present"] = s.factors
                row["unentered_names"] = int(spec_by_position[s.position].design.unentered.sum())
                sp = spec_by_position[s.position]
                row["names"] = sp.design.assets
                row["singletons"] = int(sp.design.singleton.sum())
                row["short_history"] = int(sp.short_history.sum())
                row["exact_zero"] = int(sp.exact_zero.sum())
                row["structural_r2"] = sp.structural_r_squared
                row["max_z"] = float(np.nanmax(sp.z)) if np.isfinite(sp.z).any() else np.nan
                row["bartlett_fallbacks_specific"] = sp.bartlett_fallbacks
                row["target_shift"] = sp.target_shift
                idx = int(np.flatnonzero(result.positions == s.position)[0])
                for tag, values in result.lambda_f.items():
                    row[f"lambda_f2_{tag}"] = float(values[idx])
                row["lambda_s2"] = float(result.lambda_s[idx])
            rows.append(row)
    return pd.DataFrame(rows)


def _md_table(frame: pd.DataFrame, fmt: dict[str, str]) -> list[str]:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in frame.iterrows():
        cells = []
        for c in columns:
            v = row[c]
            if isinstance(v, float):
                cells.append(fmt.get(c, "{:.4f}").format(v) if np.isfinite(v) else "--")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render(report: EquityRiskReport, cfg: config.Config) -> str:
    data = report.panel
    short = report.results["short"]
    reg = cfg.model.equity_risk.registrations
    level = cfg.model.validation.chi_square_level
    k = data.factors.factors
    pre = short.runs["psd_repair"]
    ladder = ladder_table(report)
    out: list[str] = []
    add = out.append
    add("# The equity model through the unchanged risk pipeline (SPEC.md 15.2, 15.7; W7-P4)")
    add("")
    add(
        f"Generated by `python -m {_MODULE}` at {report.generated}. The equity `(X, f, u)` of "
        "W7-P3b fed through `mafrm.risk` exactly as Model A is, under the five rulings of "
        "SPEC.md 15.7.1, scored on SPEC.md 6.2's four families and SPEC.md 6.5's battery. "
        f"Nothing here reaches `sample.holdout_start` ({data.holdout_start.date()})."
    )
    add("")
    add("## Verdicts (experiments.md rows 291-297)")
    add("")
    for v in report.verdicts:
        add(f"- row {v.row} **{'HOLDS' if v.holds else 'REFUTED'}** -- {v.leg}: {v.detail}")
    add("")
    add("## What was scored")
    add("")
    n_range = (
        min(d.assets for d in short.designs),
        max(d.assets for d in short.designs),
    )
    k_range = (min(s.factors for s in short.stages), max(s.factors for s in short.stages))
    late = data.factors.first_present[data.factors.first_present > 0].sort_values()
    add(
        f"- **The frame (ruling 1).** `K = {k}` at the end of the sample: "
        f"{len(data.factors.industry_columns)} industries present on at least one date plus the "
        f"country and {len(data.factors.style_columns)} styles; dropped as never present: "
        f"{list(data.factors.dropped)}. **Point in time**, an industry enters `K` at its first "
        "appearance -- before it, its history is all zeros and `risk/` refuses a zero variance, "
        f"correctly -- so `K_d` runs from {k_range[0]} to {k_range[1]} over the grid; entering "
        "after the first session: "
        + ", ".join(f"{n} ({data.dates[int(p)].date()})" for n, p in late.items())
        + f". {len(data.factors.intermittent)} industries are intermittent and zero-filled on "
        "absent dates after entry (table below). Returns in basis points per day "
        f"(x{cfg.model.equity_risk.basis_points_per_unit:g})."
    )
    add(
        f"- **The grid (ruling 4).** {len(report.grid)} month-ends from the arithmetic floor "
        f"`max(K + 1, lags + 1)` = {bias_report.minimum_observations(short.risk, factors=k)} rows; "
        f"forecasts at the close of each, weights held through the following month, scored on "
        "every session (daily-held) and on the month. `M = "
        f"{short.risk.eigenfactor_monte_carlo_trials:,}`."
    )
    for horizon, r in report.results.items():
        add(
            f"- **Scored window, {horizon}.** {len(r.scored)} of {len(r.positions)} month-ends "
            "with an "
            f"eigenfactor forecast, {r.scored_dates[0].date()} to {r.scored_dates[-1].date()}: "
            "dropped "
            f"{r.dropped_by_repair} month-end build(s) up to the last PSD-repair firing, "
            f"{r.dropped_by_rank} where `T <= N_subset = {len(data.always_present)}` leaves the "
            "naive "
            f"comparand singular, {r.dropped_by_regime} with no VRA multiplier yet, and the last "
            f"month-end (nothing to hold it over). {r.runs['psd_repair'].daily.observations:,} "
            "daily "
            f"sessions, {r.runs['psd_repair'].monthly.observations} months. Names per forecast "
            f"{n_range[0]}-{n_range[1]}; liquidated `(name, session)` cells "
            f"{r.runs['psd_repair'].scored.liquidated_cells:,}. Wall time {r.seconds / 60:.1f} min."
        )
    add(
        f"- **The always-present subset (construction 7)**: {len(data.always_present)} names with "
        "a "
        "return on every regression date -- a survivor set, labelled as such wherever it appears. "
        "The naive comparand lives on it; every factor variant's family 4 is also scored on it."
    )
    add(
        "- **Size deciles (ruling 2)**: ten equal-count buckets by screened cap at the close, "
        "recomputed each forecast date. Singletons held out of the target (construction 5)."
    )
    reading = cfg.model.equity_risk.singleton_specific_risk
    add(
        f"- **Singleton specific risk**: `equity_risk.singleton_specific_risk = {reading}`. "
        + (
            "SPEC.md 15.7.3 (W7-P4b): a singleton takes leg (b)'s structural estimate. Rows "
            "291-297 as registered were scored under `time_series` at commit 680a207; this render "
            "re-scores them under the fixed panel and scores row 298."
            if reading == "structural"
            else "W7-P4's reading: a singleton's own time-series estimate."
        )
    )
    add(
        "- **`src/mafrm/risk/`**: `git diff --stat HEAD -- src/mafrm/risk/` is "
        + ("**empty**." if report.risk_diff == "" else f"NOT empty: `{report.risk_diff}`")
    )
    add("")
    add(
        "## SPEC.md 15.2's acceptance: every stage, every month-end, PSD, beside the style "
        "block's condition"
    )
    add("")
    stage_rows = []
    for horizon, r in report.results.items():
        scored_builds = r.stages[r.dropped_by_repair :]
        row: dict[str, object] = {
            "horizon": horizon,
            "builds": len(r.stages),
            "scored_builds": len(scored_builds),
        }
        for name in equity_risk.CHEAP_STAGES:
            row[f"min eig `{name}`"] = min(s.minimum_eigenvalues[name] for s in scored_builds)
        for tag in r.lambda_f:
            row[f"min eig eigen {tag}"] = min(float(np.linalg.eigvalsh(e[tag])[0]) for e in r.eigen)
        row["cond_style range"] = (
            f"{min(s.style_condition for s in r.stages):.2f}-"
            f"{max(s.style_condition for s in r.stages):.2f}"
        )
        row["repair firings"] = sum(s.repair_fired for s in r.stages)
        row["bartlett firings"] = sum(s.bartlett_fired for s in r.stages)
        row["K/T_eff range"] = (
            f"{min(s.k_over_t for s in scored_builds):.4f}-"
            f"{max(s.k_over_t for s in scored_builds):.4f}"
        )
        stage_rows.append(row)
    out.extend(
        _md_table(
            pd.DataFrame(stage_rows),
            {
                "min eig `ewma`": "{:.3e}",
                "min eig `newey_west`": "{:.3e}",
                "min eig `psd_repair`": "{:.3e}",
                "min eig eigen a1": "{:.3e}",
                "min eig eigen a1.4": "{:.3e}",
            },
        )
    )
    add("")
    add(
        "The per-month-end record -- every stage's smallest eigenvalue, the regression's "
        "style-block "
        "condition number on that date (row 288(b): peak 6.09, 2009-04-07), the repair and "
        "fallback "
        "flags, `K/T_eff`, the lambda curve's ends and the specific-risk diagnostics -- is "
        "`reports/equity_risk_stages.csv`. `newey_west` is measured and not asserted (SPEC.md "
        "5.2); "
        "its successor repairs it."
    )
    add("")
    add("## The ladder: SPEC.md 6.2's four families under every variant")
    add("")
    add(
        "Daily-held: `b_t = R_t / sigma_t` on every session with the month-end forecast standing, "
        "clipped at +-4 for the standard deviation (SPEC.md 6.1). Family 1 is reported twice -- "
        "the "
        "median over the always-present names, which `validate` sees as complete columns, and the "
        "pooled statistic over every `(name, session)` cell of the full universe. Family 4 is the "
        "minimum-variance book on the full universe; `fam4_subset` the same book on the "
        "always-present "
        "names. Headline: **family 4 pre-eigen-adjustment (`psd_repair`), short** (SPEC.md 6.2.2's "
        "pre-VRA ruling and SPEC.md 15.7's target)."
    )
    add("")
    fmt = {
        "fam1_pooled_cells": "{:.0f}",
        "T_sessions": "{:.0f}",
        "T_months": "{:.0f}",
        "min_var_realised_vol_bp": "{:.3f}",
        "random_specific_inside": "{:.2f}",
        "clipped_fam4": "{:.4f}",
    }
    show = ladder[
        [
            "horizon",
            "variant",
            "fam1_subset_median",
            "fam1_pooled",
            "fam2_median",
            "fam3_median",
            "fam3_mrad",
            "fam4",
            "fam4_subset",
            "gap_fam4_fam2",
            "T_sessions",
            "min_var_realised_vol_bp",
        ]
    ]
    out.extend(_md_table(show, fmt))
    add("")
    interval = validation.chi_square_interval(pre.daily.observations, level=level)
    add(
        f"Exact interval under the null at the daily `T`: {interval.render()}. The macro model's "
        f"pre-eigen family-4 `B` on the same statistic is {reg.row_291_macro_family4_pre_eigen} "
        "(short)."
    )
    add("")
    add("### The monthly leg")
    add("")
    monthly = ladder[["horizon", "variant", "monthly_fam2", "monthly_fam4", "T_months"]]
    out.extend(_md_table(monthly, {"T_months": "{:.0f}"}))
    add("")
    add("### The rolling chart")
    add("")
    any_roll = next(iter(pre.rolling.values()))
    add(
        f"`reports/equity_bias_statistics.png`: rolling `B` over {any_roll.window} sessions "
        "stepped one "
        f"session (overlap {any_roll.overlap}; CLAUDE.md failure mode 9 -- no count of windows is "
        "an "
        "`n`), the median across members per family, short horizon, one line per variant. Bands "
        "are "
        "SPEC.md 6.1's normal band and the exact chi-square interval at the window length."
    )
    add("")
    add("## Ruling 1's accounting: the zero-fill (row 292)")
    add("")
    zf = pd.DataFrame(
        {
            "industry": data.factors.intermittent,
            "f_k": data.factors.presence[list(data.factors.intermittent)].to_numpy(),
            "understatement 1 - f_k": data.factors.understatement.to_numpy(),
        }
    )
    zf = zf.sort_values("f_k")
    out.extend(_md_table(zf, {"f_k": "{:.4f}", "understatement 1 - f_k": "{:.4f}"}))
    add("")
    m = report.membership.dropna(subset=["B"])
    add(
        "Per-name daily-held `B` (`psd_repair`, short) by industry membership: "
        "intermittent-industry "
        f"names median {m[m['intermittent']]['B'].median():.4f} over "
        f"{int(m['intermittent'].sum())} names "
        f"(median `T` {m[m['intermittent']]['T'].median():.0f} sessions), always-present-industry "
        "names "
        f"{m[~m['intermittent']]['B'].median():.4f} over {int((~m['intermittent']).sum())} names "
        f"(median `T` {m[~m['intermittent']]['T'].median():.0f}). Family 4's mean absolute weight "
        "share "
        f"on intermittent-industry names: **{report.intermittent_weight_share:.2%}**."
    )
    add("")
    add(f"## The lambda curve at K = {k} (row 293)")
    add("")
    hist = short.history.iloc[short.scored]
    raw = hist[[f"lambda_raw_{i}" for i in range(k)]].to_numpy(dtype=float)
    fit = hist[[f"lambda_fit_{i}" for i in range(k)]].to_numpy(dtype=float)
    top, bulk_min, bottom = _curve_ends(raw)
    fit_bottom, _unused, fit_top = _curve_ends(fit)
    add(
        f"Over the {len(hist)} scored month-ends (short; the stage is horizon-independent): "
        "amplitude "
        f"median **{hist['amplitude'].median():.4f}**, range "
        f"{hist['amplitude'].min():.4f}-{hist['amplitude'].max():.4f}, "
        "against 0.0099 at `K = 6`; `K/T_eff` "
        f"{hist['k_over_t'].min():.4f}-{hist['k_over_t'].max():.4f}. "
        f"Median raw `lambda` at the smallest eigenvalue {np.median(bottom):.4f}, at the "
        f"largest {np.median(top):.4f}, bulk minimum {np.median(bulk_min):.4f}; the fitted "
        f"parabola's ends {np.median(fit_bottom):.4f} / {np.median(fit_top):.4f}. The dominant "
        "eigenvalue's raw `lambda` sits above the bulk minimum on "
        f"{float((top > bulk_min).mean()):.1%} "
        "of scored month-ends. `reports/equity_lambda_curve.png` draws three dates and the "
        "amplitude through time."
    )
    add("")
    add(
        "## Specific risk on the equity panel (SPEC.md 5.5 in regime), and the singletons (row 294)"
    )
    add("")
    sp = [short.specifics[i] for i in short.scored]
    add(
        "Per scored forecast (short): names "
        f"{min(s.design.assets for s in sp)}-{max(s.design.assets for s in sp)}; "
        f"leg (b) `R^2` median {np.median([s.structural_r_squared for s in sp]):.3f} on "
        f"{int(np.median([s.structural_degrees_of_freedom for s in sp]))} residual d.o.f. "
        "(against 6 at `N = 13`); "
        f"median `Z` {np.median([np.nanmedian(s.z) for s in sp]):.2f} "
        "(its maximum is infinite wherever a name's interquartile range is zero); names below the "
        "arithmetic floor "
        f"(structural forecast) median {int(np.median([s.short_history.sum() for s in sp]))}, "
        "exact-zero histories "
        f"median {int(np.median([s.exact_zero.sum() for s in sp]))}; Bartlett fallbacks total "
        f"{sum(s.bartlett_fallbacks for s in sp)}; largest relative move of a bucket target from "
        "holding "
        f"singletons out {max(s.target_shift for s in sp):.2%}."
    )
    add("")
    s = report.singletons
    if len(s):
        pure = s[s["pure"]]
        mixed = s[~s["pure"]]
        add(
            f"Singleton cells: **{len(s)}** over {s['name'].nunique()} names and "
            f"{s['industry'].nunique()} industries; "
            "median cross-sectional percentile of the shrunk `sigma_SH` "
            f"**{s['percentile'].median():.1f}** "
            "(pure singletons -- alone on every observed session -- "
            f"{pure['percentile'].median() if len(pure) else float('nan'):.1f} "
            f"over {len(pure)} cells; intermittent "
            f"{mixed['percentile'].median() if len(mixed) else float('nan'):.1f} over "
            f"{len(mixed)} cells). "
            "Names most often singleton: "
            f"{', '.join(f'{n} ({c})' for n, c in s['name'].value_counts().head(8).items())}. "
            "Family 4's mean absolute weight share on singleton names (`psd_repair`, short): "
            f"**{report.singleton_weight_share:.2%}**."
        )
    add("")
    add("## SPEC.md 10.3's component assertion on the equity family-4 book (row 295)")
    add("")
    comp = ladder[ladder["variant"] != "sample"][
        [
            "horizon",
            "variant",
            "B_factor",
            "B_specific",
            "B_total_monthly",
            "random_specific_inside",
            "T_months",
        ]
    ]
    out.extend(_md_table(comp, {"random_specific_inside": "{:.2f}", "T_months": "{:.0f}"}))
    c = pre.components["specific"]
    drift_sd = float("nan")
    specific_sd = float("nan")
    if pre.split is not None:
        periods = pre.split.periods
        drift_sd = float(
            (periods["specific_return"] - periods["regression_specific_return"]).std(ddof=1)
        )
        specific_sd = float(periods["specific_return"].std(ddof=1))
    add("")
    add(
        f"Exact interval at `T = {c.periods}` months: [{c.lower:.3f}, {c.upper:.3f}]. Ex-post "
        "specific is the "
        "remainder `w'r - (X'w)'f` (construction 9); the gap to `w'u` from the regression's own "
        "residuals "
        "has a monthly sd of "
        f"{drift_sd:.2f} bp against a specific-return sd of {specific_sd:.2f} bp -- exposure "
        "drift inside the month."
    )
    add("")
    add("## SPEC.md 5.4's multipliers on the held forecast")
    add("")
    for horizon, r in report.results.items():
        last = r.scored[-1]
        add(
            f"- {horizon}: at the last scored month-end `lambda_F^2` "
            + ", ".join(f"{tag} {float(v[last]):.4f}" for tag, v in r.lambda_f.items())
            + f", `lambda_S^2` {float(r.lambda_s[last]):.4f}; medians over the scored window "
            + ", ".join(
                f"{tag} {float(np.nanmedian(v[r.scored])):.4f}" for tag, v in r.lambda_f.items()
            )
            + f" / {float(np.nanmedian(r.lambda_s[r.scored])):.4f}. Daily `B_t` under the held "
            "forecast, EWMA at "
            f"{r.risk.volatility_regime_halflife}d (construction 6). Post-VRA rows are reported, "
            "not tested (SPEC.md 6.2.2)."
        )
    add("")
    add("## Files")
    add("")
    add(
        "- `reports/equity_bias_statistics.csv` -- the ladder table above in full (both legs, "
        "components)."
    )
    add("- `reports/equity_risk_stages.csv` -- the per-month-end record for both horizons.")
    add(
        "- `reports/equity_bias_statistics.png` -- the rolling chart; "
        "`reports/equity_lambda_curve.png` -- the lambda curve."
    )
    add(
        "- `reports/equity_validation_battery.md` -- SPEC.md 6.5 on every variant (row 297) and "
        "the per-factor `t` table."
    )
    add(
        "- `data/processed/equity_bias/<horizon>.csv` -- the gitignored, digest-guarded "
        "eigenfactor partials (`--rebuild`)."
    )
    add("")
    return "\n".join(out)


def render_battery(report: EquityRiskReport, cfg: config.Config) -> str:
    results: dict[Horizon, battery_report.HorizonBattery] = {}
    for horizon, r in report.results.items():
        results[horizon] = battery_report.HorizonBattery(
            horizon=horizon,
            inputs=battery_report.BatteryInputs.from_config(cfg, horizon),
            variants=tuple(run.battery for run in r.runs.values()),
            window=r.window(len(report.panel.always_present)),
            factor_t=r.factor_t,
        )
    short = results["short"]
    legs = {h: battery_report.legs(results[h]) for h in HORIZONS}
    out: list[str] = []
    add = out.append
    add("# The validation battery on the equity model (SPEC.md 6.5, 15.7; W7-P4)")
    add("")
    add(
        f"Generated by `python -m {_MODULE}`. SPEC.md 6.5's tests on the equity family-4 "
        "daily-held "
        "series (min-var on the full universe, forecast refreshed at each month-end, pre-VRA by "
        "SPEC.md 6.2.2 for the headline variants), every ladder variant, both horizons. The rank "
        "correlation runs on the always-present names, the only complete member matrix. Same "
        "rulings "
        "and config as `reports/validation_battery.md`; nothing re-configured (row 297)."
    )
    add("")
    add(f"- {short.window.render()}")
    add("")
    add("## Every test, every variant")
    add("")
    add(
        "The Shepard column applies Eq. 32 at `K` for the factor variants (an upper bound, as in "
        "`reports/bias_statistics.md`). For `sample` it would be Eq. 13 at `N = "
        f"{len(report.panel.always_present)}` against `T_eff = 242` at the 84d half-life -- "
        "`N/T_eff > 1`, outside the closed form's domain -- so that cell is not a number and is "
        "printed as 0."
    )
    add("")
    out.extend(battery_report.master_table(results))
    add("")
    add(
        "## The nine legs of rows 187-188, re-read on the equity series (row 297 registers (a)-(e))"
    )
    add("")
    for horizon in HORIZONS:
        add(f"**{horizon}**")
        add("")
        for leg in legs[horizon]:
            add(f"- {leg.leg} {'HOLDS' if leg.holds else 'FAILS'} -- {leg.claim}: {leg.detail}")
        add("")
    add("## Per-factor time-series `t` (SPEC.md 15.7), short-horizon sessions")
    add("")
    add(
        "Nothing is pruned on this table (CLAUDE.md: choosing risk factors by mean return is alpha "
        "research; NLSIZE kept by SPEC.md 15.6.3 ruling 2). The rolling share is over overlapping "
        "windows and is not an `n`; the non-overlapping column carries the count."
    )
    add("")
    out.extend(battery_report.factor_table(short))
    add("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def chart(report: EquityRiskReport, cfg: config.Config, path: Path) -> None:
    short = report.results["short"]
    battery_cfg = cfg.model.validation
    families = [
        f
        for f in (*validation.FAMILIES, equity_risk.OPTIMIZED_SUBSET)
        if any(f in r.rolling for r in short.runs.values())
    ]
    shown = [s for s in bias_report.variants(cfg) if s.name != "sample"]
    window = battery_cfg.rolling_window_months * cfg.model.data.trading_days_per_month
    normal = validation.normal_band(window, z=battery_cfg.normal_band_z)
    exact = validation.chi_square_interval(window, level=battery_cfg.chi_square_level)
    figure, axes = plt.subplots(len(families), 1, figsize=(11, 3.0 * len(families)), sharex=True)
    for axis, family in zip(np.atleast_1d(axes), families, strict=True):
        for spec in shown:
            rolling = short.runs[spec.name].rolling.get(family)
            if rolling is None:
                continue
            axis.plot(rolling.index, rolling.values[:, 0], lw=1.0, label=spec.name)
        sample = short.runs["sample"].rolling.get(family)
        if sample is not None:
            axis.plot(
                sample.index,
                sample.values[:, 0],
                lw=1.0,
                ls=":",
                color="black",
                label="sample (subset)",
            )
        axis.axhspan(exact.lower, exact.upper, color="grey", alpha=0.15, lw=0)
        axis.axhline(normal.lower, color="grey", lw=0.6, ls="--")
        axis.axhline(normal.upper, color="grey", lw=0.6, ls="--")
        axis.axhline(1.0, color="black", lw=0.6)
        axis.set_ylabel(f"{family}\nrolling B ({window}d)")
        axis.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(ncol=4, fontsize=7, loc="upper left")
    np.atleast_1d(axes)[0].set_title(
        f"Equity model, K = {report.panel.factors.factors}, short horizon: rolling bias statistic "
        "by family "
        f"(window {window} sessions, overlap {window - 1} of {window}; median across members)",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=130, facecolor="white")
    plt.close(figure)


def lambda_chart(report: EquityRiskReport, path: Path) -> None:
    short = report.results["short"]
    k = report.panel.factors.factors
    hist = short.history
    scored_dates = short.scored_dates
    picks = [scored_dates[0], scored_dates[len(scored_dates) // 2], scored_dates[-1]]
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    ranks = np.arange(k)
    for date in picks:
        row = hist.loc[date]
        raw = row[[f"lambda_raw_{i}" for i in range(k)]].to_numpy(dtype=float)
        fit = row[[f"lambda_fit_{i}" for i in range(k)]].to_numpy(dtype=float)
        keep = np.isfinite(raw)
        line = left.plot(
            ranks[keep],
            raw[keep],
            lw=0.8,
            alpha=0.8,
            label=f"raw {date.date()} (K_d={int(keep.sum())})",
        )
        left.plot(ranks[keep], fit[keep], lw=1.4, ls="--", color=line[0].get_color())
    left.axhline(1.0, color="black", lw=0.6)
    left.set_xlabel("eigenvalue rank (0 = smallest)")
    left.set_ylabel("lambda(k)")
    left.set_title(
        f"SPEC.md 5.3's bias curve at K = {k} (raw, and the fitted parabola dashed)", fontsize=9
    )
    left.legend(fontsize=7)
    left.grid(alpha=0.3)
    right.plot(hist.index, hist["amplitude"], lw=1.0, label="amplitude")
    right.axhline(0.0099, color="grey", ls="--", lw=0.8, label="K = 6 macro (0.0099)")
    twin = right.twinx()
    twin.plot(hist.index, hist["k_over_t"], lw=0.8, color="tab:red", label="K/T_eff")
    twin.set_ylabel("K/T_eff", color="tab:red")
    right.set_ylabel("amplitude lambda_P(0) - lambda_P(K-1)")
    right.set_title("Amplitude and K/T_eff through the month-end grid", fontsize=9)
    right.legend(fontsize=7, loc="upper right")
    right.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=130, facecolor="white")
    plt.close(figure)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _partials_fresh(data: equity_risk.Panel, cfg: config.Config) -> bool:
    for horizon in HORIZONS:
        risk = RiskConfig.load(horizon=horizon, config=cfg)
        path = history.partial_path(equity_risk.PARTIALS, equity_risk.HorizonColumn(horizon))
        if not path.exists():
            return False
        existing = history.read_partial(path)
        digests = equity_risk.history_digests(risk, data.factors, cfg)
        if not all(existing.attrs.get(key) == value for key, value in digests.items()):
            return False
    return True


def _grid_for(data: equity_risk.Panel, cfg: config.Config) -> np.ndarray:
    ends = equity_risk.month_end_positions(data.dates)
    floor = bias_report.minimum_observations(
        RiskConfig.load(horizon=HORIZONS[0], config=cfg), factors=data.factors.factors
    )
    return np.asarray([p for p in ends if p + 1 >= floor and p < len(data.dates) - 1], dtype=int)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - entry point
    parser = argparse.ArgumentParser(
        description="SPEC.md 15.2/15.7: the equity model through risk/"
    )
    parser.add_argument("--rebuild", action="store_true", help="discard the eigenfactor partials")
    parser.add_argument(
        "--column", default=None, help="build one horizon's partial (fan-out child)"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    cfg = config.load()
    data = equity_risk.panel(cfg)
    grid = _grid_for(data, cfg)
    if args.column is not None:
        horizon = args.column
        if horizon not in HORIZONS:
            raise SystemExit(f"unknown horizon {horizon!r}")
        risk = RiskConfig.load(horizon=horizon, config=cfg)
        stages = equity_risk.stage_builds(risk, data, [int(p) for p in grid])
        positions = [s.position for s in stages[_first_clean(stages) :]]
        started = time.perf_counter()
        equity_risk.load_history(risk, data.factors, positions, cfg)
        print(f"{horizon}: eigenfactor history ready in {time.perf_counter() - started:.0f}s")
        return 0
    if args.rebuild:
        for horizon in HORIZONS:
            path = history.partial_path(equity_risk.PARTIALS, equity_risk.HorizonColumn(horizon))
            if path.exists():
                path.unlink()
    if not _partials_fresh(data, cfg):
        columns = [equity_risk.HorizonColumn(h) for h in HORIZONS]
        if history.fan_out(_MODULE, columns, flag="--column"):
            raise SystemExit(1)
    report = build(cfg)
    for v in report.verdicts:
        print(f"row {v.row} {'HOLDS ' if v.holds else 'REFUTED'} {v.leg}: {v.detail}")
    _REPORTS.mkdir(parents=True, exist_ok=True)
    _report_path("equity_bias_statistics.md").write_text(render(report, cfg), encoding="utf-8")
    _report_path("equity_validation_battery.md").write_text(
        render_battery(report, cfg), encoding="utf-8"
    )
    ladder_table(report).to_csv(
        _report_path("equity_bias_statistics.csv"),
        index=False,
        lineterminator="\n",
        float_format="%.6g",
    )
    stages_table(report).to_csv(
        _report_path("equity_risk_stages.csv"),
        index=False,
        lineterminator="\n",
        float_format="%.6g",
    )
    chart(report, cfg, _report_path("equity_bias_statistics.png"))
    lambda_chart(report, _report_path("equity_lambda_curve.png"))
    print(f"wrote {_report_path('equity_bias_statistics.md')} and companions")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
