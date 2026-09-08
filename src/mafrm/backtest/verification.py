"""W6-P1's ONE verification configuration of SPEC.md 8's optimizer, and its reports.

SPEC.md 8.5.1 ruling 6: *"W6-P1 builds the optimizer and runs ONE verification
configuration. RSTR, patient Y, a = 1.0, gamma_trade = 1.0, short horizon,
TE_target at 1x. State it as verification, not result. The grid is W6-P2's job."*

This module is that run. It is NOT the experiment grid of SPEC.md 9 and it
publishes no Sharpe ratio: it exists to show that the optimizer solves on every
rebalance date in the sample without an unhandled failure, which constraints
bind and how often, what the weight distribution and the misalignment angle look
like, and whether the ``alpha = 0`` control lands on SPEC.md 6.2's family 4.

WHAT IT READS, AND FROM WHERE
    - The risk forecast at each rebalance is the exact matrix W4-P2's family 4
      was scored on -- :func:`mafrm.factors.bias_report.forecast_panel`, the
      pre-VRA eigenfactor-adjusted ``F`` at ``a = 1.0`` on the short horizon
      (SPEC.md 6.2.2), the lagged design ``X`` and the pre-VRA specific
      variances. A decision at the close of ``t`` uses the panel's row for the
      NEXT scored date, which is the forecast built from everything through ``t``.
    - ``alpha-hat`` is RSTR (:mod:`mafrm.backtest.alpha`) on the assets' TRUE
      daily excess returns, whose full history the descriptor needs, centred
      cross-sectionally and scaled to the rebalance period.
    - The book is marked on the same true excess returns, so a strategy earns
      what its holdings earned. The four government zeros' forecast was fitted to
      their duration leg (SPEC.md 4.3, W2-P3's control), so their realised return
      carries the carry-and-roll term the forecast's regressand does not; that is
      a level difference of a few basis points a month and is stated, not hidden.
    - Costs come from :func:`mafrm.costs.inputs.build_cost_inputs`: the issuer
      half-spread INTERVAL (SPEC.md 7.1.3), trailing volatility and dollar ADV,
      each taken AS OF the rebalance date (the latest monthly value stamped at or
      before it -- causal, and at most a month stale).
    - NAV at each rebalance is the engine's own ``NAV_pre`` from an incremental
      re-run of :func:`mafrm.backtest.engine.run_backtest` over the targets so
      far, so the optimizer prices impact and the ADV cap on the book the engine
      actually holds -- one accounting, not a second copy of it.

THE BAND. The half-spread is an interval and the book size is a rule (SPEC.md
8.5.1's third construction: the two endpoints of the ruled AUM grid read off the
equal-weight book), so the RSTR configuration runs four times and every figure
is reported as a band. The ``alpha = 0`` control runs once, cost-free, at a NAV
that cannot matter to it.

Everything stops strictly before ``sample.holdout_start``: the panel, the cache,
the cost inputs and the return series all do, and the last rebalance holds
through the last in-sample date because the forecast for the date after it does
not exist.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from mafrm import config as config_mod
from mafrm import history
from mafrm.backtest import alpha as alpha_mod
from mafrm.backtest import constraints as con
from mafrm.backtest import engine
from mafrm.backtest import optimizer as opt
from mafrm.costs import inputs as cost_inputs
from mafrm.data import calendar
from mafrm.factors import betas, bias_report, macro
from mafrm.risk import validation
from mafrm.risk.config import RiskConfig

__all__ = [
    "RunConfig",
    "RunResult",
    "Universe",
    "book_sizes",
    "build_universe",
    "main",
    "render",
    "render_binding",
    "run_configuration",
    "run_configurations",
]

_REPORTS: Final[Path] = Path(__file__).resolve().parents[3] / "reports"
_MARKDOWN_PATH: Final[Path] = _REPORTS / "optimizer_verification.md"
_BINDING_PATH: Final[Path] = _REPORTS / "binding_constraints.md"
_CSV_PATH: Final[Path] = _REPORTS / "optimizer_verification.csv"
_CHART_PATH: Final[Path] = _REPORTS / "optimizer_weights.png"

#: The risk panel is in basis points per day (``bias_report.Inputs.returns``).
_BPS: Final[float] = 1e-4

#: ``gamma_risk`` for the ``alpha = 0`` control. SPEC.md 8.4 gives ``lambda = 0``
#: there; cost-free, the long-only minimum-variance portfolio is the same for
#: EVERY positive value (``tests/test_optimizer.py`` asserts the invariance), so
#: this is the scale the solver sees and not a parameter.
_CONTROL_GAMMA_RISK: Final[float] = 1.0

#: The pseudo-constraint the report counts beside the configured limits: the
#: long-only floor ``w_i >= 0`` binding on an asset.
_LONG_ONLY_FLOOR: Final[str] = "long_only_floor"


# ---------------------------------------------------------------------------
# Inputs, read once
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Universe:
    """Everything the run reads, assembled once and shared by every configuration."""

    panel: bias_report.ForecastPanel
    spec: bias_report.VariantSpec
    assets: tuple[str, ...]
    sleeves: dict[str, str]
    #: The two identity assets (R^2 = 1; W4-P1b), for the IR decomposition.
    excluded: tuple[str, ...]
    #: Daily TRUE excess returns on the panel's dates, decimal.
    excess: pd.DataFrame
    #: The risk panel's own asset returns, ``T x N`` in basis points per day on
    #: the FULL panel index (``bias_report.Inputs.returns``); ``panel.positions``
    #: index its rows. W6-P2's dense grid variants are estimated on it, so they
    #: see the same regressand the pipeline's naive comparand sees.
    asset_returns: pd.DataFrame
    #: Daily TRUE excess returns on the FULL in-sample history, complete cases
    #: across the assets, decimal. The residual panel above starts only when
    #: every rolling beta is fitted (April 2009), so it cannot feed a window
    #: that reaches back from the first rebalance; W6-P2's dense grid variants
    #: are estimated on this series instead (SPEC.md 9.1).
    excess_history: pd.DataFrame
    #: ``(1 + excess).cumprod()`` -- the engine's price panel.
    prices: pd.DataFrame
    #: RSTR per session, full history, decimal per session.
    alpha_rate: pd.DataFrame
    #: Cost inputs AS OF each rebalance date, (rebalance date x asset).
    half_spread: dict[str, pd.DataFrame]
    daily_volatility: pd.DataFrame
    adv_dollars: pd.DataFrame
    #: In-sample median dollar ADV per asset, for the book-size rule.
    adv_median: pd.Series
    rebalance_dates: pd.DatetimeIndex
    #: Month ends dropped because RSTR's 525-session window was not yet full.
    dropped_for_alpha: int
    #: Per-period TE_target (decimal) and the daily equal-weight volatility behind it.
    te_target: float
    equal_weight_daily_volatility: float
    horizon_days: int
    #: ``bps^2/day^2 -> decimal^2/period``.
    scale: float
    #: The risk horizon the panel was built at (``"short"`` or ``"long"``); W6-P3's band.
    horizon: str = "short"
    #: Ken French's daily bill rate, decimal per session, on the panel's dates --
    #: the financing leg's rate (operator ruling 1, W6-P3).
    risk_free: pd.Series | None = None
    #: ``T x K`` realised factor returns on the panel's dates, DECIMAL per day
    #: (the panel's basis points divided by 1e4), for SPEC.md 10.3's attribution.
    factor_returns: pd.DataFrame | None = None

    @property
    def terminal_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.panel.dates[-1])


def _as_of(frame: pd.DataFrame, dates: pd.DatetimeIndex, *, label: str) -> pd.DataFrame:
    """The latest value stamped AT OR BEFORE each date. Causal; at most one month stale."""
    union = frame.index.union(dates)
    out = frame.reindex(union).ffill().reindex(dates)
    if not np.all(np.isfinite(out.to_numpy(dtype=float))):
        missing = out.columns[out.isna().any()].tolist()
        raise ValueError(
            f"verification: {label} has no value at or before some rebalance date for {missing}"
        )
    return out


def _rstr_by_asset(
    log_excess: pd.DataFrame, alpha_cfg: config_mod.OptimizerAlphaConfig
) -> pd.DataFrame:
    """RSTR per asset on ITS OWN sessions, then placed back on the union calendar.

    The union calendar leaves ``NaN`` wherever one asset did not trade (the GSW
    curve prices on bond-market days the ETFs do not, and vice versa), and a
    525-session window over the union would contain one on every asset -- the
    first run of this module produced no alpha at any month end for exactly that
    reason. CNE5's descriptor is a sum over the security's own sessions, so each
    column is scored on its own non-missing rows -- its sessions that also carry
    a bill rate -- and read back on the dates it has a value, which every panel
    date is, because the panel is complete-cases.
    """
    columns = {}
    for name in log_excess.columns:
        own = log_excess[name].dropna().to_frame()
        scored = alpha_mod.rstr(
            own, window=alpha_cfg.window, halflife=alpha_cfg.halflife, lag=alpha_cfg.lag
        )
        columns[name] = scored[name].reindex(log_excess.index)
    return pd.DataFrame(columns, index=log_excess.index)


def _variant(settings: config_mod.Config) -> bias_report.VariantSpec:
    ver = settings.model.optimizer.verification
    name = f"{ver.risk_variant}_a{ver.eigenfactor_scaling:g}"
    for spec in bias_report.variants(settings):
        if spec.name == name:
            return spec
    raise ValueError(f"verification: no risk variant named {name!r} in bias_report.variants")


def build_universe(
    settings: config_mod.Config | None = None, *, horizon: str | None = None
) -> Universe:
    """Read the panel, the cache, the true returns and the cost inputs. No network.

    ``horizon`` overrides the verification horizon the panel is read at -- the
    W6-P3 horizon band reads the committed panel's ``long`` column. Everything
    else (the true returns, the alpha, the costs, the anchor) is horizon-free.
    """
    settings = settings or config_mod.load()
    data = bias_report.inputs(settings)
    return build_universe_from(
        settings, horizon=horizon, data=data, cache=bias_report.read_cache(data, settings)
    )


def build_universe_from(
    settings: config_mod.Config | None = None,
    *,
    horizon: str | None = None,
    data: bias_report.Inputs,
    cache: history.Cache,
) -> Universe:
    """:func:`build_universe` on a panel and cache the caller already holds.

    EXTRACTED IN W8-P2, AT THE SECOND CALLER
        The holdout evaluation (:mod:`mafrm.backtest.holdout`) builds its own
        panel -- longer than the committed one, which is the whole point -- and
        its own extended eigenfactor history, so it cannot go through
        :func:`bias_report.read_cache`'s panel digest. Everything below that
        point is identical, and copying it would be two universes that must
        agree and are not checked against each other.
    """
    settings = settings or config_mod.load()
    opt_cfg = settings.model.optimizer
    ver = opt_cfg.verification
    holdout = settings.require_holdout_start()
    column = horizon or ver.horizon
    if column not in ("short", "long"):
        raise ValueError(f"build_universe: horizon is 'short' or 'long', got {column!r}")

    risk = RiskConfig.load(horizon=column, config=settings)  # type: ignore[arg-type]
    panel = bias_report.forecast_panel(risk, column, data, cache)
    spec = _variant(settings)
    assets = tuple(data.assets)

    # The TRUE excess returns, full history, for RSTR and for the book.
    excess_all = calendar.align(
        betas.asset_excess_returns(config=settings), how="union", end=holdout
    )[list(assets)]
    risk_free = macro._daily_risk_free()
    total = excess_all.add(risk_free.reindex(excess_all.index), axis=0)
    log_excess = alpha_mod.log_excess_returns(total, risk_free)
    alpha_rate = _rstr_by_asset(log_excess, opt_cfg.alpha)
    excess = excess_all.reindex(panel.dates)
    if not np.all(np.isfinite(excess.to_numpy(dtype=float))):
        raise ValueError("verification: a true excess return is missing on a panel date")
    prices = (1.0 + excess).cumprod()

    # Rebalance calendar: month ends of the scored window with a next scored date
    # (the forecast for it exists) and a full RSTR window behind them.
    month_ends = calendar.month_end_dates(pd.DatetimeIndex(panel.dates))
    has_next = month_ends < panel.dates[-1]
    alpha_ready = alpha_rate.reindex(month_ends).notna().all(axis=1).to_numpy()
    keep = has_next & alpha_ready
    dropped_for_alpha = int((has_next & ~alpha_ready).sum())
    rebalance_dates = pd.DatetimeIndex(month_ends[keep], name="date")
    if len(rebalance_dates) < 2:
        raise ValueError("verification: fewer than two rebalance dates")

    costs = cost_inputs.build_cost_inputs(settings)
    half_spread = {
        "low": _as_of(
            costs.half_spread_low[list(assets)], rebalance_dates, label="half_spread_low"
        ),
        "high": _as_of(
            costs.half_spread_high[list(assets)], rebalance_dates, label="half_spread_high"
        ),
    }
    daily_volatility = _as_of(
        costs.daily_volatility[list(assets)], rebalance_dates, label="daily_volatility"
    )
    adv_dollars = _as_of(costs.adv_dollars[list(assets)], rebalance_dates, label="adv_dollars")
    adv_median = costs.adv_dollars[list(assets)].median()

    horizon_days = opt_cfg.alpha.horizon_days
    equal_weight = excess.mean(axis=1)
    ew_daily = float(equal_weight.std(ddof=1))
    te_target = ew_daily * np.sqrt(horizon_days) * ver.tracking_error_multiple

    sleeves = {asset.id: asset.sleeve for asset in settings.universe.assets}
    rf_on_panel = risk_free.reindex(panel.dates)
    if not np.all(np.isfinite(rf_on_panel.to_numpy(dtype=float))):
        raise ValueError("verification: the risk-free rate is missing on a panel date")
    # The panel's factor returns are in basis points per day; the book is in decimal.
    factor_returns = data.factors.reindex(panel.dates)[list(data.factor_names)] * _BPS
    if not np.all(np.isfinite(factor_returns.to_numpy(dtype=float))):
        raise ValueError("verification: a factor return is missing on a panel date")
    if float(np.abs(factor_returns.to_numpy(dtype=float)).max()) >= 1.0:
        raise ValueError(
            "verification: a daily factor return of 100% or more; the panel is in basis points "
            "and the conversion to decimal is wrong"
        )
    return Universe(
        panel=panel,
        spec=spec,
        assets=assets,
        sleeves={name: sleeves[name] for name in assets},
        excluded=tuple(data.excluded),
        excess=excess,
        asset_returns=data.returns[list(assets)],
        excess_history=excess_all.dropna(),
        prices=prices,
        alpha_rate=alpha_rate,
        half_spread=half_spread,
        daily_volatility=daily_volatility,
        adv_dollars=adv_dollars,
        adv_median=adv_median,
        rebalance_dates=rebalance_dates,
        dropped_for_alpha=dropped_for_alpha,
        te_target=float(te_target),
        equal_weight_daily_volatility=ew_daily,
        horizon_days=horizon_days,
        scale=_BPS**2 * horizon_days,
        horizon=column,
        risk_free=rf_on_panel,
        factor_returns=factor_returns,
    )


def book_sizes(universe: Universe, settings: config_mod.Config) -> dict[str, float]:
    """The NAV band, SPEC.md 8.5.1's third construction, or the ruled fixed NAV.

    ``A = participation * N * min_i median(V_i)``: the AUM at which the
    EQUAL-WEIGHT trade from cash into the thinnest asset is ``participation`` of
    that asset's in-sample median ADV, at the two participations
    ``costs.capacity.aum_grid`` rules.
    """
    book = settings.model.optimizer.verification.book_size
    if book.rule == "fixed":
        assert book.nav_dollars is not None
        return {"fixed": float(book.nav_dollars)}
    grid = settings.model.costs.capacity.aum_grid
    n = len(universe.assets)
    thinnest = float(universe.adv_median.min())
    return {
        "A_low": grid.low_participation_of_adv * n * thinnest,
        "A_high": grid.high_participation_of_adv * n * thinnest,
    }


# ---------------------------------------------------------------------------
# One configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    label: str
    #: ``"rstr"`` or ``"zero"``.
    alpha: str
    #: ``"low"`` / ``"high"`` end of the spread interval; ``None`` when cost-free.
    spread_end: str | None
    nav: float
    book_label: str
    cost_free: bool


@dataclass(frozen=True)
class RunResult:
    config: RunConfig
    #: One row per rebalance: weights, angle, penalties, statuses, readings.
    rows: pd.DataFrame
    targets: pd.DataFrame
    backtest: engine.BacktestResult
    readings: tuple[tuple[con.Reading, ...], ...]
    #: Monthly net and gross returns and the forecast volatility standing at the start.
    monthly: pd.DataFrame
    summary: dict[str, float] = field(default_factory=dict)


def _cost_model(
    universe: Universe, settings: config_mod.Config, run: RunConfig
) -> engine.CostModel:
    costs = settings.model.costs
    exponent = costs.total_cost_exponent
    if run.cost_free:
        return engine.CostModel.free(
            index=universe.rebalance_dates, columns=pd.Index(universe.assets), exponent=exponent
        )
    assert run.spread_end is not None
    regime = settings.model.optimizer.verification.cost_regime
    prefactor = getattr(costs.square_root_prefactor, regime)
    return engine.CostModel(
        half_spread=universe.half_spread[run.spread_end],
        daily_volatility=universe.daily_volatility,
        adv_dollars=universe.adv_dollars,
        prefactor=float(prefactor),
        exponent=exponent,
        asymmetry=costs.buy_sell_asymmetry,
    )


def run_configuration(
    universe: Universe, settings: config_mod.Config, run: RunConfig, *, progress: bool = True
) -> RunResult:
    """Solve every rebalance, feed the engine, and score the result. One configuration."""
    opt_cfg = settings.model.optimizer
    battery = settings.model.validation
    n = len(universe.assets)
    assets = list(universe.assets)
    cost_model = _cost_model(universe, settings, run)

    limits = con.limits_from_config(opt_cfg.constraints)
    if run.cost_free:
        limits = tuple(limit for limit in limits if limit.name != "adv_participation")
    names = {limit.name for limit in limits}
    ladder = tuple(name for name in opt_cfg.relaxation_ladder if name in names)

    targets = pd.DataFrame(np.nan, index=universe.rebalance_dates, columns=assets)
    rows: list[dict[str, object]] = []
    readings: list[tuple[con.Reading, ...]] = []
    for i, t in enumerate(universe.rebalance_dates):
        stamp = pd.Timestamp(t)
        if i == 0:
            nav_pre = run.nav
            prior = np.zeros(n)
        else:
            partial = engine.run_backtest(
                universe.prices.loc[:stamp],
                targets.iloc[:i],
                cost_model=cost_model,
                initial_nav=run.nav,
            )
            nav_pre = float(partial.nav.loc[stamp])
            prior = np.asarray(partial.weights.loc[stamp].to_numpy(dtype=float), dtype=float)

        position = int(universe.panel.dates.get_indexer(pd.DatetimeIndex([stamp]))[0])
        if position < 0:
            raise ValueError(f"verification: {stamp.date()} is not a scored date")
        exposures, factor, specific = universe.panel.ingredients(universe.spec, position + 1)
        risk = opt.FactorRisk(
            exposures=exposures,
            factor_covariance=factor * universe.scale,
            specific_variance=specific * universe.scale,
        )
        sigma = risk.covariance()

        if run.alpha == "rstr":
            rate = universe.alpha_rate.loc[stamp, assets]
            centred = alpha_mod.demean(rate)
            assert isinstance(centred, pd.Series)
            alpha = np.asarray(
                alpha_mod.to_horizon(centred, days=universe.horizon_days), dtype=float
            )
            ir = opt.information_ratio(alpha, sigma)
            ir_diagonal = float(np.sqrt(np.sum(alpha**2 / np.diag(sigma))))
            keep = [k for k, name in enumerate(assets) if name not in universe.excluded]
            sub_alpha = alpha[keep] - alpha[keep].mean()
            ir_without_identity = opt.information_ratio(sub_alpha, sigma[np.ix_(keep, keep)])
            gamma_risk = (
                opt.gamma_risk_from_spec(alpha, sigma, te_target=universe.te_target)
                if opt_cfg.gamma_risk is None
                else opt_cfg.gamma_risk
            )
            decomposition = alpha_mod.decompose(alpha, exposures)
            cos_theta = decomposition.cos_theta
            penalty = (
                alpha_mod.misalignment_penalty(decomposition, sigma, gamma_risk=gamma_risk)
                if opt_cfg.misalignment_penalty == "msci"
                else alpha_mod.MisalignmentPenalty(psi_unit=0.0, direction=np.zeros(n))
            )
        else:
            alpha = np.zeros(n)
            ir = ir_diagonal = ir_without_identity = 0.0
            gamma_risk = _CONTROL_GAMMA_RISK if opt_cfg.gamma_risk is None else opt_cfg.gamma_risk
            cos_theta = float("nan")
            penalty = alpha_mod.MisalignmentPenalty(psi_unit=0.0, direction=np.zeros(n))

        cost: opt.TradingCost | None = None
        if not run.cost_free:
            assert run.spread_end is not None
            cost = opt.TradingCost(
                half_spread=universe.half_spread[run.spread_end]
                .loc[stamp, assets]
                .to_numpy(dtype=float),
                daily_volatility=universe.daily_volatility.loc[stamp, assets].to_numpy(dtype=float),
                adv_over_nav=universe.adv_dollars.loc[stamp, assets].to_numpy(dtype=float)
                / nav_pre,
                prefactor=cost_model.prefactor,
                exponent=cost_model.exponent,
                asymmetry=cost_model.asymmetry,
            )
        problem = opt.RebalanceProblem(
            alpha=alpha,
            risk=risk,
            prior_weights=prior,
            penalties=opt.Penalties(
                gamma_risk=float(gamma_risk),
                gamma_trade=opt_cfg.verification.gamma_trade,
                gamma_hold=opt_cfg.gamma_hold,
                psi_unit=penalty.psi_unit,
                misalignment_direction=penalty.direction if penalty.active else None,
                rho=opt_cfg.robustification.return_forecast_rho,
                varrho=opt_cfg.robustification.covariance_varrho,
            ),
            cost=cost,
            limits=limits,
            long_only=opt_cfg.constraints.long_only,
            fully_invested=opt_cfg.constraints.fully_invested,
            relaxation_order=ladder,
            primary_solver=opt_cfg.solver.primary,
            fallback_solver=opt_cfg.solver.fallback,
            feasibility_tolerance=opt_cfg.solver.feasibility_tolerance,
        )
        result = opt.solve_rebalance(problem)
        targets.loc[stamp] = result.weights
        readings.append(result.readings)

        family4 = validation.minimum_variance_weights(sigma)
        row: dict[str, object] = {
            "configuration": run.label,
            "date": stamp,
            "nav_pre": nav_pre,
            "gamma_risk": float(gamma_risk),
            "information_ratio": ir,
            "information_ratio_diagonal": ir_diagonal,
            "information_ratio_without_identity": ir_without_identity,
            "lowest_volatility_asset": assets[int(np.argmin(np.diag(sigma)))],
            "lowest_volatility": float(np.sqrt(np.diag(sigma).min())),
            "alpha_dispersion": float(np.std(alpha)),
            "cos_theta": cos_theta,
            "psi_unit": penalty.psi_unit,
            "psi_mis": penalty.psi_mis,
            "forecast_volatility": result.forecast_volatility,
            "turnover": result.turnover,
            "trading_cost": result.trading_cost,
            "status": result.status,
            "attempt": result.attempt.label,
            "rung": result.attempt.rung,
            "relaxed": ",".join(result.relaxed),
            "fell_back": result.fell_back,
            "at_lower_bound": result.at_lower_bound,
            "largest_weight": result.largest_weight,
            "effective_assets": result.effective_assets,
            "family4_has_short": bool(family4.min() < 0.0),
            "family4_l1_distance": float(np.abs(result.weights - family4).sum()),
            "term_alpha": result.terms["alpha"],
            "term_risk": result.terms["risk"],
            "term_misalignment": result.terms["misalignment"],
            "term_trading_cost": result.terms["trading_cost"],
        }
        for reading in result.readings:
            row[f"{reading.name}_binding"] = reading.binding
            row[f"{reading.name}_multiplier"] = reading.multiplier
            row[f"{reading.name}_value"] = reading.value
        for name, weight in zip(assets, result.weights, strict=True):
            row[f"w_{name}"] = float(weight)
        rows.append(row)
        if progress and (i + 1) % 50 == 0:
            print(f"  {run.label}: {i + 1:,} / {len(universe.rebalance_dates):,}", flush=True)

    backtest = engine.run_backtest(
        universe.prices, targets, cost_model=cost_model, initial_nav=run.nav
    )
    frame = pd.DataFrame(rows).set_index("date")
    monthly = _monthly(universe, frame, backtest)
    summary = _summary(universe, settings, run, frame, backtest, monthly, readings, battery)
    return RunResult(
        config=run,
        rows=frame,
        targets=targets,
        backtest=backtest,
        readings=tuple(readings),
        monthly=monthly,
        summary=summary,
    )


def _monthly(
    universe: Universe, rows: pd.DataFrame, backtest: engine.BacktestResult
) -> pd.DataFrame:
    """Net and gross returns per holding period, non-overlapping, with the standing forecast."""
    marks = universe.rebalance_dates.append(pd.DatetimeIndex([universe.terminal_date]))
    nav = backtest.nav.reindex(marks).to_numpy(dtype=float)
    cost = backtest.total_cost.reindex(marks[1:]).fillna(0.0).to_numpy(dtype=float)
    net = nav[1:] / nav[:-1] - 1.0
    gross = (nav[1:] + cost) / nav[:-1] - 1.0
    return pd.DataFrame(
        {
            "start": marks[:-1],
            "net_return": net,
            "gross_return": gross,
            "cost_over_nav": cost / nav[:-1],
            "forecast_volatility": rows["forecast_volatility"].to_numpy(dtype=float),
        },
        index=pd.DatetimeIndex(marks[1:], name="end"),
    )


def _summary(
    universe: Universe,
    settings: config_mod.Config,
    run: RunConfig,
    rows: pd.DataFrame,
    backtest: engine.BacktestResult,
    monthly: pd.DataFrame,
    readings: Sequence[Sequence[con.Reading]],
    battery: config_mod.ValidationBatteryConfig,
) -> dict[str, float]:
    n = len(universe.assets)
    periods_per_year = settings.model.data.trading_days_per_year / universe.horizon_days
    net = monthly["net_return"].to_numpy(dtype=float)
    gross = monthly["gross_return"].to_numpy(dtype=float)
    forecast = monthly["forecast_volatility"].to_numpy(dtype=float)
    standardized = validation.standardized_returns(net[:, None], forecast[:, None])
    bias = float(validation.bias_statistic(standardized, clip=battery.standardized_return_clip)[0])
    interval = validation.chi_square_interval(len(net), level=battery.chi_square_level)
    nav_pre = rows["nav_pre"].to_numpy(dtype=float)
    spread = backtest.spread_cost.to_numpy(dtype=float).sum()
    impact = backtest.impact_cost.to_numpy(dtype=float).sum()
    total_cost = backtest.total_cost.to_numpy(dtype=float).sum()
    at_floor = rows["at_lower_bound"].to_numpy(dtype=int)
    summary: dict[str, float] = {
        "rebalances": float(len(rows)),
        "first_rung": float((rows["rung"] == 0).sum()),
        "inaccurate": float((rows["status"] == "optimal_inaccurate").sum()),
        "relaxed": float((rows["relaxed"] != "").sum()),
        "fell_back": float(rows["fell_back"].sum()),
        "cos_theta_mean": float(rows["cos_theta"].mean()),
        "cos_theta_median": float(rows["cos_theta"].median()),
        "cos_theta_min": float(rows["cos_theta"].min()),
        "gamma_risk_median": float(rows["gamma_risk"].median()),
        "information_ratio_median": float(rows["information_ratio"].median()),
        "information_ratio_diagonal_median": float(rows["information_ratio_diagonal"].median()),
        "information_ratio_without_identity_median": float(
            rows["information_ratio_without_identity"].median()
        ),
        "lowest_volatility_median": float(rows["lowest_volatility"].median()),
        "alpha_dispersion_median": float(rows["alpha_dispersion"].median()),
        "effective_assets_mean": float(rows["effective_assets"].mean()),
        "effective_assets_min": float(rows["effective_assets"].min()),
        "largest_weight_median": float(rows["largest_weight"].median()),
        "largest_weight_max": float(rows["largest_weight"].max()),
        "at_floor_mean": float(at_floor.mean()),
        "any_at_floor_share": float((at_floor > 0).mean()),
        "corner_share": float((at_floor >= n - 2).mean()),
        "turnover_annual": float(rows["turnover"].mean() * periods_per_year),
        "cost_drag_bps_per_year": float(total_cost / nav_pre.sum() * periods_per_year * 1e4),
        "spread_share_of_cost": float(spread / total_cost) if total_cost > 0.0 else float("nan"),
        "impact_share_of_cost": float(impact / total_cost) if total_cost > 0.0 else float("nan"),
        "gross_annual": float(gross.mean() * periods_per_year),
        "net_annual": float(net.mean() * periods_per_year),
        "realised_volatility": float(np.std(net, ddof=1)),
        "forecast_volatility_rms": float(np.sqrt(np.mean(forecast**2))),
        "te_attainment": float(np.sqrt(np.mean(forecast**2)) / universe.te_target),
        "bias": bias,
        "bias_lower": interval.lower,
        "bias_upper": interval.upper,
        "months": float(len(net)),
        "family4_short_share": float(rows["family4_has_short"].mean()),
        "family4_l1_mean": float(rows["family4_l1_distance"].mean()),
        "final_cash_over_nav": float(backtest.cash.iloc[-1] / backtest.nav.iloc[-1]),
    }
    table = con.summarise(
        readings, percentile=settings.model.optimizer.constraints.lagrange_percentile
    )
    label = f"multiplier_p{round(settings.model.optimizer.constraints.lagrange_percentile * 100)}"
    if "adv_participation" in table.index:
        row = table.to_dict(orient="index")["adv_participation"]
        summary["adv_binding_share"] = float(row["binding_share"])
        summary["adv_binding_count"] = float(row["binding"])
        summary["adv_multiplier_p"] = float(row[label])
        summary["adv_multiplier_max"] = float(row["multiplier_max"])
    return summary


def run_configurations(
    universe: Universe, settings: config_mod.Config, *, progress: bool = True
) -> list[RunResult]:
    """The band: RSTR at every (book size, spread end), then the cost-free ``alpha = 0`` control."""
    ver = settings.model.optimizer.verification
    sizes = book_sizes(universe, settings)
    configs: list[RunConfig] = []
    for book_label, nav in sizes.items():
        for end in ver.spread_ends:
            configs.append(
                RunConfig(
                    label=f"{ver.alpha}/{book_label}/spread_{end}",
                    alpha=ver.alpha,
                    spread_end=end,
                    nav=nav,
                    book_label=book_label,
                    cost_free=False,
                )
            )
    first_label, first_nav = next(iter(sizes.items()))
    configs.append(
        RunConfig(
            label="zero/control",
            alpha="zero",
            spread_end=None,
            nav=first_nav,
            book_label=first_label,
            cost_free=ver.control_cost_free,
        )
    )
    results = []
    for run in configs:
        if progress:
            print(f"solving {run.label} at NAV {run.nav:,.0f}", flush=True)
        results.append(run_configuration(universe, settings, run, progress=progress))
    return results


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _pct(value: float, digits: int = 1) -> str:
    return "n/a" if not np.isfinite(value) else f"{100.0 * value:.{digits}f}%"


def _num(value: float, digits: int = 3) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def _dollars(value: float) -> str:
    return f"${value / 1e6:,.1f}M" if value >= 1e6 else f"${value:,.0f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def render(universe: Universe, results: Sequence[RunResult], settings: config_mod.Config) -> str:
    ver = settings.model.optimizer.verification
    opt_cfg = settings.model.optimizer
    costs = settings.model.costs
    sizes = book_sizes(universe, settings)
    thinnest = str(universe.adv_median.idxmin())
    rstr = [r for r in results if r.config.alpha == "rstr"]
    control = next(r for r in results if r.config.alpha == "zero")
    n = len(universe.assets)
    dates = universe.rebalance_dates

    lines = [
        "# Optimizer verification: one configuration, stated as verification and not as a result",
        "",
        "Generated by `python -m mafrm.backtest.verification`. SPEC.md 8 under the rulings of SPEC.md 8.5.1, W6-P1.",
        "",
        "**This is not the experiment grid and it publishes no Sharpe ratio.** SPEC.md 8.5.1 ruling 6: W6-P1 runs ONE configuration to show that the optimizer solves on every date without an unhandled failure, what binds, and what the weights and the misalignment angle look like. SPEC.md 9's grid is W6-P2's job, and the moment any of these numbers is compared against an alternative on outcome, the rows in `experiments.md` change category.",
        "",
        "## The configuration",
        "",
        *_table(
            ["Ingredient", "Value", "Where ruled"],
            [
                [
                    "`alpha-hat`",
                    "RSTR (SPEC.md 15.4): T = 504, half-life 126, lag 21; normalised weights, equal-weighted cross-sectional centring, scaled to 21 sessions",
                    "ruling 1; construction at SPEC.md 8.5.1",
                ],
                [
                    "Risk forecast",
                    f"`{universe.spec.name}`: {universe.spec.label}; {ver.horizon} horizon, the matrix behind W4-P2's family-4 headline",
                    "ruling 6, SPEC.md 6.2.2",
                ],
                [
                    "`gamma_risk`",
                    "SPEC.md 8.4's `IR / (2 TE_target)` at each rebalance"
                    if opt_cfg.gamma_risk is None
                    else f"pinned at {opt_cfg.gamma_risk}",
                    "SPEC.md 8.4",
                ],
                [
                    "`TE_target`",
                    f"{ver.tracking_error_multiple:g} x the equal-weight book's realised in-sample volatility: {_pct(universe.equal_weight_daily_volatility, 3)}/day, {_pct(universe.te_target, 3)}/period",
                    "ruling 3",
                ],
                [
                    "Costs",
                    f"SPEC.md 7.1 inside the objective; `Y` {ver.cost_regime} = {getattr(costs.square_root_prefactor, ver.cost_regime):g}, exponent {costs.total_cost_exponent:g}, `gamma_trade` = {ver.gamma_trade:g}; half-spread at both ends of the issuer interval",
                    "ruling 6, SPEC.md 7.1.3",
                ],
                [
                    "`psi_mis`",
                    f"`{opt_cfg.misalignment_penalty}`: `lambda * u' Sigma u` on the unit `alpha_perp` direction, computed each rebalance",
                    "ruling 4; construction",
                ],
                [
                    "`rho`, `varrho`, `gamma_hold`, `w^b`",
                    f"{opt_cfg.robustification.return_forecast_rho:g}, {opt_cfg.robustification.covariance_varrho:g}, {opt_cfg.gamma_hold:g}, zero -- absences",
                    "ruling 4",
                ],
                [
                    "Constraints",
                    f"long-only, fully invested (hard, the simplex); ADV cap {_pct(opt_cfg.constraints.adv_participation.bound or 0.0, 0)} of ADV HARD (multipliers observed); box {opt_cfg.constraints.position_box.bound:g} and turnover {opt_cfg.constraints.turnover.bound:g} at their simplex maxima; TE hinge off",
                    "ruling 5",
                ],
                [
                    "Ladder",
                    f"{', '.join(opt_cfg.relaxation_ladder)} -> prior weights; solvers {opt_cfg.solver.primary} then {opt_cfg.solver.fallback}",
                    "SPEC.md 8.1",
                ],
                [
                    "Book size",
                    " / ".join(f"{k} = {_dollars(v)}" for k, v in sizes.items())
                    + f" (the ruled AUM grid's endpoints read off the equal-weight trade in `{thinnest}`, median ADV {_dollars(float(universe.adv_median.min()))})",
                    "construction at SPEC.md 8.5.1",
                ],
                [
                    "Calendar",
                    f"{len(dates)} month-end rebalances, {dates[0].date()} to {dates[-1].date()}, held through {universe.terminal_date.date()}; {universe.dropped_for_alpha} early month end(s) dropped while RSTR's 525-session window filled",
                    "SPEC.md 2",
                ],
            ],
        ),
        "",
        "## Did it solve",
        "",
        *_table(
            [
                "Configuration",
                "NAV",
                "Rebalances",
                "Solved at rung 0",
                "`optimal_inaccurate`",
                "Constraint relaxed",
                "Fell back to prior weights",
            ],
            [
                [
                    r.config.label,
                    _dollars(r.config.nav),
                    f"{int(r.summary['rebalances'])}",
                    f"{int(r.summary['first_rung'])}",
                    f"{int(r.summary['inaccurate'])}",
                    f"{int(r.summary['relaxed'])}",
                    f"{int(r.summary['fell_back'])}",
                ]
                for r in results
            ],
        ),
        "",
    ]
    events = [
        (r.config.label, str(d.date()), str(a), str(s))
        for r in results
        for d, a, s in zip(r.rows.index, r.rows["attempt"], r.rows["status"], strict=True)
        if a != settings.model.optimizer.solver.primary
    ]
    if events:
        lines += [
            "Every rebalance that did not solve at the first rung, with the rung it reached:",
            "",
        ]
        lines += _table(["Configuration", "Date", "Rung", "Status"], events)
    else:
        lines += [
            "Every rebalance of every configuration solved at the first rung with the primary solver. The ladder was never entered."
        ]
    lines += [
        "",
        "## What the RSTR optimizer did (SPEC.md 8.5.1 ruling 7)",
        "",
        f"`N = {n}`. `cos theta = ||alpha_R|| / ||alpha||` per SPEC.md 8.3; effective assets is `1 / sum w^2` (13 at equal weight, 1 at a corner); a rebalance is a CORNER when at least `N - 2` = {n - 2} weights sit on the long-only floor.",
        "",
        *_table(
            [
                "Configuration",
                "median `cos theta`",
                "min `cos theta`",
                "median `gamma_risk`",
                "median `IR`",
                "mean effective assets",
                "median largest weight",
                "max largest weight",
                "mean assets at floor",
                "corner share",
            ],
            [
                [
                    r.config.label,
                    _num(r.summary["cos_theta_median"]),
                    _num(r.summary["cos_theta_min"]),
                    _num(r.summary["gamma_risk_median"], 1),
                    _num(r.summary["information_ratio_median"]),
                    _num(r.summary["effective_assets_mean"], 2),
                    _pct(r.summary["largest_weight_median"]),
                    _pct(r.summary["largest_weight_max"]),
                    _num(r.summary["at_floor_mean"], 1),
                    _pct(r.summary["corner_share"]),
                ]
                for r in rstr
            ],
        ),
        "",
    ]
    worst = max(rstr, key=lambda r: r.summary["corner_share"])
    if worst.summary["corner_share"] >= 0.5:
        lines.append(
            f"**A position limit, not alpha, is determining the output.** On {_pct(worst.summary['corner_share'])} of rebalances ({worst.config.label}) the solution is a corner of the simplex: the long-only floor `w >= 0` binds on at least {n - 2} of {n} assets and the book is one or two names. That is SPEC.md 8.5.1 ruling 7's anticipated finding -- the unconstrained optimum for this `alpha-hat` is concentrated, the optimizer loads on the part of `alpha-hat` the six factors cannot see (median `cos theta` {_num(worst.summary['cos_theta_median'])}), and it is the reason W6-P2 needs bounds as a grid dimension. No bound was added here to make the weights look reasonable."
        )
    elif worst.summary["any_at_floor_share"] >= 0.5:
        lines.append(
            f"**The long-only floor binds on most rebalances but the book is not a corner.** On {_pct(worst.summary['any_at_floor_share'])} of rebalances ({worst.config.label}) at least one weight is at zero, with {_num(worst.summary['at_floor_mean'], 1)} assets at the floor on average and {_num(worst.summary['effective_assets_mean'], 2)} effective assets; the floor shapes the output without reducing it to one or two names. Median `cos theta` {_num(worst.summary['cos_theta_median'])}."
        )
    else:
        lines.append(
            f"**Alpha, not a position limit, is determining the output.** The long-only floor binds on {_pct(worst.summary['any_at_floor_share'])} of rebalances at most ({worst.config.label}); the book holds {_num(worst.summary['effective_assets_mean'], 2)} effective assets on average. Median `cos theta` {_num(worst.summary['cos_theta_median'])}."
        )
    lead = rstr[0]
    lowest = str(lead.rows["lowest_volatility_asset"].mode().iloc[0])
    lowest_weight = float(lead.rows[f"w_{lowest}"].mean())
    lines += [
        "",
        f"**SPEC.md 8.4's initialisation is an initialisation, not the answer, and this run shows how far.** `lambda = IR / (2 TE_target)` with Grinold-Kahn's `IR = sqrt(alpha' Sigma^-1 alpha)` gives a median `gamma_risk` of {_num(lead.summary['gamma_risk_median'], 1)} from a median per-period `IR` of {_num(lead.summary['information_ratio_median'], 2)}. That `IR` is not a forecast of anything, and it decomposes: with `Sigma` replaced by its diagonal it is {_num(lead.summary['information_ratio_diagonal_median'], 2)}, so `Sigma^-1`'s exploitation of the near-collinear curve points accounts for a factor of {_num(lead.summary['information_ratio_median'] / lead.summary['information_ratio_diagonal_median'], 1)}; without the two identity assets it is {_num(lead.summary['information_ratio_without_identity_median'], 2)}, so they are NOT the source. The rest is the return-unit `alpha-hat` meeting an order-of-magnitude range of volatilities: the cross-sectional dispersion of `alpha-hat` is {_pct(lead.summary['alpha_dispersion_median'], 2)} per period, and the lowest-volatility asset (`{lowest}`) forecasts {_pct(lead.summary['lowest_volatility_median'], 2)} per period, so an alpha of ordinary size on it is a per-asset `IR` above 2 by itself -- which is why the book holds {_pct(lowest_weight)} of NAV in it on average. **This is the sixth published construction in the project measured out of its regime** (`reports/stage_k_dependence.md`, alongside Marchenko-Pastur denoising, the parabola smoothing, the specific-risk blend, stage (d) and the PSD repair): RSTR is CNE5's descriptor for a homogeneous-volatility equity universe, and applied across assets running from {_pct(lead.summary['lowest_volatility_median'], 2)} to several percent per period its return-unit alpha meets `Sigma^-1` on the curve block and dominates the book. That is ruling 7's corner solution with its cause located. `alpha-hat` is NOT changed: the concentration is the result, and W6-P2's per-asset bounds are the grid dimension that measures what it costs. A volatility-scaled alpha (Grinold's `IC * sigma * z`) would not do this and needs an IC, which ruling 4 declines. The unconstrained `h* = Sigma^-1 alpha / (2 lambda)` carries `TE_target` by construction; projected onto the long-only simplex, which removes the leverage that optimum needed, the book's forecast volatility is {_pct(lead.summary['forecast_volatility_rms'], 2)} per period against a target of {_pct(universe.te_target, 2)} -- {_pct(lead.summary['te_attainment'], 0)} of `TE_target`. SPEC.md 8.4's second step -- replace `lambda` by a risk constraint and let the solver recover it, or tune it -- is W6-P2's, and it is a `strategy-config` row when it runs.",
        "",
        "## Turnover, cost and the bias statistic on the optimizer's own book",
        "",
        f"Per holding period ({universe.horizon_days} sessions nominal), {int(rstr[0].summary['months'])} non-overlapping months. `B = sd(r_m / sigma_m)` with `sigma_m` the forecast standing at the start of the month (SPEC.md 6.1, clip +-{settings.model.validation.standardized_return_clip:g}); the exact chi-square interval at that `n` is beside it. Returns are excess of cash, annualised arithmetically; NO Sharpe ratio is reported (ruling 6).",
        "",
        *_table(
            [
                "Configuration",
                "one-way turnover /yr",
                "cost drag bp/yr",
                "spread share",
                "impact share",
                "gross excess /yr",
                "net excess /yr",
                "realised vol /period",
                "forecast vol rms /period",
                "`B`",
                "interval",
                "ADV cap binding",
                "final cash / NAV",
            ],
            [
                [
                    r.config.label,
                    _num(r.summary["turnover_annual"], 2),
                    _num(r.summary["cost_drag_bps_per_year"], 1),
                    _pct(r.summary["spread_share_of_cost"]),
                    _pct(r.summary["impact_share_of_cost"]),
                    _pct(r.summary["gross_annual"], 2),
                    _pct(r.summary["net_annual"], 2),
                    _pct(r.summary["realised_volatility"], 2),
                    _pct(r.summary["forecast_volatility_rms"], 2),
                    _num(r.summary["bias"]),
                    f"[{_num(r.summary['bias_lower'])}, {_num(r.summary['bias_upper'])}]",
                    _pct(r.summary.get("adv_binding_share", float("nan"))),
                    _pct(r.summary["final_cash_over_nav"], 2),
                ]
                for r in results
            ],
        ),
        "",
        "Negative final cash is the engine's stated convention -- costs are debited from cash and carried at zero interest (SPEC.md 7.4.2); W6 still owes the financing leg, and its size is the last column.",
        "",
        "## The `alpha = 0` control against SPEC.md 6.2's family 4",
        "",
        f"Cost-free, long-only minimum variance on the same forecast, {int(control.summary['rebalances'])} rebalances. Family 4 (W4-P2) is minimum variance fully invested WITH shorts allowed; the two coincide exactly on dates where the unconstrained solution has no negative weight.",
        "",
        *_table(
            [
                "Dates where family 4 holds a short",
                "Mean L1 distance to family 4",
                "`B` (monthly, long-only)",
                "interval",
                "realised vol /period",
                "forecast vol rms /period",
            ],
            [
                [
                    _pct(control.summary["family4_short_share"]),
                    _num(control.summary["family4_l1_mean"]),
                    _num(control.summary["bias"]),
                    f"[{_num(control.summary['bias_lower'])}, {_num(control.summary['bias_upper'])}]",
                    _pct(control.summary["realised_volatility"], 2),
                    _pct(control.summary["forecast_volatility_rms"], 2),
                ]
            ],
        ),
        "",
        "W4-P2's family-4 `B` of 1.3321 is a DAILY statistic on a portfolio rebuilt every day with shorts allowed; the number above is monthly, held for the month, long-only, and marked on the true excess returns rather than the regressands. They measure the same mechanism on different objects and are not expected to coincide to the digit.",
        "",
        "## Chart",
        "",
        f"![weights by sleeve, cos theta and effective assets]({_CHART_PATH.name}) -- `{rstr[0].config.label}`; the other three RSTR configurations are in `{_CSV_PATH.name}`.",
        "",
        "## Caveats carried from earlier sections",
        "",
        "- `govt_30y` trades through TLT, about HALF the exposure the risk model priced (SPEC.md 7.1.1); every 30y cost figure carries that caveat.",
        "- The issuer half-spread is a 2026 disclosure applied to 2009-2024 (SPEC.md 3.4.1's accepted anachronism) and an INTERVAL (SPEC.md 7.1.3); hence the band.",
        "- `TE_target` and the EDGE calm baseline are full-sample constants by ruling and are excluded from the causality test with that reason (SPEC.md 8.5.1 ruling 3; row 204).",
        "- `alpha-hat` is in return units by construction (normalised RSTR, centred, not scaled); the alternative and the reason it was not taken are at SPEC.md 8.5.1.",
        "- The book size is a rule, not a ruled NAV (SPEC.md 8.5.1's third construction); a ruled NAV replaces it through `optimizer.verification.book_size`.",
        "- `alpha_g` is not published here. Ruling 2 measures it from this strategy's realised gross return for the capacity curve in W6-P2/P3, with the caveat that it is what this signal did.",
        "",
    ]
    return "\n".join(lines)


def render_binding(
    universe: Universe, results: Sequence[RunResult], settings: config_mod.Config
) -> str:
    percentile = settings.model.optimizer.constraints.lagrange_percentile
    label = f"multiplier_p{round(percentile * 100)}"
    n = len(universe.assets)
    lines = [
        "# Binding constraints: which limits bound, how often, and the multipliers W6-P2 reads",
        "",
        "Generated by `python -m mafrm.backtest.verification`. SPEC.md 8.1's recipe -- hinge priorities from the Lagrange multipliers of the hard-constrained problem at about the 80th percentile -- needs those multipliers OBSERVED first (SPEC.md 8.5.1 ruling 5). This is the observation.",
        "",
        f"A hard limit BINDS when its slack is within the solver's feasibility tolerance ({settings.model.optimizer.solver.feasibility_tolerance:g}, Clarabel's default) of the bound. The percentile is taken over binding rebalances only -- a zero multiplier at a slack constraint says nothing about the price of tightness -- and the count of binding rebalances is beside it so a percentile of three observations is not read as a population statistic. The long-only floor `w_i >= 0` is counted as a limit too, because it is the one that determines the output when the book is a corner.",
        "",
    ]
    for r in results:
        table = con.summarise(r.readings, percentile=percentile)
        rows_out: list[list[str]] = []
        for name, row in table.to_dict(orient="index").items():
            rows_out.append(
                [
                    f"`{name}`",
                    str(row["kind"]),
                    f"{int(row['imposed'])}",
                    f"{int(row['binding'])}",
                    _pct(float(row["binding_share"])),
                    f"{int(row['relaxed'])}",
                    _num(float(row["multiplier_median"]), 5),
                    _num(float(row[label]), 5),
                    _num(float(row["multiplier_max"]), 5),
                ]
            )
        at_floor = r.rows["at_lower_bound"].to_numpy(dtype=int)
        rows_out.append(
            [
                f"`{_LONG_ONLY_FLOOR}` (any asset)",
                "hard",
                f"{len(at_floor)}",
                f"{int((at_floor > 0).sum())}",
                _pct(float((at_floor > 0).mean())),
                "0",
                "n/a",
                "n/a",
                "n/a",
            ]
        )
        rows_out.append(
            [
                f"`{_LONG_ONLY_FLOOR}` (corner: >= {n - 2} assets)",
                "hard",
                f"{len(at_floor)}",
                f"{int((at_floor >= n - 2).sum())}",
                _pct(float((at_floor >= n - 2).mean())),
                "0",
                "n/a",
                "n/a",
                "n/a",
            ]
        )
        lines += [
            f"## `{r.config.label}` -- NAV {_dollars(r.config.nav)}",
            "",
            *_table(
                [
                    "Limit",
                    "Kind",
                    "Imposed",
                    "Binding",
                    "Binding share",
                    "Relaxed",
                    "multiplier median",
                    f"multiplier p{round(percentile * 100)}",
                    "multiplier max",
                ],
                rows_out,
            ),
            "",
        ]
        adv_share = r.summary.get("adv_binding_share", float("nan"))
        corner = float((at_floor >= n - 2).mean())
        if corner >= 0.5:
            lines.append(
                f"**The long-only floor, a position limit, is determining the output on {_pct(corner)} of rebalances.** The ADV cap binds on {_pct(adv_share)}."
            )
        elif np.isfinite(adv_share) and adv_share >= 0.5:
            lines.append(
                f"**The ADV participation cap is determining the output on {_pct(adv_share)} of rebalances**; at this NAV the book cannot move into or out of the thin names at the pace alpha asks, and the multiplier percentile above is the price it would pay to. The long-only floor binds on {_pct(float((at_floor > 0).mean()))} of rebalances."
            )
        else:
            lines.append(
                f"Neither the ADV cap ({_pct(adv_share)} of rebalances) nor a corner of the simplex ({_pct(corner)}) determines the output here; the long-only floor touches at least one asset on {_pct(float((at_floor > 0).mean()))} of rebalances."
            )
        lines.append("")
    lines += [
        "## What W6-P2 does with this",
        "",
        f"The `{label}` column of `adv_participation` at each NAV is the hinge priority SPEC.md 8.1's recipe sets when the cap becomes soft; the priority is in objective units per unit of `|z|` beyond the cap, i.e. per-period return. Where the cap never bound the column is `n/a` and there is nothing to bootstrap from -- the constraint stays hard there, as ruling 5 says.",
        "",
    ]
    return "\n".join(lines)


def chart(universe: Universe, result: RunResult, path: Path) -> None:
    """Weights by sleeve (stacked), `cos theta`, and effective assets, one column of small multiples."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink_2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
    # The validated categorical order (dataviz palette, light mode), one slot per sleeve
    # in the universe file's own order. Five sleeves hold assets, so five slots.
    slots = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    order: list[str] = []
    for asset in universe.assets:
        sleeve = universe.sleeves[asset]
        if sleeve not in order:
            order.append(sleeve)
    weights = result.rows[[f"w_{a}" for a in universe.assets]]
    weights.columns = list(universe.assets)
    by_sleeve = weights.T.groupby(pd.Series(universe.sleeves)).sum().T[order]
    x = by_sleeve.index

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, height_ratios=[3, 1.2, 1.2])
    fig.patch.set_facecolor(surface)
    ax0, ax1, ax2 = axes
    ax0.stackplot(
        x,
        [by_sleeve[s].to_numpy(dtype=float) for s in order],
        labels=order,
        colors=slots[: len(order)],
        edgecolor=surface,
        linewidth=1.0,
    )
    ax0.set_ylim(0, 1)
    ax0.set_ylabel("weight by sleeve", color=ink_2)
    ax0.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=len(order),
        frameon=False,
        fontsize=9,
        labelcolor=ink_2,
    )
    ax1.plot(x, result.rows["cos_theta"].to_numpy(dtype=float), color=slots[0], linewidth=2)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("cos theta", color=ink_2)
    ax2.plot(x, result.rows["effective_assets"].to_numpy(dtype=float), color=slots[0], linewidth=2)
    ax2.set_ylim(0, len(universe.assets))
    ax2.set_ylabel("effective assets", color=ink_2)
    for ax in axes:
        ax.set_facecolor(surface)
        ax.grid(axis="y", color=grid, linewidth=1.0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(colors=ink_2, labelsize=9)
    fig.suptitle(
        f"{result.config.label}: monthly weights by sleeve, misalignment angle and concentration",
        color=ink,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.93,
        "SPEC.md 8, W6-P1 verification run. Long-only, fully invested; RSTR alpha; the six-factor forecast. In-sample only.",
        color=ink_2,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - entry point
    parser = argparse.ArgumentParser(
        description="SPEC.md 8's optimizer: the W6-P1 verification run"
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    settings = config_mod.load()
    universe = build_universe(settings)
    print(
        f"panel {universe.panel.dates[0].date()}..{universe.terminal_date.date()}, "
        f"{len(universe.rebalance_dates)} rebalances, TE_target {universe.te_target:.4%}/period, "
        f"book sizes {book_sizes(universe, settings)}"
    )
    results = run_configurations(universe, settings, progress=not args.quiet)
    pd.concat([r.rows for r in results]).to_csv(_CSV_PATH, float_format="%.10g")
    _MARKDOWN_PATH.write_text(render(universe, results, settings), encoding="utf-8")
    _BINDING_PATH.write_text(render_binding(universe, results, settings), encoding="utf-8")
    chart(universe, results[0], _CHART_PATH)
    for r in results:
        s = r.summary
        print(
            f"  {r.config.label}: rung0 {int(s['first_rung'])}/{int(s['rebalances'])}, relaxed "
            f"{int(s['relaxed'])}, fell back {int(s['fell_back'])}, cos theta median "
            f"{s['cos_theta_median']:.3f}, eff. assets {s['effective_assets_mean']:.2f}, corner share "
            f"{s['corner_share']:.1%}, turnover {s['turnover_annual']:.2f}/yr, cost "
            f"{s['cost_drag_bps_per_year']:.1f} bp/yr, B {s['bias']:.3f}"
        )
    for path in (_CSV_PATH, _MARKDOWN_PATH, _BINDING_PATH, _CHART_PATH):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
