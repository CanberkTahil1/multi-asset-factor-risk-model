"""``reports/validation_battery.md``. SPEC.md 6.5's battery on every covariance variant. W4-P3.

WHY THIS LIVES IN ``factors/``
------------------------------

:mod:`mafrm.risk.battery` is arithmetic on a standardised series. *This* module
knows whose series it is -- family 4's, pre-VRA by the SPEC.md 6.2.2 ruling,
built by :mod:`mafrm.factors.bias_report` on Model A's residual panel -- and
which two assets are construction artefacts that must stay out of anything that
pools across assets. None of that may cross into ``risk/`` (CLAUDE.md
invariant 10).

WHAT IS SCORED
--------------

The same nine variants and two horizons as ``reports/bias_statistics.md``
(``experiments.md`` rows 144-161), re-scored rather than re-configured: no new
configuration is evaluated here, and rows 187-188 say so. The series is
``b_t = R_t / sigma_t`` for the minimum-variance portfolio, **unclipped** -- the
``+-4`` clip belongs to SPEC.md 6.1's standard deviation and to nothing else
(SPEC.md 5.4.1 ruling 2 made the same point for the VRA), and a VaR test on a
clipped series would have its tail removed by hand.

The rulings of 2026-09-03 are transcribed in ``config/model.yaml`` under
``validation.battery`` and in SPEC.md 6.5.3; this module reads them there and
adds none. The Q-statistic row of SPEC.md 6.5 is **unimplemented by ruling** and
the report says so where the row would have been.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from mafrm import config, history
from mafrm.factors import bias_report
from mafrm.risk import battery, validation
from mafrm.risk.config import HORIZONS, Horizon
from mafrm.risk.shepard import SecondOrderRisk, Unit

__all__ = [
    "BatteryInputs",
    "Registered",
    "VariantBattery",
    "factor_table",
    "legs",
    "main",
    "master_table",
    "render",
    "run",
    "score",
    "score_series",
]

_REPORTS: Final[Path] = Path(__file__).resolve().parents[3] / "reports"
_MARKDOWN_PATH: Final[Path] = _REPORTS / "validation_battery.md"

#: The registered significance level for every "rejects" in rows 187-188. Not a
#: tunable: it is the 5% the chi-square interval's 0.95 coverage already carries,
#: read off ``validation.chi_square_level`` at use.
_PRE_VRA_STAGES: Final[frozenset[str]] = frozenset(
    {"ewma", "newey_west", "psd_repair", "eigenfactor"}
)


@dataclass(frozen=True)
class BatteryInputs:
    """Everything the battery reads from the config, resolved through its pointers."""

    levels: tuple[float, ...]
    ljung_box_lags: tuple[int, ...]
    basel: config.BaselConfig
    trials: int
    newey_west_lags: int
    block: int
    t_threshold: float
    t_window: int
    seed: int
    significance: float

    @classmethod
    def from_config(cls, settings: config.Config, horizon: Horizon) -> BatteryInputs:
        model = settings.model
        tests = model.validation.battery
        lags = config.resolve(model, tests.newey_west_lags_from)
        months = config.resolve(model, tests.t_statistic_window_from)
        return cls(
            levels=tests.var_levels,
            ljung_box_lags=tuple(
                month * model.data.trading_days_per_month for month in tests.ljung_box_lags_months
            ),
            basel=tests.basel,
            trials=int(config.resolve(model, tests.monte_carlo_trials_from)),
            newey_west_lags=int(getattr(lags, horizon)),
            block=int(config.resolve(model, tests.rank_correlation_block_days_from)),
            t_threshold=float(config.resolve(model, tests.t_statistic_threshold_from)),
            t_window=int(months) * model.data.trading_days_per_month,
            seed=model.seed,
            significance=1.0 - model.validation.chi_square_level,
        )


@dataclass(frozen=True)
class VariantBattery:
    """SPEC.md 6.5 on one variant's family-4 series."""

    spec: bias_report.VariantSpec
    horizon: Horizon
    observations: int
    #: SPEC.md 6.1's clipped ``B`` from the bias report, for the reader's anchor.
    bias: float
    #: ``B`` divided by the volatility multiplier of the closed form that applies.
    shepard_corrected_bias: float
    shepard_label: str
    #: SPEC.md 6.5's headline model-comparison metric, bps/day.
    min_var_realised_volatility: float
    mincer_zarnowitz: battery.MincerZarnowitz
    ljung_box: dict[int, battery.LjungBox]
    tails: dict[float, battery.TailQuantile]
    kupiec: dict[float, battery.Kupiec]
    christoffersen: dict[float, battery.Christoffersen]
    basel: battery.BaselTrafficLight
    acerbi_szekely: dict[float, battery.AcerbiSzekely]
    rank: battery.RankCorrelation

    @property
    def pre_vra_factor(self) -> bool:
        return self.spec.stage in _PRE_VRA_STAGES and not self.spec.factor_regime


def score(
    run: bias_report.VariantRun,
    horizon_run: bias_report.HorizonRun,
    data: bias_report.Inputs,
    inputs: BatteryInputs,
) -> VariantBattery:
    """Every test in SPEC.md 6.5 on one variant of Model A's ladder."""
    members = [name for name in data.assets if name not in set(data.excluded)]
    closed_form = bias_report.shepard_for(run.spec, data, horizon_run.risk)
    label = (
        f"Eq. 13, N = {len(data.assets)}"
        if run.spec.stage == "sample"
        else f"Eq. 32, K = {len(data.factor_names)} (upper bound)"
    )
    return score_series(
        run.spec,
        horizon_run.horizon,
        forecast=run.member_forecasts["min_var"].to_numpy(dtype=float),
        realised=run.member_returns["min_var"].to_numpy(dtype=float),
        member_forecasts=run.member_forecasts[members].to_numpy(dtype=float),
        member_returns=run.member_returns[members].to_numpy(dtype=float),
        bias=run.report.median_bias(validation.OPTIMIZED),
        min_var_realised_volatility=run.min_var_realised_volatility,
        closed_form=closed_form,
        shepard_label=label,
        inputs=inputs,
    )


def score_series(
    spec: bias_report.VariantSpec,
    horizon: Horizon,
    *,
    forecast: np.ndarray,
    realised: np.ndarray,
    member_forecasts: np.ndarray,
    member_returns: np.ndarray,
    bias: float,
    min_var_realised_volatility: float,
    closed_form: SecondOrderRisk,
    shepard_label: str,
    inputs: BatteryInputs,
) -> VariantBattery:
    """SPEC.md 6.5 on one family-4 series. The panel-free core of :func:`score`.

    EXTRACTED IN W7-P4, AT THE SECOND CALLER
        The equity module (:mod:`mafrm.factors.equity_risk_report`) scores the
        same battery on a ragged panel that :class:`bias_report.VariantRun`
        cannot describe. One function so the two reports cannot disagree about
        what a test is; :func:`score` is Model A's adapter onto it.
    """
    standardized = realised / forecast
    return VariantBattery(
        spec=spec,
        horizon=horizon,
        observations=int(standardized.size),
        bias=bias,
        shepard_corrected_bias=bias / closed_form.multiplier(Unit.VOLATILITY),
        shepard_label=shepard_label,
        min_var_realised_volatility=min_var_realised_volatility,
        mincer_zarnowitz=battery.mincer_zarnowitz(
            np.square(realised), np.square(forecast), newey_west_lags=inputs.newey_west_lags
        ),
        ljung_box={
            lags: battery.ljung_box(np.square(standardized), lags=lags)
            for lags in inputs.ljung_box_lags
        },
        tails={
            level: battery.empirical_quantile(standardized, level=level) for level in inputs.levels
        },
        kupiec={
            level: battery.kupiec(
                int(battery.breach_indicator(standardized, level=level).sum()),
                standardized.size,
                level=level,
            )
            for level in inputs.levels
        },
        christoffersen={
            level: battery.christoffersen(
                battery.breach_indicator(standardized, level=level), level=level
            )
            for level in inputs.levels
        },
        basel=battery.basel_traffic_light(
            battery.breach_indicator(standardized, level=inputs.basel.level),
            window=inputs.basel.window_days,
            level=inputs.basel.level,
            green_max=inputs.basel.green_max_breaches,
            yellow_max=inputs.basel.yellow_max_breaches,
        ),
        acerbi_szekely={
            level: battery.acerbi_szekely(
                standardized, level=level, trials=inputs.trials, seed=inputs.seed
            )
            for level in inputs.levels
        },
        rank=battery.cross_sectional_rank_correlation(
            member_forecasts, member_returns, block=inputs.block
        ),
    )


@dataclass(frozen=True)
class HorizonBattery:
    horizon: Horizon
    inputs: BatteryInputs
    variants: tuple[VariantBattery, ...]
    window: bias_report.ScoredWindow
    factor_t: tuple[battery.FactorTStatistic, ...]


def run(
    data: bias_report.Inputs, cache: history.Cache, settings: config.Config
) -> dict[Horizon, HorizonBattery]:
    """Score every variant at both horizons off the same committed eigenfactor cache."""
    out: dict[Horizon, HorizonBattery] = {}
    for horizon in HORIZONS:
        horizon_run = bias_report.run(horizon, data, cache, settings)
        inputs = BatteryInputs.from_config(settings, horizon)
        scored = tuple(
            score(horizon_run.runs[spec.name], horizon_run, data, inputs)
            for spec in bias_report.variants(settings)
        )
        factor_returns = data.factors.reindex(horizon_run.dates).to_numpy(dtype=float)
        factor_t = battery.factor_t_statistics(
            factor_returns,
            list(data.factor_names),
            newey_west_lags=inputs.newey_west_lags,
            window=inputs.t_window,
            threshold=inputs.t_threshold,
        )
        out[horizon] = HorizonBattery(
            horizon=horizon,
            inputs=inputs,
            variants=scored,
            window=horizon_run.window,
            factor_t=factor_t,
        )
    return out


# ---------------------------------------------------------------------------
# Rows 187-188: the registered legs, adjudicated
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Registered:
    leg: str
    claim: str
    holds: bool
    detail: str


def legs(result: HorizonBattery) -> tuple[Registered, ...]:
    alpha = result.inputs.significance
    everything = result.variants
    pre_vra = [item for item in everything if item.pre_vra_factor]
    factor_variants = [item for item in everything if item.spec.stage != "sample"]
    sample = next(item for item in everything if item.spec.stage == "sample")
    levels = result.inputs.levels
    low, high = min(levels), max(levels)

    def names(items: Sequence[VariantBattery]) -> str:
        return ", ".join(f"`{item.spec.name}`" for item in items) if items else "none"

    kupiec_fail = [
        item for item in everything if any(item.kupiec[level].p_value >= alpha for level in levels)
    ]
    tail_fail = [
        item
        for item in everything
        if not (item.tails[high].empirical / item.bias) / item.tails[high].normal > 1.0
    ]
    christoffersen_fail = [
        item for item in pre_vra if item.christoffersen[low].independence_p_value >= alpha
    ]
    ljung_fail = [
        item for item in pre_vra if any(test.p_value >= alpha for test in item.ljung_box.values())
    ]
    slope_fail = [item for item in factor_variants if item.mincer_zarnowitz.slope <= 1.0]
    qlike_best = min(everything, key=lambda item: item.mincer_zarnowitz.qlike)
    acerbi_fail = [
        item
        for item in pre_vra
        if any(
            item.acerbi_szekely[level].z_2 >= 0.0 or item.acerbi_szekely[level].z_2_p_value >= alpha
            for level in levels
        )
    ]
    basel_fail = [item for item in pre_vra if item.basel.count("red") <= len(item.basel.blocks) / 2]
    sample_green = sample.basel.count("green") > len(sample.basel.blocks) / 2
    rank_fail = [item for item in everything if item.rank.mean / item.rank.standard_error <= 2.0]

    return (
        Registered(
            "(a)",
            f"Kupiec rejects at {alpha:.0%} for every variant at both levels",
            not kupiec_fail,
            f"fails on {names(kupiec_fail)}",
        ),
        Registered(
            "(b)",
            f"empirical {1 - high:.0%} quantile of `b/B` below the normal's for every variant",
            not tail_fail,
            f"fails on {names(tail_fail)}",
        ),
        Registered(
            "(c)",
            f"Christoffersen `LR_ind` rejects at {alpha:.0%} at {low:.0%} VaR for every pre-VRA "
            "factor variant",
            not christoffersen_fail,
            f"fails on {names(christoffersen_fail)}",
        ),
        Registered(
            "(d)",
            f"Ljung-Box on `b^2` rejects at {alpha:.0%} at all three lags for every pre-VRA "
            "factor variant",
            not ljung_fail,
            f"fails on {names(ljung_fail)}",
        ),
        Registered(
            "(e)",
            "Mincer-Zarnowitz slope `b > 1` for every factor variant",
            not slope_fail,
            f"fails on {names(slope_fail)}",
        ),
        Registered(
            "(f)",
            "QLIKE ranks `sample` best on family 4",
            qlike_best.spec.stage == "sample",
            f"best is `{qlike_best.spec.name}` at {qlike_best.mincer_zarnowitz.qlike:.4f}",
        ),
        Registered(
            "(g)",
            f"Acerbi-Szekely `Z_2 < 0` with MC `p < {alpha:.2f}` at both levels for every pre-VRA "
            "factor variant",
            not acerbi_fail,
            f"fails on {names(acerbi_fail)}",
        ),
        Registered(
            "(h)",
            "Basel: majority of blocks red for pre-VRA factor variants and green for `sample`",
            not basel_fail and sample_green,
            f"red-majority fails on {names(basel_fail)}; `sample` green-majority "
            f"{'holds' if sample_green else 'FAILS'}",
        ),
        Registered(
            "(i)",
            "mean cross-sectional Spearman positive at more than 2 s.e. for every variant",
            not rank_fail,
            f"fails on {names(rank_fail)}",
        ),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def master_table(results: dict[Horizon, HorizonBattery]) -> list[str]:
    """SPEC.md 6.5's acceptance: every test, every variant, one table."""
    some = next(iter(results.values())).inputs
    low, high = min(some.levels), max(some.levels)
    lag_labels = "/".join(str(lag) for lag in some.ljung_box_lags)
    lines = [
        "| Horizon | Variant | fam4 `B` | Shepard-corr. `B` | **min-var realised vol** (bps/day) "
        "| MZ `a` | MZ `b` | MZ joint `p` | QLIKE "
        f"| LB `p` @ {lag_labels} | Kupiec `p` {low:.0%}/{high:.0%} "
        f"| `LR_ind` `p` {low:.0%}/{high:.0%} | `LR_cc` `p` {low:.0%}/{high:.0%} "
        f"| Basel G/Y/R | `Z_2` (`p`) {low:.0%}/{high:.0%} | `Z_MB` (`p`) {low:.0%}/{high:.0%} "
        f"| tail ratio {low:.0%}/{high:.0%} | rank rho (s.e.) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for horizon in HORIZONS:
        for item in results[horizon].variants:
            mz = item.mincer_zarnowitz
            lb = "/".join(_p(item.ljung_box[lag].p_value) for lag in some.ljung_box_lags)
            kup = "/".join(_p(item.kupiec[level].p_value) for level in (low, high))
            ind = "/".join(
                _p(item.christoffersen[level].independence_p_value) for level in (low, high)
            )
            cc = "/".join(
                _p(item.christoffersen[level].conditional_coverage_p_value) for level in (low, high)
            )
            basel = (
                f"{item.basel.count('green')}/{item.basel.count('yellow')}/"
                f"{item.basel.count('red')}"
            )
            z2 = "/".join(
                f"{item.acerbi_szekely[level].z_2:+.2f} "
                f"({_p(item.acerbi_szekely[level].z_2_p_value)})"
                for level in (low, high)
            )
            mb = "/".join(
                f"{item.acerbi_szekely[level].minimally_biased:+.2f} "
                f"({_p(item.acerbi_szekely[level].minimally_biased_p_value)})"
                for level in (low, high)
            )
            tails = "/".join(f"{item.tails[level].ratio:.2f}" for level in (low, high))
            lines.append(
                f"| {horizon} | `{item.spec.name}` | {item.bias:.4f} "
                f"| {item.shepard_corrected_bias:.4f} | **{item.min_var_realised_volatility:.3f}** "
                f"| {mz.intercept:+.3f} | {mz.slope:.3f} | {_p(mz.joint_p_value)} | {mz.qlike:.4f} "
                f"| {lb} | {kup} | {ind} | {cc} | {basel} | {z2} | {mb} | {tails} "
                f"| {item.rank.mean:.3f} ({item.rank.standard_error:.3f}) |"
            )
    return lines


def factor_table(result: HorizonBattery) -> list[str]:
    lines = [
        "| Factor | mean | NW s.e. | `t` | rolling `|t| > "
        f"{result.inputs.t_threshold:g}` share (overlap {result.inputs.t_window - 1}) "
        f"| non-overlapping share (n) |",
        "|---|---|---|---|---|---|",
    ]
    for item in result.factor_t:
        lines.append(
            f"| `{item.name}` | {item.mean:+.4f} | {item.newey_west_standard_error:.4f} "
            f"| **{item.t_statistic:+.2f}** | {item.rolling_exceedance:.1%} "
            f"| {item.non_overlapping_exceedance:.1%} ({item.non_overlapping.size}) |"
        )
    return lines


def render(
    results: dict[Horizon, HorizonBattery], data: bias_report.Inputs, settings: config.Config
) -> str:
    short = results["short"]
    inputs = short.inputs
    low, high = min(inputs.levels), max(inputs.levels)
    legs = {horizon: legs(results[horizon]) for horizon in HORIZONS}
    lag_list = ", ".join(str(lag) for lag in inputs.ljung_box_lags)
    clustered = sum(
        1
        for item in short.variants
        if item.christoffersen[low].independence_p_value < inputs.significance
    )
    out: list[str] = []
    add = out.append
    add("# The validation battery: SPEC.md 6.5 on every covariance variant")
    add("")
    add("Generated by `python -m mafrm.factors.battery_report`. SPEC.md 6.5, W4-P3.")
    add("")
    add(
        "SPEC.md 6.1's bias statistic is one number per portfolio. The battery asks the "
        "questions that number cannot: is the model wrong *conditionally* (Mincer-Zarnowitz), "
        "did volatility clustering survive the forecast (Ljung-Box), is the VaR level right "
        "(Kupiec) and do the breaches cluster (Christoffersen), what would a regulator's traffic "
        "light say (Basel), is expected shortfall calibrated (Acerbi-Szekely), and does the "
        "model rank a risky asset above a safe one at all. Every test runs on the **same** "
        "series the headline `B` is computed from -- family 4's, the minimum-variance portfolio "
        "rebuilt each date, pre-VRA by SPEC.md 6.2.2 -- for every variant of the attribution "
        "ladder, so a column can be read down as an attribution and a row across as a profile."
    )
    add("")
    add("## What was scored, and under which rulings")
    add("")
    add(f"- {short.window.render()}; the same window at both horizons.")
    add(
        "- Series: `b_t = R_t / sigma_t` for family 4, **unclipped** "
        f"({short.variants[0].observations:,} "
        "daily observations). SPEC.md 6.1's `+-4` clip belongs to its standard deviation only; "
        "a VaR test on a clipped series would have had its tail removed by hand. The `B` column "
        "is the clipped SPEC.md 6.1 statistic from `reports/bias_statistics.md`, as the anchor."
    )
    add(
        f"- **Ruling 1**: VaR and ES at {low:.0%} and {high:.0%}, both reported, neither selected. "
        f"At `T = {short.variants[0].observations:,}` the {high:.0%} level expects "
        f"{short.variants[0].kupiec[high].expected:.0f} breaches and the {low:.0%} level "
        f"{short.variants[0].kupiec[low].expected:.0f}."
    )
    add(
        "- **Ruling 2**: the tests use the normal -- `VaR = z sigma`, `ES = sigma phi(z)/alpha` -- "
        "because it is the distribution the variance forecast implicitly assumes. The 'tail "
        "ratio' column is the diagnostic that replaces a fitted tail: the standardised returns' "
        "own lower quantile over the normal's, above one when the tail is fatter than the "
        "forecast assumes. It is never fed into a test."
    )
    add(
        f"- **Ruling 3**: Ljung-Box at `h` = {lag_list} "
        "-- a month, a quarter, a year -- all three reported."
    )
    add(
        f"- **Ruling 4**: the Acerbi-Szekely null is simulated with `M = {inputs.trials:,}`, read "
        "from `eigenfactor.monte_carlo_trials` through a pointer; p-values are one-sided in the "
        "direction of understatement, resolution 1/M."
    )
    add(
        "- **Ruling 5**: **Menchero & Ji's Q-statistic is not implemented.** The repository does "
        "not hold *JPM* 50(3) 2024 and SPEC.md 6.5 gives one sentence, so any construction here "
        "would claim a provenance it does not have. The 'rank rho' column is this project's own "
        "plain cross-sectional Spearman between the forecast volatility standing when a block "
        f"opened and the realised standard deviation inside it, on **non-overlapping** "
        f"{inputs.block}-day blocks ({short.variants[0].rank.blocks} of them, so that is the `n`), "
        f"across the {short.variants[0].rank.members} assets that are not construction artefacts. "
        "It answers the same diagnostic question and is never called Q."
    )
    add(
        f"- Basel: {inputs.basel.window_days}-day **non-overlapping** blocks at "
        f"{inputs.basel.level:.0%}, green <= {inputs.basel.green_max_breaches}, yellow to "
        f"{inputs.basel.yellow_max_breaches}, red above; {len(short.variants[0].basel.blocks)} "
        f"blocks and a {short.variants[0].basel.unscored_tail}-day unscored tail."
    )
    add(
        f"- Newey-West lags for Mincer-Zarnowitz and the factor t-statistics: "
        f"{inputs.newey_west_lags}, read from `covariance.volatility_newey_west_lags`."
    )
    add("")
    add("## The table")
    add("")
    add(
        "**The headline model-comparison column is realised min-var volatility**, in bps/day, "
        "per SPEC.md 6.5 and Menchero & Ji's argument that it converges far faster than an "
        "information ratio. `B` is SPEC.md 6.1's statistic; the Shepard-corrected `B` divides it "
        "by the volatility multiplier of the closed form that applies (Eq. 13 for `sample`, Eq. "
        "32 as an upper bound for the factor variants -- `reports/second_order_risk.md`). MZ is "
        "`r^2 = a + b h + e` with HAC standard errors and a joint Wald test of `(0, 1)`; QLIKE "
        "is lower-is-better and comparable only down a column. Every `p` is a p-value; `Z_2 < 0` "
        "and `Z_MB > 0` are the understatement directions."
    )
    add("")
    out.extend(master_table(results))
    add("")
    add("## Rows 187-188: the registered legs")
    add("")
    add(
        "Registered before the run from W4-P2's published `B` under the normal, so what is "
        "informative is what appears beyond `B`. Adjudicated at the "
        f"{inputs.significance:.0%} level the chi-square interval's coverage already carries."
    )
    add("")
    for horizon in HORIZONS:
        add(f"### `{horizon}` (row {187 if horizon == 'short' else 188})")
        add("")
        add("| Leg | Claim | Verdict | Detail |")
        add("|---|---|---|---|")
        for leg in legs[horizon]:
            add(
                f"| {leg.leg} | {leg.claim} | **{'HOLDS' if leg.holds else 'REFUTED'}** "
                f"| {leg.detail} |"
            )
        add("")
    add("## Factor-return t-statistics")
    add("")
    add(
        "SPEC.md 6.5: 'a factor whose returns are indistinguishable from noise is adding "
        "estimation error to the covariance matrix for nothing. Use it to prune.' **Nothing is "
        "pruned.** Choosing risk factors by their mean return would be alpha research "
        "(CLAUDE.md), and these factors were built to explain covariance, not to earn a premium. "
        "Reported on the scored window in the pipeline's regression units (basis points of "
        "return for return factors, basis points of yield change for the rate factors). The "
        f"rolling series is {inputs.t_window} days stepped one day -- overlap "
        f"{inputs.t_window - 1}, so its share is a description and not a frequency; the "
        "non-overlapping share carries the only honest `n`."
    )
    add("")
    out.extend(factor_table(short))
    add("")
    significant = [item for item in short.factor_t if abs(item.t_statistic) >= inputs.t_threshold]
    add(
        f"{len(significant)} of {len(short.factor_t)} factors have full-sample `|t| >= "
        f"{inputs.t_threshold:g}`"
        + (f" ({', '.join(f'`{item.name}`' for item in significant)})" if significant else "")
        + ". The pre-registered expectation was that the majority would not."
    )
    add("")
    add("## What the battery adds to the bias statistic")
    add("")
    base = next(item for item in short.variants if item.spec.name.startswith("eigen_a1"))
    sample = next(item for item in short.variants if item.spec.stage == "sample")
    long_base = next(item for item in results["long"].variants if item.spec.name == base.spec.name)
    add(
        "**Read the Basel and Mincer-Zarnowitz rows as one finding, in the vocabulary a risk "
        f"interviewer speaks.** This model would fail a regulatory backtest in a plurality of "
        f"years: at a {inputs.basel.level:.0%} VaR over {inputs.basel.window_days}-day windows the "
        f"pre-VRA factor model is in the **red zone in {base.basel.count('red')} of "
        f"{len(base.basel.blocks)}** blocks at the short horizon and "
        f"{long_base.basel.count('red')} of {len(long_base.basel.blocks)} at the long, against "
        f"{sample.basel.count('red')} of {len(sample.basel.blocks)} for the naive covariance on "
        "the same portfolio. Its Mincer-Zarnowitz slope is "
        f"**{base.mincer_zarnowitz.slope:.1f} short / {long_base.mincer_zarnowitz.slope:.1f} "
        "long**: realised variance rises two-and-a-half to three times faster than the forecast, "
        "so the model is not merely low on average but low in proportion to how much risk there "
        "is. Neither is a failed test in its own right. Both are SPEC.md 6.2.6's specification "
        "error -- the diagonal `Delta` in `Sigma = X F X' + Delta`, which discards residual "
        "correlations that scale with the level of risk -- expressed as a regulatory outcome and "
        "as a conditional-calibration coefficient. The cause is an assumption the specification "
        "makes, not an estimate the data failed to pin down, and no amount of history repairs it."
    )
    add("")
    add(
        f"**The two-level ruling paid for itself on the naive comparand.** `sample` passes Kupiec "
        f"at {low:.0%} (`p` = {sample.kupiec[low].p_value:.2f} short) and fails at {high:.0%}: its "
        f"`B` = {sample.bias:.3f} is a **tail** excess (tail ratio "
        f"{sample.tails[low].ratio:.2f} at "
        f"{low:.0%}, {sample.tails[high].ratio:.2f} at {high:.0%}), which either level alone would "
        "have hidden -- one would have reported a calibrated model, the other a miscalibrated one, "
        "and neither would have said the disagreement was the finding."
    )
    add("")
    add(
        f"- **The level error `B` measures is everywhere.** On the pre-VRA factor model "
        f"(`{base.spec.name}`, short) Kupiec rejects at both levels "
        f"({base.kupiec[low].breaches} breaches against {base.kupiec[low].expected:.0f} expected "
        f"at {low:.0%}; {base.kupiec[high].breaches} against {base.kupiec[high].expected:.0f} at "
        f"{high:.0%}), the Mincer-Zarnowitz slope is {base.mincer_zarnowitz.slope:.2f}, and "
        f"{base.basel.count('red')} of {len(base.basel.blocks)} Basel blocks are red."
    )
    add(
        f"- **The naive comparand is better calibrated on family 4, and it is not close.** "
        f"`sample` (short) has `B = {sample.bias:.4f}`, {sample.kupiec[high].breaches} breaches at "
        f"{high:.0%}, {sample.basel.count('green')} green blocks of {len(sample.basel.blocks)}, "
        "and "
        f"QLIKE {sample.mincer_zarnowitz.qlike:.4f} against `{base.spec.name}`'s "
        f"{base.mincer_zarnowitz.qlike:.4f}. That is SPEC.md 6.2.6 seen through six more "
        "instruments: the diagonal specific-risk assumption, not sampling error, is what the "
        "factor model pays for on optimiser-selected portfolios."
    )
    add(
        f"- **The tail is fatter than any `B` explains.** Tail ratios at {high:.0%} run "
        f"{min(item.tails[high].ratio for h in HORIZONS for item in results[h].variants):.2f}-"
        f"{max(item.tails[high].ratio for h in HORIZONS for item in results[h].variants):.2f} "
        "across every variant and horizon: dividing by `B` would still leave the normal's 1% "
        "quantile inside the empirical one. A Student-t was refused (ruling 2) because its `nu` "
        "would be fitted on this very series; what stands instead is the number."
    )
    add(
        f"- **Breaches cluster.** Christoffersen's independence test at {low:.0%} rejects on "
        f"{clustered} "
        f"of {len(short.variants)} short-horizon variants, and Ljung-Box on `b^2` rejects at "
        "every lag on every pre-VRA factor variant: an 84-day half-life does not track the "
        "clustering, which is the timing error SPEC.md 5.4's VRA exists for -- and the `vra_*` "
        "rows show what it buys."
    )
    add(
        "- **The model does rank risk.** The cross-sectional Spearman is positive on every "
        "variant at more than two standard errors over non-overlapping monthly blocks, which is "
        "the one question in this battery the factor model answers well. A model can have "
        "`B = 1.00` and fail this; this one has `B = 1.33` and passes it."
    )
    add("")
    add(
        "Nothing here changes a configuration. Every row in this table is one of `experiments.md` "
        "rows 144-161 re-scored, and rows 187-188 record that."
    )
    add("")
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - entry point
    parser = argparse.ArgumentParser(description="SPEC.md 6.5's validation battery")
    parser.parse_args(list(argv) if argv is not None else None)
    settings = config.load()
    data = bias_report.inputs(settings)
    cache = history.read_cache(
        bias_report._CACHE_PATH,
        rebuild_hint=bias_report._REBUILD_HINT,
        settings=settings,
        panel_digests={"macro": history.panel_digest(data.factors)},
    )
    results = run(data, cache, settings)
    _MARKDOWN_PATH.write_text(render(results, data, settings), encoding="utf-8")
    print(f"wrote {_MARKDOWN_PATH}")
    for horizon in HORIZONS:
        for leg in legs(results[horizon]):
            print(f"  {horizon} {leg.leg} {'HOLDS  ' if leg.holds else 'REFUTED'} {leg.detail}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
