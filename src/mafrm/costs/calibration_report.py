"""Generates ``reports/cost_calibration.md``. SPEC.md 7.2.

The derivation of both square-root prefactors from the published anchors, the
SPEC.md 7.3 scaling laws at the configured exponent, the spread construction
rule with each asset's measured EDGE baseline, and the state of the issuer
level file. Every number in the first two sections is arithmetic on
``config/model.yaml`` and needs no data; the third reads the cache and says so
if it cannot.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mafrm import config as config_mod
from mafrm.costs import impact, inputs, spread_level
from mafrm.data import cache

__all__ = [
    "AssetResolution",
    "Derivation",
    "ResolutionArithmetic",
    "derive",
    "main",
    "render",
    "report_path",
    "resolution_arithmetic",
    "resolution_table",
]

#: The two reference participations of the interval ruling: a large trade at the
#: AQR anchor's 2% of ADV and a small one at 0.1%. Presentation only.
_LARGE_PARTICIPATION = 0.02
_SMALL_PARTICIPATION = 0.001

_BPS = 1e-4


def report_path() -> Path:
    return config_mod._REPO_ROOT / "reports" / "cost_calibration.md"


@dataclass(frozen=True)
class ImpliedPoint:
    regime: str
    source: str
    participation: float
    cost_bps: float
    implied_y: float


@dataclass(frozen=True)
class Derivation:
    """Everything the first section of the report states, so a test can pin it."""

    daily_volatility: float
    exponent: float
    points: tuple[ImpliedPoint, ...]
    #: Mean of the implied values per regime -- what the config value rounds.
    patient_mean: float
    urgent_mean: float
    patient_config: float
    urgent_config: float
    #: SPEC.md 7.2's cross-check: Y_patient at the cross-check participation, in bp.
    cross_check_participation: float
    cross_check_bps: float
    cross_check_expected_bps: float

    @property
    def regime_ratio(self) -> float:
        return self.urgent_config / self.patient_config


def derive(cfg: config_mod.Config | None = None) -> Derivation:
    cfg = cfg or config_mod.load()
    costs = cfg.model.costs
    anchors = costs.calibration_anchors
    e = costs.total_cost_exponent
    sigma = anchors.daily_volatility

    points: list[ImpliedPoint] = []
    for regime, spec in (("patient", anchors.patient), ("urgent", anchors.urgent)):
        for point in spec.points:
            points.append(
                ImpliedPoint(
                    regime=regime,
                    source=spec.source,
                    participation=point.participation_of_adv,
                    cost_bps=point.cost_bps,
                    implied_y=impact.implied_prefactor(
                        cost_of_trade=point.cost_bps * _BPS,
                        participation=point.participation_of_adv,
                        daily_volatility=sigma,
                        exponent=e,
                    ),
                )
            )
    patient = [p.implied_y for p in points if p.regime == "patient"]
    urgent = [p.implied_y for p in points if p.regime == "urgent"]
    cross = anchors.cross_check
    cross_bps = (
        float(
            impact.impact_per_unit_traded(
                cross.participation_of_adv,
                daily_volatility=sigma,
                prefactor=costs.square_root_prefactor.patient,
                exponent=e,
            )
        )
        / _BPS
    )
    return Derivation(
        daily_volatility=sigma,
        exponent=e,
        points=tuple(points),
        patient_mean=sum(patient) / len(patient),
        urgent_mean=sum(urgent) / len(urgent),
        patient_config=costs.square_root_prefactor.patient,
        urgent_config=costs.square_root_prefactor.urgent,
        cross_check_participation=cross.participation_of_adv,
        cross_check_bps=cross_bps,
        cross_check_expected_bps=cross.expected_cost_bps,
    )


def _level_status(levels: spread_level.SpreadLevels, asset: str) -> str:
    entry = levels.entries.get(asset)
    if entry is None:
        return "**PLACEHOLDER** -- not supplied"
    if isinstance(entry, spread_level.IssuerLevel):
        return f"{entry.median_spread_bps:.2f}bp ({entry.disclosure_date})"
    if entry.displayed_bps is not None:
        return (
            f"{entry.displayed_bps / 100:.2f}% displayed -> [{entry.low_bps:.1f}, "
            f"{entry.high_bps:.1f})bp ({entry.disclosure_date})"
        )
    return f"band [{entry.low_bps:.2f}, {entry.high_bps:.2f}]bp ({entry.disclosure_date})"


@dataclass(frozen=True)
class ResolutionArithmetic:
    """What a half-resolution of half-spread is worth against impact at the anchor sigma."""

    half_spread_bps: float
    large_participation: float
    small_participation: float
    impact_large_bps: float
    impact_small_bps: float

    @property
    def share_large(self) -> float:
        return self.half_spread_bps / self.impact_large_bps

    @property
    def share_small(self) -> float:
        return self.half_spread_bps / self.impact_small_bps


def resolution_arithmetic(cfg: config_mod.Config | None = None) -> ResolutionArithmetic:
    """Row 195's arithmetic: 0.5bp against 16.4bp at 2% of ADV and 3.67bp at 0.1%."""
    cfg = cfg or config_mod.load()
    costs = cfg.model.costs
    resolution = costs.spread_level.display_resolution_bps
    if resolution is None:
        raise spread_level.SpreadLevelError("display_resolution_bps is null; nothing to state")
    sigma = costs.calibration_anchors.daily_volatility
    y = costs.square_root_prefactor.patient
    e = costs.total_cost_exponent

    def per_unit(q: float) -> float:
        return (
            float(impact.impact_per_unit_traded(q, daily_volatility=sigma, prefactor=y, exponent=e))
            / _BPS
        )

    return ResolutionArithmetic(
        half_spread_bps=resolution / 2.0,
        large_participation=_LARGE_PARTICIPATION,
        small_participation=_SMALL_PARTICIPATION,
        impact_large_bps=per_unit(_LARGE_PARTICIPATION),
        impact_small_bps=per_unit(_SMALL_PARTICIPATION),
    )


@dataclass(frozen=True)
class AssetResolution:
    """One asset at both ends of its band, at its own median sigma and ADV."""

    asset: str
    ticker: str
    displayed_bps: float | None
    full_low_bps: float
    full_high_bps: float
    half_low_bps: float
    half_high_bps: float
    median_sigma_pct: float
    median_adv_musd: float
    #: Per-unit impact at the two reference participations, patient regime, own sigma.
    impact_large_bps: float
    impact_small_bps: float

    @property
    def share_large(self) -> float:
        return self.half_high_bps / self.impact_large_bps

    @property
    def share_small(self) -> float:
        return self.half_high_bps / self.impact_small_bps


def resolution_table(ci: inputs.CostInputs, cfg: config_mod.Config) -> tuple[AssetResolution, ...]:
    costs = cfg.model.costs
    y = costs.square_root_prefactor.patient
    e = costs.total_cost_exponent
    rows: list[AssetResolution] = []
    for asset, entry in ci.levels.entries.items():
        if entry is None:
            continue
        bounds = entry.levels_bps
        sigma = float(np.nanmedian(ci.daily_volatility[asset].to_numpy(dtype="float64")))
        adv = float(np.nanmedian(ci.adv_dollars[asset].to_numpy(dtype="float64")))
        half_low = bounds[0] / 2.0 + costs.commission / _BPS
        half_high = bounds[-1] / 2.0 + costs.commission / _BPS

        def per_unit(q: float, sigma: float = sigma) -> float:
            return (
                float(
                    impact.impact_per_unit_traded(
                        q, daily_volatility=sigma, prefactor=y, exponent=e
                    )
                )
                / _BPS
            )

        rows.append(
            AssetResolution(
                asset=asset,
                ticker=ci.shapes[asset].ticker,
                displayed_bps=getattr(entry, "displayed_bps", None),
                full_low_bps=bounds[0],
                full_high_bps=bounds[-1],
                half_low_bps=half_low,
                half_high_bps=half_high,
                median_sigma_pct=sigma * 100.0,
                median_adv_musd=adv / 1e6,
                impact_large_bps=per_unit(_LARGE_PARTICIPATION),
                impact_small_bps=per_unit(_SMALL_PARTICIPATION),
            )
        )
    return tuple(rows)


def render(cfg: config_mod.Config | None = None) -> str:
    cfg = cfg or config_mod.load()
    costs = cfg.model.costs
    d = derive(cfg)
    laws = impact.ScalingLaws(costs.total_cost_exponent)
    e = costs.total_cost_exponent

    lines: list[str] = []
    w = lines.append
    w("# Cost model calibration")
    w("")
    w("Generated by `make report` from `mafrm.costs.calibration_report`. SPEC.md 7.1-7.3.")
    w("Sections 1 and 2 are arithmetic on `config/model.yaml` and read no data.")
    w("")
    w("## 1. The two square-root prefactors, backed out of published figures")
    w("")
    w("SPEC.md 7.2: AQR's ~10bp and Virtu's ~40bp are **both correct** and measure")
    w("different things -- one patient, liquidity-providing manager against a peer")
    w("universe that includes urgent and event-driven flow. Both regimes are modelled;")
    w("neither is selected, and there are no interior points (operator ruling,")
    w("2026-09-03, W5-P1: they are calibrated regimes, not endpoints of a continuum).")
    w("")
    w("Under the square-root law the cost of a trade, as a proportion of the trade, is")
    w("")
    w("```")
    w(f"cost/|Q| = Y * sigma * (Q/V)^(e-1)        e = {e:g}, so (Q/V)^{e - 1:g} = sqrt(Q/V)")
    w("Y        = cost / (sigma * sqrt(Q/V))")
    w("```")
    w("")
    w(f"Every anchor is quoted at sigma = {d.daily_volatility * 100:.0f}%/day.")
    w("")
    w("| Regime | Source | Participation Q/V | Published cost | sigma * sqrt(Q/V) | Implied `Y` |")
    w("|---|---|---|---|---|---|")
    for p in d.points:
        denom = d.daily_volatility * p.participation ** (e - 1.0)
        w(
            f"| {p.regime} | {p.source} | {p.participation * 100:.0f}% of ADV | "
            f"{p.cost_bps:.0f}bp | {denom:.6f} | **{p.implied_y:.3f}** |"
        )
    w("")
    w("| Regime | Mean of implied `Y` | `costs.square_root_prefactor` | Rounding |")
    w("|---|---|---|---|")
    w(
        f"| patient | {d.patient_mean:.4f} | **{d.patient_config:.2f}** | "
        f"{d.patient_config - d.patient_mean:+.4f} |"
    )
    w(
        f"| urgent | {d.urgent_mean:.4f} | **{d.urgent_config:.2f}** | "
        f"{d.urgent_config - d.urgent_mean:+.4f} |"
    )
    w("")
    w(f"Ratio urgent / patient: **{d.regime_ratio:.2f}x**.")
    w("")
    w(
        f"**Cross-check** (SPEC.md 7.2): at Q/V = {d.cross_check_participation * 100:.0f}% and "
        f"sigma = {d.daily_volatility * 100:.0f}%/day, `Y_patient` = {d.patient_config:.2f} gives "
        f"**{d.cross_check_bps:.2f}bp** of the trade against the spec's "
        f"{d.cross_check_expected_bps:.1f}bp -- consistent with AQR's value-weighted market "
        "impact of ~15bp at a mean order size of 1.2% of ADV."
    )
    w("")
    w("Both `Y` values are **borrowed coefficients**: this project has no execution data,")
    w("so any impact coefficient is inherited from a published figure rather than")
    w("estimated. The README's Limitations section names them as such.")
    w("")
    w("### Per-unit impact at the anchor participations, both regimes")
    w("")
    w("| Q/V | patient (bp of trade) | urgent (bp of trade) |")
    w("|---|---|---|")
    for q in sorted({p.participation for p in d.points} | {d.cross_check_participation}):
        pat = float(
            impact.impact_per_unit_traded(
                q, daily_volatility=d.daily_volatility, prefactor=d.patient_config, exponent=e
            )
        )
        urg = float(
            impact.impact_per_unit_traded(
                q, daily_volatility=d.daily_volatility, prefactor=d.urgent_config, exponent=e
            )
        )
        w(f"| {q * 100:.0f}% | {pat / _BPS:.1f} | {urg / _BPS:.1f} |")
    w("")
    w("## 2. Scaling laws (SPEC.md 7.3)")
    w("")
    w(f"At `costs.total_cost_exponent` = {e:g}, with the portfolio structure fixed:")
    w("")
    w("| Quantity | Scales as | Exponent |")
    w("|---|---|---|")
    w(f"| Cost per unit traded | `A^(e-1)` | {laws.aum_exponent_per_unit:g} |")
    w(
        f"| Annual drag, bps of AUM | `tau * A^(e-1)` | {laws.aum_exponent_drag_bps:g} on `A`, {laws.turnover_exponent:g} on `tau` |"
    )
    w(f"| Annual cost, dollars | `tau * A^e` | {laws.aum_exponent_dollars:g} |")
    w(f"| Cost vs volatility | linear | {laws.volatility_exponent:g} |")
    w(f"| Cost vs ADV | `V^-(e-1)` | {laws.adv_exponent:g} |")
    w("")
    w(
        f"Doubling AUM raises the bps drag by **{laws.drag_multiplier(aum_ratio=2.0):.2f}x**, not 2x; "
        f"doubling turnover raises it by **{laws.drag_multiplier(aum_ratio=1.0, turnover_ratio=2.0):.2f}x**. "
        "Turnover is the linear lever and size the sublinear one. Cost is linear in"
    )
    w("volatility, so a model without a vol term under-charges in stressed markets --")
    w("exactly when the optimizer most wants to trade (H5).")
    w("")
    w("## 3. The spread: level from the issuer, shape from EDGE")
    w("")
    w("SPEC.md 3.4.1: EDGE supplies time variation and cannot supply level for this")
    w("sleeve. The per-asset `a_i` of SPEC.md 7.1 is therefore built as")
    w("")
    w("```")
    w(
        "s_i(t) = s_issuer,i + max(0, s_EDGE,i(t) - baseline_i)     baseline_i = in-sample median of clipped EDGE"
    )
    w(
        f"a_i(t) = s_i(t) / 2 + commission                              commission = {costs.commission:g} (no published figure)"
    )
    w("```")
    w("")
    w("Additive, not multiplicative: for the liquid names the clipped EDGE baseline is")
    w("often exactly zero, and a ratio against it is the division-by-zero class SPEC.md")
    w("7.4 names. A level offset cancels in a widening (SPEC.md 3.4.1 consequence 1).")
    w("The issuer figure is a 2026 disclosure applied to 2009-2024 -- the anachronism")
    w("SPEC.md 3.4.1 accepted when it made issuer medians the level source. The five")
    w("curve points are priced through tradable proxies; each proxy's cost is the cost")
    w("of trading the ETF, not the exposure the risk model priced, and **the 30y proxy")
    w("(TLT) has roughly half the duration of a 30y zero, so its cost is the cost of")
    w("trading about half that exposure.**")
    w("")
    try:
        shapes = inputs.spread_shapes(cfg)
    except cache.CacheError as exc:
        w(f"_EDGE shape statistics not available: {exc}. Run `make data`._")
        shapes = {}
    levels: spread_level.SpreadLevels | None
    try:
        levels = inputs.load_levels(cfg)
    except spread_level.SpreadLevelError as exc:
        w(f"_Level file could not be read: {exc}_")
        levels = None
    if shapes:
        w(
            "| Asset | Priced by | Months | Negative before clip | Baseline (bp) | Max widening (bp) | Months widened | Issuer level |"
        )
        w("|---|---|---|---|---|---|---|---|")
        for asset_id, shape in shapes.items():
            priced = shape.ticker if shape.proxy_caveat is None else f"{shape.ticker} (proxy)"
            status = "n/a" if levels is None else _level_status(levels, asset_id)
            w(
                f"| {asset_id} | {priced} | {shape.months} | "
                f"{shape.negatives.fraction * 100:.0f}% | {shape.baseline_bps:.1f} | "
                f"{shape.max_widening_bps:.1f} | {shape.widening_months} | {status} |"
            )
        w("")
        w("`Negative before clip` is the resolution diagnostic of SPEC.md 3.4: near 50% means")
        w("the estimator is reporting noise around zero on a tight instrument, which is the")
        w("expected reading for every liquid name here and is why the level is not EDGE's.")
        w("")
    if levels is not None:
        if levels.placeholders:
            w(
                f"**The cost model does not run.** {len(levels.placeholders)} of "
                f"{len(levels.entries)} rows in `{costs.spread_level.file}` are placeholders: "
                + ", ".join(f"`{a}`" for a in levels.placeholders)
                + ". `mafrm.costs.inputs.build_cost_inputs` refuses each by name, and no"
            )
            w("default exists -- SPEC.md 3.4.1 forbids the EDGE level and")
            w("`costs.flat_spread_assumption_bps` as level sources.")
        else:
            w(f"All {len(levels.entries)} issuer levels are supplied; the cost model runs.")
            for asset, entry in levels.entries.items():
                second = getattr(entry, "second_source", None)
                if second is not None:
                    w("")
                    w(
                        f"Second source for `{asset}`: {second.value_bps:.1f}bp, {second.note} "
                        f"({second.url})"
                    )
        w("")
    w("## 4. What is swept and what is not")
    w("")
    w(
        f"- `Y`: the two regimes above, {d.patient_config:.2f} and {d.urgent_config:.2f}. "
        "Both always, no interior points."
    )
    grid = ", ".join(f"{g:g}" for g in costs.gamma_trade.points)
    w(
        f"- `gamma_trade`: {grid} -- half-steps from {costs.gamma_trade.sweep_min:g} to "
        f"{costs.gamma_trade.sweep_max:g}, stated as the rule. Characterising the cost "
        "model across the grid is `data-diagnostic`; selecting a point on backtest"
    )
    w("  performance in W6 is `strategy-config` and counts in the deflated-Sharpe trial")
    w("  count then (registered in `experiments.md`).")
    w("- The spread level: a point per asset from the issuer file, or a sourced band")
    w("  swept and reported as a range for any asset whose issuer discloses no median.")
    w("- No separate no-trade band exists or will be added: the `a|z|` L1 term already")
    w("  generates the no-trade region, and a band on top double-counts (SPEC.md 7.1).")
    w("")
    _render_resolution(w, cfg)
    return "\n".join(lines) + "\n"


def _render_resolution(w: Callable[[str], None], cfg: config_mod.Config) -> None:
    costs = cfg.model.costs
    resolution = costs.spread_level.display_resolution_bps
    w("## 5. What the issuers' display resolution means for the identity")
    w("")
    if resolution is None:
        w(
            "`costs.spread_level.display_resolution_bps` is null: the file's values are read as exact."
        )
        w("")
        return
    w(
        f"Every issuer displays its median bid/ask spread at {resolution / 100:.2f}% resolution, so a "
        f"displayed figure is a rounding interval of the full spread {resolution:g}bp wide, and the"
    )
    w(
        f"half-spread `a_i` is known to +-{resolution / 4:g}bp. The cost model runs at both ends of that"
    )
    w("interval -- the same code path as a sourced band for a missing level -- and no midpoint")
    w("is invented. **The band below is the honest width of that uncertainty, not a sensitivity")
    w("anyone chose.**")
    w("")
    ra = resolution_arithmetic(cfg)
    w(
        f"At the anchor volatility of {costs.calibration_anchors.daily_volatility * 100:.0f}%/day with "
        f"`Y_patient`, per-unit impact is **{ra.impact_large_bps:.1f}bp** at "
        f"{ra.large_participation * 100:g}% of ADV and **{ra.impact_small_bps:.2f}bp** at "
        f"{ra.small_participation * 100:g}%. A {ra.half_spread_bps:g}bp half-spread is therefore "
        f"**{ra.share_large * 100:.1f}%** of the impact term on the large trade and "
        f"**{ra.share_small * 100:.1f}%** on the small one: the {resolution:g}bp resolution matters"
    )
    w("for small trades and is negligible for large ones.")
    w("")
    try:
        ci = inputs.build_cost_inputs(cfg)
    except (cache.CacheError, spread_level.SpreadLevelError) as exc:
        w(f"_Per-asset table not available: {exc}_")
        w("")
        return
    rows = resolution_table(ci, cfg)
    w("Per asset, at its own in-sample median trailing volatility and median dollar ADV,")
    w(
        f"patient regime; `share` is the upper half-spread against per-unit impact at {ra.large_participation * 100:g}% / {ra.small_participation * 100:g}% of ADV:"
    )
    w("")
    w(
        "| Asset | Priced by | Displayed | Full spread (bp) | Half-spread `a_i` (bp) | Median sigma (%/day) | Median ADV ($M) | Impact @2% (bp) | Impact @0.1% (bp) | Share @2% | Share @0.1% |"
    )
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        displayed = "n/a" if r.displayed_bps is None else f"{r.displayed_bps / 100:.2f}%"
        w(
            f"| {r.asset} | {r.ticker} | {displayed} | [{r.full_low_bps:.1f}, {r.full_high_bps:.1f}) | "
            f"[{r.half_low_bps:.2f}, {r.half_high_bps:.2f}) | {r.median_sigma_pct:.2f} | "
            f"{r.median_adv_musd:,.0f} | {r.impact_large_bps:.1f} | {r.impact_small_bps:.2f} | "
            f"{r.share_large * 100:.1f}% | {r.share_small * 100:.1f}% |"
        )
    w("")
    widest = max(rows, key=lambda r: r.half_high_bps)
    thinnest = min(rows, key=lambda r: r.median_adv_musd)
    if widest.asset == thinnest.asset:
        w(
            f"**`{widest.asset}` ({widest.ticker}) has both the widest spread band "
            f"([{widest.full_low_bps:.1f}, {widest.full_high_bps:.1f})bp) and the thinnest volume "
            f"(${widest.median_adv_musd:,.0f}M/day median), so for any given dollar trade both terms "
            "of the cost function are largest there -- the spread term by its level and the impact "
            "term through participation. That is coherent: the least liquid instrument in the sleeve"
        )
        w("is the least liquid on both measures at once.**")
    else:
        w(
            f"The widest band is `{widest.asset}` ([{widest.full_low_bps:.1f}, {widest.full_high_bps:.1f})bp) "
            f"and the thinnest volume is `{thinnest.asset}` (${thinnest.median_adv_musd:,.0f}M/day): "
            "the two terms of the cost function peak on different assets."
        )
    w("")
    heavy = [r for r in rows if r.share_large >= 0.2]
    if heavy:
        names = ", ".join(f"`{r.ticker}` {r.share_large * 100:.0f}%" for r in heavy)
        w(
            f"The {ra.share_large * 100:.0f}% figure above is at the anchor's {costs.calibration_anchors.daily_volatility * 100:.0f}%/day. "
            f"The bond sleeve trades at a tenth to a third of that volatility, so its impact term is "
            f"correspondingly smaller and the same 0.5bp is a first-order share of it even on the large "
            f"trade: {names}. For those assets the cost term of the identity is spread-dominated, and "
            "the display resolution is the binding uncertainty on it rather than a rounding detail."
        )
        w("")
    w("Caveats carried from section 3: the five curve points are priced through proxies, the 30y")
    w("through TLT at roughly half a 30y zero's duration; every level is a 2026 disclosure applied")
    w("to the in-sample window.")
    w("")


def main() -> int:
    path = report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
