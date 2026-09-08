"""W6-P3's one-dimensional bands off SPEC.md 9's reference cell. SPEC.md 9.1 ruling 4, 9.3.

Each band is the reference cell -- variant 4 (eigenfactor ``a = 1.0``),
treatment D at patient ``Y``, every other dial at the W6-P1 verification point
-- with ONE dial moved, run through :func:`mafrm.backtest.grid.run_cell`
unchanged. Nothing is crossed. Twelve cells, each a ``strategy-config`` row
registered in ``experiments.md`` before it ran, each with its falsifier scored
here in :func:`registered_checks` and printed with its verdict:

    spread end       the HIGH end of the issuer half-spread interval (1)
    TE multiple      0.5x and 2x the equal-weight book's realised volatility (2)
    book size        A_low and A_high, the two W6-P1 endpoints of the ruled AUM grid (2)
    gamma_trade      1.5, 2.0, 2.5, 3.0 -- SPEC.md 7.2.1's rule less the reference's 1.0 (4)
    per-asset bound  2/N, the ruled 2x-equal-weight bound (1)
    horizon          the long horizon's committed forecast panel (1)
    risk aversion    SPEC.md 8.4's initialisation lambda (W6-P1's) against the recovered multiplier (1)

The reference cell is re-run in the same process so that every comparison is
against a result produced by the same code on the same day; it is the same
configuration as the grid's 4D/patient and is NOT a new trial. ``N`` is read
from ``experiments.md`` at run time (SPEC.md 9.1 ruling 6) and the deflated
Sharpe's ``V[SR]`` is taken across these thirteen runs' per-period net Sharpes,
which share one alpha and are not independent trials -- stated, not adjusted.

What this module does NOT do: select. Every band point is reported; the
reference stays the reference. A point that looks better is a finding about a
dial, recorded as such.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from mafrm import config as config_mod
from mafrm.backtest import grid as grid_mod
from mafrm.backtest import metrics as metrics_mod
from mafrm.backtest import verification

__all__ = [
    "Band",
    "BandsResult",
    "ChartData",
    "band_specs",
    "main",
    "metrics_section",
    "registered_checks",
    "render",
    "run_bands",
]

_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_REPORTS: Final[Path] = _ROOT / "reports"
_RESULTS: Final[Path] = _ROOT / "results"
_MARKDOWN_PATH: Final[Path] = _REPORTS / "bands.md"
_CSV_PATH: Final[Path] = _REPORTS / "bands.csv"
_CHART_PATH: Final[Path] = _REPORTS / "bands.png"
_METRICS_PATH: Final[Path] = _RESULTS / "metrics.json"
_EXPERIMENTS_PATH: Final[Path] = _ROOT / "experiments.md"

_REFERENCE_TAG: Final[str] = "reference"
_TE_LIMIT: Final[str] = "tracking_error"
_ADV_LIMIT: Final[str] = "adv_participation"


class BandsError(ValueError):
    """The bands are mis-specified. Nothing is filled or defaulted."""


@dataclass(frozen=True)
class Band:
    """One band point: which dial moved, to what, and the ``cell_inputs`` override that does it."""

    tag: str
    dial: str
    value: str
    #: The registered prediction, as written in experiments.md before the run.
    registered: str
    horizon: str = "short"
    overrides: dict[str, object] = field(default_factory=dict)


def band_specs(universe: verification.Universe, settings: config_mod.Config) -> tuple[Band, ...]:
    """The twelve band points, in the order they are run, from ``optimizer.grid.bands``."""
    bands = settings.model.optimizer.grid.bands
    grid = settings.model.optimizer.grid
    out: list[Band] = []
    for end in bands.spread_ends:
        out.append(
            Band(
                tag=f"spread_{end}",
                dial="spread end",
                value=end,
                registered="the cost term exceeds the reference cell's risk-model term (0.0544 on Lo's scale in W6-P2b)",
                overrides={"spread_end": end},
            )
        )
    for multiple in bands.tracking_error_multiples:
        out.append(
            Band(
                tag=f"te_{multiple:g}x",
                dial="TE multiple",
                value=f"{multiple:g}x",
                registered=(
                    "B exceeds 1.10; B falls as the multiple rises"
                    if multiple < grid.tracking_error_multiple
                    else "B is below the reference's; B falls as the multiple rises"
                ),
                overrides={"tracking_error_multiple": float(multiple)},
            )
        )
    sizes = verification.book_sizes(universe, settings)
    if set(sizes) != {"A_low", "A_high"}:
        raise BandsError(f"band_specs: the book-size rule must give A_low and A_high, got {sizes}")
    out.append(
        Band(
            tag="book_A_low",
            dial="book size",
            value=f"A_low = ${sizes['A_low']:,.0f}",
            registered="impact is under 10% of the realised cost and the cost term is below the reference's",
            overrides={"nav": float(sizes["A_low"])},
        )
    )
    out.append(
        Band(
            tag="book_A_high",
            dial="book size",
            value=f"A_high = ${sizes['A_high']:,.0f}",
            registered="the ADV hinge is active on at least 10% of rebalances and the cost term is above the reference's",
            overrides={"nav": float(sizes["A_high"])},
        )
    )
    for gamma in bands.gamma_trades:
        out.append(
            Band(
                tag=f"gamma_trade_{gamma:g}",
                dial="gamma_trade",
                value=f"{gamma:g}",
                registered="annual turnover and realised cost drag are non-increasing along the sweep 1.0, 1.5, 2.0, 2.5, 3.0",
                overrides={"gamma_trade": float(gamma)},
            )
        )
    n = len(universe.assets)
    box = bands.position_box(n)
    out.append(
        Band(
            tag="bounds_2_over_N",
            dial="per-asset bound",
            value=f"{bands.position_box_equal_weight_multiple}/{n} = {box:.1%}",
            registered="|B - 1| is smaller than the reference's (the diagonal suspect: a book spread across more names carries less of its risk in the specific term)",
            overrides={"position_box": float(box)},
        )
    )
    for horizon in bands.horizons:
        out.append(
            Band(
                tag=f"horizon_{horizon}",
                dial="horizon",
                value=horizon,
                registered="annual turnover is below the reference's (a slower forecast) and SPEC.md 6.1's B is inside its chi-square interval",
                horizon=horizon,
            )
        )
    for mode in bands.risk_aversions:
        out.append(
            Band(
                tag=mode,
                dial="risk aversion",
                value=mode,
                registered="rms attainment of TE_target is below 0.5 and B is above the reference's (W6-P1: 31% and 1.18 on the same panel)",
                overrides={"risk_aversion": mode},
            )
        )
    return tuple(out)


@dataclass(frozen=True)
class BandsResult:
    reference: grid_mod.CellResult
    #: In ``band_specs`` order.
    bands: tuple[Band, ...]
    results: tuple[grid_mod.CellResult, ...]
    trials: int
    trials_variance: float
    checks: tuple[dict[str, object], ...] = ()

    def by_tag(self, tag: str) -> grid_mod.CellResult:
        for band, result in zip(self.bands, self.results, strict=True):
            if band.tag == tag:
                return result
        raise BandsError(f"no band tagged {tag!r}")


def _reference_cell(settings: config_mod.Config, *, tag: str = "") -> grid_mod.CellSpec:
    bands = settings.model.optimizer.grid.bands
    return grid_mod.CellSpec(
        bands.reference_variant, bands.reference_treatment, bands.reference_regime, tag=tag
    )


def run_bands(
    settings: config_mod.Config,
    universe_for: Callable[[str], verification.Universe],
    *,
    progress: bool = True,
) -> BandsResult:
    """The reference cell, then every band point, then the statistics that need all of them.

    ``universe_for(horizon)`` supplies the universe at a horizon; the long
    horizon's is built lazily by the caller, once.
    """
    grid = settings.model.optimizer.grid
    reference_universe = universe_for(grid.horizon)
    specs = band_specs(reference_universe, settings)
    if progress:
        print(f"solving {_REFERENCE_TAG} {_reference_cell(settings).label}", flush=True)
    reference = grid_mod.run_cell(
        grid_mod.cell_inputs(reference_universe, settings, _reference_cell(settings)),
        settings,
        progress=progress,
    )
    results: list[grid_mod.CellResult] = []
    for band in specs:
        universe = universe_for(band.horizon)
        cell = _reference_cell(settings, tag=band.tag)
        if progress:
            print(f"solving band {band.tag}: {band.dial} = {band.value}", flush=True)
        inputs = grid_mod.cell_inputs(universe, settings, cell, **band.overrides)  # type: ignore[arg-type]
        results.append(grid_mod.run_cell(inputs, settings, progress=progress))

    trials = metrics_mod.read_trial_count(_EXPERIMENTS_PATH)
    if trials < 28 + len(specs):
        raise BandsError(
            f"experiments.md carries N = {trials} but the grid's 28 cells plus {len(specs)} band "
            "points are trials: the rows must be registered BEFORE the run (experiments.md rule 1)"
        )
    per_period = np.array([r.sharpe_net.per_period for r in (reference, *results)])
    trials_variance = float(np.var(per_period, ddof=1))

    def deflate(result: grid_mod.CellResult) -> grid_mod.CellResult:
        return grid_mod.replace_deflated(
            result,
            metrics_mod.deflated_sharpe(
                result.monthly["net_return"].to_numpy(dtype=float),
                trials_variance=trials_variance,
                trials=trials,
            ),
        )

    out = BandsResult(
        reference=deflate(reference),
        bands=specs,
        results=tuple(deflate(r) for r in results),
        trials=trials,
        trials_variance=trials_variance,
    )
    return BandsResult(
        reference=out.reference,
        bands=out.bands,
        results=out.results,
        trials=out.trials,
        trials_variance=out.trials_variance,
        checks=registered_checks(out, settings),
    )


# ---------------------------------------------------------------------------
# The registered legs
# ---------------------------------------------------------------------------


def registered_checks(
    out: BandsResult, settings: config_mod.Config
) -> tuple[dict[str, object], ...]:
    """Every band's falsifier, scored against the reference run in the same process."""
    ref = out.reference
    grid = settings.model.optimizer.grid
    checks: list[dict[str, object]] = []

    def add(tag: str, claim: str, holds: bool, observed: str) -> None:
        checks.append({"band": tag, "claim": claim, "holds": bool(holds), "observed": observed})

    def result(tag: str) -> grid_mod.CellResult | None:
        try:
            return out.by_tag(tag)
        except BandsError:
            return None

    ref_risk = ref.identity.risk_model_term
    ref_cost = ref.identity.cost_term
    ref_b = ref.identity.bias_ratio

    # spread end
    for band in out.bands:
        if band.dial != "spread end":
            continue
        r = out.by_tag(band.tag)
        add(
            band.tag,
            "the cost term at the high spread end exceeds the reference cell's risk-model term",
            r.identity.cost_term > ref_risk,
            f"cost term {r.identity.cost_term:.4f} against the reference risk term {ref_risk:.4f} "
            f"(reference cost term {ref_cost:.4f})",
        )
    # TE multiples
    te_points = sorted(
        [(grid.tracking_error_multiple, ref)]
        + [
            (float(b.overrides["tracking_error_multiple"]), out.by_tag(b.tag))  # type: ignore[arg-type]
            for b in out.bands
            if b.dial == "TE multiple"
        ],
        key=lambda item: item[0],
    )
    bs = [(m, r.identity.bias_ratio) for m, r in te_points]
    falls = all(b_hi < b_lo for (_, b_lo), (_, b_hi) in itertools.pairwise(bs))
    low = [r for m, r in te_points if m < grid.tracking_error_multiple]
    add(
        "te_multiples",
        "B on the reference cell falls as the TE multiple rises",
        falls,
        ", ".join(f"{m:g}x: B {b:.3f}" for m, b in bs),
    )
    if low:
        add(
            f"te_{low[0].inputs.te_target / ref.inputs.te_target:g}x",
            "B exceeds 1.10 at 0.5x",
            low[0].identity.bias_ratio > 1.10,
            f"B {low[0].identity.bias_ratio:.3f}; SPEC.md 6.1's B {low[0].summary['bias']:.3f} against "
            f"[{low[0].summary['bias_lower']:.3f}, {low[0].summary['bias_upper']:.3f}]",
        )
    # book size
    a_low, a_high = result("book_A_low"), result("book_A_high")
    if a_low is not None:
        share = a_low.summary["impact_share_of_cost"]
        add(
            "book_A_low",
            "impact is under 10% of the realised cost and the cost term is below the reference's",
            bool(share < 0.10 and a_low.identity.cost_term < ref_cost),
            f"impact share {share:.1%}; cost term {a_low.identity.cost_term:.4f} vs reference {ref_cost:.4f}",
        )
    if a_high is not None:
        hinge = a_high.summary.get(f"{_ADV_LIMIT}_binding_share", 0.0)
        add(
            "book_A_high",
            "the ADV hinge is active on at least 10% of rebalances and the cost term is above the reference's",
            bool(hinge >= 0.10 and a_high.identity.cost_term > ref_cost),
            f"hinge active {hinge:.1%}; cost term {a_high.identity.cost_term:.4f} vs reference {ref_cost:.4f}; "
            f"impact share {a_high.summary['impact_share_of_cost']:.1%}",
        )
    # gamma_trade
    sweep = sorted(
        [(grid.gamma_trade, ref)]
        + [
            (float(b.overrides["gamma_trade"]), out.by_tag(b.tag))  # type: ignore[arg-type]
            for b in out.bands
            if b.dial == "gamma_trade"
        ],
        key=lambda item: item[0],
    )
    turnovers = [r.summary["turnover_annual"] for _, r in sweep]
    drags = [r.identity.cost_drag for _, r in sweep]
    monotone = all(b <= a for a, b in itertools.pairwise(turnovers)) and all(
        b <= a for a, b in itertools.pairwise(drags)
    )
    add(
        "gamma_trade",
        "annual turnover and realised cost drag are non-increasing along the gamma_trade sweep",
        monotone,
        "; ".join(
            f"{g:g}: turnover {t:.2f}, cost {d * 1e4:.1f} bp/yr, net SR {r.sharpe_net.annualised:.3f}"
            for (g, r), t, d in zip(sweep, turnovers, drags, strict=True)
        ),
    )
    # per-asset bound
    bounded = result("bounds_2_over_N")
    if bounded is not None:
        add(
            "bounds_2_over_N",
            "|B - 1| under the 2/N bound is smaller than the reference's",
            abs(bounded.identity.bias_ratio - 1.0) < abs(ref_b - 1.0),
            f"B {bounded.identity.bias_ratio:.3f} vs reference {ref_b:.3f}; effective assets "
            f"{bounded.summary['effective_assets_mean']:.2f} vs {ref.summary['effective_assets_mean']:.2f}",
        )
    # horizon
    for band in out.bands:
        if band.dial != "horizon":
            continue
        r = out.by_tag(band.tag)
        inside = r.summary["bias_lower"] <= r.summary["bias"] <= r.summary["bias_upper"]
        add(
            band.tag,
            "turnover at the long horizon is below the reference's and SPEC.md 6.1's B is inside its interval",
            bool(r.summary["turnover_annual"] < ref.summary["turnover_annual"] and inside),
            f"turnover {r.summary['turnover_annual']:.2f} vs {ref.summary['turnover_annual']:.2f}; "
            f"B (6.1) {r.summary['bias']:.3f} in [{r.summary['bias_lower']:.3f}, {r.summary['bias_upper']:.3f}]; "
            f"{int(r.summary['rebalances'])} rebalances",
        )
    # risk aversion
    for band in out.bands:
        if band.dial != "risk aversion":
            continue
        r = out.by_tag(band.tag)
        add(
            band.tag,
            "rms attainment of TE_target is below 0.5 and B is above the reference's",
            bool(r.summary["te_attainment_rms"] < 0.5 and r.identity.bias_ratio > ref_b),
            f"attainment {r.summary['te_attainment_rms']:.3f}; B {r.identity.bias_ratio:.3f} vs {ref_b:.3f}; "
            f"gamma_risk (spec) median {r.summary['gamma_risk_spec_median']:.1f} vs recovered "
            f"{ref.summary['gamma_risk_recovered_median']:.1f}",
        )
    return tuple(checks)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _num(value: float, digits: int = 3) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def _pct(value: float, digits: int = 1) -> str:
    return "n/a" if not np.isfinite(value) else f"{100.0 * value:.{digits}f}%"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def _row(tag: str, dial: str, value: str, r: grid_mod.CellResult) -> list[str]:
    s = r.summary
    ident = r.identity
    return [
        tag,
        dial,
        value,
        str(int(s["rebalances"])),
        _num(ident.sharpe_paper),
        _num(ident.sharpe_real),
        _num(ident.risk_model_term, 4),
        _num(ident.cost_term, 4),
        _num(ident.bias_ratio),
        _num(s["bias"]),
        f"[{s['bias_lower']:.3f}, {s['bias_upper']:.3f}]",
        _num(s["turnover_annual"], 2),
        _num(s["cost_drag_bps_per_year"], 1),
        _pct(s["impact_share_of_cost"]),
        f"{r.sharpe_net.annualised:.3f} ({r.sharpe_net.standard_error:.3f})",
        _num(r.deflated.probability if r.deflated else float("nan")),
        _pct(s.get(f"{_TE_LIMIT}_binding_share", float("nan"))),
        _num(s["te_attainment_rms"]),
        _num(s["effective_assets_mean"], 2),
        _pct(s.get(f"{_ADV_LIMIT}_binding_share", float("nan"))),
        _num(s["financing_bps_per_year"], 3),
        f"{int(s['inaccurate'])}/{int(s['resolve_count'])}/{s['resolve_max_l1']:.1e}",
        (
            "**RE-RUN OWED**"
            if s["resolve_actionable"] > 0
            else ("fallback's error" if s["resolve_exceeds"] > 0 else "-")
        ),
    ]


_HEADER: Final[tuple[str, ...]] = (
    "band",
    "dial",
    "value",
    "rebalances",
    "SR_paper",
    "SR_real",
    "risk-model term",
    "cost term",
    "B",
    "B (6.1)",
    "interval",
    "turnover /yr",
    "cost bp/yr",
    "impact share",
    "net SR (Lo SE)",
    "DSR",
    "TE bound",
    "rms attainment",
    "eff. assets",
    "ADV hinge",
    "financing bp/yr",
    "inaccurate / re-solved / max L1",
    "amended criterion",
)


def render(out: BandsResult, settings: config_mod.Config) -> str:
    """``reports/bands.md``."""
    grid = settings.model.optimizer.grid
    ref = out.reference
    lines = [
        "# W6-P3's bands: one dial at a time off SPEC.md 9's reference cell",
        "",
        "Generated by `python -m mafrm.backtest.bands`. SPEC.md 9.1 ruling 4 and the operator rulings of 2026-09-04 (SPEC.md 9.3). The reference cell is variant 4 (eigenfactor `a = 1.0`), treatment D at patient `Y`, every other dial at the W6-P1 verification point; each band moves ONE dial and is a `strategy-config` row registered in experiments.md before it ran, with its falsifier scored below. `N` at run time was "
        f"**{out.trials}**. The reference was re-run in this process (same configuration as the grid's 4D/patient, not a new trial) so every comparison is against the same code on the same day.",
        "",
        "## The band table",
        "",
        "Every Sharpe-type column is on Lo's scale, as in `reports/experiment_grid.md`. `B` is the identity's `sigma_r / sigma_f`; `B (6.1)` is SPEC.md 6.1's statistic against its exact chi-square interval at the band's own number of non-overlapping months. The deflated Sharpe's `V[SR]` is "
        f"{out.trials_variance:.3e} across these {len(out.results) + 1} runs' per-period net Sharpes (they share one alpha and are not independent trials; the formula assumes they are). `financing` is the nominal size of the cash leg at RF (ruling 1; zero by identity in the excess frame the book is marked in). The last column is the count of `optimal_inaccurate` rebalances, how many were re-solved with the fallback solver, and the largest L1 distance between the two weight vectors (ruling 6: the MAX decides against "
        f"{grid.resolve.l1_threshold:g}); the final column is the AMENDED criterion's verdict (W6-P3b, SPEC.md 9.3): a re-run is owed only where the L1 is above the threshold and the fallback's point is both feasible and better in objective; otherwise the distance is the fallback's error, reported.",
        "",
    ]
    rows = [
        _row(
            _REFERENCE_TAG,
            "-",
            "4D / patient / low spread / 1x / 2% anchor / gamma_trade 1 / unbounded / short / constraint",
            ref,
        )
    ]
    rows += [
        _row(band.tag, band.dial, band.value, r)
        for band, r in zip(out.bands, out.results, strict=True)
    ]
    lines += _table(_HEADER, rows)
    lines += ["", "## Registered legs, scored", ""]
    lines += _table(
        ["band", "registered", "verdict", "observed"],
        [
            [
                str(c["band"]),
                str(c["claim"]),
                "HOLDS" if c["holds"] else "**REFUTED**",
                str(c["observed"]),
            ]
            for c in out.checks
        ],
    )
    lines += [
        "",
        "## Reading notes",
        "",
        "- The bands are DIAGNOSTIC, not a search: no point is selected and the reference stays the reference (SPEC.md 9.1 ruling 4). A point that reads better is a finding about that dial.",
        "- The spread-end band answers W6-P2b's registration: if the cost term at the high end exceeds the reference's risk-model term, the reference cell's answer flips with the spread reading and the honest answer is a range.",
        "- The TE band answers W6-P2's registered suspect for why `B` on the constrained book is a third of W6-P1's: if `B` falls as the multiple rises, the diagonal specific-risk misspecification bites on a concentrated low-volatility book and is a smaller share of the forecast at higher target risk.",
        "- The 2x multiple asks the long-only simplex for twice the equal-weight book's volatility; where the bound is slack the book is reported at the risk it attains (a finding, SPEC.md 9.1 ruling 1).",
        "- The horizon band reads the committed panel's long column; its scored window and rebalance count are its own and are stated in the table.",
        "- `spec_initialisation` is W6-P1's `lambda = IR / (2 TE_target)` on the reference book: the band that measures what SPEC.md 8.4's second step changed.",
        "- Failure mode 9: every statistic is on non-overlapping months; no rolling window is quoted as an `n`.",
        "",
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class ChartData:
    """What the figure needs, from a :class:`BandsResult` or from ``results/metrics.json``."""

    labels: tuple[str, ...]
    #: Series name -> (values per band, the reference's value).
    series: dict[str, tuple[list[float], float]]
    trials: int

    @classmethod
    def from_result(cls, out: BandsResult) -> ChartData:
        ref = out.reference
        return cls(
            labels=tuple(b.tag for b in out.bands),
            series={
                "risk-model term  (mu_g / sigma_f)(1 - 1/B)": (
                    [r.identity.risk_model_term for r in out.results],
                    ref.identity.risk_model_term,
                ),
                "cost term  TC / (B sigma_f)": (
                    [r.identity.cost_term for r in out.results],
                    ref.identity.cost_term,
                ),
                "B = sigma_r / sigma_f": (
                    [r.identity.bias_ratio for r in out.results],
                    ref.identity.bias_ratio,
                ),
            },
            trials=out.trials,
        )

    @classmethod
    def from_metrics(cls, path: Path) -> ChartData:
        section = json.loads(path.read_text(encoding="utf-8"))["bands"]
        ref = section["reference"]["identity"]
        entries = section["bands"]
        return cls(
            labels=tuple(str(e["tag"]) for e in entries),
            series={
                "risk-model term  (mu_g / sigma_f)(1 - 1/B)": (
                    [float(e["identity"]["risk_model_term"]) for e in entries],
                    float(ref["risk_model_term"]),
                ),
                "cost term  TC / (B sigma_f)": (
                    [float(e["identity"]["cost_term"]) for e in entries],
                    float(ref["cost_term"]),
                ),
                "B = sigma_r / sigma_f": (
                    [float(e["identity"]["bias_ratio"]) for e in entries],
                    float(ref["bias_ratio"]),
                ),
            },
            trials=int(section["trial_count_N"]),
        )


def chart(data: ChartData, path: Path) -> None:
    """The two identity terms and ``B`` per band point, the reference marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink_2, gridline = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
    accent, reference_colour = "#2a78d6", "#eb6834"
    labels = list(data.labels)
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.5), sharey=True)
    fig.patch.set_facecolor(surface)
    y = np.arange(len(labels))
    for ax, (title, (values, reference_value)) in zip(axes, data.series.items(), strict=True):
        ax.barh(y, values, color=accent, edgecolor=surface, height=0.7)
        ax.axvline(reference_value, color=reference_colour, linewidth=1.5, label="reference 4D")
        ax.set_title(title, color=ink, fontsize=10, loc="left", pad=10)
        ax.set_facecolor(surface)
        ax.grid(axis="x", color=gridline, linewidth=1.0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(gridline)
        ax.tick_params(colors=ink_2, labelsize=9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, color=ink_2, fontsize=9)
    axes[0].invert_yaxis()
    axes[2].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(
        "W6-P3's bands: one dial at a time off the reference cell (SPEC.md 9.1 ruling 4)",
        color=ink,
        fontsize=12,
        x=0.02,
        y=0.985,
        ha="left",
    )
    fig.text(
        0.02,
        0.945,
        f"Reference: variant 4, treatment D, patient Y. N = {data.trials}. Diagnostic bands, nothing selected; the orange line is the reference cell.",
        color=ink_2,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)


def metrics_section(out: BandsResult, settings: config_mod.Config) -> dict[str, object]:
    """The ``bands`` section of ``results/metrics.json``."""

    def clean(value: object) -> object:
        if isinstance(value, float):
            return None if not np.isfinite(value) else float(f"{value:.10g}")
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [clean(v) for v in value]
        if isinstance(value, np.bool_ | bool):
            return bool(value)
        if isinstance(value, np.integer | int):
            return int(value)
        return value

    def entry(tag: str, dial: str, value: str, r: grid_mod.CellResult) -> dict[str, object]:
        ident = r.identity
        return {
            "tag": tag,
            "dial": dial,
            "value": value,
            "cell": r.cell.key,
            "regime": r.cell.regime,
            "rebalances": len(r.inputs.rebalance_dates),
            "nav_dollars": r.inputs.nav,
            "te_target_per_period": r.inputs.te_target,
            "identity": {
                "gross_return_annual": ident.gross_return,
                "cost_drag_annual": ident.cost_drag,
                "forecast_volatility_annual": ident.forecast_volatility,
                "realised_volatility_annual": ident.realised_volatility,
                "bias_ratio": ident.bias_ratio,
                "aggregation_eta": ident.aggregation,
                "sharpe_paper": ident.sharpe_paper,
                "sharpe_real": ident.sharpe_real,
                "risk_model_term": ident.risk_model_term,
                "cost_term": ident.cost_term,
            },
            "sharpe_net": {
                "per_period": r.sharpe_net.per_period,
                "annualised": r.sharpe_net.annualised,
                "standard_error": r.sharpe_net.standard_error,
                "eta": r.sharpe_net.eta,
            },
            "deflated_sharpe_probability": r.deflated.probability if r.deflated else None,
            "summary": dict(r.summary),
        }

    payload: dict[str, object] = {
        "task": "W6-P3",
        "spec": "SPEC.md 9.1 ruling 4; SPEC.md 9.3",
        "trial_count_N": out.trials,
        "trials_variance_per_period": out.trials_variance,
        "band_points": len(out.results),
        "reference": entry(_REFERENCE_TAG, "-", "", out.reference),
        "bands": [
            entry(band.tag, band.dial, band.value, r)
            for band, r in zip(out.bands, out.results, strict=True)
        ],
        "registered_checks": list(out.checks),
    }
    return clean(payload)  # type: ignore[return-value]


def write_metrics_section(section: dict[str, object], path: Path = _METRICS_PATH) -> None:
    """Put the ``bands`` section into ``results/metrics.json``, leaving the grid's own untouched."""
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload["bands"] = section
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - entry point
    parser = argparse.ArgumentParser(description="W6-P3's bands off the reference cell")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--chart-only",
        action="store_true",
        help="redraw reports/bands.png from results/metrics.json's bands section; no solve",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    settings = config_mod.load()
    if args.chart_only:
        chart(ChartData.from_metrics(_METRICS_PATH), _CHART_PATH)
        print(f"redrew {_CHART_PATH} from {_METRICS_PATH}")
        return 0
    universes: dict[str, verification.Universe] = {}

    def universe_for(horizon: str) -> verification.Universe:
        if horizon not in universes:
            universes[horizon] = verification.build_universe(settings, horizon=horizon)
        return universes[horizon]

    out = run_bands(settings, universe_for, progress=not args.quiet)
    _RESULTS.mkdir(parents=True, exist_ok=True)
    frames = [out.reference.rows.assign(band=_REFERENCE_TAG)]
    frames += [r.rows.assign(band=band.tag) for band, r in zip(out.bands, out.results, strict=True)]
    pd.concat(frames).to_csv(_CSV_PATH, float_format="%.10g")
    _MARKDOWN_PATH.write_text(render(out, settings), encoding="utf-8")
    write_metrics_section(metrics_section(out, settings))
    chart(ChartData.from_result(out), _CHART_PATH)
    for tag, r in [(_REFERENCE_TAG, out.reference)] + [
        (b.tag, r) for b, r in zip(out.bands, out.results, strict=True)
    ]:
        s = r.summary
        print(
            f"  {tag}: B {r.identity.bias_ratio:.3f}, risk {r.identity.risk_model_term:+.4f}, "
            f"cost {r.identity.cost_term:+.4f}, net SR {r.sharpe_net.annualised:.3f}, "
            f"turnover {s['turnover_annual']:.2f}, cost {s['cost_drag_bps_per_year']:.1f} bp/yr, "
            f"attainment {s['te_attainment_rms']:.3f}, inaccurate {int(s['inaccurate'])}, "
            f"max L1 {s['resolve_max_l1']:.1e}"
        )
    for c in out.checks:
        print(f"  band {c['band']}: {'HOLDS' if c['holds'] else 'REFUTED'} -- {c['observed']}")
    print(f"N = {out.trials}; wrote {_MARKDOWN_PATH}, {_CSV_PATH}, {_CHART_PATH}, {_METRICS_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
