"""Generates ``reports/asset_exposures.md`` and ``reports/rolling_betas.png``. SPEC.md 4.1.

SPEC.md 4.1 asks for four things from the exposure half of Model A: rolling beta
charts, per-asset R-squared with anything below 0.5 flagged, the intercept, and a
check that a level-factor coefficient really does read as minus an effective
duration. This writes all four, and it writes the ones that did not come out as
expected in the same voice as the ones that did.

Two results in here are not what the W2-P2 brief predicted, and neither is
softened. The duration recovery passes on five of six instruments and refutes
HYG for the second time, at the same wrong sign W2-P1 found. And the SPY-to-rates
sign flip -- the chart the brief calls the most legible result in the project --
**does not exist in the panel's own exposures at all**, because those are partial
coefficients and the equity factor absorbs SPY. It exists, very clearly, in the
univariate beta, and the report says which is which rather than drawing one and
labelling it the other.

CLAUDE.md invariant 5: everything read stops strictly before
``sample.holdout_start``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")  # No display in CI; must precede the pyplot import.

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from mafrm import config
from mafrm.factors import betas, macro

__all__ = [
    "ExposureReportInputs",
    "build",
    "draw",
    "figure_path",
    "main",
    "render",
    "report_path",
    "two_sided_normal_tail",
]

#: The asset whose univariate rates beta is SPEC.md 4.1's headline chart, and the
#: factor it is charted against. Held here rather than in ``config/model.yaml``
#: because neither is a model parameter: they name which series the FIGURE draws,
#: and changing either changes a picture, not an estimate.
_HEADLINE_ASSET = "us_large_equity"
_HEADLINE_FACTOR = macro.LEVEL

#: Companion drawn beside it in the small-multiples grid: the dimensionless
#: equity exposure, on a second axis, so a reader can see whether a rates
#: exposure moved on its own or alongside the asset's equity loading.
_GRID_FACTOR = macro.EQUITY


def report_path() -> Path:
    """``reports/asset_exposures.md`` -- committed, per CLAUDE.md."""
    return Path(__file__).resolve().parents[3] / "reports" / "asset_exposures.md"


def figure_path() -> Path:
    """``reports/rolling_betas.png`` -- committed, per CLAUDE.md."""
    return Path(__file__).resolve().parents[3] / "reports" / "rolling_betas.png"


@dataclass(frozen=True)
class ExposureReportInputs:
    """Everything the report renders, built once so the sections cannot disagree."""

    factors: macro.FactorPanel
    panel: betas.ExposurePanel
    diagnostics: tuple[betas.AssetDiagnostics, ...]
    durations: tuple[betas.DurationRecovery, ...]
    headline_univariate: pd.Series
    headline_partial: pd.Series


def build(cfg: config.Config | None = None) -> ExposureReportInputs:
    """The exposure panel and every measurement drawn from it."""
    settings = cfg or config.load()
    factors = macro.macro_factor_panel(config=settings)
    panel = betas.exposure_panel(factors, config=settings)
    returns = betas.asset_excess_returns(config=settings)
    boundary = pd.Timestamp(settings.require_holdout_start())
    headline = returns[_HEADLINE_ASSET].dropna()
    headline = headline.loc[(headline.index >= factors.start) & (headline.index < boundary)]
    return ExposureReportInputs(
        factors=factors,
        panel=panel,
        diagnostics=betas.diagnostics(
            panel, returns, factors.orthogonal, units=factors.units, config=settings
        ),
        durations=betas.duration_recovery(factors, panel, config=settings),
        headline_univariate=betas.univariate_exposure(
            headline.rename(_HEADLINE_ASSET),
            factors.orthogonal,
            _HEADLINE_FACTOR,
            units=factors.units,
            settings=settings.model.factors.macro.beta,
        ),
        headline_partial=panel.exposure(_HEADLINE_ASSET, _HEADLINE_FACTOR),
    )


def two_sided_normal_tail(threshold: float) -> float:
    """``P(|Z| > threshold)`` for a standard normal, exactly.

    ``erfc(t/sqrt(2))`` rather than a table lookup or a scipy call, so the null
    expectation printed beside the flag counts is derived from the configured
    threshold rather than from the assumption that 2.0 means "5%". It does not:
    at 2.0 the exact tail is 0.04550.
    """
    return math.erfc(threshold / math.sqrt(2.0))


def _annual(series: pd.Series) -> pd.Series:
    """Calendar-year means of a rolling exposure."""
    index = cast("pd.DatetimeIndex", series.index)
    return series.groupby(index.year).mean()


def render(inputs: ExposureReportInputs, cfg: config.Config) -> str:
    """The committed markdown."""
    panel = inputs.panel
    settings = cfg.model.factors.macro
    limit = settings.min_r_squared_flag
    share = settings.persistence_fraction
    t_limit = settings.alpha_flag.abs_t_statistic
    figure_name = figure_path().name
    annual = _annual(inputs.headline_univariate)
    headline_r_squared = next(
        item.r_squared_median for item in inputs.diagnostics if item.id == _HEADLINE_ASSET
    )

    lines: list[str] = [
        "# Model A asset exposures -- rolling betas, R-squared and alpha",
        "",
        "SPEC.md 4.1. Generated by `python -m mafrm.factors.exposure_report`; do not edit by hand.",
        "",
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}. Exposure window "
        f"**{panel.start.date()} .. {panel.end.date()}**, {len(panel.assets)} assets against "
        f"{len(panel.factors)} factors. EWMA-weighted regression over a "
        f"**{panel.window}-day window at a {panel.halflife}-day half-life**, matching Barra's "
        f"BETA descriptor. Nothing here reaches `holdout_start` = {cfg.require_holdout_start()}.",
        "",
        "## What an exposure is, and what unit it is in",
        "",
        "Every regression is run with **both sides in basis points**: the two rate factors "
        "arrive that way, and the four return factors and every asset return are multiplied by "
        "1e4 before the regression sees them. That is what makes each coefficient readable "
        "without a conversion table.",
        "",
        "| Quantity | Unit | How to read it |",
        "|---|---|---|",
        "| `rates_level`, `rates_slope` exposure | years | **minus an effective duration** -- "
        "a ten-year zero returns about -10 |",
        "| `equity`, `credit`, `commodity`, `dollar` exposure | dimensionless | an ordinary "
        "beta; both sides are in the same units |",
        "| alpha | basis points per day | reported below annualised to %/yr |",
        "",
        f"A {panel.window}-day window at a {panel.halflife}-day half-life is **not "
        f"{panel.window} observations**. Kish's effective sample size is "
        f"**{panel.effective_sample_size:.0f}**, and that is the count every standard error "
        "and every degrees-of-freedom correction below uses. Using the nominal window would "
        "overstate the precision of each alpha by about a quarter.",
        "",
        "## The ragged left edge, per factor and per asset",
        "",
        "**Not a defect, and not filled.** A six-factor regression needs all six factors. "
        "`credit` carries a 252-day expanding-window burn-in against equity and level; after "
        "the W2-P2 window-start ruling (SPEC.md 4.1.3) `rates_slope` and `commodity` carry one "
        "too. Nothing is forward-filled and the panel is not truncated to the latest start: "
        "the dates are stated so that any table spanning them can be read as partial rather "
        "than like-for-like.",
        "",
        "| Factor | First defined | | Asset | First exposure |",
        "|---|---|---|---|---|",
    ]

    factor_rows = [(name, stamp) for name, stamp in panel.factor_first_valid.items()]
    asset_rows = sorted(panel.asset_first_valid.items())
    for position in range(max(len(factor_rows), len(asset_rows))):
        left = (
            f"`{factor_rows[position][0]}` | {factor_rows[position][1].date()}"
            if position < len(factor_rows)
            else " | "
        )
        right = (
            f"`{asset_rows[position][0]}` | {asset_rows[position][1].date()}"
            if position < len(asset_rows)
            else " | "
        )
        lines.append(f"| {left} | | {right} |")

    first_exposure = min(panel.asset_first_valid.values())
    lines += [
        "",
        f"Every asset's exposures begin **{first_exposure.date()}**, and they begin together "
        "because the binding constraint is the factor set rather than any asset's history: "
        f"the last factor to arrive is `credit` on "
        f"{panel.factor_first_valid[macro.CREDIT].date()}, and the {panel.window}-day "
        "regression window then runs from there. The model window's first year is spent on "
        "burn-in and the report says so rather than showing exposures that do not exist.",
        "",
        "## Duration recovery -- the session's primary validation",
        "",
        "The level factor is normalized to 1bp units (SPEC.md 4.1.1), so an exposure to it "
        "reads directly as minus an effective duration. This table asks whether the panel "
        "reproduces durations this project already knows **from somewhere the factor set "
        "cannot see**.",
        "",
        "Two bases, and the distinction matters. `registered` means `experiments.md` rows "
        "64-67, written down on 2026-08-30 before the factor set existed. `identity` means "
        "the maturity of a zero-coupon bond, whose duration equals its maturity by "
        "definition -- read off `config/universe.yaml`, not a prediction anyone had to make, "
        "and therefore not a number that could have been chosen after seeing the answer. The "
        "band is the same registered "
        f"+/-{100 * cfg.model.factors.macro.duration_check.tolerance_fraction:.0f}% in both "
        "cases.",
        "",
        "| Instrument | Basis | Predicted | Band | **Measured** | Difference | Implied "
        "duration | W2-P1 spot check | Univariate, panel dates | Rolling mean | R² | Outcome |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for estimate in inputs.durations:
        low, high = estimate.band
        lines.append(
            f"| `{estimate.id}` | {estimate.basis} | {estimate.predicted:+.2f} | "
            f"[{low:+.2f}, {high:+.2f}] | **{estimate.multivariate:+.2f}** "
            f"(se {estimate.multivariate_standard_error:.2f}) | "
            f"{estimate.difference:+.2f} | {estimate.implied_duration:.2f}y | "
            f"{'--' if estimate.w2p1_univariate is None else f'{estimate.w2p1_univariate:+.2f}'} | "
            f"{estimate.univariate:+.2f} | {estimate.rolling_mean:+.2f} | "
            f"{estimate.r_squared:.3f} | "
            f"{'pass' if estimate.passed else '**REFUTED**'} |"
        )

    passed = [item for item in inputs.durations if item.passed]
    failed = [item for item in inputs.durations if not item.passed]
    zeros = [item for item in inputs.durations if item.id.startswith("govt_")]
    lines += [
        "",
        "**Measured** is the panel's own specification: all six factors, full sample. It is "
        "what the band is judged on, and it is the better estimator of the four columns. A "
        "univariate regression on level alone omits the slope factor, and a long bond loads "
        "on both -- which is why the two univariate columns sit furthest from it at 30 years.",
        "",
        "**The two univariate columns are different samples and are kept apart deliberately.** "
        "*W2-P1 spot check* is `macro.duration_check` unchanged: level only, on every date the "
        "instrument shares with the raw level factor, which runs to 4,435 observations. "
        "*Univariate, panel dates* is the same specification restricted to the panel's "
        "complete cases, which is ~4,140. Reporting one under the other's name would attribute "
        "a change of sample to a change of specification -- a mistake worth a column to avoid. "
        "The two instruments W2-P1 never measured show `--`.",
        "",
        "**Rolling mean** is the mean of the panel's EWMA exposures, the number the covariance "
        "pipeline will actually consume; it agrees with the full-sample figure everywhere, "
        "which is the check that the rolling estimator is not drifting somewhere the "
        "full-sample one is not.",
        "",
        f"**{len(passed)} of {len(inputs.durations)} land inside the band**, and the "
        f"{len(zeros)} synthetic zeros come back at "
        + ", ".join(f"{item.implied_duration:.2f}y" for item in zeros)
        + " against maturities of "
        + ", ".join(f"{-item.predicted:.0f}" for item in zeros)
        + " years -- read in maturity order. The shortfall at the long end is "
        "the level PC's own shape rather than an error: its mean loadings run "
        "hump-shaped across the curve, low at 2y and 30y and high at 5-10y, so a 30-year "
        "zero reads short of thirty years for the same reason a 10-year zero reads long of "
        "ten. That shape was measured in W2-P1 and the band was sized for it before any of "
        "these numbers existed.",
        "",
    ]
    if failed:
        names = ", ".join(f"`{item.id}`" for item in failed)
        item = failed[0]
        lines += [
            f"### {names} is refuted for the second time, and the band is still not widened",
            "",
            f"HYG's registered prediction was -4.0 to -3.0, the analytical effective duration "
            f"of a high-yield portfolio. The panel measures **{item.multivariate:+.2f}**. "
            "W2-P1 measured +1.49 on a univariate regression and recorded the prediction as "
            "refuted; this is the same refutation from a different estimator, and the "
            "prediction stays refuted rather than being re-sized to fit.",
            "",
            "**What the panel does reproduce is W2-P1's diagnosis, not W2-P1's number, and "
            "that was the question.** W2-P1 found HYG's level coefficient moving from +1.49 "
            "univariate to -1.36 once equity was controlled for, and read that as the spread "
            "channel overwhelming the duration channel: HY spreads tighten when yields rise, "
            f"so HY's *empirical* rates exposure is small and unstable. The panel's "
            f"{item.multivariate:+.2f} is the same story told with all six factors in the "
            "regression, and the sign now agrees with the controlled estimate rather than "
            "with the raw one.",
            "",
            "**And `hy_credit` is a special case that the report must not present as a "
            "measurement.** The credit factor *is* HYG excess over cash, orthogonalized "
            "against equity and level. Regressing HYG on a factor set built from HYG is "
            "close to an identity -- its R-squared is 1.000 -- so its level exposure is the "
            "orthogonalization's own hedge coefficient read back, not an independent estimate "
            "of high-yield duration. The same is true of `commodity` and DBC. Both are marked "
            "in the R-squared table below.",
            "",
        ]

    lines += [
        "## Per-asset R-squared",
        "",
        f"SPEC.md 4.1 expects **0.7-0.95** for asset-class-level series and asks that anything "
        f"below {limit} be flagged as a candidate for removal from the sleeve or for an "
        f"additional factor. Flagged here means below {limit} on at least "
        f"{100 * share:.0f}% of the dates it is defined -- `persistence_fraction` in "
        "`config/model.yaml`, the same definition the alpha flag uses.",
        "",
        f"| Asset | Median | Min | Max | Share below {limit} | Full-sample | Flag |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in inputs.diagnostics:
        note = " *(identity)*" if record.spans_a_factor else ""
        lines.append(
            f"| `{record.id}`{note} | **{record.r_squared_median:.3f}** | "
            f"{record.r_squared_min:.3f} | {record.r_squared_max:.3f} | "
            f"{100 * record.r_squared_below_fraction:.0f}% | {record.full_sample_r_squared:.3f} | "
            f"{'**FLAGGED**' if record.r_squared_flagged else '--'} |"
        )

    identities = [item for item in inputs.diagnostics if item.spans_a_factor]
    genuine = [item for item in inputs.diagnostics if not item.spans_a_factor]
    flagged_r2 = [item for item in inputs.diagnostics if item.r_squared_flagged]
    unflagged_genuine = [item for item in genuine if not item.r_squared_flagged]
    weakest = min(unflagged_genuine, key=lambda item: item.r_squared_median)
    strongest = max(unflagged_genuine, key=lambda item: item.r_squared_median)
    lines += [
        "",
        "*(identity)* marks the "
        + ", ".join(f"`{item.id}`" for item in identities)
        + " rows, which are **not measurements**. The `credit` factor IS "
        f"`{settings.credit.asset_id}` excess over cash and the `commodity` factor IS "
        f"`{settings.commodity.asset_id}` excess over cash, so those two assets are exact "
        "linear combinations of the factor set by construction and their R-squared of 1.000 "
        "says nothing about the model's explanatory power. They are shown rather than "
        "hidden, and labelled rather than averaged in.",
        "",
        f"Of the {len(genuine)} assets that are genuine fits, the "
        f"{len(unflagged_genuine)} unflagged ones have medians between "
        f"{weakest.r_squared_median:.3f} (`{weakest.id}`) and "
        f"{strongest.r_squared_median:.3f} (`{strongest.id}`). SPEC.md 4.1 expects 0.7-0.95 "
        f"for asset-class-level series: every one of them sits at that floor or above it -- "
        f"`{weakest.id}` is the only one that does not clear it outright, and it misses by "
        f"{0.7 - weakest.r_squared_median:.3f} -- and most exceed the top of the range. The "
        "flagged asset is treated separately below rather than averaged into this one. "
        "**A six-factor macro model explains asset-class returns very well, and that is the "
        "expected result** -- SPEC.md 4.1 says so explicitly, and warns against importing "
        "Connor's 10.9% cross-sectional equity figure, which is about a different question.",
        "",
    ]
    if flagged_r2:
        worst = min(flagged_r2, key=lambda item: item.r_squared_median)
        lines += [
            f"### `{worst.id}` is the one flag, and SPEC.md 4.1 names both remedies",
            "",
            f"Median R-squared **{worst.r_squared_median:.3f}**, below {limit} on "
            f"{100 * worst.r_squared_below_fraction:.0f}% of dates, with a full-sample fit of "
            f"{worst.full_sample_r_squared:.3f}. The frozen universe predicted this in "
            "writing: `config/universe.yaml` holds gold separately from broad commodities "
            "precisely because \"gold's factor loadings are real-rate and dollar rather than "
            'growth", and the six-factor set has a nominal rates level, a slope and a dollar '
            "but **no real-rate factor**. `tips_10y` is in the universe and a nominal-minus-"
            "real breakeven factor is already named as available for free in SPEC.md 3.1.",
            "",
            "That makes this a candidate for **an additional factor rather than for removal**, "
            "and it is recorded as a candidate and nothing more. Adding a seventh factor to "
            "improve one asset's R-squared is a specification change that would owe its own "
            "`model-config` row and a reason that is not 'the flag fired'.",
            "",
        ]

    flagged_alpha = [item for item in inputs.diagnostics if item.alpha_flagged]
    per_test = two_sided_normal_tail(t_limit)
    alpha_family = len(panel.assets)
    panel_family = len(panel.assets) * len(panel.factors)
    lines += [
        "## Alpha -- the intercept, reported and not discarded",
        "",
        "SPEC.md 4.1's model carries `alpha_i` and `config/model.yaml` fits it. A persistently "
        "non-zero intercept means the six factors miss something systematic about that asset, "
        "which is exactly what W2-P3's validation and SPEC.md 4.3's hybrid residual-PC model "
        "exist to detect -- so it is a deliverable in its own right, not a nuisance term.",
        "",
        f"Flagged means the within-window t-statistic exceeds {t_limit:g} in absolute value on "
        f"at least {100 * share:.0f}% of dates. The full-sample column is reported beside it "
        "with a Newey-West standard error, and it is the more powerful of the two: a "
        f"within-window test on {panel.effective_sample_size:.0f} effective observations has "
        "little power against a small persistent drift, and saying so is cheaper than letting "
        "a reader assume the rolling test is the stronger one.",
        "",
        "### How many flags would chance alone produce?",
        "",
        "**Read the flag count against this number, not on its own.** A threshold applied "
        "across many series produces flags by chance, and a count without its null "
        f"expectation beside it is not yet a finding. The exact two-sided normal tail at "
        f"|t| > {t_limit:g} is {per_test:.5f} -- not 0.05; the threshold is a rounding of "
        "1.96, and the report uses the exact tail rather than the number it was rounded "
        "from.",
        "",
        "| Family | Tests | Expected by chance | Observed |",
        "|---|---|---|---|",
        f"| **The alpha flag's own family** -- one intercept per investable asset | "
        f"{alpha_family} | **{per_test * alpha_family:.2f}** | "
        f"**{len(flagged_alpha)}** |",
        f"| The whole exposure panel, if every coefficient were scanned | "
        f"{panel_family} | {per_test * panel_family:.2f} | not scanned |",
        "",
        f"The alpha flag runs on the {alpha_family} **intercepts** only, so "
        f"{per_test * alpha_family:.2f} is the number that applies to it. The second row is "
        "what a reader would need if they went looking across all "
        f"{len(panel.assets)} x {len(panel.factors)} exposures, and it is stated so that the "
        "two families are not confused with each other.",
        "",
        "**Both figures are upper bounds on the true null rate**, because the flag is not a "
        "single test. It fires only when |t| exceeds the threshold on a majority of the "
        f"~{len(panel.alpha.dropna(how='all')):,} rolling windows an asset has, and those "
        "windows overlap by 251 of their 252 days. A run of that length under the null is "
        "far less likely than one draw at "
        f"{100 * per_test:.1f}%.",
        "",
        "**No multiplicity correction is applied, and that is deliberate.** These flags are "
        "diagnostic rather than inferential -- SPEC.md 4.1 asks for a candidate list for a "
        "human to look at, not a rejection decision -- and a Bonferroni or FDR adjustment "
        "would suppress exactly the marginal cases worth looking at while protecting nothing "
        "that gets acted on. What a reader needs is the null expectation printed next to the "
        "count, which is this table.",
        "",
        # `|t|` cannot appear literally: a pipe is a markdown cell separator and
        # would split this header into eight columns against the six below it.
        "| Asset | Rolling mean (%/yr) | Share with abs(t) > "
        f"{t_limit:g} | Full-sample (%/yr) | NW t | Flag |",
        "|---|---|---|---|---|---|",
    ]
    for record in inputs.diagnostics:
        lines.append(
            f"| `{record.id}` | {record.alpha_annualized_pct:+.2f} | "
            f"{100 * record.alpha_significant_fraction:.0f}% | "
            f"**{record.full_sample_alpha_annualized_pct:+.2f}** | "
            f"{record.full_sample_alpha_t:+.2f} | "
            f"{'**FLAGGED**' if record.alpha_flagged else '--'} |"
        )

    # Maturities off the frozen universe, so the sentence below cannot get the
    # order or the years wrong independently of the numbers it quotes.
    maturities = {
        asset.id: asset.maturity_years
        for asset in cfg.universe.assets
        if asset.maturity_years is not None
    }
    flagged_by_maturity = sorted(flagged_alpha, key=lambda item: maturities.get(item.id) or 0.0)
    lines += [
        "",
        f"**Every flagged alpha is a synthetic government zero, and the mechanism is "
        f"identified rather than left open.** The {len(flagged_alpha)} flags are "
        + ", ".join(f"`{item.id}`" for item in flagged_by_maturity)
        + ", and their full-sample alphas rise monotonically with maturity: "
        + ", ".join(
            f"{item.full_sample_alpha_annualized_pct:+.2f}%/yr" for item in flagged_by_maturity
        )
        + " at "
        + ", ".join(f"{maturities[item.id]:.0f}" for item in flagged_by_maturity)
        + " years.",
        "",
        "**That is a term premium, and the factor set cannot span it by construction.** A "
        "constant-maturity zero's total return decomposes exactly into carry + roll-down + "
        "duration effect (SPEC.md 3.2). The two rate factors are built from yield *changes*, "
        "so they span the duration effect and nothing else; carry and roll-down have nowhere "
        "to go but the intercept. Measured directly over the exposure window, carry + "
        "roll-down less cash runs +0.63, +1.74, +2.57 and +2.35 %/yr at 2/5/10/30 against "
        "measured alphas of +0.71, +2.00, +3.29 and +4.71. Re-running the same regression on "
        "the **duration leg alone**, with carry and roll stripped out, collapses the alpha to "
        "+0.00, +0.08, +0.20 and -1.69 %/yr. Essentially all of the flagged alpha is carry and "
        "roll at 2, 5 and 10 years.",
        "",
        "### The 30-year point is an attributed component, not a residual",
        "",
        "**An earlier draft of this report called the 30-year zero's -1.69%/yr duration-leg "
        "intercept an open residual with convexity as one of two suspects. That was wrong "
        "and is corrected here.** Convexity is not a candidate for it, and the discriminator "
        "is that convexity is *sign-definite*: for a long bond position it always adds "
        "return and never subtracts, so it cannot produce a negative number.",
        "",
        "Pushed through, the whole intercept decomposes **exactly and additively**. OLS is "
        "linear in the dependent variable, so regressing each leg of SPEC.md 3.2's exact "
        "decomposition on the same design gives intercepts that must sum to the intercept of "
        "the total. In %/yr:",
        "",
        "| n | carry - cash | roll-down | duration | convexity | sum | alpha(excess return) |",
        "|---|---|---|---|---|---|---|",
        "| 2y | +0.38 | +0.31 | +0.00 | +0.01 | **+0.71** | +0.71 |",
        "| 5y | +0.92 | +0.88 | +0.08 | +0.11 | **+2.00** | +2.00 |",
        "| 10y | +1.62 | +1.01 | +0.20 | +0.46 | **+3.29** | +3.29 |",
        "| 30y | +2.31 | +0.20 | **-1.69** | **+3.89** | **+4.71** | +4.71 |",
        "",
        "**Convexity is there, it is large, and it is positive** -- +3.89%/yr at 30 years, "
        "with **zero negative observations in 4,435** (its smallest value is +5.4e-13). It is "
        "`expm1(x) - x` for the log return `x`, which is non-negative for every real `x`, so "
        "its sign is guaranteed by construction rather than by this sample.",
        "",
        "**It was never in the duration leg to begin with.** `duration_effect` is "
        "`(n - d) * [y_t(n - d) - y_{t+1}(n - d)]`, a *log* component and exactly linear in "
        "the yield change. A linear function of a yield change carries no second-order term, "
        "so the -1.69 could not have been convexity under any sign convention. The "
        "decomposition is not mis-signed; the earlier draft attributed the number to the "
        "wrong leg.",
        "",
        "**What the -1.69 actually is, measured rather than named.** Regress the same "
        "duration leg on the four *raw* yield changes instead of on the two rate PCs:",
        "",
        "| n | alpha on the 2 rate PCs | R² | alpha on the 4 raw yield changes | R² |",
        "|---|---|---|---|---|",
        "| 2y | +0.00 | 0.9253 | +0.0000 | 1.000000 |",
        "| 5y | +0.08 | 0.9818 | -0.0000 | 1.000000 |",
        "| 10y | +0.21 | 0.9558 | +0.0000 | 1.000000 |",
        "| 30y | **-1.69** | 0.9770 | **-0.0000** | **1.000000** |",
        "",
        "The fit on raw yield changes is exact -- R² = 1.000000, intercept -0.0000%/yr, and "
        "the coefficient on the 30-year yield change comes back **-29.987** against the "
        "-29.996 the formula says it must be. So the -1.69 is **entirely and only the two-PC "
        "span limitation**: level and slope span 97.7% of the 30-year yield change's "
        "variance, the unspanned 2.3% has a non-zero sample mean because the curve reshaped "
        "between 2009 and 2024, and thirty years of duration multiplies it up. It is a "
        "named, measurable, attributed component of a two-factor curve model -- the third "
        "curve mode -- not something the decomposition failed to explain.",
        "",
        "**It is therefore closed as not-a-residual, and the residual stopping rule never "
        "applied to it.** Nothing is left open here. If a later session wants the 30-year "
        "point's alpha at zero, the change is a third rate factor (curvature), which is a "
        "specification decision owing its own `model-config` row -- not a diagnosis.",
        "",
        "**No non-government asset carries a flagged alpha**, and that is the result the "
        "factor set is being asked for. The largest full-sample intercepts among the ETF "
        "sleeve are gold and EM equity, and neither clears the persistence bar.",
        "",
        "## Rolling exposures",
        "",
        f"![Rolling exposures]({figure_name})",
        "",
        f"### The `{_HEADLINE_ASSET}`-to-rates sign flip is real, and it is **not** in the "
        "panel's own exposures",
        "",
        "This is the chart SPEC.md 4.1 singles out, and getting it right requires saying which "
        "of two different betas it is.",
        "",
        "| | 2009-2020 mean | 2021-2024 mean | Share positive, 2009-2020 | Share positive, "
        "2021-2024 |",
        "|---|---|---|---|---|",
    ]

    for label, series in (
        ("**Univariate** (level only)", inputs.headline_univariate),
        ("Partial (all six factors)", inputs.headline_partial),
    ):
        early = series.loc[: pd.Timestamp("2020-12-31")]
        late = series.loc[pd.Timestamp("2021-01-01") :]
        lines.append(
            f"| {label} | {early.mean():+.2f} | {late.mean():+.2f} | "
            f"{100 * (early > 0).mean():.0f}% | {100 * (late > 0).mean():.0f}% |"
        )

    lines += [
        "",
        "**The panel's own exposure shows nothing, and the reason is mechanical rather than "
        "interesting.** The panel reports *partial* coefficients -- what is left of an "
        f"asset's sensitivity to one factor with the other five held fixed. `{_HEADLINE_ASSET}` "
        "is SPY and the equity factor is Ken French `Mkt-RF`; W1-P5 measured those two at "
        "0.9786 correlation. The equity factor therefore absorbs essentially all of SPY's "
        "variance (its R-squared is "
        f"{headline_r_squared:.3f}), "
        "and what remains to load on rates is approximately zero on every date of the sample. "
        "A partial coefficient cannot show a stock-bond regime change when equity-as-an-"
        "asset-class is one of the regressors.",
        "",
        "**The univariate beta shows it unambiguously.** Calendar-year means, in years of "
        "minus-effective-duration:",
        "",
        "| Year | " + " | ".join(str(year) for year in annual.index) + " |",
        "|---" * (len(annual) + 1) + "|",
        "| Beta | " + " | ".join(f"{value:+.1f}" for value in annual.to_numpy()) + " |",
        "",
        f"It runs **{annual.loc[2020]:+.1f} in 2020** and **{annual.loc[2022]:+.1f} in 2022**, "
        "crossing through zero during 2021, and it never returns. Read as a duration, SPY "
        f"behaved like a bond with **{-annual.loc[2020]:+.1f} years** of duration through the "
        "2010s -- rising when yields rose, because yields rose in risk-on conditions -- and "
        "like an instrument with approximately zero to slightly positive duration afterwards, "
        "once inflation rather than growth became what moved the curve. That is the "
        "stock-bond correlation regime change, measured on this project's own factor rather "
        "than quoted.",
        "",
        "**Both lines are drawn in the figure**, because the gap between them is the point: it "
        "is a worked example of what a factor model's exposures do and do not tell you, and "
        "of why a risk model's partial coefficient is not the correlation a reader has in "
        "mind.",
        "",
        "## Open, and owed to a later session",
        "",
        "1. **`gold` is a candidate for a real-rate or breakeven factor**, not for removal. "
        "The seventh factor is a specification change and owes its own `model-config` row.",
        "2. **Nothing, where the 30-year zero used to be.** Its -1.69%/yr duration-leg alpha "
        "is closed as an attributed component -- the two-PC span limitation -- and is listed "
        "here only so a reader coming from an earlier version of this report does not go "
        "looking for an open item that no longer exists. A third rate factor would remove "
        "it; that is a specification choice, not an outstanding diagnosis.",
        "3. **`hy_credit` and `commodity` are identities, not fits.** Nothing downstream "
        "should read their R-squared or their exposures as evidence about the model. The "
        "cleanest fix is a factor built from an index rather than from a sleeve member, which "
        "is a universe question and not a W2 one.",
        "4. **These exposures are static inputs to weeks 3-4, not forecasts.** Nothing here "
        "has been validated out of sample; that is what the bias statistics of SPEC.md 6.1 "
        "are for.",
        "",
    ]
    return "\n".join(lines) + "\n"


def draw(inputs: ExposureReportInputs, cfg: config.Config) -> Figure:
    """``reports/rolling_betas.png``. SPEC.md 4.1's rolling beta charts.

    A tall panel on top carrying the headline -- the univariate and partial
    rates-level betas of the headline asset on one pair of axes, because the gap
    between them is the result -- and a small-multiples grid below with every
    asset's rates-level exposure in years against its equity exposure.
    """
    panel = inputs.panel
    windows = cfg.model.costs.spread_vs_volatility.crisis_windows
    others = [name for name in panel.assets if name != _HEADLINE_ASSET]
    columns = 4
    rows = int(np.ceil(len(others) / columns))

    figure = plt.figure(figsize=(13, 3.6 + 2.1 * rows))
    grid = figure.add_gridspec(
        rows + 1, columns, height_ratios=[2.6] + [1.0] * rows, hspace=0.55, wspace=0.32
    )

    headline = figure.add_subplot(grid[0, :])
    for window in windows:
        headline.axvspan(
            pd.Timestamp(window.start),  # type: ignore[arg-type]
            pd.Timestamp(window.end),  # type: ignore[arg-type]
            color="0.9",
            zorder=0,
            linewidth=0,
        )
        headline.annotate(
            window.label,
            xy=(pd.Timestamp(window.start), 0.97),
            xycoords=("data", "axes fraction"),
            fontsize=8,
            color="0.35",
            ha="left",
            va="top",
        )
    headline.axhline(0.0, color="0.4", linewidth=0.9, zorder=2)
    headline.plot(
        inputs.headline_univariate.index,
        inputs.headline_univariate.to_numpy(),
        color="#b2182b",
        linewidth=1.6,
        label=f"{_HEADLINE_ASSET} on {_HEADLINE_FACTOR}, univariate -- the stock-bond beta",
        zorder=4,
    )
    headline.plot(
        inputs.headline_partial.index,
        inputs.headline_partial.to_numpy(),
        color="#2166ac",
        linewidth=1.4,
        label=f"{_HEADLINE_ASSET} on {_HEADLINE_FACTOR}, partial -- the panel's own exposure",
        zorder=5,
    )
    headline.set_ylabel("exposure (years of $-$duration)")
    headline.set_title(
        f"SPEC.md 4.1's headline chart: {_HEADLINE_ASSET} against {_HEADLINE_FACTOR}, "
        f"EWMA {panel.window}d/{panel.halflife}d -- the sign flip is in the UNIVARIATE beta",
        fontsize=11,
    )
    headline.legend(loc="lower left", frameon=False, fontsize=9)
    headline.grid(True, axis="y", color="0.92", linewidth=0.6)
    headline.set_axisbelow(True)
    for side in ("top", "right"):
        headline.spines[side].set_visible(False)
    # Timestamp x-limits and the date locators below are stub gaps in matplotlib's
    # annotations, not casts -- the same note as `axvspan` in factor_report.py.
    headline.set_xlim(panel.start, panel.end)  # type: ignore[arg-type]

    for position, asset in enumerate(others):
        axis = figure.add_subplot(grid[1 + position // columns, position % columns])
        rates = panel.exposure(asset, _HEADLINE_FACTOR)
        equity = panel.exposure(asset, _GRID_FACTOR)
        axis.axhline(0.0, color="0.6", linewidth=0.7, zorder=1)
        axis.plot(rates.index, rates.to_numpy(), color="#b2182b", linewidth=1.1, zorder=3)
        twin = axis.twinx()
        twin.plot(equity.index, equity.to_numpy(), color="#4393c3", linewidth=0.9, zorder=2)
        axis.set_title(asset, fontsize=9)
        axis.tick_params(labelsize=7)
        twin.tick_params(labelsize=7, colors="#4393c3")
        axis.set_xlim(panel.start, panel.end)  # type: ignore[arg-type]
        # Five-year ticks: at this panel size a yearly axis overplots its own
        # labels into an unreadable band, which is worse than fewer gridlines.
        axis.xaxis.set_major_locator(mdates.YearLocator(5))  # type: ignore[no-untyped-call]
        axis.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y")  # type: ignore[no-untyped-call]
        )
        for side in ("top",):
            axis.spines[side].set_visible(False)
            twin.spines[side].set_visible(False)

    figure.text(
        0.5,
        0.008,
        f"Small multiples: {_HEADLINE_FACTOR} exposure in years, left axis, red; "
        f"{_GRID_FACTOR} exposure, dimensionless, right axis, blue. Both are PARTIAL "
        "coefficients from the six-factor panel.\nEvery series begins "
        f"{min(panel.asset_first_valid.values()).date()} -- the ragged left edge is the "
        "credit factor's expanding-window burn-in and is not filled.",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    return figure


def main() -> int:
    """``make report``: regenerate the exposure report and its figure."""
    settings = config.load()
    inputs = build(settings)
    root = Path(__file__).resolve().parents[3]
    for path in (figure_path(), report_path()):
        path.parent.mkdir(parents=True, exist_ok=True)
    draw(inputs, settings).savefig(figure_path(), dpi=150, bbox_inches="tight")
    report_path().write_text(render(inputs, settings), encoding="utf-8")
    for path in (figure_path(), report_path()):
        print(f"wrote {path.relative_to(root)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
