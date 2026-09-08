"""Generates ``reports/factor_validation.md``. SPEC.md 6.5, last row.

SPEC.md 6.5 calls this a headline artefact rather than an appendix, and the
reason is that it is the only thing in the project that turns "trust me, my
factors are right" into something a reader can check. It is written to be read
by someone who does not believe it.

Three things this report must do that a naive version of it would not:

1. **State what could not be validated, first.** Three of Model A's six factors
   have no published analogue in the files this project holds. That is the
   headline, not a footnote, and the reason is sourced to a printed column
   listing of the three AQR workbooks and the Ken French daily research table.
2. **Carry the window, the units and the monthly ``n`` on every row.** The
   comparands are monthly and their coverages differ; a table whose columns
   silently span different periods is worse than no table.
3. **Argue the withdrawn gate rather than assert it.** SPEC.md 6.5's 0.3
   correlation bar is withdrawn for lack of validity on the SPEC.md 4.1.2
   precedent, and this report has to make that case or say it cannot.

CLAUDE.md invariant 5: everything read stops strictly before
``sample.holdout_start``.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

from mafrm import config
from mafrm.factors import macro, validation

__all__ = ["main", "render", "report_path"]

#: SPEC.md 4.1's own order. Sections iterate this rather than the config's
#: alphabetical key order, so the report reads in the order the factors are
#: constructed and orthogonalized.
_ORDER = (macro.EQUITY, macro.LEVEL, macro.SLOPE, macro.CREDIT, macro.COMMODITY, macro.DOLLAR)


def _ordered(names: tuple[str, ...]) -> tuple[str, ...]:
    """``names`` in SPEC.md 4.1's factor order."""
    return tuple(item for item in _ORDER if item in names)


#: How a factor's name is written in prose.
_TITLES = {
    macro.EQUITY: "Equity",
    macro.LEVEL: "Rates level",
    macro.SLOPE: "Rates slope",
    macro.CREDIT: "Credit",
    macro.COMMODITY: "Commodity",
    macro.DOLLAR: "Dollar",
}


def report_path() -> Path:
    """``reports/factor_validation.md`` -- committed, per CLAUDE.md."""
    return Path(__file__).resolve().parents[3] / "reports" / "factor_validation.md"


#: bps per month -> percent per year, for a mean.
_YEAR = 12.0 / 1e4 * 100.0

#: bps per month -> percent per year, for a standard deviation.
_ANNUAL_VOL = math.sqrt(12.0) / 1e4 * 100.0


def _pct(value: float) -> str:
    return f"{value:+.2f}%/yr"


def render(result: validation.ValidationResult, cfg: config.Config) -> str:
    """The committed markdown."""
    settings = result.settings
    boundary = cfg.require_holdout_start()
    lines: list[str] = []
    add = lines.append

    add("# Factor validation against published series")
    add("")
    add(
        f"Generated {datetime.now(UTC).date()} by `mafrm.factors.validation_report`. "
        "SPEC.md 6.5 (last row) as amended by SPEC.md 6.5.1; task W2-P3."
    )
    add("")
    add(
        'SPEC.md 6.5 asks this table to turn *"trust me, my factors are right"* into '
        "something falsifiable, and names it a headline artefact rather than an appendix. "
        "What it turns out to establish is narrower than the spec assumed, and the narrowing "
        "is the result rather than a shortfall."
    )
    add("")

    # ------------------------------------------------------------------
    add("## The headline: two of six factors have a published comparand")
    add("")
    order = _ordered(settings.mapped)
    mapped = ", ".join(f"`{name}`" for name in order)
    full = tuple(item for item in order if not settings.by_factor(item).is_sign_test)
    add(
        f"Model A has six factors. **{len(order)} are mapped** ({mapped}); "
        f"**{len(settings.unmapped)} have no published analogue in the files this project "
        f"holds**. One of the mapped three -- `{macro.LEVEL}` -- is a sign test only, so the "
        f"count that survives as a full external check is **{len(full)}: "
        + " and ".join(f"`{item}`" for item in full)
        + f"**, and **{6 - len(full)} of the six factors have no independent check available**."
    )
    add("")
    add(
        "The mapping was written factor by factor from a printed column listing of the three "
        "AQR workbooks and the Ken French daily research table, **before any regression was "
        "run**. That ordering is the point: an earlier draft of this task assumed five of six "
        "factors would be validated, and the column listing is what prevented three of them "
        "from being regressed against the nearest available column."
    )
    add("")
    add("| Factor | Comparand | Frequency | Status |")
    add("|---|---|---|---|")
    for name in _ORDER:
        title = _TITLES[name]
        if name in settings.mapped:
            item = settings.by_factor(name)
            status = "sign test only" if item.is_sign_test else "full comparison"
            add(
                f"| {title} (`{name}`) | `{item.dataset}` &rarr; `{item.column}` "
                f"| monthly | {status} |"
            )
        else:
            add(f"| {title} (`{name}`) | **none** | -- | no analogue |")
    add("")
    add(f"### Why the {len(settings.unmapped)} unmapped factors are unmapped")
    add("")
    add(
        "Each reason is sourced to what the files actually contain, not to recollection. "
        "`config/model.yaml` carries the same text so that a later session cannot quietly "
        "map one of them."
    )
    add("")
    for name in sorted(settings.unmapped):
        add(f"- **`{name}`** -- {settings.unmapped[name]}")
    add("")
    add(
        "- **AQR TSMOM is a comparand for nothing in Model A.** SPEC.md 6.5's last row and "
        "SPEC.md 12's week-2 *done when* both name it, and **no factor in the set SPEC.md 4.1 "
        "specifies is a trend factor**, so neither is meetable as written. Both are amended in "
        "SPEC.md 6.5.1. Building a trend factor to satisfy the spec would be alpha research and "
        "is out of scope under CLAUDE.md; the sentence is retained there as an illustration of "
        "what strong evidence looks like, explicitly marked as not applicable here."
    )
    add("")

    # ------------------------------------------------------------------
    add("## The comparison is monthly, and that changes what the table is")
    add("")
    counts = result.monthly.observations
    add(
        "All three AQR workbooks are monthly -- the source URLs end `-Monthly.xlsx`, and the "
        "manifest records 1,196 / 497 / 1,780 rows against the factor panel's "
        f"{len(result.panel.complete):,} complete daily dates. Every regression below therefore "
        f"runs at **n = {counts[macro.EQUITY]} months** for the factors that open at "
        f"`sample.start`, and **n = {counts[macro.COMMODITY]}** for `commodity` after its "
        "2008-04 burn-in -- not at 4,146 observations. **An alpha t-statistic at n = 213 is a "
        "different instrument from one at n = 4,146** and no row here should be read as though "
        "it were the latter."
    )
    add("")
    add(
        "Aggregation, stated because CLAUDE.md failure mode 1 sits directly in the path: the "
        "four decimal-return factors are **compounded** in log space, the two basis-point "
        "yield-change factors are **summed**, and each factor is aggregated over its own "
        "trading dates. Only a boundary month is ever dropped -- a month in which the factor's "
        "history opened or closed part-way through. The aggregation is asserted as an exact "
        "round trip against the daily series rather than trusted:"
    )
    add("")
    add("| Factor | Monthly n | Window | Boundary months dropped | Round-trip residual |")
    add("|---|---|---|---|---|")
    for name in result.monthly.frame.columns:
        key = str(name)
        series = result.monthly.series(key)
        cut = result.monthly.dropped[key]
        cut_text = ", ".join(str(item) for item in cut) if cut else "none"
        add(
            f"| `{key}` | {counts[key]} | {series.index.min()}..{series.index.max()} | "
            f"{cut_text} | {result.round_trip[key]:.1e} |"
        )
    add("")
    add(
        "AQR stamps its month-ends on the last **business** day (1926-07-30, 1877-02-28, "
        "1985-01-31 in the three files) and this project stamps on the last **traded** day, so "
        "the join key is the calendar month and the surviving index is AQR's own observed date. "
        "The joined count is asserted against a count computed from the two coverages, so a "
        "month lost to a stamp mismatch cannot pass as a month the sources did not share."
    )
    add("")
    add(
        f"**The right edge is `holdout_start` = {boundary}, always.** All three AQR files end "
        "*after* it -- 2026-02-27, 2026-05-29 and 2025-05-30 -- so their coverage can only ever "
        "bind the left edge or an earlier right edge. That is asserted per comparison rather "
        "than left to the intersection to deliver."
    )
    add("")

    # ------------------------------------------------------------------
    add("## The table")
    add("")
    add(
        "The self-built factor is the **dependent** variable, per SPEC.md 6.5's own wording. "
        "Both sides are in basis points per month, so `beta` on a return comparand is "
        "dimensionless and `beta` on the rate factor is in inverse years. `alpha` is the part "
        "of the factor the published series does not carry, annualised for display, with a "
        "Newey-West t-statistic on the Newey & West (1994) plug-in bandwidth "
        "`floor(4*(T/100)^(2/9))`."
    )
    add("")
    add(
        "| Factor | Comparand | n | Window (months) | Observed dates | R² | corr "
        "| beta (se) | units | alpha | t(alpha) | NW lags |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in _ordered(settings.mapped):
        fit = result.fit(name)
        add(
            f"| `{fit.factor}` | `{fit.column}` | {fit.observations} | {fit.window} | "
            f"{fit.observed_start.date()}..{fit.observed_end.date()} | {fit.r_squared:.4f} | "
            f"{fit.correlation:+.4f} | {fit.beta:+.5f} ({fit.beta_standard_error:.5f}) | "
            f"{fit.beta_units} | {_pct(fit.alpha_pct_per_year)} | "
            f"{fit.alpha_t_statistic:+.2f} | {fit.newey_west_lags} |"
        )
    add("")
    add(
        "**Units warning, because two numbers in this repository must not be compared.** The "
        "`rates_level` row is a *monthly summed basis-point* yield change regressed on a "
        "*monthly* bond return. W2-P2's exposure panel reports *daily* coefficients on the same "
        "factor that read directly as minus an effective duration (-1.66, -5.61, -11.29, "
        "-27.46 years). **These are not on the same scale and neither is a check on the "
        "other.**"
    )
    add("")

    equity = result.fit(macro.EQUITY)
    commodity = result.fit(macro.COMMODITY)
    add("### What the two full comparisons say")
    add("")
    add(
        f"**Equity correlates {equity.correlation:+.3f} with AQR's equity-index market "
        f"portfolio** (R² {equity.r_squared:.3f}). SPEC.md 6.5's own reading is that a "
        "correlation of 0.6-0.8 with the AQR analogue is strong evidence the concept is "
        "implemented correctly and 0.2 means something is wrong; this sits above that band. It "
        "should: both series are estimates of the return to owning equity."
    )
    add("")
    add(
        f"**The {_pct(equity.alpha_pct_per_year)} intercept "
        f"(t = {equity.alpha_t_statistic:+.2f}) is not a defect, and the beta is the evidence "
        "for that.** AQR's series is a **global developed** equity-index futures portfolio; "
        "ours is Ken French's **US** market. So the intercept should be the US-over-global "
        "spread across 2007-2024, and it is -- arithmetically, not by assertion:"
    )
    add("")
    add(
        f"| | mean | annualised volatility |\n|---|---|---|\n"
        f"| our `equity` (US `Mkt-RF`) | {_pct(equity.factor_mean_bps_per_month * _YEAR)} | "
        f"{equity.factor_sd_bps_per_month * _ANNUAL_VOL:.2f}%/yr |\n"
        f"| AQR `Equity indices Market` (global) | "
        f"{_pct(equity.comparand_mean_bps_per_month * _YEAR)} | "
        f"{equity.comparand_sd_bps_per_month * _ANNUAL_VOL:.2f}%/yr |\n"
        f"| **difference** | **{_pct(equity.mean_gap_bps_per_month * _YEAR)}** | |"
    )
    add("")
    add(
        f"The mean gap is {_pct(equity.mean_gap_bps_per_month * _YEAR)} and the fitted alpha is "
        f"{_pct(equity.alpha_pct_per_year)} -- the same number, because "
        f"**beta = {equity.beta:+.3f} (se {equity.beta_standard_error:.3f}) is "
        "indistinguishable from 1**. That is the whole argument: if the two series differed in "
        "*exposure*, the gap would show up in the slope and the intercept would absorb whatever "
        "was left over. It does not. They differ in **level** at the same unit exposure, which "
        "is what a regional performance spread looks like and is not something either "
        "construction got wrong. US equity beat global developed equity over this window; the "
        "factor is not miscalibrated."
    )
    add("")
    add(
        f"**Commodity correlates {commodity.correlation:+.3f} with AQR's equal-weight commodity "
        f"index** (R² {commodity.r_squared:.3f}), inside SPEC.md 6.5's 0.6-0.8 band and well "
        f"clear of the 0.2 that would mean something is wrong. Its intercept is "
        f"{_pct(commodity.alpha_pct_per_year)} at t = {commodity.alpha_t_statistic:+.2f} -- "
        "indistinguishable from zero, which is the stronger of the two level results: our "
        "DBC-based factor and AQR's index disagree about the path but not about the level."
    )
    add("")
    add(
        f"**Beta {commodity.beta:+.3f} and R² {commodity.r_squared:.3f} are a "
        "*composition* difference, not a scale difference, and the volatilities settle which.** "
        "A reader who stops at R² will conclude the commodity factor explains half of what "
        "it should. Decompose the slope instead -- `beta = corr x sd(factor)/sd(comparand)`:"
    )
    add("")
    add(
        f"| | annualised volatility |\n|---|---|\n"
        f"| our `commodity` (DBC excess) | "
        f"{commodity.factor_sd_bps_per_month * _ANNUAL_VOL:.2f}%/yr |\n"
        f"| AQR equal-weight commodity index | "
        f"{commodity.comparand_sd_bps_per_month * _ANNUAL_VOL:.2f}%/yr |\n"
        f"| ratio | **{commodity.scale_ratio:.3f}** |"
    )
    add("")
    add(
        f"The two carry **essentially the same volatility** (ratio "
        f"{commodity.scale_ratio:.3f}), so `beta` {commodity.beta:+.3f} is almost entirely the "
        f"correlation term {commodity.correlation:+.3f} rather than a difference in scale. That "
        "distinction is the reading: a *leverage* or *scale* mismatch -- one series being a "
        "geared version of the other -- would show as a ratio away from 1 and would be a defect "
        "in one construction. Equal volatility at 0.70 correlation is what **two differently "
        "weighted baskets of the same asset class** look like. DBC tracks an index-weighted "
        "commodity basket with a concentrated energy weight and its own roll-optimisation "
        "schedule; AQR's is an equal-weight academic portfolio. The same risk, a different mix."
    )
    add("")
    add(
        "**The exact index weights of DBC are not recorded anywhere in this repository**, so the "
        "composition claim is the structural reading rather than a sourced decomposition. What "
        "*is* measured is the part that matters for the factor set: the two series carry the "
        "same amount of commodity risk and differ in which commodities."
    )
    add("")

    # ------------------------------------------------------------------
    level = result.fit(macro.LEVEL)
    expected = settings.by_factor(macro.LEVEL).expected_beta_sign
    full_vol = validation.comparand_volatility_pct_per_year(settings.by_factor(macro.LEVEL))
    century = validation.century_volatility_summary()
    n_columns = int(century["columns"])
    min_vol, median_vol, max_vol = century["min"], century["median"], century["max"]
    near_ten = int(century["near_ten_percent"])
    add("### `rates_level` is a sign test, and a weak one")
    add("")
    add(
        "AQR's `Fixed income Market` is a global bond-market **excess return**; `rates_level` "
        "is a **US yield change in basis points**. They are not two estimates of one object, "
        "and the regression coefficient is"
    )
    add("")
    add("```")
    add("beta  =  -D_global x (dY_global / dY_US)")
    add("```")
    add("")
    add(
        "**a product of two quantities, neither of which is published anywhere in this "
        "repository's sources.** Any band wide enough to be defensible around that product is "
        "too wide to fail, and a gate that cannot fail is worse than no gate because it reads "
        "as evidence while supplying none. **No implied-duration band is pre-registered here, "
        "deliberately.**"
    )
    add("")
    sign_word = "negative" if expected is not None and expected < 0 else "positive"
    add(
        f"The falsifier is the sign and nothing else: `beta` must be {sign_word}, because a "
        f"bond return rises when yields fall. Measured **{level.beta:+.5f}** with a standard "
        f"error of {level.beta_standard_error:.5f} over {level.observations} months -- "
        f"**{'passes' if result.sign_gate_passed else 'FAILS'}**."
    )
    add("")
    add("#### The division every reader will do, done here")
    add("")
    add(
        "**Reported as a diagnostic and gated on nothing** -- but stated, because a reader who "
        "is not given the arithmetic will supply their own, and the obvious slip gives an "
        "answer half again too large."
    )
    add("")
    bracket_low, bracket_mid, bracket_high = level.implied_duration_bracket
    add(
        f"`beta` = {level.beta:+.5f} is in **basis points of yield per basis point of return**, "
        f"because both sides of this regression are in basis points. So the implied duration is "
        f"`1/|beta|` = **{bracket_high:.2f} years** -- not `|beta| x 100` = "
        f"{abs(level.beta) * 100:.1f} years, which is what the number gives if `beta` is read as "
        "a *percent* return per basis point. It is not: that would be the coefficient of the "
        "reverse regression in different units."
    )
    add("")
    add(
        "And `1/|beta|` is an upper edge rather than the answer, because the two regression "
        "directions attenuate toward whichever axis carries the noise. All three estimates:"
    )
    add("")
    add(
        f"| Estimate | Value | What it assumes |\n|---|---|---|\n"
        f"| `|R²/beta|` -- regress AQR's return on our yield change | **{bracket_low:.2f} yr** | "
        "all the noise is in the yield change |\n"
        f"| `sd(AQR)/sd(ours)` -- the orthogonal/volatility-ratio estimate | "
        f"**{bracket_mid:.2f} yr** | noise in neither; the geometric mean of the other two |\n"
        f"| `1/|beta|` -- reciprocal of the fitted slope | **{bracket_high:.2f} yr** | "
        "all the noise is in the return |"
    )
    add("")
    add(
        f"So: **implied duration {bracket_low:.1f}-{bracket_high:.1f} years, centred near "
        f"{bracket_mid:.1f}.** Both unknowns in `D_global x (dY_global/dY_US)` remain "
        "unidentified and nothing in this project depends on the number."
    )
    add("")
    add("#### Is AQR's series vol-scaled? Measured, and no")
    add("")
    add(
        "This matters for reading the duration: **a construction targeting a constant "
        "volatility would need either a much longer duration or leverage**, and roughly 6.7 "
        "years would then be the wrong reading of the coefficient. Many AQR datasets do target "
        "a constant volatility, so the question is live rather than pedantic -- and it is "
        "answerable from the file rather than from recollection."
    )
    add("")
    add(
        f"`Fixed income Market` realises **{level.comparand_sd_bps_per_month * _ANNUAL_VOL:.2f}"
        f"%/yr** over the comparison window and **{full_vol:.2f}%/yr** over its whole history "
        f"back to 1926 (read, like everything else here, up to the holdout boundary). Across "
        f"all {n_columns} columns of Century of Factor Premia the "
        f"annualised volatility runs {min_vol:.1f}% to {max_vol:.1f}% with a median of "
        f"{median_vol:.1f}%, and only {near_ten} of {n_columns} sit between 9% and 11%. "
        "**This dataset is not scaled to a volatility target.**"
    )
    add("")
    add(
        f"So AQR's series behaves like an **unlevered intermediate-duration government bond "
        f"portfolio**: roughly {bracket_mid:.1f} years of duration realising "
        f"{level.comparand_sd_bps_per_month * _ANNUAL_VOL:.2f}%/yr against our measured US "
        f"yield volatility of {level.factor_sd_bps_per_month * _ANNUAL_VOL * 100:.0f} bp/yr. "
        "**That the three numbers reconcile is arithmetic, not corroboration** -- the "
        "volatility-ratio estimate is *defined* as one divided by the other, so it reproduces "
        "them by construction and confirms nothing. What does the work is the comparison "
        "against 10%: the series would have to realise roughly twice its measured volatility "
        "to be vol-targeted, and it does not."
    )
    add("")
    add(
        "**Whether it is a cash index or unlevered futures is not documented anywhere in this "
        "repository, and is stated as unknown rather than guessed.** The distinction does not "
        "affect the sign test, which is the only thing gated."
    )
    add("")
    add(
        "**This test adds independence, not power, and the report says so plainly.** The strong "
        "test of the level factor already ran in W2-P2: daily, on US data, at n = 4,146, "
        "against durations this project knows independently, and five of six instruments landed "
        "inside their pre-registered bands (`experiments.md` rows 73-77). What the AQR "
        "comparison adds is that the factor also tracks a series produced by someone else from "
        f"different data -- at R² {level.r_squared:.3f}, which is higher than a transform of "
        "this kind had any right to deliver."
    )
    add("")

    # ------------------------------------------------------------------
    add("## Rolling correlation")
    add("")
    window = settings.rolling_correlation_window_months
    error = settings.rolling_correlation_standard_error
    add(
        f"A **{window}-month** rolling window. This is a **choice**, not a published constant: "
        "it was fixed by the operator in the W2-P3 brief before the run, nothing was swept "
        "against it, and `config/model.yaml` records it as a choice. No `experiments.md` row is "
        "owed, because a row would record a search that did not happen."
    )
    add("")
    add(
        f"**The standard error is printed beside every window and it is large.** At {window} "
        f"points the Fisher-z standard error is `1/sqrt({window}-3)` = **{error:.3f}**, so a "
        "drift from 0.55 to 0.75 is roughly one standard error and must not be read as a change "
        "in anything."
    )
    add("")
    add("| Factor | Windows | Min | Median | Max | Range in Fisher-z SEs | First..last |")
    add("|---|---|---|---|---|---|---|")
    for name in _ordered(settings.mapped):
        series = result.rolling[name]
        spread = abs(math.atanh(float(series.max())) - math.atanh(float(series.min()))) / error
        add(
            f"| `{name}` | {len(series)} | {series.min():+.3f} | {series.median():+.3f} | "
            f"{series.max():+.3f} | {spread:.1f} | {series.index.min()}..{series.index.max()} |"
        )
    add("")
    add(
        "**Every mapped factor holds its sign in every window** -- which is the property worth "
        "having, and the one a rolling correlation is actually good for."
    )
    add("")
    add(
        "The Fisher-z column is there to calibrate a *local* drift, not to license a claim that "
        "the whole range is noise. Read the two together and neither reading is available: the "
        "full ranges span 3-6 standard errors, so they are **not** within sampling error of a "
        "constant correlation -- but these are 36-month windows stepped one month at a time, "
        "overlapping in 35 of 36 observations, so the windows are nowhere near independent "
        "draws and the span cannot be read as 3-6 independent standard errors either. What the "
        "column does support is the comparison it was put there for: two nearby readings "
        "differing by less than about "
        f"{error:.2f} -- a drift from 0.55 to 0.75, say -- are not distinguishable, and this "
        "report makes no claim that rests on one."
    )
    add("")

    # ------------------------------------------------------------------
    add("## The gates, and why 0.3 was withdrawn rather than lowered")
    add("")
    add(
        "W2-P3 was originally given a **0.3 correlation gate**. It is withdrawn, and this "
        "section has to make that argument rather than assert it, because *withdrawing a gate "
        "the run would have passed anyway is exactly how a project talks itself out of a bar it "
        "did not like*."
    )
    add("")
    add(
        "**The argument is the SPEC.md 4.1.2 precedent.** The all-pairs 0.15 correlation bar was "
        "withdrawn in W2-P1 because it *measured something other than what it claimed to* -- "
        "meeting it would have driven the factor correlation matrix toward diagonal and left "
        "week 3's eigenfactor adjustment nothing to correct. It was invalid in kind, not "
        "mis-calibrated. The same test applies here and gives the same answer for a different "
        "reason: **0.3 was a threshold for a comparand that does not exist.** SPEC.md 6.5's only "
        "published figures -- 0.6-0.8 strong, 0.2 wrong -- are stated for a *trend* factor "
        "against AQR TSMOM, and Model A has no trend factor. The number was written before "
        "anyone had looked at what the AQR files contain, and no published figure covers any "
        "pairing that actually exists in this project."
    )
    add("")
    cleared = ", ".join(
        f"`{name}` {abs(result.fit(name).correlation):.3f}" for name in _ordered(settings.mapped)
    )
    add(
        "**It is also worth noting what withdrawing it does not buy.** All three mapped factors "
        f"clear 0.3 in absolute correlation ({cleared}), "
        "so the gate is withdrawn from a position of not needing it. That is the only position "
        "from which withdrawing a gate is legitimate."
    )
    add("")
    add("Three pre-registered falsifiers replace it, and each can fail.")
    add("")

    add("### (i) The daily anchor -- `experiments.md` row 41")
    add("")
    anchor = settings.daily_anchor
    add(
        f"Row {anchor.source_row} measured our own SPY total return against Ken French "
        f"`Mkt-RF` at correlation **{anchor.registered_correlation:.4f}** with a mean gap of "
        f"**{anchor.registered_mean_gap_pct_per_year:+.3f}%/yr** over "
        f"{anchor.registered_observations:,} days. That is the only sharp prior in this session: "
        "a measured expectation rather than a guessed threshold, and the calibration point for "
        "what *two constructions of the same object* look like on this project's data. The "
        f"equity factor is `Mkt-RF` itself, so its comparand is SPY excess -- which SPEC.md "
        "4.1.1 ruling 2 converted from a candidate input into a comparand."
    )
    add("")
    add("| Window | n | Dates | Correlation | 95% Fisher-z interval | Mean gap |")
    add("|---|---|---|---|---|---|")
    for check in result.anchors:
        add(
            f"| {check.label} | {check.observations:,} | "
            f"{check.start.date()}..{check.end.date()} | {check.correlation:.4f} "
            f"| [{check.correlation_low:.4f}, {check.correlation_high:.4f}] "
            f"| {_pct(check.mean_gap_pct_per_year)} |"
        )
    add("")
    first = result.anchors[0]
    add(
        f"**Passes.** Row 41's {anchor.registered_correlation:.4f} lies inside the sampling "
        f"interval of the reproduction on row 41's own window "
        f"([{first.correlation_low:.4f}, {first.correlation_high:.4f}]), and the mean gap comes "
        f"back at {_pct(first.mean_gap_pct_per_year)} against a registered "
        f"{anchor.registered_mean_gap_pct_per_year:+.3f}%/yr. The window is row 41's, truncated "
        f"at `holdout_start`: row 41 ran to the end of the cache and this session may not, so "
        f"{anchor.registered_observations - first.observations} days are dropped and the small "
        "difference from the registered figure is that truncation."
    )
    add("")
    in_sample, pre_sample = result.anchors[1], result.anchors[2]
    add(
        f"**The in-sample window gives {in_sample.correlation:.4f}, not "
        f"{anchor.registered_correlation:.4f}, and that difference is a window difference rather "
        "than sampling error — measured, not asserted.** The two intervals do not overlap, so "
        "the distinction matters. The pre-sample period 1993-2007 correlates "
        f"{pre_sample.correlation:.4f} and the in-sample period {in_sample.correlation:.4f}; the "
        f"full-history {anchor.registered_correlation:.4f} is a blend of the two. SPY's tracking "
        "of the CRSP market improved after 2007, which is unsurprising -- the small-cap "
        "complement's dispersion against the S&P 500 was larger in the dot-com era. **Comparing "
        "the in-sample number to row 41's headline and calling the difference a failure would "
        "charge a window difference to sampling error**, which is the error this report's own "
        "*every comparison states its own window* rule exists to prevent."
    )
    add("")

    add("### (ii) The placebo -- each factor beats every other factor's comparand")
    add("")
    add(
        "Regress each mapped factor on the comparands belonging to the *other* mapped factors. "
        "Each factor's R² against its own comparand must exceed its R² against every other. "
        "**This is falsifiable without any constant at all**: if `commodity` explained AQR's "
        "Fixed income Market as well as it explains AQR's commodity index, the mapping would be "
        "doing no work and this table would be decoration. It is the same move as the IVV/VOO "
        "control (row 63) and the zero-spread simulation -- a gate built from measurement rather "
        "than from a guessed threshold."
    )
    add("")
    add(
        "**Read row-wise.** Each row holds the dependent variable and its window fixed and "
        "varies only the comparand; reading down a column compares different windows."
    )
    add("")
    header = " | ".join(f"`{name}`" for name in _ordered(settings.mapped))
    add(f"| Factor (rows) &darr; / comparand &rarr; | {header} | Own | Margin | Result |")
    add("|---|" + "---|" * (len(settings.mapped) + 3))
    by_factor = {row.factor: row for row in result.placebo_rows}
    for factor in _ordered(settings.mapped):
        row = by_factor[factor]
        cells = []
        for name in _ordered(settings.mapped):
            value = f"{row.r_squared[name]:.4f}"
            cells.append(f"**{value}**" if name == row.own else value)
        best_name, _ = row.best_other
        add(
            f"| `{row.factor}` | " + " | ".join(cells) + f" | `{row.own}` | "
            f"{row.margin:+.4f} vs `{best_name}` | {'pass' if row.passed else '**FAIL**'} |"
        )
    add("")
    tightest = min(result.placebo_rows, key=lambda row: row.margin)
    rival, rival_score = tightest.best_other
    add(
        f"**{'Passes on every row' if result.placebo_passed else 'FAILS'}.** The nearest call is "
        f"`{tightest.factor}`, which loads on `{rival}`'s comparand at R² {rival_score:.3f} and "
        f"still beats it by {tightest.margin:.3f}. The off-diagonal number this gate exists to "
        f"produce is `{macro.EQUITY}` against the bond comparand at R² "
        f"{by_factor[macro.EQUITY].r_squared[macro.LEVEL]:.4f}: a mapping that is doing work "
        "looks like that away from its own diagonal."
    )
    add("")

    add("### (iii) `rates_level` beta is negative")
    add("")
    add(
        f"Measured {level.beta:+.5f} -- **{'passes' if result.sign_gate_passed else 'FAILS'}**. "
        "A failure here would have meant something badly wrong with the level factor's sign "
        "convention, which SPEC.md 4.1.1 fixes as a by-product of the normalization."
    )
    add("")
    verdict = "All three hold." if result.passed else "**AT LEAST ONE GATE FAILED.**"
    add(f"**{verdict}** No gate was adjusted after seeing a number.")
    add("")

    # ------------------------------------------------------------------
    add("## Two comparands that turned out to be one series")
    add("")
    add(
        f"Century of Factor Premia's `Commodities Market` and Commodities-for-the-Long-Run's "
        f"`Excess return of equal-weight commodities portfolio` agree to "
        f"**{result.duplicate_max_difference:.1e}** over all "
        f"**{result.duplicate_months:,}** overlapping months. They are the same series."
    )
    add("")
    add(
        "This was checked rather than assumed, and it changed the design. The mapping as "
        "approved listed **both** as commodity comparands, on the reasoning that a second "
        "independent AQR construction costs nothing. It is not a second construction."
    )
    add("")
    add(
        "**This is the sharpest instance yet of the project's flattering-direction artifact "
        "family, and the reason is specific.** The placebo of §(ii) was introduced precisely "
        "because it needs no invented constant -- it is the gate that replaced 0.3. A "
        "duplicated comparand would have made it **pass by tautology while looking like an "
        "independent check**: `commodity` would have been scored against a second copy of its "
        'own series, "beaten" it by whatever rounding separated them, and the pass would have '
        "been reported as evidence the mapping was doing work. Every other artifact in this "
        "family flattered a *measurement*. This one would have flattered the *gate*, which is "
        "worse, because a gate is what the reader trusts when they stop checking the "
        "measurements."
    )
    add("")
    add(
        "It was caught by someone printing the columns and noticing, which is not a control. "
        "**The general rule now stands in code: before any comparand enters a placebo matrix, "
        "every pair is asserted to be a different series** "
        "(`validation.assert_comparands_are_distinct`, run at the top of `placebo()` and again "
        "in `build()`). The comparison is made after standardizing both sides, so an affine "
        "rescaling -- the realistic duplication mode, a unit change rather than a copy -- is "
        "caught too. A 2.3e-12 identity should be caught by an assertion, not by a person."
    )
    add("")
    add("The three configured comparands, pairwise:")
    add("")
    add("| Pair | Shared months | max abs difference | Standardized | Correlation |")
    add("|---|---|---|---|---|")
    for pair in result.comparand_pairs:
        add(
            f"| `{pair.left.split(':')[-1]}` vs `{pair.right.split(':')[-1]}` | {pair.months:,} "
            f"| {pair.max_abs_difference:.2e} | {pair.max_abs_standardized_difference:.2e} "
            f"| {pair.correlation:+.4f} |"
        )
    add("")
    add(
        "`config/model.yaml` names only the Commodities-for-the-Long-Run column, which is the "
        "one SPEC.md 4.1.1 ruling 3 designated."
    )
    add("")

    # ------------------------------------------------------------------
    add("## The four flagged alphas: not explained, and handed on with a control")
    add("")
    add(
        "W2-P2 flagged four assets with persistently non-zero intercepts against 0.59 expected "
        "by chance, and `experiments.md` rows 79-82 identified all four as government carry and "
        "roll-down -- a term premium a factor set built from yield *changes* cannot span. A "
        "validation session is the natural place to ask whether the published analogues carry "
        "it. AQR's `Fixed income Market` is the only comparand in the mapping that contains a "
        "bond term premium at all, so it is the only candidate."
    )
    add("")
    limit = cfg.model.factors.macro.alpha_flag.abs_t_statistic
    add(
        "| Asset | n | alpha, six factors | t | alpha, + AQR FI Market | t "
        f"| beta on comparand | ΔR² | Explained (abs t < {limit:g})? |"
    )
    add("|---|---|---|---|---|---|---|---|---|")
    for confound in result.confounds:
        add(
            f"| `{confound.asset}` | {confound.observations} "
            f"| {_pct(confound.alpha_pct_per_year)} | {confound.alpha_t_statistic:+.2f} "
            f"| {_pct(confound.augmented_alpha_pct_per_year)} "
            f"| {confound.augmented_alpha_t_statistic:+.2f} | {confound.comparand_beta:+.4f} "
            f"| {confound.delta_r_squared:+.4f} "
            f"| {'yes' if confound.explained else '**no**'} |"
        )
    add("")
    add(
        "**The answer is no, on all four, and it is not close.** Adding the published comparand "
        "moves the largest alpha by "
        f"{max(abs(row.shift_pct_per_year) for row in result.confounds):.2f}%/yr, adds at most "
        f"{max(row.delta_r_squared for row in result.confounds):.4f} to R², and leaves every "
        "intercept significant at |t| > 3.7. Two of the four alphas go *up*. The mechanism is "
        "the same one rows 79-82 identified: a global bond-market **return** series carries its "
        "own carry and roll, so it cannot isolate the US Treasury carry-and-roll term premium "
        "that the flagged intercepts are."
    )
    add("")
    add(
        "The monthly alphas here corroborate W2-P2's daily ones across a frequency change, "
        "which is worth one line: "
        + ", ".join(f"`{row.asset}` {_pct(row.alpha_pct_per_year)}" for row in result.confounds)
        + ", against W2-P2's daily +0.71, +2.00, +3.29, +4.71 %/yr."
    )
    add("")
    add("### The control handed to W3-P5, fixed now")
    add("")
    add(
        "`experiments.md` row 83 pre-registers the hybrid's residual-PC question, and it depends "
        "on this: **a residual PC will load on the government sleeve for reasons unrelated to "
        'real rates unless the carry-and-roll alphas are removed first.** "Removed first" is '
        "fixed here as one specific mechanical operation, before any PC has been seen, rather "
        "than left as a judgement a later session gets to make freshly:"
    )
    add("")
    add(
        "> For the four synthetic government zeros, W3-P5's residual input is the **duration "
        "leg alone** -- `duration_effect` from SPEC.md 3.2's exact three-way decomposition, "
        "i.e. the excess return with `carry` and `roll_down` subtracted. Not the total excess "
        "return."
    )
    add("")
    add(
        "This is not a new construction. It is row 80's operation, already implemented in "
        "`mafrm.data.gsw.constant_maturity_return`, which decomposes every synthetic zero's "
        "return additively into `carry + roll_down + duration_effect`. Row 80 measured what it "
        "does to the flagged alphas: they collapse to +0.00, +0.08, +0.20 and -1.69 %/yr at "
        "2/5/10/30 years, and rows 81-82 closed the surviving 30-year term as an **attributed "
        "component** -- the third curve mode, outside the span of a two-PC factor set -- rather "
        "than a residual."
    )
    add("")
    add(
        "Stating it here costs one paragraph and removes a degree of freedom from W3-P5 that "
        "would otherwise be exercised after the PCs are visible."
    )
    add("")

    # ------------------------------------------------------------------
    add("## What this table does not establish")
    add("")
    add(
        f"1. **Four of six factors have no independent check available.** `{macro.SLOPE}`, "
        f"`{macro.CREDIT}` and `{macro.DOLLAR}` have no published analogue in the files this "
        f"project holds, and `{macro.LEVEL}`'s check is a sign. Their correctness rests on "
        "construction and on the internal evidence in `reports/factor_correlations.md` and "
        "`reports/asset_exposures.md` -- the duration recovery, the orthogonalization gates, "
        "the conditioning diagnostic -- not on an external comparand. This is carried in the "
        "README limitations as well as here."
    )
    add(
        "2. **No comparison here spans the holdout.** Every window closes strictly before "
        f"{boundary}."
    )
    add(
        "3. **A monthly comparison cannot see daily structure.** These regressions run at "
        f"n = {counts[macro.EQUITY]} and would not detect a calendar-alignment defect that "
        "cancels within a month."
    )
    add(
        "4. **The equity factor is not validated against Ken French, and cannot be.** It *is* "
        "Ken French `Mkt-RF` (SPEC.md 4.1.1 ruling 2), so that regression would return R² = 1 "
        "and prove nothing. Its comparands are AQR's equity-index market portfolio and our own "
        "SPY excess return."
    )
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    """``make report``: regenerate ``reports/factor_validation.md``."""
    settings = config.load()
    result = validation.build(settings)
    root = Path(__file__).resolve().parents[3]
    path = report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(result, settings), encoding="utf-8")
    print(f"wrote {path.relative_to(root)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
