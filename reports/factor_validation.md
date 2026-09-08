# Factor validation against published series

Generated 2026-08-31 by `mafrm.factors.validation_report`. SPEC.md 6.5 (last row) as amended by SPEC.md 6.5.1; task W2-P3.

SPEC.md 6.5 asks this table to turn *"trust me, my factors are right"* into something falsifiable, and names it a headline artefact rather than an appendix. What it turns out to establish is narrower than the spec assumed, and the narrowing is the result rather than a shortfall.

## The headline: two of six factors have a published comparand

Model A has six factors. **3 are mapped** (`equity`, `rates_level`, `commodity`); **3 have no published analogue in the files this project holds**. One of the mapped three -- `rates_level` -- is a sign test only, so the count that survives as a full external check is **2: `equity` and `commodity`**, and **4 of the six factors have no independent check available**.

The mapping was written factor by factor from a printed column listing of the three AQR workbooks and the Ken French daily research table, **before any regression was run**. That ordering is the point: an earlier draft of this task assumed five of six factors would be validated, and the column listing is what prevented three of them from being regressed against the nearest available column.

| Factor | Comparand | Frequency | Status |
|---|---|---|---|
| Equity (`equity`) | `century_of_factor_premia` &rarr; `Equity indices Market` | monthly | full comparison |
| Rates level (`rates_level`) | `century_of_factor_premia` &rarr; `Fixed income Market` | monthly | sign test only |
| Rates slope (`rates_slope`) | **none** | -- | no analogue |
| Credit (`credit`) | **none** | -- | no analogue |
| Commodity (`commodity`) | `commodities_long_run` &rarr; `Excess return of equal-weight commodities portfolio` | monthly | full comparison |
| Dollar (`dollar`) | **none** | -- | no analogue |

### Why the 3 unmapped factors are unmapped

Each reason is sourced to what the files actually contain, not to recollection. `config/model.yaml` carries the same text so that a later session cannot quietly map one of them.

- **`credit`** -- No credit series exists in any of the four files. Century of Factor Premia has no credit asset class, TSMOM has no credit leg, Commodities for the Long Run is commodities, and the Ken French research factors are equity style premia.
- **`dollar`** -- No dollar-direction series exists in any of the four files. Century's `Currencies Value/Momentum/Carry/Multi-style` are cross-sectional long/short currency STYLE factors and are dollar-neutral by construction; `TSMOM^FX` is trend-following on FX. Neither is the level of the dollar.
- **`rates_slope`** -- No curve-slope series exists in any of the four files. Century's `Fixed income Carry` is a cross-sectional long/short across COUNTRIES sorted on yield over cash -- a different construction on a different cross-section, not a US 2s30s slope.

- **AQR TSMOM is a comparand for nothing in Model A.** SPEC.md 6.5's last row and SPEC.md 12's week-2 *done when* both name it, and **no factor in the set SPEC.md 4.1 specifies is a trend factor**, so neither is meetable as written. Both are amended in SPEC.md 6.5.1. Building a trend factor to satisfy the spec would be alpha research and is out of scope under CLAUDE.md; the sentence is retained there as an illustration of what strong evidence looks like, explicitly marked as not applicable here.

## The comparison is monthly, and that changes what the table is

All three AQR workbooks are monthly -- the source URLs end `-Monthly.xlsx`, and the manifest records 1,196 / 497 / 1,780 rows against the factor panel's 4,146 complete daily dates. Every regression below therefore runs at **n = 213 months** for the factors that open at `sample.start`, and **n = 201** for `commodity` after its 2008-04 burn-in -- not at 4,146 observations. **An alpha t-statistic at n = 213 is a different instrument from one at n = 4,146** and no row here should be read as though it were the latter.

Aggregation, stated because CLAUDE.md failure mode 1 sits directly in the path: the four decimal-return factors are **compounded** in log space, the two basis-point yield-change factors are **summed**, and each factor is aggregated over its own trading dates. Only a boundary month is ever dropped -- a month in which the factor's history opened or closed part-way through. The aggregation is asserted as an exact round trip against the daily series rather than trusted:

| Factor | Monthly n | Window | Boundary months dropped | Round-trip residual |
|---|---|---|---|---|
| `equity` | 213 | 2007-04..2024-12 | none | 0.0e+00 |
| `rates_level` | 213 | 2007-04..2024-12 | none | 5.7e-14 |
| `rates_slope` | 200 | 2008-05..2024-12 | 2008-04 | 0.0e+00 |
| `credit` | 200 | 2008-05..2024-12 | 2008-04 | 5.6e-17 |
| `commodity` | 201 | 2008-04..2024-12 | none | -2.8e-16 |
| `dollar` | 213 | 2007-04..2024-12 | none | -5.6e-17 |

AQR stamps its month-ends on the last **business** day (1926-07-30, 1877-02-28, 1985-01-31 in the three files) and this project stamps on the last **traded** day, so the join key is the calendar month and the surviving index is AQR's own observed date. The joined count is asserted against a count computed from the two coverages, so a month lost to a stamp mismatch cannot pass as a month the sources did not share.

**The right edge is `holdout_start` = 2025-01-01, always.** All three AQR files end *after* it -- 2026-02-27, 2026-05-29 and 2025-05-30 -- so their coverage can only ever bind the left edge or an earlier right edge. That is asserted per comparison rather than left to the intersection to deliver.

## The table

The self-built factor is the **dependent** variable, per SPEC.md 6.5's own wording. Both sides are in basis points per month, so `beta` on a return comparand is dimensionless and `beta` on the rate factor is in inverse years. `alpha` is the part of the factor the published series does not carry, annualised for display, with a Newey-West t-statistic on the Newey & West (1994) plug-in bandwidth `floor(4*(T/100)^(2/9))`.

| Factor | Comparand | n | Window (months) | Observed dates | R² | corr | beta (se) | units | alpha | t(alpha) | NW lags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `equity` | `Equity indices Market` | 213 | 2007-04..2024-12 | 2007-04-30..2024-12-31 | 0.7700 | +0.8775 | +0.99830 (0.03756) | dimensionless | +4.73%/yr | +2.79 | 4 |
| `rates_level` | `Fixed income Market` | 213 | 2007-04..2024-12 | 2007-04-30..2024-12-31 | 0.7291 | -0.8539 | -0.12699 (0.00533) | years^-1 | +0.27%/yr | +2.52 | 4 |
| `commodity` | `Excess return of equal-weight commodities portfolio` | 201 | 2008-04..2024-12 | 2008-04-30..2024-12-31 | 0.4891 | +0.6994 | +0.67854 (0.04916) | dimensionless | +1.87%/yr | +0.70 | 4 |

**Units warning, because two numbers in this repository must not be compared.** The `rates_level` row is a *monthly summed basis-point* yield change regressed on a *monthly* bond return. W2-P2's exposure panel reports *daily* coefficients on the same factor that read directly as minus an effective duration (-1.66, -5.61, -11.29, -27.46 years). **These are not on the same scale and neither is a check on the other.**

### What the two full comparisons say

**Equity correlates +0.877 with AQR's equity-index market portfolio** (R² 0.770). SPEC.md 6.5's own reading is that a correlation of 0.6-0.8 with the AQR analogue is strong evidence the concept is implemented correctly and 0.2 means something is wrong; this sits above that band. It should: both series are estimates of the return to owning equity.

**The +4.73%/yr intercept (t = +2.79) is not a defect, and the beta is the evidence for that.** AQR's series is a **global developed** equity-index futures portfolio; ours is Ken French's **US** market. So the intercept should be the US-over-global spread across 2007-2024, and it is -- arithmetically, not by assertion:

| | mean | annualised volatility |
|---|---|---|
| our `equity` (US `Mkt-RF`) | +10.15%/yr | 16.23%/yr |
| AQR `Equity indices Market` (global) | +5.43%/yr | 14.27%/yr |
| **difference** | **+4.72%/yr** | |

The mean gap is +4.72%/yr and the fitted alpha is +4.73%/yr -- the same number, because **beta = +0.998 (se 0.038) is indistinguishable from 1**. That is the whole argument: if the two series differed in *exposure*, the gap would show up in the slope and the intercept would absorb whatever was left over. It does not. They differ in **level** at the same unit exposure, which is what a regional performance spread looks like and is not something either construction got wrong. US equity beat global developed equity over this window; the factor is not miscalibrated.

**Commodity correlates +0.699 with AQR's equal-weight commodity index** (R² 0.489), inside SPEC.md 6.5's 0.6-0.8 band and well clear of the 0.2 that would mean something is wrong. Its intercept is +1.87%/yr at t = +0.70 -- indistinguishable from zero, which is the stronger of the two level results: our DBC-based factor and AQR's index disagree about the path but not about the level.

**Beta +0.679 and R² 0.489 are a *composition* difference, not a scale difference, and the volatilities settle which.** A reader who stops at R² will conclude the commodity factor explains half of what it should. Decompose the slope instead -- `beta = corr x sd(factor)/sd(comparand)`:

| | annualised volatility |
|---|---|
| our `commodity` (DBC excess) | 14.99%/yr |
| AQR equal-weight commodity index | 15.45%/yr |
| ratio | **0.970** |

The two carry **essentially the same volatility** (ratio 0.970), so `beta` +0.679 is almost entirely the correlation term +0.699 rather than a difference in scale. That distinction is the reading: a *leverage* or *scale* mismatch -- one series being a geared version of the other -- would show as a ratio away from 1 and would be a defect in one construction. Equal volatility at 0.70 correlation is what **two differently weighted baskets of the same asset class** look like. DBC tracks an index-weighted commodity basket with a concentrated energy weight and its own roll-optimisation schedule; AQR's is an equal-weight academic portfolio. The same risk, a different mix.

**The exact index weights of DBC are not recorded anywhere in this repository**, so the composition claim is the structural reading rather than a sourced decomposition. What *is* measured is the part that matters for the factor set: the two series carry the same amount of commodity risk and differ in which commodities.

### `rates_level` is a sign test, and a weak one

AQR's `Fixed income Market` is a global bond-market **excess return**; `rates_level` is a **US yield change in basis points**. They are not two estimates of one object, and the regression coefficient is

```
beta  =  -D_global x (dY_global / dY_US)
```

**a product of two quantities, neither of which is published anywhere in this repository's sources.** Any band wide enough to be defensible around that product is too wide to fail, and a gate that cannot fail is worse than no gate because it reads as evidence while supplying none. **No implied-duration band is pre-registered here, deliberately.**

The falsifier is the sign and nothing else: `beta` must be negative, because a bond return rises when yields fall. Measured **-0.12699** with a standard error of 0.00533 over 213 months -- **passes**.

#### The division every reader will do, done here

**Reported as a diagnostic and gated on nothing** -- but stated, because a reader who is not given the arithmetic will supply their own, and the obvious slip gives an answer half again too large.

`beta` = -0.12699 is in **basis points of yield per basis point of return**, because both sides of this regression are in basis points. So the implied duration is `1/|beta|` = **7.87 years** -- not `|beta| x 100` = 12.7 years, which is what the number gives if `beta` is read as a *percent* return per basis point. It is not: that would be the coefficient of the reverse regression in different units.

And `1/|beta|` is an upper edge rather than the answer, because the two regression directions attenuate toward whichever axis carries the noise. All three estimates:

| Estimate | Value | What it assumes |
|---|---|---|
| `|R²/beta|` -- regress AQR's return on our yield change | **5.74 yr** | all the noise is in the yield change |
| `sd(AQR)/sd(ours)` -- the orthogonal/volatility-ratio estimate | **6.72 yr** | noise in neither; the geometric mean of the other two |
| `1/|beta|` -- reciprocal of the fitted slope | **7.87 yr** | all the noise is in the return |

So: **implied duration 5.7-7.9 years, centred near 6.7.** Both unknowns in `D_global x (dY_global/dY_US)` remain unidentified and nothing in this project depends on the number.

#### Is AQR's series vol-scaled? Measured, and no

This matters for reading the duration: **a construction targeting a constant volatility would need either a much longer duration or leverage**, and roughly 6.7 years would then be the wrong reading of the coefficient. Many AQR datasets do target a constant volatility, so the question is live rather than pedantic -- and it is answerable from the file rather than from recollection.

`Fixed income Market` realises **5.33%/yr** over the comparison window and **3.59%/yr** over its whole history back to 1926 (read, like everything else here, up to the holdout boundary). Across all 44 columns of Century of Factor Premia the annualised volatility runs 2.2% to 20.1% with a median of 6.2%, and only 4 of 44 sit between 9% and 11%. **This dataset is not scaled to a volatility target.**

So AQR's series behaves like an **unlevered intermediate-duration government bond portfolio**: roughly 6.7 years of duration realising 5.33%/yr against our measured US yield volatility of 79 bp/yr. **That the three numbers reconcile is arithmetic, not corroboration** -- the volatility-ratio estimate is *defined* as one divided by the other, so it reproduces them by construction and confirms nothing. What does the work is the comparison against 10%: the series would have to realise roughly twice its measured volatility to be vol-targeted, and it does not.

**Whether it is a cash index or unlevered futures is not documented anywhere in this repository, and is stated as unknown rather than guessed.** The distinction does not affect the sign test, which is the only thing gated.

**This test adds independence, not power, and the report says so plainly.** The strong test of the level factor already ran in W2-P2: daily, on US data, at n = 4,146, against durations this project knows independently, and five of six instruments landed inside their pre-registered bands (`experiments.md` rows 73-77). What the AQR comparison adds is that the factor also tracks a series produced by someone else from different data -- at R² 0.729, which is higher than a transform of this kind had any right to deliver.

## Rolling correlation

A **36-month** rolling window. This is a **choice**, not a published constant: it was fixed by the operator in the W2-P3 brief before the run, nothing was swept against it, and `config/model.yaml` records it as a choice. No `experiments.md` row is owed, because a row would record a search that did not happen.

**The standard error is printed beside every window and it is large.** At 36 points the Fisher-z standard error is `1/sqrt(36-3)` = **0.174**, so a drift from 0.55 to 0.75 is roughly one standard error and must not be read as a change in anything.

| Factor | Windows | Min | Median | Max | Range in Fisher-z SEs | First..last |
|---|---|---|---|---|---|---|
| `equity` | 178 | +0.619 | +0.879 | +0.940 | 5.8 | 2010-03..2024-12 |
| `rates_level` | 178 | -0.916 | -0.851 | -0.778 | 3.0 | 2010-03..2024-12 |
| `commodity` | 166 | +0.570 | +0.710 | +0.877 | 4.1 | 2011-03..2024-12 |

**Every mapped factor holds its sign in every window** -- which is the property worth having, and the one a rolling correlation is actually good for.

The Fisher-z column is there to calibrate a *local* drift, not to license a claim that the whole range is noise. Read the two together and neither reading is available: the full ranges span 3-6 standard errors, so they are **not** within sampling error of a constant correlation -- but these are 36-month windows stepped one month at a time, overlapping in 35 of 36 observations, so the windows are nowhere near independent draws and the span cannot be read as 3-6 independent standard errors either. What the column does support is the comparison it was put there for: two nearby readings differing by less than about 0.17 -- a drift from 0.55 to 0.75, say -- are not distinguishable, and this report makes no claim that rests on one.

## The gates, and why 0.3 was withdrawn rather than lowered

W2-P3 was originally given a **0.3 correlation gate**. It is withdrawn, and this section has to make that argument rather than assert it, because *withdrawing a gate the run would have passed anyway is exactly how a project talks itself out of a bar it did not like*.

**The argument is the SPEC.md 4.1.2 precedent.** The all-pairs 0.15 correlation bar was withdrawn in W2-P1 because it *measured something other than what it claimed to* -- meeting it would have driven the factor correlation matrix toward diagonal and left week 3's eigenfactor adjustment nothing to correct. It was invalid in kind, not mis-calibrated. The same test applies here and gives the same answer for a different reason: **0.3 was a threshold for a comparand that does not exist.** SPEC.md 6.5's only published figures -- 0.6-0.8 strong, 0.2 wrong -- are stated for a *trend* factor against AQR TSMOM, and Model A has no trend factor. The number was written before anyone had looked at what the AQR files contain, and no published figure covers any pairing that actually exists in this project.

**It is also worth noting what withdrawing it does not buy.** All three mapped factors clear 0.3 in absolute correlation (`equity` 0.877, `rates_level` 0.854, `commodity` 0.699), so the gate is withdrawn from a position of not needing it. That is the only position from which withdrawing a gate is legitimate.

Three pre-registered falsifiers replace it, and each can fail.

### (i) The daily anchor -- `experiments.md` row 41

Row 41 measured our own SPY total return against Ken French `Mkt-RF` at correlation **0.9786** with a mean gap of **-0.122%/yr** over 8,410 days. That is the only sharp prior in this session: a measured expectation rather than a guessed threshold, and the calibration point for what *two constructions of the same object* look like on this project's data. The equity factor is `Mkt-RF` itself, so its comparand is SPY excess -- which SPEC.md 4.1.1 ruling 2 converted from a candidate input into a comparand.

| Window | n | Dates | Correlation | 95% Fisher-z interval | Mean gap |
|---|---|---|---|---|---|
| row 41 window, truncated at holdout | 8,037 | 1993-02-01..2024-12-31 | 0.9779 | [0.9769, 0.9788] | -0.11%/yr |
| W2-P3 in-sample | 4,469 | 2007-04-02..2024-12-31 | 0.9911 | [0.9906, 0.9916] | -0.24%/yr |
| pre-sample only | 3,568 | 1993-02-01..2007-03-30 | 0.9548 | [0.9518, 0.9576] | +0.05%/yr |

**Passes.** Row 41's 0.9786 lies inside the sampling interval of the reproduction on row 41's own window ([0.9769, 0.9788]), and the mean gap comes back at -0.11%/yr against a registered -0.122%/yr. The window is row 41's, truncated at `holdout_start`: row 41 ran to the end of the cache and this session may not, so 373 days are dropped and the small difference from the registered figure is that truncation.

**The in-sample window gives 0.9911, not 0.9786, and that difference is a window difference rather than sampling error — measured, not asserted.** The two intervals do not overlap, so the distinction matters. The pre-sample period 1993-2007 correlates 0.9548 and the in-sample period 0.9911; the full-history 0.9786 is a blend of the two. SPY's tracking of the CRSP market improved after 2007, which is unsurprising -- the small-cap complement's dispersion against the S&P 500 was larger in the dot-com era. **Comparing the in-sample number to row 41's headline and calling the difference a failure would charge a window difference to sampling error**, which is the error this report's own *every comparison states its own window* rule exists to prevent.

### (ii) The placebo -- each factor beats every other factor's comparand

Regress each mapped factor on the comparands belonging to the *other* mapped factors. Each factor's R² against its own comparand must exceed its R² against every other. **This is falsifiable without any constant at all**: if `commodity` explained AQR's Fixed income Market as well as it explains AQR's commodity index, the mapping would be doing no work and this table would be decoration. It is the same move as the IVV/VOO control (row 63) and the zero-spread simulation -- a gate built from measurement rather than from a guessed threshold.

**Read row-wise.** Each row holds the dependent variable and its window fixed and varies only the comparand; reading down a column compares different windows.

| Factor (rows) &darr; / comparand &rarr; | `equity` | `rates_level` | `commodity` | Own | Margin | Result |
|---|---|---|---|---|---|---|
| `equity` | **0.7700** | 0.0001 | 0.2533 | `equity` | +0.5167 vs `commodity` | pass |
| `rates_level` | 0.0411 | **0.7291** | 0.0529 | `rates_level` | +0.6762 vs `commodity` | pass |
| `commodity` | 0.0646 | 0.1984 | **0.4891** | `commodity` | +0.2908 vs `rates_level` | pass |

**Passes on every row.** The nearest call is `commodity`, which loads on `rates_level`'s comparand at R² 0.198 and still beats it by 0.291. The off-diagonal number this gate exists to produce is `equity` against the bond comparand at R² 0.0001: a mapping that is doing work looks like that away from its own diagonal.

### (iii) `rates_level` beta is negative

Measured -0.12699 -- **passes**. A failure here would have meant something badly wrong with the level factor's sign convention, which SPEC.md 4.1.1 fixes as a by-product of the normalization.

**All three hold.** No gate was adjusted after seeing a number.

## Two comparands that turned out to be one series

Century of Factor Premia's `Commodities Market` and Commodities-for-the-Long-Run's `Excess return of equal-weight commodities portfolio` agree to **1.9e-12** over all **1,182** overlapping months. They are the same series.

This was checked rather than assumed, and it changed the design. The mapping as approved listed **both** as commodity comparands, on the reasoning that a second independent AQR construction costs nothing. It is not a second construction.

**This is the sharpest instance yet of the project's flattering-direction artifact family, and the reason is specific.** The placebo of §(ii) was introduced precisely because it needs no invented constant -- it is the gate that replaced 0.3. A duplicated comparand would have made it **pass by tautology while looking like an independent check**: `commodity` would have been scored against a second copy of its own series, "beaten" it by whatever rounding separated them, and the pass would have been reported as evidence the mapping was doing work. Every other artifact in this family flattered a *measurement*. This one would have flattered the *gate*, which is worse, because a gate is what the reader trusts when they stop checking the measurements.

It was caught by someone printing the columns and noticing, which is not a control. **The general rule now stands in code: before any comparand enters a placebo matrix, every pair is asserted to be a different series** (`validation.assert_comparands_are_distinct`, run at the top of `placebo()` and again in `build()`). The comparison is made after standardizing both sides, so an affine rescaling -- the realistic duplication mode, a unit change rather than a copy -- is caught too. A 2.3e-12 identity should be caught by an assertion, not by a person.

The three configured comparands, pairwise:

| Pair | Shared months | max abs difference | Standardized | Correlation |
|---|---|---|---|---|
| `Excess return of equal-weight commodities portfolio` vs `Equity indices Market` | 1,182 | 2.66e-01 | 7.26e+00 | +0.2163 |
| `Excess return of equal-weight commodities portfolio` vs `Fixed income Market` | 1,182 | 2.42e-01 | 6.07e+00 | -0.1090 |
| `Equity indices Market` vs `Fixed income Market` | 1,182 | 2.59e-01 | 8.72e+00 | +0.1210 |

`config/model.yaml` names only the Commodities-for-the-Long-Run column, which is the one SPEC.md 4.1.1 ruling 3 designated.

## The four flagged alphas: not explained, and handed on with a control

W2-P2 flagged four assets with persistently non-zero intercepts against 0.59 expected by chance, and `experiments.md` rows 79-82 identified all four as government carry and roll-down -- a term premium a factor set built from yield *changes* cannot span. A validation session is the natural place to ask whether the published analogues carry it. AQR's `Fixed income Market` is the only comparand in the mapping that contains a bond term premium at all, so it is the only candidate.

| Asset | n | alpha, six factors | t | alpha, + AQR FI Market | t | beta on comparand | ΔR² | Explained (abs t < 2)? |
|---|---|---|---|---|---|---|---|---|
| `govt_2y` | 200 | +0.72%/yr | +4.15 | +0.74%/yr | +4.08 | -0.0157 | +0.0007 | **no** |
| `govt_5y` | 200 | +2.12%/yr | +5.87 | +2.07%/yr | +5.53 | +0.0393 | +0.0005 | **no** |
| `govt_10y` | 200 | +3.26%/yr | +4.82 | +3.00%/yr | +4.65 | +0.2131 | +0.0028 | **no** |
| `govt_30y` | 200 | +4.13%/yr | +3.64 | +4.30%/yr | +3.75 | -0.1395 | +0.0002 | **no** |

**The answer is no, on all four, and it is not close.** Adding the published comparand moves the largest alpha by 0.26%/yr, adds at most 0.0028 to R², and leaves every intercept significant at |t| > 3.7. Two of the four alphas go *up*. The mechanism is the same one rows 79-82 identified: a global bond-market **return** series carries its own carry and roll, so it cannot isolate the US Treasury carry-and-roll term premium that the flagged intercepts are.

The monthly alphas here corroborate W2-P2's daily ones across a frequency change, which is worth one line: `govt_2y` +0.72%/yr, `govt_5y` +2.12%/yr, `govt_10y` +3.26%/yr, `govt_30y` +4.13%/yr, against W2-P2's daily +0.71, +2.00, +3.29, +4.71 %/yr.

### The control handed to W3-P5, fixed now

`experiments.md` row 83 pre-registers the hybrid's residual-PC question, and it depends on this: **a residual PC will load on the government sleeve for reasons unrelated to real rates unless the carry-and-roll alphas are removed first.** "Removed first" is fixed here as one specific mechanical operation, before any PC has been seen, rather than left as a judgement a later session gets to make freshly:

> For the four synthetic government zeros, W3-P5's residual input is the **duration leg alone** -- `duration_effect` from SPEC.md 3.2's exact three-way decomposition, i.e. the excess return with `carry` and `roll_down` subtracted. Not the total excess return.

This is not a new construction. It is row 80's operation, already implemented in `mafrm.data.gsw.constant_maturity_return`, which decomposes every synthetic zero's return additively into `carry + roll_down + duration_effect`. Row 80 measured what it does to the flagged alphas: they collapse to +0.00, +0.08, +0.20 and -1.69 %/yr at 2/5/10/30 years, and rows 81-82 closed the surviving 30-year term as an **attributed component** -- the third curve mode, outside the span of a two-PC factor set -- rather than a residual.

Stating it here costs one paragraph and removes a degree of freedom from W3-P5 that would otherwise be exercised after the PCs are visible.

## What this table does not establish

1. **Four of six factors have no independent check available.** `rates_slope`, `credit` and `dollar` have no published analogue in the files this project holds, and `rates_level`'s check is a sign. Their correctness rests on construction and on the internal evidence in `reports/factor_correlations.md` and `reports/asset_exposures.md` -- the duration recovery, the orthogonalization gates, the conditioning diagnostic -- not on an external comparand. This is carried in the README limitations as well as here.
2. **No comparison here spans the holdout.** Every window closes strictly before 2025-01-01.
3. **A monthly comparison cannot see daily structure.** These regressions run at n = 213 and would not detect a calendar-alignment defect that cancels within a month.
4. **The equity factor is not validated against Ken French, and cannot be.** It *is* Ken French `Mkt-RF` (SPEC.md 4.1.1 ruling 2), so that regression would return R² = 1 and prove nothing. Its comparands are AQR's equity-index market portfolio and our own SPY excess return.

