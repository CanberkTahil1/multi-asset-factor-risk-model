# Which stages of the USE4 pipeline earn their keep at K = 6 — and, since W7-P4, at K = 54

**Maintained by hand, appended to as each stage of SPEC.md 5 lands.** Started
2026-08-31 in W3-P3, at three stages rather than at the end, so that the pattern
is recorded as it accumulates rather than reconstructed afterwards. Every number
here is a cross-reference to a generated report or an `experiments.md` row; this
file synthesises and does not measure. Carried into **W8-P1** beside the `K/T`
scaling result.

## Why this file exists

The project's headline is that **estimation error scales with `K/T`**. SPEC.md
15.2 is built to demonstrate it: the same code runs a `K = 6` macro model and a
`K = 56` equity cross-sectional model, and Shepard's `[1 − K/T_eff]⁻²` moves from
a 1.7% understatement to a large one across that range.

The finding accumulating beside it is a different one, and it was not planned:

> **The industry-standard corrections for that error are themselves `K`-dependent,
> and a small macro model pays for several that do nothing for it.**

That is worth stating deliberately, because it has now been noticed three
separate times in three separate sessions, each time as a local curiosity. It is
not a criticism of USE4 — USE4 is a model with dozens of factors and every one of
these corrections is doing real work there. It is a statement about what happens
when a published pipeline designed for one regime is run faithfully in another,
which is exactly what this project set out to do and exactly the kind of thing
that is normally left unsaid.

**The distinction this table has to keep making is between a stage and its
machinery.** SPEC.md 5.3's eigenfactor adjustment earns its keep at `K = 6`
comfortably; the *parabola inside it* does almost nothing. Collapsing those two
into one claim would be the easy version of this finding and the wrong one.

## The table

**Three conventions in this table.** Cells marked **[P]** are **predictions** made
before the `K = 56` module existed; cells marked **[M W7-P4]** are the same predictions
**measured** in W7-P4 at `K_d` = 51–54 (SPEC.md 15.7.1, 15.7.2; `reports/equity_bias_statistics.md`);
every other cell is a measurement with a citation. And each row names the stage as it appears in `covariance.stages` in
`config/model.yaml`, in backticks -- `tests/test_stage_k_dependence.py` asserts
every declared stage has a row here, so a stage cannot be added without this file
finding out.

The **"where in the spectrum"** column was added on the operator's instruction
after W3-P3b, and it is the column that makes the table say something a list of
verdicts cannot: a stage that earns its keep on the bulk and does nothing for the
dominant factors is not the same object as one that acts uniformly, even if both
"earn their keep".

| Stage | SPEC.md | Where in the spectrum it acts | What it does at `K = 6` | Evidence | `K = 56` |
|---|---|---|---|---|---|
| `ewma` — separated volatility and correlation | 5.1 | Everywhere; it *is* the spectrum | Not a correction — this is the estimator. `K`-independent by construction. | `reports/mean_convention.md` | **[M W7-P4]** **Same.** Positive-definite on every one of 420 month-end builds at `K_d` = 51–54; the first build's `K/T_eff` is 0.64, the last 0.037 |
| ↳ moments about zero rather than demeaned | 5.1.1 | The diagonal only — a level effect, no rotation | Median shift **10.2 bp** in a factor volatility (short), tail **323 bp** on `dollar`. A coherence requirement, not a correction, and **not `K`-dependent**. | rows 93–95 | **[M W7-P4]** **Same in kind.** Not re-measured on the equity panel; a coherence requirement, not a correction |
| `newey_west` — Bartlett correction | 5.2 | Everywhere; rescales both legs, changes conditioning barely | Moves factor volatilities by a median **−3.5%** (short). On a fixture serially uncorrelated *by construction*, Frobenius error against the known truth **rises 1.6–3.2×** — it estimates lag terms that are zero in expectation and pays the variance. Depends on real serial correlation, not on `K`. | rows 97, 101 | **[M W7-P4]** **Measured.** Moves family-4 `B` from 1.059 (`ewma`) to 1.100 (short) — the lag terms are net *positive* on the equity factor set, the opposite sign to Model A's −3.5%. Non-PSD after it on the first month-end build only (smallest eigenvalue −0.83 at `K/T_eff` = 0.84) |
| ↳ `newey_west` non-positive variance, and its fallback | 5.2.3 | **The diagonal** — one factor's variance, not the spectrum | **FIRED in W3-P5, at `K = 3` and not the `K = 56` it was registered for.** Model B's detoned panel, `T = 7–10`, `K/T ≈ 0.43`, 2–3 dates of 4,143, 2 contiguous episodes. Denoised never fires. Row 100's registered fallback to the stage-1 EWMA variance applied unchanged. **A `K/T` phenomenon, not a `K` one** — same shape as the row below, and plain iid noise at `T = 7` against 5 lags reproduces it. | row 100 + its W3-P5 amendment, row 129, §5.2.3 | **[M W7-P4]** **Did not fire at `K_d` = 51–54** (row 293(e), 0 of 420 builds), including the first build at `K/T_eff` = 0.64 — above the 0.43 at which it fired for Model B's `K = 3` — so the ratio alone does not predict it: the sign of the lag terms does, and on the equity factor set they are net positive. Every later build sits below 0.12 |
| `psd_repair` | 5.2 | **The bottom of the spectrum only** — the single most negative eigenvalue | **Fires on 5 dates of 4,141 builds, all at `T ≤ 10`** against `K = 6`; realised `K/T_eff` 1.000 down to 0.600. Zero firings from `T = 11` on. In production a no-op. | row 98, `reports/psd_repairs.md`, §5.2.2 | **[M W7-P4]** **Fired once per horizon, on the first month-end build** (2007-06-29, `K/T_eff` = 0.84, one direction floored), never after (row 293(c)); the first eigenfactor forecast is the following month-end, so **no firing reaches a scored forecast** (row 293(d)). Live, and confined to `K/T_eff` above 0.6 exactly as at `K = 6` |
| `eigenfactor` | 5.3 | **The bulk, hardest at the smallest eigenfactors; least at isolated ones** | **Earns its keep, and by less than W3-P3 reported.** Raises the minimum-variance forecast by **+0.54%** (`a=1.0`) / **+0.76%** (`a=1.4`) against **+0.08%** for the median random portfolio — a **10×** ratio, MWO's mechanism. (W3-P3's +3.80%/+5.33% used the wrong `T`; see W3-P3b.) | `reports/eigenfactor.md` | **[M W7-P4]** **Where it earns its keep, measured.** Family-4 `B` **1.100 → 1.044** at `a = 1.0` and **1.025** at `a = 1.4` (short, daily-held, `T` = 3,985, half-width 0.022) — 55% and 76% of the gap to 1 — against **−0.0001** at `K = 6`; family 3's MRAD 0.120 → 0.090. The component split (row 295) says why: the specific `B` is 1.03, inside its interval, and the whole excess is the FACTOR component (1.19 → 1.10). Same code, `src/mafrm/risk/` untouched |
| ↳ where it does **least** | 5.3 | **Isolated / dominant eigenvalues** — λ returns toward 1 | Measured, not assumed: a well-separated eigenvalue suffers almost no eigenvector rotation, and rotation is what the bias is made of. On the golden fixture the two spikes sit above the bulk's last value. | row 107 | **[M W7-P4]** **Pronounced, as registered (row 293(b)).** Raw `λ` at the dominant eigenvalue 0.985 (median) above the bulk minimum 0.966 on 100% of scored month-ends, while the smallest eigenvalue's is 1.047 |
| ↳ the λ-curve amplitude | 5.3 | The spread across the whole spectrum | **0.0099**, identical at both horizons, against the published large-`K` shape of ≈0.55. The amplitude gate is **withdrawn** (§5.3.2); the *direction* gate is kept, as a bulk claim. | `reports/eigenfactor.md` | **[M W7-P4]** **0.1136** median (range 0.097–0.253), **eleven times** the 0.0099 at `K = 6`, tracking `K/T_eff` through the grid (2.78 at 0.64 on the first build). Falsifier not met; the `K/T` scaling holds (row 293(a)) |
| ↳ the parabola that smooths it | 5.3 step 7 | **The ends of the spectrum**, where simulation noise is worst | **Close to a no-op.** Three parameters on `K` points: **3 residual d.o.f.** at `K = 6`, **zero** at the `K = 3` fixture, where it interpolates exactly and removes no noise at all. | `reports/eigenfactor.md` | **[M W7-P4]** **51 residual d.o.f.** at `K_d` = 54; visibly smoothing (`reports/equity_lambda_curve.png`) — the regime it was designed for |
| ↳ its coverage after the correlation-space ruling | 5.3.2 | `rho` only — **`sigma`'s error passes through uncorrected** | Shepard at the correlation window is **0.84%**; at the volatility window **5.14%**. The gap is, to order of magnitude, the **uncorrected volatility leg**. A real cost of the scale-invariance ruling, open. | `reports/eigenfactor.md`, W3-P3b | **[M W7-P4]** **Same in kind.** Shepard's Eq. 32 at `K = 54`, volatility window (`T_eff` 242, upper bound) gives 1.287 on volatility against a measured pre-adjustment 1.100; at the correlation window (`T_eff` 1,454) 1.038. The measured bias sits between the two windows' closed forms, as W4-P3's disjointness argument predicts |
| `volatility_regime` | 5.4 | A **uniform** level scaling — every eigenvalue equally, correlation structure untouched | **Earns its keep, and the two horizons disagree about the sign — which is the finding.** `λ_F` = **1.016 short** (+1.6% on vol, near-calibrated) against **0.920 long** (−8.0%: a 252d half-life carries crisis variance through the calm that follows and the VRA marks it down). Responds to both reachable episodes, harder at the faster half-life: Sep 2008 **1.75 / 1.62**, Mar 2020 **1.76 / 1.40**. Not `K`-dependent in itself — a cross-sectional mean over `K` factors is not a `K`-limited estimator. | rows 112–115, `reports/volatility_regime.md` | **[M W7-P4]** **Same in kind, and the sign pattern repeats:** `λ_F²` 1.14 short / 0.94 long at the last scored month-end (medians 1.04 / 1.10 over the window), fitted on daily `B_t^F` under the held forecast. Post-VRA rows reported, not tested (SPEC.md 6.2.2) |
| ↳ how much of §5.3's correction it absorbs | 5.4.2 | The level only — it cannot see the off-diagonals | **Third-order, and predicted before it was measured: 0.017–0.033% of volatility.** `B_t^F` standardises factor by factor, so it sees only `diag(F)`, and after the correlation-space ruling §5.3 reaches the diagonal only through `ρ_adj,kk`. **A per-factor standardisation is blind to the off-diagonal correction by construction.** | rows 116–117 | **[P]** Not re-measured on the equity panel in W7-P4: larger, and to be watched in W8 where the diagonal inflation is 11x |
| **PAIRING** `psd_repair` × `eigenfactor` | 5.2 + 5.3 | The bottom of the spectrum, then **all `K` values of `γ` through the fit** | **Negligible here, and only because the repair is a no-op here.** Where the repair fires (4 dates, `K/T_eff` 0.86→0.67) the forecast inflates ~**10⁴**: a floored `1e-14` eigenvalue is a manufactured number, the Monte Carlo measures an enormous `λ(k)` for it, and the **parabola spreads that through every `γ(k)`**. Worth **≤1.1e-9** on `λ_F²`. | §5.4.4, `reports/volatility_regime.md` | **[M W7-P4]** **Not on the production path at `K_d` = 51–54** (row 293(d)): the one firing per horizon precedes the first eigenfactor forecast. Row 118's remedy stays registered, unimplemented, for W8 with that count |
| Specific risk: time-series estimate | 5.5(a) | Not the factor spectrum at all | **Works at any `N`.** EWMA plus a Bartlett term over 5 lags; moves specific vol by −41% (`govt_5y`) to +2% (`us_small_equity`), median −14.1% short. The row-100 fallback never fires. | `reports/specific_risk.md` | **[M W7-P4]** **Same, per name with gaps removed** (SPEC.md 15.7.1 ruling 3): ~400 names per month-end, the row-100 fallback fired 70 times over 190 × 400 name-months on the specific leg |
| Specific risk: structural estimate | 5.5(b) | Not the factor spectrum at all | **Barely identified.** 13 observations against 7 parameters — **6 residual d.o.f.** — `R² = 0.833`, and `σ^STR` misses `σ^TS` by **0.31× to 4.17×**. Built and reported; **never reaches the forecast**, because the blend below is pinned. | `reports/specific_risk.md`, §5.5.2 | **[M W7-P4]** **Where it earns its place, measured:** `R²` median **0.998** on **336** residual d.o.f. (against 0.833 on 6). Still never reaches the forecast except for names below the arithmetic floor (construction 4), because the blend is pinned |
| Specific risk: the data-quality blend | 5.5(c) | Not the factor spectrum at all | **INVERTED, and this is the finding.** The observation term saturates (complete panel, `min h_n = 3,889`); the fatness term does **not** (`max Z = 7.61`). `γ_n < 1` would route weight from leg (a) onto leg (b) — the *worse* estimate here — and it fires in March–April 2020, precisely where leg (b) is fitted on the same stressed data. `γ_n` pinned at 1 by ruling. | §5.5.2, rows 137–139 | **[M W7-P4]** **Not tested — γ stays pinned by ruling.** The reversal prediction needs `γ_n < 1` switched on, which is a `model-config` trial nobody has ruled on; the `Z` diagnostic is reported (its maximum is infinite for a name with a zero interquartile range) |
| Specific risk: Bayesian shrinkage | 5.5(d) | Not the factor spectrum at all | **Vacuous on 3 of 5 populated buckets, and exactly so.** At two contributors `d = σ_Δ` identically, so `v = 1/(1+q) = 0.9091` **whatever the data**; at one it is the identity. After W4-P1b holds the two identity assets out of the target, `commodity`, `credit` and `inflation` all have one contributor: stage (d) is live on **8 of 13 assets**. | §5.5.5, §5.5.4, `tests/test_specific_risk.py` | **[M W7-P4]** **Informative on ~40 contributors per decile, and it does exactly what its formula says (row 294, REFUTED): a singleton's `σ_TS ≈ 0` is an EXTREME value, `v ≈ 1`, and it is shrunk LEAST** — the 13 singleton names sit at the 0.7th cross-sectional percentile after shrinkage, 1,285 cells. The published mechanism does not rescue a construction artefact; no constant added, flagged to the operator. **W7-P4b (row 298):** by ruling the singleton takes leg (b)'s structural estimate — §5.5's own fallback for a name with no information — and moves to the 48th percentile; family 4's weight on the 13 names 7.2% → 4.3%, family-4 `B` 1.100 → 1.087 |
| ↳ what it does to a value that is a construction artefact | 5.5(d) | Not the factor spectrum at all | **It propagates it, before any aggregate is taken.** Two assets whose residual is ~0 by construction pulled their bucket-mates down **4.6–4.8%** and were themselves inflated **+1434%** / **+167%**. Fixed in W4-P1b by holding them out of the target, not by re-bucketing. **`N`-independent — it is a property of pooling, not of pool size.** | §5.5.4, row 143 | **[M W7-P4]** **Same mechanism, held out by the W4-P1b constraint (construction 5):** singletons kept out of the target, which moved a decile's target by up to 10% where one sat |
| **PAIRING** `newey_west`(5.5a) × `volatility_regime`(5.4) | 5.5(a) + 5.4 | The specific LEVEL only — `Δ` is diagonal | **The VRA undoes 82–90% of the Bartlett correction.** Signs reverse across it: without the correction `λ_S = 0.947` (model over-forecasts, VRA marks down); with it `λ_S = 1.153` (under-forecasts, marks up). Long horizon absorbs to **+0.0025** of the median correction. Failure mode 7, measured. | §5.5.3, rows 137–139 | **[M W7-P4]** **Same in kind.** `λ_S²` 1.25 short / 1.07 long at the last scored month-end; the fully specified variant's family 2 lands at 0.954, outside the interval (row 296(a)) — the specific VRA over-marks on a cross-section whose mean of squares a few understated names dominate. Suspect named, not measured further |

## What the pattern is, stated carefully

Four of the entries above are stages whose *value* at `K = 6` is measurably
smaller than the published method assumes, and in each case the reason is the same
parameter:

1. the **PSD repair** fires only where `K/T_eff` approaches 1;
2. the **λ amplitude** scales with `K/T`, so the published 1.5-to-0.95 spread is a
   large-`K` shape;
3. the **parabola** has `K − 3` degrees of freedom to smooth with, so it smooths
   almost nothing at `K = 6` and nothing at all at `K = 3`;
4. the **eigenfactor correction itself** moves the optimizer's forecast by 0.76%
   here, where at `K = 56` the same code corrects a far larger bias.

**A fifth entry is not a stage at all, and it was the one this table's own shape
was hiding.** The `psd_repair` × `eigenfactor` row is a **pairing**: both stages
pass their own checks, and the failure exists only in their composition. A table
organised one-stage-per-row cannot represent that, which is why it was found by
reading a chart rather than by reading this file — and why the pairing now has a
row. It is also the only entry here whose `K = 56` prediction is a *risk* rather
than a larger benefit: the repair fires where `K/T_eff` approaches 1, so the
stage that is a no-op at `K = 6` is expected to be live at `K = 56`, and every
firing corrupts all `K` values of `γ` rather than one.

Two entries are **not** `K`-dependent and are listed so the pattern is not
over-read: the zero-mean convention is a coherence requirement whose size depends
on factor means, and the Bartlett correction's cost depends on whether the data
has serial correlation, not on how many columns it has.

And one entry cuts the other way, which is the entry that matters most: **the
eigenfactor adjustment is worth having at `K = 6`.** A +0.76% revision to the
optimizer-selected portfolio against +0.08% for random portfolios is a 10× ratio,
not a marginal effect, and it is the term the project's identity is about.

**The spectrum column is what stops the summary being glib.** "Earns its keep" and
"does nothing" are the wrong axis on their own: the eigenfactor adjustment earns
its keep *on the bulk* and does almost nothing *for the dominant factors*, and
which of those matters depends entirely on where the portfolio sits. A
minimum-variance portfolio lives in the bulk, which is why the correction bites
there; a portfolio dominated by the first factor would barely notice it. At
`K = 56` — a spiked spectrum with a dominant market factor — that distinction
stops being a footnote.

## `N/T` — the same ratio, this time inside a published formula (added W3-P5)

The table above is about `K/T_eff`, the ratio the covariance pipeline's stages
are sensitive to. **Model B (SPEC.md 4.2) supplies the same ratio from a
completely different direction, and that is why it is recorded here rather than
only in its own report.**

Marchenko-Pastur's noise edge is

```
λ₊ = σ²(1 + √(N/T))²
```

`N/T` is not an analogy for the project's headline ratio — it **is** a
variables-over-observations ratio, appearing inside a published formula rather
than inferred from one. Every other appearance in this project is a diagnostic
the pipeline emits about itself; this one is a term in an equation the model
evaluates.

| Where | Ratio | Range on this project's data | What it governs |
|---|---|---|---|
| SPEC.md 5.2, the Bartlett fallback | `K/T` | fires at **0.43** and above; silent below | Whether the Newey-West correction drives a variance non-positive at all |
| SPEC.md 5, every covariance stage | `K/T_eff` | 0.008 (K=6, 252d) to 0.025 (K=6, 84d) | How much the corrections in the table above have to do |
| SPEC.md 6.4, Shepard | `K/T_eff` | same | `[1 − K/T_eff]⁻²`, the optimizer's understatement |
| **SPEC.md 4.2, Marchenko-Pastur** | **`N/T`** | **0.0516 down to 0.0029** (N=13, T=252→4421) | **`λ₊` directly, and therefore Model B's factor count** |
| SPEC.md 15.2, week 7 | `K/T_eff` | **0.037–0.114 on the scored window at `K_d` = 51–54** (0.64 on the first build) | The headline sweep — measured in W7-P4, read in W8-P1 |

**What Model B's reading says, and it is the same shape as the rest of this
file.** At `N/T = 0.0029` the noise band runs `[0.894, 1.111]` — a collar 0.22
wide around `σ²`, since `√(N/T)` is only 5.4%. It is
so narrow that the `σ²` fit SPEC.md 4.2 specifies **cannot resolve it**: the
0.15 KDE bandwidth quoted from López de Prado ch. 2 is 69% of the entire band's
width, where in ch. 2's own examples (`N/T ≈ 0.1`) it is 12% of it. The fit
degenerates to `σ² = 1` — the assumption SPEC.md 4.2 explicitly says not to make
— on every one of 4,170 dates. See SPEC.md 4.2.4 and `experiments.md` rows
119–122.

That is one of two instances added in W3-P5, and the first from outside
SPEC.md 5. **The other arrived by refuting a registration rather than by
measurement**: `experiments.md` row 100 pre-registered the Bartlett correction's
non-positive-variance failure as a large-`K` problem and it fired at `K = 3`,
`T = 7`, `K/T ≈ 0.43` — so that failure too is indexed on the ratio and not on
`K`, and the registration's own framing was mis-scoped. A pre-registered
hypothesis indexed on the wrong variable, refuted by its own remedy firing, says
the ratio governs even where the person writing the registration did not think to
look. See §5.2.3.

The pattern itself:

> **A published method's machinery is calibrated for a regime, and run faithfully
> in a different one it does nothing — or, here, silently returns the default it
> was specified to avoid.**

**But the σ² degeneracy is the smaller half of what Model B contributes. The
larger half is the first entry in this file where a published technique is
actively HARMFUL rather than merely inert.**

## The five `N`- and `K`-dependent techniques, and the two that cost something

Every entry in the table above is a stage that does *less* than advertised at
this scale. Three of them, restated as one row each so the fourth can be put
beside them:

| Technique | SPEC.md | Why its value depends on `N` or `K` | What it costs here |
|---|---|---|---|
| PSD repair | 5.2 | Fires only where `K/T_eff` approaches 1 | **Nothing.** 5 dates of 4,141, all at `T ≤ 10`; a no-op in production |
| Parabola smoothing inside the eigenfactor stage | 5.3 step 7 | Three parameters fitted on `K` points, so `K − 3` residual d.o.f. | **Nothing.** Close to an interpolation at `K = 6`; exactly one at `K = 3` |
| Newey-West Bartlett correction | 5.2 | Depends on real serial correlation, not on `K` — but the variance it pays is per-parameter | **Little.** On a fixture with no serial correlation to correct, Frobenius error against a known truth rises 1.6–3.2× |
| **Marchenko-Pastur denoising** | **4.2** | **It is a high-dimensional technique. Its value is proportional to how much sampling noise there is in the correlation matrix, i.e. to `N/T`** | **A 57% increase in out-of-sample minimum-variance volatility.** 1.41%/yr against 0.90% for the plain sample correlation |
| **Specific risk — the whole of §5.5** | **5.5** | **Three of its four legs are cross-sectional. Leg (b) regresses on `N` points, leg (c) discriminates between ragged histories, leg (d) needs populated buckets** | **The blend is INVERTED and stage (d) is vacuous on 4 of 5 buckets.** `γ_n < 1` would move weight to a leg fitted on 6 residual d.o.f.; `v = 1/(1+q)` regardless of data at two members. Added W4-P1 |

**The fourth and fifth are different in kind, not degree.** The first three are
corrections that find nothing to correct and cost approximately nothing for
looking. MP denoising at `N/T = 0.0029` is not idle — it acts, and acting is the
problem. The
sample correlation of 13 series from ~4,470 observations is already well
estimated; `λ₊` sits at 1.11 and the band is 0.22 wide with essentially nothing
inside it. Replacing the ten sub-band eigenvalues with their constant average
therefore **averages away real structure rather than noise**, and the structure it
destroys is exactly the small-eigenvalue directions a minimum-variance portfolio
is built from.

**The sign is measured on both sides, which is what makes this a statement about
the regime rather than about the implementation.** `experiments.md` row 127,
registered before it ran, repeats the comparison at `N/T = 0.4`: the ordering
flips, MP-denoising beats the sample correlation by 6.1%, `σ²` identifies at 0.62
instead of pinning, and the survivor count recovers the planted factors exactly.

**`K = 56` prediction, registered: [P] MP-denoising OUTPERFORMS the sample
covariance at the equity panel's `N/T` ≈ 0.11** — about 500 names against the same
history, in regime and between the two measured points. **Falsified if it fails to
outperform there**, in which case the regime explanation is wrong and the
underperformance measured at `N/T = 0.0029` needs another cause. Registered in
W3-P5, before the equity panel exists, with the sign already measured at 0.0029
and at 0.4. See SPEC.md 4.2.7.

**Why this belongs in this file rather than only in Model B's report.** It is a
fourth independent corroboration of the governing ratio, and the most independent
one: it comes from outside SPEC.md 5's pipeline, from a technique with no
connection to Shepard's second-order correction, and it is measured as a *cost*
in the units the project's identity is denominated in rather than as a diagnostic
about a matrix. The headline is that estimation error scales with `K/T`; this is
the corollary reaching the other way — **that a correction for estimation error
has negative value where there is none to correct.**

**`K = 56` prediction: [P] the fit identifies.** At `N = 500` names against a
252-day window, `N/T ≈ 2` and the MP band is enormous; the kernel resolves it
easily and `σ²` will sit strictly inside its bounds. **Falsified if `σ²` pins at
a bound at `K = 56` as well** — that would mean the objective is degenerate for a
reason other than the band-to-kernel ratio diagnosed here, and SPEC.md 4.2.4's
mechanism is wrong.

## The fifth entry, and why it is the most structural (added W4-P1)

The four entries above are **stages and machinery**: one line of a pipeline, one
smoothing step inside a stage, one kernel. SPEC.md 5.5 is a **whole section**, and
three of its four legs are cross-sectional constructions rather than time-series
ones. That makes it the largest single piece of the published methodology this
project has had to run outside the regime it was written for, and the finding is
correspondingly broader:

> **A published method's degeneracies are not always idleness. Two of the four
> shapes this file has now catalogued are corrections that act in the wrong
> direction, and both were found by measuring rather than by reading.**

The two that act:

* **MP denoising** averages away eigenvalues that were never noise, and the cost
  is 57% of out-of-sample minimum-variance volatility.
* **SPEC.md 5.5(c)'s blend** would route weight from a noisy-but-real estimate to
  a barely-identified one, and it would do so **hardest in March 2020** — the
  dates where the difference matters most. The inversion is what pinned `γ_n` at
  1 rather than a saturation argument; see §5.5.2.

And one entry here is not a technique failing at all but two of them **cancelling**:
the `newey_west`(5.5a) × `volatility_regime`(5.4) pairing, where the VRA undoes
82–90% of the Bartlett correction and the sign of `λ_S` reverses across it. That
is the **second stage pairing** in this file, after `psd_repair` × `eigenfactor`,
and it arrives the same way the first did — by reading an output rather than by
reading a spec. A table organised one-row-per-stage cannot represent either, which
is why both are entered as pairings.

**What week 7 inherits from this row is a set of registered reversals rather than
a warning.** Every `[P]` cell in the four SPEC.md 5.5 rows above predicts the leg
gets *better* at `N ≈ 500`, and §5.5.2's blend prediction carries an explicit
falsifier. If they hold, the K/N story closes symmetrically: the same code, the
same published methods, one regime where the corrections are inert or inverted and
one where they earn their keep. If the blend prediction fails, the inversion
explanation is wrong and this file's fifth entry needs another cause.

## What this becomes in W8-P1

Two results reported side by side, on the same code and the same pipeline:

- **estimation error scales with `K/T`** — Shepard's multiplier and the realised
  bias statistic, swept from `K = 6` to `K = 56`;
- **so do the corrections for it** — this table, with the `K = 56` column filled
  in from week 7 rather than predicted.

Every `K = 56` cell above is a **prediction made before the equity module exists**.
They are cheap to make now and impossible to make honestly later.

## The sixth entry: an equity descriptor's units meeting a heterogeneous universe (added W6-P1)

Ruled at the W6-P1 wrap-up (2026-09-04); `experiments.md` row 210, SPEC.md 8.5.2, `reports/optimizer_verification.md`.

| Technique | SPEC.md | Why its value depends on the universe it was written for | What it costs here |
|---|---|---|---|
| **RSTR as the optimizer's fixed alpha** | **8.5.1 ruling 1, 15.4** | **CNE5's momentum descriptor is written for a homogeneous-volatility equity universe, where a cross-sectionally centred alpha of ordinary size means roughly the same risk-adjusted return on every name.** Applied in return units across assets running from 0.26%/month (`govt_2y`) to about 5%/month (equities), the same alpha on the lowest-volatility asset is a single-asset `IR` above 2 before any correlation is used; `Σ⁻¹` on the near-collinear curve block multiplies it by a further 2.2–2.5 | **SPEC.md 8.4's initialisation gives `γ_risk` = 85.7 from a per-period `IR` of 3.86, the long-only book holds 61.5% `govt_2y` on average with 2.2 effective assets, and it attains 31% of `TE_target`.** Not the identity assets: without them the `IR` is 4.12. `α̂` is NOT changed — the concentration is ruling 7's corner solution with its cause located, and W6-P2's per-asset bounds are the grid dimension that measures what it costs |

The pattern the first five entries share — a published constant or form carrying an implicit `N`, `K` or `N/T` — has a second face here: a published *descriptor* carrying an implicit assumption about the cross-section it is standardised over. The week-7 equity panel is the regime RSTR was written for; there the same construction should behave, and that is a prediction W7 can score.


## W8-P1: the `K/T` scaling result, and what it says about this table

`reports/kt_scaling.md` (SPEC.md 15.1.2) is the chart this file was carried
toward. Three things it adds to the table above, each a cross-reference:

1. **The pipeline's second-order estimation error is the `rho` leg plus a
   `sigma` leg, and only the `sigma` leg moves with the half-life.** Rows
   318–326 (`reports/second_order_risk.md`, the `tau` grid): the `rho` leg —
   the eigenfactor stage's — is 0.61–0.75% of variance at all nine
   factor-volatility half-lives (within 2 s.e. of one value), the `sigma` leg
   runs 5.1% → 0.3%, and Shepard's whole-covariance form over-predicts the
   estimator by 4.1× at `tau = 21` and 0.9× at 504. So the eigenfactor stage's
   *target* is `K`-dependent but not `tau`-dependent, and the half-life sweep
   is a sweep of the leg the stage does not touch.
2. **`eigenfactor` at `K_d = 51–54`, re-measured on row 164's subset books:
   removes 44% of the family-4 excess** (1.159 → 1.089 monthly), against 0% at
   `K = 6` — the largest share any stage has removed on any panel here — and
   restoring the residual correlations (C1) *adds* to the excess, because a
   residual correlation on hundreds of names from an EWMA at `T_eff = 242` is
   rank-deficient. The diagonal `Delta` is in regime at large `N` and out of
   regime at `N = 13`; the eigenfactor stage is the reverse. Same code, same
   two panels, opposite verdicts, and the governing ratio is `K/T`.
3. **`psd_repair` at `K_d = 51–54` fires through the sample at `tau ≤ 42`**
   (38 and 16 of 210 month-end builds, the last on the final build, at realised
   `K_d/T_eff` 0.89 and 0.45), so the equity model has no scored window there
   at all — the "where it earns its place" cell of the `psd_repair` row above
   is not a nuisance but a boundary: below `tau = 63` the factor model does not
   exist on this panel. At the shipped 84 it fires once, on the first build, as
   the row records.

The finding this file exists to accumulate now has its number. The corrections
scale with `K`; the estimation error they correct scales with `K/T`; and the
half-life, which sets `T`, moves the error the eigenfactor stage does not
correct (the `sigma` leg and the responsiveness channel) and leaves the one it
does (the `rho` leg) where it was.
