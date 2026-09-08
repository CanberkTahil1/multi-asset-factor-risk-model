# Multi-asset factor risk model with a cost-aware implementation layer

Every quantitative portfolio has a paper track record and a real one, and the two
never match. This project measures that gap on its own data and splits it into
exactly two terms: **estimation error in the risk model**, measured by the bias
statistic `B = σ_realised / σ_forecast`, and **implementation cost**.

It builds two factor risk models — a 6-factor macro model over a 14-instrument
multi-asset sleeve, and a 54-factor cross-sectional equity model over the S&P 500
— puts both through **one unchanged covariance pipeline** (EWMA with separated
half-lives → Newey-West → PSD repair → eigenfactor adjustment → volatility regime
adjustment, plus Bayesian-shrunk specific risk), wraps the macro model in a convex
optimizer with a time-varying empirically-calibrated cost model inside the
objective, and reports the decomposition in-sample over 187 months and once, out
of sample, over a window pinned before any model was fitted.

> **This project does not forecast returns.** Expected returns enter only as a
> fixed, published, never-tuned optimizer input, so that the optimizer has
> something to trade against. Nothing here was selected on backtest performance,
> and the trial ledger in [`experiments.md`](experiments.md) exists to make that
> checkable rather than asserted.

## How this was built

A specification (`SPEC.md`) and a constitution (`CLAUDE.md`) were written before
any code, and both are committed here verbatim — including the invariants the work
had to obey and the failure modes it was told to expect. The build ran as **one
session per task**, each named by a task ID; every session's **rulings were issued
before the run they governed**, so a decision could not be made after seeing which
answer it produced. Every configuration evaluated was logged with a hypothesis, a
falsifier and a result: [`experiments.md`](experiments.md) records **332
configurations**, of which **71** fall in a category that could flatter a reported
Sharpe, and the deflated Sharpe is computed at **`N` = 70** — the trials that
existed *before* the shipped configuration was frozen.

**The refutations were kept rather than tidied**, which is the only part of this
that is hard to fake:

- **§15.7's registered target of 1.4–1.7 for the equity model's bias landed at
  1.10** — below the macro model's 1.33, the opposite end of the prediction — and
  the component split explains why rather than the prediction being restated.
- **H3's first half is refuted.** The eigenfactor adjustment was predicted to
  close most of the optimizer-selected bias gap; it moves it by **−0.0001**.
- **Row 83 was refuted five sessions after it was written**, when the residual
  panel it predicted finally existed: the leading residual PC is the third curve
  mode, not what the row said.

To which the honest addendum is that a fourth refutation is aimed at the search
itself — **PBO is 0.6876**, so ranking these 28 cells by in-sample Sharpe is worse
than useless, and the shipped cell was fixed by ruling before the grid ran
precisely so that nothing depended on that ranking.

**This repository is a snapshot, not the build.** It is a single commit taken from
the finished work. The development history — 74 commits, one or a few per session,
each named by its task ID — is kept in a private repository and is **available on
request**; it is the provenance record, and it is not public because it contains
material that was never intended to be. The fuller record of *what happened when*
is here anyway: `experiments.md`'s rows are dated, carry their task IDs, and
include the sessions that produced nothing, because a session that consumed a look
at the data counts whether or not it worked.

---

## Summary

**The identity.** `SR_paper − SR_real = (μ_g/σ_f)·(1 − 1/B) + TC/(B·σ_f)`, exact
and asserted in code. On the shipped configuration over **187 months** the two
terms are **risk-model +0.0543** and **cost +0.0400**, at `B` = 1.061 and a net
Sharpe of 0.850 ± 0.264. Out of sample over **18 months** they are **+0.0746** and
**+0.0572** at `B` = 1.0681. The risk model leads the cost model at the reference
cell under every spread reading the issuer disclosures permit — and it leads by
more, not less, the better the cost treatment gets.

**Neither term is resolved against the other at this sample size.** `B` = 1.061
lies *inside* its exact χ² interval **[0.898, 1.101]**, and the cost term is a
**lower bound** — the shipped cell prices the spread at the disclosure interval's
floor and commissions are zero, an absence rather than a value — so both known
biases push the split the same way the panel reports it, and **"the risk-model
term leads" is an ordering this panel produced under biases that favour it, not a
finding.** What does *not* depend on either statistic being distinguishable from 1
are the two contrasts below — the macro panel flat at 1.64 against the equity
panel's 1.0256 → 1.2143, and family 4's 0.335 deterioration against the
constrained book's 0.007 — because each is a *difference* of two bias statistics,
in which a common offset cancels.

**What the risk-model term is made of, and why it is not what the literature
predicts.** At `K` = 6 and `N` = 13 the bias is **specification error — the false
diagonal in `Σ = X F X' + Δ` — not estimation error**, established three ways: a
pre-registered control that restores the residual correlations off-diagonal only
closes **107% of the gap** to the naive comparand while moving random portfolios
0.002; a 24-fold sweep of `K/T_eff` moves the optimizer-selected bias by
**0.0039**; and the component assertion fires on the specific leg (1.23–1.32,
outside its interval) in all twenty factor-model cells.

**The two-panel contrast, which is the headline.** The same pipeline, unchanged
(`git diff` on `src/mafrm/risk/` is empty), was pointed at a 54-factor equity
panel. **Shepard's `[1 − K/T]⁻²` scaling is not reproduced on this data.** The
macro model's bias is **flat at 1.64 across nine half-lives**; the equity model's
**rises 1.0256 → 1.2143**, about **38× more** than the estimator's own simulated
second-order theory allows; and at the two shortest half-lives the equity model has
**no answer at all**, because the PSD repair fires through the sample. So the chart
is a **contrast, not a curve**: a model whose error is specification error does not
respond to sample size, and a model whose error is estimation error responds
strongly. The identity's own component split of `B` told the two apart **in
advance**, registered before either run. The driver of the equity panel's excess
movement is **unidentified**, the separating test is registered and unrun, and that
is stated rather than filled with a candidate.

**The holdout, evaluated once.** The window was pinned 2025-01-01 before any
covariance matrix reached an optimizer and read once, on a frozen configuration,
after a control confirmed the harness reproduces the published in-sample answer.
**The Sharpe resolves nothing and was registered as unable to before the window
opened** — at 18 months the standard error is 0.86. The one thing the window could
resolve is a contrast: over the same window and the same forecasts, the
*unconstrained* minimum-variance bias deteriorated **1.332 → 1.667** while the
*constrained* shipped book's moved **1.061 → 1.068**. The optimistic interval and
the unexplained window-length effect apply to both sides and cancel in the
comparison, which is why it is reportable when neither number is: **what protected
the shipped book was the constraint set, not the covariance estimator.** The book
also underperformed equal weight by 2.16%, and that is reported too.

**What was searched, and what that is worth.** The ledger counts **332 evaluated
configurations — 261 `data-diagnostic`, 12 `model-config`, 59 `strategy-config`**,
of which 71 are in a counted category. The **deflated Sharpe is computed at
`N` = 70**, the counted trials as they stood when the configuration was frozen, and
it is **0.7408** — short of any conventional bar. Probability of backtest overfitting across the grid is **0.6876**
with a degradation slope of −0.919: the grid's in-sample ranking carries no
out-of-sample information, so selecting on backtest Sharpe would have been worse
than choosing at random — **which is why the shipped cell was fixed by ruling
before the grid ran and never selected on.**

**Beyond the identity**, the project documents [eight published constructions
measured outside the regime they were calibrated
for](#published-methods-measured-outside-their-regime), and [three undocumented
traps in the Fed's own GSW curve file](#three-undocumented-traps-in-the-feds-own-curve-file),
one of which silently deletes 27% of the sample.

## Contents

- [How this was built](#how-this-was-built)
- [The question, stated as an identity](#the-question-stated-as-an-identity)
- [Results](#results) — [in-sample decomposition](#1-the-in-sample-decomposition-187-months) · [the two-panel K/T contrast](#2-the-two-panel-kt-contrast-and-a-closed-form-that-does-not-reproduce) · [the holdout](#3-the-holdout-run-once) · [the cost side](#4-the-cost-side) · [the pre-registered hypotheses](#5-the-pre-registered-hypotheses-registered-2026-08-25)
- [Published methods measured outside their regime](#published-methods-measured-outside-their-regime)
- [Three undocumented traps in the Fed's own curve file](#three-undocumented-traps-in-the-feds-own-curve-file)
- [Data](#data) · [Method](#method) · [Reproducibility](#reproducibility)
- **[Limitations](#limitations)** — where an experienced reader should go first, and written to reward that
- [Deliberately out of scope](#deliberately-out-of-scope) · [Positioning](#positioning) · [Repository, licence, citation](#repository-licence-citation)

The chronological record is [`experiments.md`](experiments.md) — every
configuration with its category, hypothesis, falsifier and date — and the commit
log, one commit per work session named by task. Methodology of record:
[`SPEC.md`](SPEC.md). Working constraints: [`CLAUDE.md`](CLAUDE.md).

---

## The question, stated as an identity

Let a strategy have gross annualised return `μ_g`, forecast active volatility
`σ_f`, realised volatility `σ_r`, and annual cost drag `TC`. With
`B = σ_r / σ_f`:

```
SR_paper − SR_real  =  (μ_g/σ_f)·(1 − 1/B)   +   TC/(B·σ_f)
                    =  ── risk-model term ──     ── cost term ──
```

Exact, not an approximation, and asserted in code on every cost-free cell
(`gross SR × B = SR_paper` to six decimals). Two properties are not obvious:

- **The cost term is divided by `B`.** A model that under-forecasts risk
  mechanically *understates* how much costs hurt Sharpe. The two error sources are
  not independent, and adding them without this term is wrong.
- **The risk-model term is proportional to `SR_paper`.** High-Sharpe paper
  backtests degrade harder — the quantitative form of "if it looks too good in the
  backtest, it is."

Every row of [`reports/experiment_grid.md`](reports/experiment_grid.md) is a
configuration and its two columns are those terms.

---

## Results

### 1. The in-sample decomposition (187 months)

Shipped configuration: EWMA + Newey-West + eigenfactor `a = 1.0`; a time-varying
spread (issuer level + EDGE widening) with square-root impact, inside the
objective and charged on realisation; patient impact regime; tracking-error target
at 1× the equal-weight book's realised volatility; long-only, fully invested. 187
month-end rebalances, 2009-05-29 to 2024-11-29, at $8.12M — the cost model's own
lower calibration anchor (2% of the thinnest proxy's median ADV).

| | Value |
|---|---|
| `B` on the shipped book | **1.061**, exact χ² interval at `T` = 188: **[0.898, 1.101]** |
| Risk-model term | **+0.0543** |
| Cost term | **+0.0400** |
| `SR_paper` / `SR_real` | 0.944 / 0.849 |
| Net Sharpe (Lo scale) | 0.850 ± 0.264 |
| Cost drag / turnover | 28.2 bp/yr / 5.62× |

The risk-model term leads the cost term at the reference cell, and under every
spread reading the issuer disclosures permit — the high end of the interval moves
the cost term to 0.0424 against 0.0543 (a band registered to flip it, and
refuted). The ordering is not general: it fails in 16 of 21 charged cells, but 14
of those 16 are the two cost treatments the specification itself deprecates
(trading as if trading were free, and a flat 2.5bp half-spread that over-prices
the real one by 2–10×). **The better the cost treatment, the more of the residual
gap belongs to the risk model.**

**The two terms are not resolved against each other at this sample size, and the
ordering is not a finding.** `B` = 1.061 sits inside the exact χ² interval
[0.898, 1.101] at `T` = 188, so the risk-model term is not distinguishable from
the value it would take at `B` = 1; and the cost term is a lower bound, priced at
the spread interval's floor with commissions at zero — so both known biases point
in the direction of the reported ordering, and what is recorded here is **an
ordering this panel produced under biases that favour it.** The claims that do not
depend on it are the contrasts: the macro panel's flat 1.64 against the equity
panel's 1.0256 → 1.2143 in [result 2](#2-the-two-panel-kt-contrast-and-a-closed-form-that-does-not-reproduce),
and family 4's 0.335 deterioration against the constrained book's 0.007 in
[result 3](#3-the-holdout-run-once), are each a *difference* of two bias
statistics computed on the same forecasts, and neither requires either statistic
to be distinguishable from 1.

**What the risk-model term is made of.** At `K = 6`, `N = 13` the bias is
**specification error — the false diagonal in `Σ = X F X' + Δ` — not estimation
error.** Three independent routes:

- **Control C1, pre-registered with two falsifiers, both survived.** Replacing
  `diag(δ²)` with `D_δ R_u D_δ` — off-diagonal only, all four specific-risk legs
  held exactly — takes the unconstrained minimum-variance family's `B` from
  **1.3321 to 1.0481**: 107% of the gap to the naive comparand, 86% of the excess
  over one. Random portfolios move **0.0021** against an interval half-width of
  0.0223. The effect is *direction-selective*, which is what the argument turns on
  — a change in the forecast's level would have moved random portfolios too.
- **A 24-fold sweep of the estimation-error channel moves it 0.29%.** Across nine
  volatility half-lives `K/T_eff` rises 24× (0.0041 → 0.0990); random portfolios
  rise 1.0003 → 1.0301 exactly as Shepard predicts in direction, while the
  optimizer-selected family moves **0.0039**. A bias invariant to `K/T` is not a
  `K/T` phenomenon.
- **The component assertion fires in all twenty factor-model cells, on the
  specific component only:** factor `B` 0.94–0.99 (inside its interval), specific
  `B` **1.23–1.32** (outside), total 1.01–1.06.

The diagonal is **not repaired**. Repairing it would mean not implementing the
specified model, and the specification's failure is the deliverable.

Figures: [`bias_statistics.png`](reports/bias_statistics.png) (four portfolio
families, rolling `B` with bands),
[`halflife_sensitivity.png`](reports/halflife_sensitivity.png),
[`second_order_risk.png`](reports/second_order_risk.png),
[`eigenfactor_bias.png`](reports/eigenfactor_bias.png),
[`vra_multiplier.png`](reports/vra_multiplier.png),
[`optimizer_weights.png`](reports/optimizer_weights.png),
[`bands.png`](reports/bands.png),
[`experiment_grid.png`](reports/experiment_grid.png). Tables:
[`bias_statistics.md`](reports/bias_statistics.md),
[`validation_battery.md`](reports/validation_battery.md),
[`experiment_grid.md`](reports/experiment_grid.md),
[`diagnostics.md`](reports/diagnostics.md).

### 2. The two-panel `K/T` contrast, and a closed form that does not reproduce

The equity module exists to put the same machinery on a high-`K` problem: 54
factors (country + 49 Fama-French industries present point-in-time + six CNE5
style descriptors) over 305–458 names per date, √cap-weighted WLS under a
cap-weighted industry sum-to-zero constraint. It goes through `src/mafrm/risk/`
**unchanged** — `git diff` on that directory for the session is empty, every stage
PSD on all 420 month-end builds.

**The project did not reproduce Shepard's `[1 − K/T]⁻²` scaling on its own data,
and that is the finding.**

| | Macro, `K = 6` | Equity, `K_d = 51–54` |
|---|---|---|
| `K/T_eff` swept | 0.0055 → 0.0998 | 0.043 → 0.290 |
| Family-4 `B` across nine half-lives | **flat: 1.6417 → 1.6471** | **1.0256 → 1.2143** |
| Estimator's own simulated second-order theory | — | 1.0374 → 1.0425 (range 0.005) |
| Measured range ÷ simulated range | — | **≈ 38×** |
| Two shortest half-lives | answered | **no answer at all** |

1. **The macro cluster does not move** — nine points flat at 1.64 while the curve
   moves; the registered "does not track" verdict holds 9 of 9.
2. **The equity cluster moves far more than any estimation-error account
   reaches.** The measured range is 0.189; a Monte Carlo on the pipeline's *own*
   estimators — the exact theory for a two-window estimator, not a one-window
   approximation to it — gives 0.005, with standard errors of 0.0004–0.0012 that
   cannot absorb any of it. The registered prediction that the equity points track
   that simulated curve is **refuted at exactly the two half-lives at which the
   closed form also missed**, which is what says the misses belong to the
   measurement rather than to which window a formula is evaluated at. **The driver
   is unidentified.** The simulation has no fat tails, no volatility clustering, no
   specific-risk error and no specification error, and the responsiveness channel
   is confounded with the sweep at every point. No candidate is promoted; the
   separating test is registered and unrun.
3. **At the two shortest half-lives the equity model does not exist.** The PSD
   repair fires on 38 and 16 of 210 month-end builds, the last at `K_d/T_eff` of
   0.89 and 0.45, so no month-end survives the scored-window rule. That is the
   factor-model end of "effectively singular", reached by measurement.

**What the chart shows is a contrast, not a curve.** A model whose error is
specification error does not respond to sample size; a model whose error is
estimation error responds strongly. **The identity's own component split of `B`
told the two apart in advance** — registered before either run: the macro model's
specific component sits outside its interval (1.23–1.32) and the equity model's
inside it (0.97), with the equity excess carried entirely by the factor component
(1.18, outside). The same instrument confirms it in opposite directions: control
C1 closes 86% of the macro excess and *widens* the equity book's, 1.159 → 1.268,
because a diagonal `Δ` is in regime at `N` = 447 and restoring a rank-deficient
residual correlation puts estimation error into a `Δ` that had none. The
eigenfactor stage behaves accordingly: it takes the equity book's `B` from
**1.0874 to 1.0342** against **−0.0001** at `K` = 6. Where a share of the gap is
quoted at all, it is the ledger's with its basis named in the same sentence: 55%
and 76% at the two published scalings against W7-P4's pre-singleton-fix baseline,
and 43.7% of the excess over one on row 164's 252-name always-present subset.

Figures: [`kt_scaling.png`](reports/kt_scaling.png),
[`equity_bias_statistics.png`](reports/equity_bias_statistics.png),
[`equity_lambda_curve.png`](reports/equity_lambda_curve.png),
[`second_order_tau_grid.png`](reports/second_order_tau_grid.png). Tables:
[`kt_scaling.md`](reports/kt_scaling.md),
[`kt_book_sweep.md`](reports/kt_book_sweep.md),
[`equity_validation_battery.md`](reports/equity_validation_battery.md),
[`stage_k_dependence.md`](reports/stage_k_dependence.md).

### 3. The holdout, run once

`HOLDOUT_START` was pinned at **2025-01-01** on 2026-08-30, before any covariance
matrix reached an optimizer, and made unrevisable. It was evaluated **once**, on
the frozen reference cell, after a control confirmed the same code path reproduces
the published in-sample answer (`B` 1.0611 against 1.061; net Sharpe 0.8495
against 0.850). Code, config, risk target and book size frozen; every *estimate*
walks forward. Window **2025-01-01 → 2026-07-31, 18 rebalances**; the right edge
is the minimum of the book's per-series last observations.

| | Holdout | In-sample |
|---|---|---|
| Net Sharpe (Lo scale) | 1.0374 ± **0.8586** | 0.850 ± 0.264 |
| `B`, shipped book | **1.0681**, χ² [0.667, 1.333] at `T` = 18 | 1.061, χ² [0.898, 1.101] at `T` = 188 |
| Risk-model term | **+0.0746** | +0.0543 |
| Cost term | **+0.0572** | +0.0400 |
| Cost drag / turnover | 29.9 bp/yr / 4.58× | 28.2 bp/yr / 5.62× |
| Deflated Sharpe | **0.7408** at `N` = 70 | 0.99 at `N` = 38 |

**The Sharpe of 1.0374 resolves nothing, and it was registered as unable to before
the window opened**: at 18 months the Lo standard error is 0.86 — about the size
of the in-sample Sharpe itself — and the exact interval for `B` is [0.667, 1.333]
against [0.898, 1.101] in-sample, so a holdout Sharpe of 0.4 and one of 1.3 are
both inside one standard error of 0.850. That arithmetic was written down before
the boundary moved, which is the only thing that makes reporting the number honest
rather than flattering. Cost drag (registered 20–40 bp/yr), turnover and the
identity's ordering all held.

**The result this window could resolve is a contrast between two statistics, not
either one alone.** On 391 daily scored sessions the exact interval is
[0.930, 1.070] — twenty times tighter than the monthly one:

| Family | Holdout, χ² [0.930, 1.070] at `T` = 391 | In-sample, χ² [0.978, 1.022] at `T` = 3,874 |
|---|---|---|
| 1, individual assets | 0.9766 | 1.0245 |
| 2, random portfolios | 1.0871 | 1.0101 |
| 3, eigenfactors | 1.0368 | 1.0415 |
| 4, **optimizer-selected** | **1.6673** | 1.3321 |

Family 4 is the *unconstrained, daily-rebuilt* minimum-variance portfolio; the
identity's `B` is the *constrained, monthly* book carrying the tracking-error
bound, the long-only simplex and the ADV hinge. **Over the same window, on the
same forecasts, the unconstrained statistic deteriorated by 0.335 and the shipped
book's moved 0.007.** H1 survives out of sample; H2's registered 1.2–1.5 band is
exceeded at 1.67.

**Why the comparison is reportable when neither number is.** The daily interval is
optimistic — the forecast behind those 391 observations is a slowly-moving rolling
estimate and the weights are rebuilt daily from it, so consecutive standardized
returns are not as independent as the interval assumes. And the family-4 level is
confounded with the unidentified window-length effect of
[result 2](#2-the-two-panel-kt-contrast-and-a-closed-form-that-does-not-reproduce).
**Both apply to both sides of the comparison and cancel in it**: the optimistic
interval, the window length and the unexplained effect are common to the
constrained and unconstrained statistics, and what differs between them is only
the constraint set. So the contrast survives what neither number survives alone —
**what protected the shipped book was the constraint set, not the covariance
estimator.** That is the first out-of-sample evidence for the in-sample finding
that the bias is a property of how concentrated the book is allowed to get: the
tracking-error band took `B` from 1.139 at 0.5× to 1.015 at 2×, and a 2/13
position bound took it to 0.991. Stated as consistency, not as proof — one window,
one configuration, and the mechanism was not manipulated here.

**The book underperformed equal weight by 2.16% cumulative**, and the
Brinson-Fachler decomposition puts it in allocation rather than selection. That is
what a concentrated low-volatility book does in a window where the broad book
runs. It is the comparison a reader will ask for and it is not a flattering one.

**The equity control is uninformative out of sample at this length.** Of five
legs, three are statistically indistinguishable from their in-sample values and
two are nominally significant (`p` = 0.018 and 0.042) but **neither survives a
Bonferroni correction for five comparisons** at level 0.01. Every holdout leg sits
inside its own null interval, so nothing here refutes the model. **The
diagonal-in-regime finding is an in-sample result**, and this window cannot test
it.

Per year: 2025 +7.58% net over 11 months at 17.8 bp of cost; 2026 +0.32% over 7
months at 27.0 bp, with 11.72% realised volatility against a 7.80% forecast — one
episode, not a pattern. The run was clean: 18 of 18 rebalances at the first rung,
no relaxation, no fallback. Full report: [`holdout.md`](reports/holdout.md). The
window was opened three times and read once; the disclosure is in
[Limitations](#limitations).

### 4. The cost side

Cost is a half-spread plus square-root impact at a general convex exponent, with
the total pinned against the per-unit law. Both published impact regimes are
always built and always reported — patient `Y = 0.58` and urgent `Y = 1.40`,
backed out of AQR's and Virtu's published anchors and reproducing the
specification's table at 0.601 / 0.572 / 1.414. There are no interior points: two
calibrated regimes, not the ends of a continuum.

- **Spreads are emphatically not flat.** The sleeve's 63-day EDGE median widens by
  **+105bp** through the GFC, **+129bp** through COVID and **+38bp** through the
  2022 repricing, against the preceding twelve months. In calm periods the window
  cannot resolve the sleeve's spread from zero at all; in crises it resolves
  40–130bp. ([`spread_vs_vol.png`](reports/spread_vs_vol.png))
- **H5 holds beyond its band.** In the ten worst rebalances by trailing
  volatility, realised spread cost is **4.37×** the flat assumption on the same
  trades, against 0.94× in the other 177. The registered band was 2–4×.
- **H6 holds by 0.006 Sharpe units, and the robust half is the interesting one.**
  Costs inside the objective beat gross-then-net at **all seven covariance
  variants by 0.05–0.07**, while the spread across covariance estimators under the
  best cost treatment is 0.0675 — and that range is the dense-versus-factor gap,
  not anything inside the pipeline. Both numbers are a quarter of one Lo standard
  error, so the headline ordering is recorded as what this panel produced, not as a
  statistical claim.
- **Turnover orders as predicted.** Gross-then-net trades 7.0–7.8×/yr and pays
  46–51 bp; a flat 2.5bp spread over-prices the truth and trades least (5.1–5.7×,
  32–34 bp); the time-varying model trades 5.3–5.9× and pays 27–30 bp, the least of
  the three, with the highest net Sharpe at every variant.
- **Capacity is a re-optimisation, not a rescaling.** Re-solving at fifty AUMs with
  the ADV cap hard shows the closed-form rescaling **overstates cost by 37%
  (patient) and 49% (urgent)** at the top of the grid.
  ([`capacity_reoptimised.png`](reports/capacity_reoptimised.png),
  [`capacity.png`](reports/capacity.png))

The engine reconciles against `bt` to **two or three units in the last place** on
both a synthetic and the real thirteen-asset panel; the one row registered to
breach its bound does so by a named accounting convention worth **+0.004 bp/yr**,
attributed in full by replaying `bt`'s own convention on this engine's arithmetic.
([`engine_reconciliation.md`](reports/engine_reconciliation.md))

### 5. The pre-registered hypotheses (registered 2026-08-25)

Written before any model code existed, reported against honestly.

| # | Hypothesis | Outcome |
|---|---|---|
| H1 | Bias on **random** portfolios ≈ 1.0 for every variant, naive sample estimator included | **HOLDS.** 0.957–1.011 across nine variants and two horizons; interval [0.978, 1.022]. Survives out of sample |
| H2 | Bias on **optimizer-selected** portfolios is 1.2–1.5 | **HOLDS** mid-range: 1.332 / 1.314 against a random control of 1.010 / 0.999 on the same matrix. **Exceeded out of sample at 1.667** |
| H3 | The eigenfactor adjustment closes most of the H2 gap and leaves H1 untouched | **FIRST HALF REFUTED**, second holds. −0.0001 and +0.0000 at the two published scalings; family 2 within 0.0005. At `K_d` = 54 the same stage removes ~60% of the excess |
| H4 | Shepard predicts 5.1% understatement for the factor model against 13.6% for a raw sample covariance — 2.6× from factor structure alone | **Arithmetic holds** (2.24× at `N` = 13 in variance). **Empirically split:** on the estimation-error term it holds by *more* — 2.04% measured against 11.7%, 5.7× — but as a statement about family-4 bias it is refuted, 1.33 against the naive covariance's 1.07, because the diagonal's specification error costs thirty times what factor structure saves |
| H5 | Cost drag in the worst volatility windows is 2–4× the flat assumption | **Leg 1 holds in direction, beyond the band** (4.37×); **leg 2 holds in sign**, uncertainty not quantified at ten stressed months |
| H6 | Costs inside the objective beat gross-then-net by more than the gap between any two covariance estimators | **HOLDS by the letter**, by 0.006 Sharpe units. See above for the robust half |

---

## Published methods measured outside their regime

Not in the specification. This is what the project turned out to be *about* beyond
the identity: **eight published constructions, each calibrated to a regime, each
measured outside it, each with its number.** None of this criticises a source.
Every one is a method applied to a panel whose dimensions differ from the one it
was written for — which is what a replication finds out and a citation does not.
Consolidated table: [`stage_k_dependence.md`](reports/stage_k_dependence.md).

| # | Method | The regime it assumes | Measured outside it, here |
|---|---|---|---|
| 1 | **Marchenko-Pastur denoising** | High-dimensional: its value is proportional to the sampling noise in the correlation matrix | At `N/T` = 0.0029 there is none to remove, and flattening eigenvalues that were never noise destroys the small-eigenvalue directions a min-var book is built from. Out-of-sample min-var volatility **0.896%/yr (sample), 0.931% (Ledoit-Wolf), 1.409% (MP) — 57% worse.** A pre-registered control at `N/T` = 0.4 flips the sign (−6.1%). The `σ²` fit also **degenerates to `σ² = 1` — the assumption the method says not to make — on all 4,170 dates**, the noise band being 0.22 wide against a published 0.15 kernel. Model B was not changed |
| 2 | **The eigenfactor stage's parabola** | Enough spectrum to smooth: three parameters fitted on `K` points | `K − 3` residual d.o.f.: three at `K` = 6, **zero** at the `K` = 3 fixture, where it interpolates exactly and removes no noise; 51 at `K_d` = 54, where it visibly smooths. Amplitude **0.0099 at `K` = 6 against a published large-`K` ≈0.55**, and **0.1136 at `K_d` = 51–54**. The published amplitude gate was withdrawn rather than met — holding a correct implementation to a large-`K` shape would have diagnosed it as broken |
| 3 | **The PSD repair** | A rare rescue near singularity | At `K` = 6 it fires on **5 dates of 4,141 builds, all at `T ≤ 10`**, never after: a no-op in production. At `K_d` = 51–54, once per horizon, never reaching a scored forecast. At the two shortest equity half-lives it fires **through the sample** and the model has no answer at all. It is not a safety net; it is an indicator of `K/T_eff` approaching one |
| 4 | **The Newey-West Bartlett correction** | Real serial correlation worth the variance of estimating it | On a fixture serially uncorrelated *by construction* it raises Frobenius error against the known truth by **1.6× to 3.2×** — lag terms that are zero in expectation, paid for. On real data it works, and the sign is panel-dependent: median factor volatility **−3.5%** on the macro panel, net *positive* on the equity factor set. Direction matters: a correction that lowers the forecast biases `B` upward |
| 5 | **The specific-risk blend and shrinkage** | A cross-section with enough names to identify a structural leg and populate a bucket | *The blend is inverted*: its observation term saturates, its fatness term does not (`max Z` = 7.61), and routing weight off the time-series leg moves it onto a structural leg fitted on **6 residual d.o.f.** — hardest in March–April 2020, where that leg is fitted on the same stressed data. *The shrinkage shrinks an extreme value least, by design*, so a name alone in its industry (residual ≈ 0 by construction) is not rescued: 13 such names sat at the **0.7th cross-sectional percentile** carrying 7.2% of the optimizer's absolute weight, until routed to the specification's own structural fallback (48th percentile, weight halved). At a two-member bucket the stage is **data-independent**: the coefficient is `1/(1+q)` whatever the values |
| 6 | **CNE5's RSTR as a fixed alpha** | A homogeneous-volatility equity cross-section, where a centred alpha of ordinary size means the same risk-adjusted return on every name | In return units across assets from 0.26%/month to ~5%/month, the same alpha on the lowest-volatility asset is a single-asset information ratio above 2 before any correlation is used, and `Σ⁻¹` on the near-collinear curve block multiplies it by 2.2–2.5. The specification's initialisation returns **γ_risk = 85.7**; the book holds **61.5% two-year zero** with **2.2 effective assets** and attains **31%** of its risk target. Its own second step lands at 99.7% and γ_risk 7–8 — the initialisation was **ten** times too high, not three, because it prices leverage a long-only simplex does not have |
| 7 | **EDGE on daily ETF bars** | A frequency and a spread size where one window can resolve a level | Unbiased with no resolution floor (median near zero at a true 1bp, 49.3bp recovered at 50bp), but **per-window noise is ~23bp whatever the truth**, so nothing below ~20bp is resolvable from one 63-day window and no scale factor rescues it. LQD returns the exact zero-spread signature; SPY, IVV, VOO and IWM invert to ~25–30bp against quoted spreads near 0.4bp — **about 60×**. A pre-registered control was **refuted**, eliminating the leading suspect. Consequence: a permanent design decision, not an open item — the *level* comes from issuer disclosures and EDGE supplies only the time variation, the half that is sound |
| 8 | **Shepard's `[1 − K/T]⁻²`** | A covariance estimated on **one** window | This pipeline estimates correlations on 504 days and volatilities on 84. A Monte Carlo on its own estimators splits the bias into a `ρ` leg (0.65% of variance) and a `σ` leg (1.46%), adding to within 3–5% of the joint 2.04%, so the two compose. But **the whole-covariance form at the volatility window is 5.14%, two and a half times the split estimator's actual bias**, over-predicting by 4.1× at the short end of the grid falling to 0.94× at the long. At the *correlation* window it agrees with the estimator's own simulated theory to within 0.008 — resolving a conflict in this project's own rulings in the direction nobody expected. Plus the two-panel non-reproduction of [result 2](#2-the-two-panel-kt-contrast-and-a-closed-form-that-does-not-reproduce) |

---

## Three undocumented traps in the Fed's own curve file

`feds200628.csv` — the Gürkaynak-Sack-Wright fitted nominal zero curve, the file
underneath a great deal of published term-structure research — carries three traps
that are documented neither in the file nor in its accompanying paper. All three
were found by inspection, **all three would have passed silently**, and all three
point in the direction that flatters a bond sleeve. They are reported here because
they are useful to anyone reading that file and are independent of this project's
thesis.

**1. `TAU2 = -999.99` silently deletes the Volcker era.** Before 1980-01-02 the
Fed fitted Nelson-Siegel, not Svensson: `BETA3` is 0 and `TAU2` carries the
sentinel `-999.99`. Evaluating the fourth Svensson term at a negative tau gives
`exp(-n / -999.99)`, which overflows to `inf`, and `0 * inf` is `nan`. Left
unhandled this deletes **4,620 rows — 27% of the sample** — and the block lost is
precisely the highest-rate-volatility regime in the record, the single most
informative window for a term-structure risk model. There is no error, no warning
and no gap in the index: the series simply starts in 1980. Handled by dropping the
fourth term wherever `TAU2` is missing or non-positive.

**2. Blank weekday rows inflate carry and roll by 3.5%.** Both curve files carry a
row for every weekday, market holidays included, on which every field is `NA`.
Retained, they yield 261 observations a year against a `Δ` of 1/252, so a
constant-maturity series accrues **3.5% more carry and roll than time actually
passed** — roughly 18bp a year at a 5% yield, again in the flattering direction.
Dropped at parse time, which leaves ~250.5 observations a year and makes
`Δ = 1/252` honest.

**3. One holiday the parameter filter does not catch.** On 2008-03-21, Good
Friday, the Fed wrote a degenerate fit (`BETA0 = 8.3e-14`) but published **no yield
at any tenor**. Trap 2's filter drops rows whose *parameters* are blank; this row
has parameters, so it survives, and anything pricing across it carries a four-day
weekend as a single `Δ` step. One row in 6,008 — and it moved this project's
acceptance gap by **0.08%/yr, more than half the tolerance, decided by a single
day**. Rows with no published yield at any tenor are now dropped alongside the
blank-parameter rows.

Full write-up with the series inventory and the duration identity that validates
the construction: [`gsw_validation.md`](reports/gsw_validation.md).

---

## Data

Everything is free and public. **Network access lives in exactly one module**
(`src/mafrm/data/loaders.py`); every other module reads a content-addressed cache.
`make data` refreshes and updates `data/manifest.json`; `make verify` re-hashes
every artefact and fails on drift. **No third-party raw data is committed** — the
manifest is, and that asymmetry is the point of the design: it makes someone
else's rebuild checkable against this one without redistributing anything.

| Source | What | History | Licence / redistribution |
|---|---|---|---|
| Federal Reserve `feds200628` | GSW fitted nominal zero curve (Svensson parameters) | 1961– | Public domain (US government) |
| Federal Reserve `feds200805` | GSW fitted TIPS real curve | 1999– | Public domain |
| yfinance | Raw **unadjusted** daily OHLCV plus a separately hashed corporate-actions table: 9 sleeve ETFs, TLT, 2 diagnostic controls | 1993– | Not redistributed |
| yfinance | Daily bars for every S&P 500 ticker the membership tables name (876 requested) | 1962– | Not redistributed |
| FRED / ALFRED API | `DTWEXBGS`, `DGS1MO`, plus initial-release series and vintage dates | 2001– | Not redistributed |
| FRED (ICE BofA) | `BAMLH0A0HYM2`, `BAMLC0A0CM`, `BAMLH0A0HYM2EY` | **2023-08-29–** (see below) | ICE forbids reproduction; never committed |
| Ken French Data Library | Daily research factors; the one-month bill is the risk-free leg | 1926– | No explicit redistribution licence — **never committed** |
| Ken French Data Library | `Siccodes49`, the FF49 industry definitions as SIC ranges | — | Same; a five-industry synthetic excerpt is the test fixture |
| AQR | Century of Factor Premia; TSMOM; Commodities for the Long Run | 1877– | No explicit redistribution licence — **never committed** |
| SEC EDGAR (XBRL frames + quarterly index) | Cover-page shares outstanding joined to filing dates by accession — point-in-time share counts | 2008Q4– | US government record; manifested, not committed |
| SEC EDGAR (submissions + filing headers) | Current SIC per CIK, and the SIC in each drifted name's first in-sample 10-K header | 2007– | US government record; one public header is a fixture |
| Wikipedia | S&P 500 constituents and the historical changes table, walked backwards into a **daily** membership matrix | 1957– | **CC BY-SA 4.0**, carried in the committed derived tables' own headers, separately from this repository's MIT licence |
| ETF issuers (SSGA, iShares, Invesco, Vanguard) | 30-day median bid/ask spread per fund, read by hand with URL and date | 2026 snapshot | Public fund pages; the file records URL and read date per row |
| Siblis Research | Total US equity market value, used only as a published **upper bound** on the reconstructed universe cap | 2007– | Cited, not redistributed |
| Stooq | Intended second price opinion | — | **Retired** — see below |

The two derived tables under `data/reference/` — the daily membership matrix and
its interval form — **are** committed, under CC BY-SA 4.0, because they are the
only way to check the universe reconstruction without re-running an 876-ticker
pull against a source that answers differently on different days.

**Three things to know before trusting any of this.** (i) **Yahoo rewrites history
backwards when a dividend is paid.** The mitigation is raw unadjusted bars plus a
separately hashed actions table, adjusted deterministically and forward-blind;
split continuity is asserted on every load and the trading calendar across eight
sleeve ETFs over 19 years shows zero discrepancies. **None of that detects a bar
that was already wrong in Yahoo when first cached.** (ii) **Seven of nine sleeve
tickers are Yahoo-only.** Stooq — the only free second opinion — answers every
request with a JavaScript proof-of-work browser challenge, solvable in about ten
lines of Python. **It was not solved**: it is an access control the operator put
up deliberately, and circumventing it is not something this project does. Two
genuinely independent cross-checks stand in its place — SPY against Ken French's
CRSP-derived `Mkt-RF` (ρ = 0.9786, mean gap −0.12%/yr against a 1.86%/yr dividend
signature) and TLT against a GSW par-coupon ladder (β = 1.0096) — and a contract
test asserts the single-sourced count is seven, so the limitation cannot quietly
stop being true. (iii) **FRED truncated the ICE BofA credit series to a rolling
three-year window in April 2026**; all three now start 2023-08-29, and tutorials
using them for long histories are out of date.

Generated data report with every measured number:
[`data_contracts.md`](reports/data_contracts.md).

---

## Method

The methodology of record is [`SPEC.md`](SPEC.md); this is the short version.

**Universe.** Fourteen instruments, **frozen and dated 2026-08-25 with a written
rationale per line** in [`config/universe.yaml`](config/universe.yaml), before any
backtest ran. The synthetic government series are the differentiator —
constant-maturity zeros built from the Fed's own Svensson parameters with carry,
roll-down and duration decomposed exactly, rather than an ETF whose duration
drifts. Headline window 2007-04-01 onward, with the last 18+ months held out.

**Model A.** Six macro factors: equity, two normalised rate principal components,
credit, commodity, dollar. The rate PCs are scaled so that regressing an asset's
excess return **in basis points** on the level factor returns **minus its
effective duration directly**, with no free parameter — which makes the validation
a point prediction about arithmetic rather than a guess about a market. Five of
six recover inside bands registered before the factor set existed; high yield is
refuted twice at the wrong sign, and the diagnosis (the spread channel offsetting
and then overwhelming the duration channel) is a finding rather than a fix. The
orthogonalization is a sequential expanding-window Gram-Schmidt over three
pairings. The specification's all-pairs correlation bar was **withdrawn as invalid
rather than relaxed** — unmeetable by any correct implementation, and if met would
have driven the factor correlation matrix toward diagonal and left the eigenfactor
adjustment nothing to operate on. It is replaced by the same criterion scoped to
the pairs the scheme targets (all four pass unaided) plus the **condition number of
the factor correlation matrix through time**, reported and never gated: raw
conditioning peaks at 17.7 in COVID and 14.4 in 2022 against a whole-sample median
of 9.3, while the orthogonalized set peaks at 6.7.
([`factor_condition_number.png`](reports/factor_condition_number.png),
[`factor_correlations.md`](reports/factor_correlations.md),
[`asset_exposures.md`](reports/asset_exposures.md),
[`rolling_betas.png`](reports/rolling_betas.png))

**Model B and the hybrid.** A statistical PCA model with MP denoising, and a
hybrid appending residual principal components to the macro core. Both built, both
through the pipeline unchanged, both reported — and **neither was ever
optimizer-consumable** (see [Limitations](#limitations)). One result is worth
keeping: a hypothesis registered **five sessions before the code that could test it
existed** — that a residual PC would load on gold and correlate with the real-yield
series — was **refuted**, gold ranking 11th of 13 on both components; but the
falsifier's second half ("the residual PCs are noise") is also wrong. The leading
residual PC is the **third mode of the yield curve**: `R²` = 0.6247 against the four
raw yield changes against 0.0002 against the two rate factors it was built
orthogonal to. It detects real structure the named factor set missed, and what it
detected already has a name.
([`statistical_factors.md`](reports/statistical_factors.md),
[`statistical_spectrum.png`](reports/statistical_spectrum.png),
[`hybrid_residual_pcs.md`](reports/hybrid_residual_pcs.md),
[`residual_pc_variance_share.png`](reports/residual_pc_variance_share.png))

**The covariance pipeline.** `src/mafrm/risk/` takes
`(factor_returns, exposures, residuals, config)` and returns matrices; it does not
know what asset class it is looking at, and a test asserts that by static scan.
Every constant comes from [`config/model.yaml`](config/model.yaml), traceable to
USE4 Table 4.1 or the CNE5 Descriptor Details. Two implementation decisions are
worth naming. The PSD *detection* threshold is expressed in units of machine
epsilon relative to the largest eigenvalue, because an absolute threshold correct
on one panel sat an order of magnitude *inside* the representation noise on
another. And the eigenfactor Monte Carlo is resolved in **correlation space**,
because the panel mixes decimal returns with basis points of yield and a correction
whose output changes with the numeraire is not a risk correction — invariance
measured at **7.4e-16**, against a power control showing the literal
covariance-space form fails the same test at 1.75e-2. **PSD is asserted after every
stage, not only at the end**: 420 month-end builds on the equity panel, 4,141
expanding builds per horizon on the macro panel, zero failures.

**The optimizer.** Single-period, `cvxpy` directly — never `cvxportfolio`, which is
GPL-3.0; its cost specification was reimplemented from the published form and
cited. Costs sit **inside** the objective, alongside a factor-structured risk term
with PSD asserted, an alpha-misalignment penalty, hard limits as observed
multipliers or hinge penalties, and a written-down relaxation ladder. 187 of 187
in-sample and 18 of 18 holdout rebalances solved at the first rung.

**The equity module.** Six of USE4's twelve style factors — the six needing no paid
data — plus 49 FF industries and a country factor. Shares outstanding are
point-in-time from EDGAR cover-page facts, known from each filing's date, behind a
within-name consistency screen that drops 74 filings over 49 names (each a count
tagged in thousands or millions, or a shell) and is validated against a published
upper bound on all 18 year ends. The cross-section is a **√cap-weighted** WLS —
MSCI's stated assumption that specific variance is inversely proportional to √cap,
which materially changes the factor returns — under a cap-weighted industry
sum-to-zero constraint, both identities closing to 1e-17 on all 4,468 dates. Mean
weighted daily cross-sectional `R²` is **0.3868** against Connor (1995)'s 0.39:
refuted by 0.3 percentage points and not materially, since industries alone explain
0.300 and the missing three points are FF49's cost against a proprietary GICS-based
scheme. ([`equity_regression.md`](reports/equity_regression.md),
[`equity_regression.png`](reports/equity_regression.png),
[`equity_industries.md`](reports/equity_industries.md),
[`equity_market_cap.png`](reports/equity_market_cap.png),
[`equity_descriptor_correlations.png`](reports/equity_descriptor_correlations.png),
[`universe_screen.png`](reports/universe_screen.png))

---

## Reproducibility

```
make data      # refresh cache from sources, update data/manifest.json
make verify    # re-hash everything, fail on drift
make model     # build factors, covariance, specific risk
make backtest  # run the experiment grid
make report    # regenerate reports/ and results/metrics.json
make test      # ruff + mypy + pytest
```

`uv` with a committed `uv.lock`; `requires-python >= 3.11`; `yfinance` pinned
tightly as the highest-churn dependency. **`make verify` checks integrity, not
presence**: `data/raw` and `data/processed` are gitignored, so a clean clone
legitimately has almost no cache. On one it verifies the **2 of 96** recorded
files that *are* committed — the two `data/reference/` membership tables — lists
the other 94 as not on disk with `run make data`, and **exits 0**; with nothing on
disk at all it says "no cache present". A file that *is* on disk with the wrong
bytes is drift and still fails. Presence is enforced where it matters — at read
time, by name. **CI is two workflows**: a fast one on
every push, network disabled, running `ruff` + `mypy` + `pytest` against committed
fixtures on a 3.11/3.12 matrix; and a **scheduled weekly one** that hits live
sources and fails when a data contract breaks. The second is what proves the
pipeline still works six months after anyone stops touching it.

**Which lanes are green, stated precisely rather than as "CI passes".** The
audit that produced this section found the earlier version of it was not precise
enough — it reported a developer machine's numbers under a heading that promises
a lane. Both are given below, separately, because they are different facts.

**On a developer machine** (macOS/arm64, the locked toolchain): `ruff` clean,
`ruff format` clean, `mypy` clean on 98 source files **on both 3.11 and 3.12**,
**1,347 passed / 1 skipped / 137 deselected**. The deselected are the `network`
and `dataset` markers — tests that fetch, or that read the gitignored cache — and
they are the weekly lane's job, not this one's.

**The fast lane was RED from 2026-08-30 to 2026-09-08**, twenty-six consecutive
failing runs, and this section did not say so. Three independent causes, all
fixed in W8-P4b and each a way a green developer machine can hide a red lane:

1. **`mypy` failed on 3.11 only** — one `no-any-return` that 3.12's stub
   resolution does not raise. The declared minimum was not the interpreter the
   gate was ever run on. Fixed by annotating the return; 3.11 was **not** dropped
   from `requires-python` or the matrix, because moving the claim to fit the
   result is the same move as shrinking `N` to flatter a deflated Sharpe.
2. **The golden file was pinned to one platform** — see
   [Reproducibility](#reproducibility) above and SPEC.md 11: byte-identity is now
   asserted per platform where it has been *observed*, and every platform is held
   to a stated tolerance instead.
3. **Four tests read the gitignored cache without the `dataset` marker**, so the
   lane selected them and they failed on every machine without a populated
   `data/raw`. `ci.yml`'s own header says a clean clone is green without reaching
   a vendor; the comment was the specification and the tests were the defect.

The lane's verdict on *this* commit is the run attached to it, not a claim made
here.

**The weekly lane has never passed.** Two scheduled runs have ever executed:
**2026-08-26** failed at *Install project* on a `uv` configuration error
(`Default group 'dev' … is not defined in dependency-groups`), and **2026-09-02**
failed because **`FRED_API_KEY` is not set** — the repository has **no Actions
secrets configured at all**, so `make data` fails on the two FRED series and the
contract tests cascade on the missing cache. Making it green needs two decisions,
not one: the secrets (`FRED_API_KEY`, `SEC_EDGAR_USER_AGENT`), and what to do
about three **deliberately stale** AQR recency assertions — 92, 183 and 456 days
against a 90-day contract — which are red **on purpose**, because noticing that an
upstream source stopped updating is precisely what that workflow is for.
**Its schedule is therefore disabled in this snapshot and it runs on
`workflow_dispatch` only.** The workflow is kept rather than deleted because it is
part of the design and its contracts are worth reading; what is not kept is a
permanently red scheduled badge, which would say something false about the
repository in the one section written to be precise about exactly this. Its
**offline half** — the tests that read a populated cache without fetching — exits 0
on a developer machine, carrying the single known `xfail(strict=True)`: the GSW
par-coupon ladder's gap against TLT, residual 1 in
[Limitations](#limitations), deliberately left failing rather than have its
tolerance widened. Its **live-fetch half is still not run locally**, because a
same-day re-pull writes to the same date-keyed filenames and replaces their
manifest hashes — the finding recorded when a re-pull eighteen minutes after the
S&P 500 snapshot returned different bars.

**Cache keying is unchanged, and that is stated rather than fixed.** Raw pulls
are keyed by source, name and pull *date*, not by pull identity, so two pulls on
one day collide and the second silently replaces the first. The consequence a
reader meets is documented in [Limitations](#limitations):
`reports/universe_screen.*` cannot be regenerated byte-for-byte from a clean
clone. Re-keying by pull identity is a data-layer change with test churn behind
it and it was deliberately not made during a publication audit.

**Determinism.** One `SEED` in config, passed explicitly to every stochastic call
through `np.random.default_rng(seed)` — never a global seed, which would make the
eigenfactor Monte Carlo irreproducible. The holdout evaluation was executed twice
and every reported figure is bit-identical between the two. A clean rebuild from
`uv.lock` into an empty prefix resolves 68 distributions identical to the
incumbent with a zero-line diff.

**The trial ledger.** [`experiments.md`](experiments.md) logs every configuration
evaluated, whether it worked or not, with its category written **when the row was
written** and never reclassified. Only two of three categories feed the deflated
Sharpe's trial count:

| Category | What it is | Count | Feeds `N` |
|---|---|---|---|
| `data-diagnostic` | Validating the data layer against an external reference; bounding a construction difference; a control testing whether a statistic has power. No strategy, no reported Sharpe | **261** | no |
| `model-config` | A choice inside the risk model that changes a covariance or specific-risk estimate; reaches a Sharpe only through the optimizer | **12** | yes |
| `strategy-config` | Anything that changes the backtested portfolio | **59** | yes |
| **Total evaluated** | | **332** | **71 counted** |

**Two counts are reported and they differ by one.** The holdout evaluation is
itself a backtested portfolio carrying a Sharpe, so it is correctly a
`strategy-config` row and the ledger above stands at **332 evaluated, 71 in a
counted category** — but **the holdout's deflated Sharpe is computed at `N` = 70**,
being `model-config` (12) plus `strategy-config` (58) as they stood when the
configuration was frozen. The deflated Sharpe asks how likely an observed Sharpe
is under no skill given a search over `N` configurations from which this one was
chosen: the search space is what existed before the choice, the holdout is the
test rather than a member of the space it tests, and a statistic cannot deflate
itself. The two figures differ in the fourth decimal (0.7408 against 0.7407) and
both are reported. `N` moved **up**, never down.

**Deflated Sharpe: 0.7408 at `N` = 70** — well short of any conventional bar, and
reported as such. `results/metrics.json` carries **three** trial counts and they
are not a disagreement: each artefact records `N` **as it stood when that
artefact was generated** — `bands.trial_count_N` and the top-level
`trial_count_N` are **57**, the count when the grid and the bands ran in W6;
`holdout.trial_count_N` is **70**, the count when the configuration was frozen
and the one the deflated Sharpe uses; `holdout.trial_count_including_this_row`
and `overfitting.trial_count_N` are **71**, the ledger after the holdout's own
row. `N` only ever moves up, and a reader diffing the file is being shown a
history rather than an inconsistency.

**Probability of backtest overfitting: 0.6876**, degradation slope **−0.919**, by
CSCV over the grid's 28 cells × 187 months (16 blocks, 12,870 splits, in-sample
only and asserted twice). **This characterises the 28-cell search space, not this
result.** The grid's in-sample ranking carries no out-of-sample information — the
in-sample winner lands below the out-of-sample median on 69% of splits, and a cell
that looks better in sample does worse out of sample almost one for one — **so
selecting on backtest Sharpe would have made things worse than choosing at random,
which is why the reference cell was fixed by ruling before the grid ran and was
never selected on.** The grid's actual top cell is a cost-free cell that could
never ship and never went to the holdout. The `probability_of_loss` of 0.0000 says
the same from the other side: every split's winner still made money out of sample;
what fails to survive is the *ordering*, not the returns.
([`overfitting.md`](reports/overfitting.md))

**Tests.** Cost-model unit tests with hand-computed expected values; no-look-ahead
tests in both forms — recomputing a signal at `t` from data through `t`, and the
mechanical version that perturbs all data strictly after `t`, re-runs, and asserts
the weights at `t` are unchanged across ≥50 values of `t`, with a power check that
reports a constant state as *vacuous* rather than passed; calendar and alignment
tests; covariance property tests after every stage; a committed synthetic
golden-file fixture; and data-contract tests with recency assertions.

**The golden file pins a tolerance, not a hash, and the reason was measured.**
The fixture's panel is `draws @ chol(Σ)ᵀ`. `default_rng`'s stream is bit-identical
across platforms by numpy's guarantee, but the matmul and the Cholesky go through
the wheel's BLAS, which is free to reassociate its sums. On **darwin-arm64** the
digest is one value at both `K`, unchanged across OMP thread counts 1–8; on
**linux-x86_64** it differs from macOS at both `K`, and **at `K` = 40 it is not
stable run to run** — two digests across three CI runs on identical code and an
identical `uv.lock`. Suspect: OpenBLAS kernel dispatch across heterogeneous runner
CPUs; `K` = 3 is too small to reach a vectorised path and is stable. Named, not
confirmed — there is no Linux runner in scope to bisect on. So the fixture records
**both platforms' digests with a `byte_identical` flag and its evidence**, asserts
byte-identity only where it has been observed, and holds every platform to
`CROSS_PLATFORM_RTOL` = 1e-9 on the panel's absolute sum, sum of squares, minimum
and maximum. The tolerance is chosen a priori, not fitted, and it keeps the
tripwire: **a one-step change to the seed moves those statistics by 2.5% at
`K` = 3 and 11.3% at `K` = 40**, measured — seven orders above the tolerance,
which itself sits four orders above what a 2×10⁵-element double-precision
reduction can accumulate. Cross-platform bit-identity in a float pipeline through
BLAS is not achievable in general; pinning the anchor to one platform and calling
the repository reproducible would have been the worse answer.

---

## Limitations

Written to be read first. Each item says what it costs and in which direction.

**The macro universe is hand-selected and survivorship-biased, and says so in its
own header.** Fourteen instruments chosen in 2026, all of which still trade. An
ETF sleeve picked today is a sleeve of survivors; nothing here corrects for that,
and the frozen-and-dated rationale is a record of the choice, not a defence. It
matters least for what this project measures — a bias statistic and a `K/T`
scaling are not return studies — and would matter a great deal to anyone reading
the Sharpes as a return claim.

**The S&P 500 membership reconstruction is approximate and the source is
non-deterministic.** Membership is walked backwards from two Wikipedia tables and
inherits every gap in the changes table. **173 of 511 members at 2007-04-02 have
no Yahoo history at all**, falling to 12 of 503 at 2024-12-31, and 179 of 355
departed names have none — so the early sample over-represents survivors and the
specific-risk, residual-volatility and size cross-sections are biased toward
survivors' values, declining through the sample. Worse for reproducibility: a
re-pull eighteen minutes after the committed snapshot returned **667 ok / 209
empty against 684 / 192**, seventeen delisted tickers that had served full
histories now returning Yahoo's explicit *no price data found*. The matrix is
therefore **one snapshot**, rewritten only by an explicit flag, and
**`reports/universe_screen.*` cannot be regenerated byte-for-byte from a clean
clone.** The committed derived tables exist so the reconstruction is checkable
anyway.

**Spreads are estimated, not observed, and the level is not EDGE's.** Time
variation comes from a 63-day EDGE estimator on raw OHLC; the *level* comes from
**issuer-disclosed 30-day medians read by hand in 2026 and applied to 2009–2024**.
That anachronism is accepted deliberately and is the better of two bad options,
because the measured alternative is worse (see item 7 of the previous section).
Each disclosed figure is a 1bp rounding interval, so every half-spread is known to
±0.25bp and the model runs at both ends. **The shipped cell runs at the low end,
which is an interval floor and not a value** — SPY and IWM price at exactly zero
in every cell — so every cost figure in the grid is a *lower bound* on the spread
leg. For the bond sleeve, whose daily volatility is 0.08–0.44%, that 1bp
resolution is the binding uncertainty on the cost term rather than a rounding
detail: at 2% of ADV a 0.5bp half-spread is 122% of the impact term for SHY and
21–43% for the rest.

**Impact coefficients are borrowed, not estimated.** `Y` = 0.58 and 1.40 come from
AQR's and Virtu's published anchors and the 1.5 exponent from the literature.
Nothing here estimates a price-impact coefficient from its own fills, because it
has none. Both regimes are always run side by side; there is no interior point and
no preferred regime. Commissions are zero — the **absence** of a sourced figure,
not a value — which understates cost and therefore biases the identity's split
toward the risk term.

**Two splices are declared and neither is exercised.** The universe declares an IG
credit series extended pre-2002 by a Moody's-based model and a commodity series
spliced to AQR's Commodities for the Long Run pre-2006. The headline window starts
2007-04, so no splice discontinuity enters any reported number — and the pre-2002
IG reconstruction **was never built**: it needs a spread duration per rating bucket
and an expected-loss term this repository does not hold, and inventing either
would breach the no-invented-parameters rule. The third splice slot is
deliberately unspent. A reader who extends the window backwards inherits both
discontinuities, and neither is characterised here.

**The FRED ICE credit truncation.** All three option-adjusted spread series now
begin 2023-08-29 on a rolling window. The credit factor is therefore HY over cash
rather than a spread-based construction, and the pre-2007 credit reconstruction is
deferred, not built.

**The equity module is not a USE4 replica and should not be described as one.**
Six of twelve style factors, FF49 rather than a proprietary GICS-based scheme,
single-industry rather than multi-segment exposures, no Non-Linear Beta, and an
approximate point-in-time universe. **Five style factors are omitted for lack of
data, and the omission is not neutral:** Book-to-Price, Earnings Yield, Growth,
Leverage and Dividend Yield all need point-in-time fundamentals, and the
analyst-estimate descriptors inside them (EPIBS, EGIBS) need I/B/E/S and have **no
free substitute at all**. That matters more than the count suggests, because **the
analyst-estimate descriptors carry the largest published weights in their
composites — EPIBS alone is 0.68 of Earnings Yield.** The six built here derive
entirely from daily price, volume and shares outstanding. Other measured
shortfalls, reported and unrepaired: thirteen thin industries on the median date;
NLSIZE flagged as a pruning candidate at an 11.0% `|t| > 2` frequency against a
registered 15% floor and **kept, not pruned**; 28 names on 2024-12-31 carrying
8.76% of universe dollar volume with no cap at all, because dual-class filers
report the cover-page count per class with a dimension the XBRL frames do not
carry; 2.2% of universe cells carrying splits the actions table lacks; and every
pre-first-filing share count a labelled approximation, covering 13.4% of cells with
a cap.

**The diagonal specific-risk assumption is violated, and where is measured.** Of
all 78 residual pairs on the macro panel the largest is `govt_10y`/`govt_30y` at
**0.9028** — level and slope cannot span a curve, and what is left is the third
curve mode. SPY/IWM is **−0.5488**, confirming the specification's own prediction
**with the opposite sign**, which matters: a linked-asset override assuming the
intuitive positive correlation would push the covariance the wrong way. LQD/HYG
cannot be tested on this factor set at all, because the credit factor is built from
HYG. At `N` = 447 the diagonal is **in regime**. It is nowhere repaired,
deliberately.

**Model B and the hybrid were never optimizer-consumable.** Their per-date
`(X, F, Δ)` do not exist: building one needs an asset-level specific-risk
definition for a statistical factor model that the specification does not give and
the config does not hold, and inventing one would breach the no-invented-parameters
rule. **Every claim this project makes about Model B and the hybrid is a claim
about a covariance estimator scored on bias statistics, never about a portfolio.**
The "two factor models compared head to head" is delivered on the risk-model term
and **not** on the cost term; no Model B cell appears in any Sharpe table.

**The macro panel is small, and the specification said so in advance.** `K` = 6 and
`N` = 13 put `K/T_eff` at 0.008–0.025, where Shepard's closed form predicts a
1.7–5.1% understatement — so there is very little factor-covariance estimation
error for the eigenfactor stage to correct, and the finding that it corrects
nothing there is a statement about the panel's dimensions rather than about the
technique.

**Four of the specification's own acceptance gates were withdrawn as invalid**
rather than relaxed: a daily-correlation gate that could not move when the quantity
it tested was deleted; an all-pairs factor-correlation bar; a 0.3 correlation bar
written for a comparand this project does not hold; and an eigenfactor amplitude
gate calibrated for large `K`. Each withdrawal is argued in full in
[`experiments.md`](experiments.md) with its replacement and that replacement's
power evidence. A reader who suspects a gate was moved to fit a result should read
those four arguments — they are the ones that would show it.

**Eighteen months has no power, and that was written down before the window
opened.** Lo standard error 0.86; exact interval for `B` [0.667, 1.333]. Nothing in
that window can refute anything, and the only reportable finding is the *contrast*
described in [result 3](#3-the-holdout-run-once).

**The boundary was crossed once, in week 2, and the crossing is disclosed rather
than argued away.** While scoping a validation row, a Pearson correlation between
this project's SPY excess return and Ken French's `Mkt-RF` — both data-layer price
series — was computed over 2025-01-01 to 2026-06-30. It should not have been run.
The value is deliberately not transcribed anywhere here, because writing it into
the permanent record would propagate a holdout statistic into every future session;
it reaches no config, report, test, gate or committed artefact, and it informed no
decision. **But the claim that nothing here has been read on or after
`HOLDOUT_START` is withdrawn rather than quietly restated**, and that it was
harmless is not offered as a defence: a bright line exists precisely to refuse the
argument that a particular crossing did no damage. The consequence was structural —
the guard moved out of discipline and into the loader, `cache.read` now truncates
at the boundary by default, the only route past it requires a stated non-blank
reason, and a test enumerates every file permitted to use it.

**The holdout window was opened three times and read once.** The first two
invocations of the evaluation raised before emitting, rendering or writing any
number. In the first, the equity eigenfactor history was built from the arithmetic
floor rather than from the first clean build, so the Monte Carlo's Cholesky reached
the pre-repair region and raised a linear-algebra error. In the second, a function
returning one value per column was unpacked as a pair. Both are defects in the
evaluation runner rather than in the model, both crashed inside the equity leg
*after* the macro leg had computed and *before* anything was printed or written,
and no holdout figure was seen by anyone until the third invocation completed. This
is recorded because a crash inside the crossing is still a crossing, and the honest
record is that the window was opened three times. **The diff check that makes the
walk-forward legitimate** is asserted rather than assumed: extending the data panel
past the boundary must not change any in-sample forecast, and the last 40 in-sample
eigenfactor rows rebuilt off the extended panel agree with the committed cache to
**5e-8 absolute on a scale of 8.4e3** — the cache's own 12-significant-digit CSV
formatting and nothing else. A larger gap refuses the run.

### Residuals left open, each with a named suspect

A residual may be left open only when it does not affect the production data path,
is small relative to what it perturbs, and has a named suspect with a stated reason
it cannot be settled in scope — for at most one session past the first diagnosis.
Eight are open:

| # | Residual left open | Named suspect | Why it cannot be settled in scope |
|---|---|---|---|
| 1 | The par-coupon ladder runs **+0.114%/yr** against TLT's index after every term is reconciled | GSW fits a *filtered* bond set excluding on-the-run and first off-the-run issues whose liquidity premia would distort the curve, while the index prices the actual outstanding stock; and the index is market-value weighted where this ladder is eleven equally-spaced maturities | Needs bond-level CUSIP data this project will not acquire. **The tolerance was not widened** — the test stays `xfail(strict=True)` at −0.192%/yr against a 0.15%/yr limit, so the failure stays visible |
| 2 | The EDGE level on liquid equity ETFs reads ~60× the quoted spread | Two survive after six were eliminated by measurement (including a pre-registered control that was refuted): mixed price formation between the daily range and the close, and auction prints against a continuously-formed range. One post-hoc pattern — futures-linked arbitrage producing negative serial covariance without being a spread — is fenced and explicitly not cited as a finding | Needs intraday consolidated-tape data with venue and session flags. Off the production path by permanent design, not quarantine |
| 3 | The macro panel's optimizer-portfolio bias **grows 1.06 → 1.59** across sixteen non-overlapping blocks while `K/T_eff` is fixed | A residual correlation structure that strengthened after 2013 | The test written to settle it ran and left it standing rather than settled: the growth tracks `K/T_eff` on neither panel (rank correlation −0.03 equity, −0.70 macro) |
| 4 | The equity model's bias rises as the estimation window shortens by **~38× more** than the estimator's own simulated theory predicts | **None promoted.** The simulation contains no fat tails, no clustering, no specific-risk error and no specification error; the responsiveness channel is confounded with the sweep at every point | The separating test is registered and unrun. **This is the residual that most limits what [result 2](#2-the-two-panel-kt-contrast-and-a-closed-form-that-does-not-reproduce) can claim** |
| 5 | The specific volatility regime adjustment over-marks: the fully specified equity variant's random-portfolio `B` is 0.949, outside its interval, with 76% of random books inside on the specific component against a registered 90% | A cross-sectional mean of squares on a right-skewed distribution. The first suspect — singleton names dominating — was **refuted by its own re-read**, which moved both figures the wrong way | Not chased; one session past the first diagnosis was already spent |
| 6 | On 2020-03-31 in one grid cell the primary and fallback solvers land **8.9e-3 apart in L1**, both feasible to 7e-4 on a flat optimum, in the month the book turned over 1.43× NAV | The fallback solver's first-order tolerance on a flat objective | A third normalisation or a third solver would be tuning on outcome, so the cell was **not** re-run a third time — a deliberate departure from the letter of the governing ruling, recorded as such |
| 7 | A **2.4e-16** non-determinism in a reconstructed eigenvalue floor in one generated report, seen once in nine runs | Load-dependent reduction order inside multi-threaded LAPACK | Settling it means pinning BLAS thread counts across every entry point. A quarter of one unit in the last place of a manufactured floor, on a test fixture, off the production path |
| 8 | A SIC **change-and-revert** inside an otherwise unchanged name is invisible to the two-point drift check | The same re-coding process that moved the 27 names the check did flag | Bounded above by the frequency of re-codings among names that drifted (41 code changes over 538 names in seventeen years), and it would have to change *and revert* inside the sample to matter |

**And one contract that is deliberately red.** The weekly data-contract workflow is
expected to go red when an upstream source changes; that is its job. The vendor
recency limit specified in week 1 failed on all three AQR files and was **replaced
rather than widened** — by a coverage contract over the compared window plus a
release-drift detector whose cadence is inferred from the manifest's own release
log, which with one observed release **cannot fire**. That is the honest state, and
it is stated rather than dressed up as a passing check.

---

## Deliberately out of scope

- **Return forecasting, alpha research, signal discovery of any kind.** Expected
  returns enter as one fixed, published, never-tuned input.
- **Multi-period / MPC optimization.** Boyd et al. (2024) argue single-period does
  essentially as well outside specific cases — transitions, planned liquidations,
  strongly multi-scale signals — none of which this book has. Single-period is
  implemented, the reason is cited, and the project moves on.
- **Intraday execution modelling**, child-order scheduling, venue analysis.
- **Regime-switching or HMM factor models; machine learning of any kind.**
- **SEC Rule 605 spread archaeology** — a documented two-week rabbit hole, declined
  in advance.
- **Live trading, dashboards, web UI.**

---

## Positioning

**cvxportfolio** has the best cost model in open source, no factor risk model, and
is GPL-3.0 — which is why its cost specification was reimplemented from the
published form and cited rather than imported. **toraniko** is a real
cross-sectional equity risk model and the reference implementation in that space,
but has no optimizer and no cost layer; the week-7 module here is not a competitor
to it. **skfolio** has excellent optimizer plumbing and covariance estimators but
no macro multi-asset factor structure and no data layer. **OptimalPortfolios** is
multi-asset and cost-aware but has no factor risk model and no reproducible
free-data pipeline.

**Nobody ships a free-data-reproducible multi-asset factor risk model with a
time-varying, empirically-calibrated cost layer and a published bias statistic.**
This project differentiates on four axes and deliberately no others: time-varying
empirically-estimated costs; bias-statistic validation with confidence bands;
factor validation against published sources as a falsifiable table; and
reproducibility as a feature. It does **not** differentiate on the optimizer, on
risk measures, or on plotting — reimplementing a mean-variance solver reads as not
knowing what is already solved, while reimplementing the cost model reads as
knowing exactly where the interesting part is.

---

## Repository, licence, citation

```
config/universe.yaml     frozen, dated, rationale per line
config/model.yaml        every tunable number, cited inline
data/manifest.json       provenance: URL, timestamp, SHA-256, rows, date range
data/reference/          committed S&P 500 membership tables (CC BY-SA 4.0)
src/mafrm/data/          loaders, cache, manifest, contracts  ← only network access
src/mafrm/factors/       macro, statistical, equity cross-sectional
src/mafrm/risk/          covariance, specific risk, validation  ← asset-class agnostic
src/mafrm/costs/         EDGE, ADV, impact  ← the most heavily tested module
src/mafrm/backtest/      engine, optimizer, constraints, diagnostics, reporting
tests/
reports/                 generated figures and tables, COMMITTED
results/metrics.json     every headline number, regenerated by make report
experiments.md           every configuration evaluated, with category and date
SPEC.md                  the methodology reference
```

**Licence: MIT** ([`LICENSE`](LICENSE)). Third-party raw data is never
redistributed; the Wikipedia-derived membership tables carry CC BY-SA 4.0 in their
own headers. Citation metadata: [`CITATION.cff`](CITATION.cff).

**Author:** Canberk Tahil.
