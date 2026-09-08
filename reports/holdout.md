# The holdout, evaluated once

SPEC.md 9 and 12's single out-of-sample evaluation. Window **2025-01-01 to 2026-07-31**, **18 monthly rebalances**, cell **4D/patient** frozen before the boundary moved. `experiments.md` row 334. Generated 2026-09-06T21:38:56+00:00.

## Read the precision before the numbers

At **T = 18 months** SPEC.md 6.1's exact chi-square interval for `B` is **[0.667, 1.333]** -- against [0.898, 1.101] at the in-sample 187 -- and Lo's standard error on the Sharpe is **0.859**, about the size of the in-sample Sharpe itself (0.850). **Nothing here can refute anything**, and that was registered in row 334 before the window opened rather than discovered afterwards. This is a confirmation that the frozen pipeline runs unattended on unseen data and returns figures of the right order; it is not a test with power.

## The frozen configuration

| Dial | Value |
|---|---|
| Cell | `eigenfactor_a1.0/time_varying/patient` (variant 4, treatment D, patient) |
| Horizon | short |
| `TE_target` (per period) | 0.022526389 -- IN-SAMPLE, injected |
| NAV | $8,119,893 -- IN-SAMPLE book-size rule, injected |
| `gamma_trade` | 1.0 |
| Spread end | low |
| Estimates | walk forward; every covariance, specific-risk and cost input at a rebalance is re-estimated from data strictly before it |

### The harness control, run before the crossing

The same code path re-ran the frozen cell over the in-sample window and reproduced W6-P2b's published reference cell: bias_ratio 1.0611 (published 1.061), sharpe_net 0.8495 (published 0.85), lo_standard_error 0.2643 (published 0.264), risk_model_term 0.0543 (published 0.0543), cost_term 0.0400 (published 0.04).

## The right edge, per series

The edge is **2026-07-31**, the minimum over the book's series of their last observation, binding on **ken french daily**. `make data` was not run; these are the dates the verified cache already held.

| Series | Last observation |
|---|---|
| commodity | 2026-09-04 |
| dev_ex_us_equity | 2026-09-04 |
| dollar (factor) | 2026-08-28 |
| em_equity | 2026-09-04 |
| gold | 2026-09-04 |
| govt_10y | 2026-08-28 |
| govt_2y | 2026-08-28 |
| govt_30y | 2026-08-28 |
| govt_5y | 2026-08-28 |
| hy_credit | 2026-09-04 |
| ig_credit | 2026-09-04 |
| ken french daily | 2026-07-31 **(binding)** |
| risk-free bill | 2026-09-03 |
| tips_10y | 2026-08-28 |
| us_large_equity | 2026-09-04 |
| us_small_equity | 2026-09-04 |

## Headline

| Quantity | Holdout | In-sample (W6-P2b) |
|---|---|---|
| Net Sharpe (Lo scale) | **1.0374** +- 0.8586 | 0.850 +- 0.264 |
| Gross Sharpe | 1.1065 +- 0.8614 | 0.893 |
| `B` (identity) | **1.0681** | 1.061 |
| Interval for `B` at this `T` | [0.667, 1.333] | [0.898, 1.101] |
| `SR_paper` | 1.1691 | -- |
| `SR_real` | 1.0374 | -- |
| **Risk-model term** | **+0.0746** | +0.0543 |
| **Cost term** | **+0.0572** | +0.0400 |
| Cost drag | 29.9 bp/yr | 28.2 bp/yr |
| Turnover | 4.58x/yr | 5.62x/yr |
| Months | 18 | 187 |

### Every other metric the grid reports, on the same window

| Metric | Holdout |
|---|---|
| Gross return (annualised) | 5.73% |
| Net return (annualised) | 5.43% |
| Realised volatility | 5.23% |
| Forecast volatility | 4.90% |
| SPEC.md 6.1 `B` (rolling statistic) | 1.0681 |
| its interval, lower | 0.6670 |
| its interval, upper | 1.3326 |
| MRAD | 0.1439 |
| MRAD windows | 1 |
| TE attainment (rms, target = 1) | 1.0000 |
| Effective assets (mean) | 3.17 |
| Largest weight (median) | 44.93% |
| Largest weight (max) | 60.77% |
| Corner share of rebalances | 5.56% |
| Assets at the long-only floor (mean) | 8.94 |
| Spread share of realised cost | 75.11% |
| Impact share of realised cost | 24.89% |
| Information ratio (median) | 6.9943 |
| `gamma_risk` recovered (median) | 10.316 |
| Alpha-risk alignment `cos theta` (median) | 0.8762 |
| Financing leg (nominal) | -0.1137 bp/yr |
| Mean cash / NAV | -0.02546% |
| Relaxation ladder: rebalances at rung 0 | 18 |
| Rebalances relaxed | 0 |
| Rebalances on the fallback solver | 0 |
| `optimal_inaccurate` rebalances | 0 |
| Flagged and re-solved | 0 |
| Re-solve MAX L1 | 0.00e+00 |
| Actionable under the amended criterion | 0 |

## Against the benchmark (SPEC.md 10.3, Brinson-Fachler)

The benchmark is the **equal-weight book** (W6-P3 ruling 3), which is also what the risk target is sized against. Its return series is reported and **its Sharpe is not computed** -- a benchmark that acquires one becomes a trial, and a test asserts the diagnostics module names no Sharpe.

| | Cumulative over the holdout |
|---|---|
| Portfolio | +8.42% |
| Equal-weight benchmark | +10.57% |
| **Difference** | **-2.16%** |
| Allocation (Carino-linked) | -5.81% |
| Selection | -0.88% |
| Interaction | +4.53% |

**The optimized book underperformed equal weight over the holdout**, by 2.16% cumulative, and the decomposition puts it in allocation rather than selection. That is what a concentrated low-volatility book does in a window where the broad book runs; it is one 18-month window and no claim is made from it, but it is the comparison a reader will ask for and it is not a flattering one.

## The attribution reconciliation (SPEC.md 10.3's two assertions)

The return identity is exact by construction and holds to machine precision. The *variance* legs are forecasts against realisations, so they differ by exactly what `B` measures, and each additive-in-variance component is tested against the same exact interval at 18 periods: **[0.667, 1.333]**.

| Component | `B` | Inside | In-sample (W6-P3) |
|---|---|---|---|
| factor | 0.9550 | yes | inside |
| specific | **1.8405** | **no** | 1.23-1.32, fired |
| total | 1.0699 | yes | -- |

**The specific leg fires again, and harder: 1.84 against an in-sample 1.23-1.32.** This is SPEC.md 6.2.6's diagonal specific-risk misspecification -- the same mechanism control C1 isolated in W4-P2 and the same one W6-P3's TE band traced to the book's concentration -- showing up out of sample in the component the model actually gets wrong, while the factor leg (0.955) sits inside. A refutation on real data is a finding, not a reason the run is invalid, and the cell is marked rather than dropped.

## Perold's implementation shortfall (SPEC.md 10.1)

Degenerate by construction and stated as such (W6-P3 ruling 5): a monthly simulation that executes at its decision price has no execution schedule, so delay and opportunity are zero and the whole shortfall is impact.

| Leg | bp/yr |
|---|---|
| Total shortfall | 30.35 |
| Impact | 30.35 |
| Delay (zero by construction) | 0 |
| Opportunity (zero by construction) | 0 |
| Fees | 0 |
| Predicted by the optimizer | 30.35 |
| Predicted - realised residual | -3.151e-15 |

The predicted-versus-realised residual is 1.08e-15 bp at its worst date: under treatment D the optimizer priced the same as-of spread the engine charged, so it reconciles to machine precision.

## Tracking-error decomposition and constraint activity

| | Value |
|---|---|
| Factor share of forecast variance | 73.94% |
| Specific share | 26.06% |
| Ex-post / ex-ante, fixed weights | 1.0683 |
| Ex-post / ex-ante, drifted weights | 1.0699 |
| Hwang-Satchell drift share of variance | 0.30% |
| TE bound binding | 18 of 18 (100%) |
| ADV hinge active | 1 of 18 (5.6%) |
| Position box binding | 0 of 18 |

### Factor contribution to risk (mean share)

| Sleeve | Share |
|---|---|
| commodity | 17.26% |
| credit | 1.10% |
| dollar | 22.52% |
| equity | 25.57% |
| rates_level | 5.41% |
| rates_slope | 2.09% |

## SPEC.md 1's identity

```
SR_paper - SR_real = (mu_g/sigma_f)(1 - 1/B)  +  TC/(B sigma_f)
1.1691 - 1.0374 = +0.0746 + +0.0572
```

## Bias statistics, all four families

SPEC.md 6.2's four families on the holdout's scored sessions, same forecasts, `eigen_a1`. At `T` = 391 DAILY observations the exact interval is **[0.930, 1.070]** -- far tighter than the 18-month interval above, because these are daily and there are twenty times as many of them.

**`minimum_variance` here is NOT the `B` in the headline table.** Family 4 is SPEC.md 6.2's *unconstrained, daily-rebuilt* minimum-variance portfolio; the headline `B` is the identity's, on the *constrained, monthly* optimizer book that carries the tracking-error bound, the long-only simplex and the ADV hinge. W6-P3's TE band already showed the two move apart: the bias is a property of how concentrated the book is allowed to get, and the constrained book is not allowed to get as concentrated. In-sample the same pair read 1.332 and 1.061.

| Family | `B` | `T` | In-sample (W4-P2) |
|---|---|---|---|
| individual_assets | 0.9766 | 391 | 1.0245 |
| random_portfolios | 1.0871 | 391 | 1.0101 |
| eigenfactors | 1.0368 | 391 | 1.0415 |
| minimum_variance | 1.6673 | 391 | 1.3321 |

## Per year

| Year | Months | Net | Gross | Cost (bp) | Realised vol | Forecast vol |
|---|---|---|---|---|---|---|
| 2025 | 11 | +7.58% | +7.77% | 17.8 | 5.82% | 7.80% |
| 2026 | 7 | +0.32% | +0.59% | 27.0 | 11.72% | 7.80% |

## Deflated Sharpe

`N` = **70** -- the `model-config` + `strategy-config` count in `experiments.md` at the moment the configuration was frozen (12 + 58). `V[SR]` = 8.808e-05 per period, the dispersion measured across the 28 grid cells in W6-P2 and written to `results/metrics.json` then, so the bracket is not chosen after seeing the holdout.

| | Value |
|---|---|
| Sharpe (per period) | 0.1880 |
| `E[max SR_N]` (false-strategy bracket) | 0.0225 |
| Trials `N` | 70 |
| Periods | 18 |
| Skewness | -0.4622 |
| Kurtosis | 4.2366 |
| **DSR** | **0.7408** |

At `N` = 71 -- the count including this row itself -- the DSR is 0.7407. Both are reported so the choice of denominator cannot be a manipulation; they differ by 0.0001. The cells the bracket is built from share one alpha and are not the independent trials the formula assumes, which is the same caveat `reports/experiment_grid.md` carries.

## The equity control

The equity model's family-4 bias on the same window, `eigen_a1` at `K` = 54: **0.9417** daily-held over 373 sessions (2025-01-02 to 2026-06-30), against the published in-sample `eigen_a1` figure of 1.0342. Monthly 0.9469 over 18 months.

| Component | `B` on the holdout | Inside its interval | In-sample (`eigen_a1`) |
|---|---|---|---|
| factor | 0.8787 | yes | 1.0892 |
| specific | 1.3030 | yes | 0.9479 |
| total | 0.8370 | yes | 0.9994 |

**No Sharpe is computed on any equity family** (W7-P3's benchmark rule), so this leg is `data-diagnostic` and `N` does not move for it.

## Limitations, carried forward unrepaired

- **The equity model's bias rises as the estimation window shortens by more than any available second-order account predicts, and the driver is unidentified** (fat tails, the specific leg, SPEC.md 5.1.3's responsiveness channel). W8-P1b refuted the estimator's own simulated comparand as an explanation; the bootstrapped-innovation test is registered and was not run. Not repaired before the holdout.
- **Model B and the hybrid were never optimizer-consumable** -- no asset-level specific-risk definition -- so the head-to-head is on the **risk-model term only** and there is no Model B cell in this or any Sharpe table.
- **18 months is not a sample.** Everything above is reported with its interval and none of it is read as a verdict.

