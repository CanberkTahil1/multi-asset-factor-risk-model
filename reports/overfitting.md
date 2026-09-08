# Overfitting controls: PBO via CSCV, and the false-strategy bracket

SPEC.md 6.6's two remaining controls. **In-sample only** -- they describe the *search*, which happened strictly before `HOLDOUT_START`. The matrix ends **2024-12-31** and the runner raises if it reaches the boundary. `experiments.md` W8-P2b.

## The performance matrix

SPEC.md 9's **28 cells**, re-solved in sample, as monthly net returns over **187 months** (2009-06-30 to 2024-12-31). These are the configurations already counted as rows 211-238; the grid is re-run only because its per-cell return *series* were never saved, and re-running an already-counted configuration adds no row. **Nothing was swept and nothing chosen.** The patient regime of treatment D is taken, per SPEC.md 9.1 ruling 4 -- `Y` is an ingredient inside one cell, not a separate trial -- which is also the convention `run_grid` uses for `V[SR]`, so the bracket below is comparable to the holdout's.

Best cell by per-period Sharpe: `volatility_regime/none/patient` at 0.2200; the spread across cells is 0.1884 to 0.2200, `V[SR]` = 8.808e-05.

## PBO via CSCV

`S` = 16 contiguous blocks, so **C(16, 8) = 12,870 splits**, each pairing an in-sample half with its complement. 11 row(s) dropped from the front so the blocks divide evenly.

| | Value |
|---|---|
| **PBO** = `P[logit <= 0]` | **0.6876** |
| Median relative rank of the IS winner | 0.3793 |
| Median logit | -0.4925 |
| Performance degradation (OOS on IS slope) | -0.9190 |
| Probability of loss (winner's OOS Sharpe < 0) | 0.0000 |
| Splits | 12,870 |

## The false-strategy bracket

`E[max SR_N]` under zero skill at **`N` = 71** and `V[SR]` = 8.808e-05 per period is **0.0226** per period. The best cell's own per-period Sharpe is 0.2200, which is 9.7x the bracket.

## What these two numbers do and do not say

**PBO is a property of the search, not of the winner.** It asks how often the in-sample best lands below the out-of-sample median. Read it beside the fact that **these 28 cells share one alpha** and differ only in covariance treatment and cost treatment: their Sharpes sit in a narrow band, so which one ranks top is close to a coin flip by construction, and a PBO near 0.5 is what a search that discriminates nothing looks like. That was the registered expectation, and the result below is read against it. It is the same caveat `reports/experiment_grid.md` and `reports/holdout.md` carry about `V[SR]`.

**Neither control was ever going to condemn or clear this project's headline.** The quantity the project reports is not a selected Sharpe -- nothing was selected on backtest performance, which is why `N` counts trials that were never chosen between -- it is the decomposition of a gap. These are reported because SPEC.md 6.6 asks for them and because a search of 70 configurations owes its reader the number, not because a verdict hangs on them.

