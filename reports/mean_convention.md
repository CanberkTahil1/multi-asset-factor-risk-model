# The zero-mean convention, and what it costs

Generated 2026-08-31 04:27 UTC by `mafrm.factors.mean_convention_report`. 
SPEC.md 5.1.1; `covariance.mean_convention` in `config/model.yaml`; `experiments.md` rows 93-95.

## The decision, and why this file cannot change it

SPEC.md 5.1.1 takes the covariance pipeline's EWMA moments about **zero**. The reason is coherence, not size: SPEC.md 6.1's bias statistic is `b_nt = R_nt / sigma_nt` with a **raw** return in the numerator, so a denominator forecasting variance about a drifting mean would make the two sides measure different quantities, and `B` -- the one statistic the project exists to decompose into a risk-model term and a cost term -- would be biased by the mismatch alone, before any estimation error.

The ruling carries a second clause: *and at daily frequency the mean is small relative to the standard deviation anyway*. That is a claim about data, and it is the only part this file speaks to.

**No result here would reverse the decision, and that was registered before the measurement ran** (`experiments.md` rows 93-95). A large number below would be a finding about the factor means, not a licence to break coherence with SPEC.md 6.1. Having no reversing result is precisely why these rows are `data-diagnostic` and add nothing to the deflated-Sharpe trial count.

## The identity

For weights summing to one, with `m = sum(w f)` the EWMA mean, `sum(w f^2) = sum(w (f-m)^2) + m^2` **exactly**. So

```
sigma_zero / sigma_demeaned = sqrt(1 + (m / sigma_demeaned)^2)
```

The difference is *quadratic* in the mean-to-volatility ratio, which is why it is small without needing to be argued: a factor whose EWMA mean reached a tenth of its EWMA volatility -- a very large daily drift -- would move the volatility by 0.50%. Both quantities are tabulated below, so the identity can be checked rather than taken.

The comparand subtracts the trailing EWMA mean and **omits** the `1 - w'w` reliability correction that normally accompanies demeaning. That omission is deliberate: including it would confound the cost of the mean term with the cost of a degree-of-freedom adjustment, and the mean term is what was ruled on.

## Horizon: short (volatility half-life 84d, correlation half-life 504d)

Read every 21 trading days on EWMA windows whose volatility half-life is 84d and whose correlation half-life is 504d. Consecutive readings share almost all of their weighted history, so these are near-duplicates rather than independent observations. **No count of dates is quoted as `n` and no standard error is computed.** CLAUDE.md failure mode 9.

Window 2009-08-17 to 2024-12-31, stopping strictly before the holdout at 2025-01-01.

### Volatility

`sigma_zero / sigma_demeaned - 1`, in basis points of relative difference, and the driver `m / sigma_demeaned` that produces it.

| Factor | median \|shift\| | 95th pct | max | max \|m/sigma\| |
|---|---|---|---|---|
| `equity` | 18.78 bp | 70.61 bp | 224.23 bp | 0.2130 |
| `rates_level` | 10.82 bp | 87.79 bp | 193.34 bp | 0.1976 |
| `rates_slope` | 11.09 bp | 139.09 bp | 225.77 bp | 0.2137 |
| `credit` | 9.86 bp | 45.65 bp | 71.65 bp | 0.1199 |
| `commodity` | 5.65 bp | 96.22 bp | 161.87 bp | 0.1807 |
| `dollar` | 8.53 bp | 93.08 bp | 322.99 bp | 0.2562 |

**Worst relative volatility shift across all factors and all dates: 322.99 bp (3.2299%), at a largest observed `|m/sigma|` of 0.2562.**

### Correlation

`max |rho_zero - rho_demeaned|` over the off-diagonal, at the correlation half-life of 504d. This is the number that sized the W3-P1b re-basing of `mafrm.factors.macro.condition_numbers`: that diagnostic was demeaned through W3-P1 and now calls the pipeline's own estimators, so there is one code path rather than two computing the same statistic two ways. `experiments.md` row 68's published figures stand as published; the re-based ones move by at most 1.7% and round identically.

| | median | 95th pct | max |
|---|---|---|---|
| max off-diagonal |Δrho| | 0.0019 | 0.0043 | 0.0050 |

## Horizon: long (volatility half-life 252d, correlation half-life 504d)

Read every 21 trading days on EWMA windows whose volatility half-life is 252d and whose correlation half-life is 504d. Consecutive readings share almost all of their weighted history, so these are near-duplicates rather than independent observations. **No count of dates is quoted as `n` and no standard error is computed.** CLAUDE.md failure mode 9.

Window 2012-05-03 to 2024-12-31, stopping strictly before the holdout at 2025-01-01.

### Volatility

`sigma_zero / sigma_demeaned - 1`, in basis points of relative difference, and the driver `m / sigma_demeaned` that produces it.

| Factor | median \|shift\| | 95th pct | max | max \|m/sigma\| |
|---|---|---|---|---|
| `equity` | 12.53 bp | 30.38 bp | 50.93 bp | 0.1011 |
| `rates_level` | 3.72 bp | 29.27 bp | 61.79 bp | 0.1113 |
| `rates_slope` | 4.91 bp | 43.86 bp | 73.01 bp | 0.1211 |
| `credit` | 2.84 bp | 10.11 bp | 13.78 bp | 0.0525 |
| `commodity` | 1.90 bp | 32.09 bp | 71.81 bp | 0.1201 |
| `dollar` | 2.91 bp | 49.81 bp | 60.30 bp | 0.1100 |

**Worst relative volatility shift across all factors and all dates: 73.01 bp (0.7301%), at a largest observed `|m/sigma|` of 0.1211.**

### Correlation

`max |rho_zero - rho_demeaned|` over the off-diagonal, at the correlation half-life of 504d. This is the number that sized the W3-P1b re-basing of `mafrm.factors.macro.condition_numbers`: that diagnostic was demeaned through W3-P1 and now calls the pipeline's own estimators, so there is one code path rather than two computing the same statistic two ways. `experiments.md` row 68's published figures stand as published; the re-based ones move by at most 1.7% and round identically.

| | median | 95th pct | max |
|---|---|---|---|
| max off-diagonal |Δrho| | 0.0021 | 0.0045 | 0.0050 |

## What this settles, and what it does not

**The second clause of the ruling is only half supported, and this file says so.**

The *typical* cost is negligible exactly as the ruling anticipated: the median absolute volatility shift is 10.16 bp (0.1016%) at the short horizon and 3.95 bp at the long one. The *tail* is not negligible: across both horizons, every factor and every read date, the largest single shift is **323 bp (3.23%)**. On a volatility forecast that is a real number, not a rounding difference, and it is reported as one.

The mechanism is the identity, not a defect. The largest shifts occur where a factor's EWMA mean reaches roughly a quarter of its EWMA volatility -- an 84-day half-life is short enough for a sustained trend to do that -- and the identity then predicts a shift of 322.99 bp against the 322.99 bp observed. The comparand is what it claims to be.

### When the maximum occurs, which the identity cannot tell you

The identity holds for **any** `m`, so it confirms the arithmetic and says nothing about whether the drift is real. A shift of this size has to correspond to a nameable episode in the underlying series; a maximum landing on a quiet date would mean the EWMA mean is wrong, and the identity check would pass anyway.

**It does not land on a quiet date.** The maximum is `dollar` on **2015-01-26**, at `m/sigma` = +0.2562 -- an EWMA mean of +15.31%/yr against an annualised volatility of 3.77%/yr. That is four days after the ECB announced sovereign QE (22 January 2015) and eleven days after the SNB abandoned the franc's euro floor (15 January 2015), in the middle of the 2014-15 dollar surge; the factor's trailing 252-day sum to that date is +10.52%. Confirmatory, and the ratio is large for two reasons rather than one -- the numerator is a genuine trend, and the denominator is small because `dollar` is among the least volatile factors in the panel at 3.8%/yr.

The neighbouring readings say the same thing and are worth listing, because they are **not** independent confirmations -- they are one episode seen four times through overlapping windows, which is the CLAUDE.md failure mode 9 caveat applied to this file's own headline:

| Date | Factor | Shift |
|---|---|---|
| 2015-01-26 | `dollar` | 323 bp |
| 2015-02-26 | `dollar` | 233 bp |
| 2014-12-22 | `dollar` | 231 bp |
| 2018-04-17 | `rates_slope` | 226 bp |
| 2018-01-11 | `equity` | 224 bp |
| 2018-05-16 | `rates_slope` | 212 bp |

The two clusters are the 2014-15 dollar surge and the 2018 curve flattening. Both are episodes, not artefacts.

Correlations are a different story and the answer there is unambiguous: the largest off-diagonal disagreement anywhere is **0.00498**, at the 504-day half-life both horizons share. That bound is what made the W3-P1b re-basing of `mafrm.factors.macro.condition_numbers` cheap: the SPEC.md 4.1.2 conditioning diagnostic was demeaned through W3-P1 and now calls the pipeline's own estimators, which removes a second code path computing the same statistic a second way -- a trap for any later session that compared them. It cost a re-published diagnostic whose headline figures round identically.

### None of this moves the decision

Which is what was registered in advance (`experiments.md` rows 93-95), and the registration is worth more here than it would have been had the number come back small. **The tail result is a finding about the factor means, not an argument against the convention.** The convention rests on SPEC.md 6.1: the bias statistic divides a raw return by a forecast volatility, and a forecast of variance about a drifting mean is a forecast of a different quantity than the one being realised. That holds at 3 bp and it holds at 300 bp. A decision that would have been reversed by a large number was never a coherence argument in the first place.

What the tail *does* earn is an entry on the W4 watch list: `B` is estimated on windows where a factor can carry a mean of a quarter of its volatility, and the risk-model term of the SPEC.md 1 identity is measured through `B`. The direction is known and stated here so that it is not rediscovered as a puzzle later -- forecasting about zero makes the forecast volatility *larger* than a demeaned one by exactly this amount, so it biases `B` **downward**, never upward, and cannot manufacture an apparent risk-model failure.

