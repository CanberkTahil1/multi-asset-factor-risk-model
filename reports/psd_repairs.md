# The PSD repair: how often it fires, and by how much

Generated 2026-09-02 by `python -m mafrm.factors.psd_repair_report`. SPEC.md 5.2.

SPEC.md 5.2 makes the eigenvalue repair after Newey-West mandatory and then asks for
a number: *"log how often it fires and by how much -- that number is itself a
diagnostic."* This is that log.

## What was scanned

The covariance pipeline was rebuilt on an **expanding window ending at every date**
in the factor panel, at both horizons: 4,141
builds per horizon, from `T = K = 6` (the arithmetic minimum, below
which the second moment is singular by construction) to `T = 4,146`.
Scan range: 2008-04-21 to 2024-12-31. The panel itself starts
2008-04-14; its leading 5 dates cannot be
scanned, because `T < K` is singular by construction and there is no matrix to check.
Everything ends strictly before `sample.holdout_start` = 2025-01-01.

No stride. A stride would sample the firings rather than count them, and the count is
the deliverable.

> **CLAUDE.md failure mode 9.** Consecutive readings are expanding windows differing by
> **one observation**, so they share all but one of their inputs. The number of readings
> is **not** a count of independent draws. This report quotes **no firing rate, no
> probability and no confidence interval** -- only a raw date count and a count of
> contiguous episodes, labelled as such. Five consecutive firing dates are one episode
> seen five times, not five events.

## Result

### `short` horizon (volatility half-life 84d at 5 lags, correlation half-life 504d at 2 lags)

- **Firing dates: 5** of 4,141 builds *(a date count, not a rate -- see the box above)*
- **Contiguous episodes: 1**
  - 2008-04-21 to 2008-04-25, 5 consecutive dates, most negative eigenvalue -9.645e-06
- **Every firing is at `T <= 10` against `K = 6`.** The repair never fires again in the 4,136 builds from `T = 11` onward.
- Closest approach after the last firing: `lambda_min` = +6.454e-07 on 2008-04-28 (`T = 11`), which is +6.08e-09 of `lambda_max` -- **positive**, so it is a margin and not a breach.
- **Margin against SPEC.md 5.2.4's detection threshold: the repaired matrix's least eigenvalue never falls below +0.284 ulp of its own `lambda_max`**, against a bound of -8. Worst on 2008-04-21 (`T = 6`), where `lambda_min` = +1.000e-14 beside `lambda_max` = 1.586e+02.
- Largest repaired `lambda_max` anywhere in the scan: 1.5864e+02 on 2008-04-21, so one ulp of it is 3.523e-14 -- **28x the superseded absolute threshold of -1e-12.** On THIS panel that threshold was therefore sound, and the justification `config/model.yaml` gave for it ("variances run of order 1e1-1e2") was accurate for the panel it was written against.
- **It stopped being accurate when the panel changed, and nothing re-derived it.** SPEC.md 4.3's hybrid residual panel carries the same six factors at a `lambda_max` of 5.43e+04 -- 390x this one -- where one ulp is 1.21e-11, twelve times the old threshold. That is the panel W4-P1 and W4-P2 build on, and two of W4-P2b's nine sweep half-lives were refused there on matrices that were positive semi-definite to a fifth of one ulp. SPEC.md 5.2.4 and `experiments.md` row 174. **The relative form is panel-independent by construction, which is the property the absolute one lacked and the reason this line is now printed at every build rather than only where the repair fired.**

| Date | `T` | `K/T_eff` realised | EWMA `lambda_min` | NW `lambda_min` | NW `lambda_min / lambda_max` | eigenvalues floored |
|---|---|---|---|---|---|---|
| 2008-04-21 | 6 | 1.000 | +3.923e-07 | -9.645e-06 | -6.08e-08 | 2 |
| 2008-04-22 | 7 | 0.857 | +1.045e-06 | -5.488e-06 | -4.17e-08 | 1 |
| 2008-04-23 | 8 | 0.750 | +4.710e-06 | -3.708e-06 | -3.11e-08 | 2 |
| 2008-04-24 | 9 | 0.667 | +5.336e-06 | -2.244e-06 | -1.66e-08 | 1 |
| 2008-04-25 | 10 | 0.600 | +4.853e-06 | -6.878e-07 | -4.94e-09 | 1 |

### `long` horizon (volatility half-life 252d at 5 lags, correlation half-life 504d at 2 lags)

- **Firing dates: 5** of 4,141 builds *(a date count, not a rate -- see the box above)*
- **Contiguous episodes: 1**
  - 2008-04-21 to 2008-04-25, 5 consecutive dates, most negative eigenvalue -9.710e-06
- **Every firing is at `T <= 10` against `K = 6`.** The repair never fires again in the 4,136 builds from `T = 11` onward.
- Closest approach after the last firing: `lambda_min` = +6.507e-07 on 2008-04-28 (`T = 11`), which is +6.07e-09 of `lambda_max` -- **positive**, so it is a margin and not a breach.
- **Margin against SPEC.md 5.2.4's detection threshold: the repaired matrix's least eigenvalue never falls below +0.283 ulp of its own `lambda_max`**, against a bound of -8. Worst on 2008-04-21 (`T = 6`), where `lambda_min` = +1.000e-14 beside `lambda_max` = 1.591e+02.
- Largest repaired `lambda_max` anywhere in the scan: 1.5908e+02 on 2008-04-21, so one ulp of it is 3.532e-14 -- **28x the superseded absolute threshold of -1e-12.** On THIS panel that threshold was therefore sound, and the justification `config/model.yaml` gave for it ("variances run of order 1e1-1e2") was accurate for the panel it was written against.
- **It stopped being accurate when the panel changed, and nothing re-derived it.** SPEC.md 4.3's hybrid residual panel carries the same six factors at a `lambda_max` of 5.43e+04 -- 390x this one -- where one ulp is 1.21e-11, twelve times the old threshold. That is the panel W4-P1 and W4-P2 build on, and two of W4-P2b's nine sweep half-lives were refused there on matrices that were positive semi-definite to a fifth of one ulp. SPEC.md 5.2.4 and `experiments.md` row 174. **The relative form is panel-independent by construction, which is the property the absolute one lacked and the reason this line is now printed at every build rather than only where the repair fired.**

| Date | `T` | `K/T_eff` realised | EWMA `lambda_min` | NW `lambda_min` | NW `lambda_min / lambda_max` | eigenvalues floored |
|---|---|---|---|---|---|---|
| 2008-04-21 | 6 | 1.000 | +3.919e-07 | -9.710e-06 | -6.10e-08 | 2 |
| 2008-04-22 | 7 | 0.857 | +1.045e-06 | -5.534e-06 | -4.19e-08 | 1 |
| 2008-04-23 | 8 | 0.750 | +4.701e-06 | -3.730e-06 | -3.11e-08 | 2 |
| 2008-04-24 | 9 | 0.667 | +5.336e-06 | -2.239e-06 | -1.65e-08 | 1 |
| 2008-04-25 | 10 | 0.600 | +4.844e-06 | -6.912e-07 | -4.95e-09 | 1 |

## What the firings actually are

Both horizons fire on the same 5 dates and nowhere else, and
every one of them is in the regime where six factors are being estimated from ten
observations or fewer. The `K/T_eff` column is the reading: the firings run
1.000, 0.857, 0.750, 0.667, 0.600 and stop.

The negative eigenvalue is genuinely produced by Newey-West and not by the window
being short in itself -- the EWMA stage is a Gram matrix and is positive
semi-definite by construction at every one of these dates, with `lambda_min` in the
3.9e-07 to 5.3e-06 range. It is the
Bartlett lag terms that push it below zero, on windows where the longest lag has
one or two pairs to average over.

**This is the empirical shape of SPEC.md 5.1.2's forward constraint, and it arrived
sooner than expected.** That ruling says a burn-in minimum must be expressed as a
bound on `K / T_eff` rather than as a day count, because a day count would have to
be re-chosen for every panel and every half-life and says nothing about how many
parameters are being estimated. The boundary observed here is a `K / T_eff`
boundary: it sits at the same place at both horizons even though their volatility
half-lives differ by a factor of three, which a day count could not have predicted.

**It is not a calibrated bound and must not be used as one.** It is one episode of
5 overlapping dates on one
panel with one factor set. It is a shape, not a threshold; the threshold, when
W3/W4 needs one, owes its own registration.

**And it is the first independent corroboration in this project that `K/T` is the
governing parameter -- of a different kind from the last one.** W3-P1 reproduced
SPEC.md 15.2's table by computing Shepard's formula and finding it agrees, which
validates the **arithmetic**. Nothing here computes Shepard's formula. The same
parameter arrives from a **numerical failure mode**, in an implementation that was
not aiming at `K/T` and would have reported whatever boundary the data had.
Reproducing the published table validates the arithmetic; this validates the
framing.

## The floor, against the spectrum

CLAUDE.md invariant 4 fixes the repair floor at `1e-14` and **it has not been
moved**. But an absolute floor is a statement about a matrix in particular units,
and this project's factor panel is deliberately in mixed units (SPEC.md 4.1.1). So
the floor is quoted against the spectrum rather than on its own:

| Horizon | floor | `lambda_min` (full window) | `lambda_max` | floor / `lambda_min` | orders of magnitude below `lambda_min` |
|---|---|---|---|---|---|
| `short` | 1e-14 | 3.896e-06 | 3.695e+01 | 2.57e-09 | 8.6 |
| `long` | 1e-14 | 5.741e-06 | 4.424e+01 | 1.74e-09 | 8.8 |

So the floor currently sits about eight orders of magnitude below the smallest
legitimate eigenvalue. **That is safe, and it is safe by accident of the unit
convention rather than by design.** On a differently-scaled matrix the same constant
could clip real structure: SPEC.md 5.3.1 records that this panel's covariance
condition number of 9.96e6 is almost entirely a 7.3e6 variance ratio across
mixed-unit columns, not near-singularity.

**W3-P3's numeraire decision changes what this floor means and must trigger a
re-read of this table.** Moving the panel to a common numeraire, or adjusting the
correlation matrix and rescaling back, changes `lambda_min` by whatever the
rescaling is worth -- and the floor, being absolute, does not move with it.

## Zero firings outside the degenerate regime is the expected result

Non-PSD after Newey-West is a **large-K** failure. At `K = 6`, with a correlation
matrix conditioned at 3.27 (`experiments.md` row 96) and a Bartlett kernel truncated
at 5 lags, a genuine estimation window may simply never produce a non-PSD matrix,
and the scan above says it does not: every firing is at `T <= 10`, and across the
whole pre-holdout sample from `T = 11` onward the repair does not fire once.

**That is not evidence the stage is broken.** Diagnosing it as one would burn a
session on a non-defect, which is the same error as the 0.3 correlation gate
withdrawn in SPEC.md 6.5.2 and the daily-correlation gate withdrawn in SPEC.md 3.2:
a test whose failure mode has not occurred is not a failing test.

What the requirement is instead: **the code path must be demonstrably live.** It is
shown on a synthetic case rather than only on cached data, because `data/raw` is
gitignored and a reader cloning this repository cannot run the scan above:

- `K = 6` factors from `T = 12` observations, drawn from `model.seed` = 20260825, `short` horizon
- Newey-West `lambda_min` = -7.161e-03 against `lambda_max` = 4.749e+00 -- a relative breach of -1.51e-03, about 1e13 times machine epsilon, so it does not turn on which BLAS is installed
- The repair floors 1 eigenvalue(s); the post-repair `lambda_min` is 1.004e-14 and the invariant-4
  assertion after the repair passes

`tests/test_covariance.py` pins this end to end at both horizons, alongside a
hand-computed repair of `[[1, 2], [2, 1]]` -- eigenvalues 3 and -1 -- and a
hand-computed Newey-West sum that is *itself* non-PSD (`det = -121/441`), which is the
cheapest available demonstration that the repair is not decorative.

**Week 7 is where this stage earns its place.** SPEC.md 15.2's cross-sectional
module runs `K = 56` -- one country factor, the 49 Fama-French industries of
SPEC.md 15.6 and the six descriptors of SPEC.md 15.4 -- at the same half-lives,
which raises `K / T_eff` by roughly an order of magnitude and is the regime MSCI's
own practice is written for. The expectation recorded here, before that module runs,
is that the repair fires there and does not fire on the macro model. If it fires on
neither, the honest reading is that the Bartlett kernel is doing its job, not that
the code is dead -- which is why the synthetic demonstration above exists rather
than a gate on the real firing count.

## Invariant 4, as read here

The `newey_west` stage **measures** its spectrum and does not assert it; the
assertion runs after `psd_repair`. That is a reading of CLAUDE.md invariant 4, not a
relaxation of it: the invariant exists to catch *silent* non-PSD, and a stage whose
successor in `covariance.stages` is its declared repair is not silent. Asserting
between the two would detect no defect; it would convert an expected and handled
outcome into a crash, and the only way to keep the pipeline running would then be to
widen `psd_minimum_eigenvalue` -- the move CLAUDE.md's residual stopping rule
forbids. `mafrm.risk.covariance.measured_not_asserted_stages()` names the exemption
and a test pins it to exactly one member, so it cannot spread quietly.
