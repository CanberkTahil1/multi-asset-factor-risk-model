# Multi-Asset Factor Risk Model with a Cost-Aware Implementation Layer
## Build specification — 7 weeks

**Author:** Canberk Tahil
*(Two sections — §13 and §15.9 — covered CV and interview positioning rather than
methodology, and are not part of the public snapshot; §15.1's reason-one paragraph
was shortened for the same reason. The section numbering is left unrenumbered so
that every cross-reference in `experiments.md` and `reports/` still resolves.)*

**Purpose:** a reproducible multi-asset factor risk model with a cost-aware implementation layer. Primary focus: portfolio construction / portfolio implementation and buy-side investment risk analytics.
**Effort assumption:** 12–15 productive hours/week, 85–105 hours total.

---

# 0. The one-paragraph version

Every quantitative portfolio has a paper track record and a real one, and the two never match. This project measures the gap and splits it in two. Part of it is that the risk model lied — the optimizer's forecast volatility was lower than what actually happened, because the optimizer deliberately loads onto the directions where the covariance matrix is most optimistically wrong. The rest is that trading is not free — spreads widen exactly when the model wants to rebalance most. I build a multi-asset macro factor risk model with the full MSCI Barra covariance machinery, wrap it in a convex cost-aware optimizer, and produce an exact additive decomposition of the paper-versus-real Sharpe gap into a risk-estimation term and an implementation-cost term. Then I test which published fixes actually close each one.

**Write this on day one, at the top of the README:** *This project does not forecast returns.* The moment you start trying to make the backtest look good, you have changed projects, and the new project takes six months.

---

# 1. The research question, stated as an identity

Let a strategy have gross annualised return `μ_g`, forecast active volatility `σ_f` (what the risk model said ex ante), realised volatility `σ_r`, and annual cost drag `TC`.

Define the **bias statistic** `B = σ_r / σ_f` (Section 6). Then:

```
SR_paper = μ_g / σ_f
SR_real  = (μ_g − TC) / σ_r = (μ_g − TC) / (B σ_f)

SR_paper − SR_real  =  (μ_g/σ_f)·(1 − 1/B)   +   TC/(B σ_f)
                    =  ── risk-model term ──     ── cost term ──
```

This is **exact, not an approximation**, and it is the spine of the whole project. Both terms are measurable, both are attributable to specific modelling choices, and both have published fixes with published effect sizes. The deliverable is a table where every row is a model configuration and the two columns are those two terms.

Two properties worth noticing and stating in the README:

- The cost term is divided by `B`. A risk model that *under*-forecasts risk (B > 1) mechanically *understates* how much costs hurt your Sharpe. The two error sources are not independent, and any analysis that treats them additively without this term is wrong.
- The risk-model term is proportional to `SR_paper`. The better the paper strategy looks, the more absolute Sharpe you lose to a given amount of estimation error. High-Sharpe paper backtests degrade harder. This is the quantitative form of "if it looks too good in the backtest, it is."

### Pre-registered hypotheses

Write these down before running anything, commit them with a date, and report against them honestly whether or not they hold. Pre-registration is the single cheapest credibility signal in the project.

| # | Hypothesis | Basis | Outcome |
|---|---|---|---|
| H1 | Bias statistics on **random** portfolios will be ≈1.0 for every covariance variant, including the naive sample estimator. | Menchero-Wang-Orr: sampling error is first-order unbiased for estimation-independent portfolios. | **HOLDS** (W4-P2). Family 2 runs 0.957–1.011 across all nine variants and both horizons, the naive sample covariance included; the exact interval at `T = 3,874` is `[0.978, 1.022]`. |
| H2 | Bias statistics on **optimizer-selected** portfolios from the same matrices will be 1.2–1.5. | MWO report 1.4–1.5 for equities at T=200. Fewer assets here, so expect the low end. | **HOLDS** (W4-P2). Family 4 is **1.332 short / 1.314 long**, mid-range, against a family-2 control of 1.010 / 0.999 on the same matrix. |
| H3 | The eigenfactor adjustment closes most of the H2 gap and leaves H1 untouched. | MWO Table 1: 1.45 → ≈1.0. | **REFUTED on the first half, HOLDS on the second** (W4-P2). The adjustment moves family 4 by **−0.0001 at `a = 1.0` and 0.0000 at `a = 1.4`** — it closes none of the gap — while leaving family 2 within 0.0005. §6.2.5 has the reason, and it is not that the stage is broken: control C1 shows the gap is **not** factor-covariance estimation error. |
| H4 | Shepard's second-order correction predicts a **5.1%** volatility understatement for the 6-factor model versus **13.6%** for a raw 15-asset sample covariance at the same effective window — i.e. the factor structure buys a 2.6× reduction in second-order risk *before* any adjustment. | Shepard Eq. 32 vs Eq. 13, computed in §6.4. | Arithmetic reproduced in W3-P1 and restated in both units in W4-P3 (§6.4.1: 5.1% and 13.6% are variance multipliers minus one; 2.5% and 6.6% on volatility). **Empirically SPLIT (W4-P3, §6.4.4).** On the estimation-error term it holds by more than the arithmetic — measured split-window bias 2.04% of variance against the naive comparand's 11.7% closed form, 5.7× — but as a statement about family-4 bias on this panel it is REFUTED: `B` 1.33 against 1.07, because §6.2.6's diagonal costs thirty times what the factor structure saves. |
| H5 | Under a time-varying spread model, realised cost drag in the worst 5% of volatility days will be **2–4×** the flat-spread assumption, and cost-aware optimization will recover more Sharpe in those windows than in calm ones. | Spreads are linear in σ (AQR's VIX coefficient); the optimizer's desire to trade also peaks there. | **Leg 1 HOLDS in direction, beyond the band; leg 2 HOLDS in sign** (W6-P2, §9.2). Realised spread cost of the reference cell's trades in the ten worst rebalances by trailing volatility is **4.37×** the flat assumption on the same trades, 0.94× in calm months; costs-inside minus gross-then-net is 2.6 bp per period in the stressed months against 1.9 bp in calm ones, uncertainty not quantified at ten months. |
| H6 | Putting costs *inside* the objective beats optimizing gross and then netting costs off, by more than the difference between any two covariance estimators. | Boyd et al.; and the ordering claim is the interesting one — if it fails, that is a publishable finding for the README. | **HOLDS by the letter, by 0.006 Sharpe units** (W6-P2, §9.2): D − B = 0.0737 at the reference variant against a covariance range of 0.0675 under D, both a quarter of one Lo standard error. The robust half: costs inside beat gross-then-net at all seven variants by 0.05–0.07, and the covariance range is the dense-versus-factor gap rather than anything inside the pipeline. |

---

# 2. Scope

### In scope

- 12–15 asset macro multi-asset sleeve, frozen ex ante, monthly rebalance (with a weekly-rebalance sensitivity run).
- Two factor models, compared head to head: an **economically-interpretable macro factor model** and a **statistical PCA model with Marchenko-Pastur denoising**.
- Full Barra covariance pipeline on both: separate volatility/correlation EWMA half-lives → Newey-West → PSD repair → eigenfactor adjustment → volatility regime adjustment.
- Specific risk with the time-series / structural blend and Bayesian shrinkage.
- Validation battery: bias statistics on random *and* optimizer-selected portfolios, MRAD, Shepard correction, Mincer-Zarnowitz, Kupiec/Christoffersen VaR tests, factor-return t-statistics.
- Cost model: EDGE spread estimator on raw OHLC, ADV-scaled square-root impact, two calibrated regimes (patient / urgent).
- Single-period convex optimizer (cvxpy) with costs in the objective, soft constraints, alpha-alignment penalty.
- Diagnostics: Perold implementation shortfall decomposition, ex-ante vs ex-post tracking error, capacity curve, deflated Sharpe, PBO.

### Deliberately out of scope

State this section verbatim in the README. It is a maturity signal, not an apology.

- Return forecasting, alpha research, signal discovery of any kind.
- Multi-period / MPC optimization. Boyd et al. (2024) argue single-period does essentially as well outside specific cases (transitions, planned liquidations, strongly multi-scale signals). Implement SPO, cite the reason, move on.
- Intraday execution modelling, child-order scheduling, venue analysis.
- Regime-switching or HMM factor models; ML anything.
- SEC Rule 605 spread archaeology — a documented two-week rabbit hole.
- Live trading, dashboards, web UI.
- Single-stock equity factors (Barra USE4 style). This is the *macro* analogue; the equity version is a different project and `toraniko` already does it well.

---

# 3. Universe and data layer

**Build this in week 1 and never touch a live API again during modelling.** Data plumbing eats 40–50% of the time on every project of this shape; the only defence is to finish it before you start thinking.

## 3.1 The universe

Freeze it in `config/universe.yaml`, dated, with a written rationale per line, **before** any backtest runs. State in the README that the universe is hand-selected and survivorship-biased by construction. Practitioners respect that far more than a hand-wave.

| Sleeve | Instrument | Construction | History |
|---|---|---|---|
| US large equity | SPY | ETF total return | 1993– |
| US small equity | IWM | ETF total return | 2000– |
| Developed ex-US equity | EFA | ETF total return | 2001– |
| EM equity | EEM | ETF total return | 2003– |
| Govt 2y | **synthetic** | GSW zero curve | **1961–** |
| Govt 5y | **synthetic** | GSW zero curve | **1961–** |
| Govt 10y | **synthetic** | GSW zero curve | **1961–** |
| Govt 30y | **synthetic** | GSW zero curve | **1961–** |
| TIPS 10y | **synthetic** | Fed TIPS curve | 1999– |
| IG credit | LQD + reconstruction | ETF; Moody's-based extension | 2002– (1919– modelled) |
| HY credit | HYG | ETF total return | 2007– |
| Broad commodity | DBC + AQR CLR splice | ETF; AQR index pre-2006 | 2006– (1877– spliced) |
| Gold | GLD | ETF total return | 2004– |
| Dollar | DTWEXBGS | FRED trade-weighted index | 2006– |

**Headline window: 2007-04 to present** (the naive ETF-only common start, driven by HYG). Run the full model there. Then run a **long-history sensitivity** on the subset that goes back further — the four synthetic government bonds plus SPY plus commodities gets you to 1993, and dropping SPY for Ken French Mkt-RF gets you to 1961. Reporting the model's behaviour across 60 years of rate regimes on the govvie sleeve alone is a genuine differentiator that costs almost nothing once the GSW loader exists.

Limit yourself to **three splices maximum**, half a day budgeted each, each one documented as a judgment call.

## 3.2 The synthetic government bond series — the differentiator

This is the piece that separates the project from every ETF-only repo. The Fed publishes daily Svensson parameters for the fitted nominal zero-coupon curve back to 1961.

**Source:** `https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv`
(landing page: federalreserve.gov/data/nominal-yield-curve.htm — updated weekly, Tuesdays, through the prior Friday)

Columns `BETA0, BETA1, BETA2, BETA3, TAU1, TAU2` give the continuously-compounded zero yield at any maturity `n` (years):

```
y(n) = β₀
     + β₁·[1 − exp(−n/τ₁)]/(n/τ₁)
     + β₂·{[1 − exp(−n/τ₁)]/(n/τ₁) − exp(−n/τ₁)}
     + β₃·{[1 − exp(−n/τ₂)]/(n/τ₂) − exp(−n/τ₂)}          (percent p.a.)
```

Constant-maturity zero-coupon bond total return over one day, holding maturity `n` and rolling down by `Δ = 1/252`:

```
P_t(n)      = exp(−n · y_t(n)/100)
r_{t+1}(n)  = exp( n·y_t(n)/100 − (n−Δ)·y_{t+1}(n−Δ)/100 ) − 1
excess_{t+1}= r_{t+1}(n) − rf_t·Δ                    rf from DGS1MO / TB3MS
```

Three things to get right, each worth a unit test:

1. **Evaluate the curve at `n − Δ` on day t+1, not at `n`.** The roll-down is a real component of the return and getting it wrong is the classic error. At the 30y point with a 2% slope it is worth a few bp/day.
2. **`y(n)` as `n → 0`** is numerically unstable in the Svensson form — the `(1−exp(−n/τ))/(n/τ)` terms are 0/0. Use the limit (→1) below `n = 1e-6`. You will not hit this at 2y+ but the loader should be safe.
3. **The GSW file has a multi-row header and NaN blocks for short tenors in early years.** Assert on the parsed column set in a data-contract test.

**Cross-check against TLT — amended 2026-08-26.**

> ~~Correlate your synthetic 20y+ series against TLT over 2002–present. Expect correlation > 0.98 and a small negative mean difference (TLT's expense ratio plus its non-constant maturity). Report the number. If it is not there, your roll-down is wrong.~~
>
> **Superseded.** The original gate was insensitive to the quantity it was written to test. Deleting the roll-down entirely — reading the day-`t+1` curve at `n` instead of `n − Δ` — moves the daily correlation by **1e-7**. The roll is a slow drift in the mean and contributes almost nothing to daily covariance, so a correlation threshold cannot falsify it. The correlation table is still worth reporting; it is no longer the acceptance test.

**Primary gate.** Score the mean total return against a **par-coupon ladder**, not a single zero:

```
|mean(TLT) − mean(ladder)| ≤ 0.15%/yr        and        β ∈ [0.95, 1.05]
```

where the ladder holds constant-maturity par bonds spanning 20–30y (TLT tracks the ICE US Treasury 20+ Year index), and β is the slope of TLT regressed on it.

- **Why a ladder and not the 20y zero.** A 20y zero puts all its duration at one point and earns ≈ +0.42%/yr of roll-down; TLT holds coupon bonds spanning 20–30y with a regression-implied duration near 15.4y and earns ≈ +0.21%/yr. Scoring one against the other conflates a roll differential, a cash-drag artefact of β-scaling, and a key-rate mismatch. **The ladder is a validation comparand only** — the production factor inputs stay constant-maturity zeros, because a blended ladder's loading on level/slope/curvature is a mixture rather than a clean key rate, which is exactly what §4.1 needs its inputs not to be.
- **Why 0.15%/yr.** It is TLT's expense ratio. A synthetic built from the Fed's curve pays no fee, so a correctly specified replica should beat the fund by about its fee and by nothing else. A gap materially wider than the fee means the construction is wrong; a gap of the wrong sign means the fund is beating a costless replica of itself, which cannot happen.
- **The gate has power**, which is the whole point of the amendment. Flipping the sign of the roll-down moves the gap from −0.19%/yr to +0.23%/yr — **a movement of 0.42%/yr, about 2.8× the tolerance, and a change of sign.** Compare the 1e-7 the correlation gate moved by.

  *(An earlier draft of this amendment stated the flipped gap as −0.52%/yr, 3.5× the fee. That came from a ladder run which priced 2008-03-21 — Good Friday, Svensson parameters written but no yield published at any tenor — as an ordinary observation, carrying a five-calendar-day move as one day. One row in 6,009 was worth 0.08%/yr. The figures above are from the corrected run.)*

**Current status: the gate FAILS at −0.192%/yr, outside the 0.15%/yr tolerance by 0.042%/yr**, with β = 1.010 inside its band. **This is an open question, deliberately left open.**

The fund is not the source. iShares publishes TLT's tracking difference against the ICE US Treasury 20+ Year Bond Index at −0.09%/yr since inception (−0.13% 1yr, −0.09% 5yr, −0.10% 10yr) against its 0.15% expense ratio — TLT *beats* its fee via securities lending, so no ETF implementation drag is available. Chaining, `ladder − index = +0.192 − 0.09 = +0.102%/yr`: the synthetic ladder outperforms the real bond index, and that is ours to explain.

Three structural corrections were measured independently, each with its prediction registered in `experiments.md` before the run:

| Correction | Effect on the gap |
|---|---|
| Par bonds → seasoned bonds (swept over issue age 0–10y) | **+0.005%/yr** |
| Full first period → realistic coupon-date phase | −0.026%/yr |
| `Δ = 1/252` → calendar accrual | +0.033%/yr |
| **Unexplained after corrections** | **+0.114%/yr** |

The par-versus-seasoned idealisation was predicted at +0.06%/yr rising monotonically in seasoning; **it measured −0.005%/yr, non-monotone, and the prediction is refuted.** Net, the corrections widen the gap rather than closing it.

The leading remaining suspect is the curve itself: GSW fit a smooth zero curve to a filtered set of Treasuries that deliberately excludes on-the-run and first off-the-run issues, while the index prices the actual outstanding stock including exactly those. Testing that needs bond-level data this project does not have.

**The tolerance stays at the expense ratio and is not widened to absorb the residual.** A tolerance fitted to the number it is meant to test is not a test. The assertion is `xfail(strict=True)` in `tests/test_ladder.py` with the full reasoning; see `reports/gsw_validation.md` §6. This does not block the data layer — the production series are the constant-maturity zeros, and the residual is a property of the validation comparand, not of what the factor models consume.

**Reported diagnostics, not gated.** Report the correlation of daily total returns against TLT at daily, weekly, monthly and quarterly horizons, **for both the 20y zero and the ladder**. Expect roughly 0.95/0.97/0.98/0.99 for the zero and 0.96/0.98/0.99/1.00 for the ladder. The shortfall at daily frequency is observation noise, not model error: **the Fed fits its curve to end-of-day bond quotes at about 3:30pm ET while TLT prints at the 4:00pm equity close**, and TLT's close is a traded price carrying a premium or discount to NAV. Neither is removable from a curve-based series, and both average out with horizon.

Report the evidence that rules out the alternatives, because each is the more obvious explanation and each is wrong here: cross-correlation against the lagged synthetic is ≈ 0 at ±1 and ±2 days (so it is a within-day effect, not a date misalignment); excluding the 287 ex-dividend days leaves the daily correlation at 0.9483 (so it is not dividend handling); and four-year sub-periods run 0.94–0.97 (so it is not one bad regime).

Also report the mean difference against the 20y zero and its duration-neutral α — but do not gate on α. It mixes fees with the roll differential and with the cash leg that β-scaling implicitly leaves uninvested, and is not cleanly interpretable.

Same treatment for TIPS via `feds200805` (real yields, 1999–, 10y+ from 1999 and shorter tenors from ~2004). Nominal-minus-real gives you a clean breakeven-inflation factor for free.

## 3.3 Credit

FRED's ICE BofA series were truncated to a rolling 3-year window in **April 2026**. Every tutorial that says "get HY spreads back to 1996 from FRED" is now stale — verify this yourself and note it in the README, because it is current and most readers will not know.

- **Headline:** LQD from 2002, HYG from 2007. Live with it.
- **Extension (clearly labelled as a model, not data):** Moody's seasoned Aaa/Baa yields on FRED (`AAA`, `BAA`, monthly from January 1919, **not truncated** — these are Moody's, not ICE). Reconstruct excess returns as
  ```
  xs_return ≈ (OAS/12) − SD·ΔOAS − expected_loss
  ```
  with spread duration `SD` assumed by rating bucket and expected loss from historical default/recovery tables. This is defensible, cheap, very explainable, and must be labelled a reconstruction everywhere it appears.

## 3.4 Spreads and ADV — the hardest requirement

**There is no free daily historical bid-ask series for ETFs.** You must estimate. Reframe this as the project's most novel piece rather than as a shortcoming.

**ADV is easy.** Daily volume × close, **21- or 63-day median** (not mean — one index-rebalance day wrecks a mean).

**Spread: the EDGE estimator.** Ardia, Guidotti & Kroencke (2024), *JFE* 161:103916. `pip install bidask`, MIT, github.com/eguidotti/bidask. Needs only OHLC, minimum 3 observations, handles missing values, estimates the root-mean-square **effective** spread — exactly what a cost model wants. It supersedes Corwin-Schultz and Roll, both of which are badly biased.

> **Pass `sign=True` and reset negatives to zero.** The package default returns `|estimate|`, which is not an estimate of the spread but of `E|noise| = σ·√(2/π)` wherever the true spread is small — positive by construction, and growing with volatility. W1-P5 measured that quantity and misattributed it to a finite-sample property of EDGE; the correction is `experiments.md` rows 54–58. Note also that "returns negative estimates for liquid ETFs" is **not** a defect and is not what distinguishes EDGE from Roll: a correctly-signed EDGE goes negative about half the time on a genuinely tight instrument, and that incidence is the diagnostic for whether it is resolving a spread at all.

Two implementation notes that matter more than the estimator choice:

- **Feed it RAW, unadjusted OHLC.** Spread is a property of actually-traded prices. Dividend back-adjustment smears across ex-dates and contaminates the high-low range. Pull `auto_adjust=False`; use raw columns for EDGE and adjusted columns for returns.
- **Estimate on rolling 63-day windows, monthly step.** A static spread makes the cost model uselessly optimistic, because spreads blow out in exactly the crises when the risk model wants to trade most. **The time-variation is the contribution.** A chart of estimated spread against realised volatility across 2008, 2020 and 2022, with the flat-5bp assumption drawn as a horizontal line, is the single most persuasive figure in the repo.

### 3.4.1 FINDING (amended 2026-08-30, W1-P5c): EDGE supplies time variation and cannot supply level

This section previously described a hopeful architecture — EDGE as the production estimator with issuer medians as a level anchor. Measurement has replaced it with a division of labour, and the issuer medians are **load-bearing, not a cross-check**.

**EDGE at a 63-day window of daily bars has a per-window noise floor of ≈23bp, irrespective of the true spread.** Measured on independent windows at the sleeve's volatility (`experiments.md` rows 59–60). That floor:

- **exceeds every calm-period spread in this universe.** The true quoted spreads of the sleeve run roughly 0.3–5bp, all of them far inside the noise. Signal-to-noise at 63 daily bars is 0.04 at a true 1bp and 0.87 even at a true 20bp.
- **sits far below the crisis signal**, which runs 40–130bp. The widening the figure exists to show stands clear of the floor by a factor of two to six.

**The floor is variance, not bias, and this is the evidence.** Against known true spreads of 1 / 5 / 20 / 50bp the estimator returns medians of **−1.1 / 5.0 / 19.9 / 49.3bp** — unbiased at every level, including a median near zero with 50% negatives at a true 1bp, so there is no positive resolution floor and no implementation fault. What fails is precision on a single window, and precision does not improve with more windows because it is a property of one estimate.

The authors say as much. Their FAQ: *"using a few daily prices would provide estimates closer to the spread in those days but with potentially large estimation uncertainty"*, and *"the higher the frequency, the better (e.g., minute prices are preferable to hourly and daily prices)"*. **This project runs the coarsest frequency they describe.** The published guarantee is *asymptotic* unbiasedness and minimum variance *among OHLC estimators* — neither is a finite-sample precision claim.

**Consequences, binding on every later section:**

1. **The spread series' SHAPE is a production input.** Time variation is what §3.4 always claimed as the contribution and it survives measurement intact. A constant level offset cancels in a widening, which is why the crisis statistic is reported in basis points rather than as a multiple.
2. **The spread series' LEVEL is permanently not a production input. This is design, not quarantine.** Earlier sessions held the ban as a temporary measure pending an external comparand that might have licensed a scale factor. The noise-floor measurement removes that possibility: a ~23bp per-window standard deviation *irrespective of the true spread* cannot be calibrated away, because scaling multiplies signal and noise alike and the quantity being recovered is smaller than the noise around it. There is no constant, and no per-ticker constant, that turns a series with signal-to-noise 0.04 into a level. §7.2's impact coefficients take their level from the issuer-disclosed medians permanently, and a later session must not reopen this as though it were awaiting evidence.
3. **A per-month level for a single liquid ETF is not obtainable from this estimator at this frequency, by anyone.** Not a limitation of this implementation — a property of 63 daily bars.

An unresolved residual sits underneath: inverting the estimator per asset implies true spreads of ~25bp for SPY, IVV, VOO and IWM against quoted spreads near 0.4bp, while LQD and GLD invert correctly to 0bp. Six candidate mechanisms have been eliminated by measurement and three remain named and untestable without intraday consolidated-tape data. See `experiments.md` rows 59–63. It does not affect the shape and is contained by consequence 2.

**Level calibration — now the load-bearing step, not the validation flourish.** Every major issuer publishes a current 30-day median bid/ask spread per ETF (Vanguard's is at advisors.vanguard.com/investments/bidaskspread; iShares and SPDR on per-fund pages). Point-in-time only, so it enters as a **dated static file, committed, not a loader**. Fit EDGE over the trailing 30 days, compare, and report the ratio per ticker in a table. The original instruction — "if EDGE runs 1.4× the disclosed median across the sleeve, say so and scale" — **understated the problem by an order of magnitude and should not be followed as written**: the measured discrepancy is roughly 60×, it is not uniform across the sleeve, and a single scale factor would therefore be wrong for every asset. **The table's job is to SUPPLY the level per ticker, not to decide whether EDGE can.** That decision is made and is recorded above: it cannot, at this frequency and window, for any member of this sleeve. The comparison is still worth reporting — it documents the size and non-uniformity of the discrepancy, and it is the honest record of how the level was actually obtained — but it is not a gate EDGE could pass.

## 3.5 The traps, each of which is a test

| Trap | What happens | Mitigation |
|---|---|---|
| **Adjusted-close revision** | Yahoo's dividend adjustment rewrites history backwards. Two pulls on different dates give different histories. Every statistic on adjusted prices has a subtle forward-looking component. | Cache raw unadjusted OHLCV plus a separate dividend/split action table, both content-hashed. Apply adjustments yourself, deterministically, as of a stated date. **This single change is what turns a hobby repo into a credible one.** |
| **Weekly/monthly bars are wrong at Yahoo** | yfinance issue #1273: a mid-week dividend is applied to the whole week's close, inflating returns. Silent. | **Never request `interval="1wk"` or `"1mo"`.** Pull daily, aggregate yourself. Catching this in your repo is a good tell. |
| **`Adj Close` removal** | yfinance now defaults `auto_adjust=True` and drops the column; old code silently changes meaning. | Pin yfinance tightly in the lockfile; assert on the returned column set in a test. |
| **FRED revisions** | Macro series get revised. Using current values as factors is look-ahead. | Use **ALFRED** vintages (`realtime_start`/`realtime_end` in the FRED API). Check vintage coverage per series; drop or conservatively lag anything without vintages. Most free-data projects get this wrong; getting it right is cheap and visible. |
| **Calendar misalignment** | Different assets trade on different holidays. `.last()` vs `.resample('ME')` vs business-month-end give different series. | Top source of "my Sharpe is 2.4" moments. Write calendar tests in week 1. |
| **Licensing** | AQR and Ken French have no explicit redistribution licence; ICE data explicitly forbids reproduction; JST is CC BY-NC-SA. | **Never commit third-party raw data.** Download at build time, cache to a gitignored directory, commit only the manifest (URL, timestamp, SHA-256, row count, date range) and derived aggregates. |

---

# 4. The two factor models

Estimate both. The comparison is a deliverable, not an afterthought — every serious shop runs a fundamental/macro model for attribution and communication and a statistical model as a hidden-risk detector, and the reason is worth demonstrating rather than asserting.

## 4.1 Model A — macro factor model (the MAC analogue)

Time-series regression, not cross-sectional: with 15 assets and 6 factors you estimate `β` per asset by regressing asset excess returns on observed factor returns.

```
r_{i,t} = α_i + Σ_k β_{i,k} f_{k,t} + u_{i,t}
```

**Factor set (6):**

| Factor | Construction | Rationale |
|---|---|---|
| Equity | Ken French `Mkt-RF` (US). **Settled 2026-08-30**, not a standing choice — see below | The dominant risk source in any multi-asset book |
| Rates level | First PC of GSW zero-yield daily changes across 2/5/10/30, sign-fixed so positive = yields up | Duration |
| Rates slope | Second PC, sign-fixed so positive = steepening | The 2022 regime is a slope story |
| Credit | HYG excess return **over cash** — amended 2026-08-30, see below. The duration hedge is performed by the orthogonalization, not by a matched Treasury leg | Default/liquidity premium |
| Commodity | DBC excess return over cash on the daily model window; the AQR Commodities-for-the-Long-Run index is MONTHLY and enters only as the pre-2006 splice leg (§3.1) | Inflation and real-asset exposure |
| Dollar | DTWEXBGS return (sign so positive = USD strength) | The cross-asset transmission channel |

Notes on construction, each of which you should be able to defend:

- **Orthogonalize sequentially** in the order above (Gram-Schmidt on the factor return series, expanding-window). Rates slope orthogonal to level; credit orthogonal to equity and level; commodity orthogonal to dollar. Without this, credit and equity are ~0.7 correlated and the factor covariance is near-singular in exactly the crisis periods you care about. Report the pre- and post-orthogonalization correlation matrices side by side.
- **`β` estimation window**: EWMA-weighted, 252-day window, **half-life 63 days**, matching Barra's BETA descriptor. Report rolling β charts — the SPY-to-rates β flipping sign around 2021 is the single most legible chart the project will produce and it is what a portfolio implementation interviewer will want to talk about.
- **Expect low R².** Connor (1995) finds macroeconomic factor models explain 10.9% of individual equity return variance versus 42.6% for fundamental models. **But that is a cross-sectional equity number and does not transfer** — for 15 asset-class-level series, a 6-factor macro model should reach R² of 0.7–0.95 per asset, because asset-class returns essentially *are* factor returns. Report per-asset R² and flag anything below 0.5 as a candidate for removal from the sleeve or for an additional factor.

### 4.1.1 AMENDMENT (2026-08-30, W2-P1): the credit factor was double-specified, and four rulings

**The credit row above was wrong, not ambiguous, and is corrected.** As originally written it specified the
hedge twice. "HY excess return over **duration-matched Treasury**" removes the rates exposure by subtracting a
Treasury leg at an assumed duration; "orthogonalize credit ⊥ (equity, level)" removes it again by projecting
the series onto the rates level factor. Doing both hedges the same exposure twice — the orthogonalization would
be regressing an already-duration-stripped series on the level factor and removing whatever residual rates
exposure the assumed duration failed to capture, so the resulting factor is not "credit net of rates", it is
"credit net of rates, minus a second helping of rates". The two routes are alternatives, not stages.

**The orthogonalization is the route this project takes.** The credit factor is defined as **HYG total return
less the cash rate**, and the sequential Gram-Schmidt against the rates level factor performs the duration
hedge with an **estimated, expanding-window, time-varying** hedge ratio. That is strictly better than the
subtraction it replaces on three counts: it invents no duration constant (CLAUDE.md invariant 9 — no published
value for HYG's spread duration exists anywhere in this repository or its sources); the hedge ratio is allowed
to move, and HY duration genuinely moves with the coupon and the level of rates; and the hedge is applied by
the same mechanism that hedges every other factor in the set, so there is one orthogonalization story rather
than two.

The cost is stated rather than hidden: the credit factor's rates hedge is now only as good as an
expanding-window regression coefficient, it is undefined for the first `min_window` observations of HYG's
history, and the pre-orthogonalization correlation matrix will show credit loading on rates, which the
post-orthogonalization matrix must then remove. The report shows both.

**Four further rulings, settled the same day.**

1. **The rate PCs are normalized to interpretable units.** The level PC is scaled so that its **mean loading
   across the 2/5/10/30 tenors equals one basis point**; the slope PC is scaled so that its **30y loading minus
   its 2y loading equals one basis point**. Both scalings are quadratic in the eigenvector's sign and therefore
   fix the sign as a side effect, which is the deterministic sign convention §4.1 asks for. With the factor and
   the asset return both expressed in basis points, a regression coefficient on the level factor then reads
   **directly as a negative effective duration** — a 10-year zero should return ≈ −10 — which turns the
   normalization into a falsifiable claim rather than a cosmetic rescaling. See `experiments.md` rows 64-67.

2. **Equity is Ken French `Mkt-RF`, not SPY excess.** Longer history (1926 against 1993), independent of Yahoo,
   and already cross-checked against our own SPY total return at correlation 0.9786 with a mean gap of
   −0.12%/yr (`experiments.md` row 41). **Consequence for W2-P3, recorded here so it is not discovered late:
   a factor taken directly from a published series cannot then be validated against that same series.** The
   equity factor's W2-P3 validation must be against something else — the AQR Century of Factor Premia equity
   market series, or our own SPY excess return, which is now a comparand rather than an input.

3. **The commodity factor is DBC excess return over cash.** The row above named the AQR Commodities-for-the-
   Long-Run index, which is **monthly**; Model A is estimated daily, so that series cannot be the daily factor.
   §3.1 already makes DBC the instrument and AQR CLR the pre-2006 splice leg, and the factor follows the
   instrument. AQR CLR returns to this project as a monthly comparand in W2-P3 and as the long-history splice,
   both unchanged.

4. **The pre-2007 credit reconstruction stays deferred** — settled in W1-P4 and restated here so that a later
   session does not reopen it as an oversight. §3.3's `(OAS/12) − SD·ΔOAS − expected_loss` reconstruction needs
   a spread duration per rating bucket and an expected-loss term, neither of which is a published constant
   available to this repository; §12 lists it second on the cut list; and the headline window begins 2007-04,
   where HYG exists. It is not needed for Model A and it is not built.

**Estimation windows, ruled the same day and binding on the implementation.** The PCs are estimated
**expanding-window** with a minimum of 252 observations, `NaN` before that. The Gram-Schmidt is
**expanding-window** on the same minimum. Full-sample orthogonalization is look-ahead — it uses the whole
history's covariance to decide what a factor was on its first day — and a test asserts the difference rather
than trusting the code comment.

### 4.1.2 ACCEPTANCE (amended 2026-08-30, W2-P1): the pairwise bar is WITHDRAWN

W2-P1 was originally given the acceptance criterion **"no factor pair exceeds 0.15 correlation
post-orthogonalization"**, applied to all 15 pairs. It is withdrawn, and the reason is worth more than the
number: **it was wrong in kind, not in calibration.**

**Why it was invalid.** §5.1 estimates a K×K factor correlation matrix at a 504-day half-life and §5.3 runs a
Monte Carlo eigenfactor adjustment on that matrix's *eigenspectrum*. Both techniques presume off-diagonal
structure — the eigenfactor adjustment exists precisely to correct the bias in the eigenvalues of a matrix whose
factors are correlated. A gate forcing every pair under 0.15 would drive `F` toward diagonal and leave the
project's central technique with nothing to operate on. The gate would have been satisfied by destroying the
object week 3 is built to estimate. Barra's own answer is the same: USE4 orthogonalizes two pairs and leaves
the rest correlated.

**Why it was also unmeetable.** §4.1's scheme names exactly three pairings. Equity is first in the order and is
never projected off anything, so `equity`/`dollar`, `equity`/`commodity` and `equity`/`rates_level` are
whatever the market delivers. **No correct implementation of the specified scheme could have passed the bar.**
A pair the scheme does not touch cannot be a gate on the scheme.

This is the same class of amendment as §3.2's: a gate replaced for **lack of validity**, not lowered for
inconvenience. `experiments.md` carries both.

---

**Gate — the four pairs the scheme actually targets.** `credit`/`equity`, `credit`/`rates_level`,
`commodity`/`dollar`, `rates_slope`/`rates_level`. The measured before and after must be stated for all four,
and each must end **substantially below where it started, at a small absolute level**. The 0.15 threshold
survives as the level criterion, but scoped to these four pairs only — where it asks whether the
orthogonalization did its job, rather than asking the factor set to be uncorrelated.

Measured 2026-08-30 over 2007-04-02..2024-12-31, 4,146 complete dates:

| Pair | Before | After | Level criterion |
|---|---|---|---|
| `credit` / `equity` | +0.680 | **+0.067** | pass |
| `commodity` / `dollar` | −0.412 | **+0.084** | pass |
| `credit` / `rates_level` | +0.106 | **−0.068** | pass |
| `rates_slope` / `rates_level` | +0.167 | **+0.246** | does not meet it — **superseded by §4.1.3**, which brings it to +0.096 |

> **SUPERSEDED 2026-08-30 by §4.1.3.** The paragraph below is left as written because it is the diagnosis
> that produced the fix, and because the exemption it granted was real while it stood. W2-P2 acted on exactly
> the sign-change measurement it describes: the Gram-Schmidt's expanding window now opens at `sample.start`,
> the pair measures **+0.096**, and the exemption has been **discharged and removed from the config**.

**`rates_slope`/`rates_level` at +0.246 is not a failure of the project**, and is exempted by ruling on
2026-08-30. It is 0.246 between two principal components of the same yield curve, which is an entirely ordinary
entry in a factor covariance matrix; and the reason the hedge underperforms is diagnosed rather than open. The
correlation between the two normalized PCs **changes sign across the burn-in boundary** — about −0.196 over the
pre-2007 history the expanding window is mostly estimated on, about +0.124 over the model window, and by
calendar year inside it from −0.315 to +0.837. An expanding-window hedge ratio estimated mostly on the former
comes out negative and, applied to the latter, *adds* level exposure rather than removing it. That is a real
result about expanding-window estimation, not a defect, and the alternative window start is registered as a
`model-config` question for W2-P2 with its hypothesis stated before it runs.

**Reported diagnostic, not gated — the condition number of the factor correlation matrix through time**, with
the crisis windows of §7 marked, computed at the same 504-day correlation half-life §5.1 will use. This is the
number week 3 actually cares about and pairwise correlation is not. It is the direct measurement of the failure
mode this project ranks second: *"the covariance matrix goes near-singular in exactly the crises that matter"*.
Report the raw and orthogonalized series side by side — the orthogonalization's real justification is that it
improves conditioning, and that claim should be measured rather than asserted. State explicitly what §5.3's
eigenfactor adjustment exists to handle: a high condition number means the smallest eigenvalues of `F` are the
most poorly estimated, an optimizer will load onto exactly those directions, and the resulting risk
understatement is what the Monte Carlo adjustment corrects. It is reported and never gated, because a condition
number is a property of the market in that window, not a property of the implementation.

### 4.1.3 RULING (2026-08-30, W2-P2): the orthogonalization burns in inside the sample

§4.1.1 settled that the PCs and the Gram-Schmidt are both estimated expanding-window on a 252-observation
minimum. It did not say **where the expanding window opens**, and W2-P1 opened both at the beginning of
available history — which for the rate PCs is 1961. That is now split: the PCA is unchanged, and the
**Gram-Schmidt's window opens at `sample.start`**.

**This is a design decision justified by a measurement already in hand, not a selection between outcomes.**
W2-P1 measured the correlation between the two normalized rate PCs at about **−0.196 over the pre-2007
burn-in** and about **+0.124 over the model window** — it changes sign across the boundary. An
expanding-window hedge ratio fitted mostly on the former comes out negative and, applied to the latter,
*adds* level exposure to the slope factor instead of removing it. Estimating a hedge on a period where the
relationship carries the opposite sign is wrong regardless of the correlation it happens to produce.

The test that makes this a principle rather than a search: **it would have been made even if it made the
correlation worse.** `experiments.md` row 69 records that the decision was taken before the resulting
number was known, and that the alternative was deliberately not run as a comparison.

**Why only the Gram-Schmidt moves.** Moving the PCA's window too would push `rates_level` and `rates_slope`
to a ~2008-04 start, and then push every factor hedged against them a further 252 days out to ~2009-04 —
turning two ragged edges into three. As implemented the cost lands exactly on the edge `credit` already
had: `rates_slope` and `commodity` now begin ~2008-04 against `credit`'s 2008-04-14, and **the number of
dates on which all six factors exist is unchanged at 4,146**. The change cost the panel nothing, because
credit's burn-in already dominated.

Nothing new is invented. The minimum is §4.1.1's 252, unchanged, and the start is a pointer to
`sample.start`. The alternative is named rather than deleted in `config/model.yaml`
(`expanding_window_start: sample.start | full_history`), so reverting is a config edit that owes a row.

**Measured 2026-08-30 over the same window as the §4.1.2 table:**

| Pair | Before | After (W2-P1) | After (W2-P2) | Level criterion |
|---|---|---|---|---|
| `credit` / `equity` | +0.680 | +0.067 | **+0.067** | pass |
| `commodity` / `dollar` | −0.412 | +0.084 | **+0.085** | pass |
| `credit` / `rates_level` | +0.106 | −0.068 | **−0.068** | pass |
| `rates_slope` / `rates_level` | +0.167 | +0.246 | **+0.096** | **now passes** |

The hypothesis registered in row 69 holds and its decision rule is satisfied: the conditioning of the
orthogonalized factor correlation matrix does not deteriorate — its whole-sample median falls from 4.88 to
**4.43** and its maximum from 6.65 to **6.40**.

**Consequence: the §4.1.2 exemption is discharged.** `rates_slope`/`rates_level` was exempted by name, with
its reason and the follow-up it owed. The follow-up has run and the pair now meets the criterion unaided, so
the exemption is **removed from the config rather than kept as a spare** — an exemption that is no longer
needed and is left in place is a standing licence that would silently excuse the same pair if a later change
made it fail again. The record survives here and in `experiments.md`; the licence does not. A test asserts
both that the list is empty and that the pair passes on its own, so emptying it cannot succeed by accident.

### 4.1.4 The exposure half of §4.1 (W2-P2)

`β` is estimated as §4.1 specifies — EWMA-weighted, 252-day window, 63-day half-life — one regression per
asset per date on all six orthogonalized factors, with the intercept `α_i` that §4.1's equation carries.
Three points are settled here because they are what makes the output readable:

1. **Both sides of every regression are in basis points.** The rate factors already are; asset returns and
   the four return factors are scaled by 1e4. Return-factor betas then come out dimensionless and rate
   betas come out in **years**, reading directly as minus an effective duration.
2. **A 252-day window at a 63-day half-life is not 252 observations.** Kish's effective sample size
   `(Σw)²/Σw²` is **160.5**, and that is what every standard error and degrees-of-freedom correction uses.
3. **`α` is a deliverable, not a nuisance term.** It is reported per asset and flagged when persistently
   distinguishable from zero, because that is the signal W2-P3's validation and §4.3's hybrid residual-PC
   model exist to detect.

**Duration recovery is the acceptance test**, against durations this project knows independently: the four
synthetic zeros from their own maturities (a zero's duration is its maturity, by definition), TLT from
W1-P3's regression against the GSW ladder, HYG from its registered analytical range. Measured 2026-08-30:
five of six inside their bands at **−1.66, −5.61, −11.29, −27.46** for 2/5/10/30y and **−15.97** for TLT.
HYG is refuted for the second time and its band is not widened; see `experiments.md` rows 73-78.

**The synthetic zeros' intercepts are a term premium, and the decomposition is exact.** SPEC.md 3.2's
carry + roll-down + duration-effect split is additive in log space, and OLS is linear in the dependent
variable, so regressing each leg on the same design gives intercepts that sum to the total's. At 30 years:
carry−cash **+2.31**, roll **+0.20**, duration **−1.69**, convexity **+3.89**, summing to **+4.71** = α(excess).

**Convexity is a named, sign-definite, positive component and must not be read as error.** It is
`expm1(x) − x` for the log return `x`, non-negative for every real `x`; measured at 30 years it is +3.89%/yr
with zero negative observations in 4,435. It is absent from the duration leg by construction, because
`duration_effect` is exactly linear in the yield change.

**The 30-year duration leg's −1.69%/yr is the two-PC span limitation and nothing else.** Regressed on the
four raw yield changes instead of the two PCs, the same leg fits at R² = 1.000000 with an intercept of
−0.0000%/yr and a coefficient of −29.987 on Δy(30). Level and slope span 97.7% of that tenor's variance; the
unspanned 2.3% has a non-zero sample mean because the curve reshaped over 2009-2024, and thirty years of
duration multiplies it up. **It is an attributed component of a two-factor curve model — the third curve
mode — not a residual, and the residual stopping rule does not apply to it.** Removing it means adding a
curvature factor, which is a specification decision owing a `model-config` row.

**One expectation in §4.1 is refuted and is recorded rather than quietly dropped.** §4.1 calls the
"SPY-to-rates β flipping sign around 2021" the most legible chart the project will produce. It is not in
the panel's exposures and cannot be: those are **partial** coefficients, the equity factor is Ken French
`Mkt-RF`, and SPY correlates 0.9786 with it, so the equity factor absorbs SPY's variance and its residual
rates loading is ≈0 on every date. The flip is unambiguous in the **univariate** beta — calendar-year means
of **+17.9 in 2020** against **−0.1 in 2022** — and `reports/rolling_betas.png` draws both lines, because
the gap between them is a worked example of what a risk model's exposures do and do not tell you.

## 4.2 Model B — statistical factor model

PCA on the 15×15 correlation matrix of daily excess returns, with **Marchenko-Pastur denoising** before extraction.

```
λ₊ = σ²(1 + √(N/T))²                    noise-band upper edge
```

Fit `σ²` to the empirical eigenvalue spectrum rather than assuming `σ² = 1` (López de Prado, *Machine Learning for Asset Managers* ch. 2; KDE bandwidth 0.15). Replace all eigenvalues below `λ₊` with their **constant average**, preserving trace. Keep the components above `λ₊` as factors; expect 3–5 to survive for a 15-asset sleeve.

Then also compute, for the head-to-head:

- **Detoned** variant (remove the first eigenvector — the "market" — before denoising).
- **Ledoit-Wolf** shrinkage as the standard benchmark to beat. **Note carefully: `sklearn.covariance.LedoitWolf` implements the identity target, not the constant-correlation target from the 2004 JPM paper.** These are different estimators. If you want the constant-correlation version you must implement it. Getting this right, and saying so in the README, is a small detail that a knowledgeable reader will notice.
- **OAS** (Chen et al.) as a third point of comparison.

### 4.2.1 RULING (2026-08-31, W3-P5): the input is the equally-weighted SAMPLE correlation, on an expanding window

§4.2 says "the correlation matrix of daily excess returns" and stops there. It does not say whether that is the sample correlation or the EWMA correlation of §5.1, nor whether the window is expanding or full-sample. Both are closed here, by the operator, before any component was extracted.

**The estimator is the equally-weighted sample correlation. Not the EWMA correlation.** Marchenko-Pastur is derived for `T` equally-weighted iid observations, and `λ₊ = σ²(1 + √(N/T))²` takes that `T` directly. Under EWMA weights the edge would need an effective sample size in place of `T` — and W3-P3b already measured that `T_eff = 2τ/ln2` **does not transfer** to a normalised correlation estimator: it is off by 1.31×, and the discrepancy is not constant, so there is no correction factor to substitute either. Applying MP to an EWMA correlation would be using a published formula outside its derivation with a `T` that is known to be wrong. `config/model.yaml` carries `factors.statistical.estimator: "sample_correlation"` and the loader rejects any other value rather than supporting one.

**The window is expanding.** A full-sample eigendecomposition uses the whole history's correlation to decide what the factor was on its first day, which is look-ahead; §4.1.3 ruled that out for the rates PCA and the same reading governs here.

**The burn-in is the rates PCA's number, reused — and the direction of the reuse is the whole point.** §5.1.2 forbids importing `expanding_min_window: 252` into a **risk-side** estimator: `risk/` has to stay asset-class agnostic (invariant 10), a day count is not, and a burn-in there must be expressed as a bound on `K/T_eff`. Model B lives in `factors/`, where the parameter already exists and means the same thing — a minimum before an expanding-window PCA is trusted. **Reuse within `factors/` is not the violation; importing into `risk/` is.** `config/model.yaml` therefore carries no `factors.statistical.expanding_min_window` key at all: the config layer copies the rates PCA's value across and **rejects** a second key rather than ignoring one, because an ignored key is how two numbers for one concept drift apart. A later session must not read this as a precedent for the other direction.

**`N/T` is emitted on every date, beside the fitted `λ₊`.** `N/T` is the Marchenko-Pastur parameter, so this is the same governing ratio as the project's headline result, appearing for the third time — and the first time inside a published formula rather than inferred from one. It belongs in `reports/stage_k_dependence.md` and in W8-P1 for that reason.

### 4.2.2 RULING (2026-08-31, W3-P5): `N` is 13, and §4.2's "15×15" and "3–5" are VOID rather than adjustable

§4.2 says "PCA on the 15×15 correlation matrix" and "expect 3–5 to survive for a 15-asset sleeve". The frozen universe in `config/universe.yaml` has **fourteen** lines, and one of them — `dollar`, the FRED trade-weighted index — is an index level rather than a tradable total return and is excluded from the investable set, which `mafrm.factors.betas` has enforced since W2-P2. **The panel is 13 wide.**

The count is not rescaled from 15 to 13. **A survivor count written for a 15-asset sleeve says nothing about a 13-asset one**, because the count depends on `λ₊`, which depends on `N/T`, which the sleeve size does not determine on its own. The expectation must be **re-derived from `λ₊` at the actual `N` and `T`** or not stated. This is the **sixth** number in the project written for a configuration it turned out not to be in, after the 0.15 correlation bar, the 0.3 comparand bar, zero-PSD-firings, the λ curve's 1.5→0.95, and Aug 2007 — and §4.2.4 adds the **seventh** (the KDE bandwidth) and the **eighth** (detoning "the first eigenvector — the 'market'") from inside this same section.

**And the count that replaces it is a threshold crossing with a thin margin, which must be reported with it.** On this panel `K = 3` on all 4,170 dates for the denoised variant — but the second eigenvalue clears `λ₊` by at least 38% while the **third clears it by under 5% on 59.8% of dates and by 0.6% at its tightest**. The count is stable in the sense that it does not flicker; it is *not* stable in the sense of a clean separation, and "K = 3 on every date" must not be reported as though it were. For the detoned variant the same near-tie becomes an actual change in the count: the third eigenvalue crosses below `λ₊` on 27 dates. `experiments.md` row 128, logged as **not pre-registered** — it was found while redrawing the figure, after the counts were known.

### 4.2.3 RULING (2026-08-31, W3-P5): eigenvector signs, and what the flip count is not

An eigenvector's sign is arbitrary. Run once per date on a window that grew by one row, `numpy.linalg.eigh` can return `-v` where it returned `+v` yesterday — its sign is deterministic for a given matrix but is **not continuous** in it, and LAPACK guarantees nothing about two nearly-identical inputs. Left alone, that puts a spurious `-2f` into the factor's return series on the affected dates.

**Each date's loadings are oriented against the previous date's**; the first is oriented so that its largest-magnitude loading is positive. The seed convention is chosen for having no near-zero denominator: the largest absolute element of a unit vector in `Rᴺ` is at least `1/√N`, so it cannot be the coin flip that a "positive sum" convention would be for a component whose loadings cancel.

**Two statistics are reported and only one of them means anything.** The *flip count* — how often `eigh`'s output disagreed with the accumulated orientation — is **path-dependent and not interpretable on its own**: one genuine discontinuity leaves the convention negated relative to LAPACK's for every date after it, so the count measures how long the two have been out of step rather than how often something went wrong. It is carried as provenance and must never be read as a per-date instability rate. The statistic that can actually go wrong is the *alignment* after orientation, `|v_prev · v|`, which sits at 1 while a component keeps its identity and falls toward 0 where two eigenvalues cross and the components swap. **A sign convention cannot fix a crossing and must not hide one**, so the low-alignment dates are counted and reported: the factor series has a genuine discontinuity there, and a reading of a single date's loadings near one is a reading of two different components.

### 4.2.4 FINDING (2026-08-31, W3-P5): the `σ²` fit does not identify at this `N/T`, and detoning by position is not detoning by identity

Two things in §4.2 turn out to be written for a configuration this project is not in. Both are reported rather than worked around, and in neither case was the shipped configuration changed.

**First: the `σ²` fit returns `σ² = 1` on every date — the assumption §4.2 tells you not to make.** §4.2 says to fit `σ²` to the empirical spectrum *"rather than assuming `σ² = 1`"*, citing López de Prado ch. 2 and its KDE bandwidth of 0.15. On this panel the fit converges to the upper bound of its own derived range on all 4,170 dates, and `λ₊` equals `(1 + √(N/T))²` exactly. The fitted parameter contributes nothing.

The mechanism is arithmetic. The objective is the squared error between a Gaussian KDE of the eigenvalues and the MP density, both on a grid spanning the candidate band. The MP density carries unit mass over a band of width `4σ²√(N/T)`, so its **height** is of order `1/(4√(N/T))` — about 4.6 at the full window. The KDE of thirteen eigenvalues spread over a range of about five has a height of order 0.2. They differ by more than an order of magnitude everywhere on the band, so the objective is dominated by the theoretical density's own magnitude and is **monotone decreasing in `σ²`**: the fit minimises it by making the band as wide as it is allowed to be. Equivalently, and this is the version to remember: **at `N/T = 0.0029` the noise band is 0.22 wide and the kernel that is supposed to resolve it is 0.15.** López de Prado's examples run at `N/T ≈ 0.1`, where the band is 1.26 wide and the kernel is 12% of it; `experiments.md` row 122 confirms the same code identifies `σ²` there. **The bandwidth is the seventh number written for a configuration this project is not in**, on §4.2.2's count — but read §4.2.6 before concluding anything should be done about it. The constant is out of regime; the answer it produces is nevertheless the right one, and the two statements are not in tension.

**The shipped `kde_bandwidth` is still 0.15 and Model B's `K` does not depend on it.** Row 120 swept the bandwidth from 0.01 to 1.00 as a control — with the shipped value fixed in advance regardless of the outcome — and the surviving count is 3 at every value. The row's pre-registered prediction that `σ²` would pin at *every* bandwidth is **refuted**: it leaves the bound at 0.01. That refutation is recorded rather than the prediction being softened, and the reason it does not change the model is that the falsifier's stated consequence — "Model B's `K` depends on it" — is measured false.

**Second: §4.2 specifies detoning by POSITION and justifies it by IDENTITY, and in a multi-asset sleeve those are different objects.** §4.2 says to remove "the first eigenvector — the 'market'". In this panel the first eigenvector is **not** a market factor: it loads equities at about −0.32 against government bonds at about +0.31, which is the risk-on/risk-off opposition, and the all-positive common-level mode is the **second** eigenvector. Detoning as specified therefore removes the opposition and promotes the market mode to first. The consequence is that the detoned variant is a genuinely different three-factor set — market, real assets, curve slope — rather than the denoised model with one component deleted, which is what "detoned" is usually taken to mean and is what a reader would otherwise assume. It is carried as specified; what changes is the description.

**One further divergence from the cited source, followed deliberately.** López de Prado ch. 2 detones the **already-denoised** correlation; §4.2 detones **before** denoising, so the `σ²` fit sees the deflated spectrum. This project follows §4.2. The deflated matrix is singular by construction, and denoising is what removes the singularity: a zero eigenvalue is below `λ₊` and is absorbed into the constant average with the rest of the noise.

**And one step §4.2 leaves implicit, made explicit — it is not cosmetic, and it is kept off the factor path.** Replacing eigenvalues moves the diagonal off 1, so the result is not a correlation matrix until it is rescaled, and it must remain one where it is recombined with volatilities. The rescale preserves the trace as well (a unit diagonal on `N` variables has trace `N`), so §4.2's constraint holds either way, and ch. 2 applies the same rescale.

But it is a real transformation, not a rounding tidy-up: the pre-rescale diagonal deviation runs to a **median of 0.23 and a maximum of 0.53**, it turns the leading directions by up to `1 − |cos| = 0.10`, and it moves the survivor count across `λ₊` on **528 of 4,170 dates for the detoned variant** (none for the denoised one, where there is no deflation zero for it to act on). **The factors are therefore extracted from the PRE-rescale eigenvectors**, which are exactly the input correlation's — constant-average replacement changes `Λ` and leaves `V` alone, so **denoising in the form §4.2 specifies rotates nothing and is purely a rule for how many components to keep**. Counting survivors on one spectrum and taking directions from another would be neither faithful nor self-consistent, so the two objects are kept apart: the rescaled matrix is what the Ledoit-Wolf and OAS comparison uses, where a correlation matrix is what is needed, and `survivors_after_rescale` reports where the two diverge.

### 4.2.5 RULING (2026-08-31, W3-P5): Ledoit-Wolf and OAS are COMPARANDS, not candidates

§4.2 names Ledoit-Wolf as "the standard benchmark to beat" and OAS as a third point of comparison. **They are comparands, not candidates, and the head-to-head is therefore `data-diagnostic` and adds nothing to the deflated-Sharpe trial count.** The reason is that the specification already chose: §4.2 names MP-denoised PCA as Model B. The comparands exist to characterise where that estimator sits, and reporting where it sits cannot reverse it. Denoised versus detoned is the one real fork in §4.2 and it gets the treatment §5.3 gives `a = 1.0` and `a = 1.4` — carry both, report both, pick neither.

**It came out against the specified estimator, and §4.2.7 is what that means.** Out-of-sample minimum-variance volatility is 0.90%/yr for the plain sample correlation, 0.93% for constant-correlation Ledoit-Wolf and **1.41% for MP-denoising**.

**Pre-registered before any comparand number existed (`experiments.md` row 126): if Ledoit-Wolf or OAS outperforms MP-denoising on any metric, that is a reported finding about the estimators and NOT a licence to change Model B.** Switching on a comparand's measured performance would be exactly the after-the-fact reclassification §6.6's category rules forbid. **Standing note: a later session that selects one of these on measured performance owes the `model-config` row, and owes it at the moment of selection rather than at the end.**

**One implementation trap, which §4.2 already flags and which is asserted rather than commented.** `sklearn.covariance.LedoitWolf` implements the **identity** target, not the constant-correlation target of Ledoit & Wolf (2004, *JPM*). They are different estimators with different optimal intensities. The constant-correlation version is implemented by hand in `mafrm.factors.statistical.ledoit_wolf_constant_correlation`, and a test asserts the shrunk matrix stays closer to the sample correlation than to the identity — which is the property that distinguishes the two targets. `sklearn` is not a dependency of this project in any case.

### 4.2.6 AMENDMENT (2026-08-31, W3-P5): out of regime, `σ² = 1` is the CORRECT answer, not a lazy assumption

§4.2 says to fit `σ²` *"rather than assuming `σ² = 1`"*. §4.2.4 records that on this panel the fit returns `σ² = 1` anyway. **This section amends how that is to be read, because "the specified procedure returns the thing the spec warns against" invites exactly one wrong inference — that the fit is broken and should be repaired by moving the bandwidth — and it is not.**

**The warning in §4.2 is a warning for the in-regime case.** Where `N/T` is large the noise band is wide, a real spread of eigenvalues sits inside it, and assuming `σ² = 1` throws away a measurable quantity: the fit at `N/T = 0.1` returns 0.78 and recovers the planted factor count exactly (`experiments.md` row 122). There, assuming 1 is the lazy answer §4.2 is right to forbid.

**Out of regime it is the correct answer.** At `N/T = 0.0029` the band is 0.22 wide and there is essentially nothing inside it to estimate. A correlation matrix has trace `N`, so its average eigenvalue is exactly 1; with no resolvable noise distribution to fit, the noise variance *is* 1, and the estimator returning its bound is the estimator saying so. The degenerate objective is not the estimator failing to find an answer — it is the estimator finding that no answer inside the band is distinguishable from any other.

**`kde_bandwidth` therefore stays at 0.15 and the sweep is not a reason to move it.** `experiments.md` row 120 measured that `σ²` leaves the bound at a bandwidth of 0.01 — and that **`K = 3` at every bandwidth from 0.01 to 1.00**. The invariance of `K` is the evidence, not a lucky escape: if the kernel width were the problem, the factor count would move with it. What a bandwidth of 0.01 buys is `σ²` moving *by fitting noise inside a band that should not be fitted at all* — thirteen eigenvalues resolved as thirteen spikes, which is a description of the sample rather than an estimate of a noise distribution. **No kernel can resolve a band 0.22 wide from thirteen points, and one that appears to has stopped estimating the thing the band is for.**

A constant quoted from a source moves by a ruling with a reversing result named in advance. None was named here, none exists, and the sweep that could have suggested otherwise measured the opposite.

### 4.2.7 FINDING (2026-08-31, W3-P5): denoising is worse than not denoising here, and that is a statement about `N/T`

The head-to-head §4.2.5 sets up came out against the specified estimator. Out-of-sample minimum-variance realised volatility, same expanding window, same volatilities, only the correlation estimator differing:

| estimator | min-var realised volatility (%/yr) |
|---|---|
| sample correlation | **0.8964** |
| Ledoit-Wolf, constant correlation | 0.9312 |
| OAS | 1.1726 |
| **MP-denoised (Model B)** | **1.4088** |
| MP-detoned | 2.5355 |

**This is a result about the method's regime and must not be reported as Model B being a bad model.** Marchenko-Pastur denoising is a **high-dimensional** technique. It exists to separate signal from sampling noise in a correlation matrix carrying too many parameters for its sample, and its value is proportional to how much such noise there is to separate.

At `N/T = 0.0029` there is essentially none. Thirteen series from about 4,470 observations give a sample correlation that is already well estimated; `λ₊` is 1.11, the band is 0.22 wide, and there is almost nothing inside it. **So denoising cannot help, and it hurts, because averaging ten eigenvalues that were never noise destroys real structure** — specifically the small-eigenvalue directions a minimum-variance portfolio is built out of. The 1.41% against 0.90% is that sentence measured.

**The control settles that this is the regime and not the code.** At `N/T = 0.4` (`experiments.md` row 127, registered before it ran) the ordering flips: MP-denoising beats the sample correlation by 6.1%, beats both shrinkage comparands, `σ²` identifies at 0.62 rather than pinning, and the survivor count recovers exactly the planted factors. Two points either side of the sign change, both measured.

**It is the fourth published technique in this project whose value depends on `N` or `K`, and the first that is actively harmful rather than merely inert.** §5.2's PSD repair is a no-op at `K = 6`; §5.3's parabola has three residual degrees of freedom and smooths almost nothing; §5.2's Newey-West pays variance to estimate lag terms that are zero in expectation. Those three cost approximately nothing. This one takes a well-estimated matrix and makes it measurably worse. `reports/stage_k_dependence.md` carries all four side by side, and that distinction is why they are together.

**Week 7 is the test, and it is registered now** (`experiments.md`, registered-not-counted, W3-P5). The cross-sectional equity module runs about 500 names against the same history, so `N/T ≈ 0.11` — in regime, and between the two points already measured. The registered prediction is that MP-denoising **underperforms** the sample covariance at `N/T = 0.0029` and **outperforms** it at `N/T ≈ 0.11`. Falsified if it fails to outperform there, in which case the regime explanation is wrong and the underperformance measured here needs another cause. Registered while the equity panel does not exist and the direction is measured at two points either side, which is what makes it worth anything.

## 4.3 The hybrid

Run PCA on the **residuals** of Model A and append the top 1–2 residual PCs as unnamed factors. This is the honest best-of-both: it recovers systematic risk your named factor set missed, it degrades gracefully, and it gives you a hidden-risk detector for free. Report Model A's R² alone and with the residual PCs.

If a residual PC ever loads heavily and its variance spikes, that *is* the crowding/deleveraging signal that no pre-specified factor set contains — the August 2007 quant-quake case. Charting the residual PC's variance share through 2008, 2020 and 2022 is a strong figure.

### 4.3.1 RULINGS (2026-08-31, W3-P6): row 83's four thresholds, fixed before the residual panel existed

`experiments.md` row 83 was registered on 2026-08-30, five sessions before this section was implemented and before any residual PC had been extracted. Its two clauses — "loads **materially** on gold" and "**correlates** with the TIPS real-yield series" — carried no thresholds, and neither did SPEC.md, `config/model.yaml`, or the row itself. **They were ruled by the operator on 2026-08-31, in the session brief, before the residual panel was computed.** The ordering is the whole point: a threshold chosen once the loadings are within reach is a threshold chosen to produce an outcome.

**Every one is rank- or comparison-based, so none of them is a magnitude anyone selected.**

1. **"Loads materially on gold" = `gold` holds the LARGEST ABSOLUTE LOADING on the component.** Scale-free, needs no constant, and has a computable null of `1/N` per component. `gold` is named for reasons fixed in writing beforehand: W2-P2 measured it as the **only** asset whose R² is persistently below §4.1's 0.5 line (median 0.392), and `config/universe.yaml` stated on 2026-08-25 that "gold's factor loadings are real-rate and dollar rather than growth". The full loading vector and gold's rank are reported **whatever the outcome**, so a near-miss stays visible.

2. **"Correlates with TIPS" is a PLACEBO, not a significance bar.** The criterion is `abs(rho(PC, d.real 10y)) > abs(rho(PC, d.nominal 10y))`. At `n ≈ 3,600` a two-sided 5% test passes at `abs(rho) > 0.033`, which has no teeth, and the 0.3 bar §6.5.2 **withdrew** is not reimported. The six-factor set already contains a nominal level factor, so a residual PC that is merely more nominal-rates exposure fails this while a genuine real-rate factor cannot. Same family as W2-P3's comparand placebo, and it needs no constant at all. Null rate 1/2.

3. **The comparand is the daily CHANGE in the 10-year real yield, in basis points — not the level.** Correlating a daily return series against a yield level is a spurious-regression setup. Both legs are built by `macro.curve_yield_changes_bps` at the same tenor, so they are masked and differenced identically and the comparison is between two curves rather than between two constructions. **The tenor is not a key**: it is the single entry of `data.real_curve.maturities_years`, because §3.1 puts exactly one real point in the frozen universe.

4. **A PC's sign is unidentified, so the sign clause is stated sign-invariantly.** Each date's component is oriented so that `gold`'s loading is positive; the correlation with the real-yield change must then be **negative** — gold rises when real yields fall. Fixed in code and asserted in tests, not left in prose.

**This is a TEST orientation and it is not the factor series' orientation.** The series handed downstream uses §4.2.3's continuity convention — aligned against the previous date, seeded on the largest-magnitude loading — because a factor series must not flip on a date when one loading crosses zero. The share of dates on which the two agree is **reported**: a low share would mean gold's loading is not stably signed and the test orientation is close to a coin flip. Measured at **90.8%** and **95.2%** for the two components.

`ResidualPcPanel.test_factors` is **derived** from `factors` and `loadings` rather than stored, so that negating both — which is exactly what a different LAPACK eigenvector sign produces — provably cancels. `tests/test_hybrid.py` asserts that it does, and asserts the paired control that negating the DATA does change the verdict, so the invariance is not vacuous.

### 4.3.2 RULING (2026-08-31, W3-P6): "the top 1–2" is a range over MODELS, and both are carried

§4.3 says "the top 1–2 residual PCs". That is a range, and it is **not** the same kind of range as the eigenfactor's 1000–3000 Monte Carlo trials, where any value inside changes only simulation noise. One residual PC and two are **different models**: they append different factor sets and they change what "at least one residual PC loads on gold" can mean.

**So both are built, both are reported, and neither is selected.** Third application of the precedent that carries `a = 1.0` alongside `a = 1.4` (§5.3) and `denoised` alongside `detoned` (§4.2). `config/model.yaml`'s `factors.hybrid.component_counts` is `[1, 2]` and the parser **rejects** any other value: dropping one is choosing between them, which is a `model-config` trial and owes a counted row in `experiments.md`.

**Row 83 is reported separately for the one-PC and two-PC readings, with the null rate stated for each.** Gold ranking first is `1/13` on PC1 alone and, across two components, `2/13` as a union bound or `25/169` under independence — and the components are *not* independent, because eigenvectors of one matrix are orthogonal. "At least one of two passes" is a materially weaker claim than "the first one does" and the report keeps them apart rather than quoting the stronger-sounding one.

### 4.3.3 RULING (2026-08-31, W3-P6): the PCA is on the residual CORRELATION, and there is no switch

The residuals are in basis points per day and their cross-asset volatilities span an order of magnitude — `hy_credit` at 0.9 bp/day against `gold` at 73.6. **A covariance PCA on that panel returns whichever asset is most volatile as its leading eigenvector and calls it a factor.** That is `experiments.md` row 96's units artefact arriving a second time, and it is the same reasoning that made the eigenfactor stage scale-invariant (§5.3.1).

`tests/test_hybrid.py` asserts the rejected alternative **actually fails**, on a synthetic panel that plants one common driver in four assets at two very different scales: in correlation space the quiet pair carries loadings above 0.3, in covariance space below 0.1. The ruling is a measurement rather than a preference.

**The window is expanding, and the burn-in is not a new number.** Read from `factors.macro.rates.pca.expanding_min_window`, exactly as §4.2.1 has Model B read it; a key of that name under `factors.hybrid` is **rejected** by the parser rather than ignored. A full-sample eigendecomposition uses the whole history to decide what the factor was on its first day, which §4.1.3 ruled out for the rates PCA. §5.1.2's prohibition is on importing a day count into `risk/`, and that direction stays closed.

**Nothing is denoised.** §4.3 asks for a PCA and does not ask for Marchenko-Pastur, and §4.2.7 measured what MP denoising does at this panel's `N/T = 0.0029`: it loses to the sample correlation by 57%, because averaging eigenvalues that were never noise destroys real structure. Importing it here would import a technique this project has already measured as harmful in this regime, and it would owe a counted row to do it.

### 4.3.4 CONTROL (2026-08-31, W3-P6): the four government zeros enter on their duration leg alone

W2-P3 fixed this on 2026-08-30, before any residual PC existed, and it is recorded here because it is the reason row 83 is readable by its own registered criterion. Row 83's "what would make this uninformative" field names it exactly: *"If the hybrid is fitted on residuals that still contain the carry-and-roll term premium of rows 79-80, a residual PC will load on the government sleeve for a reason that has nothing to do with real rates."*

**For universe members constructed as `synthetic_gsw_zero_curve`, the regressand is `duration_effect` from §3.2's exact three-way decomposition — not the total excess return.** Rows 79-82: a constant-maturity zero's carry and roll-down cannot be spanned by a factor set built from yield *changes*, so they land in the intercept at +0.71, +2.00, +3.29 and +4.71 %/yr at 2/5/10/30. Row 80 measured that the substitution collapses them to +0.00, +0.08, +0.20 and −1.69.

**`tips_10y` is deliberately NOT in that set.** The registered control names the four *government* zeros; `tips_10y`'s alpha was never flagged, so there is no measured term premium in it to strip, and widening a pre-registered control after the fact is the move the control exists to prevent.

### 4.3.5 FINDING (2026-08-31, W3-P6): row 83 is REFUTED, and the leading residual PC is the third curve mode

**Row 83's clause (a) fails on both components. `gold` ranks 11th of 13**, at mean absolute loadings of 0.0106 and 0.0237 against leading loadings of 0.5563 and 0.5479 — a factor of fifty, not a rounding. Clauses (b) and (c) both hold on both components, but the hypothesis is a conjunction and its falsifier is explicit that failing either branch refutes it. **Reported as registered. No third rate factor was added and no criterion was revised.**

**The registered falsifier's second half does not hold, and that is the finding the registration did not anticipate.** It reads: *"the residual PCs are noise and the hybrid adds nothing over the macro core."* They are not noise. The first residual PC is the **third mode of the yield curve** — the curvature a two-PC rates factor set cannot span — established by three controls each registered with its falsifier before it ran:

- **C1**: every government residual is better explained by the four raw yield changes (0.632–0.834) than every non-government one (largest 0.031, `tips_10y`). A factor of twenty.
- **C2**: `residual_pc1` regresses on the four raw yield changes at **R² = 0.6247** and on the model's own two rate factors at **0.0002** — the second number is the residual construction checking itself. Its correlation with the 2−2×10+30 butterfly is **+0.627**.
- **C3**: dropping the two identity assets (N = 11) leaves the verdict unchanged, so the refutation is not an artefact of two columns that were never observations.

**So the honest summary is neither of row 83's two branches.** The hybrid does detect systematic risk the named factor set missed; what it detected already has a name, and rows 81-82 closed it as an attributed component rather than a residual. A third rate factor would remove it. **That is a specification decision owing its own `model-config` row and it is not taken here** — adding one to rescue a refuted pre-registration is precisely the move the pre-registration exists to prevent.

**In basis points the ordering reverses, and both numbers are reported.** `residual_pc1` holds 22.4% of the correlation trace and **1.9%** of actual residual variance: it is four government residuals of 1.6–12.2 bp/day moving in near-lockstep. `residual_pc2` — an equity dispersion component, small against large — holds less of the trace and **15.1%** of the variance. Quoting either alone misleads, and this is **not** an argument for a covariance PCA, which would have returned `gold` and `em_equity` for being the most volatile (§4.3.3).

### 4.3.6 FINDING (2026-08-31, W3-P6): 2008 is not reachable, and this is the second time

§4.3 asks for the residual PC's variance share charted "through 2008, 2020 and 2022", and names the **August 2007** quant quake as the motivating case. **The residual PC series opens 2010-04-27.** Three burn-ins stack: the credit factor's expanding window, the 252-day beta regression, and the residual PCA's own 252-day expanding window. 2020 and 2022 are covered; 2008 is not, and August 2007 is further out of reach still.

The chart states the gap rather than opening in 2010 and letting a reader assume the years were quiet. **This is the second time a SPEC.md figure has asked for a date this project's burn-in cannot supply**, after §5.4.3's Aug 2007. Those two are a narrower family than §5.4.3's count of **five** gates calibrated for a case this project is not in — the other three are thresholds rather than dates — and both counts are correct at their own scope. The pattern here is the specific one: **a specification written before the burn-in was known will name crises the model cannot see**, and this build's burn-in is three stacked windows rather than one, so it is longer than any single section implies.

**The buckets are calendar months and therefore non-overlapping**, which is why no overlap caveat appears beside any number in that section and why no window length was invented for it. CLAUDE.md failure mode 9 governs rolling windows read as independent observations; a share stepped daily would share 251 of its 252 days with its neighbour. `residual_pc1` peaks at **0.683 in 2023-08**, not in a crisis — which is what the curve-mode diagnosis predicts, since 2022-23 is when the curve moved in ways two PCs could not span.


### 4.3.7 ACCEPTANCE (2026-08-31, W3-P6): the hybrid runs the §5 pipeline unchanged, PSD at every stage

**§15.2's architectural claim is that a second factor set costs nothing in `src/mafrm/risk/`.** W3-P5 demonstrated it for Model B across eight combinations. **The hybrid is the harder test and it was run second for that reason.**

Model B's factors are a homogeneous statistical basis — every column the same kind of object, built the same way, on one scale. The hybrid's frame is not. It carries **six named macro factors in two different units** (four decimal daily returns, two in basis points of yield) alongside **one or two dimensionless residual components** whose standard deviation is near `sqrt(λ_k)`. Three scales, spanning about three orders of magnitude, in one `T × K` matrix. **If `risk/` held an assumption about what a factor is, this is the frame that would surface it.**

**It did not. All eight runs — 1 and 2 components × short and long × `a` = 1.0 and 1.4 — completed every declared stage and were positive semi-definite after every one.** Not marginally: the minimum eigenvalues run **3.8e-06 to 6.3e-06** against a floor of 1e-14, so this is not a near-miss absorbed by a tolerance. `reports/hybrid_residual_pcs.md` carries the per-stage table.

**Nothing in `src/mafrm/risk/` was changed, added to, or special-cased.** The session's diff on that package is empty. `mafrm.factors.hybrid.factor_panel` hands `run_pipeline` a `T × K` frame and a `RiskConfig`, which is the identical call `mafrm.build` makes for Model A and `statistical_report.acceptance` makes for Model B.

**Two guards make that claim mechanical rather than asserted.** `tests/test_risk_architecture.py` already forbids `risk/` importing from `factors/`; `tests/test_hybrid.py` asserts it again at the hybrid, because a pipeline that needed to know a column was "a macro factor" or "a residual PC" would have to reach into `factors/` to find out. And `tests/test_hybrid.py` runs all eight combinations against the committed history and asserts completeness and PSD per stage, so a later change that breaks the architecture fails the gate rather than the report.

**The PSD repair and row 100's Bartlett fallback both fired ZERO times across the eight, and that is the expected reading rather than evidence they are unnecessary.** Both are `K/T_eff` phenomena — `experiments.md` rows 98 and 129 — and at `K = 7–8` against `T = 3,638` the ratio is nowhere near where either fires. These are two more measurements for `reports/stage_k_dependence.md`, not a reason to drop a stage.

**The condition number is units and §5.3.1 already ruled on it.** The finished matrices condition at **7.6e+06 to 9.5e+06**, the same order as Model A's own six-factor matrix. Appending a residual PC of standard deviation ~1.8 to a frame already spanning 0.003 to 5.1 cannot improve the conditioning and does not need to: §5.1 estimates volatility and correlation **separately**, so a level difference between factors is carried in the volatility leg rather than in the correlation matrix the eigendecompositions act on. `experiments.md` row 96 settled this on Model A's own matrix before the hybrid existed.

**`λ_F²` runs 0.868 to 1.014, and the two horizons disagree about the sign** — short over 1, long under it — which is the same pattern W3-P4 measured on Model A's own factors and has the same cause: a 252-day half-life on a panel containing 2020 and 2022.

**No forecast-quality claim is made here.** The acceptance says the pipeline *accepts* the hybrid, not that the hybrid forecasts better than Model A. That is a bias-statistic question, it needs §6.2's optimizer-selected portfolios rather than a multiplier read off the calibration series it was fitted on, and it belongs to W4. **The hybrid's panel also opens 2010-04-27 against Model A's 2008-04-14**, so any later comparison must be run on the common window or it will be comparing two different samples.

---

# 5. The covariance pipeline

Apply identically to both models so the comparison is clean. **Order matters: EWMA → Newey-West → PSD repair → eigenfactor adjustment → volatility regime adjustment.** VRA is last because it is a pure level scaling and must be calibrated against the final matrix.

**Assert positive semi-definiteness after every single stage, not just at the end.** Individually-safe corrections interact badly; this is the most common way a home-built pipeline silently breaks.

## 5.1 EWMA with separated volatility and correlation

USE4 Methodology Notes Table 4.1. Estimate variances and correlations **separately with different half-lives** and recombine:

```
F⁰_ij = ρ_ij · σ_i · σ_j
```

| Parameter | Short horizon | Long horizon |
|---|---|---|
| Factor volatility half-life | **84 days** | **252 days** |
| Volatility Newey-West lags | **5** | **5** |
| Factor correlation half-life | **504 days** | **504 days** |
| Correlation Newey-West lags | **2** | **2** |
| VRA half-life | **42 days** | **168 days** |

Volatilities mean-revert fast and need responsiveness; correlations are noisier and need stability. Note the correlation half-life is **identical** across horizons — only volatility responsiveness changes. Copy that.

Run both horizons. The short model is the optimizer-facing one; the long model is the attribution-facing one.

**Half-life selection is the main dial and there is no right answer.** Equal-weighted history "still fears 2008 in 2014"; short windows "forget the crisis by late 2009." Do not pick a number and hope nobody asks — ship a sensitivity chart of bias statistic and realised min-var portfolio volatility across half-lives from 21 to 504 days.

### 5.1.1 RULING (2026-08-31, W3-P1): the EWMA moments are taken about ZERO

The table above fixes the half-lives and says nothing about centring. This section closes that, before any covariance was estimated.

**The EWMA variance and covariance sums are taken about zero. No trailing mean — EWMA or otherwise — is subtracted, at either half-life, at either horizon.** `covariance.mean_convention: "zero"` in `config/model.yaml`; the loader rejects any other value rather than supporting one.

**This is a coherence requirement, not a convention preference,** and the whole argument is one line. §6.1's bias statistic is

```
b_nt = R_nt / sigma_nt
```

with a **raw** return in the numerator. If the denominator forecast variance about a drifting sample mean while the numerator is measured about zero, the two sides would be measuring different quantities, and `B` — the single statistic §1's identity exists to decompose into a risk-model term and a cost term — would be biased by the mismatch alone, before any estimation error entered. The forecast must be of variance about zero because that is what the realised side is measured against. RiskMetrics assumes a zero mean for the same reason.

Two consequences, both asserted in `tests/test_covariance.py` because they are how the decision stays visible rather than becoming folklore:

- As the half-life grows the estimator converges to `X'X / T`, the **uncentred** second moment — *not* to `numpy.cov`, which demeans and divides by `T-1`.
- There is no `1 - w'w` reliability correction. That correction restores the degree of freedom a *demeaned* estimator spends estimating the mean; a zero-mean estimator spends none, so the denominator is `sum(w) = 1`.

**What the decision costs is measured, not asserted, and the measurement could not have reversed it.** That was registered in advance — `experiments.md` rows 93-95 name the result that would change the decision and record that there is none — which is why those rows are `data-diagnostic` and add nothing to the deflated-Sharpe trial count. The general criterion, which applies from here on: **name the reversing result before running the measurement. If no result would change the decision it is not a trial and owes no row. If some result would, it is a sweep and counts as `model-config`.**

`reports/mean_convention.md` has the numbers and they are not uniformly small. The exact identity `sum(w f^2) = sum(w (f-m)^2) + m^2` makes the difference quadratic in the mean-to-volatility ratio, so the **median** absolute shift in a factor volatility is 10.2 bp at the short horizon and 4.0 bp at the long one — negligible, as the ruling anticipated. The **tail** is not: the largest single reading is 323 bp (3.23%), on the dollar factor at an 84-day half-life where its EWMA mean reached 0.256 of its EWMA volatility. Correlations are unambiguous — the largest off-diagonal disagreement anywhere is 0.005.

The tail is reported rather than rounded off, and it changes nothing here: the coherence argument holds at 3 bp and at 300 bp, and a decision that a large number would have reversed was never a coherence argument. What it earns is a note for week 4: zero-mean forecasting makes the forecast volatility *larger* than a demeaned one by exactly this amount, so it biases `B` **downward**, never upward, and cannot manufacture an apparent risk-model failure.

**One known divergence, left open deliberately.** `mafrm.factors.macro.condition_numbers` — the §4.1.2 conditioning diagnostic — subtracts an EWMA mean and carries the `1 - w'w` correction. It is not re-based, because its numbers are published in `reports/factor_correlations.md` and `experiments.md` row 68 and silently moving a shipped diagnostic is worse than carrying a measured divergence. The divergence is bounded at 0.005 of a correlation, which is the number that makes leaving it defensible.

### 5.1.2 FORWARD CONSTRAINT (2026-08-31, W3-P1): a burn-in minimum is a bound on K/T_eff, never a day count

No burn-in parameter exists and none was added in W3-P1, because the §15.2 signature takes its window from the caller and returns one matrix — there is no such decision to make until W3/W4 iterates matrices through time. The shape of the future parameter is fixed here so that a later session does not reach for 252 because it is lying around in `factors.macro.orthogonalization.expanding_min_window`.

**When the minimum arrives it must be expressed as a bound on `K / T_eff`.** `T_eff = 2*tau/ln 2` is already in `config/model.yaml`; the bound then depends only on the number of factors and the half-lives, which makes it **asset-class agnostic by construction** and therefore compatible with §15.2. A day count is not: it would have to be re-chosen for every panel and every half-life, and it says nothing about how many parameters are being estimated from the window. Reusing the orthogonalization's 252-day minimum would do both wrong things at once — invent a parameter, and import a macro-model assumption into `risk/`.

**`K/T_eff` is therefore emitted as a diagnostic from the first covariance stage onward**, alongside Shepard's `[1 - K/T_eff]^-2`, in both its asymptotic form (`2*tau/ln 2`, which is what §15.2's table quotes) and the realised Kish form for the actual window. It costs nothing to compute and it is the project's headline result, so §15.2's K/T table is assembled at the end from a quantity that has been reported all along rather than computed specially to make a point.

The first reading already reproduces §15.2's table on the project's own data: Model A at K=6 gives K/T_eff = 0.008 and a 1.7% understatement at a 252-day volatility half-life, and 0.025 and 5.1% at 84 days, against the table's 0.008/1.7% and 0.025/5.1%.

### 5.1.3 REGISTRATION (2026-09-02, W4-P2b): the half-life sweep is a `K/T` test, and the prediction has two legs

§5.1 asks for *"a sensitivity chart of bias statistic and realised min-var portfolio volatility across half-lives from 21 to 504 days."* This registers it in full — the grid, what is swept, what the sweep is classified as, what is predicted, and what would falsify it — **before the first history was built.** No bias statistic on any grid point had been computed when this section was written.

**What is swept: the FACTOR VOLATILITY half-life, alone.** The correlation half-life is held at 504. CLAUDE.md's parameter table pins it identically across horizons and marks that deliberate, so it is not a dial in this model and sweeping it would ask a question the specification has already answered; a joint sweep is also a 2-D grid that multiplies the history count for nothing. `validation.halflife_sensitivity.grid` in `config/model.yaml`.

**The grid is nine points, and the justification is a principle rather than a provenance list:** integer trading months — every point a multiple of `data.trading_days_per_month` — log-spaced, spanning §5.1's own endpoints 21 and 504, and hitting both shipped values exactly.

```
21   42   63   84   126   168   252   336   504     trading days
 1    2    3    4     6     8    12    16    24     months
```

Consecutive ratios alternate 2.00 / 1.50 / 1.33 for a geometric mean of `24^(1/8) = 1.4877`, so there is no dense region and no sparse one. **An earlier draft justified the nine points individually**, as constants this project already carries — 21 the trading month and §15.4's RSTR lag, 42 and 168 the two VRA half-lives, 63 BETA/HSIGMA, 84 and 252 the shipped volatility half-lives, 126 MOMENTUM, 504 the correlation half-life. That argument covers eight of nine and **fails on 336**, and it is recorded here as refused rather than quietly repaired: a provenance claim that breaks on one point is weaker than a rule that covers all nine, and this project's credibility is built on claims that survive checking.

#### This is a `K/T` test, and that is the result rather than the context

`T_eff = 2τ/ln 2`, so sweeping τ over 24× sweeps `K/T_eff` over 24× **on one panel, with `K`, `N`, the asset class, the data source, the factor construction and the scored window all held fixed.** Shepard's predicted volatility understatement across the grid, at `K = 6` and at the `N = 13` naive comparand's rank:

| τ | `T_eff` | `K/T_eff` | `[1−K/T_eff]⁻²−1` | `N/T_eff` | `[1−N/T_eff]⁻²−1` |
|---|---|---|---|---|---|
| 21 | 60.6 | 0.0990 | **23.2%** | 0.2145 | **62.1%** |
| 42 | 121.2 | 0.0495 | 10.7% | 0.1073 | 25.5% |
| 63 | 181.8 | 0.0330 | 6.9% | 0.0715 | 16.0% |
| **84** *(shipped, short)* | 242.4 | 0.0248 | **5.1%** | 0.0536 | 11.7% |
| 126 | 363.6 | 0.0165 | 3.4% | 0.0358 | 7.6% |
| 168 | 484.7 | 0.0124 | 2.5% | 0.0268 | 5.6% |
| **252** *(shipped, long)* | 727.1 | 0.0083 | **1.7%** | 0.0179 | 3.7% |
| 336 | 969.5 | 0.0062 | 1.2% | 0.0134 | 2.7% |
| 504 | 1454.2 | 0.0041 | 0.8% | 0.0089 | 1.8% |

The τ = 84 row reproduces §6.4's table exactly at `K = 6`; the `N` column differs from §6.4's 13.6% only because §6.4 was written at the universe's nominal `N = 15` and the residual panel carries 13, which is the figure `reports/bias_statistics.md` has used throughout.

**This de-risks the project's headline.** §12's week 8 deliverable is measured optimizer-portfolio bias against `K/T` with Shepard's `[1−K/T]⁻²` overlaid, and until now the second end of that range rested entirely on W7's equity module arriving. It now has a within-panel range that does not depend on W7 at all — and a within-panel range is arguably the **cleaner** evidence, because a macro-versus-equity comparison moves `K`, `N`, the asset class and the estimator's input data at once, while this moves one number. W7 remains the wider range and the harder test; it is no longer the only one. `reports/halflife_sensitivity.png` therefore carries `K/T_eff` as a companion panel and the write-up states it as a result.

**Both forms of `K/T_eff` are reported**, as §5.1.2 requires: the asymptotic `2τ/ln 2` above, which is what §15.2's table quotes and what the Shepard overlay is drawn against, and the realised Kish form for the actual expanding window, which is what the pipeline emits per date and which departs from the asymptotic form early in the sample where the window is shorter than `T_eff`.

#### The prediction, in two legs that must hold jointly

W4-P2's §6.2.6 says the family-4 bias is **specification error** in `Δ` rather than estimation error in `F`. The sweep tests that, because the two channels respond to τ differently and **the sharper form separates them**:

1. **The factor-covariance channel.** As τ falls, `K/T_eff` rises and Shepard says every family's `B` should rise. Families 1 and 2 are where this shows: they barely touch the diagonal assumption.
2. **The specification channel.** `Δ`'s misspecification — the residual correlations discarded by `Σ = X F X' + Δ` — is a property of the model's form and is **constant in τ**. It is most of family 4's gap and none of family 2's.

**So the registered prediction is two things jointly: (a) all families' `B` rise as τ falls, and (b) the family-4-minus-family-2 gap is roughly τ-invariant.** W4-P2 measured that gap at **+0.322 short / +0.315 long**.

**FALSIFIER: the gap moves materially with τ.** That puts the error back in `F` and qualifies §6.2.6. This is strictly stronger than *"family 4 moves less than the others"*, which the levels alone would satisfy for the wrong reason, and it is testable on the same nine builds.

**A third channel exists and is named here so that it cannot be reached for afterwards as a rescue.** A shorter half-life is also a **more responsive** forecast, which tracks conditional volatility better and pushes `B` toward one — the opposite sign to leg 1, and confounded with it across every point of the grid. If the levels do not rise as leg 1 predicts, responsiveness is the first suspect; but it is being named **before** the numbers exist, and neither the sweep nor anything in W4 separates the two, so it can be offered as a candidate explanation and never as a confirmed one. It does not touch leg (b), which is the load-bearing half: responsiveness acts on the forecast level for every family alike and therefore moves both terms of the difference.

#### Classification: `data-diagnostic`, with the enforcement written at the row

**`N` is unchanged.** The shipped half-lives are the published USE4 constants, pinned in CLAUDE.md's parameter table and fixed long before this grid existed, and **no result from this sweep may change which value ships.** So there is no reversing result, and W3-P3's criterion — restated at §5.1.1 and applied unchanged since — makes it a diagnostic rather than a trial: *name the reversing result before running the measurement; if none exists it is not a trial.*

The partition's own table names half-lives as `model-config`. **It does so by example, because in the general case a half-life sweep is a selection.** Here it is not, and the example does not govern.

Two enforcements, both written before the run:

1. **The temptation is the trigger, not the act.** The moment anyone proposes shipping a swept value, the sweep becomes `model-config` and counts **retroactively, in full** — all nine points, not the one proposed. There is no version of this where a selection is made and only the selected point is charged for.
2. **The grid is reported in full**, across its whole range and not only near the shipped points, so that nobody can later claim the published values were "confirmed" by a curve that examined only its own neighbourhood.

#### The scored window is the INTERSECTION across the nine points

§6.2.4's *"the window is common to every variant"* clause, extended one step and otherwise unchanged. A curve of `B` against τ scored on nine different date sets is partly a curve of `B` against **sample**, and the reason is §6.2.4's own: letting each point start where its own estimator became well behaved would let the worst-behaved one look best by being scored on the calmest sample.

§6.2.4 recorded that the PSD-repair boundary landed in the same place at both horizons though their volatility half-lives differ by a factor of three. **Across 24× that is not assumed**: each grid point's own `first_clean_position`, its dropped-date counts and its `K/T_eff` profile are measured and reported beside the chart, and the intersection is whatever they turn out to imply.

#### Nine histories, not eighteen — measured rather than argued

The only expensive quantity is §5.3's Monte Carlo (2,000 trials/date; every other stage in the battery runs at 0.8 ms/date and is rebuilt on each render). Of the four `RiskConfig` fields that differ between the horizons, three cannot reach it — `horizon` is a label, `specific_volatility_halflife` enters §5.5 which is rebuilt at render, and `volatility_regime_halflife` is post-VRA while this chart is pre-VRA — and the fourth **is** the swept dial. So the cached factor-covariance history is a function of τ alone and both horizons render from the same nine files, differing only in the specific-risk half-life applied at render.

**Checked rather than asserted**, in §5.3.3's shape: τ = 252 built under the *short* horizon's shape reproduces the committed `long` column of `reports/bias_forecast_history.csv` to **4.5e-12 relative**, which is that cache's own 12-significant-digit write precision.

τ = 84 and τ = 252 are **rebuilt rather than lifted** from that cache. It costs nothing — nine fits in one fan-out wave — it keeps every partial at `history.write_partial`'s 17 digits instead of splicing in two at 12, and it makes those two points an independent reproduction of W4-P2's published family-4 `B` of **1.3321 short / 1.3135 long**, which is a free regression check on the entire harness.

**The cache key is three-part**, because `history.risk_config_digest` hashes both horizons' full `RiskConfig` and therefore cannot separate grid points: the sweep's partials key on `(shipped risk_config_digest, τ, panel_digest)`. That strengthens the guard rather than weakening it, and a test asserts the key moves when any of the three moves.


### 5.1.4 FINDING (2026-09-02, W4-P2b): family 4's bias is invariant to a 24× sweep of `K/T_eff`, and leg (b)'s falsifier had no threshold

`reports/halflife_sensitivity.md` and `.png`. Pre-VRA, `a = 1.0`, common scored window of 3,874 dates, 2009-05-07 to 2024-12-31.

**The reproduction check passes exactly.** Both shipped points were rebuilt rather than lifted, through a refactored code path, a cache rebuilt under a new `risk_config_digest`, and §5.2.4's amended PSD assertion. τ = 84 short gives family 4 **1.3321** against W4-P2's 1.3321; τ = 252 long gives **1.3135** against 1.3135; families 2 match at 1.0101 and 0.9987. Same statistic, same dates.

**All nine scored windows are identical** — 3,874 dates, same start, nothing dropped to reach the intersection. §6.2.4 saw the repair boundary hold across a 3× half-life difference; it holds across 24× because at `T ≤ 10` against `K = 6` the window length binds rather than the nominal half-life. §5.1.2's insistence that a burn-in is a bound on `K/T_eff` and never a day count is what makes that true, and it made the intersection free rather than a compromise.

**Leg (a) holds monotonically.** As τ falls 504 → 21 and `K/T_eff` rises 0.0041 → 0.0990: family 1 **1.0095 → 1.0580**, family 2 **1.0003 → 1.0301**, family 3 **1.0277 → 1.0982**. The direction is Shepard's. The **level is about an eighth of his closed form** — 3.0% observed against 23.2% predicted at τ = 21 — which is the opposite sign of disagreement to MWO's, and is reported as an observation because no band was registered for the magnitude.

**Leg (b)'s falsifier cannot be adjudicated, and the defect is mine.** *"Materially"* was never given a number, and no interval for a **difference** of two bias statistics on the same dates and portfolios was registered or computed; the χ² interval is for a single `B` and the two terms are strongly correlated, so borrowing it would be using a yardstick that does not measure this quantity. Picking a threshold after seeing the number is what §6.6 exists to prevent, so **the verdict is withheld and the measurement is published**:

| | `tau` = 504 | `tau` = 21 | range |
|---|---|---|---|
| family 2 | 1.0003 | 1.0301 | 0.0299 |
| **family 4** | 1.3321 | 1.3285 | **0.0039** |
| gap | +0.3318 | +0.2984 | 0.0334 |

**90% of the gap's drift is family 2 rising** (96% at the long horizon), which is leg (a) operating. Family 4 moves **0.29% of its own level across a 24-fold change in `K/T_eff`.**

**The falsifier's consequence does not follow, and this is the session's result.** Its consequence was *"that puts the error back in `F` and qualifies §6.2.6"*. The term that moved is the estimation-independent family's; the term §6.2.6 is about is flat while the estimation-error channel — the thing `K/T`, Shepard and §5.3 all describe — is swept through an order of magnitude. **A bias invariant to `K/T` is not a `K/T` phenomenon.** W4-P2's control C1 tested one alternative specification; this dials the competing explanation 24-fold and finds it inert. §6.2.6 is confirmed by a harder test than the one that produced it, and this is the strongest evidence in the project that the risk-model term of §1's identity is specification error.

**It also de-risks §12's week-8 deliverable.** The `K/T` curve now has a within-panel range that does not depend on W7 arriving. W7 remains the wider range and the harder test.

**Two things this does not show.** It does not select a half-life — see §5.1.3's enforcements, and family 4's realised min-var volatility spans only 5.100 to 5.195 bps/day with no interval attached. And the naive asset-level comparand still beats the factor model on realised min-var volatility at **every point of the grid**, 4.562–4.779 against 5.100–5.221, which is §6.2.6's finding reappearing across the whole sweep for the reason recorded there.

## 5.2 Newey-West

```
F_NW = C₀ + Σ_{δ=1..L} (1 − δ/(L+1))·(C_δ + C_δᵀ)
```
`C_δ` = EWMA lag-δ autocovariance at the relevant half-life, then scaled by the horizon (×21 for monthly).

```python
FCM = cov_ewa(F, half_life, 0)
for i in range(1, max_lags + 1):
    C = cov_ewa(F, half_life, i)
    FCM += (1 - i / (1 + max_lags)) * (C + C.T)
D, U = np.linalg.eigh(FCM * multiplier)
D[D <= 0] = 1e-14                      # REQUIRED — NW is not guaranteed PSD
FCM = U @ np.diag(D) @ U.T
```

The PSD repair is not in the MSCI write-up but is mandatory in practice. Log how often it fires and by how much — that number is itself a diagnostic.

### 5.2.1 RULINGS (2026-08-31, W3-P2): separated application, daily units, and how invariant 4 is read here

Three decisions the pseudocode above does not make, settled before the stage was written rather than derived under time pressure inside it.

**1. Newey-West is applied SEPARATELY to the volatility and correlation terms, not once to an assembled `F₀`.**

The pseudocode's single `cov_ewa(F, half_life, i)` loop reads as one correction applied to an already-assembled matrix. **The two lag counts are the proof that it is not.** If the correction were applied once there would be one lag number in USE4 Table 4.1, not two; serial correlation in volatility and serial correlation in correlation are different phenomena with different persistence, which is the same reason §5.1 gives them different half-lives in the first place. The pseudocode is exposition, not specification — the same class of artefact as the `λ^a(λ−1)+1` garble §5.3 flags in one PDF rendering of the MSCI paper, where the published text and the open-source implementations agree against the rendering. The discrepancy is noted in a code comment, as this project does elsewhere.

So: Bartlett at `L=5` on the second moment at the volatility half-life → `σ`; Bartlett at `L=2` on the second moment at the correlation half-life → `ρ`; recombine as `F_ij = ρ_ij σ_i σ_j`.

**The trap that follows from it, and it is easy to miss.** Newey-West does not preserve a unit diagonal. Correcting an already-normalised `ρ` and recombining would count each variance twice — once through `σ`, and again through the inflated diagonal left in `ρ`. The correction is therefore applied to the **second moment** at the correlation half-life and the correlation is formed from the corrected matrix, which renormalises the diagonal to exactly one by construction. That the diagonal really is unit is asserted in code, not assumed.

**The assembled variant was not run as a comparison.** This is a ruling on a structural argument, not a sweep, and running the alternative would have made it one. `experiments.md` records that under W3-P2.

**2. The horizon multiplier is applied ONCE, DOWNSTREAM, and never inside a stage.**

§5.2 says the sum is *"then scaled by the horizon (×21 for monthly)"*. The pipeline stays in **daily units through all five stages** and the multiplier is applied at the point of consumption by `mafrm.risk.horizon.scale_to_horizon`, the only function in the project that applies one. `data.trading_days_per_month: 21` is in `config/model.yaml` as an arithmetic constant and is **deliberately absent from `RiskConfig`**, so nothing the pipeline can reach knows the number.

Invariant 6 puts the number in the config either way, so that is not the discriminator. The discriminator is **failure mode 7**: a scaling applied inside a stage is invisible to every later stage and to every consumer, and a matrix 21× too large is still PSD, symmetric and identically conditioned — it passes every check the pipeline runs. This project has been bitten three times by corrections that each passed their own check. Beyond that, different consumers legitimately want different horizons — daily bias statistics (§6.1), monthly VaR, the optimizer's rebalance interval — and a matrix that has already chosen one forces all the others to undo it. **`risk/` should no more decide the caller's horizon than it should know the caller's asset class**; that is invariant 10 applied to a different axis.

A test asserts the multiplier cannot be applied twice: `scale_to_horizon` takes an array and returns a `HorizonCovariance`, so feeding its own output back is a type error and a runtime refusal.

**Harmless to §5.3, stated so the ordering is not re-litigated:** a uniform scalar multiple scales every eigenvalue equally and changes neither the eigenvectors nor the `λ̃/λ` ratios the eigenfactor adjustment reads. Note this is exactly what is *not* true of §5.3.1's per-factor rescaling — which is the whole content of that ruling. The two are kept in separately named functions, one global and one diagonal, so a later reader can see at a glance which is which.

**3. `assert_psd` does not run between Newey-West and its repair, and that is a READING of invariant 4 rather than a relaxation of it.**

Invariant 4 exists to catch **silent** non-PSD. The `newey_west` stage is the one place in §5 where a non-PSD result is *declared in advance* — this section calls the repair REQUIRED and the next stage in `covariance.stages` exists for no other purpose. A stage whose successor is its repair is not silent, so asserting between the two would detect no defect; it would convert an expected and already-handled outcome into a crash, and the only way to keep the pipeline running would then be to widen `psd_minimum_eigenvalue`, which is the move CLAUDE.md's residual stopping rule forbids.

So: `newey_west` **measures and reports** its spectrum — `λ_min`, `λ_max`, negative count — without asserting; `psd_repair` floors; and the assertion runs after the repair, which is the transformation that is supposed to yield a PSD matrix. `measured_not_asserted_stages()` names the exemption, and a test pins it to exactly one member so it cannot spread. **A later session reading the missing assertion as an oversight and "fixing" it into a crash is the failure this ruling exists to prevent.**

**4. The zero-mean convention carries into the lag terms, and it is asserted rather than written twice.**

§5.1.1 rules that the EWMA moments are taken about zero. The Newey-West lag terms `Γ_j = Σ w · f_t · f_{t−j}ᵀ` are therefore also computed without demeaning. If stage 1 is zero-mean and stage 2 demeans, the pipeline estimates two different quantities and the inconsistency shows up **nowhere except in `B`** — failure mode 7 again, two corrections each passing their own sanity check. W3-P1 measured that the two conventions are not interchangeable at the extreme even though they agree at the median (10.2 bp median against a 323 bp tail on `dollar`), so this is not a distinction without a difference.

Two mechanisms rather than a comment. Structurally, `ewma_autocovariance(lag=0)` **is** `ewma_second_moment` — one implementation, not two written the same way. And at runtime the stage evaluates the lag-0 term through the *general* lagged path that also serves every lag ≥ 1 and asserts it matches, against a tolerance derived from `T · eps` rather than chosen. A later session that demeans the lagged path breaks this at the stage that did it.

### 5.2.2 FINDING (2026-08-31, W3-P2): the repair fires only where `K/T_eff` is near 1

`reports/psd_repairs.md` rebuilds the pipeline on an expanding window ending at every date in the factor panel, both horizons, no stride: 4,141 builds each. The repair fires on **5 dates, forming 1 contiguous episode**, and every one is at `T ≤ 10` against `K = 6` — realised `K/T_eff` of 1.000, 0.857, 0.750, 0.667, 0.600. From `T = 11` onward it does not fire once in 4,136 builds, at either horizon.

Three things follow.

**The boundary is a `K/T_eff` boundary, which is what §5.1.2 predicted the shape of.** It sits at the same place at both horizons even though their volatility half-lives differ by a factor of three — something a day count could not have produced. **It is not a calibrated bound and must not be used as one:** it is one episode of five overlapping dates on one panel with one factor set. The threshold, when W3/W4 needs one, owes its own registration.

**This is the first independent corroboration in this project that `K/T` is the governing parameter, and it is of a different kind from the last one.** §15.2's table was reproduced in W3-P1 by computing Shepard's formula and finding it agrees — that validates the arithmetic. This computes Shepard's formula nowhere. It arrives at the same parameter from a **numerical failure mode**, in an implementation that was not aiming at `K/T` and would have reported whatever boundary the data had. Reproducing the published table validates the arithmetic; this validates the framing.

**The negative eigenvalue is genuinely Newey-West's.** The EWMA stage is a Gram matrix and is PSD by construction at every one of those dates, with `λ_min` between 3.9e-7 and 5.3e-6. It is the Bartlett lag terms that push it below zero, on windows where the longest lag has one or two pairs to average over.

**Zero firings outside that regime is the expected result at `K = 6`, not evidence of a dead code path.** Non-PSD after Newey-West is a large-K failure; with a correlation matrix conditioned at 3.27 and a Bartlett kernel truncated at 5 lags, a genuine estimation window may simply never produce one. Diagnosing that as a bug would burn a session on a non-defect — the same error as the 0.3 correlation gate withdrawn in §6.5.2 and the daily-correlation gate withdrawn in §3.2. The requirement is therefore **not** that it fire on real data but that the path be demonstrably live, which is shown on a synthetic case (`K=6` from `T=12`, drawn from `model.seed`, relative breach −1.5e-3) so that it is checkable without the gitignored cache. **§15.2's `K = 56` cross-sectional module in week 7 is where this stage earns its place.**

**The floor is reported against the spectrum, because an absolute floor is scale-dependent.** Invariant 4 fixes it at 1e-14 and it has not been moved. On the full-window matrices it sits 8.6–8.8 orders of magnitude below `λ_min`, which is safe — and safe *by accident of the unit convention* rather than by design, since §5.3.1 records that this panel's 9.96e6 condition number is almost entirely a 7.3e6 variance ratio across mixed-unit columns. **W3-P3's numeraire decision changes what the floor means and must trigger a re-read of that table.**

### 5.2.3 AMENDMENT (2026-08-31, W3-P5): the Bartlett fallback fired, at `K = 3` rather than `K = 56`

`experiments.md` row 100 pre-registered, in W3-P2 and while nothing was broken, what to do if the Bartlett correction drives a factor variance non-positive. **It fired in W3-P5** — on Model B's *detoned* factor panel (§4.2), at `K = 3`, `T = 7…10`, `K/T ≈ 0.43`, on 2–3 dates of 4,143 with 2 contiguous episodes at each horizon. The denoised panel never fires. The registered remedy is applied unchanged: **fall back to the uncorrected stage-1 EWMA variance for that factor on that date, count the substitution, report it.** No floor, no new constant — the fallback value is one stage 1 already computes.

**The remedy attaches to a mechanism, not to a panel size.** Newey & West's non-negativity guarantee is a property of the **equally-weighted** HAC estimator: with constant weights the truncated Bartlett sum rewrites as a sum of outer products of overlapping block sums, and a sum of outer products is PSD whatever the data. Exponential weights break that rewriting — the weights differ across the observations inside each block, so the terms no longer collect into squares. USE4 specifies both an EWMA and a Bartlett correction, so this project meets a case the textbook result does not cover. That composition is asset-class and `K` agnostic, which is why row 100's `K = 56` was a **prediction about where it would first fire and not a boundary on where it applies**.

**The prediction was wrong in an informative direction, and that is the finding.** Row 100 expected a large-`K` failure. It fired at `K = 3` — but at `T = 7`. **So it is governed by `K/T`, not by `K`**, which is precisely §5.2.2's shape for the PSD repair: fires only where `K/T_eff` approaches 1, never above it. Two different failure modes of the same stage, both indexed on the same ratio. The test fixture corroborates it by construction: **plain iid standard-normal noise at `T = 7` against 5 lags provokes the failure**, with no special serial-correlation structure — seven observations give the longest lag two overlapping pairs, and two pairs is not an estimate. This is the fifth independent arrival of the governing ratio in the project and the first that arrives by *correcting* a registration.

**It cost the deflated-Sharpe count one trial.** Row 100's category was fixed as `model-config` at registration, and it was not re-argued at the moment of application even though a case exists — the alternative to substituting is that the matrix does not build, so nothing could have been flattered. Making that case in the direction that lowers the count is the reclassification §6.6 forbids. `N` = 3.

**And it fired §15.2's falsifier, which was answered with a sharper test rather than a waiver.** The change is inside `src/mafrm/risk/`, which `experiments.md` row 125 registered as the architecture early warning. The trigger was a true positive; the inference was not — a numerical fallback for a failure any factor set can provoke knows nothing about what the columns mean. Rather than accept that as an argument, the gate was strengthened: the change is permitted only because it passes the existing asset-class greps **and** a new test that demonstrates the fallback behaves identically on a synthetic panel unrelated to either model, invariant under relabelling of the columns. Same move as §5.3.1's scale-invariance test. **The precedent is the sharper test, not the accepted exception.**

### 5.2.4 RULING (2026-09-02, W4-P2b): the PSD detection threshold is scale-relative, in units of machine epsilon

**The superseded threshold was absolute. Its justification was correct for the panel it was written against, stopped being correct when the panel changed, and nothing re-derived it.** W3-P1 set `psd_minimum_eigenvalue = -1e-12`, and `config/model.yaml` defended the absolute form on the ground that *"the largest-scale factor covariance this project estimates is the rates panel in basis points, whose variances run of order 1e1-1e2, so double-precision roundoff on its eigenvalues is of order 1e-14 — two orders of magnitude inside the threshold."*

**On the macro factor panel that argument is right, and it was checked.** Its largest repaired `λ_max` over 4,141 expanding-window builds is **1.59e+02**, so one ulp is 3.52e-14 and the threshold sits **28× outside** the noise. `reports/psd_repairs.md` now prints that line.

**On the panel this project actually forecasts from, it is wrong by three orders of magnitude.** §4.3's hybrid residual panel carries the *same six factors* at a largest `λ_max` of **5.43e+04** — **390× the macro panel's** — where one ulp is **1.21e-11**, i.e. **12.1× the threshold** rather than a twenty-eighth of it. That is the panel W4-P1's specific risk and W4-P2's entire validation battery are built on. The threshold there sat an order of magnitude *inside* the representation noise, which means **it carried no information at all on the one panel whose numbers get published.**

**Nothing detected the drift because nothing was looking.** `reports/psd_repairs.md` logged how badly non-PSD the repair's *input* was at every build and never once how close its *output* sat to the detection threshold. A margin nobody measures is a margin nobody knows they have lost, and this is CLAUDE.md failure mode 7 in its quietest form: a constant that was sound when written, a panel that changed under it, and no assertion coupling the two.

**How it surfaced.** W4-P2b's sweep could not build `τ = 42` or `τ = 504`: both raised at the second date of the expanding loop, on a matrix that had just been repaired.

| | τ=42, 2009-04-28 | τ=504, 2009-04-27 |
|---|---|---|
| Newey-West `λ_min` (genuinely non-PSD; the repair is correctly required) | −1.808e+01 | −3.681e+02 |
| after flooring at `psd_eigenvalue_floor` and reconstructing, `λ_min` | −1.397e-12 | −2.505e-12 |
| `λ_max` | 4.37e+04 | 4.94e+04 |
| **`λ_min / λ_max`** | **−3.2e-17** | **−5.1e-17** |
| the violation, in ulps of `λ_max` | **0.14** | **0.23** |
| the old threshold, in ulps of `λ_max` | 0.10 | 0.09 |

The repaired matrices are positive semi-definite **to a fraction of one epsilon**. The repair was not failing; the check was refusing a matrix that is PSD to within the precision the matrix is stored in.

**THE SHIPPED PATH WAS PASSING BY LUCK, AND THAT IS THE FINDING THAT MATTERS.** This is not confined to the sweep. The shipped short horizon's worst repaired `λ_min` is **−6.66e-13** against the −1e-12 threshold, on 2009-04-28 — a **1.5× margin** on a quantity whose own noise floor is 15× the observed value. Shipped long is −8.26e-14. Neither margin was designed; both are where the arithmetic happened to land. **W4-P2's published numbers were computed one unlucky date from an assertion failure, and the numbers are unaffected — what was unsound is the confidence a passing check gave in them.** A check that cannot distinguish a repaired matrix from a broken one is worse than no check, because it is believed.

**The replacement.** `mafrm.risk.checks.assert_psd` requires

```
λ_min  ≥  −(size + psd_reconstruction_roundings) · ε · λ_max
```

with `ε = numpy.finfo(float).eps`.

**Why epsilon and not a relative constant.** A bare relative bound — `λ_min/λ_max ≥ −1e-12` — would clear this residual, and that is exactly what is wrong with it: the number would have been chosen *by* the residual. One ulp of `λ_max` is a property of the double-precision representation, fixed before anything here was measured. W1-P3's precedent requires the basis of an accommodation to be independent of the thing accommodated, and `ε` is; `1e-12` would not be. **The observed violations are below a single ulp** — 0.14 and 0.23 of one — so the residual is smaller than one epsilon relative to scale, and that sentence is the whole justification.

**Why the coefficient is a step count.** It counts the roundings a repaired matrix accumulates per entry between its floored spectrum and this assertion: `size` for the inner product of the reconstruction `(U Λ) Uᵀ`, one for the scaling by `Λ`, and one for the explicit symmetrisation — hence `psd_reconstruction_roundings: 2` and a coefficient of 8 at `K = 6`. Weyl's inequality carries each stage's perturbation to every eigenvalue one for one. The two eigendecompositions' own backward errors are the same order in `size` and are **deliberately not counted**, because what covers them is measured margin rather than argument. `size` is a property of the matrix and not of the asset class, so this stays inside invariant 10 and needs no revision for W7.

**IT IS NOT A WIDENING, AND THE STRICTER DIRECTION WAS VERIFIED BEFORE IT SHIPPED** — the ruling's second condition. On a matrix of scale 1 the new bound is −8.9e-16, about a thousand times **tighter** than the number it replaced; in correlation space, where §5.3 operates and `λ_max ≤ K`, it is ninety times tighter. Every assertion in the project was re-run under the new form: every §5 stage, both horizons, all 3,882 forecast dates, plus the eigenfactor outputs at seven of the sweep's half-lives, 2 scalings each.

- **Regressions — matrices that passed under −1e-12 and fail under the new form: ZERO.**
- Worst violation anywhere on the asserted surface: **0.072 ulp** of `λ_max` (`psd_repair`, short, 2009-05-05), against a bound of 8. A **111× margin**.
- Every eigenfactor output is strictly positive definite at every date and both `a`.
- `newey_west` is absent from that surface by design: it is the one stage whose successor *is* its repair, so it measures rather than asserts (see `mafrm.risk.checks`). Its `λ_min/λ_max` runs to −7e-3, twelve orders outside the bound, which is what a real failure looks like and why the new form still catches one.

**What this cost, and why it was paid.** `psd_minimum_eigenvalue` is a field of `RiskConfig`, so replacing it changed `history.risk_config_digest` and marked all four committed forecast-history caches stale. They were rebuilt rather than re-headed. **The rebuild is the proof the fix is inert**: the caches' bodies are byte-identical before and after, so the change alters no number anywhere in the project — only what is asserted about them. That is also why the fix owes no `experiments.md` trial row.

## 5.3 Eigenfactor risk adjustment

Menchero, Wang & Orr (2011). **This is the central technique of the project.** It is what makes the risk-model term in the §1 identity move.

The problem: sampling error in `F̂` is not neutral. Diagonalise `F̂`; the eigenfactors' variances are systematically mis-estimated — small eigenvalues biased **down**, large ones slightly **up**. An optimizer seeking minimum risk deliberately loads onto exactly the low-variance eigenfactors, i.e. onto the most under-forecast directions. Hence optimized portfolios' risk is under-predicted even though random portfolios' risk is not. In MSCI's words: *"portfolios with the lowest forecast risk are those whose risk is underestimated the most."*

**Algorithm:**

1. `F₀ = U₀ D₀ U₀ᵀ`
2. For each Monte Carlo trial `m = 1…M`: draw `b^m` as K×T with row k ~ N(0, D₀(k)); set `f^m = U₀ b^m`
3. `F^m = f^m (f^m)ᵀ / (T−1)`
4. `F^m = U^m D^m (U^m)ᵀ`
5. `D̃^m = (U^m)ᵀ F₀ U^m` — take the diagonal. This is the *true* risk of the portfolio the simulation thought was eigenfactor k.
6. `λ(k) = sqrt( (1/M) Σ_m D̃^m(k)/D^m(k) )` — empirically ≈1.5 at the smallest eigenfactor, declining monotonically to ≈0.95 at the largest
7. `γ(k) = a·[λ_P(k) − 1] + 1` where `λ_P` is a **parabola fitted to λ(k)** to smooth simulation noise at the spectrum ends
8. `D₀* = γ² D₀`, `F₀* = U₀ D₀* U₀ᵀ`

```python
D_0, U_0 = np.linalg.eigh(FCM)
D_0[D_0 <= 0] = 1e-14
Lambda = np.zeros(K)
for _ in range(M):
    b_m = np.array([rng.normal(0, d**0.5, T) for d in D_0])
    f_m = U_0 @ b_m
    F_m = f_m @ f_m.T / (T - 1)
    D_m, U_m = np.linalg.eigh(F_m)
    D_m[D_m <= 0] = 1e-14
    Lambda += np.diag(U_m.T @ FCM @ U_m) / D_m
Lambda = np.sqrt(Lambda / M)
Gamma  = coef * (Lambda - 1.0) + 1.0      # linear in (λ−1). NOT a power law.
FCM    = U_0 @ np.diag(Gamma**2 * D_0) @ U_0.T
```

**Parameters and the one real decision:**
- `M = 1000–3000`. Not published; anything in that range is fine.
- `T` must match the estimation window used for `F₀` — the bias magnitude is a function of K/T. For EWMA use the effective sample size (§6.4): `T_eff = 2τ/ln2`, so **242 days for the 84-day half-life** and **727 for the 252-day**.
- **`a = 1.4` for an optimizer-facing model; `a = 1.0` for an attribution-facing model.** MSCI publishes `a = 1.4` but USE4 *in production* uses the milder `a = 1` version, explicitly because the 1.4 scaling overstates the volatilities of the pure factors themselves and corrupts attribution. **Run both, report both, and explain the trade-off.** That paragraph in the README is worth more than the code.

Note this is applied to the **factor** covariance matrix in factor space, never to the asset-level matrix.

One PDF rendering of the MSCI paper garbles Eq. A8 into `λ^a(λ−1)+1`. That rendering is wrong; the open-source implementations confirm the linear form. Mention the discrepancy in a code comment — it shows you read the primary source and cross-checked it.

### 5.3.1 RULING (2026-08-31, W3-P1b): the adjustment must be scale-invariant

Settled now, against `experiments.md` row 96, so that W3-P3 starts from the principle rather than deriving it under time pressure.

**The eigenfactor adjustment must be invariant to the units each factor is expressed in.** A risk correction whose output changes when a factor is quoted in decimals rather than in basis points is not a risk correction — it is a correction to an arbitrary choice of numeraire, and nothing about estimation error depends on that choice.

The step above that is not automatically scale-invariant is step 1: `F₀ = U₀ D₀ U₀ᵀ`. Eigenvectors of a covariance matrix are not equivariant under a diagonal rescaling of the underlying variables, so *which direction is the smallest eigenfactor* — the direction the adjustment corrects hardest — can be decided by the units before it is decided by the data. Row 96 measures exactly that on this project's own panel: `cond(F) = 9.96e6` against `cond(corr) = 3.27`, with the whole of the gap attributable to a variance ratio of 7.3e6 across the mixed-unit factor set §4.1.1 deliberately specifies.

**This is not a defect in USE4 and the code must not imply that it is.** Every factor in USE4 is already in return units, so a diagonal rescaling across factors is not a thing that can happen there and the question never arises. Diagonalising `F` is exactly right in that setting. What is wrong is following the published step *literally into a case it does not cover* — a factor panel that deliberately mixes decimal returns with basis points of yield, which §4.1.1 chose in order to make a level-factor regression coefficient read as an effective duration directly.

The two candidate resolutions — adjusting the correlation matrix and rescaling back, or moving the panel to a common numeraire before `F` is formed — and the mechanical invariance test that settles the choice belong to **W3-P3** and are specified there. **Neither is implemented in W3-P1.** What is fixed here is the acceptance criterion any implementation has to meet: rescale any factor by any positive constant, run the adjustment, undo the rescaling, and the resulting covariance matrix must be unchanged to numerical tolerance. An implementation that fails that test is wrong whatever else it reproduces.

### 5.3.2 RULING (2026-08-31, W3-P3): the correlation-space resolution, and the tiebreak criterion stated before either was built

§5.3.1 fixed the criterion and left the choice of resolution open. This section takes it. **The tiebreak criterion is written first, and deliberately so** — a criterion recorded after two implementations exist can be reverse-engineered from whichever looked better, and this project's own category rules say that a choice made by comparing outcome numbers is a sweep, not a ruling.

**The criterion, fixed before anything was built: prefer whichever resolution keeps the units bookkeeping in fewer places.** W3-P2 established the reason on this project's own evidence — a scaling applied where later stages cannot see it is the most reliable source of silent error here, because a matrix that is uniformly wrong by a constant is still PSD, still symmetric and still identically conditioned, and passes every check the pipeline runs. Failure mode 7 twice over. Nothing about which resolution produces a better-conditioned matrix, a smaller correction or a nicer `λ` curve enters the criterion, and none of those quantities was computed for the losing candidate.

**The criterion selects (a): run the adjustment in correlation space and rescale by `σ` afterwards.** The count is not close.

- **(a) puts the units bookkeeping in exactly one place, inside one function, invisible from outside it.** `eigenfactor_adjustment` receives a covariance, splits it into `σ` and `ρ`, corrects `ρ`, and recombines with the same `σ` it split off. The split and the recombination are three lines apart in one file. Nothing upstream changes, nothing downstream changes, and `risk/` still receives a `T × K` frame and returns a `K × K` matrix in whatever units the caller handed it — invariant 10 intact, §15.2's signature untouched.
- **(b) — moving the panel to a common numeraire before `risk/` sees it — puts it in every place exposures and covariance meet.** The optimizer's `wᵀXFXᵀw`, §10.3's attribution, §5.5's specific risk, §6.1's bias statistic, and every report that quotes a factor volatility. Each is a site where a units error is silent, and §4.1.1's duration reading means `factors/` keeps basis points regardless, so (b) does not remove the conversion — it moves it to a boundary that is crossed many times instead of once.

**(b)'s arithmetic advantage is real and does not change the count.** Converting basis points to decimals collapses the 7.3e6 variance ratio to about 3.6 — `rates_slope` at 6.3 bp is 6.3e−4 in decimal against `credit`'s 2.3e−3 — and it is arithmetic rather than an estimated parameter, so it costs no degrees of freedom. That makes (b) a legitimate resolution and it is recorded as one. It loses on the criterion above, not on its arithmetic, and **it was not built and no outcome number was computed for it**, which is what keeps this a ruling rather than a sweep (`experiments.md`, W3-P3).

**Neither resolution touches `factors/`.** The panel keeps the basis-point convention §4.1.1 chose, so rows 64–89 and the effective-duration reading of the level coefficient stand unchanged.

#### What (a) obliges, stated because it is the part that can go wrong

**The adjusted matrix in correlation space is not a correlation matrix, and its diagonal must not be renormalised.** `γ²` scales the eigenvalues of `ρ`, which moves the trace and therefore the diagonal away from one. Renormalising it back would delete the correction: the eigenfactor adjustment exists to raise the risk of the low-variance directions an optimizer loads onto, and that increase arrives as exactly the diagonal inflation renormalising would remove. This is the mirror image of §5.2.1's double-counting trap — there the diagonal *had* to be renormalised, because Newey-West was correcting a term whose scale was already carried by `σ`; here it must not be, because the correction has nowhere else to live. **The two are three stages apart in the same file and a later session that makes them consistent with each other breaks one of them.** A test pins the diagonal as inflated rather than unit.

**The `σ` used to recombine is the one split off, unadjusted.** The correction is applied once, in correlation space; there is no second correction to the volatilities.

#### SUPERSEDED BY 5.3.3 — `T` was kept on the wrong authority

*W3-P3 kept SPEC §5.3's published `T = 242 / 727` and gave three reasons: the spec states it, comparability with Shepard's cross-check, and the direction being conservative. **All three fail, and §5.3.3 replaces them.** The paragraph is not deleted, because a reason that looked sufficient and was not is worth leaving visible.*

#### The λ(k) magnitude gate is WITHDRAWN, and what replaces it separates the universal part from the K-dependent part

§5.3 says `λ(k)` is *"empirically ≈1.5 at the smallest eigenfactor, declining monotonically to ≈0.95 at the largest."* **That amplitude is a large-`K` shape and is withdrawn as a gate.** It is the published curve for a model with dozens of factors, where `K/T_eff` is large enough to spread the spectrum that far. This model is `K = 6` at `T_eff = 242`, so `K/T_eff = 0.025` and Shepard puts the understatement at 5.1%; the curve should be *narrow*, nowhere near 1.5 to 0.95. Holding to the published amplitude would have a correct implementation diagnosed as broken — the fourth gate in this project calibrated for a case the project is not in, after §3.2's daily-correlation gate, §6.5.2's 0.3 correlation bar and §4.1.2's pairwise bar.

**What stays a hard gate is the direction, which is universal — AS AMENDED IN W3-P3b:**

> `λ(k)` **declines monotonically across the BULK of the spectrum**, sits **above 1** for the smallest eigenfactors and **below 1** for the largest of the bulk, and **returns toward 1 at any ISOLATED eigenvalue**.

A flat or non-monotonic *bulk* still means the implementation is wrong, at any `K`. That is the shape MWO derive rather than measure: small eigenvalues are biased down and large ones slightly up.

**The original wording — monotone in rank — is wrong for any spiked spectrum, and the amendment is measured rather than asserted.** The bias is driven by **eigenvector rotation**; an eigenvalue well separated from the bulk suffers almost none, so its own risk is forecast nearly without bias and `λ` comes back to ≈1 there, above the bulk's last value. `experiments.md` row 107 isolates it on a matrix built to contain exactly one spike: with no spike the curve is monotone 1.2091 → 0.8553; with one spike at 3× the bulk is monotone 1.1559 → 0.8852 and the spike sits at 0.9785. The golden fixture is `BB' + diag(ψ)` with a `K×2` loading matrix, so it has exactly two spikes, and its raw curve is monotone across all 38 bulk eigenvalues at `K = 40`, breaking only at those two.

> **PRE-REGISTERED (2026-08-31, W3-P3b), to be read in W7: the `K = 56` equity module WILL show this.** A cross-sectional equity panel is a spiked spectrum by construction — a dominant market factor, then industry and style structure separating from the bulk — so `λ` **is expected to return toward 1 at the dominant factor and at whichever industry factors separate**. **This is the expected result, not a defect**, and it is registered here so that week 7 does not spend a session diagnosing a correct implementation. The falsifiable half: if `λ` at `K = 56` declines monotonically **in rank** with no return toward 1 at the top, then either the equity spectrum is less spiked than expected or the mechanism above is wrong, and both are findings.

**The corollary belongs in the K-dependence table, and it changes what that table records.** The adjustment does **least** for the dominant factors and **most** for the bulk. So `reports/stage_k_dependence.md` records *where in the spectrum* each stage acts, not only whether it earns its keep: a stage that corrects the bulk and leaves the dominant factor alone is a different object from one that acts uniformly, and which of them matters depends entirely on where the portfolio sits. A minimum-variance portfolio lives in the bulk, which is why the correction bites there.

**The amplitude is pre-registered as a prediction instead of a gate,** because it scales with `K/T_eff`:

> **PRE-REGISTERED (2026-08-31, W3-P3), to be read in W7:** the `λ`-curve amplitude — `λ(0) − λ(K−1)` — is **visibly larger** in the `K = 56` equity module than in this `K = 6` macro model, on the same code and the same `T_eff`. Falsified if the amplitude is equal or smaller at `K = 56`.

It costs nothing now, has real power later, and gives the project a fourth independent corroboration that `K/T` is the governing parameter — after §15.2's table reproduced arithmetically in W3-P1, §5.2.2's PSD-repair boundary arrived at from a numerical failure mode, and §5.3's own bias magnitude.

#### The parabola at this K does almost nothing, and that is reported rather than assumed away

Step 7 fits a parabola to the raw `λ(k)` to smooth Monte Carlo noise at the ends of the spectrum. Plain degree-2 OLS in the eigenvalue **index**, which is what MWO do.

At `K = 6` that is three parameters fitted to six points: **3 residual degrees of freedom, so the smoothing is weak.** On the `K = 3` golden fixture it is three parameters on three points — an **exact interpolation that removes nothing at all**, and the fitted curve is the raw curve. Both are reported in `reports/eigenfactor.md` rather than left for a reader to assume noise was removed when it was not.

This belongs to a pattern that is now large enough to name, and §5.3's smoothing step is its third instance: **the USE4 pipeline's corrections are themselves `K`-dependent, and a small macro model pays for several that do nothing for it.** The PSD repair fires only at `K/T_eff ≥ 0.6` (§5.2.2); Newey-West on a serially uncorrelated panel spends variance for nothing (row 101); the parabola at `K = 6` smooths three degrees of freedom and at `K = 3` smooths none. `reports/stage_k_dependence.md` is the running table, appended to as each stage lands and carried into W8-P1 beside the `K/T` scaling result. It is a **second headline**: the project's first is that estimation error scales with `K/T`, and the finding accumulating beside it is that the industry-standard corrections for that error scale with `K` too.

### 5.3.3 RULING (2026-08-31, W3-P3b): there is no `T` — the simulation follows the estimator

**This section exists because §5.3 carried a defect and W3-P3 kept it.** §5.3 states `T = T_eff = 2τ/ln2` at the **volatility** half-life — 242 days and 727 — and W3-P3 kept those numbers after moving the adjustment into correlation space. The number was written for **covariance-space diagonalisation**, which §5.3.2 replaced. The three reasons W3-P3 gave all fail, and each is a plausible-sounding argument a later session could reuse:

1. *"The spec states it."* The spec stated it for a procedure that no longer runs. A number inherited from a superseded procedure is a leftover, not an authority — **and this is the reason that fails hardest, because it is the one that sounds least like a judgement call.**
2. *"Comparability with Shepard's cross-check."* Real, and it **cuts both ways**: a correction and a cross-check computed at a `K/T` that describes neither estimator are comparably wrong, not jointly right. Agreement between two quantities evaluated at the same wrong parameter is not corroboration — and W3-P3's reported agreement turned out to be exactly that (below).
3. *"The direction is conservative."* A **tiebreak, not a justification.** "It errs the safe way" is how a wrong number survives review, and §6.6 and the W1-P3 rows already forbid that shape of argument.

**The principle: `T` follows the matrix being diagonalised.** After §5.3.2 that matrix is `ρ̂`, estimated at the correlation half-life.

#### The resolution is a measurement, not a substitution

Putting 1454 in place of 242 because the argument says so repeats the error in the other direction. `T_eff = 2τ/ln2` is derived for an **equally weighted window's** effective size; whether it transfers to the **eigenvalue bias of a normalised EWMA correlation estimator** was an empirical question nobody had asked. So it was measured, in the shape of W1-P5c's EDGE recovery test: simulate from a known matrix, re-estimate through the production code path, and ask what the estimator delivers rather than what a formula says it should. `experiments.md` rows 109–111.

**The measurement refuted its own hypothesis, and the control registered beside it found out why.**

- **`2τ/ln2` transfers exactly — for the SECOND-MOMENT estimator.** Un-normalised, the equivalent equal-weight `T` is **1.007×** the Kish figure on this project's panel, and a flat-weight control (a half-life so long the weights are uniform, where the answer must be the window length) recovers **300 of 300 exactly**. Exponential weighting is not the problem.
- **Normalising to a correlation matrix is what breaks it.** The same flat-weight case normalised returns **400** instead of 300; the panel normalised returns **1890** instead of 1454. Removing the variance error from the diagonal removes part of the eigenvalue dispersion, so a *correlation* estimator behaves like a **longer** sample than its variance-matched size implies. Row 111 predicted this would be second-order; **it is the whole of the effect.**
- **The factor is not universal, so it cannot be a constant.** Across `K ∈ {3, 6, 40}` and half-lives 84d and 504d it runs **1.00 to 1.50** — stable at 1.30 for `K = 6` on two different matrices, 1.10 and 1.50 elsewhere. It depends on the spectrum, which is exactly what a config constant cannot.

#### The ruling

**The Monte Carlo simulates the estimator the pipeline actually uses.** Steps 2–3 draw `observations` rows and run them through `ewma_second_moment` at the correlation half-life and then `correlation_from_covariance` — the production path, not a model of it. Steps 1 and 4–8 are unchanged.

This **removes** the parameter rather than replacing it. There is no `T` in `config/model.yaml`, no calibrated correction factor, and nothing for a later session to get wrong. It is **asset-class agnostic by construction** (invariant 10): the simulation adapts to whatever panel and half-life it is handed, which a calibrated `T` could not do without being re-measured per panel. `K/T_eff` is still *reported* — it is the project's headline axis — but nothing consumes it.

**Three consequences, all recorded rather than discovered later.**

**1. The stage is now horizon-independent.** `ρ̂` is estimated at 504d at *both* horizons (§5.1, deliberately), so short and long now receive an identical eigenfactor correction. Under `T = 242/727` they differed, and **that difference was an artefact of the wrong sample size, not a property of the model.** A test pins the equality. It also withdraws a corroboration: W3-P3 quoted the two horizons' amplitudes (0.0741 against 0.0242) as evidence that the amplitude tracks `K/T`. That comparison no longer exists. **A corroboration that disappears when a defect is fixed was evidence for the defect, not for the claim.**

**2. W3-P3's Shepard agreement was two errors cancelling, and it is withdrawn.** It compared a revision computed at `T = 242` against a Shepard figure for the *volatility* window — a window this stage does not correct — and reported the match as a cross-check. Computed on the matched window the closed form gives **0.836%** against a minimum-variance revision of **+0.758% / +0.803%** at `a = 1.4`, which is the cross-check §6.4 actually asked for. **Two wrong numbers agreeing is the most dangerous kind of corroboration, because it reads as confirmation from an independent source.** The `Shepard / (a = 1.0)` ratio of 1.353 that was tempting to read as *deriving* MSCI's `a = 1.4` is now 1.55 and 1.46 and the reading is dead — it was labelled a coincidence rather than a finding at the time, on the grounds that it was not stable across horizons, **and that caution is the only reason no claim has to be retracted.**

**3. The stage's coverage changed with the numeraire, and §5.3.2 did not name it.** In covariance space the adjustment corrected the estimation error of `σ` and `ρ` **jointly**, because both live inside `F`. In correlation space it corrects `ρ`'s error alone: **`σ`'s estimation error passes through this stage uncorrected.** The two Shepard columns put a number on it — 5.14% at the volatility window against 0.84% at the correlation window — and the difference is, to order of magnitude, the uncorrected volatility leg. **This is a real cost of the scale-invariance ruling, it is larger than the correction the stage applies, and it is left open.** It belongs with §5.4's volatility regime adjustment, the other stage that acts on the level of risk, and W3-P4 must read this paragraph before writing it. **ANSWERED in §5.4.2 (W3-P4): the VRA absorbs it partly and invisibly, because it is fitted to realised data; the two stages correct different objects but both move the level, so W4-P3's disjointness demonstration is three-way. The overlap was measured rather than assumed.**

## 5.4 Volatility regime adjustment

The EWMA estimator is slow. When volatility jumps (Aug 2007, Sep 2008, Mar 2020) forecasts are too low for weeks. VRA is a fast level correction using the cross-section as an instantaneous volatility signal.

```
B_t^F = sqrt( (1/K) Σ_k (f_kt/σ_kt)² )              σ_kt forecast made at t−1
(λ_F)² = Σ_t w_t (B_t^F)²                            exponential w, half-life τ_VRA
F ← λ_F² · F
```

`τ_VRA` = **42 days (short) / 168 days (long)**. Scale the whole factor covariance matrix by `λ_F²` — it changes the level of risk, not the correlation structure.

The specific-risk multiplier uses a cap-weighted (here: notional-weighted) cross-sectional bias statistic across assets with the **same** `τ_VRA`. MSCI is explicit that equal half-lives for both components is deliberate — otherwise the factor/specific split drifts during regime changes.

### 5.4.1 RULINGS (2026-08-31, W3-P4): where `σ_kt` comes from, what is not clipped, and what the burn-in is not

Four decisions, taken while writing the stage, each with the alternative named rather than left implicit.

**1. `σ_kt` is the forecast from the FULL pre-VRA pipeline — stages 1 to 4, not 1 to 3.** §5.4 says *"the forecast made at t−1"*, and the model's forecast at `t−1` is what the pipeline produces after every stage preceding this one. The alternative — taking `σ` from the EWMA/Newey-West/repair legs and skipping §5.3 — was considered on cost grounds (it is roughly five hundred times cheaper, since it skips the Monte Carlo at every date) and **rejected for a reason that is not cost**: under §5.3.2's correlation-space resolution the eigenfactor stage moves `σ` only through the deliberately-inflated diagonal of `ρ_adj`, so taking `σ_kt` from stages 1–3 would make the eigenfactor stage *invisible* to the VRA. The overlap measurement below would then return zero **by construction rather than by finding**, which is the most expensive kind of null result: one that looks like evidence. Measured on this project's panel the inflation is 1.00004 to 1.0017 in variance — 0.002% to 0.085% in volatility — small, and the point is that it is not zero.

**2. Nothing is clipped, and §6.1's ±4 is deliberately not imported.** §6.1 clips standardized returns at ±4 before taking a standard deviation, because one 2020 observation otherwise dominates a 12-month window. §5.4 specifies no clip and none is applied (invariant 9). The two statistics are different objects: §6.1's is a standard deviation over a short rolling window where one outlier is most of the sample; this one is an exponentially weighted mean of `B²` over hundreds of effective observations, where the same outlier is a few per cent of the weight. Carrying a constant across sections because it is nearby is how a parameter gets invented without anyone deciding to. What replaces the clip is **reporting**: the largest single-day `B_t^F` is in `reports/volatility_regime.md`, so a reader can see how much of the multiplier one day carries.

**3. The burn-in stays with the caller, and §5.1.2's future bound is still future.** §5.1.2 fixed the shape of the burn-in rule — a bound on `K/T_eff`, never a day count — and this is the first session where matrices iterate through time, so it is the first session that *could* have set one. It does not. **There is no published bound and any value chosen here would be a guess dressed as a ruling** (invariant 9). What stands in its place is already built: W3-P1 made every build emit `K/T_eff`, so `ForecastHistory` carries the ratio at every forecast date and a matrix estimated at an unhealthy ratio is **labelled rather than suppressed**. W4's rolling harness owns the exclusion decision and owes a criterion written *before* the bias statistics are looked at — otherwise it is a sweep and owes `model-config` rows.

The one minimum the stage does enforce is arithmetic and not a burn-in: `max(K + 1, lags + 1)`. Below `lags + 1` §5.2's Bartlett sum has no pairs at the longest lag; below `K + 1` §5.3's Monte Carlo re-estimates a singular correlation matrix. These are points where a stage has no answer at all, not points where its answer is poor, which is the same distinction `reports/psd_repairs.md` scans from.

**4. The specific-risk leg is absent, not stubbed.** §5.4's second paragraph needs specific returns, which §5.5 has not built. The shared half-life already sits at `volatility_regime_adjustment.halflife` — in its own top-level section of `config/model.yaml` precisely so the factor and specific legs cannot drift apart — so the session that writes §5.5 reads the same key and adds no parameter.

### 5.4.2 AMENDMENT to 5.3.3 (2026-08-31, W3-P4): the disjointness demonstration is THREE-WAY, and the VRA has already absorbed part of the answer

§5.3.3's third consequence left the uncorrected `σ` leg open and told W3-P4 to record whether the VRA absorbs it. **It does, partly, and invisibly — and the reasoning matters more than the number, because the two stages are not doing the same thing.**

The VRA corrects for **regime**: the true level of volatility moving faster than an EWMA at an 84- or 252-day half-life can track it. That error averages out over a long enough sample; it is a timing problem. §5.3's eigenfactor adjustment and §6.4's Shepard closed form correct for **sampling error**: a systematic multiplicative understatement that does *not* average out, and that is present even in a stationary world with no regimes at all. Different objects, and a reading that collapses them is wrong.

**But this stage is fitted to realised data.** If the forecast is systematically 5% low for any reason whatsoever, `B_t^F` sits systematically above one and `λ_F²` scales the matrix up. That is not a defect in the VRA — it is what an empirical calibration *does*. The consequence is that `λ_F²` silently contains whatever level bias the stages before it left behind, including the part of Shepard's understatement that §5.3.2's correlation-space ruling declines to correct.

So **W4-P3's disjointness demonstration is three-way, not two-way.** Eigenfactor, VRA and Shepard all touch the overall scale, and the VRA is downstream of both. A demonstration that treats the eigenfactor adjustment and the VRA as independent corrections and adds their effects will double-count.

**W3-P4 measured the overlap rather than leaving W4-P3 to assume it**, in the shape §5.3.3 used for `T`: run the thing, do not argue about it. `λ_F²` is computed twice per horizon — once on a forecast history built with §5.3's stage on, once with it off — and the difference is the size of the absorption. `experiments.md` carries the numbers and `reports/volatility_regime.md` the table. **A measured overlap is what turns W4-P3 from an argument into an accounting**, and it costs one extra pass over a history that had to be built anyway.

**The absorption is third-order — 0.017% to 0.033% of volatility — and it scales exactly, which is what W4-P3 actually needs.** A nine-point grid of `a` from 0 to 2 shows `Δ(a) = λ_F²(0) − λ_F²(a)` is **exactly quadratic**, and not as a fit: `γ² = (1 + aδ)²` with `δ = λ_P − 1` makes `ρ_adj,kk(a) = 1 + 2aA_k + a²C_k` identically. `Δ(a)/a` is linear in `a` to **9 parts per million** of the leading coefficient. So W4-P3 can rescale the overlap to any correction magnitude rather than re-measuring it, and the 0.7–1.2% by which the observed `a = 1.4` / `a = 1.0` ratio exceeds 1.4 is the second-order term, accounted for to four decimal places at both horizons. Note also that `a = 0` **is** the eigenfactor-off control exactly (`γ = 1`, so `ρ_adj = ρ`), so the grid reproduces the independently-built control to nine decimals — a free check on both.

**What that ratio does NOT establish, stated because the opposite was briefly claimed.** It does not confirm the linear `γ = a(λ_P − 1) + 1` against the `λ^a(λ − 1) + 1` form one PDF rendering garbles §5.3's Eq. A8 into. Every number in the grid was produced *by* the linear implementation, so the ratio falls out of the code rather than adjudicating between two readings — **a measurement cannot discriminate against a form it never evaluated.** That is the same shape as W3-P3's Shepard agreement, withdrawn in §5.3.2 for the same reason, and the second time in this project. The ambiguity is settled where it already was: `tests/test_eigenfactor.py::test_gamma_is_linear_in_lambda_minus_one_and_not_a_power_law` evaluates **both** forms and asserts they are distinguishable on this data. That test discriminates because it runs the alternative.

### 5.4.3 FINDING (2026-08-31, W3-P4): Aug 2007 is not reachable, and that is the fifth gate written for a case this project is not in

§5.4 names Aug 2007, Sep 2008 and Mar 2020 as the episodes a working VRA must respond to. **The first is not on this panel and no claim is made about it.** The complete orthogonalized macro factor panel begins **2008-04-14**, bound by `credit` (first valid 2008-04-14) and `rates_slope` (2008-04-02) through the 252-observation expanding-window orthogonalization burn-in off `sample.start` = 2007-04-01 — a documented consequence of §4.1.3 and W2-P2's row 69, a decision taken *after* §5.4's sentence was written. Sep 2008 and Mar 2020 stand and are reported.

This is the **fifth** acceptance criterion in this project calibrated for a case the project is not in, after §3.2's daily-correlation gate, §6.5.2's 0.3 correlation bar, §4.1.2's pairwise bar and §5.3.2's `λ(k)` magnitude gate. The pattern is now frequent enough to be worth naming: **a gate written early, against a published example, survives into a build whose own later decisions have moved the case out from under it.** The failure it produces is not a false pass — it is a session spent diagnosing a correct implementation. Stating the exclusion and its cause in `reports/volatility_regime.md` costs a paragraph; a chart that quietly began after the window it claimed to cover would have read as a pass.

### 5.4.4 FINDING (2026-08-31, W3-P4): the PSD repair and the eigenfactor stage compound at `K/T_eff ~ 0.8`, and it is failure mode 7

Found while reading W3-P4's results, not predicted — recorded that way rather than dressed as a hypothesis that held.

Over the first four forecast dates the pipeline's volatility forecast is inflated by about **four orders of magnitude**. Traced stage by stage, stages 1–3 are sane and **the eigenfactor stage produces it**, on exactly the dates where §5.2's PSD repair has fired and on no others. The mechanism: the repair floors a near-zero eigenvalue of `ρ̂` at `1e-14`; §5.3's Monte Carlo then measures an enormous `λ(k)` for a direction carrying essentially no variance; and the **parabola fit spreads that through all six `γ(k)`**, which is why every factor inflates rather than one. Two corrections, each passing its own sanity check, compounding when applied in sequence — CLAUDE.md failure mode 7, in the pipeline the failure-mode list was written for.

Its reach is bounded and matches §5.2.2 exactly: four dates, `K/T_eff` from 0.857 to 0.667, and nothing from the date the repair goes quiet. Its cost to the reported `λ_F²` is **at most 1.1e-9**, because a 2008 date carries essentially no weight in an exponentially weighted sum ending in 2024.

**It is not fixed and no burn-in is invented to hide it** (§5.4.1). What it changes is the standing of §5.1.2's future bound: there are now **two independent arguments** for a `K/T_eff` minimum rather than one. The first is generic estimation error. The second is this — a specific interaction between two named stages, with a located mechanism, a measured reach and a measured cost. W4 still owes a criterion written before the bias statistics are read.

**A burn-in is the wrong remedy at `K = 56` anyway, so the right one is registered now — `experiments.md` row 118.** ≤1.1e-9 here is not a reason to leave the remedy to week 7: the repair is *predicted* to fire often at `K = 56` (`reports/stage_k_dependence.md` already carries "where it earns its place" as that stage's large-`K` cell), and **each firing corrupts all `K` values of `γ`, not one**, because the parabola is fitted across the whole spectrum. The damage is `K`-amplified twice — more firings, each spread wider — and a session meeting it as a broken build would be choosing a remedy under pressure, which is the shape §5.2.2's row 100 exists to prevent.

**The registered remedy: exclude floored eigen-directions from the `λ(k)` curve and the parabola fit.** Fit on the directions that carry an estimate; apply the fitted curve to all of them. It **invents no parameter** — a floored eigenvalue is `psd_eigenvalue_floor`, a constant the repair *wrote*, not a quantity the estimator measured, and fitting a curve through it treats a manufactured number as data. Which directions were floored is already recorded by `RepairReport`, so the exclusion is a fact the pipeline carries rather than a threshold anyone picks. It is the same class as the W4-P2 forward constraint excluding the two identity assets from bias aggregation: not a filter on values, but the removal of things that were never observations.

**And it carries a falsifier for the diagnosis, not only for the remedy.** At `K = 6` the repair fires on four dates and floors one direction, so the exclusion must leave every other date bit-identical. **If it changes `γ` materially anywhere else, the interaction is not confined to floored directions and the mechanism recorded above is mis-identified** — the remedy is then void and the cause has to be found again before anything is excluded. This section is not a licence to exclude; it is a hypothesis with a way to be wrong.

The pairing also has a row of its own in `reports/stage_k_dependence.md`, entered as a **stage pairing rather than a stage**. That distinction is the point: both stages pass their own checks and the failure exists only in their composition, so a table organised one-row-per-stage could not represent it — which is why this was found by reading a chart rather than by reading that file.

## 5.5 Specific risk

With only 15 assets the full Barra apparatus is partly overkill, but implementing it correctly is exactly the signal the project exists to send. Four stages:

**(a) Time-series estimate.** EWMA of squared specific returns, half-life **84d (short) / 252d (long)**, plus a Newey-West term over **5 lags** with autocorrelation half-life 252d.

**(b) Structural estimate.** Regress log time-series specific vol on factor exposures:
```
ln σ_n^TS = Σ_k X_nk b_k + ε_n
σ_n^STR   = E₀ · exp( Σ_k X_nk b_k )
```
`E₀` is a multiplicative bias correction removing the log-transform (Jensen) bias. Fit on assets with clean complete histories, apply to all — this is how a series with 30 days of history gets a forecast, and it is what makes the pipeline handle a new ETF gracefully.

**(c) Blend.** `σ̂_n = γ_n σ_n^TS + (1−γ_n) σ_n^STR`, with `γ_n ∈ [0,1]` a data-quality weight driven by missing observations and by return-distribution fatness (compare a robust vol estimate to the raw one).

**(d) Bayesian shrinkage.**
```
σ_n^SH = v_n σ̂_n + (1 − v_n) σ̄(s_n)

              | σ̂_n − σ̄(s_n) |
v_n = ─────────────────────────────────      q = 0.1
      | σ̂_n − σ̄(s_n) | + q·σ_Δ(s_n)
```
`s_n` = the asset's bucket (Barra uses size deciles; here use **asset class** — equity / rates / credit / commodity / FX), `σ̄` = notional-weighted mean specific vol in the bucket, `σ_Δ` = its cross-sectional dispersion, **`q = 0.1`**.

Read the formula carefully: the further an estimate is from its bucket mean *relative to that bucket's natural dispersion*, the **less** it is shrunk. That is backwards from naive shrinkage and it is deliberate — the target is estimation error, not genuine heterogeneity. Extreme observed specific vols are disproportionately noise and mean-revert. Some PDF renderings garble this into a squared form; the linear-absolute-deviation version above is correct.

Then apply the specific VRA multiplier.

**Known false assumption to state explicitly:** specific returns are assumed cross-sectionally uncorrelated, so specific risk is diagonal. This is false for linked assets — SPY/IWM residuals are not independent, nor are LQD/HYG. MSCI handles it with linked-asset overrides that are not publicly documented. Report the residual correlation matrix and note where the diagonal assumption is violated; do not pretend it holds.


### 5.5.1 RULINGS (2026-09-01, W4-P1): four decisions §5.5 leaves open, and none of them is a parameter

§5.5 is written for a large cross-sectional universe and five of its terms are undefined at `N = 13`. All five are settled here on structural arguments, and in each case the alternative was **deliberately not built** — so none of them is a sweep and none adds to the deflated-Sharpe trial count. `experiments.md` carries the reasoning at the rows.

1. **The buckets are `config/universe.yaml`'s six sleeves, unmapped.** §5.5 names five classes for `s_n` — equity / rates / credit / commodity / FX — as an illustration of what a bucket *is*, adapted from Barra's size deciles. This project has its own partition: dated, frozen, with a rationale per line. Remapping it onto §5.5's list would be the **sixth** time in this build that a shape written for one universe is forced onto another, and the previous five all ended in the same finding. Counts: `equity` 4, `government` 4, `credit` 2, `commodity` 2, `inflation` 1, `currency` 0 — the last because `dollar` is the sixth *factor* rather than an asset in the residual panel.

2. **Every cross-sectional mean is EQUALLY weighted.** §5.4 and §5.5 both ask for notional weighting. **No notionals exist**: no portfolio has been constructed at this point in the project, and `config/universe.yaml` is right not to invent any. Equal weighting is the *absence* of a weighting rather than a chosen one, it is emitted in the report beside every number it touches, and every specific-VRA figure is labelled equal-weighted. **A later session with real holdings that revisits this owes a `model-config` row at the moment of switching**, not at the end of the project.

3. **The structural design matrix carries an intercept.** §5.5(b) writes `ln σ_n = Σ_k X_nk b_k` with no constant. In a Barra model that is complete, because the exposure matrix already carries a unit column through its market or country factor. §4.1's does not, so without an intercept the log fit is forced through the origin and cannot represent a common level of specific volatility. An intercept is a **design-matrix column, not a parameter** — it introduces no number anyone chose, and no `R²` could reverse it, which is the W3-P1 criterion for a ruling rather than a sweep. The no-intercept fit is reported as a diagnostic (`R² = 0.684` against `0.833`) rather than carried as a candidate.

4. **A value that is a construction artefact is held out of the shrinkage target** (added W4-P1b; see §5.5.4). Every asset is bucketed, shrunk and reported; which values may *build* `σ̄` and `σ_Δ` is a separate question, and `mafrm.risk.specific` takes the answer from its caller because it cannot know which values are artefacts. Not a parameter and not a filter on magnitudes — it removes manufactured numbers and substitutes no estimate.

5. **`σ_Δ` is the population deviation, `ddof = 0`.** A bucket is the whole of itself rather than a sample drawn from something larger, so no degree of freedom is spent estimating its mean from outside it. At two members the two conventions differ by `√2`, which is large enough that the choice has to be stated rather than inherited from a default.

Two further constructions are **derived rather than chosen**, and are recorded here so that a reader does not go looking for them in `config/model.yaml`: the robust scale is `IQR / (2·Φ⁻¹(0.75))`, computed from the normal quantile at run time rather than written as 1.35; and `E₀` is the ratio of means `mean(σ^TS) / mean(exp(Xb))` over the fit set, which removes the Jensen bias without assuming anything about the shape of the log residuals. The log-normal closed form `exp(s²/2)` is **computed and reported beside it rather than applied** — it is exact under log-normality, which 13 points cannot support — so the gap between the two is visible instead of implicit (1.178 against 1.347 at the short horizon).

### 5.5.2 FINDING (2026-09-01, W4-P1): §5.5(c)'s fatness term does not saturate, and `γ_n` is pinned on an INVERSION rather than on a regime

**This is the one place in §5.5 where a published constant was needed and was not available.** §5.5(c) specifies `γ_n` qualitatively — *"a data-quality weight driven by missing observations and by return-distribution fatness"* — and names no formula and no constants; `config/model.yaml` has none. CLAUDE.md invariant 9 forbids reciting them from memory, and a confidently-stated wrong constant is worse than an admitted gap.

The route taken was to show the weight **saturates**, making the constants irrelevant. It half worked, and **the half that failed was pre-registered by the operator with its own falsifier before the run**, which is what makes this a result rather than an observation.

| Term | Saturates? | Evidence |
|---|---|---|
| missing observations | **YES** | the residual panel is complete-cases: `min h_n = 3,889` of 3,889, **zero** missing observations. Any ramp ceiling at or below 15.5 years returns 1 whatever its value |
| distribution fatness | **NO** | `max Z = 7.61` (short, `hy_credit`, 2020-04-06) and `5.57` (long, `hy_credit`, 2013-04-04). **4 of 13** assets breach `Z > 1`; **13 of 13** breach `Z > 0.2` |

**The premise conflated two different terms.** Completeness is a statement about *how many* observations there are; fatness is a statement about *what they look like*. A complete history of a fat-tailed series is still fat-tailed, so no amount of completeness can saturate the second term. The breach is largest on the two identity assets, whose residuals are ~0 by construction, but **it survives dropping them**: `ig_credit` reaches 3.009 and `tips_10y` 1.523.

**RULING: `γ_n = 1`, and the reason is that the term is INVERTED here rather than out of regime.** `γ_n < 1` routes weight off leg (a) and onto leg (b). That is worth doing when the structural estimate is the better-identified one; at `N = 13` it is the worse one — `R² = 0.833` on **6 residual d.o.f.**, with `σ^STR` missing `σ^TS` by **0.31× to 4.17×** — so the mechanism moves weight the wrong way *even implemented exactly as published*. The clinching detail is **where** it fires: March–April 2020, which is precisely where leg (a) is fat-tailed and therefore precisely where leg (b) is fitted on the same stressed data. **Constants would not have rescued it**, which is why this is not an invariant-9 blocker deferred to a later session.

Same shape as §4.2.6 and settled the same way — out of regime, the degenerate value is the correct answer and not a lazy assumption — but the *mechanism* is different and the distinction is load-bearing. §4.2.6's `σ² = 1` is a technique finding nothing to do. This is a technique doing something, in the wrong direction.

**The code path is kept and both diagnostics run live, with `γ_n` pinned rather than the leg deleted.** Same treatment as `a = 1.0` and `a = 1.4`: the mechanism stays exercised so that week 7 can turn it on where it belongs. `Z_n` is **reported as a reliability flag** rather than discarded — it correctly identifies where the time-series estimate is unreliable, which is real information about this panel whether or not a weight is computed from it.

**W7 PREDICTION, REGISTERED BEFORE THE PANEL EXISTS.** At `N ≈ 500` the structural model is well identified, so the blend should route weight usefully and `γ_n < 1` should **improve** the specific-risk forecast rather than degrade it. **Falsified if it fails to improve at the equity panel's `N`**, in which case the inversion explanation is wrong and the degradation measured here needs another cause. This is the sixth entry in the `K`/`N`-dependence family and it is registered-not-counted, on the same footing as rows 100 and 118.

### 5.5.3 FINDING (2026-09-01, W4-P1): the specific VRA absorbs most of §5.5(a)'s Newey-West correction — failure mode 7 on the production path

Pre-registered before the run, with the falsifier written first. §5.5(a)'s Bartlett term **lowers** `σ^TS`; §5.4's multiplier is fitted to realised data and therefore **raises** the level back. Rebuilding the forecast history with `lags = 0` — a **control**, not a candidate, since the lag count comes from USE4 Table 4.1 and no result here would change it — measures how much of the first stage the second one undoes.

| horizon | `λ_S` (lags=5) | `λ_S` (lags=0) | ratio | per-asset NW range | median | registered falsifier |
|---|---|---|---|---|---|---|
| short | 1.152640 | 0.946820 | **0.8214** | [0.5862, 1.0175] | 0.8595 | **HOLDS** |
| long | 1.017256 | 0.915778 | **0.9002** | [0.7726, 1.0184] | 0.8977 | **HOLDS** |

**Read the signs, because that is where the finding is.** Without the Bartlett term the model *over*-forecasts and the VRA marks it down (`λ_S = 0.947` short); with it the model *under*-forecasts and the VRA marks it up (`1.153`). The two stages push in opposite directions and **the VRA undoes 82–90% of what the Newey-West stage did**. At the long horizon the absorption is near-exact: a ratio of 0.9002 against a median correction of 0.8977, a gap of **+0.0025**.

This is CLAUDE.md **failure mode 7** — *"each adjustment passes its own sanity check; applied in sequence they compound or cancel"* — observed on the production path rather than anticipated. It does **not** follow that the Bartlett term is worthless: the two stages correct different objects, and the VRA is fitted rather than derived, so what is measured is that the fitted stage has already priced in whatever level the stage before it left. That is the same relationship §5.4.2 recorded between the VRA and the eigenfactor stage, **arriving a second time on a different pair of stages**, and it is the reason §5.4.2's disjointness demonstration is now four-way rather than three-way for anything that touches the specific leg.

### 5.5.4 AMENDMENT (2026-09-01, W4-P1) to the W4-P2 forward constraint: excluding the identity assets from bias aggregation is NOT sufficient

The forward constraint registered in W2-P2 excludes `hy_credit` and `commodity` — the two assets that ARE factors, at `R² = 1.000` by construction — from bias aggregation, and it is applied here: neither enters the specific VRA's cross-section. **That is not enough, and the gap was found by reading stage (d)'s output rather than by re-reading the constraint.**

Stage (d) shrinks toward a **bucket mean**, and the identity assets are members of buckets whose other members are not degenerate. So they contaminate the assets the constraint retains, through `σ̄`:

| horizon | asset | with the identity assets | without | change |
|---|---|---|---|---|
| short | `gold` | 65.4043 | 68.5084 | **+4.75%** |
| short | `ig_credit` | 9.4896 | 9.9289 | **+4.63%** |
| long | `gold` | 61.9465 | 64.8836 | **+4.74%** |
| long | `ig_credit` | 15.5727 | 16.2876 | **+4.59%** |

And the distortion to the identity assets themselves is an order of magnitude larger, because stage (d) pulls a specific volatility that is ~0 by construction toward a bucket mean set by a genuinely volatile partner: `commodity` goes from **0.2165 to 3.3206 bps/day, +1434%**, and `hy_credit` from 0.2631 to 0.7025, **+167%** (short horizon; the long horizon gives +1093% and +128%).

**AMENDED AND ACTED ON (2026-09-02, W4-P1b). The constraint is widened rather than the symptom patched.** The two identity assets are now held out of stage (d)'s `σ̄` and `σ_Δ` as well as out of bias aggregation. The justification is `experiments.md` row 119's, exactly: their residuals are ~0 by construction, so their specific volatility is a **construction artefact rather than an estimate**, and a shrinkage target computed partly from one is a target computed partly from a manufactured number. It **removes a manufactured number and substitutes no estimate**, so no result can reverse it — which is why the row is `data-diagnostic` and why nothing was re-bucketed.

The amended constraint, stated so a later session inherits the general form rather than this instance: **an asset whose value is a construction artefact must be excluded from every statistic that pools across assets — any aggregate, and any shrinkage target — not only from an average taken at the end.** Excluding a corrupted value from a mean does not uncorrupt the values it corrupted; by the time the average is taken, the artefact has already propagated.

What the fix is worth, short horizon: `gold` **65.4043 → 68.5084** and `ig_credit` **9.4896 → 9.9289**, both artefacts removed; `commodity` **3.3206 → 0.2165**, the +1434% distortion gone; `hy_credit` 0.7025 → 0.2631. `λ_S` moves toward one at both horizons — **1.1526 → 1.1417** short and **1.0173 → 1.0090** long — which is the expected direction and a check rather than a target: the two retained assets were being *under*-forecast by ~4.6%, and they are in the VRA's cross-section where the identity assets are not.

**It costs stage (d) two of its five buckets, and that is stated rather than netted off.** `commodity` and `credit` each had two members, one an identity asset, so both now have one-contributor targets and are no-ops by §5.5.5's arithmetic. **Stage (d) is live on `equity` and `government` alone — 8 of 13 assets.** The alternative was a target built partly from a manufactured number, which is not a better position; it is a worse one that looks fuller.

C2 has changed job as a result. It compared "identity assets in the bucket mean" against "identity assets deleted from the universe" and found a 4.6–4.8% gap. Under the amendment the production path holds them out of the *target* while keeping them bucketed and shrunk, and a target computed from a set is the same number whether the non-contributors are present or absent — so the two sides must now agree **exactly**. They do, to every printed digit, at both horizons. **The control that measured the defect is now the assertion that it is gone**, and `tests/test_specific_risk.py` pins the same equality on synthetic input.

### 5.5.5 FINDING (2026-09-01, W4-P1): §5.5(d) at a two-member bucket is data-independent, and the result is exact

In a bucket of two, each member's distance from the mean is half the gap between them, and the population dispersion `σ_Δ` is **also** half the gap. So `|σ̂_n − σ̄| = σ_Δ` identically, and

```
v_n = σ_Δ / (σ_Δ + q·σ_Δ) = 1/(1+q) = 0.9091     for EVERY pair of values
```

Stage (d) at two members therefore carries **no information about the data at all**: it is a fixed 9.09% proportional pull toward the bucket mean, whatever the two volatilities are. At one member it is the identity — `σ̄ = σ̂`, the numerator vanishes, and the `0/0` is removable because the map returns `σ̂` for every `v`.

The count that matters is **contributors to the target, not members**, and after §5.5.4's amendment those differ: `credit` and `commodity` are two-member buckets whose second member is an identity asset held out of the target, so both have one contributor and are no-ops; `inflation` is a singleton. **Only `equity` and `government`, at four contributors each, leave stage (d) anything to estimate — 8 of 13 assets.** The formula's whole content — *how far is this estimate, relative to how spread out this bucket is* — needs at least three members before the two quantities stop being the same number. This is not an empirical observation about this panel; it is arithmetic, and `tests/test_specific_risk.py` pins it.

### 5.5.6 ACCEPTANCE (2026-09-01, W4-P1): §15.2's specific-risk signature, met verbatim

§15.2 specifies

```python
def specific_risk(residuals, exposures, buckets, config: RiskConfig) -> pd.Series: ...
```

and `mafrm.risk.specific.specific_risk` has exactly that signature, in that order, returning a labelled `pd.Series`. `mafrm.risk` gained one module and `tests/test_risk_architecture.py`'s acceptance grep — no `equity`, `macro`, `ticker` or `asset_class` anywhere under `risk/`, comments and docstrings included — passes over it unchanged. The buckets arrive as **opaque labels** and are grouped by equality; a test asserts that renaming every label leaves the output identical, which is the mechanical form of the claim that this module cannot know what it is looking at.

Two things about the specific leg make it **cheaper than the factor leg rather than a copy of it**, and both are worth recording because they are what week 7 inherits:

* **There is no cache and none is needed.** §5.4's factor history costs ninety minutes because §5.3's Monte Carlo runs once per date, so it is checkpointed, fanned out and committed. The specific history costs about **nine seconds** a variant, because §5.5 has no Monte Carlo in it, so it is rebuilt on every `make report` and carries no staleness hazard.
* **The estimator is evaluated on an `O(N T)` path rather than an `O(N² T)` one.** §5.5 needs only the diagonal of the Bartlett sum, and forming an `N × N` matrix to discard it does not survive the move to `N ≈ 500`. The **conventions** are shared — one Bartlett kernel, one lag-pair weighting — and `tests/test_specific_risk.py` asserts the diagonal path reproduces `np.diag` of the matrix path to a derived rounding bound, because a second evaluation that is never checked against the first is a second estimator.

`make model` is **GREEN as of this session, for the first time in the project**: `mafrm.build._UNBUILT` had one entry left, §5.5, and it was removed by building the thing.

---

# 6. The validation battery

**This is the centrepiece.** Almost no open-source project ships a rolling bias-statistic chart, and it is the single most recognisable "this person has actually built a risk model" signal.

## 6.1 Bias statistics

```
b_nt = R_nt / σ_nt                       σ_nt forecast made at t−1 — out of sample by construction
B_n  = sqrt( (1/(T−1)) Σ_t (b_nt − b̄_n)² )
```

`B` is the ratio of realised risk to predicted risk. `B ≈ 1` calibrated, `B > 1` under-forecasting, `B < 1` over-forecasting.

Rolling 12-month version is what MSCI actually plots. The scalar summary for comparing models:
```
MRAD = (1/N) Σ_n | B_n(τ) − 1 |
```

**Clip standardized returns at ±4 before taking the standard deviation** (ddof=1) — one 2020 observation otherwise dominates a 12-month window.

**Confidence band:** `B ∈ 1 ± 1.96/√T`.

| T | Band |
|---|---|
| 12 (one rolling window) | ±0.57 |
| 60 | ±0.25 |
| 120 | ±0.18 |
| 250 | ±0.12 |
| 504 | ±0.09 |

**A single rolling reading of 1.4 is not evidence of miscalibration** — only a persistent level shift is. This is the most commonly misused part of the methodology and saying so in the README will separate you from people who have only read a blog post about it. Returns are fat-tailed, so treat the normal-theory interval as a floor on your uncertainty, not a hard test; report a χ² interval alongside it.

### 6.1.1 RULING (2026-09-02, W4-P2): the χ² interval decides, the normal band is a display convention, and daily is primary

Three decisions taken before the first bias statistic was computed. The second dissolves an ambiguity in this document rather than resolving it.

**1. The frequency is daily, with monthly reported alongside.** `config/model.yaml` already says so — the note at `numerics` names *"daily bias statistics (SPEC.md 6.1)"* — and that is where the power is: the scored window gives `T` in the thousands against roughly 180 monthly observations, and §6.1's own table shows a 12-observation window cannot separate `B = 1.0` from `B = 1.5` at all. A band that cannot make that distinction cannot carry this project's headline. Monthly is reported beside it because it is what MSCI plots and because a monthly-horizon consumer of the model is real; the monthly construction fixes portfolio weights at the start of each month and holds them, because a month's return is only the return *of a portfolio* if the portfolio was not rebuilt inside it.

**2. §6.1's `1 ± 1.96/√T` is a DISPLAY CONVENTION and nothing rests on it.** That line is a normal approximation and its constant is arguable: the sampling standard deviation of a standard deviation is `σ/√(2(T−1))`, so whether the denominator should be `√T` or `√(2T)` is a live question, and the two differ by a factor of 1.41. **The line was written by hand and this section is not going to let a claim depend on which of two defensible constants it happened to carry.**

What replaces it is exact rather than better-approximated. Under the null that the model is calibrated, `b_nt` is standard normal, and the ddof=1 estimator §6.1 specifies gives

```
(T − 1) B²  ~  χ²_{T−1}          exactly
```

so inverting the two-sided interval gives `B ∈ [√(χ²_{α/2,T−1}/(T−1)), √(χ²_{1−α/2,T−1}/(T−1))]`. **The χ² interval decides every claim; the normal band is drawn on the chart because §6.1 asks for it and is labelled a display convention there.** The coverage is 0.95 and is *read off* §6.1's own 1.96 — the two-sided 95% normal quantile — rather than chosen, so no level is invented either.

The two are not interchangeable and §6.1's own worked example is where they part company. At `T = 12` the band is `[0.434, 1.566]` and the exact interval is `[0.589, 1.412]`. So *"a single rolling reading of 1.4 is not evidence of miscalibration"* is **true under both**, and the margin is not: 1.4 is comfortably inside the band and a hair inside the exact interval, and a reading of 1.45 is inside one and outside the other. The exact interval is also **asymmetric**, which the normal band cannot be, and most of the difference is below one.

§6.1's fat-tail caveat stands unchanged and is repeated at every use: the χ² result inverts a normal null, so it too is a floor on the uncertainty rather than the truth. What the exactness buys is that the floor no longer depends on a constant this document chose.

**3. Every chart and table states its own `T`.** `±0.57` at `T = 12` and `±0.12` at `T = 250` are the same statistic at two window lengths and are trivially confusable; a `B` quoted without its `T` is not a reading. The overlap of any rolling stepping is stated beside it (CLAUDE.md failure mode 9), and no count of rolling windows is ever quoted as an `n`.

## 6.2 The test that matters — optimizer-selected portfolios

Run bias statistics on **four** portfolio families, and the fourth is the whole point:

1. **Individual assets** — sanity check.
2. **Random portfolios** — 100 dollar-neutral portfolios with N(0,1) weights, redrawn each period. Expect B ≈ 1.0 for *every* covariance variant including the naive sample estimator.
3. **Eigenfactors, ranked by eigenvalue** — expect 1.5 at the smallest declining to 0.95 at the largest before adjustment, flat ≈1.0 after.
4. **Optimizer-selected portfolios** — minimum-variance subject to `αᵀw = 1`, and minimum-variance fully-invested, rebuilt each period from the then-current forecast. Expect 1.2–1.5 before adjustment.

**Validating only on random or benchmark portfolios will tell you your model is fine when it is not.** That sentence, and the table demonstrating it on your own data, is the intellectual core of the project. The gap between family 2 and family 4 *is* the risk-model term in the §1 identity.

### 6.2.1 RULING (2026-09-02, W4-P2): the α-constrained portfolio is DEFERRED to W6, and `α = 1` would not have been a workaround

§6.2's family 4 names two portfolios: minimum variance subject to `αᵀw = 1`, and minimum variance fully invested. **Only the second is built here.**

`α` is unspecified. §6.3 quotes MWO at *"`α_P = 1`"*, which names the constraint's right-hand side and not the vector, and nothing in this document or in `config/model.yaml` says what `α` is. CLAUDE.md admits expected returns **only** as a fixed, documented input to the optimizer, and that input does not exist until W6.

**The obvious workaround is not one.** With `α = 1` the constraint `αᵀw = 1` becomes `Σᵢwᵢ = 1`, which is the fully-invested constraint — so the two named portfolios would be the *same portfolio* and the α-constrained row would be a duplicate column reported as a second family member. `α` therefore has to be a signal, and inventing one here would be both a parameter this project made up (invariant 9) and the beginning of return forecasting, which CLAUDE.md forbids outright.

**Nothing is lost from the headline.** The comparison §6.2 turns on is families 2 and 4 on the same matrix, and minimum-variance fully-invested carries the entire mechanism on its own: `w ∝ Σ⁻¹1` loads hardest on the smallest eigen-directions of the forecast, which are exactly the directions in which sampling error makes eigenvalues too small. W3-P3 already measured the size of that dependence from the other side — the eigenfactor adjustment moves minimum-variance forecasts **14× more** than it moves random ones.

**Registered for W6**, where it belongs with §8.3's alpha–risk misalignment. The two are the same problem seen twice: a portfolio built from `α` tilts along `α_⊥`, whose risk the model cannot see, so the α-constrained bias statistic and the misalignment angle are readings of one thing. §8.3's problem has already surfaced early once, in W2-P2's SPY partial coefficient.

### 6.2.2 RULING (2026-09-02, W4-P2): the headline `B` is the PRE-VRA one, because the post-VRA one is close to circular

**§5.4 sets `λ_F² = Σ_t w_t (B_t^F)²`.** The volatility regime adjustment is fitted to realised data expressly to drive a bias statistic to one. So a post-VRA `B ≈ 1` is substantially **what that stage was built to produce**, not evidence that the risk model is calibrated.

It is not perfectly circular, and the reason matters: `λ_F²` is a lagged exponentially weighted mean over the forecast history available at `t−1`, and the statistic it is compared against is out of sample, so `B` is not identically one. It is close enough that the project's central claim cannot rest on it.

**Both are reported, and each is labelled with the question it answers.**

* **Pre-VRA `B` measures whether the risk model understates risk.** That is the project's thesis and the `(1 − 1/B)` term of §1's identity.
* **Post-VRA `B` measures whether the regime adjustment generalises out of sample.** A different and lesser question.

Conflating them would let the project claim its central result from a number engineered to produce it.

**The specific leg is in the same position, from the other side.** W4-P1 measured the specific VRA absorbing **82–90%** of §5.5(a)'s Newey-West correction, with the sign of `λ_S` reversing across it (0.947 → 1.153 short). So both legs of a fully-adjusted forecast are close to circular with respect to `B`, which is why the pipeline-as-specified row is reported **last** rather than first.

**And the size of what is at stake was measured before this session.** W3-P4's `λ_F` is **1.0162 short and 0.9204 long**: the long-horizon model over-forecasts by 8%, which is what a 252-day volatility half-life does to a panel containing 2008 and 2020 — slow to come down after a crisis, carrying crisis variance through the calm that follows. On the long horizon the VRA therefore scales the forecast **down** by 8% while §5.2's Newey-West has already scaled it down by 1.7%, so roughly **10% of downward forecast pressure sits in the pipeline before any realised return is read**, all of it pushing `B` up toward the project's own claim. That is not a defect — it is published methodology applied as specified — and it is the reason §6.2.3 exists.

### 6.2.3 RULING (2026-09-02, W4-P2): `B` is ATTRIBUTED to pipeline stages, and the attribution is not a selection

**A raw `B` of 1.15 cannot be read as "the risk model understates by 15%."** Two published corrections in §5 move the level of the forecast, both of them were measured before any bias statistic existed, and both happen to point the same way:

* **§5.2's Newey-West is net negative on this panel.** W3-P2 measured the median volatility ratio at **0.9653 short and 0.9828 long**, with `rates_slope` down to 0.79. A lower forecast raises `B` — by roughly **30×** more than W3-P1's zero-mean convention pushes it the other way.
* **§5.4's VRA is net negative at the long horizon**, by the 8% recorded in §6.2.2.

So the report states `B` for the pipeline exactly as specified **and, beside it, what each stage contributed**, so that a reader can see the understatement that survives when the published corrections are accounted for.

**This is attribution, not selection, and the distinction is load-bearing.** The pipeline stays exactly as §5 specifies. Nothing is chosen on the basis of what `B` comes out as, no stage is disabled in the production path, and the ladder varies one stage at a time from a fixed base point rather than searching a grid. The signs of both effects were put on file in W3-P1 and W3-P2 **before** any bias statistic was computed, precisely so that this accounting could not be arranged after the fact.

### 6.2.4 REGISTRATION (2026-09-02, W4-P2): the burn-in, written before the statistics were looked at

§5.4.1 ruling 3 says the W4 harness owns the exclusion decision and **owes a criterion written before the bias statistics are read** — otherwise it is a sweep and owes `model-config` rows. §5.1.2 fixes its shape: a bound on `K/T_eff`, never a day count. This is that criterion.

**A date is scored when every variant has a forecast that exists.** Three conditions, each of them *"the estimator has no answer"* rather than *"the estimator's answer is poor"*, and not one of them a number anyone chose:

1. **It is at or after the last date §5.2's PSD repair fired.** Before that the repair has written `psd_eigenvalue_floor` into the spectrum, the matrix was not a covariance matrix, and §5.3's Monte Carlo raises on the Cholesky of the correlation it produces. `reports/psd_repairs.md` measured this boundary to be a `K/T_eff` boundary — it lands in the same place at both horizons though their volatility half-lives differ by a factor of three — which is exactly §5.1.2's required shape. Which dates those are is a fact `RepairReport` already records. Excluding them removes manufactured numbers and substitutes no estimate: the same class as `experiments.md` row 119's.
2. **The naive asset-level comparand is of full rank**, `T > N`. Below it the sample covariance is singular by construction and there is nothing for `Σ⁻¹1` to solve. Arithmetic, in the sense §5.4.1's `max(K+1, lags+1)` is.
3. **§5.4's multiplier exists**, which needs at least one earlier forecast date, since `λ²` at `t` is a weighted mean of `B_s²` for `s < t`. Costs exactly one date.

**"Last firing" and not "did not fire".** On this panel the firings are *interleaved* with non-firing dates rather than forming the single contiguous episode `reports/psd_repairs.md` found on the macro factor panel, and a history that skipped the firing dates in the middle would have interior gaps — which §5.4's exponentially weighted sum cannot take and which `history.Cache.bias` refuses outright. The scored history is the contiguous tail on which the estimator is well posed throughout.

**The window is common to every variant.** A comparison of families across variants scored on different date sets is not a comparison, and letting each variant start where its own estimator became well behaved would let the worst-behaved one look best by being scored on the calmest sample.

**What this is not.** It is not a claim that the retained early dates are well estimated — `K/T_eff` runs high at the start of the window and Shepard's `[1 − K/T_eff]⁻²` is enormous there. Those dates are **labelled rather than suppressed**, which is §5.4.1 ruling 3's own treatment, and the report carries the `K/T_eff` profile beside the chart.

### 6.2.5 FINDING (2026-09-02, W4-P2): H2 holds, H3 is REFUTED, and the reason is that the bias was never in `F`

**The headline, on the pre-VRA forecast, `a = 1.0`, short horizon, `T = 3,874` scored dates.**

| Family | `B` | Exact 95% interval at `T = 3,874` |
|---|---|---|
| 1 — individual assets | 1.0245 | `[0.9777, 1.0223]` |
| 2 — random dollar-neutral | **1.0101** | inside, at the edge |
| 3 — factor eigen-portfolios | 1.0415 | just outside |
| 4 — **minimum variance** | **1.3321** | fifteen half-widths outside |

**The gap between family 2 and family 4 is +0.322 on the same covariance matrix** (+0.315 long). That sentence, on this project's own data, is what §6.2 exists to produce: random portfolios say the model is fine; the portfolios an optimizer actually picks say it understates risk by a third. Family 2 sits at 0.957–1.011 **for every one of the nine variants including the naive sample covariance**, which is H1 exactly.

**H3 is refuted, and the interesting part is why.** §5.3's eigenfactor adjustment moves family 4 by **−0.0001** at `a = 1.0` and **0.0000** at `a = 1.4`. It closes none of the H2 gap. It is not inert everywhere — family 3, the eigen-portfolios the stage is *about*, moves 1.0464 → 1.0388 at `a = 1.4`, in the right direction — so this is not a broken stage.

**Control C1 says the gap was never factor-covariance estimation error.** Registered with both its falsifiers before it was run (`experiments.md` rows 162–163). `Σ = X F X' + diag(δ²)` discards every residual correlation, and W4-P1 had already measured `SPY/IWM` at **−0.5488** — §5.5's own closing prediction, confirmed with the opposite sign. Replacing `diag(δ²)` with `D_δ R_u D_δ`, **off-diagonal only**, with §5.5's four legs and its shrunk diagonal untouched:

| Horizon | family 4 `B`, diagonal `Δ` | family 4 `B`, residual correlations restored | family 2 |
|---|---|---|---|
| short | 1.3321 | **1.0481** | 1.0101 → 1.0122 |
| long | 1.3135 | **1.0237** | 0.9987 → 1.0007 |

**107% of the gap to the naive comparand closes**, and family 2 moves by 0.002 — inside the exact half-width, which is the second registered falsifier doing its job: this is not a level effect, it is direction-selective. Realised minimum-variance volatility falls from 5.196 to 4.679 bps/day short and from 5.157 to **4.632** long, the latter beating the naive comparand outright.

**So the reading is:** at `K = 6` and `N = 13` this project's optimizer-portfolio bias is **specific-risk misspecification, not factor-covariance sampling error**, and §5.3 corrects the second. `K/T_eff` is 0.0248 short and 0.0083 long, so Shepard's `[1 − K/T_eff]⁻²` predicts a 5.1% / 1.7% understatement — there is barely any factor-covariance bias at this `K/T` for the stage to correct, which §6.3 warned of in advance (*"your numbers will be smaller"*). The stage is aimed at a bias this panel does not have, and the bias this panel does have sits in a term the stage cannot reach. **That is a statement about `K/T`, not about the technique**, and W7 is where it is tested: at `K ≈ 56` the same stage faces a `K/T_eff` an order of magnitude larger.

**What this does NOT license.** Making `Δ` non-diagonal is **not** on the table and C1 is not a candidate. §5.5 specifies a diagonal, USE4 ships a diagonal, and at §15.2's `N ≈ 500` a full residual covariance estimated from `T_eff = 727` is exactly the singular regime §15.2 uses to argue that a sample covariance is not a thing that works. A specific-risk model the week-7 module cannot run is not a model this project may adopt. The production path is unchanged; the row is counted `model-config` anyway.

**And it does not license reading the naive comparand as the better model.** The naive asset-level sample covariance scores `B = 1.066` on family 4 against the factor model's 1.332 and a lower realised minimum-variance volatility (4.570 against 5.196 bps/day, short). It wins here for one reason, which C1 identifies: at `N = 13` it carries the residual correlations for free because it never separated them out. That advantage is a property of `N = 13`, and §15.2's table is the reason it does not survive: at `N = 500` and `T_eff = 727` the same estimator has `N/T = 0.688` and `[1 − N/T]⁻² = 10.25`. **Reported, and not adopted** — the same footing §4.2.5 puts Ledoit-Wolf and OAS on.

**A fourth finding, recorded because it is the shape of the chart rather than a number.** Family 4's rolling `B` sits near 1 through 2010–2013 and climbs steadily thereafter, running 1.3–1.8 from 2016 on. The bias is not a constant level shift; it grows through the sample. Nothing here diagnoses that, it is stated as an observation, and the natural suspects — a residual correlation structure that strengthened after 2013, and a `K/T_eff` that fell by an order of magnitude across the same window — are not separated by anything W4-P2 ran.

### 6.2.6 FINDING (2026-09-02, W4-P2): the risk-model term is SPECIFICATION error, not estimation error — and that reframes the project

**This is the headline, and it changes what the `(1 − 1/B)` term of §1's identity is made of.**

`K/T`, Shepard's second-order correction and §5.3's eigenfactor adjustment all describe one thing: **estimation error**. The covariance matrix is right in its form and wrong in its numbers, because it was fitted on a finite sample. Every correction in §5.3 and §6.4 is aimed there.

What this panel has is **specification error**: the false diagonal in `Σ = X F Xᵀ + Δ`. The matrix is wrong in its *form*, and no amount of data fixes a form.

**Two independent measurements say so, taken different ways.**

| Route | Reading |
|---|---|
| **C1, empirically** (rows 162–163) | **107%** of the family-4 gap is attributable to the residual correlations, against family 2 moving **0.002**. Direction-selective, which the second registered falsifier established. |
| **Shepard, arithmetically** (§6.4) | At `K/T_eff = 0.0248` the closed form puts the understatement at **5.1%** against an **observed 32%**. Sampling error accounts for **at most a sixth of the gap even in principle** — and that is the generous reading, since taking `[1 − K/T_eff]⁻²` as the variance multiplier Eq. 32 actually writes gives **2.5%** on volatility and a share of one thirteenth. At the long horizon `K/T_eff = 0.0083` and the share is smaller again. |

**So H3's first half being refuted is not a disappointment — it is the same measurement seen from the other side.** A stage that corrects estimation error cannot move a bias that is not estimation error, and both routes say this one is not. §5.3 is working correctly on family 3, the eigen-portfolios it is about, and correctly moving family 4 by nothing.

**As the general finding: at small `K` the industry-standard corrections address a term that is negligible, while the term that actually matters is one the model assumes away.** That belongs beside the five published techniques this project has already measured out of regime — Marchenko-Pastur denoising (§4.2.7), §5.5(c)'s blend (§5.5.2), §5.5(d) at two-member buckets (§5.5.5), the Bartlett fallback firing at `K = 3` rather than `K = 56` (§5.2.3), and now §5.3 at `K/T_eff` of a few hundredths. **It is the sixth, and it carries the largest consequence, because it is about the technique the project was built around.**

**It also explains the naive sample covariance winning at `N = 13`.** It imposes no diagonal assumption, so it has none to be wrong about. Its advantage is the absence of a misspecification rather than better estimation — it estimates 91 free covariance parameters against 21 in `F` plus 13 in `Δ`, so it is the noisier of the two — and C1 is what separates those explanations: give the factor model the residual correlations and it matches the naive comparand short and beats it long. §15.2's table is why the advantage does not survive week 7: at `N = 500` and `T_eff = 727` the same estimator sits at `N/T = 0.688` and `[1 − N/T]⁻² = 10.25`, effectively singular, where estimation error is the whole story.

**Why the diagonal is not repaired, since a reader will ask.** Repairing it would mean **not implementing the specified model**, and the specification's failure is the deliverable. That is W4-P1's ruling holding rather than a new one: §5.5.1 settled five undefined terms on structural arguments with the alternatives **deliberately not built**; §5.5.5 kept stage (d) after proving it data-independent at a two-member bucket; and in every one of the five out-of-regime findings the alternative — drop the stage because it does nothing here — was declined, because week 7 is the panel the stages were written for. A project that repairs each published technique wherever this panel embarrasses it ends up reporting a model nobody specified and can no longer say which published method failed where. **C1 is a control and not a candidate**, §5.5 still specifies a diagonal, and `config/model.yaml` is untouched by it.

**What this hands week 7.** The two error types are separable there and are confounded here. The equity module puts `K ≈ 56` at `K/T_eff = 0.039–0.231`, an order of magnitude up, where Shepard predicts 8.2–69.1% — so estimation error stops being negligible and §5.3 has something to correct. Whether specification error scales with it is exactly what `experiments.md` row 164 registers.

## 6.3 Published targets to check yourself against

Menchero-Wang-Orr, N=50 US stocks, optimized portfolios, `α_P = 1`:

| T | Bias, sample F₀ | Realised vol, F₀ | Bias, F₀* | Realised vol, F₀* |
|---|---|---|---|---|
| 60 | 7.78 | 8.04% | 0.89 | 3.86% |
| 100 | 2.25 | 4.57% | 0.97 | 3.66% |
| 200 | 1.45 | 3.77% | 1.02 | 3.53% |
| 500 | 1.20 | 3.51% | 1.05 | 3.45% |

Out of sample: mean realised volatility ratio 0.936 (**6.4% risk reduction**), ≈7% IR improvement; min-var fully-invested 14.64% → 13.98% realised vol.

The `T = 60` row is worth internalising: with a short window the sample covariance is not "slightly optimistic," it is catastrophically wrong.

**Your numbers will be smaller** — you have 15 assets and 6 factors, not 50 stocks, so `K/T` is far lower and there is less bias to correct. Say that up front rather than letting a reader think you got a weak result. Then show the sensitivity: shorten the half-life until `K/T` matches MWO's setup and demonstrate the bias appearing on schedule. **That experiment — deliberately breaking your own model to reproduce a published effect size — is worth more than any headline number in the project.**

## 6.4 Shepard's second-order risk correction

Shepard (2009), arXiv:0908.2455. Independent derivation of the same bias from the estimation-error side.

```
Σ²_SO      = E(ŵᵀΩŵ)·[1 − N/T]⁻²           asset-level covariance      (Eq. 13)
Σ̂²_SO,f    = [1 − K/T]⁻²·ŵᵀX̂F̂X̂ᵀŵ           factor model               (Eq. 32)
T_eff      = 2τ/ln 2                        EWMA half-life → effective T (Eq. 33)
```

Worked for this project's parameters (`T_eff = 242` days at the 84-day volatility half-life):

| Setting | K or N | K/T | `[1−K/T]⁻²` | Vol understatement |
|---|---|---|---|---|
| Asset-level sample covariance, N=15 | 15 | 0.062 | 1.136 | **13.6%** |
| 6-factor model | 6 | 0.025 | 1.051 | **5.1%** |

**The factor structure buys a 2.6× reduction in second-order risk before any adjustment is applied.** That is a concrete, defensible, quantified argument for why a factor model rather than a sample covariance matrix — and it is the answer to "why not just use the sample covariance with 15 assets?", which you will be asked.

Report the corrected forecast alongside the raw one everywhere. Note that MWO find Shepard's closed form *under*-predicts the empirically observed bias (2.0 predicted vs 2.25 observed at T=100), so use the Monte Carlo eigenfactor adjustment as the correction and Shepard's formula as the analytic cross-check.

### 6.4.1 RULING (2026-09-03, W4-P3): the multiplier is on VARIANCE, both readings are carried, and no figure is emitted without its unit

Eq. 13 and Eq. 32 write `[1 − K/T]⁻²` against a **variance** (`Σ²`). MWO's cross-check quoted above fixes the reading beyond argument: at `T = 100, N = 50` Shepard "predicts 2.0" against an observed bias statistic of 2.25, and `(1 − 0.5)⁻¹ = 2.0` on volatility while `(1 − 0.5)⁻² = 4.0`. **The worked table above applies the variance multiplier to volatility.** Its "5.1%" and "13.6%" are `[1 − K/T]⁻² − 1`, which on volatility are **2.5%** and **6.6%**; the 2.6× ratio survives to first order in either unit.

The table is not rewritten. Every report before W4-P3 quoted its convention, W4-P2b's half-life sweep was published in it, and W4-P2b had just shown what a number quoted without its unit does to two sessions' readings of a λ amplitude. So the ruling is the operator's: **implement both readings, label every use site, and assert that no understatement figure is emitted anywhere without its unit attached.** `mafrm.risk.shepard.SecondOrderRisk` is the one implementation — four copies existed (the covariance build, the eigenfactor report, the half-life report and the bias report), two already disagreed about what they called the result, and all four now read from it — and it refuses `format()` without a `Unit`. The spec's convention lives under the name `table_convention`, so the superseded figures stay reproducible and recognisable as what they are.

**Which `T_eff` case this is, stated rather than inherited.** W3-P3b measured `2τ/ln 2` transferring at 1.007× for an EWMA **second moment** and at 1.31× after correlation normalisation (rows 109–111), with the excess spectrum-dependent. Shepard's multiplier is a variance-scaling correction and every `σ`-leg quantity below is a second moment, so the formula transfers here; `experiments.md` row 175 checks it on Shepard's own single-window case and finds 5.06% ± 0.21% against 5.14%. The `ρ` leg is the normalised case, and row 177 predicted — correctly — that its bias would land at the row-109 end of the band rather than at 1454.

### 6.4.2 FINDING (2026-09-03, W4-P3): the three corrections are disjoint to Monte Carlo precision, and the whole-covariance closed form is NOT this pipeline's bias

§5.4.2 said the disjointness demonstration is three-way. It was done by simulation on the pipeline's own estimators (`mafrm.risk.second_order`, `reports/second_order_risk.md`; registered at `experiments.md` rows 175–186 before the run), because arguing it would have been failure mode 7 with better prose.

**The problem, exactly.** Shepard's closed form is derived for a whole covariance estimated on **one** window. §5.1's estimator is `F = D ρ D` with `D` on the volatility window (84/252) and `ρ` on the correlation window (504), and after §5.3.2 the eigenfactor stage corrects `ρ`'s sampling error alone. So the arithmetic a reader will form — the eigenfactor stage's 0.84% plus Shepard's 5.14% — is wrong in both terms: the first is the `ρ` leg of the second, and the second describes an estimator that re-estimates `ρ` on the short window, which this pipeline does not.

**The measurement.** Gaussian rows from the panel's own `F`, run through `ewma_second_moment` and `correlation_from_covariance`, minimum-variance `w` of each `F̂`, statistic `E[w'Fw / w'F̂w]` = the population `B²` of family 4, `M = 2,000` from `model.seed`. Five conditions: (A) single window at `τ_σ` — Shepard's case; (B) the split-window estimator; (C) `ρ` estimated, `σ` exact; (D) `σ` estimated, `ρ` exact; (E) the per-factor `F_kk/F̂_kk` under (B). In variance, short / long:

| | short | long |
|---|---|---|
| (A) single-window joint | 5.06% ± 0.21% (Shepard 5.14%) | 1.75% ± 0.12% (1.67%) |
| **(B) split-window joint — the pipeline** | **2.04% ± 0.15%, `B` = 1.010** | **1.15% ± 0.10%, `B` = 1.006** |
| (C) `ρ` leg — §5.3's | 0.65% ± 0.08% (Shepard at 1454: 0.83%; at row 109's 1890: 0.64%) | 0.65% (same draws) |
| (D) `σ` leg — uncorrected | 1.46% ± 0.15% (`ρ = I` form 1.27%) | 0.56% ± 0.08% (0.42%) |
| (E) per-factor — what a `B_t^F` sees | 1.04% ± 0.09% (exact `E[1/X] − 1` = 0.83%) | 0.40% ± 0.05% (0.28%) |
| interaction (B) − (C) − (D) | **−3.4%** of the joint | **−5.0%** of the joint |

**The legs add.** The interaction is inside the registered 10% at both horizons, with a small negative third-order remainder, so §5.1's second-order bias is the `ρ` leg plus the `σ` leg to Monte Carlo precision. The mechanism is the one written before the run: for Gaussian data the sample correlation is independent of the sample variances and the bias is a quadratic form in the estimation error. Hence:

1. **Eigenfactor ⟂ Shepard's `σ` leg.** The eigenfactor stage's correction (0.65%, cross-checked by Shepard at the correlation window and by W3-P3b's +0.54%/+0.76% minimum-variance revision) and the uncorrected `σ` leg (1.46% / 0.56%) may be quoted side by side. **What may not be quoted as this pipeline's bias is the whole-covariance closed form at the volatility window**: 5.14% is 2.5× the measured 2.04%, because `ρ` sits on a window six times longer than the formula assumes.
2. **Eigenfactor ⟂ VRA.** Measured in W3-P4 at 0.017–0.033% of volatility, exactly quadratic in `a` (§5.4.2). Third-order; unchanged.
3. **Shepard ⟂ VRA.** Second-order risk is a property of optimiser-selected directions and §5.4's `B_t^F` is fitted on the `K` fixed factor axes. What a fixed axis sees of the estimation error is the inverse-variance Jensen term — Shepard at `p = 1`, exactly `E[1/Σ w_i z_i²] − 1` = **0.83% short / 0.28% long** by the Laplace-transform integral the report now computes. That is the entire estimation-error content of the VRA's fitted multiplier, and the rest of the `σ` leg (0.63% / 0.29% of variance) is optimiser-specific and invisible to a per-factor statistic.

**The `σ`-leg closed form, derived before the run.** For `ρ = I` and true minimum-variance weights `w`, the `σ`-only bias is `1 + (2/T_eff)(2 − Σ_k w_k²)` to second order — the Herfindahl of the min-var weights replaces `1/K` for unequal variances, and `K = 1` returns Shepard's own `1 + 2/T`. Checked on synthetic iid data (1.01353 predicted, 1.01365 ± 0.00028 measured at 40,000 trials) and recovered by the simulation's `ρ = I` control at both horizons (rows 176, 182). On the panel's correlated `F` the `σ` leg sits 15–33% above this floor: hedged positions are more sensitive to a variance error, never less.

**Two rows refuted, and what they teach.** Rows 180 and 186 registered (E) against `1 + 2/T_eff` within 2 s.e.; the measurement came in at +2.3 s.e. at both horizons — which is *one* excursion, because both horizons draw the same Gaussian rows from one seed and `τ_σ` does not enter condition (C). A second seed (0.84%) and a 10× run (0.81% ± 0.03%) sit on the exact value; the registered number was right to three decimals. Recorded as refuted, with two lessons: **a 2 s.e. band against a comparand that can be computed exactly fails 5% of the time by construction and should not be simulated at all** (the report now prints the exact integral), and **two horizons sharing one seed are one measurement**, so a registration that wants independent replication across horizons must vary the seed and say so.

**The revision this forces on §5.3.3's "uncorrected `σ` leg".** That section put it at "5.14% against 0.84%" — the difference of two single-window figures. On the measurement it is **1.46% of variance (0.73% of volatility) at the short horizon, 0.56% (0.28%) at the long** — 71% and 49% of the split estimator's actual second-order bias. The direction of §5.3.3's concern was right and its magnitude was a third of what the subtraction implied, entirely because the subtraction treated a one-window formula as if it described a two-window estimator.

### 6.4.3 RULING (2026-09-03, W4-P3): what "the corrected forecast reported alongside the raw one" is

`reports/bias_statistics.md` now prints, beside every family-4 forecast and `B`, the closed form that applies to that variant with its unit: **Eq. 13 at `N` for the naive `sample` variant**, whose estimator is exactly the one-window whole covariance the formula is derived for, and **Eq. 32 at `K` and the volatility window for every factor variant, labelled an UPPER BOUND** for the reason in 6.4.2 point 1. The corrected `B` is the raw `B` divided by the volatility multiplier. For `sample` it takes 1.066 → 1.009 (short); for the factor variants it takes 1.332 → 1.299, which is the finding of §6.2.6 from the other side — the factor variants' excess is not sampling error and no `K/T` correction reaches it. The measured split-window bias (`B` = 1.010 / 1.006) is the honest estimation-error figure for §5.1's estimator and is reported in `reports/second_order_risk.md` rather than applied as a correction, because a Monte Carlo constant from a Gaussian truth is a measurement of the estimator, not a stage of the model. **The Monte Carlo eigenfactor adjustment remains the correction and Shepard remains the cross-check**, as this section says; what changed is that the cross-check is now applied to the leg it describes.

### 6.4.4 H4 (2026-09-03, W4-P3): arithmetic holds in both units; empirically SPLIT

At `N = 13` the factor structure divides the second-order risk by 2.24× in variance (2.6× at this table's nominal `N = 15`). **On the estimation-error term the reduction is larger than the arithmetic says**: the split-window factor estimator's measured bias is 2.04% of variance against the naive comparand's 11.7% closed form — 5.7×, because `ρ` on the long window buys more than the parameter count alone. **As a statement about family-4 bias on this panel H4 is refuted**: the factor model's `B` is 1.33 against the naive covariance's 1.07 (`experiments.md` rows 144, 148, 189), because §6.2.6's diagonal specification error costs thirty times what the factor structure saves in sampling error. H4 was a statement about sampling error, and sampling error is not what this panel is about. The `sample` variant's own excess over Eq. 13 (`B` 1.066 against 1.057 predicted) is the size of its 99% tail ratio, 1.28 — fat tails, which no `K/T` argument reaches either.

### 6.4.5 FINDING (2026-09-06, W8-P1): the `tau` grid — the `rho` leg is `tau_sigma`-independent, the joint at `tau = 21` is 5.7%, and the closed form's over-prediction grows as the window shortens

W4-P3's forward registration (`experiments.md` after row 191), run as rows 318–326: `mafrm.risk.second_order.simulate` at every point of §5.1.3's grid, the truth held at the short horizon's shipped `F`, `tau_rho = 504` throughout, `M = 2,000`, **the seed varied across grid points** (`model.seed` + grid index) so that nine points are nine measurements. `reports/second_order_risk.md` (the `tau` grid section) and `reports/second_order_tau_grid.png`.

**Every registered leg holds.** The joint split-window bias at `tau_sigma = 21` is **5.67% ± 0.32% of variance** against the registered 6.5% [4.5%, 8.5%] — the `tau`-independent `rho` leg plus the `sigma` leg scaled by `T_eff`, which is what the registration said it would be. The `rho` leg sits at 0.61–0.75% (s.e. 0.08%) at all nine points, within 2 MC s.e. of row 177's 0.647% at every one: **it does not move with `tau_sigma`, because it cannot** — `tau_rho` is pinned. The `sigma` leg does all the moving, 5.08% at `tau = 21` down to 0.27% at 504, roughly `1/T_eff` of the volatility window. The interaction stays at −1% to −6.5% of the joint, inside row 179's 10%.

**The number the `K/T` chart needs.** Shepard's whole-covariance closed form at the volatility window over-predicts the split-window estimator's bias by **4.1× at `tau = 21`, 2.3× at the shipped 84, 1.7× at 252, and 0.94× at 504** — where `tau_sigma = tau_rho` and the estimator IS the one-window case the formula is derived for. The over-prediction grows as the window shortens because the formula charges `rho`'s estimation error at a window it is not estimated on. That is why §15.1's chart evaluates the curve at the correlation window for a factor-model point and carries the volatility-window figure as §6.4.3's upper bound: on this estimator the closed form is a comparand at the window the correlation is estimated on, and an upper bound at the other. The 2.5× of §6.4.2 was one point on this curve; the curve is now measured across 24× in `tau`.

**What this does to W4-P2b's "an eighth" at `tau = 21`.** Family 2's observed 3.0% of volatility (≈ 6.1% of variance) against the closed form's 23.2% is now three accounted-for pieces: the unit (a factor of two), the one-window-versus-two-window mismatch (4.1×), and the difference between an estimation-independent family's Jensen term and an optimiser's second-order risk. None of the eighth is unexplained.

## 6.5 The rest of the battery

| Test | Formula / detail | What it catches |
|---|---|---|
| **Mincer-Zarnowitz** | Regress realised variance on predicted: `R²_t = a + b·σ̂²_t + ε`. Calibration requires a=0, b=1. Use QLIKE loss rather than MSE — variance forecast errors are heteroskedastic and MSE over-weights crises. | *Conditional* miscalibration — systematically wrong at high vol but right at low vol, which the bias statistic cannot see. |
| **Q-statistic** | Menchero & Ji, *JPM* 50(3), 2024. Cross-sectional correlation between predicted and realised risk. | Whether the model **rank-orders** risk correctly. A model can have B = 1.00 and be useless if it cannot tell a risky asset from a safe one. |
| **Realised vol of min-var portfolios** | Menchero & Ji's key methodological argument: do **not** compare risk models by realised information ratio, it is far too noisy. Compare by the realised volatility of min-var portfolios built from each. | The cleanest single "which model is better" test. Converges much faster than IR. Make this your headline model-comparison metric. |
| **Ljung-Box Q on b_t²** | Standard portmanteau test on squared standardized returns. | Remaining volatility clustering the model failed to capture. |
| **Kupiec LR_uc** | Unconditional coverage of VaR breaches. | Wrong VaR level. |
| **Christoffersen LR_ind / LR_cc** | Independence and joint coverage of breaches. | **Clustered** breaches — the failure that actually kills you, and which Kupiec alone misses entirely. |
| **Basel traffic light** | Binomial CDF on 250-day breach counts: green ≤4, yellow 5–9, red ≥10. | The regulatory framing. Free to compute, and it is the vocabulary a risk interviewer speaks. |
| **Acerbi-Szekely Z₂, Z_MB** | Backtests for expected shortfall, which is not elicitable and therefore needs these rather than a simple breach count. | ES calibration. |
| **Factor return t-statistics** | Mean factor return t-stat, plus % of periods with abs(t) > 2, per factor. | A factor whose returns are indistinguishable from noise is adding estimation error to the covariance matrix for nothing. Use it to prune. |
| **Factor validation vs AQR / Ken French** | Regress each self-built factor on the corresponding AQR Century-of-Factor-Premia series. **AMENDED 2026-08-30 — see §6.5.1: the TSMOM leg is struck, and only two of Model A's six factors have a comparand at all.** Report R², beta, alpha t-stat, rolling correlation, and on every row the monthly `n`, the window and the units. | Turns "trust me, my factors are right" into a falsifiable table. *A self-built trend factor correlating 0.6–0.8 with AQR TSMOM is strong evidence; 0.2 tells you something is wrong* — **retained as an illustration of what strong evidence looks like, and NOT APPLICABLE to Model A, which has no trend factor.** **Make this a headline artefact, not an appendix.** |

### 6.5.1 AMENDMENT (2026-08-30, W2-P3): the TSMOM comparand does not exist, and four factors have no analogue

**Two defects, not wording.** Both are corrected here rather than worked around.

**1. TSMOM is a comparand for nothing in this project.** The row above and §12's week-2 *done when*
("Validation against AQR TSMOM and Century of Factor Premia — the falsifiable table") both name the AQR
Time-Series Momentum series. **No factor in the set §4.1 specifies is a trend factor.** The six are equity,
rates level, rates slope, credit, commodity and dollar; TSMOM is trend-following, and its five columns
(`TSMOM`, `TSMOM^CM`, `TSMOM^EQ`, `TSMOM^FI`, `TSMOM^FX`) have no counterpart in that set. Neither acceptance
criterion was meetable as written.

The fix is to strike the leg, not to build the factor. **Adding a trend factor to satisfy a validation
criterion would be alpha research and is out of scope under CLAUDE.md** — and it would be a factor invented to
pass a test rather than to explain risk, which is the worse of the two objections. The 0.6–0.8 sentence stays
in the table as an illustration of what strong external evidence looks like, marked not applicable.

**2. Four of the six factors have no published analogue in the files this project holds, and three have none
at all.** The mapping was written factor by factor from a printed column listing of the three AQR workbooks
and the Ken French daily research table, **before any regression was run**, precisely so that it could not be
inferred from whatever happened to fit:

| Factor | Comparand | Status |
|---|---|---|
| Equity | AQR Century `Equity indices Market`; our own SPY excess return at daily frequency | full comparison |
| Rates level | AQR Century `Fixed income Market` | **sign test only** |
| Rates slope | — | **no analogue** |
| Credit | — | **no analogue** |
| Commodity | AQR Commodities-for-the-Long-Run `Excess return of equal-weight commodities portfolio` | full comparison |
| Dollar | — | **no analogue** |

The three unmapped factors are unmapped for stated reasons, sourced to the column listing: Century's
`Fixed income Carry` is a cross-sectional long/short across *countries*, not a US 2s30s slope; no file holds a
credit series at all; and Century's `Currencies *` columns are dollar-neutral long/short style factors while
`TSMOM^FX` is trend on FX, so neither is the level of the dollar. **A factor with no comparand is a stated
limitation and is never a row filled with the nearest available column.** It is carried in the README
limitations as well as in `reports/factor_validation.md`.

**Rates level is run as a sign test and nothing more.** AQR's `Fixed income Market` is a global bond-market
excess return and the factor is a US yield change in basis points, so the coefficient is
`−D_global × (ΔY_global/ΔY_US)` — a product of two quantities neither of which is published anywhere in this
repository's sources. Any band wide enough to be defensible around that product is too wide to fail, and a
gate that cannot fail is worse than no gate because it reads as evidence while supplying none. **No
implied-duration band is registered.** The falsifier is `β < 0`, measured at **−0.12699**. The implied
`1/|β|` = 7.87 years is reported as a diagnostic with both unknowns named and is gated on nothing. This test
adds *independence from our own data layer, not power*: the strong test of the level factor already ran in
W2-P2 at daily frequency on US data at n = 4,146 against independently known durations (`experiments.md`
rows 73–77).

**The comparison is monthly.** All three AQR workbooks are monthly, so these regressions run at n = 200–213
months, not at the panel's 4,146 daily dates. Daily factor returns are **compounded** to monthly and
basis-point yield-change factors are **summed**; the aggregation is asserted as an exact round trip. AQR
stamps month-ends on the last *business* day and this project on the last *traded* day, so the join keys on
the calendar month, carries AQR's observed date, and asserts the joined count against one computed from the
two coverages. The right edge is `holdout_start` on every comparison, asserted rather than inferred — all
three AQR files end after it.

### 6.5.2 The gates (2026-08-30, W2-P3): the 0.3 correlation bar is WITHDRAWN

W2-P3 was given a **0.3 correlation gate**. It is withdrawn on the §4.1.2 precedent — for **lack of validity,
not for inconvenience** — and the argument is owed rather than asserted.

**Why it was invalid.** §6.5's only published figures, 0.6–0.8 strong and 0.2 wrong, are stated for a *trend*
factor against AQR TSMOM. Per §6.5.1 that pairing does not exist here, and **no published number covers any
pairing that actually does**. 0.3 was a threshold for a comparand this project does not hold, written before
anyone had looked at what the AQR files contain. That is the same class of defect as the all-pairs 0.15 bar:
a gate measuring something other than what it claimed to.

**What it is replaced by.** Three pre-registered falsifiers, each of which can fail, and none of which needs
an invented constant:

| # | Falsifier | Measured 2026-08-30 |
|---|---|---|
| (i) | **The daily anchor.** Equity against our own SPY excess return reproduces `experiments.md` row 41's ρ = 0.9786 at daily frequency, on row 41's own window, to within sampling error | **0.9779**, 95% Fisher-z interval [0.9769, 0.9788] — contains 0.9786. Mean gap −0.11%/yr against a registered −0.122%/yr. **Passes** |
| (ii) | **The placebo.** Each mapped factor's R² against its own comparand exceeds its R² against every other mapped factor's comparand | equity **0.7700** vs 0.0001 / 0.2533; rates_level **0.7291** vs 0.0411 / 0.0529; commodity **0.4891** vs 0.0646 / 0.1984. **Passes on every row** |
| (iii) | **The sign.** `rates_level`'s β against `Fixed income Market` is negative | **−0.12699**. **Passes** |

The placebo is the substantive replacement and it is built from measurement rather than from a threshold: if
`commodity` explained AQR's Fixed income Market as well as it explains AQR's commodity index, the mapping
would be doing no work and the table would be decoration. It is the same move as the IVV/VOO control
(`experiments.md` row 63) and the zero-spread simulation.

**All three mapped factors clear 0.3 in absolute correlation anyway** (0.877, 0.854, 0.699), so the bar is
withdrawn from a position of not needing it — which is the only position from which withdrawing a gate is
legitimate.

**One further finding, recorded because it changed the design.** Century of Factor Premia's
`Commodities Market` and Commodities-for-the-Long-Run's `Excess return of equal-weight commodities portfolio`
are **the same series**, agreeing to 2.3e-12 over all 1,187 overlapping months. An earlier draft of the
mapping proposed listing both as commodity comparands on the reasoning that a second AQR construction costs
nothing. It is not a second construction, and listing both would have let `commodity` tie with itself in the
placebo. Only the CLR column is configured.

### 6.5.3 RULINGS (2026-09-03, W4-P3): the five parameters this table leaves open, and the row that is not implemented

Recorded before any code, on the operator's decision, and transcribed into `config/model.yaml` under `validation.battery`. None of the five is a number anyone picked.

1. **VaR level — 95% and 99%, both reported, neither selected.** Both are conventional and 99% is Basel's own, so nothing is invented and no level is chosen. Kupiec's level is *not* derived from the Basel row's provenance. Reporting both is also a statement about the tests' own power: at `T ≈ 3,900` the 99% level expects ~39 breaches against the 95% level's ~195, and Christoffersen's independence test needs breaches dense enough to cluster detectably. Acerbi-Székely runs at the same two levels, so ES and VaR are directly comparable.
2. **Distribution — the normal, with the empirical quantile reported beside it and never fed in.** The normal is not a choice: it is the distribution a variance forecast implicitly assumes and the one Basel-style backtesting assumes. A Student-t is refused because it needs `ν`, invented or fitted, and fitting it would be a `model-config` trial for a distributional claim this project never made. The standardised returns' own tail quantile against `−z_α` is a diagnostic of how fat the tail is; using it in the test would calibrate the quantile on the sample being tested.
3. **Ljung-Box `h` — 21, 63 and 252, all reported.** A month, a quarter and a year: the timescales the model's own half-lives live at, and constants the project already carries (held in months, converted with `data.trading_days_per_month`). `ln T` against `T/4` is not a choice between conventions — `T/4` is a Box-Jenkins identification heuristic, not a test lag.
4. **Acerbi-Székely null — simulated with `eigenfactor.monte_carlo_trials`**, read through a pointer. Same class of quantity ("not published; any value in range"), stated once. 2,000 draws gives p-value resolution of 0.0005.
5. **Menchero & Ji's Q-statistic is NOT implemented.** The repository does not hold *JPM* 50(3) 2024 and the row above gives one sentence; the realised-risk window is one of several degrees of freedom that would make it a different statistic, and CLAUDE.md's docstring convention requires citing a method, which cannot be done for a paper nobody here has read. A wrong constant produces a wrong number; a wrong definition produces a number that claims a provenance it does not have. **What replaces it**: a plain cross-sectional Spearman rank correlation between the forecast volatility standing when a block opens and the realised standard deviation inside it, on non-overlapping monthly blocks, named as this project's own construction and never called Q. The diagnostic question is answered; only the attribution is refused.

Two further points fixed at the same time. **The series is family 4's, pre-VRA (§6.2.2), unclipped** — §6.1's `±4` belongs to its standard deviation and a VaR test on a clipped series would have its tail removed by hand. **`[1 − K/T]⁻²` is carried in both readings and labelled at every use site** (§6.4.1).

### 6.5.4 FINDING (2026-09-03, W4-P3): the battery on every variant, and what the two refuted legs say

`reports/validation_battery.md`, one table, nine variants × two horizons, every test of this section except Q. Registered at `experiments.md` rows 187–188 from W4-P2's published `B` under the normal, so the informative content is what appears beyond `B`. Seven of nine legs hold at both horizons.

- **The level error is everywhere, and it is conditional.** Mincer-Zarnowitz slopes of **2.55–2.58 (short) and 2.94–3.00 (long)** on the factor variants against 1.10 for the naive comparand: realised variance rises two-and-a-half to three times faster than the forecast. A missing residual-correlation term that scales with the level produces exactly this, and it is what §6.2.6's diagonal looks like through this instrument. Kupiec rejects at both levels on every factor variant (377 breaches against 194 at 95%, 177 against 39 at 99%); Ljung-Box rejects at 21, 63 and 252 lags; Christoffersen's independence test rejects at 95% on every pre-VRA factor variant; `Z_2` runs −1.5 at 95% and −4.8 at 99% with every simulated `p` < 0.001; QLIKE puts `sample` first at both horizons.
- **The naive comparand's `B` = 1.07 is a TAIL excess, not a level one — leg (a) refuted.** `sample` passes Kupiec at 95% (`p` = 0.27 / 0.57) and fails at 99%; its tail ratios are 1.03 at 95% and 1.28 at 99%. The registered arithmetic scaled a normal's quantiles by `B` and that was the wrong model of the series.
- **Basel puts the factor model in the red zone in 7 of 15 years (short), 6 of 15 (long) — a plurality, leg (h) refuted by one block.** `sample` is green in 8 and 9 of 15. Fifteen non-overlapping blocks cannot deliver more than this; the reading is "about half the years".
- **Tails are fatter than any `B` explains**: 99% tail ratios 1.28–1.77 on every variant and horizon, so dividing by `B` still leaves the normal's 1% quantile inside the empirical one.
- **The model ranks risk**: cross-sectional Spearman 0.936–0.940 (s.e. 0.003) on every variant — easy at `N = 11` with volatilities spanning an order of magnitude, and reported as the answer to "does it rank" rather than as a discriminating test.
- **Five of six factors have full-sample Newey-West `|t| < 2`** (`equity` +3.19 is the exception); the non-overlapping annual exceedance is 0–6.7% on `n = 15`. **Nothing is pruned** — choosing risk factors by their mean return is alpha research.

## 6.6 Overfitting controls

You will evaluate many configurations. Count them honestly and deflate.

**What counts as a trial — decided 2026-08-26, at 28 rows rather than 200.** The trial count `N` is not "everything I ever ran". The deflated Sharpe asks a specific question: *given that I searched this many times, how much of my best Sharpe is luck?* Only searches that could have flattered a **reported strategy Sharpe** belong in `N`. `experiments.md` therefore carries a category on every row, written when the row is written:

| Category | What it is | In `N` |
|---|---|---|
| `data-diagnostic` | Validating the data layer against an external reference, bounding a construction difference, or a control run testing whether a statistic has power. No strategy attached, no reported Sharpe depends on it, and it does not alter the production data path. | **No** |
| `model-config` | A choice inside the risk model that changes the covariance or specific-risk estimate — half-lives, Newey-West lags, eigenfactor `a`, shrinkage `q`. Reaches a Sharpe only through the optimizer. | **Yes** |
| `strategy-config` | Anything that changes the backtested portfolio — `γ_trade`, constraint sets, cost regime, rebalance rule, universe subsetting. | **Yes** |

The rule was fixed in week one, before any `model-config` or `strategy-config` row existed, precisely so that it could not be chosen later when it happened to help. Two guards: a row qualifies as `data-diagnostic` only if **all three** of its conditions hold, and **categories are never reassigned after the fact** — a row moved out of `N` at the end of the project is exactly the manipulation the deflated Sharpe exists to detect.

Both directions of miscounting are errors, not one. An undercounted `N` makes the deflated Sharpe optimistic, which is the failure the statistic was built to prevent. An `N` padded with data-layer diagnostics over-deflates it, discarding a real result to look conservative. At the end of week one the honest split was **28 evaluated, 0 in `N`** — every look so far was validating the data layer, and nothing had been backtested.

- **Deflated Sharpe Ratio** (Bailey & López de Prado) using the *honest* count of `model-config` and `strategy-config` rows in `experiments.md`, tracked from day one. Report raw and deflated side by side, and report the `data-diagnostic` count alongside them so the reader can see what was excluded and why.
- **False Strategy Theorem** bracket on the maximum Sharpe expected from N independent trials of zero-skill strategies.
- **PBO via CSCV** (combinatorially symmetric cross-validation). `skfolio` implements the combinatorial purged machinery and it is genuinely hard to write yourself.
- **Lo (2002)** standard errors on every Sharpe reported, with the autocorrelation correction.
- **Purged K-fold with one-sided embargo** for anything fitted.

---

# 7. The cost model

## 7.1 Functional form

Per asset, on the normalized trade `z_i = u_i/v` (dollar trade over NAV):

```
φ_trade,i(z_i) = a_i·|z_i|  +  b·σ_i·|z_i|^{3/2}/(V_i/v)^{1/2}  +  c_i·z_i
```

- `a_i` = half-spread + commission — **from EDGE, rolling 63-day, time-varying**
- `b` — the impact coefficient; see calibration below
- `σ_i` = trailing 1-year daily volatility
- `V_i` = trailing 1-year median dollar volume
- `c_i` = buy/sell asymmetry; set 0 unless modelling short-sale asymmetry

**Why the 3/2 exponent.** This is total cost = size × per-unit impact, and per-unit impact ∝ √(x/V). So `x·σ√(x/V) = σ·x^{3/2}/V^{1/2}`. The square-root law and the 3/2 cost model are the same statement, and `3/2 > 1` makes the term convex, which is what makes the whole optimization tractable. Say this explicitly in the README — a lot of people use the form without knowing why the exponent is what it is.

The `a|z|` L1 term is what generates the economically correct **no-trade region**: an asset is left alone unless its marginal alpha exceeds its round-trip spread cost. **Do not implement a separate ad-hoc no-trade band on top of it** — you would be double-counting.

## 7.2 Two calibrated regimes

The single most important calibration fact in the area: AQR's ~10 bps and Virtu's ~40 bps are **both correct** and measure different things. AQR is one patient, liquidity-*providing* manager with ~85% of orders passive; the peer universe includes liquidity-demanding, urgent and event-driven flow and includes commissions. **Model both. Do not calibrate a single number.**

Back out the square-root prefactor `Y` from published figures, at σ = 2%/day:

| Anchor | Source | Implied `Y` |
|---|---|---|
| 2% of ADV → 17 bps | AQR *Craftsmanship Alpha*, live execution data | **0.60** |
| 6% of ADV → 28 bps | AQR *Craftsmanship Alpha* | **0.57** |
| 2% of ADV → 40 bps | Virtu Global Cost Review, US trailing-year | **1.41** |

⇒ **`Y_patient ≈ 0.58`, `Y_urgent ≈ 1.4`.** Ratio of ~2.4×. That is a clean, defensible, derived calibration rather than a number lifted from a blog post, and deriving it yourself from two published sources is exactly the kind of thing that reads well in an interview.

Cross-check at Q/V = 1%, σ = 2%/day: `Y=0.58` gives 11.6 bps, consistent with AQR's value-weighted market impact of ~15 bps at a mean order size of 1.2% of ADV.

**Sweep `Y` and report the range rather than picking a point.** You have no execution data, so any impact coefficient is borrowed, not estimated. The sensitivity chart is more honest and more informative than a point estimate, and saying so is a strength.

**`γ_trade` is a first-class documented parameter, not 1.** Most desks run 1.5–3 as a deliberate margin of safety, because cost models are calibrated on the manager's historical order sizes and extrapolate poorly to larger ones, and because they omit the opportunity cost of unfilled orders. Man Group's work gives a second concrete reason: for a strategy with autocorrelated order flow, standard slippage TCA is biased low, because each metaorder's decision price already embeds the unreverted permanent impact of the previous same-signed one. Charge the permanent component against the whole future stream of same-signed trades — in practice, a larger `γ_trade` for names where your own flow is highly autocorrelated.

## 7.3 Impact scaling — three laws to make first-class

Under the square-root law with a fixed portfolio structure, `A` = AUM, `τ_A` = annual two-way turnover:

| Quantity | Scaling |
|---|---|
| Cost per unit traded | ∝ √A |
| **Total annual cost drag (bps of AUM)** | ∝ τ_A·√A |
| Total annual cost in dollars | ∝ τ_A·A^{3/2} |
| Cost vs volatility | **linear in σ** |
| Cost vs ADV | ∝ V^{−1/2} |

Consequences worth stating:
- **Doubling AUM raises the bps drag by ×1.41, not ×2.** This is why square-root vs linear impact changes capacity estimates by an order of magnitude.
- **Doubling turnover doubles the drag.** Turnover is a *linear* lever, size a *sublinear* one — turnover reduction is the higher-leverage control at any AUM.
- **Cost is linear in volatility.** A cost model without a vol term systematically under-charges trades in stressed markets — exactly when the optimizer wants to trade most. This is H5.

## 7.4 Cost-model unit tests

The highest-value test suite in the project. The March 2026 implementation-risk paper (Yin et al.) ran identical strategy logic through multiple mainstream backtest engines and found divergence of up to **3.71%** for high-turnover rotation strategies, purely from implementation differences — including a silent division error in a production cost model. Cite it; it justifies writing your own transparent engine and testing the cost model line by line.

Required tests:
- Hand-computed expected values for each term at three trade sizes.
- Costs applied to **actual turnover, on both sides, at the correct time**.
- Round-trip symmetry: buying then selling the same notional costs twice the one-way cost plus impact asymmetry, exactly.
- Zero trade → zero cost (catches the division-by-zero class).
- Reconcile your engine against `bt` on a trivial strategy and commit the comparison table. Half a day; a hallmark of engineering maturity.

### 7.1.1 RULINGS (2026-09-03, W5-P1): the tradable proxies, the windows, and the two constants that are absences

Recorded before any code, on the operator's decision, and transcribed into `config/model.yaml` under `costs`.

0. **Four of thirteen assets are not tradable as constructed, and a fifth is not either.** `govt_2y/5y/10y/30y` are synthetic constant-maturity zeros from the GSW curve; `tips_10y` is from the real curve. A spread, an ADV and an impact cost belong to an instrument, not to a curve point, so before any cost number exists each curve point is assigned a maturity-matched Treasury ETF whose bars supply its spread, volume and volatility: **SHY, IEI, IEF, TLT, TIP**, in `costs.tradable_proxies`. This is a **modelling choice, recorded as one**, with the caveat carried beside every cost figure: the proxy's cost is the cost of trading the ETF, not the cost of trading the exposure the risk model priced. **The 30y caveat is the important one.** TLT's duration is roughly half a 30y zero's, so its trading cost is the cost of trading about half the exposure the risk model priced, and that sentence accompanies every 30y cost number. The proxies are declared in `model.yaml` rather than `universe.yaml`, which is frozen and dated, because they are not universe members — the same reasoning as `data.spread_structure_control`. Without this ruling the cost term of §1's identity would have been computed on assets nobody can trade. The four new tickers were registered in `data/manifest.json` through the loader path; `dollar` (DTWEXBGS) has no tradable instrument and carries no cost.
1. **`σ_i` window: 252 days**, in `costs.volatility_window`, citing §7.1's "trailing 1-year daily volatility". One year is 252 trading days by the project's own `data.trading_days_per_year`, and a test pins the identity. This is `costs/` taking its own spec section's value — **not** the W3-P5 concern, which forbade carrying `factors/`' 252 into a risk-side estimator: the cost-side `σ_i` is trailing and realised, not the risk model's forecast, and coupling costs to the risk pipeline would be a specification the spec does not make.
2. **`V_i` window: 252 days** per §7.1's "trailing 1-year median dollar volume". `costs.adv.window` was 63, "deliberately matched to the EDGE window", and that comment is **withdrawn**: EDGE's 63 days governs spread-estimation noise, ADV's window governs volume stability — two different quantities with no reason to share a window. The two keys stay separate.
3. **Commission is zero, recorded as the absence of a published figure rather than as a value.** It enters `a_i` additively so a published number slots in later. **Direction, stated:** zero understates cost, which biases the identity's split toward the risk-model term — not the conservative direction for a project whose thesis is that both terms matter.
4. **`c_i = 0`** per §7.1, in `costs.buy_sell_asymmetry`. The term is implemented and nets to zero over a round trip; nothing here models short-sale asymmetry.

### 7.1.2 RULING (2026-09-03, W5-P1): how `a_i` is built — level from the issuer, shape from EDGE, added

`a_i` is not the EDGE series. §3.4.1 established that EDGE supplies time variation and cannot supply level for this sleeve, that the ban is design rather than quarantine, and that no constant and no per-ticker constant fixes a signal-to-noise of 0.04. That question is **not reopened** and this section does not reason toward a scale factor.

**The level** is the issuer-disclosed 30-day median bid/ask spread, committed as a dated static file — `config/spread_levels.yaml`, one row per cost-bearing asset with ticker, source URL, disclosure date and the value in basis points — exactly as §3.4.1 prescribes. Values are read from the issuer pages by hand and supplied by the operator; they are not taken from memory, not from `costs.flat_spread_assumption_bps` (which §3.4.1 forbids as a level source), and not from EDGE. **A row without a value is a placeholder, and the code refuses to build a cost input for that asset, by name. There is no default.** An asset whose issuer discloses no median gets a *sourced* band instead, swept and reported as a range — the treatment the impact coefficient already gets, for the same reason. The operator will supply all thirteen levels, so the band mechanism exists and no bounds have to be invented.

**The combination rule:**

```
s_i(t) = s_issuer,i + max(0, s_EDGE,i(t) − baseline_i)
a_i(t) = s_i(t)/2 + commission
```

Level from the issuer, shape from EDGE as an **additive widening in basis points**, no negatives, agreement at baseline. `baseline_i` is the **in-sample median of the clipped production EDGE series**: a median is robust to the crisis tail, so it is the calm level with no date selection and no judgement about which periods count as calm, and it reads nothing at or after `holdout_start`.

**A multiplicative rule was ruled first and withdrawn the same day, and the record of why matters more than the rule.** The first ruling was `s(t) = s_issuer × s_EDGE(t) / median(s_EDGE over the 30 days ending at the disclosure date)`. Two things were wrong with it. The anchor window sat inside the holdout — the disclosures are 2026 figures — which invariant 5 forbids. And the denominator was a number the project had already measured as unusable: W1-P5c established EDGE's per-window noise at ~23bp regardless of truth, so for the liquid names the clipped value is often exactly zero, and `costs.spread_vs_volatility.minimum_baseline_for_ratio_bps` already records that a ratio against a baseline below 5bp reports 170× or −26× and is not a statement about spreads. §3.4.1 consequence 1 had ruled the same four weeks earlier: a level offset cancels in a widening, which is why the crisis statistic is in basis points rather than a multiple. The additive form keeps every principle of the ruling — level from the issuer, shape from EDGE, agreement at the anchor, nothing negative — and drops the one thing that was wrong. With it the anchor question dissolves: there is no disclosure-date window and no holdout crossing.

**The distinction, recorded because it will come up again.** The ruling said "no result reverses it", and it was not a result that reversed it. A prior measurement is not a result — it is a fact the ruling should have been checked against before it was made. The implementing session was right not to adopt the additive form unilaterally and right to bring the measurement instead; "no result reverses it" binds against *new* evidence, not against evidence already on the record. The W5-P1 section of `experiments.md` carries the same note.

**Anachronism, stated.** The issuer figure is a 2026 disclosure applied to 2009–2024. It is the same anachronism §3.4.1 accepted when it made issuer medians the permanent level source, because no historical disclosed series exists. It is a caveat on every cost figure and is reported beside them. **Do not cross the holdout to reduce it.**

**The direction of the error that matters.** An over-stated spread widens the L1 no-trade region, suppresses turnover, lowers measured cost drag and flatters the cost term of §1's identity. That failure points in the flattering direction and would not look like an error — the fourth instance of that pattern in this project, after rows 54–58's absolute value, the ADV mean and the specific-risk VRA absorption. It is why the level is refused rather than defaulted.

### 7.1.3 RULING (2026-09-03, W5-P1): a displayed figure is an interval, and the cost model runs at both ends

All thirteen issuer levels were supplied by hand from the issuer pages the same day (`config/spread_levels.yaml`, each with URL and disclosure date). Every issuer displays the figure at **0.01% resolution**, so a displayed value is not a level but a rounding interval of the full spread: "0.00%" is [0, 0.5)bp, "0.01%" is [0.5, 1.5)bp, "0.02%" is [1.5, 2.5)bp, "0.03%" is [2.5, 3.5)bp. `costs.spread_level.display_resolution_bps = 1.0` makes the loader read every point value as that interval and return the same `SpreadBand` object a sourced band for a missing level would be, so the cost model runs at both ends through one code path and reports a band. **No midpoint is invented.** The half-spread `a_i` is therefore known to ±0.25bp.

**What that resolution means for the identity** (`reports/cost_calibration.md` §5, `experiments.md` rows 195–196). At the anchor volatility of 2%/day the per-unit impact term is 16.4bp at 2% of ADV and 3.67bp at 0.1%, so a 0.5bp half-spread is 3.0% of it on the large trade and 13.6% on the small one: the resolution matters for small trades and is negligible for large ones, and the band the report shows is the honest width of that uncertainty rather than a sensitivity anyone chose. **DBC — the widest displayed spread (0.03%) on the thinnest volume ($31M/day median) — is where both terms of the cost function are largest for any given dollar trade, which is coherent.** One reading beyond the ruling, labelled as an observation: the bond sleeve trades at 0.08–0.44%/day, a tenth to a third of the anchor volatility, so its impact term is smaller by the same factor and the 0.5bp is a first-order share of it even at 2% of ADV (SHY 122%, IEI 43%, TIP 27%, LQD 24%, IEF 23%, HYG 21%). For those assets the cost term is spread-dominated and the display resolution is the binding uncertainty on it.

**SPY's second source.** SSGA's capital-markets document quotes ~0.6bp *average* bid-ask spread for calendar 2023. It is recorded in the file beside SPY's "0.00%" as a second source and does not enter the band: it is a mean over a different period against a 30-day median, so it bounds nothing from below, and it sits 0.1bp above the display interval's upper end — consistent with the interval only in the way a mean exceeds a median on a right-skewed series. The ruling asked for it to narrow the band from below; it cannot, and the reason is stated rather than the band adjusted.

### 7.2.1 RULING (2026-09-03, W5-P1): `Y` is two regimes and `γ_trade` is five points, both stated as rules

- **`Y` is the two published regimes — run both, report both, no interior points.** They are calibrated regimes, not endpoints of a continuum, and a sweep between them would manufacture a parameter from two numbers that measure different things. `reports/cost_calibration.md` derives both from their anchors: 17bp at 2% of ADV and 28bp at 6% give 0.601 and 0.572 (mean 0.586, config **0.58**); 40bp at 2% gives 1.414 (config **1.40**); ratio 2.41×. The cross-check at 1% of ADV returns 11.6bp exactly. The anchors themselves are in `costs.calibration_anchors` so the report and the test read the same figures the config claims to be derived from.
- **The `γ_trade` sweep is half-steps 1.0 to 3.0, five points**, `costs.gamma_trade.sweep_step`. Stated as the rule so the grid is not chosen at the moment it is run.
- **Both are `data-diagnostic` when they characterise the cost model**, because nothing is selected. **If W6 selects a `γ_trade` on backtest performance, that is `strategy-config` and counts then** — registered in `experiments.md` at the W5-P1 rows, before any backtest exists.

### 7.4.1 ACCEPTANCE (2026-09-03, W5-P1): what was built and tested, and what was deferred

`costs/impact.py` implements §7.1 term by term at a general convex exponent that reduces to the 3/2 form at `costs.total_cost_exponent`, with the per-unit square-root law and the total pinned against each other; `costs/spread_level.py` implements §7.1.2; `costs/inputs.py` assembles `σ_i`, `V_i` and `a_i` monthly from the cache through the proxies and refuses on any placeholder; `costs/calibration_report.py` writes `reports/cost_calibration.md`. Every test of §7.4's list is present with a hand-computed expected value: each term at three trade sizes in both regimes (inputs chosen so the powers are exact); zero trade → exactly zero with `V = 0`, `NaN` depth, `NaN` spread and `NaN` volatility, and a non-zero trade against any of those raising rather than returning `inf` or `NaN`; the round trip equal to twice the one-way cost bit-for-bit with the asymmetry term netting to zero exactly; costs on actual turnover on both sides at each date's own inputs with no fill from another date; the 11.6bp calibration check; and each §7.3 scaling law pinned against the cost function rather than asserted. **No separate no-trade band exists and a test asserts that none does.**

**Deferred, and why.** The `bt` reconciliation is W5-P3's engine cross-check and §12 lists it as a cut. The capacity curve (§10.4) needs the optimizer and is W5/W6 work. The cost model produces per-asset inputs for the whole universe (thirteen assets, 381 months, in-sample only) under §7.1.3's interval bands; **no portfolio has been costed yet**, because no trade series exists until W6's optimizer produces one. The refusal path for a placeholder level is retained and tested, and was exercised for the whole of this session's first half.

### 7.4.2 RULINGS AND ACCEPTANCE (2026-09-03, W5-P3): the engine, its `bt` reconciliation, and §11's no-look-ahead test

Four rulings, recorded before any code ran and none of them a chosen number.

1. **The reconciliation tolerance is `c · ε · T`, relative on NAV**, `T` the number of daily accounting steps, `ε` double-precision epsilon, and `c = 2 · (assets + accounting_roundings)` — the construction of §5.2.4's PSD bound: `assets` for the summation that values the book, `accounting_roundings` for the eight size-independent roundings per asset per step counted in `config/model.yaml` `backtest.reconciliation`, and the factor two because both engines round. Two implementations of the same accounting on the same prices and dates must agree to that. **Every discrepancy above it is attributed to a named cause — day-count, commission on shares versus on notional, fee financing, rounding order — before the table is committed. An unattributed discrepancy is a bug in one of the two engines, never a tolerance to widen.**
2. **Equal weight across the thirteen cost-bearing assets, rebalanced monthly on `mafrm.data.calendar.month_end_dates`**, the project's one definition of month end; equal weight is the absence of a choice (the W4-P1 reasoning). **`bt` is fed the same dates** and never resamples its own month ends, or the reconciliation tests calendar conventions instead of accounting — failure mode 1 through the back door. Synthetic first (a hand-computable answer), then the real cache (the plumbing): the W3-P1 shape of golden fixture plus real panel. The task's fixed 60/40 is the synthetic panel's strategy.
3. **`bt` reconciles accounting and the spread term as a proportional commission, and cannot reconcile impact.** The cross-check is complete in two halves — impact was pinned to §7.1's formula by W5-P1's hand-computed tests — and `reports/engine_reconciliation.md` states the split in its first section so that "reconciled against `bt`" is never read as covering the whole cost function. `bt` 1.2.0 and `ffn` 1.1.5 were verified MIT from their package metadata before being added, as a dev extra only, pinned exactly.
4. **The no-look-ahead test is generic.** It takes a callable producing "the state at `t`" and asserts bitwise invariance to perturbations strictly after `t`. With fixed weights the weights version is vacuous, so this session targets the covariance forecast, the cost inputs and the engine's NAV at `t`; W6-P1 plugs the optimizer's weights into the same harness without rewriting it — invariant 10 applied to a test. §11's "≥ 50 values of `t`" is mirrored into config as a spec constant and the parser refuses less.

**What was built.** `mafrm.backtest.engine` (about 330 lines, no framework): positions in units and cash in dollars, NAV recomputed from units every day so only a rebalance changes state; a rebalance at the close of each target date, decided on `NAV_pre`, the target dollar positions hit exactly, the §7.1 cost through `mafrm.costs.impact.trade_cost` at that date's own inputs and debited from cash on the same date; `NaN` prices refused rather than filled; no separate no-trade band, asserted by a test. Its accounting convention — fee from cash, negative cash carried at zero interest — is stated in the module docstring, and it differs from `bt`'s in exactly one named way. `mafrm.backtest.lookahead`: the harness, with a **power check** that perturbs the data through `t` and refuses to report a state that does not move as a pass. `mafrm.backtest.reconciliation` and its report: the engine and `bt` on the two panels, fee financing run both ways — `bt` given the engine's convention through a two-line algo that debits the fee after `Rebalance`, and `bt`'s native convention **replayed on the engine's arithmetic** (`shadow_trade_financed`, iterated to `bt`'s own stopping rule) so the native row's gap is shown to be the convention rather than argued to be.

**Result (experiments.md rows 199–206, all registered before the run, all HOLD).** Reconciled rows agree to two or three ulps: 4.4e-16 to 6.7e-16 relative on NAV against bounds of 3.4e-12 (two assets, 756 steps) and 4.2e-11 (thirteen assets, 4,450 steps, 2007-04-30 to 2024-12-31), i.e. 1e-4 to 1e-5 of the bound. The one row registered to breach breaches: `bt`'s native commission differs from the engine by 7.0e-6 of NAV at its widest on the real panel, **+0.0040 bp/yr annualised** (8.4e-6 and +0.0017 bp/yr on the synthetic), and the shadow replay reproduces `bt` to three ulps on both panels, so the whole of it is the named cause — the fee's own return, sitting in the risky book under one convention and in cash under the other. The task's acceptance, "within a few basis points annualised", is met by three orders of magnitude on the convention row and exactly on the reconciled rows. Row 204: the engine's NAV, the covariance forecast through §5.2's repair, and `σ_t`, `V_t` and the EDGE window at `t` are bitwise causal at 50 dates each with the power check on; the row-`t+1` control raises and the constant-state control is reported vacuous.

**Stated before the run, and recorded here because it will come up again.** Two components of §7.1.2's `a_i(t)` are **not causal by ruling** and were excluded from the causality test with the reason written before it ran: the issuer level is a 2026 disclosure applied to 2009–2024 (§3.4.1's accepted anachronism), and the calm baseline is the in-sample median of the clipped EDGE series — a full-sample constant applied at every `t`, chosen in §7.1.2 precisely because it selects no dates. A perturbation of the bars after `t` moves that median and would fail the test; that would be the ruling working as written, not a look-ahead through the estimation. The estimated components must be causal and are.

**Deferred, with the owner named.** §10.1's Perold decomposition needs the optimizer's decision timestamp and is W6; the engine supplies the accounting half. The weights version of the no-look-ahead test, the golden **weights** fixture of §11 and `results/metrics.json` need a strategy and are W6. Negative cash is carried at zero interest; W6 owns any financing leg. `CITATION.cff` and `.pre-commit-config.yaml` (ruff, ruff-format, nbstripout, large-file and private-key guards; every tag verified upstream) were added in the wrap-up as §11's small things.

---

# 8. The optimizer

## 8.1 Formulation

Single-period, cvxpy directly. **Do not import cvxportfolio** — it is GPL-3.0, which makes your repo GPL, and many quant employers' open-source policies are hostile to GPL. Use its cost specification as a documented benchmark you reimplement, cite it prominently, show a table comparing your parameterisation to theirs, and keep your repo MIT.

```
maximize   α̂ᵀz − ρᵀ|w+z| − γ_trade·φ_trade(z) − γ_hold·φ_hold(w+z)
                          − γ_risk·ψ(w+z) − ψ_mis·(α_⊥ᵀ(w+z))²
s.t.       1ᵀz = 0
           leverage, per-asset and per-sleeve limits, ADV participation
           soft tracking-error and turnover hinges

ψ(x) = (x − w^b)ᵀ (X F Xᵀ + Δ) (x − w^b)
```

**Soft constraints, not hard ones.** Convert limits into hinge penalties: `γ_risk·(σ − σ_tar)₊`, similarly for leverage and turnover. Two reasons: the problem is then **always feasible** (`z = 0` is always admissible), which eliminates the single most common production failure mode; and the optimizer can breach a limit slightly when the alternative is much worse. Set the priority parameters from the Lagrange multipliers of the hard-constrained problem observed in backtest, **at around the 80th percentile** — that is Boyd et al.'s concrete recipe.

Even so, plan for infeasibility: a relaxation ladder, fallback to prior weights, and logging of which constraint bound. Handling this gracefully is a genuine week if you don't plan for it.

## 8.2 Robustification is regularization

```
R^wc     = μᵀw − ρᵀ|w|                              worst case over μ_i ± ρ_i
(σ^wc)²  = wᵀΣw + ϱ·(Σ_i √Σ_ii·|w_i|)²              ϱ ∈ [0,1)
```

Both convex. The structural reading, which is the intellectual move worth stating in the README: **robustness to return-forecast error is mathematically an L1 penalty on leverage; robustness to covariance error is a squared weighted-L1 penalty.** You get shrinkage for free by choosing `ρ` and `ϱ` rather than by ad-hoc covariance shrinkage.

Related and worth a paragraph: **Jagannathan & Ma (2003)** show a no-short-sale constraint on a min-variance problem is *exactly equivalent* to solving the unconstrained problem with `Σ̃ = Σ − λ1ᵀ − 1λᵀ`, where `λ` are the constraint multipliers. Because binding constraints fall precisely on the assets with the most under-estimated covariances, the constraint acts as shrinkage of the largest covariance estimates toward zero — it repairs exactly the estimation error the optimizer would otherwise exploit. Imposing an economically *wrong* constraint reduced out-of-sample risk in their data. **Position and turnover limits are not risk-management furniture, they are an implicit covariance regularizer.** With a weak covariance matrix they are cheap insurance.

## 8.3 Alpha–risk misalignment

Decompose the alpha vector:
```
α_R = X(XᵀX)⁻¹Xᵀα              spanned by risk factors
α_⊥ = [I − X(XᵀX)⁻¹Xᵀ]α        orthogonal remainder
```
A tilt along `α_R` incurs factor risk; a tilt along `α_⊥` incurs **only specific risk according to the model**. The unconstrained optimum shrinks `α_R` by a factor always < 1, so the optimizer systematically **over-allocates to the orthogonal component precisely because the risk model cannot see its risk.** Result: understated ex-ante TE, overstated ex-ante IR, realised TE much larger than forecast. It gets *worse* when you add factor-neutrality constraints, because those explicitly strip out `α_R` and leave only `α_⊥`.

MSCI's fix (preferred over Axioma's synthetic-factor approach):
```
U = αᵀw − (λ/2)wᵀΣw − ψ_mis·(wᵀα_⊥)²        optimal ψ_mis ≈ λ·σ²(α_⊥)
```
Their numbers: IR 0.357 → 0.613 when `α_⊥` is pure noise; peak 0.759 vs 0.596 when it carries genuine return.

**Build-spec action:** compute and report the misalignment angle `cos θ = ‖α_R‖/‖α‖` every rebalance. This matters more for your model than for a commercial one, because your factor set is deliberately small — misalignment is guaranteed. Budget for the penalty.

## 8.4 Setting `γ_risk`

```
h* = (1/2λ)Σ⁻¹α ,   TE* = (1/2λ)√(αᵀΣ⁻¹α)   ⇒   λ = IR/(2·TE_target)
```
using Grinold-Kahn's `IR = √(αᵀΣ⁻¹α)`. Example: IR = 0.5, TE_target = 4% ⇒ λ = 6.25. **This is the right initialization, not the answer.** Then either replace it with a risk *constraint* and let the solver recover `λ` as the multiplier (what most PMs actually want, because TE is interpretable and λ is not), or tune by cyclic coordinate search: increase by 25%, evaluate, if worse decrease by 20%, repeat. `γ_risk` and `γ_trade` are **not separable** — higher `γ_trade` reduces responsiveness, which mechanically lowers realised TE and interacts with `γ_risk`.

## 8.5 Rulings and constructions (2026-09-04, W6-P1)

Every gap the W6-P1 orientation found in §8 was ruled by the operator before any optimizer code existed, and each ruling is written into `config/model.yaml` under `optimizer` with its reason beside it. This section is the short record; the config is the long one.

### 8.5.1 RULINGS (2026-09-04, W6-P1): `α̂`, `α_g`, `TE_target`, the absences, the constraints, and the one verification run

1. **`α̂` is a fixed, published construction: RSTR (§15.4), `T = 504`, half-life 126, lag 21, applied cross-sectionally across the thirteen assets.** Plus `α̂ = 0` as the no-view control, which reproduces W4-P2's family 4. Two runs, not a sweep. A random `α̂` was considered and rejected for a reason worth recording: with no realised expected return, `μ_g ≈ 0`, both Sharpes in §1's identity collapse toward zero and the decomposition reads `0 = 0 + 0`. The identity needs a `μ_g` that is *measured*, and a standard documented signal supplies one without the project claiming it works — which is exactly what CLAUDE.md's "fixed, documented input so the optimizer has something to trade against" means. **RSTR is never tuned and never compared against alternatives on performance.**
2. **The `α_g` sweep question dissolves.** With `α̂` fixed, `α_g` is *measured* as the RSTR strategy's realised in-sample gross return — the same shape as §5.3.3 removing `T` rather than measuring it — and the capacity curve uses the measured value with the caveat that it is what this signal did, not what it will do. `IR = √(α̂ᵀΣ⁻¹α̂)` is likewise computed from §8.4's own formula at each rebalance, not a parameter. `costs.capacity.gross_alpha` stays `null` and its parser still refuses a point.
3. **`TE_target` = 1× the equal-weight portfolio's realised in-sample volatility** — a measured anchor, not a constant, and equal weight is the absence of a choice. W6-P2 sweeps 0.5×, 1×, 2× as a band; W6-P1 runs the middle one. The anchor is a full-sample constant of the same class as §7.1.2's EDGE baseline and is excluded from the causality test with that reason, as row 204 excluded the baseline.
4. **`ρ = ϱ = 0` and `γ_hold = 0`, recorded as absences.** Robustification hedges error in `α̂`, and `α̂` is a fixed input with no claimed accuracy — robustifying it would be hedging a view nobody holds. A long-only ETF book has no borrow cost. **`ψ_mis` is computed from `α̂` and `Σ`** at each rebalance, never set. **`w^b = 0`** — absolute risk, matching `σ_f` in the identity.
5. **Constraints: long-only, fully invested — `Σw = 1`, `w ≥ 0`.** That is the absence of leverage; it matches family 4's structure and is what makes `γ_hold = 0` coherent. No per-asset or per-sleeve bound in W6-P1. The hinges exist, their bounds are set non-binding, and that is documented. **ADV participation cap at 6% of ADV** — the cost model's upper calibration anchor — on the principle *do not trade where the cost model is not calibrated*: the square-root law is anchored at 2% and 6% and extrapolates beyond. The 80th-percentile Lagrange recipe has nothing to bootstrap from until something binds; **it activates in W6-P2**, so W6-P1 imposes the cap hard, observes its multipliers, and reports their distribution.
6. **W6-P1 builds the optimizer and runs ONE verification configuration**: RSTR, patient `Y`, `a = 1.0`, `γ_trade = 1.0`, short horizon, `TE_target` at 1×. Stated as verification, not result. The grid is W6-P2's job.
7. **Anticipated, and to be treated as a finding rather than a bug:** with thirteen assets, long-only, and a cross-sectional `α̂`, the unconstrained optimum may be a corner solution — all-in on one or two names. That is §8.3's mechanism arriving on the first run: the optimizer loading maximally on `α_⊥`, the part of `α̂` the risk model cannot see. The weight distribution and `cos θ` are reported per rebalance. A concentrated solution reads "the unconstrained optimum for this `α̂` is concentrated", which is informative about misalignment and is precisely why W6-P2 needs bounds as a grid dimension. **No bound is added in W6-P1 to make the weights look reasonable.**

**Three constructions the rulings forced but did not name, each recorded with its alternative** (`config/model.yaml` marks them CONSTRUCTION):

- **`α̂` is in return units, not a z-score.** §8.1 adds `α̂ᵀz` to a cost that is a fraction of NAV, and §8.4's `λ = IR/(2·TE_target)` presumes `α` in return units — with a unit-variance z-score, "IR" is not an IR and the alpha-versus-cost trade-off depends on an arbitrary scale. So the exponential weights are normalised (RSTR becomes the weighted *mean* daily log excess return, a rate), the cross-section is centred with an equal-weighted mean per §15.5 — equal because the curve points carry no market cap, and because under `1ᵀw = 1` the level of `α̂` is untradeable and would only inflate `IR` — and it is **not** scaled to unit standard deviation. The rate is multiplied by `data.trading_days_per_month` so `α̂`, the risk term and the per-rebalance cost share one horizon. The alternative — a z-score mapped to return units through Grinold's `α = IC·σ·z` — needs an IC, a claimed accuracy ruling 4 says nobody holds.
- **`σ²(α_⊥)` in `ψ_mis ≈ λ·σ²(α_⊥)` is read as the model's own forecast variance of a unit position along `α_⊥`**, so the penalty prices that direction at twice what the model sees — the direction §8.3 says the model understates by construction. The repository does not hold MSCI's paper; the form is theirs, the estimate is the model's.
- **The book size.** The impact term and the ADV cap both need a NAV, and neither the spec nor the rulings supply one. Inventing one would be invariant 9, so the verification run is a **band at the two endpoints of §10.4.1's ruled AUM grid** — the AUM at which the largest position's trade is 0.1% and 10% of its proxy's ADV — with "the largest position's trade" read off the equal-weight book on the thinnest proxy's in-sample median ADV. Two configurations, two `strategy-config` rows. A ruled NAV replaces the rule by one config key.

**One departure from the letter of ruling 6, recorded rather than silent — and then RULED ON at the wrap-up (2026-09-04): the departure was correct and ruling 6's count claim is WITHDRAWN.** The operator's reasoning, in substance: *"not selected" was conflated with "not a trial". The deflated Sharpe corrects for the number of Sharpes that could have been reported, not the number the operator claims to have chosen among — and once a strategy Sharpe exists it is a trial whether or not anyone selected on it.* `N = 10` stands. **The sharpened criterion W6-P2 inherits:** the W3-P1 test — name the result that would reverse the decision — governs `model-config` decisions that produce no Sharpe; **once a strategy Sharpe exists, it counts.** Every cell of §9's grid is a `strategy-config` trial; reporting all of them does not reduce `N`, it makes the DSR honest. **The grid is to be designed knowing `N` may reach 50 or more, and that is to be said before it is built** (`experiments.md`, the W6-P2 forward statement). The original record follows. Ruling 6 said `N` does not move because nothing is selected. `experiments.md` rows 68–69 already rejected that argument for the production path — *"arguing that a decision taken on principle is not a search and therefore not a trial is exactly the reclassification the category rules forbid"*, and *"the conservative count is the one that costs nothing"* — and categories are never reassigned afterwards. The RSTR verification runs are the first backtested portfolios on the production path and are logged `strategy-config`; the cost-free `α̂ = 0` control satisfies all three `data-diagnostic` conditions and is logged so. The operator upheld this at the wrap-up, above.

### 8.5.2 FINDING (2026-09-04, W6-P1): it solves everywhere, the book is concentrated as anticipated, and §8.4's initialisation lands a third of the way

`experiments.md` rows 207–210; `reports/optimizer_verification.md`, `reports/binding_constraints.md`, `reports/optimizer_weights.png`. Registered before the run and all scoreable legs hold.

**It solves.** 187 month-end rebalances, 2009-05-29 to 2024-11-29, five configurations (RSTR at two book sizes × two spread ends, plus the `α̂ = 0` control): every one at the first rung with Clarabel, no constraint relaxed, no fall-back to prior weights, one `optimal_inaccurate` accepted and recorded. The ladder exists and is tested on a deliberately infeasible cap; the sample never needed it.

**What binds.** The long-only floor binds on at least one asset on every rebalance and on 8.6 of 13 on average; the book holds 2.2–2.3 effective assets (`1/Σw²`), a median largest weight of 61–63%, and reaches a full corner (≥ 11 assets at zero) on 4–8% of rebalances. The 6% ADV cap never binds at `A_low` ($0.4M — the largest participation any trade reached was 1.9%) and binds on 20% of rebalances at `A_high` ($40.6M), where its multiplier's 80th percentile is 0.0048 per period per unit of `|z|` beyond the cap — the priority W6-P2 sets when the cap becomes a hinge. Neither the box nor the turnover limit touched its simplex maximum. **A position limit — the long-only floor — rather than alpha is shaping the output on every date**, which is ruling 7's anticipated finding, and no bound was added to soften it.

**`B` on the optimizer's own book is 1.18** (both spread ends, both book sizes; interval [0.898, 1.101] at 187 non-overlapping months) and 1.16 on the control: H2's mechanism is present in the portfolio that actually trades, monthly, long-only, held for the month. One-way turnover 2.5–3.7 per year, cost drag 7–12 bp/yr, spread-dominated at the small book and half impact at the large one. No Sharpe ratio is reported.

**The control is not family 4 and cannot be compared to it directly.** Long-only minimum variance on this forecast is 97% `govt_2y` (1.06 effective assets), and the unconstrained family-4 portfolio holds a short on every one of the 187 dates, so the registered equality leg is vacuous on this panel; the reproduction is pinned by a unit test on a synthetic forecast instead. That the lowest-volatility asset absorbs a long-only minimum-variance book is arithmetic, not a defect.

**§8.4's initialisation gives `γ_risk` = 85.7 and a third of the target risk, and the reason is measured rather than guessed (row 210, not pre-registered).** The median per-period `IR = √(α̂ᵀΣ⁻¹α̂)` is 3.86. Without the two identity assets it is 4.12 — they are **not** the source, which refuted the session's working suspect. With `Σ` replaced by its diagonal it is 1.55, so `Σ⁻¹`'s pricing of the near-collinear curve points accounts for a factor of 2.2–2.5 (median per-date ratio, and ratio of medians). The level is the return-unit `α̂` meeting an order-of-magnitude range of volatilities: a cross-sectional alpha dispersion of 0.77% per period assigned to `govt_2y`, whose forecast volatility is 0.26% per period, is a single-asset `IR` above 2 before any correlation is used — and the RSTR book holds 61.5% `govt_2y` on average for that reason. The unconstrained `h*` carries `TE_target` by construction; the long-only simplex removes the leverage it needed, and the forecast volatility lands at 0.69% per period against 2.25%. §8.4 calls its formula *"the right initialization, not the answer"* and names the second step — a risk constraint with `λ` recovered as the multiplier, or the cyclic search — which is W6-P2's and a `strategy-config` row when it runs. A volatility-scaled alpha would not do this and needs an IC, which ruling 4 declines. **Ruled at the wrap-up (2026-09-04): this is the sixth published construction in the project measured out of its regime** — RSTR is CNE5's descriptor for a homogeneous-volatility equity universe, and applied across assets running from 0.26%/month to about 5%/month a return-unit `α̂` of ordinary size on the lowest-volatility asset meets `Σ⁻¹` on a near-collinear curve block and dominates the book. It joins Marchenko-Pastur denoising, the parabola smoothing, the specific-risk blend, stage (d) and the PSD repair in `reports/stage_k_dependence.md`. That is ruling 7's corner solution with its cause located. **`α̂` is not changed.** The concentration is the result; W6-P2's per-asset bounds are the grid dimension that measures what it costs. The three constructions of §8.5.1 are approved as recorded, each with its alternative and one config key to replace.

**Two defects found by the tests and fixed before any number was seen.** The penalty's `1/‖α_⊥‖²` exploded on an alpha inside the factor span (a least-squares residual is `O(ε)`, not zero), so `ψ_mis` is carried on the unit direction with an `N·ε·‖α‖` floor; and two runs that must coincide differed in the fifth digit because per-period objectives of `O(1e-5)` sit at the solvers' absolute tolerances, so the objective is normalised by `γ_risk` times the mean variance, which leaves the argmax where it was and rescales the duals back. A third, found by the first launch: RSTR scored on the union calendar had no value at any month end, because any asset's non-trading day poisons every 525-session window; it is scored per asset on its own sessions.

**Deferred, with the owner named.** The grid (§9), `results/metrics.json`, the golden weights fixture and `make backtest` going green are W6-P2. The hinge priorities, the `TE_target` band, `γ_risk`'s second step and the capacity re-optimisation are W6-P2/P3. `α_g` is measured but not published here. The financing leg is still unowned (final cash −0.01% of NAV).

---

# 9. The experiment grid

This is what generates the answer table. Two axes.

**Axis 1 — covariance treatment (7 variants):**

| # | Variant |
|---|---|
| 1 | Sample covariance, equal-weighted window |
| 2 | EWMA, separated vol/correlation half-lives |
| 3 | + Newey-West |
| 4 | + eigenfactor adjustment, `a = 1.0` |
| 5 | + eigenfactor adjustment, `a = 1.4` |
| 6 | + volatility regime adjustment |
| 7 | Ledoit-Wolf (constant-correlation target) — the benchmark to beat |

Plus, orthogonally, Model A (macro) vs Model B (PCA + MP denoise) vs the hybrid.

**Axis 2 — cost treatment (4 variants):**

| # | Variant |
|---|---|
| A | No costs anywhere (the paper portfolio) |
| B | Optimize gross, subtract realised costs afterwards |
| C | Costs in the objective, flat 5bp spread assumption |
| D | Costs in the objective, time-varying EDGE spread + square-root impact, both regimes |

**Report, for each cell:** the two terms of the §1 identity, plus bias statistic on the optimizer-selected portfolio, MRAD, realised min-var volatility, annual turnover, cost drag in bps, gross and net Sharpe with Lo standard errors, and deflated Sharpe.

**Hold out the last 18–24 months.** Write the exact dates down before anything runs. Do not touch it until the grid is frozen. Once you look at it, it is spent.

**Expected headline finding**, to be confirmed or refuted honestly: the risk-model term is small when measured on random portfolios and material when measured on optimizer-selected ones; the eigenfactor adjustment removes most of it; and the cost term — under a time-varying spread model — is larger than the risk-model term in absolute Sharpe, but the cost term is the one that responds to putting costs inside the objective rather than outside it. If that is not what you find, report what you did find. A clean refutation of your own hypothesis is a better interview story than a confirmation.


## 9.1 RULINGS (2026-09-04, W6-P2): the grid is 28 cells, and eight decisions taken before any cell ran

Every gap the W6-P2 orientation found in this section was ruled by the operator before the runner existed, and each ruling is in `config/model.yaml` under `optimizer.grid` with its reason beside it. This is the short record. `experiments.md` rows 211–238 carry the registered legs; `reports/experiment_grid.md` and `results/metrics.json` carry the result.

1. **§8.4's second step happens here, before the grid.** The initialisation `λ = IR/(2·TE_target)` attained 31% of target in W6-P1; 28 cells at a `γ_risk` several times too high would be 28 trials on a configuration nobody would ship. The spec's own route is taken — *replace it with a risk constraint and let the solver recover `λ` as the multiplier* — and not the cyclic search, which evaluates on outcome and is a trial. The `γ_risk` term leaves the objective; `‖Mw‖₂ ≤ TE_target` is imposed hard; the multiplier `μ` is recovered and the `γ_risk` it implies is `μ/(2·TE_target)` (KKT at the bound). **Hinge priorities land here too:** the ADV cap becomes a hinge at 0.0048, the 80th-percentile multiplier W6-P1 observed at `A_high`. The TE band, the capacity re-optimisation and the bounds sweep are W6-P3.
2. **Variant 1's window is derived, not chosen:** `T = T_eff` of the short horizon `= 2τ/ln 2` at `τ = 84` → 242 days, equal-weighted, moments about zero — the equal-weighted estimator with the same effective sample size as the EWMA it is compared against; §6.4's "isolates factor structure at fixed `T_eff`" made literal.
3. **Variant 7 (Ledoit-Wolf) does not go through the pipeline.** Constant-correlation Ledoit-Wolf (2004 JPM; the repository's own, not sklearn's identity target) on the same 242-day window, intensity solved per date, no Monte Carlo, no eigenfactor, no VRA. Registered expectation: it sits near variant 1 (W3-P5 measured `N/T = 0.003` as the regime where shrinkage barely moves; here `N/T = 0.054`). If it does not, the window is short enough that shrinkage has purchase, and 7 becomes the first ladder point where estimation error is measurable on this panel.
4. **The grid is 7 × 4 = 28 cells and nothing else is crossed.** Every cell sits at the W6-P1 verification point on every other dial (patient `Y`, `a = 1.0`, `γ_trade = 1.0`, short horizon, `TE_target` at 1×, no bounds). `γ_trade` (5), TE multiples (3), book size (2), spread ends (2), per-asset bounds (2) and horizon are each a one-dimensional **band** off the reference cell, run in W6-P3 as a diagnostic — the W4-P2b idiom, not a grid. A fully crossed grid of thousands of cells answers a question nobody asked and inflates `N` for nothing. `Y` is an axis-2 ingredient — treatment D runs both regimes inside its one cell — and is not also a band.
5. **Model B and the hybrid are OUT of W6-P2.** Their optimizer-consumable per-date matrices do not exist (the committed histories carry bias series), the cache build is about two hours, and "does the model matter" is a different question from axis 1's "does the treatment matter". Registered for W6-P3 or W8-P1; W3-P5 already predicts Model B's cell lands near the sample-covariance cell. **Horizon: short only.** **Per-asset bounds are a W6-P3 band at two values,** unbounded (1.0) and 2× equal weight (`2/13 ≈ 15.4%`) — the tightest integer multiple that leaves the optimizer a choice, since 1× forces equal weight; recorded as a rule, not a number.
6. **Lo (2002) with `q = 12` for monthly is a derivation, not a parameter.** The GMM standard error uses a Bartlett kernel at the same `q − 1` lags the aggregation factor reads. **The deflated Sharpe reads `N` from `experiments.md`'s running total at run time** so it cannot go stale, against a zero-skill null; `V[SR]` is the variance of the cells' per-period net Sharpes, and the report states that the cells share one alpha and are not the independent trials the formula assumes. PBO and `skfolio` go to W8.
7. **Cost treatment C: 5bp is the FULL spread, so `a_i = 2.5bp`** — the convention `spread_levels.yaml` declares and the code already halves; it is not passed through the halving.
8. **`HOLDOUT_START` is enforced in code, not inherited.** `mafrm.backtest.grid.assert_in_sample` refuses any rebalance, and the close of the last holding period, on or after 2025-01-01, and runs before the first solve of every cell; `tests/test_grid.py` drives it past the boundary. **`N`: 10 → 38 after W6-P2**, stated back and agreed before a single Sharpe existed.

**The book size, ruled at the state-back:** neither W6-P1 endpoint. The reference cell runs at the AUM where the largest equal-weight trade is **2% of its proxy's ADV** — the cost model's lower calibration anchor — by the same rule that placed `A_low` and `A_high`. At `A_low` impact is a rounding error and the four cost treatments cannot be told apart; at `A_high` the 6% cap bound on 20% of rebalances, so every cell would measure a book shaped by a constraint the cost model cannot price beyond. At the 2% anchor impact is of order 40% of cost and the square-root law is measured where it is calibrated. The exact figure is computed by the run: `A = 0.02 × 13 × $31.2M ≈ $8.1M`.

**Two readings the rulings left open, recorded with their alternatives** (`config/model.yaml` marks them READING):

- **Treatment C keeps the square-root impact term** beside the flat 2.5bp half-spread, so that C against D isolates the spread model — H5's question. The alternative, C as a flat spread with no impact, conflates impact with spread time-variation.
- **The reference cell runs at the LOW end of the issuer half-spread interval.** Ruling 4 makes the ends a W6-P3 band, so the reference needs one; low makes the time-varying cost term smallest, the conservative side of H6's ordering claim. One config key; the alternative is `high`.

**One construction the rulings forced:** §8.3's `ψ_mis ≈ λ·σ²(α_⊥)` needs a `λ` before the solve and in constraint mode `λ` is recovered by the solve. Two passes per rebalance — the first with the penalty off recovers `μ`, the second prices the penalty at the implied `γ_risk` and is the solution. Deterministic, causal, no lag, no invented number, one extra 10ms solve. The alternatives — the initialisation `λ` (three times too high) or the previous rebalance's multiplier (path-dependent) — are named and not taken.

**One construction found by the first launch:** the dense variants are estimated on the assets' true daily excess returns (complete cases from the 2007 sample start), because the residual panel the pipeline variants forecast starts in April 2009 and holds fifteen rows before the first rebalance. They forecast exactly the series the book is marked on; the pipeline variants' regressand differs from it for the four government zeros by the carry-and-roll leg (§4.3). Stated in the report.

### 9.2 FINDING (2026-09-04, W6-P2): every cell solves on target, `B` on the constrained book is 1.03–1.07, and the cost term is the larger one where costs are not priced

`experiments.md` rows 211–238 and their RESULT section; `reports/experiment_grid.md`, `reports/experiment_grid.png`, `results/metrics.json`. Registered before the run; nine legs hold and three are refuted.

**The second step works, and the initialisation was ten times off.** The hard forecast-volatility bound binds on 98–99% of rebalances in every cell; rms attainment 0.997–0.999 against W6-P1's 31%. The `γ_risk` the recovered multiplier implies is 7–8.5 against §8.4's `λ = IR/(2·TE_target)` of 57–86 on the same dates, because the initialisation prices the unconstrained optimum's leverage and the long-only simplex has none. Effective assets 2.5–2.9, median largest weight 46–53%, a full corner on 9–19% of rebalances. The ADV hinge was active on at most 1.1% of rebalances.

**`B` is 1.03–1.07 in every cell — a third of W6-P1's 1.18 on the same matrices — and inside the χ² interval [0.898, 1.101] in all twenty factor-model cells.** Direction consistent (`B > 1` in all 35 runs; dense variants 1.03–1.05, factor variants 1.06–1.07; the eigenfactor stage moves it by ≤ 0.003, H3's refutation persisting), magnitude not. Suspect, post hoc and untested: control C1 located family 4's bias in the diagonal specific-risk assumption, which bites on a concentrated low-volatility book and is a smaller share of the forecast at target risk, where the book sits in the equity, commodity and long-duration sleeves. **Registered for W6-P3's `TE_target` band:** `B` on the reference cell falls as the multiple rises, and exceeds 1.10 at 0.5×. **Under the operator's name (W6-P2b):** the initialisation was 10× off, not 3×; the 3× was a linear guess from 31% attainment, and the simplex has no leverage to price.

**The identity, split — on Lo's scale (W6-P2b).** Volatilities are annualised by `12/η` of the net series so that `SR_paper` and `SR_real` share the Sharpe columns' scale; on every cost-free cell `gross SR × B = SR_paper` to six decimals, asserted in code. Risk-model term 0.013–0.058 and cost term 0.038–0.068 in annualised Sharpe units on Sharpes of 0.7–0.9 with Lo standard errors of 0.25–0.27. **§9's expected headline, clause by clause:** *the cost term responds to putting costs inside the objective* — **holds** at every variant (4B 0.068 → 4D 0.040); *the eigenfactor adjustment removes most of the risk term* — **refuted again**, `|ΔB| ≤ 0.003`; *under a time-varying spread model the cost term is larger* — that clause is treatment D and nothing else, and it is **refuted at four of the five factor-model cells under D and at the dense sample cell** (4D: risk 0.054 against cost 0.040), holding only at variant 2 (EWMA, the grid's smallest risk term) and at Ledoit-Wolf. Under B and C, the two treatments this section deprecates, the cost term is ahead everywhere — B trades as if trading were free (46–51 bp/yr), C over-prices the spread 2–10×. **The better the cost treatment, the more of the residual gap belongs to the risk model.** Confound, named: the dense variants forecast the assets' true excess returns while the pipeline variants' regressand differs for the four zeros by carry-and-roll, so the ~0.03 gap in `B` between dense and factor cells is not the diagonal alone, and 0.03 is inside the per-cell interval regardless. The cost term responds to placement exactly as §9 expected: D pays the least (27–30 bp at patient `Y`, 34–38 at urgent) and has the highest net Sharpe at every variant.

**Every cost figure is a lower bound on the spread leg.** The reference cell runs at the LOW end of the issuer interval, which `config/spread_levels.yaml` itself says is a floor and not a value: SPY and IWM trade at exactly zero spread in every cell (SSGA puts SPY near 0.6bp). The 28 cells stay as run. **Registered for W6-P3's spread-end band:** at the high end the cost term at 4D exceeds 0.0544, the reference cell's risk-model term. If it does the reference cell's answer flips with the spread reading and the honest answer is a range; if not, the risk model is ahead at 4D under any reading the disclosures permit.

**Ledoit-Wolf sits on the sample** (ruling 3's registered expectation): `|ΔB|` 0.008, realised min-var volatility within 4.6%, median intensity 0.040. The five pipeline variants' min-var volatilities reproduce `reports/bias_statistics.md` to the third decimal.

**H5 and H6 are recorded at §1.** The deflated Sharpe is 0.99 in every cell at `N = 38` and says almost nothing: `V[SR]` across cells sharing one alpha is 8.8e-5 per period, so the false-strategy bracket is 0.02 per period against Sharpes of 0.19–0.22; the search it would deflate was never run. Lo's `η(12)` is 3.8–4.1 against `√12 = 3.46`.

**Residual, named.** Clarabel returns `optimal_inaccurate` on 22–37 of 187 rebalances in the treatment-D factor cells (one in all of W6-P1); every such point is finite, on the simplex and inside the bound, and recorded. Suspect: the TE second-order cone meeting the impact power cone at an objective scale of `max|α|` once the `γ_risk` term no longer sets it. W6-P3 re-solves those dates with the fallback solver and reports the L1 distance; above 1e-3 the normalisation changes and the grid re-runs as new rows.

**Deferred, with the owner named.** **First: the financing leg is still unowned (final cash within 0.01% of NAV), and it must be owned before W6-P3's bands produce a Sharpe** (operator, W6-P2b wrap-up). Then the bands (`γ_trade`, TE multiple, book size, spread end, per-asset bounds, horizon), the capacity re-optimisation, the `optimal_inaccurate` re-solve and the W6-P1-style `spec_initialisation` comparison, all W6-P3. Model B and the hybrid cells are W6-P3 or W8-P1.

## 9.3 RULINGS (2026-09-04, W6-P3): six decisions taken before any band ran, and the twelve bands registered

The session that first received rulings 1-4 was lost to a network drop; they were re-issued verbatim with rulings 5 and 6 and are recorded here once. Every ruling is in `config/model.yaml` (`backtest.financing`, `optimizer.grid.bands`, `.resolve`, `.capacity`, `.diagnostics`) with its reason beside it; `experiments.md` rows 239–253 carry the registered legs; `reports/bands.md`, `reports/capacity_reoptimised.md`, `reports/diagnostics.md` and `results/metrics.json` carry the results. The order ruled: own the financing leg first, then the re-solve, then the bands, then the capacity re-optimisation; §10.1–10.3's diagnostics run on every cell in the same pass.

1. **Financing: cash earns and pays the Ken French daily RF both ways, no spread, in every cell uniformly.** This invents no rate — §1's identity is written on returns already in excess of cash, so a fully invested book financed at RF has excess return exactly `Σ w_i (r_i − rf)`; cash at RF is the definition the Sharpe already assumes. A borrowing spread would be an invented number, and the leg is −0.01% of NAV by construction (long-only, `Σw = 1`; negative cash is cost paid out of NAV), so any spread's effect is below the cost column's resolution. Under treatment A the cash is zero, so the leg is zero; uniformity costs nothing. **Construction:** the engine marks the book on the excess-return index `(1 + excess).cumprod()`, in which frame the accrual on cash is `rf − rf = 0` by identity, so the engine carries cash at zero and the leg is REPORTED per cell in nominal bp/yr beside the cost drag, so that "negligible" is a computed line. `engine.run_backtest` also takes a `cash_rate` for a caller on nominal prices, and a test pins that the nominal run at `cash_rate = rf`, deflated by the bill, reproduces the excess-frame run exactly.
2. **Reconciliation (§10.3): two assertions, one exact and one statistical.** (i) Returns: `realised active return = Σ_k x_k f_k + specific` is an identity given the same `X` and residuals defined as the remainder — asserted to `1e-12` of scale, the `identity_terms` form. (ii) Variance: ex-ante shares are forecasts and ex-post shares realisations, so they differ by exactly what `B` measures; the test is §6.1's statistic on each additive-in-variance component — the ex-post/ex-ante factor-variance ratio and specific-variance ratio, square-rooted, each against the same exact χ² interval `B` uses at the same sample size (`[0.898, 1.101]` at 187 months). Checked against the spec's own example: 60% versus 5% gives a factor-component `B` of `√(5/60) ≈ 0.29`, and it fires (a test pins it). No new constant. In tests and on corrupted input it raises; on the real run it writes ratio and interval to the report and marks the cell — a refutation on real data is a finding, not a reason the grid cannot run.
3. **Brinson-Fachler benchmark: the equal-weight book.** The project's standing absence of a choice and already the anchor the book is sized against (`TE_target` = 1× EW realised volatility). `w^b = 0` stays the risk benchmark in the identity; the two are different objects and the report says so. The EW return series is reported for attribution; its Sharpe is not computed — a benchmark that acquires one becomes a trial (a test asserts the diagnostics module names no Sharpe).
4. **Capacity points are `data-diagnostic`, on one condition:** the curve reports `TC(A)`, turnover and net-return drag per AUM point and NO Sharpe per point. The Sharpe-bearing points on the curve are `A_low` and `A_high`, already `strategy-config` in the book-size band. Reversing result, named now: if any AUM point acquires a Sharpe, all ~100 become `strategy-config` and `N` goes to ~150. `N = 50` after the bands (12: `γ_trade` 4, TE 2, book 2, spread 1, bounds 1, horizon 1, `spec_initialisation` 1), stated before the first band ran; the running total is confirmed after. **Construction:** the ADV cap is imposed HARD ("with the ADV constraints binding") and the ladder drops it before the TE bound so a relaxation is logged as the cap's; `TC(A)` on the curve is the RECURRING drag — per-rebalance cost over `NAV_pre`, annualised, the build from cash excluded, the same normalisation `structure_kappa` uses, so at the reference NAV the closed form on the reference cell's own structure reproduces its recurring cost exactly (a test pins it).
5. **Perold: decision price = the rebalance close, upheld, and it is a finding.** A monthly simulation that executes at its decision price has no execution schedule, so delay and opportunity are zero by construction and the decomposition is degenerate; the report says so, and says that the "you own the decision timestamp" advantage is realisable only with an intra-period schedule that does not exist. An earlier timestamp would be invented. The predicted-versus-realised residual is defined so it is not identically zero: predicted = the cost the optimizer priced in its objective at the decision; realised = the §7.1 cost charged on realisation. Zero under A; under B nothing was priced, so the residual is the realised cost itself and carries no calibration information; the measurement is C (flat spread against the true one) and D (the as-of spread the optimizer saw against the as-of spread charged on the same trade).
6. **The re-solve criterion is the MAX per-date L1, not the median** — the median hides the worst date and this is a correctness check. A cell re-runs as new rows if any flagged date exceeds `1e-3` (the threshold registered at §9.2). **AMENDED (2026-09-04, W6-P3b, under the operator's name: the criterion as first ruled was mis-designed — "L1 to the fallback's point" presumed the fallback was the reference, and on 2015-01-30 the L1 was the fallback's error).** Two points are compared only when both are feasible to the solver's tolerance, and the feasible point with the better objective — higher, since §8.1's problem is a maximisation — is the solution; the L1 to an infeasible or worse-objective fallback is the fallback's error, reported and not acted on. `grid.run_cell` now records, per flagged date, the re-solved point's `σ_f/TE_target`, the relative objective gap and the verdict, and a cell owes a re-run only where the L1 is above `1e-3` AND the fallback's point is feasible AND better. Under the amended criterion 3D/patient 2020-03-31 stands (the fallback's objective is 0.34% higher but the point is 0.07% outside the bound), and `A_high` 2015-08-31 — the one date on which both points were feasible — is settled by the objectives: primary 0.0076412, fallback 0.0076315, the primary's better by 0.13%; nothing moves and `N` stays 57. A third normalisation or a third solver would have been tuning on outcome and was rightly not taken.

**Ruled values added to the config with their source:** the per-asset bound at `2/N` (§9.1 ruling 5), the L1 threshold `1e-3` (§9.2), and `spec_initialisation` as a band mode (§9.1 ruling 1). **Model B and the hybrid** have no per-date optimizer-consumable caches (`data/processed/statistical` and `hybrid` hold bias series only); their build is the two hours W6-P2 expected and they are registered for W8-P1.

**The twelve bands, each one dial off cell 4D/patient, each a `strategy-config` row with its falsifier written before it ran** (`experiments.md` rows 239–250): spread at the HIGH end (registered W6-P2b: the cost term at 4D exceeds 0.0544, the reference's risk term); `TE_target` at 0.5× and 2× (registered W6-P2: `B` exceeds 1.10 at 0.5× and falls as the multiple rises); book size at `A_low` and `A_high` (impact under 10% of cost and a smaller cost term at `A_low`; the hinge active on ≥ 10% of rebalances and a larger cost term at `A_high`); `γ_trade` at 1.5, 2.0, 2.5, 3.0 (turnover and cost drag non-increasing along the sweep; net Sharpe reported, not predicted); the per-asset bound at 2/13 (`|B − 1|` smaller than the reference's); the long horizon (lower turnover and §6.1's `B` inside its interval); and `spec_initialisation` (attainment below 0.5 and `B` above the reference's). The capacity re-optimisation carries four registered legs (rows 251–252): the holdings differ across AUM (the assertion §10.4.1 requires), the re-optimised `TC(A)` at or below the rescaled one at or above the reference NAV, the re-optimised break-even at or beyond the closed form's `A_BE`, and the hard cap binding at the grid's top. The re-solve (row 253): MAX L1 below `1e-3` in every cell.

### 9.4 FINDING (2026-09-04, W6-P3): the financing leg is 0.07 bp/yr at most, the `optimal_inaccurate` flags were the normalisation and the fallback solver, and the grid reproduces

`experiments.md` rows 253–260; `reports/experiment_grid.md` (the financing and re-solve table), `results/metrics.json`.

**The financing leg, owned.** Under ruling 1 the engine carries cash at zero in the excess frame and reports the nominal size of the leg per cell: −0.037 to −0.072 bp/yr in every charged cell, zero under A, on a mean cash balance of −0.02% to −0.04% of NAV and a final balance within −0.006% of NAV. "Negligible" is now a computed line, and it is a hundredth of the smallest cost drag in the grid.

**The re-solve, and what it found (ruling 6).** On the first pass 22–37 flagged rebalances per treatment-D factor cell, as W6-P2 reported, and **seven cells with a MAX L1 above `1e-3`** (2D/urgent 7.4e-3, 3D/patient 2.0e-3, 4D/patient 3.1e-3, 4D/urgent 4.7e-2, 5D/patient 1.1e-2, 6D/patient 5.2e-2, 6D/urgent 3.0e-3) — row 253 **REFUTED**. Diagnosed on the worst date (4D/urgent, 2015-01-30): the primary's flagged point is feasible (`σ_f/TE_target` = 1.0000) and within 3e-7 of the point the same problem gives with status `optimal` once the objective is left in its own units, while the fallback's point violates the TE bound by 2.7% and is the whole of the 4.7e-2 distance. W6-P2's suspect — the objective scale of `max|α|` once the `γ_risk` term no longer sets it — is confirmed as the cause of the flag, not of a wrong solution. As registered, the normalisation changed (scale 1 when there is no risk term; `optimizer._objective_scale`), the seven cells re-ran as rows 254–260, and every cell was checked against its W6-P2b weights: **largest per-date L1 8.0e-4 over all 35 runs (median per cell 1e-7 to 3e-5), every Sharpe-type figure identical to the third decimal, the flags down from 22–37 to 0–1 per cell.** Rows 254–260's prediction HOLDS. The registered criterion needed the feasibility of both points to be a correctness check; the re-solved point's `σ_f/TE_target` is now recorded beside every L1.

**Residual, named — and, in W6-P3b, settled under the amended criterion.** One flag survives the normalisation change: 3D/patient on 2020-03-31, L1 8.9e-3, the fallback's point 0.07% outside the TE bound with an objective 0.34% higher — a flat optimum along a 0.9%-of-weight direction on the month the book turned over 1.43× its NAV. It is one date of 187 in one cell, below the third decimal of that cell's Sharpe. The cell was not re-run a third time, which departed from the letter of ruling 6 as first written; the operator accepted the departure and amended the criterion (ruling 6 above): the fallback's point is infeasible, so the distance is the fallback's error, reported and not acted on. The grid report's re-solve table now carries the verdict per flagged date.

**Two things confirmed by reproduction.** The 28 cells, re-run twice under the new rulings (once with the financing line and diagnostics, once with the normalisation), reproduce W6-P2b's answer table figure for figure; and the golden-weights fixture moved by at most 8.7e-7 per weight under the normalisation and was regenerated deliberately.

### 9.5 FINDING (2026-09-04, W6-P3; long-horizon paragraph restated W6-P3b): the bands — the spread reading does not flip the answer, `B` is a property of the book's concentration, and the long-horizon band's lower `B` is two errors cancelling

`experiments.md` rows 239–250; `reports/bands.md`, `reports/bands.csv`, `reports/bands.png`; `results/metrics.json` (`bands`). Twelve points, one dial each off cell 4D/patient (`B` 1.061, risk-model term 0.0543, cost term 0.0400, net Sharpe 0.850 with Lo SE 0.264), `N` = 57 at run time; nine registered legs HOLD, two are REFUTED, nothing selected.

**The spread reading does not flip the reference cell's answer (row 239, REFUTED as registered).** At the HIGH end of the issuer interval the cost term is 0.0424 against the reference's risk-model term 0.0543: the risk model is ahead at 4D under any reading the disclosures permit, which is the alternative W6-P2b named. The high end moves the cost term by 0.0024 and turnover from 5.62 to 5.48; `B` does not move.

**`B` falls as the TE multiple rises, and exceeds 1.10 at 0.5× (rows 240–241, HOLD).** 1.139 at 0.5×, 1.061 at 1×, 1.015 at 2×. W6-P2's post-hoc suspect — that the diagonal specific-risk misspecification bites on a concentrated low-volatility book and is a smaller share of the forecast at target risk — is now a tested one. At 2× the simplex cannot carry the target: the bound binds on 60% of rebalances, rms attainment 0.949, effective assets 1.64, and the net Sharpe falls to 0.638 because the book is a corner most of the time (SPEC.md 9.1 ruling 1: reported at the risk it attains).

**The per-asset bound takes `B` below one and the risk-model term negative (row 248, HOLDS).** At 2/13 the book is spread across 6.96 effective assets against 2.71, `B` is 0.991 and the risk-model term −0.0070; the TE bound is slack on 41% of rebalances (attainment 0.938) because the bounded simplex cannot reach 2.25% a month either. The same mechanism from the other side: take the book off the diagonal and the bias goes with it — at a cost of 0.12 in net Sharpe, since the bound also takes the book off the alpha.

**The long-horizon band (row 249, HOLDS as registered) has a lower total `B` — 1.017, risk term 0.0156 against 0.0543, net Sharpe 0.874 against 0.850, turnover 5.36 against 5.62 — and its components say why: factor `B` 0.932 against the reference's 0.972, specific `B` 1.278 against 1.312. Two errors cancel.** The specific component is still 1.2+, the factor component moved further BELOW one, and the total closed because the two errors offset on this book — CLAUDE.md failure mode 7, not the window closing the term. A band moved one dial and a better `B` came out, which is what a band reports; the reference cell stays the short horizon and nothing is selected. Nor is the horizon one dial: the long panel moves the factor volatility half-life (84 → 252) AND the specific-risk half-life (84 → 252) together, and its VRA half-life (42 → 168) differs too, though variant 4 is pre-VRA so that one does not enter this band; "the estimation window" is therefore an inference, not a measurement. **Registered for W8-P1, not run:** `B` on the reference book across W4-P2b's τ sweep — the factor volatility half-life alone, every other half-life at the short horizon's value — each a `strategy-config` row; prediction, `B` falls monotonically in `T_eff`; reversing result, `B` at τ = 252 with the specific-risk half-life held at 84 stays near 1.06, in which case the long band's factor `B` came from the specific leg or from the two moving together.

**Book size (rows 242–243): one leg REFUTED, one HOLDS.** At `A_low` ($406k) impact is 19.5% of the realised cost, not under 10% — the registration's "impact is a rounding error at `A_low`" was wrong, because the thinnest proxies (the 30-year and the 2-year through their ETFs) carry impact at any book the equal-weight rule sizes off the thinnest ADV; the cost term is 0.0294 against 0.0400 as predicted. At `A_high` ($40.6M) the hinge is active on 10.7% of rebalances, impact is 57% of cost, the cost term 0.0408 — and the net Sharpe is 0.891, the highest in the band, with `SR_paper` 0.991: **the ADV hinge acts as a turnover regulariser** (4.45 against 5.62) and the book it produces earns more gross. Reported, not selected.

**`γ_trade` (rows 244–247, HOLD): turnover and cost drag fall monotonically, 5.62 → 4.08 and 28.2 → 14.8 bp/yr, while the net Sharpe falls 0.850 → 0.826** (0.847, 0.836, 0.820, 0.826): on this book a safety margin above 1.0 buys cost and pays alpha, and the desks' ">1" carries no Sharpe on this panel. `B` is unmoved (1.063–1.065).

**`spec_initialisation` (row 250, HOLDS): attainment 0.305, `B` 1.109 (§6.1's 1.180), the initialisation's `λ` 85.7 against the recovered 7.4.** What §8.4's second step changed, measured on the reference book: the initialisation trades a third of the target's risk at a third of the turnover, and its `B` is W6-P1's, not the grid's.

**The re-solve on the bands, settled under the amended criterion (W6-P3b).** Four band cells carry a flagged date with an L1 above `1e-3` after the normalisation change (book `A_high` 2.5e-3, `γ_trade` 2 4.6e-3, 2.5 1.1e-3, 3 2.4e-3); on three of them the fallback's point is 0.07–0.14% outside the TE bound — the fallback's error, reported. On `A_high` 2015-08-31 both points are feasible (`σ_f/TE_target` 1.0000 and 0.9984) and the objectives decide: primary 0.0076412, fallback 0.0076315, the primary's better by 0.13%. Nothing moves; `N` stays 57.

## 9.6 RULINGS (2026-09-06, W8-P2): five decisions taken before the boundary moved, and the ledger reconciled first

Every gap the W8-P2 orientation found was ruled by the operator before the runner existed. `experiments.md` row 334 carries the registration; `reports/holdout.md` and `results/metrics.json` (`holdout`) carry the result. The order ruled: reconcile the count, then freeze, then run.

0. **The trial count is `N` = model-config + strategy-config, and the W6-P1 gap is closed HERE, not at W8-P4.** The W8-P2 brief's original wording — "the DSR trial number is the strategy-config count" — was written in week one, before the category system existed, and is **superseded**: §6.6, `experiments.md`'s category table and `mafrm.backtest.metrics.read_trial_count` all define `N` as both counted columns, and moving to the smaller number at the final step is exactly the reclassification §6.6 forbids, in the direction where a smaller `N` flatters the deflated Sharpe. The three-row gap carried since W6-P1 turned out **not** to be three uncategorised rows: rows **100** and **118** are registered-and-never-run and were wrongly swept into the *total* at W6-P1, and row **164** ran in W8-P1 and was never added to a *column*. Corrected: 261 / 12 / 58 = **331** evaluated, `N` = **70**. This happens before the run because the holdout is evaluated once and its DSR reads the count at run time; a later audit moving a row into a counted category would make the holdout's DSR wrong and unfixable.
1. **One cell, one row.** Running 28 cells would spend the window on a comparison nobody may act on and would invite *"which cell did best out of sample"*, which is selection on the holdout. **One addition, registered:** the equity model's family-4 bias and its factor/specific split on the same window, in the same pass — the control for the whole thesis, whose code is frozen and which cannot be run later. **No Sharpe on any equity family** (W7-P3's benchmark rule), so it stays `data-diagnostic`.
2. **The estimates walk forward; the code and config are frozen.** Freezing the covariance at 2024-12-31 and marking twenty months against it would test a stale matrix, not the method: a risk model is used by re-estimating it from data available at each rebalance. *"Do not tune"* means no parameter changes, not a frozen estimate. **The two exceptions are configuration rather than estimates and are injected from the in-sample universe:** `TE_target` (0.022526389 per period) and the NAV ($8,119,893). Recomputing either on the extended panel would set the holdout's risk target from the holdout's own realised volatility, and the runner asserts the injected target survives.
3. **`make data` is not run.** The cache is verified with no drift at `9ba15f3`, and the row must cite a manifest that exists when the run happens; a fresh pull adds a few days while rewriting every hash. **The right edge is the last date on which every series in the book has an observation** — the minimum of the per-series last observations, recorded by name — so the book is marked on a complete cross-section and a ragged tail cannot silently change the universe.
4. **Expectations, not gates.** The row states what is expected and why, so that a surprise is visible as a surprise; none of it is a threshold to re-run against. **Nothing is tuned after the result**, no second configuration is run, `experiments.md` is not re-partitioned and no tolerance is widened. If the holdout is worse than in-sample, that is the result and it goes in the README as the result.

**The mechanism, and why it is shaped to be spent.** `mafrm.data.holdout.evaluating_holdout` is the one sanctioned crossing: it moves the boundary globally for the duration of a block and restores it with an assertion, refuses a nested crossing, refuses an empty reason, and has exactly one caller, which a test pins. It moves the boundary through `config.load` because `cache.read` takes no config — the boundary is enforced where data enters the process, so moving it has to move there too, and both mechanisms then move together and cannot disagree.

**Three controls, ruled to run before or around the crossing.** (i) The same code path re-runs the frozen cell over the **in-sample** window and the run refuses to proceed unless it reproduces W6-P2b's published reference cell; (ii) the last 40 in-sample eigenfactor rows are rebuilt off the *extended* panel and must agree with the committed cache — extending the panel cannot change an in-sample forecast, and rather than assume it, it is measured; (iii) the evaluation is deterministic and was executed twice, bit-identically.

## 9.7 FINDING (2026-09-06, W8-P2): the shipped book's `B` moved 0.007, and the one statistic the window could resolve says the risk-model term is LARGER out of sample

`experiments.md` row 334; `reports/holdout.md`, `reports/holdout_per_year.csv`, `reports/holdout_monthly.csv`, `results/metrics.json` (`holdout`). Window **2025-01-01 to 2026-07-31**, 18 monthly rebalances, right edge bound by Ken French's daily table.

**Read the standard error first, and it was registered first.** At `T` = 18 the exact χ² interval for `B` is **[0.667, 1.333]** against [0.898, 1.101] at 187, and Lo's standard error is **0.86** — about the size of the in-sample Sharpe itself. Net Sharpe **1.0374 ± 0.8586** against the in-sample 0.850 ± 0.264 is therefore **not** evidence of anything, and the deflated Sharpe of **0.7408** at `N` = 70 clears no conventional bar. Five of the seven registered expectations hold: `B` **1.0681** against 1.061; cost drag **29.9 bp/yr** against 28.2; turnover **4.58×** against 5.62; the identity's ordering held (risk **+0.0746** ahead of cost **+0.0572**, as in-sample); the DSR is uninformative as predicted.

**The resolvable finding is daily, and it goes against the project's own registration.** The daily bias statistics have `T` = 391 and an exact interval of **[0.930, 1.070]**, twenty times tighter. Family 1 reads 0.977, family 2 **1.087**, family 3 1.037 and family 4 **1.6673** against an in-sample 1.3321: **the family-4-minus-family-2 gap is +0.580 out of sample against +0.322 in.** H1 survives — random portfolios are near 1 for the same matrices, out of sample as in. **H2's registered 1.2–1.5 band is exceeded at 1.67.** The interval is optimistic (the forecast series behind the 391 observations is a slowly-moving rolling estimate, so consecutive `b_t` are not as independent as it assumes, failure mode 9), which is why family 2's 1.087 — 0.017 outside — is recorded as *not distinguishable from 1*, while family 4's 1.667, 0.60 outside, would survive a large widening.

**What protected the shipped book was the constraint set.** Family 4 is the *unconstrained, daily-rebuilt* minimum-variance portfolio; the identity's `B` is the *constrained, monthly* book carrying the TE bound, the long-only simplex and the ADV hinge. In-sample the pair read 1.332 / 1.061; out of sample **1.667 / 1.068** — the unconstrained statistic deteriorated by 0.335 and the shipped book's by 0.007. That is consistent with §9.5's `TE_target` and 2/13 position-bound bands, which found in-sample that the bias is a property of how concentrated the book is allowed to get, and it is the first out-of-sample evidence for it. **One window, one configuration: consistency, not proof.**

**§10.3's specific leg fires again, and harder.** The attribution reconciliation's component `B` is factor 0.955 (inside), **specific 1.840 (outside)**, total 1.070, against an in-sample specific of 1.23–1.32 that already fired in all twenty factor-model cells. Same mechanism, further out: §6.2.6's diagonal specific-risk misspecification, in the component the model actually gets wrong.

**The book underperformed its benchmark.** Brinson-Fachler against the equal-weight book (§9.3 ruling 3): portfolio **+8.42%** cumulative against the benchmark's **+10.57%**, the −2.16% sitting in allocation (−5.81%) rather than selection (−0.88%). A concentrated low-volatility book in a window where the broad book ran. Reported because it is the comparison a reader asks for, not because 18 months supports a claim.

**The equity control is uninformative out of sample at this length (AMENDED 2026-09-08, W8-P2b).** The first write-up called it *"refuted on both halves"*; that was too strong and is withdrawn. Family-4 `B` at `K` = 54 is **0.9417** daily-held (`T` = 373) and **0.9469** monthly (`T` = 18), against in-sample 1.0342 and 1.1077; the components are factor **0.879**, specific **1.303**, total **0.837** against 1.089 / 0.948 / 0.999. Tested two-sample (`F = B_out²/B_in²`), **three of the five legs cannot be distinguished from their in-sample values at all** (`p` = 0.46, 0.31, 0.40), and the two that can are nominal only: `p` = 0.018 (daily) and 0.042 (specific) against a Bonferroni level of **0.01** for the five comparisons made. Every holdout leg sits inside its own null interval, so nothing here refutes the model. The one leg with real power — the daily statistic at `T` = 373 — carries §15.1.3's named confound directly: the equity bias moves with window length for reasons no second-order account reaches, so a change between two windows of different length is confounded with exactly the unexplained mechanism. **The registration expected persistence and did not get it; "the diagonal left its regime" is not a reading these numbers support, and W8-P3 writes what they do.**

**The run was clean and the record is complete.** All 18 rebalances solved at rung 0 — no relaxation, no fallback solver, no `optimal_inaccurate`, nothing flagged. Perold reconciles to 1e-15 (degenerate by construction, §9.3 ruling 5). 2026 realised 11.72% volatility against a 7.80% forecast, one episode and not a pattern. **Two aborted invocations are logged at row 334**: both raised inside the equity leg, after the macro leg had computed and before anything was printed, rendered or written, so no holdout figure was seen until the third completed. The window was opened three times and read once.

---

# 10. Diagnostics and reporting

## 10.1 Perold implementation shortfall

Measure at the **parent** (metaorder) level; child fills netted against a moving benchmark understate impact because the benchmark has already absorbed the parent's own footprint.

```
IS = r_paper − r_real
```
where the paper portfolio transacts the full intended quantity instantaneously at the **decision price** with zero commissions and zero impact. Decompose into **delay / impact / opportunity (unfilled) / fees**.

Because your build generates its own trade list, **you own the decision timestamp** and can do the full four-way Perold decomposition — vendor TCA usually cannot, because the decision timestamp is not reliably logged, and falls back to arrival-price IS. Say this. It is a genuine advantage of a simulated book and it demonstrates you know what the standard measurement actually is.

Report predicted-vs-realised cost residual every period. That residual is what recalibrates the model — the loop is: TCA measures cost → cost model updates → optimizer updates → trade list updates.

### 10.1.1 FINDING (2026-09-04, W6-P3): the decomposition is degenerate by construction, and the residual measures the spread model

`reports/diagnostics.md` (section 10.1), `reports/diagnostics.csv` (per period, the reference cell). Under ruling 5 the decision price is the rebalance close and the parent order is filled at that close in full, so **delay = 0 and opportunity = 0 by construction, `IS = impact + fees` exactly, and `fees = 0`** (`costs.commission_bps` is an absence). The four-way decomposition this section calls a genuine advantage of a simulated book is realisable only with an intra-period execution schedule that does not exist here; an earlier decision timestamp would be invented. Said in the report rather than padded. `IS` per cell is the realised §7.1 cost over `NAV_pre`, 27–53 bp/yr across the charged cells.

**The predicted-versus-realised residual.** Zero under A. Under B nothing was priced, so the residual is the realised cost itself (48–53 bp/yr) and carries no calibration information. **Under C the flat 2.5bp half-spread UNDER-prices its own realised cost by 5.8–8.1 bp/yr** (predicted 24–28 against realised 32–34): the realised half-spread per unit turnover is 3.7 bp at the LOW end of the issuer interval once EDGE's widening is added (governments and credit 0.5–1.9 bp, equities, gold and commodities 5–8.5 bp), and C trades the expensive names as if they cost 2.5. W6-P2's "C over-prices the spread 2–10×" was a statement about the issuer LEVEL alone; both are true of different objects. **Under D the residual is exactly zero:** the optimizer priced the as-of spread the engine charged on the same trade, so a "forecast against realised EDGE" gap is not measurable in this build — the engine charges the as-of value, at most a month stale, and no spread realised after the decision is carried. The recalibration loop this section describes therefore has nothing to update from D and a spread-model error of 6–8 bp/yr to update from C.

## 10.2 Tracking error

```
Σ = X F Xᵀ + Δ
x_a = Xᵀh                                active factor exposures
σ_a² = x_aᵀ F x_a + hᵀΔh                 exactly additive in VARIANCE
```

**Additivity is in variance only.** Reporting a "factor TE" and a "specific TE" that sum to total TE is wrong. Report `√(x_aᵀFx_a)` and `√(hᵀΔh)` and state that the *variances* add.

Risk contributions via **x-σ-ρ** (Menchero & Davis, *JPM* 37(2), 2011): `CTR_k = x_k·σ_k·ρ_k` — exposure times volatility times correlation with total active return. Algebraically identical to the Euler decomposition but far more interpretable: it separates how much you bet from how risky the bet is from how much it diversifies. Percent contributions can be negative for diversifying exposures — that is correct and informative.

**Ex-post will exceed ex-ante and you must explain why**, not apologise for it. Hwang & Satchell (2001): ex-ante conditions on *fixed* weights; ex-post weights are stochastic (they drift between rebalances), and the extra terms are non-negative. Lawton-Browne suggest the effect can roughly *double* annual ex-ante TE. Add to that: the risk model lags regime shifts in both directions, and the missing-factor problem (§8.3) is by construction absent from `σ_a` but present in realised returns.

### 10.2.1 FINDING (2026-09-04, W6-P3): 90% factor variance, volatilities that would overshoot by 20%, and a quarter of the contributions negative

`reports/diagnostics.md` (section 10.2). On every factor-model cell the factor part is **90.3–91.3% of forecast VARIANCE** and the specific part 8.7–9.7%; the two volatilities `√(x'Fx)` and `√(h'Δh)` would sum to **1.20× `σ_a`**, which is the size of the error this section warns against, and `mafrm.backtest.diagnostics.assert_variance_additive` refuses a decomposition whose volatilities add (tested). On the reference cell the x-σ-ρ contributions (Menchero & Davis 2011) average equity 46%, rates level 21%, commodity 10%, dollar 6%, slope 5%, credit 2.5%, specific 9%; **23–25% of the (factor, rebalance) contributions are negative** — diversifying exposures, reported as such. The dense variants enter as `X = I`, so every "factor" is an asset, the specific share is zero by construction and 13% of (asset, rebalance) contributions are negative.

## 10.3 Attribution, reconciled against the risk model

```
r_p − r_b = Σ_k (x_k − X_k)·f_k + specific
```

**The same `X`, `F`, `Δ` that generate ex-ante TE must generate ex-post attribution.** Risk and attribution share one model object. If the ex-ante decomposition says "60% of TE is the rates level factor" and the ex-post attribution says the rates factor contributed 5% of active return variation, **that is a model failure and the pipeline should catch it automatically.** Make it an assertion, not a chart someone might look at.

Use **Brinson-Fachler**, not Brinson-Hood-Beebower, for the sleeve-level sanity check: `A_i = (w_i − W_i)(b_i − b)`. The `− b` correctly penalizes overweighting a sleeve that underperformed the overall benchmark. Link multi-period with **Carino** (logarithmic scaling, order-independent).

### 10.3.1 FINDING (2026-09-04, W6-P3): the factor component reconciles, the specific component does not — in all twenty factor-model cells — and that is §6.2.6 measured a third way

`reports/diagnostics.md` (section 10.3). Assertion (i) holds on every day of every period in every cell to 1e-12 (it is why no column for it appears). Assertion (ii): **the factor component's `B` is 0.94–0.99, inside `[0.898, 1.101]`, in every factor-model cell; the specific component's `B` is 1.23–1.32, OUTSIDE the interval, in all twenty; the total is 1.01–1.06, inside.** The assertion fires on real data, the cells are marked, and the grid runs (ruling 2). The reading is the one §6.2.5–6.2.6 reached from control C1 and §6.4.4 from Shepard: the factor covariance is calibrated on the trading book and the diagonal specific-risk assumption under-forecasts the remainder's variance by 50–75% — on a book that carries 9% of its forecast variance there, which is why the total stays inside. Variant 6's VRA moves the specific `B` nearest to one (1.23–1.27) and the factor `B` nearest to one (0.98–0.99), and still fires. The named confound is in the same place as before: the four government zeros' remainder carries the carry-and-roll term the forecast's regressand does not, in every cell alike. The dense variants have no specific component (`n/a`) and reconcile on the total at 1.01–1.04.

**Ex post exceeds ex ante, as Hwang & Satchell say it must.** Ex-post over ex-ante volatility at the drifted weights is 1.02–1.06; the drift between rebalances accounts for **1.1–1.8% of realised variance** at a monthly horizon — the effect Lawton-Browne suggest can double annual TE is small here because the weights move for a month, not a year. **Brinson-Fachler against the equal-weight book (ruling 3), Carino-linked:** the reference cell's cumulative active return over EW is +69% over the 15.6 years, and it is **selection (+68%) and interaction (+41%) against a negative allocation (−39%)** — the RSTR book earned its excess inside sleeves and paid for overweighting sleeves that underperformed the whole benchmark, which is exactly the `−b` term of Brinson-Fachler doing what it is there for. The benchmark's return series is reported; its Sharpe is not computed.

## 10.4 Capacity

```
Cost drag:  TC(A) = τ·κ√A
Net alpha:  α_n(A) = α_g − τκ√A
Break-even: A_BE = (α_g/(τκ))²
Marginal:   A_eff = ((α_g − α_min)/((3/2)τκ))²   ⇒  A_eff = (4/9)·A_BE at α_min = 0
```

**Effective capacity under square-root impact is 4/9 ≈ 44% of break-even capacity.** Hard-code that relationship and mark both points on the curve.

Two refinements the literature insists on and most write-ups skip:

- **The right capacity experiment is a re-optimization, not a rescaling.** Re-run portfolio construction at each candidate AUM with the ADV constraints binding and let the optimizer choose *different* holdings. Simply scaling the small-AUM portfolio and applying an impact function overstates costs badly.
- **Capacity is quadratically sensitive to `κ`.** Halving the cost coefficient quadruples capacity. This is why AQR's estimates come out ~17× larger than Korajczyk-Sadka's. **Any capacity number you publish must be accompanied by its cost-model sensitivity.**

### 10.4.1 RULING (2026-09-03, W5-P2): §10.4 WAITS FOR W6 — the closed form is built, labelled as the rescaling, and publishes no number

**The ruling.** Every input this section's curve needs — `α_g`, `τ_A`, and the portfolio structure that fixes `κ` — comes from a trade series, and none exists until §8's optimizer produces one. Producing a capacity number now would require inventing `α_g`, which CLAUDE.md invariant 9 most explicitly forbids. So W5-P2 built the capacity module, its tests, and the `κ`-sensitivity machinery against synthetic inputs with hand-computed expected values, and **published no number**. The re-optimised curve with ADV constraints binding is the real deliverable and lands in W6-P2 or W6-P3.

**What was built** (`src/mafrm/costs/capacity.py`, `tests/test_capacity.py`). The closed form at the configured exponent `e` — `TC(A) = τκA^{e−1}`, `α_n(A) = α_g − TC(A)`, `A_BE = (α_g/(τκ))^{1/(e−1)}`, `A_eff = ((α_g − α_min)/(e·τκ))^{1/(e−1)}` — with the 4/9 relation hard-coded as this section instructs and the general `A_eff/A_BE = e^{−1/(e−1)}` pinned to it at `e = 3/2`; at the marginal point exactly one third of the gross alpha survives, `1 − √(4/9) = 1/3`. `κ` is **derived** from a fixed structure — `Σ_i Yσ_i|z_i|^e/V_i^{e−1}` over `Σ_i|z_i|` — and a test asserts that `R ×` §7.1's engine impact at `V_i/A` equals `τκA^{e−1}` to `1e−12` at every AUM on a grid, rather than trusting the algebra (`experiments.md` row 198). Capacity is quadratic in `κ`: `capacity_multiplier(m) = m^{−1/(e−1)}`, halving the coefficient quadruples capacity.

**The closed form is the RESCALING this section warns overstates cost badly, and it is labelled as such in the module docstring, the report and the figure title.** It holds the small-AUM structure fixed and scales the impact function. It is therefore the upper bound on cost and the lower bound on capacity that the W6 re-optimised curve is drawn beside.

**`α_min`** is parameterised (`costs.capacity.minimum_net_alpha`) and **only the `α_min = 0` special case is implemented**. A non-zero threshold is a statement about what return the strategy must clear and no source supplies one; the config parser and the module both refuse any other value until one is sourced.

**The AUM grid is a rule, not a pair of endpoints** (`costs.capacity.aum_grid`). Log-spaced between the AUM at which the largest position's trade is 0.1% of its proxy's ADV and the AUM at which it is 10%: `A = p·V/z_max`. Both ends derive from the ADV data once a portfolio exists and nothing is invented; the endpoints are computed in W6. The point count is display resolution — both marked points are closed-form and a test asserts they are read off no grid.

**The L1 term.** §7.1's spread cost `a_i|z_i|` is AUM-invariant and this section's impact-only form omits it. The module carries it exactly as `aum_invariant_drag`, subtracted from `α_g` wherever `α_g` appears — an algebraic identity, not a parameter — and a test pins `A_BE(α_g, d) = A_BE(α_g − d, 0)`. A gross alpha that does not cover the spread has capacity exactly zero.

**REGISTRATION for W6-P1 — `α_g` is swept, never picked.** `α_g` is the one number the whole cost side depends on that is published nowhere. It enters §1's risk term `(μ_g/σ_f)(1 − 1/B)` and the break-even condition `α_g = TC(A)`. **W6-P1 must not pick it.** Ruled here, in advance: sweep it and report the band — the project's standing idiom — with the **principle for the sweep range written down before W6-P1 opens**, not chosen inside it. A point value picked in the optimizer session, however reasonable it looks, would be the first invented parameter to reach the headline. `costs.capacity.gross_alpha` is `null` and the parser refuses any other value, so the attempt fails loudly. When the sweep's principle is ruled it gets its own key with the principle beside it.

**REGISTRATION for W6-P2/W6-P3 — the re-optimisation.** Re-run portfolio construction at each grid AUM with the ADV participation constraints binding and let the optimizer choose different holdings. **A test must assert the holdings actually differ across AUM levels.** Both `Y` regimes, both points marked on each, the band shown, in dollars, with the 30y-through-TLT and 2026-disclosure caveats of §7.1.1–7.1.2 carried beside every figure. The three §7.3 scaling laws are then checked empirically on the backtest's own cost output and charted; W5-P1 pinned them against the cost function because no backtest existed.

**`Y` stays two regimes.** The session's task text asked for a `Y` sweep spanning both regimes. §7.2.1 ruled the same day that `Y` is two calibrated regimes with **no interior points** — they measure different things, and a sweep between them manufactures a parameter from two numbers. That ruling stands: the capacity sensitivity band is evaluated **at** the two regimes, `(Y_patient/Y_urgent)^{1/(e−1)} = (0.58/1.40)^2 = 0.1716` — patient capacity is 5.83× urgent — and nothing between them is computed (`experiments.md` row 197). The machinery accepts any `κ`, so a sourced third regime slots in. Recorded here rather than silently resolved either way.

**What the report shows** (`reports/capacity.md`, `reports/capacity.png`). The curve in the units the closed form is scale-free in: AUM as a multiple of the patient break-even, net alpha over gross. In those units the patient curve is `1 − √x`, the urgent curve `1 − √(x/0.1716)`, the marked points `x = 1` and `x = 4/9` per regime, and the only project values that enter are the two `Y` regimes and the exponent. No dollar appears and a test asserts it. The report states that, with no execution data, any impact coefficient is borrowed rather than estimated, and that the band rather than a midpoint is what accompanies every capacity figure this project will publish.

### 10.4.2 FINDING (2026-09-04, W6-P3): the re-optimised curve — the rescaling overstates cost by 37–49% where the cap binds, and break-even is beyond the ruled grid in both regimes

`experiments.md` rows 251–252 (`data-diagnostic` under §9.3 ruling 4); `reports/capacity_reoptimised.md`, `reports/capacity_reoptimised.csv`, `reports/capacity_reoptimised.png`; `results/metrics.json` (`capacity_reoptimised`). The reference cell re-optimised at each of 50 AUMs from $406k to $40.6M (the ruled grid: 0.1% to 10% participation of the thinnest proxy's median ADV on the equal-weight trade) with the ADV cap HARD, both `Y` regimes, no Sharpe per point.

**The holdings differ (leg (a), HOLDS):** the largest per-rebalance L1 distance between the grid's ends is 1.27 (patient) and 1.41 (urgent) — the top-of-grid book is a different book, not a rescaled one, as §10.4.1 required the test to show. **The cap binds at the top (leg (d), HOLDS):** on 27.6% and 25.4% of rebalances at $40.6M, first appearing near $10M, and it empties the feasible set on two rebalances, where the ladder drops it and logs it.

**Where the cap binds the rescaling overstates cost badly, as this section says; where it does not the two coincide (leg (b): urgent HOLDS at every point, patient REFUTED at 6 of 18).** At the grid top the re-optimised recurring drag is 27.8 bp/yr against the rescaled 38.2 (patient) and 36.5 against 71.0 (urgent): the closed form overstates by 37% and 49% because the re-optimised book turns over less (4.39 against 6.50 a year at the top, 3.55 in the urgent regime) and holds thicker names. Below the AUM at which the cap first binds the re-optimised book IS the reference book, and the two drags differ by the NAV drift between rebalances (±2.6 bp/yr on 22–28): the six patient points "above" the rescaling are that noise, on the near side of the cap, and the leg as registered had no tolerance for it. **Recurring cost rises from 21 to 28 bp/yr across the grid in the patient regime and 25 to 37 in the urgent, while the gross return does not fall — it rises at the top (6.56% against 6.39%), the ADV cap acting as the turnover regulariser the `A_high` band found.** Dollar net alpha is monotone in AUM on the whole grid.

**Break-even is beyond the ruled grid in both regimes (leg (c), HOLDS), and the closed form puts it at $28.3B patient and $4.85B urgent** — 5.83× apart, the two-regime band of §7.2.1 — with `A_eff = 4/9 A_BE` at $12.6B and $2.16B. Those are the RESCALING's numbers and therefore the lower bound on capacity this section warned about; the re-optimised curve never approaches zero inside the grid because the measured gross alpha (6.2% a year, RSTR in sample: what the signal did, not what it will do) is twenty times the cost drag at the largest book the ADV data define. **The honest statement is that capacity in this section's sense is not reachable inside the range the disclosures calibrate the cost model for**, and that the number a reader wants — the AUM at which this strategy's alpha is gone — sits where the square-root law is extrapolated an order of magnitude past its 6% anchor, through five curve points traded as ETF proxies (the 30-year through TLT), on 2026 spread disclosures applied to the whole history (§7.1.1–7.1.2). Both regimes are shown, both marked points are drawn, and the band rather than a midpoint accompanies every figure.

---

# 11. Repository and reproducibility

```
├── README.md              # the paper: question, data, method, results, limitations, scope-out
├── pyproject.toml / uv.lock
├── Makefile               # data / features / model / backtest / report / verify / all
├── config/
│   ├── universe.yaml      # frozen, dated, rationale per line
│   └── model.yaml         # half-lives, shrinkage, cost params — no magic numbers in code
├── src/mafrm/
│   ├── data/              # loaders, cache, manifest, contracts
│   ├── factors/           # construction + AQR/French validation
│   ├── risk/              # covariance pipeline, specific risk, bias stats
│   ├── costs/             # EDGE, ADV, impact  ← the heavily-tested core
│   └── backtest/          # engine, optimizer, constraints, reporting
├── tests/
├── notebooks/             # exploration only, outputs stripped, never imported from
├── experiments.md         # every configuration evaluated, with rationale and result
└── reports/               # generated figures + tables, COMMITTED
```

**Environment:** `uv` with a committed `uv.lock` — the 2026 default, largely displaced Poetry. `requires-python = ">=3.11"`. Pin yfinance tightly; it is your highest-churn dependency.

**Determinism:** one `SEED` constant in config, passed explicitly to every stochastic call. `np.random.default_rng(seed)` per function. Never a global `np.random.seed()`.

**Data caching:** raw downloads to a gitignored `data/raw/` as parquet, one file per source per pull date. A **committed `data/manifest.json`** recording per file: source URL, download timestamp, SHA-256, row count, date range. Downloads idempotent, skipping on hash match. `make verify` re-hashes everything and fails loudly on drift. Never commit third-party raw data; always commit the manifest — that is what makes someone else's rebuild checkable against yours.

**Tests** — the non-negotiable categories:
- **Cost model unit tests** with hand-computed expected values (§7.4).
- **No-look-ahead tests**: computing a signal at `t` using data through `t` equals the value the backtest used at `t`. Plus the mechanical version — randomly perturb all data strictly after `t`, re-run, assert weights at `t` are bitwise unchanged, across ≥50 values of `t`.
- **Calendar/alignment tests**: monthly resampling produces the expected period count; no asset contributes a return on a date it did not trade.
- **Covariance property tests**: symmetric, PSD, correct shape, **after every adjustment stage**.
- **Golden-file regression**: a tiny committed synthetic fixture (3 assets, 5 years) producing known weights and metrics.

**AMENDMENT (2026-09-08, W8-P4b): what the golden file pins is a tolerance, not a hash, and the reason is measured.** The covariance fixture's panel is `draws @ chol(Sigma).T`. `default_rng`'s stream is bit-identical across platforms by numpy's guarantee, but the matmul and the Cholesky go through the wheel's BLAS, which may reassociate its sums. Measured rather than assumed: on **darwin-arm64** (Accelerate) the digest is one value at both `K`, unchanged at OMP thread counts 1, 2, 4 and 8; on **linux-x86_64** (GitHub Actions, OpenBLAS) it differs from macOS at both `K`, and **at `K` = 40 it is not stable run to run** — two digests across three runs on identical code and an identical `uv.lock`. Suspect: kernel dispatch across heterogeneous runner CPUs; `K` = 3 is too small to reach a vectorised path and is stable. Named, not confirmed — there is no Linux runner in scope to bisect on.

So `tests/fixtures/covariance_golden.json` records **both platforms' digests, with a `byte_identical` flag and the evidence for it**, and asserts byte-identity only where it has been observed to hold. What every platform is held to instead is `panel_summary` — absolute sum, sum of squares, min and max — at `CROSS_PLATFORM_RTOL` = 1e-9. That tolerance is chosen a priori, not fitted: it sits four orders above the ~1e-13 a 2×10⁵-element double-precision reduction can accumulate, and the tripwire it has to catch is seven orders the other way — **a one-step change to the seed moves these statistics by 2.5% at `K` = 3 and 11.3% at `K` = 40**, measured. Pinning the anchor to macOS and calling the repository reproducible would have been the worse answer: this section's whole value is that it says precisely what holds.
- **Data-contract tests**: expected columns, no all-NaN columns, recency assertions (e.g. "AQR data ends within 90 days of today" — this will save you from silently backtesting a frozen series).

**CI:** GitHub Actions, matrix on 3.11/3.12, `ruff` + `mypy` on the core module + `pytest` with coverage. **Two workflows:** a fast one on every push running against committed fixtures with network disabled, and a **scheduled weekly one** that hits live sources and fails if data contracts break. The second proves the pipeline still works six months after you stop touching it. That is cvxportfolio's trick, scaled down, and it is the strongest single reproducibility signal available.

**Small things that signal seriousness:** `CITATION.cff`; MIT or Apache-2.0 (not GPL); pre-commit hooks; `nbstripout`; `results/metrics.json` regenerated by CI so changes to headline numbers appear in the diff.

**Version control and publication.** `main` only, one commit per work session made after tests pass, commit messages naming the task. The `.gitignore` must exist before the first data pull — once raw parquet or an API key is in history, removing it means rewriting history on a repo you intend to publish. `data/raw/` and `data/processed/` are ignored; `data/manifest.json` is committed deliberately, and that asymmetry is the whole point of the caching design. Keep the repository **private through the build and flip it public only after an audit** of history for secrets, third-party data and licence contamination, plus a clean-clone `make verify && make test`. The commit history is part of the artefact — a log that reads as a coherent project timeline is itself evidence of how the work was done, which is why history is never rewritten. Separately, check your employer's outside-business-activity and publication policy before making a public repository under your own name; start that early, since approval processes take time.

**The README is the actual differentiator.** Structure it like a short paper: question, data (with a provenance table and every licence noted), method, results (figures committed so the repo is legible without running anything), and an honest **Limitations** section — survivorship bias in the frozen universe, estimated-rather-than-observed spreads, splice discontinuities, the FRED credit truncation, borrowed impact coefficients. Every experienced reader scrolls to Limitations first. A repo with one written well is instantly in the top decile.

## Positioning against what already exists

Say this explicitly in the README, because it demonstrates you surveyed the field:

- **cvxportfolio** — best cost model, no factor risk model, GPL.
- **toraniko** — a real cross-sectional equity risk model, but no optimizer and no costs; this project is the macro analogue, not a competitor.
- **skfolio** — excellent optimizer plumbing and covariance estimators, no macro multi-asset factor structure, no data layer.
- **OptimalPortfolios** (Sepp) — multi-asset and cost-aware, no factor risk model, no reproducible free-data pipeline.

**Nobody ships a free-data-reproducible multi-asset macro factor risk model with a time-varying empirically-calibrated cost layer and a published bias statistic.** Differentiate on four axes in this order: (1) time-varying empirically-estimated costs, (2) bias-statistic validation with confidence bands, (3) factor validation against AQR/French as a falsifiable table, (4) reproducibility as a feature.

Explicitly do **not** differentiate on the optimizer, risk measures, or plotting. Reimplementing a mean-variance solver reads as not knowing what is already solved; reimplementing the cost model reads as knowing exactly where the interesting part is. **Build what your thesis is about, import everything else, and write a paragraph explaining each choice.** That paragraph is what a hiring manager actually reads.

---

# 12. Week by week

**Eight weeks, not seven** — the equity cross-sectional module (§15) adds one. Seven is still achievable if you cut the long-history sensitivity, the credit reconstruction and the `bt` reconciliation up front; decide that in week 1 rather than discovering it in week 7.

| Week | Deliverable | Done when |
|---|---|---|
| **1** | **Data layer, complete.** GSW loader with Svensson evaluation and roll-down; TIPS curve; ETF loader with raw-and-adjusted caching and manifest; FRED with ALFRED vintages; AQR/French parsers; EDGE spreads on rolling windows; ADV. Calendar tests. `make verify`. | You never call a live API again. Synthetic 20y+ correlates >0.98 with TLT. |
| **2** | **Factors.** Macro factor set with sequential orthogonalization; PCA + MP denoising; rolling betas; per-asset R². **Validation against the published comparands the mapping of §6.5.1 actually supports — the falsifiable table.** (AMENDED 2026-08-30: the earlier wording named AQR TSMOM, which is a comparand for no factor in §4.1's set.) | `reports/factor_validation.md` exists and states, factor by factor, either the comparand it was regressed on or the reason no analogue exists; the three falsifiers of §6.5.2 are registered before the run and their outcomes are recorded whichever way they came out. |
| **3** | **Covariance pipeline — written generically.** EWMA with separated half-lives; Newey-West with PSD repair; eigenfactor adjustment (both `a` values); VRA. PSD assertion after every stage. Half-life sensitivity. **See the architectural instruction in §15.2 — this module must not know what asset class it is looking at.** | Eigenfactor bias curve reproduces the published 1.5→0.95 shape on your data. |
| **4** | **Specific risk + the validation battery.** Time-series/structural blend, Bayesian shrinkage. Bias statistics on all four portfolio families. MRAD. Shepard correction. MZ, Q, Ljung-Box, Kupiec, Christoffersen, Basel, Acerbi-Szekely. | The rolling bias-statistic chart with confidence bands exists. **This is the moment the project becomes credible.** |
| **5** | **Cost model.** EDGE integration, two regimes calibrated from published anchors, `Y` sweep, impact scaling laws, capacity curve. Hand-computed unit tests. `bt` reconciliation. | Cost tests pass with hand-computed values; the spread-vs-volatility chart exists. |
| **6** | **Optimizer + backtest.** cvxpy SPO, soft constraints, relaxation ladder, misalignment penalty. The full experiment grid. Perold decomposition, TE decomposition, attribution reconciliation assertion. | The grid runs end to end unattended and writes `results/metrics.json`. |
| **7** | **Equity cross-sectional module (§15).** Point-in-time-ish S&P 500 universe; six price-only Barra descriptors; Fama-French 49 industries; WLS cross-section with the cap-weighted sum-to-zero constraint. Then feed it through the **existing** weeks 3–4 pipeline unchanged. | Daily cross-sectional R² ≥ 39%, and the same bias-statistic chart regenerates for the equity model with no changes to `risk/`. |
| **8** | **The K/T scaling result, holdout, write-up, ship.** Plot measured optimizer-portfolio bias against K/T across both models and overlay Shepard's `[1−K/T]⁻²`. One evaluation on the held-out window. Deflated Sharpe with the honest count from `experiments.md`. README as a paper. CI green on both workflows. | The holdout has been run **once**, and the K/T curve exists. |

**If you are behind at week 5**, cut in this order: the long-history sensitivity, the credit reconstruction, Model B's detoned/OAS variants, the `bt` reconciliation. **Never cut the validation battery** — it is the project. **Never cut the equity module once week 3 is written generically** — at that point it is the cheapest week in the plan and it carries the headline result.

---


---

# 14. Failure modes, ranked

The parts most likely to break a build of this shape, in order:

1. **Calendar and alignment bugs.** The top source of "my Sharpe is 2.4" moments. Write the tests in week 1.
2. **Skipping the orthogonalizations.** Un-orthogonalized credit and equity factors are ~0.7 correlated and the covariance matrix goes near-singular in crises.
3. **Non-PSD matrices after Newey-West.** Guaranteed to happen. Assert and repair after every stage.
4. **Validating only on random portfolios**, which will look fine while the model is badly broken for its actual use.
5. **Look-ahead through adjusted closes.** Yahoo rewrites history backwards. Cache raw.
6. **Optimizer infeasibility** once turnover, box constraints and costs stack. Soft constraints plus a relaxation ladder, planned from the start.
7. **Individually-safe corrections interacting badly.** Each adjustment passes its own sanity check; applied in sequence they compound or cancel.
8. **Scope creep into return forecasting.** The moment you try to make the backtest look good, it is a six-month project.
9. **Chasing the last 5% of data history.** Cap it. The credit problem is the obvious trap.

**Models underforecast after calm periods** — August 2007 (tuned on the placid mid-2000s) and 2022 (tuned on the 2021 melt-up) are the canonical cases. If your bias statistic sits below 1 through a quiet stretch, that is not conservatism, it is a loaded spring. Say that in the README next to the chart.

**Correlations spike in crises** — diversification evaporates exactly when the model says it should not. Report conditional correlations in stress windows, not just unconditional ones.

---

# 15. The equity cross-sectional module (week 7)

## 15.1 Why this exists

Two reasons, and the second is the better one.

**Reason one — vocabulary.** Barra's USE4/CNE5 methodology is the common vocabulary of equity factor risk: it is what published risk-model documentation, vendor comparisons and buy-side risk postings are written in. This module is what lets "Barra USE4 methodology" be claimed and defended line by line rather than cited.

**Reason two — the result.** This is the part that actually matters. Shepard's second-order correction says the optimizer's volatility understatement scales as `[1 − K/T_eff]⁻²`. The macro model has K = 6; the equity model has K ≈ 56. Building both sweeps K/T across a **10× range on your own data**:

| Model | K | Vol half-life | T_eff | K/T | `[1−K/T]⁻²` | Understatement |
|---|---|---|---|---|---|---|
| Macro | 6 | 252d | 727 | 0.008 | 1.017 | 1.7% |
| Macro | 6 | 84d | 242 | 0.025 | 1.051 | 5.1% |
| Asset-level sample cov, N=15 | — | 84d | 242 | 0.062 | 1.136 | 13.6% |
| Equity | 56 | 504d | 1454 | 0.039 | 1.082 | 8.2% |
| Equity | 56 | 252d | 727 | 0.077 | 1.174 | 17.4% |
| Equity | 56 | 84d | 242 | 0.231 | 1.691 | **69.1%** |
| Asset-level sample cov, N=500 | — | 252d | 727 | 0.688 | 10.25 | ~925% (effectively singular) |

**Plot measured optimizer-portfolio bias against K/T and overlay the theoretical curve.** If your points track `[1−K/T]⁻²`, you have empirically reproduced a published theoretical result across two asset classes with two independently-built factor models. That is a categorically different artefact from "my bias statistic was 1.1."

It also answers, with a number, the two questions you will be asked:
- *"Why a factor model and not the sample covariance?"* — at N=500 and a 252-day effective window the sample covariance is singular. There is no version of this that works.
- *"How do you choose the half-life?"* — at K=56 the choice moves the second-order correction from 8% to 69%. In the macro model the same choice moves it from 1.7% to 5.1%. **Half-life selection is nearly irrelevant in a small model and decisive in a large one**, and almost nobody frames it that way.

### 15.1.1 RULINGS (2026-09-06, W8-P1): what "tracks" means, what the x-axis is, and what is swept — written before any point existed

Operator rulings, transcribed in `config/model.yaml` under `validation.kt_scaling` and registered at `experiments.md` rows 299–326 before the first point ran.

1. **"Tracks" is defined with the interval this project already uses for `B`.** A model's points track the curve if the curve's predicted `B` lies inside each point's own exact chi-square interval — the interval derived from the number of MONTHS that point's `B` pools over, in the unit `B` is measured in. `B` is a ratio of standard deviations, so the curve is `(1 − K/T)⁻¹` in that unit and `(1 − K/T)⁻²` in variance; both are overlaid, the test is in sd units. Stated per model, because the two clusters carry different error types and the predictions are opposite: **the equity points are registered to TRACK** (`B_specific` inside its interval, `B_factor` outside — the excess is estimation error and should scale with `K/T`), **the macro points are registered NOT to track** (family-4 `B` flat at 1.33 across nine `tau` while the curve moves — the excess is the diagonal's specification error and does not scale). The reversing result is one point at a time: a macro point whose interval contains the curve, or an equity point whose interval excludes it. *Implementation note:* "the point's own interval" is the set of true biases the measurement is consistent with, `[B/u_T, B/l_T]` from `(T−1)B²/B_true² ~ χ²_{T−1}` with `[l_T, u_T]` the null interval at `T` months — not the null interval itself, which would test the comparand against 1 rather than against the point.
2. **The x-axis is `K_d / T_eff`** with `T_eff` the realised Kish effective sample size of the volatility window averaged over the scored dates each `B` pools over; the asymptotic `2τ/ln 2` is an annotation and the per-date range is printed beside each point. The early builds sit at short `T`, and a pooled `B` does not pretend otherwise; for the macro model the two coincide after burn-in.
3. **The equity model is swept across the same nine-`tau` grid as the macro model.** Cheap, not multi-hour: the pre-eigenfactor `B` needs stages 1–3 only, no Monte Carlo, and the chart is pre-adjustment by design. The factor-volatility half-life alone moves; every other half-life at the short value (correlation 504, specific 84) — W6-P3's design. A 504d row is a grid point, not a new key. The same category as the macro sweep's rows (`data-diagnostic`), with the same enforcement.
4. **In this session:** the reference-book `tau` sweep (nine `strategy-config` rows, `N` 61 → 70, the W6-P3b registration) — with the honest note that the macro family-4 sweep being flat makes refutation the likely and informative outcome; row 164, C1 on the equity residual panel (the same instrument on both panels, `data-diagnostic`); Model B and the hybrid at the reference cell only IF their two-hour cache build can run unattended — **not run**, for a reason stronger than time (§15.1.2); the MP-denoising prediction at `N/T ≈ 0.11` **OUT** — it needs an equity statistical model that does not exist, recorded as untested.
5. **The second-order simulation at the `tau` grid** (6.5% at `tau = 21`, band [4.5%, 8.5%]) runs as registered — §6.4.5.
6. **The Shepard overlay's `T` is the formula's own** (task item iii): the correlation window for each factor-model point, because W3-P3b's withdrawn agreement and W4's 2.5× over-prediction were both window mismatches, and the curve is only a comparand where its `T` is the estimator's. Eq. 32 at the volatility window is carried beside every factor point as §6.4.3's UPPER BOUND. The naive asset-level comparand is a one-window estimator, so Eq. 13 at the volatility window is its comparand.

**Expected `N` before the first strategy-config row ran, stated back:** 61 + 9 = 70, plus 2 if Model B and the hybrid had run. They did not; `N` = 70.

#### The comparand conflict in these rulings, recorded under the operator's name (2026-09-06, W8-P1b)

**Ruling 6 and ruling 2 name different `T`s, and the conflict is the operator's rather than the session's.** Ruling 6 (the inheritance block) says the Shepard overlay must use *the formula's own `T` — the correlation window*, because W3-P3b's withdrawn agreement and W4's 2.5× over-prediction were both window mismatches. Ruling 2 puts the chart's x-axis on `K_d / T_eff` where `T_eff` is the realised Kish size of the **volatility** window. A curve drawn as a function of the x-axis quantity and a comparand evaluated at a different window are not the same object, and §15.1.2's two refutations at `tau` = 63 and 84 sit exactly on that difference: against the correlation-window comparand they are refutations, against the volatility-window one they would not be.

**W8-P1 tested against the registered comparand and recorded the volatility-window agreement as an observation. That was correct.** Switching comparands after seeing which verdicts each produced is selection on outcome, and it is the specific manipulation §6.6 and the category rules exist to detect. The session is not amended and neither verdict is revised.

**What the conflict actually reveals is that neither single-`T` curve is the right comparand for a two-window estimator**, which §6.4.5 had already measured from the other direction: the whole-covariance closed form over-predicts this estimator by 4.1× at `tau = 21` falling to 0.94× at 504, and its `rho` leg is `tau_sigma`-independent. A formula derived for one window has no `T` that is both of this estimator's windows, and choosing between them is choosing which half of the estimator to describe. §15.1.3 is the resolution: the estimator's own second-order theory, simulated, tested under the same interval rule — registered at `experiments.md` rows 327–333 before it was computed, `data-diagnostic`, `N` unchanged at 70.


### 15.1.2 FINDING (2026-09-06, W8-P1): the chart exists, the macro cluster does not track and the equity cluster mostly does, the equity model stops existing below `tau = 63`, and the walk along `tau` is not the walk along the curve

`reports/kt_scaling.md`, `.csv`, `.png`; `reports/kt_book_sweep.md`; `experiments.md` rows 299–326 and row 164. All numbers below are monthly `B` with the point's own exact interval at its month count, as ruled.

**The verdicts, per model.** Macro family 4 (pre-eigenfactor, `K = 6`): **1.642–1.647 at every one of nine half-lives**, `K/T_eff` 0.0055–0.0998, comparand 1.0056 outside every interval — **not-track HOLDS 9 of 9**. Equity family 4 (pre-eigenfactor, `K_d` 51–54): **1.214 → 1.026** across `tau` 63 → 504 (`K/T_eff` 0.29 → 0.043), comparand 1.045 at the correlation window — **track HOLDS 5 of 7**, REFUTED at `tau` 63 and 84 where the measured `B` exceeds what the comparand admits. `B_factor` sits above its interval (1.10–1.20) and `B_specific` inside or just below (0.90–1.00) at every answered point: the equity excess is the factor component throughout, as registered. **At `tau` 21 and 42 the equity model has no answer**: §5.2's repair fires on 38 and 16 of 210 month-end builds, the last on the final build, at realised `K_d/T_eff` 0.89 and 0.45, and no month-end survives §6.2.4's rule. That is this table's "effectively singular" row reached by a factor model, two grid points below the shipped half-life, and it is the strongest form of "decisive at `K = 56`": below `tau = 63` there is no model to select. The naive `N = 447` comparand is singular (`N/T_eff` > 1, Eq. 13 without a value) at `tau ≤ 126`, with the near-singular estimator's book realising 2.7–3.5× its forecast; at `tau = 504` (`N/T_eff` 0.36) it measures 1.77 against Eq. 13's 1.57.

**The reference book** (rows 299–307, `N` 61 → 70): `B` falls monotonically 1.132 → 0.999 across the nine points, the monotone leg holding at every consecutive pair; `tau = 84` reproduces the grid's 4D/patient cell (1.0611 against 1.061); `B_factor` runs 1.062 → 0.912 while `B_specific` stays 1.28–1.33 (§6.2.6's diagonal, a fourth time). Row 305's reversing result is obtained by the interval's width and refuted by the point estimates: `B_factor` 0.932 at `tau = 252` reproduces the long band's 0.932 to three decimals with the specific half-life held, so W6-P3's lower `B` was the factor-volatility window. Under ruling 1 the book cluster tracks at 8 of 9 — anticipated at registration as the interval's width, since a book whose `B` sits within 0.13 of one at 187 months cannot exclude a comparand of 1.006.

**What the chart shows that the rulings did not register, stated so it cannot be quoted as more than it is.** Against the volatility-window `K/T_eff` — the x-axis — the equity points and the reference-book points lie on or near the sd-unit curve `(1 − K/T)⁻¹` from `x` = 0.01 to 0.29. Three things say that is not a demonstration of Shepard's mechanism: rows 318–326 measured this estimator's Gaussian estimation-error bias at these windows at a third to a quarter of that curve (§6.4.5); the `K = 6` book moves 1.13 → 1.00 across the grid while Shepard at `K = 6` spans 1.11 → 1.006; and the book's `B_factor` falls *below* one at long half-lives, which no estimation-error story produces. The walk along `tau` within one model is the responsiveness channel §5.1.3 named before any sweep ran, and it is monotone in the window length exactly as the curve is. **§15.1.3 settles this, and against the observation**: the estimator's own second-order theory, simulated at every answered half-life, is flat at 1.037–1.043 and reproduces none of the equity model's 0.19 rise. The observation stands as recorded and is now explained rather than promoted.

**MWO's direction, and where the sign lands here.** Against the ruled comparand the two refuted equity points are under-predicted (1.045 against 1.18–1.21), MWO's sign; against the volatility-window form they are over-predicted (1.28 and 1.41), §6.4.2's sign. The truth for a two-window estimator sits between its windows, and the closed form has no `T` that is both.

**Row 164.** Leg (a) HOLDS: C1 removes −69% of the equity excess — it makes the book worse, because a residual correlation on 252 names from an EWMA at `T_eff = 242` is rank-deficient (smallest eigenvalue 2e-5) and restoring it puts estimation error into a `Delta` that is in regime. Leg (b) REFUTED on its boundary: §5.3 removes 44% of the excess, the largest share any stage has removed on any panel here, but the half-width is 63% of a 0.16 excess at 190 months. The registered falsifier is not met; the two error types move in opposite directions across the panels. The second falsifier does not fire: on 16 non-overlapping 12-month blocks the block `B` has a Spearman correlation with block `K/T_eff` of −0.03 (equity) and −0.70 (macro, `B` rising while `K/T_eff` falls to its asymptote). The macro panel's within-sample growth stays with W4-P2's residual-correlation suspect, logged under the stopping rule.

**Not run, and the reason is a limitation rather than a schedule.** **Model B and the hybrid have never been optimizer-consumable models**, and that is the phrase W8-P3's limitations section carries. Their committed histories hold bias series, not per-date `(X, F, Delta)`; building those needs an **asset-level specific-risk definition for Model B that §4.2 does not give and `config/model.yaml` does not hold** — is the specific variance the residual of a PCA reconstruction, or the discarded eigenvalues' constant average? Inventing one is CLAUDE.md invariant 9, so the gap is not two hours of compute but a specification that was never written. Every statement this project makes about Model B and the hybrid is therefore a statement about a **covariance estimator scored on bias statistics**, never about a portfolio: they appear in §4.2, §4.3 and §5's ladder and they appear nowhere in §8's optimizer, §9's grid or §1's identity. That is a real limit on the head-to-head §2 promised — "two factor models, compared head to head" is delivered on the risk-model term and not on the cost term — and it goes in the README in those words rather than as an absence a reader has to notice. MP-denoising at `N/T ≈ 0.11`: out by ruling; registered and untested. `N` = 70.

**Two practical consequences, in the form the measurements support.** (1) A 500-stock sample covariance at a 252-day effective window is singular — here `N = 447` at `N/T_eff` 0.61–2.5, no minimum-variance answer where the ratio exceeds one and a 2–3.5× miss where it does not; no `K/T` correction reaches a matrix that is not invertible. (2) Half-life selection moves the estimation-error term in proportion to `K/T`, moves the responsiveness term at any `K`, and moves the specification-error term not at all: irrelevant to the macro family 4 (0.005 across 24×), decisive for the equity model (0.19 across the answered grid, and no model at all below `tau = 63`), and — the part the table did not anticipate — material for a constrained `K = 6` book through responsiveness rather than through `K/T`.

### 15.1.3 FINDING (2026-09-06, W8-P1b): the estimator's own second-order theory is flat at 1.04, the equity rise is not sampling error, and the correlation window was the right comparand after all

`experiments.md` rows 327–333, registered before the simulation was written; `reports/kt_scaling.md` (the rows 327–333 section) and the third curve on `reports/kt_scaling.png`. `data-diagnostic`; `N` unchanged at 70.

**Why a third curve at all.** §15.1.1 records a conflict in the W8-P1 rulings under the operator's name: ruling 6 evaluates Shepard at the correlation window, ruling 2 draws the x-axis at the volatility window. Neither is obviously right, because **a formula derived for a covariance estimated on one window has no `T` that is both of a two-window estimator's windows**. The estimator's own second-order theory is not a formula at all — it is `mafrm.risk.second_order.simulate`, which draws Gaussian rows from the panel's own `F`, runs them through §5.1's actual estimator and measures `E[w'Fw / w'F̂w]` for the minimum-variance `w` of each draw. Its `.bias` is the square root of that ratio, which is **the same identity Shepard uses to turn a variance multiplier into a `B`**, so the third curve is available without inventing a mapping. One truth at the shipped 84d/504d half-lives, `K = 54`, `T` = 4,468 rows, `M = 2,000`, seed varied across grid points.

**The registered prediction is REFUTED, at exactly the two points that were already refuted.** The hypothesis was that the equity points track the simulated curve at all seven answered `tau`, including 63 and 84; they track it at five, missing at 63 and 84 in the same direction as the closed form did. The falsifier said what that means and it is taken: **the two refutations are a property of the measurement, not of which window a one-window formula is evaluated at.**

**The simulated curve is flat, and that is the finding.**

| | `tau` = 63 | 84 | 126 | 168 | 252 | 336 | 504 |
|---|---|---|---|---|---|---|---|
| simulated `B` | 1.0425 | 1.0406 | 1.0399 | 1.0392 | 1.0385 | 1.0379 | 1.0374 |
| MC s.e. | 0.0012 | 0.0010 | 0.0008 | 0.0007 | 0.0006 | 0.0005 | 0.0004 |
| `rho` leg, variance | 7.61% | 7.62% | 7.59% | 7.61% | 7.63% | 7.57% | 7.60% |
| `sigma` leg, variance | 1.01% | 0.62% | 0.51% | 0.36% | 0.22% | 0.14% | 0.02% |
| **measured `B`** | **1.2143** | **1.1838** | **1.1401** | **1.1118** | **1.0765** | **1.0541** | **1.0256** |
| measured `B_factor` | 1.1951 | 1.1825 | 1.1637 | 1.1502 | 1.1323 | 1.1202 | 1.1022 |

The simulated range is **0.005** against a measured range of **0.189** — thirty-eight times smaller, against Monte Carlo errors of 0.0004–0.0012 that cannot absorb any of it. The `rho` leg is 7.6% of variance at every point because the correlation half-life is pinned at 504 and cannot move; the `sigma` leg is 1.01% falling to 0.02%, because **at `K = 54` the split-window estimator's volatility-window leg is a Jensen term of order `1/T_eff` and not a `K/T_eff` term.** That is §6.4.5's macro measurement reappearing at nine times the `K`, and it is the reason Shepard's whole-covariance form over-predicts this estimator rather than describing it.

**So the equity model's rise from 1.03 to 1.21 as the window shortens is not sampling error in `F`.** Three accounts were available and none reaches it: the closed form at the correlation window (1.045, flat), the closed form at the volatility window (a different estimator), and the estimator's own simulated theory (1.037–1.043, flat). Against `B_factor` — the component the simulated comparand actually describes — the measured value is above the comparand at **all seven** points, not five. **What the rise is instead is not settled here and nothing is promoted to explain it:** the simulation is Gaussian on a fixed `F`, so it contains no fat tails, no volatility clustering, no specific-risk error and no specification error; §5.1.3's responsiveness channel is live at every point. Named, not chosen. **The measurement that would settle it, stated so a later session inherits a test rather than a hunch:** re-run this simulation with the rows bootstrapped from the panel's own standardised factor returns instead of drawn Gaussian — same estimator, same windows, same `M`, the only change being the innovation distribution. If the simulated curve then rises with `K/T_eff`, the equity excess is fat tails and clustering; if it stays flat, it is responsiveness or the specific leg, and a `lambda`-off variant separates those two. Not run here: it is a new estimator-level construction in a session whose scope is the chart.

**The conflict at §15.1.1 resolves in ruling 6's favour, which was not the expected direction.** The correlation-window closed form is 1.0451 at every equity point; the estimator's own simulated theory is 1.0374–1.0425. **They agree to within 0.008**, while the volatility-window form is 1.4117 at `tau = 63` — 0.37 away. Evaluating a one-window formula at the window the correlation is actually estimated on turns out to be a good approximation to the two-window estimator's own theory, because the `rho` leg dominates at large `K`. The ruling that looked like the arbitrary half of a conflict was the correct one, and the reason is now measured rather than argued.

**What this costs the headline, stated plainly.** §15.1 promised that if the measured points track `[1−K/T]⁻²` the project has "empirically reproduced a published theoretical result across two asset classes with two independently-built factor models". On the ruled comparand the macro cluster does not track by construction (its excess is specification error) and the equity cluster tracks at five of seven — but the tracking that holds is against a **flat** comparand over a range where the measured points move 0.19, so five inside is the interval's width and not the curve's shape. **The project has not reproduced Shepard's scaling on its own data, and the honest headline is the one the measurements support:** the governing ratio `K/T` determines *which stage of the pipeline earns its keep* (`reports/stage_k_dependence.md`), it determines *whether an estimator exists at all* (the naive comparand at `N/T > 1`, and this model below `tau = 63`), and it does **not** determine the realised calibration of either factor model on this data — the macro model's is specification error and the equity model's is something the Gaussian second-order theory does not contain. That is a weaker claim than §15.1 hoped for and a more defensible one, and it is what the README will say.

## 15.2 The architectural instruction — act on this in week 3, not week 7

**The `risk/` module must never know what asset class it is looking at.** Write it in week 3 against a generic interface:

```python
def build_covariance(
    factor_returns: pd.DataFrame,     # T × K
    config: RiskConfig,               # half-lives, NW lags, eigen a, VRA half-life, M
) -> np.ndarray: ...

def specific_risk(
    residuals: pd.DataFrame,          # T × N
    exposures: pd.DataFrame,          # N × K
    buckets: pd.Series,               # N, for Bayesian shrinkage
    config: RiskConfig,
) -> pd.Series: ...

def validate(
    forecasts: pd.DataFrame, realised: pd.DataFrame, portfolios: PortfolioSet
) -> ValidationReport: ...
```

If week 3 is written this way, the entire equity module is a new `factors/equity_cross_sectional.py` producing `(X, f, u)` and nothing in `risk/`, `costs/` or `backtest/` changes. Week 7 becomes the cheapest week in the plan. If week 3 is written with macro assumptions baked in, week 7 becomes three weeks and you will cut it.

**Test that enforces it:** run the full covariance pipeline on the committed synthetic golden fixture with K=3 and K=40 and assert both produce PSD matrices of the right shape. That test failing is your early warning.

## 15.3 Universe

The honest free-data route, in order of preference:

1. **Point-in-time-ish S&P 500 membership.** Wikipedia maintains both the current constituent list and a *"Selected changes to the list of S&P 500 components"* table with dates, which reconstructs approximate historical membership back to roughly 2000. It is not perfect — the change table is incomplete in places — but it is a genuine attempt at survivorship control and is far better than using today's list. **Snapshot it once, commit the reconstructed membership matrix, and state its provenance and known gaps in the README.**
2. Failing that, current constituents with survivorship bias **stated explicitly and prominently**, not buried.

Do not claim a selection procedure you did not run. Delisted tickers silently vanish from Yahoo and Stooq; free sources cannot fix this. Practitioners respect a clearly-labelled limitation far more than a hand-wave.

Universe screen, mirroring USE4's estimation-universe logic: exclude names with fewer than 252 days of history, and require a minimum dollar ADV. Log how many names the screen drops per date.

### 15.3.1 RULINGS (2026-09-05, W7-P1): the liquidity screen is inherited, history means bars, the matrix is a licensed daily table, no second source, and the loader's status contract

Seven rulings. The first four were issued once in a session lost to a network drop and re-issued with three more; all seven are recorded here before any code ran on the cache, and transcribed into `config/model.yaml` under `equity_universe`.

1. **The minimum dollar-ADV screen has no published value and is not invented.** The key is in the config as `null` with the reason beside it. The liquidity screen is **inherited from the index constructor**: S&P's own published eligibility rule — a float-adjusted liquidity ratio of at least 0.75 plus a minimum monthly share volume, tested at addition — is applied by S&P at every addition, so membership already carries a liquidity screen with a source. This project has no float data to re-apply it and no published USE4 threshold to substitute, and this week builds a risk model and runs no trades, so there is no cost-model anchor from which a threshold could be derived the way the ETF ADV cap was (§8.5.1). **The only in-sample screen applied is the one §15.3 gives a number for: 252 days of history.** The drop-count chart is that screen's. The ADV screen is implemented behind the null so that a sourced figure slots in without touching code; while null it drops nothing and the report says so.
2. **"252 days of history" means 252 valid daily bars for the ticker in the trailing window, not 252 days since index entry.** A name has trading history before addition, and USE4's screen is on history, not tenure.
3. **The membership matrix is committed, in a new tracked `data/reference/`.** It is a **licensed derived table**, not third-party raw data in invariant 8's sense: that invariant's list (AQR, French, ICE, JST) is data with no redistribution licence, whereas Wikipedia's text is CC BY-SA 4.0, which permits redistribution with attribution and share-alike. The file header carries the source URLs, the snapshot timestamp, the CC BY-SA 4.0 attribution, and a statement that the file itself is CC BY-SA 4.0, separately from the repository's code licence. `data/manifest.json` carries its SHA-256. `data/raw/` and `data/processed/` stay ignored; the Yahoo bars behind the matrix are raw and are not committed.
4. **Calendar.** Membership is reconstructed **as far back as the changes table reaches** — §15.3's "roughly 2000" is the reach, not a cutoff — and known gaps are counted **per year**. The equity estimation sample starts at the existing `sample.start` (2007-04-01), so both models' bias statistics and W8-P1's `K/T` points sit on one calendar: one existing key, no new one. **The matrix is DAILY** — the changes table gives dates, and the pipeline reads month ends off a daily matrix, which invents no calendar.
5. **No second data source in this session.** A member whose bars are absent is marked **member, no data** — a third state, distinct from *not a member* — and counted per date as its own series on the drop chart. Filling from Stooq would mix two providers' adjustment conventions (failure mode 5) through a loader that does not exist; dropping silently would reintroduce survivorship bias exactly where the matrix is meant to control it. **That count is the survivorship control's measured coverage; the report states the residual bias as that number.** If coverage of delisted names is poor, that is the finding, and a second loader is a W7-P1b decided on the number.
6. **The loader records a per-ticker status in the manifest** — `ok`, `empty` (the provider answered and has no history), or `failed` (exception, timeout, HTTP error) — **retries `failed` once, and only `empty` becomes member, no data.** A ticker still `failed` after the retry is listed by name in the report, never folded into the coverage count. The lost attempt found that a delisted name and a failed request were indistinguishable in what it built; this is the contract that closes it, and it is a contract, not a parameter.
7. **The stash and the orphaned cache.** `stash@{0}` holds the lost attempt (W6-P3's earlier WIP is `stash@{1}`); both are kept, neither popped, both read as reference only, and nothing enters the tree without its tests. The 2026-09-05 parquet files under `data/raw/` are re-pulled — not hand-registered; the loader writes the manifest as it always has.

**Two facts about the source, found by the lost attempt at its orientation and recorded here because §15.3 was written against a page that no longer has this shape.** (a) Wikipedia has split the *"Selected changes to the list of S&P 500 components"* table out of the constituents article into its own article, *Historical components of the S&P 500*; the constituents article now carries the current list and its `Date added` column only, so the loader pulls **two** pages and both URLs are in the config. (b) The changes table had 407 rows in the 2026-09-04 snapshot, of which **19 are before 2007** — two in 1976, one in 1994 and seven in 2000 among them — against 16–30 a year from 2011 onwards.

**What the implementing session had seen before it registered anything, stated so the reader can discount it.** The lost attempt's stash was read in full as reference (ruling 7), including its rendered `reports/universe_screen.md`: a month-end construction with 878 spells over 858 tickers, member counts 502–511 over the sample, 876 tickers requested and 659 with any bars, 355 departed names of which 152 had bars, 186 of 511 members without bars at 2007-04-30 falling to 12 of 503 at 2024-12-31, and a peak of 11 members with insufficient history at 2020-02-28. Its four registered rows and their verdicts are transcribed into `experiments.md` as rows 261–264 with that provenance; this session's rows 265–268 are registered against the daily construction and ruling 6's status split, and each says which of those numbers informed it.

**Constructions taken by the implementing session, none a parameter.** The reconstruction walks the changes table **backwards** from the current list: before each row's effective date, its added tickers were not members and its removed tickers were. An added ticker that is not in the running set, or a removed ticker that already is, is an **inconsistency** — the signature of a missing later row or of a ticker rename — and is counted per year rather than resolved. One deterministic reconciliation is applied before the walk and its count is reported: a current member whose `Date added` has no addition row under its own ticker is linked to the addition row on that exact date if, and only if, exactly one such member and exactly one such row share the date (the FB → META shape); anything ambiguous stays a gap. Wikipedia's `BRK.B` is Yahoo's `BRK-B`; the dot-to-dash map is a vendor symbol convention held in code and logged under *constants held in code*. The daily matrix's calendar is the **union of sessions any cached bar exists on** — the calendar `mafrm.data.calendar` aligns on — restricted to `[reach, snapshot]`; the interval table is committed beside it as the canonical form and a test asserts the two agree. **The third state is per session:** a member is *member, no data* on a session if the cache holds no valid bar for it that day, a valid bar being one with close and volume both present and positive — the same rule the estimation screen uses, so the committed state and the screen's `member_no_data` agree by construction and the screen asserts it. For ruling 6, "the provider answered" is tested directly: yfinance reports a swallowed transport error and a delisted symbol with the same *no timezone found* error, so on that error (or an empty frame) the loader asks Yahoo's chart endpoint itself and counts the ticker `empty` only if Yahoo returns its explicit *no data found* error payload; anything else is `failed` and retried once. The probe goes through the browser-impersonating HTTP client yfinance itself uses (`curl_cffi`), one request at a time: the first pull probed from eight threads with a plain `urllib` client carrying a browser user-agent string, and Yahoo answered *429 Too Many Requests* for 112 of the 120 candidates — an honest `failed` under the contract, and a defect in the probe's transport rather than in the source, diagnosed and fixed before the second pull (the first pull's counts are kept at `experiments.md` row 265). Eight further tickers failed inside yfinance's own parsing under pandas 3 — it assigns an integer into a dividend column pandas has inferred as its string dtype when every value is missing — and the loader restores object inference for the duration of the vendor's parsing; the price columns are numeric either way and the action columns are cast to float before caching.


### 15.3.2 FINDING (2026-09-05, W7-P1): the coverage number, and what the status contract caught

On the 2026-09-05 snapshot the walk yields 878 spells over 858 tickers, a member count of 502–511 on every session of the sample, and a daily matrix of 12,649 sessions. Of 876 tickers requested, 684 have Yahoo history, 192 are `empty` under ruling 6's contract and **none is `failed`** after the retry. **The residual survivorship bias, as ruling 5 asks for it:** 173 of 511 members at the sample's first session have no history, falling to 12 of 503 at 2024-12-31; 179 of 355 departed names (50.4%) have none. The estimation universe runs 330–488 names over 2007–2024. The lost attempt's absence set was 25 names larger than the snapshot's — eight lost to a vendor parsing bug and seventeen that Yahoo serves on one pull and not on another (a re-pull eighteen minutes after the snapshot returned *no price data found* for 17 names it had just served in full) — which is the third thing, beside a delisting and a failed request, that ruling 6's split turned out to separate: the source's own non-determinism on delisted names. The coverage number is a property of the snapshot, one more reason the matrix is one snapshot. A second price source is a W7-P1b decided on the 173, not here.

**W7-P1b (2026-09-05), decided: no second loader.** Stooq — the only free candidate — is dead, and no free source carries delisted names; a W7-P1b with no viable source is a placeholder. The survivorship bias is stated with its number and its direction: 173 of 511 members at the sample start have no bars, and the missing names are overwhelmingly departures (179 of 355, against 12 of 503 current), so the early-sample universe over-represents survivors and the specific-risk, RESVOL and SIZE cross-sections are biased toward survivors' values, declining through the sample. For this project's question it is second-order — the equity module tests whether a diagonal `Δ` is in regime and how `B` scales with `K/T`, neither of which is a return study — but it goes in the README and in every equity report as a number, not a caveat. **Consequence for W8-P1, registered now: the equity `K/T` points use the actual per-date estimation-universe size, 330–488, not 500.** Three housekeeping rulings from the same wrap-up: Stooq is retired from the refresh with a manifest note (nothing reads it; the refresh target's non-zero-on-any-failure contract stays); the count identity `requested = ok + empty + failed` is asserted in the loader, not printed; and the committed matrix is **one snapshot**, kept by `make data` and rewritten only by a deliberate `--resnapshot`.

## 15.4 Descriptors — the six that need no paid data

The clean split from the research: **Beta, Momentum, Size, Non-Linear Size, Residual Volatility and Liquidity derive entirely from daily price, volume and shares outstanding.** Book-to-Price, Earnings Yield, Growth, Leverage and Dividend Yield need point-in-time fundamentals, and the analyst-estimate descriptors (EPIBS, EGIBS) — which carry the *largest* published weights in their composites, 0.68 of Earnings Yield — have no free substitute at all. Build the six. Say plainly in the README which five you omitted and why.

Exact parameters, from the CNE5 Descriptor Details document (the only public MSCI source with both formulas *and* weights):

| Factor | Construction | Window |
|---|---|---|
| **BETA** | `r_t − r_ft = α + β·R_t + e_t`, R = cap-weighted market excess return of the estimation universe | 252 days, exponential weights, **half-life 63d** |
| **MOMENTUM** (RSTR) | `Σ_t w_t[ln(1+r_t) − ln(1+r_ft)]` | **T = 504 days, lagged L = 21 days** (sum runs t = 21…525, skipping the most recent month), **half-life 126d** |
| **SIZE** | `LNCAP = ln(total market cap)` | — |
| **NON-LINEAR SIZE** | Cube of the *standardized* Size exposure, regressed on Size with the cross-sectional regression's √cap weights (USE4: "regression-weighted"; this table first said cap-weighted — superseded by §15.4.5 ruling 2), residual taken, then winsorized and re-standardized | — |
| **RESIDUAL VOLATILITY** | `0.74·DASTD + 0.16·CMRA + 0.10·HSIGMA` | DASTD: EW sd of daily excess returns, 252d, **HL 42d**. CMRA: `Z_max − Z_min` over `Z(T) = Σ_{τ≤T}[ln(1+r_τ) − ln(1+r_fτ)]`, T = 1…12 months. HSIGMA: sd of BETA-regression residuals, same 252d/63d window |
| **LIQUIDITY** | `0.35·STOM + 0.35·STOQ + 0.30·STOA` where `STOM = ln(Σ_{t=1..21} V_t/S_t)`, `STOQ = ln((1/3)Σ exp(STOM_τ))`, `STOA = ln((1/12)Σ exp(STOM_τ))` | 21d / 3m / 12m |

**The two orthogonalizations are not optional.** Non-Linear Size must be orthogonalized against Size or it is ~collinear with it. Residual Volatility must be orthogonalized against Beta (and in USE4, Beta *and* Size) and re-standardized, or it is simply a beta proxy. Skipping these is one of the top three ways a home-built Barra replica goes wrong, and the resulting factor correlation matrix will tell you — check it.

### 15.4.1 RULINGS (2026-09-05, W7-P2a): market cap needs a point-in-time share count, and the source is EDGAR

Five operator rulings were issued for W7-P2 before any descriptor code was written. The first restructures the session and is implemented here; the other four govern the descriptors and are recorded now so that W7-P2 proper reads them rather than re-deciding them.

**Ruling 1 — shares outstanding.** Market cap needs a point-in-time share count, and the source is SEC EDGAR's XBRL cover-page fact `dei:EntityCommonStockSharesOutstanding`, through a new loader in `loaders.py`. It is the only free point-in-time source, and it covers delisted filers because a CIK outlives a ticker. It is pulled through the **frames** endpoint — one request per calendar quarter, every filer, keyed by CIK — never per ticker. The User-Agent EDGAR requires comes from the environment (`equity_shares.user_agent_env`, default name `SEC_EDGAR_USER_AGENT`), never a literal. **Point-in-time rule:** a share count is known from its FILING date and forward-filled to the next filing; the count is put on the same basis as the cached close using the cache's own actions table, so shares and price carry no look-ahead. The **pre-XBRL gap** — `sample.start` (2007-04) to a name's first filing — is back-filled with the earliest filed count, split-adjusted, **labelled an approximation with its direction** (buybacks overstate early caps, issuance understates them) and its share of the sample stated in `reports/equity_market_cap.md`. yfinance's `sharesOutstanding` is **rejected**: a current count times a historical price is look-ahead in a descriptor. **CIK mapping:** SEC's `company_tickers.json` for current names, the changes table's company names for departed ones; a name that does not map drops from the estimation universe with a per-date count — the W7-P1 discipline. The loader, its manifest entries, the cap panel and its tests are their own commit, **W7-P2a**, before any descriptor is built; if it consumes the session, the descriptors are W7-P2 proper next session and the record says so rather than compressing them.

**Two orientation findings that changed how ruling 1 is implemented, recorded before the pull.**

(a) **EDGAR's frames records carry no filing date.** Each record has an accession number, CIK, entity name, an as-of date (`end`) and a value — nothing else (probed at CY2009Q2I, CY2012Q4I, CY2019Q1I and CY2024Q4I; every record has exactly those keys). The point-in-time rule therefore cannot be implemented from the frames alone. The loader pulls EDGAR's **quarterly XBRL index** (`full-index/{year}/QTR{q}/xbrl.idx`: CIK, company name, form type, DATE FILED, accession-bearing filename) beside every frame — one request per quarter for every filer, the same shape the ruling asked for — and the join is by accession number. A fact whose accession is in no cached index has no known-from date and is **dropped and counted**, never guessed. The index tables are indexed by filing date and the frames by as-of date, so `cache.read`'s holdout truncation applies to both, and a count filed on or after the boundary can be in force on no in-sample session.

(b) **The cached close is split-adjusted backwards.** `mafrm.data.prices.check_splits_applied` (W1-P4) asserts that Yahoo's `auto_adjust=False` close is back-adjusted for splits and only dividends are left raw. "Shares and price on the same basis" is therefore implemented as: a filed count is multiplied by **every split ratio in the actions table with an ex-date after the count's as-of date**, then by the cached close. That product equals the actual count times the actual price on every session — including the back-filled sessions before the first filing, where the actual count is the first filed count with the intervening splits undone and the actual price is the cached close with the same splits redone. The as-of date, not the filing date, anchors the split adjustment, because the cover-page count is stated as of that date. Because the adjustment basis is set by the **pull** date, 32 splits after the holdout boundary (in the 2026-09-05 pull) rescale every cached close before it, so the actions table is read in full for its `Stock Splits` column only — a sanctioned source-question crossing pinned in `tests/test_holdout_guard.py`, beside the undated ticker map. The share counts and filing dates themselves go through `cache.read`.

**Three further facts about the source.** Multi-class filers report the fact **per class with a dimension** the frames do not carry, so the non-dimensional total is absent for Alphabet (CIK 1652044), Berkshire Hathaway (1067983) and News Corp (1564708) in every probed frame; such a name is `no_facts` and drops with a count. Fox Corp's CY2019Q1I record carries the value **1** — a placeholder — which the panel uses as filed and the report lists under *unexplained jumps* (a display threshold, `_JUMP_RATIO = 2`, held in code) for the operator to rule on; nothing is filtered. The frames begin at CY2008Q4I with a handful of voluntary filers; the mandate begins with fiscal periods ending after 2009-06-15 for the largest filers and phases in through 2011, so the pre-XBRL approximation runs longer than "mid-2009" for most of the universe, which is why row 270 registers a band rather than the 13.1% arithmetic alone.

**Constructions taken by the implementing session, none a parameter.** A current-list ticker maps by ticker through SEC's map (a current list's tickers are current; Yahoo's `BRK-B` is SEC's `BRK-B`); a departed name maps by **exact normalised company name only**, because tickers are recycled and the company holding a departed ticker today is not the one the index dropped. Normalisation: upper-case, parentheticals dropped, EDGAR's trailing `/DE/` dropped, `&` read as `AND`, punctuation to spaces, and a modest list of legal-form tokens removed (`INC`, `CORP`, `CO`, `LTD`, `PLC`, ...; `HOLDINGS` and `GROUP` deliberately kept). A name matching more than one filer is `ambiguous`, one matching none is `unmapped`; neither is resolved by hand and both are listed. The status contract is SPEC.md 15.3.1 ruling 6's: `ok`, `empty` (HTTP 404 — the server answered that it has no such quarter) or `failed` (anything else), one retry, the identity asserted, a still-`failed` quarter listed by name and never read as "no filers". Within one filing date the latest as-of date is kept. A filing on a non-session day is in force from the first session on or after it.

**The documents on hand.** Neither the CNE5 Descriptor Details nor the USE4 methodology document is in the repository (`find` for either name returns nothing), so for rulings 2 and 5 below the "check the document first" clause could not be executed here: the winsorization bound keeps its READING mark, and RESVOL follows USE4 as the project's primary reference.

**Rulings 2–5, for W7-P2 proper, recorded now.** (2) **Winsorization** at ±3 equal-weighted cross-sectional standard deviations around the equal-weighted mean of the raw descriptor, one pass, marked READING in the config with its source named as the Barra convention; the sensitivity at 2.5 and 3.5 is a `data-diagnostic` on the descriptor correlation matrix and VIFs only — no factor returns, no Sharpe — and the bound is chosen by convention, not by that measurement; the reversing result is named now: a correlation or VIF that moves materially across the three bounds makes the bound a `model-config` decision and it is logged as one. (3) **Exposures are DAILY**, and so will W7-P3's regression be, so the equity factor returns go through `risk/` unchanged with the same half-lives in days (invariant 10). (4) **CMRA's month is a 21-session block** (12 × 21 = 252 sessions), consistent with the daily calendar, `data.trading_days_per_month`, and RSTR's 21-day lag. (5) **RESVOL is orthogonalized against Beta AND Size** — USE4's construction; if CNE5 orthogonalizes against Beta only, the difference is recorded and USE4 followed.

### 15.4.2 FINDING (2026-09-05, W7-P2a): coverage, the join verified, and the source's unit-scale errors

On the 2026-09-05 pull: 72 calendar quarters (2008Q4–2026Q3) of frames and index, all `ok`, none `empty`, none `failed`; 302,043 frame records, 1,225,394 index rows, 10,412 ticker-map rows. **Mapping (row 269, holds):** 503 of 503 current tickers by ticker; 275 of 355 departed names (77.5%) by exact normalised name, 21 ambiguous, 59 unmapped. With usable counts: 465 current, 265 departed. **The point-in-time join (row 272, holds on all three legs):** Apple's count in force on 2024-12-31 is the 10-K's, filed 2024-11-01, 15,115,823,000 to the share, and the January-2025 10-Q's is invisible; on the 4-for-1 ex-date the count in force is 4 × the July 10-Q's, filed 2020-07-31, and it is identical on the session before, so the cap moves only with the close. **The pre-XBRL approximation (row 270, holds):** 13.38% of (session, name) pairs with a cap are back-filled — every name on 2007-04-02, 61 names at 2009-12-31, 12 at 2010-12-31, none at 2024-12-31. **Coverage of the estimation universe (row 271, refuted on both legs):** the universe runs 326–488 names and 307–460 of them have a cap; on 2007-04-02 only 19 lack one, because the departed-before-XBRL names the registration reasoned about are mostly the names Yahoo has no bars for, so the survivorship gap of §15.3.2 removed them before this module could; on 2024-12-31, 28 lack one, of which 21 are dual-class filers reporting the fact per class with a dimension the frames do not carry, and one (XOM) is a **successor CIK**: SEC's current map points to the 2025 re-incorporated registrant, under which no pre-boundary XBRL exists, and the predecessor is not consulted.

**The finding that governs what happens next (row 273, not pre-registered).** The cover-page fact is mis-scaled in a material minority of filings: 96 of the 136 consecutive-count jumps among universe names are at or beyond 100× (or below 1/100) — a count tagged in thousands or millions in one filing and in shares in the next — over 49 names; 40 more, over 36 names, are splits the actions table lacks, class restructurings, or shells (FOX/FOXA's only non-dimensional fact is `1`, on the pre-spin registration; SCANA's last is `100`, after Dominion acquired it). In-force counts more than 10× off a name's own median cover 0.94% of the universe's (session, name) cells over 39 names, and because the earliest filed count is back-filled, a first filing carrying the error carries it over 2007–2009. **The raw total cap of the universe reads 35,162 USD trn at 2007-12-31 against 10.74 trn excluding those cells**, and 46.50 against 46.24 at 2024-12-31 — the excluded series is of the same order as the index's published total throughout. **No filter is applied and the panel is committed as ruling 1 specifies.** The descriptors must not read `cap` until three rulings are issued at the start of W7-P2 proper, each with a number the operator chooses: (i) the **within-name consistency screen** — drop a filed count more than `R×` off the name's own median split-adjusted count (10 and 100 both separate the unit-scale errors from the corporate actions on this pull; `R` is a parameter and is not chosen here), with the dropped count logged and the previous count staying in force; (ii) a **predecessor-CIK route** for current names whose mapped CIK has no facts — a second pass by name, or a hand table with sources — so that a re-incorporation does not cost a name its history; (iii) the **dual-class gap** — accept `no_facts` and the per-date count (the present state), or find a per-class source, of which no free one is known. Until (i) is ruled, `MarketCapPanel.cap` is a loader output, not a model input.
### 15.4.3 RULINGS (2026-09-05, W7-P2): the consistency screen, successor CIKs, dual-class filers — and what the screen's own test found

Three operator rulings, issued at the start of W7-P2 before any descriptor read `cap`, on the numbers §15.4.2 left open; two confirmations on the descriptors; then the record of what the rulings met on the cache.

**Ruling (i) — the within-name consistency screen, `R = 100`.** A filed count more than 100× the name's own median split-adjusted count, or below 1/100 of it, is dropped; the previous count stays in force; every drop is logged (`MarketCapPanel.dropped`, and by name in `reports/equity_market_cap.md`); the pre-XBRL back-fill then reads the earliest SURVIVING count. `R` is not tuned: the ruling placed it at the geometric midpoint of a band, (10, 1000), that the 2026-09-05 pull was read as leaving empty between the largest corporate action the actions table lacks (10×) and the smallest unit-scale error (1000×), and asked for that emptiness to be asserted in a dataset-marked test — any jump inside the band on a re-pull fails the test and RE-OPENS `R`. Shells (FOX's `1`, SCANA's `100`) fall to the screen; splits the actions table lacks (2–10×) do not and are counted by name and cell-share as a known residual. The universe total cap is reported with and without the screen beside a published total. Until the screen is applied and its test passes on the real cache, `MarketCapPanel.cap` is a loader output (`is_model_input` false) and no descriptor reads it. Config: `equity_shares.consistency_screen.{ratio, empty_band}`; code: `edgar.consistency_screen`, applied in `market_cap.assemble`.

**Ruling (ii) — successor CIKs from a hand table.** `config/cik_overrides.yaml`, in the `spread_levels.yaml` pattern: per re-incorporated registrant, the predecessor CIK, the EDGAR URL each figure was read from and the date read. The predecessor's cover-page facts are read beside the mapped CIK's; the loader refuses a row whose `current_cik` is not what SEC's map gives on the pull being built, or whose predecessor has no fact in the cached frames. One sourced row now: XOM (2115436 ← 34088, both EDGAR company pages read 2026-09-05). **No automated second pass by name** — that would re-introduce the ambiguity the exact-name mapping was built to avoid.

**Ruling (iii) — dual-class filers stay `no_facts`.** No free per-class source exists. The report lists them, and the cap-weighted market return states, per date, how many universe names it is missing and their share of universe DOLLAR VOLUME — a proxy, because the quantity missing is the cap itself; the exact cap share is what a per-class XBRL parse (a possible W7-P2c) would measure first. That proxy, with row 278's SPY correlation, decides whether W7-P2c happens; it is not prejudged here.

**Confirmations (d), (e).** The ±3 sd winsorization bound goes into the config marked READING with its source named as the Barra convention (`equity_descriptors.winsorization.bound_sd`), exactly as ruling 2 of §15.4.1 specifies — one key, not a new decision. NLSIZE's post-orthogonalization winsorization reuses the same key: the same convention at a different stage, and a second bound would be a second unpublished number. The 2.5/3.5 sensitivity covers both uses in one run.

**What the screen met on the 2026-09-05 pull (rows 274–277).** The screen at `R = 100` dropped **74 filings over 49 universe names** (121 over every mapped name), every one either a count tagged in thousands or millions (ratios to the name's median 488–1.2×10⁶ and 5×10⁻⁵–1.3×10⁻³) or a shell (`1`, `100`, `173`). The universe's year-end total cap reads **11.36 USD trn screened against 35,162 unscreened at 2007-12-31**, and sits below the published total US equity market value (`config/published_market_totals.yaml`; Siblis Research, a SUPERSET of the index — no free citable history of the index's own total was found) on all 18 year ends, where the unscreened total exceeds it on 7 (row 274 holds on both legs). XOM's predecessor route delivers status `ok`, 62 in-sample facts and the 10-Q filed 2024-11-04 in force on 2024-12-31 at **4,395,094,536 shares, the cover page's figure to the share** — but row 276 as REGISTERED is refuted, because the registration typed 4,395,095,000 from a rounded scientific-notation print of the cached value; the config keeps the registered figure and records the cover-page one beside it. The 27 universe names without a cap on 2024-12-31 (22 of them `no_facts`, dual-class filers) carry **8.06% of universe dollar volume** (4.5% at 2010, 8.2% at 2015, 5.4% at 2020; row 277 holds).

**The band was NOT empty as first stated, and the premise was mis-stated twice (row 275 REFUTED; `R` confirmed in W7-P2b).** Among universe names the unscreened consecutive-count ratios inside (10, 1000) or its reciprocal number **21 over 17 names**: nineteen are unit-scale errors whose two filings straddle a buyback or issuance, so the ratio lands at 972–999 or 1/999–1/972 rather than exactly 1000; one is AIG's 2011 recapitalisation (135M → 1.80B shares, 13.3×, the Treasury's conversion); one is BRK-B's 2010 filing pair (52.8M → 976K after the cache's 50:1 split adjustment, 1/54). **Two errors, the operator's first.** §15.4.2 said "10 and 100 both separate the unit-scale errors from the corporate actions on this pull" — a statement about two candidate values of `R` — and the operator turned it into "the pull found no jump between 10× and 1000×", which nobody had measured. The implementing session then built the screen and its test on that statement without probing the ratio distribution at orientation, when one groupby would have shown it false; that is the second error and it is recorded as one. **What was measured is the right quantity.** On the median-relative ratios the screen actually acts on (count / the name's median, folded to ≥ 1, every usable filing among universe names), the empty band is **(54.7, 489)** — BRK-B's Class-A fact below, REG's count in thousands above — and `R = 100` sits inside it with a factor-of-two margin either side. The 972–999× jumps are the ×1000 family with drift and the screen catches every one on the median-relative ratio; AIG's 13.3× is kept, correctly, as a corporate action. **Ruling 1 of W7-P2b: `R = 100` stands, and the test is rewritten** to assert what is true and checkable — (a) `R` lies strictly inside the empty band of median-relative ratios recomputed on every pull, with the band's ends printed, and the band still contains the recorded (55, 488); (b) no count the screen drops has a ratio the actions table explains, so no corporate action is ever dropped. A separation test on the populations as observed with `R` fixed, not tuning; a future pull that puts a filing between 55 and 488 fails it and re-opens `R` with numbers. `consistency_screen.empty_band` now holds the measured band, and the original (10, 1000) is kept in the report for the record.

**BRK-B: a dual-class filer wearing a per-total fact (ruling 3 of W7-P2b).** Berkshire's only non-dimensional cover-page facts (four filings, 2009-11 to 2011-05, 941K–1.06M shares) are its **Class A count**, and the frames carry nothing for it after 2011 — so the panel would forward-fill a Class-A count against the Class-B price for thirteen years and BRK-B would enter SIZE and the market weight as a ~0.4 bn name. It is neither a unit-scale error (54.7× off its own median, under `R`) nor `no_facts` by the source's own record; it is the dual-class gap with a per-total-looking number. **Ruling: a `treat_as: no_facts` row in `config/cik_overrides.yaml`, sourced from EDGAR's company page (CIK 1067983, read 2026-09-05).** BRK-B joins Alphabet, Meta and News Corp in the `no_facts` group; its facts are counted and never used; **the 1,500:1 A/B conversion is not fabricated** to keep the name. The 2024-12-31 no-cap count is 28 of 488 (23 `no_facts`), and their dollar-volume share 8.76%.

**The residual the screen leaves.** After the screen, 40 consecutive-count jumps over 36 universe names remain (2–13×): splits the actions table lacks, class restructurings, the AIG recapitalisation. Cells whose in-force screened count is 2× or more off the name's median — an upper bound on the cells such a jump mis-states — are **37,545 of 1,707,927 (2.20%) over 48 names**. A known residual, not screened: the screen is for unit-scale errors, and a 2:1 the vendor lacks is a different defect with a different fix.

### 15.4.4 FINDING (2026-09-05, W7-P2 and W7-P2b): the six descriptors on the panel, the orthogonalization weights, and two refutations that are properties of the construction

Built in `mafrm.factors.equity` under rulings 2–5 of §15.4.1, the confirmations of §15.4.3 and the five rulings of §15.4.5, at `R = 100`. Every window, half-life, lag, block and weight from `config/model.yaml`; exposures daily, 2007-04-02 to 2024-12-31, on the estimation universe intersected with the names holding a screened cap (307–460 names; MOMENTUM's 525-session window costs a further 2–6). The rolling stage runs from 525 sessions before the sample so every window is complete on the first sample session; on that burn-in the estimation universe is the first sample session's, frozen, and used only for the market return. Report: `reports/equity_descriptor_correlations.md`.

**The asymmetry holds and does what it is for.** Cap-weighted centring, equal-weighted scaling: the cap-weighted portfolio's exposure to every factor is zero to 10⁻¹⁴ over 4,469 sessions (unchanged by the weighting ruling below, as row 282(c) required), every factor's equal-weighted cross-sectional sd is one, and a unit test pins both on a synthetic panel. The market return BETA is regressed on — the cap-weighted excess return of the universe with the previous session's caps — correlates **0.9907** with SPY's excess return daily (0.9751–0.9989 by calendar year, 17.3 bp/day tracking sd; row 278 holds): neither the survivorship gap nor the 4–9% of universe dollar volume without a cap (row 277, a proxy, because the cap is what those names lack) reaches the proxy.

**The orthogonalizations are not optional here either, and they are weighted the way USE4 words them.** Before: the standardized cube of SIZE correlates **0.732** with SIZE and the RESVOL composite **0.690** with BETA (equal-weighted time averages of daily cross-sectional correlations; row 279(a) holds). W7-P2 first ran both orthogonalizations cap-weighted, as §15.4's table and the task paste said; that zeroed the cap-weighted moment and left NLSIZE/SIZE at +0.601 equal-weighted and **+0.286 √cap-weighted** — the metric §15.6's regression uses, so the paste's own reason, "or it is collinear with Size", was not served. **Ruling 2 of W7-P2b (§15.4.5): both orthogonalizations use the regression weights, √cap — USE4's "orthogonalized … on a regression-weighted basis" — decided by reference to USE4's text, not by the outcome; had USE4 said cap-weighted, cap-weighted would stand.** Under it (row 282): NLSIZE/SIZE **−0.016 √cap-weighted** (the trace is the winsorization after the regression, §15.5's exception), +0.453 equal-weighted, −0.415 cap-weighted; RESVOL/BETA −0.001 and RESVOL/SIZE +0.001 √cap-weighted, zero to floating precision before the re-standardization's affine map. Row 279(b) as registered — equal-weighted after-correlations below 0.2 — stays refuted on NLSIZE/SIZE at 0.453: a regression-weighted orthogonalization is not an equal-weighted one, and the report shows all three metrics. Cap-weighted CENTRING is a different operation with a different purpose and is unchanged.

**VIFs.** √cap-weighted, the regression's metric: BETA 1.35, MOMENTUM 1.04, SIZE 1.52, NLSIZE 1.04, RESVOL 1.64, **LIQUIDITY 2.50**; equal-weighted 1.25 / 1.08 / 1.72 / 1.34 / 1.64 / 2.58 (row 281 refuted at < 2, accepted as reported by ruling 5). LIQUIDITY correlates −0.46 with SIZE, +0.50 with RESVOL and +0.40 with BETA in the regression's metric — turnover is high where names are small, volatile and high-beta — a collinearity no published orthogonalization addresses; W7-P3's regression tolerates it and none is added.

**The winsorization bound is a convention the structure does not feel** (row 280 holds): at 2.5 and 3.5 sd no off-diagonal correlation moves by more than 0.027 and no VIF by more than 0.024. It stays a READING and is not reclassified `model-config`.

**Coverage.** 11 names the committed matrix marks MEMBER have no valid bar on this pull over 8,135 (session, name) cells — BMC, CBE, CFC, GR, HAR, MEE, MOLX, RSH, RX, TIE, TLAB — listed, not dropped; such a cell has no return and no exposure.

### 15.4.5 RULINGS (2026-09-05, W7-P2b): `R` confirmed and its test rewritten, regression-weighted orthogonalizations, BRK-B to the hand table, the cover-page figure in config, four findings accepted as reported

Five operator rulings on W7-P2's findings; none adds a trial.

1. **`R = 100` stands; the band test is rewritten** as a separation test on the median-relative ratios the screen acts on, with `R` fixed (§15.4.3). Both errors in the original premise are recorded there, the operator's first.
2. **NLSIZE is orthogonalized against SIZE, and RESVOL against BETA and SIZE, with the REGRESSION weights √cap** — USE4's "regression-weighted" — superseding the task paste's "cap-weighted" and §15.4's table. The paste's own reason is only served in the regression's metric. §15.5's cap-weighted centring stays, as does the order orthogonalize → winsorize → re-standardize. Decided by reference, not by outcome; row 282, `data-diagnostic`, with the reversing result named.
3. **BRK-B: a `treat_as: no_facts` row** in `config/cik_overrides.yaml`, sourced. It is a dual-class filer and joins that group; no conversion ratio is fabricated.
4. **Row 276: the config carries the cover-page figure**, 4,395,094,536, not the registered one. The config is the record of the source; the row is the record of the registration and its transcription error. "Print and check before writing" applies to transcribed values as much as to computed ones.
5. **Accepted as reported, no action:** LIQUIDITY's VIF (2.50 √cap-weighted) against the registered < 2 — no published orthogonalization exists and W7-P3's regression tolerates it; the no-cap names' 8.76% is of dollar volume, labelled the proxy it is, because a cap share cannot be computed for names without a cap; the ±3 sd bound moves no correlation past 0.03, so it stays a convention; the Siblis total-US series is an upper bound, labelled.

## 15.5 Standardization — the asymmetry most replicas get wrong

USE4 §2, Eq. 2.4:

```
d_nl = (Raw_d_nl − μ_l) / σ_l
```

- **`μ_l` is the CAP-WEIGHTED mean.** So a cap-weighted portfolio — the market — has ≈ zero exposure to every style factor, style factors carry no market beta, and the Country factor absorbs the market.
- **`σ_l` is the EQUAL-WEIGHTED standard deviation.** MSCI's stated rationale: equal weights *"prevent large-cap stocks from having an undue influence on the overall scale."*

**Cap-weighted centering, equal-weighted scaling. Not the same weights for both.** Winsorize at the raw-data level *before* standardization (NLSIZE is the exception — it is winsorized *after* orthogonalization). Composite factors are built as weighted sums of standardized descriptors, then re-standardized at the factor level.

## 15.6 Industries and the cross-sectional regression

**Industries:** GICS is proprietary. Use the **Fama-French 49** definitions mapped from SIC codes — free, well-documented, academically accepted, and Ken French publishes the mapping. Note honestly that GICS is arguably the single largest contributor to Barra's explanatory power, so this substitution costs you more than it looks like it should.

**The regression** (USE4 Eq. 3.1):
```
r_n = f_c + Σ_i X_ni f_i + Σ_s X_ns f_s + u_n
```

Two things to get exactly right, both of which are the difference between a replica and a toy:

**1. Weighted least squares with `w_n ∝ √(market cap_n)`,** normalized to sum to 1. MSCI's stated assumption, verbatim: *"the variance of specific returns is inversely proportional to the square root of total market capitalization."* Not cap-weighted, not equal-weighted. This is a heteroskedasticity correction and it materially changes the factor returns.

**2. The cap-weighted industry sum-to-zero constraint** (Eq. 3.3):
```
Σ_i w_i f_i = 0        w_i = cap weight of industry i in the estimation universe
```
Industry exposures sum to 1 for every stock and the country exposure is 1 for every stock, so the industry block is exactly collinear with the country factor. This constraint resolves it — and it is *cap*-weighted, not equal-weighted, and not "industries sum to zero" naively.

What it buys you is the whole reason Barra output is readable: **the Country factor return becomes the cap-weighted market return, and industry returns become pure industry effects relative to market.** "Your Energy exposure cost you X above and beyond the market" is a sentence you can only say if this constraint holds.

Implement via a K×(K−1) constraint matrix `R` with `f = R·f̃`, giving `f̃ = (R'X'WXR)⁻¹R'X'W r`. Or drop one industry and reparameterize — equivalent, uglier, easier to debug.

### 15.6.1 RULINGS (2026-09-05, W7-P3): Siccodes49 from Ken French, the current SIC from EDGAR's submissions endpoint checked against the first in-sample 10-K, no thin-industry rule, and t−1 timing

Four operator rulings issued before any code was written for this section, transcribed into `config/model.yaml` (`data.ken_french_siccodes49`, `equity_industries`, `equity_regression`); then the constructions the implementing session took, and one fact about the construction established by derivation before the run.

**Ruling 1 — the industry definitions are Ken French's own `Siccodes49` file.** Added to the data block with its URL, pulled through `loaders.py`, recorded in the manifest, never committed — the treatment the daily-factor entry already has. A SIC code that falls in none of the 48 listed industries' ranges goes to industry 49, "Other"; that is French's own rule, not this project's. (The file also lists a few ranges explicitly under 49; a code in those, or in none, is 49 either way.)

**Ruling 2 — the per-name SIC comes from EDGAR's submissions endpoint, one request per CIK, delisted names included** (a submissions record outlives a ticker). It serves the CURRENT SIC, and applied backwards that is an approximation, so it is not left as a claim: the SEC-HEADER of each name's FIRST in-sample 10-K carries the SIC assigned at that date. One header per name is pulled and the fraction of names whose SIC differs between first filing and now is reported, listed by name. If the disagreement is small, current-applied-backwards stands with the list beside it; if large, that is the finding, and a per-filing header parse is a W7-P3b decided on the number. **A name with a CIK but no SIC is excluded from the regression with a per-date count — a missing code is not an "Other" code.** Names with no CIK have no cap and are already out.

**Ruling 3 — thin-industry rule: explicitly NONE.** No minimum membership, no shrinkage constant. A zero-member industry on a date is absent from that date's regression — its factor return is NaN, not zero, and the constraint matrix is built on the industries present. A one-member industry is estimated as-is. The report states, per date, the count of industries with ≤ 2 members and the count of names in them. The consequence is named for W7-P4 rather than patched here: a name alone in its industry has a residual of ≈ 0 by construction, so its specific variance is understated. §5.5's Bayesian shrinkage toward the size-decile prior (`q = 0.1`) is the published mechanism for exactly this and is already in the pipeline; **W7-P4 measures whether it is enough — registered: the singleton names' shrunk specific risk sits inside the cross-sectional distribution, not at its floor — and adds no thin-industry constant unless it is not.** FF49 on a 307–460-name universe is the spec's choice, and its thinness is a fact the report states, not a reason to change the scheme.

**Ruling 4 — timing.** Exposures and industry weights at the close of `t−1` against returns over `t`. The constraint's `w_i` are industry cap shares at `t−1`, the same cap the market return uses (the previous session's screened cap).

**Inherited from W7-P2, restated so this section does not re-decide it.** Both orthogonalizations (NLSIZE on SIZE; RESVOL on BETA and SIZE) were done at the regression weights, √cap, so the design matrix is orthogonal in this regression's metric and in no other: nothing is re-orthogonalized here, and equal-weighted correlations are not expected to be zero (NLSIZE/SIZE is 0.45 equal-weighted by construction). LIQUIDITY's VIF is 2.50 with no published remedy; the design matrix's condition number is printed per date and its range reported — that is the number that says whether the collinearity matters. The regression universe is the estimation universe minus the no-cap names (dual-class filers and BRK-B), whose dollar-volume share (8.76% at 2024-12-31) is stated beside the factor returns. 2.2% of universe cells carry splits the actions table lacks and every pre-first-filing count is a labelled approximation — both stated, neither fixed here.

**Constructions taken by the implementing session, none a parameter.**

- *Which CIKs are pulled.* Every CIK the W7-P2a mapping assigns to any ticker the membership tables name (764 distinct on the 2026-09-05 pull), plus the predecessor CIK of every `cik_overrides.yaml` row — not only the 539 estimation-universe names with a cap. The loader does not depend on the universe screen, so the provenance of the SIC table does not change when a screen parameter does; the report states the universe subset.
- *The first in-sample 10-K.* Per ticker, across the mapped CIK and its predecessor (XOM's 2008 10-K was filed under CIK 34088, and the successor CIK's submissions record begins in 2026): the earliest filing of form exactly `10-K` (`equity_industries.first_filing_forms`; amendments are not the original and are excluded) with a filing date on or after `sample.start`. The submissions endpoint pages its history — `filings.recent` holds the newest 1,000 and `filings.files` names older pages with their date spans — so every older page whose span reaches `sample.start` is pulled before the earliest is chosen. A ticker with no such filing (delisted before its first in-sample 10-K, or a filer of another form) is `no_10k`, counted and listed; it keeps its current SIC and is not excluded on that ground.
- *The header.* `https://www.sec.gov/Archives/edgar/data/{cik}/{accession without dashes}/{accession}.hdr.sgml`, the SGML header that exists for every filing (probed on XOM's 10-K of 2008-02-28, accession 0001193125-08-041781: `<ASSIGNED-SIC>2911` inside the `<FILER>` block whose `<CIK>` is 34088; the `-index-headers.html` route served a 337-byte stub and is not used). The SIC is read from the `<FILER>` block whose CIK is the one the filing was searched under, because a filing can carry several filers. A header without an `ASSIGNED-SIC` is `no_sic_in_header`, counted.
- *Two tables, both undated.* `raw/edgar/sic_current` (one row per CIK: SIC, its description, the entity name, status) and `raw/edgar/sic_first_10k` (one row per ticker: CIK searched, accession, form, filing date, period, the header's SIC, status). Like SEC's ticker map (§15.4.1) they are identifier tables with no date axis and are read through `cache.read_unrestricted` with the reason stated; `tests/test_holdout_guard.py` lists the module. The status contract is §15.3.1 ruling 6's: `ok` / `empty` (HTTP 404, the server answered) / `failed`, one retry of `failed`, every item with exactly one status.
- *The industry exposure* of a name is its current SIC's FF49 industry, constant through the sample — ruling 2's approximation, measured by the drift table. A name whose current SIC is missing has no industry exposure on any date and is excluded with the count ruling 2 asks for.
- *The constraint matrix.* On each date, with `I` industries present among the regression set and `w_i` their cap shares at `t−1` (summing to one over the set), `R` is the identity on the country and style columns and, on the industry block, maps the `I−1` free industry returns to `I` by `f_e = −Σ_{i≠e} (w_i / w_e) f̃_i`, where `e` is the industry with the LARGEST cap share that date. The choice of `e` does not change `f` (any `e` with `w_e > 0` gives the same constrained solution) and is made for numerical stability — the divisor is the largest weight — and so that a singleton industry is never the eliminated one, which keeps ruling 3's "residual ≈ 0" literal.
- *The weights.* `w_n = √cap_n / Σ √cap` over the regression set. The regression set on date `t` is: in the estimation universe at `t−1`, with a screened cap at `t−1`, with all six style exposures at `t−1`, with a current SIC, and with an excess return over `t`. Names failing each test are counted per date.
- *Solving.* `f̃ = (R'X'WXR)⁻¹ R'X'W r` by a solve on the normal matrix (it is `K̃ × K̃`, `K̃ ≤ 56`), then `f = R f̃` and `u = r − X f`. The residual variance is `σ̂² = Σ_n w_n u_n² / (n − K̃)` (the WLS estimator with weights summing to one), `Var(f̃) = σ̂² (R'X'WXR)⁻¹`, `Var(f) = R Var(f̃) R'`, and each date's t-statistic is `f_k / √Var(f)_kk`. The daily cross-sectional R² is the WEIGHTED one, `1 − Σ w u² / Σ w (r − r̄_w)²`, the regression's own metric, reported beside the equal-weighted one; §15.7's 39% is scored on the weighted figure and the other is printed so the choice is visible.
- *Condition numbers.* The 2-norm condition number of `W^{1/2} X R` (the matrix the solve inverts) per date, and of the style block `W^{1/2} X_S` alone. The full one is driven by the thinnest industry present — a singleton's column has norm `√w_n` — and the style one is the number LIQUIDITY's VIF is about; both are reported because they answer different questions.
- *`|t| > 2` frequencies* are counted over DAILY cross-sectional t-statistics, one regression per date, so consecutive readings share no observations and failure mode 9 does not apply to the count; the time-series t-statistic of each factor's mean return is printed beside it as information and decides nothing.

**A fact about the construction, established by derivation before the run and stated so the test is written against it rather than discovered by it.** Let `c` be the cap-weight vector of the regression set at `t−1` (`Σ c_n = 1`). Summing the fitted model with weights `c`:

```
c'r = f_c + Σ_i w_i f_i + Σ_s (c'x_s) f_s + c'u
```

The constraint makes the second term zero on every date. The third is the cap-weighted style exposure over the REGRESSION set, which is zero to 1e-14 over the standardization set (§15.5) and small but not zero over its subset. The fourth, `c'u`, is the market portfolio's specific return, and it is exactly zero **only when the regression weights are the cap weights**: WLS residuals are orthogonal to the columns of `XR` in the `W` metric, so `c'u = 0` for every `r` needs `c ∈ W·col(XR)`, which holds for `W = diag(cap)` (then `c = W·1` and `1` is the country column) and fails for `W = diag(√cap)`, where the residuals satisfy `Σ √cap_n u_n = 0` instead. So under §15.6's √cap weights **the country factor return equals the cap-weighted market return of the regression set minus the market's specific return, exactly**, and the equality the constraint is said to buy is exact up to `c'u`. The test therefore has two legs: (a) `|Σ_i w_i f_i| < 1e-10` on every date — the constraint binds; (b) the identity above closes to 1e-10 on every date — the decomposition is complete; and the gap `c'r − f_c` is MEASURED (its daily standard deviation relative to the market return's, the correlation of `f_c` with `c'r`, and with the estimation universe's cap-weighted return that BETA is regressed on) and registered as row 286, not asserted away. USE4's text describes the country factor as the cap-weighted market under the constraint; the arithmetic above is what that statement is exact about.


### 15.6.2 FINDING (2026-09-05, W7-P3): the factor set is built, the constraint's identity is exact to 1e-17, R² lands at 38.7%, and five registrations are refuted without a defect

Built in `mafrm.factors.equity_regression` under the four rulings of §15.6.1, on the W7-P2 exposures at `t−1` against excess returns over `t`, 2007-04-03 to 2024-12-31 (4,468 dates; 305–458 names per date, mean 379). Report: `reports/equity_regression.md`; the daily factor set: `reports/equity_factor_returns.csv`; the industry table and drift list: `reports/equity_industries.md`. experiments.md rows 283–289.

**Industries (rulings 1–2).** Every one of the 539 estimation-universe names with a screened cap has a current SIC through EDGAR's submissions record (765 CIKs pulled, all `ok`), and four fall in 49 "Other" (AES, AMCR, RSG, WM). 43–45 of the 49 industries are present on a date and three (Books, FabPr, Coal) never; **13 industries have at most two members on the median date** (range 9–15, falling as the universe grows), holding 18 names between them. The drift check: among 538 names with both a current SIC and a first in-sample 10-K header SIC, **41 (7.6%) changed SIC code and 27 (5.02%) changed FF49 industry** — six REIT conversions (AMT, CCI, EQIX, IRM, SBAC, WY → 6798), three casino operators (Fun → Meals), the rest single re-codings; real changes, listed by name. Row 284(b) registered < 5% and is refuted by 0.02 pp. Ruling 2's decision — whether 27 of 538 is "small", or W7-P3b's per-filing header parse is owed — is the operator's, on that number.

**The two constraint tests (rulings 3–4, and §15.6.1's derivation).** On every date the cap-weighted sum of the present industries' returns is zero to **3.5e-18** and the identity `c'r = f_c + Σ w_i f_i + Σ (c'x_s) f_s + c'u` closes to **2.8e-17** (row 285). The country factor correlates **0.9998** with the cap-weighted return of the regression set and with the estimation universe's `market_excess`; the gap `c'r − f_c`, which under √cap weights is the market's specific return plus a residual style term, has a daily sd of **2.3 bp against the market's 127 bp** (ratio 0.018; the style term contributes 0.3 bp of it). Row 286 registered a ratio near 0.15–0.20 from an independent-draws argument about the cap weights' effective name count and was wrong by an order of magnitude in the direction that makes the identity more useful: the √cap fit already leans on the large names, so the cap-weighted residual is not a sum of independent draws. Recorded as a prediction that missed, not as a pass.

**Explanatory power (§15.7).** Mean weighted daily cross-sectional R² **0.3868** (equal-weighted 0.345; median weighted 0.370; by calendar year 0.316 in 2013 to 0.463 in 2022, with 2008–09 and 2020–22 above 0.42). Row 287 registered ≥ 0.39 and is refuted by 0.3 pp — not materially. The diagnosis the falsifier asked for, a NOT PRE-REGISTERED decomposition on the same regression set and weights: **country + industries alone 0.300, country + styles alone 0.173, together 0.387.** The industry block carries the model, and it is the block §15.6 warns about: FF49 is a scheme built for the whole CRSP universe, assigns one SIC-based industry per name, and is thin here. Connor's 39.0% is for statistical models and 42.6% for fundamental models with a proprietary industry scheme; 38.7% with six styles and FF49 is where a free-data build lands. Nothing is changed on the strength of it; row 287 is the session's one `model-config` row (`N` = 58, row 68's precedent, written at registration).

**Conditioning and t-statistics.** The full design matrix `W^{1/2} X R` has a median condition number of 47 (range 40–80; row 288(c) holds); the style block alone 3.8 median and **6.09 at its worst (2009-04-07)** against a registered 5 — LIQUIDITY's collinearity is somewhat worse in the regression's metric than its VIF of 2.50 suggested and peaks in dislocations (row 288(b) refuted; benign). `|t| > 2` frequencies over daily cross-sectional t-statistics: BETA 57%, MOMENTUM 47%, SIZE 34%, RESVOL 26%, LIQUIDITY 24%, **NLSIZE 11%** (COUNTRY 84%); row 289 registered > 15% for every style and NLSIZE refutes it. NLSIZE is the pruning candidate §15.7 asks for — twice the 5% noise rate, with a time-series t of −2.2 on a mean of −0.5 bp/day, a small persistent premium rather than a per-date signal — flagged to the operator and not pruned here.

**Left for W7-P4, as ruling 3 named it.** A name alone in its industry has a residual of exactly zero (18 names on the mean date sit in industries of one or two), so its specific variance is understated; §5.5's Bayesian shrinkage toward the size-decile prior is the published mechanism, and W7-P4's registered measurement is that the singleton names' shrunk specific risk sits inside the cross-sectional distribution rather than at its floor. No thin-industry constant is added unless it does not.

### 15.6.3 RULINGS (2026-09-05, W7-P3b): point-in-time SIC for the 27 drifted names, NLSIZE flagged and kept

Two operator rulings on §15.6.2's findings, issued before any further code ran; then the constructions.

**Ruling 1 — a targeted W7-P3b, on cost rather than on the number.** 5.02% against a 5% bar is at the bar, and at the bar the number does not decide — the cost does. For the 27 names whose FF49 industry differs between the first in-sample 10-K and the current record, every in-sample 10-K SGML header is pulled through the loader already built (about 460 requests), and their SIC is point-in-time: **known from its filing date and forward-filled to the next filing**; before the first in-sample filing it is back-filled with that filing's SIC, labelled, the same convention as the pre-XBRL share count. The 511 unchanged names keep the current code. **Named as the residual the two-point check cannot see:** a change-and-revert between two 10-Ks of an unchanged name. Six REIT conversions mis-classified for up to seven years each is a known look-ahead in the industry loadings, removable for one loader call per filing; that is worth doing before the factor returns go into `risk/`. The regression outputs are regenerated, rows 284–289 re-scored on the corrected panel, and the numbers that moved are stated (row 290, registered before the pull).

**Ruling 2 — NLSIZE is not pruned.** The registration said "flag", not "prune". NLSIZE is a published USE4 factor and this project replicates the published model; a t-frequency on a 400-name survivor universe is not the evidence MSCI selected on; pruning would be a `model-config` decision taken on outcome; and `K = 56` is the design the `K/T` result needs. Reported, flagged, kept.

**Confirmed.** `N = 58` after W7-P3. Row 287 is a `model-config` row: its reversing result is named (a regression at these weights and this factor set reaching 39%) and the decision it informs is the report's statement, not a change. The post-hoc decomposition stays labelled post hoc.

**Constructions taken by the implementing session, none a parameter.** The drifted set is COMPUTED from the two cached SIC tables (`industry_differs` in the drift table), not hand-listed, so a re-pull that changes the set changes the targets. The history table (`edgar/sic_history`) is indexed by FILING DATE and read through `cache.read`, so a 10-K filed on or after the holdout boundary can never set an in-sample industry. A filing is in force from the first session on or after its filing date (a filing on `t−1` is known at the close of `t−1`). The industry exposure is now a sessions × names table, `equity_industries.point_in_time = "drifted_names_by_filing_date"`, and the regression reads it at `t−1` like every other exposure; for the 511 unchanged names the table is the current industry on every session, so nothing else moves. The report counts the regression cells the rule re-classifies and re-scores rows 284–289 beside W7-P3's figures.

**For W7-P4, restated from ruling 3 of §15.6.1 so the next session inherits it as a registration and not a note.** 18 names on the mean date sit alone or paired in their industry, so their residuals are ≈ 0 and their specific variance is understated; §5.5's shrinkage toward the size-decile prior is the published mechanism, and W7-P4 measures whether it is enough — registered: the singleton names' shrunk specific risk sits inside the cross-sectional distribution, not at its floor.

### 15.6.4 FINDING (2026-09-05, W7-P3b): the look-ahead reached 16 names and 1.74% of cells, and nothing moved by more than a rounding

Under §15.6.3's rulings. The per-filing pull flagged 38 names over every mapped ticker (the 27 universe names of §15.6.2 plus 11 departed or no-cap names) and delivered 655 headers, all `ok`, 591 of them in-sample after the boundary cut; the drifted names' industry exposure is now known from each 10-K's filing date. Report: `reports/equity_regression.md`, section "The point-in-time correction"; experiments.md row 290.

**What the two-point check could not say.** Of the 27 universe names, **11 changed industry before they joined the S&P 500** (AOS, AYI, BR, CRL, HUBB, LDOS, MGM, MSCI, PENN, SBAC, TDY), so their early-sample code never entered a regression. The correction reaches **16 names and 29,412 of 1,692,791 regression cells (1.74%)**, eleven cells per date in 2007 falling to none by 2024; JCI, BKNG, HON, JEF and IRM carry 2,200–3,900 cells each and the five REIT conversions in the set (AMT, CCI, EQIX, IRM, WY) 235–2,242.

**What moved.** Mean weighted daily R² **0.3874 from 0.3868** (+0.0006; median 0.3713 from 0.3703); `|t| > 2` frequencies by at most +0.5 pp (LIQUIDITY 24.1%, RESVOL 26.7%, BETA 57.6%, MOMENTUM 47.5%, SIZE 33.6%, **NLSIZE 11.0%** — flagged and kept by ruling 2); the thin-industry range 9–14 from 9–15; the constraint tests close to 2.6e-18 and 2.8e-17; the country gap's sd 2.4 bp. Rows 284–289 re-scored on the corrected panel: no verdict changes in kind, and §15.6.2's statement of the factor set's explanatory power stands as written rather than as a look-ahead artefact. Row 290 holds on all three legs and is `model-config` as registered: `N = 59`.

**The residual, named.** A change-and-revert between two 10-Ks of one of the 511 unchanged names is invisible to the two-point check and was not pulled. Its size is bounded by the re-coding rate among the names that did drift (41 code changes over 538 names in seventeen years) and it would have to change and revert inside the sample to matter.

## 15.7 Targets and validation

| Metric | Target | Source |
|---|---|---|
| Daily cross-sectional R² | **≥ 39%** | Connor (1995): statistical models reach 39.0%, fundamental 42.6%, macro 10.9%. 39% is the bar a free-data build should clear. |
| Per-factor rolling t-stats | report `abs(t) > 2` frequency per factor | Prune any style factor indistinguishable from noise — it adds estimation error to the covariance matrix for nothing. |
| Exposure VIFs | report | Catches the collinearity the orthogonalizations were meant to fix. |
| Optimizer-portfolio bias, pre-adjustment | **expect 1.4–1.7** at HL 84d | This is where the eigenfactor adjustment finally bites hard. Contrast with the macro model's much milder bias. |

**Optional, cheap, high-value:** run PCA on the *residuals* `u_n` and append the top 1–2 residual PCs as unnamed factors. Report R² with and without. If a residual PC's variance share spikes — the crowding/deleveraging signal no pre-specified factor set contains, as in August 2007 — that chart is worth more than another decimal place of R².

### 15.7.1 RULINGS (2026-09-05, W7-P4): the T×K frame at K = 53, size-decile buckets, per-name residual histories, the month-end grid, and scope

Five operator rulings issued before any code ran, then the constructions the implementing session took under them. The session's task is §15.2's test — feed the equity `(X, f, u)` through the unchanged `risk/` pipeline and §6's battery — and §15.7's target; **no file under `src/mafrm/risk/` may change**. If one has to, that is week 3's design failing and the session says so rather than patching.

**Ruling 1 — the T×K frame.** Industries never present on any regression date are **dropped from `K`**: they are not factors on this universe. `K = 53` (1 country + 46 industries + 6 styles), and every place the `K/T` result reads `K` reads 53. Intermittent industries stay, and their absent dates are **zero-filled in `factors/`**, with the bias quantified rather than hidden: on an absent date no name is exposed to the factor, so a zero return is consistent with every realised asset return, and the cost is that the factor's EWMA variance is understated by exactly the fraction of absent dates, `1 − f_k`. `f_k` and the implied understatement are reported per intermittent industry. Zero-fill is the one option that leaves `risk/` untouched and PSD-guaranteed — pairwise omission is not PSD, and restricting to the always-present set drops a dozen factors. **Reversing result, named now:** attribute the equity `B` by industry membership; if it is carried by names in intermittent industries, the zero-fill is the cause and the restricted variant runs as a `model-config` row. For W8-P1 the equity `K/T` points use `K` present per date, as `N` already does.

**Ruling 2 — the size-decile prior.** Ten equal-count buckets by screened cap at `t−1`, recomputed each date. A decile is equal-count by definition and daily recomputation is what the daily specific-risk stage already assumes. No new constant.

**Ruling 3 — the residual panel.** Each name's specific variance is estimated on its **own observed residuals with gaps removed**. A name out of the regression set contributes nothing on those dates and its EWMA resumes on re-entry; the 252-bar screen already excludes short histories. `risk/specific` refuses NaN by design, so the wrapper in `factors/` passes per-name observed series — per-name calls are not a change to `risk/`. If nothing short of a change to `risk/` works, §15.2's claim has failed and the session stops there.

**Ruling 4 — the month-end grid.** The pipeline runs on the **month-end dates** the bias statistic and the optimizer-selected family actually read, not on all 4,468 sessions: the EWMA/NW/VRA recursions are cheap and only §5.3's Monte Carlo is per-date expensive, and it is needed only where a forecast is scored. Order: short horizon at `a = 1.0` and `1.4` first — §15.7's target is at HL 84 and the eigenfactor stage is the one it says removes the bias — the long horizon after, if the session holds. `M = 2000` for comparability with the macro model unless the first date's measured wall time says otherwise, in which case `M = 1000` with the reason and the `1/√M` difference logged. One date is measured before the run is committed to.

**Ruling 5 — scope.** Both optional items are **out**: W3-P5's MP-denoising prediction at `N/T ≈ 0.11` goes to W8-P1 with the other regime predictions, and the residual-PC addition is out — "optional and cheap" is not a reason to add a factor to a model whose `K` the result depends on, in the week's heaviest session. The four pre-registered `K = 56` predictions now read at **`K = 53`**, stated beside each before it is scored.

**The wall-time measurement (ruling 4).** One `K = 53`, `T = 4,468` pre-VRA build on a synthetic frame at the short horizon: **2.6 s at `M = 2000`**, 1.3 s at `M = 1000`, cheap stages under 10 ms. 213 month-ends per horizon is about nine minutes; **`M` stays at 2000 and both horizons run**, fanned out as two processes.

**Constructions taken by the implementing session, none a parameter.** Each is a reading of a ruling or of an existing constraint, recorded so a later session does not mistake it for a choice with a number in it.

1. **The forecast made at the close of month-end `d` uses factor rows through `d` inclusive** (the regression's row `d` is the return over `d`, known at the close) and is **scored on every session of the following month with the weights held** — §6.2's "held between rebuilds" leg, the daily observations giving the exact interval its `T`, and the monthly aggregate (`R_m = Σ_t R_t`, `σ_m = σ_d √(days in m)`) reported beside it exactly as `reports/bias_statistics.md` does. The daily-held series is the headline because it is the series the macro model's chart is drawn from in shape, and W8-P1's contrast reads the two models on the same statistic.
2. **The names in a forecast are those with a screened cap, an industry and all six exposures at `d`**, the regression set the next session will use, with the design `X_d` built from them: a unit country column, one dummy per industry in `K`'s order, the six styles. Membership does not condition on the next session's return.
3. **A missing return inside the held month is zero** — the position is liquidated at its last bar — and the number of `(name, session)` cells it touches is counted and reported. Conditioning membership on a complete next-month return would be a look-ahead.
4. **Names with fewer observed residuals than the arithmetic floor `lags + 1` have no leg (a) estimate** and take leg (b)'s structural estimate outright: `γ_n` forced to 0 by absence, not by a ramp, so no observation-term constant is introduced; every other name keeps `γ_n = 1` as pinned in §5.5.2. The count per date is reported.
5. **Singleton names — alone in their industry at `d`, residual ≈ 0 by construction — are held out of stage (d)'s shrinkage target and out of the specific VRA's cross-section**, exactly the W4-P1b treatment of the two identity assets (§5.5.1 ruling 4, §5.5.4): a value that is a construction artefact may not build `σ̄` or `σ_Δ`. They stay in the panel, are bucketed, shrunk and reported. The registered singleton measurement is made on this production path, and the arithmetic that says what to expect of it is written at `experiments.md` row 294 before the run.
6. **The VRA multipliers are fitted on daily standardised returns under the held forecast**, half-lives in days as published: `B_t^F` from the daily factor returns against the held `√diag(F)`, `B_t^S` from the daily residuals of names present that session against the held `σ_n`, each `λ²` an EWMA at `τ_VRA` over the daily history through the month-end it is applied at (lagged by construction: the forecast made at `d_{i+1}` carries the history through `d_{i+1}`). No half-life is converted to months.
7. **The naive asset-level comparand runs on the always-present names** — those with a return on every session of the sample — because a zero-filled ragged return panel understates a recent joiner's variance by its whole absent fraction and an EWMA on it is not the estimator §6.2 names. Every factor variant's family 4 is **also** scored on that subset (`Σ_sub = X_sub F X_sub' + Δ_sub`, a sub-block of the same forecast) so the naive comparison is like for like; the full-universe series stays the headline. The subset is survivorship-labelled where it is reported.
8. **Family 1 is `validate` on the always-present names plus the pooled cross-sectional statistic over every `(name, session)` cell**, because a per-name `B` on a ragged membership has a different `T` for every name and the pooled statistic is the one with a single stated `T`.
9. **§10.3's component assertion defines the ex-post specific component as the remainder** `w'r − (X_d'w)'f`, as `backtest.diagnostics.attribute` does, so the identity closes exactly per period; `w'u` from the regression's own residuals is reported beside it with the gap, which measures exposure drift inside the month.
10. **The scored window is common to every variant and is §6.2.4's three conditions on the month-end grid**: at or after the last PSD-repair firing among month-end builds, `T` above the always-present name count so the naive comparand is of full rank, and at least one earlier forecast so a VRA multiplier exists. Drop counts are reported for each.
11. **Row 118's registered remedy is measured, not implemented.** Excluding floored eigen-directions from the `λ(k)` fit is a change under `risk/`; this session counts the firings and states whether any reaches a scored forecast, and W8 rules on the remedy with that number in hand.
12. **`M = 2000`** (config, unchanged) after the measurement above; the seed is `model.seed` through `default_rng` as everywhere.
13. **`K` is point in time, and it is 54 at the end of the sample, not 53.** Two facts met at the first build, before any covariance was read. First, the committed panel carries **Books** on 1,726 dates through 2014-02-07 — W7-P3's "three never present" was the pre-P3b count and row 292(a)'s registered list was copied from it — so only FabPr and Coal are dropped and `K = 54` (1 + 47 + 6); row 292(a) is refuted on its own falsifier and, as it says, `K` is corrected before anything is scored. Second, three industries first appear years into the sample (Soda 2012-07-02, Txtls 2013-12-24, Agric 2021-06-25), so a fixed-`K` frame hands `risk/` an all-zero column and the EWMA stage refuses a zero variance — correctly, there being no observation to estimate from. The reading of ruling 1 that invents nothing and touches nothing in `risk/` is **point-in-time `K`: an industry enters the frame at its first appearance** ("never present" applied as of each date), so `K_d` grows from 50 to 54 across the grid; reading the end-of-sample industry list at an earlier date would in any case be a look-ahead in the design. The `K_d × K_d` matrices are embedded in the full `K × K` layout with zeros outside their factors, which is exact downstream — no name is exposed to an industry that has not appeared, so `X F X'` is unchanged, and family 3 reads the present block back through the zero diagonal. A name whose industry first appears on the very next session meets a zero factor variance for that one month; the count is reported. The four pre-registered predictions are read at `K_d`, and W8's `K/T` points read it per date as ruling 1 already said.


### 15.7.2 FINDING (2026-09-05, W7-P4): §15.2 holds with `risk/` untouched; §15.7's 1.4–1.7 is refuted at **1.10**, below the macro model's 1.33 — and the split says why

Under the rulings of §15.7.1. Report: `reports/equity_bias_statistics.md` (with `.csv`, `.png`), `reports/equity_validation_battery.md`, `reports/equity_risk_stages.csv`, `reports/equity_lambda_curve.png`; experiments.md rows 291–297. Both horizons, both published `a`, `M = 2000`, 210 month-end builds each, 190 scored (2009-01-30 to 2024-10-31; 3,985 daily sessions, 190 months), names per forecast 305–458, always-present subset 447 names, 42 liquidated `(name, session)` cells. Wall time: about six minutes per horizon for the Monte Carlo, four for the render.

**§15.2 holds, verbatim (row 291(a)).** `git diff --stat HEAD -- src/mafrm/risk/` is empty, and every declared pre-VRA stage ran and passed the pipeline's own PSD assertion on every month-end build at both horizons (smallest eigenvalue after `psd_repair` 2.7e-6 bps², after the eigenfactor stage −8e-11 against a maximum near 10⁵, inside `checks.eigenvalue_floor`). The one thing the equity panel asked of the architecture that Model A never had — an industry that is a factor only from its first appearance — was met in `factors/` by point-in-time `K` (construction 13), which is the correct no-look-ahead reading anyway. The bias-statistic chart regenerated with zero changes to `risk/`, which was the week's acceptance criterion.

**§15.7's target is refuted (row 291(b)): the optimizer-selected portfolio's pre-adjustment bias is `B` = 1.1003 daily-held (short, `T` = 3,985, exact half-width 0.022), 1.191 on the monthly leg — not 1.4–1.7.** And the W8 contrast reverses (row 291(c)): 1.10 is **below** the macro model's 1.33, not above it. **The component assertion (row 295) says why, and it is the finding of the week.** On the same family-4 book, monthly, `T` = 190: **`B_specific` = 1.030, inside [0.899, 1.101]** — a diagonal `Δ` is in regime at `N ≈ 400`, exactly as §5.5.6 predicted — and **`B_factor` = 1.193, outside above**: the whole excess is estimation error in the factor covariance, the term §5.3 exists to correct. The macro model's 1.33 was never that term: C1 (rows 162–163) showed 107% of it was the diagonal's specification error at `N = 13`, and the equity panel is where the two are separable. Read together: **`K/T` scales the estimation-error term, and that term is 1.19 at `K_d` = 51–54 against a factor-component excess the macro model could not even show; the macro model's larger total was a different error.** The zero-fill is not the cause (row 292(c): intermittent-industry names' `B` 0.974 against 0.969 for the rest, family 4's weight on them 2.2%), and the naive comparand confirms the regime (row 296(b)): on the same 447-name subset the sample covariance's family-4 `B` is **3.22** against the factor model's 1.10 — the reversal of row 190, where at `N = 13` the naive estimator *beat* the factor model because it carried no diagonal to be wrong about.

**The eigenfactor adjustment finally bites, and it bites where the component split says it should (rows 291(d)–(e), 295).** Family 4 moves **1.100 → 1.044 at `a = 1.0` and 1.025 at `a = 1.4`** (55% and 76% of the gap to 1, against −0.0001 at `K = 6`); `B_factor` moves 1.193 → 1.097, back inside its interval, while `B_specific` stays inside (1.010); family 3's MRAD falls 0.120 → 0.090. H3's first half, refuted at `K = 6`, holds at `K = 54`.

**The four `K = 56` predictions, read at `K_d` = 51–54, all hold (row 293).** The parabola amplitude is **0.114** median (0.097–0.253), eleven times the 0.0099 at `K = 6`, and tracks `K/T_eff` through the grid (2.78 on the first build at 0.64; 0.10 at 0.037). Raw `λ` at the dominant eigenvalue (0.985) sits above the bulk minimum (0.966) on **100%** of scored month-ends while the smallest eigenvalue's is 1.047 — the spiked-spectrum return toward 1, registered in W3-P3b so that no session would diagnose it. The PSD repair fired **once per horizon, on the first build only** (2007-06-29, `K/T_eff` = 0.84, one direction floored), never at `K/T_eff` ≤ 0.64, and **no firing reaches a scored forecast** — so row 118's remedy stays registered and unimplemented for W8, with the count. Row 100's Bartlett fallback never fired.

**§5.5's shrinkage does not rescue a singleton, and the arithmetic written at row 294 before the run said it would not.** 13 names sit alone in their FF49 industry on 1,285 scored cells (NEM, LMT, CBRE, MNST, NWL, MHK, GD, HAS …); their residual is ≈ 0 by construction, so `σ_TS` is an *extreme* value, `v ≈ 1`, and stage (d) shrinks it **least**: after shrinkage they sit at the **0.7th** cross-sectional percentile (pure singletons 0.6, intermittent 0.9), 190 cells at the pre-shrinkage floor. Held out of the target (construction 5) they moved a decile's target by up to 10%; family 4 puts **7.2%** of its absolute weight on them. **No thin-industry constant is added on this session's authority** (§15.6.1 ruling 3); the number is flagged to the operator, and the two options that invent nothing are named: a singleton takes leg (b)'s structural estimate (construction 4 extended from "no observations" to "no information"), or its industry is merged under a documented rule. Either is a `model-config` row.

**What else moved, stated rather than netted.** Leg (b) is identified at last: `R²` 0.998 on 336 residual d.o.f. against 0.833 on 6 — and still reaches no forecast except below the arithmetic floor, because `γ` is pinned by ruling; the `Z` diagnostic's median is 0.25 and its maximum infinite wherever a name's interquartile range is zero. The specific-leg Bartlett fallback fired 70 times over 190 × 400 name-months. The random-book control on the component assertion lands at **80%** inside against a registered 90% (row 295(c) refuted; median random `B_specific` 0.951), and the fully specified variant's family 2 lands at 0.954, outside the interval (row 296(a) refuted on that variant alone): the typical book's specific risk is over-forecast by about 5% (family 1 pooled 0.967) while the VRA's cross-sectional statistic, a mean of squares that a few understated names dominate, marks it up further (`λ_S²` 1.25 short at the last month-end) — the two disagree about the level, suspect named, sign consistent, not chased (residual stopping rule). The battery's legs that follow from a large `B` (Kupiec, Mincer-Zarnowitz slope, Basel) refute at `B` = 1.10 for the reason row 297 anticipated — it is row 291(b) failing, not the tests; clustering (Ljung-Box, Christoffersen) and the fat tails hold as at `K = 6`. Newey-West's lag terms are net **positive** on the equity factor set (family 4 1.059 → 1.100 from `ewma` to `newey_west`), the opposite sign to Model A's −3.5%. The monthly leg reads higher than the daily-held one throughout (1.19 against 1.10 pre-adjustment); the held forecast is stale by up to a month and the monthly sum has its own autocorrelation, and which of the two W8 should read against the macro model is stated there, not chosen here.

**Residuals, named.** A change-and-revert of SIC inside the 511 unchanged names (from W7-P3b), unmeasured. The exposure drift inside a held month: the remainder `w'r − (X'w)'f` differs from `w'u` with a monthly sd of 37 bp against a specific-return sd of 85 bp — reported, and the reason the component split uses the remainder. The specific VRA's over-marking above, suspect named.

**Trial count.** Row 291 is `model-config`; `N` = 60. Rows 292–297 are `data-diagnostic`. Nothing was switched on any result; the two `model-config` decisions the session could have taken — the restricted-`K` variant (row 292(c), not owed: the zero-fill is not the cause) and row 118's remedy (not owed: not on the production path) — are recorded as not taken, with the numbers that decided them.

### 15.7.3 RULING (2026-09-06, W7-P4b): a singleton takes leg (b)'s structural estimate — construction 4 extended from "no observations" to "no information"

**Row 294's option (a).** A name alone in its FF49 industry has a residual that is ≈ 0 by the regression's identification, and that is not an observation of specific risk; §5.5 already says what to do with a name that has none — leg (b), the structural model, now identified on this panel at `R²` 0.998 on 336 residual d.o.f. Construction 4 ("no leg (a) estimate below the arithmetic floor") is extended to "no leg (a) estimate where the residual carries no information", and the singleton takes `σ^STR` outright, exactly as a short-history name does. It invents nothing: the fallback is §5.5's own published leg applied where its premise holds. Option (b) — merging industries — would change the factor set and needs a merge rule that is a choice; `K` stays as designed. Row 294's registration said no constant would be added unless the published mechanism was not enough; it is not, by its own arithmetic, so the published fallback applies. Singletons stay out of stage (d)'s target and the specific VRA's cross-section (construction 5).

**Counted `model-config`**, `N` → 61: it changes the production specific-risk estimate for the names it touches. **Specific risk only** — the factor covariance and §5.3's Monte Carlo are not re-run; the eigenfactor partials are reused unchanged. Registered before the run at experiments.md row 298 with three thresholds: (i) the median singleton cell's shrunk specific risk at or above the **10th** cross-sectional percentile; (ii) family 4's absolute weight on the singleton names **falls** from 7.2%, reported beside their count share (13 of 447 always-present names, 2.9%; 13 of ~380 names per date, 3.4%); (iii) family-4 `B` and the component split re-scored on the fixed panel with the change stated. If the fix takes the names off the floor, the specific VRA's family-2 figure (0.954 on the fully specified variant) and the random control (80% inside) are re-read **once** — the suspect named at §15.7.2 is a mean of squares those names dominate — and whether they moved is reported; nothing is chased further.

**Two corrections to the record, under the operator's name.** `K` = 54, not 53 — "three never present" was the pre-P3b count; and `K` is point in time, `K_d` 51–54, which the fixed-frame ruling did not anticipate and `risk/` correctly refused. Both stand as recorded at construction 13.

**Like-for-like, corrected against the macro report before it was written.** `reports/bias_statistics.md` gives the macro model's pre-eigen family-4 `B` as **1.3322 daily-rebalanced** and **1.6417 on its monthly leg** (weights fixed at the first scored date of the month, `R_m` the sum of daily returns, `σ_m` scaled by √days). The equity model's figures are **1.10 daily-held** (weights fixed at the month-end, scored on every session) and **1.19 monthly** (the same construction as the macro's monthly leg). The write-up compares 1.19 against 1.64 and 1.10 against 1.33, with the holding convention stated on each; neither pairing is identical, and the daily one differs in that the macro book is rebuilt every session while the equity book is held for a month.

### 15.7.4 FINDING (2026-09-06, W7-P4b): the structural fallback takes the singletons off the floor, the optimizer's loading on them halves, and the once-only re-read refutes the suspect named at §15.7.2

Row 298, `equity_risk.singleton_specific_risk = structural`; specific risk only, eigenfactor partials reused; `reports/equity_bias_statistics.md` re-rendered under the fix (rows 291–297's registered results stand as scored at commit 680a207 and are re-scored beside them).

**All three thresholds hold.** (i) The 13 singleton names' shrunk specific risk sits at the **48th** cross-sectional percentile over the same 1,285 cells (from the 0.7th; pure singletons 62nd, intermittent 37th), and no cell is at the pre-shrinkage floor; holding them out now moves a decile target by at most 6% (from 10%). (ii) Family 4's absolute weight on them falls to **4.3% from 7.2%**, against a count share of 2.9% of the always-present names and 3.4% of the mean regression set — still above their share, no longer double it. (iii) Family-4 `B` moves **1.1003 → 1.0874** daily-held and 1.191 → 1.184 monthly; on the component split the specific `B` moves 1.030 → **0.968** (inside) and the factor `B` 1.193 → 1.183 (outside); after the eigenfactor adjustment family 4 is 1.034 (`a = 1.0`) and `B_total` 0.999. The excess is now entirely the factor component, and the specific component reads a slight over-forecast.

**The re-read, once, and it refutes the suspect.** The fully specified variant's family 2 moved **0.954 → 0.949**, further from 1, and the random control **80% → 76%** inside with a median specific `B` of 0.943; the two VRA variants' family 2 now sits at 0.978, a hair outside the interval's 0.978 lower edge. So the singletons were **not** what dominated the specific VRA's mean of squares. What the numbers now say is a sign disagreement: the typical name's specific risk is over-forecast by 3–6% before the VRA (family 1 pooled 0.966, random books' median specific `B` 0.943), while the cross-sectional `B_t^S` — an equally weighted mean of squares — sits above 1 and marks the whole cross-section **up** (`λ_S²` 1.25 at the last month-end), which is a fat right tail of `(u_nt/σ_n)²` across names rather than any identifiable subset. That is the residual, with its new suspect named (the mean-of-squares aggregation against a right-skewed cross-section; the median would read below 1); it is not chased, per the ruling and the stopping rule.

**Unchanged by the fix, as it should be:** rows 291(a), 293 (every leg), 296(b) at 3.22 against 1.088; row 292(c) still holds with the intermittent-industry names' `B` now 0.851 (six names, several of them the singletons whose σ moved). Row 297's refutations stand for the same reason as before.

**Trial count.** Row 298 is `model-config`; `N` = 61.

## 15.8 What this module is NOT

State this in the README so nobody mistakes the claim:

- It is **not** a USE4 replica. It has 6 of 12 style factors, Fama-French 49 rather than 60 custom GICS-based industries, single-industry rather than multi-segment exposures, no Non-Linear Beta, and an approximate point-in-time universe.
- It is **not** competing with `toraniko`, which does cross-sectional equity well and should be cited as the reference implementation in the space.
- It exists to (a) demonstrate the estimation machinery on a high-K problem and (b) produce the K/T scaling result. Both of those are true of a 6-style model and neither requires the missing fundamentals.

**Do not oversell it in an interview.** The line that works is: *"Six of the twelve USE4 style factors — the ones you can get without point-in-time fundamentals and I/B/E/S — plus Fama-French 49 industries, √cap-weighted WLS, cap-weighted industry sum-to-zero constraint. I built it to have a high-K problem to run the covariance machinery on, and the interesting result is how the optimizer bias scales with K/T."* That sentence is honest, specific, and every clause of it is checkable.


---

# 16. Primary sources

**Risk model construction**
- Menchero, Orr & Wang (2011), *The Barra US Equity Model (USE4) Methodology Notes* — the primary build spec. Sections 2–5 and Appendices A–B. `top1000funds.com/wp-content/uploads/2011/09/USE4_Methodology_Notes_August_2011.pdf`
- MSCI, *CNE5 Descriptor Details* (Sept 2013) — the only public source with descriptor formulas **and weights**. `cslt.org/mediawiki/images/1/12/MSCI-CNE5-201309.pdf`
- Menchero, Wang & Orr (2011/2012), *Eigen-Adjusted Covariance Matrices* / *Improving Risk Forecasts for Optimized Portfolios*, FAJ 68(3) 40–50. SSRN 1915318. **The central paper.**
- Shepard (2009), *Second Order Risk*, arXiv:0908.2455.
- Menchero & Ji (2024), *Evaluating and Comparing Risk Model Performance*, JPM 50(3) — the modern successor to the USE4 validation appendix.
- Connor (1995), *The Three Types of Factor Models*, FAJ — macro 10.9% / statistical 39.0% / fundamental 42.6%.
- Ledoit & Wolf (2004), *Honey, I Shrunk the Sample Covariance Matrix*, JPM. SSRN 433840.
- López de Prado (2020), *Machine Learning for Asset Managers*, ch. 2 (denoising) and ch. 7 (NCO).
- Jagannathan & Ma (2003), *Risk Reduction in Large Portfolios*, JF 58, 1651–1684.
- *It's Just Beta*, ch. 8 "Risk Model Assembly" — itsjustbeta.com. Best free write-up of the production pipeline.

**Costs and implementation**
- Perold (1988), *The Implementation Shortfall: Paper versus Reality*, JPM 14(3) 4–9.
- Almgren & Chriss (2000), *Optimal Execution of Portfolio Transactions*, J. Risk 3(2) 5–39.
- Almgren, Thum, Hauptmann & Li (2005), *Direct Estimation of Equity Market Impact*, Risk 18 — γ=0.314, η=0.142, β=0.600.
- Tóth et al. (2011), *Anomalous Price Impact and the Critical Nature of Liquidity*, PRX 1, 021006 — the square-root law.
- Frazzini, Israel & Moskowitz (2018), *Trading Costs*. SSRN 3229719.
- Israel, Jiang & Ross (2017), *Craftsmanship Alpha*, JPM 44(2) 23–35.
- Boyd, Busseti, Diamond, Kahn, Koh, Nystrup & Speth (2017), *Multi-Period Trading via Convex Optimization*, arXiv:1705.00109.
- Boyd, Johansson, Kahn, Schiele & Schmelzer (2024), *Markowitz Portfolio Construction at Seventy*, arXiv:2401.05080, JPM 50(8).
- Gârleanu & Pedersen (2013), *Dynamic Trading with Predictable Returns and Transaction Costs*, JF 68(6).
- MSCI, *Active Portfolio Construction When Risk and Alpha Factors Are Misaligned*.
- Harvey & Ustinov (Man Group), *Quantifying Long-Term Market Impact*.
- Hwang & Satchell (2001), *Tracking Error: Ex-Ante versus Ex-Post Measures*, JAM.
- Menchero & Davis (2011), *Risk Contribution Is Exposure Times Volatility Times Correlation*, JPM 37(2).

**Data and method**
- Gürkaynak, Sack & Wright — Fed GSW nominal yield curve, `federalreserve.gov/data/nominal-yield-curve.htm`; TIPS curve, `federalreserve.gov/data/tips-yield-curve-and-inflation-compensation.htm`
- Ardia, Guidotti & Kroencke (2024), *Efficient Estimation of Bid-Ask Spreads from Open, High, Low, and Close Prices*, JFE 161:103916. `github.com/eguidotti/bidask`
- Ken French Data Library; AQR Data Library (Century of Factor Premia, TSMOM, Commodities for the Long Run).
- Bailey & López de Prado — Deflated Sharpe Ratio; PBO via CSCV.
- Lo (2002), *The Statistics of Sharpe Ratios*, FAJ.
- Yin, Miki, Lesnichenko & Gural (2026), *Implementation Risk in Portfolio Backtesting*, alphaXiv 2603.20319.
- Tidy Finance (`tidy-finance.org/python/`) — the reproducibility template to steal wholesale.

**Equity module additions**
- Fama & French 49-industry SIC mapping — Ken French Data Library.
- Kakushadze & Yu (2017), *Open Source Fundamental Industry Classification*, Data 2(2):20 — a published free GICS substitute built for risk-model use.
- Hornli Quant (2025), *Building an Open-Source Barra-Like Specific Return Model*, SSRN 5624990 — closest existing free-data equity replica.
- `github.com/UePG-21/Barra-risk-model` (covariance adjustments), `github.com/YTZzzzz/Barra_CNE5` (descriptor construction), `github.com/0xfdf/toraniko` (the reference cross-sectional implementation).
