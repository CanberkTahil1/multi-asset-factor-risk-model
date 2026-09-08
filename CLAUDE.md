# CLAUDE.md

Project constitution. Claude Code reads this at the start of every session. Keep it in the repo root.

Reference document: `SPEC.md` — the full methodology with formulas, published parameter values and sources. **When a task references a section (§5.3, §15.4), read that section of `SPEC.md` before writing code.** Do not reconstruct the methodology from memory.

---

## What this project is

Two factor risk models sharing one covariance pipeline, wrapped in a cost-aware convex optimizer, built to answer one question:

> How much of the gap between a model portfolio's paper returns and its realised returns is estimation error in the risk model, and how much is implementation cost — and which fixes actually close each?

The answer is an exact identity, and every deliverable exists to fill in its two terms:

```
SR_paper − SR_real = (μ_g/σ_f)·(1 − 1/B)  +  TC/(B·σ_f)
                     ─── risk-model ───      ─── cost ───

B = σ_realised / σ_forecast     (the bias statistic)
```

---

## THIS PROJECT DOES NOT FORECAST RETURNS

If a task starts to look like alpha research, signal discovery, or making a backtest look better, **stop and say so**. That is a different project and it takes six months. Expected returns enter only as a fixed, documented input to the optimizer so that it has something to trade against — never as something to improve.

---

## Hard invariants

Violating any of these is a bug even if the tests pass.

1. **Network access lives only in `src/mafrm/data/loaders.py`.** Every other module reads from the cache. If you need data in `factors/`, `risk/`, `costs/` or `backtest/`, add a loader — do not fetch inline.
2. **Never request `interval="1wk"` or `"1mo"` from yfinance.** Yahoo mis-applies dividend adjustments to weekly and monthly bars (yfinance issue #1273) and silently inflates returns. Pull daily, aggregate yourself.
3. **Never import `cvxportfolio`.** It is GPL-3.0 and would make this repo GPL. Reimplement its cost specification from `SPEC.md` §7 and cite it. Use `cvxpy` directly.
4. **Assert PSD after every covariance transformation stage**, not just at the end. Newey-West routinely produces non-PSD matrices. Repair by eigenvalue flooring at `1e-14` and log when it fires.
5. **Never read or evaluate anything on or after `HOLDOUT_START`** (in `config/model.yaml`). All development, tuning and model selection happens strictly before it. Once looked at, it is spent. If a task seems to need holdout data, stop and ask.
6. **No magic numbers in code.** Every half-life, lag count, shrinkage constant and cost coefficient comes from `config/model.yaml`. If a value is not in the config, add it there first.
7. **Log every evaluated configuration to `experiments.md`** — hypothesis, what would falsify it, result, date. The deflated Sharpe at the end needs an honest count and there is no way to reconstruct it later.
8. **Never commit third-party raw data.** AQR and Ken French have no explicit redistribution licence; ICE data forbids reproduction; JST is CC BY-NC-SA. Commit `data/manifest.json` (URL, timestamp, SHA-256, row count, date range), never the files.
9. **Do not invent a parameter.** Every published constant is in the table below or in `SPEC.md`. If you need one that is not, stop and ask rather than guessing a plausible value.
10. **`src/mafrm/risk/` must not know what asset class it is looking at.** It takes `(factor_returns, exposures, residuals, config)` and returns matrices. This is what makes the week-7 equity module cost one week instead of three. See §15.2.

---

## Parameter table — single source of truth

Mirror this into `config/model.yaml`. Sources are in `SPEC.md`; USE4 Table 4.1 unless noted.

| Parameter | Short horizon | Long horizon | Note |
|---|---|---|---|
| Factor volatility EWMA half-life | 84d | 252d | |
| Factor correlation EWMA half-life | 504d | 504d | identical across horizons — deliberate |
| Volatility Newey-West lags | 5 | 5 | |
| Correlation Newey-West lags | 2 | 2 | |
| VRA half-life | 42d | 168d | same value for factor and specific — deliberate |
| Specific risk EWMA half-life | 84d | 252d | |
| Specific risk NW lags / AC half-life | 5 / 252d | 5 / 252d | |
| Bayesian shrinkage `q` | 0.1 | 0.1 | |
| Eigenfactor Monte Carlo trials `M` | 1000–3000 | | not published; any value in range |
| Eigenfactor scaling `a` | **run both 1.0 and 1.4** | | 1.4 = optimizer-facing, 1.0 = attribution-facing (USE4 production) |
| EWMA effective sample size | `T_eff = 2τ/ln2` | | 84d → 242, 252d → 727, 504d → 1454 |

**Equity descriptors** (CNE5 Descriptor Details, §15.4):

| Descriptor | Window | Half-life | Other |
|---|---|---|---|
| BETA | 252d | 63d | vs cap-weighted market excess return |
| MOMENTUM (RSTR) | T=504d | 126d | **lagged L=21d** — sum runs t=21…525 |
| DASTD | 252d | 42d | |
| HSIGMA | 252d | 63d | sd of BETA-regression residuals |
| RESVOL composite | | | `0.74·DASTD + 0.16·CMRA + 0.10·HSIGMA` |
| LIQUIDITY composite | | | `0.35·STOM + 0.35·STOQ + 0.30·STOA` |

**Costs** (§7.2):

| Parameter | Value | Source |
|---|---|---|
| Square-root prefactor `Y`, patient | 0.58 | backed out of AQR: 2% ADV → 17bps, 6% → 28bps at σ=2%/day |
| Square-root prefactor `Y`, urgent | 1.40 | backed out of Virtu US trailing-year ~40bps |
| Cost exponent | 1.5 | total cost = size × √impact |
| `γ_trade` | sweep 1.0–3.0 | desks run >1 as a safety margin; document the choice |
| EDGE spread window | 63d rolling, monthly step | must be time-varying |

---

## Definition of done

A task is complete when **all** of the following hold. Do not report a task finished otherwise — say what is blocking instead.

- [ ] Code written, `ruff` clean, `mypy` clean on `src/`
- [ ] Tests written **and passing**, including at least one with a hand-computed expected value
- [ ] PSD assertions in place if the task touched a covariance matrix
- [ ] No new magic numbers — anything tunable is in `config/model.yaml`
- [ ] `experiments.md` updated if any configuration was evaluated
- [ ] Any generated figure or table written to `reports/` and committed
- [ ] A one-paragraph summary of what was built, what was assumed, and what is still wrong

**If tests fail, or you could not find a data source, or a result looks wrong — say so plainly.** Do not report partial work as complete, and do not adjust a test to make it pass.

---

## Residual stopping rule

A residual may be documented and left open when **all three** hold:

1. it does not affect the production data path;
2. it is small relative to the quantity it perturbs;
3. it has a named suspect with a stated reason it is untestable within scope.

**Spend at most one additional session past the first diagnosis.** Log it, name it, move on.

This is a budget, not permission to hand-wave. A residual left open still owes the full record: what was measured, what was ruled out, what the suspect is, and why it cannot be settled here. "Unexplained, suspect named, sign consistent, untestable within scope" is a far stronger statement than "unexplained", and it costs one paragraph. What it must never become is a tolerance widened until the residual fits inside it — see §6.6 and the W1-P3 rows in `experiments.md`.

The reason for the budget is arithmetic. Weeks 3 and 4 are nothing but residuals — the covariance pipeline generates them by construction — and a project that spends two sessions chasing four basis points on a validation comparand takes six months instead of eight weeks. W1-P3 is the worked example: three sessions of diagnosis produced three real data traps and a refuted pre-registered prediction, and the fourth moved the answer by 0.04% on a series the factor models never see. The first three were worth it. The fourth was the one the rule exists to prevent.

---

## Known failure modes — check yourself against these

Ranked by how often they break builds of this shape. Items 1–8 were listed at the outset; item 9 was added in W2-P3 after the same mistake produced three separate wrong readings.

1. **Calendar and alignment bugs.** Different assets trade on different holidays. `.last()` vs `.resample('ME')` vs business-month-end give different series. This is the top source of "my Sharpe is 2.4" moments.
2. **Skipping the orthogonalizations.** Un-orthogonalized credit/equity factors run ~0.7 correlated and the covariance matrix goes near-singular in exactly the crises that matter. Same for Non-Linear Size vs Size, and Residual Volatility vs Beta.
3. **Non-PSD after Newey-West.** Guaranteed to happen. Invariant 4.
4. **Validating only on random portfolios.** They look fine for *any* covariance matrix including the naive sample estimator. The test that matters is optimizer-selected portfolios. See §6.2.
5. **Look-ahead through adjusted closes.** Yahoo rewrites history backwards when a dividend is paid. Cache raw unadjusted OHLCV plus a separate actions table; apply adjustments deterministically yourself.
6. **Optimizer infeasibility** once turnover, box constraints and costs stack. Soft constraints plus a relaxation ladder, planned from the start, not patched later.
7. **Corrections interacting badly.** Each adjustment passes its own sanity check; applied in sequence they compound or cancel. Assert after each.
8. **Using equal or cap weights in the equity WLS instead of √cap.** MSCI's stated assumption is that specific return variance is inversely proportional to √(market cap). It materially changes factor returns.
9. **Reading overlapping rolling windows as independent observations.** A rolling statistic stepped one period at a time shares all but one observation with its neighbour, so its series is enormously autocorrelated: consecutive readings are near-duplicates, the effective sample size is a small fraction of the window count, and a normal-theory standard error computed as if the windows were independent is far too small. This has now produced **three separate wrong readings in this project**, which is why it is listed rather than re-learned:
   - **`experiments.md` row 58** — an apparent +7.6bp effect at 62-of-63 overlap;
   - **the alpha flag** — 251-of-252 overlap, where a "persistent" breach is mostly one breach seen 252 times;
   - **W2-P3's rolling correlation** — 35-of-36 overlap, where the 36-month range spans 3–6 Fisher-z standard errors and *neither* "within noise" nor "significantly varying" is an available reading.

   Weeks 4 and 6 are built almost entirely from rolling windows — the bias statistic, the rolling bias chart, the VRA, every rolling exposure — so assume the failure is present by default. **Never quote a count of rolling windows as `n`.** State the overlap next to any rolling statistic, and where a claim depends on one, either use non-overlapping windows or say explicitly that the uncertainty is not quantified. §6.1's confidence bands are the worked example of doing it properly.

---

## Code conventions

- Python ≥ 3.11. `uv` with a committed `uv.lock`. Pin `yfinance` tightly — highest-churn dependency.
- `numpy`, `pandas`, `scipy`, `cvxpy`, `pyyaml`, `bidask`. `skfolio` only for combinatorial purged CV. **Not** `cvxportfolio`.
- Type hints on every public function. `mypy` on `src/`, not on tests or notebooks.
- One `SEED` constant in `config/`, passed explicitly to every stochastic call via `np.random.default_rng(seed)`. **Never a global `np.random.seed()`** — it makes the eigenfactor Monte Carlo irreproducible.
- Notebooks are for exploration only. Never import from them. `nbstripout` on commit.
- Docstrings on anything implementing a published method must cite it: `"""USE4 Eq. 4.3. Menchero, Orr & Wang (2011)."""`

## Version control

- **`main` only.** Solo project. Do not create feature branches, and do not open pull requests.
- **One or a few commits per work session, all named by task ID, made at the end** after tests pass. The purpose is a log that reads as a project timeline, not a commit count. **Do not squash to hit a number.** Do not commit mid-task, and do not commit unless the end-of-session checklist has been run.
- **Commit message format:** `<task-id>: <what changed>` — for example `W3-P3: eigenfactor adjustment, both a=1.0 and a=1.4`. Put anything deliberately left broken or unfinished in the commit body. The log is the project timeline and the next session reads it to orient.
- **Never commit if `make test` fails.** A red commit is worse than no commit, because the next session starts from it and assumes it works.
- **Never `git push --force`, never rewrite history that has been pushed, never amend a pushed commit.** Local unpushed commits may be reorganised freely, but there is no reason to. The repository will be made public at the end of the project and its history is part of the artefact.
- **Never commit:** anything under `data/raw/` or `data/processed/`, `.env`, API keys, or any third-party raw data. **Always commit:** `data/manifest.json`, `uv.lock`, everything in `reports/`, `results/metrics.json`, and the synthetic golden-file test fixture.
- **The repository is private during the build** and is made public only after the W8-P4 audit. Do not change its visibility.
- Secrets come from environment variables loaded from a gitignored `.env`. Never a literal in code, never a default value in the source.

## Layout

```
config/universe.yaml     frozen, dated, rationale per line
config/model.yaml        every tunable number
src/mafrm/data/          loaders, cache, manifest, contracts  ← only network access
src/mafrm/factors/       macro, statistical, equity cross-sectional
src/mafrm/risk/          covariance, specific risk, validation  ← asset-class agnostic
src/mafrm/costs/         EDGE, ADV, impact  ← most heavily tested module
src/mafrm/backtest/      engine, optimizer, constraints, reporting
tests/
reports/                 generated figures and tables, COMMITTED
experiments.md           every configuration evaluated
SPEC.md                  the methodology reference
```

## Commands

```
make data      # refresh cache from sources, update manifest
make verify    # re-hash everything, fail on drift
make model     # build factors, covariance, specific risk
make backtest  # run the experiment grid
make report    # regenerate reports/ and results/metrics.json
make test      # ruff + mypy + pytest
```
