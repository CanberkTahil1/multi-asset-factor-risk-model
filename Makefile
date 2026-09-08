# Makefile -- the commands named in CLAUDE.md. Every target is reproducible
# from a clean clone; none of them takes an argument.
#
# `uv` is the project runner (CLAUDE.md, code conventions). If it is not on
# PATH we fall back to whatever python is active, so a clean clone can still
# run `make test` before uv is installed.

# Put src/ on the import path for every recipe. The package is installed
# editable, but on macOS uv writes the editable .pth with the UF_HIDDEN flag and
# CPython >= 3.12 silently SKIPS hidden .pth files, so `python -m mafrm...`
# fails while ruff and mypy -- which do not need the package importable -- stay
# green. pyproject sets the equivalent for pytest; this covers the module-entry
# targets (data, verify, model, backtest, report). Prepended, so an existing
# PYTHONPATH is preserved rather than clobbered.
export PYTHONPATH := src$(if $(PYTHONPATH),:$(PYTHONPATH))

# `--extra dev --frozen` is load-bearing, not decoration (W8-P4c). A bare
# `uv run` syncs only the BASE dependencies, so on a project environment that
# has no dev extra installed, `uv run mypy` silently falls through to whatever
# mypy is on PATH -- an ambient 1.18.2 was measured doing exactly that against
# the lock's 2.3.1, and reporting a gate result from an unpinned type checker
# is the one failure this target must not have. `--extra dev` puts the locked
# toolchain in the environment before anything runs; `--frozen` makes uv refuse
# to re-lock behind your back rather than quietly resolving something new.
# `make report` needs it too: the bt reconciliation runs off the dev extra.
UV := $(shell command -v uv 2>/dev/null)
ifeq ($(UV),)
RUN :=
PY  := python
else
RUN := uv run --extra dev --frozen
PY  := uv run --extra dev --frozen python
endif

# pytest exits 5 when it collects no tests. That is not a failure for a repo
# whose test suite is still being written -- ruff and mypy have already run.
PYTEST_OK = || [ $$? -eq 5 ]

.PHONY: all install data verify model backtest holdout overfitting report test lint typecheck unit clean

all: test

## Install the project and its dev tooling into the active environment.
install:
ifeq ($(UV),)
	pip install -e ".[dev]"
else
	uv sync --extra dev
endif

## Refresh the cache from live sources and update data/manifest.json.
## CLAUDE.md invariant 1: this is the only target permitted to touch the network.
data:
	$(PY) -m mafrm.data.loaders refresh
# SPEC.md 15.3 (W7-P1): rebuild the committed S&P 500 membership tables in
# data/reference/ from the cached Wikipedia pages and Yahoo bars, and record
# them in the manifest. Tracked files, licensed CC BY-SA 4.0 in their header.
	$(PY) -m mafrm.data.sp500_reference

## Re-hash every cached file against data/manifest.json and fail on drift.
## Succeeds on an empty manifest -- a clean clone has nothing cached yet.
verify:
	$(PY) -m mafrm.data.cache verify

## Build factors, the covariance pipeline and specific risk.
##
## ============================================================================
## GREEN AS OF W4-P1 -- the first time in the project. Red from W3-P1 to W3-P6,
## and by construction rather than by accident. If you have arrived here because
## it is red again, THAT IS THE TARGET WORKING. DO NOT "FIX" IT.
##
## There are exactly two correct responses to a red `make model` -- implement
## the missing piece, or leave it red. There is no third. In particular, do NOT:
##   - delete a stage from `covariance.stages` in config/model.yaml (the
##     declaration IS the specification; removing a stage does not implement it);
##   - weaken or skip either audit in mafrm.build;
##   - empty `mafrm.build._UNBUILT` by hand rather than by building the thing;
##   - append `|| true` here, or drop mafrm.build from the recipe.
## Any of those restores a green build target that passes by omission, which is
## the exact defect the audit was added to detect -- the same class of defect as
## a test that cannot detect its own failure mode. CI does not run this target,
## so a red `make model` blocks nothing; `make test` is the commit gate.
## ============================================================================
##
## TWO AUDITS RUN HERE AND BOTH MUST BE EMPTY FOR THE TARGET TO PASS.
## `mafrm.build` compares the pipeline stages DECLARED in config/model.yaml
## (`covariance.stages`) against those implemented in mafrm.risk.covariance, and
## compares this target's NAME -- factors, covariance, specific risk -- against
## `mafrm.build._UNBUILT`. It prints both lists and EXITS NON-ZERO on any gap.
##
## Until W3-P1 this target exited green while building Model A alone, with the
## covariance pipeline, Model B and the hybrid all absent -- a build target that
## passes by omission is the same class of defect as a test that cannot detect
## its own failure mode. W4-P1 built SPEC.md 5.5, which emptied the second audit,
## so a green line here now means both audits ran and found nothing rather than
## that nobody asked. Do NOT route around a future red by dropping a stage from
## the config: the declaration is the specification.
model:
	$(PY) -m mafrm.factors.macro
	$(PY) -m mafrm.factors.betas
	$(PY) -m mafrm.build

## Run the experiment grid.
##
## Two steps. W6-P1's ONE verification configuration first (SPEC.md 8.5.1
## ruling 6; reports/optimizer_verification.md, reports/binding_constraints.md),
## then SPEC.md 9's grid (W6-P2; SPEC.md 9.1): seven covariance treatments by
## four cost treatments, Model A, every cell a strategy-config trial, writing
## reports/experiment_grid.md, reports/experiment_grid.csv,
## reports/experiment_grid.png and results/metrics.json. The grid reads the
## deflated Sharpe's trial count N from experiments.md at run time and refuses
## to run if the cells have not been registered there first. Nothing on or
## after sample.holdout_start is read, and the runner asserts that in code.
## Reads the committed eigenfactor cache and data/raw; about ten minutes.
## W6-P3 adds, after the grid: SPEC.md 10.1-10.3's diagnostics (written by the
## grid run into reports/diagnostics.md), the twelve one-dimensional bands off
## the reference cell (reports/bands.md; about five minutes; each a
## strategy-config row registered in experiments.md first), and SPEC.md 10.4's
## re-optimised capacity curve at fifty AUMs in both Y regimes
## (reports/capacity_reoptimised.md; data-diagnostic; about half an hour).
backtest:
	$(PY) -m mafrm.backtest.verification
	$(PY) -m mafrm.backtest.grid
	$(PY) -m mafrm.backtest.bands
	$(PY) -m mafrm.backtest.capacity_curve
# W8-P1: the reference cell re-solved at every point of SPEC.md 5.1.3's
# factor-volatility half-life grid, read off the half-life sweep's own committed
# cache (reports/kt_book_sweep.md; about eight minutes; nine strategy-config rows
# registered in experiments.md first, and the runner refuses to solve until they
# are counted). These are the macro model's optimizer-book points on the K/T chart.
	$(PY) -m mafrm.backtest.halflife_band

## SPEC.md 9 and 12's SINGLE evaluation of the held-out window (W8-P2).
##
## NOT part of `make backtest`, and deliberately not wired into any other
## target: it is the one thing in this repository that reads data on or after
## sample.holdout_start, and CLAUDE.md invariant 5 makes that a visible choice
## rather than a step in a pipeline. It re-runs the frozen cell over the
## IN-SAMPLE window first and refuses to proceed unless it reproduces W6-P2b's
## published reference cell; then it moves the boundary once, walks the frozen
## configuration forward over 2025-01-01..the right edge, scores SPEC.md 6.2's
## four families and the equity control, and writes reports/holdout.md,
## reports/holdout_per_year.csv, reports/holdout_monthly.csv and the `holdout`
## section of results/metrics.json. About twenty minutes; deterministic.
##
## The window is spent. Re-running reproduces the same figures bit for bit --
## it does NOT buy another look, and running it against a second configuration
## is selection on the holdout. experiments.md row 334 is the record.
holdout:
	$(PY) -m mafrm.backtest.holdout

## SPEC.md 6.6's overfitting controls: PBO via CSCV and the false-strategy
## bracket (W8-P2b). IN-SAMPLE ONLY -- they describe the search, and the runner
## asserts the performance matrix stops before sample.holdout_start and refuses
## to run inside the holdout crossing at all. Re-solves SPEC.md 9's 28 cells to
## recover their monthly return series (the grid saved only summary statistics),
## which evaluates no new configuration and moves no count. About two minutes.
overfitting:
	$(PY) -m mafrm.backtest.overfitting

## Regenerate reports/ and results/metrics.json.
## Reads the local cache only -- no network. The spread figure needs data/raw,
## so this target fails cleanly on a clean clone until `make data` has run.
##
## `vra_report` READS the committed reports/vra_forecast_history.csv rather than
## rebuilding it: SPEC.md 5.4 standardises by the forecast made at t-1, so the
## history costs one full pipeline run -- including SPEC.md 5.3's Monte Carlo --
## per date. Rebuild it deliberately when a risk-config value changes --
##     python -m mafrm.factors.vra_report --rebuild --all
## which fans the six variants across processes and resumes from any partial
## already in data/processed/vra/. tests/test_regime.py fails until you do.
report:
	$(PY) -m mafrm.costs.spread_report
# SPEC.md 7.2's calibration (W5-P1): both square-root prefactors derived from
# their published anchors, the SPEC.md 7.3 scaling laws, and each asset's EDGE
# baseline for the spread construction. Arithmetic plus one pass over the
# cached bars; seconds.
	$(PY) -m mafrm.costs.calibration_report
# SPEC.md 10.4's capacity machinery (W5-P2) in NORMALISED units: no project input
# reaches it and no number is published. Arithmetic on the two Y regimes and the
# exponent; W6 redraws the same figure in dollars from the trade series.
	$(PY) -m mafrm.costs.capacity_report
# SPEC.md 7.4's engine reconciliation against bt (W5-P3): the engine and bt on
# the synthetic panel and on the real cache (thirteen cost tickers, in-sample
# only), the comparison table with the c*eps*T bound, and a non-zero exit if
# any row is not as registered in experiments.md rows 199-206. Needs the dev
# extra (bt) and data/raw; about a minute.
	$(PY) -m mafrm.backtest.reconciliation_report
	$(PY) -m mafrm.factors.factor_report
	$(PY) -m mafrm.factors.exposure_report
	$(PY) -m mafrm.factors.validation_report
	$(PY) -m mafrm.factors.mean_convention_report
	$(PY) -m mafrm.factors.psd_repair_report
	$(PY) -m mafrm.factors.eigenfactor_report
	$(PY) -m mafrm.factors.vra_report
# Model B (SPEC.md 4.2). Renders from the COMMITTED forecast-history cache;
# rebuilding that cache is `python -m mafrm.factors.statistical_report --rebuild`
# and takes about ninety minutes, so it is deliberately not part of this target.
	$(PY) -m mafrm.factors.statistical_report
# The hybrid (SPEC.md 4.3), experiments.md row 83, and SPEC.md 15.2's acceptance
# of the hybrid factor set by the unchanged covariance pipeline. Renders from the
# COMMITTED forecast-history cache; rebuilding that cache is
# `python -m mafrm.factors.hybrid_report --rebuild` and takes about half an hour,
# so it is deliberately not part of this target.
	$(PY) -m mafrm.factors.hybrid_report
# SPEC.md 5.5's specific risk and its residual-correlation report. NO CACHE and
# none needed: the specific forecast history costs about nine seconds a variant,
# because SPEC.md 5.5 has no Monte Carlo in it. Rebuilt every run, so there is no
# staleness hazard to manage here.
	$(PY) -m mafrm.factors.specific_report
# SPEC.md 6.1 and 6.2's validation battery -- the rolling bias-statistic chart
# and the family-2 vs family-4 table. Renders from the COMMITTED cache of
# SPEC.md 5.3's eigenfactor-adjusted matrices; rebuilding that cache is
# `python -m mafrm.factors.bias_report --rebuild` and takes about a quarter of
# an hour, so it is deliberately not part of this target. Everything else in the
# battery -- the cheap covariance stages, SPEC.md 5.5's specific risk, the four
# portfolio families and the statistic itself -- is rebuilt on every run.
	$(PY) -m mafrm.factors.bias_report
# SPEC.md 5.1's half-life sensitivity sweep (W4-P2b), registered in SPEC.md 5.1.3.
# Renders from its OWN committed cache of nine eigenfactor histories, one per grid
# point; rebuilding that cache is `python -m mafrm.factors.halflife_report
# --rebuild` and fans nine processes out for about a quarter of an hour, so it is
# deliberately not part of this target. The render itself scores eighteen runs --
# nine half-lives under both horizon shapes -- off those nine files, because the
# cached history depends on the volatility half-life alone (SPEC.md 5.1.3).
	$(PY) -m mafrm.factors.halflife_report
# SPEC.md 6.4's second-order risk and the three-way disjointness accounting
# (W4-P3). A Monte Carlo on the pipeline's own estimators, seconds per horizon;
# no cache. Registered comparisons are printed with their verdicts.
	$(PY) -m mafrm.factors.second_order_report
# SPEC.md 6.5's battery (W4-P3): Mincer-Zarnowitz/QLIKE, Ljung-Box, Kupiec,
# Christoffersen, Basel, Acerbi-Szekely, the rank correlation and the factor
# t-statistics, on every covariance variant's family-4 series. Re-runs the bias
# battery's cheap half off the same committed eigenfactor cache (about a minute)
# and adds the Acerbi-Szekely null simulation on top.
	$(PY) -m mafrm.factors.battery_report
# SPEC.md 15.3's estimation-universe screen (W7-P1): the reconstructed daily
# membership against the cached bars, drop counts per session by cause, the
# per-year gap table, the loader's per-ticker status and the coverage of
# departed names. Seconds; reads the committed data/reference tables and
# data/raw. Scores experiments.md rows 265-267 and prints each verdict.
	$(PY) -m mafrm.factors.universe_report
# SPEC.md 15.4.1's point-in-time market cap (W7-P2a): the EDGAR share counts
# joined to the cached bars under ruling 1, CIK mapping coverage, the pre-XBRL
# approximation's share, per-session counts of estimation-universe names with
# and without a cap, and the unexplained-jump listing. Seconds; reads
# data/raw and the committed data/reference tables. Scores experiments.md
# rows 269-272 and prints each verdict.
	$(PY) -m mafrm.factors.cap_report
# SPEC.md 15.4 and 15.5's six price-only style exposures (W7-P2): the rolling
# descriptors on complete EWMA windows, cap-centred / equal-scaled
# standardization, the two orthogonalizations, the exposure correlation matrix
# under three weightings with VIFs, the winsorization sensitivity and the
# cap-weighted market against SPY. About twenty seconds; reads data/raw and the
# committed data/reference tables. Scores experiments.md rows 278-281.
	$(PY) -m mafrm.factors.equity_report
# SPEC.md 15.6's daily cross-sectional regression (W7-P3): FF49 industries from
# EDGAR SIC codes, sqrt-cap WLS under the cap-weighted industry sum-to-zero
# constraint, the two constraint tests on every date, the daily R^2 series
# against SPEC.md 15.7's 39%, per-factor |t| > 2 frequencies, condition
# numbers, thin-industry counts and the SIC drift table. About a minute; reads
# data/raw and the committed data/reference tables. Scores experiments.md rows
# 283-289 and prints each verdict.
	$(PY) -m mafrm.factors.equity_regression_report
# SPEC.md 15.2's test and 15.7's target (W7-P4): the equity (X, f, u) through
# the UNCHANGED risk/ pipeline on the month-end grid, SPEC.md 6.2's four
# families daily-held, SPEC.md 6.5's battery, 10.3's component assertion.
# SPEC.md 5.3's Monte Carlo per horizon is a digest-guarded partial under
# data/processed/equity_bias/ (gitignored; about ten minutes fanned out across
# the two horizons on a clean clone, reused afterwards); rebuild it deliberately
# with `python -m mafrm.factors.equity_risk_report --rebuild`. The render itself
# is about five minutes. Scores experiments.md rows 291-297 and prints each verdict.
	$(PY) -m mafrm.factors.equity_risk_report
# SPEC.md 15.1 and 12's week-8 headline (W8-P1): measured optimizer-portfolio
# bias against K/T_eff across both models and every half-life of the sweep,
# Shepard's curve overlaid in both units, the tracking test per model, row 164's
# C1 on the equity panel and the non-overlapping sub-period split. Reads the
# half-life cache, the equity partial and reports/kt_book_sweep.csv (drawn
# without the book cluster if the sweep has not run); rebuilds the equity model
# at nine half-lives with no Monte Carlo. About a quarter of an hour.
	$(PY) -m mafrm.factors.kt_report

## ruff + mypy + pytest. Nothing is committed unless this is green.
test: lint typecheck unit

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

## CLAUDE.md: mypy on src/, not on tests or notebooks. Paths come from pyproject.
typecheck:
	$(RUN) mypy

## Network is blocked for unmarked tests by the autouse guard in tests/conftest.py.
## `dataset` is deselected too: those tests read data/raw, which is gitignored and
## absent on a clean clone. data-contracts.yml runs them weekly after `make data`.
unit:
	$(RUN) pytest -m "not network and not dataset" $(PYTEST_OK)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
