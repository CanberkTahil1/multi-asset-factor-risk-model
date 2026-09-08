"""SPEC.md 6.6's remaining overfitting controls: PBO via CSCV, and the false-strategy bracket.

W8-P2b. **In-sample only**, and asserted so: these describe the *search*, and the
search happened strictly before ``sample.holdout_start``. Running them across the
holdout would make the window part of the thing being audited.

WHY THIS IS WRITTEN HERE RATHER THAN TAKEN FROM ``skfolio``
    SPEC.md 6.6 names ``skfolio`` for this, on the grounds that *"the
    combinatorial purged machinery ... is genuinely hard to write yourself"*.
    That is true of **purged** combinatorial CV -- the version with purging and a
    one-sided embargo, needed when a model is *refitted inside each fold* and
    training labels can overlap test ones in time. **Nothing is refitted here.**
    The input is a fixed ``T x N`` matrix of per-period returns from trials that
    have already been run, so there is no leakage channel for purging to close
    and no embargo to set: CSCV reduces to partitioning rows, ranking columns and
    counting. That is the thirty lines below, with a hand-computed test.

    The deviation is recorded rather than taken silently. Adding ``skfolio`` at
    the end of the project would relock the environment (it pulls
    ``scikit-learn`` and a plotting stack) immediately before the repository is
    made public, which is a larger risk to a *working* project -- W1-P5's rebuild
    control is the reason that distinction is taken seriously here -- than a
    self-contained implementation of a published, fully specified procedure.
    SPEC.md 6.6's purged-K-fold line still stands for anything that IS fitted.

WHAT CSCV ANSWERS, AND WHAT IT DOES NOT
    PBO is *"given that I picked the best of `N` configurations in sample, how
    often does that pick land below median out of sample?"* It is a property of
    the **search**, not of the winner: a PBO near 0.5 says selection carried no
    information. It says nothing about whether the strategy makes money, and on a
    grid whose cells share one alpha it is largely a statement about how little
    the cells differ. That caveat is reported beside the number.

Bailey, Borwein, Lopez de Prado & Zhu (2016), *The Probability of Backtest
Overfitting*, Journal of Computational Finance 20(4):39-69.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mafrm import config as config_mod
from mafrm.backtest import grid as grid_mod
from mafrm.backtest import metrics as metrics_mod
from mafrm.backtest import verification
from mafrm.data import holdout as holdout_mod

__all__ = [
    "CSCVResult",
    "OverfittingError",
    "cscv",
    "main",
    "performance_matrix",
    "render",
    "run",
]

FloatArray = NDArray[np.float64]

_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_REPORTS: Final[Path] = _ROOT / "reports"
_RESULTS: Final[Path] = _ROOT / "results"
_EXPERIMENTS: Final[Path] = _ROOT / "experiments.md"

#: ``S`` in Bailey et al.: the number of equal contiguous blocks the sample is
#: cut into, giving ``C(S, S/2)`` splits. 16 -> 12,870, the value the paper uses
#: throughout and the one it recommends as "large enough that the result is
#: stable, small enough to enumerate". Must be even.
BLOCKS: Final[int] = 16


class OverfittingError(ValueError):
    """The inputs cannot carry the statistic asked of them. Nothing is defaulted."""


@dataclass(frozen=True)
class CSCVResult:
    """One CSCV run over a ``T x N`` performance matrix."""

    #: ``P[logit <= 0]``: how often the in-sample winner lands below the OOS median.
    pbo: float
    #: One relative rank per split, in ``(0, 1)``.
    relative_ranks: FloatArray
    logits: FloatArray
    #: Per split: the in-sample winner's IS and OOS Sharpe.
    is_sharpe: FloatArray
    oos_sharpe: FloatArray
    trials: int
    periods: int
    blocks: int
    splits: int
    #: Rows dropped so the blocks divide evenly; reported, never silently absorbed.
    dropped_periods: int
    #: OLS slope of the winner's OOS Sharpe on its IS Sharpe. Bailey et al.'s
    #: "performance degradation": a negative slope means a better in-sample pick
    #: does worse out of sample.
    degradation_slope: float
    #: Share of splits on which the winner's OOS Sharpe is below zero.
    probability_of_loss: float


def _sharpe(block: FloatArray) -> FloatArray:
    """Per-period Sharpe of each column. No annualisation: ranks are scale-free."""
    sd = np.std(block, axis=0, ddof=1)
    if np.any(sd <= 0.0):
        raise OverfittingError("a trial has zero variance on a split; its rank is undefined")
    return np.asarray(np.mean(block, axis=0) / sd, dtype=float)


def cscv(returns: FloatArray, *, blocks: int = BLOCKS) -> CSCVResult:
    """Combinatorially symmetric cross-validation. Bailey et al. (2016) §3.

    ``returns`` is ``T x N`` -- one column per trial, one row per period, on a
    common calendar. The sample is cut into ``S`` equal contiguous blocks; every
    one of the ``C(S, S/2)`` ways of choosing half of them forms an in-sample set
    with its complement as out-of-sample. The trial with the best IS Sharpe is
    selected, its **relative rank** among the OOS Sharpes is taken, and
    ``PBO = P[logit(rank) <= 0]`` -- the frequency with which the in-sample
    winner is a below-median performer out of sample.

    Blocks are **contiguous** rather than interleaved, which is the paper's own
    construction and the one that respects serial dependence in the return
    series; the trials share a calendar, so a split is the same partition for
    every column and the comparison is like-for-like.
    """
    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2:
        raise OverfittingError(f"returns must be T x N, got shape {matrix.shape}")
    periods, trials = matrix.shape
    if trials < 2:
        raise OverfittingError(f"CSCV needs at least two trials, got {trials}")
    if blocks < 2 or blocks % 2 != 0:
        raise OverfittingError(f"blocks must be even and at least 2, got {blocks}")
    if periods < 2 * blocks:
        raise OverfittingError(
            f"{periods} periods cannot be cut into {blocks} blocks with two rows each"
        )
    if not np.all(np.isfinite(matrix)):
        raise OverfittingError("returns carries a non-finite entry")

    size = periods // blocks
    dropped = periods - size * blocks
    # Drop from the FRONT: the tail is the most recent data and the least
    # arbitrary thing to keep whole. Reported either way.
    trimmed = matrix[dropped:]
    parts = [trimmed[i * size : (i + 1) * size] for i in range(blocks)]

    ranks: list[float] = []
    logits: list[float] = []
    is_best: list[float] = []
    oos_best: list[float] = []
    half = blocks // 2
    for chosen in combinations(range(blocks), half):
        rest = [i for i in range(blocks) if i not in set(chosen)]
        inside = _sharpe(np.concatenate([parts[i] for i in chosen]))
        outside = _sharpe(np.concatenate([parts[i] for i in rest]))
        winner = int(np.argmax(inside))
        # Rank of the winner among the OOS Sharpes, 1 = worst .. N = best, with
        # ties averaged so a tie cannot be resolved in the winner's favour.
        order = pd.Series(outside).rank(method="average").to_numpy()
        relative = float(order[winner] / (trials + 1))
        ranks.append(relative)
        logits.append(math.log(relative / (1.0 - relative)))
        is_best.append(float(inside[winner]))
        oos_best.append(float(outside[winner]))

    logit_array = np.asarray(logits, dtype=float)
    is_array = np.asarray(is_best, dtype=float)
    oos_array = np.asarray(oos_best, dtype=float)
    slope = float(np.polyfit(is_array, oos_array, 1)[0]) if np.std(is_array) > 0.0 else float("nan")
    return CSCVResult(
        pbo=float(np.mean(logit_array <= 0.0)),
        relative_ranks=np.asarray(ranks, dtype=float),
        logits=logit_array,
        is_sharpe=is_array,
        oos_sharpe=oos_array,
        trials=trials,
        periods=periods,
        blocks=blocks,
        splits=len(ranks),
        dropped_periods=dropped,
        degradation_slope=slope,
        probability_of_loss=float(np.mean(oos_array < 0.0)),
    )


# ---------------------------------------------------------------------------
# The matrix, built from the in-sample grid
# ---------------------------------------------------------------------------


def performance_matrix(
    settings: config_mod.Config | None = None, *, progress: bool = True
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """SPEC.md 9's cells re-solved in sample, as a ``T x N`` frame of monthly net returns.

    **These are the same configurations already counted as rows 211-238**; the
    grid is re-run only because its per-cell return *series* were never saved,
    and re-running an already-counted configuration adds no row (experiments.md
    rule 3: one row per configuration). Nothing is swept and nothing is chosen.

    Raises if any date reaches the holdout boundary -- ``run_grid`` already
    asserts it, and this asserts it again on the assembled frame, because the
    claim "in-sample only" is the whole point of this module.
    """
    settings = settings or config_mod.load()
    if holdout_mod.is_evaluating_holdout():
        raise OverfittingError(
            "the overfitting controls describe the SEARCH, which happened strictly before the "
            "boundary; running them inside the crossing would audit the holdout with itself"
        )
    boundary = pd.Timestamp(settings.require_holdout_start())
    universe = verification.build_universe(settings, horizon=settings.model.optimizer.grid.horizon)
    # SPEC.md 9's 28 CELLS, not its 35 runs. Ruling 4 (SPEC.md 9.1) says `Y` is an
    # axis-2 ingredient -- "treatment D runs both regimes inside its one cell" --
    # so the urgent-regime runs are the same configuration priced at a different
    # prefactor, not separate trials. `run_grid` takes the patient one for exactly
    # this reason when it computes `V[SR]`, and matching it here is what makes the
    # bracket below comparable to the one `reports/holdout.md` uses.
    columns: dict[str, pd.Series] = {}
    for spec in [c for c in grid_mod.cells(settings) if c.regime == "patient"]:
        if progress:
            print(f"  {spec.cell_id} {spec.label}", flush=True)
        result = grid_mod.run_cell(grid_mod.cell_inputs(universe, settings, spec), settings)
        columns[spec.label] = result.monthly["net_return"]
    frame = pd.DataFrame(columns)
    if frame.isna().to_numpy().any():
        raise OverfittingError("the cells do not share one calendar")
    last = pd.Timestamp(pd.DatetimeIndex(frame.index).max())
    if last >= boundary:
        raise OverfittingError(
            f"the performance matrix reaches {last.date()}, on or after HOLDOUT_START "
            f"{boundary.date()}. These controls are in-sample only (CLAUDE.md invariant 5)."
        )
    return frame, last


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render(
    result: CSCVResult,
    frame: pd.DataFrame,
    *,
    trials_n: int,
    trials_variance: float,
    bracket: float,
    last: pd.Timestamp,
) -> str:
    """``reports/overfitting.md``."""
    index = pd.DatetimeIndex(frame.index)
    per_period = np.asarray(
        [metrics_mod.sharpe_ratio(frame[c].to_numpy(dtype=float)) for c in frame.columns]
    )
    best = int(np.argmax(per_period))
    out: list[str] = []
    out.append("# Overfitting controls: PBO via CSCV, and the false-strategy bracket")
    out.append("")
    out.append(
        "SPEC.md 6.6's two remaining controls. **In-sample only** -- they describe the "
        f"*search*, which happened strictly before `HOLDOUT_START`. The matrix ends "
        f"**{last.date()}** and the runner raises if it reaches the boundary. `experiments.md` "
        "W8-P2b."
    )
    out.append("")
    out.append("## The performance matrix")
    out.append("")
    out.append(
        f"SPEC.md 9's **{result.trials} cells**, re-solved in sample, as monthly net returns "
        f"over **{result.periods} months** ({index.min().date()} to {index.max().date()}). These "
        "are the configurations already counted as rows 211-238; the grid is re-run only "
        "because its per-cell return *series* were never saved, and re-running an "
        "already-counted configuration adds no row. **Nothing was swept and nothing chosen.** "
        "The patient regime of treatment D is taken, per SPEC.md 9.1 ruling 4 -- `Y` is an "
        "ingredient inside one cell, not a separate trial -- which is also the convention "
        "`run_grid` uses for `V[SR]`, so the bracket below is comparable to the holdout's."
    )
    out.append("")
    out.append(
        f"Best cell by per-period Sharpe: `{frame.columns[best]}` at {per_period[best]:.4f}; "
        f"the spread across cells is {per_period.min():.4f} to {per_period.max():.4f}, "
        f"`V[SR]` = {float(np.var(per_period, ddof=1)):.3e}."
    )
    out.append("")
    out.append("## PBO via CSCV")
    out.append("")
    out.append(
        f"`S` = {result.blocks} contiguous blocks, so **C({result.blocks}, {result.blocks // 2}) "
        f"= {result.splits:,} splits**, each pairing an in-sample half with its complement. "
        f"{result.dropped_periods} row(s) dropped from the front so the blocks divide evenly."
    )
    out.append("")
    out.append("| | Value |")
    out.append("|---|---|")
    out.append(f"| **PBO** = `P[logit <= 0]` | **{result.pbo:.4f}** |")
    out.append(
        f"| Median relative rank of the IS winner | {np.median(result.relative_ranks):.4f} |"
    )
    out.append(f"| Median logit | {np.median(result.logits):+.4f} |")
    out.append(f"| Performance degradation (OOS on IS slope) | {result.degradation_slope:+.4f} |")
    out.append(
        f"| Probability of loss (winner's OOS Sharpe < 0) | {result.probability_of_loss:.4f} |"
    )
    out.append(f"| Splits | {result.splits:,} |")
    out.append("")
    out.append("## The false-strategy bracket")
    out.append("")
    out.append(
        f"`E[max SR_N]` under zero skill at **`N` = {trials_n}** and "
        f"`V[SR]` = {trials_variance:.3e} per period is **{bracket:.4f}** per period. The best "
        f"cell's own per-period Sharpe is {per_period[best]:.4f}, which is "
        f"{per_period[best] / bracket:.1f}x the bracket."
    )
    out.append("")
    out.append("## What these two numbers do and do not say")
    out.append("")
    out.append(
        "**PBO is a property of the search, not of the winner.** It asks how often the "
        "in-sample best lands below the out-of-sample median. Read it beside the fact that "
        f"**these {result.trials} cells share one alpha** and differ only in covariance "
        "treatment and cost treatment: their Sharpes sit in a narrow band, so which one ranks "
        "top is close to a coin flip by construction, and a PBO near 0.5 is what a search "
        "that discriminates nothing looks like. That was the registered expectation, and the "
        "result below is read against it. It is the same caveat "
        "`reports/experiment_grid.md` and `reports/holdout.md` carry about `V[SR]`."
    )
    out.append("")
    out.append(
        "**Neither control was ever going to condemn or clear this project's headline.** The "
        "quantity the project reports is not a selected Sharpe -- nothing was selected on "
        "backtest performance, which is why `N` counts trials that were never chosen between "
        "-- it is the decomposition of a gap. These are reported because SPEC.md 6.6 asks for "
        "them and because a search of 70 configurations owes its reader the number, not "
        "because a verdict hangs on them."
    )
    out.append("")
    return "\n".join(out) + "\n"


def run(settings: config_mod.Config | None = None, *, progress: bool = True) -> None:
    """Build the matrix, run both controls, write the report and the metrics section."""
    settings = settings or config_mod.load()
    frame, last = performance_matrix(settings, progress=progress)
    result = cscv(frame.to_numpy(dtype=float))

    trials_n = metrics_mod.read_trial_count(_EXPERIMENTS)
    per_period = np.asarray(
        [metrics_mod.sharpe_ratio(frame[c].to_numpy(dtype=float)) for c in frame.columns]
    )
    trials_variance = float(np.var(per_period, ddof=1))
    bracket = metrics_mod.expected_maximum_sharpe(trials_variance=trials_variance, trials=trials_n)

    _REPORTS.mkdir(parents=True, exist_ok=True)
    (_REPORTS / "overfitting.md").write_text(
        render(
            result,
            frame,
            trials_n=trials_n,
            trials_variance=trials_variance,
            bracket=bracket,
            last=last,
        ),
        encoding="utf-8",
    )
    frame.to_csv(_REPORTS / "overfitting_matrix.csv")

    path = _RESULTS / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    metrics["overfitting"] = {
        "task": "W8-P2b",
        "in_sample_only": True,
        "matrix_last_date": str(last.date()),
        "trials": result.trials,
        "periods": result.periods,
        "blocks": result.blocks,
        "splits": result.splits,
        "dropped_periods": result.dropped_periods,
        "pbo": result.pbo,
        "median_relative_rank": float(np.median(result.relative_ranks)),
        "median_logit": float(np.median(result.logits)),
        "degradation_slope": result.degradation_slope,
        "probability_of_loss": result.probability_of_loss,
        "trial_count_N": trials_n,
        "trials_variance_per_period": trials_variance,
        "false_strategy_bracket": bracket,
        "best_cell_sharpe_per_period": float(per_period.max()),
    }
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print((_REPORTS / "overfitting.md").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - entry point
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
