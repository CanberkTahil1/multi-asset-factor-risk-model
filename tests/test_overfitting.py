"""SPEC.md 6.6's PBO/CSCV, with a hand-computed case. W8-P2b."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from mafrm.backtest.overfitting import BLOCKS, CSCVResult, OverfittingError, cscv


def test_cscv_hand_computed_on_a_two_trial_four_block_case() -> None:
    """Four blocks, two trials, every one of the six splits worked by hand.

    Two columns, eight rows, cut into ``S`` = 4 blocks of two rows. Trial A wins
    in sample on the first half of the sample and loses on the second; trial B is
    the mirror. With ``C(4, 2)`` = 6 splits and ties averaged, the arithmetic is
    small enough to write out.

    Blocks (rows are per-period returns):

        block 0: A = (+3, +3)  B = (-1, -1)
        block 1: A = (+3, +3)  B = (-1, -1)
        block 2: A = (-1, -1)  B = (+3, +3)
        block 3: A = (-1, -1)  B = (+3, +3)

    Every block has zero within-block variance, so a Sharpe is only defined on a
    union of two blocks that differ; the six splits are the six unordered pairs:

        IS {0,1} -> A mean +3, B mean -1; A wins.  OOS {2,3}: A -1, B +3.
            But {0,1} has zero variance in both columns -> refused, see below.

    To keep every split well defined the values alternate WITHIN each block, so
    each block is (+x, -x) shifted: block ``k`` for trial A is ``(a_k + 1, a_k - 1)``.
    Then the standard deviation on any union is positive and the mean is the mean
    of the block levels. Levels:

        A: (+3, +3, -1, -1)      B: (-1, -1, +3, +3)

    On a split with in-sample blocks ``I`` the mean of A is ``mean(a_k for k in I)``
    and its sd is constant across splits and equal for both columns (each block
    contributes the same +-1 wiggle), so **ranking by Sharpe is ranking by mean**.

        IS {0,1}: A +3.0 B -1.0 -> A wins.  OOS {2,3}: A -1.0 B +3.0 -> A ranks 1 of 2
        IS {0,2}: A +1.0 B +1.0 -> tie, argmax takes A.  OOS {1,3}: A +1.0 B +1.0 -> tie
        IS {0,3}: A +1.0 B +1.0 -> tie, argmax takes A.  OOS {1,2}: A +1.0 B +1.0 -> tie
        IS {1,2}: A +1.0 B +1.0 -> tie, argmax takes A.  OOS {0,3}: A +1.0 B +1.0 -> tie
        IS {1,3}: A +1.0 B +1.0 -> tie, argmax takes A.  OOS {0,2}: A +1.0 B +1.0 -> tie
        IS {2,3}: A -1.0 B +3.0 -> B wins.  OOS {0,1}: A +3.0 B -1.0 -> B ranks 1 of 2

    Relative rank is ``rank / (N + 1)`` with N = 2, ties averaged at rank 1.5:

        split 1: winner ranks 1   -> 1/3   = 0.3333 -> logit ln(0.5)  = -0.6931 <= 0
        splits 2-5: ties, rank 1.5 -> 1.5/3 = 0.5000 -> logit ln(1.0)  =  0.0000 <= 0
        split 6: winner ranks 1   -> 1/3   = 0.3333 -> logit -0.6931          <= 0

    All six logits are <= 0, so **PBO = 6/6 = 1.0** -- the in-sample winner is
    never an above-median performer out of sample, which is exactly right for a
    pair of trials constructed to swap places. The medians follow: relative rank
    0.5, logit 0.0.
    """
    levels = {"A": [3.0, 3.0, -1.0, -1.0], "B": [-1.0, -1.0, 3.0, 3.0]}
    rows = []
    for k in range(4):
        rows.append([levels["A"][k] + 1.0, levels["B"][k] + 1.0])
        rows.append([levels["A"][k] - 1.0, levels["B"][k] - 1.0])
    matrix = np.asarray(rows, dtype=float)
    assert matrix.shape == (8, 2)

    result = cscv(matrix, blocks=4)

    assert result.splits == len(list(combinations(range(4), 2))) == 6
    assert result.trials == 2
    assert result.periods == 8
    assert result.dropped_periods == 0
    assert result.pbo == pytest.approx(1.0)
    assert float(np.median(result.relative_ranks)) == pytest.approx(0.5)
    assert float(np.median(result.logits)) == pytest.approx(0.0)
    # The two decisive splits carry the 1/3 rank and its logit; four are ties.
    assert sorted(np.round(result.relative_ranks, 4)) == pytest.approx(
        [1 / 3, 1 / 3, 0.5, 0.5, 0.5, 0.5], abs=1e-4
    )
    assert min(result.logits) == pytest.approx(math.log(0.5), abs=1e-12)


def test_an_independent_search_lands_near_one_half() -> None:
    """Pure noise: selection carries no information, so PBO should sit near 0.5.

    Not a threshold on the project's own number -- a control that the estimator
    is not biased toward 0 or 1 by construction.
    """
    rng = np.random.default_rng(20260825)
    matrix = rng.standard_normal((192, 20))
    result = cscv(matrix, blocks=BLOCKS)
    assert 0.35 < result.pbo < 0.65
    assert result.splits == 12870


def test_a_genuinely_better_trial_is_not_called_overfitted() -> None:
    """One column with real edge everywhere: the winner is stable, so PBO is low."""
    rng = np.random.default_rng(20260825)
    matrix = rng.standard_normal((192, 8))
    matrix[:, 3] += 0.8
    result = cscv(matrix, blocks=BLOCKS)
    assert result.pbo < 0.05


def test_the_blocks_must_be_even_and_the_sample_long_enough() -> None:
    rng = np.random.default_rng(1)
    with pytest.raises(OverfittingError, match="even"):
        cscv(rng.standard_normal((64, 4)), blocks=7)
    with pytest.raises(OverfittingError, match="cannot be cut"):
        cscv(rng.standard_normal((10, 4)), blocks=16)
    with pytest.raises(OverfittingError, match="at least two trials"):
        cscv(rng.standard_normal((64, 1)), blocks=4)
    with pytest.raises(OverfittingError, match="non-finite"):
        bad = rng.standard_normal((64, 4))
        bad[3, 2] = np.nan
        cscv(bad, blocks=4)


def test_the_result_is_a_frozen_record() -> None:
    rng = np.random.default_rng(7)
    result = cscv(rng.standard_normal((96, 5)), blocks=8)
    assert isinstance(result, CSCVResult)
    assert result.relative_ranks.size == result.splits
    assert np.all(result.relative_ranks > 0.0) and np.all(result.relative_ranks < 1.0)
    with pytest.raises(AttributeError):
        result.pbo = 0.0  # type: ignore[misc]


def test_dropped_periods_are_reported_not_absorbed() -> None:
    """195 rows into 16 blocks leaves 3; the count must surface."""
    rng = np.random.default_rng(3)
    result = cscv(rng.standard_normal((195, 4)), blocks=16)
    assert result.dropped_periods == 195 - (195 // 16) * 16 == 3
    assert result.periods == 195


def test_the_matrix_builder_refuses_inside_the_crossing() -> None:
    """These controls describe the search, which is in-sample by definition."""
    from datetime import date

    from mafrm import config as config_mod
    from mafrm.backtest import overfitting
    from mafrm.data import holdout as holdout_mod

    with (
        holdout_mod.evaluating_holdout(edge=date(2026, 7, 31), reason="test"),
        pytest.raises(OverfittingError, match="describe the SEARCH"),
    ):
        overfitting.performance_matrix(config_mod.load(), progress=False)


def test_the_report_states_the_shared_alpha_caveat() -> None:
    """A PBO on cells that share one alpha must not be read as a clean verdict."""
    from mafrm.backtest import overfitting

    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        rng.standard_normal((64, 4)),
        index=pd.date_range("2019-01-31", periods=64, freq="ME"),
        columns=[f"cell_{i}" for i in range(4)],
    )
    text = overfitting.render(
        cscv(frame.to_numpy(dtype=float), blocks=8),
        frame,
        trials_n=70,
        trials_variance=8.8e-05,
        bracket=0.0225,
        last=pd.Timestamp("2024-04-30"),
    )
    assert "share one alpha" in text
    assert "property of the search" in text.lower()
    assert "In-sample only" in text
