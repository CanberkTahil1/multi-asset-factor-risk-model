"""SPEC.md 11's mechanical no-look-ahead test, written generic. W5-P3.

    *"randomly perturb all data strictly after t, re-run, assert weights at t
    are bitwise unchanged, across >= 50 values of t."*

The harness takes a callable producing **the state at t** from a data panel and
asserts that state is bitwise invariant to perturbations of every row strictly
after ``t`` (operator ruling 4, 2026-09-03). It does not know what the state
is. This session points it at the covariance forecast, the cost inputs and the
engine's NAV; W6-P1 points it at the optimizer's weights without rewriting it --
invariant-10 discipline applied to a test.

TWO THINGS A NAIVE VERSION GETS WRONG, AND THIS ONE CHECKS
    - **Power.** A state that never changes passes the invariance check for
      free -- with fixed target weights the "weights at t" version is vacuous,
      which is why it is not what this session runs. So the harness also
      perturbs the data **through** ``t`` and requires the state to move at
      every evaluated date; a test that cannot fail is reported as vacuous, not
      as passed.
    - **Bitwise means bitwise.** The comparison is on the float64 byte string,
      not on ``allclose``: an off-by-one-ulp change is a change, because it
      means some later row reached the arithmetic.

The evaluation count comes from ``config/model.yaml``
(``backtest.no_lookahead.minimum_evaluation_dates``, SPEC.md 11's 50) and is
passed in by the caller, so the harness cannot quietly run on fewer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "LookaheadError",
    "LookaheadReport",
    "Perturbation",
    "StateAt",
    "assert_no_lookahead",
    "evaluation_dates",
    "perturb_additive",
    "perturb_multiplicative",
]

#: ``state_at(data, t)`` -- the state at ``t`` computed from ``data``. The
#: callable decides what "through t" means (inclusive for a trailing statistic,
#: exclusive for a forecast made at ``t-1``); the harness perturbs strictly after.
StateAt = Callable[[pd.DataFrame, pd.Timestamp], object]

#: ``perturb(rows, rng)`` -- replacement values for ``rows``, same shape.
Perturbation = Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]


class LookaheadError(AssertionError):
    """The state at ``t`` depended on data after ``t`` -- or could not be shown to depend on
    data through ``t`` at all."""


@dataclass(frozen=True)
class LookaheadReport:
    """What was checked. ``evaluated`` are the dates, in order."""

    evaluated: tuple[pd.Timestamp, ...]
    minimum_required: int
    #: True when the state moved at every date under the through-``t`` perturbation.
    sensitive: bool

    @property
    def count(self) -> int:
        return len(self.evaluated)


def perturb_additive(rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add unit-variance noise scaled per column by that column's own dispersion.

    For returns and other signed data. A column with zero dispersion is
    perturbed by unit noise so that a constant column is still moved.
    """
    values = rows.to_numpy(dtype="float64")
    scale = np.nanstd(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    noise = rng.standard_normal(values.shape) * scale
    return pd.DataFrame(values + noise, index=rows.index, columns=rows.columns)


def perturb_multiplicative(rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Multiply by ``exp(N(0, 1))`` factors -- stays positive, for prices and volumes."""
    values = rows.to_numpy(dtype="float64")
    factors = np.exp(rng.standard_normal(values.shape))
    return pd.DataFrame(values * factors, index=rows.index, columns=rows.columns)


def evaluation_dates(
    index: pd.DatetimeIndex,
    *,
    count: int,
    rng: np.random.Generator,
    minimum_prefix: int,
    minimum_suffix: int = 1,
) -> tuple[pd.Timestamp, ...]:
    """``count`` distinct dates drawn from the interior of ``index``.

    ``minimum_prefix`` rows are kept before every evaluated date so the state
    has something to be computed from (a burn-in the caller states, never a
    default here), and ``minimum_suffix`` rows after it so there is something to
    perturb. Raises rather than drawing fewer.
    """
    if count < 1:
        raise LookaheadError(f"evaluation_dates: count must be >= 1, got {count}")
    if minimum_prefix < 1 or minimum_suffix < 1:
        raise LookaheadError("evaluation_dates: minimum_prefix and minimum_suffix must be >= 1")
    low = minimum_prefix
    high = len(index) - minimum_suffix  # exclusive
    if high - low < count:
        raise LookaheadError(
            f"evaluation_dates: only {max(high - low, 0)} interior positions for {count} dates "
            f"(index length {len(index)}, prefix {minimum_prefix}, suffix {minimum_suffix})"
        )
    positions = np.sort(rng.choice(np.arange(low, high), size=count, replace=False))
    return tuple(pd.Timestamp(index[int(i)]) for i in positions)


def _as_bytes(state: object, *, label: str) -> tuple[tuple[int, ...], bytes]:
    """Shape and the float64 byte string. NaN-safe, because bytes are bytes."""
    if isinstance(state, pd.DataFrame | pd.Series):
        array = np.ascontiguousarray(state.to_numpy(dtype="float64"))
    else:
        try:
            array = np.ascontiguousarray(np.asarray(state, dtype="float64"))
        except (TypeError, ValueError) as exc:
            raise LookaheadError(
                f"{label}: the state must be numeric (array, Series, DataFrame or scalar), got "
                f"{type(state).__name__}"
            ) from exc
    return tuple(array.shape), array.tobytes()


def assert_no_lookahead(
    state_at: StateAt,
    data: pd.DataFrame,
    *,
    times: Sequence[pd.Timestamp],
    minimum_evaluations: int,
    rng: np.random.Generator,
    perturb: Perturbation,
    require_sensitivity: bool = True,
) -> LookaheadReport:
    """Assert ``state_at(data, t)`` is bitwise invariant to perturbing rows after ``t``.

    For each ``t`` in ``times``: compute the baseline state; replace every row
    with index strictly greater than ``t`` by ``perturb(rows, rng)``; recompute;
    require identical bytes. With ``require_sensitivity`` (the default) also
    replace every row with index at most ``t`` and require the bytes to
    **differ** -- the power check. Raises :class:`LookaheadError` naming the
    date on the first failure of either kind; returns a report otherwise.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise LookaheadError("assert_no_lookahead: data must be indexed by DatetimeIndex")
    if not data.index.is_monotonic_increasing or not data.index.is_unique:
        raise LookaheadError("assert_no_lookahead: data index must be increasing and unique")
    stamps = tuple(pd.Timestamp(t) for t in times)
    if len(set(stamps)) < minimum_evaluations:
        raise LookaheadError(
            f"assert_no_lookahead: {len(set(stamps))} distinct evaluation dates against a "
            f"required minimum of {minimum_evaluations} (SPEC.md 11; "
            "config backtest.no_lookahead.minimum_evaluation_dates)"
        )

    sensitive = True
    for t in stamps:
        after = data.index > t
        if not after.any():
            raise LookaheadError(f"{t.date()}: no rows after t, so there is nothing to perturb")
        baseline = _as_bytes(state_at(data, t), label=f"state at {t.date()}")

        future = data.copy()
        future.loc[after, :] = perturb(data.loc[after], rng).to_numpy(dtype="float64")
        shifted = _as_bytes(state_at(future, t), label=f"state at {t.date()} (future perturbed)")
        if shifted != baseline:
            raise LookaheadError(
                f"LOOK-AHEAD at {t.date()}: the state changed when rows strictly after t were "
                "perturbed. Some later row reached the arithmetic."
            )

        if require_sensitivity:
            past = data.copy()
            past.loc[~after, :] = perturb(data.loc[~after], rng).to_numpy(dtype="float64")
            moved = _as_bytes(state_at(past, t), label=f"state at {t.date()} (past perturbed)")
            if moved == baseline:
                sensitive = False
                raise LookaheadError(
                    f"VACUOUS at {t.date()}: the state did not change when every row through t "
                    "was perturbed, so this test could not have detected look-ahead here. "
                    "Point the harness at a state that depends on the data (operator ruling 4)."
                )

    return LookaheadReport(
        evaluated=stamps, minimum_required=minimum_evaluations, sensitive=sensitive
    )
