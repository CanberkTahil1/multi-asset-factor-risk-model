"""The holdout boundary, enforced at the point data enters the process.

CLAUDE.md invariant 5 says never read or evaluate anything on or after
``HOLDOUT_START``. Until W2-P3 that was enforced by *discipline plus downstream
assertions*: every shipped panel passed ``Config.require_holdout_start()`` into
its construction and tests asserted the panels ended before the boundary.

**That control was insufficient and it failed.** During W2-P3 a scoping read
computed a SPY/``Mkt-RF`` correlation over 2025-01..2026-06 -- a window on the
far side of the boundary -- because the cache read that produced it was an
ordinary call returning full history, and a test on the shipped path has no
purchase on an ad-hoc one. The read reached no config, report, test or gate and
its value was discarded, but *that it was harmless is not the point*: a bright
line exists precisely to refuse the argument that a particular crossing did no
damage. See ``experiments.md``, W2-P3.

So the guard moved from discipline to the loader. :func:`truncate` is applied by
:func:`mafrm.data.cache.read`, which every cached artefact in this project is
read through, so **the default behaviour of a normal call is to stop at the
boundary**. Reaching past it is still legitimate -- manifest coverage, release
drift and the data-layer cross-checks all have to see what a source actually
published -- but it now requires :func:`mafrm.data.cache.read_unrestricted` and a
stated ``reason``, which makes it a visible choice at the call site rather than
the silent default.

**Why this lives here and not in ``loaders.py``.** ``loaders.py`` is the natural
home for a boundary policy and is where the instruction placed it, but it
imports ``cache`` and every per-source reader, so any of them importing it back
would be a cycle at module import time. This module imports ``config`` and
nothing else in the project, which lets the readers depend on it. ``loaders.py``
documents the boundary and points here.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, timedelta
from typing import Final, TypeVar

import pandas as pd

from mafrm import config as config_mod
from mafrm.config import Config, load

__all__ = [
    "HoldoutError",
    "boundary",
    "evaluating_holdout",
    "is_evaluating_holdout",
    "is_truncated",
    "truncate",
]

Dated = TypeVar("Dated", pd.DataFrame, pd.Series)


class HoldoutError(ValueError):
    """A cached object cannot be placed relative to the holdout boundary."""


def boundary(config: Config | None = None) -> date:
    """``model.sample.holdout_start``. Raises while it is unpinned."""
    return (config or load()).require_holdout_start()


# ---------------------------------------------------------------------------
# The single sanctioned crossing -- W8-P2, and nothing else
# ---------------------------------------------------------------------------

#: True only inside :func:`evaluating_holdout`. Read by tests and by the runner's
#: own assertions; never a way to ask for a wider read.
_EVALUATING: bool = False

#: Recorded so the runner can print WHY the boundary moved, in the report and in
#: the ``experiments.md`` row. Cleared on exit with everything else.
_REASON: str | None = None

_W8_REASON: Final[str] = (
    "SPEC.md 9 and 12: the ONE evaluation of the held-out window, W8-P2. "
    "The grid was frozen first and nothing is tuned after."
)


def is_evaluating_holdout() -> bool:
    """Whether the process is inside the one sanctioned crossing."""
    return _EVALUATING


@contextmanager
def evaluating_holdout(*, edge: date, reason: str) -> Iterator[date]:
    """Move the boundary past ``edge`` for the duration. **Spends the holdout.**

    CLAUDE.md invariant 5 says never read or evaluate anything on or after
    ``HOLDOUT_START``, and every other path in this project obeys it: normal
    reads truncate (:func:`truncate`), and a source-level question uses
    :func:`mafrm.data.cache.read_unrestricted`, which sees the full history but
    reaches no model. **Neither of those can run a backtest over 2025-2026**, and
    SPEC.md 9 and 12 require exactly one that does: *"the holdout has been run
    once"*.

    So this is that one, and it is deliberately shaped to be spent rather than
    used. It moves the boundary GLOBALLY -- ``config.load`` is
    :func:`functools.lru_cache`-d and both :func:`boundary` and every
    ``settings.require_holdout_start()`` read through it, so the two mechanisms
    move together and cannot disagree -- for the duration of the block, and
    restores it afterwards with an assertion that the restore happened.

    ``edge`` is the last date the evaluation may see: the boundary is set to the
    day AFTER it, because :func:`truncate` and :func:`mafrm.data.calendar.align`
    are both half-open. The operator's ruling (2026-09-06, W8-P2) fixes ``edge``
    as **the minimum over the book's series of their last observation**, so the
    book is marked on a complete cross-section and a ragged tail cannot silently
    change the universe.

    **What this is not.** It is not a way to look at recent numbers, not a
    development tool, and not re-runnable against a second configuration: the
    window is spent the first time it is read, and re-running the grid against it
    to pick a winner is selection on the holdout. There is exactly one caller,
    :mod:`mafrm.backtest.holdout`, and a test pins that.
    """
    global _EVALUATING, _REASON
    if not reason.strip():
        raise HoldoutError(
            "evaluating_holdout requires a stated reason. Spending the holdout is a visible "
            "choice, not a default (CLAUDE.md invariant 5)."
        )
    if _EVALUATING:
        raise HoldoutError(
            "evaluating_holdout is already active. The holdout is evaluated ONCE; a nested "
            "or repeated crossing is the thing this contextmanager exists to make visible."
        )
    original = load()
    pinned = original.require_holdout_start()
    if edge < pinned:
        raise HoldoutError(
            f"evaluating_holdout: edge {edge} is before HOLDOUT_START {pinned}, so there is "
            "no holdout to evaluate and the crossing buys nothing."
        )
    moved = dataclasses.replace(
        original,
        model=dataclasses.replace(
            original.model,
            sample=dataclasses.replace(
                original.model.sample, holdout_start=edge + timedelta(days=1)
            ),
        ),
    )
    _EVALUATING, _REASON = True, reason
    config_mod._OVERRIDE = moved
    try:
        seated = load().require_holdout_start()
        if seated != moved.model.sample.holdout_start:
            raise HoldoutError(
                f"evaluating_holdout: the boundary did not move ({seated}); the crossing "
                "would have run against the in-sample window and reported it as the holdout."
            )
        yield edge
    finally:
        _EVALUATING, _REASON = False, None
        config_mod._OVERRIDE = None
        restored = load().require_holdout_start()
        if restored != pinned:
            raise HoldoutError(
                f"evaluating_holdout: the boundary did not restore ({restored} != {pinned}). "
                "The process is no longer safe to read anything in."
            )


def truncate(obj: Dated, *, label: str, config: Config | None = None) -> Dated:
    """Drop every observation on or after the holdout boundary.

    **Half-open**, matching :func:`mafrm.data.calendar.align`: the holdout begins
    *on* ``holdout_start``, so the boundary date itself is excluded.

    Raises rather than passing an object through unfiltered. A cached artefact
    this project cannot date is one it cannot guarantee anything about, and
    silently returning it is the failure this module was written after.
    """
    index = obj.index
    if not isinstance(index, pd.DatetimeIndex):
        raise HoldoutError(
            f"{label}: index is {type(index).__name__}, not a DatetimeIndex, so the holdout "
            "boundary cannot be applied. Every artefact in this project's cache is dated; if a "
            "genuinely undated one is added, read it through cache.read_unrestricted with a "
            "reason rather than widening this."
        )
    return obj[index < pd.Timestamp(boundary(config))]


def is_truncated(obj: pd.DataFrame | pd.Series, *, config: Config | None = None) -> bool:
    """Whether ``obj`` already stops before the boundary. For tests and asserts."""
    index = obj.index
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return True
    return bool(index.max() < pd.Timestamp(boundary(config)))
