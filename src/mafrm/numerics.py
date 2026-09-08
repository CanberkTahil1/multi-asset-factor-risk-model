"""Numerical helpers shared by :mod:`mafrm.factors` and :mod:`mafrm.risk`.

This module sits **beneath both**, and the placement is a rule rather than a
convenience.

CLAUDE.md invariant 10 says ``mafrm.risk`` must not know what asset class it is
looking at. The corollary that will actually get violated one day is not a
stray ``if asset_class ==`` -- nobody writes that on purpose -- it is the
**import edge**, because reaching sideways for a helper that already exists
always looks harmless at the moment it is done. Once ``mafrm.risk`` imports
anything from ``mafrm.factors``, the week-7 equity module inherits whatever
assumptions that helper carried, and the failure surfaces three weeks later as
"why does the equity covariance need the rates calendar".

So the direction is fixed: **``mafrm.risk`` imports from nothing above it, and
never from ``mafrm.factors``.** The shared EWMA weighting lives here rather than
in either package, which leaves no edge between them in *either* direction and
therefore nothing for a later session to follow backwards.
``tests/test_risk_architecture.py`` asserts this mechanically rather than
trusting this paragraph.

Everything here is pure arithmetic on arrays. No config, no I/O, no state.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

__all__ = [
    "LN2",
    "effective_sample_size",
    "ewma_weights",
    "realised_effective_sample_size",
]

#: ``ln 2``. Appears in every half-life conversion in the project; named so that
#: the conversion is greppable and so no call site writes ``0.693``.
LN2: Final[float] = math.log(2.0)


def ewma_weights(n_obs: int, halflife: float, *, normalize: bool = True) -> np.ndarray:
    """Exponentially decaying weights for a window ending at its last row.

    Returned **oldest first**, matching the row order of a time-indexed frame,
    so ``weights @ values`` needs no reversal at the call site. The most recent
    observation carries the largest weight; an observation ``halflife`` days
    older carries half of it.

    ``normalize`` divides by the sum, which is what every estimator here wants:
    an EWMA moment is a weighted average, and leaving the weights unnormalised
    makes the estimate depend on how much history happens to be available.
    Pass ``normalize=False`` only to inspect the raw decay.

    USE4 Table 4.1 states half-lives in days; this is the conversion from a
    half-life to the weights it implies.
    """
    if n_obs < 1:
        raise ValueError(f"ewma_weights: n_obs must be at least 1, got {n_obs}")
    if not halflife > 0.0:
        raise ValueError(f"ewma_weights: halflife must be positive, got {halflife}")
    age = np.arange(n_obs - 1, -1, -1, dtype=float)
    weights: np.ndarray = np.exp(-LN2 * age / halflife)
    if normalize:
        weights = weights / weights.sum()
    return weights


def effective_sample_size(halflife: float) -> float:
    """``T_eff = 2*tau/ln 2`` -- the asymptotic EWMA effective sample size.

    CLAUDE.md's parameter table and ``model.numerics.effective_sample_size``.
    This is the limit reached by an *infinitely long* exponentially weighted
    sample, which is what SPEC.md 15.2's K/T table quotes: 84d -> 242,
    252d -> 727, 504d -> 1454.

    It is therefore an **upper bound** on what a finite window actually delivers.
    Where the distinction matters -- and for a K/T diagnostic it does, because
    the whole point of that number is how much estimation error there is --
    :func:`realised_effective_sample_size` gives the finite-sample counterpart
    and the two are reported side by side.
    """
    if not halflife > 0.0:
        raise ValueError(f"effective_sample_size: halflife must be positive, got {halflife}")
    return 2.0 * halflife / LN2


def realised_effective_sample_size(weights: np.ndarray) -> float:
    """Kish's effective sample size ``(sum w)^2 / sum w^2`` for actual weights.

    The finite-sample counterpart to :func:`effective_sample_size`. For a window
    long relative to the half-life the two agree; for a window shorter than
    ``2*tau/ln 2`` this one is smaller, because the weights have not had room to
    decay and the estimate is closer to an equally-weighted average over
    whatever history exists.

    This is the honest denominator for a K/T ratio computed on a real window,
    and it is the quantity a burn-in rule should eventually bound. A day count
    could not serve that purpose: it would have to be different for every
    half-life and would say nothing about how many factors are being estimated.
    """
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError(
            f"realised_effective_sample_size: expected 1-D weights, got {weights.shape}"
        )
    if np.any(weights < 0.0):
        raise ValueError("realised_effective_sample_size: weights must be non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("realised_effective_sample_size: weights sum to zero")
    return total * total / float(np.square(weights).sum())
