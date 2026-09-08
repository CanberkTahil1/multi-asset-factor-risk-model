"""Horizon scaling for a daily covariance matrix. SPEC.md 5.2, W3-P2 ruling.

SPEC.md 5.2 writes the Newey-West sum as ``C0 + sum(...)``, *"then scaled by the
horizon (x21 for monthly)"*. **That scaling does not happen inside the stage, or
inside any stage.** The covariance pipeline stays in daily units through all five
of SPEC.md 5's stages and the multiplier is applied here, once, by the caller,
at the point of consumption.

WHY THE SCALING IS NOT IN THE PIPELINE
--------------------------------------

Two reasons, and the first is the binding one.

**CLAUDE.md failure mode 7 -- corrections interacting badly.** A scaling applied
inside a stage is invisible to every later stage and to every consumer, and each
stage would still pass its own sanity check: a matrix 21x too large is PSD,
symmetric and well conditioned. This project has already been bitten three times
by corrections that each passed their own check, which is why the multiplier is
in one named function with a test asserting it cannot be applied twice rather
than in whichever stage happened to need it first.

**Different consumers legitimately want different horizons.** SPEC.md 6.1's bias
statistic is daily; a VaR number is monthly; the optimizer's is its rebalance
interval. A matrix that has already chosen one forces every other consumer to
undo it, and undoing a scaling is exactly the operation nobody remembers to do.
``risk/`` should no more decide the caller's horizon than it should know the
caller's asset class -- the same principle as CLAUDE.md invariant 10, applied to
a different axis.

The mechanism, rather than the intention: ``trading_days_per_month`` lives under
``data`` in ``config/model.yaml`` and is **deliberately absent from**
:class:`~mafrm.risk.config.RiskConfig`. Nothing the pipeline can reach knows the
number, so a stage cannot apply it even by accident.

THIS IS NOT THE OTHER RESCALING
-------------------------------

Two different rescalings arrive in week 3 and confusing them would be easy, so
they are in separately named functions in separate modules:

:func:`scale_to_horizon` (here)
    **Global.** One scalar, the same for every factor, converting the time unit
    of the whole matrix. Changes the units of risk, not the units of any factor.

W3-P3's numeraire decision (SPEC.md 5.3.1, ``experiments.md`` row 96)
    **Per factor.** A *diagonal* rescaling, different per column, fixing the fact
    that this project's factor panel deliberately mixes decimal returns with
    basis points of yield. Its purpose is to make the eigenfactor adjustment
    invariant to a choice of units; it is not implemented yet.

When the units are next questioned, the distinction to check is scalar versus
diagonal.

HARMLESS TO SPEC.md 5.3, AND WHY THAT IS WORTH ASSERTING
--------------------------------------------------------

``c * F`` has eigenvalues ``c * lambda_k`` and the **same** eigenvectors, so a
uniform scalar multiple changes neither which direction is the smallest
eigenfactor nor the ratios ``lambda_tilde / lambda`` that SPEC.md 5.3's
adjustment reads. Deferring the multiplier past the eigenfactor stage therefore
costs nothing, and ``tests/test_horizon.py`` asserts it rather than leaving the
reader to take this paragraph's word for it. Note that this is exactly what is
*not* true of the per-factor rescaling above -- which is the whole content of
SPEC.md 5.3.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["HorizonCovariance", "HorizonError", "scale_to_horizon"]


class HorizonError(ValueError):
    """A horizon scaling was asked for that cannot be, or should not be, applied."""


@dataclass(frozen=True)
class HorizonCovariance:
    """A covariance matrix that has been scaled to a horizon, and by how much.

    A distinct **type** rather than a plain array, so that "has this already
    been scaled?" is a question the type system answers instead of a comment.
    :func:`scale_to_horizon` accepts ``np.ndarray`` and returns this, so feeding
    its own output back is a mypy error on ``src/`` and a :class:`HorizonError`
    at runtime.

    Reaching past it -- ``scale_to_horizon(scale_to_horizon(f, ...).matrix, ...)``
    -- is still possible, and is meant to be: it is legible at the call site,
    which a silently doubled multiplier inside a stage would not be.
    """

    #: ``K x K``, in units of ``trading_days`` days of variance.
    matrix: np.ndarray
    #: How many daily variances one unit of this matrix represents. 21 = monthly.
    trading_days: int

    @property
    def factors(self) -> int:
        """``K``."""
        return int(self.matrix.shape[0])

    def render(self) -> str:
        """One line, naming the horizon so a report cannot quote it unlabelled."""
        return (
            f"covariance scaled x{self.trading_days} "
            f"({self.factors}x{self.factors}, {self.trading_days}-day variance units)"
        )


def scale_to_horizon(daily: np.ndarray, *, trading_days: int) -> HorizonCovariance:
    """Scale a **daily** covariance matrix to a ``trading_days``-day horizon.

    ``F_h = trading_days * F_daily``. The square-root-of-time rule applied to a
    covariance rather than to a volatility, which is where it is linear.

    This is the **only** place in the project that applies a horizon multiplier.
    Pass ``model.data.trading_days_per_month`` for a monthly matrix; the argument
    is not defaulted, because a default would be this module deciding the
    caller's horizon, which is the thing the W3-P2 ruling forbids.

    Raises :class:`HorizonError` if handed a :class:`HorizonCovariance`, which is
    the double-application this function exists to make impossible.
    """
    if isinstance(daily, HorizonCovariance):
        raise HorizonError(
            "scale_to_horizon: this matrix has already been scaled to a "
            f"{daily.trading_days}-day horizon. Scaling it again would multiply the variance "
            "by the horizon twice and every downstream check -- PSD, symmetry, conditioning -- "
            "would still pass. Scale the daily matrix once, at the point of consumption."
        )
    if not isinstance(daily, np.ndarray):
        raise HorizonError(f"scale_to_horizon: expected a numpy array, got {type(daily).__name__}")
    if daily.ndim != 2 or daily.shape[0] != daily.shape[1]:
        raise HorizonError(f"scale_to_horizon: expected a square matrix, got {daily.shape}")
    if daily.shape[0] == 0:
        raise HorizonError("scale_to_horizon: empty matrix")
    if trading_days < 1:
        raise HorizonError(f"scale_to_horizon: trading_days must be at least 1, got {trading_days}")
    scaled: np.ndarray = float(trading_days) * daily
    return HorizonCovariance(matrix=scaled, trading_days=int(trading_days))
