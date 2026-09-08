"""Limits as HARD constraints or HINGE penalties, the ladder, and what bound. SPEC.md 8.1, W6-P1.

SPEC.md 8.1: *"Soft constraints, not hard ones. Convert limits into hinge
penalties ... the problem is then always feasible ... Set the priority parameters
from the Lagrange multipliers of the hard-constrained problem observed in
backtest, at around the 80th percentile."* And: *"Even so, plan for infeasibility:
a relaxation ladder, fallback to prior weights, and logging of which constraint
bound."*

This module is the bookkeeping half of that. Each :class:`Limit` names a
quantity of the decision (``|z_i|`` against ``cap_i``, ``w_i`` against an upper
bound, one-way turnover, forecast volatility), a ``bound``, and a ``priority``
that is either ``None`` -- impose it HARD and OBSERVE its multiplier, which is
what W6-P1 does by ruling 5 (SPEC.md 8.5.1) because the recipe has nothing to
bootstrap from until something binds -- or a positive number, which makes it
``priority * pos(violation)`` in the objective and can never cause infeasibility.

:func:`ladder` writes down, before the first solve, the exact sequence of
attempts a failed solve walks: the primary solver, the fallback solver, then the
same pair with each hard constraint on ``optimizer.relaxation_ladder`` dropped in
order, then PRIOR WEIGHTS. The sequence is a value, so a test can assert it and a
report can print which rung a date reached. :func:`summarise` turns the per-date
readings into the table ``reports/binding_constraints.md`` shows: how often each
limit bound, and the multiplier percentiles W6-P2 reads its priorities off.

Nothing here knows what the assets are.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mafrm.config import OptimizerConstraint, OptimizerConstraintsConfig

__all__ = [
    "PRIOR_WEIGHTS",
    "Attempt",
    "ConstraintError",
    "Limit",
    "Reading",
    "ladder",
    "limits_from_config",
    "summarise",
]

#: The last rung of every ladder.
PRIOR_WEIGHTS = "prior_weights"


class ConstraintError(ValueError):
    """A limit or a ladder is mis-specified."""


@dataclass(frozen=True)
class Limit:
    """One limit and how it is imposed. Mirrors :class:`mafrm.config.OptimizerConstraint`.

    ``bound`` is a scalar for ``position_box`` (per-asset upper weight),
    ``turnover`` (one-way ``sum |z|``) and ``tracking_error`` (forecast volatility
    per period); for ``adv_participation`` it is the participation cap and the
    per-asset bound on ``|z_i|`` is ``cap * V_i / NAV``, supplied at solve time.
    """

    name: str
    bound: float
    #: ``None`` -> hard; a positive number -> hinge priority.
    priority: float | None

    @property
    def hard(self) -> bool:
        return self.priority is None

    def relaxed(self) -> Limit:
        """The same limit switched off -- what a ladder rung does to it."""
        return Limit(name=self.name, bound=np.inf, priority=self.priority)

    @property
    def off(self) -> bool:
        return not np.isfinite(self.bound)


def limits_from_config(config: OptimizerConstraintsConfig) -> tuple[Limit, ...]:
    """Every ACTIVE limit in ``optimizer.constraints``, in the config's order."""
    out: list[Limit] = []
    for spec in config.limits:
        if spec.active:
            assert spec.bound is not None
            out.append(Limit(name=spec.name, bound=float(spec.bound), priority=spec.priority))
    return tuple(out)


def _check_ladder(names: Sequence[str], limits: Sequence[Limit]) -> None:
    by_name = {limit.name: limit for limit in limits}
    if len(set(names)) != len(names):
        raise ConstraintError(f"relaxation ladder repeats a constraint: {list(names)}")
    for name in names:
        if name not in by_name:
            raise ConstraintError(f"relaxation ladder names {name!r}, which is not an active limit")
        if not by_name[name].hard:
            raise ConstraintError(
                f"relaxation ladder names {name!r}, a hinge; only a HARD constraint can make the "
                "problem infeasible"
            )


@dataclass(frozen=True)
class Attempt:
    """One rung of the ladder: which solver, with which hard constraints dropped."""

    rung: int
    solver: str
    #: Names of the hard constraints switched off at this rung, in ladder order.
    relaxed: tuple[str, ...]

    @property
    def is_prior_weights(self) -> bool:
        return self.solver == PRIOR_WEIGHTS

    @property
    def label(self) -> str:
        if self.is_prior_weights:
            return PRIOR_WEIGHTS
        dropped = f" without {', '.join(self.relaxed)}" if self.relaxed else ""
        return f"{self.solver}{dropped}"


def ladder(
    limits: Sequence[Limit], *, order: Sequence[str], primary: str, fallback: str
) -> tuple[Attempt, ...]:
    """The attempt sequence, written down before the first solve.

    ``(primary, nothing dropped)``, ``(fallback, nothing dropped)``, then for each
    prefix of ``order`` the same two solvers with that prefix dropped, then
    :data:`PRIOR_WEIGHTS`. The fallback solver comes BEFORE any relaxation because
    a numerical failure is not an infeasibility, and dropping a constraint to cure
    one would log the wrong cause.
    """
    _check_ladder(order, limits)
    if primary == fallback:
        raise ConstraintError(f"primary and fallback solvers must differ, both {primary!r}")
    attempts: list[Attempt] = []
    rung = 0
    for depth in range(len(order) + 1):
        dropped = tuple(order[:depth])
        for solver in (primary, fallback):
            attempts.append(Attempt(rung=rung, solver=solver, relaxed=dropped))
            rung += 1
    attempts.append(Attempt(rung=rung, solver=PRIOR_WEIGHTS, relaxed=tuple(order)))
    return tuple(attempts)


@dataclass(frozen=True)
class Reading:
    """What one limit did at one rebalance."""

    name: str
    hard: bool
    #: The limit's bound as imposed (participation cap, weight, turnover, volatility).
    bound: float
    #: The realised value of the constrained quantity at the solution -- for a
    #: per-asset limit, the tightest asset's (largest ratio of value to bound).
    value: float
    #: Whether the solution sits on the bound (hard) or beyond it (hinge).
    binding: bool
    #: Hard: the largest Lagrange multiplier across the limit's rows. Hinge: NaN.
    multiplier: float
    #: Hinge: the summed violation ``sum pos(value - bound)``. Hard: 0.
    violation: float
    #: Whether this rebalance dropped the limit on a ladder rung.
    relaxed: bool


@dataclass
class _Tally:
    """Per-limit accumulator behind :func:`summarise`."""

    kind: str
    rebalances: int = 0
    imposed: int = 0
    binding: int = 0
    relaxed: int = 0
    multipliers: list[float] = field(default_factory=list)
    violations: list[float] = field(default_factory=list)


def summarise(readings: Sequence[Sequence[Reading]], *, percentile: float) -> pd.DataFrame:
    """Per limit: how often it bound, how often it was relaxed, and the multiplier quantiles.

    ``readings[i]`` is the tuple of readings at rebalance ``i``. The
    ``multiplier_p<percentile>`` column is SPEC.md 8.1's recipe -- the priority
    W6-P2 will set for the hinge -- taken over the rebalances at which the limit
    was IMPOSED (not relaxed) and BOUND; a multiplier of zero at a slack
    constraint says nothing about the price of tightness and is excluded, which
    the report states. The count of binding dates is beside it so a percentile of
    three observations cannot be read as a population statistic.
    """
    if not (0.0 < percentile < 1.0):
        raise ConstraintError(f"summarise: percentile must be in (0, 1), got {percentile}")
    tallies: dict[str, _Tally] = {}
    for date_readings in readings:
        for reading in date_readings:
            tally = tallies.setdefault(
                reading.name, _Tally(kind="hard" if reading.hard else "hinge")
            )
            tally.rebalances += 1
            if reading.relaxed:
                tally.relaxed += 1
                continue
            tally.imposed += 1
            if reading.binding:
                tally.binding += 1
                if reading.hard and np.isfinite(reading.multiplier):
                    tally.multipliers.append(reading.multiplier)
                if not reading.hard:
                    tally.violations.append(reading.violation)
    label = f"multiplier_p{round(percentile * 100)}"
    table: list[dict[str, object]] = []
    for name, tally in tallies.items():
        multipliers = np.asarray(tally.multipliers, dtype=float)
        violations = np.asarray(tally.violations, dtype=float)
        table.append(
            {
                "constraint": name,
                "kind": tally.kind,
                "rebalances": tally.rebalances,
                "imposed": tally.imposed,
                "binding": tally.binding,
                "relaxed": tally.relaxed,
                "binding_share": (tally.binding / tally.imposed) if tally.imposed else np.nan,
                "multiplier_median": float(np.median(multipliers)) if multipliers.size else np.nan,
                label: float(np.quantile(multipliers, percentile)) if multipliers.size else np.nan,
                "multiplier_max": float(multipliers.max()) if multipliers.size else np.nan,
                "violation_mean": float(violations.mean()) if violations.size else np.nan,
            }
        )
    return pd.DataFrame(table).set_index("constraint") if table else pd.DataFrame()


def as_config_limit(limit: Limit) -> OptimizerConstraint:
    """Back to the config's shape, for callers that round-trip through it."""
    return OptimizerConstraint(
        name=limit.name,
        bound=None if limit.off else float(limit.bound),
        priority=limit.priority,
    )
