"""SPEC.md 8's single-period cost-aware optimizer, in cvxpy directly. W6-P1.

**cvxportfolio is not imported and must never be** (CLAUDE.md invariant 3: it is
GPL-3.0). Its cost specification -- Boyd, Busseti, Diamond, Kahn, Koh, Nystrup &
Speth, "Multi-Period Trading via Convex Optimization" (2017) -- is the benchmark
SPEC.md 7 reimplements and SPEC.md 8.1 cites; the objective below is that
specification with this project's parameterisation, and nothing of their code.

THE PROBLEM, per rebalance (SPEC.md 8.1, with the W6-P1 rulings of SPEC.md 8.5)

    maximize   alpha' w  -  rho' |w|                        (SPEC.md 8.2, rho = 0 by ruling)
             - gamma_risk * [ w' Sigma w + varrho (sum_i sigma_i |w_i|)^2 ]
             - psi_mis * (alpha_perp' w)^2                   (SPEC.md 8.3, on the unit direction)
             - gamma_trade * phi_trade(z)                    (SPEC.md 7.1, inside the objective)
             - gamma_hold * sum_i pos(-w_i)                  (gamma_hold = 0 by ruling)
             - sum_hinges priority * pos(violation)
    s.t.       z = w - w_prior
               1' w = 1,  w >= 0                             (ruling 5: the simplex)
               hard limits, if any                           (W6-P1: the 6% ADV cap)

The risk term uses the FACTOR STRUCTURE ``Sigma = X F X' + Delta`` rather than
a dense matrix: ``w' Sigma w = ||F^{1/2} X' w||^2 + sum_i delta_i^2 w_i^2``, two
sums of squares over ``K + N`` terms. ``F`` is asserted PSD before its square
root is taken and ``Delta >= 0`` is asserted (CLAUDE.md invariant 4).

``gamma_risk`` IS SPEC.md 8.4's ``lambda``. :func:`gamma_risk_from_spec` computes
``IR / (2 TE_target)`` with ``IR = sqrt(alpha' Sigma^-1 alpha)`` -- "the right
initialization, not the answer" -- and the caller may pin it instead
(``optimizer.gamma_risk``). **``gamma_risk`` and ``gamma_trade`` are not
separable**: a higher ``gamma_trade`` reduces responsiveness, which mechanically
lowers realised risk and interacts with ``gamma_risk``; a change to either owes
a look at the other, and a tuned pair is a ``strategy-config`` row.

FEASIBILITY, AND WHAT HAPPENS WHEN A SOLVE FAILS
    The simplex is never empty, and a hinge cannot make it so; only a HARD limit
    can. When the solver does not return an optimal point, :func:`solve_rebalance`
    walks the :func:`~mafrm.backtest.constraints.ladder`: fallback solver, then
    each hard limit on ``optimizer.relaxation_ladder`` dropped in order with both
    solvers, then the PRIOR WEIGHTS (``z = 0``). Every rung tried is recorded
    on the result with its solver status, and a fallback to prior weights is a
    named event -- **a failed solve never silently returns zeros**, which is
    asserted at the end of the function rather than hoped for.

UNITS. ``alpha`` is expected return per rebalance period as a fraction;
``factor_covariance`` and ``specific_variance`` are per-period variances in
fraction^2; the cost inputs are SPEC.md 7.1's proportional units; ``te_target``
is a per-period volatility as a fraction. The caller owns the conversion from
the risk pipeline's basis points per day (:mod:`mafrm.backtest.verification`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from mafrm.backtest import constraints as constraints_mod
from mafrm.backtest.constraints import PRIOR_WEIGHTS, Attempt, Limit, Reading

__all__ = [
    "FactorRisk",
    "OptimizerError",
    "Penalties",
    "RebalanceProblem",
    "RebalanceResult",
    "TradingCost",
    "gamma_risk_from_spec",
    "information_ratio",
    "solve_rebalance",
]

FloatArray = NDArray[np.float64]

#: Solver statuses accepted as a solution. ``OPTIMAL_INACCURATE`` is accepted
#: and RECORDED, never silently.
_ACCEPTED: frozenset[str] = frozenset({cp.OPTIMAL, cp.OPTIMAL_INACCURATE})


class OptimizerError(ValueError):
    """The problem is mis-specified. Nothing is filled or defaulted."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorRisk:
    """``Sigma = X F X' + diag(delta^2)`` for one rebalance, per-period fraction units."""

    exposures: FloatArray
    factor_covariance: FloatArray
    specific_variance: FloatArray

    def __post_init__(self) -> None:
        x = np.asarray(self.exposures, dtype=float)
        f = np.asarray(self.factor_covariance, dtype=float)
        d = np.asarray(self.specific_variance, dtype=float)
        if x.ndim != 2:
            raise OptimizerError(f"FactorRisk: exposures must be N x K, got {x.shape}")
        n, k = x.shape
        if f.shape != (k, k):
            raise OptimizerError(f"FactorRisk: F is {f.shape} against K = {k}")
        if d.shape != (n,):
            raise OptimizerError(f"FactorRisk: specific variance is {d.shape} against N = {n}")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(f)) and np.all(np.isfinite(d))):
            raise OptimizerError("FactorRisk: every input must be finite")
        if np.any(d < 0.0):
            raise OptimizerError("FactorRisk: a specific VARIANCE is negative")
        if not np.allclose(f, f.T, atol=0.0, rtol=1e-10):
            raise OptimizerError("FactorRisk: F is not symmetric")
        # CLAUDE.md invariant 4: assert PSD at the point the matrix is consumed.
        smallest = float(np.linalg.eigvalsh(f).min())
        scale = float(max(np.abs(np.diag(f)).max(), np.finfo(float).tiny))
        if smallest < -k * np.finfo(float).eps * scale:
            raise OptimizerError(
                f"FactorRisk: F is not PSD (smallest eigenvalue {smallest:.3e} against a diagonal "
                f"scale of {scale:.3e}). Repair belongs upstream in mafrm.risk, not here."
            )

    @property
    def assets(self) -> int:
        return int(self.exposures.shape[0])

    @property
    def factors(self) -> int:
        return int(self.exposures.shape[1])

    def factor_sqrt(self) -> FloatArray:
        """A symmetric ``F^{1/2}`` with negative rounding-noise eigenvalues clipped to zero."""
        values, vectors = np.linalg.eigh(self.factor_covariance)
        root: FloatArray = (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T
        return root

    def covariance(self) -> FloatArray:
        """The dense ``N x N`` matrix, for diagnostics and for SPEC.md 8.4's ``IR``."""
        x = self.exposures
        dense: FloatArray = x @ self.factor_covariance @ x.T + np.diag(self.specific_variance)
        symmetric: FloatArray = 0.5 * (dense + dense.T)
        return symmetric

    def risk_rows(self) -> FloatArray:
        """``M`` with ``||M w||^2 = w' Sigma w``: ``F^{1/2} X'`` stacked over ``diag(delta)``."""
        rows: FloatArray = np.vstack(
            [self.factor_sqrt() @ self.exposures.T, np.diag(np.sqrt(self.specific_variance))]
        )
        return rows

    def volatilities(self) -> FloatArray:
        """``sqrt(Sigma_ii)``, for SPEC.md 8.2's ``varrho`` term."""
        return np.sqrt(np.diag(self.covariance()))


@dataclass(frozen=True)
class TradingCost:
    """SPEC.md 7.1's per-asset inputs for one rebalance, proportional units."""

    half_spread: FloatArray
    daily_volatility: FloatArray
    adv_over_nav: FloatArray
    prefactor: float
    exponent: float
    asymmetry: float = 0.0

    def __post_init__(self) -> None:
        for name in ("half_spread", "daily_volatility", "adv_over_nav"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                raise OptimizerError(f"TradingCost: {name} must be a finite vector")
        if np.any(self.half_spread < 0.0) or np.any(self.daily_volatility < 0.0):
            raise OptimizerError("TradingCost: half_spread and daily_volatility must be >= 0")
        if np.any(self.adv_over_nav <= 0.0):
            raise OptimizerError("TradingCost: adv_over_nav must be positive for every asset")
        if not np.isfinite(self.prefactor) or self.prefactor < 0.0:
            raise OptimizerError(f"TradingCost: prefactor must be >= 0, got {self.prefactor}")
        if not np.isfinite(self.exponent) or self.exponent <= 1.0:
            raise OptimizerError(
                f"TradingCost: the exponent must exceed 1 for a convex cost, got {self.exponent}"
            )

    def impact_coefficient(self) -> FloatArray:
        """``Y sigma_i / (V_i/v)^(e-1)``: the coefficient on ``|z_i|^e``."""
        coefficient: FloatArray = (
            self.prefactor
            * self.daily_volatility
            / np.power(self.adv_over_nav, self.exponent - 1.0)
        )
        return coefficient

    def evaluate(self, trade: FloatArray) -> float:
        """``phi_trade(z)`` at a point, the same arithmetic the objective carries."""
        z = np.asarray(trade, dtype=float)
        size = np.abs(z)
        total = (
            self.half_spread @ size
            + self.impact_coefficient() @ np.power(size, self.exponent)
            + self.asymmetry * float(z.sum())
        )
        return float(total)


@dataclass(frozen=True)
class Penalties:
    """Every multiplier in the objective, for one rebalance."""

    gamma_risk: float
    gamma_trade: float
    gamma_hold: float = 0.0
    #: SPEC.md 8.3's penalty as ``psi_unit * (direction' w)^2`` -- see
    #: :class:`mafrm.backtest.alpha.MisalignmentPenalty`. ``psi_unit = 0`` switches it off.
    psi_unit: float = 0.0
    misalignment_direction: FloatArray | None = None
    #: SPEC.md 8.2's two robustification amplitudes. Zero by ruling 4.
    rho: float = 0.0
    varrho: float = 0.0

    def __post_init__(self) -> None:
        for name in ("gamma_risk", "gamma_trade", "gamma_hold", "psi_unit", "rho", "varrho"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise OptimizerError(f"Penalties: {name} must be finite and >= 0, got {value}")
        if not (0.0 <= self.varrho < 1.0):
            raise OptimizerError(f"Penalties: SPEC.md 8.2 has varrho in [0, 1), got {self.varrho}")
        if self.psi_unit > 0.0 and self.misalignment_direction is None:
            raise OptimizerError("Penalties: psi_unit > 0 needs misalignment_direction")


@dataclass(frozen=True)
class RebalanceProblem:
    """Everything one rebalance needs. Assets are positions in a vector, nothing more."""

    alpha: FloatArray
    risk: FactorRisk
    prior_weights: FloatArray
    penalties: Penalties
    cost: TradingCost | None
    limits: tuple[Limit, ...] = ()
    long_only: bool = True
    fully_invested: bool = True
    #: Hard limits dropped in this order on failure; then prior weights.
    relaxation_order: tuple[str, ...] = ()
    primary_solver: str = "CLARABEL"
    fallback_solver: str = "SCS"
    #: The solver's feasibility tolerance; a hard limit BINDS within it of its bound.
    feasibility_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        n = self.risk.assets
        for name in ("alpha", "prior_weights"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (n,) or not np.all(np.isfinite(values)):
                raise OptimizerError(f"RebalanceProblem: {name} must be a finite vector of {n}")
        if self.cost is not None and self.cost.half_spread.shape != (n,):
            raise OptimizerError("RebalanceProblem: the cost inputs do not match N")
        direction = self.penalties.misalignment_direction
        if direction is not None and direction.shape != (n,):
            raise OptimizerError("RebalanceProblem: misalignment_direction does not match N")
        names = [limit.name for limit in self.limits]
        if len(set(names)) != len(names):
            raise OptimizerError(f"RebalanceProblem: duplicate limits {names}")
        if not (0.0 < self.feasibility_tolerance < 1e-2):
            raise OptimizerError("RebalanceProblem: feasibility_tolerance is a solver tolerance")
        for name in self.relaxation_order:
            if name not in names:
                raise OptimizerError(f"RebalanceProblem: ladder names {name!r}, not a limit")

    @property
    def assets(self) -> int:
        return self.risk.assets

    def attempts(self) -> tuple[Attempt, ...]:
        return constraints_mod.ladder(
            self.limits,
            order=self.relaxation_order,
            primary=self.primary_solver,
            fallback=self.fallback_solver,
        )


# ---------------------------------------------------------------------------
# SPEC.md 8.4
# ---------------------------------------------------------------------------


def information_ratio(alpha: FloatArray, covariance: FloatArray) -> float:
    """Grinold-Kahn's ``IR = sqrt(alpha' Sigma^-1 alpha)``. SPEC.md 8.4."""
    a = np.asarray(alpha, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    if sigma.shape != (a.size, a.size):
        raise OptimizerError(f"information_ratio: Sigma {sigma.shape} against N = {a.size}")
    if not np.any(a):
        return 0.0
    solved = np.linalg.solve(sigma, a)
    quadratic = float(a @ solved)
    if not np.isfinite(quadratic) or quadratic < 0.0:
        raise OptimizerError(
            f"information_ratio: alpha' Sigma^-1 alpha = {quadratic}; Sigma is singular or not PSD"
        )
    return float(np.sqrt(quadratic))


def gamma_risk_from_spec(alpha: FloatArray, covariance: FloatArray, *, te_target: float) -> float:
    """``lambda = IR / (2 TE_target)``. SPEC.md 8.4's initialisation, not the answer.

    The unconstrained optimum ``h* = Sigma^-1 alpha / (2 lambda)`` then carries
    exactly ``TE_target`` of forecast risk, whatever alpha's scale. Raises at
    ``alpha = 0``: the formula gives ``lambda = 0`` there, which would delete the
    risk term, so a no-view run must pin ``gamma_risk`` (and cost-free, any
    positive value gives the same minimum-variance answer).
    """
    if not np.isfinite(te_target) or te_target <= 0.0:
        raise OptimizerError(f"gamma_risk_from_spec: TE_target must be positive, got {te_target}")
    ir = information_ratio(alpha, covariance)
    if ir == 0.0:
        raise OptimizerError(
            "gamma_risk_from_spec: alpha is zero, so IR = 0 and SPEC.md 8.4 gives lambda = 0. "
            "Pin gamma_risk for a no-view run; cost-free, any positive value is the same portfolio."
        )
    return float(ir / (2.0 * te_target))


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RebalanceResult:
    """Weights, how they were reached, and what bound. Never zeros by accident."""

    weights: FloatArray
    trade: FloatArray
    #: The rung that produced ``weights``.
    attempt: Attempt
    #: Every rung tried, in order, with the solver status each returned.
    attempts: tuple[tuple[Attempt, str], ...]
    status: str
    readings: tuple[Reading, ...]
    #: Forecast variance ``w' Sigma w`` and its square root, per period.
    forecast_variance: float
    #: ``phi_trade(z)`` at the solution as a fraction of NAV; 0 when cost-free.
    trading_cost: float
    #: Assets at the long-only floor (``w_i`` within tolerance of zero).
    at_lower_bound: int
    #: The objective's pieces at the solution, for the report.
    terms: dict[str, float] = field(default_factory=dict)

    @property
    def fell_back(self) -> bool:
        return self.attempt.is_prior_weights

    @property
    def relaxed(self) -> tuple[str, ...]:
        return self.attempt.relaxed

    @property
    def forecast_volatility(self) -> float:
        return float(np.sqrt(max(self.forecast_variance, 0.0)))

    @property
    def turnover(self) -> float:
        """One-way, ``sum |z|``."""
        return float(np.abs(self.trade).sum())

    @property
    def largest_weight(self) -> float:
        return float(self.weights.max())

    @property
    def effective_assets(self) -> float:
        """``1 / sum w_i^2``, the inverse Herfindahl: 13 at equal weight, 1 at a corner."""
        total = float(self.weights @ self.weights)
        return float(1.0 / total) if total > 0.0 else 0.0


def _build(
    problem: RebalanceProblem, imposed: tuple[Limit, ...], adv_bounds: FloatArray | None
) -> tuple[cp.Problem, cp.Variable, dict[str, cp.Constraint | cp.Expression]]:
    """The cvxpy problem for one rung. ``imposed`` are the limits still in force."""
    n = problem.assets
    w = cp.Variable(n, name="w")
    z = w - problem.prior_weights
    pen = problem.penalties
    risk = problem.risk

    objective = problem.alpha @ w
    if pen.rho > 0.0:
        objective = objective - pen.rho * cp.norm1(w)  # SPEC.md 8.2, worst case over alpha +- rho
    variance = cp.sum_squares(risk.risk_rows() @ w)
    if pen.varrho > 0.0:
        variance = variance + pen.varrho * cp.square(risk.volatilities() @ cp.abs(w))
    objective = objective - pen.gamma_risk * variance
    if pen.psi_unit > 0.0 and pen.misalignment_direction is not None:
        objective = objective - pen.psi_unit * cp.square(pen.misalignment_direction @ w)
    if problem.cost is not None and pen.gamma_trade > 0.0:
        cost = problem.cost
        size = cp.abs(z)
        trading = cost.half_spread @ size
        coefficient = cost.impact_coefficient()
        if np.any(coefficient > 0.0):
            trading = trading + coefficient @ cp.power(size, cost.exponent)
        if cost.asymmetry != 0.0:
            trading = trading + cost.asymmetry * cp.sum(z)
        objective = objective - pen.gamma_trade * trading
    if pen.gamma_hold > 0.0:
        objective = objective - pen.gamma_hold * cp.sum(cp.pos(-w))

    hard: list[cp.Constraint] = []
    handles: dict[str, cp.Constraint | cp.Expression] = {}
    if problem.fully_invested:
        hard.append(cp.sum(w) == 1.0)
    if problem.long_only:
        nonneg = w >= 0.0
        hard.append(nonneg)
        handles["__long_only__"] = nonneg

    for limit in imposed:
        if limit.name == "adv_participation":
            if adv_bounds is None:
                raise OptimizerError("adv_participation needs cost inputs (ADV over NAV)")
            expr: cp.Expression = cp.abs(z) - adv_bounds
        elif limit.name == "position_box":
            expr = w - limit.bound
        elif limit.name == "turnover":
            expr = cp.sum(cp.abs(z)) - limit.bound
        elif limit.name == "tracking_error":
            expr = cp.norm(risk.risk_rows() @ w, 2) - limit.bound
        else:
            raise OptimizerError(f"unknown limit {limit.name!r}")
        if limit.hard:
            constraint = expr <= 0.0
            hard.append(constraint)
            handles[limit.name] = constraint
        else:
            assert limit.priority is not None
            objective = objective - limit.priority * cp.sum(cp.pos(expr))
            handles[limit.name] = expr

    # NORMALISATION, not a change to the problem. Per-period variances are O(1e-4)
    # and alphas O(1e-3), so the raw objective sits near the solvers' absolute
    # tolerances and two runs that should coincide differ in the fifth digit
    # (found by tests/test_optimizer.py's gamma_risk-invariance test). Dividing
    # the whole objective by a positive constant leaves the argmax where it was;
    # the constant is gamma_risk times the mean forecast variance, the natural
    # unit of the risk term, and the duals are multiplied back by it in
    # _readings so the reported multipliers stay in objective units.
    scale = _objective_scale(problem)
    handles["__scale__"] = scale
    return cp.Problem(cp.Maximize(objective / scale), hard), w, handles


def _objective_scale(problem: RebalanceProblem) -> float:
    """``gamma_risk * mean_i Sigma_ii``, the risk term's unit; ``1`` when there is no risk term.

    THE FALLBACK CHANGED IN W6-P3 (SPEC.md 9.2's registered residual, resolved
    under SPEC.md 9.3 ruling 6). In constraint mode ``gamma_risk`` is zero, and
    W6-P2 normalised by the largest ``|alpha|`` instead (about 0.03): Clarabel
    then returned ``optimal_inaccurate`` on 22-37 of 187 rebalances in every
    treatment-D factor cell. The re-solve found the flagged points CORRECT --
    feasible, and within ``3e-7`` (L1) of the point the same problem gives at
    scale ``1`` with status ``optimal`` -- while the fallback solver's point on
    the same date violated the TE bound by 2.7% and was the whole of the
    ``4.7e-2`` distance. The normalisation exists for the risk term's units;
    with no risk term the objective is left in its own units, which is where
    the solver converges cleanly. Registered before it was measured: above
    ``1e-3`` "the normalisation changes and the affected cells re-run as new
    rows" (experiments.md rows 254-260).
    """
    pen = problem.penalties
    variance = float(np.mean(np.diag(problem.risk.covariance())))
    scale = pen.gamma_risk * variance
    return scale if scale > 0.0 else 1.0


def _solve(prob: cp.Problem, solver: str) -> str:
    try:
        prob.solve(solver=solver)
    except (
        cp.SolverError,
        ValueError,
        ArithmeticError,
    ) as exc:  # pragma: no cover - solver-specific
        return f"exception: {type(exc).__name__}: {exc}"[:200]
    return str(prob.status)


def _valid(weights: FloatArray | None, problem: RebalanceProblem) -> bool:
    if weights is None or not np.all(np.isfinite(weights)):
        return False
    tol = 1e3 * problem.feasibility_tolerance
    if problem.fully_invested and abs(float(weights.sum()) - 1.0) > tol:
        return False
    return not (problem.long_only and float(weights.min()) < -tol)


def _max_dual(constraint: Any, scale: float) -> float:
    """The largest multiplier across the constraint's rows, back in objective units."""
    dual = constraint.dual_value
    if dual is None:
        return float("nan")
    return float(np.max(np.abs(np.asarray(dual, dtype=float))) * scale)


def _readings(
    problem: RebalanceProblem,
    attempt: Attempt,
    weights: FloatArray,
    handles: dict[str, Any],
    adv_bounds: FloatArray | None,
) -> tuple[Reading, ...]:
    tol = problem.feasibility_tolerance
    z = weights - problem.prior_weights
    out: list[Reading] = []
    for limit in problem.limits:
        relaxed = limit.name in attempt.relaxed
        if limit.name == "adv_participation":
            if adv_bounds is None:
                continue
            ratios = np.abs(z) / adv_bounds
            value = float(ratios.max()) * limit.bound
            violation = float(np.clip(np.abs(z) - adv_bounds, 0.0, None).sum())
            slack = float((adv_bounds - np.abs(z)).min())
            size = float(adv_bounds.min())
        elif limit.name == "position_box":
            value = float(weights.max())
            violation = float(np.clip(weights - limit.bound, 0.0, None).sum())
            slack = limit.bound - value
            size = limit.bound
        elif limit.name == "turnover":
            value = float(np.abs(z).sum())
            violation = max(value - limit.bound, 0.0)
            slack = limit.bound - value
            size = limit.bound
        else:  # tracking_error
            value = float(np.linalg.norm(problem.risk.risk_rows() @ weights))
            violation = max(value - limit.bound, 0.0)
            slack = limit.bound - value
            size = limit.bound
        if relaxed or attempt.is_prior_weights:
            binding = False
            multiplier = float("nan")
        elif limit.hard:
            handle = handles.get(limit.name)
            scale = float(handles.get("__scale__", 1.0))
            multiplier = _max_dual(handle, scale) if handle is not None else float("nan")
            binding = slack <= tol * max(1.0, size)
        else:
            multiplier = float("nan")
            binding = violation > tol * max(1.0, size)
        out.append(
            Reading(
                name=limit.name,
                hard=limit.hard,
                bound=limit.bound,
                value=value,
                binding=bool(binding),
                multiplier=multiplier,
                violation=float(violation if not limit.hard else 0.0),
                relaxed=bool(relaxed),
            )
        )
    return tuple(out)


def solve_rebalance(problem: RebalanceProblem) -> RebalanceResult:
    """Solve one rebalance, walking the ladder on failure. See the module docstring.

    Returns the first rung that yields an accepted status AND a point that is
    finite and on the simplex (a solver can report optimal and hand back a point
    that is not, so the check is on the point). If every solver rung fails the
    result is the PRIOR WEIGHTS with ``fell_back`` true and every status kept.
    """
    adv_bounds: FloatArray | None = None
    adv = next((limit for limit in problem.limits if limit.name == "adv_participation"), None)
    if adv is not None:
        if problem.cost is None:
            raise OptimizerError(
                "adv_participation is imposed but the problem is cost-free, so there is no ADV "
                "to cap against. Drop the limit or supply cost inputs."
            )
        adv_bounds = adv.bound * problem.cost.adv_over_nav

    tried: list[tuple[Attempt, str]] = []
    for attempt in problem.attempts():
        if attempt.is_prior_weights:
            weights = np.asarray(problem.prior_weights, dtype=float).copy()
            tried.append((attempt, PRIOR_WEIGHTS))
            handles: dict[str, Any] = {}
            status = PRIOR_WEIGHTS
            break
        imposed = tuple(limit for limit in problem.limits if limit.name not in attempt.relaxed)
        prob, variable, handles = _build(problem, imposed, adv_bounds)
        status = _solve(prob, attempt.solver)
        tried.append((attempt, status))
        candidate = None if variable.value is None else np.asarray(variable.value, dtype=float)
        if status in _ACCEPTED and _valid(candidate, problem):
            assert candidate is not None
            weights = candidate
            if problem.long_only:
                weights = np.clip(weights, 0.0, None)
            if problem.fully_invested:
                weights = weights / weights.sum()
            break
    else:  # pragma: no cover - the ladder always ends in prior weights
        raise OptimizerError("the ladder ended without a prior-weights rung")

    if not np.all(np.isfinite(weights)):
        raise OptimizerError(
            "solve_rebalance: non-finite weights survived the ladder; this is a bug"
        )
    if not np.any(weights) and np.any(problem.prior_weights):
        raise OptimizerError(
            "solve_rebalance: an all-zero weight vector was about to be returned from a non-empty "
            "prior. A failed solve must fall back to the prior weights, never to zeros."
        )

    trade = weights - problem.prior_weights
    readings = _readings(problem, attempt, weights, handles, adv_bounds)
    variance = float(weights @ problem.risk.covariance() @ weights)
    trading_cost = problem.cost.evaluate(trade) if problem.cost is not None else 0.0
    pen = problem.penalties
    terms = {
        "alpha": float(problem.alpha @ weights),
        "risk": float(pen.gamma_risk * variance),
        "misalignment": (
            float(pen.psi_unit * (pen.misalignment_direction @ weights) ** 2)
            if pen.psi_unit > 0.0 and pen.misalignment_direction is not None
            else 0.0
        ),
        "trading_cost": float(pen.gamma_trade * trading_cost),
    }
    tol = problem.feasibility_tolerance
    return RebalanceResult(
        weights=weights,
        trade=trade,
        attempt=attempt,
        attempts=tuple(tried),
        status=status,
        readings=readings,
        forecast_variance=variance,
        trading_cost=trading_cost,
        at_lower_bound=int(np.sum(weights <= tol * 1e2)) if problem.long_only else 0,
        terms=terms,
    )
