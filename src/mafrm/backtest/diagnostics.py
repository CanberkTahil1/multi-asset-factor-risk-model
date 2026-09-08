"""SPEC.md 10.1-10.3's diagnostics on a backtested cell. W6-P3, under SPEC.md 9.3's rulings.

Three things, each an identity first and a measurement second, run on every
grid cell by :func:`mafrm.backtest.grid.run_cell` and rendered to
``reports/diagnostics.md``.

PEROLD IMPLEMENTATION SHORTFALL (SPEC.md 10.1), at the PARENT level
    ``IS = r_paper - r_real`` where the paper portfolio transacts the full
    intended quantity at the DECISION PRICE with zero cost. Operator ruling 5:
    the decision price is the rebalance close -- the engine decides on
    ``NAV_pre`` at that close and fills at that close, hitting the targets
    exactly -- and that is a FINDING, not a padding: a monthly simulation that
    executes at its decision price has no execution schedule, so the delay and
    opportunity legs are ZERO BY CONSTRUCTION and the four-way decomposition is
    degenerate, ``IS = impact + fees`` exactly. SPEC.md 10.1's "you own the
    decision timestamp" advantage over vendor TCA is realisable only with an
    intra-period schedule that does not exist here; an earlier timestamp would
    be invented. The predicted-versus-realised residual is defined so that it
    is not identically zero: PREDICTED is the cost the optimizer priced in its
    objective at the decision, REALISED is the SPEC.md 7.1 cost the engine
    charged on realisation. Under treatment A both are zero; under B nothing
    was priced, so the residual is the realised cost itself and carries no
    calibration information; the measurement is treatment C (a flat 2.5bp
    against the issuer level plus EDGE) and D (the as-of spread the optimizer
    saw against the as-of spread the engine charged on the same trade).

TRACKING ERROR (SPEC.md 10.2), additive in VARIANCE only
    ``sigma_a^2 = x_a' F x_a + h' Delta h`` with ``x_a = X' h`` the active factor
    exposures (``w^b = 0``: absolute risk, the identity's ``sigma_f``). The
    decomposition carries the two VARIANCES and the shares; the volatilities
    ``sqrt(x'Fx)`` and ``sqrt(h'Delta h)`` are reported beside them and DO NOT
    add to the total -- :func:`assert_variance_additive` refuses a
    decomposition whose volatilities add, and a test pins it. Risk contributions
    are Menchero & Davis (2011, *JPM* 37(2))'s x-sigma-rho: ``CTR_k = x_k sigma_k
    rho_k`` with ``rho_k`` the correlation of factor ``k`` with the total active
    return, ``(F x)_k / (sigma_k sigma_a)``; the ``K`` factor contributions plus
    the specific one sum to ``sigma_a`` exactly, and a contribution can be
    NEGATIVE for a diversifying exposure -- correct, and reported as such.

ATTRIBUTION, RECONCILED AGAINST THE RISK MODEL (SPEC.md 10.3; ruling 2)
    The same ``X``, ``F``, ``Delta`` that give the ex-ante decomposition give
    the ex-post attribution. Two assertions:

    (i) RETURNS -- on every day of every holding period, with the design ``X_i``
        of the forecast standing at that period's rebalance and the drifted
        weights ``h_{t-1}`` the engine held at the previous close,
        ``r_p,t = sum_k x_{t,k} f_{t,k} + specific_t`` is an IDENTITY with the
        specific return defined as the remainder. Asserted to 1e-12 of scale
        (the form of :func:`mafrm.backtest.metrics.identity_terms`).
    (ii) VARIANCE -- the ex-ante shares are forecasts and the ex-post shares are
        realisations, so they differ by exactly what ``B`` measures. Each
        additive-in-variance component's ex-post/ex-ante variance ratio, square
        rooted, is SPEC.md 6.1's statistic for that component and is scored
        against the SAME exact chi-square interval ``B`` uses at the same sample
        size (``validation.chi_square_level``; [0.898, 1.101] at 187 months).
        SPEC.md 10.3's own example -- a factor at 60% of ex-ante variance that
        contributes 5% ex post -- gives ``sqrt(5/60) = 0.29`` and fires. No new
        constant. ``strict=True`` RAISES (tests, corrupted input); the real run
        writes ratio and interval and marks the cell.

    Ruling 3: the Brinson-Fachler sleeve-level check is against the EQUAL-WEIGHT
    book -- the absence of a choice and the anchor ``TE_target`` is sized on.
    ``A_i = (w_i - W_i)(b_i - b)``, the ``- b`` penalising an overweight to a
    sleeve that underperformed the whole benchmark (Brinson-Fachler, not BHB);
    selection ``W_i (b_i^p - b_i)`` and interaction ``(w_i - W_i)(b_i^p - b_i)``
    complete the identity, which is asserted per period. Multi-period linking
    is Carino's logarithmic scaling, order-independent, and the linked effects
    are asserted to sum to the cumulative active return. ``w^b = 0`` stays the
    RISK benchmark in the identity; the equal-weight book is the ATTRIBUTION
    benchmark; the two are different objects. The benchmark's return series is
    reported and its Sharpe is NEVER computed (a benchmark that acquires one
    becomes a trial); ``tests/test_diagnostics.py`` asserts this module never
    calls a Sharpe.

    Ex-post exceeding ex-ante is EXPECTED, not a bug (Hwang & Satchell 2001):
    ex-ante conditions on FIXED weights, ex-post weights drift between
    rebalances and the extra terms are non-negative. Both ex-post variances --
    at the fixed rebalance weights and at the drifted weights -- are reported,
    so the drift term is a computed line.

UNITS. Returns are decimal per day or per period; variances per period; costs
are fractions of ``NAV_pre``. Nothing here reads data or knows an asset class.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mafrm.risk import validation

__all__ = [
    "Attribution",
    "CellDiagnostics",
    "ComponentBias",
    "DiagnosticsError",
    "RiskForecast",
    "TrackingErrorDecomposition",
    "assert_variance_additive",
    "attribute",
    "brinson_fachler",
    "carino_link",
    "cell_diagnostics",
    "component_bias",
    "decompose_tracking_error",
    "implementation_shortfall",
    "reconcile",
    "render",
]

FloatArray = NDArray[np.float64]

#: The identity tolerance, the form of :func:`mafrm.backtest.metrics.identity_terms`.
_IDENTITY_TOLERANCE = 1e-12


class DiagnosticsError(ValueError):
    """An identity failed or an input cannot carry the statistic. Nothing is filled."""


class RiskForecast(Protocol):
    """``(X, F, delta^2)`` for one rebalance, per-period units; :class:`mafrm.backtest.grid.Forecast`."""

    @property
    def exposures(self) -> FloatArray: ...

    @property
    def factor_covariance(self) -> FloatArray: ...

    @property
    def specific_variance(self) -> FloatArray: ...


def _vector(values: object, *, name: str, length: int | None = None) -> FloatArray:
    out = np.asarray(values, dtype=float)
    if out.ndim != 1:
        raise DiagnosticsError(f"{name}: expected a vector, got shape {out.shape}")
    if length is not None and out.shape != (length,):
        raise DiagnosticsError(f"{name}: expected {length} entries, got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise DiagnosticsError(f"{name}: every entry must be finite")
    return out


# ---------------------------------------------------------------------------
# SPEC.md 10.1 -- Perold
# ---------------------------------------------------------------------------


def implementation_shortfall(
    *,
    nav_before_costs: pd.Series,
    spread_cost: pd.DataFrame,
    impact_cost: pd.DataFrame,
    asymmetry_cost: pd.DataFrame,
    turnover: pd.Series,
    predicted: pd.Series,
    commission_rate: float = 0.0,
) -> pd.DataFrame:
    """Perold's four legs per parent order (one rebalance), as fractions of ``NAV_pre``.

    ``delay`` and ``opportunity`` are zero by construction (ruling 5: decision
    price = fill price, targets hit exactly); ``impact`` is SPEC.md 7.1's three
    terms charged by the engine; ``fees`` is the commission on traded notional
    (``costs.commission_bps``, an ABSENCE in this repository). The decomposition
    is asserted to add to the realised cost, and ``residual = realised -
    predicted`` with ``predicted`` the cost the optimizer priced at the decision.
    """
    index = pd.DatetimeIndex(nav_before_costs.index)
    nav = _vector(nav_before_costs.to_numpy(dtype=float), name="nav_before_costs")
    if np.any(nav <= 0.0):
        raise DiagnosticsError("implementation_shortfall: NAV_pre must be positive")
    n = len(index)
    spread = _vector(spread_cost.reindex(index).sum(axis=1).to_numpy(), name="spread", length=n)
    impact = _vector(impact_cost.reindex(index).sum(axis=1).to_numpy(), name="impact", length=n)
    asym = _vector(asymmetry_cost.reindex(index).sum(axis=1).to_numpy(), name="asym", length=n)
    turn = _vector(turnover.reindex(index).to_numpy(), name="turnover", length=n)
    priced = _vector(predicted.reindex(index).to_numpy(), name="predicted", length=n)
    if not (math.isfinite(commission_rate) and commission_rate >= 0.0):
        raise DiagnosticsError(f"commission_rate must be >= 0, got {commission_rate}")
    frame = pd.DataFrame(index=index)
    frame["paper_cost"] = 0.0
    frame["delay"] = 0.0
    frame["impact_spread"] = spread / nav
    frame["impact_market"] = impact / nav
    frame["impact_asymmetry"] = asym / nav
    frame["impact"] = frame["impact_spread"] + frame["impact_market"] + frame["impact_asymmetry"]
    frame["opportunity"] = 0.0
    frame["fees"] = commission_rate * turn
    frame["shortfall"] = frame["delay"] + frame["impact"] + frame["opportunity"] + frame["fees"]
    frame["realised"] = (spread + impact + asym) / nav + commission_rate * turn
    frame["predicted"] = priced
    frame["residual"] = frame["realised"] - frame["predicted"]
    gap = np.abs(frame["shortfall"].to_numpy() - frame["realised"].to_numpy())
    scale = np.maximum(np.abs(frame["realised"].to_numpy()), 1.0)
    if np.any(gap > _IDENTITY_TOLERANCE * scale):
        raise DiagnosticsError(  # pragma: no cover - the identity is exact
            "implementation_shortfall: the four legs do not add to the realised cost"
        )
    frame.index.name = "date"
    return frame


# ---------------------------------------------------------------------------
# SPEC.md 10.2 -- tracking error, additive in variance; x-sigma-rho
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackingErrorDecomposition:
    """``sigma_a^2 = x_a' F x_a + h' Delta h`` and Menchero & Davis's ``x sigma rho``.

    The VARIANCES add; the volatilities do not, and :func:`assert_variance_additive`
    refuses an instance whose volatilities add.
    """

    total_variance: float
    factor_variance: float
    specific_variance: float
    #: ``x_a = X' h``.
    active_exposures: FloatArray
    #: ``sigma_k = sqrt(F_kk)``.
    factor_volatilities: FloatArray
    #: ``rho_k = (F x)_k / (sigma_k sigma_a)``; zero where either volatility is zero.
    factor_correlations: FloatArray
    #: ``CTR_k = x_k sigma_k rho_k``, in volatility units.
    factor_contributions: FloatArray
    #: ``h' Delta h / sigma_a``.
    specific_contribution: float

    def __post_init__(self) -> None:
        assert_variance_additive(self)
        total = self.total_volatility
        summed = float(self.factor_contributions.sum()) + self.specific_contribution
        if abs(summed - total) > _IDENTITY_TOLERANCE * max(total, 1.0):
            raise DiagnosticsError(  # pragma: no cover - Euler's identity is exact
                f"x-sigma-rho contributions {summed} do not sum to sigma_a {total}"
            )

    @property
    def total_volatility(self) -> float:
        return math.sqrt(max(self.total_variance, 0.0))

    @property
    def factor_volatility(self) -> float:
        """``sqrt(x' F x)``. Does NOT add to ``specific_volatility``; see the module docstring."""
        return math.sqrt(max(self.factor_variance, 0.0))

    @property
    def specific_volatility(self) -> float:
        return math.sqrt(max(self.specific_variance, 0.0))

    @property
    def factor_share(self) -> float:
        """Share of VARIANCE, the quantity that adds."""
        return (
            self.factor_variance / self.total_variance
            if self.total_variance > 0.0
            else float("nan")
        )

    @property
    def contribution_shares(self) -> FloatArray:
        """``CTR_k / sigma_a``; negative for a diversifying exposure."""
        total = self.total_volatility
        if total <= 0.0:
            return np.zeros_like(self.factor_contributions)
        out: FloatArray = self.factor_contributions / total
        return out


def assert_variance_additive(decomposition: TrackingErrorDecomposition) -> None:
    """Refuse a decomposition that is not additive in variance, or whose volatilities add.

    SPEC.md 10.2: "Reporting a 'factor TE' and a 'specific TE' that sum to
    total TE is wrong." With both components positive, ``sqrt(a) + sqrt(b) >
    sqrt(a + b)`` strictly, so a pair of volatilities that adds to the total
    cannot have come from this decomposition.
    """
    d = decomposition
    for name, value in (
        ("total_variance", d.total_variance),
        ("factor_variance", d.factor_variance),
        ("specific_variance", d.specific_variance),
    ):
        if not (math.isfinite(value) and value >= 0.0):
            raise DiagnosticsError(f"{name} must be a finite non-negative variance, got {value}")
    gap = abs(d.factor_variance + d.specific_variance - d.total_variance)
    if gap > _IDENTITY_TOLERANCE * max(d.total_variance, 1e-300):
        raise DiagnosticsError(
            f"tracking error is additive in VARIANCE: {d.factor_variance} + "
            f"{d.specific_variance} != {d.total_variance}"
        )
    if d.factor_variance > 0.0 and d.specific_variance > 0.0:
        summed = math.sqrt(d.factor_variance) + math.sqrt(d.specific_variance)
        if not summed > math.sqrt(d.total_variance) * (1.0 + _IDENTITY_TOLERANCE):
            raise DiagnosticsError(  # pragma: no cover - unreachable once the variances add
                "a 'factor TE' and a 'specific TE' that sum to the total TE are not this "
                "decomposition (SPEC.md 10.2)"
            )


def decompose_tracking_error(
    weights: FloatArray,
    exposures: FloatArray,
    factor_covariance: FloatArray,
    specific_variance: FloatArray,
    *,
    benchmark: FloatArray | None = None,
) -> TrackingErrorDecomposition:
    """SPEC.md 10.2 on one rebalance. ``benchmark=None`` is ``w^b = 0``, absolute risk."""
    x_design = np.asarray(exposures, dtype=float)
    if x_design.ndim != 2:
        raise DiagnosticsError(f"exposures must be N x K, got {x_design.shape}")
    n, k = x_design.shape
    w = _vector(weights, name="weights", length=n)
    h = w - (_vector(benchmark, name="benchmark", length=n) if benchmark is not None else 0.0)
    f = np.asarray(factor_covariance, dtype=float)
    if f.shape != (k, k):
        raise DiagnosticsError(f"factor_covariance is {f.shape} against K = {k}")
    delta = _vector(specific_variance, name="specific_variance", length=n)
    if np.any(delta < 0.0):
        raise DiagnosticsError("specific_variance has a negative entry")
    x = x_design.T @ h
    fx = f @ x
    factor_var = float(x @ fx)
    specific_var = float(h @ (delta * h))
    if factor_var < 0.0:
        if factor_var < -_IDENTITY_TOLERANCE * max(abs(specific_var), 1e-300):
            raise DiagnosticsError(f"x' F x = {factor_var} < 0: F is not PSD")
        factor_var = 0.0
    total_var = factor_var + specific_var
    sigma_a = math.sqrt(total_var)
    sigma_k = np.sqrt(np.clip(np.diag(f), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where((sigma_k > 0.0) & (sigma_a > 0.0), fx / (sigma_k * sigma_a), 0.0)
    ctr = x * sigma_k * rho
    specific_ctr = specific_var / sigma_a if sigma_a > 0.0 else 0.0
    return TrackingErrorDecomposition(
        total_variance=total_var,
        factor_variance=factor_var,
        specific_variance=specific_var,
        active_exposures=np.asarray(x, dtype=float),
        factor_volatilities=np.asarray(sigma_k, dtype=float),
        factor_correlations=np.asarray(rho, dtype=float),
        factor_contributions=np.asarray(ctr, dtype=float),
        specific_contribution=float(specific_ctr),
    )


# ---------------------------------------------------------------------------
# SPEC.md 10.3 -- attribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Attribution:
    """Per holding period: the return identity's pieces and the ex-ante variances beside them.

    ``periods`` columns: ``portfolio_return`` (the arithmetic sum of the daily
    gross returns at the DRIFTED weights), ``factor_return``, ``specific_return``
    (the remainder), one ``contrib_<factor>`` per factor, ``fixed_weight_return``
    (the same days at the rebalance weights held fixed), ``buy_and_hold_return``
    (the target weights times the assets' period returns, the Brinson
    portfolio), ``benchmark_return`` (the equal-weight book's period return),
    ``ex_ante_factor_variance``, ``ex_ante_specific_variance``,
    ``ex_ante_total_variance``.
    """

    periods: pd.DataFrame
    factor_names: tuple[str, ...]
    #: Per period, the Brinson-Fachler table by sleeve (``brinson_fachler``'s frame).
    brinson: dict[pd.Timestamp, pd.DataFrame] = field(default_factory=dict)


def attribute(
    *,
    prices: pd.DataFrame,
    weights_daily: pd.DataFrame,
    targets: pd.DataFrame,
    terminal_date: pd.Timestamp,
    forecasts: Sequence[RiskForecast],
    factor_returns: pd.DataFrame,
    sleeves: Mapping[str, str] | None = None,
) -> Attribution:
    """SPEC.md 10.3's return identity on every day, aggregated to holding periods.

    ``prices`` is the engine's daily panel, ``weights_daily`` the engine's
    drifted weights at each close, ``targets`` the rebalance weights,
    ``forecasts`` the ``(X, F, delta^2)`` standing at each rebalance (per-period
    units), ``factor_returns`` the daily realised factor returns (decimal) on
    the price dates with ``K`` columns in the forecasts' factor order.
    """
    assets = list(prices.columns)
    n = len(assets)
    rebalance_dates = pd.DatetimeIndex(targets.index)
    r = len(rebalance_dates)
    if len(forecasts) != r:
        raise DiagnosticsError(f"attribute: {len(forecasts)} forecasts against {r} rebalances")
    if list(targets.columns) != assets or list(weights_daily.columns) != assets:
        raise DiagnosticsError(
            "attribute: targets and weights must carry the price assets in order"
        )
    k = int(np.asarray(forecasts[0].exposures).shape[1]) if r else 0
    if factor_returns.shape[1] != k:
        raise DiagnosticsError(
            f"attribute: factor_returns has {factor_returns.shape[1]} columns against K = {k}"
        )
    names = tuple(str(c) for c in factor_returns.columns)
    marks = rebalance_dates.append(pd.DatetimeIndex([pd.Timestamp(terminal_date)]))
    if not marks.is_monotonic_increasing or marks[-1] <= marks[-2]:
        raise DiagnosticsError("attribute: the terminal date must follow the last rebalance")
    price_values = prices.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(prices.index)
    daily = price_values[1:] / price_values[:-1] - 1.0  # return on dates[1:]
    day_index = pd.DatetimeIndex(dates[1:])
    weights_prev = weights_daily.reindex(dates).to_numpy(dtype=float)[:-1]
    factors = factor_returns.reindex(day_index).to_numpy(dtype=float)
    if not np.all(np.isfinite(factors)):
        raise DiagnosticsError("attribute: a factor return is missing on a price date")
    if not np.all(np.isfinite(weights_prev)):
        raise DiagnosticsError("attribute: a drifted weight is missing on a price date")
    sleeve_of = dict(sleeves) if sleeves is not None else {a: "all" for a in assets}
    sleeve_list = [str(sleeve_of[a]) for a in assets]
    benchmark_weights = np.full(n, 1.0 / n)

    rows: list[dict[str, object]] = []
    brinson: dict[pd.Timestamp, pd.DataFrame] = {}
    for i in range(r):
        start, end = marks[i], marks[i + 1]
        mask = (day_index > start) & (day_index <= end)
        if not mask.any():
            raise DiagnosticsError(f"attribute: no trading day in ({start.date()}, {end.date()}]")
        ret = daily[mask]
        w_prev = weights_prev[mask]
        f = factors[mask]
        forecast = forecasts[i]
        x_design = np.asarray(forecast.exposures, dtype=float)
        w_target = targets.iloc[i].to_numpy(dtype=float)
        portfolio = (w_prev * ret).sum(axis=1)
        exposures_daily = w_prev @ x_design  # T x K
        contributions = exposures_daily * f
        factor_part = contributions.sum(axis=1)
        specific = portfolio - factor_part
        # (i) the return identity, per day, to 1e-12 of scale.
        gap = np.abs(factor_part + specific - portfolio)
        if np.any(gap > _IDENTITY_TOLERANCE * np.maximum(np.abs(portfolio), 1.0)):
            raise DiagnosticsError(  # pragma: no cover - the remainder closes by construction
                f"attribute: the return identity fails in the period ending {end.date()}"
            )
        fixed = ret @ w_target
        period_asset = price_values[dates.get_loc(end)] / price_values[dates.get_loc(start)] - 1.0
        buy_and_hold = float(w_target @ period_asset)
        benchmark = float(period_asset.mean())
        decomposition = decompose_tracking_error(
            w_target,
            x_design,
            np.asarray(forecast.factor_covariance, dtype=float),
            np.asarray(forecast.specific_variance, dtype=float),
        )
        row: dict[str, object] = {
            "end": end,
            "start": start,
            "days": int(mask.sum()),
            "portfolio_return": float(portfolio.sum()),
            "factor_return": float(factor_part.sum()),
            "specific_return": float(specific.sum()),
            "fixed_weight_return": float(fixed.sum()),
            "buy_and_hold_return": buy_and_hold,
            "benchmark_return": benchmark,
            "ex_ante_factor_variance": decomposition.factor_variance,
            "ex_ante_specific_variance": decomposition.specific_variance,
            "ex_ante_total_variance": decomposition.total_variance,
        }
        for j, name in enumerate(names):
            row[f"contrib_{name}"] = float(contributions[:, j].sum())
        rows.append(row)
        brinson[pd.Timestamp(end)] = brinson_fachler(
            w_target, benchmark_weights, period_asset, sleeve_list
        )
    periods = pd.DataFrame(rows).set_index("end")
    return Attribution(periods=periods, factor_names=names, brinson=brinson)


@dataclass(frozen=True)
class ComponentBias:
    """One additive-in-variance component's ex-post/ex-ante ratio against the exact interval."""

    name: str
    ex_post_variance: float
    ex_ante_variance: float
    #: ``sqrt(ex_post / ex_ante)``; NaN when the component is absent ex ante.
    bias: float
    lower: float
    upper: float
    periods: int

    @property
    def defined(self) -> bool:
        return math.isfinite(self.bias)

    @property
    def inside(self) -> bool:
        """Inside the interval; an undefined ratio (absent component) is not a refutation."""
        return (not self.defined) or (self.lower <= self.bias <= self.upper)


def component_bias(
    ex_post: FloatArray, ex_ante_variance: FloatArray, *, name: str, level: float
) -> ComponentBias:
    """``B_component = sqrt(var(ex_post) / mean(ex_ante_variance))`` against the chi-square interval.

    ``ex_post`` is the component's realised return per period; ``ex_ante_variance``
    its forecast variance per period. The interval is SPEC.md 6.1's exact one at
    the number of periods. An ex-ante variance of zero (the component is not in
    the model, as specific risk is not in a dense variant) gives NaN, reported
    as undefined rather than as a pass or a failure.
    """
    post = _vector(ex_post, name=f"{name}: ex_post")
    ante = _vector(ex_ante_variance, name=f"{name}: ex_ante_variance", length=post.size)
    if post.size < 2:
        raise DiagnosticsError(f"{name}: at least two periods are needed")
    if np.any(ante < 0.0):
        raise DiagnosticsError(f"{name}: a forecast variance is negative")
    interval = validation.chi_square_interval(post.size, level=level)
    mean_ante = float(ante.mean())
    var_post = float(np.var(post, ddof=1))
    bias = math.sqrt(var_post / mean_ante) if mean_ante > 0.0 else float("nan")
    return ComponentBias(
        name=name,
        ex_post_variance=var_post,
        ex_ante_variance=mean_ante,
        bias=bias,
        lower=float(interval.lower),
        upper=float(interval.upper),
        periods=int(post.size),
    )


def reconcile(
    attribution: Attribution, *, level: float, strict: bool = False
) -> dict[str, ComponentBias]:
    """Assertion (ii): the factor, specific and total components against the interval.

    ``strict=True`` raises :class:`DiagnosticsError` on any defined component
    outside the interval -- the behaviour on corrupted input and in tests. The
    real run reports and marks (ruling 2).
    """
    periods = attribution.periods
    out = {
        "factor": component_bias(
            periods["factor_return"].to_numpy(dtype=float),
            periods["ex_ante_factor_variance"].to_numpy(dtype=float),
            name="factor",
            level=level,
        ),
        "specific": component_bias(
            periods["specific_return"].to_numpy(dtype=float),
            periods["ex_ante_specific_variance"].to_numpy(dtype=float),
            name="specific",
            level=level,
        ),
        "total": component_bias(
            periods["portfolio_return"].to_numpy(dtype=float),
            periods["ex_ante_total_variance"].to_numpy(dtype=float),
            name="total",
            level=level,
        ),
    }
    if strict:
        outside = [c for c in out.values() if not c.inside]
        if outside:
            worst = outside[0]
            raise DiagnosticsError(
                f"attribution does not reconcile with the risk model: the {worst.name} "
                f"component's ex-post/ex-ante bias is {worst.bias:.3f} against the exact "
                f"interval [{worst.lower:.3f}, {worst.upper:.3f}] at {worst.periods} periods "
                "(SPEC.md 10.3: the same X, F, Delta must generate both)"
            )
    return out


def brinson_fachler(
    weights: FloatArray,
    benchmark_weights: FloatArray,
    asset_returns: FloatArray,
    sleeves: Sequence[str],
) -> pd.DataFrame:
    """Brinson-Fachler by sleeve for one period; the effects add to ``r_p - r_b`` exactly.

    ``A_i = (w_i - W_i)(b_i - b)``, ``S_i = W_i (b_i^p - b_i)``,
    ``I_i = (w_i - W_i)(b_i^p - b_i)``. A sleeve the portfolio does not hold has
    ``b_i^p := b_i`` (nothing selected), so its selection and interaction are
    zero and its allocation carries the whole effect.
    """
    n = len(sleeves)
    w = _vector(weights, name="weights", length=n)
    wb = _vector(benchmark_weights, name="benchmark_weights", length=n)
    ret = _vector(asset_returns, name="asset_returns", length=n)
    if abs(w.sum() - wb.sum()) > 1e-9:
        raise DiagnosticsError("brinson_fachler: portfolio and benchmark weights must sum alike")
    total_b = float(wb @ ret)
    total_p = float(w @ ret)
    names = list(dict.fromkeys(str(s) for s in sleeves))
    rows = []
    for name in names:
        members = np.array([str(s) == name for s in sleeves])
        w_i, wb_i = float(w[members].sum()), float(wb[members].sum())
        b_i = float(wb[members] @ ret[members] / wb_i) if wb_i > 0.0 else float("nan")
        if not math.isfinite(b_i):
            raise DiagnosticsError(f"brinson_fachler: benchmark weight zero in sleeve {name!r}")
        bp_i = float(w[members] @ ret[members] / w_i) if w_i > 0.0 else b_i
        allocation = (w_i - wb_i) * (b_i - total_b)
        selection = wb_i * (bp_i - b_i)
        interaction = (w_i - wb_i) * (bp_i - b_i)
        rows.append(
            {
                "sleeve": name,
                "portfolio_weight": w_i,
                "benchmark_weight": wb_i,
                "portfolio_return": bp_i,
                "benchmark_return": b_i,
                "allocation": allocation,
                "selection": selection,
                "interaction": interaction,
                "total": allocation + selection + interaction,
            }
        )
    frame = pd.DataFrame(rows).set_index("sleeve")
    active = total_p - total_b
    if abs(float(frame["total"].sum()) - active) > _IDENTITY_TOLERANCE * max(abs(active), 1.0):
        raise DiagnosticsError(  # pragma: no cover - the identity is exact
            "brinson_fachler: the effects do not add to the active return"
        )
    return frame


def _carino_k(portfolio: float, benchmark: float) -> float:
    if portfolio <= -1.0 or benchmark <= -1.0:
        raise DiagnosticsError("carino: a return of -100% or worse cannot be linked")
    if portfolio == benchmark:
        return 1.0 / (1.0 + portfolio)
    return (math.log1p(portfolio) - math.log1p(benchmark)) / (portfolio - benchmark)


def carino_link(
    effects: pd.DataFrame, portfolio_returns: FloatArray, benchmark_returns: FloatArray
) -> pd.Series:
    """Carino's logarithmic linking: ``sum_t (k_t / k) effect_t`` per column.

    ``k_t = (ln(1+r_p,t) - ln(1+r_b,t)) / (r_p,t - r_b,t)`` and ``k`` the same on
    the cumulative returns; order-independent, and the linked effects add to
    the cumulative active return ``R_p - R_b`` exactly when the per-period
    effects add to ``r_p,t - r_b,t``, which is asserted.
    """
    rp = _vector(portfolio_returns, name="portfolio_returns")
    rb = _vector(benchmark_returns, name="benchmark_returns", length=rp.size)
    if len(effects) != rp.size:
        raise DiagnosticsError("carino_link: effects and returns must align")
    per_period = effects.to_numpy(dtype=float).sum(axis=1)
    if np.any(np.abs(per_period - (rp - rb)) > 1e-9 * np.maximum(np.abs(rp - rb), 1.0)):
        raise DiagnosticsError("carino_link: the per-period effects do not add to r_p - r_b")
    cum_p = float(np.prod(1.0 + rp) - 1.0)
    cum_b = float(np.prod(1.0 + rb) - 1.0)
    k = _carino_k(cum_p, cum_b)
    weights = np.array([_carino_k(p, b) for p, b in zip(rp, rb, strict=True)]) / k
    linked = pd.Series(effects.mul(weights, axis=0).sum(axis=0), name="linked")
    total = float(linked.sum())
    if abs(total - (cum_p - cum_b)) > 1e-9 * max(abs(cum_p - cum_b), 1.0):
        raise DiagnosticsError(  # pragma: no cover - Carino's identity is exact
            f"carino_link: linked effects {total} do not add to R_p - R_b {cum_p - cum_b}"
        )
    return linked


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellDiagnostics:
    """Everything SPEC.md 10.1-10.3 produced on one cell, plus a flat summary for the tables."""

    shortfall: pd.DataFrame
    tracking_error: pd.DataFrame
    attribution: Attribution | None
    reconciliation: dict[str, ComponentBias]
    #: Carino-linked Brinson-Fachler effects by sleeve (rows) and effect (columns).
    brinson_linked: pd.DataFrame | None
    summary: dict[str, float]
    #: Whether any DEFINED component fell outside the interval on this cell.
    reconciliation_flag: bool


def cell_diagnostics(
    *,
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    terminal_date: pd.Timestamp,
    forecasts: Sequence[RiskForecast],
    nav_before_costs: pd.Series,
    spread_cost: pd.DataFrame,
    impact_cost: pd.DataFrame,
    asymmetry_cost: pd.DataFrame,
    weights_daily: pd.DataFrame,
    turnover: pd.Series,
    predicted_cost: pd.Series,
    factor_returns: pd.DataFrame | None,
    sleeves: Mapping[str, str] | None,
    periods_per_year: float,
    level: float,
    commission_rate: float = 0.0,
    strict: bool = False,
    per_factor_summary: bool = True,
) -> CellDiagnostics:
    """Run all three diagnostics on one cell. ``factor_returns=None`` skips the attribution.

    ``per_factor_summary=False`` keeps the per-factor contribution columns out of
    the flat summary (a dense variant's "factors" are its assets).
    """
    shortfall = implementation_shortfall(
        nav_before_costs=nav_before_costs,
        spread_cost=spread_cost,
        impact_cost=impact_cost,
        asymmetry_cost=asymmetry_cost,
        turnover=turnover,
        predicted=predicted_cost,
        commission_rate=commission_rate,
    )
    names: tuple[str, ...] = (
        tuple(str(c) for c in factor_returns.columns)
        if factor_returns is not None
        else tuple(f"f{j}" for j in range(int(np.asarray(forecasts[0].exposures).shape[1])))
    )
    te_rows = []
    for i, stamp in enumerate(pd.DatetimeIndex(targets.index)):
        forecast = forecasts[i]
        d = decompose_tracking_error(
            targets.iloc[i].to_numpy(dtype=float),
            np.asarray(forecast.exposures, dtype=float),
            np.asarray(forecast.factor_covariance, dtype=float),
            np.asarray(forecast.specific_variance, dtype=float),
        )
        row: dict[str, object] = {
            "date": stamp,
            "total_variance": d.total_variance,
            "factor_variance": d.factor_variance,
            "specific_variance": d.specific_variance,
            "factor_share": d.factor_share,
            "total_volatility": d.total_volatility,
            "factor_volatility": d.factor_volatility,
            "specific_volatility": d.specific_volatility,
            "specific_contribution_share": (
                d.specific_contribution / d.total_volatility if d.total_volatility > 0 else np.nan
            ),
        }
        shares = d.contribution_shares
        for j, name in enumerate(names):
            row[f"x_{name}"] = float(d.active_exposures[j])
            row[f"rho_{name}"] = float(d.factor_correlations[j])
            row[f"ctr_share_{name}"] = float(shares[j])
        te_rows.append(row)
    tracking = pd.DataFrame(te_rows).set_index("date")

    ppy = periods_per_year
    summary: dict[str, float] = {
        "perold_decision_price_is_fill_price": 1.0,
        "perold_delay_bps_per_year": float(shortfall["delay"].mean() * ppy * 1e4),
        "perold_impact_bps_per_year": float(shortfall["impact"].mean() * ppy * 1e4),
        "perold_opportunity_bps_per_year": float(shortfall["opportunity"].mean() * ppy * 1e4),
        "perold_fees_bps_per_year": float(shortfall["fees"].mean() * ppy * 1e4),
        "perold_shortfall_bps_per_year": float(shortfall["shortfall"].mean() * ppy * 1e4),
        "perold_predicted_bps_per_year": float(shortfall["predicted"].mean() * ppy * 1e4),
        "perold_residual_bps_per_year": float(shortfall["residual"].mean() * ppy * 1e4),
        "perold_residual_abs_mean_bps": float(shortfall["residual"].abs().mean() * 1e4),
        "perold_residual_max_abs_bps": float(shortfall["residual"].abs().max() * 1e4),
        "te_factor_variance_share_mean": float(tracking["factor_share"].mean()),
        "te_specific_variance_share_mean": float(1.0 - tracking["factor_share"].mean()),
        "te_negative_contribution_share": float(
            (tracking[[f"ctr_share_{n}" for n in names]] < 0.0).to_numpy().mean()
        )
        if names
        else 0.0,
        "te_volatility_sum_over_total_mean": float(
            (
                (tracking["factor_volatility"] + tracking["specific_volatility"])
                / tracking["total_volatility"]
            ).mean()
        ),
    }
    if per_factor_summary:
        for name in names:
            summary[f"ctr_share_mean_{name}"] = float(tracking[f"ctr_share_{name}"].mean())

    attribution: Attribution | None = None
    reconciliation: dict[str, ComponentBias] = {}
    linked: pd.DataFrame | None = None
    flag = False
    if factor_returns is not None:
        attribution = attribute(
            prices=prices,
            weights_daily=weights_daily,
            targets=targets,
            terminal_date=terminal_date,
            forecasts=forecasts,
            factor_returns=factor_returns,
            sleeves=sleeves,
        )
        reconciliation = reconcile(attribution, level=level, strict=strict)
        flag = any(not c.inside for c in reconciliation.values())
        periods = attribution.periods
        for key, comp in reconciliation.items():
            summary[f"attribution_bias_{key}"] = comp.bias
            summary[f"attribution_{key}_inside"] = float(comp.inside)
        first = next(iter(reconciliation.values()))
        summary["attribution_interval_lower"] = first.lower
        summary["attribution_interval_upper"] = first.upper
        summary["attribution_periods"] = float(first.periods)
        summary["attribution_reconciliation_flag"] = float(flag)
        # Hwang & Satchell: fixed weights versus drifted weights, per-period variances.
        var_fixed = float(np.var(periods["fixed_weight_return"].to_numpy(dtype=float), ddof=1))
        var_drift = float(np.var(periods["portfolio_return"].to_numpy(dtype=float), ddof=1))
        ante = float(periods["ex_ante_total_variance"].mean())
        summary["ex_ante_rms_volatility_per_period"] = math.sqrt(ante)
        summary["ex_post_volatility_fixed_weights_per_period"] = math.sqrt(var_fixed)
        summary["ex_post_volatility_drifted_weights_per_period"] = math.sqrt(var_drift)
        summary["ex_post_over_ex_ante_fixed_weights"] = (
            math.sqrt(var_fixed / ante) if ante > 0 else float("nan")
        )
        summary["ex_post_over_ex_ante_drifted_weights"] = (
            math.sqrt(var_drift / ante) if ante > 0 else float("nan")
        )
        summary["hwang_satchell_drift_variance_share"] = (
            (var_drift - var_fixed) / var_drift if var_drift > 0 else float("nan")
        )
        # Brinson-Fachler, Carino-linked over the periods.
        stamps = list(periods.index)
        sleeves_seen = list(attribution.brinson[pd.Timestamp(stamps[0])].index)
        effects = pd.DataFrame(
            {
                f"{sleeve}|{effect}": [
                    float(attribution.brinson[pd.Timestamp(s)].loc[sleeve, effect]) for s in stamps
                ]
                for sleeve in sleeves_seen
                for effect in ("allocation", "selection", "interaction")
            },
            index=periods.index,
        )
        linked_series = carino_link(
            effects,
            periods["buy_and_hold_return"].to_numpy(dtype=float),
            periods["benchmark_return"].to_numpy(dtype=float),
        )
        linked = pd.DataFrame(
            {
                effect: [float(linked_series[f"{sleeve}|{effect}"]) for sleeve in sleeves_seen]
                for effect in ("allocation", "selection", "interaction")
            },
            index=pd.Index(sleeves_seen, name="sleeve"),
        )
        linked["total"] = linked.sum(axis=1)
        cum_p = float(np.prod(1.0 + periods["buy_and_hold_return"].to_numpy(dtype=float)) - 1.0)
        cum_b = float(np.prod(1.0 + periods["benchmark_return"].to_numpy(dtype=float)) - 1.0)
        summary["brinson_cumulative_portfolio_return"] = cum_p
        summary["brinson_cumulative_benchmark_return"] = cum_b
        summary["brinson_linked_allocation"] = float(linked["allocation"].sum())
        summary["brinson_linked_selection"] = float(linked["selection"].sum())
        summary["brinson_linked_interaction"] = float(linked["interaction"].sum())
    return CellDiagnostics(
        shortfall=shortfall,
        tracking_error=tracking,
        attribution=attribution,
        reconciliation=reconciliation,
        brinson_linked=linked,
        summary=summary,
        reconciliation_flag=flag,
    )


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticsEntry:
    cell_id: str
    label: str
    treatment: str
    regime: str
    diagnostics: CellDiagnostics


def _num(value: float, digits: int = 3) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def _pct(value: float, digits: int = 1) -> str:
    return "n/a" if not np.isfinite(value) else f"{100.0 * value:.{digits}f}%"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def render(entries: Sequence[DiagnosticsEntry], *, level: float, trials: int) -> str:
    """``reports/diagnostics.md``: the three SPEC.md 10 diagnostics across the cells."""
    lines = [
        "# SPEC.md 10.1-10.3's diagnostics on every grid cell",
        "",
        "Generated by `python -m mafrm.backtest.grid`. W6-P3, under operator rulings 2, 3 and 5 of 2026-09-04 (SPEC.md 9.3). One row per run; treatment D at both `Y` regimes. `N` at run time was "
        f"**{trials}**; nothing here is a trial.",
        "",
        "## 10.1 Perold implementation shortfall, at the parent level",
        "",
        "**The decomposition is degenerate, and that is the finding.** The build generates its own trade list, so the decision timestamp is owned: the decision price is the rebalance close, the parent order is filled at that close in full (the engine hits the targets exactly), and therefore `delay = 0` and `opportunity = 0` BY CONSTRUCTION and `IS = impact + fees` exactly, with `fees` the commission on traded notional -- an absence in this repository (`costs.commission_bps` = 0, no disclosed figure). The full four-way Perold decomposition that vendor TCA usually cannot do -- because the decision timestamp is not reliably logged -- is realisable here only with an intra-period execution schedule that does not exist; an earlier decision timestamp would be invented. `impact` is SPEC.md 7.1's three terms as charged. The predicted-versus-realised residual is `realised - predicted`, with `predicted` the cost the optimizer priced in its objective at the decision: zero under A (nothing priced, nothing charged); under B nothing was priced, so the residual is the realised cost itself and carries no calibration information; the measurement is C (a flat 2.5bp half-spread against the issuer level plus EDGE widening, on the same trade) and D (the as-of spread the optimizer saw against the as-of spread the engine charged on the same trade). All in bp of NAV per year (mean per rebalance x periods per year).",
        "",
    ]
    rows = []
    for e in entries:
        s = e.diagnostics.summary
        rows.append(
            [
                e.cell_id,
                e.regime,
                _num(s["perold_delay_bps_per_year"], 1),
                _num(s["perold_impact_bps_per_year"], 1),
                _num(s["perold_opportunity_bps_per_year"], 1),
                _num(s["perold_fees_bps_per_year"], 1),
                _num(s["perold_shortfall_bps_per_year"], 1),
                _num(s["perold_predicted_bps_per_year"], 1),
                _num(s["perold_residual_bps_per_year"], 1),
                _num(s["perold_residual_max_abs_bps"], 2),
            ]
        )
    lines += _table(
        [
            "cell",
            "Y",
            "delay",
            "impact",
            "opportunity",
            "fees",
            "IS = shortfall",
            "predicted (priced)",
            "residual (realised - predicted)",
            "max |residual| per rebalance, bp",
        ],
        rows,
    )
    lines += [
        "",
        "## 10.2 Tracking error: additive in variance, x-sigma-rho contributions",
        "",
        "`sigma_a^2 = x_a' F x_a + h' Delta h` (`w^b = 0`: absolute risk, the identity's `sigma_f`). The VARIANCE shares below add to one; the volatilities `sqrt(x'Fx)` and `sqrt(h'Delta h)` do not add to `sigma_a` -- the column `(sigma_F + sigma_S) / sigma_a` shows by how much they overshoot, and `mafrm.backtest.diagnostics.assert_variance_additive` refuses a decomposition whose volatilities add. `CTR_k = x_k sigma_k rho_k` (Menchero & Davis 2011) as a share of `sigma_a`, averaged over rebalances; a NEGATIVE share is a diversifying exposure and is correct. The dense variants (1, 7) enter as `X = I`, so every 'factor' is an asset and the specific share is zero by construction.",
        "",
    ]
    names = sorted(
        {
            k[len("ctr_share_mean_") :]
            for e in entries
            for k in e.diagnostics.summary
            if k.startswith("ctr_share_mean_")
        }
    )
    header = [
        "cell",
        "Y",
        "factor var share",
        "specific var share",
        "(sigma_F + sigma_S) / sigma_a",
        "negative CTR share",
    ] + [f"CTR {n}" for n in names]
    rows = []
    for e in entries:
        s = e.diagnostics.summary
        rows.append(
            [
                e.cell_id,
                e.regime,
                _pct(s["te_factor_variance_share_mean"]),
                _pct(s["te_specific_variance_share_mean"]),
                _num(s["te_volatility_sum_over_total_mean"]),
                _pct(s["te_negative_contribution_share"]),
            ]
            + [_pct(s.get(f"ctr_share_mean_{n}", float("nan"))) for n in names]
        )
    lines += _table(header, rows)
    lines += [
        "",
        "## 10.3 Attribution, reconciled against the risk model",
        "",
        f"The same `X`, `F`, `Delta` that give the ex-ante decomposition give the ex-post attribution. (i) On every day of every holding period the realised gross return at the drifted weights equals `sum_k x_k f_k + specific` with `specific` the remainder -- asserted to 1e-12 of scale in code, and it is why no 'return identity' column appears: it cannot fail. (ii) Each additive-in-variance component's ex-post/ex-ante variance ratio, square-rooted, is SPEC.md 6.1's `B` for that component, scored against the SAME exact chi-square interval at the same sample size (level {level:.0%}). SPEC.md 10.3's example -- 60% ex ante against 5% ex post -- gives `sqrt(5/60) = 0.29` and fires; `tests/test_diagnostics.py` pins it. A component outside the interval is MARKED here (a refutation on real data is a finding); on corrupted input the same code raises. `n/a` is a component absent ex ante (specific risk in a dense variant). Ex post exceeding ex ante is EXPECTED (Hwang & Satchell 2001: ex ante conditions on fixed weights, ex post weights drift and the extra terms are non-negative); both ex-post volatilities are shown, at the rebalance weights held fixed and at the drifted weights the engine held, and `drift share` is the fraction of the drifted variance the drift accounts for. Brinson-Fachler by sleeve against the EQUAL-WEIGHT book (ruling 3; the risk benchmark in the identity stays `w^b = 0` and the two are different objects), Carino-linked over the periods so the effects add to the cumulative active return exactly; the benchmark's return is reported and its Sharpe is not computed.",
        "",
    ]
    header = [
        "cell",
        "Y",
        "B factor",
        "B specific",
        "B total",
        "interval",
        "marked",
        "ex-ante rms vol /period",
        "ex-post vol, fixed w",
        "ex-post vol, drifted w",
        "ex-post/ex-ante (drifted)",
        "drift share of variance",
        "cum. active return vs EW",
        "linked allocation",
        "linked selection",
        "linked interaction",
    ]
    rows = []
    for e in entries:
        s = e.diagnostics.summary
        if "attribution_bias_total" not in s:
            rows.append([e.cell_id, e.regime] + ["n/a"] * (len(header) - 2))
            continue
        marked = []
        for key in ("factor", "specific", "total"):
            if s[f"attribution_{key}_inside"] == 0.0:
                marked.append(key)
        rows.append(
            [
                e.cell_id,
                e.regime,
                _num(s["attribution_bias_factor"]),
                _num(s["attribution_bias_specific"]),
                _num(s["attribution_bias_total"]),
                f"[{s['attribution_interval_lower']:.3f}, {s['attribution_interval_upper']:.3f}]",
                "**" + ", ".join(marked) + "**" if marked else "none",
                _pct(s["ex_ante_rms_volatility_per_period"], 3),
                _pct(s["ex_post_volatility_fixed_weights_per_period"], 3),
                _pct(s["ex_post_volatility_drifted_weights_per_period"], 3),
                _num(s["ex_post_over_ex_ante_drifted_weights"]),
                _pct(s["hwang_satchell_drift_variance_share"], 2),
                _pct(
                    s["brinson_cumulative_portfolio_return"]
                    - s["brinson_cumulative_benchmark_return"]
                ),
                _pct(s["brinson_linked_allocation"]),
                _pct(s["brinson_linked_selection"]),
                _pct(s["brinson_linked_interaction"]),
            ]
        )
    lines += _table(header, rows)
    lines += [
        "",
        "Reading notes:",
        "",
        "- The specific component's `B` is the ex-post variance of the remainder against the forecast `h' Delta h`. For the four government zeros the forecast's regressand is the duration leg (SPEC.md 4.3) while the book is marked on the true excess return, so the remainder carries the carry-and-roll term the forecast does not: the specific `B` on the factor-model cells is where that confound lands, in every cell alike.",
        "- `B total` here is on the arithmetic sum of daily gross returns at the drifted weights; the answer table's `B` is on the compounded net period return. The two agree to the second decimal and neither is substituted for the other.",
        "- Failure mode 9: every statistic above is on the non-overlapping holding periods; no rolling window is quoted as an `n`.",
        "",
    ]
    return "\n".join(lines)
