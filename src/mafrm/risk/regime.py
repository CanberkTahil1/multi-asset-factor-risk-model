"""SPEC.md 5.4's volatility regime adjustment. USE4 section 4.4.

WHAT IT IS FOR, AND WHAT IT IS NOT FOR
--------------------------------------

The EWMA estimator is slow. When volatility jumps -- Sep 2008, Mar 2020 -- the
forecast is too low for weeks, because a half-life of 84 days needs 84 days to
put half its weight on the new regime. The VRA is a **fast level correction**
that uses the cross-section as an instantaneous volatility signal: if every
factor's realised return today is large relative to its own forecast, the level
of the whole matrix is too low *now*, and that is knowable from one day's
cross-section without waiting for the EWMA to catch up::

    B_t^F   = sqrt( (1/K) sum_k (f_kt / sigma_kt)^2 )    sigma_kt made at t-1
    lambda_F^2 = sum_t w_t (B_t^F)^2                     exponential w at tau_VRA
    F <- lambda_F^2 * F

**It corrects for REGIME, not for SAMPLING ERROR.** That distinction is the
subject of the amendment at SPEC.md 5.3.3 and it is restated here because this
is the file where it will be forgotten. The regime problem is that the true
level of volatility moves faster than the estimator tracks it; it averages out
over a long enough sample. Shepard's second-order correction and SPEC.md 5.3's
eigenfactor adjustment address a *systematic multiplicative understatement from
estimation error* that does not average out. Different objects.

**But this stage is fitted to realised data, so empirically it absorbs any
systematic level bias, Shepard's included, and it does so invisibly.** If the
forecast is systematically 5% low for any reason at all, ``B_t^F`` sits
systematically above one and ``lambda_F^2`` scales the matrix up. That is what an
empirical calibration *does*. The consequence, recorded here because W4-P3 has to
build on it: **the disjointness demonstration is three-way, not two-way.** The
eigenfactor adjustment, the VRA and Shepard's closed form all touch the overall
scale, and this stage will have quietly absorbed whatever the other two left on
the table. ``experiments.md`` carries the measured size of that overlap --
``lambda_F^2`` computed with the eigenfactor stage on and with it off -- so that
W4-P3 argues from a number rather than from an assumption.

THE INPUT IS A FORECAST HISTORY, AND THE CALLER SUPPLIES IT
-----------------------------------------------------------

``sigma_kt`` is *"the forecast made at t-1"*, so this is the first stage in the
project that consumes a **history** of forecasts rather than one window of
returns. SPEC.md 5.1.2 fixed the shape of the burn-in rule that governs such a
history -- a bound on ``K / T_eff``, never a day count -- and W3-P4 does not set
one. There is no published bound, and inventing one would be CLAUDE.md invariant
9. What stands in its place is already built: every build emits ``K / T_eff``, so
:class:`ForecastHistory` carries the ratio at every forecast date and a matrix
estimated at an unhealthy ratio is **labelled rather than suppressed**. The W4
rolling harness owns the exclusion decision and owes a criterion written before
the bias statistics are looked at.

WHY sigma_kt COMES FROM THE FULL PRE-VRA PIPELINE
--------------------------------------------------

Stages 1-4, not 1-3. Two reasons, and the second is the one that is easy to get
wrong.

First, it is the faithful reading: SPEC.md 5.4 says *the forecast*, and the
model's forecast at ``t-1`` is what the pipeline produces after every stage that
precedes this one.

Second, **the eigenfactor stage moves ``sigma``, and only through its diagonal.**
SPEC.md 5.3.2 puts that stage in correlation space, where ``gamma^2`` rescales
the eigenvalues of ``rho_hat`` and leaves an inflated diagonal that is
deliberately not renormalised -- see
:mod:`mafrm.risk.eigenfactor`, which explains at length why renormalising it
would delete the correction. So ``diag(F)`` after stage 4 is
``sigma_k^2 * rho_adj_kk`` with ``rho_adj_kk > 1``. Measured on this project's
panel that inflation is 1.00004 to 1.0017 in variance, 0.002% to 0.085% in
volatility. Small, but **not zero, and that is what makes the on/off control
well-posed**: taken from stages 1-3, disabling the eigenfactor stage would not
move ``sigma_kt`` at all and the overlap measurement would return zero by
construction rather than by finding.

NO CLIPPING, AND THAT IS A DECISION
------------------------------------

SPEC.md 6.1 clips standardized returns at +-4 before taking a standard deviation,
because one 2020 observation otherwise dominates a 12-month window. **SPEC.md 5.4
specifies no clip and none is applied here.** The two statistics are not the same
object: 6.1's is a standard deviation over a short rolling window where a single
outlier is most of the sample, and this one is an exponentially weighted mean of
``B^2`` over hundreds of effective observations, where the same outlier is a few
per cent of the weight. Importing 6.1's constant into 5.4 would be inventing a
parameter for this stage (CLAUDE.md invariant 9). What is done instead is
reporting: the largest single-day ``B_t^F`` and the date it fell on go into
``reports/volatility_regime.md``, so a reader can see how much of the multiplier
one day is carrying rather than take it on trust.

THE SPECIFIC-RISK LEG IS NOT HERE
----------------------------------

SPEC.md 5.4's second paragraph specifies a notional-weighted cross-sectional bias
statistic across *assets*, at the **same** ``tau_VRA``, feeding a specific-risk
multiplier. That needs specific returns, which SPEC.md 5.5 has not built yet. It
is deliberately absent rather than stubbed. The shared half-life already sits at
``volatility_regime_adjustment.halflife`` in ``config/model.yaml`` -- its own
top-level section precisely so the two legs cannot drift apart -- so the session
that writes SPEC.md 5.5 reads the same key and adds no parameter.

WHAT THIS MODULE MAY KNOW
-------------------------

A ``T x K`` frame, a :class:`~mafrm.risk.config.RiskConfig`, and a vector of
bias statistics. CLAUDE.md invariant 10: no branch here on what the columns are
or how many there are.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mafrm.numerics import ewma_weights, realised_effective_sample_size
from mafrm.risk.config import RiskConfig

__all__ = [
    "ForecastHistory",
    "ForecastRow",
    "RegimeError",
    "RegimeMultiplier",
    "cross_sectional_bias",
    "forecast_history",
    "regime_multiplier",
    "rolling_multiplier",
]


class RegimeError(ValueError):
    """The regime adjustment was handed a forecast history it cannot use."""


# ---------------------------------------------------------------------------
# The cross-sectional bias statistic -- SPEC.md 5.4, first line
# ---------------------------------------------------------------------------


def cross_sectional_bias(returns: np.ndarray, deviations: np.ndarray) -> np.ndarray:
    """``B_t^F = sqrt(mean_k (f_kt / sigma_kt)^2)``, one value per row. SPEC.md 5.4.

    Both arrays are ``T x K`` and must be aligned so that row ``t`` of
    ``deviations`` is the forecast made at ``t-1`` for the return in row ``t`` of
    ``returns``. **This function cannot check that** -- it sees two arrays and no
    time -- so the lead-lag is established and asserted where the history is
    built, in :func:`forecast_history`, and pinned by a leak test in
    ``tests/test_regime.py``.

    The mean is over ``k``, the cross-section of factors. It is a *cross-sectional*
    statistic evaluated at a single date, which is what makes it an instantaneous
    volatility signal rather than another slow time-series average.
    """
    if returns.shape != deviations.shape:
        raise RegimeError(
            f"cross_sectional_bias: returns {returns.shape} and deviations "
            f"{deviations.shape} must be the same T x K shape"
        )
    if returns.ndim != 2:
        raise RegimeError(f"cross_sectional_bias: expected a T x K array, got {returns.shape}")
    if returns.shape[1] < 1:
        raise RegimeError("cross_sectional_bias: expected at least one factor column")
    if np.any(deviations <= 0.0):
        raise RegimeError(
            "cross_sectional_bias: a forecast volatility is zero or negative. Standardising by "
            "it would produce an infinite or sign-flipped bias statistic, which would then be "
            "squared into the multiplier and be invisible. A non-positive forecast is an "
            "upstream estimator failure and is refused here rather than floored -- there is no "
            "published floor for it (CLAUDE.md invariant 9)."
        )
    standardized = returns / deviations
    result: np.ndarray = np.sqrt(np.mean(np.square(standardized), axis=1))
    return result


# ---------------------------------------------------------------------------
# The multiplier -- SPEC.md 5.4, second and third lines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeMultiplier:
    """``lambda_F^2`` and enough context to say what it was computed from."""

    #: ``lambda_F^2``, the factor applied to the WHOLE covariance matrix.
    lambda_squared: float
    #: ``B_t^F``, oldest first. The series the chart plots.
    bias: np.ndarray
    halflife: int
    #: Kish effective sample size of the VRA weights over this history.
    realised_effective_sample_size: float

    @property
    def lambda_(self) -> float:
        """``lambda_F``, the multiplier on a VOLATILITY rather than a variance."""
        return float(np.sqrt(self.lambda_squared))

    @property
    def observations(self) -> int:
        """How many dates the exponentially weighted sum ran over."""
        return int(self.bias.size)

    @property
    def maximum_bias(self) -> float:
        """The largest single-day ``B_t^F``. Reported because nothing is clipped."""
        return float(np.max(self.bias))

    def render(self) -> str:
        """One line for ``make model``."""
        return (
            f"volatility regime: lambda_F^2 = {self.lambda_squared:.6f} "
            f"(lambda_F = {self.lambda_:.6f}, {100.0 * (self.lambda_ - 1.0):+.2f}% on vol) "
            f"from {self.observations:,} forecast dates at half-life {self.halflife}d, "
            f"Kish T_eff {self.realised_effective_sample_size:.0f}, "
            f"max B_t^F {self.maximum_bias:.3f}, no clipping"
        )


def regime_multiplier(bias: np.ndarray, *, halflife: int) -> RegimeMultiplier:
    """``lambda_F^2 = sum_t w_t (B_t^F)^2`` at ``tau_VRA``. SPEC.md 5.4.

    ``bias`` is oldest-first, matching :func:`mafrm.numerics.ewma_weights`, so the
    most recent date carries the largest weight. The weights are normalised, so
    ``lambda_F^2`` is an exponentially weighted **mean** of ``B^2`` and a
    perfectly calibrated model returns exactly 1 regardless of how much history
    it is handed.
    """
    if bias.ndim != 1:
        raise RegimeError(f"regime_multiplier: expected a 1-D bias series, got {bias.shape}")
    if bias.size < 1:
        raise RegimeError(
            "regime_multiplier: an empty forecast history. This stage consumes a history the "
            "caller supplies (SPEC.md 5.1.2 leaves the burn-in with the caller), so an empty "
            "one is a caller bug and not a burn-in condition to be tolerated here."
        )
    weights = ewma_weights(bias.size, float(halflife))
    return RegimeMultiplier(
        lambda_squared=float(weights @ np.square(bias)),
        bias=bias,
        halflife=halflife,
        realised_effective_sample_size=realised_effective_sample_size(weights),
    )


def rolling_multiplier(bias: np.ndarray, *, halflife: int) -> np.ndarray:
    """``lambda_F^2`` as it stood at every date. Same estimator, expanding window.

    Element ``t`` is exactly ``regime_multiplier(bias[: t + 1]).lambda_squared``
    -- the multiplier the model would have applied on that date, using only the
    forecast history available by then. Computed by the two-term recursion rather
    than by re-running the sum ``T`` times, and ``tests/test_regime.py`` asserts
    the two agree, because a chart drawn from a second implementation of an
    estimator is a chart of the second implementation.

    **This is a rolling statistic and it is drawn as one.** CLAUDE.md failure
    mode 9: consecutive values share all but one observation, so the series is
    enormously autocorrelated and its wiggles are not independent readings. It is
    plotted, and no count of its points is ever quoted as an ``n``.
    """
    if bias.ndim != 1:
        raise RegimeError(f"rolling_multiplier: expected a 1-D bias series, got {bias.shape}")
    if bias.size < 1:
        raise RegimeError("rolling_multiplier: an empty forecast history")
    decay = 0.5 ** (1.0 / float(halflife))
    squared = np.square(bias)
    numerator = 0.0
    denominator = 0.0
    out = np.empty(bias.size, dtype=float)
    for index, value in enumerate(squared):
        numerator = decay * numerator + float(value)
        denominator = decay * denominator + 1.0
        out[index] = numerator / denominator
    return out


# ---------------------------------------------------------------------------
# The forecast history -- the caller's half of SPEC.md 5.4
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastRow:
    """One date's forecast, handed to :func:`forecast_history`'s ``on_row``.

    Exists so a caller can **checkpoint**: building a history over a long panel
    runs the whole pre-VRA pipeline once per date, and an interrupted run that
    kept nothing has to start from zero. Nothing in the estimator depends on
    this -- it is a progress hook, and a caller that passes none gets exactly the
    same result.
    """

    date: pd.Timestamp
    #: ``sigma_kt``, the forecast made at ``t-1``.
    deviations: np.ndarray
    #: The realised factor returns on :attr:`date`.
    returns: np.ndarray
    observations: int
    k_over_realised_t_eff: float

    @property
    def bias(self) -> float:
        """This date's ``B_t^F``. SPEC.md 5.4."""
        return float(cross_sectional_bias(self.returns[None, :], self.deviations[None, :])[0])


@dataclass(frozen=True)
class ForecastHistory:
    """Out-of-sample factor volatility forecasts, one row per date.

    Row ``i`` holds the forecast for :attr:`dates` ``[i]`` built from the rows of
    the panel that fall **strictly before** it, the realised returns on that date,
    and the ``K / T_eff`` the forecast was estimated at. The last of those is
    SPEC.md 5.1.2's substitute for a burn-in parameter: an unhealthy ratio is
    carried and labelled, and W4 decides what to do about it on a criterion
    stated before the bias statistics are read.
    """

    dates: pd.DatetimeIndex
    columns: tuple[str, ...]
    #: ``sigma_kt``, the forecast made at ``t-1``. ``n x K``.
    deviations: np.ndarray
    #: The realised factor returns on :attr:`dates`. ``n x K``.
    returns: np.ndarray
    #: How many panel rows fed each forecast. Strictly increasing by one.
    observations: np.ndarray
    #: ``K`` over the Kish effective sample size of the window each forecast saw.
    k_over_realised_t_eff: np.ndarray
    #: Which pipeline stages ran. ``stop_after`` truncates this, which is how the
    #: eigenfactor on/off control is built.
    stages: tuple[str, ...]
    horizon: str
    #: The eigenfactor ``a`` this history was built at, or ``None`` when the
    #: eigenfactor stage did not run.
    eigenfactor_variant: float | None

    @property
    def factors(self) -> int:
        """``K``."""
        return len(self.columns)

    @property
    def bias(self) -> np.ndarray:
        """``B_t^F`` for every date. SPEC.md 5.4."""
        return cross_sectional_bias(self.returns, self.deviations)

    def multiplier(self, *, halflife: int) -> RegimeMultiplier:
        """``lambda_F^2`` over this whole history."""
        return regime_multiplier(self.bias, halflife=halflife)

    def rolling(self, *, halflife: int) -> np.ndarray:
        """``lambda_F^2`` at every date, from the history available by then."""
        return rolling_multiplier(self.bias, halflife=halflife)

    def frame(self) -> pd.DataFrame:
        """Date-indexed ``B_t^F`` and ``K/T_eff``, for the cache and the chart."""
        return pd.DataFrame(
            {
                "bias": self.bias,
                "observations": self.observations,
                "k_over_realised_t_eff": self.k_over_realised_t_eff,
            },
            index=self.dates,
        )


def forecast_history(
    factor_returns: pd.DataFrame,
    config: RiskConfig,
    *,
    minimum_observations: int,
    stop_after: str,
    on_row: Callable[[ForecastRow], None] | None = None,
) -> ForecastHistory:
    """Run the pre-VRA pipeline on an expanding window ending at every date.

    ``minimum_observations`` is **required and has no default**. SPEC.md 5.1.2
    leaves the burn-in with the caller and no bound on ``K / T_eff`` is published,
    so this function will not choose one; what it does instead is report the ratio
    every forecast was estimated at.

    ``stop_after`` is forwarded to :func:`~mafrm.risk.covariance.run_pipeline`
    and is **required, with no default**, because the one value that would be a
    natural default is the one value that cannot work: running to the end would
    need the very bias history this function exists to produce. Passing
    ``"eigenfactor"`` gives the model's own pre-VRA forecast, which is what
    SPEC.md 5.4 standardises by; passing ``"psd_repair"`` gives the history the
    eigenfactor stage did *not* touch, which is the control half of the overlap
    measurement recorded in ``experiments.md``.

    ``on_row`` is called after each date is computed and is a **progress hook for
    checkpointing**, nothing more. It cannot change the result: a caller that
    passes none gets the same arrays. Resumption needs no separate parameter --
    a caller holding checkpointed rows through position ``p`` calls this again
    with ``minimum_observations = p + 1`` and concatenates.

    THE LEAD-LAG IS ASSERTED, NOT ASSUMED
        The forecast for date ``t`` is built from ``factor_returns.iloc[:j]``
        where ``j`` is ``t``'s own position, so row ``t`` itself is excluded by
        construction. That construction is then **checked** -- the recorded
        observation count must equal the position -- because this is the first
        thing in the project to consume a forecast history and a contemporaneous
        forecast leaking in here would flatter ``B_t^F`` toward one, shrink
        ``lambda_F^2``, and silently corrupt every statistic later calibrated
        against it. ``tests/test_regime.py`` perturbs the return on date ``t`` and
        asserts the forecast for ``t`` does not move.
    """
    # Local, and for the same reason the eigenfactor import in ``covariance`` is
    # local: ``covariance`` reaches back here for the VRA stage, so the cycle has
    # to be broken on one side. It is broken on the side that runs the loop.
    from mafrm.risk.covariance import run_pipeline

    periods, factors = factor_returns.shape
    if minimum_observations < factors:
        raise RegimeError(
            f"forecast_history: minimum_observations={minimum_observations} is below K="
            f"{factors}. The second moment is singular by construction below T = K -- that is "
            "the arithmetic minimum, not a burn-in rule, and it is the same floor "
            "reports/psd_repairs.md scans from."
        )
    if stop_after == "volatility_regime":
        raise RegimeError(
            "forecast_history: stop_after='volatility_regime' would run SPEC.md 5.4's stage to "
            "build the input SPEC.md 5.4's stage needs. Stop after 'eigenfactor' for the "
            "model's own pre-VRA forecast, or after 'psd_repair' for the eigenfactor-off "
            "control."
        )
    if minimum_observations >= periods:
        raise RegimeError(
            f"forecast_history: minimum_observations={minimum_observations} leaves no dates to "
            f"forecast in a panel of {periods} rows."
        )

    dates: list[pd.Timestamp] = []
    deviations: list[np.ndarray] = []
    realised: list[np.ndarray] = []
    counts: list[int] = []
    ratios: list[float] = []
    stages: tuple[str, ...] = ()

    values = factor_returns.to_numpy(dtype=float)
    for position in range(minimum_observations, periods):
        build = run_pipeline(factor_returns.iloc[:position], config, stop_after=stop_after)
        if build.observations != position:
            raise RegimeError(
                f"forecast_history: the build for {factor_returns.index[position]!s} saw "
                f"{build.observations} observations but sits at position {position}. The "
                "forecast for date t must be made from data strictly before t; this is the "
                "look-ahead check and it has failed."
            )
        row = ForecastRow(
            date=factor_returns.index[position],
            deviations=np.sqrt(np.diag(build.matrix)),
            returns=values[position],
            observations=position,
            k_over_realised_t_eff=build.k_over_realised_t_eff,
        )
        dates.append(row.date)
        deviations.append(row.deviations)
        realised.append(row.returns)
        counts.append(row.observations)
        ratios.append(row.k_over_realised_t_eff)
        stages = tuple(stage.name for stage in build.stages)
        if on_row is not None:
            on_row(row)

    return ForecastHistory(
        dates=pd.DatetimeIndex(dates),
        columns=tuple(str(column) for column in factor_returns.columns),
        deviations=np.asarray(deviations, dtype=float),
        returns=np.asarray(realised, dtype=float),
        observations=np.asarray(counts, dtype=int),
        k_over_realised_t_eff=np.asarray(ratios, dtype=float),
        stages=stages,
        horizon=config.horizon,
        eigenfactor_variant=config.eigenfactor_variant if "eigenfactor" in stages else None,
    )
