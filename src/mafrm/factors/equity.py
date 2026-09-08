"""The six price-only CNE5 style descriptors and their standardization. SPEC.md 15.4 and 15.5.

BETA, MOMENTUM (RSTR), SIZE (LNCAP), NON-LINEAR SIZE, RESIDUAL VOLATILITY
(DASTD, CMRA, HSIGMA) and LIQUIDITY (STOM, STOQ, STOA), from daily price,
volume and the point-in-time share count of :mod:`mafrm.data.market_cap`.
Every window, half-life, lag, block length and composite weight is read from
``config/model.yaml`` ``equity_descriptors``; nothing numerical is a literal
here. Exposures are DAILY (SPEC.md 15.4.1 ruling 3).

Two stages, kept apart because the second is re-run at three winsorization
bounds while the first is not:

1. **Rolling** (:func:`raw_descriptors`): per-name time-series statistics on
   EWMA-weighted trailing windows. A window must be COMPLETE -- every session
   valid -- or the descriptor is missing on that date; no weights are
   renormalised over partial windows, because the fraction that would make a
   partial window acceptable is a number nobody published.
2. **Cross-sectional** (:func:`exposures_from_raw`): per date, over the names
   in the estimation universe WITH a screened cap, winsorize the raw
   descriptor, standardize it with SPEC.md 15.5's asymmetry -- CAP-WEIGHTED
   mean, EQUAL-WEIGHTED standard deviation -- build the composites as weighted
   sums of standardized descriptors and re-standardize them, orthogonalize
   NLSIZE against SIZE and RESVOL against BETA and SIZE (regressions weighted
   by sqrt(cap), the cross-sectional regression's own weights), and
   re-standardize. A cap-weighted portfolio therefore has
   exactly zero exposure to every style factor, which is the property the
   centring exists to produce and the test that pins it.

The market return for BETA is CNE5's: the cap-weighted excess return of the
estimation universe, weights the PREVIOUS session's cap, so no weight is set
by the return it multiplies. Its correlation with SPY is a report cross-check
(experiments.md row 278), never an input.

Nothing here reads the network (CLAUDE.md invariant 1) and nothing reaches
the holdout: every input arrives through ``cache.read`` or the committed
reference tables, and :func:`build` cuts at ``holdout_start``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import numpy.typing as npt
import pandas as pd

from mafrm import config as config_mod
from mafrm.data import cache, french, loaders, market_cap, sp500_reference
from mafrm.factors import equity_universe
from mafrm.numerics import ewma_weights

__all__ = [
    "FACTORS",
    "Exposures",
    "RawDescriptors",
    "build",
    "cap_weighted_market_excess",
    "cmra",
    "exposures_from_raw",
    "lncap",
    "orthogonalize",
    "raw_descriptors",
    "rolling_beta_hsigma",
    "rolling_dastd",
    "rstr",
    "standardize",
    "total_returns",
    "turnover",
    "windowed_sums",
    "winsorize",
]

Array = npt.NDArray[np.float64]

#: The six style factors, in the order every report and matrix uses.
FACTORS: tuple[str, ...] = ("BETA", "MOMENTUM", "SIZE", "NLSIZE", "RESVOL", "LIQUIDITY")
#: The raw descriptors the rolling stage produces.
DESCRIPTORS: tuple[str, ...] = (
    "beta",
    "hsigma",
    "dastd",
    "rstr",
    "cmra",
    "stom",
    "stoq",
    "stoa",
    "lncap",
)


class EquityDescriptorError(ValueError):
    """An input this module will not accept, named."""


# ---------------------------------------------------------------------------
# Rolling stage -- time-series statistics on complete trailing windows
# ---------------------------------------------------------------------------


def windowed_sums(values: Array, weights: Array) -> tuple[Array, Array]:
    """Trailing weighted sums of every column, and the count of valid rows in each window.

    ``values`` is T x N with NaN for a missing observation; ``weights`` has
    length W and is OLDEST FIRST (:func:`mafrm.numerics.ewma_weights`). Row
    ``t`` of the first output is ``sum_j weights[j] * values[t - (W-1) + j]``
    with missing values contributing zero; row ``t`` of the second is how many
    of those W rows were valid, so the caller can require a COMPLETE window
    (count == W). The first W-1 rows are NaN / zero. Exact arithmetic: a direct
    convolution per column, not an FFT.
    """
    if values.ndim != 2:
        raise EquityDescriptorError("windowed_sums: values must be T x N")
    window = len(weights)
    n_obs, n_cols = values.shape
    sums = np.full((n_obs, n_cols), np.nan)
    counts = np.zeros((n_obs, n_cols))
    if n_obs < window:
        return sums, counts
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    kernel = weights[::-1]
    ones = np.ones(window)
    for j in range(n_cols):
        sums[window - 1 :, j] = np.convolve(filled[:, j], kernel, mode="full")[window - 1 : n_obs]
        counts[window - 1 :, j] = np.convolve(valid[:, j].astype(float), ones, mode="full")[
            window - 1 : n_obs
        ]
    return sums, counts


def _complete(sums: Array, counts: Array, window: int) -> Array:
    return np.where(counts == window, sums, np.nan)


def _frame(values: Array, like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(values, index=like.index, columns=like.columns)


def rolling_beta_hsigma(
    excess: pd.DataFrame, market: pd.Series, *, window: int, halflife: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BETA and HSIGMA. CNE5 Descriptor Details; USE4 Table 4.1's 252d / 63d.

    ``r_nt - r_ft = alpha_n + beta_n R_t + e_nt`` by exponentially weighted
    least squares over the trailing ``window`` sessions, half-life
    ``halflife``. With normalised weights ``w``: ``beta = Cov_w(y, R) /
    Var_w(R)`` and ``HSIGMA = sqrt(Var_w(e)) = sqrt(Var_w(y) - beta^2 Var_w(R))``,
    the exact residual variance of a weighted regression with an intercept.
    Both need a complete window of the name AND of the market.
    """
    if not excess.index.equals(market.index):
        raise EquityDescriptorError("rolling_beta_hsigma: excess and market disagree on dates")
    w = ewma_weights(window, halflife)
    y = excess.to_numpy(dtype=float)
    x = market.to_numpy(dtype=float)[:, None]
    # Every product needs BOTH observations present, so mask y by x's validity.
    y = np.where(np.isfinite(x), y, np.nan)
    x_rep = np.broadcast_to(x, y.shape)
    sy, cy = windowed_sums(y, w)
    sx, _ = windowed_sums(np.where(np.isfinite(y), x_rep, np.nan), w)
    sxy, _ = windowed_sums(y * x_rep, w)
    sxx, _ = windowed_sums(np.where(np.isfinite(y), x_rep**2, np.nan), w)
    syy, _ = windowed_sums(y**2, w)
    cov = sxy - sx * sy
    var_x = sxx - sx**2
    var_y = syy - sy**2
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = cov / var_x
        resid_var = var_y - beta**2 * var_x
    hsigma = np.sqrt(np.clip(resid_var, 0.0, None))
    return (
        _frame(_complete(beta, cy, window), excess),
        _frame(_complete(hsigma, cy, window), excess),
    )


def rolling_dastd(excess: pd.DataFrame, *, window: int, halflife: int) -> pd.DataFrame:
    """DASTD: exponentially weighted standard deviation of daily excess returns. CNE5."""
    w = ewma_weights(window, halflife)
    y = excess.to_numpy(dtype=float)
    s1, c = windowed_sums(y, w)
    s2, _ = windowed_sums(y**2, w)
    var = np.clip(s2 - s1**2, 0.0, None)
    return _frame(_complete(np.sqrt(var), c, window), excess)


def rstr(log_excess: pd.DataFrame, *, window: int, halflife: int, lag: int) -> pd.DataFrame:
    """RSTR: exponentially weighted sum of log excess returns, LAGGED. CNE5.

    ``sum_{t=lag}^{lag+window-1} w_t [ln(1 + r_t) - ln(1 + r_ft)]`` -- the
    window ENDS ``lag`` sessions before the estimation date, skipping the most
    recent month (CLAUDE.md parameter table: "the part replicas get wrong").
    Weights normalised to one; the scale is removed by standardization.
    """
    if lag < 0:
        raise EquityDescriptorError("rstr: lag must be non-negative")
    w = ewma_weights(window, halflife)
    s, c = windowed_sums(log_excess.to_numpy(dtype=float), w)
    unlagged = _complete(s, c, window)
    out = np.full_like(unlagged, np.nan)
    if lag == 0:
        out = unlagged
    else:
        out[lag:] = unlagged[:-lag]
    return _frame(out, log_excess)


def cmra(log_excess: pd.DataFrame, *, months: int, days_per_month: int) -> pd.DataFrame:
    """CMRA: the range of trailing cumulative log excess returns over T = 1 ... months. CNE5.

    ``Z(T) = sum over the trailing T months of [ln(1 + r) - ln(1 + r_f)]`` with
    a month a block of ``days_per_month`` sessions (SPEC.md 15.4.1 ruling 4),
    and ``CMRA = max_T Z(T) - min_T Z(T)``. Every session of the
    ``months x days_per_month`` window must be valid.
    """
    if months <= 0 or days_per_month <= 0:
        raise EquityDescriptorError("cmra: months and days_per_month must be positive")
    window = months * days_per_month
    values = log_excess.to_numpy(dtype=float)
    n_obs = values.shape[0]
    valid = np.isfinite(values)
    cum = np.cumsum(np.where(valid, values, 0.0), axis=0)
    cum = np.vstack([np.zeros((1, values.shape[1])), cum])  # cum[t] = sum of rows < t
    ones = np.ones(window)
    complete = np.zeros(values.shape, dtype=bool)
    if n_obs >= window:
        for j in range(values.shape[1]):
            counts = np.convolve(valid[:, j].astype(float), ones, mode="full")[window - 1 : n_obs]
            complete[window - 1 :, j] = counts == window
    z_max = np.full(values.shape, -np.inf)
    z_min = np.full(values.shape, np.inf)
    for t_months in range(1, months + 1):
        span = t_months * days_per_month
        z = np.full(values.shape, np.nan)
        # Z over rows (t - span + 1 .. t) = cum[t + 1] - cum[t + 1 - span]
        z[span - 1 :] = cum[span:] - cum[: n_obs - span + 1]
        z_max = np.fmax(z_max, z)
        z_min = np.fmin(z_min, z)
    out = np.where(complete, z_max - z_min, np.nan)
    return _frame(out, log_excess)


def turnover(
    volume: pd.DataFrame,
    shares: pd.DataFrame,
    *,
    months: Mapping[str, int],
    days_per_month: int,
) -> dict[str, pd.DataFrame]:
    """STOM, STOQ, STOA: log mean monthly share turnover over 1, 3 and 12 months. CNE5.

    ``STOM = ln(sum_{t=1..21} V_t / S_t)``; ``STOQ = ln((1/3) sum_tau
    exp(STOM_tau))`` over three consecutive monthly blocks, which equals ln of
    the mean monthly turnover over 63 sessions; STOA likewise over 12 months.
    Volume and the share count are on the same split-adjusted basis (the
    vendor back-adjusts both; :mod:`mafrm.data.market_cap` puts the count
    there). A window with any invalid bar, or zero total turnover, is missing.
    """
    if not volume.index.equals(shares.index) or not volume.columns.equals(shares.columns):
        raise EquityDescriptorError("turnover: volume and shares disagree on dates or tickers")
    v = volume.to_numpy(dtype=float)
    s = shares.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where((v > 0) & (s > 0), v / s, np.nan)
    out: dict[str, pd.DataFrame] = {}
    for name, n_months in months.items():
        window = n_months * days_per_month
        sums, counts = windowed_sums(q, np.ones(window))
        mean_monthly = _complete(sums, counts, window) / n_months
        with np.errstate(divide="ignore", invalid="ignore"):
            logged = np.where(mean_monthly > 0, np.log(mean_monthly), np.nan)
        out[name] = _frame(logged, volume)
    return out


def lncap(cap: pd.DataFrame) -> pd.DataFrame:
    """SIZE's raw descriptor, ``ln(total market cap)``. CNE5 / USE4."""
    values = cap.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(values > 0, np.log(values), np.nan)
    return _frame(out, cap)


def total_returns(close: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """``(P_t + D_t) / P_{t-1} - 1`` per name from raw closes and ex-date cash dividends.

    The same construction as :func:`mafrm.data.prices.total_return` for the
    ETFs, on a wide panel: nothing is back-adjusted, so a later dividend cannot
    change an earlier return. A return needs a positive close on both sessions.
    """
    p = close.to_numpy(dtype=float)
    d = (
        dividends.reindex(index=close.index, columns=close.columns)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    prev = np.vstack([np.full((1, p.shape[1]), np.nan), p[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((p > 0) & (prev > 0), (p + d) / prev - 1.0, np.nan)
    return _frame(r, close)


def cap_weighted_market_excess(
    excess: pd.DataFrame, cap: pd.DataFrame, universe: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """CNE5's market: the cap-weighted excess return of the estimation universe.

    Weights on session ``t`` are the PREVIOUS session's caps of the names in
    the universe on ``t`` with a return on ``t``, normalised to one, so no
    weight depends on the return it multiplies. Returns the series and the
    per-session count of names it averaged; a session with no eligible name is
    NaN.
    """
    if not (excess.index.equals(cap.index) and excess.index.equals(universe.index)):
        raise EquityDescriptorError("cap_weighted_market_excess: inputs disagree on dates")
    tickers = excess.columns
    r = excess.to_numpy(dtype=float)
    c = cap.reindex(columns=tickers).to_numpy(dtype=float)
    u = universe.reindex(columns=tickers).fillna(False).to_numpy(dtype=bool)
    prev_cap = np.vstack([np.full((1, c.shape[1]), np.nan), c[:-1]])
    eligible = u & np.isfinite(r) & np.isfinite(prev_cap) & (prev_cap > 0)
    w = np.where(eligible, prev_cap, 0.0)
    total = w.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        market = np.where(total > 0, (w * np.where(eligible, r, 0.0)).sum(axis=1) / total, np.nan)
    return (
        pd.Series(market, index=excess.index, name="market_excess"),
        pd.Series(eligible.sum(axis=1), index=excess.index, name="names"),
    )


@dataclass(frozen=True)
class RawDescriptors:
    """The rolling stage's output: one sessions x tickers frame per raw descriptor."""

    frames: dict[str, pd.DataFrame]
    market_excess: pd.Series
    market_names: pd.Series

    def __post_init__(self) -> None:
        missing = set(DESCRIPTORS) - set(self.frames)
        if missing:
            raise EquityDescriptorError(f"RawDescriptors lacks {sorted(missing)}")


def raw_descriptors(
    *,
    excess: pd.DataFrame,
    risk_free: pd.Series,
    volume: pd.DataFrame,
    shares: pd.DataFrame,
    cap: pd.DataFrame,
    universe: pd.DataFrame,
    descriptors: config_mod.EquityDescriptorConfig,
    days_per_month: int,
) -> RawDescriptors:
    """Every raw descriptor on every session of ``excess``'s calendar. Pure."""
    for name, frame in (("volume", volume), ("shares", shares), ("cap", cap)):
        if not frame.index.equals(excess.index) or not frame.columns.equals(excess.columns):
            raise EquityDescriptorError(f"raw_descriptors: {name} disagrees with excess on shape")
    if not risk_free.index.equals(excess.index):
        raise EquityDescriptorError("raw_descriptors: risk_free disagrees with excess on dates")
    market, names = cap_weighted_market_excess(excess, cap, universe)
    rf = risk_free.to_numpy(dtype=float)[:, None]
    with np.errstate(invalid="ignore"):
        log_excess = _frame(np.log1p(excess.to_numpy(dtype=float) + rf) - np.log1p(rf), excess)
    beta, hsigma = rolling_beta_hsigma(
        excess, market, window=descriptors.beta.window, halflife=descriptors.beta.halflife
    )
    if (descriptors.hsigma.window, descriptors.hsigma.halflife) != (
        descriptors.beta.window,
        descriptors.beta.halflife,
    ):
        # CNE5 defines HSIGMA on the BETA regression's own window; a different
        # setting would need a second regression, and the config says both.
        _, hsigma = rolling_beta_hsigma(
            excess,
            market,
            window=descriptors.hsigma.window,
            halflife=descriptors.hsigma.halflife,
        )
    horizons = descriptors.liquidity_horizons_months
    turn = turnover(
        volume,
        shares,
        months={"stom": horizons.stom, "stoq": horizons.stoq, "stoa": horizons.stoa},
        days_per_month=days_per_month,
    )
    frames = {
        "beta": beta,
        "hsigma": hsigma,
        "dastd": rolling_dastd(
            excess, window=descriptors.dastd.window, halflife=descriptors.dastd.halflife
        ),
        "rstr": rstr(
            log_excess,
            window=descriptors.momentum.window,
            halflife=descriptors.momentum.halflife,
            lag=descriptors.momentum.lag,
        ),
        "cmra": cmra(log_excess, months=descriptors.cmra_months, days_per_month=days_per_month),
        "stom": turn["stom"],
        "stoq": turn["stoq"],
        "stoa": turn["stoa"],
        "lncap": lncap(cap),
    }
    return RawDescriptors(frames=frames, market_excess=market, market_names=names)


# ---------------------------------------------------------------------------
# Cross-sectional stage -- winsorize, standardize, orthogonalize
# ---------------------------------------------------------------------------


def _row_winsorize(x: Array, bound: float) -> Array:
    valid = np.isfinite(x)
    if valid.sum() < 2:
        return x
    mean = float(np.mean(x[valid]))
    sd = float(np.std(x[valid], ddof=1))
    if not sd > 0:
        return x
    return np.where(valid, np.clip(x, mean - bound * sd, mean + bound * sd), np.nan)


def winsorize(raw: pd.DataFrame, *, bound: float) -> pd.DataFrame:
    """Clip every row (date) to the equal-weighted mean +/- ``bound`` equal-weighted sd.

    SPEC.md 15.4.1 ruling 2: one pass, equal weights, at the RAW level before
    standardization (SPEC.md 15.5). The sd is the sample sd (``ddof=1``) over
    the row's valid cells.
    """
    if not bound > 0:
        raise EquityDescriptorError("winsorize: bound must be positive")
    values = raw.to_numpy(dtype=float).copy()
    for i in range(values.shape[0]):
        values[i] = _row_winsorize(values[i], bound)
    return _frame(values, raw)


def _row_standardize(x: Array, cap: Array) -> Array:
    valid = np.isfinite(x) & np.isfinite(cap) & (cap > 0)
    if valid.sum() < 2:
        return np.full_like(x, np.nan)
    w = cap[valid] / cap[valid].sum()
    mean = float(np.dot(w, x[valid]))
    sd = float(np.std(x[valid], ddof=1))
    if not sd > 0:
        return np.full_like(x, np.nan)
    return np.where(valid, (x - mean) / sd, np.nan)


def standardize(raw: pd.DataFrame, cap: pd.DataFrame) -> pd.DataFrame:
    """USE4 Eq. 2.4, SPEC.md 15.5: ``(Raw - mu) / sigma``, mu CAP-WEIGHTED, sigma EQUAL-WEIGHTED.

    The asymmetry is the point. Cap-weighted centring makes the cap-weighted
    portfolio's exposure exactly zero; equal-weighted scaling stops the
    largest names setting the scale. A cell needs a value AND a positive cap
    to be standardized; others are missing. Sample sd (``ddof=1``).
    """
    if not raw.index.equals(cap.index) or not raw.columns.equals(cap.columns):
        raise EquityDescriptorError("standardize: raw and cap disagree on shape")
    values = raw.to_numpy(dtype=float)
    c = cap.to_numpy(dtype=float)
    out = np.empty_like(values)
    for i in range(values.shape[0]):
        out[i] = _row_standardize(values[i], c[i])
    return _frame(out, raw)


def _row_orthogonalize(y: Array, xs: Sequence[Array], weights: Array) -> Array:
    cap = weights
    valid = np.isfinite(y) & np.isfinite(cap) & (cap > 0)
    for x in xs:
        valid &= np.isfinite(x)
    k = len(xs) + 1
    if valid.sum() <= k:
        return np.full_like(y, np.nan)
    design = np.column_stack([np.ones(int(valid.sum())), *[x[valid] for x in xs]])
    root_w = np.sqrt(cap[valid] / cap[valid].sum())
    coef, *_ = np.linalg.lstsq(design * root_w[:, None], y[valid] * root_w, rcond=None)
    resid = np.full_like(y, np.nan)
    resid[valid] = y[valid] - design @ coef
    return resid


def orthogonalize(
    y: pd.DataFrame, xs: Sequence[pd.DataFrame], weights: pd.DataFrame
) -> pd.DataFrame:
    """Per date, the residual of a WEIGHTED regression of ``y`` on ``xs`` with an intercept.

    USE4's construction for Non-Linear Size (on Size) and Residual Volatility
    (on Beta and Size), "orthogonalized ... on a regression-weighted basis":
    the weights are the cross-sectional regression's, ``sqrt(cap)`` (SPEC.md
    15.6; W7-P2b ruling 2), so the residual is uncorrelated with every
    regressor in the metric the regression uses. Any positive weight frame is
    accepted; the caller passes ``cap ** 0.5``. The equal- and cap-weighted
    correlations the report shows are what is left in the other metrics.
    """
    for x in xs:
        if not x.index.equals(y.index) or not x.columns.equals(y.columns):
            raise EquityDescriptorError("orthogonalize: a regressor disagrees with y on shape")
    if not weights.index.equals(y.index) or not weights.columns.equals(y.columns):
        raise EquityDescriptorError("orthogonalize: weights disagree with y on shape")
    values = y.to_numpy(dtype=float)
    regressors = [x.to_numpy(dtype=float) for x in xs]
    c = weights.to_numpy(dtype=float)
    out = np.empty_like(values)
    for i in range(values.shape[0]):
        out[i] = _row_orthogonalize(values[i], [x[i] for x in regressors], c[i])
    return _frame(out, y)


def _composite(parts: Mapping[str, pd.DataFrame], weights: Mapping[str, float]) -> pd.DataFrame:
    """A weighted sum of standardized descriptors; missing if any part is missing."""
    first = next(iter(parts.values()))
    total = np.zeros(first.shape)
    for name, weight in weights.items():
        total = total + weight * parts[name].to_numpy(dtype=float)
    return _frame(total, first)


@dataclass(frozen=True)
class Exposures:
    """The six standardized daily exposures and everything the report needs to explain them."""

    #: ``FACTORS`` -> sessions x tickers standardized exposure.
    exposures: dict[str, pd.DataFrame]
    #: Standardized (winsorized, cap-centred, equal-scaled) single descriptors.
    standardized: dict[str, pd.DataFrame]
    #: The two composites and NLSIZE's cube BEFORE orthogonalization (the
    #: report measures what the orthogonalizations removed).
    pre_orthogonal: dict[str, pd.DataFrame]
    #: Names in the estimation universe WITH a screened cap: the standardization set.
    universe: pd.DataFrame
    cap: pd.DataFrame
    market_excess: pd.Series
    winsor_bound: float
    #: Per session: universe, with_cap, and one column per factor with an exposure.
    counts: pd.DataFrame
    sample_start: pd.Timestamp = field(default=pd.Timestamp("1970-01-01"))

    def cap_weighted_exposure(self) -> pd.DataFrame:
        """``sum_n w_n X_nk`` per session and factor, weights the cap over names with an exposure.

        Zero to floating precision on every session for every factor: the
        property SPEC.md 15.5's cap-weighted centring exists to produce.
        """
        out: dict[str, pd.Series] = {}
        c = self.cap.to_numpy(dtype=float)
        for factor in FACTORS:
            x = self.exposures[factor].to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(c) & (c > 0)
            w = np.where(valid, c, 0.0)
            total = w.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                value = np.where(
                    total > 0, (w * np.where(valid, x, 0.0)).sum(axis=1) / total, np.nan
                )
            out[factor] = pd.Series(value, index=self.cap.index)
        return pd.DataFrame(out)


def exposures_from_raw(
    raw: RawDescriptors,
    *,
    cap: pd.DataFrame,
    universe: pd.DataFrame,
    descriptors: config_mod.EquityDescriptorConfig,
    winsor_bound: float,
    sample_start: pd.Timestamp | None = None,
) -> Exposures:
    """The cross-sectional stage on every session of ``universe``. Pure.

    ``universe`` is the estimation-universe mask; the standardization set on a
    session is its names with a positive cap. ``winsor_bound`` is passed
    explicitly because the report runs this at the READING and at the two
    sensitivity bounds.
    """
    frames = raw.frames
    sessions = pd.DatetimeIndex(universe.index)
    tickers = next(iter(frames.values())).columns
    in_set = universe.reindex(columns=tickers).fillna(False).astype(bool)
    cap_set = cap.reindex(index=sessions, columns=tickers).where(in_set)
    if sample_start is not None:
        cap_set = cap_set.loc[cap_set.index >= sample_start]
        in_set = in_set.loc[in_set.index >= sample_start]
        sessions = pd.DatetimeIndex(cap_set.index)

    def prepared(name: str) -> pd.DataFrame:
        frame = frames[name].reindex(index=sessions, columns=tickers).where(in_set)
        return standardize(winsorize(frame, bound=winsor_bound), cap_set)

    standardized = {name: prepared(name) for name in DESCRIPTORS}

    size = standardized["lncap"]
    beta = standardized["beta"]
    rv = descriptors.resvol_composite
    resvol_raw = standardize(
        _composite(standardized, {"dastd": rv.dastd, "cmra": rv.cmra, "hsigma": rv.hsigma}),
        cap_set,
    )
    lq = descriptors.liquidity_composite
    liquidity = standardize(
        _composite(standardized, {"stom": lq.stom, "stoq": lq.stoq, "stoa": lq.stoa}), cap_set
    )
    cube = _frame(size.to_numpy(dtype=float) ** 3, size)
    # The regression weights of SPEC.md 15.6 -- sqrt(cap) -- for BOTH
    # orthogonalizations (USE4's "regression-weighted basis"; W7-P2b ruling 2).
    # Centring stays cap-weighted (standardize): a different operation.
    regression_weights = cap_set.pow(0.5)
    # NLSIZE: orthogonalize the cube against SIZE, THEN winsorize, then re-standardize
    # (SPEC.md 15.5: the one exception to winsorizing before standardization).
    nlsize = standardize(
        winsorize(orthogonalize(cube, [size], regression_weights), bound=winsor_bound), cap_set
    )
    # RESVOL: orthogonalize against BETA and SIZE (USE4; SPEC.md 15.4.1 ruling 5), re-standardize.
    resvol = standardize(orthogonalize(resvol_raw, [beta, size], regression_weights), cap_set)

    exposures = {
        "BETA": beta,
        "MOMENTUM": standardized["rstr"],
        "SIZE": size,
        "NLSIZE": nlsize,
        "RESVOL": resvol,
        "LIQUIDITY": liquidity,
    }
    counts = pd.DataFrame(index=sessions)
    counts.index.name = "date"
    counts["universe"] = in_set.sum(axis=1)
    counts["with_cap"] = cap_set.notna().sum(axis=1)
    for factor in FACTORS:
        counts[factor] = exposures[factor].notna().sum(axis=1)
    return Exposures(
        exposures=exposures,
        standardized=standardized,
        pre_orthogonal={"NLSIZE": cube, "RESVOL": resvol_raw},
        universe=in_set & cap_set.notna(),
        cap=cap_set,
        market_excess=raw.market_excess.reindex(sessions),
        winsor_bound=winsor_bound,
        counts=counts.astype(int),
        sample_start=sessions[0] if len(sessions) else pd.Timestamp("1970-01-01"),
    )


# ---------------------------------------------------------------------------
# Inputs from the cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Inputs:
    """Everything the two stages read, on one calendar, cut at the holdout boundary.

    The calendar starts ``burn_in`` sessions before ``sample.start`` so the
    longest window (RSTR's lag + window) is complete on the first sample
    session. Before ``sample.start`` the estimation universe is the FIRST
    sample session's universe, frozen -- a construction for the market
    return's burn-in only, flagged in ``universe_frozen``; nothing is
    standardized on those sessions.
    """

    excess: pd.DataFrame
    risk_free: pd.Series
    volume: pd.DataFrame
    shares: pd.DataFrame
    cap: pd.DataFrame
    universe: pd.DataFrame
    universe_frozen: pd.Series
    panel: market_cap.MarketCapPanel
    screen: equity_universe.Screen
    dividends: pd.DataFrame
    close: pd.DataFrame
    sample_start: pd.Timestamp
    holdout_start: pd.Timestamp


def _dividends(manifest: cache.Manifest, entry: cache.ManifestEntry | None) -> pd.DataFrame:
    actions_entry = entry or manifest.latest(source="yfinance", name=loaders.SP500_ACTIONS_NAME)
    long = cache.read(actions_entry, manifest=manifest)
    if "Dividends" not in long.columns:
        return pd.DataFrame()
    cash = pd.to_numeric(long["Dividends"], errors="coerce")
    wide = (
        long.assign(_cash=cash)[["ticker", "_cash"]]
        .reset_index()
        .pivot_table(index="date", columns="ticker", values="_cash", aggfunc="sum")
        .sort_index()
    )
    wide.columns = pd.Index([str(c) for c in wide.columns], name="ticker")
    return wide


def load_inputs(cfg: config_mod.Config | None = None) -> Inputs:
    """Read the cache and the committed tables; build the screened cap panel and the universe."""
    cfg = cfg or config_mod.load()
    manifest = cache.Manifest.load()
    sample_start = pd.Timestamp(cfg.model.sample.start)
    holdout_start = pd.Timestamp(cfg.require_holdout_start())
    desc = cfg.model.equity_descriptors
    burn_in = desc.momentum.lag + desc.momentum.window

    snapshot = sp500_reference.snapshot_input_entries(cfg, manifest)
    prices_entry = actions_entry = None
    if snapshot is not None:
        by_key = {(e.source, e.name): e for e in snapshot}
        prices_entry = by_key.get(("yfinance", "sp500_prices"))
        actions_entry = by_key.get(("yfinance", "sp500_actions"))

    close, volume = equity_universe.load_bars(cfg, entry=prices_entry)
    membership = equity_universe.load_membership(cfg)
    u = cfg.model.equity_universe
    screen = equity_universe.screen(
        close,
        volume,
        membership,
        min_history_days=u.min_history_days,
        min_dollar_adv=u.min_dollar_adv,
        start=sample_start,
        end=holdout_start,
        strict=snapshot is not None,
    )
    tickers = pd.Index(sorted(str(t) for t in screen.universe.columns), name="ticker")
    all_sessions = pd.DatetimeIndex(close.index)
    all_sessions = all_sessions[all_sessions < holdout_start]
    first_pos = int(np.searchsorted(all_sessions.to_numpy(), sample_start.to_datetime64()))
    start_pos = max(first_pos - burn_in, 0)
    sessions = all_sessions[start_pos:]

    panel = market_cap.build(
        cfg,
        manifest=manifest,
        prices_entry=prices_entry,
        actions_entry=actions_entry,
        start=sessions[0].date(),
    )
    panel.require_model_input()

    universe = screen.universe.reindex(index=sessions, columns=tickers)
    first_universe = screen.universe.iloc[0].reindex(tickers).fillna(False).astype(bool)
    frozen = pd.Series(False, index=sessions, name="universe_frozen")
    before = sessions < sample_start
    universe.loc[before] = np.broadcast_to(
        first_universe.to_numpy(), (int(before.sum()), len(tickers))
    )
    frozen.loc[before] = True
    universe = universe.fillna(False).astype(bool)

    close_w = close.reindex(index=sessions, columns=tickers).astype(float)
    volume_w = volume.reindex(index=sessions, columns=tickers).astype(float)
    dividends = _dividends(manifest, actions_entry)
    total = total_returns(close_w, dividends)
    # Ken French publishes RF in percent per trading day -- already the accrual
    # over one session, so the conversion is a division by 100 (the same
    # reading as mafrm.factors.macro's).
    rf = (french.load_daily_rf() / 100.0).reindex(sessions)
    rf.name = "risk_free"
    excess = total.sub(rf, axis=0)
    excess = excess.where(np.broadcast_to(rf.notna().to_numpy()[:, None], excess.shape))

    return Inputs(
        excess=excess,
        risk_free=rf,
        volume=volume_w,
        shares=panel.shares.reindex(index=sessions, columns=tickers),
        cap=panel.cap.reindex(index=sessions, columns=tickers),
        universe=universe,
        universe_frozen=frozen,
        panel=panel,
        screen=screen,
        dividends=dividends.reindex(index=sessions, columns=tickers),
        close=close_w,
        sample_start=sample_start,
        holdout_start=holdout_start,
    )


def build(
    cfg: config_mod.Config | None = None,
    *,
    inputs: Inputs | None = None,
    winsor_bound: float | None = None,
) -> tuple[Exposures, RawDescriptors, Inputs]:
    """The six exposures on every session from ``sample.start`` to the holdout boundary."""
    cfg = cfg or config_mod.load()
    data = inputs or load_inputs(cfg)
    desc = cfg.model.equity_descriptors
    raw = raw_descriptors(
        excess=data.excess,
        risk_free=data.risk_free,
        volume=data.volume,
        shares=data.shares,
        cap=data.cap,
        universe=data.universe,
        descriptors=desc,
        days_per_month=cfg.model.data.trading_days_per_month,
    )
    bound = desc.winsorization.bound_sd if winsor_bound is None else winsor_bound
    exposures = exposures_from_raw(
        raw,
        cap=data.cap,
        universe=data.universe,
        descriptors=desc,
        winsor_bound=bound,
        sample_start=data.sample_start,
    )
    return exposures, raw, data


def holdout_boundary(cfg: config_mod.Config | None = None) -> date:
    """The boundary every frame this module emits stops before (CLAUDE.md invariant 5)."""
    return (cfg or config_mod.load()).require_holdout_start()
