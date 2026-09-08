"""The optimizer's fixed expected-return input, and SPEC.md 8.3's alpha-risk decomposition. W6-P1.

THIS MODULE DOES NOT FORECAST RETURNS. CLAUDE.md admits expected returns only as
"a fixed, documented input to the optimizer so that it has something to trade
against -- never as something to improve". The input is RSTR, SPEC.md 15.4's
momentum descriptor from the CNE5 Descriptor Details, fixed by operator ruling 1
(SPEC.md 8.5.1) and never tuned or compared against alternatives on performance:

    RSTR_t = sum_{j=0}^{T-1} w_j * x_{t-L-j} / sum_j w_j ,   x_s = ln(1+r_s) - ln(1+r_f,s)

with ``T = 504``, half-life 126 (``w_j = 0.5^{j/126}``) and lag ``L = 21`` -- the
sum runs over the 504 sessions ending 21 sessions before ``t``, skipping the most
recent month, which is the part replicas get wrong (CLAUDE.md parameter table).
All three constants are ``equity_descriptors.momentum`` in ``config/model.yaml``,
reached through ``optimizer.alpha``'s pointers.

TWO CONSTRUCTIONS, FORCED BY DIMENSIONS AND RECORDED AT SPEC.md 8.5.1
    - **The weights are normalised**, so RSTR is the weighted MEAN daily log
      excess return -- a RATE in return/day -- rather than a sum. SPEC.md 8.1
      adds ``alpha'z`` to a cost that is a fraction of NAV and SPEC.md 8.4's
      ``lambda = IR/(2 TE)`` presumes ``alpha`` in return units; a dimensionless
      score would make the alpha-versus-cost trade-off depend on an arbitrary
      scale. :func:`to_horizon` then multiplies the rate by the rebalance
      period's day count so alpha, risk and cost share one horizon.
    - **The cross-section is CENTRED and not scaled.** SPEC.md 15.5's centring,
      with an equal-weighted mean because the curve points carry no market cap;
      not the scaling to unit standard deviation, which would destroy the units.
      Under ``1'w = 1`` the level of alpha is untradeable anyway, and leaving it
      in would only inflate SPEC.md 8.4's ``IR``.

Nothing here reads the cache. The caller supplies the return panel, and the RSTR
at ``t`` reads rows ``t-L-T+1 .. t-L`` only -- asserted by the no-look-ahead test
in ``tests/test_optimizer.py`` through the generic harness of
:mod:`mafrm.backtest.lookahead`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "AlphaError",
    "Misalignment",
    "MisalignmentPenalty",
    "decompose",
    "demean",
    "log_excess_returns",
    "misalignment_penalty",
    "rstr",
    "rstr_weights",
    "to_horizon",
]

FloatArray = NDArray[np.float64]


class AlphaError(ValueError):
    """The alpha input cannot be built as asked. Nothing is filled or defaulted."""


# ---------------------------------------------------------------------------
# RSTR
# ---------------------------------------------------------------------------


def log_excess_returns(total_return: pd.DataFrame, risk_free: pd.Series) -> pd.DataFrame:
    """``ln(1 + r_t) - ln(1 + r_f,t)`` per asset and date. CNE5's summand, exactly.

    ``total_return`` is (date x asset) in decimal per session; ``risk_free`` is the
    decimal per-session bill return on (a superset of) the same dates. A date on
    which the bill rate is missing gets ``NaN`` for every asset -- never a filled
    rate -- and a date on which an asset did not trade stays ``NaN`` for it.
    """
    if not isinstance(total_return.index, pd.DatetimeIndex):
        raise AlphaError("log_excess_returns: total_return must be indexed by DatetimeIndex")
    rate = risk_free.reindex(total_return.index)
    if (total_return <= -1.0).any().any():
        raise AlphaError("log_excess_returns: a total return at or below -100% has no logarithm")
    logged = (
        np.log1p(total_return.to_numpy(dtype=float)) - np.log1p(rate.to_numpy(dtype=float))[:, None]
    )
    return pd.DataFrame(logged, index=total_return.index, columns=total_return.columns)


def rstr_weights(window: int, halflife: int) -> FloatArray:
    """``0.5^{j/halflife}`` for ``j = 0 .. window-1``, normalised to sum to one.

    ``j = 0`` is the most recent session INSIDE the lagged window (``t - L``), so
    the weights are those CNE5 writes up to the constant ``0.5^{L/halflife}`` --
    which the normalisation removes. Half-life in sessions.
    """
    if window < 1 or halflife < 1:
        raise AlphaError(f"rstr_weights: window {window} and halflife {halflife} must be >= 1")
    ages = np.arange(window, dtype=float)
    raw = np.power(0.5, ages / float(halflife))
    weights: FloatArray = raw / raw.sum()
    return weights


def rstr(log_excess: pd.DataFrame, *, window: int, halflife: int, lag: int) -> pd.DataFrame:
    """SPEC.md 15.4's RSTR at every date, as a per-session RATE. ``NaN`` until the window is full.

    Row ``t`` reads rows ``t-lag-window+1 .. t-lag`` inclusive and nothing after
    ``t-lag``. A window containing any ``NaN`` produces ``NaN`` -- no filling, no
    shortened windows (the W1-P5 ``require_full_window`` reasoning: a descriptor
    over 30 of 504 sessions is not the same object as one over 504).
    """
    if lag < 0:
        raise AlphaError(f"rstr: lag must be >= 0, got {lag}")
    if not isinstance(log_excess.index, pd.DatetimeIndex):
        raise AlphaError("rstr: log_excess must be indexed by DatetimeIndex")
    if not log_excess.index.is_monotonic_increasing or not log_excess.index.is_unique:
        raise AlphaError("rstr: the index must be strictly increasing")
    weights = rstr_weights(window, halflife)
    values = log_excess.to_numpy(dtype=float)
    periods, assets = values.shape
    out = np.full((periods, assets), np.nan)
    if periods >= window + lag:
        # sliding_window_view over the LAGGED series: slot t sees x[t-lag-window+1 .. t-lag].
        windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
        # windows[s] covers rows s .. s+window-1; the RSTR at t uses s = t - lag - window + 1.
        # The most recent row in the window (index window-1) gets weight j = 0.
        scored = windows @ weights[::-1]
        first_t = lag + window - 1
        out[first_t:] = scored[: periods - first_t]
    return pd.DataFrame(out, index=log_excess.index, columns=log_excess.columns)


def demean(alpha: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Equal-weighted cross-sectional centring, per date. SPEC.md 15.5's mean, nothing else.

    A date on which any asset is ``NaN`` is left entirely ``NaN``: a mean over a
    partial cross-section would centre the survivors on a different universe.
    """
    if isinstance(alpha, pd.Series):
        if alpha.isna().any():
            return alpha * np.nan
        return alpha - float(alpha.mean())
    complete = alpha.notna().all(axis=1)
    out = alpha.sub(alpha.mean(axis=1), axis=0)
    out.loc[~complete, :] = np.nan
    return out


def to_horizon(rate: pd.DataFrame | pd.Series, *, days: int) -> pd.DataFrame | pd.Series:
    """Scale a per-session rate to a ``days``-session rebalance period. Linear, as a mean is."""
    if days < 1:
        raise AlphaError(f"to_horizon: days must be >= 1, got {days}")
    return rate * float(days)


# ---------------------------------------------------------------------------
# SPEC.md 8.3 -- alpha-risk misalignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Misalignment:
    """``alpha = alpha_R + alpha_perp`` against the risk model's design ``X``. SPEC.md 8.3.

    ``alpha_R = X (X'X)^-1 X' alpha`` is the part the factors span -- a tilt along
    it incurs factor risk the model can see. ``alpha_perp`` is the remainder, whose
    risk the model sees only through the diagonal ``Delta``. ``cos_theta =
    ||alpha_R|| / ||alpha||`` is the misalignment angle reported at every
    rebalance; it is 1 when alpha lies in the factor span and 0 when the model
    cannot see any of it.
    """

    spanned: FloatArray
    orthogonal: FloatArray
    cos_theta: float

    @property
    def alpha(self) -> FloatArray:
        total: FloatArray = self.spanned + self.orthogonal
        return total


def decompose(alpha: FloatArray, exposures: FloatArray) -> Misalignment:
    """Project ``alpha`` onto the column span of ``exposures`` (``N x K``).

    Least squares rather than an explicit ``(X'X)^-1``: the design is 13 x 6 and
    well posed here, but ``lstsq`` is the right tool for a projection whatever the
    conditioning, and it is what makes the K = 56 equity design of week 7 a
    non-event.
    """
    a = np.asarray(alpha, dtype=float)
    x = np.asarray(exposures, dtype=float)
    if x.ndim != 2 or a.shape != (x.shape[0],):
        raise AlphaError(f"decompose: alpha {a.shape} against an N x K design {x.shape}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(x)):
        raise AlphaError("decompose: alpha and exposures must be finite")
    coefficients, *_ = np.linalg.lstsq(x, a, rcond=None)
    spanned: FloatArray = x @ coefficients
    orthogonal: FloatArray = a - spanned
    norm = float(np.linalg.norm(a))
    cos_theta = float(np.linalg.norm(spanned) / norm) if norm > 0.0 else 0.0
    return Misalignment(spanned=spanned, orthogonal=orthogonal, cos_theta=min(cos_theta, 1.0))


@dataclass(frozen=True)
class MisalignmentPenalty:
    """SPEC.md 8.3's penalty, carried as ``psi_unit * (direction' w)^2``.

    Algebraically ``psi_mis * (alpha_perp' w)^2`` with ``psi_mis = psi_unit /
    ||alpha_perp||^2`` -- the form SPEC.md 8.3 writes -- but carried on the UNIT
    direction so that a small ``alpha_perp`` never puts ``1 / ||alpha_perp||^2``
    into the solver. When ``alpha`` lies in the factor span to within rounding,
    ``direction`` is zero and the term vanishes.
    """

    #: ``lambda * u' Sigma u``, the penalty per unit of squared exposure along ``u``.
    psi_unit: float
    #: ``u = alpha_perp / ||alpha_perp||``, or zeros when there is nothing to penalise.
    direction: FloatArray

    @property
    def active(self) -> bool:
        return self.psi_unit > 0.0

    @property
    def psi_mis(self) -> float:
        """MSCI's coefficient on ``(alpha_perp' w)^2``, for the report. ``NaN`` when inactive."""
        norm_squared = float(self.direction @ self.direction)
        if not self.active or norm_squared == 0.0:
            return float("nan")
        return self.psi_unit / self._alpha_perp_norm_squared

    _alpha_perp_norm_squared: float = 0.0


def misalignment_penalty(
    decomposition: Misalignment, covariance: FloatArray, *, gamma_risk: float
) -> MisalignmentPenalty:
    """``psi_mis`` for ``psi_mis * (alpha_perp' w)^2``: MSCI's ``lambda * sigma^2(alpha_perp)``.

    THE READING OF ``sigma^2(alpha_perp)`` IS A CONSTRUCTION (SPEC.md 8.5.1)
        The repository does not hold MSCI's paper. ``sigma^2(alpha_perp)`` is
        taken as the model's own forecast variance of a UNIT position along
        ``alpha_perp`` -- ``u' Sigma u`` with ``u = alpha_perp / ||alpha_perp||``
        -- so that the penalty prices that direction at twice what the model
        sees. The division by ``||alpha_perp||^2`` makes ``psi_mis (alpha_perp'
        w)^2 = lambda * sigma_u^2 * (u'w)^2`` a squared weight times a variance
        times ``lambda``, the same units as ``lambda w' Sigma w``; without it the
        term would carry alpha's units squared and be incommensurate.

    ``alpha_perp`` IS ZERO WHEN IT IS ROUNDING
        A least-squares residual of an alpha inside the factor span is not zero
        but ``O(eps * ||alpha||)``, and normalising it would manufacture a
        direction out of noise (found by ``tests/test_optimizer.py``). The
        floor is ``N * eps * ||alpha||`` -- machine epsilon times the length of
        the sum, the same construction as SPEC.md 5.2.4's PSD threshold -- and
        below it the penalty is inactive.
    """
    perp = np.asarray(decomposition.orthogonal, dtype=float)
    alpha_norm = float(np.linalg.norm(decomposition.alpha))
    perp_norm = float(np.linalg.norm(perp))
    floor = perp.size * np.finfo(float).eps * alpha_norm
    zeros = np.zeros_like(perp)
    if perp_norm <= floor or gamma_risk == 0.0:
        return MisalignmentPenalty(psi_unit=0.0, direction=zeros)
    sigma = np.asarray(covariance, dtype=float)
    if sigma.shape != (perp.size, perp.size):
        raise AlphaError(f"misalignment_penalty: covariance {sigma.shape} against N = {perp.size}")
    unit: FloatArray = perp / perp_norm
    variance = float(unit @ sigma @ unit)
    if variance < 0.0:
        raise AlphaError(f"misalignment_penalty: u' Sigma u = {variance} < 0; Sigma is not PSD")
    return MisalignmentPenalty(
        psi_unit=float(gamma_risk * variance),
        direction=unit,
        _alpha_perp_norm_squared=perp_norm**2,
    )
