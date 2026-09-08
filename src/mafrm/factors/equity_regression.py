"""SPEC.md 15.6's daily cross-sectional regression. USE4 Eq. 3.1 and 3.3 (W7-P3).

``r_n = f_c + sum_i X_ni f_i + sum_s X_ns f_s + u_n`` on every session, by
weighted least squares with ``w_n`` proportional to ``sqrt(cap_n)`` normalised
to one -- MSCI's stated assumption that specific variance is inversely
proportional to the square root of market cap (CLAUDE.md failure mode 8) --
under the CAP-weighted industry sum-to-zero constraint ``sum_i w_i f_i = 0``
of Eq. 3.3, implemented through a ``K x (K-1)`` matrix ``R`` with ``f = R f~``
and ``f~ = (R'X'WXR)^-1 R'X'W r``. Industries are Fama-French 49 from SIC codes
(:mod:`mafrm.data.sic`); every ruling is at SPEC.md 15.6.1 and every number
comes from ``config/model.yaml`` (``equity_regression``).

Timing (ruling 4): exposures, caps and industry shares at the close of ``t-1``
against excess returns over ``t``. No thin-industry rule (ruling 3): an
industry with no member on a date is absent from that date's regression and
its factor return is NaN; a one-member industry is estimated as-is.

The identity the constraint buys, derived at SPEC.md 15.6.1 and asserted here
on every date to ``identity_tolerance``::

    c'r = f_c + sum_i w_i f_i + sum_s (c'x_s) f_s + c'u

with ``c`` the cap weights of the regression set. The second term is zero by
the constraint; the last is the market's specific return, which sqrt-cap
weights do not zero. Both are measured, not assumed.

Nothing here reads the network; the panel comes from
:func:`mafrm.factors.equity.build` and the cache through ``cache.read`` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from mafrm import config as config_mod
from mafrm.data import market_cap, sic
from mafrm.factors import equity

__all__ = [
    "COUNTRY",
    "DayResult",
    "FactorReturns",
    "RegressionError",
    "build",
    "constraint_matrix",
    "industry_labels",
    "regress_day",
    "run",
]

#: The country factor's column name; industries carry French's abbreviations; styles theirs.
COUNTRY: Final[str] = "COUNTRY"

#: Per-date summary columns, in order.
SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "names",
    "industries_present",
    "thin_industries",
    "thin_names",
    "eliminated_industry",
    "r2_weighted",
    "r2_equal",
    "cond_full",
    "cond_style",
    "industry_term",
    "country",
    "cap_weighted_return",
    "style_term",
    "cap_weighted_specific",
    "identity_gap",
    "market_excess",
    "reclassified",
    "with_cap",
    "excluded_no_sic",
    "excluded_no_exposure",
    "excluded_no_return",
)


class RegressionError(ValueError):
    """A date the regression cannot be run on, or an inconsistent input."""


# ---------------------------------------------------------------------------
# Pure pieces
# ---------------------------------------------------------------------------


def constraint_matrix(industry_weights: np.ndarray, eliminate: int) -> np.ndarray:
    """USE4 Eq. 3.3's ``R`` for the industry block alone: ``I x (I-1)``.

    ``industry_weights`` are the cap shares ``w_i`` of the present industries
    (positive, summing to one); ``eliminate`` is the index of the industry
    written as a function of the others, ``f_e = -sum_{i != e} (w_i / w_e) f~_i``.
    Any ``e`` with ``w_e > 0`` gives the same constrained solution; the caller
    picks the largest for numerical stability.
    """
    w = np.asarray(industry_weights, dtype=float)
    count = w.shape[0]
    if count == 0:
        return np.zeros((0, 0))
    if not (w > 0).all():
        raise RegressionError("constraint_matrix: every present industry needs a positive weight")
    if not 0 <= eliminate < count:
        raise RegressionError(f"constraint_matrix: eliminate {eliminate} outside 0..{count - 1}")
    free = [i for i in range(count) if i != eliminate]
    r = np.zeros((count, count - 1))
    for col, i in enumerate(free):
        r[i, col] = 1.0
        r[eliminate, col] = -w[i] / w[eliminate]
    return r


@dataclass(frozen=True)
class DayResult:
    """One date's regression. Arrays are over the PRESENT industries and the styles given."""

    #: Present industry ids (as passed in), in the order of ``industry_returns``.
    present: np.ndarray
    members: np.ndarray
    eliminated: int
    country: float
    industry_returns: np.ndarray
    style_returns: np.ndarray
    country_t: float
    industry_t: np.ndarray
    style_t: np.ndarray
    residuals: np.ndarray
    r2_weighted: float
    r2_equal: float
    cond_full: float
    cond_style: float
    #: ``sum_i w_i f_i`` -- zero to tolerance by the constraint.
    industry_term: float
    #: ``c'r``, ``sum_s (c'x_s) f_s``, ``c'u`` and ``c'r - f_c``.
    cap_weighted_return: float
    style_term: float
    cap_weighted_specific: float
    identity_gap: float
    columns: int


def regress_day(
    returns: np.ndarray,
    styles: np.ndarray,
    industry: np.ndarray,
    cap: np.ndarray,
    *,
    weight_exponent: float,
    tolerance: float,
) -> DayResult:
    """The constrained WLS on one date. Pure numpy.

    ``returns`` (n,), ``styles`` (n, S), ``industry`` (n,) integer ids, ``cap``
    (n,) positive. Every row is in the regression set: the caller has already
    dropped names without a cap, an exposure, an industry or a return.
    """
    r = np.asarray(returns, dtype=float)
    s = np.asarray(styles, dtype=float)
    ind = np.asarray(industry)
    c_raw = np.asarray(cap, dtype=float)
    n = r.shape[0]
    if s.shape[0] != n or ind.shape[0] != n or c_raw.shape[0] != n:
        raise RegressionError("regress_day: inputs disagree on the name count")
    if n == 0:
        raise RegressionError("regress_day: no names")
    if not (np.isfinite(r).all() and np.isfinite(s).all() and np.isfinite(c_raw).all()):
        raise RegressionError("regress_day: a non-finite input reached the regression")
    if not (c_raw > 0).all():
        raise RegressionError("regress_day: every name needs a positive cap")

    present, codes = np.unique(ind, return_inverse=True)
    count = present.shape[0]
    dummies = np.zeros((n, count))
    dummies[np.arange(n), codes] = 1.0
    members = dummies.sum(axis=0)

    w = c_raw**weight_exponent
    w = w / w.sum()
    c = c_raw / c_raw.sum()
    industry_weights = c @ dummies
    eliminated = int(np.argmax(industry_weights))
    r_ind = constraint_matrix(industry_weights, eliminated)

    n_styles = s.shape[1]
    x = np.hstack([np.ones((n, 1)), dummies, s])
    k = 1 + count + n_styles
    big_r = np.zeros((k, k - 1))
    big_r[0, 0] = 1.0
    big_r[1 : 1 + count, 1 : 1 + count - 1] = r_ind
    big_r[1 + count :, count:] = np.eye(n_styles)
    xr = x @ big_r
    k_free = xr.shape[1]
    if n <= k_free:
        raise RegressionError(f"regress_day: {n} names cannot estimate {k_free} free parameters")

    wxr = xr * w[:, None]
    a = wxr.T @ xr
    b = wxr.T @ r
    f_free = np.linalg.solve(a, b)
    f = big_r @ f_free
    u = r - x @ f
    sigma2 = float((w * u * u).sum() / (n - k_free))
    var_free = sigma2 * np.linalg.inv(a)
    var_f = big_r @ var_free @ big_r.T
    sd = np.sqrt(np.clip(np.diag(var_f), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(sd > 0, f / sd, np.nan)

    r_bar_w = float((w * r).sum())
    ss_w = float((w * (r - r_bar_w) ** 2).sum())
    r2_w = 1.0 - float((w * u * u).sum()) / ss_w if ss_w > 0 else np.nan
    ss_e = float(((r - r.mean()) ** 2).sum())
    r2_e = 1.0 - float((u * u).sum()) / ss_e if ss_e > 0 else np.nan

    sqrt_w = np.sqrt(w)[:, None]
    cond_full = float(np.linalg.cond(sqrt_w * xr))
    cond_style = float(np.linalg.cond(sqrt_w * s)) if n_styles else np.nan

    f_c = float(f[0])
    f_ind = f[1 : 1 + count]
    f_sty = f[1 + count :]
    industry_term = float(industry_weights @ f_ind)
    cap_weighted_return = float(c @ r)
    style_term = float((c @ s) @ f_sty) if n_styles else 0.0
    cap_weighted_specific = float(c @ u)
    closing = cap_weighted_return - (f_c + industry_term + style_term + cap_weighted_specific)
    if abs(industry_term) > tolerance:
        raise RegressionError(
            f"the cap-weighted industry sum is {industry_term:.3e}, above {tolerance:g}: "
            "the constraint did not bind"
        )
    if abs(closing) > tolerance:
        raise RegressionError(
            f"the identity c'r = f_c + sum w_i f_i + sum (c'x_s) f_s + c'u fails by "
            f"{closing:.3e}, above {tolerance:g}"
        )
    return DayResult(
        present=present,
        members=members,
        eliminated=int(present[eliminated]),
        country=f_c,
        industry_returns=f_ind,
        style_returns=f_sty,
        country_t=float(t[0]),
        industry_t=t[1 : 1 + count],
        style_t=t[1 + count :],
        residuals=u,
        r2_weighted=float(r2_w),
        r2_equal=float(r2_e),
        cond_full=cond_full,
        cond_style=cond_style,
        industry_term=industry_term,
        cap_weighted_return=cap_weighted_return,
        style_term=style_term,
        cap_weighted_specific=cap_weighted_specific,
        identity_gap=cap_weighted_return - f_c,
        columns=k_free,
    )


def industry_labels(siccodes: pd.DataFrame) -> dict[int, str]:
    """Industry index -> French's abbreviation (``Agric``, ``Oil``, ...), from the parsed file."""
    pairs = siccodes[["industry", "abbrev"]].drop_duplicates()
    if pairs["industry"].duplicated().any():
        raise RegressionError("industry_labels: an industry carries two abbreviations")
    return {int(i): str(a) for i, a in pairs.itertuples(index=False)}


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorReturns:
    """The equity factor set and everything the report needs to explain it."""

    #: Sessions x [COUNTRY, industries (abbreviations), styles]; an absent industry is NaN.
    returns: pd.DataFrame
    tstats: pd.DataFrame
    #: Sessions x tickers; NaN outside the regression set.
    residuals: pd.DataFrame
    #: Sessions x industries: member counts (0 when absent).
    members: pd.DataFrame
    summary: pd.DataFrame
    #: Per ticker: cik, sic, sic_description, industry, status
    #: (:func:`mafrm.data.sic.industries_for`).
    industries: pd.DataFrame
    drift: pd.DataFrame
    #: Sessions x tickers point-in-time industry ids (SPEC.md 15.6.3), the exposure the
    #: regression read at t-1; equals the current industry everywhere for undrifted names.
    industries_pit: pd.DataFrame
    #: Per drifted ticker: regression cells whose industry the rule changed, and the sessions.
    reclassified: pd.DataFrame
    labels: dict[int, str]
    style_factors: tuple[str, ...]
    weight_exponent: float
    tolerance: float
    thin_max_members: int
    sample_start: pd.Timestamp
    holdout_start: pd.Timestamp
    #: Names in the estimation universe whose industry the regression never saw (no SIC).
    no_sic_names: tuple[str, ...] = field(default_factory=tuple)

    @property
    def industry_columns(self) -> tuple[str, ...]:
        return tuple(
            c for c in self.returns.columns if c != COUNTRY and c not in self.style_factors
        )


def run(
    exposures: equity.Exposures,
    inputs: equity.Inputs,
    industries: pd.DataFrame,
    drift: pd.DataFrame,
    labels: dict[int, str],
    *,
    regression: config_mod.EquityRegressionConfig,
    industries_pit: pd.DataFrame | None = None,
) -> FactorReturns:
    """The regression on every session from the second sample session to the boundary.

    Regression set on ``t``: names with a screened cap in the estimation
    universe at ``t-1`` (:attr:`equity.Exposures.cap` is NaN elsewhere), with
    all style exposures at ``t-1``, with an industry AT ``t-1``, and with an
    excess return over ``t``. Each exclusion is counted per date, in that
    order. ``industries_pit`` is the sessions x tickers point-in-time industry
    table of SPEC.md 15.6.3; when absent, the current industry applies on
    every session (W7-P3's construction).
    """
    styles = tuple(equity.FACTORS)
    sessions = pd.DatetimeIndex(exposures.cap.index)
    if len(sessions) < 2:
        raise RegressionError("run: fewer than two sessions")
    tickers = pd.Index([str(t) for t in exposures.cap.columns], name="ticker")
    n = len(tickers)
    cap = exposures.cap.reindex(columns=tickers).to_numpy(dtype=float)
    expo = np.stack(
        [
            exposures.exposures[f].reindex(index=sessions, columns=tickers).to_numpy(dtype=float)
            for f in styles
        ],
        axis=2,
    )
    excess = inputs.excess.reindex(index=sessions, columns=tickers).to_numpy(dtype=float)
    ind_series = industries["industry"].reindex(tickers)
    current = np.where(ind_series.notna(), ind_series.fillna(-1).astype(int), -1)
    if industries_pit is None:
        pit = pd.DataFrame(
            np.broadcast_to(current, (len(sessions), n)).copy(), index=sessions, columns=tickers
        )
    else:
        pit = industries_pit.reindex(index=sessions, columns=tickers)
    ind_panel = np.where(pit.notna().to_numpy(), pit.fillna(-1).to_numpy(dtype=int), -1)
    changed = (ind_panel != current[None, :]) & (ind_panel >= 0) & (current[None, :] >= 0)
    market = exposures.market_excess.reindex(sessions)

    industry_ids = sorted(labels)
    industry_cols = [labels[i] for i in industry_ids]
    id_pos = {i: p for p, i in enumerate(industry_ids)}
    columns = [COUNTRY, *industry_cols, *styles]
    dates = sessions[1:]
    t_count = len(dates)
    ret = np.full((t_count, len(columns)), np.nan)
    tst = np.full((t_count, len(columns)), np.nan)
    resid = np.full((t_count, n), np.nan)
    memb = np.zeros((t_count, len(industry_ids)), dtype=int)
    summary_rows: list[dict[str, object]] = []
    seen_no_sic: set[str] = set()
    reclassified_cells = np.zeros(n, dtype=int)
    reclassified_first = np.full(n, -1, dtype=int)

    for p in range(1, len(sessions)):
        prev = p - 1
        ind = ind_panel[prev]
        base = np.isfinite(cap[prev]) & (cap[prev] > 0)
        has_ind = ind >= 0
        has_expo = np.isfinite(expo[prev]).all(axis=1)
        has_ret = np.isfinite(excess[p])
        no_sic = base & ~has_ind
        no_expo = base & has_ind & ~has_expo
        no_ret = base & has_ind & has_expo & ~has_ret
        use = base & has_ind & has_expo & has_ret
        seen_no_sic.update(tickers[no_sic].tolist())
        row: dict[str, object] = {
            "with_cap": int(base.sum()),
            "excluded_no_sic": int(no_sic.sum()),
            "excluded_no_exposure": int(no_expo.sum()),
            "excluded_no_return": int(no_ret.sum()),
            "market_excess": float(market.iloc[p]),
        }
        idx = np.flatnonzero(use)
        if idx.size == 0:
            summary_rows.append(row)
            continue
        moved = use & changed[prev]
        reclassified_cells += moved
        newly = moved & (reclassified_first < 0)
        reclassified_first[newly] = prev
        row["reclassified"] = int(moved.sum())
        day = regress_day(
            excess[p, idx],
            expo[prev][idx],
            ind[idx],
            cap[prev, idx],
            weight_exponent=regression.weight_exponent,
            tolerance=regression.identity_tolerance,
        )
        k = p - 1
        ret[k, 0] = day.country
        tst[k, 0] = day.country_t
        for j, industry_id in enumerate(day.present.tolist()):
            col = 1 + id_pos[int(industry_id)]
            ret[k, col] = day.industry_returns[j]
            tst[k, col] = day.industry_t[j]
            memb[k, id_pos[int(industry_id)]] = int(day.members[j])
        ret[k, 1 + len(industry_ids) :] = day.style_returns
        tst[k, 1 + len(industry_ids) :] = day.style_t
        resid[k, idx] = day.residuals
        thin = day.members <= regression.thin_industry_report_max_members
        row.update(
            {
                "names": int(idx.size),
                "industries_present": int(day.present.size),
                "thin_industries": int(thin.sum()),
                "thin_names": int(day.members[thin].sum()),
                "eliminated_industry": labels[day.eliminated],
                "r2_weighted": day.r2_weighted,
                "r2_equal": day.r2_equal,
                "cond_full": day.cond_full,
                "cond_style": day.cond_style,
                "industry_term": day.industry_term,
                "country": day.country,
                "cap_weighted_return": day.cap_weighted_return,
                "style_term": day.style_term,
                "cap_weighted_specific": day.cap_weighted_specific,
                "identity_gap": day.identity_gap,
            }
        )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows, index=dates).reindex(columns=list(SUMMARY_COLUMNS))
    summary.index.name = "date"
    hit = np.flatnonzero(reclassified_cells > 0)
    reclassified = pd.DataFrame(
        {
            "cells": reclassified_cells[hit],
            "first_session": [sessions[i] for i in reclassified_first[hit]],
            "industry_first_pit": [
                labels.get(int(ind_panel[0, i]), str(ind_panel[0, i])) for i in hit
            ],
            "industry_current": [labels.get(int(current[i]), str(current[i])) for i in hit],
        },
        index=pd.Index([str(tickers[int(i)]) for i in hit], name="ticker"),
    )
    return FactorReturns(
        returns=pd.DataFrame(ret, index=dates, columns=columns),
        tstats=pd.DataFrame(tst, index=dates, columns=columns),
        residuals=pd.DataFrame(resid, index=dates, columns=tickers),
        members=pd.DataFrame(memb, index=dates, columns=industry_cols),
        summary=summary,
        industries=industries,
        drift=drift,
        industries_pit=pit.astype("Int64"),
        reclassified=reclassified,
        labels=labels,
        style_factors=styles,
        weight_exponent=regression.weight_exponent,
        tolerance=regression.identity_tolerance,
        thin_max_members=regression.thin_industry_report_max_members,
        sample_start=exposures.sample_start,
        holdout_start=inputs.holdout_start,
        no_sic_names=tuple(sorted(seen_no_sic)),
    )


def build(
    cfg: config_mod.Config | None = None,
    *,
    exposures: equity.Exposures | None = None,
    inputs: equity.Inputs | None = None,
) -> tuple[FactorReturns, equity.Exposures, equity.Inputs]:
    """Exposures from :func:`mafrm.factors.equity.build`, industries from the cache, then
    :func:`run`.
    """
    cfg = cfg or config_mod.load()
    if exposures is None or inputs is None:
        exposures, _raw, inputs = equity.build(cfg)
    unassigned = cfg.model.data.ken_french_siccodes49.unassigned_industry
    siccodes = sic.load_siccodes()
    current = sic.load_current()
    first_10k = sic.load_first_10k()
    mapping = inputs.panel.mapping
    industries = sic.industries_for(mapping, current, siccodes, unassigned=unassigned)
    # W7-P2b ruling 3: a hand-table no_facts name has no cap and is out of the
    # regression already; its industry row is kept for the report.
    assert isinstance(inputs.panel, market_cap.MarketCapPanel)
    drift = sic.drift_table(industries, first_10k, siccodes, unassigned=unassigned)
    labels = industry_labels(siccodes)
    # SPEC.md 15.6.3: the drifted names' industry is point-in-time from their
    # in-sample 10-K headers; without the history table cached, the rule cannot
    # be applied and the build refuses rather than silently falling back.
    if cfg.model.equity_industries.point_in_time != "drifted_names_by_filing_date":
        raise RegressionError("equity_industries.point_in_time: unknown reading")
    history = sic.load_history()
    sessions = pd.DatetimeIndex(exposures.cap.index)
    tickers = [str(t) for t in exposures.cap.columns]
    pit = sic.point_in_time_industries(
        industries, history, sessions, siccodes, unassigned=unassigned, tickers=tickers
    )
    return (
        run(
            exposures,
            inputs,
            industries,
            drift,
            labels,
            regression=cfg.model.equity_regression,
            industries_pit=pit,
        ),
        exposures,
        inputs,
    )
