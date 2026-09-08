"""Corroborating the Yahoo price series against sources that share nothing with it.

SPEC.md 3.5's first trap is that Yahoo rewrites adjusted-close history backwards
on every dividend, so a series built from it can be wrong in a way that leaves no
trace inside Yahoo. The only defence that is not circular is a comparand from a
different vendor, with a different dividend convention, reached by a different
code path. Stooq was to have been that comparand and is unreachable, so these two
are the independence the project actually has:

``spy_versus_market``
    SPY's total return -- computed by :mod:`mafrm.data.prices` from raw
    unadjusted bars and our own dividend handling -- against Ken French's
    ``Mkt-RF``, the CRSP value-weighted US market excess return. CRSP is not
    Yahoo, its dividend treatment is its own, and the two series touch at no
    point in this repository.

``tlt_versus_ladder``
    TLT's Yahoo closes against a par-coupon ladder priced off the Fed's published
    Svensson parameters (W1-P3). Gated on the regression **beta**, which asks
    whether the price series still moves one-for-one with a curve-derived
    comparand. That is a different question from W1-P3's mean-gap gate, which
    asks whether our *ladder construction* is right and remains open.

Both report the measured numbers rather than a verdict alone. Their tolerances
live in ``config/model.yaml`` under ``data.cross_checks`` and each is derived
from what it must discriminate -- a dropped dividend stream, a missed split --
rather than fitted to what was measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mafrm.config import load
from mafrm.data import cache, french, prices

__all__ = ["MarketCrossCheck", "spy_versus_market"]


@dataclass(frozen=True)
class MarketCrossCheck:
    """SPY's own excess return against the CRSP market excess return."""

    ticker: str
    observations: int
    start: pd.Timestamp
    end: pd.Timestamp
    correlation: float
    #: Arithmetic mean difference, annualised at ``trading_days_per_year``.
    mean_gap_pct_per_year: float
    #: Largest single-day absolute difference, in percent.
    max_abs_daily_pct: float
    #: The date it occurred on -- a real dispersion day should be identifiable.
    max_abs_daily_date: pd.Timestamp
    #: Scale for the gap: a dropped dividend stream would show as this.
    dividend_yield_pct_per_year: float

    @property
    def passed(self) -> bool:
        limits = load().model.data.cross_checks.spy_versus_market
        return (
            self.correlation >= limits.min_correlation
            and abs(self.mean_gap_pct_per_year) <= limits.max_abs_mean_gap_pct_per_year
            and self.max_abs_daily_pct <= limits.max_abs_daily_difference_pct
        )

    def render(self) -> str:
        limits = load().model.data.cross_checks.spy_versus_market
        return (
            f"{self.ticker} vs {limits.benchmark_column}: n={self.observations:,} "
            f"{self.start.date()}..{self.end.date()}; corr {self.correlation:.4f} "
            f"(min {limits.min_correlation}); mean gap {self.mean_gap_pct_per_year:+.4f}%/yr "
            f"(limit {limits.max_abs_mean_gap_pct_per_year}, dividend signature "
            f"{self.dividend_yield_pct_per_year:.3f}); worst day "
            f"{self.max_abs_daily_pct:.3f}% on {self.max_abs_daily_date.date()} "
            f"(limit {limits.max_abs_daily_difference_pct}) -- "
            f"{'pass' if self.passed else 'FAIL'}"
        )


#: Why this module reads past the holdout boundary. It asks whether a VENDOR
#: series agrees with an independent one over everything cached -- a question
#: about the data, not about the model -- and its reported n and window are
#: pinned by experiments.md rows 41-42. Nothing here reaches a factor, a
#: covariance or a strategy. See mafrm.data.holdout.
_REASON = "data-layer vendor cross-check: SPY vs CRSP over the full cached history"


def _cached(source: str, name: str) -> pd.DataFrame:
    manifest = cache.Manifest.load()
    return cache.read_unrestricted(
        manifest.latest(source=source, name=name), manifest=manifest, reason=_REASON
    )


def spy_versus_market() -> MarketCrossCheck:
    """Compare our SPY total return with Ken French's CRSP market excess return.

    Both legs are put on the same footing before comparison: our total return has
    Ken French's own risk-free rate subtracted, so what is compared is two
    estimates of the same quantity -- the excess return on US equity -- and any
    difference is index composition or a defect, not a cash-rate convention.
    """
    config = load().model.data
    limits = config.cross_checks.spy_versus_market
    ticker = limits.ticker

    raw = _cached("yfinance", f"{ticker.lower()}_prices")
    actions = _cached("yfinance", f"{ticker.lower()}_actions")
    ours = prices.total_return(raw, actions)

    factors = french.load_daily_factors_unrestricted(reason=_REASON)
    common = ours.index.intersection(factors.index)
    if len(common) == 0:
        raise ValueError(f"{ticker} and the Ken French table share no dates")

    rf_column = config.risk_free.ken_french_daily_rf.rf_column
    # Ken French publishes both columns in percent per trading day.
    ours_excess = ours.loc[common] - factors.loc[common, rf_column] / 100.0
    theirs = factors.loc[common, limits.benchmark_column] / 100.0
    difference = ours_excess - theirs

    # Median annual dividend yield: the scale a dropped dividend stream would
    # show up at, and therefore what the mean-gap tolerance has to sit below.
    index = pd.DatetimeIndex(raw.index)
    dividends = actions["Dividends"].reindex(index).fillna(0.0)
    by_year = dividends.groupby(index.year).sum() / raw["Close"].groupby(index.year).mean()

    worst = difference.abs().idxmax()
    return MarketCrossCheck(
        ticker=ticker,
        observations=len(common),
        start=pd.Timestamp(common.min()),
        end=pd.Timestamp(common.max()),
        correlation=float(ours_excess.corr(theirs)),
        mean_gap_pct_per_year=float(difference.mean() * config.trading_days_per_year * 100.0),
        max_abs_daily_pct=float(difference.abs().max() * 100.0),
        max_abs_daily_date=pd.Timestamp(worst),
        dividend_yield_pct_per_year=float(by_year.median() * 100.0),
    )
