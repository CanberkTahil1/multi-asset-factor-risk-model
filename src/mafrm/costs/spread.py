"""Effective bid-ask spreads by the EDGE estimator. SPEC.md 3.4.

There is no free daily historical bid-ask series for ETFs, so the spread has to
be estimated from bars. This module implements SPEC.md 3.4's answer:

    Ardia, D., Guidotti, E. & Kroencke, T. (2024). "Efficient estimation of
    bid-ask spreads from open, high, low, and close prices." *Journal of
    Financial Economics* 161:103916.

EDGE is a generalised method-of-moments estimator of the **root-mean-square
effective spread** -- which is exactly the quantity a cost model wants, since a
trade crosses half the effective spread -- and it needs only OHLC. It supersedes
Corwin-Schultz and Roll, both of which are badly biased and routinely return
negative estimates for liquid ETFs. The estimation itself is delegated to the
authors' own MIT-licensed ``bidask`` package rather than reimplemented, so that
the numbers are theirs and not a transcription of their paper.

Two implementation choices matter more here than the choice of estimator, and
both come straight from SPEC.md 3.4.

**RAW, unadjusted OHLC.** A spread is a property of prices that actually traded.
Yahoo's dividend back-adjustment rescales the whole history by a factor that
changes on every ex-date, and the high-low range is rescaled with it, so an
adjusted bar's range is contaminated by dividends paid years later. Worse, the
contamination is *retroactive*: the same day's estimate changes when the next
dividend is paid. :func:`effective_spread` therefore reads the raw cache
(``auto_adjust=False``) and asserts on it -- see :mod:`mafrm.data.prices` for the
other half of that rule, where adjustment is applied deterministically for
returns and only for returns.

**Rolling 63-day windows, monthly step.** A single static spread makes the cost
model uselessly optimistic, because spreads blow out in exactly the crises when
the risk model most wants to trade. The time variation IS the contribution, and
``reports/spread_vs_vol.png`` is the artefact that shows it.

**Signed estimates, clipped once.** ``bidask`` computes the estimator on the
SQUARED spread ``s2`` and returns ``sqrt(|s2|)``, multiplying through by
``sign(s2)`` only when it is passed ``sign=True``. Its default is therefore
``|estimate|``, and averaging an absolute value is not an estimate of the thing
inside it: where the true spread is near zero and the estimate is noise ``X``
with standard deviation ``s``, the mean of ``|X|`` is ``s*sqrt(2/pi) = 0.798*s``
-- positive for every ``s``, and growing with the estimate's noise, which itself
grows with volatility. W1-P5 measured exactly that shape and attributed it to a
finite-sample property of EDGE; it was the absolute value. See experiments.md
rows 54-58.

This module therefore asks for signed estimates and treats the sign as
information. :func:`effective_spread` returns what the estimator returned,
negatives included, because the incidence of negatives is the diagnostic that
tells you whether a spread series is resolving a real spread or reporting its
own noise. Non-negativity is imposed once, at the boundary where a cost model
needs a number that is a spread, by :func:`clip_negative`. That clip is
one-sided and costs half the bias back -- ``E[max(X,0)] = s/sqrt(2*pi) =
0.399*s`` -- so anything that AVERAGES spreads must read the signed series.
:class:`SpreadPanel` carries both and never conflates them.

A note on units, because it is the easiest thing in this module to get wrong:
EDGE returns a **proportional** spread, where ``0.01`` means 1%. Everything here
stays in those units and conversion is explicit via :func:`to_basis_points`. The
issuer-disclosed medians used for level calibration are quoted in percent or
basis points depending on the issuer, which is precisely why no function in this
module silently returns "the spread" in unnamed units.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from bidask import edge, edge_rolling

from mafrm.data import calendar

__all__ = [
    "NegativeEstimates",
    "SpreadError",
    "SpreadPanel",
    "SpreadRecovery",
    "ZeroSpreadBias",
    "clip_negative",
    "count_negative",
    "effective_spread",
    "monthly_effective_spread",
    "monthly_spread_panel",
    "recovery_reference",
    "to_basis_points",
    "zero_spread_reference",
]

#: Raw OHLC columns EDGE consumes. ``bidask`` matches these case-insensitively;
#: they are asserted here so that a renamed upstream column fails loudly instead
#: of reaching the estimator as a silently missing input.
_OHLC = ("Open", "High", "Low", "Close")

#: One basis point as a proportion. Not a tunable -- a unit.
_BPS = 1e-4


class SpreadError(ValueError):
    """Bars cannot support an EDGE estimate."""


def _validate_ohlc(prices: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Check the raw bars are the shape EDGE assumes, and hand back just OHLC."""
    missing = [column for column in _OHLC if column not in prices.columns]
    if missing:
        raise SpreadError(
            f"{label}: raw bars are missing {missing}; got {list(prices.columns)}. EDGE needs "
            "open, high, low and close, and they must be the RAW unadjusted columns."
        )
    frame = pd.DataFrame(
        {column: pd.to_numeric(prices[column], errors="coerce") for column in _OHLC},
        index=prices.index,
    )

    populated = frame.dropna(how="all")
    if populated.empty:
        raise SpreadError(f"{label}: every OHLC row is missing, so no window can be estimated")
    if (populated <= 0).to_numpy().any():
        raise SpreadError(
            f"{label}: non-positive price in OHLC. EDGE takes logs; a zero or negative bar is a "
            "parse defect, not a market observation."
        )

    high_below_low = (populated["High"] < populated["Low"]).sum()
    if high_below_low:
        raise SpreadError(
            f"{label}: {high_below_low} bar(s) with High < Low. The columns are transposed or "
            "the parse is wrong; EDGE would return a meaningless estimate rather than fail."
        )
    outside = (
        (populated["Close"] > populated["High"])
        | (populated["Close"] < populated["Low"])
        | (populated["Open"] > populated["High"])
        | (populated["Open"] < populated["Low"])
    ).sum()
    if outside:
        raise SpreadError(
            f"{label}: {outside} bar(s) with open or close outside the high-low range. This is "
            "the signature of adjusted and unadjusted columns mixed in one frame."
        )
    return frame


def effective_spread(
    prices: pd.DataFrame,
    *,
    window: int,
    min_periods: int,
    signed: bool,
    label: str = "prices",
) -> pd.Series:
    """Rolling EDGE effective spread, one estimate per trading day.

    Ardia, Guidotti & Kroencke (2024), *JFE* 161:103916, via the authors'
    ``bidask`` package. The result is a **proportional** spread: ``0.0005`` is
    5bp. Windows are counted in rows, so a market holiday shortens the calendar
    span of a window but never its sample size.

    ``min_periods`` is the caller's policy, not the estimator's: EDGE is defined
    from ``costs.edge_spread.min_observations`` rows, but this project sets
    ``require_full_window`` and passes ``window``, because an estimate from six
    sessions plotted beside one from sixty-three invites the reader to compare
    them.

    ``signed`` is passed straight through to ``bidask`` as ``sign``. It is a
    required keyword and has no default on purpose: the package's own default is
    ``False``, which returns ``|estimate|`` and silently biases every average
    taken over the result upward by ``0.798`` times the estimate's noise. This
    project passes ``costs.edge_spread.signed_estimates``, which is ``true``.
    The returned series may therefore contain negative values, which are not
    spreads and are not meant to be -- they are the estimator saying it cannot
    resolve one. Impose non-negativity with :func:`clip_negative`, once, where a
    cost model actually needs it.
    """
    if window < 1:
        raise SpreadError(f"{label}: window must be positive, got {window}")
    if min_periods < 3:
        raise SpreadError(
            f"{label}: EDGE is undefined below 3 observations (Ardia, Guidotti & Kroencke 2024 "
            f"section 3), got min_periods={min_periods}"
        )
    if min_periods > window:
        raise SpreadError(
            f"{label}: min_periods {min_periods} exceeds window {window}, so no window could "
            "ever produce an estimate"
        )

    frame = _validate_ohlc(prices, label=label)
    # bidask matches column names case-insensitively; lowercased here so the
    # call does not depend on that behaviour continuing to hold.
    frame = frame.rename(columns=str.lower)
    estimates = edge_rolling(df=frame, window=window, min_periods=min_periods, sign=signed)
    spread = pd.Series(estimates, index=prices.index, dtype="float64")
    spread.name = "effective_spread"
    return spread


def monthly_effective_spread(
    prices: pd.DataFrame,
    *,
    window: int,
    min_periods: int,
    signed: bool,
    label: str = "prices",
) -> pd.Series:
    """:func:`effective_spread` sampled at each month's last traded date.

    This is the "rolling 63-day window, monthly step" of SPEC.md 3.4. Computing
    daily and then sampling is identical to estimating only on month ends -- the
    rolling estimate at date ``t`` reads exactly the trailing ``window`` rows
    either way -- and a test pins that equality against a direct
    :func:`bidask.edge` call so the claim is checked rather than asserted.

    The step is monthly and the window is 63 days, so consecutive observations
    overlap by roughly two thirds. That is deliberate and it is why these are a
    monthly *sample of a rolling estimator*, not 63-day non-overlapping blocks:
    overlapping windows are what let the series track a spread blow-out within
    the quarter it happens.
    """
    daily = effective_spread(
        prices, window=window, min_periods=min_periods, signed=signed, label=label
    )
    sampled = calendar.monthly(daily, label=label)
    assert isinstance(sampled, pd.Series)  # a Series in gives a Series out
    return sampled


def clip_negative(spread: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Reset negative estimates to zero. The ONE place non-negativity is imposed.

    A spread cannot be negative, so a cost model needs a series that is not. The
    estimator can still return a negative number, because it is a moment
    estimator and a negative estimate is what it says when the spread it is
    looking for is smaller than the noise it is looking through. Both facts are
    true at once, and this function is where they are reconciled -- deliberately
    as a named step rather than inside :func:`effective_spread`, so that the
    signed series survives for the diagnostics that need it.

    **This is not free and it is not a fix.** Clipping is one-sided, so on a
    market whose true spread is zero it leaves ``E[max(X,0)] = s/sqrt(2*pi) =
    0.399*s`` where the absolute-value default left ``0.798*s``: exactly half,
    measured at experiments.md row 55. Averaging a clipped series therefore
    still overstates the spread of a tight instrument, and any statistic that
    takes a mean or a median across assets or months should read the signed
    series. See :class:`SpreadPanel`, which hands back both.
    """
    return spread.clip(lower=0.0)


@dataclass(frozen=True)
class SpreadPanel:
    """Per-asset, per-month spreads in both forms, plus the negative incidence.

    Both series are returned from one call because they must not drift apart,
    and because choosing between them is a real decision that a caller should
    have to make explicitly:

    - :attr:`signed` is the estimator's output. Use it for anything that
      AVERAGES -- medians across a sleeve, crisis multiples, the level
      calibration against issuer-disclosed medians -- because averaging a
      one-sided series reintroduces the bias this module exists to remove.
    - :attr:`production` is :attr:`signed` with negatives reset to zero. Use it
      wherever a number has to be a spread, which in this project means the
      cost model and nothing else yet.
    - :attr:`negatives` counts, per asset, how often the estimator went below
      zero BEFORE the clip. That fraction is the honest read on whether a
      spread series is resolving a spread: near 50% means the estimator is
      reporting noise around zero, and near 0% on a liquid name is the signature
      of an absolute value having been taken somewhere upstream.
    """

    signed: pd.DataFrame
    production: pd.DataFrame
    negatives: Mapping[str, NegativeEstimates]


def monthly_spread_panel(
    prices_by_asset: Mapping[str, pd.DataFrame],
    *,
    window: int,
    min_periods: int,
    signed: bool,
    clip: bool,
    end: date | None = None,
) -> SpreadPanel:
    """Per-asset, per-month effective spreads on one calendar.

    Assets keep their own month-end dates until the final alignment, because a
    month end is a property of the asset's own trading calendar
    (:mod:`mafrm.data.calendar`). The union is taken at the end and never filled:
    a month before an asset existed is ``NaN``, not a carried spread.

    ``end`` is the holdout boundary. Callers pass
    ``config.require_holdout_start()``; the panel is half-open and stops strictly
    before it, per CLAUDE.md invariant 5.

    ``signed`` and ``clip`` come from ``costs.edge_spread.signed_estimates`` and
    ``costs.edge_spread.clip_negative_to_zero``. Negatives are counted before
    the clip, always, so the diagnostic survives the production policy.
    """
    if not prices_by_asset:
        raise SpreadError("monthly_spread_panel: no assets given")
    series = {
        asset: monthly_effective_spread(
            prices, window=window, min_periods=min_periods, signed=signed, label=asset
        )
        for asset, prices in prices_by_asset.items()
    }
    aligned = calendar.align(series, how="union", end=end)
    assert isinstance(aligned, pd.DataFrame)
    production = clip_negative(aligned) if clip else aligned
    assert isinstance(production, pd.DataFrame)
    return SpreadPanel(
        signed=aligned,
        production=production,
        negatives={asset: count_negative(aligned[asset]) for asset in aligned.columns},
    )


def to_basis_points(spread: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Proportional spread to basis points. ``0.0005 -> 5.0``."""
    return spread / _BPS


def _simulate_bars(
    rng: np.random.Generator,
    *,
    days: int,
    ticks_per_day: int,
    daily_volatility: float,
    overnight_gap_share: float,
    true_spread: float,
    price: float = 100.0,
) -> pd.DataFrame:
    """Synthetic raw OHLC with a KNOWN effective spread. The shared test market.

    A driftless geometric random walk in the efficient (mid) price, observed
    ``ticks_per_day`` times a session, with every trade placed half a spread
    either side of the mid at a random sign -- so the high-low range carries the
    bid-ask bounce EDGE is built to extract, and open/high/low/close are formed
    from traded prices exactly as an exchange would form them.

    ``overnight_gap_share`` is the fraction of daily VARIANCE realised as a jump
    between one session's close and the next session's open. At zero the two
    lines below collapse to one continuous walk, which is the process rows 48-51
    and 54-55 used.

    ``price`` is carried explicitly but does not affect any estimate: EDGE works
    on log prices, so the level cancels. It would matter only under tick
    discreteness, which this market does not impose -- see experiments.md row 61
    for why that omission is defensible for the post-decimalization sample and
    NOT for the pre-2001 one.
    """
    intraday_volatility = daily_volatility * np.sqrt(1.0 - overnight_gap_share)
    gap_volatility = daily_volatility * np.sqrt(overnight_gap_share)
    step = intraday_volatility / np.sqrt(ticks_per_day)

    # Within-session path, then the level each session opens at. Splitting them
    # is what lets the close-to-open jump exist at all.
    intraday = np.cumsum(rng.normal(0.0, step, size=(days, ticks_per_day)), axis=1)
    gaps = rng.normal(0.0, gap_volatility, size=days)
    opens = np.concatenate([[0.0], np.cumsum(intraday[:-1, -1] + gaps[1:])])
    mid = np.exp(opens[:, None] + intraday + np.log(price))

    signs = rng.choice([-1.0, 1.0], size=(days, ticks_per_day))
    traded = mid * (1.0 + signs * true_spread / 2.0)
    return pd.DataFrame(
        {
            "open": traded[:, 0],
            "high": traded.max(axis=1),
            "low": traded.min(axis=1),
            "close": traded[:, -1],
        }
    )


@dataclass(frozen=True)
class ZeroSpreadBias:
    """What EDGE returns on a market whose true spread is exactly zero.

    Four numbers rather than one, because the single number this used to return
    was ``absolute_mean`` and reading it as "the bias of EDGE" is what produced
    the W1-P5 misdiagnosis. They separate cleanly:

    - :attr:`signed_mean` is the estimator's actual bias. On a zero-spread
      market it should be approximately zero, and it is.
    - :attr:`absolute_mean` is what ``bidask``'s ``sign=False`` default returns.
      It is ``E|X| ~ 0.798*sd(X)`` and is positive by construction, whatever the
      estimator does.
    - :attr:`clipped_mean` is the bias the PRODUCTION series carries, after
      negatives are reset to zero. It is ``E[max(X,0)] ~ 0.399*sd(X)``, half of
      ``absolute_mean``, and it is the number the figure's band should show.
    - :attr:`negative_fraction` should sit near one half. It is the check that
      the estimator is centred rather than shifted.
    """

    signed_mean: float
    absolute_mean: float
    clipped_mean: float
    negative_fraction: float


def zero_spread_reference(
    *,
    daily_volatility: float,
    window: int,
    seed: int,
    days: int = 1500,
    ticks_per_day: int = 390,
    overnight_gap_share: float = 0.0,
) -> ZeroSpreadBias:
    """Expected EDGE estimate when the TRUE spread is zero. A diagnostic, not a model.

    The measurement is a simulation: a driftless geometric random walk observed
    ``ticks_per_day`` times a day, trades placed at the mid with no spread at
    all, from which open, high, low and close are formed exactly as an exchange
    would. Anything the estimator returns above zero on that input is bias.

    W1-P5 ran this, found a large positive number, and concluded that EDGE
    carries a finite-sample bias scaling with volatility. That was wrong in its
    cause: the number it read was :attr:`ZeroSpreadBias.absolute_mean`, and the
    absolute value is taken by ``bidask`` before the caller ever sees it.
    :attr:`ZeroSpreadBias.signed_mean` is the estimator's real bias and is
    roughly zero. See experiments.md rows 54-58.

    ``overnight_gap_share`` is the fraction of daily variance realised as a jump
    between one session's close and the next session's open, rather than inside
    the session. It defaults to ``0.0``, which is the continuous walk rows 48-51
    used and the process the committed figure's band is drawn from -- kept at
    zero so the reference stays comparable across the correction. It is exposed
    because a nonzero gap is the named suspect for the level that survives the
    sign fix on the most liquid names: EDGE's price process has no overnight
    jump, and a real one is read partly as spread. Row 58 measures it.

    **Read the result as an order of magnitude, not a correction.** It depends on
    the assumed intraday process -- measured across 78 to 6,000 ticks per day the
    answer moves by a factor of ~3.6 -- so it is drawn on the figure as an
    indicative band and is never subtracted from an estimate. Subtracting it
    would import this data-generating process into the production series, which
    is a far stronger assumption than the one it is trying to qualify.

    ``seed`` is passed explicitly to ``np.random.default_rng`` (CLAUDE.md, code
    conventions); there is no global seeding anywhere in this project.
    """
    if daily_volatility < 0:
        raise SpreadError(f"daily_volatility must be non-negative, got {daily_volatility}")
    if not 0.0 <= overnight_gap_share < 1.0:
        raise SpreadError(
            f"overnight_gap_share is a share of daily variance and must be in [0, 1), got "
            f"{overnight_gap_share}"
        )
    if window < 3 or days <= window or ticks_per_day < 2:
        raise SpreadError(
            f"zero_spread_reference: need window >= 3, days > window and ticks_per_day >= 2; "
            f"got window={window}, days={days}, ticks_per_day={ticks_per_day}"
        )

    rng = np.random.default_rng(seed)
    frame = _simulate_bars(
        rng,
        days=days,
        ticks_per_day=ticks_per_day,
        daily_volatility=daily_volatility,
        overnight_gap_share=overnight_gap_share,
        true_spread=0.0,
    )
    estimates = pd.Series(
        edge_rolling(df=frame, window=window, min_periods=window, sign=True)
    ).dropna()
    clipped = clip_negative(estimates)
    assert isinstance(clipped, pd.Series)
    return ZeroSpreadBias(
        signed_mean=float(estimates.mean()),
        absolute_mean=float(estimates.abs().mean()),
        clipped_mean=float(clipped.mean()),
        negative_fraction=float((estimates < 0).mean()),
    )


@dataclass(frozen=True)
class SpreadRecovery:
    """What EDGE recovers at one window length when the TRUE spread is known.

    The null was tested first (:class:`ZeroSpreadBias`) and the estimator passed
    it. That says nothing about whether it can RESOLVE a spread, which is a
    different question and the one that decides whether this instrument can
    deliver a level for a liquid ETF.

    Estimates come from INDEPENDENT windows, never overlapping ones, so
    :attr:`standard_error` is honest. A rolling series shares 62 days in 63
    between neighbours and its apparent precision is mostly that overlap.

    Read :attr:`standard_deviation` against ``true_spread``, not
    :attr:`standard_error`. The standard error says how well the LONG-RUN mean
    is pinned down and shrinks with more windows; the standard deviation is what
    a single window actually delivers, and it does not shrink at all.
    """

    true_spread: float
    mean: float
    standard_error: float
    median: float
    standard_deviation: float
    negative_fraction: float
    windows: int

    @property
    def recovers_in_mean(self) -> bool:
        """Is the true spread inside a 95% interval for the mean estimate?"""
        return abs(self.mean - self.true_spread) <= 1.96 * self.standard_error

    @property
    def signal_to_noise(self) -> float:
        """True spread over the per-window standard deviation.

        Below 1 a single window cannot distinguish this spread from zero, whatever
        the estimator's bias. This is the number that decides whether a LEVEL can
        be read off one window.
        """
        return self.true_spread / self.standard_deviation if self.standard_deviation else np.inf


def recovery_reference(
    *,
    true_spread: float,
    daily_volatility: float,
    window: int,
    seed: int,
    windows: int = 500,
    ticks_per_day: int = 390,
    overnight_gap_share: float = 0.0,
    price: float = 100.0,
) -> SpreadRecovery:
    """What EDGE returns at ``window`` days when the true spread is ``true_spread``.

    The recovery counterpart to :func:`zero_spread_reference`. Ardia, Guidotti &
    Kroencke (2024) prove the estimator ASYMPTOTICALLY unbiased and variance-
    minimising among OHLC estimators; neither property promises that a single
    short window pins down a small spread, and the authors' own FAQ says as much
    -- "using a few daily prices would provide estimates closer to the spread in
    those days but with potentially large estimation uncertainty". This function
    measures that uncertainty at the window this project actually runs.

    Each of ``windows`` estimates is computed on its OWN freshly simulated
    ``window`` days, so they are independent by construction. See
    experiments.md rows 59-61.
    """
    if true_spread < 0:
        raise SpreadError(f"true_spread must be non-negative, got {true_spread}")
    if window < 3:
        raise SpreadError(f"recovery_reference: window must be at least 3, got {window}")
    if windows < 2:
        raise SpreadError(f"recovery_reference: need at least 2 windows, got {windows}")

    rng = np.random.default_rng(seed)
    estimates = np.array(
        [
            edge(
                *(frame[column].to_numpy() for column in ("open", "high", "low", "close")),
                sign=True,
            )
            for frame in (
                _simulate_bars(
                    rng,
                    days=window,
                    ticks_per_day=ticks_per_day,
                    daily_volatility=daily_volatility,
                    overnight_gap_share=overnight_gap_share,
                    true_spread=true_spread,
                    price=price,
                )
                for _ in range(windows)
            )
        ]
    )
    estimates = estimates[~np.isnan(estimates)]
    if estimates.size < 2:
        raise SpreadError("recovery_reference: the simulation produced no usable estimates")

    deviation = float(estimates.std(ddof=1))
    return SpreadRecovery(
        true_spread=true_spread,
        mean=float(estimates.mean()),
        standard_error=deviation / float(np.sqrt(estimates.size)),
        median=float(np.median(estimates)),
        standard_deviation=deviation,
        negative_fraction=float((estimates < 0).mean()),
        windows=int(estimates.size),
    )


@dataclass(frozen=True)
class NegativeEstimates:
    """How often EDGE returned a negative spread, which is not a spread.

    EDGE is a moment estimator, so in a finite window it can land below zero even
    though the quantity it estimates cannot be. The paper's own claim is that
    this is rare relative to Roll and Corwin-Schultz, not that it is impossible.

    The incidence is a diagnostic in its own right and this project reports it
    rather than only clipping. Read it as a signal-to-noise ratio: a fraction
    near one half says the estimator is centred on zero and resolving nothing,
    a fraction near zero on a liquid instrument says an absolute value has been
    taken somewhere upstream, and a small nonzero fraction on a wide instrument
    is the estimator behaving as published. It must always be computed on the
    SIGNED series, before :func:`clip_negative` -- counting negatives after the
    clip returns zero by construction, which is precisely the reading that let
    the W1-P5 misdiagnosis stand (experiments.md rows 53-54).
    """

    total: int
    negative: int
    most_negative: float | None

    @property
    def fraction(self) -> float:
        return self.negative / self.total if self.total else 0.0

    def render(self) -> str:
        if not self.total:
            return "no estimates"
        return f"{self.negative}/{self.total} ({100 * self.fraction:.2f}%) negative" + (
            f", most negative {self.most_negative:.6f}" if self.negative else ""
        )


def count_negative(spread: pd.Series) -> NegativeEstimates:
    """Incidence of negative EDGE estimates in a spread series."""
    values = spread.dropna()
    negative = values[values < 0]
    return NegativeEstimates(
        total=int(values.size),
        negative=int(negative.size),
        most_negative=float(negative.min()) if negative.size else None,
    )
