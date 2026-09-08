"""The equity estimation-universe screen. SPEC.md 15.3, rulings at 15.3.1.

USE4's estimation universe excludes names without enough history and names
too thin to trade. This module applies the one screen SPEC.md 15.3 gives a
number for -- ``min_history_days`` valid daily bars for the ticker in the
trailing window of the same length, history rather than tenure in the index
(ruling 2) -- and carries the dollar-ADV screen behind a config value that is
``null`` until a published threshold exists (ruling 1): while null it drops
nothing and the report says so.

Every count is stated per session and by cause, because the causes are
different claims (ruling 5):

* ``members``              -- Wikipedia says the name was in the index;
* ``member_no_data``       -- and the cache holds no valid bar for it on that
                              session (the survivorship control's measured
                              coverage, and the committed matrix's state 2);
* ``insufficient_history`` -- it has a bar, and fewer than ``min_history_days``
                              of the trailing ``min_history_days`` sessions
                              carry a valid bar;
* ``below_dollar_adv``     -- it has the history and its trailing median dollar
                              volume is under the threshold (zero while null);
* ``estimation_universe``  -- what is left.

The screen reads the COMMITTED daily membership matrix and the cached bars,
and asserts that the matrix's third state agrees with the bars on every
in-sample session -- a committed file that drifted from the cache it was
built from is a provenance failure, not something to screen around.

Asset-class agnostic in the sense that matters for CLAUDE.md invariant 10: the
screen is a function of bars and a membership matrix, and nothing in
``risk/`` reads it. Reads are model-facing: the bars go through
:func:`mafrm.data.cache.read` and the matrix is cut at the holdout boundary
on load, so nothing here can see the holdout (invariant 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from mafrm import config as config_mod
from mafrm.data import cache, loaders, sp500, sp500_reference

__all__ = ["COUNT_COLUMNS", "Screen", "load_bars", "load_membership", "screen"]

#: Column order of :attr:`Screen.counts`.
COUNT_COLUMNS = (
    "members",
    "member_no_data",
    "insufficient_history",
    "below_dollar_adv",
    "estimation_universe",
)


@dataclass(frozen=True)
class Screen:
    """The screen on every session of the window."""

    #: Per-session counts, one column per cause, in :data:`COUNT_COLUMNS` order.
    counts: pd.DataFrame
    #: Boolean session x ticker: passes the screen.
    universe: pd.DataFrame
    #: Boolean session x ticker, the two causes the report names names for.
    member_no_data: pd.DataFrame
    insufficient_history: pd.DataFrame
    min_history_days: int
    min_dollar_adv: float | None
    #: Cells where the committed matrix's state 2 disagrees with the bars. Zero
    #: under ``strict``; reported when the bars are a later pull than the
    #: snapshot's (one-snapshot rule, W7-P1b).
    state_disagreements: int = 0

    def __post_init__(self) -> None:
        total = (
            self.counts["member_no_data"]
            + self.counts["insufficient_history"]
            + self.counts["below_dollar_adv"]
            + self.counts["estimation_universe"]
        )
        if not total.equals(self.counts["members"]):
            raise ValueError("screen counts do not partition the member count")


def load_bars(
    cfg: config_mod.Config | None = None, *, entry: cache.ManifestEntry | str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide ``close`` and ``volume`` (sessions x tickers), truncated at the holdout.

    ``entry`` pins a specific cached pull -- the snapshot's recorded prices
    input -- instead of the latest.
    """
    cfg = cfg or config_mod.load()
    manifest = cache.Manifest.load()
    if entry is None:
        entry = manifest.latest(source="yfinance", name=loaders.SP500_PRICES_NAME)
    long = cache.read(entry, manifest=manifest)
    if not isinstance(long.index, pd.DatetimeIndex):
        raise ValueError("sp500_prices: expected a date index")
    close = long.pivot(columns="ticker", values="Close").sort_index()
    volume = long.pivot(columns="ticker", values="Volume").sort_index()
    close.columns = pd.Index([str(c) for c in close.columns], name="ticker")
    volume.columns = pd.Index([str(c) for c in volume.columns], name="ticker")
    return close, volume


def load_membership(cfg: config_mod.Config | None = None) -> pd.DataFrame:
    """The committed daily matrix, cut at the holdout boundary (CLAUDE.md invariant 5)."""
    cfg = cfg or config_mod.load()
    universe = cfg.model.equity_universe
    path = sp500_reference.reference_dir(cfg) / universe.membership_file
    return sp500.load_membership(path, end=cfg.require_holdout_start())


def screen(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    min_history_days: int,
    min_dollar_adv: float | None,
    start: date | pd.Timestamp,
    end: date | pd.Timestamp,
    strict: bool = True,
) -> Screen:
    """Apply the screen on every session in ``[start, end)``.

    ``membership`` is the daily three-state matrix (:mod:`mafrm.data.sp500`).
    The trading calendar is the union of sessions in ``close``; every session
    in the window must be a row of ``membership``, and the matrix's state 2
    must equal "member with no valid bar" computed from these bars. A bar is
    *valid* when close and volume are both present and positive
    (:func:`mafrm.data.sp500_reference.valid_bars`). ``insufficient_history``
    is a member with a valid bar whose trailing ``min_history_days`` sessions
    hold fewer than ``min_history_days`` valid bars. The dollar-ADV screen is
    the trailing median of ``close x volume`` over valid bars in the same
    window, applied only when a threshold is set.

    ``strict`` makes a disagreement between the committed third state and the
    bars a refusal; the report passes ``False`` only when the bars are a later
    pull than the snapshot's, and then the count is carried on the result.
    """
    if min_history_days <= 0:
        raise ValueError("min_history_days must be positive")
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError("close must be indexed by date")
    tickers = sorted(set(map(str, close.columns)) | set(map(str, membership.columns)))
    close = close.reindex(columns=tickers)
    volume = volume.reindex(index=close.index, columns=tickers)
    valid = sp500_reference.valid_bars(close, volume)

    sessions = pd.DatetimeIndex(close.index)
    window = sessions[(sessions >= pd.Timestamp(start)) & (sessions < pd.Timestamp(end))]
    if len(window) == 0:
        raise ValueError("screen: no session inside the window")
    missing = window.difference(pd.DatetimeIndex(membership.index))
    if len(missing) > 0:
        raise ValueError(
            f"screen: {len(missing)} session(s) in the window are not rows of the membership "
            f"matrix, first {missing[0].date()}; rebuild data/reference (make data)"
        )
    states = membership.reindex(index=window, columns=tickers).fillna(sp500.NOT_MEMBER)
    member = states != sp500.NOT_MEMBER
    committed_no_data = states == sp500.MEMBER_NO_DATA
    present = valid.reindex(index=window).fillna(False).astype(bool)
    with_data = member & present
    no_data = member & ~present
    disagreements = int((committed_no_data != no_data).to_numpy().sum())
    if disagreements and strict:
        raise ValueError(
            f"screen: the committed matrix's 'member, no data' state disagrees with the cached "
            f"bars in {disagreements} cell(s); the reference file has drifted from the cache "
            "it was built from -- rebuild data/reference (make data) and commit the change"
        )

    history_ok = (
        valid.rolling(min_history_days, min_periods=min_history_days).sum().reindex(window)
        >= min_history_days
    )
    if min_dollar_adv is None:
        adv_ok = pd.DataFrame(True, index=window, columns=pd.Index(tickers, name="ticker"))
    else:
        dollar = (close * volume).where(valid)
        median = dollar.rolling(min_history_days, min_periods=min_history_days).median()
        adv_ok = median.reindex(window) >= min_dollar_adv

    insufficient = with_data & ~history_ok.fillna(False)
    eligible = with_data & history_ok.fillna(False)
    below_adv = eligible & ~adv_ok.fillna(False)
    universe = eligible & adv_ok.fillna(False)

    counts = pd.DataFrame(
        {
            "members": member.sum(axis=1),
            "member_no_data": no_data.sum(axis=1),
            "insufficient_history": insufficient.sum(axis=1),
            "below_dollar_adv": below_adv.sum(axis=1),
            "estimation_universe": universe.sum(axis=1),
        },
        index=window,
    ).astype(int)
    counts.index.name = "date"
    return Screen(
        counts=counts[list(COUNT_COLUMNS)],
        universe=universe,
        member_no_data=no_data,
        insufficient_history=insufficient,
        min_history_days=min_history_days,
        min_dollar_adv=min_dollar_adv,
        state_disagreements=disagreements,
    )
