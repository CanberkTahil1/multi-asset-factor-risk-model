"""SPEC.md 15.4.1's panel on the REAL cache: data-contract tests (``dataset``-marked).

Deselected by ``make test`` and the fast CI workflow; run weekly after ``make
data``. Each test skips, naming what is missing, when the EDGAR artefacts are
not cached. Row 272's expectations are read from ``config/model.yaml``, where
they were registered before the pull, not typed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mafrm.config import load
from mafrm.data import cache, edgar, market_cap


@pytest.fixture(scope="module")
def panel() -> market_cap.MarketCapPanel:
    manifest = cache.Manifest.load()
    for name in (edgar.FRAMES_NAME, edgar.INDEX_NAME, edgar.COMPANY_TICKERS_NAME):
        try:
            manifest.latest(source=edgar.SOURCE, name=name)
        except cache.CacheError as exc:
            pytest.skip(f"no cached {edgar.SOURCE}/{name}: {exc}")
    return market_cap.build(load(), manifest=manifest)


@pytest.mark.dataset
def test_the_panel_stops_before_the_holdout(panel: market_cap.MarketCapPanel) -> None:
    """CLAUDE.md invariant 5: nothing filed on or after the boundary is in force in-sample."""
    boundary = pd.Timestamp(load().require_holdout_start())
    assert panel.sessions.max() < boundary
    for table in panel.counts.values():
        if len(table):
            assert pd.DatetimeIndex(table.index).max() < boundary


@pytest.mark.dataset
def test_every_ticker_has_exactly_one_status(panel: market_cap.MarketCapPanel) -> None:
    """The mapping partitions the universe's tickers: ok + no_facts + unmapped + ambiguous."""
    status = panel.mapping["status"]
    counts = status.value_counts()
    assert set(counts.index) <= {"ok", *market_cap.NO_CAP_REASONS}
    assert int(counts.sum()) == len(panel.mapping)
    # A cap needs a count AND a close: every ticker with a cap has status ok, and an
    # ok ticker without a cap is one the cache holds no in-sample close for.
    has_cap = panel.cap.notna().any(axis=0)
    ok = set(status[status == "ok"].index)
    assert set(has_cap[has_cap].index) <= ok
    for ticker in ok - set(has_cap[has_cap].index):
        assert panel.shares[ticker].notna().any(), ticker
        assert panel.cap[ticker].isna().all(), ticker


@pytest.mark.dataset
def test_row_272_the_point_in_time_join_on_apple(panel: market_cap.MarketCapPanel) -> None:
    """experiments.md row 272, legs (a) and (b), as registered in config."""
    r = load().model.equity_shares.registrations.row_272
    assert int(panel.mapping.loc[r.ticker, "cik"]) == r.cik
    table = panel.counts[r.ticker]
    in_force = table[table.index <= pd.Timestamp(r.in_force_on)].iloc[-1]
    assert in_force.name == pd.Timestamp(r.expected_filed)
    assert float(in_force["shares"]) == float(r.expected_shares)
    ex = pd.Timestamp(r.split_ex_date)
    on_split = table[table.index <= ex].iloc[-1]
    assert on_split.name == pd.Timestamp(r.pre_split_filed)
    assert float(on_split["shares"]) == float(r.split_ratio * r.pre_split_shares)
    # (c) no jump at the split on the back-adjusted basis
    sessions = panel.sessions
    before = sessions[sessions < ex][-1]
    assert panel.shares.loc[before, r.ticker] == panel.shares.loc[ex, r.ticker]


# ---------------------------------------------------------------------------
# SPEC.md 15.4.3 (W7-P2): the three cap rulings on the real cache
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def universe_ever(panel: market_cap.MarketCapPanel) -> set[str]:
    """Names ever in the estimation universe (SPEC.md 15.3's screen), for the band test."""
    from mafrm.factors import equity_universe

    cfg = load()
    close, volume = equity_universe.load_bars(cfg)
    membership = equity_universe.load_membership(cfg)
    u = cfg.model.equity_universe
    screen = equity_universe.screen(
        close,
        volume,
        membership,
        min_history_days=u.min_history_days,
        min_dollar_adv=u.min_dollar_adv,
        start=cfg.model.sample.start,
        end=cfg.require_holdout_start(),
        strict=False,
    )
    return {str(t) for t in screen.universe.columns[screen.universe.any(axis=0)]}


@pytest.mark.dataset
def test_the_panel_is_a_model_input_only_because_the_screen_was_applied(
    panel: market_cap.MarketCapPanel,
) -> None:
    cfg = load()
    assert panel.is_model_input
    assert panel.consistency_ratio == cfg.model.equity_shares.consistency_screen.ratio
    # every drop is logged with the median it was compared against
    assert set(panel.dropped.columns) >= {"ticker", "filed", "shares", "median", "ratio"}
    ratio = panel.dropped["ratio"].astype(float)
    assert ((ratio > panel.consistency_ratio) | (ratio < 1.0 / panel.consistency_ratio)).all()
    # and no surviving count is that far off its name's median
    for ticker, table in panel.counts.items():
        if len(table) > 1:
            rel = table["shares"].to_numpy(dtype=float) / float(table["shares"].median())
            assert (rel <= panel.consistency_ratio).all() and (
                rel >= 1.0 / panel.consistency_ratio
            ).all(), ticker


@pytest.mark.dataset
def test_row_274_at_most_the_registered_number_of_universe_filings_drop(
    panel: market_cap.MarketCapPanel, universe_ever: set[str]
) -> None:
    """experiments.md row 274(b): the screen removes unit-scale errors and shells, nothing else."""
    reg = load().model.equity_shares.registrations
    dropped = panel.dropped[panel.dropped["ticker"].isin(universe_ever)]
    assert len(dropped) <= reg.row_274_max_dropped_universe_filings, len(dropped)


def _unscreened_universe_counts(universe_ever: set[str]) -> dict[str, pd.DataFrame]:
    """Every filer's usable counts among universe names, UNSCREENED and without the hand table."""
    cfg = load()
    manifest = cache.Manifest.load()
    from mafrm.data import sp500, sp500_reference

    frames = cache.read(
        manifest.latest(source=edgar.SOURCE, name=edgar.FRAMES_NAME), manifest=manifest
    )
    index = cache.read(
        manifest.latest(source=edgar.SOURCE, name=edgar.INDEX_NAME), manifest=manifest
    )
    close = market_cap.load_close(manifest)
    sessions = pd.DatetimeIndex(close.index)
    sessions = sessions[
        (sessions >= pd.Timestamp(cfg.model.sample.start))
        & (sessions < pd.Timestamp(cfg.require_holdout_start()))
    ]
    unscreened = market_cap.assemble(
        frames=frames,
        index=index,
        company_tickers=market_cap.load_company_tickers(manifest),
        close=close.reindex(sessions),
        splits=market_cap.load_splits(manifest),
        intervals=sp500.load_intervals(
            sp500_reference.reference_dir(cfg) / cfg.model.equity_universe.intervals_file
        ),
        sessions=sessions,
    )
    return {t: tb for t, tb in unscreened.counts.items() if t in universe_ever and len(tb)}


def _folded_median_ratios(counts: dict[str, pd.DataFrame]) -> pd.Series:
    """count / the name's median, folded to >= 1: the quantity the screen acts on."""
    parts: list[pd.Series] = []
    for ticker, table in counts.items():
        shares = table["shares"].to_numpy(dtype=float)
        rel = shares / float(np.median(shares))
        parts.append(pd.Series(np.maximum(rel, 1.0 / rel), index=[ticker] * len(rel)))
    return pd.concat(parts)


@pytest.mark.dataset
def test_r_sits_inside_the_measured_empty_band(universe_ever: set[str]) -> None:
    """SPEC.md 15.4.3 ruling (i), as rewritten in W7-P2b ruling 1: a SEPARATION test with R fixed.

    On the median-relative ratios the screen acts on -- every usable filing
    among universe names, unscreened -- the empty band around R is (largest
    ratio below R, smallest ratio above R). The 2026-09-05 pull measured it as
    (54.7, 489), recorded in config as ``consistency_screen.empty_band``. This
    test recomputes it, prints its ends, and asserts (a) R lies strictly inside
    it and (b) it still contains the recorded one: a future filing landing
    between the recorded ends fails here and RE-OPENS R with the numbers.
    Nothing here tunes R.
    """
    cfg = load()
    screen = cfg.model.equity_shares.consistency_screen
    ratios = _folded_median_ratios(_unscreened_universe_counts(universe_ever))
    below = ratios[ratios < screen.ratio]
    above = ratios[ratios > screen.ratio]
    assert len(below) and len(above), "no filings on one side of R -- nothing to separate"
    low, high = float(below.max()), float(above.min())
    print(
        f"\nmeasured empty band around R = {screen.ratio:g}: ({low:.3f}, {high:.3f}); "
        f"nearest below {below.idxmax()}, nearest above {above.idxmin()}; "
        f"{len(ratios):,} filings over {len(set(ratios.index))} names"
    )
    assert not (ratios == screen.ratio).any(), "a filing sits exactly at R"
    assert low < screen.ratio < high
    assert low <= screen.empty_band[0], (
        f"a filing at {low:.3f}x ({below.idxmax()}) has entered the recorded empty band "
        f"({screen.empty_band[0]:g}, {screen.empty_band[1]:g}) from below; R is re-opened"
    )
    assert high >= screen.empty_band[1], (
        f"a filing at {high:.3f}x ({above.idxmin()}) has entered the recorded empty band "
        f"({screen.empty_band[0]:g}, {screen.empty_band[1]:g}) from above; R is re-opened"
    )


@pytest.mark.dataset
def test_no_dropped_count_is_a_corporate_action(
    panel: market_cap.MarketCapPanel, universe_ever: set[str]
) -> None:
    """W7-P2b ruling 1(b): the screen never drops a count the actions table explains.

    Every dropped filing's ratio to the name's median is compared with every
    split ratio the vendor's actions table holds for the ticker, with the
    running products of consecutive splits, and with their reciprocals; a
    match within 1% would mean the screen removed a corporate action rather
    than a unit-scale error or a shell.
    """
    splits = market_cap.load_splits(cache.Manifest.load())
    dropped = panel.dropped[panel.dropped["ticker"].isin(universe_ever)]
    assert len(dropped), "the screen dropped nothing among universe names"
    offenders: list[str] = []
    for row in dropped.itertuples(index=False):
        ratio = float(row.ratio)
        folded = max(ratio, 1.0 / ratio)
        series = splits.get(str(row.ticker))
        if series is None or len(series) == 0:
            continue
        values = series.to_numpy(dtype=float)
        candidates: list[float] = list(values)
        for i in range(len(values)):
            running = 1.0
            for j in range(i, len(values)):
                running *= values[j]
                candidates.append(running)
        for c in candidates:
            if c <= 0:
                continue
            if abs(np.log(folded) - abs(np.log(c))) < np.log(1.01):
                offenders.append(
                    f"{row.ticker} {pd.Timestamp(str(row.filed)).date()} {ratio:.4g} ~ {c:g}"
                )
    assert not offenders, "dropped counts explained by the actions table:\n" + "\n".join(offenders)


@pytest.mark.dataset
def test_row_276_the_predecessor_route_on_xom(panel: market_cap.MarketCapPanel) -> None:
    """experiments.md row 276, as registered in config."""
    r = load().model.equity_shares.registrations.row_276
    m = panel.mapping.loc[r.ticker]
    assert m["status"] == "ok" and int(m["predecessor_cik"]) == r.predecessor_cik
    assert panel.overrides == {r.ticker: r.predecessor_cik}
    table = panel.counts[r.ticker]
    in_force = table[table.index <= pd.Timestamp(r.in_force_on)].iloc[-1]
    assert in_force.name == pd.Timestamp(r.expected_filed)
    # The config carries the cover page's figure (W7-P2b ruling 4); the registration's
    # transcription error is recorded in experiments.md row 276.
    assert float(in_force["shares"]) == float(r.expected_shares)


@pytest.mark.dataset
def test_row_274_screened_total_sits_below_the_published_us_total(
    panel: market_cap.MarketCapPanel, universe_ever: set[str]
) -> None:
    """experiments.md row 274(a). The universe is a subset of the US market."""
    from mafrm import config as config_mod
    from mafrm.factors import equity_universe

    cfg = load()
    published = market_cap.load_published_totals(
        config_mod.config_dir() / cfg.model.equity_shares.published_totals_file
    )
    close, volume = equity_universe.load_bars(cfg)
    membership = equity_universe.load_membership(cfg)
    u = cfg.model.equity_universe
    universe = equity_universe.screen(
        close,
        volume,
        membership,
        min_history_days=u.min_history_days,
        min_dollar_adv=u.min_dollar_adv,
        start=cfg.model.sample.start,
        end=cfg.require_holdout_start(),
        strict=False,
    ).universe
    sessions = panel.sessions
    in_universe = universe.reindex(index=sessions, columns=panel.cap.columns).fillna(False)
    for day, value in published.values_usd.items():
        year_end = sessions[sessions.year == pd.Timestamp(str(day)).year]
        if len(year_end) == 0:
            continue
        total = float(panel.cap.where(in_universe.astype(bool)).loc[year_end[-1]].sum())
        assert total < float(value), (year_end[-1].date(), total, value)


@pytest.mark.dataset
def test_brk_b_is_no_facts_by_the_hand_table(panel: market_cap.MarketCapPanel) -> None:
    """W7-P2b ruling 3: the Class-A fact never meets the Class-B price."""
    assert panel.excluded == {"BRK-B": market_cap.TREAT_AS_NO_FACTS}
    m = panel.mapping.loc["BRK-B"]
    assert m["status"] == "no_facts" and m["treat_as"] == market_cap.TREAT_AS_NO_FACTS
    assert int(m["facts"]) > 0 and int(m["used"]) == 0
    assert panel.cap["BRK-B"].isna().all()
