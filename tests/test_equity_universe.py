"""The estimation-universe screen (SPEC.md 15.3) on a synthetic panel, by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mafrm.config import ConfigError, _parse_model, load
from mafrm.data import sp500
from mafrm.factors import equity_universe


def _panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """400 business days; X full history, Y bars from session 300, Z none, W leaves.

    Returns ``(close, volume, membership)`` where the membership matrix is built
    the way the committed file is: from intervals and the same validity rule.
    """
    sessions = pd.bdate_range("2019-01-01", periods=400, name="date")
    close = pd.DataFrame(np.nan, index=sessions, columns=pd.Index(["W", "X", "Y"], name="ticker"))
    volume = close.copy()
    close["X"] = 10.0
    volume["X"] = 1_000.0  # dollar volume 10,000 a day
    close.iloc[300:, close.columns.get_loc("Y")] = 20.0
    volume.iloc[300:, volume.columns.get_loc("Y")] = 500.0
    close["W"] = 5.0
    volume["W"] = 100.0
    intervals = pd.DataFrame(
        {
            "ticker": ["W", "X", "Y", "Z"],
            "start": [pd.NaT, pd.NaT, sessions[300], pd.NaT],
            "end": [sessions[350], pd.NaT, pd.NaT, pd.NaT],
            "start_known": [False, False, True, False],
            "security": ["W", "X", "Y", "Z"],
        }
    )
    member = sp500.member_on(intervals, sessions)
    valid = (
        (close.notna() & (close > 0) & volume.notna() & (volume > 0))
        .reindex(columns=member.columns)
        .fillna(False)
        .astype(bool)
    )
    membership = pd.DataFrame(
        sp500.NOT_MEMBER, index=sessions, columns=member.columns, dtype="int8"
    )
    membership[member & valid] = sp500.MEMBER
    membership[member & ~valid] = sp500.MEMBER_NO_DATA
    return close, volume, membership


def _screen(min_dollar_adv: float | None = None, **overrides: object) -> equity_universe.Screen:
    close, volume, membership = _panel()
    kwargs: dict[str, object] = dict(
        min_history_days=252,
        min_dollar_adv=min_dollar_adv,
        start=close.index[0],
        end=close.index[-1] + pd.Timedelta(days=1),
    )
    kwargs.update(overrides)
    return equity_universe.screen(close, volume, membership, **kwargs)  # type: ignore[arg-type]


def test_counts_partition_members_by_cause_on_every_session() -> None:
    result = _screen()
    assert len(result.counts) == 400
    last = result.counts.iloc[-1]
    # Last session: W left at session 350, X has 400 valid bars, Y has 100,
    # Z has never had a bar.
    assert last["members"] == 3
    assert last["member_no_data"] == 1  # Z
    assert last["insufficient_history"] == 1  # Y
    assert last["below_dollar_adv"] == 0
    assert last["estimation_universe"] == 1  # X
    # First session: X and W have 1 valid bar of 252 needed; Y is not yet a
    # member; Z no data.
    first = result.counts.iloc[0]
    assert first["members"] == 3 and first["insufficient_history"] == 2
    assert first["member_no_data"] == 1 and first["estimation_universe"] == 0
    assert bool(result.universe.iloc[-1]["X"]) and not bool(result.universe.iloc[-1]["Y"])
    # W is a member through session 349 and not on 350 (end exclusive).
    assert result.counts.iloc[349]["members"] == 4
    assert result.counts.iloc[350]["members"] == 3


def test_history_screen_turns_on_exactly_at_the_window_length() -> None:
    result = _screen()
    universe = result.universe["X"]
    # Session 251 (0-based) is the first with 252 valid bars behind it.
    threshold = result.counts.index[251]
    assert not universe[universe.index < threshold].any()
    assert universe[universe.index >= threshold].all()


def test_dollar_adv_screen_applies_only_when_a_threshold_is_set() -> None:
    unset = _screen(None)
    assert unset.counts["below_dollar_adv"].sum() == 0
    # X's median dollar volume is exactly 10,000; a threshold just above it drops X.
    above = _screen(10_000.5)
    assert above.counts.iloc[-1]["below_dollar_adv"] == 1
    assert above.counts.iloc[-1]["estimation_universe"] == 0
    at = _screen(10_000.0)
    assert at.counts.iloc[-1]["estimation_universe"] == 1


def test_window_bounds_are_half_open() -> None:
    close, _, _ = _panel()
    result = _screen(end=close.index[-1])
    assert result.counts.index.max() < close.index[-1]
    assert len(result.counts) == 399


def test_a_matrix_that_drifted_from_the_bars_is_refused() -> None:
    close, volume, membership = _panel()
    drifted = membership.copy()
    drifted.iloc[10, drifted.columns.get_loc("X")] = sp500.MEMBER_NO_DATA  # X has a bar there
    with pytest.raises(ValueError, match="drifted from the cache"):
        equity_universe.screen(
            close,
            volume,
            drifted,
            min_history_days=252,
            min_dollar_adv=None,
            start=close.index[0],
            end=close.index[-1],
        )


def test_a_session_missing_from_the_matrix_is_refused() -> None:
    close, volume, membership = _panel()
    with pytest.raises(ValueError, match="not rows of the membership matrix"):
        equity_universe.screen(
            close,
            volume,
            membership.iloc[5:],
            min_history_days=252,
            min_dollar_adv=None,
            start=close.index[0],
            end=close.index[-1],
        )


def test_config_null_adv_is_the_absence_of_a_threshold() -> None:
    cfg = load().model.equity_universe
    assert cfg.min_dollar_adv is None
    assert cfg.min_history_days == 252
    assert cfg.membership_count_band == (480, 520)


def test_config_refuses_a_nonpositive_adv_threshold_or_an_inverted_band() -> None:
    import yaml

    from mafrm.config import _REPO_ROOT

    document = yaml.safe_load((_REPO_ROOT / "config" / "model.yaml").read_text())
    bad_adv = yaml.safe_load(yaml.safe_dump(document))
    bad_adv["equity_universe"]["min_dollar_adv"] = -1.0
    with pytest.raises(ConfigError, match="min_dollar_adv"):
        _parse_model(bad_adv)
    bad_band = yaml.safe_load(yaml.safe_dump(document))
    bad_band["equity_universe"]["membership_count_band"] = [520, 480]
    with pytest.raises(ConfigError, match="membership_count_band"):
        _parse_model(bad_band)


def test_equity_shares_block_parses_and_validates() -> None:
    """W7-P2a: the EDGAR source block. URLs carry their placeholders; nothing is tuned."""
    import yaml

    from mafrm.config import _REPO_ROOT

    shares = load().model.equity_shares
    assert shares.tag == "EntityCommonStockSharesOutstanding" and shares.taxonomy == "dei"
    assert shares.frame_url("CY2019Q1I").endswith(
        "/dei/EntityCommonStockSharesOutstanding/shares/CY2019Q1I.json"
    )
    assert shares.quarter_index_url(2009, 3).endswith("/2009/QTR3/xbrl.idx")
    assert shares.user_agent_env.isidentifier()
    document = yaml.safe_load((_REPO_ROOT / "config" / "model.yaml").read_text())
    bad = yaml.safe_load(yaml.safe_dump(document))
    bad["equity_shares"]["first_period"] = "2009-Q2"
    with pytest.raises(ConfigError, match="first_period"):
        _parse_model(bad)
    bad = yaml.safe_load(yaml.safe_dump(document))
    bad["equity_shares"]["point_in_time"] = "known_from_period_end"
    with pytest.raises(ConfigError, match="point_in_time"):
        _parse_model(bad)
