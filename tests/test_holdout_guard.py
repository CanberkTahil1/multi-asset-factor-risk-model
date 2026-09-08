"""The holdout boundary as a mechanical control. CLAUDE.md invariant 5.

Written after the control it replaces failed. Through W2-P2 the invariant was
enforced by discipline plus downstream assertions: shipped panels passed
``Config.require_holdout_start()`` into their construction and tests asserted
those panels ended before the boundary. During W2-P3 an ad-hoc scoping read
computed a SPY/``Mkt-RF`` correlation over 2025-01..2026-06 -- and *no test on
the shipped path could have caught it*, because the read never touched the
shipped path. The cache read that produced it was an ordinary call returning
full history.

So the guard moved to the loader, and these are its tests. The one that matters
most is :func:`test_the_sanctioned_crossings_are_exactly_these` -- it pins the
set of call sites permitted to see past the boundary, so adding a new one is a
deliberate edit to this file rather than a line nobody reviews.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from mafrm.config import load
from mafrm.data import aqr, cache, french, gsw, holdout

_SRC = Path(__file__).resolve().parents[1] / "src"


def _dated(start: str, periods: int = 6) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="D", name="date")
    return pd.DataFrame({"value": range(periods)}, index=index)


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------


def test_truncate_is_half_open_on_the_boundary_date() -> None:
    """The holdout begins ON holdout_start, so that date is excluded.

    Matches `calendar.align`, which takes the same boundary as a half-open upper
    bound. A closed bound here would leak exactly one day, which is the kind of
    off-by-one an invariant like this cannot afford.
    """
    boundary = pd.Timestamp(load().require_holdout_start())
    frame = _dated("2024-12-30", periods=5)  # 12-30, 12-31, 01-01, 01-02, 01-03
    kept = holdout.truncate(frame, label="t")
    assert list(pd.DatetimeIndex(kept.index).date) == [
        pd.Timestamp("2024-12-30").date(),
        pd.Timestamp("2024-12-31").date(),
    ]
    assert boundary not in kept.index


def test_truncate_leaves_a_wholly_in_sample_frame_alone() -> None:
    frame = _dated("2020-01-01")
    pd.testing.assert_frame_equal(holdout.truncate(frame, label="t"), frame)


def test_truncate_refuses_an_undated_object_rather_than_passing_it_through() -> None:
    """A silent pass-through is the failure this module exists to prevent."""
    frame = pd.DataFrame({"value": [1, 2]}, index=["a", "b"])
    with pytest.raises(holdout.HoldoutError, match="not a DatetimeIndex"):
        holdout.truncate(frame, label="undated")


def test_truncate_works_on_a_series_as_well_as_a_frame() -> None:
    series = _dated("2024-12-30", periods=5)["value"]
    assert len(holdout.truncate(series, label="t")) == 2


def test_is_truncated_reports_the_property_it_names() -> None:
    assert holdout.is_truncated(_dated("2020-01-01"))
    assert not holdout.is_truncated(_dated("2026-01-01"))
    assert holdout.is_truncated(_dated("2020-01-01").iloc[:0])


# ---------------------------------------------------------------------------
# The accessors
# ---------------------------------------------------------------------------


def test_read_unrestricted_requires_a_reason_that_says_something() -> None:
    """The reason is the record of why a crossing is entitled. Blank is not one."""
    with pytest.raises(cache.CacheError, match="requires a stated reason"):
        cache.read_unrestricted("anything", reason="   ")


def test_the_two_accessors_are_named_so_the_default_is_the_safe_one() -> None:
    """`read` is short and truncates; seeing past the boundary is the long name.

    Stated as a test because it is the whole ergonomic argument: the guard works
    only if the obvious call is the safe call.
    """
    assert "read" in cache.__all__
    assert "read_unrestricted" in cache.__all__
    assert "holdout" in cache.read.__doc__.lower()


def test_the_sanctioned_crossings_are_exactly_these() -> None:
    """Pin every call site permitted to read past the boundary.

    This is the control with teeth. Each entry is a question about what a
    *source* published -- coverage, release cadence, or a vendor cross-check --
    never a question about model behaviour. Adding a crossing means editing this
    list, which is a reviewable act; without this test a new one is a single
    line nobody looks at twice.
    """
    permitted = {
        # Vendor cross-checks: do two independent constructions of the same
        # series agree over everything cached? (experiments.md rows 41-42, W1-P3)
        "mafrm/data/crosschecks.py",
        "mafrm/data/gsw_report.py",
        # The unrestricted accessors themselves, which exist to be called by the
        # above and by the data-contract tests.
        "mafrm/data/cache.py",
        "mafrm/data/french.py",
        "mafrm/data/gsw.py",
        # W7-P1, SPEC.md 15.3.1: the S&P 500 membership tables are a SOURCE
        # snapshot dated by column, not by index -- a current constituent list
        # is read after the boundary by construction -- and the committed daily
        # matrix's data state spans the snapshot. Both are questions about what
        # the source published. Model-facing reads of the same bars go through
        # `cache.read` (mafrm.factors.equity_universe) and the committed matrix
        # is cut at the boundary on load.
        "mafrm/data/loaders.py",
        "mafrm/data/sp500_reference.py",
        # W7-P2a, SPEC.md 15.4.1: two source questions. SEC's ticker -> CIK map
        # is an undated identifier table (cache.read refuses undated frames by
        # design), and the vendor's split back-adjustment basis is set by the
        # PULL date -- a split after the boundary rescales every cached close
        # before it -- so the actions table's Stock Splits column is read in
        # full to put a filed share count on the close's basis. The share
        # counts and filing dates themselves go through cache.read.
        "mafrm/data/market_cap.py",
        # W7-P3, SPEC.md 15.6.1: the FF49 definitions and the two per-CIK SIC
        # tables are undated identifier tables, like SEC's ticker map. The SIC
        # applied is the CURRENT one by ruling 2 -- an approximation applied
        # backwards by decision, measured by the drift table, not a read past
        # the boundary of anything the model forecasts.
        "mafrm/data/sic.py",
    }
    found = {
        str(path.relative_to(_SRC))
        for path in _SRC.rglob("*.py")
        if re.search(r"read_unrestricted\s*\(|load_\w+_unrestricted\s*\(", path.read_text())
    }
    assert found == permitted, (
        f"unsanctioned holdout crossing(s): {sorted(found - permitted)}. "
        "A read past the boundary needs a stated reason and an entry here."
    )


def test_no_module_on_the_model_path_crosses_the_boundary() -> None:
    """factors/, risk/, costs/ and backtest/ have no business reading past it."""
    for package in ("factors", "risk", "costs", "backtest"):
        for path in (_SRC / "mafrm" / package).rglob("*.py"):
            assert "read_unrestricted" not in path.read_text(), path


# ---------------------------------------------------------------------------
# The regression test for the actual breach
# ---------------------------------------------------------------------------


@pytest.mark.dataset
def test_every_model_facing_loader_stops_at_the_boundary() -> None:
    """The exact reads the W2-P3 scoping breach was built from, now truncated.

    Not a re-statement of the panel assertions in `test_factor_validation`: those
    check that a *shipped* artefact ends in time. This checks that the raw cache
    read a scratch script would reach for cannot hand back a post-boundary
    observation in the first place.
    """
    boundary = pd.Timestamp(load().require_holdout_start())
    manifest = cache.Manifest.load()
    reads = {
        "spy_prices": cache.read(
            manifest.latest(source="yfinance", name="spy_prices"), manifest=manifest
        ),
        "spy_actions": cache.read(
            manifest.latest(source="yfinance", name="spy_actions"), manifest=manifest
        ),
        "ken_french": french.load_daily_factors(),
        "aqr_century": aqr.load_dataset("century_of_factor_premia"),
        "aqr_clr": aqr.load_dataset("commodities_long_run"),
        "gsw_nominal": gsw.load_curve("nominal"),
    }
    for name, frame in reads.items():
        assert pd.DatetimeIndex(frame.index).max() < boundary, name
        assert holdout.is_truncated(frame), name


@pytest.mark.dataset
def test_the_unrestricted_accessor_still_sees_what_the_source_published() -> None:
    """The other half: the escape hatch has to actually work, or coverage breaks."""
    boundary = pd.Timestamp(load().require_holdout_start())
    full = french.load_daily_factors_unrestricted(reason="test: source coverage")
    assert pd.DatetimeIndex(full.index).max() > boundary
    assert len(full) > len(french.load_daily_factors())
