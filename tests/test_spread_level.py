"""The spread level file and the additive combination rule. SPEC.md 3.4.1, 7.1."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mafrm import config
from mafrm.costs import inputs, spread_level

EXPECTED = {"a_asset": "AAA", "b_asset": "BBB"}


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "levels.yaml"
    path.write_text("schema_version: 1\nentries:\n" + body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The committed file
# ---------------------------------------------------------------------------


def test_the_committed_file_covers_every_cost_bearing_asset_with_its_cost_ticker() -> None:
    cfg = config.load()
    expected = inputs._cost_assets(cfg)
    assert len(expected) == 13
    assert "dollar" not in expected  # an index level, not an instrument
    levels = spread_level.load_spread_levels(
        config._REPO_ROOT / cfg.model.costs.spread_level.file, expected=expected
    )
    assert set(levels.entries) == set(expected)
    # the five curve points are keyed to their proxies
    assert expected["govt_30y"] == "TLT" and expected["tips_10y"] == "TIP"


def test_a_placeholder_row_is_refused_by_name_and_has_no_default() -> None:
    cfg = config.load()
    expected = inputs._cost_assets(cfg)
    levels = spread_level.load_spread_levels(
        config._REPO_ROOT / cfg.model.costs.spread_level.file, expected=expected
    )
    for asset in levels.placeholders:
        with pytest.raises(spread_level.SpreadLevelError, match=f"'{asset}'.*placeholder"):
            levels.require(asset)


def test_build_cost_inputs_refuses_while_any_placeholder_remains() -> None:
    """The refusal comes BEFORE any bar is read, so it needs no cache."""
    cfg = config.load()
    expected = inputs._cost_assets(cfg)
    levels = spread_level.load_spread_levels(
        config._REPO_ROOT / cfg.model.costs.spread_level.file, expected=expected
    )
    if not levels.placeholders:
        pytest.skip("every level supplied; the refusal path is not reachable")
    with pytest.raises(spread_level.SpreadLevelError, match="placeholder"):
        inputs.build_cost_inputs(cfg)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_a_sourced_point_value_parses(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "  - {asset: a_asset, ticker: AAA, source_url: 'https://x/a', disclosure_date: '2026-09-03', median_spread_bps: 1.25}\n"
        "  - {asset: b_asset, ticker: BBB, source_url: null, disclosure_date: null, median_spread_bps: null}\n",
    )
    levels = spread_level.load_spread_levels(path, expected=EXPECTED)
    a = levels.require("a_asset")
    assert isinstance(a, spread_level.IssuerLevel)
    assert a.median_spread_bps == 1.25
    assert a.disclosure_date == date(2026, 9, 3)
    assert a.levels_bps == (1.25,)
    assert levels.placeholders == ("b_asset",)
    assert levels.supplied == ("a_asset",)


def test_a_sourced_band_parses_and_sweeps_both_ends(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "  - {asset: a_asset, ticker: AAA, source_url: 'https://x/a', disclosure_date: '2026-09-03', band_bps: [2.0, 6.0]}\n"
        "  - {asset: b_asset, ticker: BBB, source_url: null, disclosure_date: null, median_spread_bps: null}\n",
    )
    band = spread_level.load_spread_levels(path, expected=EXPECTED).require("a_asset")
    assert isinstance(band, spread_level.SpreadBand)
    assert band.levels_bps == (2.0, 6.0)


@pytest.mark.parametrize(
    ("row", "fragment"),
    [
        # a value without provenance is not a disclosed figure
        (
            "{asset: a_asset, ticker: AAA, source_url: null, disclosure_date: '2026-09-03', median_spread_bps: 1.0}",
            "source_url",
        ),
        (
            "{asset: a_asset, ticker: AAA, source_url: 'https://x', disclosure_date: null, median_spread_bps: 1.0}",
            "disclosure_date",
        ),
        # a placeholder carrying provenance reads as a lost value
        (
            "{asset: a_asset, ticker: AAA, source_url: 'https://x', disclosure_date: null, median_spread_bps: null}",
            "placeholder",
        ),
        # both forms at once
        (
            "{asset: a_asset, ticker: AAA, source_url: 'https://x', disclosure_date: '2026-09-03', median_spread_bps: 1.0, band_bps: [1, 2]}",
            "not both",
        ),
        # a band must be ordered
        (
            "{asset: a_asset, ticker: AAA, source_url: 'https://x', disclosure_date: '2026-09-03', band_bps: [3.0, 3.0]}",
            "low must be below high",
        ),
        # a negative spread
        (
            "{asset: a_asset, ticker: AAA, source_url: 'https://x', disclosure_date: '2026-09-03', median_spread_bps: -1.0}",
            "non-negative",
        ),
        # the wrong ticker for the asset
        (
            "{asset: a_asset, ticker: ZZZ, source_url: null, disclosure_date: null, median_spread_bps: null}",
            "priced by AAA",
        ),
    ],
)
def test_malformed_rows_are_refused(tmp_path: Path, row: str, fragment: str) -> None:
    path = _write(
        tmp_path,
        f"  - {row}\n"
        "  - {asset: b_asset, ticker: BBB, source_url: null, disclosure_date: null, median_spread_bps: null}\n",
    )
    with pytest.raises(spread_level.SpreadLevelError, match=fragment):
        spread_level.load_spread_levels(path, expected=EXPECTED)


def test_the_file_must_hold_exactly_the_cost_bearing_assets(tmp_path: Path) -> None:
    only_a = _write(
        tmp_path,
        "  - {asset: a_asset, ticker: AAA, source_url: null, disclosure_date: null, median_spread_bps: null}\n",
    )
    with pytest.raises(spread_level.SpreadLevelError, match="no row for \\['b_asset'\\]"):
        spread_level.load_spread_levels(only_a, expected=EXPECTED)
    extra = _write(
        tmp_path,
        "  - {asset: a_asset, ticker: AAA, source_url: null, disclosure_date: null, median_spread_bps: null}\n"
        "  - {asset: b_asset, ticker: BBB, source_url: null, disclosure_date: null, median_spread_bps: null}\n"
        "  - {asset: dollar, ticker: DTWEXBGS, source_url: null, disclosure_date: null, median_spread_bps: null}\n",
    )
    with pytest.raises(spread_level.SpreadLevelError, match="carry no cost"):
        spread_level.load_spread_levels(extra, expected=EXPECTED)


# ---------------------------------------------------------------------------
# The combination rule, by hand
# ---------------------------------------------------------------------------

BP = 1e-4


def _series(values_bps: list[float]) -> pd.Series:
    index = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30"])[
        : len(values_bps)
    ]
    return pd.Series([v * BP for v in values_bps], index=index)


def test_calm_baseline_is_the_median_and_widening_clips_at_zero() -> None:
    """Production 0, 5, 10, 40bp: median (5+10)/2 = 7.5bp; widening 0, 0, 2.5, 32.5bp."""
    production = _series([0.0, 5.0, 10.0, 40.0])
    baseline = spread_level.calm_baseline(production)
    assert baseline == pytest.approx(7.5 * BP, rel=1e-12)
    widened = spread_level.widening(production, baseline) / BP
    assert list(widened.round(12)) == [0.0, 0.0, 2.5, 32.5]


def test_calm_baseline_refuses_the_signed_series() -> None:
    with pytest.raises(spread_level.SpreadLevelError, match="CLIPPED"):
        spread_level.calm_baseline(_series([-3.0, 5.0, 10.0]))


def test_widening_keeps_nan_as_nan() -> None:
    production = _series([0.0, 5.0, 10.0, 40.0])
    production.iloc[1] = np.nan
    widened = spread_level.widening(production, 5.0 * BP)
    assert np.isnan(widened.iloc[1])
    assert widened.iloc[3] == pytest.approx(35.0 * BP, rel=1e-12)


def test_spread_series_hand_computed_and_agrees_with_the_level_at_baseline() -> None:
    """Level 3bp on production 0, 5, 10, 40bp.

    baseline 7.5bp; widening 0, 0, 2.5, 32.5; full 3, 3, 5.5, 35.5; half (no
    commission) 1.5, 1.5, 2.75, 17.75bp. The first two months sit at or below
    the baseline, so the full spread there IS the issuer level, exactly.
    """
    out = spread_level.spread_series(
        level_bps=3.0, production=_series([0.0, 5.0, 10.0, 40.0]), commission=0.0
    )
    assert out.level == pytest.approx(3.0 * BP, rel=1e-12)
    assert out.baseline == pytest.approx(7.5 * BP, rel=1e-12)
    assert list((out.full / BP).round(12)) == [3.0, 3.0, 5.5, 35.5]
    assert list((out.half / BP).round(12)) == [1.5, 1.5, 2.75, 17.75]
    assert out.full.iloc[0] == out.level and out.full.iloc[1] == out.level


def test_commission_enters_additively_after_the_halving() -> None:
    """Half of 3bp is 1.5bp; a 0.2bp commission makes a_i 1.7bp, not 1.6 or 3.2."""
    out = spread_level.spread_series(
        level_bps=3.0, production=_series([0.0, 0.0, 0.0]), commission=0.2 * BP
    )
    assert list((out.half / BP).round(12)) == [1.7, 1.7, 1.7]


def test_the_level_is_never_scaled_by_the_edge_series() -> None:
    """Doubling every EDGE estimate moves the full spread by the widening only."""
    base = spread_level.spread_series(
        level_bps=3.0, production=_series([0.0, 5.0, 10.0, 40.0]), commission=0.0
    )
    doubled = spread_level.spread_series(
        level_bps=3.0, production=_series([0.0, 10.0, 20.0, 80.0]), commission=0.0
    )
    # baseline doubles to 15bp; widening 0, 0, 5, 65; full 3, 3, 8, 68 -- the level term is untouched
    assert list((doubled.full / BP).round(12)) == [3.0, 3.0, 8.0, 68.0]
    assert doubled.level == base.level


# ---------------------------------------------------------------------------
# The interval ruling: a displayed figure is a rounding interval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("displayed", "low", "high"),
    [
        (0.0, 0.0, 0.5),  # "0.00%" -> [0, 0.5)
        (1.0, 0.5, 1.5),  # "0.01%" -> [0.5, 1.5)
        (2.0, 1.5, 2.5),
        (3.0, 2.5, 3.5),
    ],
)
def test_a_displayed_point_becomes_a_rounding_interval_band(
    tmp_path: Path, displayed: float, low: float, high: float
) -> None:
    path = _write(
        tmp_path,
        f"  - {{asset: a_asset, ticker: AAA, source_url: 'https://x/a', disclosure_date: '2026-09-03', median_spread_bps: {displayed}}}\n"
        "  - {asset: b_asset, ticker: BBB, source_url: null, disclosure_date: null, median_spread_bps: null}\n",
    )
    entry = spread_level.load_spread_levels(
        path, expected=EXPECTED, display_resolution_bps=1.0
    ).require("a_asset")
    assert isinstance(entry, spread_level.SpreadBand)
    assert entry.levels_bps == (low, high)
    assert entry.displayed_bps == displayed
    # without a resolution the same row is an exact point
    exact = spread_level.load_spread_levels(path, expected=EXPECTED).require("a_asset")
    assert isinstance(exact, spread_level.IssuerLevel) and exact.levels_bps == (displayed,)


def test_a_second_source_is_parsed_and_does_not_enter_the_band(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "  - asset: a_asset\n    ticker: AAA\n    source_url: 'https://x/a'\n    disclosure_date: '2026-09-03'\n"
        "    median_spread_bps: 0.0\n    second_source: {url: 'https://x/b', value_bps: 0.6, note: 'a 2023 mean'}\n"
        "  - {asset: b_asset, ticker: BBB, source_url: null, disclosure_date: null, median_spread_bps: null}\n",
    )
    entry = spread_level.load_spread_levels(
        path, expected=EXPECTED, display_resolution_bps=1.0
    ).require("a_asset")
    assert isinstance(entry, spread_level.SpreadBand)
    assert entry.levels_bps == (0.0, 0.5)
    assert entry.second_source is not None and entry.second_source.value_bps == 0.6


def test_a_non_positive_resolution_is_refused(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "  - {asset: a_asset, ticker: AAA, source_url: null, disclosure_date: null, median_spread_bps: null}\n",
    )
    with pytest.raises(spread_level.SpreadLevelError, match="positive"):
        spread_level.load_spread_levels(
            path, expected={"a_asset": "AAA"}, display_resolution_bps=0.0
        )


def test_has_band_ignores_nan_on_both_sides() -> None:
    idx = pd.to_datetime(["2020-01-31", "2020-02-28"])
    low = pd.DataFrame({"A": [np.nan, 1e-4]}, index=idx)
    same = inputs.CostInputs(
        half_spread_low=low,
        half_spread_high=low.copy(),
        daily_volatility=low,
        adv_dollars=low,
        shapes={},
        levels=spread_level.SpreadLevels(path=Path("x"), entries={}),
    )
    assert same.has_band is False
    high = pd.DataFrame({"A": [np.nan, 2e-4]}, index=idx)
    differs = inputs.CostInputs(
        half_spread_low=low,
        half_spread_high=high,
        daily_volatility=low,
        adv_dollars=low,
        shapes={},
        levels=spread_level.SpreadLevels(path=Path("x"), entries={}),
    )
    assert differs.has_band is True


@pytest.mark.dataset
def test_the_committed_file_builds_inputs_as_bands_at_one_bp_resolution() -> None:
    cfg = config.load()
    ci = inputs.build_cost_inputs(cfg)
    assert ci.levels.placeholders == ()
    assert ci.has_band
    assert ci.half_spread_low.shape[1] == 13
    assert ci.half_spread_low.index.max() < pd.Timestamp(cfg.require_holdout_start())
    width = (ci.half_spread_high - ci.half_spread_low).median()
    # half of a 1bp interval, or half of a [0, 0.5) floor interval
    assert set(width.round(9)) <= {0.5e-4, 0.25e-4}
