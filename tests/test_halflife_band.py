"""W8-P1's reference-book half-life sweep: what it refuses, and what a point reports."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mafrm import config
from mafrm.backtest import halflife_band

CFG = config.load()


def _panel(dates: pd.DatetimeIndex, column: str) -> Any:
    return SimpleNamespace(dates=dates, column=column)


@dataclasses.dataclass(frozen=True)
class _Universe:
    panel: Any
    horizon: str
    other: str


def test_universe_at_swaps_the_panel_only_when_the_windows_agree() -> None:
    """The universe's calendar is derived from the panel's dates, so a different window is refused."""
    dates = pd.bdate_range("2010-01-04", periods=50)
    base = _Universe(panel=_panel(dates, "short"), horizon="long", other="kept")
    swapped = halflife_band.universe_at(base, _panel(dates, "hl21"))  # type: ignore[arg-type]
    assert swapped.panel.column == "hl21"
    assert swapped.horizon == "short"
    assert swapped.other == "kept"
    with pytest.raises(halflife_band.HalflifeBandError, match="differs"):
        halflife_band.universe_at(base, _panel(dates[1:], "hl21"))  # type: ignore[arg-type]


def test_the_sweep_refuses_to_run_before_its_rows_are_counted(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """experiments.md rule 1, enforced the way bands.run_bands enforces it."""
    fake = tmp_path / "experiments.md"
    fake.write_text("| **Total evaluated** | **298** | **`N` = 61** |\n", encoding="utf-8")
    monkeypatch.setattr(halflife_band, "_EXPERIMENTS_PATH", fake)
    with pytest.raises(halflife_band.HalflifeBandError, match="register the rows BEFORE"):
        halflife_band.run_sweep(CFG, progress=False)


def test_the_trials_before_constant_is_the_w7_p4b_count_and_the_grid_adds_nine() -> None:
    grid = CFG.model.validation.halflife_sensitivity.grid
    assert halflife_band._TRIALS_BEFORE == 61
    assert halflife_band._TRIALS_BEFORE + len(grid) == 70


def test_a_book_point_reports_the_summary_and_the_kish_sizes() -> None:
    summary = {
        "months": 187.0,
        "bias": 1.065,
        "bias_lower": 0.898,
        "bias_upper": 1.101,
        "bias_ratio": 1.061,
        "risk_model_term": 0.0543,
        "cost_term": 0.04,
        "sharpe_paper": 0.944,
        "turnover_annual": 5.62,
        "cost_drag_bps_per_year": 28.2,
        "te_attainment_rms": 0.997,
        "inaccurate": 0.0,
        "resolve_max_l1": 0.0,
    }
    result = SimpleNamespace(
        summary=summary,
        diagnostics=None,
        sharpe_net=SimpleNamespace(annualised=0.85),
        deflated=SimpleNamespace(probability=0.993),
    )
    point = halflife_band.BookPoint(
        halflife=84,
        result=result,  # type: ignore[arg-type]
        factors=6,
        k_over_t=np.array([0.03, 0.025, 0.0248]),
        kish_volatility=np.array([200.0, 240.0, 242.0]),
        kish_correlation=np.array([1000.0, 1400.0, 1454.0]),
    )
    row = point.row
    assert row["halflife"] == 84.0
    assert row["factors"] == 6.0
    assert row["bias"] == 1.065 and row["bias_ratio"] == 1.061
    assert np.isnan(row["b_factor"]) and np.isnan(row["b_specific"])
    assert row["k_over_t_mean"] == pytest.approx(np.mean([0.03, 0.025, 0.0248]))
    assert row["kish_correlation_mean"] == pytest.approx(np.mean([1000.0, 1400.0, 1454.0]))
    assert row["deflated_sharpe"] == 0.993
    assert row["sharpe_net"] == 0.85
