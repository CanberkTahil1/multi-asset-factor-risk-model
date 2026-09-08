"""SPEC.md 7.4's engine reconciliation against bt, on the synthetic panel. W5-P3.

experiments.md rows 199-200 and 205-206. The real-cache rows (201-203) read
``data/raw`` and run under the ``dataset`` marker, weekly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mafrm import config
from mafrm.backtest import reconciliation as rec
from mafrm.data import calendar


@pytest.fixture(scope="module")
def synthetic_rows() -> tuple[rec.Comparison, ...]:
    cfg = config.load()
    return rec.reconcile_panel(
        rec.synthetic_panel(cfg),
        cfg.model.backtest.reconciliation.synthetic_weights,
        panel="synthetic 60/40",
        cfg=cfg,
    )


def test_the_synthetic_panel_is_byte_reproducible_from_the_seed() -> None:
    a = rec.synthetic_panel().to_numpy().tobytes()
    b = rec.synthetic_panel().to_numpy().tobytes()
    assert a == b
    panel = rec.synthetic_panel()
    assert list(panel.columns) == ["synthetic_a", "synthetic_b"]
    assert (panel.to_numpy() > 0.0).all()


def test_constant_targets_sit_on_the_projects_month_ends() -> None:
    panel = rec.synthetic_panel()
    targets = rec.constant_targets(panel, (0.6, 0.4))
    assert targets.index.equals(calendar.month_end_dates(pd.DatetimeIndex(panel.index)))
    assert (targets.sum(axis=1) == 1.0).all()
    equal = rec.constant_targets(panel, "equal_weight")
    assert np.allclose(equal.to_numpy(), 0.5)


def test_the_bound_is_the_configured_construction() -> None:
    rules = config.load().model.backtest.reconciliation
    eps = np.finfo(np.float64).eps
    assert rules.coefficient(2) == 2 * (2 + 8)
    assert rules.coefficient(13) == 2 * (13 + 8)
    assert rules.tolerance(assets=2, steps=756) == 20 * eps * 756


def test_row_199_zero_cost_engine_and_bt_agree_to_the_bound(
    synthetic_rows: tuple[rec.Comparison, ...],
) -> None:
    row = synthetic_rows[0]
    assert row.fee_financing == "none" and row.half_spread == 0.0
    assert row.tolerance == config.load().model.backtest.reconciliation.tolerance(
        assets=2, steps=row.steps
    )
    assert row.within_tolerance, f"gap {row.max_abs_gap:.3e} vs bound {row.tolerance:.3e}"
    assert row.as_expected


def test_row_200_cash_financed_spread_agrees_to_the_bound(
    synthetic_rows: tuple[rec.Comparison, ...],
) -> None:
    row = synthetic_rows[1]
    assert row.fee_financing == "cash"
    assert row.half_spread == pytest.approx(2.5e-4)
    assert row.within_tolerance, f"gap {row.max_abs_gap:.3e} vs bound {row.tolerance:.3e}"
    # The cost actually bit: both books end below the zero-cost book.
    free = synthetic_rows[0]
    assert row.reference.iloc[-1] < free.reference.iloc[-1]
    assert row.comparand.iloc[-1] < free.comparand.iloc[-1]


def test_row_205_bt_native_commission_is_a_convention_gap_not_noise(
    synthetic_rows: tuple[rec.Comparison, ...],
) -> None:
    row = synthetic_rows[2]
    assert row.expectation == "exceeds"
    assert not row.within_tolerance
    assert abs(row.annualised_gap_bps) < 1.0
    assert row.as_expected


def test_row_206_the_shadow_replay_attributes_the_whole_gap(
    synthetic_rows: tuple[rec.Comparison, ...],
) -> None:
    row = synthetic_rows[3]
    assert row.fee_financing == "trade"
    assert row.within_tolerance, (
        f"UNATTRIBUTED gap {row.max_abs_gap:.3e} vs bound {row.tolerance:.3e}"
    )
    # And the shadow is not the engine: it differs from the cash-financed book.
    assert (row.reference - synthetic_rows[1].reference).abs().max() > row.tolerance


def test_fee_financing_is_one_of_two_conventions() -> None:
    panel = rec.synthetic_panel().iloc[:40]
    targets = rec.constant_targets(panel, (0.6, 0.4))
    with pytest.raises(ValueError, match="fee_financing"):
        rec.run_bt(panel, targets, half_spread=0.0, initial_nav=1.0e6, fee_financing="margin")  # type: ignore[arg-type]


def test_the_shadow_stops_where_bt_stops() -> None:
    """bt leaves numpy.isclose's atol at its default; the replay reads that default."""
    assert rec._bt_absolute_stop() == 1e-8


@pytest.mark.dataset
def test_rows_201_to_203_on_the_real_cache() -> None:
    result = rec.reconciliation(include_real=True)
    real = [row for row in result.rows if row.panel.startswith("real")]
    assert len(real) == 4 and all(row.assets == 13 for row in real)
    for row in real:
        assert row.as_expected, (
            f"{row.label}: gap {row.max_abs_gap:.3e} vs bound {row.tolerance:.3e}"
        )
    native = real[2]
    assert abs(native.annualised_gap_bps) < 1.0
