"""reports/cost_calibration.md: the derivation section, pinned by hand. SPEC.md 7.2."""

from __future__ import annotations

import pytest

from mafrm import config
from mafrm.costs import calibration_report, inputs
from mafrm.data import cache


def test_derivation_reproduces_spec_7_2_by_hand() -> None:
    d = calibration_report.derive(config.load())
    implied = {(p.regime, p.participation): p.implied_y for p in d.points}
    # 0.0017 / (0.02 * sqrt(0.02)); 0.0028 / (0.02 * sqrt(0.06)); 0.0040 / (0.02 * sqrt(0.02))
    assert implied[("patient", 0.02)] == pytest.approx(0.601, abs=5e-4)
    assert implied[("patient", 0.06)] == pytest.approx(0.572, abs=5e-4)
    assert implied[("urgent", 0.02)] == pytest.approx(1.414, abs=5e-4)
    assert d.patient_mean == pytest.approx((0.6010 + 0.5715) / 2, abs=1e-3)
    assert d.patient_config == 0.58 and d.urgent_config == 1.40
    assert d.regime_ratio == pytest.approx(2.41, abs=0.01)
    # 0.58 * 0.02 * sqrt(0.01) = 0.00116 = 11.6bp
    assert d.cross_check_bps == pytest.approx(11.6, rel=1e-12)
    assert d.cross_check_expected_bps == 11.6


def test_render_needs_no_cache_for_sections_one_and_two(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_cache(cfg: object = None) -> dict[str, object]:
        raise cache.CacheError("no cached bars in this test")

    monkeypatch.setattr(inputs, "spread_shapes", no_cache)
    text = calibration_report.render(config.load())
    assert "**0.601**" in text and "**0.572**" in text and "**1.414**" in text
    assert "11.60bp" in text
    assert "1.41x" in text  # doubling AUM
    assert "not available" in text
    assert "The cost model does not run" in text or "issuer levels are supplied" in text
    assert "could not be read" not in text
    assert "half the duration" in text or "half that exposure" in text


@pytest.mark.dataset
def test_render_against_the_cache_tabulates_all_thirteen() -> None:
    text = calibration_report.render(config.load())
    for asset in ("us_large_equity", "govt_2y", "govt_30y", "tips_10y", "gold"):
        assert f"| {asset} |" in text
    assert "TLT (proxy)" in text


def test_resolution_arithmetic_by_hand() -> None:
    """0.58 * 0.02 * sqrt(0.02) = 16.40bp; 0.58 * 0.02 * sqrt(0.001) = 3.668bp.

    A 0.5bp half-spread is 3.05% of the first and 13.6% of the second.
    """
    ra = calibration_report.resolution_arithmetic(config.load())
    assert ra.half_spread_bps == 0.5
    assert ra.impact_large_bps == pytest.approx(16.405, abs=5e-3)
    assert ra.impact_small_bps == pytest.approx(3.668, abs=5e-3)
    assert ra.share_large == pytest.approx(0.0305, abs=5e-4)
    assert ra.share_small == pytest.approx(0.1363, abs=5e-4)
