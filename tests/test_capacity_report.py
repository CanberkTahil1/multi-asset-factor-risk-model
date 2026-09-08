"""reports/capacity.md and reports/capacity.png in normalised units, pinned by hand. SPEC.md 10.4, W5-P2."""

from __future__ import annotations

from pathlib import Path

import pytest

from mafrm import config
from mafrm.costs import capacity_report

EXACT = {"rel": 1e-12, "abs": 0.0}


def test_normalised_curve_reproduces_the_ratios_by_hand() -> None:
    """A_BE(patient) = 1 by construction; A_BE(urgent) = (0.58/1.40)^2; A_eff = 4/9 of each."""
    norm = capacity_report.normalised(config.load())
    points = norm.curve.points
    assert norm.y_patient == 0.58 and norm.y_urgent == 1.40
    assert points.loc["patient", "break_even_aum"] == pytest.approx(1.0, **EXACT)
    assert points.loc["urgent", "break_even_aum"] == pytest.approx((0.58 / 1.40) ** 2, rel=1e-12)
    assert norm.band_ratio == pytest.approx(0.171633, abs=5e-7)
    assert norm.effective_ratio == pytest.approx(4.0 / 9.0, **EXACT)
    for regime in ("patient", "urgent"):
        ratio = points.loc[regime, "effective_aum"] / points.loc[regime, "break_even_aum"]
        assert ratio == pytest.approx(4.0 / 9.0, **EXACT)
    assert norm.halving_multiplier == pytest.approx(4.0, **EXACT)
    # Normalised units: net alpha is 1 at zero AUM, 1 - sqrt(x) on the patient curve.
    net = norm.curve.net_alpha["patient"]
    assert net.iloc[0] == 1.0
    x = norm.curve.aum
    assert net.to_numpy() == pytest.approx(1.0 - x**0.5, rel=1e-12, abs=1e-15)


def test_no_project_input_reaches_the_normalised_curve() -> None:
    """The only values that enter are the two Y regimes and the exponent: alpha_g = tau = 1."""
    norm = capacity_report.normalised(config.load())
    assert norm.curve.gross_alpha == 1.0
    assert norm.curve.turnover == 1.0
    assert norm.curve.aum_invariant_drag == 0.0
    assert norm.curve.points.loc["patient", "kappa"] == 1.0
    assert norm.curve.points.loc["urgent", "kappa"] == pytest.approx(1.40 / 0.58, rel=1e-12)


def test_chart_marks_both_points_per_regime_and_writes_the_figure(tmp_path: Path) -> None:
    norm = capacity_report.normalised(config.load())
    # Both marked points exist for both regimes before anything is drawn ...
    assert set(norm.curve.points.columns) >= {"break_even_aum", "effective_aum"}
    assert list(norm.curve.points.index) == ["patient", "urgent"]
    # ... and the figure is written where the acceptance says it lives.
    path = tmp_path / "capacity.png"
    capacity_report.chart(norm, path)
    assert path.exists() and path.stat().st_size > 0
    assert capacity_report.figure_path().name == "capacity.png"


def test_render_says_what_the_closed_form_is_and_publishes_no_number() -> None:
    text = capacity_report.render(config.load())
    assert "NO PROJECT NUMBER IS PUBLISHED" in text
    assert "RESCALING" in text and "overstates cost" in text
    assert "borrowed rather than estimated" in text
    assert "**0.1716**" in text  # the band
    assert "**4.0x**" in text  # halving kappa
    assert "0.444444" in text  # 4/9
    assert "W6-P1" in text and "W6-P2/W6-P3" in text
    assert "no interior points" in text  # SPEC.md 7.2.1 stands
    assert "alpha_min = 0" in text
    # No dollar amount anywhere: the report is in normalised units.
    assert "$" not in text
