"""``reports/eigenfactor.md``. SPEC.md 5.3.

``build()`` reads the gitignored cache and is exercised by ``make report``, not
here. What is pinned here is what the report is not allowed to claim: it reports
a **forecast revision**, and a forecast revision is not a bias statistic. The
distinction is easy to lose in a sentence and it is the difference between a
diagnostic and a result about H1-H3 that nothing in this session has earned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mafrm.factors.eigenfactor_report import (
    FixtureReading,
    Reading,
    _minimum_variance,
    draw,
    render,
)
from mafrm.risk.eigenfactor import eigenfactor_adjustment


def _adjustment(size: int = 6):  # type: ignore[no-untyped-def]
    rotation, _ = np.linalg.qr(np.random.default_rng(11).standard_normal((size, size)))
    spectrum = np.linspace(0.5, 1.8, size)
    matrix = rotation @ np.diag(spectrum) @ rotation.T
    matrix = matrix / np.sqrt(np.outer(np.diag(matrix), np.diag(matrix)))
    return matrix, eigenfactor_adjustment(
        matrix,
        trials=200,
        halflife=84.0,
        observations=1000,
        scalings=(1.0, 1.4),
        seed=5,
        floor=1e-14,
    )


def _reading(horizon: str = "short") -> Reading:
    matrix, adjustment = _adjustment()
    return Reading(
        horizon=horizon,  # type: ignore[arg-type]
        columns=tuple(f"f{index}" for index in range(6)),
        observations=4146,
        start=pd.Timestamp("2008-04-14"),
        end=pd.Timestamp("2024-12-31"),
        volatility_halflife=84,
        correlation_halflife=504,
        trials=2000,
        shepard_multiplier=1.0514,
        unadjusted=matrix,
        adjustment=adjustment,
        minimum_variance_ratio={1.0: 1.038, 1.4: 1.053},
        random_median_ratio={1.0: 1.0026, 1.4: 1.0038},
        random_upper_ratio={1.0: 1.020, 1.4: 1.028},
        invariance_residual=7.4e-16,
    )


def _fixtures() -> tuple[FixtureReading, ...]:
    return (
        FixtureReading(
            factors=3,
            amplitude=0.0175,
            residual_degrees_of_freedom=0,
            exact_interpolation=True,
            bias_minimum=0.998,
            bias_maximum=1.015,
        ),
        FixtureReading(
            factors=40,
            amplitude=0.6650,
            residual_degrees_of_freedom=37,
            exact_interpolation=False,
            bias_minimum=0.784,
            bias_maximum=1.574,
        ),
    )


def _rendered() -> str:
    return render((_reading("short"), _reading("long")), _fixtures())


# ---------------------------------------------------------------------------
# What the report may not claim
# ---------------------------------------------------------------------------


def test_the_report_says_a_forecast_revision_is_not_a_bias_statistic() -> None:
    """The one sentence that keeps this a diagnostic rather than a claim.

    Nothing in W3-P3 compares a forecast against a realised return. H1, H2 and H3
    are week 4's and they need the bias machinery; a report that quietly implied
    otherwise would be claiming a pre-registered hypothesis had been tested when
    it had not.
    """
    text = _rendered()
    assert "not a bias statistic" in text
    assert "H1, H2 or H3" in text


def test_the_report_does_not_call_any_hypothesis_confirmed() -> None:
    """No H-number may be reported as settled here, in either direction."""
    text = _rendered().lower()
    for verdict in ("h1 confirmed", "h2 confirmed", "h3 confirmed", "h3 holds", "confirms h"):
        assert verdict not in text


def test_the_report_withdraws_the_shepard_agreement_w3p3_claimed() -> None:
    """W3-P3's cross-check was two errors cancelling, and the report must say so.

    It compared a revision computed at ``T = 242`` -- the wrong effective sample
    size for a correlation estimator -- against a Shepard figure for a window the
    stage does not correct, and reported the agreement as corroboration. Two wrong
    numbers agreeing is the most dangerous kind of corroboration, because it reads
    as confirmation from an independent source.
    """
    text = _rendered()
    assert "two errors cancelling" in text
    assert "1.353" in text
    assert "coincidence" in text


def test_the_report_states_the_uncorrected_volatility_leg() -> None:
    """The cost of the correlation-space ruling, which the ruling did not name.

    In covariance space the adjustment corrected ``sigma`` and ``rho`` jointly.
    In correlation space it corrects ``rho`` alone, and the two Shepard columns
    are what put a number on the difference.
    """
    text = _rendered()
    assert "uncorrected `sigma` leg" in text
    assert "passes through this stage **uncorrected**" in text


# ---------------------------------------------------------------------------
# What it must contain
# ---------------------------------------------------------------------------


def test_the_report_states_the_invariance_residual_as_a_number() -> None:
    """SPEC.md 5.3.1's criterion is mechanical, so it is published as a number.

    "Passes" is not a measurement. A tolerance that is never printed is one
    nobody checks.
    """
    assert "7.40e-16" in _rendered() or "7.4e-16" in _rendered()


def test_the_report_carries_both_published_scalings_everywhere() -> None:
    text = _rendered()
    assert "a = 1.0" in text
    assert "a = 1.4" in text
    assert "attribution-facing" in text and "optimizer-facing" in text


def test_the_report_says_the_parabola_interpolates_at_k3() -> None:
    """SPEC.md 5.3.2 asks for the weak smoothing to be reported, not assumed away."""
    text = _rendered()
    assert "exact interpolation" in text
    assert "residual d.o.f." in text


def test_the_report_states_the_amplitude_gate_is_withdrawn_and_why() -> None:
    text = _rendered()
    assert "withdrawn" in text
    assert "large-K" in text or "large-`K`" in text
    assert "PRE-REGISTERED" in text


def test_the_report_contrasts_optimizer_selected_with_random_portfolios() -> None:
    """SPEC.md 6.2: random portfolios validate any covariance matrix at all."""
    text = _rendered()
    assert "random portfolios validate any covariance matrix" in text
    assert "minimum-variance" in text


# ---------------------------------------------------------------------------
# The arithmetic the report publishes
# ---------------------------------------------------------------------------


def test_the_minimum_variance_portfolio_is_hand_computable_on_a_diagonal_matrix() -> None:
    """HAND-COMPUTED. For ``diag(1, 4)`` the min-variance weights are 4/5 and 1/5.

    ``w = S^-1 1 / (1' S^-1 1)`` gives ``(1, 1/4) / (5/4) = (0.8, 0.2)``, and the
    forecast variance is ``0.8^2 * 1 + 0.2^2 * 4 = 0.64 + 0.16 = 0.8``.
    """
    weights = _minimum_variance(np.diag([1.0, 4.0]))
    np.testing.assert_allclose(weights, [0.8, 0.2], rtol=0.0, atol=1e-12)
    assert weights @ np.diag([1.0, 4.0]) @ weights == pytest.approx(0.8)


def test_the_minimum_variance_portfolio_is_fully_invested() -> None:
    matrix, _ = _adjustment()
    assert _minimum_variance(matrix).sum() == pytest.approx(1.0)


def test_the_figure_has_one_panel_per_horizon_with_both_curves() -> None:
    figure = draw((_reading("short"), _reading("long")))
    axes = figure.get_axes()
    assert len(axes) == 2
    for axis in axes:
        # raw curve, fitted parabola, and the reference line at 1.
        assert len(axis.get_lines()) == 3
    figure.clf()
