"""assert_psd. CLAUDE.md invariant 4.

The point of these is that the assertion detects each failure separately and
says which one it was: a message that reports "not PSD" for an asymmetric matrix
sends the next session to the wrong stage.
"""

from __future__ import annotations

import numpy as np
import pytest

from mafrm.config import load
from mafrm.risk.checks import PsdError, assert_psd, eigenvalue_floor


def test_identity_passes_and_reports_its_eigenvalues() -> None:
    check = assert_psd(np.eye(4), "unit test")
    assert check.size == 4
    assert check.minimum_eigenvalue == pytest.approx(1.0)
    assert check.maximum_eigenvalue == pytest.approx(1.0)
    assert check.condition_number == pytest.approx(1.0)
    assert check.maximum_asymmetry == 0.0


def test_a_singular_but_valid_matrix_passes_and_reports_infinite_condition() -> None:
    """A zero eigenvalue is PSD. It is not an error, and the report says so."""
    matrix = np.outer([1.0, 2.0], [1.0, 2.0])
    check = assert_psd(matrix, "rank one")
    assert check.minimum_eigenvalue == pytest.approx(0.0, abs=1e-15)
    assert not np.isfinite(check.condition_number)
    assert "singular" in check.render()


def test_roundoff_negative_eigenvalue_is_tolerated() -> None:
    """The threshold exists because eigh on a valid matrix returns -1e-17, not 0."""
    numerics = load().model.numerics
    matrix = np.diag([1.0, eigenvalue_floor(2, 1.0, numerics=numerics) / 2.0])
    assert assert_psd(matrix, "roundoff").minimum_eigenvalue < 0.0


def test_a_genuinely_negative_eigenvalue_fails() -> None:
    numerics = load().model.numerics
    matrix = np.diag([1.0, eigenvalue_floor(2, 1.0, numerics=numerics) * 1000.0])
    with pytest.raises(PsdError, match="not positive semi-definite"):
        assert_psd(matrix, "newey_west")


def test_eigenvalue_floor_is_four_epsilons_at_the_hand_computed_case() -> None:
    """HAND-COMPUTED. W4-P2b ruling: the bound is -(size + roundings) * eps * lambda_max.

    A 2x2 with ``lambda_max = 1`` and ``psd_reconstruction_roundings = 2`` gives
    ``-(2 + 2) * 2.220446049250313e-16 * 1 = -8.881784197001252e-16`` exactly,
    because the coefficient is an integer and eps is a power of two, so the
    product is representable with no rounding at all.
    """
    numerics = load().model.numerics
    assert numerics.psd_reconstruction_roundings == 2
    assert eigenvalue_floor(2, 1.0, numerics=numerics) == -8.881784197001252e-16
    # And at the scale that motivated the change: K = 6, lambda_max = 4.4e4.
    assert eigenvalue_floor(6, 4.4e4, numerics=numerics) == pytest.approx(
        -8 * 2.220446049250313e-16 * 4.4e4, rel=1e-15
    )


def test_the_bound_is_scale_relative_which_is_the_whole_W4_P2b_RULING() -> None:
    """The same relative violation passes at one scale and fails at another.

    SPEC.md 5.2.4. ``lambda_min = -1.4e-12`` beside ``lambda_max = 4.4e4`` is a
    seventh of one ulp and is accepted; the identical *absolute* eigenvalue beside
    ``lambda_max = 1`` is 6.3 million ulp and is refused. The superseded absolute
    threshold could not tell those two matrices apart, which is why it refused a
    matrix that was PSD to a fraction of an epsilon.
    """
    assert_psd(np.diag([4.4e4, -1.4e-12]), "the tau = 42 case")
    with pytest.raises(PsdError, match="not positive semi-definite"):
        assert_psd(np.diag([1.0, -1.4e-12]), "the same absolute eigenvalue at scale 1")


def test_the_new_bound_is_stricter_than_the_one_it_replaced_at_unit_scale() -> None:
    """Not a widening. W4-P2b's first condition, as an assertion.

    ``-1e-12`` was the superseded absolute threshold. At ``lambda_max = 1`` the
    new bound is ``-8.9e-16``, about a thousand times tighter, and in correlation
    space (``lambda_max <= K``) it stays two orders inside it.
    """
    numerics = load().model.numerics
    superseded = -1.0e-12
    assert eigenvalue_floor(2, 1.0, numerics=numerics) > superseded
    assert eigenvalue_floor(6, 6.0, numerics=numerics) > superseded
    with pytest.raises(PsdError, match="not positive semi-definite"):
        assert_psd(np.diag([1.0, superseded / 2.0]), "would have passed before")


def test_a_negative_definite_matrix_gets_a_zero_bound_rather_than_a_permissive_one() -> None:
    """``max(lambda_max, 0)``: a negative scale must not buy a negative matrix room."""
    numerics = load().model.numerics
    assert eigenvalue_floor(2, -5.0, numerics=numerics) == 0.0
    with pytest.raises(PsdError, match="not positive semi-definite"):
        assert_psd(np.diag([-1.0, -2.0]), "negative definite")


def test_the_failure_message_names_the_stage() -> None:
    """Invariant 4 is per-stage precisely so a failure says which stage."""
    with pytest.raises(PsdError, match=r"eigenfactor \[short\]"):
        assert_psd(np.diag([1.0, -1.0]), "eigenfactor [short]")


def test_asymmetry_is_reported_as_asymmetry_not_as_non_psd() -> None:
    matrix = np.array([[1.0, 0.5], [0.4, 1.0]])
    with pytest.raises(PsdError, match="not symmetric"):
        assert_psd(matrix, "unit test")


def test_shape_failures() -> None:
    with pytest.raises(PsdError, match="2-D"):
        assert_psd(np.ones(3), "unit test")
    with pytest.raises(PsdError, match="square"):
        assert_psd(np.ones((2, 3)), "unit test")
    with pytest.raises(PsdError, match="3x3"):
        assert_psd(np.eye(2), "unit test", expected_size=3)
    with pytest.raises(PsdError, match="empty"):
        assert_psd(np.empty((0, 0)), "unit test")


def test_non_finite_entries_are_caught_before_the_eigendecomposition() -> None:
    matrix = np.eye(2)
    matrix[0, 0] = np.nan
    with pytest.raises(PsdError, match="non-finite"):
        assert_psd(matrix, "unit test")


def test_margin_is_positive_on_a_pass_and_recorded_with_its_threshold() -> None:
    check = assert_psd(np.eye(3), "unit test")
    assert check.eigenvalue_margin > 0.0
    assert check.minimum_eigenvalue_threshold == eigenvalue_floor(
        3, 1.0, numerics=load().model.numerics
    )
