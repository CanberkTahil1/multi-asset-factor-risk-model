"""SPEC.md 15.6's constrained WLS on a hand-built cross-section.

Every expected value below was worked by hand from the inputs before the code
ran (CLAUDE.md, definition of done). The design is chosen so the three
effective columns -- country, the industry contrast, the style -- are
W-orthogonal, which makes every coefficient a weighted projection a reader can
check on paper, and so the cap-weighted specific return is NOT zero: the
identity's last term is exercised, not assumed away.

Four names A, B, C, D; caps 4, 1, 1, 4, so sqrt-cap weights w = [2, 1, 1, 2]/6
and cap weights c = [4, 1, 1, 4]/10. Industries: A, B in I1, C, D in I2, so
w_1 = w_2 = 0.5 and the constraint reads f_2 = -f_1. One style
s = [1, -2, 2, -1] (cap-centred: c's = 0). With the contrast g = [1, 1, -1, -1]
the free design is [1, g, s], and sum w 1 g = sum w 1 s = sum w g s = 0, with
W-norms squared 1, 1, 2.

Returns, in units of 1e-4: r = [160, 40, 140, 55].
  f_c = sum w r           = (320 + 40 + 140 + 110) / 6 = 610 / 6 = 101.6667
  f_1 = sum w g r / 1     = (320 + 40 - 140 - 110) / 6 = 110 / 6 = 18.3333;  f_2 = -f_1
  f_s = sum w s r / 2     = (320 - 80 + 280 - 110) / 6 / 2 = 410 / 12 = 34.1667
  fitted = [154.1667, 51.6667, 151.6667, 49.1667]; u = [5.8333, -11.6667, -11.6667, 5.8333]
  sum w u = 0 (the country column absorbs the sqrt-cap mean)
  c'u = 0.4 * 5.8333 * 2 - 0.1 * 11.6667 * 2 = 4.6667 - 2.3333 = 2.3333  (NOT zero)
  c'r = 64 + 4 + 14 + 22 = 104 = f_c + c'u = 101.6667 + 2.3333            (the identity)
  sum w u^2 = (2 * 34.0278 + 136.1111 + 136.1111 + 2 * 34.0278) / 6 = 408.3333 / 6 = 68.0556
  sum w (r - 101.6667)^2 = (2 * 3402.78 + 3802.78 + 1469.44 + 2 * 2177.78) / 6 = 2738.89
  R^2_w = 1 - 68.0556 / 2738.89 = 0.975152
  sigma^2 = 68.0556 / (4 - 3) = 68.0556;  A = diag(1, 1, 2)
  t_c = 101.6667 / 8.2496 = 12.3238;  t_1 = 18.3333 / 8.2496 = 2.2223 = -t_2
  t_s = 34.1667 / (8.2496 / sqrt 2) = 34.1667 / 5.8333 = 5.8571
  cond(W^1/2 X R) = sqrt(2) / 1 (singular values 1, 1, sqrt 2);  cond(style block) = 1
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mafrm.factors import equity_regression as reg

CAP = np.array([4.0, 1.0, 1.0, 4.0])
INDUSTRY = np.array([1, 1, 2, 2])
STYLE = np.array([[1.0], [-2.0], [2.0], [-1.0]])
RETURNS = np.array([160.0, 40.0, 140.0, 55.0]) * 1e-4


def _day() -> reg.DayResult:
    return reg.regress_day(RETURNS, STYLE, INDUSTRY, CAP, weight_exponent=0.5, tolerance=1e-10)


def test_constraint_matrix_by_hand() -> None:
    """Weights [0.5, 0.3, 0.2], eliminate 0: f_0 = -(0.3/0.5) f~_1 - (0.2/0.5) f~_2."""
    r = reg.constraint_matrix(np.array([0.5, 0.3, 0.2]), eliminate=0)
    assert r.shape == (3, 2)
    np.testing.assert_allclose(r, [[-0.6, -0.4], [1.0, 0.0], [0.0, 1.0]])
    # Any f = R f~ satisfies the cap-weighted sum-to-zero exactly.
    f = r @ np.array([0.7, -1.1])
    assert abs(np.array([0.5, 0.3, 0.2]) @ f) < 1e-15


def test_factor_returns_by_hand() -> None:
    day = _day()
    assert day.country == pytest.approx(101.6667e-4, abs=1e-8)
    np.testing.assert_allclose(day.industry_returns, [18.3333e-4, -18.3333e-4], atol=1e-8)
    assert day.style_returns[0] == pytest.approx(34.1667e-4, abs=1e-8)
    np.testing.assert_allclose(
        day.residuals, np.array([5.8333, -11.6667, -11.6667, 5.8333]) * 1e-4, atol=1e-8
    )
    assert day.eliminated in (1, 2)  # equal industry shares: argmax picks the first, I1
    assert day.eliminated == 1
    assert day.members.tolist() == [2.0, 2.0] and day.present.tolist() == [1, 2]


def test_the_constraint_binds_and_the_identity_closes_with_a_nonzero_specific_term() -> None:
    day = _day()
    assert abs(day.industry_term) < 1e-15
    assert day.cap_weighted_return == pytest.approx(104.0e-4, abs=1e-12)
    assert day.style_term == pytest.approx(0.0, abs=1e-12)
    assert day.cap_weighted_specific == pytest.approx(2.3333e-4, abs=1e-8)
    assert day.identity_gap == pytest.approx(2.3333e-4, abs=1e-8)
    closing = day.cap_weighted_return - (
        day.country + day.industry_term + day.style_term + day.cap_weighted_specific
    )
    assert abs(closing) < 1e-15


def test_r_squared_t_statistics_and_condition_numbers_by_hand() -> None:
    day = _day()
    assert day.r2_weighted == pytest.approx(1 - (408.3333 / 6) / (16433.33 / 6), abs=2e-5)
    assert day.country_t == pytest.approx(12.3238, abs=2e-3)
    np.testing.assert_allclose(day.industry_t, [2.2223, -2.2223], atol=2e-3)
    assert day.style_t[0] == pytest.approx(5.8571, abs=2e-3)
    assert day.cond_full == pytest.approx(np.sqrt(2.0), abs=1e-9)
    assert day.cond_style == pytest.approx(1.0, abs=1e-12)
    assert day.columns == 3


def test_sqrt_cap_weights_are_neither_cap_nor_equal() -> None:
    """Failure mode 8. Under CAP weights the country return would be c'r = 104 exactly and
    c'u zero; under equal weights it would be the plain mean 98.75. sqrt-cap gives 101.67."""
    cap_w = reg.regress_day(RETURNS, STYLE, INDUSTRY, CAP, weight_exponent=1.0, tolerance=1e-10)
    assert cap_w.country == pytest.approx(104.0e-4, abs=1e-10)
    assert abs(cap_w.cap_weighted_specific) < 1e-12
    equal = reg.regress_day(RETURNS, STYLE, INDUSTRY, CAP, weight_exponent=0.0, tolerance=1e-10)
    assert equal.country == pytest.approx(98.75e-4, abs=1e-10)
    assert _day().country == pytest.approx(101.6667e-4, abs=1e-8)


def test_a_singleton_industry_has_a_zero_residual_and_the_largest_is_eliminated() -> None:
    """Ruling 3: D alone in I3 -> its residual is 0 by construction; I1 (cap 5 of 10) is
    eliminated, never the singleton."""
    industry = np.array([1, 1, 2, 3])
    styles = np.zeros((4, 0))
    day = reg.regress_day(RETURNS, styles, industry, CAP, weight_exponent=0.5, tolerance=1e-10)
    assert day.present.tolist() == [1, 2, 3]
    assert day.members.tolist() == [2.0, 1.0, 1.0]
    assert day.eliminated == 1
    assert abs(day.residuals[3]) < 1e-15 and abs(day.residuals[2]) < 1e-15
    assert abs(day.industry_term) < 1e-15


def test_too_few_names_and_bad_inputs_raise() -> None:
    with pytest.raises(reg.RegressionError, match="free parameters"):
        reg.regress_day(
            RETURNS[:3], STYLE[:3], INDUSTRY[:3], CAP[:3], weight_exponent=0.5, tolerance=1e-10
        )
    with pytest.raises(reg.RegressionError, match="positive cap"):
        reg.regress_day(
            RETURNS,
            STYLE,
            INDUSTRY,
            np.array([4.0, 0.0, 1.0, 4.0]),
            weight_exponent=0.5,
            tolerance=1e-10,
        )
    with pytest.raises(reg.RegressionError, match="non-finite"):
        reg.regress_day(
            np.array([np.nan, 1.0, 2.0, 3.0]),
            STYLE,
            INDUSTRY,
            CAP,
            weight_exponent=0.5,
            tolerance=1e-10,
        )


def test_industry_labels_from_the_parsed_file() -> None:
    table = pd.DataFrame(
        {"industry": [1, 1, 30], "abbrev": ["Agric", "Agric", "Oil"], "sic_low": [100, 200, 1300]}
    )
    assert reg.industry_labels(table) == {1: "Agric", 30: "Oil"}
    bad = pd.DataFrame({"industry": [1, 1], "abbrev": ["Agric", "Food"]})
    with pytest.raises(reg.RegressionError):
        reg.industry_labels(bad)
