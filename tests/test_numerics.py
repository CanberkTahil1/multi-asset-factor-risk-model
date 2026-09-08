"""EWMA weighting and effective sample size. mafrm.numerics.

Hand-computed throughout. At a half-life of one period the weights are exactly
``..., 1/4, 1/2, 1`` before normalisation, which makes every expectation here a
fraction that can be checked by eye rather than a number produced by the code
under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mafrm.config import load
from mafrm.numerics import (
    LN2,
    effective_sample_size,
    ewma_weights,
    realised_effective_sample_size,
)


def test_weights_at_unit_halflife_are_powers_of_one_half() -> None:
    """HAND-COMPUTED. halflife=1, n=3 -> raw [1/4, 1/2, 1], normalised [1/7, 2/7, 4/7]."""
    raw = ewma_weights(3, 1.0, normalize=False)
    np.testing.assert_allclose(raw, [0.25, 0.5, 1.0], rtol=0, atol=1e-15)

    normalised = ewma_weights(3, 1.0)
    np.testing.assert_allclose(normalised, [1 / 7, 2 / 7, 4 / 7], rtol=0, atol=1e-15)
    assert normalised.sum() == pytest.approx(1.0, abs=1e-15)


def test_weights_are_oldest_first_and_most_recent_is_largest() -> None:
    weights = ewma_weights(10, 5.0)
    assert weights[-1] == weights.max()
    assert np.all(np.diff(weights) > 0.0)


def test_an_observation_one_halflife_older_carries_half_the_weight() -> None:
    weights = ewma_weights(101, 20.0)
    assert weights[-21] / weights[-1] == pytest.approx(0.5, rel=1e-12)


def test_single_observation_normalises_to_one() -> None:
    np.testing.assert_allclose(ewma_weights(1, 84.0), [1.0], rtol=0, atol=1e-15)


@pytest.mark.parametrize("bad", [0, -1])
def test_weights_reject_an_empty_window(bad: int) -> None:
    with pytest.raises(ValueError, match="n_obs"):
        ewma_weights(bad, 84.0)


@pytest.mark.parametrize("bad", [0.0, -84.0, float("nan")])
def test_weights_reject_a_non_positive_halflife(bad: float) -> None:
    with pytest.raises(ValueError, match="halflife"):
        ewma_weights(10, bad)


def test_effective_sample_size_matches_the_config_reference_table() -> None:
    """The config carries 84 -> 242, 252 -> 727, 504 -> 1454 for exactly this check.

    CLAUDE.md's parameter table states the closed form and the three values it
    produces. The reference table exists so the closed form in code can be
    checked against hand-computed values rather than against itself.
    """
    reference = load().model.numerics.effective_sample_size.reference
    assert reference, "model.numerics.effective_sample_size.reference is empty"
    for halflife, expected in reference.items():
        assert math.floor(effective_sample_size(float(halflife))) == expected


def test_effective_sample_size_is_the_stated_closed_form() -> None:
    """HAND-COMPUTED. 2 * 84 / ln 2 = 242.3728..."""
    assert effective_sample_size(84.0) == pytest.approx(2 * 84 / LN2, rel=1e-15)
    assert effective_sample_size(84.0) == pytest.approx(242.3728, abs=1e-4)


def test_realised_effective_sample_size_of_equal_weights_is_the_count() -> None:
    """HAND-COMPUTED. Kish on n equal weights: (n*w)^2 / (n*w^2) = n."""
    weights = np.full(37, 1.0 / 37.0)
    assert realised_effective_sample_size(weights) == pytest.approx(37.0, rel=1e-12)


def test_realised_effective_sample_size_hand_computed_at_unit_halflife() -> None:
    """HAND-COMPUTED. w = [1/7, 2/7, 4/7]: 1 / (1+4+16)/49 = 49/21 = 7/3."""
    weights = ewma_weights(3, 1.0)
    assert realised_effective_sample_size(weights) == pytest.approx(7.0 / 3.0, rel=1e-12)


def test_realised_effective_sample_size_approaches_the_asymptote_on_a_long_window() -> None:
    """A window long relative to the half-life realises 2*tau/ln2; a short one does not."""
    long_window = realised_effective_sample_size(ewma_weights(5040, 84.0))
    assert long_window == pytest.approx(effective_sample_size(84.0), rel=1e-3)

    short_window = realised_effective_sample_size(ewma_weights(100, 84.0))
    assert short_window < effective_sample_size(84.0)
    assert short_window < 100.0


def test_realised_effective_sample_size_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        realised_effective_sample_size(np.ones((2, 2)))
    with pytest.raises(ValueError, match="non-negative"):
        realised_effective_sample_size(np.array([1.0, -1.0]))
    with pytest.raises(ValueError, match="sum to zero"):
        realised_effective_sample_size(np.zeros(3))
