"""Horizon scaling. SPEC.md 5.2's ``x21``, and the W3-P2 ruling that moved it out.

The multiplier is applied **once, downstream of the whole pipeline**, by one
named function. Three things are pinned here:

* the arithmetic, by hand;
* that it cannot be applied twice -- the failure-mode-7 case, where a doubled
  multiplier leaves every downstream check (PSD, symmetry, conditioning) passing;
* that no stage of the pipeline can apply it, because nothing the pipeline can
  reach knows the number.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mafrm.config import load
from mafrm.risk.config import HORIZONS, RiskConfig
from mafrm.risk.covariance import run_pipeline
from mafrm.risk.horizon import HorizonCovariance, HorizonError, scale_to_horizon

_DAILY = np.array([[4.0, 1.0], [1.0, 9.0]])


def test_scale_to_horizon_hand_computed() -> None:
    """HAND-COMPUTED. 21 * [[4, 1], [1, 9]] = [[84, 21], [21, 189]].

    Linear in the covariance, which is where the square-root-of-time rule is
    linear: the volatility 2.0 becomes ``sqrt(84) = 9.165`` = ``2 * sqrt(21)``.
    """
    scaled = scale_to_horizon(_DAILY, trading_days=21)
    np.testing.assert_allclose(scaled.matrix, [[84.0, 21.0], [21.0, 189.0]], rtol=0, atol=0)
    assert scaled.trading_days == 21
    assert np.sqrt(scaled.matrix[0, 0]) == pytest.approx(2.0 * np.sqrt(21.0))


def test_the_multiplier_is_the_config_constant_and_is_252_over_12() -> None:
    """CLAUDE.md invariant 6. The number is in the config, and the two agree."""
    data = load().model.data
    assert data.trading_days_per_month == 21
    assert data.trading_days_per_month == data.trading_days_per_year // 12


def test_it_cannot_be_applied_twice() -> None:
    """The W3-P2 ruling's acceptance test. CLAUDE.md failure mode 7.

    A matrix scaled twice is 441x too large and still passes every check the
    pipeline runs -- it is PSD, symmetric and identically conditioned. The type
    is what catches it: :func:`scale_to_horizon` takes an array and returns a
    :class:`HorizonCovariance`, so feeding its own output back is a mypy error on
    ``src/`` and this error at runtime.
    """
    once = scale_to_horizon(_DAILY, trading_days=21)
    with pytest.raises(HorizonError, match="already been scaled"):
        scale_to_horizon(once, trading_days=21)  # type: ignore[arg-type]


def test_a_doubled_multiplier_would_pass_every_other_check() -> None:
    """Why the guard above is a type and not an assertion on the values.

    This is the failure the ruling names: nothing about a 441x matrix looks
    wrong. Asserting it here keeps the reason for the guard visible next to the
    guard.
    """
    once = scale_to_horizon(_DAILY, trading_days=21).matrix
    twice = scale_to_horizon(once, trading_days=21).matrix
    assert np.allclose(twice, 441.0 * _DAILY)
    assert np.linalg.eigvalsh(twice)[0] > 0.0
    assert np.allclose(twice, twice.T)
    assert np.linalg.cond(twice) == pytest.approx(np.linalg.cond(_DAILY))


def test_scaling_leaves_the_eigenvectors_and_the_eigenvalue_ratios_unchanged() -> None:
    """Why deferring the multiplier past SPEC.md 5.3 costs nothing.

    ``c * F`` has eigenvalues ``c * lambda_k`` and the same eigenvectors, so a
    uniform scalar changes neither which direction is the smallest eigenfactor
    nor the ``lambda_tilde / lambda`` ratios the eigenfactor adjustment reads.
    This is exactly what is *not* true of SPEC.md 5.3.1's per-factor rescaling,
    which is why the two live in separately named functions.
    """
    values, vectors = np.linalg.eigh(_DAILY)
    scaled_values, scaled_vectors = np.linalg.eigh(scale_to_horizon(_DAILY, trading_days=21).matrix)

    np.testing.assert_allclose(np.abs(scaled_vectors), np.abs(vectors), rtol=1e-12)
    np.testing.assert_allclose(scaled_values, 21.0 * values, rtol=1e-12)
    np.testing.assert_allclose(scaled_values / scaled_values[-1], values / values[-1], rtol=1e-12)


def test_the_pipeline_cannot_reach_the_multiplier() -> None:
    """The mechanism, not the intention. ``RiskConfig`` does not carry it.

    A stage takes ``(matrix, returns, RiskConfig)`` and nothing else, so if no
    field of ``RiskConfig`` names a horizon in trading days, no stage can scale
    to one even by accident.
    """
    risk = RiskConfig.load(horizon=HORIZONS[0])
    fields = vars(risk)
    assert not [name for name in fields if "trading_day" in name or "horizon_days" in name]
    assert 21 not in [value for value in fields.values() if isinstance(value, int)]


def test_no_module_in_risk_applies_a_horizon_multiplier_except_this_one() -> None:
    """Static scan for the literal multiplication, plus the import edge.

    Prose may name the constant -- the pipeline's own modules explain why they do
    not apply it, and that explanation is worth having. What may not appear is a
    module in ``risk`` multiplying by a horizon, or importing the function that
    does. The import edge is the one that would actually be crossed: a stage that
    wanted a monthly matrix would reach for ``scale_to_horizon`` rather than
    write ``21`` itself.
    """
    risk = Path(__file__).resolve().parents[1] / "src" / "mafrm" / "risk"
    multiplication = re.compile(r"(?<![_.\w])21\b\s*\*|\*\s*(?<![_.\w])21\b")
    imports_horizon = re.compile(
        r"^\s*(?:from\s+mafrm\.risk\.horizon|import\s+mafrm\.risk\.horizon|from\s+mafrm\.risk\s+import\s+.*horizon)"
    )
    offenders = []
    for path in sorted(risk.rglob("*.py")):
        if path.name in {"horizon.py", "__init__.py"}:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if multiplication.search(line) or imports_horizon.search(line):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, (
        "a horizon multiplier has appeared inside the pipeline. SPEC.md 5.2's x21 belongs in "
        "mafrm.risk.horizon.scale_to_horizon and nowhere else:\n" + "\n".join(offenders)
    )


def test_the_pipeline_output_is_daily_and_scaling_it_is_the_caller_s_step() -> None:
    """End to end: the pipeline returns daily units; the caller converts."""
    rng = np.random.default_rng(load().model.seed)
    frame = pd.DataFrame(rng.standard_normal((300, 3)), columns=["f0", "f1", "f2"])
    # All five stages. SPEC.md 5.4's stage is handed a perfectly calibrated bias
    # history (B = 1 everywhere, so lambda_F^2 = 1), which keeps this a statement
    # about UNITS and not about the regime multiplier's size.
    daily = run_pipeline(
        frame,
        RiskConfig.load(horizon=HORIZONS[0]).for_scaling(1.0),
        regime_bias=np.ones(300),
    ).matrix

    # Unit variance daily input -> daily variances of order 1, not of order 21.
    assert np.all(np.diag(daily) < 5.0)
    monthly = scale_to_horizon(daily, trading_days=load().model.data.trading_days_per_month)
    np.testing.assert_allclose(monthly.matrix, 21.0 * daily, rtol=1e-14)
    assert isinstance(monthly, HorizonCovariance)


@pytest.mark.parametrize("bad", [0, -1])
def test_a_non_positive_horizon_is_refused(bad: int) -> None:
    with pytest.raises(HorizonError, match="at least 1"):
        scale_to_horizon(_DAILY, trading_days=bad)


def test_a_non_square_matrix_is_refused() -> None:
    with pytest.raises(HorizonError, match="square"):
        scale_to_horizon(np.ones((2, 3)), trading_days=21)
