"""The committed golden fixture. CLAUDE.md: *"always commit ... the synthetic
golden-file test fixture"*; SPEC.md 15.2 names it as the invariant-10 tripwire.

WHAT IS COMMITTED, AND WHY IT IS NOT THE ARRAY
-----------------------------------------------

``tests/fixtures/covariance_golden.json`` holds the fixture's digests, a
platform-independent summary of the panel, and the pipeline outputs it produces
-- not the 5040x40 panel itself. That panel is 4 MB of float text in a public
repository and committing it would buy nothing these do not.

BYTE-IDENTITY IS A PROPERTY OF A PLATFORM, NOT OF THIS CODE (W8-P4b)
---------------------------------------------------------------------

The panel is ``draws @ factorisation.T``. The draw is bit-identical everywhere
-- numpy guarantees ``default_rng``'s stream -- but the matmul and the Cholesky
run through whatever BLAS the wheel was built against, and a BLAS may
reassociate its sums. Measured, not assumed:

* **darwin-arm64** (Accelerate): one digest at both K, unchanged across OMP
  thread counts 1, 2, 4 and 8.
* **linux-x86_64** (GitHub Actions, OpenBLAS): a *different* digest from macOS
  at both K -- and at **K = 40 it is not stable run to run**, two digests
  observed across three runs on identical code and an identical ``uv.lock``.
  Suspect: kernel dispatch across heterogeneous runner CPUs. Named, not
  confirmed; there is no Linux runner here to bisect it on.

So this file asserts byte-identity **only where byte-identity has been observed
to hold**, and holds every platform to ``CROSS_PLATFORM_RTOL`` on
``panel_summary`` instead. That tolerance is what carries the tripwire: a
reassociated sum moves those statistics in the thirteenth significant figure, a
changed seed or generator moves them in the first. Pinning the anchor to one
platform and calling it reproducible would be the worse answer -- see SPEC.md 11.

Regenerate, and expect to justify the diff::

    python -m mafrm.risk.synthetic > tests/fixtures/covariance_golden.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mafrm.risk.config import HORIZONS, RiskConfig
from mafrm.risk.covariance import run_pipeline
from mafrm.risk.synthetic import (
    CROSS_PLATFORM_RTOL,
    GOLDEN_FACTOR_COUNTS,
    OBSERVATIONS,
    golden_regime_bias,
    panel,
    panel_summary,
    platform_key,
    specification,
)

_GOLDEN = Path(__file__).resolve().parent / "fixtures" / "covariance_golden.json"


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    assert _GOLDEN.exists(), f"the golden fixture is not committed: {_GOLDEN}"
    loaded: dict[str, Any] = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    return loaded


def test_the_golden_file_describes_the_fixture_this_code_builds(golden: dict[str, Any]) -> None:
    assert golden["observations"] == OBSERVATIONS
    assert tuple(golden["factor_counts"]) == GOLDEN_FACTOR_COUNTS
    assert golden["seed"] == specification(3).seed


@pytest.mark.parametrize("factors", GOLDEN_FACTOR_COUNTS)
def test_the_panel_reproduces_within_the_stated_tolerance(
    golden: dict[str, Any], factors: int
) -> None:
    """The tripwire, on every platform. See this module's docstring for why it is not a hash.

    A reassociated BLAS sum moves these statistics in the thirteenth significant
    figure; a changed seed, generator or specification moves them in the first.
    The tolerance sits between the two by four orders of magnitude and was
    chosen a priori rather than fitted to an observed disagreement.
    """
    values = panel(specification(factors)).to_numpy()
    case = golden["cases"][str(factors)]
    assert list(values.shape) == case["panel_shape"]

    measured = panel_summary(values)
    recorded = case["panel_summary"]
    assert set(measured) == set(recorded), "the summary's fields moved"
    for field, value in measured.items():
        assert value == pytest.approx(recorded[field], rel=CROSS_PLATFORM_RTOL), (
            f"the synthetic panel changed: {field} is {value!r} against a recorded "
            f"{recorded[field]!r}, outside {CROSS_PLATFORM_RTOL}. Either "
            "mafrm.risk.synthetic moved or numpy's bit stream did; both invalidate every "
            "other covariance test's expectations. Regenerate with "
            "`python -m mafrm.risk.synthetic` only after establishing which."
        )


@pytest.mark.parametrize("factors", GOLDEN_FACTOR_COUNTS)
def test_the_panel_is_byte_identical_WHERE_THAT_HOLDS(golden: dict[str, Any], factors: int) -> None:
    """Byte-identity, asserted per platform and only where it has been observed.

    An unrecorded platform is not a failure -- it is a platform nobody has run
    this on yet, and the tolerance test above is what covers it. A platform
    recorded as ``byte_identical: false`` is not asserted either, because the
    fixture records that it is *known* not to hold there; the entry names what
    was observed and the suspect for it.
    """
    values = panel(specification(factors)).to_numpy()
    digest = hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()
    recorded = golden["cases"][str(factors)]["panel_digests"].get(platform_key())

    if recorded is None:
        pytest.skip(f"no digest recorded for {platform_key()}; the tolerance test covers it")
    if not recorded["byte_identical"]:
        assert recorded["observed"], "a platform excused from byte-identity owes its evidence"
        pytest.skip(
            f"{platform_key()} is recorded as NOT byte-identical at K={factors}: "
            f"{recorded['observed']}"
        )
    assert digest in recorded["sha256"], (
        f"{platform_key()} is recorded as byte-identical at K={factors} and is not: "
        f"{digest} against {recorded['sha256']}. Either the panel changed -- in which case "
        "the tolerance test above has also failed and that is the one to read -- or this "
        "platform's byte-identity claim is wrong and the fixture entry must say so."
    )


@pytest.mark.parametrize("factors", GOLDEN_FACTOR_COUNTS)
def test_the_true_covariance_is_the_closed_form_the_golden_file_records(
    golden: dict[str, Any], factors: int
) -> None:
    """The truth involves no draw at all, so these are exact to roundoff."""
    spec = specification(factors)
    case = golden["cases"][str(factors)]
    eigenvalues = np.linalg.eigvalsh(spec.covariance)
    assert float(np.trace(spec.covariance)) == pytest.approx(case["true_trace"], rel=1e-12)
    assert float(eigenvalues[0]) == pytest.approx(case["true_minimum_eigenvalue"], rel=1e-10)
    assert float(eigenvalues[-1]) == pytest.approx(case["true_maximum_eigenvalue"], rel=1e-10)


@pytest.mark.parametrize("factors", GOLDEN_FACTOR_COUNTS)
@pytest.mark.parametrize("horizon", HORIZONS)
def test_the_pipeline_reproduces_its_recorded_outputs(
    golden: dict[str, Any], factors: int, horizon: str
) -> None:
    """SPEC.md 15.2's shape-and-PSD assertion, plus the numbers behind it.

    A failure here is an ARCHITECTURE warning before it is a numerical one --
    SPEC.md 15.2 says so in as many words -- because the same code runs at K=3
    and K=40 and nothing in it may branch on which.
    """
    frame = panel(specification(factors))
    base = RiskConfig.load(horizon=horizon)  # type: ignore[arg-type]
    # Stages 1-3 are pinned on a build STOPPED at the repair, which is the
    # matrix those recorded numbers describe; the eigenfactor stage's output is
    # per-scaling and is pinned below.
    build = run_pipeline(frame, base, stop_after="psd_repair")
    recorded = golden["cases"][str(factors)]["horizons"][horizon]
    eigenvalues = np.linalg.eigvalsh(build.matrix)

    assert build.matrix.shape == (factors, factors)
    assert float(np.trace(build.matrix)) == pytest.approx(recorded["trace"], rel=1e-10)
    assert float(eigenvalues[0]) == pytest.approx(recorded["minimum_eigenvalue"], rel=1e-9)
    assert float(eigenvalues[-1]) == pytest.approx(recorded["maximum_eigenvalue"], rel=1e-9)
    assert build.k_over_t_eff == pytest.approx(recorded["k_over_t_eff"], rel=1e-12)
    assert build.shepard_multiplier == pytest.approx(recorded["shepard_multiplier"], rel=1e-12)

    # Stage 2 and stage 3, pinned from W3-P2. The fixture is serially
    # UNCORRELATED by construction, so Newey-West has no serial correlation to
    # correct here and the recorded result -- the repair never fires at either K
    # -- is the expected one rather than a lucky one.
    newey_west = next(stage for stage in build.stages if stage.name == "newey_west")
    assert newey_west.minimum_eigenvalue == pytest.approx(
        recorded["newey_west_minimum_eigenvalue"], rel=1e-9
    )
    assert build.repair_fired == recorded["repair_fired"]
    assert not build.repair_fired

    # SPEC.md 5.3, W3-P3. BOTH published scalings are pinned, from one Monte
    # Carlo each, so a session that quietly ships only one breaks this file.
    for scaling in base.eigenfactor_scaling:
        adjusted = run_pipeline(frame, base.for_scaling(scaling), stop_after="eigenfactor")
        pinned = recorded["eigenfactor_variants"][f"{scaling:.1f}"]
        spectrum = np.linalg.eigvalsh(adjusted.matrix)
        assert float(np.trace(adjusted.matrix)) == pytest.approx(pinned["trace"], rel=1e-9)
        assert float(spectrum[0]) == pytest.approx(pinned["minimum_eigenvalue"], rel=1e-8)
        assert float(spectrum[-1]) == pytest.approx(pinned["maximum_eigenvalue"], rel=1e-8)
        assert adjusted.eigenfactor is not None
        assert adjusted.eigenfactor.halflife == recorded["eigenfactor_halflife"]
        assert adjusted.eigenfactor.observations == recorded["eigenfactor_observations"]
        np.testing.assert_allclose(
            adjusted.eigenfactor.bias, recorded["eigenfactor_bias"], rtol=1e-9
        )


@pytest.mark.parametrize("factors", GOLDEN_FACTOR_COUNTS)
@pytest.mark.parametrize("horizon", HORIZONS)
def test_the_regime_stage_is_pinned_and_is_a_pure_scalar(
    golden: dict[str, Any], factors: int, horizon: str
) -> None:
    """SPEC.md 5.4, W3-P4. The fixture now pins the pipeline to its END.

    Three claims, and the third is the one worth a golden file. ``lambda_F^2``
    reproduces from ``model.seed``; the five-stage trace reproduces; and the two
    are consistent with the stage being a **pure positive scalar multiplication**
    -- ``trace(F5) == lambda_F^2 * trace(F4)`` exactly. A stage that had started
    doing anything else to the matrix would break the third assertion while
    passing the first two.
    """
    frame = panel(specification(factors))
    base = RiskConfig.load(horizon=horizon)  # type: ignore[arg-type]
    recorded = golden["cases"][str(factors)]["horizons"][horizon]
    bias = golden_regime_bias(specification(factors))
    assert bias.size == recorded["regime_bias_dates"]

    scaling = base.eigenfactor_scaling[0]
    through_four = run_pipeline(frame, base.for_scaling(scaling), stop_after="eigenfactor")
    build = run_pipeline(frame, base.for_scaling(scaling), regime_bias=bias)

    assert build.complete
    assert build.regime is not None
    assert build.regime.lambda_squared == pytest.approx(
        recorded["regime_lambda_squared"], rel=1e-12
    )
    assert float(np.trace(build.matrix)) == pytest.approx(recorded["regime_trace"], rel=1e-9)
    np.testing.assert_allclose(
        build.matrix, recorded["regime_lambda_squared"] * through_four.matrix, rtol=1e-12
    )


def test_the_lambda_amplitude_is_wider_at_k40_than_at_k3(golden: dict[str, Any]) -> None:
    """The in-fixture form of SPEC.md 5.3.2's pre-registered W7 prediction.

    The amplitude of the eigenfactor bias curve scales with ``K/T``. This is the
    comparison W7 repeats at ``K = 56`` against the production model's ``K = 6``,
    run here on the one fixture where both sizes exist and nothing else differs.
    The residual degrees of freedom are pinned alongside it, because at ``K = 3``
    the parabola is an exact interpolation and smooths nothing at all.
    """
    small = golden["cases"]["3"]["horizons"]["short"]
    large = golden["cases"]["40"]["horizons"]["short"]
    assert large["eigenfactor_amplitude"] > 10.0 * small["eigenfactor_amplitude"]
    assert small["eigenfactor_residual_degrees_of_freedom"] == 0
    assert large["eigenfactor_residual_degrees_of_freedom"] == 37


def test_the_two_k_cases_sweep_the_ratio_spec_152_is_about(golden: dict[str, Any]) -> None:
    """The fixture is not two sizes for the sake of it: K/T_eff has to move.

    SPEC.md 15.2's argument is that half-life selection is nearly irrelevant in a
    small model and decisive in a large one. A fixture whose two cases sat at the
    same K/T_eff could not exercise that at all.
    """
    small = golden["cases"]["3"]["horizons"]["short"]
    large = golden["cases"]["40"]["horizons"]["short"]
    assert large["k_over_t_eff"] > 10.0 * small["k_over_t_eff"]
    assert large["shepard_multiplier"] > 1.4
    assert small["shepard_multiplier"] < 1.05
