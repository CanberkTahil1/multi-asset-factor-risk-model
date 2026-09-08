"""The synthetic golden fixture the covariance pipeline is validated against.

SPEC.md 15.2 names the test that guards CLAUDE.md invariant 10:

    *"Run the full covariance pipeline on the committed synthetic golden fixture
    with K=3 and K=40 and assert both produce PSD matrices of the right shape.
    That test failing is your early warning."*

This module generates that fixture. Nothing here knows what a column is; the
columns are ``f00``, ``f01``, ... precisely so that no test written against them
can accidentally encode an assumption the week-7 module would then violate.

THE COVARIANCE IS KNOWN, AND IT IS NOT DIAGONAL
-----------------------------------------------

An independent generator would be useless here. If every column were drawn
independently the true correlation matrix would be the identity, and a bug in
the correlation half of SPEC.md 5.1 -- the half that exists *because* USE4
estimates correlations separately -- would hide behind correct variances and the
K=40 case would be unable to detect the failure it was written to catch.

So the truth is a two-factor structure::

    Sigma = B B' + diag(psi)

with ``B`` a ``K x 2`` loading matrix and ``psi`` a strictly positive
idiosyncratic variance. All three of the properties the fixture needs follow
from closed forms rather than from a draw, so ``Sigma`` is auditable by reading
this file:

``b1_k = 0.30 + 0.70 * k/(K-1)``
    A first loading rising across the columns, which makes the **variances
    heterogeneous** -- roughly a 4x spread at K=3 and wider at K=40. A fixture
    with equal variances cannot tell a covariance from a correlation.

``b2_k = 0.25 * (-1)^k``
    A sign-alternating second loading, which makes the **correlations
    heterogeneous in magnitude** and pushes some pairs well away from others.
    Without it every off-diagonal would be a monotone function of one index and
    a transposition bug would be invisible.

``psi_k = 0.20 + 0.06 * ((7k) mod 11)``
    A non-monotone idiosyncratic variance in ``[0.20, 0.80]``. Non-monotone so
    that the variance ordering is not the column ordering; bounded away from
    zero so that ``lambda_min(Sigma) >= 0.20`` and the fixture is **well
    conditioned by construction** at every K.

T = 20 * 252 = 5040
-------------------

Two requirements set it, both from SPEC.md 15.2's own test:

* The longest half-life the pipeline runs is 504 days, whose asymptotic
  ``T_eff = 2*504/ln 2 = 1454``. At T = 5040 the window is 3.5x that, so the
  EWMA weights have room to decay and the fixture exercises the estimator
  rather than an equally-weighted average of a truncated window.
* The K=40 case is compared against a sample covariance, which has to be well
  conditioned itself for the comparison to mean anything. T/K = 126.

``20 * 252`` is twenty trading years at ``data.trading_days_per_year``, which is
where the number comes from; it is not a round figure chosen to look like one.

FIXTURE CONSTANTS, NOT MODEL PARAMETERS
---------------------------------------

``T``, the loading and idiosyncratic closed forms, and the two K values are
**held in code on purpose**. They are properties of a test fixture, not
tunables: putting them in ``config/model.yaml`` would imply that a later session
could adjust them, and a fixture that can be adjusted to make a test pass is not
a fixture. They are logged in ``experiments.md`` under the constants-held-in-code
section for W3-P1, which is where the project records exactly this category.

The draw is keyed to ``model.seed`` through ``numpy.random.default_rng``, never
a global seed (CLAUDE.md, code conventions), so the panel is byte-reproducible.
``tests/fixtures/covariance_golden.json`` records its SHA-256 and the pipeline
outputs it produces, so drift in the generator or in numpy's bit stream fails
loudly instead of quietly changing what the tests test.
"""

from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from mafrm.config import Config, load
from mafrm.risk.config import HORIZONS, RiskConfig

__all__ = [
    "CROSS_PLATFORM_RTOL",
    "GOLDEN_FACTOR_COUNTS",
    "GOLDEN_REGIME_DATES",
    "OBSERVATIONS",
    "SyntheticSpec",
    "golden_record",
    "golden_regime_bias",
    "panel",
    "panel_summary",
    "platform_key",
    "specification",
]

#: Relative tolerance for comparing the panel ACROSS platforms (W8-P4b ruling).
#:
#: The panel is ``draws @ factorisation.T``: the draw is bit-identical
#: everywhere -- numpy guarantees ``default_rng``'s stream across platforms --
#: but the matmul and the Cholesky go through whatever BLAS the wheel was built
#: against, and a BLAS is free to reassociate its sums. Byte-identity is
#: therefore a property of a platform, not of this code, and is asserted only
#: where it has been observed to hold. This tolerance is what is asserted
#: everywhere instead. It is chosen a priori, not fitted: ~1e-9 sits four orders
#: above the ~1e-13 relative accumulation a 2e5-element double-precision
#: reduction can produce, and many orders below any change to the draw itself,
#: which moves these statistics in the first significant figure.
CROSS_PLATFORM_RTOL: Final[float] = 1e-9

#: SPEC.md 15.2 names both: a small model and a large one, on the same code.
GOLDEN_FACTOR_COUNTS: Final[tuple[int, ...]] = (3, 40)

#: Twenty trading years at ``data.trading_days_per_year``. See the module
#: docstring for the two requirements that set it.
OBSERVATIONS: Final[int] = 20 * 252

#: How many forecast dates the golden file's SPEC.md 5.4 bias history carries.
#: A **fixture dimension**, like :data:`OBSERVATIONS`, not a model parameter:
#: nothing in the pipeline reads it and no result depends on the value. Two
#: trading years is long against the 168-day VRA half-life, so ``lambda_F^2``
#: is determined by the series rather than by where it happens to start.
GOLDEN_REGIME_DATES: Final[int] = 2 * 252

_BASE_LOADING: Final[float] = 0.30
_LOADING_RANGE: Final[float] = 0.70
_SECOND_LOADING: Final[float] = 0.25
_BASE_IDIOSYNCRATIC: Final[float] = 0.20
_IDIOSYNCRATIC_STEP: Final[float] = 0.06
_IDIOSYNCRATIC_STRIDE: Final[int] = 7
_IDIOSYNCRATIC_MODULUS: Final[int] = 11

#: Keeps :func:`golden_regime_bias` off the panel's own random stream. Any
#: non-zero offset does; nothing depends on the value.
_REGIME_SEED_OFFSET: Final[int] = 1


@dataclass(frozen=True)
class SyntheticSpec:
    """A fixture panel's exact generating parameters. Everything is closed form."""

    factors: int
    observations: int
    seed: int
    #: ``K x 2``.
    loadings: np.ndarray
    #: ``K``, strictly positive.
    idiosyncratic: np.ndarray

    @property
    def covariance(self) -> np.ndarray:
        """The true ``Sigma = B B' + diag(psi)``. Non-diagonal, well conditioned."""
        matrix: np.ndarray = self.loadings @ self.loadings.T + np.diag(self.idiosyncratic)
        symmetric: np.ndarray = (matrix + matrix.T) / 2.0
        return symmetric

    @property
    def correlation(self) -> np.ndarray:
        """The true correlation implied by :attr:`covariance`."""
        deviation = np.sqrt(np.diag(self.covariance))
        correlation: np.ndarray = self.covariance / np.outer(deviation, deviation)
        np.fill_diagonal(correlation, 1.0)
        return correlation

    @property
    def columns(self) -> tuple[str, ...]:
        """``f00``, ``f01``, ... Deliberately meaningless."""
        return tuple(f"f{index:02d}" for index in range(self.factors))


def specification(
    factors: int,
    *,
    observations: int = OBSERVATIONS,
    seed: int | None = None,
    config: Config | None = None,
) -> SyntheticSpec:
    """The closed-form generating parameters for a ``K``-column fixture."""
    if factors < 1:
        raise ValueError(f"specification: factors must be at least 1, got {factors}")
    if observations < 1:
        raise ValueError(f"specification: observations must be at least 1, got {observations}")
    index = np.arange(factors, dtype=float)
    span = float(factors - 1) if factors > 1 else 1.0
    first = _BASE_LOADING + _LOADING_RANGE * index / span
    second = _SECOND_LOADING * np.where(np.arange(factors) % 2 == 0, 1.0, -1.0)
    idiosyncratic = _BASE_IDIOSYNCRATIC + _IDIOSYNCRATIC_STEP * (
        (_IDIOSYNCRATIC_STRIDE * np.arange(factors)) % _IDIOSYNCRATIC_MODULUS
    )
    return SyntheticSpec(
        factors=factors,
        observations=observations,
        seed=(config or load()).model.seed if seed is None else seed,
        loadings=np.column_stack([first, second]),
        idiosyncratic=idiosyncratic.astype(float),
    )


def panel(spec: SyntheticSpec) -> pd.DataFrame:
    """Draw the ``T x K`` fixture panel. Byte-reproducible from ``spec.seed``.

    Rows are iid mean-zero Gaussian with covariance :attr:`SyntheticSpec.covariance`.
    Mean zero in population is not incidental: SPEC.md 5.1.1 estimates moments
    about zero, so a fixture with a non-zero true mean would make the estimator
    look biased against a truth it was never forecasting.
    """
    rng = np.random.default_rng(spec.seed)
    factorisation = np.linalg.cholesky(spec.covariance)
    draws = rng.standard_normal((spec.observations, spec.factors))
    values = draws @ factorisation.T
    index = pd.RangeIndex(spec.observations, name="period")
    return pd.DataFrame(values, index=index, columns=list(spec.columns))


def _digest(values: np.ndarray) -> str:
    """SHA-256 of the panel's IEEE-754 bytes, little-endian."""
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


def platform_key() -> str:
    """The key a panel digest is recorded under. See :data:`CROSS_PLATFORM_RTOL`."""
    return f"{platform.system()}-{platform.machine()}".lower()


def panel_summary(values: np.ndarray) -> dict[str, float]:
    """The platform-INDEPENDENT comparand for the panel (W8-P4b ruling).

    Four statistics over every element, each far from zero so that a relative
    tolerance means something, and each moved in its first significant figure by
    any change to the draw. This is what carries the golden file's tripwire
    where the digest cannot: a reassociated BLAS sum perturbs these in the
    thirteenth figure, a changed seed or generator perturbs them in the first.

    ``sum`` is deliberately absent: the panel is mean-zero, so its sum is a
    near-cancellation whose relative error is meaningless.
    """
    return {
        "absolute_sum": float(np.abs(values).sum()),
        "sum_of_squares": float(np.square(values).sum()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def golden_regime_bias(spec: SyntheticSpec) -> np.ndarray:
    """A deterministic ``B_t^F`` history for the golden file. SPEC.md 5.4.

    The bias statistic of a **perfectly calibrated** model: ``K`` standard
    normals per date, standardised by a forecast of exactly one. So
    ``lambda_F^2`` sits near one without being one, which is what makes it worth
    pinning -- an all-ones history would give ``lambda_F^2 = 1`` exactly and
    would pin nothing at all.

    It is a draw rather than a closed form, and therefore reproducible only
    through ``model.seed``; the offset keeps it off the panel's own stream so a
    change to one cannot silently move the other.
    """
    from mafrm.risk.regime import cross_sectional_bias  # local: keeps the graph acyclic

    generator = np.random.default_rng(spec.seed + _REGIME_SEED_OFFSET)
    draw = generator.standard_normal((GOLDEN_REGIME_DATES, spec.factors))
    return cross_sectional_bias(draw, np.ones_like(draw))


def golden_record(config: Config | None = None) -> dict[str, Any]:
    """The committed golden file's contents, regenerated.

    Written to ``tests/fixtures/covariance_golden.json`` and compared there by
    ``tests/test_covariance_golden.py``. Regenerate with::

        python -m mafrm.risk.synthetic > tests/fixtures/covariance_golden.json

    and expect to justify any diff: a change here means either the generator
    moved or numpy's bit stream did, and both change what every other
    covariance test is testing.
    """
    from mafrm.risk.covariance import run_pipeline  # local: keeps the import graph acyclic

    settings = config or load()
    record: dict[str, Any] = {
        "_what": (
            "Golden values for the SPEC.md 15.2 synthetic fixture. Regenerate with "
            "`python -m mafrm.risk.synthetic`. A diff here means the generator or "
            "numpy's bit stream moved; both change what every covariance test tests."
        ),
        "seed": settings.model.seed,
        "observations": OBSERVATIONS,
        "factor_counts": list(GOLDEN_FACTOR_COUNTS),
        "cases": {},
    }
    for factors in GOLDEN_FACTOR_COUNTS:
        spec = specification(factors, config=settings)
        frame = panel(spec)
        regime_bias = golden_regime_bias(spec)
        true_eigenvalues = np.linalg.eigvalsh(spec.covariance)
        values = frame.to_numpy()
        case: dict[str, Any] = {
            # Byte-identity is per platform (W8-P4b); this run can only speak
            # for the one it is on. `_main` merges observations already in the
            # committed fixture so regenerating here never erases them.
            "panel_digests": {
                platform_key(): {
                    "byte_identical": True,
                    "sha256": [_digest(values)],
                    "observed": "regenerated by `python -m mafrm.risk.synthetic`",
                }
            },
            "panel_summary": panel_summary(values),
            "panel_shape": list(frame.shape),
            "true_trace": float(np.trace(spec.covariance)),
            "true_minimum_eigenvalue": float(true_eigenvalues[0]),
            "true_maximum_eigenvalue": float(true_eigenvalues[-1]),
            "horizons": {},
        }
        for horizon in HORIZONS:
            base = RiskConfig.load(horizon=horizon, config=settings)
            # BOTH published scalings, from ONE pipeline run each. SPEC.md 5.3
            # says run both and the golden file pins both, so a later session
            # cannot quietly ship one of them.
            variants: dict[str, Any] = {}
            for scaling in base.eigenfactor_scaling:
                variant_build = run_pipeline(
                    frame, base.for_scaling(scaling), stop_after="eigenfactor"
                )
                variant_eigenvalues = np.linalg.eigvalsh(variant_build.matrix)
                variants[f"{scaling:.1f}"] = {
                    "trace": float(np.trace(variant_build.matrix)),
                    "minimum_eigenvalue": float(variant_eigenvalues[0]),
                    "maximum_eigenvalue": float(variant_eigenvalues[-1]),
                    "frobenius_error_vs_truth": float(
                        np.linalg.norm(variant_build.matrix - spec.covariance, ord="fro")
                    ),
                }
            adjustment = variant_build.eigenfactor
            assert adjustment is not None
            # SPEC.md 5.4, W3-P4, so the fixture pins the pipeline to its END
            # rather than to its fourth stage. Recorded once per horizon and NOT
            # per scaling: the bias history is supplied by the caller, so
            # `lambda_F^2` cannot depend on the eigenfactor variant, and putting
            # it inside `eigenfactor_variants` would imply a dependence that does
            # not exist. It DOES depend on the horizon, through the 42d/168d VRA
            # half-life. The stage-5 trace is recorded beside it because a
            # multiplier that could only be checked by dividing two other
            # recorded numbers is not really pinned.
            regime_build = run_pipeline(
                frame, base.for_scaling(base.eigenfactor_scaling[0]), regime_bias=regime_bias
            )
            assert regime_build.regime is not None
            build = run_pipeline(frame, base, stop_after="psd_repair")
            eigenvalues = np.linalg.eigvalsh(build.matrix)
            newey_west = next(stage for stage in build.stages if stage.name == "newey_west")
            case["horizons"][horizon] = {
                # SPEC.md 5.3, recorded from W3-P3. The raw curve rather than the
                # fit, because the fit is a deterministic function of it and the
                # raw curve is what the Monte Carlo actually produced.
                "eigenfactor_halflife": adjustment.halflife,
                "eigenfactor_observations": adjustment.observations,
                "eigenfactor_bias": [float(value) for value in adjustment.bias],
                "eigenfactor_amplitude": adjustment.amplitude,
                "eigenfactor_residual_degrees_of_freedom": (
                    adjustment.fit.residual_degrees_of_freedom
                ),
                "eigenfactor_variants": variants,
                "regime_bias_dates": int(regime_bias.size),
                "regime_lambda_squared": float(regime_build.regime.lambda_squared),
                "regime_trace": float(np.trace(regime_build.matrix)),
                # Stage 2 and stage 3, recorded from W3-P2 so the fixture pins
                # the whole pipeline rather than only its last matrix. The
                # Newey-West spectrum is the quantity SPEC.md 5.2's repair acts
                # on, and whether the repair fired is the diagnostic that section
                # asks to be logged.
                "newey_west_minimum_eigenvalue": float(newey_west.minimum_eigenvalue),
                "repair_fired": bool(build.repair_fired),
                # The next four describe the matrix AS OF THE PSD REPAIR, which
                # is where the pipeline was pinned in W3-P2. The eigenfactor
                # stage's output is per-scaling and is in `eigenfactor_variants`.
                "trace": float(np.trace(build.matrix)),
                "minimum_eigenvalue": float(eigenvalues[0]),
                "maximum_eigenvalue": float(eigenvalues[-1]),
                "frobenius_error_vs_truth": float(
                    np.linalg.norm(build.matrix - spec.covariance, ord="fro")
                ),
                "k_over_t_eff": build.k_over_t_eff,
                "shepard_multiplier": build.shepard_multiplier,
            }
        record["cases"][str(factors)] = case
    return record


def _main() -> int:  # pragma: no cover - regeneration entry point
    """Print the record, MERGING any platform digests already committed.

    A regeneration run on one machine must not silently delete another
    platform's recorded digest -- that is how a fixture quietly becomes a
    single-platform anchor while still claiming to be the golden file.
    """
    import json
    from pathlib import Path

    record = golden_record()
    committed = (
        Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "covariance_golden.json"
    )
    if committed.is_file():
        previous = json.loads(committed.read_text(encoding="utf-8"))
        for factors, case in record["cases"].items():
            was = previous.get("cases", {}).get(factors, {}).get("panel_digests", {})
            for key, entry in was.items():
                case["panel_digests"].setdefault(key, entry)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(_main())
