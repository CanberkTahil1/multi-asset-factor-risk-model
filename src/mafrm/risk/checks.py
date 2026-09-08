"""Positive-semi-definiteness assertions. CLAUDE.md invariant 4.

Invariant 4 is *"assert PSD after every covariance transformation stage, not
just at the end"*, and the reason it is an invariant rather than a nicety is
CLAUDE.md failure mode 7: each correction in SPEC.md 5 passes its own sanity
check, and applied in sequence they compound or cancel. A single assertion at
the end of the pipeline tells you that something broke, and nothing about which
stage broke it.

Two thresholds, and keeping them apart is the point of this module:

``psd_eigenvalue_floor`` (1e-14)
    The **repair** value. What a negative eigenvalue is pushed up to in
    SPEC.md 5.2's mandatory PSD repair. Used by the repair stage, not here.

``psd_reconstruction_roundings`` (2)
    The **detection** threshold, in units of machine epsilon. A matrix is
    refused when ``lambda_min < -(size + 2) * eps * lambda_max``. Used here, and
    nowhere else. See :func:`eigenvalue_floor` for why it is expressed this way
    and not as an absolute number.

If those two were one number, a matrix could be silently repaired without
anything noticing, which is precisely the diagnostic SPEC.md 5.2 asks to be
logged: *"log how often the PSD repair fires and by how much -- that number is
itself a diagnostic."*

Neither threshold is a dial. A stage that needs one loosened is a stage with a
bug, and widening a tolerance until the residual fits inside it is the failure
CLAUDE.md's residual stopping rule exists to prevent. :class:`PsdCheck`
therefore reports the observed **margin** at every stage rather than only
pass/fail, so an approach to a threshold is visible several sessions before it
becomes a failure.

THE DETECTION THRESHOLD IS RELATIVE TO SCALE, AND WAS NOT ALWAYS (W4-P2b)
-------------------------------------------------------------------------

Until 2026-09-02 the threshold was the absolute ``-1e-12`` of the W3-P1 ruling,
justified in ``config/model.yaml`` on the ground that this project's covariance
matrices have eigenvalues of order 1e1-1e2, putting roundoff two orders of
magnitude inside it. **That premise was never measured and is false**: the factor
covariance's largest eigenvalue reaches 5.43e+04, so one ulp of it is 1.21e-11 --
twelve times the old threshold. An absolute bound an order of magnitude inside
the representation noise carries no information: it can refuse a matrix that is
positive semi-definite to a fraction of one epsilon, and it did, at two of
W4-P2b's nine sweep half-lives.

The bound is therefore ``-(size + psd_reconstruction_roundings) * eps *
lambda_max``. **This is not a widening.** On a matrix of scale 1 it is tighter
than the number it replaced, and in correlation space -- where SPEC.md 5.3
operates and ``lambda_max <= K`` -- it is about ninety times tighter. It still
rejects a real failure by orders of magnitude: the pre-repair Newey-West matrices
that motivated the change sit at ``lambda_min / lambda_max`` of -4e-4, which is
twelve orders outside the bound.

MEASUREMENT AND ASSERTION ARE TWO DIFFERENT CALLS (W3-P2)
---------------------------------------------------------

:func:`measure_spectrum` reports a matrix's spectrum and never gates on it;
:func:`assert_psd` measures and then refuses. Both run the shape, finiteness and
symmetry checks, because those are bugs at any stage and in any convention -- an
asymmetric matrix is not a matrix anybody built.

**The split is a reading of invariant 4, not a relaxation of it, and the
distinction is worth stating precisely because a later session will be tempted
to "fix" it into a crash.** Invariant 4 exists to catch *silent* non-PSD: a
stage that quietly hands a broken matrix to the next one. SPEC.md 5.2's
Newey-West stage is the one place in the pipeline where a non-PSD result is
**declared in advance** -- the SPEC calls the repair that follows it "REQUIRED",
and the next stage in ``covariance.stages`` exists for no other purpose. A stage
whose successor is its repair is not silent, so asserting between the two would
not detect a defect; it would only convert an expected and already-handled
outcome into a crash, and the only way to keep the pipeline running would then be
to widen ``psd_minimum_eigenvalue`` -- the exact move CLAUDE.md's residual
stopping rule and this module's own docstring forbid.

So: ``newey_west`` measures, ``psd_repair`` floors, and the assertion runs after
the repair, which is the transformation that is *supposed* to yield a PSD matrix.
Every other stage asserts. ``tests/test_covariance.py`` pins that list, so the
exemption cannot spread to a second stage without a test changing.

This module knows nothing about what the rows and columns of its input mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from mafrm.config import Config, NumericsConfig, load

#: Machine epsilon for float64. A property of the representation, not a choice.
_EPS: Final[float] = float(np.finfo(np.float64).eps)

__all__ = [
    "PsdCheck",
    "PsdError",
    "Spectrum",
    "assert_psd",
    "eigenvalue_floor",
    "measure_spectrum",
]


class PsdError(ValueError):
    """A matrix failed a shape, symmetry or positive-semi-definiteness check."""


@dataclass(frozen=True)
class Spectrum:
    """What a matrix's eigenvalues are, with no judgement about whether that is ok.

    Returned by :func:`measure_spectrum` for a stage whose output is *expected*
    to be non-PSD and is repaired by the stage that follows it. Carries no
    threshold, because nothing was checked against one.
    """

    stage: str
    size: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    #: ``max |A - A.T|`` over all entries.
    maximum_asymmetry: float
    #: Eigenvalues strictly below zero. This is the count SPEC.md 5.2 asks to be
    #: logged -- *"log how often the PSD repair fires and by how much"* -- and it
    #: is measured before the repair, which is the only place it exists.
    negative_eigenvalues: int

    @property
    def condition_number(self) -> float:
        """Largest over smallest eigenvalue; ``inf`` at or below zero."""
        if self.minimum_eigenvalue <= 0.0:
            return float("inf")
        return self.maximum_eigenvalue / self.minimum_eigenvalue

    def render(self) -> str:
        """One line, for ``make model`` and for a report. Says PSD or does not."""
        condition = (
            "singular" if not np.isfinite(self.condition_number) else f"{self.condition_number:.1f}"
        )
        verdict = (
            f"{self.negative_eigenvalues} negative eigenvalue(s), repair pending"
            if self.negative_eigenvalues
            else "PSD as measured, not asserted"
        )
        return (
            f"{self.stage}: {self.size}x{self.size} {verdict}, "
            f"eigenvalues [{self.minimum_eigenvalue:+.6e}, {self.maximum_eigenvalue:.6e}], "
            f"condition {condition}, asymmetry {self.maximum_asymmetry:.2e}"
        )


@dataclass(frozen=True)
class PsdCheck:
    """What :func:`assert_psd` observed, kept whether or not it passed.

    Carried rather than discarded because the margins are a diagnostic in their
    own right: a stage whose least eigenvalue is drifting toward the threshold
    over successive builds is a stage about to fail, and only the recorded
    number shows that before it does.
    """

    #: Name of the pipeline stage whose output this is, for the failure message.
    stage: str
    size: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    #: ``max |A - A.T|`` over all entries.
    maximum_asymmetry: float
    #: The thresholds this was checked against, so a report is self-contained.
    minimum_eigenvalue_threshold: float
    symmetry_threshold: float

    @property
    def condition_number(self) -> float:
        """Largest over smallest eigenvalue; ``inf`` for a singular matrix.

        Reported, never gated. A high condition number is a property of the data
        -- factors genuinely do co-move in a crisis -- not of the implementation,
        so there is no value of it that would mean the code is wrong. What it
        measures is the size of the problem SPEC.md 5.3's eigenfactor adjustment
        exists to correct.
        """
        if self.minimum_eigenvalue <= 0.0:
            return float("inf")
        return self.maximum_eigenvalue / self.minimum_eigenvalue

    @property
    def eigenvalue_margin(self) -> float:
        """How far the least eigenvalue sits above its threshold. Negative fails."""
        return self.minimum_eigenvalue - self.minimum_eigenvalue_threshold

    def render(self) -> str:
        """One line, for ``make model`` and for a report."""
        condition = (
            "singular" if not np.isfinite(self.condition_number) else f"{self.condition_number:.1f}"
        )
        return (
            f"{self.stage}: {self.size}x{self.size} PSD ok, "
            f"eigenvalues [{self.minimum_eigenvalue:+.6e}, {self.maximum_eigenvalue:.6e}], "
            f"condition {condition}, asymmetry {self.maximum_asymmetry:.2e}"
        )


def measure_spectrum(
    matrix: np.ndarray,
    stage: str,
    *,
    expected_size: int | None = None,
    numerics: NumericsConfig | None = None,
    config: Config | None = None,
) -> Spectrum:
    """Report ``matrix``'s spectrum. **Runs no PSD gate.**

    Shape, finiteness and symmetry are still enforced -- they are bugs at any
    stage, and symmetry must precede the eigendecomposition because
    ``numpy.linalg.eigvalsh`` reads only one triangle, so an asymmetric matrix
    would yield the eigenvalues of a matrix nobody built.

    Use this only where the next stage is the repair for what this one produces;
    see the module docstring. Everywhere else, :func:`assert_psd`.
    """
    settings = numerics if numerics is not None else (config or load()).model.numerics
    _check_shape(matrix, stage, expected_size)
    asymmetry = _check_symmetry(matrix, stage, settings)
    eigenvalues = np.linalg.eigvalsh(matrix)
    return Spectrum(
        stage=stage,
        size=matrix.shape[0],
        minimum_eigenvalue=float(eigenvalues[0]),
        maximum_eigenvalue=float(eigenvalues[-1]),
        maximum_asymmetry=asymmetry,
        negative_eigenvalues=int(np.count_nonzero(eigenvalues < 0.0)),
    )


def _check_shape(matrix: np.ndarray, stage: str, expected_size: int | None) -> None:
    if matrix.ndim != 2:
        raise PsdError(f"{stage}: expected a 2-D matrix, got shape {matrix.shape}")
    rows, columns = matrix.shape
    if rows != columns:
        raise PsdError(f"{stage}: expected a square matrix, got shape {matrix.shape}")
    if expected_size is not None and rows != expected_size:
        raise PsdError(
            f"{stage}: expected a {expected_size}x{expected_size} matrix, got {rows}x{columns}"
        )
    if rows == 0:
        raise PsdError(f"{stage}: empty matrix")
    if not np.all(np.isfinite(matrix)):
        count = int(np.count_nonzero(~np.isfinite(matrix)))
        raise PsdError(f"{stage}: {count} non-finite entr{'y' if count == 1 else 'ies'}")


def _check_symmetry(matrix: np.ndarray, stage: str, settings: NumericsConfig) -> float:
    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    if asymmetry > settings.symmetry_absolute_tolerance:
        raise PsdError(
            f"{stage}: not symmetric. max|A - A.T| = {asymmetry:.3e}, tolerance "
            f"{settings.symmetry_absolute_tolerance:.1e} "
            "(model.numerics.symmetry_absolute_tolerance)"
        )
    return asymmetry


def eigenvalue_floor(size: int, maximum_eigenvalue: float, *, numerics: NumericsConfig) -> float:
    """The least eigenvalue a ``size x size`` matrix of this scale may show.

    ``-(size + psd_reconstruction_roundings) * eps * lambda_max``. W4-P2b ruling.

    WHY EPSILON AND NOT A RELATIVE CONSTANT
        A bare relative tolerance -- ``lambda_min / lambda_max >= -1e-12`` -- would
        clear the residual that motivated this change, and that is exactly what
        is wrong with it: the number would have been chosen by the residual. One
        ulp of ``lambda_max`` is a property of the double-precision
        representation, fixed before anything here was measured, and the observed
        violations are **below a single ulp** (0.14 and 0.23 of one). The W1-P3
        precedent requires the basis of an accommodation to be independent of the
        thing accommodated, and ``eps`` is; ``1e-12`` would not be.

    WHY THE COEFFICIENT IS A STEP COUNT
        It counts the roundings a repaired matrix accumulates per entry between
        its floored spectrum and this assertion: ``size`` for the inner product
        of the reconstruction ``(V L) V'``, one for the scaling by ``L``, and one
        for the explicit symmetrisation. Weyl's inequality carries each
        perturbation to every eigenvalue one for one. The two eigendecompositions'
        own backward errors are the same order in ``size`` and are deliberately
        NOT counted, because what covers them is measured margin rather than
        argument -- see ``reports/psd_repairs.md``.

        ``size`` is a property of the matrix and not of the asset class, so this
        stays within CLAUDE.md invariant 10 and needs no revision for W7.

    ``max(lambda_max, 0)`` because a negative-definite matrix must not be handed
    a negative scale and thereby a permissive bound; it gets a bound of zero and
    fails, which is correct.
    """
    scale = max(maximum_eigenvalue, 0.0)
    return -float(size + numerics.psd_reconstruction_roundings) * _EPS * scale


def assert_psd(
    matrix: np.ndarray,
    stage: str,
    *,
    expected_size: int | None = None,
    numerics: NumericsConfig | None = None,
    config: Config | None = None,
) -> PsdCheck:
    """Assert ``matrix`` is a square, symmetric, positive-semi-definite matrix.

    Four checks, in the order that makes a failure message useful: shape, then
    finiteness, then symmetry, then the eigenvalue floor. The first three are
    :func:`measure_spectrum`'s; this adds the gate.

    ``stage`` names the pipeline stage being checked and appears in the error, so
    a failure says *which* of SPEC.md 5's stages produced the bad matrix rather
    than only that one of them did.

    Returns the observed :class:`PsdCheck` on success; raises :class:`PsdError`
    with the observed value and the threshold on failure.
    """
    settings = numerics if numerics is not None else (config or load()).model.numerics
    spectrum = measure_spectrum(
        matrix, stage, expected_size=expected_size, numerics=settings, config=config
    )
    threshold = eigenvalue_floor(spectrum.size, spectrum.maximum_eigenvalue, numerics=settings)
    if spectrum.minimum_eigenvalue < threshold:
        ulps = spectrum.minimum_eigenvalue / (_EPS * spectrum.maximum_eigenvalue)
        raise PsdError(
            f"{stage}: not positive semi-definite. Least eigenvalue "
            f"{spectrum.minimum_eigenvalue:.6e} is below {threshold:.6e}, which is "
            f"-(size {spectrum.size} + {settings.psd_reconstruction_roundings}) x eps x "
            f"lambda_max {spectrum.maximum_eigenvalue:.6e} "
            "(model.numerics.psd_reconstruction_roundings). The violation is "
            f"{ulps:.2f} ulp of lambda_max. CLAUDE.md invariant 4: repair by eigenvalue "
            "flooring and log that it fired -- do NOT widen this threshold."
        )
    return PsdCheck(
        stage=stage,
        size=spectrum.size,
        minimum_eigenvalue=spectrum.minimum_eigenvalue,
        maximum_eigenvalue=spectrum.maximum_eigenvalue,
        maximum_asymmetry=spectrum.maximum_asymmetry,
        minimum_eigenvalue_threshold=threshold,
        symmetry_threshold=settings.symmetry_absolute_tolerance,
    )
