"""``make model``, its two audits, and the session it finally went green. SPEC.md 5.

WHY THIS MODULE EXISTS AT ALL
-----------------------------

After W2-P1, ``make model`` exited green while building Model A alone. The
covariance pipeline, Model B and the hybrid were all absent, so a green
``make model`` no longer meant what its name says. **A build target that passes
by omission is the same class of defect as a test that cannot detect its own
failure mode** -- both report success for a question they never asked.

So the stage list is declared in ``covariance.stages`` in ``config/model.yaml``
rather than implied by which functions happen to exist, and this module compares
the declaration against :func:`mafrm.risk.covariance.implemented_stages`. A
stage named in the config and absent from the code makes ``make model`` **fail**,
with the missing names printed.

**AS OF W4-P1 THIS TARGET IS GREEN, for the first time in the project**, and the
paragraphs below are written for the next session that makes it red rather than
as a description of today. It was red by construction from W3-P1 to W3-P6: two of
SPEC.md 5's five stages were missing, then three components above them were.
Every one of those was closed by building the thing, which is the only way an
entry is ever allowed to leave either audit.

**A red ``make model`` is the audit working, and there are exactly two correct
responses -- implement the missing piece, or leave it red.** There is no third.
Deleting a stage from ``covariance.stages``, weakening either audit, emptying
:data:`_UNBUILT` by hand, or appending ``|| true`` in the Makefile each restores
a build target that passes by omission, which is the defect the audits exist to
detect. CI does not run this target, so a red ``make model`` blocks nothing;
``make test`` is the commit gate.

TWO AUDITS, AND THE SECOND WAS ADDED IN W3-P4 FOR A SPECIFIC REASON
-------------------------------------------------------------------

The **stage audit** compares ``covariance.stages`` against
:func:`mafrm.risk.covariance.implemented_stages`. W3-P4 wrote SPEC.md 5.4, the
last of the five, so **that audit now passes** -- and on its own it would have
turned this target green while SPEC.md 5.5's specific risk, Model B and the
hybrid were all still absent. A green ``make model`` would then have meant
exactly what it meant before W3-P1: *the part that exists, works*. That is the
defect this module was written to remove, and it would have come back as a side
effect of finishing an unrelated stage.

So the **component audit** in :data:`_UNBUILT` was added in the same session,
listing what this target's own name claims and what does not yet exist. It
declares nothing numeric and adds no configuration; it is the same mechanism as
the stage list, applied to the pieces above a stage. Each entry names the SPEC.md
section and the task that closes it, and an entry is deleted **when the thing is
built** -- never to make the target green.

WHY IT IS HERE AND NOT IN ``mafrm.risk``
-----------------------------------------

This is the only piece that needs to know both which factor set is being built
and how the generic pipeline runs, so it sits above both packages. Putting it in
``mafrm.risk`` would mean ``risk`` importing ``factors``, which CLAUDE.md
invariant 10's corollary forbids and ``tests/test_risk_architecture.py``
enforces. Putting it in ``mafrm.factors`` would make a covariance build a
property of one factor set, which is exactly the coupling SPEC.md 15.2 exists to
prevent: the week-7 module gets a second call here and nothing else changes.

CLAUDE.md invariant 5: the factor panel stops strictly before
``sample.holdout_start``.
"""

from __future__ import annotations

from mafrm import config
from mafrm.factors import macro, specific_report, statistical, statistical_report, vra_report
from mafrm.risk.config import HORIZONS, RiskConfig
from mafrm.risk.covariance import declared_stages, implemented_stages, missing_stages, run_pipeline

__all__ = ["main"]

#: What ``make model``'s name claims and what does not exist yet. Emptied one
#: entry at a time, by building the thing. See the module docstring for why this
#: exists and why it was added in W3-P4 rather than in week 8.
#:
#: SPEC.md 4.2's entry was removed in W3-P5, SPEC.md 4.3's in W3-P6 and SPEC.md
#: 5.5's in W4-P1 -- each by building the thing, never to make the target green.
#: The list is now empty; see the docstring below it for what that does and does
#: not license.
_UNBUILT: tuple[tuple[str, str, str], ...] = ()
"""Components this target's name claims and that do not yet exist.

**EMPTY AS OF W4-P1, and that is the first time.** SPEC.md 5.5 was the last
entry and it was deleted because the thing was built, which is the only reason an
entry may ever be deleted. `make model` therefore goes GREEN in this session for
the first time in the project.

Going green is not a licence to stop auditing. If a later week adds a component
this target's name covers -- the week-7 cross-sectional factor set is the obvious
candidate, since `make model` claims to build factors -- it belongs here the
moment it is planned and comes out the moment it is built.
"""


def main() -> int:
    """Build the covariance stage of ``make model`` and audit it against the config."""
    settings = config.load()
    frame = macro.macro_factor_panel(config=settings).complete

    print(
        f"covariance pipeline (SPEC.md 5) on {frame.shape[1]} factors x "
        f"{len(frame):,} dates, {frame.index[0].date()}..{frame.index[-1].date()}"
    )

    reference = RiskConfig.load(horizon=HORIZONS[0], config=settings)
    declared = declared_stages(reference)
    implemented = implemented_stages()
    missing = missing_stages(reference)
    print(f"  declared    ({len(declared)}): {', '.join(declared)}")
    print(
        f"  implemented ({len(implemented)}): "
        f"{', '.join(stage for stage in declared if stage in implemented) or '(none)'}"
    )

    # The stage audit runs BEFORE anything is built. A missing stage makes every
    # matrix below partial, and `CovarianceBuild.forecast` would raise on it --
    # correctly, but with a message about one build rather than about the gap.
    if missing:
        print()
        print(f"make model FAILS: {len(missing)} of {len(declared)} declared stage(s) are not")
        print(f"  implemented: {', '.join(missing)}")
        print("  No matrix was built: any it produced would be PARTIAL.")
        print()
        _why_red()
        return 1

    # SPEC.md 5.4's stage consumes a HISTORY of forecasts, which costs one full
    # pipeline run per date to build. It is therefore built out of band by
    # `python -m mafrm.factors.vra_report --rebuild`, committed under reports/,
    # and read here. `read_cache` refuses a cache built against a different risk
    # config rather than quietly publishing numbers from a superseded one.
    cache = vra_report.read_cache()
    print(
        f"  regime bias history: {cache.provenance['rows']:,} dates from "
        f"{cache.provenance['panel_start']}, built at commit "
        f"{str(cache.provenance['git_commit'])[:12]}"
    )

    # BOTH published eigenfactor scalings, at both horizons. CLAUDE.md's
    # parameter table: a = 1.0 is attribution-facing and is what USE4 runs in
    # production, a = 1.4 is optimizer-facing. They are two model variants and
    # this target builds both rather than choosing.
    for horizon in HORIZONS:
        base = RiskConfig.load(horizon=horizon, config=settings)
        for scaling in base.eigenfactor_scaling:
            bias = cache.bias(vra_report.Column(horizon=horizon, scaling=float(scaling)))
            build = run_pipeline(frame, base.for_scaling(scaling), regime_bias=bias)
            for line in build.render():
                print(f"  {line}")
            # `.forecast` raises unless every declared stage ran, so touching it
            # here is what stops this target printing a green line over a partial
            # matrix. Not an `assert`: `python -O` would strip it.
            if build.forecast.shape != (frame.shape[1], frame.shape[1]):  # pragma: no cover
                raise RuntimeError(f"covariance forecast has the wrong shape for {horizon}")

    print()
    print(f"SPEC.md 5 is COMPLETE: all {len(declared)} declared stage(s) ran and the matrices")
    print("  above are finished forecasts. `CovarianceBuild.forecast` was touched for each,")
    print("  so a partial matrix could not have reached this line.")

    # SPEC.md 4.2, built in W3-P5. The SAME `run_pipeline` above, on a factor set
    # it knows nothing about -- which is the SPEC.md 15.2 claim, and the reason
    # Model B appears in this target rather than only in its own report.
    print()
    print("Model B (SPEC.md 4.2): PCA with Marchenko-Pastur denoising, both variants")
    panel = statistical.asset_panel(config=settings)
    for variant in settings.model.factors.statistical.variants:
        built = statistical.statistical_factor_panel(panel, variant=variant, config=settings)
        for line in built.render():
            print(f"  {line}")
    passed = statistical_report.acceptance(settings)
    failures = [result for result in passed if not result.passed]
    print(
        f"  SPEC.md 15.2 acceptance: {len(passed) - len(failures)}/{len(passed)} combinations "
        "PSD at every stage through the UNCHANGED risk pipeline"
    )
    if failures:  # pragma: no cover - the architecture early warning
        raise RuntimeError(
            "SPEC.md 15.2's early warning has fired: Model B's factors did not pass the "
            f"covariance pipeline at {[result.column.name for result in failures]}. That is "
            "an architecture problem, not a numerical one -- do not patch it at the call site."
        )

    # SPEC.md 5.5, built in W4-P1. Unlike SPEC.md 5.4's factor leg this needs no
    # cache: the specific forecast history costs seconds rather than ninety
    # minutes, because SPEC.md 5.5 has no Monte Carlo in it.
    print()
    print("Specific risk (SPEC.md 5.5): time series, structural, blend, shrinkage, VRA")
    data = specific_report.inputs(settings)
    for horizon in HORIZONS:
        specific_build = specific_report.run(horizon, data, settings).build
        for line in specific_build.render():
            print(f"  {line}")
        # `.forecast` raises unless SPEC.md 5.4's specific multiplier was applied,
        # so touching it here is what stops this target printing a green line over
        # a pre-VRA estimate. Not an `assert`: `python -O` would strip it.
        if specific_build.forecast.shape != (data.residuals.shape[1],):  # pragma: no cover
            raise RuntimeError(f"specific risk forecast has the wrong shape for {horizon}")

    if _UNBUILT:
        print()
        print(f"make model FAILS anyway: {len(_UNBUILT)} component(s) this target's name claims")
        print("  do not exist yet.")
        for section, what, task in _UNBUILT:
            print(f"    - {section}: {what} ({task})")
        print()
        print("  This audit was added in W3-P4, the session that finished SPEC.md 5. Without")
        print("  it, completing the last covariance stage would have turned this target green")
        print("  while the three components above were still missing -- a green line meaning")
        print("  'the part that exists, works', which is the defect this module removes.")
        print()
        _why_red()
        return 1

    print()
    print("make model is GREEN, for the first time in the project. Every component this")
    print("  target's name claims now exists: Model A's factors, SPEC.md 5's five covariance")
    print("  stages at both horizons and both published eigenfactor scalings, Model B, the")
    print("  hybrid, and SPEC.md 5.5's specific risk. The two audits above -- the stage list")
    print("  against `covariance.stages`, and `_UNBUILT` against this target's name -- both")
    print("  ran and both are empty. A green line here means they were checked, not skipped.")
    return 0


def _why_red() -> None:
    """The paragraph that stops the next session routing around a red build."""
    print("  THIS FAILURE IS THE FEATURE. `make model` is designed to exit 1 until")
    print("  the model is finished in week 8. There are two correct responses --")
    print("  implement the missing piece, or leave it red. Do NOT delete a stage from")
    print("  `covariance.stages` or an entry from `_UNBUILT` (the declaration IS the")
    print("  specification), weaken this audit, or append `|| true` in the Makefile: each")
    print("  of those restores a build target that passes by omission, which is what this")
    print("  audit exists to detect. CI does not run this target, so a red `make model`")
    print("  blocks nothing.")


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
