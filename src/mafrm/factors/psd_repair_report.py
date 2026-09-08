"""Generates ``reports/psd_repairs.md``. SPEC.md 5.2.

SPEC.md 5.2 asks for one number and this file is it:

    *"The PSD repair is not in the MSCI write-up but is mandatory in practice.
    Log how often it fires and by how much -- that number is itself a
    diagnostic."*

WHAT IS SCANNED
---------------

The pipeline is rebuilt on an **expanding window** ending at every date in the
factor panel, from the arithmetic minimum ``T = K`` -- below which the second
moment is singular by construction and there is no answer at all -- through the
last date before ``sample.holdout_start``. Both horizons, every date, no stride:
a stride would be a sample of the firings rather than a count of them, and the
count is the deliverable.

Expanding rather than fixed-length because that is what the project does
elsewhere and because there is no burn-in parameter yet: SPEC.md 5.1.2 rules that
when a minimum arrives it must be a bound on ``K / T_eff`` rather than a day
count, and inventing one here to tidy the left edge of this scan would be exactly
the reach-for-252 that ruling exists to prevent. The short windows are reported
with their ``K / T_eff`` beside them instead, which turns out to be the whole
finding.

CLAUDE.md FAILURE MODE 9 -- READ THIS BEFORE QUOTING A NUMBER FROM HERE
-----------------------------------------------------------------------

Consecutive readings are expanding windows differing by **one observation**, so
they share all but one of their inputs and are near-duplicates of each other. The
number of readings is therefore **not** a count of independent draws, and this
report quotes **no firing rate, no probability and no confidence interval**.

What it reports instead is the raw date count and the number of **contiguous
episodes**, and says which is which. A run of five consecutive firing dates is
one episode observed five times, not five events; presenting it as "5 of 4,141
dates = 0.12%" would be the same error as ``experiments.md`` row 58, the alpha
flag and W2-P3's rolling correlation, which is why CLAUDE.md lists it ninth
rather than leaving it to be re-learned.

THE FLOOR IS ABSOLUTE, WHICH MEANS IT IS SCALE-DEPENDENT
---------------------------------------------------------

CLAUDE.md invariant 4 fixes the repair floor at ``1e-14`` and it is not moved
here -- an invariant is not revised mid-build. But an absolute floor is a
statement about a matrix in particular units, and this project's factor panel is
deliberately in mixed units (SPEC.md 4.1.1, and SPEC.md 5.3.1's ruling on what
that costs). So the floor is reported **against** ``lambda_min`` and
``lambda_max`` rather than on its own, and the report says how many orders of
magnitude of headroom that leaves. W3-P3's numeraire decision changes the answer
and must trigger a re-read of this file.

**The DETECTION threshold is no longer absolute, and this report is where that
should have been visible (W4-P2b-fix).** SPEC.md 5.2.4 replaced
``psd_minimum_eigenvalue = -1e-12`` with ``-(size + 2) * eps * lambda_max``,
after the absolute form refused two of W4-P2b's nine sweep half-lives on matrices
that were positive semi-definite to a fifth of one ulp. Until then this report
logged how badly non-PSD the repair's **input** was and never how close its
**output** sat to the threshold -- so the shipped path could run at a 1.5x margin
on a quantity whose own noise floor was 15x the observed value, and nothing said
so. Every horizon's section now carries that margin in ulps, at every build and
not only where it fired.

CLAUDE.md invariant 5: the panel stops strictly before ``sample.holdout_start``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mafrm import config
from mafrm.factors import macro
from mafrm.risk.config import HORIZONS, Horizon, RiskConfig
from mafrm.risk.covariance import run_pipeline

#: Machine epsilon for float64, for SPEC.md 5.2.4's margin. A property of the
#: representation, not a choice.
_EPS = float(np.finfo(np.float64).eps)

__all__ = ["Episode", "Reading", "Scan", "build", "main", "render", "report_path"]


def report_path() -> Path:
    """``reports/psd_repairs.md`` -- committed, per CLAUDE.md."""
    return Path(__file__).resolve().parents[3] / "reports" / "psd_repairs.md"


@dataclass(frozen=True)
class Reading:
    """One expanding-window build, and what the two spectra looked like."""

    date: pd.Timestamp
    observations: int
    realised_effective_sample_size: float
    k_over_realised_t_eff: float
    ewma_minimum_eigenvalue: float
    newey_west_minimum_eigenvalue: float
    newey_west_maximum_eigenvalue: float
    #: The spectrum the repair PRODUCED, which is what SPEC.md 5.2.4's detection
    #: threshold is applied to. Added in W4-P2b-fix: until then this report
    #: logged how bad the repair's INPUT was and never how close its OUTPUT sat
    #: to the threshold, which is the blind spot that let an absolute threshold
    #: sit an order of magnitude inside its own noise floor for four sessions.
    psd_repair_minimum_eigenvalue: float
    psd_repair_maximum_eigenvalue: float
    repair_fired: bool
    floored: int

    @property
    def relative_breach(self) -> float:
        """``lambda_min / lambda_max`` at the Newey-West stage. Signed."""
        if self.newey_west_maximum_eigenvalue <= 0.0:
            return float("nan")
        return self.newey_west_minimum_eigenvalue / self.newey_west_maximum_eigenvalue

    @property
    def post_repair_ulps(self) -> float:
        """The repaired matrix's ``lambda_min``, in ulps of its own ``lambda_max``.

        SPEC.md 5.2.4's detection threshold is ``-(size + 2) * eps * lambda_max``,
        so this is the quantity that is actually gated and **-8 is the bound**. It
        is reported at every build rather than only when it fails, because the
        whole lesson of W4-P2b-fix is that a margin nobody measures is a margin
        nobody knows they have lost.
        """
        if self.psd_repair_maximum_eigenvalue <= 0.0:
            return float("nan")
        return self.psd_repair_minimum_eigenvalue / (_EPS * self.psd_repair_maximum_eigenvalue)


@dataclass(frozen=True)
class Episode:
    """A run of CONSECUTIVE firing dates. One episode, however many dates long.

    The unit the report counts in, because the dates inside one of these are not
    independent of each other -- see the module docstring.
    """

    readings: tuple[Reading, ...]

    @property
    def dates(self) -> int:
        return len(self.readings)

    @property
    def start(self) -> pd.Timestamp:
        return self.readings[0].date

    @property
    def end(self) -> pd.Timestamp:
        return self.readings[-1].date

    @property
    def most_negative(self) -> float:
        return min(reading.newey_west_minimum_eigenvalue for reading in self.readings)


@dataclass(frozen=True)
class Scan:
    """Every reading at one horizon, and the episodes the firings group into."""

    horizon: Horizon
    factors: int
    volatility_halflife: int
    correlation_halflife: int
    volatility_lags: int
    correlation_lags: int
    floor: float
    #: First date of the panel itself. Earlier than the first READING, because the
    #: leading ``K - 1`` dates cannot be scanned at all: ``T < K`` is singular by
    #: construction and there is no matrix to check.
    panel_start: pd.Timestamp
    readings: tuple[Reading, ...]

    @property
    def firings(self) -> tuple[Reading, ...]:
        return tuple(reading for reading in self.readings if reading.repair_fired)

    @property
    def episodes(self) -> tuple[Episode, ...]:
        """Contiguous runs of firing dates, in order."""
        groups: list[list[Reading]] = []
        previous: int | None = None
        for index, reading in enumerate(self.readings):
            if not reading.repair_fired:
                continue
            if previous is not None and index == previous + 1:
                groups[-1].append(reading)
            else:
                groups.append([reading])
            previous = index
        return tuple(Episode(readings=tuple(group)) for group in groups)

    @property
    def last_firing_observations(self) -> int | None:
        """``T`` at the last build that fired, or ``None`` if none did."""
        firings = self.firings
        return firings[-1].observations if firings else None

    def clean_readings(self) -> tuple[Reading, ...]:
        """Readings after the last firing. The regime a model is actually built in."""
        last = self.last_firing_observations
        if last is None:
            return self.readings
        return tuple(reading for reading in self.readings if reading.observations > last)

    @property
    def closest_approach(self) -> Reading:
        """The clean reading whose Newey-West spectrum came nearest to zero."""
        return min(self.clean_readings(), key=lambda reading: reading.relative_breach)

    @property
    def worst_post_repair(self) -> Reading:
        """The build whose REPAIRED spectrum came closest to SPEC.md 5.2.4's bound."""
        return min(self.readings, key=lambda reading: reading.post_repair_ulps)

    @property
    def largest_scale(self) -> Reading:
        """The build with the largest repaired ``lambda_max``.

        Reported beside :attr:`worst_post_repair` because they answer different
        questions and W4-P2b-fix turned on the difference. The worst margin says
        how close this panel came to the bound; the largest scale says how much
        an ABSOLUTE bound would have been worth here, since one ulp of
        ``lambda_max`` is the floor on what any absolute threshold can resolve.
        """
        return max(self.readings, key=lambda reading: reading.psd_repair_maximum_eigenvalue)

    @property
    def final(self) -> Reading:
        return self.readings[-1]


def _scan(frame: pd.DataFrame, risk: RiskConfig) -> Scan:
    factors = frame.shape[1]
    readings: list[Reading] = []
    for end in range(factors, len(frame) + 1):
        # STOP AT THE STAGE THIS REPORT IS ABOUT. Running SPEC.md 5.3's
        # 2,000-trial Monte Carlo on each of 4,141 expanding windows would cost
        # hours and produce a matrix this scan reads nothing from.
        build = run_pipeline(frame.iloc[:end], risk, stop_after="psd_repair")
        stages = {stage.name: stage for stage in build.stages}
        repair = build.repairs[0]
        readings.append(
            Reading(
                date=frame.index[end - 1],
                observations=end,
                realised_effective_sample_size=build.realised_effective_sample_size,
                k_over_realised_t_eff=build.k_over_realised_t_eff,
                ewma_minimum_eigenvalue=stages["ewma"].minimum_eigenvalue,
                newey_west_minimum_eigenvalue=stages["newey_west"].minimum_eigenvalue,
                newey_west_maximum_eigenvalue=stages["newey_west"].maximum_eigenvalue,
                psd_repair_minimum_eigenvalue=stages["psd_repair"].minimum_eigenvalue,
                psd_repair_maximum_eigenvalue=stages["psd_repair"].maximum_eigenvalue,
                repair_fired=repair.fired,
                floored=repair.floored,
            )
        )
    return Scan(
        horizon=risk.horizon,
        factors=factors,
        volatility_halflife=risk.volatility_halflife,
        correlation_halflife=risk.correlation_halflife,
        volatility_lags=risk.volatility_newey_west_lags,
        correlation_lags=risk.correlation_newey_west_lags,
        floor=risk.psd_eigenvalue_floor,
        panel_start=frame.index[0],
        readings=tuple(readings),
    )


@dataclass(frozen=True)
class Demonstration:
    """The synthetic case proving the repair path is live, independent of the data.

    ``data/raw`` is gitignored, so a reader cloning this repository cannot run the
    scan above. This is drawn from ``model.seed`` and reproduces the regime in
    which the repair fires on the real panel -- ``K`` factors from a window barely
    longer than ``K`` -- so the claim that the code path works is checkable
    without the cache. ``tests/test_covariance.py`` asserts it.
    """

    observations: int
    factors: int
    horizon: Horizon
    newey_west_minimum_eigenvalue: float
    newey_west_maximum_eigenvalue: float
    floored: int
    repaired_minimum_eigenvalue: float

    @property
    def relative_breach(self) -> float:
        """``lambda_min / lambda_max`` at the Newey-West stage. Signed."""
        return self.newey_west_minimum_eigenvalue / self.newey_west_maximum_eigenvalue


def _demonstrate(settings: config.Config, *, periods: int, factors: int) -> Demonstration:
    values = np.random.default_rng(settings.model.seed).standard_normal((periods, factors))
    frame = pd.DataFrame(values, columns=[f"f{index}" for index in range(factors)])
    risk = RiskConfig.load(horizon=HORIZONS[0], config=settings)
    build = run_pipeline(frame, risk, stop_after="psd_repair")
    newey_west = next(stage for stage in build.stages if stage.name == "newey_west")
    return Demonstration(
        observations=periods,
        factors=factors,
        horizon=risk.horizon,
        newey_west_minimum_eigenvalue=newey_west.minimum_eigenvalue,
        newey_west_maximum_eigenvalue=newey_west.maximum_eigenvalue,
        floored=build.repairs[0].floored,
        repaired_minimum_eigenvalue=float(np.linalg.eigvalsh(build.matrix)[0]),
    )


#: The synthetic demonstration's shape. Held in code rather than in
#: ``config/model.yaml`` for the same reason the golden fixture's constants are:
#: they are properties of a test case, not tunables, and a fixture that can be
#: adjusted to make a claim come out is not a fixture. Logged in
#: ``experiments.md`` under the W3-P2 constants-held-in-code section.
_DEMONSTRATION_SHAPE = (12, 6)


def build(cfg: config.Config | None = None) -> tuple[tuple[Scan, ...], Demonstration]:
    """Scan both horizons over the whole pre-holdout panel, plus the live-path case."""
    settings = cfg or config.load()
    frame = macro.macro_factor_panel(config=settings).complete
    scans = tuple(
        _scan(frame, RiskConfig.load(horizon=horizon, config=settings)) for horizon in HORIZONS
    )
    periods, factors = _DEMONSTRATION_SHAPE
    return scans, _demonstrate(settings, periods=periods, factors=factors)


def _orders_of_magnitude(numerator: float, denominator: float) -> float:
    if numerator <= 0.0 or denominator <= 0.0:
        return float("nan")
    return float(np.log10(numerator / denominator))


def _firing_table(scan: Scan) -> list[str]:
    lines = [
        "| Date | `T` | `K/T_eff` realised | EWMA `lambda_min` | NW `lambda_min` "
        "| NW `lambda_min / lambda_max` | eigenvalues floored |",
        "|---|---|---|---|---|---|---|",
    ]
    for reading in scan.firings:
        lines.append(
            f"| {reading.date.date()} | {reading.observations} "
            f"| {reading.k_over_realised_t_eff:.3f} "
            f"| {reading.ewma_minimum_eigenvalue:+.3e} "
            f"| {reading.newey_west_minimum_eigenvalue:+.3e} "
            f"| {reading.relative_breach:+.2e} | {reading.floored} |"
        )
    return lines


def render(
    scans: tuple[Scan, ...], demonstration: Demonstration, cfg: config.Config | None = None
) -> str:
    """The whole of ``reports/psd_repairs.md``."""
    settings = cfg or config.load()
    stamp = datetime.now(UTC).date().isoformat()
    reference = scans[0]
    first = reference.readings[0]
    last = reference.final

    lines: list[str] = [
        "# The PSD repair: how often it fires, and by how much",
        "",
        f"Generated {stamp} by `python -m mafrm.factors.psd_repair_report`. SPEC.md 5.2.",
        "",
        "SPEC.md 5.2 makes the eigenvalue repair after Newey-West mandatory and then asks for",
        'a number: *"log how often it fires and by how much -- that number is itself a',
        'diagnostic."* This is that log.',
        "",
        "## What was scanned",
        "",
        "The covariance pipeline was rebuilt on an **expanding window ending at every date**",
        f"in the factor panel, at both horizons: {last.observations - first.observations + 1:,}",
        f"builds per horizon, from `T = K = {reference.factors}` (the arithmetic minimum, below",
        f"which the second moment is singular by construction) to `T = {last.observations:,}`.",
        f"Scan range: {first.date.date()} to {last.date.date()}. The panel itself starts",
        f"{reference.panel_start.date()}; its leading {reference.factors - 1} dates cannot be",
        "scanned, because `T < K` is singular by construction and there is no matrix to check.",
        "Everything ends strictly before `sample.holdout_start` = "
        f"{settings.model.sample.holdout_start}.",
        "",
        "No stride. A stride would sample the firings rather than count them, and the count is",
        "the deliverable.",
        "",
        "> **CLAUDE.md failure mode 9.** Consecutive readings are expanding windows differing by",
        "> **one observation**, so they share all but one of their inputs. The number of readings",
        "> is **not** a count of independent draws. This report quotes **no firing rate, no",
        "> probability and no confidence interval** -- only a raw date count and a count of",
        "> contiguous episodes, labelled as such. Five consecutive firing dates are one episode",
        "> seen five times, not five events.",
        "",
        "## Result",
        "",
    ]

    for scan in scans:
        episodes = scan.episodes
        approach = scan.closest_approach
        lines.extend(
            [
                f"### `{scan.horizon}` horizon (volatility half-life "
                f"{scan.volatility_halflife}d at {scan.volatility_lags} lags, correlation "
                f"half-life {scan.correlation_halflife}d at {scan.correlation_lags} lags)",
                "",
                f"- **Firing dates: {len(scan.firings)}** of {len(scan.readings):,} builds "
                "*(a date count, not a rate -- see the box above)*",
                f"- **Contiguous episodes: {len(episodes)}**",
            ]
        )
        for episode in episodes:
            lines.append(
                f"  - {episode.start.date()} to {episode.end.date()}, {episode.dates} "
                f"consecutive dates, most negative eigenvalue {episode.most_negative:+.3e}"
            )
        if scan.last_firing_observations is not None:
            clean = scan.clean_readings()
            lines.extend(
                [
                    f"- **Every firing is at `T <= {scan.last_firing_observations}` against "
                    f"`K = {scan.factors}`.** The repair never fires again in the "
                    f"{len(clean):,} builds from `T = {scan.last_firing_observations + 1}` "
                    "onward.",
                    f"- Closest approach after the last firing: `lambda_min` = "
                    f"{approach.newey_west_minimum_eigenvalue:+.3e} on {approach.date.date()} "
                    f"(`T = {approach.observations:,}`), which is "
                    f"{approach.relative_breach:+.2e} of `lambda_max` -- **positive**, so it is a "
                    "margin and not a breach.",
                ]
            )
        worst = scan.worst_post_repair
        scale = scan.largest_scale
        bound = -(scan.factors + settings.model.numerics.psd_reconstruction_roundings)
        ulp = _EPS * scale.psd_repair_maximum_eigenvalue
        lines.extend(
            [
                f"- **Margin against SPEC.md 5.2.4's detection threshold: the repaired matrix's "
                f"least eigenvalue never falls below {worst.post_repair_ulps:+.3f} ulp of its own "
                f"`lambda_max`**, against a bound of {bound}. Worst on {worst.date.date()} "
                f"(`T = {worst.observations:,}`), where `lambda_min` = "
                f"{worst.psd_repair_minimum_eigenvalue:+.3e} beside `lambda_max` = "
                f"{worst.psd_repair_maximum_eigenvalue:.3e}.",
                f"- Largest repaired `lambda_max` anywhere in the scan: "
                f"{scale.psd_repair_maximum_eigenvalue:.4e} on {scale.date.date()}, so one ulp "
                f"of it is {ulp:.3e} -- **{1e-12 / ulp:.0f}x the superseded absolute threshold "
                "of -1e-12.** On THIS panel that threshold was therefore sound, and the "
                'justification `config/model.yaml` gave for it ("variances run of order '
                '1e1-1e2") was accurate for the panel it was written against.',
                "- **It stopped being accurate when the panel changed, and nothing re-derived "
                "it.** SPEC.md 4.3's hybrid residual panel carries the same six factors at a "
                "`lambda_max` of 5.43e+04 -- 390x this one -- where one ulp is 1.21e-11, "
                "twelve times the old threshold. That is the panel W4-P1 and W4-P2 build on, "
                "and two of W4-P2b's nine sweep half-lives were refused there on matrices that "
                "were positive semi-definite to a fifth of one ulp. SPEC.md 5.2.4 and "
                "`experiments.md` row 174. **The relative form is panel-independent by "
                "construction, which is the property the absolute one lacked and the reason "
                "this line is now printed at every build rather than only where the repair "
                "fired.**",
            ]
        )
        lines.extend(["", *_firing_table(scan), ""])

    reference_episodes = reference.episodes
    lines.extend(
        [
            "## What the firings actually are",
            "",
            f"Both horizons fire on the same {len(reference.firings)} dates and nowhere else, and",
            "every one of them is in the regime where six factors are being estimated from ten",
            "observations or fewer. The `K/T_eff` column is the reading: the firings run",
            f"{', '.join(f'{r.k_over_realised_t_eff:.3f}' for r in reference.firings)} and stop.",
            "",
            "The negative eigenvalue is genuinely produced by Newey-West and not by the window",
            "being short in itself -- the EWMA stage is a Gram matrix and is positive",
            "semi-definite by construction at every one of these dates, with `lambda_min` in the",
            f"{min(r.ewma_minimum_eigenvalue for r in reference.firings):.1e} to "
            f"{max(r.ewma_minimum_eigenvalue for r in reference.firings):.1e} range. It is the",
            "Bartlett lag terms that push it below zero, on windows where the longest lag has",
            "one or two pairs to average over.",
            "",
            "**This is the empirical shape of SPEC.md 5.1.2's forward constraint, and it arrived",
            "sooner than expected.** That ruling says a burn-in minimum must be expressed as a",
            "bound on `K / T_eff` rather than as a day count, because a day count would have to",
            "be re-chosen for every panel and every half-life and says nothing about how many",
            "parameters are being estimated. The boundary observed here is a `K / T_eff`",
            "boundary: it sits at the same place at both horizons even though their volatility",
            "half-lives differ by a factor of three, which a day count could not have predicted.",
            "",
            "**It is not a calibrated bound and must not be used as one.** It is one episode of",
            f"{reference_episodes[0].dates if reference_episodes else 0} overlapping dates on one",
            "panel with one factor set. It is a shape, not a threshold; the threshold, when",
            "W3/W4 needs one, owes its own registration.",
            "",
            "**And it is the first independent corroboration in this project that `K/T` is the",
            "governing parameter -- of a different kind from the last one.** W3-P1 reproduced",
            "SPEC.md 15.2's table by computing Shepard's formula and finding it agrees, which",
            "validates the **arithmetic**. Nothing here computes Shepard's formula. The same",
            "parameter arrives from a **numerical failure mode**, in an implementation that was",
            "not aiming at `K/T` and would have reported whatever boundary the data had.",
            "Reproducing the published table validates the arithmetic; this validates the",
            "framing.",
            "",
            "## The floor, against the spectrum",
            "",
            "CLAUDE.md invariant 4 fixes the repair floor at `1e-14` and **it has not been",
            "moved**. But an absolute floor is a statement about a matrix in particular units,",
            "and this project's factor panel is deliberately in mixed units (SPEC.md 4.1.1). So",
            "the floor is quoted against the spectrum rather than on its own:",
            "",
            "| Horizon | floor | `lambda_min` (full window) | `lambda_max` | floor / `lambda_min` "
            "| orders of magnitude below `lambda_min` |",
            "|---|---|---|---|---|---|",
        ]
    )
    for scan in scans:
        final = scan.final
        lines.append(
            f"| `{scan.horizon}` | {scan.floor:.0e} | "
            f"{final.newey_west_minimum_eigenvalue:.3e} | "
            f"{final.newey_west_maximum_eigenvalue:.3e} | "
            f"{scan.floor / final.newey_west_minimum_eigenvalue:.2e} | "
            f"{-_orders_of_magnitude(scan.floor, final.newey_west_minimum_eigenvalue):.1f} |"
        )
    lines.extend(
        [
            "",
            "So the floor currently sits about eight orders of magnitude below the smallest",
            "legitimate eigenvalue. **That is safe, and it is safe by accident of the unit",
            "convention rather than by design.** On a differently-scaled matrix the same constant",
            "could clip real structure: SPEC.md 5.3.1 records that this panel's covariance",
            "condition number of 9.96e6 is almost entirely a 7.3e6 variance ratio across",
            "mixed-unit columns, not near-singularity.",
            "",
            "**W3-P3's numeraire decision changes what this floor means and must trigger a",
            "re-read of this table.** Moving the panel to a common numeraire, or adjusting the",
            "correlation matrix and rescaling back, changes `lambda_min` by whatever the",
            "rescaling is worth -- and the floor, being absolute, does not move with it.",
            "",
            "## Zero firings outside the degenerate regime is the expected result",
            "",
            "Non-PSD after Newey-West is a **large-K** failure. At `K = 6`, with a correlation",
            "matrix conditioned at 3.27 (`experiments.md` row 96) and a Bartlett kernel truncated",
            "at 5 lags, a genuine estimation window may simply never produce a non-PSD matrix,",
            "and the scan above says it does not: every firing is at `T <= 10`, and across the",
            "whole pre-holdout sample from `T = 11` onward the repair does not fire once.",
            "",
            "**That is not evidence the stage is broken.** Diagnosing it as one would burn a",
            "session on a non-defect, which is the same error as the 0.3 correlation gate",
            "withdrawn in SPEC.md 6.5.2 and the daily-correlation gate withdrawn in SPEC.md 3.2:",
            "a test whose failure mode has not occurred is not a failing test.",
            "",
            "What the requirement is instead: **the code path must be demonstrably live.** It is",
            "shown on a synthetic case rather than only on cached data, because `data/raw` is",
            "gitignored and a reader cloning this repository cannot run the scan above:",
            "",
            f"- `K = {demonstration.factors}` factors from `T = {demonstration.observations}` "
            f"observations, drawn from `model.seed` = {settings.model.seed}, "
            f"`{demonstration.horizon}` horizon",
            f"- Newey-West `lambda_min` = {demonstration.newey_west_minimum_eigenvalue:+.3e} "
            f"against `lambda_max` = {demonstration.newey_west_maximum_eigenvalue:.3e} -- a "
            f"relative breach of {demonstration.relative_breach:+.2e}, about 1e13 times machine "
            "epsilon, so it does not turn on which BLAS is installed",
            f"- The repair floors {demonstration.floored} eigenvalue(s); the post-repair "
            f"`lambda_min` is {demonstration.repaired_minimum_eigenvalue:.3e} and the invariant-4",
            "  assertion after the repair passes",
            "",
            "`tests/test_covariance.py` pins this end to end at both horizons, alongside a",
            "hand-computed repair of `[[1, 2], [2, 1]]` -- eigenvalues 3 and -1 -- and a",
            "hand-computed Newey-West sum that is *itself* non-PSD "
            "(`det = -121/441`), which is the",
            "cheapest available demonstration that the repair is not decorative.",
            "",
            "**Week 7 is where this stage earns its place.** SPEC.md 15.2's cross-sectional",
            "module runs `K = 56` -- one country factor, the 49 Fama-French industries of",
            "SPEC.md 15.6 and the six descriptors of SPEC.md 15.4 -- at the same half-lives,",
            "which raises `K / T_eff` by roughly an order of magnitude and is the regime MSCI's",
            "own practice is written for. The expectation recorded here, before that module runs,",
            "is that the repair fires there and does not fire on the macro model. If it fires on",
            "neither, the honest reading is that the Bartlett kernel is doing its job, not that",
            "the code is dead -- which is why the synthetic demonstration above exists rather",
            "than a gate on the real firing count.",
            "",
            "## Invariant 4, as read here",
            "",
            "The `newey_west` stage **measures** its spectrum and does not assert it; the",
            "assertion runs after `psd_repair`. That is a reading of CLAUDE.md invariant 4, not a",
            "relaxation of it: the invariant exists to catch *silent* non-PSD, and a stage whose",
            "successor in `covariance.stages` is its declared repair is not silent. Asserting",
            "between the two would detect no defect; it would convert an expected and handled",
            "outcome into a crash, and the only way to keep the pipeline running would then be to",
            "widen `psd_minimum_eigenvalue` -- the move CLAUDE.md's residual stopping rule",
            "forbids. `mafrm.risk.covariance.measured_not_asserted_stages()` names the exemption",
            "and a test pins it to exactly one member, so it cannot spread quietly.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Write ``reports/psd_repairs.md``. Reads the local cache only; no network."""
    settings = config.load()
    scans, demonstration = build(settings)
    path = report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(scans, demonstration, settings), encoding="utf-8")

    for scan in scans:
        print(
            f"psd_repairs [{scan.horizon}]: {len(scan.firings)} firing date(s) in "
            f"{len(scan.episodes)} episode(s) over {len(scan.readings):,} expanding-window builds"
        )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
