"""``reports/psd_repairs.md``. SPEC.md 5.2's *"log how often it fires"*.

``build()`` reads the cache and is exercised by ``make report``, not here. What is
pinned here is the arithmetic the report publishes: how firing dates group into
episodes, and the CLAUDE.md failure-mode-9 discipline that stops the date count
being quoted as a rate.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from mafrm.factors.psd_repair_report import Demonstration, Reading, Scan, render


def _reading(day: int, *, fired: bool, minimum: float = -1e-6) -> Reading:
    return Reading(
        date=pd.Timestamp("2008-04-21") + pd.Timedelta(days=day),
        observations=6 + day,
        realised_effective_sample_size=float(6 + day),
        k_over_realised_t_eff=6.0 / (6 + day),
        ewma_minimum_eigenvalue=1e-6,
        newey_west_minimum_eigenvalue=minimum if fired else 1e-6,
        newey_west_maximum_eigenvalue=10.0,
        # The spectrum the repair PRODUCED, added in W4-P2b-fix. SPEC.md 5.2.4's
        # detection threshold is applied to this and not to the Newey-West input,
        # which is the distinction the absolute threshold could not see: a floored
        # eigenvalue beside a lambda_max of 10 is well inside -(size + 2) * eps *
        # lambda_max, while the Newey-West input above it is twelve orders outside.
        psd_repair_minimum_eigenvalue=1e-14 if fired else 1e-6,
        psd_repair_maximum_eigenvalue=10.0,
        repair_fired=fired,
        floored=1 if fired else 0,
    )


def _scan_from(readings: tuple[Reading, ...]) -> Scan:
    return Scan(
        horizon="short",
        factors=6,
        volatility_halflife=84,
        correlation_halflife=504,
        volatility_lags=5,
        correlation_lags=2,
        floor=1e-14,
        panel_start=pd.Timestamp("2008-04-14"),
        readings=readings,
    )


def _scan(pattern: str) -> Scan:
    """``pattern`` is a string of X (fired) and . (did not), one character per date."""
    return _scan_from(tuple(_reading(day, fired=mark == "X") for day, mark in enumerate(pattern)))


def test_consecutive_firings_are_one_episode_not_several() -> None:
    """The whole point of the episode count. CLAUDE.md failure mode 9.

    Five consecutive firing dates on expanding windows are one episode seen five
    times. Counting them as five events -- or as "5 of 4,141 dates" -- is the
    error that has already produced three wrong readings in this project.
    """
    scan = _scan("XXXXX....................")
    assert len(scan.firings) == 5
    assert len(scan.episodes) == 1
    assert scan.episodes[0].dates == 5
    assert scan.episodes[0].start == scan.readings[0].date
    assert scan.episodes[0].end == scan.readings[4].date


def test_separated_firings_are_separate_episodes() -> None:
    """POWER: the grouping can tell one run from three."""
    scan = _scan("XX...X.......XXX...")
    assert len(scan.firings) == 6
    assert [episode.dates for episode in scan.episodes] == [2, 1, 3]


def test_no_firings_is_no_episodes_and_every_reading_is_clean() -> None:
    scan = _scan("..........")
    assert scan.episodes == ()
    assert scan.last_firing_observations is None
    assert scan.clean_readings() == scan.readings


def test_clean_readings_are_the_ones_after_the_last_firing() -> None:
    """The regime a model is actually built in, which is what the report reports."""
    scan = _scan("XXX.......")
    assert scan.last_firing_observations == 8
    assert len(scan.clean_readings()) == 7
    assert all(reading.observations > 8 for reading in scan.clean_readings())


def test_the_episode_reports_its_most_negative_eigenvalue() -> None:
    """*"and by how much"* -- SPEC.md 5.2 asks for the magnitude, not only the count."""
    readings = (
        _reading(0, fired=True, minimum=-1e-6),
        _reading(1, fired=True, minimum=-9e-6),
        _reading(2, fired=True, minimum=-3e-6),
    )
    scan = _scan_from(readings)
    assert scan.episodes[0].most_negative == pytest.approx(-9e-6)


def test_the_relative_breach_is_signed_and_scaled_by_lambda_max() -> None:
    """HAND-COMPUTED. -1e-6 / 10.0 = -1e-7.

    Reported relative to ``lambda_max`` rather than in absolute terms because the
    floor and the eigenvalues are both in the matrix's own units, and those units
    are not fixed until SPEC.md 5.3.1's numeraire decision in W3-P3.
    """
    assert _reading(0, fired=True).relative_breach == pytest.approx(-1e-7)
    assert _reading(0, fired=False).relative_breach == pytest.approx(1e-7)


def test_the_closest_approach_is_the_smallest_positive_margin() -> None:
    scan = _scan("XX........")
    approach = scan.closest_approach
    assert approach.relative_breach > 0.0
    assert approach in scan.clean_readings()


def _demonstration() -> Demonstration:
    return Demonstration(
        observations=12,
        factors=6,
        horizon="short",
        newey_west_minimum_eigenvalue=-7.161e-03,
        newey_west_maximum_eigenvalue=4.749,
        floored=1,
        repaired_minimum_eigenvalue=1.028e-14,
    )


def test_the_report_never_quotes_the_firing_count_as_a_rate() -> None:
    """CLAUDE.md failure mode 9, enforced on the OUTPUT and not only in prose.

    Expanding windows differing by one observation are not independent draws, so
    "5 of 4,141 dates = 0.12%" is not an available statement. This asserts no line
    that mentions firing carries a percentage or the word "rate" -- which is what
    a later session would add without noticing that it is a claim about
    independence.
    """
    rendered = render((_scan("XXXXX" + "." * 40), _scan("XXXXX" + "." * 40)), _demonstration())
    # Whole words: "degenerate" contains "rate", and the disclaimers say "not a
    # rate" on purpose.
    claim = re.compile(r"%|\brates?\b|\bprobabilit", re.IGNORECASE)
    disclaimer = re.compile(r"not a rate|no firing rate", re.IGNORECASE)
    offenders = [
        line
        for line in rendered.splitlines()
        if ("firing" in line.lower() or "fires" in line.lower())
        and claim.search(line)
        and not disclaimer.search(line)
    ]
    assert not offenders, "a firing rate has appeared in the report:\n" + "\n".join(offenders)


def test_the_report_states_the_overlap_and_the_episode_count() -> None:
    """The two things that replace a rate."""
    rendered = render((_scan("XXXXX" + "." * 40), _scan("XXXXX" + "." * 40)), _demonstration())
    assert "failure mode 9" in rendered.lower()
    assert "differing by" in rendered and "one observation" in rendered
    assert "Contiguous episodes: 1" in rendered
    assert "a count of independent draws" in rendered


def test_the_report_quotes_the_floor_against_the_spectrum() -> None:
    """The floor is absolute and therefore scale-dependent. SPEC.md 5.3.1."""
    rendered = render((_scan("XXXXX" + "." * 40), _scan("XXXXX" + "." * 40)), _demonstration())
    assert "orders of magnitude below `lambda_min`" in rendered
    assert "W3-P3" in rendered
    assert "1e-14" in rendered
