"""W8-P2's holdout evaluation: the boundary mechanism, the edge rule, and the ledger.

**What these tests can and cannot cover.** The evaluation itself reads the
holdout window, so it is not run here -- a test suite that spent the holdout on
every ``make test`` would spend it a hundred times. What is pinned instead is
everything *around* the crossing: that the boundary moves and restores, that a
nested crossing is refused, that the edge is the minimum of the book's series,
that the frozen configuration refuses to be computed inside the crossing, and
that ``experiments.md``'s counts still reconcile against its own rows.

The one hand-computed test is :func:`test_expected_maximum_sharpe_hand_computed`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pytest

from mafrm import config as config_mod
from mafrm.backtest import holdout as holdout_run
from mafrm.backtest import metrics as metrics_mod
from mafrm.data import holdout as holdout_mod

_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS = _ROOT / "experiments.md"

#: Rows registered for a later session and never run, so they are not evaluated
#: configurations and must not be in the total. Both say so in their own
#: registration: "the count does not move until it runs".
_REGISTERED_NOT_RUN = {100, 118}
#: Rows whose category is stated in prose rather than in a table cell: 164 fixed
#: it at registration ("`data-diagnostic` ... not reassignable"), and 191 is
#: covered by W4-P3's "Every row is `data-diagnostic`" over rows 175-191.
_CATEGORY_IN_PROSE = {164: "data-diagnostic", 191: "data-diagnostic"}

_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(20\d\d-\d\d-\d\d)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|")
_TOTAL = re.compile(
    r"^\|\s*\*\*Total evaluated\*\*\s*\|\s*\*\*(?P<total>\d+)\*\*\s*\|"
    r"\s*\*\*`N`\s*=\s*(?P<n>\d+)\*\*\s*\|"
)


def _row_categories() -> dict[int, str]:
    out: dict[int, str] = {}
    for line in _EXPERIMENTS.read_text(encoding="utf-8").splitlines():
        match = _ROW.match(line)
        if match is None:
            continue
        category = match.group(4).replace("*", "").replace("`", "").strip().split("--")[0].strip()
        out.setdefault(int(match.group(1)), category)
    out.update(_CATEGORY_IN_PROSE)
    return out


# ---------------------------------------------------------------------------
# The boundary mechanism
# ---------------------------------------------------------------------------


def test_the_boundary_moves_inside_the_crossing_and_restores_after() -> None:
    pinned = config_mod.load().require_holdout_start()
    edge = date(2026, 7, 31)
    assert not holdout_mod.is_evaluating_holdout()
    with holdout_mod.evaluating_holdout(edge=edge, reason="test") as yielded:
        assert yielded == edge
        assert holdout_mod.is_evaluating_holdout()
        # Half-open, like calendar.align: the boundary is the day AFTER the edge,
        # so the edge date itself is inside the window.
        assert config_mod.load().require_holdout_start() == edge + timedelta(days=1)
    assert not holdout_mod.is_evaluating_holdout()
    assert config_mod.load().require_holdout_start() == pinned


def test_the_boundary_restores_even_when_the_body_raises() -> None:
    pinned = config_mod.load().require_holdout_start()
    with (
        pytest.raises(RuntimeError, match="boom"),
        holdout_mod.evaluating_holdout(edge=date(2026, 7, 31), reason="test"),
    ):
        raise RuntimeError("boom")
    assert config_mod.load().require_holdout_start() == pinned
    assert not holdout_mod.is_evaluating_holdout()


def test_a_nested_crossing_is_refused() -> None:
    # The nesting is the subject of the test: the outer crossing has to be OPEN
    # before the inner one is attempted, so SIM117's "combine them" would test
    # nothing. Kept nested deliberately.
    with holdout_mod.evaluating_holdout(edge=date(2026, 7, 31), reason="test"):  # noqa: SIM117
        with (
            pytest.raises(holdout_mod.HoldoutError, match="already active"),
            holdout_mod.evaluating_holdout(edge=date(2026, 7, 31), reason="test"),
        ):
            pass
    assert config_mod.load().require_holdout_start() == date(2025, 1, 1)


def test_a_crossing_without_a_reason_is_refused() -> None:
    with (
        pytest.raises(holdout_mod.HoldoutError, match="stated reason"),
        holdout_mod.evaluating_holdout(edge=date(2026, 7, 31), reason="   "),
    ):
        pass


def test_an_edge_before_the_boundary_is_refused() -> None:
    with (
        pytest.raises(holdout_mod.HoldoutError, match="before HOLDOUT_START"),
        holdout_mod.evaluating_holdout(edge=date(2024, 6, 30), reason="test"),
    ):
        pass


def test_the_frozen_configuration_refuses_to_run_inside_the_crossing() -> None:
    """It is what fixes ``TE_target`` and the NAV at their in-sample values."""
    with (
        holdout_mod.evaluating_holdout(edge=date(2026, 7, 31), reason="test"),
        pytest.raises(holdout_run.HoldoutRunError, match="BEFORE the crossing"),
    ):
        holdout_run.frozen_configuration(config_mod.load())


def test_only_the_holdout_runner_opens_the_crossing() -> None:
    """The claim in ``evaluating_holdout``'s docstring, made checkable.

    **Opening** a crossing is entering the context manager. Merely asking whether
    one is open -- ``is_evaluating_holdout()`` -- is the opposite: every caller of
    that predicate in this repository uses it to REFUSE to run inside the
    crossing. So the thing to count is ``with ... evaluating_holdout(``, and
    there must be exactly one, in the runner.

    The first version of this test matched the bare substring
    ``evaluating_holdout(`` and therefore also matched ``is_evaluating_holdout()``
    in ``mafrm.backtest.overfitting``, which refuses the crossing rather than
    opening it. It failed in ``make test`` and the matcher was corrected to test
    the claim the docstring actually makes -- which is stricter, since it no
    longer has to exempt a file by name.
    """
    openers = sorted(
        f"{path.relative_to(_ROOT).as_posix()}:{number}"
        for path in (_ROOT / "src").rglob("*.py")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"\bwith\s+\S*\bevaluating_holdout\s*\(", line)
    )
    assert len(openers) == 1, f"the holdout crossing is opened in more than one place: {openers}"
    assert openers[0].startswith("src/mafrm/backtest/holdout.py:"), openers


# ---------------------------------------------------------------------------
# The right edge
# ---------------------------------------------------------------------------


def test_the_edge_is_the_minimum_of_the_books_series() -> None:
    """Operator ruling (d): a complete cross-section, not a ragged tail."""
    where = holdout_run.evaluation_edge(config_mod.load())
    assert where.edge == min(where.last_observation.values())
    assert where.binding
    for name in where.binding:
        assert where.last_observation[name] == where.edge
    # Every book member is priced, and the support series the run cannot mark
    # a date without are here too.
    assert set(holdout_run._BOOK_SOURCES) <= set(where.last_observation)
    assert set(holdout_run._SUPPORT_SOURCES) <= set(where.last_observation)


def test_the_edge_is_after_the_boundary_so_there_is_a_holdout_to_evaluate() -> None:
    settings = config_mod.load()
    assert holdout_run.evaluation_edge(settings).edge > settings.require_holdout_start()


# ---------------------------------------------------------------------------
# The frozen cell
# ---------------------------------------------------------------------------


def test_the_frozen_cell_is_the_reference_cell() -> None:
    """4D/patient, confirmed by the operator on 2026-09-06 and not settable at run time."""
    cell = holdout_run.FROZEN_CELL
    assert cell.variant == "eigenfactor_a1.0"
    assert cell.treatment == "time_varying"
    assert cell.regime == "patient"
    assert cell.charged and not cell.dense


# ---------------------------------------------------------------------------
# The trial count and the ledger
# ---------------------------------------------------------------------------


def test_the_running_totals_reconcile_against_the_rows() -> None:
    """W8-P2 closed the W6-P1 gap; this is what stops it reopening.

    The category columns must sum to the stated total, and the total must be the
    number of rows that have actually RUN -- every numbered row less the ones
    still registered-and-not-run.
    """
    text = _EXPERIMENTS.read_text(encoding="utf-8")
    totals = [m for line in text.splitlines() if (m := _TOTAL.match(line))]
    assert len(totals) == 1, "exactly one live running-totals line"
    stated_total, stated_n = int(totals[0].group("total")), int(totals[0].group("n"))

    rows = _row_categories()
    highest = max(max(rows), max(_REGISTERED_NOT_RUN))
    assert set(rows) | _REGISTERED_NOT_RUN == set(range(1, highest + 1)), "no gaps, no duplicates"
    assert not (set(rows) & _REGISTERED_NOT_RUN), "a not-run row must carry no category"

    counts = Counter(rows.values())
    assert set(counts) == {"data-diagnostic", "model-config", "strategy-config"}
    assert sum(counts.values()) == stated_total
    assert counts["model-config"] + counts["strategy-config"] == stated_n
    assert metrics_mod.read_trial_count(_EXPERIMENTS) == stated_n


def test_the_trial_count_never_moves_down() -> None:
    """`N` is the one number a project of this shape has an incentive to shrink."""
    assert metrics_mod.read_trial_count(_EXPERIMENTS) >= 70


# ---------------------------------------------------------------------------
# The hand-computed one
# ---------------------------------------------------------------------------


def test_expected_maximum_sharpe_hand_computed() -> None:
    """Bailey & Lopez de Prado's false-strategy bracket, computed by hand at `N` = 70.

    ``E[max SR_N] = sqrt(V) [ (1 - g) Z(1 - 1/N) + g Z(1 - 1/(N e)) ]`` with
    ``g`` the Euler-Mascheroni constant. At the holdout's own inputs --
    ``N`` = 70 and ``V[SR]`` = 8.80830807e-05, the dispersion W6-P2 measured
    across the 28 grid cells -- every step, to ten places:

        1/N              = 0.014285714286
        1 - 1/N          = 0.985714285714
        Z(.)             = 2.1893497555
        N e              = 190.2797279921
        1/(N e)          = 0.005255420588
        1 - 1/(N e)      = 0.994744579412
        Z(.)             = 2.5585542731
        g                = 0.5772156649
        sqrt(V)          = 9.385258691160e-03
        (1 - g) Z(1-1/N) = 0.4227843351 * 2.1893497555 = 0.9256227807
        g Z(1-1/(N e))   = 0.5772156649 * 2.5585542731 = 1.4768376059
        sum              = 2.4024603866
        E[max SR_N]      = 9.385258691160e-03 * 2.4024603866 = 2.254771222366e-02

    which is the 0.0225 the holdout report carries. Written out because a bracket
    the deflated Sharpe is measured against is exactly the number that must not
    be taken on trust from the library that produces it.

    **The first draft of this test asserted ``Z(1 - 1/(N e)) = 2.55868``, which is
    wrong in the fourth decimal** -- ``1/(70 e)`` is 0.005255420588, and the
    2.55868 implies 0.00525491. The test failed, the arithmetic was redone rather
    than the tolerance widened, and the wrong digit is left on the record here
    because that is the direction of correction this file exists to enforce.
    """
    from scipy.stats import norm

    trials, variance = 70, 8.80830807e-05
    gamma = metrics_mod.EULER_MASCHERONI
    z_one = norm.ppf(1.0 - 1.0 / trials)
    z_two = norm.ppf(1.0 - 1.0 / (trials * math.e))
    by_hand = math.sqrt(variance) * ((1.0 - gamma) * z_one + gamma * z_two)

    assert z_one == pytest.approx(2.1893497555, abs=5e-10)
    assert z_two == pytest.approx(2.5585542731, abs=5e-10)
    assert by_hand == pytest.approx(2.254771222366e-02, rel=1e-12)
    assert metrics_mod.expected_maximum_sharpe(
        trials_variance=variance, trials=trials
    ) == pytest.approx(by_hand, rel=1e-12)


# ---------------------------------------------------------------------------
# The three invocations, made checkable rather than asserted
# ---------------------------------------------------------------------------

#: The commit the session started from, and the commit that introduced the
#: holdout evaluation. Both fixed, so this comparison does not drift as the
#: repository grows.
_SESSION_BASE = "9ba15f3"
_SESSION_HEAD = "8f3c6cc"

#: If any of these changed while the holdout was being run, the three
#: invocations would not have been three attempts at ONE evaluation -- they
#: would have been three different models, and the window would be spent three
#: times over. This is the claim W8-P2's disclosure rests on.
_MODEL_PATH = (
    "src/mafrm/risk",
    "src/mafrm/costs",
    "src/mafrm/backtest/optimizer.py",
    "config/model.yaml",
)
#: What the session DID touch outside the runner. Plumbing, every piece of it a
#: no-op unless the holdout runner asks: ``config.load`` gains an ``_OVERRIDE``
#: that is ``None`` everywhere else, ``verification.build_universe`` delegates to
#: an extracted ``build_universe_from`` with identical arithmetic,
#: ``equity_risk_report.run_horizon`` gains a ``history_frame`` defaulting to
#: ``None``, and ``data/holdout.py`` gains the crossing itself. Listed by name
#: because "the runner only" is not literally true and the disclosure says so.
_PLUMBING = (
    "src/mafrm/config.py",
    "src/mafrm/data/holdout.py",
    "src/mafrm/backtest/verification.py",
    "src/mafrm/factors/equity_risk_report.py",
)


def _changed_paths(base: str, head: str) -> list[str]:
    import subprocess

    out = subprocess.run(
        ["git", "diff", "--name-only", base, head],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        pytest.skip(f"git unavailable or commits absent: {out.stderr.strip()}")
    return sorted(line for line in out.stdout.splitlines() if line)


def test_the_holdout_run_changed_nothing_on_the_model_path() -> None:
    """W8-P2's disclosure, made checkable.

    The holdout runner was invoked three times: twice it raised inside the equity
    leg on defects in the runner, and the third completed. **There is no git diff
    *between* those invocations** -- they were successive uncommitted working-tree
    states and no commit was made until the session ended -- so the check that can
    be made is the stronger one: across the ENTIRE session, from the commit it
    started at to the commit that introduced the evaluation, **not one file on the
    model path changed at all**. If nothing moved across the whole session, nothing
    moved between the invocations either.

    What that buys: the three invocations were three attempts at one evaluation of
    one frozen model, not three evaluations of three models. What it does not buy:
    it is evidence about the code, not about what was read, and the reason the
    window is intact is the operator's ruling that what spends a holdout is
    information reaching the decision-maker. The first two invocations raised
    before anything was printed, rendered or written.
    """
    changed = _changed_paths(_SESSION_BASE, _SESSION_HEAD)
    assert changed, "the session commit touched nothing, which cannot be right"
    offending = [
        path
        for path in changed
        if any(path == entry or path.startswith(f"{entry}/") for entry in _MODEL_PATH)
    ]
    assert offending == [], (
        "the holdout ran while the model path changed, so the three invocations were "
        f"not three attempts at one evaluation: {offending}"
    )


def test_the_disclosure_lists_every_source_file_the_session_touched() -> None:
    """ "The runner only" is not literally true, and the record must not say it is."""
    changed = _changed_paths(_SESSION_BASE, _SESSION_HEAD)
    source = [p for p in changed if p.startswith("src/")]
    expected = sorted(("src/mafrm/backtest/holdout.py", *_PLUMBING))
    assert source == expected, (
        "a source file changed that the W8-P2 disclosure does not name; either the "
        f"disclosure is now incomplete or something moved that should not have: {source}"
    )
