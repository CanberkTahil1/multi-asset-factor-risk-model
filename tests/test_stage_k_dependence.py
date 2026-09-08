"""``reports/stage_k_dependence.md`` cannot fall behind ``covariance.stages``.

The file is **maintained by hand**, which is right for it -- it synthesises
findings across three reports and several sessions, and generating it would mean
re-running everything to regenerate prose. But a hand-maintained file has exactly
one failure mode that bites: **a stage gets added and the table does not know.**
The pattern it exists to record is about the pipeline's stages, so a missing stage
is a hole in the argument rather than a stale line.

That failure is mechanical, so it is closed mechanically here rather than by
asking a future session to remember.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mafrm.config import load

_REPORT = Path("reports/stage_k_dependence.md")

#: Every `K = 56` cell is EITHER a forecast made before the equity module
#: existed OR that forecast measured in W7-P4, and must say which. The markers
#: are deliberately not words: a reader skimming the column has to be able to
#: see which cells are predictions and which are results without reading them.
_PREDICTION = "**[P]**"
_MEASURED = "**[M W7-P4]**"


@pytest.fixture(scope="module")
def report() -> str:
    if not _REPORT.exists():  # pragma: no cover - the file is committed
        pytest.fail(f"{_REPORT} is missing; it is committed and carried into W8-P1")
    return _REPORT.read_text(encoding="utf-8")


def _table_rows(report: str) -> list[list[str]]:
    """The stage table's data rows, split on ``|``."""
    rows = []
    for line in report.splitlines():
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 6 and not cells[0].startswith("Stage"):
            rows.append(cells)
    return rows


def test_every_declared_pipeline_stage_has_a_row(report: str) -> None:
    """The guard. A stage in the config with no row here is a hole in the table.

    Matched on the stage name in backticks, exactly as ``covariance.stages``
    spells it, so a rename breaks this too rather than silently orphaning a row.
    """
    missing = [stage for stage in load().model.covariance.stages if f"`{stage}`" not in report]
    assert not missing, (
        f"{_REPORT} has no row for {missing}. The table is hand-maintained and this is the "
        "one way that bites: a stage was added to covariance.stages and the K-dependence "
        "argument silently stopped covering the whole pipeline."
    )


def test_the_table_is_not_empty_and_is_well_formed(report: str) -> None:
    """Guards the guard: a parser that matches nothing would pass the test above."""
    rows = _table_rows(report)
    assert len(rows) >= len(load().model.covariance.stages)


def test_every_k56_claim_is_marked_as_a_prediction(report: str) -> None:
    """Guard 3. Every last-column cell says whether it is a forecast or W7-P4's result.

    Before W7-P4 every non-empty cell was a forecast made before the ``K = 56``
    module existed; W7-P4 measured them and re-marked the measured cells. A cell
    with neither marker is one a reader cannot place, which is the failure.
    """
    unmarked = [
        cells[0]
        for cells in _table_rows(report)
        if cells[-1] and _PREDICTION not in cells[-1] and _MEASURED not in cells[-1]
    ]
    assert not unmarked, (
        f"unmarked K=56 cells in {_REPORT}: {unmarked}. Every claim about K=56 is a "
        f"prediction until week 7 measures it, and must carry {_PREDICTION}."
    )


def test_the_prediction_marker_is_explained_where_it_is_used(report: str) -> None:
    """A marker nobody can decode is decoration."""
    assert re.search(r"\*\*\[P\]\*\*.{0,80}predictions?", report, re.IGNORECASE | re.DOTALL)


def test_the_table_records_where_in_the_spectrum_each_stage_acts(report: str) -> None:
    """The column added after W3-P3b, and why it is not decoration.

    "Earns its keep" and "does nothing" are the wrong axis alone: the eigenfactor
    adjustment earns its keep *on the bulk* and does almost nothing *for the
    dominant factors*, and which matters depends on where the portfolio sits.
    """
    assert "Where in the spectrum it acts" in report
    assert "bulk" in report
    for cells in _table_rows(report):
        assert cells[2], f"row {cells[0]!r} does not say where in the spectrum it acts"
