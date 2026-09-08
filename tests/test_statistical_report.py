"""``reports/statistical_factors.md`` and the SPEC.md 15.2 acceptance.

The test that matters here is
:func:`test_model_b_runs_through_the_unchanged_risk_pipeline` -- ``experiments.md``
row 125. Everything else guards the committed artefacts: that the cached forecast
history still describes the config it was built against, and that nothing in any
of them reaches the holdout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mafrm.config import load
from mafrm.factors import statistical_report as sr

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def settings() -> object:
    return load()


def test_the_committed_cache_exists() -> None:
    """It is a build artefact of a forty-minute run and is committed for that reason."""
    assert sr.CACHE_PATH.exists(), (
        f"{sr.CACHE_PATH} is missing. Rebuild with "
        "`python -m mafrm.factors.statistical_report --rebuild` (about ninety minutes)."
    )


def test_the_cached_forecast_history_still_describes_the_current_config() -> None:
    """The staleness guard, mirroring ``test_regime.py``'s for Model A.

    A committed cache is numbers about a model. Change a half-life, a lag count
    or the eigenfactor trial count and every one of them describes a model that no
    longer exists, with nothing in the file to say so. The digest is over every
    ``RiskConfig`` field at both horizons, so any of those edits breaks this test
    rather than silently republishing.
    """
    header: list[str] = []
    with sr.CACHE_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            header.append(line[1:].strip())
    provenance = json.loads("\n".join(header))
    assert provenance["risk_config_digest"] == sr.risk_config_digest(), (
        "reports/statistical_forecast_history.csv was built against a different RiskConfig. "
        "Rebuild it with `python -m mafrm.factors.statistical_report --rebuild`."
    )


@pytest.mark.dataset
def test_the_cache_still_describes_the_current_factor_panel() -> None:
    """The digest the ``RiskConfig`` one does not cover, added after it was needed.

    W3-P5 changed the factor extraction while every value in ``config/model.yaml``
    stayed identical, which moved the factor series and left the risk-config
    digest matching. This is the guard for that class: it rebuilds both panels and
    compares their digests against the ones the cache recorded.
    """
    settings = load()
    panels = sr._panels(settings)
    sr.read_cache(panels=panels)


def test_every_declared_column_is_present_in_the_cache() -> None:
    """Both variants x both horizons x both published ``a``. Eight, not six."""
    cache = sr.read_cache()
    for column in sr.columns():
        assert column.name in cache.frame.columns, f"{column.name} missing from the cache"
    assert len(sr.columns()) == 8


@pytest.mark.dataset
def test_model_b_runs_through_the_unchanged_risk_pipeline() -> None:
    """``experiments.md`` row 125. The SPEC.md 15.2 acceptance.

    Model B's factors, all five covariance stages, both horizons, both published
    eigenfactor ``a``, both variants. Every stage's matrix must come back positive
    semi-definite and every build must be complete.

    **The claim is architectural, not numerical.** If this ever fails, the first
    question is not "which eigenvalue went negative" -- it is whether something
    under ``src/mafrm/risk/`` has acquired an assumption about what its columns
    are. ``tests/test_risk_architecture.py`` asserts that package cannot import
    from ``mafrm.factors`` at all, and this asserts the consequence: a second,
    structurally unrelated factor set goes through it unchanged.
    """
    results = sr.acceptance()
    assert len(results) == 8
    failures = [result.column.name for result in results if not result.passed]
    assert not failures, (
        f"SPEC.md 15.2's early warning has fired at {failures}. That is an architecture "
        "problem, not a numerical one -- do not patch it at the call site."
    )
    for result in results:
        assert len(result.stages) == 5, f"{result.column.name} ran {len(result.stages)}/5 stages"
        assert result.factors >= 2
        assert result.lambda_squared > 0.0


def test_the_report_and_its_artefacts_are_committed() -> None:
    for path in (sr.report_path(), sr.figure_path(), sr.diagnostics_path()):
        assert path.exists(), f"{path} is missing; reports/ is committed (CLAUDE.md)"


def test_nothing_committed_reaches_the_holdout() -> None:
    """CLAUDE.md invariant 5, asserted on every dated artefact this module writes."""
    boundary = pd.Timestamp(load().require_holdout_start())
    for path in (sr.CACHE_PATH, sr.diagnostics_path()):
        frame = pd.read_csv(path, comment="#")
        dates = pd.to_datetime(frame["date"])
        assert dates.max() < boundary, f"{path} reaches {dates.max()}, on or after {boundary}"


def test_the_report_states_that_n_is_thirteen_and_not_fifteen() -> None:
    """SPEC.md 4.2.2. A reader must not have to know the discrepancy to spot it."""
    text = sr.report_path().read_text(encoding="utf-8")
    assert "N = 13" in text
    assert "15" in text
    assert "void" in text.lower()


def test_the_report_names_the_sklearn_ledoit_wolf_trap() -> None:
    """SPEC.md 4.2 asks for this to be said out loud, and a knowledgeable reader checks."""
    text = sr.report_path().read_text(encoding="utf-8")
    assert "sklearn.covariance.LedoitWolf" in text
    assert "IDENTITY" in text or "identity" in text


def test_the_report_frames_the_comparand_result_as_a_regime_finding() -> None:
    """The comparand table is the most misreadable thing this report contains.

    On its face it shows the specified estimator losing to the benchmark it was
    meant to beat, and a reader who stops there concludes Model B is a bad model.
    It is a statement about `N/T`: MP denoising is a high-dimensional technique
    and at 0.0029 there is no noise left for it to separate, which row 127
    confirms by flipping the sign at 0.4. If the report ever stops saying that,
    the table is worse than not publishing it.
    """
    text = sr.report_path().read_text(encoding="utf-8")
    assert "high-dimensional" in text
    assert "regime" in text
    assert "actively harmful" in text


def test_the_report_says_the_factor_count_is_a_near_tie() -> None:
    """`K = 3` on every date is not the same as a clean separation, and says so."""
    text = sr.report_path().read_text(encoding="utf-8")
    assert "near-tie" in text
    assert "not a\nrobust count" in text or "not a robust count" in text


def test_the_report_carries_the_no_switching_pre_registration() -> None:
    """The clause is the only thing making row 126's category defensible.

    If the report ever stops carrying it, a reader has a table showing a comparand
    beating Model B and nothing saying why Model B did not change.
    """
    text = sr.report_path().read_text(encoding="utf-8")
    assert "comparands, not candidates" in text
    assert "model-config" in text
