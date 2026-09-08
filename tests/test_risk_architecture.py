"""CLAUDE.md invariant 10, asserted mechanically. SPEC.md 15.2.

SPEC.md 15.2 is a bet with a stated payoff: if week 3 is written against a
generic interface, the week-7 cross-sectional module is a new file under
``factors/`` and nothing in ``risk/``, ``costs/`` or ``backtest/`` changes -- and
week 7 costs one week instead of three. Everything here exists to make that bet
checkable now rather than discoverable in week 7.

The import direction is the half that will actually be violated. Nobody writes
``if asset_class == ...`` on purpose; what happens is that a helper already
exists one package over and reaching for it looks free. So the edge is asserted
directly, and against the source text rather than against the import graph,
because a lazily-imported module would not show up in ``sys.modules``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_RISK = Path(__file__).resolve().parents[1] / "src" / "mafrm" / "risk"

#: The acceptance grep from the W3-P1 ruling, verbatim.
_FORBIDDEN_WORDS = re.compile(r"equity|macro|ticker|asset_class", re.IGNORECASE)

_FORBIDDEN_IMPORTS = re.compile(
    r"^\s*(from\s+mafrm\.factors|import\s+mafrm\.factors)", re.MULTILINE
)


def _sources() -> list[Path]:
    files = sorted(_RISK.rglob("*.py"))
    assert files, f"no python files under {_RISK}"
    return files


@pytest.mark.parametrize("path", _sources(), ids=lambda p: p.name)
def test_no_source_file_in_risk_names_an_asset_class(path: Path) -> None:
    """``grep -ri "equity|macro|ticker|asset_class" src/mafrm/risk/`` must be empty.

    Comments and docstrings count. A module whose prose explains what it does
    "for the equity model" has an assumption in it even when the code does not,
    and the next person to edit it will act on the prose.
    """
    hits = [
        f"{path.name}:{number}: {line.strip()}"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if _FORBIDDEN_WORDS.search(line)
    ]
    assert not hits, "mafrm.risk must not know what asset class it sees:\n" + "\n".join(hits)


@pytest.mark.parametrize("path", _sources(), ids=lambda p: p.name)
def test_no_source_file_in_risk_imports_from_factors(path: Path) -> None:
    """The corollary of invariant 10, and the one that gets violated first.

    ``mafrm.risk`` imports from nothing above it. Shared numerics live in
    ``mafrm.numerics``, beneath both packages, so there is no edge between
    ``risk`` and ``factors`` in either direction.
    """
    text = path.read_text(encoding="utf-8")
    assert not _FORBIDDEN_IMPORTS.search(text), (
        f"{path.name} imports from mafrm.factors. Move the shared piece into "
        "mafrm.numerics instead -- see the module docstring there for why."
    )


def test_risk_imports_only_from_beneath_it() -> None:
    """Positive form of the same rule: name the packages ``risk`` may reach for."""
    permitted = {"mafrm.config", "mafrm.numerics", "mafrm.risk"}
    offenders: list[str] = []
    for path in _sources():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            match = re.match(r"^\s*(?:from|import)\s+(mafrm[\w.]*)", line)
            if match is None:
                continue
            module = match.group(1)
            if not any(module == root or module.startswith(root + ".") for root in permitted):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, (
        "mafrm.risk may import only from "
        + ", ".join(sorted(permitted))
        + ":\n"
        + "\n".join(offenders)
    )


def test_the_shared_ewma_helper_lives_beneath_both_packages() -> None:
    """Moved out of ``factors`` rather than duplicated, and there is exactly one.

    ``betas`` is the remaining direct consumer in ``factors``. ``macro`` stopped
    needing it in W3-P1b, when ``condition_numbers`` was re-based onto the
    pipeline's own estimators -- so the invariant is not "both packages import
    the helper" but "nobody has a private copy of it".
    """
    from mafrm.factors import betas
    from mafrm.numerics import ewma_weights
    from mafrm.risk import covariance

    assert betas.ewma_weights is ewma_weights
    assert covariance.ewma_weights is ewma_weights

    factors = Path(__file__).resolve().parents[1] / "src" / "mafrm" / "factors"
    for name in ("macro.py", "betas.py"):
        source = (factors / name).read_text()
        assert "def _ewma_weights" not in source, f"{name}: a local copy is back"
        assert "def _effective_sample_size" not in source, f"{name}: a local copy is back"


def test_factors_may_import_risk_but_never_the_reverse() -> None:
    """The allowed direction, asserted so the re-basing cannot be read as symmetric.

    W3-P1b pointed ``macro.condition_numbers`` at ``mafrm.risk.covariance`` so that
    one statistic has one implementation. That edge is fine -- ``risk`` sits
    beneath ``factors`` in the dataflow. The reverse edge is the one invariant 10
    forbids, and it is asserted separately above for every file in ``risk``.
    """
    from mafrm.factors import macro
    from mafrm.risk import covariance

    assert macro.ewma_second_moment is covariance.ewma_second_moment
    assert macro.correlation_from_covariance is covariance.correlation_from_covariance
