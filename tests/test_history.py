"""The shared forecast-history builder. ``mafrm.history``.

**These are the tests that would have caught the divergence.** Before W3-P6 the
shape existed three times: ``vra_report`` (W3-P4), ``statistical_report``
(W3-P5), ``hybrid_report`` (W3-P6). Nothing asserted that the three computed the
same digest from the same inputs, and by the time it was checked they did not --
the newest copy used ``vars()`` where the others used ``asdict()``, skipped their
tuple normalisation, and hashed the panel in a different order.

So the first test here is the one that was missing: every caller's digest
functions are the same object. It is deliberately an identity check rather than
a value comparison, because a value comparison passes for as long as two
implementations happen to agree and stops meaning anything the moment one is
edited.

The second group is the constraint that makes the shared module hard to change
carelessly: **the three committed caches must still validate.** Each cost between
twelve minutes and an hour and a half to rebuild, and a silent digest change
would invalidate all three at once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mafrm import history
from mafrm.factors import hybrid_report, statistical_report, vra_report


@dataclass(frozen=True)
class _Variant:
    """A minimal :class:`mafrm.history.Variant`. One member, as the protocol says."""

    name: str


def _frame(seed: int = 0, columns: tuple[str, ...] = ("a", "b")) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=40, name="date")
    return pd.DataFrame(
        {name: generator.standard_normal(len(index)) for name in columns}, index=index
    )


# ---------------------------------------------------------------------------
# The test that was missing
# ---------------------------------------------------------------------------


def test_every_caller_uses_the_same_digest_functions() -> None:
    """Identity, not equality. THE regression test for the W3-P6 divergence.

    A value comparison would pass for as long as two implementations happened to
    agree and would stop meaning anything the moment one was edited -- which is
    exactly the failure this replaces. Asserting the callers hold the *same
    object* cannot drift.
    """
    for module in (vra_report, statistical_report, hybrid_report):
        assert module.risk_config_digest is history.risk_config_digest, module.__name__
        assert module.panel_digest is history.panel_digest, module.__name__
        assert module._git_commit is history.git_commit, module.__name__


def test_every_caller_shares_the_cache_type() -> None:
    """One ``Cache``, so ``bias()``'s NaN handling cannot differ between callers.

    It differed before: ``statistical_report``'s trimmed the index union's
    padding and refused interior gaps, ``vra_report``'s was a bare
    ``to_numpy()``. The strict one is now everyone's.
    """
    for module in (vra_report, statistical_report, hybrid_report):
        assert module.Cache is history.Cache, module.__name__


def test_the_three_committed_caches_still_validate() -> None:
    """The constraint that makes :mod:`mafrm.history` hard to change carelessly.

    Each of these cost between twelve minutes and an hour and a half to build. A
    change to either digest function invalidates all three at once, and this is
    where that shows up rather than in a report that silently refuses to render.
    """
    for read in (vra_report.read_cache, statistical_report.read_cache, hybrid_report.read_cache):
        cache = read()
        assert cache.provenance["risk_config_digest"] == history.risk_config_digest()
        assert cache.provenance["rows"] == len(cache.frame)
        assert len(cache.provenance["columns"]) >= 6


# ---------------------------------------------------------------------------
# The primitives
# ---------------------------------------------------------------------------


def test_a_partial_round_trips_a_float64_exactly(tmp_path: Path) -> None:
    """17 significant digits in, the identical bits out.

    ``statistical_report`` found this first: at 12 digits a resumed assembly
    differed from an unbroken one in the last ulp, which would make the committed
    cache depend on whether a run was interrupted. The writer and the
    ``round_trip`` reader only work as a pair, so both are asserted here.
    """
    awkward = np.array([0.1, 1.0 / 3.0, np.pi, 1e-17, 0.30000000000000004])
    frame = pd.DataFrame(
        {"bias": awkward}, index=pd.bdate_range("2021-01-01", periods=len(awkward), name="date")
    )
    path = tmp_path / "one.csv"
    history.write_partial(path, frame, {"risk_config_digest": "x", "panel_digest": "y"})
    back = history.read_partial(path)
    assert np.array_equal(back["bias"].to_numpy(dtype=float), awkward)
    assert back.attrs["risk_config_digest"] == "x"


def test_a_partial_with_stale_digests_is_discarded_not_reused(tmp_path: Path) -> None:
    """A resume that could splice two models together is worse than no resume."""
    path = tmp_path / "one.csv"
    frame = pd.DataFrame(
        {"bias": [1.0, 2.0]}, index=pd.bdate_range("2021-01-01", periods=2, name="date")
    )
    history.write_partial(path, frame, {"risk_config_digest": "old", "panel_digest": "old"})

    calls: list[int] = []

    def compute() -> pd.DataFrame:
        calls.append(1)
        return pd.DataFrame(
            {"bias": [9.0, 9.0]}, index=pd.bdate_range("2021-01-01", periods=2, name="date")
        )

    same = history.load_or_build(
        path, {"risk_config_digest": "old", "panel_digest": "old"}, compute
    )
    assert calls == [], "a matching partial must be returned without recomputing"
    assert same["bias"].tolist() == [1.0, 2.0]

    rebuilt = history.load_or_build(
        path, {"risk_config_digest": "new", "panel_digest": "old"}, compute
    )
    assert calls == [1], "a stale partial must be discarded and rebuilt"
    assert rebuilt["bias"].tolist() == [9.0, 9.0]
    assert history.read_partial(path).attrs["risk_config_digest"] == "new"


def test_the_panel_digest_moves_when_the_series_moves() -> None:
    """The guard that catches what ``config/model.yaml`` cannot. W3-P5's failure."""
    base = _frame()
    assert history.panel_digest(base) == history.panel_digest(base.copy())

    moved = base.copy()
    moved.iloc[0, 0] += 1e-12
    assert history.panel_digest(moved) != history.panel_digest(base)

    renamed = base.rename(columns={"a": "z"})
    assert history.panel_digest(renamed) != history.panel_digest(base)

    shifted = base.copy()
    shifted.index = shifted.index + pd.Timedelta(days=7)
    assert history.panel_digest(shifted) != history.panel_digest(base)


def test_a_stale_cache_is_refused_on_read(tmp_path: Path) -> None:
    """The message says STALE, which is what ``tests/test_regime.py`` matches."""
    path = tmp_path / "cache.csv"
    frame = pd.DataFrame(
        {"bias_x": [1.0, 2.0]}, index=pd.bdate_range("2021-01-01", periods=2, name="date")
    )
    history.write_cache(
        history.Cache(provenance={"risk_config_digest": "not-the-current-one"}, frame=frame), path
    )
    with pytest.raises(ValueError, match="is STALE"):
        history.read_cache(path, rebuild_hint="rebuild it")


def test_a_moved_panel_is_refused_on_read(tmp_path: Path) -> None:
    """And the panel guard fires separately, with its own message."""
    path = tmp_path / "cache.csv"
    frame = pd.DataFrame(
        {"bias_x": [1.0, 2.0]}, index=pd.bdate_range("2021-01-01", periods=2, name="date")
    )
    provenance = {
        "risk_config_digest": history.risk_config_digest(),
        "panel_digest": {"one": "recorded"},
    }
    history.write_cache(history.Cache(provenance=provenance, frame=frame), path)

    history.read_cache(path, rebuild_hint="x", panel_digests={"one": "recorded"})
    with pytest.raises(ValueError, match="factor series itself moved"):
        history.read_cache(path, rebuild_hint="x", panel_digests={"one": "different"})


def test_a_single_string_panel_digest_is_understood(tmp_path: Path) -> None:
    """``vra_report``'s variants share one panel, so its digest is a bare string.

    Both forms are carried in the header as written and both are checked. A
    reader that only understood the map form would silently skip the guard on the
    oldest of the three caches, which is the one ``make model`` reads on every
    run.
    """
    path = tmp_path / "cache.csv"
    frame = pd.DataFrame(
        {"bias_x": [1.0]}, index=pd.bdate_range("2021-01-01", periods=1, name="date")
    )
    provenance = {
        "risk_config_digest": history.risk_config_digest(),
        "panel_digest": "a-single-digest",
    }
    history.write_cache(history.Cache(provenance=provenance, frame=frame), path)

    history.read_cache(path, rebuild_hint="x", panel_digests={"only": "a-single-digest"})
    with pytest.raises(ValueError, match="factor series itself moved"):
        history.read_cache(path, rebuild_hint="x", panel_digests={"only": "something-else"})


def test_the_bias_accessor_trims_padding_and_refuses_interior_gaps() -> None:
    """Leading and trailing NaN are alignment; a hole in the middle is a defect."""
    index = pd.bdate_range("2021-01-01", periods=6, name="date")
    padded = pd.DataFrame({"v": [np.nan, 1.0, 2.0, 3.0, np.nan, np.nan]}, index=index)
    assert history.Cache(provenance={}, frame=padded).bias(_Variant("v")).tolist() == [
        1.0,
        2.0,
        3.0,
    ]

    holed = pd.DataFrame({"v": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0]}, index=index)
    with pytest.raises(ValueError, match="interior gap"):
        history.Cache(provenance={}, frame=holed).bias(_Variant("v"))

    empty = pd.DataFrame({"v": [np.nan] * 6}, index=index)
    with pytest.raises(ValueError, match="entirely missing"):
        history.Cache(provenance={}, frame=empty).bias(_Variant("v"))


def test_assemble_refuses_a_column_the_partials_do_not_carry() -> None:
    """A silently missing column would become a KeyError somewhere far away."""
    from mafrm.config import load

    index = pd.bdate_range("2021-01-01", periods=3, name="date")
    piece = pd.DataFrame({"bias_a": [1.0, 2.0, 3.0]}, index=index)
    with pytest.raises(ValueError, match="do not carry"):
        history.assemble(
            [piece],
            order=["bias_a", "bias_missing"],
            what="test",
            settings=load(),
            panel_digests={"one": "d"},
            variants=[_Variant("bias_a")],
        )


def test_the_provenance_header_carries_the_same_keys_everywhere() -> None:
    """One header shape, plus whatever ``extra`` a caller adds."""
    from mafrm.config import load

    settings = load()
    header = history.cache_provenance(
        what="test",
        settings=settings,
        panel_digests={"one": "d"},
        variants=[_Variant("bias_a")],
        rows=3,
        extra={"model_yaml_sha256": "abc"},
    )
    for key in ("_what", "generated_utc", "git_commit", "risk_config_digest", "panel_digest"):
        assert key in header
    assert header["rows"] == 3
    assert header["columns"] == ["bias_a"]
    assert header["model_yaml_sha256"] == "abc"
    assert json.dumps(header, sort_keys=True)
