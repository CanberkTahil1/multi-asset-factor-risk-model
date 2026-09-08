"""Cache and manifest tests.

Nothing here touches the network: the ``block_network`` autouse fixture in
conftest fails any unmarked test that opens a socket, and every fetch in this
file is a fake closure. The cache module itself imports no network library, so
the only way data reaches it is the injected fetcher -- that seam is what makes
CLAUDE.md invariant 1 mechanical.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from mafrm.data import cache
from mafrm.data.cache import (
    CacheDriftError,
    CacheError,
    Fetched,
    Manifest,
    _main,
    check,
    fetch_raw,
    read_unrestricted,
    store_processed,
    store_raw,
    verify,
)

URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
MOMENT = datetime(2026, 8, 25, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the whole module at a throwaway data tree."""
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("MAFRM_DATA_DIR", str(root))
    cache._writer_fingerprint.cache_clear()
    yield root


#: Why the storage-layer tests below read past the holdout boundary. They fix a
#: synthetic frame in 2026 and ask whether bytes round-trip, which is a question
#: about the cache and not about the model -- exactly the case
#: `cache.read_unrestricted` exists for. `test_read_stops_at_the_holdout_boundary`
#: pins the other half.
_STORAGE = "storage-layer round trip on a synthetic frame; no model output involved"


def frame(n: int = 5, value: float = 1.0) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=n, freq="B", name="date")
    return pd.DataFrame({"beta0": [value] * n, "tau1": [2.0] * n}, index=index)


def fake_fetcher(payload: pd.DataFrame, source_bytes: bytes | None = None) -> cache.Fetcher:
    """A loader stand-in. Records how many times it was called."""

    def fetch() -> Fetched:
        fetch.calls += 1  # type: ignore[attr-defined]
        return Fetched(frame=payload, source_bytes=source_bytes)

    fetch.calls = 0  # type: ignore[attr-defined]
    return fetch


# ---------------------------------------------------------------------------
# Hashing -- published test vectors, hand-checkable
# ---------------------------------------------------------------------------


def test_sha256_matches_published_vectors() -> None:
    # FIPS 180-4 / NIST CSRC published SHA-256 vectors. These are the anchor for
    # every digest in the manifest: if this fails nothing else here means anything.
    assert cache._sha256(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert (
        cache._sha256(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


# ---------------------------------------------------------------------------
# Layout and the manifest entry
# ---------------------------------------------------------------------------


def test_store_raw_writes_the_expected_path_and_entry(data_root: Path) -> None:
    result = store_raw(
        source="fed",
        name="gsw",
        url=URL,
        frame=frame(),
        pull_date=date(2026, 8, 25),
        now=MOMENT,
    )

    assert result.action == "written"
    assert result.entry.path == "raw/fed/gsw_2026-08-25.parquet"
    assert (data_root / "raw" / "fed" / "gsw_2026-08-25.parquet").is_file()

    entry = result.entry
    assert entry.url == URL
    assert entry.downloaded_at == "2026-08-25T12:00:00Z"
    assert entry.rows == 5
    assert entry.first_date == "2026-01-01"
    assert entry.last_date == "2026-01-07"  # five business days from Thursday 1 Jan
    assert entry.sha256 == cache._sha256(result.path.read_bytes())
    assert entry.size_bytes == result.path.stat().st_size
    assert entry.source_sha256 is None


def test_manifest_is_written_and_reloads_identically(data_root: Path) -> None:
    store_raw(source="fed", name="gsw", url=URL, frame=frame(), now=MOMENT)

    manifest_file = data_root / "manifest.json"
    assert manifest_file.is_file()
    raw = json.loads(manifest_file.read_text())
    assert raw["schema_version"] == 1
    assert list(raw["entries"]) == ["raw/fed/gsw_2026-08-25.parquet"]

    reloaded = Manifest.load()
    assert len(reloaded) == 1
    assert reloaded.get("raw/fed/gsw_2026-08-25.parquet") is not None
    # Round-trips byte for byte, so an unchanged re-run leaves an empty git diff.
    assert reloaded.render() == manifest_file.read_text()


def test_manifest_render_is_sorted_and_deterministic() -> None:
    store_raw(source="fed", name="tips", url=URL, frame=frame(), now=MOMENT)
    store_raw(source="fed", name="gsw", url=URL, frame=frame(3), now=MOMENT)
    rendered = Manifest.load().render()
    assert list(json.loads(rendered)["entries"]) == [
        "raw/fed/gsw_2026-08-25.parquet",
        "raw/fed/tips_2026-08-25.parquet",
    ]
    assert rendered.endswith("}\n")
    assert Manifest.load().render() == rendered


def test_source_bytes_are_hashed_separately(data_root: Path) -> None:
    result = store_raw(
        source="fed", name="gsw", url=URL, frame=frame(), source_bytes=b"abc", now=MOMENT
    )
    # Upstream provenance, independent of any serializer version.
    assert result.entry.source_sha256 == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert result.entry.source_sha256 != result.entry.sha256


def test_rejects_path_unsafe_tokens() -> None:
    with pytest.raises(CacheError, match="not a valid path component"):
        store_raw(source="../etc", name="gsw", url=URL, frame=frame(), now=MOMENT)
    with pytest.raises(CacheError, match="not a valid path component"):
        store_raw(source="fed", name="a/b", url=URL, frame=frame(), now=MOMENT)


def test_rejects_an_empty_frame_unless_asked() -> None:
    empty = pd.DataFrame({"beta0": []}, index=pd.DatetimeIndex([], name="date"))
    with pytest.raises(CacheError, match="silently broken loader"):
        store_raw(source="fed", name="gsw", url=URL, frame=empty, now=MOMENT)

    result = store_raw(source="fed", name="gsw", url=URL, frame=empty, now=MOMENT, allow_empty=True)
    assert result.entry.rows == 0
    assert result.entry.first_date is None


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_identical_refetch_does_not_rewrite(data_root: Path) -> None:
    first = store_raw(
        source="fed", name="gsw", url=URL, frame=frame(), pull_date=date(2026, 8, 25), now=MOMENT
    )
    mtime = first.path.stat().st_mtime_ns

    later = datetime(2026, 9, 1, 9, 30, tzinfo=UTC)
    second = store_raw(
        source="fed", name="gsw", url=URL, frame=frame(), pull_date=date(2026, 9, 1), now=later
    )

    assert second.action == "unchanged"
    assert second.entry.path == first.entry.path
    # No second dated copy, and the original download timestamp is preserved --
    # the manifest records when the data was obtained, not when it was rechecked.
    assert second.entry.downloaded_at == "2026-08-25T12:00:00Z"
    assert first.path.stat().st_mtime_ns == mtime
    assert len(Manifest.load()) == 1
    assert sorted(p.name for p in (data_root / "raw" / "fed").iterdir()) == [
        "gsw_2026-08-25.parquet"
    ]


def test_changed_data_writes_a_new_dated_file(data_root: Path) -> None:
    store_raw(
        source="fed", name="gsw", url=URL, frame=frame(), pull_date=date(2026, 8, 25), now=MOMENT
    )
    later = datetime(2026, 9, 1, 9, 30, tzinfo=UTC)
    second = store_raw(
        source="fed",
        name="gsw",
        url=URL,
        frame=frame(6),
        pull_date=date(2026, 9, 1),
        now=later,
    )

    assert second.action == "written"
    assert second.entry.path == "raw/fed/gsw_2026-09-01.parquet"
    assert len(Manifest.load()) == 2
    assert Manifest.load().latest(source="fed", name="gsw").path == second.entry.path


def test_same_day_revision_overwrites_that_day(data_root: Path) -> None:
    pull = date(2026, 8, 25)
    store_raw(source="fed", name="gsw", url=URL, frame=frame(), pull_date=pull, now=MOMENT)
    revised = store_raw(
        source="fed", name="gsw", url=URL, frame=frame(value=2.0), pull_date=pull, now=MOMENT
    )

    assert revised.action == "written"
    assert len(Manifest.load()) == 1
    assert read_unrestricted(revised.entry, reason=_STORAGE)["beta0"].iloc[0] == 2.0


def test_a_corrupted_file_is_repaired_by_a_refetch(data_root: Path) -> None:
    first = store_raw(source="fed", name="gsw", url=URL, frame=frame(), now=MOMENT)
    first.path.write_bytes(b"not parquet")

    second = store_raw(source="fed", name="gsw", url=URL, frame=frame(), now=MOMENT)

    assert second.action == "repaired"
    assert second.entry.sha256 == first.entry.sha256
    assert verify().ok


def test_fetch_raw_fetches_once_and_delegates(data_root: Path) -> None:
    fetcher = fake_fetcher(frame(), source_bytes=b"abc")
    result = fetch_raw(source="fed", name="gsw", url=URL, fetcher=fetcher, now=MOMENT)
    assert result.action == "written"
    assert fetcher.calls == 1  # type: ignore[attr-defined]

    again = fetch_raw(source="fed", name="gsw", url=URL, fetcher=fetcher, now=MOMENT)
    # The fetch always happens -- you cannot hash a payload you did not download --
    # but the disk write does not.
    assert fetcher.calls == 2  # type: ignore[attr-defined]
    assert again.action == "unchanged"


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_read_round_trips_the_frame() -> None:
    original = frame()
    result = store_raw(source="fed", name="gsw", url=URL, frame=original, now=MOMENT)
    # check_freq=False: see test_round_trip_drops_the_index_frequency below.
    pd.testing.assert_frame_equal(
        read_unrestricted(result.entry, reason=_STORAGE), original, check_freq=False
    )
    pd.testing.assert_frame_equal(
        read_unrestricted(result.entry.path, reason=_STORAGE), original, check_freq=False
    )


def test_round_trip_drops_the_index_frequency() -> None:
    """Parquet stores index *values*, not a pandas ``freq`` attribute.

    Pinned rather than worked around. A ``freq`` on a market calendar is a claim
    the data cannot support once holidays exist, and silently reacquiring one
    through the cache is how CLAUDE.md failure mode 1 starts. Anything needing a
    calendar must state it explicitly rather than inherit it from a cached frame.
    """
    original = frame()
    assert original.index.freq is not None
    result = store_raw(source="fed", name="gsw", url=URL, frame=original, now=MOMENT)

    restored = read_unrestricted(result.entry, reason=_STORAGE)
    assert isinstance(restored.index, pd.DatetimeIndex)
    assert restored.index.freq is None
    assert restored.index.name == "date"
    assert list(restored.index) == list(original.index)


def test_read_refuses_a_drifted_file(data_root: Path) -> None:
    result = store_raw(source="fed", name="gsw", url=URL, frame=frame(), now=MOMENT)
    result.path.write_bytes(result.path.read_bytes() + b"\x00")
    with pytest.raises(CacheError, match="does not match the manifest"):
        read_unrestricted(result.entry, reason=_STORAGE)


def test_latest_raises_when_nothing_is_cached() -> None:
    with pytest.raises(CacheError, match="run `make data`"):
        Manifest.load().latest(source="fed", name="gsw")


# ---------------------------------------------------------------------------
# Processed
# ---------------------------------------------------------------------------


def test_processed_records_its_inputs_and_is_undated(data_root: Path) -> None:
    raw = store_raw(source="fed", name="gsw", url=URL, frame=frame(), now=MOMENT)
    derived = store_processed(
        name="govt_returns", frame=frame(4), inputs=[raw.entry.path], now=MOMENT
    )

    assert derived.entry.path == "processed/govt_returns.parquet"
    assert derived.entry.stage == "processed"
    assert derived.entry.url == "derived:govt_returns"
    assert derived.entry.inputs == (raw.entry.path,)
    assert (data_root / "processed" / "govt_returns.parquet").is_file()

    again = store_processed(
        name="govt_returns", frame=frame(4), inputs=[raw.entry.path], now=MOMENT
    )
    assert again.action == "unchanged"


def test_processed_rejects_an_unrecorded_input() -> None:
    with pytest.raises(CacheError, match="not in the manifest"):
        store_processed(name="govt_returns", frame=frame(), inputs=["raw/fed/nope.parquet"])


def test_regenerating_processed_data_overwrites_in_place(data_root: Path) -> None:
    raw = store_raw(source="fed", name="gsw", url=URL, frame=frame(), now=MOMENT)
    store_processed(name="govt_returns", frame=frame(4), inputs=[raw.entry.path], now=MOMENT)
    changed = store_processed(
        name="govt_returns", frame=frame(4, value=9.0), inputs=[raw.entry.path], now=MOMENT
    )

    assert changed.action == "written"
    assert len(list((data_root / "processed").iterdir())) == 1
    assert read_unrestricted(changed.entry, reason=_STORAGE)["beta0"].iloc[0] == 9.0


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def test_verify_succeeds_on_an_empty_manifest(data_root: Path) -> None:
    # A clean clone has a committed manifest and no cache; `make verify` must pass.
    (data_root / "manifest.json").write_text(Manifest().render())
    report = verify()
    assert report.ok
    assert report.checked == 0
    assert "nothing to verify" in report.render()


def test_verify_succeeds_with_no_manifest_file_at_all() -> None:
    assert verify().ok


def test_verify_detects_a_hand_corrupted_file(data_root: Path) -> None:
    result = store_raw(source="fed", name="gsw", url=URL, frame=frame(), now=MOMENT)
    good = result.path.read_bytes()
    result.path.write_bytes(good[:-1] + bytes([good[-1] ^ 0xFF]))

    with pytest.raises(CacheDriftError) as excinfo:
        verify()

    report = excinfo.value.report
    assert [d.kind for d in report.drift] == ["modified"]
    message = str(excinfo.value)
    assert "raw/fed/gsw_2026-08-25.parquet" in message
    assert result.entry.sha256[:16] in message
    assert "expected sha256" in message and "actual   sha256" in message
    assert "make data" in message


def test_verify_REPORTS_a_deleted_file_and_does_not_call_it_drift(data_root: Path) -> None:
    """W8-P4b operator ruling: `make verify` checks INTEGRITY, not presence.

    This test asserted the opposite until W8-P4b, and the change is a change of
    SPECIFICATION rather than a test bent to fit a result. data/raw is
    gitignored, so a clean clone has no cache at all and the old semantics made
    the README's first command fail for every stranger who ran it. Presence is
    enforced at read time instead -- see the read-side test below.
    """
    result = store_raw(source="fed", name="gsw", url=URL, frame=frame(), now=MOMENT)
    result.path.unlink()

    report = check()
    assert report.ok, "an absent file is not an integrity failure"
    assert report.drift == ()
    assert [d.kind for d in report.absent] == ["missing"]
    assert report.present == 0
    assert "no cache present" in report.render()
    assert "make data" in report.render()


def test_verify_on_a_CLEAN_CLONE_succeeds_and_says_what_to_do(data_root: Path) -> None:
    """The clean-clone case exactly: a full manifest, an empty data directory."""
    for name in ("gsw", "tips", "breakeven"):
        store_raw(source="fed", name=name, url=URL, frame=frame(), now=MOMENT).path.unlink()

    report = check()
    assert report.ok and report.checked == 3 and report.present == 0
    assert len(report.absent) == 3
    rendered = report.render()
    assert "no cache present" in rendered and "3 file(s) recorded" in rendered
    assert _main(["verify"]) == 0, "`make verify` must exit 0 on a clean clone"


def test_a_PRESENT_file_with_the_wrong_bytes_still_fails(data_root: Path) -> None:
    """The other half of the ruling: integrity is exactly what is still checked."""
    kept = store_raw(source="fed", name="gsw", url=URL, frame=frame(), now=MOMENT)
    gone = store_raw(source="fed", name="tips", url=URL, frame=frame(4), now=MOMENT)
    gone.path.unlink()
    kept.path.write_bytes(kept.path.read_bytes() + b"corruption")

    report = check()
    assert not report.ok
    assert [d.kind for d in report.drift] == ["modified"]
    assert [d.kind for d in report.absent] == ["missing"]
    assert _main(["verify"]) == 1, "drift on a file that IS on disk must still fail"


def test_verify_detects_a_derived_file_whose_input_vanished(data_root: Path) -> None:
    raw = store_raw(source="fed", name="gsw", url=URL, frame=frame(), now=MOMENT)
    store_processed(name="govt_returns", frame=frame(4), inputs=[raw.entry.path], now=MOMENT)

    book = Manifest.load()
    book.remove(raw.entry.path)
    book.save()
    (data_root / raw.entry.path).unlink()

    report = check()
    assert [d.kind for d in report.drift] == ["orphaned-input"]
    assert "provenance cannot be reconstructed" in report.render()


def test_verify_reports_untracked_files_without_failing(data_root: Path) -> None:
    store_raw(source="fed", name="gsw", url=URL, frame=frame(), now=MOMENT)
    stray = data_root / "raw" / "fed" / "stray_2020-01-01.parquet"
    stray.write_bytes(b"whatever")

    report = check()
    assert report.ok
    assert report.untracked == ("raw/fed/stray_2020-01-01.parquet",)
    assert "not in the manifest" in report.render()


def test_verify_flags_a_serializer_upgrade_as_such(data_root: Path) -> None:
    result = store_raw(source="fed", name="gsw", url=URL, frame=frame(), now=MOMENT)
    book = Manifest.load()
    # Simulate a manifest written by a different pyarrow/pandas: same data, other bytes.
    book.upsert(
        cache.ManifestEntry.from_json(
            {
                **result.entry.to_json(),
                "writer": "pyarrow=0.0.0;pandas=0.0.0;parquet=2.6",
                "sha256": "0" * 64,
            },
            result.entry.path,
        )
    )
    book.save()

    report = check()
    assert [d.kind for d in report.drift] == ["modified"]
    assert "probably a dependency upgrade rather than corruption" in report.render()


# ---------------------------------------------------------------------------
# Manifest robustness
# ---------------------------------------------------------------------------


def test_manifest_rejects_a_wrong_schema_version(data_root: Path) -> None:
    (data_root / "manifest.json").write_text(json.dumps({"schema_version": 99, "entries": {}}))
    with pytest.raises(CacheError, match="schema_version"):
        Manifest.load()


def test_manifest_rejects_invalid_json(data_root: Path) -> None:
    (data_root / "manifest.json").write_text("{not json")
    with pytest.raises(CacheError, match="not valid JSON"):
        Manifest.load()


def test_manifest_rejects_a_key_that_disagrees_with_its_entry(data_root: Path) -> None:
    result = store_raw(source="fed", name="gsw", url=URL, frame=frame(), now=MOMENT)
    payload = json.loads((data_root / "manifest.json").read_text())
    payload["entries"]["raw/fed/elsewhere.parquet"] = payload["entries"].pop(result.entry.path)
    (data_root / "manifest.json").write_text(json.dumps(payload))
    with pytest.raises(CacheError, match="records a different path"):
        Manifest.load()


# ---------------------------------------------------------------------------
# W7-P1: the committed reference stage (SPEC.md 15.3.1 ruling 3) and the
# per-item status on a manifest entry (ruling 6)
# ---------------------------------------------------------------------------


def test_store_reference_records_a_tracked_text_file(data_root: Path) -> None:
    """A CSV under data/reference is hashed like the cache and names its inputs."""
    raw = cache.store_raw(
        source="wikipedia",
        name="sp500_changes",
        url="u",
        frame=frame(3),
        pull_date=date(2026, 9, 4),
    )
    text = "# header\nticker,start\nAAA,2020-01-01\n"
    result = cache.store_reference(
        filename="sp500_membership_intervals.csv",
        text=text,
        inputs=[raw.entry.path],
        rows=1,
        first_date="2020-01-01",
        last_date="2026-09-04",
    )
    assert result.action == "written"
    entry = result.entry
    assert entry.stage == "reference" and entry.path == "reference/sp500_membership_intervals.csv"
    assert entry.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert entry.inputs == (raw.entry.path,)
    assert entry.input_sha256 == (raw.entry.sha256,)  # the input AS IT WAS
    assert cache.Manifest.load().get(entry.path).input_sha256 == (raw.entry.sha256,)
    assert (data_root / entry.path).read_text(encoding="utf-8") == text
    # Idempotent on identical text, and verified by the same check as the cache.
    again = cache.store_reference(
        filename="sp500_membership_intervals.csv",
        text=text,
        inputs=[raw.entry.path],
        rows=1,
        first_date="2020-01-01",
        last_date="2026-09-04",
    )
    assert again.action == "unchanged"
    assert cache.check().ok
    # Hand-editing the committed file is drift, like any other recorded file.
    (data_root / entry.path).write_text(text + "BBB,2021-01-01\n", encoding="utf-8")
    report = cache.check()
    assert not report.ok and report.drift[0].kind == "modified"


def test_store_reference_refuses_an_unrecorded_input_and_an_empty_file(data_root: Path) -> None:
    with pytest.raises(cache.CacheError, match="not in the manifest"):
        cache.store_reference(
            filename="x.csv",
            text="a\n",
            inputs=["raw/nowhere.parquet"],
            rows=0,
            first_date=None,
            last_date=None,
        )
    with pytest.raises(cache.CacheError, match="empty reference file"):
        cache.store_reference(
            filename="x.csv", text="  \n", inputs=[], rows=0, first_date=None, last_date=None
        )


def test_reference_relpath_requires_an_extension() -> None:
    assert cache.reference_relpath("sp500_membership.csv") == "reference/sp500_membership.csv"
    with pytest.raises(cache.CacheError, match="extension"):
        cache.reference_relpath("sp500_membership")


def test_a_per_item_status_round_trips_through_the_manifest(data_root: Path) -> None:
    """Ruling 6: ok / empty / failed per ticker, on the entry, sorted, and only when given."""
    fetched = cache.Fetched(frame=frame(2), status={"ZZZ": "failed", "AAA": "ok", "MMM": "empty"})
    result = cache.fetch_raw(
        source="yfinance", name="sp500_coverage", url="u", fetcher=lambda: fetched
    )
    assert result.entry.status == (("AAA", "ok"), ("MMM", "empty"), ("ZZZ", "failed"))
    assert result.entry.status_counts() == {"ok": 1, "empty": 1, "failed": 1}
    reloaded = cache.Manifest.load().get(result.entry.path)
    assert reloaded is not None and reloaded.status == result.entry.status
    payload = json.loads((data_root / "manifest.json").read_text())
    assert payload["entries"][result.entry.path]["status"] == {
        "AAA": "ok",
        "MMM": "empty",
        "ZZZ": "failed",
    }
    # An entry without a status carries no key at all, so older entries are byte-identical.
    plain = cache.store_raw(source="s", name="plain", url="u", frame=frame(2))
    assert "status" not in plain.entry.to_json()


def test_a_retired_source_round_trips_through_the_manifest(data_root: Path) -> None:
    """W7-P1b: a retirement is a note in the provenance record, with its reason."""
    book = cache.Manifest.load()
    with pytest.raises(cache.CacheError, match="stated reason"):
        book.retire("stooq", "  ")
    book.retire("stooq", "unreachable since W1-P5 and read by nothing")
    book.save()
    payload = json.loads((data_root / "manifest.json").read_text())
    assert payload["retired_sources"] == {"stooq": "unreachable since W1-P5 and read by nothing"}
    assert cache.Manifest.load().retired_sources == book.retired_sources
    # A manifest without the key loads as no retirements and renders without it.
    empty = cache.Manifest()
    assert empty.retired_sources == {} and "retired_sources" not in empty.render()
