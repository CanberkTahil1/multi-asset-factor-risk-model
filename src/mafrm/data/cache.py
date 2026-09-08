"""Content-addressed cache and provenance manifest for every file this project pulls.

CLAUDE.md invariant 1 says network access lives only in ``mafrm.data.loaders``.
This module is the other half of that rule: it is the door everything else goes
through. It never imports a network library and never fetches. A loader hands it
an already-fetched frame (or a callable that produces one, see :func:`fetch_raw`)
and this module decides whether anything needs writing, records the provenance,
and can prove later that nothing on disk has moved since.

CLAUDE.md invariant 8 and SPEC.md section 11 set the layout:

* ``data/raw/{source}/{name}_{pull_date}.parquet`` -- gitignored, never committed.
* ``data/processed/{name}.parquet`` -- gitignored, deterministic from raw.
* ``data/manifest.json`` -- **committed**. URL, UTC download timestamp, SHA-256,
  row count and date range per file. That asymmetry is the whole caching design:
  the data is not ours to redistribute, but the record of exactly which bytes we
  used is, and it is what makes someone else's rebuild checkable against ours.

Raw files carry the pull date in the name, derived files do not. A raw pull is a
dated observation of an upstream source that may be revised; a derived file is a
pure function of raw plus code, so a second copy under a second date would be
noise rather than provenance.

**What the hash is over.** ``sha256`` is the digest of the parquet bytes actually
on disk, so :func:`verify` can re-hash a file and detect a corrupted or
hand-edited cache. Parquet embeds the writing library's version in its footer, so
the same data serialized by a different pyarrow or pandas build hashes
differently. ``uv.lock`` pins both, and each entry additionally records the
``writer`` fingerprint so that a mismatch after a dependency upgrade reports
itself as an upgrade rather than as corruption. Where a source delivers real
bytes over the wire (a CSV from the Fed, say), the loader may also pass those
bytes through and they are recorded separately as ``source_sha256`` -- that
digest is upstream provenance and is independent of any library version.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd

from mafrm.data import holdout

__all__ = [
    "CacheDriftError",
    "CacheError",
    "CacheResult",
    "Drift",
    "Fetched",
    "Fetcher",
    "Manifest",
    "ManifestEntry",
    "VerifyReport",
    "check",
    "data_dir",
    "fetch_raw",
    "manifest_path",
    "read",
    "read_unrestricted",
    "store_processed",
    "store_raw",
    "store_reference",
    "verify",
]

# Repo root: src/mafrm/data/cache.py -> data -> mafrm -> src -> <root>.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

#: Bumped only when the on-disk shape of manifest.json changes incompatibly.
MANIFEST_SCHEMA_VERSION: Final[int] = 1

# Serialization format constants, NOT model tunables -- they do not belong in
# config/model.yaml under CLAUDE.md invariant 6, which governs half-lives, lags,
# shrinkage constants and cost coefficients. They are pinned here because the
# file hash depends on them: changing one rewrites every digest in the manifest.
_PARQUET_VERSION: Final[str] = "2.6"
#: ``None`` means no compression: codec output is not stable across versions.
_PARQUET_COMPRESSION: Final[Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None] = None

#: ``source`` and ``name`` become path components, so they are constrained.
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

#: ``reference`` (W7-P1) is a COMMITTED, tracked text file under ``data/reference/``:
#: a licensed derived table (SPEC.md 15.3.1 ruling 3), hashed here like everything
#: else so that ``make verify`` covers it, but never a parquet in the ignored cache.
Stage = Literal["raw", "processed", "reference"]
Action = Literal["written", "unchanged", "repaired"]
DriftKind = Literal["missing", "modified", "orphaned-input"]


class CacheError(RuntimeError):
    """The cache was asked for something impossible or inconsistent."""


class CacheDriftError(CacheError):
    """A cached file no longer matches its manifest entry."""

    def __init__(self, report: VerifyReport) -> None:
        super().__init__(report.render())
        self.report = report


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


def data_dir() -> Path:
    """Root of the data tree. Override with ``MAFRM_DATA_DIR`` (tests do)."""
    override = os.environ.get("MAFRM_DATA_DIR")
    return Path(override) if override else _REPO_ROOT / "data"


def manifest_path() -> Path:
    """Path of the committed manifest."""
    return data_dir() / "manifest.json"


def _check_token(value: str, label: str) -> str:
    if not _TOKEN_RE.match(value):
        raise CacheError(
            f"{label} {value!r} is not a valid path component: expected lowercase "
            "letters, digits, dot, dash or underscore, starting alphanumeric"
        )
    return value


def raw_relpath(source: str, name: str, pull_date: date) -> str:
    """Manifest key for a raw pull. Relative to :func:`data_dir`."""
    _check_token(source, "source")
    _check_token(name, "name")
    return f"raw/{source}/{name}_{pull_date.isoformat()}.parquet"


def processed_relpath(name: str) -> str:
    """Manifest key for a derived file. Relative to :func:`data_dir`."""
    _check_token(name, "name")
    return f"processed/{name}.parquet"


def reference_relpath(filename: str) -> str:
    """Manifest key for a committed reference file. Relative to :func:`data_dir`.

    ``filename`` carries its own extension (``.csv``): reference files are text
    a reader can open, not parquet, and the manifest key is the path as
    committed.
    """
    stem, dot, ext = filename.rpartition(".")
    if not dot or not ext:
        raise CacheError(f"reference filename {filename!r} needs an extension")
    _check_token(stem, "reference filename")
    _check_token(ext, "reference extension")
    return f"reference/{filename}"


# ---------------------------------------------------------------------------
# Hashing and serialization
# ---------------------------------------------------------------------------


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=1)
def _writer_fingerprint() -> str:
    """Identifies the serializer, because the parquet bytes depend on it."""
    return f"pyarrow={_dist_version('pyarrow')};pandas={pd.__version__};parquet={_PARQUET_VERSION}"


def _dist_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:  # pragma: no cover - pyarrow is a hard dependency
        return "unknown"


def _serialize(frame: pd.DataFrame) -> bytes:
    """Frame -> parquet bytes, deterministically for a fixed pyarrow/pandas."""
    buffer = io.BytesIO()
    frame.to_parquet(
        buffer,
        engine="pyarrow",
        index=True,
        compression=_PARQUET_COMPRESSION,
        version=_PARQUET_VERSION,
        write_statistics=False,
    )
    return buffer.getvalue()


def _deserialize(payload: bytes) -> pd.DataFrame:
    # engine is left at "auto", which resolves to pyarrow -- a hard dependency and
    # the only engine that writes these files.
    return pd.read_parquet(io.BytesIO(payload))


def _write_atomic(path: Path, payload: bytes) -> None:
    """Write via a sibling temp file so an interrupt cannot truncate the cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------


def _bounds(values: pd.Series) -> tuple[str | None, str | None]:
    stamps = pd.Series(pd.to_datetime(values, errors="coerce")).dropna()
    if stamps.empty:
        return None, None
    return (
        pd.Timestamp(stamps.min()).date().isoformat(),
        pd.Timestamp(stamps.max()).date().isoformat(),
    )


def _date_range(frame: pd.DataFrame, date_column: str | None) -> tuple[str | None, str | None]:
    """First and last date in the frame, or ``(None, None)`` if it has none.

    Order of preference: an explicitly named column, a ``DatetimeIndex``, a
    column literally called ``date``, then any datetime-typed column. The
    manifest records the range for provenance only; nothing downstream reads it,
    so a frame with no date axis is cached with nulls rather than rejected.
    """
    if date_column is not None:
        if date_column not in frame.columns:
            raise CacheError(f"date_column {date_column!r} is not a column of the frame")
        return _bounds(frame[date_column])
    if isinstance(frame.index, pd.DatetimeIndex):
        return _bounds(pd.Series(frame.index))
    for column in frame.columns:
        if str(column).lower() == "date":
            return _bounds(frame[column])
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            return _bounds(frame[column])
    return None, None


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestEntry:
    """Provenance for one cached file. Field order is the JSON key order."""

    #: Path relative to :func:`data_dir`, so the manifest is machine-portable.
    path: str
    stage: Stage
    source: str
    name: str
    #: Source URL, or ``derived:<name>`` for a processed file.
    url: str
    #: UTC, ISO-8601, second resolution, ``Z`` suffix.
    downloaded_at: str
    #: SHA-256 of the parquet bytes on disk.
    sha256: str
    size_bytes: int
    rows: int
    first_date: str | None
    last_date: str | None
    #: SHA-256 of the upstream payload, when the source delivered real bytes.
    source_sha256: str | None
    #: Serializer fingerprint; a mismatch explains a hash change after an upgrade.
    writer: str
    #: For processed files: the manifest paths this was derived from.
    inputs: tuple[str, ...] = ()
    #: Per-item status of a multi-item pull, as ``(item, status)`` pairs sorted by
    #: item -- the S&P 500 bars pull records ``ok`` / ``empty`` / ``failed`` per
    #: ticker here (SPEC.md 15.3.1 ruling 6), so that a name the provider has no
    #: history for and a request that failed are two different records rather
    #: than one absence. Empty for every other entry, and then not serialised.
    status: tuple[tuple[str, str], ...] = ()
    #: For reference files: the SHA-256 of each input AS IT WAS when the file was
    #: built, one per ``inputs`` entry. Raw pulls are keyed by date, so a same-day
    #: re-pull with different content replaces the file under the same path and
    #: its manifest hash with it; this is what lets a reader tell that the cache
    #: has moved past the snapshot even when every input path still resolves
    #: (W7-P1b, the one-snapshot rule).
    input_sha256: tuple[str, ...] = ()

    def status_counts(self) -> dict[str, int]:
        """How many items carry each status, in first-seen order."""
        counts: dict[str, int] = {}
        for _, value in self.status:
            counts[value] = counts.get(value, 0) + 1
        return counts

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "stage": self.stage,
            "source": self.source,
            "name": self.name,
            "url": self.url,
            "downloaded_at": self.downloaded_at,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
            "first_date": self.first_date,
            "last_date": self.last_date,
            "source_sha256": self.source_sha256,
            "writer": self.writer,
            "inputs": list(self.inputs),
        }
        if self.status:
            payload["status"] = dict(self.status)
        if self.input_sha256:
            payload["input_sha256"] = list(self.input_sha256)
        return payload

    @classmethod
    def from_json(cls, raw: Any, key: str) -> ManifestEntry:
        if not isinstance(raw, dict):
            raise CacheError(f"manifest entry {key!r}: expected an object")

        def need(field_name: str) -> Any:
            if field_name not in raw:
                raise CacheError(f"manifest entry {key!r}: missing {field_name!r}")
            return raw[field_name]

        stage = need("stage")
        if stage not in ("raw", "processed", "reference"):
            raise CacheError(f"manifest entry {key!r}: unknown stage {stage!r}")
        status_raw = raw.get("status", {})
        if not isinstance(status_raw, dict):
            raise CacheError(f"manifest entry {key!r}: 'status' must be an object")
        return cls(
            path=str(need("path")),
            stage=stage,
            source=str(need("source")),
            name=str(need("name")),
            url=str(need("url")),
            downloaded_at=str(need("downloaded_at")),
            sha256=str(need("sha256")),
            size_bytes=int(need("size_bytes")),
            rows=int(need("rows")),
            first_date=_opt_str(raw.get("first_date")),
            last_date=_opt_str(raw.get("last_date")),
            source_sha256=_opt_str(raw.get("source_sha256")),
            writer=str(need("writer")),
            inputs=tuple(str(item) for item in raw.get("inputs", ())),
            status=_status_pairs(status_raw),
            input_sha256=tuple(str(item) for item in raw.get("input_sha256", ())),
        )


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _status_pairs(status: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    """A per-item status mapping as the sorted tuple the entry stores."""
    if not status:
        return ()
    return tuple((str(item), str(value)) for item, value in sorted(status.items()))


@dataclass
class Manifest:
    """``data/manifest.json``, parsed. Entries are keyed by relative path."""

    schema_version: int = MANIFEST_SCHEMA_VERSION
    entries: dict[str, ManifestEntry] = field(default_factory=dict)
    #: Sources ``refresh_all`` no longer pulls, with the reason. A retirement is a
    #: note in the provenance record rather than a config edit, so that the fact
    #: that a source was once part of the refresh -- and why it stopped being --
    #: travels with the manifest (W7-P1b: Stooq, unreachable since W1-P5 and
    #: read by nothing).
    retired_sources: dict[str, str] = field(default_factory=dict)

    def retire(self, source: str, reason: str) -> None:
        """Record that ``source`` is no longer refreshed, and why."""
        _check_token(source, "source")
        if not reason.strip():
            raise CacheError("retiring a source requires a stated reason")
        self.retired_sources[source] = reason.strip()

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[ManifestEntry]:
        """Iterate in sorted-path order, which is the on-disk order."""
        return iter(self.entries[key] for key in sorted(self.entries))

    def get(self, path: str) -> ManifestEntry | None:
        return self.entries.get(path)

    def find(self, *, stage: Stage, source: str, name: str) -> tuple[ManifestEntry, ...]:
        """Every entry for one logical series, newest pull last."""
        matched = [
            entry
            for entry in self
            if entry.stage == stage and entry.source == source and entry.name == name
        ]
        return tuple(sorted(matched, key=lambda entry: (entry.downloaded_at, entry.path)))

    def latest(self, *, stage: Stage = "raw", source: str, name: str) -> ManifestEntry:
        """Most recent pull of one series, or raise."""
        matched = self.find(stage=stage, source=source, name=name)
        if not matched:
            raise CacheError(
                f"nothing cached for stage={stage} source={source!r} name={name!r}; run `make data`"
            )
        return matched[-1]

    def upsert(self, entry: ManifestEntry) -> None:
        self.entries[entry.path] = entry

    def remove(self, path: str) -> None:
        self.entries.pop(path, None)

    # -- persistence --------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> Manifest:
        """Read the manifest. A missing file is an empty manifest, not an error.

        A clean clone with no cache has nothing to verify, and `make verify`
        must still succeed there.
        """
        target = path or manifest_path()
        if not target.is_file():
            return cls()
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CacheError(f"{target} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise CacheError(f"{target}: expected an object at the top level")
        schema_version = raw.get("schema_version")
        if schema_version != MANIFEST_SCHEMA_VERSION:
            raise CacheError(
                f"{target}: schema_version {schema_version!r}, expected {MANIFEST_SCHEMA_VERSION}"
            )
        raw_entries = raw.get("entries", {})
        if not isinstance(raw_entries, dict):
            raise CacheError(f"{target}: 'entries' must be an object")
        entries = {
            str(key): ManifestEntry.from_json(value, str(key)) for key, value in raw_entries.items()
        }
        for key, entry in entries.items():
            if entry.path != key:
                raise CacheError(f"manifest entry {key!r} records a different path {entry.path!r}")
        raw_retired = raw.get("retired_sources", {})
        if not isinstance(raw_retired, dict):
            raise CacheError(f"{target}: 'retired_sources' must be an object")
        retired = {str(source): str(reason) for source, reason in raw_retired.items()}
        return cls(schema_version=MANIFEST_SCHEMA_VERSION, entries=entries, retired_sources=retired)

    def render(self) -> str:
        """Canonical JSON text: sorted keys, two-space indent, trailing newline.

        Deterministic so that a re-run with unchanged data produces a byte-identical
        file and therefore an empty git diff.
        """
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "entries": {key: self.entries[key].to_json() for key in sorted(self.entries)},
        }
        if self.retired_sources:
            payload["retired_sources"] = {
                source: self.retired_sources[source] for source in sorted(self.retired_sources)
            }
        return json.dumps(payload, indent=2) + "\n"

    def save(self, path: Path | None = None) -> Path:
        target = path or manifest_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(target, self.render().encode("utf-8"))
        return target


# ---------------------------------------------------------------------------
# Storing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheResult:
    """What :func:`store_raw` / :func:`store_processed` did."""

    entry: ManifestEntry
    #: ``unchanged`` means the fetch matched what was already cached.
    action: Action

    @property
    def path(self) -> Path:
        return data_dir() / self.entry.path


@dataclass(frozen=True)
class Fetched:
    """What a fetcher returns: the parsed frame, optionally the wire bytes."""

    frame: pd.DataFrame
    #: The literal bytes the source served, when there are any. Hashed for
    #: provenance and never written to disk in that form.
    source_bytes: bytes | None = None
    #: Per-item status of a multi-item pull, recorded on the manifest entry.
    status: Mapping[str, str] | None = None


#: A zero-argument callable that performs one download. Supplied by
#: ``mafrm.data.loaders`` in production and by a fake in tests -- which is the
#: seam that keeps CLAUDE.md invariant 1 mechanical rather than aspirational.
Fetcher = Callable[[], Fetched]


def _utc_now(now: datetime | None) -> datetime:
    return (now or datetime.now(UTC)).astimezone(UTC)


def _stamp(moment: datetime) -> str:
    return moment.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def store_raw(
    *,
    source: str,
    name: str,
    url: str,
    frame: pd.DataFrame,
    source_bytes: bytes | None = None,
    pull_date: date | None = None,
    now: datetime | None = None,
    date_column: str | None = None,
    allow_empty: bool = False,
    manifest: Manifest | None = None,
    status: Mapping[str, str] | None = None,
) -> CacheResult:
    """Cache one raw pull and record it, skipping the write if nothing changed.

    Idempotency is by content across pull dates, not by filename: if the freshly
    fetched frame serializes to bytes already cached under *any* pull date for
    this ``(source, name)``, the existing file and its original timestamp are
    kept and nothing is written. Otherwise a file dated ``pull_date`` is written.
    So ``data/raw`` holds one file per date on which the upstream data actually
    changed, rather than one identical copy per day the pipeline was run.

    If the recorded file has gone missing or its bytes no longer hash to the
    recorded digest, it is rewritten from this fetch and the action is
    ``repaired`` -- ``make data`` heals a corrupted cache, ``make verify``
    reports one.
    """
    if frame.empty and not allow_empty:
        raise CacheError(
            f"refusing to cache an empty frame for {source}/{name}: a fetch that returns "
            "no rows is a silently broken loader. Pass allow_empty=True if it is genuine."
        )

    moment = _utc_now(now)
    payload = _serialize(frame)
    digest = _sha256(payload)
    book = manifest if manifest is not None else Manifest.load()

    for candidate in book.find(stage="raw", source=source, name=name):
        if candidate.sha256 != digest:
            continue
        target = data_dir() / candidate.path
        if target.is_file() and _sha256(target.read_bytes()) == digest:
            return CacheResult(candidate, "unchanged")
        _write_atomic(target, payload)
        return CacheResult(candidate, "repaired")

    relpath = raw_relpath(source, name, pull_date or moment.date())
    first_date, last_date = _date_range(frame, date_column)
    entry = ManifestEntry(
        path=relpath,
        stage="raw",
        source=source,
        name=name,
        url=url,
        downloaded_at=_stamp(moment),
        sha256=digest,
        size_bytes=len(payload),
        rows=len(frame),
        first_date=first_date,
        last_date=last_date,
        source_sha256=None if source_bytes is None else _sha256(source_bytes),
        writer=_writer_fingerprint(),
        status=_status_pairs(status),
    )
    _write_atomic(data_dir() / relpath, payload)
    book.upsert(entry)
    book.save()
    return CacheResult(entry, "written")


def fetch_raw(
    *,
    source: str,
    name: str,
    url: str,
    fetcher: Fetcher,
    pull_date: date | None = None,
    now: datetime | None = None,
    date_column: str | None = None,
    allow_empty: bool = False,
) -> CacheResult:
    """Run ``fetcher`` and cache what it returns.

    The fetch always happens -- you cannot know a payload's hash without
    downloading it -- but the disk write does not. This module never fetches
    anything itself; ``fetcher`` comes from ``mafrm.data.loaders`` (CLAUDE.md
    invariant 1) and from a fake in tests.
    """
    fetched = fetcher()
    return store_raw(
        source=source,
        name=name,
        url=url,
        frame=fetched.frame,
        source_bytes=fetched.source_bytes,
        pull_date=pull_date,
        now=now,
        date_column=date_column,
        allow_empty=allow_empty,
        status=fetched.status,
    )


def store_processed(
    *,
    name: str,
    frame: pd.DataFrame,
    inputs: Sequence[str],
    source: str = "derived",
    now: datetime | None = None,
    date_column: str | None = None,
    allow_empty: bool = False,
    manifest: Manifest | None = None,
) -> CacheResult:
    """Cache one derived table, recording the raw entries it was built from.

    Derived files are a pure function of their inputs plus the code, so they are
    not dated: regenerating overwrites in place and a changed hash is the signal.
    ``inputs`` are manifest-relative raw paths and must already be recorded --
    a derived file whose provenance cannot be named is not reproducible, which
    is the point of the manifest.
    """
    if frame.empty and not allow_empty:
        raise CacheError(f"refusing to cache an empty derived frame for {name!r}")

    book = manifest if manifest is not None else Manifest.load()
    for input_path in inputs:
        if book.get(input_path) is None:
            raise CacheError(
                f"derived file {name!r} names input {input_path!r}, which is not in the "
                "manifest. Cache the raw input first."
            )

    moment = _utc_now(now)
    payload = _serialize(frame)
    digest = _sha256(payload)
    relpath = processed_relpath(name)
    target = data_dir() / relpath

    existing = book.get(relpath)
    if (
        existing is not None
        and existing.sha256 == digest
        and tuple(inputs) == existing.inputs
        and target.is_file()
        and _sha256(target.read_bytes()) == digest
    ):
        return CacheResult(existing, "unchanged")

    first_date, last_date = _date_range(frame, date_column)
    entry = ManifestEntry(
        path=relpath,
        stage="processed",
        source=source,
        name=name,
        url=f"derived:{name}",
        downloaded_at=_stamp(moment),
        sha256=digest,
        size_bytes=len(payload),
        rows=len(frame),
        first_date=first_date,
        last_date=last_date,
        source_sha256=None,
        writer=_writer_fingerprint(),
        inputs=tuple(inputs),
    )
    _write_atomic(target, payload)
    book.upsert(entry)
    book.save()
    return CacheResult(entry, "written")


def store_reference(
    *,
    filename: str,
    text: str,
    inputs: Sequence[str],
    rows: int,
    first_date: str | None,
    last_date: str | None,
    source: str = "derived",
    now: datetime | None = None,
    manifest: Manifest | None = None,
) -> CacheResult:
    """Write one committed reference file under ``data/reference/`` and record it.

    The file is TRACKED (SPEC.md 15.3.1 ruling 3): a licensed derived table the
    repository redistributes under its own licence header, not raw third-party
    data. It is recorded here so that ``make verify`` re-hashes it like the
    cache, and so that its provenance names the raw entries it was built from.
    Text is written as UTF-8 exactly as given; the hash is of those bytes.
    """
    if not text.strip():
        raise CacheError(f"refusing to write an empty reference file {filename!r}")
    book = manifest if manifest is not None else Manifest.load()
    input_hashes: list[str] = []
    for input_path in inputs:
        recorded = book.get(input_path)
        if recorded is None:
            raise CacheError(
                f"reference file {filename!r} names input {input_path!r}, which is not in the "
                "manifest. Cache the raw input first."
            )
        input_hashes.append(recorded.sha256)
    moment = _utc_now(now)
    payload = text.encode("utf-8")
    digest = _sha256(payload)
    relpath = reference_relpath(filename)
    target = data_dir() / relpath
    existing = book.get(relpath)
    if (
        existing is not None
        and existing.sha256 == digest
        and tuple(inputs) == existing.inputs
        and tuple(input_hashes) == existing.input_sha256
        and target.is_file()
        and _sha256(target.read_bytes()) == digest
    ):
        return CacheResult(existing, "unchanged")
    entry = ManifestEntry(
        path=relpath,
        stage="reference",
        source=source,
        name=filename,
        url=f"derived:{filename}",
        downloaded_at=_stamp(moment),
        sha256=digest,
        size_bytes=len(payload),
        rows=rows,
        first_date=first_date,
        last_date=last_date,
        source_sha256=None,
        writer="text/csv;utf-8",
        inputs=tuple(inputs),
        input_sha256=tuple(input_hashes),
    )
    _write_atomic(target, payload)
    book.upsert(entry)
    book.save()
    return CacheResult(entry, "written")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read(entry: ManifestEntry | str, *, manifest: Manifest | None = None) -> pd.DataFrame:
    """Load a cached file, **truncated at the holdout boundary**.

    This is the analysis-side read and it is deliberately the short, obvious name:
    CLAUDE.md invariant 5 is enforced here, at the point data enters the process,
    rather than by every caller remembering to slice. A normal call cannot return
    an observation on or after ``model.sample.holdout_start``.

    Work that legitimately needs the full cached history -- manifest coverage,
    release drift, and the data-layer cross-checks, all of which are questions
    about what a *source* published rather than about model behaviour -- calls
    :func:`read_unrestricted` and states why. See :mod:`mafrm.data.holdout` for
    what prompted the split.

    Index *values* round-trip; a pandas ``freq`` attribute on the index does not,
    because parquet has nowhere to put it. That is deliberate -- a frequency on a
    market calendar is a claim the data cannot support once holidays exist.
    """
    resolved = _resolve(entry, manifest)
    return holdout.truncate(_read_bytes(resolved), label=f"{resolved.source}:{resolved.name}")


def read_unrestricted(
    entry: ManifestEntry | str, *, manifest: Manifest | None = None, reason: str
) -> pd.DataFrame:
    """Load a cached file over its **full** history, past the holdout boundary.

    ``reason`` is required and is not decorative: it is the one-line record of why
    this call is entitled to see data the model may not, and it makes every
    crossing greppable. Legitimate reasons are questions about the *source* --
    what it published and when -- not about the model:

    * manifest coverage and date ranges (``data/manifest.json``);
    * the AQR release-drift detector, which needs observed release history;
    * the data-layer cross-checks (SPY vs CRSP, TLT vs the GSW ladder), which
      validate a vendor series rather than a model output.

    If the reason is "I wanted to look at the recent numbers", it is not one.
    """
    if not reason.strip():
        raise CacheError(
            "read_unrestricted requires a stated reason. Reaching past the holdout boundary "
            "is a visible choice, not a default (CLAUDE.md invariant 5)."
        )
    return _read_bytes(_resolve(entry, manifest))


def _resolve(entry: ManifestEntry | str, manifest: Manifest | None) -> ManifestEntry:
    """A manifest key or an entry, as an entry."""
    if isinstance(entry, str):
        book = manifest if manifest is not None else Manifest.load()
        found = book.get(entry)
        if found is None:
            raise CacheError(f"{entry!r} is not in {manifest_path()}")
        entry = found
    return entry


def _read_bytes(entry: ManifestEntry) -> pd.DataFrame:
    """The untruncated read. Both public accessors go through this."""
    target = data_dir() / entry.path
    if not target.is_file():
        raise CacheError(f"{target} is recorded in the manifest but missing; run `make data`")
    payload = target.read_bytes()
    digest = _sha256(payload)
    if digest != entry.sha256:
        raise CacheError(
            f"{target} does not match the manifest (expected sha256 {entry.sha256[:16]}..., "
            f"got {digest[:16]}...). Run `make verify` for the full report."
        )
    return _deserialize(payload)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Drift:
    """One way in which the cache disagrees with the manifest."""

    path: str
    kind: DriftKind
    detail: str


@dataclass(frozen=True)
class VerifyReport:
    """Result of re-hashing everything the manifest claims.

    **This target checks INTEGRITY, not presence** (operator ruling, W8-P4b).
    A recorded file that is absent is reported and is NOT drift: ``data/raw``
    and ``data/processed`` are gitignored, so a clean clone legitimately has
    none of them and `make verify` must still succeed there -- it is the first
    command the README gives a stranger. A file that is *present* with the
    wrong bytes is drift and fails, which is the whole job of the target.

    Presence is enforced where it actually matters, at read time: :func:`read`
    raises :class:`CacheError` naming the file and telling the caller to run
    `make data`. Nothing can silently consume a cache that is not there.
    """

    checked: int
    drift: tuple[Drift, ...]
    untracked: tuple[str, ...]
    manifest_file: str
    #: Recorded files that are not on disk. Reported, never drift.
    absent: tuple[Drift, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.drift

    @property
    def present(self) -> int:
        """Recorded files actually on disk, i.e. those whose bytes were hashed."""
        return self.checked - len(self.absent)

    def render(self) -> str:
        lines: list[str] = []
        if self.ok:
            if self.checked == 0:
                lines.append(
                    f"data cache: {self.manifest_file} records no files; nothing to verify."
                )
            elif self.present == 0:
                lines.append(
                    f"data cache: no cache present -- {self.checked} file(s) recorded in "
                    f"{self.manifest_file}, none on disk. Nothing is wrong: data/raw and "
                    "data/processed are gitignored and rebuildable. Run `make data` to "
                    "populate the cache."
                )
            else:
                lines.append(
                    f"data cache: {self.present} of {self.checked} recorded file(s) verified "
                    f"against {self.manifest_file}, no drift."
                )
                if self.absent:
                    lines.append("")
                    lines.append(
                        f"  {len(self.absent)} recorded file(s) are not on disk. That is not "
                        "drift -- run `make data` to refetch them:"
                    )
                    lines.extend(f"    {item.path}" for item in self.absent)
        else:
            lines.append(
                f"data cache drift: {len(self.drift)} of {self.present} file(s) on disk "
                f"do not match {self.manifest_file}."
            )
            lines.append("")
            for item in self.drift:
                lines.append(f"  {item.kind:<15} {item.path}")
                lines.extend(f"      {line}" for line in item.detail.splitlines())
                lines.append("")
            lines.append(
                "  data/raw and data/processed are gitignored and rebuildable: "
                "run `make data` to refetch."
            )
            if self.absent:
                lines.append("")
                lines.append(
                    f"  separately, {len(self.absent)} recorded file(s) are not on disk; "
                    "that is not drift."
                )
        if self.untracked:
            lines.append("")
            lines.append(
                f"  note: {len(self.untracked)} cached file(s) are not in the manifest and "
                "are ignored by every loader:"
            )
            lines.extend(f"    {path}" for path in self.untracked)
        return "\n".join(lines)


def _scan_untracked(known: set[str]) -> tuple[str, ...]:
    root = data_dir()
    found: list[str] = []
    for stage, pattern in (
        ("raw", "*.parquet"),
        ("processed", "*.parquet"),
        ("reference", "*.csv"),
    ):
        stage_dir = root / stage
        if not stage_dir.is_dir():
            continue
        for path in sorted(stage_dir.rglob(pattern)):
            relpath = path.relative_to(root).as_posix()
            if relpath not in known:
                found.append(relpath)
    return tuple(found)


def check(manifest: Manifest | None = None) -> VerifyReport:
    """Re-hash every recorded file and report, without raising.

    Two kinds of drift, both about INTEGRITY: a recorded file whose bytes
    changed, and a derived file naming an input that is no longer recorded. A
    recorded file that is simply absent is reported in
    :attr:`VerifyReport.absent` and is **not** drift -- see :class:`VerifyReport`
    for why, and :func:`read` for where presence is enforced instead.
    """
    book = manifest if manifest is not None else Manifest.load()
    root = data_dir()
    drift: list[Drift] = []
    absent: list[Drift] = []

    for entry in book:
        target = root / entry.path
        if not target.is_file():
            absent.append(
                Drift(
                    path=entry.path,
                    kind="missing",
                    detail=(
                        f"recorded {entry.downloaded_at}, {entry.rows:,} rows, "
                        f"{entry.first_date or '?'}..{entry.last_date or '?'}\n"
                        f"from {entry.url}\n"
                        "the file is not on disk"
                    ),
                )
            )
            continue

        payload = target.read_bytes()
        digest = _sha256(payload)
        if digest != entry.sha256:
            running = _writer_fingerprint()
            detail = (
                f"expected sha256 {entry.sha256[:16]}...  ({entry.size_bytes:,} bytes)\n"
                f"actual   sha256 {digest[:16]}...  ({len(payload):,} bytes)"
            )
            if running != entry.writer:
                detail += (
                    f"\nwriter recorded {entry.writer}"
                    f"\nwriter running  {running}"
                    "\nthe serializer differs, so identical data would hash differently;"
                    "\nthis is probably a dependency upgrade rather than corruption."
                )
            drift.append(Drift(path=entry.path, kind="modified", detail=detail))

        for input_path in entry.inputs:
            if book.get(input_path) is None:
                drift.append(
                    Drift(
                        path=entry.path,
                        kind="orphaned-input",
                        detail=(
                            f"derived from {input_path}, which is no longer in the manifest;\n"
                            "its provenance cannot be reconstructed"
                        ),
                    )
                )

    return VerifyReport(
        checked=len(book),
        drift=tuple(drift),
        untracked=_scan_untracked(set(book.entries)),
        manifest_file=manifest_path().as_posix(),
        absent=tuple(absent),
    )


def verify(manifest: Manifest | None = None) -> VerifyReport:
    """Re-hash every recorded file; raise :class:`CacheDriftError` on any drift.

    An absent file is not drift and does not raise (W8-P4b ruling); see
    :class:`VerifyReport`.
    """
    report = check(manifest)
    if not report.ok:
        raise CacheDriftError(report)
    return report


# ---------------------------------------------------------------------------
# `make verify`
# ---------------------------------------------------------------------------


def _main(argv: Sequence[str]) -> int:
    if list(argv) != ["verify"]:
        print("usage: python -m mafrm.data.cache verify", file=sys.stderr)
        return 2
    try:
        report = check()
    except CacheError as exc:
        print(f"data cache: {exc}", file=sys.stderr)
        return 1
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
