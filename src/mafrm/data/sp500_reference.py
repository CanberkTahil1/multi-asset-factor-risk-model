"""Build and commit ``data/reference/`` -- the S&P 500 membership tables. SPEC.md 15.3.1.

``python -m mafrm.data.sp500_reference`` reads the cached Wikipedia tables and
the cached Yahoo bars, runs :func:`mafrm.data.sp500.reconstruct`, and writes
two tracked CSV files with a licence header, recording each in
``data/manifest.json`` under stage ``reference`` with the raw entries it was
built from. Wired to ``make data`` after the refresh.

**One snapshot (W7-P1b).** The committed files are written once. When both
exist and are recorded in the manifest, ``make data`` keeps them and says so;
rewriting them from a newer pull is a deliberate ``--resnapshot``, committed
with its reason. The report reads the snapshot's own recorded inputs where the
cache still holds them, so a refreshed cache does not silently move the
comparison off the snapshot.

Reads are UNRESTRICTED by design and the reason is stated at each call: the
matrix is a property of the *source snapshot* -- a current constituent list is
read after the holdout boundary by construction, and the data state for
2025-2026 members is as real as for 2007 members. Model-facing reads of the
same bars (the estimation screen, the descriptors) go through
:func:`mafrm.data.cache.read`, which stops at the boundary.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from mafrm import config as config_mod
from mafrm.data import cache, calendar, loaders, sp500

__all__ = [
    "ReferenceBuild",
    "ReferenceInputs",
    "bars_present_by_session",
    "build",
    "load_inputs",
    "main",
    "reference_dir",
    "valid_bars",
]

_REASON_TABLES = (
    "the membership tables are a SOURCE snapshot dated by column, not by index; the "
    "current list is by construction read after the holdout boundary (SPEC.md 15.3.1)"
)
_REASON_BARS = (
    "the matrix's data state is a property of the source snapshot and spans it "
    "(SPEC.md 15.3.1 ruling 5); model-facing reads of these bars go through cache.read"
)


def reference_dir(cfg: config_mod.Config | None = None) -> Path:
    """``data/reference/``, resolved against the repository root."""
    cfg = cfg or config_mod.load()
    return Path(__file__).resolve().parents[3] / cfg.model.equity_universe.reference_dir


@dataclass(frozen=True)
class ReferenceInputs:
    constituents: pd.DataFrame
    changes: pd.DataFrame
    #: Long-format bars, date index, ``ticker`` column. Full history.
    prices: pd.DataFrame
    #: One row per requested ticker: ``status``, ``attempts``, ``rows``, dates.
    coverage: pd.DataFrame
    constituents_entry: cache.ManifestEntry
    changes_entry: cache.ManifestEntry
    prices_entry: cache.ManifestEntry
    coverage_entry: cache.ManifestEntry


@dataclass(frozen=True)
class ReferenceBuild:
    reconstruction: sp500.Reconstruction
    gaps: sp500.GapCounts
    matrix: pd.DataFrame
    provenance: sp500.Provenance
    inputs: ReferenceInputs
    intervals_path: Path
    membership_path: Path


def snapshot_input_entries(
    cfg: config_mod.Config, manifest: cache.Manifest
) -> tuple[cache.ManifestEntry, ...] | None:
    """The raw entries the COMMITTED matrix records as its inputs, if all are cached.

    ``None`` when the reference file is not recorded, when any of its inputs is
    no longer in the manifest or on disk -- a fresh clone after ``make data``,
    say, whose raw pulls carry a later date than the snapshot's -- or when an
    input path still resolves but its hash is not the one recorded at build
    time: raw pulls are keyed by date, so a same-day re-pull with different
    content replaces the snapshot's file under the same name (W7-P1b found
    exactly this, twenty minutes after the snapshot).
    """
    relpath = cache.reference_relpath(cfg.model.equity_universe.membership_file)
    entry = manifest.get(relpath)
    if entry is None:
        return None
    if len(entry.input_sha256) != len(entry.inputs):
        return None
    resolved: list[cache.ManifestEntry] = []
    for path, digest in zip(entry.inputs, entry.input_sha256, strict=True):
        found = manifest.get(path)
        if found is None or not (cache.data_dir() / path).is_file():
            return None
        if found.sha256 != digest:
            return None
        resolved.append(found)
    return tuple(resolved)


def load_inputs(
    manifest: cache.Manifest | None = None,
    *,
    entries: tuple[cache.ManifestEntry, ...] | None = None,
) -> ReferenceInputs:
    """The four cached artefacts the build reads, full history, reasons stated.

    ``entries`` pins the exact manifest entries -- the snapshot's recorded
    inputs, from :func:`snapshot_input_entries` -- instead of the latest pull
    of each series.
    """
    book = manifest if manifest is not None else cache.Manifest.load()
    if entries is not None:
        by_key = {(e.source, e.name): e for e in entries}
        try:
            constituents_entry = by_key[(sp500.SOURCE, sp500.CONSTITUENTS_NAME)]
            changes_entry = by_key[(sp500.SOURCE, sp500.CHANGES_NAME)]
            prices_entry = by_key[("yfinance", loaders.SP500_PRICES_NAME)]
            coverage_entry = by_key[("yfinance", loaders.SP500_COVERAGE_NAME)]
        except KeyError as exc:
            raise sp500.SP500Error(f"snapshot inputs lack {exc.args[0]}") from exc
    else:
        constituents_entry = book.latest(source=sp500.SOURCE, name=sp500.CONSTITUENTS_NAME)
        changes_entry = book.latest(source=sp500.SOURCE, name=sp500.CHANGES_NAME)
        prices_entry = book.latest(source="yfinance", name=loaders.SP500_PRICES_NAME)
        coverage_entry = book.latest(source="yfinance", name=loaders.SP500_COVERAGE_NAME)
    return ReferenceInputs(
        constituents=cache.read_unrestricted(
            constituents_entry, manifest=book, reason=_REASON_TABLES
        ),
        changes=cache.read_unrestricted(changes_entry, manifest=book, reason=_REASON_TABLES),
        prices=cache.read_unrestricted(prices_entry, manifest=book, reason=_REASON_BARS),
        coverage=cache.read_unrestricted(coverage_entry, manifest=book, reason=_REASON_BARS),
        constituents_entry=constituents_entry,
        changes_entry=changes_entry,
        prices_entry=prices_entry,
        coverage_entry=coverage_entry,
    )


def valid_bars(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """A bar is valid when close and volume are both present and positive.

    One definition, shared by the matrix's data state and the estimation
    screen (:mod:`mafrm.factors.equity_universe`), so that the committed
    third state and the screen's ``member_no_data`` agree by construction.
    """
    aligned_volume = volume.reindex(index=close.index, columns=close.columns)
    return close.notna() & (close > 0) & aligned_volume.notna() & (aligned_volume > 0)


def bars_present_by_session(prices: pd.DataFrame) -> pd.DataFrame:
    """Session x ticker: does the cache hold a valid bar on that session?

    The index is the union of dates any ticker has a bar on -- the trading
    calendar the pipeline aligns on (:mod:`mafrm.data.calendar`).
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise sp500.SP500Error("bars_present_by_session: prices must be indexed by date")
    close = prices.pivot(columns="ticker", values="Close").sort_index()
    volume = prices.pivot(columns="ticker", values="Volume").sort_index()
    present = valid_bars(close, volume)
    present.index = pd.DatetimeIndex(calendar.trading_dates(close, label="sp500_prices"))
    present.index.name = "date"
    present.columns = pd.Index([str(c) for c in present.columns], name="ticker")
    return present


def _snapshot_date(entry: cache.ManifestEntry) -> pd.Timestamp:
    return pd.Timestamp(entry.downloaded_at.rstrip("Z")[:10])


def _status_summary(
    entry: cache.ManifestEntry,
) -> tuple[tuple[tuple[str, int], ...], tuple[str, ...]]:
    counts = tuple(sorted(entry.status_counts().items()))
    failed = tuple(sorted(t for t, s in entry.status if s == loaders.FAILED))
    return counts, failed


def committed_files_exist(cfg: config_mod.Config, manifest: cache.Manifest) -> bool:
    """Both reference files on disk and recorded -- the one-snapshot condition."""
    universe = cfg.model.equity_universe
    directory = reference_dir(cfg)
    for name in (universe.intervals_file, universe.membership_file):
        if manifest.get(cache.reference_relpath(name)) is None:
            return False
        if not (directory / name).is_file():
            return False
    return True


def build(
    cfg: config_mod.Config | None = None,
    *,
    inputs: ReferenceInputs | None = None,
    now: datetime | None = None,
    write: bool = True,
    resnapshot: bool = False,
) -> ReferenceBuild:
    """Reconstruct, render, write (unless ``write=False``) and record.

    With ``write=True`` and the committed files already present and recorded,
    nothing is written unless ``resnapshot=True`` (the one-snapshot rule,
    W7-P1b); the reconstruction is still returned for the report.
    """
    cfg = cfg or config_mod.load()
    universe = cfg.model.equity_universe
    data = inputs if inputs is not None else load_inputs()
    moment = now if now is not None else datetime.now(UTC)
    if write and not resnapshot and committed_files_exist(cfg, cache.Manifest.load()):
        write = False

    snapshot = _snapshot_date(data.constituents_entry)
    rec = sp500.reconstruct(data.constituents, data.changes, snapshot=snapshot)
    gaps = sp500.gap_table(rec, data.changes)
    present = bars_present_by_session(data.prices)
    matrix = sp500.membership_matrix(
        rec, sessions=pd.DatetimeIndex(present.index), bars_present=present
    )
    status_counts, failed = _status_summary(data.coverage_entry)

    intervals_prov = sp500.Provenance(
        constituents_url=universe.constituents_url,
        changes_url=universe.changes_url,
        licence=universe.licence,
        licence_url=universe.licence_url,
        snapshot=snapshot,
        pulled_at=data.constituents_entry.downloaded_at,
        bars_pull=None,
        generated_at=moment,
    )
    matrix_prov = sp500.Provenance(
        constituents_url=universe.constituents_url,
        changes_url=universe.changes_url,
        licence=universe.licence,
        licence_url=universe.licence_url,
        snapshot=snapshot,
        pulled_at=data.constituents_entry.downloaded_at,
        bars_pull=data.prices_entry.downloaded_at,
        generated_at=moment,
        status_counts=status_counts,
        failed_tickers=failed,
    )

    directory = reference_dir(cfg)
    intervals_path = directory / universe.intervals_file
    membership_path = directory / universe.membership_file
    if write:
        directory.mkdir(parents=True, exist_ok=True)
        table_inputs = [data.constituents_entry.path, data.changes_entry.path]
        cache.store_reference(
            filename=universe.intervals_file,
            text=sp500.render_intervals_csv(rec, intervals_prov),
            inputs=table_inputs,
            rows=len(rec.intervals),
            first_date=rec.reach.date().isoformat(),
            last_date=snapshot.date().isoformat(),
            source=sp500.SOURCE,
            now=moment,
        )
        cache.store_reference(
            filename=universe.membership_file,
            text=sp500.render_membership_csv(matrix, matrix_prov),
            inputs=[*table_inputs, data.prices_entry.path, data.coverage_entry.path],
            rows=len(matrix),
            first_date=pd.Timestamp(matrix.index.min()).date().isoformat(),
            last_date=pd.Timestamp(matrix.index.max()).date().isoformat(),
            source=sp500.SOURCE,
            now=moment,
        )
    return ReferenceBuild(
        reconstruction=rec,
        gaps=gaps,
        matrix=matrix,
        provenance=matrix_prov,
        inputs=data,
        intervals_path=intervals_path,
        membership_path=membership_path,
    )


def _main(argv: Sequence[str]) -> int:
    args = list(argv)
    if args not in ([], ["--resnapshot"]):
        print("usage: python -m mafrm.data.sp500_reference [--resnapshot]", file=sys.stderr)
        return 2
    resnapshot = args == ["--resnapshot"]
    cfg = config_mod.load()
    kept = committed_files_exist(cfg, cache.Manifest.load()) and not resnapshot
    result = build(cfg, resnapshot=resnapshot)
    if kept:
        print(
            f"kept {result.intervals_path.name} and {result.membership_path.name}: one snapshot "
            "(SPEC.md 15.3.1 ruling 3); rewrite deliberately with --resnapshot"
        )
        return 0
    rec = result.reconstruction
    counts = (result.matrix != sp500.NOT_MEMBER).sum(axis=1)
    print(f"wrote {result.intervals_path} ({len(rec.intervals)} spells, reach {rec.reach.date()})")
    print(
        f"wrote {result.membership_path} ({result.matrix.shape[0]} sessions x "
        f"{result.matrix.shape[1]} tickers; members {int(counts.min())}..{int(counts.max())}; "
        f"{result.membership_path.stat().st_size:,} bytes)"
    )
    print(
        f"inconsistencies {len(rec.inconsistencies)}, rename links {len(rec.rename_links)}, "
        f"additions without row {len(rec.additions_without_row)}, "
        f"undated members {len(rec.undated_members)}; loader status "
        f"{dict(result.provenance.status_counts)}, failed after retry "
        f"{list(result.provenance.failed_tickers) or 'none'}"
    )
    return 0


def main() -> int:
    return _main(sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
