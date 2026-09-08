"""The shared forecast-history cache. One builder, four callers.

SPEC.md 5.4's volatility regime adjustment consumes a history of out-of-sample
forecasts, and building one costs a full pre-VRA pipeline run **per date**. Every
model that reaches that stage therefore needs the same machinery: a list of
variants, a digest-guarded partial per variant, a fan-out across processes, and
an assembled cache committed under ``reports/`` so that ``make report`` renders
in seconds rather than in hours.

WHY THIS MODULE EXISTS
----------------------

**It was written three times before it was written once.** ``vra_report`` grew
the shape in W3-P4, ``statistical_report`` grew it again in W3-P5, and
``hybrid_report`` grew it a third time in W3-P6. Parallelism arrived in each at a
different session, so at the moment this module was extracted the three copies
were at three different levels of capability: ``vra_report`` had fan-out **and**
mid-column checkpointing, ``statistical_report`` had fan-out, and the newest copy
had neither -- which is why the W3-P6 history build was first started as a
100-minute sequential run when the same work fans out to about twelve.

**And they had already diverged in a way that matters.** The two older copies'
``risk_config_digest`` and ``panel_digest`` were byte-identical; the third's were
not. It used ``vars()`` where they used ``asdict()``, skipped their
tuple-to-list normalisation, and hashed the panel's columns, endpoints and values
in a different order entirely. Nothing had been built with it yet, so the
divergence cost nothing this time -- but a digest is a **staleness guard**, and a
third implementation that computes a different number from the same inputs is a
guard that silently protects something other than what it claims to. That is the
defect, not the duplication.

This is the same class as W3-P1's ``ewma_weights`` consolidation, and it is
recorded here for the same reason: **three copies of a thing that must behave
identically is a defect even while all three happen to work.** Here they
demonstrably did not.

WHAT IS SHARED AND WHAT IS NOT
------------------------------

Shared, because it must be identical everywhere: the two digests, the partial
format, the cache format, the staleness refusal, and the fan-out.

Not shared, because it is genuinely per-model: what a variant *is*
(:class:`Variant` is a protocol with one required member, ``name``), what
computing one produces, and which factor panel a variant belongs to. Those
arrive as callables. A caller that wants mid-column checkpointing supplies its
own ``compute`` that does it -- :func:`load_or_build` only owns the
finished-partial case, which is the part all four share.

W4-P2 is the fourth caller and the reason this was extracted rather than ported a
third time: its stage attribution needs a dozen or more histories, so it is the
one where rediscovering the fan-out under time pressure would actually hurt.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from mafrm import config
from mafrm.risk.config import HORIZONS, RiskConfig

__all__ = [
    "Cache",
    "Variant",
    "assemble",
    "cache_provenance",
    "fan_out",
    "git_commit",
    "load_or_build",
    "panel_digest",
    "partial_path",
    "read_cache",
    "read_partial",
    "risk_config_digest",
    "write_cache",
    "write_partial",
]

_ROOT = Path(__file__).resolve().parents[2]


class Variant(Protocol):
    """What every caller's column/variant object must provide.

    One member. A variant is whatever the calling module says it is -- a horizon
    and an eigenfactor ``a``, a Model B variant, a residual-PC count -- and this
    module only ever needs to name it, so that is all it asks for.
    """

    @property
    def name(self) -> str: ...


# ---------------------------------------------------------------------------
# The digests. THE reason this module exists -- see the module docstring.
# ---------------------------------------------------------------------------


def risk_config_digest(settings: config.Config | None = None) -> str:
    """SHA-256 of every ``RiskConfig`` field, both horizons. The staleness key.

    These are the values that DETERMINE a cached bias series: change any of them
    and every number in the file describes a model that no longer exists.
    Nothing else in ``config/model.yaml`` can reach it.

    **This function must not change without invalidating every committed cache
    in the repository.** ``reports/vra_forecast_history.csv``,
    ``reports/statistical_forecast_history.csv`` and
    ``reports/hybrid_forecast_history.csv`` all carry its output, and each costs
    between twelve minutes and an hour and a half to rebuild. A test asserts the
    committed caches still validate against it.
    """
    loaded = settings or config.load()
    payload = {}
    for horizon in HORIZONS:
        fields = asdict(RiskConfig.load(horizon=horizon, config=loaded))
        payload[horizon] = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in sorted(fields.items())
        }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def panel_digest(frame: pd.DataFrame) -> str:
    """SHA-256 of a factor panel's values, index and column names.

    The guard that actually catches things. A changed half-life shows up in
    :func:`risk_config_digest` and in ``config/model.yaml``; a **moved factor
    series** shows up in neither. W3-P5 hit exactly that: an extraction fix
    inside ``mafrm.factors.statistical`` moved Model B's factors while every
    value in ``config/model.yaml`` stayed byte-identical, so a cache built
    minutes earlier described a model that no longer existed and no digest in the
    file would have said so.

    The same caveat as :func:`risk_config_digest` applies -- changing the hashed
    quantities or their order invalidates every committed cache.
    """
    values = hashlib.sha256(np.ascontiguousarray(frame.to_numpy(dtype=float)).tobytes())
    index = pd.DatetimeIndex(frame.index).astype("int64").to_numpy()
    values.update(np.ascontiguousarray(index).tobytes())
    values.update("|".join(str(column) for column in frame.columns).encode("utf-8"))
    return values.hexdigest()


def git_commit() -> str:
    """The commit a cache was built at, or ``"unknown"`` outside a checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_ROOT,
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - not a checkout
        return "unknown"
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Partials -- one per variant, digest-guarded, gitignored
# ---------------------------------------------------------------------------


def partial_path(partials: Path, variant: Variant) -> Path:
    return partials / f"{variant.name}.csv"


def write_partial(path: Path, table: pd.DataFrame, digests: dict[str, str]) -> None:
    """Write one variant's history with its digests in a ``#`` comment header.

    **17 significant digits, which is the shortest precision that round-trips a
    float64 exactly** -- and deliberately NOT the precision the assembled cache
    uses. A partial is an intermediate that may be read back and re-assembled, so
    a value that came back changed in the last ulp would make the committed cache
    depend on *whether a run was interrupted*, which is not a property that file
    may have. The assembled cache is a published artefact read by people and
    stays at 12.

    ``statistical_report`` found this first and ``vra_report``'s checkpoint writer
    found it independently; ``hybrid_report``'s copy had neither and wrote 12.
    Consolidating gives it to all four. :func:`read_partial` is the other half --
    they only work as a pair.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps(digests, indent=2, sort_keys=True)
    path.write_text(
        "".join(f"# {line}\n" for line in header.splitlines()) + table.to_csv(float_format="%.17g"),
        encoding="utf-8",
    )


def read_partial(path: Path) -> pd.DataFrame:
    """Read one partial, with its digests restored into ``frame.attrs``.

    ``float_precision="round_trip"``, not the default: pandas' fast C parser is
    not round-trip exact, so a resumed assembly would differ from an unbroken one
    in the last ulp even though :func:`write_partial` was exact. The pair only
    works together.
    """
    header: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            header.append(line[1:].strip())
    frame = pd.read_csv(
        path,
        comment="#",
        index_col="date",
        parse_dates=["date"],
        float_precision="round_trip",
    )
    frame.attrs.update(json.loads("\n".join(header)))
    return frame


def load_or_build(
    path: Path,
    digests: dict[str, str],
    compute: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """Return a matching partial, or compute one and write it.

    A partial whose digests do not match is **discarded rather than reused**.
    That is the whole reason resumption is safe to have: a resume that could
    splice two different models together would be worse than no resume at all,
    and W3-P5 moved a factor series once mid-session, so this is not
    hypothetical.

    Mid-column checkpointing is deliberately NOT here. Only ``vra_report`` has
    it, it needs the compute step to cooperate, and pushing it in would mean
    every caller paying for a feature one of them uses. A caller that wants it
    supplies a ``compute`` that does it.
    """
    if path.exists():
        existing = read_partial(path)
        if all(existing.attrs.get(key) == value for key, value in digests.items()):
            return existing
    table = compute()
    write_partial(path, table, digests)
    table.attrs.update(digests)
    return table


# ---------------------------------------------------------------------------
# The assembled cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cache:
    """The committed bias series and the provenance header that guards them."""

    provenance: dict[str, Any]
    frame: pd.DataFrame

    def bias(self, variant: Variant) -> np.ndarray:
        """One variant's ``B_t^F``, with the index union's padding removed.

        **Variants need not share a date range.** Model B's ``detoned`` panel
        loses the dates whose own ``K`` fell short of the maximum, so its history
        is shorter than ``denoised``'s and the columns are joined on the union of
        the two indices -- which pads the shorter one with NaN at the edges.
        Passing that padding into SPEC.md 5.4 would put NaN into ``lambda_F^2``
        and from there into every entry of the matrix, and the PSD assertion
        would report it as non-finite entries with nothing pointing back to here.

        Leading and trailing NaN are therefore dropped. An **interior** gap is
        not: that would mean a forecast date went missing inside a history built
        by an unbroken expanding loop, which is a defect rather than an alignment
        artefact, and it is refused rather than silently closed up.
        """
        series = self.frame[variant.name]
        present = series.notna().to_numpy()
        if not present.any():
            raise ValueError(f"{variant.name}: the cached bias series is entirely missing")
        first, last = int(np.argmax(present)), len(present) - int(np.argmax(present[::-1]))
        if not present[first:last].all():
            missing = int((~present[first:last]).sum())
            raise ValueError(
                f"{variant.name}: {missing} interior gap(s) in the cached bias series. The "
                "history is built by an unbroken expanding loop, so a hole inside it is a "
                "defect and not an index-alignment artefact. Rebuild the cache."
            )
        return np.asarray(series.to_numpy(dtype=float)[first:last])


def cache_provenance(
    *,
    what: str,
    settings: config.Config,
    panel_digests: dict[str, str] | str,
    variants: Sequence[Variant],
    rows: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The header every committed history carries. Same keys in all four.

    ``extra`` is for keys one caller carries and the others do not --
    ``vra_report`` records ``model_yaml_sha256``, the panel endpoints and the
    arithmetic minimum, and ``tests/test_regime.py`` asserts they are present.
    Merged last, so a caller can also override a shared key deliberately.
    """
    from datetime import UTC, datetime

    header: dict[str, Any] = {
        "_what": what,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "risk_config_digest": risk_config_digest(settings),
        "panel_digest": panel_digests,
        "holdout_start": str(settings.model.sample.holdout_start),
        "rows": rows,
        "columns": [variant.name for variant in variants],
    }
    header.update(extra or {})
    return header


def assemble(
    pieces: Iterable[pd.DataFrame],
    *,
    order: Sequence[str],
    what: str,
    settings: config.Config,
    panel_digests: dict[str, str] | str,
    variants: Sequence[Variant],
    extra: dict[str, Any] | None = None,
) -> Cache:
    """Join every partial into one cache, in a fixed column order.

    ``panel_digests`` is a map for the callers with one panel per variant, and a
    bare string for ``vra_report``, whose variants all share Model A's single
    panel. Both forms are carried through to the header as written, and
    :func:`read_cache` understands both.
    """
    table = pd.concat(list(pieces), axis=1)
    table = table.loc[:, ~table.columns.duplicated()]
    missing = [name for name in order if name not in table.columns]
    if missing:
        raise ValueError(f"assemble: the partials do not carry {missing}")
    table = table[list(order)]
    table.index.name = "date"
    return Cache(
        provenance=cache_provenance(
            what=what,
            settings=settings,
            panel_digests=panel_digests,
            variants=variants,
            rows=len(table),
            extra=extra,
        ),
        frame=table,
    )


def write_cache(cache: Cache, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps(cache.provenance, indent=2, sort_keys=True)
    body = cache.frame.to_csv(float_format="%.12g")
    path.write_text("".join(f"# {line}\n" for line in header.splitlines()) + body, encoding="utf-8")


def read_cache(
    path: Path,
    *,
    rebuild_hint: str,
    settings: config.Config | None = None,
    panel_digests: dict[str, str] | None = None,
) -> Cache:
    """Read a committed cache and REFUSE a stale one.

    Two digests, for the two ways it goes stale. ``risk_config_digest`` catches a
    changed half-life or lag count -- the obvious failure, and the one visible in
    ``config/model.yaml``. ``panel_digest`` catches a moved factor series, which
    is not visible there at all and is the failure W3-P5 actually hit.

    ``panel_digests`` is a parameter rather than recomputed here because building
    the panels costs real time and every caller about to run the pipeline already
    holds them. A caller that passes ``None`` gets the first guard only, and
    that is a deliberate weakening the caller is choosing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"forecast-history cache is missing: {path}. Rebuild: {rebuild_hint}"
        )
    header: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            header.append(line[1:].strip())
    provenance = json.loads("\n".join(header))
    frame = pd.read_csv(path, comment="#", index_col="date", parse_dates=["date"])

    current = risk_config_digest(settings)
    if provenance.get("risk_config_digest") != current:
        raise ValueError(
            f"{path.name} is STALE. It was built against risk config "
            f"{provenance.get('risk_config_digest')} and config/model.yaml now produces "
            f"{current}. Every number derived from it describes the old config. "
            f"Rebuild: {rebuild_hint}"
        )
    if panel_digests is not None:
        recorded = provenance.get("panel_digest", {})
        if isinstance(recorded, str):  # a single-panel cache written before the map form
            recorded = dict.fromkeys(panel_digests, recorded)
        stale = [key for key, value in panel_digests.items() if recorded.get(key) != value]
        if stale:
            raise ValueError(
                f"{path} was built against a different factor panel for {stale}. The risk "
                "config is unchanged, so nothing in config/model.yaml would show this -- the "
                f"factor series itself moved. Rebuild: {rebuild_hint}"
            )
    return Cache(provenance=provenance, frame=frame)


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


def fan_out(module: str, variants: Sequence[Variant], *, flag: str = "--column") -> int:
    """One process per variant. Returns a shell exit code.

    **Processes, not loop iterations.** The histories are independent -- each is
    its own expanding-window pipeline run over its own factor panel -- so nothing
    is shared and nothing has to be locked. Wall clock is one history rather than
    ``n``.

    The durability argument is as strong as the speed one: a single long process
    that is killed leaves nothing behind and looks exactly like one that has not
    finished, whereas a fanned-out run that is interrupted has already written
    every partial that completed.
    """
    running = [
        subprocess.Popen(
            [sys.executable, "-m", module, flag, variant.name],
            cwd=_ROOT,
        )
        for variant in variants
    ]
    codes = [process.wait() for process in running]
    if any(codes):
        print(f"history build(s) failed with {codes}")
        return 1
    return 0
