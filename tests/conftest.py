"""Shared test fixtures.

The autouse network guard is the mechanical enforcement of CLAUDE.md invariant
1: only ``mafrm.data.loaders`` may touch the network, and it does so at
``make data`` time, never during a test. Any test that genuinely needs a live
source must say so with ``@pytest.mark.network`` -- those are deselected by the
fast CI workflow and run weekly by ``data-contracts.yml``.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest

from mafrm.data import cache


class NetworkAccessBlocked(RuntimeError):
    """A test tried to open a socket without the ``network`` marker."""


@pytest.fixture(autouse=True)
def block_network(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail loudly on any socket opened by an unmarked test."""
    if request.node.get_closest_marker("network") is not None:
        yield
        return

    def guard(*args: Any, **kwargs: Any) -> Any:
        raise NetworkAccessBlocked(
            "network access from an unmarked test. CLAUDE.md invariant 1: only "
            "mafrm.data.loaders may fetch, and tests read from fixtures. Mark the "
            "test with @pytest.mark.network if it is a data-contract test."
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(socket, "socket", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    try:
        yield
    finally:
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# The local data cache
# ---------------------------------------------------------------------------
#
# `dataset`-marked tests read `data/raw`, which is gitignored and absent on a
# clean clone. They are deselected by `make test` and by the fast CI workflow
# (`-m "not network and not dataset"`) and run weekly by data-contracts.yml
# after `make data`. Where an individual source failed to refresh -- Stooq is
# behind a browser challenge, and FRED needs a key CI may not hold -- the
# helpers below SKIP with the reason rather than failing, so one unavailable
# vendor does not mask a real contract breach in another.


_CONTRACT_REASON = "data-contract test: source coverage and release history, not model output"


def cached(source: str, name: str) -> pd.DataFrame:
    """Read one cached table, or skip the test naming what is missing."""
    manifest = cache.Manifest.load()
    try:
        entry = manifest.latest(source=source, name=name)
    except cache.CacheError as exc:
        pytest.skip(f"no cached {source}/{name}: {exc}")
    try:
        # UNRESTRICTED on purpose. `dataset`-marked tests are data CONTRACT tests:
        # they ask what a source actually published -- its coverage, its end date,
        # its release cadence -- which is a question about the vendor and not
        # about the model. Model-facing reads go through `cache.read`, which stops
        # at the holdout boundary by default (mafrm.data.holdout).
        return cache.read_unrestricted(entry, manifest=manifest, reason=_CONTRACT_REASON)
    except cache.CacheError as exc:  # pragma: no cover - drift is `make verify`'s job
        pytest.skip(f"cached {source}/{name} does not match the manifest: {exc}")
