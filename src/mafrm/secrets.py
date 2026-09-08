"""Secrets from the environment, or from a gitignored ``.env``.

CLAUDE.md, version control: "Secrets come from environment variables loaded from
a gitignored ``.env``. Never a literal in code, never a default value in the
source." This module is the only place that reads them, and it has no default
for anything -- a missing key raises with the name of the variable and where to
get one, rather than falling through to an anonymous request that will fail
later with a less useful error.

The real environment wins over ``.env``. That is what lets CI inject a key as a
repository secret without a file, and what lets a developer override one for a
single command without editing anything.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Final

__all__ = ["MissingSecretError", "env_file_path", "parse_env_file", "require"]

# Repo root: src/mafrm/secrets.py -> src/mafrm -> src -> <root>.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


class MissingSecretError(RuntimeError):
    """A required secret is in neither the environment nor ``.env``."""


def env_file_path() -> Path:
    """Location of the gitignored ``.env``. ``MAFRM_ENV_FILE`` overrides (tests do)."""
    override = os.environ.get("MAFRM_ENV_FILE")
    return Path(override) if override else _REPO_ROOT / ".env"


def parse_env_file(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` lines. Blank lines and ``#`` comments are skipped.

    Deliberately not a shell parser: no interpolation, no ``export``, no multi-line
    values. A quoted value has one matching pair of surrounding quotes stripped, so
    that a key with a trailing ``#`` or spaces can be written unambiguously.
    """
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


@lru_cache(maxsize=1)
def _env_file_values(path: str, mtime: float) -> dict[str, str]:
    # Keyed on (path, mtime) so an edited .env is picked up without a restart
    # while a hot loop does not re-read the file on every call.
    del mtime
    return parse_env_file(Path(path).read_text(encoding="utf-8"))


def require(name: str, *, hint: str = "") -> str:
    """Return the secret ``name``, or raise :class:`MissingSecretError`.

    Order: the process environment first, then ``.env``. An empty value counts as
    missing -- an unset secret and a secret set to the empty string fail the same
    way upstream, and failing here names the problem.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        path = env_file_path()
        if path.is_file():
            value = _env_file_values(str(path), path.stat().st_mtime).get(name, "").strip()
    if not value:
        suffix = f" {hint}" if hint else ""
        raise MissingSecretError(
            f"{name} is not set. Put it in the environment or in {env_file_path()} "
            f"(gitignored; copy .env.example).{suffix}"
        )
    return value
