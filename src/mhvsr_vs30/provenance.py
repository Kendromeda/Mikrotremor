"""Provenance for anything this pipeline writes.

Every artifact records enough to answer one question later: which code, which
config, which dependency set, and which interpreter produced these bytes. A
dirty working tree is recorded rather than hidden, because an artifact built
from uncommitted code is not reproducible and should say so.
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any

from mhvsr_vs30 import __version__
from mhvsr_vs30.hashing import sha256_file, stable_id

__all__ = ["environment_provenance", "source_tree_hash"]

_SOURCE_ROOT = Path(__file__).resolve().parent


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def source_tree_hash(root: Path | None = None) -> str:
    """Hash every tracked Python source file, by path and content.

    This does not depend on git, so it stays meaningful for an uncommitted tree
    and it changes the moment any source file changes.
    """
    base = Path(root) if root is not None else _SOURCE_ROOT
    parts: list[str] = []
    for path in sorted(base.rglob("*.py"), key=lambda p: p.as_posix()):
        if "__pycache__" in path.parts:
            continue
        parts.append(path.relative_to(base).as_posix())
        parts.append(sha256_file(path))
    if not parts:
        return stable_id("empty-source-tree")
    return stable_id(*parts)


def environment_provenance(lockfile: Path | str = Path("uv.lock")) -> dict[str, Any]:
    """Collect the identity of the code and environment doing the work."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    lock_path = Path(lockfile)

    return {
        "tool_version": __version__,
        "code_commit": commit or "unknown",
        "code_dirty": bool(status) if status is not None else None,
        "source_tree_hash": source_tree_hash(),
        "lockfile_hash": sha256_file(lock_path) if lock_path.is_file() else None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
