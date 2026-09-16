"""Deterministic identifiers and file checksums.

Identifiers are derived from content and provenance, never from a counter, so
that rebuilding a manifest from the same tree yields the same ids and two
snapshots can be diffed row by row.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = [
    "make_asset_id",
    "make_label_id",
    "make_recording_id",
    "make_site_id",
    "sha256_file",
    "stable_id",
]

# ASCII record separator: cannot occur in a path, a site code or an id, so a
# field can never impersonate a field boundary.
_SEPARATOR = "\x1e"

_DEFAULT_CHUNK_SIZE = 1 << 20


def stable_id(*parts: str) -> str:
    """Hash an ordered sequence of fields into a lowercase hex SHA-256."""
    if not parts:
        raise ValueError("stable_id needs at least one part")
    digest = hashlib.sha256()
    digest.update(_SEPARATOR.join(parts).encode("utf-8"))
    return digest.hexdigest()


def _normalise_path(relative_path: str) -> str:
    return relative_path.strip().replace("\\", "/")


def make_site_id(source_id: str, site_code: str) -> str:
    """Namespace a site code by its source.

    Site ids stay human-readable because they are the join key an analyst reads
    most often, and because a collision across sources would be a merge this
    project refuses to make implicitly.
    """
    code = " ".join(site_code.split()).upper()
    if not code:
        raise ValueError("site code must not be empty")
    return f"{source_id}:{code}"


def make_asset_id(source_id: str, relative_path: str) -> str:
    return stable_id(source_id, _normalise_path(relative_path))


def make_recording_id(
    source_id: str,
    site_id: str,
    relative_path: str,
    segment: str | None = None,
) -> str:
    """Identify one recording, optionally one segment of a multi-segment file."""
    return stable_id(source_id, site_id, _normalise_path(relative_path), segment or "")


def make_label_id(
    site_id: str,
    label_method: str,
    reference: str,
    vs30_mps: float,
) -> str:
    """Identify a label by site, method, citation and value.

    Changing any of those is a different claim and therefore a different label.
    """
    return stable_id(site_id, label_method, reference, repr(float(vs30_mps)))


def sha256_file(path: Path | str, chunk_size: int = _DEFAULT_CHUNK_SIZE) -> str:
    """Checksum a file without reading it all into memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
