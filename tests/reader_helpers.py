"""Helpers for building reader requests in tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mhvsr_vs30.contracts import MediaType, RawAsset, Recording
from mhvsr_vs30.hashing import make_asset_id, make_recording_id, make_site_id, sha256_file
from mhvsr_vs30.io.base import ReadRequest

SOURCE_ID = "test_source"
SITE_CODE = "XX.AAA"


def make_request(
    path: Path, raw_format: str, root: Path | None = None, **overrides: Any
) -> ReadRequest:
    """Build a ReadRequest for a file, as the CLI would from a manifest row."""
    base = root or path.parent
    relative_path = path.relative_to(base).as_posix()
    site_id = make_site_id(SOURCE_ID, SITE_CODE)

    asset = RawAsset(
        asset_id=make_asset_id(SOURCE_ID, relative_path),
        source_id=SOURCE_ID,
        relative_path=relative_path,
        media_type=MediaType.AMBIENT_NOISE_WAVEFORM,
        byte_size=path.stat().st_size,
        sha256=sha256_file(path),
    )
    recording = Recording(
        recording_id=make_recording_id(SOURCE_ID, site_id, relative_path),
        site_id=site_id,
        asset_id=asset.asset_id,
        raw_format=raw_format,
        sampling_rate_hz=overrides.get("declared_sampling_rate_hz"),
        channel_mapping={"Z": "column_1", "N": "column_2", "E": "column_3"},
        units=overrides.get("units"),
    )
    return ReadRequest(recording=recording, asset=asset, path=path, root=base, **overrides)
