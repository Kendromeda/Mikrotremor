"""Turn a manifest row into something a reader can open.

A manifest row names an asset, not a file path. Resolution is kept here so that
readers stay ignorant of manifests and manifests stay ignorant of readers.
"""

from __future__ import annotations

from pathlib import Path

from mhvsr_vs30.contracts import Recording
from mhvsr_vs30.exceptions import ConfigError, RecordingReadError
from mhvsr_vs30.io.base import ReadRequest
from mhvsr_vs30.manifest.schemas import Manifest, SourceConfig

__all__ = ["resolve_request"]

_BUNDLE_FORMATS = {"sac_bundle", "gcf"}


def _siblings(path: Path, config: SourceConfig, raw_format: str) -> tuple[Path, ...]:
    if raw_format not in _BUNDLE_FORMATS:
        return ()
    if config.recording.bundle_group_by != "directory":
        raise ConfigError(
            f"format {raw_format} stores one component per file, so the source config "
            "must declare bundle_group_by"
        )
    return tuple(sorted(p for p in path.parent.iterdir() if p.is_file()))


def resolve_request(
    manifest: Manifest,
    recording: Recording,
    config: SourceConfig,
    workspace: Path | str = Path("."),
) -> ReadRequest:
    """Locate the bytes behind a recording and attach the documented metadata."""
    assets = {asset.asset_id: asset for asset in manifest.assets}
    asset = assets.get(recording.asset_id)
    if asset is None:
        raise RecordingReadError(
            f"recording {recording.recording_id[:12]} points at asset "
            f"{recording.asset_id[:12]}, which is not in this manifest"
        )

    root = Path(workspace) / config.root
    path = root / asset.relative_path
    if not path.is_file():
        raise RecordingReadError(f"{asset.relative_path} is missing from {root}")

    return ReadRequest(
        recording=recording,
        asset=asset,
        path=path,
        root=root,
        channel_codes=config.recording.channel_codes,
        component_traces=config.recording.component_traces,
        curve_columns=config.recording.curve_columns,
        siblings=_siblings(path, config, recording.raw_format),
        declared_sampling_rate_hz=recording.sampling_rate_hz,
        units=recording.units,
        component_order=config.recording.component_order,
        drop_incomplete_final_row=config.recording.drop_incomplete_final_row,
    )
