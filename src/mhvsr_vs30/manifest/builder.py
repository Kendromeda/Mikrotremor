"""Discovery and assembly of a recording-level manifest.

Nothing in this module opens a raw recording for its samples, and nothing
writes into the source tree. Discovery reads bytes to checksum them and
metadata to size them; that is all.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from mhvsr_vs30.contracts import MediaType, RawAsset, Recording, Site, Vs30Label
from mhvsr_vs30.exceptions import ConfigError, ManifestValidationError
from mhvsr_vs30.hashing import (
    make_asset_id,
    make_label_id,
    make_recording_id,
    make_site_id,
    sha256_file,
)
from mhvsr_vs30.manifest.schemas import Manifest, SiteConfig, SourceConfig
from mhvsr_vs30.manifest.validation import validate_manifest

__all__ = ["build_manifest", "classify_asset", "discover_assets"]

CACHE_VERSION = 2

# Conservative fallbacks. Anything whose meaning depends on the study belongs in
# the asset_rules of that study, not here, so an unrecognised file stays UNKNOWN
# rather than being silently promoted into training data.
_EXTENSION_DEFAULTS: dict[str, MediaType] = {
    ".jpg": MediaType.IMAGE,
    ".jpeg": MediaType.IMAGE,
    ".png": MediaType.IMAGE,
    ".tif": MediaType.IMAGE,
    ".tiff": MediaType.IMAGE,
    ".pdf": MediaType.SITE_REPORT,
    ".doc": MediaType.SITE_METADATA,
    ".docx": MediaType.SITE_METADATA,
    ".xls": MediaType.SITE_METADATA,
    ".xlsx": MediaType.SITE_METADATA,
    ".csv": MediaType.SITE_METADATA,
    ".kmz": MediaType.SITE_METADATA,
    ".zip": MediaType.ARCHIVE,
    ".gz": MediaType.ARCHIVE,
    ".tar": MediaType.ARCHIVE,
    ".7z": MediaType.ARCHIVE,
}


def _matches_any(relative_path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatchcase(relative_path, pattern) for pattern in patterns)


def classify_asset(relative_path: str, config: SourceConfig) -> MediaType:
    """Decide what a file is, preferring the documented rules of the study."""
    for rule in config.asset_rules:
        if fnmatchcase(relative_path, rule.pattern):
            return rule.media_type
    return _EXTENSION_DEFAULTS.get(Path(relative_path).suffix.lower(), MediaType.UNKNOWN)


def _load_cache(cache_path: Path | None) -> dict[str, dict[str, Any]]:
    if cache_path is None or not cache_path.is_file():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
        return {}
    entries = payload.get("entries")
    return entries if isinstance(entries, dict) else {}


def _save_cache(cache_path: Path | None, entries: Mapping[str, dict[str, Any]]) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"version": CACHE_VERSION, "entries": dict(sorted(entries.items()))}, indent=2),
        encoding="utf-8",
    )


def discover_assets(
    config: SourceConfig,
    workspace: Path | str = Path("."),
    cache_path: Path | str | None = None,
) -> tuple[RawAsset, ...]:
    """Record every file under the source root, checksum included.

    Checksums are cached on size and modification time so re-running a build
    over a multi-gigabyte tree does not re-read every byte of it.
    """
    root = Path(workspace) / config.root
    if not root.is_dir():
        raise ConfigError(f"source root does not exist: {root}")

    cache_file = Path(cache_path) if cache_path is not None else None
    cache = _load_cache(cache_file)
    fresh: dict[str, dict[str, Any]] = {}

    assets: list[RawAsset] = []
    for path in sorted(root.rglob("*"), key=lambda p: p.as_posix()):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        if _matches_any(relative_path, config.exclude):
            continue

        stat = path.stat()
        key = f"{config.source_id}/{relative_path}"
        cached = cache.get(key)
        if (
            isinstance(cached, dict)
            and cached.get("byte_size") == stat.st_size
            and cached.get("mtime_ns") == stat.st_mtime_ns
            and isinstance(cached.get("sha256"), str)
        ):
            checksum = str(cached["sha256"])
        else:
            checksum = sha256_file(path)

        fresh[key] = {
            "byte_size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": checksum,
        }
        assets.append(
            RawAsset(
                asset_id=make_asset_id(config.source_id, relative_path),
                source_id=config.source_id,
                relative_path=relative_path,
                media_type=classify_asset(relative_path, config),
                byte_size=stat.st_size,
                sha256=checksum,
            )
        )

    _save_cache(cache_file, fresh)
    return tuple(assets)


def _build_sites(config: SourceConfig) -> tuple[Site, ...]:
    return tuple(
        Site(
            site_id=make_site_id(config.source_id, site.site_code),
            source_id=config.source_id,
            latitude=site.latitude,
            longitude=site.longitude,
            elevation_m=site.elevation_m,
            country=config.country,
            location_accuracy_m=site.location_accuracy_m,
            site_code=site.site_code,
        )
        for site in config.sites
    )


def _owning_sites(relative_path: str, config: SourceConfig) -> list[SiteConfig]:
    return [site for site in config.sites if fnmatchcase(relative_path, site.effective_path_glob())]


def _build_recordings(
    config: SourceConfig,
    assets: Iterable[RawAsset],
) -> tuple[tuple[Recording, ...], list[str]]:
    """Promote ambient-noise assets to recordings, attributing each to one site.

    A file that no configured site claims, or that two sites claim, is reported
    rather than guessed at: both cases mean the config is wrong about the tree.
    """
    recordings: list[Recording] = []
    failures: list[str] = []

    for asset in assets:
        if asset.media_type is not MediaType.AMBIENT_NOISE_WAVEFORM:
            continue
        if config.recording.include and not _matches_any(
            asset.relative_path, config.recording.include
        ):
            continue
        if _matches_any(asset.relative_path, config.recording.exclude):
            continue

        owners = _owning_sites(asset.relative_path, config)
        if len(owners) != 1:
            failures.append(
                f"{asset.relative_path} is claimed by {len(owners)} configured sites; "
                "exactly one site must own each recording"
            )
            continue

        site_id = make_site_id(config.source_id, owners[0].site_code)
        recordings.append(
            Recording(
                recording_id=make_recording_id(config.source_id, site_id, asset.relative_path),
                site_id=site_id,
                asset_id=asset.asset_id,
                raw_format=config.recording.format,
                sampling_rate_hz=config.recording.sampling_rate_hz,
                start_time=None,
                duration_s=None,
                channel_mapping=config.recording.channel_mapping(),
                units=config.recording.units,
                segment=None,
            )
        )

    return tuple(recordings), failures


def _build_labels(config: SourceConfig) -> tuple[Vs30Label, ...]:
    labels: list[Vs30Label] = []
    for site in config.sites:
        if site.label is None:
            continue
        site_id = make_site_id(config.source_id, site.site_code)
        labels.append(
            Vs30Label(
                label_id=make_label_id(
                    site_id, site.label.label_method, site.label.reference, site.label.vs30_mps
                ),
                site_id=site_id,
                vs30_mps=site.label.vs30_mps,
                vs30_std_mps=site.label.vs30_std_mps,
                label_method=site.label.label_method,
                label_independence=site.label.label_independence,
                reference=site.label.reference,
                reference_locator=site.label.reference_locator,
            )
        )
    return tuple(labels)


def build_manifest(
    config: SourceConfig,
    workspace: Path | str = Path("."),
    cache_path: Path | str | None = None,
) -> Manifest:
    """Build and validate a manifest, refusing to return an inconsistent one."""
    assets = discover_assets(config, workspace, cache_path=cache_path)
    recordings, attribution_failures = _build_recordings(config, assets)

    manifest = Manifest(
        source=config.to_source(),
        sites=_build_sites(config),
        assets=assets,
        recordings=recordings,
        labels=_build_labels(config),
    )

    failures = attribution_failures + validate_manifest(manifest, config)
    if failures:
        listed = "\n".join(f"  - {failure}" for failure in failures)
        raise ManifestValidationError(
            f"manifest for {config.source_id} failed {len(failures)} check(s):\n{listed}",
            failures,
        )
    return manifest
