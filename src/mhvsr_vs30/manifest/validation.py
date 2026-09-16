"""Manifest validation.

Every check returns a human-readable failure string rather than raising, so one
run reports every problem at once. The caller decides whether a non-empty list
is fatal; for builds, it always is.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from mhvsr_vs30.manifest.schemas import Manifest, SourceConfig

__all__ = ["validate_manifest"]


def _duplicates(values: Sequence[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _check_identity(manifest: Manifest) -> list[str]:
    failures: list[str] = []
    for table, ids in (
        ("sites", [site.site_id for site in manifest.sites]),
        ("assets", [asset.asset_id for asset in manifest.assets]),
        ("recordings", [r.recording_id for r in manifest.recordings]),
        ("labels", [label.label_id for label in manifest.labels]),
    ):
        duplicated = _duplicates(ids)
        if duplicated:
            failures.append(f"{table}: duplicate ids {duplicated[:5]}")
    return failures


def _check_references(manifest: Manifest) -> list[str]:
    failures: list[str] = []
    site_ids = {site.site_id for site in manifest.sites}
    asset_ids = {asset.asset_id for asset in manifest.assets}

    for recording in manifest.recordings:
        if recording.site_id not in site_ids:
            failures.append(
                f"recording {recording.recording_id[:12]} has site_id {recording.site_id} "
                "which is not in the sites table"
            )
        if recording.asset_id not in asset_ids:
            failures.append(
                f"recording {recording.recording_id[:12]} has asset_id "
                f"{recording.asset_id[:12]} which is not in the assets table"
            )

    for label in manifest.labels:
        if label.site_id not in site_ids:
            failures.append(
                f"label {label.label_id[:12]} has site_id {label.site_id} "
                "which is not in the sites table"
            )
    return failures


def _check_source_consistency(manifest: Manifest) -> list[str]:
    failures: list[str] = []
    source_id = manifest.source.source_id

    for asset in manifest.assets:
        if asset.source_id != source_id:
            failures.append(f"asset {asset.relative_path} belongs to source {asset.source_id}")
        if asset.relative_path.startswith("/") or ".." in asset.relative_path.split("/"):
            failures.append(f"asset path escapes the source root: {asset.relative_path}")
        if len(asset.sha256) != 64:
            failures.append(f"asset {asset.relative_path} has no usable checksum")

    for site in manifest.sites:
        if site.source_id != source_id:
            failures.append(f"site {site.site_id} belongs to source {site.source_id}")
        if not site.site_id.startswith(f"{source_id}:"):
            failures.append(f"site {site.site_id} is not namespaced by {source_id}")
    return failures


def _check_physical_plausibility(manifest: Manifest, config: SourceConfig) -> list[str]:
    failures: list[str] = []
    low, high = config.vs30_range_mps

    for label in manifest.labels:
        if not low <= label.vs30_mps <= high:
            failures.append(
                f"label {label.label_id[:12]} for {label.site_id} has vs30 "
                f"{label.vs30_mps} m/s, outside the configured range {low}-{high} m/s"
            )

    declared = config.recording.sampling_rate_hz
    if declared is not None:
        for recording in manifest.recordings:
            if recording.sampling_rate_hz != declared:
                failures.append(
                    f"recording {recording.recording_id[:12]} claims "
                    f"{recording.sampling_rate_hz} Hz but the source documents {declared} Hz"
                )
    return failures


def _check_coverage(manifest: Manifest) -> list[str]:
    """Every configured site must actually be represented in the tree."""
    failures: list[str] = []
    per_site = Counter(recording.site_id for recording in manifest.recordings)
    labelled = {label.site_id for label in manifest.labels}

    for site in manifest.sites:
        if per_site.get(site.site_id, 0) == 0:
            failures.append(f"site {site.site_id} has no recordings under the source root")
        if site.site_id not in labelled:
            failures.append(f"site {site.site_id} has no Vs30 label")
    return failures


def _check_expectations(manifest: Manifest, config: SourceConfig) -> list[str]:
    """A count that differs from the documented one fails the build, not a warning.

    A silently shrinking corpus is the failure mode this project can least
    afford, because it shows up as a quietly different model, not as an error.
    """
    failures: list[str] = []
    expected = config.expectations

    observed = {
        "site_count": len(manifest.sites),
        "asset_count": len(manifest.assets),
        "recording_count": len(manifest.recordings),
    }
    for name, actual in observed.items():
        wanted = getattr(expected, name)
        if wanted is not None and wanted != actual:
            failures.append(f"{name}: expected {wanted}, found {actual}")

    if expected.recordings_per_site is not None:
        per_site = Counter(recording.site_id for recording in manifest.recordings)
        for site in manifest.sites:
            actual_count = per_site.get(site.site_id, 0)
            if actual_count != expected.recordings_per_site:
                failures.append(
                    f"recordings_per_site: {site.site_id} has {actual_count}, "
                    f"expected {expected.recordings_per_site}"
                )
    return failures


def validate_manifest(manifest: Manifest, config: SourceConfig) -> list[str]:
    """Return every reason this manifest should not be trusted."""
    return [
        *_check_identity(manifest),
        *_check_references(manifest),
        *_check_source_consistency(manifest),
        *_check_physical_plausibility(manifest, config),
        *_check_coverage(manifest),
        *_check_expectations(manifest, config),
    ]
