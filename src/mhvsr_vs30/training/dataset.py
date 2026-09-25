"""Assemble a training snapshot from curves, recordings and labels.

The join is by identifier only. Nothing is matched on file name, on proximity,
or on anything the file tree happens to imply, so a row exists only when a curve,
a recording and a cited label all agree on the same site.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mhvsr_vs30.exceptions import MhvsrVs30Error
from mhvsr_vs30.features.hvsr import FEATURE_COUNT, feature_columns
from mhvsr_vs30.manifest.schemas import Manifest
from mhvsr_vs30.preprocessing.config import PreprocessingProfile
from mhvsr_vs30.preprocessing.store import curve_directory, read_curve_arrays
from mhvsr_vs30.provenance import environment_provenance

__all__ = ["DatasetSnapshot", "build_dataset", "combine_snapshots", "read_dataset_snapshot"]

FEATURE_COLUMNS = feature_columns()


class DatasetError(MhvsrVs30Error):
    """A snapshot could not be assembled from the available artifacts."""


class DatasetSnapshot:
    """A table of recordings with features, labels and per-site weights."""

    def __init__(self, frame: pd.DataFrame, metadata: dict[str, Any]) -> None:
        self.frame = frame
        self.metadata = metadata

    @property
    def sites(self) -> list[str]:
        return sorted(self.frame["site_id"].unique())

    def features(self) -> np.ndarray:
        return self.frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)

    def targets(self) -> np.ndarray:
        return self.frame["vs30_mps"].to_numpy(dtype=np.float64)

    def weights(self) -> np.ndarray:
        return self.frame["sample_weight"].to_numpy(dtype=np.float64)

    def groups(self) -> np.ndarray:
        return self.frame["site_id"].to_numpy()

    def write(self, directory: Path | str) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(target / "dataset.parquet", index=False)
        (target / "dataset.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return target


def read_dataset_snapshot(directory: Path | str) -> DatasetSnapshot:
    """Load a snapshot written by :meth:`DatasetSnapshot.write` for evaluation."""
    source = Path(directory)
    parquet_path = source / "dataset.parquet"
    metadata_path = source / "dataset.json"
    if not parquet_path.is_file() or not metadata_path.is_file():
        raise DatasetError(f"snapshot directory needs dataset.parquet and dataset.json: {source}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"invalid snapshot metadata JSON: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise DatasetError(f"snapshot metadata must be a JSON object: {metadata_path}")
    return DatasetSnapshot(pd.read_parquet(parquet_path), metadata)


def combine_snapshots(
    snapshots: Sequence[DatasetSnapshot], *, site_crosswalk: dict[str, str] | None = None
) -> DatasetSnapshot:
    """Combine independently built snapshots for cross-study evaluation.

    Inputs must use the same non-empty preprocessing profile. Recording
    identifiers stay globally unique. Cross-source inputs need either an
    explicit ``canonical_site_id`` column or a ``source_id::site_id`` crosswalk;
    proximity is deliberately never used to merge physical sites.
    """
    if len(snapshots) < 2:
        raise DatasetError("combine_snapshots needs at least 2 snapshots")

    frames: list[pd.DataFrame] = []
    input_metadata: list[dict[str, Any]] = []
    profile_keys: set[str] = set()
    input_source_ids: set[str] = set()
    for index, snapshot in enumerate(snapshots):
        _validate_snapshot_for_combination(snapshot, index)
        profile = snapshot.metadata.get("profile_identity")
        if profile is None or profile == "" or profile == {}:
            raise DatasetError(f"snapshot {index} lacks non-empty profile_identity provenance")
        profile_keys.add(json.dumps(profile, sort_keys=True))
        frames.append(snapshot.frame.copy(deep=True))
        input_metadata.append(deepcopy(snapshot.metadata))
        input_source_ids.add(str(snapshot.metadata["source_id"]))

    if len(profile_keys) > 1:
        raise DatasetError("cannot combine snapshots made with different preprocessing profiles")

    frame = pd.concat(frames, ignore_index=True)
    _apply_canonical_site_ids(frame, site_crosswalk, require_canonical=len(input_source_ids) > 1)
    duplicate_recordings = sorted(
        frame.loc[frame["recording_id"].duplicated(keep=False), "recording_id"].unique().tolist()
    )
    if duplicate_recordings:
        raise DatasetError(
            f"duplicate recording_id values across snapshots: {duplicate_recordings}"
        )
    _validate_site_consistency(frame)

    frame = frame.sort_values("recording_id").reset_index(drop=True)
    per_site = frame["site_id"].value_counts()
    frame["sample_weight"] = frame["site_id"].map(lambda site: 1.0 / float(per_site[site]))

    source_ids = sorted({str(metadata["source_id"]) for metadata in input_metadata})
    metadata: dict[str, Any] = {
        "source_id": "combined",
        "source_ids": source_ids,
        "input_snapshots": input_metadata,
        "profile_identity": deepcopy(snapshots[0].metadata.get("profile_identity")),
        "row_count": len(frame),
        "site_count": int(frame["site_id"].nunique()),
        "recordings_per_site": {
            site: int(count) for site, count in frame["site_id"].value_counts().sort_index().items()
        },
        "vs30_by_site": {
            site: float(value)
            for site, value in frame.groupby("site_id")["vs30_mps"].first().sort_index().items()
        },
        "label_independence": {
            key: int(value) for key, value in frame["label_independence"].value_counts().items()
        },
        "feature_count": FEATURE_COUNT,
    }
    metadata.update(environment_provenance())
    return DatasetSnapshot(frame, metadata)


def _validate_snapshot_for_combination(snapshot: DatasetSnapshot, index: int) -> None:
    required = {"recording_id", "site_id", "vs30_mps", "label_independence"} | set(FEATURE_COLUMNS)
    missing = sorted(required - set(snapshot.frame.columns))
    if missing:
        raise DatasetError(f"snapshot {index} lacks required columns: {missing}")
    source_id = snapshot.metadata.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        raise DatasetError(f"snapshot {index} lacks source_id provenance")


def _validate_site_consistency(frame: pd.DataFrame) -> None:
    """Refuse contradictory labels or grouping metadata for one physical site."""
    invariant_columns = (
        "country",
        "geography_id",
        "vs30_mps",
        "label_independence",
    )
    present = [column for column in invariant_columns if column in frame.columns]
    conflicts: dict[str, list[str]] = {}
    for site_id, group in frame.groupby("site_id", sort=True):
        inconsistent = [column for column in present if group[column].nunique(dropna=False) != 1]
        if inconsistent:
            conflicts[str(site_id)] = inconsistent
    if conflicts:
        raise DatasetError(f"conflicting metadata for physical sites: {conflicts}")


def _apply_canonical_site_ids(
    frame: pd.DataFrame,
    site_crosswalk: dict[str, str] | None,
    *,
    require_canonical: bool,
) -> None:
    """Apply an audited site crosswalk before any physical-site validation."""
    has_canonical = "canonical_site_id" in frame.columns
    if require_canonical and not has_canonical and site_crosswalk is None:
        raise DatasetError(
            "cross-source combination requires canonical_site_id metadata "
            "or an explicit site_crosswalk"
        )
    if site_crosswalk is not None:
        source_sites = frame["source_id"].astype(str) + "::" + frame["site_id"].astype(str)
        mapped = source_sites.map(site_crosswalk)
        if mapped.isna().any() or any(not str(value).strip() for value in mapped):
            missing = sorted(source_sites[mapped.isna()].unique().tolist())
            raise DatasetError(f"site_crosswalk has no canonical site for: {missing}")
        frame["canonical_site_id"] = mapped.astype(str)
        has_canonical = True
    if has_canonical:
        canonical = frame["canonical_site_id"]
        if canonical.isna().any() or any(not str(value).strip() for value in canonical):
            raise DatasetError("canonical_site_id must be present and non-empty for every row")
        frame["source_site_id"] = frame["site_id"].astype(str)
        frame["site_id"] = canonical.astype(str)


def build_dataset(
    manifest: Manifest,
    profile: PreprocessingProfile,
    curves_root: Path | str = Path("artifacts/curves"),
    manifest_dir: Path | str | None = None,
    require_independent_labels: bool = True,
) -> DatasetSnapshot:
    """Join model-ready curves to their sites and labels.

    ``require_independent_labels`` is on by default: a Vs30 derived from the same
    HVSR the model will read is circular, and letting it in would make any score
    computed afterwards meaningless.
    """
    labels_by_site = {label.site_id: label for label in manifest.labels}
    recordings = {r.recording_id: r for r in manifest.recordings}
    sites_by_id = {site.site_id: site for site in manifest.sites}

    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for recording_id, recording in sorted(recordings.items()):
        directory = curve_directory(curves_root, profile.profile_id, recording_id)
        if not directory.is_dir():
            skipped["no_curve"] += 1
            continue

        label = labels_by_site.get(recording.site_id)
        if label is None:
            skipped["no_label"] += 1
            continue
        site = sites_by_id.get(recording.site_id)
        if site is None:
            skipped["no_site_metadata"] += 1
            continue
        independence = str(label.label_independence)
        if require_independent_labels and independence != "independent_of_hvsr":
            skipped[f"label_{independence}"] += 1
            continue

        arrays = read_curve_arrays(directory)
        if "model_amplitude" not in arrays:
            skipped["not_model_ready"] += 1
            continue
        features = np.asarray(arrays["model_amplitude"], dtype=np.float64)
        if features.shape != (FEATURE_COUNT,):
            skipped["bad_feature_shape"] += 1
            continue

        qc = json.loads((directory / "qc.json").read_text(encoding="utf-8"))
        provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))

        row: dict[str, Any] = {
            "recording_id": recording_id,
            "site_id": recording.site_id,
            "source_id": manifest.source.source_id,
            # A source is the smallest cited study unit available in the
            # manifest. More granular studies can replace this column when
            # combining snapshots, but a source must never be split across
            # train/test merely because it contains multiple recordings.
            "study_id": f"source:{manifest.source.source_id}",
            "country": str(site.country),
            "geography_id": f"country:{site.country}",
            "label_id": label.label_id,
            "vs30_mps": label.vs30_mps,
            "label_method": label.label_method,
            "label_independence": independence,
            "qc_status": qc.get("qc_status"),
            "qc_flags": ";".join(qc.get("qc_flags", [])),
            "processing_hash": provenance.get("processing_hash"),
            "input_sha256": provenance.get("input_sha256"),
        }
        row.update(dict(zip(FEATURE_COLUMNS, features, strict=True)))
        rows.append(row)

    if not rows:
        raise DatasetError(
            f"no model-ready curves joined to an eligible label for profile "
            f"{profile.profile_id}; skipped: {dict(skipped)}"
        )

    frame = pd.DataFrame(rows)
    # One site contributes one unit of weight however many times it was recorded,
    # so a site with six recordings cannot outvote a site with one.
    per_site = frame["site_id"].value_counts()
    frame["sample_weight"] = frame["site_id"].map(lambda s: 1.0 / float(per_site[s]))
    frame = frame.sort_values("recording_id").reset_index(drop=True)

    metadata = _metadata(frame, manifest, profile, skipped, manifest_dir)
    return DatasetSnapshot(frame, metadata)


def _metadata(
    frame: pd.DataFrame,
    manifest: Manifest,
    profile: PreprocessingProfile,
    skipped: Counter[str],
    manifest_dir: Path | str | None,
) -> dict[str, Any]:
    snapshot_hash = None
    if manifest_dir is not None:
        path = Path(manifest_dir) / "snapshot.json"
        if path.is_file():
            snapshot_hash = json.loads(path.read_text(encoding="utf-8")).get("config_hash")

    metadata: dict[str, Any] = {
        "source_id": manifest.source.source_id,
        "profile_id": profile.profile_id,
        "profile_identity": profile.identity(),
        "manifest_snapshot_hash": snapshot_hash,
        "manifest_dir": str(manifest_dir) if manifest_dir is not None else None,
        "row_count": len(frame),
        "site_count": int(frame["site_id"].nunique()),
        "recordings_per_site": {
            site: int(count) for site, count in frame["site_id"].value_counts().sort_index().items()
        },
        "vs30_by_site": {
            site: float(value)
            for site, value in frame.groupby("site_id")["vs30_mps"].first().sort_index().items()
        },
        "label_independence": {
            key: int(value) for key, value in frame["label_independence"].value_counts().items()
        },
        "skipped": dict(sorted(skipped.items())),
        "feature_count": FEATURE_COUNT,
    }
    metadata.update(environment_provenance())
    return metadata
