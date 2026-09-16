"""Assemble a training snapshot from curves, recordings and labels.

The join is by identifier only. Nothing is matched on file name, on proximity,
or on anything the file tree happens to imply, so a row exists only when a curve,
a recording and a cited label all agree on the same site.
"""

from __future__ import annotations

import json
from collections import Counter
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

__all__ = ["DatasetSnapshot", "build_dataset"]

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
