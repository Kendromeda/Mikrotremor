"""Atomic artifact writing for processed curves.

Each recording gets its own directory holding the arrays, the QC record and the
provenance needed to rebuild it. Writes go to a temporary directory and are
moved into place, so an interrupted run leaves either the previous version or
nothing, never a half-written curve that looks complete.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from mhvsr_vs30.preprocessing.config import PreprocessingProfile
from mhvsr_vs30.preprocessing.contracts import ProcessedHvsr
from mhvsr_vs30.provenance import environment_provenance

__all__ = ["curve_directory", "read_curve_arrays", "write_curve"]

ARRAY_NAMES = (
    "frequency_hz",
    "window_amplitudes",
    "accepted_window_mask",
    "mean_curve",
    "log_std_curve",
    "model_frequency_hz",
)


def curve_directory(root: Path | str, profile_id: str, recording_id: str) -> Path:
    return Path(root) / profile_id / recording_id


def write_curve(
    curve: ProcessedHvsr,
    profile: PreprocessingProfile,
    root: Path | str,
    input_sha256: str,
    manifest_snapshot_hash: str | None = None,
) -> Path:
    """Write curve.npz, qc.json and provenance.json atomically."""
    target = curve_directory(root, curve.profile_id, curve.recording_id)
    staging = target.with_name(target.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        arrays: dict[str, Any] = {name: getattr(curve, name) for name in ARRAY_NAMES}
        if curve.model_amplitude is not None:
            arrays["model_amplitude"] = curve.model_amplitude
        np.savez(staging / "curve.npz", **arrays)
        (staging / "qc.json").write_text(
            json.dumps(_qc_payload(curve), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "provenance.json").write_text(
            json.dumps(
                _provenance_payload(curve, profile, input_sha256, manifest_snapshot_hash),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except BaseException:
        # Leave no debris behind: a half-written staging directory is not a
        # curve, and a later run must not have to guess whether it is.
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if target.exists():
        shutil.rmtree(target)
    staging.replace(target)
    return target


def _qc_payload(curve: ProcessedHvsr) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "recording_id": curve.recording_id,
        "profile_id": curve.profile_id,
        "initial_window_count": curve.window_count,
        "accepted_window_count": curve.accepted_window_count,
        "rejected_window_count": curve.window_count - curve.accepted_window_count,
        "valid_frequency_range_hz": [
            curve.valid_frequency_min_hz,
            curve.valid_frequency_max_hz,
        ],
        "model_ready": curve.model_ready,
        "extrapolated": curve.extrapolated,
        "qc_status": curve.qc_status(),
        "qc_flags": list(curve.qc_flags),
        "window_qc": [w.as_dict() for w in curve.window_qc],
    }
    if curve.coverage is not None:
        payload["frequency_coverage"] = curve.coverage.as_dict()
        payload["window_length_s"] = curve.coverage.window_length_s
        payload["nyquist_hz"] = curve.coverage.nyquist_hz
    if curve.sesame is not None:
        payload["sesame"] = curve.sesame.as_dict()
    return payload


def _provenance_payload(
    curve: ProcessedHvsr,
    profile: PreprocessingProfile,
    input_sha256: str,
    manifest_snapshot_hash: str | None,
) -> dict[str, Any]:
    import numpy

    payload = {
        "recording_id": curve.recording_id,
        "input_sha256": input_sha256,
        "manifest_snapshot_hash": manifest_snapshot_hash,
        "profile_id": profile.profile_id,
        "profile_identity": profile.identity(),
        "config_hash": profile.config_hash,
        "processing_hash": curve.processing_hash,
        "hvsrpy_version": _hvsrpy_version(),
        "numpy_version": numpy.__version__,
    }
    payload.update(environment_provenance())
    return payload


def _hvsrpy_version() -> str:
    import hvsrpy

    return str(hvsrpy.__version__)


def read_curve_arrays(directory: Path | str) -> dict[str, np.ndarray]:
    """Load a stored curve. Pickle stays off: these files hold arrays only."""
    with np.load(Path(directory) / "curve.npz", allow_pickle=False) as data:
        return {name: data[name] for name in data.files}
