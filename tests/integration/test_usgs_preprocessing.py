"""Phase 4 integration: real USGS recordings through both profiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mhvsr_vs30.io.registry import get_reader
from mhvsr_vs30.io.resolve import resolve_request
from mhvsr_vs30.manifest.schemas import load_source_config
from mhvsr_vs30.manifest.store import read_manifest
from mhvsr_vs30.preprocessing.config import load_profile
from mhvsr_vs30.preprocessing.pipeline import safe_process_recording

pytestmark = pytest.mark.requires_real_data

MANIFEST = Path("artifacts/manifests/usgs_arra_v1")
CONFIG = Path("configs/datasets/usgs_arra.yaml")
QC_REJECTED_RECORDING_ID = "0e1f5de2b2c72307dd785d0892361dd81a259351ce07e46904b0808dcd2a1ff1"
MODEL_READY_RECORDING_ID = "bfd3c0056919f3556d5726af264d4b2829d998543773470ec4d9cb0e089e5fcf"


@pytest.fixture(scope="module")
def usgs_records():
    if not MANIFEST.is_dir() or not CONFIG.is_file():
        pytest.skip("USGS manifest is not built in this workspace")
    manifest = read_manifest(MANIFEST)
    config = load_source_config(CONFIG)
    if not (Path(".") / config.root).is_dir():
        pytest.skip("USGS ARRA dataset is not present in this workspace")

    selected = {
        recording.recording_id: recording
        for recording in manifest.recordings
        if recording.recording_id in {QC_REJECTED_RECORDING_ID, MODEL_READY_RECORDING_ID}
    }
    assert selected.keys() == {QC_REJECTED_RECORDING_ID, MODEL_READY_RECORDING_ID}
    return {
        recording_id: get_reader(recording.raw_format).read(
            resolve_request(manifest, recording, config, ".")
        )
        for recording_id, recording in selected.items()
    }


def test_training_profile_rejects_recording_below_acceptance_fraction(usgs_records) -> None:
    outcome = safe_process_recording(
        usgs_records[QC_REJECTED_RECORDING_ID],
        load_profile("configs/preprocessing/training_v1.yaml"),
    )

    assert outcome.status == "failed"
    assert outcome.failure_reason == "insufficient_accepted_windows"
    assert outcome.message == "only 23/60 windows survived, below the minimum fraction 0.5"
    assert outcome.curve is None


def test_training_profile_produces_a_model_ready_curve(usgs_records) -> None:
    profile = load_profile("configs/preprocessing/training_v1.yaml")
    outcome = safe_process_recording(usgs_records[MODEL_READY_RECORDING_ID], profile)

    assert outcome.status in {"passed", "flagged"}
    assert outcome.curve is not None

    curve = outcome.curve
    assert curve.extrapolated is False
    assert curve.model_amplitude is not None
    assert curve.model_amplitude.shape == (35,)
    assert np.isfinite(curve.model_amplitude).all()
    assert (curve.model_amplitude > 0).all()


def test_window_length_differs_between_profiles(usgs_records) -> None:
    """The compatibility profile scales windows to duration; training does not."""
    training = safe_process_recording(
        usgs_records[MODEL_READY_RECORDING_ID],
        load_profile("configs/preprocessing/training_v1.yaml"),
    )
    compat = safe_process_recording(
        usgs_records[MODEL_READY_RECORDING_ID],
        load_profile("configs/preprocessing/published_compat_v1.yaml"),
    )
    assert compat.curve is not None
    assert compat.curve.window_count == 35
    if training.curve is not None:
        assert training.curve.coverage.window_length_s == 60.0
        assert compat.curve.coverage.window_length_s != 60.0


def test_processing_is_reproducible(usgs_records) -> None:
    profile = load_profile("configs/preprocessing/training_v1.yaml")
    first = safe_process_recording(usgs_records[MODEL_READY_RECORDING_ID], profile)
    second = safe_process_recording(usgs_records[MODEL_READY_RECORDING_ID], profile)

    assert first.status == second.status
    assert first.curve is not None
    assert second.curve is not None
    assert first.curve.processing_hash == second.curve.processing_hash
    assert np.array_equal(first.curve.mean_curve, second.curve.mean_curve)
    assert np.array_equal(first.curve.accepted_window_mask, second.curve.accepted_window_mask)


def test_preprocessing_does_not_touch_the_raw_tree(usgs_records) -> None:
    config = load_source_config(CONFIG)
    root = Path(".") / config.root
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}

    safe_process_recording(
        usgs_records[MODEL_READY_RECORDING_ID],
        load_profile("configs/preprocessing/training_v1.yaml"),
    )

    assert {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()} == before
    assert not list(root.rglob("*_updated.*"))
