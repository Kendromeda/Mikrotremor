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


@pytest.fixture(scope="module")
def first_record():
    if not MANIFEST.is_dir() or not CONFIG.is_file():
        pytest.skip("USGS manifest is not built in this workspace")
    manifest = read_manifest(MANIFEST)
    config = load_source_config(CONFIG)
    if not (Path(".") / config.root).is_dir():
        pytest.skip("USGS ARRA dataset is not present in this workspace")

    recording = sorted(manifest.recordings, key=lambda r: r.recording_id)[0]
    request = resolve_request(manifest, recording, config, ".")
    return get_reader(recording.raw_format).read(request)


def test_training_profile_produces_a_model_ready_curve(first_record) -> None:
    profile = load_profile("configs/preprocessing/training_v1.yaml")
    outcome = safe_process_recording(first_record, profile)

    assert outcome.status in {"passed", "flagged", "failed"}
    if outcome.curve is None:
        pytest.skip(f"this recording failed with {outcome.failure_reason}")

    curve = outcome.curve
    assert curve.extrapolated is False
    assert curve.model_amplitude is not None
    assert curve.model_amplitude.shape == (35,)
    assert np.isfinite(curve.model_amplitude).all()
    assert (curve.model_amplitude > 0).all()


def test_window_length_differs_between_profiles(first_record) -> None:
    """The compatibility profile scales windows to duration; training does not."""
    training = safe_process_recording(
        first_record, load_profile("configs/preprocessing/training_v1.yaml")
    )
    compat = safe_process_recording(
        first_record, load_profile("configs/preprocessing/published_compat_v1.yaml")
    )
    assert compat.curve is not None
    assert compat.curve.window_count == 35
    if training.curve is not None:
        assert training.curve.coverage.window_length_s == 60.0
        assert compat.curve.coverage.window_length_s != 60.0


def test_processing_is_reproducible(first_record) -> None:
    profile = load_profile("configs/preprocessing/training_v1.yaml")
    first = safe_process_recording(first_record, profile)
    second = safe_process_recording(first_record, profile)

    assert first.status == second.status
    if first.curve is None:
        return
    assert first.curve.processing_hash == second.curve.processing_hash
    assert np.array_equal(first.curve.mean_curve, second.curve.mean_curve)
    assert np.array_equal(first.curve.accepted_window_mask, second.curve.accepted_window_mask)


def test_preprocessing_does_not_touch_the_raw_tree(first_record) -> None:
    config = load_source_config(CONFIG)
    root = Path(".") / config.root
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}

    safe_process_recording(first_record, load_profile("configs/preprocessing/training_v1.yaml"))

    assert {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()} == before
    assert not list(root.rglob("*_updated.*"))
