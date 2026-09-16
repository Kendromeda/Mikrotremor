"""Phase 4 unit tests: pipeline behaviour, rejection and artifact writing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mhvsr_vs30.io.base import RepairRecord, ThreeComponentRecord
from mhvsr_vs30.preprocessing.config import load_profile
from mhvsr_vs30.preprocessing.contracts import ProcessedHvsr
from mhvsr_vs30.preprocessing.pipeline import process_recording, safe_process_recording
from mhvsr_vs30.preprocessing.store import read_curve_arrays, write_curve

TRAINING = Path("configs/preprocessing/training_v1.yaml")
COMPAT = Path("configs/preprocessing/published_compat_v1.yaml")


def synthetic_record(
    duration_s: float = 1200.0, rate: float = 200.0, **kwargs
) -> ThreeComponentRecord:
    rng = np.random.default_rng(1234)
    n = int(duration_s * rate)
    defaults = dict(
        z=rng.normal(size=n),
        north=rng.normal(size=n),
        east=rng.normal(size=n),
        sampling_rate_hz=rate,
        start_time=None,
        units="counts",
        source_channels={"Z": "column_1", "N": "column_2", "E": "column_3"},
        recording_id="c" * 64,
    )
    defaults.update(kwargs)
    return ThreeComponentRecord(**defaults)


@pytest.fixture(scope="module")
def training():
    return load_profile(TRAINING)


@pytest.fixture(scope="module")
def curve(training) -> ProcessedHvsr:
    return process_recording(synthetic_record(), training)


def test_arrays_are_read_only(curve: ProcessedHvsr) -> None:
    for name in ("frequency_hz", "mean_curve", "log_std_curve", "model_frequency_hz"):
        with pytest.raises(ValueError):
            getattr(curve, name)[0] = 1.0


def test_window_count_follows_the_fixed_duration_policy(curve: ProcessedHvsr) -> None:
    assert curve.window_count == 20  # 1200 s / 60 s


def test_rejection_mask_is_deterministic(training) -> None:
    first = process_recording(synthetic_record(), training)
    second = process_recording(synthetic_record(), training)

    assert np.array_equal(first.accepted_window_mask, second.accepted_window_mask)
    assert np.array_equal(first.mean_curve, second.mean_curve)
    assert first.processing_hash == second.processing_hash


def test_rejected_windows_do_not_enter_the_mean(training) -> None:
    """The stored mean must be the mean of accepted windows only."""
    result = process_recording(synthetic_record(), training)
    accepted = result.accepted_window_mask.astype(bool)
    if accepted.all():
        pytest.skip("this fixture produced no rejections to compare against")

    kept = np.log(result.window_amplitudes[accepted])
    expected = np.exp(kept.mean(axis=0))
    all_windows = np.exp(np.log(result.window_amplitudes).mean(axis=0))

    assert np.allclose(result.mean_curve, expected, rtol=1e-8)
    assert not np.allclose(result.mean_curve, all_windows, rtol=1e-8)


def test_window_qc_records_one_entry_per_window(curve: ProcessedHvsr) -> None:
    assert len(curve.window_qc) == curve.window_count
    assert [w.index for w in curve.window_qc] == list(range(curve.window_count))
    assert all(w.reason for w in curve.window_qc)


def test_processing_hash_changes_with_config(tmp_path: Path, training) -> None:
    text = TRAINING.read_text(encoding="utf-8").replace("bandwidth: 40", "bandwidth: 30")
    altered = tmp_path / "altered.yaml"
    altered.write_text(text, encoding="utf-8")

    record = synthetic_record()
    assert (
        process_recording(record, training).processing_hash
        != process_recording(record, load_profile(altered)).processing_hash
    )


def test_a_repaired_input_flags_the_curve(training) -> None:
    repaired = synthetic_record(
        repairs=(RepairRecord(action="drop_incomplete_final_row", reason="cut", samples_dropped=1),)
    )
    assert "repaired_source_row" in process_recording(repaired, training).qc_flags


def test_a_short_record_fails_instead_of_producing_a_curve(training) -> None:
    outcome = safe_process_recording(synthetic_record(duration_s=120.0), training)

    assert outcome.status == "failed"
    assert outcome.curve is None
    assert outcome.failure_reason == "insufficient_initial_windows"


def test_training_profile_never_extrapolates(curve: ProcessedHvsr) -> None:
    assert curve.extrapolated is False
    assert "legacy_extrapolation_applied" not in curve.qc_flags


def test_model_vector_is_35_positive_finite_values(curve: ProcessedHvsr) -> None:
    assert curve.model_ready is True
    assert curve.model_amplitude is not None
    assert curve.model_amplitude.shape == (35,)
    assert np.isfinite(curve.model_amplitude).all()
    assert (curve.model_amplitude > 0).all()


def test_store_round_trips_without_pickle(curve: ProcessedHvsr, training, tmp_path: Path) -> None:
    written = write_curve(curve, training, tmp_path, input_sha256="d" * 64)
    arrays = read_curve_arrays(written)

    assert np.array_equal(arrays["mean_curve"], curve.mean_curve)
    assert np.array_equal(arrays["model_amplitude"], curve.model_amplitude)
    assert (written / "qc.json").is_file()
    assert (written / "provenance.json").is_file()


def test_atomic_write_leaves_no_partial_output(
    curve: ProcessedHvsr, training, tmp_path: Path, monkeypatch
) -> None:
    """A crash mid-write must not leave a directory that looks like a finished curve."""
    import mhvsr_vs30.preprocessing.store as store

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(store.json, "dumps", explode)
    with pytest.raises(OSError):
        write_curve(curve, training, tmp_path, input_sha256="d" * 64)

    final = store.curve_directory(tmp_path, curve.profile_id, curve.recording_id)
    assert not final.exists()
    assert not list(tmp_path.rglob("curve.npz")) or final.exists()
