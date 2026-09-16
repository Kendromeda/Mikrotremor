"""Golden tests: the numbers must not move without someone saying why.

These compare against fixtures committed under tests/fixtures. Regenerating
them is a deliberate act (tests/fixtures/make_goldens.py) so that a change in
the curves shows up as a reviewed diff rather than a silently passing suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from mhvsr_vs30.preprocessing.config import load_profile
from mhvsr_vs30.preprocessing.pipeline import process_recording

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
sys.path.insert(0, str(FIXTURES))

RTOL = 1e-9
ATOL = 1e-12


def load_golden(name: str) -> dict[str, np.ndarray]:
    path = FIXTURES / f"golden_{name}.npz"
    if not path.is_file():
        pytest.skip(f"golden fixture {path.name} is absent")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def compare(curve, golden: dict[str, np.ndarray]) -> None:
    assert curve.window_count == int(golden["window_count"][0])
    assert curve.accepted_window_count == int(golden["accepted_window_count"][0])
    np.testing.assert_allclose(curve.frequency_hz, golden["frequency_hz"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(curve.mean_curve, golden["mean_curve"], rtol=RTOL, atol=ATOL)

    expected_model = golden["model_amplitude"]
    if expected_model.size:
        assert curve.model_amplitude is not None
        np.testing.assert_allclose(curve.model_amplitude, expected_model, rtol=RTOL, atol=ATOL)
    else:
        assert curve.model_amplitude is None


def test_training_profile_matches_its_golden() -> None:
    from make_goldens import synthetic_record

    curve = process_recording(
        synthetic_record(), load_profile("configs/preprocessing/training_v1.yaml")
    )
    compare(curve, load_golden("training_v1"))


@pytest.mark.requires_real_data
def test_published_compat_profile_matches_its_golden() -> None:
    """Built from the recording the reference notebooks ship as their example."""
    from make_goldens import NOTEBOOK_RECORDING, notebook_record

    if not NOTEBOOK_RECORDING.is_file():
        pytest.skip("notebook example recording is absent")

    curve = process_recording(
        notebook_record(), load_profile("configs/preprocessing/published_compat_v1.yaml")
    )
    compare(curve, load_golden("published_compat_v1"))


@pytest.mark.requires_real_data
def test_notebook_recording_sits_exactly_at_nyquist() -> None:
    """A 100 Hz record puts the 50 Hz model ceiling exactly at Nyquist.

    This is the case that decides whether training_v1 should cap the corpus at
    40 Hz instead, so it is pinned here rather than left as a remark.
    """
    from make_goldens import NOTEBOOK_RECORDING, notebook_record

    if not NOTEBOOK_RECORDING.is_file():
        pytest.skip("notebook example recording is absent")

    curve = process_recording(
        notebook_record(), load_profile("configs/preprocessing/published_compat_v1.yaml")
    )
    assert curve.coverage is not None
    assert curve.coverage.nyquist_hz == 50.0
    assert curve.coverage.nyquist_margin == pytest.approx(1.0)
    assert "low_nyquist_margin" in curve.qc_flags
