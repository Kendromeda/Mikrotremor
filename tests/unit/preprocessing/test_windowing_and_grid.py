"""Phase 4 unit tests: window policy, frequency validity and the model grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mhvsr_vs30.exceptions import ConfigError
from mhvsr_vs30.preprocessing.config import load_profile
from mhvsr_vs30.preprocessing.contracts import PreprocessingError
from mhvsr_vs30.preprocessing.resampling import model_grid_hz, resample_to_model_grid
from mhvsr_vs30.preprocessing.windowing import (
    expected_window_count,
    minimum_valid_frequency_hz,
    window_length_seconds,
)

TRAINING = Path("configs/preprocessing/training_v1.yaml")
COMPAT = Path("configs/preprocessing/published_compat_v1.yaml")


@pytest.fixture
def training():
    return load_profile(TRAINING)


@pytest.fixture
def compat():
    return load_profile(COMPAT)


def test_fixed_duration_window_count(training) -> None:
    assert window_length_seconds(training, 3600.0) == 60.0
    assert expected_window_count(3600.0, 60.0) == 60
    assert expected_window_count(1451.0, 60.0) == 24  # partial tail is not a window


def test_fixed_count_window_length_tracks_duration(compat) -> None:
    """The notebook divides duration by 35, so window length varies per record."""
    assert window_length_seconds(compat, 3600.0) == pytest.approx(3600.0 / 35)
    assert window_length_seconds(compat, 1451.0) == pytest.approx(1451.0 / 35)


def test_fixed_duration_refuses_a_record_shorter_than_one_window(training) -> None:
    with pytest.raises(PreprocessingError) as excinfo:
        window_length_seconds(training, 30.0)
    assert excinfo.value.reason == "insufficient_initial_windows"


def test_significant_cycles_lower_bound(training) -> None:
    """15 cycles in a 60 s window puts the floor at 0.25 Hz, under the 0.3 Hz grid."""
    assert minimum_valid_frequency_hz(training, 60.0) == pytest.approx(0.25)
    assert minimum_valid_frequency_hz(training, 60.0) < training.model_grid.min_hz


def test_shorter_windows_raise_the_frequency_floor(training) -> None:
    assert minimum_valid_frequency_hz(training, 30.0) == pytest.approx(0.5)
    assert minimum_valid_frequency_hz(training, 120.0) == pytest.approx(0.125)


def test_model_grid_has_exactly_35_points(training, compat) -> None:
    for profile in (training, compat):
        grid = model_grid_hz(profile)
        assert grid.shape == (35,)
        assert grid[0] == pytest.approx(0.3)
        assert grid[-1] == pytest.approx(50.0)
        assert (np.diff(grid) > 0).all()


def test_resampling_never_extrapolates_under_training_profile(training) -> None:
    """A curve that stops at 10 Hz yields no model vector rather than invented values."""
    frequency = np.geomspace(0.3, 10.0, 128)
    amplitude = np.ones_like(frequency) * 2.0

    values, covered, extrapolated = resample_to_model_grid(
        frequency, amplitude, training, valid_min_hz=0.3, valid_max_hz=10.0
    )
    assert values is None
    assert covered is False
    assert extrapolated is False


def test_resampling_succeeds_when_the_grid_is_covered(training) -> None:
    frequency = np.geomspace(0.1, 60.0, 256)
    amplitude = np.linspace(1.0, 3.0, 256)

    values, covered, extrapolated = resample_to_model_grid(
        frequency, amplitude, training, valid_min_hz=0.2, valid_max_hz=55.0
    )
    assert covered is True
    assert extrapolated is False
    assert values is not None and values.shape == (35,)
    assert np.isfinite(values).all() and (values > 0).all()


def test_compatibility_profile_extrapolates_and_says_so(compat) -> None:
    """Legacy extrapolation is reproduced, but never silently."""
    frequency = np.geomspace(1.0, 60.0, 128)
    amplitude = np.linspace(1.0, 2.0, 128)

    values, _, extrapolated = resample_to_model_grid(
        frequency, amplitude, compat, valid_min_hz=1.0, valid_max_hz=55.0
    )
    assert extrapolated is True
    assert values is not None and values.shape == (35,)


def test_profiles_disagree_on_extrapolation_by_design(training, compat) -> None:
    assert training.model_grid.allow_extrapolation is False
    assert compat.model_grid.allow_extrapolation is True
    assert training.identity() != compat.identity()


def test_profile_identity_changes_with_settings(tmp_path: Path, training) -> None:
    text = TRAINING.read_text(encoding="utf-8").replace(
        "window_length_s: 60", "window_length_s: 90"
    )
    altered = tmp_path / "altered.yaml"
    altered.write_text(text, encoding="utf-8")

    assert load_profile(altered).identity() != training.identity()


def test_profile_identity_ignores_renaming(tmp_path: Path, training) -> None:
    """Renaming a profile must not invalidate numerically identical curves."""
    text = TRAINING.read_text(encoding="utf-8").replace(
        "profile_id: training_v1", "profile_id: training_v1_renamed"
    )
    renamed = tmp_path / "renamed.yaml"
    renamed.write_text(text, encoding="utf-8")

    assert load_profile(renamed).identity() == training.identity()


def test_missing_window_length_is_a_config_error(tmp_path: Path) -> None:
    text = TRAINING.read_text(encoding="utf-8").replace("  window_length_s: 60\n", "")
    broken = tmp_path / "broken.yaml"
    broken.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_profile(broken)
