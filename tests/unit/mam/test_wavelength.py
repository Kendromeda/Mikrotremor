from __future__ import annotations

import numpy as np
import pytest

from mhvsr_vs30.mam.wavelength import (
    depth_guidelines,
    initial_model_from_wavelength,
    wavelength_diagnostics,
    wavelength_guides,
)


def test_wavelength_guides_preserve_input_order_and_do_not_mutate_inputs() -> None:
    frequencies = np.array([4.0, 2.0])
    velocities = np.array([200.0, 300.0])

    guides = wavelength_guides(frequencies, velocities)

    np.testing.assert_allclose(guides["wavelength_m"], [50.0, 150.0])
    np.testing.assert_allclose(guides["guide_depth_m"], [50.0 / 3.0, 50.0])
    np.testing.assert_allclose(guides["frequency_hz"], frequencies)
    np.testing.assert_allclose(guides["phase_velocity_m_s"], velocities)
    np.testing.assert_allclose(frequencies, [4.0, 2.0])
    np.testing.assert_allclose(velocities, [200.0, 300.0])


@pytest.mark.parametrize(
    ("frequencies", "velocities"),
    [([], []), ([1.0, 2.0], [300.0]), ([[1.0]], [300.0]), ([0.0], [300.0]), ([1.0], [np.inf])],
)
def test_wavelength_guides_reject_invalid_curves(
    frequencies: object, velocities: object
) -> None:
    with pytest.raises(ValueError):
        wavelength_guides(frequencies, velocities)  # type: ignore[arg-type]


def test_initial_model_uses_thickness_midpoints_and_depth_interpolated_guides() -> None:
    model = initial_model_from_wavelength(
        np.array([4.0, 0.5, 1.0, 2.0]),
        np.array([400.0, 240.0, 300.0, 300.0]),
        np.array([[79.9, 80.1], [119.9, 120.1]]),
        np.array([[100.0, 500.0], [100.0, 500.0], [100.0, 500.0]]),
    )

    np.testing.assert_allclose(model["parameters"], [80.0, 120.0, 360.0, 260.0, 240.0])
    np.testing.assert_allclose(model["sample_depth_m"], [40.0, 140.0, 200.0])
    np.testing.assert_allclose(model["phase_velocity_guide_m_s"], [360.0, 260.0, 240.0])
    np.testing.assert_allclose(model["vs_unclipped_m_s"], [360.0, 260.0, 240.0])
    assert not np.any(model["clipped_to_bounds"])
    np.testing.assert_array_equal(model["outside_guide_depth"], [False, False, True])


def test_initial_model_averages_duplicate_depths_and_flags_endpoint_clamping() -> None:
    model = initial_model_from_wavelength(
        np.array([2.0, 4.0, 1.0]),
        np.array([300.0, 600.0, 300.0]),
        np.array([[1.9, 2.1]]),
        np.array([[150.0, 250.0], [100.0, 250.0]]),
        velocity_factor=0.5,
    )

    # The first two samples share guide depth 50 m, so their phase speeds average to 450 m/s.
    np.testing.assert_allclose(model["phase_velocity_guide_m_s"], [450.0, 450.0])
    np.testing.assert_allclose(model["vs_unclipped_m_s"], [225.0, 225.0])
    np.testing.assert_array_equal(model["clipped_to_bounds"], [False, False])
    np.testing.assert_array_equal(model["outside_guide_depth"], [True, True])


def test_initial_model_clips_guides_to_each_vs_bound() -> None:
    model = initial_model_from_wavelength(
        np.array([2.0, 1.0]),
        np.array([600.0, 600.0]),
        np.array([[3.9, 4.1]]),
        np.array([[100.0, 250.0], [300.0, 400.0]]),
    )

    np.testing.assert_allclose(model["vs_unclipped_m_s"], [600.0, 600.0])
    np.testing.assert_allclose(model["parameters"], [4.0, 250.0, 400.0])
    np.testing.assert_array_equal(model["clipped_to_bounds"], [True, True])


@pytest.mark.parametrize(
    "thickness_bounds,vs_bounds,velocity_factor",
    [
        ([], [[100.0, 200.0]], 1.0),
        ([[1.0, 2.0]], [[100.0, 200.0]], 1.0),
        ([[2.0, 1.0]], [[100.0, 200.0], [100.0, 200.0]], 1.0),
        ([[1.0, 2.0]], [[100.0, 200.0], [100.0, 200.0]], 0.0),
    ],
)
def test_initial_model_rejects_invalid_bounds(
    thickness_bounds: object, vs_bounds: object, velocity_factor: float
) -> None:
    with pytest.raises(ValueError):
        initial_model_from_wavelength(
            np.array([1.0]),
            np.array([300.0]),
            thickness_bounds,  # type: ignore[arg-type]
            vs_bounds,  # type: ignore[arg-type]
            velocity_factor=velocity_factor,
        )


def test_diagnostics_use_configurable_cutoff_with_inclusive_boundary() -> None:
    diagnostics = wavelength_diagnostics(
        np.array([2.0, 1.0]), np.array([100.0, 300.0]), maximum_spacing_m=50.0
    )
    np.testing.assert_allclose(diagnostics["wavelength_to_aperture"], [1.0, 6.0])
    np.testing.assert_array_equal(diagnostics["within_wavelength_limit"], [True, True])

    limited = wavelength_diagnostics(
        np.array([2.0, 1.0]),
        np.array([100.0, 300.0]),
        maximum_spacing_m=50.0,
        maximum_wavelength_ratio=6.0,
    )
    np.testing.assert_array_equal(limited["within_wavelength_limit"], [True, True])


def test_diagnostics_reject_invalid_policy_values() -> None:
    with pytest.raises(ValueError):
        wavelength_diagnostics(np.array([1.0]), np.array([100.0]), maximum_spacing_m=0.0)
    with pytest.raises(ValueError):
        wavelength_diagnostics(
            np.array([1.0]),
            np.array([100.0]),
            maximum_spacing_m=1.0,
            maximum_wavelength_ratio=np.nan,
        )


def test_depth_guidelines_use_curve_extrema() -> None:
    guidelines = depth_guidelines(np.array([4.0, 1.0]), np.array([200.0, 300.0]))
    assert guidelines == {"minimum_depth_m": 50.0 / 3.0, "maximum_depth_m": 150.0}
