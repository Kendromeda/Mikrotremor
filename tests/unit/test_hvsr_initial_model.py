from __future__ import annotations

import numpy as np
import pytest

from mhvsr_vs30.hvsr_initial_model import (
    PowerLawVs,
    apply_bound_overrides,
    konno_ohmachi_smooth,
    suggest_initial_model,
)
from mhvsr_vs30.hvsr_inversion import (
    HVSRCurve,
    InversionBounds,
    forward_body_wave_hvsr,
    l2_hvsr_misfit,
    run_pso_hvsr_inversion,
    split_particle,
    validate_inversion_bounds,
)


def _three_layer_curve() -> tuple[np.ndarray, np.ndarray]:
    frequency = np.geomspace(0.8, 12.0, 120)
    amplitude = forward_body_wave_hvsr(
        frequency,
        thickness_m=np.array([6.0, 18.0]),
        vs_m_s=np.array([180.0, 350.0, 900.0]),
        qp=30.0,
        qs=10.0,
    )
    return frequency, amplitude


def test_konno_ohmachi_keeps_a_constant_curve_and_is_finite() -> None:
    frequency = np.geomspace(0.5, 20.0, 50)
    smoothed = konno_ohmachi_smooth(frequency, np.full(50, 2.5), bandwidth=40.0)

    np.testing.assert_allclose(smoothed, 2.5)


def test_konno_ohmachi_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="positive"):
        konno_ohmachi_smooth(np.array([0.0, 1.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="bandwidth"):
        konno_ohmachi_smooth(np.array([1.0, 2.0]), np.array([1.0, 1.0]), bandwidth=0.0)


def test_power_law_depth_methods_agree_without_velocity_gradient() -> None:
    frequency = np.array([0.5, 1.0, 4.0])
    constant = PowerLawVs(v0_m_s=200.0, exponent=0.0)

    np.testing.assert_allclose(
        constant.depth_from_frequency(frequency, "power_law_travel_time"),
        200.0 / (4.0 * frequency),
    )
    gradient = PowerLawVs(v0_m_s=175.0, exponent=0.099)
    quarter = gradient.depth_from_frequency(frequency, "quarter_wavelength_v0")
    travel = gradient.depth_from_frequency(frequency, "power_law_travel_time")
    assert np.all(travel > quarter)
    np.testing.assert_allclose(gradient.vs_at(np.array([0.0])), 175.0)


def test_power_law_travel_time_depth_matches_quarter_period_travel_time() -> None:
    profile = PowerLawVs(v0_m_s=175.0, exponent=0.099)
    depth = float(profile.depth_from_frequency(np.array([2.0]), "power_law_travel_time")[0])
    z = np.linspace(0.0, depth, 20001)
    travel_time = np.trapezoid(1.0 / profile.vs_at(z), z)

    assert travel_time == pytest.approx(1.0 / (4.0 * 2.0), rel=1e-6)


def test_power_law_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="V0"):
        PowerLawVs(v0_m_s=0.0)
    with pytest.raises(ValueError, match="exponent"):
        PowerLawVs(exponent=1.0)
    with pytest.raises(ValueError, match="unknown depth method"):
        PowerLawVs().depth_from_frequency(np.array([1.0]), "bogus")  # type: ignore[arg-type]


def test_suggestion_adapts_to_the_curve_and_yields_valid_pso_bounds() -> None:
    frequency, amplitude = _three_layer_curve()
    suggestion = suggest_initial_model(frequency, amplitude, max_boundaries=4)

    assert 1 <= suggestion.finite_layer_count <= 4
    assert not suggestion.used_f0_fallback
    assert np.all(np.diff(suggestion.boundary_depths_m) >= 1.0)
    np.testing.assert_allclose(np.cumsum(suggestion.thickness_m), suggestion.boundary_depths_m)
    assert np.all(np.diff(suggestion.vs_m_s) >= 0)
    bounds = InversionBounds(suggestion.thickness_bounds_m, suggestion.vs_bounds_m_s)
    validate_inversion_bounds(bounds)
    assert np.all(suggestion.particle >= bounds.lower)
    assert np.all(suggestion.particle <= bounds.upper)
    assert suggestion.vs_bounds_m_s[-1, 1] == pytest.approx(3000.0)


def test_suggestion_changes_with_the_depth_conversion() -> None:
    frequency, amplitude = _three_layer_curve()
    quarter = suggest_initial_model(frequency, amplitude)
    travel = suggest_initial_model(frequency, amplitude, depth_method="power_law_travel_time")

    assert travel.boundary_depths_m[-1] > quarter.boundary_depths_m[-1]


def test_suggestion_can_reproduce_the_symmetric_dinver_tolerance() -> None:
    frequency, amplitude = _three_layer_curve()
    suggestion = suggest_initial_model(
        frequency, amplitude, vs_tolerance=0.10, halfspace_vs_max_m_s=None
    )

    np.testing.assert_allclose(suggestion.vs_bounds_m_s[:, 0], 0.9 * suggestion.vs_m_s)
    np.testing.assert_allclose(suggestion.vs_bounds_m_s[:, 1], 1.1 * suggestion.vs_m_s)


def test_flat_curve_falls_back_to_the_f0_depth() -> None:
    frequency = np.geomspace(0.5, 10.0, 40)
    suggestion = suggest_initial_model(frequency, np.full(40, 1.5))

    assert suggestion.used_f0_fallback
    assert suggestion.finite_layer_count == 1


def test_suggestion_rejects_invalid_settings() -> None:
    frequency, amplitude = _three_layer_curve()
    with pytest.raises(ValueError, match="vs_tolerance"):
        suggest_initial_model(frequency, amplitude, vs_tolerance=1.0)
    with pytest.raises(ValueError, match="max_boundaries"):
        suggest_initial_model(frequency, amplitude, max_boundaries=0)
    with pytest.raises(ValueError, match="halfspace_vs_max_m_s"):
        suggest_initial_model(frequency, amplitude, halfspace_vs_max_m_s=5000.0)
    with pytest.raises(ValueError, match="positive"):
        suggest_initial_model(frequency, -amplitude)
    with pytest.raises(ValueError, match="min_layer_thickness_m"):
        suggest_initial_model(frequency, amplitude, min_layer_thickness_m=1e6)


def test_overrides_replace_selected_layers_only() -> None:
    thickness = np.array([[1.0, 5.0], [5.0, 20.0]])
    vs = np.array([[100.0, 300.0], [200.0, 500.0], [400.0, 900.0]])
    new_thickness, new_vs = apply_bound_overrides(
        thickness, vs, {"thickness_m": {2: (8, 12)}, "vs_m_s": {3: (600.0, 2000.0)}}
    )

    np.testing.assert_allclose(new_thickness, [[1.0, 5.0], [8.0, 12.0]])
    np.testing.assert_allclose(new_vs[:2], vs[:2])
    np.testing.assert_allclose(new_vs[2], [600.0, 2000.0])
    np.testing.assert_allclose(thickness[1], [5.0, 20.0])
    unchanged, _ = apply_bound_overrides(thickness, vs, None)
    np.testing.assert_allclose(unchanged, thickness)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"density": {1: (1, 2)}}, "unknown override"),
        ({"thickness_m": {3: (1, 2)}}, "outside"),
        ({"vs_m_s": {1: (300, 200)}}, "min < max"),
    ],
)
def test_overrides_reject_invalid_entries(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        apply_bound_overrides(
            np.array([[1.0, 5.0], [5.0, 20.0]]),
            np.array([[100.0, 300.0], [200.0, 500.0], [400.0, 900.0]]),
            overrides,
        )


def test_pso_seeds_particle_zero_with_the_initial_model() -> None:
    frequency, amplitude = _three_layer_curve()
    observed = HVSRCurve(frequency_hz=frequency, amplitude=amplitude)
    suggestion = suggest_initial_model(frequency, amplitude)
    bounds = InversionBounds(suggestion.thickness_bounds_m, suggestion.vs_bounds_m_s)
    result = run_pso_hvsr_inversion(
        observed,
        bounds,
        n_particles=4,
        n_iterations=1,
        seed=7,
        qp=30.0,
        qs=10.0,
        initial_particle=suggestion.particle,
    )

    np.testing.assert_allclose(result.evaluated_models[0], suggestion.particle)
    thickness, vs = split_particle(suggestion.particle, bounds.finite_layer_count)
    initial_misfit = l2_hvsr_misfit(
        amplitude, forward_body_wave_hvsr(frequency, thickness, vs, qp=30.0, qs=10.0)
    )
    assert result.initial_best_misfit <= initial_misfit + 1e-12


def test_pso_clips_the_initial_particle_and_checks_its_size() -> None:
    frequency, amplitude = _three_layer_curve()
    observed = HVSRCurve(frequency_hz=frequency, amplitude=amplitude)
    bounds = InversionBounds(np.array([[5.0, 30.0]]), np.array([[150.0, 400.0], [500.0, 1500.0]]))
    result = run_pso_hvsr_inversion(
        observed, bounds, n_particles=3, n_iterations=1, initial_particle=[100.0, 50.0, 800.0]
    )

    np.testing.assert_allclose(result.evaluated_models[0], [30.0, 150.0, 800.0])
    with pytest.raises(ValueError, match="initial_particle"):
        run_pso_hvsr_inversion(
            observed, bounds, n_particles=3, n_iterations=1, initial_particle=[10.0, 200.0]
        )
