from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mhvsr_vs30.hvsr_inversion import (
    HVSRCurve,
    InversionBounds,
    acceptable_model_ensemble,
    brocher_density_from_vp,
    brocher_vp_from_vs,
    forward_body_wave_hvsr,
    l2_hvsr_misfit,
    parse_geopsy_hv,
    run_pso_hvsr_inversion,
    split_particle,
    validate_inversion_bounds,
    vs30_from_layers,
)


def _observed_curve() -> HVSRCurve:
    frequency_hz = np.linspace(0.5, 8.0, 80)
    amplitude = forward_body_wave_hvsr(
        frequency_hz,
        thickness_m=np.array([18.0]),
        vs_m_s=np.array([240.0, 800.0]),
    )
    return HVSRCurve(frequency_hz=frequency_hz, amplitude=amplitude)


def _bounds() -> InversionBounds:
    return InversionBounds(
        thickness_m=np.array([[10.0, 30.0]]),
        vs_m_s=np.array([[160.0, 360.0], [600.0, 1000.0]]),
    )


def test_parse_geopsy_hv_uses_frequency_amplitude_table_and_ignores_peak_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "station.hv"
    path.write_text(
        "# GEOPSY HVSR export\n"
        "f0 amplitude\n"
        "0.50 1.20\n"
        "1.00 2.40\n"
        "Peak amplitude\n"
        "2.40\n",
        encoding="utf-8",
    )

    curve = parse_geopsy_hv(path)

    np.testing.assert_allclose(curve.frequency_hz, [0.5, 1.0])
    np.testing.assert_allclose(curve.amplitude, [1.2, 2.4])


def test_parse_geopsy_four_column_table_preserves_envelope_log_std_and_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "station-with-envelope.hv"
    path.write_text(
        "# Number of windows = 12\n"
        "# Number of windows for f0 = 8\n"
        "# f0 from average = 2.75\n"
        "# f0 from windows 2.70 2.50 3.00\n"
        "# Peak amplitude = 4.20\n"
        "f0 amplitude\n"
        "1.0 2.0 1.5 2.5\n"
        "2.0 3.0 2.0 4.0\n",
        encoding="utf-8",
    )

    curve = parse_geopsy_hv(path)

    np.testing.assert_allclose(curve.minimum, [1.5, 2.0])
    np.testing.assert_allclose(curve.maximum, [2.5, 4.0])
    np.testing.assert_allclose(curve.log_std, 0.5 * np.log(np.array([2.5 / 1.5, 2.0])))
    assert curve.number_of_windows == 12
    assert curve.number_of_windows_for_f0 == 8
    assert curve.f0_from_average_hz == pytest.approx(2.75)
    assert curve.f0_from_windows_hz == pytest.approx(2.70)
    assert curve.f0_min_hz == pytest.approx(2.50)
    assert curve.f0_max_hz == pytest.approx(3.00)
    assert curve.f0_amplitude == pytest.approx(4.20)


def test_parse_geopsy_rejects_mixed_two_and_four_column_numeric_rows(tmp_path: Path) -> None:
    path = tmp_path / "mixed-columns.hv"
    path.write_text("1.0 2.0\n2.0 3.0 2.5 3.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inconsistent column count"):
        parse_geopsy_hv(path)


@pytest.mark.parametrize("malformed_row", ["1.5 broken", "1.5 2.0 1.0", "1.5,2.0"])
def test_parse_geopsy_rejects_malformed_rows_after_curve_starts(
    tmp_path: Path, malformed_row: str
) -> None:
    path = tmp_path / "malformed.hv"
    path.write_text(
        "# Frequency Average\n0.5 1.2\n1.0 2.4\n" + malformed_row + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="data line"):
        parse_geopsy_hv(path)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("f0 amplitude\n0.0 1.2\n", "frequency"),
        ("f0 amplitude\n1.0 -0.1\n", "amplitude"),
        ("f0 amplitude\n2.0 1.2\n1.0 1.3\n", "increasing"),
        ("Peak amplitude\n2.4\n", "f0 amplitude"),
    ],
)
def test_parse_geopsy_hv_rejects_invalid_frequency_or_hvsr_envelope(
    tmp_path: Path, contents: str, message: str
) -> None:
    path = tmp_path / "invalid.hv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        parse_geopsy_hv(path)


def test_brocher_relations_return_physical_vp_and_density() -> None:
    vs_m_s = np.array([200.0, 500.0, 1000.0])

    vp_m_s = brocher_vp_from_vs(vs_m_s)
    density_kg_m3 = brocher_density_from_vp(vp_m_s)

    assert np.all(np.isfinite(vp_m_s))
    assert np.all(vp_m_s > vs_m_s)
    assert np.all(np.diff(vp_m_s) > 0)
    assert np.all(np.isfinite(density_kg_m3))
    assert np.all((density_kg_m3 > 1500.0) & (density_kg_m3 < 3500.0))


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (
            lambda: HVSRCurve(
                frequency_hz=np.array([1.0, 2.0]),
                amplitude=np.array([2.0, 3.0]),
                minimum=np.array([1.0, 1.0]),
            ),
            "together",
        ),
        (
            lambda: HVSRCurve(
                frequency_hz=np.array([1.0, 2.0]),
                amplitude=np.array([2.0, 3.0]),
                minimum=np.array([2.1, 1.0]),
                maximum=np.array([3.0, 4.0]),
            ),
            "Min <= Average <= Max",
        ),
    ],
)
def test_hvsr_curve_rejects_incomplete_or_invalid_envelopes(
    constructor: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        constructor()  # type: ignore[operator]


def test_brocher_and_misfit_reject_nonphysical_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="Brocher Vs"):
        brocher_vp_from_vs(np.array([0.0]))
    with pytest.raises(ValueError, match="Brocher Vp"):
        brocher_density_from_vp(np.array([np.nan]))
    with pytest.raises(ValueError, match="finite"):
        l2_hvsr_misfit(np.array([1.0]), np.array([np.nan]))


def test_forward_body_wave_is_flat_for_a_uniform_medium() -> None:
    frequency_hz = np.linspace(0.2, 12.0, 300)

    synthetic = forward_body_wave_hvsr(
        frequency_hz,
        thickness_m=np.array([20.0]),
        vs_m_s=np.array([400.0, 400.0]),
    )

    assert np.all(np.isfinite(synthetic))
    assert np.all(synthetic > 0.0)
    assert np.ptp(synthetic) < 1e-6


def test_forward_body_wave_validates_model_and_quality_input() -> None:
    with pytest.raises(ValueError, match="one fewer"):
        forward_body_wave_hvsr(np.array([1.0]), np.array([10.0]), np.array([200.0]))
    with pytest.raises(ValueError, match="Qp"):
        forward_body_wave_hvsr(
            np.array([1.0]), np.array([10.0]), np.array([200.0, 600.0]), qp=0.0
        )


def test_forward_body_wave_resonance_moves_lower_when_sediment_is_thicker() -> None:
    frequency_hz = np.linspace(0.3, 12.0, 2000)
    thin = forward_body_wave_hvsr(
        frequency_hz,
        thickness_m=np.array([10.0]),
        vs_m_s=np.array([240.0, 900.0]),
    )
    thick = forward_body_wave_hvsr(
        frequency_hz,
        thickness_m=np.array([20.0]),
        vs_m_s=np.array([240.0, 900.0]),
    )

    thin_peak_hz = frequency_hz[np.argmax(thin)]
    thick_peak_hz = frequency_hz[np.argmax(thick)]

    assert thick_peak_hz < thin_peak_hz * 0.7


def test_finite_q_damps_resonance_instead_of_amplifying_it() -> None:
    frequency_hz = np.linspace(0.3, 12.0, 2000)

    elastic = forward_body_wave_hvsr(
        frequency_hz,
        thickness_m=np.array([20.0]),
        vs_m_s=np.array([240.0, 900.0]),
    )
    attenuated = forward_body_wave_hvsr(
        frequency_hz,
        thickness_m=np.array([20.0]),
        vs_m_s=np.array([240.0, 900.0]),
        qp=30.0,
        qs=10.0,
    )

    assert np.max(attenuated) < np.max(elastic)
    fundamental = attenuated[(frequency_hz >= 2.0) & (frequency_hz <= 4.5)]
    first_overtone = attenuated[(frequency_hz >= 8.0) & (frequency_hz <= 11.0)]
    assert np.max(fundamental) > np.max(first_overtone)


def test_elastic_single_layer_matches_closed_form_transfer_ratio() -> None:
    frequency_hz = np.array([0.7, 1.3, 2.5, 4.0])
    thickness_m = 18.0
    vs_m_s = np.array([260.0, 900.0])
    vp_m_s = brocher_vp_from_vs(vs_m_s)
    density = brocher_density_from_vp(vp_m_s)
    omega = 2.0 * np.pi * frequency_hz

    def transfer(velocity: np.ndarray) -> np.ndarray:
        phase = omega * thickness_m / velocity[0]
        impedance_ratio = density[0] * velocity[0] / (density[1] * velocity[1])
        return np.abs(2.0 / (np.cos(phase) + 1j * impedance_ratio * np.sin(phase)))

    expected = transfer(vs_m_s) / transfer(vp_m_s)
    actual = forward_body_wave_hvsr(frequency_hz, [thickness_m], vs_m_s)

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)


def test_l2_hvsr_misfit_is_norm_2_and_rejects_mismatched_curves() -> None:
    observed = np.array([1.0, 2.0, 4.0])
    synthetic = np.array([1.0, 1.0, 2.0])

    assert l2_hvsr_misfit(observed, synthetic) == pytest.approx(np.sqrt(5.0))
    with pytest.raises(ValueError, match="same shape"):
        l2_hvsr_misfit(observed, synthetic[:2])


def test_bounds_validate_layer_counts_finite_intervals_and_positive_velocities() -> None:
    valid = _bounds()
    validate_inversion_bounds(valid)

    invalid = InversionBounds(
        thickness_m=np.array([[30.0, 10.0]]),
        vs_m_s=np.array([[160.0, 360.0], [600.0, 1000.0]]),
    )
    with pytest.raises(ValueError, match="lower"):
        validate_inversion_bounds(invalid)


def test_bounds_and_particle_split_reject_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="Vs bounds"):
        validate_inversion_bounds(
            InversionBounds(thickness_m=np.array([[10.0, 20.0]]), vs_m_s=np.array([[200.0, 400.0]]))
        )
    with pytest.raises(ValueError, match="contain 3"):
        split_particle(np.array([10.0, 200.0]), finite_layer_count=1)


def test_pso_is_deterministic_constrained_and_improves_the_initial_population() -> None:
    observed = _observed_curve()
    bounds = _bounds()

    first = run_pso_hvsr_inversion(
        observed, bounds, n_particles=18, n_iterations=30, seed=2026
    )
    second = run_pso_hvsr_inversion(
        observed, bounds, n_particles=18, n_iterations=30, seed=2026
    )

    np.testing.assert_allclose(first.best_particle, second.best_particle)
    assert first.best_misfit == pytest.approx(second.best_misfit)
    assert first.best_misfit <= first.initial_best_misfit
    assert np.all(first.best_particle >= bounds.lower)
    assert np.all(first.best_particle <= bounds.upper)
    assert len(first.models) == 18


def test_pso_projects_particles_to_nondecreasing_vs() -> None:
    result = run_pso_hvsr_inversion(
        _observed_curve(),
        _bounds(),
        n_particles=8,
        n_iterations=3,
        seed=7,
        require_nondecreasing_vs=True,
    )

    assert np.all(np.diff(result.models[:, 1:], axis=1) >= 0.0)


def test_pso_rejects_bounds_without_a_nondecreasing_vs_solution() -> None:
    impossible = InversionBounds(
        thickness_m=np.array([[10.0, 30.0]]),
        vs_m_s=np.array([[900.0, 1000.0], [100.0, 200.0]]),
    )

    with pytest.raises(ValueError, match="do not admit"):
        run_pso_hvsr_inversion(
            _observed_curve(),
            impossible,
            n_particles=4,
            n_iterations=1,
            require_nondecreasing_vs=True,
        )


def test_pso_rejects_invalid_population_size_and_coefficients() -> None:
    with pytest.raises(ValueError, match="at least two"):
        run_pso_hvsr_inversion(_observed_curve(), _bounds(), n_particles=1, n_iterations=1)
    with pytest.raises(ValueError, match="coefficients"):
        run_pso_hvsr_inversion(
            _observed_curve(), _bounds(), n_particles=2, n_iterations=1, inertia=-1
        )


def test_ensemble_depth_profile_and_vs30_summarize_low_misfit_models() -> None:
    models = np.array(
        [
            [10.0, 200.0, 500.0],
            [12.0, 220.0, 550.0],
            [30.0, 350.0, 900.0],
        ]
    )
    misfits = np.array([0.10, 0.12, 1.50])

    ensemble = acceptable_model_ensemble(models, misfits, top_n=2)
    profile = ensemble.depth_profile(depth_m=np.array([0.0, 10.0, 20.0]))

    assert ensemble.n_models == 2
    assert np.max(ensemble.misfits) == pytest.approx(0.12)
    assert profile.vs_mean_m_s.shape == (3,)
    assert np.all(profile.vs_std_m_s >= 0.0)
    assert vs30_from_layers(np.array([10.0]), np.array([200.0, 400.0])) == pytest.approx(300.0)


def test_ensemble_filters_by_relative_misfit_discards_nonfinite_and_validates_options() -> None:
    models = np.array(
        [[10.0, 200.0, 500.0], [12.0, 220.0, 550.0], [14.0, 240.0, 600.0], [16.0, 260.0, 650.0]]
    )
    misfits = np.array([1.0, 1.09, 1.5, np.inf])

    ensemble = acceptable_model_ensemble(
        models, misfits, top_n=10, maximum_relative_misfit=0.10
    )

    assert ensemble.n_models == 2
    np.testing.assert_allclose(ensemble.misfits, [1.0, 1.09])
    with pytest.raises(ValueError, match="nonnegative"):
        acceptable_model_ensemble(models, misfits, maximum_relative_misfit=-0.1)
    with pytest.raises(ValueError, match="positive"):
        acceptable_model_ensemble(models, misfits, top_n=0)
