from __future__ import annotations

import numpy as np
import pytest
from scipy.special import j0

from mhvsr_vs30.mam.spac import fit_bessel_grid, real_coherency


def test_real_coherency_identical_signals_is_one() -> None:
    spectrum = np.array([[[1 + 1j, 2 + 0j], [1 + 1j, 2 + 0j]]])
    result = real_coherency(spectrum, [(0, 1)], [(0, 1), (1, 2)])
    np.testing.assert_allclose(result, 1.0)


def test_real_coherency_orthogonal_phase_is_zero() -> None:
    spectrum = np.array([[[1 + 0j], [0 + 1j]]])
    result = real_coherency(spectrum, [(0, 1)], [(0, 1)])
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


def test_bessel_fit_recovers_known_speed() -> None:
    distances = np.array([2.1, 2.3, 2.6, 3.6, 4.1, 4.5])
    frequencies = np.array([8.0, 12.0, 16.0])
    observed = j0(2 * np.pi * frequencies[:, None] * distances[None, :] / 220.0)
    fit = fit_bessel_grid(frequencies, distances, observed, np.linspace(80, 600, 1041))
    np.testing.assert_allclose(fit["velocity_m_s"], 220.0, atol=0.5)
    np.testing.assert_allclose(fit["fit_rmse"], 0.0, atol=1e-8)


def test_bessel_fit_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="distances"):
        fit_bessel_grid(
            np.array([10.0]), np.array([0.0]), np.array([[0.5]]), np.array([100.0, 200.0])
        )
