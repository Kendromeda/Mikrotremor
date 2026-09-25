"""Small, auditable SPAC estimators for the Solo pilot.

Coherency is pooled across accepted time windows and a narrow frequency band.
The Bessel fit is an automatic candidate only; it does not select a physical mode.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks  # type: ignore[import-untyped]
from scipy.special import j0  # type: ignore[import-untyped]


def real_coherency(
    spectra: NDArray[np.complex128],
    pairs: Sequence[tuple[int, int]],
    bands: Sequence[tuple[int, int]],
) -> NDArray[np.float64]:
    """Return real, power-normalized cross spectra [band, pair]."""
    if spectra.ndim != 3 or spectra.shape[0] < 1:
        raise ValueError("spectra must have shape [window, station, frequency]")
    n_stations, n_frequency = spectra.shape[1:]
    result = np.full((len(bands), len(pairs)), np.nan, dtype=np.float64)
    for band_index, (low, high) in enumerate(bands):
        if not (0 <= low < high <= n_frequency):
            raise ValueError("frequency band is outside the spectra")
        for pair_index, (left, right) in enumerate(pairs):
            if not (0 <= left < n_stations and 0 <= right < n_stations and left != right):
                raise ValueError("station pair is invalid")
            x = spectra[:, left, low:high]
            y = spectra[:, right, low:high]
            cross = np.mean(np.conj(x) * y)
            power = np.sqrt(np.mean(np.abs(x) ** 2) * np.mean(np.abs(y) ** 2))
            if power > 0:
                result[band_index, pair_index] = float(np.real(cross / power))
    return result


def fit_bessel_grid(
    frequencies_hz: NDArray[np.float64],
    distances_m: NDArray[np.float64],
    observed: NDArray[np.float64],
    velocities_m_s: NDArray[np.float64],
) -> dict[str, NDArray[np.float64] | NDArray[np.bool_]]:
    """Fit J0(2πfr/c) independently at each frequency on a stated velocity grid."""
    frequency = np.asarray(frequencies_hz, dtype=np.float64)
    distances = np.asarray(distances_m, dtype=np.float64)
    coefficient = np.asarray(observed, dtype=np.float64)
    velocities = np.asarray(velocities_m_s, dtype=np.float64)
    if (
        frequency.ndim != 1
        or distances.ndim != 1
        or velocities.ndim != 1
        or coefficient.shape != (len(frequency), len(distances))
    ):
        raise ValueError("frequency, distances, observed and velocities have incompatible shapes")
    if not np.all(np.isfinite(distances)) or np.any(distances <= 0):
        raise ValueError("distances must be positive and finite")
    if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0):
        raise ValueError("frequencies must be positive and finite")
    if len(velocities) < 2 or not np.all(np.isfinite(velocities)) or np.any(velocities <= 0):
        raise ValueError("velocities must contain at least two positive finite values")
    if np.any(np.diff(velocities) <= 0):
        raise ValueError("velocities must be strictly increasing")

    speed = np.full(len(frequency), np.nan)
    error = np.full(len(frequency), np.nan)
    second_error = np.full(len(frequency), np.nan)
    null_error = np.full(len(frequency), np.nan)
    at_bound = np.zeros(len(frequency), dtype=bool)
    for index, hz in enumerate(frequency):
        valid = np.isfinite(coefficient[index])
        if valid.sum() < 3:
            continue
        curve = coefficient[index, valid]
        predicted = j0(2 * np.pi * hz * distances[valid, None] / velocities[None, :])
        misfit = np.sqrt(np.mean((predicted - curve[:, None]) ** 2, axis=0))
        best = int(np.argmin(misfit))
        minima = list(find_peaks(-misfit)[0])
        minima.extend([0, len(velocities) - 1])
        alternatives = [candidate for candidate in set(minima) if candidate != best]
        speed[index] = velocities[best]
        error[index] = misfit[best]
        second_error[index] = min((misfit[candidate] for candidate in alternatives), default=np.nan)
        null_error[index] = float(np.sqrt(np.mean((1.0 - curve) ** 2)))
        at_bound[index] = best in (0, len(velocities) - 1)
    return {
        "velocity_m_s": speed,
        "fit_rmse": error,
        "second_minimum_rmse": second_error,
        "null_rmse": null_error,
        "velocity_at_grid_bound": at_bound,
    }
