"""Wavelength-guided, bounds-safe initial models for MAM dispersion inversion.

The velocity guide deliberately keeps measured Rayleigh phase velocity separate
from shear-wave velocity. ``velocity_factor`` is an explicit local project
approximation, rather than a claimed Hayashi phase-to-Vs conversion.
Depth guidance: Hayashi et al. (2022), Sections 6.2 and 7.2,
https://doi.org/10.1007/s10950-021-10051-y.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]
type GuideResult = dict[str, FloatArray]
type ModelResult = dict[str, FloatArray | BoolArray]
type DiagnosticResult = dict[str, FloatArray | BoolArray]


def _curve_arrays(
    frequencies_hz: ArrayLike, phase_velocity_m_s: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Return copied, valid one-dimensional positive dispersion observations."""
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    velocities = np.asarray(phase_velocity_m_s, dtype=np.float64)
    if (
        frequencies.ndim != 1
        or velocities.ndim != 1
        or len(frequencies) == 0
        or len(frequencies) != len(velocities)
        or not np.all(np.isfinite(frequencies))
        or not np.all(np.isfinite(velocities))
        or np.any(frequencies <= 0)
        or np.any(velocities <= 0)
    ):
        raise ValueError(
            "frequencies and phase velocities must be matching nonempty positive finite 1D arrays"
        )
    return frequencies.copy(), velocities.copy()


def wavelength_guides(frequencies_hz: ArrayLike, phase_velocity_m_s: ArrayLike) -> GuideResult:
    """Calculate wavelength and the ``lambda / 3`` guide depth for each observation."""
    frequencies, velocities = _curve_arrays(frequencies_hz, phase_velocity_m_s)
    wavelength = velocities / frequencies
    return {
        "frequency_hz": frequencies,
        "phase_velocity_m_s": velocities,
        "wavelength_m": wavelength,
        "guide_depth_m": wavelength / 3.0,
    }


def _bounds_arrays(
    thickness_bounds_m: ArrayLike, vs_bounds_m_s: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    thickness = np.asarray(thickness_bounds_m, dtype=np.float64)
    velocity = np.asarray(vs_bounds_m_s, dtype=np.float64)
    if thickness.ndim != 2 or thickness.shape[0] < 1 or thickness.shape[1] != 2:
        raise ValueError(
            "thickness bounds must have shape (n_finite_layers, 2), with n_finite_layers >= 1"
        )
    if velocity.ndim != 2 or velocity.shape != (thickness.shape[0] + 1, 2):
        raise ValueError(
            "Vs bounds must have shape (n_finite_layers + 1, 2), including the halfspace"
        )
    if (
        not np.all(np.isfinite(thickness))
        or not np.all(np.isfinite(velocity))
        or np.any(thickness <= 0)
        or np.any(velocity <= 0)
        or np.any(thickness[:, 0] >= thickness[:, 1])
        or np.any(velocity[:, 0] >= velocity[:, 1])
    ):
        raise ValueError(
            "all bounds must be finite, positive, and have lower values below upper values"
        )
    return thickness.copy(), velocity.copy()


def _unique_depth_velocity(
    depth_m: FloatArray, velocity_m_s: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Sort guide points and average phase velocity at duplicate depths."""
    order = np.argsort(depth_m, kind="stable")
    sorted_depth = depth_m[order]
    sorted_velocity = velocity_m_s[order]
    unique_depth, inverse = np.unique(sorted_depth, return_inverse=True)
    totals = np.bincount(inverse, weights=sorted_velocity)
    counts = np.bincount(inverse)
    return unique_depth, totals / counts


def initial_model_from_wavelength(
    frequencies_hz: ArrayLike,
    phase_velocity_m_s: ArrayLike,
    thickness_bounds_m: ArrayLike,
    vs_bounds_m_s: ArrayLike,
    *,
    velocity_factor: float = 1.0,
) -> ModelResult:
    """Build a model from thickness midpoints and wavelength-depth guides.

    Finite-layer thicknesses use the midpoint of their specified search bounds.
    Phase velocity is interpolated at each finite-layer midpoint and at the top
    of the halfspace. Values beyond the guide's depth range are endpoint
    clamped and explicitly marked as unsupported.
    """
    if not np.isfinite(velocity_factor) or velocity_factor <= 0:
        raise ValueError("velocity_factor must be positive and finite")
    guides = wavelength_guides(frequencies_hz, phase_velocity_m_s)
    thickness_bounds, velocity_bounds = _bounds_arrays(thickness_bounds_m, vs_bounds_m_s)
    thickness = np.mean(thickness_bounds, axis=1)
    layer_tops = np.r_[0.0, np.cumsum(thickness)]
    sample_depth = np.r_[layer_tops[:-1] + thickness / 2.0, layer_tops[-1]]
    guide_depth, guide_velocity = _unique_depth_velocity(
        guides["guide_depth_m"], guides["phase_velocity_m_s"]
    )
    interpolated = np.interp(sample_depth, guide_depth, guide_velocity)
    outside = (sample_depth < guide_depth[0]) | (sample_depth > guide_depth[-1])
    unclipped_vs = interpolated * float(velocity_factor)
    vs = np.clip(unclipped_vs, velocity_bounds[:, 0], velocity_bounds[:, 1])
    clipped = vs != unclipped_vs
    return {
        "parameters": np.r_[thickness, vs],
        "sample_depth_m": sample_depth,
        "phase_velocity_guide_m_s": interpolated,
        "vs_unclipped_m_s": unclipped_vs,
        "clipped_to_bounds": clipped,
        "outside_guide_depth": outside,
    }


def wavelength_diagnostics(
    frequencies_hz: ArrayLike,
    phase_velocity_m_s: ArrayLike,
    maximum_spacing_m: float,
    *,
    maximum_wavelength_ratio: float | None = None,
) -> DiagnosticResult:
    """Assess wavelengths against a configurable project aperture policy.

    ``None`` records the ratio while marking every wavelength as within limit;
    it does not impose a universal array-spacing rule.
    """
    if not np.isfinite(maximum_spacing_m) or maximum_spacing_m <= 0:
        raise ValueError("maximum_spacing_m must be positive and finite")
    if maximum_wavelength_ratio is not None and (
        not np.isfinite(maximum_wavelength_ratio) or maximum_wavelength_ratio <= 0
    ):
        raise ValueError("maximum_wavelength_ratio must be positive and finite when provided")
    guides = wavelength_guides(frequencies_hz, phase_velocity_m_s)
    ratio = guides["wavelength_m"] / float(maximum_spacing_m)
    within = np.ones(ratio.shape, dtype=np.bool_)
    if maximum_wavelength_ratio is not None:
        within = ratio <= maximum_wavelength_ratio
    return {**guides, "wavelength_to_aperture": ratio, "within_wavelength_limit": within}


def depth_guidelines(frequencies_hz: ArrayLike, phase_velocity_m_s: ArrayLike) -> dict[str, float]:
    """Return the minimum ``lambda/3`` and maximum ``lambda/2`` depth guides."""
    guides = wavelength_guides(frequencies_hz, phase_velocity_m_s)
    wavelength = guides["wavelength_m"]
    return {
        "minimum_depth_m": float(np.min(wavelength) / 3.0),
        "maximum_depth_m": float(np.max(wavelength) / 2.0),
    }
