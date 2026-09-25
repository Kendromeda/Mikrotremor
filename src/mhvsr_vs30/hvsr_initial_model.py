"""Data-driven initial model and search bounds for single-site HVSR inversion.

This ports the heuristic of the ``Titik_1 (Coba_pakai_initial_model)`` notebook
into a tested module:

1. smooth the H/V curve with a Konno-Ohmachi window;
2. treat peaks of ``|H/V - smoothed H/V|`` as candidate layer boundaries;
3. convert their frequency to depth with a power-law velocity profile
   ``Vs(z) = V0 * (1 + z) ** x``;
4. assign each layer the power-law Vs at its mid-depth and widen thickness
   and Vs by relative tolerances to obtain PSO search bounds.

The result is a starting point that adapts to each curve.  It is not a
measured profile: the power-law coefficients, tolerances and boundary count
are user assumptions, and every value can be overridden layer by layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks  # type: ignore[import-untyped]

FloatArray = NDArray[np.float64]
DepthMethod = Literal["quarter_wavelength_v0", "power_law_travel_time"]
BROCHER_VS_LIMIT_M_S = 4500.0


def _readonly(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64).copy()
    array.setflags(write=False)
    return array


def konno_ohmachi_smooth(
    frequency_hz: ArrayLike, amplitude: ArrayLike, bandwidth: float = 40.0
) -> FloatArray:
    """Smooth a curve sampled at ``frequency_hz`` with a Konno-Ohmachi window."""
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    values = np.asarray(amplitude, dtype=np.float64)
    if frequency.ndim != 1 or values.shape != frequency.shape or frequency.size < 2:
        raise ValueError("frequency and amplitude must be matching 1D arrays of length >= 2")
    if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0):
        raise ValueError("frequency must be positive and finite")
    if not np.all(np.isfinite(values)):
        raise ValueError("amplitude must be finite")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be positive and finite")
    argument = bandwidth * np.log10(frequency[None, :] / frequency[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = (np.sin(argument) / argument) ** 4
    weights[argument == 0] = 1.0
    return np.asarray(weights @ values / weights.sum(axis=1), dtype=np.float64)


@dataclass(frozen=True)
class PowerLawVs:
    """Empirical ``Vs(z) = V0 * (1 + z) ** exponent`` profile (z in metres)."""

    v0_m_s: float = 175.0
    exponent: float = 0.099

    def __post_init__(self) -> None:
        if not np.isfinite(self.v0_m_s) or self.v0_m_s <= 0:
            raise ValueError("V0 must be positive and finite")
        if not np.isfinite(self.exponent) or not 0 <= self.exponent < 1:
            raise ValueError("power-law exponent must be in [0, 1)")

    def vs_at(self, depth_m: ArrayLike) -> FloatArray:
        depth = np.asarray(depth_m, dtype=np.float64)
        if not np.all(np.isfinite(depth)) or np.any(depth < 0):
            raise ValueError("depth must be finite and nonnegative")
        return np.asarray(self.v0_m_s * (1.0 + depth) ** self.exponent, dtype=np.float64)

    def depth_from_frequency(
        self, frequency_hz: ArrayLike, method: DepthMethod = "quarter_wavelength_v0"
    ) -> FloatArray:
        """Convert resonance frequency to depth.

        ``quarter_wavelength_v0`` is the ``z = V0 / (4 f)`` rule of the source
        notebook.  ``power_law_travel_time`` instead solves ``f = 1 / (4 t(z))``
        with the vertical travel time through the power-law profile, which gives
        ``z = (1 + (1 - x) V0 / (4 f)) ** (1 / (1 - x)) - 1`` and reduces to the
        first rule when ``x = 0``.
        """
        frequency = np.asarray(frequency_hz, dtype=np.float64)
        if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0):
            raise ValueError("frequency must be positive and finite")
        quarter = self.v0_m_s / (4.0 * frequency)
        if method == "quarter_wavelength_v0":
            return np.asarray(quarter, dtype=np.float64)
        if method == "power_law_travel_time":
            power = 1.0 - self.exponent
            depth = (1.0 + power * quarter) ** (1.0 / power) - 1.0
            return np.asarray(depth, dtype=np.float64)
        raise ValueError(f"unknown depth method: {method!r}")


@dataclass(frozen=True)
class InitialModelSuggestion:
    """Curve-derived starting model plus the PSO bounds built around it."""

    frequency_hz: FloatArray
    depth_m: FloatArray
    smoothed_amplitude: FloatArray
    contrast: FloatArray
    boundary_indices: NDArray[np.int64]
    boundary_depths_m: FloatArray
    halfspace_reference_depth_m: float
    thickness_m: FloatArray
    vs_m_s: FloatArray
    thickness_bounds_m: FloatArray
    vs_bounds_m_s: FloatArray
    used_f0_fallback: bool

    @property
    def finite_layer_count(self) -> int:
        return int(self.thickness_m.size)

    @property
    def particle(self) -> FloatArray:
        """Initial model in PSO order ``[h_1..h_L, Vs_1..Vs_(L+1)]``."""
        return _readonly(np.r_[self.thickness_m, self.vs_m_s])


def _tolerance(value: float, label: str, *, upper: float) -> float:
    if not np.isfinite(value) or not 0 < value < upper:
        raise ValueError(f"{label} must be in (0, {upper:g})")
    return float(value)


def suggest_initial_model(
    frequency_hz: ArrayLike,
    amplitude: ArrayLike,
    *,
    profile: PowerLawVs | None = None,
    depth_method: DepthMethod = "quarter_wavelength_v0",
    konno_ohmachi_bandwidth: float = 40.0,
    max_boundaries: int = 5,
    min_peak_distance: int = 3,
    min_layer_thickness_m: float = 1.0,
    min_relative_contrast: float = 1e-3,
    vs_tolerance: float = 0.50,
    thickness_tolerance: float = 0.50,
    halfspace_vs_max_m_s: float | None = 3000.0,
) -> InitialModelSuggestion:
    """Derive layer boundaries, initial Vs and PSO bounds from one H/V curve.

    The ``max_boundaries`` strongest contrast peaks become the tops of layers
    2..L+1; the deepest one is the top of the halfspace.  Boundaries closer than
    ``min_layer_thickness_m`` to the previous one, and peaks weaker than
    ``min_relative_contrast`` times the median amplitude, are dropped.  When no peak
    survives, the depth of the H/V maximum (f0) is used as the single boundary.

    Vs bounds are ``Vs * (1 -/+ vs_tolerance)``.  The source notebook used 0.10
    for Dinver; for HVSR-only PSO that is usually too narrow to reproduce the
    impedance contrast behind the H/V peak, so the default is 0.50 and the
    halfspace upper bound is raised to ``halfspace_vs_max_m_s`` (``None`` keeps
    the symmetric tolerance).
    """
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    values = np.asarray(amplitude, dtype=np.float64)
    if frequency.ndim != 1 or values.shape != frequency.shape or frequency.size < 3:
        raise ValueError("frequency and amplitude must be matching 1D arrays of length >= 3")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("amplitude must be positive and finite")
    if max_boundaries < 1 or min_peak_distance < 1:
        raise ValueError("max_boundaries and min_peak_distance must be at least 1")
    if not np.isfinite(min_layer_thickness_m) or min_layer_thickness_m <= 0:
        raise ValueError("min_layer_thickness_m must be positive and finite")
    if not np.isfinite(min_relative_contrast) or min_relative_contrast < 0:
        raise ValueError("min_relative_contrast must be finite and nonnegative")
    vs_tol = _tolerance(vs_tolerance, "vs_tolerance", upper=1.0)
    if halfspace_vs_max_m_s is not None and not (
        np.isfinite(halfspace_vs_max_m_s) and 0 < halfspace_vs_max_m_s <= BROCHER_VS_LIMIT_M_S
    ):
        raise ValueError(f"halfspace_vs_max_m_s must be in (0, {BROCHER_VS_LIMIT_M_S:g}]")
    thickness_tol = _tolerance(thickness_tolerance, "thickness_tolerance", upper=1.0)
    velocity_profile = profile or PowerLawVs()

    order = np.argsort(frequency)
    frequency = frequency[order]
    values = values[order]
    if np.any(np.diff(frequency) <= 0):
        raise ValueError("frequency values must be unique")

    smoothed = konno_ohmachi_smooth(frequency, values, konno_ohmachi_bandwidth)
    contrast = values - smoothed
    depth = velocity_profile.depth_from_frequency(frequency, depth_method)
    strength = np.abs(contrast)

    peaks = np.asarray(find_peaks(strength, distance=min_peak_distance)[0], dtype=np.int64)
    peaks = peaks[strength[peaks] >= min_relative_contrast * float(np.median(values))]
    strongest = peaks[np.argsort(strength[peaks], kind="stable")[::-1][:max_boundaries]]
    kept: list[int] = []
    previous = 0.0
    for index in sorted(strongest.tolist(), key=lambda item: depth[item]):
        if depth[index] - previous >= min_layer_thickness_m:
            kept.append(index)
            previous = float(depth[index])
    used_fallback = not kept
    if used_fallback:
        f0_index = int(np.argmax(values))
        if depth[f0_index] < min_layer_thickness_m:
            raise ValueError(
                "no layer boundary deeper than min_layer_thickness_m; "
                "lower it or use a wider/lower frequency band"
            )
        kept = [f0_index]

    boundaries = depth[np.asarray(kept, dtype=np.int64)]
    deepest = float(depth.max())
    thickness = np.diff(np.r_[0.0, boundaries])
    tops = np.r_[0.0, boundaries[:-1]]
    halfspace_mid = 0.5 * (boundaries[-1] + deepest)
    vs = np.minimum(
        velocity_profile.vs_at(np.r_[tops + thickness / 2.0, halfspace_mid]),
        BROCHER_VS_LIMIT_M_S / (1.0 + vs_tol),
    )
    thickness_bounds = np.column_stack(
        (thickness * (1.0 - thickness_tol), thickness * (1.0 + thickness_tol))
    )
    vs_bounds = np.column_stack((vs * (1.0 - vs_tol), vs * (1.0 + vs_tol)))
    if halfspace_vs_max_m_s is not None and halfspace_vs_max_m_s > vs_bounds[-1, 1]:
        vs_bounds[-1, 1] = halfspace_vs_max_m_s
    return InitialModelSuggestion(
        frequency_hz=_readonly(frequency),
        depth_m=_readonly(depth),
        smoothed_amplitude=_readonly(smoothed),
        contrast=_readonly(contrast),
        boundary_indices=np.asarray(kept, dtype=np.int64),
        boundary_depths_m=_readonly(boundaries),
        halfspace_reference_depth_m=float(halfspace_mid),
        thickness_m=_readonly(thickness),
        vs_m_s=_readonly(vs),
        thickness_bounds_m=_readonly(thickness_bounds),
        vs_bounds_m_s=_readonly(vs_bounds),
        used_f0_fallback=used_fallback,
    )


def apply_bound_overrides(
    thickness_bounds_m: ArrayLike,
    vs_bounds_m_s: ArrayLike,
    overrides: Mapping[str, Mapping[int, Sequence[float]]] | None,
) -> tuple[FloatArray, FloatArray]:
    """Replace selected layer bounds; layer numbers start at 1.

    ``overrides = {"thickness_m": {2: (5, 15)}, "vs_m_s": {1: (120, 220)}}``.
    For Vs the last layer number is the halfspace.
    """
    thickness = np.array(thickness_bounds_m, dtype=np.float64)
    vs = np.array(vs_bounds_m_s, dtype=np.float64)
    if not overrides:
        return _readonly(thickness), _readonly(vs)
    unknown = set(overrides) - {"thickness_m", "vs_m_s"}
    if unknown:
        raise ValueError(f"unknown override keys: {sorted(unknown)}")
    for key, table in (("thickness_m", thickness), ("vs_m_s", vs)):
        for layer, interval in overrides.get(key, {}).items():
            if not 1 <= int(layer) <= table.shape[0]:
                raise ValueError(f"{key} override layer {layer} is outside 1..{table.shape[0]}")
            low, high = (float(value) for value in interval)
            if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high):
                raise ValueError(f"{key} override for layer {layer} needs 0 < min < max")
            table[int(layer) - 1] = (low, high)
    return _readonly(thickness), _readonly(vs)
