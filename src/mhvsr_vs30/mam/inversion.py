"""Unit-safe forward modeling and depth averaging for layered Vs models."""

from __future__ import annotations

import numpy as np
from disba import PhaseDispersion  # type: ignore[import-untyped]
from numpy.typing import NDArray


def _validated_layers(
    thickness_m: NDArray[np.float64], velocity_s_m_s: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    thickness = np.asarray(thickness_m, dtype=np.float64)
    velocity = np.asarray(velocity_s_m_s, dtype=np.float64)
    if thickness.ndim != 1 or velocity.ndim != 1 or len(thickness) != len(velocity) - 1:
        raise ValueError("thickness must have one fewer entry than Vs, which includes a halfspace")
    if len(velocity) < 2 or not np.all(np.isfinite(velocity)) or np.any(velocity <= 0):
        raise ValueError("Vs must contain at least two positive finite layer speeds")
    if not np.all(np.isfinite(thickness)) or np.any(thickness <= 0):
        raise ValueError("finite layer thicknesses must be positive and finite")
    return thickness, velocity


def forward_rayleigh_phase(
    periods_s: NDArray[np.float64],
    thickness_m: NDArray[np.float64],
    velocity_s_m_s: NDArray[np.float64],
    *,
    poisson: float = 0.3,
    density_g_cm3: float = 2.0,
) -> NDArray[np.float64]:
    """Return fundamental Rayleigh phase speed in m/s at ascending periods."""
    thickness, velocity = _validated_layers(thickness_m, velocity_s_m_s)
    periods = np.asarray(periods_s, dtype=np.float64)
    if periods.ndim != 1 or not np.all(np.isfinite(periods)) or np.any(periods <= 0):
        raise ValueError("periods must be positive and finite")
    if np.any(np.diff(periods) <= 0):
        raise ValueError("periods must be strictly increasing")
    if not 0 <= poisson < 0.5 or not density_g_cm3 > 0:
        raise ValueError("Poisson ratio and density must be physically valid")
    vp_vs = np.sqrt(2 * (1 - poisson) / (1 - 2 * poisson))
    solver = PhaseDispersion(
        np.r_[thickness / 1000.0, 1.0],
        velocity * vp_vs / 1000.0,
        velocity / 1000.0,
        np.full(len(velocity), density_g_cm3),
    )
    curve = solver(periods, mode=0, wave="rayleigh")
    if len(curve.period) != len(periods) or not np.allclose(curve.period, periods):
        raise RuntimeError("forward solver did not return every requested period")
    return np.asarray(curve.velocity, dtype=np.float64) * 1000.0


def vs30_from_layers(
    thickness_m: NDArray[np.float64], velocity_s_m_s: NDArray[np.float64]
) -> float:
    """Calculate mathematical Vs30, without claiming the model resolves 30 m."""
    thickness, velocity = _validated_layers(thickness_m, velocity_s_m_s)
    depth = 0.0
    travel_time = 0.0
    for layer_thickness, layer_velocity in zip(np.r_[thickness, np.inf], velocity, strict=True):
        used = min(float(layer_thickness), 30.0 - depth)
        travel_time += used / float(layer_velocity)
        depth += used
        if depth >= 30.0:
            break
    return 30.0 / travel_time
