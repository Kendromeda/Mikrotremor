"""Unit-safe forward modeling and depth averaging for layered Vs models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from disba import PhaseDispersion  # type: ignore[import-untyped]
from numpy.typing import NDArray

VpMethod = Literal["groundwater", "constant_poisson"]
KITSUNEZAKI_SLOPE = 1.11
KITSUNEZAKI_INTERCEPT_M_S = 1290.0


@dataclass(frozen=True)
class VpRule:
    """How Vp follows Vs in each layer of a dispersion forward model.

    ``groundwater`` follows Hayashi et al. (2022, Section 7.2): above the water
    table ``Vp = above_vp_vs_ratio * Vs`` (2 by default); at and below it the
    Kitsunezaki et al. (1990) relation ``Vp = 1.11 Vs + 1290 m/s``.  A layer is
    saturated when its mid-depth is at or below ``groundwater_depth_m``; the
    halfspace is saturated whenever the water table is finite.  The default
    water table at the surface is the paper's conservative choice when no
    groundwater information exists.  ``constant_poisson`` keeps the earlier
    project assumption of one Poisson ratio for every layer.
    """

    method: VpMethod = "groundwater"
    groundwater_depth_m: float = 0.0
    above_vp_vs_ratio: float = 2.0
    poisson: float = 0.3

    def __post_init__(self) -> None:
        if self.method not in ("groundwater", "constant_poisson"):
            raise ValueError(f"unknown Vp method: {self.method!r}")
        if np.isnan(self.groundwater_depth_m) or self.groundwater_depth_m < 0:
            raise ValueError("groundwater_depth_m must be nonnegative (inf means dry)")
        if not np.isfinite(self.above_vp_vs_ratio) or self.above_vp_vs_ratio <= np.sqrt(2):
            raise ValueError("above_vp_vs_ratio must exceed sqrt(2)")
        if not 0 <= self.poisson < 0.5:
            raise ValueError("Poisson ratio must be in [0, 0.5)")

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None, *, poisson: float = 0.3) -> VpRule:
        """Build a rule from a notebook dict; ``None`` keeps the constant-Poisson rule."""
        if config is None:
            return cls(method="constant_poisson", poisson=poisson)
        values = {"poisson": poisson, **dict(config)}
        if values.get("groundwater_depth_m") is None:
            values["groundwater_depth_m"] = float("inf")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        if not np.isfinite(self.groundwater_depth_m):
            values["groundwater_depth_m"] = None
        return values

    def vp(self, thickness_m: NDArray[np.float64], velocity_s_m_s: NDArray[np.float64]) -> (
        NDArray[np.float64]
    ):
        """Return Vp in m/s for finite layers plus the halfspace."""
        thickness, velocity = _validated_layers(thickness_m, velocity_s_m_s)
        if self.method == "constant_poisson":
            ratio = np.sqrt(2 * (1 - self.poisson) / (1 - 2 * self.poisson))
            return np.asarray(velocity * ratio, dtype=np.float64)
        tops = np.r_[0.0, np.cumsum(thickness)]
        mid_depth = np.r_[tops[:-1] + thickness / 2.0, np.inf]
        saturated = mid_depth >= self.groundwater_depth_m
        saturated[-1] = bool(np.isfinite(self.groundwater_depth_m))
        vp = np.where(
            saturated,
            KITSUNEZAKI_SLOPE * velocity + KITSUNEZAKI_INTERCEPT_M_S,
            self.above_vp_vs_ratio * velocity,
        )
        if np.any(vp <= np.sqrt(2) * velocity):
            raise ValueError("Vp rule gives a negative Poisson ratio for this Vs")
        return np.asarray(vp, dtype=np.float64)


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
    vp_rule: VpRule | None = None,
) -> NDArray[np.float64]:
    """Return fundamental Rayleigh phase speed in m/s at ascending periods.

    ``vp_rule`` overrides the constant ``poisson`` assumption when given.
    """
    thickness, velocity = _validated_layers(thickness_m, velocity_s_m_s)
    periods = np.asarray(periods_s, dtype=np.float64)
    if periods.ndim != 1 or not np.all(np.isfinite(periods)) or np.any(periods <= 0):
        raise ValueError("periods must be positive and finite")
    if np.any(np.diff(periods) <= 0):
        raise ValueError("periods must be strictly increasing")
    if not 0 <= poisson < 0.5 or not density_g_cm3 > 0:
        raise ValueError("Poisson ratio and density must be physically valid")
    rule = vp_rule or VpRule(method="constant_poisson", poisson=poisson)
    solver = PhaseDispersion(
        np.r_[thickness / 1000.0, 1.0],
        rule.vp(thickness, velocity) / 1000.0,
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
