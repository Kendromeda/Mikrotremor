"""Constrained, shape-only HVSR comparison for layered Vs candidates."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from disba import DispersionError, Ellipticity  # type: ignore[import-untyped]
from numpy.typing import NDArray

from mhvsr_vs30.mam.inversion import VpRule, forward_rayleigh_phase


def forward_ellipticity(
    periods_s: NDArray[np.float64],
    thickness_m: NDArray[np.float64],
    velocity_s_m_s: NDArray[np.float64],
    *,
    poisson: float,
    density_g_cm3: float,
    vp_rule: VpRule | None = None,
) -> NDArray[np.float64]:
    """Return absolute fundamental-mode Rayleigh H/V eigenfunction ratio."""
    periods = np.asarray(periods_s, dtype=np.float64)
    thickness = np.asarray(thickness_m, dtype=np.float64)
    velocity = np.asarray(velocity_s_m_s, dtype=np.float64)
    if periods.ndim != 1 or np.any(np.diff(periods) <= 0) or np.any(periods <= 0):
        raise ValueError("periods must be positive and strictly increasing")
    if len(thickness) != len(velocity) - 1 or np.any(thickness <= 0) or np.any(velocity <= 0):
        raise ValueError("layer dimensions and speeds are invalid")
    rule = vp_rule or VpRule(method="constant_poisson", poisson=poisson)
    solver = Ellipticity(
        np.r_[thickness / 1000.0, 1.0],
        rule.vp(thickness, velocity) / 1000.0,
        velocity / 1000.0,
        np.full(len(velocity), density_g_cm3),
    )
    curve = solver(periods, mode=0)
    values = np.abs(np.asarray(curve.ellipticity, dtype=np.float64))
    if len(values) != len(periods) or not np.isfinite(values).all() or np.any(values <= 0):
        raise RuntimeError("ellipticity is incomplete or invalid")
    return values


def log_shape_correlation(observed: NDArray[np.float64], theory: NDArray[np.float64]) -> float:
    """Compare curve shapes without fitting their absolute amplitudes."""
    left = np.log(np.asarray(observed, dtype=np.float64))
    right = np.log(np.asarray(theory, dtype=np.float64))
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("shape curves must be equal-length, positive and finite")
    if np.std(left) < 0.01 or np.std(right) < 0.01:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def refine_candidate(
    initial: NDArray[np.float64],
    bounds: Sequence[tuple[float, float]],
    dispersion_periods_s: NDArray[np.float64],
    dispersion_velocity_m_s: NDArray[np.float64],
    hvsr_periods_s: NDArray[np.float64],
    hvsr_amplitude: NDArray[np.float64],
    *,
    poisson: float,
    density_g_cm3: float,
    seed: int,
    candidate_count: int,
    perturbation_std: float,
    max_rmse_increase_m_s: float,
    vp_rule: VpRule | None = None,
) -> tuple[NDArray[np.float64], list[dict[str, float | bool | int]]]:
    """Sample small perturbations while preserving the dispersion misfit bound."""
    base = np.asarray(initial, dtype=np.float64)
    n_finite = (len(base) - 1) // 2
    if len(bounds) != len(base) or len(base) != 2 * n_finite + 1:
        raise ValueError("parameter bounds do not match layer structure")
    if candidate_count < 0 or perturbation_std <= 0 or max_rmse_increase_m_s < 0:
        raise ValueError("candidate count and perturbation settings are invalid")

    def rmse(parameters: NDArray[np.float64]) -> float:
        predicted = forward_rayleigh_phase(
            dispersion_periods_s,
            parameters[:n_finite],
            parameters[n_finite:],
            poisson=poisson,
            density_g_cm3=density_g_cm3,
            vp_rule=vp_rule,
        )
        return float(np.sqrt(np.mean((predicted - dispersion_velocity_m_s) ** 2)))

    def score(parameters: NDArray[np.float64]) -> float:
        theory = forward_ellipticity(
            hvsr_periods_s,
            parameters[:n_finite],
            parameters[n_finite:],
            poisson=poisson,
            density_g_cm3=density_g_cm3,
            vp_rule=vp_rule,
        )
        return log_shape_correlation(hvsr_amplitude, theory)

    baseline_rmse = rmse(base)
    baseline_score = score(base)
    best, best_score = base.copy(), baseline_score
    rows: list[dict[str, float | bool | int]] = [
        {"candidate": 0, "shape_correlation": baseline_score,
         "dispersion_rmse_m_s": baseline_rmse, "accepted_dispersion": True}
    ]
    rng = np.random.default_rng(seed)
    for index in range(1, candidate_count + 1):
        candidate = base * (1 + rng.normal(0, perturbation_std, len(base)))
        candidate = np.array(
            [
                np.clip(value, low, high)
                for value, (low, high) in zip(candidate, bounds, strict=True)
            ]
        )
        try:
            error = rmse(candidate)
            if error > baseline_rmse + max_rmse_increase_m_s:
                rows.append({"candidate": index, "shape_correlation": float("nan"),
                             "dispersion_rmse_m_s": error, "accepted_dispersion": False})
                continue
            similarity = score(candidate)
        except (DispersionError, RuntimeError, ValueError):
            continue
        rows.append({"candidate": index, "shape_correlation": similarity,
                     "dispersion_rmse_m_s": error, "accepted_dispersion": True})
        if np.isfinite(similarity) and (not np.isfinite(best_score) or similarity > best_score):
            best, best_score = candidate, similarity
    return best, rows
