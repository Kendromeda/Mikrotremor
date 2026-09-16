"""Resampling onto the model feature grid.

The only interesting decision here is what to do when the curve does not reach
the whole grid. Extrapolating fills the gap with numbers no measurement
supports, so it happens only where a profile explicitly asks for it in order to
reproduce legacy behaviour.
"""

from __future__ import annotations

import numpy as np

from mhvsr_vs30.preprocessing.config import PreprocessingProfile

__all__ = ["model_grid_hz", "resample_to_model_grid"]


def model_grid_hz(profile: PreprocessingProfile) -> np.ndarray:
    grid = profile.model_grid
    return np.logspace(np.log10(grid.min_hz), np.log10(grid.max_hz), grid.points)


def resample_to_model_grid(
    frequency_hz: np.ndarray,
    amplitude: np.ndarray,
    profile: PreprocessingProfile,
    valid_min_hz: float,
    valid_max_hz: float,
) -> tuple[np.ndarray | None, bool, bool]:
    """Return (model amplitudes, covered, extrapolated).

    ``covered`` says the whole grid sits inside the validated frequency range.
    When it does not, a profile that forbids extrapolation returns None rather
    than a curve that looks complete but is not.
    """
    grid = model_grid_hz(profile)
    covered = bool(grid[0] >= valid_min_hz and grid[-1] <= valid_max_hz)

    if not covered and not profile.model_grid.allow_extrapolation:
        return None, False, False

    usable = (frequency_hz >= valid_min_hz) & (frequency_hz <= valid_max_hz)
    if np.count_nonzero(usable) < 2:
        return None, False, False

    source_f = frequency_hz[usable]
    source_a = amplitude[usable]
    inside = (grid >= source_f[0]) & (grid <= source_f[-1])
    extrapolated = bool(not inside.all())

    if extrapolated and not profile.model_grid.allow_extrapolation:
        return None, covered, False

    # Interpolate in log-frequency, matching the geometric spacing of both grids.
    values = np.interp(
        np.log10(grid),
        np.log10(source_f),
        source_a,
        left=np.nan,
        right=np.nan,
    )
    if extrapolated:
        # Legacy behaviour: linear extension beyond the measured range.
        values = np.interp(np.log10(grid), np.log10(source_f), source_a)
    if not np.isfinite(values).all() or (values <= 0).any():
        return None, covered, extrapolated
    return values, covered, extrapolated
