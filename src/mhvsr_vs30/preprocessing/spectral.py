"""Spectral core, delegated to hvsrpy.

The detrend, taper, FFT, Konno-Ohmachi smoothing and horizontal combination all
run through hvsrpy 2.0.0 rather than being reimplemented here. That is a
deliberate choice: the compatibility profile has to match the library that
produced the published models, and reimplementing the same maths a second way
would guarantee the two profiles stopped being comparable to each other.

What this project owns is the policy around that core: window length, which
frequencies are admissible, which windows survive, and what gets written down.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mhvsr_vs30.io.base import ThreeComponentRecord
from mhvsr_vs30.preprocessing.config import PreprocessingProfile
from mhvsr_vs30.preprocessing.contracts import PreprocessingError

__all__ = ["compute_windowed_hvsr", "internal_grid_hz", "to_seismic_recording"]


def internal_grid_hz(profile: PreprocessingProfile, minimum_frequency_hz: float) -> np.ndarray:
    """The smoothing grid, trimmed to frequencies the windows can resolve."""
    smoothing = profile.smoothing
    grid = np.geomspace(
        smoothing.internal_grid_min_hz,
        smoothing.internal_grid_max_hz,
        smoothing.internal_grid_points,
    )
    usable = grid > minimum_frequency_hz
    if np.count_nonzero(usable) < 2:
        raise PreprocessingError(
            "insufficient_initial_windows",
            f"no smoothing frequency exceeds the {minimum_frequency_hz:.4g} Hz floor",
        )
    return grid[usable]


def to_seismic_recording(record: ThreeComponentRecord) -> Any:
    """Wrap a canonical record as an hvsrpy SeismicRecording3C.

    hvsrpy takes north, east and vertical in that order; getting it wrong would
    silently swap the numerator and denominator of the ratio.
    """
    from hvsrpy import SeismicRecording3C, TimeSeries

    dt = 1.0 / float(record.sampling_rate_hz)
    return SeismicRecording3C(
        ns=TimeSeries(np.asarray(record.north, dtype=np.float64), dt),
        ew=TimeSeries(np.asarray(record.east, dtype=np.float64), dt),
        vt=TimeSeries(np.asarray(record.z, dtype=np.float64), dt),
    )


def compute_windowed_hvsr(
    record: ThreeComponentRecord,
    profile: PreprocessingProfile,
    window_length_s: float,
    center_frequencies_hz: np.ndarray,
) -> Any:
    """Run hvsrpy preprocessing and processing, returning the per-window curves."""
    import hvsrpy

    pre = hvsrpy.settings.HvsrPreProcessingSettings()
    pre.detrend = profile.windowing.detrend
    pre.window_length_in_seconds = float(window_length_s)

    proc = hvsrpy.settings.HvsrTraditionalProcessingSettings()
    proc.window_type_and_width = (profile.windowing.taper, profile.windowing.taper_width)
    proc.smoothing = dict(
        operator=profile.smoothing.operator,
        bandwidth=profile.smoothing.bandwidth,
        center_frequencies_in_hz=center_frequencies_hz,
    )
    proc.method_to_combine_horizontals = profile.method_to_combine_horizontals
    proc.handle_dissimilar_time_steps_by = "frequency_domain_resampling"

    records = hvsrpy.preprocess([to_seismic_recording(record)], pre)
    if not records:
        raise PreprocessingError(
            "insufficient_initial_windows", "preprocessing produced no windows"
        )
    hvsr = hvsrpy.process(records, proc)
    return hvsrpy.HvsrTraditional(frequency=hvsr.frequency, amplitude=hvsr.amplitude)
