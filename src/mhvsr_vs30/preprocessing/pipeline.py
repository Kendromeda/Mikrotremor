"""Turn one canonical three-component record into one mHVSR curve.

The pipeline never writes into the source tree and never returns a partial
result: a recording either yields a curve carrying its whole audit trail, or it
raises a PreprocessingError naming the reason.
"""

from __future__ import annotations

import numpy as np

from mhvsr_vs30.hashing import stable_id
from mhvsr_vs30.io.base import ThreeComponentRecord
from mhvsr_vs30.preprocessing import windowing
from mhvsr_vs30.preprocessing.config import PreprocessingProfile
from mhvsr_vs30.preprocessing.contracts import (
    FrequencyCoverageQc,
    PreprocessingError,
    ProcessedHvsr,
    ProcessingOutcome,
)
from mhvsr_vs30.preprocessing.rejection import reject_windows
from mhvsr_vs30.preprocessing.resampling import model_grid_hz, resample_to_model_grid
from mhvsr_vs30.preprocessing.sesame_qc import evaluate_sesame
from mhvsr_vs30.preprocessing.spectral import compute_windowed_hvsr, internal_grid_hz

__all__ = ["process_recording", "safe_process_recording"]

_THIRTY_MINUTES_S = 1800.0
_NYQUIST_MARGIN_WARN = 0.9


def process_recording(record: ThreeComponentRecord, profile: PreprocessingProfile) -> ProcessedHvsr:
    """Process one record, raising rather than returning a curve it cannot defend."""
    duration_s = record.duration_s
    nyquist_hz = record.sampling_rate_hz / 2.0

    window_length_s = windowing.window_length_seconds(profile, duration_s)
    available = windowing.expected_window_count(duration_s, window_length_s)
    if available < profile.windowing.minimum_windows:
        raise PreprocessingError(
            "insufficient_initial_windows",
            f"{available} whole windows of {window_length_s:.1f} s fit in "
            f"{duration_s:.1f} s, below the minimum of {profile.windowing.minimum_windows}",
        )

    minimum_frequency_hz = windowing.minimum_valid_frequency_hz(profile, window_length_s)
    center_frequencies = internal_grid_hz(profile, minimum_frequency_hz)

    hvsr = compute_windowed_hvsr(record, profile, window_length_s, center_frequencies)
    mask, window_qc = reject_windows(hvsr, profile)

    frequency = np.asarray(hvsr.frequency, dtype=np.float64)
    amplitudes = np.asarray(hvsr.amplitude, dtype=np.float64)
    mean_curve = np.asarray(hvsr.mean_curve(distribution=profile.distribution), dtype=np.float64)
    std_curve = np.asarray(hvsr.std_curve(distribution=profile.distribution), dtype=np.float64)

    if not np.isfinite(mean_curve).all():
        raise PreprocessingError("nonpositive_hvsr", "mean curve contains non-finite values")
    if (mean_curve <= 0).any():
        raise PreprocessingError("nonpositive_hvsr", "mean curve contains non-positive values")

    valid_min_hz = float(max(minimum_frequency_hz, frequency[0]))
    valid_max_hz = float(min(nyquist_hz, frequency[-1]))
    if valid_max_hz <= valid_min_hz:
        raise PreprocessingError(
            "invalid_output_shape",
            f"validated frequency range is empty ({valid_min_hz:.4g}-{valid_max_hz:.4g} Hz)",
        )

    model_amplitude, covered, extrapolated = resample_to_model_grid(
        frequency, mean_curve, profile, valid_min_hz, valid_max_hz
    )
    grid = model_grid_hz(profile)
    coverage = FrequencyCoverageQc(
        window_length_s=window_length_s,
        significant_cycles=profile.windowing.significant_cycles,
        nyquist_hz=nyquist_hz,
        valid_frequency_min_hz=valid_min_hz,
        valid_frequency_max_hz=valid_max_hz,
        model_grid_min_hz=float(grid[0]),
        model_grid_max_hz=float(grid[-1]),
        model_grid_covered=covered,
        nyquist_margin=float(grid[-1] / nyquist_hz) if nyquist_hz else float("inf"),
    )

    sesame = evaluate_sesame(hvsr, window_length_s, profile.distribution)
    flags = _collect_flags(record, profile, coverage, mask, sesame, covered, extrapolated)

    return ProcessedHvsr(
        recording_id=record.recording_id,
        profile_id=profile.profile_id,
        frequency_hz=frequency,
        window_amplitudes=amplitudes,
        accepted_window_mask=mask.astype(np.float64),
        mean_curve=mean_curve,
        log_std_curve=std_curve,
        valid_frequency_min_hz=valid_min_hz,
        valid_frequency_max_hz=valid_max_hz,
        model_frequency_hz=grid,
        model_amplitude=model_amplitude,
        model_ready=model_amplitude is not None,
        processing_hash=stable_id(profile.identity(), record.recording_id, repr(window_length_s)),
        window_qc=window_qc,
        sesame=sesame,
        coverage=coverage,
        qc_flags=flags,
        extrapolated=extrapolated,
    )


def _collect_flags(
    record: ThreeComponentRecord,
    profile: PreprocessingProfile,
    coverage: FrequencyCoverageQc,
    mask: np.ndarray,
    sesame: object,
    covered: bool,
    extrapolated: bool,
) -> tuple[str, ...]:
    """Soft flags: reasons to look closer, not reasons to discard."""
    flags: list[str] = []

    if record.repairs:
        flags.append("repaired_source_row")
    if record.duration_s < _THIRTY_MINUTES_S:
        flags.append("shorter_than_30_minutes")
    if not covered:
        flags.append("frequency_grid_not_fully_covered")
    if coverage.nyquist_margin >= _NYQUIST_MARGIN_WARN:
        flags.append("low_nyquist_margin")
    if extrapolated:
        flags.append("legacy_extrapolation_applied")

    total = int(mask.size)
    accepted = int(np.count_nonzero(mask))
    if total and (total - accepted) / total > 0.5:
        flags.append("high_rejection_fraction")

    reliability = getattr(sesame, "reliability_passed", None)
    clarity = getattr(sesame, "clarity_passed", None)
    if reliability is not True:
        flags.append("sesame_reliability_failed")
    if clarity is not True:
        flags.append("sesame_clarity_failed")
    if record.units in (None, "", "unknown"):
        flags.append("unknown_instrument_response")

    return tuple(flags)


def safe_process_recording(
    record: ThreeComponentRecord, profile: PreprocessingProfile
) -> ProcessingOutcome:
    """Process one record, reporting failure as data instead of an exception.

    Batch runs need every recording accounted for, so a failure becomes an
    outcome row naming its reason rather than aborting the run.
    """
    try:
        curve = process_recording(record, profile)
    except PreprocessingError as error:
        return ProcessingOutcome(
            recording_id=record.recording_id,
            profile_id=profile.profile_id,
            status="failed",
            failure_reason=error.reason,
            message=str(error),
        )
    return ProcessingOutcome(
        recording_id=record.recording_id,
        profile_id=profile.profile_id,
        status=curve.qc_status(),
        curve=curve,
    )
