"""Feature vectors derived from a processed mHVSR curve.

The model consumes the curve resampled onto a fixed grid. Nothing here invents
values: a curve that was not model-ready in preprocessing has no feature vector,
and asking for one is an error rather than a silent zero-fill.
"""

from __future__ import annotations

import numpy as np

from mhvsr_vs30.preprocessing.contracts import ProcessedHvsr

__all__ = ["FEATURE_COUNT", "feature_columns", "feature_vector"]

FEATURE_COUNT = 35


def feature_columns(count: int = FEATURE_COUNT) -> tuple[str, ...]:
    """Stable, zero-padded column names so the order survives any round trip."""
    return tuple(f"feature_{index:02d}" for index in range(count))


def feature_vector(curve: ProcessedHvsr) -> np.ndarray:
    """The model-grid amplitudes for one curve.

    Raises when the curve never reached the grid, because a caller that wanted a
    feature vector and got a filled-in one would not be able to tell.
    """
    if not curve.model_ready or curve.model_amplitude is None:
        raise ValueError(
            f"recording {curve.recording_id[:12]} is not model-ready: "
            "its curve does not cover the model frequency grid"
        )
    values = np.asarray(curve.model_amplitude, dtype=np.float64)
    if values.shape != (FEATURE_COUNT,):
        raise ValueError(f"expected {FEATURE_COUNT} features, got {values.shape}")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("feature vector must be finite and strictly positive")
    return values
