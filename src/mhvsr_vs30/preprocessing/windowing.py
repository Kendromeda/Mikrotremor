"""Window-length policy and the frequency floor it implies.

This module decides how long a window is and, as a direct consequence, the
lowest frequency the record can support. Those two are kept together because
separating them is how a pipeline ends up reporting amplitudes at frequencies
the windows were never long enough to resolve.
"""

from __future__ import annotations

import math

from mhvsr_vs30.preprocessing.config import PreprocessingProfile
from mhvsr_vs30.preprocessing.contracts import PreprocessingError

__all__ = [
    "expected_window_count",
    "minimum_valid_frequency_hz",
    "window_length_seconds",
]


def window_length_seconds(profile: PreprocessingProfile, duration_s: float) -> float:
    """Window length for this recording under this profile."""
    windowing = profile.windowing
    if windowing.mode == "fixed_duration":
        assert windowing.window_length_s is not None
        length = float(windowing.window_length_s)
        if length > duration_s:
            raise PreprocessingError(
                "insufficient_initial_windows",
                f"window length {length} s exceeds the {duration_s:.1f} s record",
            )
        return length

    assert windowing.window_count is not None
    if duration_s <= 0:
        raise PreprocessingError("insufficient_initial_windows", "record has no duration")
    return duration_s / float(windowing.window_count)


def expected_window_count(duration_s: float, window_length_s: float) -> int:
    """How many whole windows fit. Partial trailing data is not a window."""
    if window_length_s <= 0:
        raise ValueError("window length must be positive")
    return math.floor(duration_s / window_length_s)


def minimum_valid_frequency_hz(profile: PreprocessingProfile, window_length_s: float) -> float:
    """The lowest frequency a window of this length resolves.

    Follows the significant-cycles rule the reference notebooks use: a window
    must contain at least N full cycles of a frequency before that frequency
    means anything.
    """
    if window_length_s <= 0:
        raise ValueError("window length must be positive")
    return float(profile.windowing.significant_cycles) / float(window_length_s)
