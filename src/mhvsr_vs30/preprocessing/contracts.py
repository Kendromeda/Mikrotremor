"""Contracts for the output of preprocessing.

A processed curve is evidence, not a working buffer: the arrays are read-only
and every judgement made on the way here travels with it. A consumer that only
reads ``mean_curve`` still cannot lose the fact that half the windows were
rejected or that the record never covered the model grid.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from mhvsr_vs30.exceptions import MhvsrVs30Error

__all__ = [
    "FrequencyCoverageQc",
    "PreprocessingError",
    "ProcessedHvsr",
    "ProcessingOutcome",
    "SesameQc",
    "WindowQc",
]


class PreprocessingError(MhvsrVs30Error):
    """A recording could not be turned into a curve worth keeping."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _readonly(values: Any, name: str) -> np.ndarray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim not in (1, 2):
        raise ValueError(f"{name} must be 1- or 2-dimensional, got shape {array.shape}")
    array.setflags(write=False)
    return array


@dataclasses.dataclass(frozen=True)
class WindowQc:
    """Why one window was kept or dropped."""

    index: int
    accepted: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SesameQc:
    """SESAME 2004 reliability and clarity outcomes.

    Stored as results, never as a filter: a curve that fails clarity is still a
    real measurement, and dropping it here would bias the corpus toward sites
    with sharp peaks.
    """

    reliability_passed: bool | None
    clarity_passed: bool | None
    reliability_detail: dict[str, Any] = dataclasses.field(default_factory=dict)
    clarity_detail: dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class FrequencyCoverageQc:
    """Which part of the frequency axis this record can actually support."""

    window_length_s: float
    significant_cycles: float
    nyquist_hz: float
    valid_frequency_min_hz: float
    valid_frequency_max_hz: float
    model_grid_min_hz: float
    model_grid_max_hz: float
    model_grid_covered: bool
    nyquist_margin: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ProcessedHvsr:
    """One recording turned into an mHVSR curve, with its full audit trail."""

    recording_id: str
    profile_id: str
    frequency_hz: np.ndarray
    window_amplitudes: np.ndarray
    accepted_window_mask: np.ndarray
    mean_curve: np.ndarray
    log_std_curve: np.ndarray
    valid_frequency_min_hz: float
    valid_frequency_max_hz: float
    model_frequency_hz: np.ndarray
    model_amplitude: np.ndarray | None
    model_ready: bool
    processing_hash: str
    window_qc: tuple[WindowQc, ...] = ()
    sesame: SesameQc | None = None
    coverage: FrequencyCoverageQc | None = None
    qc_flags: tuple[str, ...] = ()
    extrapolated: bool = False

    def __post_init__(self) -> None:
        for name in (
            "frequency_hz",
            "window_amplitudes",
            "accepted_window_mask",
            "mean_curve",
            "log_std_curve",
            "model_frequency_hz",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), name))
        if self.model_amplitude is not None:
            object.__setattr__(self, "model_amplitude", _readonly(self.model_amplitude, "model"))

        points = len(self.frequency_hz)
        if self.window_amplitudes.shape[1:] != (points,):
            raise ValueError(
                f"window_amplitudes has shape {self.window_amplitudes.shape}, "
                f"which does not match {points} frequencies"
            )
        if len(self.mean_curve) != points or len(self.log_std_curve) != points:
            raise ValueError("mean and log-std curves must match the frequency axis")
        if len(self.accepted_window_mask) != self.window_amplitudes.shape[0]:
            raise ValueError("accepted_window_mask must have one entry per window")
        if self.model_ready and self.model_amplitude is None:
            raise ValueError("model_ready cannot be true without a model amplitude vector")

    @property
    def accepted_window_count(self) -> int:
        return int(np.count_nonzero(self.accepted_window_mask))

    @property
    def window_count(self) -> int:
        return int(self.window_amplitudes.shape[0])

    def qc_status(self) -> str:
        """passed, flagged or failed. Reaching this contract at all rules out failed."""
        return "flagged" if self.qc_flags else "passed"

    def summary(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "profile_id": self.profile_id,
            "window_count": self.window_count,
            "accepted_window_count": self.accepted_window_count,
            "rejected_window_count": self.window_count - self.accepted_window_count,
            "valid_frequency_range_hz": [
                self.valid_frequency_min_hz,
                self.valid_frequency_max_hz,
            ],
            "model_ready": self.model_ready,
            "extrapolated": self.extrapolated,
            "qc_status": self.qc_status(),
            "qc_flags": list(self.qc_flags),
            "processing_hash": self.processing_hash,
        }


@dataclasses.dataclass(frozen=True)
class ProcessingOutcome:
    """What happened to one recording, whether or not a curve came out."""

    recording_id: str
    profile_id: str
    status: str
    curve: ProcessedHvsr | None = None
    failure_reason: str | None = None
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "recording_id": self.recording_id,
            "profile_id": self.profile_id,
            "status": self.status,
        }
        if self.curve is not None:
            payload.update(self.curve.summary())
            payload["status"] = self.status
        if self.failure_reason is not None:
            payload["failure_reason"] = self.failure_reason
            payload["message"] = self.message
        return payload
