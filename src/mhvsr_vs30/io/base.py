"""Canonical in-memory records and the reader interface.

A reader either returns a valid record or raises. It never returns None, never
returns a partially populated object, and never writes anything to disk. That
keeps failure visible in a QC report instead of hidden in a null column.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mhvsr_vs30.contracts import RawAsset, Recording
from mhvsr_vs30.exceptions import (
    MetadataConflictError,
    NonFiniteSampleError,
    SamplingRateMismatchError,
)

__all__ = [
    "HvsrCurve",
    "ReadRequest",
    "RecordingReader",
    "RepairRecord",
    "SurveyInspection",
    "ThreeComponentRecord",
]

COMPONENT_CODES = ("Z", "N", "E")


def _freeze(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise MetadataConflictError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise NonFiniteSampleError(f"{name} contains NaN or infinity")
    frozen = array.copy()
    frozen.setflags(write=False)
    return frozen


@dataclasses.dataclass(frozen=True)
class RepairRecord:
    """One repair that was actually applied, and what it cost.

    Carried on the record rather than logged, so a curve built from repaired
    input can never be mistaken downstream for a clean one.
    """

    action: str
    reason: str
    samples_dropped: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ThreeComponentRecord:
    """Three equal-length components on one clock, in source units.

    The arrays are read-only: every later stage receives the same samples the
    reader produced, so a preprocessing bug cannot quietly rewrite the input.
    """

    z: np.ndarray
    north: np.ndarray
    east: np.ndarray
    sampling_rate_hz: float
    start_time: datetime | None
    units: str
    source_channels: Mapping[str, str]
    recording_id: str
    repairs: tuple[RepairRecord, ...] = ()

    def __post_init__(self) -> None:
        for name in ("z", "north", "east"):
            object.__setattr__(self, name, _freeze(getattr(self, name), name))

        lengths = {name: len(getattr(self, name)) for name in ("z", "north", "east")}
        if len(set(lengths.values())) != 1:
            raise MetadataConflictError(f"components have unequal sample counts: {lengths}")
        if not lengths["z"]:
            raise MetadataConflictError("components are empty")
        if not np.isfinite(self.sampling_rate_hz) or self.sampling_rate_hz <= 0:
            raise SamplingRateMismatchError(
                f"sampling rate must be positive and finite, got {self.sampling_rate_hz}"
            )
        object.__setattr__(self, "source_channels", dict(self.source_channels))

    @property
    def n_samples(self) -> int:
        return len(self.z)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sampling_rate_hz

    def structural_qc(self) -> dict[str, Any]:
        """Facts a reviewer needs before trusting this record as an mHVSR input."""
        return {
            "recording_id": self.recording_id,
            "samples": self.n_samples,
            "sampling_rate_hz": self.sampling_rate_hz,
            "duration_s": self.duration_s,
            "components": list(COMPONENT_CODES),
            "source_channels": dict(self.source_channels),
            "units": self.units,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "structural_qc": "passed_with_repair" if self.repairs else "passed",
            "repair_actions": [repair.as_dict() for repair in self.repairs],
            "samples_dropped_by_repair": sum(r.samples_dropped for r in self.repairs),
        }


@dataclasses.dataclass(frozen=True)
class HvsrCurve:
    """An already-processed H/V curve.

    Deliberately not a ThreeComponentRecord: a published curve carries somebody
    else processing choices, so it must never be mixed with raw waveforms when
    evaluating a preprocessing pipeline.
    """

    frequency_hz: np.ndarray
    mean_amplitude: np.ndarray
    std_amplitude: np.ndarray | None
    recording_id: str
    input_level: str = "processed_curve"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frequency_hz", _freeze(self.frequency_hz, "frequency_hz"))
        object.__setattr__(self, "mean_amplitude", _freeze(self.mean_amplitude, "mean_amplitude"))
        if self.std_amplitude is not None:
            object.__setattr__(self, "std_amplitude", _freeze(self.std_amplitude, "std_amplitude"))

        if (self.frequency_hz <= 0).any():
            raise MetadataConflictError("frequencies must be strictly positive")
        if not (np.diff(self.frequency_hz) > 0).all():
            raise MetadataConflictError("frequencies must be strictly increasing")
        if len(self.frequency_hz) != len(self.mean_amplitude):
            raise MetadataConflictError("frequency and amplitude arrays differ in length")
        if self.std_amplitude is not None and len(self.std_amplitude) != len(self.frequency_hz):
            raise MetadataConflictError("standard deviation array differs in length")

    def structural_qc(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "input_level": self.input_level,
            "points": len(self.frequency_hz),
            "frequency_hz": [float(self.frequency_hz[0]), float(self.frequency_hz[-1])],
            "has_std": self.std_amplitude is not None,
            "structural_qc": "passed",
        }


@dataclasses.dataclass(frozen=True)
class SurveyInspection:
    """What a multichannel survey file turned out to be.

    SEG-2 and friends are inspected, not converted: a shot gather is not an
    ambient-noise recording just because it has three or more channels.
    """

    recording_id: str
    detected_format: str
    trace_count: int
    sampling_rate_hz: float
    samples_per_trace: int
    channel_codes: tuple[str, ...]
    is_three_component_candidate: bool
    reason: str

    def structural_qc(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ReadRequest:
    """Everything a reader needs to find and interpret one recording.

    A manifest row alone cannot locate bytes on disk, so the resolved path and
    the source root travel with it.
    """

    recording: Recording
    asset: RawAsset
    path: Path
    root: Path
    channel_codes: Mapping[str, str] | None = None
    component_traces: Mapping[str, int] | None = None
    curve_columns: Sequence[str] = ()
    siblings: Sequence[Path] = ()
    declared_sampling_rate_hz: float | None = None
    units: str | None = None
    component_order: Sequence[str] = ("vertical", "north", "east")
    # Permission to repair this exact file, already checked against its checksum
    # by the resolver. None means the reader must refuse defective input.
    repair: Any | None = None


@runtime_checkable
class RecordingReader(Protocol):
    """Turns one raw asset into a canonical record."""

    format_name: str

    def read(self, request: ReadRequest) -> Any:
        """Return a ThreeComponentRecord, HvsrCurve or SurveyInspection, or raise."""
        ...
