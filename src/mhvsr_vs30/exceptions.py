"""Structured errors.

Every failure mode this pipeline knows about gets its own type so a QC report
can count failures by cause instead of grepping message strings.
"""

from __future__ import annotations

__all__ = [
    "AmbiguousChannelError",
    "ConfigError",
    "ContractViolationError",
    "MalformedAsciiError",
    "ManifestValidationError",
    "MetadataConflictError",
    "MhvsrVs30Error",
    "MissingComponentError",
    "NoCommonTimeWindowError",
    "NonFiniteSampleError",
    "RecordingReadError",
    "SamplingRateMismatchError",
    "UnsupportedFormatError",
]


class MhvsrVs30Error(Exception):
    """Base class for every error this package raises deliberately."""


class ConfigError(MhvsrVs30Error):
    """A dataset configuration file is missing, malformed, or self-contradictory."""


class ContractViolationError(MhvsrVs30Error):
    """A value cannot be expressed as a valid manifest row."""


class ManifestValidationError(MhvsrVs30Error):
    """A built manifest failed its referential or physical-plausibility checks."""

    def __init__(self, message: str, failures: list[str] | None = None) -> None:
        super().__init__(message)
        self.failures = failures or []


class RecordingReadError(MhvsrVs30Error):
    """Base for every failure to turn a raw asset into a canonical record."""


class UnsupportedFormatError(RecordingReadError):
    """No reader is registered for the declared raw format."""


class MissingComponentError(RecordingReadError):
    """One or more of the vertical/north/east components is absent."""


class AmbiguousChannelError(RecordingReadError):
    """Channel codes cannot be mapped to Z/N/E without guessing."""


class SamplingRateMismatchError(RecordingReadError):
    """Components disagree on sampling rate, or contradict the source metadata."""


class NoCommonTimeWindowError(RecordingReadError):
    """The three components never overlap in time."""


class NonFiniteSampleError(RecordingReadError):
    """Samples contain NaN or infinity."""


class MalformedAsciiError(RecordingReadError):
    """An ASCII recording does not match its documented column layout."""


class MetadataConflictError(RecordingReadError):
    """File headers and source configuration make incompatible claims."""
