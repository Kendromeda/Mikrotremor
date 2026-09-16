"""Immutable data contracts for the recording-level manifest.

Every row that ends up in a manifest table is validated through one of these
models first. The models are frozen: a manifest is a record of what was
observed on disk, not a mutable working object.

Nothing here reads files or computes anything about a recording's contents;
that belongs to :mod:`mhvsr_vs30.manifest` and :mod:`mhvsr_vs30.io`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import Annotated, Any, Self

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer

__all__ = [
    "FormatHint",
    "LabelIndependence",
    "ManifestRecord",
    "MediaType",
    "RawAsset",
    "Recording",
    "RelativePath",
    "RepairAction",
    "Site",
    "Source",
    "Vs30Label",
]

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\/]")
_SOURCE_ID = r"^[a-z0-9][a-z0-9_]*$"
_SITE_ID = r"^[a-z0-9][a-z0-9_]*:\S.*$"
_SHA256 = r"^[0-9a-f]{64}$"


# --- vocabularies ------------------------------------------------------------


class MediaType(StrEnum):
    """What a raw file *is*, independent of its extension.

    Assigning a media type is a claim about provenance, so ``UNKNOWN`` is a
    legitimate answer and is preferred over a guess.
    """

    AMBIENT_NOISE_WAVEFORM = "ambient_noise_waveform"
    PROCESSED_HVSR_CURVE = "processed_hvsr_curve"
    SURFACE_WAVE_SURVEY = "surface_wave_survey"
    SITE_METADATA = "site_metadata"
    VS_PROFILE = "vs_profile"
    SITE_REPORT = "site_report"
    FIELD_NOTE = "field_note"
    IMAGE = "image"
    ARCHIVE = "archive"
    UNKNOWN = "unknown"


class FormatHint(StrEnum):
    """The wire format of a raw file, as identified by the release itself.

    Separate from :class:`MediaType`, which says what a file *means*. A SEG-2
    MASW gather and an HP SDF SASW sweep are both surface-wave surveys, but
    nothing can read one with the other reader.
    """

    ASCII_3C = "ascii_3c"
    MINISEED = "miniseed"
    SAC = "sac"
    GCF = "gcf"
    SEG2 = "seg2"
    HP_SDF = "hp_sdf"


class RepairAction(StrEnum):
    """A defect in published data that this pipeline is allowed to work around.

    The vocabulary is closed on purpose: every entry is a documented decision
    about real bytes, not a general-purpose cleaning step.
    """

    DROP_INCOMPLETE_FINAL_ROW = "drop_incomplete_final_row"


class LabelIndependence(StrEnum):
    """How independent a Vs30 value is from the HVSR input it will be paired with.

    Mixing these in one training target without weighting leaks the label.
    """

    INDEPENDENT_OF_HVSR = "independent_of_hvsr"
    HVSR_DERIVED = "hvsr_derived"
    PROXY = "proxy"
    UNKNOWN = "unknown"


# --- annotated scalars -------------------------------------------------------


def _normalise_relative_path(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip().replace("\\", "/")
    if not text:
        raise ValueError("relative path must not be empty")
    if text.startswith("/") or _WINDOWS_DRIVE.match(value.strip()):
        raise ValueError(f"path must be relative to the source root, got {value!r}")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts:
        raise ValueError(f"path resolves to nothing, got {value!r}")
    if ".." in parts:
        raise ValueError(f"path must not escape the source root, got {value!r}")
    return "/".join(parts)


def _coerce_mapping(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _freeze_mapping(value: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(value))


RelativePath = Annotated[str, BeforeValidator(_normalise_relative_path), Field(min_length=1)]
Sha256Hex = Annotated[str, Field(pattern=_SHA256)]
SourceId = Annotated[str, Field(pattern=_SOURCE_ID)]
SiteId = Annotated[str, Field(pattern=_SITE_ID)]
NonEmptyStr = Annotated[str, Field(min_length=1)]
Latitude = Annotated[float, Field(ge=-90.0, le=90.0, allow_inf_nan=False)]
Longitude = Annotated[float, Field(ge=-180.0, le=180.0, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
ChannelMapping = Annotated[
    Mapping[str, str],
    BeforeValidator(_coerce_mapping),
    AfterValidator(_freeze_mapping),
    PlainSerializer(lambda mapping: dict(mapping), return_type=dict),
]


# --- models ------------------------------------------------------------------


class ManifestRecord(BaseModel):
    """Base for every manifest row: frozen, closed, and tabular-serialisable."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    def to_row(self) -> dict[str, Any]:
        """Render as a flat dict that Parquet and CSV can both hold.

        Mappings become sorted JSON strings so the same bytes come out of every
        rebuild; :meth:`from_row` accepts them back unchanged.
        """
        row: dict[str, Any] = {}
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, Enum):
                value = value.value
            elif isinstance(value, Mapping):
                value = json.dumps(dict(value), sort_keys=True)
            row[name] = value
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Self:
        """Rebuild from a tabular row, treating pandas' missing values as ``None``."""
        cleaned = {
            key: None if _is_missing(value) else value
            for key, value in row.items()
            if key in cls.model_fields
        }
        return cls(**cleaned)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, float) and value != value


class Source(ManifestRecord):
    """One published dataset release."""

    source_id: SourceId
    title: NonEmptyStr
    doi: str | None = None
    license: str | None = None
    root_path: RelativePath
    source_url: str | None = None


class Site(ManifestRecord):
    """One physical measurement location within a source.

    Sites are never merged across sources on proximity alone; see
    ``docs`` in the manifest builder for why.
    """

    site_id: SiteId
    source_id: SourceId
    latitude: Latitude | None = None
    longitude: Longitude | None = None
    elevation_m: FiniteFloat | None = None
    country: NonEmptyStr
    location_accuracy_m: NonNegativeFloat | None = None
    site_code: str | None = None


class RawAsset(ManifestRecord):
    """Any file under a source root, recording or not.

    PDFs, spreadsheets, photographs and field notes are assets too: they carry
    the provenance that makes a label defensible, so they get checksums.
    """

    asset_id: Sha256Hex
    source_id: SourceId
    relative_path: RelativePath
    media_type: MediaType
    byte_size: int = Field(ge=0)
    sha256: Sha256Hex
    format_hint: FormatHint | None = None


class Recording(ManifestRecord):
    """An asset that has been claimed as a parseable three-component recording."""

    recording_id: Sha256Hex
    site_id: SiteId
    asset_id: Sha256Hex
    raw_format: NonEmptyStr
    sampling_rate_hz: PositiveFloat | None = None
    start_time: datetime | None = None
    duration_s: PositiveFloat | None = None
    channel_mapping: ChannelMapping = MappingProxyType({})
    units: str | None = None
    segment: str | None = None


class Vs30Label(ManifestRecord):
    """A Vs30 value together with the provenance that justifies it.

    ``reference`` is required on purpose: a label with no citable source is a
    guess, and this project does not train on guesses.
    """

    label_id: Sha256Hex
    site_id: SiteId
    vs30_mps: PositiveFloat
    vs30_std_mps: NonNegativeFloat | None = None
    label_method: NonEmptyStr
    label_independence: LabelIndependence
    reference: NonEmptyStr
    reference_locator: str | None = None
