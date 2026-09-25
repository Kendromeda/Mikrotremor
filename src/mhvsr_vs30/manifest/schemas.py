"""Dataset configuration and the in-memory manifest container.

A configuration file is a set of claims about a published dataset that a human
checked against the source documentation. It is deliberately not inferred from
the file tree: the tree says what exists, the config says what it means.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from mhvsr_vs30.contracts import (
    FormatHint,
    LabelIndependence,
    MediaType,
    RawAsset,
    Recording,
    RelativePath,
    RepairAction,
    Site,
    Source,
    Vs30Label,
)
from mhvsr_vs30.exceptions import ConfigError
from mhvsr_vs30.hashing import sha256_file

__all__ = [
    "AssetRule",
    "Expectations",
    "LabelConfig",
    "Manifest",
    "RecordingConfig",
    "RepairAction",
    "RepairRule",
    "SiteConfig",
    "SourceConfig",
    "load_source_config",
]

TABLE_NAMES = ("sources", "sites", "assets", "recordings", "labels")
CSV_TABLE_NAMES = ("sources", "sites", "assets", "recordings", "labels")
MANIFEST_VERSION = "1.0"

_COMPONENT_NAMES = ("vertical", "north", "east")
_COMPONENT_TO_CODE = {"vertical": "Z", "north": "N", "east": "E"}


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LabelConfig(_Config):
    """A Vs30 claim transcribed from a source document."""

    vs30_mps: float = Field(gt=0.0, allow_inf_nan=False)
    vs30_std_mps: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    label_method: str = Field(min_length=1)
    label_independence: LabelIndependence
    reference: str = Field(min_length=1)
    reference_locator: str | None = None


class SiteConfig(_Config):
    """One physical location, plus how to recognise its files on disk."""

    site_code: str = Field(min_length=1)
    path_glob: str | None = None
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0, allow_inf_nan=False)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0, allow_inf_nan=False)
    elevation_m: float | None = Field(default=None, allow_inf_nan=False)
    location_accuracy_m: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    label: LabelConfig | None = None

    def effective_path_glob(self) -> str:
        return self.path_glob or f"{self.site_code}/*"


class RecordingConfig(_Config):
    """How this source stores the recordings we intend to parse."""

    format: str = Field(min_length=1)
    component_order: tuple[str, str, str] = _COMPONENT_NAMES
    sampling_rate_hz: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    units: str | None = None
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    start_time_from_filename: str | None = None

    # Orientation for channel codes that carry none of their own, such as
    # BH1/BH2 or the unnamed traces in a SEG-2 file. Absent means the reader
    # refuses to guess.
    channel_codes: dict[str, str] | None = None
    component_traces: dict[str, int] | None = None

    # Column roles for processed H/V curves, e.g. [frequency, mean, std] for an
    # .asc export or [frequency, mean, min, max] for Geopsy .hv.
    curve_columns: tuple[str, ...] = ()

    # How components that live in separate files are grouped into one recording.
    bundle_group_by: str | None = None

    @field_validator("bundle_group_by")
    @classmethod
    def _known_bundle_rule(cls, value: str | None) -> str | None:
        if value is not None and value != "directory":
            raise ValueError("bundle_group_by currently supports only: directory")
        return value

    @field_validator("component_order")
    @classmethod
    def _known_components(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if sorted(value) != sorted(_COMPONENT_NAMES):
            raise ValueError(f"component_order must be a permutation of {_COMPONENT_NAMES}")
        return value

    def channel_mapping(self) -> dict[str, str]:
        """Map canonical component codes to the source's own column positions."""
        return {
            _COMPONENT_TO_CODE[name]: f"column_{index + 1}"
            for index, name in enumerate(self.component_order)
        }


class AssetRule(_Config):
    """A path pattern that asserts what a file is.

    Rules beat extensions, because extensions lie: in USGS ARRA the MASW
    lowercase .dat files are SEG-2 while the SASW uppercase .DAT files are
    HP SDF, as stated in the release's own FILE FORMATS document.
    """

    pattern: str = Field(min_length=1)
    media_type: MediaType
    format_hint: FormatHint | None = None
    note: str | None = None


class RepairRule(_Config):
    """Permission to repair exactly one file, pinned to its exact bytes.

    A repair is an admission that the published data is defective, so it is
    granted per file and per checksum rather than as a per-source switch. If the
    file is ever replaced, the checksum stops matching and the pipeline refuses
    to reuse an audit that was written about different bytes.
    """

    relative_path: RelativePath
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    action: RepairAction
    reason: str = Field(min_length=1)
    expected_samples_dropped: int | None = Field(default=None, ge=0)


class Expectations(_Config):
    """Counts a correct build must reproduce. A mismatch fails the build."""

    site_count: int | None = Field(default=None, ge=0)
    asset_count: int | None = Field(default=None, ge=0)
    recording_count: int | None = Field(default=None, ge=0)
    recordings_per_site: int | None = Field(default=None, ge=0)
    # Label curation can lag waveform curation. Keep the historical strict
    # behaviour unless a source explicitly declares itself QC/pretraining-only.
    require_labels: bool = True


class SourceConfig(_Config):
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    country: str = Field(min_length=1)
    root: str = Field(min_length=1)
    doi: str | None = None
    license: str | None = None
    source_url: str | None = None
    recording: RecordingConfig
    asset_rules: tuple[AssetRule, ...] = ()
    repairs: tuple[RepairRule, ...] = ()
    exclude: tuple[str, ...] = ()
    vs30_range_mps: tuple[float, float] = (50.0, 3000.0)
    expectations: Expectations = Expectations()
    sites: tuple[SiteConfig, ...] = ()
    config_hash: str = ""
    config_path: str | None = None

    @field_validator("vs30_range_mps")
    @classmethod
    def _ordered_range(cls, value: tuple[float, float]) -> tuple[float, float]:
        low, high = value
        if not 0 < low < high:
            raise ValueError("vs30_range_mps must be an increasing, positive pair")
        return value

    @field_validator("sites")
    @classmethod
    def _unique_site_codes(cls, value: tuple[SiteConfig, ...]) -> tuple[SiteConfig, ...]:
        codes = [site.site_code for site in value]
        duplicates = sorted({code for code in codes if codes.count(code) > 1})
        if duplicates:
            raise ValueError(f"duplicate site_code entries: {duplicates}")
        return value

    def to_source(self) -> Source:
        return Source(
            source_id=self.source_id,
            title=self.title,
            doi=self.doi,
            license=self.license,
            root_path=self.root,
            source_url=self.source_url,
        )


def load_source_config(path: Path | str) -> SourceConfig:
    """Read and validate one dataset configuration.

    The file's own checksum travels with the config so a manifest snapshot can
    record exactly which claims produced it.
    """
    target = Path(path)
    if not target.is_file():
        raise ConfigError(f"dataset config not found: {target}")

    try:
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"{target} is not valid YAML: {error}") from error

    if not isinstance(payload, dict):
        raise ConfigError(f"{target} must contain a YAML mapping at the top level")

    payload["config_hash"] = sha256_file(target)
    payload["config_path"] = target.as_posix()

    try:
        return SourceConfig(**payload)
    except ValidationError as error:
        raise ConfigError(f"{target} failed validation:\n{error}") from error


@dataclasses.dataclass(frozen=True, slots=True)
class Manifest:
    """The five manifest tables, in memory."""

    source: Source
    sites: tuple[Site, ...]
    assets: tuple[RawAsset, ...]
    recordings: tuple[Recording, ...]
    labels: tuple[Vs30Label, ...]

    def replace(self, **changes: Any) -> Manifest:
        return dataclasses.replace(self, **changes)

    def table(self, name: str) -> Sequence[Any]:
        if name == "sources":
            return (self.source,)
        if name in {"sites", "assets", "recordings", "labels"}:
            return getattr(self, name)  # type: ignore[no-any-return]
        raise KeyError(f"unknown manifest table: {name}")

    def summary(self) -> dict[str, Any]:
        """Counts and distributions an analyst should read before trusting a build."""
        per_site = Counter(recording.site_id for recording in self.recordings)
        vs30_values = sorted(label.vs30_mps for label in self.labels)

        summary: dict[str, Any] = {
            "source_id": self.source.source_id,
            "source_count": 1,
            "site_count": len(self.sites),
            "asset_count": len(self.assets),
            "recording_count": len(self.recordings),
            "label_count": len(self.labels),
            "assets_by_media_type": dict(
                sorted(Counter(asset.media_type.value for asset in self.assets).items())
            ),
            "recordings_per_site": {
                site.site_id: per_site.get(site.site_id, 0) for site in self.sites
            },
            "labels_by_independence": dict(
                sorted(Counter(label.label_independence.value for label in self.labels).items())
            ),
            "vs30_mps": None,
            "total_raw_bytes": sum(asset.byte_size for asset in self.assets),
        }
        if vs30_values:
            summary["vs30_mps"] = {
                "count": len(vs30_values),
                "min": vs30_values[0],
                "max": vs30_values[-1],
                "mean": sum(vs30_values) / len(vs30_values),
                "values": vs30_values,
            }
        return summary
