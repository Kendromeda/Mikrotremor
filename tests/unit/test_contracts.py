"""Phase 1 contract tests.

These pin the data contract itself: what a manifest row is allowed to say.
They are deliberately independent of any dataset on disk.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mhvsr_vs30.contracts import (
    LabelIndependence,
    MediaType,
    RawAsset,
    Recording,
    Site,
    Source,
    Vs30Label,
)

SHA = "a" * 64


def make_source(**overrides: object) -> Source:
    kwargs: dict[str, object] = {
        "source_id": "usgs_arra_2013",
        "title": "USGS Open-File Report 2013-1102",
        "doi": None,
        "license": "public domain (USGS)",
        "root_path": "datasets_global/source_05_usgs_arra_2013/raw/extracted",
    }
    kwargs.update(overrides)
    return Source(**kwargs)  # type: ignore[arg-type]


def make_site(**overrides: object) -> Site:
    kwargs: dict[str, object] = {
        "site_id": "usgs_arra_2013:CI.DRE",
        "source_id": "usgs_arra_2013",
        "latitude": 35.9,
        "longitude": -117.6,
        "elevation_m": 700.0,
        "country": "USA",
        "location_accuracy_m": None,
    }
    kwargs.update(overrides)
    return Site(**kwargs)  # type: ignore[arg-type]


def make_asset(**overrides: object) -> RawAsset:
    kwargs: dict[str, object] = {
        "asset_id": SHA,
        "source_id": "usgs_arra_2013",
        "relative_path": "CI.DRE/a.txt",
        "media_type": MediaType.AMBIENT_NOISE_WAVEFORM,
        "byte_size": 10,
        "sha256": SHA,
    }
    kwargs.update(overrides)
    return RawAsset(**kwargs)  # type: ignore[arg-type]


def make_recording(**overrides: object) -> Recording:
    kwargs: dict[str, object] = {
        "recording_id": SHA,
        "site_id": "usgs_arra_2013:CI.DRE",
        "asset_id": SHA,
        "raw_format": "usgs_ascii_3c",
        "sampling_rate_hz": 200.0,
        "start_time": datetime(2011, 2, 2, 19, 26, 37, tzinfo=UTC),
        "duration_s": 3600.0,
        "channel_mapping": {"Z": "column_1", "N": "column_2", "E": "column_3"},
        "units": "counts",
    }
    kwargs.update(overrides)
    return Recording(**kwargs)  # type: ignore[arg-type]


def make_label(**overrides: object) -> Vs30Label:
    kwargs: dict[str, object] = {
        "label_id": SHA,
        "site_id": "usgs_arra_2013:CI.DRE",
        "vs30_mps": 196.0,
        "vs30_std_mps": None,
        "label_method": "surface_wave_joint_inversion",
        "label_independence": LabelIndependence.INDEPENDENT_OF_HVSR,
        "reference": "USGS OFR 2013-1102",
        "reference_locator": "Table 3",
    }
    kwargs.update(overrides)
    return Vs30Label(**kwargs)  # type: ignore[arg-type]


# --- geographic bounds -------------------------------------------------------


@pytest.mark.parametrize("latitude", [-90.1, 90.1, 1000.0, float("nan")])
def test_invalid_latitude_rejected(latitude: float) -> None:
    with pytest.raises(ValidationError):
        make_site(latitude=latitude)


@pytest.mark.parametrize("longitude", [-180.1, 180.1, float("inf")])
def test_invalid_longitude_rejected(longitude: float) -> None:
    with pytest.raises(ValidationError):
        make_site(longitude=longitude)


def test_missing_coordinates_are_allowed() -> None:
    site = make_site(latitude=None, longitude=None)
    assert site.latitude is None


# --- physical bounds ---------------------------------------------------------


@pytest.mark.parametrize("vs30", [0.0, -1.0, -196.0])
def test_negative_vs30_rejected(vs30: float) -> None:
    with pytest.raises(ValidationError):
        make_label(vs30_mps=vs30)


def test_negative_vs30_std_rejected() -> None:
    with pytest.raises(ValidationError):
        make_label(vs30_std_mps=-5.0)


@pytest.mark.parametrize("rate", [0.0, -200.0])
def test_non_positive_sampling_rate_rejected(rate: float) -> None:
    with pytest.raises(ValidationError):
        make_recording(sampling_rate_hz=rate)


def test_negative_byte_size_rejected() -> None:
    with pytest.raises(ValidationError):
        make_asset(byte_size=-1)


# --- paths -------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_path",
    [
        "",
        "/absolute/unix/path.txt",
        "C:/absolute/windows/path.txt",
        "../escapes/root.txt",
        "nested/../../escapes.txt",
    ],
)
def test_relative_path_required(bad_path: str) -> None:
    with pytest.raises(ValidationError):
        make_asset(relative_path=bad_path)


def test_backslash_paths_are_normalised_to_posix() -> None:
    asset = make_asset(relative_path=r"CI.DRE\CI_DRE_Raw_Data\a.txt")
    assert asset.relative_path == "CI.DRE/CI_DRE_Raw_Data/a.txt"


def test_source_root_path_must_be_relative() -> None:
    with pytest.raises(ValidationError):
        make_source(root_path="/var/data")


# --- required identity -------------------------------------------------------


@pytest.mark.parametrize("model_factory", [make_source, make_site, make_asset])
def test_source_id_required(model_factory) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError):
        model_factory(source_id="")


def test_site_id_must_be_namespaced_by_source() -> None:
    with pytest.raises(ValidationError):
        make_site(site_id="CI.DRE")


def test_checksum_must_be_lowercase_hex_sha256() -> None:
    with pytest.raises(ValidationError):
        make_asset(sha256="not-a-checksum")
    with pytest.raises(ValidationError):
        make_asset(sha256=SHA.upper())


def test_label_requires_provenance() -> None:
    with pytest.raises(ValidationError):
        make_label(reference="")


def test_label_independence_is_a_closed_vocabulary() -> None:
    with pytest.raises(ValidationError):
        make_label(label_independence="probably_fine")


def test_media_type_is_a_closed_vocabulary() -> None:
    with pytest.raises(ValidationError):
        make_asset(media_type="spreadsheet_ish")


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_label(confidence=0.9)


# --- immutability ------------------------------------------------------------


@pytest.mark.parametrize(
    "instance",
    [make_source(), make_site(), make_asset(), make_recording(), make_label()],
)
def test_models_are_immutable(instance: object) -> None:
    field = next(iter(type(instance).model_fields))  # type: ignore[attr-defined]
    with pytest.raises(ValidationError):
        setattr(instance, field, "mutated")


def test_channel_mapping_cannot_be_mutated_in_place() -> None:
    recording = make_recording()
    with pytest.raises(TypeError):
        recording.channel_mapping["Z"] = "column_9"  # type: ignore[index]


def test_channel_mapping_is_decoupled_from_caller_dict() -> None:
    mapping = {"Z": "column_1", "N": "column_2", "E": "column_3"}
    recording = make_recording(channel_mapping=mapping)
    mapping["Z"] = "column_9"
    assert recording.channel_mapping["Z"] == "column_1"


def test_models_round_trip_through_plain_dicts() -> None:
    recording = make_recording()
    restored = Recording(**recording.to_row())
    assert restored == recording
