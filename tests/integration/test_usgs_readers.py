"""Phase 3 integration: parse every real USGS ARRA recording."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mhvsr_vs30.io.base import ThreeComponentRecord
from mhvsr_vs30.io.registry import get_reader
from mhvsr_vs30.io.resolve import resolve_request
from mhvsr_vs30.manifest.builder import build_manifest
from mhvsr_vs30.manifest.schemas import Manifest, SourceConfig, load_source_config

pytestmark = pytest.mark.requires_real_data

CONFIG_PATH = Path("configs/datasets/usgs_arra.yaml")
CACHE = Path("artifacts/cache/checksums.json")

# Range reported for this subset in datasets_global/README.md.
DOCUMENTED_DURATION_S = (1451.0, 3600.0)


@pytest.fixture(scope="module")
def usgs() -> tuple[Manifest, SourceConfig]:
    if not CONFIG_PATH.is_file():
        pytest.skip("USGS config is absent")
    config = load_source_config(CONFIG_PATH)
    if not (Path(".") / config.root).is_dir():
        pytest.skip("USGS ARRA dataset is not present in this workspace")
    return build_manifest(config, ".", cache_path=CACHE), config


@pytest.fixture(scope="module")
def records(usgs: tuple[Manifest, SourceConfig]) -> list[ThreeComponentRecord]:
    manifest, config = usgs
    reader = get_reader("usgs_ascii_3c")
    return [
        reader.read(resolve_request(manifest, recording, config, "."))
        for recording in manifest.recordings
    ]


def test_all_eighteen_recordings_parse(records: list[ThreeComponentRecord]) -> None:
    assert len(records) == 18
    assert all(isinstance(record, ThreeComponentRecord) for record in records)


def test_components_have_equal_length(records: list[ThreeComponentRecord]) -> None:
    for record in records:
        assert len(record.z) == len(record.north) == len(record.east) == record.n_samples
        assert record.n_samples > 0


def test_sampling_rate_matches_the_documented_two_hundred_hz(
    records: list[ThreeComponentRecord],
) -> None:
    assert {record.sampling_rate_hz for record in records} == {200.0}
    assert {record.units for record in records} == {"counts"}


def test_durations_fall_inside_the_documented_range(
    records: list[ThreeComponentRecord],
) -> None:
    low, high = DOCUMENTED_DURATION_S
    durations = [record.duration_s for record in records]
    assert min(durations) >= low - 1.0
    assert max(durations) <= high + 1.0


def test_samples_are_finite_and_read_only(records: list[ThreeComponentRecord]) -> None:
    record = records[0]
    assert np.isfinite(record.z).all()
    with pytest.raises(ValueError):
        record.north[0] = 0.0


def test_column_order_is_vertical_north_east(records: list[ThreeComponentRecord]) -> None:
    for record in records:
        assert record.source_channels == {"Z": "column_1", "N": "column_2", "E": "column_3"}


def test_parsing_is_deterministic(usgs: tuple[Manifest, SourceConfig]) -> None:
    manifest, config = usgs
    recording = manifest.recordings[0]
    reader = get_reader("usgs_ascii_3c")

    first = reader.read(resolve_request(manifest, recording, config, "."))
    second = reader.read(resolve_request(manifest, recording, config, "."))

    assert np.array_equal(first.z, second.z)
    assert first.structural_qc() == second.structural_qc()


def test_parser_writes_nothing_into_the_source_tree(
    usgs: tuple[Manifest, SourceConfig], records: list[ThreeComponentRecord]
) -> None:
    manifest, config = usgs
    root = Path(".") / config.root
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}

    assert on_disk == {asset.relative_path for asset in manifest.assets}
    assert not list(root.rglob("*_updated.*"))
