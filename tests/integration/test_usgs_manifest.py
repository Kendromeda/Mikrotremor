"""Phase 2 integration tests against the real USGS ARRA subset.

These assert the counts the release itself documents. If the tree on disk
changes, these tests are supposed to fail loudly rather than absorb the change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mhvsr_vs30.contracts import MediaType
from mhvsr_vs30.manifest.builder import build_manifest
from mhvsr_vs30.manifest.schemas import Manifest, load_source_config
from mhvsr_vs30.manifest.store import read_manifest, write_manifest

pytestmark = pytest.mark.requires_real_data

EXPECTED_SITES = ["usgs_arra_2013:CI.CCC", "usgs_arra_2013:CI.CLC", "usgs_arra_2013:CI.DRE"]
EXPECTED_VS30 = {
    "usgs_arra_2013:CI.DRE": 196.0,
    "usgs_arra_2013:CI.CCC": 432.0,
    "usgs_arra_2013:CI.CLC": 1464.0,
}
CACHE = Path("artifacts/cache/checksums.json")


@pytest.fixture(scope="module")
def usgs_manifest() -> Manifest:
    config_path = Path("configs/datasets/usgs_arra.yaml")
    if not config_path.is_file():
        pytest.skip("USGS config is absent")
    config = load_source_config(config_path)
    if not (Path(".") / config.root).is_dir():
        pytest.skip("USGS ARRA dataset is not present in this workspace")
    return build_manifest(config, ".", cache_path=CACHE)


def test_build_usgs_manifest_finds_three_sites(usgs_manifest: Manifest) -> None:
    assert sorted(site.site_id for site in usgs_manifest.sites) == EXPECTED_SITES


def test_build_usgs_manifest_finds_18_recordings(usgs_manifest: Manifest) -> None:
    assert len(usgs_manifest.recordings) == 18
    assert all(r.raw_format == "usgs_ascii_3c" for r in usgs_manifest.recordings)
    assert all(r.sampling_rate_hz == 200.0 for r in usgs_manifest.recordings)


def test_each_site_has_six_recordings(usgs_manifest: Manifest) -> None:
    per_site = usgs_manifest.summary()["recordings_per_site"]
    assert per_site == dict.fromkeys(EXPECTED_SITES, 6)


def test_surveys_are_recorded_but_not_treated_as_recordings(usgs_manifest: Manifest) -> None:
    """The 127 MASW/SASW files are tracked, yet none of them is an mHVSR input."""
    surveys = [a for a in usgs_manifest.assets if a.media_type is MediaType.SURFACE_WAVE_SURVEY]
    assert len(surveys) == 127
    assert not {a.asset_id for a in surveys} & {r.asset_id for r in usgs_manifest.recordings}


def test_labels_match_the_published_table(usgs_manifest: Manifest) -> None:
    labels = {label.site_id: label for label in usgs_manifest.labels}
    assert {site: label.vs30_mps for site, label in labels.items()} == EXPECTED_VS30
    for label in labels.values():
        assert label.reference == "USGS Open-File Report 2013-1102"
        assert label.reference_locator == "Table 3"
        assert label.label_independence.value == "independent_of_hvsr"


def test_every_file_in_the_tree_has_a_checksum(usgs_manifest: Manifest) -> None:
    root = Path("datasets_global/source_05_usgs_arra_2013/raw/extracted")
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}

    assert {a.relative_path for a in usgs_manifest.assets} == on_disk
    assert all(len(a.sha256) == 64 and a.byte_size > 0 for a in usgs_manifest.assets)


def test_build_does_not_modify_raw_files(usgs_manifest: Manifest) -> None:
    root = Path("datasets_global/source_05_usgs_arra_2013/raw/extracted")
    sizes = {a.relative_path: a.byte_size for a in usgs_manifest.assets}
    for relative_path, byte_size in sizes.items():
        assert (root / relative_path).stat().st_size == byte_size
    assert not list(root.rglob("*_updated.*"))


def test_manifest_rebuild_is_identical(tmp_path: Path) -> None:
    """Two builds of unchanged data differ only in the snapshot timestamp."""
    config = load_source_config(Path("configs/datasets/usgs_arra.yaml"))
    if not (Path(".") / config.root).is_dir():
        pytest.skip("USGS ARRA dataset is not present in this workspace")

    first = write_manifest(build_manifest(config, ".", cache_path=CACHE), tmp_path / "a", config)
    second = write_manifest(build_manifest(config, ".", cache_path=CACHE), tmp_path / "b", config)

    volatile = {"created_at"}
    for name in sorted(p.name for p in first.iterdir()):
        left = (first / name).read_bytes()
        right = (second / name).read_bytes()
        if name == "snapshot.json":
            left_json = {k: v for k, v in json.loads(left).items() if k not in volatile}
            right_json = {k: v for k, v in json.loads(right).items() if k not in volatile}
            assert left_json == right_json
        else:
            assert left == right, f"{name} is not reproducible"


def test_manifest_round_trips_through_parquet(usgs_manifest: Manifest, tmp_path: Path) -> None:
    config = load_source_config(Path("configs/datasets/usgs_arra.yaml"))
    write_manifest(usgs_manifest, tmp_path / "m", config)
    restored = read_manifest(tmp_path / "m")

    assert restored.source == usgs_manifest.source
    assert sorted(restored.sites, key=lambda s: s.site_id) == sorted(
        usgs_manifest.sites, key=lambda s: s.site_id
    )
    assert sorted(restored.labels, key=lambda label: label.label_id) == sorted(
        usgs_manifest.labels, key=lambda label: label.label_id
    )
    assert len(restored.recordings) == len(usgs_manifest.recordings)
    assert len(restored.assets) == len(usgs_manifest.assets)
