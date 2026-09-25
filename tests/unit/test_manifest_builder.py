"""Phase 2 tests: discovery, attribution and manifest construction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mhvsr_vs30.contracts import MediaType
from mhvsr_vs30.exceptions import ConfigError, ManifestValidationError
from mhvsr_vs30.hashing import sha256_file
from mhvsr_vs30.manifest.builder import build_manifest, discover_assets
from mhvsr_vs30.manifest.schemas import load_source_config
from mhvsr_vs30.manifest.validation import validate_manifest


def _tree_state(root: Path) -> dict[str, tuple[int, int, str]]:
    return {
        str(path.relative_to(root)): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            sha256_file(path),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_discovery_uses_relative_paths(workspace: Path, demo_config: Path) -> None:
    config = load_source_config(demo_config)
    assets = discover_assets(config, workspace)

    assert assets
    for asset in assets:
        assert not Path(asset.relative_path).is_absolute()
        assert ".." not in asset.relative_path
        assert str(workspace) not in asset.relative_path
        assert (workspace / config.root / asset.relative_path).is_file()


def test_discovery_does_not_modify_raw_files(workspace: Path, demo_config: Path) -> None:
    config = load_source_config(demo_config)
    root = workspace / config.root
    before = _tree_state(root)

    build_manifest(load_source_config(demo_config), workspace)

    assert _tree_state(root) == before


def test_asset_checksum_matches_file(workspace: Path, demo_config: Path) -> None:
    config = load_source_config(demo_config)
    root = workspace / config.root
    for asset in discover_assets(config, workspace):
        target = root / asset.relative_path
        assert asset.sha256 == sha256_file(target)
        assert asset.byte_size == target.stat().st_size


def test_checksum_cache_is_reused_but_invalidated_by_content(
    workspace: Path, demo_config: Path, tmp_path: Path
) -> None:
    config = load_source_config(demo_config)
    cache = tmp_path / "checksums.json"

    first = {
        a.relative_path: a.sha256 for a in discover_assets(config, workspace, cache_path=cache)
    }
    assert cache.exists()
    second = {
        a.relative_path: a.sha256 for a in discover_assets(config, workspace, cache_path=cache)
    }
    assert first == second

    victim = next(iter(first))
    (workspace / config.root / victim).write_bytes(b"different bytes entirely")
    third = {
        a.relative_path: a.sha256 for a in discover_assets(config, workspace, cache_path=cache)
    }
    assert third[victim] != first[victim]


def test_site_id_is_stable(workspace: Path, demo_config: Path) -> None:
    first = build_manifest(load_source_config(demo_config), workspace)
    second = build_manifest(load_source_config(demo_config), workspace)

    assert [s.site_id for s in first.sites] == [s.site_id for s in second.sites]
    assert [s.site_id for s in first.sites] == ["demo_source:XX.AAA", "demo_source:XX.BBB"]


def test_ids_are_stable_across_rebuilds(workspace: Path, demo_config: Path) -> None:
    first = build_manifest(load_source_config(demo_config), workspace)
    second = build_manifest(load_source_config(demo_config), workspace)

    assert [a.asset_id for a in first.assets] == [a.asset_id for a in second.assets]
    assert [r.recording_id for r in first.recordings] == [r.recording_id for r in second.recordings]
    assert [label.label_id for label in first.labels] == [label.label_id for label in second.labels]


def test_unknown_extension_is_preserved(workspace: Path, demo_config: Path) -> None:
    """Files nobody classified still get a row and a checksum."""
    manifest = build_manifest(load_source_config(demo_config), workspace)
    unknown = [a for a in manifest.assets if a.media_type is MediaType.UNKNOWN]

    assert [Path(a.relative_path).name for a in unknown] == ["Thumbs.db", "Thumbs.db"]
    assert all(len(a.sha256) == 64 for a in unknown)


def test_every_discovered_file_becomes_exactly_one_asset(
    workspace: Path, demo_config: Path
) -> None:
    config = load_source_config(demo_config)
    on_disk = {
        str(p.relative_to(workspace / config.root)).replace("\\", "/")
        for p in (workspace / config.root).rglob("*")
        if p.is_file()
    }
    manifest = build_manifest(config, workspace)

    assert {a.relative_path for a in manifest.assets} == on_disk
    assert len({a.asset_id for a in manifest.assets}) == len(manifest.assets)


def test_recording_references_existing_asset(workspace: Path, demo_config: Path) -> None:
    manifest = build_manifest(load_source_config(demo_config), workspace)
    asset_ids = {a.asset_id for a in manifest.assets}
    site_ids = {s.site_id for s in manifest.sites}

    assert manifest.recordings
    for recording in manifest.recordings:
        assert recording.asset_id in asset_ids
        assert recording.site_id in site_ids


def test_recordings_carry_documented_acquisition_metadata(
    workspace: Path, demo_config: Path
) -> None:
    manifest = build_manifest(load_source_config(demo_config), workspace)
    recording = manifest.recordings[0]

    assert recording.raw_format == "usgs_ascii_3c"
    assert recording.sampling_rate_hz == 200.0
    assert recording.units == "counts"
    assert dict(recording.channel_mapping) == {
        "Z": "column_1",
        "N": "column_2",
        "E": "column_3",
    }


def test_only_ambient_noise_assets_become_recordings(workspace: Path, demo_config: Path) -> None:
    manifest = build_manifest(load_source_config(demo_config), workspace)
    by_id = {a.asset_id: a for a in manifest.assets}

    assert all(
        by_id[r.asset_id].media_type is MediaType.AMBIENT_NOISE_WAVEFORM
        for r in manifest.recordings
    )
    survey = [a for a in manifest.assets if a.media_type is MediaType.SURFACE_WAVE_SURVEY]
    assert survey
    assert not {a.asset_id for a in survey} & {r.asset_id for r in manifest.recordings}


def test_label_requires_provenance(workspace: Path, demo_config: Path) -> None:
    text = demo_config.read_text(encoding="utf-8")
    demo_config.write_text(
        text.replace("reference: Demo report", 'reference: ""'),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_source_config(demo_config)


def test_label_is_not_inferred_from_file_names(workspace: Path, demo_config: Path) -> None:
    """A site whose config carries no label gets none, however many files it has."""
    text = demo_config.read_text(encoding="utf-8")
    head, _, tail = text.partition("  - site_code: XX.BBB")
    unlabelled = head + "  - site_code: XX.BBB" + tail.split("    label:")[0]
    demo_config.write_text(unlabelled, encoding="utf-8")

    with pytest.raises(ManifestValidationError) as excinfo:
        build_manifest(load_source_config(demo_config), workspace)

    assert "demo_source:XX.BBB has no Vs30 label" in str(excinfo.value)


def test_unlabelled_site_can_build_when_labels_are_explicitly_optional(
    workspace: Path, demo_config: Path
) -> None:
    """QC/pretraining corpora may contain recordings before Vs30 curation."""
    text = demo_config.read_text(encoding="utf-8")
    head, _, tail = text.partition("  - site_code: XX.BBB")
    unlabelled = head + "  - site_code: XX.BBB" + tail.split("    label:")[0]
    demo_config.write_text(
        unlabelled.replace("expectations:\n", "expectations:\n  require_labels: false\n"),
        encoding="utf-8",
    )

    manifest = build_manifest(load_source_config(demo_config), workspace)

    assert [label.site_id for label in manifest.labels] == ["demo_source:XX.AAA"]
    assert {recording.site_id for recording in manifest.recordings} == {
        "demo_source:XX.AAA",
        "demo_source:XX.BBB",
    }


def test_unconfigured_site_is_reported_not_invented(workspace: Path, demo_config: Path) -> None:
    """Files belonging to no configured site stop the build instead of being absorbed."""
    text = demo_config.read_text(encoding="utf-8")
    start = text.index("  - site_code: XX.BBB")
    demo_config.write_text(
        text[:start]
        .replace("site_count: 2", "site_count: 1")
        .replace("recording_count: 4", "recording_count: 2"),
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError) as excinfo:
        build_manifest(load_source_config(demo_config), workspace)

    assert "claimed by 0 configured sites" in str(excinfo.value)


def test_config_rejects_unattributable_site(workspace: Path, demo_config: Path) -> None:
    text = demo_config.read_text(encoding="utf-8")
    demo_config.write_text(text.replace("site_code: XX.AAA", "site_code: XX.ZZZ"), encoding="utf-8")

    with pytest.raises(ManifestValidationError):
        build_manifest(load_source_config(demo_config), workspace)


def test_expectation_mismatch_fails_the_build(workspace: Path, demo_config: Path) -> None:
    text = demo_config.read_text(encoding="utf-8")
    demo_config.write_text(
        text.replace("recording_count: 4", "recording_count: 5"), encoding="utf-8"
    )

    with pytest.raises(ManifestValidationError) as excinfo:
        build_manifest(load_source_config(demo_config), workspace)
    assert "recording_count" in str(excinfo.value)


def test_vs30_outside_configured_range_fails_validation(workspace: Path, demo_config: Path) -> None:
    text = demo_config.read_text(encoding="utf-8")
    demo_config.write_text(text.replace("vs30_mps: 800", "vs30_mps: 9000"), encoding="utf-8")

    with pytest.raises(ManifestValidationError):
        build_manifest(load_source_config(demo_config), workspace)


def test_validation_detects_dangling_reference(workspace: Path, demo_config: Path) -> None:
    config = load_source_config(demo_config)
    manifest = build_manifest(config, workspace)
    broken = manifest.replace(sites=manifest.sites[:1])

    failures = validate_manifest(broken, config)
    assert any("site_id" in failure for failure in failures)


def test_missing_config_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_source_config(tmp_path / "nope.yaml")


def test_missing_root_is_a_config_error(workspace: Path, demo_config: Path) -> None:
    config = load_source_config(demo_config)
    with pytest.raises(ConfigError):
        discover_assets(config, workspace / "elsewhere")


def test_summary_reports_counts_and_vs30_distribution(workspace: Path, demo_config: Path) -> None:
    manifest = build_manifest(load_source_config(demo_config), workspace)
    summary = manifest.summary()

    assert summary["site_count"] == 2
    assert summary["recording_count"] == 4
    assert summary["label_count"] == 2
    assert summary["asset_count"] == len(manifest.assets)
    assert summary["vs30_mps"]["min"] == 200.0
    assert summary["vs30_mps"]["max"] == 800.0
    assert summary["assets_by_media_type"]["ambient_noise_waveform"] == 4
    json.dumps(summary)  # must stay serialisable
