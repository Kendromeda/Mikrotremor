"""Phase 3.5 tests: repair allowlisting, format hints, exports and provenance."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mhvsr_vs30.contracts import FormatHint, RepairAction
from mhvsr_vs30.exceptions import RecordingReadError
from mhvsr_vs30.io.resolve import resolve_request
from mhvsr_vs30.manifest.builder import build_manifest
from mhvsr_vs30.manifest.schemas import load_source_config
from mhvsr_vs30.manifest.store import read_manifest, write_manifest
from mhvsr_vs30.provenance import environment_provenance, source_tree_hash


def add_repair(config_path: Path, relative_path: str, sha256: str) -> None:
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + "\nrepairs:\n"
        + f'  - relative_path: "{relative_path}"\n'
        + f'    sha256: "{sha256}"\n'
        + "    action: drop_incomplete_final_row\n"
        + "    reason: synthetic fixture cut mid-sample\n",
        encoding="utf-8",
    )


def test_format_hint_reaches_the_asset_row(workspace: Path, demo_config: Path) -> None:
    """A rule that names a wire format puts it on every asset it matches."""
    hints = {
        '  - pattern: "*_HVSR/*.txt"': "ascii_3c",
        '  - pattern: "*_SASW/*.DAT"': "hp_sdf",
    }
    lines = demo_config.read_text(encoding="utf-8").splitlines()
    annotated: list[str] = []
    for line in lines:
        annotated.append(line)
        if line in hints:
            annotated.append(f"    format_hint: {hints[line]}")
    demo_config.write_text("\n".join(annotated) + "\n", encoding="utf-8")

    manifest = build_manifest(load_source_config(demo_config), workspace)
    counts: dict[FormatHint | None, int] = {}
    for asset in manifest.assets:
        counts[asset.format_hint] = counts.get(asset.format_hint, 0) + 1

    assert counts[FormatHint.ASCII_3C] == 4
    assert counts[FormatHint.HP_SDF] == 2
    assert None in counts  # photos and reports assert no wire format


def test_unhinted_assets_carry_no_invented_format(workspace: Path, demo_config: Path) -> None:
    manifest = build_manifest(load_source_config(demo_config), workspace)
    assert all(asset.format_hint is None for asset in manifest.assets)


def test_a_repair_is_refused_when_the_file_changed(workspace: Path, demo_config: Path) -> None:
    """A repair audited against different bytes must not be reused."""
    config = load_source_config(demo_config)
    manifest = build_manifest(config, workspace)
    recording = manifest.recordings[0]
    asset = next(a for a in manifest.assets if a.asset_id == recording.asset_id)

    add_repair(demo_config, asset.relative_path, "b" * 64)
    stale = load_source_config(demo_config)

    with pytest.raises(RecordingReadError) as excinfo:
        resolve_request(manifest, recording, stale, workspace)
    assert "re-audit" in str(excinfo.value)


def test_a_repair_applies_only_to_its_own_file(workspace: Path, demo_config: Path) -> None:
    config = load_source_config(demo_config)
    manifest = build_manifest(config, workspace)
    recordings = manifest.recordings
    assets = {a.asset_id: a for a in manifest.assets}
    target = assets[recordings[0].asset_id]

    add_repair(demo_config, target.relative_path, target.sha256)
    updated = load_source_config(demo_config)

    repaired = resolve_request(manifest, recordings[0], updated, workspace)
    other = resolve_request(manifest, recordings[1], updated, workspace)

    assert repaired.repair is not None
    assert repaired.repair.action is RepairAction.DROP_INCOMPLETE_FINAL_ROW
    assert other.repair is None


def test_assets_are_exported_as_csv(workspace: Path, demo_config: Path, tmp_path: Path) -> None:
    config = load_source_config(demo_config)
    manifest = build_manifest(config, workspace)
    out = write_manifest(manifest, tmp_path / "m", config)

    with open(out / "assets.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(manifest.assets)
    assert "format_hint" in rows[0]
    assert {row["relative_path"] for row in rows} == {a.relative_path for a in manifest.assets}


def test_format_hint_survives_a_parquet_round_trip(
    workspace: Path, demo_config: Path, tmp_path: Path
) -> None:
    config = load_source_config(demo_config)
    manifest = build_manifest(config, workspace)
    write_manifest(manifest, tmp_path / "m", config)

    restored = {a.relative_path: a.format_hint for a in read_manifest(tmp_path / "m").assets}
    assert restored == {a.relative_path: a.format_hint for a in manifest.assets}


def test_snapshot_records_the_full_environment(
    workspace: Path, demo_config: Path, tmp_path: Path
) -> None:
    config = load_source_config(demo_config)
    manifest = build_manifest(config, workspace)
    out = write_manifest(manifest, tmp_path / "m", config)
    snapshot = json.loads((out / "snapshot.json").read_text(encoding="utf-8"))

    for field in (
        "code_commit",
        "code_dirty",
        "source_tree_hash",
        "config_hash",
        "lockfile_hash",
        "python_version",
    ):
        assert field in snapshot, field
    assert snapshot["python_version"].startswith("3.12")
    assert len(snapshot["source_tree_hash"]) == 64


def test_source_tree_hash_reacts_to_source_changes(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    first = source_tree_hash(tmp_path)

    assert source_tree_hash(tmp_path) == first
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert source_tree_hash(tmp_path) != first


def test_environment_provenance_is_json_serialisable() -> None:
    json.dumps(environment_provenance())
