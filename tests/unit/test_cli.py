"""Tests for the command line surface, including its exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mhvsr_vs30.cli as cli
from mhvsr_vs30.cli import main
from mhvsr_vs30.training.dataset import DatasetSnapshot


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(list(argv))
    captured = capsys.readouterr()
    payload = json.loads(captured.out) if captured.out.strip() else {}
    return code, payload


@pytest.fixture
def built(workspace: Path, demo_config: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    output = workspace / "artifacts" / "manifests" / "demo_v1"
    code, _ = run(
        capsys,
        "manifest",
        "build",
        "--config",
        str(demo_config),
        "--output",
        str(output),
        "--workspace",
        str(workspace),
        "--cache",
        str(workspace / "cache.json"),
    )
    assert code == 0
    return output


def test_build_writes_every_table(built: Path) -> None:
    names = {path.name for path in built.iterdir()}
    assert {
        "sources.parquet",
        "sites.parquet",
        "assets.parquet",
        "recordings.parquet",
        "labels.parquet",
        "summary.json",
        "snapshot.json",
    } <= names


def test_snapshot_pins_the_config_that_produced_it(built: Path, demo_config: Path) -> None:
    snapshot = json.loads((built / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["source_id"] == "demo_source"
    assert snapshot["recording_count"] == 4
    assert len(snapshot["config_hash"]) == 64
    assert snapshot["config_path"].endswith("demo.yaml")


def test_validate_passes_on_a_fresh_build(
    built: Path, demo_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = run(capsys, "manifest", "validate", str(built), "--config", str(demo_config))

    assert code == 0
    assert payload["status"] == "passed"
    assert payload["failures"] == []


def test_validate_reports_failures_and_exits_nonzero(
    built: Path, demo_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = demo_config.read_text(encoding="utf-8")
    demo_config.write_text(text.replace("recording_count: 4", "recording_count: 9"), "utf-8")

    code, payload = run(capsys, "manifest", "validate", str(built), "--config", str(demo_config))

    assert code == 1
    assert payload["status"] == "failed"
    assert any("recording_count" in failure for failure in payload["failures"])


def test_build_exits_two_when_the_config_is_wrong(
    workspace: Path, demo_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = demo_config.read_text(encoding="utf-8")
    demo_config.write_text(text.replace("site_count: 2", "site_count: 7"), "utf-8")

    code = main(
        [
            "manifest",
            "build",
            "--config",
            str(demo_config),
            "--output",
            str(workspace / "out"),
            "--workspace",
            str(workspace),
            "--cache",
            str(workspace / "cache.json"),
        ]
    )
    assert code == 2
    assert "site_count" in capsys.readouterr().err


def test_missing_manifest_exits_two(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code = main(["manifest", "validate", str(tmp_path), "--config", str(tmp_path / "gone.yaml")])
    assert code == 2
    assert "missing manifest table" in capsys.readouterr().err


def test_missing_config_exits_two(
    built: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code = main(["manifest", "validate", str(built), "--config", str(tmp_path / "gone.yaml")])
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_recording_inspect_reports_structural_qc(
    built: Path, demo_config: Path, workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from mhvsr_vs30.manifest.store import read_manifest

    recording_id = read_manifest(built).recordings[0].recording_id
    code, payload = run(
        capsys,
        "recording",
        "inspect",
        "--manifest",
        str(built),
        "--config",
        str(demo_config),
        "--recording-id",
        recording_id[:12],
        "--workspace",
        str(workspace),
    )

    assert code == 0
    assert payload["reader"] == "usgs_ascii_3c"
    assert payload["components"] == ["Z", "N", "E"]
    assert payload["samples"] == 40
    assert payload["sampling_rate_hz"] == 200.0
    assert payload["structural_qc"] == "passed"


def test_recording_inspect_rejects_an_unknown_id(
    built: Path, demo_config: Path, workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "recording",
            "inspect",
            "--manifest",
            str(built),
            "--config",
            str(demo_config),
            "--recording-id",
            "ffffffffffff",
            "--workspace",
            str(workspace),
        ]
    )
    assert code == 2
    assert "no recording" in capsys.readouterr().err


def test_recording_validate_all_writes_a_report(
    built: Path, demo_config: Path, workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = workspace / "artifacts" / "reports" / "demo_qc.json"
    code, payload = run(
        capsys,
        "recording",
        "validate-all",
        "--manifest",
        str(built),
        "--config",
        str(demo_config),
        "--workspace",
        str(workspace),
        "--report",
        str(report),
    )

    assert code == 0
    assert payload["status"] == "passed"
    assert payload["passed_count"] == 4
    assert json.loads(report.read_text(encoding="utf-8"))["failed"] == []


def test_recording_validate_all_reports_a_broken_file(
    built: Path, demo_config: Path, workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A corrupt recording is named in the report, not swallowed."""
    from mhvsr_vs30.manifest.schemas import load_source_config
    from mhvsr_vs30.manifest.store import read_manifest

    config = load_source_config(demo_config)
    manifest = read_manifest(built)
    assets = {a.asset_id: a for a in manifest.assets}
    victim = workspace / config.root / assets[manifest.recordings[0].asset_id].relative_path
    victim.write_text("1\t2\n3\t4\n", encoding="ascii")

    code, payload = run(
        capsys,
        "recording",
        "validate-all",
        "--manifest",
        str(built),
        "--config",
        str(demo_config),
        "--workspace",
        str(workspace),
    )

    assert code == 1
    assert payload["status"] == "failed"
    assert payload["failed"][0]["error_type"] == "MalformedAsciiError"


def test_train_evaluate_combines_snapshots_for_a_grouped_split(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = DatasetSnapshot.__new__(DatasetSnapshot)
    loaded: list[Path] = []
    captured: dict[str, object] = {}

    def fake_read(directory: Path) -> DatasetSnapshot:
        loaded.append(directory)
        return snapshot

    def fake_combine(snapshots: list[DatasetSnapshot]) -> DatasetSnapshot:
        captured["combined"] = snapshots
        return snapshot

    def fake_evaluate(
        dataset: DatasetSnapshot, output: Path, *, split: str, ridge_alpha: float
    ) -> dict:
        captured.update(dataset=dataset, output=output, split=split, alpha=ridge_alpha)
        return {"split": "leave_one_study_out", "site_leakage_detected": []}

    monkeypatch.setattr(cli, "read_dataset_snapshot", fake_read)
    monkeypatch.setattr(cli, "combine_snapshots", fake_combine)
    monkeypatch.setattr(cli, "run_grouped_evaluation", fake_evaluate)

    code, payload = run(
        capsys,
        "train",
        "evaluate",
        "--dataset",
        str(tmp_path / "us"),
        "--dataset",
        str(tmp_path / "id"),
        "--split",
        "study",
        "--output",
        str(tmp_path / "run"),
        "--alpha",
        "0.5",
    )

    assert code == 0
    assert loaded == [tmp_path / "us", tmp_path / "id"]
    assert captured["split"] == "study"
    assert captured["alpha"] == 0.5
    assert payload["split"] == "leave_one_study_out"


def test_train_evaluate_runs_an_external_country_holdout(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = DatasetSnapshot.__new__(DatasetSnapshot)
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "read_dataset_snapshot", lambda _: snapshot)

    def fake_external(
        dataset: DatasetSnapshot, output: Path, *, country: str, ridge_alpha: float
    ) -> dict:
        captured.update(dataset=dataset, output=output, country=country, alpha=ridge_alpha)
        return {"split": "external_country_holdout", "site_leakage_detected": []}

    monkeypatch.setattr(cli, "run_external_holdout", fake_external)

    code, payload = run(
        capsys,
        "train",
        "evaluate",
        "--dataset",
        str(tmp_path / "combined"),
        "--external-country",
        "Indonesia",
        "--output",
        str(tmp_path / "run"),
    )

    assert code == 0
    assert captured["country"] == "Indonesia"
    assert payload["split"] == "external_country_holdout"
