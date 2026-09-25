"""Exercise the Medan tabular evaluation through the package CLI."""

from __future__ import annotations

import json
from pathlib import Path

from mhvsr_vs30.cli import main


def test_medan_tabular_cli_evaluates_audited_pairs(tmp_path: Path) -> None:
    data = Path("datasets/processed/medan_supplement_v1")
    report_path = tmp_path / "medan_evaluation.json"
    result = main(
        [
            "tabular",
            "evaluate-medan",
            "--data",
            str(data),
            "--report",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert result == 0
    assert report["n_sites"] == 185
    assert report["site_leakage_detected"] == []
    assert report["external_validation"] is False
    assert len(report["folds"]) == 5
    assert sum(fold["n_test_sites"] for fold in report["folds"]) == 185
    assert all(fold["n_test_sites"] == 37 for fold in report["folds"])
