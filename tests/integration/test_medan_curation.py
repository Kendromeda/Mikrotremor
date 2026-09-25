"""Regression checks for the audited Medan supplementary-table extraction."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path("datasets/processed/medan_supplement_v1")
RAW_DOCUMENT = Path("datasets/source_08_medan_supplement/raw/40677_2022_227_MOESM1_ESM.docx")
SOURCE_SHA256 = "f522ef6780fe1f9c4b413b794509f884f0abc019afc2238fbc1ab699c51399c7"


def _rows(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _extractor_module() -> object:
    path = ROOT / "extract_medan_tables.py"
    spec = importlib.util.spec_from_file_location("medan_extractor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_medan_crosswalk_is_safe_and_source_pinned() -> None:
    masw = _rows("masw_vs30.csv")
    hvsr = _rows("hvsr.csv")
    paired = _rows("hvsr_masw_crosswalk.csv")
    reconciliation = {row["site_id"]: row for row in _rows("reconciliation.csv")}
    provenance = json.loads((ROOT / "source_provenance.json").read_text(encoding="utf-8"))

    assert len(masw) == 198
    assert len(hvsr) == 202
    assert len(paired) == 185
    assert len({row["site_id"] for row in paired}) == len(paired)
    assert all(
        sum(row["site_id"] == paired_id["site_id"] for row in masw) == 1
        for paired_id in paired
    )
    assert all(
        sum(row["site_id"] == paired_id["site_id"] for row in hvsr) == 1
        for paired_id in paired
    )
    assert all(row["match_method"] == "exact_site_id_coordinate_checked" for row in paired)
    assert all(float(row["coordinate_delta_m_approx"]) <= 100.0 for row in paired)
    assert all(
        reconciliation[row["site_id"]]["reconciliation_status"] == "paired" for row in paired
    )
    assert reconciliation["MD3"]["reconciliation_status"] == "coordinate_conflict"
    assert "MD3" not in {row["site_id"] for row in paired}
    assert provenance["source_sha256"] == SOURCE_SHA256
    assert provenance["masw_row_count"] == len(masw)
    assert provenance["hvsr_row_count"] == len(hvsr)
    assert provenance["paired_row_count"] == len(paired)


@pytest.mark.requires_real_data
def test_regeneration_from_source_docx_matches_committed_outputs(tmp_path: Path) -> None:
    if not RAW_DOCUMENT.is_file():
        pytest.skip("Medan source DOCX is not present in this workspace")

    extractor = _extractor_module()
    extractor.extract(RAW_DOCUMENT, tmp_path)  # type: ignore[attr-defined]

    for name in ("masw_vs30.csv", "hvsr.csv", "hvsr_masw_crosswalk.csv", "reconciliation.csv"):
        assert (tmp_path / name).read_bytes() == (ROOT / name).read_bytes()
    actual = json.loads((tmp_path / "source_provenance.json").read_text(encoding="utf-8"))
    expected = json.loads((ROOT / "source_provenance.json").read_text(encoding="utf-8"))
    assert actual == expected
