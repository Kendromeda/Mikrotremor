"""Tests for the independent, tabular Medan evaluation path."""

from __future__ import annotations

import pandas as pd
import pytest

from mhvsr_vs30.training.medan_tabular import MedanDataError, evaluate_medan, load_medan_pairs


def test_load_medan_pairs_joins_only_audited_sites(tmp_path) -> None:
    pd.DataFrame(
        [
            {"site_id": "MD1", "latitude": 3.0, "longitude": 98.0, "f0_hz": 1.0, "a0": 2.0},
            {"site_id": "MD2", "latitude": 3.1, "longitude": 98.1, "f0_hz": 2.0, "a0": 3.0},
        ]
    ).to_csv(tmp_path / "hvsr.csv", index=False)
    pd.DataFrame(
        [
            {"site_id": "MD1", "vs30_mps": 200.0},
            {"site_id": "MD2", "vs30_mps": 300.0},
        ]
    ).to_csv(tmp_path / "masw_vs30.csv", index=False)
    pd.DataFrame([{"site_id": "MD1", "coordinate_delta_m_approx": 50.0,
                   "match_method": "exact_site_id_coordinate_checked"}]).to_csv(
        tmp_path / "hvsr_masw_crosswalk.csv", index=False
    )

    frame = load_medan_pairs(tmp_path)
    assert frame["site_id"].tolist() == ["MD1"]
    assert frame["vs30_mps"].tolist() == [200.0]


def test_load_medan_pairs_rejects_unsafe_pair(tmp_path) -> None:
    pd.DataFrame(
        [{"site_id": "MD1", "latitude": 3.0, "longitude": 98.0, "f0_hz": 1.0, "a0": 2.0}]
    ).to_csv(tmp_path / "hvsr.csv", index=False)
    pd.DataFrame([{"site_id": "MD1", "vs30_mps": 200.0}]).to_csv(
        tmp_path / "masw_vs30.csv", index=False
    )
    pd.DataFrame([{"site_id": "MD1", "coordinate_delta_m_approx": 150.0,
                   "match_method": "exact_site_id_coordinate_checked"}]).to_csv(
        tmp_path / "hvsr_masw_crosswalk.csv", index=False
    )

    with pytest.raises(MedanDataError, match="coordinate"):
        load_medan_pairs(tmp_path)


def test_load_medan_pairs_requires_audited_match_method(tmp_path) -> None:
    pd.DataFrame(
        [{"site_id": "MD1", "latitude": 3.0, "longitude": 98.0, "f0_hz": 1.0, "a0": 2.0}]
    ).to_csv(tmp_path / "hvsr.csv", index=False)
    pd.DataFrame([{"site_id": "MD1", "vs30_mps": 200.0}]).to_csv(
        tmp_path / "masw_vs30.csv", index=False
    )
    pd.DataFrame([{"site_id": "MD1", "coordinate_delta_m_approx": 50.0}]).to_csv(
        tmp_path / "hvsr_masw_crosswalk.csv", index=False
    )
    with pytest.raises(MedanDataError, match="match_method"):
        load_medan_pairs(tmp_path)


def test_spatial_evaluation_holds_out_whole_sites_and_disclaims_scope() -> None:
    rows = [
        {
            "site_id": f"MD{cluster}-{index}",
            "latitude": 3.0 + cluster * 0.01 + index * 0.001,
            "longitude": 98.0 + cluster * 0.01 + index * 0.001,
            "f0_hz": 0.5 + cluster * 0.2 + index * 0.01,
            "a0": 1.0 + index * 0.1,
            "vs30_mps": 150.0 + cluster * 100 + index,
        }
        for cluster in range(5)
        for index in range(4)
    ]
    report = evaluate_medan(pd.DataFrame(rows), n_regions=5)

    assert report["n_sites"] == 20
    assert report["n_studies"] == 1
    assert report["evaluation_scope"] == "exploratory_single_study"
    assert report["external_validation"] is False
    assert report["site_leakage_detected"] == []
    assert len(report["folds"]) == 5
    assert set(report["models"]) == {"median_vs30", "ridge_log_f0_a0"}
    longitude_by_site = {row["site_id"]: row["longitude"] for row in rows}
    for fold in report["folds"]:
        assert set(fold["train_sites"]).isdisjoint(fold["test_sites"])
        for train_site in fold["train_sites"]:
            assert all(
                abs(longitude_by_site[train_site] - longitude_by_site[test_site]) > 0.0089
                for test_site in fold["test_sites"]
            )


def test_spatial_evaluation_rejects_duplicate_site() -> None:
    frame = pd.DataFrame(
        [
            {
                "site_id": "MD1",
                "latitude": 3.0,
                "longitude": 98.0,
                "f0_hz": 1.0,
                "a0": 2.0,
                "vs30_mps": 200.0,
            },
            {
                "site_id": "MD1",
                "latitude": 3.1,
                "longitude": 98.1,
                "f0_hz": 2.0,
                "a0": 3.0,
                "vs30_mps": 300.0,
            },
        ]
    )
    with pytest.raises(MedanDataError, match="duplicate"):
        evaluate_medan(frame, n_regions=2)
