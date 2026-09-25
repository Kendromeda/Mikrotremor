"""Evaluation protocols must keep physical sites and label provenance isolated."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mhvsr_vs30.features.hvsr import feature_columns
from mhvsr_vs30.training.dataset import DatasetError, DatasetSnapshot, combine_snapshots
from mhvsr_vs30.training.evaluate import (
    EvaluationError,
    run_external_holdout,
    run_grouped_evaluation,
)
from mhvsr_vs30.training.splits import external_holdout, leave_one_group_out


def _dataset(*, dependent: bool = False) -> DatasetSnapshot:
    columns = feature_columns()
    rows = []
    for index, (site, study, country, target) in enumerate(
        (
            ("site:us-a", "study:us", "USA", 300.0),
            ("site:us-b", "study:us", "USA", 600.0),
            ("site:id-a", "study:id", "Indonesia", 250.0),
            ("site:id-b", "study:id", "Indonesia", 450.0),
        )
    ):
        row = {
            "recording_id": f"recording:{index}",
            "site_id": site,
            "study_id": study,
            "country": country,
            "geography_id": f"country:{country}",
            "vs30_mps": target,
            "sample_weight": 1.0,
            "label_independence": "hvsr_derived" if dependent else "independent_of_hvsr",
        }
        row.update({column: float(index + 1) for column in columns})
        rows.append(row)
    return DatasetSnapshot(pd.DataFrame(rows), {"source_id": "combined"})


def test_leave_one_group_out_holds_out_an_entire_study() -> None:
    groups = np.asarray(["study:a", "study:a", "study:b", "study:c"])

    folds = leave_one_group_out(groups, group_name="study")

    assert [fold.test_group for fold in folds] == ["study:a", "study:b", "study:c"]
    for fold in folds:
        assert set(groups[fold.test_index]) == {fold.test_group}
        assert fold.test_group not in set(groups[fold.train_index])


def test_external_holdout_keeps_each_physical_site_on_one_side() -> None:
    sites = np.asarray(["us:a", "us:b", "id:a", "id:b"])
    indonesia = np.asarray([False, False, True, True])

    fold = external_holdout(sites, indonesia, holdout_name="Indonesia")

    assert set(sites[fold.train_index]) == {"us:a", "us:b"}
    assert set(sites[fold.test_index]) == {"id:a", "id:b"}


def test_external_holdout_rejects_a_site_split_across_countries() -> None:
    with pytest.raises(ValueError, match="physical site"):
        external_holdout(
            np.asarray(["shared", "shared", "other"]),
            np.asarray([False, True, False]),
            holdout_name="Indonesia",
        )


def test_grouped_study_evaluation_reports_site_and_group_leakage_checks(tmp_path) -> None:
    report = run_grouped_evaluation(_dataset(), tmp_path, split="study")

    assert report["split"] == "leave_one_study_out"
    assert report["site_leakage_detected"] == []
    assert report["group_leakage_detected"] == []
    assert report["evaluation_scope"] == "engineering_only"


def test_external_indonesia_holdout_is_identified_and_never_promotion_eligible(tmp_path) -> None:
    report = run_external_holdout(_dataset(), tmp_path, country="Indonesia")

    assert report["split"] == "external_country_holdout"
    assert report["external_holdout"]["country"] == "Indonesia"
    assert report["external_holdout"]["test_sites"] == ["site:id-a", "site:id-b"]
    assert report["promotion_eligible"] is False


def test_evaluation_refuses_dependent_labels_even_for_a_permissive_snapshot(tmp_path) -> None:
    with pytest.raises(EvaluationError, match="independent_of_hvsr"):
        run_grouped_evaluation(_dataset(dependent=True), tmp_path, split="study")


def test_external_holdout_requires_country_metadata(tmp_path) -> None:
    frame = _dataset().frame.drop(columns=["country"])
    with pytest.raises(EvaluationError, match="country"):
        run_external_holdout(DatasetSnapshot(frame, {}), tmp_path, country="Indonesia")


def test_combine_snapshots_preserves_source_provenance_and_rebalances_site_weights() -> None:
    us = _dataset().frame.iloc[:2].copy()
    indonesia = _dataset().frame.iloc[2:].copy()
    us["canonical_site_id"] = us["site_id"]
    indonesia["canonical_site_id"] = indonesia["site_id"]
    first = DatasetSnapshot(us, {"source_id": "us-study", "profile_identity": {"id": "v1"}})
    second = DatasetSnapshot(
        indonesia, {"source_id": "id-study", "profile_identity": {"id": "v1"}}
    )

    combined = combine_snapshots([first, second])

    assert combined.sites == ["site:id-a", "site:id-b", "site:us-a", "site:us-b"]
    assert combined.metadata["source_ids"] == ["id-study", "us-study"]
    assert len(combined.metadata["input_snapshots"]) == 2
    assert combined.frame.groupby("site_id")["sample_weight"].sum().eq(1.0).all()


def test_combined_snapshots_enable_external_indonesia_evaluation(tmp_path) -> None:
    dataset = _dataset()
    us = dataset.frame.iloc[:2].copy()
    indonesia = dataset.frame.iloc[2:].copy()
    us["canonical_site_id"] = us["site_id"]
    indonesia["canonical_site_id"] = indonesia["site_id"]
    combined = combine_snapshots(
        [
            DatasetSnapshot(
                us,
                {"source_id": "us-study", "profile_identity": {"id": "v1"}},
            ),
            DatasetSnapshot(
                indonesia,
                {"source_id": "id-study", "profile_identity": {"id": "v1"}},
            ),
        ]
    )

    report = run_external_holdout(combined, tmp_path, country="Indonesia")

    assert report["external_holdout"]["test_sites"] == ["site:id-a", "site:id-b"]


def test_combine_snapshots_rejects_duplicate_recording_ids() -> None:
    snapshot = _dataset()
    snapshot = DatasetSnapshot(
        snapshot.frame,
        {"source_id": "combined", "profile_identity": {"id": "v1"}},
    )
    with pytest.raises(DatasetError, match="duplicate recording_id"):
        combine_snapshots([snapshot, snapshot])


def test_combine_snapshots_rejects_conflicting_site_labels() -> None:
    original = _dataset()
    original_frame = original.frame.copy()
    original_frame["canonical_site_id"] = original_frame["site_id"]
    changed = original_frame.copy()
    changed.loc[changed["site_id"] == "site:us-a", "vs30_mps"] = 999.0
    changed.loc[changed["site_id"] == "site:us-a", "recording_id"] = "recording:replacement"
    with pytest.raises(DatasetError, match="conflicting metadata"):
        combine_snapshots(
            [
                DatasetSnapshot(
                    original_frame.iloc[:1].copy(),
                    {"source_id": "first", "profile_identity": {"id": "v1"}},
                ),
                DatasetSnapshot(
                    changed.iloc[:1].copy(),
                    {"source_id": "second", "profile_identity": {"id": "v1"}},
                ),
            ]
        )


def test_combine_snapshots_rejects_mismatched_preprocessing_profiles() -> None:
    first = DatasetSnapshot(
        _dataset().frame.iloc[:2].copy(), {"source_id": "first", "profile_identity": {"id": "v1"}}
    )
    second = DatasetSnapshot(
        _dataset().frame.iloc[2:].copy(), {"source_id": "second", "profile_identity": {"id": "v2"}}
    )
    with pytest.raises(DatasetError, match="preprocessing profiles"):
        combine_snapshots([first, second])


def test_combine_snapshots_rejects_a_snapshot_without_profile_provenance() -> None:
    dataset = _dataset()
    with pytest.raises(DatasetError, match="profile_identity"):
        combine_snapshots(
            [
                DatasetSnapshot(dataset.frame.iloc[:2].copy(), {"source_id": "first"}),
                DatasetSnapshot(
                    dataset.frame.iloc[2:].copy(),
                    {"source_id": "second", "profile_identity": {"id": "v1"}},
                ),
            ]
        )
