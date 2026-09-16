"""Phase 5 integration: the whole slice from manifest to a scored fold."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mhvsr_vs30.manifest.store import read_manifest
from mhvsr_vs30.preprocessing.config import load_profile
from mhvsr_vs30.training.dataset import DatasetError, build_dataset
from mhvsr_vs30.training.evaluate import run_leave_one_site_out
from mhvsr_vs30.training.splits import leave_one_site_out

pytestmark = pytest.mark.requires_real_data

MANIFEST = Path("artifacts/manifests/usgs_arra_v1")
PROFILE = Path("configs/preprocessing/training_v1.yaml")
CURVES = Path("artifacts/curves")


@pytest.fixture(scope="module")
def dataset():
    if not MANIFEST.is_dir() or not (CURVES / "training_v1").is_dir():
        pytest.skip("manifest or curves are not built in this workspace")
    return build_dataset(
        read_manifest(MANIFEST), load_profile(PROFILE), CURVES, manifest_dir=MANIFEST
    )


def test_dataset_joins_only_by_identifier(dataset) -> None:
    manifest = read_manifest(MANIFEST)
    label_by_site = {label.site_id: label.vs30_mps for label in manifest.labels}

    for _, row in dataset.frame.iterrows():
        assert row["vs30_mps"] == label_by_site[row["site_id"]]
    assert set(dataset.frame["site_id"]) <= set(label_by_site)


def test_every_site_carries_unit_weight(dataset) -> None:
    totals = dataset.frame.groupby("site_id")["sample_weight"].sum()
    assert np.allclose(totals.to_numpy(), 1.0)


def test_features_are_35_positive_finite_columns(dataset) -> None:
    features = dataset.features()
    assert features.shape[1] == 35
    assert np.isfinite(features).all()
    assert (features > 0).all()


def test_only_hvsr_independent_labels_enter_by_default(dataset) -> None:
    """A Vs30 inverted from the same HVSR would make any later score circular."""
    assert set(dataset.frame["label_independence"]) == {"independent_of_hvsr"}


def test_dataset_points_back_at_its_inputs(dataset) -> None:
    metadata = dataset.metadata
    assert metadata["manifest_snapshot_hash"]
    assert metadata["profile_identity"]
    assert metadata["source_tree_hash"]
    assert dataset.frame["processing_hash"].notna().all()
    assert dataset.frame["input_sha256"].str.len().eq(64).all()


def test_no_site_appears_on_both_sides_of_a_fold(dataset) -> None:
    groups = dataset.groups()
    for fold in leave_one_site_out(groups):
        assert fold.test_site not in set(groups[fold.train_index])


def test_smoke_run_is_labelled_as_engineering_only(dataset, tmp_path: Path) -> None:
    report = run_leave_one_site_out(dataset, tmp_path / "run")

    assert report["evaluation_scope"] == "engineering_only"
    assert report["promotion_eligible"] is False
    assert report["n_sites"] == 3
    assert report["site_leakage_detected"] == []
    assert "below" in report["not_a_scientific_result_because"]

    written = json.loads((tmp_path / "run" / "evaluation.json").read_text(encoding="utf-8"))
    assert written["promotion_eligible"] is False


def test_both_baselines_produce_finite_scores(dataset, tmp_path: Path) -> None:
    report = run_leave_one_site_out(dataset, tmp_path / "run")

    for name in ("median_vs30", "ridge_log_vs30"):
        pooled = report["models"][name]["pooled"]
        assert np.isfinite(pooled["sigma_ln"])
        assert np.isfinite(pooled["bias_ln"])
        assert 0.0 <= pooled["nehrp_class_accuracy"] <= 1.0
        assert len(report["models"][name]["per_fold"]) == 3


def test_the_run_is_reproducible(dataset, tmp_path: Path) -> None:
    first = run_leave_one_site_out(dataset, tmp_path / "a")
    second = run_leave_one_site_out(dataset, tmp_path / "b")

    assert first["models"] == second["models"]


def test_dependent_labels_are_refused_unless_asked_for() -> None:
    """Turning the guard off must be an explicit act, not a default."""
    manifest = read_manifest(MANIFEST)
    swapped = manifest.replace(
        labels=tuple(
            type(label)(**{**label.to_row(), "label_independence": "hvsr_derived"})
            for label in manifest.labels
        )
    )
    with pytest.raises(DatasetError, match="no model-ready curves"):
        build_dataset(swapped, load_profile(PROFILE), CURVES)

    permissive = build_dataset(
        swapped, load_profile(PROFILE), CURVES, require_independent_labels=False
    )
    assert set(permissive.frame["label_independence"]) == {"hvsr_derived"}
