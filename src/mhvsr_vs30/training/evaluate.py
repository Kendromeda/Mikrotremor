"""Leave-one-site-out evaluation for the vertical slice.

With three independent sites this is an engineering smoke test. It proves the
pipeline runs end to end without a notebook and without leaking a site across
the split; it says nothing about whether the model works. Every result carries
that statement in its own metadata so a number cannot be quoted out of context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mhvsr_vs30.provenance import environment_provenance
from mhvsr_vs30.training.dataset import DatasetSnapshot
from mhvsr_vs30.training.models import MedianVs30, RidgeLogVs30, load_model, save_model
from mhvsr_vs30.training.splits import (
    Fold,
    external_holdout,
    leave_one_group_out,
    leave_one_site_out,
)

__all__ = [
    "NEHRP_BOUNDS",
    "EvaluationError",
    "nehrp_class",
    "run_external_holdout",
    "run_grouped_evaluation",
    "run_leave_one_site_out",
]

# ASCE 7 site classes by Vs30 in m/s. The class is what an engineer acts on, so
# getting the class right matters more than the exact number.
NEHRP_BOUNDS = ((1500.0, "A"), (760.0, "B"), (360.0, "C"), (180.0, "D"))

MINIMUM_SITES_FOR_SCIENCE = 30


class EvaluationError(ValueError):
    """The requested validation protocol is not supported by the snapshot."""


def nehrp_class(vs30_mps: float) -> str:
    for threshold, name in NEHRP_BOUNDS:
        if vs30_mps > threshold:
            return name
    return "E"


def _metrics(observed: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> dict[str, Any]:
    residual = np.log(predicted) - np.log(observed)
    total = float(weight.sum())
    bias = float((residual * weight).sum() / total)
    variance = float((weight * (residual - bias) ** 2).sum() / total)

    observed_class = [nehrp_class(v) for v in observed]
    predicted_class = [nehrp_class(v) for v in predicted]
    correct = float(
        sum(w for w, a, b in zip(weight, observed_class, predicted_class, strict=True) if a == b)
    )
    # Predicting a stiffer class than reality understates amplification, which is
    # the unconservative direction.
    order = ["E", "D", "C", "B", "A"]
    unconservative = float(
        sum(
            w
            for w, a, b in zip(weight, observed_class, predicted_class, strict=True)
            if order.index(b) > order.index(a)
        )
    )
    return {
        "n_rows": len(observed),
        "weighted_n_sites": total,
        "bias_ln": bias,
        "sigma_ln": float(np.sqrt(variance)),
        "mean_abs_residual_ln": float((np.abs(residual) * weight).sum() / total),
        "nehrp_class_accuracy": correct / total,
        "nehrp_unconservative_rate": unconservative / total,
    }


def run_leave_one_site_out(
    dataset: DatasetSnapshot,
    output_dir: Path | str,
    ridge_alpha: float = 1.0,
) -> dict[str, Any]:
    """Fit and score both baselines on every held-out site."""
    return _run_evaluation(
        dataset,
        output_dir,
        folds=leave_one_site_out(dataset.groups()),
        split_name="leave_one_site_out",
        group_column="site_id",
        ridge_alpha=ridge_alpha,
    )


def run_grouped_evaluation(
    dataset: DatasetSnapshot,
    output_dir: Path | str,
    *,
    split: str,
    ridge_alpha: float = 1.0,
) -> dict[str, Any]:
    """Evaluate with site, study, or geography groups and strict site isolation.

    Study and geography columns are required rather than guessed from row order.
    ``build_dataset`` supplies source-as-study and country-as-geography; combined
    snapshots may supply a more specific cited study identifier.
    """
    group_columns = {"site": "site_id", "study": "study_id", "geography": "geography_id"}
    try:
        group_column = group_columns[split]
    except KeyError as exc:
        raise EvaluationError("split must be one of: site, study, geography") from exc
    _validate_group_metadata(dataset, group_column)
    return _run_evaluation(
        dataset,
        output_dir,
        folds=leave_one_group_out(dataset.frame[group_column].to_numpy(), group_name=split),
        split_name=f"leave_one_{split}_out",
        group_column=group_column,
        ridge_alpha=ridge_alpha,
    )


def run_external_holdout(
    dataset: DatasetSnapshot,
    output_dir: Path | str,
    *,
    country: str = "Indonesia",
    ridge_alpha: float = 1.0,
) -> dict[str, Any]:
    """Train outside one country and score only that country's whole sites."""
    _validate_group_metadata(dataset, "country")
    target_country = country.strip()
    if not target_country:
        raise EvaluationError("external holdout country must be non-empty")
    countries = dataset.frame["country"].astype(str).to_numpy()
    sites = dataset.groups()
    try:
        fold = external_holdout(sites, countries == target_country, holdout_name=target_country)
    except ValueError as exc:
        raise EvaluationError(str(exc)) from exc

    report = _run_evaluation(
        dataset,
        output_dir,
        folds=[fold],
        split_name="external_country_holdout",
        group_column="country",
        ridge_alpha=ridge_alpha,
    )
    report["external_holdout"] = {
        "country": target_country,
        "train_sites": sorted(set(sites[fold.train_index].tolist())),
        "test_sites": sorted(set(sites[fold.test_index].tolist())),
        "train_countries": sorted(set(countries[fold.train_index].tolist())),
    }
    report["evaluation_scope"] = "external_validation_candidate"
    report["not_a_scientific_result_because"] = (
        "external country holdout is a validation design, not evidence by itself; "
        + report["not_a_scientific_result_because"]
    )
    _write_report(Path(output_dir), report)
    return report


def _run_evaluation(
    dataset: DatasetSnapshot,
    output_dir: Path | str,
    *,
    folds: list[Fold],
    split_name: str,
    group_column: str,
    ridge_alpha: float,
) -> dict[str, Any]:
    """Fit and score supplied folds after enforcing label and site invariants."""
    _validate_independent_labels(dataset)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    features = dataset.features()
    targets = dataset.targets()
    weights = dataset.weights()
    groups = dataset.groups()
    results: dict[str, Any] = {}
    for factory in (MedianVs30, lambda: RidgeLogVs30(alpha=ridge_alpha)):
        model_name = factory().name
        per_fold: list[dict[str, Any]] = []
        observed_all: list[float] = []
        predicted_all: list[float] = []
        weight_all: list[float] = []

        for fold in folds:
            model = factory()
            model.fit(
                features[fold.train_index], targets[fold.train_index], weights[fold.train_index]
            )

            # A model that predicts differently after a round trip cannot be
            # deployed, so the check happens on every fold rather than once.
            path = save_model(
                model,
                target_dir / "models" / f"{model_name}__{fold.test_group.replace(':', '_')}.joblib",
            )
            before = model.predict(features[fold.test_index])
            after = load_model(path).predict(features[fold.test_index])
            if not np.array_equal(before, after):
                raise RuntimeError(
                    f"{model_name} predictions changed after reload on fold {fold.test_group}"
                )

            observed = targets[fold.test_index]
            fold_weight = weights[fold.test_index]
            fold_result = {
                "test_site": fold.test_site if group_column == "site_id" else None,
                "test_group": fold.test_group,
                "test_sites": sorted(set(groups[fold.test_index].tolist())),
                "train_sites": sorted(set(groups[fold.train_index].tolist())),
                "n_train_rows": int(fold.train_index.size),
                "n_test_rows": int(fold.test_index.size),
                "observed_vs30_mps_mean": float(observed.mean()),
                "predicted_vs30_mps_mean": float(before.mean()),
                "observed_nehrp": nehrp_class(float(observed.mean())),
                "predicted_nehrp": nehrp_class(float(before.mean())),
                **_metrics(observed, before, fold_weight),
            }
            # Preserve the field used by version-1 LOSO reports while making
            # its single-site meaning explicit for wider held-out groups.
            if group_column == "site_id":
                fold_result["observed_vs30_mps"] = float(observed[0])
            per_fold.append(fold_result)
            observed_all.extend(observed.tolist())
            predicted_all.extend(before.tolist())
            weight_all.extend(fold_weight.tolist())

        results[model_name] = {
            "per_fold": per_fold,
            "pooled": _metrics(
                np.asarray(observed_all), np.asarray(predicted_all), np.asarray(weight_all)
            ),
        }

    report = _report(dataset, folds, results, ridge_alpha, split_name, group_column)
    _write_report(target_dir, report)
    return report


def _report(
    dataset: DatasetSnapshot,
    folds: list[Fold],
    results: dict[str, Any],
    ridge_alpha: float,
    split_name: str,
    group_column: str,
) -> dict[str, Any]:
    """Wrap the numbers in the context that keeps them honest."""
    site_count = len(dataset.sites)
    sites = dataset.groups()
    split_groups = dataset.frame[group_column].astype(str).to_numpy()
    site_leakage = [
        fold.test_group
        for fold in folds
        if set(sites[fold.train_index].tolist()) & set(sites[fold.test_index].tolist())
    ]
    group_leakage = [
        fold.test_group
        for fold in folds
        if (
            set(split_groups[fold.train_index].tolist())
            & set(split_groups[fold.test_index].tolist())
        )
    ]

    return {
        "evaluation_scope": "engineering_only",
        "promotion_eligible": False,
        "not_a_scientific_result_because": (
            f"{site_count} independent sites is far below the {MINIMUM_SITES_FOR_SCIENCE} "
            "needed for a defensible generalisation estimate; these numbers show only "
            "that the pipeline runs and does not leak a site across the split"
        ),
        "n_sites": site_count,
        "n_rows": len(dataset.frame),
        "sites": dataset.sites,
        "split": split_name,
        "group_column": group_column,
        "site_leakage_detected": site_leakage,
        "group_leakage_detected": group_leakage,
        "ridge_alpha": ridge_alpha,
        "dataset": {
            key: dataset.metadata.get(key)
            for key in (
                "source_id",
                "profile_id",
                "profile_identity",
                "manifest_snapshot_hash",
                "manifest_dir",
                "recordings_per_site",
                "vs30_by_site",
                "label_independence",
            )
        },
        "models": results,
        "environment": environment_provenance(),
    }


def _validate_independent_labels(dataset: DatasetSnapshot) -> None:
    if "label_independence" not in dataset.frame:
        raise EvaluationError("dataset has no label_independence provenance column")
    independence = set(dataset.frame["label_independence"].astype(str).tolist())
    if independence != {"independent_of_hvsr"}:
        raise EvaluationError(
            "evaluation requires labels with label_independence=independent_of_hvsr; "
            f"got {sorted(independence)}"
        )


def _validate_group_metadata(dataset: DatasetSnapshot, column: str) -> None:
    if column not in dataset.frame:
        raise EvaluationError(f"dataset has no {column!r} metadata required for this split")
    values = dataset.frame[column]
    if values.isna().any() or any(not str(value).strip() for value in values):
        raise EvaluationError(f"dataset {column!r} metadata must be present for every row")
    # A physical site cannot be simultaneously assigned to two countries,
    # studies, or geography groups. That would undermine every group protocol.
    cardinality = dataset.frame.groupby("site_id", sort=False)[column].nunique(dropna=False)
    inconsistent = sorted(cardinality[cardinality != 1].index.tolist())
    if inconsistent:
        raise EvaluationError(
            f"{column!r} metadata is inconsistent within physical sites: {inconsistent}"
        )


def _write_report(target_dir: Path, report: dict[str, Any]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
