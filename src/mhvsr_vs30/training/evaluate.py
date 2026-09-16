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
from mhvsr_vs30.training.splits import leave_one_site_out

__all__ = ["NEHRP_BOUNDS", "nehrp_class", "run_leave_one_site_out"]

# ASCE 7 site classes by Vs30 in m/s. The class is what an engineer acts on, so
# getting the class right matters more than the exact number.
NEHRP_BOUNDS = ((1500.0, "A"), (760.0, "B"), (360.0, "C"), (180.0, "D"))

MINIMUM_SITES_FOR_SCIENCE = 30


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
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    features = dataset.features()
    targets = dataset.targets()
    weights = dataset.weights()
    groups = dataset.groups()
    folds = leave_one_site_out(groups)

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
                target_dir / "models" / f"{model_name}__{fold.test_site.replace(':', '_')}.joblib",
            )
            before = model.predict(features[fold.test_index])
            after = load_model(path).predict(features[fold.test_index])
            if not np.array_equal(before, after):
                raise RuntimeError(
                    f"{model_name} predictions changed after reload on fold {fold.test_site}"
                )

            observed = targets[fold.test_index]
            fold_weight = weights[fold.test_index]
            per_fold.append(
                {
                    "test_site": fold.test_site,
                    "train_sites": sorted(set(groups[fold.train_index].tolist())),
                    "n_train_rows": int(fold.train_index.size),
                    "n_test_rows": int(fold.test_index.size),
                    "observed_vs30_mps": float(observed[0]),
                    "predicted_vs30_mps_mean": float(before.mean()),
                    "observed_nehrp": nehrp_class(float(observed[0])),
                    "predicted_nehrp": nehrp_class(float(before.mean())),
                    **_metrics(observed, before, fold_weight),
                }
            )
            observed_all.extend(observed.tolist())
            predicted_all.extend(before.tolist())
            weight_all.extend(fold_weight.tolist())

        results[model_name] = {
            "per_fold": per_fold,
            "pooled": _metrics(
                np.asarray(observed_all), np.asarray(predicted_all), np.asarray(weight_all)
            ),
        }

    report = _report(dataset, folds, results, ridge_alpha)
    (target_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _report(
    dataset: DatasetSnapshot,
    folds: list[Any],
    results: dict[str, Any],
    ridge_alpha: float,
) -> dict[str, Any]:
    """Wrap the numbers in the context that keeps them honest."""
    site_count = len(dataset.sites)
    leakage = [
        fold.test_site
        for fold in folds
        if set(dataset.groups()[fold.train_index].tolist()) & {fold.test_site}
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
        "split": "leave_one_site_out",
        "site_leakage_detected": leakage,
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
