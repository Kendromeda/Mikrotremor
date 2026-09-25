"""Exploratory spatial evaluation of audited Medan HVSR/MASW table pairs.

This uses published f0 and A0 values, not waveform-derived 35-point curves.
One study cannot establish transfer to a different study or geography.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mhvsr_vs30.exceptions import MhvsrVs30Error


class MedanDataError(MhvsrVs30Error):
    """An audited Medan pair or evaluation input is invalid."""


def _unique_sites(frame: pd.DataFrame, name: str) -> None:
    if "site_id" not in frame or frame["site_id"].isna().any():
        raise MedanDataError(f"{name} needs non-null site_id values")
    if frame["site_id"].duplicated().any():
        raise MedanDataError(f"{name} has duplicate site_id values")


def load_medan_pairs(directory: Path | str) -> pd.DataFrame:
    """Join only sites accepted in the audited exact-ID crosswalk."""
    root = Path(directory)
    try:
        hvsr = pd.read_csv(root / "hvsr.csv", dtype={"site_id": str})
        masw = pd.read_csv(root / "masw_vs30.csv", dtype={"site_id": str})
        crosswalk = pd.read_csv(root / "hvsr_masw_crosswalk.csv", dtype={"site_id": str})
    except (OSError, pd.errors.ParserError) as exc:
        raise MedanDataError(f"cannot read Medan curation in {root}: {exc}") from exc

    _unique_sites(crosswalk, "crosswalk")
    required_hvsr = {"site_id", "latitude", "longitude", "f0_hz", "a0"}
    required_masw = {"site_id", "vs30_mps"}
    required_crosswalk = {"site_id", "coordinate_delta_m_approx", "match_method"}
    for name, frame, required in (
        ("HVSR", hvsr, required_hvsr),
        ("MASW", masw, required_masw),
        ("crosswalk", crosswalk, required_crosswalk),
    ):
        missing = required - set(frame)
        if missing:
            raise MedanDataError(f"{name} lacks columns: {sorted(missing)}")

    delta = pd.to_numeric(crosswalk["coordinate_delta_m_approx"], errors="coerce")
    if delta.isna().any() or (delta < 0).any() or (delta > 100).any():
        raise MedanDataError("crosswalk has invalid or >100 m coordinate separation")
    if not crosswalk["match_method"].eq("exact_site_id_coordinate_checked").all():
        raise MedanDataError("crosswalk contains an unaudited match method")

    matched_hvsr = hvsr.loc[hvsr["site_id"].isin(crosswalk["site_id"]), list(required_hvsr)]
    matched_masw = masw.loc[masw["site_id"].isin(crosswalk["site_id"]), list(required_masw)]
    _unique_sites(matched_hvsr, "matched HVSR")
    _unique_sites(matched_masw, "matched MASW")
    selected = (
        crosswalk[["site_id"]]
        .merge(matched_hvsr, on="site_id", how="left", validate="one_to_one")
        .merge(matched_masw, on="site_id", how="left", validate="one_to_one")
    )
    if selected.isna().any().any():
        raise MedanDataError("crosswalk references a missing or incomplete source row")
    return selected.sort_values("site_id").reset_index(drop=True)


def _validated_arrays(
    frame: pd.DataFrame, n_regions: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    needed = {"site_id", "latitude", "longitude", "f0_hz", "a0", "vs30_mps"}
    missing = needed - set(frame)
    if missing:
        raise MedanDataError(f"evaluation lacks columns: {sorted(missing)}")
    _unique_sites(frame, "evaluation")
    if n_regions < 2 or len(frame) < n_regions * 2:
        raise MedanDataError("need at least two sites per spatial region")
    try:
        coordinates = frame[["latitude", "longitude"]].to_numpy(dtype=float)
        features = frame[["f0_hz", "a0"]].to_numpy(dtype=float)
        targets = frame["vs30_mps"].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise MedanDataError("features, coordinates and Vs30 must be numeric") from exc
    if not all(np.isfinite(values).all() for values in (coordinates, features, targets)):
        raise MedanDataError("evaluation contains missing or non-finite numeric values")
    if (np.abs(coordinates[:, 0]) > 90).any() or (np.abs(coordinates[:, 1]) > 180).any():
        raise MedanDataError("latitude or longitude is outside the valid range")
    if (features <= 0).any() or (targets <= 0).any():
        raise MedanDataError("f0, A0 and Vs30 must be positive")
    return coordinates, features, targets


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    return {
        "mae_mps": float(np.mean(np.abs(error))),
        "median_absolute_error_mps": float(np.median(np.abs(error))),
        "mean_absolute_log_error": float(np.mean(np.abs(np.log(predicted / observed)))),
        "bias_log": float(np.mean(np.log(predicted / observed))),
    }


def evaluate_medan(
    frame: pd.DataFrame, *, n_regions: int = 5, buffer_km: float = 1.0
) -> dict[str, Any]:
    """Hold out each coordinate-derived region, fitting every model within a fold."""
    coordinates, features, targets = _validated_arrays(frame, n_regions)
    if not np.isfinite(buffer_km) or buffer_km < 0:
        raise MedanDataError("buffer_km must be finite and non-negative")
    km_per_lon_degree = 111.32 * np.cos(np.deg2rad(np.abs(coordinates[:, 0]).max()))
    if km_per_lon_degree <= 1e-6:
        raise MedanDataError("longitude-band buffer is undefined near the poles")
    buffer_degrees = buffer_km / km_per_lon_degree
    site_ids = frame["site_id"].astype(str).to_numpy()
    # Equal-sized east-west bands avoid a remote point becoming a one-site fold.
    order = np.lexsort((site_ids, coordinates[:, 0], coordinates[:, 1]))
    regions = np.empty(len(frame), dtype=int)
    regions[order] = np.arange(len(frame)) * n_regions // len(frame)
    predictions = {
        "median_vs30": np.full(len(frame), np.nan),
        "ridge_log_f0_a0": np.full(len(frame), np.nan),
    }
    folds: list[dict[str, Any]] = []
    for region in sorted(np.unique(regions)):
        test = regions == region
        test_min_lon = float(coordinates[test, 1].min())
        test_max_lon = float(coordinates[test, 1].max())
        train = ~test & (
            (coordinates[:, 1] < test_min_lon - buffer_degrees)
            | (coordinates[:, 1] > test_max_lon + buffer_degrees)
        )
        train_sites = sorted(site_ids[train].tolist())
        test_sites = sorted(site_ids[test].tolist())
        if train.sum() < 2 or not test.any():
            raise MedanDataError("a buffered spatial fold needs at least two training sites")
        predictions["median_vs30"][test] = float(np.median(targets[train]))
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        ridge.fit(np.log(features[train]), np.log(targets[train]))
        predictions["ridge_log_f0_a0"][test] = np.exp(ridge.predict(np.log(features[test])))
        folds.append(
            {
                "region": int(region),
                "n_train_sites": len(train_sites),
                "n_test_sites": len(test_sites),
                "n_buffer_excluded_sites": int((~test & ~train).sum()),
                "test_longitude_min": test_min_lon,
                "test_longitude_max": test_max_lon,
                "train_sites": train_sites,
                "test_sites": test_sites,
                "models": {
                    name: _metrics(targets[test], values[test])
                    for name, values in predictions.items()
                },
            }
        )

    leaks = sorted(
        {site for fold in folds for site in set(fold["train_sites"]) & set(fold["test_sites"])}
    )
    return {
        "dataset": "medan_supplement_v1",
        "features": ["f0_hz", "a0"],
        "target": "masw_vs30_mps",
        "split": "leave_one_longitude_band_out",
        "buffer_km": buffer_km,
        "n_regions": n_regions,
        "n_sites": len(frame),
        "n_studies": 1,
        "evaluation_scope": "exploratory_single_study",
        "external_validation": False,
        "promotion_eligible": False,
        "site_leakage_detected": leaks,
        "limitation": (
            "All sites come from one Medan study. Buffered longitude-band holdout tests "
            "within-study geographic interpolation only; no external study or country "
            "has been evaluated."
        ),
        "folds": folds,
        "models": {name: _metrics(targets, values) for name, values in predictions.items()},
    }


def write_medan_report(report: dict[str, Any], output: Path | str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
