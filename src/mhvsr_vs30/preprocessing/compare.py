"""Compare two preprocessing profiles over the same recordings.

The point is not to declare a winner. It is to make the cost of each legacy
choice visible: how many curves only reach the model grid by extrapolating, and
how far the two profiles disagree where both are defensible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mhvsr_vs30.preprocessing.store import curve_directory, read_curve_arrays

__all__ = ["compare_profiles"]


def _index(index_root: Path, source_id: str, profile_id: str) -> pd.DataFrame:
    """Load one curve index, treating a failed recording as not model-ready.

    A failed recording has no curve, so its boolean columns arrive as NA. That
    is the correct storage but the wrong thing to reason with here.
    """
    frame = pd.read_parquet(Path(index_root) / f"{source_id}__{profile_id}.parquet")
    for column in ("model_ready", "extrapolated"):
        if column in frame:
            frame[column] = frame[column].fillna(False).astype(bool)
    return frame


def compare_profiles(
    source_id: str,
    left_profile: str,
    right_profile: str,
    index_root: Path | str = Path("artifacts/curve_indexes"),
    curves_root: Path | str = Path("artifacts/curves"),
) -> dict[str, Any]:
    """Summarise where two profiles agree, differ, and why."""
    left = _index(Path(index_root), source_id, left_profile).set_index("recording_id")
    right = _index(Path(index_root), source_id, right_profile).set_index("recording_id")
    shared = sorted(set(left.index) & set(right.index))

    both_ready: list[str] = []
    ratios: list[float] = []
    for recording_id in shared:
        if not (left.loc[recording_id, "model_ready"] and right.loc[recording_id, "model_ready"]):
            continue
        try:
            a = read_curve_arrays(curve_directory(curves_root, left_profile, recording_id))
            b = read_curve_arrays(curve_directory(curves_root, right_profile, recording_id))
        except FileNotFoundError:
            continue
        both_ready.append(recording_id)
        ratios.append(float(np.max(np.abs(np.log(a["model_amplitude"] / b["model_amplitude"])))))

    def side(frame: pd.DataFrame, profile_id: str) -> dict[str, Any]:
        return {
            "profile_id": profile_id,
            "recordings": len(frame),
            "status_counts": {k: int(v) for k, v in frame["status"].value_counts().items()},
            "model_ready": int(frame["model_ready"].sum()),
            "extrapolated": int(frame["extrapolated"].sum()),
            "mean_accepted_windows": (
                float(frame["accepted_window_count"].mean())
                if "accepted_window_count" in frame
                else None
            ),
        }

    return {
        "source_id": source_id,
        "left": side(left, left_profile),
        "right": side(right, right_profile),
        "comparable_recordings": len(both_ready),
        "max_abs_log_ratio": {
            "max": max(ratios) if ratios else None,
            "median": float(np.median(ratios)) if ratios else None,
            "note": (
                "Largest absolute natural-log difference between the two model "
                "feature vectors, per recording. A value near 0 means the profiles "
                "agree; large values mean the legacy choices dominate the curve."
            ),
        },
        "only_model_ready_in_left": sorted(
            set(left.index[left["model_ready"]]) - set(right.index[right["model_ready"]])
        ),
        "only_model_ready_in_right": sorted(
            set(right.index[right["model_ready"]]) - set(left.index[left["model_ready"]])
        ),
    }
