"""Run preprocessing across every recording in a manifest."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from mhvsr_vs30.exceptions import MhvsrVs30Error
from mhvsr_vs30.io.registry import get_reader
from mhvsr_vs30.io.resolve import resolve_request
from mhvsr_vs30.manifest.schemas import Manifest, SourceConfig
from mhvsr_vs30.preprocessing.config import PreprocessingProfile
from mhvsr_vs30.preprocessing.contracts import ProcessingOutcome
from mhvsr_vs30.preprocessing.pipeline import safe_process_recording
from mhvsr_vs30.preprocessing.store import write_curve

__all__ = ["run_profile"]


def run_profile(
    manifest: Manifest,
    source_config: SourceConfig,
    profile: PreprocessingProfile,
    workspace: Path | str = Path("."),
    curves_root: Path | str = Path("artifacts/curves"),
    index_root: Path | str = Path("artifacts/curve_indexes"),
    snapshot_hash: str | None = None,
) -> dict[str, Any]:
    """Process every recording, writing one curve directory each, plus an index."""
    assets = {asset.asset_id: asset for asset in manifest.assets}
    rows: list[dict[str, Any]] = []
    outcomes: list[ProcessingOutcome] = []

    for recording in manifest.recordings:
        asset = assets[recording.asset_id]
        try:
            request = resolve_request(manifest, recording, source_config, workspace)
            record = get_reader(recording.raw_format).read(request)
        except MhvsrVs30Error as error:
            outcome = ProcessingOutcome(
                recording_id=recording.recording_id,
                profile_id=profile.profile_id,
                status="failed",
                failure_reason=type(error).__name__,
                message=str(error),
            )
        else:
            outcome = safe_process_recording(record, profile)
            if outcome.curve is not None:
                write_curve(outcome.curve, profile, curves_root, asset.sha256, snapshot_hash)

        outcomes.append(outcome)
        row = outcome.as_dict()
        row["site_id"] = recording.site_id
        row["source_id"] = manifest.source.source_id
        row["relative_path"] = asset.relative_path
        row["input_sha256"] = asset.sha256
        row["qc_flags"] = ";".join(row.get("qc_flags", []) or [])
        rows.append(row)

    index_path = Path(index_root) / f"{manifest.source.source_id}__{profile.profile_id}.parquet"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values("recording_id").reset_index(drop=True)
    frame.to_parquet(index_path, index=False)

    status_counts = Counter(o.status for o in outcomes)
    failure_counts = Counter(o.failure_reason for o in outcomes if o.failure_reason is not None)
    return {
        "profile_id": profile.profile_id,
        "source_id": manifest.source.source_id,
        "recording_count": len(outcomes),
        "status_counts": dict(sorted(status_counts.items())),
        "failure_reasons": dict(sorted(failure_counts.items())),
        "model_ready_count": sum(
            1 for o in outcomes if o.curve is not None and o.curve.model_ready
        ),
        "extrapolated_count": sum(
            1 for o in outcomes if o.curve is not None and o.curve.extrapolated
        ),
        "curve_index": str(index_path),
    }


def write_report(summary: dict[str, Any], path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
