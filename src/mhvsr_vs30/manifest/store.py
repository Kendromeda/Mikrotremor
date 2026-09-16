"""Reading and writing manifest tables.

Parquet is the table of record; CSV exists so a human can open a manifest
without tooling. Both are written from the same sorted rows through an explicit
schema, so a rebuild of unchanged data produces identical bytes.
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from mhvsr_vs30 import __version__
from mhvsr_vs30.contracts import ManifestRecord, RawAsset, Recording, Site, Source, Vs30Label
from mhvsr_vs30.exceptions import ManifestValidationError
from mhvsr_vs30.manifest.schemas import (
    CSV_TABLE_NAMES,
    MANIFEST_VERSION,
    TABLE_NAMES,
    Manifest,
    SourceConfig,
)

__all__ = ["read_manifest", "write_manifest"]

_STRING = pa.string()

TABLE_SCHEMAS: dict[str, pa.Schema] = {
    "sources": pa.schema(
        [
            ("source_id", _STRING),
            ("title", _STRING),
            ("doi", _STRING),
            ("license", _STRING),
            ("root_path", _STRING),
            ("source_url", _STRING),
        ]
    ),
    "sites": pa.schema(
        [
            ("site_id", _STRING),
            ("source_id", _STRING),
            ("latitude", pa.float64()),
            ("longitude", pa.float64()),
            ("elevation_m", pa.float64()),
            ("country", _STRING),
            ("location_accuracy_m", pa.float64()),
            ("site_code", _STRING),
        ]
    ),
    "assets": pa.schema(
        [
            ("asset_id", _STRING),
            ("source_id", _STRING),
            ("relative_path", _STRING),
            ("media_type", _STRING),
            ("byte_size", pa.int64()),
            ("sha256", _STRING),
        ]
    ),
    "recordings": pa.schema(
        [
            ("recording_id", _STRING),
            ("site_id", _STRING),
            ("asset_id", _STRING),
            ("raw_format", _STRING),
            ("sampling_rate_hz", pa.float64()),
            ("start_time", pa.timestamp("us", tz="UTC")),
            ("duration_s", pa.float64()),
            ("channel_mapping", _STRING),
            ("units", _STRING),
            ("segment", _STRING),
        ]
    ),
    "labels": pa.schema(
        [
            ("label_id", _STRING),
            ("site_id", _STRING),
            ("vs30_mps", pa.float64()),
            ("vs30_std_mps", pa.float64()),
            ("label_method", _STRING),
            ("label_independence", _STRING),
            ("reference", _STRING),
            ("reference_locator", _STRING),
        ]
    ),
}

_MODELS: dict[str, type[ManifestRecord]] = {
    "sources": Source,
    "sites": Site,
    "assets": RawAsset,
    "recordings": Recording,
    "labels": Vs30Label,
}

_SORT_KEYS = {
    "sources": "source_id",
    "sites": "site_id",
    "assets": "relative_path",
    "recordings": "recording_id",
    "labels": "label_id",
}


def _rows(manifest: Manifest, table: str) -> list[dict[str, Any]]:
    key = _SORT_KEYS[table]
    rows = [record.to_row() for record in manifest.table(table)]
    return sorted(rows, key=lambda row: str(row[key]))


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return str(value)


def _write_csv(path: Path, schema: pa.Schema, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(schema.names)
        for row in rows:
            writer.writerow([_csv_value(row[name]) for name in schema.names])


def _code_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def write_manifest(
    manifest: Manifest,
    output_dir: Path | str,
    config: SourceConfig,
    created_at: datetime | None = None,
) -> Path:
    """Write the five tables plus a summary and a snapshot.

    The snapshot is what makes two builds comparable: it pins the config
    checksum, the code commit and the row counts that produced these files.
    """
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    for table in TABLE_NAMES:
        schema = TABLE_SCHEMAS[table]
        rows = _rows(manifest, table)
        arrow_table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(arrow_table, target / f"{table}.parquet", compression="snappy")
        if table in CSV_TABLE_NAMES:
            _write_csv(target / f"{table}.csv", schema, rows)

    summary = manifest.summary()
    (target / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    snapshot = {
        "manifest_version": MANIFEST_VERSION,
        "tool_version": __version__,
        "source_id": manifest.source.source_id,
        "config_path": config.config_path,
        "config_hash": config.config_hash,
        "source_count": 1,
        "site_count": len(manifest.sites),
        "asset_count": len(manifest.assets),
        "recording_count": len(manifest.recordings),
        "label_count": len(manifest.labels),
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "code_commit": _code_commit(),
    }
    (target / "snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def read_manifest(manifest_dir: Path | str) -> Manifest:
    """Rebuild the in-memory manifest from its Parquet tables."""
    source_dir = Path(manifest_dir)
    tables: dict[str, list[Any]] = {}

    for table in TABLE_NAMES:
        path = source_dir / f"{table}.parquet"
        if not path.is_file():
            raise ManifestValidationError(f"missing manifest table: {path}")
        model = _MODELS[table]
        tables[table] = [model.from_row(row) for row in pq.read_table(path).to_pylist()]

    sources = tables["sources"]
    if len(sources) != 1:
        raise ManifestValidationError(
            f"{source_dir} holds {len(sources)} sources; one manifest describes exactly one"
        )

    return Manifest(
        source=sources[0],
        sites=tuple(tables["sites"]),
        assets=tuple(tables["assets"]),
        recordings=tuple(tables["recordings"]),
        labels=tuple(tables["labels"]),
    )
