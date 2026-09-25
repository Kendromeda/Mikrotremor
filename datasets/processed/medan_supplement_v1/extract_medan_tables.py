"""Extract the audited MASW and HVSR tables from the Medan supplement.

The DOCX is immutable source material.  This extractor writes normalised CSV
tables beside itself and intentionally pairs rows only by the published site
identifier; nearest-coordinate matching is not a valid substitute.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from docx import Document

DEFAULT_RAW_DOCUMENT = Path(
    "datasets/source_08_medan_supplement/raw/40677_2022_227_MOESM1_ESM.docx"
)
DEFAULT_OUTPUT = Path("datasets/processed/medan_supplement_v1")

# The supplemental document has 56 OOXML tables. Tables 13--33 are the 21
# district fragments of manuscript Table 14 (MASW Vs30); tables 35--55 are
# the 21 district fragments of manuscript Table 16 (HVSR).
MASW_TABLE_INDEXES = range(12, 33)
HVSR_TABLE_INDEXES = range(34, 55)


def _text(cell: Any) -> str:
    return str(cell.text).strip().replace("\n", " ")


def _source_rows(document: Document, indexes: range) -> list[tuple[int, list[str]]]:
    extracted: list[tuple[int, list[str]]] = []
    for index in indexes:
        table = document.tables[index]
        for row in table.rows[2:]:
            values = [_text(cell) for cell in row.cells]
            if values and values[0]:
                extracted.append((index + 1, values))
    return extracted


def _write_csv(
    output: Path, name: str, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    target = output / name
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _coordinate_delta_m(left: dict[str, object], right: dict[str, object]) -> float:
    # Sufficient only to detect transcription/crosswalk mistakes at this scale;
    # it is not used to create matches.
    latitude_delta = abs(float(left["latitude"]) - float(right["latitude"])) * 111_320
    longitude_delta = abs(float(left["longitude"]) - float(right["longitude"])) * 111_320
    return round((latitude_delta**2 + longitude_delta**2) ** 0.5, 3)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(raw_document: Path, output: Path) -> None:
    """Extract deterministic tables from one verified Medan supplementary DOCX."""
    document = Document(raw_document)
    output.mkdir(parents=True, exist_ok=True)

    masw = [
        {
            "site_id": values[1],
            "latitude": values[2],
            "longitude": values[3],
            "vs30_mps": values[4],
            "site_class": values[5],
            "source_table": "Table 14",
            "docx_table_index": table_index,
            "source_row_number": values[0],
        }
        for table_index, values in _source_rows(document, MASW_TABLE_INDEXES)
    ]
    hvsr = [
        {
            "site_id": values[1],
            "latitude": values[2],
            "longitude": values[3],
            "f0_hz": values[4],
            "a0": values[5],
            "tdom_s": values[6],
            "kg": values[7],
            "gss_1000yr": values[8],
            "gss_2500yr": values[9],
            "source_table": "Table 16",
            "docx_table_index": table_index,
            "source_row_number": values[0],
        }
        for table_index, values in _source_rows(document, HVSR_TABLE_INDEXES)
    ]
    masw_by_id: dict[str, list[dict[str, object]]] = {}
    hvsr_by_id: dict[str, list[dict[str, object]]] = {}
    for row in masw:
        masw_by_id.setdefault(str(row["site_id"]), []).append(row)
    for row in hvsr:
        hvsr_by_id.setdefault(str(row["site_id"]), []).append(row)

    # A repeated published ID is ambiguity, even if its two rows happen to be
    # byte-for-byte identical. Preserve those rows in their source table but
    # exclude the ID from the paired corpus until a human resolves it.
    unique_on_both_sides = sorted(
        site_id
        for site_id in set(masw_by_id) & set(hvsr_by_id)
        if len(masw_by_id[site_id]) == 1 and len(hvsr_by_id[site_id]) == 1
    )
    # All ordinary coordinate differences are below 72 m, which is consistent
    # with Table 16's three-decimal coordinate rounding. MD3 differs by about
    # 4 km despite its shared ID, so it is retained in reconciliation.csv as a
    # coordinate conflict rather than silently paired.
    paired_ids = [
        site_id
        for site_id in unique_on_both_sides
        if _coordinate_delta_m(masw_by_id[site_id][0], hvsr_by_id[site_id][0]) <= 100.0
    ]

    paired = [
        {
            "site_id": site_id,
            "match_method": "exact_site_id_coordinate_checked",
            "coordinate_delta_m_approx": _coordinate_delta_m(
                masw_by_id[site_id][0], hvsr_by_id[site_id][0]
            ),
            "masw_docx_table_index": masw_by_id[site_id][0]["docx_table_index"],
            "hvsr_docx_table_index": hvsr_by_id[site_id][0]["docx_table_index"],
        }
        for site_id in sorted(paired_ids, key=lambda value: int(value.removeprefix("MD")))
    ]

    reconciliation = []
    all_site_ids = sorted(
        set(masw_by_id) | set(hvsr_by_id), key=lambda value: int(value.removeprefix("MD"))
    )
    for site_id in all_site_ids:
        masw_count = len(masw_by_id.get(site_id, []))
        hvsr_count = len(hvsr_by_id.get(site_id, []))
        if masw_count == 0:
            status = "only_hvsr"
        elif hvsr_count == 0:
            status = "only_masw"
        elif masw_count != 1 or hvsr_count != 1:
            status = "duplicate_published_site_id"
        elif site_id not in paired_ids:
            status = "coordinate_conflict"
        else:
            status = "paired"
        reconciliation.append(
            {
                "site_id": site_id,
                "masw_row_count": masw_count,
                "hvsr_row_count": hvsr_count,
                "reconciliation_status": status,
            }
        )

    _write_csv(output, "masw_vs30.csv", list(masw[0]), masw)
    _write_csv(output, "hvsr.csv", list(hvsr[0]), hvsr)
    _write_csv(output, "hvsr_masw_crosswalk.csv", list(paired[0]), paired)
    _write_csv(output, "reconciliation.csv", list(reconciliation[0]), reconciliation)
    (output / "source_provenance.json").write_text(
        json.dumps(
            {
                "source_path": raw_document.as_posix(),
                "source_sha256": _sha256(raw_document),
                "source_table_14_docx_indexes": list(MASW_TABLE_INDEXES.start + 1 + index for index in range(len(MASW_TABLE_INDEXES))),
                "source_table_16_docx_indexes": list(HVSR_TABLE_INDEXES.start + 1 + index for index in range(len(HVSR_TABLE_INDEXES))),
                "masw_row_count": len(masw),
                "hvsr_row_count": len(hvsr),
                "paired_row_count": len(paired),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_DOCUMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    extract(args.input, args.output)


if __name__ == "__main__":
    main()
