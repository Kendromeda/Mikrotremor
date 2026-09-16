# Data pipeline: manifests and canonical readers

This document covers the `mhvsr_vs30` package under `src/`, which builds
recording-level manifests from published datasets and parses their raw files
into one in-memory representation. It does no preprocessing and no training.

The notebooks (`low_dim_models.ipynb`, `high_dim_models.ipynb`) remain the
reference for inference and do not depend on this package.

## Principles

- **Raw data is immutable.** Discovery reads bytes to checksum them and stats to
  size them. Nothing under `datasets/` or `datasets_global/` is ever written.
- **Claims come from documents, not from file names.** A Vs30 label exists only
  because a source configuration cites a document for it.
- **Counts are contracts.** If a build finds a different number of sites,
  recordings or assets than the configuration expects, the build fails.
- **Identifiers are derived, not assigned.** The same tree rebuilds to the same
  ids, so two snapshots can be diffed row by row.
- **Readers raise, never return None.** Each failure mode has its own exception
  type so a QC report can count causes instead of grepping messages.

## Install

The package targets Python 3.12 and pins its dependency graph in `uv.lock`:

```bash
uv sync
```

## Build and check a manifest

```bash
uv run mhvsr-vs30 manifest build --config configs/datasets/usgs_arra.yaml --output artifacts/manifests/usgs_arra_v1
```

```bash
uv run mhvsr-vs30 manifest validate artifacts/manifests/usgs_arra_v1 --config configs/datasets/usgs_arra.yaml
```

A manifest directory holds five tables (`sources`, `sites`, `assets`,
`recordings`, `labels`) as Parquet, four of them also as CSV for inspection,
plus `summary.json` and a `snapshot.json` that pins the config checksum, the
code commit and the row counts.

## Parse recordings

```bash
uv run mhvsr-vs30 recording validate-all --manifest artifacts/manifests/usgs_arra_v1 --config configs/datasets/usgs_arra.yaml --report artifacts/reports/usgs_arra_v1_recording_qc.json
```

```bash
uv run mhvsr-vs30 recording inspect --manifest artifacts/manifests/usgs_arra_v1 --config configs/datasets/usgs_arra.yaml --recording-id 0e1f5de2b2c7
```

Readers are selected by the `raw_format` recorded in the manifest, never by file
extension. Registered formats: `usgs_ascii_3c`, `miniseed`, `sac_bundle`, `gcf`,
`processed_hvsr`, `seg2`.

Raw three-component data becomes a `ThreeComponentRecord` with read-only arrays.
A published H/V curve becomes an `HvsrCurve` marked `input_level=processed_curve`
and is never mixed with raw waveforms. A SEG-2 file becomes a `SurveyInspection`
unless the configuration states which trace is Z, N and E.

## Adding a source

Write a YAML file in `configs/datasets/`. It records only claims you verified in
the release itself, with `asset_rules` describing what each file family is,
`expectations` giving the counts a correct build must reproduce, and one entry
per site carrying its label and the document that supports it. See
`configs/datasets/usgs_arra.yaml`, which cites the FILE FORMATS document shipped
in each USGS ARRA site archive and Table 3 of Open-File Report 2013-1102.

## Tests

```bash
uv run pytest
```

Tests marked `requires_real_data` skip themselves when the dataset trees are
absent. To run only the tests that need no data:

```bash
uv run pytest -m "not requires_real_data"
```
