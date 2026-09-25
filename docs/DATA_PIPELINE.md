# Data pipeline: manifests and canonical readers

This document covers the `mhvsr_vs30` package under `src/`, which builds
recording-level manifests from published datasets, parses raw files,
preprocesses mHVSR curves, and evaluates small baseline models.

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

## Repairing defective source data

Published data is sometimes broken. USGS ARRA has one recording whose
acquisition was cut mid-sample, leaving a final line with one column instead of
three. Working around that is a judgement call about real bytes, so permission
is granted per file and pinned to its checksum:

```yaml
repairs:
  - relative_path: "CI.CCC/.../20110616181108.Q289.txt"
    sha256: "575f9bd09772..."
    action: drop_incomplete_final_row
    expected_samples_dropped: 1
    reason: >-
      Acquisition was cut mid-sample. Line 435214 holds only the vertical value.
```

If that file is ever re-downloaded or re-extracted, the checksum stops matching
and the build refuses rather than reusing an audit written about different data.
A record built from repaired input reports `structural_qc: passed_with_repair`
and carries the action, the reason and the number of samples discarded, so no
downstream stage can mistake it for clean input.

## Asset format hints

`media_type` says what a file means; `format_hint` says how it is encoded. They
are separate because a SEG-2 MASW gather and an HP SDF SASW sweep are both
surface-wave surveys, yet nothing can read one with the other reader. Values:
`ascii_3c`, `miniseed`, `sac`, `gcf`, `seg2`, `hp_sdf`. An asset that no rule
claims carries no hint at all, because the extension is not evidence.

## Preprocessing profiles

Raw components become an mHVSR curve under one of two profiles, because two
different questions are being asked and one configuration cannot answer both.

| | `published_compat_v1` | `training_v1` |
|---|---|---|
| Purpose | compare against the published models | produce new training data |
| Window length | duration / 35, so it varies per record | fixed 60 s |
| Window rejection | none (the notebook rejects by hand) | deterministic frequency-domain |
| Model grid | 0.3-50 Hz, 35 points | same |
| Extrapolation | allowed, reproducing legacy behaviour, always flagged | forbidden |

Every value in `published_compat_v1` was read out of the reference notebooks,
including `significant_cycles = 15`, Tukey 0.2, Konno-Ohmachi bandwidth 40, the
0.05-50 Hz / 256 point internal grid, and the `fill_value="extrapolate"` call
that fills the model grid below the usable frequency floor. Curves from that
profile carry `extrapolated=true` and are not automatically eligible as training
data.

The spectral core (detrend, taper, FFT, Konno-Ohmachi, horizontal combination)
is delegated to hvsrpy 2.0.0 rather than reimplemented. Reimplementing it would
guarantee the compatibility profile stopped matching the library that produced
the published models, and would make the two profiles incomparable to each
other. What this project owns is the policy around that core: window length,
which frequencies are admissible, which windows survive, and what is written
down.

```bash
uv run mhvsr-vs30 preprocess run --manifest artifacts/manifests/usgs_arra_v1 --config configs/datasets/usgs_arra.yaml --profile configs/preprocessing/training_v1.yaml --report artifacts/reports/usgs_arra_v1__training_v1.json
```

```bash
uv run mhvsr-vs30 preprocess compare --source-id usgs_arra_2013 --left published_compat_v1 --right training_v1
```

Each recording produces `curve.npz`, `qc.json` and `provenance.json` under
`artifacts/curves/<profile>/<recording_id>/`, written atomically through a
staging directory. Arrays load with `allow_pickle=False`. A run also writes
`artifacts/curve_indexes/<source>__<profile>.parquet` for corpus-level analysis.

A `preprocess run` exits non-zero when any recording failed to produce a curve.
That is intentional: a failed recording is recorded in the index with its
reason, and an operator should have to notice.

## Dataset snapshots and the vertical slice

A snapshot joins model-ready curves to their sites and labels by identifier
only. Nothing is matched on file name or on proximity, so a row exists only when
a curve, a recording and a cited label agree on the same site.

```bash
uv run mhvsr-vs30 dataset build --manifest artifacts/manifests/usgs_arra_v1 --profile configs/preprocessing/training_v1.yaml --output artifacts/datasets/usgs_arra_v1__training_v1
```

Two guards are on by default and both have to be switched off deliberately:

- Only labels marked `independent_of_hvsr` enter. A Vs30 inverted from the same
  HVSR the model reads is circular, and a score computed against it would be
  meaningless. `--allow-dependent-labels` overrides this.
- Each site carries one unit of weight however many times it was recorded, so a
  site with six recordings cannot outvote a site with one.

```bash
uv run mhvsr-vs30 train smoke --manifest artifacts/manifests/usgs_arra_v1 --profile configs/preprocessing/training_v1.yaml --output artifacts/runs/loso --report artifacts/reports/loso.json
```

The split is leave-one-site-out, because a physical location is the unit of
independence: six recordings of one site are six views of the same ground. Only
two estimators run, a weighted median and ridge on log Vs30, both deliberately
dull. With three independent sites, anything with capacity to spare would fit
the fold rather than the physics.

Every report is stamped `evaluation_scope: engineering_only`,
`promotion_eligible: false` and `n_sites`, and carries a sentence saying why the
numbers are not a scientific result. That stamp is the point of the command: it
proves the pipeline runs without a notebook and does not leak a site across the
split, and it proves nothing else.

## Medan table-only evaluation

The Medan supplement supplies 185 audited site-ID matches between published
HVSR `f0`/`A0` values and MASW Vs30 values. It does not supply the raw
three-component recordings needed by the 35-point curve pipeline, so its
evaluation is deliberately separate:

```bash
uv run mhvsr-vs30 tabular evaluate-medan --report artifacts/reports/medan_supplement_v1__spatial_tabular.json
```

The evaluator holds out five equal-sized longitude bands (37 sites each) and
removes training sites within a 1 km longitude buffer around each held-out band.
Within each fold it fits a median-Vs30 baseline and a ridge regression using
only log `f0` and log `A0`; the output lists the train/test sites, longitude
ranges, buffer exclusions, and errors. All 185 sites come from one study, so
the report is stamped
`exploratory_single_study`, `external_validation: false`, and
`promotion_eligible: false`. It cannot establish performance in another study
or country. The curated source tables and join rules are described in
`datasets/processed/medan_supplement_v1/CURATION.md`.

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
