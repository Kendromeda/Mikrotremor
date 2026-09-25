# Medan supplementary-table extraction

`extract_medan_tables.py` reads the immutable Springer supplementary DOCX and
writes the three CSV files in this directory. The 21 fragments corresponding to
manuscript Table 14 produce `masw_vs30.csv`; the 21 fragments corresponding to
manuscript Table 16 produce `hvsr.csv`.

`hvsr_masw_crosswalk.csv` joins only a published `MD` site ID that occurs once
in each source table. Coordinates are retained on both source tables and their
approximate delta is reported solely as a cross-check. No nearest-neighbour
pairing is performed. Repeated IDs remain in the source extracts and are
deliberately excluded as ambiguous until reviewed.

`reconciliation.csv` makes the exclusions inspectable. In this source version,
one exact-ID candidate (MD3) has a roughly 4 km coordinate discrepancy and is
excluded as `coordinate_conflict`; repeated or table-specific IDs are likewise
not paired.

`source_provenance.json` pins the input DOCX as SHA-256
`f522ef6780fe1f9c4b413b794509f884f0abc019afc2238fbc1ab699c51399c7` and records
the DOCX table fragments and output row counts. Regeneration must match the
committed CSV and provenance outputs exactly.

The resulting data are paired tabular measurements, not waveform examples: the
supplement does not supply raw ambient-noise recordings. Vs30 comes from Table
14 MASW and is marked by provenance in that table; it must not be conflated
with a Vs30 inferred from the Table 16 HVSR results.
