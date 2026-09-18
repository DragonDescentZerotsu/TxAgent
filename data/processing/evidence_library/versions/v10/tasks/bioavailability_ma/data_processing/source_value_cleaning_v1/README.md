# Bioavailability source-value cleaning v1

Stage 01 is the authoritative source-visible value layer. The immutable
`Direct_HF/records.parquet` snapshot is never edited. The directory name is
historical; Stage 00 ingests all 163,815 rows as the single physical source
`hf_bioavailability` and persists each row's `direct` or `nondirect` evidence
scope. Deterministic syntax
repairs and reviewed row-specific repairs are applied during the staged build
and recorded in `01_cleaned/source_value_cleaning_audit.parquet`.

`reviewed_repairs.jsonl` is hand-maintained. Each entry pins the source hash,
logical source row, source record ID, exact before/after fields, and supporting
evidence. `reviewed_drops.jsonl` uses the same exact row identity for records
that are known to be molecularly wrong but cannot be repaired without guessing.
A full build fails if either ledger drifts, targets a row twice, or is not fully
applied. `smiles_identity_audit.jsonl` records the supporting identity review;
the dated sample-audit JSON preserves the reproducible reviewed sample,
including rows that were retained as unresolved.

`support_text` is immutable after the shared ingestion-only Unicode and
whitespace cleanup. Reviewed repairs may target `measurement_text`, `unit_text`,
or `smiles`; support text can corroborate a repair or drop but is never
rewritten to match it. Missing-SMILES rows are retained unless the molecular
identity is independently proven wrong.

The shared deterministic cleaner handles the literal `%0020` encoded-space
artifact. Comma-bearing measurements are preserved and sent to endpoint-aware
LLM extraction instead of being rewritten as decimal or thousands separators.
Missing semantic separators require a reviewed entry. Direct-HF unitless numbers remain
source-visible as written but are not assigned a canonical unit in finalized Stage 01 and therefore
cannot enter pair buckets or assay-transfer calibration. Only a percent or
literal fraction unit present in the value itself can make a direct HF number
scalar; support text and numeric magnitude never rescue a missing unit.

Finalized Stage 01 also stores the frozen canonical endpoint, main measurement,
and unit. Oral direct-F duplicates are then removed both across and within source
datasets using exact post-repair canonical SMILES, PMID, canonical unit, and an
equal-or-within-one numeric canonical measurement. Every discarded physical row
remains in the Stage 01 audit with its retained `source_row_uid`.

All file-backed mapping ownership is declared once in
`../mapping_registry.v1.json`. Each entry states its application stage,
applicable sources, output fields, dependencies, path, and SHA-256. The Python
policy, measurement resolver, endpoint-concept loader, auxiliary-context loader,
and reference-semantics loader all resolve their defaults from that registry.
