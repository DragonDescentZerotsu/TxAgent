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
evidence. A full build fails if an entry drifts, targets a row twice, or is not
applied. Partial smoke builds may leave entries unapplied but record that fact
in `01_cleaned/source_inventory.json`.

`support_text` is immutable after the shared ingestion-only Unicode and
whitespace cleanup. Deterministic and reviewed source-value repairs may target
only `measurement_text` or `unit_text`; support text can corroborate a repair
but is never rewritten to match it.

The shared deterministic cleaner handles unambiguous decimal commas and the
literal `%0020` encoded-space artifact. It does not guess missing semantic
separators; those require a reviewed entry. Direct-HF unitless fractions remain
source-visible as written and are converted to percent only in Stage 02 for
internal comparison and assay-transfer values.
