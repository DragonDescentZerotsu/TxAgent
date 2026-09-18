# Gold labels

Active benchmark labels are versioned by task. Historical label constructions
are retained under `data/gold_labels/legacy/`; they are provenance inputs, not
active benchmark defaults.

## Versioned level mappings

Gold-release-specific evidence-library level mappings are published at
`<Task>/level_mappings/<gold-version>/`. The evidence-library pipeline owns and
produces the scientific assignment of L1 and L3+; gold-label code must not derive
or rewrite it from voter membership. Gold membership may be used only as a
validation input for the evidence-library publication. Each gold version gets a
separate immutable mapping, while the evidence pipeline keeps L3+ consistent
across gold versions.

The preserved publications are Ames v1, BBB v1/v2, Bioavailability v1/v2,
Carcinogens v1, DILI v1, and Skin Reaction v1. `CURRENT` remains v1 for every
task. ClinTox is intentionally excluded because it is a legacy task.

Each publication retains the exact Tianang-derived input as
`original_level_mapping.parquet` (or the `original_level_mapping/` Parquet
dataset for sharded tasks) and exposes the reviewed result as
`level_mapping.parquet` or `level_mapping/`. BBB is byte-identical to the
original. Bioavailability changes only reviewed voter UIDs: v1 has 13 L2-to-L1
promotions plus 13 additions, and v2 has 11 additions. Ames voter identity is
independently checked against its accepted-record votes; the other unchanged
tasks preserve their pinned Tianang L1 contracts. No L3+ assignment changes
between gold versions.

`level_mappings.v1.json` is the preserved multi-task publication index for the
gold-owned v1 mappings. It points into the task-owned directories and contains
no copied mapping rows. BBB and Bioavailability runtime consumers instead use
the release-owned mappings indexed by
`data/evidence_libraries/level_mappings.v1.json`; the other four active tasks
keep their gold-owned mappings until reviewed replacements arrive. Historical
artifact manifests remain audit provenance, not active runtime inputs.
