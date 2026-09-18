# Conditioned Benchmark data

This directory is the single active gold dataset for BBB_Martins,
Bioavailability_Ma, ClinTox, and Skin_Reaction.

- `manifest.json`: task roots, target definitions, and split sizes.
- `migration_receipt.json`: exact source-path and row/hash equivalence audit.
- `<Task>/scaffold/{train,valid,test}.jsonl`: evaluation inputs.
- `<Task>/scaffold/*_molecule_condition_labels.jsonl`: detailed provenance.

The active path intentionally has no task-specific generation suffix. Internal
contract evolution is recorded in the manifest, not by accumulating competing
evaluation directories.

## BBB and Oral v2 reconstruction assets

The versioned BBB and Oral v2 gold releases live under
`data/gold_labels/<Task>/v2/scaffold/`; both `CURRENT` pointers resolve to v2.
Reconstruction inputs and audits in this directory are:

- `v2_voter_lineage/`: exact v1 vote lineage and Stage-1 preferred-UID inputs.
- `v2_stage1/`: immutable, exact-deduplicated Stage-1 snapshot and publication
  receipt used by gold v2.
- `v2_condition_migration/`: physical Oral review migration and gap audit.
- `v2_condition_ledgers/`: payload-bound terminal condition decisions for every
  surviving, label-eligible physical voter.

Gold-v2 migration receipts pin these artifacts. The edge-level
`voter_membership.parquet` and per-card `gold_label_record_index.parquet` are the
canonical physical-record mappings for L1. The active UID-level map is
`data/artifacts/starling/source_uid_levels/manifest.json` (`source_uid_levels.v3`).

The v2 aggregation gate is applied before split assignment and therefore covers
train, valid, and test uniformly. Null-condition contexts normally require 70%
agreement. An existing v1 null-condition context with the same majority label
may use the two-thirds migration-preservation floor without changing its
physical voters or vote mean; exact ties remain rejected. External-condition
contexts retain their 60% threshold.
