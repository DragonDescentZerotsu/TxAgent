# Current Starling level-record collaborator dataset

This Git-tracked snapshot has two deliberately separate tables for each task.
It is stored as ordinary Parquet files so a collaborator receives it with a
normal repository clone; no archive unpacking or Git LFS installation is
required. Only the current adopted snapshot belongs here.

The current public naming model is `source_group_id -> family_key -> level`.
`legacy_family_id` is included only to locate older catalogs and traces; it is
not a second semantic classification.

- `source_record_level_membership.parquet` (or the same-named directory without
  the suffix, containing ordinary Parquet parts when the ledger exceeds 90 MB) contains every current
  `retrieval_eligible=True` source record whose purity-overlay `group_id` maps
  to a current progressive level. This is the static, pre-split level ledger.
- `<split>/indexed_representative_cards.parquet` contains compact references to
  the exact record cards materialized in that split's frozen assay-molecule
  index after direct heldout filtering. The index keeps at most three cards per
  assay×molecule; card text stays normalized in the source ledger rather than
  being duplicated in every split table.

Join the two tables on `molecule_id` plus `card_fingerprint_sha256` when source
record lineage is needed. More than one canonical record can have the same
visible card payload; all matching source rows remain in the membership table.
Export fails unless every indexed-card key resolves to at least one source row.

Read `manifest.json` first. It records row counts and SHA-256 hashes for every
file and for the frozen inputs. Pass each table's `path` to `pyarrow.parquet.read_table`;
both a single file and a directory of parts have the same schema. Sharded tables
list individual file hashes under `parts`. A physical assay can occur at several record
levels; `assay_first_level` is catalog metadata, not a retrieval gate.

Regenerate the complete directory from the repository root with:

```bash
python -m tools.chembl_tool.paper_experiments.export_current_starling_level_records
```

The exporter first verifies every current input against the frozen retrieval
contract. Commit a regenerated snapshot only when that current lineage changes;
do not keep parallel versioned copies of this directory.

These are candidate surfaces, not the cards selected for a particular query.
Exact model-visible cards for one completed query are stored in that run's
`<task>/queries/query_idxNNNNN/levels/level_N/prepared.json` under
`active_evidence`; `new_card_ids` identifies what was newly unlocked at that
level.
