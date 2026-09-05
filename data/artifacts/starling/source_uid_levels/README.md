# Upstream level assignments keyed by acquisition UID

This is the routing-reference import from main commit
`00800a5b890440ac52e4e7689c4a157d0ff2a365` (2026-09-04).
It does not merge main or change this branch's retrieval, gold labels, or published traces.

The byte-exact upstream package is retained at
`artifacts/chembl_tool/starling/current_level_records/`, including its original
README and manifest. All nine Parquet hashes match that manifest. Paths and
regeneration instructions in that original README describe the upstream pipeline.
The package first appeared in `23ded557a94106a364542da6800e7fa9b3de87d9`;
the imported tip includes the subsequent Skin revision.

## UID lookup

This directory contains one Parquet per task with columns:
`source_row_uid`, `canonical_record_id`, `source_group_id`, `family_key`, `level`.
Each imported membership row has one verified acquisition UID, and each UID has
exactly one level and family in this snapshot.

| Task | Mapped records / unique UIDs | Levels | Unmatched / conflicting |
|---|---:|---|---:|
| BBB | 498,556 | L1–L5 | 0 / 0 |
| Oral bioavailability | 435,472 | L1–L6 | 0 / 0 |
| Skin | 66,967 | L1–L3 | 0 / 0 |

```python
import pandas as pd

reference = pd.read_parquet("data/artifacts/starling/source_uid_levels/bbb_martins.parquet")
uid_to_level = reference.set_index("source_row_uid")["level"]
# For an existing dataframe containing source_row_uid:
audited = records.merge(reference, on="source_row_uid", how="left", validate="many_to_one")
```

The UID bridge uses exact frozen lineage:

`canonical_record_id + source_id + source_record_id`
→ frozen Stage-03 `cleaned_record_id`
→ UID ledger `legacy_cleaned_record_id`
→ permanent `source_row_uid`.

Source IDs, source record IDs, and acquisition row numbers are checked together.
`source_record_id` alone is unsafe: identifiers such as `ext_1` repeat.
Our locally rebuilt V7 canonical IDs differ for 4,515 imported records, so the
builder uses the exact upstream Stage-03 archives, verifies their hashes, and
reads them in memory. It does not reconstruct UIDs from molecule structure or text.

## Where source_group_id comes from

The field is a renamed export of the purity overlay's `group_id`, not a new UID
or a similarity cluster. Task normalization supplies an initial group. The
task-specific purity classifier applies voter membership, endpoint rules, and
reviewed source decisions; the overlay writes the resulting `group_id` and
preserves its original value and reassignment reason. The exporter uses the
family catalog to map that group to `family_key` and `level`.

Pinned upstream code:

- [Overlay assignment and audit](https://github.com/DragonDescentZerotsu/TxAgent/blob/00800a5b890440ac52e4e7689c4a157d0ff2a365/tools/chembl_tool/paper_experiments/build_conditioned_source_family_purity.py#L129): classifier invocation at line 139; `group_id` replacement at line 256.
- [BBB classifier](https://github.com/DragonDescentZerotsu/TxAgent/blob/00800a5b890440ac52e4e7689c4a157d0ff2a365/tools/chembl_tool/tasks/bbb_martins/source_family_purity.py#L500): explicit predictions and accepted voter membership precede near-direct/mechanism routing.
- [Oral classifier](https://github.com/DragonDescentZerotsu/TxAgent/blob/00800a5b890440ac52e4e7689c4a157d0ff2a365/tools/chembl_tool/tasks/bioavailability_ma/source_family_purity.py#L116).
- [Skin classifier](https://github.com/DragonDescentZerotsu/TxAgent/blob/00800a5b890440ac52e4e7689c4a157d0ff2a365/tools/chembl_tool/tasks/skin_reaction/source_family_purity.py#L174).
- [Exporter](https://github.com/DragonDescentZerotsu/TxAgent/blob/00800a5b890440ac52e4e7689c4a157d0ff2a365/tools/chembl_tool/paper_experiments/export_current_starling_level_records.py#L138): catalog mapping at line 138; overlay filtering and export at line 186.

## How to use the reference

First join each task's current record inventory by acquisition UID and compare
its existing routing against `family_key` and the reference level. Report matched,
disagreeing, and unmapped records before adopting a redirect. Preserve the original
group, upstream commit, and reason in any future routing overlay.

Level numbers are specific to a protocol. Our context-record workflow's L2 is
associated nonvoting records for the selected contexts; upstream L2 is a broader
near-direct family. Upstream Skin has three levels, whereas the existing website
Skin traces have four. Family semantics must be reconciled before changing
production caches. Newly adopted routing needs new cache provenance and new runs;
old traces describe the evidence actually supplied at the time.

This mapping covers only retrieval-eligible, level-assigned upstream records.
An absent UID is unmapped, not proof of exclusion and not a default L1 candidate.
One acquisition row can later split into several measurements: such descendants
need record-level review if they no longer share a family. The source membership
is pre-split; apply held-out-parent and query scaffold/parent eligibility filters
separately. The imported indexed-card tables describe upstream split candidates,
not this branch's exact query selections. This import does not certify the
upstream decisions as universally correct or change benchmark labels.

## Reproduce

If the pinned commit is absent from a fresh clone, fetch main first:

```bash
git fetch ssh://git@ssh.github.com:443/DragonDescentZerotsu/TxAgent.git main
python -m data.processing.evidence_library.build_source_uid_levels
python -m tests.chembl_tool.common.test_source_uid_levels
```

The builder verifies imported tables, frozen archive parts, canonical records,
and all UID-ledger partitions. It fails on missing joins, ambiguous keys, row
number mismatches, and conflicting UID assignments. `manifest.json` records
input/output hashes and coverage. Validation used `/usr/bin/python` 3.12.3 on
`epyc-4-10`, pandas 3.0.2, PyArrow 24.0.0, and zstandard 0.22.0.
