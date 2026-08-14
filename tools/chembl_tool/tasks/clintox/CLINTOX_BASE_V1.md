# ClinTox base v1 candidate lineage

`clintox_base_v1` is the active source candidate for
`ClinTox_Human_Toxicity`. It replaces the candidate built from raw-v1
organ-specific toxicity while preserving that older source, adapter, and
artifact store as historical lineage. The previous active snapshot is commit
`f056084332fbb5cec87cce5ec2a0d82a9b25b2fc`.

This task predicts whether the controlled source category reports human
clinical toxicity (`Y=1`) or explicitly reports `toxicity_absent` (`Y=0`). It
is not TDC/MoleculeNet CT_TOX clinical-trial failure.

## Raw source

The exact Parquet and extraction guide are tracked at:

```text
data/starling_data/clintox/clintox_base_v1/
```

The Parquet has 338,780 rows. `SOURCE_MANIFEST.json` records its checksum,
schema, completeness, and known provenance limitations. In particular, the
guide requires `qualifying_conditions`, but the delivered Parquet omits that
column. The field is unavailable, not empty after cleaning.

## Build policy

The adapter accepts only exact guide-declared toxicity categories, an explicit
toxicity outcome, `needs_more_context=false`, required provenance text, and a
resolvable source SMILES. `toxicity_absent` is negative and the other 11
declared categories are positive. Off-schema values are rejected without
rewriting. FDA status, confidence, prose, and numeric values never infer or
override the label.

Every accepted source row is one vote. Parent labels require at least 70%
record agreement; exact ties are rejected. The final split uses the shared
`record_supported_v2` scaffold-disjoint allocator.

The fixed build has 297,262 labeled source rows before structure resolution,
6,047 accepted binary parents (`Y=0`: 548; `Y=1`: 5,499), and 568 rejected
parents (203 exact ties and 365 below 70% agreement). The final split is:

| Split | Parents | Y=0 | Y=1 | Multi-record | Singleton |
|---|---:|---:|---:|---:|---:|
| train | 4,839 | 467 | 4,372 | 2,662 | 2,177 |
| valid | 604 | 42 | 562 | 604 | 0 |
| test | 604 | 39 | 565 | 604 | 0 |

Parent identity and Bemis–Murcko scaffold overlap are zero across all splits.

Only accepted gold-eligible rows enter the `clintox_base` direct evidence
catalog. Held-out parents are removed from that catalog, and query-time
retrieval must additionally use `parent_disjoint`.

The full direct catalog contains 296,545 accepted source records aggregated
into 7,424 structure-level evidence rows. The held-out-filtered view removes
1,365 evidence rows, retains 6,059, and has zero held-out parent overlap.

## Commands

```bash
python -m tools.chembl_tool.tasks.clintox.clintox_base build-library --workers 8
python -m tools.chembl_tool.tasks.clintox.clintox_base build-benchmark
python -m tools.chembl_tool.tasks.clintox.clintox_base build-heldout-view --workers 8

python -m tools.chembl_tool.tasks.clintox.clintox_base_artifact_store package
python -m tools.chembl_tool.tasks.clintox.clintox_base_artifact_store verify-tracked
python -m tools.chembl_tool.tasks.clintox.clintox_base_artifact_store verify-local
```

The result remains `candidate_pending_qa` until every row in the deterministic
30-per-category source-record sample passes review. It must not enter formal
paper matrices before that gate passes.
