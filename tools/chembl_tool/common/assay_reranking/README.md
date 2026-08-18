# V11 assay-transfer reranker cache

This module builds frozen, read-only assay-transfer score caches from a
lineage-owned Starling V7 paper evidence view. The scaffold-validation contract is:

- up to 50 Morgan/Tanimoto molecules per reasoning group after identity exclusion;
- minimum similarity 0.0;
- `parent_disjoint` identity exclusion before the eligible top-50 truncation;
- every in-scope Stage 07 record for each candidate molecule;
- no Stage 04 pair-bucket or Stage 05 calibration filter;
- record-level ranking with a BF16 backbone and FP32 selected A/B head logits.

The prompt projection and Jinja templates are vendored from the pinned
`starling_assay_transfer` revision recorded in `assets/v11/SOURCE.json`.

## Runtime

The tokenizer requires `transformers==4.57.6`. On dgx012, use the existing
environment and keep model downloads on `/vast`:

```bash
export HF_HOME=/vast/projects/myatskar/design-documents/joseph/.cache/huggingface_assay_transfer_v11
export PYTORCH_ALLOC_CONF=expandable_segments:True
V11_PYTHON=/vast/projects/myatskar/design-documents/conda_env/open_rlhf_intern/bin/python
```

The worker rejects any other Transformers version.

## Build and resume

Supply all three lineage mappings for each task. Preparation is CPU-only:

```bash
$V11_PYTHON -m tools.chembl_tool.common.assay_reranking.build_v11_cache \
  --tasks bioavailability_ma --phase prepare \
  --task-evidence-view bioavailability_ma=<paper-root>/evidence/bioavailability_starling_v7 \
  --task-query-jsonl bioavailability_ma=<current-scaffold-valid.jsonl> \
  --task-cache-dir bioavailability_ma=<paper-root>/assay_transfer_rerank/bioavailability_starling_v7/v11_with_categorical/scaffold/valid
```

Scoring is append-only and resumable until finalization:

```bash
$V11_PYTHON -m tools.chembl_tool.common.assay_reranking.build_v11_cache \
  --tasks bioavailability_ma --phase score \
  --task-evidence-view bioavailability_ma=<paper-evidence-view> \
  --task-query-jsonl bioavailability_ma=<current-scaffold-valid.jsonl> \
  --task-cache-dir bioavailability_ma=<lineage-cache-dir> \
  --devices 0,1,2,3 --batch-size 128 --local-files-only
```

Use the same mappings with `--phase verify` for a read-only integrity check.

### Record scope

`--record-scope all` is the backward-compatible default. Two narrower scopes
filter the record bridge before identity exclusion and Morgan top-50 selection:

- `labelable_direct` retains only direct-source records accepted by the frozen
  task labeler;
- `numeric_direct` additionally requires a finite direct numeric value and is
  supported only for Bioavailability.

For example, a classification cache can be prepared with:

```bash
$V11_PYTHON -m tools.chembl_tool.common.assay_reranking.build_v11_cache \
  --tasks bioavailability_ma --phase prepare \
  --record-scope labelable_direct \
  --task-evidence-view bioavailability_ma=<paper-evidence-view> \
  --task-query-jsonl bioavailability_ma=<current-scaffold-valid.jsonl> \
  --task-cache-dir bioavailability_ma=<new-lineage-cache-dir>
```

`VERSION.json` records the scope, preselection record and molecule counts, and
the `record_scope_then_tanimoto_identity_exclusion_then_eligible_pool.v1`
candidate contract. Resume and verification require the same scope.

Each finalized lineage retains only:

```text
scores.sqlite3
VERSION.json
```

Rendered prompts and preparation joins live in temporary SQLite tables while
scoring is incomplete. Successful finalization checks exact coverage, drops the
build tables, vacuums the database, and requires `PRAGMA quick_check=ok`.
Reasoning runners use direct query/group/molecule assignments from SQLite.

## Models

The task-specific Bioavailability, BBB, and Skin Reaction checkpoints and
immutable revisions are pinned in `assets/v11/models.json`. Historical
task-local caches and the old BBB multitask checkpoint remain separate lineage
artifacts.

On four full 180 GiB B200s, worst-prompt profiling found that larger batches
fit but did not improve throughput. For BBB's 984-token maximum, batch 128 used
29.71 GiB and delivered 36.71 prompts/s/GPU; batch 1,408 used 173.58 GiB and
delivered 35.50 prompts/s/GPU. For Skin Reaction's 1,214-token maximum, batch
128 used 33.11 GiB and delivered 29.75 prompts/s/GPU. Batch 128 is therefore the
measured throughput-oriented default, not a memory ceiling.
