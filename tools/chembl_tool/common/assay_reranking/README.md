# V11 assay-transfer reranker cache

This module builds frozen, read-only assay-transfer score caches for the V7
Starling retrieval artifacts. It supports Bioavailability_Ma, Skin_Reaction,
and BBB_Martins through the same builder and runtime cache contract.

The cache is scoped to scaffold validation queries and uses this fixed
retrieval contract by default:

- raw Morgan/Tanimoto pool of 50 molecules per reasoning group;
- minimum similarity 0.0;
- `parent_disjoint` identity exclusion after the raw top-50 pool;
- every retrieval-eligible Stage07 record for each candidate molecule;
- no Stage04 pair-bucket or Stage05 calibration eligibility filter;
- record-level assay-transfer ranking;
- BF16 model backbone with FP32 selected A/B output-head accumulation.

## Runtime

The pinned model tokenizer requires `transformers==4.57.6`. Keep this scoring
runtime isolated from the main `txagent-glm` package set. On node002, the
current node-local overlay is:

```text
/local/joseph/huggingface/assay_transfer_v11/python_transformers_4_57_6
```

Prefix scoring commands with:

```bash
PYTHONPATH=/local/joseph/huggingface/assay_transfer_v11/python_transformers_4_57_6 \
HF_HUB_CACHE=/local/joseph/huggingface/assay_transfer_v11/hub \
HF_HOME=/local/joseph/huggingface/assay_transfer_v11
```

The worker rejects any other Transformers version so an environment change
cannot silently alter tokenization or selected-token scoring.

## Build and resume

Preparation is CPU-only and freezes the catalog, raw Morgan candidate manifest,
and exact prompt demand:

```bash
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.common.assay_reranking.build_v11_cache \
  --tasks bioavailability_ma skin_reaction \
  --benchmark-split scaffold --evaluation-subset valid --phase prepare
```

Scoring is append-only and resumable. The default batch size is 128. A full
Skin_Reaction run showed that 192 can exceed an 80 GB GPU on later, longer
prompt batches even when early batches appear to leave ample memory.

```bash
PYTHONPATH=/local/joseph/huggingface/assay_transfer_v11/python_transformers_4_57_6 \
HF_HUB_CACHE=/local/joseph/huggingface/assay_transfer_v11/hub \
HF_HOME=/local/joseph/huggingface/assay_transfer_v11 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
/data1/joseph/miniconda3/condabin/conda run --no-capture-output -n txagent-glm \
  python -m tools.chembl_tool.common.assay_reranking.build_v11_cache \
  --tasks bioavailability_ma skin_reaction \
  --benchmark-split scaffold --evaluation-subset valid --phase score \
  --devices 0,1,2,3,4,5,6,7 --batch-size 128 --local-files-only
```

Verify exact demand coverage and SQLite integrity without loading a model:

```bash
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.common.assay_reranking.build_v11_cache \
  --tasks bioavailability_ma skin_reaction \
  --benchmark-split scaffold --evaluation-subset valid --phase verify
```

Each task writes `catalog.jsonl`, `candidate_manifest.jsonl`,
`prompt_demand.jsonl`, `scores.sqlite3`, and `VERSION.json` under:

```text
outputs/chembl_tool/tasks/<task>/evidence_library/assay_transfer_rerank/
  v11_with_categorical/scaffold/valid/
```

`VERSION.json` becomes `complete` only after exact cache coverage and
`PRAGMA quick_check=ok`. Reasoning runners open the SQLite cache read-only.
Distinct record references that render to the same immutable prompt cache key
remain in the catalog and candidate manifest but share one score row. The
version manifest records the pre-deduplication task count, the number collapsed,
and the exact unique score count separately.

## BBB model

BBB uses the task-aware multitask checkpoint pinned in `assets/v11/models.json`:

```text
jiosephlee/assay-transfer-tool-soft-v11-multitask-with-categorical
```

Its cache uses the vendored BBB templates, the compact V7 held-out scaffold
index, and the same Morgan-pool/precision provenance as the other tasks.
