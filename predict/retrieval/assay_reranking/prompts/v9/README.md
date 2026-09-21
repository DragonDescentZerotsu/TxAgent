# Gold-v1 direct-transfer caches

`prompt.jinja` and `prompt_config.json` are the vendored prompt assets used by
`predict.retrieval.assay_reranking.v9.V9PromptRenderer`. The published caches
record their hashes, exact model revision, rendered prompt hashes, and Gold
input hashes in each `VERSION.json`.

## Published caches

BBB and Oral use `v10_3_best_scaffold_morgan100_v1`; Skin uses
`v9_skin_gold_v1_scaffold_morgan100_v1`. Each profile contains independent
valid and test Parquet files. No file is split or packed. Verify every published
file and reconstruct its deterministic candidate membership with:

```bash
python -m predict.retrieval.assay_reranking.v9 validate-release
```

The validator accepts a relocated checkout only when its canonical Gold-v1
input hashes match the frozen cache. The aggregate file and size receipts are in
`predict/retrieval/assay_reranking/direct_gold_v1_release.json`.

## Rebuild one split

Use a fresh output directory; never overwrite the published cache. BBB and Oral
use lineage `v10_3_best`, and Skin uses `v9_scaffold`. The scorer requires the
pinned model revision, `transformers==4.57.6`, PyTorch, RDKit, and a GPU.

```bash
TASK=bbb_martins
SUBSET=valid
LINEAGE=v10_3_best
CACHE_ROOT=/path/to/staging/$TASK/scaffold/$SUBSET

python -m predict.retrieval.assay_reranking.v9 download \
  --task "$TASK" --lineage "$LINEAGE"
python -m predict.retrieval.assay_reranking.v9 prepare \
  --task "$TASK" --subset "$SUBSET" --lineage "$LINEAGE" \
  --cache-root "$CACHE_ROOT" --no-reuse
python -m predict.retrieval.assay_reranking.v9 score \
  --task "$TASK" --subset "$SUBSET" --lineage "$LINEAGE" \
  --cache-root "$CACHE_ROOT" --num-shards 1 --shard-index 0
python -m predict.retrieval.assay_reranking.v9 finalize \
  --task "$TASK" --subset "$SUBSET" --lineage "$LINEAGE" \
  --cache-root "$CACHE_ROOT" --num-shards 1
python -m predict.retrieval.assay_reranking.v9 validate \
  --task "$TASK" --subset "$SUBSET" --lineage "$LINEAGE" \
  --cache-root "$CACHE_ROOT"
```

Repeat with `SUBSET=test`; use `TASK=bioavailability_ma` with lineage
`v10_3_best`, or `TASK=skin_reaction` with lineage `v9_scaffold`. Multiple score
shards may run independently when every shard uses the same `--num-shards`.
Fresh Parquet bytes can vary with the writer environment, so the committed cache
and its SHA-256 are the exact release; validation establishes the scientific
candidate, prompt, score, and rank contract of a rebuild.
