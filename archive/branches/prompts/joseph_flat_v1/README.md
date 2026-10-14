# Joseph flat v1

This bundle freezes the group-stage prompt for the cache-matched flat harness.
It keeps the active BBB and oral task contracts from `tianang_flat_v1`, while
describing the level-aware Morgan, assay-transfer, and joint selection surfaces.

Single-molecule and final synthesis prompts remain owned by the selected task
profile. Selection is owned by `cache_matched_retrieval.v1`; this bundle only
controls how its already-selected records are presented. Pool names and cache
paths are provenance-only and are never rendered to the model. This prompt is
unevaluated until a real pilot is completed.

Run one condition:

```bash
python -m predict.harnesses.branches --organization flat \
  --task bioavailability_ma \
  --reranking assay-transfer \
  --record_pool assay-transfer-trained
```

The other modes are `morgan` and `joint`; the other pools are
`all_transfer_eligible` and `all`. `--evaluation-subset test` is supported but
does not run automatically. Validate its frozen inputs without inference with
`--prepare-only`. Reproduce the historical harness explicitly with
`--harness-version tianang-flat-v1` and its legacy retrieval flags.
