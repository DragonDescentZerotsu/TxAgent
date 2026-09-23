# Full-flat V5 artifact manifests

These manifests are the tracked contracts for the ignored evidence-library and
retrieval-cache payloads. One manifest is generated per task and benchmark.

Inventory a local restore, verify it once, and then run the harness against the
receipt:

```bash
python -m predict.retrieval.assay_reranking.artifact_bundle inventory \
  --task bbb_martins --benchmark gold_v1 \
  --cache-config predict/retrieval/assay_reranking/ranked_level_retrieval_v4.yaml \
  --output predict/retrieval/assay_reranking/flat_v5_manifests/gold_v1/bbb_martins.json

python -m predict.retrieval.assay_reranking.artifact_bundle verify \
  --manifest predict/retrieval/assay_reranking/flat_v5_manifests/gold_v1/bbb_martins.json
```

Use `package` to create deterministic split `tar.zst` payloads and `restore` to
verify and install them into their canonical repository-relative paths. The
runtime performs only receipt stat checks; a missing or stale receipt fails
before retrieval preparation.

Canonical cache identities use:

```text
flat_v5/<benchmark>/<task>/<role>/<method>/<release>/<variant>
```

Historical profile names remain valid through the explicit compatibility map.

The opt-in V27 successor uses separate manifests under
`v27_successors/<benchmark>/<task>.json` and ignored receipts under
`data/caches/assay_reranking/active/flat_v5/receipts/v27_successors/<benchmark>/<task>.json`.
Pass those paths with `--flat-artifact-manifest` and `--flat-artifact-receipt`,
plus the matching `ranked_level_retrieval_<benchmark>_v27_successors_v1.yaml`
with `--assay-transfer-cache`. This keeps historical defaults and receipts
unchanged. Gold Ames, TDC DILI, and TDC Carcinogens have verified successor
bundles; Gold Carcinogens and TDC Ames wait for their top-40 cache publication.

For opt-in inference before top-40 scoring completes, use
`ranked_level_retrieval_gold_v1_v27_partial_snapshot_v1.yaml` or
`ranked_level_retrieval_tdc_v1_v27_partial_snapshot_v1.yaml` with its
task-specific manifest under `v27_partial_snapshots/<benchmark>/<task>.json`
and matching verified receipt. These immutable snapshots retain the full
Morgan universe but omit unscored rows from assay-transfer selection; even an
empty query/level is valid. The normal successor bundles remain strict.
