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
