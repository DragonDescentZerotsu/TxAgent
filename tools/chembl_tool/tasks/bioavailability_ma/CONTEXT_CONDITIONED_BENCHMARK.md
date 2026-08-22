# Bioavailability selected context-conditioned benchmark

The current formal conditioned lineage is
`bioavailability_context_conditioned_selected_v1` under:

```text
data/processed_starling_context_conditioned_selected_v1/Bioavailability_Ma/scaffold/
```

It preserves all 2,092 `record_supported_v2` null-group rows and adds 397
parent-condition labels across seven selected external groups. The resulting
2,489 rows are split `1,958/262/269` for train/valid/test, with zero parent and
scaffold overlap.

The source is `bioavailability_canonical_direct.v2`; review candidates and
model traces are under
`data/starling_data/bioavailability_ma/context_conditioned_review_v2/`.
Relative-effect, comparator-only, non-human, indirect-analyte, predicted,
simulated, ambiguous-route, and non-direct F records are rejected before
semantic review. Two independent `gpt-oss-120b` passes use the hashed
`condition_semantic_prereview.v2` contract. Only unanimous acceptance is
promoted; this is model-assisted review, not human annotation. Exact execution
settings are pinned in both `model_prereview*.summary.json` files and their
hashes are pinned by the dataset `summary.json`.

The seven selected external groups are:

```text
prandial_state=fasted
prandial_state=fed_unspecified
prandial_state=fed_high_fat
co_treatment=rifampin
disease=cirrhosis
disease=cystic_fibrosis
release_profile=modified_release
```

Shared filtering, 60% voting, exact-signature grouping, group promotion,
split, audit, and version rules are defined once in
[`../../common/starling/REVIEWED_CONDITIONED_BENCHMARK.md`](../../common/starling/REVIEWED_CONDITIONED_BENCHMARK.md).

Rebuild the current data:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.reviewed_context_conditioned_benchmark \
  build-selected
```

The former rule-only `bioavailability_context_conditioned_v1` dataset and its
baseline outputs were removed because they predated the terminal review and
relative-effect exclusion contracts. They must not be cited as current. The
condition-aware baseline runner now defaults to selected v1, but selected-v1
baseline results require a fresh run and must not reuse the deleted v1 metrics.
