# Bioavailability condition-aware benchmark

The active Bioavailability benchmark is:

```text
data/gold_labels/Bioavailability_Ma/v1/scaffold/
```

It contains 2,489 molecule-condition rows split 1,958/262/269 across
train/valid/test. Of these, 2,092 are `no_reported_external_condition` rows and
397 carry one of seven reviewed external conditions:

```text
prandial_state=fasted
prandial_state=fed_unspecified
prandial_state=fed_high_fat
co_treatment=rifampin
disease=cirrhosis
disease=cystic_fibrosis
release_profile=modified_release
```

The direct source is `bioavailability_canonical_direct.v2`. Review candidates
and model-assisted semantic-review traces are retained under
`data/artifacts/starling/bioavailability_ma/source_reviews/context_conditioned_review_v2/`.
Relative-effect, comparator-only, non-human, indirect-analyte, predicted,
simulated, ambiguous-route, and non-direct F records do not vote for an
external-condition label.

The historical names of the null-condition source and selected condition build
are recorded only in `data/artifacts/gold_labels/conditioned_benchmark/migration_receipt.json`.
Neither is a second active benchmark. The current valid/test rows are unchanged,
so existing results may be reused only after their input hashes match that
receipt.

The shared schema, split contract, and publication entrypoint are documented in
[`../../common/starling/CONDITIONED_BENCHMARK.md`](../../common/starling/CONDITIONED_BENCHMARK.md).
