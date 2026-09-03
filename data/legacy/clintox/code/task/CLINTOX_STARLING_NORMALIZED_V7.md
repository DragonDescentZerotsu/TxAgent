# ClinTox source preparation and normalized v7 status

ClinTox uses `clinical_trial_failure_v1` as its only benchmark. Frozen AACT
toxicity-failure rows and SWEETLEAD/FDA comparator rows define the label;
Starling literature records never vote.

## Stage 1 source contract

The source-reconstructed gold and benchmark live under:

```text
data/starling_data/clintox/canonical_clinical_trial_failure_v1/
data/processed_clintox_clinical_trial_failure_v1/ClinTox/scaffold/
```

The seven independent literature sources remain under:

```text
data/starling_data/clintox/send_v2/<source_id>/extractions.parquet
```

`human_clinical_toxicity` is an indirect clinical-safety source. The frozen
338-row strict trial-failure subset is represented by
`direct_residual_candidates.parquet`, which points back to the corresponding
`send_v2` rows without copying them. These mappings remain
`pending_manual_review` and are not retrieval eligible.

The benchmark split publishes `voting_records.jsonl`. Each eligible AACT/FDA
source row retains its record vote, split, final parent label, and whether the
record vote agrees with that final label. Invalid source structures remain in
the canonical source audit and do not vote.

## Light cleaning boundary

The shared normalized-v7 builder may run only through `clean` for Stage 1:

```bash
python -m tools.chembl_tool.tasks.clintox.build_normalized_starling_evidence_library \
  --from-stage source --through-stage clean
```

This performs source-faithful whitespace/null cleaning, stable record
identification, and structure resolution. It preserves source IDs, source
columns, invalid rows, source hashes, and row counts.

Stage 1 does not decide canonical endpoints, construct measurements or units,
cluster fields, deduplicate sources, form pair buckets, collapse records, or
build retrieval indices. Existing ClinTox v7 artifacts from the removed
`ClinTox_Human_Toxicity` candidate are incompatible and must not be reused.

## Stage 2 contract and current gate

The endpoint-defining fields are now frozen source by source:

| source | endpoint | measurement | context kept separate |
|---|---|---|---|
| `human_clinical_toxicity` | `toxicity_category` | `outcome_measure` | clinical context |
| `nonclinical_in_vivo_toxicity` | `evidence_type` | `endpoint_value` + `endpoint_unit` | animal context |
| `organ_specific_toxicity` | `toxicity_endpoint` | quantitative result | organ system and effect status |
| `genotoxicity_carcinogenicity` | `endpoint` | `result_direction` | evidence category and assay type |
| `cellular_stress` | `stress_endpoint` | `effect_direction` | evidence basis |
| `general_cytotoxicity` | `endpoint_type` | `result_value` + `result_unit` | assay method |
| `off_target_ddi_exposure` | `result_metric` | `result_value` + `result_unit` | target and evidence type |

Implementation lives in `starling_normalization_sources.py`,
`starling_schema.py`, `starling_categorical_response.py`, and
`starling_measurement_resolution.py`. The measurement prompt uses
`gpt-5.4-mini`, batches 10 rows, and forbids invented units.

`build_stage2_review_samples.py` creates bounded review inputs and scores frozen
gold replays
under `data_processing/stage2_review_v1/`:

- `measurement_resolution_gold_candidates.jsonl`: 350 deterministic cases,
  50 per source, stratified by measurement form. This remains the immutable
  sampling ledger.
- `measurement_resolution_endpoint_review.jsonl`: the production-visible
  endpoint view, with all 350 sampled IDs excluded from profile examples.
- `measurement_resolution_gold.jsonl`: the frozen 350-case
  Codex-subagent-adjudicated gold. It contains 278 resolver expectations and
  72 controlled categorical expectations; two independent reviews are embedded
  per case, 32 resolver disagreements and 12 scope cases were adjudicated, and
  no model output was used as label evidence.
- `cluster_pilot.json`: frequent-500 plus stable-tail-1500 samples for each of
  15 endpoint/context namespaces, MiniLM target 50 and maximum 75 clusters,
  with at most 10 representative clusters marked for review.

The gold audit found five deterministic routing conflicts: three exact accepts
whose support makes them bounded, relative, or exposure-only, and two rejected
rows with a selected relative quantity. The current router remains unchanged;
the conflicts are explicit gold diagnostics.

The frozen 200-row extraction replay uses `gpt-5.4-mini`. Minimal prompt
iterations improved canonical case accuracy from 81.5% (v2) to 87.0% (v3)
and 92.5% (v4). Later additions plateaued at 92.5% (v5) and regressed to
89.5% (v6); replacing the mature shared template with a compact ClinTox
template regressed to 86.5% with four rejected rows (v7). Those branches were
not promoted.

Prompt v4 is the retained implementation and canonical replay. Its per-source
canonical accuracy is 100.0% general cytotoxicity, 100.0% genotoxicity, 86.1%
human clinical, 95.1% nonclinical in vivo, 88.1% off-target/DDI, and 92.1%
organ-specific. It still fails the frozen gate: overall accuracy is below 95%,
human clinical and off-target/DDI are below 90%, and two bound/range errors
remain. The complete receipt and mismatches are in
`data_processing/measurement_resolution_v1/gold_replay.metrics.json`.

An independent DeepSeek validation used the same frozen prompt and endpoint
profiles with `deepseek-ai/DeepSeek-V4-Flash-0731` and a 32,768-token
completion/reasoning allowance. All 25 batches were structurally valid with no
rejected rows. Canonical agreement over the 200 extraction cases was 89.5%:
100.0% general cytotoxicity, 100.0% genotoxicity, 75.0% human clinical, 95.1%
nonclinical in vivo, 88.1% off-target/DDI, and 86.8% organ-specific. The receipt
and 21 disagreements are in
`data_processing/measurement_resolution_v1/deepseek_gold_validation_32k_v1.metrics.json`.

Those disagreements were checked against the blind endpoint cards and both
embedded reviews. GPT matched the gold on 14 of 21. The seven disagreements
shared by both models were retained after endpoint review: four are fractions
or ambiguous multi-value selectors, two are experiment-relative changes, and
one requires a more specific severity-qualified count unit. No gold label was
changed. The full 350-card endpoint view also reconciles exactly with the gold:
IDs, endpoints, resolver inputs, profile blocks, and review provenance match;
the 72 categorical expectations reproduce from the controlled encoder; and all
five deterministic routing conflicts remain explicitly reviewed.

No full-corpus LLM normalization or clustering has been run. Full measurement
resolution must wait for a newly frozen prompt to pass this gold gate;
cluster names/mappings must wait for representative-cluster review. Only after
those gates pass should Stage 02 canonicalization and later Stage 04/05
pair-bucket/dedup artifacts be materialized for all 4,846,914 rows.

## Build and validation

```bash
python -m tools.chembl_tool.tasks.clintox.build_clinical_trial_failure_benchmark
pytest -q tests/chembl_tool/tasks/clintox
```

The source inventory must reconcile nine provenance sources: two gold sources
and seven `send_v2` evidence sources. The 338 residual candidates overlap the
human clinical source and therefore do not increase the source-row total.
