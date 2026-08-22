# ClinTox send_v2 gold-source review protocol

This audit reviews the frozen 360-row, category-stratified sample at
`outputs/chembl_tool/tasks/clintox/evidence_library/clintox_send_v2_direct/05_audits/gold_qa_sample.parquet`.
The sample SHA-256 is
`953387d884c42fb7d966b8a3f063cdd2612f1ae23bea0241e8183f416f4c8188`.
It contains 30 accepted records from each of the 12 declared
`toxicity_category` values. Failed rows are retained in the sample; they are
not replaced.

## Review question

Does the cited human-clinical passage directly support the proposed binary
label for the molecule represented by the supplied structure?

A row passes only when all of the following are true:

1. **Entity alignment:** the named clinical entity is the molecule represented
   by the supplied structure. Formulations, conjugates, mixtures, metabolites,
   co-therapies, and protective agents are not silently reduced to a component.
2. **Human exposure:** the passage reports human clinical exposure to the
   molecule, rather than animal, in-vitro, prediction-only, disease-background,
   or general class evidence.
3. **Causal attribution:** the outcome is attributed to the molecule. Events
   caused by disease, another treatment, a comparator, or an unresolved
   combination do not pass.
4. **Direction:** a positive row reports toxicity caused by the molecule; a
   negative row explicitly reports absence of molecule-attributed toxicity.
   Protective, preventive, reduced-risk, and null-comparison findings are not
   positive toxicity claims.
5. **Outcome specificity:** the outcome supports the declared category, or
   supports explicit global absence for `toxicity_absent`.
6. **Qualifiers:** dose, route, formulation, co-medication, comparator, and
   population conditions that materially change the claim are explicit and do
   not make the unqualified molecule-level label misleading.
7. **Source support:** the outcome, context, and support text are mutually
   consistent and sufficiently specific to re-derive the decision.

## Review statuses

- `pass`: the proposed label is directly supported under every criterion.
- `fail`: at least one criterion is violated.
- `uncertain`: the supplied fields are insufficient to decide; this counts as
  a failure for promotion.

Failure reasons use a small source-policy vocabulary:

- `wrong_outcome_direction`
- `protective_or_preventive_context`
- `combination_or_comparator_confounded`
- `background_event_not_attributed`
- `formulation_or_entity_mismatch`
- `not_human_clinical_exposure`
- `category_outcome_mismatch`
- `absence_not_global_safety`
- `insufficient_attribution`
- `other_source_semantic_failure`

The current category-only adapter can be promoted only if every sampled row
passes. Any failure requires a source-policy revision and a new frozen audit of
the rebuilt accepted population.

## Frozen result

The review failed: 252 rows passed, 96 failed, and 12 were uncertain. All 30
proposed `toxicity_absent` rows were non-passing. The full row-level receipt is
`reviewed_rows.tsv`; aggregate counts and SHA-256 receipts are in
`summary.json`. The existing 7,628-parent split remains diagnostic and must not
enter paper-facing experiments.

## Required source correction

`reextraction_contract_v1.json` extends the immutable send_v2 extraction guide
with the smallest structured fields needed to resolve the observed failures:
entity alignment, human-evidence scope, causal attribution, direction,
clinical significance, absence scope, and material qualifying conditions. It
supports two possible label thresholds without another extraction pass. The
recommended profile predicts clinically meaningful toxicity; the broader
profile treats any attributed adverse event as positive. No profile is active
until a new source delivery populates these fields and passes a new frozen
audit.
