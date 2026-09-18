# Semantic-bucket prompts

These templates are versioned inputs to the semantic grouping, within-semantic
readout grouping, ranking, and retrieval-eligibility workflows. Generated
manifests must record the hashes of every template used.

`semantic_weight_sliding_v2/` contains the active initial and locked-anchor
prompts for absolute 0.00–1.00 bucket utility weights, with fixed calibration
bands. Their model-facing cards deliberately omit semantic-bucket identity,
rank, and frequency counts. V1 was prepared but never executed.

`semantic_weight_calibration_v6/` is the immutable second-pass calibration
bundle. It shows the V5 score as a fallible prior, adds normalized experimental
context, and requires each rationale to identify whether the prior was retained,
increased, or decreased. Missing metadata alone cannot justify a zero weight.
