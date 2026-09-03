# Version-independent evidence-library analysis prompts

This directory contains prompts for analyses that consume a completed evidence
library rather than construct a particular release. `bucket_informativeness_v1.jinja`
is used by `data.processing.evidence_library.bucket_informativeness`.
`record_informativeness_v1.jinja` is the reviewed row-level scoring prompt used
by `data.processing.evidence_library.record_informativeness`.
`relevance_bucket_tournament_v1.jinja` compares V9 BBB relevance buckets for
relative downstream usefulness without exposing reference semantics or molecule
identity.
`relevance_bucket_diagnostic_v2.jinja` uses inline, outcome-blind candidates and
exact bucket-ID winners to isolate repeatability, order, batch-context, and batch-
size effects after the V1 calibration gate failed.
