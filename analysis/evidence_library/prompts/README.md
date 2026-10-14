# Version-independent evidence-library analysis prompts

This directory contains prompts for analyses that consume a completed evidence
library rather than construct a particular release. `bucket_informativeness_v1.jinja`
is used by `analysis.evidence_library.bucket_informativeness`.
`record_informativeness_v1.jinja` is the reviewed row-level scoring prompt used
by `analysis.evidence_library.record_informativeness`.
The relevance tournament and diagnostic templates are retained for frozen
Bradley–Terry experiment provenance; their fitting/ranking executables are retired.
New semantic utility scoring uses the canonical two-pass scorer under
`optimization/semantic_buckets/`.
