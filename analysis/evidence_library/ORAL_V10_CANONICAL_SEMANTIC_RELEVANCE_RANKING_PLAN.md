# Oral V10 canonical semantic relevance ranking plan

> Historical reference only. Do not launch requests from this plan: every new
> LLM request must use `reasoning_effort=high` and a fresh approved run identity.

This plan supersedes the partial V1 oral relevance run. Preserve that paid cache unchanged.

1. Read the Bioavailability_Ma V10 pair-bucket records and the frozen L1-L6 source-row mapping. Exclude L1 and fail closed unless every retained L2-L6 row maps exactly once.
2. Define endpoint-centered semantic groups from exact canonical fields only. Every group includes `source_id` and `canonical_endpoint_concept`. `hf_bioavailability` also includes `canonical_bioavailability_report_type` and `canonical_bioavailability_evidence_scope`; `oral_exposure` also includes `canonical_biological_matrix`; `fa`, `fg`, and `fh` add no other identity field.
3. Preserve the canonical-field provenance versions: `bioavailability_endpoint_concepts.v1`, `bioavailability_report_type_normalization.v1`, `bioavailability_evidence_scope.v1`, `starling_auxiliary.globally_reconciled.v2`, and `bioavailability_ma_pair_buckets.v19`.
4. Build one deterministic degree-10 comparison graph independently inside each progressive level. The expected universe is 268 level-scoped groups and 1,340 comparisons: L2 64/320, L3 159/795, L4 17/85, L5 11/55, and L6 17/85.
5. Put exactly one pair in every request. For each group, aim for three deterministic sample records: prefer one unique record from each of three distinct pair buckets, then backfill with other unique records. If only one or two unique records exist, show all available and never duplicate a source record.
6. Expose only canonical pair-bucket descriptors in sample cards. Withhold measurements, outcomes, labels, reference semantics, raw aliases, and evidence quantities.
7. Render the exact reviewed oral-bioavailability prompt in `bioavailability_relevance_bucket_level_v2.jinja`. Require the exact response object `{"winner_bucket_id":"<winning bucket ID>"}` with the winner constrained to the two supplied bucket IDs. Do not request a rationale, probability, score, or comparison ID.
8. Use `deepseek-ai/DeepSeek-V4-Flash-0731` only at `http://dgx027:50001/v1`, with provider `local`, no fallback, reasoning effort `low`, 4,096 maximum completion tokens, and 128 as the eventual full-run concurrency.
9. Before model calls, build the immutable request artifact, validate all counts and hashes, and render representative examples from L2-L6 for inspection.
10. Run only a 25-pair order-reversal pilot: five deterministic pairs per level, 50 single-comparison requests total. L2 must collectively cover all five sources. Preserve prompts, responses, attempt receipts, served-model identities, token usage, and returned reasoning content when available.
11. Pass the pilot only if all 50 requests complete with the exact served model and at least 23 of 25 reversed-order pairs agree. Save a compact pilot report and its hash.
12. Stop after reporting the build, rendered prompts, and pilot result. Do not launch the 1,340-request full ranking without a separate approval tied to the passing pilot report hash.
