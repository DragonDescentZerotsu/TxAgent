# Ranking, weight, and eligibility audit

## Result

The active Bradley–Terry ordering and expert weights are largely independent.
Across all 90 top-ten entries, the Spearman correlation between ranking priority
and weight is `0.130` (`p=0.221`). BBB L5 is the only level with a clearly stronger
relationship (`0.743`, `p=0.014`). The other levels do not show evidence that the
expert weights reproduce the pairwise ordering.

This follows from the two review questions. Bradley–Terry comparisons ask which
bucket is more relevant in a forced pairwise choice. Expert weights encode how
transferable and useful the evidence should be after selection. They were assigned
manually from rank, score, source composition, and pair-bucket identity; there was
no weight-assignment model prompt.

## Prompt surfaces

The pairwise prompts are `prompts/bbb_relevance_bucket_level_v3.jinja` and
`prompts/bioavailability_relevance_bucket_level_v2.jinja`. They show two compact
bucket summaries containing source-specific canonical dimensions and up to five
canonicalized sample profiles. They do not show a query molecule, molecular
structure, raw measurements, support text, or full records.

The include/exclude prompt is
`prompts/semantic_indirect_retrieval_eligibility_v2.jinja`. It asks whether a
bucket should ever be eligible as auxiliary evidence and excludes only
intrinsically unusable evidence. It explicitly retains indirect, low-priority,
cross-species, or weakly transferable evidence when it may still be informative.

## Incremental decisions and V2 policy

- BBB: 118 newly created L2-L4 buckets; 109 passed and 9 were rejected.
- Oral: 4 newly created buckets; 2 passed and 2 were rejected.
- Seven new BBB L5 atoms have deterministic bucket IDs and remain outside
  assay-transfer eligibility.
- BBB L4 `sb_1f5dba8db75a3bd2fc9b` is rank 5, passed, and has weight `0.7`.
- Oral L2 `sb_44b6c9e5770cd98c32c8` is rank 3, rejected, and has weight `0.0`.

The V2 eligibility generation restores request IDs, model, provider, token counts,
and prompt hashes for incremental reviews. Existing Oral L2 nonzero weights whose
rationales still say “undefined reference” are retained for reproducibility and
flagged for later scientific review rather than silently changed here.

The supporting tables are `top10_policy.tsv`, `rank_weight_correlations.tsv`,
`incremental_bucket_decisions.tsv`, and `bbb_l5_non_assay_transfer.tsv`. Exact
rendered pairwise and eligibility examples, including responses and request
metadata, are under `rendered_requests/`.
