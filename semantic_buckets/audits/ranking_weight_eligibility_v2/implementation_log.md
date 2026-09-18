# Implementation log

- Read the repository and simplicity-first instructions before changes.
- Inventoried semantic-bucket code, prompts, consumers, active releases, and history.
- Moved the semantic package to the repository-top-level `semantic_buckets/` owner.
- Moved active BBB and Oral V10 releases under `releases/`.
- Moved pre-Gold-v2 semantic artifacts under `history/pre_gold_v2/`.
- Promoted the complete incremental review worktree under `provenance/`.
- Published `semantic_weighted_top10.v2` and a new weighted eligibility generation.
- Restored request IDs and request receipts in V2 bucket decisions.
- Updated imports, resolvers, policy consumers, documentation, and release manifests.
- Generated TSV audits for top-ten policy, correlations, incremental decisions, and L5 exclusions.
- Verified both release inventories and 58 focused semantic-bucket, layout, and retrieval-cache tests.

No LLM inference or Bradley–Terry comparison was rerun.
