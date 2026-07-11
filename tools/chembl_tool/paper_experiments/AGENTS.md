# Paper experiment operating contract

This directory owns the frozen, paper-facing molecular evidence agent matrix. Keep orchestration generic. Task-specific endpoint mappings belong only in `tools/chembl_tool/tasks/<task>/experiment_config.py`.

For reported runs, do not change these conditions between experiments:

```text
model: zai-org/GLM-5.2-FP8
temperature: 0
max_tokens: 20480
identity_blind: true
top_k_per_group: 3
min_similarity: 0.30
exact_query_exclusion: true
```

`full_flat` and `full_mechanism` must be derived from the same selected source evidence. Their only intended difference is whether the evidence union is presented in one group or separated by task mechanism.

Run each task's `none` condition first. All retrieved conditions must use `--single-analysis-source-batch` to reuse that condition's frozen single-molecule branch. Do not independently regenerate the prior for each ablation.

Do not put API key values in code, commands saved to files, manifests, reports, or traces. Use `GLM_API_KEY` or another explicitly selected environment variable.

The OpenAI SDK transport retry limit is explicitly fixed at 2, matching the baseline run. Structured JSON validation separately allows at most four total attempts inside the recorded call path. The final two attempts only reserialize the same JSON evidence to escape deterministic provider degeneration; they do not remove evidence or change inference settings. Any run with an incomplete single, group, or final branch is excluded by the batch completion invariant and rerun visibly through batch `--skip-existing`.

Group prompts also have a provider transport guard. Cleaned payloads at or below 750 KB are unchanged. Above that threshold, each oversized neighbor is reduced to at most 100 deterministically evenly spaced evidence rows, and the prompt records the original and retained row counts, original byte count, threshold, and sampling method. This prevents endpoint-level HTTP 413 failures without source- or task-specific rules.

Before reporting a run:

1. Require `n_failed_runs == 0`, or rerun failed samples with `--skip-existing`.
2. Require `query_smiles_trace_leaks == 0` from `summarize_results.py`.
3. Require identity-blind preflight to pass and use sanitized group outputs for final synthesis. `group_reasoning_outputs_raw.jsonl` is audit-only and must never be used as final input.
4. Report retrieval coverage next to performance; a missing-neighbor sample is a real property of that source/index and must not be silently removed.
5. Use paired test-set comparisons, bootstrap intervals, and exact McNemar tests from the shared summarizer.
6. Keep the scalar KNN result separate from LLM agent conditions; it is a numerical direct-F control.

Generated evidence and run outputs live under `outputs/paper/molecular_evidence_agent/` and are reproducible artifacts, not source code.
