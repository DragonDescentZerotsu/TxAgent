# RL and worktree refactor status

Snapshot date: 2026-08-11.

## Organized in the current pass

- Tinker and NeMo share one reward metadata/dispatch surface and one frozen
  training recipe.
- NeMo derived configs inherit one formal 20B Bio config and are validated
  before worker creation.
- One-pass versions, visibility modes, task label values, and nested schema
  extraction live in the dependency-light `one_pass_contract.py`.
- Canonical RL data generation and historical trace replay are explicitly
  separated.
- OpenAI-compatible and Tinker evaluators share one audited-data/runtime gate,
  6,144-token default, nested schema validation, prediction parser, result
  layout, and inference-contract hash.
- Training requires adjacent data audit v2; historical v1 artifacts remain
  available for read-only reproduction.
- Task pipelines expose a public `build_group_prompt_payload()` adapter instead
  of requiring the RL materializer to call private task functions.
- Shared reasoning payload/trace helpers, structured-output validation,
  versioned task prompt profiles, final-decision profiles, manifest provenance,
  and branch-reuse gates are isolated behind one documented contract.
- Hosted 120B and local NeMo 20B are both documented as intentionally stopped;
  neither run may be resumed automatically.

## Still unresolved inside the RL subsystem

1. Shared materialized data still lives under the historical
   `rl_lora_gpt_oss_120b` output namespace. There is no longer an active-run
   blocker, but a model-neutral artifact migration must preserve hashes and
   leave redirects/receipts; it is not required for analysis of the frozen run.
2. `audit_one_pass_tokens.py` currently records the Tinker renderer/context
   surface. A separate NeMo/Hugging Face rendered-token receipt should be
   generated before a fresh/restarted local campaign; the stopped 20B prompts were
   bounded indirectly by the same GPT-OSS tokenizer and existing lengths.
3. Provider optimizer behavior remains intentionally different: Tinker uses
   Adam beta2 0.95 and drops constant-reward groups; NeMo inherits beta2 0.999
   and gives constant groups zero advantage. This is documented, not yet a
   matched optimizer ablation.
4. OpenAI-compatible retry usage reports only the selected response usage from
   the shared client; Tinker records every attempt. This affects exact retry
   cost accounting, not predictions or reward.
5. Reward v2 checks structure and binary judgments but does not semantically
   grade the analysis text. Branch correctness still means agreement with the
   final task gold because independent branch labels do not exist.
6. The stopped 20B run's step/checkpoint and scalar-trend receipt is now frozen
   in `LOCAL_NEMO_RUNBOOK.md`. Curated sample-output review and checkpoint-valid
   evaluation are still incomplete; no checkpoint has passed a promotion gate.

## Repository publication status

The isolated RL subsystem, shared reasoning/profile contract, BBB/Skin Starling
lineages, paper diagnostics, tests, and processed audit artifacts were reviewed
and published to `main` through commits `4ea18e4`, `fa8eec5`, and `d359170`.
After that publication, local `main` and `origin/main` matched and the worktree
was clean. Future changes must still be scoped and reviewed normally; the old
33-modified/76-untracked inventory is historical and no longer an open cleanup
queue.
