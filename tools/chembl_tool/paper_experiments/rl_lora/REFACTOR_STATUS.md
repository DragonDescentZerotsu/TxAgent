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
- Hosted 120B is documented as intentionally stopped; local NeMo 20B is the
  active run.

## Still unresolved inside the RL subsystem

1. Shared materialized data still lives under the historical
   `rl_lora_gpt_oss_120b` output namespace. Moving it during active 20B training
   would break provenance/path assumptions; a later model-neutral artifact
   migration must preserve hashes and leave redirects/receipts.
2. `audit_one_pass_tokens.py` currently records the Tinker renderer/context
   surface. A separate NeMo/Hugging Face rendered-token receipt should be
   generated before a fresh/restarted local campaign; current 20B prompts were
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
6. Current 20B training-result analysis and sample-output review are not yet
   frozen into a repository receipt; this was deliberately deferred until code
   organization completed.

## Dirty worktree outside the RL subsystem

The repository is not ready for a single catch-all commit. Before the dedicated
RL publication, `git status --porcelain -uall` showed 53 modified tracked files
plus 141 untracked files. The isolated publication tracks the 55 RL/test paths,
the three public task adapters, and the scaffold-disjoint identity dependency
with its focused tests. After that scope is removed, 49 modified tracked files
and 86 untracked paths remain in the groups below:

| Group | Modified | Untracked | Main concern |
|---|---:|---:|---|
| shared common/Starling code | 9 | 5 | identity, reasoning payload/validation, benchmark builders |
| paper experiments outside RL | 15 | 26 | matrix, summaries, figures, diagnostics, result ledgers |
| task adapters/pipelines | 13 | 5 | prompt profiles, BBB lineage, Skin evidence scope |
| tests outside RL | 11 | 13 | mixed coverage for several independent experiment lines |
| data artifacts | 0 | 37 | two BBB processed lineages and audit outputs |
| root files | 1 | 0 | repository-wide instructions |

These changes represent several scientific lineages and should not be hidden in
one “cleanup” commit. After publishing the isolated RL/one-pass scope, the safe
order for the remaining work is:

1. Shared reasoning payload/validation and task prompt-profile changes.
2. BBB/Skin Starling dataset/index lineage and processed data artifacts.
3. Paper matrix, analyses, summaries, figures, and result-ledger updates.
4. Remaining diagnostics/router/train-ratio experiments and their tests.

Each group needs its own targeted test list and artifact/protocol review before
staging. Existing unrelated user changes must be preserved; do not stage the
entire worktree with `git add -A` until those scopes are reviewed.
