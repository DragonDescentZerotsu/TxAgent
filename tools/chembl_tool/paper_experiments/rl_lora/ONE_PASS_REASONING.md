# One-pass reasoning for RL

One-pass reasoning is the RL-specific front end for the Starling evidence
agent. It fuses the previous single-molecule, full-flat analog, and final
adjudication calls into one generated response so one trajectory contains all
three trainable judgments. It does not replace the canonical three-stage
inference pipeline outside this experiment.

## Frozen output contract

The response is exactly one JSON object containing:

```text
single_prediction
single_analysis
analog_prediction
analog_analysis
<task final prediction and final fields>
```

The supported prompt contracts are:

- `one_pass_full_flat.v1`: identity-blind query and anonymous neighbors.
- `one_pass_full_flat_visible_prefetched.v1`: visible structures and allowed
  source identifiers, with retrieval and molecular tools still fixed by the
  harness.

Both use global `parent_disjoint` retrieval for formal current-lineage data.
Gold is stored only in environment-private metadata. The model has no live tool
surface.

## Code ownership

```text
one_pass_contract.py
  dependency-light versions, visibility modes, task labels, schema extraction

one_pass.py
  historical three-stage trace -> one-pass prompt builder only

materialize_split_one_pass.py
  canonical RL dataset builder: source split -> retrieval -> prefetched tools

materialize_one_pass.py
  historical trace-replay builder for aligned architecture comparisons only

audit_one_pass_data.py / audit_one_pass_tokens.py
  privacy, identity, tool-receipt, hash, and context gates

one_pass_runtime.py
  shared audited-data loading, nested schema validation, prediction parsing,
  result paths, and inference-contract hashes

run_one_pass.py / evaluate_tinker_one_pass.py
  thin OpenAI-compatible base evaluator / Tinker checkpoint evaluator

experiment_contract.py -> training_contract.py -> one_pass_reward.py
  shared training settings -> provider-neutral dispatch -> reward arithmetic
```

Task pipelines expose `build_group_prompt_payload()` as the public adapter used
by the direct materializer. Their historical private alias remains temporarily
for old callers and tests. The materializer consumes only the adapter's
normalized `group` and `neighbors` evidence surface; the frozen one-pass
template, not the task adapter, owns the recorded prompt profile and output
instructions.

## Canonical lifecycle

1. Read one frozen scaffold split and its held-out-filtered retrieval index.
2. Run `materialize_split_one_pass.py` with `parent_disjoint` retrieval and the
   explicit visibility mode.
3. Produce the JSONL plus its materialization manifest.
4. Run data audit v2. Training launchers reject missing, failed, or stale v2
   audits before creating a hosted client or NeMo workers.
5. Run the backend-specific token/context audit with 6,144 completion tokens
   reserved.
6. Evaluate the frozen base or a checkpoint. Both evaluators require a fresh
   adjacent audit, the same nested schema, and the same 6,144-token default.
7. Resume only results with the exact same prompt hash and inference-contract
   hash. Model, checkpoint, renderer, temperature, token cap, retry count, and
   seed changes therefore cannot silently reuse old results.
8. Summarize branch/final metrics and perform paired comparison only on aligned
   sample indices.

The old `materialize_one_pass.py` path is intentionally outside formal RL data
generation: it reads completed three-stage traces and exists to reproduce the
original architecture comparison. Its v1 audit remains acceptable for
read-only evaluation, while every new training launch requires audit v2.

## Current state

- All three visible-prefetched base-valid sets finished: 820/820 outputs.
- Formal Bio train data contains 1,674 audited rows and is shared byte-for-byte
  by hosted/local backends despite its historical `rl_lora_gpt_oss_120b` path.
- The hosted 120B training task was intentionally stopped and must not be
  resumed automatically.
- The local 20B NeMo Bio run was also intentionally stopped after metrics were
  recorded through step 174; its last complete checkpoint is step 160. It must
  not be resumed automatically.
- No local checkpoint-valid promotion or test evaluation was performed. The
  partial-run scalar receipt is recorded in `LOCAL_NEMO_RUNBOOK.md`.
- Reward version remains `one_pass_hierarchical_reward.v2`; this cleanup does
  not change any reward number.

## Known scientific limitation

There is no independent gold label for the single or analog branch. A branch is
currently called correct when its binary prediction equals the final task gold
label. Analysis text is schema-checked but not semantically graded. Therefore
branch reward is useful shaping, not evidence that each intermediate medicinal
chemistry explanation is factually correct. Any semantic reward redesign must
use a new reward version and remain shared by Tinker and NeMo.
