# E18: one-pass full-flat LoRA-RL contract

Status update (2026-08-11): prompt/data/base-valid and bounded mechanical gates
are complete. The hosted 120B training run was intentionally stopped and must
not be resumed automatically. The active training path is Bio-only GPT-OSS-20B
on local NeMo RL with the same one-pass reward and shared training profile.
Current code ownership and launch gates are documented in
`ONE_PASS_REASONING.md`; this file remains the scientific design contract.

## Question and comparison order

E18 tests whether the current single, flat-group, and final LLM stages can be fused into one generation without
losing current-valid performance, and whether RL over that complete generation improves held-out performance more
than historical final-only RL. It does not overwrite E17 or the canonical three-stage pipeline.

The required order is:

1. Build `one_pass_full_flat.v1` prompts from the exact current-valid LLM-visible query/tool/evidence payloads.
2. Compare frozen GPT-OSS-120B one-pass against the aligned current three-stage full-flat predictions.
3. Build all-train one-pass rows with the same global `parent_disjoint` Full Flat retrieval used by the canonical three-stage valid condition.
4. Train task-specific rank-32 all-module LoRA adapters on all train rows and select checkpoints on valid. The
   earlier rank-16 attention-only run is a bounded observability smoke, not the formal adapter configuration.
5. Freeze adapter, decode, prompt, parser, and reward; then compare once on scaffold test.

## One-pass information contract

Retrieval and tool preparation remain outside the model. The one model-visible user payload contains:

- anonymous query plus harness-prefetched `molecule_properties`;
- the same bounded `Flat.all_evidence` neighbor cards used by the current group branch;
- harness-prefetched `mmp_structure_compare` and `properties_compare` text for every retained neighbor;
- retrieval coverage and the current task-specific single, group, and final instructions;
- no source assistant answer and no gold label.

The response keeps the task's current final schema and adds four fields:

```text
single_prediction
single_analysis
analog_prediction
analog_analysis
```

The two prediction fields use the same task values as the final prediction (`pass/fail`, `high/low`, or
`risk/no_risk`). `single_prediction` must use query properties only. `analog_prediction` may use query properties
only to judge analog transferability, matching the current group stage.

There is no independent branch gold dataset. In the reward table, a "correct"
single or analog prediction means that the branch's binary outcome matches the
private task gold label. The semantic analysis text itself is not graded as
true/false; that would require a separate curated supervision contract.

## Reward contract

Reward version: `one_pass_hierarchical_reward.v2`. Version 2 keeps the same
numeric hierarchy and additionally requires every deterministic nested
single/analog schema key before awarding the `+0.03` completeness bonus.

| component | value |
|---|---:|
| final prediction correct | +1.00 |
| final prediction wrong/missing | -1.00 |
| single prediction correct | +0.15 |
| analog prediction correct | +0.15 |
| single/analog disagree and final prediction is correct | +0.15 |
| parseable JSON | +0.02 |
| complete top-level plus nested branch schema | +0.03 |

There is no semantic hallucination classifier, no generic consistency reward, and no reward for citing a valid
neighbor identifier. Branch disagreement is logged. The extra `+0.15` applies only when
`conflict_resolved_correctly=true`; this makes a correctly adjudicated one-right/one-wrong conflict tie
both-branches-correct, without rewarding an incorrectly resolved conflict.

The possible score ranges preserve the intended hierarchy:

```text
final correct: +1.00 to +1.35
final wrong:   -1.00 to -0.65
```

Thus a wrong final decision can never outscore a correct final decision. Among final-correct trajectories,
both-branches-correct and one-right/one-wrong with correct conflict resolution tie above zero correct branches.

## Selection and reporting

Valid is explicitly a development set for E18. Report frozen base one-pass, historical frozen three-stage,
one-pass LoRA checkpoints, and historical final-only LoRA separately. Select one checkpoint per task by valid
macro-F1, breaking ties by accuracy then earlier checkpoint. Do not average or tune on test.

Report per task accuracy, macro-F1, class recall, schema success, branch accuracy, branch disagreement resolution,
paired bootstrap macro-F1 delta, McNemar, token usage, wall time, and hosted billing receipts.
