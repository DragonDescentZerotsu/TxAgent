# Visible-prefetched one-pass valid and RL smoke receipt

Status: complete on 2026-08-10. The current scaffold-valid sets completed
`820/820` with zero failed outputs. A bounded three-step Bio RL smoke and
checkpoint reload passed under the historical rank-16 attention-only smoke
configuration. The user subsequently selected rank-32 all-module LoRA, and its
matched three-step hosted smoke, 96-rollout length audit, W&B sync, checkpoint
export, and valid-3 reload also passed. After this receipt, the hosted 120B
all-train run was started and intentionally stopped; it must not be resumed.
The local NeMo 20B Bio run is active, and every test split remains untouched.

## Contract

- prompt: `one_pass_full_flat_visible_prefetched.v1`
- model: `openai/gpt-oss-120b:peft:131072`
- visibility: query/neighbor structures and allowed source IDs visible
- identity policy: global `parent_disjoint`
- tools: harness-prefetched `molecule_properties`, `mmp_structure_compare`, and
  `properties_compare`; no live LLM tool surface
- labels: gold exists only in private reward/evaluation metadata
- current profiles: BBB `meaningful_cns_access_v1`, Bio
  `f20_evidence_calibrated_v2`, Skin `sensitization_aligned_v2`
- Skin index: sensitization/contact-allergy scoped v2
- generation: greedy base evaluation, 6,144 maximum output tokens, up to four
  schema attempts

## Current base-valid result

| task | n | macro-F1 | accuracy | single F1 | analog F1 | attempts 1 / 2 |
|---|---:|---:|---:|---:|---:|---:|
| BBB | 366 | 0.6825 | 0.7213 | 0.6630 | 0.6801 | 363 / 3 |
| Bioavailability | 209 | 0.6778 | 0.7129 | 0.5317 | 0.6605 | 207 / 2 |
| Skin Reaction | 245 | 0.6276 | 0.6571 | 0.6235 | 0.5989 | 245 / 0 |

All five retries succeeded on attempt two. No output reached the 6,144-token
cap. Compared with the current blind one-pass rows, macro-F1 changes were BBB
`+0.0074`, Bio `+0.0167`, and Skin `+0.0591`. The paired 95% bootstrap
intervals were respectively `[-0.0332,+0.0489]`, `[-0.0342,+0.0657]`, and
approximately `[0,+0.1185]`. BBB/Bio keep the matched index; Skin also adopts
the subsequently frozen scoped-v2 index, so its delta is not a pure visibility
effect.

## Comparison with previous visible three-stage artifacts

| task | previous visible result | current visible one-pass | interpretation |
|---|---:|---:|---|
| BBB | full-flat 0.6814 | 0.6825 | descriptive only: previous BBB used historical record-supported gold and n=500 |
| Bioavailability | full-flat 0.6787 | 0.6778 | nearly identical point estimate, but previous prompt/tool-execution contract was legacy agentic visible |
| Skin Reaction | full-flat 0.5981; best mechanism 0.6151 | 0.6276 | current one-pass is higher, but also uses current aligned prompt and scoped-v2 evidence index |

The historical comparison is not a paired architecture test. The old visible
runner allowed model-selected comparison tool calls; the new upper-bound
control fixes the exact same tool bundle in the harness. BBB also changed gold
lineage, while Skin changed evidence scope. These rows answer whether the new
system is in the range of prior visible performance, not whether one-pass is
statistically superior to the old three-stage system.

## Data, tool, length, and cost gate

The three audits contain 10,640 explicit tool receipts: 10,597 successful and
43 deterministic missing-evidence receipts. Error receipts contain only the
generic unavailable-result message; gold, source assistant answers, and live
tools are absent from all prompts.

| task | prompt total | p50 / p95 / max | context overflow |
|---|---:|---:|---:|
| BBB | 6,421,334 | 17,259 / 30,727 / 38,264 | 0 |
| Bioavailability | 5,376,481 | 28,803 / 41,058 / 44,696 | 0 |
| Skin Reaction | 6,079,773 | 28,560 / 37,742 / 47,383 | 0 |

Actual evaluation attempts consumed 17,995,238 prompt tokens, including
263,424 cache-hit tokens, and 2,621,469 completion tokens. At the current
public 128K rates this is an estimated `$18.96`; Tinker's billing export can
lag by hours, so this token-derived estimate is retained until the matching
session rows appear.

## Bounded RL observability gate

| step | all-rollout reward | final / single / analog correct | retained reward | signed loss sum | entropy | KL v2 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7750 | 0.7500 / 0.2812 / 0.7500 | 1.3500 | 0.0025 | 0.3560 | 0.0101 |
| 1 | 1.0038 | 0.8438 / 0.8750 / 0.8438 | 0.6575 | 1587.3147 | 0.3237 | 0.0124 |
| 2 | 0.7059 | 0.7188 / 0.6875 / 0.7188 | 1.2119 | -2969.0113 | 0.3221 | 0.0120 |

The signed importance-sampling loss is allowed to cross zero. Different
molecule batches make these three train rewards non-comparable as a learning
curve; the smoke validates that every metric is emitted and synchronized, not
that reward is already improving. The W&B run is:
<https://wandb.ai/reasonv/txagent-one-pass-rl/runs/jjwsc0d8>.

The step-3 sampler checkpoint loaded successfully and produced three complete,
first-attempt Bio valid outputs. Local curves and their machine-readable audit:

```text
outputs/paper/rl_lora_gpt_oss_120b/one_pass_full_flat_visible_prefetched_v1/
  tinker_smoke/bio_3step_rank16_attention_128k/
    training_curves.png
    training_curves.json
    metrics.jsonl
    checkpoints.jsonl
    eval_valid3/
```

## Formal Bio train data and forecast

The complete visible-prefetched train artifact has 1,674 rows with 461 label-0
and 1,213 label-1 examples. Its audit found 24,445 successful tool receipts,
241 generic missing-evidence receipts, no gold/live-tool leak, and no context
overflow. Prompt lengths are min/p50/p90/p95/p99/max
`2,473 / 30,576 / 39,001 / 40,668 / 43,683 / 48,301`; total unique prompt
tokens are 44,148,331.

With eight rollouts, seven cache hits after each first prefill, the smoke's
2,983.9-token mean completion, and its 41.7% retained-rollout fraction, a full
419-step Bio pass is approximately `$542`; the every-group-retained ceiling is
`$1,076`. Three valid checkpoint evaluations and one selected test evaluation
add about `$76`, giving a campaign forecast of roughly `$618-$1,152`, plus
storage and queue variation. The smoke's 60.6-second average step implies about
7.0 train hours before queue/checkpoint-evaluation time.

```text
outputs/paper/rl_lora_gpt_oss_120b/one_pass_full_flat_visible_prefetched_v1/
  data/train/
    bioavailability_ma.jsonl
    bioavailability_ma.manifest.json
    bioavailability_ma.audit.json
    bioavailability_ma.tokens.json
```

## Gate decision

Implementation, full base-valid, private-label/tool audits, 128K context,
multi-step training, signed loss export, all-rollout reward export, W&B sync,
checkpoint export, and checkpoint reload all pass for rank-16 attention-only.
The subsequent rank-32 all-module matched smoke also passes: 96/96 clean stop,
95/96 parseable JSON, 92/96 full schema, maximum 4,911 completion tokens, zero
6,144-token cap hits, three optimizer steps, W&B, step-3/final checkpoints, and
3/3 first-attempt full-schema valid reload. The mechanical gate is therefore
complete and the next action is explicit user confirmation for the formal Bio
run. No test split has been read.
