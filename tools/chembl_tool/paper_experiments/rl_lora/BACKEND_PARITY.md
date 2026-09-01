# Tinker and NeMo backend parity

## Shared scientific contract

The current profile is `one_pass_bio_grpo.v1`. Prompt visibility remains a
separate versioned data contract. These fields are owned
by `experiment_contract.py` and must match for an aligned run:

| Field | Frozen value |
|---|---:|
| reward | `one_pass_hierarchical_reward.v2` |
| prompts per optimizer step | 4 |
| generations per prompt | 8 |
| rollout batch | 32 |
| maximum completion | 6,144 tokens |
| temperature / top-p | 1.0 / 1.0 |
| learning rate | 1e-4 |
| LoRA rank / target profile | 32 / all |
| seed | 42 |
| reference KL penalty | 0 |
| reward normalization / leave-one-out baseline | true / true |

The reward weights are one shared object: final correctness is `+1/-1`, each
correct branch contributes `+0.15`, correct final resolution of a branch
conflict contributes `+0.15`, and JSON/full-schema bonuses are `+0.02/+0.03`.
Changing a weight or parser therefore changes both providers simultaneously.

Both adapters follow this exact call path:

```text
materialized row
  -> reward_metadata_from_row()
  -> score_training_response()
  -> score_one_pass_response()
  -> shared result-to-metrics/log mapping
```

The NeMo config validator resolves config inheritance and compares the
effective values, so a copied YAML value cannot silently drift. The Tinker
launcher performs the same comparison against its command-line settings.

The currently audited materialized Bio rows still live below the original
`rl_lora_gpt_oss_120b` artifact namespace. That path name is historical; the
rows contain no model weights and are deliberately reused byte-for-byte by
20B through their adjacent SHA-256 audit. Do not duplicate them into a second
model directory ad hoc. Both formal runs are now stopped, so there is no
active-run blocker; a future artifact migration should use one model-neutral
data namespace and preserve the existing hash receipt.

## Backend/model-specific settings

These remain explicit provider differences and are not reward differences:

| Area | Tinker 120B | local NeMo 20B | Reason |
|---|---|---|---|
| model/context | hosted 120B, 128K | local 20B, 65,536 | model and hardware capacity |
| topology | hosted service | 8 x A100, TP=4/EP=2 train and TP=8 generation | runtime implementation |
| optimizer beta2 | 0.95 | 0.999 inherited from pinned NeMo | provider optimizer default; changing it would be a new run contract |
| constant-reward groups | removed before Tinker update | zero advantage under NeMo normalization | framework behavior; compare effective batch diagnostics |
| response transport | renderer extracts generated assistant response | adapter concatenates assistant message content | parser tolerates surrounding text, but truncation/stop receipts must still be compared |
| checkpoint format | hosted sampler/state paths | NeMo/Megatron optimizer and adapter checkpoints | backend storage |

These differences mean a 20B-vs-120B result is not a pure model-size ablation.
Reward, data, sampling group shape, and core LoRA settings are aligned; runtime
and model-capacity differences must remain in the result manifest.

## Change rule

- Scientific reward or shared training changes start in
  `experiment_contract.py` and require backend-contract tests.
- Provider-only runtime changes stay in `run_tinker.py`, `nemo_adapter.py`, or
  NeMo YAML and must be recorded as backend-specific.
- Do not edit `one_pass_reward.py` in one backend branch or add reward math to
  an adapter.
- A deliberate shared-setting ablation uses a new profile/version. It must not
  masquerade as the frozen comparison.

This organization guarantees implementation parity; it does not claim the
current reward is already scientifically optimal. Reward redesign should be a
separate versioned step after the present training behavior is reviewed.
