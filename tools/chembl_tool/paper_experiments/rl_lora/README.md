# Starling one-pass LoRA RL

This directory owns the train-only Starling RL experiments and keeps the
scientific contract separate from provider runtime code. The active comparison
is Bioavailability one-pass RL: hosted GPT-OSS-120B uses Tinker and local
GPT-OSS-20B uses NeMo RL. Historical E17 final-only and local 120B NeMo
feasibility artifacts remain reproducible but are not active launch defaults.

Gold labels are environment-private reward metadata. They never enter a
model-visible prompt. Training and tuning use scaffold-train only; checkpoint
selection may use the frozen scaffold-valid contract, and test remains closed
until a separate promotion decision.

## Ownership layers

```text
prompt/data contract
  one_pass_contract.py, one_pass.py, materialize*_one_pass.py, audit_one_pass_*.py
                         |
shared inference runtime
  one_pass_runtime.py (audit, nested schema, prediction and resume identity)
                         |
shared experiment contract
  experiment_contract.py (reward version, weights, shared GRPO/LoRA settings)
                         |
shared reward surface
  training_contract.py -> one_pass_reward.py / historical reward.py
                         |
backend adapters
  run_tinker.py                     nemo_adapter.py -> run_grpo.py
  hosted 120B                       local 20B
                         |
evaluation and diagnostics
  evaluate_tinker_one_pass.py, summarize_one_pass.py,
  plot_training_curves.py, probe_reward_variance.py
```

`training_contract.py` is the only supported reward dispatch. Both adapters
must project rows with `reward_metadata_from_row()` and score responses with
`score_training_response()`. Backend code may transport tokens and schedule
workers, but it must not implement reward logic.

`experiment_contract.py` freezes the settings that must match for the current
cross-backend one-pass comparison. Tinker refuses one-pass deviations unless
an explicit ablation flag is passed. Current NeMo 20B launch configs are
validated before any NeMo worker is created, and the effective receipt is
written as `backend_contract.json` under the run log directory.

See [BACKEND_PARITY.md](BACKEND_PARITY.md) for the exact shared fields and the
remaining intentional backend differences.

See [ONE_PASS_REASONING.md](ONE_PASS_REASONING.md) for the RL-specific
reasoning lifecycle and the distinction between the canonical direct
materializer and the historical trace-replay builder.

## Active and historical entry points

| Status | Model/backend | Entry point | Contract |
|---|---|---|---|
| active | GPT-OSS-20B / local NeMo | `run_grpo.py` + `configs/grpo_gpt_oss_20b_one_pass_bio.yaml` | visible-prefetched Bio one-pass |
| active comparison | GPT-OSS-120B / hosted Tinker | `run_tinker.py` | same one-pass reward/profile |
| historical | GPT-OSS-120B / local NeMo | `configs/grpo_gpt_oss_120b_lora_*.yaml` | E17 feasibility only |
| historical | final-only E17 | `materialize.py`, `reward.py` | frozen final-synthesis contract |

Short NeMo gates inherit the formal 20B config instead of copying it. Their
only allowed differences are step count, data subset, checkpoint directory,
and logging destination. The config-specific map is in
[`configs/README.md`](configs/README.md).

## Documents

- `AGENTS.md`: non-negotiable lineage, privacy, and backend-sharing rules.
- `ONE_PASS_IMPLEMENTATION_PLAN.md`: one-pass scientific design and gates.
- `ONE_PASS_REASONING.md`: current code ownership and runnable lifecycle.
- `LOCAL_NEMO_RUNBOOK.md`: pinned local runtime, patches, and receipts.
- `VISIBLE_ONE_PASS_VALID_RESULTS.md`: visible-prefetched base-valid/RL receipts.

Large checkpoints, Ray state, generated worker environments, and caches stay
under `/local/tianang/txagent_rl_lora`. Repository code, small manifests,
audits, and reports stay here.
