# NeMo RL config map

The frozen formal config root is `grpo_gpt_oss_20b_one_pass_bio.yaml`. It
contains the complete local 20B Bio recipe used by the stopped partial run. The
shorter gates inherit it:

- `grpo_gpt_oss_20b_one_pass_bio_smoke.yaml`: 12-row, one-step mechanical gate.
- `grpo_gpt_oss_20b_one_pass_bio_2step.yaml`: same smoke data, two steps.
- `grpo_gpt_oss_20b_one_pass_bio_full_2step.yaml`: full data, two-step gate.

Derived configs must only override run length, data subset, checkpoint cadence
and destination, or logging destination. Shared reward/sampling/LoRA fields are
checked by `validate_backend_contract.py` before launch.

`grpo_gpt_oss_120b_lora_smoke.yaml` and
`grpo_gpt_oss_120b_lora_feasibility.yaml` are historical local NeMo E17
feasibility configs. They are not the current 120B path; current hosted 120B RL
used Tinker before it was intentionally stopped. Keep them for receipts, but do
not derive or resume one-pass runs without a new explicit user decision.
