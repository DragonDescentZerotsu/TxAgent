# Local NeMo RL runbook

This runbook records the pinned local NeMo RL environment. It was originally
proven with the historical GPT-OSS-120B feasibility gate; the active Bio-only
GPT-OSS-20B one-pass LoRA-GRPO job now uses the same environment on `node002`.
The initial gate was completed on 2026-08-10. It is an operational receipt,
not a claim that the one-step smoke improved model quality.

## Proven host and pinned software

| item | pinned value |
|---|---|
| host | `node002`, 8 x NVIDIA A100-SXM4-80GB |
| interconnect | NVSwitch/NVLink single node |
| NeMo RL checkout | `/local/tianang/nemo-rl` |
| NeMo RL commit | `037fb36bdad40f6336edcdfd39f7c5150b152324` |
| driver Python | 3.13.14 |
| PyTorch / CUDA | 2.11.0+cu130 / CUDA 13.0 |
| Ray | 2.56.1 |
| policy-worker TE | 2.15.0+42b84005 |
| HF snapshot | `/data1/tianang/cache/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a` |
| converted Megatron checkpoint | `/local/tianang/txagent_rl_lora/megatron_cache/model__data1_tianang_cache_hub_models--openai--gpt-oss-120b_snapshots_b5c939de8f754692c1647ca79fbf85e8c1e70f8a` |

Current local footprint is approximately 581 MiB for the NeMo checkout,
31 GiB for worker environments, and 218 GiB for the converted checkpoint.

## Runtime layout

```text
/local/tianang/txagent_rl_lora/
  home/ cache/ tmp/                     isolated process state
  worker_venvs/                         Ray component environments
  megatron_cache/                       converted GPT-OSS checkpoint
  data/                                 node-local materialized prompt rows
  logs/                                 driver, TensorBoard, and train traces
  checkpoints/                          NeMo optimizer/adapter checkpoints
  receipts/                             service stop/start and host audits
```

Do not put credentials in this tree.  Do not use a shared `/tmp` or the login
home for Triton, TorchInductor, vLLM, uv, or CUDA caches: concurrent jobs and
quota pressure made those locations unreliable during bring-up.

## How the working environment was assembled

The environment was first resolved from the pinned NeMo RL checkout and
official `uv.lock` on node001, then copied to node002 with
`scripts/sync_node001_to_node002.sh`.  That script copies the checkout, uv
Python/runtime, component worker environments, converted checkpoint, and
mechanical data, then validates both the driver and Megatron worker imports.
It is a bootstrap/recovery tool, not a per-run command.

The GPT-OSS HF checkpoint is imported through `convert_gpt_oss_checkpoint.py`.
The wrapper delegates to Megatron Bridge but disables gradient-accumulation
fusion during conversion: GPT-OSS retains a native `ColumnParallelLinear`
output projection and otherwise requests an Apex fused weight-gradient
extension that is irrelevant for checkpoint import and absent here.

Before launching, `scripts/run_node_smoke.sh` isolates every mutable cache and
binds the policy worker's bundled CUDA 13 toolkit.  This is required because a
long-lived tmux server may retain CUDA 12.8 paths while the generated NeMo
workers use PyTorch cu130.

Important exported values are:

```text
HOME=/local/tianang/txagent_rl_lora/home
NEMO_RL_ROOT=/local/tianang/nemo-rl
NEMO_RL_VENV_DIR=/local/tianang/txagent_rl_lora/worker_venvs
NRL_MEGATRON_CHECKPOINT_DIR=/local/tianang/txagent_rl_lora/megatron_cache
NRL_REFIT_BUFFER_MEMORY_RATIO=0.05
HF_HOME=/data1/tianang/cache
CUDA_HOME=<MegatronPolicyWorker venv>/site-packages/nvidia/cu13
```

The script also isolates `XDG_CACHE_HOME`, `TMPDIR`, `PIP_CACHE_DIR`,
`UV_CACHE_DIR`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`,
`VLLM_CACHE_ROOT`, `TORCH_EXTENSIONS_DIR`, and `CUDA_CACHE_PATH` beneath the
runtime root.

## Required upstream patches

Four small patches are versioned under `patches/` and must apply cleanly to the
pinned NeMo commit:

1. `nemo_rl_refit_empty_cache.patch` releases unused cached blocks before the
   fallback oversized IPC allocation.
2. `nemo_rl_refit_zero_copy_oversized.patch` avoids a second 4.25 GB CUDA copy
   for an exact, contiguous, self-owned EP-resharded expert tensor.  It exports
   the existing storage only when CUDA, contiguity, zero storage offset, and
   exact storage size all hold; receiver ACK remains the lifetime boundary.
3. `megatron_bridge_grouped_expert_lora_export_order.patch` merges grouped-MoE
   adapters in Megatron layout before GPT-OSS applies its HF export transpose
   and uninterleave mapping. This is required by all-module LoRA: merging after
   conversion shape-mismatches FC1 and silently transposes square FC2 updates.
4. `nemo_rl_grpo_ragged_final_batch.patch` preserves the final incomplete
   prompt-group batch and passes its actual rollout count to Megatron training.
   Bio has 1,674 rows and four groups per step, so step 419 contains two groups;
   the default NeMo loader would otherwise silently drop those rows.

The second patch contains the first patch's fallback behavior.  Apply the
zero-copy patch to a clean pinned checkout; do not stack the first two patches
blindly. The grouped-expert patch targets the nested Megatron Bridge checkout
and is independent of the zero-copy NeMo patch.
After any NeMo update, re-run the one-step refit gate before longer training.

## Proven topology and launch

The Bio-only GPT-OSS-20B topology proven by the exact one-step smoke is:

```text
policy: Megatron Core, BF16, TP=4, EP=2, PP=1, sequence parallel
LoRA: linear_qkv + linear_proj + linear_fc1 + linear_fc2 + output_layer,
      rank=32, alpha=32
rollout: colocated vLLM, TP=8, eager, Triton MoE
context/output: max input 65,536; max completion 6,144
activation checkpointing: enabled
```

Run only after confirming all eight GPUs are intentionally available and after
stopping only the four owned GPT-OSS vLLM backends and their dedicated HAProxy.
Preserve unrelated users' processes and the resident tool service on port 8765.

```bash
ssh node002
cd /data1/tianang/Projects/TxAgent
bash tools/chembl_tool/paper_experiments/rl_lora/scripts/run_node_smoke.sh <receipt-label>
```

The launcher must be the only NeMo/Ray launcher using the node.  A successful
receipt requires all of the following, not merely a driver process:

- generation and deterministic reward complete;
- policy logprobs, GRPO advantage, LoRA backward, and optimizer step complete;
- async checkpoint reports the requested step;
- driver exits successfully and Ray actors disappear;
- checkpoint metadata, train trace, config, and full log are retained.

## 2026-08-11 GPT-OSS-20B Bio receipt and formal launch

The exact Bio configuration completed a full one-step gate, an independent
consecutive two-step gate, and a final two-step gate drawn from the complete
1,674-row shuffled training dataset with the rank-32 all-module adapter.
Generation, hierarchical reward, policy logprobs, backward, optimizer update,
the second weights/KV-cache refit, and checkpoint save all passed. The formal
configuration uses `gpu_memory_utilization=0.40`: it exposes a 7,723,865-token
vLLM KV cache (117.86 concurrent 65,536-token requests) while leaving a wide
post-update Megatron/refit memory margin. Observed KV usage in the complete-data
gate peaked at only 1.9%.

```text
/local/tianang/txagent_rl_lora/checkpoints/gpt_oss_20b/one_pass_visible_bio_smoke/step_1
/local/tianang/txagent_rl_lora/logs/smoke_driver_gpt_oss_20b_one_pass_bio_smoke_retry12.log
/local/tianang/txagent_rl_lora/checkpoints/gpt_oss_20b/one_pass_visible_bio_2step/step_2
/local/tianang/txagent_rl_lora/logs/smoke_driver_gpt_oss_20b_one_pass_bio_2step.log
/local/tianang/txagent_rl_lora/checkpoints/gpt_oss_20b/one_pass_visible_bio_full_2step/step_2
/local/tianang/txagent_rl_lora/logs/smoke_driver_gpt_oss_20b_one_pass_bio_full_2step.log
```

The formal Bio-only run uses the same 1,674-row visible-prefetched data as the
hosted 120B run. It has 419 steps: 418 batches of four prompt groups and one
ragged final batch of two prompt groups. The durable session and log are:

```text
tmux: txagent_gpt20b_bio_nemo_rl
/local/tianang/txagent_rl_lora/logs/smoke_driver_gpt_oss_20b_one_pass_bio_formal.log
W&B: https://wandb.ai/reasonv/txagent-one-pass-rl/runs/eajazu8s
```

## 2026-08-10 receipt

The proven one-step run used one real Bioavailability train-OOF prompt with
eight generations, 4,096 total tokens and 256 new tokens.  It completed in
83.48 s: generation 22.49 s, logprob 25.52 s, policy training 12.87 s, and
checkpoint 2.70 s.  The 100 MiB checkpoint is at:

```text
/local/tianang/txagent_rl_lora/checkpoints/smoke/step_1
```

The driver log and training trace are:

```text
/local/tianang/txagent_rl_lora/logs/smoke_driver_node002_06.log
/local/tianang/txagent_rl_lora/logs/smoke/exp_006/train_data_step1.jsonl
```

All eight generations were truncated and received `-1`, giving zero normalized
advantage and zero loss.  This proves mechanics only.  The next short run must
show mixed rewards, nonzero advantage, finite nonzero gradient norm, and at
least one changed LoRA tensor.  It must also cleanly tear down: the receipt run
saved its checkpoint but later reported one CUDA allocator error during vLLM
actor cleanup.

A later exact-full-prompt diagnostic used a 3,135-token BBB prompt and eight
768-token rollouts.  One full generation/reward/logprob/backward/optimizer
step completed in 117.04 s (generation 46.77 s, logprob 28.49 s, policy
training 15.68 s, initial/refit preparation 24.37 s), but all rollouts again
hit the artificial 768-token cap, so loss remained zero.  A second-step refit
then OOMed while stacking a 3.96 GiB expert tensor after allocator
fragmentation.  This is a negative multi-step gate, not a successful training
receipt.

Historical current-valid traces show that real GPT-OSS final outputs reach
870/856/977 tokens at maximum for BBB/Bioavailability/Skin, while complete
prompt+response sequences reach about 5K tokens.  The local retry must preserve
the original prompt, set the GPT-OSS `<|return|>` token (`200002`) explicitly,
use an 8K-or-larger total sequence budget, and give generation at least 2K
tokens.  Do not infer a need to shorten the prompt from the 768-token failure.

## Timing protocol for full-train estimates

Do not extrapolate from model load or the zero-advantage receipt alone.  Use a
short run with real full-flat prompts after warm-up, record steps 2 onward, and
report separately:

- prompts/step and generations/prompt;
- prompt and completion token distributions;
- generation, logprob, policy-train, refit, checkpoint, and total step seconds;
- effective prompt rows/hour and generated tokens/second;
- fixed startup/conversion time versus marginal training time.

Estimate each task from its exact materialized train row count and the frozen
epoch/step budget.  Include a range based on observed p50 and slow-step p90;
never multiply the 83.48 s mechanical step by dataset size without matching its
token and batch shape.

## Cleanup and service restoration

After a run, verify no NeMo Ray process remains.  Restore the established
node002 service topology only after GPU memory is free:

```text
GPUs 0-1 -> vLLM 8001
GPUs 2-3 -> vLLM 8002
GPUs 4-5 -> vLLM 8003
GPUs 6-7 -> vLLM 8004
HAProxy  -> 127.0.0.1:9001
tools    -> 127.0.0.1:8765 (never stopped for RL)
```

Completion means all four `/v1/models` endpoints and proxy 9001 return
`gpt-oss-120b`, and `/health` on 8765 remains healthy.  NFS weight prefetch can
make simultaneous backend restoration take several minutes; wait for endpoint
health rather than treating tmux session existence as readiness.
