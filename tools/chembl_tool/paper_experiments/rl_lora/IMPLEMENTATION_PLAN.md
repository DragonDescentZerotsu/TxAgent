# E17: GPT-OSS-120B train-only LoRA-GRPO implementation plan

## Research question

在不改变 current Starling data、retrieval、tools、single/group branch 和 final JSON schema 的条件下，
仅用 scaffold-train gold reward 优化 GPT-OSS-120B 的 final synthesis policy，是否能显著改善 current
scaffold-valid 的 macro-F1，并减少无证据推翻强 analog signal 的错误？

这个实验优先针对现有 trace 暴露的 calibration failure，而不是声称重新学习完整 medicinal chemistry。
Phase 1 只训练 final synthesis，因此不会把 branch 变化和 final decision 变化混在一起。

## Frozen data contract

| task | current train | current valid | train lineage | Phase 1 evidence condition |
|---|---:|---:|---|---|
| BBB_Martins | 2,935 | 366 | `experimental_meaningful_cns_access_v2` | `starling_full_flat` |
| Bioavailability_Ma | 1,674 | 209 | `record_supported_v2` | `starling_full_flat` |
| Skin_Reaction | 1,966 | 245 | `record_supported_v2` | `starling_full_flat` |

用 deterministic 5-fold OOF 对每个 task 独立生成 final-synthesis prompts。每个 query 的 retrieval 固定为
`mixed_family_identity_policy.v1`：direct family 排除同 parent 和同 Bemis-Murcko scaffold neighbor；其它
full-flat family 只排除同 parent，允许同 scaffold mechanism neighbor。该 policy 与 current valid/test 的
可见性一致，不能用全局 `scaffold_disjoint` 误伤 non-direct evidence。对 fold `i`：

- query 只来自 fold `i`；
- direct retrieval 排除 current valid/test、fold `i` query parents，并逐 query 执行 scaffold-disjoint；
- non-direct retrieval 排除 current valid/test 和 query parent，但不执行 scaffold exclusion；
- evidence、single/group outputs 和 final user prompt 都由冻结 base GPT-OSS-120B 生成；
- 保存最终 system/user messages，但删除 base assistant response；
- gold label 仅作为 environment-private reward metadata，不出现在 prompt。

fold 0-3 用于首轮 GRPO optimization，fold 4 是 train-only development gate。配置冻结后，可按固定 step budget
在全部五 fold prompt 上重新拟合一个 final adapter；这个 refit 不再选择超参数。scaffold-valid 只做一次最终
paired readout。

## Reward contract

每条 generation 先按 task schema 确定性解析 prediction：

- exact label correct: `+1.0`
- exact label incorrect: `-1.0`
- valid JSON object: `+0.05`
- 完整包含 task required keys 且取值在枚举内: `+0.05`
- JSON 无法解析或 prediction 缺失: label reward 视为 incorrect，并且 format bonus 为 0

Format shaping 总幅度不得超过 `0.1`，避免模型以格式优化替代分类。训练 sampler 对 task 和 gold class 分层，
不通过更改 metric 分母来处理 class imbalance。reward manifest 保存实现版本和 SHA-256。

## Hardware/config ladder

环境固定到 NeMo RL commit `037fb36bdad40f6336edcdfd39f7c5150b152324`，使用官方 `uv.lock`。

1. `parse`: CPU config/dataset/environment import。
2. `convert`: 8 GPU HF -> Megatron conversion，不做 optimizer step。
3. `lora-fwd-bwd`: `EP=8, TP=1, PP=1`、attention-only LoRA、micro-batch 1、sequence 2,048。
4. `refit`: colocated vLLM `TP=8`，验证 base/rollout logprob tolerance 和一次 LoRA weight refit。
5. `grpo-1`: 1 prompt x 2 generations x 1 step。
6. `grpo-10`: 2 prompts x 4 generations x 10 steps；要求 step 2-10 peak memory 不持续增长。
7. `development`: 4-fold train，fold-4 readout；先 2,048，再只在 prompt coverage 不足时升 4,096/8,192。

首个 LoRA 候选：

```yaml
policy:
  model_name: openai/gpt-oss-120b
  precision: bfloat16
  train_micro_batch_size: 1
  logprob_batch_size: 1
  megatron_cfg:
    enabled: true
    expert_model_parallel_size: 8
    tensor_model_parallel_size: 1
    pipeline_model_parallel_size: 1
    sequence_parallel: false
    activation_checkpointing: true
    peft:
      enabled: true
      target_modules: [linear_qkv, linear_proj]
      dim: 16
      alpha: 32
  generation:
    vllm_cfg:
      tensor_parallel_size: 8
cluster:
  num_nodes: 1
  gpus_per_node: 8
```

如果 attention-only 无法形成 train-only signal，不直接读取 valid 搜索配置。只允许在 fold-4 上预先登记的第二候选：
加入 `linear_fc1/linear_fc2`、`rank=8`；仍失败则终止 E17，不做 valid-driven sweep。

## Mechanical feasibility receipt (2026-08-10)

`node002` 的单节点 `8 x A100-SXM4-80GB` 已完成真实 NeMo RL colocated smoke：

- NeMo RL commit: `037fb36bdad40f6336edcdfd39f7c5150b152324`；
- policy: `EP=8, TP=1, PP=1`，BF16，activation checkpointing；
- LoRA: `linear_qkv + linear_proj`, rank 16, alpha 32；每 rank 16,473,883,968 parameters，
  其中 8,626,176 trainable (`0.05%`)；
- rollout: vLLM `TP=8`, `gpu_memory_utilization=0.40`, eager mode；
- batch: 1 train-only prompt x 8 generations，256 new tokens，1 optimizer step；
- 8 个 mechanical rows 只来自 Bioavailability train OOF fold 4，未读取 current valid/test label；
- generation、reward、policy logprob、LoRA backward、optimizer step、async checkpoint 全部完成；
- step wall time 83.48 s；generation 22.49 s，logprob 25.52 s，policy training 12.87 s；
- checkpoint receipt: `/local/tianang/txagent_rl_lora/checkpoints/smoke/step_1`，约 100 MiB，
  `latest_checkpoint_status.json` 记录 `last_checkpoint_step=1`；
- full driver log: `/local/tianang/txagent_rl_lora/logs/smoke_driver_node002_06.log`；
- train trace: `/local/tianang/txagent_rl_lora/logs/smoke/exp_006/train_data_step1.jsonl`。

GPT-OSS-120B 的 EP reshard 会产生最大 4,246,732,800-byte expert tensor；原 NeMo RL IPC refit 会再复制
一份同尺寸 CUDA buffer，导致 OOM。当前 smoke 使用受限零拷贝补丁：只有 tensor contiguous、storage offset=0、
底层 storage 大小恰好等于 tensor 大小时才直接导出 CUDA IPC byte view，并在 receiver ACK 后释放；其它 tensor
保持原复制路径。补丁位于 `patches/nemo_rl_refit_zero_copy_oversized.patch`。

本次 reward 全为 `-1`：256-token mechanical outputs 均被截断，归一化 advantage 为 0，loss 为 0。因此该 receipt
只证明单节点 A100 可以执行完整 LoRA-GRPO 机械闭环，不构成“RL 有效”或“性能提升”的证据。进入 `grpo-10`
之前必须增加以下 gate：至少一个 batch 同时含两个不同 reward、非零 advantage、有限且非零 grad norm，并确认
LoRA checkpoint 相对 step 0 至少一个 tensor 发生变化。正式 development 仍只使用 fold 0-3 训练、fold 4 调参；
在冻结前不得读取 current valid/test。

训练与 checkpoint 完成后，vLLM actor teardown 的 CUDA expandable-segment allocator 曾报告一次
`CUDA error: invalid argument`。该异常发生在成功 receipt 之后，不推翻机械可行性；但 `grpo-10` 还必须验证
多步显存不持续增长、Ray actor 全部退出且 driver clean exit，不能把仅有 checkpoint 当作长期运行稳定性证明。

## Evaluation and promotion gate

Base 与 LoRA 必须使用完全相同的 prompt rows、retrieval hash、decode setting 和 parser。报告：

- per task accuracy、macro-F1、Y=0/Y=1 recall；
- prediction distribution 和 JSON/schema success；
- paired macro-F1 bootstrap CI、exact McNemar；
- base-only / LoRA-only correct，以及按 KNN/full-evidence agreement 强度分层的 rescue/harm；
- adapter checkpoint、NeMo commit、base checkpoint、dataset/prompt/reward SHA-256。

“明显变好”的预注册门槛：至少两个 task 的 macro-F1 point estimate 提高 `>=0.03`，其余 task accuracy 不下降
超过 1 percentage point，并且 task-balanced pooled paired bootstrap macro-F1 delta lower 95% bound `>0`。
未通过时完整保留 negative result，不运行 formal test。通过也只获得一次冻结 formal-test 的资格，不自动修改
默认 pipeline。

## Artifact layout

```text
outputs/paper/rl_lora_gpt_oss_120b/
  contract.json
  data/
    <task>/fold_00..04/{prompts.jsonl,manifest.json,audit.json}
    train.jsonl
    development.jsonl
    manifest.json
  runs/<run_id>/{config.yaml,manifest.json,logs/,metrics.json}
  adapters/<run_id>/
  valid/<run_id>/{predictions.jsonl,metrics.json,paired_statistics.json,report.md}
```

Node-local heavy artifacts:

```text
/local/tianang/huggingface/
/local/tianang/nemo-rl/
/local/tianang/txagent_rl_lora/{megatron_cache,checkpoints,ray,logs}/
```
