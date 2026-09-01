# Starling train-only LoRA-RL

本目录隔离维护 Starling train-only LoRA-RL 实验。当前 one-pass Bio 对照包括 hosted Tinker
GPT-OSS-120B 与本地 NeMo RL GPT-OSS-20B；两者只允许 backend/runtime 层不同，必须共享数据合同、
reward 实现和冻结的公共训练设置。它不是新的 benchmark lineage，也不得修改现有 task pipeline、
prompt profile、router OOF 或正式 valid/test artifact。
历史 E17 final-only 合同保留不变；2026-08-10 用户明确批准的 E18
`one_pass_full_flat.v1` 使用独立代码和 artifact namespace，详细合同见
`ONE_PASS_IMPLEMENTATION_PLAN.md`。

## 不可破坏的边界

1. RL prompt 只来自 current scaffold `train`。任何训练、checkpoint 选择、reward 调整和超参数选择都不得读取
   scaffold valid/test label。
2. train prompt 必须是 OOF 且使用 `mixed_family_identity_policy.v1`：direct family 对 query 执行
   `scaffold_disjoint`，其它 full-flat mechanism family 只执行 `parent_disjoint`，允许同 scaffold 的机制 analog。
   不得对整个 full-flat union 全局套用 scaffold filter，也不得直接用未过滤的 full-train direct index。
3. Phase 1 三个 task 都使用 current `full_flat`，并且只训练 final synthesis。single/group 输出由冻结的 base GPT-OSS-120B 生成并作为 prompt context；
   LoRA adapter 不用于重写这些 branch。这样 held-out 结果只归因于 final decision policy。
4. current task lineage 固定为：
   - BBB_Martins: `experimental_meaningful_cns_access_v2`
   - Bioavailability_Ma: `record_supported_v2`
   - Skin_Reaction: `record_supported_v2`
5. current prompt profile 固定为：BBB `meaningful_cns_access_v1`、Bioavailability
   `f20_evidence_calibrated_v2`、Skin `sensitization_aligned_v2`。
6. 正式训练数据不得复用 historical BBB 18,425-parent router OOF artifact；它只允许用于硬件/解析 smoke。
   Bioavailability/Skin 既有 OOF artifact 也只有在 prompt/retrieval/profile hash 完全匹配时才可复用。
7. reward 只读取 train gold label、结构化输出合法性和确定性长度/完整性约束。不得把 valid/test metric、
   baseline prediction 或人工 case selection 写入 reward。
8. valid 只在 train-only 配置冻结后运行一次 paired evaluation；test 默认不运行。是否运行 test 必须经过单独
   promotion gate，并保持 adapter、decode、prompt、retrieval 和 threshold 全部冻结。
9. 所有数据、checkpoint、metrics 和 manifest 写入 model-specific 独立 namespace：
   120B 使用 `outputs/paper/rl_lora_gpt_oss_120b/`；2026-08-11 用户批准的本地 NeMo RL 20B 对照使用
   `outputs/paper/rl_lora_gpt_oss_20b/`。大 checkpoint/cache 放 node-local `/local/tianang/`，共享目录只保留
   adapter、manifest、汇总和必要审计。
10. 不保存 API key、sudo 密码或 Hugging Face token。远程启动命令和 manifest 只能记录环境变量名。

## 双后端共享规则

1. `experiment_contract.py` 是 one-pass reward version、reward weights 和跨后端训练设置的唯一来源；
   adapter 和 YAML 不得另存一套“等价”常量。
2. Tinker 与 NeMo 都必须通过 `training_contract.py` 的
   `reward_metadata_from_row()` 和 `score_training_response()` 计分。adapter 只负责 response transport、
   worker 调度和 backend metrics，不得实现 label parser 或 reward arithmetic。
3. 当前正式 NeMo 20B config 及其 smoke/2-step 派生 config 必须通过
   `validate_backend_contract.py`；Tinker one-pass CLI 的公共字段必须通过 launcher 内同一 profile gate。
4. 模型 context、并行拓扑、optimizer implementation、constant-reward group 处理和 checkpoint format
   属于 backend-specific manifest 字段，不能伪装成共享合同。若要把其中一项变成 matched ablation，必须
   新建显式 profile/version，不能改写已经运行中的合同。
5. 修改 reward 必须新建 reward version、同时更新双后端一致性测试，并重新审计训练曲线；不得因某个
   backend 当前 reward 不上涨而只修改该 backend 的 reward。

以上第 1-8 条描述历史 E17 final-only 线，不得借 E18 覆盖其 artifact。E18 blind 的显式差异是：不做 OOF，
对全部 scaffold train 逐 query 使用与 canonical valid 相同的 global `parent_disjoint` policy；valid 用于 checkpoint 选择；设置冻结后 test
只运行一次。其余 lineage、label privacy 和独立 namespace 边界继续适用。

2026-08-10 用户另行批准 `one_pass_full_flat_visible_prefetched.v1` 作为上限控制。它不得覆盖 blind
  artifact，固定写入 `one_pass_full_flat_visible_prefetched_v1/`：query/neighbor structure 与允许的 source ID
可见，但 retrieval、`molecule_properties`、`mmp_structure_compare`、`properties_compare` 仍由 harness 在
generation 前固定执行；LLM 没有 live tool surface，gold 仍只在 reward metadata。Skin 必须使用 current
sensitization/contact-allergy scoped v2 index。三任务完整 base-valid、128K token audit、训练曲线、W&B 和
bounded RL smoke 全部通过。Hosted 120B 正式任务后来由用户明确停止，不得自动恢复；本地 NeMo
GPT-OSS-20B Bio run 也在 step-174 metrics 后由用户明确停止，最后完整 checkpoint 为 step 160。两个
backend 均不得自动恢复；新训练必须由用户另行明确批准。

## E18 one-pass primary contract

- prompt contract: `one_pass_full_flat.v1`
- one model call emits `single_prediction`, `analog_prediction`, task final prediction, and the corresponding analyses
- model-visible input reuses current bounded query properties, anonymous full-flat evidence, and prefetched pairwise tool text
- no live LLM tool call; retrieval and tools remain deterministic harness preparation
- train: all current scaffold train rows with the same global `parent_disjoint` Full Flat retrieval used by the canonical three-stage valid condition
- checkpoint selection: current scaffold valid; test remains untouched until the selected checkpoint and decode are frozen
- reward: final correctness `+1/-1`; independently correct single and analog predictions `+0.15` each; a correctly resolved single/analog conflict receives `+0.15`; JSON/schema bonuses total at most `0.05`
- no semantic hallucination reward, no generic consistency reward, and no reward for merely citing a neighbor identifier
- strict reward ordering: every final-correct trajectory scores above every final-wrong trajectory; both-correct/final-correct ties one-correct/one-wrong/final-correct when the latter correctly resolves the conflict
- artifacts: `outputs/paper/rl_lora_gpt_oss_120b/one_pass_full_flat_v1/`
- shared Bio training profile: rank-32 all-module LoRA (MLP + attention +
  unembedding), `group_size=8`, `groups_per_batch=4`, and `max_tokens=6144`;
  hosted 120B uses `openai/gpt-oss-120b:peft:131072` when explicitly run, while
  the frozen local comparison candidate is GPT-OSS-20B on NeMo. The shorter 4,096-token
  BBB smoke cap is not valid for Bio.
- every one-pass training file requires an adjacent passing audit whose data
  hash matches exactly; both `run_tinker.py` and `run_grpo.py` require a fresh
  v2 audit before creating backend workers/clients
- explicit failed tool receipts remain `status=error` but backend exception
  messages and structure fragments must be replaced by the frozen generic
  missing-evidence text before prompting

Visible-prefetched 上限控制复用同一输出 schema 和 reward ordering；只改变显式 versioned visibility
contract，不得把 visible prompt 混进 blind 数据文件或 checkpoint selection。当前 128K 模型 ID 为
`openai/gpt-oss-120b:peft:131072`；32K ID 不满足完整 Bio/Skin prompt 加 6,144 输出 token 的 gate。
2026-08-10 已通过的 rank-16 attention-only 三步 smoke 只作历史 observability receipt；用户随后把正式候选
改为 Tinker SDK 默认组件集合和默认 rank，即 rank-32 all-module。新的 matched 三步 hosted smoke 已完成：
96 条 rollout 最大 4,911 tokens，0 条触及 6,144 上限，96 clean stop、95 parseable JSON、92 full schema；
三个 optimizer step、W&B、step-3/final checkpoint 和 final sampler valid-3 回读全部通过。该 gate 允许请求
正式 Bio 训练确认，但 smoke 本身不是正式训练授权，且不得读取 test。

## Historical E17 Phase 1 primary contract

- algorithm: synchronous GRPO
- model: `openai/gpt-oss-120b`
- backend: NeMo RL Megatron Core
- trainable parameters: attention-only LoRA (`linear_qkv`, `linear_proj`)，首个候选 `rank=16, alpha=32`
- train topology: node001 single node, 8 x A100-SXM4-80GB, `EP=8, TP=1, PP=1`
- generation topology: colocated vLLM, `TP=8`
- evidence condition: all tasks `full_flat`; direct `scaffold_disjoint`, non-direct `parent_disjoint`
- sampling: one task-specific adapter per run, label-balanced prompts, 8 generations per prompt
- primary reward: exact task label correctness；JSON/schema completeness 为小幅 shaping，不能超过 label reward
- primary evaluation: per-task accuracy、macro-F1、class recall、schema success、paired bootstrap 和 McNemar

先后 gate 为：数据/identity audit -> config parse -> model conversion -> LoRA forward/backward -> vLLM refit/logprob
parity -> 1-step GRPO -> 10-step memory plateau -> train-only development run -> frozen scaffold-valid evaluation。
