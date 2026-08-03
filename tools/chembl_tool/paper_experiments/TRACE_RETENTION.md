# 论文 Reasoning Trace 保留策略

本文档定义最终论文实验唯一允许长期保留的 reasoning trace。目标是让产物目录与论文方法一一对应，避免旧 Tier、细粒度 group、expert policy 和失败重跑继续占用空间或干扰分析。

## 唯一正式目录

2026-08-01 起，新 `record_agreement70_split811_v1` lineage 的正式 trace 只保留在：

```text
outputs/paper/molecular_evidence_agent_starling_random_record_agreement70_split811_v1/
  runs_identity_blind_parent_disjoint/
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1/
  runs_identity_blind_parent_disjoint/
```

这两个 root 是新 v4 主矩阵；不依赖 operational 或 `reuse.json`。下述四个
`outputs/paper/molecular_evidence_agent/` roots 是旧 TDC/strict-conflict lineage 的 historical allowlist，
继续保留但不得与 v4 混表或作为新默认。

历史论文 trace 保留在：

```text
outputs/paper/molecular_evidence_agent/
  runs/
  runs_deployment_visible_prefetched/
  runs_deployment_visible/
  runs_deployment_visible_parent_disjoint/
```

`runs_deployment_visible/` 对应 agentic 主实验候选，`runs/` 对应 `identity_blind` 补充控制，
`runs_deployment_visible_prefetched/` 对应 matched-prefetch 补充控制。正式 condition 集合以
`analysis/experiment_summary.tsv` 为准。`runs_deployment_visible_parent_disjoint/` 对应 analog-only
消融，其正式 condition 集合以 `analysis/parent_disjoint_ablation/condition_results.tsv` 为准。
不在这两个 allowlist 中的 condition 目录不是论文结果。

独立的 ChEMBL activity-transfer benchmark 不属于这次 task pipeline 清理范围，其输出和 trace 继续保留。

## 保留单位

每个成功样本只保留一个 run 目录，其中包括：

```text
trace_messages.jsonl
retrieval.json
single_molecule_reasoning_output.json
group_reasoning_outputs.jsonl
final_reasoning_output.json
```

condition 级继续保留 `predictions.jsonl`、`metrics.json`、`manifest.json`、报告和日志。`predictions.jsonl` 中每一行必须指向一个实际存在的 run 目录，且该目录必须包含自己的 `trace_messages.jsonl`。

不再生成或保留 condition-level 合并 `trace_messages.jsonl`。它只是样本级 trace 的重复副本，会显著增加空间占用。Paper runner 必须传入 `--no-combine-traces`。

## 必须删除的产物

- `outputs/chembl_tool/tasks/<task>/reasoning/` 下旧通用 pipeline、细粒度 Tier/group 和 expert-policy run。
- 不在最终实验矩阵中的 condition，包括 `_failed_attempts`、临时补跑和命名为 `invalid_*` 的目录。
- 未被 condition 的 `predictions.jsonl` 引用的孤立 run 目录。
- condition 根目录下由样本 trace 拼接得到的重复 `trace_messages.jsonl`。
- 旧 task reasoning 目录中的 `.trace_viewer.html` 链接，以及只服务旧 task pipeline 的 viewer 适配。

失败样本完成重跑并进入正式 `predictions.jsonl` 后，应立即删除失败尝试，不把调试历史混入论文产物。需要长期保存的错误分析应转成小型表格、报告或人工 annotation，而不是复制整批 trace。

## 2026-07-12 清理记录

本次清理前盘点：

```text
旧 task reasoning roots: 5
旧 task batch directories: 119
旧 task trace files: 7,242
旧 task files: 57,678
旧 task size: 11,908,137,244 bytes

最终矩阵 conditions: 42
最终预测样本: 8,912
最终应保留 per-run traces: 8,912
重复 condition-level traces: 42
重复 condition-level trace size: 1,217,144,932 bytes
非最终/失败 condition directories: 61
孤立 invalid run directories: 9
旧 deployment-visible smoke trace files: 4
预计释放空间: 13,180,239,737 bytes（约 12.3 GiB）
```

清理后必须满足：

```text
identity-blind 26 个、matched-prefetch 21 个、agentic 26 个 condition
15,428 行 predictions
15,428 个被 predictions 引用的 run 目录
15,428 个 per-run trace_messages.jsonl
0 个 condition-level 合并 trace
0 个 invalid/failed condition 或孤立 run
```

## 2026-07-14 Parent-disjoint 纳入与二次清理

Parent-disjoint 是论文 analog-retrieval claim 的正式消融，因此纳入长期 trace 和 viewer。当前冻结产物为：

```text
identity-blind: 26 conditions / 5,486 traces
deployment-visible matched-prefetch: 21 conditions / 4,456 traces
deployment-visible agentic: 26 conditions / 5,486 traces
deployment-visible parent-disjoint: 22 retrieval conditions / 4,598 traces
total: 95 conditions / 20,026 predictions / 20,026 run directories / 20,026 traces
missing referenced runs: 0
orphan run directories: 0
```

Parent-disjoint 的 4,598 个 sample manifest 全部记录 `neighbor_identity_policy=parent_disjoint`；其中
494 个 sample 因 retrieval 变化而重跑，4,104 个 sample 通过相同 LLM-visible input hash 复用，并在
`reuse.json` 中保存 provenance。Viewer 必须明确展示二者，不能把复用 trace 误解成未执行 policy。

本次删除不在上述 allowlist 中的 `repair_strict_pair_20260713_093731/` 归档（其正式修复结果已写回
对应 condition）以及 activity-transfer benchmark 中两个明确命名为 `smoke1` 的调试 run。独立
activity-transfer benchmark 的正式 runs 继续保留；task assay/evidence library 中不含 reasoning trace
的 smoke 预处理产物不属于本次 trace 清理范围。

## Viewer 约束

`tools/trace_viewer/viewer.html` 对新 v4 扫描 `runs_identity_blind_parent_disjoint/`，对历史 dataset 扫描上述
四个 legacy paper roots。它通过 `predictions.jsonl` 构建样本目录，再按需加载样本级 trace、retrieval、
sample manifest 和可选 reuse provenance；渲染依据统一 stage、message、tool call 和 JSON 数据结构，
不硬编码 task prediction 字段、旧 Tier 或 expert-policy schema。

启动方式：

```bash
bash tools/trace_viewer/start_viewer.sh 8776
```
