# Starling assay-level retrieval

更新时间：2026-08-18。

## 目的与边界

Assay-level retrieval 与现有 group-level retrieval 是两个平行的 evidence assembly 版本。两者共享
分子标准化、Morgan 相似度、identity policy、minimal-evidence contract、reasoning batch、tool
prefetch、single analysis 和 final reasoning；但 assay 排名、prefix schedule、flat merge 和 compact prompt
view 都封装在 assay-level 模块内，不修改 group-level 的 family mapping 或 branch 数量。

Assay unit 定义为 `canonical_assay_context`。同一 context 下的 endpoint 只作为 assay 描述，不再拆成多个
assay；缺少 context 时才使用 `endpoint_fallback::<canonical_endpoint_name>`。Catalog 不做额外 quality gate，
仅要求至少 5 个 unique molecules。Codex-authored v6 rubric 离线为每个 catalog row 打 0–100 biological
relevance score；score/rank 只用于选择累计 prefix，不进入 LLM-visible evidence。

每个被选 assay 独立检索 Morgan top-3、Tanimoto `>=0.3` 的 reference molecules。随后把相同 reference
molecule 在不同 assays 下的 evidence rows 合并到一个 `Flat.assay_ranked_evidence` branch，并继续执行
`identity_blind`。Current/default v5 在 query time 使用 `scaffold_disjoint`：相同 parent 和相同非空
Bemis–Murcko scaffold 的候选都排除，并在相似度门槛内向后补足 Top-3。

Reference pool 是显式、版本化的实验维度：

- historical v1 `train_only` 只索引 train parents；
- historical v2 `direct_only_heldout_filtered` 从 normalized-v7 Stage 03 的全部
  `retrieval_eligible=true` records 开始，只删除 current valid+test parent union 在 benchmark-defining direct
  source/scope 中的记录；相同 heldout parent 的其他 source records 保留。BBB 使用当前
  `experimental_meaningful_cns_access_v2` heldout union，Bioavailability/Skin 使用当前
  `record_supported_v2` heldout union；
- current/default v5 `direct_only_heldout_filtered_scaffold_disjoint` 复用 v2 source index：先删除全部 current
  valid+test parents 的 benchmark-defining direct-outcome rows，保留它们的 mechanism/non-direct rows；再对每个
  query 动态排除 same parent 和 same non-empty scaffold。

## 维护入口

```text
tools/chembl_tool/common/starling/assay_catalog.py
  构建 context-level assay catalog；属于 Starling 数据合同，不依赖 paper experiment。

tools/chembl_tool/paper_experiments/codex_assay_relevance.py
  冻结的离线 v6 relevance rubric；无外部模型调用。

tools/chembl_tool/common/assay_retrieval.py
  构建显式 reference-pool assay index、累计 prefix retrieval、molecule merge 和 replay batch。
  v2/v5 直接读取 Stage 03 eligible rows，避免继承旧 Stage 06/07 heldout lineage；通用 retrieval policy
  已原生支持 `scaffold_disjoint` 和 threshold-preserving backfill。

tools/chembl_tool/common/evidence_contract.py
  `evidence_for_group_llm()` 保持普通 group 的完整 minimal-evidence view，仅对 assay flat group
  使用 bounded assay view。旧 v1 replay 通过固定 group id 兼容。

tools/chembl_tool/paper_experiments/run_assay_retrieval_curve.py
  三任务 launcher/audit。默认 setting 为 `direct_only_heldout_filtered_scaffold_disjoint`；仍可显式运行 historical
  pools。Prefix 为 5 起步、每次乘 4，并强制最后一点等于该任务全部 assays。Audit 同时验证 held-out direct
  rows 为零、实际 query-neighbor scaffold overlap 为零和 prefix 嵌套。

tools/chembl_tool/paper_experiments/plot_assay_retrieval_curve.py
  唯一 assay-scaling 汇总和英文五-panel图入口；A–D 汇总 performance/group comparison/retrieval volume，
  E 展示累计 mean assay-relevance score；从 artifacts 读取指标，不硬编码曲线数值。
```

Task pipeline 不新增 assay-specific runner。BBB、Bioavailability 和 Skin 仍使用原有
`run_reasoning_pipeline.py`；reasoning batch 通过 frozen retrieval replay 注入一个 flat group。这样 assay 和
group 两条实验线共享运行时，但 source selection 与 prompt compression 不会相互改变。

## Historical train-only scaffold-valid 实验（PARCC DeepSeek-V4-Flash-0731）

三任务均已完成四倍递增 schedule 及全部-assay端点，共 24 个 performance rows（包含 3 个 historical none）和
24 个 retrieval-volume rows。新增 1,797 个 sample-condition 全部成功，0 failed。

| task | eligible assays | full-catalog macro-F1 | best prefix | best macro-F1 | historical group best | delta |
|---|---:|---:|---:|---:|---:|---:|
| BBB | 22,820 | 0.716634 | 20,480 | **0.720856** | full-flat 0.676241 | +0.044615 |
| Bioavailability | 1,840 | 0.579121 | 20 | **0.640151** | full-flat 0.677652 | -0.037501 |
| Skin | 12,015 | 0.613003 | 80 | **0.626714** | direct 0.616345 | +0.010369 |

三个 task-best assay points 的未加权平均 macro-F1 为 `0.662574`；historical group-best 平均为 `0.656746`，
描述性 delta 为 `+0.005828`。全量 assay 不是三个任务中任何一个的最佳点，说明加入更 distant assays 会产生
task-dependent noise。

`none` 和 historical group-level reference 复用既有 OpenRouter `deepseek/deepseek-v4-flash` artifacts；新
assay-level points 使用 PARCC `deepseek-ai/DeepSeek-V4-Flash-0731`。因此 group comparison 是 descriptive
historical reference，不是 endpoint-matched rerun。

```text
outputs/paper/starling_assay_relevance_all_v1/
outputs/paper/starling_assay_retrieval_v1/
outputs/paper/starling_assay_retrieval_curve_v1/
  scaffold_valid_train_only_epyc_deepseek_v4_flash_0731/
    experiment_manifest.json
    audit.json
    analysis/{summary.json,assay_curve_metrics.tsv,assay_retrieval_volume.tsv,assay_vs_group_best.tsv}
    analysis/figures/assay_retrieval_scaling.{svg,png}
```

`outputs/` 不进入 Git；本文件保存稳定合同、入口和最终数值，artifact manifest 保存完整路径、模型、并发、
prefix、identity policy 和 replay audit。

## Historical direct-only-heldout-filtered scaffold-valid 实验

三任务 historical v2 index、valid replay、全量 parent/direct-overlap audit 和 21-condition worst-case smoke
均已完成，smoke 为 0 failed。该 reference pool 不是“允许 query 自己从 non-direct source 回流”：index 构建只
删除 benchmark-defining direct rows，而 query-time 对所有 source 继续执行 `parent_disjoint`。

| task | heldout parents | excluded heldout direct records | retained heldout non-direct records | heldout direct records after filter |
|---|---:|---:|---:|---:|
| BBB | 732 | 70,077 | 49,133 | 0 |
| Bioavailability | 418 | 9,470 | 70,817 | 0 |
| Skin | 490 | 7,479 | 54,369 | 0 |

这些计数来自三个 v2 index manifest；replay audit 还逐 query 重算 normalized molecular parent，并验证实际
neighbor parent 与 query parent 的交集为零。正式 artifact roots：

```text
outputs/paper/starling_assay_retrieval_v2/
outputs/paper/starling_assay_retrieval_curve_v2/
  scaffold_valid_direct_only_heldout_filtered_epyc_deepseek_v4_flash_0731/
```

Full valid 运行模型为 PARCC `deepseek-ai/DeepSeek-V4-Flash-0731`，endpoint
`http://127.0.0.1:50001/v1`，global prompt pool 默认 64 slots，经 checkpoint 恢复可提升到已允许的
128-slot 上限，900 秒 transport timeout。BBB 和 Bioavailability 的全部保留档位均已完整、0 failed；Skin
已完成至 Top-5,120，full-catalog 在 64/245 final 时按用户要求停止并保留 checkpoint。已完成曲线的当前结果：

| task | 5 | 20 | 80 | 320 | 1,280 | 5,120 | all |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBB | 0.646214 | 0.641403 | 0.648371 | 0.680329 | 0.724722 | **0.763566** | 0.759553 |
| Bioavailability | 0.675789 | 0.702195 | 0.703897 | 0.740909 | — | — | **0.787754** |
| Skin | 0.612634 | 0.601418 | 0.589724 | **0.633422** | 0.630177 | 0.602137 | incomplete |

## Discarded pre-LLM source-contract probes

两个短暂的 source-contract probes 在启动 LLM 前即被否决：一个从所有 source 全局删除 valid+test parents，
过度限制 mechanism analog；另一个保留 full source，未删除 held-out direct-outcome rows。两者从未产生
reasoning/performance 结果，相关临时代码分支和大型 index/replay artifacts 已清理，不作为可运行或可引用的
lineage。正式历史只保留 v1/v2，当前默认为 v5。

## Current/default direct-filtered scaffold-disjoint lineage

v5 的 source index 与 historical v2 完全相同，先删除 valid+test parent union 的 benchmark-defining direct rows，
保留相同 held-out parents 的 mechanism/non-direct records。Query-time policy 从 v2 的 `parent_disjoint` 改为
`scaffold_disjoint`。因此不同 scaffold 的 valid/test molecules 只可能通过保留下来的 non-direct evidence 作为
mechanism analog；它们的 direct-outcome rows 不在 index 中。独立 roots：

```text
index: outputs/paper/starling_assay_retrieval_v2/<task>/
         scaffold_valid_direct_only_heldout_filtered_all/assay_neighbor_index.pkl
replay: outputs/paper/starling_assay_retrieval_v5/<task>/
          scaffold_valid_direct_only_heldout_filtered_scaffold_disjoint/replay_batches/
curve: outputs/paper/starling_assay_retrieval_curve_v5/
         scaffold_valid_direct_only_heldout_filtered_scaffold_disjoint_epyc_deepseek_v4_flash_0731/
```

相对 historical v2（相同 index、query-time `parent_disjoint`），v5 exact neighbor-set 发生变化的 valid query
比例为：

| task | Top-5 | Top-20 | Top-80 | Top-320 | Top-1,280 | Top-5,120 | all |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBB | 0.27% | 0.27% | 3.28% | 12.57% | 20.49% | 41.80% | 45.08% |
| Bioavailability | 42.11% | 43.06% | 44.02% | 53.59% | — | — | 55.02% |
| Skin | 14.69% | 19.18% | 26.12% | 31.43% | 35.92% | 43.27% | 47.35% |

Full-catalog query-neighbor pairs 从 BBB `9,038` 变为 `8,471`（删除 718、backfill 新增 151，保留旧 pair
92.06%），Bioavailability 从 `2,797` 变为 `2,513`（删除 451、新增 167，保留 83.88%），Skin 从
`6,018` 变为 `5,607`（删除 500、新增 89，保留 91.69%）。三任务 audit 均确认 held-out direct rows after
filter 为零、actual query-neighbor parent/scaffold overlap 为零，并且 prefix assays/evidence 严格嵌套。
尚未启动 v5 LLM reasoning。

## 共享省调用量与恢复策略

以下优化互相独立，并且不改变 retrieval rows、prompt profile、model 或 prediction policy：

1. **删除过密档位。** 默认几何档位始终保留 full catalog；若 full catalog 不到前一个 4× point 的 2 倍，
   删除倒数第二点。当前因此不跑 BBB Top-20,480 和 Bioavailability Top-1,280。
2. **跳过完整 batch。** 默认只提交缺少 metrics、样本数不完整或 `n_failed_runs>0` 的 condition；
   `--include-complete-batches` 仅用于显式诊断。
3. **single 只计算一次。** Single-molecule 分支不依赖 assay prefix。Runner 在同一 task/output lineage 中寻找
   最早的完整兼容 batch，并通过 `--single-analysis-source-batch` 供后续普通运行和 `--exact-prefixes`
   恢复使用。Prompt-profile 兼容性继续由公共 batch contract 强制校验；不存在兼容 source 时才调用模型。
4. **相邻 prefix evidence-equivalent carry-forward。** 逐 query 对 query identity、有序 neighbors 和
   `evidence_for_group_llm()` 后的 compact rows 做稳定 hash；完全相同时复制已成功的 single/group/final/trace，
   并在 target-local `reuse.json`、run manifest 和 batch-level `carry_forward_receipt.json` 保存来源。
   该策略有意忽略 selected-assay/coverage-count-only 变化，因此是冻结的 evidence-equivalence policy，
   **不是**整个 final prompt 的 byte-identical reuse。任何 group-visible molecule、顺序、similarity 或 evidence
   row 变化都必须重跑 group/final；target 已有与 source 不同的 reasoning artifact 时也拒绝覆盖。
5. **stage checkpoint 恢复。** Global prompt pool 在 single/group/final 成功后原子落盘；重启只调度缺失或失败
   stage。局部修复使用 `--exact-prefixes`，不会自动追加 full-catalog condition。

Historical v2 的 19-condition valid 计划中，carry-forward census 为 1,323 个 final，其中 333 个原本还会包含
non-empty flat group call；结合删除两个过密档位，理论调用数从 11,174 降到约 8,379。BBB v2 receipts 实际
登记了 858 个 carry-forward final，Top-22,820 另预填 24 个。v5 尚未启动 reasoning，不能直接沿用这些 v2
group/final receipts；single 仍按兼容合同复用，其他复用必须重新通过 v5 evidence hash。

## 常用入口

```bash
# index/replay builder 的完整参数合同
python -m tools.chembl_tool.common.assay_retrieval build-index --help
python -m tools.chembl_tool.common.assay_retrieval materialize-batch --help

# 审计三任务 replay、记录完整计划但不调用模型
python -m tools.chembl_tool.paper_experiments.run_assay_retrieval_curve --manifest-only

# 显式复现 historical v2 parent-disjoint 合同
python -m tools.chembl_tool.paper_experiments.run_assay_retrieval_curve \
  --retrieval-contract direct_only_heldout_filtered --manifest-only

# 只恢复一个指定档位；不追加其它档位
python -m tools.chembl_tool.paper_experiments.run_assay_retrieval_curve \
  --tasks bbb_martins --prefixes 22820 --exact-prefixes --parallelism 128

# 对已经 preparation 的相邻 batch 物化 evidence-equivalent reuse
python -m tools.chembl_tool.paper_experiments.carry_forward_assay_prefix \
  --source-batch <previous-prefix-batch> --target-batch <next-prefix-batch>

# 生成 TSV、英文五-panel scaling figure 和独立 relevance-decay 图
python -m tools.chembl_tool.paper_experiments.plot_assay_retrieval_curve \
  --output-root <completed-curve-root>
python -m tools.chembl_tool.paper_experiments.plot_assay_retrieval_curve \
  --output-root <curve-root> --relevance-decay-only
```

主绘图入口默认拒绝缺失、失败或样本数不完整的 condition；只有明确制作运行中诊断图时才加
`--allow-incomplete`，该图不得进入正式结果。

`run_assay_retrieval_curve.py` 的 root manifest 是跨兼容 resume launches 的累计视图；不同 retrieval contract、
model、visibility、identity policy 或 index hash 不能写入同一 root。每次并发与启动时间仍保存在
`scheduler_launch_history`，避免一次 `--exact-prefixes` 恢复把完整三任务计划覆盖掉。
