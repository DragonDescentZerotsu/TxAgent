# Starling assay-level retrieval

更新时间：2026-08-22。

## Conditioned cumulative-family flat assay experiment（current）

新的 conditioned 实验与下面的 4× relevance-prefix 曲线平行，旧入口、排序和 artifacts 均不删除。BBB、
Bioavailability、Skin 读取
`data/processed_starling_context_conditioned_selected_v1/<Task>/scaffold/`；ClinTox 继续读取冻结的
`clinical_trial_failure_v1`，没有合格 condition 时不向 agent 添加 condition 句子。

每个 level 累计加入一个 task 已冻结的 Starling mechanism family；被纳入的每个 physical assay 仍独立检索
Morgan Top-3、Tanimoto `>=0.3`，随后按 molecule 合并为一个 full-flat branch。Reference contract 固定为：

1. 删除 valid+test parent union 在 benchmark-defining direct source/scope 中的 rows；
2. 保留相同 heldout parents 的 non-direct/mechanism rows；
3. query time 对所有候选执行 `scaffold_disjoint`；
4. 同一个 physical assay 若出现在多个 families，只在最早累计 level 引入一次，family/rank 不进入 LLM prompt。

Condition visibility 固定为 `flat_evidence_and_final_only`：external condition 由公共 deterministic renderer
转成自然英语后只进入 flat branch 和 final branch；`no_reported_external_condition` 完全省略，single-molecule
branch 始终保持 condition-independent，因而可跨所有 levels 复用。重叠 physical assay 保持原子性：在最早
level 加入后携带该 assay 的完整 representative records，不按后续 family 再切碎 records。

Family catalog 入口为
`tools/chembl_tool/paper_experiments/build_assay_family_catalog.py`。当前冻结的 physical-assay 累计数量是：

| task | cumulative physical assays by level | multi-family assays |
|---|---|---:|
| BBB | 8,231 / 8,305 / 22,952 / 22,958 | 806 |
| Bioavailability | 4 / 304 / 434 / 1,403 / 1,871 / 2,140 | 208 |
| Skin | 479 / 1,219 | 117 |
| ClinTox | 12 / 13 / 2,356 / 5,665 / 6,002 / 6,064 / 12,207 / 12,485 | 0 |

Bioavailability 的第二级是 `Observed.nondirect_oral_bioavailability`。Normalized-v7 的 HF source 没有
原生 `assay_system`，历史 family catalog 因此曾遗漏该组全部 111,303 条 retrieval-eligible records。当前先由
`tools/chembl_tool/tasks/bioavailability_ma/build_nondirect_assay_context.py` 生成版本化 source overlay：只根据
source-visible report type、粗粒度 population 和 oral exposure mode，把这些 rows 映射到 300 个有解释性的
assay contexts；不按 PMID、molecule 或单条 record 拆分，也不改 endpoint、measurement、condition 或
support text。修复后 family union 覆盖全部 435,478 条 Stage-03 retrieval-eligible records，held-out direct
过滤仍在其后执行。Overlay manifest 和修复后的 raw-v3 index 分别位于：

```text
outputs/paper/starling_conditioned_assay_family_curve_v1/
  source_overlays/bioavailability_nondirect_assay_context_v1/{records.parquet,manifest.json}
  family_catalogs/bioavailability_ma_nondirect_context_v1/
  indices/bioavailability_ma/raw_v3_nondirect_context_v1/
```

修复后的 index 在 conditioned held-out direct filter 后保留 425,833 条 source records、103,274 个
assay×molecule rows 和 27,067 个 molecules；held-out direct、实际 query-neighbor parent overlap 和 scaffold
overlap均为 0。262 条 valid full-catalog prompt 的 `input + 20,480 completion reserve` 最大值为 110,524
tokens，未触发 raw-v3 guard。旧缺失-nondirect lineage 继续使用原 catalog/index/replay，避免正在运行或历史
checkpoint 被新 source hash 污染；修复版 reasoning 独立写入
`scaffold_valid_epyc_deepseek_v4_flash_0731_bio_nondirect_context_v1/`。

### Current raw support cards（无 summary 模型）

正式 conditioned curve 使用 `assay_compact.raw_v3`。每个 assay×molecule 最多保留三张 representative
record cards；每张 card 独立保留 endpoint、value、unit、species、conditions 和完整原始 `support_text`。
只删除重复 contract/source boilerplate、opaque assay id、默认 role/transferability、confidence 和 extraction
diagnostics。raw-v3 不截断或重写任何被选中 card 的字段，也不在 preparation 或运行时调用 GPT-OSS。

PARCC `deepseek-ai/DeepSeek-V4-Flash-0731` 的实测窗口是 1,048,576 tokens，并按
`input tokens + max_tokens` 校验；正式 completion reserve 为 20,480。raw-v3 独立使用 900,000-token payload
gate，给 system/chat envelope 留出超过 128k tokens 的余量。只有 payload 超过安全阈值才按完整 evidence rows
确定性均匀采样；普通 group-level pipeline 保持原 400 KB guard，不受这次切换影响。

在 1,048 个 valid queries、full-catalog、每 assay Top-3、Morgan `>=0.3`、`scaffold_disjoint`，并为每个
neighbor 重复历史 trace 中最大的实际 tool-comparison payload 的保守模拟中，完整 raw support 的
`input + 20,480 completion reserve` 最大值为：BBB 459,798、Bioavailability 163,581、Skin 165,409、
ClinTox 353,728。去除所有 raw-v3 字段上限后，全局最坏仍有 588,778 tokens 余量，且 1,048 个 query
均未触发 evidence sampling。
历史 v2 summary 将全部 long-support corpus token 数减少 41.12%，但旧端到端审计中对完整 prompt 的总 token
只减少约 0.54%，收益不足以承担额外模型依赖和语义改写风险。

ClinTox source-native bridge 仍为
`tools/chembl_tool/tasks/clintox/build_flat_assay_support_evidence.py`。当前 raw-v3 与历史 summary artifacts
分开保存：

```text
outputs/paper/starling_conditioned_assay_family_curve_v1/
  family_catalogs/<task>/{family_assays.jsonl,manifest.json}
  clintox/{assay_support_evidence.jsonl,assay_support_evidence_manifest.json}
  indices/<task>/raw_v3/{assay_molecule_evidence.jsonl,assay_neighbor_index.pkl,manifest.json}
  tokenizer/deepseek_v4_flash_0731/{tokenizer.json,tokenizer_config.json,...}
  prompt_length_audit_raw_v3_untruncated/{prompt_lengths.jsonl,summary.json}
  replay_batches/valid/<task>/assay_flat_top<prefix>/
  scaffold_valid_epyc_deepseek_v4_flash_0731/<task>/
  support_summary_gpt_oss_120b_v1/
    {inventory.jsonl,inventory_manifest.json,summaries.jsonl,summary_manifest.json}
  indices/<task>/compact_v2/  # historical comparison only
```

历史 summary cache/artifacts 不删除，但不再是代码入口、索引构建或 reasoning 的依赖。四任务 replay audit、
conditioned query-prompt 接线及 raw-v3/compact-v2 的 assay ranking、molecule table、assay→molecule indices
和 fingerprint-count parity 已逐 task 通过；切换只改变 LLM-visible evidence view，不改变 retrieval 结果。

### Scaffold-valid 运行快照（2026-08-22）

以下只登记 398/398、262/262、245/245 或 142/142 且 `n_failed_runs=0` 的完整 batch。所有 agent batch 都是
`identity_blind + scaffold_disjoint`；BBB/Bio/Skin 的非 null condition 只在 flat/final 可见，ClinTox 没有
condition。数值为 macro-F1：

| task | none | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBB | 0.612483 | **0.727936** | 0.726247 | 0.721957 | — | — | — |
| Bioavailability | 0.541203 | 0.643302 | **0.678445** | 0.644597 | 0.668787 | 0.668787 | 0.675972 |
| Skin | 0.611851 | 0.623892 | **0.627364** | — | — | — | — |
| ClinTox | 0.506945 | **0.519946** | 0.518328 | 0.506945 | — | — | — |

Bioavailability 和 Skin 已到 full catalog。BBB L4 及 ClinTox L4–L8 没有完整 metrics receipt，不能把当前
task-best 当作最终曲线结论。对应 roots 是
`scaffold_valid_epyc_deepseek_v4_flash_0731/` 和 Bio 修复后的
`scaffold_valid_epyc_deepseek_v4_flash_0731_bio_nondirect_context_v1/`。

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

tools/chembl_tool/paper_experiments/run_conditioned_assay_family_curve.py
  四任务 conditioned cumulative-family launcher。自动验证 selected-v1/ClinTox split、family overlap policy、
  raw-v3 index、held-out hashes 和 direct/scaffold gates；一次物化所有严格嵌套 replays，先运行/恢复 none，
  后续每个 level 复用 none single，并在相邻 level 启动前执行 evidence-equivalent carry-forward。各 task 的同序
  level 共享一个 128-slot global prompt pool，旧 4× relevance-prefix runner 不受影响。

tools/chembl_tool/paper_experiments/audit_conditioned_assay_prompt_lengths.py
  不调用 LLM；用 endpoint-matched tokenizer 对四任务 full-catalog raw-v3 prompt 做保守长度审计，同时核对
  tool-payload、completion reserve 和 evidence-sampling guard。

tools/chembl_tool/paper_experiments/plot_assay_retrieval_curve.py
  唯一 assay-scaling 可视化入口。默认生成 historical 4× curve 的英文五-panel 图；
  `--conditioned-family-baselines` 从完整零失败 artifacts 生成 conditioned levels 与当前 MiniMol/Morgan
  baselines 的 2×2 比较图，并输出 paired one-sided permutation p-value、Holm 校正和遗漏 level 审计。
  两种模式都从 receipts 读取指标，不硬编码曲线数值。
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
python -m tools.chembl_tool.common.assay_retrieval build-preaggregated-index --help
python -m tools.chembl_tool.common.assay_retrieval materialize-batch --help

# raw-v3 超限检查使用冻结的 endpoint-matched tokenizer；不会调用模型
export TXAGENT_DEEPSEEK_TOKENIZER_JSON=outputs/paper/starling_conditioned_assay_family_curve_v1/tokenizer/deepseek_v4_flash_0731/tokenizer.json

# 审计三任务 replay、记录完整计划但不调用模型
python -m tools.chembl_tool.paper_experiments.run_assay_retrieval_curve --manifest-only

# conditioned cumulative-family valid；默认 PARCC DeepSeek、单一 128-slot pool，可断点恢复
python -m tools.chembl_tool.paper_experiments.run_conditioned_assay_family_curve

# Bioavailability HF nondirect context overlay；catalog/index 必须读取该版本化 artifact
python -m tools.chembl_tool.tasks.bioavailability_ma.build_nondirect_assay_context
python -m tools.chembl_tool.paper_experiments.build_assay_family_catalog --tasks bioavailability_ma

# 只重跑受 source 修复影响的 Bioavailability；复用冻结 none/single，并保护旧 output lineage
python -m tools.chembl_tool.paper_experiments.run_conditioned_assay_family_curve \
  --tasks bioavailability_ma \
  --bioavailability-source-revision nondirect_context_v1 \
  --single-analysis-source-batch \
    outputs/paper/starling_conditioned_assay_family_curve_v1/scaffold_valid_epyc_deepseek_v4_flash_0731/bioavailability_ma/none

# 无中断接管已经运行的 conditioned curve；断 tunnel 后选择 Duo 2 并从成功 stage 恢复
python -m tools.chembl_tool.paper_experiments.watch_glm_tunnel_and_matrix \
  --completion-mode recursive_reasoning_runs --expected-results 5578 \
  --output-root outputs/paper/starling_conditioned_assay_family_curve_v1/scaffold_valid_epyc_deepseek_v4_flash_0731 \
  --launcher-module tools.chembl_tool.paper_experiments.run_conditioned_assay_family_curve \
  --launcher-args-json '["--parallelism","128","--preparation-workers","32","--materialize-workers","4","--model","deepseek-ai/DeepSeek-V4-Flash-0731","--base-url","http://127.0.0.1:50001/v1","--api-key-env","DEEPSEEK_API_KEY","--timeout-s","900","--max-stage-requeues","3"]' \
  --allow-implicit-output-root --base-url http://127.0.0.1:50001/v1 \
  --api-key-env DEEPSEEK_API_KEY --ssh-host parcc \
  --ssh-local-forward 127.0.0.1:50001:epyc-1-4:50000 --duo-option 2 \
  --watchdog-log <watchdog.log> --launcher-log <launcher.log>

# 只物化/检查 conditioned replays、condition prompt 和 batch manifests，不调用 LLM
python -m tools.chembl_tool.paper_experiments.run_conditioned_assay_family_curve --prepare-only

# 绘制当前完整 conditioned levels、latest baselines 和 exploratory paired p-values
python -m tools.chembl_tool.paper_experiments.plot_assay_retrieval_curve \
  --conditioned-family-baselines

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
