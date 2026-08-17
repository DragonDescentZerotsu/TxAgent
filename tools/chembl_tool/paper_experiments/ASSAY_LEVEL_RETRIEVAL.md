# Starling assay-level retrieval

更新时间：2026-08-17。

## 目的与边界

Assay-level retrieval 与现有 group-level retrieval 是两个平行的 evidence assembly 版本。两者共享
分子标准化、Morgan 相似度、parent-disjoint policy、minimal-evidence contract、reasoning batch、tool
prefetch、single analysis 和 final reasoning；但 assay 排名、prefix schedule、flat merge 和 compact prompt
view 都封装在 assay-level 模块内，不修改 group-level 的 family mapping 或 branch 数量。

Assay unit 定义为 `canonical_assay_context`。同一 context 下的 endpoint 只作为 assay 描述，不再拆成多个
assay；缺少 context 时才使用 `endpoint_fallback::<canonical_endpoint_name>`。Catalog 不做额外 quality gate，
仅要求至少 5 个 unique molecules。Codex-authored v6 rubric 离线为每个 catalog row 打 0–100 biological
relevance score；score/rank 只用于选择累计 prefix，不进入 LLM-visible evidence。

每个被选 assay 独立检索 Morgan top-3、Tanimoto `>=0.3` 的 train-reference molecules。随后把相同 reference
molecule 在不同 assays 下的 evidence rows 合并到一个 `Flat.assay_ranked_evidence` branch。Valid query 只能从
train molecules 检索，并继续执行 `identity_blind + parent_disjoint`。

## 维护入口

```text
tools/chembl_tool/common/starling/assay_catalog.py
  构建 context-level assay catalog；属于 Starling 数据合同，不依赖 paper experiment。

tools/chembl_tool/paper_experiments/codex_assay_relevance.py
  冻结的离线 v6 relevance rubric；无外部模型调用。

tools/chembl_tool/common/assay_retrieval.py
  构建 train-only assay index、累计 prefix retrieval、molecule merge 和 replay batch。

tools/chembl_tool/common/evidence_contract.py
  `evidence_for_group_llm()` 保持普通 group 的完整 minimal-evidence view，仅对 assay flat group
  使用 bounded assay view。旧 v1 replay 通过固定 group id 兼容。

tools/chembl_tool/paper_experiments/run_assay_retrieval_curve.py
  三任务 launcher/audit。默认 prefix 为 5 起步、每次乘 4，并强制最后一点等于该任务全部 assays。

tools/chembl_tool/paper_experiments/plot_assay_retrieval_curve.py
  唯一 assay-scaling 汇总和英文四联图入口；从 artifacts 读取指标，不硬编码曲线数值。
```

Task pipeline 不新增 assay-specific runner。BBB、Bioavailability 和 Skin 仍使用原有
`run_reasoning_pipeline.py`；reasoning batch 通过 frozen retrieval replay 注入一个 flat group。这样 assay 和
group 两条实验线共享运行时，但 source selection 与 prompt compression 不会相互改变。

## Scaffold-valid 实验（PARCC DeepSeek-V4-Flash-0731）

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
