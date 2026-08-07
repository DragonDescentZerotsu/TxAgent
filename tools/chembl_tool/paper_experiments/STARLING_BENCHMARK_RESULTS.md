# Starling benchmark：当前决策、结果与入口

更新时间：2026-08-07。

本文件是 Starling benchmark 迁移和实验的集中总账。当前 paper-facing lineage 是 scaffold-only
`record_supported_v2`；第一版 `record_agreement70_split811_v1` random/scaffold 结果作为 historical
comparison 保留。旧 TDC `test` / `valid` 的历史结果仍见 `RESULTS.md`，不得跨 lineage 混表。

## 0. 当前 record-supported v2 scaffold benchmark

v2 继续使用第一版相同的 70% record-weighted parent label，只改变 scaffold split 分配：优先保证
scaffold-disjoint，并按 lexicographic MILP 依次最小化 held-out singleton、valid/test singleton imbalance、
label imbalance，最后才最大化第一版 valid molecule 复用。BBB valid/test 上限各 500；Bioavailability
和 Skin 各取总 parent 的 10%。

| task | train / valid / test | valid multi/single | test multi/single | valid/test Y=0,Y=1 | first-version valid reuse |
|---|---:|---:|---:|---:|---:|
| BBB_Martins | 18,425 / 500 / 500 | 500 / 0 | 500 / 0 | 139,361 / 139,361 | 81 |
| Bioavailability_Ma | 1,674 / 209 / 209 | 209 / 0 | 209 / 0 | 58,151 / 58,151 | 110 |
| Skin_Reaction | 1,966 / 245 / 245 | 240 / 5 | 240 / 5 | 73,172 / 73,172 | 72 |

所有 identity/scaffold pairwise overlap 都是 0。Skin 的 10 个 held-out singleton 是精确 245/245 下的
全局最小值，并均衡分配为 valid/test 各 5 个。当前入口与 canonical roots：

```text
tools/chembl_tool/common/starling/build_record_supported_benchmark.py
data/processed_starling_record_supported_v2/
tools/chembl_tool/paper_experiments/build_starling_benchmark_indices.py
outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2/
tools/chembl_tool/paper_experiments/seed_starling_matrix_reuse.py
tools/chembl_tool/paper_experiments/starling_benchmark_matrix.py
tools/chembl_tool/paper_experiments/analyze_starling_direct_significance.py
```

五个 v2 Starling held-out indices 均通过 `zero_parent_overlap=true` 和
`n_residual_heldout_parent_identities=0`。正式 valid matrix 使用完整 22-condition scaffold matrix、
`parent_disjoint` retrieval，分别运行 GPT-OSS-120B 与 GLM-5.2 的 identity-blind 和 deployment-visible。
历史复用按 molecule key 和严格 stage contract 审计；不会按旧 query index 搬运。

GPT-OSS-120B 的 blind/visible 两套矩阵均已完成 22 conditions、6,887/6,887 sample-conditions，失败为 0。
最佳 macro-F1 与三种 train-label baseline 如下：

| task | best 120B blind | blind | best 120B visible | visible | MiniMol head | Morgan KNN | MiniMol KNN |
|---|---|---:|---|---:|---:|---:|---:|
| BBB_Martins | Starling Full / Flat | 0.6753 | Starling Direct | 0.6900 | **0.7226** | 0.5857 | 0.6907 |
| Bioavailability_Ma | Starling Direct / Full | 0.6031 | Starling Full / Flat | **0.6787** | 0.6755 | 0.5856 | 0.6149 |
| Skin_Reaction | Starling Full / Mechanism | 0.6010 | Starling Full / Mechanism | **0.6151** | 0.5749 | 0.5201 | 0.5749 |

结果 roots：

```text
outputs/paper/starling_benchmark_results_scaffold_record_supported_v2_valid_gpt_oss_120b_blind/
outputs/paper/starling_benchmark_results_scaffold_record_supported_v2_valid_gpt_oss_120b_visible/
outputs/baselines/minimol_starling_record_supported_v2/
outputs/baselines/structure_knn_starling_record_supported_v2/
outputs/baselines/minimol_embedding_knn_starling_record_supported_v2/
```

新旧 dataset 的 GPT-OSS-120B 全设置与三种 baseline 由唯一总图入口生成，图中明确标注 v2 的
multi-record-heavy held-out 特征：

```text
outputs/paper/starling_benchmark_results_scaffold_record_supported_v2_valid/figures/
  gpt_oss_120b_dataset_version_comparison.svg
  gpt_oss_120b_dataset_version_comparison.png
```

Morgan KNN 的 v2 下降不是 train size 变小造成。严格 leak-free 的 old/new train x valid 交叉分解显示：
BBB 总下降 `-0.1270` 中 valid cohort 替换贡献 `-0.1291`；Skin 总下降 `-0.0862` 中 valid cohort 替换贡献
`-0.0505`，同时移除/加入 train rows 共贡献 `-0.0357`；Bioavailability 总下降 `-0.0343` 主要来自新增
train composition（`-0.0218`）。新 valid 的 unique scaffold 数从 BBB/Skin/Bio 的 `277/133/151`
增至 `377/187/175`，Morgan 的负类 recall 分别从 `0.520/0.392/0.361` 降至
`0.324/0.274/0.276`。因此最稳妥的结论是：v2 把 held-out 重心移到 multi-record、更多样且更难迁移的
scaffold，暴露了 Morgan local-neighborhood 的 domain-shift 弱点；不是 label disagreement，也不是 leakage。
可复核诊断位于：

```text
outputs/paper/starling_benchmark_results_scaffold_record_supported_v2_valid/diagnostics/morgan_dataset_shift/
```

GLM blind 首轮在 endpoint/tunnel 中断前完成 5,876/6,887 sample-conditions，仍有 1,011 个失败；其中
BBB 仅余 3 个，Skin 余 314 个，Bioavailability 余 694 个。该 root 尚未通过 zero-failure gate，visible
矩阵尚未启动，因此所有 partial GLM macro-F1 都不得进入正式表或显著性分析。恢复 endpoint 后必须使用
同一 root 的 `--skip-existing` 修复，再启动独立 visible root。

曾生成的 exploratory `record_supported_v1` 因 held-out 分布不符合最终设计，数据、indices、agent runs、
baselines 和显著性 artifact 已于 2026-08-07 删除；这里只保留这条 lineage tombstone，不再引用旧路径。

## 第一版 record-agreement benchmark（historical comparison）

上一版 strict-conflict lineage 的机器可读结果和 canonical bar chart：

```text
outputs/paper/starling_benchmark_results/summary.json
outputs/paper/starling_benchmark_results/metrics.tsv
outputs/paper/starling_benchmark_results/report.md
outputs/paper/starling_benchmark_results/figures/starling_benchmark_overview.svg
outputs/paper/starling_benchmark_results/figures/starling_benchmark_overview_highres.png
```

当前 70% record-agreement v4 的 GPT-OSS-20B identity-blind scaffold-valid 与 train-label baseline
对比单独保存在：

```text
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b/summary.json
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b/metrics.tsv
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b/report.md
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b/figures/starling_benchmark_overview.svg
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b/figures/starling_benchmark_overview_highres.png
```

这一轮只评估 scaffold `valid`，不使用 test。MiniMol head 使用 scaffold train 全量训练、5-member
ensemble、25 epochs 和固定 0.5 threshold；两种 KNN 都只使用 scaffold train labels、`k=3` 和未加权多数票。
结果如下（macro-F1）：

| task | best GPT-OSS-20B condition | best GPT-OSS | MiniMol head | Morgan KNN | MiniMol KNN |
|---|---|---:|---:|---:|---:|
| BBB_Martins | Starling Full / Mechanism | 0.6934 | 0.7108 | **0.7127** | 0.7007 |
| Skin_Reaction | Starling Full / Mechanism | 0.5585 | 0.5900 | 0.6064 | **0.6166** |
| Bioavailability_Ma | ChEMBL Full / Flat | 0.6443 | **0.6577** | 0.6199 | 0.5757 |

GPT-OSS 共 22 个 condition、6,887 个 sample-condition；其中
`Bioavailability_Ma / chembl_full_flat / idx00076` 因输入超过 131,072 context limit 失败，未补跑。
汇总按预先明确的 `count_as_incorrect_opposite_label` policy 将该样本计错：该 condition 的
failure-inclusive confusion matrix 为 TN=40、FP=21、FN=47、TP=101，macro-F1 从只统计成功样本的
0.6476 调整为 0.6443。其余 6,886 个 sample-condition 成功。

同一 frozen scaffold-valid contract 的 GPT-OSS-120B 结果独立保存在：

```text
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid_gpt_oss_120b/
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_120b/summary.json
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_120b/metrics.tsv
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_120b/report.md
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_120b/figures/starling_benchmark_overview.svg
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_120b/figures/starling_benchmark_overview_highres.png
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b_vs_120b/figures/starling_model_comparison.svg
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b_vs_120b/figures/starling_model_comparison_highres.png
```

120B 运行只更换 served model；仍使用 scaffold `valid`、`identity_blind + parent_disjoint`、同一 held-out
retrieval index、同一 prompt/tool contract，并复用上表同一批 train-label baselines。launcher 使用 4 个
TP=2 vLLM backends，经 HAProxy 统一暴露，执行形状为 4 个 condition workers x 每 condition 64 requests，
全局 effective concurrency 为 256。最佳 macro-F1 与 20B 对照如下：

| task | best GPT-OSS-120B condition | 120B | best 20B | delta | strongest baseline |
|---|---|---:|---:|---:|---:|
| BBB_Martins | Starling Direct | **0.7138** | 0.6934 | +0.0203 | Morgan KNN 0.7127 |
| Skin_Reaction | Starling Full / Mechanism | 0.5700 | 0.5585 | +0.0115 | MiniMol KNN 0.6166 |
| Bioavailability_Ma | ChEMBL Full / Flat | **0.6498** | 0.6443 | +0.0055 | MiniMol head 0.6577 |

120B 同样完成 22 个 condition、6,887 个 sample-condition。初次运行的 Bioavailability flat branches
包含 49 个 request timeout（`chembl_full_flat` 27 个、`starling_full_flat` 22 个）和一个确定性的 context
failure：`chembl_full_flat/idx00076` 输入 140,665 tokens，超过 131,072 上限。49 个 timeout 使用 600 秒
timeout、最多 50 个实际并发请求全部成功补齐；旧失败目录和日志保存在同一 task root 下的
`retry_archive_20260802_timeout300/`。最终只有 idx00076 按
`count_as_incorrect_opposite_label` policy 计错，6,886 个 sample-condition 成功；重汇总后
`chembl_full_flat` 与 `starling_full_flat` 的正式 macro-F1 分别为 0.6498 和 0.6346。

### GLM-5.2 NVFP4 identity-blind scaffold-valid（2026-08-03 完成）

GLM 在同一 frozen scaffold `valid`、同一 held-out-filtered index 和
`identity_blind + parent_disjoint` contract 上完成 22 个 condition、6,887 个 sample-condition，严格完整性
检查为 `6887/6887`，没有使用 failure-inclusive 计错。机器可读结果位于：

```text
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid_glm_5_2_nvfp4/
outputs/paper/starling_benchmark_results_scaffold_valid_glm_5_2_nvfp4/metrics.tsv
outputs/paper/starling_benchmark_results_scaffold_valid_glm_5_2_nvfp4/summary.json
outputs/paper/starling_benchmark_results_scaffold_valid_glm_5_2_nvfp4/report.md
```

最佳 agent macro-F1 为：

| task | best GLM condition | GLM | best GPT-OSS-120B blind | delta |
|---|---|---:|---:|---:|
| BBB_Martins | Starling Full / Flat | **0.7337** | 0.7138 | +0.0199 |
| Skin_Reaction | Starling Direct | **0.6046** | 0.5700 | +0.0346 |
| Bioavailability_Ma | Starling Full / Mechanism | 0.6467 | **0.6498** | -0.0031 |

逐 trace 审计覆盖 22 条件/6,887 样本：query-SMILES trace leak 为 0，prompt structure/identifier leak 为 0。
最初报告的 3 个 name leak 均来自 Starling source name `PER` 与英文介词 `per` 的词边界碰撞；`PER` 已加入
generic-name audit allowlist 后重新审计为 0。全量 28,298 个 retained neighbor slots 均标记为
`structural_analog`，parent-policy conflict 和低于 0.30 threshold 的补位均为 0；对应 held-out index metadata
记录 `zero_parent_overlap=true` 和 residual held-out parent 为 0。

注意 canonical matrix manifest 会被最后一次 selection/repair launcher 原子更新，因此该 root 当前 manifest
显示最后的 `1x1` finalization，而不是整轮历史峰值并发。完整启动/修复形状保存在同 root 的
`run_*` / `repair_*` logs；不要仅凭最后一个 manifest 反推整轮吞吐。

### GPT-OSS deployment-visible + parent-disjoint 补充矩阵（2026-08-03）

两个 GPT-OSS 模型随后在同一 frozen scaffold `valid` 和同一 held-out-filtered retrieval index 上补齐
`deployment_visible + parent_disjoint`。这里的 visible 同时表示 query/neighbor identity 对 LLM 可见，以及
comparison tools 由模型通过 function call 执行；它不是只改变 SMILES 脱敏的单因素消融。结果分别保存在：

```text
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid_gpt_oss_20b_visible_parent_disjoint/
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b_visible_parent_disjoint/
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid_gpt_oss_120b_visible_parent_disjoint/
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_120b_visible_parent_disjoint/
```

每个模型均完成 22 个 condition、6,887 个 sample-condition。两边唯一最终失败都是
`Bioavailability_Ma / chembl_full_flat / idx00076`：visible prompt 为 151,630 tokens，超过两个 served
model 的 131,072 context limit。该样本无法在不改变冻结 evidence/prompt contract 的情况下修复，故按
`count_as_incorrect_opposite_label` 计错；每个模型其余 6,886 个 sample-condition 成功。两套矩阵的 19 个
retrieval conditions 全量审计均为 parent-policy conflict=0、below-similarity-threshold=0，manifest 均为
`fresh_parent_disjoint=true`、`operational_staging_used=false`。

20B 首轮有 177 个失败 run（失败 group 合计为 177 个 Harmony tool-header parser error、1 个 request
timeout 和上述 1 个 context error；少数 run 同时包含多个失败 group）。先以 32/16/8 的递减全局并发补跑，
最后对单个长尾使用并发 1；所有可重试失败均成功。各轮原始失败 run、日志和 metrics 保存在该 model root
下的 `retry_archive_20260803_*`。120B 首轮只有上述确定性 context failure。20B 服务最终使用 node001 上
8 个单-GPU backend 和 HAProxy；120B 使用 node002 上常驻的 4 个 TP=2 backend 和 HAProxy。

visible 主矩阵的最佳 macro-F1 为：

| task | best 20B visible condition | 20B visible | best 120B visible condition | 120B visible |
|---|---|---:|---|---:|
| BBB_Martins | Starling Direct | 0.6751 | Starling Full / Flat | **0.6926** |
| Skin_Reaction | Starling Full / Mechanism | 0.5525 | Starling Full / Flat | **0.5913** |
| Bioavailability_Ma | Starling Full / Flat | 0.6545 | ChEMBL Full / Mechanism | **0.6703** |

这些数值与 identity-blind 主矩阵共享数据、retrieval identity policy、prompt schema 和模型权重，但 visible
合同还改变了结构可见性与 tool execution，因此图中的 blind/visible 差异应解释为完整部署合同差异，不能只
归因于分子 identity visibility。

GLM 的同合同 `deployment_visible + parent_disjoint` scaffold-valid 矩阵已于 2026-08-04 完成，输出到：

```text
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid_glm_5_2_nvfp4_visible_parent_disjoint/
outputs/paper/starling_benchmark_results_scaffold_valid_glm_5_2_nvfp4_visible_parent_disjoint/metrics.tsv
outputs/paper/starling_benchmark_results_scaffold_valid_glm_5_2_nvfp4_visible_parent_disjoint/summary.json
outputs/paper/starling_benchmark_results_scaffold_valid_glm_5_2_nvfp4_visible_parent_disjoint/report.md
```

该运行最初使用静态 condition lanes；在 `6765/6887` 完整样本处安全停止旧 launcher，并迁移到唯一的
`global_prompt_ready_pool.v1`。首次恢复扫描直接还原 6765 个完整样本，把跨 task/condition 的 144 个缺失
group stage 放进同一 128-slot pool；46 分钟后完整样本增至 6879，余下 8 个极端长请求随后以相同模型、prompt、
reasoning 和 validation 合同、仅提高单请求 timeout 完成。迁移审查还发现 staged final 必须保持旧 pipeline 的
`group_id` 排序；受影响的 38 个 Bioavailability mechanism final/trace 已先按 SHA-256 归档，再只重建 final。
归档及审计位于 `scheduler_migration_audit/final_prompt_order_pre_fix_20260804/`。

最终 gate 为 22 个 batch、6887 predictions、6887 run directories、`sum(n_failed_runs)=0`，逐 run 的
task-specific prediction、single、expected group、final 和 trace 完整性错误均为 0；38 个重建 final 的实际 prompt
顺序和归档 hash 也全部通过。旧 static/condition-lane scheduler、`--condition-workers`、batch
`--group-workers` 和 scheduler 选择开关已从正式与 generic paper runner 删除；batch CLI、Starling matrix 和
MiniMol orchestrator 统一复用 global ready pool，不再保留可绕过全局预算的 endpoint fan-out 路径。

GLM visible 汇总已加入同一 canonical blind+visible 总图。它不是 visible-only 小图：图中同时保留
GPT-OSS-20B、GPT-OSS-120B 和 GLM-5.2 NVFP4 的 blind/visible 六套完整 series，覆盖三个 task 的全部
22 个 agent conditions，并保留共享 train-label baselines 和 matched opt-in experiments。各 task 的最佳
GLM visible macro-F1 为：BBB `starling_full_flat` 0.7096、Skin `starling_direct` 0.6208、Bioavailability
`starling_full_mechanism` 0.6832；相对 GLM blind 的同 task 最佳值分别为 -0.0242、+0.0162、+0.0365。

blind/visible 全条件总图入口为：

```bash
python -m tools.chembl_tool.paper_experiments.plot_starling_model_comparison \
  --reference-metrics outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b/metrics.tsv \
  --candidate-metrics outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_120b/metrics.tsv \
  --comparison-metrics outputs/paper/starling_benchmark_results_scaffold_valid_glm_5_2_nvfp4/metrics.tsv \
  --comparison-metrics outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b_visible_parent_disjoint/metrics.tsv \
  --comparison-metrics outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_120b_visible_parent_disjoint/metrics.tsv \
  --comparison-metrics outputs/paper/starling_benchmark_results_scaffold_valid_glm_5_2_nvfp4_visible_parent_disjoint/metrics.tsv \
  --experiment-metrics outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/three_way_metrics.tsv \
  --experiment-metrics outputs/paper/coverage_mmp_ledger_gpt_oss_120b_scaffold_valid/analysis/figure_metrics.tsv \
  --paired-ci-metrics outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b_vs_120b/analysis/best_agent_paired_baseline_bootstrap_ci.tsv \
  --paired-significance-display pvalue \
  --output outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b_vs_120b/figures/starling_model_comparison.svg \
  --png-output outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b_vs_120b/figures/starling_model_comparison_highres.png
```

这是 Starling model、visibility、baseline 和 matched ablation 的唯一正式总图。完整 model/visibility
summary 通过重复 `--comparison-metrics` 追加；matched experiment 通过重复 `--experiment-metrics` 追加，
不再生成单实验 overview/bar chart。experiment TSV 中的 anchor 必须与 candidate summary 的既有 condition
在 subset、样本数和 macro-F1 上一致，绘图时只用于合同校验、不重复显示。
总图现在通过 `--paired-significance-display pvalue` 在每个 task panel 内只显示 best agent
相对 MiniMol train-all 和 Morgan KNN 的单侧 paired-permutation p-value，检验方向为
`H1: best agent > baseline`，图内不再显示 95% CI。六个检验均未达 `p < 0.05`；且因为
单侧方向和 best agent 均是看到同一 scaffold-valid 结果后确定，该 p-value 展示明确标为
exploratory，不写成预注册 confirmatory 检验。

本轮 baseline 输出根：

```text
outputs/baselines/minimol_starling_valid/<Task>/scaffold/
outputs/baselines/structure_knn_starling_valid/<Task>/scaffold/
outputs/baselines/minimol_embedding_knn_starling_valid/<Task>/scaffold/
```

## 1. 第一版冻结的数据与 label 决策

公共协议和唯一数据构建入口：

```text
tools/chembl_tool/common/starling/STARLING_BENCHMARK_PROTOCOL.md
tools/chembl_tool/common/starling/benchmark_dataset.py
tools/chembl_tool/common/starling/build_benchmark_datasets.py
```

当前冻结 v4 规则：

- parent agreement 为 record-weighted `max(n0,n1)/(n0+n1)`，threshold 为 70%，精确 tie 拒绝；
- valid 与 test size 分别为 `min(500, floor(0.1 * n_binary_molecules))`；
- `random` 是固定 seed 的 label-stratified stable-hash split；
- `scaffold` 以 canonical Bemis–Murcko scaffold 为不可拆分 group；
- 两套 split 从同一批 accepted binary parents 独立构建；
- 每套 train/valid/test parent identity 两两零重叠，scaffold 还要求 scaffold 两两零重叠；
- source row 先按 task adapter 独立转成 `0/1/ambiguous`，再按
  `rdkit_fragment_parent.v1` 聚合；
- 同一个 normalized parent 的 accepted rows 同时出现 0 和 1 时，达到 70% 就接受多数 label；
- `within_record_label_conflict` 指同一条 row 内可用信号互相矛盾；这种 row 在 parent 聚合前拒绝。

冻结规模：

| task | binary parents | rejected | valid/test | random valid / test Y=0,Y=1 | scaffold valid / test Y=0,Y=1 |
|---|---:|---:|---:|---:|---:|
| BBB_Martins | 19,425 | 1,091 | 500 / 500 | 139,361 / 139,361 | 150,350 / 148,352 |
| Bioavailability_Ma | 2,092 | 106 | 209 / 209 | 58,151 / 58,151 | 61,148 / 65,144 |
| Skin_Reaction | 2,456 | 465 | 245 / 245 | 75,170 / 75,170 | 74,171 / 71,174 |

Task label：

```text
BBB_Martins:
  Y=1 pass；Y=0 fail；明确 logBB 使用 >= -1 threshold。

Bioavailability_Ma:
  human oral F >= 20% -> Y=1；F < 20% -> Y=0。
  2026-08-01 起使用 `bioavailability_canonical_direct.v2`：固定 HF snapshot 与 local 中明确
  absolute/oral-IV rows 合并，跨来源近等值 claim 去重；relative/ambiguous local rows 只留在 residual
  inference evidence。当前 direct claims SHA-256 为
  `045261cbda785092143eeadd636f78399f7b02f951b23480b16fb8dde22661c5`。

Skin_Reaction:
  Y=1 skin sensitizer；Y=0 non-sensitizer。
  irritation、phototoxicity、generic local damage 和 skin exposure 不构成这个 gold label。

ClinTox:
  当前没有与 toxicity-caused clinical-trial failure 同定义的 Starling direct source，
  所以没有构造 Starling split。
```

此前所有 17,893/1,828/1,900-parent strict-conflict benchmark 的 baseline、agent 和图表，以及更早的
Bioavailability 1,862-parent mixed-source 结果，均为 historical lineage，不得与当前 lineage 混表。
第一版 record-agreement agent roots 为：

```text
outputs/paper/molecular_evidence_agent_starling_random_record_agreement70_split811_v1/
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1/
```

当前 record-supported v2 的 canonical index root 和 model-specific valid roots 见第 0 节。

数据与诊断入口：

```text
Bioavailability canonical source:
  tools/chembl_tool/tasks/bioavailability_ma/build_canonical_starling_source.py

record/PMID provenance distributions:
  tools/chembl_tool/paper_experiments/analyze_starling_parent_provenance.py

50/60/70/80/90% threshold comparison:
  tools/chembl_tool/paper_experiments/analyze_starling_majority_thresholds.py

70% gold split build:
  tools/chembl_tool/common/starling/build_benchmark_datasets.py

valid+test-heldout index build:
  tools/chembl_tool/paper_experiments/build_starling_benchmark_indices.py

identity-blind parent-disjoint valid/test matrix:
  tools/chembl_tool/paper_experiments/starling_benchmark_matrix.py
```

具体 numerical/unit/free-text 映射以三个 task adapter 为代码真相，汇总规则见
`STARLING_BENCHMARK_PROTOCOL.md`：

```text
tools/chembl_tool/tasks/bbb_martins/starling_benchmark.py
tools/chembl_tool/tasks/bioavailability_ma/starling_benchmark.py
tools/chembl_tool/tasks/skin_reaction/starling_benchmark.py
```

## 2. Held-out retrieval 隔离

Gold split 与 retrieval evidence library 是两个步骤。正式 retrieval condition 分别使用对应构造方法的
`heldout_molecule_labels.jsonl` 从 full Starling evidence 中删除全部 valid+test parents，再重建 train-only index；
不能只靠 query-time exact-SMILES exclusion，也不能直接使用 full-source index。

入口：

```text
tools/chembl_tool/common/starling/heldout_index.py
tools/chembl_tool/paper_experiments/build_starling_benchmark_indices.py
```

当前 v4 formal pipeline condition 使用 `identity_blind + parent_disjoint` fresh-run 制度：

```text
none:
  没有 retrieval，identity policy 标为 not applicable。

retrieval conditions:
  直接从 valid+test-heldout-filtered index 做 fresh parent_disjoint retrieval；
  retained neighbor 与 query parent identity 零重叠，不读 operational reuse plan。
```

2026-08-01 的 random-valid BBB query-only 首次 512 并发压力运行完成 499/500，唯一失败是
single branch transport timeout，不是 structured-output validation 失败。按预定 contingency 将正式
launcher 稍降为 `parallelism=384, group_workers=1`，只重跑该样本后达到 500/500、0 failed。
但 384 并发的 BBB ChEMBL full-flat 又产生 193/500 失败，其中 190 个为 group request timeout；
矩阵随后以较低并发恢复。当前 random-valid 已生成 22 个 condition metrics 和 6,886 个 final artifacts，
严格完整性为 `6885/6887`；剩余两个失败分别位于 Bioavailability `chembl_full_flat` 和
`chembl_full_mechanism`。机器可读 failure-inclusive pipeline summary 位于
`outputs/paper/starling_benchmark_results_random_valid_glm_5_2_nvfp4/`，但该 split 仍未通过 zero-failure
valid gate，也没有匹配的 v4 random-valid train-label baselines，因此不进入 scaffold-valid model comparison
或正式图。当前默认保持单一 `parallelism=128` global prompt pool；完整 valid gate 通过前不启动 test。

## 3. Historical strict-conflict performance

以下数值全部来自上一版 strict-conflict train/test split，不是当前 70%-agreement scaffold benchmark 的结果。
它们仅用于历史复现；第一版 record-agreement 已完成 scaffold-valid 的 GLM、GPT-OSS-20B/120B blind agent matrix、两套
GPT-OSS visible matrix 和 matched MiniMol/Morgan/MiniMol-KNN baselines，结果见本文开头。正式 test 尚未
启动；random-valid GLM 保留两个失败，是独立 lineage，不与这里的 scaffold-valid 模型对照混表。
V4 主合同已在 `AGENTS.md` / `EXPERIMENT_PLAN.md` 冻结为
`identity_blind + parent_disjoint` fresh-run、operational staging disabled、endpoint 上限 512/当前
GLM launcher 128；GPT-OSS scaffold-valid 使用独立 model-specific roots 和 4×64 condition lanes。

下表均为完整 test 的 macro-F1。`Starling direct` 对 Bioavailability 指 full direct-F condition；
另有 numeric-only direct-F：random `0.6652`、scaffold `0.6392`。所有表内 formal pipeline、
MiniMol 和 KNN 条件均为 0 failed samples。

| split / task | none | ChEMBL direct | ChEMBL flat | ChEMBL mechanism | Starling direct | Starling flat | Starling mechanism | MiniMol head | Morgan KNN | MiniMol KNN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random / BBB | 0.5992 | 0.6221 | 0.6378 | 0.6420 | 0.7527 | 0.7462 | 0.7422 | **0.7567** | 0.6950 | 0.7314 |
| random / Bioavailability | 0.4677 | 0.6114 | 0.6265 | 0.6152 | 0.6741 | 0.6871 | 0.6991 | 0.7120 | 0.6824 | **0.7256** |
| random / Skin | 0.6223 | 0.6061 | 0.6023 | 0.6303 | **0.6431** | 0.6353 | 0.6300 | 0.6066 | 0.5973 | 0.5905 |
| scaffold / BBB | 0.5788 | 0.6129 | 0.6484 | 0.6439 | 0.6995 | 0.7024 | 0.7005 | **0.7188** | 0.6745 | 0.6946 |
| scaffold / Bioavailability | 0.4529 | 0.6350 | 0.6617 | 0.6712 | 0.6516 | 0.6565 | 0.6743 | 0.6421 | 0.6952 | **0.7028** |
| scaffold / Skin | 0.5884 | 0.5862 | **0.6107** | 0.5918 | 0.5973 | 0.5921 | 0.5836 | 0.6050 | 0.5372 | 0.5892 |

这些是 point estimates，不支持“增加更多 evidence 或 mechanism branches 必然提升”的普遍结论：

- BBB 的 Starling direct/full 明显高于 ChEMBL，但 direct/flat/mechanism 内部差异较小；
- Bioavailability 的 MiniMol embedding KNN 在 random/scaffold 都高于 Morgan KNN 和 MiniMol head；
- Skin 的 Starling direct 到 full mechanism 在两套 split 都下降，见第 8 节；
- random 与 scaffold 是不同 test sets，不能把两者的绝对高低直接解释为方法对 scaffold 的因果效应。

## 4. Historical strict-conflict MiniMol 选择口径

上一版 strict-conflict Starling MiniMol 使用 `--train-all`：

- 使用该 split 的全部 `train.jsonl`；
- 不读取或构造 `valid.jsonl`；
- 每个 ensemble member 固定 epoch；
- test threshold 固定为 `0.5`；
- 不用 test 选择 epoch、hyperparameter 或 threshold；
- 输出中的 validation metrics 为 null。

因此 MiniMol 最终 performance 不是“在 test 上选最好 epoch”，而是冻结训练设置后对 test 评估一次。
入口：

```text
baselines/minimol/run_bioavailability_ma.py --train-all
```

正式产物：

```text
outputs/baselines/minimol_starling/<Task>/<random|scaffold>/
```

## 5. Morgan KNN baseline 的精确定义

入口：

```text
baselines/structure_knn/run.py
```

正式设置：

```text
fingerprint: RDKit Morgan radius=2, 2048 bits, no chirality
candidate pool: 同 split 的 train.jsonl
k: 3
ranking: query-to-train Tanimoto similarity
vote: 三个 neighbor molecules 各一票，未加权多数票
score/AUROC: 三票中的正类比例
```

它不是“top-k records 的 flat vote”。Benchmark builder 已经先把 source records 按 normalized molecule
parent 聚合，冲突 parent 已排除，`train.jsonl` 每个 accepted parent 只有一个 binary label。因此 KNN 的
top 3 是三个 train molecules，每个 molecule 恰好贡献一票；不会让一个有多篇文献/多条 source records
的 molecule 获得更多票。

Bioavailability 的 scaffold KNN macro-F1 `0.6952` 高于 random 的 `0.6824`，但 accuracy 反而更低
（`0.7634` vs `0.7742`），AUROC 几乎相同（`0.7302` vs `0.7295`）。Scaffold test 有更多负类
（113 vs 99），且 negative-class 表现较好，因而 macro-F1 上升。这不是异常，也不能据此声称 scaffold
split 一般比 random 更容易。

正式产物：

```text
outputs/baselines/structure_knn_starling/<Task>/<random|scaffold>/
```

### MiniMol embedding KNN

2026-07-28 新增的 retrieval-only 对照保持 Morgan KNN 的 train pool、`k=3`、未加权多数票和
正类邻居比例 score 不变；只把 ranking feature 从 Morgan fingerprint/Tanimoto 换成正式 MiniMol
baseline 已缓存的 512 维 embedding/L2-normalized cosine similarity。Cache loader 会逐条校验 SMILES、
label 和 row order 与当前 split 完全一致，并在 manifest 中记录 train/test JSONL 与 embedding cache
的 SHA-256。

入口与产物：

```text
baselines/minimol/run_embedding_knn.py
outputs/baselines/minimol_embedding_knn_starling/<Task>/<random|scaffold>/
machine-readable method: minimol_embedding_cosine_knn_k3
```

相对 Morgan KNN 的完整 test 结果：

| split / task | macro-F1 | Δ macro-F1 | accuracy | Δ accuracy | AUROC | Δ AUROC |
|---|---:|---:|---:|---:|---:|---:|
| random / BBB | 0.7314 | +0.0363 | 0.8040 | +0.0220 | 0.7693 | +0.0150 |
| scaffold / BBB | 0.6946 | +0.0201 | 0.7740 | +0.0060 | 0.7790 | +0.0606 |
| random / Bioavailability | 0.7256 | +0.0432 | 0.7957 | +0.0215 | 0.7741 | +0.0445 |
| scaffold / Bioavailability | 0.7028 | +0.0076 | 0.7634 | +0.0000 | 0.7301 | -0.0001 |
| random / Skin | 0.5905 | -0.0068 | 0.6526 | +0.0132 | 0.6374 | +0.0196 |
| scaffold / Skin | 0.5892 | +0.0521 | 0.6316 | +0.0368 | 0.6197 | +0.0930 |

这组 point estimates 表明 MiniMol cosine neighborhood 在 5/6 条件上提高 macro-F1，在 6/6 条件上
不降低 accuracy，在 5/6 条件上提高 AUROC；唯一明显不一致的是 Bioavailability scaffold AUROC
几乎持平而略低 `0.0001`，Skin random 则 accuracy/AUROC 提高但 macro-F1 下降 `0.0068`。
这不等于 learned embedding 普遍支配 Morgan：两种 split 的 test 集不同，而且当前只冻结了 `k=3`
和 cosine 一个设置。

该 baseline 已进入 `summarize_starling_benchmark.py` 生成的 `metrics.tsv`、`summary.json`、
`report.md` 和 canonical `starling_benchmark_overview.{svg,png}`，不是只存在于单独实验目录。

## 6. MiniMol embedding agent retrieval（operational + parent-disjoint）

2026-07-30 完成 MiniMol/cosine agent retrieval 的 random/scaffold operational 矩阵。该实验不是
train-label KNN：它保持 evidence source、direct/flat/mechanism organization、top-k、GLM、prompt、
tool execution 和 inference settings 不变，只将 agent 的 neighbor ranking feature 从
Morgan/Tanimoto 换为 L2-normalized MiniMol embedding/cosine。

完整 gate：

```text
conditions: 38/38（random 19，scaffold 19）
sample-conditions: 15,768/15,768 successful
failed: 0
neighbor identity policy: operational
```

失败的 structured-output 分支均按原设置定点重跑；每次失败版本保存在
`outputs/paper/minimol_retrieval_agent_results/failed_attempts/operational/attempt_*/`，没有通过
postprocess 补字段或改写 prediction。

与已有正式 Morgan agent retrieval 的配对 point estimates：

| split / task | paired win / loss / tie | best MiniMol operational condition | MiniMol macro-F1 | best Morgan formal condition | Morgan macro-F1 |
|---|---:|---|---:|---|---:|
| random / BBB | 0 / 5 / 1 | Starling full / flat | 0.7353 | Starling direct | 0.7527 |
| random / Bioavailability | 7 / 0 / 0 | Starling full / flat | 0.7089 | Starling full / mechanism | 0.6991 |
| random / Skin | 2 / 4 / 0 | Starling direct | 0.6267 | Starling direct | 0.6431 |
| scaffold / BBB | 4 / 2 / 0 | Starling direct | 0.7224 | Starling full / flat | 0.7024 |
| scaffold / Bioavailability | 7 / 0 / 0 | Starling full / mechanism | 0.7111 | Starling full / mechanism | 0.6743 |
| scaffold / Skin | 6 / 0 / 0 | Starling full / mechanism | 0.6252 | ChEMBL full / flat | 0.6107 |

38 个 paired retrieval conditions 中 MiniMol operational 为 26 win / 11 loss / 1 tie；平均
`Δ macro-F1 = +0.0111`，中位数 `+0.0098`。最大提高是 scaffold Bioavailability 的
ChEMBL direct（`+0.0549`），最大下降是 random BBB 的 ChEMBL full/mechanism（`-0.0250`）。
Bioavailability 两套 split 的 14/14 paired conditions 都提高；random BBB 则没有提高，
Skin random 也有 4/6 下降。这些异质性不支持“MiniMol feature 普遍支配 Morgan”的结论。

2026-07-30 的 operational-only 配对图只是结果比较，不是纯 retrieval-feature causal attribution：已有 Morgan bars 使用正式
`parent_disjoint` policy，新 MiniMol bars是本轮 `operational` policy，因此同时改变了 feature 和
neighbor identity policy。No-retrieval、MiniMol train-all head、Morgan KNN 和 MiniMol KNN 仅作为同图
context。当时尚不能作 feature-only claim；下述 2026-08-01 matched 结果解决了这项限制。

2026-08-01 已补齐同轮 MiniMol `parent_disjoint`、paired summary 和正式图：

```text
conditions: 38/38（random 19，scaffold 19）
missing conditions: 0
Morgan failed runs: 0
MiniMol failed runs: 0
neighbor identity policy: parent_disjoint（两侧 matched）
bootstrap replicates: 10,000 per condition
```

在 identity policy matched 后，38 个条件仍为 26 win / 11 loss / 1 tie；平均
`Delta macro-F1 = +0.0103`，中位数 `+0.0112`。六个 task/split 的最优条件与上表一致。
10,000 次 paired bootstrap 中有 6 个条件的 95% CI 不跨 0：random BBB ChEMBL full/mechanism
偏向 Morgan；random Bioavailability ChEMBL full/mechanism、scaffold Skin Starling full/mechanism，
以及 scaffold Bioavailability 的 ChEMBL direct、Starling full/flat、Starling full/mechanism 偏向 MiniMol。
这些是 condition-level 未做多重比较校正的结果，仍不支持把平均改善解释成普遍支配。

正式 parent-disjoint 产物：

```text
outputs/paper/minimol_retrieval_agent_results/summary.json
outputs/paper/minimol_retrieval_agent_results/condition_results.tsv
outputs/paper/minimol_retrieval_agent_results/report.md
outputs/paper/minimol_retrieval_agent_results/all_results_comparison.tsv
outputs/paper/minimol_retrieval_agent_results/figures/minimol_vs_morgan_agent_retrieval.svg
outputs/paper/minimol_retrieval_agent_results/figures/minimol_vs_morgan_agent_retrieval_highres.png
outputs/paper/minimol_retrieval_agent_results/figures/starling_benchmark_with_minimol_agent.svg
outputs/paper/minimol_retrieval_agent_results/figures/starling_benchmark_with_minimol_agent_highres.png
```

其中 `minimol_vs_morgan_agent_retrieval` 是 38 个 feature-only paired 条件；
`starling_benchmark_with_minimol_agent` 将 matched parent-disjoint MiniMol agent bars 与 no-retrieval、
MiniMol train-all head、Morgan KNN、MiniMol KNN 和既有 Morgan agent 结果放在同一张 bar chart。

历史 operational-only 入口与产物仍保留用于 sensitivity audit：

```text
tools/chembl_tool/paper_experiments/run_minimol_retrieval_agent_experiment.py
tools/chembl_tool/paper_experiments/plot_starling_with_minimol_agent.py

outputs/paper/minimol_retrieval_agent_results/operational_all_results_comparison.tsv
outputs/paper/minimol_retrieval_agent_results/figures/starling_benchmark_with_minimol_agent_operational.svg
outputs/paper/minimol_retrieval_agent_results/figures/starling_benchmark_with_minimol_agent_operational_highres.png
```

## 7. Morgan similarity vs query-feature coverage agent retrieval

2026-07-28 完成 coverage selector 的六个 Starling task/split 全样本 matched LLM 对比，全部
`n_failed_runs=0`。两组条件固定使用 Starling full evidence、`full_mechanism`、deployment-visible、
`parent_disjoint`、`top_k_per_group=3`、`min_similarity=0.30`、相同 GLM/single analysis/group reuse 和
decoding；唯一变化是 neighbor selector：按 Morgan/Tanimoto similarity 排序，或在 similarity 不低于
`0.30` 的候选中贪心最大化 query Morgan-bit union coverage。

| split / task | Morgan accuracy | Coverage accuracy | Δ accuracy | Morgan macro-F1 | Coverage macro-F1 | Δ macro-F1 | flips | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random / BBB | 0.7720 | 0.7440 | -0.0280 | 0.7422 | 0.7078 | -0.0344 | 34 | 0.0243 |
| scaffold / BBB | 0.7380 | 0.7200 | -0.0180 | 0.7005 | 0.6862 | -0.0142 | 33 | 0.1628 |
| random / Bioavailability | 0.7231 | 0.7016 | -0.0215 | 0.6991 | 0.6779 | -0.0213 | 28 | 0.1849 |
| scaffold / Bioavailability | 0.6855 | 0.6801 | -0.0054 | 0.6743 | 0.6694 | -0.0049 | 30 | 0.8555 |
| random / Skin | 0.6526 | 0.6447 | -0.0079 | 0.6300 | 0.6200 | -0.0100 | 23 | 0.6776 |
| scaffold / Skin | 0.6211 | 0.6395 | +0.0184 | 0.5836 | 0.6017 | +0.0181 | 25 | 0.2295 |

Coverage 在 6 个条件中有 5 个降低 accuracy 和 macro-F1；唯一上升是 Skin scaffold，但 paired
McNemar 不显著，macro-F1 paired-bootstrap 95% CI `[-0.0058, +0.0426]` 仍跨零。BBB random 的下降
最明确：Morgan-only correct / coverage-only correct 为 `24 / 10`，McNemar `p=0.0243`，macro-F1
delta bootstrap 95% CI `[-0.0612, -0.0094]`。因此当前结果不支持用 pure coverage selector 全局替代
Morgan similarity selector。它仍可作为 hybrid selector 的互补项，但下一版必须同时保留 analog relevance
约束并预先冻结组合规则，不能根据这六个 test 结果调权后再在同一 test 上作确认性结论。

2026-07-31 对全部 retrieval 和 reasoning trace 做了进一步配对审计。`2,504` 个 query 中有
`1,699 (67.9%)` 实际换入至少一个 neighbor，`1,195 (47.7%)` 至少一个 mechanism group 的 top-1
发生变化；跨 query 的平均 replacement 数为 `2.15`，实际变化 query 中为 `3.17`。selector 的确提高了
Morgan query-feature coverage（六条件平均增幅范围 `+0.0357` 至 `+0.0632`）并降低 neighbor-neighbor
Tanimoto（`-0.0977` 至 `-0.1344`），所以负结果不是 selector 没有改变 retrieval。真正发生 set replacement
的样本中 Morgan-only correct / coverage-only correct 为 `85 / 51`，损失集中在换入 evidence 的下游效用。

Trace audit 还发现 `543` 个 query 的 ranked retrieval 完全相同，其中 `27` 个 final prediction 仍发生
flip；这 `27` 个 query 的 final LLM 输入逐字相同，因此属于模型调用波动，不能归因于 selector。当前 group
prompt 会逐 neighbor 提供 whole-molecule Tanimoto、MCS/MMP、property delta 和 assay evidence，但不会把
selector 的 marginal Morgan-bit coverage、query atom/region mapping 或 neighbor 独有覆盖区域传给 LLM。
因此这轮实验验证的是“coverage-selected analog set + 既有 whole-molecule transferability reasoning”，
不能解释为已经完整检验 fragment-wise compositional reasoning。

### 7.1 Coverage-aware reasoning context pilot（2026-08-03 完成）

为直接检验上述输入缺口，新增了与 selector 正交的 `neighbor_context_profile`。`standard` 保持原 prompt；
`coverage_aware` 在不暴露结构/身份的前提下，向每个 mechanism group 增加 query Morgan feature coverage、
由这些 feature 映射得到的 query atom-environment coverage、每个 neighbor 的 marginal/redundant contribution、
累计 coverage 和 marginal connected-region sizes。Prompt 明确要求模型只在 analog 可迁移时组合互补 evidence，
并明确 coverage 不是 fragment causality 或 label vote。

当前 pilot 固定为 Starling scaffold-valid、GPT-OSS-120B、`identity_blind + parent_disjoint`、Starling
`full_mechanism`、`top_k=3`、`min_similarity=0.30` 和 `query_feature_coverage` selector；两边复用同一批冻结的
120B single-molecule analyses，唯一变化是 `standard` vs `coverage_aware` context。控制组和候选组分别写入：

```text
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid_gpt_oss_120b_coverage_standard/
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid_gpt_oss_120b_coverage_aware/
```

完整 valid run 两边均为 `954/954` 成功、`n_failed_runs=0`。两边 `954/954` raw retrieval SHA-256
逐 query 相同；threshold violation、非 structural-analog neighbor、query-parent conflict、held-out parent overlap
和 identity-blind prompt leak 均为 `0`。Standard trace 中没有 coverage message；coverage-aware 的
`3,002/3,002` 个有 neighbor 的 group branch 恰好各收到一条 coverage context，其中 `2,998/3,002`
在输出中显式讨论 coverage、complementarity 或 redundancy。由此可确认新输入确实改变了 reasoning pattern，
而不是再次出现“selector 换了 neighbor，但模型不知道为什么”的旧条件。

| task | Standard accuracy | Aware accuracy | Δ accuracy | Standard macro-F1 | Aware macro-F1 | Δ macro-F1 | flips | McNemar p | macro-F1 Δ 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BBB | 0.7040 | 0.7200 | +0.0160 | 0.6796 | 0.6939 | +0.0143 | 54 | 0.3409 | [-0.0140, +0.0428] |
| Bioavailability | 0.6220 | 0.6220 | +0.0000 | 0.6129 | 0.6151 | +0.0022 | 28 | 1.0000 | [-0.0445, +0.0499] |
| Skin Reaction | 0.6000 | 0.5796 | -0.0204 | 0.5633 | 0.5348 | -0.0285 | 35 | 0.4996 | [-0.0809, +0.0220] |

结论是 mixed / no-go for promotion：BBB 有小幅正向 point estimate，Bioavailability 基本不变，Skin 反而下降；
三项 McNemar 均不显著，三个 macro-F1 bootstrap interval 都跨零。因此 coverage-aware context 解决了“LLM
没有使用 coverage 信息”的机制问题，但没有带来跨 task 稳定性能提升，不能替代 standard context，也不据此
进入 Starling test。它保留为 opt-in 插件，后续若继续，应先在 valid 冻结更有针对性的 region representation 或
task-specific evidence transfer gate，再做独立确认。

与同一 GPT-OSS-120B scaffold-valid 的原始 `Morgan selector + standard context` 主线相比，总图新增的
coverage experiment rows 显示：
BBB macro-F1 为 `0.6583 → 0.6796 → 0.6939`，Bioavailability 为
`0.5791 → 0.6129 → 0.6151`，Skin 为 `0.5700 → 0.5633 → 0.5348`（依次为 Morgan-standard、
coverage-standard、coverage-aware）。Morgan-standard vs coverage-aware 的 BBB 增幅为 `+0.0356`，McNemar
`p=0.0265`，macro-F1 bootstrap 95% CI `[+0.0017, +0.0698]`；但 Bioavailability interval 跨零，Skin point
estimate 为负。因此新 prompt 在 BBB 上使 coverage 路线显著超过 Morgan，但不能把这个 task-specific signal
解释成通用方法胜出；核心结论仍是跨 task 不稳定。

代价方面，三 task 合计 group+final total tokens 从 `35,248,740` 增至 `37,667,006`（`+6.86%`）；
同机并发运行下累计 wall latency proxy 增加 `+2.95%`。因此当前 mixed 性能还伴随确定的上下文和生成成本。

新增分析产物：

```text
outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/metrics.tsv
outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/comparison.json
outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/report.md
outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/contract_audit.json
outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/contract_audit.md
outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/three_way_metrics.tsv
outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/morgan_vs_coverage_standard/
outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/morgan_vs_coverage_aware/
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b_vs_120b/figures/starling_model_comparison.svg
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b_vs_120b/figures/starling_model_comparison_highres.png
```

### 7.2 Visible MMP-ledger context ablation（2026-08-03 完成）

为让 visible group reasoning 明确看到每个 retrieved neighbor 对应的结构重合信息，新增 opt-in
`coverage_mmp_ledger` profile。该 profile 保持 `query_feature_coverage` selector、raw retrieval、neighbor set、
single-molecule analysis 和 final schema 不变；对每个 selected analog 复用常驻
`mmp_structure_compare` 的 MCS/MMP 文本，并用 Morgan marginal feature 统计组织 rank-by-rank 的互补、冗余
和未覆盖区域。Morgan feature coverage 不解释为 atom coverage；没有 matched-pair transformation 时具体
fragment correspondence 必须标为 unresolved。

当前 matched valid 实验固定为 Starling scaffold、GPT-OSS-120B、`deployment_visible + parent_disjoint`、
Starling `full_mechanism`、`top_k=3`、`min_similarity=0.30` 和 `query_feature_coverage` selector。Standard 与
MMP-ledger 两边分别完成 BBB `500/500`、Bioavailability `209/209`、Skin `245/245`，均为
`n_failed_runs=0`；两边复用相同 frozen single analyses，smoke 中 raw retrieval 逐字节一致。

| task | Standard macro-F1 | MMP ledger macro-F1 | Δ macro-F1 | flips | McNemar p | macro-F1 Δ 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| BBB | 0.6843 | 0.6829 | -0.0014 | 66 | 1.0000 | [-0.0346, +0.0311] |
| Bioavailability | 0.6216 | 0.6511 | +0.0295 | 36 | 0.4050 | [-0.0238, +0.0848] |
| Skin Reaction | 0.5642 | 0.5610 | -0.0032 | 33 | 1.0000 | [-0.0575, +0.0517] |

结果仍为 mixed / no-go for promotion：BBB 和 Skin 基本持平，Bioavailability 有正向 point estimate，但三项
paired-bootstrap interval 均跨零，McNemar 也不显著。因此当前证据不支持将 MMP-ledger 升级为默认 prompt；
它继续作为 visible-only 可插拔 ablation 保留。该结果只在 canonical
`starling_model_comparison.{svg,png}` 总图中追加两行 visible experiment，不生成独立 performance figure。

配对统计与总图输入：

```text
outputs/paper/coverage_mmp_ledger_gpt_oss_120b_scaffold_valid/analysis/metrics.tsv
outputs/paper/coverage_mmp_ledger_gpt_oss_120b_scaffold_valid/analysis/comparison.json
outputs/paper/coverage_mmp_ledger_gpt_oss_120b_scaffold_valid/analysis/report.md
outputs/paper/coverage_mmp_ledger_gpt_oss_120b_scaffold_valid/analysis/figure_metrics.tsv
```

入口与产物：

```text
tools/chembl_tool/paper_experiments/summarize_coverage_selector_llm_matrix.py
tools/chembl_tool/paper_experiments/analyze_coverage_selector_retrieval_changes.py
tools/chembl_tool/paper_experiments/plot_coverage_selector_llm_matrix.py
outputs/paper/coverage_selector_llm/analysis/metrics.tsv
outputs/paper/coverage_selector_llm/analysis/comparison.json
outputs/paper/coverage_selector_llm/analysis/report.md
outputs/paper/coverage_selector_llm/analysis/retrieval_change_analysis.json
outputs/paper/coverage_selector_llm/analysis/retrieval_change_analysis.tsv
outputs/paper/coverage_selector_llm/analysis/retrieval_change_report.md
outputs/paper/coverage_selector_llm/analysis/figures/coverage_selector_llm_matrix.svg
outputs/paper/coverage_selector_llm/analysis/figures/coverage_selector_llm_matrix_highres.png
```

## 8. Identity-blind 补充控制：当前实测状态

Blind 条件隐藏 query/neighbor 的结构、名称和 source ID，并由 harness 预先提供脱敏后的 properties /
comparison tool evidence。它与 deployment-visible 同时改变 identity visibility 和 tool execution，
所以不是“纯 identity effect”；正式 visibility attribution 仍需 matched-prefetch replay。

统一入口：

```text
tools/chembl_tool/paper_experiments/starling_benchmark_matrix.py
  --benchmark-split random|scaffold
  --visibility-mode identity_blind
  --neighbor-identity-policy operational
```

两套 split 各已有 22 个 condition metrics，但当前还没有达到正式报告 gate：

```text
random:
  22/22 condition artifacts present
  9 failed sample-condition runs，全部在 Bioavailability

scaffold:
  22/22 condition artifacts present
  5 failed sample-condition runs：
    BBB ChEMBL flat 2
    Bioavailability Starling direct-full / flat / mechanism 各 1

BBB random、Skin random/scaffold 以及其余列出的条件为 0 failure。
```

Blind 当前 point estimates 可用于进度诊断，但在 repair 到 `n_failed_runs=0`、完成 query-SMILES leak 和
visibility-contract audit 前，不进入 formal Starling bar chart。完整逐 condition metrics 位于：

```text
outputs/paper/molecular_evidence_agent_starling_random/runs/
outputs/paper/molecular_evidence_agent_starling_scaffold/runs/
```

## 9. Skin_Reaction：label scope、retrieval distance 与性能下降

Skin gold label 是 sensitization/contact allergy，不是所有 adverse skin effects 的并集。Paper-facing
Starling mechanism families 是 evidence-scope expansion，而不是像 oral bioavailability
`F = Fa × Fg × Fh` 那样的层层因果分解：

```text
Tier 1: direct sensitization/contact-allergy anchors，最接近 gold
Tier 2: sensitization AOP key events，仍与 gold 对齐但不是 clinical outcome
Tier 3: phototoxicity、irritation、corrosion、local damage；相关但属于不同 skin hazard
Tier 4: permeability/retention/exposure context；不能单独证明 sensitization
```

Observed Starling macro-F1：

| split | direct | full flat | full mechanism |
|---|---:|---:|---:|
| random | 0.6431 | 0.6353 | 0.6300 |
| scaffold | 0.5973 | 0.5921 | 0.5836 |

Full-flat 与 full-mechanism 在两种 source、两种 split 上逐 query 的 LLM-visible evidence-row multiset
完全一致（各 380/380 matches，0 mismatch）。因此 flat-to-mechanism 变化来自 evidence organization、
独立 branch reasoning 和 final synthesis，不是 mechanism 看到了更多 rows。

Starling direct-to-mechanism flips：

```text
random:
  38 flips；17 corrected，21 broken；net -4 correct

scaffold:
  29 flips；12 corrected，17 broken；net -5 correct
```

Trace 中反复出现：

- phototoxicity/irritation/local-damage evidence 被提升成 sensitization risk；
- Tier 4 skin exposure support 被误用为 hazard 支持；
- weak AOP signal 或 distant analog narrative 被 branch packaging 放大；
- broad mixed negatives 稀释一个较近的 positive anchor；
- 没有 neighbor 时仍可因 prompt/organization 边界发生 prediction instability。

Token burden 同时明显增加。保存的 successful-response logical token mean：

| split | direct | full flat | full mechanism |
|---|---:|---:|---:|
| random | 27.7k | 77.4k | 96.1k |
| scaffold | 26.9k | 73.4k | 91.6k |

这说明更多 tokens 主要来自更多 evidence rows、group calls 和重复 synthesis，不等于更多
label-aligned information。完整量化与逐 flip artifacts：

```text
outputs/paper/skin_reaction_retrieval_diagnostic/agent_quant_summary.json
outputs/paper/skin_reaction_retrieval_diagnostic/agent_random_flips.jsonl
outputs/paper/skin_reaction_retrieval_diagnostic/agent_scaffold_flips.jsonl
outputs/paper/skin_reaction_retrieval_diagnostic/report_flip_summary.csv
outputs/paper/skin_reaction_retrieval_diagnostic/report_trace_examples.csv
```

### Tier 1+2 final-only post-hoc

为隔离 Tier 3/4，复用原 full-mechanism 的 frozen single、Tier 1、Tier 2 branch，删除 Tier 3/4 后只重跑
final。两套均为 380/380、0 failure，760 个 run 的 single/group artifact reuse 和 scope audit 为 0 mismatch：

| split | Tier 1+2 macro-F1 | 相对 full mechanism | 相对 direct |
|---|---:|---:|---:|
| random | 0.6310 | +0.0010 | -0.0121 |
| scaffold | 0.5948 | +0.0112 | -0.0025 |

两套 bootstrap 95% CI 均跨 0。裁掉 Tier 3/4 在 scaffold 有小幅 point-estimate recovery，但没有超过
direct，也没有证明稳定改善；Tier 2 analog noise、branch synthesis 和生成不稳定性仍可能贡献误差。

真正的 final-only scope 过滤入口是：

```text
--final-only-source-batch <source batch>
--final-only-groups Mechanism.tier_1 Mechanism.tier_2
```

普通 `--groups` 只控制 fresh pipeline，不能用于 resume-final artifact filtering。

## 10. 主要运行与汇总入口

```text
data builder:
  python -m tools.chembl_tool.common.starling.build_benchmark_datasets

held-out index:
  python -m tools.chembl_tool.paper_experiments.build_starling_benchmark_indices

formal/blind matrix:
  python -m tools.chembl_tool.paper_experiments.starling_benchmark_matrix

formal summary:
  python -m tools.chembl_tool.paper_experiments.summarize_starling_benchmark

formal figure:
  python -m tools.chembl_tool.paper_experiments.plot_starling_benchmark_overview

MiniMol:
  python -m baselines.minimol.run_bioavailability_ma \
    --data-dir data/processed_starling/<Task>/<random|scaffold> \
    --output-dir <model-and-split-specific-output> \
    --train-all --evaluation-split valid

Morgan KNN:
  python -m baselines.structure_knn.run \
    --data-dir data/processed_starling/<Task>/<random|scaffold> \
    --output-dir <model-and-split-specific-output> \
    --k 3 --evaluation-split valid

MiniMol embedding KNN:
  python -m baselines.minimol.run_embedding_knn \
    --data-dir data/processed_starling/<Task>/<random|scaffold> \
    --embedding-cache-dir <MiniMol-output>/embeddings \
    --output-dir <model-and-split-specific-output> \
    --k 3 --evaluation-split valid

MiniMol embedding agent retrieval:
  python -m tools.chembl_tool.paper_experiments.run_minimol_retrieval_agent_experiment
  python -m tools.chembl_tool.paper_experiments.plot_starling_with_minimol_agent

generic final-only group filtering:
  tools/chembl_tool/common/task_workflows/reasoning_batch.py
```

### 2026-08-02 至 2026-08-04 入口与 artifact contract 更新

- `starling_benchmark_matrix.py` 提供显式 `--output-root`、`--evaluation-subset`、`--timeout-s`、
  `--neighbor-selector`、`--neighbor-context-profile` 和 `--single-analysis-root`。旧 condition lane/phase
  scheduler 已移除；single/group/final branch 统一进入全局 ready queue，frozen single 按 sample 解锁。
- `molecular_evidence_agent.py` 和四个 task runner 透传 selector/context/timeout；fresh
  `deployment_visible + parent_disjoint` 允许写入独立 ablation root，不再要求 operational reuse plan。
- `coverage_reasoning.py` 提供 `standard`、blind-safe `coverage_aware` 和 visible-only
  `coverage_mmp_ledger` 三个 profile；group reuse 现在要求 source/target profile 一致。
- `summarize_starling_benchmark.py` 支持 model-specific pipeline root、valid/test subset、visibility 和
  failure-inclusive policy。显式 v4/valid summary 不再隐式加载 historical test baselines；baseline roots 必须
  明确传入且与同一 lineage/subset 匹配。`summarize_results.py` 仍只用于旧 TDC visibility matrix，不能对
  `runs_identity_blind_parent_disjoint/` 生成 v4 审计；此前生成的空
  `analysis_identity_blind_parent_disjoint/report.md` 不是 canonical 结果。
- 三个 baseline CLI 新增 `--evaluation-split valid|test`。MiniMol 的 valid 评估只允许与 `--train-all`
  配对，并可通过重复 `--reuse-embedding-cache-dir` 按 exact SMILES 复用 molecule-only embeddings；manifest
  不复用 label 字段。KNN/MiniMol metrics 同时保存 `evaluation_split` 和 `n_evaluation`，旧 `n_test` 仅为兼容。
- `plot_starling_model_comparison.py` 是 model、visibility、baseline 和 matched method rows 的唯一总图入口；
  comparison summary 必须 task/split/subset/n/baseline 对齐，experiment TSV 必须提供不会重复绘制的 anchor。
- Resident tool service 新增 `/tools/batch`、bounded workers、process-resident MolGpKa、native-thread cap、
  SQLite/WAL + LRU/single-flight cache；部署与滚动切换规范见 `tools/service/README.md`。
- `watch_glm_tunnel_and_matrix.py` 监控 `/v1/models`、SSH tunnel 和唯一 resumable matrix。完成计数与 batch
  gate 对齐：必须有可归一化的 task prediction、single/final status=ok、准确的 expected group 数和 0 failed
  groups；它不保存或重放 SSH 密码，Duo 仍需用户批准。

### GLM endpoint 默认值与 2026-08-01 benchmark

上述 paper/Starling GLM 入口现在默认使用：

```text
http://127.0.0.1:50000/v1
nvidia/GLM-5.2-NVFP4
reasoning_effort=""（省略 API 参数，保持历史 GLM reasoning contract）
```

本机端口由 `ssh -fNT parcc-glm` 转发到 `dgx008:50000`。旧 LiteLLM 请求名
`zai-org/GLM-5.2-FP8` 实际也解析到 `hosted_vllm/nvidia/GLM-5.2-NVFP4`，因此这里不是两种模型精度的比较。

同一约 3.3k input-token structured-output 请求的 64 并发 smoke：

| runtime contract | valid JSON | wall time | p50 latency | aggregate token/s |
|---|---:|---:|---:|---:|
| old LiteLLM + historical default thinking | 0 / 64 | 16.41 s | 7.99 s | 14,767 |
| direct dgx008 + `reasoning_effort=none`（诊断，不采用） | 64 / 64 | 6.68 s | 3.43 s | 32,221 |

上表的 `2.46x` wall-time、`2.33x` p50 和 `2.18x` aggregate throughput 改善来自关闭 reasoning，
不属于正式 agent 设置，不能作为新默认相对旧端点的速度结论。正式 runner 保持历史
`--disable-thinking --reasoning-effort ""`：不发送 reasoning-effort 参数，但继续接收并保存 GLM reasoning。

为找 endpoint 吞吐上限，另用约 2.1k token/request、短 JSON 输出和共享 prompt 前缀测试直连端点：

| concurrency | valid JSON | wall time | aggregate token/s |
|---:|---:|---:|---:|
| 128 | 128 / 128 | 4.73 s | 57,128 |
| 256 | 256 / 256 | 6.16 s | 87,883 |
| 512 | 512 / 512 | 8.80 s | 122,927 |

这组最高约 `122.9k token/s` 同样使用 `reasoning_effort=none`，只保留为关闭 reasoning 的 endpoint
ceiling 诊断；它不是当前 reasoning-enabled 默认，也不能换算真实 agent pipeline 完成时间。

### Task-local KNN–agent router v2（2026-08-05，scaffold valid）

`router_oof/` 使用各 task 的完整 scaffold train set 构造 5-fold OOF 训练数据，并保持 Morgan `k=3`、
GPT-OSS-120B `identity_blind + parent_disjoint` direct agent 的冻结合同。v2 删除了随 fold reference-pool 大小
漂移的特征以及 k=3 下的确定性冗余特征，分别估计 `P(agent_only_correct)` 与 `P(knn_only_correct)`；只有
train-only nested paired-bootstrap promotion gate 通过时才允许从 KNN 切换到 agent，否则部署为严格 KNN
fallback。v1 artifact 保留，v2 写入独立 `router_v2/` lineage。

BBB OOF agent 的最后一个缺失样本也已补齐，三个 task 均通过 zero-failure/paired-completeness gate：
BBB `18,425/18,425`、Bioavailability `1,674/1,674`、Skin Reaction `1,966/1,966`。冻结 v2 后一次性得到的
valid 结果为：

| task | valid n | promotion gate | KNN acc / macro-F1 | direct agent acc / macro-F1 | deployed router acc / macro-F1 | router - KNN | switch / rescue / harm |
|---|---:|---|---:|---:|---:|---:|---:|
| BBB | 500 | PASS | 0.7740 / 0.7127 | 0.7300 / 0.7138 | **0.7900 / 0.7330** | +0.0160 / +0.0203 | 42 / 25 / 17 |
| Bioavailability | 209 | PASS | 0.7177 / 0.6199 | 0.6507 / 0.6434 | **0.7177 / 0.6339** | +0.0000 / +0.0139 | 24 / 12 / 12 |
| Skin Reaction | 245 | FAIL, KNN fallback | **0.6857 / 0.6064** | 0.5633 / 0.5487 | **0.6857 / 0.6064** | +0.0000 / +0.0000 | 0 / 0 / 0 |

BBB 的 point estimate 同时超过 KNN 与 direct agent；但其 10,000-repeat label-stratified paired bootstrap
accuracy delta 95% CI 为 `[-0.0100, 0.0420]`，macro-F1 delta CI 为 `[-0.0096, 0.0506]`，均跨 0，现阶段
只能解释为 promising signal。Bioavailability 在 accuracy 不变时 macro-F1 增加 1.39 pp，但 CI 也跨 0。
Skin 的 ungated candidate 在 valid 上实际为 0.6531 / 0.5768（12 rescues、20 harms）；promotion gate 将其
挡住，因此 deployed policy 没有重复 v1 的退化。这说明 v2 已解决“坏 router 必须安全退回 KNN”的工程问题，
尚未证明三 task 都能显著优于 KNN。

Canonical receipt：
`outputs/paper/router_oof/gpt_oss_120b/scaffold/valid_evaluation_result_v2.json`。每 task 的 model、feature、
prediction、report 和输入 SHA-256 位于同一 root 下的 `<task>/router_v2/`。这些 valid 结果不得用于回写
feature、threshold 或 promotion policy；formal test 尚未运行。

### Output-aware post-selector v3（2026-08-05，scaffold valid development）

v3 不覆盖 v2，而是在 KNN 与 GPT-OSS-120B direct agent 已产生不同 label 后，只对 disagreement rows 学习
`agent_only_correct` 对 `knn_only_correct`。它保留原始 query、KNN 和 evidence feature，并比较三档输入：
18-feature output/query/KNN、49-feature evidence、70-feature evidence+structured-trace；每档同时比较 Logistic
与小型 HistGBDT。`KNN=0, agent=1` 与 `KNN=1, agent=0` 使用独立 train-only thresholds。Nested OOF promotion
gate 未通过时部署严格回退 KNN，不能在看到 valid 后解锁。

| task | train gate / frozen profile | KNN acc / macro-F1 | candidate acc / macro-F1 | deployed acc / macro-F1 | candidate switch / rescue / harm | oracle acc |
|---|---|---:|---:|---:|---:|---:|
| BBB | PASS / HistGBDT + evidence | 0.7740 / 0.7127 | **0.7760 / 0.7382** | **0.7760 / 0.7382** | 71 / 36 / 35 | 0.9120 |
| Bioavailability | FAIL / HistGBDT + evidence+trace | 0.7177 / 0.6199 | 0.7177 / **0.6499** | 0.7177 / 0.6199 | 28 / 14 / 14 | 0.9091 |
| Skin Reaction | FAIL / HistGBDT + evidence | **0.6857** / 0.6064 | 0.6694 / 0.6123 | **0.6857** / 0.6064 | 42 / 19 / 23 | 0.8122 |

结果支持“原始 evidence features 有用但还不够”的判断：三个 task 的 full-train deployment choice 都包含
evidence，Bioavailability 还选择了 structured trace；BBB nested outer folds 也全部选择 evidence 或
evidence+trace profile。不过 BBB valid 的 71 次切换只有净 1 次 rescue，accuracy 仅比 KNN 高 0.2 pp；其
macro-F1 delta 为 +2.55 pp，但 10,000-repeat paired-bootstrap 95% CI `[-0.0114, 0.0653]` 跨 0。
Bioavailability candidate 的 14 rescues/14 harms 保持 accuracy、改善类别平衡，但 train accuracy uncertainty
未通过预注册 guardrail；Skin candidate harms 多于 rescues且 accuracy 下降。因此 v3 仍未接近 oracle，瓶颈
不是 feature 是否全部保留，而是现有 trace/evidence summary 对“这一次 disagreement 谁正确”的辨别力仍弱。

Canonical receipt：
`outputs/paper/router_oof/gpt_oss_120b/scaffold/post_selector_valid_result_v3.json`。每 task 的 model、nested OOF
prediction、metrics、manifest、valid feature 和 report 位于 `<task>/post_selector_v3/`。这是经过既有 valid
观察后设计的 development experiment，只能用于方法迭代；formal test 尚未运行，也没有用 valid 调整 v3
模型、profile 或 thresholds。

### Direction-calibrated post-selector v3.1（2026-08-05，scaffold valid development）

v3.1 是在观察 v3 valid 后冻结的独立 development lineage，不覆盖 v3，也不新增 LLM 请求。两个 disagreement
direction 分别选择 Logistic/HistGBDT、feature profile 与 threshold；每个模型使用 fold-heldout sigmoid
calibration 的五组件 ensemble。一个 direction 只有在 train 上至少 route 20 rows 且 agent-win precision 的
one-sided 95% Wilson lower bound `> 0.5` 时才允许启用；overall train promotion 还要求 paired-bootstrap accuracy
delta 的 95% CI lower bound `> 0`。Valid 不拟合、不校准、不调 threshold。

| task | train gate | KNN acc / macro-F1 | v3.1 acc / macro-F1 | delta | switch / rescue / harm | valid accuracy delta 95% CI | held-out evidence gate |
|---|---|---:|---:|---:|---:|---:|---|
| BBB | PASS | 0.7740 / 0.7127 | **0.8060 / 0.7473** | **+0.0320 / +0.0346** | 26 / 21 / 5 | **[+0.0140, +0.0520]** | **PASS** |
| Bioavailability | PASS | 0.7177 / 0.6199 | 0.7225 / 0.6020 | +0.0048 / -0.0180 | 9 / 5 / 4 | [-0.0239, +0.0335] | FAIL |
| Skin Reaction | PASS | 0.6857 / 0.6064 | 0.6980 / 0.5962 | +0.0122 / -0.0102 | 13 / 8 / 5 | [-0.0163, +0.0408] | FAIL |

BBB 是目前唯一同时满足 train promotion、held-out accuracy evidence gate，并在 valid 上同时提高 accuracy 与
macro-F1 的 task。其两个方向都发生了少量选择性切换：`KNN=0, agent=1` 为 14/18 正确，`KNN=1, agent=0`
为 7/8 正确。Bioavailability 与 Skin 的 point-estimate accuracy 小幅上升，但 CI 跨 0 且 macro-F1 分别下降
1.80 pp 和 1.02 pp，因此不能称为稳定改善，也不能因为 valid 结果去调整 frozen policy。Skin 的复杂 selector
在 train 上也没有超过简单的 `route KNN=0/agent=1 only` baseline，提示该 task 当前主要利用输出方向 prior，
而不是学到可迁移的 trace/evidence 条件边界。

Canonical receipt：
`outputs/paper/router_oof/gpt_oss_120b/scaffold/post_selector_valid_result_v31.json`。每 task 的 calibrated ensemble、
manifest、nested predictions、valid features/predictions/metrics 位于 `<task>/post_selector_v31/`。这是 valid-informed
development experiment；formal test 未运行，不能把 BBB 结果当成最终 test claim。

### Router termination diagnostics（2026-08-05，train-only）

为区分 data-limited 与 signal-limited，后续诊断不再读取 valid/test，也不重新搜索 v3.1 family/profile。首先将
BBB outer-train disagreement supervision 按 direction/fold/target 分层下采样，固定 full-train direction specs，
主比较改为 learned selector 相对不读 feature 的 direction-only OR rule：

| full-equivalent disagreement budget | seeds | realized outer-train n | accuracy delta vs OR | macro-F1 delta vs OR | positive-seed fraction (acc / F1) |
|---:|---:|---:|---:|---:|---:|
| 800 | 5 | 640.0 | -0.10 pp | +0.50 pp | 0.00 / 1.00 |
| 1,600 | 5 | 1,282.4 | +0.05 pp | +0.91 pp | 0.80 / 1.00 |
| 3,200 | 5 | 2,559.2 | +0.49 pp | +2.33 pp | 1.00 / 1.00 |
| 7,130 | 1 deterministic | 5,704.0 | **+0.74 pp** | **+3.37 pp** | 1.00 / 1.00 |

Full-size train OOF 的 learned-vs-OR paired 95% CI 为 accuracy `[+0.45,+1.02] pp`、macro-F1
`[+2.90,+3.82] pp`。因此 BBB 确实存在随训练量上升的 task-local routing signal；在约 800 disagreements 时，
复杂 selector 还不能提高 accuracy。这解释了 Oral/Skin 为什么更难，但不能覆盖既有 scaffold-valid 结论：BBB
valid 上 learned selector 相对 OR 的 accuracy 增量仍为 0，macro-F1 `+0.95 pp` 且 CI 跨 0，说明跨 scaffold
transfer/calibration 仍是独立瓶颈。

条件触发的共享表示实验固定为两个 direction 的 task-balanced Logistic、完整 68-feature generic profile、
task one-hot，以及 task-specific sigmoid calibration/threshold/Wilson gate。每个 target outer/inner heldout fold
还从其它 task training rows 排除了相同 molecule identity 和 fold-group/scaffold：

| task | shared acc / macro-F1 | shared - direction-only OR | shared - frozen task-local | shared-vs-local macro-F1 95% CI |
|---|---:|---:|---:|---:|
| BBB | 0.7607 / 0.6510 | +0.04 / +0.53 pp | **-0.69 / -2.84 pp** | [-3.29,-2.41] pp |
| Bioavailability | 0.7437 / 0.6239 | -0.30 / -0.18 pp | -0.42 / **-1.01 pp** | [-1.81,-0.32] pp |
| Skin Reaction | 0.6846 / 0.5758 | -0.05 / +0.38 pp | +0.05 / +0.22 pp | [-0.39,+0.89] pp |

共享表示没有让任一小 task 相对 task-local 得到可信提升，且显著伤害 Bioavailability macro-F1；transfer
continuation gate 因此为 **FAIL / `stop_router_main_method`**。结论是：更多 task-local data 能帮助 BBB，但现有
query/KNN/evidence/trace representation 的可迁移性不足。这里的 shared model 监督仍是最终 agent-win label，
不是 counterfactual evidence utility。Router 保留为 reliability baseline 和 negative diagnosis，不再继续做
valid-informed family/profile sweep，也不消耗 formal test。下一条方法线应改变监督信号本身，例如
counterfactual evidence add/drop utility，而不是继续调 post-selector。

Canonical receipts：
`outputs/paper/router_oof/gpt_oss_120b/scaffold/post_selector_v31_learning_curve_result.json` 与
`outputs/paper/router_oof/gpt_oss_120b/scaffold/post_selector_v31_transfer_result.json`；task-local job metrics、
compressed predictions、cross-task fold exclusions 和 thresholds 保存在相应 task artifact directory。

## 11. 当前未完成项

- `record_supported_v2` GLM blind 首轮为 5,876/6,887，尚有 1,011 个 endpoint/tunnel 失败待
  `--skip-existing` 修复；GLM visible 尚未启动，二者均未进入正式结果表；
- `record_supported_v2` 的 paired direct-agent significance 尚未在完整 GLM 矩阵上生成；任何 partial
  GLM p-value 都无效；
- GLM random-valid blind 仍有 Bioavailability 两个 ChEMBL full 条件各 1 个失败，尚未通过 zero-failure gate；
- v4 random-valid 的 matched train-label baselines 尚未生成；
- v4 formal test 尚未启动，valid setting 冻结后需按同合同 fresh-run；
- test matched-prefetch 尚未扩展到全部当前 Starling conditions；
- ECFP RF/XGBoost 和 matched-neighbor evidence retrieval-only vote 未完成；
- 用于确认表示选择稳健性的独立第二种 pretrained encoder baseline 未完成；
- source-quality 双人 annotation 未完成；
- GLM/GPT-OSS 当前每项主要只有一次 run，关键 comparisons 仍需独立 repeats；
- task-local router v2、output-aware post-selector v3 和 direction-calibrated v3.1 已完成 scaffold-valid
  development；matched-size curve 证明 BBB signal 随 data 增强，但 shared transfer gate 失败，router 主方法线
  已停止，formal test 按 gate 决定不启动（不是待补运行）；
- Skin Tier 1+2 是 test-triggered post-hoc diagnosis，不得升级成预注册 primary condition。

## 12. Git 发布里程碑

已直接推送到 `origin/main` 的 Starling migration：

```text
4595fde  Add Starling benchmark data splits
8c16ae6  Add Starling benchmark evaluation pipeline
```

`8c16ae6` 包含 random/scaffold matrix、held-out provenance、MiniMol train-all、Morgan KNN、bar chart
及相关入口。本轮 scoped publish 在其上补充 2026-07-27 的 Skin Tier 1+2 final-only
filtering/results、集中结果总账，以及 2026-07-28 的 MiniMol embedding cosine KNN、统一汇总和
canonical bar chart；同时存在的 viewer/coverage-selector 独立改动不属于该 publish scope。

2026-07-30 的 scoped publish 进一步加入 MiniMol/cosine operational agent retrieval 的 38-condition
零失败 gate、与已有结果同图比较的独立 SVG/PNG/TSV、绘图入口和本节结果记录；viewer 与
coverage-selector 独立改动仍不属于该 publish scope。
