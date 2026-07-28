# Starling random/scaffold benchmark：当前决策、结果与入口

更新时间：2026-07-28。

本文件是 2026-07-24 至 2026-07-27 Starling benchmark 迁移和实验的集中总账。它只记录当前
Starling-held-out `random` / `scaffold` lineage；旧 TDC `test` / `valid` 的历史结果仍见
`RESULTS.md`，不得混表或改称 Starling。

机器可读主结果和 canonical bar chart：

```text
outputs/paper/starling_benchmark_results/summary.json
outputs/paper/starling_benchmark_results/metrics.tsv
outputs/paper/starling_benchmark_results/report.md
outputs/paper/starling_benchmark_results/figures/starling_benchmark_overview.svg
outputs/paper/starling_benchmark_results/figures/starling_benchmark_overview_highres.png
```

## 1. 冻结的数据与 label 决策

公共协议和唯一数据构建入口：

```text
tools/chembl_tool/common/starling/STARLING_BENCHMARK_PROTOCOL.md
tools/chembl_tool/common/starling/benchmark_dataset.py
tools/chembl_tool/common/starling/build_benchmark_datasets.py
```

当前冻结规则：

- test size 为 `min(500, floor(0.2 * n_binary_molecules))`；
- `random` 是固定 seed 的 label-stratified stable-hash split；
- `scaffold` 以 canonical Bemis–Murcko scaffold 为不可拆分 group；
- 两套 split 从同一批 accepted binary parents 独立构建；
- 每套 train/test parent identity 为零重叠，scaffold 还要求 scaffold 零重叠；
- source row 先按 task adapter 独立转成 `0/1/ambiguous`，再按
  `rdkit_fragment_parent.v1` 聚合；
- 同一个 normalized parent 的 accepted rows 同时出现 0 和 1 时，整个 parent 排除，不做多数票；
- `within_record_label_conflict` 指同一条 row 内可用信号互相矛盾；这种 row 在 parent 聚合前拒绝。

冻结规模：

| task | binary parents | test target | random Y=0 / Y=1 | scaffold Y=0 / Y=1 |
|---|---:|---:|---:|---:|
| BBB_Martins | 17,893 | 500 | 139 / 361 | 122 / 378 |
| Bioavailability_Ma | 1,862 | 372 | 99 / 273 | 113 / 259 |
| Skin_Reaction | 1,900 | 380 | 129 / 251 | 117 / 263 |

Task label：

```text
BBB_Martins:
  Y=1 pass；Y=0 fail；明确 logBB 使用 >= -1 threshold。

Bioavailability_Ma:
  human oral F >= 20% -> Y=1；F < 20% -> Y=0。

Skin_Reaction:
  Y=1 skin sensitizer；Y=0 non-sensitizer。
  irritation、phototoxicity、generic local damage 和 skin exposure 不构成这个 gold label。

ClinTox:
  当前没有与 toxicity-caused clinical-trial failure 同定义的 Starling direct source，
  所以没有构造 Starling split。
```

具体 numerical/unit/free-text 映射以三个 task adapter 为代码真相，汇总规则见
`STARLING_BENCHMARK_PROTOCOL.md`：

```text
tools/chembl_tool/tasks/bbb_martins/starling_benchmark.py
tools/chembl_tool/tasks/bioavailability_ma/starling_benchmark.py
tools/chembl_tool/tasks/skin_reaction/starling_benchmark.py
```

## 2. Held-out retrieval 隔离

Gold split 与 retrieval evidence library 是两个步骤。正式 retrieval condition 分别使用对应 split 的
`test_molecule_labels.jsonl` 从 full Starling evidence 中删除全部 test parents，再重建 held-out index；
不能只靠 query-time exact-SMILES exclusion，也不能直接使用 full-source index。

入口：

```text
tools/chembl_tool/common/starling/heldout_index.py
tools/chembl_tool/paper_experiments/build_starling_benchmark_indices.py
```

当前 formal pipeline condition 使用 deployment-visible agentic 制度：

```text
none:
  operational；没有 retrieval。

retrieval conditions:
  parent_disjoint；retained neighbor 与 query parent identity 零重叠。
```

## 3. Formal performance

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
- Skin 的 Starling direct 到 full mechanism 在两套 split 都下降，见第 7 节；
- random 与 scaffold 是不同 test sets，不能把两者的绝对高低直接解释为方法对 scaffold 的因果效应。

## 4. MiniMol baseline 的选择口径

当前 Starling MiniMol 使用 `--train-all`：

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

## 6. Identity-blind 补充控制：当前实测状态

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

## 7. Skin_Reaction：label scope、retrieval distance 与性能下降

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

## 8. 主要运行与汇总入口

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
  python -m baselines.minimol.run_bioavailability_ma --train-all

Morgan KNN:
  python -m baselines.structure_knn.run --k 3

MiniMol embedding KNN:
  python -m baselines.minimol.run_embedding_knn --k 3

generic final-only group filtering:
  tools/chembl_tool/common/task_workflows/reasoning_batch.py
```

## 9. 当前未完成项

- repair identity-blind 的 14 个 failed sample-condition runs，并生成独立 blind analysis/audit；
- test matched-prefetch 尚未扩展到全部当前 Starling conditions；
- ECFP RF/XGBoost 和 matched-neighbor evidence retrieval-only vote 未完成；
- 用于确认表示选择稳健性的独立第二种 pretrained encoder baseline 未完成；
- source-quality 双人 annotation 未完成；
- 当前 GLM condition 每项主要只有一次 run，关键 comparisons 仍需 repeats/第二模型验证；
- Skin Tier 1+2 是 test-triggered post-hoc diagnosis，不得升级成预注册 primary condition。

## 10. Git 发布里程碑

已直接推送到 `origin/main` 的 Starling migration：

```text
4595fde  Add Starling benchmark data splits
8c16ae6  Add Starling benchmark evaluation pipeline
```

`8c16ae6` 包含 random/scaffold matrix、held-out provenance、MiniMol train-all、Morgan KNN、bar chart
及相关入口。本轮 scoped publish 在其上补充 2026-07-27 的 Skin Tier 1+2 final-only
filtering/results、集中结果总账，以及 2026-07-28 的 MiniMol embedding cosine KNN、统一汇总和
canonical bar chart；同时存在的 viewer/coverage-selector 独立改动不属于该 publish scope。
