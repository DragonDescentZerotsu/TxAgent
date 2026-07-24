# 论文实验操作约定

该目录负责维护冻结的、面向论文的分子证据 agent 实验矩阵。orchestration 必须保持通用。任务特定的 endpoint 映射只能放在 `tools/chembl_tool/tasks/<task>/experiment_config.py` 中。

## 文档职责

```text
ICLR_2027_EXECUTION_PLAN.md
  面向投稿的项目级执行总计划。记录贡献、实验编号、依赖、资源、缺口、验收标准、时间表和投稿 gate。

EXPERIMENT_PLAN.md
  当前冻结实验矩阵的研究问题、运行参数和统计协议。不得用项目管理状态覆盖协议内容。

VISIBILITY_ANALYSIS.md
  Identity-blind 与 deployment-visible 的集中结果解释，包括上升、下降、identity audit、metric caveat
  和尚未完成的因果验证。

RESULTS.md
  已实际测得的结果和统计结论。不能写入尚未运行的预期结果。

PIPELINE.md
  通用 pipeline、数据流、可见性制度和 task mechanism families。

DISTANCE_EXPANSION_DESIGN.md
  E12 的 D-root/C-family/H1/H2 tree 定义、flat 主曲线、tree-node mechanism supplementary、
  顺序 branch reuse、独立代码路径、资源控制、artifact contract 和回归 gate。

TRACE_RETENTION.md
  最终论文 trace 的唯一目录、保留单位、清理边界和 viewer 约束。
```

新增实验或得到新结果时：先在执行总计划更新对应 E 编号状态，再运行生成器更新机器可读统计和
`analysis/report.md`，然后根据生成产物把已测结果同步到 `RESULTS.md`，最后更新
`VISIBILITY_ANALYSIS.md` 等解释文档。当前生成器不会自动改写 `RESULTS.md`；不得只在聊天中保留结论。

## Benchmark input lineage 与 Starling 迁移边界

当前 frozen `test` / `valid` matrix、`RESULTS.md` 和
`outputs/paper/molecular_evidence_agent{,_valid}/` 均来自迁移前的 TDC task splits。它们是可复现的
历史结果，不应删除；但不得改称 Starling benchmark。现有 CLI 的 `--split test|valid` 仍是旧数据选择器，
不能重载为 Starling 的 `random|scaffold`。

新的 Starling direct gold benchmark 由以下公共入口构建：

```text
tools/chembl_tool/common/starling/STARLING_BENCHMARK_PROTOCOL.md
tools/chembl_tool/common/starling/build_benchmark_datasets.py

data/processed_starling/<Task>/random/test.jsonl
data/processed_starling/<Task>/scaffold/test.jsonl
```

将它接入 paper runner 时必须满足：

1. condition manifest 显式记录 `benchmark_source=starling`、task、`random|scaffold`、builder/source
   revision 和 test input hash；
2. random 与 scaffold 分别使用对应 `test_molecule_labels.jsonl` 构建的 train-only evidence index，
   不能复用从 full Starling direct source 构建的旧 index；
3. 在运行 LLM 前审计 test parent identity 与 index molecule identity 为零重叠；scaffold split 还需保留
   train/test scaffold 零重叠的 builder audit；
4. 两种 Starling split 使用不同 output root/batch ID，且都与旧 TDC test/valid root 隔离；
5. 汇总器和图表必须按 benchmark lineage 分区，不得把 TDC、Starling-random 和
   Starling-scaffold sample-condition 合并成一个指标。

当前仅完成 dataset builder 和 split artifacts；在上述 runner/index 接线与 leakage audit 完成前，
`RESULTS.md` 中不存在正式 Starling rerun 结果。

## 代码与命令入口

```text
molecular_evidence_agent.py
  冻结实验矩阵与统一运行入口；负责 visibility mode、neighbor identity policy、结果 root 和跨条件复用参数。

summarize_results.py
  汇总 identity-blind、matched-prefetch 和 agentic operational 条件，生成 coverage、token、visibility audit、
  paired bootstrap、McNemar/Holm 和 `analysis/report.md`。

analyze_coverage_performance.py
  只读取 deployment-visible 的 predictions，快速生成 coverage/class-conditional coverage、相对同任务
  none 的 paired macro-F1 增幅、10,000 次 bootstrap、机器可读 TSV/JSON 和报告段落。

audit_prefetch_contract.py
  逐 condition、逐样本验证 matched-prefetch 是否完整 replay identity-blind 的 retrieval 和工具输出。

parent_disjoint_ablation.py
  审计 operational retrieval 的 same-parent overlap，生成选择性重跑/整条复用/family-branch 复用计划；
  加 `--materialize` 后只物化输入未变化的整条 reuse artifact 和 reuse plan。输入变化的样本仍由统一 matrix
  runner 以 `--neighbor-identity-policy parent_disjoint` 执行。

summarize_parent_disjoint_results.py
  在指定 split 的完整样本上配对比较 operational 与 parent-disjoint，并审计 identity policy、threshold 和
  reuse provenance。当前入口通过显式 `--operational-root`、`--parent-disjoint-root`、`--output-dir`
  复用于 test/valid。

plot_retrieval_claims_overview.py
  从已生成的 analysis TSV 绘制统一横向 grouped-bar chart；它不重新计算指标。
  `--data-split {test,valid}` 控制 split 文案与结论，`--analysis-dir` / `--output` 用于分区产物。

plot_coverage_performance.py
  从 `coverage_performance.tsv` 绘制 overall/class-conditional coverage 与 paired macro-F1 增幅；
  只负责展示，不在图内重算指标。默认输出 deployment-visible agentic 的 canonical SVG。

paper_figure_style.py
  论文 SVG 共用的颜色、字体和基础绘图 primitive；新增 figure 不复制视觉常量。
```

常用审计顺序：

```bash
python -m tools.chembl_tool.paper_experiments.summarize_results
python -m tools.chembl_tool.paper_experiments.analyze_coverage_performance
python -m tools.chembl_tool.paper_experiments.audit_prefetch_contract
python -m tools.chembl_tool.paper_experiments.summarize_parent_disjoint_results
python -m tools.chembl_tool.paper_experiments.plot_retrieval_claims_overview
python -m tools.chembl_tool.paper_experiments.plot_coverage_performance
```

Valid 诊断重跑的对应顺序：

```bash
python -m tools.chembl_tool.paper_experiments.summarize_results --split valid
python -m tools.chembl_tool.paper_experiments.audit_prefetch_contract --split valid
python -m tools.chembl_tool.paper_experiments.summarize_parent_disjoint_results \
  --operational-root outputs/paper/molecular_evidence_agent_valid/runs_deployment_visible \
  --parent-disjoint-root outputs/paper/molecular_evidence_agent_valid/runs_deployment_visible_parent_disjoint \
  --output-dir outputs/paper/molecular_evidence_agent_valid/analysis/parent_disjoint_ablation
python -m tools.chembl_tool.paper_experiments.plot_retrieval_claims_overview \
  --analysis-dir outputs/paper/molecular_evidence_agent_valid/analysis \
  --output outputs/paper/molecular_evidence_agent_valid/analysis/figures/retrieval_claims_overview.svg \
  --png-output outputs/paper/molecular_evidence_agent_valid/analysis/figures/retrieval_claims_overview_highres.png \
  --data-split valid
```

对应回归测试集中在 `tests/chembl_tool/common/`：identity/policy、retrieval view、replay/hash reuse、prefetch
contract、paper matrix/result summary、parent-disjoint summary 和 viewer root 都必须随相关入口一起运行。

## 语言约定

- 内部分析、推理过程、代码标识符和机器可读字段可以使用英文。
- 所有面向用户展示的说明、实验结论、Markdown 文档、生成报告和进度汇报必须使用中文。
- 命令、路径、模型名、实验 ID、字段名和通行的技术术语可以保留英文，但正文解释必须使用中文。
- 修改报告生成器时，必须确保重新生成的用户可见报告仍为中文，不能只手工翻译生成产物。

论文主结果表使用真实部署导向的 agentic 制度：

```text
deployment_visible:
  query 只暴露结构，不暴露名称；
  retrieved molecule 暴露结构，并在数据源提供时暴露 source name/source ID；
  LLM 自主决定是否调用比较工具；
  作为 none/direct/full_flat/full_mechanism 和 source comparison 的论文主制度。

identity_blind 与 deployment_visible_prefetched:
  只作为 visibility/tool-execution 补充控制；
  不进入论文主结果表，不得代替 deployment_visible 的端到端 agentic 结果；
  既有 parity audit 和结果必须保留用于 provenance/appendix。
```

主实验和补充控制共享以下基础条件：

```text
model: zai-org/GLM-5.2-FP8
temperature: 0
max_tokens: 20480
top_k_per_group: 3
min_similarity: 0.30
operational_exact_record_exclusion: true
primary_analog_neighbor_identity_policy: parent_disjoint
operational_policy_role: mandatory staging and deployment-sensitivity reference
```

所有 paper-facing structural-analog retrieval 主结果默认使用 `parent_disjoint`：标准化 query/source parent、
排除相同 parent，再用仍高于原 similarity threshold 的后续候选补足 top-k。`operational` 仍必须先运行，
因为它代表真实部署中可能利用盐型、溶剂化物、重复组分和 formulation-linked record 的设置，也是
parent-disjoint 差异检测、整条 run 复用和 branch 复用的 staging/reference；但它不再作为 analog-retrieval
claim 的默认最终结果，只作为 deployment-sensitivity 对照单独报告。这里的 parent 是 RDKit FragmentParent
标准化结果，不是 active moiety，也不推断 prodrug 或代谢物关系。

新增任何 retrieval condition 时，完成 operational run 后必须在同一轮工作中生成 parent-disjoint 计划、
物化不变样本并补跑变化样本；不能等到最后统一补消融。统一 runner 的 CLI 默认已设为
`deployment_visible + parent_disjoint`，且会要求目标 batch 已有 `reuse_plan.json`；第一阶段 operational 必须
显式传 `--neighbor-identity-policy operational`，防止误把 staging 设置当成最终主实验。

Parent-disjoint 产物统一放在：

```text
# test
outputs/paper/molecular_evidence_agent/runs_deployment_visible_parent_disjoint/
outputs/paper/molecular_evidence_agent/analysis/parent_disjoint_ablation/

# valid
outputs/paper/molecular_evidence_agent_valid/runs_deployment_visible_parent_disjoint/
outputs/paper/molecular_evidence_agent_valid/analysis/parent_disjoint_ablation/
```

先运行 `parent_disjoint_ablation.py` 做两阶段审计：只有 operational top-k 中实际出现 same-parent 的
sample-condition 才重新执行 fingerprint retrieval 和 GLM。无 same-parent 的输入按 hash 直接复用；
mechanism condition 还应通过 `--group-analysis-source-batch` 复用 hash 未变化的 family branch。禁止通过
删除受影响样本、只评估子集或从 similarity threshold 以下补候选来节省计算。

计划报告必须同时统计 operational retrieval 的 same-parent 暴露量：受影响 query-condition 数、受影响
group 数、same-parent neighbor slots、rank-1 slots、全部 neighbor slots 中的占比，以及在单个
query-condition 内去重后的 same-parent record 数。`slot` 是一个 neighbor 在一个 retrieval group 中的一次
LLM-visible 出现；同一 record 若进入多个 mechanism branch，应计为多个 slots。该统计描述暴露频率，不能
单独解释为性能因果效应；因果敏感性仍以 operational/parent-disjoint 配对 prediction flip 和指标变化为准。

`--materialize` 之后，按计划只选择 retrieval conditions 运行统一入口（不要把 `none` 重复写入
parent-disjoint root）：

```bash
python -m tools.chembl_tool.paper_experiments.molecular_evidence_agent \
  --visibility-mode deployment_visible \
  --neighbor-identity-policy parent_disjoint \
  --experiments <retrieval_condition_names_from_plan>
```

完成 parent-disjoint 条件后统一运行：

```bash
python -m tools.chembl_tool.paper_experiments.summarize_parent_disjoint_results
```

该汇总器会生成 condition-level 配对指标、逐样本 prediction flips、修正/破坏计数、整条 run 与 family
branch 复用计数，并从最终 `retrieval.json` 重新审计 retained neighbor 的 identity relation 和 similarity
threshold。最终报告前必须满足所有条件完成、`n_failed_runs == 0`、parent-policy conflict 为 0、低于
threshold 的补位为 0。

`full_flat` 和 `full_mechanism` 必须来自同一组选定的数据源证据。两者唯一预期差异是：前者将证据集合放在一个 group 中，后者按任务机制拆分证据。

E12 distance expansion 同时运行 flat primary 和 mechanism supplementary。Mechanism 顺序固定为
`D+C -> D+C+H1 -> D+C+H1+H2`，但内部是 C-family tree 而非平行 D/C/H1/H2 groups：D 为 root，每个当前
C mechanism family 恰有一个聚合 H1 child，每个 H1 至多有一个 optional H2 child。H1 condition 复用 hash
未变化的 D/C family branches，只运行各 C family 的 H1 child 后重跑 final；H2 condition 必须以前一个 H1
condition 为 `--group-analysis-source-batch`，复用 D/C/H1，只运行 available H2 children 后
再次重跑 final。加入 H2 后 H1 的 group ID、neighbors、evidence serialization 和 branch hash 必须不变；
flat/mechanism 同 prefix 的 evidence-row multiset 必须完全一致。

每个 C/H1/H2 tree node 独立最多检索 3 个 unique molecular neighbors，并使用同一 similarity threshold 与 identity
policy。聚合 H1/H2 node 内的不同 target/measurement families 共享 node 的 top-3，不能按 target 各取 3 个。
每个 extension measurement family 必须唯一归属一个 C parent；H2 到任意 D/C measured node 存在一跳捷径时，
validation 必须失败并将其重分到相应 H1 或排除。

E12 是全程 ChEMBL-only，而不是仅 H1/H2 使用 ChEMBL：D/C 来自当前 ChEMBL direct/full evidence，H1/H2
来自同一冻结 ChEMBL release。禁止 `Starling D/C + ChEMBL H1/H2` 及其它跨 source prefix；distance audit 必须
验证所有层共享同一 ChEMBL source manifest、标准化版本、quality gate 和 provenance contract。Starling 只用于
独立的 E2/RQ2 matched-scope source comparison。

以下 BBB v2 命令与产物只保留为 historical retrieval prototype，不是当前 tree ontology 的可运行入口：

```bash
python -m tools.chembl_tool.tasks.bbb_martins.build_distance_assay_manifest \
  --chembl-sqlite tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db \
  --export-activities
python -m tools.chembl_tool.tasks.bbb_martins.build_distance_extension_library --workers 128
python -m tools.chembl_tool.paper_experiments.build_distance_index \
  --base-index outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_neighbor_index.pkl \
  --extension-index outputs/chembl_tool/tasks/bbb_martins/distance_expansion/evidence_library/v3/extension/bbb_distance_extension_index.pkl \
  --output-index outputs/chembl_tool/tasks/bbb_martins/distance_expansion/evidence_library/v3/bbb_distance_superset_index.pkl \
  --output-meta outputs/chembl_tool/tasks/bbb_martins/distance_expansion/evidence_library/v3/bbb_distance_superset_index.meta.json \
  --index-version bbb_distance_expansion.v2 --source-release "ChEMBL 36" --workers 128
python -m tools.chembl_tool.paper_experiments.audit_distance_expansion

python -m tools.chembl_tool.paper_experiments.materialize_distance_retrieval
```

BBB v2 通过 D/C measured-state census 排除了 node overlap，但随后发现 MMP-3 可直接影响 tight-junction integrity，
所以其 shortest path 为 1，原 `H2 MMP-3` mapping 无效。v2 的 392-query audit 和 coverage 只说明旧 prototype 的
retrieval 工程性质，不能送入 E12 LLM，也不能作为新 tree ontology 的 coverage/performance。下一版必须重建为：
每个 BBB C family 一个 H1 aggregate child，MMP-9/MMP-3 同属 passive/barrier H1；其它 C families 也分别建立
唯一 H1，H2 只在无全局一跳 shortcut 时建立。独立 E12 tree runner、LLM macro-F1 curve 和 mechanism execution
尚未完成。

2026-07-21 新增 same-molecule causal continuity gate。node-to-node path 正确仍不足以发布：必须证明 inference
没有从 assay molecule A 切换到另一个未观测的 substrate B。BBB v3 审计中 MMP-9/MMP-3 保持同一 barrier
perturbagen，暂时通过但保留强 scope 条件；NRF2/KEAP1-NRF2 需要 query 自身是 induced ABC transporter
substrate，HIF-1/PHD2 需要 query 自身是 GLUT1 substrate，当前 retrieval contract 均未提供这些角色。因此 v3
manifest/index/replay 只保留为 engineering artifact，不能启动正式 E12 LLM。所有未来 task 发布前必须有完整
`FamilySelfRelevanceAudit`，并通过
`validate_self_relevance_audit(config, audits, require_publishable=True)`；prompt 不能代替缺失角色证据。
BBB 详细记录见 `tools/chembl_tool/tasks/bbb_martins/DISTANCE_SELF_RELEVANCE_AUDIT.md`。

每个任务必须先运行 `none` 条件。所有包含检索的条件都必须使用 `--single-analysis-source-batch`，以复用该条件中冻结的 single-molecule 分支。不得为每个消融条件独立重新生成先验。

不得将 API key 的值写入代码、保存到文件中的命令、manifest、报告或 trace。使用 `GLM_API_KEY` 或另一个明确指定的环境变量。

OpenAI SDK 的传输层 retry 上限明确固定为 2，与 baseline run 一致。结构化 JSON 验证在记录的调用路径中最多允许 4 次总尝试。最后两次尝试只重新序列化相同的 JSON evidence，以避开 provider 的确定性退化；不得删除证据或改变 inference setting。single、group 或 final 分支不完整的 run 会被 batch 完整性约束排除，并通过 batch `--skip-existing` 显式重跑。

Group prompt 还具有 provider 传输保护。清理后不超过 750 KB 的 payload 保持不变。超过该阈值时，每个超限 neighbor 最多保留 100 条按确定性等间距采样的 evidence row；prompt 必须记录原始行数、保留行数、原始字节数、阈值和采样方法。该机制在不引入数据源或任务专用规则的情况下避免 endpoint 层 HTTP 413 失败。

报告一个 run 之前：

1. 必须满足 `n_failed_runs == 0`；否则使用 `--skip-existing` 重跑失败样本。
2. 必须满足 `summarize_results.py` 输出的 `query_smiles_trace_leaks == 0`。
3. `identity_blind` 必须通过身份盲化 preflight，并使用脱敏后的 group 输出进行 final synthesis。`group_reasoning_outputs_raw.jsonl` 仅用于审计，绝不能作为 final 输入。
4. 主实验 `deployment_visible` 必须通过正向可见性 contract。补充的 `deployment_visible_prefetched` 必须逐 query replay 对应 `identity_blind` run 的冻结 `retrieval.json` 和 prefetched tool outputs，不能按当前 task config 重新检索或重新执行可能非确定的 MCS。当前 test matched-prefetch 保留 21 个条件并必须覆盖 4,456/4,456 samples；valid 已扩展到 26 个条件并必须覆盖 2,713/2,713 samples。两者都要求 missing、extra、mismatch 全为 0；交集匹配不能替代覆盖率检查。
5. 报告性能时必须同时报告 retrieval coverage；缺少 neighbor 是数据源/索引的真实属性，不得静默删除相应样本。
6. 使用共享汇总程序生成制度内和跨制度的配对测试集比较、bootstrap 区间和精确 McNemar 检验。
7. 标量 KNN 结果必须与 LLM agent 条件分开报告；它是数值型 direct-F 对照，且不属于任何 LLM 可见性制度。

生成的证据和运行输出位于 `outputs/paper/molecular_evidence_agent/`：`runs_deployment_visible/` 是 agentic
主实验候选，`runs/` 是 identity-blind 补充控制，`runs_deployment_visible_prefetched/` 是 matched-prefetch
补充控制。
这些是可复现产物，不是源代码。

## 可视化与产物清理

论文 performance 可视化统一使用 `plot_retrieval_claims_overview.py` 的横向 grouped-bar 设计。
后续 test/valid 或新 split 的性能图应扩展这一入口，不再并行保留另一套 overview 绘图代码。
每个 split 的 `analysis/figures/` 只保留两个正式产物：

```text
retrieval_claims_overview.svg
retrieval_claims_overview_highres.png
```

`preview`、`qa`、`pre_parent_disjoint` 和已被这张图取代的其它 overview 文件不得留在正式
figures 目录。SVG 是可复现源图；`--png-output` 通过 ImageMagick 导出便于查看的
唯一 raster 版本。

## Valid 诊断矩阵记录

2026-07-22 至 2026-07-23 已将 valid 的 Identity-blind、matched-prefetch 和 deployment-visible
扩展到各 26 个条件，每套 2,713 个 sample-condition，失败均为 0；prefetch audit 为
2,713/2,713，无 missing、extra 或 mismatch。Parent-disjoint 的 22 个检索条件共 2,275 个
sample-condition，失败为 0，最终 neighbor identity conflict 和低于 threshold 的补位均为 0。
Bioavailability scalar KNN 在 valid 上的 macro-F1 为 0.6621。详细数值与
解释必须同步查阅 `RESULTS.md` 和 `outputs/paper/molecular_evidence_agent_valid/analysis/`。

Skin Reaction valid visibility trace 的规范化分析写在
`SKIN_REACTION_VISIBILITY_TRACE_AUDIT.md`；可分享的双语静态报告及其 artifact/source/config 位于
`reports/skin_reaction_visibility_trace_casebook/`。Portable report 的语言按钮必须通过通用
`inject_portable_report_language_toggle.py` 注入，report-specific 文案放在 `language_config.json`，
不得再把专用 title、description 或两语言逻辑复制进一次性脚本。

Paper runner 必须保存每个样本自己的 `trace_messages.jsonl`，并传 `--no-combine-traces`，避免再生成
condition-level 的重复大文件。最终 trace viewer 只服务四个 paper run root：`runs/`、
`runs_deployment_visible_prefetched/`、`runs_deployment_visible/` 和
`runs_deployment_visible_parent_disjoint/`，通过 `predictions.jsonl` 定位 per-run trace；不得重新加入旧
task、Tier、expert policy 或 task-specific prediction 字段的硬编码适配。Parent-disjoint 仍必须通过
`analysis/parent_disjoint_ablation/` 的汇总产物审计；viewer 还必须从 manifest 和 `reuse.json` 明确显示
`parent_disjoint` policy，以及 sample 是因 retrieval 变化而重跑，还是因 LLM-visible input hash 未变化而复用。
