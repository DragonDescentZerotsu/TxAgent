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

STARLING_BENCHMARK_RESULTS.md
  当前 Starling random/scaffold lineage 的集中总账：frozen split/label 决策、formal pipeline、
  MiniMol head、Morgan KNN、MiniMol embedding cosine KNN、identity-blind 进度、
  Skin retrieval degradation、Tier 1+2 post-hoc 和所有入口。
  它与主要记录旧 TDC test/valid matrix 的 RESULTS.md 分开，不能跨 lineage 混表。

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
tools/chembl_tool/common/starling/heldout_index.py
tools/chembl_tool/paper_experiments/build_starling_benchmark_indices.py
tools/chembl_tool/paper_experiments/starling_benchmark_matrix.py
tools/chembl_tool/paper_experiments/summarize_starling_benchmark.py
tools/chembl_tool/paper_experiments/plot_starling_benchmark_overview.py

data/processed_starling/<Task>/random/test.jsonl
data/processed_starling/<Task>/scaffold/test.jsonl
```

将它接入 paper runner 时必须满足：

1. matrix manifest 顶层显式记录 `benchmark_source=starling` 和 `random|scaffold`；每个 experiment
   记录 task、`input_jsonl_sha256` 和 task provenance reference；top-level `benchmark_provenance`
   保存 protocol、identity normalizer、seed、source revision/metadata，以及 task/split summary、
   test input 和 `test_molecule_labels.jsonl` 的 SHA-256；
2. random 与 scaffold 分别使用对应 `test_molecule_labels.jsonl` 构建的 train-only evidence index，
   不能复用从 full Starling direct source 构建的旧 index；
3. 在运行 LLM 前审计 test parent identity 与 index molecule identity 为零重叠；scaffold split 还需保留
   train/test scaffold 零重叠的 builder audit；
4. 两种 Starling split 使用不同 output root/batch ID，且都与旧 TDC test/valid root 隔离；
5. 汇总器和图表必须按 benchmark lineage 分区，不得把 TDC、Starling-random 和
   Starling-scaffold sample-condition 合并成一个指标。

`build_starling_benchmark_indices.py` 从既有 full-source Starling evidence rows 中删除对应 split 的
全部 test parents，然后重建 direct/full index；因此可使用不属于 gold train 的其它 Starling records，
但不能保留任何 test-parent record。构建时必须用当前 normalizer 从 `drug` 重算 test parent key 并与
artifact 中保存的 key 一致；无法解析 parent 的 source evidence row 采用保守排除，不能在无法证明 disjoint
时仍写入 evidence artifact。`starling_benchmark_matrix.py` 复用冻结的 GLM、prompt、retrieval mode、
tool 和 batch pipeline，仅替换 test input、Starling index 与隔离 output root：

```text
outputs/paper/molecular_evidence_agent_starling_random/
outputs/paper/molecular_evidence_agent_starling_scaffold/
```

### MiniMol feature agent retrieval

MiniMol agent retrieval 是与 Morgan/Tanimoto formal pipeline 配对的 feature ablation：evidence source、
paper-facing group mapping、`top_k=3`、数值 `min_similarity=0.3`、GLM/prompt/tool 设置和
operational -> parent-disjoint 顺序全部不变，只把 candidate ranking 改为 MiniMol v1 embedding
L2-normalized cosine。Cosine 不能使用 Morgan 的结构相似性 bucket 文案，也不能使用
`query_feature_coverage` 这个 Morgan-bit selector。

MiniMol 模型不能在每个 reasoning worker 内即时加载。先对 random/scaffold 的 13 个实际 base index 和
六套 test query 构建共享 registry、按 index 顺序物化 candidate store，再生成轻量 descriptor。Builder
必须记录 checkpoint/config SHA-256，并通过 candidate row order、query coverage、formal MiniMol
`test.pt` cosine parity 和 Starling test-parent exclusion gate：

若 legacy evidence candidate 的 whole-record SMILES 可用于旧 index/fingerprint、但 Graphium 无法图化，
builder 只能对该 candidate 使用 `minimol_parseable_parent_or_fragment.v1` fallback，并逐条保存
registry row、原始 SMILES、实际 embedded SMILES 和原因；不得静默丢 row或填充伪向量。Gold test query
不允许 fallback，仍必须与 formal MiniMol cache 逐条通过 cosine parity。

```bash
env CUDA_VISIBLE_DEVICES=0 /data1/tianang/anaconda3/envs/intern/bin/python \
  -m tools.chembl_tool.paper_experiments.build_minimol_retrieval_features \
  --splits random scaffold
```

产物与 Morgan index、Morgan agent runs 隔离：

```text
outputs/paper/minimol_retrieval_features/
outputs/paper/molecular_evidence_agent_starling_random_minimol_retrieval/
outputs/paper/molecular_evidence_agent_starling_scaffold_minimol_retrieval/
outputs/paper/minimol_retrieval_agent_results/
```

每个 split 仍先跑 19 个 deployment-visible operational retrieval conditions。Query-only single branch
直接复用同 split 已冻结的 `none` batch；Morgan condition 的 group output 只有在
`retrieval_prompt_hash` 完全相同时才允许复用。随后按 MiniMol operational artifact 生成
parent-disjoint plan，再跑正式 parent-disjoint：

```bash
python -m tools.chembl_tool.paper_experiments.starling_benchmark_matrix \
  --benchmark-split <random|scaffold> \
  --retrieval-feature minimol \
  --visibility-mode deployment_visible \
  --neighbor-identity-policy operational \
  --parallelism 12 \
  --group-workers 4

python -m tools.chembl_tool.paper_experiments.parent_disjoint_ablation \
  --benchmark-split <random|scaffold> \
  --retrieval-feature minimol \
  --materialize

python -m tools.chembl_tool.paper_experiments.starling_benchmark_matrix \
  --benchmark-split <random|scaffold> \
  --retrieval-feature minimol \
  --visibility-mode deployment_visible \
  --neighbor-identity-policy parent_disjoint \
  --parallelism 12 \
  --group-workers 4

python -m tools.chembl_tool.paper_experiments.summarize_minimol_retrieval_agent

python -m tools.chembl_tool.paper_experiments.plot_minimol_retrieval_agent \
  --png-output outputs/paper/minimol_retrieval_agent_results/figures/minimol_vs_morgan_agent_retrieval_highres.png

# operational MiniMol agent 与已有正式 Morgan agent / supervised baselines 同图比较
python -m tools.chembl_tool.paper_experiments.plot_starling_with_minimol_agent \
  --png-output outputs/paper/minimol_retrieval_agent_results/figures/starling_benchmark_with_minimol_agent_operational_highres.png
```

上述长矩阵也可以用单一可恢复入口串联；它内部仍调用同一 matrix、parent-disjoint materializer、
summarizer 和 plotter，不定义第二套实验逻辑：

```bash
python -m tools.chembl_tool.paper_experiments.run_minimol_retrieval_agent_experiment
```

`--limit 1` 只用于首个 end-to-end smoke；正式汇总必须覆盖 random/scaffold 全部 38 个 retrieval
condition 且 Morgan/MiniMol 均为零 failed sample。该 ablation 使用独立配对汇总，不能把
MiniMol/cosine agent row 冒充或覆盖 canonical Morgan agent row。

正式 Starling performance bar chart 从合并后的 `metrics.tsv` 读取 random/scaffold、三个 task、
parent-disjoint pipeline 条件、MiniMol train-all baseline、Morgan fingerprint KNN 和 MiniMol
embedding cosine KNN。两种 KNN 都只从同 split 的 `train.jsonl` 检索，固定 `k=3`，按未加权多数票预测，
并用正类邻居比例计算 AUROC。图中主指标为 macro-F1，输出只保留 canonical SVG 和一份高分辨率 PNG：

```bash
python -m baselines.structure_knn.run \
  --data-dir data/processed_starling/<Task>/<random|scaffold> \
  --output-dir outputs/baselines/structure_knn_starling/<Task>/<random|scaffold> \
  --k 3

python -m baselines.minimol.run_embedding_knn \
  --data-dir data/processed_starling/<Task>/<random|scaffold> \
  --embedding-cache-dir outputs/baselines/minimol_starling/<Task>/<random|scaffold>/embeddings \
  --output-dir outputs/baselines/minimol_embedding_knn_starling/<Task>/<random|scaffold> \
  --k 3

python -m tools.chembl_tool.paper_experiments.summarize_starling_benchmark

python -m tools.chembl_tool.paper_experiments.plot_starling_benchmark_overview \
  --png-output outputs/paper/starling_benchmark_results/figures/starling_benchmark_overview_highres.png
```

Coverage-selector KNN 是 retrieval selector 的诊断实验，不替代上述正式、全 test 的 Morgan KNN baseline。
其 matched strict-threshold 口径固定 `k=3`、`min_similarity=0.3`，分别运行 `similarity` 和
`query_feature_coverage`；不足 3 个合格 train neighbors 的 query 必须以
`status=insufficient_neighbors` 保留在 predictions 中并排除出 supported-cohort metrics，不得用低于阈值的
neighbors 回填，也不得用 class prior 猜测。结果必须同时报告 `n_evaluated` / `evaluation_coverage`，
尤其 Bioavailability 和 scaffold split 的 supported cohort 可能只覆盖少数 test samples：

```bash
python -m baselines.structure_knn.run \
  --data-dir data/processed_starling/<Task>/<random|scaffold> \
  --output-dir outputs/baselines/structure_knn_coverage_starling/<Task>/<random|scaffold>/minsim0p3_supported_k3/<selector> \
  --k 3 \
  --min-similarity 0.3 \
  --neighbor-selector <similarity|query_feature_coverage>
```

### Coverage-selector matched LLM ablation

Coverage-selector LLM ablation 固定 Starling full evidence、`full_mechanism`、deployment-visible、
`parent_disjoint`、`top_k_per_group=3`、`min_similarity=0.30` 和同一 GLM/prompt，只比较 Morgan similarity
与 `query_feature_coverage` selector。六个 task/split 条件统一由一套 canonical 汇总和绘图入口处理；早期
BBB scaffold pilot artifact 仍可作为其中一行输入，但不再维护独立 pilot 汇总器或独立 overview 绘图代码：

```bash
python -m tools.chembl_tool.paper_experiments.summarize_coverage_selector_llm_matrix
python -m tools.chembl_tool.paper_experiments.analyze_coverage_selector_retrieval_changes
python -m tools.chembl_tool.paper_experiments.plot_coverage_selector_llm_matrix \
  --png-output outputs/paper/coverage_selector_llm/analysis/figures/coverage_selector_llm_matrix_highres.png
```

`summarize_coverage_selector_llm_matrix.py` 是全样本配对指标、McNemar 和 paired-bootstrap 的唯一入口；
`analyze_coverage_selector_retrieval_changes.py` 进一步审计 neighbor set/rank/top-1 变化、Morgan feature
coverage、neighbor redundancy、group reasoning content 和相同 final input 下的模型波动。两者共享同一
condition/path resolver，新增 split 或迁移 artifact 路径时不得复制 special case。

正式运行顺序仍是 deployment-visible operational、`parent_disjoint_ablation --materialize`、
deployment-visible parent-disjoint。`parent_disjoint_ablation.py` 的 `--benchmark-split random|scaffold`
用于读取新 matrix；旧 `--split test|valid` 语义不变。在完整矩阵、failure audit 和汇总完成前，
`RESULTS.md` 中不存在正式 Starling rerun 结果。

### Starling identity-blind 补充控制

Starling random/scaffold 的 identity-blind 使用同一个 `starling_benchmark_matrix.py`，但必须显式选择
`identity_blind + operational`。该制度隐藏 query/neighbor 的结构、名称和 source ID，由 harness 预先计算
properties/comparison tool text，再把脱敏后的 evidence 和 branch output 交给 LLM。它是补充控制，不取代
deployment-visible parent-disjoint 主结果；identity-blind 与 agentic deployment-visible 同时改变了身份可见性
和工具执行方式，不能单独解释为纯 identity effect。严格的可见性比较仍需后续
`deployment_visible_prefetched` 对同一 blind run 做 frozen retrieval/tool replay，并通过
`audit_prefetch_contract.py`。

正式入口：

```bash
python -m tools.chembl_tool.paper_experiments.starling_benchmark_matrix \
  --benchmark-split random \
  --visibility-mode identity_blind \
  --neighbor-identity-policy operational \
  --parallelism 12 \
  --group-workers 4

python -m tools.chembl_tool.paper_experiments.starling_benchmark_matrix \
  --benchmark-split scaffold \
  --visibility-mode identity_blind \
  --neighbor-identity-policy operational \
  --parallelism 12 \
  --group-workers 4
```

可用 `--experiments <condition...>` 按 task/condition 分队列并行；batch runner 内置 `--skip-existing`，
所以相同命令也是唯一 resume/failed-sample repair 入口。并行选择性运行时，matrix manifest 使用
selection-specific 文件名和原子替换，不能让多个 launcher 共享写一个 manifest。只需重建/检查完整
canonical manifest 而不启动 condition 时，在上述命令加 `--manifest-only`。产物固定写入：

```text
outputs/paper/molecular_evidence_agent_starling_random/runs/
outputs/paper/molecular_evidence_agent_starling_scaffold/runs/
```

两套 split 分别汇总，不能写入正式 parent-disjoint `metrics.tsv` 或在 blind 尚未完成时进入主 bar chart：

```bash
python -m tools.chembl_tool.paper_experiments.summarize_results \
  --split test \
  --paper-root outputs/paper/molecular_evidence_agent_starling_random \
  --output-dir outputs/paper/molecular_evidence_agent_starling_random/analysis_identity_blind

python -m tools.chembl_tool.paper_experiments.summarize_results \
  --split test \
  --paper-root outputs/paper/molecular_evidence_agent_starling_scaffold \
  --output-dir outputs/paper/molecular_evidence_agent_starling_scaffold/analysis_identity_blind
```

每个 split 必须恰有 22 个 identity-blind condition，并同时满足
`n_failed=0`、`query_smiles_trace_leaks=0`、`visibility_contract_satisfied=true`，才能报告结果。
`summarize_starling_benchmark.py` 仍只汇总 deployment-visible parent-disjoint 主 pipeline、
MiniMol train-all、正式全 test Morgan KNN 和 MiniMol embedding cosine KNN；blind 使用上面的独立
analysis 目录。

2026-07-27 artifact snapshot 已有 random/scaffold 各 22 个 blind condition metrics，但尚未通过完成 gate：
random 有 9 个 failed sample-condition（均在 Bioavailability），scaffold 有 5 个（BBB ChEMBL flat 2 个，
Bioavailability Starling direct-full/flat/mechanism 各 1 个）。这些 point estimates 只能用于 repair
进度诊断，不得写入 formal Starling bar chart；修复后还必须重跑 leak/visibility audit。当前逐条件状态和
结果集中记录在 `STARLING_BENCHMARK_RESULTS.md`。

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
  论文 SVG 共用的颜色、字体、基础绘图 primitive、纵向数值网格、标准 metrics-to-SVG CLI 和唯一
  ImageMagick PNG export helper；新增 figure 不复制视觉常量、网格、标准 CLI 或 `export_png` 实现。

summarize_coverage_selector_llm_matrix.py
  汇总六个 Starling task/split 的 Morgan-vs-coverage matched LLM 指标、配对 outcome、McNemar 和
  macro-F1 bootstrap；同时集中维护 control/coverage artifact 路径。

analyze_coverage_selector_retrieval_changes.py
  在同一配对矩阵上审计 query/group/slot 级 neighbor replacement、top-1/reorder、结构 coverage、
  neighbor redundancy、group reasoning 变化和相同 final input 下的 prediction 波动。

plot_coverage_selector_llm_matrix.py
  只读取 canonical `metrics.tsv` 绘制六条件 accuracy/macro-F1 matched bar chart，不重算指标。
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
condition-level 的重复大文件。最终 trace viewer 默认注册 Starling random/scaffold 与历史 TDC test；
每个 dataset 内只服务四个 paper run root：`runs/`、`runs_deployment_visible_prefetched/`、
`runs_deployment_visible/` 和 `runs_deployment_visible_parent_disjoint/`，通过 `predictions.jsonl` 定位
per-run trace，且不得跨 dataset 合并指标。不得重新加入旧 task、Tier、expert policy 或 task-specific
prediction 字段的硬编码适配。Parent-disjoint 仍必须通过 `analysis/parent_disjoint_ablation/` 的汇总产物
审计；viewer 还必须从 manifest 和 `reuse.json` 明确显示 `parent_disjoint` policy，以及 sample 是因
retrieval 变化而重跑，还是因 LLM-visible input hash 未变化而复用。
