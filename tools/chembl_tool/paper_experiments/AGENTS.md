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

FINAL_EVIDENCE_SURFACE_EXPERIMENT.md
  E13 固定 retrieval/single/group 的 final-only surface 诊断、复现命令、metadata census 和 no-go 结论。
  Card surface 仅为可复现实验插件，不进入默认 prompt 或 formal test。

run_train_ratio_prior_experiment.py / train_ratio_prior_analysis.py
  E14 的 final-only orchestration 与离线 paired/trigger/provenance audit。只运行 BBB/Bio current full-flat valid；
  frozen source 与 train prior 不匹配时必须失败，未通过 promotion gate 时不得运行 test 或升级默认 prompt。

TRACE_RETENTION.md
  最终论文 trace 的唯一目录、保留单位、清理边界和 viewer 约束。
```

新增实验或得到新结果时：先在执行总计划更新对应 E 编号状态，再运行生成器更新机器可读统计和
`analysis/report.md`，然后根据生成产物把已测结果同步到 `RESULTS.md`，最后更新
`VISIBILITY_ANALYSIS.md` 等解释文档。当前生成器不会自动改写 `RESULTS.md`；不得只在聊天中保留结论。

## 当前 scaffold benchmark 合同（2026-08-09）

BBB 当前使用 `experimental_meaningful_cns_access_v2`，valid/test 各 366；Bioavailability 与 Skin 继续使用
`record_supported_v2`，分别各 209 和 245。BBB 目标是系统给药后 meaningful/adequate CNS access vs
restricted/poor access，不是 passive-permeability label，也不是任意微量可检出。各 builder 在 scaffold 不跨
split 的硬约束下，按顺序最小化
held-out singleton、valid/test singleton imbalance、label imbalance，再最大化第一版 valid overlap。
BBB valid/test 分别为 345 multi + 21 singleton 和 344 multi + 22 singleton；Bioavailability 的 held-out
全是 multi-record parents；Skin 每个 held-out split 为
240 multi-record + 5 singleton，这是精确 245/245 下的全局最小 singleton 解。三个 task 的 parent identity
和 scaffold pairwise overlap 都是 0。

```text
builder:
  tools/chembl_tool/common/starling/build_bbb_experimental_meaningful_cns_access.py
  tools/chembl_tool/common/starling/build_record_supported_benchmark.py
data:
  data/processed_starling_experimental_meaningful_cns_access_v2/BBB_Martins/scaffold/
  data/processed_starling_record_supported_v2/{Bioavailability_Ma,Skin_Reaction}/scaffold/
held-out indices:
  outputs/paper/molecular_evidence_agent_starling_scaffold_experimental_meaningful_cns_access_v2/
  outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2/
strict first-version reuse:
  tools/chembl_tool/paper_experiments/seed_starling_matrix_reuse.py
paired direct-agent statistics:
  tools/chembl_tool/paper_experiments/analyze_starling_direct_significance.py
current best-agent vs all-baseline paired statistics:
  tools/chembl_tool/paper_experiments/analyze_starling_best_agent_baselines.py
trace and retrieval-family audits:
  tools/chembl_tool/paper_experiments/audit_bbb_retrieval_coverage.py
historical valid-only BBB E16 (no-go; paths frozen for reproducibility):
  tools/chembl_tool/paper_experiments/matched_train_label_agent/bbb_property_compatibility_audit.py
  tools/chembl_tool/paper_experiments/matched_train_label_agent/bbb_property_compatible_contract.py
  tools/chembl_tool/paper_experiments/matched_train_label_agent/bbb_property_compatible_experiment.py
  tools/chembl_tool/paper_experiments/matched_train_label_agent/bbb_property_compatible_report.py
  tools/chembl_tool/paper_experiments/matched_train_label_agent/bbb_property_compatible_trace_diagnosis.py
current diagnostic audits:
  tools/chembl_tool/paper_experiments/audit_skin_reasoning_bottleneck.py
  tools/chembl_tool/paper_experiments/audit_starling_trace_failure_causes.py
```

`build_starling_benchmark_indices.py` 和 `starling_benchmark_matrix.py` 对非默认 lineage 必须显式传
`--benchmark-data-root`、`--benchmark-lineage` 和 `--canonical-paper-root`；model/visibility 仍用独立
`--output-root`。历史 artifact 只允许按 molecule key 复用；target retrieval 必须重新物化，single 需要
完全一致的 model/reasoning/visibility 合同，group 还需相同 LLM-visible group input，final 只在完整
`retrieval_prompt_hash` 相同且全部 expected groups 已复用时复制。复用审计写入 target root 的
`reuse_audit/`，不得按 query index 直接搬运。

第一版 `record_agreement70_split811_v1` 正式结果保留为 historical comparison。曾生成的 exploratory
`record_supported_v1` 数据、indices、agent runs 和 baselines 已删除；不得在文档或汇总中恢复引用。

## 第一版新数据集默认运行合同（2026-08-01 冻结）

本节记录第一版相对旧 TDC/strict-conflict lineage 的默认描述。对
`record_agreement70_split811_v1`，正式默认是：

```text
base_url: http://127.0.0.1:50000/v1
model: nvidia/GLM-5.2-NVFP4
reasoning_effort: ""（省略参数，保留历史 GLM reasoning 行为）
visibility_mode: identity_blind
neighbor_identity_policy: parent_disjoint
operational_staging_used: false
endpoint_concurrency_budget: 512
default launcher shape: one global prompt pool with --parallelism 128
```

`identity_blind + parent_disjoint` 是新主矩阵，不再是补充控制。它直接从 valid+test-heldout-filtered index
fresh retrieval，并由 harness 预取脱敏工具证据；operational 不再预跑，`reuse_plan.json` 不再是主矩阵输入。
既有 deployment-visible、operational、matched-prefetch 和 reuse 结果全部保留为 historical lineage。

512 是同一 endpoint 的全局 outstanding-request budget。首次 512 并发 valid 压力运行在 500 个 BBB
query-only samples 中出现 1 个 transport timeout；随后 384 并发 BBB ChEMBL full-flat 又有 193/500
失败，其中 190 个是 group request timeout。因此当前默认降为 128，以适配 reasoning-enabled
长 group prompt 的实际吞吐；512 仅保留为 endpoint ceiling，不再是 agent launcher 默认值。
默认只允许一个 matrix launcher 占用预算；不得再并行
启动多个 split/task launcher，也不得通过静态 condition lanes 绕过 global pool。第一轮 valid 运行若出现
429、连接/超时错误、服务队列持续增长或 structured-output failure rate 上升，
再下调并发；不得因为并发调整而改变 prompt、reasoning、temperature、max tokens 或 validation policy。

matrix 唯一调度器是通用全局 prompt pool。它不按 task 或 condition 预留
并发槽，而是在同一个 `--parallelism` 上限下混合调度所有 ready single/group/final branch；final 只有在该
sample-condition 的 single 和全部 expected group 均为 `status=ok` 后才进入 ready queue。首次没有可恢复
stage artifact 的 sample 必须通过 task pipeline 的 `--prepare-only` 只生成 retrieval seed，不得在 seed job
发出 LLM 请求；parent scheduler 随后写兼容 manifest，并从第一条 single/group prompt 起进入共享 pool。
跨 condition 的 frozen-single 依赖按 query index 动态解锁，不得保留整批 `none` phase barrier。
`--max-stage-requeues N` 控制 validation 内置尝试耗尽后的即时 stage 重排队次数。

该 scheduler 必须保持既有 artifact contract：canonical `retrieval.json`、single/group/final output、
`trace_messages.jsonl`、batch predictions/metrics/report 的路径和 schema 不变；只允许增加 manifest 中的
`scheduler`/`stage_pool` provenance 和 `.run.lock`。每个成功 stage 必须原子落盘，重启后
只从缺失或失败 stage 恢复；任何 prerequisite branch 改写时必须先使旧 final/trace 失效，再动态重建 final。
final trace 仍调用 task 原有 `_write_trace_jsonl()` 生成。同一 root 同时只能有一个 matrix launcher。

Harness-prefetched molecular tools 统一通过 `ToolServiceClient.invoke_many()` 将每个 sample 的 query
properties 和全部 neighbor pair comparisons 作为一个 batch 提交。tool service 的 persistent cache key
只依赖 tool contract/input/version，与 Morgan、MiniMol、coverage-based 或 future retrieval feature 无关；
因此替换 retrieval 方法只会产生新的 neighbor pairs，不得增加另一套 task/retriever-specific tool
materializer。node002 正式服务拓扑和验证入口见 `tools/service/README.md`。

runner 已实现该目标合同：Starling v4 CLI 默认先跑 `valid`，使用
`identity_blind + parent_disjoint`、`parallelism=128` 的单一 global prompt pool，并且 fresh 主矩阵
不读 operational reuse plan。`valid` 产物写入 lineage 同名 `_valid` root，与正式 test root 隔离。
超过 512 的 `parallelism` 会在 launcher 启动前拒绝。旧 generic runner 的
deployment-visible/operational 路径仅保留用于 historical ablation。

非默认模型对照必须给 `starling_benchmark_matrix.py` 传显式 `--output-root`，并在目录名中包含模型标识；
该参数只隔离 run/manifest 产物，canonical split-specific held-out evidence index 仍从原 Starling lineage root
读取。不同模型不得复用或覆盖彼此的 `runs_identity_blind_parent_disjoint/`。例如：

```bash
python -m tools.chembl_tool.paper_experiments.starling_benchmark_matrix \
  --benchmark-split scaffold --evaluation-subset valid \
  --model gpt-oss-20b --base-url http://127.0.0.1:9001/v1 \
  --api-key-env GPT_OSS_LOCAL_API_KEY \
  --output-root outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid_gpt_oss_20b
```

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
tools/chembl_tool/paper_experiments/plot_starling_model_comparison.py
tools/chembl_tool/common/task_workflows/global_prompt_pool.py
tools/chembl_tool/common/task_workflows/reasoning_stage_runtime.py

data/processed_starling/<Task>/random/{train,valid,test}.jsonl
data/processed_starling/<Task>/scaffold/{train,valid,test}.jsonl
```

`plot_starling_model_comparison.py` 是当前 Starling model/baseline/ablation 的唯一正式总图。后续完整
model/visibility summary 用可重复的 `--comparison-metrics` 传入；matched method 实验用可重复的
`--experiment-metrics` 传入。两类实验都不得新增一次性 overview 图，继续输出到
`starling_benchmark_results_scaffold_valid_gpt_oss_20b_vs_120b/figures/starling_model_comparison.{svg,png}`。
每个 comparison summary 必须与 reference 使用相同 task/split/subset/sample count 和 baseline。每个
experiment TSV 的每个 task/split 必须有且仅有一个 `comparison_role=anchor`（兼容历史
`method=morgan_standard`），可用 `base_method` 指向总图 candidate metrics 中的既有 condition；anchor 的
`base_model_label` 可进一步指向任一已通过 `--comparison-metrics` 加载的 model/visibility series，未提供时
仍默认主 candidate。Anchor 的 sample count、evaluation subset 和 macro-F1 必须完全对齐，且不会重复画出。新增方法行可用
`plot_label` 提供短标签；没有时使用 `method_label`。

将它接入 paper runner 时必须满足：

1. matrix manifest 顶层显式记录 `benchmark_source=starling` 和 `random|scaffold`；每个 experiment
   记录 task、`input_jsonl_sha256` 和 task provenance reference；top-level `benchmark_provenance`
   保存 protocol、identity normalizer、seed、source revision/metadata、parent agreement policy，以及 task/split
   summary、valid/test inputs、各自 label audit 和 valid+test union `heldout_molecule_labels.jsonl` 的 SHA-256；
2. random 与 scaffold 分别使用对应 `heldout_molecule_labels.jsonl` 构建的 train-only evidence index，
   不能复用从 full Starling direct source 构建的旧 index；
3. 在运行 LLM 前审计 valid+test parent identity union 与 index molecule identity 为零重叠；scaffold split
   还需保留 train/valid/test scaffold 两两零重叠的 builder audit；
4. 两种 Starling split 使用不同 output root/batch ID，且都与旧 TDC test/valid root 隔离；
5. 汇总器和图表必须按 benchmark lineage 分区，不得把 TDC、Starling-random 和
   Starling-scaffold sample-condition 合并成一个指标。

`build_starling_benchmark_indices.py` 从既有 full-source Starling evidence rows 中删除对应构造方法的
全部 valid+test parents，然后重建 direct/full index；因此可使用不属于 gold train 的其它 Starling records，
但不能保留任何 evaluation-parent record。构建时必须用当前 normalizer 从 `drug` 重算 held-out parent key 并与
artifact 中保存的 key 一致；无法解析 parent 的 source evidence row 采用保守排除，不能在无法证明 disjoint
时仍写入 evidence artifact。`starling_benchmark_matrix.py` 复用冻结的 GLM、prompt、retrieval mode、
tool 和 batch pipeline，仅替换 test input、Starling index 与隔离 output root：

```text
outputs/paper/molecular_evidence_agent_starling_random_record_agreement70_split811_v1/
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1/
```

### MiniMol feature agent retrieval

MiniMol agent retrieval 是与 Morgan/Tanimoto formal pipeline 配对的 feature ablation：evidence source、
paper-facing group mapping、`top_k=3`、数值 `min_similarity=0.3`、GLM/prompt/tool 设置和
`identity_blind + parent_disjoint` fresh-run 合同全部不变，只把 candidate ranking 改为 MiniMol v1 embedding
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

新 v4 每个 split 直接跑完整 `identity_blind + parent_disjoint` 条件。Query-only `none` 同轮 fresh-run，
其 identity policy 在 manifest 中标为不适用；retrieval conditions 不从 Morgan 或 operational artifacts
复用 group/final output：

```bash
python -m tools.chembl_tool.paper_experiments.starling_benchmark_matrix \
  --benchmark-split <random|scaffold> \
  --retrieval-feature minimol \
  --visibility-mode identity_blind \
  --neighbor-identity-policy parent_disjoint \
  --parallelism 128

python -m tools.chembl_tool.paper_experiments.summarize_minimol_retrieval_agent

python -m tools.chembl_tool.paper_experiments.plot_minimol_retrieval_agent \
  --png-output outputs/paper/minimol_retrieval_agent_results/figures/minimol_vs_morgan_agent_retrieval_highres.png

# matched parent-disjoint MiniMol agent 与 Morgan agent / supervised baselines 同图比较
python -m tools.chembl_tool.paper_experiments.plot_starling_with_minimol_agent \
  --png-output outputs/paper/minimol_retrieval_agent_results/figures/starling_benchmark_with_minimol_agent_highres.png
```

代码迁移完成后，长矩阵仍由单一可恢复入口串联；该入口应直接调用 fresh parent-disjoint matrix、
summarizer 和 plotter，不再调用 operational matrix 或 parent-disjoint materializer，也不定义第二套实验逻辑：

```bash
python -m tools.chembl_tool.paper_experiments.run_minimol_retrieval_agent_experiment
```

`--limit 1` 只用于首个 end-to-end smoke；正式汇总必须覆盖 random/scaffold 全部 38 个 retrieval
condition 且 Morgan/MiniMol 均为零 failed sample。该 ablation 使用独立配对汇总，不能把
MiniMol/cosine agent row 冒充或覆盖 canonical Morgan agent row。

正式 Starling performance bar chart 从合并后的 `metrics.tsv` 读取 random/scaffold、三个 task、
parent-disjoint pipeline 条件、lineage-frozen MiniMol head baseline、Morgan fingerprint KNN 和 MiniMol
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

Coverage-aware reasoning context 是与 selector 正交的 opt-in ablation：
`--neighbor-context-profile standard|coverage_aware|coverage_mmp_ledger` 只改变 group branch 的附加输入，
不改变 neighbor set、raw retrieval、task schema 或 final prompt。`coverage_aware` 提供匿名 Morgan feature /
query atom-environment marginal coverage；visible-only `coverage_mmp_ledger` 复用常驻
`mmp_structure_compare` 的逐 neighbor MCS/MMP 文本，并用 Morgan marginal feature 统计组织集合级 ledger。
后者禁止用于 `identity_blind`，不得把 Morgan feature coverage 描述成 atom coverage；没有 matched-pair
transformation 时具体 fragment 对应必须保持 unresolved。
非默认 selector/profile 必须写入显式独立 `--output-root`；group artifact reuse 还必须 profile 一致。
2026-08-03 GPT-OSS-120B scaffold-valid、blind+parent-disjoint matched run 的三 task point estimates 为
BBB `+0.0143` macro-F1、Bioavailability `+0.0022`、Skin `-0.0285`，三项 interval 均跨零，结论为
mixed / no-go for promotion，不进入 test。相对原始 Morgan-standard，coverage-aware 在 BBB 上为
`+0.0356` macro-F1 且 interval 不跨零，但 Bioavailability/Skin 不一致，所以只能视为 task-specific signal。
完整合同审计位于下面的 analysis root；正式图只追加到前述
`starling_model_comparison.{svg,png}` 总图，不在该 analysis root 保留单实验 figure：

```text
outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/
```

新 v4 正式运行顺序是 valid `identity_blind + parent_disjoint` fresh-run、valid audit、设置冻结、test
fresh-run、test audit 和汇总。`parent_disjoint_ablation.py` 仅用于旧 operational lineage 或显式 opt-in
sensitivity，不参与默认路径。在完整矩阵、failure/leak/identity/held-out audit 和汇总完成前，
`RESULTS.md` 中不存在正式 v4 Starling rerun 结果。

### Starling identity-blind parent-disjoint 主矩阵

Starling random/scaffold 的新 v4 主矩阵使用同一个 `starling_benchmark_matrix.py`，默认选择
`identity_blind + parent_disjoint`。该制度隐藏 query/neighbor 的结构、名称和 source ID，由 harness 预先计算
properties/comparison tool text，再把脱敏后的 evidence 和 branch output 交给 LLM。当前主张只针对这一冻结
blind contract；不得把它解释为 deployment-visible agentic 性能。若以后需要可见性比较，另开显式
deployment-visible-prefetched/agentic ablation，不得让它阻塞主矩阵。

正式入口：

```bash
python -m tools.chembl_tool.paper_experiments.starling_benchmark_matrix \
  --benchmark-split random \
  --visibility-mode identity_blind \
  --neighbor-identity-policy parent_disjoint \
  --parallelism 128

python -m tools.chembl_tool.paper_experiments.starling_benchmark_matrix \
  --benchmark-split scaffold \
  --visibility-mode identity_blind \
  --neighbor-identity-policy parent_disjoint \
  --parallelism 128
```

可用 `--experiments <condition...>` 按 task/condition 选择队列，但默认顺序运行，不能在多个 launcher 中
各自占用 512 并发。batch runner 内置 `--skip-existing`，所以相同命令也是唯一 resume/failed-sample repair
入口。选择性运行时，matrix manifest 使用 selection-specific 文件名和原子替换，不能让多个 launcher
共享写一个 manifest。只需重建/检查完整
canonical manifest 而不启动 condition 时，在上述命令加 `--manifest-only`。产物固定写入：

```text
outputs/paper/molecular_evidence_agent_starling_random_record_agreement70_split811_v1/runs_identity_blind_parent_disjoint/
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1/runs_identity_blind_parent_disjoint/
```

两套 split 分别汇总；blind parent-disjoint 未完整通过 gate 前不能进入主 bar chart：

```bash
python -m tools.chembl_tool.paper_experiments.summarize_starling_benchmark \
  --splits <random|scaffold> \
  --evaluation-subset <valid|test> \
  --pipeline-root <model-and-subset-specific-v4-root> \
  --model-label <label> \
  --minimol-root <lineage-matched-minimol-root> \
  --structure-knn-root <lineage-matched-morgan-knn-root> \
  --minimol-embedding-knn-root <lineage-matched-minimol-knn-root> \
  --output-dir <model-and-subset-specific-summary-root>
```

`summarize_results.py` 只扫描旧 TDC `experiments_for_split(test|valid)` 和 legacy visibility roots；不得对 v4
root 使用它。旧命令生成的空 `analysis_identity_blind_parent_disjoint/report.md` 不是 canonical v4 结果。
显式 v4/valid summary 不会隐式加载 historical test baselines；需要 baseline 时必须传入同 lineage/subset roots。

Historical combined v4 每个 split 恰有 22 个 identity-blind condition；current BBB experimental-v2 与
Bio/Skin record-v2 已拆分，必须改为按 `lineage × task × declared conditions` 校验 expected set。所有 current
run 仍同时审计 `n_failed`、`query_smiles_trace_leaks=0`、`visibility_contract_satisfied=true`、retained parent
conflict=0 和 held-out overlap=0。正式 test 的默认完成 gate 仍是 `n_failed=0`。若遇到经过重试仍可确定为 non-retryable 的单样本
失败，valid 诊断可以保留失败 trace，并用汇总器的 `count_as_incorrect_opposite_label` policy 计入分母；报告
必须显式写出失败数、原因和 policy，且不能把它称为通过 zero-failure gate。
`summarize_starling_benchmark.py` 的 v4 汇总必须读取这套 blind parent-disjoint 主 pipeline、对应 lineage
冻结的 MiniMol head、正式全 test Morgan KNN 和 MiniMol embedding cosine KNN。

2026-07-27 的 random/scaffold 各 22-condition blind artifact 属于上一版 strict-conflict split，曾分别残留
9/5 个 failed sample-condition；它只保留作 historical repair lineage，不能冒充当前 benchmark。第一版
`record_agreement70_split811_v1` 的 valid 结果已冻结为 historical comparison；当前
Bio/Skin `record_supported_v2` 与 BBB `experimental_meaningful_cns_access_v2` 的逐条件状态记录在
`STARLING_BENCHMARK_RESULTS.md`。各 task 在完整 valid gate 通过前不得启动或解释 test；Bio 已在 gate
通过并冻结设置后于2026-08-10完成一次 selected-condition formal scaffold-test，不能再用该 test 调参。

## 代码与命令入口

```text
molecular_evidence_agent.py
  冻结实验矩阵与统一运行入口；负责 visibility mode、neighbor identity policy、结果 root 和跨条件复用参数。

router_oof/
  train-only KNN-vs-direct-agent router 的独立模块。v1–v3.1 可部署版本按 task 分别训练，不共享参数；使用
  scaffold-or-parent
  5-fold OOF、fold-specific parent-filtered index、gpt-oss-120b identity-blind parent-disjoint direct agent、
  固定 pre-decision features 和 nested OOF Logistic/HistGBDT 评估。v2 删除 fold-dependent reference-size
  与确定性重复 features，使用双风险 score 和 paired-bootstrap promotion gate；v1 `router/` 与 v2
  `router_v2/` 必须保留独立 lineage。v3 `post_selector_v3/` 只在 disagreement rows 学 agent-win probability，
  用 nested OOF 比较 output/query/KNN、完整 evidence、evidence+structured-trace 三档 profile，并按两个输出
  方向分别冻结 threshold；train gate 失败时部署严格回退 KNN。不得把 train fold 塞进正式
  `starling_benchmark_matrix.py` 的 valid/test 枚举。v3.1 `post_selector_v31/` 为两个方向分别选择并校准
  fold ensemble，以最低 route count、Wilson precision lower bound 和 accuracy-first paired-bootstrap gate 控制
  风险；valid 只报告独立 evidence gate，不允许回调 frozen policy。后续 matched-size curve 固定 direction
  specs；唯一不可部署的 shared-transfer termination diagnosis 固定 task-balanced Logistic、完整 generic profile
  和 task-specific calibration，并执行 cross-task identity/scaffold exclusion。它仍以 agent-win 为监督，不是
  counterfactual evidence utility。当前 transfer continuation gate 已失败，禁止继续 valid-informed router
  family/profile sweep，也不启动 router formal test。协议与 gate 见
  `ROUTER_OOF_IMPLEMENTATION_PLAN.md`。

summarize_results.py
  仅汇总旧 TDC identity-blind、matched-prefetch 和 agentic operational 条件，生成 coverage、token、
  visibility audit、paired bootstrap、McNemar/Holm 和 `analysis/report.md`；不读取 v4 run roots。

summarize_starling_benchmark.py
  汇总 v4 model/split/subset-specific pipeline 和显式 lineage-matched baselines；对失败样本使用记录的
  failure-inclusive policy，并禁止 valid summary 静默回退到 historical test baselines。

paired_binary_predictions.py
  离线 audit 的共享二分类配对统计入口：统一 macro-F1 delta、paired bootstrap、prediction flips 和 exact
  McNemar。新的确定性诊断不得各自复制这套基础统计；需要不同假设方向的 permutation test 时再保留独立实现。

watch_glm_tunnel_and_matrix.py
  监控 GLM endpoint、SSH tunnel 和唯一 resumable matrix；严格完成计数要求 task prediction、single/final
  status、expected group 数和 group status 全部有效。支持透传 `--timeout-s`，但不保存或重放密码。

analyze_coverage_performance.py
  只读取 deployment-visible 的 predictions，快速生成 coverage/class-conditional coverage、相对同任务
  none 的 paired macro-F1 增幅、10,000 次 bootstrap、机器可读 TSV/JSON 和报告段落。

audit_prefetch_contract.py
  逐 condition、逐样本验证 matched-prefetch 是否完整 replay identity-blind 的 retrieval 和工具输出。

parent_disjoint_ablation.py
  历史/显式 sensitivity 工具：审计 operational retrieval 的 same-parent overlap，生成选择性重跑/整条复用/family-branch 复用计划；
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
  只负责展示，不在图内重算指标。当前代码默认输出历史 deployment-visible agentic 的 canonical SVG；
  v4 汇总接线前不能拿它冒充 blind 主结果。

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

新 v4 论文主结果表使用 identity-blind 制度；以下 deployment-visible 定义仅用于历史结果或显式 ablation：

```text
identity_blind:
  query/neighbor 的结构、名称和 source ID 对 LLM 隐藏；
  harness 预取 properties/comparison tool text；
  与 parent-disjoint retrieval 配对，作为 v4 none/direct/full_flat/full_mechanism 和 source comparison 主制度。

deployment_visible:
  query 只暴露结构，不暴露名称；
  retrieved molecule 暴露结构，并在数据源提供时暴露 source name/source ID；
  LLM 自主决定是否调用比较工具；
  只作为显式 deployment ablation，不是 v4 默认。

deployment_visible_prefetched:
  只作为 visibility/tool-execution 补充控制；
  不进入 v4 主结果表；既有 parity audit 和结果保留用于 provenance/appendix。
```

主实验和补充控制共享以下基础条件：

```text
default endpoint: http://127.0.0.1:50000/v1 (requires `ssh -fNT parcc-glm`)
model: nvidia/GLM-5.2-NVFP4
reasoning_effort: "" (omit the parameter; preserve the historical GLM reasoning contract)
temperature: 0
max_tokens: 20480
top_k_per_group: 3
min_similarity: 0.30
exact_record_exclusion: true
primary_analog_neighbor_identity_policy: parent_disjoint
visibility_mode: identity_blind
operational_policy_role: opt-in historical/deployment-sensitivity ablation only
endpoint_concurrency_budget: 512
default_launcher_shape: one global prompt pool with parallelism=128, one launcher at a time
```

所有 paper-facing structural-analog retrieval 主结果默认使用 `parent_disjoint`：标准化 query/source parent、
排除相同 parent，再用仍高于原 similarity threshold 的后续候选补足 top-k。新 v4 直接 fresh-run，
不需要 operational staging、diff、整条 run reuse 或 branch reuse。这里的 parent 是 RDKit FragmentParent
标准化结果，不是 active moiety，也不推断 prodrug 或代谢物关系。Operational 只能显式 opt in 到独立 root，
用于 deployment sensitivity；不得成为新 retrieval condition 的依赖。

Parent-disjoint 产物统一放在：

```text
# test
outputs/paper/molecular_evidence_agent_starling_random_record_agreement70_split811_v1/runs_identity_blind_parent_disjoint/
outputs/paper/molecular_evidence_agent_starling_random_record_agreement70_split811_v1/analysis_identity_blind_parent_disjoint/

# scaffold
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1/runs_identity_blind_parent_disjoint/
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1/analysis_identity_blind_parent_disjoint/
```

新 v4 直接运行完整 fresh matrix；`none` 同轮运行但 identity policy 为不适用：

```bash
python -m tools.chembl_tool.paper_experiments.molecular_evidence_agent \
  --visibility-mode identity_blind \
  --neighbor-identity-policy parent_disjoint \
  --parallelism 128
```

最终汇总直接审计 fresh retrieval：所有条件完成、`n_failed_runs == 0`、parent-policy conflict 为 0、
held-out overlap 为 0、低于 threshold 的补位为 0、identity leak 为 0。旧
`summarize_parent_disjoint_results.py` 的 operational/parent 配对输出只属于历史 sensitivity analysis。

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

不得将 API key 的值写入代码、保存到文件中的命令、manifest、报告或 trace。直连 loopback vLLM 默认使用
`GLM_LOCAL_API_KEY` 的非敏感占位值；旧 LiteLLM fallback 使用 `GLM_API_KEY` 或另一个明确指定的环境变量。
旧 `zai-org/GLM-5.2-FP8` 是请求别名，实际 response model 为 `hosted_vllm/nvidia/GLM-5.2-NVFP4`；
不得把 endpoint 对比误写成 FP8 与 NVFP4 模型精度对比。

OpenAI SDK 的传输层 retry 上限明确固定为 2，与 baseline run 一致。结构化 JSON 验证在记录的调用路径中最多允许 4 次总尝试。最后两次尝试只重新序列化相同的 JSON evidence，以避开 provider 的确定性退化；不得删除证据或改变 inference setting。single、group 或 final 分支不完整的 run 会被 batch 完整性约束排除，并通过 batch `--skip-existing` 显式重跑。

Group prompt 还具有 provider 传输保护。清理后不超过 750 KB 的 payload 保持不变。超过该阈值时，每个超限 neighbor 最多保留 100 条按确定性等间距采样的 evidence row；prompt 必须记录原始行数、保留行数、原始字节数、阈值和采样方法。该机制在不引入数据源或任务专用规则的情况下避免 endpoint 层 HTTP 413 失败。

报告一个 run 之前：

1. 必须满足 `n_failed_runs == 0`；否则使用 `--skip-existing` 重跑失败样本。
2. 必须满足 `summarize_results.py` 输出的 `query_smiles_trace_leaks == 0`。
3. v4 主实验 `identity_blind` 必须通过身份盲化 preflight，并使用脱敏后的 group 输出进行 final synthesis；
   `group_reasoning_outputs_raw.jsonl` 仅用于审计，绝不能作为 final 输入。同时逐 retrieval 审计
   parent conflict=0、held-out overlap=0 和 threshold violation=0。
4. `deployment_visible` / `deployment_visible_prefetched` 只在显式 visibility ablation 中运行。Prefetched
   条件必须逐 query replay 对应 `identity_blind` run 的冻结 `retrieval.json` 和 prefetched tool outputs，
   不能按当前 task config 重新检索或重新执行可能非确定的 MCS。历史 matched-prefetch coverage gate
   仍保留用于旧 lineage；它不阻塞新 v4 blind 主矩阵。
5. 报告性能时必须同时报告 retrieval coverage；缺少 neighbor 是数据源/索引的真实属性，不得静默删除相应样本。
6. 使用共享汇总程序生成制度内和跨制度的配对测试集比较、bootstrap 区间和精确 McNemar 检验。
7. 标量 KNN 结果必须与 LLM agent 条件分开报告；它是数值型 direct-F 对照，且不属于任何 LLM 可见性制度。

旧 lineage 的运行输出位于 `outputs/paper/molecular_evidence_agent/`：`runs_deployment_visible/`、`runs/`
和 `runs_deployment_visible_prefetched/` 分别是历史 agentic、blind 和 matched-prefetch artifacts。
新 v4 主输出位于各 record-agreement lineage root 的 `runs_identity_blind_parent_disjoint/`。
这些都是可复现产物，不是源代码。

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
新 v4 dataset 服务 `runs_identity_blind_parent_disjoint/`，历史 dataset 继续服务四个 legacy roots：
`runs/`、`runs_deployment_visible_prefetched/`、`runs_deployment_visible/` 和
`runs_deployment_visible_parent_disjoint/`，通过 `predictions.jsonl` 定位
per-run trace，且不得跨 dataset 合并指标。不得重新加入旧 task、Tier、expert policy 或 task-specific
prediction 字段的硬编码适配。Parent-disjoint 仍必须通过 `analysis/parent_disjoint_ablation/` 的汇总产物
审计；viewer 还必须从 manifest 和 `reuse.json` 明确显示 `parent_disjoint` policy，以及 sample 是因
retrieval 变化而重跑，还是因 LLM-visible input hash 未变化而复用。
