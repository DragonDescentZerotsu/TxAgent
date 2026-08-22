# 分子证据 Agent 论文实验

本目录冻结面向论文的实验矩阵，但不把任务逻辑写进通用 runner。

## 文档入口

- [ICLR 2027 执行计划](ICLR_2027_EXECUTION_PLAN.md)：贡献、实验、资源、缺口、时间表和投稿 gate。
- [当前冻结实验协议](EXPERIMENT_PLAN.md)：第一轮矩阵、参数、指标和统计方法。
- [Pipeline](PIPELINE.md)：数据流、retrieval views、visibility 和 mechanism families。
- [Visibility 分析](VISIBILITY_ANALYSIS.md)：structure visible 后性能上升或下降的集中解释。
- [Skin Reaction trace casebook](reports/skin_reaction_visibility_trace_casebook/index.html)：可直接分享的
  中英文静态案例报告，右上角按钮切换语言。
- [当前结果](RESULTS.md)：已完成 full run 的实测结果。
- [Starling v4 结果总账](STARLING_BENCHMARK_RESULTS.md)：当前 random/scaffold lineage、跨模型、visible 和
  coverage-context 实验的唯一集中记录。
- [Assay-level retrieval](ASSAY_LEVEL_RETRIEVAL.md)：与 group-level 平行的 context-level assay catalog、
  relevance-prefix retrieval、current/default direct-heldout-filtered + query-time `scaffold_disjoint` 与 historical
  reference-pool 合同、conditioned cumulative-family curve、无 summary 模型的 raw support cards、scaffold audit、
  single/evidence-equivalent reuse、英文五-panel scaling figure 和实测结果。
- [ClinTox clinical-trial-failure v1](../tasks/clintox/CLINTOX_CLINICAL_TRIAL_FAILURE_V1.md)：独立
  AACT/FDA source-reconstructed lineage、retrieval hierarchy、v3 prompt、DeepSeek 结果和 no-promotion 结论。
- [外部 Starling Table 2 MiniMol 复现](../../../baselines/minimol/STARLING_TABLE2_REPRODUCTION.md)：released
  CSV、作者补充的 `n_extractions` weighted-BCE 复现、结果和不可与当前 gold 混表的 lineage 边界。
- [MiniMol 入口索引](../../../baselines/minimol/README.md)：共享 head/embedding runtime、current TxAgent
  baselines、冻结 GPT-OSS-120B top-5 sensitivity 和外部 Table 2 lineage。
- [KNN-Agent Router OOF 计划](ROUTER_OOF_IMPLEMENTATION_PLAN.md)：按 task 独立训练的 train-only OOF
  数据隔离、特征、nested evaluation、运行 gate 和当前执行状态。
- [Trace 保留策略](TRACE_RETENTION.md)：最终 trace 的唯一目录、清理边界和一致性约束。
- [Final evidence surface 诊断](FINAL_EVIDENCE_SURFACE_EXPERIMENT.md)：固定 retrieval/single/group 的
  summary/card final-only 消融、metadata census、输出 lineage 和已冻结的 no-go 结论。
- [One-pass LoRA-RL](rl_lora/README.md)：single/group/final 融合、共享 reward/runtime、已停止的 hosted
  120B 与同样已停止的本地 NeMo 20B；[one-pass reasoning 合同](rl_lora/ONE_PASS_REASONING.md) 单独记录
  RL-specific prompt/data/evaluation lifecycle。
- Train-ratio final-only 诊断：`run_train_ratio_prior_experiment.py` 负责 source/train contract、恢复运行和
  versioned output；`train_ratio_prior_analysis.py` 独立完成 paired statistics、trigger audit 和 copied-artifact parity。
## 当前 Starling gold lineages

BBB 当前 paper-facing dataset 是 `experimental_meaningful_cns_access_v2`：3,667 parents，scaffold
train/valid/test 为 2,935/366/366，目标是系统给药后的 meaningful/adequate CNS access，而不是 mixed passive
permeability。Bioavailability 与 Skin 继续使用 scaffold-only `record_supported_v2`，分别为
1,674/209/209 和 1,966/245/245；Skin 每个 held-out split 只有 5 个 unavoidable singleton。旧 BBB
`record_supported_v2`、`experimental_direct_cns_v1` 和第一版 `record_agreement70_split811_v1` 都只作 historical
comparison；不合理的 exploratory `record_supported_v1` 已删除。

ClinTox 是独立的 `clinical_trial_failure_v1`，不是第四个 Starling gold task。它从冻结 AACT
toxicity-failure positives 与 SWEETLEAD/FDA-approved comparators 构造 1,428 个 parent labels；Starling
clinical/mechanistic rows 只用于 retrieval。构建、运行和审计入口为：

```text
tools/chembl_tool/tasks/clintox/build_clinical_trial_failure_benchmark.py
tools/chembl_tool/tasks/clintox/starling_retrieval.py
tools/chembl_tool/tasks/clintox/audit_clinical_trial_failure_agent.py
tools/chembl_tool/paper_experiments/starling_benchmark_matrix.py
```

```text
tools/chembl_tool/common/starling/build_record_supported_benchmark.py
tools/chembl_tool/common/starling/build_bbb_experimental_meaningful_cns_access.py
tools/chembl_tool/paper_experiments/build_starling_benchmark_indices.py
tools/chembl_tool/paper_experiments/seed_starling_matrix_reuse.py
tools/chembl_tool/paper_experiments/starling_benchmark_matrix.py
tools/chembl_tool/paper_experiments/analyze_starling_direct_significance.py
tools/chembl_tool/paper_experiments/audit_bbb_retrieval_coverage.py
tools/chembl_tool/paper_experiments/audit_skin_reasoning_bottleneck.py
tools/chembl_tool/paper_experiments/audit_starling_trace_failure_causes.py
tools/chembl_tool/paper_experiments/run_train_ratio_prior_experiment.py
tools/chembl_tool/paper_experiments/train_ratio_prior_analysis.py
```

BBB/Bio train-ratio tie-break valid 诊断可重复运行；已有完整 artifact 时 `--analyze-only` 不产生模型调用：

```bash
python -m tools.chembl_tool.paper_experiments.run_train_ratio_prior_experiment
python -m tools.chembl_tool.paper_experiments.run_train_ratio_prior_experiment --analyze-only
```

非默认 lineage 的 index/matrix 命令必须显式传 `--benchmark-data-root`、`--benchmark-lineage` 和
`--canonical-paper-root`。历史 agent artifact 只按 molecule key 和严格 stage contract 复用；target
retrieval 始终重新物化，完整复用审计保存在 target root 的 `reuse_audit/`。

## 新 v4 默认运行约定

所有 LLM 实验共享以下条件：

- 默认通过 SSH tunnel 直连 `http://127.0.0.1:50000/v1`，模型为 `nvidia/GLM-5.2-NVFP4`
- 保持历史 `--disable-thinking --reasoning-effort ""`，GLM reasoning 仍保存到 trace
- temperature 0，最大输出 20,480 tokens
- 主矩阵固定 `identity_blind + parent_disjoint`；直接排除相同 RDKit molecular parent，并只用原
  similarity threshold 以上的后续 structural analog 回填 top-k
- operational 不再预跑，不生成或依赖 `reuse_plan.json`；只作为显式 historical/deployment ablation
- endpoint 全局并发预算上限为 512；首次 512 压力运行出现 1/500 transport timeout 后，
  随后 384 的 full-flat 运行仍发生大量长 group-request timeout，因此当前单 launcher 默认使用
  单一 `--parallelism 128` global prompt pool，禁止多个
  launcher 各自再占 512
- 每个 task/query 冻结一个 single-molecule analysis，并在各检索条件间复用
- 每个 evidence group 最多检索 3 个 Morgan similarity 不低于 0.30 的 neighbor
- 共享 single、group、final、JSON validation、retry、trace 和 batch evaluation workflow

新 v4 论文主结果表使用 identity-blind 制度：

- `identity_blind`：LLM 不看 query/neighbor 的结构和身份；harness 预取结构工具结果；用于主表的
  none/direct/flat/mechanism 和 source comparison。
- `deployment_visible`：只作为显式 agentic/deployment ablation，不是默认。
- `deployment_visible_prefetched`：只作为显式 matched visibility control，不阻塞主矩阵。

实验模式定义在 `common/experiment_retrieval.py`：

- `none`：只有 query properties，不提供检索证据
- `direct`：只使用任务声明的直接结果证据
- `full_flat`：把所有任务机制证据合并到一个 group
- `full_mechanism`：使用完全相同的证据集合，但按 mechanism family 分组

每个任务只在 `experiment_config.py` 中声明 source endpoint group 如何映射到 direct 和 mechanism family。

## 冻结矩阵

运行矩阵前先构建冻结的 Starling index：

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.build_canonical_starling_source

python -m tools.chembl_tool.tasks.bbb_martins.build_starling_full_evidence_library \
  --out-dir outputs/paper/molecular_evidence_agent/evidence/bbb_starling_full \
  --workers 128

python -m tools.chembl_tool.tasks.skin_reaction.build_canonical_starling_source

python -m tools.chembl_tool.tasks.skin_reaction.build_starling_evidence_library \
  --workers 128

# Historical source profiles remain reproducible only by explicit opt-in:
# --source-profile sensitization_contact_allergy_v2
# --source-profile broad_skin_reaction_v1

python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_factor_evidence_library \
  --scope full \
  --evidence-content full \
  --out-dir outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_full_v2 \
  --workers 128

python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_factor_evidence_library \
  --scope direct \
  --evidence-content numeric_only \
  --out-dir outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_direct_numeric_v2 \
  --workers 128
```

### 数据审计与构建入口

```text
tools/chembl_tool/tasks/bioavailability_ma/build_canonical_starling_source.py
  固定 HF snapshot，将 local 中明确 absolute/oral-IV 的记录移入 canonical direct source，跨来源按
  parent+PMID 近等值 claim 一对一去重，并生成 residual exposure 与完整 partition/dedup audit。

tools/chembl_tool/paper_experiments/analyze_starling_parent_provenance.py
  重建每个 parent 的 accepted-record 和 unique-PMID 分布；其中 strict status 仅表示是否出现过两种 label。

tools/chembl_tool/paper_experiments/analyze_starling_majority_thresholds.py
  比较 50/60/70/80/90% record agreement 下的 keep/reject、label、record 和 publication 分布。

tools/chembl_tool/common/starling/build_benchmark_datasets.py
  唯一正式 gold builder：70% record-majority、精确 tie 拒绝、random/scaffold 8:1:1 split，并生成
  valid+test union 的 heldout audit artifact。

tools/chembl_tool/common/starling/build_record_supported_benchmark.py
  从冻结的 accepted binary parents 构造当前 scaffold-only v2；lexicographic MILP 先优化 held-out
  record support 和 label balance，再最大化第一版 valid overlap。

tools/chembl_tool/paper_experiments/build_starling_benchmark_indices.py
  默认读取 heldout union，从 inference evidence 删除全部 valid/test parents，构建 train-reference index。
  `--heldout-subsets test` 只用于已声明的 test post-selection sensitivity：删除 test parents、允许 valid
  parents 进入 reference pool。两种 scope 会写入不同 index version/metadata；valid-only 或重复声明会拒绝，
  `test valid` 会规范化为稳定的 `valid test` receipt。
  非默认 source lineage 通过 `--source-evidence INDEX_NAME=EVIDENCE_JSONL` 显式覆盖，historical spec 不改写。

tools/chembl_tool/paper_experiments/starling_benchmark_matrix.py
  当前正式 valid/test runner；默认 identity_blind + parent_disjoint、单一 128-slot global prompt pool，
  valid 与 test root 隔离。默认 `--reference-pool train` 会核验 index 确实排除了 valid+test；显式
  `--reference-pool train_valid` 只允许 test，并核验 index 只排除 test。MiniMol descriptor 可通过
  `--minimol-feature-root` 指向同一 scope 的 feature root。

tools/chembl_tool/paper_experiments/seed_starling_matrix_reuse.py
  为新 lineage 新建 retrieval/manifest，再按 molecule key 严格复用兼容的 single/group/final stage。

tools/chembl_tool/paper_experiments/analyze_starling_direct_significance.py
  在完整 valid predictions 上计算 direct agent 对 Morgan/MiniMol KNN 的 paired macro-F1 randomization、
  paired bootstrap CI、exact McNemar 和 Holm-adjusted p-values。

tools/chembl_tool/paper_experiments/plot_starling_model_comparison.py
  唯一 Starling model/visibility/dataset-lineage 总图入口；同 dataset 使用共享 baseline，不同 lineage 使用
  `--baseline-display series` 和 `--baseline-series-group` 合并 visibility-duplicate baseline；补充实验可用
  `--experiment-legend` 增加清晰 legend，不新增一次性绘图模块。

tools/chembl_tool/paper_experiments/starling_paired_figure.py
  总图的内部 paired-statistics TSV 校验和 CI/p-value SVG fragment；不是第二个 CLI 或独立图入口。

Reference-pool 默认保持 train：index 排除 valid+test，KNN 只读 `train.jsonl`。只有已明确标为 test
post-selection sensitivity 时，才组合使用 index builder 的 `--heldout-subsets test`、matrix 的
`--reference-pool train_valid`，以及两个 KNN 的 `--reference-splits train valid`。这些入口都会记录 scope；
matrix 还会读取 base-index metadata 反向核验。选择性删除 Bio valid-reference 的重跑没有新增 task-specific
module：先从 frozen retrieval payload 确定变化 indices，再复用 matrix 的 `--indices` 与 shared prompt pool，
最终只在 artifact analysis 合并未变化 predictions。

tools/chembl_tool/paper_experiments/router_oof/
  独立的 train-only router 实验入口；不扩展正式 matrix 的 valid/test 枚举。它按 task 分别生成
  scaffold-aware folds、过滤冻结 direct index、计算 OOF Morgan KNN、运行 gpt-oss-120b none/direct、
  汇总 pre-decision features，并用 nested folds 比较 Logistic 与小型 GBDT。当前 v2 使用双风险
  `P(agent-only)-P(KNN-only)` score、train-only feature-profile selection 和保守 KNN fallback gate；
  v1/v2 artifacts 分别位于 `router/` 与 `router_v2/`。v3 是 output-aware disagreement-only post-selector，
  保留 query/KNN/original evidence features，并逐层加入 decision-relative evidence 与 structured trace features；
  使用两个 disagreement direction thresholds，产物独立位于 `post_selector_v3/`。v3.1 再把两个 direction
  拆成独立的 calibrated ensemble heads，并加入最低 route count、Wilson lower-bound 风险门和 accuracy-first
  promotion gate；产物位于 `post_selector_v31/`。终止性 matched-size curve 与 shared-transfer diagnosis 分别
  位于 `post_selector_v31_learning_curve/` 和 `post_selector_v31_transfer/`，均严格 train-only。
```

Router 命令严格按以下顺序执行，所有产物位于
`outputs/paper/router_oof/gpt_oss_120b/scaffold/`：

```bash
python -m tools.chembl_tool.paper_experiments.router_oof prepare
python -m tools.chembl_tool.paper_experiments.router_oof build-indices
python -m tools.chembl_tool.paper_experiments.router_oof run-knn
python -m tools.chembl_tool.paper_experiments.router_oof audit
python -m tools.chembl_tool.paper_experiments.router_oof run-agent --limit 8 --parallelism 64
# pilot zero-failure 后去掉 --limit，用同一 root 和 --skip-existing 续跑全量
python -m tools.chembl_tool.paper_experiments.router_oof progress
python -m tools.chembl_tool.paper_experiments.router_oof finalize
# 在完整 v2 feature/trace 上建立独立 v3 post-selector lineage
python -m tools.chembl_tool.paper_experiments.router_oof.cli build-post-features
python -m tools.chembl_tool.paper_experiments.router_oof.cli train-post-selector
python -m tools.chembl_tool.paper_experiments.router_oof.cli evaluate-post-valid
# v3.1 direction-specific calibrated selector（复用既有 features，不发新的 LLM 请求）
python -m tools.chembl_tool.paper_experiments.router_oof.cli train-post-selector-v31
python -m tools.chembl_tool.paper_experiments.router_oof.cli evaluate-post-valid-v31
# train-only termination diagnostics；不读取 valid/test
python -m tools.chembl_tool.paper_experiments.router_oof.cli diagnose-post-v31-learning-curve --workers 16
python -m tools.chembl_tool.paper_experiments.router_oof.cli diagnose-post-v31-transfer --workers 15
```

v1–v3.1 的三个可部署 task-local router 从不共享训练 rows 或参数；唯一例外是最后的、不可部署的
shared-representation termination diagnosis，它只用于检验 transfer gate。Agent 固定为 gpt-oss-120b、
identity-blind、parent-disjoint 和 task-specific direct Starling condition；`none` 只提供 direct 必须复用的
single analysis。
`finalize` 只有在 22,065 个 samples 的 none/direct 都通过 single、expected groups、final 四层 artifact gate 后，
才会依次构建 features 和训练三个 task-local routers；未完成时直接失败，不会用 partial rows 训练。

v2 feature contract 删除 `knn_reference_size_log1p` 和确定性重复的 k=3 margin/entropy；候选 profile 为
`knn_compact`、`query_knn`、`query_knn_evidence`。训练分别拟合 agent-only 与 KNN-only risk head，routing
score 为两者概率差。Nested OOF promotion gate 要求 macro-F1 delta 的 95% paired-bootstrap lower bound > 0，
且 accuracy delta lower bound 不低于 -0.5 pp；失败时冻结 threshold=1.01，部署结果严格等于 KNN。

v3 只拟合 disagreement rows 上的 agent-win probability，并保留三档 profile（18/49/70 features），让 nested
OOF 在小 feature set、完整 evidence set 和 evidence+trace set 之间选择，而不是手工删除原始 evidence features。
Logistic/HistGBDT、profile 和两个方向阈值都只由 train OOF 决定；valid 不拟合模型、不调 threshold。当前
scaffold-valid canonical receipt 为
`outputs/paper/router_oof/gpt_oss_120b/scaffold/post_selector_valid_result_v3.json`，formal test 未运行。

v3.1 为两个 direction 分别选择 Logistic/HistGBDT 和上述三档 profile；每个 component 的 sigmoid calibrator
只读取对应 held-out fold，external score 为五个 calibrated components 的均值。Train-only risk gate 要求每个
启用方向至少 20 次 route 且 precision 的 one-sided 95% Wilson lower bound > 0.5，整体 accuracy delta 的 paired
bootstrap 95% CI lower bound > 0。Valid 仅报告 held-out evidence gate，绝不重选模型、profile 或 threshold。
Canonical receipt 为
`outputs/paper/router_oof/gpt_oss_120b/scaffold/post_selector_valid_result_v31.json`；formal test 未运行。

Matched-size curve 固定 v3.1 full-train direction specs，只改变 outer-train disagreement supervision。BBB 的
learned-vs-direction-only OR accuracy/macro-F1 增量随 budget 从 `800` 的 `-0.10/+0.50 pp` 单调改善到 full
`+0.74/+3.37 pp`，说明 task-local signal 确实 data-limited；但既有 scaffold-valid learned-vs-OR 增量仍未
显著。条件触发的 shared transfer 固定 task-balanced Logistic + 完整 generic profile，并保留 task-specific
calibration/threshold；它没有改善 Oral/Skin，termination gate 因此冻结为 `stop_router_main_method`。该实验
仍以 agent-win 为监督，不能称为 counterfactual evidence utility。不得继续用 valid 搜索 shared family/profile，
也不启动 router formal test。

```bash
python -m tools.chembl_tool.paper_experiments.molecular_evidence_agent --list
```

新 v4 runner 已实现 `identity_blind + parent_disjoint` fresh-run，并将 endpoint/model/reasoning、
512 全局并发预算和 valid/test 隔离接入统一 CLI。默认先跑 valid；冻结设置后显式加
`--evaluation-subset test` 跑正式 test。命令形状为：

```bash
python -m tools.chembl_tool.paper_experiments.starling_benchmark_matrix \
  --benchmark-split <random|scaffold> \
  --visibility-mode identity_blind \
  --neighbor-identity-policy parent_disjoint \
  --parallelism 128
```

### 全局 prompt ready pool 与断点恢复

matrix 只保留全局 prompt ready pool：

```text
tools/chembl_tool/common/task_workflows/global_prompt_pool.py
  跨 batch/task/condition 的 ready queue、公平 seed 交错和唯一并发预算。
tools/chembl_tool/common/task_workflows/reasoning_stage_runtime.py
  single/group/final checkpoint DAG、依赖失效、原子 artifact 和 trace 发布。
```

```bash
python -m tools.chembl_tool.paper_experiments.starling_benchmark_matrix \
  --benchmark-split <random|scaffold> \
  --evaluation-subset <valid|test> \
  --visibility-mode identity_blind \
  --neighbor-identity-policy parent_disjoint \
  --parallelism 128 \
  --max-stage-requeues 1
```

该模式把所有 task/condition 的 ready single、group 和 final branch 放入同一个 pool；task/condition 只作为
prompt adapter、artifact key 和 provenance，不再拥有固定 lane。final 等待相应 single 和全部 expected group
成功后动态入队。全新 sample 先用 `--prepare-only` 只冻结 retrieval，不发 LLM 请求，随后从第一条
single/group prompt 起进入全局 pool。跨 condition 的 frozen-single 依赖按 sample 解锁，不再等待整个 `none`
condition 批次结束。成功 branch 原子写回原 canonical 文件；重启同一命令时，完整 sample 直接恢复，部分完成
sample 只补失败 stage。run manifest 同时冻结 `expected_group_ids`；恢复和完成判定校验精确 group ID 集合，
不会把数量相同但重复或被替换的 group 误判为完成。输出仍是原有 `retrieval.json`、`single_molecule_reasoning_output.json`、
`group_reasoning_outputs.jsonl`、`final_reasoning_output.json`、`trace_messages.jsonl`、predictions/metrics/report；
新增的 `stage_pool` manifest provenance 和 `.run.lock` 只用于审计恢复行为。

同一 output root 只能运行一个 matrix/watchdog；断点恢复复用相同 root、batch ID、模型和 prompt 参数。

### Coverage retrieval 与 reasoning context 插件

Neighbor set 选择和 LLM 输入增强是两个互不绑定的开关：

- `--neighbor-selector similarity|query_feature_coverage` 决定 eligible candidates 中选哪三个 molecule；
- `--neighbor-context-profile standard|coverage_aware|coverage_mmp_ledger` 决定是否给 group branch 增加
  analog-set coverage 信息。

默认 `similarity + standard` 路径保持原有 Morgan/Tanimoto retrieval 和 prompt 不变。`coverage_aware` 只增加
匿名统计：query Morgan feature/atom-environment 总覆盖率、每个 neighbor 的 shared/marginal/redundant coverage、
累计覆盖率和 marginal region size。它不发送 SMILES、fingerprint bit ID、原子元素标签或 molecule identity，
不改变 retrieval.json、task-specific JSON schema、final prompt 或 tool-prefetch/cache。当前该 profile 只支持
Morgan retrieval feature；任何非默认 selector/profile 都必须显式指定独立 `--output-root`，runner 会阻止其写入
canonical result root。

Visible-only `coverage_mmp_ledger` 是另一条平行的 opt-in 输入路径。它不重新实现结构匹配，而是对选中的每个
neighbor 预取现有 `mmp_structure_compare` 文本，让 group branch 同时看到 MCS coverage、可用的 MMP
shared constant / transformation，以及 Morgan feature 的逐 rank marginal/redundant 统计。Morgan 统计只衡量
集合互补性，不能定位 fragment 或宣称 atom coverage；没有 matched-pair transformation 时具体 region mapping
必须标为 unresolved。该 profile 不支持 `identity_blind`，也不改变 retrieval.json、neighbor set、默认 prompt、
task schema 或普通 tool cache。

Matched prompt ablation 应固定同一个 selector，只改变 context profile。例如：

```bash
python -m tools.chembl_tool.paper_experiments.starling_benchmark_matrix \
  --benchmark-split scaffold \
  --evaluation-subset valid \
  --visibility-mode identity_blind \
  --neighbor-identity-policy parent_disjoint \
  --neighbor-selector query_feature_coverage \
  --neighbor-context-profile <standard|coverage_aware> \
  --output-root <model-and-profile-specific-root> \
  --single-analysis-root <frozen-model-root>/runs_identity_blind_parent_disjoint \
  --experiments \
    bbb_martins__starling_full_mechanism \
    bioavailability_ma__starling_full_mechanism \
    skin_reaction__starling_full_mechanism
```

Visible MMP-ledger matched ablation 使用相同 selector，并将上述 visibility/profile 两项改为：

```bash
  --visibility-mode deployment_visible \
  --neighbor-context-profile coverage_mmp_ledger
```

实现入口为 `common/neighbor_selection.py`、`common/coverage_reasoning.py`、
`common/identity_blind.py::prepare_reasoning_retrieval` 和
`common/reasoning_calls.py::call_group_branch`。Matrix、batch 和 single-run manifest 都保存 selector/profile，
便于从 trace 审计实际输入；group artifact reuse 也要求 source/target profile 一致，防止把 standard branch
误复用到 coverage-aware run。

Operational 和 deployment-visible 以后只能显式 opt in，并写入独立 historical/ablation root：

```bash
python -m tools.chembl_tool.paper_experiments.molecular_evidence_agent \
  --visibility-mode deployment_visible \
  --neighbor-identity-policy operational \
  --experiments bioavailability_ma__none bioavailability_ma__starling_full_mechanism
```

Starling v4 的 visible parent-disjoint ablation 可以 fresh-run，不依赖 operational artifact 或 reuse plan；
必须同时显式传入两项 policy，并使用独立 model-specific output root：

```bash
python -m tools.chembl_tool.paper_experiments.starling_benchmark_matrix \
  --benchmark-split scaffold \
  --evaluation-subset valid \
  --visibility-mode deployment_visible \
  --neighbor-identity-policy parent_disjoint \
  --output-root <model-specific-visible-parent-disjoint-root> \
  --base-url <openai-compatible-endpoint> \
  --model <served-model>
```

输出位于 `<output-root>/runs_deployment_visible_parent_disjoint/`，manifest 必须记录
`fresh_parent_disjoint=true` 和 `operational_staging_used=false`。该 ablation 不改变默认正式主结果仍为
`identity_blind + parent_disjoint` 的约定。请求 timeout 默认 300 秒；只有在首轮 metrics 证明失败属于
transport timeout 时，才用 `--timeout-s 600` 重新启动 matrix 做 targeted repair（matrix 内置
`--skip-existing`），不能借此改变 prompt、证据或其它 inference setting。

当前正式 root 按 task lineage 分开：BBB 使用 `experimental_meaningful_cns_access_v2`，Bioavailability/Skin
使用 `record_supported_v2`；各自的 agent batch 均位于对应 root 的
`runs_identity_blind_parent_disjoint/`。先在 valid 做 completeness/contract 检查，冻结设置后再运行 test；
test 不用于模型、prompt、threshold 或 label-policy 选择。

以下旧 TDC validation 流程仅保留作历史复现，不是 v4 默认。需要以完全相同的旧设置诊断重跑时，为
matrix、prefetch audit 和汇总命令统一加
`--split valid`。输入自动从各任务的 `valid.jsonl` 读取，全部产物隔离写入
`outputs/paper/molecular_evidence_agent_valid/`；默认不加参数时仍使用 test split 和原结果 root。例如：

```bash
python -m tools.chembl_tool.paper_experiments.molecular_evidence_agent \
  --split valid \
  --visibility-mode deployment_visible \
  --neighbor-identity-policy operational \
  --experiments bioavailability_ma__none bioavailability_ma__starling_full_mechanism

python -m tools.chembl_tool.paper_experiments.summarize_results --split valid
python -m tools.chembl_tool.paper_experiments.audit_prefetch_contract --split valid
python -m tools.chembl_tool.paper_experiments.parent_disjoint_ablation --split valid --materialize
```

2026-07-23 已扩展完成该 valid 诊断矩阵：identity-blind、matched-prefetch 和 deployment-visible
各 26 个条件、2,713 个 sample-condition，均为完整样本且 0 失败；22 个 parent-disjoint 条件、
2,275 个 sample-condition 也全部完成。主报告位于
`outputs/paper/molecular_evidence_agent_valid/analysis/report.md`，parent-disjoint 配对审计位于
`outputs/paper/molecular_evidence_agent_valid/analysis/parent_disjoint_ablation/result_report.md`。

当前 paper/Starling runner 默认使用本机 tunnel `http://127.0.0.1:50000/v1`、
`nvidia/GLM-5.2-NVFP4` 和历史一致的 `--disable-thinking --reasoning-effort ""`。空值使 client 不发送
`reasoning_effort` 参数，但 GLM 返回的 reasoning 仍会保存在 trace。运行前用 `ssh -fNT parcc-glm` 建立 tunnel；
loopback vLLM 无鉴权时，runner 会在进程内为 `GLM_LOCAL_API_KEY` 注入非敏感占位值。
旧 LiteLLM 只作为显式 fallback，使用 `--api-key-env GLM_API_KEY --base-url
https://litellm.parcc.upenn.edu/v1 --model zai-org/GLM-5.2-FP8 --reasoning-effort ""`。
不得将任何真实 API key 的值写入命令记录、manifest 或 trace。

长矩阵可使用 `watch_glm_tunnel_and_matrix.py` 同时监控 `/v1/models`、SSH tunnel 和唯一 matrix launcher：

```bash
python -m tools.chembl_tool.paper_experiments.watch_glm_tunnel_and_matrix \
  --benchmark-split scaffold --evaluation-subset valid \
  --visibility-mode deployment_visible --neighbor-identity-policy parent_disjoint \
  --output-root <model-specific-output-root> \
  --expected-results 6887 \
  --parallelism 128 --max-stage-requeues 1 \
  --timeout-s 300 \
  --watchdog-log <watchdog.log> --launcher-log <launcher.log>
```

Watchdog 完成计数不是简单统计 `final_reasoning_output.json`：它要求 task prediction 可归一化、single/final
均为 `status=ok`、group 数与 manifest 的 `n_groups_with_neighbors` 完全相等且没有失败 group。Tunnel 失效时
它终止当前 matrix process group，重建 tunnel 后由 `--skip-existing` 恢复；它不会保存或重放密码，Duo Push
仍需用户批准。若长尾修复需要 600 秒 request timeout，必须在 watchdog 和 matrix 两边都显式传
`--timeout-s 600`。

同一入口也可监控其它支持 `--output-root` 且能从成功 stage 断点恢复的 Python launcher。使用
`--launcher-module`、`--launcher-args-json` 和 `--completion-mode recursive_reasoning_runs`；通用完成 gate
要求 single/final、精确 expected group set 全部成功，但不硬编码 task prediction 字段。正在运行且省略了
`--output-root` 的 singleton launcher 只能显式加 `--allow-implicit-output-root` 后接管；watchdog 自身不会被
误识别为 launcher。`--ssh-local-forward` 同时冻结重建用的 `-L` 参数和允许替换的 SSH tunnel 范围。
Endpoint 必须连续达到 `--unhealthy-threshold` 次失败才会停止 launcher，避免一次短暂探测失败造成中断。

V4 parent-disjoint 使用公共 identity normalizer、retrieval policy、top-k backfill 和 manifest provenance，
但直接 fresh-run，不读取 operational retrieval，也不生成 reuse plan。汇总必须从最终 `retrieval.json`
逐条验证 parent conflict=0、held-out overlap=0、threshold violation=0，并验证 identity-blind leak=0。
不得降低 `min_similarity`、删除受影响样本或只在 supported/overlap 子集上计算主指标。

## 汇总分析

```bash
# 旧 TDC visibility matrix only
python -m tools.chembl_tool.paper_experiments.summarize_results

python -m tools.chembl_tool.paper_experiments.plot_retrieval_claims_overview
```

以下 operational/parent 配对汇总只用于旧 lineage：

```bash
python -m tools.chembl_tool.paper_experiments.summarize_parent_disjoint_results \
  --operational-root outputs/paper/molecular_evidence_agent_valid/runs_deployment_visible \
  --parent-root outputs/paper/molecular_evidence_agent_valid/runs_deployment_visible_parent_disjoint \
  --analysis-dir outputs/paper/molecular_evidence_agent_valid/analysis/parent_disjoint_ablation

python -m tools.chembl_tool.paper_experiments.plot_retrieval_claims_overview \
  --analysis-dir outputs/paper/molecular_evidence_agent_valid/analysis \
  --output outputs/paper/molecular_evidence_agent_valid/analysis/figures/retrieval_claims_overview.svg \
  --png-output outputs/paper/molecular_evidence_agent_valid/analysis/figures/retrieval_claims_overview_highres.png \
  --data-split valid
```

Performance 可视化统一使用上述横向 grouped-bar chart。每个 split 的正式 figures
目录只保留 `retrieval_claims_overview.svg` 和 `retrieval_claims_overview_highres.png`；不保留
preview/QA 导出或另一套 overview 绘图代码。

### Matched train-label direct agent vs Morgan KNN（valid-only 历史诊断）

`matched_train_label_agent/` 是与正式 Starling matrix 隔离的 valid-only 诊断。Materializer 将当前 Morgan
KNN 的 exact top-3 scaffold-train neighbors、顺序、相似度和 `Y` 原样物化成一个 label-visible direct group；
agent 不读取其它 Starling records。Launcher 复用各 task 已冻结的 `none` single branch，并在一个 global
prompt pool 中运行 group/final；summarizer 强制逐 query retrieval parity 后生成 paired JSON/TSV/中文报告。
当前 scaffold-valid 已完成 820/820、0 failure、0 retrieval mismatch、0 prompt identity leak。BBB/Skin 的
macro-F1 point estimate 较 KNN 为 `+0.0514/+0.0408`，但区间均跨 0；Bio 为 `-0.0595`，且 accuracy
下降 `0.1818`。因此 valid gate 为 no-go，formal test 不运行；完整解释以
`STARLING_BENCHMARK_RESULTS.md` 为准。

目录内当前/历史模块边界见 `matched_train_label_agent/README.md`。下列第一组命令复现 E15 和 Bio
diagnosis；不会修改当前默认 prompt：

```bash
python -m tools.chembl_tool.paper_experiments.matched_train_label_agent.materialize
python -m tools.chembl_tool.paper_experiments.matched_train_label_agent.run --limit 1
python -m tools.chembl_tool.paper_experiments.matched_train_label_agent.run
python -m tools.chembl_tool.paper_experiments.matched_train_label_agent.summarize
python -m tools.chembl_tool.paper_experiments.matched_train_label_agent.compare_full_pool
python -m tools.chembl_tool.paper_experiments.matched_train_label_agent.diagnose_bio_unanimous_positive
```

下列第二组仅复现已终止的 BBB E16，不是 continuation plan：

```bash
python -m tools.chembl_tool.paper_experiments.matched_train_label_agent.bbb_property_compatibility_audit
python -m tools.chembl_tool.paper_experiments.matched_train_label_agent.bbb_property_compatible_experiment all
python -m tools.chembl_tool.paper_experiments.matched_train_label_agent.bbb_property_compatible_trace_diagnosis
```

其中 `diagnose_bio_unanimous_positive` 是 deterministic Bio trace audit：只读取既有 valid predictions、single/group/final traces、
full-pool direct 对照和 frozen gold audit，不调用模型。它将三邻居全为 `Y=1` 的 109 个样本分成误伤、救回、
正确保留和错误保留四类，并把逐样本 provenance 与汇总写到
`analysis/bio_unanimous_positive_diagnosis/`。

E16 availability command 不调用 LLM。它验证正式 Morgan
top-3 parity 后，在同一 BBB train pool 的 top-20 内测量 ionization/property-compatible neighbors 的可用性；
selector 不读取 label，valid label 只用于冻结选择后的描述性 vote audit。产物写入
`outputs/paper/bbb_property_compatibility_availability_experimental_meaningful_cns_access_v2_valid/`。当前最保守的
0.02 similarity-cost budget 将 KNN vote macro-F1 从 0.5896 提高到 0.6335，但 paired CI 仍轻微跨 0。

E16 experiment command 是该 audit 唯一运行过的 frozen 0.02 matched-v3 valid candidate；`all` 依次 materialize、
run、summarize，并可通过三个 action 单独断点重跑。它已完成 366/366、0 failure，macro-F1 仅从 0.6696
变为 0.6708（paired delta `+0.0012`，95% CI `[-0.0410,+0.0450]`），promotion gate 失败。产物写入
`outputs/paper/matched_train_label_direct_agent_bbb_property_compatible_v1_scaffold_valid_gpt_oss_120b/`；不得运行
formal test 或继续调 selector。详细 trace-control 统计见 `STARLING_BENCHMARK_RESULTS.md`。

E16 diagnosis command 不调用模型；它严格区分 exact ordered input、same-membership reorder 与真实
membership change，并配对 group core、final state、neighbor vote 和 gold。产物写入上述 candidate root 的
`analysis/trace_diagnosis/`。

该 audit 之后冻结的唯一 Bio 修复是 task-local prompt profile `f20_evidence_calibrated_v2`。它不改变 retrieval，
只要求 evidence direction 与 transferability 分开，且 `low` 必须有明确 `F<20%` 证据。GPT-OSS-120B
scaffold-valid 的 matched/full-direct/full-flat/full-mechanism macro-F1 分别从
`0.5261/0.5577/0.6117/0.5869` 提高到 `0.6140/0.6375/0.7004/0.6844`；full-flat 和
full-mechanism 的 paired 95% CI 均高于 0。所有 run 为 209/209、0 failed，evidence-bearing pair 的
retrieval SHA mismatch 为 0。`none` 降到 `0.4129` 且几乎全预测 high，因此它是 evidence adjudication
修复，不是无证据 prior。2026-08-10 冻结设置下的首次 Bio scaffold-test 已完成：full-flat/full-mechanism
macro-F1 为 `0.6663/0.6720`，Morgan KNN、MiniMol embedding KNN、MiniMol trained head 分别为
`0.6801/0.6484/0.7027`；五项均覆盖同一209条 test，两个 agent batch 为0 failed。agent root 使用 canonical
`molecular_evidence_agent_starling_scaffold_record_supported_v2`，三个 baseline 分别写入带
`_record_supported_v2_test` 后缀的独立 root。详细合同、paired CI、p-value 和 frozen paths 仍以
`STARLING_BENCHMARK_RESULTS.md` 为准；不得根据 test 继续调参。

上述约束只针对旧 TDC `test|valid` lineage。Starling `random|scaffold` 使用独立的汇总、图表和
output root，不能写入或替代上述 TDC figures：

```bash
python -m tools.chembl_tool.paper_experiments.summarize_starling_benchmark \
  --splits scaffold --evaluation-subset valid \
  --pipeline-root <model-specific-v4-root> \
  --minimol-root outputs/baselines/minimol_starling_valid \
  --structure-knn-root outputs/baselines/structure_knn_starling_valid \
  --minimol-embedding-knn-root outputs/baselines/minimol_embedding_knn_starling_valid \
  --output-dir <model-specific-summary-root>
python -m tools.chembl_tool.paper_experiments.plot_starling_benchmark_overview \
  --png-output outputs/paper/starling_benchmark_results/figures/starling_benchmark_overview_highres.png

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

`summarize_results.py` 仍按旧 TDC `experiments_for_split(test|valid)` 和 legacy visibility roots 扫描；它不能
用于 v4 `runs_identity_blind_parent_disjoint/`。V4 performance 只能由
`summarize_starling_benchmark.py` 读取。对显式 v4/valid pipeline，三个 baseline root 不再隐式回退到历史
strict-conflict test 输出；需要 baseline 时必须像上面一样显式传入同 lineage/subset roots，否则生成
pipeline-only summary。

这张 `starling_model_comparison` 是 model、baseline 和 matched ablation 的唯一正式总图。新增完整
model/visibility summary 重复添加 `--comparison-metrics <metrics.tsv>`；局部 matched 方法重复添加
`--experiment-metrics <metrics.tsv>`，不再为单个实验创建独立 overview/bar chart。comparison summary
必须与 reference 的 split/subset/sample count 和 baseline 一致。每个追加 experiment TSV 的每个 task/split
需要一个 `comparison_role=anchor` 行（历史 `morgan_standard` 自动视为 anchor）；可选
`base_method` 指定已有 condition，`base_model_label` 指定主 candidate 或通过 `--comparison-metrics` 加载的
其它 model/visibility series。绘图器先校验 subset、样本数和 macro-F1，
再隐藏这个重复 anchor，只增加新的实验行。
`--paired-ci-metrics` 提供每个 task 的 best agent 相对 train-derived baselines 的 paired 统计量；historical
artifact 可以保留 MiniMol train-all + Morgan KNN 两项，current figure 必须同时包含 MiniMol train-all、Morgan
KNN 和 MiniMol embedding cosine KNN 三项。当指定 `--paired-significance-display pvalue` 时，正式总图只显示
单侧 paired-permutation p-value（`H1: best agent > baseline`），不显示 95% CI。当前方向是在看到
valid 结果后确定，best agent 也由同一 valid set 选出，因此图中明确标注为 exploratory。
输入必须与图中的 best agent、baseline、subset 和样本数一致。

当 current release 的不同 task 已迁移到不同 dataset/prompt lineage 时，先用
`summarize_starling_benchmark.py --task-metrics task=metrics.tsv` 逐 task 选择各自完整、lineage-matched 的
summary，再用同一绘图器的 `--single-series` 模式绘制一个 latest agent series 和三类 train-label baseline。
该模式允许 partial current matrix，但仍要求每个 task 有结果、method 名称属于冻结矩阵且所有纳入行通过
failure gate。额外 condition 的既有 metrics 可用显式
`--condition-metrics task__condition=metrics.json` 纳入，输出会保留原 artifact path。

当前 BBB experimental-v2、Bio f20-v2、Skin aligned-v2 的合并 summary 已包含同 setting 的 9 个 ChEMBL
conditions。ChEMBL matrix artifact root 为
`outputs/paper/molecular_evidence_agent_chembl_scaffold_current_latest_valid_gpt_oss_120b/`；三个 task 合计
`2460/2460` final 完整。合并 summary 与图位于：

```bash
python -m tools.chembl_tool.paper_experiments.analyze_starling_best_agent_baselines \
  --metrics outputs/paper/starling_benchmark_results_scaffold_current_latest_valid_gpt_oss_120b/metrics.tsv \
  --output-prefix outputs/paper/starling_benchmark_results_scaffold_current_latest_valid_gpt_oss_120b/analysis/best_agent_paired_all_baselines

python -m tools.chembl_tool.paper_experiments.plot_starling_model_comparison \
  --reference-metrics outputs/paper/starling_benchmark_results_scaffold_current_latest_valid_gpt_oss_120b/metrics.tsv \
  --single-series \
  --paired-ci-metrics outputs/paper/starling_benchmark_results_scaffold_current_latest_valid_gpt_oss_120b/analysis/best_agent_paired_all_baselines.tsv \
  --paired-significance-display pvalue \
  --chart-title 'Current Starling Scaffold-Valid Results' \
  --chart-subtitle 'Latest task contracts · Macro-F1 with lineage-matched train-label baselines' \
  --comparison-title 'GPT-OSS-120B latest agent' \
  --context-line 'BBB: meaningful CNS access v2 (n=366) · Bioavailability: record-supported v2 + f20-calibrated prompt (n=209)' \
  --context-line 'Skin Reaction: record-supported v2 + sensitization-aligned prompt (n=245)' \
  --context-line 'Identity-blind · parent-disjoint retrieval · external ChEMBL may include same-scaffold analogs · validation only' \
  --output outputs/paper/starling_benchmark_results_scaffold_current_latest_valid_gpt_oss_120b/figures/starling_model_comparison.svg \
  --png-output outputs/paper/starling_benchmark_results_scaffold_current_latest_valid_gpt_oss_120b/figures/starling_model_comparison_highres.png
```

Analyzer 从 metrics 自动选择每个 task 的最高 macro-F1 agent，按 molecule+gold 严格对齐三个 baseline，运行
100,000 次 one-sided paired permutation。JSON/TSV 同时保存 10,000 次 paired-bootstrap CI 和跨全部 9 项
比较的 Holm p-value；图沿用历史约定显示 raw exploratory p-value，不能解释为 confirmatory inference。
若 best agent 来自总图追加的 experiment TSV，而不是主 metrics series，使用
`--agent-metrics <experiment_metrics.tsv> --agent-model <exact model_label>`；analyzer 仍从主 metrics 读取并
校验同 task/split/subset 的三个 baseline。绘图器随后会反向校验所选 experiment row 的 model、method、n 和
macro-F1，避免把未画出的条件或错位样本统计量放进图中。
当前 `parent_disjoint` 不等于 scaffold-disjoint：它只排除 exact、same-connectivity 和 same-parent。外部
ChEMBL 同-scaffold analog 可以进入检索，因此 ChEMBL 与 train-only scaffold KNN 的数据可见性不完全对称。
共享 identity policy registry 另提供 opt-in `scaffold_disjoint`：在 `parent_disjoint` 上额外排除标准化 parent
的相同非空 Bemis–Murcko scaffold，并保持 similarity threshold 内 top-k 回填；空 scaffold 的无环分子不互相
排除。它适用于任意 task/source，fresh runs 写入 `runs_<visibility>_scaffold_disjoint/`，不得覆盖 canonical
`parent_disjoint` roots。当前计划只允许把它用于 **ChEMBL direct**；Starling direct 和所有 full-flat/
full-mechanism condition 保持 `parent_disjoint`。2026-08-10 曾误把该 policy 应用于 18-batch 全矩阵，以下 root
仅作为 overbroad diagnostic archive，不得作为正式 matched-source comparison：

```text
outputs/paper/molecular_evidence_agent_scaffold_disjoint_source_matched_current_valid_gpt_oss_120b/

python -m tools.chembl_tool.paper_experiments.summarize_results \
  --split valid \
  --paper-root outputs/paper/molecular_evidence_agent_scaffold_disjoint_source_matched_current_valid_gpt_oss_120b \
  --neighbor-identity-policy scaffold_disjoint \
  --bootstrap-replicates 10000
```

三个 task 的精确 benchmark/canonical-index/single-root/model 参数保存在同一 output root 的
`experiment_matrix_identity_blind_scaffold_disjoint_*.json` manifests；结果、设置纠正以及 Bio identical-input
full-flat repeat 的波动审计见 `STARLING_BENCHMARK_RESULTS.md`。

```text
outputs/paper/starling_benchmark_results_scaffold_current_latest_valid_gpt_oss_120b/
  metrics.tsv
  summary.json
  report.md
  analysis/best_agent_paired_all_baselines.{tsv,json,md}
  figures/starling_model_comparison.svg
  figures/starling_model_comparison_highres.png
```

同一模型跨 dataset lineage 的描述性比较仍使用这个入口：通过
`--method-family molecular_evidence_agent` 排除因训练 split 改变而不可共享的 baselines，按输入顺序重复
`--series-label` 区分 dataset/visibility，并用 `--chart-title`、`--chart-subtitle`、重复
`--context-line` 和 `--comparison-title` 把新数据集的构建特征与非 paired 限制写进图内。各 metrics 仍须有
相同 task、method、evaluation subset 和样本数；不同 valid molecules 的差值只能作描述性比较。
如果要同时比较各 lineage 在各自 train split 上训练的 MiniMol/Morgan baselines，则不使用
`--method-family`，并显式设置 `--baseline-display series`；默认 `shared` 仍要求 baseline 指标完全相同，防止
普通 model comparison 把不同训练口径误画成共享 baseline。
当多个 agent series 只是同一 dataset lineage 的不同 visibility 时，按 metrics 顺序重复
`--baseline-series-group <lineage>`；绘图器会先验证组内 baseline 完全一致，再只画一条 lineage baseline，避免
把与 visibility 无关的同一结果重复展示。

当前新旧 dataset + baseline 对比图的完整复现命令：

```bash
python -m tools.chembl_tool.paper_experiments.plot_starling_model_comparison \
  --reference-metrics outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_120b/metrics.tsv \
  --candidate-metrics outputs/paper/starling_benchmark_results_scaffold_record_supported_v2_valid_gpt_oss_120b_blind/metrics.tsv \
  --comparison-metrics outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_120b_visible_parent_disjoint/metrics.tsv \
  --comparison-metrics outputs/paper/starling_benchmark_results_scaffold_record_supported_v2_valid_gpt_oss_120b_visible/metrics.tsv \
  --series-label 'Original v1 · Blind' \
  --series-label 'Record-supported v2 · Blind' \
  --series-label 'Original v1 · Visible' \
  --series-label 'Record-supported v2 · Visible' \
  --baseline-display series \
  --baseline-series-group old --baseline-series-group new \
  --baseline-series-group old --baseline-series-group new \
  --chart-title 'GPT-OSS-120B and Baselines Across Starling Dataset Versions' \
  --chart-subtitle 'Scaffold valid · Parent-disjoint retrieval · Macro-F1' \
  --comparison-title 'Original 70%-agreement v1 vs record-supported v2' \
  --context-line 'Record-supported v2 prioritizes scaffold-disjoint assignment and held-out molecules with at least 2 accepted records.' \
  --context-line 'New valid multi-record coverage: BBB 500/500 · Bioavailability 209/209 · Skin 240/245; train/valid/test scaffold overlap = 0.' \
  --context-line 'Baselines use each lineage-specific train split and are shown once per lineage; visibility is not applicable.' \
  --context-line 'Old and new valid sets have equal sizes but different molecules; dataset-version changes are descriptive, not paired estimates.' \
  --output outputs/paper/starling_benchmark_results_scaffold_record_supported_v2_valid/figures/gpt_oss_120b_dataset_version_comparison.svg \
  --png-output outputs/paper/starling_benchmark_results_scaffold_record_supported_v2_valid/figures/gpt_oss_120b_dataset_version_comparison.png
```

Starling 图从统一 `metrics.tsv` 读取。现有 parent-disjoint agent conditions、MiniMol train-all head、
Morgan KNN 和 MiniMol embedding cosine KNN 的旧 random/scaffold test 图属于上一版 strict-conflict lineage；
第一版 70% record-majority 8:1:1 lineage 已完成 GLM、GPT-OSS-20B、GPT-OSS-120B scaffold-valid blind matrix、
GPT-OSS 两套和 GLM 一套 visible matrix、三种 matched baseline，以及 coverage-aware/MMP-ledger valid ablations；
单模型和总图使用独立 output roots，不得与旧 test 图混表。GLM visible 已在完整 gate 后加入上述同一
blind+visible 总图；不得追加未完成运行的中间指标。
三种 baseline 的 v4 valid CLI 共同要求显式传入 split 和隔离 output root：

```bash
python -m baselines.minimol.run_bioavailability_ma \
  --data-dir data/processed_starling/<Task>/scaffold \
  --output-dir outputs/baselines/minimol_starling_valid/<Task>/scaffold \
  --train-all --evaluation-split valid

python -m baselines.structure_knn.run \
  --data-dir data/processed_starling/<Task>/scaffold \
  --output-dir outputs/baselines/structure_knn_starling_valid/<Task>/scaffold \
  --k 3 --evaluation-split valid

python -m baselines.minimol.run_embedding_knn \
  --data-dir data/processed_starling/<Task>/scaffold \
  --embedding-cache-dir outputs/baselines/minimol_starling_valid/<Task>/scaffold/embeddings \
  --output-dir outputs/baselines/minimol_embedding_knn_starling_valid/<Task>/scaffold \
  --k 3 --evaluation-split valid
```

MiniMol head 的 train-only scaffold CV 诊断入口为 `python -m baselines.minimol.run_train_cv`，完整合同、
scheduler 修复、regularization sweep 和不替换 canonical baseline 的结论记录在
`baselines/minimol/HEAD_TRAINING_DIAGNOSTICS.md`。该入口只读取 train embedding cache，不把 outer valid/test
用于 epoch selection。

MiniMol 入口总索引见 `baselines/minimol/README.md`。外部 Starling 论文 Table 2 的 released-CSV 复现使用
独立的 `run_starling_table2.py` 和 output root；其任务定义与 split 不属于本项目当前 gold，结果不进入本页
v4 总账或 canonical figures。

MiniMol head 可重复传 `--reuse-embedding-cache-dir`，按 exact SMILES 从既有 molecule-only cache 复用
embeddings；命中与未命中统计进入 `metrics.json`。valid-only run 的 canonical 指标键是
`evaluation_metrics` / `evaluation_metrics_fixed_0.5`；历史 `test_metrics*` alias 仅为兼容旧汇总器保留。
完整口径、结果和 baseline 入口见
[`STARLING_BENCHMARK_RESULTS.md`](STARLING_BENCHMARK_RESULTS.md)。

`audit_prefetch_contract` 同时输出逐样本 `prefetch_contract_audit.tsv` 和逐 condition
`prefetch_contract_conditions.tsv`。当前 21 条件矩阵只有在 `n_conditions_complete=21`、
test 上 `n_expected=n_audited=4456`，valid 上 `n_expected=n_audited=2203`；两者都只有在
missing/extra/mismatch 均为 0 时才通过。审计同时要求 raw retrieval
精确一致，并在规范化 identity redaction 后比较 prefetched tool outputs；只比较两边已有样本的交集
不构成完整审计。

该命令生成每个条件的指标和 95% bootstrap 区间、retrieval coverage、token/retry 计数、可见性审计、制度内和跨制度的配对 macro-F1 差异，以及精确 McNemar 检验：

```text
outputs/paper/molecular_evidence_agent/analysis/
```

冻结的 full-run 结果和解释见 [RESULTS.md](RESULTS.md)。

Bioavailability scalar KNN 对照单独运行，因为它只使用数值 direct-F 且不进行 LLM reasoning：

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.run_scalar_knn
```

## 查看最终 paper trace

```bash
bash tools/trace_viewer/start_viewer.sh 8776
```

Viewer 只扫描 `outputs/paper/molecular_evidence_agent/` 下四个正式 root：`runs`、
`runs_deployment_visible_prefetched`、`runs_deployment_visible` 和
`runs_deployment_visible_parent_disjoint`。它按 condition 读取轻量预测列表，再按样本加载唯一的 per-run
`trace_messages.jsonl`、`retrieval.json`、manifest 和可选 `reuse.json`；不再支持旧 task-specific reasoning
目录或旧 Tier/policy 专用渲染。
