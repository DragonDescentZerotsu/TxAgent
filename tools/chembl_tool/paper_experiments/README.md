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
- [Trace 保留策略](TRACE_RETENTION.md)：最终 trace 的唯一目录、清理边界和一致性约束。

## 新 v4 默认运行约定

所有 LLM 实验共享以下条件：

- PARCC LiteLLM is the default: top-level launchers load `LITELLM_BASE_URL` and `LITELLM_API_KEY` from the
  ignored root `keys.py`, and request `nvidia/GLM-5.2-NVFP4`
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

python -m tools.chembl_tool.tasks.skin_reaction.build_starling_evidence_library \
  --out-dir outputs/paper/molecular_evidence_agent/evidence/skin_reaction_starling_full \
  --workers 128

python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_factor_evidence_library \
  --scope full \
  --evidence-content full \
  --out-dir outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_full_v3 \
  --workers 128

python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_factor_evidence_library \
  --scope direct \
  --evidence-content numeric_only \
  --out-dir outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_direct_numeric_v3 \
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

tools/chembl_tool/paper_experiments/build_starling_benchmark_indices.py
  读取 heldout union，从 inference evidence 删除全部 valid/test parents，构建 split-specific index。

tools/chembl_tool/paper_experiments/starling_benchmark_matrix.py
  当前正式 valid/test runner；默认 identity_blind + parent_disjoint、单一 128-slot global prompt pool，
  valid 与 test root 隔离。
```

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

新正式 root 固定为各 `record_agreement70_split811_v1` lineage 下的
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

The paper/Starling launchers now default to the PARCC LiteLLM URL and credential stored as
`LITELLM_BASE_URL` and `LITELLM_API_KEY` in the ignored root `keys.py`. They request
`nvidia/GLM-5.2-NVFP4` and preserve the historical `--disable-thinking --reasoning-effort ""` request contract.
Those flags omit optional request controls; they do not disable provider-default GLM reasoning, which remains in
the trace. Launchers inject the key into their child environment and persist only the environment-variable name.

The direct loopback route remains available with explicit `--base-url http://127.0.0.1:50000/v1` and
`--api-key-env GLM_LOCAL_API_KEY`. Establish `ssh -fNT parcc-glm` before selecting it; unauthenticated loopback
vLLM receives only the existing non-sensitive placeholder.

For the explicit loopback route, `watch_glm_tunnel_and_matrix.py` can monitor `/v1/models`, the SSH tunnel, and
the single matrix launcher. It remains tunnel-only and must not wrap the default remote LiteLLM run:

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

Starling 图从统一 `metrics.tsv` 读取。现有 parent-disjoint agent conditions、MiniMol train-all head、
Morgan KNN 和 MiniMol embedding cosine KNN 的旧 random/scaffold test 图属于上一版 strict-conflict lineage；
当前 70% record-majority 8:1:1 v4 已完成 GLM、GPT-OSS-20B、GPT-OSS-120B scaffold-valid blind matrix、
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
