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
- [Trace 保留策略](TRACE_RETENTION.md)：最终 trace 的唯一目录、清理边界和一致性约束。

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
  `--parallelism 128 --group-workers 1`，禁止多个
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

tools/chembl_tool/paper_experiments/build_starling_benchmark_indices.py
  读取 heldout union，从 inference evidence 删除全部 valid/test parents，构建 split-specific index。

tools/chembl_tool/paper_experiments/starling_benchmark_matrix.py
  当前正式 valid/test runner；默认 identity_blind + parent_disjoint、128×1，valid 与 test root 隔离。
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
  --parallelism 128 \
  --group-workers 1
```

Operational 和 deployment-visible 以后只能显式 opt in，并写入独立 historical/ablation root：

```bash
python -m tools.chembl_tool.paper_experiments.molecular_evidence_agent \
  --visibility-mode deployment_visible \
  --neighbor-identity-policy operational \
  --experiments bioavailability_ma__none bioavailability_ma__starling_full_mechanism
```

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

当前 paper/Starling runner 默认使用本机 tunnel `http://127.0.0.1:50000/v1`、
`nvidia/GLM-5.2-NVFP4` 和历史一致的 `--disable-thinking --reasoning-effort ""`。空值使 client 不发送
`reasoning_effort` 参数，但 GLM 返回的 reasoning 仍会保存在 trace。运行前用 `ssh -fNT parcc-glm` 建立 tunnel；
loopback vLLM 无鉴权时，runner 会在进程内为 `GLM_LOCAL_API_KEY` 注入非敏感占位值。
旧 LiteLLM 只作为显式 fallback，使用 `--api-key-env GLM_API_KEY --base-url
https://litellm.parcc.upenn.edu/v1 --model zai-org/GLM-5.2-FP8 --reasoning-effort ""`。
不得将任何真实 API key 的值写入命令记录、manifest 或 trace。

V4 parent-disjoint 使用公共 identity normalizer、retrieval policy、top-k backfill 和 manifest provenance，
但直接 fresh-run，不读取 operational retrieval，也不生成 reuse plan。汇总必须从最终 `retrieval.json`
逐条验证 parent conflict=0、held-out overlap=0、threshold violation=0，并验证 identity-blind leak=0。
不得降低 `min_similarity`、删除受影响样本或只在 supported/overlap 子集上计算主指标。

## 汇总分析

```bash
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
python -m tools.chembl_tool.paper_experiments.summarize_starling_benchmark
python -m tools.chembl_tool.paper_experiments.plot_starling_benchmark_overview \
  --png-output outputs/paper/starling_benchmark_results/figures/starling_benchmark_overview_highres.png
```

Starling 图从统一 `metrics.tsv` 读取。现有 parent-disjoint agent conditions、MiniMol train-all head、
Morgan KNN 和 MiniMol embedding cosine KNN 都属于上一版 strict-conflict lineage；当前 70% record-majority
8:1:1 v4 dataset 尚未重跑这些模型结果。完整口径、历史结果和 baseline 入口见
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
