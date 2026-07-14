# 分子证据 Agent 论文实验

本目录冻结面向论文的实验矩阵，但不把任务逻辑写进通用 runner。

## 文档入口

- [ICLR 2027 执行计划](ICLR_2027_EXECUTION_PLAN.md)：贡献、实验、资源、缺口、时间表和投稿 gate。
- [当前冻结实验协议](EXPERIMENT_PLAN.md)：第一轮矩阵、参数、指标和统计方法。
- [Pipeline](PIPELINE.md)：数据流、retrieval views、visibility 和 mechanism families。
- [Visibility 分析](VISIBILITY_ANALYSIS.md)：structure visible 后性能上升或下降的集中解释。
- [当前结果](RESULTS.md)：已完成 full run 的实测结果。
- [Trace 保留策略](TRACE_RETENTION.md)：最终 trace 的唯一目录、清理边界和一致性约束。

## 共享约定

所有 LLM 实验共享以下条件：

- 通过 Penn LiteLLM OpenAI-compatible endpoint 使用 GLM-5.2
- temperature 0，最大输出 20,480 tokens
- operational retrieval 排除完全相同的 source record，但允许并标记 same-parent formulation record
- parent-disjoint 消融排除相同 RDKit molecular parent，并只用原 similarity threshold 以上的后续
  structural analog 回填 top-k；这里的 parent 不是药理学 active moiety
- 每个 task/query 冻结一个 single-molecule analysis，并在各检索条件间复用
- 每个 evidence group 最多检索 3 个 Morgan similarity 不低于 0.30 的 neighbor
- 共享 single、group、final、JSON validation、retry、trace 和 batch evaluation workflow

论文主结果表使用 deployment-visible agentic tool-use 制度：

- `deployment_visible`：结构与允许身份信息可见；LLM 自主选择工具；用于主表的 none/direct/flat/mechanism 和 source comparison。
- `identity_blind`：LLM 不看 query/neighbor 的结构和身份；harness 预取结构工具结果；只作为补充诊断。
- `deployment_visible_prefetched`：结构与允许身份信息可见；逐 query replay `identity_blind` 的冻结 retrieval 和工具结果；只作为 matched control。

实验模式定义在 `common/experiment_retrieval.py`：

- `none`：只有 query properties，不提供检索证据
- `direct`：只使用任务声明的直接结果证据
- `full_flat`：把所有任务机制证据合并到一个 group
- `full_mechanism`：使用完全相同的证据集合，但按 mechanism family 分组

每个任务只在 `experiment_config.py` 中声明 source endpoint group 如何映射到 direct 和 mechanism family。

## 冻结矩阵

运行矩阵前先构建冻结的 Starling index：

```bash
python -m tools.chembl_tool.tasks.bbb_martins.build_starling_evidence_library \
  --mode all \
  --out-root outputs/paper/molecular_evidence_agent/evidence/bbb_starling \
  --workers 128

python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_factor_evidence_library \
  --scope full \
  --evidence-content full \
  --out-dir outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_full \
  --workers 128

python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_factor_evidence_library \
  --scope direct \
  --evidence-content numeric_only \
  --out-dir outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_direct_numeric \
  --workers 128
```

```bash
python -m tools.chembl_tool.paper_experiments.molecular_evidence_agent --list
```

运行主实验时必须显式选择 deployment-visible agentic 制度：

```bash
python -m tools.chembl_tool.paper_experiments.molecular_evidence_agent \
  --visibility-mode deployment_visible \
  --experiments bioavailability_ma__none bioavailability_ma__starling_full_mechanism
```

运行补充的 identity-blind 制度时省略 `--visibility-mode`。运行 matched-prefetch 补充控制：

```bash
python -m tools.chembl_tool.paper_experiments.molecular_evidence_agent \
  --visibility-mode deployment_visible_prefetched \
  --experiments bioavailability_ma__none bioavailability_ma__starling_full_mechanism
```

该命令要求对应 identity-blind condition 已完整存在。batch 会自动传入
`--retrieval-replay-source-batch` 和 `--prefetched-tool-replay-source-batch`；不要用当前 index 重新构建
matched-visible retrieval，否则 task mapping 漂移或 MCS 非确定性会破坏严格配对。

API key 默认从 `GLM_API_KEY` 读取，其值不得写入命令记录、manifest 或 trace。

Parent-disjoint 已由公共 identity normalizer、retrieval policy、top-k backfill 和 manifest provenance
实现。先审计并生成选择性重跑计划；确认后用 `--materialize` 物化输入未变化的整条 reuse artifact 和
reuse plan：

```bash
python -m tools.chembl_tool.paper_experiments.parent_disjoint_ablation
python -m tools.chembl_tool.paper_experiments.parent_disjoint_ablation --materialize
```

然后按 plan 只选择 retrieval conditions 运行统一 matrix 入口；`none` 不写入 parent-disjoint root：

```bash
python -m tools.chembl_tool.paper_experiments.molecular_evidence_agent \
  --visibility-mode deployment_visible \
  --neighbor-identity-policy parent_disjoint \
  --experiments <retrieval_condition_names_from_plan>
```

该流程只重跑 operational top-k 中实际出现 same-parent 且 LLM-visible input hash 变化的 sample-condition；
未变化的整条 run 已复用，`full_mechanism` 还可复用未变化的独立 family branch。不得降低
`min_similarity`、删除受影响样本或只在 overlap 子集上计算指标。

## 汇总分析

```bash
python -m tools.chembl_tool.paper_experiments.summarize_results

python -m tools.chembl_tool.paper_experiments.audit_prefetch_contract

python -m tools.chembl_tool.paper_experiments.summarize_parent_disjoint_results

python -m tools.chembl_tool.paper_experiments.plot_retrieval_claims_overview
```

`audit_prefetch_contract` 同时输出逐样本 `prefetch_contract_audit.tsv` 和逐 condition
`prefetch_contract_conditions.tsv`。当前 21 条件矩阵只有在 `n_conditions_complete=21`、
`n_expected=n_audited=4456`、missing/extra/mismatch 均为 0 时才通过。审计同时要求 raw retrieval
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
