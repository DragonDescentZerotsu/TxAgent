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

TRACE_RETENTION.md
  最终论文 trace 的唯一目录、保留单位、清理边界和 viewer 约束。
```

新增实验或得到新结果时：先在执行总计划更新对应 E 编号状态，再运行生成器更新机器可读统计和
`analysis/report.md`，然后根据生成产物把已测结果同步到 `RESULTS.md`，最后更新
`VISIBILITY_ANALYSIS.md` 等解释文档。当前生成器不会自动改写 `RESULTS.md`；不得只在聊天中保留结论。

## 代码与命令入口

```text
molecular_evidence_agent.py
  冻结实验矩阵与统一运行入口；负责 visibility mode、neighbor identity policy、结果 root 和跨条件复用参数。

summarize_results.py
  汇总 identity-blind、matched-prefetch 和 agentic operational 条件，生成 coverage、token、visibility audit、
  paired bootstrap、McNemar/Holm 和 `analysis/report.md`。

audit_prefetch_contract.py
  逐 condition、逐样本验证 matched-prefetch 是否完整 replay identity-blind 的 retrieval 和工具输出。

parent_disjoint_ablation.py
  审计 operational retrieval 的 same-parent overlap，生成选择性重跑/整条复用/family-branch 复用计划；
  加 `--materialize` 后只物化输入未变化的整条 reuse artifact 和 reuse plan。输入变化的样本仍由统一 matrix
  runner 以 `--neighbor-identity-policy parent_disjoint` 执行。

summarize_parent_disjoint_results.py
  在完整 test set 上配对比较 operational 与 parent-disjoint，并审计 identity policy、threshold 和 reuse provenance。

plot_retrieval_claims_overview.py
  从已生成的 analysis TSV 绘制 retrieval claims 总览 SVG；它不重新计算指标。
```

常用审计顺序：

```bash
python -m tools.chembl_tool.paper_experiments.summarize_results
python -m tools.chembl_tool.paper_experiments.audit_prefetch_contract
python -m tools.chembl_tool.paper_experiments.summarize_parent_disjoint_results
python -m tools.chembl_tool.paper_experiments.plot_retrieval_claims_overview
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
parent_disjoint_ablation: required for analog-retrieval claims
```

Operational 主结果允许 same RDKit molecular parent 的盐型、溶剂化物、重复组分和 formulation-linked record，
但必须显式标记其 entity relation，并与真正 structural analog 分开报告。用于证明“similar-molecule
retrieval”的 parent-disjoint 消融必须标准化 query/source parent、排除相同 parent、再用后续候选补足
top-k。现有 17 个 retrieval 条件的首轮消融已完成，但最终 Starling 数据补齐和 source freeze 后仍需按
同一 contract 重跑，因此现阶段不得改称最终主表。这里的 parent 是 RDKit FragmentParent 标准化结果，
不是 active moiety，也不推断 prodrug 或代谢物关系。

Parent-disjoint 产物统一放在：

```text
outputs/paper/molecular_evidence_agent/runs_deployment_visible_parent_disjoint/
outputs/paper/molecular_evidence_agent/analysis/parent_disjoint_ablation/
```

先运行 `parent_disjoint_ablation.py` 做两阶段审计：只有 operational top-k 中实际出现 same-parent 的
sample-condition 才重新执行 fingerprint retrieval 和 GLM。无 same-parent 的输入按 hash 直接复用；
mechanism condition 还应通过 `--group-analysis-source-batch` 复用 hash 未变化的 family branch。禁止通过
删除受影响样本、只评估子集或从 similarity threshold 以下补候选来节省计算。

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

每个任务必须先运行 `none` 条件。所有包含检索的条件都必须使用 `--single-analysis-source-batch`，以复用该条件中冻结的 single-molecule 分支。不得为每个消融条件独立重新生成先验。

不得将 API key 的值写入代码、保存到文件中的命令、manifest、报告或 trace。使用 `GLM_API_KEY` 或另一个明确指定的环境变量。

OpenAI SDK 的传输层 retry 上限明确固定为 2，与 baseline run 一致。结构化 JSON 验证在记录的调用路径中最多允许 4 次总尝试。最后两次尝试只重新序列化相同的 JSON evidence，以避开 provider 的确定性退化；不得删除证据或改变 inference setting。single、group 或 final 分支不完整的 run 会被 batch 完整性约束排除，并通过 batch `--skip-existing` 显式重跑。

Group prompt 还具有 provider 传输保护。清理后不超过 750 KB 的 payload 保持不变。超过该阈值时，每个超限 neighbor 最多保留 100 条按确定性等间距采样的 evidence row；prompt 必须记录原始行数、保留行数、原始字节数、阈值和采样方法。该机制在不引入数据源或任务专用规则的情况下避免 endpoint 层 HTTP 413 失败。

报告一个 run 之前：

1. 必须满足 `n_failed_runs == 0`；否则使用 `--skip-existing` 重跑失败样本。
2. 必须满足 `summarize_results.py` 输出的 `query_smiles_trace_leaks == 0`。
3. `identity_blind` 必须通过身份盲化 preflight，并使用脱敏后的 group 输出进行 final synthesis。`group_reasoning_outputs_raw.jsonl` 仅用于审计，绝不能作为 final 输入。
4. 主实验 `deployment_visible` 必须通过正向可见性 contract。补充的 `deployment_visible_prefetched` 必须逐 query replay 对应 `identity_blind` run 的冻结 `retrieval.json` 和 prefetched tool outputs，不能按当前 task config 重新检索或重新执行可能非确定的 MCS。当前 21 条件矩阵的 prefetch audit 必须覆盖 21/21 conditions 和 4,456/4,456 samples，且 missing、extra、mismatch 均为 0；交集匹配不能替代覆盖率检查。
5. 报告性能时必须同时报告 retrieval coverage；缺少 neighbor 是数据源/索引的真实属性，不得静默删除相应样本。
6. 使用共享汇总程序生成制度内和跨制度的配对测试集比较、bootstrap 区间和精确 McNemar 检验。
7. 标量 KNN 结果必须与 LLM agent 条件分开报告；它是数值型 direct-F 对照，且不属于任何 LLM 可见性制度。

生成的证据和运行输出位于 `outputs/paper/molecular_evidence_agent/`：`runs_deployment_visible/` 是 agentic
主实验候选，`runs/` 是 identity-blind 补充控制，`runs_deployment_visible_prefetched/` 是 matched-prefetch
补充控制。
这些是可复现产物，不是源代码。

Paper runner 必须保存每个样本自己的 `trace_messages.jsonl`，并传 `--no-combine-traces`，避免再生成
condition-level 的重复大文件。最终 trace viewer 只服务四个 paper run root：`runs/`、
`runs_deployment_visible_prefetched/`、`runs_deployment_visible/` 和
`runs_deployment_visible_parent_disjoint/`，通过 `predictions.jsonl` 定位 per-run trace；不得重新加入旧
task、Tier、expert policy 或 task-specific prediction 字段的硬编码适配。Parent-disjoint 仍必须通过
`analysis/parent_disjoint_ablation/` 的汇总产物审计；viewer 还必须从 manifest 和 `reuse.json` 明确显示
`parent_disjoint` policy，以及 sample 是因 retrieval 变化而重跑，还是因 LLM-visible input hash 未变化而复用。
