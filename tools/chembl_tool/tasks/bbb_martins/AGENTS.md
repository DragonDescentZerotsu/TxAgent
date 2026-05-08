# BBB_Martins ChEMBL 工具说明

本目录实现 BBB_Martins 任务的 ChEMBL evidence 筛选、molecule-level evidence library、neighbor retrieval 和 reasoning pipeline。

目标不是训练模型，而是构建一个可审计的 BBB evidence library，并在 query-time 先预取相似分子 evidence，再交给 reasoning LLM 判断 analog evidence 是否能 transfer 到 query molecule。

当前边界：

```text
ChEMBL neighbor retrieval 不是 DeepSeek 可调用 tool。
ChEMBL neighbor retrieval 也不是当前 FastAPI service tool。
它是 run_reasoning_pipeline.py 内部的 evidence prefetch / context assembly 步骤。

DeepSeek group-level analysis 可调用的工具只有：
  mmp_structure_compare
  properties_compare

DeepSeek single-molecule analysis 可调用的工具只有：
  molecule_properties
```

---

## 目录结构

```text
tools/chembl_tool/tasks/bbb_martins/
  __init__.py
  AGENTS.md
  endpoint_groups.py
  build_evidence_library.py
  retrieve_neighbors.py
  run_reasoning_pipeline.py
  run_reasoning_batch.py
  rules.py
  scoring.py
  screen_assays.py
  rescore_outputs.py
  summarize_outputs.py
  report.py
```

依赖的通用工具在：

```text
tools/chembl_tool/common/
  assay_loader.py
  export.py
  schema.py
  sqlite.py
  text.py
```

通用层负责 SQLite 连接、ChEMBL 表读取、assay/activity/target 聚合、文本标准化和 CSV/JSONL 导出。任务层只放 BBB_Martins 专属规则和 CLI。

---

## 核心脚本

### `screen_assays.py`

全量筛选入口。读取 ChEMBL SQLite，按统一 BBB scoring 逻辑筛出候选 assay，并生成候选表、报告和可选 activity evidence。

推荐命令：

```bash
python -m tools.chembl_tool.tasks.bbb_martins.screen_assays \
  --chembl-sqlite tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db \
  --out-dir outputs/chembl_bbb \
  --min-score 40 \
  --progress-every 10000 \
  --export-activities
```

主要输出：

```text
outputs/chembl_bbb/bbb_assay_candidates.csv
outputs/chembl_bbb/bbb_assay_candidates.jsonl
outputs/chembl_bbb/bbb_assay_report.md
outputs/chembl_bbb/bbb_activity_evidence.csv
```

说明：

```text
bbb_assay_candidates.*: assay-level 候选结果
bbb_assay_report.md: Top examples 和 Tier 统计
bbb_activity_evidence.csv: 候选 assay 下的 molecule-level activity evidence
```

`--export-activities` 会额外导出每个候选 assay 对应的分子实验记录，包括 molecule ChEMBL ID、canonical SMILES、standard_type、standard_value、units 等。这个文件是后续相似分子 evidence retrieval 的基础。

常用参数：

```text
--chembl-sqlite      ChEMBL SQLite 数据库路径
--out-dir            输出目录
--min-score          保留阈值，默认 40
--progress-every     每扫描 N 个 assay 输出一次进度；设为 0 可关闭
--export-activities  导出候选 assay 对应的 activity evidence
--limit              只扫描前 N 个 assay，用于 smoke test
```

### `rescore_outputs.py`

旧结果重打分工具。它不包含独立业务规则，只读取已有 `bbb_assay_candidates.csv`，调用统一的 `scoring.scored_row()` 重新打分并过滤。

用途：

```text
1. scoring/rules 更新后，不想重新扫描 189 万 assay 时，快速重算旧候选。
2. 尝试不同 --min-score。
3. 同步过滤已有 bbb_activity_evidence.csv。
```

推荐命令：

```bash
python -m tools.chembl_tool.tasks.bbb_martins.rescore_outputs \
  --in-dir outputs/chembl_bbb \
  --out-dir outputs/chembl_bbb_cleaned_v6 \
  --min-score 40 \
  --filter-activities
```

注意：

```text
rescore_outputs.py 只能处理已有候选。
如果新规则变宽，可能发现原候选表之外的新 assay，这种情况必须重新跑 screen_assays.py。
如果新规则变严、重排 Tier 或调整分数，rescore_outputs.py 通常足够。
```

### `summarize_outputs.py`

健康检查报告生成工具。读取候选 CSV，统计 Tier 分布、matched keywords/endpoints、negative flags，并列出每个 Tier 的 top examples。

命令：

```bash
python -m tools.chembl_tool.tasks.bbb_martins.summarize_outputs \
  --out-dir outputs/chembl_bbb_cleaned_v6 \
  --report-path outputs/chembl_bbb_cleaned_v6/bbb_health_check.md
```

输出：

```text
bbb_health_check.md
```

### `endpoint_groups.py`

定义第二阶段的 `Tier.endpoint_group` 规则，并为 BBB evidence library 内部审计生成
`evidence_direction`、`evidence_strength`、`endpoint_group_reason` 等派生字段。

这些派生字段可以保留在 library/debug 文件中，但不要发送给 reasoning LLM。

### `build_evidence_library.py`

把 `bbb_assay_candidates.csv` 和 `bbb_activity_evidence.csv` 构建成 molecule-level evidence
library 和 query-time neighbor index。当前会使用预计算 ChEMBL fingerprints：

```text
tools/chembl_tool/chembl_data/chembl_36_fps/chembl_36.fps.gz
```

当前主要输出：

```text
outputs/bbb_martins/evidence_library/bbb_molecule_evidence.jsonl
outputs/bbb_martins/evidence_library/bbb_neighbor_index.pkl
outputs/bbb_martins/evidence_library/bbb_neighbor_index.meta.json
```

### `retrieve_neighbors.py`

给定 query SMILES，按 `Tier.endpoint_group` 从 BBB evidence molecule subset 中检索 top-k
non-identical neighbors。当前默认 `top_k_per_group=3`、`min_similarity=0.3`，过滤 very distant analog；
保留的低相似度 analog 会带上 `similarity_bucket`，交给 LLM 判断 transferability。

这是 pipeline 内部 evidence retrieval / context assembly，不是 DeepSeek tool。

### `chembl_exact_context.py`

可选 evidence-rich 增强。它会用 query full InChIKey 查 ChEMBL exact molecule，并在 retrieved
neighbor 涉及的 assay 中查 query activity，生成：

```text
same_endpoint_activity:
  same assay_chembl_id + same normalized standard_type

same_assay_different_endpoint_activity:
  same assay_chembl_id + different standard_type
```

这会使用 query molecule 的已知 ChEMBL 实验记录，可能造成 prospective benchmark 的数据泄漏。
因此默认关闭；只有显式传 `--enable-chembl-exact-context` 时才用于 retrospective / evidence-rich
case study。默认批量评估不要开启。

### `run_reasoning_pipeline.py`

BBB_Martins 端到端 reasoning 入口。当前流程：

```text
1. 读取 test_efflux.jsonl 的 query molecule。
2. 调用 retrieve_neighbors.py 预取每个 group 的 ChEMBL neighbor evidence。
3. 并发执行 single-molecule analysis；DeepSeek 只可调用 molecule_properties。
4. 并发执行 group-level analysis；DeepSeek 只可调用 mmp_structure_compare 和 properties_compare。
5. final summary 读取 single + all group outputs，不暴露任何 tool。
6. 保存 retrieval/single/group/final/trace/manifest。
```

支持只重跑 final summary：

```bash
python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline \
  --resume-final-from-run-dir outputs/bbb_martins/reasoning_runs/<run_id>
```

### `run_reasoning_batch.py`

批量 reasoning 入口。它按 query index 调用 `run_reasoning_pipeline.py`，负责 molecule 级并行、
日志、trace 合并和评估报告。

当前 label 约定：

```text
Y=1 -> bbb_prediction=pass
Y=0 -> bbb_prediction=fail
final summary 必须在 pass/fail 中二选一；不再允许 uncertain prediction
```

推荐命令：

```bash
python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_batch \
  --input-jsonl data/processed/BBB_Martins/B3DB_cleaned/test/test_efflux.jsonl \
  --parallelism 1 \
  --group-workers 4 \
  --batch-id <batch_id>
```

批量输出：

```text
outputs/bbb_martins/reasoning_batches/<batch_id>/predictions.jsonl
outputs/bbb_martins/reasoning_batches/<batch_id>/metrics.json
outputs/bbb_martins/reasoning_batches/<batch_id>/report.md
outputs/bbb_martins/reasoning_batches/<batch_id>/trace_messages.jsonl
outputs/bbb_martins/reasoning_runs/<batch_id>_combined/trace_messages.jsonl
```

`<batch_id>_combined` 会被当前 trace viewer 自动扫到；打开后可通过 `Molecule package`
下拉框切换不同分子的 trace。

### `report.py`

供 `screen_assays.py` 和 `rescore_outputs.py` 调用的 Markdown 报告生成模块。通常不直接运行。

### `rules.py`

定义 BBB_Martins 专属规则：

```text
Tier 1: direct BBB / brain exposure
Tier 2: passive permeability / barrier model
Tier 3: efflux transporter
Tier 4: influx transporter
negative keywords
weak terms
transporter target genes
```

不要在脚本各处散落硬编码规则；新增或调整关键词优先改这里和 `scoring.py`。

### `scoring.py`

统一业务逻辑入口。全量筛选和旧结果重打分都调用：

```python
scored_row(row, min_score=40)
```

当前所有 BBB assay 保留/剔除逻辑都应在 `rules.py` 和 `scoring.py` 中维护。不要在 `rescore_outputs.py` 里新增另一套 postprocess 规则。

---

## 当前最终结果目录

当前推荐使用：

```text
outputs/chembl_bbb_cleaned_v6/
```

其中：

```text
bbb_assay_candidates.csv
bbb_assay_candidates.jsonl
bbb_assay_report.md
bbb_health_check.md
bbb_activity_evidence.csv
```

当前 v6 统计：

```text
candidate assays: 20,369
activity evidence rows: 113,929
Tier 1: 4,306
Tier 2: 4,609
Tier 3: 10,752
Tier 4: 702
```

旧的 `outputs/chembl_bbb_cleaned*` 中间版本只用于调试对比。确认 v6 后，可以清理中间版本。

---

## Evidence 类型解释

### Tier 1

直接 BBB / brain exposure 证据，例如：

```text
brain/plasma ratio
brain to plasma ratio
logBB
Kp,uu,brain
brain concentration
brain level
brain perfusion
CSF/plasma
```

这是最接近“分子是否进入脑内”的证据。

### Tier 2

被动通透或体外屏障模型证据，例如：

```text
PAMPA-BBB
Caco-2 Papp
MDCK Papp
brain endothelial cell model
hCMEC/D3
bEnd.3
BMEC / BBMEC / RBEC
```

这类是 proxy evidence，不等价于体内脑暴露。

### Tier 3

外排转运体证据，例如：

```text
ABCB1 / P-gp / MDR1
ABCG2 / BCRP
ABCC / MRP
efflux ratio
bidirectional Papp
rhodamine 123 efflux
calcein AM
Hoechst 33342 accumulation
```

P-gp/BCRP substrate、efflux ratio、bidirectional Papp 比单纯 inhibitor IC50 更接近 BBB reasoning。

### Tier 4

摄取转运体证据，例如：

```text
LAT1 / SLC7A5
GLUT1 / SLC2A1
OATP1A2 / SLCO1A2
OAT3 / SLC22A8
MCT1 / SLC16A1
TFRC
```

Tier 4 必须是 functional uptake/transport/substrate 相关 assay。单纯 gene expression、Western blot、phosphorylation、binding affinity 等不保留。

---

## 重要过滤规则

### `brain plasma membrane` 不是 brain/plasma

`brain/plasma` 表示脑组织暴露量与血浆暴露量的比值，是 BBB evidence。

但：

```text
brain plasma membrane
brain plasma membranes
```

表示脑组织来源的细胞膜/膜制备物，经常出现在 receptor binding assay 中，不是 brain/plasma ratio。当前逻辑会过滤这类误命中。

### 不保留非功能性 influx assay

例如：

```text
GLUT1 expression
SLC2A1 RNA stability
Western blot
phosphorylation
Kinobead pull down
binding affinity
```

如果没有 uptake/transport/substrate 等功能性读数，不作为 BBB influx evidence。

### 非 transporter target 的 resistant-cell-line phenotype 噪声

例如 target 是 MAP3K5、proteasome 等，但 description 里出现：

```text
ABCB1-substrate-selected resistant cell line
ABCG2-substrate-selected resistant cell line
overexpressing ABCB1
overexpressing ABCG2
```

这类通常是 phenotypic/cytotoxicity setting，不作为高置信 BBB transporter assay。当前逻辑会过滤明显噪声。

但会保留功能性 transporter readout，例如：

```text
P-gp-mediated rhodamine 123 efflux
calcein AM assay
Hoechst 33342 accumulation
mitoxantrone accumulation
doxorubicin accumulation
transepithelial transport
```

---

## canonical SMILES 为空的情况

`bbb_activity_evidence.csv` 中少量行的 `canonical_smiles` 为空，这是 ChEMBL 原始结构数据缺失，不是 join 错误。

在 v6 中：

```text
activity evidence rows: 113,929
missing canonical_smiles: 279
affected molecule_chembl_id: 147
```

常见原因：

```text
structure_type = NONE
structure_type = SEQ
金属/无机物
放射性 technetium complex
蛋白/肽/序列类 molecule
ChEMBL 没有标准结构
```

后续构建相似性检索索引时，应过滤 `canonical_smiles` 为空的 molecule。原始 evidence 行可以保留，用于审计和报告。

---

## 长任务监控

一次性监控脚本：

```text
tools/chembl_tool/utils/BBB_Martins/monitor_chembl_assay.sh
```

用途：每隔固定时间检查一个 tmux session 是否完成，完成后自动生成健康检查报告。

示例：

```bash
tmux new-session -d -s chembl_assay_monitor \
  'tools/chembl_tool/utils/BBB_Martins/monitor_chembl_assay.sh chembl_assay outputs/chembl_bbb 5400 /tmp/chembl_assay_monitor.log'
```

参数：

```text
chembl_assay: 被监控的 tmux session
outputs/chembl_bbb: 输出目录
5400: 检查间隔秒数，即 90 分钟
/tmp/chembl_assay_monitor.log: monitor 日志
```

这个脚本不是核心业务逻辑，可留作长任务运维工具。

---

## 测试

运行：

```bash
pytest -q tests/chembl_tool
```

当前覆盖：

```text
文本标准化
P-gp / Caco-2 / blood-brain barrier / Kp,uu brain 匹配
generic permeability / generic uptake 过滤
ABCB1 target + substrate 保留
brain plasma membrane 误命中过滤
Tier 4 非功能性 GLUT1/SLC2A1 assay 过滤
非 transporter target resistant-cell-line 噪声过滤
functional rhodamine efflux 保留
```

---

## 当前 retrieval / reasoning 状态

第二阶段已经基于 `bbb_activity_evidence.csv` 构建 molecule-level evidence library：

```text
1. 过滤 canonical_smiles 为空的 molecule
2. 标准化 molecule identity，排除 query exact same molecule，包括 full InChIKey、InChIKey connectivity layer 和 canonical SMILES 相同的记录
3. 用 Morgan fingerprint Tanimoto 检索相似 analog
4. 按 Tier、endpoint type、similarity bucket 聚合 evidence
5. 输出 analog evidence summary，而不是直接硬判定 BBB pass/fail
```

当前实现入口：

```text
build_evidence_library.py
retrieve_neighbors.py
run_reasoning_pipeline.py
```

ChEMBL neighbor retrieval 当前作为 pipeline 内部 evidence prefetch，不作为 LLM tool。
后续新增其他 ChEMBL task 时，继续复用 `tools/chembl_tool/common/` 的通用工具。
