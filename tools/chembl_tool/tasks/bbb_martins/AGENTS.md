# BBB_Martins task notes

本文件只记录 BBB_Martins 的 task-specific 语义：label、当前数据版本、BBB evidence tier、过滤规则和 reasoning 边界。通用 ChEMBL workflow、wrapper 结构、batch/resume、viewer、cost 和目录规范统一记录在仓库根 `AGENTS.md`。

## Task 定义

目标不是训练 BBB classifier，而是构建可审计的 BBB evidence library：给定 query molecule，先预取相似分子的 BBB / permeability / transporter evidence，再交给 reasoning LLM 判断 analog evidence 是否能 transfer 到 query molecule。

当前评估约定：

```text
Y=1 -> bbb_prediction=pass
Y=0 -> bbb_prediction=fail
final summary 必须在 pass/fail 中二选一；不再允许 uncertain prediction
```

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

## Task-specific 文件

```text
endpoint_groups.py
  BBB Tier.endpoint_group、evidence_direction、evidence_strength 规则。

rules.py
  BBB assay screening 关键词、negative keywords、weak terms、transporter target genes。

scoring.py
  BBB assay 保留/剔除和打分统一入口。screen_assays.py 和 rescore_outputs.py 都调用 scored_row()。

run_reasoning_pipeline.py
  BBB_Martins prompt、single/group/final schema、retrieval/prompt assembly 和 final-only rerun。
```

下面这些文件是 task-specific 配置 wrapper，公共实现见根 `AGENTS.md` 的 `tools/chembl_tool/common/task_workflows/` 说明：

```text
screen_assays.py
rescore_outputs.py
summarize_outputs.py
report.py
build_evidence_library.py
retrieve_neighbors.py
chembl_exact_context.py
run_reasoning_batch.py
```

不要在 wrapper 中新增业务规则；BBB assay 保留/剔除逻辑应只放在 `rules.py` 和 `scoring.py`，endpoint-group 语义应只放在 `endpoint_groups.py`。

## 当前数据和输出

当前推荐使用：

```text
outputs/chembl_tool/tasks/bbb_martins/assay_screening/v6/
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

当前 evidence library 和 neighbor index：

```text
outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_molecule_evidence.jsonl
outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_neighbor_index.pkl
outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_neighbor_index.meta.json
```

当前 test set：

```text
data/processed/BBB_Martins/B3DB_cleaned/test/test_efflux.jsonl

fields:
  drug: query SMILES
  Y: BBB label
```

reasoning 产物统一放在：

```text
outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs/
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/
```

## BBB reasoning pipeline

```text
1. 读取 test_efflux.jsonl 的 query molecule。
2. 调用 retrieve_neighbors.py 预取每个 Tier.endpoint_group 的 ChEMBL neighbor evidence。
3. 并发执行 single-molecule analysis；DeepSeek 只可调用 molecule_properties。
4. 并发执行 group-level analysis；DeepSeek 只可调用 mmp_structure_compare 和 properties_compare。
5. final summary 读取 single + all group outputs，不暴露任何 tool。
6. 保存 retrieval/single/group/final/trace/manifest。
```

final-only rerun：

```bash
python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline \
  --resume-final-from-run-dir outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs/<run_id>
```

## Exact ChEMBL context

`chembl_exact_context.py` 是可选 evidence-rich 增强。它会用 query full InChIKey 查 ChEMBL exact molecule，并在 retrieved neighbor 涉及的 assay 中查 query activity，生成：

```text
same_endpoint_activity:
  same assay_chembl_id + same normalized standard_type

same_assay_different_endpoint_activity:
  same assay_chembl_id + different standard_type
```

这会使用 query molecule 的已知 ChEMBL 实验记录，可能造成 prospective benchmark 的数据泄漏。因此默认关闭；只有显式传 `--enable-chembl-exact-context` 时才用于 retrospective / evidence-rich case study。默认批量评估不要开启。

默认 single-molecule prompt 不包含任何 ChEMBL 相关 payload 或 instruction。只有开启 exact context 且命中 query exact context 时，single-molecule payload 才包含 `exact_query_chembl_context`，并提示模型区分 direct same-molecule ChEMBL BBB evidence 和 physicochemical prior。ChEMBL neighbor evidence 仍只进入 group-level context。

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
  'tools/chembl_tool/utils/BBB_Martins/monitor_chembl_assay.sh chembl_assay outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw 5400 /tmp/chembl_assay_monitor.log'
```

参数：

```text
chembl_assay: 被监控的 tmux session
outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw: 输出目录
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
