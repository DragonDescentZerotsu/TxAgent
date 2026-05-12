# TxAgent 当前系统：常驻分子工具服务与证据检索推理

## 当前目标

本项目要构建一个可复用的分子证据检索与 reasoning 系统。BBB_Martins 是第一个概念验证任务：给定一个 query molecule，先通过常驻 FastAPI 工具服务计算分子属性、结构差异和属性差异，再从 ChEMBL BBB evidence library 中检索相似分子的实验读数，最后把工具输出和 assay evidence 交给 reasoning LLM，综合判断该分子是否可能通过 BBB。

当前 BBB 数据基础：

```text
assay candidates:
  outputs/chembl_tool/tasks/bbb_martins/assay_screening/v6/bbb_assay_candidates.csv

activity evidence:
  outputs/chembl_tool/tasks/bbb_martins/assay_screening/v6/bbb_activity_evidence.csv

ChEMBL fingerprints:
  tools/chembl_tool/chembl_data/chembl_36_fps/chembl_36.fps.gz

test molecules:
  data/processed/BBB_Martins/B3DB_cleaned/test/test_efflux.jsonl
```

`test_efflux.jsonl` 当前字段：

```text
drug: query SMILES
Y: BBB label
```

## 设计原则

1. 不把 BBB 逻辑写成一次性脚本。BBB 是第一个 task，但工具服务、检索协议、LLM 输入输出格式应能支持后续更多任务。
2. ChEMBL neighbor retrieval 是 pipeline 的 evidence prefetch / context assembly 步骤，不是当前暴露给 LLM 的 function tool，也不是当前 FastAPI service tool。后续 pKa、logD、solubility、toxicity、target affinity、PK property 等模型才按通用 tool contract 接入。
3. 长初始化模型要常驻。慢启动模型和大索引应在服务启动时加载，通过 FastAPI endpoint 调用，避免每个 query 反复初始化。
4. evidence retrieval 只提供证据，不直接替代 reasoning。retrieval payload 必须保留 assay 描述、activity 数值、endpoint 语义、similarity 和不确定性。
5. retrieval 单元优先是 molecule-level evidence，不是 assay-level evidence。assay 信息要保留，但 query-time ranking 应先找相似 molecule，再展开其 assay/activity evidence。
6. LLM reasoning 分为并发证据分支和 final 汇总：single-molecule 分支判断理化性质先验，
   group-level 分支判断每个 Tier.endpoint_group 的 analog transferability，final-level 汇总所有证据。

## 当前常驻工具服务

当前已经实现的通用工具服务入口：

```text
tools/service/app.py
  FastAPI app。注册工具并提供 /health、/tools、/tools/{tool_name}/invoke、/tools/invoke、/tools/{tool_name}。

tools/service/config.py
  服务配置。MolGpKa 相关开关在这里读取；mmpdb 使用当前 Python 环境中已安装的 mmpdblib，不需要源码路径环境变量。

tools/service/registry.py
  ToolRegistry。负责初始化工具、复用共享实例、统一 invoke。

tools/service/schemas.py
  ToolRequest / ToolResponse / ToolError 等统一 schema。

tools/service/errors.py
  统一工具异常。

tools/service/tools/base.py
  BaseTool 抽象。

tools/service/tools/rdkit_properties.py
  molecule_properties v1。计算 RDKit descriptors、MolGpKa pKa/logD、AccFG 顶层 functional groups。

tools/service/tools/properties_compare.py
  properties_compare v1。比较两个分子的 molecule_properties 输出，functional groups 除外。

tools/service/tools/mmp_structure_compare.py
  mmp_structure_compare v1。只比较结构：Morgan Tanimoto、similarity bucket、mmpdb matched-pair transformation、MCS。
```

当前相关测试入口：

```text
tests/service/test_registry.py
tests/service/test_rdkit_properties.py
tests/service/test_properties_compare.py
tests/service/test_mmp_structure_compare.py
```

服务启动命令：

```bash
uvicorn tools.service.app:app --host 127.0.0.1 --port 8765
```

当前服务层暂时只冻结三个通用工具：

```text
molecule_properties
properties_compare
mmp_structure_compare
```

之前规划里的 `rdkit_properties`、`ml_pka` 不再作为独立 service tool 暴露；它们已经合并进 `molecule_properties`。`chembl_neighbors` 当前也不是常驻 service tool，也不是 DeepSeek 可调用 tool。BBB neighbor retrieval 仍在 `tools/chembl_tool/tasks/bbb_martins/retrieve_neighbors.py` 中，由 `run_reasoning_pipeline.py` 在调用 LLM 前预取并注入为 group evidence context；后续需要时再包装成 service tool 或 task endpoint。

## LLM 可见输出约定

工具响应会保留结构化 JSON，供 workflow、debug、缓存和测试使用；最终展示给 LLM 的内容只使用：

```text
ToolResponse.output.text
```

不要把 `raw_features`、`comparisons`、`mcs`、`transformation`、`metadata` 等结构化字段整包塞进 LLM prompt。需要在 prompt 中区分工具时，可以加简短标题，例如：

```text
[molecule_properties]
<output.text>

[properties_compare]
<output.text>

[mmp_structure_compare]
<output.text>
```

所有工具面向 LLM 的文本中，数字最多保留两位小数。None / 缺失 / 不适用值应以自然语言说明，例如 `not applicable`，避免把 Python/JSON 内部表示直接暴露给 LLM。

BBB evidence library 内部可以保留 `evidence_direction`、`evidence_strength`、`endpoint_group_reason`、`assay_reason` 等规则派生字段，方便 debug 和审计；但这些字段不要发送给 reasoning LLM。LLM payload 中的 activity evidence row 应只包含原始 ChEMBL assay/activity 字段和必要 metadata，例如 assay id、tier、description、target、standard_type/value/units、activity_comment、confidence_score、relationship_type。分组层可以保留 `tier` / `endpoint_group`，因为并行分析本身按这个分组运行。

## BBB 代码入口

```text
tools/chembl_tool/tasks/bbb_martins/endpoint_groups.py
  BBB endpoint_group、evidence_direction、evidence_strength 的规则。

tools/chembl_tool/tasks/bbb_martins/build_evidence_library.py
  从 assay candidates 和 activity evidence 构建 molecule-level evidence library。

tools/chembl_tool/tasks/bbb_martins/retrieve_neighbors.py
  按 Tier.endpoint_group 检索 BBB evidence neighbors。

tools/chembl_tool/tasks/bbb_martins/run_reasoning_pipeline.py
  BBB_Martins reasoning pipeline 入口。负责 neighbor retrieval、single-molecule analysis、
  group-level 并发 reasoning、final summary、trace 保存，以及 final-only rerun。

tools/chembl_tool/tasks/bbb_martins/run_reasoning_batch.py
  BBB_Martins 批量 reasoning 入口。按 query_index 调用单分子 pipeline，支持 molecule 级并行、
  可选 trace 保存/合并、prediction report、accuracy 和 macro-F1 评估。

tools/trace_viewer/viewer.html
  本地 trace 可视化页面。支持选择 run、选择 molecule trace package、查看单个分子的
  single/group/final messages、reasoning、tool calls 和 parsed JSON response。

tools/trace_viewer/start_viewer.sh
  启动通用 trace viewer 的静态 HTTP server。查看 standalone 单分子 run 时指向
  reasoning/single_runs；查看 batch run 时指向 reasoning/batches。

tools/chembl_tool/tasks/bbb_martins/
  其他 BBB evidence 清洗、打分、报告和输出汇总脚本。
```

## BBB evidence 分组标准

BBB 第二阶段不按每个 assay 单独检索。应按：

```text
Tier -> endpoint_group
```

组合生成 retrieval groups。

### Tier 1: direct BBB / brain exposure

建议 endpoint groups：

```text
direct_brain_plasma
  bpr
  brain/plasma
  b/p
  bbr
  ratio
  ratio auc

direct_unbound_brain
  k(p,uu,brain)
  k(p,uu,csf)
  kp
  fu

direct_logbb_or_brain_level
  logbb
  log bb
  brain level
  brain concentration
  brain penetration index
  bpi

direct_brain_uptake_or_perfusion
  drug uptake
  drug uptake(free)
  uptake
  brain uptake
```

### Tier 2: passive permeability / barrier model

建议 endpoint groups：

```text
passive_papp
  papp
  logpapp
  logp app
  papp e-6

passive_caco2
  caco-2 papp
  caco-2 permeability
  pcaco2

passive_generic_permeability
  permeability
  permeability coefficient
  peff
  logpeff
  log pe
  pc
  pm
  pbbb

passive_transport_or_recovery
  drug transport
  drug recovery
```

### Tier 3: efflux transporter

Tier 3 必须拆分强弱证据，不能把 transporter inhibition 直接解释成 efflux substrate。

建议 endpoint groups：

```text
efflux_functional_ratio_or_bidirectional
  efflux ratio
  ratio
  ratio_papp
  papp a to b (mean)
  papp b to a (mean)
  papp

efflux_transport_or_accumulation
  drug transport
  drug uptake
  activity
  flu intensity
  rfu
  fluorescence

efflux_atpase_or_probe
  ratio_atpase activity
  atpase
  relative jc-1 accumulation

efflux_inhibition_or_binding
  inhibition
  ic50
  ki
  ec50
  kd
  km
  kon
  k_off
  ratio ic50
  ratio ec50
  fc
```

### Tier 4: influx transporter

Tier 4 也必须区分 functional uptake 和普通 binding/inhibition。

建议 endpoint groups：

```text
influx_functional_uptake_or_transport
  drug uptake
  uptake
  drug transport
  transport

influx_kinetic_or_substrate
  km
  vmax
  jmax
  kin

influx_inhibition_or_binding
  inhibition
  ic50
  ki
  kd
  kon
  k_off
  ec50
  ratio ic50
```

### Unknown / weak context

下面 endpoint 只能作为 context-dependent evidence，不能单独强解释：

```text
activity
ratio
inhibition
survival
cc50
gi50
ec90
mic
flu intensity
rfu
fluorescence
```

如果这些 endpoint 出现在明确 assay context 中，可以被 endpoint group 规则提升；否则应标记为：

```text
endpoint_group: context_dependent
evidence_strength: weak
```

## BBB evidence library

应从当前 assay candidates 和 activity evidence 构建 molecule-level library。每一条 evidence 至少包含：

```text
molecule_chembl_id
canonical_smiles
standard_inchi_key
assay_chembl_id
assay_id
assay_tier
endpoint_group
endpoint_group_reason
assay_description
target_chembl_id
target_pref_name
target_genes
standard_type
standard_relation
standard_value
standard_units
pchembl_value
data_validity_comment
activity_comment
evidence_direction
evidence_strength
evidence_reason
```

其中：

```text
evidence_direction:
  supports_bbb_crossing
  argues_against_bbb_crossing
  efflux_risk
  influx_support
  permeability_support
  context_dependent
  unknown_direction

evidence_strength:
  strong
  moderate
  weak
  context_dependent
```

`endpoint_group` 由 `assay_tier + standard_type + assay_description + target_genes` 共同决定。不能只看 `standard_type`，因为 `ratio`、`activity`、`inhibition` 等 endpoint 在不同 assay context 下含义不同。

## Neighbor retrieval 设计

输入：

```text
query_smiles
top_k_per_group: default 3
min_similarity: default 0.3
groups: optional list[Tier.endpoint_group]
exclude_exact: default true
```

流程：

```text
1. 标准化 query molecule，生成 canonical SMILES、InChIKey、Morgan fingerprint。
2. 按 evidence library 中的 Tier.endpoint_group 建立 group membership。
3. 对每个 group 独立计算 query 与该 group molecule fingerprints 的 Tanimoto。
4. 每个 group 返回 top 3 non-identical neighbors，默认过滤 Tanimoto < 0.3 的 very distant analog。
5. 对同一 neighbor molecule 聚合其在该 group 下的所有 assay/activity evidence。
6. 返回 group-level retrieval payload。
```

同一分子排除标准：

```text
same molecule_chembl_id
same full standard_inchi_key
same InChIKey connectivity layer, i.e. the first block before "-"
same canonical_smiles
```

相似度 bucket：

```text
very_close_analog: Tanimoto >= 0.95
close_analog: 0.80 <= Tanimoto < 0.95
moderate_analog: 0.60 <= Tanimoto < 0.80
weak_analog: 0.40 <= Tanimoto < 0.60
distant_analog: 0.20 <= Tanimoto < 0.40
very_distant_analog: Tanimoto < 0.20
```

默认使用 `min_similarity=0.3` 过滤 very distant analog，减少 sparse group 中几乎不可迁移的 evidence。保留的低相似度 analog 仍应交给 group-level LLM 判断 transferability。LLM prompt 必须明确：`distant_analog` 和 `very_distant_analog` 不能作为正负证据，除非共享 scaffold 和 assay mechanism 有很强的药化理由。

当前实现会用预计算 ChEMBL fingerprints：

```text
tools/chembl_tool/chembl_data/chembl_36_fps/chembl_36.fps.gz
```

构建 BBB evidence molecule subset 的 fingerprint/group index。这个 subset 当前约 5 万个
molecule，query-time 对每个 group 做 `BulkTanimotoSimilarity` 足够快。后续如需扩展为
全 ChEMBL 背景邻居检索，可以在不改变 BBB MVP payload contract 的前提下新增背景索引。

### Query ChEMBL exact context / shared-assay enrichment

当前实现保留一个可选的 evidence-rich 增强：

```text
tools/chembl_tool/tasks/bbb_martins/chembl_exact_context.py
```

它可以用 query full InChIKey 查 ChEMBL exact molecule，读取 ChEMBL compound properties、
query molecule 的 BBB-relevant evidence rows，并在 retrieved neighbor assay 中查 query activity，
生成两类 shared-assay context：

```text
same_endpoint_activity:
  same assay_chembl_id + same normalized standard_type

same_assay_different_endpoint_activity:
  same assay_chembl_id + different standard_type
```

这类信息使用了 query molecule 的已知 ChEMBL 实验记录。为避免 prospective evaluation 中的数据泄漏，
默认必须关闭，不进入 retrieval、single-molecule prompt、group prompt 或 batch 评估。只有显式传：

```bash
--enable-chembl-exact-context
```

才允许作为 retrospective / evidence-rich case study 使用。默认 benchmark、accuracy 和 macro-F1 报告
都应保持该开关关闭。

## LLM reasoning 流程

BBB_Martins reasoning 当前分为并发 evidence branches 和 final summary。

### Group-level reasoning

每个 `Tier.endpoint_group` 独立执行：

```text
input:
  query molecule
  top 3 neighbors for this group
  cleaned raw ChEMBL assay/activity evidence rows
  tier and endpoint_group

available tools:
  mmp_structure_compare
  properties_compare

not sent:
  query standard_inchi_key
  query fingerprint
  group n_candidate_molecules
  evidence_direction
  evidence_strength
  endpoint_group_reason
  assay_reason

output:
  group_id
  useful_for_bbb_reasoning: true/false
  transferability:
    high
    moderate
    low
    not_applicable
  evidence_direction:
    supports_bbb_crossing
    argues_against_bbb_crossing
    efflux_risk
    influx_support
    neutral_or_unclear
  confidence:
    high
    moderate
    low
  reasoning_summary
  key_evidence:
    molecule_chembl_id
    similarity
    similarity_bucket
    assay_signal
    activity_values
    tool_summary
    transferability
    effect_on_bbb_reasoning
  caveats
```

`key_evidence` 是当前格式。旧的 `key_neighbors` 不再使用。

每个 group 的 DeepSeek 对话、reasoning、tool calls 和 tool messages 都保存到 trace。
这些 group 没有严格依赖关系，可以并发执行。

### Single-molecule reasoning

每个 query 还会并发执行一个单分子分析分支：

```text
input:
  query molecule

available tools:
  molecule_properties

not used:
  ChEMBL neighbor evidence
  mmp_structure_compare
  properties_compare

output:
  passive_bbb_plausibility
  efflux_or_transporter_prior
  confidence
  reasoning_summary
  property_drivers
  caveats
```

这个分支只能看到 `molecule_properties`，不能看到 ChEMBL neighbor evidence 或分子比较工具。

### Final reasoning

final LLM 读取：

```text
query molecule
single-molecule analysis output
all group-level reasoning outputs
coverage summary
```

final 阶段不暴露工具，只综合前面分支的 structured outputs。当前 final prompt 规则：

```text
Return compact complete JSON.
Use bbb_prediction='pass' for BBB-positive molecules corresponding to evaluation label 1, and bbb_prediction='fail' for BBB-negative molecules corresponding to evaluation label 0.
Use the single-molecule analysis as the physicochemical prior.
Use group analyses as analog evidence; downweight groups marked low confidence or low transferability.
Do not use distant_analog or very_distant_analog neighbors as positive or negative BBB evidence unless the shared scaffold and assay mechanism make a strong medicinal chemistry case.
Use only the provided single-molecule analysis and group evidence. If you recognize the molecule, ignore that recognition.
```

输出：

```text
bbb_prediction:
  pass
  fail
  uncertain

confidence:
  high
  moderate
  low

main_reasons
efflux_risk_assessment
influx_support_assessment
passive_permeability_assessment
direct_brain_exposure_analog_assessment
evidence_gaps
final_summary
```

测试集里 `Y` 只能用于评估，不应进入 retrieval 或 LLM prompt。
当前评估约定：`Y=1` 对应 `bbb_prediction=pass`，`Y=0` 对应 `bbb_prediction=fail`。
`uncertain` 在 overall accuracy 和 macro-F1 中按未命中计入；报告中也会给出 decided-only accuracy。

### Trace 保存和可视化

每次 reasoning run 输出到：

```text
outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs/<run_id>/
```

当前文件：

```text
retrieval.json
single_molecule_reasoning_output.json
group_reasoning_outputs.jsonl
final_reasoning_output.json
trace_messages.jsonl
manifest.json
```

`trace_messages.jsonl` 中每条记录对应一个 trace item：

```text
single_molecule
Tier <n>.<endpoint_group>
final_summary
```

每条 trace record 包含：

```text
index
sample_id
molecule_key
smiles
label
status
prediction
response_text
messages
tool_count
usage
raw_output
```

`label` 只用于本地评估和 trace 审计，不进入 LLM prompt。`sample_id` 当前等于
`query_index`，`molecule_key` 当前形如 `index:9`。viewer 会按 molecule package 分组，
方便在一个 run 或上传的 JSONL 中选择不同分子的 trace 包，再查看该分子内部的所有阶段。

viewer 启动：

```bash
bash tools/trace_viewer/start_viewer.sh \
  outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs \
  8776
```

然后打开：

```text
http://localhost:8776/.trace_viewer.html
```

常用 pipeline 命令：

```bash
# 完整运行一个 test_efflux 分子
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline \
  --query-index 0 \
  --top-k-per-group 3 \
  --min-similarity 0.3 \
  --max-workers 4 \
  --max-tool-rounds 10 \
  --timeout-s 300 \
  --max-tokens 8192 \
  --model deepseek-v4-pro \
  --run-id <run_id>

# 只重跑已有 run 的 final summary
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline \
  --resume-final-from-run-dir outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs/<run_id> \
  --timeout-s 300 \
  --max-tokens 8192 \
  --model deepseek-v4-pro

# 批量运行一个 JSONL 中的分子，并生成评估报告
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_batch \
  --input-jsonl data/processed/BBB_Martins/B3DB_cleaned/test/test_efflux.jsonl \
  --parallelism 1 \
  --group-workers 4 \
  --top-k-per-group 3 \
  --min-similarity 0.3 \
  --max-tool-rounds 10 \
  --timeout-s 300 \
  --max-tokens 8192 \
  --model deepseek-v4-pro \
  --batch-id <batch_id>
```

默认会把每个单分子 run 的 stderr 进度实时打印到控制台，例如
`[idx00003 stderr] [bbb_reasoning_pipeline] group done: ...`，同时完整保存到
`outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/logs/`。如果只想写日志文件、不想在控制台显示进度，可加
`--no-stream-logs`。

批量输出：

```text
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/manifest.json
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/predictions.jsonl
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/metrics.json
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/report.md
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/trace_messages.jsonl
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/logs/
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/runs/
```

batch 中每个 molecule 的独立 run 保留在 batch 目录内部：

```text
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/runs/<batch_id>_idx00000/
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/runs/<batch_id>_idx00001/
...
```

`trace_messages.jsonl` 会合并每个 molecule 的 trace。查看 batch trace 时启动 viewer 指向
`reasoning/batches`，然后选择 `<batch_id>`：

```bash
bash tools/trace_viewer/start_viewer.sh \
  outputs/chembl_tool/tasks/bbb_martins/reasoning/batches \
  8776
```

viewer 可以选择 `<batch_id>`，再通过 `Molecule package` 下拉框切换分子。
若传 `--no-save-trace`，不会生成 batch combined trace。

## FastAPI 常驻服务标准

服务启动时初始化慢资源，后续每次 tool invoke 复用已加载对象：

```text
RDKit standardization config
MolGpKa import/model state
AccFG import/state
mmpdb Python package import
后续其他慢启动 ML models 或大索引
```

当前 endpoint：

```text
GET /health
GET /tools
POST /tools/{tool_name}/invoke
POST /tools/invoke
POST /tools/{tool_name}
```

其中 `/tools/{tool_name}/invoke` 是标准通用工具接口。`/tools/invoke` 和 `/tools/{tool_name}` 是兼容入口。`/tasks/bbb_martins/retrieve`、`/tasks/bbb_martins/reason`、`/tasks/bbb_martins/predict` 后续如果需要 task-level orchestration 再新增。

### 通用 ToolRequest

```json
{
  "request_id": "string",
  "tool_name": "molecule_properties",
  "version": "v1",
  "input": {},
  "options": {
    "timeout_s": 120,
    "return_debug": false
  }
}
```

### 通用 ToolResponse

```json
{
  "request_id": "string",
  "tool_name": "molecule_properties",
  "version": "v1",
  "status": "ok",
  "output": {
    "text": "LLM-readable natural language output",
    "..."
  },
  "warnings": [],
  "errors": [],
  "metadata": {
    "started_at": "ISO-8601",
    "finished_at": "ISO-8601",
    "latency_ms": 0,
    "model_or_index_version": "string"
  }
}
```

失败时：

```json
{
  "status": "error",
  "output": null,
  "warnings": [],
  "errors": [
    {
      "code": "INVALID_SMILES",
      "message": "Could not parse query SMILES.",
      "recoverable": true
    }
  ]
}
```

## Tool: molecule_properties v1

输入：

```json
{
  "query_smiles": "CCO"
}
```

职责：

```text
1. 标准化 query SMILES，返回 canonical_smiles 和 standard_inchi_key。
2. 计算易解释 RDKit descriptors。
3. 通过 MolGpKa 计算 acidic/basic pKa 和 logD。
4. 通过 AccFG 识别最顶层 functional groups。
5. 生成自然语言 output.text，作为 LLM 唯一直接消费的工具文本。
```

主要输出字段：

```text
output.text
output.query
output.properties
output.functional_groups
output.raw_features
```

`properties` 至少覆盖：

```text
MolGpKa pKa/logD features
RDKit molecular weight
logP
TPSA
HBD/HBA
rotatable bonds
formal charge
heavy atom count
aromatic rings
fraction Csp3
QED
rule flags
```

## Tool: properties_compare v1

输入：

```json
{
  "query_smiles": "CCN(CC)CC",
  "reference_smiles": "CCO",
  "query_label": "query",
  "reference_label": "reference"
}
```

职责：

```text
1. 复用 registry 中同一个 molecule_properties 工具实例，避免重复初始化 MolGpKa/AccFG。
2. 比较 molecule_properties 中所有非 functional-group properties。
3. delta 定义为 query_value - reference_value。
4. 生成自然语言 output.text，说明哪些 properties 增加、降低或不适用。
```

主要输出字段：

```text
output.text
output.query
output.reference
output.comparisons
```

functional groups 不在此工具中比较。FG 信息只由 `molecule_properties` 单独返回。

## Tool: mmp_structure_compare v1

输入：

```json
{
  "query_smiles": "CCN(CC)CC",
  "reference_smiles": "CCO",
  "query_label": "query",
  "reference_label": "reference"
}
```

职责：

```text
1. 计算 Morgan fingerprint Tanimoto similarity。
2. 按 similarity bucket 标记结构相似度。
3. 调用 mmpdb 的 fragmentation / matched-pair 逻辑，描述可解释的 matched-pair transformation。
4. 计算 RDKit MCS coverage，辅助判断共同骨架比例。
5. 生成自然语言 output.text，作为 LLM 可读结构差异说明。
```

主要输出字段：

```text
output.text
output.query
output.reference
output.similarity
output.transformation
output.mcs
```

`mmp_structure_compare` 不返回 descriptor/property deltas。所有属性差异比较必须使用 `properties_compare`。

结构 similarity bucket：

```text
very_close_analog: Tanimoto >= 0.95
close_analog: 0.80 <= Tanimoto < 0.95
moderate_analog: 0.60 <= Tanimoto < 0.80
weak_analog: 0.40 <= Tanimoto < 0.60
distant_analog: 0.20 <= Tanimoto < 0.40
very_distant_analog: Tanimoto < 0.20
```

## ChEMBL neighbor retrieval 状态

`chembl_neighbors` 仍是 BBB_Martins reasoning 的重要 evidence retrieval 概念，但当前实现不是
DeepSeek function tool，也没有作为 `tools/service/` 的已注册常驻工具暴露。当前代码入口是：

```text
tools/chembl_tool/tasks/bbb_martins/retrieve_neighbors.py
```

当前调用方式：

```text
run_reasoning_pipeline.py 在 LLM 调用前读取 BBB neighbor index，
按 Tier.endpoint_group 预取每组 top 3 neighbors，
清理内部派生字段后把 neighbors/evidence_rows 放进 group-level user prompt。
```

因此 group-level DeepSeek 看到 neighbor evidence，但不能主动调用 `chembl_neighbors`。它能调用的工具只有：

```text
mmp_structure_compare
properties_compare
```

后续如果需要把 BBB evidence retrieval 也放进常驻服务，应复用现有 ToolRequest / ToolResponse contract，并把服务端启动时加载 BBB evidence library 和 fingerprint group index。不要复制服务框架。

## 并发策略

服务内部可并发执行：

```text
molecule_properties
properties_compare
mmp_structure_compare
ChEMBL neighbor retrieval / evidence prefetch for each group（pipeline 内部；后续可 service 化）
group-level LLM reasoning for each group
```

建议边界：

```text
1. retrieval / evidence prefetch 的逻辑粒度是 endpoint group。
2. group-level reasoning 并发粒度也是 endpoint group。
3. final reasoning 必须等待所有 group reasoning 完成。
4. 每个 query 要有 request_id/run_id，所有中间产物可追踪。
```

## 实施计划

### Phase 0: 文档与接口冻结

状态：

```text
已完成，并在本文件中记录当前工具入口和 LLM 可见输出约定。
```

当前冻结的 service tool 名称：

```text
molecule_properties
properties_compare
mmp_structure_compare
```

### Phase 1: BBB evidence library

状态：

```text
已实现核心入口：
tools/chembl_tool/tasks/bbb_martins/endpoint_groups.py
tools/chembl_tool/tasks/bbb_martins/build_evidence_library.py
```

目标保持不变：

```text
能从 v6 assay/activity 输出生成 molecule-level evidence。
每条 evidence 有 endpoint_group、evidence_direction、evidence_strength。
测试覆盖 Tier.endpoint_group 映射规则。
```

### Phase 2: BBB neighbor retrieval

状态：

```text
已实现核心入口：
tools/chembl_tool/tasks/bbb_martins/retrieve_neighbors.py
```

当前行为：

```text
给定一个 SMILES，每个 Tier.endpoint_group 返回 top 3 non-identical neighbors。
返回完整 assay description 和 activity rows。
能批量处理 test_efflux.jsonl。
默认使用 min_similarity=0.3 过滤 very distant analog；保留的低相似度 analog 继续标记 bucket。
```

### Phase 3: 常驻 FastAPI service

状态：

```text
已实现：
tools/service/app.py
tools/service/config.py
tools/service/registry.py
tools/service/schemas.py
tools/service/errors.py
tools/service/tools/base.py
tools/service/tools/rdkit_properties.py
tools/service/tools/properties_compare.py
tools/service/tools/mmp_structure_compare.py
```

当前验收：

```text
服务启动时初始化 MolGpKa、AccFG、mmpdb 等慢资源。
POST /tools/{tool_name}/invoke 可调用 molecule_properties、properties_compare、mmp_structure_compare。
GET /tools 可列出工具 schema 和版本。
所有工具 output.text 为 LLM 可见文本，数字最多保留两位小数。
```

注意：当前 `chembl_neighbors` 没有注册为 service tool，也不是 LLM 可调用 tool；它是 BBB pipeline 内部的 evidence prefetch/context assembly。DeepSeek 只接收预取后的 group evidence。后续如果需要 service 化，应新增 `tools/service/tools/chembl_neighbors.py`，但不要替换或复制现有通用工具框架。

### Phase 4: LLM reasoning payloads

状态：

```text
已实现核心 orchestration：
tools/chembl_tool/tasks/bbb_martins/run_reasoning_pipeline.py
```

当前行为：

```text
1. 从 test_efflux.jsonl 读取 query。
2. 用 retrieve_neighbors.py 获取每个 Tier.endpoint_group 的 top 3 neighbors。
3. 并发执行 single-molecule analysis，只暴露 molecule_properties。
4. 并发执行每个 group-level analysis，只暴露 mmp_structure_compare 和 properties_compare。
5. 等待所有 group 和 single 分支完成后执行 final summary，不暴露工具。
6. 保存 retrieval、single、group、final、trace_messages 和 manifest。
7. 支持 --resume-final-from-run-dir 复用已有 retrieval/single/group 输出，只重跑 final summary。
```

当前 DeepSeek 调用约定：

```text
OpenAI SDK
base_url=https://api.deepseek.com
model=deepseek-v4-pro
thinking enabled
reasoning_effort=high
```

当前 trace 验收：

```text
完整保存 system/user/assistant/tool messages。
保存 assistant reasoning_content。
保存 assistant tool_calls。
保存 ToolResponse.output.text 作为 tool message content。
保存 molecule_key，方便 viewer 按分子 trace package 分组。
```

### Phase 5: BBB_Martins evaluation

状态：

```text
已完成单分子端到端 smoke runs，输出目录：
outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs/<run_id>/
```

已验证的示例：

```text
query_index=0:
  run_id=first_efflux_full_key_evidence_20260505_182029
  final prediction=fail
  confidence=moderate
  19 group outputs, 21 trace records, 57 key_evidence items

query_index=9:
  run_id=last_efflux_full_key_evidence_20260505_185201
  final prediction=fail after final-only prompt rerun
  confidence=moderate
  19 group outputs, 21 trace records, 57 key_evidence items
```

后续 evaluation 工作：

```text
对 test_efflux.jsonl 批量运行 retrieval + reasoning。
评估时只在最后对照 Y，不把 Y 传给工具或 LLM。
报告 coverage、prediction accuracy、uncertain rate、典型成功/失败案例。
支持一个 run 中包含多个 molecule trace package，viewer 按 molecule_key 分组浏览。
```

### Phase 6: 扩展到其他任务

新任务只新增：

```text
tools/<domain_tool>/tasks/<task_name>/
tools/service/tasks/<task_name>.py
```

对应输出统一放在：

```text
outputs/<domain_tool>/tasks/<task_name>/
  assay_screening/
  evidence_library/
  reasoning/
```

如果需要新模型或新索引，则新增：

```text
tools/service/tools/<tool_name>.py
```

不要复制已有服务框架、tool schema、request/response 标准。

## 当前不做的事情

```text
不训练 BBB classifier。
不把每个 assay 作为独立 retrieval group。
不把 IC50/inhibition 直接解释成 substrate 或 transport。
不把 test_efflux.jsonl 的 Y 暴露给 retrieval 或 LLM prompt。
不在每次 query 时重新 import 或初始化 MolGpKa、AccFG、mmpdb、ML model 或大索引。
不把 descriptor/property deltas 放进 mmp_structure_compare；属性差异统一走 properties_compare。
不把工具结构化 JSON 整包展示给 LLM；LLM 默认只看 output.text。
不为已安装的 mmpdb 增加源码路径环境变量。
```

## 下一步

当前通用工具层暂时冻结，BBB_Martins 单分子端到端 MVP 已能运行。下一步优先级：

```text
1. 批量评估 test_efflux.jsonl，形成 prediction/label 对照表和错误分析。
2. 继续审计 final summary 的证据加权，必要时增加 explicit adjudication fields。
3. 将 reasoning run 的多分子输出组织成更稳定的 batch run layout。
4. 如果需要让其他系统复用 retrieval，再把 chembl_neighbors 包装为 service tool 或 task endpoint；当前 BBB pipeline 继续把它作为内部 evidence prefetch。
5. 后续接入更多常驻 ML tools，例如更慢的 pKa/logD、solubility、PK 或 toxicity 模型。
```
