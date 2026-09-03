# AGENTS.md: ChEMBL BBB Assay 筛查实现说明

## Testing discipline

Do not add circular tests that merely assert newly written prompt prose or copy
implementation literals into the test. Prompt wording is validated with reviewed
input/output fixtures or a real pilot/evaluation. Automated tests should cover
executable behavior, failure modes, schemas, rendering validity, and provenance.

## Env instructions

If you are on `node002`, default to the `vllm` conda environment when you need RDKit or the local project dependencies. conda is at: /data1/tianang/anaconda3/condabin/conda

## 目标

实现一个可复用脚本，从 ChEMBL SQLite/数据库中筛出能帮助判断分子是否能够通过 BBB 的 assay，并按证据强度分层排序。

当前状态说明：

```text
本文件主要保留早期 assay screening 计划和规则说明。
当前正式实现已经迁移到 tools/chembl_tool/tasks/bbb_martins/。
第二阶段已经实现 molecule-level evidence library、neighbor retrieval 和 reasoning pipeline。
```

当前关键入口：

```text
tools/chembl_tool/tasks/bbb_martins/screen_assays.py
tools/chembl_tool/tasks/bbb_martins/rescore_outputs.py
tools/chembl_tool/tasks/bbb_martins/endpoint_groups.py
tools/chembl_tool/tasks/bbb_martins/build_evidence_library.py
tools/chembl_tool/tasks/bbb_martins/retrieve_neighbors.py
tools/chembl_tool/tasks/bbb_martins/run_reasoning_pipeline.py
```

当前 ChEMBL neighbor retrieval 的角色：

```text
不是 DeepSeek 可调用 tool。
不是当前 FastAPI service tool。
是 run_reasoning_pipeline.py 在 LLM 调用前执行的 evidence prefetch / context assembly。
group-level DeepSeek 只能调用 mmp_structure_compare 和 properties_compare。
single-molecule DeepSeek 只能调用 molecule_properties。
```

`screen_assays.py` 的 raw screening 输出：

```text
outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw/bbb_assay_candidates.csv
outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw/bbb_assay_candidates.jsonl
outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw/bbb_assay_report.md
```

当前正式 BBB_Martins task 输出已经按 pipeline stage 归档：

```text
outputs/chembl_tool/tasks/bbb_martins/assay_screening/v6/
outputs/chembl_tool/tasks/bbb_martins/evidence_library/
outputs/chembl_tool/tasks/bbb_martins/reasoning/runs/
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/
```

不要训练模型。只做 deterministic assay 检索、打分、分层、导出。

## 输入

优先支持 ChEMBL SQLite，例如：

```bash
python -m tools.chembl_tool.tasks.bbb_martins.screen_assays \
  --chembl-sqlite /data1/tianang/Projects/TxAgent/tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db \
  --out-dir outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw \
  --min-score 40 \
  --progress-every 10000
```

如果当前项目已有 ChEMBL 访问工具，优先复用，不要重复造复杂 ORM。

---

## 需要读取的 ChEMBL 表

至少使用：

```text
assays
activities
target_dictionary
target_components
component_sequences
component_synonyms
molecule_dictionary
compound_structures
```

建议额外使用：

```text
docs
```

`docs.title` / `docs.abstract` 只作为弱召回辅助，不能单独决定保留一个 assay。它们用于发现 BBB 论文中的 assay，但最终仍必须由 assay description、activity endpoint、cell model 或 target annotation 支撑。

核心字段：

```text
assays.assay_id
assays.doc_id
assays.chembl_id
assays.description
assays.assay_type
assays.assay_test_type
assays.assay_category
assays.assay_cell_type
assays.assay_tissue
assays.assay_organism
assays.confidence_score
assays.relationship_type
assays.tid

activities.assay_id
activities.standard_type
activities.standard_relation
activities.standard_value
activities.standard_units
activities.molregno
activities.pchembl_value
activities.data_validity_comment
activities.potential_duplicate
activities.activity_comment

target_dictionary.tid
target_dictionary.chembl_id
target_dictionary.pref_name
target_dictionary.target_type
target_dictionary.organism

component_sequences.component_id
component_sequences.accession
component_sequences.description
component_synonyms.component_id
component_synonyms.component_synonym
component_synonyms.syn_type

docs.doc_id
docs.title
docs.abstract
docs.pubmed_id
docs.doi
```

字段不存在时要 graceful fallback，不要直接崩溃。

---

## 输出字段

`bbb_assay_candidates.csv/jsonl` 至少包含：

```text
assay_chembl_id
assay_id
tier
score
assay_type
description
target_chembl_id
target_pref_name
target_genes
target_synonyms
organism
confidence_score
relationship_type
assay_cell_type
assay_tissue
n_activities
n_unique_molecules
standard_types
matched_keywords
matched_endpoints
matched_targets
negative_flags
weak_context_flags
reason
keep_for_bbb_reasoning
```

`reason` must explain in one English sentence why the assay is retained, for example:

```text
Matches brain/plasma and the logBB endpoint, so it is direct brain-exposure evidence.
```

---

## 文本标准化

所有 keyword matching 前先标准化：

```python
text = text.lower()
text = text.replace("-", " ").replace("/", " ")
text = text.replace(",", " ")
text = collapse_multiple_spaces(text)
```

同时保留原始 description 用于输出。

注意兼容这些写法：

```text
blood-brain barrier / blood brain barrier
P-gp / P gp / P glycoprotein / P-glycoprotein
Caco-2 / Caco2
MDCK-MDR1 / MDR1-MDCK / MDCKII-MDR1 / MDCK2-MDR1
B-A/A-B / BA/AB / basolateral to apical / apical to basolateral
Kp,uu / Kpuu / Kp uu / K(p,uu,brain)
brain/plasma / brain to plasma / brain:blood / brain blood
CSF/plasma / CSF to plasma / cerebrospinal fluid plasma
```

实现上优先用 token/phrase matching 或预编译正则，避免单字符或过短 token 造成误报。

---

## 证据分层

### Tier 1: 直接 BBB / brain exposure，最高优先级

这一层表示分子已经有脑暴露、脑/血浆、脑/血、CSF 或 BBB 通过相关实验读数。

关键词：

```text
blood brain barrier
bbb permeability
brain penetration
brain uptake
brain exposure
brain plasma
brain to plasma
brain blood
brain to blood
brain concentration
brain level
brain auc
logbb
kp uu brain
kpuu brain
k p uu brain
unbound brain
unbound plasma
fraction unbound brain
fu brain
brain binding
in situ brain perfusion
brain perfusion
brain uptake index
brain penetration index
bui
bpi
permeability surface area
ps product
csf plasma
csf to plasma
cerebrospinal fluid plasma
```

endpoint：

```text
logbb
k(p,uu,brain)
kp,uu,brain
kp uu brain
kpuu brain
brain/plasma
brain plasma
brain/plasma ratio
brain to plasma ratio
brain/blood
brain blood
brain to blood ratio
brain uptake
brain uptake index
brain penetration index
brain concentration
brain level
csf/plasma
csf plasma
csf to plasma ratio
kin
ps product
```

基础分：`100`

注意：

```text
kp
ps
kin
csf
```

这些短词不能单独触发 Tier 1。必须和 brain、BBB、CSF/plasma、perfusion、unbound brain 等上下文同时出现，或者标准 endpoint 是明确的 `K(p,uu,brain)` / `brain/plasma` / `logBB`。

---

### Tier 2: 被动通透 / barrier model permeability

这一层表示分子有 BBB 相关体外屏障模型、PAMPA-BBB、brain endothelial cell model、MDCK/Caco-2 permeability 或 Papp 证据。Caco-2 更偏肠吸收，但可以作为 general permeability/P-gp 辅助证据。

关键词：

```text
pampa bbb
bbb pampa
parallel artificial membrane
mdck mdr1
mdr1 mdck
mdckii mdr1
mdck2 mdr1
mdck-mdr1
mdck
mdckii
mdck2
caco 2
caco2
hcmec d3
bend 3
bmec
bbmec
rbec
brain endothelial
brain microvessel endothelial
transwell
apical to basolateral
basolateral to apical
papp
logpapp
apparent permeability
permeability coefficient
```

endpoint：

```text
papp
logpapp
apparent permeability
permeability coefficient
caco-2 papp
caco-2 permeability
efflux ratio
```

基础分：`70`

注意：

```text
pampa
permeability
pe
a b
b a
```

这些词不能单独触发 Tier 2。`pampa` 优先要求同时出现 BBB、PAMPA-BBB 或 parallel artificial membrane；`permeability` 必须和 Caco-2、MDCK、PAMPA、BBB、brain endothelial、transwell、Papp 或 efflux ratio 同时出现；`a b` / `b a` 只在 basolateral/apical 或 Papp 上下文中使用。

---

### Tier 3: 主动外排 / efflux transporter

这一层表示分子可能是 BBB 外排转运体底物、抑制剂或存在 bidirectional transport / efflux ratio 证据。

重点 transporter：

```text
ABCB1, MDR1, P-gp, P glycoprotein, P-glycoprotein
ABCG2, BCRP, breast cancer resistance protein
ABCC1, MRP1
ABCC2, MRP2
ABCC4, MRP4
ABCC5, MRP5
```

关键词：

```text
p gp
p glycoprotein
pglycoprotein
mdr1
abcb1
bcrp
abcg2
breast cancer resistance protein
mrp1
mrp2
mrp4
mrp5
abcc1
abcc2
abcc4
abcc5
efflux
efflux ratio
substrate
bidirectional
basolateral to apical
apical to basolateral
rhodamine 123
calcein am
digoxin
prazosin
vinblastine
tariquidar
elacridar
verapamil
cyclosporin a
ko143
atpase
```

endpoint：

```text
efflux ratio
substrate
transport
papp
atpase
ic50
ki
```

基础分：`85`

注意：

`P-gp/BCRP substrate`、`bidirectional Papp`、`efflux ratio` 比单纯 inhibitor `IC50` 更重要。`transport`、`substrate`、`IC50`、`Ki` 不能单独触发 Tier 3；必须和 ABCB1/ABCG2/ABCC gene symbol、P-gp/BCRP/MRP target、efflux ratio、bidirectional assay 或典型 probe/inhibitor context 同时出现。

---

### Tier 4: 主动摄取 / influx transporter

这一层表示分子有可能借助 BBB 相关摄取转运体进入脑内。

重点 transporter：

```text
SLC7A5, LAT1
SLC2A1, GLUT1
SLCO1A2, OATP1A2
SLC22A8, OAT3
SLC16A1, MCT1
TFRC, transferrin receptor
```

关键词：

```text
lat1
slc7a5
glut1
slc2a1
oatp1a2
slco1a2
oat3
slc22a8
mct1
slc16a1
transferrin receptor
tfrc
influx
uptake
transport
```

endpoint：

```text
uptake
transport
substrate
```

基础分：`55`

注意：

`uptake` 和 `transport` 是弱词，不能单独触发 Tier 4。必须同时命中 Tier 4 transporter target/gene/synonym，或 description 明确是 LAT1/GLUT1/OATP1A2/OAT3/MCT1/TFRC 相关 assay。

---

## 打分规则

对每个 assay 计算 `score`：

```text
score = tier_base_score
      + keyword_bonus
      + endpoint_bonus
      + target_bonus
      + quality_bonus
      - weak_context_penalty
      - negative_penalty
```

建议规则：

```text
每个强关键词 +3，最多 +20
命中关键 endpoint +15
命中多个 Tier 1 direct endpoint +10
命中 ABCB1/ABCG2 单蛋白 target +20
命中 ABCC/SLC BBB-relevant transporter target +10
命中 transporter substrate/efflux ratio/bidirectional Papp +20
confidence_score >= 8 +10
relationship_type == 'D' +5
n_unique_molecules >= 20 +5
n_unique_molecules >= 100 +10
只命中弱词但上下文不足 -50
```

弱词包括：

```text
kp
ps
kin
csf
pampa
permeability
pe
a b
b a
transport
uptake
substrate
ic50
ki
```

弱词必须满足本文件各 Tier 中的上下文要求，否则不要作为保留依据。

负向规则：

```text
如果只命中 CNS receptor binding，而没有 BBB/permeability/transport 关键词，-80
如果只命中 cytotoxicity/cell viability/tumor proliferation，-60
如果只命中 generic kinase/receptor/enzyme inhibition，-60
如果只命中 generic uptake，例如 thymidine/glucose/oxygen/tumor uptake，-60
如果 activity 全部 invalid 或 data_validity_comment 严重异常，-30
```

负向关键词：

```text
cytotoxicity
cell viability
proliferation
tumor
cancer cell line
dopamine receptor
serotonin receptor
gaba receptor
muscarinic receptor
histamine receptor
opioid receptor
cannabinoid receptor
kinase inhibition
thymidine uptake
glucose uptake
oxygen uptake
tumor uptake
```

但如果同时命中 direct BBB、PAMPA-BBB、MDCK-MDR1、Caco-2 Papp、ABCB1、ABCG2 等强证据，不要因为出现 receptor/cell line 就直接丢弃。

---

## 保留规则

默认保留：

```text
score >= min_score
```

并且至少满足之一：

```text
命中 Tier 1 direct brain/BBB exposure 关键词或 endpoint
命中 Tier 2 BBB-relevant permeability/cell model 关键词或 endpoint
命中 Tier 3 efflux transporter target/关键词/endpoint
命中 Tier 4 influx transporter target/关键词/endpoint
```

对于 transporter target assay：

```text
如果 target 是 ABCB1/ABCG2/ABCC/SLC 等单蛋白，优先要求 confidence_score >= 8 或 relationship_type == 'D'
如果 confidence_score 低，但 description/endpoint 明确是 efflux ratio、substrate、bidirectional Papp，也可以保留
```

对于 cell-based assay：

```text
Caco-2、MDCK、hCMEC/D3、bEnd.3、BMEC、BBMEC、RBEC 不要因为 confidence_score 低就丢弃
但必须有 permeability、Papp、efflux ratio、transwell 或 BBB/barrier context
```

---

## 实现结构

当前 `tools/chembl_tool` 已经按可复用 Python 包结构实现。不要新增平级的 `tools/chembl_bbb/`，否则后续每个 ChEMBL 任务都会形成一套互相割裂的工具。

当前结构仍应保持：`common/` 放所有任务都能复用的 SQLite/schema/assay 聚合逻辑，`tasks/` 放具体任务。BBB_Martins 只是一个 task。

当前核心结构：

```text
tools/chembl_tool/
  __init__.py
  common/
    __init__.py
    sqlite.py
    schema.py
    assay_loader.py
    text.py
    export.py
  tasks/
    __init__.py
    bbb_martins/
      __init__.py
      endpoint_groups.py
      build_evidence_library.py
      retrieve_neighbors.py
      run_reasoning_pipeline.py
      screen_assays.py
      rules.py
      scoring.py
      report.py
  utils/
    BBB_Martins/
      AGENTS.md
tests/
  chembl_tool/
    common/
      test_text.py
      test_schema.py
      test_assay_loader.py
    tasks/
      bbb_martins/
        test_rule_matching.py
        test_scoring.py
```

目录职责：

```text
common/sqlite.py: 连接 SQLite、执行查询、流式读取、检查表是否存在
common/schema.py: 字段存在性检查、graceful fallback、ChEMBL 版本/路径探测
common/assay_loader.py: 读取并聚合 assay metadata、activity summary、target annotations、doc annotations
common/text.py: 通用文本标准化、phrase/token matching helper
common/export.py: CSV/JSONL 导出 helper

tasks/bbb_martins/rules.py: BBB_Martins 专属规则、Tier 定义、target gene 列表、弱词上下文规则
tasks/bbb_martins/scoring.py: BBB_Martins 专属 match/score/keep 逻辑
tasks/bbb_martins/report.py: BBB_Martins 报告
tasks/bbb_martins/screen_assays.py: CLI 入口，只做参数解析、调用 common loader、调用 task scoring、导出结果
```

这样以后新增其他 ChEMBL 任务时，只需要新增：

```text
tools/chembl_tool/tasks/<task_name>/
  __init__.py
  screen_assays.py 或 run.py
  rules.py
  scoring.py
  report.py
```

不要在新任务里复制 `assay_loader.py`、SQLite 连接、target synonym 聚合、文本标准化等通用逻辑。

---

## 主要函数

### `common/sqlite.py`

```python
connect_sqlite(path: str) -> sqlite3.Connection
table_exists(conn, table_name: str) -> bool
get_table_columns(conn, table_name: str) -> set[str]
read_sql(conn, query: str, params: dict | None = None) -> DataFrame
```

### `common/schema.py`

```python
require_any_columns(conn, table: str, candidates: list[str]) -> list[str]
build_select_columns(conn, table: str, requested: dict[str, str | None]) -> list[str]
```

字段不存在时在这里统一 graceful fallback，不要让 task 代码到处写 `PRAGMA table_info`。

### `common/assay_loader.py`

负责读取并聚合 assay 信息：

```python
load_assay_metadata(conn) -> DataFrame
load_activity_summary(conn) -> DataFrame
load_target_annotations(conn) -> DataFrame
load_doc_annotations(conn) -> DataFrame
merge_assay_table(...) -> DataFrame
```

每个 assay 聚合：

```text
standard_types: 去重列表
n_activities: activity 数量
n_unique_molecules: molregno 去重数量
target_genes / target_synonyms: 去重列表
doc_title / doc_abstract: 可选弱召回字段
```

### `common/text.py`

```python
normalize_text(text: str) -> str
contains_phrase(text: str, phrase: str) -> bool
match_phrases(text: str, phrases: Iterable[str]) -> list[str]
```

### `tasks/bbb_martins/rules.py`

```python
BBB_RULES = {
    "direct_bbb": {...},
    "passive_permeability": {...},
    "efflux": {...},
    "influx": {...},
}

NEGATIVE_RULES = {...}
WEAK_CONTEXT_RULES = {...}
TRANSPORTER_TARGETS = {...}
```

### `tasks/bbb_martins/scoring.py`

```python
match_rules(row) -> MatchResult
score_assay(row, matches) -> ScoredAssay
should_keep_assay(row, scored) -> bool
```

### `tasks/bbb_martins/report.py`

生成 markdown 报告：

```text
总 assay 数
候选 assay 数
各 tier 数量
Top 50 direct BBB assays
Top 50 passive permeability assays
Top 50 efflux assays
Top 50 influx assays
弱词命中但被过滤的示例
负向规则过滤的示例
```

### `tasks/bbb_martins/screen_assays.py`

CLI 入口：

```python
main(argv: list[str] | None = None) -> int
```

职责：

```text
解析参数
连接 ChEMBL SQLite
调用 common.assay_loader 构造 assay-level table
调用 tasks.bbb_martins.scoring 打分和过滤
导出 CSV/JSONL/report
可选导出候选 assay 的 activity evidence
按 `--progress-every` 输出扫描进度
```

---

## SQL 查询建议

不要一次性拉出所有 activity 明细用于最终表。先聚合：

```sql
SELECT
  assay_id,
  COUNT(*) AS n_activities,
  COUNT(DISTINCT molregno) AS n_unique_molecules,
  GROUP_CONCAT(DISTINCT LOWER(standard_type)) AS standard_types
FROM activities
WHERE standard_type IS NOT NULL
GROUP BY assay_id;
```

assay metadata：

```sql
SELECT
  a.assay_id,
  a.chembl_id AS assay_chembl_id,
  a.description,
  a.assay_type,
  a.assay_test_type,
  a.assay_category,
  a.assay_cell_type,
  a.assay_tissue,
  a.assay_organism,
  a.confidence_score,
  a.relationship_type,
  a.tid,
  a.doc_id,
  td.chembl_id AS target_chembl_id,
  td.pref_name AS target_pref_name,
  td.target_type,
  td.organism AS target_organism
FROM assays a
LEFT JOIN target_dictionary td ON a.tid = td.tid;
```

component/synonym 信息单独聚合后按 `tid` merge。gene symbol 必须优先来自 `component_synonyms.syn_type = 'GENE_SYMBOL'`，同义词可同时保留 `UNIPROT` / `GENE_SYMBOL_OTHER` 等。

doc metadata 可选：

```sql
SELECT
  doc_id,
  title,
  abstract,
  pubmed_id,
  doi
FROM docs;
```

---

## 可选：导出 activity evidence

增加参数：

```bash
--export-activities
```

如果开启，对候选 assay 额外导出：

```text
outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw/bbb_activity_evidence.csv
```

字段：

```text
assay_chembl_id
molecule_chembl_id
canonical_smiles
standard_type
standard_relation
standard_value
standard_units
pchembl_value
data_validity_comment
activity_comment
```

只导出候选 assay 对应 activities，避免全库导出过大。

---

## 当前实现：相似分子 evidence retrieval

筛出 BBB-relevant assay 只是第一步。当前已经构建 molecule-level BBB evidence library，用于给定一个新分子时，检索有 BBB 相关实验读数的相似分子，并用这些 analog evidence 辅助判断 query 是否可能通过 BBB。

### Evidence library

基于候选 assay 的 activity evidence，当前由 `build_evidence_library.py` 构建分子级证据表：

```text
molecule_chembl_id
canonical_smiles
standard_inchi_key
assay_chembl_id
assay_tier
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
evidence_direction
evidence_strength
evidence_reason
```

`evidence_direction` 用于区分证据含义：

```text
supports_bbb_crossing
argues_against_bbb_crossing
efflux_risk
influx_support
context_dependent
unknown_direction
```

不同 endpoint 不要简单混平均。`logBB`、`Kp,uu,brain`、brain/plasma、CSF/plasma、Papp、efflux ratio、transporter substrate/inhibition 的含义不同，需要分别解释后再聚合。

### Query-time retrieval

给一个 query molecule 时：

```text
1. 标准化 query molecule，生成 canonical SMILES / InChIKey / Morgan fingerprint
2. 在 evidence library 中排除同一分子
3. 用 Morgan fingerprint Tanimoto 检索非同一分子的相似邻居
4. 按 similarity bucket、assay tier、endpoint type 聚合证据
5. 输出 analog evidence summary，而不是直接给绝对判断
```

必须避免直接 retrieve 到 query 的 ground truth：

```text
排除 same molecule
排除 same full standard InChIKey
排除 same InChIKey connectivity layer（InChIKey 第一段）
排除 same canonical standardized SMILES
```

不要默认排除高相似 analog。只要不是同一个分子，`Tanimoto >= 0.95` 的邻居应保留，因为它们通常是最有价值的 analog evidence。

建议相似度分层：

```text
very_close_analog: Tanimoto >= 0.95，保留并标记
close_analog: 0.80 <= Tanimoto < 0.95
moderate_analog: 0.60 <= Tanimoto < 0.80
weak_analog: 0.40 <= Tanimoto < 0.60
distant_analog: 0.20 <= Tanimoto < 0.40
very_distant_analog: Tanimoto < 0.20
```

不要用 `0.40` 作为默认硬截断。每个 group 可以返回 top 3 non-identical neighbors，即使相似度很低也保留并标记 similarity bucket。后续 group-level reasoning LLM 应判断这些 analog 是否 transferable。`distant_analog` 和 `very_distant_analog` 不能作为强正负证据，除非共享 scaffold 和 assay mechanism 有很强的药化理由。

如果排除同一分子后没有可检索 BBB evidence 邻居，应明确返回：

```text
no reliable analog evidence found
```

不要在 evidence sparse 的情况下硬判定能否通过 BBB。

### 当前输出与入口

早期计划里的 `retrieve_analogs` CLI 没有采用。当前已实现入口是：

```bash
python -m tools.chembl_tool.tasks.bbb_martins.retrieve_neighbors \
  --query-smiles "CCN(CC)..."
```

端到端 reasoning 入口：

```bash
python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline \
  --query-index 0 \
  --top-k-per-group 3 \
  --model deepseek-v4-pro \
  --run-id <run_id>
```

只重跑 final summary：

```bash
python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline \
  --resume-final-from-run-dir outputs/chembl_tool/tasks/bbb_martins/reasoning/runs/<run_id>
```

当前 reasoning run 输出：

```text
outputs/chembl_tool/tasks/bbb_martins/reasoning/runs/<run_id>/retrieval.json
outputs/chembl_tool/tasks/bbb_martins/reasoning/runs/<run_id>/single_molecule_reasoning_output.json
outputs/chembl_tool/tasks/bbb_martins/reasoning/runs/<run_id>/group_reasoning_outputs.jsonl
outputs/chembl_tool/tasks/bbb_martins/reasoning/runs/<run_id>/final_reasoning_output.json
outputs/chembl_tool/tasks/bbb_martins/reasoning/runs/<run_id>/trace_messages.jsonl
outputs/chembl_tool/tasks/bbb_martins/reasoning/runs/<run_id>/manifest.json
```

---

## 测试要求

必须有最小单元测试：

```text
P-gp、P gp、P glycoprotein、P-glycoprotein 能归一到 efflux
Caco-2、Caco2 都能命中 passive permeability
blood-brain barrier、blood brain barrier 都能命中 direct BBB
K(p,uu,brain)、Kp,uu brain、Kpuu brain 都能命中 direct BBB
MDCK-MDR1 同时命中 passive 和 efflux，但最终 tier 应偏 efflux/passive 高分
只有 dopamine receptor binding 不应被保留为 BBB evidence
ABCB1 target + substrate endpoint 应高分保留
只有 generic permeability 不应保留
glucose uptake 只有在 GLUT1/SLC2A1 target context 下才可作为 influx evidence
```

测试不要依赖完整 ChEMBL 数据库，用小 DataFrame/mock SQLite 即可。

---

## 验收标准

完成后应能：

```bash
python -m tools.chembl_tool.tasks.bbb_martins.screen_assays \
  --chembl-sqlite /data1/tianang/Projects/TxAgent/tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db \
  --out-dir outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw \
  --min-score 40 \
  --progress-every 10000
```

成功生成：

```text
bbb_assay_candidates.csv
bbb_assay_candidates.jsonl
bbb_assay_report.md
```

报告中必须能看到以下类别的候选：

```text
Direct BBB / brain exposure
Passive permeability / PAMPA-BBB / Caco-2 / MDCK / brain endothelial model
Efflux transporter / ABCB1 / ABCG2 / P-gp / BCRP / efflux ratio
Influx transporter / LAT1 / GLUT1 / OATP / MCT1
```

## 注意事项

1. 不要把 CNS receptor binding 当作 BBB 通过证据。
2. P-gp inhibitor 不等于 P-gp substrate，substrate/efflux ratio 权重更高。
3. PAMPA-BBB 只能说明 passive permeability，不能说明没有 efflux。
4. Caco-2 更偏肠吸收，但可作为 general permeability/P-gp 辅助证据。
5. MDCK-MDR1、bidirectional Papp、efflux ratio 对 BBB reasoning 很有价值。
6. 直接 brain/plasma、logBB、Kp,uu,brain、brain perfusion、CSF/plasma evidence 优先级最高。
7. 所有规则放在可编辑配置或 `rules.py`，不要硬编码散落在脚本各处。
