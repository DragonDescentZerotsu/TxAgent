# ChEMBL evidence-distance expansion：D/C/H1/H2 定义与代码设计

> 2026-07-23 status：四任务 census 得到 `0/4` publishable strict H2，因此本文的 strict H1/H2
> 四级主曲线已判定 no-go。本文保留为 strict causal secondary/historical contract，不得继续物化
> `D+C+H1+H2` 主实验。新的 controlled relevance-dilution 主实验见
> `RELEVANCE_DILUTION_EXPERIMENT_PLAN.md`。

本文档冻结 E12 前两步：如何可执行地定义 one-hop/two-hop，以及如何在不改变现有论文 pipeline 的前提下实现
独立的 cumulative distance-expansion 实验。本文档暂不列出四个 task 需要新增的具体 assay；那是 graph/schema
实现完成后的第三步。

## 1. 研究对象与非目标

E12 要回答：在同一个 ChEMBL 数据源内，保留现有直接和核心机制 evidence，再逐步加入机制上更远但数量更多的
assay evidence，LLM 的 macro-F1 在哪里停止改善或开始下降？

这里的“同一个 ChEMBL 数据源”覆盖整条曲线，而不只是新增层：`D`、`C`、`H1`、`H2` 都来自同一冻结
ChEMBL release。`D/C` 不从 Starling 继承，`H1/H2` 也不运行 Starling；禁止构造
`Starling D/C + ChEMBL H1/H2` 或任何其它跨 source prefix。四层必须共享 assay/activity 标准化、
measurement-quality gate、molecule aggregation、provenance contract 和 source manifest。Starling 与 ChEMBL
的比较保留在独立的 RQ2/E2 matched-scope source experiment 中。

这里的 distance 是 **task relevance distance** 的一个预注册 proxy，不是 assay measurement quality：

```text
mechanism distance  -> 由 biological graph 的额外推理桥数量定义
scope match         -> 物种、组织、route、assay system 是否可迁移，单独记录
measurement quality -> assay/measurement/provenance 是否可靠，单独记录
quantity            -> 实际进入 retrieval/prompt 的 family、assay、unique molecule、row 和 token 数
```

不同物种、组织或 assay format 本身不增加 hop；低质量记录也不会因为“更远”而被允许进入。所有层级使用同一
measurement-quality gate。E12 研究的是 relevance 与 quantity 的联合扩展，不把“远”写成“数据本身更差”。

## 2. 一句话定义

**one-hop assay**：它测量的 biological state/process 不在当前 pipeline 的冻结 evidence envelope 中；相对它所
归属的一个 C mechanism family，只需要增加一条有方向、有适用条件、有文献依据的生物学推理桥。

**two-hop assay**：它不在当前 envelope，也不存在到任何 D/C measured state 的合格一跳路径；相对同一 C
family 的 H1 child，需要再增加一条合格推理桥，因此到对应 C family 的最短合格路径为两条边。

例如，假设当前 core 已经接受 `process Z`：

```text
assay measures A -> Z                       one-hop extension
assay measures A -> intermediate Y -> Z    two-hop extension
```

如果实际上存在 Y，但只因为 A 和 Z 常在同一领域出现就省略 Y，不能把 two-hop 写成 one-hop。

## 3. D/C/H1/H2 层级树的精确定义

每个 task 先从当前 `experiment_config.py` 冻结两个互斥集合：

```text
D = 当前 ChEMBL direct_groups 解析得到的 source groups
C = 当前 ChEMBL mechanism_groups 的 source-group union 减去 D
```

因此必须满足：

```text
D ∩ C = empty
D ∪ C = 当前 chembl_full_flat / chembl_full_mechanism 的 evidence union
```

D/C 不是 hop 编号，也不是重新解释旧 Tier。D 是层级树的 root evidence；当前 paper-facing C mechanism
families 是 D 的第一圈子节点。每个 C family 独立向外延伸一条纵向链，不能把全局 H1/H2 当成与所有 C family
平行的 source groups：

```text
D root
  C.family_a
    H1.family_a                     exactly one aggregate child
      H2.family_a                   zero or one aggregate child
  C.family_b
    H1.family_b                     exactly one aggregate child
      H2.family_b                   zero or one aggregate child
  ...
```

一个 H1/H2 tree node 是一个 retrieval/reasoning group，可以聚合多个 measurement families、targets 和 assay
IDs；但每条 assay 必须保留自己的 measured node、具体 admissible path、scope、quality 和 provenance。graph edge
连接的是 measured biological nodes，不是 assay ID 彼此连接。

这里的“C mechanism family”只指非 direct 的可解释机制分支；direct-exposure family 属于 D root，不再要求向下
扩展。若 endpoint normalization 在 direct tier 内留下 `context_dependent` 弱证据，该 source group 仍保留在
`D+C` evidence union 以保证旧 full condition 完全不变，但它不是一个可扩展 mechanism family，也不为了满足树形
外观强行创建 H1。每个 task 必须显式冻结 `c_family_ids`。BBB v3 冻结为
`Mechanism.tier_2`（passive/barrier）、`Mechanism.tier_3`（efflux）和 `Mechanism.tier_4`（influx）；
`Mechanism.tier_1` 是 direct-exposure organization，归 D root 语义。

令 `B = D ∪ C` 为冻结的 base measured-state envelope，`C_k` 为一个冻结 C mechanism family：

```text
H1_k:
  C_k 有且只有一个 H1 child spec；其中可纳入多个 measurement families f
  对每个 f，f 不属于 B，且 f 到 C_k 中任一 measured node 的最短合格路径为 1

H2_k:
  H1_k 至多有一个 H2 child spec；其中可纳入多个 measurement families g
  对每个 g，g 到 H1_k 中至少一个 measured node 有一条合格边，且 d(g, B) = 2
  若 g 到任一 D/C measured node 存在一跳捷径，则必须重分到相应 H1，不能进入 H2_k
```

每个 extension measurement family 必须映射到唯一 `parent_c_family_id`，避免同一 evidence row 在不同 mechanism
branches 重复计量。若存在多个等长 parent path，使用预注册、与 performance 无关的 specificity/tie-break 规则；
仍无法唯一判定则标为 `unresolved`。H1 spec 是每个 C family 的必需结构；冻结前若找不到任何可信且可用的 H1
assay family，该 task graph 不能静默通过，也不能制造弱 edge。H2 是可选结构，可显式记为 `unavailable`。

因此 `H1/H2` 表示各 C branch 向外展开的 tree depth，同时仍接受全局 shortest-path shortcut audit；它不是 assay
到 benchmark label 的绝对生化步数，也不意味着 `C -> D` 是一条固定机制边。

累计条件固定为：

```text
none
D
D + C
D + C + H1
D + C + H1 + H2
```

计划使用以下不与现有 paper condition 冲突的 machine IDs：

```text
# flat primary
distance_none
distance_d
distance_dc
distance_dc_h1
distance_dc_h1_h2

# mechanism supplementary
distance_mechanism_dc
distance_mechanism_dc_h1
distance_mechanism_dc_h1_h2
```

累计条件是这棵树的 depth prefixes，而不是四类平行 source 的简单相加。`D+C+H1` 表示加入每个 C family 的唯一
H1 child；`D+C+H1+H2` 只加入实际 available 的 H2 children。

如果新扫描发现的是漏掉的 direct/core assay，而不是更远的 assay，它属于 base-library completeness 问题：先标记
`base_repair_candidate`，不允许偷偷放进 H1/H2。是否重冻 base 必须形成新版本，旧 D/C 和旧结果继续保留。

## 4. 什么算一条推理桥

一条 graph edge 必须同时满足以下条件：

1. 两端是命名明确、可测量的 biological state/process，不是模糊主题词。
2. 两端粒度相近；不能一端是单个 receptor binding，另一端直接写整个 clinical outcome，中间机制全部省略。
3. 生物学关系具有方向：记录 `upstream_node -> downstream_node`。
4. 另行记录允许的 inference direction；默认只允许 upstream 推断 downstream。只有经过验证的 mechanistic
   readout 才能声明 downstream readout 可反推相邻 core state。
5. 记录 relation type、正/负/条件依赖方向、适用物种/组织/route/system 和至少一个可核查来源。
6. 条件依赖关系必须把关键条件写入 edge；若 ChEMBL row 缺少这些条件，相关 assay 不得借该 edge 入选。

文献准入至少满足其一：一份明确陈述该关系的 authoritative guideline/endorsed AOP；或两份相互独立的支持
来源，其中至少一份是直接 mechanistic study。只在 narrative review 中提到、但找不到直接或权威支持的 edge
可以保留为 candidate，不能进入冻结 graph。

允许的 relation type 只包括：

```text
causal_process
  upstream biological change 会驱动 downstream change。

functional_component
  可测功能是 downstream core process 的直接组成或限速决定因素。

exposure_link
  disposition/exposure change 会直接改变 core process 所需的 target-site exposure。

validated_mechanistic_readout
  该 downstream readout 已被验证为相邻 core state 的可预测机制读数，因而可以显式允许反向 inference；
  普通 biomarker association 不够。
```

以下情况明确不算 edge：

- 同属 ADME、tox、skin、CNS 等大领域；
- 同一器官、疾病、pathway、target family 或 assay format；
- assay description 的文本相似；
- molecule structure 相似；
- 只有统计相关、共表达或共现，没有相邻 causal/predictive relation；
- generic target binding，但没有 functional modulation 与 task-relevant process 的连接；
- generic cytotoxicity，但没有指向 task core 的特异机制桥；
- 用最终 benchmark label 或 preliminary performance 反推 edge。

### 4.1 Same-molecule causal continuity gate

graph shortest path 只验证 biological state 之间能否连接；它不能自动保证整条路径仍在预测 **同一个 assay/query
molecule**。每个 extension measurement family 还必须单独追踪 causal subject：

```text
允许：molecule A perturbs process X -> shared physical barrier changes -> A encounters that barrier

断点：molecule A induces transporter abundance
      -> transporter changes disposition of substrate B
      -> 没有 evidence 证明 A == B 或 A 也具有该 substrate role
```

因此，以下命题必须分开，不能互相推出：

- molecule activates a transcription factor；
- transporter abundance/activity changes；
- the same molecule is a substrate of that transporter。

对每个 family 保存 `FamilySelfRelevanceAudit`：

```text
family_id
status: pass_same_molecule | requires_query_role | context_only | unresolved
causal_subject
required_query_roles
rationale
citations
```

执行规则：

1. `pass_same_molecule`：path 没有切换到另一个未观测 molecule；必要的物种、组织、疾病状态、route 和时间尺度
   仍写入 scope/applicability。
2. `requires_query_role`：path 只有在 query 同时是 transporter/enzyme substrate、precursor、sensitizer 或具备其它
   molecule-specific role 时才成立，但当前 row/prefetch 没有该角色证据。
3. `context_only`：assay molecule 只改变 system/environment，当前证据支持的是其它 molecule 或人群的 outcome。
4. `unresolved`：无法确定 causal subject 是否连续。
5. 只有 `pass_same_molecule` 可以直接进入主 H1/H2。`requires_query_role` 只有在 retrieval contract 对同一 molecule
   显式联合提供并验证所需角色后，才能重新审计为 pass；LLM 猜测或 prompt warning 不算证据。

该 gate 与 hop、scope 和 assay quality 正交。一个 family 可以是文献充分的一跳、assay quality 也为 pass，但仍然
因为 causal subject 从 A 换成 B 而不能用于 self-molecule prediction。

定义参考 OECD AOP/KER 思路：Key Event 是可测量的 biological change，Key Event Relationship 是连接上下游
event、支持 causal/predictive inference 的科学关系。E12 借用这种结构化思想，但不宣称四个 task graph 都是
OECD endorsed AOP：

- OECD AOP overview: https://www.oecd.org/en/topics/sub-issues/testing-of-chemicals/adverse-outcome-pathways.html
- OECD AOP Users' Handbook: https://doi.org/10.1787/5jlv1m9d1g32-en
- OECD AOP Knowledge Base: https://aopkb.oecd.org/background.html

## 5. Assay 到 distance level 的执行流程

分层单位不是单条 activity row，也不是单个 assay ID，而是语义一致的 **measurement family**。一个 family 中的
assay 必须测量相同 biological node，并有兼容的 endpoint 语义；不同单位/物种/系统可以作为 row metadata 保留，
但若 biological interpretation 不同就必须拆 family。

执行顺序固定为：

1. 从 assay `description + target + standard_type/measurement + organism + tissue/cell/system + route/context`
   判定它测量哪个 node；只靠 keyword 命中不能完成映射。
2. 先映射到唯一 `measurement_family_id`；无法唯一判定则记为 `unresolved`。
3. 用冻结 graph 的 `admissible_inference_directions` 计算该 node 到 B 的 shortest admissible path；不能默认逆着
   causal edge 行走。
4. 独立计算 `scope_match`：`matched`、`transportable` 或 `out_of_scope`。
5. 独立计算 `quality_status`：`pass` 或具体失败原因。
6. 只有 path 唯一可解释、scope 非 `out_of_scope`、quality 为 `pass` 的 family 才进入主实验。

优先级和冲突处理：

```text
已经属于 D/C          -> 保留当前 base membership，不重新贴 hop 标签
同时有一跳和两跳路径   -> 取到整个 B 的最短合格路径，归对应 C family 的 H1，并保存所有候选路径
声明 H2 但发现一跳捷径 -> validation error，不能发布 manifest
方向或条件不清         -> unresolved，排除
一个 assay 命中多 family -> 拆分 endpoint；无法拆分则 unresolved
多个 C parent 等长      -> 按预注册 specificity/tie-break 唯一归属；仍不能唯一判定则 unresolved
某个 C 没有可信 H1     -> graph freeze failure；不得创建空 LLM branch 或伪造弱 edge
某个 H1 没有可信 H2    -> 该 C family 的 H2 显式 unavailable
```

在没有外部领域专家的当前条件下，第三步采用 citation-backed 两遍独立审计：第一遍建立 mapping；第二遍隐藏第一遍
level，只读取 assay fields、node definitions 和 edge sources，重新计算路径。任何不一致项进入 `unresolved`。
这是一种可重复的 AI-assisted curation，不等同于 external expert validation，论文必须如实说明这个限制。

## 6. 两条代码线及复用边界

### 6.1 现有论文主线：完全冻结

以下接口和语义不改：

```text
tools/chembl_tool/common/experiment_retrieval.py
  EXPERIMENT_MODES = none/direct/full_flat/full_mechanism/native

tools/chembl_tool/tasks/<task>/experiment_config.py
  当前 direct/mechanism source-group mapping

tools/chembl_tool/paper_experiments/molecular_evidence_agent.py
  当前 paper matrix、condition 名称和旧 index 路径
```

不向 `EXPERIMENT_MODES` 塞入 H1/H2，不重命名旧 Tier，不改旧 `direct/full_flat/full_mechanism` 的 retrieval 或
prompt assembly。现有结果继续回答原 RQ；只有 LLM-visible input hash 完全相同的 condition 才允许在 E12 中
复用 prediction。`experiment_retrieval.py` 最多新增一个公开、无状态的 flat-assembly wrapper，内部仍调用当前
实现；现有调用路径和输出 hash 必须不变。

### 6.2 新 distance-expansion 线：独立入口

当前已实现或计划补齐：

```text
tools/chembl_tool/common/evidence_distance.py
  通用 dataclass、shortest-path 计算、schema validation、manifest hash。

tools/chembl_tool/common/distance_retrieval.py
  将 D root、C family nodes、每个 C 的聚合 H1 child 和 optional H2 child 作为 maximal tree view；
  每个 tree node 独立执行同一 top-k/similarity policy，再物化累计 flat primary 和 mechanism supplementary prefixes。

tools/chembl_tool/paper_experiments/materialize_distance_retrieval.py
  对每个 query 只计算一次 maximal tree view，写出标准 batch replay 目录；不调用 LLM。

tools/chembl_tool/tasks/<task>/distance_config.py
  task outcome/core nodes、D/C source groups、H1/H2 nodes/edges/families 和 citations。

tools/chembl_tool/paper_experiments/distance_expansion.py
  待实现：独立 E12 matrix runner；复用现有 task batch/pipeline、visibility、single branch 和 replay contract。

tools/chembl_tool/paper_experiments/audit_distance_expansion.py
  审计 graph、prefix nestedness、base parity、scope/quality、coverage、budget 和 input hashes。

tools/chembl_tool/paper_experiments/summarize_distance_expansion.py
  待实现：只汇总 E12，生成 macro-F1 curve、paired bootstrap、quantity/token tables。
```

四个 task 不复制 runner。它们只提供 `distance_config.py`；后续新增 assay 的筛选规则仍放 task-local adapter，
标准化、molecule-level aggregation、identity policy、retrieval ranking、evidence contract、LLM client、batch resume
和统计全部复用 common 实现。

distance runner 先把每个 prefix 物化为现有 batch 目录形状的 per-query `retrieval.json`，再通过已有
`--retrieval-replay-source-batch` 送入四个 task 的通用 batch/pipeline。task pipeline 不需要认识 H1/H2，也不
需要新增 experiment mode：flat primary 收到一个合并 group，mechanism supplementary 收到 prefix 内稳定的
family groups。

## 7. 配置与机器可读 contract

公共配置的最小结构计划为：

```python
DistanceExpansionConfig(
    task_name=...,
    base_source_config=CHEMBL,  # applies to D, C, H1, and H2
    nodes=(BiologicalNode(...),),
    edges=(MechanismEdge(...),),
    c_family_chains=(DistanceFamilyChainSpec(...),),
    retrieval_budget=DistanceBudget(...),
)
```

其中：

```text
BiologicalNode
  node_id, display_name, definition, measurable_state, core_anchor

MechanismEdge
  upstream_node, downstream_node, relation_type, effect_direction,
  admissible_inference_directions, applicability, rationale, citations

DistanceFamilyChainSpec
  parent_c_family_id, h1_group_id, optional_h2_group_id,
  h1_measurement_families, h2_measurement_families, tie_break_policy

DistanceMeasurementFamilySpec
  measurement_family_id, measured_node, declared_level, parent_c_family_id,
  parent_h1_measurement_nodes, admissible_path, scope_rule, quality_rule

DistanceBudget
  min_similarity, per-tree-node unique-neighbor cap,
  per-neighbor representative-row cap, selection policy
```

validator 至少强制：

- D/C/H1/H2 的 evidence source 均为冻结 source manifest 中的同一 ChEMBL release，拒绝 Starling/其它 source row；
- D/C 互斥且 union 与当前 full source-group union 完全相等；
- 每个冻结 C family 恰有一个 H1 child spec；每个 H1 至多有一个 H2 child spec；
- 每个 extension measurement family 只属于一个 parent C family，source/evidence row 不跨 branch 重复；
- 每个 H1 measurement family 到其 parent C family 的 path length 恰为 1；
- 每个 H2 measurement family 到其 H1 parent 有一条边、到整个 B 的 shortest path 恰为 2；
- H2 若到任何 D/C measured node 存在一跳 shortcut，validation 必须失败；
- edge 两端存在、biological/inference direction、类型、citation、适用范围不为空；
- 每个 extension family 恰有一个 `FamilySelfRelevanceAudit`；发布时全部通过
  `validate_self_relevance_audit(..., require_publishable=True)`；
- `requires_query_role` 必须明确列出缺失角色，且在角色未由同一 molecule evidence 满足前不得进入 retrieval；
- `unresolved`、`out_of_scope`、quality fail 不进入 retrieval config；
- cumulative source-group union 严格 nested；
- 所有配置和 source manifest 产生稳定 hash。

发布 artifact：

```text
distance_graph.json
distance_family_tree.json
distance_measurement_family_manifest.json
assay_to_family.tsv
assay_exclusions.tsv
source_manifest.json
retrieval_prefix_manifest.json
```

## 8. Evidence library 与旧结果隔离

旧 evidence library/index 路径保持不变。E12 使用一个新版本的 superset library：

```text
frozen base evidence rows (D/C)
  + new ChEMBL extension rows (H1/H2)
  -> distance-specific molecule evidence library/index
```

其中 frozen base rows 也必须是当前 ChEMBL D/C，而不是 Starling D/C。Superset builder 不接受跨 source merge；
source manifest 必须同时覆盖 base 与 extension，并由 audit 检查每个 row 的 source、release 和 normalization version。

新 index 内 D/C 的 source-group ID 和 evidence row 必须原样保留。加入 extension 后，同一分子的不同 group 可以
合并在 molecule record 中，但 D/C view 只能读取原 group rows。base parity audit 必须证明：

```text
distance D evidence content       == 当前 direct evidence content
distance D+C evidence content     == 当前 full_flat evidence union
distance mechanism D+C branches   == 当前 full_mechanism family inputs
```

这里的 parity 按最终 LLM-visible `minimal_evidence.v1` 计算；旧 base index 中是否预先缓存
`minimal_evidence` 字段不是语义差异。另需区分两种 nestedness：D 和 D+C 必须 source-group union 累计，但旧
direct 与旧 full-mechanism 的 Tier-1 family 组织不同，D neighbor 可能在 D+C 的固定 top-k 中被 context neighbor
替换，因此只记录 D->D+C replacement diagnostic；D+C->D+C+H1->D+C+H1+H2 必须逐 evidence row 严格保留。

### 6.3 BBB v2 historical prototype（2026-07-20）

BBB v2 historical prototype 曾实现第一个旧式 vertical slice：`evidence_distance.py`、`distance_index.py`、`distance_retrieval.py`、通用
`distance_assay_manifest.py`、BBB `distance_config.py`/`distance_assay_rules.py`、建库 wrapper 和 retrieval audit。
v2 在 family 定义前对全部 D/C assays 生成 measured-state census，并要求 extension measured node 与 base node
集合不相交。census 证明旧 D/C 已含 tight-junction、efflux/influx abundance 和 functional PXR/CAR readouts，
因此 v1 的 tight-junction/PXR-CAR extension 作废。v2 当时映射 H1 MMP-9 activity 和 H2 MMP-3 activity；全库扫描
纳入 696 个 extension assays，ID 与 20,369 个 base assays 的交集为 0，extension index 为 2,747 molecules，
合并后 superset 为 55,260 molecules。392-query retrieval audit 的 base parity、source-group nestedness、
D+C->H1->H2 evidence retention 和 H1-after-H2 branch stability failures 均为 0；累计 coverage 为
D 0.673、D+C 0.878、D+C+H1 0.888、D+C+H1+H2 0.888。该结果只验证检索集合和 quantity，独立 E12
LLM runner 尚未实现，因此不能解释为 macro-F1 增益。

后续 shortcut audit 找到 `MMP-3 activity -> tight-junction integrity` 的直接支持，因此 MMP-3 到 B 的最短路径
为 1，不能继续作为 H2。v2 产物只保留为 historical retrieval prototype，不得送入 E12 LLM。下一版必须按 C-family
tree 重建：MMP-9/MMP-3 可作为 passive/barrier C family 的同一个聚合 H1 node 内部 measurement families；每个
其它 C family 也必须有唯一 H1 child，H2 只在不存在任何到 B 的一跳捷径时建立。旧 v2 coverage 不得被当成新
tree ontology 的 coverage。

### 6.4 BBB v3 C-family tree 实现与 retrieval gate（2026-07-20）

BBB v3 已冻结三个可扩展 C parents 和五个 aggregate tree nodes：passive H1 聚合 MMP-9/MMP-3，efflux
H1/H2 分别为 functional NRF2 state 与 KEAP1-NRF2 PPI，influx H1/H2 分别为 functional HIF-1 state 与
direct PHD2 hydroxylase。MMP-3 已按 shortcut audit 归回 passive H1；passive H2 显式 unavailable。

ChEMBL 36 全库 1,890,749 assays 扫描后，v3 纳入 1,514 个 extension assays，和 20,369 个 frozen D/C assays
的 ID 交集为 0。分支 assay 数为 passive H1 696、efflux H1 389、efflux H2 121、influx H1 146、influx H2
162。extension library 含 34,838 activity rows、21,127 indexed molecules；直接合并已冻结 indices 后 superset
为 73,175 molecules、24 source groups。merge 不重新标准化 base molecules，因而避免无谓计算并保留原 D/C rows。

392-query retrieval audit 的四类严格 failure 均为 0：D/D+C base parity、source-group nestedness、
D+C->H1->H2 evidence retention、H1-after-H2 stability。累计 coverage 为 D 0.673、D+C 0.878、
D+C+H1 0.923、D+C+H1+H2 0.923；mean unique neighbors 分别为 1.39、5.51、8.28、8.45。tree-node coverage
为 passive H1 0.204、efflux H1 0.666、influx H1 0.528、efflux H2 0.043、influx H2 0.071。H2 没有提高
总体 coverage，只为少数已覆盖 query 增加 evidence，因此正式 LLM run 必须把 H2 描述为 low-coverage distant
evidence condition，不能预设会改善 macro-F1。

这些结果只完成 ontology、data 和 retrieval gates；尚未运行 E12 LLM macro-F1。下一步先物化 flat prefixes，
再按 D+C+H1 -> D+C+H1+H2 顺序运行 mechanism supplementary，并复用冻结 D/C/H1 branch outputs。

`materialize_distance_retrieval.py` 已把 7 个非 none 条件各物化为 392 个标准 batch-style `retrieval.json`；
manifest 冻结 config hash、index、input split 和 retrieval policy。后续 task reasoning pipeline 只通过
`--retrieval-replay-source-batch` 消费这些 artifacts，因此不需要把 H1/H2 mode 塞进旧 `EXPERIMENT_MODES`，也不会
改变既有 paper matrix。

若 group ID/serialization 等 prompt metadata 为实现累计 flat view 而发生变化，则 evidence parity 仍需通过，但
旧 prediction 不得按“看起来一样”复用；必须由现有 input-hash 机制决定是否重跑。

因此现有结果的使用边界是：旧 none/direct/full-flat/full-mechanism 继续完整用于原 RQ；E12 的 none、D 或
D+C 只有在 model、split、visibility、retrieval、serialization 和 LLM-visible hash 全部一致时才能复用。
这保证“保留旧结果”不等于“强行把旧输出塞进新曲线”。

## 9. Retrieval、parallel reasoning 与资源控制

Flat primary 每个累计条件只有一个合并 group branch，用它回答 distance/quantity 主问题。Mechanism
supplementary 按 family 并行 reasoning，用来检查 evidence distance 与 reasoning organization 的交互；它不替代
flat 主曲线。两套视图来自同一次 family retrieval：

1. 每个 query 对 superset index 只计算一次 fingerprint similarity。
2. 对 D root、每个 C family node、每个 C 的唯一 H1 child 和 optional H2 child 生成完整、确定性的候选 ranking。
3. 依次物化 D、D+C、D+C+H1、D+C+H1+H2 flat prefixes。
4. 另行物化 D+C、D+C+H1、D+C+H1+H2 mechanism prefixes；tree-node group ID 和输入在后续 prefix 中保持不变。
5. 每个 tree node 独立最多保留 3 个 unique molecular neighbors；node 内不同 target/measurement families 共享这
   3 个位置，不得按 target 各取 3 个。
6. molecule 跨 tree node 重复时，flat union 只计一个 unique molecule，但各 node 保留自己的 source groups 与
   evidence rows；同一 assay/evidence row 不得因多 parent path 被复制到多个 H1/H2 branches。
7. 所有 tree nodes/prefixes 使用相同 `min_similarity` 和 identity policy；不能为了补满 quota 降阈值。

Mechanism supplementary 固定按以下顺序执行：

```text
distance_mechanism_dc
  当前 D+C full_mechanism baseline；hash 相同时复用已有 branches。

distance_mechanism_dc_h1
  --group-analysis-source-batch = distance_mechanism_dc
  复用 D/C；只运行 H1；重跑 final。

distance_mechanism_dc_h1_h2
  --group-analysis-source-batch = distance_mechanism_dc_h1
  复用 D/C/H1；只运行 H2；重跑 final。
```

加入 H2 后，任何 H1 branch 的 group ID、neighbors、evidence rows、serialization 或 LLM-visible hash 发生变化，
该 branch 都不能复用。无论 group branches 是否全部复用，只要 prefix 扩展，final 输入集合就改变，因此 final
必须重新运行。`distance_mechanism_dc_h1_h2` 只加入 available H2 children；若所有 C-family H2 均 unavailable，
task 停在 `distance_mechanism_dc_h1`。

为了让 quantity 可以增长、又不让成本失控，预算按 **tree node** 冻结：

```text
D/C:
  保持当前 top-k=3/family 的 base semantics，确保与旧 full union 可比。

H1/H2:
  每个 C family 的聚合 H1 node 最多 3 个 unique neighbors；其 optional H2 node也最多 3 个。
  node 内所有 measurement families 先形成同一个 molecule-level candidate pool，再按 similarity 和稳定 molecule ID
  排序；不得给每个 target 单独 quota。所有节点使用与 C 相同的 similarity threshold，不读取 macro-F1 调参。
```

每个 neighbor 的大量同类 measurements 用确定性 representative-row policy 压缩，只能聚合 endpoint、unit、species、
route 和 assay system 兼容的记录；保留 `n/median/range` 与 provenance samples。压缩规则对 H1/H2 一致，并报告原始
与可见 row 数。不要依赖当前 750 KB transport emergency truncation 作为日常资源策略。

工具与 LLM 成本控制：

- query `molecule_properties` 只计算一次，并复用 frozen single prior；
- pair comparison 以 query/neighbor standardized structure + tool version 为 cache key，一次计算、跨 prefix 复用；
- 完整 flat 曲线与 H1/H2 mechanism supplementary 先跑 `deployment_visible_prefetched`；
- flat agentic 只跑 none、D、D+C 和最大可辩护层；mechanism agentic 必跑 D+C+H1，并在至少一个 H2 child
  available 时再跑 D+C+H1+H2，按前述顺序复用 branches；
- 先做四任务 retrieval-only audit，再用少量 valid samples 检查 payload/token/runtime；只按资源 envelope 调 cap，
  不按 performance 调 ontology；
- 如果某 C-family H2 coverage 极低或 graph 不可信，将该 child 记为 unavailable，而不是扩大到泛化 background assay。

## 10. 输出目录与回归 gate

E12 产物与当前 21/29-condition matrix 分开：

```text
outputs/paper/molecular_evidence_agent/distance_expansion/
outputs/paper/molecular_evidence_agent_valid/distance_expansion/
```

实现完成前至少增加以下测试：

```text
test_evidence_distance.py
  path length、shortcut、cycle、missing citation、ambiguous mapping、H2 unavailable。

test_distance_retrieval.py
  cumulative nestedness、proximal retention、dedup、budget、threshold、deterministic hash、flat/mechanism parity。

test_distance_base_parity.py
  D/direct 与 D+C/full union 的 source groups、neighbors、evidence rows parity。

test_distance_experiment_matrix.py
  flat 与 mechanism condition IDs、split/root 隔离、single/retrieval/group replay wiring。

test_distance_audit.py
  graph/source/config hash、coverage、prompt volume、H1 prefix-invariance、branch reuse 和 final-rerun 完整性。
```

旧路径回归必须同时满足：

- `EXPERIMENT_MODES` 不变；
- 现有 paper condition 名称、index path 和命令不变；
- 现有 direct/full retrieval fixtures 的 JSON 与 LLM-visible hash 不变；
- `tests/chembl_tool/common/` 全部通过；
- 新代码不向 LLM 暴露 `evidence_direction`、label vote、threshold policy 或 deterministic override。

## 11. 第三步开始前的完成 gate

只有以下事项完成后，才开始回答“四个 task 具体缺哪些 one-hop/two-hop assay”：

1. 本文定义得到确认；
2. common schema 与 validator 可运行；
3. 四个 `distance_config.py` 可以先只声明 D/C，并通过 base parity；
4. retrieval-only runner 能从同一 query ranking 物化 nested flat/mechanism prefixes，并证明同 prefix evidence parity；
5. 资源 cap 的冻结方法已经实现且不读取模型性能。

随后第三步将按 task 输出：candidate node、graph path、支持文献、ChEMBL query/mapping rule、纳入/排除例子、
预计 assay/molecule coverage 和 H1/H2 是否可用；不会直接凭现有 Tier 名称批量归类。
