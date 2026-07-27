# ChEMBL relevance-dilution / evidence-volume experiment plan

日期：2026-07-23
状态：下一阶段执行计划；替代 strict H1/H2 四级主曲线，但不删除 strict-hop census 或历史 prototype

## 1. 决策与研究问题

四任务 strict-hop census 的结论是：

```text
engineering-feasible strict H1: 3 / 4 tasks
publishable strict H2:          0 / 4 tasks
```

因此不再把 `D+C+H1+H2` 作为跨任务主实验。严格 H1 继续作为 task-specific secondary analysis；
主实验改为 **controlled relevance dilution**：

> 当 evidence measurement quality 和数据源保持不变，逐步加入与当前 task mechanism envelope
> 关系更弱、但数量更多的 ChEMBL assay evidence 时，LLM 的分子任务性能在什么位置停止改善或开始下降？

这里必须区分三个概念：

```text
measurement quality
  assay/measurement/provenance 本身是否可靠；所有主实验 band 使用同一 quality gate

task relevance
  assay measured state 到当前 D/C envelope 的 biological-context distance；这是主实验操纵轴

evidence volume
  实际检索到的 neighbors、assays、rows 和 prompt tokens；随 prefix 增长并完整记录
```

论文中不把 relevance 下降写成 assay 数据质量下降。建议名称为
`evidence relevance-volume expansion` 或 `controlled relevance dilution`。

## 2. 两条互不混淆的分析线

### 2.1 Primary：contextual relevance bands

主实验允许 context-only evidence，因为问题正是检验 LLM 能否从较远上下文获益；但它不允许把这些
evidence 表述成对 query molecule label 的直接因果证据。

```text
D
D+C
D+C+R1
D+C+R1+R2
D+C+R1+R2+R3
```

### 2.2 Secondary：strict causal H1

只在 strict census 已通过的 task/family 上比较：

```text
D+C
D+C+strict-H1
```

当前候选：

- BBB_Martins：passive/barrier strict H1；
- Skin_Reaction：GSK3B graph/data pass，但必须保留 sensitizer-scope caveat；
- Bioavailability_Ma：pKa、LogD/LogP、PPB/Fu；必须报告 property-tool semantic overlap；
- ClinTox：无 publishable strict H1。

不再寻找或运行 strict H2。Secondary 不能被画成四任务统一 causal-distance curve。

## 3. R1/R2/R3 的机器可执行定义

### 3.1 共同前提

一个 assay family 进入任何 R band 前必须：

1. 来自冻结的 ChEMBL 36；
2. assay ID 不属于冻结 D/C；
3. 通过与 D/C 相同的 measurement-quality、validity、duplicate 和 provenance gate；
4. measured node、target/property、scope、mapping reason 和 provenance 可审计；
5. 唯一归属一个 band，不能跨 band 重复 activity row；
6. 不使用 benchmark label 或 valid/test performance 决定归属；
7. 保留 `causal_status`，但 band membership 不以 causal status 作为充分条件。

### 3.2 Context graph

新增一张只用于 **relevance connectivity** 的 context graph。它与 strict causal graph 分开：

- node 是命名明确的 measurable biological state/process/property；
- edge 必须有可核查的 pathway database relation 或文献支持；
- edge 保留方向和 relation type，但 relevance distance 只表示连接远近，不授权 causal inference；
- 共同器官、疾病词、assay format 或文本相似度不能单独构成 graph edge；
- task-domain membership 由预注册 ontology/rule 定义，不由 embedding 相似度事后决定。

### 3.3 R1：proximal contextual evidence

```text
shortest context-graph distance to any frozen C measured node = 1
```

- 可以是 `pass_same_molecule`、`requires_query_role` 或 `context_only`；
- strict causal H1 是其中 `pass_same_molecule` 的可发布子集；
- role/scope 断点必须记录，不能在 prompt 中暗示其已被解决。

### 3.4 R2：distal pathway context

```text
shortest context-graph distance to frozen C measured nodes = 2 or 3
and no R1 shortcut
```

- 必须能给出完整 node path；
- 允许 pathway-connected context-only evidence；
- 不声称 assay molecule 的作用沿路径传递为它自己的 task label。

### 3.5 R3：task-domain background

```text
passes a pre-registered task-domain ontology/rule
and has no verified context-graph path of length <= 3 to frozen C
```

- 不是跨任务 source，也不是 endpoint-shuffled 或随机噪声；
- 必须排除 direct label endpoint、D/C synonym 和已知 R1/R2 shortcut；
- 用来表示“仍在相同 ADME/tox/skin/BBB 广义领域，但机制关联明显更弱”的 assay pool。

若某 task 的某一 band 不满足发布 gate，则显式记为 unavailable；不能为了完整曲线放宽规则。

## 4. 每条 evidence 的审计字段

新的 manifest 至少保存：

```text
relevance_band
measured_node
parent_c_family_ids
context_graph_path
context_graph_distance
causal_status
required_query_roles
scope_status
quality_status
mapping_reason
citations
assay_id
document_id
source_release
```

其中 `relevance_band`、内部 relevance score、rule-derived vote 和 mapping policy 只用于 audit，
不得发送给 reasoning LLM。LLM 只看到经 `minimal_evidence.v1` 标准化的真实 assay/measurement/context、
scope/uncertainty 和 provenance；不能在 group title 中告诉模型某条 evidence 是 “distant” 或 “low relevance”。

## 5. Candidate 与 band 发布 gate

每个 task/band 在不看 label/performance 的前提下通过以下 gate：

```text
unique molecular parents                 >= 50
independent documents                    >= 3
largest-document parent fraction         <= 0.80
parent-disjoint valid-query coverage      >= 0.10
incremental valid-query coverage          >= 0.10
D/C assay overlap                         = 0
cross-band activity-row overlap           = 0
quality failures in published rows        = 0
source release                            = ChEMBL 36 only
```

`incremental coverage` 指相对前一个 prefix，至少 10% valid queries 获得一个此前未出现的新 molecular-parent
neighbor。数量很多但没有新增 query coverage 的 band 仍可保留在 census，但不进入 LLM 主曲线。

任务曲线在最后一个通过 gate 的 band 停止。若少于两个 task 具有 R2 或 R3，论文将该部分降为
task-specific case study，不声称跨任务通用 boundary。

## 6. Retrieval budget 与 reasoning 成本控制

### 6.1 Retrieval

D/C 保持现有 ChEMBL retrieval input 不变。每个新增 R band 是一个全 task 聚合 retrieval budget：

```text
min similarity:                 0.30
primary identity policy:        parent_disjoint
operational staging/sensitivity: required
unique molecular parents:       max 3 per band per query
representative rows:            max 8 per retrieved parent per band
```

因此每个 query 最多新增：

```text
R1: 3 parents / 24 rows
R2: 3 parents / 24 rows
R3: 3 parents / 24 rows
total: 9 parents / 72 rows
```

row selection 必须确定性完成：优先覆盖不同 assay、measurement type 和 document，再按质量与稳定 ID
tie-break。不能在运行中因 token 不足临时改变规则。

每个 band 的三个位置由该 band 的全部 families 共享，不按 target 或 C family 各取 top-3。row 内仍保留
parent C family/path metadata，避免某个 task 因 C family 多而产生无上限分支和 prompt。

### 6.2 Flat primary

每个 prefix 使用一个 flat evidence branch；新增 prefix 需要重跑 flat branch 和 final。
single-molecule prior、tool replay、model parameters 和 serialization 全部冻结复用。

### 6.3 Supplementary grouped reasoning

现有 D/C mechanism branches 在 input hash 不变时复用。每个新增 R band 只增加一个聚合 parallel branch：

```text
D/C branches + R1 branch -> final
D/C branches + reused R1 + R2 branch -> final
D/C branches + reused R1/R2 + R3 branch -> final
```

这样新增 parallel branches 最多为 3，不随 assay family 数增长。每次只运行新 band branch并重跑 final；
旧 branch 的 group ID、evidence rows、serialization 和 hash 不变。

### 6.4 Token gate

建库后先在每个 task/band 运行 5-query smoke，冻结正式 row cap。进入 full run 前必须满足：

- 没有 request 触发 750 KB transport guard；
- P95 input token 不超过同 task `D+C` 的 1.5 倍目标值；
- structured-output failure/retry 没有相对 D+C 明显恶化；
- flat 与 grouped view 的 evidence-row multiset 一致。

若超限，只能在 freeze 前统一降低所有 task/band 的 row cap并重新物化全部 retrieval；freeze 后不得按
condition 单独削减。

## 7. 代码结构

严格 H1/H2 prototype 保留用于历史与 secondary，不在原模块内改写语义。新增独立代码线：

```text
tools/chembl_tool/common/evidence_relevance.py
  RelevanceConfig、ContextNode/Edge、RelevanceFamilySpec、band validator 和 stable hash。

tools/chembl_tool/common/relevance_retrieval.py
  R-band aggregate top-3、cumulative prefixes、flat/grouped views 和 branch-stability assembly。

tools/chembl_tool/common/task_workflows/relevance_assay_manifest.py
  通用 ChEMBL scan/export；复用 distance_assay_manifest 的 DB、normalization 和 provenance primitives。

tools/chembl_tool/tasks/<task>/relevance_config.py
tools/chembl_tool/tasks/<task>/relevance_assay_rules.py
  只放 task ontology、candidate mapping、scope/quality rule 和 citations。

tools/chembl_tool/paper_experiments/build_relevance_index.py
tools/chembl_tool/paper_experiments/materialize_relevance_retrieval.py
tools/chembl_tool/paper_experiments/audit_relevance_dilution.py
tools/chembl_tool/paper_experiments/relevance_dilution.py
tools/chembl_tool/paper_experiments/summarize_relevance_dilution.py
  独立建库、replay、audit、LLM runner、统计和绘图入口。
```

必须复用：

- `minimal_evidence.v1`；
- molecule normalization、parent-disjoint/operational policy；
- `experiment_retrieval.py` 的 group/flat assembly；
- replay、identity-blind、input hash 和 branch reuse；
- task batch/pipeline、OpenAI-compatible client、validation；
- paired bootstrap、McNemar/Holm 和 paper figure style。

明确禁止：

- 向旧 `EXPERIMENT_MODES` 注册 R1/R2/R3；
- 改写旧 D/C index、condition name 或已有结果；
- 把 Starling row 合并进 relevance curve；
- 在 task wrapper 复制 common retrieval/LLM runtime；
- 用 test performance 修改 graph、band 或 assay mapping。

新 index/output：

```text
outputs/chembl_tool/tasks/<task>/relevance_dilution/
  manifests/v1/
  evidence_library/v1/
  retrieval_replay/v1/
  analysis/v1/

outputs/paper/relevance_dilution/
outputs/paper/relevance_dilution_valid/
```

## 8. 执行顺序

### Phase 0：冻结 contract

1. 写 `evidence_relevance.py` schema 和 validator；
2. 明确 R1/R2/R3、hidden audit fields、global top-3 budget；
3. 将 strict-hop design 标为 secondary/historical；
4. 为 schema、shortest path、shortcut、overlap、prefix 和 hash 写单元测试。

完成 gate：纯 synthetic config 可以稳定通过/失败；旧 paper tests 全部不变。

### Phase 1：四任务 retrieval-only census

1. 为每个 task 冻结 D/C measured-node envelope；
2. 建 candidate target/property/pathway list；
3. 生成 R1/R2/R3 assay manifest 和 exclusion audit；
4. 计算 parents/documents、operational/parent-disjoint coverage 和 incremental coverage；
5. 不调用 LLM，形成四任务 go/no-go table。

建议顺序：BBB vertical slice -> Skin -> Bioavailability -> ClinTox。BBB 用来验证工程路径；其它任务用于验证
band 定义是否具有跨任务可执行性。

### Phase 2：BBB vertical slice

1. 建 BBB relevance superset index；
2. 物化 operational，再物化 parent-disjoint；
3. 审计 D/D+C parity、band disjointness、prefix nestedness、top-3 budget、flat/group parity；
4. 对 valid 的 5 queries 做 flat/grouped smoke；
5. 检查 trace 中 R-band audit label 未泄漏给 LLM。

完成 gate：retrieval audit 0 critical failure；旧 BBB paper inputs/hash 不变。

### Phase 3：扩展到其它合格 task

只为通过 Phase 1 gate 的 task/band建库。每个 task 复用同一 runner，不能复制实现。

完成 gate：所有正式 prefix 都有 source manifest、config hash、coverage/quantity table 和 reproducible replay。

### Phase 4：受控 LLM 主实验

Primary：

- parent-disjoint；
- deployment-visible matched-prefetch；
- flat cumulative prefixes；
- frozen single prior；
- primary metric 为 test macro-F1。

Supplementary：

- 同一 prefix 的 grouped cumulative reasoning；
- D/C 与早期 R branches 按 input hash 复用；
- operational 作为 sensitivity，不替代 parent-disjoint analog claim。

执行顺序固定：

```text
valid 5-query smoke
valid full integrity run
test parent-disjoint matched-prefetch flat
test parent-disjoint grouped supplementary
operational sensitivity
```

不得根据 valid/test performance 删除某个已冻结 band。

### Phase 5：Agentic confirmation

不跑完整 agentic curve。每个合格 task只确认：

```text
D+C
maximal defensible relevance prefix
```

若现有 `D+C` LLM-visible hash 完全一致则直接复用。新增 maximal point 同轮运行 operational 与
parent-disjoint，并报告 same-parent exposure。

### Phase 6：统计与论文产物

每个 task 分别报告：

- prefix macro-F1 和 95% paired bootstrap interval；
- 相邻 prefix 及相对 D+C 的 paired macro-F1 delta；
- McNemar exact test，预注册 comparison family 内 Holm correction；
- overall/class-conditional coverage；
- actual neighbors、unique parents、assays、documents、rows 和 tokens；
- LLM calls、retry/failure 和 branch-reuse counts。

不增加 rescue/harm rate；不把 coverage 与 performance 的相关性解释为因果效应；不拟合跨任务统一
distance threshold。若曲线方向不同，结论就是 task-specific relevance boundary。

主图：

```text
x: D, D+C, +R1, +R2, +R3
y: macro-F1
annotation: retrieved parents / rows / prompt tokens
```

Secondary 图单独显示 strict `D+C` vs `D+C+H1`，不能与 contextual R bands 合并为同一 causal x-axis。

## 9. 预计运行量与资源策略

当前四个 test split 共约 888 queries。若全部 task 有三个新增 band：

```text
flat:     3 prefixes * (branch + final) * 888  ~= 5,328 calls
grouped:  3 new-band branches + 3 finals       ~= 5,328 calls
upper bound before reuse/unavailable bands     ~= 10,656 calls
```

真实调用数会因 unavailable bands、existing D/C reuse 和 branch reuse 降低。Agentic 只跑 base/maximal，
不把完整 curve 重复一遍。

正式 token 预算不从历史平均值猜测；Phase 2/3 的每 task 5-query smoke 必须输出：

```text
tokens/query by prefix
calls/query
P50/P95 prompt bytes
retry rate
projected full-run tokens and wall time
```

只有用户确认 projection 后才启动 full LLM run。Retrieval census、manifest、index 和 replay 可以先完整完成，
因为它们不产生昂贵 API reasoning 成本。

## 10. Stop rules

以下任一情况触发 stop，而不是继续扩张：

1. band 不满足 coverage/data gate；
2. 无法排除 D/C 或跨-band evidence overlap；
3. context path 只能由共同关键词或文本相似度支持；
4. R3 domain rule 过宽到无法稳定复现；
5. prompt 泄漏 relevance label，或 flat/group evidence multiset 不一致；
6. P95 payload 超过冻结预算且统一 row cap 仍无法解决；
7. 少于两个 task 有可运行的 distant band。

第 7 项发生时，保留最强的 task-specific case study，不再把该实验作为跨任务主 claim。

## 11. 近期交付顺序

```text
1. relevance schema + tests
2. four-task task-domain/context-node specification
3. four-task R1/R2/R3 retrieval-only census
4. go/no-go review
5. BBB index/replay vertical slice
6. full eligible-task indices and retrieval audits
7. 5-query cost projection
8. user approval for full LLM run
9. matched-prefetch curves
10. key agentic confirmation, statistics, figure and paper text
```

在第 4 步前不实现昂贵 LLM runner；在第 8 步前不启动 full LLM batch。
