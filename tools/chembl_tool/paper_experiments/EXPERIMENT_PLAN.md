# 实验计划

> 状态说明（2026-08-27）：本文冻结的是第一轮 group-level 矩阵，不再充当最新 benchmark/assay-level
> 运行入口。当前 paper-facing lineage 与结果见 `STARLING_BENCHMARK_RESULTS.md`；conditioned progressive
> 协议、source-family purity、最终 valid 曲线和重画命令见 `ASSAY_LEVEL_RETRIEVAL.md`。

本文档记录当前第一轮冻结矩阵的研究问题和运行协议。面向 ICLR 2027 的后续数据补齐、same-parent
retrieval 消融、第二模型、重复运行、source-quality annotation、baseline 和投稿时间表见
[ICLR_2027_EXECUTION_PLAN.md](ICLR_2027_EXECUTION_PLAN.md)。Visibility 升降原因的集中分析见
[VISIBILITY_ANALYSIS.md](VISIBILITY_ANALYSIS.md)。

## 2026-08-01 v4 协议覆盖

For `record_agreement70_split811_v1` and later datasets, the default remains an
`identity_blind + parent_disjoint` fresh run; operational staging is neither run first nor required. The launcher
loads the PARCC LiteLLM URL and credential from the ignored root `keys.py`, requests
`nvidia/GLM-5.2-NVFP4`, and retains the historical empty `reasoning_effort`. The endpoint-wide hard limit remains
512. Because reasoning-enabled long prompts were unstable at 384/512, one global prompt pool with
`parallelism=128` remains the default. Later deployment-visible/operational sections describe only historical
lineage or explicit ablations and do not override this section.

第一轮先在 valid 运行完整条件和 audit，冻结设置后再运行 test。Runner 已支持 blind+parent-disjoint
fresh-run、独立 root、无需 reuse plan 的完整 `none`+retrieval 矩阵，以及跨 task/condition 的单一 ready
queue；多个 launcher 仍不得各自占用 512。

## 研究问题

### RQ1：相似分子检索能否提高分子性质分类性能？

V4 主分析在 identity-blind setting 中比较 `none` 与 `direct`，retrieval 固定 parent-disjoint：排除同一
RDKit molecular parent 后用后续真正 analog 补齐 top-k。Operational 只保留为历史或显式 deployment
sensitivity，不属于默认因果链。

### RQ2：Starling 原文抽取 evidence 与 ChEMBL curated structured assay records 有何差异？

- BBB：比较 ChEMBL direct 与 Starling direct。
- Bioavailability：比较 ChEMBL direct 与 Starling direct full。
- Bioavailability mechanism：比较 ChEMBL full mechanism 与 Starling full mechanism。

ChEMBL 本身也包含从科学文献人工抽取、整理和标准化的 bioactivity data，因此这里不是“literature vs
non-literature”，而是保留 source text/context 的 literature-extracted records 与 curated structured
records 的比较。Primary
source comparison 必须匹配 task、query、endpoint scope 和 retrieval view；性能必须与 coverage、evidence
volume、各 group availability 和独立 source-quality annotation 一起报告。只有跨任务方向稳定时才使用
“Starling generally better”这一结论，否则报告 task/family-specific advantage。

### RQ3：按任务机制拆分证据能否进一步提高 reasoning 性能？

每个任务在同一数据源内比较 `full_flat` 与 `full_mechanism`。两者包含相同的 evidence row，只改变分组方式和并行 mechanism reasoning。Bioavailability 同时提供 ChEMBL 和 Starling 的该项消融。

### RQ4：非数值文献证据是否带来额外价值？

对 Bioavailability direct Starling evidence 比较：

- 只使用数值 direct-F 的 scalar KNN
- 只接收数值 direct-F evidence 的 GLM
- 接收数值、非数值 evidence 和实验条件的 GLM

Starling content 只分 `numeric_only` 和 `full` 两种，不在主实验中引入更细的分类。

### RQ5：在冻结的 identity-blind contract 下，retrieval evidence 如何改变判断？

V4 主实验使用 `identity_blind`：LLM 不看 query/neighbor 的结构和身份，harness 提供脱敏的 properties/
comparison 工具文本。论文结论限定在该 contract。`deployment_visible` 和
`deployment_visible_prefetched` 只作为后续显式可见性/部署消融，不能反向决定 blind 主矩阵设置。

### RQ7：Retrieval coverage 与 macro-F1 增幅有什么关系？

在 identity-blind parent-disjoint 主制度中，以每个 retrieval condition 为观察单位，报告 overall、正类和
负类 coverage，并计算相对同任务 `none` 的 paired macro-F1 差值。主性能量使用 macro-F1，以避免标签
不平衡使 accuracy 掩盖少数类行为；class-conditional coverage 是解释 retrieval availability 是否偏向
某一类别的诊断量。

该分析是描述性关联，不将 coverage 解释为独立操纵的 evidence quality，也不做额外的
`none/hybrid/retrieval` counterfactual。不同 task、source 和 retrieval view 分层保留，最终结果需同时
报告样本分母与 paired bootstrap 区间。

### RQ8：超出当前 curated mechanism envelope 多远后，增加更多 assay evidence 不再帮助 LLM？

> 2026-07-23 update：四任务 strict-hop census 得到 `0/4` publishable strict H2，因此下文
> `D/C/H1/H2` 四级主曲线已判定 no-go，只保留为 strict causal secondary/historical design。
> RQ8 主实验改为 `RELEVANCE_DILUTION_EXPERIMENT_PLAN.md` 定义的
> `D -> D+C -> D+C+R1 -> D+C+R1+R2 -> D+C+R1+R2+R3`
> controlled relevance-dilution curve。不得按下文旧 contract 启动 strict H2 LLM run。

这一实验从 base 到 extension 全程只使用 ChEMBL：`D/C` 来自当前 ChEMBL direct/full evidence，`H1/H2`
来自同一冻结 ChEMBL release 的扩展 assay。不得为任何一层运行 Starling extraction，也不得构造
`Starling D/C + ChEMBL H1/H2` 或其它跨 source 混合累计曲线。四层使用相同的 ChEMBL release、标准化流程、
measurement-quality gate、molecule aggregation、provenance contract 和 source manifest。它与 RQ2 正交：
RQ2 在匹配 scope 下比较 source；RQ8 固定 source 为 ChEMBL，研究 mechanistic distance 与 evidence quantity
的联合变化。

Distance 由每个 task 预先冻结的 mechanism-family graph 定义，而不是 assay description 的文本相似度。
`D` 是层级树的 root，当前 paper-facing C mechanism families 是第一圈子节点；H1/H2 是每条 C branch 向外展开的
tree depth，不是与 C 平行的全局 source groups：

```text
D root
  C.family_k
    H1.family_k: 该 C family 有且只有一个聚合 child；内部可含多个一跳 measurement families
      H2.family_k: zero or one 聚合 child；内部可含多个到 H1 一跳的 measurement families
```

`D` 与 `C` 必须互斥，且 `D+C` 必须逐 source group 精确等于当前 ChEMBL full evidence union。C 内已有
families 不要求相对 D 具有统一 hop 数；E12 不重新分层它们。每个 C family 必须声明唯一 H1 child spec，H1
至多声明一个 H2 child；一个 child node 可以聚合多个 targets/measurement families，但每条 assay 保留具体 path。
extension measurement family 必须唯一归属一个 parent C family，不能跨 branches 重复 evidence。H1 measured node
到 parent C 的最短合格路径必须为 1；H2 到其 H1 parent 有一条边、到整个 `B=D+C` 的最短路径必须为 2，发现
任何到 D/C 的一跳捷径即 validation failure。在声明 extension 前必须对所有 D/C assays 生成 measured-state
census；新增 assay ID 和 measured node 都必须与 base 不相交。
主累计曲线为 `none`、`D`、`D+C`、`D+C+H1`、`D+C+H1+H2`。

可扩展 `C mechanism families` 不包括 direct-exposure organization：direct family 属于 D root。为保持旧
`D+C` evidence union，direct tier 内 endpoint normalization 产生的 `context_dependent` 弱证据仍原样保留，但不
为它制造 H1。BBB 显式冻结可扩展 parents 为 passive/barrier、efflux 和 influx 三个 families。

一条合格推理桥必须连接两个命名明确、可测量、粒度相近的 biological state/process，并记录有方向的
biological relation、允许的 inference direction、适用条件和支持文献。默认只允许顺 causal direction 推断；
只有经过验证的 mechanistic readout 才允许反向 inference。共同器官/疾病/pathway/target class、文本相似度、统计相关、
分子结构相似或相同 assay format 都不构成 hop。物种、组织和实验系统差异单独记录为 `scope_match`，assay
可靠性单独记录为 `quality_status`；它们不计入 hop。映射或最短路径不确定的 assay 进入 `unresolved` 并从
主曲线排除。graph、edge rationale 和 assay mapping 在查看 test performance 前冻结。每个 C family 冻结前必须
找到可信 H1，否则 graph freeze 失败；某个 H1 没有可信 H2 时，只把该 family 的 H2 记为 unavailable，不为形成
完整层级强行纳入弱相关或 cross-task assay。

完整受控曲线使用 `deployment_visible_prefetched` matched-prefetch、固定 flat assembly、相同 retrieval
policy 和 frozen single prior。Flat 主曲线只在 `none`、`D`、`D+C` 与最大可辩护层级做 deployment-visible agentic
confirmation。主要指标为 macro-F1；coverage、unique molecule/assay/family counts、evidence rows 和 tokens
只作 quantity 解释。不增加 fixed-token replacement、rescue/harm rate 或 cross-task source 条件。

E12 同时要求 mechanism supplementary，且不能只跑 maximal point：

```text
D+C mechanism
D+C+H1 mechanism
D+C+H1+H2 mechanism
```

先完成 `D+C+H1`：若 D/C family 的 LLM-visible input hash 与当前 `D+C` mechanism condition 相同，则复用
D/C group outputs，只运行每个 C family 的聚合 H1 branch，并重跑 final。随后运行 `D+C+H1+H2`：复用 D/C/H1 branches，
只运行新增 H2 branches，并再次重跑 final。加入 H2 后，H1 的 group ID、neighbor selection、evidence rows、
serialization 和 branch hash 必须保持不变；否则 H1 branch 必须重跑且复用审计不能通过。

Retrieval budget 按 tree node 计算：每个 C、H1 或 H2 node 最多 3 个 unique molecular neighbors，并使用相同
similarity threshold/identity policy。聚合 child 内的不同 target/measurement families 共享这 3 个位置，不能按
target 各取 3 个。

完整 matched-prefetch supplementary 在 H1、H2 两点都运行；deployment-visible agentic 也确认这两个 mechanism
points。每个 prefix 的 flat 与 mechanism 必须使用相同 evidence-row multiset，只允许 grouping 和并行 reasoning
不同。`D+C+H1+H2` 只加入 available H2 children；若所有 C-family H2 均 unavailable，task 只运行到
`D+C+H1`，并逐 family 记录 H2 unavailable。

RQ8 的 ChEMBL-only 结果不能用于证明 Starling 优于 ChEMBL；该 source claim 只能来自 RQ2 的 matched-scope
比较和独立 source-quality annotation。E12 审计必须确认每个 retrieval prefix 的全部 evidence rows 都指向冻结的
ChEMBL source manifest，出现 Starling 或其它 source row 时该 condition 不进入曲线。

可执行的 distance ontology、配置 schema、独立代码路径、资源上限与回归验收标准见
`DISTANCE_EXPANSION_DESIGN.md`。

## 冻结实验矩阵

| Task | None | ChEMBL direct | ChEMBL flat | ChEMBL mechanism | Starling direct | Starling flat | Starling mechanism | Numeric Starling | Scalar KNN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BBB_Martins | yes | yes | yes | yes | yes | yes | yes | no | no |
| Skin_Reaction | yes | yes | yes | yes | yes | yes | yes | no | no |
| ClinTox | yes | yes | yes | yes | no | no | no | no | no |
| Bioavailability_Ma | yes | yes | yes | yes | yes | yes | yes | yes | yes |

Historical combined v4 包含 BBB、Skin 和 Bioavailability 的 22 个 identity-blind conditions，其中 3 个
`none` 的 identity policy 不适用，19 个 retrieval conditions 使用 parent-disjoint；其三模型 blind/visible、
baselines 和 ablations 均保留作 historical lineage。Current 已按 task 拆分：BBB 使用
`experimental_meaningful_cns_access_v2`，Bio/Skin 使用 `record_supported_v2`，不再共享单一 22-condition/
6,887 completion gate。2026-08-10 已完成 GPT-OSS-120B current scaffold-valid：BBB 7×366，Bio 7×209，
Skin 7×245，均零失败；其中新补齐的 9 个 ChEMBL batch 为 `2460/2460` 完整。2026-08-10 之后仅 Bio
frozen full-flat/full-mechanism 与三项对应 baseline 已完成一次 formal scaffold-test；BBB/Skin formal test、
Bio direct test、current visible 与跨模型 confirmation 仍未启动。Bio full-flat matched valid 另有5次完全
同输入重复，macro-F1 均值 `0.6444`、sample SD `0.0357`。详细状态以
`STARLING_BENCHMARK_RESULTS.md` 为准。旧 26-condition TDC visibility/operational 结果也只属 historical lineage。
ClinTox 当前没有 Starling gold split，不计入这轮 v4 矩阵。

同日已完成独立 `scaffold_disjoint` matched-source ablation：ChEMBL/Starling × direct/full-flat/
full-mechanism 共 18 个 batch、`4920/4920` 完整，实际保留 neighbor policy conflict 为 0。九个 Starling-minus-
ChEMBL 点估计均为正，但只有 BBB 三个 mode 的 paired-bootstrap 95% CI 不跨 0；Bio/Skin source claim 仍不确定。
artifact 与完整统计见 `STARLING_BENCHMARK_RESULTS.md`，该 ablation 不替代 canonical parent-disjoint current runs。

RQ8 的 ChEMBL-only cumulative distance curve 是与 historical 22-condition v4 source/grouping matrix 正交的新增
实验，不计入上述条件数；完整 curve 先作为 matched-prefetch 受控实验运行，再做少量 agentic confirmation。

## 固定模型和检索设置

```text
base URL: keys.py:LITELLM_BASE_URL
requested model: nvidia/GLM-5.2-NVFP4
API key env: LITELLM_API_KEY (injected from keys.py by top-level launchers)
reasoning effort: "" (API parameter omitted; historical GLM reasoning behavior preserved)
temperature: 0
maximum output tokens: 20480
primary visibility mode: identity_blind with harness-prefetched redacted tool evidence
primary retrieval policy: parent-disjoint exclusion with top-k backfill
operational staging: disabled by default; explicit historical/deployment ablation only
endpoint concurrency budget: 512 total
default launcher shape: one global prompt pool with --parallelism 128; one launcher at a time
Morgan radius/bits: 2 / 2048
top k per group: 3
minimum similarity: 0.30
single prior: frozen from each task's none condition
SDK transport retry limit: 2; structured JSON validation: at most 4 total attempts; the last two attempts deterministically reserialize the same input to recover provider-side degeneration; remaining incomplete samples are excluded and rerun visibly at batch level

Provider request-size guard: cleaned group payloads above 750 KB use deterministic even-spacing with at most 100 evidence rows per oversized neighbor; truncation counts and byte metadata remain visible in the prompt trace.
```

The endpoint-returned model identifier is recorded from every response rather than inferred from the requested alias.

## 主要与次要指标

主要指标：test macro-F1。

次要指标：

- accuracy and class-specific confusion matrix
- retrieval coverage overall and by mechanism group
- retrieval coverage by true class, and condition-level association with paired macro-F1 gain versus same-task `none`
- prompt, completion, and total token count
- structured-output retries and failed runs
- number of LLM calls and frozen-prior reuse count
- query-SMILES trace leak count
- prompt-boundary structure, molecule-identifier, and source-name leak counts

## 统计分析

- 每个条件 macro-F1 的 95% test-set bootstrap 区间
- 每项预注册 macro-F1 差异的配对 bootstrap 区间
- 基于配对正确性的精确双侧 McNemar 检验
- 对预注册比较族的 McNemar p-value 做 Holm 校正
- 不根据 test set 选择阈值，也不做事后 deterministic label correction

配对比较预先声明在 `summarize_results.py` 中。存在失败样本或不满足相应可见性 contract 的结果，在修复并重新审计前不能用于论文。Prompt audit 检查保存的 system/user/tool request input；assistant response 不视为上游身份披露。

## 补充的非 Agent baseline

现有 MiniMol 结果可作为 learned baseline 背景，但不属于 retrieval ablation，也不能用来选择 agent setting。历史 DeepSeek 和 Bioavailability expert-policy run 仅作为 provenance 参考；它们与冻结的 GLM 矩阵不可直接比较，不进入论文主表。

上一版 strict-conflict Starling random/scaffold 已完成 MiniMol `--train-all` head、full-test Morgan KNN
`k=3` 和复用冻结 MiniMol embedding 的 cosine KNN `k=3`；当前 v4 已完成三者的 scaffold-valid 对照。
它们进入各自 lineage 的 Starling benchmark performance overview，但都不是 agent retrieval condition；
训练/选择口径、结果与入口见
`STARLING_BENCHMARK_RESULTS.md`。下文若仍出现 29-condition 计数，只描述旧 TDC-lineage historical
agentic proposal，不覆盖本节的 22-condition v4 contract。
