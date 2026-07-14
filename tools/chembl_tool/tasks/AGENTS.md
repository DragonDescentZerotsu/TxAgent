# 分子证据任务扩展约定

该文件适用于 `tools/chembl_tool/tasks/` 下的所有现有和未来任务。任务目录可以补充自己的数据路径、endpoint 语义和运行命令，但不得破坏这里定义的通用分层。

## 两层证据分类

每个任务必须明确区分以下两层，不能把二者都实现成 LLM reasoning branch。

### Mechanism family

Mechanism family 是任务级、数据源无关的语义类别，也是 retrieval 和 parallel reasoning 的基本单位。例如 Bioavailability_Ma 使用 Observed direct F、Observed oral exposure、Fa、Fg、Fh；BBB 使用 direct exposure、passive permeability、efflux、influx。

要求：

- 在接入 ChEMBL、Starling 或其它数据源之前，先根据任务机制定义少量稳定的 mechanism family。
- `full_mechanism` 每个 mechanism family 最多对应一个并行 group reasoning branch。
- `direct` 通常只启用直接结果 family；`full_flat` 使用与 `full_mechanism` 相同的 evidence union，但合并成一个 reasoning branch。
- 不设跨任务统一的固定 family 数量，但必须保持克制；通常应是少量可解释类别，而不是数十个 endpoint 分支。
- family 的定义不能依赖某个数据源当前恰好有哪些 assay，也不能包含 benchmark label policy、投票规则或 deterministic override。

### Source-local endpoint group

Source-local endpoint group 是数据接入、语义归一化、筛选和审计标签。例如 ChEMBL 中需要区分 AUC、Cmax、Papp、efflux ratio、hERG、Ames 等不可直接混用的 endpoint。

要求：

- 细粒度 group 可以保留在 evidence row、index 和 provenance 中，用于发现错误归类、检查覆盖率和解释来源差异。
- 多个细粒度 group 必须先映射到一个 mechanism family；query-time 在 family 内合并候选分子后再进行 top-k retrieval。
- 同一分子在一个 family 的多个细粒度 group 中出现时，只占一个 neighbor 位置，并携带该 family 下的相关记录。
- 细粒度 group 不得各自启动 parallel LLM reasoning。否则会造成 evidence fragmentation、重复 neighbor、token 成本膨胀和 final synthesis 过载。
- 无法可靠分类的 `context_dependent` 或等价 bucket 默认不进入论文 retrieval view。只有语义明确、事先声明的 background/context family 可以进入 reasoning。

因此，旧的细粒度 ontology 可以继续存在，但它属于 source adapter 和 audit layer，不是论文方法中的专家推理 policy。论文和用户文档应将其描述为 `source-local endpoint normalization`，将 mechanism family 描述为 agent 的 reasoning organization。

## 新任务的配置位置

新增任务时遵循以下职责划分：

```text
tools/chembl_tool/tasks/<task>/experiment_config.py
  声明数据源无关的 mechanism family，以及各 source group 到 family 的映射。

tools/chembl_tool/tasks/<task>/endpoint_groups.py
  只负责 ChEMBL 等异构原始数据的细粒度 endpoint 归一化和审计标签。

tools/chembl_tool/common/evidence_contract.py
  将不同来源转换为 minimal_evidence.v1；不做 label prediction。

tools/chembl_tool/common/experiment_retrieval.py
  按 mechanism family 合并候选、检索 neighbor，并构造 direct、full_flat、full_mechanism 视图。

tools/chembl_tool/common/molecule_identity.py
  使用版本化 RDKit FragmentParent 标准化 whole record、fragment/molecular parent 和 mixture components。

tools/chembl_tool/common/retrieval_policy.py
  统一定义 operational 与 parent_disjoint 候选排除策略；task 和 source adapter 不得复制该逻辑。

tools/chembl_tool/common/retrieval_ablation.py
  对 LLM-visible sample/family input 做稳定 hash，物化整条 run 或独立 branch 复用并写 provenance。

tools/chembl_tool/common/retrieval_replay.py
  读取冻结 retrieval artifact 并校验 query identity，供 matched-prefetch 严格 replay；不得按当前 index 重检索。

tools/chembl_tool/common/identity_blind.py
  统一实现 identity redaction、harness prefetch 与 visible prefetched-tool replay；task runner 不得自行分叉该逻辑。
```

任务代码不得复制公共 retrieval、source aggregation、LLM client、validation 或 batch orchestration。

## Molecule identity 与 parent-disjoint

Molecule identity 是 retrieval ranking/filtering metadata，不属于 `minimal_evidence.v1` 的 evidence
语义。ChEMBL、Starling 和未来数据源都必须调用公共 normalizer，不得根据 source ID、名称或 task label
自行判断 same-parent。

```text
operational:
  排除完整记录 exact match 和 whole-record connectivity variant；
  保留盐型、溶剂化物、质子化形式等 same molecular parent evidence。

parent_disjoint:
  在 operational 基础上额外排除 same molecular parent；
  继续向排名后方取候选，但只在原 min_similarity 以上回填；
  候选不足时允许少于 top-k，绝不能降低阈值补满。
```

`same_parent` 只允许主 parent 相同，或一方主 parent 明确出现在另一方 mixture components 中。不能用
任意 component key 相交，否则两个无关盐会因为共享 chloride、sodium 等 counterion 被误归为同一 parent。
这里的 parent 是 RDKit 结构标准化概念，不是药理学 active moiety。共价 prodrug、代谢物和
active-moiety relation 不由 parent key 推断，必须保留为 structural analog 或由独立
PK scope annotation 表达。

论文消融的选择性重跑以 LLM-visible retrieval contract 的稳定 hash 为准。sample 输入完全相同时复用整个
run；`full_mechanism` 中只有部分 family 变化时，可复用其它独立 group outputs，再重跑变化 branch 和 final。
所有复用必须记录 `reused_from`、`reuse_reason` 和输入 hash；最终指标仍在完整 test set 上计算。

## Starling 系统背景

这里的 Starling 特指论文 *Self-Driving Datasets: From 20 Million Papers to Nuanced Biomedical Knowledge at Scale* 中的 Starling，不是其它同名软件。官方描述中，Starling 是一个面向大规模生物医学文献的 multi-agent deep research system：给定自然语言 extraction task，它会设计兼顾 precision/recall 的 corpus retrieval probes、从样本文献归纳统一 extraction schema，然后在检索子语料上生成带 supporting passage 和实验条件的结构化记录。

官方资料：

- 论文：https://arxiv.org/abs/2605.07022
- 代码：https://github.com/starling-labs/starling
- Oral Bioavailability 示例数据：https://huggingface.co/datasets/starling-labs/Oral_Bioavailability

论文报告其底层语料包含约 22.5M 篇 PubMed 论文，并强调实验条件和 supporting passages 是相对传统表格数据库的重要增量。即使论文报告了较低的单条 extraction 成本，完整的 corpus retrieval、schema induction、extraction 和 validation 仍会随任务数量重复执行；在本项目中，Starling acquisition 应视为昂贵操作。

## Starling acquisition 粒度

Starling 默认以 mechanism family 为 acquisition task/prompt/schema 的粒度，不以细粒度 endpoint group 为粒度。

具体要求：

1. 每个 task 先冻结 mechanism family，再提交 Starling acquisition。
2. 每个 mechanism family 原则上只运行一个 Starling task。不要为 family 下的每个 endpoint subtype、单位、assay system、species 或 formulation 单独启动 Starling。
3. 同一 family 内的差异通过 extraction schema 字段表达，例如 `endpoint_type`、`value`、`unit`、`species`、`assay_system`、`dose`、`formulation`、`comparator`、`support_text` 和 `confidence`。
4. Starling 输出接入 TxAgent 后，可以根据这些结构化字段派生细粒度 audit group，但这些 group 不产生新的 acquisition prompt，也不产生独立 LLM reasoning branch。
5. 只有当一个 family 无法用一致的检索语义和统一 schema 表达，并且抽样审计证明 precision/recall 明显受损时，才允许拆成多个 Starling tasks；拆分理由和额外成本必须记录在 task-local `AGENTS.md`。
6. 不得因为 ChEMBL 中存在很多旧 Tier.endpoint_group，就一一复制成 Starling prompts。ChEMBL ontology 用于整理已有异构 assay；Starling schema 应直接围绕 mechanism family 获取带条件的文献证据。

因此，Starling 数据不需要像 ChEMBL 那样维护大量细碎 group 作为采集单元。它仍然必须保留 endpoint subtype 和实验条件供 audit、scope 判断和 provenance 使用，但这些信息是 family 内字段，不是额外的 agent 分支。

## 新任务检查清单

新增任务或新数据源前必须确认：

1. mechanism family 数量少且有明确任务语义。
2. direct evidence 与 mechanistic/surrogate evidence 的边界明确。
3. 每个 source-local group 都映射到至多一个默认 mechanism family；需要复用时必须解释原因。
4. `full_flat` 与 `full_mechanism` 使用完全相同的 evidence union。
5. parallel group reasoning 数等于启用的 mechanism family 数，而不是底层 endpoint group 数。
6. Starling task 数默认等于需要采集的 mechanism family 数，不随 endpoint subtype 数量增长。
7. schema 保留数值、单位、实验条件、scope、supporting passage、quality/uncertainty 和 provenance。
8. 缺失 SMILES、结构标准化失败、重复 source molecule 和无法分类记录都有可审计统计。
