# Starling assay-level retrieval protocol

更新时间：2026-09-05。

本文只冻结当前 retrieval、leakage、progressive reasoning 和 source-purity 合同。当前数据的恢复、逐层 records
和协作者分享见 `CURRENT_STARLING_RETRIEVAL.md`；当前 metrics、artifact roots 和 freshness 见
`current_conditioned_results.json` 与 `RESULTS.md`。避免在三处重复维护路径和结果数字。

## Scope

当前 progressive curve 包含 BBB、Bioavailability 和 Skin，使用统一 Conditioned Benchmark：

```text
data/conditioned_benchmark/<Task>/{scaffold,random}/
```

ClinTox 属于同一 benchmark，但没有 progressive L1：其 label 来自冻结 AACT toxicity-failure positives 与
SWEETLEAD/FDA-approved comparators，broad Starling toxicity records 不投票。

one-shot cumulative full-flat、append-only progressive 和 4x assay-prefix scaling 是三个独立实验族。它们可共享
底层检索组件，但 runner、manifest、artifact root 和结论不能混用。

## Retrieval and leakage contract

三个 progressive tasks 共享以下合同：

1. source preparation 只删除 valid+test parents 的 benchmark-defining direct records；同一 heldout parent 的
   eligible nondirect/mechanism records仍可作为 analog evidence。
2. query identity policy 与 split 对齐：scaffold 使用 `scaffold_disjoint`，random 使用 `parent_disjoint`；两者
   都排除 query parent 本身。
3. Morgan Tanimoto minimum similarity 为 `0.3`。
4. 单重原子 query 的 candidate 必须含同一种元素（`monatomic_query_element_match.v1`），避免 folded Morgan
   collision；普通多原子 query 不受影响。
5. candidate generation 在截至当前 level 的 cumulative record-family pool 内按 molecule 全局排序，没有
   per-assay neighbor cap。
6. family assignment 是 record-level；同一 physical assay 的不同 records 可以属于不同 families。
7. assay identity 仅作 provenance 和 card-diversity tie-break，不是 visibility gate。
8. query SMILES 和 analog identity 在当前 progressive protocol 中可见；prompt 禁止按名字识别 query，但允许
   使用一般化学知识。

`direct_only_heldout_filtered + scaffold_disjoint` 不能简写为“只从 train molecules retrieve”：heldout parent
的 nondirect evidence 可以保留，但 direct outcome 和相同 scaffold candidate 均被对应 gate 排除。

## Record and family contract

Starling run 的列名差异在 normalization 阶段由逐 source mapping 处理，retrieval 不猜 alias。当前输入是仓库
冻结的 Stage-03 canonical Parquet，不是 raw run。Stage-03 保留 endpoint、measurement、unit、molecule、assay、
context 和 `support_text`；purity overlay 只重写 `group_id` 并增加审计列。

当前分类关系只有：

```text
source_group_id -> family_key -> progressive level
```

- `source_group_id` 是 overlay record 的 `group_id`，表达 source-local 分类和 purity；
- `family_key` 是跨 ChEMBL/Starling 对齐的 canonical biological family；
- `level` 是该 source config 的累计展示次序，不是跨 source identity。

冻结 artifact 中的 `endpoint_group` 等价于 `family_key`；`family_id`/`Mechanism.tier_N` 是 legacy branch ID。
新配置通过 `evidence_family()` 声明 mapping。若一个 source group 被映射到多个 progressive families，catalog
builder 直接失败，防止 silent membership drift。

当前 `assay_compact.raw_v3` 在每个 assay×molecule 最多保留三张 representative cards，包含完整 raw card
fields 和 `support_text`；`max_support_text_chars=0` 表示不做字段级截断。Card selection 仍然意味着不是每条
source record 都会进入 index。静态 membership、split index cards 和 per-query visible cards 的区别与分享表
见 `CURRENT_STARLING_RETRIEVAL.md`。

## Append-only progressive protocol

当前协议为 `conditioned_assay_progressive_visible.v8`：

- L1 从 direct pool 选择最多 10 个不同 molecules；默认每 molecule 最多 4 张 cards；
- 后续每层最多新增 3 个此前未 active 的 molecules，每个新增最多 2 张 cards；
- 后续每层最多为 3 个 active molecules 补充新 family cards，每个新增最多 2 张；
- 新 molecule 和 active augmentation 的名额不互借；
- active evidence 只增不删，没有跨层累计的 per-molecule card cap；
- 没有 evidence delta 的 level 直接 carry forward，不调用模型；
- 下一层仍可见所有旧 cards、prior reasoning 和新增 cards。

每层只调用一次模型，不按 assay/card 单独调用。默认不启用 flip verifier。Prompt profile 是
`progressive_evidence_revision.v5`（历史 v2/v3/v4 保留各自实验合同），每层展示完整 level plan；已否决的 prefix-only 分支不保留 alternate
CLI。completion cap 为 20,480 tokens，API `reasoning_effort` 参数省略，provider 自身 reasoning 保持默认。

正式 4/2 是默认 card budget。2/1 和 8/4 只通过同一 runner 的 `--initial-card-limit` 与
`--delta-card-limit` 改变 card limits；molecule quotas、candidate ranking、prompt、identity policy 和 append-only
语义不变，不定义新协议或独立 runner。

Bounded source-cleanup diagnostics may opt into `selection_policy=endpoint_diverse_delta.v1`
inside the hash-bound `--record-review-overlay`. Later-level card selection first spans
reported endpoint types, then fills the existing budget using assay diversity and the
original card order. It does not rank result direction, consult gold, pin individual cards,
or change L1 selection. The default remains `assay_diverse.v1`. The overlay hash and
selection policy are saved in the manifest; every query must be prepared again, and a
changed visible prefix invalidates its entire progressive suffix. Cleanup plus endpoint
diversity is a combined diagnostic, not an isolated estimate of either effect.

The progressive runner also supports an opt-in `--initial-condition-map` diagnostic,
bound to the task/index SHA256 and containing only card IDs and source condition groups.
For explicit query conditions, L1 prefers exact-condition cards/molecules, then unspecified
conditions, then other conditions; Morgan ranking and the existing card selector break
ties within each bucket. Cross-condition fallback retains its raw restrictions. Null-condition
queries keep the original ordering. Budgets and later progressive updates do not change.
The complete L1 pool is materialized and map coverage is checked before inference; the
default similarity-only shortlist would be invalid for this selector. `--max-level 1` limits
preparation/execution/metrics to L1 while preserving the complete level plan in prompts.
These optional fields are resume invariants; changed selection can reuse only exact
model-visible input matches under the same budget and inference contract.
With `--initial-condition-scope cards_only`, condition priority applies only inside each
of the original Morgan-selected molecules. The top-ten molecule set/order stays unchanged,
so the existing similarity shortlist remains valid. This scope has a distinct manifest
version and must not resume the earlier molecule-and-card-priority run. Source-only map
coverage, card budgets, null-condition behavior and the full prompt level plan are unchanged.
With `--initial-condition-scope cards_exact_only`, the same Morgan molecule order is
preserved, but only exact-condition cards receive priority. Missing, unspecified and
explicit other conditions share one fallback pool with the original card selector.
An analog with no exact match retains its original cards and order. The distinct
`initial_condition_priority.cards_exact_only.v1` manifest prevents resuming older policies.

The true-query-name pilot was completed but rejected because it reveals query identity.
Its artifacts remain audit-only; the active runner rejects `query_identity_anchor`.
The completed replacement L1 diagnostic was also retired by user: do not run future
query-specific exclusions. Its historical preparations used `query_identity_exclusion` with only
an excluded name and exact query SMILES binding. It adds only
`query.identity_exclusion`: “The query molecule is not <wrong name>.” No true query
name is added. The exclusion participates in the reuse signature; default messages
are unchanged. Excluded names come from previously reviewed misidentifications,
so this targeted diagnostic is not a blind or full-cohort benchmark result.

`--query-prior-source-root` 只复用 identity-checked none/single prior 与 query tool summary，不复用 level
prediction。`--progressive-reuse-source-root` 只有在完整 selection 和 inference contract 相同时才可复用逐 query
完全相同的 visible prefix，并在首个不同 level 永久停止。跨 retrieval lineage 的 prediction 复用必须有逐 split
selected-surface zero-change receipt。

## Matched independent full-flat control

### Evidence-grounding ablation

共享 progressive preparation 支持 `--min-similarity`（默认 0.3）、
`--omit-query-prior-with-evidence` 和 `--evidence-grounding`。后两项按
`reasoning_policy=evidence_grounding_ablation.v1` 冻结并由 matched full-flat 自动继承：有证据时不把
single/None 的模型判断送入 prompt，首次获得证据不继承 None state；后续 progressive 的 evidence-based
state、append-only、update/flip 规则均不变。Grounding 只补充研究对象归属、目标标签范围和正负证据
对称评估。None 与原始 query 工具可以在同模型、同输入及显式 reuse receipt 下复用。

threshold=0 仍按相似度排序并使用原 10/3/3 molecule quotas 和 4/2 card limits。准备阶段先算出原规则
可能选中的 molecule union，再展开这些 molecules 在所有层的完整卡片，避免为每个 query 重复展开全库。
这不改变选择结果；审计保留全量检索候选数，并把实际展开数记为 `n_materialized_*`。
有 heldout alias guard 的任务继续使用完整展开路径。阈值和 reasoning policy 变化禁止复用旧 level predictions。
CPU-heavy preparation 可显式使用 `--preparation-executor process`；Linux fork workers 共享只读 index，
`--preparation-workers` 只限制准备进程数，不改变模型请求并发或科学设置。

`conditioned_assay_matched_full_flat.v1` 使用当前 progressive 每个 level 的相同累计 cards、query prior、
query/analog 工具文本、condition、模型和生成参数，每层独立判断。它不读取上一层 prediction/state，
不显示 prior-use/new-card 标记，也不使用 flip/update 指令；非空累计证据即使没有新增卡仍重新调用模型。
无任何 evidence 的层只使用同一冻结 query-only 结果。模型输出仍使用共享 card-citation schema，
`new_evidence_assessment` 在此模式下可引用全部可见卡，`revision_action` 固定为 `initial`。

新准备的 progressive/full-flat 输入采用 `progressive_evidence_revision.v5`，沿用 v4 的工具展示：query 与每个 neighbor
各有 `functional_group_tree`，数值属性摘要不再重复官能团列表。树来自常驻工具服务已有的
`molecule_properties` / `properties_compare`，同一 `/tools/batch` 预取并复用缓存；prompt 层不计算结构。
树隐藏 atom indices，表示 AccFG 子结构包含关系，不表示化学键连接；一条简短说明约束其解释。
检索 v8、卡片预算、任务 label policy 和输出 schema 不变，不增加逐 neighbor label 推断。
新 profile 和工具缓存 namespace 阻止静默复用旧输入；fresh single/None 也记录并校验 tool text profile。
旧 v2/v3 结果仍是原合同的历史结果。v4 修复跨匹配位置的树删边，不改变任务指令或输出 schema。
single/None prior 的来源继续单独记录，冻结 prior 的配对诊断不能计作整条 pipeline 的 fresh rerun。
prior 当前通过 `prefetched_molecule_properties.content` 接收完整属性和树段落；其 JSON 布局与
progressive/full-flat 的独立 tree 字段不同，未额外创建一套 prior 提取器。

V5 简化共享格式/标签指令，修正 progressive 中的 group 术语及 Skin 输出字段要求。
`query_prior` / `prior_state` 是可撤销的模型判断，实验性 claims（包括继承的说法）须引用原始卡片；
决定性 claims 说明研究对象、条件、迁移依据及对目标预测的影响，decision_summary 处理最强相关反证。
flip 允许新证据或纠正旧事实/归属/推断错误；validator 保留引用与预测变化检查，要求非空解释，
不再强制 basis 含新卡。它不验证解释的科学真实性，也不为没有证据增量的层增加模型调用。
输出字段、完整 level plan、原始证据、工具及 Bioavailability high/low 判据不变；query-only prompt 未改。
V5 六任务 valid/test progressive 与 valid matched full-flat 已完成，结果登记在
`replicate_suites.reasoning_prompt_v5_scaffold_20260913`；不得把 v4 及更早版本的 predictions 标为 v5 结果。

入口为 `run_conditioned_assay_family_curve.py --matched-progressive-root PATH --output-root NEW_PATH`，
可先加 `--prepare-only`。共享池用 `--parallelism`，每 task 可用 `--parallelism-per-task` 限流；
超过默认全局 512 必须显式指定已授权的 `--endpoint-concurrency-budget`。
split/subset、evaluation indices 和选卡预算从 source manifest 继承；input/index/
family 与当前 canonical 文件的 hash 必须匹配，逐层 prepared 文件另存 SHA，resume 拒绝输入漂移。
该入口复用共享 query executor、validator 和 summarizer，不复制 progressive 的 level outputs。

这是与当前选卡严格匹配的 one-shot 对照，区别于历史 identity-blind、per-assay Top-3 full-flat；
两者分别记录 protocol 和结果。完成状态和重复实验结果只维护在 `RESULTS.md` 与 registry；
运行中进度读取注册的 `suite_status.json`。

跨模型 matched suite 显式加 `--refresh-query-priors`：先用新模型在冻结 query 工具上生成 single/None，
再让本模型的两种 organization 共享这些 priors。原有 tools/cards 不变，完整 prepared hash 和移除
model-derived prior 后的 evidence/tool hash 分开校验。源协议、输出目录隔离在刷新 prior 前检查；
自定义 `--env-file` 和工具服务配置透传到 single/None。该阶段沿用普通 branch validation，不做竞速。

## Unattended retries and status

Progressive 和 matched full-flat 共用失败 query 补跑逻辑：单次 JSON validation 最多 4 次；
一轮结束后，只把失败 query 重新入队，已成功的 level checkpoints 不重新推理。默认
`--max-stage-requeues 3`（额外三轮），`--retry-delay-s 60`，冷却时间指数增加、上限 900 秒。
重试冻结模型、证据、阈值和生成设置；JSON repair 继续沿用共享 validator 的修复提示。
无上限重试不作为默认行为。

新增重复实验显式使用 `--retry-race-width 6`：首次 level 请求为单次；JSON repair 或失败 level
补跑时，同一提示并发提交六个副本，以首个通过完整 schema/content 校验的响应为唯一结果，不查看 gold。
其余异步 HTTP 请求取消并等待客户端连接清理；实际服务端 abort 传播取决于 endpoint。
六个副本全部计入同一个全局和 per-task request budget。`retry_races/` 保存逐副本状态、校验错误、
已返回的响应和取消记录；常规 token 指标只含保留结果，不能当作包括竞速浪费的总开销。
旧第一遍保留原顺序重试协议，新第二/三遍记录六路竞速，因此三遍的恢复策略并非完全相同。

`run_conditioned_assay_family_curve.py --matched-progressive-root SOURCE
--matched-organizations progressive full_flat --replicate-ids 2 3` 复用冻结工具、None/query prior 和累计
cards，独立重算各遍的 level 输出；progressive 只携带本遍的状态。各遍依次运行，单遍内部三个 task
共享并发池。此设计估计的是固定预取输入后的推理变异，不包含 None/query-prior 的重复采样变异。

`execution_status.json` 原子更新总 query 数、成功数、当前轮进度、失败 query、下次重试时间及
`running / retry_wait / complete / needs_attention` 状态。失败输出和 transport errors 在下次尝试前
保存到对应 `failed_attempts/`；达到上限则非零退出，保留可续跑 checkpoints。重新启动相同命令可继续
缺失/失败层，并开始新的有限重试预算；进程退出或机器重启本身不由该循环自动恢复。

程序运行时不需要 Codex 持续轮询；用户询问进度时读取状态和必要日志即可。准备模式不调用 LLM，
也不进行延时补跑。此机制不能自动修复数据/代码错误或更新 Kerberos/Duo 凭据。

## Multi-provider execution

同一 progressive runner 可使用共享应用层 provider pool。公共调度器是
`tools/chembl_tool/common/openai_provider_pool.py`，当前配置为：

```text
provider_pools/deepseek_v4_flash_mixture.json
provider_pools/deepseek_v4_flash_hosted_openrouter.json
provider_pools/deepseek_v4_flash_openrouter.json
```

配置只保存 API-key 环境变量名。调度按每端 `max_inflight` 和 latency EWMA 做 work-conserving 分配；连续
transport/429/5xx failure 会临时熔断，一次调用最多 fail over 到一个未尝试 provider。SDK transport retry 为
0，避免隐藏的重复 generation。

每次调用将 provider、requested/served model、request ID、latency 和失败链写入
`llm.execution_provider_attempts`。在 canonical model identity 相同的前提下 resume 可更换 execution provider；
prompt、retrieval、max tokens 和其它语义合同仍须完全一致。

共享 progressive/matched runner 另支持 `transport: openrouter_batch`，通过同一 provider
配置选用；请求合批、remote job 恢复和计费 provenance 由 `common/openrouter_batch.py` 负责。
每个 query 的逐层依赖不变，Batch 不使用竞速或自动 failover。完整调用、配置与恢复规则见
[`README.md#openrouter-batch-transport`](README.md#openrouter-batch-transport)。

## Current source-purity rules

### BBB

当前五层：

| level | family | cumulative assays |
|---:|---|---:|
| 1 | direct measured CNS access voters | 796 |
| 2 | functional, predicted, generic or near-direct proxy | 7,976 |
| 3 | passive permeability | 8,356 |
| 4 | efflux transport | 22,858 |
| 5 | influx transport | 22,958 |

`bbb_source_family_purity.v6` 审计全部 581,708 source rows。L1 仅允许当前 benchmark lineage 实际输出
accepted non-prediction vote 的 records；gold-contract replay 可把 nonvoter 归为 L2 near-direct，但不能授予
L1 membership。PAMPA 不进 L1；MDCK 按实际 passive/efflux/influx readout 分配；QikProp、BOILED-Egg 和
missing/generic outcomes 可作 evidence 但不是 direct measurement。硬 gate：L1 nonvoter=0、voter outside L1=0、
explicit prediction in L1=0。

### Bioavailability

当前六层累计 assays 为 `35 / 478 / 595 / 1,467 / 1,890 / 2,140`：direct oral-F voters、nondirect
bioavailability、oral AUC/Cmax、Fa、Fg、Fh。L1 membership 重放 null-condition 和 reviewed external-condition
的 actual accepted votes，不按 endpoint 关键词决定。

删除 6 条已核验 nitrendipine mismatch 后，overlay 保留 463,549 rows。L1 有 20,538 rows，L1 nonvoter=0，
可映射 voter outside L1=0。该 overlay 的 family reassignment 不修改 benchmark gold。scaffold-valid agent
prediction 的复用仅由 `receipts/bioavailability_scaffold_valid_nitrendipine_fix_zero_change.json` 授权；random 和
依赖旧 train rows 的 baselines 不在授权范围。

### Skin

当前 source-purity v5 三层累计 physical assays 为 `384 / 531 / 1,031`（各层新增 `384 / 147 / 500`）：direct voter outcomes、near-direct
outcomes/classifications、sensitization mechanisms。LLNA final outcome 只有实际参与 current gold vote 的 record
才能进入 L1；其他 measured outcome 以及 predicted/defined-approach overall classification 进入 L2；experimental
或 predicted MIE/KE 及有实质内容的 unspecified mechanisms 进入 L3。photo/light-dependent、irritation-only 和
non-contact severe cutaneous reaction records 不进入任何 level。最终 MDAM reproducibility rebuild 保持 gold/split
不变，但改变了 1/2/3 个 scaffold-valid queries 在 2/1、4/2、8/4 下的 selected surface；这些 bounded prefixes
需要 targeted replay，精确清单见 `receipts/skin_scaffold_valid_mdam_family_rebuild.json`。random cells 仍需完整 replay。

## Maintained entrypoints

```text
restore/rebuild/verify:
  rebuild_current_starling_retrieval.py

share ledgers:
  export_current_starling_level_records.py

progressive runner:
  run_conditioned_assay_progressive_curve.py

source purity:
  build_bbb_source_family_purity.py
  build_bioavailability_vote_pure_source.py
  build_conditioned_source_family_purity.py
  audit_bbb_source_family_purity.py

catalog/index:
  build_assay_family_catalog.py
  tools/chembl_tool/common/assay_retrieval.py

analysis/plot:
  analyze_progressive_trace_adoption.py
  plot_assay_retrieval_curve.py
```

公共实现放在 `common/`；task directories 只维护 family/endpoint descriptions、benchmark adapter 与 prompt
schema。新的 budget、model row 或 rerun 不应增加一次性 launcher/plotter。

## Reproduction boundary

恢复并验证 current data：

```bash
python -m tools.chembl_tool.paper_experiments.rebuild_current_starling_retrieval \
  restore-records
python -m tools.chembl_tool.paper_experiments.rebuild_current_starling_retrieval \
  verify
```

运行新的 progressive experiment 只使用：

```bash
python -m tools.chembl_tool.paper_experiments.run_conditioned_assay_progressive_curve \
  --tasks <task> [--split-scheme random] [--prepare-only]
```

确切 current input paths、hashes、run roots、可用配置与缺失 cells 均从机器可读 registry 读取，不在本页复制。
绘图器会校验 evaluation indices、family/index lineage、visibility、identity policy、selection、prompt/generation/
tool contract 和 model identity；缺失配置明确留空，不能用 historical output 补齐。

## Historical boundaries

- 4x assay-count scaling 使用 `run_assay_retrieval_curve.py`，不代表 progressive family protocol。
- cumulative-family flat runner 保留历史 per-assay Top-3 合同，不代表当前 global-molecule candidate generation。
- fixed Top-20、causal-bridge、indirect-only cap-4、prefix-only prompt 和 Bioavailability empirical-only branches
  已被否决；维护代码不保留其专用 launcher/plot mode。
- 旧 source-purity、broad-L1 和 replay artifacts 只保留 lineage/receipt；不能进入 current result cells。
- 历史细节从 Git history 或 compact receipts 查阅，不在 active 文档和文件树中复制可运行分支。
