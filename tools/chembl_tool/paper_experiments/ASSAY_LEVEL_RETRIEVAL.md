# Starling assay-level retrieval

更新时间：2026-08-31。

本文档只维护当前合同、最终 valid 结果、可复现入口和历史边界。逐次 smoke、失败重试、endpoint
切换和被替代的 trace 不在这里重复记录；它们已集中归档，避免主输出目录继续膨胀。

## Canonical snapshot

当前 gold 统一属于 Conditioned Benchmark：

```text
data/conditioned_benchmark/<Task>/scaffold/
```

并列 random split 位于 `data/conditioned_benchmark/<Task>/random/`，但本页现有结果仍全部是 scaffold-valid。
random agent run 必须先按 random valid+test parents 重建 heldout-filtered index，不能复用本页的 scaffold-heldout index。

历史的 molecule-only、BBB gold-vN 和 selected-vN 名称只出现在 migration receipt 中，不再作为 runner
输入或结果标签。ClinTox 没有合格 external-condition taxonomy，因此全部使用 null condition；它仍是统一
四任务 benchmark 的一部分，但当前 progressive 曲线只包含 BBB、Bioavailability 和 Skin。

当前三任务 valid artifact 状态：

机器可读 current index：`current_conditioned_results.json`。其中旧版本串只作为已经完成的 artifact storage
pointer；benchmark identity 始终是 `conditioned_benchmark`。

| task | benchmark input | source contract | n | artifact status |
|---|---|---|---:|---|
| BBB | `conditioned_benchmark/BBB_Martins` | audited BBB family purity | 397 | current |
| Bioavailability | `conditioned_benchmark/Bioavailability_Ma` | direct voter-pure L1 | 262 | current via scaffold-valid zero-change retrieval receipt |
| Skin | `conditioned_benchmark/Skin_Reaction` | voter-only direct / nonvoter outcome / AOP | 246 | v2 index current; v1 predictions stale |

三个任务均为 scaffold-valid、`deployment_visible_prefetched`、DeepSeek-V4-Flash-0731。BBB 的完整曲线当前且
零失败。Skin strict-voter-L1 v2 已完成 source overlay、catalog 和 scaffold/random index，但现有完整曲线来自
旧 broad-L1 v1，必须 replay。Bioavailability 删除六条错误 nitrendipine identity records 后 index hash 虽然变化，
但按冻结 protocol 对 262 条 valid rows、L1-L6 逐 query 重建后，selected active molecules/cards 的 canonical
surface hash 与修复前完全相同，变化 query 为 0。因此 retained agent score 通过 split-scoped zero-change
receipt 继续有效，不需要新 LLM 调用；random split 尚未执行同类审计。旧 broad-L1 和早期 progressive curves
不是当前结果。

## Retrieval and leakage contract

三个任务共享以下合同：

1. source preparation 只删除 valid/test parents 的 benchmark-defining direct rows；同一 heldout parent 的
   non-direct mechanism rows仍可作为 analog evidence。
2. query-time identity policy 与 split 对齐：scaffold split 使用 `scaffold_disjoint`，random split 使用
   `parent_disjoint`；两者都不允许 query parent 自身进入 retrieval。
3. Morgan Tanimoto 最低阈值为 `0.3`。
4. 对只有一个重原子的 query，candidate 必须含同一种元素
   (`monatomic_query_element_match.v1`)。这是 folded 2048-bit Morgan 的退化保护，避免 `[Pb]`/`[U]`
   这类不同元素因单 bit collision 被误报为 Tanimoto 1.0；普通多原子 query 不受影响。
5. candidate generation 以 cumulative record-family pool 中的 molecule 为单位，按全局 Morgan similarity
   排序；没有 per-assay neighbor cap。
6. assay identity 只作为 card provenance 和 card-selection diversity tie-break，不决定 family 可见性，
   也不再把一个 assay 强制映射到唯一 family。
7. family assignment 是 record-level；同一 physical assay 的不同 records 可以在不同 levels 出现，但一张
   record card 只能按自己的 family 解锁。
8. query SMILES 与 analog identity 对模型可见；prompt 保留
   `Do not identify the query by name even if its structure is recognizable.`，不禁止模型使用一般化学知识。

`direct_only_heldout_filtered + scaffold_disjoint` 同时保留了 mechanism analog 的覆盖能力和 scaffold-split
隔离，不能简写为“只从 train molecules retrieve”。

## Append-only progressive protocol

L1 从 direct pool 中选择最多 10 个不同 molecules；每个 molecule 最多 4 张 cards。后续每个 level：

- 最多新增 3 个此前从未 active 的 molecules，每个从该 level 新增最多 2 张 cards；
- 最多为 3 个已经 active 的 molecules 增补该 level 新解锁的 cards，每个最多 2 张；
- 两类名额不互借；已经 active 的 molecule 不占“新增 molecule”名额；
- active evidence 只增加、不删除，没有全程累计的 per-molecule card cap；
- 如果本 level 没有 evidence delta，直接 carry forward，不调用模型；
- 所有旧 cards 在下一轮仍可见，并标记上一轮是否进入 decision basis；模型先评估新增 cards 对 prior
  decision 的影响，但仍可回看旧 evidence 修正早期判断。

模型每个 level 只调用一次，不按 card 或 assay 分别调用。默认不启用 flip verifier；没有差异化新增信息时，
第二次同模型确认只会增加成本和保守偏差。

Prompt profile 为 `progressive_compact_tools_short_aliases.v2`。Card 使用短 alias，省略重复 ID 列表；原始
endpoint、measurement、species、conditions、support text 和 provenance 保留，不调用 GPT-OSS summary，
不做字段级截断。`reasoning_effort` 参数省略、provider thinking 保持默认，completion cap 为 20,480 tokens。

### Multi-provider execution

Progressive runner 可用一个共享的应用层 provider pool 混合本机、PARCC tunnel 与 OpenRouter。这里不用
HAProxy 直接代理异构后端，因为三端具有不同 model alias、鉴权和 reasoning 参数。公共调度器位于
`tools/chembl_tool/common/openai_provider_pool.py`。包含本机 endpoint 的完整配置位于
`provider_pools/deepseek_v4_flash_mixture.json`；本机不可用时使用只包含 PARCC 与 OpenRouter 的
`provider_pools/deepseek_v4_flash_parcc_openrouter.json`；只使用 OpenRouter 双 key failover 时使用
`provider_pools/deepseek_v4_flash_openrouter.json`。三份配置都只记录密钥环境变量名。

调度是 work-conserving least-normalized-load：每端有独立 `max_inflight`，空闲 slot 按近期 latency EWMA
动态接收下一条请求。连续 transport/429/5xx failure 会暂时熔断该端；一次调用最多 fail over 到一个尚未尝试
的 provider。每端还可设置独立 timeout，使慢端长尾在有 checkpoint 的应用层转交给其它 provider。SDK
transport retry 仍保持 0，避免不可见的重复长 generation。OpenRouter 显式发送
`reasoning.enabled=true`；本机和 PARCC 保持 provider-default reasoning。

每次真实调用在 level output 的 `llm.execution_provider_attempts` 保存 provider、requested/served model、
request ID、latency 和失败链；manifest 只保存 API-key 环境变量名，不保存密钥。Resume 允许在相同 canonical
model identity 下改变 execution provider，但 prompt、retrieval、max tokens 和其它语义合同仍必须完全一致。

## Source-family purity

### BBB v5

`bbb_source_family_purity.v5` 审计全部 581,708 条 source rows，只修改 `group_id` 并附加审计字段，不改
measurement/support text。L1 仅允许当前 accepted non-prediction voters，或能够重放
`experimental_meaningful_cns_access_v4` gold contract 的实验 CNS-access outcomes。

五层依次为：

| level | family | cumulative assays |
|---:|---|---:|
| 1 | direct measured CNS access | 1,161 |
| 2 | central functional / prediction / generic near-direct proxy | 7,976 |
| 3 | passive permeability | 8,356 |
| 4 | efflux transport | 22,858 |
| 5 | influx transport | 22,958 |

PAMPA 不在 L1；MDCK records 按实际 permeability/efflux/influx readout 分配；QikProp、BOILED-Egg 和
missing/generic BBB outcomes 保留为 evidence，但不能冒充 gold-compatible direct measurement。明确 efflux
signal 优先于泛化 uptake/transporter 词。v5 的 purity gate 为：L1 residual prediction=0、L1 residual
non-gold-contract=0、cross-family efflux/influx precedence violation=0、被移动 gold-vote rows=0。

Gold-v4 本身来自 source-index 级 qualitative-direction review：批准 65 条新增 votes，得到 3,675 parents，
train/valid/test 为 2,945/365/365；shared-parent label/split changes 均为 0，identity/scaffold overlap 为 0。
Conditioned-v3 将其展开为 3,053/397/396 个 parent-condition rows，并保持 parent/scaffold disjoint。

### Bioavailability current vote-pure source and Skin

Bioavailability 当前六层累计 assays 为 `35 / 478 / 595 / 1,467 / 1,890 / 2,140`：actual benchmark-voter oral F、non-direct
bioavailability、oral AUC/Cmax exposure、absorption/solubility/permeability、gut-wall/efflux/metabolism、
hepatic clearance/metabolic stability。L2 仍记为 indirect information，不能与 L1 合并描述成 direct。
L1 membership 重放当前 benchmark provenance 中 null-condition 与 reviewed external-condition 的实际
accepted votes，而不是 endpoint 关键词。全部 463,555 source rows 保留；L1 20,543 rows、L2 153,402 rows，
L1 nonvoter=0、可映射 voter outside L1=0。相对旧 broad-L1 v1，106,963 条 L1 nonvoters 下沉 L2，710 条此前
漏在 L2 的真实 voters 校正到 L1。被拒绝的 empirical-only candidate 及其专用 provenance 已删除，不属于
当前维护范围。

Skin 两层累计 assays 为 `530 / 1,219`：gold-compatible sensitization outcome 与 sensitization AOP evidence。
LLNA final outcome 只能在 direct；MIE/KE evidence 只进入 AOP。

## Last completed valid results and freshness

Macro-F1：

| task | None | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBB | 0.6207 | 0.7314 | 0.7475 | 0.7475 | 0.7563 | **0.7563** | — |
| Bioavailability current via zero-change receipt | 0.6314 | 0.6900 | 0.7312 | 0.7378 | 0.7545 | 0.7545 | **0.7608** |
| Skin | **0.6119** | 0.6004 | 0.5939 | — | — | — | — |

最近一次 matched baselines：

| task | MiniMol head | MiniMol KNN condition | MiniMol KNN all | Morgan KNN condition | Morgan KNN all |
|---|---:|---:|---:|---:|---:|
| BBB gold-v4 | **0.6674** | 0.6071 | 0.5708 | 0.5958 | 0.6174 |
| Bioavailability | 0.5872 | 0.5984 | **0.6229** | 0.5891 | 0.5888 |
| Skin | **0.6030** | 0.5755 | 0.5755 | 0.5208 | 0.5180 |

BBB 与 Skin baseline 行仍匹配当前 cohort。Bioavailability baseline 使用了修复前多两条 train rows 的训练集，
因此必须重训；上表 baseline 的 Bio 行仅保留为 pre-fix reference，不能与当前 agent 行当作 matched comparison。

最终 level 的平均 active evidence 与模型调用长度：

| task | molecules | cards | cards/molecule | prompt tokens/call | reasoning tokens/call |
|---|---:|---:|---:|---:|---:|
| BBB L5 | 9.98 | 23.39 | 2.26 | 15,360 | 2,680 |
| Bioavailability L6 | 10.87 | 35.05 | 3.10 | 15,985 | 2,534 |
| Skin L2 | 6.02 | 15.91 | 2.13 | 10,168 | 8,832 |

Prompt/reasoning 均只对实际模型调用求平均；carry-forward 和 reused-none checkpoints 不进入长度均值。

## Canonical artifacts

完整最终 traces：

```text
BBB:
outputs/paper/
  starling_conditioned_assay_progressive_visible_bbb_source_purity_v6_card_budget_4_2_v1/
  scaffold_valid_deepseek_v4_flash_0731/

Bioavailability retained vote-pure run (current agent predictions via zero-change receipt):

outputs/paper/
  starling_conditioned_assay_progressive_visible_v10_bio_legacy_gold_vote_pure_v1/
  scaffold_valid_deepseek_v4_flash_0731/

Skin:
outputs/paper/
  starling_conditioned_assay_progressive_visible_v7_source_purity_v1/
  scaffold_valid_deepseek_v4_flash_0731/
```

当前可复现 inputs：

```text
outputs/paper/starling_conditioned_assay_family_curve_v1/
  source_overlays/bbb_source_family_purity_v6/
  source_overlays/bioavailability_source_family_purity_legacy_record_supported_v2_vote_pure_v1/
  source_overlays/source_family_purity_v2/skin_reaction/
  family_catalogs/bioavailability_ma_legacy_record_supported_v2_vote_pure_v1/
  family_catalogs_mechanism_tagged_v1/
    {bbb_martins_source_purity_v6,skin_reaction_source_purity_v2}/
  indices/
    bbb_martins/mechanism_tagged_v4_source_purity_v6/
    bioavailability_ma/mechanism_tagged_v4_legacy_record_supported_v2_vote_pure_v1/
    skin_reaction/mechanism_tagged_v4_source_purity_v2/
```

Source-purity 的正式构建链只有四个共享入口，不为单个版本新增 launcher：

1. `build_bbb_source_family_purity.py`、`build_bioavailability_vote_pure_source.py` 或
   `build_conditioned_source_family_purity.py` 生成 task-specific voter-pure overlay；
2. `build_assay_family_catalog.py` 生成 record-family catalog；
3. `python -m tools.chembl_tool.common.assay_retrieval build-index` 按 scaffold/random 的 valid+test heldout
   parent union 构建对应 index；
4. `run_conditioned_assay_progressive_curve.py` 是唯一 progressive runner。

各 entrypoint 的参数与当前 artifact 路径由 `--help`、runner defaults 和
`current_conditioned_results.json` 共同约束；历史 `selected_vN` 或 source-purity v1 路径不能作为新 run 默认输入。

Bioavailability scaffold-valid source repair 的复用边界固定在：

```text
tools/chembl_tool/paper_experiments/receipts/
  bioavailability_scaffold_valid_nitrendipine_fix_zero_change.json
```

该 receipt 只授权 scaffold-valid agent predictions；不授权 random artifacts 或依赖修复前 train rows 的
MiniMol/Morgan baselines。

最后一版英文总图与数据表（Bioavailability agent curve 已由 zero-change receipt 复用；图中 Bio baseline
仍为 pre-fix reference，重训后需重画）：

```text
outputs/paper/
  starling_conditioned_assay_progressive_visible_v10_bio_legacy_gold_vote_pure_v1/
  analysis/three_task_progressive_overview_current_matched_baselines/
    progressive_overview_metrics.tsv
    summary.json
    figures/three_task_progressive_overview_current_matched_baselines.{png,svg}
```

Top-20 historical control 只在主目录保留 compact analysis 与 visible-standard `none` single-cache；其余完整
traces、旧 progressive versions、replay batches、旧 indices 和 prompt audits 可逆归档到：

```text
outputs/archive/conditioned_assay_superseded_20260827/
```

归档不参与当前 evidence retrieval；它保留原相对子目录，必要时可移回。历史 family-curve 与 Top-20
canonical paths 只保留 compatibility symlink，以继续解析 aggregate metrics 和 progressive single-cache。
Gold/source datasets、migration receipts、current metrics/predictions 和最终三任务 traces 未删除。
唯一移出 `data/` 主树的是从未进入正式实验的 unselected
`processed_starling_context_conditioned_reviewed_v1` 和已被 v2 取代的 BBB source-family review ledger；两者
仍完整保存在 archive。

## Maintained entrypoints

```text
progressive runner:
  tools/chembl_tool/paper_experiments/run_conditioned_assay_progressive_curve.py

multi-provider scheduler/config:
  tools/chembl_tool/common/openai_provider_pool.py
  tools/chembl_tool/paper_experiments/provider_pools/deepseek_v4_flash_mixture.json
  tools/chembl_tool/paper_experiments/provider_pools/deepseek_v4_flash_parcc_openrouter.json
  tools/chembl_tool/paper_experiments/provider_pools/deepseek_v4_flash_openrouter.json

shared retrieval/state:
  tools/chembl_tool/common/assay_retrieval.py
  tools/chembl_tool/common/progressive_assay_reasoning.py

source purity and audits:
  tools/chembl_tool/paper_experiments/build_bbb_source_family_purity.py
  tools/chembl_tool/paper_experiments/build_bioavailability_vote_pure_source.py
  tools/chembl_tool/paper_experiments/build_conditioned_source_family_purity.py
  tools/chembl_tool/paper_experiments/audit_bbb_source_family_purity.py
  tools/chembl_tool/paper_experiments/analyze_source_family_purity_retrieval_changes.py

catalog/index:
  tools/chembl_tool/paper_experiments/build_assay_family_catalog.py
  tools/chembl_tool/common/assay_retrieval.py

analysis/plot:
  tools/chembl_tool/paper_experiments/analyze_progressive_trace_adoption.py
  tools/chembl_tool/paper_experiments/plot_assay_retrieval_curve.py

parallel historical experiments:
  tools/chembl_tool/paper_experiments/run_conditioned_assay_family_curve.py
  tools/chembl_tool/paper_experiments/run_assay_retrieval_curve.py
```

公共逻辑只放在 `common/`；task directories 只维护 family/endpoint descriptions、benchmark adapter 和 prompt
schema。Progressive、cumulative-family 和 4x assay-prefix 三条 pipeline 共享检索组件，但 runner、manifest 和
artifact root 分离，不能互相覆盖。

## Reproduction

重建当前 Bioavailability vote-pure source overlay：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.build_bioavailability_vote_pure_source
```

Bioavailability scaffold-valid 已由 zero-change receipt 授权复用，不执行下列 LLM replay。只有未来需要独立
fresh replication 时，才使用新的 output root 运行：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.run_conditioned_assay_progressive_curve \
  --tasks bioavailability_ma \
  --parallelism 128
```

可选 fresh replication 的三端 mixture 使用同一新 output root 和 checkpoint：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.run_conditioned_assay_progressive_curve \
  --tasks bioavailability_ma \
  --provider-pool-config \
    tools/chembl_tool/paper_experiments/provider_pools/deepseek_v4_flash_mixture.json \
  --parallelism 256 --transport-max-retries 0
```

Random split 必须使用 `--split-scheme random` 和按当前 random valid+test union 重建的 heldout-filtered
index。BBB/Skin random artifacts 当前；Bioavailability random index 同样在 source-identity repair 后失配，
尚未执行 zero-change 审计，必须单独审计或 targeted replay。若本机 endpoint 不可用，可把上例配置替换为
`deepseek_v4_flash_parcc_openrouter.json`；任何旧 checkpoint 都必须通过完整 hash gate。

重画最后一版完整图（当前 root 继续用于 Bioavailability agent curve；baseline 重训后替换其 baseline root）：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.plot_assay_retrieval_curve \
  --conditioned-progressive-overview \
  --conditioned-progressive-task-root \
    bbb_martins=outputs/paper/starling_conditioned_assay_progressive_visible_bbb_source_purity_v6_card_budget_4_2_v1/scaffold_valid_deepseek_v4_flash_0731 \
  --conditioned-progressive-task-root \
    bioavailability_ma=outputs/paper/starling_conditioned_assay_progressive_visible_v10_bio_legacy_gold_vote_pure_v1/scaffold_valid_deepseek_v4_flash_0731 \
  --conditioned-progressive-task-root \
    skin_reaction=outputs/paper/starling_conditioned_assay_progressive_visible_v7_source_purity_v1/scaffold_valid_deepseek_v4_flash_0731 \
  --conditioned-progressive-baseline-root \
    bbb_martins=outputs/baselines/starling_conditioned_bbb_gold_v4_valid_v1 \
  --conditioned-progressive-baseline-root \
    bioavailability_ma=outputs/baselines/starling_conditioned_valid_v1 \
  --conditioned-progressive-baseline-root \
    skin_reaction=outputs/baselines/starling_conditioned_valid_v1 \
  --analysis-dir \
    outputs/paper/starling_conditioned_assay_progressive_visible_v10_bio_legacy_gold_vote_pure_v1/analysis/three_task_progressive_overview_current_matched_baselines \
  --output-stem three_task_progressive_overview_current_matched_baselines
```

绘图器会校验 task/sample count、完整 agent metrics 和 `n_failed_runs=0`；baseline cohort 不匹配时默认报错，
只有显式 `--omit-mismatched-progressive-baselines` 才允许诊断性省略，并在 summary 中记录原因。

同一张图比较多个 progressive 配置时，使用通用的
`CONFIG:TASK=PATH` 注册方式。绘图器校验相同 task、level family、evaluation input、sample count、visibility、
identity policy、evaluation indices、非 card-budget selection 语义、prompt/generation/tool contract 和 model
identity；解析 level metrics 与 query checkpoints，固定输出 Macro-F1、active molecules、cards/molecule、
prompt tokens 和 reasoning tokens 五排对比。若 index 或 family-manifest hash 不同，必须用
`--conditioned-progressive-config-lineage-receipt TASK=PATH` 显式注册 selected-surface zero-change receipt，
否则拒绝画图。当前保留的 4/2 与 8/4 完整对比图入口如下；其中 BBB 是 strict-voter-L1 v6 current，
Skin panel 是 pre-v2 historical，Bioavailability 的 lineage 边界见下文 receipt：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.plot_assay_retrieval_curve \
  --conditioned-progressive-config-comparison \
  --conditioned-progressive-config-task-root '4/2:bbb_martins=outputs/paper/starling_conditioned_assay_progressive_visible_bbb_source_purity_v6_card_budget_4_2_v1/scaffold_valid_deepseek_v4_flash_0731' \
  --conditioned-progressive-config-task-root '4/2:bioavailability_ma=outputs/paper/starling_conditioned_assay_progressive_visible_v10_bio_legacy_gold_vote_pure_v1/scaffold_valid_deepseek_v4_flash_0731' \
  --conditioned-progressive-config-task-root '4/2:skin_reaction=outputs/paper/starling_conditioned_assay_progressive_visible_card_budget_4_2_control_v1/scaffold_valid_deepseek_v4_flash_0731/skin' \
  --conditioned-progressive-config-task-root '8/4:bbb_martins=outputs/paper/starling_conditioned_assay_progressive_visible_bbb_source_purity_v6_card_budget_8_4_v1/scaffold_valid_deepseek_v4_flash_0731' \
  --conditioned-progressive-config-task-root '8/4:bioavailability_ma=outputs/paper/starling_conditioned_assay_progressive_visible_card_budget_8_4_tool_prefetch_fixed_v1/scaffold_valid_deepseek_v4_flash_0731/bio' \
  --conditioned-progressive-config-task-root '8/4:skin_reaction=outputs/paper/starling_conditioned_assay_progressive_visible_card_budget_8_4_v1/scaffold_valid_deepseek_v4_flash_0731/skin' \
  --conditioned-progressive-config-lineage-receipt 'bioavailability_ma=tools/chembl_tool/paper_experiments/receipts/bioavailability_scaffold_valid_nitrendipine_fix_zero_change.json' \
  --conditioned-progressive-baseline-root 'bbb_martins=outputs/baselines/starling_conditioned_bbb_gold_v4_valid_v1' \
  --analysis-dir outputs/paper/analysis/progressive_record_card_budget_4_2_vs_8_4 \
  --output-stem progressive_record_card_budget_4_2_vs_8_4
```

配置比较图的 performance 行先画两种配置完全一致的 matched `None`，再画
4/2 与 8/4 progressive levels，最后画 task-specific MiniMol head、MiniMol
KNN condition/all 和 Morgan KNN condition/all references。`None` 必须逐 task
跨配置严格相等；baseline 必须覆盖相同 evaluation sample count。下面四行只
描述 progressive levels 的 evidence/token resources，不给 `None` 或 baseline
伪造资源值。Bioavailability baseline 仍是删除两条 train rows 之前的 pre-fix
reference，需在重训前保持该标注。

BBB 必须使用上面成对的 strict-voter-L1 v6 4/2 与 8/4 roots。两组各 397/397、
0 failed queries，使用相同 input、family manifest、heldout-filtered index、query prior、
model identity 和 1,493 个 model-called checkpoints；两组均为 fresh run，没有跨预算
复用 level prediction，也不需要跨 lineage receipt。旧 v5 `tool_prefetch_fixed_v1`
roots 只保留历史 provenance，不能进入当前结果图。

Bioavailability 8/4 必须使用上面的 `tool_prefetch_fixed_v1` lineage。该次
preparation 在 262 个 query、1,572 个 query-level 上没有出现任何
`127.0.0.1:8765` 连接或权限失败，并与 4/2 的 5,694 个共享 analog×tool
surface 完全一致。两种预算共同存在的 72 个确定性 MMP runtime error 不构成
配置间差异。旧 `...card_budget_8_4_v1.../bio` 受 sandbox loopback 失败污染，
不得进入结果图；完整 receipt 见
`receipts/bioavailability_scaffold_valid_8_4_tool_prefetch_fixed.json`。

预算实验不使用独立 runner。正式通用入口仍是
`run_conditioned_assay_progressive_curve.py`：默认 `--initial-card-limit 4 --delta-card-limit 2`，8/4 仅改为
`--initial-card-limit 8 --delta-card-limit 4`。`--query-prior-source-root` 只从完成的 progressive artifact 复用
identity-checked none/single prior 与 query tool summary，不复用 level prediction；
`--progressive-reuse-source-root` 只允许 selection contract 和其它冻结设置完全相同的 run，并只复用逐 query
从 L1 开始完全相同的 model-visible prefix，在首个变化 level 永久停止复用。4/2 与 8/4 不跨预算复用 level
prediction；实验准备或审计使用 `--prepare-only`。仓库内没有为本轮保留一次性 audit Python 入口。

## Historical boundaries

- 4x assay-count scaling 仍使用 `run_assay_retrieval_curve.py`，按 frozen assay relevance prefix 扩展到 all；
  它不是 progressive family curve。
- cumulative-family flat runner 仍使用每 assay Top-3 的历史合同；它不代表当前 global-molecule v8 candidate
  generation。它的 frozen catalog/index/replay defaults 显式指向 archive，不会污染当前 source-purity inputs。
- fixed Top-20 standard/causal-bridge 与 blind/visible matrix 是已完成且失败的 ablation；只保留 frozen
  artifacts/receipts，专用 launcher、causal-adoption wrapper 和绘图模式已从维护代码删除。
- indirect-only cap-4 是历史诊断 artifact；专用 schedule/plot mode 已删除，不是当前 append-only evidence
  budget。
- Bioavailability empirical-only gold/selected-v2/source-purity-v2/v9 是已拒绝且已删除的分支；当前代码不再提供
  其构建或绘图入口。若未来重新提出该方法，必须作为新的版本化 candidate 从头审查，不能复用当前结果名称。
- historical roots 在归档中保留 lineage；不得把其中指标与当前 BBB gold-v4 matched cohort 混表。
