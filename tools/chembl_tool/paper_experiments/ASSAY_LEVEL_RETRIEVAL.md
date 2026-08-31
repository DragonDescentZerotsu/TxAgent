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

当前三任务 valid artifact：

机器可读 current index：`current_conditioned_results.json`。其中旧版本串只作为已经完成的 artifact storage
pointer；benchmark identity 始终是 `conditioned_benchmark`。

| task | benchmark input | progressive protocol | source contract | n |
|---|---|---|---|---:|
| BBB | `conditioned_benchmark/BBB_Martins` | append-only progressive | audited BBB family purity | 397 |
| Bioavailability | `conditioned_benchmark/Bioavailability_Ma` | append-only progressive | direct voter-pure L1 | 262 |
| Skin | `conditioned_benchmark/Skin_Reaction` | append-only progressive | sensitization direct/AOP | 246 |

三个任务均为 scaffold-valid、`deployment_visible_prefetched`、DeepSeek-V4-Flash-0731。当前完整曲线均为
零失败。旧 broad-L1、被拒绝的 empirical Bioavailability 和早期 progressive curves 不是当前结果。

## Retrieval and leakage contract

三个任务共享以下合同：

1. source preparation 只删除 valid/test parents 的 benchmark-defining direct rows；同一 heldout parent 的
   non-direct mechanism rows仍可作为 analog evidence。
2. query-time 对每个候选执行 `scaffold_disjoint`；相同 scaffold 不能进入 retrieval。
3. Morgan Tanimoto 最低阈值为 `0.3`。
4. candidate generation 以 cumulative record-family pool 中的 molecule 为单位，按全局 Morgan similarity
   排序；没有 per-assay neighbor cap。
5. assay identity 只作为 card provenance 和 card-selection diversity tie-break，不决定 family 可见性，
   也不再把一个 assay 强制映射到唯一 family。
6. family assignment 是 record-level；同一 physical assay 的不同 records 可以在不同 levels 出现，但一张
   record card 只能按自己的 family 解锁。
7. query SMILES 与 analog identity 对模型可见；prompt 保留
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
`provider_pools/deepseek_v4_flash_parcc_openrouter.json`。两份配置都只记录密钥环境变量名。

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

### Bioavailability legacy-gold vote-pure v1 and Skin

Bioavailability 当前六层累计 assays 为 `35 / 478 / 595 / 1,467 / 1,890 / 2,140`：actual legacy-gold-voter oral F、non-direct
bioavailability、oral AUC/Cmax exposure、absorption/solubility/permeability、gut-wall/efflux/metabolism、
hepatic clearance/metabolic stability。L2 仍记为 indirect information，不能与 L1 合并描述成 direct。
L1 membership 重放当前 benchmark provenance 中 null-condition 与 reviewed external-condition 的实际
accepted votes，而不是 endpoint 关键词。全部 463,555 source rows 保留；L1 20,543 rows、L2 153,402 rows，
L1 nonvoter=0、可映射 voter outside L1=0。相对旧 broad-L1 v1，106,963 条 L1 nonvoters 下沉 L2，710 条此前
漏在 L2 的真实 voters 校正到 L1。被拒绝的 empirical-only candidate 及其专用 provenance 已删除，不属于
当前维护范围。

Skin 两层累计 assays 为 `530 / 1,219`：gold-compatible sensitization outcome 与 sensitization AOP evidence。
LLNA final outcome 只能在 direct；MIE/KE evidence 只进入 AOP。

## Completed valid results

Macro-F1：

| task | None | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBB | 0.6207 | 0.7314 | 0.7475 | 0.7475 | 0.7563 | **0.7563** | — |
| Bioavailability legacy vote-pure v1 | 0.6314 | 0.6900 | 0.7312 | 0.7378 | 0.7545 | 0.7545 | **0.7608** |
| Skin | **0.6119** | 0.6004 | 0.5939 | — | — | — | — |

最新 matched baselines：

| task | MiniMol head | MiniMol KNN condition | MiniMol KNN all | Morgan KNN condition | Morgan KNN all |
|---|---:|---:|---:|---:|---:|
| BBB gold-v4 | **0.6674** | 0.6071 | 0.5708 | 0.5958 | 0.6174 |
| Bioavailability | 0.5872 | 0.5984 | **0.6229** | 0.5891 | 0.5888 |
| Skin | **0.6030** | 0.5755 | 0.5755 | 0.5208 | 0.5180 |

最终 level 的平均 active evidence 与模型调用长度：

| task | molecules | cards | cards/molecule | prompt tokens/call | reasoning tokens/call |
|---|---:|---:|---:|---:|---:|
| BBB L5 | 10.61 | 25.43 | 2.32 | 16,257 | 4,898 |
| Bioavailability L6 | 10.87 | 35.05 | 3.10 | 15,985 | 2,534 |
| Skin L2 | 6.02 | 15.91 | 2.13 | 10,168 | 8,832 |

Prompt/reasoning 均只对实际模型调用求平均；carry-forward 和 reused-none checkpoints 不进入长度均值。

## Canonical artifacts

完整最终 traces：

```text
BBB:
outputs/paper/
  starling_conditioned_assay_progressive_visible_v8_global_molecule_source_purity_v5/
  scaffold_valid_deepseek_v4_flash_0731/

Bioavailability current legacy-gold vote-pure rerun:

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
  source_overlays/bbb_source_family_purity_v5/
  source_overlays/bioavailability_source_family_purity_legacy_record_supported_v2_vote_pure_v1/
  source_overlays/source_family_purity_v1/skin_reaction/
  family_catalogs/bioavailability_ma_legacy_record_supported_v2_vote_pure_v1/
  family_catalogs_mechanism_tagged_v1/
    {bbb_martins_source_purity_v5,skin_reaction_source_purity_v1}/
  indices/
    bbb_martins/mechanism_tagged_v4_source_purity_v5/
    bioavailability_ma/mechanism_tagged_v4_legacy_record_supported_v2_vote_pure_v1/
    skin_reaction/mechanism_tagged_v4_source_purity_v1/
```

当前英文总图与数据表：

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

Resume 当前 Bioavailability v10（默认 task 与 output root 已配对，避免把 BBB/Skin 写入 Bio 目录）：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.run_conditioned_assay_progressive_curve \
  --tasks bioavailability_ma \
  --parallelism 128
```

三端 mixture resume 使用同一 output root 和 checkpoint：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.run_conditioned_assay_progressive_curve \
  --tasks bioavailability_ma \
  --provider-pool-config \
    tools/chembl_tool/paper_experiments/provider_pools/deepseek_v4_flash_mixture.json \
  --parallelism 256 --transport-max-retries 0
```

Random split 必须使用新 output root、`--split-scheme random` 和按当前 random valid+test union
重建的 heldout-filtered index。若本机 endpoint 不可用，可把上例配置替换为
`deepseek_v4_flash_parcc_openrouter.json`；旧 random cohort 的 checkpoint 不能通过 hash gate。

重画当前完整图：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.plot_assay_retrieval_curve \
  --conditioned-progressive-overview \
  --conditioned-progressive-task-root \
    bbb_martins=outputs/paper/starling_conditioned_assay_progressive_visible_v8_global_molecule_source_purity_v5/scaffold_valid_deepseek_v4_flash_0731 \
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
