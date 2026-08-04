# Skin_Reaction task notes

本文件记录 Skin_Reaction 的 task-specific 语义、ChEMBL evidence ontology、assay screening 规则方向、
endpoint group 设计和 reasoning 约束。通用 ChEMBL workflow、batch/resume、viewer 和输出目录规范仍以仓库根
`AGENTS.md` 为准。

当前论文路径由 `experiment_config.py` 将 source-local groups 合并为 4 个 mechanism families：direct skin
reaction、sensitisation AOP、phototoxicity/irritation/local damage 和 skin exposure。Tier 5 只留在 source
library 做数据审计。下文 `retrieve_neighbors.py` 和历史 v1 batch 按 `Tier.endpoint_group` 运行的说明，是旧
native task runner 的复现记录；新实验不得据此为每个细粒度 group 启动并行 reasoning。

2026-07-23 strict-hop availability census 见
`outputs/chembl_tool/tasks/skin_reaction/distance_expansion/analysis/hop_availability_census/`。GSK3B activity
到 NRF2 state 的 graph/data gate 通过（7,950 parents；parent-disjoint >=1 coverage 34.15%），但直接药理性
NRF2 regulation 不是 sensitizer-specific，必须保留 scope caveat。KEAP1-NRF2 PPI coverage 只有 2.44%，
CUL3 H2 为 0；本 task 没有可用 strict H2。不得据此新增旧 paper condition。

2026-07-22 已导入 Starling direct Skin_Reaction parquet，以及 sensitization AOP、phototoxicity/irritation/local
damage 和 skin exposure 三个 mechanism-family acquisition。`build_starling_evidence_library.py` 通过公共 profile
reader 将四类数据构建成一个 molecule-level index，供 Starling direct/full-flat/full-mechanism 三个论文条件共用。

## Task 定义

目标不是训练一个单纯的 QSAR skin-reaction classifier，而是构建可审计的 skin-reaction evidence retrieval
和 reasoning workflow：给定 query molecule，从 ChEMBL 中检索与皮肤不良反应判断相关的相似分子实验读数，
再由 reasoning LLM 判断这些 analog evidence 是否能 transfer 到 query molecule。

当前 Starling gold benchmark：

```text
data/processed_starling/Skin_Reaction/random/{train.jsonl,valid.jsonl,test.jsonl}
data/processed_starling/Skin_Reaction/scaffold/{train.jsonl,valid.jsonl,test.jsonl}

fields:
  drug: query SMILES
  Y: Skin_Reaction label
```

当前 frozen build 有 2,456 个 binary parents，两种构造方法的 valid/test target 均为 245。冲突 parent
按 accepted source records 计算 70% agreement，同 PMID 多条 record 分别计票，精确 tie 拒绝。正式运行前
必须按各自 valid+test union 的 `heldout_molecule_labels.jsonl` 重建 train-only retrieval index。构建命令和审计协议见
`tools/chembl_tool/common/starling/STARLING_BENCHMARK_PROTOCOL.md`。

当前二分类约定：

```text
Y=1 -> skin sensitizer / positive
Y=0 -> non-sensitizer / negative
```

原始 404-molecule task 来自 binary LLNA skin-sensitization 数据，不是任意 clinical dermatologic
reaction。新的 Starling-held-out benchmark 因此只接受 sensitization 或 allergic contact
dermatitis/contact allergy scope 的明确 positive/negative record；irritation、generic local damage、
skin exposure 和 inconclusive 不转成 gold label。

### Paper-facing tier 的距离语义

Skin 的 `Mechanism.tier_1` 至 `tier_4` 是逐步扩大的 evidence scope，不是像 oral bioavailability
`F = Fa × Fg × Fh` 那样的层层因果分解：

```text
Tier 1:
  direct sensitization/contact-allergy anchors；最接近当前 gold。

Tier 2:
  sensitization AOP key events；与 gold 对齐，但单一 key event 不等于最终 clinical outcome。

Tier 3:
  phototoxicity、irritation、corrosion、local skin damage；都是 skin hazard，
  但不是当前 sensitization label 的组成机制。

Tier 4:
  skin permeability、retention 和 exposure context；只改变 hazard 表现的 plausibility，
  不能单独证明 sensitization。
```

因此从 direct 扩展到 full 并不是加入越来越完整的同一条 causal chain，而是加入越来越远、语义可能
不完全对齐的 evidence。`experiment_config.py` 是 paper-facing source/group mapping 的代码真相。

### 2026-07-27 Starling retrieval degradation trace audit

Deployment-visible parent-disjoint Starling 的 observed macro-F1：

| split | direct | full flat | full mechanism |
|---|---:|---:|---:|
| random | 0.643141 | 0.635255 | 0.629991 |
| scaffold | 0.597332 | 0.592139 | 0.583574 |

Full-flat 与 full-mechanism 的 LLM-visible evidence-row multiset 在 random/scaffold 均为 380/380
逐 query exact matches；mechanism 没有获得额外 rows，只是把同一 union 拆成多个 branches 再做 final。
Direct 到 mechanism 的 prediction flips 为：

```text
random:   38 flips，17 corrected / 21 broken，net -4 correct
scaffold: 29 flips，12 corrected / 17 broken，net -5 correct
```

Trace 中主要 failure modes 是：phototoxicity/irritation 被提升成 sensitization hazard、Tier 4 exposure
support 被误当 risk、weak/distant AOP narrative 被 branch packaging 放大、broad mixed negatives 稀释
较近 positive anchor，以及无 neighbor 时的 prompt-boundary instability。Starling random 的平均 logical
tokens 从 direct 27.7k 增至 flat 77.4k、mechanism 96.1k；scaffold 为 26.9k、73.4k、91.6k。
更多 tokens 表示更多 group calls/重复 synthesis，不等于更多 label-aligned information。

完整量化和逐 flip trace：

```text
outputs/paper/skin_reaction_retrieval_diagnostic/agent_quant_summary.json
outputs/paper/skin_reaction_retrieval_diagnostic/agent_random_flips.jsonl
outputs/paper/skin_reaction_retrieval_diagnostic/agent_scaffold_flips.jsonl
outputs/paper/skin_reaction_retrieval_diagnostic/report_trace_examples.csv
```

当前 ChEMBL evidence 版本：

```text
assay screening raw:
  outputs/chembl_tool/tasks/skin_reaction/assay_screening/raw/

assay screening v1:
  outputs/chembl_tool/tasks/skin_reaction/assay_screening/v1/skin_reaction_assay_candidates.csv
  outputs/chembl_tool/tasks/skin_reaction/assay_screening/v1/skin_reaction_activity_evidence.csv
  outputs/chembl_tool/tasks/skin_reaction/assay_screening/v1/skin_reaction_health_check.md

evidence library:
  outputs/chembl_tool/tasks/skin_reaction/evidence_library/skin_reaction_molecule_evidence.jsonl
  outputs/chembl_tool/tasks/skin_reaction/evidence_library/skin_reaction_neighbor_index.pkl
  outputs/chembl_tool/tasks/skin_reaction/evidence_library/skin_reaction_neighbor_index.meta.json
```

当前 v1 screening 状态：

```text
retained assays: 3,648
activity evidence rows: 18,299
index molecules: 8,428
retrieval groups: 21

Tier 1 direct anchors: 326 assays
Tier 2 sensitisation AOP key-event: 303 assays
Tier 3 phototoxicity / irritation / local skin damage: 1,308 assays
Tier 4 skin exposure modifiers: 62 assays
Tier 5 weak/context background: 1,649 assays
```

2026-07-12 的论文视图审计后，Tier 5 继续保留在 ChEMBL source library 中用于 endpoint/data-quality
审计，但不再进入 paper-facing `full_flat` 或 `full_mechanism`，也不再创建独立 LLM reasoning branch。
现有两套 82-sample mechanism run 中，Tier 5 各覆盖 42 个样本并产生 84 次 group call；identity-blind
没有一次判为 useful，deployment-visible 只有一次判为 useful 且方向仍为 `neutral_or_unclear`，合计消耗
约 816k tokens。它主要重复说明 generic cytotoxicity、efficacy 或 target binding 不能决定 Skin label，
没有观察到强正向作用。

该删除只影响 reasoning view，不删除原始 Tier 5 evidence。有效 assay 的 concentration、vehicle、
formulation、duration、light condition、skin model 等 context 必须继续随其所属 Tier 1-4 evidence row
保留。2026-07-12 之前生成的 Skin full-mechanism exploratory runs 含 Tier 5，配置变更后的 run 必须使用
新 batch ID，不能与旧 run 混合断点续跑。

## 当前代码入口和运行状态

主要入口：

```text
constants.py
  label / prediction mapping。当前二分类约定：Y=1 -> risk，Y=0 -> no_risk。

rules.py
  Skin_Reaction assay keyword、negative keyword、context/weak evidence family 配置。

scoring.py
  assay screening / rescore 的保留、剔除和打分入口。

endpoint_groups.py
  Tier.endpoint_group、evidence_direction、evidence_strength 和 endpoint assignment 规则。

experiment_config.py
  Paper-facing direct、full_flat 和 4-family full_mechanism retrieval view；Tier 5 不进入 reasoning view。

screen_assays.py / rescore_outputs.py / summarize_outputs.py / report.py
  thin wrappers，复用 common task workflow 生成候选 assay、activity evidence、health check 和 report。

build_evidence_library.py
  从 v1 assay candidates + activity evidence 构建 molecule-level evidence library 和 neighbor index。

build_starling_evidence_library.py
  从 `data/starling_data/skin_reaction/` 构建 Starling 四-family evidence/index；正式产物写入
  `outputs/paper/molecular_evidence_agent/evidence/skin_reaction_starling_full/`。

  构建命令：
  `python -m tools.chembl_tool.tasks.skin_reaction.build_starling_evidence_library --workers 32 --progress-every 10000`

starling_benchmark.py
  将 direct Skin_Reaction parquet 的 sensitization/contact-allergy records 转成可审计的
  parent-level binary label；不把其它 skin mechanism families 当作 gold outcome。

retrieve_neighbors.py
  旧 native runner 对每个 source-local Tier.endpoint_group 做 analog retrieval。历史 v1 benchmark 使用 top-k-per-group=3、
  min-similarity=0.35；min-similarity=0 的排查显示 index/retrieval 正常，低覆盖主要来自
  chemical-space similarity threshold。

chembl_exact_context.py
  exact-query ChEMBL context wrapper。benchmark 默认不开启，避免 retrospective leakage。

run_reasoning_pipeline.py
  旧 native 单分子 reasoning pipeline：retrieval prefetch、single-molecule branch、endpoint-group 并发 reasoning、
  final summary、trace 保存，以及 --resume-final-from-run-dir final-only rerun。

run_reasoning_batch.py
  批量 reasoning wrapper。复用 common reasoning_batch.py，输出 predictions、metrics、report、logs、
  runs 和 combined trace；支持 --skip-existing 断点续跑。
```

## Starling Tier 1+2 final-only 诊断（2026-07-27）

为检查较远的 Tier 3（phototoxicity/irritation/local damage）和 Tier 4（skin exposure）evidence 是否拖累
sensitization gold label，已对 random/scaffold 的 deployment-visible parent-disjoint
`starling_full_mechanism` 做 post-hoc scope ablation。该条件：

```text
保留：冻结的 single-molecule output、Mechanism.tier_1、Mechanism.tier_2
删除：Mechanism.tier_3、Mechanism.tier_4
重跑：仅 final synthesis
不变：query、retrieval policy、source artifacts、branch outputs、final prompt/schema、model/config
```

公共入口为 `reasoning_batch.py --final-only-source-batch ... --final-only-groups ...`。必须使用
`--final-only-groups` 做 artifact 级过滤；普通 `--groups` 只控制 fresh pipeline 的 group reasoning，
不能替代 resume-final 过滤。当前 batch：

```text
random:
  outputs/paper/molecular_evidence_agent_starling_random/runs_deployment_visible_parent_disjoint/
    skin_reaction/skin_reaction__starling_tier12_final_only/

scaffold:
  outputs/paper/molecular_evidence_agent_starling_scaffold/runs_deployment_visible_parent_disjoint/
    skin_reaction/skin_reaction__starling_tier12_final_only/
```

两套均为 380/380 successful、0 failure。逐样本检查确认 single output 与 source batch byte-identical，
retained group outputs 与 source 中 Tier 1/2 子集完全相同，retrieval/group/trace 无 Tier 3/4 group 泄漏。
结果：

| split | condition | macro-F1 | accuracy | TN / FP / FN / TP |
|---|---|---:|---:|---|
| random | direct | 0.643141 | 0.663158 | 81 / 48 / 80 / 171 |
| random | Tier 1+2 final-only | 0.631003 | 0.652632 | 78 / 51 / 81 / 170 |
| random | full mechanism | 0.629991 | 0.652632 | 77 / 52 / 80 / 171 |
| scaffold | direct | 0.597332 | 0.634211 | 63 / 54 / 85 / 178 |
| scaffold | Tier 1+2 final-only | 0.594785 | 0.626316 | 66 / 51 / 91 / 172 |
| scaffold | full mechanism | 0.583574 | 0.621053 | 61 / 56 / 88 / 175 |

相对 full mechanism，Tier 1+2 的 paired macro-F1 delta 为 random `+0.001013`
（13 better / 13 worse；bootstrap 95% CI `[-0.026550, 0.028667]`）和 scaffold `+0.011211`
（10 better / 8 worse；95% CI `[-0.009740, 0.033671]`）。它说明裁掉 Tier 3/4 在 scaffold 上有小幅
point-estimate recovery，但两套区间均跨 0，且都没有超过 direct；该 test-driven post-hoc 结果只能作为
failure diagnostic，不能当作新的预注册 primary condition。

历史 native v1 全量 test 结果（TDC lineage，不是当前 Starling split）：

```text
batch:
  outputs/chembl_tool/tasks/skin_reaction/reasoning/batches/skin_reaction_calib_50_v1

run settings:
  input=data/processed/Skin_Reaction/test.jsonl
  indices=0-81
  parallelism=3
  group-workers=20
  top-k-per-group=3
  min-similarity=0.35

metrics after idx00074 final-only rerun:
  n_total=82
  n_evaluable=82
  n_successful=82
  n_failed_runs=0
  accuracy=0.682927
  macro-F1=0.678140
  positive precision=0.733333
  positive recall=0.702128
  positive F1=0.717391
  confusion matrix: TN=23 FP=12 FN=14 TP=33
  prediction distribution: no_risk=37 risk=45
```

## Layered normalized-record library (v6)

Skin_Reaction uses the same layered Starling builder as Bioavailability_Ma. The staged driver is shared:

```text
tools/chembl_tool/common/starling/build_normalized_evidence_library.py
```

Everything task-specific is supplied by a `StarlingTaskPolicy` published as `POLICY` in
`tools/chembl_tool/tasks/skin_reaction/starling_policy.py`. The builder and the directory-index loader
resolve it by that convention; there is no separate registry. The task entry point is a thin wrapper:

```bash
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.skin_reaction.build_normalized_starling_evidence_library \
  --workers 128
```

Stages 01-03 are built by the shared normalized-record driver. The split-aware stages 04-09 are built
transactionally by `common/starling/split_downstream.py`, with thin task bindings in
`build_starling_downstream_artifacts.py`:

```text
starling_normalized_v6/
  01_cleaned/{records.parquet, manifest.json, source_inventory.json, endpoint_inventory.json}
  02_normalized/{records.parquet, manifest.json, endpoint_registry.json,
                 record_validity_policy.json, auxiliary_mapping_manifest.json, source_contract.json}
  03_records/{records.parquet, duplicates.parquet, exclusions.parquet,
              scalar_distribution.parquet, manifest.json}
  04_pair_buckets/{pair_bucket_records.parquet, pair_bucket_metadata.json}
  05_assay_transfer_policy/pair_bucket_transfer_policy.json.gz
  06_remove_heldout_overlap/{excluded_direct_skin_reaction_records.parquet,
                             random/records.parquet, scaffold/records.parquet, manifest.json}
  07_molecule_evidence/{random,scaffold}/{molecule_families.parquet,
                                          molecule_family_records.parquet, manifest.json}
  08_neighbor_index/{random,scaffold}/{molecules.parquet, fingerprints.npz,
                                       group_membership.parquet, manifest.json}
  09_audits/
  manifest.json
```

### How the four skin sources differ from Bioavailability_Ma

Two structural differences drive most of the task-specific policy, and neither is incidental.

First, every skin source carries its own structure column, spelled `SMILES` in uppercase, so all four
profiles use `structure_mode="direct"`. There is no shared identifier-to-SMILES mapping and no
separately pinned HuggingFace direct snapshot. Row counts and file digests are pinned from
`data/starling_data/skin_reaction/SOURCE_MANIFEST.json` and asserted on every build.

Second, only two of the four sources carry numeric measurements:

| source | rows | endpoint column | measurement / unit | scalar |
|---|---:|---|---|---|
| `direct_skin_reaction` | 66,597 | `reaction_type` | `effect_metric`, embedded | no |
| `sensitization_aop` | 45,985 | `endpoint_or_target` | `result_value` / `result_unit` | yes |
| `phototoxicity_irritation_local_damage` | 382,726 | `evidence_endpoint` | `observed_effect`, free prose | no |
| `skin_exposure` | 311,834 | `evidence_type` | `result_value` / `result_unit` | yes |

`direct_skin_reaction` and `phototoxicity_irritation_local_damage` are categorical: their outcome lives
in `outcome_label` / `result_label`, and their measurement columns are semi-quantitative scores (`++`,
`+++`) or free prose. The v6 contract keeps such records as retrieval evidence with a **null scalar**,
which meant they could never reach a pair bucket — 296,567 structurally resolved records excluded for
`missing_canonical_unit` alone, a larger pool than the entire measured set. They reach a bucket now, but
only through the explicit encoders below; they are still never coerced into a number by the measurement
parser.

### Categorical response encoding

`starling_categorical_response.py` places the informative subset of the categorical records on a named
latent scale, after which nothing downstream changes: bucket membership, the within-bucket SD
standardisation, the empirical percentile curve and the transfer label all work unmodified.

Five encoders on three scales. Each is confined to one source, and precedence is strict — the first that
matches wins, so a record backed by real counts is never downgraded to an anchor:

| encoder | rows | `canonical_unit` | definition |
|---|---:|---|---|
| `count_logit` | 19,025 | `logit_response` | `η = logit((k + ½)/(n + 1))`, Jeffreys, for `n >= 2` |
| `ordinal_severity_grade` | 3,789 | `ordinal_severity_grade` | the `+`/`++`/`+++`/`++++` ladder, 0-4 |
| `percent_positive_logit` | 2,106 | `logit_response` | `logit` of a reported percentage with no denominator |
| `single_subject_logit` | 12,662 | `logit_response` | the count computation for `n == 1` |
| `signed_direction` | 356,976 | `signed_effect_direction` | `protective −1`, `negative 0`, `positive +1` |

Four decisions here are load-bearing and were made against the data, not by analogy:

**Shrinkage is mandatory, not cosmetic.** 9,308 count rows report `p = 0` and 12,057 report `p = 1`;
an unshrunk logit is infinite for both. The Jeffreys posterior mean also makes the sample size matter
rather than only the ratio: `0/1 → −1.10`, `0/10 → −3.04`, `0/100 → −5.30`.

**The `+` ladder is severity, not incidence.** Of the graded rows that also carry counts, 94.7% have
`n == 1` — the grade says how strongly one subject reacted, not what fraction of a group did. Encoding
`++` as "50% positive" would conflate the two, so it gets its own ordinal scale. Because the transfer
policy standardises by within-bucket SD, the absolute spacing of the ladder is irrelevant; only the
ratios between grades matter.

**A single subject is not an incidence study.** 42.7% of count rows are `n == 1`. `single_subject_logit`
is held apart from `count_logit` so a 1/1 report can never set the comparison scale for a 45/50 one.
It takes only two values (`±1.10`), which the distinct-level gate then handles.

**`protective` is a direction, not a weaker positive.** It is 95,392 phototoxicity rows, and an ordinal
ladder would place it on the wrong side of `negative`. `not_classified` (22,896) and
`mixed_or_inconclusive` (2,336) receive **no value at all** — they are absence of information, and
mapping them to the midpoint would fabricate evidence of no effect. 518 junk rows (LLM output leakage
such as `positivetext>`) are dropped, not repaired.

Three guards keep the encoded records honest:

- `categorical_encoder_id` is part of the pair-bucket key for both categorical sources. `canonical_unit`
  alone is not enough, because three encoders share `logit_response`.
- An encoder only ever fills a record with no scalar of its own, and
  `validate_measurement_pairs` fails the build if an encoding ever overwrote a source measurement the
  parser could have scored.
- Encoded units are validated by `encoded_unit_validity_status`, not by the physical domains — a
  log-odds is legitimately negative, and the physical check would otherwise reject roughly half of them
  as `nonpositive_positive_scalar`.

### Auxiliary context reconciliation

Pair buckets need reconciled `global_context` and `global_species_context`, produced by:

```text
tools/chembl_tool/tasks/skin_reaction/data_processing/
  auxiliary_value_prompts.json                       data-driven prompt registry
  build_embedding_bucket_mapping.py                  MiniLM cluster builder, budget ledger and resume
  study_design_reviewed_mapping.json                 reviewed table, no LLM calls
  auxiliary_mapping_helpers/reconciliation.py        global reconciler; owns MAPPING_VERSION, NULL_LIKE
```

All four sources are covered because categorical encoding lets every source reach a pair bucket:

| source | outputs | planned calls |
|---|---|---:|
| `direct_skin_reaction` | context, species, explicit severity grade | 288 |
| `sensitization_aop` | context, species, reconciled endpoint concept | 200 |
| `phototoxicity_irritation_local_damage` | context, species | 938 actual |
| `skin_exposure` | reviewed study-design context, species | 223 |
| **total** |  | **1,649 actual** |

The runner clusters roughly 100 embedding-neighbour values per ordinary request. The 157,737-value
phototoxicity assay inventory alone uses MiniBatchKMeans with a target and hard request cap of 250;
oversized uneven clusters are split deterministically. Every smaller inventory uses exact Lloyd KMeans.
The recommended run order is the 711-call non-phototoxicity phase, then the 938-call
phototoxicity phase. Each invocation has a 500,000-token guard and a lifetime ledger; status 75 is a
clean resumable budget stop, and completed caches are removed only after final publication.

The accepted prompt lineage is `starling_skin_embedding_bucket_mapping.v3`, currently using standard
OpenAI `gpt-5.4-mini` with `reasoning_effort=low`. The completed direct and sensitization snapshots retain
their earlier `gpt-5.4` provenance and are accepted during mixed-source finalization only through the
explicit compatible-snapshot model flag. Every open-vocabulary request performs neighbourhood-level
reconciliation: it considers all values in the embedding cluster together, reuses the same label for
the same core concept, and minimizes the scientifically defensible label inventory. The earlier v2
row-wise prompt semantics were rejected before publication and its cache must never be resumed into v3.
All species outputs are base species; human occupations, nationalities, ages, and clinical populations
normalize to `human` rather than creating comparison strata.

The completed local pass is preserved separately from the cross-cluster proposal under
`data_processing/auxiliary_reconciliation_v2/`. Its 2026-08-03 review covers all 17,368 non-null labels
in the eight open namespaces across 58 cluster-atomic packets and 218,198 assignments. Three independent
primary/checker/adjudicator rotations accepted 839 of 856 proposed changes and rejected 17; the accepted
changes affect 8,316 assignments. `proposal/FINAL_INTEGRITY_AUDIT.json` passes with no issues and the
runtime-shaped proposal has SHA-256
`5cd011cbd536dd5e720e9e7e42c18619788b54e33ec384f56933ee24bb50c6b5`.
After explicit approval on 2026-08-03, six cleaned-tuple conflicts in phototoxicity context were resolved
to existing reviewed labels and recorded in `auxiliary_reconciliation_v2/PUBLICATION_RECORD.json`.
The published runtime mapping SHA-256 is
`34db0efcd64769699e4b86f7cdcb16e9dc12e466e5f04eb2352eca43440ebd45`.
Stages 02-09 were rebuilt and atomically published: Stage 04 now contains 27,081 pair buckets, including
1,957 with at least 25 records and 1,099 that pass every assay-transfer eligibility gate.

Until that mapping exists, `--allow-missing-auxiliary-mapping` builds stages 01-03 with
`auxiliary_mapping_status=not_available`. Such a build deliberately fails its
`globally_reconciled_auxiliary_coverage` validation so it can never be mistaken for a complete artifact,
and does not attempt pair buckets or later stages. `build_starling_pair_bucket_transfer_policy.py`
also refuses an incomplete mapping: its
`global_context_contract` check compares the recorded `mapping_version` against
`auxiliary_mapping_helpers/reconciliation.MAPPING_VERSION`, and a `not_built` manifest carries
`null`. Do not work around either gate.

### Evidence-catalog family labels

`compact_persisted_records` strips `assay_tier`, `endpoint_group`, `evidence_role` and
`target_pref_name` from the organize-stage artifact because they are derivable. The evidence catalog
therefore takes a `family_resolver` and re-derives them, so a resumed `--from-stage index` build and a
full end-to-end build produce the same catalog. Without it a resumed build silently emits empty labels.

### Downstream rebuilds

```bash
python -m tools.chembl_tool.tasks.skin_reaction.build_starling_pair_bucket_sidecar
python -m tools.chembl_tool.tasks.skin_reaction.build_starling_pair_bucket_transfer_policy
```

A successful `clean`, `normalize`, or `organize` stage invalidates every active downstream artifact.
Stages 04 and 05 are rebuilt from the complete Stage-03 records. Stage 06 then materializes separate
random/scaffold record views; Stages 07 and 08 are built only from those filtered views. The shared
publisher validates the complete candidate tree and atomically replaces Stages 04-09, restoring the
previous tree if publication fails.

### Assay-transfer policy and the heldout filter

Stage 05 follows the same contract Bioavailability_Ma uses: the pair bucket is the only comparison
stratum, eligible buckets need at least 25 records, the raw candidate columns feed an ω² heterogeneity
gate that can disqualify a bucket but never subdivides it, and distance is
`|left − right| / bucket_SD` with a boolean label at 1 SD and a within-bucket empirical percentile.
Skin adds two things to the shared builder:

- `minimum_distinct_levels = 3`. An anchor-encoded bucket can clear the record-count gate while
  carrying almost no spread; `single_subject_logit` in particular takes only two values. A bucket
  rejected for this reason is reported as `fewer_than_minimum_distinct_levels`, not silently dropped.

  A candidate column must never also be a pair-bucket identity field: identity fields are constant
  inside their own bucket, so they can only ever score zero levels and are dead configuration. This is
  why `sensitization_aop` lists neither `aop_event` (a bucket field) nor `assay_type` (the raw input to
  its reconciled context and species). `test_candidate_fields_never_double_as_bucket_identity` pins it.
- Stage 06 filters only `direct_skin_reaction`, separately for random and scaffold. A direct record is
  the benchmark label source and would leak; flux, phototoxicity, and AOP rows remain legitimate assay
  evidence. This filtering is deliberately downstream of the complete-data pair-bucket and transfer
  policy statistics.

The heldout key set is the union of `{random,scaffold}/heldout_molecule_labels.jsonl` — **880 parents**
(490 per split, overlapping by 100), which is the valid+test union, not the 2,456 binary parents in the
benchmark as a whole.

### Soft transfer target

Stage 05 publishes `distance_contract.soft_transfer_target`
(`assay_transfer_soft_probability.v1`) alongside the boolean label:

```text
y = 1 / (1 + exp((d − 1.0) / tau)),  tau = 0.5 / ln(9) = 0.2275980...
```

The midpoint is the boolean decision boundary, so the two never disagree about which side of the
threshold a pair falls on: `y = 0.5` exactly at `d = 1` SD, and `tau → 0` recovers the step. The
temperature is derived rather than tuned — half an SD either side reads as 0.9 and 0.1.

This matters most for the encoded categorical records. A signed direction takes three values and a
severity grade five, so their raw distances are discrete; the sigmoid is where their continuity comes
from. Like every other score in this pipeline, it is **not** a calibrated probability, and the contract
says so explicitly.

### Contract handoff to the assay-transfer repo

TxAgent stops before pair enumeration — `04_pair_buckets/pair_bucket_metadata.json` asserts
`{"pair_enumeration": false, "pair_labels": false, "modeling_dataset": false}`. The modeling target
itself lives in `/data1/joseph/starling_assay_transfer`: `continuous_target = mean_j |y_A − y_Bj|`
normalised as `continuous_target / not_transfer_min` (`ml/starling_ml/data.py`), plus
`transfer_fraction = n_transfer / n_records` (`pipeline/stages/pairs.py`).

What that repo can now consume from here:

- The eligible record set contains categorical records carrying `finite_scalar_value`,
  `canonical_unit ∈ {logit_response, ordinal_severity_grade, signed_effect_direction}`,
  `categorical_encoder_id`, and `categorical_sample_size` where a denominator was reported. The sample
  size is the natural per-record weight — a 1/1 report and a 45/50 report are both single records but
  are not equally informative.
- `soft_transfer_probability` replaces the linear `/ not_transfer_min` normalisation with a bounded
  target that is already commensurable across buckets, since the distance is SD-standardised first.
- `ml/starling_ml/data.py:_binary_to_int_present` already tolerates a `None` label behind a `present`
  mask; that is the existing mechanism for the records the encoders deliberately decline
  (`not_classified`, `mixed_or_inconclusive`, `inconclusive`).

### Tests

```bash
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m pytest tests/chembl_tool/tasks/skin_reaction -q
```

Covers the pinned source contracts, the policy plug-in contract, the catalog family-label round-trip,
endpoint-policy v2, and a bounded end-to-end build asserting that the two qualitative sources produce
zero scalars while the two scalar sources produce canonical units.

## Historical TRIM / DeepSeek properties-only baselines

2026-06-29 跑了 3 个 Intern-S1/TRIM no-retrieval properties-only DeepSeek-v4-pro baseline，
用于和当前 ChEMBL retrieval pipeline 做历史参考比较。它们不使用 TxAgent 当前
`run_reasoning_pipeline.py`，也不使用 ChEMBL/Starling retrieval；prompt 来自
`trim.reasoning.task_user_prompts.render_task_user_message`，tool mode 为 `properties`，
唯一可见工具是 `get_mol_properties_and_fg`。数据 split 使用
`/data1/tianang/Projects/Intern-S1/DataPrepare/TDC_no_conflict_labels_salt_removed/test/Skin_Reaction.jsonl`，
与当前 Skin_Reaction test split 的 82 条样本口径一致。

```text
identity allowed:
  log: /data1/tianang/Projects/Intern-S1/logs/deepseek-v4-pro_properties_Skin_Reaction_DILI_test_20260629_194945.log
  trace: /data1/tianang/Projects/Intern-S1/reasoning-trajectory/deepseek-v4-pro_properties_Skin_Reaction_DILI_test_20260629_194945.log/Skin_Reaction.jsonl
  n=82, failed parses=0
  accuracy=0.7683
  macro-F1=0.7640
  class 0 precision/recall/F1=0.7222/0.7429/0.7324
  class 1 precision/recall/F1=0.8043/0.7872/0.7957
  tool usage: 81/82 questions with tools, avg tools/sample=0.99

strict no identity / no memory comparison:
  log: /data1/tianang/Projects/Intern-S1/logs/deepseek-v4-pro_properties_no_smiles_identity_Skin_Reaction_DILI_test_20260629_195403.log
  trace: /data1/tianang/Projects/Intern-S1/reasoning-trajectory/deepseek-v4-pro_properties_no_smiles_identity_Skin_Reaction_DILI_test_20260629_195403.log/Skin_Reaction.jsonl
  n=82, failed parses=0
  accuracy=0.7439
  macro-F1=0.7408
  class 0 precision/recall/F1=0.6842/0.7429/0.7123
  class 1 precision/recall/F1=0.7955/0.7447/0.7692
  tool usage: 82/82 questions with tools, avg tools/sample=1.00

identity forbidden but memory comparison allowed:
  log: /data1/tianang/Projects/Intern-S1/logs/deepseek-v4-pro_properties_no_smiles_identity_allow_memory_compare_Skin_Reaction_DILI_test_20260629_195652.log
  trace: /data1/tianang/Projects/Intern-S1/reasoning-trajectory/deepseek-v4-pro_properties_no_smiles_identity_allow_memory_compare_Skin_Reaction_DILI_test_20260629_195652.log/Skin_Reaction.jsonl
  n=82, failed parses=0
  accuracy=0.6707
  macro-F1=0.6695
  class 0 precision/recall/F1=0.5952/0.7143/0.6494
  class 1 precision/recall/F1=0.7500/0.6383/0.6897
  tool usage: 82/82 questions with tools, avg tools/sample=1.00
```

Strict vs memory-allowed trace audit:

```text
The two no-identity runs differ on only 8/82 predictions.
Memory-allowed is worse on 7 flips and better on 1 flip.
Main degradation: positive recall drops from 35/47 to 30/47.
Failure mode: memory-allowed often falls back to a narrow "classic direct electrophile only" checklist
and misses pro-hapten / pre-hapten / autoxidation / less-canonical sensitizer mechanisms
such as azlactone/reactive lactone, isothiourea-like reactivity, ortho-quinone-methide formation,
terpene autoxidation, squaric-acid-like dicarbonyl chemistry, and pyrazolone/pro-hapten behavior.
The one useful memory-allowed flip was a nitroaromatic pro-hapten case.
```

当前 v1 规则已主动排除：

```text
generic PubChem/Tox21 Nrf2 assays without HaCaT/keratinocyte/ARE-luc/sensitisation context
EGFR/FGFR/other kinase biochemical assays triggered only by target names such as epidermal/fibroblast
PAF-induced vascular permeability assays misread as skin permeability
anti-inflammatory / dermatology efficacy in reconstructed human epidermis models
permeation-enhancer assays where the tested molecule promotes another compound's transdermal permeation
```

原始 task 定义已经按 source chain 确认为 binary LLNA skin sensitisation。Irritation、phototoxicity、
local damage 和 skin exposure 仍只能作为 mechanistic/context evidence，不能被等同于 gold label。

## Skin_Reaction evidence 原则

Skin reaction 不是一个单一 assay endpoint。ChEMBL 里有价值的 evidence 大致分成三条轴：

```text
hazard axis:
  compound 是否能引发皮肤致敏、刺激、腐蚀、光毒性或局部细胞损伤。

mechanism axis:
  是否命中皮肤致敏 AOP 中的关键事件，例如蛋白共价结合、角质细胞激活、树突状细胞激活。

exposure axis:
  compound 是否能进入、滞留或穿透皮肤，从而让 hazard 有机会表现出来。
```

重要约束：

```text
1. skin permeability / dermal absorption 不是 skin reaction hazard。
   它只能增强或削弱 exposure plausibility，不能单独支持 Y=1。

2. general cytotoxicity 不是 skin reaction hazard。
   非皮肤细胞系的 CC50 / GI50 / viability 只能作为 weak background，不应单独决定 positive。

3. dermatology efficacy 不是 skin reaction hazard。
   抗炎、抗菌、抗银屑病、抗痤疮、melanoma efficacy、skin whitening、wound healing 等治疗性 assay
   不能当作 adverse skin reaction evidence。

4. target binding / enzyme inhibition 通常不是 skin reaction evidence。
   除非 assay description 明确指向 skin sensitisation、irritation、phototoxicity、keratinocyte /
   dendritic-cell activation、haptenation 或 dermal toxicity。

5. validated skin sensitisation AOP assays 的一致阳性 evidence 比单个弱 cytotoxicity assay 更重要。
   DPRA/ADRA/kDPRA、KeratinoSens/LuSens/EpiSensA、h-CLAT/U-SENS/IL-8 Luc/GARDskin 对应不同
   key event；多 key-event 一致时 evidence strength 应上调。
```

## Evidence families and endpoint groups

Skin_Reaction 不应照搬 BBB / Bioavailability 的 tier 语义。这里的 `assay_tier` 应表示 skin-reaction
reasoning 价值，而不是体内/体外距离的简单排序。

建议初版分层：

```text
Tier 1: direct skin reaction anchors
Tier 2: validated skin sensitisation AOP key-event assays
Tier 3: phototoxicity and local skin irritation / corrosion / dermal toxicity
Tier 4: skin exposure, barrier penetration, and local context modifiers
Tier 5: weak or context-dependent background
```

### Tier 1: direct skin reaction anchors

这些是最接近 Skin_Reaction label 的 evidence。命中时应优先保留。

Endpoint groups:

```text
human_patch_or_clinical_skin_reaction
  human patch test
  human maximization test
  repeated insult patch test
  HRIPT / RIPT
  allergic contact dermatitis
  contact allergy
  skin rash / erythema / edema / pruritus when explicitly adverse
  clinical skin reaction / dermatologic adverse event

llna_or_lymph_node_proliferation
  local lymph node assay
  LLNA
  BrdU-ELISA / BrdU-FCM LLNA
  stimulation index / SI
  EC3
  lymph node proliferation after dermal application

guinea_pig_sensitization
  guinea pig maximization test
  GPMT
  Buehler test
  OECD TG 406-like skin sensitization

in_vivo_dermal_irritation_or_toxicity
  dermal irritation
  dermal toxicity
  skin edema
  skin erythema
  Draize skin irritation
  repeat-dose dermal toxicity
```

Direction rules:

```text
supports_skin_reaction_risk:
  positive human patch / HRIPT / clinical skin adverse reaction
  LLNA SI >= 3, positive LLNA call, low EC3 indicating sensitizer potency
  positive GPMT / Buehler / dermal sensitization call
  clear dermal irritation, erythema, edema, necrosis, ulceration, or local toxicity

argues_against_skin_reaction_risk:
  non-sensitizer / non-irritant calls from adequately described validated assays
  LLNA SI < 3 at adequate tested concentrations
  negative human patch / HRIPT only if exposure and concentration are meaningful
```

Evidence strength:

```text
strong:
  human patch / clinical adverse skin reaction with clear positive or negative call
  LLNA with SI / EC3 or explicit positive/negative interpretation

moderate:
  GPMT, Buehler, in vivo dermal irritation/toxicity with clear assay context

weak:
  ambiguous clinical skin terms without dose, route, or adverse-event context
```

### Tier 2: validated skin sensitisation AOP key-event assays

These assays are highly relevant even when they are not direct clinical reactions. They map to the accepted skin
sensitisation adverse outcome pathway: covalent protein binding, keratinocyte activation, and dendritic-cell activation.

Endpoint groups:

```text
protein_binding_or_haptenation
  DPRA
  ADRA
  kDPRA
  direct peptide reactivity assay
  cysteine depletion
  lysine depletion
  peptide depletion
  amino acid derivative reactivity
  haptenation
  protein binding
  covalent binding to protein

keratinocyte_activation_are_nrf2
  KeratinoSens
  LuSens
  EpiSensA
  ARE-Nrf2 luciferase
  antioxidant response element
  Nrf2 activation
  Keap1 / Nrf2
  keratinocyte activation
  HaCaT when assay is explicitly sensitisation-related

dendritic_cell_activation
  h-CLAT
  U-SENS
  IL-8 Luc
  GARDskin
  dendritic cell activation
  THP-1 activation
  U937 activation
  CD54
  CD86
  IL-8 reporter
  genomic allergen rapid detection

glutathione_or_thiol_reactivity
  glutathione depletion
  GSH reactivity
  thiol reactivity
  cysteine adduct formation
```

Direction rules:

```text
sensitization_risk:
  positive DPRA/ADRA/kDPRA, high cysteine/lysine depletion, high peptide reactivity
  positive KeratinoSens/LuSens/EpiSensA, ARE-Nrf2 induction, EC1.5-like activation
  positive h-CLAT/U-SENS/IL-8 Luc/GARDskin, CD54/CD86 upregulation, IL-8 reporter induction
  strong GSH/thiol reactivity when assay explicitly links to sensitisation or electrophilic reactivity

argues_against_skin_reaction_risk:
  negative call in a validated AOP assay, especially if multiple key events are negative
  no peptide depletion / no ARE-Nrf2 activation / no dendritic-cell activation at adequate non-cytotoxic concentrations

context_dependent:
  raw GSH depletion or generic protein binding without skin-sensitisation assay context
```

Evidence strength:

```text
strong:
  validated method with explicit positive/negative call and interpretable concentration-response
  two or more AOP key events agree for the same molecule or close analog

moderate:
  one validated AOP key-event assay with clear positive/negative signal

weak:
  partial mechanistic readout, missing concentration context, or assay only weakly linked to skin sensitisation
```

### Tier 3: phototoxicity, irritation, corrosion, and local skin damage

These assays can support skin-reaction risk, but they answer different questions from allergic sensitisation.

Endpoint groups:

```text
phototoxicity_3t3_nru_or_rhe
  3T3 NRU phototoxicity
  photoirritation factor / PIF
  mean photo effect / MPE
  UVA / UVB photocytotoxicity
  reconstructed human epidermis phototoxicity
  RhE phototoxicity
  phototoxicity / photoallergy / photosafety when assay is adverse

skin_irritation_rhe
  reconstructed human epidermis irritation
  RhE skin irritation
  EpiSkin / EpiDerm / SkinEthic / LabCyte irritation
  MTT viability in skin irritation assay
  OECD TG 439-like assay

skin_corrosion_rhe
  reconstructed human epidermis corrosion
  RhE skin corrosion
  irreversible tissue damage
  necrosis
  OECD TG 431-like assay

keratinocyte_or_skin_cell_cytotoxicity
  keratinocyte viability
  HaCaT viability
  epidermal cell cytotoxicity
  dermal fibroblast cytotoxicity
  skin cell MTT / NRU / LDH release

skin_inflammation_or_barrier_stress
  IL-1 alpha
  IL-6
  IL-8
  TNF alpha
  PGE2
  COX-2
  barrier disruption
  oxidative stress in skin cells
```

Direction rules:

```text
phototoxicity_risk:
  positive 3T3 NRU phototoxicity, high PIF/MPE, cytotoxicity only or much stronger under irradiation
  positive RhE phototoxicity or explicit photosafety concern

irritation_or_corrosion_risk:
  RhE viability below validated irritation/corrosion thresholds
  explicit irritant/corrosive call
  strong local erythema/edema/necrosis in dermal models

local_skin_damage_risk:
  potent keratinocyte or dermal-fibroblast cytotoxicity in a skin-relevant assay
  inflammatory cytokine induction in skin cells with adverse context

context_dependent:
  generic cell viability loss without skin cell type or skin assay context
  anti-inflammatory activity, cytokine inhibition, wound-healing efficacy
```

Evidence strength:

```text
strong:
  validated phototoxicity, RhE irritation, or RhE corrosion assay with explicit positive/negative call

moderate:
  skin-cell cytotoxicity or inflammatory stress with clear skin-relevant cell model and concentration-response

weak:
  cytokine, oxidative stress, or viability readout without clear adverse skin-reaction framing
```

### Tier 4: skin exposure and barrier penetration modifiers

These assays are useful because skin reaction requires local exposure, but they do not define hazard by themselves.

Endpoint groups:

```text
skin_permeability_or_absorption
  skin absorption
  dermal absorption
  percutaneous absorption
  Franz diffusion cell
  diffusion cell
  skin permeation
  skin permeability
  transdermal permeation
  flux
  Jmax
  Kp / logKp
  permeability coefficient
  OECD TG 428-like assay

skin_retention_or_distribution
  skin retention
  epidermis retention
  dermis retention
  stratum corneum retention
  tape stripping
  skin deposition

skin_pampa_or_artificial_membrane
  skin PAMPA
  artificial membrane skin permeability
  silicone / isopropyl myristate skin PAMPA
```

Direction rules:

```text
skin_exposure_support:
  high dermal absorption, high flux, high permeability, strong skin retention
  exposure support can strengthen a hazard signal from Tier 1-3

reduced_skin_exposure:
  low or absent dermal absorption/permeability can weaken but not eliminate hazard concern

context_dependent:
  permeability evidence without any hazard evidence
```

Evidence strength:

```text
moderate:
  validated or well-described skin absorption/permeation assay with quantitative flux/Kp/retention

weak:
  artificial membrane or qualitative permeability call without formulation, dose, or skin model details
```

### Tier 5: weak or context-dependent background

These rows may help the LLM understand analogs, but they should not dominate final prediction.

Endpoint groups:

```text
general_cytotoxicity_context
  CC50
  GI50
  IC50 viability
  LDH release
  apoptosis
  cell proliferation
  non-skin cell viability

immune_or_inflammation_context
  cytokine modulation
  immune-cell activation
  COX / LOX / NF-kB activity
  anti-inflammatory or pro-inflammatory assay without skin context

dermatology_efficacy_context
  anti-acne
  anti-psoriasis
  anti-atopic dermatitis efficacy
  wound healing
  skin whitening
  melanogenesis
  melanoma efficacy
  antimicrobial activity for skin pathogens

target_binding_context
  receptor binding
  enzyme inhibition
  kinase activity
  transporter activity
  target-based pharmacology not explicitly framed as adverse skin reaction
```

Default direction:

```text
context_dependent
```

These rows should usually receive `evidence_strength=weak` or `context_dependent`. They can be retained only when they help
explain a close analog, but they should not be used as primary positive evidence for Skin_Reaction.

## Screening keywords

Initial positive keyword families for `rules.py`:

```text
skin sensitization / sensitisation
skin reaction
contact dermatitis
contact allergy
allergic contact dermatitis
skin allergy
dermal allergy
human patch
patch test
HRIPT
repeated insult patch
maximization test
LLNA
local lymph node
stimulation index
EC3
BrdU-ELISA
BrdU-FCM
guinea pig maximization
Buehler
GPMT

DPRA
ADRA
kDPRA
peptide reactivity
peptide depletion
cysteine depletion
lysine depletion
hapten
haptenation
protein binding
covalent binding
glutathione
GSH
thiol reactivity

KeratinoSens
LuSens
EpiSensA
ARE-Nrf2
Nrf2
Keap1
keratinocyte activation
HaCaT

h-CLAT
U-SENS
IL-8 Luc
GARDskin
dendritic cell activation
THP-1
U937
CD54
CD86
IL-8 reporter

phototoxicity
photoallergy
photoirritation
photosafety
3T3 NRU
PIF
MPE
UVA
UVB

skin irritation
dermal irritation
skin corrosion
dermal corrosion
reconstructed human epidermis
RhE
EpiSkin
EpiDerm
SkinEthic
LabCyte
MTT skin
erythema
edema
necrosis

skin absorption
dermal absorption
percutaneous absorption
skin permeation
skin permeability
transdermal
Franz diffusion
diffusion cell
skin retention
stratum corneum
tape stripping
logKp
Kp
flux
skin PAMPA
```

Negative / exclusion keywords:

```text
melanoma
melanogenesis
tyrosinase inhibition
skin whitening
anti-aging
wrinkle
collagenase
elastase
hair growth
alopecia
sebocyte
acne efficacy
psoriasis efficacy
atopic dermatitis efficacy
eczema treatment
wound healing
antimicrobial
antifungal
antiviral
anti-inflammatory
COX inhibition
LOX inhibition
NF-kB inhibition
cytokine inhibition
topical formulation release only
permeation enhancer assay where the tested molecule is the enhancer vehicle, not the query compound
```

Do not hard-exclude every row containing these terms. If the same description also contains clear adverse skin-reaction terms
such as sensitisation, LLNA, irritation, phototoxicity, or dermal toxicity, keep the row and let `endpoint_groups.py` assign the
more specific group.

## Endpoint assignment requirements

`endpoint_group` must be assigned from the combination of:

```text
assay_tier
standard_type
assay_description
target_pref_name
target_genes
activity_comment
standard_units
```

Do not assign from `standard_type` alone. Examples:

```text
viability
  RhE skin irritation/corrosion context -> skin_irritation_rhe or skin_corrosion_rhe
  HaCaT / keratinocyte context -> keratinocyte_or_skin_cell_cytotoxicity
  generic cancer cell context -> general_cytotoxicity_context

IC50
  phototoxicity +/- irradiation context -> phototoxicity_3t3_nru_or_rhe
  skin-cell viability context -> keratinocyte_or_skin_cell_cytotoxicity
  target inhibition context -> target_binding_context

activity
  h-CLAT / dendritic activation context -> dendritic_cell_activation
  anti-inflammatory efficacy context -> dermatology_efficacy_context
  unclear context -> context_dependent

permeability / flux / Kp
  skin / dermal / Franz / transdermal context -> skin_permeability_or_absorption
  PAMPA skin context -> skin_pampa_or_artificial_membrane
  generic Caco-2 or BBB context -> exclude or context_dependent, not skin evidence
```

## Evidence library row fields

Evidence rows should preserve the same raw ChEMBL fields used by other tasks, plus Skin_Reaction-specific derived fields.

Minimum fields:

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

Skin_Reaction-specific directions:

```text
supports_skin_reaction_risk
argues_against_skin_reaction_risk
sensitization_risk
irritation_or_corrosion_risk
phototoxicity_risk
local_skin_damage_risk
skin_exposure_support
reduced_skin_exposure
context_dependent
unknown_direction
```

Skin_Reaction-specific strengths:

```text
strong
moderate
weak
context_dependent
```

Derived fields such as `endpoint_group_reason`, `evidence_direction`, and `evidence_strength` are for debug and audit. They
should not be sent directly to the reasoning LLM as if they were raw evidence.

## LLM payload rules

The LLM payload should include:

```text
assay_chembl_id
assay_tier
endpoint_group
assay_description
target_pref_name
target_genes
standard_type
standard_relation
standard_value
standard_units
pchembl_value
activity_comment
data_validity_comment
confidence_score
relationship_type
similarity
similarity_bucket
```

Do not include internal rule fields in the LLM evidence rows:

```text
endpoint_group_reason
evidence_direction
evidence_strength
evidence_reason
assay_reason
```

The group-level prompt must make these distinctions explicit:

```text
1. Direct human/LLNA/validated skin reaction evidence can support or oppose final label.
2. AOP key-event evidence supports sensitisation hazard, but one isolated key event is not equal to clinical skin reaction.
3. Phototoxicity is a skin-reaction subtype; it should be separated from allergic sensitisation.
4. Skin irritation/corrosion is local damage evidence; it should not be conflated with immune sensitisation.
5. Skin permeability/retention modifies exposure only; it cannot by itself prove skin reaction risk.
6. Generic cytotoxicity, dermatology efficacy, target binding, and antimicrobial assays are weak context only.
```

## Reasoning schema expectations

Single-molecule branch should focus on:

```text
reactive_or_haptenation_prior
electrophile_or_thiol_reactivity_alerts
skin_permeation_prior
phototoxicity_structural_prior
irritation_or_corrosion_structural_prior
physicochemical_exposure_prior
```

Group-level output should include:

```text
useful_for_skin_reaction_reasoning
transferability
evidence_direction
confidence
reasoning_summary
key_evidence[].effect_on_skin_reaction_reasoning
caveats
```

Final output should include:

```text
skin_reaction_prediction: risk | no_risk
confidence: low | moderate | high
main_evidence_type:
  direct_skin_reaction_anchor
  sensitization_aop
  phototoxicity
  irritation_or_corrosion
  exposure_context_only
  weak_or_no_evidence
key_evidence
conflicting_evidence
caveats
```

Final predictions used for accuracy and macro-F1 must be binary: `risk` or `no_risk`. If the model is uncertain, keep that in
`confidence` and `caveats`, not in the prediction field.

## Exact ChEMBL context

As with the other ChEMBL reasoning tasks, exact-query ChEMBL context can cause retrospective evidence leakage. It must remain
off by default for benchmark runs. Only enable exact-query context for retrospective case studies with an explicit flag such as:

```bash
--enable-chembl-exact-context
```

By default the single-molecule prompt contains no ChEMBL-specific payload or instruction. Only when exact context is enabled and
query exact context is found should the single-molecule payload include `exact_query_chembl_context`, with an instruction to
distinguish direct same-molecule ChEMBL skin-reaction evidence from the physicochemical prior. ChEMBL neighbor evidence still
belongs only in group-level context.

## Initial references used for ontology design

The evidence ontology above follows the regulatory skin-safety assay landscape:

```text
OECD TG 429 / 442B:
  LLNA and non-radioactive LLNA variants for skin sensitisation.

OECD TG 442C:
  DPRA, ADRA, and kDPRA for covalent protein binding / peptide reactivity.

OECD TG 442D:
  KeratinoSens, LuSens, and EpiSensA for keratinocyte activation through ARE-Nrf2-related pathways.

OECD TG 442E:
  h-CLAT, U-SENS, IL-8 Luc, and GARDskin for dendritic-cell activation.

OECD TG 439:
  reconstructed human epidermis skin irritation.

OECD TG 431:
  reconstructed human epidermis skin corrosion.

OECD TG 432 / TG 498 and ICH S10:
  3T3 NRU and RhE phototoxicity / photosafety assessment.

OECD TG 428:
  in vitro skin absorption / dermal absorption.
```
