# Molecular-evidence agent paper experiments

This directory contains the maintained paper-facing experiment layer. Task
science stays in `tools/chembl_tool/tasks/<task>/`; retrieval, visibility,
reasoning, validation, checkpointing, analysis, and plotting are shared.

DILI/Carcinogens 的当前数据合同和全部 source-build 入口见
[`NEW_TASK_SOURCE_DATA.md`](../common/starling/NEW_TASK_SOURCE_DATA.md)。二者已接入统一
Stage-03 恢复、catalog/index 重建和普通 Parquet level 导出；无需单独维护 runner 或画图程序。
当前 benchmark 均为 Starling-only gold_v4，清理后的检索源统一冻结在 `retrieval_final`。
最新已完成诊断为 DILI v5 与 Carcinogens R18；结果、baseline 和图均从
`current_conditioned_results.json` 的对应 suite 读取。旧混合标签、旧 gold 和清理中间轮次
只作历史参考。此次源整合没有新增模型结果，已完成预测保留其实际输入和复用凭据。

## Paper scope

The retained paper matrix has six result families and two visibility controls:

| Question | Retrieval source | Evidence organization | Maintained runner |
|---|---|---|---|
| Does literature-scale Starling retrieval outperform ChEMBL retrieval? | ChEMBL versus Starling | matched direct, full-flat, and full-mechanism views | `starling_benchmark_matrix.py` |
| What happens as progressively more distant evidence becomes available all at once? | Starling | one-shot cumulative full-flat levels | `run_conditioned_assay_family_curve.py` |
| Can the model revise an earlier decision when new indirect evidence is appended? | Starling | append-only progressive state and evidence | `run_conditioned_assay_progressive_curve.py` |
| Does mechanism grouping help relative to one flat evidence branch? | ChEMBL or Starling | full-flat versus full-mechanism over the same evidence | `starling_benchmark_matrix.py` |
| Does evaluation difficulty depend on structural generalization? | matched source and reasoning setup | scaffold versus parent-grouped random split | the same runner with the split flag |
| Does molecular identity visibility change behavior? | matched retrieval payload | `identity_blind` versus `deployment_visible_prefetched` | shared visibility contract and replay |

`deployment_visible` remains available for end-to-end agentic deployment
experiments. The paper-facing matched visibility comparison uses
`deployment_visible_prefetched`, which restores identities/structures while
replaying the blind retrieval and tool payload.

The historical one-shot family curve is identity-blind. The matched full-flat
control and append-only progressive curve both use deployment-visible-prefetched
inputs; they compare reasoning organization over identical cumulative cards and
tools. Neither comparison is a matched visibility ablation. Blind and visible
settings remain supported by the source/reasoning matrix.

The existing `plot_assay_retrieval_curve.py --conditioned-progressive-config-comparison`
accepts both matched full-flat and progressive roots through repeated
`--conditioned-progressive-config-task-root CONFIG:TASK=PATH` arguments. The frozen
prompt profiles may differ only after cumulative prepared-input, selection, and
retrieval hashes match. Add `--performance-only` for one three-task Macro-F1 row
with None and the five right-side baselines; omit it to retain resource panels.
Use `--conditioned-baseline-root` to select baselines from the same evaluation subset.

## Canonical benchmark

All active evaluation data lives at:

```text
data/conditioned_benchmark/<Task>/{scaffold,random}/
```

The two split schemes contain the same molecule-condition rows and labels:

- `scaffold`: parent- and Bemis-Murcko-scaffold-disjoint;
- `random`: exact parent-grouped 80/10/10 allocation; parent-disjoint but not
  scaffold-disjoint, with all conditions represented in all three splits.

The benchmark creation code, task-specific voting boundaries, condition and
parent construction, multi-vote held-out priority, complete lexicographic split
objectives, deletion-repair behavior, leakage policy, and required audit files
are frozen in:

```text
tools/chembl_tool/common/starling/CONDITIONED_BENCHMARK.md
data/conditioned_benchmark/manifest.json
data/conditioned_benchmark/migration_receipt.json
data/conditioned_benchmark/random_split_receipt.json
```

Historical molecule-only, `selected_vN`, task gold-vN, and source-build names
are provenance only. Runners and plots must use the canonical roots above.

## Maintained entrypoints

### Benchmark and evidence construction

```text
tools/chembl_tool/paper_experiments/rebuild_current_starling_retrieval.py
tools/chembl_tool/paper_experiments/export_current_starling_level_records.py
tools/chembl_tool/common/starling/publish_conditioned_benchmark.py
tools/chembl_tool/common/starling/build_conditioned_random_split.py
tools/chembl_tool/common/starling/build_record_supported_benchmark.py
tools/chembl_tool/paper_experiments/build_starling_benchmark_indices.py
tools/chembl_tool/paper_experiments/build_assay_family_catalog.py
tools/chembl_tool/paper_experiments/build_conditioned_source_family_purity.py
tools/chembl_tool/paper_experiments/build_bbb_source_family_purity.py
tools/chembl_tool/paper_experiments/build_bioavailability_vote_pure_source.py
```

`CURRENT_STARLING_RETRIEVAL.md` 是当前 Starling records、purity overlays、family catalogs 和
scaffold/random indices 的唯一操作说明。当前 retrieval 只冻结并读取实验实际使用的最新 Stage-03
canonical snapshot；统一入口负责
恢复、重建和逐哈希验证，不依赖个人 checkout。该文档也区分静态 record membership、split-specific
indexed representative cards 和 per-query model-visible cards，并说明 GitHub 协作数据。旧版本号只作为
run/receipt provenance 保留。

当前 family naming contract 是 `source_group_id -> family_key -> level`。新配置只通过
`common.experiment_retrieval.evidence_family()` 声明 canonical `family_key` 与 source-native groups；冻结
artifact 中的 `family_id`/`endpoint_group` 仅由兼容边界保留，跨 ChEMBL/Starling 对齐必须使用
`family_key`，不能使用 level number 或 `Mechanism.tier_N`。跨 source 对照应直接以 `family_key` join，
同时保留各 source 自己的 level。

Task-specific voter and source-review entrypoints are indexed in
`CONDITIONED_BENCHMARK.md`; they are intentionally not duplicated here.

### Model experiments

```text
tools/chembl_tool/paper_experiments/molecular_evidence_agent.py
tools/chembl_tool/paper_experiments/starling_benchmark_matrix.py
tools/chembl_tool/paper_experiments/run_conditioned_assay_family_curve.py
tools/chembl_tool/paper_experiments/run_conditioned_assay_progressive_curve.py
```

### Analysis and figures

```text
tools/chembl_tool/paper_experiments/summarize_starling_benchmark.py
tools/chembl_tool/paper_experiments/analyze_starling_direct_significance.py
tools/chembl_tool/paper_experiments/analyze_starling_best_agent_baselines.py
tools/chembl_tool/paper_experiments/analyze_progressive_trace_adoption.py
tools/chembl_tool/paper_experiments/plot_starling_model_comparison.py
tools/chembl_tool/paper_experiments/plot_assay_retrieval_curve.py
```

`plot_starling_model_comparison.py` is the single model/source/visibility
comparison figure entrypoint. `plot_assay_retrieval_curve.py` is the single
level-curve/resource figure entrypoint. Do not add a one-off plot module for a
new model row or a new progressive rerun.

### Baselines

```text
baselines/minimol/run_train_cv.py
baselines/minimol/run_embedding_knn.py
baselines/conditioned_knn.py
baselines/structure_knn/run.py
```

Condition-aware MiniMol is the paper-facing trained baseline whenever a task
has accepted condition groups. The two KNN families retain both
same-condition-then-null and unrestricted-train variants.

## Shared contracts

- Retrieval source, evidence organization, split, visibility, model, prompt,
  input, index, and family-catalog identities are independent manifest axes.
- A stored prediction may be reused only when all LLM-visible semantic hashes
  match. Directory names or matching molecule counts are insufficient.
- `none` may reuse a compatible frozen single/final state. Evidence-equivalent
  levels may carry forward without a model call.
- The current cumulative-family/progressive evaluation uses
  `scaffold_disjoint` for scaffold and `parent_disjoint` for random. The
  source/reasoning matrix exposes its identity policy explicitly in the
  manifest; a split name alone never silently changes an old matrix contract.
  Every policy prohibits the query parent itself.
- Evaluation indices remove valid+test benchmark-defining direct rows while
  preserving eligible non-direct mechanism evidence from held-out parents.
- Morgan retrieval uses the configured `0.30` minimum similarity. The current
  progressive protocol also applies its monatomic-element collision guard.
- Identity-blind prompts remove query/neighbor structures, identifiers, and
  source names. Visible-prefetched prompts restore them while holding retrieval
  and tool outputs fixed.
- LLM outputs pass shared JSON validation. Validation retries and provider
  failover are recorded in the trace; they do not alter label policy.
- The scaffold-valid progressive record-card budget ablation uses only the
  2/1, 4/2, and 8/4 roots and receipts registered in
  `current_conditioned_results.json`. BBB has current strict-voter-L1 results
  for all three budgets. Bioavailability 8/4 has two exact-contract full-curve
  runs, so the combined figure reports their mean and observed min-max range.
  Skin source-purity v5 applies the strict target-scope gate to gold and
  L1/L2/L3. A final reproducibility rebuild added 37 MDAM nonvoter outcomes to
  L2 without changing the benchmark split; this changed the index-selected
  surface for 1, 2, and 3 validation queries under 2/1, 4/2, and 8/4,
  respectively. The completed agent roots are retained as last-complete
  references pending targeted replay. The matched MiniMol/Morgan baselines
  remain current because their train/validation rows did not change.
- The ablation remains on the shared progressive runner: card limits are CLI
  parameters (`--initial-card-limit`, `--delta-card-limit`), while molecule
  quotas, selection order, prompt profile, and append-only semantics are fixed.
  Multi-configuration figures remain in the shared assay-curve plotter. Repeat
  the same `CONFIG:TASK=PATH` argument with a distinct path only for a verified
  exact-contract replicate; omitted cells are recorded explicitly. A
  selected-surface receipt is required whenever retrieval lineage hashes differ.
  The saved combined PNG/SVG, TSVs, and summary are under the following root;
  its Skin cells are stale until the targeted replay completes:
  `outputs/paper/analysis/progressive_record_card_budget_2_1_4_2_8_4/`.

Detailed group-level data flow is in [PIPELINE.md](PIPELINE.md). The cumulative
level and progressive protocols are in
[ASSAY_LEVEL_RETRIEVAL.md](ASSAY_LEVEL_RETRIEVAL.md).

## Current artifact registry

The only machine-readable index for paper-facing result roots and freshness is:

```text
tools/chembl_tool/paper_experiments/current_conditioned_results.json
```

It distinguishes:

- `current`: benchmark and retrieval hashes match;
- `current_via_zero_change_retrieval_receipt`: a split-scoped audit proves that
  every model-visible selected retrieval surface is unchanged across a
  lineage-only source/index repair;
- `last_complete_reference_pending_*`: a complete historical reference is
  retained because the corresponding current-condition matrix is incomplete;
- `stale_after_*_requires_*_replay`: a retained run is complete, but a later
  source/index repair changed model-visible retrieval for a bounded query set;
- `stale_retrieval_index_requires_targeted_replay`: labels may still match, but
  changed retrieval inputs prohibit publishing the old score as current.

At the 2026-09-05 snapshot, BBB scaffold-valid has fresh 2/1, 4/2, and 8/4
runs over the strict-voter-L1 v6 index; its random v6 index is built but still
requires LLM replay. Skin source-purity v5 has current matched trained/KNN
baselines, but the final MDAM L2 rebuild leaves 1/2/3 scaffold-validation
queries pending targeted replay for 2/1, 4/2, and 8/4. Its random agent cell
still requires full replay. The exact affected prefixes are frozen in
`receipts/skin_scaffold_valid_mdam_family_rebuild.json`.

Bioavailability scaffold-valid has fresh 2/1 and exact 8/4 replay
results. Its retained 4/2 curve is current through the nitrendipine
selected-surface zero-change receipt. The rejected L1-only 0.7148 run used a
different prompt contract and is neither a replicate nor a variance estimate.
Bioavailability random remains unaudited and requires a separate change audit
or replay. Its scaffold-test baselines have been retrained after the train-row
correction; older valid baseline scores remain pre-fix references.

The progressive runner accepts `--evaluation-subset test` with an explicit
output root and a complete test-specific `--single-source-root`. It uses the
same valid+test-heldout-filtered index as validation, records the selected
subset, and rejects cross-subset resume. Freeze the validation protocol before
running test. `--parallelism` is the shared pool size;
`--parallelism-per-task` optionally caps each task within that pool. The default
endpoint budget remains 512; an authorized override must be explicit in
`--endpoint-concurrency-budget`. Task caps are enforced before submission so
capped tasks do not occupy workers needed by other ready queries.

Both methods' three-task scaffold-test 4/2 curves have three complete runs,
including the original as replicate 1; all five matched baselines are complete.
Results, figures, tool audits, and the BBB/Skin L2→L3 trace analysis are registered
in `current_conditioned_results.json` and summarized in `RESULTS.md`. Earlier
valid/random tool outputs have not been audited against the serialized MolGpKa
correction; source-lineage compatibility alone does not establish tool equivalence.

For a controlled one-shot comparison, the existing family runner accepts
`--matched-progressive-root PATH --output-root NEW_PATH [--prepare-only]`.
It inherits the source split/subset and exact cumulative cards/tools, validates
current lineage, and judges every level independently without previous decisions.
Frozen inputs are copied atomically; altered destination inputs abort resume
instead of being overwritten alongside existing predictions.
After a source repair, a single matched run can add
`--level-reuse-source-root OLD_RUN`. The shared reuse gate requires the same
organization, model, generation settings, benchmark and priors, then compares
every prepared model input. Only unchanged prefixes are copied; the first
changed level and its successors run again. Each repetition must point to its
own old run. This option cannot be combined with `--replicate-ids`.
This visible matched control is distinct from the historical identity-blind,
per-assay full-flat curve; see `ASSAY_LEVEL_RETRIEVAL.md` for the contract.

Progressive and matched full-flat retry failed queries automatically after a
cooldown, skipping successful level checkpoints. Defaults are three extra rounds
(`--max-stage-requeues 3`) with 60-second initial backoff (`--retry-delay-s 60`),
doubling up to 900 seconds. Read `execution_status.json` on demand; exhausted
retries end as `needs_attention` with failure history retained. Restarting the
same command resumes checkpoints; no Codex polling loop is required.

For frozen-input repeats, use the same family runner with
`--matched-organizations progressive full_flat --replicate-ids 2 3`.
The suite runs the four jobs sequentially, with all selected tasks sharing each
job's prompt budget. It creates `replicate_02/{progressive,full_flat}/` and
`replicate_03/{progressive,full_flat}/`; source level predictions are never copied.
The original completed run remains replicate 1. None/query priors and tools stay
frozen, so these repeats measure level-reasoning variability conditional on those inputs.
To overlap complete runs, set `--concurrent-runs N`; each run uses `--parallelism`
slots, and their product must fit `--endpoint-concurrency-budget`. Provider-pool
capacity must also fit each run's slot count. The default remains sequential.

For a different model, add `--refresh-query-priors` to a matched suite. The same
runner generates fresh single/None branches under `single_cache/` using the
source's exact retrieval and prefetched tools, then replaces only the query prior,
None result and source index in the frozen level inputs. Both organizations share
the new model's priors. Source and resulting prepared hashes are recorded separately;
completed priors are hash-checked on resume. The prior stage shares the global and
per-task query caps, with sequential single/final calls and the legacy branch JSON
validation policy; six-way racing applies to subsequent level retries.
Tasks using `conditioned_query_priors.v1`, including Ames, refresh both branches
through the shared fresh-prior runtime, with the configured failure-only racing.
Only frozen tool retrievals are copied; old-model single/None predictions are not.
`--provider-pool-config PATH` forwards routing options without changing prompts and
freezes the public provider configuration in the suite manifest. Credentials are
read from `--env-file` (default `.env`) and are never written to receipts.
The one-run V4 Pro 0813 test suite is registered as
`replicate_suites.scaffold_test_pro_0813_4_2`; its `launch_receipt` contains the
exact command and `provider_pool.json` contains the
price-first routing policy and price caps. Resume with that command.

To plot repeats, register each run with the existing plotter's repeated
`--conditioned-progressive-config-task-root CONFIG:TASK=PATH` option and select
`--replicate-interval sd` for arithmetic mean ±1 sample SD (`ddof=1`) of
run-level scores. At least two runs per curve point are required. The default
remains observed min-max (`range`). Neither interval is a confidence interval;
fixed None and baseline references remain single points. The completed three-run
scaffold-test comparison, including all levels and baselines, is registered under
`replicate_suites.scaffold_test_4_2.comparison_analysis`.
The plotter compares actual prepared-input hashes across every replicate, not
only configuration/index hashes. Configuration comparisons read unique task/run
artifacts with at most eight workers, then aggregate in the supplied order;
all completeness, input-equivalence and lineage checks still apply.
The registered `comparison_receipt` stores the
exact plotting command, individual run scores and verification results; the
suite's `launch_receipt` stores the run command. Use these receipts for exact
reproduction rather than creating another launcher or duplicating root lists.

Ames uses the same level-curve plotter: add
`--conditioned-progressive-config-task-root CONFIG:ames=PATH` and
`--conditioned-progressive-baseline-root ames=outputs/baselines/ames_scaffold_valid_v2`.
`--plot-tasks bbb_martins bioavailability_ma skin_reaction ames` fixes the four-column
order and leaves unavailable current task curves explicitly empty. The progressive
overview also accepts Ames task roots and scales its width with the column count.
All five Ames v2 baselines cover 274 rows: condition-first KNN fills remaining
slots from unrestricted train molecules. Historical partial-coverage scores can
still be omitted with `--omit-mismatched-progressive-baselines`.
`--omit-baseline-tasks TASK` keeps None but excludes unavailable or
stale baselines for that task. The current four-column valid figure and exact
generation command are registered under
`result_families.record_card_budget_ablation.ames_valid_extension`.

To overlay different models in the same level/baseline figure, use the existing
configuration comparison with `--allow-model-comparison`. This additionally
requires identical prepared evidence/tools, card selection and retrieval lineage;
only model-derived priors/None may differ. Full-flat/progressive within each model
must still share exact prepared inputs, including priors. None references are checked within
each model and plotted separately, while matched baselines appear once.
`--replicate-interval sd_if_repeated` shows sample SD for repeated configurations
and no interval for single runs; `sd` still rejects single-run configurations.
The completed Flash (three runs) versus Pro 0813 (one run) figure is registered in
`replicate_suites.scaffold_test_pro_0813_4_2.comparison_analysis`, with its exact
command, source checks and PNG/SVG hashes in the associated figure receipt.
This compact comparison (PNG/SVG, metric/reference TSVs, summary and receipt)
is tracked in Git for GitHub review. Raw run directories remain local artifacts.

`--retry-race-width 6` makes JSON repair calls and failed-level requeues race six
identical requests. The first fully validated response wins; remaining async HTTP
requests are cancelled and cleaned up before the caller publishes a single output.
Every request counts against both the global and per-task caps. SDK transport
retries must be zero. Full race receipts live in each level's `retry_races/`,
including invalid responses, failures, and cancellation outcomes; server-side
abort propagation is endpoint-dependent. Retained-output token metrics exclude
discarded/cancelled attempts; their available usage is in these receipts.

`suite_manifest.json` freezes the repeat plan; `suite_status.json` identifies the
active root, whose `execution_status.json` has query progress. An exclusive
`launcher.lock` prevents duplicate suite launches. Exhausted failures remain
`needs_attention`, while later suite jobs still run. No periodic Codex monitor is used.

Source-repair diagnostics use `--record-review-overlay` before cumulative
selection. Hash-bound `correct` decisions may repair endpoint/measurement/context
fields while preserving raw support, molecule identity and family membership;
corrected cards receive new content-addressed IDs. `--query-structure-map` accepts
an input-hash-bound `verified_query_structure.v1` map for DILI/Carcinogens. It uses
verified source-parent structures consistently for retrieval, tools and reasoning
while retaining frozen benchmark rows and leakage identities. Pass the same map
to fresh-prior preparation and evidence preparation; stale prior structures fail
closed. Task-specific rules under `--evidence-grounding` are pinned in manifests
and preserve the existing progressive update mechanism. The Carcinogens R5
Valid/Test repair and its source-only decisions are linked from the result registry.

Ames fresh single/None preparation is available through the progressive runner
with `--tasks ames --fresh-query-priors` and an explicit output root. Task-local
`query_prior.py` owns its scientific prompt; tools, validation and concurrency
use the shared implementations. Completed compatible priors are resumable.
Changing `--parallelism` or `--endpoint-concurrency-budget` preserves compatible
checkpoints and records the previous settings in `execution_history`; model,
prompt, output-limit, and dataset changes still reject reuse.
The Ames full-flat control requires `--matched-progressive-root`; it does not
use the historical unpaired family-curve branch.

The family runner owns matched full-flat and repeat-suite CLI orchestration;
the progressive runner owns their shared query queue, retry rounds and summary.
`common/reasoning_race.py` is an internal adapter for cancellable first-valid
races through `openai_provider_pool.py` and `openai_reasoning_client.py`.
`reasoning_validation.py` remains the single validator. There is no separate
retry or replicate executable.

## Documentation map

- [Conditioned benchmark contract](../common/starling/CONDITIONED_BENCHMARK.md)
- [Current Starling data, rebuild, and sharing](CURRENT_STARLING_RETRIEVAL.md)
- [Pipeline and visibility](PIPELINE.md)
- [Assay/family-level and progressive retrieval](ASSAY_LEVEL_RETRIEVAL.md)
- [Current result status](RESULTS.md)
- [Trace retention](TRACE_RETENTION.md)
- [Current progressive trace viewer](../../trace_viewer/AGENTS.md)
- [MiniMol and KNN baselines](../../../baselines/minimol/README.md)
- [ClinTox source contract](../tasks/clintox/CLINTOX_BENCHMARK.md)

Historical experiments that are not part of the paper scope must not be linked
as active entrypoints from this page. Git history and explicitly retained
receipts provide provenance for removed no-go launchers and transient smoke
runs.
