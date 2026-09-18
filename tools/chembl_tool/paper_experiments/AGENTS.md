# Paper Experiment Operations Conventions

## Active data ownership

Active data lives with its semantic owner: gold-bound data under
`data/gold_labels/<Task>/<version>/`, evidence data under its task/release, and
shared reusable caches under `data/caches/`. `data/artifacts/` is audit-only and
must not be a required build or runtime input; complete retired products belong
under `data/legacy/`. Do not add compatibility symlinks.

The evidence-library pipeline owns scientific level assignment. Preserved
gold-version mappings live under `data/gold_labels/<Task>/level_mappings/<version>/`.
BBB and Bioavailability runtime consumers use the active release-owned
`data/evidence_libraries/<task>/<release>/level_mapping/`; Ames, DILI,
Carcinogens, and Skin keep their gold-owned mappings until reviewed replacements.
Voter membership may validate L1 coverage but must never derive or rewrite levels.
Corrections and publication belong to the evidence-library pipeline and must use
reviewed UID decisions with pinned input hashes.

## Testing discipline

Do not add circular tests that merely assert newly written prompt prose or copy
implementation literals into the test. Prompt wording is validated with reviewed
input/output fixtures or a real pilot/evaluation. Automated tests should cover
executable behavior, failure modes, schemas, rendering validity, and provenance.

This directory maintains the frozen, paper-oriented molecular evidence agent experiment matrix. Orchestration must remain generic. Task-specific endpoint mappings can only be placed in `tools/chembl_tool/tasks/<task>/experiment_config.py`.

## Documentation Responsibilities

```text
ICLR_2027_EXECUTION_PLAN.md
  Project-level master execution plan for submission. Records contributions, experiment numbers, dependencies, resources, gaps, acceptance criteria, timeline, and submission gates.

EXPERIMENT_PLAN.md
  Research questions, run parameters, and statistical protocols for the current frozen experiment matrix. Do not overwrite protocol content with project management status.

VISIBILITY_ANALYSIS.md
  Centralized interpretation of identity-blind and deployment-visible results, including rises, falls, identity audit, metric caveats, and causal validation not yet completed.

RESULTS.md
  Actually measured results and statistical conclusions. Do not write expected results that have not yet been run.

STARLING_BENCHMARK_RESULTS.md
  Central ledger for the current Starling random/scaffold lineage: frozen split/label decisions, formal pipeline, MiniMol head, Morgan KNN, MiniMol embedding cosine KNN, identity-blind progress, Skin retrieval degradation, Tier 1+2 post-hoc, and all entries. It is separate from RESULTS.md, which primarily records the old TDC test/valid matrix, and must not mix tables across lineages.

PIPELINE.md
  Generic pipeline, data flow, visibility discipline, and task mechanism families.

DISTANCE_EXPANSION_DESIGN.md
  E12's D-root/C-family/H1/H2 tree definitions, flat principal curves, tree-node mechanism supplementary, sequential branch reuse, independent code paths, resource control, artifact contract, and regression gates.

ASSAY_LEVEL_RETRIEVAL.md
  Starling context-assay ranking/prefix experiments parallel to group-level; centrally records train-only historical and historical direct-filtered/parent-disjoint, current direct-filtered/scaffold-disjoint contracts, leakage audits, call reuse strategies, run entries, and result status.

FINAL_EVIDENCE_SURFACE_EXPERIMENT.md
  E13 fixed retrieval/single/group final-only surface diagnostics, reproduction commands, metadata census, and no-go conclusions. Card surface is only a reproducible experiment plugin, not part of the default prompt or formal test.

run_train_ratio_prior_experiment.py / train_ratio_prior_analysis.py
  E14's final-only orchestration and offline paired/trigger/provenance audit. Only run BBB/Bio current full-flat valid; must fail if frozen source and train prior do not match; must not run test or upgrade default prompt if promotion gate is not passed.

TRACE_RETENTION.md
  The only directory for final paper traces, retention units, cleanup boundaries, and viewer constraints.
```

When adding a new experiment or obtaining new results: first update the corresponding E-number status in the master execution plan, then run the generator to update machine-readable statistics and `analysis/report.md`, then sync measured results to `RESULTS.md` based on generated artifacts, and finally update interpretation documents such as `VISIBILITY_ANALYSIS.md`. The current generator does not automatically rewrite `RESULTS.md`; do not keep conclusions only in chat.

## Current Scaffold Benchmark Contract (2026-08-28)

All current experiments only read `data/gold_labels/<Task>/v1/scaffold/`. BBB/Bioavailability/Skin train/valid/test are 3,053/397/393, 1,958/262/269, 1,997/246/248 respectively. Parent identity and scaffold pairwise overlap for the three active tasks are all 0. Old molecule-only, gold-vN, selected-vN, and ClinTox source-build paths are only for migration provenance and cannot be new runner defaults. Public paths must be imported from `common/starling/conditioned_benchmark.py`; full contract and hash audit are in `common/starling/CONDITIONED_BENCHMARK.md` and `data/artifacts/gold_labels/conditioned_benchmark/migration_receipt.json`.

```text
builder:
  tools/chembl_tool/tasks/bbb_martins/build_conditioned_source.py
  tools/chembl_tool/tasks/bioavailability_ma/reviewed_context_conditioned_benchmark.py
  tools/chembl_tool/tasks/skin_reaction/context_conditioned_benchmark.py
publisher:
  data/processing/gold_labels/publish_conditioned_benchmark.py
data:
  data/gold_labels/{BBB_Martins,Bioavailability_Ma,Skin_Reaction}/v1/scaffold/
historical v2 held-out indices:
  outputs/paper/molecular_evidence_agent_starling_scaffold_experimental_meaningful_cns_access_v2/
  outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2/
strict first-version reuse:
  tools/chembl_tool/paper_experiments/seed_starling_matrix_reuse.py
paired direct-agent statistics:
  tools/chembl_tool/paper_experiments/analyze_starling_direct_significance.py
current best-agent vs all-baseline paired statistics:
  tools/chembl_tool/paper_experiments/analyze_starling_best_agent_baselines.py
trace and retrieval-family audits:
  tools/chembl_tool/paper_experiments/audit_bbb_retrieval_coverage.py
historical valid-only BBB E16 (no-go; paths frozen for reproducibility):
  tools/chembl_tool/paper_experiments/matched_train_label_agent/bbb_property_compatibility_audit.py
  tools/chembl_tool/paper_experiments/matched_train_label_agent/bbb_property_compatible_contract.py
  tools/chembl_tool/paper_experiments/matched_train_label_agent/bbb_property_compatible_experiment.py
  tools/chembl_tool/paper_experiments/matched_train_label_agent/bbb_property_compatible_report.py
  tools/chembl_tool/paper_experiments/matched_train_label_agent/bbb_property_compatible_trace_diagnosis.py
current diagnostic audits:
  tools/chembl_tool/paper_experiments/audit_skin_reasoning_bottleneck.py
  tools/chembl_tool/paper_experiments/audit_starling_trace_failure_causes.py
```

Old BBB molecule-only builds and selected conditioned candidates are only migration/source provenance for the current cohort; they are no longer promotion candidates or runner inputs. Source-family purity, five-layer results, and unique redraw commands are uniformly recorded in `STARLING_BENCHMARK_PROTOCOL.md` and `ASSAY_LEVEL_RETRIEVAL.md`.

`build_starling_benchmark_indices.py` and `starling_benchmark_matrix.py` must explicitly pass `--benchmark-data-root`, `--benchmark-lineage`, and `--canonical-paper-root` for non-default lineages; model/visibility still use independent `--output-root`. Historical artifacts may only be reused by molecule key; target retrieval must be re-materialized, single requires exactly the same model/reasoning/visibility contract, group also requires the same LLM-visible group input, and final is only copied when the full `retrieval_prompt_hash` is identical and all expected groups have been reused. Reuse audits are written to the target root's `reuse_audit/`, and must not be directly moved by query index.

First-version `record_agreement70_split811_v1` formal results are retained as historical comparison. Previously generated exploratory `record_supported_v1` data, indices, agent runs, and baselines have been deleted; do not restore references in documents or summaries.

## First-Version New Dataset Default Run Contract (frozen 2026-08-01)

This section records the default description of the first version relative to the old TDC/strict-conflict lineage. For `record_agreement70_split811_v1`, the formal default is:

```text
base_url: keys.py:LITELLM_BASE_URL
model: nvidia/GLM-5.2-NVFP4
api_key_env: LITELLM_API_KEY (injected from keys.py by the top-level launcher)
reasoning_effort: "" (omit the parameter; preserve historical provider-default GLM reasoning)
visibility_mode: identity_blind
neighbor_identity_policy: parent_disjoint
operational_staging_used: false
endpoint_concurrency_budget: 512
default launcher shape: one global prompt pool with --parallelism 128
```

`identity_blind + parent_disjoint` is the new main matrix, no longer a supplementary control. It directly retrieves fresh from the valid+test-heldout-filtered index, and the harness prefetches desensitized tool evidence; operational is no longer pre-run, and `reuse_plan.json` is no longer a main matrix input. Existing deployment-visible, operational, matched-prefetch, and reuse results are all retained as historical lineage.

512 is the global outstanding-request budget for the same endpoint. The first 512-concurrency valid stress run had 1 transport timeout in 500 BBB query-only samples; subsequently, 384-concurrency BBB ChEMBL full-flat had 193/500 failures, of which 190 were group request timeouts. Therefore, the current default is reduced to 128 to match the actual throughput of reasoning-enabled long group prompts; 512 is retained only as the endpoint ceiling, no longer the agent launcher default.
By default, only one matrix launcher is allowed to occupy the budget; do not launch multiple split/task launchers in parallel, and do not bypass the global pool via static condition lanes. If the first valid run shows 429, connection/timeout errors, continuously growing service queue, or rising structured-output failure rate, reduce concurrency further; do not change prompt, reasoning, temperature, max tokens, or validation policy due to concurrency adjustments.

The matrix's sole scheduler is the generic global prompt pool. It does not reserve concurrency slots per task or condition, but mixes all ready single/group/final branches under the same `--parallelism` limit; final only enters the ready queue after the single and all expected groups for that sample-condition are `status=ok`. Samples without a recoverable stage artifact must first generate only a retrieval seed via the task pipeline's `--prepare-only`, and must not issue LLM requests in the seed job; the parent scheduler then writes a compatible manifest and enters the shared pool from the first single/group prompt. Cross-condition frozen-single dependencies are dynamically unlocked by query index, and no batch `none` phase barrier is retained. `--max-stage-requeues N` controls the number of immediate stage requeues after built-in validation attempts are exhausted.

This scheduler must maintain the existing artifact contract: canonical `retrieval.json`, single/group/final output, `trace_messages.jsonl`, batch predictions/metrics/report paths and schemas unchanged; only allow adding `scheduler`/`stage_pool` provenance and `.run.lock` in the manifest. Each successful stage must be atomically written; after restart, only recover from missing or failed stages; when any prerequisite branch is rewritten, first invalidate the old final/trace, then dynamically rebuild the final. Final traces still call the task's original `_write_trace_jsonl()` generator. Only one matrix launcher can run on the same root at a time.

Harness-prefetched molecular tools uniformly submit each sample's query properties and all neighbor pair comparisons as a single batch via `ToolServiceClient.invoke_many()`. The tool service's persistent cache key depends only on tool contract/input/version, not on Morgan, MiniMol, coverage-based, or future retrieval features; therefore, replacing retrieval methods only produces new neighbor pairs and does not add another task/retriever-specific tool materializer. The node002 formal service topology and validation entry are in `tools/service/README.md`.

The runner has implemented this target contract: the Starling v4 CLI first runs `valid` by default, uses a single global prompt pool of `identity_blind + parent_disjoint` and `parallelism=128`, and the fresh main matrix does not read operational reuse plans. `valid` artifacts are written to the lineage-named `_valid` root, isolated from the formal test root. `parallelism` exceeding 512 is rejected before launcher startup. The old generic runner's deployment-visible/operational paths are retained only for historical ablation.

Non-default model comparisons must pass explicit `--output-root` to `starling_benchmark_matrix.py` and include the model identifier in the directory name; this parameter only isolates run/manifest artifacts, while the canonical split-specific held-out evidence index is still read from the original Starling lineage root. Different models must not reuse or overwrite each other's `runs_identity_blind_parent_disjoint/`. For example:

```bash
python -m tools.chembl_tool.paper_experiments.starling_benchmark_matrix \
  --benchmark-split scaffold --evaluation-subset valid \
  --model gpt-oss-20b --base-url http://127.0.0.1:9001/v1 \
  --api-key-env GPT_OSS_LOCAL_API_KEY \
  --output-root outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid_gpt_oss_20b
```

## Benchmark Input Lineage and Starling Migration Boundary

Current frozen `test` / `valid` matrix, `RESULTS.md`, and `outputs/paper/molecular_evidence_agent{,_valid}/` all come from pre-migration TDC task splits. They are reproducible historical results and should not be deleted; but they must not be renamed as Starling benchmarks. The existing CLI's `--split test|valid` is still the old data selector and cannot be overloaded as Starling's `random|scaffold`.

The new Starling direct gold benchmark is built from the following public entries:

```text
tools/chembl_tool/paper_experiments/STARLING_BENCHMARK_RESULTS.md
data/processing/gold_labels/build_conditioned_benchmark.py
data/processing/evidence_library/heldout_index.py
tools/chembl_tool/paper_experiments/build_starling_benchmark_indices.py
tools/chembl_tool/paper_experiments/starling_benchmark_matrix.py
tools/chembl_tool/paper_experiments/summarize_starling_benchmark.py
tools/chembl_tool/paper_experiments/plot_starling_benchmark_overview.py
tools/chembl_tool/paper_experiments/plot_starling_model_comparison.py
tools/chembl_tool/common/task_workflows/global_prompt_pool.py
tools/chembl_tool/common/task_workflows/reasoning_stage_runtime.py

data/gold_labels/legacy/processed_starling/<Task>/random/{train,valid,test}.jsonl
data/gold_labels/legacy/processed_starling/<Task>/scaffold/{train,valid,test}.jsonl
```

`plot_starling_model_comparison.py` is the only formal master chart for current Starling model/baseline/ablation. Subsequent full model/visibility summaries use reproducible `--comparison-metrics`; matched method experiments use reproducible `--experiment-metrics`. Both types of experiments must not add one-off overview charts and continue outputting to `starling_benchmark_results_scaffold_valid_gpt_oss_20b_vs_120b/figures/starling_model_comparison.{svg,png}`.
Each comparison summary must use the same task/split/subset/sample count and baseline as the reference. Each experiment TSV must have exactly one `comparison_role=anchor` per task/split (compatible with historical `method=morgan_standard`), and can use `base_method` to point to an existing condition in the master chart's candidate metrics; the anchor's `base_model_label` can further point to any model/visibility series loaded via `--comparison-metrics`, defaulting to the main candidate if not provided. The anchor's sample count, evaluation subset, and macro-F1 must be fully aligned and not duplicated. New method rows can use `plot_label` for short labels; if absent, use `method_label`.

When integrating into the paper runner, the following must be satisfied:

1. The matrix manifest top level explicitly records `benchmark_source=starling` and `random|scaffold`; each experiment records task, `input_jsonl_sha256`, and task provenance reference; top-level `benchmark_provenance` stores protocol, identity normalizer, seed, source revision/metadata, parent agreement policy, and SHA-256 of task/split summary, valid/test inputs, respective label audits, and valid+test union `heldout_molecule_labels.jsonl`;
2. Random and scaffold splits use corresponding `heldout_molecule_labels.jsonl` to build direct-source-filtered evidence views, and must not reuse old indices that did not perform this exclusion; mechanism records are retained and used at query time via `parent_disjoint`;
3. Before running LLM, audit that the valid+test parent identity union and direct gold source have zero overlap, retain counts of non-direct mechanism overlaps, and runtime same-parent exclusion; scaffold splits also require a builder audit preserving zero pairwise overlap among train/valid/test scaffolds;
4. The two Starling splits use different output roots/batch IDs and are isolated from the old TDC test/valid root;
5. Summarizers and charts must partition by benchmark lineage and must not merge TDC, Starling-random, and Starling-scaffold sample-conditions into one metric.

Reference pools have two valid contracts. `train` is the default and excludes valid and test parents using `heldout_molecule_labels.jsonl`. `train_valid` is only a post-selection test sensitivity and excludes test parents using `test_molecule_labels.jsonl`. `--heldout-subsets`, the MiniMol feature builder, and `--reference-pool` must agree; the matrix verifies the recorded held-out scope before model calls.

For v7 evidence, `--heldout-filter-mode direct_source_only` removes evaluation
parents only from the task-declared direct gold source and retains mechanism
records, which query-time `parent_disjoint` retrieval must still filter. The
historical `all_parents` mode removes those parents from every source. Each
paper evidence view owns its Stage 06 records, Stage 07 molecule evidence,
Stage 08 neighbor index, and Stage 09 audits. The matrix reuses the frozen LLM,
prompt, retrieval, tool, and batch pipeline while replacing only benchmark
inputs, the Starling index, and the isolated output root:

```text
outputs/paper/molecular_evidence_agent_starling_random_record_agreement70_split811_v1/
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1/
```

### MiniMol feature agent retrieval

MiniMol agent retrieval is a feature ablation paired with the Morgan/Tanimoto formal pipeline: evidence source,
paper-facing group mapping, `top_k=3`, numeric `min_similarity=0.3`, GLM/prompt/tool settings, and
`identity_blind + parent_disjoint` fresh-run contract all remain unchanged, only candidate ranking is switched to MiniMol v1 embedding
L2-normalized cosine. Cosine cannot use Morgan's structural similarity bucket wording, nor can it use
`query_feature_coverage`, the Morgan-bit selector.

The MiniMol model cannot be loaded on the fly within each reasoning worker. First build a shared registry for the 13 actual base indexes of random/scaffold and six test query sets, materialize the candidate store in index order, then generate lightweight descriptors. The builder
must record checkpoint/config SHA-256 and pass the candidate row order, query coverage, formal MiniMol
`test.pt` cosine parity, and Starling test-parent exclusion gate:

If a legacy evidence candidate's whole-record SMILES can be used for the old index/fingerprint but Graphium cannot graph it,
the builder may only use the `minimol_parseable_parent_or_fragment.v1` fallback for that candidate, and must save each
registry row, original SMILES, actual embedded SMILES, and reason; it must not silently drop rows or fill in fake vectors. Gold test queries
are not allowed to fall back and must still pass cosine parity with the formal MiniMol cache item by item.

```bash
env CUDA_VISIBLE_DEVICES=0 /data1/joseph/miniconda3/condabin/conda run -n txagent-glm python \
  -m tools.chembl_tool.paper_experiments.build_minimol_retrieval_features \
  --splits random scaffold
```

Artifacts are isolated from the Morgan index and Morgan agent runs:

```text
outputs/paper/minimol_retrieval_features_v7_record_agreement70_split811_v1/
outputs/paper/molecular_evidence_agent_starling_random_minimol_retrieval/
outputs/paper/molecular_evidence_agent_starling_scaffold_minimol_retrieval/
outputs/paper/minimol_retrieval_agent_results/
```

The new v4 runs the full `identity_blind + parent_disjoint` condition directly for each split. Query-only `none` is a same-round fresh run,
and its identity policy is marked as not applicable in the manifest; retrieval conditions do not reuse group/final output from Morgan or operational artifacts:

```bash
python -m tools.chembl_tool.paper_experiments.starling_benchmark_matrix \
  --benchmark-split <random|scaffold> \
  --retrieval-feature minimol \
  --visibility-mode identity_blind \
  --neighbor-identity-policy parent_disjoint \
  --parallelism 128

python -m tools.chembl_tool.paper_experiments.summarize_minimol_retrieval_agent

python -m tools.chembl_tool.paper_experiments.plot_minimol_retrieval_agent \
  --png-output outputs/paper/minimol_retrieval_agent_results/figures/minimol_vs_morgan_agent_retrieval_highres.png

# Compare matched parent-disjoint MiniMol agent with Morgan agent / supervised baselines in the same figure
python -m tools.chembl_tool.paper_experiments.plot_starling_with_minimol_agent \
  --png-output outputs/paper/minimol_retrieval_agent_results/figures/starling_benchmark_with_minimol_agent_highres.png
```

After code migration is complete, the long matrix is still chained by a single resumable entry point; that entry point should directly call the fresh parent-disjoint matrix,
summarizer, and plotter, and must not call the operational matrix or parent-disjoint materializer, nor define a second set of experiment logic:

```bash
python -m tools.chembl_tool.paper_experiments.run_minimol_retrieval_agent_experiment
```

`--limit 1` is only for the first end-to-end smoke; the formal summary must cover all 38 retrieval conditions across random/scaffold
and both Morgan/MiniMol must have zero failed samples. This ablation uses an independent paired summary and must not
pass off or overwrite MiniMol/cosine agent rows as canonical Morgan agent rows.

The formal Starling performance bar chart reads random/scaffold, three tasks,
parent-disjoint pipeline conditions, lineage-frozen MiniMol head baseline, Morgan fingerprint KNN, and MiniMol
embedding cosine KNN from the merged `metrics.tsv`. Both KNNs retrieve only from the same split's `train.jsonl`, fix `k=3`, predict by unweighted majority vote,
and compute AUROC using the positive-class neighbor proportion. The main metric in the figure is macro-F1, and the output keeps only the canonical SVG and one high-resolution PNG:

```bash
python -m baselines.structure_knn.run \
  --data-dir data/gold_labels/legacy/processed_starling/<Task>/<random|scaffold> \
  --output-dir outputs/baselines/structure_knn_starling/<Task>/<random|scaffold> \
  --k 3

python -m baselines.minimol.run_embedding_knn \
  --data-dir data/gold_labels/legacy/processed_starling/<Task>/<random|scaffold> \
  --embedding-cache-dir outputs/baselines/minimol_starling/<Task>/<random|scaffold>/embeddings \
  --output-dir outputs/baselines/minimol_embedding_knn_starling/<Task>/<random|scaffold> \
  --k 3

python -m tools.chembl_tool.paper_experiments.summarize_starling_benchmark

python -m tools.chembl_tool.paper_experiments.plot_starling_benchmark_overview \
  --png-output outputs/paper/starling_benchmark_results/figures/starling_benchmark_overview_highres.png
```

The coverage-selector KNN is a diagnostic experiment for the retrieval selector and does not replace the above formal, full-test Morgan KNN baseline.
Its matched strict-threshold setting fixes `k=3` and `min_similarity=0.3`, and runs `similarity` and
`query_feature_coverage` respectively; queries with fewer than 3 qualified train neighbors must be
kept in predictions with `status=insufficient_neighbors` and excluded from supported-cohort metrics; they must not be backfilled with below-threshold
neighbors or guessed with class prior. Results must report both `n_evaluated` / `evaluation_coverage`,
especially since the supported cohort for Bioavailability and scaffold split may cover only a few test samples:

```bash
python -m baselines.structure_knn.run \
  --data-dir data/gold_labels/legacy/processed_starling/<Task>/<random|scaffold> \
  --output-dir outputs/baselines/structure_knn_coverage_starling/<Task>/<random|scaffold>/minsim0p3_supported_k3/<selector> \
  --k 3 \
  --min-similarity 0.3 \
  --neighbor-selector <similarity|query_feature_coverage>
```

### Coverage-selector matched LLM ablation

The coverage-selector LLM ablation fixes Starling full evidence, `full_mechanism`, deployment-visible,
`parent_disjoint`, `top_k_per_group=3`, `min_similarity=0.30`, and the same GLM/prompt, and only compares Morgan similarity
with the `query_feature_coverage` selector. All six task/split conditions are uniformly handled by one canonical summary and plotting entry; the early
BBB scaffold pilot artifact can still be used as one row input, but a separate pilot summarizer or separate overview plotting code is no longer maintained:

```bash
python -m tools.chembl_tool.paper_experiments.summarize_coverage_selector_llm_matrix
python -m tools.chembl_tool.paper_experiments.analyze_coverage_selector_retrieval_changes
python -m tools.chembl_tool.paper_experiments.plot_coverage_selector_llm_matrix \
  --png-output outputs/paper/coverage_selector_llm/analysis/figures/coverage_selector_llm_matrix_highres.png
```

`summarize_coverage_selector_llm_matrix.py` is the only entry point for full-sample paired metrics, McNemar, and paired-bootstrap;
`analyze_coverage_selector_retrieval_changes.py` further audits neighbor set/rank/top-1 changes, Morgan feature
coverage, neighbor redundancy, group reasoning content, and model fluctuation under the same final input. Both share the same
condition/path resolver, and no special cases may be duplicated when adding splits or migrating artifact paths.

Coverage-aware reasoning context is an opt-in ablation orthogonal to the selector:
`--neighbor-context-profile standard|coverage_aware|coverage_mmp_ledger` only changes the additional input to the group branch,
and does not change the neighbor set, raw retrieval, task schema, or final prompt. `coverage_aware` provides anonymous Morgan feature /
query atom-environment marginal coverage; visible-only `coverage_mmp_ledger` reuses the resident
`mmp_structure_compare` per-neighbor MCS/MMP text and uses Morgan marginal feature statistics to organize a set-level ledger.
The latter is prohibited for `identity_blind`; Morgan feature coverage must not be described as atom coverage; without a matched-pair
transformation, specific fragment correspondence must remain unresolved.
Non-default selectors/profiles must be written to an explicit independent `--output-root`; group artifact reuse must also be profile-consistent.
The 2026-08-03 GPT-OSS-120B scaffold-valid, blind+parent-disjoint matched run's three-task point estimates are
BBB `+0.0143` macro-F1, Bioavailability `+0.0022`, Skin `-0.0285`, and all three intervals cross zero, so the conclusion is
mixed / no-go for promotion and does not enter test. Relative to the original Morgan-standard, coverage-aware on BBB is
`+0.0356` macro-F1 with an interval that does not cross zero, but Bioavailability/Skin are inconsistent, so it can only be treated as a task-specific signal.
The full contract audit is located in the analysis root below; formal figures are only appended to the aforementioned
`starling_model_comparison.{svg,png}` master figure, and no single-experiment figures are kept in that analysis root:

```text
outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/
```

The new v4 formal run order is valid `identity_blind + parent_disjoint` fresh-run, valid audit, setting freeze, test
fresh-run, test audit, and summary. `parent_disjoint_ablation.py` is only for the old operational lineage or explicit opt-in
sensitivity and does not participate in the default path. Until the full matrix, failure/leak/identity/held-out audit, and summary are complete,
no formal v4 Starling rerun results exist in `RESULTS.md`.

### Starling identity-blind parent-disjoint main matrix

The new v4 main matrix for Starling random/scaffold uses the same `starling_benchmark_matrix.py` and defaults to
`identity_blind + parent_disjoint`. This regime hides the structure, name, and source ID of query/neighbor, and the harness precomputes
properties/comparison tool text before handing the desensitized evidence and branch output to the LLM. The current claim applies only to this frozen
blind contract; it must not be interpreted as deployment-visible agentic performance. If a visibility comparison is needed later, open an explicit
deployment-visible-prefetched/agentic ablation and do not let it block the main matrix.

Formal entry point:

```bash
python -m tools.chembl_tool.paper_experiments.starling_benchmark_matrix \
  --benchmark-split random \
  --visibility-mode identity_blind \
  --neighbor-identity-policy parent_disjoint \
  --parallelism 128

python -m tools.chembl_tool.paper_experiments.starling_benchmark_matrix \
  --benchmark-split scaffold \
  --visibility-mode identity_blind \
  --neighbor-identity-policy parent_disjoint \
  --parallelism 128
```

`--experiments <condition...>` can be used to select queues by task/condition, but the default order runs sequentially and must not let multiple launchers
each occupy 512 concurrency. The batch runner has built-in `--skip-existing`, so the same command is also the only resume/failed-sample repair
entry point. For selective runs, the matrix manifest uses selection-specific filenames and atomic replacement, and multiple launchers must not
share writes to one manifest. To only rebuild/check the full
canonical manifest without starting conditions, add `--manifest-only` to the above command. Artifacts are fixed to write to:

```text
outputs/paper/molecular_evidence_agent_starling_random_record_agreement70_split811_v1/runs_identity_blind_parent_disjoint/
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1/runs_identity_blind_parent_disjoint/
```

The two splits are summarized separately; blind parent-disjoint must not enter the main bar chart until it fully passes the gate:

```bash
python -m tools.chembl_tool.paper_experiments.summarize_starling_benchmark \
  --splits <random|scaffold> \
  --evaluation-subset <valid|test> \
  --pipeline-root <model-and-subset-specific-v4-root> \
  --model-label <label> \
  --minimol-root <lineage-matched-minimol-root> \
  --structure-knn-root <lineage-matched-morgan-knn-root> \
  --minimol-embedding-knn-root <lineage-matched-minimol-knn-root> \
  --output-dir <model-and-subset-specific-summary-root>
```

`summarize_results.py` only scans old TDC `experiments_for_split(test|valid)` and legacy visibility roots; it must not be used on v4
roots. Empty `analysis_identity_blind_parent_disjoint/report.md` generated by old commands are not canonical v4 results.
An explicit v4/valid summary does not implicitly load historical test baselines; when baselines are needed, the same lineage/subset roots must be passed.

Historical combined v4 has exactly 22 identity-blind conditions per split; after the subsequent BBB experimental-v2 and
Bio/Skin record-v2 split, the expected set must be validated by `lineage × task × declared conditions`. All runs of this generation
still audit `n_failed`, `query_smiles_trace_leaks=0`, `visibility_contract_satisfied=true`, retained parent
conflict=0, and held-out overlap=0. The default completion gate for formal test is still `n_failed=0`. If a single-sample
failure is confirmed non-retryable after retries, the valid diagnosis may keep the failure trace and use the summarizer's `count_as_incorrect_opposite_label` policy to include it in the denominator; the report
must explicitly state the failure count, reason, and policy, and must not call it passing the zero-failure gate.
The v4 summary of `summarize_starling_benchmark.py` must read this blind parent-disjoint main pipeline, the corresponding lineage-frozen
MiniMol head, the formal full-test Morgan KNN, and the MiniMol embedding cosine KNN.

The 2026-07-27 random/scaffold 22-condition blind artifacts belong to the previous strict-conflict split and had 9/5
failed sample-conditions remaining respectively; they are kept only as a historical repair lineage and must not be presented as the current benchmark. The first version of
`record_agreement70_split811_v1` valid results are frozen as historical comparison; the current per-condition status of
Bio/Skin `record_supported_v2` and BBB `experimental_meaningful_cns_access_v2` is recorded in
`STARLING_BENCHMARK_RESULTS.md`. Test must not be started or interpreted for any task until the full valid gate passes; Bio has already
completed one selected-condition formal scaffold-test on 2026-08-10 after passing the gate and freezing settings, and that test must not be used for tuning.

## Code and command entry points

```text
molecular_evidence_agent.py
  Frozen experiment matrix and unified run entry point; responsible for visibility mode, neighbor identity policy, result root, and cross-condition reuse parameters.

router_oof/
  Standalone module for train-only KNN-vs-direct-agent router. v1–v3.1 deployable versions are trained per task, do not share parameters; use
  scaffold-or-parent
  5-fold OOF, fold-specific parent-filtered index, gpt-oss-120b identity-blind parent-disjoint direct agent,
  fixed pre-decision features and nested OOF Logistic/HistGBDT evaluation. v2 removes fold-dependent reference-size
  and deterministic duplicate features, uses dual-risk score and paired-bootstrap promotion gate; v1 `router/` and v2
  `router_v2/` must retain independent lineage. v3 `post_selector_v3/` only learns agent-win probability on disagreement rows,
  uses nested OOF to compare output/query/KNN, full evidence, evidence+structured-trace three profile tiers, and freezes thresholds separately for the two output
  directions; if the train gate fails, deploy strict fallback to KNN. Do not put train folds into the formal
  `starling_benchmark_matrix.py` valid/test enumeration. v3.1 `post_selector_v31/` selects and calibrates a
  fold ensemble for each direction, controlling risk with minimum route count, Wilson precision lower bound, and accuracy-first paired-bootstrap gate;
  valid only reports independent evidence gate, no callback to frozen policy. Subsequent matched-size curve fixes direction
  specs; the only non-deployable shared-transfer termination diagnosis fixes task-balanced Logistic, full generic profile,
  and task-specific calibration, and performs cross-task identity/scaffold exclusion. It still uses agent-win as supervision, not
  counterfactual evidence utility. The current transfer continuation gate has failed; do not continue valid-informed router
  family/profile sweep, and do not start router formal test. Protocol and gates are in
  `ROUTER_OOF_IMPLEMENTATION_PLAN.md`.

summarize_results.py
  Only summarizes old TDC identity-blind, matched-prefetch, and agentic operational conditions, generating coverage, token,
  visibility audit, paired bootstrap, McNemar/Holm, and `analysis/report.md`; does not read v4 run roots.

summarize_starling_benchmark.py
  Summarizes v4 model/split/subset-specific pipeline and explicit lineage-matched baselines; uses recorded
  failure-inclusive policy for failed samples, and prohibits valid summary from silently falling back to historical test baselines.

paired_binary_predictions.py
  Shared binary paired statistics entry for offline audit: unified macro-F1 delta, paired bootstrap, prediction flips, and exact
  McNemar. New deterministic diagnostics must not each duplicate this base statistics; keep independent implementations only when a permutation test with a different hypothesis direction is needed.

watch_glm_tunnel_and_matrix.py
  Monitors GLM endpoint, SSH tunnel, and the only resumable matrix; strict completion count requires task prediction, single/final
  status, expected group count, and group status all valid. Supports passing through `--timeout-s`, but does not save or replay passwords.

analyze_coverage_performance.py
  Only reads deployment-visible predictions, quickly generates coverage/class-conditional coverage, paired macro-F1 improvement relative to same-task
  none, 10,000 bootstrap, machine-readable TSV/JSON, and report paragraphs.

audit_prefetch_contract.py
  Verifies per condition and per sample whether matched-prefetch fully replays identity-blind retrieval and tool outputs.

parent_disjoint_ablation.py
  Historical/explicit sensitivity tool: audits same-parent overlap in operational retrieval, generates selective rerun/whole-reuse/family-branch reuse plans;
  with `--materialize`, only materializes whole-reuse artifacts and reuse plans for inputs that have not changed. Samples with changed inputs are still executed by the unified matrix
  runner with `--neighbor-identity-policy parent_disjoint`.

summarize_parent_disjoint_results.py
  Paired comparison of operational vs parent-disjoint on the full samples of the specified split, and audits identity policy, threshold, and
  reuse provenance. The current entry is reused for test/valid via explicit `--operational-root`, `--parent-disjoint-root`, `--output-dir`.

plot_retrieval_claims_overview.py
  Draws unified horizontal grouped-bar chart from generated analysis TSV; it does not recompute metrics.
  `--data-split {test,valid}` controls split text and conclusions, `--analysis-dir` / `--output` are for partitioned artifacts.

plot_coverage_performance.py
  Draws overall/class-conditional coverage and paired macro-F1 improvement from `coverage_performance.tsv`;
  only for display, does not recompute metrics in the figure. Current code defaults to outputting historical deployment-visible agentic canonical SVG;
  before v4 summary wiring, it must not be used to impersonate blind main results.

paper_figure_style.py
  Shared colors, fonts, basic drawing primitives, vertical numeric grid, standard metrics-to-SVG CLI, and the only
  ImageMagick PNG export helper for paper SVG; new figures must not duplicate visual constants, grid, standard CLI, or `export_png` implementation.

summarize_coverage_selector_llm_matrix.py
  Summarizes Morgan-vs-coverage matched LLM metrics, paired outcomes, McNemar, and
  macro-F1 bootstrap for six Starling task/split; also centrally maintains control/coverage artifact paths.

analyze_coverage_selector_retrieval_changes.py
  On the same paired matrix, audits query/group/slot-level neighbor replacement, top-1/reorder, structural coverage,
  neighbor redundancy, group reasoning changes, and prediction fluctuations under identical final input.

plot_coverage_selector_llm_matrix.py
  Only reads canonical `metrics.tsv` to draw six-condition accuracy/macro-F1 matched bar chart, does not recompute metrics.
```

Common audit order:

```bash
python -m tools.chembl_tool.paper_experiments.summarize_results
python -m tools.chembl_tool.paper_experiments.analyze_coverage_performance
python -m tools.chembl_tool.paper_experiments.audit_prefetch_contract
python -m tools.chembl_tool.paper_experiments.summarize_parent_disjoint_results
python -m tools.chembl_tool.paper_experiments.plot_retrieval_claims_overview
python -m tools.chembl_tool.paper_experiments.plot_coverage_performance
```

Corresponding order for valid diagnostic reruns:

```bash
python -m tools.chembl_tool.paper_experiments.summarize_results --split valid
python -m tools.chembl_tool.paper_experiments.audit_prefetch_contract --split valid
python -m tools.chembl_tool.paper_experiments.summarize_parent_disjoint_results \
  --operational-root outputs/paper/molecular_evidence_agent_valid/runs_deployment_visible \
  --parent-disjoint-root outputs/paper/molecular_evidence_agent_valid/runs_deployment_visible_parent_disjoint \
  --output-dir outputs/paper/molecular_evidence_agent_valid/analysis/parent_disjoint_ablation
python -m tools.chembl_tool.paper_experiments.plot_retrieval_claims_overview \
  --analysis-dir outputs/paper/molecular_evidence_agent_valid/analysis \
  --output outputs/paper/molecular_evidence_agent_valid/analysis/figures/retrieval_claims_overview.svg \
  --png-output outputs/paper/molecular_evidence_agent_valid/analysis/figures/retrieval_claims_overview_highres.png \
  --data-split valid
```

Corresponding regression tests are centralized in `tests/chembl_tool/common/`: identity/policy, retrieval view, replay/hash reuse, prefetch
contract, paper matrix/result summary, parent-disjoint summary, and viewer root must all run with the relevant entry points.

## Language convention

- Use English for internal analysis, reasoning, user-facing explanations, experiment conclusions, Markdown
  documentation, generated reports, and progress updates.
- Commands, paths, model names, experiment IDs, field names, and established technical terms remain unchanged.
- When modifying a report generator, ensure regenerated user-visible reports are produced in English; do not
  rely on manually translating generated artifacts afterward.

The new v4 paper main results table uses identity-blind regime; the following deployment-visible definitions are only for historical results or explicit ablation:

```text
identity_blind:
  query/neighbor structure, name, and source ID are hidden from the LLM;
  harness prefetches properties/comparison tool text;
  paired with parent-disjoint retrieval, as the main regime for v4 none/direct/full_flat/full_mechanism and source comparison.

deployment_visible:
  query exposes only structure, not name;
  retrieved molecule exposes structure, and source name/source ID when provided by the data source;
  LLM autonomously decides whether to call comparison tools;
  only as explicit deployment ablation, not v4 default.

deployment_visible_prefetched:
  Only as a supplementary control for visibility/tool-execution;
  does not enter v4 main results table; existing parity audit and results are retained for provenance/appendix.
```

Main experiments and supplementary controls share the following base conditions:

```text
default endpoint: keys.py:LITELLM_BASE_URL (PARCC LiteLLM)
model: nvidia/GLM-5.2-NVFP4
default API key env: LITELLM_API_KEY (injected from the ignored root keys.py)
reasoning_effort: "" (omit the parameter; preserve the historical GLM reasoning contract)
temperature: 0
max_tokens: 20480
top_k_per_group: 3
min_similarity: 0.30
exact_record_exclusion: true
primary_analog_neighbor_identity_policy: parent_disjoint
visibility_mode: identity_blind
operational_policy_role: opt-in historical/deployment-sensitivity ablation only
endpoint_concurrency_budget: 512
default_launcher_shape: one global prompt pool with parallelism=128, one launcher at a time
```

All paper-facing structural-analog retrieval main results default to `parent_disjoint`: standardized query/source parent,
exclude same parent, then fill top-k with subsequent candidates still above the original similarity threshold. New v4 directly fresh-runs,
no operational staging, diff, whole-run reuse, or branch reuse needed. Here parent is the RDKit FragmentParent
standardization result, not active moiety, and does not infer prodrug or metabolite relationships. Operational can only explicitly opt in to a separate root,
for deployment sensitivity; must not become a dependency of new retrieval conditions.

Parent-disjoint artifacts are uniformly placed in:

```text
# test
outputs/paper/molecular_evidence_agent_starling_random_record_agreement70_split811_v1/runs_identity_blind_parent_disjoint/
outputs/paper/molecular_evidence_agent_starling_random_record_agreement70_split811_v1/analysis_identity_blind_parent_disjoint/

# scaffold
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1/runs_identity_blind_parent_disjoint/
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1/analysis_identity_blind_parent_disjoint/
```

New v4 directly runs the full fresh matrix; `none` runs in the same round but identity policy is not applicable:

```bash
python -m tools.chembl_tool.paper_experiments.molecular_evidence_agent \
  --visibility-mode identity_blind \
  --neighbor-identity-policy parent_disjoint \
  --parallelism 128
```

Final summary directly audits fresh retrieval: all conditions complete, `n_failed_runs == 0`, parent-policy conflict is 0,
held-out overlap is 0, below-threshold fill is 0, identity leak is 0. Old
`summarize_parent_disjoint_results.py` operational/parent paired outputs are only for historical sensitivity analysis.

`full_flat` and `full_mechanism` must come from the same set of selected data source evidence. The only expected difference between them is: the former puts the evidence set in one group, the latter splits evidence by task mechanism.

E12 distance expansion runs both flat primary and mechanism supplementary. Mechanism order is fixed as
`D+C -> D+C+H1 -> D+C+H1+H2`, but internally it is a C-family tree rather than parallel D/C/H1/H2 groups: D is the root, each current
C mechanism family has exactly one aggregate H1 child, each H1 has at most one optional H2 child. H1 condition reuses hash
unchanged D/C family branches, only runs each C family's H1 child then reruns final; H2 condition must use the previous H1
condition as `--group-analysis-source-batch`, reuses D/C/H1, only runs available H2 children then
reruns final again. After adding H2, H1's group ID, neighbors, evidence serialization, and branch hash must remain unchanged;
flat/mechanism same-prefix evidence-row multiset must be exactly identical.

Each C/H1/H2 tree node independently retrieves at most 3 unique molecular neighbors, using the same similarity threshold and identity
policy. Different target/measurement families within an aggregate H1/H2 node share the node's top-3, cannot take 3 per target.
Each extension measurement family must uniquely belong to one C parent; if H2 has a one-hop shortcut to any D/C measured node,
validation must fail and reassign it to the corresponding H1 or exclude it.

E12 is ChEMBL-only throughout, not just H1/H2 using ChEMBL: D/C comes from current ChEMBL direct/full evidence, H1/H2
comes from the same frozen ChEMBL release. Prohibit `Starling D/C + ChEMBL H1/H2` and other cross-source prefixes; distance audit must
verify all layers share the same ChEMBL source manifest, standardized version, quality gate, and provenance contract. Starling is only used for
independent E2/RQ2 matched-scope source comparison.

The following BBB v2 commands and artifacts are retained only as historical retrieval prototype, not as runnable entry points for the current tree ontology:

```bash
python -m tools.chembl_tool.tasks.bbb_martins.build_distance_assay_manifest \
  --chembl-sqlite tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db \
  --export-activities
python -m tools.chembl_tool.tasks.bbb_martins.build_distance_extension_library --workers 128
python -m tools.chembl_tool.paper_experiments.build_distance_index \
  --base-index outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_neighbor_index.pkl \
  --extension-index outputs/chembl_tool/tasks/bbb_martins/distance_expansion/evidence_library/v3/extension/bbb_distance_extension_index.pkl \
  --output-index outputs/chembl_tool/tasks/bbb_martins/distance_expansion/evidence_library/v3/bbb_distance_superset_index.pkl \
  --output-meta outputs/chembl_tool/tasks/bbb_martins/distance_expansion/evidence_library/v3/bbb_distance_superset_index.meta.json \
  --index-version bbb_distance_expansion.v2 --source-release "ChEMBL 36" --workers 128
python -m tools.chembl_tool.paper_experiments.audit_distance_expansion

python -m tools.chembl_tool.paper_experiments.materialize_distance_retrieval
```

BBB v2 excluded node overlap via D/C measured-state census, but later found MMP-3 can directly affect tight-junction integrity,
so its shortest path is 1, making the original `H2 MMP-3` mapping invalid. v2's 392-query audit and coverage only indicate the old prototype's
retrieval engineering nature, cannot be fed into E12 LLM, nor serve as coverage/performance for the new tree ontology. The next version must be rebuilt as:
each BBB C family has one H1 aggregate child, MMP-9/MMP-3 belong to the same passive/barrier H1; other C families also each establish a unique
H1, H2 only when there is no global one-hop shortcut. Independent E12 tree runner, LLM macro-F1 curve, and mechanism execution
are not yet complete.

2026-07-21 added same-molecule causal continuity gate. Correct node-to-node path is still insufficient for release: must prove inference
did not switch from assay molecule A to another unobserved substrate B. In BBB v3 audit, MMP-9/MMP-3 maintain the same barrier
perturbagen, temporarily pass but retain strong scope conditions; NRF2/KEAP1-NRF2 requires the query itself to be an induced ABC transporter
substrate, HIF-1/PHD2 requires the query itself to be a GLUT1 substrate, and the current retrieval contract does not provide these roles. Therefore v3
manifest/index/replay are retained only as engineering artifacts, cannot start formal E12 LLM. All future task releases must have complete
`FamilySelfRelevanceAudit`, and pass
`validate_self_relevance_audit(config, audits, require_publishable=True)`; prompts cannot replace missing role evidence.
BBB detailed records are in `tools/chembl_tool/tasks/bbb_martins/DISTANCE_SELF_RELEVANCE_AUDIT.md`.

Each task must first run the `none` condition. All conditions that include retrieval must use `--single-analysis-source-batch`, to reuse the frozen single-molecule branch in that condition. Do not independently regenerate priors for each ablation condition.

Never write an API-key value to code, persisted commands, manifests, reports, or traces. The default top-level
paper launchers load `LITELLM_BASE_URL` and `LITELLM_API_KEY` from the ignored root `keys.py`, inject only the key
into the child environment, and pass `--api-key-env LITELLM_API_KEY`. Task runners remain environment-only.
Explicit loopback vLLM runs retain the non-sensitive `GLM_LOCAL_API_KEY` placeholder behavior. The historical
`zai-org/GLM-5.2-FP8` LiteLLM request alias resolved to `hosted_vllm/nvidia/GLM-5.2-NVFP4`; endpoint comparisons
must not be presented as FP8-versus-NVFP4 model comparisons. `watch_glm_tunnel_and_matrix.py` remains specific
to the explicit loopback SSH-tunnel route and is not the default LiteLLM launcher.

The OpenAI SDK transport-layer retry limit is explicitly fixed at 2, consistent with the baseline run. Structured JSON validation allows at most 4 total attempts in the recorded call path. The last two attempts only re-serialize the same JSON evidence to avoid provider deterministic degradation; must not drop evidence or change inference settings. Runs with incomplete single, group, or final branches are excluded by the batch completeness constraint and explicitly re-run via batch `--skip-existing`.

The group prompt also has provider transport protection. Payloads that after cleanup do not exceed 750 KB remain unchanged. Above that threshold, each over-limit neighbor retains at most 100 evidence rows sampled at deterministic equal intervals; the prompt must record the original row count, retained row count, original byte count, threshold, and sampling method. This mechanism avoids endpoint-layer HTTP 413 failures without introducing data-source or task-specific rules.

Before reporting a run:

1. Must satisfy `n_failed_runs == 0`; otherwise use `--skip-existing` to re-run failed samples.
2. Must satisfy the `query_smiles_trace_leaks == 0` output of `summarize_results.py`.
3. The v4 main experiment `identity_blind` must pass the identity-blinding preflight and use the de-identified group output for final synthesis;
   `group_reasoning_outputs_raw.jsonl` is for audit only and must never be used as final input. Also audit per retrieval
   parent conflict=0, held-out overlap=0, and threshold violation=0.
4. `deployment_visible` / `deployment_visible_prefetched` run only in explicit visibility ablations. Prefetched
   conditions must replay per query the frozen `retrieval.json` and prefetched tool outputs of the corresponding `identity_blind` run,
   and must not re-retrieve or re-execute potentially non-deterministic MCS according to the current task config. The historical matched-prefetch coverage gate
   remains for old lineage; it does not block the new v4 blind main matrix.
5. When reporting performance, must also report retrieval coverage; missing neighbors are a real property of the data source/index and must not silently drop corresponding samples.
6. Use the shared summarizer to generate within- and cross-regime paired test set comparisons, bootstrap intervals, and exact McNemar tests.
7. Scalar KNN results must be reported separately from LLM agent conditions; it is a numerical direct-F control and does not belong to any LLM visibility regime.

Old lineage run outputs are in `outputs/paper/molecular_evidence_agent/`: `runs_deployment_visible/`, `runs/`,
and `runs_deployment_visible_prefetched/` are the historical agentic, blind, and matched-prefetch artifacts, respectively.
New v4 main outputs are in `runs_identity_blind_parent_disjoint/` of each record-agreement lineage root.
These are reproducible artifacts, not source code.

## Visualization and artifact cleanup

The paper performance visualizations uniformly use the horizontal grouped-bar design of `plot_retrieval_claims_overview.py`.
Subsequent test/valid or new split performance plots should extend this entry point and not keep a separate overview plotting code in parallel.
Each split's `analysis/figures/` retains only two official artifacts:

```text
retrieval_claims_overview.svg
retrieval_claims_overview_highres.png
```

`preview`, `qa`, `pre_parent_disjoint`, and other overview files replaced by this figure must not remain in the official
figures directory. The SVG is the reproducible source figure; `--png-output` is the only
raster version exported via ImageMagick for viewing convenience.

## Valid diagnostic matrix record

From 2026-07-22 to 2026-07-23, the valid Identity-blind, matched-prefetch, and deployment-visible
conditions were extended to 26 conditions each, with 2,713 sample-conditions per set, all with zero failures; the prefetch audit was
2,713/2,713, with no missing, extra, or mismatch. The parent-disjoint 22 retrieval conditions totaled 2,275
sample-conditions, with zero failures, and final neighbor identity conflicts and below-threshold fill-ins were both zero.
The Bioavailability scalar KNN macro-F1 on valid was 0.6621. Detailed values and
interpretations must be consulted in `RESULTS.md` and `outputs/paper/molecular_evidence_agent_valid/analysis/`.

The normalized analysis of the Skin Reaction valid visibility trace is in
`SKIN_REACTION_VISIBILITY_TRACE_AUDIT.md`; the shareable bilingual static report and its artifact/source/config are in
`reports/skin_reaction_visibility_trace_casebook/`. The language buttons of the portable report must be injected via the generic
`inject_portable_report_language_toggle.py`, with report-specific copy in `language_config.json`,
and must not copy dedicated titles, descriptions, or two-language logic into one-off scripts.

Paper runners must save each sample's own `trace_messages.jsonl` and pass `--no-combine-traces`, avoiding regenerating
condition-level duplicate large files. The final trace viewer by default registers Starling random/scaffold and historical TDC test;
the new v4 dataset service serves `runs_identity_blind_parent_disjoint/`, and the historical dataset continues to serve four legacy roots:
`runs/`, `runs_deployment_visible_prefetched/`, `runs_deployment_visible/`, and
`runs_deployment_visible_parent_disjoint/`, locating per-run traces via `predictions.jsonl`,
and must not merge metrics across datasets. Must not re-add hardcoded adaptations for old tasks, Tiers, expert policies, or task-specific
prediction fields. Parent-disjoint must still be audited via the summary artifacts of `analysis/parent_disjoint_ablation/`;
the viewer must also clearly display the `parent_disjoint` policy from the manifest and `reuse.json`, and whether a sample was re-run due to
retrieval changes or reused because the LLM-visible input hash did not change.
