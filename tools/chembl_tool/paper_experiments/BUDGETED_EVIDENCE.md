# Direct + budgeted indirect evidence

## Status and ownership

**Joint/Quality paused by user on 2026-09-17.** Offline extraction, further source/quality
repairs and Joint experiments must not resume without a new explicit request.
The current checkpoint and verified stop are recorded in
[`receipts/joint_pause_review_20260917.json`](receipts/joint_pause_review_20260917.json).
Joint is implemented; its current-revision prediction comparison is not complete.
The six simple full-flat baselines are authorized on scaffold validation and test;
Quality and Joint remain deferred. Run status is registered separately from results.

| Owner | Responsibility |
|---|---|
| `common/record_diversity.py`, `record_numeric_features.py` | Source-bound assessment schema and feature encoding |
| `prepare_record_diversity.py`, `run_record_diversity_extraction.py` | Offline input inventory and resumable annotation; no inference runner |
| `common/record_selection_context.py`, `record_feature_selection.py` | Shared condition/rule features and retained molecule-local selectors |
| `common/budgeted_evidence.py` | Q, normalized features and the shared-pool selectors |
| `common/budgeted_retrieval.py` | Direct/indirect retrieval, identity exclusions and exact source/cache binding |
| `common/budgeted_reasoning.py` | Evidence payload, visibility and single-prediction contract |
| `common/flat_prompt_alignment.py` | upstream task mapping, versioned exact-field display deduplication and legacy rendering |
| `budgeted_suite.py` | Six-setting global request queue, streamed preparation and independent retry scheduling |
| `common/budgeted_preparation.py` | Lazy per-worker index and exact source-bound candidate cache across simple settings |
| `paper_experiments/budgeted_evidence.py` | Preparation and orchestration through the existing family CLI |

Do not create a second provider client, tool-prefetch layer, inference launcher
or plotter. Suite-local audits under `outputs/` are provenance, not maintained APIs.
The aligned-scaling code/results review is recorded in
[`receipts/aligned_scaling_maintenance_20260918.json`](receipts/aligned_scaling_maintenance_20260918.json).
The subsequent TDC/mixed-ML/prompt review, resume checks and bounded cleanup are
recorded in [`receipts/tdc_mixed_prompt_maintenance_20260919.json`](receipts/tdc_mixed_prompt_maintenance_20260919.json).
The Direct-3 and current validation-selected ML review is recorded in
[`receipts/tdc_baseline_maintenance_20260919.json`](receipts/tdc_baseline_maintenance_20260919.json).

## Baseline handoff

<a id="adopted-naive-llm-baseline-direct-neighbor-fill"></a>
### Adopted naive LLM baseline: Direct and Indirect Neighbor-fill

2026-10-13 follow-up: `--indirect-identity-policy scaffold_disjoint` enables
query-relative scaffold exclusion before candidate ranking/backfill, using the
existing task identity contracts. The default remains `parent_disjoint`.
The policy is pinned in the experiment manifest and candidate cache binding.
DeepSeek and Luna completed the user-requested Conditioned/TDC six-task test
comparison (4,273 predictions / 12 cells each), registered as
`indirect_scaffold_disjoint` and `indirect_scaffold_disjoint_luna`.
Their [results and matched-reference protocols](RESULTS.md#indirect-scaffold-disjoint-follow-up-2026-10-13)
preserve original Direct records, priors, prompt adapters and generation settings.
This excludes the query's scaffold, not the global valid/test scaffold union;
empty scaffolds and DILI/Carcinogens use the existing task identity contracts.
`--record-rendering full_fields` preserves adapter-v1 reference cells; the default
remains adapter-v2 exact-field deduplication. Historical scores are unchanged.

The user clarified on 2026-09-22 that **both Direct and Indirect use
Neighbor-fill** in the naive LLM baseline. The fixed-budget comparison is:

- **Direct:** fill 50 Direct records.
- **Direct + Indirect:** retain the same 50 Direct records and fill 50 Indirect records.

For each evidence pool, retrieve records from similarity-ordered molecules,
use deterministic seeded card order within each molecule, and fill the record
budget without a per-parent record cap. Each setting makes one full-flat LLM
prediction with matched query condition, cached single-molecule analysis,
prompt/model and retrieval identity contract. The Direct-only row is one arm
of this comparison; the naive designation also includes Direct + Indirect.

Current retrieval uses up to 128 candidate molecules per evidence pool, with
similarity cutoff zero for these baselines. The completed aligned suites reached
their configured record budgets. This observed result is separate from the
selector's general behavior: if eligible candidates run out, it reports the
shortfall rather than duplicating cards. The older 2026-09-17 tool-visible suite
retained a 10-molecule Direct limit and did have Direct shortfalls; see registry
`baseline_no_threshold.record_count_audit`.

Report scaling budgets explicitly: the original val-selected Direct Neighbor-fill
budget is 50; the additional fixed-Direct test comparison uses 100 and is paired
with `direct_fill_b100`. Preserve all completed 25/50/100 scaling points and their
saved predictions. Registry `naive_llm_baseline` owns the designation and references.
The [scaling inventory](RESULTS.md#aligned-scaling-record-inventory) links all
three completed suites, their scores and verification receipts.

The completed Gemini 3.8 Flash implementation of this naive baseline is registered
as `gemini38_flash_api_b50`: None / Direct 50 / Direct 50 + Indirect 50 across both
benchmarks, including current seed-42 TDC Skin/Carcinogens. Its
[baseline record](../../../outputs/paper/gemini38_flash_api_b50_20260925/README.md)
links all 36 scores and frozen input/generation/reuse contracts. Additional
provenance and cost checks are deferred at user request; figure integration is
recorded separately from baseline designation.

Original paper figures use registry `llm_vs_standard_baselines`: DeepSeek/Luna/Gemini
Neighbor-fill 50/50, supplied Ours scores and five grouped ML columns. The separate
`indirect_scaffold_disjoint_luna.figures` variant includes DeepSeek/Luna scaffold
results and omits Ours Direct+Indirect and Gemini naive Direct+Indirect.
Gemini scaffold inference has only been cost-estimated, not run. Plotting
contracts and commands live in [Current comparison figures](README.md#current-comparison-figures).
Direct-3/tool-visible figures remain historical; the complete ML appendix keeps
all variants. Test maxima are display upper bounds, separate from the original
models' validation selection and fixed-k=3 KNN (threshold 0.5).

### Conditioned standard baselines complete (2026-09-21)

All six frozen MiniMol heads now follow the same validation-selected protocol
as the TDC heads. BBB/Oral reuse their adopted frozen controls; the other four
tasks add 20 seeds. The [current table](../../../outputs/paper/conditioned_minimol_val_selected_20260921/RESULTS.md)
contains all 30 head/KNN cells; registry `conditioned_minimol_val_selected` pins
verification and commands. All KNN variants use fixed k=3. Earlier CV heads
are historical; encoder/indirect finetuning is reported separately below. Reuse the maintained MiniMol/KNN CLIs.

### Final adopted MiniMol + indirect assay (six tasks, 2026-09-22)

All six conditioned scaffold tasks now have frozen, main-only encoder and
indirect-assay encoder controls: 18 ensembles, with 80 new training seeds for
Skin/Ames/DILI/Carcinogens and unchanged existing frozen/BBB/Oral results.
Checkpoint and auxiliary-weight selection use validation BCE; ensemble thresholds
use validation Macro-F1. Auxiliary data excludes the entire validation/test
parent/connectivity union, while shared scaffolds remain allowed. LLM indirect
excludes each query parent at retrieval time; these rules do not imply identical
numeric/text evidence exposure. The separate TDC mixed-label control retains
its stricter scaffold exclusion.

Use the [MiniMol protocol and generic CLI](../../../baselines/minimol/README.md#minimol--indirect-assay).
The [six-task results](../../../outputs/paper/minimol_indirect_assay_20260922/RESULTS.md)
and source-suite receipts are owned by `budgeted_evidence_protocol.minimol_indirect_assay`.
Experiment-local scripts record completed runs; they are not task entrypoints.

### TDC encoder/indirect completion (2026-09-22)

Six tasks and both unchanged main training pools now have frozen, main-only
encoder and indirect-assay controls (36 cells). Auxiliary exclusion uses the
TDC valid/test parent/connectivity union, permitting shared scaffolds. This is
separate from the stricter exclusion for added Starling main labels. Original
TDC main overlap is preserved. All checkpoint/λ/threshold selection uses TDC val;
see [results](../../../outputs/paper/tdc_minimol_indirect_assay_20260922/RESULTS.md),
registry `tdc_minimol_indirect_assay`, and the shared MiniMol CLI documentation.

### TDC MiniMol external-validation selection (2026-09-19)

The user replaced train-only CV with the existing external-validation head path.
Registry `tdc_minimol_val_selected` records the original suite; official task
parameters and both training pools remain unchanged. Train five seeds on train
only, select each checkpoint by TDC valid BCE within 25 epochs, calibrate the
ensemble probability threshold by TDC valid Macro-F1, then evaluate test. No CV
or train+valid refit; KNN is unchanged. Use the maintained head runner with
`--threshold-strategy valid_macro_f1`, omitting `--train-all`. Mixed data retains
`--condition-field model_condition`. Commands and validation/test audit outputs
are under `outputs/paper/tdc_minimol_val_selected_20260919/`.
The original 12 heads are verified; Skin/Carcinogens cells and that root's
combined LLM table are now historical. Current reporting uses
`llm_vs_standard_baselines` with seed-42 Skin/Carcinogens ML and matched naive
LLM successors. See [current sources and export commands](README.md#appendix-result-sources).
The older `export_comparison.py` only reproduces its frozen historical suite.

### Historical TDC MiniMol SWEEP_RESULTS with CV (2026-09-19)

Registry `tdc_minimol_official_sweep` records the user-requested task-specific
MiniMol hyperparameters for both TDC-only and mixed ML pools. Use the matching
upstream `SWEEP_RESULTS` entry, falling back to Bioavailability for Skin and
Carcinogens. The maintained CV/head entrypoints and train-only epoch/OOF-threshold
selection remain unchanged. Only BBB, Ames and DILI change configuration;
other heads and all KNN are reused after verification. Frozen commands, source
hash and results live in `outputs/paper/tdc_minimol_sweep_results_20260919/`.
See `baselines/minimol/README.md` for the parameter table. This profile does not
change LLM retrieval or prior historical ML scores.

### TDC matched Morgan KNN Direct-3 (2026-09-19)

The separately authorized control uses the same six TDC test cohorts. Add
`--direct-knn-root outputs/paper/tdc_scaffold_baselines_b50_20260918/ml
--direct-budget 3` to the existing family `--baseline-suite` with the TDC manifest.
It runs `direct_knn`, `both_fill`, `both_cap`, `both_group` (four settings;
8,900 final predictions). Direct-fill/cap are identical at one label per neighbor
and are not duplicated. None is referenced only after byte-identical current
messages have been verified against the existing None run.

Direct replays `<root>/<task>/morgan_knn/all_train_unique_molecules/test_predictions.jsonl`:
three train structures, labels, similarities and rank order, validated against
hash-bound TDC train rows. It includes no Starling L1 cards and applies no extra
parent/scaffold filtering. Query labels and KNN predictions never enter context.
Two existing same-normalized-parent cases (BBB #382, Ames #1146) are retained to
match KNN; report the corresponding excluded-query sensitivity separately.
These are zero-based indices. Do not describe Direct as parent-disjoint.

Indirect retains the previous L2+ parent-disjoint candidate pool and the exact
selected records for each strategy: B=50, candidate molecules=128, minimum
similarity=0, seed=0, cap=5 where applicable. Its molecules are not restricted to
the three Direct neighbors. Direct structures retain the exact KNN representation;
external records keep their own source identity. Distinct source identities are
not silently merged into one structure. The cached prior, task semantics, model,
generation settings and local output schema remain fixed. New record contexts use
the offline-verified upstream adapter v2 deduplicated rendering.

The implementation extends `common/tdc_evidence.py` and the existing budgeted
runner/suite; there is no new maintained runner. Command, verification receipts,
status and completed comparison points live under
`outputs/paper/tdc_knn3_baselines_20260919/`, registered as
`budgeted_evidence_protocol.tdc_knn3_baselines`. `verify.py --neighbors-only`
checks all train neighbors and None reuse; `verify.py` checks prepared messages
and unchanged Indirect; `verify.py --final` verifies predictions and scores.
The latter is an experiment receipt, not a maintained entrypoint. Direct-3 versus
Morgan KNN is a matched-neighbor comparison; Direct+Indirect additionally consumes
external evidence. Existing TDC-only ML predictions are reused, not retrained.

### TDC scaffold test profile (2026-09-18)

The user authorized six fixed-budget settings (Direct 50, Indirect 50), with
the same five ML report names (three unique methods without conditions).
TDC is an explicit external benchmark, separate from the conditioned results.
Pass `--benchmark-manifest data/tdc_benchmark/benchmark_manifest.json` to the
existing family `--baseline-suite`; retain `--evaluation-subset test`,
`--direct-budget 50 --indirect-budget 50 --min-similarity 0` and cap 5.
The manifest pins inputs, train labels, heldout identities, source indices and
catalogs. `common/tdc_benchmark.py` owns profile validation and the TDC-only
Ames/Carcinogens prompt adaptation. Conditioned task contracts are unchanged.

`common/tdc_evidence.py` merges train-label records with Starling records by
normalized parent before the shared 128-molecule candidate limit. Raw molecular
forms and source IDs remain on records. Train labels consume the same Direct
budget/cap, with no source priority. Direct excludes the full valid+test parent
union and the current query scaffold; there is no global heldout-scaffold
exclusion. Indirect retains its existing parent-disjoint and task source gates.

`prepare_cached_single_analysis.py` prepares only the shared structure/property
analysis through existing tools, native single prompts and the shared provider
queue. It emits a hash-bound `single_analysis_cache.v1` bundle accepted by
`--cached-prior-sources`; no intermediate final/None prediction is generated.
Fresh final predictions still use the ordinary suite. Reuse requires matching
task/profile/contract, exact structure and condition, input hash and cache hash.

Frozen data/provenance: `data/tdc_benchmark/data_manifest.json` and
`data/tdc_benchmark/upstream/acquisition.json`. BBB/Bioavailability/Ames/DILI
use ADMET_Group fixed test files; Skin/Carcinogens use the official scaffold
splitter with seed 1, fractions .7/.1/.2. Raw TDC split membership and labels
are retained; stricter parent overlaps are recorded and excluded from Direct.
Run/progress: `outputs/paper/tdc_scaffold_baselines_b50_20260918/PROGRESS.md`.
Reviewed configuration and reproducible acquisition/index commands:
`outputs/paper/tdc_scaffold_baselines_b50_review_20260918/`.
The run root contains `launch_command.json`, ML command receipts and
`ml_verification.json`; cached single analyses are under `priors/`, final suite
artifacts under `llm/`. The fixed test counts are BBB 406, Bioavailability 128,
Skin 82, Ames 1,457, DILI 96 and Carcinogens 56 (2,225 rows total).
ML preserves the original TDC train files, including the official BBB/Ames
train/test parent overlap; the stricter Direct filter removes those parents.
BBB's retained prompt targets meaningful CNS access, which may differ from the TDC
BBB label. These external data/endpoint differences must accompany comparisons.

### Simple mixed-training ML control (2026-09-19)

All five ML baselines also evaluated the same TDC test rows using TDC train plus
filtered current Starling scaffold train. This is an explicit exception for the
added ML supervision: exclude the union of TDC valid/test parents and raw or
normalized scaffolds, including empty scaffolds. It does not change the LLM
retrieval profile above. Mixed rows have equal weight. Current heads use TDC
external validation as specified above; five-fold mixed scaffold CV describes
only the original historical head run. MLP uses
source-namespaced condition features; KNN keeps scientific conditions and the
existing unique-molecule policies. Full protocol and reused model entrypoints:
[`baselines/minimol/README.md`](../../../baselines/minimol/README.md#tdc-test-with-added-current-starling-labels).
Registry `budgeted_evidence_protocol.tdc_starling_mixed_baselines` links the
completed data, commands, verification and results; experiment-local scripts
are reproduction receipts, not new maintained APIs.

### New prompt alignment (2026-09-18)

New budgeted runs default to `--budgeted-prompt full_flat_v5`. The pinned upstream
is upstream commit `3c36c2640ada12119f4a74a9bdf8511b0d7b8f6e`,
`predict/harnesses/branches/prompts/full_flat_context_v5/`.
`common/flat_prompt_alignment.py` and its three small prompt assets own the
adaptation; task science for Ames, Skin, DILI and Carcinogens remains in their
existing `experiment_config.py`. BBB/Bio use the upstream scientific prose,
with Bio's pass/fail mapped to our high/low labels.

The local output contract is unchanged: task prediction, confidence, supportive/
contradictory/basis card lists, claims, gaps and summary. It does not add molecule
citation fields, upstream's cross-role molecule exclusivity, or mandatory nonempty
claims/references. Existing record-role validation remains unchanged. Source
records pass through `minimal_evidence.v1` before text rendering. The new input
shows molecule blocks, record aliases and source level/family provenance.

Since 2026-09-19, newly prepared local inputs use adapter
`full_flat_v5_adapted.v2` and payload marker
`protocol.record_rendering=exact_duplicate_fields.v1`. Named endpoint/result/
condition/support fields are printed once; `Additional details` retains all
other fields and any raw/normalized disagreement. Only type-and-value-identical
copies at explicitly mapped paths are removed. Unknown fields, empty values,
source structures, qualifiers and multiple source examples remain intact; no
summarization or text truncation occurs. The canonical audit payload is unchanged
apart from the rendering marker. This is preservation relative to the existing
sanitized evidence contract, not a claim that upstream extraction/normalization
retains every source field: e.g. the pre-existing PMID/DOI visibility policy is
unchanged. Old payloads without the marker still render exactly as before.
The original exported upstream bundle and completed predictions remain versioned
historical artifacts. Rendering verification and before/after examples are in
`outputs/paper/prompt_rendering_dedup_20260919/`; no new inference was launched.

Main reasoning receives **cached single-molecule analysis + selected records**,
with no raw query or analog tool output and no new tool-service calls. This prior
is not the None final prediction: only explicit single-analysis fields are
projected. The obsolete `exact_chembl_evidence_assessment` field is excluded from
the local input and portable template; current retrieval uses Starling, not
ChEMBL. Historical caches remain unchanged. None makes a fresh one-call prediction over the same cached prior
and query/condition, with zero retrieved records. Cached prior generation is
separate historical computation, not a second inference stage launched here.

Pass `--cached-prior-sources FILE`, a JSON mapping each task to a frozen
progressive root with `experiment_manifest.json` and level-1 prepared snapshots.
Lookup checks the source input hash, exact query SMILES + condition, and snapshot
hash. Missing/mismatched priors fail explicitly; no positional fallback or silent
generation. Use a fresh output root; old predictions are not a prompt-aligned
result. `prompt_payload` is an audit sidecar: analysis re-renders it and compares
the actual messages before trusting its card/condition mappings.

Validated validation-source mapping and rendered examples are in
`outputs/paper/prompt_alignment_full_flat_v5_20260918/`.
Add the following to the suite command below for the new configuration:

```text
--budgeted-prompt full_flat_v5
--cached-prior-sources outputs/paper/prompt_alignment_full_flat_v5_20260918/prior_sources.valid.json
```

For collaborator integration, use the checked-in
[shared prompt bundle](shared/README.md#task-prompts). Apply `tasks.yaml` and
`provenance.json` together; this is the maintained handoff, not a byte-identical
copy of our runtime prompt. The local progressive and aligned full-flat paths
retain native labels. The historical portable exporter emits the superseded
pass/fail contract and must not be used to regenerate the maintained bundle.
Label-encoding observations and the limits of the offline validation are in the
[DILI handoff](shared/DILI_HANDOFF.md) and
[other-task trace audit](shared/LABEL_ENCODING_AUDIT.md).
Provider/reasoning settings remain independent of prompt alignment.

### Aligned validation record scaling (2026-09-18)

Run the existing family CLI with `--baseline-suite --baseline-record-scaling`.
The common candidate pool is 128 molecules for **both** Direct and Indirect;
the former hidden Direct limit of 10 molecules is removed. Molecule-cap still
limits each parent to five records. Similarity cutoff is zero; all indirect
retrieval is parent-disjoint. Direct retains scaffold-disjoint retrieval on
scaffold splits. Shortfalls are recorded, never padded by duplicate cards.

1. Run shared None and both Direct selectors at 25, 50 and 100 records.
2. Select a separate Direct budget for each selector using equal-task mean
   validation Macro-F1 across all six complete tasks; ties prefer fewer records.
3. Freeze those Direct sets and sweep Indirect at 25, 50 and 100 records for
   Neighbor-fill, Molecule-cap and Group-balanced. Group-balanced uses the
   Molecule-cap Direct winner. Every prediction is a single full-flat call.

The runner verifies complete predictions and current input hashes before writing
`direct_selection.json` and unlocking stage 2. The two stages share one provider
pool at a time and a reusable preparation cache. Total: 32,688 validation
predictions (14,301 Direct-stage, 18,387 Indirect-stage). No test result is used
to choose budgets. `PROGRESS.md` updates every 15 seconds.

**Complete and verified:** all 32,688 predictions and prompt conditions passed;
every requested record budget was filled. Direct winners are Neighbor-fill=50
and Molecule-cap=100. Scores and four figures are registered under
`budgeted_evidence_protocol.aligned_record_scaling`; see `RESULTS.md` for the table.
The 2026-09-21 conditioned test sweep is separately registered under
`budgeted_evidence_protocol.aligned_record_scaling_test`; see the reusable
test/resume contract below. The separate fixed-budget TDC test is complete.

Artifact root: `outputs/paper/aligned_record_scaling_valid_20260918/`.
`launch.json` and `resume2_launch.json` are historical commands, not instructions
to resume this completed run with the current checkout. Preparation code changed
after the run finished preparation; existing manifests remain frozen. New runs
need a fresh output root. Reuse requires the original code contract or an explicit
verified migration, never relaxed manifest checks. Prior mappings and provider
configuration remain in that root. Current collaborator prompts are linked from
[`shared/README.md`](shared/README.md). Historical plotting points, originally
published at main `89728341`, have separate
[plotting instructions](shared/baseline_curves/README.md).

### Test record scaling and resume

**The test sweep completed on 2026-09-22 at 12:40 UTC:** 32,768 successful
predictions across all 96 task/setting cells (14,336 Direct-stage + 18,432
Indirect-stage), with no final failed queries. The final recovery used DGX027
at 512 concurrent requests. `resume_dgx027.json` records preservation of all
23,418 earlier successful outputs; `launch.json` records the historical resume
of only the incomplete Indirect phase. `scaling_status.json` and the registry
own completion/verification status. Scores and validation/test comparisons are
in `val_test_scores.csv`; the offline audit is reproducible with
`python outputs/paper/aligned_record_scaling_test_20260921/verify_results.py`.
The 52-second real-prompt endpoint preflight is recorded in
`dgx027_real_query_verification.json`.

The previous execution used dgx020:50002 plus dgx009:50001. Both endpoints passed
model discovery and actual scientific-prompt validation on DeepSeek-V4-Flash-0731.
Each has 512 admission slots; the global limit is 1024 requests, with 12 CPU
preparation processes. `resume_dual_dgx.json` pins the preserved predictions.
The originally requested dgx020 port 50001 and dgx005 ports 50000–50002 refused
connections. Earlier OpenRouter 1024/256 trials hit `402 in_flight_budget_exhausted`
with about USD 0.49 remaining. The full local disk also required moving only this
run's candidate cache to its project output root. `resource_pause.json` is the
historical pause; `resume_dual_dgx.json` records the previous two-endpoint resume.

On resume, `Prepared` counts checkpoints rediscovered after fresh manifest
validation, not all files already on disk or newly generated prompts. It can
temporarily fall while saved outputs remain intact. The coordinator registers stage cells serially and hashes their inputs.
Since 2026-09-22 it no longer loads all task indexes during registration:
workers load identity-audit metadata on demand into a shared, source/code-bound
`identity_audit/` cache. A file lock ensures only one worker builds each missing
entry; its content hash is verified on reuse. Candidate pools keep their original
scientific binding and remain reusable across budgets. Cold audit entries still
require one index load each; manifest hashing remains serial. Check readiness
receipts, worker children and provider inflight/completion counters together.
Existing validated checkpoints can feed inference while registration continues.

Use the same family runner for validation and test. Validation selects separate
Direct budgets; test requires its completed `direct_selection.json` and sibling
`direct/` artifacts. Before starting test, the runner recomputes the validation
selection, checks current validation input hashes and exact task coverage, then
pins a copy and the source receipt hash. It runs all seven Direct-stage settings
(None plus two selectors at 25/50/100) and all nine Indirect-stage settings. It
never changes the second-stage Direct context based on test scores. All test
scores, curves and descriptive rankings remain available for setting-by-setting
validation/test comparison; no grid points are omitted. For the six current scaffold tasks this is
2,048 rows × 16 settings = **32,768 predictions**.

The 2026-09-21 run uses `full_flat_v5_adapted.v2`: exact duplicate displayed
record fields are removed, with unchanged selectors, labels, budgets and full-flat
organization. The 2026-09-18 validation run remains a frozen v1 result. Test priors
are reused by exact structure and condition from the corresponding frozen test
progressive runs; `prior_verification.json` records all 2,048 checks. No raw tools
or benchmark labels are included in the analysis presented to the model.

The command below documents the completed grid. For a new experiment use a fresh
output/cache root and matching priors; do not restart the frozen completed run:

```bash
python -m tools.chembl_tool.paper_experiments.run_conditioned_assay_family_curve \
  --baseline-suite --baseline-record-scaling \
  --tasks bbb_martins bioavailability_ma skin_reaction ames dili carcinogens \
  --split-scheme scaffold --evaluation-subset test \
  --direct-budget-selection outputs/paper/aligned_record_scaling_valid_20260918/direct_selection.json \
  --visibility-mode deployment_visible_prefetched --budgeted-prompt full_flat_v5 \
  --cached-prior-sources outputs/paper/aligned_record_scaling_test_20260921/prior_sources.test.json \
  --candidate-molecules 128 --min-similarity 0 --selection-seed 0 \
  --direct-records-per-molecule 5 --records-per-molecule 5 \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --provider-pool-config outputs/paper/aligned_record_scaling_test_20260921/providers.json \
  --parallelism 512 --endpoint-concurrency-budget 512 \
  --preparation-processes 12 --preparation-workers 24 \
  --max-tokens 16384 --timeout-s 900 --retry-race-width 1 \
  --max-stage-requeues 3 --retry-delay-s 15 \
  --preparation-cache-root outputs/paper/aligned_record_scaling_test_20260921/candidate_cache \
  --output-root outputs/paper/aligned_record_scaling_test_20260921
```

The final `providers.json` uses local port 50227 for DGX027:50001 through
PARCC. If a future authorized run needs this forward and it does not already
exist, use:

```bash
ssh -fNT -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:50227:dgx027:50001 parcc
```

The final provider uses `deepseek-ai/DeepSeek-V4-Flash-0731`, temperature 1,
reasoning low/enabled and `max_inflight=512`, with a shared 512-permit admission
broker. The suite limit is shared across all tasks/settings. OpenRouter was
disabled. Earlier two-endpoint execution and connectivity failures remain in
the frozen receipts. Transport changes preserved scientific manifests and
successful prediction hashes. Endpoint verification is in
`dgx027_real_query_verification.json`; recheck availability before future runs.

For future runs, choose a fresh root and matching subset prior mapping. For val,
change `--evaluation-subset` to `valid` and omit `--direct-budget-selection`.
For test, use a completed matching validation receipt. Keep source, model,
generation and scientific settings pinned before execution. CPU preparation and
provider queues are bounded independently; raise CPU workers only when preparation
limits throughput. Set BLAS/OMP/MKL threads to 1 for the launcher. No per-task
runner copies or experiment-specific inference scripts are needed.

Resume only after the previous launcher has exited, using the same command and
output root. Locks prevent duplicate launchers. Compatible prepared contexts and
successful predictions are reused automatically; changing pinned code/inputs
requires a fresh root or an explicitly verified migration. `launch.json` records
the exact command; `PROGRESS.md`, each stage's `suite_status.json`, and the final
`scaling_status.json` expose progress. Completion requires both stages and all
96 task-setting cells, followed by row/prompt/source validation and registry
publication. Do not treat a launched or partial run as a completed result.

### Explicit fixed-Direct follow-up

`--baseline-suite --baseline-phase indirect_fixed` uses explicit
`--direct-budget`, `--direct-selector`, and one `--indirect-selector` to run
Indirect 25/50/100 in the existing shared preparation/inference pool. It runs
three settings, does not run None/Direct again, and rejects a validation selection
receipt or `--baseline-record-scaling`. Use a fresh output root when changing
the fixed context; the original validation-selected sweep remains frozen.

The user-requested 2026-09-22 follow-up fixes Neighbor-fill Direct=100 and
Neighbor-fill Indirect=25/50/100 on the same conditioned scaffold **test**
cohort (2,048 rows × 3 = 6,144 predictions, 18 cells). It is explicitly chosen
after inspecting the previous test sweep, not validation-selected. Registry
`budgeted_evidence_protocol.aligned_record_scaling_test_direct100_fill` owns
its verified results, command and cache binding. All 6,144 predictions completed
on 2026-09-22; mean Macro-F1 for Indirect 25/50/100 is 0.6642/0.6821/0.6789.
Full response/condition/hash checks, recomputed metrics and all-query selection
parity are recorded in `verification.json`; `comparison_scores.csv` compares
the original Direct 50 context. Root:
`outputs/paper/aligned_record_scaling_test_direct100_fill_20260922/`.

The relevant options, appended to the common model/input settings above, are:

```bash
--baseline-suite --baseline-phase indirect_fixed \
--direct-budget 100 --direct-selector neighbor_fill --indirect-selector neighbor_fill \
--parallelism 128 --endpoint-concurrency-budget 128 --preparation-processes 12
```

The user reduced this follow-up from 512 to **128 concurrent requests** on
2026-09-22. `resume_128.json` pins the 5,486 preserved successful outputs; only
the remaining 658 requests needed inference at that restart. Both the dispatcher
and its provider enforce 128; the existing shared admission broker is unchanged.

Its `launch.json` contains the complete executable command. It reuses the same
adapter v2, frozen test prior mapping, scientific code hashes and candidate cache
as the completed sweep. None and Direct-fill 100 remain references from that
verified run; the three new prediction cells are fresh. All selectors, conditions,
identity policies, model and generation settings stay fixed. Compare each new
Indirect budget with the original Direct-fill 50 + same Indirect budget.

### Completed tool-visible results

The completed validation and test reruns use **no similarity cutoff**
(`--min-similarity 0`, for both direct and indirect). They are registered under
`budgeted_evidence_protocol.baseline_no_threshold`; each root has a 15-second
`PROGRESS.md`. The completed validation pilot used 0.3, and the matching 0.3
test was stopped. Those artifacts remain separate historical references.
All six settings were rerun fresh; 12,258 validation and 12,288 test predictions
are fully verified. The five ML baselines remain unchanged. Curve figures, CSV
scores and verification receipts are registered for both subsets.

Both completed suites use **nominal direct B=50 and additional indirect B=50**.
Their historical Direct candidate pool was limited to 10 molecules. In the
Neighbor-fill arms, Direct fell short for 1,038/2,043 validation queries and
1,086/2,048 test queries; Indirect reached 50 for every query. The Direct sets
match between `direct_fill` and `both_fill`. These counts come from saved
`prepared.json` payloads, recorded in registry `baseline_no_threshold.record_count_audit`.
All settings make one full-flat prediction. Both suites use
`--visibility-mode deployment_visible_prefetched` by explicit user request; query
and neighbor SMILES, raw evidence and full tool text are visible. Blind outputs
are retained separately and are not reused as visible predictions.
Only the following six settings run:

| Setting | Direct selector | Indirect selector |
|---|---|---|
| None | none | none |
| Direct / Neighbor-fill | neighbor_fill | none |
| Direct + Indirect / Neighbor-fill | neighbor_fill | neighbor_fill |
| Direct / Molecule-cap | molecule_cap | none |
| Direct + Indirect / Molecule-cap | molecule_cap | molecule_cap |
| Direct + Indirect / Group-balanced | molecule_cap | group_balanced |

Use `--direct-selector` explicitly. Neighbor-fill has no per-parent record cap;
Molecule-cap and Group-balanced cap each parent at **5**, globally across groups
within the indirect pool. All use the same stable random record order/seed.
Group-balanced cycles through semantic families, taking one eligible card per
nonempty group per round; capped records are skipped within that group's turn.
Exhausted groups yield their remaining budget. Direct remains a separate pool.

Shared CLI overrides:

```text
--direct-budget 50 --direct-records-per-molecule 5
--indirect-budget 50 --records-per-molecule 5 --selection-seed 0
--min-similarity 0
```

Direct + Indirect contains at most 100 records; direct-only at most 50. Shortfalls
remain explicit; never duplicate cards. The direct set is identical within each
strategy's Direct versus Direct + Indirect pair, and between Molecule-cap and
Group-balanced. It can differ between Neighbor-fill and Molecule-cap.
Quality is deferred; Joint and its extraction queue remain paused.
These are the historical nominal 50+50 settings; see the
[aligned scaling protocol](#aligned-validation-record-scaling-2026-09-18) for
the current 128-molecule candidate pools and completed full-budget results.

The existing five methods remain `minimol_head`, `minimol_knn_condition`,
`minimol_knn_all`, `morgan_knn_condition` and `morgan_knn_all`. Their maintained
entrypoints are `baselines/minimol/run_train_cv.py`,
`baselines/minimol/run_embedding_knn.py`, `baselines/structure_knn/run.py` and
shared `baselines/conditioned_knn.py`, relative to the repository root.
Use canonical `data/conditioned_benchmark/<Task>/<split>/` inputs and registry
result roots. The local `outputs/paper/joint_validation_20260917/baseline_lineage_audit.json`
checks 60 existing six-task scaffold valid/test result cells against current rows,
conditions, labels and KNN train references. It does not reconstruct head training
or establish results on random splits. The Skin source-identity repair did not
change benchmark rows, so it alone does not invalidate these baselines.

Simple controls operate directly on eligible indexed source cards and do not
read annotation caches, compute Q/T/features, or resume extraction. Their audit
reports selection order, counts, groups and budget shortfalls; it does not invent
quality scores. Quality/Joint retain the exact source-assessment requirement.
The runner uses shared bounded preparation workers and the existing tool cache,
provider pool, request-admission brokers, retry logic and summary machinery.

The maintained single-setting entrypoint is
`run_conditioned_assay_family_curve.py --evidence-setting ...`. This mode makes one independent prediction, using a
fixed direct set and an optional selected indirect set. It consumes no previous
prediction or level state. The five existing supervised/KNN baselines are unchanged.

### Global scheduling for the six baselines

Use the same entrypoint with `--baseline-suite` to run all six simple settings
through **one shared provider pool**. Its default visibility is
`deployment_visible_prefetched`; explicit `--visibility-mode identity_blind`
remains available. Explicitly pass the 50+50 budgets and `--min-similarity 0`
as below: shared CLI defaults remain indirect B=64 and similarity 0.3 for the
paused research mode. The single-setting visibility default remains blind.

For the three Neighbor-fill arms only, pass `--baseline-settings none direct_fill both_fill --direct-budget 50 --indirect-budget 50`. This filter applies only to the standard suite, not record scaling or matched Direct-3. OpenRouter Batch uses the same runner and prompt with a provider whose transport is `openrouter_batch`; the shared client journals submissions and polls `/api/v1/batches`. Keep the base model ID (without `:batch`) in the provider configuration. Accepted jobs are asynchronous and are not completed predictions.

The completed GPT-6 Luna paper baseline is registered as `gpt6_luna_batch_b50`:
12,819 verified conditioned/TDC test predictions and $22.90500961 actual Batch cost.
Its `*_command.json` files record complete shared-CLI invocations; retained
`verify_results.py` re-audits saved outputs only. The one-off completed launcher
and startup diagnostics are archived, not additional active entrypoints.
Current figure sources and the TDC adapter-version caveat are owned by
`llm_vs_standard_baselines` and [the plotting guide](README.md#current-comparison-figures).

Context preparation and inference have separate limits. One preparation coordinator
validates each cell's manifest and submits query jobs to a shared CPU process pool.
Each job loads a candidate pool once and generates all missing settings for that
query through the existing selectors and renderer. Workers can move across tasks;
when only one task remains, all workers can help finish it. The coordinator owns
an isolated process group so failures/cancellation also stop its CPU workers.
Atomic `prepared.json`
files stream to inference immediately, without waiting for a task or setting.
The ready queue enumerates created query directories and checks only unpublished
results within them, avoiding a per-poll filesystem lookup for every expected
query. A 1,000-query regression checks that a single created query needs only one
prepared-file lookup; completed queries are not checked again.

`--preparation-processes` bounds the total CPU workers across the suite (six in
the completed validation sweep; twelve in the current test command). `--preparation-workers` remains the thread limit
for standalone single-setting preparation; the suite does not multiply these two
limits. `--parallelism` is the separate suite-wide inference budget. Provider
configuration and shared admission brokers enforce endpoint limits. All cells
share provider load/latency state; retries wait outside request slots.

`--preparation-cache-root` selects a shared candidate cache (default:
`candidate_cache/` inside the suite). Keys bind task, exact molecule text,
source/index/catalog and implementation hashes, split, similarity cutoff and
candidate count. They exclude selector, record budget, gold and external condition
because those do not enter candidate retrieval. Atomic compressed entries and
per-key locks prevent partial or duplicate fills. A worker loads its task index
only on a cache miss; the coordinator reads each task index once for identity
exclusion audit metadata. Selection, tools and prompt rendering remain shared
with the single-setting runner. Existing prepared contexts are validated and
reused, and compatible completed predictions do not trigger new model calls.
Per-cell `preparation_done.json` records the manifest, errors, query-group count
and process limit. Code changes still require a fresh experiment manifest;
a scheduling change is not permission to ignore checkpoint hash mismatches.


Shared-process validation (2026-09-18): 74 tests passed; 153 complete prepared
contexts across six tasks matched the saved contexts exactly. On 12 Carcinogens
queries × nine settings with an existing candidate cache, preparation took
81.34 s with the original task worker versus 15.90 s with six shared processes
(5.12×). This measures preparation, excluding source initialization and model
latency. Receipt: `outputs/paper/aligned_record_scaling_valid_20260918/preparation_upgrade_20260918/receipt.json`.
The completed scaling run finished preparation under its original loaded runtime;
the upgrade was applied afterward without restarting its inference.

Example for a **new** validation run using the saved provider configuration
(check endpoint availability before launch; this explicit flag reproduces the
historical tool-visible prompt, not the new alignment):

```bash
python -m tools.chembl_tool.paper_experiments.run_conditioned_assay_family_curve \
  --baseline-suite --split-scheme scaffold --evaluation-subset valid \
  --budgeted-prompt budgeted_v1 \
  --visibility-mode deployment_visible_prefetched \
  --direct-budget 50 --indirect-budget 50 \
  --direct-records-per-molecule 5 --records-per-molecule 5 \
  --min-similarity 0 \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --provider-pool-config outputs/paper/llm_baselines_50_visible_no_threshold_valid_20260917/providers.json \
  --parallelism 1536 --endpoint-concurrency-budget 1536 \
  --preparation-processes 6 --preparation-workers 24 \
  --retry-delay-s 15 --max-stage-requeues 3 --max-tokens 16384 \
  --output-root outputs/paper/llm_baselines_50_global_valid
```

The root owns `suite_status.json` and a `PROGRESS.md` refreshed every 15 seconds.
The completed roots are `outputs/paper/llm_baselines_50_visible_no_threshold_valid_20260917/`
and `outputs/paper/llm_baselines_50_visible_no_threshold_test_20260917/`.
Each `launch.json` contains its exact command; `preflight.json` binds its inputs
to the prior validation suite. Validation has 12,258 predictions and test has
12,288. The two global schedulers shared the same admission brokers: dgx005 512
and OpenRouter 1024 slots in total, with six resident task workers per subset.
The operational upgrade is recorded under
`outputs/paper/analysis/baseline_preparation_upgrade_20260917/`: exact prompt
parity checks, test results and a checkpoint migration receipt. Resume after a
code change still requires explicit provenance; the cache never bypasses the
runner's scientific manifest checks.
The final verification and each figure's receipt are linked from the registry;
`verification_provenance.json` identifies the frozen verifier used for the run.
Per-cell manifests, prepared prompts, outputs and metrics retain the existing
`runs/<setting>/<task>/` layout. Compatible completed responses are validated and
reused; a suite lock prevents duplicate coordinators. A mismatched scientific or
implementation manifest is rejected before saved contexts become eligible.
Provider concurrency may change on a compatible resume; prompts and selected
records do not change with scheduling.

The historical 0.3 visible pilot finished under its original cell coordinator; it
was not hot-migrated to this scheduler and is not a throughput evaluation of it.
Its loaded budgeted runner is preserved under the pilot's `frozen_runtime/`.
New suite code must not be used to overwrite that pilot's pinned manifests.
Quality/Joint and annotation extraction remain paused. The post-run code review,
focused checks and pre-review code snapshots are indexed in
[`baseline_maintenance_20260917.json`](receipts/baseline_maintenance_20260917.json).
Completed manifests remain pinned; maintenance does not authorize an in-place
resume under changed code.

For a completed, verified suite, compare all six settings with the five frozen
baselines through the canonical plotter:

```bash
python -m tools.chembl_tool.paper_experiments.plot_starling_model_comparison \
  --budgeted-suite-root outputs/paper/llm_baselines_50_visible_no_threshold_valid_20260917 \
  --baseline-audit outputs/paper/joint_validation_20260917/baseline_lineage_audit.json \
  --output outputs/paper/llm_baselines_50_visible_no_threshold_valid_20260917/figures/baseline_comparison.png
```

This recomputes Macro-F1/Accuracy, verifies baseline/cohort hashes and checks
the external condition in every actual prompt. It exports PNG/PDF/SVG figures,
score CSV and an audit receipt. Means weight tasks equally; a single run does
not supply replicate error bars or a significance result.

### Record-scaling figures

Aligned validation scaling figures and all score points live in
`outputs/paper/aligned_record_scaling_valid_20260918/figures/`.
The Direct figure shows None and budgets 25/50/100 with one line per selector.
The three Indirect figures retain None → Direct → Direct+Indirect and use one
line per indirect budget; the selected Direct budget is fixed within each figure
(fill=50, cap/group=100). Every figure uses the same y-axis limits and matched
five-baseline references. These are independent full-flat predictions, not
successive reasoning calls.

The maintained renderer remains `plot_assay_retrieval_curve.py`. Portable CSV
metadata accepts `curve_settings`, optional `stage_labels`, `title` and
`budget_label`; each curve's method list must match the stage count. For example:

```bash
python -m tools.chembl_tool.paper_experiments.plot_assay_retrieval_curve \
  --baseline-curves-csv outputs/paper/aligned_record_scaling_valid_20260918/figures/cap_scaling.csv \
  --baseline-curves-metadata outputs/paper/aligned_record_scaling_valid_20260918/figures/cap_metadata.json \
  --analysis-dir outputs/paper/aligned_record_scaling_valid_20260918/figures \
  --output-stem cap_scaling --macro-f1-limits 0.45 0.85
```

### Supportive and contradictory citation statistics

Use the existing offline trace-analysis entrypoint; this reads completed outputs
without model calls or changes to the prompt, retrieval, or predictions:

```bash
python -m tools.chembl_tool.paper_experiments.analyze_progressive_trace_adoption \
  --budgeted-suite-root outputs/paper/llm_baselines_50_visible_no_threshold_valid_20260917 \
  --output-dir outputs/paper/llm_baselines_50_visible_no_threshold_valid_20260917/citations
```

Replace `no_threshold_valid` with `no_threshold_test` for test. The existing vllm
runtime supplies `ijson`, used to read only selected evidence from large ledgers.
Each query reports supportive, contradictory, and their deduplicated union,
separately for all/direct/indirect records: distinct cited records, saved retrieval
parent count, effective cited parents, and citation fraction of the actual visible
record pool. Effective parents are `N^2 / sum_m(c_m^2)` for per-parent distinct
record counts; undefined when no records are cited. Role overlaps are retained
and counted, never forced into exclusive classes. TSV means are per query within
each task/setting; effective counts condition on nonempty citations and fractions
on nonempty visible pools, with both denominators reported explicitly.

Roles are model-assigned final-answer provenance, not gold labels, reasoning
mentions, or causal evidence use. Current outputs supply role arrays; a future
upstream-format response can supply `claims[].evidence_role`. This analysis does
not implement the proposed upstream prompt rerun or report prediction transitions,
costs, or stability metrics. `citations.per_query.jsonl`, `citations.summary.tsv`
and `manifest.json` retain query-level mappings, input hashes and aggregate scope.

## Inputs and settings

- `none`: query context only (cached single-molecule analysis in the aligned profile;
  property tools in the historical tool-visible profile).
- `direct`: the shared `--candidate-molecules` pool (default 128), an explicit total `--direct-budget`
  (default 50), and `--direct-selector neighbor_fill|molecule_cap`; only the latter
  applies `--direct-records-per-molecule` (default 5).
- `direct_indirect`: the same direct set plus the selected indirect records.

L1 is direct; **every L2+ record is indirect**. All indirect candidates use
**parent-disjoint** retrieval, including L2. Direct uses scaffold-disjoint on
the scaffold split and parent-disjoint on random. The existing source eligibility,
heldout prefilter and task-specific tautomer identity contracts still apply.
Unresolved parent identities (including wildcard fragments) are excluded before
ranking, with backfill under the same similarity threshold. The frozen index and
source are unchanged; selection audits record the excluded count and ID hash.
Physical assays are partitioned at record level before retrieval; a shared assay
cannot leak L1 cards into the indirect pool.

The default additional indirect budget is **64 records**; the agreed sweep is
**16, 64, 128**. A shared pool retrieves the top 128 eligible indirect molecules
by descending similarity and expands all their distinct indexed cards. Active
baseline runs explicitly use `--min-similarity 0`; the generic CLI's historical
default remains 0.3, so do not omit the override. Candidate pool
size and direct selection do not depend on B, selector or weights. No implicit
per-assay record cap is allowed. Exact duplicate card surfaces within a molecule
are deduplicated; source rows/assay IDs are retained in the selection audit.
Near duplicates share features and have diminishing benefit; there is no claim
that semantic duplicate detection is complete.

| `--indirect-selector` | Selection |
|---|---|
| `neighbor_fill` | Similarity-ordered molecules; stable random card order within each; fill B. |
| `molecule_cap` | Same ordering, at most 5 cards per parent; fill from subsequent molecules. |
| `group_balanced` | Round-robin over existing semantic families in the shared pool; similarity-ordered molecules and random cards; same parent cap. Empty/exhausted families are skipped. |
| `quality` | Top B by the same Q*T used by Joint. |
| `joint` | Global marginal-gain greedy for the objective below; no per-parent cap. |

Random controls use `--selection-seed`. Quality/Joint stop when no positive gain
remains. Every method reports actual selected counts and budget shortfall;
it never repeats a card just to fill B. Group balance is an allocation control,
not a claim that equal family representation is scientifically optimal.

## Score and objective

For record r under query task/condition q:

```
Q_q(r) = max_{o in observations(r)} V(o) * E_q(o) * (1 + C_q(o) + O_q(o) + I(o)) / 4
p_q(r) = Q_q(r) * T_q(r)
A(S) = sum_r p_q(r)
D(S) = sum_m sum_j sqrt(sum_{r in S_m} p_q(r)^2 * Xbar_rj)
H(S) = sum_g sqrt(sum_m max_{r in S_mg} p_q(r)^2)
F(S) = a*A(S) + b*D(S) + c*H(S), subject to |S| <= B
```

An empty max is zero. The direct set is fixed input to the reader, outside F.
`g` is the existing semantic family, `m` is the normalized parent (the task's
explicit tautomer leakage group for DILI/Carcinogens). H counts each parent once
per family, through its strongest selected p; adding another record from that
same parent/family does not increase H unless its p is higher.

The rule rubric is explicit, versioned and **not a calibrated probability**:

- E: endpoint relevance from existing task readout/assay-scope vocabularies.
  Related=1, unrecognized=.5, explicitly unrelated=0. A regex miss never means
  unrelated. For DILI, generic mechanisms without an identified hepatic context
  receive .5; a negated hepatic model does not establish hepatic scope. Explicit
  plant systems receive at most .25 for BBB, DILI, Carcinogens and Skin.
  Positive and negative outcomes receive identical relevance treatment.
- C: query-condition compatibility using existing task rules. Matched=1,
  limited/unknown=.5, explicitly mismatched=0. No external query condition means
  not-applicable=1. Tested species comes from specimen/system fields, not a
  human-equivalent-dose phrase.
- O: focal tested object=1, unknown or qualified scope=.5.
  Existing metabolite/radiolabel, modified specimen, special formulation and
  v4 observation-level attribution/scope assessments contribute. Perturbation,
  mixture, metabolite-measurement scope and explicitly previously sensitized
  elicitation populations receive at most .5. A molecule being a metabolite does
  not by itself mean the assay measures a different object; the measurement scope
  must establish that distinction. Unclear attribution is not proof of the wrong
  molecule. Free-text unresolved notes remain audit
  notes and do not by themselves lower every observation's score.
- I: fraction of four checks present: endpoint, result, source support, assay/system.
  Qualitative outcomes count as results. Prediction/context-review flags cap I at .5.
- V: reported measurement=1, model prediction=.5, inference=.35,
  classification=.6, methods-only=.1, unclear basis=.4. Explicit source limitations
  cap V: inadequate evaluation=.2, method limitation=.5, conflicting results=.35,
  missing study details=.75, explicitly inconclusive assay result=.5. A narrowly
  checkable replicate-count contradiction also caps V at .35 without changing the
  source result. These are provisional ordinal weights, not calibrated reliability.

Each observation uses its own source-bound conditions. Only the reported aspect
overrides the original condition; a dose quote must not erase a known species.
Taking the strongest usable observation prevents an unrelated background statement
from downgrading a well-supported measurement. All observation scores remain in
the audit, and the reader still sees the complete original record.

Missing information and scientific limitations retain distinct status/reason
fields even where their provisional numeric scores coincide. Identity-invalid
records remain excluded upstream. Rules are heuristic and require empirical
validation; neither rule completeness nor prediction improvement is assumed.

`X` uses the existing source-only endpoint/outcome/assay/system/exposure/role
blocks, plus v4 bound observations. The Joint feature encoder removes source field
names, deduplicates canonical/raw repetitions, normalizes common species and
transporter aliases, and gives each block the same square-root mass. It is a
shared lexical/observation representation, not an exhaustive semantic ontology.
With an annotation, unstructured `extra_details` and `qualifying_conditions`
are not treated as typed exposure features; explicit exposure fields, extracted
tags and bound condition quotes supply that block. A separate `source_content`
block represents the original support paragraph. This preserves overlap between
reader-visible repeated passages whose native assay focuses differ, without
discarding the distinct readouts or treating identical prose as proof of the same
experiment. Bibliographic notes remain in source provenance rather than being
rewarded as exposure diversity.
When an assessment is present, native direction codes embedded in measurement
strings do not add a second direction feature; the assessed observation supplies
direction while the source measurement is retained.
The v4 extractor uses `record_assessment_input.v3`: prior categorical direction
codes (including numeric binary-outcome encodings), generated scalar/unit fields and purely qualitative canonical measurement summaries are omitted from
the extraction input and quote catalog. Literal measurements and source prose
remain unchanged. The complete native input is still retained for source binding
and cache identity. This prevents an earlier extraction code from acting as an
independent experimental observation; it does not certify the new interpretation.
Primary record linkage does not attribute every background treatment arm to the
focal molecule. Extraction distinguishes a reported property claim's direction
from its experimental basis. The extraction runner supports `--enable-thinking`;
changing reasoning, prompt or input settings requires a new annotation cache.
When only the source inventory changes, `run_record_diversity_extraction.py
--reuse-annotations <frozen-backup.sqlite>` can seed the new cache. It requires
identical model/prompt settings, exact request keys and native input payloads,
revalidates source quotes, and preserves original attempt/cost receipts. Changed
records remain pending. Use an atomic SQLite backup, never a live WAL database.
An asserted classification supported only by native carcinogenic-role/narrative
codes receives at most inferred-evidence strength, rather than the weight of an
independently reported authoritative evaluation. The original claim remains in
the audit and raw reader context.
Explicitly unrelated observations are removed from relevant endpoint features.
With `Z_r=sum_j sqrt(X_rj)`, use `Xbar_rj=X_rj/Z_r^2`. This fixes a singleton's
total square-root feature mass at <=1. A formal context card with empty features
fails preparation. Scoring
and features stay fixed throughout selection. D rewards additional information
within each parent with diminishing returns. It is not a global pairwise
chemical-distance penalty.

Joint defaults provisionally to `(a,b,c)=(.5,.375,.125)`. Validation grid:

```
1,0,0           0,.75,.25       .25,.5625,.1875
.5,.375,.125    .75,.1875,.0625 .5,.5,0        .5,.25,.25
```

Weights are nonnegative and sum to one. With fixed scores/features/groups,
F is monotone submodular, so marginal greedy has the standard cardinality
approximation guarantee for **F**, not for downstream prediction accuracy.
The implementation uses lazy marginal updates with deterministic similarity/ID
ties. `--uniform-record-quality` provides Q=1 while retaining the same eligibility
and features. The scalar quality term remains enabled by default.

## Source features and future transfer

The scorer binds each indexed card back to its exact current native source
surface. Ames uses its exact source-row ordinal join. Missing bindings fail;
no fuzzy attribution or automatic return of excluded source rows is allowed.
Duplicate surfaces with differing native metadata use minimum quality and common
features, with all source IDs/input hashes retained.

Quality/Joint runs require `--annotation-db PATH` with v4 source assessments
for every candidate. Inputs, extraction prompt hash, schema and exact source quotes
must match. Every alternative native input of an ambiguous source binding requires
its own assessment. A partial cache fails before tools/prediction calls. Use a
frozen SQLite snapshot; never pass a live WAL database.

Use `--feature-inputs-only` to export the exact retrieved source inputs to
`feature_inputs/*.features.parquet`. The shared `run_record_diversity_extraction.py`
accepts that directory with `--annotation-version record_diversity_annotation.v4`.
This source-only export makes zero model/tool calls. Explicit diagnostic
`--allow-native-features` permits native/v3 fallback, but is not a formal v4 run.
An optional extraction `--priority-records` JSON list of `{task, record_id}`
objects schedules complete query pools first without changing inputs, cache keys
or candidate eligibility. Failed assessments remain missing; formal preparation
cannot silently accept them. Use the extractor's SQLite backup for a consistent
frozen snapshot while the larger queue continues.

`selection.json` also reports annotated/native candidate counts, annotated
selected counts, direct-card coverage and annotations with unresolved notes.
Feature extraction is offline and
source-only; reuse an input-bound annotation across queries, then compute Q for
each task/condition. It does not call the extractor during greedy selection.

The earlier bounded 2026-09-17 audit used `record_quality.v2`; its artifacts are in
`outputs/paper/analysis/joint_semantic_rule_audit_20260917/README.md`.
It is historical relative to the working `direct_indirect_budgeted.v3`,
`record_quality.v3` and `budgeted_record_features.v3` implementation. The paused
source/candidate audit is at `outputs/paper/joint_validation_20260917/`.
Schema/quote validity does not establish scientific validity. These changes have
not yet established benchmark improvement or exhaustive rule coverage.

**T=1 is the default.** No structure-similarity score substitutes for transfer.
The future tool only evaluates query-molecule to neighbor-record transferability.
`--transfer-scores FILE` accepts this strict interface:

```json
{
  "version": "query_record_transfer.v1",
  "model": "tool/version-or-artifact-hash",
  "scores": [
    {"task": "ames", "query_hash": "from selection.json", "record_id": "candidate id", "score": 0.8}
  ]
}
```

Every candidate for each evaluated query must have one finite score in [0,1].
Missing, duplicate or invalid scores fail; the tool itself is not implemented
here. p=Q*T affects all three terms; compare real T to constant 1 when available.

## Running and artifacts

Use the existing family runner with a fresh output root:

```bash
python -m \
  tools.chembl_tool.paper_experiments.run_conditioned_assay_family_curve \
  --evidence-setting direct_indirect --indirect-selector joint \
  --tasks bbb_martins bioavailability_ma skin_reaction ames dili carcinogens \
  --indirect-budget 64 --joint-weights .5 .375 .125 \
  --annotation-db /path/to/frozen_current_v4_annotations.sqlite \
  --split-scheme scaffold --evaluation-subset valid \
  --visibility-mode identity_blind \
  --model nvidia/GLM-5.2-NVFP4 --base-url http://127.0.0.1:50000/v1 \
  --api-key-env GLM_LOCAL_API_KEY --tool-service-url http://127.0.0.1:8765 \
  --parallelism 128 --output-root outputs/paper/direct_joint_valid
```

Model/provider settings are explicit choices, not a newly selected paper model.
Use `--prepare-only --indices 0` for a real-tool preparation smoke check.
`--prepare-only --skip-tool-prefetch` is an offline diagnostic in a separate root;
its outputs cannot be used for inference. Both visibility contracts are supported.
Identity-blind hides query/neighbor names and structures while preserving computed
tool text. Score, feature, label and selection-audit fields never enter the prompt.

The default scaffold indexes come from the registry's verified
`all_distinct_cards_v1` and, for Skin/Bioavailability, `source_reported_fields_v1`
variants; a variant's `task_roots` overrides its shared root for a reviewed task
revision (currently Skin v8). For a different split/build, supply `--retrieval-index-root ROOT`, with
`ROOT/<task>/<split>/assay_neighbor_index.pkl` and its manifest. Only current
heldout-filtered, uncapped indices are accepted. Random uncapped assets are not
automatically built or scheduled by this mode.

The same CLI runs the other six settings. Keep identical candidate/direct limits,
model, visibility, source hashes and task rows when comparing them. Change only
setting/selector, or B for a budget sweep. Use distinct output roots. Existing
outputs are resumed only under the same contract; changed code, inputs, weights,
budget, model or preparation mode require a fresh root. Changing the list of
indices can extend a compatible root.

The canonical curve plotter provides the None → Direct → Direct + Indirect
view with five matched ML baselines on the right. It reuses the overview's
cohort/hash and actual-condition-prompt verification; no inference is repeated:

```bash
python -m tools.chembl_tool.paper_experiments.plot_assay_retrieval_curve \
  --budgeted-suite-root outputs/paper/llm_baselines_50_visible_no_threshold_valid_20260917 \
  --baseline-audit outputs/paper/joint_validation_20260917/baseline_lineage_audit.json \
  --analysis-dir outputs/paper/llm_baselines_50_visible_no_threshold_valid_20260917/figures \
  --output-stem baseline_curves --macro-f1-limits 0.45 0.80
```

Replace `no_threshold_valid` with `no_threshold_test` in both paths for the test
figure. Each figure exports PNG/PDF/SVG, CSV scores and a verification receipt.

Artifacts:

- `experiment_manifest.json`: source/index/input/code hashes and configuration.
- `runs/<task>/query_*/selection.json`: candidate Q components, feature/annotation
  hashes, selected source cards and marginal-gain trace; no gold labels.
- `prepared.json`: exact messages, request hash and card aliases.
- `tools.json`: full tool receipts and unavailable-tool diagnostics.
- `evaluation.json`: labels, kept separate from selection and prompt assembly.
- `output.json`: validated response and request binding; successful checkpoints
  resume without repeating inference. Invalid JSON/citations use shared repair
  and retry logic, not another reasoning level.
- `summary/<task>/{metrics.json,predictions.jsonl}` and `execution_status.json`:
  completion, failures, accuracy and macro-F1 using existing summary machinery.

Transport, provider admission, racing retries, tool prefetch and JSON/citation
validation reuse existing shared implementations. No new service or standalone
launcher is introduced. Full LLM benchmark performance remains an experiment,
not a consequence of preparation tests.


### Frozen prepared-request replay

The family runner accepts `--run-prepared` for independently prepared budgeted
requests in an explicit output root. It checks the manifest, generation settings,
benchmark input hashes, request hashes and evaluation binding, then uses the
ordinary provider pool, JSON/citation validation, resumable scheduler and summaries.
Use `--retry-race-width` for bounded first-valid retry races; multiple processes
can share a provider `admission_socket`. This mode does not retrieve, select or
rewrite evidence. Changed scientific requests require a new manifest/output root.
The explicitly test-tuned BBB development experiment is registered separately as
`tdc_bbb_indirect_test_iteration_20260924`; it does not replace the frozen baseline.

The BBB cleanup extracted the shared replay implementation into `prepared_evidence.py`
and deterministic simple selection into `common/record_budget.py`; the family CLI
is unchanged. Portable BBB inputs and collaborator commands are maintained in
[tasks/bbb_martins/INDIRECT_ABLATION.md](../tasks/bbb_martins/INDIRECT_ABLATION.md).
The guide identifies the official 406-row TDC BBB test set and the single
maintained assay-aware filter. Historical scores and request-parity receipts used
the retired v1 filter; current-filter validation is offline only.
Original output-root preparation scripts remain hash-bound provenance only.
