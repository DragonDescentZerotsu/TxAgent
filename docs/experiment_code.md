# A small core for many experiments

[发布整合状态（中文）](release_refactor.md)记录本地 baseline 的迁移进度、保留的科学规则和待验收项目。

Keep the implementation small enough that a reader can follow an experiment
from its command to its evidence, prompt and result. Put reusable scientific
behavior in Python. Put the settings for a particular experiment in a Bash
script that calls that behavior. A new experiment should usually add a recipe,
not another copy of a prediction or scoring program.

The unit of reuse is a distinct capability. Prediction uses the existing flat
harness and its runner, scheduler and task adapters. Context optimization,
semantic grouping, semantic utility scoring, query-prior generation, baseline
training and saved-result analysis have different inputs and scientific jobs;
they retain their own commands. Shared functions belong with their existing
owner when several real callers need them. Small single-use operations can stay
inline. Use existing libraries before writing another implementation.

Minimizing code does not mean hiding every operation behind one universal
command. It does not mean combining different objectives, silently changing a
prompt, replacing a frozen policy, or erasing useful controls. Versioned V7–V10
construction retains its scientific differences. Historical prompt renderers
and artifact readers remain available for reproduction. Common transport,
retrieval, UID validation and metrics can be shared without merging those
contracts. Consolidate only when behavior is actually the same and the shared
implementation is easier to read.

Expose parameters that experiments really vary: inputs and cohorts, task/level,
selection budgets, objective weights, prompt/prior choice, chains, overlap,
replica counts and provider configuration. Avoid speculative configuration
layers, wrapper classes and launch frameworks. A recipe should show these
choices directly. Python still owns validation, scientific calculations,
rendering, aggregation, checkpoints, resume and provenance; Bash owns concrete
invocations and sweeps. Preserve performance controls and traces when removing
duplication. A concise program that changes the experiment is an incorrect
simplification.

Before retiring an executable, trace its callers and retain useful behavior in
its capability command. Verify the replacement on a focused fixture, including
evidence order and identity. Completed experiments and immutable releases stay
intact. New commands write new run identities; they do not select a new active
release merely because a candidate finished. This guide and the capability READMEs record the public maintenance contract.

## Capability map

| Capability | Maintained owner | Inputs and outputs |
| --- | --- | --- |
| Query prediction | `predict/harnesses/branches/` | Frozen query/cache/selection, prompt and prior → requests, resumable predictions and traces |
| Paper LLM baseline | `predict/harnesses/branches/paper_baseline.py`, `predict/prompts/paper_baseline.py` | Restored paper inputs → new candidate selection, portable prepared-input bundles, embedded frozen query priors (no regeneration), exact messages, shared API execution, resumable native predictions and offline verification |
| Paper record controls | `predict/retrieval/paper_baseline.py`, `optimization/selection/record_budget.py` | Frozen source index / TDC train labels → shared identity filtering and deterministic record-budget selection |
| Model transport and retries | `predict/api_client/{client,pool,batch,retry}.py` | Shared request/response handling, durable Batch journals and cancellable first-valid retry races; paper execution supplies native validators and bounded stage requeues |
| Parent/scaffold exclusions | `predict/retrieval/policies.py`, `task_identity.py` | Molecular identities and declared task policy → candidate eligibility; never rewrites benchmark splits |
| Morgan / embedding KNN | `predict/baselines/structure_knn/run.py`, `minimol/run_embedding_knn.py` | Frozen train reference and query splits → neighbors, scores and decisions |
| MiniMol head | `predict/baselines/minimol/run_bioavailability_ma.py` | Frozen embeddings and train/valid/test → validation-selected ensemble and predictions for all six tasks |
| Encoder / auxiliary training | `predict/baselines/minimol/run_multitask.py` | Main splits and filtered auxiliary targets → graph inputs, checkpoints and ensemble predictions |
| RASAR | `predict/baselines/retrieval_features.py`, `retrieval_forest.py` | Main/auxiliary labels → shared nearest-neighbor features, validation-selected RF and predictions |
| Evidence and gold construction | `data/processing/` | Versioned source records and task rules → published data releases |
| Selection and scoring | `optimization/` | Frozen candidates and objectives → reviewed scores or ordered physical evidence selections |
| Result analysis | `analysis/`, [result index](results.md) | Saved predictions and source bindings → derived analyses and verified reports; `analysis.prediction.paper_results` exports frozen paper tables and figure values; `--figures` renders the three current manuscript PDFs |

[Baseline recipes](../experiments/baselines/README.md) expose the paper settings.
The [integration record](release_refactor.md) owns the migration status; a mapped
capability is not a claim that all its historical experiments are distributable.

The local `experiments/retrieval_cache/molecular_policy.sh` and
`experiments/evidence_library/score_skin_gold_voters.sh` recipes locate this
checkout from their script paths and default to `python` in the active environment.
Use `TXAGENT_PYTHON` and `SCORER_PYTHON`, respectively, for an explicit interpreter;
set `HF_HOME` when reusing an existing model cache. The separate distributed
stage/collect recipes still describe a historical cluster deployment and require
further portability review before publication.

## Canonical semantic utility scorer

`optimization.semantic_buckets.bucket_scoring` is the maintained
engine. Bradley–Terry fitting and its ranking executables are retired. The
older V5 sliding and V6 shelf-calibration executables are also retired;
only the latest maintained construction, refinement and scoring commands remain.
Semantic intermediates were removed; selected releases and required pinned inputs remain.
Grouping, refinement, eligibility and publication retain their distinct jobs.

Pass 1 scores independent batches. Pass 2 sorts each task-level by descending
Pass-1 utility, breaking ties by bucket ID. It divides that order into batches
and streams waves of parallel chains. Completed chains supply their last
`overlap` candidates as locked anchors for the next wave. The first
`advance_after` completed chains freeze the anchor decision; later completions
cannot change it. All scheduled candidates must still complete.

Automatic mode uses one reference across **all requested task-levels**:
`largest bucket count / max_chains`. Each task-level chooses a power-of-two
chain count that makes its count per chain closest to that reference, capped
by available batches. Manual mode uses `--chains N`, with optional
`--chain-count TASK:LEVEL=N` overrides. Defaults are batch sizes 12/12,
fanout/averaging 1/1 then 3/3, overlap 2 and advancement after 2 chains.
Averaging uses the first required valid replicas and the existing aggregation
and representative-rationale rules; late receipts are retained.

Inputs are an explicit JSON descriptor. Paths are relative to the descriptor:

```json
{
  "tasks": {
    "dili": {
      "target": "drug-induced liver injury",
      "world": "world.parquet",
      "payloads": "bucket_prompt_payloads.json.gz",
      "levels": ["L2", "L3"]
    }
  }
}
```

The world needs `level` and `semantic_bucket_id`. Payloads are the existing
bucket-ID-to-card mapping with identity/source components and sample records;
`source_labels` can optionally name evidence families. Resolve scientific
inputs through the selected manifests. Preparation reads existing cards and
writes schedules, reviews, bindings and hashes; it makes no provider probes or
completion requests. Execution uses the shared provider pool, records actual
served models/routes and replica receipts, and rejects changed inputs, code,
prompts or provider configurations on resume.

```bash
python -m optimization.semantic_buckets.bucket_scoring prepare \
  --input-manifest inputs.json --root outputs/analysis/semantic_buckets/new-run \
  --chain-mode auto --max-chains 8 \
  --pass1-provider-config provider.json --pass2-provider-config provider.json
python -m optimization.semantic_buckets.bucket_scoring run --through-pass 1 \
  --root outputs/analysis/semantic_buckets/new-run
python -m optimization.semantic_buckets.bucket_scoring run \
  --root outputs/analysis/semantic_buckets/new-run
```

Each pass accepts its own batch size, fanout, averaging count and provider JSON
at preparation. A provider JSON can declare `fixed_replica_counts` for a pinned
provider mix; its positive counts must sum to that pass's fanout. Otherwise
replicas use the shared pool's dynamic routing. Local DGX and OpenRouter use
`predict/api_client/transport.py` for SDK construction and
`predict/api_client/pool.py` for routing/capacity. Provider JSONs are under
`predict/api_client/providers/`. Finished scores remain unselected candidates;
publication is a separate operation.

## Physical UID selections

`predict/retrieval/selection.py` owns the common reader, validation and export.
Direct, indirect and composed outputs from `optimization.selection.search` expose a
`uid_selection` manifest. Their scientific manifests, objectives and original
context/record tables remain available. The UID manifest binds its records,
query IDs, cache manifests, budgets and optimizer snapshot.

For L1, the export contains the underlying physical voters, expanded in frozen
within-context order up to the record cap. `context_id`, `context_rank` and
`within_context_rank` preserve grouping and the frozen context label. L2+ rows
contain ordered physical evidence UIDs. Both export and prediction validate
candidate membership, query identity and declared parent/scaffold exclusions.
A context ID alone is insufficient for the prediction UID interface.

Manual selections use `--uids UID... --uid-level L1 --indices INDEX` for one
query/level, or `--uids-file PATH` for a batch. A file can be the optimizer
manifest, its canonical UID manifest, or a TSV:

```tsv
benchmark_row_id	level	selection_rank	source_row_uid	context_id
query-1	L1	1	physical-uid-1	context-1
query-1	L1	2	physical-uid-2	context-1
query-1	L2	1	physical-uid-3	
```

The required TSV columns are `benchmark_row_id`, `level`, `source_row_uid`.
Ranks are per query/level; file order determines selection order. Context IDs
are optional when membership is unambiguous. Files scope the default batch to
their declared query IDs. Explicit query selection must match those identities.
Unknown, repeated, leaked or ambiguous UIDs fail before inference. Manual
selection never silently fills an omitted level or substitutes another record.
The requested budgets act as caps.

The flat CLI and matrix CLI accept these arguments. Direct-label evidence
baselines (`direct_l1_knn`, `selected_direct_knn`, `assay_transfer_knn`) also
accept them with explicit `--input-jsonl`, `--cache-index` and task/subset.
Their `--indices` names benchmark query IDs; the prediction harness uses input
row indices. Direct voting accepts L1 only and votes once per frozen context.
Query-prior and structure-only predictors do not consume evidence UIDs.

## Experiment recipes

- `experiments/scoring/two_pass_auto.sh` and `two_pass_manual.sh`: canonical
  scoring settings; prepare offline by default, execute with `RUN=1`.
- `experiments/prediction/manual_uids.sh`: manual/optimizer UID prediction;
  prepares by default, executes with `RUN=1`.
- `experiments/selection/gold_indirect.sh`: explicit Gold selection recipe.
- `experiments/analysis/`: L1 widths/priors/contrasts, progressive pair-bucket
  distribution, and L2 selection-overlap studies on saved artifacts.
- `predict/baselines/minimol/run_direct_gpu_sweep.sh`: the eleven declared
  MiniMol configurations, one sequential worker per GPU, cache skips and
  failure propagation. `PYTHON_BIN` selects the existing environment;
  `DRY_RUN=1` prints commands and `FORCE=1` reruns cached configurations.
  `summarize_sweep.py` preserves selection by validation AUROC, then macro F1,
  then lower threshold. Training remains in its existing baseline command.

All recipes accept extra capability arguments. Use a fresh output identity
when changing scientific settings. These recipes were syntax-checked and the
scoring, UID and sweep paths tested with fixtures; no live inference, training
or active-release publication was launched for this refactor.
