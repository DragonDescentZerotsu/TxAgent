# A small core for many experiments

[发布整合状态（中文）](release_refactor.md)记录交接分支、已验收范围、保留的科学规则和待补依赖。

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
| Paper LLM baseline | `predict/harnesses/branches/paper_baseline.py`, `predict/prompts/paper_baseline.py` | Restored paper inputs → new candidate selection, portable prepared-input bundles, embedded frozen query priors (no regeneration), protocol-bound messages, shared API execution, resumable native predictions and offline verification |
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

The paper baseline adapter retains main's prompt templates, native labels and
prepared/output format. `legacy_paper` additionally enforces exact original inputs;
`murcko` permits evidence and message changes under the revised retrieval rules.
Evidence FINAL retains its own prompt and execution contract. The two paths share
provider transport, response validation and identity-filtering code, but source
contracts and frozen cache contents still determine which rules apply; see the
[retrieval contract](../predict/retrieval/assay_reranking/CACHE_LAYOUT.md#current-exclusion-rules).
Substituting FINAL templates or checkpoints would change the paper experiment.
The paper-specific modules do not introduce another SDK client, task-specific runner or prior generator. They support the manuscript's
None / Direct50 / Direct50+Indirect50 settings; matched Direct-3 LLM is not a
maintained migration target.

The local `experiments/retrieval_cache/molecular_policy.sh` and
`experiments/evidence_library/score_skin_gold_voters.sh` recipes locate this
checkout from their script paths and default to `python` in the active environment.
Use `TXAGENT_PYTHON` and `SCORER_PYTHON`, respectively, for an explicit interpreter;
set `HF_HOME` when reusing an existing model cache.

The distributed recipes under `experiments/retrieval_cache/` retain their pinned
historical studies and scoring parameters. They are existing cache-maintenance
workflows, not additional paper baseline experiments. Machine locations are
supplied by the operator:

| Recipe | Required deployment settings | Optional settings |
| --- | --- | --- |
| `molecular_policy_weighted.sh prepare/collect TASK` | Original study inputs and dispatch receipts | `TXAGENT_STUDY` (repository-relative), `TXAGENT_PYTHON` (default `python`), exported `HF_HOME` |
| `molecular_policy_weighted.sh node002` | `TXAGENT_REMOTE`: absolute remote **checkout** path | `TXAGENT_REMOTE_HOST` (default `node002`), `TXAGENT_REMOTE_PYTHON` (default `python`), `TXAGENT_REMOTE_HF_HOME`; remote cache otherwise uses its own environment/default |
| `molecular_policy_weighted.sh dgx018/dgx009` | `TXAGENT_ALLOCATION`: current Slurm allocation ID | Local Python/cache settings above; original shard ranges and memory fractions stay fixed |
| `scaffold_disjoint_node002_{stage,score,collect}.sh` | `TXAGENT_REMOTE`: absolute **staging root**, containing `TxAgent/`; original run manifest, staging table and frozen scorer archive | `TXAGENT_RUN` (repository-relative), `TXAGENT_PYTHON`, `TXAGENT_REMOTE_HOST` for stage/collect; score uses exported `HF_HOME` or `<staging root>/hf_home` |

Set the same staging root for all three scaffold scripts; run `score` on the
remote machine. The weighted recipe's remote path includes the checkout itself,
whereas the scaffold trio appends `TxAgent/`. Local scripts locate the repository
from their own paths. Explicit interpreter/cache settings can reproduce the old
deployment without committing personal paths. These scripts still require their
frozen study inputs; the path cleanup does not regenerate or publish a cache.

V9 direct-cache reuse defaults to the historical `therapeutic-tuning/results/`
tree beside this checkout. Use its existing `--reuse-root` / `--old-root` option
after relocation, or `--no-reuse` for a new build without inherited scores.
Reuse still requires the exact input, prompt and model bindings.

## Canonical semantic utility scorer

`optimization.semantic_buckets.bucket_scoring` owns batched scoring, sorted
sliding rescoring and replica aggregation. The [semantic bucket guide](../optimization/semantic_buckets/README.md#score)
is the single reference for descriptors, multi-pass behavior, locked anchors,
chain/replica controls, seeded refreshes and publication. The recipes under
`experiments/scoring/` supply concrete settings to that shared implementation.

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

Use a fresh output identity when changing scientific settings. Recipe-specific
environment variables and forwarded arguments are documented by their capability
guides. Scoring, UID and sweep checks used fixtures; the later paper-baseline
replay, limited retraining and live smoke are recorded in the
[integration acceptance status](release_refactor.md#baseline-配置与验收).
