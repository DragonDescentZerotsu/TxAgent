# Prediction package

`predict/` owns FINAL inference, retrieval, provider transport, baselines and
traces for BBB, Oral, Skin, Ames, DILI and Carcinogens. Context selection lives
in `optimization/`; data publication lives in `data/processing/`.

The defaults below describe this branch's FINAL workflow. The frozen paper's
naive LLM baseline used explicit 50/50 Neighbor-fill, a pinned earlier prompt
adapter and parent-disjoint indirect evidence. Its explicit paper mode rebuilds candidates from restored source batches and
uses the shared model client; selecting today's default cache/prompt does not reproduce it.
See [paper results and migration limits](../docs/results.md) and the
[paper preparation, execution and offline verification](prompts/README.md#frozen-paper-baseline).

```bash
python -m predict.harnesses.branches --help
python -m predict.harnesses.branches.matrix --help
python -m predict.harnesses.branches.six_task_query_priors --help
python -m predict.harnesses.branches.tdc_query_priors --help
```

Prediction composes an immutable retrieval cache, explicit ranking/selection policies,
a prompt artifact and model settings. There is one flat execution path; numbered
harness selectors are not prediction modes. `--prompt` defaults to `final_full_flat`.
`--retrieval-cache` defaults to the named `scaffold_disjoint_parent100_v2` cache set,
covering all six Gold tasks and all six canonical TDC tasks. Cached level
metadata owns score availability, disjointness and supported levels. A schema
version describes file structure only; it never selects a prediction algorithm.

Use `--reranking morgan|assay-transfer` and evidence budgets for cached ranking,
or `--uids-file` for an ordered UID TSV or optimizer manifest. Matrix runs use
the same resolver and accept `--layout` and `--query-prior`. The reader rejects
foreign UIDs and disjointness violations before model calls. Prompt releases do
not change the candidate pool. The task/split bundle must have a current verification
receipt before preparation. Historical numbered cache recipes remain available
only through the historical decoder; active commands reject `--harness-version`,
`--organization` and `--assay-transfer-cache`.

Direct L1 is always scaffold-disjoint. Indirect L2+ offers two named cache sets
under `predict/retrieval/assay_reranking/flat_v5_manifests/`:

| Catalog | L1 | L2+ |
| --- | --- | --- |
| `scaffold_disjoint_parent100_v2/catalog.json` (default) | Scaffold-disjoint | Scaffold-disjoint |
| `parent_disjoint_parent100_v1/catalog.json` | Same scaffold-disjoint binding | Parent-disjoint |

Select a catalog with `--retrieval-cache`. Published score coverage remains
explicit: the parent-disjoint Gold Ames/DILI/Carcinogens indirect bindings are
Morgan-only, while the scaffold-disjoint successors have assay-transfer scores.
Assay-transfer contrastive selection and its grid sweeps are retired; ordinary
assay-transfer ranking and frozen optimizer selections use the shared launcher.

```bash
python -m predict.harnesses.branches --task dili --benchmark tdc \
  --evaluation-subset test --prompt final_full_flat --uids-file SELECTION \
  --query-prior none --batch-root FRESH_OUTPUT --batch-id STUDY --prepare-only
```

Native query priors read shared task profiles and templates from `predict/prompts/`.
Historical renderers remain readable for frozen replay. Reusing a batch identity
restores its pinned cache and scientific settings; incompatible changes require a
new identity. Missing historical identities cannot inherit the current defaults.

New runs use Gold-v1 or canonical TDC. The single TDC configuration is
`data/gold_labels/TDC/manifest.json`: Oral/Skin use the frozen scaffold cohorts;
BBB/Ames/DILI/Carcinogens use frozen original-source cohorts. There is no TDC
version option. Historical requests keep their recorded cohort identities.
Internal cache bindings, storage paths, receipts and newly written batch metadata
also use `tdc`; historical names remain in audit records. Prepared batch bindings
can be migrated with `migrate_manifests --cache-catalog CATALOG --roots ROOT --report REPORT`.
The default is a dry run; `--apply` requires matching frozen query and cache payload
identities and preserves completed batches.
Full batches launch directly in throughput mode with explicit `--parallelism`.
The selected provider JSON owns launcher-local `max_inflight`. Joseph defaults
to fixed `deepseek/deepseek-v4-flash-0731`, automatic OpenRouter provider routing,
and high reasoning effort. Changed model or route requires a new run identity;
record the actual served model and provider. `--prepare-only` uses no model calls
or provider probes. Incompatible input identities must fail before inference.

The runner/runtime/scheduler own preparation, retries and completion; `flat.py`
owns selected-record projection and exact rendering. `api_client/` supplies the
shared transport and `tools/` supplies chemistry-tool access. Inference imports
canonical data contracts and never imports retired executable namespaces.

See [retrieval/assay_reranking/CACHE_LAYOUT.md](retrieval/assay_reranking/CACHE_LAYOUT.md).
Development uses manifests, receipts and indexed reads; full hashes and SQLite
integrity checks belong to publication, restore or explicit verification.
Completed artifacts are immutable.

```bash
python -m predict.baselines.direct_l1_knn --help
python -m predict.baselines.selected_direct_knn --help
python -m predict.baselines.structure_knn.run --help
python -m predict.baselines.minimol.run_embedding_knn --help
python -m predict.baselines.minimol.run_bioavailability_ma --help
bash predict/traces/viewer/start_viewer.sh 8776
```

Selected-card kNN supports `--benchmark gold_v1|tdc` using the matching labels
and frozen selection/cache identities. `--neighbor-order assay-rank` requires
scores for the selected contexts; `policy-relevance` uses the frozen objective
and its missing-score policy. The benchmark does not choose the voting method.

The [conditioned baseline recipe](../experiments/baselines/conditioned.sh) runs
the shared Morgan KNN, MiniMol KNN and MiniMol head commands. Its
[settings and output contract](baselines/minimol/README.md#paper-baseline-recipe)
apply to supplied task splits, including Oral; there is no separate Oral runner.

New raw runs: `outputs/predictions/<harness>/<prompt>/<run-name>/<benchmark>/<task>/<split>/<profile>/`.
Supply `--run-name semantic-mixed-top50` and optionally `--output-root PATH` to
the flat or matrix launcher. The harness and selected prompt determine
their namespaces; each batch keeps its existing manifest, query snapshots and
metrics. Query folders preserve the actual selected retrieval and exact request.
Default launcher traces stay inside the named run. Explicit trace roots and
low-level batch-root overrides remain supported.

Historical raw locators resolve through `outputs/predictions/relocations.json`.
Published FINAL stays under `outputs/paper/assay_transfer_harness/joseph/final/`.
To inventory an authorized relocation, run
`python -m predict.harnesses.branches.migrate_manifests --relocate-runs --report /tmp/output-relocations.json`;
repeat with `--apply` to use that reviewed map. `--classifications JSON` supplies
explicit identities for unresolved folders. Unknown identities and unresolved
name collisions stay in place. Moves preserve run UIDs and frozen payload bytes.
See the [migration report](../docs/look/prediction_output_migration.md) for the
completed moves and remaining manual classifications.

Resume checks the currently requested query prior against the saved validated
content. A mismatch raises an error without replacing checkpoints or deleting
completed results; matching source run UIDs do not bypass this comparison.
Query-prior generation validates required fields and allowed labels before saving
an output with its successful validation receipt. Invalid outputs are not
published; exhausted generation retries raise an error. Reuse checks the saved
receipt without repeating schema validation or computing a prior-validation hash.
Missing or failed receipts are rejected. Historical priors need a one-time receipt
backfill before reuse; prompt and response content must be preserved.
Derived reports/TSVs/provenance: `outputs/analysis/<domain>/<study-id>/`.
See [../docs/finalize.md](../docs/finalize.md) for publication and
[../docs/results.md](../docs/results.md) for selected scores and caveats.
Historical assets and relocation metadata live in [../archive/](../archive/).

Manual evidence uses `--uids` with one query/level or `--uids-file` with a UID TSV
or optimizer manifest. See [the selection contract and recipes](../docs/experiment_code.md#physical-uid-selections).
BBB L2+ evidence excludes prediction-only records by default. The shared task
predicate retains measured experimental assays and mixed experimental/predicted
records. Automatic ranked panels skip excluded records before applying limits;
manual/frozen panels containing them fail before inference. New prepared BBB
batches record `bbb_prediction_only_excluded_l2plus.v2`; older unmarked prepared
batches require a fresh run identity. Explicit saved-request replay retains its
original request contract. Frozen L1 cards and other tasks retain their policies.
SDK construction is centralized in `api_client/transport.py`; `api_client/pool.py`
owns provider routing and capacity.

## Saved prediction metadata

`manifest.json` is one batch contract (`prediction_batch.v2`), shared by its query runs. It records the model and prompt versions, effective selection parameters, named input releases, ordered query UIDs, and any genuine per-query scientific overrides. Read it with `predict.harnesses.branches.manifests.read_record`; read an effective query contract with `read_run`. Query folders contain retrieval, requests, responses, predictions and `stage_resume.json`, with no copied query manifest. Actual served model/provider stays in the response and trace.

Inputs are resolved using the named releases, explicit CLI roots and local `inputs/` snapshots. Paths and experiment fingerprints are not saved as scientific identity. Query UIDs and named versions must agree before reuse; unknown old identities stay empty and cannot silently authorize inference. The metadata-only migration command is `python -m predict.harnesses.branches.migrate_manifests --report /tmp/manifest-migration.jsonl`; review its dry run, then repeat with `--apply`. The same report lets an interrupted migration resume. It includes readable FINAL arms and excludes `final/git_bundle/`.
