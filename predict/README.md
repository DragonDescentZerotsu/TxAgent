# Prediction package

`predict/` owns FINAL inference, retrieval, provider transport, baselines and
traces for BBB, Oral, Skin, Ames, DILI and Carcinogens. Context selection lives
in `optimization/`; data publication lives in `data/processing/`.

```bash
python -m predict.harnesses.branches --organization flat --help
python -m predict.harnesses.branches.matrix --help
python -m predict.harnesses.branches.six_task_query_priors --help
python -m predict.harnesses.branches.tdc_query_priors --help
```

The public CLI defaults to `final-full-flat`. Direct-only and mixed inference
use frozen selection manifests with this same execution path. It preserves UID
order and per-level budgets, renders one FINAL request per query, and writes
resumable checkpoints, predictions, metrics and traces. Native query priors
read shared task profiles and query-prior templates from `predict/prompts/`; a query-only batch ends when its
prior is complete. Historical renderers remain readable for frozen replay.

New runs use Gold-v1 or canonical TDC. The single TDC configuration is
`data/gold_labels/TDC/manifest.json`: Oral/Skin use the frozen scaffold cohorts;
BBB/Ames/DILI/Carcinogens use frozen original-source cohorts. There is no TDC
version option. Historical requests keep their recorded cohort identities.
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

The [conditioned baseline recipe](../experiments/baselines/conditioned.sh) runs
the shared Morgan KNN, MiniMol KNN and MiniMol head commands. Its
[settings and output contract](baselines/minimol/README.md#conditioned-baseline-recipe)
apply to supplied task splits, including Oral; there is no separate Oral runner.

Raw Joseph runs: `outputs/paper/assay_transfer_harness/joseph/`.
Derived reports/TSVs/provenance: `outputs/analysis/<domain>/<study-id>/`.
See [../docs/finalize.md](../docs/finalize.md) for publication and
[../docs/results.md](../docs/results.md) for selected scores and caveats.
Historical assets and relocation metadata live in [../archive/](../archive/).

Manual evidence uses `--uids` with one query/level or `--uids-file` with a UID TSV
or optimizer manifest. See [the selection contract and recipes](../docs/experiment_code.md#physical-uid-selections).
SDK construction is centralized in `api_client/transport.py`; `api_client/pool.py`
owns provider routing and capacity.

## Saved prediction metadata

`manifest.json` is one batch contract (`prediction_batch.v2`), shared by its query runs. It records the model and prompt versions, effective selection parameters, named input releases, ordered query UIDs, and any genuine per-query scientific overrides. Read it with `predict.harnesses.branches.manifests.read_record`; read an effective query contract with `read_run`. Query folders contain retrieval, requests, responses, predictions and `stage_resume.json`, with no copied query manifest. Actual served model/provider stays in the response and trace.

Inputs are resolved using the named releases, explicit CLI roots and local `inputs/` snapshots. Paths and experiment fingerprints are not saved as scientific identity. Query UIDs and named versions must agree before reuse; unknown old identities stay empty and cannot silently authorize inference. The metadata-only migration command is `python -m predict.harnesses.branches.migrate_manifests --report /tmp/manifest-migration.jsonl`; review its dry run, then repeat with `--apply`. The same report lets an interrupted migration resume. It includes readable FINAL arms and excludes `final/git_bundle/`.
