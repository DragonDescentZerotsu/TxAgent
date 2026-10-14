# Inference

This owner reads benchmark queries and frozen selections, renders model requests, and writes resumable predictions. Paths here are relative to `predict/`.

- `harnesses/branches/__main__.py` routes `python -m predict.harnesses.branches --organization flat`; `harnesses/branches/flat.py` projects selected evidence and renders prompts.
- `harnesses/branches/runner.py`, `runtime.py` and `scheduler.py` in that directory own batch preparation, execution, retries and checkpoints; `matrix.py` launches matrices.
- `prompts/flat/final_full_flat/` is the default prediction bundle; `prompts/tasks/<task>/profiles.yaml` and `prompts/query_prior/` own all six task profiles and prior templates; `harnesses/branches/tasks/` and `tasks/` own task adapters and native prior behavior.
- `harnesses/branches/six_task_query_priors.py` and `tdc_query_priors.py` prepare prior overlays; `retrieval/assay_reranking/` resolves and hydrates cached evidence.
- `api_client/` owns model transport; `tools/` consumes chemistry-service schemas; `baselines/` owns comparisons and `traces/` owns trace I/O and the viewer.
- Batch outputs live under repository `outputs/paper/assay_transfer_harness/joseph/`; `traces/io.py` defaults recorder files to repository `outputs/paper/live/`. Honor explicit batch/trace roots and keep per-query `retrieval.json`, request/prompt provenance and traces aligned.

- Keep the FINAL flat CLI, native query priors, request replay, checkpoint resume, traces and metrics compatible.
- Parameterize experiments through existing harness entry points; keep full/flat orchestration, query-prior generation and offline analysis separate capabilities. Evidence-consuming predictors accept ordered physical UIDs through the common selection reader and reject membership/disjointness violations before inference.
- Bash experiment scripts specify CLI arguments and run identities; they must not duplicate prediction or analysis logic. Analysis-only modes read saved artifacts without model calls or provider probes.
- Conditioned baselines share `baselines/conditioned_knn.py`, the Morgan/MiniMol KNN runners and the MiniMol head runner. Use `experiments/baselines/conditioned.sh` for concrete runs rather than task-specific Python copies. Keep condition matching and fixed-epoch head settings explicit in recipes; retain row identities, subgroup metrics, insufficient-neighbor coverage and train-derived categories. Generic defaults and train-only CV selection remain separate contracts.
- Use the shared provider transport and provider JSON capacity; fail on incompatible query, selection or input identities before a call.
- `api_client/transport.py` constructs model SDK clients and reads credentials; `client.py` handles model requests; `pool.py` loads provider profiles and manages routing/concurrency. Select DGX/OpenRouter through provider JSONs or the transport's explicit provider/base URL arguments.
- Molecular-tool prefetch uses `tools/client.py`'s `retry_failed_tool_calls`: retry failed results once, preserve successful results and ordering, and share it between batch prefetch and direct query-prior property calls.
- Preserve model-visible evidence order, UID budgets and frozen prompt contracts. Historical renderers support artifact replay.
- Follow [README.md](README.md) and the retrieval owner's scientific contracts.

- Prediction metadata uses one shared batch/FINAL contract with named versions and query UIDs, read through `predict.harnesses.branches.manifests`. Paths locate runtime inputs; they and content fingerprints are not experiment identity. Per-query state belongs in existing checkpoints, never duplicated manifests. Preserve sparse scientific overrides and reject incompatible or unknown identities before reuse or inference.
- Explicitly approved metadata migrations may convert historical prediction headers and operational indexes. Preserve frozen scientific payloads and leave compressed FINAL bundles untouched unless a push is requested.

TDC reads one unversioned configuration at `data/gold_labels/TDC/manifest.json` and exposes no label-version argument. Optimizer UID files may bind the configured full-molecule L1 card cache; validate its manifest and hydrate its physical members. Frozen parent-grouped rankings remain a distinct input.
