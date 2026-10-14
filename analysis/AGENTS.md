# Analysis

- Reusable prediction analyses expose explicit arguments for input runs, tasks, cohorts, comparisons and output paths through their own capability commands. Keep mode-specific analyses separate when unification would obscure their behavior.
- Put specific studies and parameter sweeps in Bash scripts that call those shared capability commands; do not add a Python executable with hardcoded experiment settings for each study.
- Reuse shared retrieval, selection, metrics and trace readers. Keep reusable implementations with their existing owners; do not copy the prediction pipeline into analysis code.
- Analysis-only modes consume saved artifacts without model calls, provider probes or rerunning inference. Write new derived reports under `outputs/analysis/<domain>/<study-id>/`; preserve completed runs and source provenance.
- Before retiring an existing analysis script, preserve its useful behavior through the appropriate capability CLI arguments and a Bash recipe, and verify equivalent outputs on a focused fixture.
- Evidence-library construction and independent scientific analyses retain their own shared implementations and scientific contracts; the prediction entry point does not absorb those pipelines.

- Prediction metadata uses one shared batch/FINAL contract with named versions and query UIDs, read through `predict.harnesses.branches.manifests`. Paths locate runtime inputs; they and content fingerprints are not experiment identity. Per-query state belongs in existing checkpoints, never duplicated manifests. Preserve sparse scientific overrides and reject incompatible or unknown identities before reuse or inference.
- Explicitly approved metadata migrations may convert historical prediction headers and operational indexes. Preserve frozen scientific payloads and leave compressed FINAL bundles untouched unless a push is requested.
