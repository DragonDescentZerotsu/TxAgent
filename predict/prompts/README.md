# Prediction prompts

All six tasks have `tasks/<task>/profiles.yaml`. Each contains the default and
historical native profiles, their schemas/instructions, and the canonical flat
task contract. Profile names describe scientific prompt contracts; they are
independent of benchmark storage.

`flat/final_full_flat/` contains the default system/user Jinja templates, level
descriptions and provenance. `query_prior/` contains shared native/generic prior
templates and generic prior instructions. `variants/` holds deliberate prompt
experiments, with task overrides rather than copied six-task definitions.

`__init__.py` loads these assets and renders prior messages; the flat harness
uses the same loader. Rendering, evidence processing, validation and inference
remain Python. Prompt prose and output schemas live here. Retrieval-scoring,
semantic-ranking and evidence-construction prompts stay with their capabilities.

Historical flat bundles are read explicitly from `archive/branches/prompts/`.
They retain their frozen contracts; there is no old-path symlink directory.

## Frozen paper baseline

`variants/paper_baseline_v1/` preserves the paper baseline system/user templates,
six native task/output contracts and the two TDC overrides (Ames and Carcinogens).
It retains task-specific prediction fields, confidence, supportive/contradictory
card IDs, prediction basis, bounded claims/gaps and decision summaries. These
are distinct from the current FINAL `claims` / `final_prediction` contract.
The source/asset hashes are recorded in the bundle's `provenance.json`.

`paper_baseline.py` renders frozen `prompt_payload` objects and supplies response
validation to the existing `predict.utils.response` functions. Exact field
deduplication is retained; older full-field payloads remain readable. It neither
selects evidence nor regenerates priors. Historical task definitions are explicit
and do not inherit the current default profiles.

From the repository root, verify a restored original batch without model calls:

```bash
python -m predict.harnesses.branches --verify-paper-baseline /path/to/frozen/batch
```

The command reads `experiment_manifest.json`, `runs/<task>/query_*/` and the
saved task summaries. It checks request/manifest bindings, exact re-rendering,
record aliases, raw response parsing, native-label validation, saved decisions
and metrics. It prints a report and does not modify the batch. This verifies
saved artifacts; it does not rebuild candidates or independently re-audit gold
labels, providers or costs.

This bundle is not yet an executable `--prompt` choice for fresh FINAL runs.
The next integration step must connect the original selection/payload preparation
and inference settings to the shared runner. Do not substitute the current named
cache, prior generation, output schema or high-reasoning default for the paper
configuration. Approved retrieval-policy changes require new prepared inputs;
old results retain their original protocol.
