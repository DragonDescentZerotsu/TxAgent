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
