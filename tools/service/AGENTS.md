# Chemistry service

This owner computes reusable molecule and molecule-pair evidence for prediction. Paths here are relative to `tools/service/`.

- `app.py` exposes FastAPI routes; `registry.py` owns tool lifecycle, batching and single-flight coordination; `schemas.py` defines request/response contracts.
- `tools/rdkit_properties.py`, `tools/properties_compare.py` and `tools/mmp_structure_compare.py` implement molecule properties, property differences and structural comparisons.
- `molgpka_predictor.py` retains resident pKa models; `cache.py` caches versioned results; `config.py` and `runtime.py` control workers and native-library threads.
- `../../predict/tools/` consumes these outputs; repository `tests/tools/service/` contains focused checks. Retrieval selection belongs to prediction/retrieval owners.

- Keep chemistry tools deterministic and independent of paid model clients.
- Preserve request/response schemas used by `predict/tools/`; validate supported behavior with existing focused service tests.
- Do not embed task labels, credentials or inference prompt policy in tool responses.
- Keep reusable deterministic chemistry operations independent. Experiment recipes call existing capabilities; do not add task-specific copies or force chemistry services into the prediction/scoring CLI.
