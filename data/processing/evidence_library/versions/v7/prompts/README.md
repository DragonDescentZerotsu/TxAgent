# Starling evidence-library LLM prompts

This directory is the editable source of truth for prompts used by the
API-driven Starling evidence-library construction jobs. Prompt versions remain
declared by their consumer modules, and future runs record the prompt file hash.
Changing a prompt does not rewrite an existing frozen mapping or artifact.

## Construction and audit prompts

| Process | Prompt assets |
|---|---|
| Scalar measurement and unit extraction | `measurement_resolution/` |
| Auxiliary assay-context and species canonicalization | `auxiliary_canonicalization/` |
| Endpoint canonicalization | `endpoint_canonicalization/` |
| Measurement reference-scope classification | `reference_semantics/` |
| Oral study-context extraction | `study_context/` |
| Final endpoint pruning | `endpoint_pruning/` |

Completed SMILES review, record-collapse, semantic aggregation, and collapsed-
record informativeness prompts are not active construction inputs and are not
kept in this release. Bucket-informativeness scoring is version-independent and
uses `data/processing/evidence_library/prompts/`.

Downstream retrieval-agent and benchmark reasoning prompts remain task-local;
they are not evidence-library construction prompts and are intentionally outside
this directory.
