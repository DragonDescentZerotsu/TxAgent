# ChEMBL tool code

This tree contains resident tool implementations and historical task-native
experiments. Current benchmark labels, evidence levels, and assay-transfer
eligibility are owned by the published releases under `data/gold_labels/` and
`data/evidence_libraries/`, with builders in `data/processing/evidence_library/`.
Do not infer a current scientific level or evaluation split from an old native
runner or an archived experiment. Versioned experiment reports and Git history
retain the earlier task ontologies and run instructions.

Prediction code lives in `predict/`; it must not import these historical task
runners. Changes to active evidence classification belong in a reviewed
release-owned mapping, not in an agent instruction file.
