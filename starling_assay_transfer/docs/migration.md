# Canonical migration

This document describes the canonical migration in the original `/vast/projects/myatskar/design-documents/joseph/starling_assay_transfer` checkout. The TxAgent copy retains the active package, tests and documentation; archives and generated runs referenced here remain in that original checkout.

The historical setup is preserved under the source checkout's `archive/historical-20261013/`. The earlier cleanup archive remains under its `archive/unused-20261013/`. The archive manifests record original locations and preserve symlinks without traversing their external targets. Original code includes uncommitted changes and existing datasets remain scientifically unchanged.

The active package is `assay_transfer`. Historical commands such as `assay_transfer.record_level.v*` and `assay_transfer.context_conditioned.v*` are retired. The original migration did not edit downstream repositories. Training/evaluation callers should select the new task and family path explicitly, preserve the run's pinned input manifest and effective contract, and verify the task receipt's scope.

The canonical builders initialize existing methods on the current V10 source snapshot. They create successors rather than exact copies of historical frozen releases. Gold-v1 target calibration uses official training and validation records; test labels can participate only in the declared eligibility gates. Molecule identity uses the shipped fragment-parent normalizer and pinned RDKit environment.

The Arrow schema is frozen inside the package, so no old Parquet release is needed to construct new outputs. `prompt`, `completion`, `target_a`, `target_b`, and `metadata.soft_targets` remain available to training consumers. New fields identify the family and effective contract. Pair identities and paths belong to the canonical successor and must not be mixed with historical release identifiers.

BBB and Oral exclude L5 from record construction, record evaluation and indirect context. Historical L5 datasets and copy identities remain archived; no L5 copy rows are constructed in the canonical setup.

Representative smoke construction uses entire selected bucket-level groups plus a deterministic sample of Gold records for graph generation. Gold target calibration still uses the full pinned training/validation population. Smoke mixture shortfalls are reported, never promoted to a full release. Full construction requires exact augmentation budgets after prompt cleanup; duplicate/conflicting prompt removals are recorded separately.

Version 1.2.0 organizes the active package by scientific stage. Old helper module paths are retired. The builders expose `pool`, `sample`, `augment`, `targets`, `dataset`, `verify`, `upload`, and `all`. Output directories follow those stages; `all` stops after verification. Source and target semantics are preserved, including fitting binary targets from the full sampled pool before mixture selection. New generated runs live in `runs/`; completed workspaces move to `archive/task-workspaces-20261013/`. `.agent-tasks/` is for current progress notes and small logs.
