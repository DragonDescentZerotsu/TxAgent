# Release integration status

This branch starts from Joseph `73bddf2e` and integrates the approved local
implementation preserved in source snapshot `41403cf4`. Main remains the final
publication branch. This is an incomplete integration, not a replacement for
the published main branch or a new set of scientific results.

## First port: indirect ML and RASAR

The existing implementation has moved into the canonical prediction package:

| Capability | Entrypoint or owner |
| --- | --- |
| MiniMol encoder fine-tuning and indirect multitask training | `python -m predict.baselines.minimol.run_multitask` |
| Main/auxiliary targets and heldout-parent filtering | `predict/baselines/minimol/indirect_data.py` |
| Differentiable MiniMol encoder and graph cache | `predict/baselines/minimol/finetune_runtime.py` |
| RASAR direct and direct+indirect features | `predict/baselines/retrieval_features.py` |
| RASAR fitting and validation selection | `python -m predict.baselines.retrieval_forest` |

These modules reuse the existing MiniMol head, metrics, condition vocabulary,
fingerprints and JSON utilities. No root `baselines/` alias or duplicate provider,
selector, runner or plotter was added. The five ported modules retain their
original algorithm bodies; imports now resolve to the canonical package.

Use the original input manifests, explicit `--data-dir`, auxiliary target files
and model checkpoint. These commands do not select a new Gold/TDC cohort through
the collaborator's CURRENT pointers. Fixed train/valid/test rows, raw molecular
inputs, labels and conditions remain the acceptance reference.

MiniMol selects checkpoints and auxiliary weight using validation BCE, then
selects an ensemble threshold using validation Macro-F1. RASAR keeps the
three direct neighbors' six features; the indirect variant appends one
target/similarity pair per auxiliary bucket. Auxiliary pools exclude the
valid/test parent-connectivity union and permit shared scaffolds. Existing main
split overlaps are preserved only with the original explicit option.

Shared conditioned-neighbor selection now preserves the local reference's first
source row for duplicate molecule-condition labels and supports the existing
`same_condition_then_null_then_all` fallback. This is an intentional difference
from Joseph's highest-ranked duplicate behavior, not a silent directory move.
The implementation remains shared by Morgan KNN, embedding KNN and RASAR.

Validation uses the transferred baseline tests, existing KNN/condition tests,
prediction import-boundary checks, CLI help, and differential checks against the
preserved local source: 50 tests passed, both CLI help commands succeeded,
18,656 distinct benchmark structures had identical parent keys, and 2,700
seeded condition-selection comparisons had no differences. The five ported
module bodies match the immutable source snapshot after import relocation.
Molecular-key comparison covers the current conditioned
and TDC scaffold cohorts (including the Skin/Carcinogens seed-42 overrides),
not the entire auxiliary evidence universe. No formal experiment, model-service
restart or frozen-result rewrite is part of this port.

## Remaining acceptance work

- Integrate the corrected local identity policies and DILI/Carcinogens
  query-time parent refinement into retrieval. Retain the agreed permissive
  multicomponent behavior and unchanged scaffold definitions.
- Bind the exact local benchmark manifests, evidence pools, budgets, card
  identity/deduplication, prompts and cached priors; verify selected evidence and
  rendered requests before using any new pipeline for results.
- Complete MiniMol head/runtime option parity, portable runtime/checkpoint
  discovery and whole auxiliary-pool verification. Unit tests and benchmark
  parent-key parity do not certify full encoder training or prepared caches.
- Integrate the local shared budgeted runner, provider admission/retry behavior,
  result registry, exporters and frozen-replay entrypoints without duplicating
  maintained implementations. Preserve progressive capabilities still used by
  those workflows until dependencies are removed deliberately.
- Reconcile main's history and privacy cleanup. Preserve the BBB/DILI handoff
  branch until its independent work and provenance references are accounted for.
- Validate the completed integration and merge normally into main; no force
  replacement of main or automatic adoption of Joseph's defaults is authorized
  by the structural migration itself.
