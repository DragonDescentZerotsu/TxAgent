# Canonical builder verification

This document records historical validation in the original
`/vast/projects/myatskar/design-documents/joseph/starling_assay_transfer` checkout.
All `runs/`, `archive/` and `.agent-tasks/` paths below refer to that checkout and
are not included in this copy. Validation of the TxAgent copy is recorded
separately in [its validation receipt](../provenance/validation.json).

## Scientific pipeline refactor, version 1.2.0

The independently installed wheel passed 25 regression checks and 15 representative builds: all six tasks with both builders in scaffold mode, BBB with both builders in parent mode, and Gold-only BBB context construction. All 45 emitted train/validation/test splits match the prior scientific rows exactly, including their schema, prompts, ordering and targets. These are smoke checks; full scientific dataset regeneration and remote publication were not performed.

Sampling and augmentation now emit unscored pairs; the targets stage computes probabilities afterward, preserving the full sampled numeric fit population. Split, IID/OOD evaluation and filtering prepare the pool. Separate record and context builders then run sampling, augmentation, targets, Jinja dataset creation and verification. Upload is an explicit command requiring a verified full release and a supplied destination repository. Mocked tests cover upload, tamper rejection and smoke rejection; no remote upload occurred.

Active sources match the independently installed wheel. All 183 active functions are at most 54 lines. Validation status, exact row comparisons, source proof and completion evidence are in `runs/canonical-refactor-20261013/`, using `installed-v2/`, `validated-*`, `validated-status.json`, `semantic-comparison.json` and `source-verification-v2.json`. Earlier provisional checks remain separately labeled.

Completed task workspaces are preserved under `archive/task-workspaces-20261013/`. Historical `.agent-tasks/<name>` references below now map to `archive/task-workspaces-20261013/<name>`; original receipts were not rewritten. The archive inventory covers 109 roots and 7,105 files, totaling 25,770,879,301 bytes. Current `.agent-tasks/` contains only this task's notes and small diagnostic logs.

The reports below certify their named historical implementations, rather than version 1.2.0.

## Gold-derived holdouts, version 1.1.1

Removed the legacy BBB parent lists, fixed excluded buckets, and test-to-validation copies. All record validation/test memberships now derive from pinned Gold splits; OOD bucket assignments derive from pinned V10 records. The frozen legacy asset is preserved in `archive/legacy-holdouts-20261013/`.

22 regression checks passed in the independently installed package. Freshly pinned current V10/Gold inputs passed split audits for all six tasks. BBB record and augmented context successor builds passed in both parent and scaffold modes; a Gold-only BBB context build also passed. All receipts are representative (`scope: smoke`); full scientific regeneration and remote publication were not performed.

Active sources match the installed wheel. Every active function is at most 60 lines. Record verification rejects validation/test queries outside the corresponding pinned Gold parent cohort. Current pointers still match the pinned releases. Evidence and completion receipts are in `.agent-tasks/v10-gold-holdouts-20261013/`.

Earlier verification below describes prior implementations and retains their original receipts; it does not certify version 1.1.1.

## Separate builders, version 1.1.0

The independently installed wheel passed 21 regression checks and 33 representative builds: all six tasks, both builders, both disjointness modes; six Gold-only builds; and DILI ratios 0, 0.5, and 2. Active sources match the tested wheel, and every active function is at most 60 lines.

Default scaffold-mode coverage:

| Task | Builder | Train pairs | Validation pairs | Test pairs | Indirect budget shortfall |
| --- | --- | ---: | ---: | ---: | ---: |
| Ames | context | 49,152 | 1,175 | 1,106 | 0 |
| Ames | record | 92,448 | 0 | 0 | — |
| BBB_Martins | context | 49,152 | 907 | 908 | 0 |
| BBB_Martins | record | 250,102 | 11,993 | 2,289 | — |
| Bioavailability_Ma | context | 14,296 | 936 | 944 | 10280 |
| Bioavailability_Ma | record | 16,688 | 14,000 | 16,200 | — |
| Carcinogens | context | 49,152 | 918 | 920 | 0 |
| Carcinogens | record | 180,192 | 0 | 0 | — |
| DILI | context | 24,576 | 1,055 | 1,053 | 0 |
| DILI | record | 66,862 | 0 | 0 | — |
| Skin_Reaction | context | 42,962 | 900 | 900 | 6190 |
| Skin_Reaction | record | 13,632 | 614 | 250 | — |

These receipts have `scope: smoke`. Representative Oral/Skin augmentation pools can be smaller than the requested default 1:1 budget; coverage records the actual ratio and exclusions. Full builds reject budget shortfalls. No full scientific release or remote publication was performed.

Gold-only pinning/building was checked using a source root with no evidence library. Context evaluation stayed direct Gold only; record and indirect training used task-wide holdout checks. The record builders used identical heldout identity files in both disjointness modes for every task. Regression checks include cross-level leakage, retained holdouts, refitted calibration, frozen OOD selections, alternate ratios, matched resume, and incompatible-mode rejection.

Current receipts, hashes, exclusion audits, logs, and completion evidence are under `.agent-tasks/separate-builders-20261013/`, using the `validated-v2-*` and `options-v2-*` outputs. Earlier provisional checks remain separate. Archives and previous receipts are unchanged.

## Earlier cutover verification, version 1.0.0

The standalone installed package passed representative builds for all six tasks and both families. These checks have `scope: smoke`; full dataset generation and remote publication were not run.

| Task | Family | Training pairs | Validation pairs | Test pairs |
| --- | --- | ---: | ---: | ---: |
| Ames | context | 49,039 | 1,175 | 1,106 |
| Ames | record | 92,448 | 0 | 0 |
| BBB_Martins | context | 49,006 | 907 | 908 |
| BBB_Martins | record | 284,012 | 13,350 | 2,295 |
| Bioavailability_Ma | context | 15,102 | 936 | 944 |
| Bioavailability_Ma | record | 22,438 | 19,750 | 24,750 |
| Carcinogens | context | 49,138 | 918 | 920 |
| Carcinogens | record | 180,192 | 0 | 0 |
| DILI | context | 24,555 | 1,055 | 1,053 |
| DILI | record | 68,038 | 0 | 0 |
| Skin_Reaction | context | 69,903 | 900 | 900 |
| Skin_Reaction | record | 59,718 | 764 | 592 |

Ten unit checks passed. Verification replayed every emitted row and checked lineage, variance gates, parent restrictions, degree caps, schemas, hashes, and scope. The active implementation matches the tested wheel. All active Python functions are at most 60 lines.

BBB and Oral record and indirect context construction exclude L5. The remaining task-specific caps, gates and holdout policies are declared in `assay_transfer/canonical.json`.

The historical archive contains 7,567 files across 17 roots (50.98 GiB). All approved roots were moved and verified; the previous archive and administrative files were preserved.

Pinned inputs, full coverage and mixture shortfalls, individual verification receipts, archive inventories, and the implementation completion report are recorded under `.agent-tasks/canonical-implementation-20261013/`. No full-release claim is made from these samples.
