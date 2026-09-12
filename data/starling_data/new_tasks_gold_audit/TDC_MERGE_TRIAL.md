> 历史阶段记录（2026-09-12 归档说明）：正文中的计数、状态、路径和命令描述该阶段，不代表当前运行配置。当前两任务均使用 Starling-only gold_v4 和 `retrieval_final`；旧暂存 source/index 及一次性脚本已清理，保留的账本仍用于来源追溯。恢复与维护请见[当前合同](../../../tools/chembl_tool/common/starling/NEW_TASK_SOURCE_DATA.md)。

# TDC null-condition augmentation trial

> Publication update (2026-09-07): the reviewed merged cohort is now active at
> `data/conditioned_benchmark/{DILI,Carcinogens}/{scaffold,random}`. All staged
> input/heldout bytes and existing progressive indices are unchanged. References
> below to the staged trial describe construction provenance; current path mapping
> is in `data/conditioned_benchmark/external_augmentation_publication.json`.


This is a **staged data experiment**, built after the repaired Starling v2.1 release.
The initial merge preserved the source-only release. Subsequent progressive review
repaired source votes and rebuilt both the source-only and merged cohorts; the
current figures below reflect that reviewed release. Raw Stage-03 records remain conserved. No model evaluation has been run or prediction lineage reused.

Trial datasets and their machine-readable manifests are in:

- `data/.build/tdc_augmented_conditioned_benchmark/DILI/{scaffold,random}/`
- `data/.build/tdc_augmented_conditioned_benchmark/Carcinogens/{scaffold,random}/`
- `data/.build/tdc_augmented_conditioned_benchmark/validation.json`

## Meaning and scope

All accepted **TDC positive and negative labels** use
`condition_group=no_reported_external_condition`, `condition_scope=none_reported`,
and empty condition atoms. The shared renderer emits no condition sentence.
This means no per-row external condition is supplied, **not** that the label is
valid under every dose, route, host or species.

The current BBB and Bioavailability scaffold cohorts contain respectively3,675
and2,091 null-condition rows. Their inspected provenance is Starling-derived;
existence of this schema does not establish that they were built by importing TDC.

[TDC's official task descriptions](https://tdcommons.ai/single_pred_tasks/tox/)
define DILI as a drug-level binary liver-injury prediction dataset and reference
Xu2015. Carcinogens_Lagunin references rodent carcinogenicity research. These labels
are imported with their original dataset provenance. They are **not fabricated
individual studies or new Starling assay votes**: `source_record_count=0`,
`original_study_count=null`, no invented PMIDs, and `external_label_count=1`.
The source SMILES, row ordinal, original Drug_ID, download URL and raw-file hash
are retained. Positive TDC rows are imported along with negatives.

This trial adds external null-condition labels to all currently selected
Starling benchmark rows. It does not erase their conditions, re-vote their
labels, or reintroduce sparse conditioned gold excluded by the existing
three-way condition-coverage policy. Complete source gold remains in `gold_v2`.

## Imported rows and identity review

| Task | TDC raw rows | Accepted TDC parent labels | TDC negative / positive | Preserved Starling rows | Combined rows | Combined negative / positive |
|---|---:|---:|---:|---:|---:|---:|
| DILI | 475 | 465 | 231 / 234 | 569 | 1,034 | 274 / 760 |
| Carcinogens | 280 | 276 | 218 / 58 | 1,188 | 1,464 | 278 / 1,186 |

All19 multicomponent raw SMILES were individually read (18 DILI,1 Carcinogens),
with notes in `tdc_structure_reviews.jsonl`. This is Codex structure-text review,
not human-expert adjudication of original biological assays. Ordinary counterion
salts and a hydrate are accepted under the documented parent convention; original
salt forms remain in provenance. Their labels remain source assertions.

Ten DILI rows are pending identity because fragment-parent normalization would
misassign a metal drug label to its free ligand, or collapse a multicomponent
formulation/material or inorganic salt into an unsuitable component. Examples:
platinum compound→diamine; gadolinium compound→free ligand; iron/cyanide/nitrosyl
compound→N=O. The full list and specific reasons are in `tdc_identity_pending.jsonl`.
The earlier apparent diphenhydramine duplicate is one such multicomponent row;
withholding it leaves465 accepted unique parents, not464.

Two Carcinogens parent identities have opposing TDC labels (Drug_2/Drug_98 and
Drug_4/Drug_131). All four original rows remain in `tdc_internal_conflicts.jsonl`;
neither label is arbitrarily selected. The official page's278 count is not the
row count of the frozen downloaded table, which contains280 rows.

There are102 exact parent overlaps with current DILI Starling rows and10 with
Carcinogens.21 and7 overlapping parents, respectively, have at least one opposite
conditioned label. `cross_source_disagreements.jsonl` retains those comparisons.
Distinct condition units remain distinct; TDC never overrides a source condition,
and all conditions of a parent must remain in one split. Opposite directions
across different scopes are not automatically same-unit label conflicts, nor
proof that both source assertions are scientifically correct.

## Rebuilt splits and evaluation scope

Counts below are unique parent molecules, followed by molecule-condition rows. A parent may have several condition rows, but never spans splits. The combined cohorts contain **845 DILI parents / 1,034 rows** and **719 Carcinogens parents / 1,464 rows**.

| Task | Scheme | Train parents (rows) | Valid parents (rows) | Test parents (rows) | TDC-only valid negative / positive | TDC-only test negative / positive |
|---|---|---:|---:|---:|---:|---:|
| DILI | scaffold | 689 (828) | 78 (103) | 78 (103) | 25 / 23 | 25 / 26 |
| DILI | random | 702 (828) | 70 (103) | 73 (103) | 23 / 23 | 23 / 23 |
| Carcinogens | scaffold | 458 (908) | 121 (278) | 140 (278) | 21 / 8 | 50 / 34 |
| Carcinogens | random | 604 (1,164) | 55 (150) | 60 (150) | 21 / 5 | 21 / 5 |

The shared allocators enforce disjoint parents, full condition coverage and both
labels in each split; scaffold additionally enforces disjoint scaffolds and keeps
empty scaffolds in train. Both schemes contain exactly the same rows and labels.
TDC label-specific allocation strata require each held-out partition to contain
at least floor(10% of each TDC class), bounded below by1. These are allocation
constraints only; they never enter query conditions. Nominal held-out sizes grow
only when whole-parent/scaffold feasibility requires it. Solver seed20260807 is
frozen. The first exploratory allocation exposed a degenerate TDC-positive
random subset; the final frozen split enforces these source-specific minima.

The first merged build incorrectly disabled the quality objective globally. The corrected build restores the shared lexicographic policy: **first minimize Starling singleton rows in valid+test, then minimize their valid/test imbalance, then balance labels**. Whole-parent/scaffold separation, all-condition coverage and TDC class minima remain hard constraints. External labels have `record_support_eligible=false`; their zero assay counts are exempt from the quality objective, not counted as high-quality studies. Existing rows default to eligible, preserving other tasks' allocation behavior.

Multi-vote here means at least two accepted source votes for the molecule-condition row; it is a support-count criterion, not a guarantee of biological correctness. Fractions below use only Starling rows as the denominator.

| Task | Scheme | Valid multi-vote / Starling rows | Test multi-vote / Starling rows | Held-out singletons | Solver minimum |
|---|---|---:|---:|---:|---:|
| DILI | scaffold | 43/55 (78.2%) | 40/52 (76.9%) | 24 | 24 |
| DILI | random | 48/57 (84.2%) | 47/57 (82.5%) | 19 | 19 |
| Carcinogens | scaffold | 57/249 (22.9%) | 65/194 (33.5%) | 321 | 321 |
| Carcinogens | random | 43/124 (34.7%) | 48/124 (38.7%) | 157 | 157 |

The selected singleton totals and imbalance match the current solver optima. Carcinogens still has weak Starling support:322/1,188 source rows have multiple votes. Full condition coverage and group integrity prevent an all-multi-vote evaluation set. TDC improves class coverage without supplying additional assay-study votes.

Report the TDC-null and Starling-conditioned subsets **separately**, alongside any
explicitly named pooled exploratory score. Label prevalence is correlated with
source/condition presence; pooled gains alone do not establish improved reasoning
or label validity. Carcinogens' TDC-positive held-out support is still small.
`evaluation_groups.json` gives the exact source-specific row IDs for each split.
These are newly constructed joint splits, not the original TDC official test set.

Progressive overlays and split-specific indices are tracked separately in
`data/starling_data/<task>/progressive_v1/`; see `PROGRESSIVE_LEVELS.md` and each
`validation_all.json` release gate. They remove all valid+test parents' direct
and near-direct evidence across sources/conditions. TDC labels never enter
cards. Source-only indices and predictions cannot substitute for these inputs.

## Validation and reproduction

107 focused tests pass after progressive integration, including external source hash checks, duplicate/conflict behavior,
metal-ligand rejection, absence of invented assay votes, null-condition rendering,
source-specific minima in both shared group solvers, and mixed-source quality optimization without fabricated study counts. Independent full-row
validation checks imported labels against frozen TDC, unchanged Starling rows,
source gold hashes, exact split unions, condition coverage, parent/scaffold
disjointness, held-out inventories, exact agreement with the minimum singleton and imbalance objectives, and150 unchanged active benchmark files.

Use the existing vllm environment; this build reads the cached755 TDC identities
and current small benchmark tables, rechecks the identities, and reuses the shared
solvers. It does not rebuild the4.9M source records or call an LLM/PubChem endpoint.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/data1/tianang/anaconda3/envs/vllm/bin/python \
  -m tools.chembl_tool.common.starling.build_tdc_augmented_benchmark
```

The builder refuses to write into the active benchmark root. Per-task manifests
pin source/code/output hashes, counts, constraints and optimizer receipts.
