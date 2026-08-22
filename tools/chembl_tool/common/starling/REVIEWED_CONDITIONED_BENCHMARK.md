# Reviewed context-conditioned benchmark protocol

This is the canonical contract for the selected context-conditioned Starling
benchmarks. It extends a frozen molecule-only benchmark without changing any
existing label or split assignment.

The benchmark grain is:

```text
(normalized parent molecule, exact external-condition signature)
```

Consequently, one parent molecule appears at most once in one group, but may
appear in multiple condition groups. Every row in `train.jsonl`, `valid.jsonl`,
and `test.jsonl` explicitly records `condition_group`; the detailed
`*_molecule_condition_labels.jsonl` files retain its vote and source lineage.

## Current selected lineages

Published data use one root and one version:

```text
data/processed_starling_context_conditioned_selected_v1/<Task>/scaffold/
```

| Task | Frozen null-group source | Selected external groups |
|---|---|---|
| Bioavailability_Ma | `record_supported_v2` over `bioavailability_canonical_direct.v2` | `prandial_state=fasted`; `prandial_state=fed_unspecified`; `prandial_state=fed_high_fat`; `co_treatment=rifampin`; `disease=cirrhosis`; `disease=cystic_fibrosis`; `release_profile=modified_release` |
| BBB_Martins | `experimental_meaningful_cns_access_v2` | `barrier_state=disrupted`; `co_treatment=cyclosporine`; `disease=meningitis_unspecified`; `disease=bacterial_meningitis`; `disease=pneumococcal_meningitis`; `disease=tuberculous_meningitis`; `disease=brain_tumor_or_glioma`; `disease=cerebral_ischemia` |
| Skin_Reaction | `record_supported_v2` | `disease=atopic_dermatitis` |

The null group is always `no_reported_external_condition`. It is a statement
about reported external context, not proof that no unreported context existed.
The older `processed_starling_context_conditioned_reviewed_v1` root is the
broader exploratory group inventory and is not the selected benchmark.

ClinTox remains on the frozen unconditioned `clinical_trial_failure_v1`
benchmark. `tasks/clintox/context_conditioned_benchmark.py` preserves the
source-specific AACT proposal/review audit that established this boundary, but
no candidate taxonomy passed the publication gate and no conditioned ClinTox
dataset is selected or consumed by the current agent or baseline runners.

SLS was exhaustively cleaned but is not a formal Skin group: only four strict
eligible parents remained, all in train and without three distinct usable
scaffolds/split coverage. Occlusion, vehicle, route, age, formulation, intact
barrier, and other common keyword groups were not selected merely because they
were frequent.

## Record eligibility

Pattern rules only propose candidates for semantic review; keyword matches do
not create gold. A source record may vote only when all of the following hold:

1. It is a direct empirical record for the task endpoint and query parent.
2. Its qualifying condition describes the tested arm's actual external
   exposure, host state, or disease state—not a mechanism, intrinsic molecular
   property, literature aside, comparator-only context, or query material.
3. It supports an absolute endpoint label. Relative-only effects are always
   discarded and never vote, even when the relative direction is clear.
4. Outcome attribution, molecule identity, and condition attribution are
   unambiguous and refer to the same experimental arm.
5. Predicted, simulated, PBPK/population-model, and duplicate study-arm records
   are excluded where applicable.
6. The exact condition signature is preserved. Composite conditions are not
   collapsed into a simpler group; a sparse composite is excluded from the
   selected benchmark rather than treated as its component group.
7. Every candidate record has a payload hash and a unique terminal `accepted`
   or `rejected` verdict. Pending, missing, duplicated, or stale verdicts make
   the build fail closed.

Task-specific constraints remain in the adapters. Skin selected records require
explicit human context and exhaustive manual adjudication of every pure
atopic-dermatitis candidate. BBB records use the meaningful/adequate CNS-access
endpoint contract; a merely relative exposure change cannot substitute for an
absolute access outcome.

Dose and species remain source provenance, not group atoms. Mechanisms and
intrinsic attributes such as poor solubility or first-pass metabolism do not
become external-condition groups.

## Voting and group promotion

- One terminally accepted direct source claim is one vote inside one exact
  parent-condition unit.
- A parent-condition label requires at least 60% agreement. Exact ties are
  rejected.
- A formal external group requires at least three accepted parents, three
  distinct Bemis-Murcko scaffolds, and at least one parent in each of train,
  valid, and test.
- The explicit task allowlist is applied before split allocation. A group can
  pass the numeric gate and still remain exploratory if its causal meaning is
  unsuitable for the endpoint.

Frozen parents and scaffolds keep their original split. Novel scaffolds are
assigned together by the shared MILP allocator. Parent identity and scaffold
overlap across train, valid, and test must both remain zero.

## Audit and row contract

Each task has one canonical full review root (Bioavailability currently uses
`context_conditioned_review_v2`; the other selected tasks use v1):

```text
data/starling_data/<task>/context_conditioned_review_v<version>/
```

It contains the complete queue, proposal audit, prereviews, terminal verdicts,
and any manual record decisions. These files preserve decisions for selected
and non-selected groups and are lineage-pinned by SHA-256 in `summary.json`.

The selected benchmark does not duplicate that entire ledger. It contains only
the review and parent-condition audit rows relevant to its exact allowlist:

```text
train.jsonl, valid.jsonl, test.jsonl
*_molecule_condition_labels.jsonl
heldout_molecule_condition_labels.jsonl
source_condition_review.jsonl
accepted_parent_conditions_before_group_gate.jsonl
rejected_parent_conditions.jsonl
group_gate_rejected_parent_conditions.jsonl
group_distribution.csv
summary.json
```

Minimal split rows contain `drug`, `Y`, `condition_group`, `condition_scope`,
`molecule_identity_key`, `bemis_murcko_scaffold`, and `benchmark_row_id`.
`benchmark_row_id` identifies the parent-condition unit. The detailed label
files additionally retain vote counts, agreement, source record IDs, PMIDs,
label methods, reviewers, condition examples, and split provenance.

### Review-model provenance

Bioavailability's 2,452 queued candidates were each evaluated in two separate
passes by the local OpenAI-compatible `gpt-oss-120b` endpoint using the hashed
`condition_semantic_prereview.v2` contract. Both passes used temperature 0,
reasoning effort `medium`, thinking/tool calls disabled, and a 1,024-token
completion budget. The terminal policy accepts a record only when both passes
accept it; disagreements and unanimous rejections are rejected. The per-pass
summary files record endpoint and execution settings, while every record stores
payload hash, contract hash, returned model ID, token usage, reasoning, and raw
JSON. The exact served checkpoint revision/quantization was not captured by the
historical endpoint and therefore is not claimed.

This is exhaustive model-assisted semantic review, not human annotation.
`review_verdicts.jsonl` may be terminal for the automated benchmark build, but
its reviewer provenance must remain `dual_semantic_prereview_unanimity_v2`.

`summary.json` distinguishes the size of the complete canonical review ledger
(`n_full_review_ledger_records`) from the selected audit subset
(`n_published_group_review_records`). It also records the exact allowlist,
source hashes, group/split counts, base preservation policy, and identity and
scaffold overlap checks.

## Paper-facing MiniMol baseline

For tasks with a selected condition taxonomy, the default trained MiniMol
baseline represents each benchmark row as the frozen 512-dimensional molecule
embedding concatenated with a one-hot `condition_group` feature. The categorical
vocabulary comes only from outer train; an unseen evaluation category fails
closed. ClinTox has no accepted condition taxonomy and therefore retains the
same molecule-only head.

Model selection is entirely train-only. Scaffold CV selects the epoch by mean
inner-valid AUROC, then the selected epoch's pooled OOF scores select one
macro-F1 decision threshold. A five-member head ensemble is fit on all train
rows and evaluated using that frozen threshold. Outer valid/test labels do not
select epochs, hyperparameters, categories, or thresholds. The shared entries
are:

```text
baselines/minimol/condition_features.py
baselines/minimol/run_train_cv.py
baselines/minimol/run_bioavailability_ma.py
```

Current scaffold-valid results and complete OOF/feature contracts are recorded
under `outputs/baselines/starling_conditioned_valid_v1/`; valid macro-F1 is
0.6912 BBB, 0.5872 Bioavailability, 0.6030 Skin, and 0.6520 unconditioned
ClinTox.

## Implementation and rebuild

Shared validation, voting, allocation, and publication live in:

```text
tools/chembl_tool/common/starling/reviewed_conditioned_benchmark.py
```

Task adapters contain only source-specific condition and endpoint semantics:

```text
tools/chembl_tool/tasks/bbb_martins/context_conditioned_benchmark.py
tools/chembl_tool/tasks/bioavailability_ma/reviewed_context_conditioned_benchmark.py
tools/chembl_tool/tasks/skin_reaction/context_conditioned_benchmark.py
```

Rebuild the current selected data in place—do not create a new version for a
non-semantic cleanup:

```bash
python -m tools.chembl_tool.tasks.bbb_martins.context_conditioned_benchmark build-selected
python -m tools.chembl_tool.tasks.bioavailability_ma.reviewed_context_conditioned_benchmark build-selected
python -m tools.chembl_tool.tasks.skin_reaction.context_conditioned_benchmark build-selected
```

A new lineage/version is warranted only if the source snapshot, endpoint/gold
semantics, selected group set, vote threshold, or split contract changes. Code
cleanup, deterministic regeneration, or removal of duplicated audit copies
must overwrite the same selected v1 artifacts.
