# Conditioned Benchmark

This is the only active conditioned evaluation dataset. It provides two frozen
split schemes over the same molecule-condition rows and labels:

```text
data/conditioned_benchmark/<Task>/{scaffold,random}/
```

The public name is **Conditioned Benchmark**. The machine-readable contract is
`conditioned_benchmark.v1`; task-specific historical version strings are not
part of active paths, runner flags, or result labels.

## Tasks and sizes

| Task | Train | Valid | Test | Target |
|---|---:|---:|---:|---|
| BBB_Martins | 3,053 | 397 | 393 | experimentally meaningful systemic CNS access |
| Bioavailability_Ma | 1,956 | 262 | 269 | oral bioavailability under the reported condition |
| ClinTox | 1,144 | 142 | 142 | clinical-trial toxicity failure versus approved comparator |
| Skin_Reaction | 1,941 | 239 | 241 | skin sensitization/contact allergy |
| Ames | 1,926 | 274 | 274 | bacterial reverse mutation under the reported strain panel and metabolic activation |

The table shows scaffold counts. Ames random counts are 1,980 / 247 / 247 over
the same 2,474 rows; random model evaluation has not been performed.
Ames scaffold valid/test names were exchanged on explicit user request on
2026-09-08. Current test is the former validation cohort, which has already
been inspected; it must not be described as an untouched test set. Train and
random remain fixed. Exact row/hash mappings and heldout-set equivalence are
recorded in `data/conditioned_benchmark/Ames/provenance/scaffold_valid_test_swap_20260908.json`.

Every split row has the same condition-aware schema:

```text
drug, Y, condition_group, condition_scope, molecule_identity_key,
bemis_murcko_scaffold, benchmark_row_id
```

ClinTox has no accepted external condition groups. Its rows use
`no_reported_external_condition`; the prompt renderer omits the condition
sentence for this value. The other tasks use the same null-group behavior.

## Construction contract

The benchmark is built in four separate stages.  They must not be collapsed
into a script that reads the final JSONL files and guesses how they were made.

1. **Task-specific source review and voting.** Each task decides which physical
   source records may vote, how a record maps to `Y`, the unit at which duplicate
   claims are collapsed, and how conflicts are resolved. Rejected records and
   reasons remain in task-specific audit artifacts.
2. **Parent and condition construction.** Valid structures are normalized to a
   molecular-parent identity. Accepted external experimental conditions follow
   the task's recorded review method and become separate molecule-condition rows. Missing
   accepted external context uses `no_reported_external_condition`; it is not
   silently merged with a real condition.
3. **Scaffold allocation.** Complete Bemis-Murcko scaffold groups are assigned
   to train, valid, or test. This makes both molecular parents and non-empty
   scaffolds disjoint across the three partitions.
4. **Random allocation.** The exact union of the scaffold cohort is reassigned
   by complete molecular-parent groups. Labels and molecule-condition rows do
   not change; only split membership changes.

The active scientific voting contracts are:

| Task | Voting unit and acceptance rule | Conflict rule |
|---|---|---|
| BBB_Martins | Experimental, systemically administered, meaningful CNS-access outcomes; explicit source-native direction or a frozen manually reviewed qualitative direction | Aggregate accepted records at molecular-parent grain; preserve record-level reasons and rejected/conflicting-parent audits |
| Bioavailability_Ma | Canonical direct absolute oral bioavailability claim for the reported condition; human conditioned rows additionally require reviewed external context and exclude relative effects, non-IV comparisons, indirect analytes, predictions/simulations, and unresolved populations | `canonical_claim_id` is the claim unit; aggregate accepted claims within parent-condition only |
| Skin_Reaction | Measured final skin-sensitization/contact-allergy outcome under the reviewed condition contract | Aggregate accepted final-outcome votes within parent-condition; model predictions, photo/irritation endpoints, integrated approaches, and mechanistic AOP rows do not vote |
| ClinTox | Frozen source roles, not assay voting: AACT toxicity-failure evidence gives `Y=1`; an FDA-approved comparator with no positive source gives `Y=0` | A parent present in both roles is positive; there is no record-majority threshold |
| Ames | Experimental bacterial reverse mutation under the exact reported strain panel and activation regime; deterministic gates plus payload-pinned semantic reviews and exact PubChem name-parent match | One unambiguous PMID-parent-condition vote; within-study conflicts do not vote; across studies require at least 60% agreement for external conditions (70% without a reported condition), excluding ties |

Task-specific code remains authoritative for the scientific criteria. The
shared publisher only normalizes the interface and is forbidden from revoting:

```text
BBB_Martins:
  tools/chembl_tool/tasks/bbb_martins/experimental_meaningful_cns_access_benchmark_v4.py
  tools/chembl_tool/tasks/bbb_martins/build_conditioned_source.py
Bioavailability_Ma:
  tools/chembl_tool/tasks/bioavailability_ma/build_canonical_starling_source.py
  tools/chembl_tool/tasks/bioavailability_ma/reviewed_context_conditioned_benchmark.py
Skin_Reaction:
  tools/chembl_tool/tasks/skin_reaction/build_canonical_starling_source.py
  tools/chembl_tool/tasks/skin_reaction/context_conditioned_benchmark.py
ClinTox:
  tools/chembl_tool/tasks/clintox/clinical_trial_failure_benchmark.py
  tools/chembl_tool/tasks/clintox/build_clinical_trial_failure_benchmark.py
Ames:
  tools/chembl_tool/tasks/ames/source_contract.py
  tools/chembl_tool/tasks/ames/build_dataset.py
```

Ames is a name-parent-verified source subset, with unresolved identity requests
explicitly excluded and retained in a pending ledger. Source review v4 adds
individual agent review of 96 L1 and 48 L2 records and a hash-pinned decision
ledger for claims from any source. It adds 12 study votes and withdraws 25; the
current source has 3,333 actual voters. This is not exhaustive human/full-paper
review, and a PubChem name match does not prove correct original-paper subject
attribution. The task README and source_review_v4 retain the per-record evidence.
See `tools/chembl_tool/tasks/ames/README.md` for its panel-any-positive semantics,
source exclusions, audit artifacts and data-only integration status.

## Split and leakage contract

### Hard constraints shared by both schemes

- The evaluation unit is one molecule-condition row; all rows belonging to the
  same normalized molecular parent stay in one split.
- Train, valid, and test therefore have zero parent-identity overlap.
- Labels, conditions, and row identities are immutable between scaffold and
  random schemes. Their union hashes are checked in `random_split_receipt.json`.
- A null condition is a real benchmark unit, not missing schema.
- No row is duplicated or fabricated to improve balance.

### Scaffold split

- The default split assigns complete Bemis-Murcko scaffold groups, so non-empty
  scaffold overlap and parent overlap are both zero.
- The historical record-supported allocator uses exact held-out sizes. For BBB,
  each evaluation split is `min(500, floor(0.1 * n))`; Bioavailability and Skin
  use `floor(0.1 * n)`. Current conditioned cohorts may preserve a frozen,
  already-reviewed assignment rather than reoptimize after a source repair.
- A fresh Ames cohort starts at `max(number of eligible conditions, floor(0.1 * n))`
  rows per held-out split and finds the smallest feasible equal valid/test size.
  Every retained condition must have three parents and three nonempty scaffolds
  and occur in all three partitions. The resolved size is frozen before quality
  optimization; no condition is silently dropped to meet the nominal 10% target.
  Explicitly requested sizes remain strict.
- When a fresh scaffold allocation is required for a voter-based task, its
  lexicographic objectives are: minimize singleton-vote rows in valid+test;
  minimize their valid/test imbalance; minimize positive-label deviation;
  maximize reuse of the frozen prior-valid parents; then apply a seeded stable
  tie-break. A multi-vote row means `source_record_count >= 2`.
- A deletion-only gold/source repair preserves every surviving parent's split.
  A new parent is rejected by default; if explicitly allowed, it inherits an
  existing scaffold assignment, and a genuinely new scaffold defaults to train.
  Such a repair must retain scaffold disjointness and writes its method to the
  split summary.

### Random split

`build_conditioned_random_split.py` solves a deterministic parent-grouped MILP
with seed `20260830` and exact rounded 80/10/10 row counts. Before any objective
is optimized, it enforces:

- one parent group can enter at most one held-out split;
- every condition has at least one parent in train, valid, and test;
- valid and test each have their exact target row count.

For BBB, Bioavailability, Skin, and Ames, the following objectives are strictly
lexicographic: each optimum is frozen before the next stage is considered.

1. Minimize the total number of `source_record_count == 1` rows in valid+test.
   Equivalently, this prioritizes multi-vote rows for held-out evaluation.
2. Minimize the absolute difference in unavoidable singleton rows between valid
   and test.
3. Minimize total absolute deviation of positive-label counts from their 10%
   targets in valid and test.
4. Minimize total absolute deviation of every condition's row count from its
   10% target in valid and test.
5. Resolve remaining ties with a stable seed-and-parent hash order.

ClinTox skips objectives 1 and 2 because `source_record_count` records source
role provenance rather than independent assay votes. It still enforces the hard
constraints and optimizes label balance, condition balance, and the stable
tie-break in that order.

A deletion-only repair may use `--preserve-existing-split`. It fails if a new
parent appears, a row changes split, exact row counts no longer hold, or any
condition loses three-way coverage. This prevents a small source correction
from silently selecting a more favorable validation/test cohort.

### Conditioned family/progressive retrieval leakage policy

- Before building an evaluation index, valid+test direct-outcome rows are
  removed from the source library.
- Query-time neighbor exclusion in the current cumulative-family/progressive
  experiment is split-specific: `scaffold_disjoint` for scaffold evaluation and
  `parent_disjoint` for random evaluation. Other retained runners must record an
  explicit identity policy and may not infer or change it from a directory name.
- Random split intentionally permits the same scaffold across partitions; using
  `scaffold_disjoint` there would evaluate a different, stricter experiment.

## Provenance and result reuse

`data/conditioned_benchmark/migration_receipt.json` records the former source
artifact, hashes, row counts, label counts, and ordered `(drug, Y)` checks.
BBB split files remain byte-identical to their previously evaluated
selected conditioned cohorts. Bioavailability excludes six frozen HF records
whose nitrendipine text was bound to the non-nitrendipine structure identity
`CNYREWGHOWSYCJ`; the two resulting benchmark rows and all six retrievable
records were removed in place. ClinTox adds only null-condition metadata.
Skin excludes prediction-only, photo/light-dependent, irritation-only,
non-contact severe cutaneous reactions, integrated/defined-approach, and mechanistic AOP rows from gold voting. Source-purity v5 applies the
same strict target-scope gate to every retrieval level while retaining target-aligned nonvoter outcomes/classifications in L2 and mechanisms in L3.
This changes the cohort, both split lineages, and both retrieval indices, so all pre-v5 Skin results are stale. Exact migrations are frozen in
`data/conditioned_benchmark/Skin_Reaction/provenance/semantic_gold_v2_migration.json` and
`data/conditioned_benchmark/Skin_Reaction/provenance/semantic_gold_v3_migration.json`.
Existing predictions are reusable only when their manifest input hash matches
the current receipt. Retrieval-dependent predictions additionally require the
same held-out index hash, source-family catalog hash, visibility contract,
prompt profile, model contract, and split policy; no score is copied by name
alone.

The old molecule-only and selected-vN names are source-lineage provenance, not
alternative active gold datasets. New code must import paths from
`tools.chembl_tool.common.starling.conditioned_benchmark` and must not hard-code
those historical paths.

## Entrypoints

```bash
python -m tools.chembl_tool.tasks.skin_reaction.build_canonical_starling_source
python -m tools.chembl_tool.tasks.skin_reaction.context_conditioned_benchmark build-selected
python -m tools.chembl_tool.tasks.ames.build_dataset --phase all --workers 16
python -m tools.chembl_tool.common.starling.publish_conditioned_benchmark
python -m tools.chembl_tool.common.starling.build_conditioned_random_split
python -m tools.chembl_tool.paper_experiments.run_conditioned_assay_progressive_curve
```

The first command is the Skin-specific canonical direct/AOP source build; the
second regenerates the reviewed Skin scaffold publication input. Task builders
stage reproducible inputs under the Git-ignored `data/.build/` tree. The
publisher copies the active, reviewable dataset and required audit ledgers into
`data/conditioned_benchmark/`; runners never read the staging tree.

The random builder is deterministic (`seed=20260830`), uses contract
`conditioned_random_parent_grouped_quality_stratified.v2`, and writes the full
lexicographic optimization audit to `random_split_receipt.json`. Run it after
republishing the scaffold cohort.
Agent evaluation also requires random-heldout-filtered retrieval indices; a
scaffold-heldout index must not be reused merely because the source rows match.

Task-specific source review and voting code remains responsible for scientific
label construction. The publisher only standardizes the final interface and
does not revote or change task labels.

## Required audit artifacts

The following are part of the benchmark, not disposable build intermediates:

- `manifest.json`: active roots, target definitions, split contracts and counts;
- `migration_receipt.json`: source-to-canonical hashes and ordered label checks;
- `random_split_receipt.json`: complete MILP/preservation audit, objective
  values, union hashes, overlap checks and condition coverage;
- each split's `summary.json` and `group_distribution.csv`;
- detailed `*_molecule_condition_labels.jsonl` files, including parent,
  scaffold, condition, `source_record_count`, and provenance fields;
- task-specific accepted/rejected/conflict/review ledgers copied under
  `provenance/` or referenced by the task summary.

A benchmark release is incomplete if it contains only `train.jsonl`,
`valid.jsonl`, and `test.jsonl`. Those minimal files are runner inputs; they do
not preserve enough information to reproduce or audit the cohort.


## DILI / Carcinogens publication

Earlier external-label and tautomer-repair receipts are retained as historical
provenance. Their TDC augmentation builder/publisher has been retired; neither
current gold includes TDC labels. Source repair does not modify frozen votes.

These tasks use `new_task_tautomer_identity.v2`: stable-stereo gold identity,
connectivity/formula/charge leakage groups and terminal-pruned Murcko topology
scaffold groups. Other tasks retain their own identity contracts. The DILI gold_v4
builder explicitly swaps allocated valid/test names, preserving the earlier naming
policy on its rebuilt cohort. Both subsets were inspected during diagnostics;
source cleanup does not make test unseen.

### Current reviewed Starling-only releases (2026-09-09)

Both tasks now publish from `data/starling_data/<task>/gold_v4/`, reusing the
completed full-base direction ledgers without additional model review or TDC labels.
DILI has 4,024 benchmark rows; Carcinogens has 4,692, using exactly five organism
groups (rodent/human/dog/monkey/rabbit). The shared `--reviewed-release` publisher
validates the staged split/index hashes and preserves previous active inputs.
Each task's `gold_v4/retrieval/canonical_publication.json` records the initial
gold_v4 publication. The retained final retrieval source and its two indices are
identified by `retrieval_final/publication.json` and `current_starling_retrieval.json`;
the initial temporary build/publish scripts have been removed.
L1 contains actual publication-vote representatives; only heldout L1 is prefiltered,
and L2 remains subject to query-time disjoint retrieval. Fresh cohort scores must
not be inferred from historical gold_v3 runs. Full source policies and build
entrypoints are in [NEW_TASK_SOURCE_DATA.md](NEW_TASK_SOURCE_DATA.md).
