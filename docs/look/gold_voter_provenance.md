# Gold voters: source truth, canonical UIDs, and publication checks

Audited on 2026-10-15. The machine-readable result is
[gold_voter_audit.json](gold_voter_audit.json). All six Gold membership contracts pass.
Skin Gold v1 now follows the pinned source replay and original molecule table;
the pre-correction files are archived with its correction receipt. Query labels,
splits, historical retrieval releases, predictions and FINAL results are preserved.

## What the identifiers mean

A canonical evidence-library `source_row_uid` (`sr_` followed by 32 hexadecimal
characters) names one physical retained row in that evidence release. An external
`record_id`, such as Skin's `ext_7`, is an extraction identifier supplied by the
source. It can recur in different papers and rows. It cannot replace the UID.

A Gold card aggregates votes for a molecule and condition. Its membership is a
relation between a card, a vote unit, and physical row UIDs. In Skin's frozen
construction, a physical row may support both a condition-free card and a
condition-specific card. These are two membership edges referencing one physical
row, not two physical records. Within a card, a member must not be repeated.
Manual selection therefore needs a context ID when a UID belongs to multiple
candidate cards.

Oral also distinguishes votes from physical members: one canonical claim can
have multiple physical source rows. Each claim contributes one vote; its members
carry weights `1 / physical_member_count`. Counting each member as a full vote
would change Gold. The membership's `vote_label`, rather than a current library
display or calibration label, reconstructs the original Gold vote.

TDC L1 cards are a separate contract: their members are benchmark training-label
records. They do not need fabricated V10 UIDs, and this audit does not convert them
into experimental evidence-library rows.

## Where the frozen Skin cards came from

The import code is
[`import_upstream_conditioned_benchmark.py`](../../data/processing/gold_labels/import_upstream_conditioned_benchmark.py).
It pins commit `26d44f121141f3738764c0b00dc09e185445341b` and imports the files
under `data/conditioned_benchmark/Skin_Reaction/scaffold/` into the current
[`Skin_Reaction/v1/scaffold`](../../data/gold_labels/Skin_Reaction/v1/scaffold/).

At that original commit, the construction chain was:

1. `tools/chembl_tool/tasks/skin_reaction/starling_benchmark.py` labeled rows
   from `data/starling_data/skin_reaction/direct_skin_reaction/extractions.parquet`.
   `canonical_starling_source.py` supplied the measured sensitization eligibility
   rules. These frozen rules differ from the currently migrated task code.
2. `tools/chembl_tool/common/starling/benchmark_dataset.py` grouped accepted
   records by normalized molecule. Every accepted source row contributed one
   binary vote. Exact ties were rejected; the condition-free agreement floor was
   70%. **The `grouped` lists before summary construction were the full voters.**
3. `build_record_supported_benchmark.py` allocated scaffold splits. The frozen
   molecule-only cards lived under
   `data/processed_starling_record_supported_v2/Skin_Reaction/scaffold/`.
4. `tools/chembl_tool/tasks/skin_reaction/context_conditioned_benchmark.py`, using
   the shared `reviewed_conditioned_benchmark.py`, wrapped those 2,386 cards as
   condition-free and added 35 externally conditioned cards from 57 reviewed
   source votes. The external-condition agreement floor was 60%.

The combined release contains 2,421 cards: 1,941 train, 239 validation, and 241
test. Its current
[`source_condition_review.jsonl`](../../data/gold_labels/Skin_Reaction/v1/scaffold/source_condition_review.jsonl)
is the same Git blob as the original review file
(`5455af3d54fc9c01697e96b98c63fb1e22e2ced6`). Its accepted entries name raw
acquisition coordinates: 56 `direct_skin_reaction:<index>` rows and one
`sensitization_aop:22321` row. `source_index` in a review is not necessarily that
raw acquisition coordinate; use the coordinate in `source_record_id`.

## Why the old Skin adapter was insufficient

The original aggregate writer capped independently deduplicated summaries:
`source_record_ids` at 50, PMIDs at 50, raw-value examples at 20, and context
examples at 10. They do not form parallel arrays, and the record IDs are not
unique physical identifiers. They cannot recover all voters or attach an example
to a particular source row.

The historical adapter used these summaries to synthesize
`skin_gold_v1_member_...` IDs and distribute examples across them. Those IDs were
never canonical evidence-library UIDs. The maintained builder now requires a
verified uncapped membership artifact, reads the actual source rows, and keeps
Gold labels/conditions on contexts. Both the Gold and mixed Gold/TDC Skin builders
use those canonical members. Missing, ambiguous, or structurally incompatible
source rows block publication; shortened examples are never a fallback.

New hydration of the known synthetic Skin member IDs raises an error before
inference. Historical cache files and completed retrieval snapshots remain
unchanged. The active successor uses canonical voter UIDs from the corrected
Gold membership and `v10_main_universe_v7`.

## The historical Skin conflict and approved correction

The pinned raw direct table has SHA-256
`e7c4819c98af47eae4dc0563f40dc39f8b09eb8e03c9626daee603147125f9c1`.
The present raw table contains exactly the same original column values in the
same row order, plus an added UID column, so the UID coordinates can be bound
exactly.

Before correction, the imported condition-free card files contained **34,963 votes**, while
replaying the pinned classifier and normalizer gives **34,947**. There are 12
card-count differences, totaling 16 votes. The original committed
`data/processed_starling/Skin_Reaction/semantic_gold_v3_migration.json` explicitly
reports 34,947 accepted votes, matching the replay. The scaffold summary reports
34,963 and different exclusion counts. Both files exist in the same pinned Git
snapshot; Git provenance alone does not reconcile them.

A further direct comparison found that the original committed
`data/processed_starling/Skin_Reaction/molecule_labels.jsonl` contains 2,386
cards and 34,947 votes. Its vote counts agree with replay on all 12 disputed
cards. Its parent set and all final binary labels exactly match the frozen
condition-free cards. Thus the replay is supported by both an original Gold
table and the migration receipt. The evidence points to stale downstream
record-supported/conditioned voter summaries; the exact build invocation that
failed to refresh them has not been established. For reconstructing voters,
the approved correction uses the replay corroborated by that original table.
It corrects counts in 12 cards while preserving every frozen query label and split.

The 22 replayed voter UIDs missing from `v10_main_universe_v6` were also
absent from its authoritative main source input and Stage 1 protection set.
Restoration used each UID's own acquisition row and permanent ledger coordinate.
The new source release is `data/raw/starling/main_universe_skin_gold_voters_v1/`;
its manifest records all 22 restored rows and source hashes.

The broader reviewed Skin L1 set is intentionally preserved: 42,435 prior
protected UIDs plus the missing 22 gives **42,457 protected UIDs**. All survive
cleaning, canonicalization and final Stage 3 in `v10_main_universe_v7`, which
contains **548,453 physical rows**. All 548,431 prior canonical row IDs and every
previously mapped level assignment remain unchanged. The complete release level
map explicitly uses level zero for rows outside the reviewed mapping.

The corrected membership contains **35,004 card-to-UID edges**: 34,947
condition-free votes plus 57 reviewed conditioned votes. There are **34,950
distinct physical UIDs**; 54 rows belong to both a condition-free and a reviewed
conditioned card. It is not appropriate to deduplicate those membership edges
across cards, or to use the 42,457-row protection set as the voter membership.

The correction receipt and original files live under
`data/gold_labels/Skin_Reaction/v1/reviews/voter_membership_replay_v1/`.
The older 36,532-entry legacy voter file and the original 42,435-row protection
artifact do not reconstruct the corrected card membership.

Seven training cards render different rounded vote percentages. The original
pinned V9 Skin scorer recomputed exactly **197** affected query/card pairs
(104 validation, 93 test); **48,443** exact compatible scores were reused.
Both resulting rankings passed deterministic prompt/candidate validation.
The canonical L1 adapter preserves the original scaffold-disjoint Morgan pools.
The active cache set is `scaffold_disjoint_parent100_v2`; its other eleven
benchmark/task bindings and all indirect ranking inputs remain unchanged.
Native TDC Skin direct cards still use the 282 training-label records.

The matching semantic release preserves all 24,494 reviewed L2/L3 physical rows
and all reviewed bucket weights. Its rebuilt ledger includes all 548,453 UIDs;
the 22 additions are L1, while level-zero rows remain unleveled. The optimizer
loads exactly the same 22,537 indirect record weights. The equivalence receipt is
`outputs/analysis/evidence_library/skin_gold_voters_replay_v1/semantic_equivalence.json`.

## Verification across the six Gold datasets

| Gold task | Cards | Vote units | Physical membership edges | Result |
| --- | ---: | ---: | ---: | --- |
| BBB | 3,843 | 7,634 | 7,634 | Verified |
| Oral | 2,489 | 18,853 | 19,479 | Verified, claim weights preserved |
| Skin | 2,421 | 35,004 | 35,004 (34,950 distinct UIDs) | Verified, replay correction |
| Ames | 2,474 | 3,131 | 3,131 | Verified |
| DILI | 4,024 | 100,522 | 100,522 | Verified, reviewed exclusion preserved |
| Carcinogens | 4,692 | 87,563 | 87,563 | Verified |

For all six tasks, the checks cover unique membership edges, exact
vote-unit membership, per-member vote labels and weights, indexed vote membership,
parent/condition/split identity, card means/labels/counts, and unique resolution of
every member in current canonical Stage 3. For the other five tasks, member UID sets also matched
their current L1 maps in the source investigation. Skin retains the explicitly
approved broader L1 protection set described above.

The independent source investigation established:

- **BBB and Oral:** membership edges matched the frozen
  `data/artifacts/gold_labels/conditioned_benchmark/v2_voter_lineage/` JSONLs.
  The lineage builder is
  [`build_v1_voter_lineage.py`](../../data/processing/gold_labels/build_v1_voter_lineage.py).
  Oral's canonical-claim expansion is part of that builder.
- **Ames:** every vote matched the detailed card `source_votes` and the pinned
  original L1 mapping through canonical record identity to permanent row UID.
- **DILI and Carcinogens:** independently reconstructing the committed original
  reviewed inputs at `8972834103833c18a8d269a92f898bad07f7e5ab` gave exactly
  100,522 and 87,563 `(card, UID, label)` edges, respectively, with no extras or
  omissions. For each task, source truth is under
  `data/starling_data/<task>/gold_v4/`: `record_dispositions.parquet` selects
  `study_vote_representative` rows; `identity_ledger.parquet` supplies identities;
  `tautomer_identities.jsonl` maps parents to final Gold molecule identities;
  `gold_labels.jsonl` supplies aggregate checks. Original construction code is
  `tools/chembl_tool/tasks/<task>/build_reviewed_starling.py` at that commit.
  Carcinogens conditions come from dispositions. DILI additionally needs the
  original `new_tasks_gold_audit/targeted_review_v2/dili_condition_review/` source
  votes, formulation review, condition changes, and the builder's
  `reused_metadata` logic. Preserve the reviewed exclusion
  `sr_3e62e0aa73804ed7a2f5c0f9acf399c3` recorded in the current DILI review.

DILI/Carcinogens training-detail JSONLs were generated replay outputs, **not
committed Git files**. Historical contract manifests incorrectly described those
paths as Git blobs. They remain immutable audit records; future publisher
metadata now identifies them as generated outputs and names the actual committed
inputs and builder. This corrects provenance without changing voter membership.

Native TDC direct-card checks also matched the canonical train labels and
conditions: BBB 1,421; Oral 448; Skin 282; Ames 5,094; DILI 332; Carcinogens 196.
These counts describe TDC label records, not Gold evidence voters.

## Repeat the checks

Run the shared audit without model calls:

```bash
bash experiments/evidence_library/audit_gold_voters.sh
```

It returns exit code 1 if any task is blocked. A report is written under
`outputs/analysis/evidence_library/`; set `REPORT` to select a new report path.

To reproduce the Skin source replay, export these three paths from commit
`26d44f121141f3738764c0b00dc09e185445341b` into a temporary directory, preserving
relative paths:

```text
tools/chembl_tool/tasks/skin_reaction/canonical_starling_source.py
tools/chembl_tool/common/molecule_identity.py
data/starling_data/skin_reaction/direct_skin_reaction/extractions.parquet
```

Then pass that directory as the Bash recipe's first argument. The Python audit
checks all three pinned hashes before executing the original pure source rules,
compares the original and UID-bearing raw rows, and records card-level conflicts
and absent canonical UIDs. It never publishes partial membership.

To rebuild an unpublished Skin successor, also export the original
`data/processed_starling/Skin_Reaction/molecule_labels.jsonl` and run:

```bash
bash experiments/evidence_library/rebuild_skin_gold_voters.sh /path/to/export
bash experiments/evidence_library/score_skin_gold_voters.sh prepare
bash experiments/evidence_library/score_skin_gold_voters.sh score  # allocated GPU
bash experiments/evidence_library/score_skin_gold_voters.sh finalize
bash experiments/evidence_library/score_skin_gold_voters.sh validate
```

Existing releases are immutable. For a fresh reproduction, select new
`RELEASE_ROOT`, `SOURCE_ROOT`, `GOLD_ROOT` and `SCORING_ROOT` destinations.
The build reuses frozen measurement/unit/reference mappings and exact compatible
pruning reviews; it does not generate replacement measurements or alter the
existing reviewed exclusion decisions.

## Uniform Skin cache location

The corrected Skin L1 prediction cache and compact scoring source now share
`data/caches/assay_reranking/active/flat_v5/gold_v1/skin_reaction/l1/assay_transfer/v9_0_2/scaffold_disjoint_parent100_v1/`.
The source rankings live in `scoring/{valid,test}/`; the prediction cache owns
`RELEASE_INDEX.json`, `evidence/` and `scaffold/{valid,test}/L1/`.
All eleven relocated files preserve their bytes, including scores, ordered
physical membership, content IDs and historical provenance. Previous corrected
cache locations resolve through `data/caches/assay_reranking/relocations.json`.
The obsolete pre-correction scaffold ranking source is retired, not aliased to
corrected results. Direct L1 always remains scaffold-disjoint; the choice of
parent-disjoint versus scaffold-disjoint retrieval applies only to indirect L2+.
