# Evidence library V10

## Active BBB and Oral Gold-v1-keyed releases (2026-09-17)

BBB and Oral evidence-library `CURRENT` pointers resolve to v10, while all three
conditioned benchmark `CURRENT` pointers resolve to Gold v1. The active BBB and
Oral libraries use the current V10 normalization, measurement-resolution, unit,
pair-bucket, and record-pruning formula with physical Gold-v1 voters protected at
Stage 1. BBB has 496,148 Stage-3 records and exactly 7,634 L1 physical voters.
Oral has 431,440 Stage-3 records and exactly 19,479 L1 physical voters. Rows that
do not belong to a published Gold-v1 card remain retrievable outside L1;
record-level assay-transfer pruning never deletes retrieval records.
The active pruning review excludes 33 BBB and 83 Oral records from calibration;
BBB also has one task-owned manual calibration exclusion. All exclusions remain
present and retrievable, and the review manifests freeze predecessor prunes so a
refresh cannot silently reverse a reviewed exclusion.

The gold-to-record mappings are:

- `data/gold_labels/<Task>/v1/scaffold/gold_label_record_index.parquet`: one row
  per published card with nested vote units and physical `source_row_uids`.
- `data/gold_labels/<Task>/v1/scaffold/vote_units.parquet`: the frozen Gold-v1
  aggregation units. BBB has 7,634; Oral has 18,853.
- `data/gold_labels/<Task>/v1/scaffold/voter_membership.parquet`: denested
  vote-unit-to-physical-row edges. Its historical structure/hash columns remain
  audit-only; V10 protection consumes UID membership, not those values.
- `data/gold_labels/<Task>/level_mappings/v1/level_mapping.parquet`: the
  preserved base-v1 mapping used for lineage and explicit version-specific
  workflows; it is not the current BBB/Oral runtime mapping.
- `data/evidence_libraries/<task>/v10/level_mapping/records.parquet`: the
  complete mapping resolved against the rebuilt V10 Stage-3 universe, with L1
  exactly equal to Gold-v1 physical voter membership. The adjacent manifest
  pins both Stage 3 and the voter contract, and
  `data/evidence_libraries/level_mappings.v1.json` indexes both active maps.
- `data/evidence_libraries/<task>/v10/level_mapping/assay_transfer_record_eligibility.parquet`:
  one record-level scientific-suitability mark per mapped UID at every level.
  This sidecar intentionally excludes bucket support, variance, and calibration
  gates.

The immediately preceding active V10 trees, which omitted record-pruning inputs
from final Stage 3, are preserved intact at
`data/legacy/artifacts/evidence_libraries/v10_before_pruning_restore_20260917/`.
The earlier pre-main-universe rebuild remains at
`data/legacy/artifacts/evidence_libraries/v10_before_main_universe_repair_rebuild_20260917/`.
The superseded local-path pruning manifests are retained compactly at
`data/legacy/artifacts/evidence_libraries/v10_before_pruning_provenance_repair_20260917/`;
the repair changed manifests only and left every scientific artifact byte-identical.
Existing cache-matched SQLite releases remain immutable and must be reused only
when their input manifests match these active row and level maps.

## BBB and Oral protected Stage 1

The active builds use `stage1_exact_deduplication.v3`. Gold-v1 voter membership
protects the pinned main-universe structure for every physical voter, while all
frozen reviewed structure mappings apply to nonvoters and all non-structure
reviewed corrections remain active. Protected duplicate members all survive.
Exact identity remains repaired SMILES + PMID + canonical measurement + canonical
unit, with configured scientific fields preventing distinct arms from
collapsing. Gold condition/context is never part of that identity or the later
pair-bucket key. The ordered contract is in `ARCHITECTURE.md`.

The completed BBB and Oral LLM measurement decisions were replayed from the
hash-pinned durable artifacts under
`data/artifacts/evidence_library_assets/<task>/main_universe_replay_v1/`; no new
model calls or new scientific decisions were made. The final parent-identity
audit found every protected UID equal to its pinned main-universe parent and no
Gold card containing multiple main-universe parents. One BBB and eleven Oral
protected rows differ only in equivalent SMILES serialization.

The active semantic releases preserve every compatible prior assignment and
review only level-local atoms newly exposed by the Gold-v1 level map. BBB
reviewed 867 L2-L4 atoms into 119 source-local buckets (one attached to an
existing semantic bucket and 118 created); Oral reviewed four atoms into four
new semantic buckets. Their paid request caches and reviewed decisions are kept
under each selected `gold_v1_protected_incremental_v1` generation.

## Imported Stage 1 releases

Ames, DILI, Carcinogens, and Skin_Reaction use the pinned upstream level mapping
from commit `26d44f121141f3738764c0b00dc09e185445341b`. Their Stage 1 scope is exactly
the cleaned acquisition rows whose permanent `source_row_uid` occurs in that
retrieval-eligible mapping. Unmapped clean rows are retained in the compatibility
exclusion sidecar; this is record-level source scoping, not pair-bucket pruning.

The live core-stage trees are local builds. Portable Git payloads contain deterministic
tar+zstd archives split below 100 MB and can be packaged or verified with
`python -m data.processing.evidence_library.versions.v10.stage1_artifact_store`.
All four imported-task profiles are Stage-1-only. Skin Reaction's preceding coherent
Stage-2 bundle is preserved under
`data/legacy/artifacts/evidence_libraries/skin_reaction/v10_stage2_before_stage1_exact_dedup_20260917/`.

## Historical Skin Reaction Stage 2

Skin Reaction resumes from its hash-verified V10 clean stage and publishes only
the canonical normalization stage:

```bash
python -m data.processing.evidence_library.versions.v10.tasks.skin_reaction.build_normalized_starling_evidence_library \
  --from-stage normalize --through-stage normalize \
  --validation-level full --cache-mode off --workers 64
```

The build reuses the frozen Skin measurement-resolution, exact-unit, auxiliary,
reference-semantics, and assay-transfer policies. It must not rebuild Stage 1 or
add or revise any SMILES repair, identity-audit, or name/SMILES conflict asset.

The 2026-09-15 EPYC build retained all 66,962 cleaned records, including 33,542
finite scalar records and 31,451 absolute continuous measurements. Measurement-unit
and final assay-transfer validation both passed. The canonical library retains the
72 records excluded from assay-transfer calibration by the frozen measurement policy
and the 149 fail-closed records without an assay-transfer reference-semantic mapping.
That snapshot is now historical because the ordered Stage 1 successor changed its
record universe. No replacement Stage 2 or pair buckets have been built, and
`CURRENT` remains `v9`.

## Skin Reaction V10 Stage 1 publication (2026-09-17)

The pre-dedup clean universe contains 66,962 scoped records and no structure
rejections. Its measurement router partitions them into 23,402 deterministic
accepts, 8,985 extraction candidates, and 34,575 terminal numeric rejections.
The completed extraction is release-owned at
`data/evidence_libraries/skin_reaction/v10/measurement_resolution_v8/`; all
8,985 candidates are covered exactly once, with 6,962 `ok`, 668 `relative`,
901 `unavailable`, and 454 `unsure` decisions.

Stage 1c unit reconciliation reuses the 8,178-entry v2 reviewed map instead of
starting over. The successor `exact_measurement_unit_map.v3.json` covers all
source-declared units, noncanonical deterministic units, and successful LLM
units in the V10 clean universe. Its adjacent manifest pins the clean,
extraction, and v2 hashes and records the reviewed successor decisions. A
strict in-memory application over all 66,962 rows produced 30,363 mapped
scalars and one explicit non-unit exclusion, with no missing extraction or unit
mapping. Stage 1d uses the task-local
`skin_reaction_stage1_exact_deduplication.v1`
policy layered on the immutable BBB/Oral v3 base contract. It removes 295 exact
within-source rows: 263 direct Skin Reaction rows and 32 sensitization-AOP rows.
Empty cross-source scientific-field intersections never deduplicate; this
preserved 290 initially proposed matches whose real records included different
populations, concentrations, or study arms.

The final Stage 1 contains 66,667 records and 30,068 mapped canonical
measurements, with zero structure rejections and all row-conservation,
UID-uniqueness, measurement-coverage, and unit-coverage checks passing. The
records SHA-256 is
`8516f104d5582e949a76081f10eb76f8989f95be4b7d571a0d48959227b285c5`;
the duplicate-audit SHA-256 is
`6146b533b137d71e2a675b052d6f0c5850895dbd083c49d3278365392f5a1fef`.
An independent EPYC replay reproduced every scientific output hash. The active
portable bundle is Stage-1-only and `CURRENT` remains `v9`.

## DILI Stage 1 measurement resolution

DILI V10 Stage 1 completed and was verified on 2026-09-08 with
`/usr/bin/python` on `epyc-4-10`. The frozen six-schema source snapshot contains
1,645,109 rows; cleaning retains 1,627,666 rows and records 17,443 structure
rejections. Each schema uses its own outcome field. Scientific endpoint aliases
are absent: the inherited `canonical_endpoint_name` field is only the cleaned raw
endpoint literal, the fixed base-source identity, or a missing-value batching key.

Routing partitions all cleaned rows into 140 narrow deterministic V5 potency
accepts, 478,469 model-extraction candidates, and 1,149,057 numeric rejections.
The completed extraction mapping is
`tasks/dili/data_processing/measurement_resolution_v1/measurement_resolution.parquet`
(SHA-256 `d8d979c6415b4385a64e3346da00a64b91b7e0a001d779782dd1e43e27d3c1af`).
It contains 55,887 `ok`, 123,085 `relative`, 121,408 `unsure`, and 178,089
`unavailable` rows, with exact `source_row_uid` coverage and no rejected or
terminal-failure rows.

Inference used one 512-worker pool against `http://dgx027:50001/v1`, requesting
and receiving only `deepseek-ai/DeepSeek-V4-Flash-0731` with no credential,
temperature 0, and reasoning effort `none`. The reviewed precision-first gate
recovered 197/335 usable outcomes and matched 195/197 complete pairs among `ok`
predictions. The detailed contract and limitations are in `tasks/dili/README.md`;
the final audit is adjacent to the mapping as `provider_endpoint_receipt.json`.
Stage 2 remains unimplemented, the incomplete-build marker is retained, and no
DILI `CURRENT` pointer has been created.

## BBB complete training-context compatibility mapping

The selected BBB review asset is `reviewed_name_smiles_conflicts.v5.parquet`.
The remaining-49 audit approved 16 repairs and deferred 33 record-local overrides:
23 require new parents, nine lead to held-out parents, and one creates conflicting
votes. Together with the earlier 21 deferrals, 54 overrides are deferred. Original
scientific review decisions and corrected structures remain in the prior assets
and the exact-UID compatibility ledger; indirect records are unchanged.

The verified local rebuild has exactly 33 additional `canonical_smiles` changes, no other
record-field or membership changes, and 498,507 Stage-3 records. The 641 reviewed
pruning rows are identical. Five calibration buckets change only their
distinct-molecule counts; eligibility is unchanged.

The new membership mapping resolves all 4,754 frozen training references:
4,731 unchanged and 23 redistributed into 21 existing contexts. Joint vote checks
preserve labels and eligibility for surviving contexts; 16 empty source contexts
retire from the derived membership, not from frozen gold files. There are no
unresolved references and no held-out destination parents.

The earlier 27-context vote-replay warning was an adapter mismatch, not a SMILES
effect. The library's v2 adapter misses 31 source-index-reviewed directions from
the frozen v4 gold contract. Applying those existing decisions in the membership
audit reproduces all 3,053 original context vote counts exactly. The library vote
sidecar itself is unchanged; no new label decisions were made.

The mapping is wired into BBB's active harness through `--gold-context-mapping`.
Morgan L1 passes offline retrieval on all 397 validation queries; assay-transfer
and joint require either rescoring five changed vote-percentage contexts or the
explicit `--allow-frozen-l1-vote-scores` approximation accepted by the user.
Scoring caches were not rebuilt. Gold, the mapper, and `CURRENT` remain unchanged. Verification receipts
and the mapping are under
`outputs/paper/assay_transfer_harness/joseph/bbb_remaining49_rebuild_20260908/`.
The preceding local V10 is preserved under
`data/legacy/artifacts/evidence_libraries/v10_before_bbb_remaining49_20260908/`.

## Historical BBB first L1 case-set compatibility deferrals

The selected BBB review asset is now `reviewed_name_smiles_conflicts.v4.parquet`.
At user direction, 20 additional record-local overrides were deferred from the
27-record L1 coverage case set: 19 whose corrected parents were absent from the
frozen training pool (including two held-out parents), and one introducing
conflicting acetylcholine votes. Row 32568 remains deferred, for 21 total.
All 3,693 remaining effective repairs are unchanged, including indirect records.
These are compatibility deferrals, not reversals of the chemical review evidence.

BBB V10 was rebuilt with exactly 20 additional `canonical_smiles` changes and
no other record-field or membership changes. The 498,507 Stage-3 records remain.
Three calibration buckets have different distinct-molecule counts; their
eligibility and all other calibration fields are unchanged. Prior artifacts
are retained under
`data/legacy/artifacts/evidence_libraries/v10_before_bbb_l1_compatibility_20260908/`.

The UID-backed training-context mapping redistributes seven approved records
into six existing contexts, preserving their labels and eligibility. It is
explicitly **partial and not wired into the harness**: the complete training
audit exposed 49 additional changed-parent source references outside that case
set. All 4,754 training source references resolve, including the alternate
`starling-labs/BBB:row:` ID prefix; none are missing or unmapped in this BBB audit.
Gold files, mapper, `CURRENT`, scoring caches and checkpoints remain unchanged.
Results and the partial mapping are under
`outputs/paper/assay_transfer_harness/joseph/bbb_l1_context_compatibility_20260908/`.

## Historical first deferral: BBB row 32568

At user direction on 2026-09-08, only the lomustine SMILES override for
`sr_9641f68913ff4d6d9365662289c68ec5` (`direct_bbb`, source row `32568`)
is temporarily disabled. The selected BBB review asset is
`reviewed_name_smiles_conflicts.v3.parquet`; its `reject` decision rejects the
proposed override, not the source record. The original lomustine rationale is
retained, and the unchanged v2 asset remains historical review provenance.
All 3,713 other effective BBB SMILES overrides are unchanged.

The exact before/after exception is recorded in `level_mapping_compatibility.json`.
The source structure `NC(=O)N(CCCl)N=O` now agrees with frozen gold at L1.
This is a compatibility deferral, not a chemical reassessment of the review.
BBB V10 still contains 498,507 Stage-3 records; only this row's
`canonical_smiles` changes. Calibration buckets, pruning, gold, mapper and
`CURRENT` are unchanged. Frozen measurement extraction is validated against its
hash-pinned original Stage-1 archive, not relabeled as newly extracted data.

The prior V10 is preserved under
`data/legacy/artifacts/evidence_libraries/v10_before_bbb_32568_reversion_20260908/`.
The pending BBB three-pool cache manifests now point to that identical archived
input at their original hashes; scores, assignments and model identities were
not changed. The verification and remaining L1 coverage gaps are recorded in
`outputs/paper/assay_transfer_harness/joseph/bbb_32568_smiles_reversion_20260908/`.

## Historical level-mapping compatibility rebuild (superseded)

The 2026-09-08 compatibility build excluded 88 BBB and 132 oral acquisition
UIDs that lacked imported level assignments. That exclusion is no longer part
of the active architecture: the reviewed level maps now include every retained
BBB and oral UID, and Stage 1 has no level-mapping row-drop path. The historical
artifacts remain immutable provenance only.

The original V10 artifacts are hash-verified under
`data/legacy/artifacts/evidence_libraries/v10_before_level_mapping_compatibility_20260908/`.
The rebuild preserves existing source SMILES serialization after verifying
isomeric equivalence; this retains BBB's previously documented 711-row carry-forward.
Raw acquisition inputs, the level mapper, published training sets and old score
caches remain unchanged. Rebuilt libraries are verified in staging before replacement.

The verified local replacement has 498,507 BBB and 435,372 oral Stage-3 rows,
with zero unmapped UIDs. `CURRENT` stayed BBB V9 and oral V10. Receipts are in
`outputs/paper/assay_transfer_harness/joseph/v10_level_mapping_compatibility_rebuild_20260908/`.

## Historical Oral V10 construction before the compatibility rebuild

Completed and verified on 2026-09-06 using `/usr/bin/python` on `epyc-4-10`.
The unpublished build contains 435,603 Stage-2 records and 435,465 Stage-3
records after 138 audited duplicates. Its 26,104 pair buckets include 691
calibration-eligible buckets after pruning (692 before). All 85 pruned records
remain in the library. The reused reference map leaves 3,855 records explicitly
unknown and ineligible; auxiliary-column mapping coverage is complete.
The compact build receipt is
`data/artifacts/evidence_library_assets/bioavailability_ma/v10_percentage_delta/completed_build.json`.

Oral uses the completed extraction at
`data/artifacts/evidence_library_assets/bioavailability_ma/v10_percentage_delta/single_measurement_repair/measurement_resolution.parquet`
and the task-local `bioavailability_unit_reconciliation.v1.json`. The extraction
covers all 188,806 routed rows. Unit reconciliation maps 5,207 reviewed strings
to 2,478 units without changing numerical values. Other column and reference
mappings are reused from the copied release.

As in BBB V10, the legacy upstream raw/log10 measurement policy is disabled.
Canonical evidence retains the resolved numerical value and unit; selected
logarithmic distance geometry belongs to downstream assay-transfer construction.
Unmapped reference scopes remain explicitly unknown and assay-transfer-ineligible
under the existing reference policy. Pruning separately excludes individual
records from calibration while retaining them in the evidence library.

The construction sequence is Stage 2, preliminary Stage 3, record-pruning review,
then final Stage 3. Pruning uses `gpt-5.4-mini` through the first OpenAI key with
its own `oral_v10_pruning_20260906` ledger capped at 10 million tokens. Resume
reuses completed review responses and the same ledger. `CURRENT` remains V9.

## Extraction interface and historical development

Measurement extraction uses batch-local `r1`, `r2`, ... IDs on the model wire.
The request cache records the `batch_local_ids.v1` mapping to immutable record
IDs; assignments retain full IDs and raw API responses remain unchanged.
Missing, duplicate or unknown response IDs fail closed. Existing successful
long-ID cache entries are reused without inference or migration.
`--retry-failed --retry-max-attempts 0` explicitly permits unlimited technical
retries; omitting the override preserves the task's attempt cap. Exit code 4
means unresolved rows are capped and no new requests can be sent; controllers
must not treat it as retryable exit code 3.

V10 is an independent copy of V9. For BBB and oral bio, previously accepted
percentages now route to measurement extraction with additional source context.
Other measurement decisions, structure cleaning, canonicalization maps and
downstream reference/assay-transfer policies are unchanged. Do not use the copied
non-target task defaults to publish another task.

Oral V10 is being completed from the V9 Stage 1 baseline; it does not depend on
V9 Stage 2 finishing. The completed 100-row-per-task percentage pilots and reused
assignments are indexed in
`data/artifacts/evidence_library_assets/<task>/v10_percentage_delta/`.
`pilot_gold.jsonl` is source-only Codex review, not independent human gold.
`audit.json` preserves the pilot disagreements and provider/usage evidence.
The partial mappings are explicitly incomplete, not publication inputs.

The following pilot and launch notes describe historical development states;
the completed extraction selected above supersedes their paused/partial status.

Oral's V6 experiment used `bioavailability_v6.jinja`: a wording-only revision
for selected outcomes, standard denominators, bounds and printed scales. Matched
GPT/Baidu-DeepSeek replays of the unchanged 100+500 development-gold cases are in
`data/artifacts/evidence_library_assets/bioavailability_ma/v10_percentage_delta/V6_REPLAY.json`.
GPT scored 99/100 and 480/500; DeepSeek scored 100/100 and 488/500. These are
development-set results, not independent validation or automatic promotion.
Full extraction remains paused; evaluation outputs do not replace the reused
production assignments. Earlier prompt templates remain as paid-cache provenance.

The selected prompt is now `bioavailability_v7.jinja`, adding one paragraph about
defined fitted parameters and oral bioavailability coefficients. Gold
`tests/chembl_tool/common/measurement_resolution_quality/gold/bioavailability_ma.v10.1.jsonl`
corrects only Emax 0.761 and oral cyclosporine bioavailability 0.51 to `ok`, with
source-review rationale and provenance. The original V9.2 gold is unchanged.
`V7_REPLAY.json` in the same artifact directory separates re-scoring the V6 run
against corrected gold from the new 500-row Baidu DeepSeek replay.
Gold `bioavailability_ma.v10.2.jsonl` additionally accepts the Emax source's
literal `unitless` spelling alongside `ratio`; this scoring-only correction was
reviewed after the V7 output and is recorded separately. Against that gold, V6
scores 490/500 strict and 495/500 scalar-output agreement; V7 scores 493/500 and
497/500. The unadjusted V7 score of 492/500 against V10.1 is also preserved.
These changes do not modify the 700 reusable production assignments or launch
full extraction.

Oral's full resume must use `--base-mapping` for the 700 completed assignments,
a new request-cache directory, and `--budget-ledger-dir` pointing to the original
V9 two-key ledgers with their existing epoch. Do not reset the 10M-per-key caps.
`--two-key-baidu-run` uses only keys one and two, then `baidu/fp8` with fallback
disabled. No CURRENT pointer has been changed.

## Retained V9 lineage contract

V9 adds permanent `source_row_uid` lineage without changing the selected
scientific policy of each active library. BBB inherits V8 policy; Bioavailability,
Skin, and AMES inherit V7 policy. Construction is pinned to `shared/v2` and fails
closed unless every authoritative raw row is present in the central UID ledger.

The UID is retained through cleaning, canonicalization, pair-bucket records,
deduplication mappings, and record-level assay-transfer eligibility. It is
provenance only and is not part of the LLM-visible evidence contract.

Oral Stage 1 build entrypoint (no paid inference):

```bash
python -m data.processing.evidence_library.versions.v10.tasks.bioavailability_ma.build_normalized_starling_evidence_library --through-stage clean --workers 8
```
