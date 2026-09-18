# SMILES repair v2 and downstream rebuild plan

This change repairs molecule identity at the source-record boundary, then rebuilds
gold and retrieval artifacts from that same reviewed identity. Gold construction
remains independent of assay-transfer calibration and V10 retrieval selection.

## Phase 1: reviewed source identity

1. Keep the immutable raw Parquets and the published v1 gold datasets unchanged.
2. Apply the complete reviewed BBB and Bioavailability repair ledgers, including
   the previously deferred rows and high-confidence name/SMILES conflict decisions.
3. Retain the stored SMILES when the named material is only a salt, isotope,
   stereoisomer, formulation, or close-core variant. Replace it only when the
   stored molecule is a different chemical identity. Preserve exclusions whose
   cited paper describes a different referent.
4. Publish field-level repair audits and pin every raw input and review-ledger hash.
5. Stage 1 is the single repaired, level-independent source universe. It stores
   the frozen canonical endpoint, main measurement, and canonical unit before
   any duplicate decision. Keep retained HF and local rows source-native; do not
   merge their fields into a synthetic canonical claim before gold labeling.
6. Deduplicate every BBB and Oral source both across and within source datasets.
   Require present, literally equal post-repair canonical SMILES, PMID, canonical
   measurement, and canonical unit. Within one source, require every configured
   raw scientific field to match; across sources, compare the intersection of
   configured raw scientific field names. When exact duplicates contain a frozen
   v1 physical voter, retain a frozen voter before applying the lexicographic UID
   tie-break; this prevents a non-gold duplicate from deleting an inherited
   voter. Publish a complete physical-row lineage audit.

Bioavailability has six repaired Nitrendipine source rows. Five are restored gold voters; the
sixth remains a repaired source record but is not promoted merely because its
identity is now correct.

## Phase 2: conditioned gold v2

1. Recover the exact published v1 voter universe before reading Stage 1. Expand
   Oral canonical claims to distinct HF/local physical members. Intersect this
   lineage with retained repaired Stage-1 rows; never admit other label-eligible
   Stage-1 records.
2. Apply the current physical-row label adapter as a hard gate. A lineage member
   that fails is dropped and written to the lineage rejection audit; it does not
   inherit the old canonical-claim label.
3. Preserve the split and row ID of every surviving v1 parent-condition card. A
   new card inherits an existing parent or scaffold split; genuinely new
   scaffolds deterministically fill retired v1 split deficits, then go to train.
   For an existing v1 null-condition card whose majority label is unchanged,
   allow a two-thirds migration-preservation agreement floor. Keep the rebuilt
   physical voters and vote mean unchanged, and continue to reject exact ties.
4. Rebuild the reviewed external-condition rows. Reuse a verdict only when the
   source lineage, condition signature, label, and label method are semantically
   unchanged; otherwise stop for review.
5. Publish `data/gold_labels/{BBB_Martins,Bioavailability_Ma}/v2/scaffold/` with
   overlap checks, input hashes, migration receipts, and a complete
   `voter_membership.parquet`.
6. `voter_membership.parquet` contains every surviving, label-eligible physical
   `source_row_uid`. It is the sole L1 membership authority; capped parent-level
   source-record lists and V10 inference records are not valid substitutes.
7. Switch each `CURRENT` pointer only after v2 is complete and independently
   validated. Skin remains on v1.

## Phase 3: level mappings and V10 evidence

1. Rebuild the complete BBB and Oral source-UID level mappings from gold-v2 voter
   membership. L1 is exactly the accepted physical voter set; remaining direct
   records enter the task policy's next direct level. Indirect levels retain their
   existing scientific family rules.
2. Rebuild V10 Stage 1 through the pair-bucket stage from the repaired sources.
   Regenerate record-level assay-transfer pruning with `reasoning_effort=high`
   against the new record hashes; never reuse hash-pinned decisions from the old
   Stage 2.
3. Validate parent ownership for every direct card, no duplicate retained UID, no
   label/condition/partition conflicts, complete UID coverage, and zero cross-split
   parent or scaffold overlap.
4. Replace the active BBB and Oral V10 evidence-library artifacts only after a
   staged build, hash receipt, and rollback snapshot. Historical releases remain
   immutable.

## Phase 4: training data, models, and caches

1. Recreate the context-conditioned molecule-assay-transfer training datasets in
   `starling_assay_transfer` from gold v2 and the rebuilt V10 evidence library.
   Pin the gold-v2, level-map, evidence-library, prompt, and source-repair hashes.
2. Confirm the v10.3 BBB and Oral prompt renderers display BBB canonical molecule
   identity and Oral source identity as intended, while all grouping and joins use
   the shared normalized parent identity.
3. Retrain successor models rather than relabeling the existing BBB step-120 and
   Oral step-140 checkpoints. Preserve those checkpoint revisions as historical
   baselines.
4. Build new L1 caches for BBB and Oral from the successor checkpoints and gold-v2
   mappings. Cache publication must fail closed on missing voter membership,
   parent mismatches, incomplete assay scores, or version/hash disagreement.
5. Generate the corrected L1 and V10 evidence-library pools only after the caches
   and successor model identities are frozen. The other three previously discussed
   pools are deferred because they are not L1 inputs.

## Current staged checkpoint

- Repaired Stage 1 is independent of the source-UID level mapping.
- The earlier 1,630-delete/867-retained Oral overlap split is superseded. The
  rebuilt Stage 1 applies the canonical measurement rule uniformly across and
  within sources.
- The final node-local exact-dedup artifacts contain 496,092 BBB rows after
  2,506 duplicate removals and 431,006 Oral rows after 4,597 removals. These are
  published as the gold-v2 Stage-1 snapshot, but have not replaced the active
  full V10 evidence libraries.
- The fifth Nitrendipine voter (observed F=54% under cirrhosis) is one of the
  2,452 exact migrations and now attaches to its repaired parent. The predicted
  57% cirrhosis row remains non-voting.
- The frozen lineage contains 7,634 BBB physical members and 18,853 Oral claim
  votes expanded to 19,479 physical members. Exact dedup removes 2 BBB and 495
  Oral inherited duplicates. The current row adapter then drops 51 Oral members;
  five additional Oral rows have excluded external conditions. The resulting
  voter sets are 7,632 BBB and 18,928 Oral rows.
- The staged v2 aggregation has no v1-to-v2 label flips. BBB publishes 3,832
  cards; Oral publishes 2,484 cards, including two audited scaffold-driven split
  reallocations.
- Versioned v2 gold is published under each task's `v2/scaffold/` directory.
  Both `CURRENT` files deliberately remain `v1`; promotion is a separate reviewed
  action after downstream level mappings and retrieval artifacts are rebuilt.

Any source row whose repaired identity changes its scientific label, condition
interpretation, or paper referent requires explicit review before publication.
