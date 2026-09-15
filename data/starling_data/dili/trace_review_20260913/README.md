# DILI trace source review, 2026-09-13

Bounded source-only repair authorized from the DILI trace discussion. The replay
reference is **`reasoning_prompt_v5_scaffold_20260913`**, prompt
`progressive_evidence_revision.v5`; the earlier legacy-v2 trace diagnostic is
historical and is not the replay configuration.

`content_repairs.json` binds 13 decisions to the previous final-source SHA256 and
immutable source payloads. Decisions were fixed before any fresh model call:

- PMID 26948886: the review really names naloxone, while the historical FDA
  hepatotoxicity warning concerns excessive naltrexone exposure. Exclude this
  non-attributable association; do not silently reassign its frozen vote.
- PMID 24575896: remove the unsupported reactive-acylglucuronide annotation from
  the naloxone extraction and quarantine the record. Naloxone has no carboxyl
  group; an independent primary study identifies naloxone-3-O-glucuronide.
  The original numerical table was not obtained. Retained values are not newly
  verified, and no negative DILI outcome is inferred.
- PMID 21940394: correct APO to apocynin (PubChem CID 2214), preserving the raw
  acquisition. Exclude both sperm ROS records from DILI retrieval.
- Replacement review exposed PMID 25446915: the original paper explicitly tests
  baicalein-6-alpha-glucoside (BG). All 12 same-PMID records were inspected;
  nine BG records carry either a sugar-free positional isomer or baicalin's
  glucuronide structure and are excluded. Three correctly bound baicalein rows
  are retained. This is an identity review, not certification of all study
  measurements or a new global macrophage-evidence exclusion rule.

Raw records, historical voter membership, gold labels, and both benchmark split
schemes remain frozen. A historical voter can become ineligible for retrieval;
this does not amend the frozen benchmark. Source validity is judged from the
paper/chemical identity, independently of query labels or observed accuracy.

Use the existing `stage_new_task_retrieval.apply_content_repairs()` API and
`build_assay_family_catalog`, `assay_retrieval build-index`, and
`validate_new_task_retrieval_identity` entrypoints. Source and index validation
receipts document the release. The official source remains
`../retrieval_final/`; staging directories are not alternative benchmark roots.

The replay and exact selected-input audit live in
`outputs/paper/dili_trace_source_replay_20260913/`. Query prior and tool outputs
are frozen from the original v5 run; common query-analog tools are reused.
Only the changed level and its downstream progressive states require inference.
The initial four-record preparation was invalidated before inference when BG
replacement errors were confirmed; it is not an experiment result.
