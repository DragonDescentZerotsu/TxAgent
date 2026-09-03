# Sensitization species-context reconciliation

This directory holds the human-gated global review of the completed Skin sensitization species mapping.
It reconciles and audits the local cluster decisions without making another model or API call. The v3
candidate remains frozen provenance. The reviewed v4 proposal was explicitly approved and published on
2026-08-13; Stage 02 accepts it only through the matching `PUBLICATION_RECORD.json`.

## Inputs and label layers

The immutable inputs are the completed responses in `../cluster_cache/`, the v3 runtime section in
`../../globally_reconciled_auxiliary_value_mapping.json`, and `../mapping_manifest.json` with their recorded
hashes. Review always uses the final post-validation v3 label, rather than the raw GPT response alone.

The workflow keeps three label layers separate:

1. **Cluster-local output:** the raw label returned for each structured
   `(assay_type, experimental_conditions, support_text)` packet.
2. **Validated v3 candidate:** the label after literal-support checks, fail-closed demotion, and controlled
   base-species alias normalization. It contains 44,919 tuples: 24,219 non-null and 20,700 null.
3. **Globally reconciled v4 mapping:** the independently reviewed proposal, published after explicit user
   approval on 2026-08-13.

## Review contract

All 20,700 v3 null assignments are frozen. They are retained for retrieval and cannot be promoted during
this review. Every one of the 24,219 non-null assignments must be present in exactly one review packet and
receive:

1. an independent primary Codex review;
2. an independent checker review by a different Codex reviewer; and
3. a third, distinct Codex adjudication for every disagreement.

The reviewer JSONL files are written manually from the complete review packets. This phase must not call
OpenAI, LiteLLM, or another model API. Packet, assignment, and reviewer coverage must be complete and
hash-checked before a proposal can be assembled.

Allowed decisions are deliberately narrow:

- retain a supported controlled base-species label;
- normalize a supported synonym such as `bovine` or `cow` to `cattle`;
- correct the label to another base species explicitly identified as the measurement-producing subject,
  donor, tissue, or cell system; or
- demote the label to null when the mention describes a reagent, background comparison, conflicting or
  pooled species, or otherwise cannot be attributed unambiguously to the measurement.

Reviewers must not promote a v3 null, infer from outside knowledge, create strain/population/reagent labels,
create a pooled multi-species label, or change any namespace other than
`sensitization_aop/global_species_context`.

## Artifacts and publication gate

The reconciliation artifacts use the following layout:

```text
provenance/       frozen v3 assignment and cluster hashes
review_packets/   complete, disjoint, cluster-atomic reviewer inputs
reviews/          primary, checker, and adjudicator JSONL plus coverage summaries
proposal/         replayable unpublished v4 mapping, deltas, manifests, and integrity audit
```

An interrupted or incomplete review changes neither the v3 runtime mapping nor downstream artifacts. After
the proposal passes complete coverage, reviewer-independence, replay, and hash audits, explicit human
approval is still required to publish it. Publication atomically replaces only the sensitization species
section, records the v3 and v4 hashes and approval, and then rebuilds Skin Stages 02-09 exactly once.

The v4 proposal was approved and published on 2026-08-13. Its proposal SHA-256 is
`76ebdecfad4eee87e7f338bcf446f62e6d54151cc60a222b4daf7c1481a455dc`; the formatted runtime mapping
SHA-256 is `152bb6e26658a6d57c6c6f38aa32da34a862b9f9b74db5a740f660bb7a81bf07`. The rebuilt v7 artifact
removed one previously eligible sensitization record and its singleton Stage-04 bucket. The 889
calibration-valid buckets, 643 ordinal category-CDF buckets, and 232 continuous value-CDF buckets were
unchanged. Stages 02-09 were packaged and both local and tracked stores passed checksum verification.

`../../auxiliary_reconciliation_v2/` is immutable historical lineage. It documents the earlier all-namespace
reconciliation, but it is neither rewritten nor used as the scientific input for this focused v4 review.
