# SMILES identity review v2

This workflow independently checks uncertain or mechanically unsafe name-to-SMILES
overrides before they can change Stage 1. It never edits immutable source data, drops
an uncertain row, publishes a task policy, or rebuilds an evidence library.

## Frozen input and review population

The input is the immutable v1 review:

```text
outputs/chembl_tool/smiles_identity_audit_v2/name_smiles_comparison/v1/
  review/v1/reviewed_conflict_candidates.v1.parquet
SHA-256: 52295101f5dcf1aac876d368e09b96830be945363d54c8f194c156fe3708213c
```

`smiles_identity_review.py prepare` selects every medium-confidence override plus
high-confidence overrides with invalid SMILES, an identical-context override/reject
conflict, or the reviewed `PfOS` powder-for-oral-suspension abbreviation collision.
The frozen v1 population contains 1,222 targets: 1,205 medium and 17 high confidence.

## Reproducible workflow

Run from the authoritative TxAgent checkout:

```bash
python -m tools.chembl_tool.common.starling.smiles_identity_review prepare \
  --batch-size 100
```

This writes 13 deterministic checker packets under `review/v2/checker_packets/`.
Exact-context target groups are never split. Each card marks one target row and may
include decision-free comparison rows with the same support text.

The existing v1 reviewer is the primary reviewer. A distinct Codex agent writes one
JSONL checker decision per target under `review/v2/checker_reviews/`. Allowed values:

```json
{
  "candidate_id": "...",
  "review_role": "checker",
  "reviewer": "distinct-reviewer-id",
  "decision": "override | retain_original | unresolved",
  "override_smiles": null,
  "rationale": "row-grounded explanation of at least 30 characters",
  "reference_id": null,
  "reference_url": null
}
```

An override is allowed only when the row unambiguously measures that molecule and
the replacement is RDKit-valid. A replacement different from the frozen PubChem
candidate must include an authoritative identifier and URL. Reviewers must not infer
an unstated analyte or invent a structure. `retain_original` and `unresolved` both
preserve the stored SMILES and keep the row.

After checker coverage is complete:

```bash
python -m tools.chembl_tool.common.starling.smiles_identity_review \
  prepare-adjudication --batch-size 100
```

Every primary/checker disagreement, alternate structure, invalid prior override, and
identical-context conflict is routed to a third distinct adjudicator. Adjudicator
JSONL uses the same schema with `review_role` set to `adjudicator` and is stored under
`review/v2/adjudicator_reviews/`.

Create the unpublished proposal only after complete adjudication coverage:

```bash
python -m tools.chembl_tool.common.starling.smiles_identity_review consolidate
```

The proposal directory contains the v2 Parquet, manifest, impact report, all input
hashes, reviewer coverage, and Python/RDKit/pandas versions. Consolidation fails on
missing or duplicate decisions, reviewer reuse, invalid or no-op replacements,
unreferenced alternate structures, or any remaining identical-context conflict.
If an adjudicated target is rejected while an otherwise identical context still has
an older override, consolidation propagates the non-override decision across that
exact-context group. Propagated rows use
`review_resolution=exact_context_propagated`, name their adjudicated anchor in
`propagated_from_candidate_id`, and are counted separately in the impact report.
The report also records checker/adjudicator outcomes, task/source outcomes, and the
net override-count change from the frozen v1 ledger.

## Runtime safety and publication

Stage-1 loading validates every override string before considering confidence. An
invalid replacement fails the build. Only high-confidence overrides are eligible;
medium and low overrides are reported as quarantined and leave the original SMILES
unchanged.

The consolidated proposal received user approval on 2026-08-31. BBB,
Bioavailability, and Skin task policies now use
`review/v2/proposal/reviewed_conflict_candidates.v2.parquet` at SHA-256
`2c141da3e4b0152cd04c7741be0df95167143f615298ee3c6c99413150ecb708`.
Their active Stage-1 manifests pin the same path and hash. The proposal manifest
retains `proposal_only` as the immutable status of the consolidation command itself;
the task policies and Stage-1 manifests are the publication receipts.

The publication rebuild used the normal Stage 1 through Stage 3 task entrypoints.
For an isolated future Stage-1 verification that preserves later physical artifacts:

```bash
python -m tools.chembl_tool.tasks.bbb_martins.build_normalized_starling_evidence_library \
  --from-stage clean --through-stage clean --preserve-downstream-artifacts
python -m tools.chembl_tool.tasks.bioavailability_ma.build_normalized_starling_evidence_library \
  --from-stage clean --through-stage clean --preserve-downstream-artifacts
python -m tools.chembl_tool.tasks.skin_reaction.build_normalized_starling_evidence_library \
  --from-stage clean --through-stage clean --preserve-downstream-artifacts
```

Publication must verify that each Stage-1 manifest pins the approved v2 SHA-256,
reports zero invalid applied overrides, and leaves Stage 2 and later untouched.

## Resolved v1 defects

- The invalid BBB CCNU override strings are no longer applied.
- The `PfOS` formulation-abbreviation collisions retain the stored structure.
- The 12 v1 identical-context override/reject conflicts were adjudicated or
  propagated consistently; the v2 proposal reports zero remaining conflicts.
