# Oral Bioavailability V10 known issues

## Deferred: direct qualitative HF records lose their report type

Status: deferred on 2026-09-19. Keep the affected records out of current
assay-transfer datasets until the evidence library is rebuilt correctly. Do not
introduce replacement UIDs or patch the records only in Stage 3.

During an authoritative-source refresh, `bioavailability_report_type` values of
`unspecified` are converted to null in Stage 1. The raw HF source still contains
the value. The ordinary HF source profile declares this field as literal text,
but the generic profile reconstructed by the authoritative-source adapter does
not preserve that declaration. Shared cleaning therefore treats `unspecified`
as a textual null.

The missing report type prevents the existing direct oral-bioavailability
encoder from establishing direct evidence scope. Qualitative measurements such
as `very low`, `poor`, `moderate`, `good`, and `excellent` consequently become
non-scalar, nondirect, and assay-transfer-ineligible instead of ordinal values.
The records retain their original `source_row_uid` and L2 assignment, but the
record-level Oral dataset loader excludes them at its eligibility gate.

The 2026-09-19 audit found 9,178 L2 UIDs that were direct ordinal records in the
historical ranked evidence cache but are non-scalar and ineligible in the
refreshed V10 library. All 9,178 already have semantic-bucket assignments and
retrieval eligibility. Of these, 5,533 were in the V25 training partition. This
explains most of the reduction in ordinal Oral L2 training evidence in V27.

### Required repair

1. Preserve source-declared literal taxonomy fields when reconstructing a
   `NormalizedSourceProfile` from an authoritative record contract. In
   particular, preserve `bioavailability_report_type` for
   `hf_bioavailability`.
2. Add a regression test proving that raw `unspecified` survives Stage 1 and
   activates `direct_oral_bioavailability_ordinal.v1` for a controlled
   qualitative outcome.
3. Rebuild Oral V10 from Stage 1 through canonicalization, pair buckets, the
   release-owned level mapping, and the assay-transfer eligibility sidecar.
4. Require the rebuilt source UID universe to match the pinned input universe;
   this repair must not add or substitute UIDs.
5. Rebuild dependent Oral record-level datasets and refresh provenance bindings.
   Reuse semantic assignments and cached scores only when their UID, rendered
   prompt, model, and cache identities still match exactly.

Until that repair is performed and verified, the current exclusion is the
accepted behavior; downstream artifacts should not locally re-ordinalize these
records.
