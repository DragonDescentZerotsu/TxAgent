# Stage-01 source-value audit corpus

This corpus freezes two cleaning-audit rounds. Round 1 contains 1,000 records
each for BBB Martins and Bioavailability Ma. Round 2 adds a disjoint 1,000
records each for BBB Martins, Bioavailability Ma, and Skin Reaction, for 5,000
total cases. It tests the authoritative Stage-01 `measurement_text` and
`unit_text` values while requiring exact preservation of the already-ingested
`support_text`; it contains no Stage-02 canonical values.

The v3 fixture schema retains the v2 migration of 734 sampled HF rows from the
historical `direct_hf` / `hf_nondirect_bioavailability` partitions to their physical
source ID, `hf_bioavailability`. Their row numbers now refer to the complete
pinned source. Directness is a Stage-02 row scope derived from
`bioavailability_report_type`, so it is intentionally outside this
Stage-01-only corpus. V3 additionally makes support text immutable after the
common ingestion-only Unicode and whitespace cleanup.

Round 1 uses seed `20260805`. To ensure rare edits were actually reviewed, it
includes the complete census of records changed by the first Stage-01 build,
then fills each task to 1,000 with a seeded random sample without replacement
from unchanged records. Round 2 uses seed `20260806` and samples without
replacement after excluding every existing corpus record. None of the 3,000
new rows changed under the cleaner. The accepted states were frozen after
checking that every edit preserved the reported digits and units, that support
text remained exact, and that cleaning introduced no adjacent-percent
concatenation.

Where punctuation-only measurement presentations are scientifically
equivalent, a case may list more than one acceptable state. Support text has
exactly one allowed state. Unambiguous values remain exact so the flexible
expectation format does not authorize arbitrary rewriting.

The audit found one unsafe oral-bio edit: `40-70,50` had become `40-70.50`
without independent support-text corroboration. The cleaner now preserves this
ambiguous source form, and the corpus carries it as an explicit fail-closed
regression.
