# Evidence library V9 construction release

V9 adds permanent `source_row_uid` lineage without changing the selected
scientific policy of each active library. BBB inherits V8 policy; Bioavailability,
Skin, and AMES inherit V7 policy. Construction is pinned to `shared/v2` and fails
closed unless every authoritative raw row is present in the central UID ledger.

The UID is retained through cleaning, canonicalization, pair-bucket records,
deduplication mappings, and record-level assay-transfer eligibility. It is
provenance only and is not part of the LLM-visible evidence contract.

Example build entrypoint:

```bash
python -m data.processing.evidence_library.versions.v9.tasks.bbb_martins.build_normalized_starling_evidence_library
```
