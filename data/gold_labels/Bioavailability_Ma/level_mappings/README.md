# Bioavailability Ma gold level mappings

The evidence-library pipeline publishes one immutable mapping per gold release
under this directory (`v1/`, `v2/`, and so on). Gold-label code does not derive
or rewrite levels. It may validate that the mapping's L1 membership matches the
release voter contract; L3+ assignment remains shared evidence-library logic.

`v1/level_mapping.parquet` is published from the retained original plus the 26
UIDs in `v1/reviewed_l1_corrections.json`: 13 promotions and 13 additions. All
19,479 v1 physical voters are present at L1, and every unreviewed row is unchanged.
The separate v2 publication needs only 11 of those reviewed additions; all 18,926
published v2 voters are at L1. Both versions retain identical L3+ rows.

The pre-versioning mapping is retained only as historical provenance at
`data/legacy/artifacts/evidence_libraries/v10_before_level_mapping_compatibility_20260908/source_uid_levels/bioavailability_ma.parquet`.
