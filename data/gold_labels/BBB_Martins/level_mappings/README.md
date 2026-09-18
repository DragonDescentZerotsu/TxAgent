# BBB Martins gold level mappings

The evidence-library pipeline publishes one immutable mapping per gold release
under this directory (`v1/`, `v2/`, and so on). Gold-label code does not derive
or rewrite levels. It may validate that the mapping's L1 membership matches the
release voter contract; L3+ assignment remains shared evidence-library logic.

`v1/level_mapping.parquet` is published and byte-identical to its retained
`original_level_mapping.parquet`; all 7,634 v1 physical voters are present at L1.
The v2 publication is also byte-identical to the original and contains all 7,630
published v2 voters at L1.

The pre-versioning mapping is retained only as historical provenance at
`data/legacy/artifacts/evidence_libraries/v10_before_level_mapping_compatibility_20260908/source_uid_levels/bbb_martins.parquet`.
