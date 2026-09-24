# Skin indirect ML level-mapped subset

This audit-only successor retains only assay-transfer-eligible Stage-3 records whose source UID is present in the release-owned Skin level map. It groups measurements by normalized parent SMILES and intrinsic pair bucket, retaining duplicate measurements and buckets with at least 10 distinct mapped parents. The original artifact was reproduced exactly before filtering. See `summary.tsv` and `manifest.json` for counts and pinned hashes.
