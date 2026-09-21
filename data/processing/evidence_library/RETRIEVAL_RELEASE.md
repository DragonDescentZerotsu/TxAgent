# Portable retrieval release

The `retrieval-gold-v1-all-tasks-v1` GitHub Release contains the exact valid/test
SQLite indexes and evidence projections selected by
`ranked_level_retrieval_gold_v1_all_tasks_v1.yaml`. It supports reproduction of
published retrieval for BBB, Oral Bioavailability, Skin Reaction, Ames, DILI,
and Carcinogens; it does not support indexing new query molecules.

Download every Release asset into one directory, then restore into a clone of
the tagged repository revision:

```bash
gh release download retrieval-gold-v1-all-tasks-v1 \
  --pattern 'retrieval-bundle.part-*' \
  --pattern 'manifest.json*' \
  --dir /path/to/retrieval-bundle
python -m data.processing.evidence_library.retrieval_artifact_store restore \
  --archive-root /path/to/retrieval-bundle
python -m data.processing.evidence_library.retrieval_artifact_store verify-local \
  --archive-root /path/to/retrieval-bundle
```

Restore refuses to replace populated cache or projection directories unless
`--force` is explicit. Every part, reconstructed archive, restored file, and the
tracked cache configuration is SHA-256 verified.

Maintainers package from a committed revision, verify the closed local tree,
copy it to durable storage, verify that copy, and upload every file as a Release
asset:

```bash
python -m data.processing.evidence_library.retrieval_artifact_store package \
  --archive-root /local/$USER/retrieval-gold-v1-all-tasks-v1 \
  --temporary-root /local/$USER
python -m data.processing.evidence_library.retrieval_artifact_store verify-tracked \
  --archive-root /local/$USER/retrieval-gold-v1-all-tasks-v1
gh release create retrieval-gold-v1-all-tasks-v1 --verify-tag \
  /path/on/vast/retrieval-gold-v1-all-tasks-v1/*
```

The Release tag must resolve to the `source_git_commit` recorded in
`manifest.json`, and the cache configuration must be tracked at that revision.
