# Frozen FINAL publication collection

The split archive contains the complete frozen collection: results, per-query
artifacts, traces and assembly snapshots. This is the reviewed publication copy.
Private locators and author-specific namespaces use portable, neutral names;
archive ownership metadata is normalized. Model messages, predictions, labels,
scores and benchmark splits retain their scientific content.

The archive root `final/publication_projection.json` describes the transformation.
Inventories distinguish current-copy hashes from `original_sha256`; assembly
bindings likewise retain original hashes separately. External dependencies still
require the exact versions described in the [model contracts](../../../../../../predict/retrieval/assay_reranking/MODEL_CONTRACTS.md).
See [release status](../../../../../../docs/release_refactor.md) for validation limits.

To verify and restore from a clone:

```sh
cd outputs/paper/assay_transfer_harness/evidence/final/git_bundle
sha256sum -c SHA256SUMS
cat payload.tar.zst.part-* | zstd -tq
cat payload.tar.zst.part-* | zstd -d | tar -C ../.. -xf -
```

For an approved new arm, follow the [update procedure](../../../../../../docs/finalize.md#updating-the-final-files)
and run `bash build.sh` here. Preserve published arms and refresh all manifest
bindings. Do not repack internal private paths into the publication collection.
