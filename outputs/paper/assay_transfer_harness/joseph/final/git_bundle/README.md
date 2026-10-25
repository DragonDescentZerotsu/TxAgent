# Frozen FINAL collection in Git

The payload.tar.zst.part files contain the complete final collection,
including every result, per-query run artifact, and trace. The readable tree
beside this directory is ignored by Git and is restored from these parts.
The manifest pins the archive, parts, and collection index.

This is the original reproduction snapshot, **not an anonymous distribution**.
Its manifests and run records retain private machine paths; frozen assembly
source also contains identity clues. Preserve these original bytes for hash
verification. A separately reviewed publication copy must identify its changes
and preserve model messages, predictions, labels, and scores. Moving a path does
not authorize replacing its recorded source hash with a hash of different data.
The [release status](../../../../../../docs/release_refactor.md#可维护性与匿名发布)
tracks the remaining checks; an internal path-conversion draft is not a
replacement for this archive or a resume-ready bundle.

Follow the [repository update procedure](../../../../../../docs/finalize.md#updating-the-final-files)
to add an approved arm and update the collection index. Then run `bash build.sh`
in this directory. The parts are ordinary Git files, so `git add` picks up the
refreshed bundle without force. Preserve published arms.

To verify and restore from a clone:

~~~bash
cd outputs/paper/assay_transfer_harness/joseph/final/git_bundle
sha256sum -c SHA256SUMS
cat payload.tar.zst.part-* | zstd -tq
cat payload.tar.zst.part-* | zstd -d | tar -C ../.. -xf -
~~~
