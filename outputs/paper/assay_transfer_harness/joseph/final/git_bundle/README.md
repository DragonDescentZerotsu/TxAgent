# Canonical Joseph final collection in Git

The two payload.tar.zst.part files contain the complete final collection,
including every result, per-query run artifact, and trace. The readable tree
beside this directory is ignored by Git and is restored from these parts.
The manifest pins the archive, parts, and collection index.

After adding or changing a final trace, run bash build.sh in this directory.
The parts are ordinary Git files, so git add picks up the refreshed bundle
without force. Keep this collection immutable after publication; use a new
successor collection for scientific changes.

To verify and restore from a clone:

~~~bash
cd outputs/paper/assay_transfer_harness/joseph/final/git_bundle
sha256sum -c SHA256SUMS
cat payload.tar.zst.part-* | zstd -tq
cat payload.tar.zst.part-* | zstd -d | tar -C ../.. -xf -
~~~
