#!/usr/bin/env bash
set -euo pipefail

bundle_dir="$(cd "$(dirname "$0")" && pwd)"
joseph_dir="$(cd "$bundle_dir/../.." && pwd)"
work_dir="$(mktemp -d /tmp/joseph_final_git.XXXXXX)"
trap 'rm -rf "$work_dir"' EXIT

tar --sort=name --mtime='UTC 2026-09-24' --owner=0 --group=0 \
  --numeric-owner --exclude='final/git_bundle' -C "$joseph_dir" -cf - final \
  | zstd -T0 -9 -o "$work_dir/payload.tar.zst"
split -b 64m -d -a 4 "$work_dir/payload.tar.zst" \
  "$work_dir/payload.tar.zst.part-"
(cd "$work_dir" && sha256sum payload.tar.zst.part-* > SHA256SUMS)
cat "$work_dir"/payload.tar.zst.part-* | zstd -tq

BUNDLE_DIR="$bundle_dir" WORK_DIR="$work_dir" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

bundle = Path(os.environ["BUNDLE_DIR"])
work = Path(os.environ["WORK_DIR"])
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
parts = sorted(work.glob("payload.tar.zst.part-*"))
manifest = {
    "schema_version": "joseph_final_git_bundle.v1",
    "source_collection_sha256": digest(bundle.parent / "collection.json"),
    "archive_sha256": digest(work / "payload.tar.zst"),
    "parts": [{"name": p.name, "size": p.stat().st_size, "sha256": digest(p)}
              for p in parts],
    "restore_root": "outputs/paper/assay_transfer_harness/joseph",
}
(work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY

rm -f "$bundle_dir"/payload.tar.zst.part-*
cp "$work_dir"/payload.tar.zst.part-* "$work_dir/SHA256SUMS" \
  "$work_dir/manifest.json" "$bundle_dir/"
(cd "$bundle_dir" && sha256sum -c SHA256SUMS)
