# Starling raw sources

## Active data ownership

Active data lives with its semantic owner: gold-bound data under
`data/gold_labels/<Task>/<version>/`, evidence data under its task/release, and
shared reusable caches under `data/caches/`. `data/artifacts/` is audit-only and
must not be a required build or runtime input; complete retired products belong
under `data/legacy/`. Do not add compatibility symlinks.

The evidence-library pipeline owns scientific level assignment. Preserved
gold-version mappings live under `data/gold_labels/<Task>/level_mappings/<version>/`.
BBB and Bioavailability runtime consumers use the active release-owned
`data/evidence_libraries/<task>/<release>/level_mapping/`; Ames, DILI,
Carcinogens, and Skin keep their gold-owned mappings until reviewed replacements.
Voter membership may validate L1 coverage but must never derive or rewrite levels.
Corrections and publication belong to the evidence-library pipeline and must use
reviewed UID decisions with pinned input hashes.

Every committed Parquet under this directory already contains its permanent
`source_row_uid`. Never regenerate or renumber that column. The partitioned
`source_row_uid_ledger/` is the global identity authority.

Git tracks every raw file at or below 100,000,000 bytes. Seven larger
authoritative Parquets are ignored; their lossless, sub-100 MB Parquet parts and
manifests are committed under each task's `compressed/` directory.

## Restore ignored authoritative Parquets

Run this from the repository root with PyArrow 24.0.0. It restores only missing
files and verifies every input part and reconstructed file before publishing it.
The `combine_chunks()` and 25,000-row write batches are required for byte-exact
reconstruction, not merely logical table equality.

```bash
python - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

assert pa.__version__ == "24.0.0", pa.__version__
root = Path("data/raw/starling")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for manifest_path in sorted(root.glob("*/compressed/manifest.json")):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for source in manifest["sources"]:
        destination = root / source["authoritative_path"]
        if destination.exists():
            continue
        parts = [manifest_path.parent / part["name"] for part in source["parts"]]
        for path, expected in zip(parts, source["parts"], strict=True):
            assert sha256(path) == expected["sha256"], path
        table = pa.concat_tables([pq.read_table(path) for path in parts]).combine_chunks()
        assert table.num_rows == source["rows"]
        assert "source_row_uid" in table.column_names
        assert table["source_row_uid"].null_count == 0

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".parquet.tmp")
        writer = pq.ParquetWriter(
            temporary,
            table.schema,
            compression="zstd",
            compression_level=3,
        )
        try:
            for batch in table.to_batches(max_chunksize=25_000):
                writer.write_batch(batch)
        finally:
            writer.close()
        assert sha256(temporary) == source["authoritative_sha256"], destination
        os.replace(temporary, destination)
        print(f"restored {destination}")
PY

python -m data.processing.starling_source_row_identity verify
python -m data.processing.starling_source_row_identity verify-compressed
```

Do not run `bootstrap` after cloning: the committed ledger already owns the UIDs.
Use `ingest` only when intentionally adding or retiring authoritative source rows.
