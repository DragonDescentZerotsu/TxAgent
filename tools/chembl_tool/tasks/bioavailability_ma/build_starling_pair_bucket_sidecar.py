"""Build the Bioavailability_Ma source-aware pair-bucket sidecar.

The command reads normalized v6 ``records.parquet`` and never rewrites it.
It materializes bucket membership and audits only; it does not enumerate or
label molecular pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.normalization.audit import write_parquet
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.pair_buckets import (
    materialize_pair_buckets,
    read_pair_bucket_input,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_pair_buckets import (
    BIOAVAILABILITY_PAIR_BUCKET_VERSION,
    SOURCE_PAIR_FIELDS,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_schema import RECORD_CONTRACT
from tools.chembl_tool.tasks.bioavailability_ma.starling_policy import DEFAULT_OUT_DIR as DEFAULT_LIBRARY_ROOT


DEFAULT_NORMALIZED_DIR = Path(DEFAULT_LIBRARY_ROOT)
DEFAULT_OUT_DIR = DEFAULT_NORMALIZED_DIR / "04_pair_buckets"
PAIR_BUCKET_RECORDS_FILENAME = "pair_bucket_records.parquet"
PAIR_BUCKET_METADATA_FILENAME = "pair_bucket_metadata.json"
LEGACY_FILENAMES = (
    "endpoint_pair_registry.json",
    "pair_bucket_audit.json",
    "manifest.json",
)


def build_sidecar(
    *,
    records_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    records_path = Path(records_path)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    records, v7, pair_fields = read_pair_bucket_input(
        records_path,
        v7_source_fields={
            source: spec.additional_dimensions
            for source, spec in RECORD_CONTRACT.pair_buckets.items()
        },
        legacy_source_fields=SOURCE_PAIR_FIELDS,
    )
    sidecar_rows, metadata = materialize_pair_buckets(
        records,
        source_required_fields=pair_fields,
        contract_version=(
            RECORD_CONTRACT.version if v7 else BIOAVAILABILITY_PAIR_BUCKET_VERSION
        ),
    )
    if not all(metadata["validations"].values()):
        raise ValueError(
            f"pair-bucket identity/coverage audit failed: "
            f"{metadata['validations']}"
        )

    records_output = target / PAIR_BUCKET_RECORDS_FILENAME
    write_parquet(records_output, sidecar_rows)
    metadata.update(
        {
            "input": {
                "path": str(records_path),
                "sha256": file_sha256(records_path),
                "records": len(records),
            },
            "output": {
                "path": str(records_output),
                "sha256": file_sha256(records_output),
            },
            "scope": {
                "pair_enumeration": False,
                "pair_labels": False,
                "comparison_thresholds": False,
                "modeling_dataset": False,
            },
        }
    )
    _write_json(target / PAIR_BUCKET_METADATA_FILENAME, metadata)
    for filename in LEGACY_FILENAMES:
        (target / filename).unlink(missing_ok=True)
    return metadata


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    metadata = build_sidecar(
        records_path=args.records,
        out_dir=args.out_dir,
    )
    print(
        "[build_starling_pair_bucket_sidecar] "
        f"records={metadata['stats']['sidecar_records']:,} "
        f"eligible={metadata['stats']['eligible_records']:,} "
        f"buckets={metadata['stats']['buckets']:,} out={args.out_dir}",
        flush=True,
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", default=str(DEFAULT_NORMALIZED_DIR / "03_records/records.parquet")
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args(argv)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
