"""Build BBB Martins source-aware pair-bucket membership from Stage 03."""

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
from tools.chembl_tool.tasks.bbb_martins.starling_pair_buckets import (
    BBB_MARTINS_PAIR_BUCKET_VERSION,
    BBB_MARTINS_V7_PAIR_BUCKET_VERSION,
    ENDPOINT_FIELD_BY_SOURCE,
    SOURCE_PAIR_FIELDS,
)
from tools.chembl_tool.tasks.bbb_martins.starling_schema import RECORD_CONTRACT
from tools.chembl_tool.tasks.bbb_martins.starling_policy import DEFAULT_OUT_DIR, POLICY


DEFAULT_NORMALIZED_DIR = Path(DEFAULT_OUT_DIR)
DEFAULT_PAIR_BUCKET_DIR = DEFAULT_NORMALIZED_DIR / "03_pair_buckets"
PAIR_BUCKET_RECORDS_FILENAME = "pair_bucket_records.parquet"
PAIR_BUCKET_METADATA_FILENAME = "pair_bucket_metadata.json"


def build_sidecar(
    *,
    records_path: str | Path,
    out_dir: str | Path,
    assay_transfer_record_ineligibility: dict[str, str] | None = None,
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
        v7_endpoint_field_by_source=ENDPOINT_FIELD_BY_SOURCE,
    )
    rows, metadata = materialize_pair_buckets(
        records,
        source_required_fields=pair_fields,
        contract_version=(
            BBB_MARTINS_V7_PAIR_BUCKET_VERSION
            if v7
            else BBB_MARTINS_PAIR_BUCKET_VERSION
        ),
        endpoint_field_by_source=ENDPOINT_FIELD_BY_SOURCE if v7 else None,
        assay_transfer_record_ineligibility=assay_transfer_record_ineligibility,
        canonical_record_contract=v7,
    )
    if not all(metadata["validations"].values()):
        raise ValueError(f"pair-bucket audit failed: {metadata['validations']}")
    output = target / PAIR_BUCKET_RECORDS_FILENAME
    write_parquet(output, rows)
    metadata.update(
        {
            "input": {
                "path": str(records_path),
                "sha256": file_sha256(records_path),
                "records": len(records),
            },
            "output": {"path": str(output), "sha256": file_sha256(output)},
            "scope": {
                "pair_enumeration": False,
                "pair_labels": False,
                "comparison_thresholds": False,
                "modeling_dataset": False,
            },
        }
    )
    _write_json(target / PAIR_BUCKET_METADATA_FILENAME, metadata)
    return metadata


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", default=str(DEFAULT_NORMALIZED_DIR / "02_canonicalized/records.parquet")
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_PAIR_BUCKET_DIR))
    args = parser.parse_args(argv)
    metadata = build_sidecar(records_path=args.records, out_dir=args.out_dir)
    print(json.dumps(metadata["stats"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
