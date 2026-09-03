"""Materialize the corpus-level BBB v7 measurement-semantics audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_measurement_semantics import (
    measurement_semantics_audit,
)


DEFAULT_ROOT = Path(
    "outputs/chembl_tool/tasks/bbb_martins/evidence_library/starling_normalized_v7"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        default=str(DEFAULT_ROOT / "02_canonicalized/records.parquet"),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_ROOT / "09_audits/measurement_semantics.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    requested_columns = (
        "source_id",
        "endpoint_name",
        "canonical_endpoint",
        "canonical_endpoint_name",
        "canonical_unit",
        "canonical_unit_text",
        "categorical_encoder_id",
        "canonical_measurement_scale_id",
        "normalization_validity_status",
        "canonicalization_status",
    )
    available = set(pq.read_schema(args.records).names)
    records = pd.read_parquet(
        args.records,
        columns=[column for column in requested_columns if column in available],
    ).to_dict(orient="records")
    audit = measurement_semantics_audit(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "inventory_count": audit["inventory_count"],
                "record_status_counts": audit["record_status_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
