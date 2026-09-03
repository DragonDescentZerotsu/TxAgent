"""Deferred v6.5 reconciliation outside the pair-bucket sidecar contract."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from data.processing.evidence_library.shared.v1.normalization.audit import write_parquet


RECONCILIATION_VERSION = "bioavailability_v65_reconciliation.deferred_v3"


def reconcile_v65_records(
    records: Sequence[Mapping[str, Any]],
    v65_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Account for legacy rows without asserting semantic or scalar comparability."""
    del records
    audit_rows = [
        {
            "reconciliation_version": RECONCILIATION_VERSION,
            "v65_child_id": old.get("child_id"),
            "v65_source_id": old.get("source_id"),
            "canonical_smiles": old.get("canonical_smiles"),
            "v65_endpoint_family": old.get("endpoint_family"),
            "v65_endpoint_subtype": old.get("endpoint_subtype"),
            "v65_metric_type": old.get("metric_type"),
            "v65_scalar_value": old.get("scalar_value"),
            "v65_unit_basis": old.get("unit_basis"),
            "reconciliation_status": "not_performed_by_pair_bucket_contract",
        }
        for old in v65_records
    ]
    return audit_rows, {
        "reconciliation_version": RECONCILIATION_VERSION,
        "status": "not_performed_by_pair_bucket_contract",
        "matching_performed": False,
        "n_v65_records": len(audit_rows),
        "n_value_matched": 0,
    }


def write_v65_reconciliation(
    *,
    records_path: str | Path,
    v65_records_path: str | Path,
    output_parquet: str | Path,
    output_summary: str | Path,
) -> dict[str, Any]:
    records = pd.read_parquet(records_path).to_dict(orient="records")
    v65_records = pd.read_parquet(v65_records_path).to_dict(orient="records")
    rows, summary = reconcile_v65_records(records, v65_records)
    write_parquet(output_parquet, rows)
    Path(output_summary).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "RECONCILIATION_VERSION",
    "reconcile_v65_records",
    "write_v65_reconciliation",
]
