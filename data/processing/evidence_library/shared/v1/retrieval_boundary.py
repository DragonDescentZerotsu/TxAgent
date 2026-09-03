"""Freeze retrieval identity while rebuilding assay-transfer canonical fields."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


FROZEN_RETRIEVAL_FIELDS = (
    "source_smiles",
    "canonical_smiles",
    "structure_status",
    "molecule_id",
    "duplicate_group_id",
    "duplicate_group_size",
    "retrieval_eligible",
    "organization_status",
)
FROZEN_STRUCTURE_FIELDS = FROZEN_RETRIEVAL_FIELDS[:4]


def freeze_retrieval_boundary(
    records: Sequence[Mapping[str, Any]], frozen_records_path: str | Path
) -> list[dict[str, Any]]:
    """Copy retrieval-owned identity/status fields by stable cleaned row ID."""
    return _freeze_fields(records, frozen_records_path, FROZEN_RETRIEVAL_FIELDS)


def freeze_normalized_retrieval_identity(
    records: Sequence[Mapping[str, Any]], frozen_records_path: str | Path
) -> list[dict[str, Any]]:
    """Restore structure identity before retrieval deduplication."""
    return _freeze_fields(records, frozen_records_path, FROZEN_STRUCTURE_FIELDS)


def _freeze_fields(
    records: Sequence[Mapping[str, Any]],
    frozen_records_path: str | Path,
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    columns = ("cleaned_record_id", *fields)
    path = Path(frozen_records_path)
    missing = set(columns) - set(pq.read_schema(path).names)
    if missing:
        raise ValueError(f"frozen retrieval records lack columns: {sorted(missing)}")
    frozen: dict[str, dict[str, Any]] = {}
    for batch in pq.ParquetFile(path).iter_batches(8192, columns=list(columns)):
        for row in batch.to_pylist():
            key = str(row["cleaned_record_id"])
            if key in frozen:
                raise ValueError(f"duplicate frozen cleaned_record_id: {key}")
            frozen[key] = row
    current_ids = [str(row.get("cleaned_record_id") or "") for row in records]
    if len(set(current_ids)) != len(current_ids):
        raise ValueError("current cleaned_record_id values are not unique")
    if set(current_ids) != set(frozen):
        current_only = sorted(set(current_ids) - set(frozen))
        frozen_only = sorted(set(frozen) - set(current_ids))
        raise ValueError(
            "frozen/current retrieval row identities differ: "
            f"current_only={len(current_only)} {current_only[:5]}, "
            f"frozen_only={len(frozen_only)} {frozen_only[:5]}"
        )
    output: list[dict[str, Any]] = []
    for row, key in zip(records, current_ids, strict=True):
        updated = dict(row)
        updated.update({field: frozen[key][field] for field in fields})
        output.append(updated)
    return output


__all__ = [
    "FROZEN_RETRIEVAL_FIELDS",
    "FROZEN_STRUCTURE_FIELDS",
    "freeze_normalized_retrieval_identity",
    "freeze_retrieval_boundary",
]
