"""Snapshot or verify the retrieval boundary around a Stage-03 rebuild."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from tools.chembl_tool.common.starling.build_normalized_evidence_library import (
    load_task_policy,
)
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.retrieval_boundary import (
    FROZEN_RETRIEVAL_FIELDS,
)


AUDIT_VERSION = "assay_transfer_retrieval_invariance.v5"


def retrieval_boundary_digest(
    task_id: str, records_path: str | Path
) -> dict[str, Any]:
    """Digest only fields that canonical assay-transfer values cannot change."""
    path = Path(records_path)
    policy = load_task_policy(task_id)
    contract = policy.record_contract
    if contract is None:
        raise ValueError(f"{task_id} has no v7 record contract")
    source_fields = sorted(
        {
            field
            for profile in contract.sources.values()
            for field in profile.source_visible_fields
        }
    )
    required_columns = [
        "cleaned_record_id",
        "group_id",
        "source_id",
        *source_fields,
    ]
    schema = set(pq.read_schema(path).names)
    missing = sorted(set(required_columns) - schema)
    if missing:
        raise ValueError(f"retrieval audit input lacks columns: {missing}")
    available_retrieval_fields = [
        field for field in FROZEN_RETRIEVAL_FIELDS if field in schema
    ]
    columns = [*required_columns, *available_retrieval_fields]
    row_digests: list[bytes] = []
    count = eligible = 0
    for batch in pq.ParquetFile(path).iter_batches(8192, columns=columns):
        for record in batch.to_pylist():
            payload = _retrieval_payload(record, contract)
            row_digests.append(
                hashlib.sha256(_canonical_json(payload).encode("utf-8")).digest()
            )
            count += 1
            if "retrieval_eligible" in schema:
                eligible += int(payload["retrieval_eligible"])
    digest = hashlib.sha256()
    for row_digest in sorted(row_digests):
        digest.update(row_digest)
    return {
        "audit_version": AUDIT_VERSION,
        "task_id": task_id,
        "record_contract_version": contract.version,
        "record_count": count,
        "retrieval_eligible_count": (
            eligible if "retrieval_eligible" in schema else None
        ),
        "retrieval_boundary_sha256": digest.hexdigest(),
        "available_retrieval_fields": available_retrieval_fields,
        "source_projection_fields": source_fields,
        "records_path": str(path),
        "records_file_sha256": file_sha256(path),
    }


def _retrieval_payload(record: Mapping[str, Any], contract: Any) -> dict[str, Any]:
    projection = contract.source_projection(record)
    return {
        # Cleaned IDs are the stable source-row identity at the retrieval
        # boundary; canonical-row identifiers are not retrieval fields.
        "cleaned_record_id": str(record["cleaned_record_id"]),
        "group_id": record.get("group_id"),
        "retrieval_fields": {
            field: record.get(field) for field in FROZEN_RETRIEVAL_FIELDS
        },
        "retrieval_eligible": bool(record.get("retrieval_eligible")),
        "source_projection": projection,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def compare_snapshot(snapshot: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    fields = (
        "audit_version",
        "task_id",
        "record_contract_version",
        "record_count",
        "retrieval_eligible_count",
        "retrieval_boundary_sha256",
        "available_retrieval_fields",
        "source_projection_fields",
    )
    changed = [field for field in fields if snapshot.get(field) != current.get(field)]
    if changed:
        raise ValueError(f"retrieval boundary changed after rebuild: {changed}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--compare", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    current = retrieval_boundary_digest(args.task, args.records)
    if args.compare:
        snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
        compare_snapshot(snapshot, current)
        print(_canonical_json({"status": "ok", **current}))
        return
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    args.snapshot.write_text(
        json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(_canonical_json({"status": "snapshot_written", **current}))


if __name__ == "__main__":
    main()
