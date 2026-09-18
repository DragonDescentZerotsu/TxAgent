"""Authoritative exact-row deduplication for V10 Stage 1."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.shared.v2.normalization.source_value_cleaning import (
    SourceValueCleaningResult,
)


POLICY_PATH = Path(__file__).with_name("stage1_exact_deduplication.v3.json")
POLICY_SHA256 = "3126102794c9e6174abbaec09cd9cbd4f60faf37b00f7fd009184765bc5dea99"
VERSION = "stage1_exact_deduplication.v3"
PROTECTION_VERSION = "gold_v1_voter_protection.v1"
BASE_FIELDS = (
    "canonical_smiles",
    "pmid",
    "canonical_measurement_text",
    "canonical_unit_text",
)


@lru_cache(maxsize=1)
def load_policy() -> dict[str, Any]:
    if file_sha256(POLICY_PATH) != POLICY_SHA256:
        raise ValueError("Stage-1 exact-deduplication policy hash mismatch")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if (
        policy.get("version") != VERSION
        or policy.get("authoritative_stage1_deduplication") is not True
        or tuple(policy.get("base_fields") or ()) != BASE_FIELDS
    ):
        raise ValueError("Invalid Stage-1 exact-deduplication policy contract")
    _validate_source_policies(policy)
    return policy


@lru_cache(maxsize=None)
def load_task_policy(path: str, expected_sha256: str) -> dict[str, Any]:
    """Load a task extension without changing the frozen shared v3 policy."""
    policy_path = Path(path)
    if file_sha256(policy_path) != expected_sha256:
        raise ValueError("Stage-1 task deduplication policy hash mismatch")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if (
        policy.get("authoritative_stage1_deduplication") is not True
        or policy.get("base_policy_sha256") != POLICY_SHA256
        or tuple(policy.get("base_fields") or ()) != BASE_FIELDS
        or not policy.get("version")
    ):
        raise ValueError("Invalid Stage-1 task deduplication policy contract")
    _validate_source_policies(policy)
    return policy


def _validate_source_policies(policy: Mapping[str, Any]) -> None:
    for task in policy.get("tasks", {}).values():
        for source in task.get("sources", {}).values():
            scientific = source.get("scientific_fields", [])
            ignored = [
                field
                for fields in source.get("ignored_source_fields", {}).values()
                for field in fields
            ]
            if (
                len(scientific) != len(set(scientific))
                or len(ignored) != len(set(ignored))
                or set(scientific) & set(ignored)
            ):
                raise ValueError("Invalid configured Stage-1 scientific fields")


def deduplicate_stage1_exact_records(
    records: list[dict[str, Any]],
    _args: Any = None,
    *,
    task_id: str,
    source_columns: Mapping[str, tuple[str, ...]],
    task_policy_path: str | Path | None = None,
    task_policy_sha256: str | None = None,
) -> SourceValueCleaningResult:
    """Deduplicate exact rows, preferring frozen-gold voters and then UID."""
    base_policy = load_policy()
    policy = (
        load_task_policy(str(task_policy_path), str(task_policy_sha256 or ""))
        if task_policy_path is not None
        else base_policy
    )
    task = policy.get("tasks", {}).get(task_id)
    if task is None:
        raise ValueError(f"No Stage-1 exact-deduplication policy for {task_id}")
    source_fields = {
        source_id: tuple(spec["scientific_fields"])
        for source_id, spec in task["sources"].items()
    }
    common_fields = set(BASE_FIELDS) | {
        field
        for fields in base_policy["common_record_fields"].values()
        for field in fields
    }
    if set(source_columns) != set(source_fields):
        raise ValueError("Stage-1 source contract and deduplication policy differ")
    for source_id, spec in task["sources"].items():
        ignored = {
            field
            for fields in spec["ignored_source_fields"].values()
            for field in fields
        }
        classified = (
            set(source_fields[source_id])
            | ignored
            | (set(source_columns[source_id]) & common_fields)
        )
        if set(source_columns[source_id]) != classified:
            raise ValueError(
                f"Unclassified Stage-1 source fields for {source_id}"
            )
    uids = [str(record.get("source_row_uid") or "") for record in records]
    if any(not uid for uid in uids) or len(uids) != len(set(uids)):
        raise ValueError("Stage-1 source_row_uid values must be nonempty and unique")
    unknown_sources = {
        str(record.get("source_id") or "") for record in records
    } - set(source_fields)
    if unknown_sources:
        raise ValueError(f"Unconfigured Stage-1 sources: {sorted(unknown_sources)}")
    protected, preferred_path = load_voter_protection(_args, task_id)
    preferred_uids = set(protected)
    require_shared_cross_source_field = (
        policy.get("empty_cross_source_intersection") == "never_duplicate"
    )
    missing_preferred = preferred_uids - set(uids)
    if missing_preferred:
        raise ValueError(
            "Preferred gold-v1 voter UIDs are absent from pre-dedup Stage 1: "
            f"{sorted(missing_preferred)[:20]}"
        )
    preferred_uids.intersection_update(uids)
    for record in records:
        source_id = str(record.get("source_id") or "")
        unclassified = set(record) - common_fields - set(source_columns[source_id])
        if unclassified:
            raise ValueError(
                f"Unclassified Stage-1 record fields for {source_id}: "
                f"{sorted(unclassified)}"
            )

    candidates: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if any(not _present(record.get(field)) for field in BASE_FIELDS):
            candidates[("incomplete_base", uids[index])].append(index)
        else:
            candidates[
                tuple(_exact(record, field) for field in BASE_FIELDS)
            ].append(index)

    discarded: dict[int, int] = {}
    for indices in candidates.values():
        clusters: list[list[int]] = []
        for index in sorted(indices, key=lambda item: uids[item]):
            cluster = next(
                (
                    members
                    for members in clusters
                    if all(
                        _scientific_match(
                            records[index],
                            records[member],
                            source_fields,
                            require_shared_cross_source_field=require_shared_cross_source_field,
                        )
                        for member in members
                    )
                ),
                None,
            )
            if cluster is None:
                clusters.append([index])
            else:
                cluster.append(index)
        for cluster in clusters:
            protected_members = [
                index for index in cluster if uids[index] in preferred_uids
            ]
            retained = min(protected_members or cluster, key=lambda item: uids[item])
            discarded.update(
                {
                    index: retained
                    for index in cluster
                    if index != retained and index not in protected_members
                }
            )

    audit_rows = []
    scopes: Counter[str] = Counter()
    for dropped_index, retained_index in sorted(
        discarded.items(), key=lambda item: uids[item[0]]
    ):
        dropped, retained = records[dropped_index], records[retained_index]
        scope = (
            "within_source"
            if dropped.get("source_id") == retained.get("source_id")
            else "cross_source"
        )
        scopes[scope] += 1
        comparison_fields = _comparison_fields(dropped, retained, source_fields)
        audit_rows.append(
            {
                "cleaning_version": policy["version"],
                "cleaned_record_id": str(dropped.get("cleaned_record_id") or ""),
                "source_row_uid": uids[dropped_index],
                "source_id": str(dropped.get("source_id") or ""),
                "source_sha256": str(dropped.get("source_sha256") or ""),
                "source_row_number": int(dropped.get("source_row_number") or 0),
                "source_record_id": str(dropped.get("source_record_id") or ""),
                "field": "record",
                "before": "retained",
                "after": "dropped",
                "rule_id": policy["version"],
                "review_status": "deterministic",
                "evidence_id": str(dropped.get("pmid") or ""),
                "canonical_smiles": str(dropped.get("canonical_smiles") or ""),
                "canonical_measurement_text": dropped.get(
                    "canonical_measurement_text"
                ),
                "canonical_unit_text": dropped.get("canonical_unit_text"),
                "rationale": (
                    "Exact canonical base and configured source scientific fields; "
                    "representative prefers a frozen gold-v1 physical voter, then minimum UID."
                ),
                "duplicate_scope": scope,
                "retained_source_row_uid": uids[retained_index],
                "retained_source_id": str(retained.get("source_id") or ""),
                "comparison_fields": list(comparison_fields),
            }
        )

    retained_records = [
        record for index, record in enumerate(records) if index not in discarded
    ]
    retained_uids = sorted(
        str(record["source_row_uid"]) for record in retained_records
    )
    lineage = sorted(
        (uids[dropped], uids[retained], audit_rows[index]["duplicate_scope"])
        for index, (dropped, retained) in enumerate(
            sorted(discarded.items(), key=lambda item: uids[item[0]])
        )
    )
    manifest = {
        "version": policy["version"],
        "policy_path": str(task_policy_path or POLICY_PATH),
        "policy_sha256": task_policy_sha256 or POLICY_SHA256,
        "base_policy_path": str(POLICY_PATH),
        "base_policy_sha256": POLICY_SHA256,
        "authoritative_stage1_deduplication": True,
        "deduplication_complete": True,
        "base_fields": list(BASE_FIELDS),
        "source_scientific_fields": {
            source_id: list(fields)
            for source_id, fields in sorted(source_fields.items())
        },
        "comparison": {
            **base_policy["comparison"],
            "empty_cross_source_intersection": policy.get(
                "empty_cross_source_intersection", "base_fields_only"
            ),
        },
        "preferred_voter_uids": {
            "path": str(preferred_path) if preferred_path else "",
            "sha256": file_sha256(preferred_path) if preferred_path else "",
            "count": len(protected),
            "present_in_input": len(preferred_uids),
            "absent_from_authoritative_universe": len(missing_preferred),
        },
        "input_records": len(records),
        "output_records": len(retained_records),
        "duplicates_removed": len(discarded),
        "duplicate_scope_counts": dict(sorted(scopes.items())),
        "retained_source_row_uid_count": len(retained_uids),
        "retained_source_row_uid_sha256": _rows_digest((uid,) for uid in retained_uids),
        "duplicate_audit": {
            "cleaning_version": policy["version"],
            "row_selector": {"field": "record", "after": "dropped"},
            "count": len(lineage),
            "discarded_source_row_uid_sha256": _rows_digest(
                (discarded_uid,) for discarded_uid, _, _ in lineage
            ),
            "lineage_sha256": _rows_digest(lineage),
        },
        "validations": {
            "input_equals_retained_plus_duplicates": (
                len(records) == len(retained_records) + len(discarded)
            ),
            "retained_source_row_uids_unique": len(retained_uids)
            == len(set(retained_uids)),
            "all_protected_voters_retained": preferred_uids <= set(retained_uids),
            "nonprotected_representatives_follow_uid_order": all(
                uids[retained] < uids[dropped]
                or uids[retained] in preferred_uids
                for dropped, retained in discarded.items()
            ),
        },
    }
    if not all(manifest["validations"].values()):
        raise ValueError(f"Stage-1 exact deduplication failed: {manifest['validations']}")
    return SourceValueCleaningResult(
        records=retained_records,
        audit_rows=audit_rows,
        manifest=manifest,
        input_paths=(
            POLICY_PATH,
            *((Path(task_policy_path),) if task_policy_path else ()),
            *((preferred_path,) if preferred_path else ()),
        ),
    )


def _load_preferred_uids(args: Any, task_id: str) -> tuple[set[str], Path | None]:
    raw_path = str(getattr(args, "stage1_preferred_voter_uids", "") or "").strip()
    if not raw_path:
        return set(), None
    path = Path(raw_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != "gold_v1_physical_voter_lineage.v1":
        raise ValueError("Unsupported preferred-voter lineage version")
    if payload.get("task") != task_id:
        raise ValueError(
            f"Preferred-voter task mismatch: {payload.get('task')} != {task_id}"
        )
    values = [str(value) for value in payload.get("source_row_uids") or ()]
    if any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError("Preferred-voter UIDs must be nonempty and unique")
    if payload.get("source_row_uid_sha256") != _rows_digest(
        (value,) for value in sorted(values)
    ):
        raise ValueError("Preferred-voter UID digest mismatch")
    return set(values), path


def load_voter_protection(
    args: Any, task_id: str
) -> tuple[dict[str, dict[str, Any]], Path | None]:
    raw_path = str(
        getattr(args, "stage1_protected_voter_contract", "") or ""
    ).strip()
    if not raw_path:
        uids, path = _load_preferred_uids(args, task_id)
        return {uid: {} for uid in uids}, path
    path = Path(raw_path)
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    if metadata.get(b"schema_version", b"").decode() != PROTECTION_VERSION:
        raise ValueError("Unsupported protected-voter contract version")
    required = {"task", "source_row_uid"}
    if missing := required - set(table.column_names):
        raise ValueError(f"Protected-voter contract lacks fields: {sorted(missing)}")
    rows = table.to_pylist()
    if any(str(row["task"]) != task_id for row in rows):
        raise ValueError(f"Protected-voter task mismatch for {task_id}")
    output = {str(row["source_row_uid"]): row for row in rows}
    if len(output) != len(rows) or any(not uid for uid in output):
        raise ValueError("Protected-voter UIDs must be nonempty and unique")
    return output, path


def _comparison_fields(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    source_fields: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    left_fields = source_fields[str(left.get("source_id") or "")]
    right_fields = source_fields[str(right.get("source_id") or "")]
    return (
        left_fields
        if left.get("source_id") == right.get("source_id")
        else tuple(sorted(set(left_fields) & set(right_fields)))
    )


def _scientific_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    source_fields: Mapping[str, tuple[str, ...]],
    *,
    require_shared_cross_source_field: bool = False,
) -> bool:
    fields = _comparison_fields(left, right, source_fields)
    if (
        require_shared_cross_source_field
        and left.get("source_id") != right.get("source_id")
        and not fields
    ):
        return False
    return all(
        _exact(left, field) == _exact(right, field)
        for field in fields
    )


def _exact(record: Mapping[str, Any], field: str) -> str:
    if field not in record:
        return '["absent"]'
    return json.dumps(
        ["present", record[field]],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _present(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _rows_digest(rows: Any) -> str:
    payload = "".join(
        "\t".join(str(value) for value in row) + "\n" for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "POLICY_PATH",
    "POLICY_SHA256",
    "PROTECTION_VERSION",
    "deduplicate_stage1_exact_records",
    "load_voter_protection",
    "load_policy",
    "load_task_policy",
]
