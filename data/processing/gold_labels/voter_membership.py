"""Publish exact source-record membership for aggregated gold labels."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = "gold_voter_membership.v1"
INDEX_SCHEMA_VERSION = "gold_label_record_index.v1"
NO_REPORTED_CONDITION = "no_reported_external_condition"
SOURCE_ROW_UID_PATTERN = re.compile(r"sr_[0-9a-f]{32}")
AGGREGATE_STATUSES = {
    "published",
    "parent_label_tie",
    "parent_agreement_below_threshold",
    "group_gate_rejected",
}
PARENT_REJECTION_STATUSES = {
    "parent_record_label_tie": "parent_label_tie",
    "parent_condition_label_tie": "parent_label_tie",
    "parent_record_agreement_below_threshold": "parent_agreement_below_threshold",
    "parent_condition_agreement_below_threshold": "parent_agreement_below_threshold",
}
MEMBERSHIP_SCHEMA = pa.schema(
    [
        ("vote_context_id", pa.string(), False),
        ("task", pa.string(), False),
        ("source_row_uid", pa.string(), False),
        ("source_id", pa.string(), False),
        ("source_record_id", pa.string(), False),
        ("molecule_identity_key", pa.string(), False),
        ("parent_smiles", pa.string(), False),
        ("condition_group", pa.string(), False),
        ("condition_atoms", pa.list_(pa.string()), False),
        ("vote_label", pa.int8(), False),
        ("label_method", pa.string(), False),
        ("aggregate_status", pa.string(), False),
        ("aggregate_reason", pa.string(), False),
        ("aggregate_label", pa.int8()),
        ("benchmark_row_id", pa.string()),
        ("split", pa.string()),
    ]
)
INDEX_SCHEMA = pa.schema(
    [
        ("benchmark_row_id", pa.string(), False),
        ("split", pa.string(), False),
        ("molecule_identity_key", pa.string(), False),
        ("condition_group", pa.string(), False),
        ("source_row_uids", pa.list_(pa.string()), False),
        ("physical_voter_count", pa.int32(), False),
        ("negative_vote_count", pa.int32(), False),
        ("positive_vote_count", pa.int32(), False),
        ("voter_mean", pa.float64(), False),
    ]
)


def materialize_voter_membership(
    *,
    task_name: str,
    vote_groups: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    published_aggregates: Sequence[Mapping[str, Any]],
    rejected_parent_aggregates: Sequence[Mapping[str, Any]] = (),
    group_gate_rejected_aggregates: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Expand accepted source votes without using capped aggregate summaries."""
    aggregate_by_context: dict[tuple[str, str], dict[str, Any]] = {}
    for row in published_aggregates:
        _add_aggregate(aggregate_by_context, row, status="published")
    for row in rejected_parent_aggregates:
        reason = _text(row.get("drop_reason"))
        try:
            status = PARENT_REJECTION_STATUSES[reason]
        except KeyError as exc:
            raise ValueError(f"Unsupported parent aggregate rejection: {reason}") from exc
        _add_aggregate(aggregate_by_context, row, status=status)
    for row in group_gate_rejected_aggregates:
        _add_aggregate(aggregate_by_context, row, status="group_gate_rejected")

    vote_contexts = set(vote_groups)
    aggregate_contexts = set(aggregate_by_context)
    if vote_contexts != aggregate_contexts:
        missing = sorted(vote_contexts - aggregate_contexts)
        extra = sorted(aggregate_contexts - vote_contexts)
        raise ValueError(
            "Aggregate status coverage does not match vote groups: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    output: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    for (parent, condition_group), source_rows in sorted(vote_groups.items()):
        aggregate = aggregate_by_context[(parent, condition_group)]
        context_id = vote_context_id(task_name, parent, condition_group)
        for source_row in source_rows:
            source_uid = _required_text(source_row, "source_row_uid")
            if not SOURCE_ROW_UID_PATTERN.fullmatch(source_uid):
                raise ValueError(f"Invalid source_row_uid: {source_uid}")
            if source_row.get("molecule_identity_key") not in (None, "", parent):
                raise ValueError(f"Vote {source_uid} belongs to a different parent")
            if source_row.get("condition_group") not in (None, "", condition_group):
                raise ValueError(f"Vote {source_uid} belongs to a different condition")
            if source_uid in seen_uids:
                raise ValueError(
                    f"Physical source row belongs to multiple vote contexts: {source_uid}"
                )
            seen_uids.add(source_uid)
            label = int(source_row["Y"])
            if label not in {0, 1}:
                raise ValueError(f"Non-binary vote for {source_uid}: {label}")
            output.append(
                {
                    "vote_context_id": context_id,
                    "task": task_name,
                    "source_row_uid": source_uid,
                    "source_id": _required_text(source_row, "source_id"),
                    "source_record_id": _required_text(source_row, "source_record_id"),
                    "molecule_identity_key": parent,
                    "parent_smiles": _required_text(aggregate, "drug"),
                    "condition_group": condition_group,
                    "condition_atoms": [
                        str(value) for value in aggregate.get("condition_atoms", [])
                    ],
                    "vote_label": label,
                    "label_method": _required_text(source_row, "label_method"),
                    "aggregate_status": aggregate["aggregate_status"],
                    "aggregate_reason": aggregate["aggregate_reason"],
                    "aggregate_label": aggregate["aggregate_label"],
                    "benchmark_row_id": aggregate["benchmark_row_id"],
                    "split": aggregate["split"],
                }
            )
    return output


def write_voter_membership(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    stage1_sha256: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write deterministic membership Parquet and a compact provenance manifest."""
    if stage1_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", stage1_sha256
    ):
        raise ValueError("stage1_sha256 must be a lowercase SHA-256 hex digest")
    normalized = sorted(
        (dict(row) for row in rows),
        key=lambda row: (str(row["vote_context_id"]), str(row["source_row_uid"])),
    )
    _validate_membership_rows(normalized)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    table = pa.Table.from_pylist(
        normalized, schema=MEMBERSHIP_SCHEMA
    ).replace_schema_metadata({b"schema_version": SCHEMA_VERSION.encode("utf-8")})
    pq.write_table(table, temporary_path, compression="zstd")
    temporary_path.replace(output_path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "path": output_path.name,
        "sha256": _sha256_file(output_path),
        "stage1_sha256": stage1_sha256,
        "n_rows": len(normalized),
        "n_vote_contexts": len({row["vote_context_id"] for row in normalized}),
        "aggregate_status_counts": dict(
            sorted(Counter(row["aggregate_status"] for row in normalized).items())
        ),
        "provenance": dict(provenance or {}),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return manifest


def materialize_gold_label_record_index(
    membership_rows: Sequence[Mapping[str, Any]],
    published_aggregates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Index each published gold card by its exact physical voter UIDs."""
    cards = {str(row["benchmark_row_id"]): row for row in published_aggregates}
    if len(cards) != len(published_aggregates):
        raise ValueError("Published gold cards contain duplicate benchmark_row_id values")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in membership_rows:
        if row["aggregate_status"] != "published":
            continue
        benchmark_row_id = _required_text(row, "benchmark_row_id")
        grouped.setdefault(benchmark_row_id, []).append(row)
    if set(grouped) != set(cards):
        raise ValueError("Published voter membership does not cover gold cards exactly")

    output = []
    for benchmark_row_id, card in sorted(cards.items()):
        members = sorted(grouped[benchmark_row_id], key=lambda row: row["source_row_uid"])
        counts = Counter(int(row["vote_label"]) for row in members)
        expected_counts = {int(key): int(value) for key, value in card["label_counts"].items()}
        if counts != Counter(expected_counts):
            raise ValueError(f"Voter counts disagree with gold card {benchmark_row_id}")
        if len(members) != int(card["source_record_count"]):
            raise ValueError(f"Voter count disagrees with gold card {benchmark_row_id}")
        if any(
            row[field] != card[field]
            for row in members
            for field in ("molecule_identity_key", "condition_group", "split")
        ):
            raise ValueError(f"Voter identity disagrees with gold card {benchmark_row_id}")
        output.append(
            {
                "benchmark_row_id": benchmark_row_id,
                "split": str(card["split"]),
                "molecule_identity_key": str(card["molecule_identity_key"]),
                "condition_group": str(card["condition_group"]),
                "source_row_uids": [str(row["source_row_uid"]) for row in members],
                "physical_voter_count": len(members),
                "negative_vote_count": counts[0],
                "positive_vote_count": counts[1],
                "voter_mean": counts[1] / len(members),
            }
        )
    return output


def write_gold_label_record_index(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    membership_sha256: str,
) -> dict[str, Any]:
    """Write the compact card-to-UID index and its hash-pinned manifest."""
    output_path = Path(path)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    table = pa.Table.from_pylist(list(rows), schema=INDEX_SCHEMA).replace_schema_metadata(
        {b"schema_version": INDEX_SCHEMA_VERSION.encode("utf-8")}
    )
    pq.write_table(table, temporary_path, compression="zstd")
    temporary_path.replace(output_path)
    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "path": output_path.name,
        "sha256": _sha256_file(output_path),
        "voter_membership_sha256": membership_sha256,
        "n_gold_labels": len(rows),
        "n_physical_voters": sum(int(row["physical_voter_count"]) for row in rows),
        "edge_view": "voter_membership.parquet where aggregate_status == published",
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest


def vote_context_id(task_name: str, parent: str, condition_group: str) -> str:
    payload = f"{task_name}\0{parent}\0{condition_group}".encode("utf-8")
    return f"VOTE_{hashlib.sha256(payload).hexdigest()[:24].upper()}"


def _add_aggregate(
    output: dict[tuple[str, str], dict[str, Any]],
    row: Mapping[str, Any],
    *,
    status: str,
) -> None:
    parent = _required_text(row, "molecule_identity_key")
    condition_group = _text(row.get("condition_group")) or NO_REPORTED_CONDITION
    key = (parent, condition_group)
    if key in output:
        raise ValueError(f"Multiple terminal aggregate outcomes for {key}")
    reason = _text(row.get("drop_reason") or row.get("label_decision")) or status
    aggregate_label = int(row["Y"]) if status == "published" else None
    if aggregate_label is not None and aggregate_label not in {0, 1}:
        raise ValueError(f"Non-binary aggregate label for {key}: {aggregate_label}")
    benchmark_row_id = _nullable_text(row.get("benchmark_row_id"))
    split = _nullable_text(row.get("split"))
    if status != "published" and (benchmark_row_id is not None or split is not None):
        raise ValueError(f"Rejected aggregate {key} cannot have benchmark identity")
    if (benchmark_row_id is None) != (split is None):
        raise ValueError(f"benchmark_row_id and split must be set together for {key}")
    if split is not None and split not in {"train", "valid", "test"}:
        raise ValueError(f"Unsupported benchmark split for {key}: {split}")
    output[key] = {
        **dict(row),
        "aggregate_status": status,
        "aggregate_reason": reason,
        "aggregate_label": aggregate_label,
        "benchmark_row_id": benchmark_row_id,
        "split": split,
    }


def _validate_membership_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    seen_uids: set[str] = set()
    for row in rows:
        for field in MEMBERSHIP_SCHEMA.names[:13]:
            if row.get(field) in (None, ""):
                raise ValueError(f"Voter membership row lacks {field}")
        status = str(row["aggregate_status"])
        if status not in AGGREGATE_STATUSES:
            raise ValueError(f"Unsupported aggregate_status: {status}")
        source_uid = str(row["source_row_uid"])
        if source_uid in seen_uids:
            raise ValueError(f"Duplicate physical voter membership: {source_uid}")
        seen_uids.add(source_uid)


def _required_text(row: Mapping[str, Any], field: str) -> str:
    value = _text(row.get(field))
    if not value:
        raise ValueError(f"Voter membership source row lacks {field}")
    return value


def _nullable_text(value: Any) -> str | None:
    return _text(value) or None


def _text(value: Any) -> str:
    return str(value or "").strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
