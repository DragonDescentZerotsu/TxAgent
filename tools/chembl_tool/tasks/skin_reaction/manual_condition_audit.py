"""Reusable record-level audit for manually adjudicated Skin condition groups."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.json_utils import (
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


@dataclass(frozen=True)
class ManualConditionAuditConfig:
    condition_atom: str
    pure_condition_group: str
    audit_version: str
    manual_review_path: Path
    audit_path: Path
    summary_path: Path
    review_root: Path
    frozen_root: Path
    conditioned_root: Path | None = None
    agreement_threshold: float = 0.60


def _manual_decisions(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    decisions: dict[str, dict[str, str]] = {}
    eligible = list(manifest.get("eligible_records", []))
    eligible.extend(
        {"source_record_id": record_id, "manual_reason": "Passed strict manual review."}
        for record_id in manifest.get("eligible_record_ids", [])
    )
    for row in eligible:
        decisions[row["source_record_id"]] = {
            "clean_status": "eligible",
            "clean_reason": "strict_manual_condition_eligibility",
            "manual_note": row["manual_reason"],
        }
    for reason, record_ids in manifest["excluded_records_by_reason"].items():
        for record_id in record_ids:
            if record_id in decisions:
                raise ValueError(f"Duplicate manual condition decision: {record_id}")
            decisions[record_id] = {
                "clean_status": "excluded",
                "clean_reason": reason,
                "manual_note": "",
            }
    return decisions, manifest


def _split_map(config: ManualConditionAuditConfig) -> dict[str, str]:
    output: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        for row in read_jsonl(config.frozen_root / f"{split}_molecule_labels.jsonl"):
            output[str(row["molecule_identity_key"])] = split
    if config.conditioned_root is None:
        return output
    for split in ("train", "valid", "test"):
        path = config.conditioned_root / f"{split}_molecule_condition_labels.jsonl"
        for row in read_jsonl(path):
            if row.get("condition_group") != config.pure_condition_group:
                continue
            key = str(row["molecule_identity_key"])
            prior = output.get(key)
            if prior is not None and prior != split:
                raise ValueError(f"Parent split conflicts with frozen assignment: {key}")
            output[key] = split
    return output


def audit_condition(config: ManualConditionAuditConfig) -> dict[str, Any]:
    decisions, manifest = _manual_decisions(config.manual_review_path)
    for filename, expected_sha256 in manifest["source_artifacts"].items():
        actual = sha256_file(config.review_root / filename)
        if actual != expected_sha256:
            raise ValueError(
                f"Source hash mismatch for {filename}: expected {expected_sha256}, got {actual}"
            )

    queue = read_jsonl(config.review_root / "review_queue.jsonl")
    verdicts = {
        row["source_record_id"]: row
        for row in read_jsonl(config.review_root / "review_verdicts.jsonl")
    }
    condition_rows = [
        row
        for row in queue
        if config.condition_atom in row.get("proposed_condition_atoms", [])
    ]
    pure_ids = {
        str(row["source_record_id"])
        for row in condition_rows
        if row["proposed_condition_group"] == config.pure_condition_group
    }
    if set(decisions) != pure_ids:
        raise ValueError(
            "Manual decisions must exactly cover the pure-signature records; "
            f"missing={sorted(pure_ids - set(decisions))}, "
            f"extra={sorted(set(decisions) - pure_ids)}"
        )

    split_by_parent = _split_map(config)
    audit_rows = []
    for row in sorted(condition_rows, key=lambda item: str(item["source_record_id"])):
        record_id = str(row["source_record_id"])
        exact_pure_signature = (
            row["proposed_condition_group"] == config.pure_condition_group
        )
        clean = decisions[record_id] if exact_pure_signature else {
            "clean_status": "excluded",
            "clean_reason": "exact_composite_signature_not_collapsed",
            "manual_note": "Composite condition remains a separate exact signature.",
        }
        model_review = verdicts.get(record_id)
        if model_review is None:
            raise ValueError(f"Missing terminal review verdict for {record_id}")
        if model_review.get("source_payload_sha256") != row["source_payload_sha256"]:
            raise ValueError(f"Review payload hash mismatch for {record_id}")
        audit_rows.append(
            {
                "source_record_id": record_id,
                "source_payload_sha256": row["source_payload_sha256"],
                "parent_key": row["molecule_identity_key"],
                "drug": row["drug"],
                "bemis_murcko_scaffold": row["bemis_murcko_scaffold"],
                "split": split_by_parent.get(row["molecule_identity_key"]),
                "Y": int(row["Y"]),
                "proposed_condition_group": row["proposed_condition_group"],
                "exact_pure_signature": exact_pure_signature,
                "condition_text": row["condition_text"],
                "support_text": row["support_text"],
                "model_review_status": model_review.get("review_status"),
                "model_review_reason": model_review.get("review_reason"),
                **clean,
            }
        )

    eligible = [row for row in audit_rows if row["clean_status"] == "eligible"]
    rows_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        rows_by_parent[row["parent_key"]].append(row)
    parent_labels = []
    vote_rejected = []
    for parent_key, source_rows in sorted(rows_by_parent.items()):
        counts = Counter(row["Y"] for row in source_rows)
        winning_label, winning_count = counts.most_common(1)[0]
        agreement = winning_count / len(source_rows)
        parent = {
            "parent_key": parent_key,
            "Y": winning_label,
            "agreement": agreement,
            "record_count": len(source_rows),
            "source_record_ids": [row["source_record_id"] for row in source_rows],
            "scaffold": source_rows[0]["bemis_murcko_scaffold"],
            "split": source_rows[0]["split"],
        }
        if agreement >= config.agreement_threshold:
            parent_labels.append(parent)
        else:
            vote_rejected.append(parent)

    split_counts = Counter(row["split"] or "unassigned" for row in parent_labels)
    scaffold_count = len({row["scaffold"] for row in parent_labels})
    gates = {
        "at_least_3_parents": len(parent_labels) >= 3,
        "at_least_3_distinct_scaffolds": scaffold_count >= 3,
        "train_valid_test_each_nonempty": all(
            split_counts[split] >= 1 for split in ("train", "valid", "test")
        ),
        "all_parents_assigned": split_counts["unassigned"] == 0,
    }
    summary = {
        "version": config.audit_version,
        "condition_group": config.pure_condition_group,
        "exact_signature_policy": "Composite signatures are excluded, not collapsed.",
        "agreement_threshold": config.agreement_threshold,
        "record_counts": {
            "all_condition_atom_records": len(audit_rows),
            "pure_signature_records": len(pure_ids),
            "composite_signature_records": len(audit_rows) - len(pure_ids),
            "strict_eligible_records": len(eligible),
            "strict_excluded_records": len(audit_rows) - len(eligible),
        },
        "strict_exclusion_reason_counts": dict(
            sorted(
                Counter(
                    row["clean_reason"]
                    for row in audit_rows
                    if row["clean_status"] == "excluded"
                ).items()
            )
        ),
        "parent_label_counts": {
            "total": len(parent_labels),
            "Y=0": sum(row["Y"] == 0 for row in parent_labels),
            "Y=1": sum(row["Y"] == 1 for row in parent_labels),
            "vote_rejected": len(vote_rejected),
        },
        "split_parent_counts": {
            split: split_counts[split]
            for split in ("train", "valid", "test", "unassigned")
        },
        "distinct_scaffold_count": scaffold_count,
        "parent_labels": parent_labels,
        "vote_rejected_parents": vote_rejected,
        "promotion_gates": gates,
        "formal_benchmark_decision": (
            "promote_group" if all(gates.values()) else "reject_group_from_formal_benchmark"
        ),
    }
    write_jsonl_atomic(config.audit_path, audit_rows)
    write_json_atomic(config.summary_path, summary)
    return summary
