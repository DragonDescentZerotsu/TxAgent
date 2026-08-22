"""Review-queue and terminal-verdict contract for conditioned gold records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic
from tools.chembl_tool.common.starling.benchmark_dataset import sha256_file
from tools.chembl_tool.common.molecule_identity import (
    bemis_murcko_scaffold,
    normalize_molecule_identity,
)


REVIEW_CONTRACT = "conditioned_gold_record_review.v1"


def attach_parent_identity(candidate: Mapping[str, Any], smiles: object) -> dict[str, Any]:
    row = dict(candidate)
    text = str(smiles or "").strip()
    identity = normalize_molecule_identity(text).to_dict() if text else {}
    key = identity.get("parent_inchi_key") or identity.get("parent_smiles")
    row.update(
        {
            "drug": identity.get("parent_smiles", ""),
            "molecule_identity_key": key or "",
            "molecule_identity": identity,
            "bemis_murcko_scaffold": (
                bemis_murcko_scaffold(identity["parent_smiles"])
                if identity.get("status") == "ok" and identity.get("parent_smiles")
                else ""
            ),
        }
    )
    return row


def write_review_queue(
    *,
    rows: Sequence[Mapping[str, Any]],
    queue_path: Path,
    manifest_path: Path,
    task_name: str,
    source_artifacts: Sequence[Path],
    proposal_version: str,
    proposal_audit_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: row["source_record_id"])
    ids = [str(row["source_record_id"]) for row in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("Review queue contains duplicate source_record_id values")
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(queue_path, ordered)
    proposal_audit_path = queue_path.with_name("proposal_audit.jsonl")
    if proposal_audit_rows:
        write_jsonl_atomic(
            proposal_audit_path,
            sorted(
                (dict(row) for row in proposal_audit_rows),
                key=lambda row: str(row["source_record_id"]),
            ),
        )
    manifest = {
        "review_contract": REVIEW_CONTRACT,
        "task": task_name,
        "proposal_version": proposal_version,
        "n_candidates": len(ordered),
        "queue_path": str(queue_path),
        "queue_sha256": sha256_file(queue_path),
        "proposal_audit": (
            {
                "path": str(proposal_audit_path),
                "sha256": sha256_file(proposal_audit_path),
                "n_rows": len(proposal_audit_rows),
            }
            if proposal_audit_rows
            else None
        ),
        "source_artifacts": [
            {"path": str(path), "sha256": sha256_file(path)} for path in source_artifacts
        ],
        "policy": (
            "Every queue row requires exactly one terminal accepted/rejected verdict; "
            "the benchmark builder rejects incomplete or stale ledgers."
        ),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def merge_terminal_verdicts(
    *,
    queue_path: Path,
    verdict_path: Path,
) -> list[dict[str, Any]]:
    queue = _read_jsonl(queue_path)
    verdicts = _read_jsonl(verdict_path)
    by_id = {str(row["source_record_id"]): row for row in queue}
    if len(by_id) != len(queue):
        raise ValueError("Review queue has duplicate source_record_id values")
    verdict_by_id: dict[str, dict[str, Any]] = {}
    for verdict in verdicts:
        record_id = str(verdict.get("source_record_id") or "")
        if not record_id or record_id in verdict_by_id:
            raise ValueError(f"Missing or duplicate review verdict id: {record_id!r}")
        verdict_by_id[record_id] = verdict
    missing = sorted(set(by_id) - set(verdict_by_id))
    extra = sorted(set(verdict_by_id) - set(by_id))
    if missing or extra:
        raise ValueError(
            f"Review coverage mismatch: missing={len(missing)} extra={len(extra)}; "
            f"examples={missing[:3]} {extra[:3]}"
        )

    merged = []
    for record_id, candidate in by_id.items():
        verdict = verdict_by_id[record_id]
        if verdict.get("source_payload_sha256") != candidate.get("source_payload_sha256"):
            raise ValueError(f"Stale review verdict payload hash for {record_id}")
        status = str(verdict.get("review_status") or "")
        if status not in {"accepted", "rejected"}:
            raise ValueError(f"Non-terminal verdict for {record_id}: {status}")
        row = {
            **candidate,
            "review_status": status,
            "reviewer": str(verdict.get("reviewer") or ""),
            "review_reason": str(verdict.get("review_reason") or ""),
            "review_notes": str(verdict.get("review_notes") or ""),
            "review_contract": REVIEW_CONTRACT,
        }
        if status == "accepted":
            row["condition_group"] = str(
                verdict.get("condition_group") or candidate.get("proposed_condition_group") or ""
            )
            row["condition_atoms"] = list(
                verdict.get("condition_atoms") or candidate.get("proposed_condition_atoms") or []
            )
            if verdict.get("Y") is not None:
                row["Y"] = int(verdict["Y"])
            if verdict.get("label_method"):
                row["label_method"] = str(verdict["label_method"])
        merged.append(row)
    return merged


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
