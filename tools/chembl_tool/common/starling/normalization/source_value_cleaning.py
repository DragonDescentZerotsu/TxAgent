"""Audited source-visible measurement/unit cleaning for Starling v7 records.

The immutable source files remain untouched.  This module changes only the
Stage-01 source view, records every changed or dropped row in a sidecar, and
requires an exact reviewed ledger entry for nondeterministic changes.
Support text is immutable after the common ingestion-only Unicode/whitespace
cleanup; it may inform a repair but is never itself a repair target.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.evidence_library import starling_molecule_id

from .cleaning import (
    clean_measurement_text,
    clean_text,
    file_sha256,
    resolve_structure_value,
)


SOURCE_VALUE_CLEANING_VERSION = "starling_source_value_cleaning.v6"
SUPPORT_TEXT_POLICY_VERSION = "support_text_immutable_after_ingestion.v1"
ENCODED_SPACE_RULE = "percent_0020_space.v1"

_ALLOWED_REPAIR_FIELDS = frozenset({"measurement_text", "unit_text", "smiles"})
_SMILES_REPAIR_CLASSIFICATIONS = frozenset({"confirmed_mismatch", "missing_smiles"})
_REPAIR_KEYS = frozenset(
    {
        "repair_id",
        "task_id",
        "source_id",
        "source_sha256",
        "source_row_number",
        "source_record_id",
        "before",
        "after",
        "evidence",
    }
)
_DROP_KEYS = frozenset(
    {
        "drop_id",
        "task_id",
        "source_id",
        "source_sha256",
        "source_row_number",
        "source_record_id",
        "reason_code",
        "evidence",
    }
)
_POWER_OF_TEN_UNIT = re.compile(
    r"10\s*(?:\^|\*\*)?\s*[+\-−–]?\s*\d+", flags=re.IGNORECASE
)
_SCIENTIFIC_VALUE = re.compile(
    r"(?P<coefficient>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(?:[x×*·]\s*)?10\s*(?:\^|\*\*)?\s*"
    r"(?P<exponent>[+\-−–]\s*\d+)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ReviewedSourceValueRepair:
    repair_id: str
    task_id: str
    source_id: str
    source_sha256: str
    source_row_number: int
    source_record_id: str
    before: Mapping[str, Any]
    after: Mapping[str, Any]
    evidence: Mapping[str, Any]

    @property
    def row_key(self) -> tuple[str, int, str]:
        return self.source_id, self.source_row_number, self.source_record_id


@dataclass(frozen=True)
class SourceValueCleaningResult:
    records: list[dict[str, Any]]
    audit_rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    input_paths: tuple[Path, ...] = ()


def load_reviewed_repairs(
    path: str | Path | None,
    *,
    task_id: str,
) -> list[ReviewedSourceValueRepair]:
    if path is None:
        return []
    target = Path(path)
    repairs: list[ReviewedSourceValueRepair] = []
    for line_number, line in enumerate(
        target.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if set(payload) != _REPAIR_KEYS:
            raise ValueError(
                f"{target}:{line_number} has invalid repair fields: "
                f"{sorted(set(payload) ^ _REPAIR_KEYS)}"
            )
        before = payload["before"]
        after = payload["after"]
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise ValueError(f"{target}:{line_number} before/after must be objects")
        if not before or set(before) != set(after):
            raise ValueError(
                f"{target}:{line_number} before/after must name the same fields"
            )
        if not set(before) <= _ALLOWED_REPAIR_FIELDS:
            raise ValueError(f"{target}:{line_number} repairs a forbidden field")
        if all(before[field] == after[field] for field in before):
            raise ValueError(f"{target}:{line_number} repair changes nothing")
        if "smiles" in after:
            from rdkit import Chem

            if not isinstance(after["smiles"], str) or Chem.MolFromSmiles(
                after["smiles"]
            ) is None:
                raise ValueError(
                    f"{target}:{line_number} has an invalid replacement SMILES"
                )
        evidence = payload["evidence"]
        if not isinstance(evidence, Mapping) or len(str(evidence.get("note") or "")) < 40:
            raise ValueError(
                f"{target}:{line_number} requires a substantive evidence note"
            )
        repair = ReviewedSourceValueRepair(
            repair_id=str(payload["repair_id"]),
            task_id=str(payload["task_id"]),
            source_id=str(payload["source_id"]),
            source_sha256=str(payload["source_sha256"]),
            source_row_number=int(payload["source_row_number"]),
            source_record_id=str(payload["source_record_id"]),
            before=dict(before),
            after=dict(after),
            evidence=dict(evidence),
        )
        if repair.task_id != task_id:
            raise ValueError(
                f"{target}:{line_number} belongs to task {repair.task_id!r}, "
                f"not {task_id!r}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", repair.source_sha256):
            raise ValueError(f"{target}:{line_number} has an invalid source SHA-256")
        repairs.append(repair)

    repair_ids = [repair.repair_id for repair in repairs]
    row_keys = [repair.row_key for repair in repairs]
    if len(repair_ids) != len(set(repair_ids)):
        raise ValueError(f"{target} contains duplicate repair IDs")
    if len(row_keys) != len(set(row_keys)):
        raise ValueError(f"{target} targets one source row more than once")
    return repairs


def load_reviewed_smiles_conflicts(
    path: str | Path | None, *, task_id: str
) -> tuple[dict[tuple[str, int, str], dict[str, Any]], Counter[str]]:
    """Load the frozen name/SMILES decisions for one task."""
    if path is None:
        return {}, Counter()
    import pyarrow.parquet as pq

    target = Path(path)
    required = {
        "candidate_id",
        "task_id",
        "source_id",
        "source_sha256",
        "source_row_number",
        "source_record_id",
        "canonical_smiles",
        "decision",
        "override_smiles",
        "confidence",
        "rationale",
    }
    schema = set(pq.read_schema(target).names)
    if missing := required - schema:
        raise ValueError(f"{target} lacks reviewed SMILES fields: {sorted(missing)}")
    decisions: Counter[str] = Counter()
    overrides: dict[tuple[str, int, str], dict[str, Any]] = {}
    candidate_ids: set[str] = set()
    row_keys: set[tuple[str, int, str]] = set()
    for row in pq.read_table(target, columns=sorted(required)).to_pylist():
        if str(row.get("task_id") or "") != task_id:
            continue
        candidate_id = str(row.get("candidate_id") or "")
        decision = str(row.get("decision") or "")
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError(f"{target} has a missing or duplicate candidate ID")
        if decision not in {"override", "reject"}:
            raise ValueError(f"{target} has invalid decision for {candidate_id}")
        confidence = str(row.get("confidence") or "")
        if confidence not in {"high", "medium", "low"}:
            raise ValueError(f"{target} has invalid confidence for {candidate_id}")
        key = (
            str(row.get("source_id") or ""),
            int(row.get("source_row_number") or 0),
            str(row.get("source_record_id") or ""),
        )
        if key in row_keys:
            raise ValueError(f"{target} reviews one source row more than once")
        candidate_ids.add(candidate_id)
        row_keys.add(key)
        decisions[decision] += 1
        if decision == "reject":
            if row.get("override_smiles") is not None:
                raise ValueError(f"rejected candidate {candidate_id} has an override")
            continue
        replacement = str(row.get("override_smiles") or "")
        if not replacement:
            raise ValueError(f"candidate {candidate_id} has no override SMILES")
        _, status = resolve_structure_value(replacement, structure_mode="direct")
        if status != "resolved":
            raise ValueError(
                f"candidate {candidate_id} has invalid override SMILES"
            )
        if confidence != "high":
            decisions["override"] -= 1
            decisions["quarantined_override"] += 1
            continue
        overrides[key] = {
            **row,
            "override_smiles": replacement,
            "override_structure_status": status,
        }
    return overrides, decisions


def load_smiles_identity_audit(
    path: str | Path | None, *, task_id: str
) -> dict[str, Mapping[str, Any]]:
    if path is None:
        return {}
    target = Path(path)
    entries: dict[str, Mapping[str, Any]] = {}
    required = {
        "audit_id",
        "task_id",
        "source_id",
        "source_sha256",
        "source_row_number",
        "source_record_id",
        "stored_smiles",
        "classification",
    }
    for line_number, line in enumerate(
        target.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        missing = required - set(payload)
        if missing:
            raise ValueError(
                f"{target}:{line_number} lacks audit fields: {sorted(missing)}"
            )
        if payload["task_id"] != task_id:
            raise ValueError(
                f"{target}:{line_number} belongs to task {payload['task_id']!r}, "
                f"not {task_id!r}"
            )
        audit_id = str(payload["audit_id"])
        if audit_id in entries:
            raise ValueError(f"{target} contains duplicate audit ID {audit_id!r}")
        entries[audit_id] = payload
    return entries


def load_reviewed_drops(
    path: str | Path | None, *, task_id: str
) -> list[dict[str, Any]]:
    if path is None:
        return []
    target = Path(path)
    drops: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        target.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if set(payload) != _DROP_KEYS:
            raise ValueError(
                f"{target}:{line_number} has invalid drop fields: "
                f"{sorted(set(payload) ^ _DROP_KEYS)}"
            )
        if payload["task_id"] != task_id:
            raise ValueError(
                f"{target}:{line_number} belongs to task {payload['task_id']!r}, "
                f"not {task_id!r}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", str(payload["source_sha256"])):
            raise ValueError(f"{target}:{line_number} has an invalid source SHA-256")
        if not re.fullmatch(r"[a-z0-9_]+", str(payload["reason_code"])):
            raise ValueError(f"{target}:{line_number} has an invalid reason code")
        evidence = payload["evidence"]
        if not isinstance(evidence, Mapping) or len(str(evidence.get("note") or "")) < 40:
            raise ValueError(
                f"{target}:{line_number} requires a substantive evidence note"
            )
        drops.append(dict(payload))

    drop_ids = [str(drop["drop_id"]) for drop in drops]
    row_keys = [
        (
            str(drop["source_id"]),
            int(drop["source_row_number"]),
            str(drop["source_record_id"]),
        )
        for drop in drops
    ]
    if len(drop_ids) != len(set(drop_ids)):
        raise ValueError(f"{target} contains duplicate drop IDs")
    if len(row_keys) != len(set(row_keys)):
        raise ValueError(f"{target} drops one source row more than once")
    return drops


def clean_source_values(
    records: Sequence[Mapping[str, Any]],
    *,
    task_id: str,
    reviewed_repairs_path: str | Path | None = None,
    reviewed_drops_path: str | Path | None = None,
    smiles_identity_audit_path: str | Path | None = None,
    reviewed_smiles_conflicts_path: str | Path | None = None,
    require_all_reviewed_repairs: bool = True,
    require_all_reviewed_drops: bool = True,
    require_all_reviewed_smiles_overrides: bool = True,
    require_scientific_scale_review: bool = False,
) -> SourceValueCleaningResult:
    """Return cleaned records and a complete field-level change audit."""
    repairs = load_reviewed_repairs(reviewed_repairs_path, task_id=task_id)
    smiles_audit = load_smiles_identity_audit(
        smiles_identity_audit_path, task_id=task_id
    )
    smiles_overrides, smiles_decisions = load_reviewed_smiles_conflicts(
        reviewed_smiles_conflicts_path, task_id=task_id
    )
    drops = load_reviewed_drops(reviewed_drops_path, task_id=task_id)
    repairs_by_row = {repair.row_key: repair for repair in repairs}
    drops_by_row = {
        (
            str(drop["source_id"]),
            int(drop["source_row_number"]),
            str(drop["source_record_id"]),
        ): drop
        for drop in drops
    }
    overlap = sorted(set(repairs_by_row) & set(drops_by_row))
    if overlap:
        raise ValueError(f"source rows cannot be both repaired and dropped: {overlap[0]}")
    applied_repairs: set[str] = set()
    applied_drops: set[str] = set()
    applied_smiles_overrides: set[str] = set()
    output: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    scale_candidates = 0
    unresolved_scale_candidates: list[str] = []

    for source_record in records:
        # Builder-owned rows are mutable dictionaries.  Reuse them so a full
        # 800k-row task does not temporarily duplicate the entire Stage-01
        # object graph; callers that supply another Mapping still get a dict.
        record = (
            source_record
            if isinstance(source_record, dict)
            else dict(source_record)
        )
        row_key = (
            str(record.get("source_id") or ""),
            int(record.get("source_row_number") or 0),
            str(record.get("source_record_id") or ""),
        )
        drop = drops_by_row.get(row_key)
        if drop is not None:
            audit_id = str(drop["evidence"].get("audit_id") or "")
            if audit_id:
                entry = smiles_audit.get(audit_id)
                audit_key = (
                    str(entry.get("source_id")) if entry else "",
                    int(entry.get("source_row_number")) if entry else 0,
                    str(entry.get("source_record_id")) if entry else "",
                )
                if entry is None or audit_key != row_key:
                    raise ValueError(
                        f"reviewed drop {drop['drop_id']!r} disagrees with its "
                        "SMILES identity audit entry"
                    )
            actual_sha = str(record.get("source_sha256") or "")
            if actual_sha != drop["source_sha256"]:
                raise ValueError(
                    f"reviewed drop {drop['drop_id']!r} source drift: "
                    f"expected {drop['source_sha256']}, found {actual_sha or '<missing>'}"
                )
            audit.append(
                _audit_row(
                    record,
                    "record",
                    "retained",
                    "dropped",
                    f"reviewed_drop:{drop['drop_id']}",
                    review_status="reviewed",
                    evidence=drop["evidence"],
                )
            )
            applied_drops.add(str(drop["drop_id"]))
            continue
        original_support_text = record.get("support_text")
        original_measurement = _optional_text(record.get("measurement_text"))
        measurement, measurement_steps = _clean_measurement(original_measurement)
        record["measurement_text"] = measurement
        for rule_id, before, after in measurement_steps:
            audit.append(
                _audit_row(record, "measurement_text", before, after, rule_id)
            )
        scale_candidate = _has_scientific_scale_conflict(record)
        scale_candidates += scale_candidate

        repair = repairs_by_row.get(row_key)
        if repair is not None:
            if "smiles" in repair.after:
                audit_id = str(repair.evidence.get("audit_id") or "")
                entry = smiles_audit.get(audit_id)
                if entry is None:
                    raise ValueError(
                        f"reviewed SMILES repair {repair.repair_id!r} lacks a matching "
                        "audit entry"
                    )
                audit_key = (
                    str(entry["source_id"]),
                    int(entry["source_row_number"]),
                    str(entry["source_record_id"]),
                )
                if (
                    audit_key != repair.row_key
                    or entry["stored_smiles"] != repair.before["smiles"]
                ):
                    raise ValueError(
                        f"reviewed SMILES repair {repair.repair_id!r} disagrees with "
                        "its audit entry"
                    )
                if entry["classification"] not in _SMILES_REPAIR_CLASSIFICATIONS:
                    raise ValueError(
                        f"reviewed SMILES repair {repair.repair_id!r} has non-repair "
                        f"audit classification {entry['classification']!r}"
                    )
            actual_sha = str(record.get("source_sha256") or "")
            if actual_sha != repair.source_sha256:
                raise ValueError(
                    f"reviewed repair {repair.repair_id!r} source drift: "
                    f"expected {repair.source_sha256}, found {actual_sha or '<missing>'}"
                )
            for field, expected_before in repair.before.items():
                actual_before = (
                    record.get("source_smiles", record.get("smiles"))
                    if field == "smiles"
                    else record.get(field)
                )
                if actual_before != expected_before:
                    raise ValueError(
                        f"reviewed repair {repair.repair_id!r} precondition drift for "
                        f"{field}: expected {expected_before!r}, found {actual_before!r}"
                    )
            for field, after in repair.after.items():
                if field == "smiles":
                    before = record.get("source_smiles", record.get("smiles"))
                    _apply_smiles_repair(record, str(after))
                else:
                    before = record.get(field)
                    record[field] = after
                audit.append(
                    _audit_row(
                        record,
                        field,
                        before,
                        after,
                        f"reviewed:{repair.repair_id}",
                        review_status="reviewed",
                        evidence=repair.evidence,
                    )
                )
            applied_repairs.add(repair.repair_id)
        smiles_override = smiles_overrides.get(row_key)
        if smiles_override is not None:
            candidate_id = str(smiles_override["candidate_id"])
            actual_sha = str(record.get("source_sha256") or "")
            if actual_sha != str(smiles_override["source_sha256"]):
                raise ValueError(
                    f"reviewed SMILES override {candidate_id!r} source drift"
                )
            before = record.get("canonical_smiles")
            expected_before = smiles_override.get("canonical_smiles")
            expected_canonical, expected_status = resolve_structure_value(
                expected_before, structure_mode="direct"
            )
            actual_canonical, actual_status = resolve_structure_value(
                before, structure_mode="direct"
            )
            if (
                expected_status != "resolved"
                or actual_status != "resolved"
                or expected_canonical != actual_canonical
            ):
                raise ValueError(
                    f"reviewed SMILES override {candidate_id!r} precondition drift: "
                    f"expected {expected_before!r}, "
                    f"found {before!r}"
                )
            replacement = str(smiles_override["override_smiles"])
            _apply_smiles_repair(record, replacement)
            audit.append(
                _audit_row(
                    record,
                    "smiles",
                    before,
                    replacement,
                    f"reviewed_name_smiles_override:{candidate_id}",
                    review_status="reviewed",
                    evidence={"note": smiles_override.get("rationale")},
                )
            )
            applied_smiles_overrides.add(candidate_id)
        if scale_candidate and _has_scientific_scale_conflict(record):
            unresolved_scale_candidates.append(
                str(record.get("cleaned_record_id") or record.get("source_record_id") or "")
            )
        if record.get("support_text") != original_support_text:
            raise ValueError(
                "source-value cleaning modified immutable support_text for "
                f"{record.get('cleaned_record_id')!r}"
            )
        output.append(record)

    unapplied = sorted({repair.repair_id for repair in repairs} - applied_repairs)
    if require_all_reviewed_repairs and unapplied:
        raise ValueError(
            "reviewed source-value repairs were not applied: " + ", ".join(unapplied)
        )
    unapplied_drops = sorted(
        {str(drop["drop_id"]) for drop in drops} - applied_drops
    )
    if require_all_reviewed_drops and unapplied_drops:
        raise ValueError(
            "reviewed source-row drops were not applied: " + ", ".join(unapplied_drops)
        )
    unapplied_smiles_overrides = sorted(
        {
            str(decision["candidate_id"])
            for decision in smiles_overrides.values()
        }
        - applied_smiles_overrides
    )
    if require_all_reviewed_smiles_overrides and unapplied_smiles_overrides:
        raise ValueError(
            "reviewed SMILES overrides were not applied: "
            + ", ".join(unapplied_smiles_overrides[:10])
        )
    if require_scientific_scale_review and unresolved_scale_candidates:
        raise ValueError(
            f"{len(unresolved_scale_candidates)} source value(s) already include the "
            "scientific factor retained in their unit; add reviewed repairs; "
            f"first={unresolved_scale_candidates[0]}"
        )

    changed_records = {str(row["cleaned_record_id"]) for row in audit}
    rule_counts = Counter(str(row["rule_id"]) for row in audit)
    field_counts = Counter(str(row["field"]) for row in audit)
    registry_path = Path(reviewed_repairs_path) if reviewed_repairs_path else None
    drop_registry_path = Path(reviewed_drops_path) if reviewed_drops_path else None
    smiles_audit_path = (
        Path(smiles_identity_audit_path) if smiles_identity_audit_path else None
    )
    smiles_conflicts_path = (
        Path(reviewed_smiles_conflicts_path)
        if reviewed_smiles_conflicts_path
        else None
    )
    manifest = {
        "version": SOURCE_VALUE_CLEANING_VERSION,
        "task_id": task_id,
        "n_input_records": len(records),
        "n_output_records": len(output),
        "n_dropped_records": len(applied_drops),
        "n_changed_records": len(changed_records),
        "n_field_changes": len(audit),
        "rule_counts": dict(sorted(rule_counts.items())),
        "field_counts": dict(sorted(field_counts.items())),
        "scientific_scale_review": {
            "n_candidates": scale_candidates,
            "n_resolved_by_reviewed_repair": (
                scale_candidates - len(unresolved_scale_candidates)
            ),
            "n_unresolved": len(unresolved_scale_candidates),
            "all_candidates_resolved": not unresolved_scale_candidates,
        },
        "support_text_policy": {
            "version": SUPPORT_TEXT_POLICY_VERSION,
            "mode": "immutable_after_ingestion",
        },
        "reviewed_repairs": {
            "path": str(registry_path) if registry_path else None,
            "sha256": file_sha256(registry_path) if registry_path else None,
            "n_declared": len(repairs),
            "n_applied": len(applied_repairs),
            "unapplied_repair_ids": unapplied,
            "all_required_repairs_applied": not unapplied,
        },
        "reviewed_drops": {
            "path": str(drop_registry_path) if drop_registry_path else None,
            "sha256": file_sha256(drop_registry_path) if drop_registry_path else None,
            "n_declared": len(drops),
            "n_applied": len(applied_drops),
            "unapplied_drop_ids": unapplied_drops,
            "all_required_drops_applied": not unapplied_drops,
        },
        "smiles_identity_audit": {
            "path": str(smiles_audit_path) if smiles_audit_path else None,
            "sha256": file_sha256(smiles_audit_path) if smiles_audit_path else None,
            "n_entries": len(smiles_audit),
        },
        "reviewed_name_smiles_conflicts": {
            "path": str(smiles_conflicts_path) if smiles_conflicts_path else None,
            "sha256": (
                file_sha256(smiles_conflicts_path) if smiles_conflicts_path else None
            ),
            "decision_counts": dict(sorted(smiles_decisions.items())),
            "n_declared_overrides": (
                smiles_decisions.get("override", 0)
                + smiles_decisions.get("quarantined_override", 0)
            ),
            "n_eligible_overrides": len(smiles_overrides),
            "n_quarantined_overrides": smiles_decisions.get(
                "quarantined_override", 0
            ),
            "n_applied_overrides": len(applied_smiles_overrides),
            "n_rdkit_invalid_override_strings": 0,
            "unapplied_override_ids": unapplied_smiles_overrides,
            "all_required_overrides_applied": not unapplied_smiles_overrides,
        },
    }
    return SourceValueCleaningResult(
        records=output,
        audit_rows=audit,
        manifest=manifest,
        input_paths=tuple(
            path
            for path in (
                registry_path,
                drop_registry_path,
                smiles_audit_path,
                smiles_conflicts_path,
            )
            if path is not None
        ),
    )


def _clean_measurement(
    value: str | None,
) -> tuple[str | None, list[tuple[str, str, str]]]:
    return _replace_encoded_spaces(value)


def _has_scientific_scale_conflict(record: Mapping[str, Any]) -> bool:
    """Flag a nonzero value that support text already expresses with its unit scale."""
    unit = str(record.get("unit_text") or "")
    if _POWER_OF_TEN_UNIT.search(unit) is None:
        return False
    try:
        value = Decimal(str(record.get("measurement_text") or "").strip())
    except (InvalidOperation, ValueError):
        return False
    if not value.is_finite() or value == 0:
        return False
    for match in _SCIENTIFIC_VALUE.finditer(str(record.get("support_text") or "")):
        exponent = match.group("exponent").replace("−", "-").replace("–", "-")
        expressed = Decimal(match.group("coefficient")).scaleb(
            int(exponent.replace(" ", ""))
        )
        if expressed == value:
            return True
    return False


def _replace_encoded_spaces(
    value: str | None,
) -> tuple[str | None, list[tuple[str, str, str]]]:
    if value is None or "%0020" not in value:
        return value, []
    replaced = clean_measurement_text(value.replace("%0020", "% "))
    if replaced == value:
        return value, []
    return replaced, [(ENCODED_SPACE_RULE, value, str(replaced))]


def _audit_row(
    record: Mapping[str, Any],
    field: str,
    before: Any,
    after: Any,
    rule_id: str,
    *,
    review_status: str = "deterministic",
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = evidence or {}
    return {
        "cleaning_version": SOURCE_VALUE_CLEANING_VERSION,
        "cleaned_record_id": str(record.get("cleaned_record_id") or ""),
        "source_id": str(record.get("source_id") or ""),
        "source_sha256": str(record.get("source_sha256") or ""),
        "source_row_number": int(record.get("source_row_number") or 0),
        "source_record_id": str(record.get("source_record_id") or ""),
        "field": field,
        "before": before,
        "after": after,
        "rule_id": rule_id,
        "review_status": review_status,
        "evidence_id": str(evidence.get("pmid") or evidence.get("source") or ""),
        "rationale": str(evidence.get("note") or ""),
    }


def _apply_smiles_repair(record: dict[str, Any], smiles: str) -> None:
    canonical, status = resolve_structure_value(smiles, structure_mode="direct")
    record.update(
        {
            "canonical_smiles": canonical,
            "structure_status": status,
            "molecule_id": starling_molecule_id(canonical) if canonical else None,
        }
    )


def _optional_text(value: Any) -> str | None:
    return clean_text(value)


__all__ = [
    "ENCODED_SPACE_RULE",
    "SOURCE_VALUE_CLEANING_VERSION",
    "SUPPORT_TEXT_POLICY_VERSION",
    "SourceValueCleaningResult",
    "clean_source_values",
    "load_reviewed_drops",
    "load_reviewed_repairs",
    "load_reviewed_smiles_conflicts",
    "load_smiles_identity_audit",
]
