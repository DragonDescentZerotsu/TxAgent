"""Compile constrained AMES candidate selections into exact scalar pairs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

CANDIDATE_SELECTION_VERSION = "ames_measurement_candidate_selection.v1"
CANDIDATE_CONTRACT_VERSION = "ames_measurement_candidates.v1"
SELECTION_COMPILER_VERSION = "ames_measurement_selection_compile.v1"
CANDIDATE_UNIT_SENTINEL = "candidate_id"
TERMINAL_STATUSES = frozenset({"ok", "relative", "unsure", "unavailable"})
REASON_TO_STATUS = {
    "absolute": "ok",
    "experiment_relative": "relative",
    "missing_unit": "unsure",
    "bound_or_range": "unsure",
    "multiple_or_conflict": "unsure",
    "no_eligible_numeric": "unavailable",
}
RESPONSE_FIELDS = {"id", "reason", "candidate_id", "status", "measurements"}
RESPONSE_ID = re.compile(r"r[1-9]\d*")
PLAIN_DECIMAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")


def _validated_measurement(value: object) -> str:
    text = str(value or "").strip()
    try:
        finite = Decimal(text).is_finite()
    except InvalidOperation:
        finite = False
    if not PLAIN_DECIMAL.fullmatch(text) or not finite:
        raise ValueError("measurement candidate must be a finite plain decimal")
    return text


def load_candidates(value: object) -> list[dict[str, Any]]:
    """Load one row's numbered pair inventory and reject ambiguous IDs."""
    try:
        rows = json.loads(str(value or "[]"))
    except json.JSONDecodeError as error:
        raise ValueError("invalid measurement_candidates_json") from error
    if not isinstance(rows, list):
        raise TypeError("measurement candidates must be a list")
    cleaned = [_candidate(row) for row in rows]
    ids = [row["candidate_id"] for row in cleaned]
    pairs = [(row["measurement"], row["unit"]) for row in cleaned]
    if len(ids) != len(set(ids)) or len(pairs) != len(set(pairs)):
        raise ValueError("measurement candidates contain duplicate IDs or pairs")
    return cleaned


def _candidate(row: object) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise TypeError("measurement candidate is not an object")
    required = ("candidate_id", "measurement", "unit", "rule_id")
    values: dict[str, Any] = {
        field: str(row.get(field) or "").strip() for field in required
    }
    values["measurement"] = _validated_measurement(row.get("measurement"))
    evidence, hints = row.get("evidence"), row.get("hints")
    if any(not values[field] for field in required):
        raise ValueError("measurement candidate has an empty required field")
    if values["unit"] == CANDIDATE_UNIT_SENTINEL:
        raise ValueError("measurement candidate uses the reserved selection sentinel")
    if (
        not isinstance(evidence, list)
        or not evidence
        or any(not isinstance(item, str) or not item.strip() for item in evidence)
    ):
        raise ValueError("measurement candidate evidence must be a nonempty text list")
    if not isinstance(hints, list) or any(not isinstance(item, str) for item in hints):
        raise ValueError("measurement candidate hints must be a text list")
    values.update(evidence=evidence, hints=hints)
    candidate_id = values["candidate_id"]
    if (
        not candidate_id.isdigit()
        or int(candidate_id) < 1
        or str(int(candidate_id)) != candidate_id
    ):
        raise ValueError("measurement candidate ID must be a positive integer")
    return values


def candidate_set_sha256(value: object) -> str:
    """Hash the validated canonical candidate list, independent of JSON spacing."""
    candidates = load_candidates(value)
    payload = json.dumps(
        candidates, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selected_id(row: Mapping[str, Any]) -> str:
    try:
        measurements = json.loads(str(row.get("measurements_json") or "[]"))
    except json.JSONDecodeError as error:
        raise ValueError("invalid selection measurements_json") from error
    if not isinstance(measurements, list) or len(measurements) != 1:
        raise ValueError("ok candidate selection must contain exactly one ID")
    selected = measurements[0]
    if not isinstance(selected, Mapping):
        raise TypeError("candidate selection ID is not an object")
    candidate_id = str(selected.get("measurement") or "").strip()
    if selected.get("unit") != CANDIDATE_UNIT_SENTINEL or not candidate_id.isdigit():
        raise ValueError("ok selection did not use the candidate-ID sentinel")
    return candidate_id


def _selection_contract(row: Mapping[str, Any]) -> tuple[str, str | None]:
    try:
        raw = json.loads(str(row.get("raw_response_json") or ""))
    except json.JSONDecodeError as error:
        raise ValueError("candidate selection lacks valid raw response JSON") from error
    if not isinstance(raw, Mapping):
        raise TypeError("candidate selection raw response is not an object")
    if set(raw) != RESPONSE_FIELDS:
        raise ValueError("candidate selection raw response fields disagree")
    if not RESPONSE_ID.fullmatch(str(raw.get("id") or "")):
        raise ValueError("candidate selection lacks a batch-local response ID")
    reason = str(raw.get("reason") or "")
    if reason not in REASON_TO_STATUS:
        raise ValueError(f"unsupported candidate selection reason: {reason!r}")
    status = str(row.get("status") or "")
    if status != REASON_TO_STATUS[reason] or raw.get("status") != status:
        raise ValueError("candidate selection reason and status disagree")
    raw_id = raw.get("candidate_id")
    candidate_id = str(raw_id).strip() if raw_id is not None else None
    if candidate_id is not None and (
        not candidate_id.isdigit() or str(int(candidate_id)) != candidate_id
    ):
        raise ValueError("selected candidate ID is not canonical")
    if reason == "absolute" and candidate_id != _selected_id(row):
        raise ValueError("absolute response candidate IDs disagree")
    if reason != "absolute" and candidate_id is not None:
        raise ValueError("non-absolute response carries a candidate ID")
    try:
        selected = json.loads(str(row.get("measurements_json") or "[]"))
    except json.JSONDecodeError as error:
        raise ValueError("invalid selection measurements_json") from error
    if raw.get("measurements") != selected:
        raise ValueError("raw and validated selection measurements disagree")
    if reason != "absolute" and selected:
        raise ValueError("non-ok candidate selection carries a hidden measurement")
    return reason, candidate_id


def compile_selection(
    selection: Mapping[str, Any], candidate_row: Mapping[str, Any]
) -> dict[str, Any]:
    """Replace one model-selected integer with its frozen source-grounded pair."""
    status = str(selection.get("status") or "")
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"nonterminal candidate selection status: {status!r}")
    candidates = load_candidates(candidate_row.get("measurement_candidates_json"))
    expected_hash = str(candidate_row.get("candidate_set_sha256") or "")
    if (
        candidate_set_sha256(candidate_row.get("measurement_candidates_json"))
        != expected_hash
    ):
        raise ValueError("measurement candidate-set hash mismatch")
    if candidate_row.get("candidate_contract_version") != CANDIDATE_CONTRACT_VERSION:
        raise ValueError("measurement candidate contract version mismatch")
    reason, candidate_id = _selection_contract(selection)
    expected_quantity = 1 if status == "ok" else 0
    if int(selection.get("quantity_count") or 0) != expected_quantity:
        raise ValueError("candidate selection quantity count disagrees")
    indexed = {row["candidate_id"]: row for row in candidates}
    if candidate_id is not None and candidate_id not in indexed:
        raise ValueError(f"selected candidate ID is absent: {candidate_id}")
    pair = indexed.get(candidate_id or "")
    measurements = (
        [{"measurement": pair["measurement"], "unit": pair["unit"]}]
        if pair is not None
        else []
    )
    return {
        **dict(selection),
        "status": status,
        "measurements_json": json.dumps(measurements, ensure_ascii=False),
        "quantity_count": len(measurements),
        "selection_reason": reason,
        "selected_candidate_id": candidate_id,
        "selected_candidate_evidence_json": (
            json.dumps(pair["evidence"], ensure_ascii=False) if pair else None
        ),
        "selected_candidate_rule_id": pair["rule_id"] if pair else None,
        "candidate_set_sha256": expected_hash,
        "candidate_contract_version": CANDIDATE_CONTRACT_VERSION,
        "resolution_method": "constrained_candidate_selection",
        "selection_compiler_version": SELECTION_COMPILER_VERSION,
    }


def compile_selections(
    selections: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compile a complete identity-matched selection table."""
    candidates = _index_rows(candidate_rows, "cleaned_record_id")
    selected = _index_rows(selections, "cleaned_record_id")
    if set(candidates) != set(selected):
        raise ValueError("candidate and selection row coverage differs")
    output = []
    for record_id in sorted(selected):
        source_uid = candidates[record_id].get("source_row_uid")
        if selected[record_id].get("source_row_uid") != source_uid:
            raise ValueError(f"candidate selection source UID mismatch: {record_id}")
        source_id = candidates[record_id].get("source_id")
        if selected[record_id].get("source_id") != source_id:
            raise ValueError(f"candidate selection source mismatch: {record_id}")
        compiled = compile_selection(selected[record_id], candidates[record_id])
        compiled["canonical_endpoint_name"] = candidates[record_id].get(
            "canonical_endpoint_name"
        )
        output.append(compiled)
    return output


def _index_rows(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = str(row.get(field) or "")
        if not key or key in indexed:
            raise ValueError(f"missing or duplicate {field}: {key!r}")
        indexed[key] = row
    return indexed


__all__ = [
    "CANDIDATE_CONTRACT_VERSION",
    "CANDIDATE_SELECTION_VERSION",
    "CANDIDATE_UNIT_SENTINEL",
    "SELECTION_COMPILER_VERSION",
    "candidate_set_sha256",
    "compile_selection",
    "compile_selections",
    "load_candidates",
]
