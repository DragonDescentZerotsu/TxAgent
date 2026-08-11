"""Audit one-pass JSONL identity, tool, prompt-hash, and label privacy gates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.json_utils import (
    canonical_json_bytes,
    sha256_file,
    write_json_atomic,
)

from .one_pass_contract import (
    ANALOG_PREDICTION_FIELD,
    IDENTITY_BLIND,
    SINGLE_PREDICTION_FIELD,
    TASK_CONTRACTS,
    VISIBILITY_MODES,
    contract_version_for_visibility,
)


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit(
    path: Path,
    output: Path,
    *,
    expected_rows: int | None = None,
    source_data: Path | None = None,
    expected_visibility_mode: str | None = None,
) -> dict[str, Any]:
    rows = _read(path)
    errors = []
    indices = []
    tool_counts = Counter()
    labels = Counter()
    tasks = Counter()
    subsets = Counter()
    visibility_modes = Counter()
    for line_index, row in enumerate(rows):
        prefix = f"row:{line_index}"
        visibility_mode = str(row.get("visibility_mode") or IDENTITY_BLIND)
        visibility_modes[visibility_mode] += 1
        if visibility_mode not in VISIBILITY_MODES:
            errors.append(f"{prefix}:visibility_mode")
            expected_contract = ""
        else:
            expected_contract = contract_version_for_visibility(visibility_mode)
        if row.get("contract_version") != expected_contract:
            errors.append(f"{prefix}:contract_version")
        source_index = int(row.get("source_index", -1))
        indices.append(source_index)
        labels[int(row.get("gold_label", -1))] += 1
        task = str(row.get("source_task") or "")
        subset = str(row.get("source_subset") or "")
        tasks[task] += 1
        subsets[subset] += 1
        contract = TASK_CONTRACTS.get(task)
        if contract is None:
            errors.append(f"{prefix}:source_task")
        else:
            expected_metadata = {
                "prediction_field": contract.prediction_field,
                "single_prediction_field": SINGLE_PREDICTION_FIELD,
                "analog_prediction_field": ANALOG_PREDICTION_FIELD,
                "negative_value": contract.negative_value,
                "positive_value": contract.positive_value,
            }
            for field, expected in expected_metadata.items():
                if row.get(field) != expected:
                    errors.append(f"{prefix}:{field}")
        if row.get("gold_location") != "environment_private_metadata_only":
            errors.append(f"{prefix}:gold_location")
        if row.get("retrieval_identity_policy") != "parent_disjoint":
            errors.append(f"{prefix}:retrieval_identity_policy")
        if row.get("tool_calls_available_to_llm") is not False:
            errors.append(f"{prefix}:live_tools_enabled")
        messages = row.get("messages") or []
        if [message.get("role") for message in messages] != ["system", "user"]:
            errors.append(f"{prefix}:message_roles")
            continue
        visible = json.dumps(messages, ensure_ascii=False)
        if "gold_label" in visible:
            errors.append(f"{prefix}:gold_visible")
        expected_hash = hashlib.sha256(canonical_json_bytes(messages)).hexdigest()
        if row.get("prompt_sha256") != expected_hash:
            errors.append(f"{prefix}:prompt_hash")
        try:
            payload = json.loads(str(messages[1].get("content") or ""))
        except json.JSONDecodeError:
            errors.append(f"{prefix}:user_json")
            continue
        if payload.get("contract_version") != expected_contract:
            errors.append(f"{prefix}:payload_contract_version")
        schema = payload.get("required_json_schema")
        if not isinstance(schema, dict) or not schema:
            errors.append(f"{prefix}:required_json_schema")
        elif list(schema) != list(row.get("required_fields") or []):
            errors.append(f"{prefix}:required_fields")
        query = payload.get("query") or {}
        if visibility_mode == IDENTITY_BLIND:
            if (
                query.get("identity_hidden") is not True
                or query.get("molecule_id") != "query"
            ):
                errors.append(f"{prefix}:query_identity")
            if query.get("canonical_smiles") or query.get("input_smiles"):
                errors.append(f"{prefix}:query_smiles_visible")
        else:
            if query.get("identity_hidden") is True:
                errors.append(f"{prefix}:query_identity_hidden")
            if not (query.get("canonical_smiles") or query.get("input_smiles")):
                errors.append(f"{prefix}:query_smiles_missing")
        query_tool = query.get("prefetched_molecule_properties") or {}
        tool_counts[str(query_tool.get("status") or "missing")] += 1
        neighbors = payload.get("neighbors") or []
        observed_ids = []
        row_tool_results = [query_tool]
        for neighbor in neighbors:
            identifier = str(neighbor.get("molecule_chembl_id") or "")
            observed_ids.append(identifier)
            if visibility_mode == IDENTITY_BLIND:
                if not identifier.startswith("neighbor_"):
                    errors.append(f"{prefix}:neighbor_id")
                if neighbor.get("canonical_smiles") != "[hidden]":
                    errors.append(f"{prefix}:neighbor_smiles_visible")
            else:
                if identifier.startswith("neighbor_"):
                    errors.append(f"{prefix}:neighbor_id_anonymous")
                if neighbor.get("canonical_smiles") in {None, "", "[hidden]"}:
                    errors.append(f"{prefix}:neighbor_smiles_missing")
            for result in neighbor.get("prefetched_comparisons") or []:
                tool_counts[str(result.get("status") or "missing")] += 1
                row_tool_results.append(result)
        if observed_ids != list(row.get("source_neighbor_ids") or []):
            errors.append(f"{prefix}:neighbor_receipt")
        if visibility_mode == IDENTITY_BLIND and len(observed_ids) != len(
            set(observed_ids)
        ):
            errors.append(f"{prefix}:duplicate_neighbor_alias")
        actual_ok = sum(result.get("status") == "ok" for result in row_tool_results)
        actual_error = sum(
            result.get("status") == "error" for result in row_tool_results
        )
        if row.get("prefetched_tool_count") != len(row_tool_results):
            errors.append(f"{prefix}:prefetched_tool_count")
        if row.get("successful_prefetched_tool_count") != actual_ok:
            errors.append(f"{prefix}:successful_prefetched_tool_count")
        if row.get("failed_prefetched_tool_count") != actual_error:
            errors.append(f"{prefix}:failed_prefetched_tool_count")
        for result in row_tool_results:
            status = result.get("status")
            if status not in {"ok", "error"}:
                continue
            content = str(result.get("content") or "")
            serialized_result = json.dumps(result, ensure_ascii=False)
            if status == "error":
                expected = "Tool result unavailable. Treat this tool comparison as missing evidence."
                if expected not in content:
                    errors.append(f"{prefix}:unredacted_tool_error")
            if any(
                term in serialized_result
                for term in (
                    "AssertionError",
                    "IndexError",
                    "edge_index",
                    "TOOL_RUNTIME_ERROR",
                )
            ):
                errors.append(f"{prefix}:backend_error_detail_visible")

    if len(indices) != len(set(indices)):
        errors.append("duplicate_source_indices")
    if expected_rows is not None and len(rows) != expected_rows:
        errors.append(f"row_count:{len(rows)}!={expected_rows}")
    if expected_rows is not None and sorted(indices) != list(range(expected_rows)):
        errors.append("source_indices_not_contiguous")
    if any(label not in {0, 1} for label in labels):
        errors.append("non_binary_label")
    if any(status not in {"ok", "error"} for status in tool_counts):
        errors.append("prefetched_tool_receipt_missing")
    if len(tasks) != 1:
        errors.append("mixed_source_tasks")
    if len(subsets) != 1:
        errors.append("mixed_source_subsets")
    if len(visibility_modes) != 1:
        errors.append("mixed_visibility_modes")
    if expected_visibility_mode is not None and visibility_modes != Counter(
        {expected_visibility_mode: len(rows)}
    ):
        errors.append(f"visibility_mode_mismatch:{dict(visibility_modes)}")
    source_data_sha256 = ""
    if source_data is not None:
        source_rows = _read(source_data)
        source_data_sha256 = sha256_file(source_data)
        if len(source_rows) < len(rows):
            errors.append(f"source_row_count:{len(source_rows)}<{len(rows)}")
        for row in rows:
            source_index = int(row.get("source_index", -1))
            if source_index < 0 or source_index >= len(source_rows):
                continue
            if int(source_rows[source_index].get("Y", -1)) != int(
                row.get("gold_label", -2)
            ):
                errors.append(f"row:{source_index}:source_gold_mismatch")
            visibility_mode = str(row.get("visibility_mode") or IDENTITY_BLIND)
            if visibility_mode != IDENTITY_BLIND:
                messages = row.get("messages") or []
                try:
                    payload = json.loads(str(messages[1].get("content") or ""))
                except (IndexError, AttributeError, json.JSONDecodeError):
                    continue
                query = payload.get("query") or {}
                if str(query.get("input_smiles") or "") != str(
                    source_rows[source_index].get("drug") or ""
                ):
                    errors.append(f"row:{source_index}:source_query_smiles_mismatch")
    result = {
        "audit_contract": "one_pass_data_audit.v2",
        "status": "pass" if not errors else "fail",
        "data": str(path),
        "data_sha256": sha256_file(path),
        "source_data": str(source_data) if source_data is not None else "",
        "source_data_sha256": source_data_sha256,
        "n_rows": len(rows),
        "label_counts": {str(key): value for key, value in sorted(labels.items())},
        "source_tasks": dict(sorted(tasks.items())),
        "source_subsets": dict(sorted(subsets.items())),
        "visibility_modes": dict(sorted(visibility_modes.items())),
        "tool_status_counts": dict(sorted(tool_counts.items())),
        "duplicate_source_indices": len(indices) - len(set(indices)),
        "source_assistant_outputs_in_prompt": False,
        "gold_visible_in_prompt": any(
            error.endswith("gold_visible") for error in errors
        ),
        "live_tool_calls_available_to_llm": False,
        "errors": errors,
    }
    write_json_atomic(output, result)
    if errors:
        raise ValueError(f"one-pass data audit failed: {errors[:10]}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--source-data", type=Path)
    parser.add_argument("--visibility-mode", choices=sorted(VISIBILITY_MODES))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            audit(
                args.data,
                args.output,
                expected_rows=args.expected_rows,
                source_data=args.source_data,
                expected_visibility_mode=args.visibility_mode,
            ),
            indent=2,
        )
    )
