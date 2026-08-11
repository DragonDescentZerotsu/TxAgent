"""Build frozen one-pass full-flat prompts from three-stage traces.

The prompt is built from the exact LLM-visible inputs of the current three-stage
trace.  It does not expose any assistant output or the evaluation label.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tools.chembl_tool.common.json_utils import canonical_json_bytes, sha256_file

from .one_pass_contract import (
    ANALOG_PREDICTION_FIELD,
    CONTRACT_VERSION,
    SINGLE_PREDICTION_FIELD,
    TASK_CONTRACTS,
)


def _json_user_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("trace row has no messages list")
    payloads = []
    for message in messages:
        if message.get("role") != "user":
            continue
        try:
            payload = json.loads(str(message.get("content") or ""))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(
            payload.get("required_json_schema"), dict
        ):
            payloads.append(payload)
    if len(payloads) != 1:
        raise ValueError(
            f"expected one schema-bearing user payload, got {len(payloads)}"
        )
    return payloads[0]


def _load_trace_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _select_stage(rows: list[dict[str, Any]], task: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row.get("task")) == task]
    if len(selected) != 1:
        raise ValueError(f"expected one {task} trace row, got {len(selected)}")
    if selected[0].get("status") != "ok":
        raise ValueError(f"source {task} trace is not successful")
    return selected[0]


def _identity_safe_neighbor_ids(group_payload: Mapping[str, Any]) -> list[str]:
    identifiers = []
    for neighbor in group_payload.get("neighbors") or []:
        identifier = str(neighbor.get("molecule_chembl_id") or "")
        if not identifier.startswith("neighbor_"):
            raise ValueError(f"non-anonymous neighbor identifier: {identifier}")
        identifiers.append(identifier)
        if str(neighbor.get("canonical_smiles") or "") != "[hidden]":
            raise ValueError(f"neighbor identity is visible: {identifier}")
    return identifiers


def build_one_pass_row_from_trace(
    path: Path,
    *,
    task: str,
    subset: str,
    group_template: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fuse one successful three-stage trace into one label-private prompt row."""

    contract = TASK_CONTRACTS[task]
    rows = _load_trace_rows(path)
    single_row = _select_stage(rows, "single_molecule")
    final_row = _select_stage(rows, "final_summary")
    group_rows = [row for row in rows if str(row.get("task")) == "Flat.all_evidence"]
    if len(group_rows) > 1:
        raise ValueError(
            f"expected at most one Flat.all_evidence trace row, got {len(group_rows)}"
        )
    group_row = group_rows[0] if group_rows else None
    stage_rows = tuple(
        row for row in (single_row, group_row, final_row) if row is not None
    )

    indices = {int(row["index"]) for row in stage_rows}
    labels = {int(row["label"]) for row in stage_rows}
    if len(indices) != 1 or len(labels) != 1:
        raise ValueError("source stages do not describe one aligned sample")
    gold_label = labels.pop()
    if gold_label not in {0, 1}:
        raise ValueError(f"gold label is not binary: {gold_label}")

    single_payload = _json_user_payload(single_row)
    final_payload = _json_user_payload(final_row)
    query = single_payload.get("query")
    if not isinstance(query, dict) or query.get("identity_hidden") is not True:
        raise ValueError("one-pass source query is not identity blind")
    if query != final_payload.get("query"):
        raise ValueError("source stages have different query payloads")
    if group_row is not None:
        group_payload = _json_user_payload(group_row)
        if query != group_payload.get("query"):
            raise ValueError("source stages have different query payloads")
    else:
        if group_template is None:
            raise ValueError("no analog group and no group prompt template")
        group_payload = {
            **dict(group_template),
            "query": query,
            "group": {
                "group_id": "Flat.all_evidence",
                "tier": "Flat",
                "endpoint_group": "all_evidence",
            },
            "neighbors": [],
        }
    neighbor_ids = _identity_safe_neighbor_ids(group_payload)

    single_schema = single_payload.get("required_json_schema")
    analog_schema = group_payload.get("required_json_schema")
    final_schema = final_payload.get("required_json_schema")
    if not all(
        isinstance(schema, dict) and schema
        for schema in (
            single_schema,
            analog_schema,
            final_schema,
        )
    ):
        raise ValueError("source stage is missing a required JSON schema")
    if contract.prediction_field not in final_schema:
        raise ValueError("final schema does not contain the task prediction field")

    required_schema = {
        SINGLE_PREDICTION_FIELD: contract.prediction_schema,
        "single_analysis": single_schema,
        ANALOG_PREDICTION_FIELD: contract.prediction_schema,
        "analog_analysis": analog_schema,
        **final_schema,
    }
    payload = {
        "task": f"One-pass {final_payload.get('task') or task}",
        "contract_version": CONTRACT_VERSION,
        "query": query,
        "retrieval_coverage": final_payload.get("retrieval_coverage") or {},
        "analog_group": group_payload.get("group") or {},
        "neighbors": group_payload.get("neighbors") or [],
        "analysis_order": [
            (
                "First assess the query alone. Set single_prediction using only the "
                "query properties and place the detailed result in single_analysis."
            ),
            (
                "Then assess analog evidence and transferability. Set analog_prediction "
                "using the analog evidence plus query-neighbor comparisons, and place the "
                "detailed result in analog_analysis."
            ),
            (
                f"Finally integrate both assessments and set {contract.prediction_field}. "
                "The branch predictions are intermediate judgments and must not be rewritten "
                "after seeing the integrated conclusion."
            ),
        ],
        "single_instructions": single_payload.get("instructions") or [],
        "analog_instructions": group_payload.get("instructions") or [],
        "final_instructions": final_payload.get("instructions") or [],
        "output_instructions": [
            "Return exactly one compact JSON object and no prose outside it.",
            "Do not identify or name the anonymous query or neighbors.",
            "Do not call tools; every allowed tool result is already in this prompt.",
            (
                "single_prediction, analog_prediction, and the final task prediction must "
                f"each be exactly one of: {contract.positive_value}, {contract.negative_value}."
            ),
        ],
        "required_json_schema": required_schema,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior medicinal-chemistry evidence reasoning model. Perform the "
                "single-molecule prior, full-flat analog transferability analysis, and final "
                "binary adjudication in one response. Keep the three judgments explicit and "
                "return only valid JSON."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    return {
        "messages": messages,
        "gold_label": gold_label,
        "prediction_field": contract.prediction_field,
        "single_prediction_field": SINGLE_PREDICTION_FIELD,
        "analog_prediction_field": ANALOG_PREDICTION_FIELD,
        "negative_value": contract.negative_value,
        "positive_value": contract.positive_value,
        "required_fields": list(required_schema),
        "source_task": task,
        "source_subset": subset,
        "source_index": indices.pop(),
        "source_trace": str(path),
        "source_trace_sha256": sha256_file(path),
        "source_neighbor_ids": neighbor_ids,
        "source_group_available": group_row is not None,
        "materialization_source": "three_stage_trace_replay",
        "formal_rl_contract_eligible": False,
        "prompt_sha256": hashlib.sha256(canonical_json_bytes(messages)).hexdigest(),
        "contract_version": CONTRACT_VERSION,
        "gold_location": "environment_private_metadata_only",
    }
