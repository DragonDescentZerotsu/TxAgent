"""Provider-neutral one-pass data and inference runtime gates.

Prompt construction stays in ``one_pass.py`` and reward arithmetic stays in
``one_pass_reward.py``.  This module owns the shared boundary between an
audited materialized dataset and any frozen-base/checkpoint evaluator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tools.chembl_tool.common.json_utils import canonical_json_bytes, sha256_file
from tools.chembl_tool.common.reasoning_validation import response_validation_errors

from .one_pass_contract import (
    is_one_pass_contract,
    nested_required_fields,
    prediction_to_label,
)


def read_one_pass_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"empty one-pass data: {path}")
    contracts = {str(row.get("contract_version") or "") for row in rows}
    if len(contracts) != 1 or not all(
        is_one_pass_contract(value) for value in contracts
    ):
        raise ValueError("data is not a uniform supported one-pass contract")
    return rows


def load_fresh_one_pass_audit(
    data_path: Path,
    *,
    minimum_contract: str = "one_pass_data_audit.v1",
) -> dict[str, Any]:
    """Require the adjacent privacy/tool audit before training or evaluation."""

    audit_path = data_path.with_suffix(".audit.json")
    if not audit_path.exists():
        raise ValueError(f"one-pass data audit is missing: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    contract_order = {
        "one_pass_data_audit.v1": 1,
        "one_pass_data_audit.v2": 2,
    }
    actual_contract = str(audit.get("audit_contract") or "")
    if actual_contract not in contract_order or minimum_contract not in contract_order:
        raise ValueError(f"unexpected one-pass data audit contract: {audit_path}")
    if contract_order[actual_contract] < contract_order[minimum_contract]:
        raise ValueError(
            f"one-pass data audit {actual_contract} is older than required "
            f"{minimum_contract}: {audit_path}"
        )
    if audit.get("status") != "pass" or audit.get("errors"):
        raise ValueError(f"one-pass data audit did not pass: {audit_path}")
    data_sha256 = sha256_file(data_path)
    if audit.get("data_sha256") != data_sha256:
        raise ValueError(f"one-pass data changed after audit: {data_path}")
    if int(audit.get("n_rows") or 0) <= 0:
        raise ValueError(f"one-pass data audit has no rows: {audit_path}")
    if audit.get("gold_visible_in_prompt") is not False:
        raise ValueError(f"one-pass audit does not prove gold privacy: {audit_path}")
    if audit.get("source_assistant_outputs_in_prompt") is not False:
        raise ValueError(f"one-pass audit allows source assistant output: {audit_path}")
    if audit.get("live_tool_calls_available_to_llm") is not False:
        raise ValueError(f"one-pass audit enables live tools: {audit_path}")
    return {
        **audit,
        "audit_path": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
    }


def one_pass_allowed_values(row: Mapping[str, Any]) -> dict[str, set[str]]:
    values = {str(row["negative_value"]), str(row["positive_value"])}
    return {
        str(field): set(values)
        for field in (
            row["single_prediction_field"],
            row["analog_prediction_field"],
            row["prediction_field"],
        )
    }


def one_pass_analysis_schema_errors(
    content: Mapping[str, Any],
    row: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    for field, required in nested_required_fields(row).items():
        value = content.get(field)
        if not isinstance(value, Mapping):
            errors.append(f"{field}_not_object")
            continue
        missing = [key for key in required if value.get(key) in (None, "")]
        if missing:
            errors.append(f"{field}_missing_fields:{','.join(missing)}")
    return errors


def one_pass_validation_errors(
    content: Mapping[str, Any] | None,
    row: Mapping[str, Any],
) -> list[str]:
    return response_validation_errors(
        {"content": content},
        required_fields=row["required_fields"],
        allowed_values=one_pass_allowed_values(row),
        content_validator=lambda value: one_pass_analysis_schema_errors(value, row),
    )


def one_pass_predictions(
    content: Mapping[str, Any] | None,
    row: Mapping[str, Any],
) -> dict[str, int | None]:
    value = content or {}
    return {
        "single": prediction_to_label(value.get(row["single_prediction_field"]), row),
        "analog": prediction_to_label(value.get(row["analog_prediction_field"]), row),
        "final": prediction_to_label(value.get(row["prediction_field"]), row),
    }


def one_pass_result_path(root: Path, row: Mapping[str, Any]) -> Path:
    return root / "runs" / f"idx{int(row['source_index']):05d}" / "result.json"


def inference_contract_sha256(contract: Mapping[str, Any]) -> str:
    """Bind resumable results to model, decode, validation, and provider settings."""

    return hashlib.sha256(canonical_json_bytes(contract)).hexdigest()
