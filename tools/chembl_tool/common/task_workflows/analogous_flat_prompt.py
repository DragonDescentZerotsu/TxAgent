"""Versioned, tool-free flat analog-evidence prompt shared by paper tasks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tools.chembl_tool.common.assay_transfer_prompt_policy import (
    public_assay_transfer_families,
)
from tools.chembl_tool.common.evidence_contract import evidence_for_llm
from tools.chembl_tool.common.reasoning_calls import call_group_branch
from tools.chembl_tool.common.reasoning_validation import (
    call_with_json_validation,
    structured_response_is_valid,
    validated_branch_content,
)


PROMPT_VERSION = "analogous_flat_v1"
PROMPT_IDENTITY_VIEW = "query_blind_neighbor_smiles"
TEMPLATE_DIR = Path(__file__).with_name("prompt_templates")

BRANCH_OUTPUT_SCHEMA: dict[str, Any] = {
    "evidence_direction": "supports_positive | supports_negative | mixed | insufficient",
    "confidence": "high | moderate | low",
    "reasoning_summary": "string",
    "key_evidence": [
        {
            "molecule_ref": "Molecule N",
            "observation": "string",
            "implication": "string",
        }
    ],
    "conflicting_evidence": ["string"],
    "evidence_gaps": ["string"],
}

_TASKS: dict[str, dict[str, Any]] = {
    "bbb_martins": {
        "name": "meaningful CNS access",
        "positive": "meaningful or adequate CNS access (pass)",
        "negative": "restricted or poor CNS access (fail)",
        "prediction_field": "bbb_prediction",
        "prediction_values": "pass | fail",
        "positive_value": "pass",
        "negative_value": "fail",
    },
    "bioavailability_ma": {
        "name": "absolute oral bioavailability at the F=20% threshold",
        "positive": "F >= 20% (high)",
        "negative": "F < 20% (low)",
        "prediction_field": "bioavailability_prediction",
        "prediction_values": "high | low",
        "positive_value": "high",
        "negative_value": "low",
    },
    "skin_reaction": {
        "name": "skin sensitization/contact allergy",
        "positive": "sensitizer or contact-allergy positive (risk)",
        "negative": "non-sensitizer (no_risk)",
        "prediction_field": "skin_reaction_prediction",
        "prediction_values": "risk | no_risk",
        "positive_value": "risk",
        "negative_value": "no_risk",
    },
}

_OMIT_SOURCE_FIELDS = {
    "confidence",
    "extraction_id",
    "global_identifier",
    "molecule_name",
    "needs_more_context",
    "paragraph_idx",
    "pmid",
    "smiles",
    "source_index",
    "source_record_id",
}

_FIELD_LABELS = {
    "endpoint_name": "Endpoint",
    "measurement_text": "Measurement",
    "unit_text": "Unit",
    "support_text": "Evidence text",
    "extra_details": "Additional scientific context",
}


def task_contract(task_id: str) -> Mapping[str, Any]:
    try:
        return _TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported analogous-flat task: {task_id!r}") from exc


def final_output_schema(task_id: str) -> dict[str, Any]:
    task = task_contract(task_id)
    return {
        str(task["prediction_field"]): str(task["prediction_values"]),
        "confidence": "high | moderate | low",
        "main_reasons": ["string"],
        "conflicting_evidence": ["string"],
        "evidence_gaps": ["string"],
        "final_summary": "string",
    }


def branch_validation() -> dict[str, Any]:
    return {
        "required_fields": (
            "evidence_direction",
            "confidence",
            "reasoning_summary",
            "key_evidence",
            "conflicting_evidence",
            "evidence_gaps",
        ),
        "allowed_values": {
            "evidence_direction": {
                "supports_positive",
                "supports_negative",
                "mixed",
                "insufficient",
            },
            "confidence": {"high", "moderate", "low"},
        },
        "forbidden_field_names": (
            "molecule_chembl_id",
            "global_identifier",
            "pmid",
            "paragraph_idx",
            "tool_summary",
        ),
    }


def final_validation(task_id: str) -> dict[str, Any]:
    task = task_contract(task_id)
    return {
        "required_fields": (str(task["prediction_field"]),),
        "allowed_values": {
            str(task["prediction_field"]): {
                str(task["positive_value"]),
                str(task["negative_value"]),
            }
        },
    }


def build_group_messages(
    group: dict[str, Any],
    *,
    task_id: str,
) -> list[dict[str, str]]:
    if group.get("group_id") != "Flat.all_evidence":
        raise ValueError(
            "analogous_flat_v1 requires the single Flat.all_evidence branch"
        )
    task = task_contract(task_id)
    molecules, retrieval_basis = _molecule_context(group)
    context = {
        "task": task,
        "molecules": molecules,
        "retrieval_basis": retrieval_basis,
        "output_schema": json.dumps(BRANCH_OUTPUT_SCHEMA, indent=2),
    }
    evidence_description = (
        "evidence and assay-transfer likelihoods"
        if retrieval_basis == "assay_transfer"
        else "evidence and Morgan structural similarities"
    )
    return [
        {
            "role": "system",
            "content": (
                f"You are an analog-evidence reasoning model for {task['name']}. "
                f"Use only the supplied {evidence_description}. "
                "The query structure is hidden. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": _env().get_template("analogous_flat_v1.jinja").render(**context),
        },
    ]


def build_final_messages(
    group_outputs: list[dict[str, Any]],
    *,
    task_id: str,
) -> list[dict[str, str]]:
    if len(group_outputs) != 1 or group_outputs[0].get("group_id") != "Flat.all_evidence":
        raise ValueError(
            "analogous_flat_v1 final synthesis requires exactly one Flat.all_evidence analysis"
        )
    task = task_contract(task_id)
    analyses = [
        {
            "status": str(output.get("status") or "error"),
            "content": validated_branch_content(output),
        }
        for output in group_outputs
    ]
    return [
        {
            "role": "system",
            "content": (
                f"You are a senior analog-evidence synthesis model for {task['name']}. "
                "Use only the supplied flat analog analysis. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": _env().get_template("analogous_flat_final_v1.jinja").render(
                task=task,
                analyses=analyses,
                output_schema=json.dumps(final_output_schema(task_id), indent=2),
            ),
        },
    ]


def prompt_provenance(task_id: str) -> dict[str, Any]:
    contract = {
        "prompt_version": PROMPT_VERSION,
        "identity_view": PROMPT_IDENTITY_VIEW,
        "task_id": task_id,
        "group_template_sha256": _sha256(TEMPLATE_DIR / "analogous_flat_v1.jinja"),
        "final_template_sha256": _sha256(
            TEMPLATE_DIR / "analogous_flat_final_v1.jinja"
        ),
        "branch_schema_sha256": _json_sha256(BRANCH_OUTPUT_SCHEMA),
        "final_schema_sha256": _json_sha256(final_output_schema(task_id)),
    }
    contract["contract_sha256"] = _json_sha256(contract)
    return contract


def reason_group(client: Any, group: dict[str, Any], *, task_id: str) -> dict[str, Any]:
    response = call_group_branch(
        client,
        build_group_messages(group, task_id=task_id),
        group=group,
        tools=[],
        **branch_validation(),
    )
    return {
        "group_id": group["group_id"],
        "status": "ok" if structured_response_is_valid(response) else "error",
        "tier": group["tier"],
        "endpoint_group": group["endpoint_group"],
        "n_neighbors": len(group["neighbors"]),
        "llm": response,
    }


def reason_final(
    client: Any,
    group_outputs: list[dict[str, Any]],
    *,
    task_id: str,
) -> dict[str, Any]:
    response = call_with_json_validation(
        client.chat_json,
        build_final_messages(group_outputs, task_id=task_id),
        branch_name="analogous-flat-final",
        **final_validation(task_id),
    )
    return {
        "status": "ok" if structured_response_is_valid(response) else "error",
        "llm": response,
    }


def _molecule_context(
    group: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    neighbors = list(group.get("neighbors") or [])
    retrieval_basis = (
        "assay_transfer"
        if any("transfer_selection_score" in neighbor for neighbor in neighbors)
        else "morgan"
    )
    molecules = []
    for molecule_number, neighbor in enumerate(neighbors, start=1):
        smiles = str(neighbor.get("canonical_smiles") or "")
        if not smiles or smiles in {"[hidden]", "[identity hidden]"}:
            raise ValueError(
                "analogous_flat_v1 requires prompt-visible neighbor canonical SMILES"
            )
        if retrieval_basis == "assay_transfer":
            evidence = _assay_transfer_evidence(neighbor, group)
            morgan_similarity = None
        else:
            evidence = _morgan_evidence(neighbor)
            similarity = neighbor.get("similarity")
            if not isinstance(similarity, (int, float)):
                raise ValueError(
                    "analogous_flat_v1 Morgan neighbors require numeric similarity"
                )
            morgan_similarity = round(float(similarity), 2)
        molecules.append(
            {
                "number": molecule_number,
                "smiles": smiles,
                "morgan_similarity": morgan_similarity,
                "evidence": evidence,
            }
        )
    return molecules, retrieval_basis


def _assay_transfer_evidence(
    neighbor: dict[str, Any], group: dict[str, Any]
) -> list[dict[str, Any]]:
    evidence = []
    for family in public_assay_transfer_families(neighbor, group):
        for record in family["records"]:
            source = record.get("record") or {}
            evidence.append(
                {
                    "mechanism_family": family["group_id"],
                    "rank": record["record_rank"],
                    "assay_transfer_score": record["assay_transfer_score"],
                    "fields": _scientific_fields(source),
                }
            )
    return evidence


def _morgan_evidence(neighbor: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = []
    for row_number, row in enumerate(neighbor.get("evidence_rows") or [], start=1):
        public = evidence_for_llm(row)
        examples = public.get("examples") or []
        if examples:
            for example in examples:
                evidence.append(
                    {
                        "mechanism_family": _minimal_group_id(public),
                        "rank": row_number,
                        "assay_transfer_score": None,
                        "fields": _scientific_fields(example),
                    }
                )
        else:
            evidence.append(
                {
                    "mechanism_family": _minimal_group_id(public),
                    "rank": row_number,
                    "assay_transfer_score": None,
                    "fields": _minimal_scientific_fields(public),
                }
            )
    return evidence


def _minimal_group_id(record: Mapping[str, Any]) -> str:
    group = record.get("group") or {}
    if isinstance(group, Mapping):
        return str(group.get("id") or group.get("endpoint_group") or "all_evidence")
    return "all_evidence"


def _minimal_scientific_fields(record: Mapping[str, Any]) -> list[dict[str, str]]:
    endpoint = record.get("endpoint") or {}
    text = record.get("text") or {}
    fields: dict[str, Any] = {}
    if isinstance(endpoint, Mapping):
        fields["endpoint_name"] = endpoint.get("name")
        measurement = endpoint.get("measurement") or {}
        if isinstance(measurement, Mapping):
            fields["relation"] = measurement.get("relation")
            fields["measurement_text"] = measurement.get("value")
            fields["unit_text"] = measurement.get("unit")
    if isinstance(text, Mapping):
        fields["support_text"] = text.get("evidence")
        fields["context"] = text.get("context")
    return _scientific_fields(fields)


def _scientific_fields(record: dict[str, Any]) -> list[dict[str, str]]:
    source_fields = record.get("source_fields")
    fields = source_fields if isinstance(source_fields, dict) else record
    rows = []
    for key, value in fields.items():
        normalized_key = str(key).lower()
        if normalized_key in _OMIT_SOURCE_FIELDS or value in (None, "", [], {}):
            continue
        label = _FIELD_LABELS.get(normalized_key, normalized_key.replace("_", " ").title())
        rows.append({"label": label, "value": _display(value)})
    return rows


def _display(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "BRANCH_OUTPUT_SCHEMA",
    "PROMPT_IDENTITY_VIEW",
    "PROMPT_VERSION",
    "branch_validation",
    "build_final_messages",
    "build_group_messages",
    "final_output_schema",
    "final_validation",
    "prompt_provenance",
    "reason_final",
    "reason_group",
]
