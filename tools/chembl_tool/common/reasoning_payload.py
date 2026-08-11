"""Source-independent prompt payload and pipeline input helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def llm_query_payload(query: dict[str, Any]) -> dict[str, Any]:
    """Expose only the query fields allowed by the active identity contract."""
    if query.get("identity_hidden"):
        return {
            "molecule_id": "query",
            "identity_hidden": True,
            "prefetched_molecule_properties": query.get(
                "prefetched_molecule_properties"
            )
            or {},
        }
    payload = {
        "input_smiles": query.get("input_smiles", ""),
        "canonical_smiles": query.get("canonical_smiles", ""),
    }
    if query.get("prefetched_molecule_properties"):
        payload["tools_prefetched"] = True
        payload["prefetched_molecule_properties"] = query[
            "prefetched_molecule_properties"
        ]
    return payload


def clean_exact_match(match: dict[str, Any]) -> dict[str, Any]:
    """Keep the frozen molecule metadata surface for exact ChEMBL matches."""
    fields = (
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_inchi_key",
        "mw_freebase",
        "alogp",
        "hba",
        "hbd",
        "psa",
        "rtb",
        "num_ro5_violations",
        "full_mwt",
        "aromatic_rings",
        "heavy_atoms",
        "qed_weighted",
        "full_molformula",
        "np_likeness_score",
    )
    return {field: match.get(field, "") for field in fields}


def clean_shared_assay_context(context: dict[str, Any]) -> dict[str, Any]:
    """Keep paired activity values without leaking source-internal fields."""
    return {
        "same_endpoint_activity": [
            _clean_shared_activity_card(card)
            for card in context.get("same_endpoint_activity", [])
        ],
        "same_assay_different_endpoint_activity": [
            _clean_shared_activity_card(card)
            for card in context.get("same_assay_different_endpoint_activity", [])
        ],
        "n_same_endpoint_activity": context.get("n_same_endpoint_activity", 0),
        "n_same_assay_different_endpoint_activity": context.get(
            "n_same_assay_different_endpoint_activity", 0
        ),
    }


def _clean_shared_activity_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "assay_chembl_id": card.get("assay_chembl_id", ""),
        "standard_type_match": card.get("standard_type_match", False),
        "units_match": card.get("units_match", False),
        "query_activity": _clean_activity_value(card.get("query_activity") or {}),
        "neighbor_activity": _clean_activity_value(card.get("neighbor_activity") or {}),
    }


def _clean_activity_value(activity: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "assay_chembl_id",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "activity_comment",
        "data_validity_comment",
        "standard_text_value",
        "action_type",
    )
    return {field: activity.get(field, "") for field in fields}


def read_jsonl_record(path: Path, index: int) -> dict[str, Any]:
    """Read one zero-based JSONL record without materializing the full file."""
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise SystemExit(f"No record at index {index}: {path}")


def load_env_file(path: Path) -> None:
    """Load the explicit run configuration file using the historical parser."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        # The explicit --env-file is the run configuration source of truth.
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def write_trace_jsonl(
    path: Path,
    *,
    prediction_field: str,
    query_record: dict[str, Any],
    query_index: int,
    smiles: str,
    single_output: dict[str, Any],
    group_outputs: list[dict[str, Any]],
    final_output: dict[str, Any],
) -> None:
    """Write the frozen single/group/final trace contract for one query."""
    outputs = [
        ("single_molecule", single_output),
        *(
            (str(group.get("group_id") or "unknown_group"), group)
            for group in group_outputs
        ),
        ("final_summary", final_output),
    ]
    with path.open("w", encoding="utf-8") as handle:
        for task, output in outputs:
            handle.write(
                json.dumps(
                    _trace_record(
                        task,
                        output,
                        prediction_field=prediction_field,
                        query_record=query_record,
                        query_index=query_index,
                        smiles=smiles,
                    ),
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )


def _trace_record(
    task: str,
    output: dict[str, Any],
    *,
    prediction_field: str,
    query_record: dict[str, Any],
    query_index: int,
    smiles: str,
) -> dict[str, Any]:
    llm = output.get("llm") or {}
    content = llm.get("content")
    return {
        "task": task,
        "index": query_index,
        "sample_id": query_index,
        "molecule_key": f"index:{query_index}",
        "smiles": smiles,
        "label": query_record.get("Y"),
        "status": output.get("status"),
        "prediction": content.get(prediction_field)
        if isinstance(content, dict)
        else None,
        "response_text": (
            json.dumps(content, ensure_ascii=False, indent=2)
            if content is not None
            else output.get("error")
        ),
        "messages": llm.get("messages") or [],
        "tool_count": len(llm.get("tool_calls") or []),
        "usage": llm.get("usage") or {},
        "raw_output": {key: value for key, value in output.items() if key != "llm"},
    }
