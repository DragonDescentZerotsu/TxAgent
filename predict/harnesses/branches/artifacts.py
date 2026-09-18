"""Read branch inputs and write the historical branch trace artifact.

The three task pipelines use this module to stream one requested JSONL row and
to preserve their existing single/group/final trace format. It does not select
evidence, construct prompts, or call a model.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
from typing import Any

from predict.llm_io.evidence import evidence_for_group_llm
from predict.retrieval.policies import (
    NEIGHBOR_SELECTORS,
    SIMILARITY_SELECTOR,
    selector_metadata,
)


def read_jsonl_record(path: Path, index: int) -> dict[str, Any]:
    """Read one zero-based JSONL record without materializing the full file."""
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise SystemExit(f"No record at index {index}: {path}")


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


def _validated_selector_metadata(metadata: Any, *, location: str) -> dict[str, Any]:
    if (
        not isinstance(metadata, dict)
        or metadata.get("name") not in NEIGHBOR_SELECTORS
        or not isinstance(metadata.get("version"), str)
        or not metadata["version"]
    ):
        raise ValueError(
            "Retrieval replay neighbor-selector provenance is malformed at "
            f"{location}: {metadata!r}"
        )
    return metadata


def _selector_from_retrieval_payload(retrieval: dict[str, Any]) -> dict[str, Any]:
    observed: list[tuple[str, dict[str, Any]]] = []
    for section_name in ("retrieval_policy", "experiment"):
        section = retrieval.get(section_name)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ValueError(
                "Retrieval replay selector-provenance section is malformed: "
                f"{section_name}={section!r}"
            )
        metadata = section.get("neighbor_selector")
        if metadata is not None:
            observed.append(
                (
                    f"{section_name}.neighbor_selector",
                    _validated_selector_metadata(metadata, location=section_name),
                )
            )
    if not observed:
        return selector_metadata(SIMILARITY_SELECTOR)
    if len(observed) == 2 and observed[0][1] != observed[1][1]:
        raise ValueError(
            "Retrieval replay neighbor-selector provenance conflicts between "
            f"{observed[0][0]}={observed[0][1]!r} and "
            f"{observed[1][0]}={observed[1][1]!r}"
        )
    return observed[0][1]


def load_retrieval_replay(
    source_run_dir: str,
    query_smiles: str,
    *,
    expected_neighbor_selector: str,
    expected_reranker_provenance: dict[str, Any] | None = None,
    expected_assay_transfer_selection_policy: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Load frozen retrieval after validating its query and selection provenance."""
    if not source_run_dir:
        return None
    source_path = Path(source_run_dir) / "retrieval.json"
    if not source_path.exists():
        raise FileNotFoundError(f"Retrieval replay artifact does not exist: {source_path}")
    retrieval = json.loads(source_path.read_text(encoding="utf-8"))
    if retrieval.get("status") != "ok":
        raise ValueError(f"Retrieval replay artifact is not successful: {source_path}")
    source_smiles = str((retrieval.get("query") or {}).get("input_smiles") or "")
    if source_smiles and source_smiles != query_smiles:
        raise ValueError(
            "Retrieval replay query mismatch: "
            f"expected {query_smiles!r}, found {source_smiles!r} in {source_path}"
        )
    expected_selector = selector_metadata(expected_neighbor_selector)
    observed_selector = _selector_from_retrieval_payload(retrieval)
    mismatches = {
        key: {"expected": expected_selector.get(key), "observed": observed_selector.get(key)}
        for key in ("name", "version")
        if expected_selector.get(key) != observed_selector.get(key)
    }
    if mismatches:
        raise ValueError(f"Retrieval replay neighbor-selector provenance mismatch: {mismatches}")
    if expected_reranker_provenance is not None:
        observed = (retrieval.get("experiment") or {}).get("retrieval_reranker") or {"name": "none"}
        expected = expected_reranker_provenance
        keys = (
            "name", "model", "model_revision", "scoring_contract_version",
            "template_hash", "catalog_version",
        )
        mismatches = {
            key: {"expected": expected.get(key), "observed": observed.get(key)}
            for key in keys
            if expected.get(key) != observed.get(key)
        }
        if mismatches:
            raise ValueError(f"Retrieval replay reranker provenance mismatch: {mismatches}")
    if expected_assay_transfer_selection_policy is not None:
        observed_policy = (retrieval.get("experiment") or {}).get(
            "assay_transfer_selection_policy"
        ) or {}
        observed_diversity = observed_policy.get("diversity") or {}
        expected_diversity = expected_assay_transfer_selection_policy.get("diversity") or {}
        mismatches = {
            key: {"expected": expected_diversity.get(key), "observed": observed_diversity.get(key)}
            for key in ("version", "mode", "score_slack")
            if expected_diversity.get(key) != observed_diversity.get(key)
        }
        expected_score = expected_assay_transfer_selection_policy.get("min_score")
        if expected_score != observed_policy.get("min_score"):
            mismatches["min_score"] = {
                "expected": expected_score,
                "observed": observed_policy.get("min_score"),
            }
        if mismatches:
            raise ValueError(
                "Retrieval replay assay-transfer selection-policy mismatch: "
                f"{mismatches}"
            )
    return retrieval


def retrieval_prompt_contract(retrieval: dict[str, Any]) -> dict[str, Any]:
    """Project retrieval output to the fields visible to branch prompts."""
    query = retrieval.get("query") or {}
    score_policy = (retrieval.get("experiment") or {}).get("llm_neighbor_score_policy") or {}
    score_visible = score_policy.get("name") == "assay_transfer_scored_neighbors.v1"
    contract = {
        "query": {
            "input_smiles": query.get("input_smiles", ""),
            "canonical_smiles": query.get("canonical_smiles", ""),
            "external_condition": query.get("external_condition", ""),
        },
        "groups": [
            {
                "group_id": group.get("group_id", ""),
                "tier": group.get("tier", ""),
                "endpoint_group": group.get("endpoint_group", ""),
                **({"llm_neighbor_score_policy": score_policy} if score_visible else {}),
                "neighbors": [
                    _prompt_neighbor_contract(neighbor, group=group, score_visible=score_visible)
                    for neighbor in group.get("neighbors") or []
                ],
            }
            for group in retrieval.get("groups") or []
        ],
    }
    if score_visible:
        contract["llm_neighbor_score_policy"] = score_policy
    return contract


def _prompt_neighbor_contract(
    neighbor: dict[str, Any], *, group: dict[str, Any], score_visible: bool
) -> dict[str, Any]:
    payload = {
        "rank": neighbor.get("rank"),
        "molecule_chembl_id": neighbor.get("molecule_chembl_id", ""),
        "canonical_smiles": neighbor.get("canonical_smiles", ""),
        "similarity": neighbor.get("similarity"),
        "similarity_bucket": neighbor.get("similarity_bucket", ""),
        "evidence_rows": [
            evidence_for_group_llm(row, group)
            for row in neighbor.get("evidence_rows") or []
        ],
    }
    if score_visible:
        payload["assay_transfer_score"] = round(float(neighbor["transfer_selection_score"]), 2)
        selected = neighbor.get("transfer_selected_records") or [
            {
                "transfer_selection_score": neighbor["transfer_selection_score"],
                "transfer_winning_record": neighbor.get("transfer_winning_record") or {},
            }
        ]
        payload["assay_transfer_records"] = [
            {
                "assay_transfer_score": round(float(record["transfer_selection_score"]), 2),
                "source_contract": (record.get("transfer_winning_record") or {}).get("source_contract"),
                "source_fields": (record.get("transfer_winning_record") or {}).get("source_fields"),
            }
            for record in selected
        ]
    return payload


def retrieval_prompt_hash(retrieval: dict[str, Any]) -> str:
    serialized = json.dumps(
        retrieval_prompt_contract(retrieval),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def changed_group_ids(baseline: dict[str, Any], target: dict[str, Any]) -> list[str]:
    baseline_groups = _group_contracts(baseline)
    target_groups = _group_contracts(target)
    return sorted(
        group_id
        for group_id in set(baseline_groups) | set(target_groups)
        if baseline_groups.get(group_id) != target_groups.get(group_id)
    )


def materialize_reused_run(source_run: Path, target_run: Path, provenance: dict[str, Any]) -> None:
    """Hard-link an immutable completed run and add target-local provenance."""
    if not target_run.exists():
        target_run.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(source_run, target_run, copy_function=os.link)
        except OSError:
            if target_run.exists():
                shutil.rmtree(target_run)
            shutil.copytree(source_run, target_run)
    (target_run / "reuse.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_target_manifest(source_run, target_run, provenance)


def load_reusable_group_outputs(
    source_run_dir: str,
    target_retrieval: dict[str, Any],
    *,
    target_neighbor_context_profile: str = "standard",
) -> list[dict[str, Any]]:
    """Reuse successful branches whose model-visible retrieval is unchanged."""
    if not source_run_dir:
        return []
    source_dir = Path(source_run_dir)
    baseline_path = source_dir / "retrieval.json"
    outputs_path = source_dir / "group_reasoning_outputs.jsonl"
    if not baseline_path.exists() or not outputs_path.exists():
        raise FileNotFoundError(f"Missing reusable group artifacts in {source_dir}")
    manifest_path = source_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    if str(manifest.get("neighbor_context_profile") or "standard") != target_neighbor_context_profile:
        return []
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    reusable = {
        str(group.get("group_id") or "")
        for group in target_retrieval.get("groups") or []
        if group.get("neighbors")
    } - set(changed_group_ids(baseline, target_retrieval))
    outputs = []
    with outputs_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            output = json.loads(line)
            if str(output.get("group_id") or "") not in reusable or output.get("status") != "ok":
                continue
            output["reused_from"] = str(outputs_path)
            output["reuse_reason"] = "identical_llm_visible_group_input"
            outputs.append(output)
    return outputs


def _group_contracts(retrieval: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contract = retrieval_prompt_contract(retrieval)
    return {str(group.get("group_id") or ""): group for group in contract["groups"]}


def _write_target_manifest(source_run: Path, target_run: Path, provenance: dict[str, Any]) -> None:
    source_manifest = source_run / "manifest.json"
    if not source_manifest.exists():
        return
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    if provenance.get("neighbor_identity_policy"):
        payload["neighbor_identity_policy"] = provenance["neighbor_identity_policy"]
    payload["artifact_reuse"] = {
        "reused_from": provenance.get("reused_from"),
        "reuse_reason": provenance.get("reuse_reason"),
        "baseline_prompt_hash": provenance.get("baseline_prompt_hash"),
        "target_prompt_hash": provenance.get("target_prompt_hash"),
    }
    target_manifest = target_run / "manifest.json"
    target_manifest.unlink(missing_ok=True)
    target_manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
