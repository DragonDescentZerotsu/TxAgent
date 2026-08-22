"""Deterministic retrieval-input comparison and artifact reuse helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from tools.chembl_tool.common.evidence_contract import evidence_for_group_llm


def retrieval_prompt_contract(retrieval: dict[str, Any]) -> dict[str, Any]:
    """Project retrieval output to source-independent fields visible to group prompts."""
    query = retrieval.get("query") or {}
    return {
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
                "neighbors": [
                    {
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
                    for neighbor in group.get("neighbors") or []
                ],
            }
            for group in retrieval.get("groups") or []
        ],
    }


def retrieval_prompt_hash(retrieval: dict[str, Any]) -> str:
    payload = retrieval_prompt_contract(retrieval)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
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
    """Hard-link an immutable completed run and add target-local reuse provenance."""
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
    """Reuse only independent group branches whose LLM-visible retrieval input is unchanged."""
    if not source_run_dir:
        return []
    source_dir = Path(source_run_dir)
    baseline_path = source_dir / "retrieval.json"
    outputs_path = source_dir / "group_reasoning_outputs.jsonl"
    if not baseline_path.exists() or not outputs_path.exists():
        raise FileNotFoundError(f"Missing reusable group artifacts in {source_dir}")
    source_manifest_path = source_dir / "manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.exists()
        else {}
    )
    source_profile = str(source_manifest.get("neighbor_context_profile") or "standard")
    if source_profile != target_neighbor_context_profile:
        return []
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    changed = set(changed_group_ids(baseline, target_retrieval))
    target_group_ids = {
        str(group.get("group_id") or "")
        for group in target_retrieval.get("groups") or []
        if group.get("neighbors")
    }
    reusable = target_group_ids - changed
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
    policy = provenance.get("neighbor_identity_policy")
    if policy:
        payload["neighbor_identity_policy"] = policy
    payload["artifact_reuse"] = {
        "reused_from": provenance.get("reused_from"),
        "reuse_reason": provenance.get("reuse_reason"),
        "baseline_prompt_hash": provenance.get("baseline_prompt_hash"),
        "target_prompt_hash": provenance.get("target_prompt_hash"),
    }
    target_manifest = target_run / "manifest.json"
    if target_manifest.exists():
        target_manifest.unlink()
    target_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
