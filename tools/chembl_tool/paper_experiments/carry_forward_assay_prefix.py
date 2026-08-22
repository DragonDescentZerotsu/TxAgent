"""Carry forward completed assay-prefix predictions when group evidence is unchanged."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
from typing import Any

from tools.chembl_tool.common.json_utils import atomic_output_path, write_json_atomic
from tools.chembl_tool.common.retrieval_ablation import retrieval_prompt_hash


RUN_INDEX = re.compile(r"_idx(\d+)$")
BRANCH_FILES = (
    "single_molecule_reasoning_output.json",
    "group_reasoning_outputs.jsonl",
    "group_reasoning_outputs_raw.jsonl",
    "trace_messages.jsonl",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_atomic(source: Path, target: Path) -> None:
    if not source.is_file():
        target.unlink(missing_ok=True)
        return
    with atomic_output_path(target) as temporary:
        shutil.copyfile(source, temporary)


def _valid_source_run(source: Path, retrieval: dict[str, Any]) -> bool:
    single = source / "single_molecule_reasoning_output.json"
    final = source / "final_reasoning_output.json"
    groups = source / "group_reasoning_outputs.jsonl"
    if not single.is_file() or not final.is_file() or not groups.is_file():
        return False
    if _read_json(single).get("status") != "ok" or _read_json(final).get("status") != "ok":
        return False
    outputs = [
        json.loads(line)
        for line in groups.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = sorted(
        str(group.get("group_id") or "")
        for group in retrieval.get("groups") or []
        if group.get("neighbors")
    )
    observed = sorted(
        str(output.get("group_id") or "")
        for output in outputs
        if output.get("status") == "ok"
    )
    return observed == expected


def _target_has_divergent_reasoning(source: Path, target: Path) -> bool:
    """Do not overwrite reasoning already produced for the target condition."""
    for name in BRANCH_FILES:
        target_path = target / name
        if not target_path.is_file():
            continue
        source_path = source / name
        if not source_path.is_file() or target_path.read_bytes() != source_path.read_bytes():
            return True
    return False


def carry_forward_run(source: Path, target: Path) -> bool:
    """Materialize one target run under the frozen evidence-equivalence policy."""
    source_retrieval_path = source / "retrieval.json"
    target_retrieval_path = target / "retrieval.json"
    target_manifest_path = target / "manifest.json"
    if not source_retrieval_path.is_file() or not target_retrieval_path.is_file():
        return False
    if not target_manifest_path.is_file():
        return False
    target_final_path = target / "final_reasoning_output.json"
    if target_final_path.is_file() and _read_json(target_final_path).get("status") == "ok":
        return False

    source_retrieval = _read_json(source_retrieval_path)
    target_retrieval = _read_json(target_retrieval_path)
    source_hash = retrieval_prompt_hash(source_retrieval)
    target_hash = retrieval_prompt_hash(target_retrieval)
    if (
        source_hash != target_hash
        or not _valid_source_run(source, source_retrieval)
        or _target_has_divergent_reasoning(source, target)
    ):
        return False

    for name in BRANCH_FILES:
        _copy_atomic(source / name, target / name)
    final_output = deepcopy(_read_json(source / "final_reasoning_output.json"))
    source_coverage = source_retrieval.get("coverage") or {}
    target_coverage = target_retrieval.get("coverage") or {}
    changed_coverage_fields = sorted(
        key
        for key in set(source_coverage) | set(target_coverage)
        if source_coverage.get(key) != target_coverage.get(key)
    )
    provenance = {
        "reused_from": str(source),
        "reuse_reason": "identical_llm_visible_assay_evidence_carry_forward",
        "reuse_contract": "query_and_group_evidence_equal_coverage_metadata_may_differ.v1",
        "source_prompt_hash": source_hash,
        "target_prompt_hash": target_hash,
        "source_assay_prefix": (source_retrieval.get("experiment") or {}).get("assay_prefix"),
        "target_assay_prefix": (target_retrieval.get("experiment") or {}).get("assay_prefix"),
        "changed_final_coverage_fields_ignored_by_policy": changed_coverage_fields,
        "final_prompt_byte_identical": not changed_coverage_fields,
    }
    final_output["artifact_reuse"] = provenance
    write_json_atomic(target_final_path, final_output)
    write_json_atomic(target / "reuse.json", provenance)

    manifest = _read_json(target_manifest_path)
    manifest["artifact_reuse"] = provenance
    stage_pool = manifest.setdefault("stage_pool", {})
    stage_pool.setdefault("events", []).append(
        {"stage": "final", "status": "ok", "reuse_reason": provenance["reuse_reason"]}
    )
    write_json_atomic(target_manifest_path, manifest)
    return True


def carry_forward_batch(source_batch: Path, target_batch: Path) -> dict[str, Any]:
    if not source_batch.is_dir() or not target_batch.is_dir():
        raise FileNotFoundError("source and target batch directories must already exist")
    reused: list[int] = []
    for target in sorted((target_batch / "runs").glob("*_idx*")):
        match = RUN_INDEX.search(target.name)
        if match is None:
            continue
        index = int(match.group(1))
        source = source_batch / "runs" / f"{source_batch.name}_idx{index:05d}"
        if carry_forward_run(source, target):
            reused.append(index)
    receipt = {
        "source_batch": str(source_batch),
        "target_batch": str(target_batch),
        "n_reused": len(reused),
        "query_indices": reused,
    }
    write_json_atomic(target_batch / "carry_forward_receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--target-batch", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(carry_forward_batch(args.source_batch, args.target_batch), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
