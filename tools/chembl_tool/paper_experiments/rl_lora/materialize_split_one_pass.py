"""Canonical RL one-pass materializer from frozen retrieval and tool contracts.

Unlike ``materialize_one_pass``, this path does not need three prior LLM calls.
It retrieves the same full-flat evidence, performs the same harness-side fixed
tool bundle, and substitutes those label-private inputs into a frozen one-pass
prompt template from the corresponding task.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import threading
from typing import Any

from tools.chembl_tool.common.identity_blind import prepare_reasoning_retrieval
from tools.chembl_tool.common.experiment_retrieval import retrieve_experiment_view
from tools.chembl_tool.common.json_utils import (
    canonical_json_bytes,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.common.openai_reasoning_client import ToolServiceClient
from tools.chembl_tool.common.task_workflows.retrieve_neighbors import load_index

from .mixed_retrieval import (
    MIXED_IDENTITY_POLICY_VERSION,
    retrieve_full_flat_mixed_identity,
)
from .one_pass_contract import (
    DEPLOYMENT_VISIBLE_PREFETCHED,
    IDENTITY_BLIND,
    TASK_CONTRACTS,
    VISIBILITY_MODES,
    contract_version_for_visibility,
    is_one_pass_contract,
)


TASK_SPECS = {
    "bbb_martins": {
        "config": "tools.chembl_tool.tasks.bbb_martins.experiment_config",
        "pipeline": "tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline",
        "prompt_profile": "meaningful_cns_access_v1",
    },
    "bioavailability_ma": {
        "config": "tools.chembl_tool.tasks.bioavailability_ma.experiment_config",
        "pipeline": "tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_pipeline",
        "prompt_profile": "f20_evidence_calibrated_v2",
    },
    "skin_reaction": {
        "config": "tools.chembl_tool.tasks.skin_reaction.experiment_config",
        "pipeline": "tools.chembl_tool.tasks.skin_reaction.run_reasoning_pipeline",
        "prompt_profile": "sensitization_aligned_v2",
    },
}
MATERIALIZER_VERSION = "one_pass_direct_rl_materializer.v2"


def _llm_query_payload(query: dict[str, Any]) -> dict[str, Any]:
    """Expose only query fields allowed by the active one-pass visibility contract."""
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_template(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError(f"empty one-pass template: {path}")
    template = rows[0]
    if not is_one_pass_contract(template.get("contract_version")):
        raise ValueError(
            f"unexpected template contract: {template.get('contract_version')}"
        )
    return template


def _apply_visibility_contract(row: dict[str, Any], visibility_mode: str) -> None:
    contract_version = contract_version_for_visibility(visibility_mode)
    payload = json.loads(row["messages"][1]["content"])
    payload["contract_version"] = contract_version
    output_instructions = [
        instruction
        for instruction in payload.get("output_instructions") or []
        if "anonymous query or neighbors" not in str(instruction)
    ]
    if visibility_mode == DEPLOYMENT_VISIBLE_PREFETCHED:
        output_instructions.append(
            "Molecular structures and allowed source identifiers are visible; use them as "
            "chemical evidence but do not infer identities that are not supplied."
        )
        row["messages"][0]["content"] = (
            "You are a senior medicinal-chemistry evidence reasoning model. Molecular "
            "structures and allowed source identifiers are visible, while retrieval and all "
            "tool comparisons are fixed by the harness. Perform the single-molecule prior, "
            "full-flat analog transferability analysis, and final binary adjudication in one "
            "response. Keep the three judgments explicit and return only valid JSON."
        )
    else:
        output_instructions.append(
            "Do not identify or name the anonymous query or neighbors."
        )
    payload["output_instructions"] = output_instructions
    row["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False)
    row["contract_version"] = contract_version
    row["visibility_mode"] = visibility_mode


def _tool_receipt(payload: dict[str, Any]) -> tuple[int, int, int]:
    results = [payload["query"]["prefetched_molecule_properties"]]
    for neighbor in payload.get("neighbors") or []:
        results.extend(neighbor.get("prefetched_comparisons") or [])
    return (
        len(results),
        sum(result.get("status") == "ok" for result in results),
        sum(result.get("status") == "error" for result in results),
    )


def _materialize_row(
    source_row: dict[str, Any],
    source_index: int,
    *,
    task: str,
    subset: str,
    index: dict[str, Any],
    source_config: Any,
    pipeline: Any,
    template: dict[str, Any],
    tool_service: ToolServiceClient,
    top_k_per_group: int,
    min_similarity: float,
    identity_policy: str,
    visibility_mode: str,
) -> dict[str, Any]:
    query_smiles = str(source_row["drug"])
    gold = int(source_row["Y"])
    if gold not in {0, 1}:
        raise ValueError(f"non-binary label at index {source_index}: {gold}")
    if identity_policy == "parent_disjoint":
        retrieval = retrieve_experiment_view(
            query_smiles,
            index,
            mode="full_flat",
            config=source_config,
            top_k_per_group=top_k_per_group,
            min_similarity=min_similarity,
            neighbor_identity_policy="parent_disjoint",
        )
        retrieval_policy = "parent_disjoint"
    else:
        retrieval = retrieve_full_flat_mixed_identity(
            query_smiles,
            index,
            config=source_config,
            top_k_per_group=top_k_per_group,
            min_similarity=min_similarity,
        )
        retrieval_policy = MIXED_IDENTITY_POLICY_VERSION
    if retrieval.get("status") != "ok":
        raise ValueError(
            f"retrieval failed at index {source_index}: {retrieval.get('errors')}"
        )
    reasoning_retrieval = prepare_reasoning_retrieval(
        retrieval,
        tool_service,
        identity_blind=visibility_mode == IDENTITY_BLIND,
        harness_prefetch_tools=True,
    )
    groups = reasoning_retrieval.get("groups") or []
    if len(groups) != 1:
        raise ValueError(
            f"expected one full-flat group at index {source_index}, got {len(groups)}"
        )
    query = _llm_query_payload(reasoning_retrieval["query"])
    # Only the normalized evidence surface is consumed here. The one-pass
    # template owns the frozen prompt profile and output instructions.
    group_payload = pipeline.build_group_prompt_payload(query, groups[0])

    row = deepcopy(template)
    _apply_visibility_contract(row, visibility_mode)
    payload = json.loads(row["messages"][1]["content"])
    payload["query"] = query
    payload["retrieval_coverage"] = reasoning_retrieval.get("coverage") or {}
    payload["analog_group"] = group_payload.get("group") or {}
    payload["neighbors"] = group_payload.get("neighbors") or []
    row["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False)
    neighbor_ids = [
        str(neighbor.get("molecule_chembl_id") or "")
        for neighbor in payload["neighbors"]
    ]
    if visibility_mode == IDENTITY_BLIND:
        if any(not identifier.startswith("neighbor_") for identifier in neighbor_ids):
            raise ValueError(f"non-anonymous neighbor at index {source_index}")
    elif any(
        not str(neighbor.get("canonical_smiles") or "")
        for neighbor in payload["neighbors"]
    ):
        raise ValueError(f"visible neighbor structure missing at index {source_index}")
    tool_count, successful_tool_count, failed_tool_count = _tool_receipt(payload)
    if successful_tool_count + failed_tool_count != tool_count:
        raise ValueError(
            f"prefetched tool receipt missing at index {source_index}: "
            f"ok={successful_tool_count} error={failed_tool_count} total={tool_count}"
        )
    row.update(
        {
            "gold_label": gold,
            "source_task": task,
            "source_subset": subset,
            "source_index": source_index,
            "source_trace": "",
            "source_trace_sha256": "",
            "source_neighbor_ids": neighbor_ids,
            "source_group_available": bool(neighbor_ids),
            "prompt_sha256": hashlib.sha256(
                canonical_json_bytes(row["messages"])
            ).hexdigest(),
            "contract_version": contract_version_for_visibility(visibility_mode),
            "gold_location": "environment_private_metadata_only",
            "retrieval_identity_policy": retrieval_policy,
            "prefetched_tool_count": tool_count,
            "successful_prefetched_tool_count": successful_tool_count,
            "failed_prefetched_tool_count": failed_tool_count,
            "tool_calls_available_to_llm": False,
            "materializer_version": MATERIALIZER_VERSION,
            "materialization_source": "direct_retrieval_tool_prefetch",
            "formal_rl_contract_eligible": identity_policy == "parent_disjoint",
        }
    )
    return row


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = _read_jsonl(args.input_jsonl)
    source_total = len(source_rows)
    if args.limit is not None:
        source_rows = source_rows[: args.limit]
    template = _load_template(args.template)
    if template.get("source_task") != args.task:
        raise ValueError("template task does not match --task")
    spec = TASK_SPECS[args.task]
    config_module = importlib.import_module(spec["config"])
    pipeline = importlib.import_module(spec["pipeline"])
    source_config = config_module.get_source_config("starling")
    index = load_index(args.index)
    tool_service = ToolServiceClient(args.tool_service_url, timeout_s=args.timeout_s)
    work_root = args.work_root or args.output.parent / f".{args.output.stem}_rows"
    work_root.mkdir(parents=True, exist_ok=True)

    pending = [
        (index_value, source_row)
        for index_value, source_row in enumerate(source_rows)
        if not (work_root / f"idx{index_value:05d}.json").exists()
    ]
    print(
        f"one-pass materialization total={len(source_rows)} "
        f"existing={len(source_rows) - len(pending)} pending={len(pending)}",
        flush=True,
    )
    completed = 0
    lock = threading.Lock()

    def run_one(item: tuple[int, dict[str, Any]]) -> None:
        nonlocal completed
        source_index, source_row = item
        row = _materialize_row(
            source_row,
            source_index,
            task=args.task,
            subset=args.subset,
            index=index,
            source_config=source_config,
            pipeline=pipeline,
            template=template,
            tool_service=tool_service,
            top_k_per_group=args.top_k_per_group,
            min_similarity=args.min_similarity,
            identity_policy=args.identity_policy,
            visibility_mode=args.visibility_mode,
        )
        write_json_atomic(work_root / f"idx{source_index:05d}.json", row)
        with lock:
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == len(pending):
                print(
                    f"completed={completed}/{len(pending)} idx={source_index}",
                    flush=True,
                )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.parallelism
    ) as executor:
        futures = [executor.submit(run_one, item) for item in pending]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    rows = [
        json.loads(
            (work_root / f"idx{source_index:05d}.json").read_text(encoding="utf-8")
        )
        for source_index in range(len(source_rows))
    ]
    write_jsonl_atomic(args.output, rows)
    label_counts = Counter(int(row["gold_label"]) for row in rows)
    manifest = {
        "materializer_version": MATERIALIZER_VERSION,
        "contract_version": contract_version_for_visibility(args.visibility_mode),
        "task": args.task,
        "subset": args.subset,
        "formal_eligible": args.limit is None
        and args.identity_policy == "parent_disjoint",
        "formal_eligibility": {
            "full_source_rows": args.limit is None,
            "parent_disjoint": args.identity_policy == "parent_disjoint",
            "requires_adjacent_v2_audit": True,
        },
        "n_rows": len(rows),
        "source_total_rows": source_total,
        "label_counts": {
            str(key): value for key, value in sorted(label_counts.items())
        },
        "input_jsonl": str(args.input_jsonl),
        "input_sha256": sha256_file(args.input_jsonl),
        "index": str(args.index),
        "template": str(args.template),
        "template_sha256": sha256_file(args.template),
        "prompt_profile": spec["prompt_profile"],
        "identity_policy": args.identity_policy,
        "visibility_mode": args.visibility_mode,
        "tool_execution": "harness_prefetched_before_prompt",
        "tool_calls_available_to_llm": False,
        "gold_location": "environment_private_metadata_only",
        "top_k_per_group": args.top_k_per_group,
        "min_similarity": args.min_similarity,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    write_json_atomic(args.output.with_suffix(".manifest.json"), manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASK_CONTRACTS), required=True)
    parser.add_argument("--subset", choices=("train", "valid", "test"), required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--tool-service-url", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--parallelism", type=int, default=32)
    parser.add_argument("--top-k-per-group", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.3)
    parser.add_argument(
        "--identity-policy",
        choices=("parent_disjoint", "mixed_family_identity_policy.v1"),
        default="parent_disjoint",
    )
    parser.add_argument(
        "--visibility-mode",
        choices=sorted(VISIBILITY_MODES),
        default=IDENTITY_BLIND,
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    manifest = materialize(parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
