"""Measure full-catalog conditioned assay prompts without calling an LLM."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import pickle
from statistics import median
from typing import Any

from tokenizers import Tokenizer

from tools.chembl_tool.common import evidence_contract
from tools.chembl_tool.common.assay_retrieval import retrieve_assay_prefix
from tools.chembl_tool.common.json_utils import (
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.common.reasoning_calls import bound_group_prompt_payload
from tools.chembl_tool.common.reasoning_payload import external_condition_sentence
from data.processing.gold_labels.conditioned_benchmark import split_path


@dataclass(frozen=True)
class TaskSpec:
    split: Path
    summary_index: Path
    raw_index: Path
    pipeline_module: str


ROOT = Path("outputs/paper/starling_conditioned_assay_family_curve_v1")
SPECS = {
    "bbb_martins": TaskSpec(
        split_path("bbb_martins", "valid"),
        ROOT / "indices/bbb_martins/compact_v2/assay_neighbor_index.pkl",
        ROOT / "indices/bbb_martins/raw_v3/assay_neighbor_index.pkl",
        "tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline",
    ),
    "bioavailability_ma": TaskSpec(
        split_path("bioavailability_ma", "valid"),
        ROOT / "indices/bioavailability_ma/compact_v2/assay_neighbor_index.pkl",
        ROOT
        / "indices/bioavailability_ma/raw_v3_nondirect_context_v1/assay_neighbor_index.pkl",
        "tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_pipeline",
    ),
    "skin_reaction": TaskSpec(
        split_path("skin_reaction", "valid"),
        ROOT / "indices/skin_reaction/compact_v2/assay_neighbor_index.pkl",
        ROOT / "indices/skin_reaction/raw_v3/assay_neighbor_index.pkl",
        "tools.chembl_tool.tasks.skin_reaction.run_reasoning_pipeline",
    ),
}


def _system_message(task: str, module: Any) -> str:
    if task == "bbb_martins":
        return (
            "You are a medicinal chemistry BBB analog evidence analyst. Reason about whether "
            "analog evidence in one endpoint group is transferable to the query molecule. "
            "Use the harness-prefetched comparison results; do not call tools. Return only valid JSON."
        )
    if task == "bioavailability_ma":
        profile = module.get_bioavailability_prompt_profile(
            module.DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE
        )
    else:
        profile = module.get_skin_prompt_profile(module.DEFAULT_SKIN_PROMPT_PROFILE)
    return (
        profile.group_system_role
        + "Use the harness-prefetched comparison results; do not call tools. Return only valid JSON."
    )


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def _user_payload(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in messages:
        if message.get("role") != "user" or not isinstance(message.get("content"), str):
            continue
        try:
            payload = json.loads(message["content"])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _load_empirical_templates(
    trace_paths: list[Path],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    largest_properties: dict[str, Any] | None = None
    largest_comparisons: list[dict[str, Any]] = []
    properties_bytes = 0
    comparison_bytes = 0
    for path in trace_paths:
        for row in read_jsonl(path):
            payload = _user_payload(row.get("messages") or [])
            if not payload:
                continue
            properties = (payload.get("query") or {}).get(
                "prefetched_molecule_properties"
            )
            current_properties_bytes = len(
                json.dumps(
                    properties, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            )
            if properties and current_properties_bytes > properties_bytes:
                largest_properties = deepcopy(properties)
                properties_bytes = current_properties_bytes
            for neighbor in payload.get("neighbors") or []:
                comparisons = neighbor.get("prefetched_comparisons") or []
                current_comparison_bytes = len(
                    json.dumps(
                        comparisons, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                )
                if comparisons and current_comparison_bytes > comparison_bytes:
                    largest_comparisons = deepcopy(comparisons)
                    comparison_bytes = current_comparison_bytes
    metadata = {
        "trace_paths": [str(path) for path in trace_paths],
        "query_properties_template_bytes": properties_bytes,
        "neighbor_comparisons_template_bytes": comparison_bytes,
        "mode": "largest_observed_templates_repeated_per_neighbor",
    }
    return largest_properties, largest_comparisons, metadata


def _build_payload(
    module: Any,
    query: dict[str, Any],
    group: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_bound = module.bound_group_prompt_payload
    module.bound_group_prompt_payload = lambda payload, **_: payload
    try:
        unbounded = module.build_group_prompt_payload(query, group)
    finally:
        module.bound_group_prompt_payload = original_bound
    return (
        bound_group_prompt_payload(
            deepcopy(unbounded),
            evidence_prompt_profile=str(group.get("evidence_prompt_profile") or ""),
        ),
        unbounded,
    )


def audit(
    *,
    tasks: list[str],
    tokenizer_path: Path,
    output_dir: Path,
    limit: int,
    max_completion_tokens: int,
    context_windows: list[int],
    empirical_trace_paths: list[Path],
    support_mode: str,
) -> dict[str, Any]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
    tokenizer_config = json.loads(
        (tokenizer_path / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    properties_template, comparisons_template, template_metadata = (
        _load_empirical_templates(empirical_trace_paths)
    )
    rows_out: list[dict[str, Any]] = []
    for task in tasks:
        spec = SPECS[task]
        module = importlib.import_module(spec.pipeline_module)
        records = read_jsonl(spec.split)
        if limit > 0:
            records = records[:limit]
        index_path = spec.raw_index if support_mode == "raw_full" else spec.summary_index
        with index_path.open("rb") as handle:
            index = pickle.load(handle)
        assay_count = len(index.get("assay_ranking") or [])
        system = _system_message(task, module)
        for position, record in enumerate(records):
            retrieval = retrieve_assay_prefix(
                str(record["drug"]),
                index,
                assay_prefix=assay_count,
                top_k_per_assay=3,
                min_similarity=0.3,
                neighbor_identity_policy="scaffold_disjoint",
            )
            groups = retrieval.get("groups") or []
            if not groups:
                payload = {}
                unbounded_payload = {}
                group = None
            else:
                group = groups[0]
                group["tools_prefetched"] = True
                if properties_template:
                    query = {
                        "molecule_id": "query",
                        "identity_hidden": True,
                        "tools_prefetched": True,
                        "prefetched_molecule_properties": deepcopy(properties_template),
                    }
                else:
                    query = dict(retrieval["query"])
                for neighbor_position, neighbor in enumerate(
                    group.get("neighbors") or [], start=1
                ):
                    if comparisons_template:
                        neighbor["prefetched_comparisons"] = deepcopy(
                            comparisons_template
                        )
                    neighbor["molecule_chembl_id"] = f"neighbor_1_{neighbor_position}"
                    neighbor["canonical_smiles"] = "[hidden]"
                    neighbor["standard_inchi_key"] = ""
                condition = external_condition_sentence(record)
                if condition:
                    query["external_condition"] = condition
                payload, unbounded_payload = _build_payload(
                    module,
                    query,
                    group,
                )
            user = json.dumps(payload, ensure_ascii=False)
            unbounded_user = json.dumps(unbounded_payload, ensure_ascii=False)
            input_tokens = len(tokenizer.encode(system + "\n" + user).ids) + 32
            unbounded_input_tokens = (
                len(tokenizer.encode(system + "\n" + unbounded_user).ids) + 32
            )
            payload_bytes = len(user.encode("utf-8"))
            transport = payload.get("prompt_transport") or {}
            rows_out.append(
                {
                    "task": task,
                    "query_index": position,
                    "condition_group": record.get("condition_group") or "",
                    "n_selected_assays": assay_count,
                    "n_assays_with_neighbors": (retrieval.get("coverage") or {}).get(
                        "n_assays_with_neighbors", 0
                    ),
                    "n_unique_neighbors": (retrieval.get("coverage") or {}).get(
                        "n_neighbors_total", 0
                    ),
                    "payload_bytes": payload_bytes,
                    "input_tokens": input_tokens,
                    "input_plus_reserved_completion": input_tokens
                    + max_completion_tokens,
                    "unbounded_payload_bytes": len(unbounded_user.encode("utf-8")),
                    "unbounded_input_tokens": unbounded_input_tokens,
                    "unbounded_input_plus_reserved_completion": (
                        unbounded_input_tokens + max_completion_tokens
                    ),
                    "guard_applied": bool(transport.get("oversize_guard_applied")),
                    "transport_limit_type": str(transport.get("limit_type") or ""),
                    "original_payload_bytes": len(unbounded_user.encode("utf-8")),
                    "original_payload_tokens": int(
                        transport.get("original_payload_tokens")
                        or unbounded_input_tokens
                    ),
                    "neighbors_in_prompt": len(payload.get("neighbors") or []),
                    "evidence_rows_in_prompt": sum(
                        len(neighbor.get("evidence_rows") or [])
                        for neighbor in payload.get("neighbors") or []
                    ),
                }
            )
            if (position + 1) % 25 == 0 or position + 1 == len(records):
                print(f"[{task}] audited {position + 1}/{len(records)}", flush=True)

    by_task = {}
    for task in tasks:
        task_rows = [row for row in rows_out if row["task"] == task]
        token_values = [row["input_tokens"] for row in task_rows]
        total_values = [row["input_plus_reserved_completion"] for row in task_rows]
        unbounded_total_values = [
            row["unbounded_input_plus_reserved_completion"] for row in task_rows
        ]
        worst = max(task_rows, key=lambda row: row["input_tokens"])
        by_task[task] = {
            "n_queries": len(task_rows),
            "input_tokens_p50": int(median(token_values)) if token_values else 0,
            "input_tokens_p95": _percentile(token_values, 0.95),
            "input_tokens_max": max(token_values, default=0),
            "input_plus_reserved_completion_max": max(total_values, default=0),
            "unbounded_input_tokens_max": max(
                (row["unbounded_input_tokens"] for row in task_rows), default=0
            ),
            "unbounded_input_plus_reserved_completion_max": max(
                unbounded_total_values, default=0
            ),
            "unbounded_context_overflow_queries": {
                str(window): sum(value > window for value in unbounded_total_values)
                for window in context_windows
            },
            "guard_applied_queries": sum(row["guard_applied"] for row in task_rows),
            "worst_query_index": worst["query_index"],
            "worst_condition_group": worst["condition_group"],
            "worst_payload_bytes": worst["payload_bytes"],
            "worst_original_payload_bytes": worst["original_payload_bytes"],
            "worst_neighbors_in_prompt": worst["neighbors_in_prompt"],
            "worst_evidence_rows_in_prompt": worst["evidence_rows_in_prompt"],
        }
    max_total = max(
        (row["input_plus_reserved_completion"] for row in rows_out), default=0
    )
    summary = {
        "artifact_type": "conditioned_assay_prompt_length_audit.v2",
        "tokenizer": str(tokenizer_path.resolve()),
        "tokenizer_model_max_length": int(tokenizer_config["model_max_length"]),
        "max_completion_tokens": max_completion_tokens,
        "chat_template_overhead_tokens": 32,
        "retrieval": {
            "assay_level": "full_catalog",
            "top_k_per_assay": 3,
            "min_similarity": 0.3,
            "neighbor_identity_policy": "scaffold_disjoint",
            "prompt_profile": (
                evidence_contract.ASSAY_RAW_CARD_PROMPT_PROFILE
                if support_mode == "raw_full"
                else evidence_contract.ASSAY_COMPACT_V2_PROMPT_PROFILE
            ),
            "support_mode": support_mode,
            "prefetched_comparisons": (
                "empirical_conservative_stress"
                if comparisons_template
                else "not_materialized"
            ),
        },
        "empirical_templates": template_metadata,
        "n_queries": len(rows_out),
        "by_task": by_task,
        "context_windows": {
            str(window): {
                "headroom_at_global_worst": window - max_total,
                "n_overflow": sum(
                    row["input_plus_reserved_completion"] > window for row in rows_out
                ),
            }
            for window in context_windows
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_dir / "prompt_lengths.jsonl", rows_out)
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(SPECS), default=list(SPECS))
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "prompt_length_audit_raw_v3_untruncated"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-completion-tokens", type=int, default=20_480)
    parser.add_argument(
        "--support-mode",
        choices=("summary", "raw_full"),
        default="raw_full",
        help="Use the current raw-v3 cards or the historical frozen-summary v2 cards.",
    )
    parser.add_argument(
        "--empirical-trace",
        action="append",
        default=[],
        help="Historical trace_messages.jsonl used to select the largest real tool payloads.",
    )
    parser.add_argument(
        "--context-windows",
        type=int,
        nargs="+",
        default=[65_536, 131_072, 1_048_576],
    )
    args = parser.parse_args(argv)
    summary = audit(
        tasks=args.tasks,
        tokenizer_path=Path(args.tokenizer),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        max_completion_tokens=args.max_completion_tokens,
        context_windows=args.context_windows,
        empirical_trace_paths=[Path(path) for path in args.empirical_trace],
        support_mode=args.support_mode,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
