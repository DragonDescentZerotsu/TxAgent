"""Compile current-lineage V11 group prompts without calling an LLM.

This is a prompt-review utility, not a benchmark launcher.  Retrieval is real
(scaffold-valid query, Morgan-50 pool, no floor, parent-disjoint, compact V11
cache), while tool text is an explicit placeholder so the audit can run without
starting the resident tool service.  Formal GLM launches still prefetch the real
tool bundle through the normal reasoning harness.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
from pathlib import Path
from typing import Any, Callable

from tools.chembl_tool.common.assay_reranking.v11 import V11CachedAssayReranker
from tools.chembl_tool.common.assay_transfer_prompt_policy import (
    prepare_assay_transfer_selected_neighbors,
)
from tools.chembl_tool.common.experiment_retrieval import retrieve_experiment_view
from tools.chembl_tool.common.experiment_retrieval import (
    flatten_retrieval_groups,
)
from tools.chembl_tool.common.identity_blind import (
    prepare_reasoning_retrieval,
    query_without_prefetched_tools,
)
from tools.chembl_tool.tasks.bbb_martins.experiment_config import (
    get_source_config as bbb_source_config,
)
from tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline import (
    build_group_prompt_payload as build_bbb_group_prompt_payload,
)
from tools.chembl_tool.tasks.bbb_martins.starling_compact_artifacts import (
    load_compact_neighbor_index as load_bbb_index,
)
from tools.chembl_tool.tasks.bioavailability_ma.experiment_config import (
    get_source_config as bioavailability_source_config,
)
from tools.chembl_tool.tasks.bioavailability_ma.group_prompt_render import (
    build_group_messages as build_bioavailability_group_messages,
)
from tools.chembl_tool.tasks.bioavailability_ma.prompt_profiles import (
    DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
    get_bioavailability_prompt_profile,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_compact_artifacts import (
    load_compact_neighbor_index as load_bioavailability_index,
)
from tools.chembl_tool.tasks.skin_reaction.experiment_config import (
    get_source_config as skin_source_config,
)
from tools.chembl_tool.tasks.skin_reaction.run_reasoning_pipeline import (
    build_group_prompt_payload as build_skin_group_prompt_payload,
)
from tools.chembl_tool.tasks.skin_reaction.starling_compact_artifacts import (
    load_compact_neighbor_index as load_skin_index,
)
from tools.chembl_tool.paper_experiments.run_v11_assay_transfer_scaffold_valid import (
    TASKS,
)


OUTPUT_ROOT = Path("outputs/paper/v11_assay_transfer_scaffold_valid_launcher/prompt_review")
SELECTIONS = (("scored_record", 1), ("mean_score_molecule", 6))
MODES = ("full_flat", "full_mechanism")
PROMPT_TASKS = TASKS


class _PromptAuditToolClient:
    """Supply a visible placeholder through the production identity-blind path."""

    def invoke_many(self, calls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        return [
            {
                "tool_name": tool_name,
                "status": "ok",
                "content": (
                    f"[{tool_name}]\nPrompt-review placeholder. The formal run will insert "
                    "the resident tool service result here."
                ),
                "warnings": [],
                "errors": [],
            }
            for tool_name, _ in calls
        ]


def _first_jsonl_record(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


def _task_runtime(task_id: str) -> tuple[Callable[..., dict[str, Any]], Any]:
    if task_id == "bbb_martins":
        return load_bbb_index, bbb_source_config("starling")
    if task_id == "bioavailability_ma":
        return load_bioavailability_index, bioavailability_source_config("starling")
    if task_id == "skin_reaction":
        return load_skin_index, skin_source_config("starling")
    raise ValueError(f"Unsupported prompt-audit task: {task_id}")


def _render_prompt(
    task_id: str,
    retrieval: dict[str, Any],
    *,
    flat_tools_disabled: bool,
) -> tuple[str, str]:
    group = max(retrieval["groups"], key=lambda row: len(row.get("neighbors") or []))
    query = (
        query_without_prefetched_tools(retrieval["query"])
        if flat_tools_disabled
        else retrieval["query"]
    )
    if task_id in {"bbb_martins", "skin_reaction"}:
        task_name = "BBB" if task_id == "bbb_martins" else "skin sensitization"
        system = (
            f"You are a medicinal chemistry {task_name} analog evidence analyst. Reason about "
            "whether analog evidence in one endpoint group is transferable to the query "
            "molecule. "
            + (
                "No tools are available for this branch. Do not infer query identity. "
                if flat_tools_disabled
                else "Use the harness-prefetched comparison results; do not call tools. "
                "Do not infer query identity. "
            )
            + "Return only valid JSON."
        )
        user = json.dumps(
            (
                build_bbb_group_prompt_payload(
                    query,
                    group,
                    include_query_tool_guidance=not flat_tools_disabled,
                )
                if task_id == "bbb_martins"
                else build_skin_group_prompt_payload(
                    query,
                    group,
                    include_query_tool_guidance=not flat_tools_disabled,
                )
            ),
            ensure_ascii=False,
            indent=2,
        )
        return system, user
    profile = get_bioavailability_prompt_profile(
        DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE
    )
    messages = build_bioavailability_group_messages(
        query,
        group,
        prompt_format="assay_transfer_tool",
        options={
            "output_schema_profile": "assay-transfer",
            "additional_instructions": list(profile.group_instructions),
            "system_role": profile.group_system_role,
            "group_tools_enabled": False,
            "prompt_version": "bioavailability_text_v1",
            "omit_query_tools": flat_tools_disabled,
        },
    )
    return str(messages[0]["content"]), str(messages[1]["content"])


def build_prompt_examples(
    output_root: Path = OUTPUT_ROOT,
    *,
    tasks: tuple[Any, ...] = PROMPT_TASKS,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    selected_ids = {task.task_id for task in tasks}
    previous = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    examples: list[dict[str, Any]] = [
        row
        for row in previous.get("examples") or []
        if row.get("task_id") not in selected_ids
    ]
    for task in tasks:
        load_index, config = _task_runtime(task.task_id)
        index = load_index(task.index)
        record = _first_jsonl_record(task.input_jsonl)
        query_smiles = str(record["drug"])
        for selection_unit, records_per_molecule in SELECTIONS:
            for mode in MODES:
                reranker = V11CachedAssayReranker(
                    task_id=task.task_id,
                    cache_path=task.cache,
                    cache_mode="read_only",
                )
                try:
                    retrieval = retrieve_experiment_view(
                        query_smiles,
                        index,
                        mode=mode,
                        config=config,
                        top_k_per_group=3,
                        min_similarity=0.0,
                        neighbor_identity_policy="parent_disjoint",
                        reranker=reranker,
                        assay_transfer_initial_morgan_filter=50,
                        assay_transfer_selection_unit=selection_unit,
                        assay_transfer_records_per_molecule=records_per_molecule,
                    )
                finally:
                    reranker.cache.close()
                prepare_assay_transfer_selected_neighbors(retrieval, expose_scores=True)
                flat_tools_disabled = mode == "full_flat"
                reasoning_retrieval = prepare_reasoning_retrieval(
                    retrieval,
                    _PromptAuditToolClient(),
                    identity_blind=True,
                    harness_prefetch_tools=False,
                    include_neighbor_tools=not flat_tools_disabled,
                )
                system, user = _render_prompt(
                    task.task_id,
                    reasoning_retrieval,
                    flat_tools_disabled=flat_tools_disabled,
                )
                stem = f"{task.task_id}__{mode}__{selection_unit}"
                path = output_root / f"{stem}.txt"
                path.write_text(
                    "## SYSTEM\n" + system + "\n\n## USER\n" + user + "\n",
                    encoding="utf-8",
                )
                examples.append(
                    {
                        "task_id": task.task_id,
                        "mode": mode,
                        "selection_unit": selection_unit,
                        "records_per_molecule": records_per_molecule,
                        "query_index": 0,
                        "identity_blind": True,
                        "neighbor_identity_policy": "parent_disjoint",
                        "top_k_per_group": 3,
                        "initial_morgan_filter": 50,
                        "min_similarity": 0.0,
                        "tool_text": (
                            "none" if flat_tools_disabled else "prompt_review_placeholder"
                        ),
                        "flat_tools_disabled": flat_tools_disabled,
                        "group_prompt_version": task.group_prompt_version,
                        "prompt_path": str(path),
                    }
                )
        del index
        gc.collect()
    manifest = {
        "status": "complete",
        "n_prompts": len(examples),
        "llm_called": False,
        "formal_tool_service_called": False,
        "examples": examples,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def audit_all_queries(
    output_root: Path = OUTPUT_ROOT,
    *,
    tasks: tuple[Any, ...] = PROMPT_TASKS,
) -> dict[str, Any]:
    """Exercise both selections and both topologies for every valid query."""
    path = output_root / "all_query_retrieval_audit.json"
    previous = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    )
    selected_ids = {task.task_id for task in tasks}
    task_results: dict[str, Any] = {
        task_id: result
        for task_id, result in (previous.get("tasks") or {}).items()
        if task_id not in selected_ids
    }
    for task in tasks:
        load_index, config = _task_runtime(task.task_id)
        index = load_index(task.index)
        records = [
            json.loads(line)
            for line in Path(task.input_jsonl).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        selections: dict[str, Any] = {}
        for selection_unit, records_per_molecule in SELECTIONS:
            reranker = V11CachedAssayReranker(
                task_id=task.task_id,
                cache_path=task.cache,
                cache_mode="read_only",
            )
            n_mechanism_groups = 0
            n_mechanism_neighbors = 0
            n_flat_neighbors = 0
            maximum_records_displayed = 0
            try:
                for query_index, record in enumerate(records):
                    retrieval = retrieve_experiment_view(
                        str(record["drug"]),
                        index,
                        mode="full_mechanism",
                        config=config,
                        top_k_per_group=3,
                        min_similarity=0.0,
                        neighbor_identity_policy="parent_disjoint",
                        reranker=reranker,
                        assay_transfer_initial_morgan_filter=50,
                        assay_transfer_selection_unit=selection_unit,
                        assay_transfer_records_per_molecule=records_per_molecule,
                    )
                    prepare_assay_transfer_selected_neighbors(
                        retrieval, expose_scores=True
                    )
                    groups = retrieval.get("groups") or []
                    if any(len(group.get("neighbors") or []) != 3 for group in groups):
                        raise ValueError(
                            f"{task.task_id}/{selection_unit}/query {query_index} "
                            "did not produce top-3 in every mechanism family"
                        )
                    n_mechanism_groups += len(groups)
                    n_mechanism_neighbors += sum(
                        len(group.get("neighbors") or []) for group in groups
                    )
                    for group in groups:
                        for neighbor in group.get("neighbors") or []:
                            displayed = neighbor.get("transfer_selected_records") or [
                                neighbor.get("transfer_winning_record") or {}
                            ]
                            maximum_records_displayed = max(
                                maximum_records_displayed, len(displayed)
                            )

                    flat = {
                        "groups": [flatten_retrieval_groups(groups)],
                        "experiment": {
                            **dict(retrieval.get("experiment") or {}),
                            "mode": "full_flat",
                        },
                        "coverage": dict(retrieval.get("coverage") or {}),
                    }
                    prepare_assay_transfer_selected_neighbors(flat, expose_scores=True)
                    n_flat_neighbors += len(flat["groups"][0].get("neighbors") or [])
                    del flat, retrieval, groups
                    gc.collect()
                    if query_index and query_index % 10 == 0:
                        ctypes.CDLL("libc.so.6").malloc_trim(0)
            finally:
                reranker.cache.close()
            selections[selection_unit] = {
                "status": "pass",
                "n_queries": len(records),
                "records_per_molecule": records_per_molecule,
                "n_mechanism_groups": n_mechanism_groups,
                "n_mechanism_neighbors": n_mechanism_neighbors,
                "n_flat_neighbors": n_flat_neighbors,
                "maximum_records_displayed": maximum_records_displayed,
            }
        task_results[task.task_id] = selections
        del index
        gc.collect()
    audit = {
        "status": "pass",
        "llm_called": False,
        "formal_tool_service_called": False,
        "neighbor_identity_policy": "parent_disjoint",
        "initial_morgan_filter": 50,
        "top_k_per_group": 3,
        "min_similarity": 0.0,
        "tasks": task_results,
    }
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--all-query-audit", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Skip prompt compilation and run only --all-query-audit.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(task.task_id for task in PROMPT_TASKS),
        required=True,
        help="Audit one task per process on memory-constrained hosts.",
    )
    args = parser.parse_args(argv)
    selected = tuple(task for task in PROMPT_TASKS if task.task_id in set(args.tasks))
    if args.audit_only and not args.all_query_audit:
        parser.error("--audit-only requires --all-query-audit")
    output = {}
    if not args.audit_only:
        output["prompt_examples"] = build_prompt_examples(
            args.output_root,
            tasks=selected,
        )
    if args.all_query_audit:
        output["all_query_audit"] = audit_all_queries(
            args.output_root,
            tasks=selected,
        )
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
