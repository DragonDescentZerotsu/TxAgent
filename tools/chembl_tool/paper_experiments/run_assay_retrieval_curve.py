"""Run the train-only Starling assay-prefix curve through one global prompt pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from tools.chembl_tool.common.assay_retrieval import geometric_assay_prefixes
from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic
from tools.chembl_tool.common.reasoning_payload import load_env_file
from tools.chembl_tool.common.task_workflows.global_prompt_pool import (
    BatchCommand,
    SCHEDULER_VERSION,
    run_global_prompt_pool,
)


WORST_CASE_VALID_INDEX = {
    "bbb_martins": 122,
    "bioavailability_ma": 102,
    "skin_reaction": 47,
}
TASKS: dict[str, dict[str, str]] = {
    "bbb_martins": {
        "module": "tools.chembl_tool.tasks.bbb_martins.run_reasoning_batch",
        "input": "data/processed_starling_experimental_meaningful_cns_access_v2/BBB_Martins/scaffold/valid.jsonl",
        "ranked_assays": "outputs/paper/starling_assay_relevance_all_v1/bbb_martins/scores_codex_gpt_5_6_sol_v6/ranked_assays.jsonl",
        "index": "outputs/paper/starling_assay_retrieval_v1/bbb_martins/scaffold_valid_train_only_all/assay_neighbor_index.pkl",
        "replay_root": "outputs/paper/starling_assay_retrieval_v1/bbb_martins/scaffold_valid_train_only_all/replay_batches",
    },
    "bioavailability_ma": {
        "module": "tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch",
        "input": "data/processed_starling_record_supported_v2/Bioavailability_Ma/scaffold/valid.jsonl",
        "ranked_assays": "outputs/paper/starling_assay_relevance_all_v1/bioavailability_ma/scores_codex_gpt_5_6_sol_v6/ranked_assays.jsonl",
        "index": "outputs/paper/starling_assay_retrieval_v1/bioavailability_ma/scaffold_valid_train_only_all/assay_neighbor_index.pkl",
        "replay_root": "outputs/paper/starling_assay_retrieval_v1/bioavailability_ma/scaffold_valid_train_only_all/replay_batches",
    },
    "skin_reaction": {
        "module": "tools.chembl_tool.tasks.skin_reaction.run_reasoning_batch",
        "input": "data/processed_starling_record_supported_v2/Skin_Reaction/scaffold/valid.jsonl",
        "ranked_assays": "outputs/paper/starling_assay_relevance_all_v1/skin_reaction/scores_codex_gpt_5_6_sol_v6/ranked_assays.jsonl",
        "index": "outputs/paper/starling_assay_retrieval_v1/skin_reaction/scaffold_valid_train_only_all/assay_neighbor_index.pkl",
        "replay_root": "outputs/paper/starling_assay_retrieval_v1/skin_reaction/scaffold_valid_train_only_all/replay_batches",
    },
}


def _catalog_size(task: str) -> int:
    ranked_path = Path(TASKS[task]["ranked_assays"])
    return sum(1 for line in ranked_path.read_text(encoding="utf-8").splitlines() if line.strip())


def prefix_plan(
    tasks: list[str],
    requested_prefixes: list[int] | None = None,
) -> dict[str, list[int]]:
    """Build task-specific schedules and always include the complete catalog."""
    plan: dict[str, list[int]] = {}
    for task in tasks:
        total = _catalog_size(task)
        if requested_prefixes:
            prefixes = sorted({int(value) for value in requested_prefixes if 0 < int(value) < total})
            prefixes.append(total)
        else:
            prefixes = list(geometric_assay_prefixes(total, start=5, multiplier=4))
        plan[task] = prefixes
    return plan


def _require_artifacts(tasks: list[str], prefixes_by_task: dict[str, list[int]]) -> None:
    missing: list[str] = []
    for task in tasks:
        spec = TASKS[task]
        for value in (spec["input"], spec["index"]):
            if not Path(value).is_file():
                missing.append(value)
        for prefix in prefixes_by_task[task]:
            batch = Path(spec["replay_root"]) / f"assay_flat_top{prefix}"
            if not (batch / "manifest.json").is_file():
                missing.append(str(batch / "manifest.json"))
    if missing:
        raise SystemExit("Missing required assay-curve artifacts:\n" + "\n".join(missing))


def _evidence_keys(retrieval: dict[str, Any]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for group in retrieval.get("groups") or []:
        for neighbor in group.get("neighbors") or []:
            molecule_id = str(neighbor.get("molecule_chembl_id") or "")
            for row in neighbor.get("evidence_rows") or []:
                assay_id = str(
                    row.get("assay_chembl_id")
                    or ((row.get("minimal_evidence") or {}).get("provenance") or {}).get("assay_id")
                    or ""
                )
                keys.add((molecule_id, assay_id))
    return keys


def _retrieved_evidence_counts(retrieval: dict[str, Any]) -> tuple[int, int]:
    evidence_rows = [
        row
        for group in retrieval.get("groups") or []
        for neighbor in group.get("neighbors") or []
        for row in neighbor.get("evidence_rows") or []
    ]
    source_records = sum(max(0, int(row.get("source_record_count") or 0)) for row in evidence_rows)
    return len(evidence_rows), source_records


def _audit_replays(
    tasks: list[str],
    prefixes_by_task: dict[str, list[int]],
) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for task in tasks:
        spec = TASKS[task]
        index_manifest_path = Path(spec["index"]).with_name("manifest.json")
        index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
        if index_manifest.get("train_molecule_only") is not True:
            raise ValueError(f"{task} index is not train-molecule-only")
        if int(index_manifest.get("n_unexpected_parent_identities", -1)) != 0:
            raise ValueError(f"{task} index has non-train parent identities")
        input_rows = [
            json.loads(line)
            for line in Path(spec["input"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        previous_selected: dict[int, set[str]] = {}
        previous_evidence: dict[int, set[tuple[str, str]]] = {}
        prefix_stats: dict[str, Any] = {}
        for prefix in prefixes_by_task[task]:
            batch = Path(spec["replay_root"]) / f"assay_flat_top{prefix}"
            manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("neighbor_identity_policy") != "parent_disjoint":
                raise ValueError(f"{task} top{prefix} is not parent_disjoint")
            if int(manifest.get("n_items", -1)) != len(input_rows):
                raise ValueError(f"{task} top{prefix} replay count does not match valid")
            values = {
                "assays_with_neighbors": [],
                "assay_neighbor_slots": [],
                "unique_neighbor_molecules": [],
                "assay_molecule_evidence_rows": [],
                "source_records_represented": [],
                "retrieval_json_bytes": [],
            }
            n_with_neighbors = 0
            for query_index, input_row in enumerate(input_rows):
                run_dir = batch / "runs" / f"{batch.name}_idx{query_index:05d}"
                retrieval_path = run_dir / "retrieval.json"
                retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
                if retrieval.get("status") != "ok":
                    raise ValueError(f"invalid replay: {retrieval_path}")
                experiment = retrieval.get("experiment") or {}
                if int(experiment.get("assay_prefix", -1)) != prefix:
                    raise ValueError(f"wrong assay prefix: {retrieval_path}")
                if experiment.get("neighbor_identity_policy") != "parent_disjoint":
                    raise ValueError(f"wrong identity policy: {retrieval_path}")
                groups = retrieval.get("groups") or []
                if len(groups) > 1 or (groups and groups[0].get("group_id") != "Flat.assay_ranked_evidence"):
                    raise ValueError(f"replay is not one flat group: {retrieval_path}")
                if "relevance_score" in json.dumps(groups) or "relevance_rank" in json.dumps(groups):
                    raise ValueError(f"relevance cue leaked to LLM group: {retrieval_path}")
                query_smiles = str((retrieval.get("query") or {}).get("input_smiles") or "")
                if query_smiles != str(input_row.get("drug") or ""):
                    raise ValueError(f"query mismatch: {retrieval_path}")
                selected = set(map(str, experiment.get("selected_assay_ids") or []))
                evidence = _evidence_keys(retrieval)
                if query_index in previous_selected and not previous_selected[query_index] <= selected:
                    raise ValueError(f"assay prefixes are not nested: {retrieval_path}")
                if query_index in previous_evidence and not previous_evidence[query_index] <= evidence:
                    raise ValueError(f"evidence prefixes are not nested: {retrieval_path}")
                previous_selected[query_index] = selected
                previous_evidence[query_index] = evidence
                coverage = retrieval.get("coverage") or {}
                assays_with_neighbors = int(coverage.get("n_assays_with_neighbors", 0))
                slots = int(coverage.get("n_assay_neighbor_slots", 0))
                neighbors = int(coverage.get("n_neighbors_total", 0))
                evidence_rows, source_records = _retrieved_evidence_counts(retrieval)
                values["assays_with_neighbors"].append(assays_with_neighbors)
                values["assay_neighbor_slots"].append(slots)
                values["unique_neighbor_molecules"].append(neighbors)
                values["assay_molecule_evidence_rows"].append(evidence_rows)
                values["source_records_represented"].append(source_records)
                values["retrieval_json_bytes"].append(retrieval_path.stat().st_size)
                n_with_neighbors += int(neighbors > 0)
            prefix_stats[str(prefix)] = {
                "n_valid_queries": len(input_rows),
                "n_queries_with_neighbors": n_with_neighbors,
                "coverage_fraction": n_with_neighbors / len(input_rows),
                **{
                    f"{name}_{suffix}": (max(rows) if suffix == "max" else mean(rows))
                    for name, rows in values.items()
                    for suffix in ("mean", "max")
                },
            }
        audit[task] = {
            "train_molecule_only": True,
            "n_unexpected_parent_identities": 0,
            "strictly_nested_assays_and_evidence": True,
            "prefixes": prefix_stats,
        }
    return audit


def _commands(
    args: argparse.Namespace,
    prefixes_by_task: dict[str, list[int]],
) -> list[BatchCommand]:
    commands: list[BatchCommand] = []
    output_root = Path(args.output_root)
    for task in args.tasks:
        spec = TASKS[task]
        task_root = output_root / task
        expected_items = sum(
            1 for line in Path(spec["input"]).read_text(encoding="utf-8").splitlines() if line.strip()
        )
        first_prefix = prefixes_by_task[task][0]
        shared_single_batch = task_root / f"assay_flat_top{first_prefix}"
        for prefix in prefixes_by_task[task]:
            batch_dir = task_root / f"assay_flat_top{prefix}"
            metrics_path = batch_dir / "metrics.json"
            if metrics_path.is_file() and not args.include_complete_batches:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if (
                    int(metrics.get("n_total", -1)) == expected_items
                    and int(metrics.get("n_failed_runs", -1)) == 0
                ):
                    continue
            name = f"{task}__assay_flat_top{prefix}"
            command = [
                args.python_executable,
                "-m",
                spec["module"],
                "--input-jsonl",
                spec["input"],
                "--index",
                spec["index"],
                "--experiment-mode",
                "full_flat",
                "--retrieval-source",
                "starling",
                "--neighbor-identity-policy",
                "parent_disjoint",
                "--identity-blind",
                "--retrieval-replay-source-batch",
                str(Path(spec["replay_root"]) / f"assay_flat_top{prefix}"),
                "--batch-root",
                str(task_root),
                "--batch-id",
                f"assay_flat_top{prefix}",
                "--model",
                args.model,
                "--base-url",
                args.base_url,
                "--api-key-env",
                args.api_key_env,
                "--tool-service-url",
                args.tool_service_url,
                "--top-k-per-group",
                "3",
                "--min-similarity",
                "0.3",
                "--max-tokens",
                str(args.max_tokens),
                "--timeout-s",
                str(args.timeout_s),
                "--disable-thinking",
                "--reasoning-effort",
                "",
                "--max-stage-requeues",
                str(args.max_stage_requeues),
                "--skip-existing",
                "--no-stream-logs",
            ]
            if prefix != first_prefix:
                command.extend(["--single-analysis-source-batch", str(shared_single_batch)])
            if args.worst_case_smoke:
                command.extend(["--indices", str(WORST_CASE_VALID_INDEX[task])])
            elif args.limit:
                command.extend(["--limit", str(args.limit)])
            if args.indices and not args.worst_case_smoke:
                command.append("--indices")
                command.extend(args.indices)
            commands.append(BatchCommand(name, command))
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=list(TASKS))
    parser.add_argument(
        "--prefixes",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional shared intermediate prefixes. The complete task-specific assay catalog "
            "is always appended. By default use 5, then x4, then the complete catalog."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="outputs/paper/starling_assay_retrieval_curve_v1/scaffold_valid_train_only_epyc_deepseek_v4_flash_0731",
    )
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    parser.add_argument("--base-url", default="http://127.0.0.1:50001/v1")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--tool-service-url", default="http://127.0.0.1:8765")
    parser.add_argument("--parallelism", type=int, default=64)
    parser.add_argument("--max-stage-requeues", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=20480)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--indices", nargs="*", default=None)
    parser.add_argument("--worst-case-smoke", action="store_true")
    parser.add_argument("--python-executable", default="/data1/tianang/anaconda3/envs/vllm/bin/python")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument(
        "--include-complete-batches",
        action="store_true",
        help="Also submit already complete zero-failure batches; default launches only missing conditions.",
    )
    args = parser.parse_args(argv)
    if args.prefixes and any(value <= 0 for value in args.prefixes):
        parser.error("--prefixes must contain positive integers")
    if args.parallelism < 1 or args.parallelism > 64:
        parser.error("PARCC DeepSeek assay curve parallelism must be between 1 and 64")
    prefixes_by_task = prefix_plan(args.tasks, args.prefixes)
    _require_artifacts(args.tasks, prefixes_by_task)
    load_env_file(Path(args.env_file))
    replay_audit = _audit_replays(args.tasks, prefixes_by_task)
    commands = _commands(args, prefixes_by_task)
    manifest: dict[str, Any] = {
        "experiment": "starling_assay_retrieval_curve.v1",
        "scheduler": SCHEDULER_VERSION,
        "evaluation_subset": "valid",
        "reference_pool": "train_only",
        "tasks": args.tasks,
        "prefixes_by_task": prefixes_by_task,
        "assay_catalog_size_by_task": {
            task: _catalog_size(task) for task in args.tasks
        },
        "prefix_policy": {
            "start": 5,
            "multiplier": 4,
            "always_include_full_catalog": True,
            "zero_retrieval_reuses_historical_none": True,
        },
        "model": args.model,
        "base_url": args.base_url,
        "visibility_mode": "identity_blind",
        "neighbor_identity_policy": "parent_disjoint",
        "top_k_per_assay": 3,
        "min_similarity": 0.3,
        "parallelism": args.parallelism,
        "reasoning_effort": "",
        "thinking": "disabled",
        "max_tokens": args.max_tokens,
        "inputs": {
            task: {
                "valid_jsonl": TASKS[task]["input"],
                "valid_jsonl_sha256": sha256_file(Path(TASKS[task]["input"])),
                "train_only_index": TASKS[task]["index"],
                "train_only_index_sha256": sha256_file(Path(TASKS[task]["index"])),
                "replay_root": TASKS[task]["replay_root"],
            }
            for task in args.tasks
        },
        "conditions": [
            f"{task}__assay_flat_top{prefix}"
            for task in args.tasks
            for prefix in prefixes_by_task[task]
        ],
        "launched_conditions": [command.experiment_name for command in commands],
        "complete_batches_reused": not args.include_complete_batches,
        "replay_audit": replay_audit,
        "limit": args.limit,
        "indices": args.indices or [],
        "worst_case_smoke": args.worst_case_smoke,
        "worst_case_valid_indices": (
            {task: WORST_CASE_VALID_INDEX[task] for task in args.tasks}
            if args.worst_case_smoke
            else {}
        ),
    }
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_root / "experiment_manifest.json", manifest)
    write_json_atomic(output_root / "replay_audit.json", replay_audit)
    if args.manifest_only:
        print(json.dumps(manifest, indent=2))
        return 0
    failed = run_global_prompt_pool(
        commands,
        max_workers=args.parallelism,
        max_stage_requeues=args.max_stage_requeues,
    )
    if failed:
        print(json.dumps({"failed": failed}, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
