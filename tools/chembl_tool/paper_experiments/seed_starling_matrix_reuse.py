"""Prepare a Starling matrix and seed strictly compatible prior-run artifacts.

Reuse is molecule-keyed rather than query-index-keyed.  Single branches require
an identical inference/visibility contract; group branches additionally require
identical LLM-visible group input; finals require an identical complete
retrieval prompt contract.  Target retrieval and manifests are always freshly
materialized before any historical artifact is considered.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.json_utils import (
    atomic_output_path,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.common.retrieval_ablation import (
    load_reusable_group_outputs,
    retrieval_prompt_hash,
)
from tools.chembl_tool.common.task_workflows.global_prompt_pool import (
    BatchCommand,
    _prepare_command,
)
from tools.chembl_tool.common.task_workflows.reasoning_stage_runtime import (
    prepare_stage_item,
)
from tools.chembl_tool.paper_experiments.molecular_evidence_agent import (
    DEPLOYMENT_VISIBLE,
    IDENTITY_BLIND,
    PARENT_DISJOINT,
    _command,
    _select_experiments,
    experiment_run_root,
)
from tools.chembl_tool.paper_experiments.starling_benchmark_matrix import (
    experiments_for_starling_benchmark,
)


CONTRACT_FIELDS = (
    "model",
    "reasoning_effort",
    "temperature",
    "thinking",
    "identity_blind",
    "tool_execution_mode",
    "experiment_mode",
    "retrieval_source",
    "max_tool_rounds",
    "top_k_per_group",
    "min_similarity",
    "neighbor_selector",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    experiments = experiments_for_starling_benchmark(
        "scaffold",
        evaluation_subset="valid",
        data_root=args.benchmark_data_root,
        canonical_paper_root=args.canonical_paper_root,
    )
    selected = _select_experiments(args.experiments, experiments=experiments)
    runtime_args = argparse.Namespace(
        python_executable=args.python_executable,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        visibility_mode=args.visibility_mode,
        neighbor_identity_policy=PARENT_DISJOINT,
        fresh_parent_disjoint=True,
        paper_root=str(args.target_root),
        split="valid",
        parallelism=args.parallelism,
        timeout_s=args.timeout_s,
        max_stage_requeues=0,
        limit=args.limit,
        experiments=args.experiments,
        single_analysis_root="",
        group_analysis_root="",
        neighbor_selector="similarity",
        neighbor_context_profile="standard",
    )
    prepared = {
        experiment.name: _prepare_command(
            BatchCommand(experiment.name, _command(experiment, runtime_args)),
            max_workers=args.parallelism,
            max_stage_requeues=0,
        )
        for experiment in selected
    }
    preparation_failures = _prepare_targets(prepared, args.prepare_workers)
    if preparation_failures:
        write_json_atomic(
            args.target_root / "reuse_audit/preparation_failures.json",
            preparation_failures,
        )
        raise SystemExit(f"Target preparation failed for {len(preparation_failures)} rows")

    audit_rows = _seed_targets(
        selected,
        prepared,
        args,
        workers=args.seed_workers,
    )

    summary = _reuse_summary(audit_rows, args)
    audit_root = args.target_root / "reuse_audit"
    write_jsonl_atomic(audit_root / f"{args.visibility_mode}.jsonl", audit_rows)
    write_json_atomic(audit_root / f"{args.visibility_mode}_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def _prepare_targets(prepared: dict[str, Any], workers: int) -> list[dict[str, Any]]:
    jobs = [
        (name, batch, item)
        for name, batch in prepared.items()
        for item in batch.items
        if not (
            batch.batch_run_root
            / f"{batch.batch_id}_idx{item.index:05d}"
            / "retrieval.json"
        ).exists()
    ]
    failures: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(prepare_stage_item, batch, item): (name, item.index)
            for name, batch, item in jobs
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            name, index = futures[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001 - retain all seed failures.
                result = {"status": "error", "error": str(error)}
            if result.get("status") != "ok":
                failures.append({"experiment": name, "query_index": index, **result})
            if completed % 500 == 0 or completed == len(jobs):
                print(
                    f"[matrix_reuse] prepared={completed}/{len(jobs)} "
                    f"failures={len(failures)}"
                )
    return failures


def _seed_targets(
    selected: list[Any],
    prepared: dict[str, Any],
    args: argparse.Namespace,
    *,
    workers: int,
) -> list[dict[str, Any]]:
    jobs: list[tuple[Any, Any, Path | None, str]] = []
    for experiment in selected:
        batch = prepared[experiment.name]
        source_runs = _source_runs(
            args.source_roots,
            args.visibility_mode,
            experiment.task,
            experiment.name,
        )
        for item in batch.items:
            smiles = str(item.record.get(batch.args.smiles_field) or "")
            jobs.append((experiment, item, source_runs.get(smiles), smiles))

    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for experiment, item, source_run, smiles in jobs:
            batch = prepared[experiment.name]
            target_run = (
                batch.batch_run_root / f"{batch.batch_id}_idx{item.index:05d}"
            )
            future = executor.submit(
                _seed_one,
                source_run,
                target_run,
                item.record,
                item.index,
                batch.config.pipeline_module,
                identity_blind=args.visibility_mode == IDENTITY_BLIND,
            )
            futures[future] = (experiment, item, smiles)

        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            experiment, item, smiles = futures[future]
            rows.append(
                future.result()
                | {
                    "experiment": experiment.name,
                    "task": experiment.task,
                    "query_index": item.index,
                    "drug": smiles,
                }
            )
            if completed % 500 == 0 or completed == len(jobs):
                print(f"[matrix_reuse] seeded={completed}/{len(jobs)}")
    return sorted(rows, key=lambda row: (row["experiment"], row["query_index"]))


def _source_runs(
    roots: list[Path],
    visibility_mode: str,
    task: str,
    experiment: str,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for root in roots:
        run_root = experiment_run_root(
            visibility_mode,
            PARENT_DISJOINT,
            paper_root=root,
        )
        batch = run_root / task / experiment
        predictions = batch / "predictions.jsonl"
        if not predictions.exists():
            continue
        for row in _read_jsonl(predictions):
            if row.get("status") != "ok":
                continue
            smiles = str(row.get("smiles") or row.get("drug") or "")
            run_dir = Path(str(row.get("run_dir") or ""))
            if smiles and run_dir.exists() and smiles not in result:
                result[smiles] = run_dir
    return result


def _seed_one(
    source_run: Path | None,
    target_run: Path,
    query_record: dict[str, Any],
    query_index: int,
    pipeline_module: str,
    *,
    identity_blind: bool,
) -> dict[str, Any]:
    base = {
        "source_run": str(source_run or ""),
        "source_available": bool(source_run),
        "contract_compatible": False,
        "single_reused": False,
        "n_group_branches_reused": 0,
        "final_reused": False,
        "reuse_reason": "no_compatible_source",
    }
    if source_run is None:
        return base
    source_manifest = _read_json(source_run / "manifest.json")
    target_manifest = _read_json(target_run / "manifest.json")
    mismatches = _contract_mismatches(source_manifest, target_manifest)
    if mismatches:
        return {**base, "contract_mismatches": mismatches}

    source_retrieval = _read_json(source_run / "retrieval.json")
    target_retrieval = _read_json(target_run / "retrieval.json")
    if _query_smiles(source_retrieval) != _query_smiles(target_retrieval):
        return {**base, "contract_mismatches": ["query_smiles"]}

    single_source = source_run / "single_molecule_reasoning_output.json"
    single_target = target_run / "single_molecule_reasoning_output.json"
    single_output = _read_json(single_source)
    if single_output.get("status") != "ok":
        return {**base, "reuse_reason": "source_single_incomplete"}
    target_single_output = _read_json(single_target) if single_target.exists() else {}
    if target_single_output.get("status") != "ok":
        single_output["reused_from"] = str(single_source)
        write_json_atomic(single_target, single_output)

    group_outputs = load_reusable_group_outputs(
        str(source_run),
        target_retrieval,
        target_neighbor_context_profile=str(
            target_manifest.get("neighbor_context_profile") or "standard"
        ),
    )
    group_target = target_run / "group_reasoning_outputs.jsonl"
    existing_group_outputs = _read_jsonl(group_target)
    merged_group_outputs = _merge_successful_group_outputs(
        existing_group_outputs,
        group_outputs,
    )
    if merged_group_outputs:
        write_jsonl_atomic(group_target, merged_group_outputs)

    source_hash = retrieval_prompt_hash(source_retrieval)
    target_hash = retrieval_prompt_hash(target_retrieval)
    expected_groups = set(target_manifest.get("expected_group_ids") or [])
    reused_groups = {
        str(output.get("group_id") or "")
        for output in group_outputs
        if output.get("status") == "ok"
    }
    final_reused = False
    source_final = source_run / "final_reasoning_output.json"
    target_final = target_run / "final_reasoning_output.json"
    target_final_output = _read_json(target_final) if target_final.exists() else {}
    source_final_output = _read_json(source_final) if source_final.exists() else {}
    if (
        target_final_output.get("status") == "ok"
        and source_final_output.get("status") == "ok"
        and source_hash == target_hash
        and target_final_output == source_final_output
    ):
        final_reused = True
    if (
        target_final_output.get("status") != "ok"
        and
        source_hash == target_hash
        and reused_groups == expected_groups
        and source_final.exists()
    ):
        final_output = source_final_output
        if final_output.get("status") == "ok":
            write_json_atomic(target_run / "final_reasoning_output.json", final_output)
            _write_trace(
                target_run,
                query_record,
                query_index,
                pipeline_module,
                identity_blind=identity_blind,
            )
            final_reused = True

    provenance = {
        "schema_version": "starling.molecule_keyed_stage_reuse.v1",
        "reused_from": str(source_run),
        "single_reused": True,
        "n_group_branches_reused": len(group_outputs),
        "final_reused": final_reused,
        "source_retrieval_prompt_hash": source_hash,
        "target_retrieval_prompt_hash": target_hash,
        "reuse_reason": (
            "identical_complete_llm_visible_retrieval_input"
            if final_reused
            else "compatible_single_and_exact_unchanged_group_branches_only"
        ),
    }
    write_json_atomic(target_run / "reuse.json", provenance)
    target_manifest["artifact_reuse"] = provenance
    write_json_atomic(target_run / "manifest.json", target_manifest)
    return {
        **base,
        "contract_compatible": True,
        "single_reused": True,
        "n_group_branches_reused": len(group_outputs),
        "final_reused": final_reused,
        "source_retrieval_prompt_hash": source_hash,
        "target_retrieval_prompt_hash": target_hash,
        "reuse_reason": provenance["reuse_reason"],
    }


def _merge_successful_group_outputs(
    existing: list[dict[str, Any]],
    reusable: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add reusable branches without discarding successful target computation."""
    by_group = {
        str(row.get("group_id") or ""): row
        for row in reusable
        if row.get("status") == "ok" and str(row.get("group_id") or "")
    }
    by_group.update(
        {
            str(row.get("group_id") or ""): row
            for row in existing
            if row.get("status") == "ok" and str(row.get("group_id") or "")
        }
    )
    return [by_group[group_id] for group_id in sorted(by_group)]


def _write_trace(
    target_run: Path,
    query_record: dict[str, Any],
    query_index: int,
    pipeline_module: str,
    *,
    identity_blind: bool,
) -> None:
    module = importlib.import_module(pipeline_module)
    group_outputs = _read_jsonl(target_run / "group_reasoning_outputs.jsonl")
    with atomic_output_path(target_run / "trace_messages.jsonl") as temporary:
        module._write_trace_jsonl(
            temporary,
            query_record=query_record,
            query_index=query_index,
            smiles=("[identity_blind]" if identity_blind else str(query_record.get("drug") or "")),
            single_output=_read_json(target_run / "single_molecule_reasoning_output.json"),
            group_outputs=group_outputs,
            final_output=_read_json(target_run / "final_reasoning_output.json"),
        )


def _contract_mismatches(source: dict[str, Any], target: dict[str, Any]) -> list[str]:
    mismatches = []
    for field in CONTRACT_FIELDS:
        source_value = source.get(field)
        target_value = target.get(field)
        if field == "neighbor_selector":
            source_value = source_value or "similarity"
            target_value = target_value or "similarity"
        if source_value != target_value:
            mismatches.append(field)
    source_profile = source.get("neighbor_context_profile") or "standard"
    target_profile = target.get("neighbor_context_profile") or "standard"
    if source_profile != target_profile:
        mismatches.append("neighbor_context_profile")
    return mismatches


def _query_smiles(retrieval: dict[str, Any]) -> str:
    query = retrieval.get("query") or {}
    return str(query.get("input_smiles") or query.get("canonical_smiles") or "")


def _reuse_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "starling.molecule_keyed_matrix_reuse_summary.v1",
        "model": args.model,
        "visibility_mode": args.visibility_mode,
        "target_root": str(args.target_root),
        "source_roots": [str(root) for root in args.source_roots],
        "n_sample_conditions": len(rows),
        "n_source_available": sum(row["source_available"] for row in rows),
        "n_contract_compatible": sum(row["contract_compatible"] for row in rows),
        "n_single_reused": sum(row["single_reused"] for row in rows),
        "n_group_branches_reused": sum(row["n_group_branches_reused"] for row in rows),
        "n_final_reused": sum(row["final_reused"] for row in rows),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-data-root", type=Path, required=True)
    parser.add_argument("--canonical-paper-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--source-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument(
        "--visibility-mode",
        choices=(IDENTITY_BLIND, DEPLOYMENT_VISIBLE),
        required=True,
    )
    parser.add_argument("--experiments", nargs="*", default=[])
    parser.add_argument("--parallelism", type=int, default=128)
    parser.add_argument("--prepare-workers", type=int, default=8)
    parser.add_argument("--seed-workers", type=int, default=16)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional per-condition query limit for an isolated smoke test.",
    )
    parser.add_argument("--python-executable", default="python")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
