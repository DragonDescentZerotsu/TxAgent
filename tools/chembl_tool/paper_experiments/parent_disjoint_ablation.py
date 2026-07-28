"""Plan and materialize selective parent-disjoint deployment-visible runs."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.experiment_retrieval import retrieve_experiment_view
from tools.chembl_tool.common.molecule_identity import identity_from_record, normalize_molecule_identity
from tools.chembl_tool.common.retrieval_ablation import (
    changed_group_ids,
    materialize_reused_run,
    retrieval_prompt_hash,
)
from tools.chembl_tool.common.retrieval_policy import MoleculeRelation, classify_molecule_relation
from tools.chembl_tool.common.task_workflows.retrieve_neighbors import load_index
from tools.chembl_tool.paper_experiments.molecular_evidence_agent import (
    DEPLOYMENT_VISIBLE,
    EXPERIMENTS,
    experiments_for_split,
    experiment_run_root,
    paper_root_for_split,
)


# Backward-compatible default for callers that summarize the frozen test split.
# Split-aware execution resolves the corresponding validation path in main().
DEFAULT_OUTPUT = Path("outputs/paper/molecular_evidence_agent/analysis/parent_disjoint_ablation")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.benchmark_split:
        from tools.chembl_tool.paper_experiments.build_starling_benchmark_indices import (
            paper_root_for_benchmark_split,
        )
        from tools.chembl_tool.paper_experiments.starling_benchmark_matrix import (
            experiments_for_starling_benchmark,
        )
        from tools.chembl_tool.paper_experiments.minimol_retrieval_contract import (
            MINIMOL_RETRIEVAL_FEATURE,
            paper_root_for_minimol_retrieval,
        )

        paper_root = (
            Path(args.paper_root)
            if args.paper_root
            else (
                paper_root_for_minimol_retrieval(args.benchmark_split)
                if args.retrieval_feature == MINIMOL_RETRIEVAL_FEATURE
                else paper_root_for_benchmark_split(args.benchmark_split)
            )
        )
        experiments = experiments_for_starling_benchmark(
            args.benchmark_split,
            retrieval_feature=args.retrieval_feature,
        )
        data_split = f"starling_{args.benchmark_split}"
    else:
        paper_root = Path(args.paper_root) if args.paper_root else paper_root_for_split(args.split)
        experiments = experiments_for_split(args.split)
        data_split = args.split
    operational_root = (
        Path(args.operational_root)
        if args.operational_root
        else experiment_run_root(DEPLOYMENT_VISIBLE, paper_root=paper_root)
    )
    target_root = (
        Path(args.target_root)
        if args.target_root
        else experiment_run_root(
            DEPLOYMENT_VISIBLE,
            "parent_disjoint",
            paper_root=paper_root,
        )
    )
    output_dir = Path(args.output_dir) if args.output_dir else paper_root / "analysis/parent_disjoint_ablation"
    selected = _select_experiments(
        args.experiments,
        experiments=experiments,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summaries = []
    for experiment in selected:
        if experiment.mode == "none":
            continue
        rows, summary = build_experiment_plan(
            experiment,
            operational_root=operational_root,
            target_root=target_root,
            materialize=args.materialize,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"[parent_disjoint] {experiment.name}: changed={summary['n_changed']} "
            f"reused={summary['n_reused']} total={summary['n_total']}",
            flush=True,
        )

    _write_tsv(output_dir / "sample_condition_diff.tsv", all_rows)
    _write_tsv(
        output_dir / "same_parent_exposure_by_condition.tsv",
        [_same_parent_exposure_row(summary) for summary in summaries],
    )
    payload = {
        "policy": "parent_disjoint",
        "data_split": data_split,
        "operational_root": str(operational_root),
        "target_root": str(target_root),
        "materialized": args.materialize,
        "n_sample_conditions": len(all_rows),
        "n_changed": sum(bool(row["changed"]) for row in all_rows),
        "n_reused": sum(not bool(row["changed"]) for row in all_rows),
        "n_queries_with_same_parent": sum(bool(row["same_parent_group_ids"]) for row in all_rows),
        "n_groups_with_same_parent": sum(
            len(str(row["same_parent_group_ids"]).split("|"))
            for row in all_rows
            if row["same_parent_group_ids"]
        ),
        "n_retrieved_neighbor_slots": sum(int(row["n_retrieved_neighbor_slots"]) for row in all_rows),
        "n_same_parent_neighbor_slots": sum(int(row["n_same_parent_neighbor_slots"]) for row in all_rows),
        "n_same_parent_unique_neighbors_summed_per_query": sum(
            int(row["n_same_parent_unique_neighbors"]) for row in all_rows
        ),
        "n_same_parent_rank1_slots": sum(int(row["n_same_parent_rank1_slots"]) for row in all_rows),
        "experiments": summaries,
    }
    payload["same_parent_neighbor_slot_fraction"] = _safe_fraction(
        payload["n_same_parent_neighbor_slots"],
        payload["n_retrieved_neighbor_slots"],
    )
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


def build_experiment_plan(
    experiment: Any,
    *,
    operational_root: Path,
    target_root: Path,
    materialize: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_batch = operational_root / experiment.task / experiment.name
    if not source_batch.exists():
        raise FileNotFoundError(f"Operational source batch does not exist: {source_batch}")
    batch_manifest = _read_json(source_batch / "manifest.json")
    top_k = int(batch_manifest.get("top_k_per_group") or 3)
    min_similarity = float(batch_manifest.get("min_similarity") or 0.3)
    records = _read_jsonl(Path(experiment.input_jsonl))
    target_batch = target_root / experiment.task / experiment.name
    if materialize:
        (target_batch / "runs").mkdir(parents=True, exist_ok=True)
        (target_batch / "logs").mkdir(parents=True, exist_ok=True)

    baseline_inputs = []
    for query_index, record in enumerate(records):
        source_run = source_batch / "runs" / f"{experiment.name}_idx{query_index:05d}"
        baseline_path = source_run / "retrieval.json"
        if not baseline_path.exists():
            raise FileNotFoundError(f"Missing operational retrieval: {baseline_path}")
        baseline = _read_json(baseline_path)
        same_parent_stats = _same_parent_stats(str(record.get("drug") or ""), baseline)
        baseline_inputs.append((query_index, record, source_run, baseline, same_parent_stats))

    index = None
    config = None
    if any(item[4]["group_ids"] for item in baseline_inputs):
        index = load_index(Path(experiment.index))
        config_module = importlib.import_module(f"tools.chembl_tool.tasks.{experiment.task}.experiment_config")
        config = config_module.get_source_config(experiment.source)

    rows = []
    for query_index, record, source_run, baseline, same_parent_stats in baseline_inputs:
        same_parent_groups = same_parent_stats["group_ids"]
        baseline_hash = retrieval_prompt_hash(baseline)
        if same_parent_groups:
            target = retrieve_experiment_view(
                str(record.get("drug") or ""),
                index,
                mode=experiment.mode,
                config=config,
                top_k_per_group=top_k,
                min_similarity=min_similarity,
                neighbor_identity_policy="parent_disjoint",
            )
            target_hash = retrieval_prompt_hash(target)
            changed_groups = changed_group_ids(baseline, target)
        else:
            target_hash = baseline_hash
            changed_groups = []
        changed = baseline_hash != target_hash
        row = {
            "experiment": experiment.name,
            "task": experiment.task,
            "source": experiment.source,
            "mode": experiment.mode,
            "query_index": query_index,
            "changed": changed,
            "n_changed_groups": len(changed_groups),
            "changed_group_ids": "|".join(changed_groups),
            "same_parent_group_ids": "|".join(same_parent_groups),
            "n_retrieved_neighbor_slots": same_parent_stats["n_retrieved_neighbor_slots"],
            "n_same_parent_neighbor_slots": same_parent_stats["n_same_parent_neighbor_slots"],
            "n_same_parent_unique_neighbors": same_parent_stats["n_same_parent_unique_neighbors"],
            "n_same_parent_rank1_slots": same_parent_stats["n_same_parent_rank1_slots"],
            "baseline_prompt_hash": baseline_hash,
            "parent_disjoint_prompt_hash": target_hash,
        }
        rows.append(row)
        if materialize and not changed:
            target_run = target_batch / "runs" / f"{experiment.name}_idx{query_index:05d}"
            materialize_reused_run(
                source_run,
                target_run,
                {
                    "reused_from": str(source_run),
                    "reuse_reason": "identical_llm_visible_retrieval_input",
                    "baseline_prompt_hash": baseline_hash,
                    "target_prompt_hash": target_hash,
                    "neighbor_identity_policy": "parent_disjoint",
                },
            )
    summary = {
        "experiment": experiment.name,
        "task": experiment.task,
        "source": experiment.source,
        "mode": experiment.mode,
        "n_total": len(rows),
        "n_changed": sum(bool(row["changed"]) for row in rows),
        "n_reused": sum(not bool(row["changed"]) for row in rows),
        "n_queries_with_same_parent": sum(bool(row["same_parent_group_ids"]) for row in rows),
        "n_groups_with_same_parent": sum(
            len(str(row["same_parent_group_ids"]).split("|"))
            for row in rows
            if row["same_parent_group_ids"]
        ),
        "n_retrieved_neighbor_slots": sum(int(row["n_retrieved_neighbor_slots"]) for row in rows),
        "n_same_parent_neighbor_slots": sum(int(row["n_same_parent_neighbor_slots"]) for row in rows),
        "n_same_parent_unique_neighbors_summed_per_query": sum(
            int(row["n_same_parent_unique_neighbors"]) for row in rows
        ),
        "n_same_parent_rank1_slots": sum(int(row["n_same_parent_rank1_slots"]) for row in rows),
        "changed_indices": [row["query_index"] for row in rows if row["changed"]],
    }
    summary["same_parent_neighbor_slot_fraction"] = _safe_fraction(
        summary["n_same_parent_neighbor_slots"],
        summary["n_retrieved_neighbor_slots"],
    )
    if materialize:
        (target_batch / "reuse_plan.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return rows, summary


def _same_parent_stats(query_smiles: str, retrieval: dict[str, Any]) -> dict[str, Any]:
    """Count LLM-visible same-parent exposure before parent-disjoint filtering.

    A neighbor slot is one appearance in one retrieval group. The same source
    record can therefore contribute multiple slots when it is independently
    exposed to multiple mechanism branches. ``n_same_parent_unique_neighbors``
    deduplicates records within this query-condition only.
    """
    query_identity = normalize_molecule_identity(query_smiles)
    affected: list[str] = []
    n_retrieved = 0
    n_same_parent = 0
    n_same_parent_rank1 = 0
    unique_same_parent: set[str] = set()
    for group in retrieval.get("groups") or []:
        has_same_parent = False
        for rank, neighbor in enumerate(group.get("neighbors") or [], start=1):
            n_retrieved += 1
            candidate_identity = identity_from_record(neighbor)
            if classify_molecule_relation(query_identity, candidate_identity) is not MoleculeRelation.SAME_PARENT:
                continue
            has_same_parent = True
            n_same_parent += 1
            n_same_parent_rank1 += int(rank == 1)
            unique_same_parent.add(_neighbor_record_key(neighbor, candidate_identity))
        if has_same_parent:
            affected.append(str(group.get("group_id") or ""))
    return {
        "group_ids": sorted(affected),
        "n_retrieved_neighbor_slots": n_retrieved,
        "n_same_parent_neighbor_slots": n_same_parent,
        "n_same_parent_unique_neighbors": len(unique_same_parent),
        "n_same_parent_rank1_slots": n_same_parent_rank1,
    }


def _same_parent_group_ids(query_smiles: str, retrieval: dict[str, Any]) -> list[str]:
    """Backward-compatible helper for callers that only need affected groups."""
    return list(_same_parent_stats(query_smiles, retrieval)["group_ids"])


def _neighbor_record_key(neighbor: dict[str, Any], identity: Any) -> str:
    return str(
        identity.standard_inchi_key
        or identity.canonical_smiles
        or neighbor.get("molecule_chembl_id")
        or neighbor.get("source_record_id")
        or neighbor.get("canonical_smiles")
        or neighbor.get("smiles")
        or json.dumps(neighbor, sort_keys=True, default=str)
    )


def _safe_fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _same_parent_exposure_row(summary: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "experiment",
        "task",
        "source",
        "mode",
        "n_total",
        "n_queries_with_same_parent",
        "n_groups_with_same_parent",
        "n_retrieved_neighbor_slots",
        "n_same_parent_neighbor_slots",
        "same_parent_neighbor_slot_fraction",
        "n_same_parent_unique_neighbors_summed_per_query",
        "n_same_parent_rank1_slots",
        "n_changed",
        "n_reused",
    )
    return {field: summary.get(field) for field in fields}


def _select_experiments(
    names: list[str],
    *,
    experiments: list[Any] = EXPERIMENTS,
) -> list[Any]:
    candidates = [experiment for experiment in experiments if experiment.mode != "none"]
    if not names:
        return candidates
    by_name = {experiment.name: experiment for experiment in candidates}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise SystemExit(f"Unknown retrieval experiments: {', '.join(missing)}")
    return [by_name[name] for name in names]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["experiment"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Parent-disjoint 选择性重跑计划",
        "",
        f"- sample-condition 总数：{summary['n_sample_conditions']}",
        f"- LLM 可见 retrieval 输入发生变化：{summary['n_changed']}",
        f"- 可直接复用：{summary['n_reused']}",
        f"- 至少检索到一个 same-parent neighbor 的 query-condition：{summary['n_queries_with_same_parent']}",
        f"- 含 same-parent neighbor 的 mechanism/group：{summary['n_groups_with_same_parent']}",
        f"- same-parent neighbor slots：{summary['n_same_parent_neighbor_slots']} / "
        f"{summary['n_retrieved_neighbor_slots']} "
        f"({_format_percent(summary['same_parent_neighbor_slot_fraction'])})",
        f"- rank-1 same-parent slots：{summary['n_same_parent_rank1_slots']}",
        f"- query-condition 内去重后 same-parent neighbors（跨 query 求和）："
        f"{summary['n_same_parent_unique_neighbors_summed_per_query']}",
        "",
        "slot 表示一个 neighbor 在一个 retrieval group 中的一次 LLM-visible 出现；同一记录若进入多个 "
        "mechanism branch 会计为多个 slots。unique neighbor 只在单个 query-condition 内去重。",
        "",
        "| 实验 | 总数 | 变化 | 复用 | same-parent queries | same-parent slots | slot 占比 | rank-1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["experiments"]:
        lines.append(
            f"| {row['experiment']} | {row['n_total']} | {row['n_changed']} | {row['n_reused']} | "
            f"{row['n_queries_with_same_parent']} | {row['n_same_parent_neighbor_slots']} | "
            f"{_format_percent(row['same_parent_neighbor_slot_fraction'])} | "
            f"{row['n_same_parent_rank1_slots']} |"
        )
    return "\n".join(lines) + "\n"


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="*", default=[])
    parser.add_argument("--split", choices=("test", "valid"), default="test")
    parser.add_argument("--benchmark-split", choices=("random", "scaffold"), default="")
    parser.add_argument(
        "--retrieval-feature",
        choices=("morgan", "minimol"),
        default="morgan",
    )
    parser.add_argument("--paper-root", default="")
    parser.add_argument("--operational-root", default="")
    parser.add_argument("--target-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--materialize", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
