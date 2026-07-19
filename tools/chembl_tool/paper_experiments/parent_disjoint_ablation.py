"""Plan and materialize selective parent-disjoint deployment-visible runs."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
import pickle
from typing import Any

from tools.chembl_tool.common.experiment_retrieval import retrieve_experiment_view
from tools.chembl_tool.common.molecule_identity import identity_from_record, normalize_molecule_identity
from tools.chembl_tool.common.retrieval_ablation import (
    changed_group_ids,
    materialize_reused_run,
    retrieval_prompt_hash,
)
from tools.chembl_tool.common.retrieval_policy import MoleculeRelation, classify_molecule_relation
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
    paper_root = Path(args.paper_root) if args.paper_root else paper_root_for_split(args.split)
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
        experiments=experiments_for_split(args.split),
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
    payload = {
        "policy": "parent_disjoint",
        "data_split": args.split,
        "operational_root": str(operational_root),
        "target_root": str(target_root),
        "materialized": args.materialize,
        "n_sample_conditions": len(all_rows),
        "n_changed": sum(bool(row["changed"]) for row in all_rows),
        "n_reused": sum(not bool(row["changed"]) for row in all_rows),
        "experiments": summaries,
    }
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
        same_parent_groups = _same_parent_group_ids(str(record.get("drug") or ""), baseline)
        baseline_inputs.append((query_index, record, source_run, baseline, same_parent_groups))

    index = None
    config = None
    if any(item[4] for item in baseline_inputs):
        with Path(experiment.index).open("rb") as handle:
            index = pickle.load(handle)
        config_module = importlib.import_module(f"tools.chembl_tool.tasks.{experiment.task}.experiment_config")
        config = config_module.get_source_config(experiment.source)

    rows = []
    for query_index, record, source_run, baseline, same_parent_groups in baseline_inputs:
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
        "changed_indices": [row["query_index"] for row in rows if row["changed"]],
    }
    if materialize:
        (target_batch / "reuse_plan.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return rows, summary


def _same_parent_group_ids(query_smiles: str, retrieval: dict[str, Any]) -> list[str]:
    query_identity = normalize_molecule_identity(query_smiles)
    affected = []
    for group in retrieval.get("groups") or []:
        has_same_parent = any(
            classify_molecule_relation(query_identity, identity_from_record(neighbor))
            is MoleculeRelation.SAME_PARENT
            for neighbor in group.get("neighbors") or []
        )
        if has_same_parent:
            affected.append(str(group.get("group_id") or ""))
    return sorted(affected)


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
        "",
        "| 实验 | 总数 | 变化 | 复用 |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["experiments"]:
        lines.append(f"| {row['experiment']} | {row['n_total']} | {row['n_changed']} | {row['n_reused']} |")
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="*", default=[])
    parser.add_argument("--split", choices=("test", "valid"), default="test")
    parser.add_argument("--paper-root", default="")
    parser.add_argument("--operational-root", default="")
    parser.add_argument("--target-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--materialize", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
