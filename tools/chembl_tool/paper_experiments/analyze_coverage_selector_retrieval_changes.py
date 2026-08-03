"""Analyze matched retrieval and trace changes for the coverage selector matrix."""

from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .summarize_coverage_selector_llm_matrix import (
    CONDITIONS,
    prediction_paths,
    read_predictions,
    run_paths,
    safe_ratio,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT / "outputs/paper/coverage_selector_llm/analysis"
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=False,
    useBondTypes=True,
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_group_outputs(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                group_id = str(row["group_id"])
                if group_id in rows:
                    raise ValueError(f"{path}: duplicate group_id={group_id}")
                rows[group_id] = row
    return rows


def final_input_signature(path: Path) -> str:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{path}: empty trace")
    messages = rows[-1].get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{path}: final trace has no serialized messages")
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"{path}: final trace does not end with the assistant response")
    # The serialized trace appends the generated assistant answer to the input messages.
    return json.dumps(messages[:-1], sort_keys=True, ensure_ascii=False)


def neighbor_identity(row: dict[str, Any]) -> str:
    identity = (
        row.get("molecule_chembl_id")
        or row.get("standard_inchi_key")
        or row.get("canonical_smiles")
    )
    if not identity:
        raise ValueError("Neighbor row has no stable molecule identity")
    return str(identity)


@lru_cache(maxsize=None)
def fingerprint(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    return MORGAN_GENERATOR.GetFingerprint(molecule)


def query_feature_coverage(query_smiles: str, neighbors: list[dict[str, Any]]) -> float | None:
    query_bits = set(fingerprint(query_smiles).GetOnBits())
    if not query_bits or not neighbors:
        return None
    union_bits: set[int] = set()
    for neighbor in neighbors:
        union_bits.update(fingerprint(str(neighbor["canonical_smiles"])).GetOnBits())
    return len(query_bits & union_bits) / len(query_bits)


def mean_pairwise_similarity(neighbors: list[dict[str, Any]]) -> float | None:
    if len(neighbors) < 2:
        return None
    fps = [fingerprint(str(neighbor["canonical_smiles"])) for neighbor in neighbors]
    return mean(
        DataStructs.TanimotoSimilarity(left, right)
        for left, right in combinations(fps, 2)
    )


def branch_signature(row: dict[str, Any] | None) -> str:
    if row is None:
        return ""
    return json.dumps(row.get("llm", {}).get("content"), sort_keys=True, ensure_ascii=False)


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def analyze_condition(
    split: str,
    task: str,
    task_label: str,
    expected_n: int,
) -> dict[str, Any]:
    control_predictions_path, coverage_predictions_path = prediction_paths(split, task)
    control_predictions = read_predictions(control_predictions_path, expected_n=expected_n)
    coverage_predictions = read_predictions(coverage_predictions_path, expected_n=expected_n)
    control_root, coverage_root, control_prefix, coverage_prefix = run_paths(split, task)

    query_any_list_change = 0
    query_any_set_replacement = 0
    query_reorder_only = 0
    query_any_top1_change = 0
    query_any_top1_new = 0
    query_any_top1_reordered = 0
    query_any_branch_change = 0
    query_replacements: list[int] = []
    changed_query_replacements: list[int] = []
    group_total = 0
    group_nonempty = 0
    group_list_changed = 0
    group_set_changed = 0
    group_top1_changed = 0
    group_top1_new = 0
    group_top1_reordered = 0
    group_branch_changed = 0
    changed_group_branch_changed = 0
    total_slots = 0
    ranked_slot_changes = 0
    total_replacements = 0
    prediction_flips = 0
    confidence_changes = 0
    exact_same_final_input = 0
    identical_retrieval_prediction_flips = 0
    exact_same_final_input_prediction_flips = 0
    set_changed_morgan_only_correct = 0
    set_changed_coverage_only_correct = 0
    control_similarities: list[float] = []
    coverage_similarities: list[float] = []
    control_top1_similarities: list[float] = []
    coverage_top1_similarities: list[float] = []
    changed_top1_similarity_deltas: list[float] = []
    control_coverages: list[float] = []
    coverage_coverages: list[float] = []
    coverage_deltas: list[float] = []
    control_pairwise: list[float] = []
    coverage_pairwise: list[float] = []
    pairwise_deltas: list[float] = []

    for index in range(expected_n):
        control_dir = control_root / f"{control_prefix}_idx{index:05d}"
        coverage_dir = coverage_root / f"{coverage_prefix}_idx{index:05d}"
        control_retrieval = load_json(control_dir / "retrieval.json")
        coverage_retrieval = load_json(coverage_dir / "retrieval.json")
        control_groups = {group["group_id"]: group for group in control_retrieval["groups"]}
        coverage_groups = {group["group_id"]: group for group in coverage_retrieval["groups"]}
        if set(control_groups) != set(coverage_groups):
            raise ValueError(f"{task}/{split}/{index}: group mismatch")
        control_query = str(control_retrieval["query"]["canonical_smiles"])
        coverage_query = str(coverage_retrieval["query"]["canonical_smiles"])
        if control_query != coverage_query:
            raise ValueError(f"{task}/{split}/{index}: query mismatch")

        control_branches = load_group_outputs(control_dir / "group_reasoning_outputs.jsonl")
        coverage_branches = load_group_outputs(coverage_dir / "group_reasoning_outputs.jsonl")
        any_list_change = False
        any_set_change = False
        any_top1_change = False
        any_top1_new = False
        any_top1_reordered = False
        any_branch_change = False
        replacements_for_query = 0

        for group_id in sorted(control_groups):
            group_total += 1
            control_neighbors = control_groups[group_id]["neighbors"]
            coverage_neighbors = coverage_groups[group_id]["neighbors"]
            control_ids = [neighbor_identity(neighbor) for neighbor in control_neighbors]
            coverage_ids = [neighbor_identity(neighbor) for neighbor in coverage_neighbors]
            if len(control_ids) != len(set(control_ids)):
                raise ValueError(f"{task}/{split}/{index}/{group_id}: duplicate Morgan neighbor")
            if len(coverage_ids) != len(set(coverage_ids)):
                raise ValueError(f"{task}/{split}/{index}/{group_id}: duplicate coverage neighbor")
            if control_neighbors or coverage_neighbors:
                group_nonempty += 1
            total_slots += max(len(control_ids), len(coverage_ids))
            slot_changes = sum(
                (control_ids[position] if position < len(control_ids) else None)
                != (coverage_ids[position] if position < len(coverage_ids) else None)
                for position in range(max(len(control_ids), len(coverage_ids)))
            )
            ranked_slot_changes += slot_changes
            list_changed = control_ids != coverage_ids
            set_replacements = len(set(coverage_ids) - set(control_ids))
            set_changed = set(control_ids) != set(coverage_ids)
            top1_changed = (control_ids[0] if control_ids else None) != (
                coverage_ids[0] if coverage_ids else None
            )
            coverage_top1 = coverage_ids[0] if coverage_ids else None
            top1_new = bool(top1_changed and coverage_top1 not in set(control_ids))
            top1_reordered = bool(top1_changed and coverage_top1 in set(control_ids))
            branch_changed = branch_signature(control_branches.get(group_id)) != branch_signature(
                coverage_branches.get(group_id)
            )
            group_list_changed += int(list_changed)
            group_set_changed += int(set_changed)
            group_top1_changed += int(top1_changed)
            group_top1_new += int(top1_new)
            group_top1_reordered += int(top1_reordered)
            group_branch_changed += int(branch_changed)
            changed_group_branch_changed += int(list_changed and branch_changed)
            total_replacements += set_replacements
            replacements_for_query += set_replacements
            any_list_change |= list_changed
            any_set_change |= set_changed
            any_top1_change |= top1_changed
            any_top1_new |= top1_new
            any_top1_reordered |= top1_reordered
            any_branch_change |= branch_changed

            control_similarities.extend(float(row["similarity"]) for row in control_neighbors)
            coverage_similarities.extend(float(row["similarity"]) for row in coverage_neighbors)
            if control_neighbors:
                control_top1_similarities.append(float(control_neighbors[0]["similarity"]))
            if coverage_neighbors:
                coverage_top1_similarities.append(float(coverage_neighbors[0]["similarity"]))
            if top1_changed and control_neighbors and coverage_neighbors:
                changed_top1_similarity_deltas.append(
                    float(coverage_neighbors[0]["similarity"])
                    - float(control_neighbors[0]["similarity"])
                )

            control_feature_coverage = query_feature_coverage(control_query, control_neighbors)
            coverage_feature_coverage = query_feature_coverage(control_query, coverage_neighbors)
            if control_feature_coverage is not None and coverage_feature_coverage is not None:
                control_coverages.append(control_feature_coverage)
                coverage_coverages.append(coverage_feature_coverage)
                coverage_deltas.append(coverage_feature_coverage - control_feature_coverage)
            control_diversity = mean_pairwise_similarity(control_neighbors)
            coverage_diversity = mean_pairwise_similarity(coverage_neighbors)
            if control_diversity is not None and coverage_diversity is not None:
                control_pairwise.append(control_diversity)
                coverage_pairwise.append(coverage_diversity)
                pairwise_deltas.append(coverage_diversity - control_diversity)

        query_any_list_change += int(any_list_change)
        query_any_set_replacement += int(any_set_change)
        query_reorder_only += int(any_list_change and not any_set_change)
        query_any_top1_change += int(any_top1_change)
        query_any_top1_new += int(any_top1_new)
        query_any_top1_reordered += int(any_top1_reordered)
        query_any_branch_change += int(any_branch_change)
        query_replacements.append(replacements_for_query)
        if any_set_change:
            changed_query_replacements.append(replacements_for_query)
        prediction_flip = (
            control_predictions[index]["pred_label"] != coverage_predictions[index]["pred_label"]
        )
        prediction_flips += int(prediction_flip)
        confidence_changes += int(
            control_predictions[index].get("confidence") != coverage_predictions[index].get("confidence")
        )
        if not any_list_change:
            identical_retrieval_prediction_flips += int(prediction_flip)
            same_final_input = final_input_signature(
                control_dir / "trace_messages.jsonl"
            ) == final_input_signature(coverage_dir / "trace_messages.jsonl")
            exact_same_final_input += int(same_final_input)
            exact_same_final_input_prediction_flips += int(same_final_input and prediction_flip)
        if any_set_change:
            control_correct = bool(control_predictions[index]["correct"])
            coverage_correct = bool(coverage_predictions[index]["correct"])
            set_changed_morgan_only_correct += int(control_correct and not coverage_correct)
            set_changed_coverage_only_correct += int(coverage_correct and not control_correct)

    return {
        "split": split,
        "task": task,
        "task_label": task_label,
        "n_queries": expected_n,
        "query_level": {
            "any_ranked_list_change": query_any_list_change,
            "any_ranked_list_change_rate": safe_ratio(query_any_list_change, expected_n),
            "any_set_replacement": query_any_set_replacement,
            "any_set_replacement_rate": safe_ratio(query_any_set_replacement, expected_n),
            "reorder_only": query_reorder_only,
            "any_top1_change": query_any_top1_change,
            "any_top1_change_rate": safe_ratio(query_any_top1_change, expected_n),
            "any_top1_new": query_any_top1_new,
            "any_top1_new_rate": safe_ratio(query_any_top1_new, expected_n),
            "any_top1_reordered": query_any_top1_reordered,
            "any_top1_reordered_rate": safe_ratio(query_any_top1_reordered, expected_n),
            "mean_replacements_all_queries": safe_mean(query_replacements),
            "mean_replacements_changed_queries": safe_mean(changed_query_replacements),
            "any_group_reasoning_change": query_any_branch_change,
            "prediction_flips": prediction_flips,
            "confidence_changes": confidence_changes,
            "identical_ranked_retrieval": expected_n - query_any_list_change,
            "exact_same_final_input": exact_same_final_input,
            "prediction_flips_with_identical_retrieval": identical_retrieval_prediction_flips,
            "prediction_flips_with_exact_same_final_input": exact_same_final_input_prediction_flips,
            "set_changed_morgan_only_correct": set_changed_morgan_only_correct,
            "set_changed_coverage_only_correct": set_changed_coverage_only_correct,
        },
        "group_and_slot_level": {
            "n_groups_total": group_total,
            "n_groups_nonempty": group_nonempty,
            "ranked_list_changed": group_list_changed,
            "ranked_list_changed_rate_nonempty": safe_ratio(group_list_changed, group_nonempty),
            "set_changed": group_set_changed,
            "set_changed_rate_nonempty": safe_ratio(group_set_changed, group_nonempty),
            "top1_changed": group_top1_changed,
            "top1_changed_rate_nonempty": safe_ratio(group_top1_changed, group_nonempty),
            "top1_new": group_top1_new,
            "top1_reordered": group_top1_reordered,
            "n_neighbor_slots": total_slots,
            "ranked_slot_changes": ranked_slot_changes,
            "ranked_slot_change_rate": safe_ratio(ranked_slot_changes, total_slots),
            "new_neighbors_selected": total_replacements,
            "mean_new_neighbors_per_nonempty_group": safe_ratio(total_replacements, group_nonempty),
            "group_reasoning_changed": group_branch_changed,
            "changed_retrieval_group_reasoning_changed": changed_group_branch_changed,
        },
        "structure_effect": {
            "mean_query_feature_coverage_morgan": safe_mean(control_coverages),
            "mean_query_feature_coverage_coverage": safe_mean(coverage_coverages),
            "mean_query_feature_coverage_delta": safe_mean(coverage_deltas),
            "mean_neighbor_pairwise_tanimoto_morgan": safe_mean(control_pairwise),
            "mean_neighbor_pairwise_tanimoto_coverage": safe_mean(coverage_pairwise),
            "mean_neighbor_pairwise_tanimoto_delta": safe_mean(pairwise_deltas),
            "mean_query_neighbor_similarity_morgan": safe_mean(control_similarities),
            "mean_query_neighbor_similarity_coverage": safe_mean(coverage_similarities),
            "mean_top1_similarity_morgan": safe_mean(control_top1_similarities),
            "mean_top1_similarity_coverage": safe_mean(coverage_top1_similarities),
            "mean_top1_similarity_delta_when_changed": safe_mean(changed_top1_similarity_deltas),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    results = [
        analyze_condition(split, task, task_label, expected_n)
        for split, task, task_label, expected_n in CONDITIONS
    ]
    total_queries = sum(result["n_queries"] for result in results)
    total_queries_changed = sum(
        result["query_level"]["any_set_replacement"] for result in results
    )
    total_replacements = sum(
        result["group_and_slot_level"]["new_neighbors_selected"] for result in results
    )
    total_nonempty_groups = sum(
        result["group_and_slot_level"]["n_groups_nonempty"] for result in results
    )
    aggregate = {
        "n_queries": total_queries,
        "queries_with_actual_replacement": total_queries_changed,
        "queries_with_actual_replacement_rate": safe_ratio(total_queries_changed, total_queries),
        "queries_with_ranked_list_change": sum(
            result["query_level"]["any_ranked_list_change"] for result in results
        ),
        "reorder_only_queries": sum(
            result["query_level"]["reorder_only"] for result in results
        ),
        "queries_with_any_top1_change": sum(
            result["query_level"]["any_top1_change"] for result in results
        ),
        "queries_with_any_top1_new": sum(
            result["query_level"]["any_top1_new"] for result in results
        ),
        "queries_with_any_top1_reordered": sum(
            result["query_level"]["any_top1_reordered"] for result in results
        ),
        "total_new_neighbors_selected": total_replacements,
        "mean_replacements_per_query": safe_ratio(total_replacements, total_queries),
        "mean_replacements_per_changed_query": safe_ratio(
            total_replacements, total_queries_changed
        ),
        "mean_replacements_per_nonempty_group": safe_ratio(
            total_replacements, total_nonempty_groups
        ),
        "prediction_flips": sum(
            result["query_level"]["prediction_flips"] for result in results
        ),
        "identical_ranked_retrieval": sum(
            result["query_level"]["identical_ranked_retrieval"] for result in results
        ),
        "exact_same_final_input": sum(
            result["query_level"]["exact_same_final_input"] for result in results
        ),
        "prediction_flips_with_identical_retrieval": sum(
            result["query_level"]["prediction_flips_with_identical_retrieval"]
            for result in results
        ),
        "prediction_flips_with_exact_same_final_input": sum(
            result["query_level"]["prediction_flips_with_exact_same_final_input"]
            for result in results
        ),
        "set_changed_morgan_only_correct": sum(
            result["query_level"]["set_changed_morgan_only_correct"] for result in results
        ),
        "set_changed_coverage_only_correct": sum(
            result["query_level"]["set_changed_coverage_only_correct"] for result in results
        ),
    }
    payload = {
        "unit": "query x mechanism group; one replacement is one newly selected molecule relative to the matched Morgan neighbor set",
        "aggregate": aggregate,
        "conditions": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "retrieval_change_analysis.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    tsv_path = args.output_dir / "retrieval_change_analysis.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "task", "split", "n_queries", "queries_changed", "queries_changed_pct",
            "queries_top1_changed", "queries_top1_changed_pct", "mean_replacements_all_queries",
            "mean_replacements_changed_queries", "groups_changed", "groups_top1_changed",
            "ranked_slot_change_pct", "feature_coverage_delta", "neighbor_pairwise_tanimoto_delta",
            "query_neighbor_similarity_delta", "prediction_flips", "confidence_changes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for result in results:
            query = result["query_level"]
            groups = result["group_and_slot_level"]
            structure = result["structure_effect"]
            writer.writerow({
                "task": result["task"], "split": result["split"], "n_queries": result["n_queries"],
                "queries_changed": query["any_set_replacement"],
                "queries_changed_pct": f"{100 * query['any_set_replacement_rate']:.2f}",
                "queries_top1_changed": query["any_top1_change"],
                "queries_top1_changed_pct": f"{100 * query['any_top1_change_rate']:.2f}",
                "mean_replacements_all_queries": f"{query['mean_replacements_all_queries']:.4f}",
                "mean_replacements_changed_queries": f"{query['mean_replacements_changed_queries']:.4f}",
                "groups_changed": groups["set_changed"], "groups_top1_changed": groups["top1_changed"],
                "ranked_slot_change_pct": f"{100 * groups['ranked_slot_change_rate']:.2f}",
                "feature_coverage_delta": f"{structure['mean_query_feature_coverage_delta']:.6f}",
                "neighbor_pairwise_tanimoto_delta": f"{structure['mean_neighbor_pairwise_tanimoto_delta']:.6f}",
                "query_neighbor_similarity_delta": f"{structure['mean_query_neighbor_similarity_coverage'] - structure['mean_query_neighbor_similarity_morgan']:.6f}",
                "prediction_flips": query["prediction_flips"], "confidence_changes": query["confidence_changes"],
            })

    report_path = args.output_dir / "retrieval_change_report.md"
    lines = [
        "# Coverage selector 检索变化审计", "",
        "计数单位：同一个 query 的同一个 mechanism group；replacement 指 coverage set 相比 Morgan set 新出现一个 molecule。", "",
        (
            f"全体：{total_queries_changed}/{total_queries} "
            f"({100 * aggregate['queries_with_actual_replacement_rate']:.1f}%) query 实际换了 neighbor；"
            f"{aggregate['queries_with_any_top1_change']}/{total_queries} "
            f"({100 * safe_ratio(aggregate['queries_with_any_top1_change'], total_queries):.1f}%) "
            "至少一个 group 的 top-1 改变；"
            f"平均每个 query 替换 {aggregate['mean_replacements_per_query']:.2f} 个 neighbor。"
        ), "",
        (
            f"Trace audit：{aggregate['prediction_flips_with_exact_same_final_input']} 个 prediction flip "
            f"发生在 final LLM 输入完全相同的 query，属于模型调用波动，不能归因于 retrieval。"
        ), "",
        "| 任务 | Split | 实际换过 neighbor 的 query | top-1 有变化 | 平均替换数/query | 平均替换数/变化 query | prediction flips |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        query = result["query_level"]
        lines.append(
            f"| {result['task_label']} | {result['split']} | {query['any_set_replacement']}/{result['n_queries']} "
            f"({100*query['any_set_replacement_rate']:.1f}%) | {query['any_top1_change']}/{result['n_queries']} "
            f"({100*query['any_top1_change_rate']:.1f}%) | {query['mean_replacements_all_queries']:.2f} | "
            f"{query['mean_replacements_changed_queries']:.2f} | {query['prediction_flips']} |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json_path)
    print(tsv_path)
    print(report_path)


if __name__ == "__main__":
    main()
