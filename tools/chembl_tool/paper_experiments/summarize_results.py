"""Aggregate the frozen molecular-evidence-agent experiments for reporting."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import random
import re
from typing import Any

from tools.chembl_tool.common.evidence_contract import evidence_for_llm
from tools.chembl_tool.common.identity_blind import find_identity_blind_leaks

from .molecular_evidence_agent import (
    DEPLOYMENT_VISIBLE,
    DEPLOYMENT_VISIBLE_PREFETCHED,
    EXPERIMENTS,
    IDENTITY_BLIND,
    PAPER_ROOT,
    VISIBILITY_MODES,
    experiment_result_name,
    experiment_run_root,
)


COMPARISONS = {
    "bbb_martins": [
        ("none", "chembl_direct"),
        ("chembl_direct", "chembl_full_flat"),
        ("chembl_full_flat", "chembl_full_mechanism"),
        ("chembl_direct", "starling_direct"),
    ],
    "skin_reaction": [
        ("none", "chembl_direct"),
        ("chembl_direct", "chembl_full_flat"),
        ("chembl_full_flat", "chembl_full_mechanism"),
    ],
    "clintox": [
        ("none", "chembl_direct"),
        ("chembl_direct", "chembl_full_flat"),
        ("chembl_full_flat", "chembl_full_mechanism"),
    ],
    "bioavailability_ma": [
        ("none", "chembl_direct"),
        ("none", "starling_direct_scalar_knn"),
        ("starling_direct_scalar_knn", "starling_direct_numeric"),
        ("chembl_direct", "starling_direct_numeric"),
        ("chembl_direct", "starling_direct_full"),
        ("starling_direct_numeric", "starling_direct_full"),
        ("chembl_direct", "chembl_full_flat"),
        ("chembl_full_flat", "chembl_full_mechanism"),
        ("starling_direct_full", "starling_full_flat"),
        ("starling_full_flat", "starling_full_mechanism"),
        ("chembl_full_mechanism", "starling_full_mechanism"),
    ],
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    group_coverage: list[dict[str, Any]] = []
    prediction_sets: dict[str, dict[int, dict[str, Any]]] = {}

    for visibility_mode in VISIBILITY_MODES:
        for experiment in EXPERIMENTS:
            batch_dir = experiment_run_root(visibility_mode) / experiment.task / experiment.name
            metrics_path = batch_dir / "metrics.json"
            predictions_path = batch_dir / "predictions.jsonl"
            if not metrics_path.exists() or not predictions_path.exists():
                continue
            result_name = experiment_result_name(experiment.name, visibility_mode)
            predictions = _read_jsonl(predictions_path)
            prediction_sets[result_name] = {
                int(row["query_index"]): row for row in predictions if row.get("pred_label") is not None
            }
            group_coverage.extend(
                _group_coverage_rows(result_name, batch_dir, predictions, visibility_mode=visibility_mode)
            )
            summaries.append(
                summarize_experiment(
                    result_name,
                    experiment.task,
                    batch_dir,
                    json.loads(metrics_path.read_text(encoding="utf-8")),
                    predictions,
                    bootstrap_replicates=args.bootstrap_replicates,
                    visibility_mode=visibility_mode,
                )
            )

    knn_dir = PAPER_ROOT / "bioavailability_ma" / "starling_direct_scalar_knn"
    if (knn_dir / "metrics.json").exists() and (knn_dir / "predictions.jsonl").exists():
        knn_predictions = _read_jsonl(knn_dir / "predictions.jsonl")
        normalized_knn = [
            {
                **row,
                "pred_label": row.get("predicted_label"),
                "n_groups_with_neighbors": int(not row.get("used_fallback")),
            }
            for row in knn_predictions
        ]
        knn_name = "bioavailability_ma__starling_direct_scalar_knn"
        prediction_sets[knn_name] = {int(row["query_index"]): row for row in normalized_knn}
        group_coverage.append(
            {
                "experiment": knn_name,
                "group_id": "Direct.scalar_knn",
                "n_samples": len(normalized_knn),
                "n_samples_with_neighbors": sum(not row.get("used_fallback") for row in normalized_knn),
                "coverage": sum(not row.get("used_fallback") for row in normalized_knn) / len(normalized_knn),
                "mean_neighbors": sum(int(row.get("n_neighbors") or 0) for row in normalized_knn)
                / len(normalized_knn),
            }
        )
        summaries.append(
            summarize_experiment(
                knn_name,
                "bioavailability_ma",
                knn_dir,
                json.loads((knn_dir / "metrics.json").read_text(encoding="utf-8")),
                normalized_knn,
                bootstrap_replicates=args.bootstrap_replicates,
                visibility_mode="visibility_independent",
            )
        )

    comparisons = []
    for visibility_mode in VISIBILITY_MODES:
        comparisons.extend(
            build_comparisons(
                prediction_sets,
                args.bootstrap_replicates,
                visibility_mode=visibility_mode,
            )
        )
    visibility_comparisons = build_visibility_comparisons(
        prediction_sets,
        args.bootstrap_replicates,
    )
    source_inventory = build_source_inventory()
    contextual_baselines = load_contextual_baselines()
    _write_tsv(output_dir / "experiment_summary.tsv", summaries)
    _write_tsv(output_dir / "paired_comparisons.tsv", comparisons)
    _write_tsv(output_dir / "visibility_comparisons.tsv", visibility_comparisons)
    _write_tsv(output_dir / "group_coverage.tsv", group_coverage)
    _write_tsv(output_dir / "source_inventory.tsv", source_inventory)
    _write_tsv(output_dir / "contextual_baselines.tsv", contextual_baselines)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "experiments": summaries,
                "source_inventory": source_inventory,
                "contextual_baselines": contextual_baselines,
                "group_coverage": group_coverage,
                "comparisons": comparisons,
                "visibility_comparisons": visibility_comparisons,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        _render_report(summaries, comparisons, visibility_comparisons, contextual_baselines),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(summaries)} experiment summaries, {len(comparisons)} within-regime comparisons, "
        f"and {len(visibility_comparisons)} visibility comparisons to {output_dir}"
    )
    return 0


def summarize_experiment(
    name: str,
    task: str,
    batch_dir: Path,
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    bootstrap_replicates: int,
    visibility_mode: str = IDENTITY_BLIND,
) -> dict[str, Any]:
    evaluable = [row for row in predictions if row.get("pred_label") is not None]
    labels = [int(row["label"]) for row in evaluable]
    predicted = [int(row["pred_label"]) for row in evaluable]
    macro_low, macro_high = bootstrap_metric_ci(labels, predicted, bootstrap_replicates)
    group_counts = [int(row.get("n_groups_with_neighbors") or 0) for row in predictions]
    usage = _collect_usage(batch_dir, predictions)
    trace_matches = _count_identity_leaks(predictions)
    prompt_audit = _audit_prompt_identities(batch_dir, predictions)
    deployment_audit = _audit_deployment_visibility(batch_dir, predictions)
    identity_blind = visibility_mode == IDENTITY_BLIND
    if identity_blind:
        visibility_contract_satisfied: bool | None = prompt_audit["prompt_identity_leak_runs"] == 0
    elif visibility_mode in {DEPLOYMENT_VISIBLE, DEPLOYMENT_VISIBLE_PREFETCHED}:
        visibility_contract_satisfied = (
            deployment_audit["deployment_visibility_audited_runs"] == len(predictions)
            and deployment_audit["deployment_contract_failed_runs"] == 0
        )
    else:
        visibility_contract_satisfied = None
    return {
        "experiment": name,
        "task": task,
        "visibility_mode": visibility_mode,
        "identity_blind": identity_blind,
        "n_total": metrics.get("n_total", len(predictions)),
        "n_successful": metrics.get("n_successful", len(evaluable)),
        "n_failed": metrics.get("n_failed_runs", len(predictions) - len(evaluable)),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "macro_f1_ci_low": macro_low,
        "macro_f1_ci_high": macro_high,
        "tn": metrics.get("confusion_matrix", {}).get("tn"),
        "fp": metrics.get("confusion_matrix", {}).get("fp"),
        "fn": metrics.get("confusion_matrix", {}).get("fn"),
        "tp": metrics.get("confusion_matrix", {}).get("tp"),
        "retrieval_coverage": (
            sum(count > 0 for count in group_counts) / len(group_counts) if group_counts else None
        ),
        "mean_groups_with_neighbors": (
            sum(group_counts) / len(group_counts) if group_counts else None
        ),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "retried_calls": usage["retried_calls"],
        "llm_calls": usage["llm_calls"],
        "reused_single_analyses": usage["reused_single_analyses"],
        "served_models": usage["served_models"],
        "query_smiles_trace_matches": trace_matches,
        "query_smiles_trace_leaks": trace_matches if identity_blind else 0,
        "visibility_contract_satisfied": visibility_contract_satisfied,
        **prompt_audit,
        **deployment_audit,
    }


def build_source_inventory() -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for experiment in EXPERIMENTS:
        if experiment.mode == "none":
            continue
        item = grouped.setdefault(
            experiment.index,
            {"tasks": set(), "sources": set(), "experiments": []},
        )
        item["tasks"].add(experiment.task)
        item["sources"].add(experiment.source)
        item["experiments"].append(experiment.name)
    rows = []
    for index_path, item in grouped.items():
        meta_path = Path(index_path).with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        direct_stats = meta.get("direct_source_stats") or {}
        groups = meta.get("groups") or []
        rows.append(
            {
                "index": index_path,
                "meta": str(meta_path),
                "tasks": ";".join(sorted(item["tasks"])),
                "sources": ";".join(sorted(item["sources"])),
                "experiments": ";".join(sorted(item["experiments"])),
                "scope": meta.get("scope", ""),
                "evidence_content": meta.get("evidence_content", ""),
                "n_evidence_rows": meta.get("n_evidence_rows"),
                "n_index_molecules": meta.get("n_index_molecules"),
                "n_groups": meta.get("n_groups", len(groups) if groups else None),
                "n_molecules_with_numeric_evidence": direct_stats.get(
                    "n_molecules_with_numeric_evidence",
                    meta.get("n_molecules_with_numerical_evidence"),
                ),
                "n_molecules_with_qualitative_evidence": direct_stats.get(
                    "n_molecules_with_qualitative_evidence",
                    meta.get("n_molecules_with_qualitative_evidence"),
                ),
                "n_molecules_with_qualitative_only_evidence": direct_stats.get(
                    "n_molecules_with_qualitative_only_evidence"
                ),
            }
        )
    return sorted(rows, key=lambda row: (row["tasks"], row["sources"], row["index"]))


def load_contextual_baselines() -> list[dict[str, Any]]:
    rows = []
    fixed = {
        "bbb_martins": Path("outputs/baselines/minimol/bbb_martins/metrics.json"),
        "bioavailability_ma": Path("outputs/baselines/minimol/bioavailability_ma/metrics.json"),
    }
    for task, path in fixed.items():
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        test = data.get("test_metrics_fixed_0.5") or data.get("test_metrics") or {}
        rows.append(
            {
                "task": task,
                "baseline": "MiniMol",
                "selection": "published_or_fixed_task_config",
                "macro_f1": test.get("macro_f1"),
                "accuracy": test.get("accuracy"),
                "auroc": test.get("auroc"),
                "artifact": str(path),
            }
        )
    for task in ("clintox", "skin_reaction"):
        path = Path(f"outputs/baselines/minimol_sweeps_gpu/{task}/sweep_summary.json")
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        test = (data.get("best") or {}).get("test") or {}
        rows.append(
            {
                "task": task,
                "baseline": "MiniMol",
                "selection": str(data.get("selection_metric") or "validation_selected"),
                "macro_f1": test.get("macro_f1"),
                "accuracy": test.get("accuracy"),
                "auroc": test.get("auroc"),
                "artifact": str(path),
            }
        )
    return sorted(rows, key=lambda row: row["task"])


def build_comparisons(
    prediction_sets: dict[str, dict[int, dict[str, Any]]],
    bootstrap_replicates: int,
    *,
    visibility_mode: str = IDENTITY_BLIND,
) -> list[dict[str, Any]]:
    rows = []
    for task, task_pairs in COMPARISONS.items():
        for left_suffix, right_suffix in task_pairs:
            left_name = _visibility_condition_name(f"{task}__{left_suffix}", visibility_mode)
            right_name = _visibility_condition_name(f"{task}__{right_suffix}", visibility_mode)
            row = _paired_comparison(
                task,
                left_name,
                right_name,
                prediction_sets,
                bootstrap_replicates,
            )
            if row is None:
                continue
            row["comparison_type"] = "within_visibility_regime"
            row["visibility_mode"] = visibility_mode
            rows.append(row)
    _add_holm_adjusted_p(rows)
    return rows


def build_visibility_comparisons(
    prediction_sets: dict[str, dict[int, dict[str, Any]]],
    bootstrap_replicates: int,
) -> list[dict[str, Any]]:
    """Build controlled visibility and agentic tool-use comparisons."""
    rows = []
    comparison_specs = (
        (
            IDENTITY_BLIND,
            DEPLOYMENT_VISIBLE_PREFETCHED,
            "identity_blind_vs_deployment_visible_prefetched",
        ),
        (
            DEPLOYMENT_VISIBLE_PREFETCHED,
            DEPLOYMENT_VISIBLE,
            "deployment_visible_prefetched_vs_agentic",
        ),
    )
    for left_mode, right_mode, comparison_type in comparison_specs:
        family = []
        for experiment in EXPERIMENTS:
            left_name = experiment_result_name(experiment.name, left_mode)
            right_name = experiment_result_name(experiment.name, right_mode)
            row = _paired_comparison(
                experiment.task,
                left_name,
                right_name,
                prediction_sets,
                bootstrap_replicates,
            )
            if row is None:
                continue
            row["condition"] = experiment.name
            row["comparison_type"] = comparison_type
            row["visibility_mode"] = "cross_visibility"
            family.append(row)
        _add_holm_adjusted_p(family)
        rows.extend(family)
    return rows


def _paired_comparison(
    task: str,
    left_name: str,
    right_name: str,
    prediction_sets: dict[str, dict[int, dict[str, Any]]],
    bootstrap_replicates: int,
) -> dict[str, Any] | None:
    if left_name not in prediction_sets or right_name not in prediction_sets:
        return None
    left = prediction_sets[left_name]
    right = prediction_sets[right_name]
    common = sorted(set(left) & set(right))
    labels = [int(left[index]["label"]) for index in common]
    left_pred = [int(left[index]["pred_label"]) for index in common]
    right_pred = [int(right[index]["pred_label"]) for index in common]
    left_f1 = macro_f1(labels, left_pred)
    right_f1 = macro_f1(labels, right_pred)
    delta_low, delta_high = paired_bootstrap_delta_ci(
        labels, left_pred, right_pred, bootstrap_replicates
    )
    left_only = sum(lp == y and rp != y for y, lp, rp in zip(labels, left_pred, right_pred))
    right_only = sum(lp != y and rp == y for y, lp, rp in zip(labels, left_pred, right_pred))
    return {
        "task": task,
        "left": left_name,
        "right": right_name,
        "n_paired": len(common),
        "left_macro_f1": left_f1,
        "right_macro_f1": right_f1,
        "delta_macro_f1": right_f1 - left_f1,
        "delta_ci_low": delta_low,
        "delta_ci_high": delta_high,
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "mcnemar_exact_p": mcnemar_exact_p(left_only, right_only),
    }


def _visibility_condition_name(condition_name: str, visibility_mode: str) -> str:
    if condition_name.endswith("__starling_direct_scalar_knn"):
        return condition_name
    return experiment_result_name(condition_name, visibility_mode)


def _add_holm_adjusted_p(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: float(item[1]["mcnemar_exact_p"]))
    running_max = 0.0
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, float(row["mcnemar_exact_p"]) * (len(rows) - rank))
        running_max = max(running_max, adjusted)
        rows[index]["mcnemar_holm_p"] = running_max


def _group_coverage_rows(
    experiment: str,
    batch_dir: Path,
    predictions: list[dict[str, Any]],
    *,
    visibility_mode: str = IDENTITY_BLIND,
) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, int]] = {}
    for prediction in predictions:
        run_dir = _resolve_run_dir(batch_dir, prediction)
        retrieval_path = run_dir / "retrieval.json"
        if not retrieval_path.exists():
            continue
        retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
        for group in retrieval.get("groups") or []:
            group_id = str(group.get("group_id") or "unknown")
            neighbors = group.get("neighbors") or []
            item = counts.setdefault(group_id, {"samples": 0, "covered": 0, "neighbors": 0})
            item["samples"] += 1
            item["covered"] += int(bool(neighbors))
            item["neighbors"] += len(neighbors)
    return [
        {
            "experiment": experiment,
            "visibility_mode": visibility_mode,
            "group_id": group_id,
            "n_samples": item["samples"],
            "n_samples_with_neighbors": item["covered"],
            "coverage": item["covered"] / item["samples"] if item["samples"] else None,
            "mean_neighbors": item["neighbors"] / item["samples"] if item["samples"] else None,
        }
        for group_id, item in sorted(counts.items())
    ]


def macro_f1(labels: list[int], predictions: list[int]) -> float:
    scores = []
    for label in (0, 1):
        tp = sum(y == label and p == label for y, p in zip(labels, predictions))
        fp = sum(y != label and p == label for y, p in zip(labels, predictions))
        fn = sum(y == label and p != label for y, p in zip(labels, predictions))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return sum(scores) / 2


def bootstrap_metric_ci(
    labels: list[int], predictions: list[int], replicates: int, *, seed: int = 17
) -> tuple[float | None, float | None]:
    if not labels or replicates <= 0:
        return None, None
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        sampled = [rng.randrange(len(labels)) for _ in labels]
        values.append(macro_f1([labels[i] for i in sampled], [predictions[i] for i in sampled]))
    return _percentile(values, 0.025), _percentile(values, 0.975)


def paired_bootstrap_delta_ci(
    labels: list[int], left: list[int], right: list[int], replicates: int, *, seed: int = 23
) -> tuple[float | None, float | None]:
    if not labels or replicates <= 0:
        return None, None
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        sampled = [rng.randrange(len(labels)) for _ in labels]
        sampled_labels = [labels[i] for i in sampled]
        values.append(
            macro_f1(sampled_labels, [right[i] for i in sampled])
            - macro_f1(sampled_labels, [left[i] for i in sampled])
        )
    return _percentile(values, 0.025), _percentile(values, 0.975)


def mcnemar_exact_p(left_only_correct: int, right_only_correct: int) -> float:
    discordant = left_only_correct + right_only_correct
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(left_only_correct, right_only_correct) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _collect_usage(batch_dir: Path, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "retried_calls": 0,
        "llm_calls": 0,
        "reused_single_analyses": 0,
    }
    models: Counter[str] = Counter()
    for prediction in predictions:
        run_dir = _resolve_run_dir(batch_dir, prediction)
        paths = [run_dir / "single_molecule_reasoning_output.json", run_dir / "final_reasoning_output.json"]
        calls: list[dict[str, Any]] = []
        for path in paths:
            if path.exists():
                output = json.loads(path.read_text(encoding="utf-8"))
                if output.get("reused_from"):
                    result["reused_single_analyses"] += 1
                else:
                    calls.append(output.get("llm", {}))
        group_path = run_dir / "group_reasoning_outputs.jsonl"
        if group_path.exists():
            calls.extend(row.get("llm", {}) for row in _read_jsonl(group_path))
        for call in calls:
            if call:
                result["llm_calls"] += 1
            if call.get("model"):
                models[str(call["model"])] += 1
            usage = call.get("usage") or {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                result[key] += int(usage.get(key) or 0)
            result["retried_calls"] += int(bool((call.get("structured_output_validation") or {}).get("retried")))
    result["served_models"] = "; ".join(f"{model}:{count}" for model, count in sorted(models.items()))
    return result


def _resolve_run_dir(batch_dir: Path, prediction: dict[str, Any]) -> Path:
    run_dir_text = str(prediction.get("run_dir") or "")
    run_dir = Path(run_dir_text) if run_dir_text else Path("__missing_run_dir__")
    if not run_dir.is_absolute() and not run_dir.exists():
        run_dir = batch_dir / "runs" / str(prediction.get("run_id", ""))
    return run_dir


def _count_identity_leaks(predictions: list[dict[str, Any]]) -> int:
    leaks = 0
    for prediction in predictions:
        smiles = str(prediction.get("smiles") or "")
        trace_path = Path(prediction.get("trace_messages") or "")
        pattern = rf"(?<![A-Za-z0-9]){re.escape(smiles)}(?![A-Za-z0-9])"
        if smiles and trace_path.is_file() and re.search(
            pattern,
            trace_path.read_text(encoding="utf-8"),
            flags=re.IGNORECASE,
        ):
            leaks += 1
    return leaks


def _audit_prompt_identities(batch_dir: Path, predictions: list[dict[str, Any]]) -> dict[str, int]:
    """Audit actual request histories while excluding the final assistant response."""
    category_runs = {"structures": 0, "identifiers": 0, "names": 0}
    runs_with_any = 0
    audited_runs = 0
    for prediction in predictions:
        run_dir = _resolve_run_dir(batch_dir, prediction)
        retrieval_path = run_dir / "retrieval.json"
        if not retrieval_path.exists():
            continue
        retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
        payloads = _llm_prompt_payloads(run_dir)
        if not payloads:
            continue
        audited_runs += 1
        found = {category: set() for category in category_runs}
        for payload in payloads:
            leaks = find_identity_blind_leaks(retrieval, payload)
            for category in found:
                found[category].update(leaks[category])
        for category in category_runs:
            category_runs[category] += int(bool(found[category]))
        runs_with_any += int(any(found.values()))
    return {
        "prompt_identity_audited_runs": audited_runs,
        "prompt_identity_leak_runs": runs_with_any,
        "prompt_structure_leak_runs": category_runs["structures"],
        "prompt_identifier_leak_runs": category_runs["identifiers"],
        "prompt_name_leak_runs": category_runs["names"],
    }


def _audit_deployment_visibility(
    batch_dir: Path,
    predictions: list[dict[str, Any]],
) -> dict[str, int]:
    """Verify the positive deployment contract against actual request histories."""
    counts = {
        "deployment_visibility_audited_runs": 0,
        "deployment_query_structure_visible_runs": 0,
        "deployment_neighbor_structure_expected_runs": 0,
        "deployment_neighbor_structure_visible_runs": 0,
        "deployment_neighbor_identifier_expected_runs": 0,
        "deployment_neighbor_identifier_visible_runs": 0,
        "deployment_neighbor_name_expected_runs": 0,
        "deployment_neighbor_name_visible_runs": 0,
        "deployment_contract_satisfied_runs": 0,
        "deployment_contract_failed_runs": 0,
    }
    for prediction in predictions:
        run_dir = _resolve_run_dir(batch_dir, prediction)
        retrieval_path = run_dir / "retrieval.json"
        if not retrieval_path.exists():
            continue
        payloads = _llm_prompt_payloads(run_dir)
        if not payloads:
            continue
        retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
        counts["deployment_visibility_audited_runs"] += 1

        query = retrieval.get("query") or {}
        query_terms = [query.get("input_smiles"), query.get("canonical_smiles")]
        query_visible = any(_term_is_visible(term, payloads) for term in query_terms if term)
        counts["deployment_query_structure_visible_runs"] += int(query_visible)

        neighbors = [
            neighbor
            for group in retrieval.get("groups") or []
            for neighbor in group.get("neighbors") or []
        ]
        structure_expected = bool(neighbors)
        structure_visible = all(
            _term_is_visible(neighbor.get("canonical_smiles"), payloads)
            for neighbor in neighbors
            if neighbor.get("canonical_smiles")
        )
        structure_visible = structure_visible and all(
            bool(neighbor.get("canonical_smiles")) for neighbor in neighbors
        )
        counts["deployment_neighbor_structure_expected_runs"] += int(structure_expected)
        counts["deployment_neighbor_structure_visible_runs"] += int(
            structure_expected and structure_visible
        )

        identifiers = [
            str(neighbor.get("molecule_chembl_id") or "").strip()
            for neighbor in neighbors
            if str(neighbor.get("molecule_chembl_id") or "").strip()
        ]
        identifier_expected = bool(identifiers)
        identifier_visible = all(_term_is_visible(term, payloads) for term in identifiers)
        counts["deployment_neighbor_identifier_expected_runs"] += int(identifier_expected)
        counts["deployment_neighbor_identifier_visible_runs"] += int(
            identifier_expected and identifier_visible
        )

        neighbor_name_sets = [names for neighbor in neighbors if (names := _neighbor_names(neighbor))]
        name_expected = bool(neighbor_name_sets)
        name_visible = all(
            any(_term_is_visible(name, payloads) for name in names)
            for names in neighbor_name_sets
        )
        counts["deployment_neighbor_name_expected_runs"] += int(name_expected)
        counts["deployment_neighbor_name_visible_runs"] += int(name_expected and name_visible)

        contract_ok = (
            query_visible
            and (not structure_expected or structure_visible)
            and (not identifier_expected or identifier_visible)
            and (not name_expected or name_visible)
        )
        counts["deployment_contract_satisfied_runs"] += int(contract_ok)
        counts["deployment_contract_failed_runs"] += int(not contract_ok)
    return counts


def _neighbor_names(neighbor: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for row in neighbor.get("evidence_rows") or []:
        record = evidence_for_llm(row)
        names.update(str(name).strip() for name in (record.get("molecule") or {}).get("names") or [])
        for example in record.get("examples") or []:
            if isinstance(example, dict):
                names.add(str(example.get("molecule_name") or "").strip())
    return {name for name in names if name}


def _term_is_visible(term: Any, payloads: list[Any]) -> bool:
    expected = str(term or "").strip().lower()
    if not expected:
        return False
    # Prompt content often contains an embedded JSON string, where SMILES
    # stereobond backslashes are escaped a second time.
    escaped = json.dumps(expected, ensure_ascii=False)[1:-1]
    candidates = {expected, escaped}
    return any(
        candidate in value.lower()
        for payload in payloads
        for value in _string_values(payload)
        for candidate in candidates
    )


def _string_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _string_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _string_values(item)


def _llm_prompt_payloads(run_dir: Path) -> list[list[dict[str, Any]]]:
    outputs: list[dict[str, Any]] = []
    for name in ("single_molecule_reasoning_output.json", "final_reasoning_output.json"):
        path = run_dir / name
        if path.exists():
            outputs.append(json.loads(path.read_text(encoding="utf-8")))
    raw_group_path = run_dir / "group_reasoning_outputs_raw.jsonl"
    clean_group_path = run_dir / "group_reasoning_outputs.jsonl"
    group_path = raw_group_path if raw_group_path.exists() else clean_group_path
    if group_path.exists():
        outputs.extend(_read_jsonl(group_path))

    payloads = []
    for output in outputs:
        messages = list((output.get("llm") or {}).get("messages") or [])
        if not messages:
            continue
        messages = [message for message in messages if message.get("role") != "assistant"]
        if messages:
            payloads.append(messages)
    return payloads


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _render_report(
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    visibility_comparisons: list[dict[str, Any]],
    contextual_baselines: list[dict[str, Any]],
) -> str:
    lines = ["# 分子证据 Agent 实验报告", "", "## 实验汇总", ""]
    lines.append("| 可见性模式 | 实验 | N | Macro-F1 | 95% CI | 准确率 | 检索覆盖率 | 失败数 | Tokens |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summaries:
        ci = f"{_fmt(row['macro_f1_ci_low'])}-{_fmt(row['macro_f1_ci_high'])}"
        lines.append(
            f"| {row['visibility_mode']} | {row['experiment']} | {row['n_total']} | "
            f"{_fmt(row['macro_f1'])} | {ci} | "
            f"{_fmt(row['accuracy'])} | {_fmt(row['retrieval_coverage'])} | {row['n_failed']} | "
            f"{row['total_tokens']} |"
        )
    lines.extend(["", "## 配对比较", ""])
    lines.append("| 任务 | 左侧条件 | 右侧条件 | Macro-F1 差值 | 95% CI | McNemar p | Holm p |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in comparisons:
        ci = f"{_fmt(row['delta_ci_low'])}-{_fmt(row['delta_ci_high'])}"
        lines.append(
            f"| {row['task']} | {row['left']} | {row['right']} | "
            f"{_fmt(row['delta_macro_f1'])} | {ci} | {_fmt(row['mcnemar_exact_p'])} | "
            f"{_fmt(row['mcnemar_holm_p'])} |"
        )
    lines.extend(["", "## 可见性与 Agentic Tool-use 配对比较", ""])
    lines.append("| 比较 | 任务 | 条件 | 左侧 Macro-F1 | 右侧 Macro-F1 | 差值 | 95% CI | McNemar p | Holm p |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in visibility_comparisons:
        ci = f"{_fmt(row['delta_ci_low'])}-{_fmt(row['delta_ci_high'])}"
        lines.append(
            f"| {row['comparison_type']} | {row['task']} | {row['condition']} | {_fmt(row['left_macro_f1'])} | "
            f"{_fmt(row['right_macro_f1'])} | {_fmt(row['delta_macro_f1'])} | {ci} | "
            f"{_fmt(row['mcnemar_exact_p'])} | {_fmt(row['mcnemar_holm_p'])} |"
        )
    lines.extend(["", "## MiniMol 参考基线", ""])
    lines.append("| 任务 | Macro-F1 | 准确率 | AUROC | 选择方式 |")
    lines.append("|---|---:|---:|---:|---|")
    for row in contextual_baselines:
        lines.append(
            f"| {row['task']} | {_fmt(row['macro_f1'])} | {_fmt(row['accuracy'])} | "
            f"{_fmt(row['auroc'])} | {row['selection']} |"
        )
    lines.extend(
        [
            "",
            "Bootstrap 区间使用固定随机种子的测试集配对重采样。McNemar p 值采用精确双侧检验。",
            "MiniMol 行是已有的、根据验证集选择或采用固定配置的参考基线，不属于配对检索消融实验。",
            "论文主结果使用 deployment-visible agentic workflow，回答真实部署中的端到端性能、工具调用和 evidence 使用。",
            "identity-blind 与 deployment-visible-prefetched 使用完全相同的 harness-prefetched 工具证据，只作为 parity-controlled visibility 补充控制；它们不进入主结果表。",
            "当前结果使用 operational retrieval；parent-disjoint analog 消融完成前，这些数值仍属于 exploratory result。",
            "仅在 assistant response 中发现的 query-SMILES 匹配属于单独的重构诊断，需要人工复核。",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(PAPER_ROOT / "analysis"))
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
