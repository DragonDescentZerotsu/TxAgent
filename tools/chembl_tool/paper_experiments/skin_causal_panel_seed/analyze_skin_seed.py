"""Merge a gated Skin causal-panel seed with direct and report paired effects."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.chembl_tool.common.json_utils import read_jsonl, write_jsonl_atomic
from tools.chembl_tool.paper_experiments.skin_causal_panel_seed.build_skin_seed import (
    DEFAULT_OUTPUT_ROOT,
)


DEFAULT_DIRECT_BATCH = Path(
    "outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2_valid_"
    "deepseek_v4_pro_skin_canonical_v3_minimol_top3_audited/"
    "runs_identity_blind_parent_disjoint/skin_reaction/skin_reaction__starling_direct"
)
DEFAULT_SEED_BATCH = DEFAULT_OUTPUT_ROOT / (
    "runs_identity_blind_parent_disjoint/skin_reaction/"
    "skin_reaction__starling_causal_panel_seed_v1"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = analyze_skin_seed(
        direct_batch=args.direct_batch,
        seed_batch=args.seed_batch,
        output_root=args.output_root,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


def analyze_skin_seed(
    *,
    direct_batch: Path,
    seed_batch: Path,
    output_root: Path,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    direct_rows = read_jsonl(direct_batch / "predictions.jsonl")
    seed_rows = read_jsonl(seed_batch / "predictions.jsonl")
    selection = json.loads((output_root / "selected_indices.json").read_text(encoding="utf-8"))
    retrieval_audit = {
        int(row["query_index"]): row
        for row in read_jsonl(output_root / "retrieval_audit.jsonl")
    }
    selected_indices = [int(value) for value in selection["selected_indices"]]
    selected_set = set(selected_indices)
    direct_by_index = _unique_by_index(direct_rows, "direct")
    seed_by_index = _unique_by_index(seed_rows, "seed")
    if set(seed_by_index) != selected_set:
        missing = sorted(selected_set - set(seed_by_index))
        extra = sorted(set(seed_by_index) - selected_set)
        raise ValueError(f"Seed prediction coverage mismatch: missing={missing}, extra={extra}")
    if any(str(row.get("status")) != "ok" for row in seed_rows):
        raise ValueError("Seed batch contains failed predictions")

    merged_rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []
    for query_index in sorted(direct_by_index):
        direct = direct_by_index[query_index]
        candidate = seed_by_index.get(query_index, direct)
        for field in ("smiles", "label"):
            if candidate.get(field) != direct.get(field):
                raise ValueError(f"Seed/direct {field} mismatch at query {query_index}")
        merged = dict(candidate)
        merged["policy_source"] = "causal_panel_seed" if query_index in selected_set else "direct"
        merged_rows.append(merged)
        if int(candidate["pred_label"]) != int(direct["pred_label"]):
            direct_correct = int(direct["pred_label"]) == int(direct["label"])
            seed_correct = int(candidate["pred_label"]) == int(candidate["label"])
            audit = retrieval_audit[query_index]
            flip_rows.append(
                {
                    "query_index": query_index,
                    "label": int(direct["label"]),
                    "direct_pred": int(direct["pred_label"]),
                    "seed_pred": int(candidate["pred_label"]),
                    "direct_correct": direct_correct,
                    "seed_correct": seed_correct,
                    "flip_effect": (
                        "beneficial" if seed_correct and not direct_correct
                        else "harmful" if direct_correct and not seed_correct
                        else "neutral"
                    ),
                    "top_causal_direction": audit["top_causal_direction"],
                    "top_causal_similarity": audit["top_causal_similarity"],
                    "direct_summary": str(direct.get("final_summary") or ""),
                    "seed_summary": str(candidate.get("final_summary") or ""),
                }
            )

    direct_metrics = binary_metrics(direct_rows)
    merged_metrics = binary_metrics(merged_rows)
    selected_direct = [direct_by_index[index] for index in selected_indices]
    selected_seed = [seed_by_index[index] for index in selected_indices]
    selected_direct_metrics = binary_metrics(selected_direct)
    selected_seed_metrics = binary_metrics(selected_seed)
    bootstrap = paired_bootstrap(
        direct_rows,
        merged_rows,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    flip_counts = Counter(row["flip_effect"] for row in flip_rows)
    direction_strata = {}
    for direction in ("negative", "positive"):
        indices = [
            index
            for index in selected_indices
            if retrieval_audit[index]["top_causal_direction"] == direction
        ]
        direction_strata[direction] = {
            "n": len(indices),
            "direct": binary_metrics([direct_by_index[index] for index in indices]),
            "seed": binary_metrics([seed_by_index[index] for index in indices]),
        }

    reuse_audit = audit_branch_reuse(seed_batch, selected_indices)
    promotion_checks = {
        "complete_64_of_64": len(seed_rows) == len(selected_indices) == 64,
        "single_reuse_64_of_64": reuse_audit["n_single_reused"] == len(selected_indices),
        "tier1_group_reuse_64_of_64": reuse_audit["n_tier1_reused"] == len(selected_indices),
        "causal_group_fresh_64_of_64": reuse_audit["n_causal_fresh"] == len(selected_indices),
        "merged_macro_f1_above_direct": merged_metrics["macro_f1"] > direct_metrics["macro_f1"],
        "merged_accuracy_not_below_direct": merged_metrics["accuracy"] >= direct_metrics["accuracy"],
        "beneficial_flips_exceed_harmful": flip_counts["beneficial"] > flip_counts["harmful"],
    }
    result = {
        "type": "skin_causal_panel_seed_analysis.v1",
        "evaluation": {
            "task": "Skin_Reaction",
            "split": "scaffold",
            "subset": "valid",
            "model": "deepseek-v4-pro",
            "n_total": len(direct_rows),
            "n_seed_triggered": len(selected_indices),
            "policy": "direct except frozen label-blind causal-panel seed subset",
        },
        "direct": direct_metrics,
        "causal_panel_gated_policy": merged_metrics,
        "delta": {
            "macro_f1": merged_metrics["macro_f1"] - direct_metrics["macro_f1"],
            "accuracy": merged_metrics["accuracy"] - direct_metrics["accuracy"],
        },
        "selected_subset": {
            "direct": selected_direct_metrics,
            "causal_panel": selected_seed_metrics,
            "direction_strata": direction_strata,
        },
        "paired_bootstrap": bootstrap,
        "flips": {
            "n": len(flip_rows),
            "counts": dict(sorted(flip_counts.items())),
            "path": str(output_root / "analysis" / "paired_flips.jsonl"),
        },
        "branch_reuse": reuse_audit,
        "promotion_gate": {
            "passed": all(promotion_checks.values()),
            "checks": promotion_checks,
            "next_step": (
                "eligible_for_independent_replication_before_any_BBB_work"
                if all(promotion_checks.values())
                else "stop_and_diagnose_skin_source_or_transfer_failure"
            ),
        },
    }
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(analysis_dir / "merged_predictions.jsonl", merged_rows)
    write_jsonl_atomic(analysis_dir / "paired_flips.jsonl", flip_rows)
    (analysis_dir / "analysis.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (analysis_dir / "report.md").write_text(_render_report(result), encoding="utf-8")
    return result


def binary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    preds = np.asarray([int(row["pred_label"]) for row in rows], dtype=int)
    per_class = {}
    f1s = []
    for label in (0, 1):
        tp = int(((labels == label) & (preds == label)).sum())
        fp = int(((labels != label) & (preds == label)).sum())
        fn = int(((labels == label) & (preds != label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class[str(label)] = {
            "support": int((labels == label).sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "n": len(rows),
        "accuracy": float((labels == preds).mean()) if len(rows) else 0.0,
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_matrix_y0_y1": [
            [int(((labels == 0) & (preds == 0)).sum()), int(((labels == 0) & (preds == 1)).sum())],
            [int(((labels == 1) & (preds == 0)).sum()), int(((labels == 1) & (preds == 1)).sum())],
        ],
    }


def paired_bootstrap(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if len(baseline) != len(candidate):
        raise ValueError("Paired bootstrap requires aligned row counts")
    rng = np.random.default_rng(seed)
    macro_deltas = np.empty(n_bootstrap, dtype=float)
    accuracy_deltas = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        indices = rng.integers(0, len(baseline), size=len(baseline))
        base_sample = [baseline[index] for index in indices]
        candidate_sample = [candidate[index] for index in indices]
        base_metrics = binary_metrics(base_sample)
        candidate_metrics = binary_metrics(candidate_sample)
        macro_deltas[iteration] = candidate_metrics["macro_f1"] - base_metrics["macro_f1"]
        accuracy_deltas[iteration] = candidate_metrics["accuracy"] - base_metrics["accuracy"]
    return {
        "n_resamples": n_bootstrap,
        "seed": seed,
        "macro_f1_delta_ci95": [
            float(np.quantile(macro_deltas, 0.025)),
            float(np.quantile(macro_deltas, 0.975)),
        ],
        "accuracy_delta_ci95": [
            float(np.quantile(accuracy_deltas, 0.025)),
            float(np.quantile(accuracy_deltas, 0.975)),
        ],
        "p_delta_macro_f1_gt_0": float((macro_deltas > 0).mean()),
        "p_delta_accuracy_gt_0": float((accuracy_deltas > 0).mean()),
    }


def audit_branch_reuse(seed_batch: Path, indices: list[int]) -> dict[str, Any]:
    n_single_reused = 0
    n_tier1_reused = 0
    n_causal_fresh = 0
    failures = []
    for query_index in indices:
        run_dir = seed_batch / "runs" / f"{seed_batch.name}_idx{query_index:05d}"
        single = json.loads((run_dir / "single_molecule_reasoning_output.json").read_text(encoding="utf-8"))
        groups = read_jsonl(run_dir / "group_reasoning_outputs.jsonl")
        if str(single.get("reused_from") or ""):
            n_single_reused += 1
        tier1 = [row for row in groups if row.get("group_id") == "Mechanism.tier_1"]
        causal = [row for row in groups if row.get("group_id") == "Mechanism.causal_panel"]
        if len(tier1) == 1 and str(tier1[0].get("reused_from") or ""):
            n_tier1_reused += 1
        if len(causal) == 1 and not str(causal[0].get("reused_from") or ""):
            n_causal_fresh += 1
        if len(tier1) != 1 or len(causal) != 1:
            failures.append(query_index)
    return {
        "n_single_reused": n_single_reused,
        "n_tier1_reused": n_tier1_reused,
        "n_causal_fresh": n_causal_fresh,
        "group_shape_failure_indices": failures,
    }


def _unique_by_index(rows: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row["query_index"])
        if index in output:
            raise ValueError(f"Duplicate {label} query index {index}")
        output[index] = row
    return output


def _render_report(result: dict[str, Any]) -> str:
    direct = result["direct"]
    candidate = result["causal_panel_gated_policy"]
    delta = result["delta"]
    flips = result["flips"]
    gate = result["promotion_gate"]
    return (
        "# Skin causal-panel seed experiment\n\n"
        f"- Direct: macro-F1 {direct['macro_f1']:.6f}, accuracy {direct['accuracy']:.6f}.\n"
        f"- Gated causal panel: macro-F1 {candidate['macro_f1']:.6f}, "
        f"accuracy {candidate['accuracy']:.6f}.\n"
        f"- Delta: macro-F1 {delta['macro_f1']:+.6f}, accuracy {delta['accuracy']:+.6f}.\n"
        f"- Paired flips: {flips['counts']}.\n"
        f"- Promotion gate: {'PASS' if gate['passed'] else 'FAIL'}; {gate['next_step']}.\n"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-batch", type=Path, default=DEFAULT_DIRECT_BATCH)
    parser.add_argument("--seed-batch", type=Path, default=DEFAULT_SEED_BATCH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
