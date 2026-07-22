"""Audit scored assay-transfer validation batches and write comparison tables."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
from typing import Any

from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.retrieval_policy import classify_molecule_relation


FORBIDDEN_RELATIONS = {"exact_record", "same_connectivity_variant", "same_parent"}
HIDDEN_PROMPT_FIELDS = {
    "transfer_selection_score",
    "transfer_selection_rank",
    "transfer_winning_record_id",
    "transfer_scored_record_count",
    "structural_rank",
}
EXPECTED_MODEL = "jiosephlee/assay-transfer-tool"
EXPECTED_REVISION = "9515603b1a5c4586e41c221dcdbc5e7487c0c3f5"
EXPECTED_SCORING_CONTRACT = "assay_transfer_chat_first_divergent_token_logits.v1"
EXPECTED_SCORE_POLICY = "assay_transfer_scored_neighbors.v1"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cache_audit = _audit_cache(Path(args.cache))
    baselines = dict(_label_path(value) for value in args.baseline)
    pinned_dir = Path(args.pinned_batch)
    pinned_metrics = _read_json(pinned_dir / "metrics.json")
    pinned_predictions = _predictions(pinned_dir)
    audits = []
    for text in args.batch:
        label, path = _label_path(text)
        baseline_dir = Path(baselines[label])
        audit = _audit_batch(label, Path(path))
        audit["comparison_to_morgan"] = _compare(Path(path), baseline_dir)
        audit["comparison_to_superseded_pinned_hf_k5"] = _compare_predictions(
            Path(path), pinned_dir, pinned_metrics, pinned_predictions
        )
        audits.append(audit)
        (Path(path) / "reranked_retrieval_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": (
            "pass"
            if cache_audit["audit_status"] == "pass" and all(item["audit_status"] == "pass" for item in audits)
            else "fail"
        ),
        "model": EXPECTED_MODEL,
        "model_revision": EXPECTED_REVISION,
        "scoring_contract": EXPECTED_SCORING_CONTRACT,
        "cache": str(Path(args.cache)),
        "cache_version": "v2",
        "cache_audit": cache_audit,
        "batches": audits,
        "superseded_pinned_hf_batch": str(pinned_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_dir": str(out_dir)}, indent=2))
    return 0 if summary["status"] == "pass" else 1


def _audit_cache(cache_path: Path) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    version_path = cache_path.parent / "VERSION.json"
    if not cache_path.is_file():
        return {
            "audit_status": "fail",
            "n_prompt_scores": 0,
            "violations": [{"reason": "missing_cache", "path": str(cache_path)}],
        }
    if not version_path.is_file():
        violations.append({"reason": "missing_cache_version", "path": str(version_path)})
        version: dict[str, Any] = {}
    else:
        version = _read_json(version_path)
    for key, expected in {
        "cache_version": "v2",
        "model": EXPECTED_MODEL,
        "model_revision": EXPECTED_REVISION,
    }.items():
        if version.get(key) != expected:
            violations.append({"reason": "cache_version_mismatch", "field": key, "observed": version.get(key)})
    try:
        uri = f"file:{cache_path.resolve()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            n_prompt_scores = int(connection.execute("SELECT COUNT(*) FROM prompt_scores").fetchone()[0])
    except (sqlite3.Error, TypeError, IndexError) as exc:
        n_prompt_scores = 0
        violations.append({"reason": "unreadable_cache", "error": str(exc)})
    if n_prompt_scores <= 0:
        violations.append({"reason": "empty_cache"})
    return {
        "audit_status": "pass" if not violations else "fail",
        "version_path": str(version_path),
        "n_prompt_scores": n_prompt_scores,
        "violations": violations,
    }


def _audit_batch(label: str, batch_dir: Path) -> dict[str, Any]:
    manifest = _read_json(batch_dir / "manifest.json")
    metrics = _read_json(batch_dir / "metrics.json")
    top_k = int(manifest["top_k_per_group"])
    floor = float(manifest["min_similarity"])
    violations: list[dict[str, Any]] = []
    relation_counts: Counter[str] = Counter()
    group_sizes: Counter[int] = Counter()
    similarities: list[float] = []
    scores: list[float] = []
    n_group_prompts = 0
    n_expected_group_prompts = 0
    n_groups_with_unscoreable_drop = 0

    expected_manifest = {
        "experiment_mode": "full_mechanism",
        "retrieval_source": "starling",
        "retrieval_reranker": "assay_transfer",
        "neighbor_identity_policy": "parent_disjoint",
        "enable_assay_transfer_scores": True,
        "llm_neighbor_score_policy": EXPECTED_SCORE_POLICY,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            violations.append({"reason": "manifest_mismatch", "field": key, "observed": manifest.get(key)})
    provenance = (manifest.get("rerank_cache_preflight") or {}).get("provenance") or {}
    for key, expected in {
        "model": EXPECTED_MODEL,
        "model_revision": EXPECTED_REVISION,
        "scoring_contract_version": EXPECTED_SCORING_CONTRACT,
    }.items():
        if provenance.get(key) != expected:
            violations.append({"reason": "provenance_mismatch", "field": key, "observed": provenance.get(key)})
    if metrics.get("n_total") != 64 or metrics.get("n_successful") != 64 or metrics.get("n_failed_runs") != 0:
        violations.append({"reason": "incomplete_batch", "metrics": metrics})

    retrieval_paths = sorted((batch_dir / "runs").glob("*/retrieval.json"))
    if len(retrieval_paths) != 64:
        violations.append({"reason": "retrieval_file_count", "observed": len(retrieval_paths)})
    for path in retrieval_paths:
        query_index = int(path.parent.name.rsplit("idx", 1)[1])
        retrieval = _read_json(path)
        query_identity = normalize_molecule_identity(retrieval["query"]["canonical_smiles"])
        score_policy = (retrieval.get("experiment") or {}).get("llm_neighbor_score_policy") or {}
        if score_policy.get("name") != EXPECTED_SCORE_POLICY or score_policy.get("round_decimals") != 2:
            violations.append({"reason": "retrieval_score_policy", "query_index": query_index})
        for group in retrieval.get("groups") or []:
            neighbors = group.get("neighbors") or []
            n_expected_group_prompts += bool(neighbors)
            group_sizes[len(neighbors)] += 1
            selection = group.get("transfer_neighbor_selection") or {}
            if len(neighbors) > top_k:
                violations.append({"reason": "over_k", "query_index": query_index, "group": group["group_id"]})
            if len(neighbors) < top_k:
                n_groups_with_unscoreable_drop += 1
            previous = float("inf")
            for neighbor in neighbors:
                relation = classify_molecule_relation(
                    query_identity, normalize_molecule_identity(neighbor["canonical_smiles"])
                ).value
                relation_counts[relation] += 1
                similarity = float(neighbor["similarity"])
                score = float(neighbor["transfer_selection_score"])
                similarities.append(similarity)
                scores.append(score)
                if relation in FORBIDDEN_RELATIONS:
                    violations.append({"reason": relation, "query_index": query_index, "group": group["group_id"]})
                if similarity < floor:
                    violations.append({"reason": "below_similarity_floor", "query_index": query_index, "similarity": similarity})
                if not 0.0 <= score <= 1.0:
                    violations.append({"reason": "invalid_transfer_score", "query_index": query_index, "score": score})
                if score > previous:
                    violations.append({"reason": "rerank_order", "query_index": query_index, "group": group["group_id"]})
                previous = score
                if int(neighbor.get("transfer_selection_rank") or 0) > top_k:
                    violations.append({"reason": "selection_rank_over_k", "query_index": query_index})
            if int(selection.get("n_selected") or 0) != len(neighbors):
                violations.append({"reason": "selection_count_mismatch", "query_index": query_index, "group": group["group_id"]})
        prompt_result = _audit_trace_prompts(path.parent, retrieval)
        n_group_prompts += prompt_result["n_group_prompts"]
        violations.extend({"query_index": query_index, **item} for item in prompt_result["violations"])

    expected_dropped = int((manifest.get("rerank_cache_preflight") or {}).get("n_unscoreable_selected_dropped") or 0)
    expected_scoreable = int((manifest.get("rerank_cache_preflight") or {}).get("n_scoreable_selected") or 0)
    if len(scores) != expected_scoreable:
        violations.append({
            "reason": "scoreable_selected_count_mismatch",
            "expected": expected_scoreable,
            "observed": len(scores),
        })
    if n_group_prompts != n_expected_group_prompts:
        violations.append({
            "reason": "group_prompt_count",
            "expected": n_expected_group_prompts,
            "observed": n_group_prompts,
        })

    return {
        "label": label,
        "batch_dir": str(batch_dir),
        "audit_status": "pass" if not violations else "fail",
        "metrics": metrics,
        "top_k_per_group": top_k,
        "min_similarity": floor,
        "n_retrieval_files": len(retrieval_paths),
        "n_groups": sum(group_sizes.values()),
        "n_neighbors": len(similarities),
        "group_neighbor_count_distribution": {str(k): v for k, v in sorted(group_sizes.items())},
        "n_groups_below_k": n_groups_with_unscoreable_drop,
        "n_unscoreable_selected_dropped": expected_dropped,
        "n_candidates_absent_due_to_insufficient_eligible_neighbors": 320 * top_k - expected_scoreable - expected_dropped,
        "neighbor_relation_counts": dict(sorted(relation_counts.items())),
        "similarity_min": min(similarities) if similarities else None,
        "similarity_max": max(similarities) if similarities else None,
        "transfer_score_min": min(scores) if scores else None,
        "transfer_score_max": max(scores) if scores else None,
        "n_group_prompts_with_visible_rounded_scores": n_group_prompts,
        "violations": violations,
    }


def _audit_trace_prompts(run_dir: Path, retrieval: dict[str, Any]) -> dict[str, Any]:
    traces = _read_jsonl(run_dir / "trace_messages.jsonl")
    expected = {
        group["group_id"]: {
            row["molecule_chembl_id"]: round(float(row["transfer_selection_score"]), 2)
            for row in group.get("neighbors") or []
        }
        for group in retrieval.get("groups") or []
    }
    violations: list[dict[str, Any]] = []
    n_group_prompts = 0
    for trace in traces:
        user_messages = [m.get("content", "") for m in trace.get("messages") or [] if m.get("role") == "user"]
        if not user_messages:
            continue
        initial = user_messages[0]
        if trace.get("task") in expected:
            n_group_prompts += 1
            try:
                payload = json.loads(initial.split("Input JSON:\n", 1)[-1])
            except json.JSONDecodeError:
                violations.append({"reason": "invalid_group_prompt_json", "task": trace.get("task")})
                continue
            observed = {row["molecule_chembl_id"]: row.get("assay_transfer_score") for row in payload.get("neighbors") or []}
            if observed != expected[trace["task"]]:
                violations.append({"reason": "visible_score_mismatch", "task": trace.get("task")})
            leaked = sorted(field for field in HIDDEN_PROMPT_FIELDS if field in initial)
            if leaked:
                violations.append({"reason": "audit_metadata_leak", "task": trace.get("task"), "fields": leaked})
        elif trace.get("task") == "final" and "assay_transfer_score" in initial:
            violations.append({"reason": "score_leaked_to_final_prompt"})
    return {"n_group_prompts": n_group_prompts, "violations": violations}


def _compare(current: Path, baseline: Path) -> dict[str, Any]:
    metrics = _read_json(current / "metrics.json")
    baseline_metrics = _read_json(baseline / "metrics.json")
    current_predictions = _predictions(current)
    baseline_predictions = _predictions(baseline)
    overlap = sorted(set(current_predictions) & set(baseline_predictions))
    return {
        "baseline_batch": str(baseline),
        "n_paired": len(overlap),
        "n_prediction_flips": sum(current_predictions[i] != baseline_predictions[i] for i in overlap),
        "accuracy_delta": round(float(metrics["accuracy"]) - float(baseline_metrics["accuracy"]), 6),
        "macro_f1_delta": round(float(metrics["macro_f1"]) - float(baseline_metrics["macro_f1"]), 6),
        "baseline_metrics": baseline_metrics,
    }


def _compare_predictions(
    current: Path,
    pinned: Path,
    pinned_metrics: dict[str, Any],
    pinned_predictions: dict[int, Any],
) -> dict[str, Any]:
    metrics = _read_json(current / "metrics.json")
    current_predictions = _predictions(current)
    overlap = sorted(set(current_predictions) & set(pinned_predictions))
    return {
        "pinned_batch": str(pinned),
        "pinned_status": "superseded_audit_artifact",
        "n_paired": len(overlap),
        "n_prediction_flips": sum(current_predictions[i] != pinned_predictions[i] for i in overlap),
        "accuracy_delta": round(float(metrics["accuracy"]) - float(pinned_metrics["accuracy"]), 6),
        "macro_f1_delta": round(float(metrics["macro_f1"]) - float(pinned_metrics["macro_f1"]), 6),
        "pinned_metrics": pinned_metrics,
    }


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Assay-transfer-reranked prepared-HF validation",
        "",
        f"Audit status: **{summary['status']}**. Model: `{summary['model']}@{summary['model_revision']}`; cache: v2.",
        "The pinned-HF k=5 output remains a **superseded audit artifact**.",
        "",
    ]
    for floor, title in [(0.3, "With similarity floor 0.30"), (0.0, "Without a similarity floor")]:
        rows = [item for item in summary["batches"] if float(item["min_similarity"]) == floor]
        lines += [
            f"## {title}",
            "",
            "| k | Accuracy | Macro-F1 | Δ accuracy vs Morgan | Δ macro-F1 vs Morgan | Failures | Audit |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
        for item in sorted(rows, key=lambda row: row["top_k_per_group"]):
            metrics = item["metrics"]
            comparison = item["comparison_to_morgan"]
            lines.append(
                f"| {item['top_k_per_group']} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} "
                f"| {comparison['accuracy_delta']:+.4f} | {comparison['macro_f1_delta']:+.4f} "
                f"| {metrics['n_failed_runs']} | {item['audit_status']} |"
            )
        lines.append("")
    return "\n".join(lines)


def _predictions(batch_dir: Path) -> dict[int, Any]:
    return {int(row["query_index"]): row.get("pred_label") for row in _read_jsonl(batch_dir / "predictions.jsonl")}


def _label_path(text: str) -> tuple[str, str]:
    label, path = text.split("=", 1)
    return label, path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", action="append", required=True, help="LABEL=BATCH_DIR")
    parser.add_argument("--baseline", action="append", required=True, help="Matching LABEL=BATCH_DIR")
    parser.add_argument("--pinned-batch", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
