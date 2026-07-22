"""Audit prepared-HF validation retrieval and compare it with superseded pinned-HF results."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import pickle
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.experiment_retrieval import retrieve_experiment_view
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.retrieval_policy import classify_molecule_relation
from tools.chembl_tool.tasks.bioavailability_ma.experiment_config import STARLING


FORBIDDEN_RELATIONS = {"exact_record", "same_connectivity_variant", "same_parent"}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    with Path(args.index).open("rb") as handle:
        index = pickle.load(handle)
    inputs = _read_jsonl(Path(args.input_jsonl))
    audits = []
    for batch_text in args.batch:
        label, path_text = batch_text.split("=", 1)
        audits.append(_audit_batch(label, Path(path_text), index, inputs))

    pinned_dir = Path(args.pinned_batch)
    pinned_metrics = _read_json(pinned_dir / "metrics.json")
    pinned_predictions = {row["query_index"]: row for row in _read_jsonl(pinned_dir / "predictions.jsonl")}
    for audit in audits:
        batch_dir = Path(audit["batch_dir"])
        current = {row["query_index"]: row for row in _read_jsonl(batch_dir / "predictions.jsonl")}
        overlap = sorted(set(current) & set(pinned_predictions))
        audit["comparison_to_superseded_pinned_hf_k5"] = {
            "pinned_batch": str(pinned_dir),
            "pinned_status": "superseded_audit_artifact",
            "n_paired": len(overlap),
            "n_prediction_flips": sum(
                current[i].get("pred_label") != pinned_predictions[i].get("pred_label") for i in overlap
            ),
            "accuracy_delta": round(
                float(audit["metrics"]["accuracy"]) - float(pinned_metrics["accuracy"]), 6
            ),
            "macro_f1_delta": round(
                float(audit["metrics"]["macro_f1"]) - float(pinned_metrics["macro_f1"]), 6
            ),
            "pinned_metrics": pinned_metrics,
        }
        _write_batch_audit(batch_dir, audit)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "pass" if all(item["audit_status"] == "pass" for item in audits) else "fail",
        "prepared_hf_batches": audits,
        "superseded_pinned_hf_batch": str(pinned_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(_comparison_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "pass" else 1


def _audit_batch(label: str, batch_dir: Path, index: dict[str, Any], inputs: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = _read_json(batch_dir / "manifest.json")
    metrics = _read_json(batch_dir / "metrics.json")
    top_k = int(manifest["top_k_per_group"])
    floor = float(manifest["min_similarity"])
    retrieval_paths = sorted((batch_dir / "runs").glob("*/retrieval.json"))
    relation_counts: Counter[str] = Counter()
    group_size_counts: Counter[int] = Counter()
    violations: list[dict[str, Any]] = []
    below_k: list[tuple[int, str, int]] = []
    similarities: list[float] = []

    for path in retrieval_paths:
        query_index = int(path.parent.name.rsplit("idx", 1)[1])
        payload = _read_json(path)
        query_identity = normalize_molecule_identity(payload["query"]["canonical_smiles"])
        for group in payload["groups"]:
            neighbors = group.get("neighbors") or []
            group_size_counts[len(neighbors)] += 1
            if len(neighbors) > top_k:
                violations.append({"query_index": query_index, "group": group["group_id"], "reason": "over_k"})
            if len(neighbors) < top_k:
                below_k.append((query_index, group["group_id"], len(neighbors)))
            for neighbor in neighbors:
                relation = classify_molecule_relation(
                    query_identity, normalize_molecule_identity(neighbor["canonical_smiles"])
                ).value
                relation_counts[relation] += 1
                similarity = float(neighbor["similarity"])
                similarities.append(similarity)
                if relation in FORBIDDEN_RELATIONS or similarity < floor:
                    violations.append({
                        "query_index": query_index,
                        "group": group["group_id"],
                        "reason": relation if relation in FORBIDDEN_RELATIONS else "below_similarity_floor",
                        "similarity": similarity,
                    })

    expanded_counts: dict[tuple[int, str], int] = {}
    for query_index in sorted({item[0] for item in below_k}):
        expanded = retrieve_experiment_view(
            inputs[query_index]["drug"], index, mode="full_mechanism", config=STARLING,
            top_k_per_group=top_k + 1, min_similarity=floor, neighbor_identity_policy="parent_disjoint",
        )
        expanded_counts.update({
            (query_index, group["group_id"]): len(group.get("neighbors") or [])
            for group in expanded["groups"]
        })
    unexplained = [
        {"query_index": i, "group": group, "retained": count, "expanded_retained": expanded_counts[(i, group)]}
        for i, group, count in below_k if expanded_counts[(i, group)] != count
    ]
    violations.extend({**item, "reason": "below_k_despite_additional_eligible_neighbor"} for item in unexplained)

    return {
        "label": label,
        "batch_dir": str(batch_dir),
        "audit_status": "pass" if not violations else "fail",
        "metrics": metrics,
        "n_retrieval_files": len(retrieval_paths),
        "n_groups": sum(group_size_counts.values()),
        "n_neighbors": len(similarities),
        "neighbor_relation_counts": dict(sorted(relation_counts.items())),
        "group_neighbor_count_distribution": {str(k): v for k, v in sorted(group_size_counts.items())},
        "n_groups_below_k": len(below_k),
        "n_below_k_explained_by_insufficient_eligible_neighbors": len(below_k) - len(unexplained),
        "similarity_min": min(similarities) if similarities else None,
        "similarity_max": max(similarities) if similarities else None,
        "top_k_per_group": top_k,
        "min_similarity": floor,
        "violations": violations,
    }


def _write_batch_audit(batch_dir: Path, audit: dict[str, Any]) -> None:
    (batch_dir / "retrieval_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    comparison = audit["comparison_to_superseded_pinned_hf_k5"]
    text = f"""# Prepared-HF retrieval audit: {audit['label']}

- Status: `{audit['audit_status']}`
- Final outputs: {audit['metrics']['n_successful']}/{audit['metrics']['n_total']}; failures: {audit['metrics']['n_failed_runs']}
- Accuracy: {audit['metrics']['accuracy']:.4f}; macro-F1: {audit['metrics']['macro_f1']:.4f}
- Retained neighbors: {audit['n_neighbors']}; minimum similarity: {audit['similarity_min']}
- Identity relations: `{json.dumps(audit['neighbor_relation_counts'], sort_keys=True)}`
- Groups below k: {audit['n_groups_below_k']}; explained by insufficient eligible neighbors above the floor: {audit['n_below_k_explained_by_insufficient_eligible_neighbors']}
- Retrieval violations: {len(audit['violations'])}

The prior pinned-HF k=5 batch is a **superseded audit artifact**. Relative to it, this run changes accuracy by {comparison['accuracy_delta']:+.4f}, macro-F1 by {comparison['macro_f1_delta']:+.4f}, and flips {comparison['n_prediction_flips']}/{comparison['n_paired']} paired predictions.
"""
    (batch_dir / "retrieval_report.md").write_text(text, encoding="utf-8")


def _comparison_report(summary: dict[str, Any]) -> str:
    lines = ["# Prepared-HF validation comparison", "", "The pinned-HF k=5 result is retained only as a **superseded audit artifact**.", ""]
    lines += ["| Condition | Accuracy | Macro-F1 | Failures | Retrieval audit |", "|---|---:|---:|---:|---|"]
    for item in summary["prepared_hf_batches"]:
        metrics = item["metrics"]
        lines.append(f"| {item['label']} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | {metrics['n_failed_runs']} | {item['audit_status']} |")
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", action="append", required=True, help="LABEL=BATCH_DIR")
    parser.add_argument("--index", required=True)
    parser.add_argument("--input-jsonl", default="data/processed/Bioavailability_Ma/valid.jsonl")
    parser.add_argument("--pinned-batch", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
