"""Run a simple Starling v1 numeric KNN-3 majority-vote baseline."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.task_workflows.retrieve_neighbors import retrieve_neighbors


DEFAULT_INPUT = "data/processed/Bioavailability_Ma/test.jsonl"
DEFAULT_INDEX = (
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling/"
    "starling_oral_bioavailability_neighbor_index.pkl"
)
DEFAULT_OUT_DIR = (
    "outputs/chembl_tool/tasks/bioavailability_ma/baselines/"
    "starling_v1_knn3_majority"
)
GROUP_ID = "Starling.direct_oral_bioavailability"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    records = _read_jsonl(Path(args.input_jsonl))
    with Path(args.index).open("rb") as handle:
        full_index = pickle.load(handle)
    numeric_index = numeric_only_index(full_index, group_id=args.group)

    predictions = [
        predict_record(
            query_index=index,
            record=record,
            index=numeric_index,
            smiles_field=args.smiles_field,
            label_field=args.label_field,
            threshold_percent=args.threshold_percent,
            group_id=args.group,
        )
        for index, record in enumerate(records)
    ]
    metrics = compute_metrics(predictions)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "predictions.jsonl", predictions)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "baseline": "Starling v1 numeric KNN-3 majority vote",
        "input_jsonl": args.input_jsonl,
        "source_index": args.index,
        "group_id": args.group,
        "n_input_records": len(records),
        "n_numeric_candidate_molecules": len(numeric_index["group_to_molecule_indices"][args.group]),
        "k": 3,
        "min_similarity": None,
        "vote_weighting": "unweighted majority",
        "neighbor_label_rule": f"median oral F% >= {args.threshold_percent:g}% => high; otherwise low",
        "query_exact_exclusion": (
            "full InChIKey, InChIKey connectivity layer, or canonical SMILES"
        ),
        "fingerprint": numeric_index.get("fingerprint", {}),
        "outputs": {
            "predictions": str(out_dir / "predictions.jsonl"),
            "metrics": str(out_dir / "metrics.json"),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(
        _render_report(manifest, metrics),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def numeric_only_index(index: dict[str, Any], *, group_id: str = GROUP_ID) -> dict[str, Any]:
    """Return the clean numeric Starling v1 slice of the current Starling index."""
    numeric_indices: list[int] = []
    for molecule_index in index["group_to_molecule_indices"].get(group_id, []):
        molecule = index["molecules"][molecule_index]
        rows = index["evidence_by_molecule_group"][molecule["molecule_chembl_id"]][group_id]
        if any(_numeric_median(row) is not None for row in rows):
            numeric_indices.append(molecule_index)
    return {
        **index,
        "group_to_molecule_indices": {group_id: numeric_indices},
        "source": {
            **index.get("source", {}),
            "type": "starling_oral_bioavailability_numeric_v1_slice",
            "qualitative_only_molecules_included": False,
        },
    }


def predict_record(
    *,
    query_index: int,
    record: dict[str, Any],
    index: dict[str, Any],
    smiles_field: str = "drug",
    label_field: str = "Y",
    threshold_percent: float = 20.0,
    group_id: str = GROUP_ID,
) -> dict[str, Any]:
    query_smiles = str(record.get(smiles_field) or "")
    retrieval = retrieve_neighbors(
        query_smiles,
        index,
        top_k_per_group=3,
        min_similarity=0.0,
        groups=[group_id],
    )
    neighbors = retrieval["groups"][0]["neighbors"] if retrieval.get("groups") else []
    if len(neighbors) != 3:
        raise RuntimeError(
            f"Expected exactly 3 Starling v1 numeric neighbors for query {query_index}, "
            f"found {len(neighbors)}."
        )

    neighbor_outputs = []
    high_votes = 0
    for neighbor in neighbors:
        evidence_rows = neighbor["evidence_rows"]
        medians = [value for row in evidence_rows if (value := _numeric_median(row)) is not None]
        if not medians:
            raise RuntimeError(
                f"Numeric Starling neighbor {neighbor['molecule_chembl_id']} has no median F%."
            )
        median_percent = medians[0]
        prediction = "high" if median_percent >= threshold_percent else "low"
        high_votes += int(prediction == "high")
        neighbor_outputs.append(
            {
                "rank": neighbor["rank"],
                "molecule_id": neighbor["molecule_chembl_id"],
                "canonical_smiles": neighbor["canonical_smiles"],
                "standard_inchi_key": neighbor.get("standard_inchi_key", ""),
                "similarity": neighbor["similarity"],
                "median_bioavailability_percent": median_percent,
                "neighbor_label": prediction,
            }
        )

    predicted_label = 1 if high_votes >= 2 else 0
    true_label = _parse_binary_label(record.get(label_field))
    return {
        "query_index": query_index,
        "smiles": query_smiles,
        "label": true_label,
        "prediction": "high" if predicted_label == 1 else "low",
        "pred_label": predicted_label,
        "correct": predicted_label == true_label,
        "high_votes": high_votes,
        "low_votes": 3 - high_votes,
        "neighbors": neighbor_outputs,
        "query": retrieval.get("query", {}),
        "status": "ok",
    }


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {
        "tn": sum(row["label"] == 0 and row["pred_label"] == 0 for row in rows),
        "fp": sum(row["label"] == 0 and row["pred_label"] == 1 for row in rows),
        "fn": sum(row["label"] == 1 and row["pred_label"] == 0 for row in rows),
        "tp": sum(row["label"] == 1 and row["pred_label"] == 1 for row in rows),
    }
    per_class = {
        str(label): _class_metrics(rows, label)
        for label in (0, 1)
    }
    return {
        "n_total": len(rows),
        "n_successful": len(rows),
        "accuracy": _safe_div(sum(row["correct"] for row in rows), len(rows)),
        "macro_f1": round((per_class["0"]["f1"] + per_class["1"]["f1"]) / 2, 6),
        "per_class": per_class,
        "positive_class_precision": per_class["1"]["precision"],
        "positive_class_recall": per_class["1"]["recall"],
        "positive_class_f1": per_class["1"]["f1"],
        "confusion_matrix": confusion,
        "prediction_distribution": {
            "high": sum(row["pred_label"] == 1 for row in rows),
            "low": sum(row["pred_label"] == 0 for row in rows),
        },
        "retrieval": {
            "k": 3,
            "min_similarity": None,
            "queries_with_exactly_3_neighbors": sum(len(row["neighbors"]) == 3 for row in rows),
            "returned_neighbors": sum(len(row["neighbors"]) for row in rows),
            "minimum_returned_similarity": min(
                neighbor["similarity"]
                for row in rows
                for neighbor in row["neighbors"]
            ),
        },
    }


def _class_metrics(rows: list[dict[str, Any]], label: int) -> dict[str, Any]:
    tp = sum(row["label"] == label and row["pred_label"] == label for row in rows)
    fp = sum(row["label"] != label and row["pred_label"] == label for row in rows)
    fn = sum(row["label"] == label and row["pred_label"] != label for row in rows)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _numeric_median(row: dict[str, Any]) -> float | None:
    value = row.get("source_value_median_percent", row.get("standard_value"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_binary_label(value: Any) -> int:
    if value in (0, 1):
        return int(value)
    text = str(value).strip().lower()
    if text in {"0", "false", "low", "negative"}:
        return 0
    if text in {"1", "true", "high", "positive"}:
        return 1
    raise ValueError(f"Unsupported binary label: {value!r}")


def _safe_div(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render_report(manifest: dict[str, Any], metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Starling v1 KNN-3 Majority Baseline",
            "",
            "## Definition",
            "",
            "- Source: clean numeric Starling v1 molecule-level evidence.",
            "- Retrieval: Morgan Tanimoto top-3 after exact-query connectivity exclusion.",
            "- Minimum similarity: none.",
            "- Neighbor label: molecule median F% >= 20% is high; otherwise low.",
            "- Prediction: unweighted majority vote over exactly three neighbors.",
            "",
            "## Results",
            "",
            f"- n: {metrics['n_total']}",
            f"- accuracy: {metrics['accuracy']:.6f}",
            f"- macro-F1: {metrics['macro_f1']:.6f}",
            f"- positive precision: {metrics['positive_class_precision']:.6f}",
            f"- positive recall: {metrics['positive_class_recall']:.6f}",
            f"- confusion matrix: `{json.dumps(metrics['confusion_matrix'])}`",
            f"- prediction distribution: `{json.dumps(metrics['prediction_distribution'])}`",
            f"- minimum returned similarity: {metrics['retrieval']['minimum_returned_similarity']:.6f}",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python -m tools.chembl_tool.tasks.bioavailability_ma.run_starling_v1_knn_baseline",
            "```",
            "",
            f"Source index: `{manifest['source_index']}`",
            "",
        ]
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--smiles-field", default="drug")
    parser.add_argument("--label-field", default="Y")
    parser.add_argument("--group", default=GROUP_ID)
    parser.add_argument("--threshold-percent", type=float, default=20.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
