"""Apply calibrated Bioavailability_Ma-specific postprocess policies.

This stage is intentionally downstream of a completed reasoning batch. It does
not call the final LLM again; it applies a selected deterministic fallback rule
to already-compiled feature rows and writes a normal predictions/metrics bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.task_workflows.reasoning_batch import compute_metrics, prediction_to_label
from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch_specific import CONFIG
from tools.chembl_tool.tasks.bioavailability_ma.specific_fallback_policy import candidate_rule_prediction


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rule_name = _resolve_rule_name(args)
    prediction_rows = _read_jsonl(Path(args.predictions_jsonl))
    feature_rows = _read_jsonl(Path(args.features_jsonl))
    post_rows = apply_postprocess_policy(prediction_rows, feature_rows, rule_name)
    metrics = compute_metrics(CONFIG, post_rows)
    manifest = {
        "postprocess_policy": "bioavailability_specific_valid_selected_fallback_postprocess",
        "rule_name": rule_name,
        "predictions_jsonl": args.predictions_jsonl,
        "features_jsonl": args.features_jsonl,
        "selected_rule_json": args.selected_rule_json,
        "n_input_predictions": len(prediction_rows),
        "n_feature_rows": len(feature_rows),
        "paths": {
            "out_dir": str(out_dir),
            "predictions": str(out_dir / "predictions.jsonl"),
            "metrics": str(out_dir / "metrics.json"),
            "manifest": str(out_dir / "manifest.json"),
            "report": str(out_dir / "report.md"),
        },
    }
    _write_jsonl(out_dir / "predictions.jsonl", post_rows)
    _write_json(out_dir / "metrics.json", metrics)
    _write_json(out_dir / "manifest.json", manifest)
    _write_report(out_dir / "report.md", manifest, metrics, post_rows)
    print(
        json.dumps(
            {
                "rule_name": rule_name,
                "metrics": {
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "confusion_matrix": metrics.get("confusion_matrix"),
                    "prediction_distribution": metrics.get("prediction_distribution"),
                },
                "n_changed": sum(1 for row in post_rows if row.get("postprocess_policy", {}).get("changed")),
                "out_dir": str(out_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


def apply_postprocess_policy(
    prediction_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    rule_name: str,
) -> list[dict[str, Any]]:
    features_by_index = {int(row["query_index"]): row for row in feature_rows}
    post_rows: list[dict[str, Any]] = []
    for prediction_row in prediction_rows:
        query_index = int(prediction_row["query_index"])
        feature_row = features_by_index.get(query_index)
        if feature_row is None:
            raise KeyError(f"Missing feature row for query_index={query_index}")
        old_prediction = str(prediction_row.get(CONFIG.prediction_field) or "").strip().lower()
        new_prediction = candidate_rule_prediction(feature_row, rule_name)
        pred_label = prediction_to_label(CONFIG, new_prediction)
        label = prediction_row.get("label")
        correct = bool(pred_label == label) if pred_label is not None and label in (0, 1) else False
        row = dict(prediction_row)
        row[CONFIG.prediction_field] = new_prediction
        row["pred_label"] = pred_label
        row["correct"] = correct
        row["postprocess_policy"] = {
            "rule_name": rule_name,
            "old_prediction": old_prediction,
            "new_prediction": new_prediction,
            "changed": new_prediction != old_prediction,
        }
        if new_prediction != old_prediction:
            row["pre_postprocess_prediction"] = old_prediction
            summary = str(row.get("final_summary") or "").strip()
            suffix = (
                f" Postprocess policy {rule_name} changed the forced label "
                f"from {old_prediction} to {new_prediction}."
            )
            row["final_summary"] = (summary + suffix).strip()
        post_rows.append(row)
    return post_rows


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", required=True, help="Original batch predictions JSONL.")
    parser.add_argument("--features-jsonl", required=True, help="Feature rows JSONL extracted from the same batch.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rule-name", help="Candidate rule name to apply.")
    group.add_argument("--selected-rule-json", help="selected_rule.json written by specific_fallback_policy.")
    parser.add_argument("--out-dir", required=True, help="Directory for postprocessed predictions and metrics.")
    return parser.parse_args(argv)


def _resolve_rule_name(args: argparse.Namespace) -> str:
    if args.rule_name:
        return str(args.rule_name)
    payload = _read_json(Path(args.selected_rule_json))
    return str(payload["selected_rule"])


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _write_report(path: Path, manifest: dict[str, Any], metrics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    changed = [row for row in rows if row.get("postprocess_policy", {}).get("changed")]
    lines = [
        "# Bioavailability_Ma Specific Postprocess Report",
        "",
        f"rule_name: `{manifest['rule_name']}`",
        f"n_changed: {len(changed)}",
        f"accuracy: {metrics.get('accuracy')}",
        f"macro_f1: {metrics.get('macro_f1')}",
        f"confusion_matrix: `{metrics.get('confusion_matrix')}`",
        "",
        "## Changed Rows",
        "",
    ]
    for row in changed:
        policy = row.get("postprocess_policy") or {}
        lines.append(
            f"- idx {row.get('query_index')}: {policy.get('old_prediction')} -> "
            f"{policy.get('new_prediction')} label={row.get('label')} correct={row.get('correct')}"
        )
    if not changed:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
