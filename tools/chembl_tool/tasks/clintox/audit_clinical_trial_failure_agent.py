"""Audit strict clinical-trial-failure agent runs and their retrieval contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

DIRECT_GROUP_ID = "Direct.clinical_trial_failure"
DEFAULT_CONDITIONS = (
    "clintox__none",
    "clintox__starling_direct",
    "clintox__starling_full_flat",
    "clintox__starling_full_mechanism",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.run_root)
    summaries = {
        condition: _audit_condition(root / condition)
        for condition in args.conditions
    }
    baseline = _prediction_map(root / args.baseline / "predictions.jsonl")
    for condition, summary in summaries.items():
        current = _prediction_map(root / condition / "predictions.jsonl")
        summary["paired_vs_baseline"] = _paired_flips(baseline, current)

    report = {
        "audit_version": "clintox_clinical_trial_failure_agent_trace_audit.v3",
        "run_root": str(root),
        "strict_direct_group_id": DIRECT_GROUP_ID,
        "conditions": summaries,
    }
    output = Path(args.output) if args.output else root / "clinical_trial_failure_trace_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--baseline", default="clintox__none")
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prediction_map(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["query_index"]): row for row in _read_jsonl(path)}


def _audit_condition(batch_dir: Path) -> dict[str, Any]:
    predictions = _prediction_map(batch_dir / "predictions.jsonl")
    metrics = _read_json(batch_dir / "metrics.json")
    manifest = _read_json(batch_dir / "manifest.json")
    prompt_profile = str(manifest.get("task_prompt_profile") or "unversioned")
    counts: Counter[str] = Counter()
    toxic_indices: list[int] = []
    toxic_without_direct: list[int] = []
    positive_label_without_direct: list[int] = []
    prompt_label_leak_indices: list[int] = []
    non_direct_vote_violations: list[dict[str, Any]] = []
    direct_provenance_violations: list[dict[str, Any]] = []
    missing_direct_group_indices: list[int] = []
    direct_neighbor_counts: list[int] = []

    for index, prediction in sorted(predictions.items()):
        run_dir = Path(prediction.get("run_dir") or prediction.get("source_run_dir") or "")
        if not run_dir.is_absolute():
            run_dir = Path.cwd() / run_dir
        retrieval = _read_json(run_dir / "retrieval.json")
        groups = retrieval.get("groups") or []
        groups_by_id = {
            str(group.get("group_id") or ""): group
            for group in groups
        }
        direct_rows = _direct_evidence_rows(groups)
        has_direct = bool(direct_rows)
        direct_neighbor_counts.append(len(direct_rows))
        counts["samples_with_direct_evidence"] += int(has_direct)
        label = int(prediction["label"])
        if label == 1:
            counts["positive_label_samples"] += 1
            counts["positive_label_samples_with_direct_evidence"] += int(has_direct)
            if not has_direct:
                positive_label_without_direct.append(index)

        is_toxic = prediction.get("clintox_prediction") == "toxic"
        if is_toxic:
            toxic_indices.append(index)
            counts["toxic_predictions_with_direct_evidence"] += int(has_direct)
            if not has_direct:
                toxic_without_direct.append(index)

        group_ids = {str(group.get("group_id") or "") for group in groups}
        if manifest.get("experiment_mode") == "full_mechanism" and DIRECT_GROUP_ID not in group_ids:
            missing_direct_group_indices.append(index)

        group_path = run_dir / "group_reasoning_outputs.jsonl"
        if group_path.exists():
            for group_output in _read_jsonl(group_path):
                _add_validation_counts(counts, group_output)
                group_id = str(group_output.get("group_id") or "")
                group_has_direct = bool(
                    _direct_evidence_rows([groups_by_id.get(group_id, {})])
                )
                direction = str(
                    ((group_output.get("llm") or {}).get("content") or {}).get("evidence_direction") or ""
                )
                if not group_has_direct and direction in {
                    "supports_toxicity_trial_failure",
                    "argues_against_toxicity_trial_failure",
                }:
                    non_direct_vote_violations.append(
                        {"query_index": index, "group_id": group_id, "evidence_direction": direction}
                    )
                direct_status = str(
                    ((group_output.get("llm") or {}).get("content") or {}).get(
                        "direct_evidence_status"
                    )
                    or ""
                )
                if direct_status:
                    status_claims_direct = direct_status != "no_direct_rows"
                    if status_claims_direct != group_has_direct:
                        direct_provenance_violations.append(
                            {
                                "query_index": index,
                                "stage": "group",
                                "group_id": group_id,
                                "direct_evidence_status": direct_status,
                                "retrieval_has_direct_row": group_has_direct,
                            }
                        )

        if _llm_messages_contain_eval_label(run_dir):
            prompt_label_leak_indices.append(index)
        for path in (
            run_dir / "single_molecule_reasoning_output.json",
            run_dir / "final_reasoning_output.json",
        ):
            if path.exists():
                _add_validation_counts(counts, _read_json(path))
        final_path = run_dir / "final_reasoning_output.json"
        if final_path.exists():
            final_status = str(
                ((_read_json(final_path).get("llm") or {}).get("content") or {}).get(
                    "direct_evidence_status"
                )
                or ""
            )
            if final_status:
                status_claims_direct = final_status != "no_direct_analog_retrieved"
                if status_claims_direct != has_direct:
                    direct_provenance_violations.append(
                        {
                            "query_index": index,
                            "stage": "final",
                            "direct_evidence_status": final_status,
                            "retrieval_has_direct_row": has_direct,
                        }
                    )

    counts["toxic_predictions"] = len(toxic_indices)
    counts["positive_predictions_without_direct_evidence"] = len(toxic_without_direct)
    counts["non_direct_vote_violations"] = len(non_direct_vote_violations)
    counts["direct_provenance_violations"] = len(direct_provenance_violations)
    counts["prompt_label_leak_samples"] = len(prompt_label_leak_indices)
    counts["full_mechanism_samples_missing_direct_group"] = len(missing_direct_group_indices)
    return {
        "batch_dir": str(batch_dir),
        "model": manifest.get("model"),
        "task_prompt_profile": prompt_profile,
        "experiment_mode": manifest.get("experiment_mode"),
        "neighbor_identity_policy": manifest.get("neighbor_identity_policy"),
        "visibility_mode": manifest.get("visibility_mode"),
        "harness_prefetch_tools": manifest.get("harness_prefetch_tools"),
        "metrics": {
            key: metrics.get(key)
            for key in (
                "n_total",
                "n_successful",
                "n_failed_runs",
                "accuracy",
                "macro_f1",
                "positive_class_precision",
                "positive_class_recall",
                "positive_class_f1",
                "confusion_matrix",
                "prediction_distribution",
            )
        },
        "counts": dict(sorted(counts.items())),
        "toxic_prediction_indices": toxic_indices,
        "toxic_without_direct_evidence_indices": toxic_without_direct,
        "positive_label_without_direct_evidence_indices": positive_label_without_direct,
        "non_direct_vote_violations": non_direct_vote_violations,
        "direct_provenance_violations": direct_provenance_violations,
        "full_mechanism_missing_direct_group_indices": missing_direct_group_indices,
        "prompt_label_leak_indices": prompt_label_leak_indices,
        "direct_evidence_rows_per_sample": {
            "min": min(direct_neighbor_counts, default=0),
            "max": max(direct_neighbor_counts, default=0),
            "mean": round(sum(direct_neighbor_counts) / len(direct_neighbor_counts), 6)
            if direct_neighbor_counts
            else 0.0,
        },
    }


def _direct_evidence_rows(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        for neighbor in group.get("neighbors") or []:
            for evidence in neighbor.get("evidence_rows") or []:
                minimal_group = (((evidence.get("minimal_evidence") or {}).get("group") or {}).get("id"))
                if minimal_group == DIRECT_GROUP_ID or evidence.get("group_id") == DIRECT_GROUP_ID:
                    rows.append(evidence)
    return rows


def _llm_messages_contain_eval_label(run_dir: Path) -> bool:
    paths = [run_dir / "single_molecule_reasoning_output.json", run_dir / "final_reasoning_output.json"]
    group_path = run_dir / "group_reasoning_outputs.jsonl"
    payloads: list[Any] = []
    for path in paths:
        if path.exists():
            payloads.append(_read_json(path))
    if group_path.exists():
        payloads.extend(_read_jsonl(group_path))
    for payload in payloads:
        messages = ((payload.get("llm") or {}).get("messages") or [])
        serialized = json.dumps(messages, sort_keys=True)
        if "query_label_for_eval_only" in serialized:
            return True
    return False


def _add_validation_counts(counts: Counter[str], payload: dict[str, Any]) -> None:
    validation = ((payload.get("llm") or {}).get("structured_output_validation") or {})
    if not validation:
        return
    counts["structured_validation_calls"] += 1
    counts["structured_validation_retried_calls"] += int(
        bool(validation.get("retried")) or int(validation.get("attempt_count") or 1) > 1
    )
    counts["structured_validation_invalid_calls"] += int(not bool(validation.get("valid")))


def _paired_flips(
    baseline: dict[int, dict[str, Any]],
    current: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    beneficial = 0
    harmful = 0
    for index in sorted(set(baseline) & set(current)):
        base = baseline[index]
        candidate = current[index]
        before = base.get("clintox_prediction")
        after = candidate.get("clintox_prediction")
        if before == after:
            continue
        label = int(candidate["label"])
        before_label = int(before == "toxic")
        after_label = int(after == "toxic")
        is_beneficial = after_label == label and before_label != label
        beneficial += int(is_beneficial)
        harmful += int(not is_beneficial)
        rows.append(
            {
                "query_index": index,
                "label": label,
                "before": before,
                "after": after,
                "effect": "beneficial" if is_beneficial else "harmful",
            }
        )
    return {"n_flips": len(rows), "beneficial": beneficial, "harmful": harmful, "rows": rows}


if __name__ == "__main__":
    raise SystemExit(main())
