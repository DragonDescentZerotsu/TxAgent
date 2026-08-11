"""Offline paired analysis for final-only train-ratio prior experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.final_decision_prior import (
    EVIDENCE_STATES,
    TRAIN_RATIO_TIEBREAK_V1,
    TrainRatioPrior,
    final_decision_validation_errors,
)
from tools.chembl_tool.common.json_utils import sha256_file
from tools.chembl_tool.paper_experiments.paired_binary_predictions import (
    paired_binary_summary,
)


REUSED_ARTIFACTS = (
    "retrieval.json",
    "single_molecule_reasoning_output.json",
    "group_reasoning_outputs.jsonl",
)


@dataclass(frozen=True)
class TrainRatioAnalysisSpec:
    task: str
    prediction_field: str
    source_batch: Path
    candidate_batch: Path
    prior: TrainRatioPrior


def analyze_train_ratio_prior(
    specs: list[TrainRatioAnalysisSpec],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_summaries: dict[str, Any] = {}
    all_samples: list[dict[str, Any]] = []
    for spec in specs:
        task_summary, sample_rows = _analyze_task(spec)
        task_summaries[spec.task] = task_summary
        all_samples.extend(sample_rows)
    return {
        "schema_version": "starling.train_ratio_prior_analysis.v1",
        "profile": TRAIN_RATIO_TIEBREAK_V1,
        "tasks": task_summaries,
        "any_task_promoted": any(
            summary["promotion_gate_pass"] for summary in task_summaries.values()
        ),
        "formal_test_started": False,
    }, all_samples


def _analyze_task(
    spec: TrainRatioAnalysisSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_rows = _prediction_rows(spec.source_batch)
    candidate_rows = _prediction_rows(spec.candidate_batch)
    if set(source_rows) != set(candidate_rows):
        raise ValueError(f"Source/candidate indices differ for {spec.task}")
    labels: list[int] = []
    source_predictions: list[int] = []
    candidate_predictions: list[int] = []
    state_counts: Counter[str] = Counter()
    prior_used_count = 0
    prior_flips = 0
    prior_corrected = 0
    prior_broken = 0
    non_prior_flips = 0
    non_prior_corrected = 0
    non_prior_broken = 0
    sample_rows: list[dict[str, Any]] = []
    for query_index in sorted(source_rows):
        source = source_rows[query_index]
        candidate = candidate_rows[query_index]
        if source.get("status") != "ok" or candidate.get("status") != "ok":
            raise ValueError(f"Incomplete paired row: {spec.task} index={query_index}")
        label = int(source["label"])
        if int(candidate["label"]) != label:
            raise ValueError(f"Label mismatch: {spec.task} index={query_index}")
        source_prediction = int(source["pred_label"])
        candidate_prediction = int(candidate["pred_label"])
        state = str(candidate.get("evidence_state") or "")
        prior_used = candidate.get("prior_used")
        if state not in EVIDENCE_STATES or not isinstance(prior_used, bool):
            raise ValueError(
                f"Missing decision audit fields: {spec.task} index={query_index}"
            )
        validation_errors = final_decision_validation_errors(
            {
                spec.prediction_field: candidate[spec.prediction_field],
                "evidence_state": state,
                "prior_used": prior_used,
            },
            profile=TRAIN_RATIO_TIEBREAK_V1,
            prior=spec.prior,
            prediction_field=spec.prediction_field,
        )
        if validation_errors:
            raise ValueError(
                f"Invalid decision audit: {spec.task} index={query_index} "
                f"errors={validation_errors}"
            )
        flipped = source_prediction != candidate_prediction
        corrected = source_prediction != label and candidate_prediction == label
        broken = source_prediction == label and candidate_prediction != label
        state_counts[state] += 1
        if prior_used:
            prior_used_count += 1
            prior_flips += int(flipped)
            prior_corrected += int(corrected)
            prior_broken += int(broken)
        else:
            non_prior_flips += int(flipped)
            non_prior_corrected += int(corrected)
            non_prior_broken += int(broken)
        labels.append(label)
        source_predictions.append(source_prediction)
        candidate_predictions.append(candidate_prediction)
        sample_rows.append(
            {
                "task": spec.task,
                "query_index": query_index,
                "label": label,
                "source_prediction": source_prediction,
                "candidate_prediction": candidate_prediction,
                "evidence_state": state,
                "prior_used": prior_used,
                "prediction_flipped": flipped,
                "corrected": corrected,
                "broken": broken,
            }
        )
    paired = paired_binary_summary(labels, source_predictions, candidate_predictions)
    ci_low = paired["delta_macro_f1_bootstrap_95ci"][0]
    promote = bool(
        paired["delta_macro_f1"] > 0
        and ci_low > 0
        and paired["right_accuracy"] >= paired["left_accuracy"]
    )
    summary = {
        "source_batch": str(spec.source_batch),
        "candidate_batch": str(spec.candidate_batch),
        "paired": paired,
        "source_positive_predictions": sum(source_predictions),
        "candidate_positive_predictions": sum(candidate_predictions),
        "evidence_state_counts": dict(sorted(state_counts.items())),
        "prior_used_count": prior_used_count,
        "prior_used_fraction": prior_used_count / len(labels),
        "prior_used_prediction_flips": prior_flips,
        "prior_used_corrected": prior_corrected,
        "prior_used_broken": prior_broken,
        "non_prior_prediction_flips": non_prior_flips,
        "non_prior_corrected": non_prior_corrected,
        "non_prior_broken": non_prior_broken,
        "reuse_audit": _audit_reused_artifacts(spec, source_rows, candidate_rows),
        "promotion_gate_pass": promote,
    }
    return summary, sample_rows


def _audit_reused_artifacts(
    spec: TrainRatioAnalysisSpec,
    source_rows: dict[int, dict[str, Any]],
    candidate_rows: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    checked = 0
    manifest_checked = 0
    for query_index in sorted(source_rows):
        source_run = Path(source_rows[query_index]["run_dir"])
        candidate_run = Path(candidate_rows[query_index]["run_dir"])
        manifest = json.loads(
            (candidate_run / "manifest.json").read_text(encoding="utf-8")
        )
        if str(manifest.get("final_only_source_batch") or "") != str(spec.source_batch):
            raise ValueError(
                f"Final-only source mismatch: {spec.task} index={query_index}"
            )
        if manifest.get("final_decision_profile") != TRAIN_RATIO_TIEBREAK_V1:
            raise ValueError(
                f"Final profile mismatch: {spec.task} index={query_index}"
            )
        manifest_checked += 1
        for name in REUSED_ARTIFACTS:
            source_path = source_run / name
            candidate_path = candidate_run / name
            if sha256_file(source_path) != sha256_file(candidate_path):
                raise ValueError(
                    f"Reused artifact differs: {spec.task} index={query_index} "
                    f"artifact={name}"
                )
            checked += 1
    return {
        "status": "pass",
        "artifact_names": list(REUSED_ARTIFACTS),
        "n_artifact_pairs_checked": checked,
        "n_candidate_manifests_checked": manifest_checked,
        "n_mismatches": 0,
    }


def render_train_ratio_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Train-ratio tie-break v1 valid result",
        "",
        "Only the final stage was rerun; retrieval, single, and group artifacts were copied from frozen full-flat controls.",
        "",
        "| task | source F1 | candidate F1 | delta | 95% CI | source acc | candidate acc | prior used | corrected / broken | non-prior corrected / broken | promote |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    details: list[str] = []
    for task, task_summary in summary["tasks"].items():
        paired = task_summary["paired"]
        ci = paired["delta_macro_f1_bootstrap_95ci"]
        lines.append(
            f"| {task} | {paired['left_macro_f1']:.4f} | {paired['right_macro_f1']:.4f} | "
            f"{paired['delta_macro_f1']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"{paired['left_accuracy']:.4f} | {paired['right_accuracy']:.4f} | "
            f"{task_summary['prior_used_count']} | "
            f"{task_summary['prior_used_corrected']} / {task_summary['prior_used_broken']} | "
            f"{task_summary['non_prior_corrected']} / {task_summary['non_prior_broken']} | "
            f"{'PASS' if task_summary['promotion_gate_pass'] else 'FAIL'} |"
        )
        details.extend(
            [
                f"- `{task}` evidence states: `{json.dumps(task_summary['evidence_state_counts'], sort_keys=True)}`",
                f"- `{task}` copied-artifact audit: `{json.dumps(task_summary['reuse_audit'], sort_keys=True)}`",
            ]
        )
    lines.extend(
        [
            "",
            *details,
            "",
            "Promotion requires positive macro-F1 delta, a paired-bootstrap 95% lower bound above zero, and non-decreasing accuracy. Formal test was not started by this runner.",
            "",
        ]
    )
    return "\n".join(lines)


def _prediction_rows(batch: Path) -> dict[int, dict[str, Any]]:
    metrics = json.loads((batch / "metrics.json").read_text(encoding="utf-8"))
    if int(metrics.get("n_failed_runs") or 0) != 0:
        raise ValueError(f"Batch has failed runs: {batch}")
    rows = [
        json.loads(line)
        for line in (batch / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indexed = {int(row["query_index"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate query indices: {batch}")
    return indexed
