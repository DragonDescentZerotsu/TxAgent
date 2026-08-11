"""Audit whether legacy Skin_Reaction errors were recoverable at the final stage.

This is a deterministic, no-model-call diagnostic over the frozen scaffold-valid
full-mechanism traces.  It does not estimate the performance of the corrected
prompt profile.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.json_utils import (
    atomic_output_path,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.common.prompt_profile import prompt_profile_from_manifest
from tools.chembl_tool.paper_experiments.paired_binary_predictions import (
    paired_binary_summary,
)
from tools.chembl_tool.tasks.skin_reaction.prompt_profiles import (
    HISTORICAL_SKIN_PROMPT_PROFILE,
)


SCHEMA_VERSION = "skin_reasoning_bottleneck_audit.v3"
ALIGNED_GROUPS = frozenset({"Mechanism.tier_1", "Mechanism.tier_2"})
POSITIVE_DIRECTIONS = frozenset(
    {"supports_skin_reaction_risk", "sensitization_risk", "supports_sensitizer"}
)
NEGATIVE_DIRECTIONS = frozenset(
    {"argues_against_skin_reaction_risk", "argues_against_sensitizer"}
)
OUT_OF_SCOPE_FINAL_TYPES = frozenset(
    {"phototoxicity", "irritation_or_corrosion", "exposure_context_only"}
)
DEFAULT_SOURCE_BATCH = Path(
    "outputs/paper/"
    "molecular_evidence_agent_starling_scaffold_record_supported_v2_valid_gpt_oss_120b/"
    "runs_identity_blind_parent_disjoint/skin_reaction/"
    "skin_reaction__starling_full_mechanism"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/paper/"
    "skin_reaction_task_alignment_audit_record_supported_v2_valid_gpt_oss_120b"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary, rows = audit_batch(
        Path(args.source_batch),
        expected_rows=args.expected_rows,
        reference_batch=Path(args.reference_batch) if args.reference_batch else None,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "summary.json", summary)
    write_jsonl_atomic(output_dir / "sample_audit.jsonl", rows)
    with atomic_output_path(output_dir / "report.md") as temporary:
        temporary.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"summary": summary, "output_dir": str(output_dir)}, indent=2))
    return 0


def audit_batch(
    source_batch: Path,
    *,
    expected_rows: int = 245,
    reference_batch: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions_path = source_batch / "predictions.jsonl"
    metrics_path = source_batch / "metrics.json"
    manifest_path = source_batch / "manifest.json"
    for path in (predictions_path, metrics_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)

    predictions = _read_jsonl(predictions_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    batch_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_source_contract(predictions, metrics, batch_manifest, expected_rows)
    prompt_profile = prompt_profile_from_manifest(
        batch_manifest,
        historical_profile=HISTORICAL_SKIN_PROMPT_PROFILE,
    )

    rows = [_audit_sample(source_batch, prediction) for prediction in predictions]
    rows.sort(key=lambda row: row["query_index"])
    errors = [row for row in rows if not row["correct"]]
    category_counts = Counter(row["bottleneck_category"] for row in errors)
    error_type_counts = Counter(row["error_type"] for row in errors)
    scope_counts = Counter(
        row["error_type"] for row in errors if row["scope_contamination"]
    )
    signal_state_counts = Counter(
        f"{'correct' if row['correct'] else 'incorrect'}_label{row['label']}:{row['aligned_signal_state']}"
        for row in rows
    )

    n_errors = len(errors)
    n_final_recoverable = category_counts["final_recoverable"]
    n_upstream = n_errors - n_final_recoverable
    n_scope = sum(scope_counts.values())
    n_scope_and_recoverable = sum(
        row["scope_contamination"]
        and row["bottleneck_category"] == "final_recoverable"
        for row in errors
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis_type": "deterministic_existing_trace_audit_no_model_calls",
        "source": {
            "batch": str(source_batch),
            "input_jsonl": batch_manifest.get("input_jsonl"),
            "dataset_lineage": "record_supported_v2",
            "benchmark_split": "scaffold",
            "evaluation_subset": "valid",
            "condition": "starling_full_mechanism",
            "visibility_mode": batch_manifest.get("visibility_mode"),
            "neighbor_identity_policy": batch_manifest.get("neighbor_identity_policy"),
            "prompt_profile": prompt_profile,
            "predictions_sha256": sha256_file(predictions_path),
            "metrics_sha256": sha256_file(metrics_path),
        },
        "source_metrics": {
            "n": len(rows),
            "n_errors": n_errors,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "error_type_counts": dict(sorted(error_type_counts.items())),
        },
        "decision_rule": {
            "aligned_groups": sorted(ALIGNED_GROUPS),
            "strict_signal_gate": (
                "useful=true, transferability in {high,moderate}, confidence in "
                "{high,moderate}, and explicit positive/negative direction"
            ),
            "final_recoverable": (
                "at least one gold-aligned strict Tier 1/2 signal and no opposite strict signal"
            ),
            "scope_contamination": sorted(OUT_OF_SCOPE_FINAL_TYPES),
        },
        "bottleneck_counts": {
            category: category_counts.get(category, 0)
            for category in (
                "final_recoverable",
                "upstream_conflict",
                "upstream_wrong_direction",
                "upstream_insufficient",
            )
        },
        "headline": {
            "final_recoverable": _rate_with_wilson(n_final_recoverable, n_errors),
            "upstream_not_final_recoverable": _rate_with_wilson(n_upstream, n_errors),
            "scope_contaminated_errors": _rate_with_wilson(n_scope, n_errors),
            "scope_contaminated_and_final_recoverable": n_scope_and_recoverable,
            "scope_contamination_by_error_type": dict(sorted(scope_counts.items())),
        },
        "aligned_signal_state_counts": dict(sorted(signal_state_counts.items())),
        "interpretation": {
            "primary": _primary_interpretation(n_final_recoverable, n_upstream),
            "scope_bug": _scope_interpretation(prompt_profile, n_scope),
            "causal_limit": (
                "This descriptive audit cannot predict the corrected profile's metric change and "
                "does not prove that a strict signal is chemically correct."
            ),
            "continuation_gate": "final_only_icl_deferred_not_run",
        },
    }
    if reference_batch is not None:
        summary["paired_reference"] = _paired_reference_summary(
            predictions,
            prompt_profile,
            reference_batch,
        )
    return summary, rows


def _primary_interpretation(n_final: int, n_upstream: int) -> str:
    if n_upstream > n_final:
        return "Upstream single/group interpretation is the dominant observed bottleneck."
    if n_final > n_upstream:
        return "Final aggregation is the dominant observed bottleneck."
    return "Upstream and final-recoverable error counts are tied."


def _scope_interpretation(prompt_profile: str, n_scope: int) -> str:
    if prompt_profile == HISTORICAL_SKIN_PROMPT_PROFILE:
        return f"The legacy broad-skin contract produced {n_scope} scope-contaminated errors."
    return (
        f"The aligned sensitization contract produced {n_scope} scope-contaminated errors; "
        "zero is the compliance target."
    )


def _paired_reference_summary(
    source_predictions: list[dict[str, Any]],
    source_profile: str,
    reference_batch: Path,
) -> dict[str, Any]:
    reference_path = reference_batch / "predictions.jsonl"
    reference_manifest_path = reference_batch / "manifest.json"
    reference = _read_jsonl(reference_path)
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    reference_profile = prompt_profile_from_manifest(
        reference_manifest,
        historical_profile=HISTORICAL_SKIN_PROMPT_PROFILE,
    )
    source_map = _prediction_map(source_predictions, "source")
    reference_map = _prediction_map(reference, "reference")
    if source_map.keys() != reference_map.keys():
        raise ValueError("Source/reference query-index sets do not match")
    indices = sorted(source_map)
    labels = [int(source_map[index]["label"]) for index in indices]
    if labels != [int(reference_map[index]["label"]) for index in indices]:
        raise ValueError("Source/reference gold labels do not match")
    left = [int(reference_map[index]["pred_label"]) for index in indices]
    right = [int(source_map[index]["pred_label"]) for index in indices]
    paired = paired_binary_summary(
        labels,
        left,
        right,
        seed=20260809,
    )
    return {
        "reference_batch": str(reference_batch),
        "reference_profile": reference_profile,
        "source_profile": source_profile,
        **paired,
        "causal_limit": (
            "Same-cohort paired observation; fresh model generation means the delta can "
            "include inference nondeterminism and is not a prompt-only causal estimate."
        ),
    }


def _prediction_map(
    rows: list[dict[str, Any]],
    name: str,
) -> dict[int, dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        query_index = int(row["query_index"])
        if query_index in by_index:
            raise ValueError(f"Duplicate {name} query_index: {query_index}")
        by_index[query_index] = row
    return by_index


def classify_aligned_signals(group_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    positive: list[str] = []
    negative: list[str] = []
    for output in group_outputs:
        group_id = str(output.get("group_id") or "")
        if group_id not in ALIGNED_GROUPS or output.get("status") != "ok":
            continue
        content = (output.get("llm") or {}).get("content") or {}
        useful = content.get("useful_for_skin_reaction_reasoning")
        if useful is None:
            useful = content.get("useful_for_skin_sensitization_reasoning")
        if useful is not True:
            continue
        if content.get("transferability") not in {"high", "moderate"}:
            continue
        if content.get("confidence") not in {"high", "moderate"}:
            continue
        direction = str(
            content.get("evidence_direction")
            or content.get("sensitization_evidence_direction")
            or ""
        )
        if direction in POSITIVE_DIRECTIONS:
            positive.append(group_id)
        elif direction in NEGATIVE_DIRECTIONS:
            negative.append(group_id)
    state = _signal_state(bool(positive), bool(negative))
    return {
        "state": state,
        "positive_group_ids": sorted(positive),
        "negative_group_ids": sorted(negative),
    }


def classify_error(label: int, signal_state: str) -> str:
    gold_only = (
        label == 1 and signal_state == "positive_only"
    ) or (
        label == 0 and signal_state == "negative_only"
    )
    wrong_only = (
        label == 1 and signal_state == "negative_only"
    ) or (
        label == 0 and signal_state == "positive_only"
    )
    if gold_only:
        return "final_recoverable"
    if signal_state == "both":
        return "upstream_conflict"
    if wrong_only:
        return "upstream_wrong_direction"
    return "upstream_insufficient"


def _audit_sample(source_batch: Path, prediction: dict[str, Any]) -> dict[str, Any]:
    query_index = int(prediction["query_index"])
    run_id = str(prediction["run_id"])
    run_dir = source_batch / "runs" / run_id
    group_path = run_dir / "group_reasoning_outputs.jsonl"
    final_path = run_dir / "final_reasoning_output.json"
    manifest_path = run_dir / "manifest.json"
    for path in (group_path, final_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(run_manifest.get("query_index")) != query_index:
        raise ValueError(f"Query index mismatch in {run_dir}")
    label = int(prediction["label"])
    pred_label = int(prediction["pred_label"])
    if int(run_manifest.get("query_label_for_eval_only")) != label:
        raise ValueError(f"Label mismatch in {run_dir}")
    groups = _read_jsonl(group_path)
    signals = classify_aligned_signals(groups)
    final_content = (
        json.loads(final_path.read_text(encoding="utf-8")).get("llm") or {}
    ).get("content") or {}
    main_type = str(final_content.get("main_evidence_type") or "")
    correct = label == pred_label
    return {
        "query_index": query_index,
        "run_id": run_id,
        "label": label,
        "pred_label": pred_label,
        "correct": correct,
        "error_type": "none" if correct else ("false_positive" if label == 0 else "false_negative"),
        "aligned_signal_state": signals["state"],
        "positive_signal_group_ids": signals["positive_group_ids"],
        "negative_signal_group_ids": signals["negative_group_ids"],
        "bottleneck_category": "correct" if correct else classify_error(label, signals["state"]),
        "final_main_evidence_type": main_type,
        "scope_contamination": (not correct and main_type in OUT_OF_SCOPE_FINAL_TYPES),
        "run_dir": str(run_dir),
    }


def _validate_source_contract(
    predictions: list[dict[str, Any]],
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    expected_rows: int,
) -> None:
    if len(predictions) != expected_rows:
        raise ValueError(f"Expected {expected_rows} predictions, found {len(predictions)}")
    indices = [int(row["query_index"]) for row in predictions]
    if len(set(indices)) != expected_rows:
        raise ValueError("Prediction query indices are not unique")
    if int(metrics.get("n_successful") or 0) != expected_rows:
        raise ValueError("Source metrics are incomplete")
    if int(metrics.get("n_failed_runs") or 0) != 0:
        raise ValueError("Source metrics contain failed runs")
    input_jsonl = str(manifest.get("input_jsonl") or "")
    if "record_supported_v2/Skin_Reaction/scaffold/valid.jsonl" not in input_jsonl:
        raise ValueError(f"Unexpected audit input: {input_jsonl}")
    if manifest.get("visibility_mode") != "identity_blind":
        raise ValueError("Expected identity_blind source")
    if manifest.get("neighbor_identity_policy") != "parent_disjoint":
        raise ValueError("Expected parent_disjoint source")


def _signal_state(has_positive: bool, has_negative: bool) -> str:
    if has_positive and has_negative:
        return "both"
    if has_positive:
        return "positive_only"
    if has_negative:
        return "negative_only"
    return "none"


def _rate_with_wilson(count: int, total: int) -> dict[str, Any]:
    if total <= 0:
        return {"count": count, "total": total, "rate": None, "wilson_95": None}
    z = 1.959963984540054
    proportion = count / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return {
        "count": count,
        "total": total,
        "rate": round(proportion, 6),
        "wilson_95": [round(center - radius, 6), round(center + radius, 6)],
    }


def render_report(summary: dict[str, Any]) -> str:
    counts = summary["bottleneck_counts"]
    headline = summary["headline"]
    source = summary["source_metrics"]
    prompt_profile = summary["source"]["prompt_profile"]
    if prompt_profile == HISTORICAL_SKIN_PROMPT_PROFILE:
        profile_interpretation = (
            "Fixing the Skin label scope remains necessary because phototoxicity, "
            "irritation/corrosion, generic local damage, and exposure do not define the "
            "benchmark label. The corrected profile must update single, group, and final "
            "contracts together."
        )
    else:
        profile_interpretation = (
            "This run already uses the corrected sensitization/contact-allergy scope. "
            "The scope-contamination count therefore tests whether the new contract is "
            "actually obeyed in completed traces; remaining errors must still be separated "
            "into upstream and final-stage paths."
        )
    paired = summary.get("paired_reference")
    false_positive = source["error_type_counts"].get("false_positive", 0)
    false_negative = source["error_type_counts"].get("false_negative", 0)
    final_rate = _format_rate(headline["final_recoverable"])
    upstream_rate = _format_rate(headline["upstream_not_final_recoverable"])
    scope_rate = _format_rate(headline["scope_contaminated_errors"])
    paired_text = ""
    if paired:
        ci_low, ci_high = paired["delta_macro_f1_bootstrap_95ci"]
        paired_text = f"""
## Paired reference comparison

`{paired['reference_profile']}` to `{paired['source_profile']}` changed macro-F1 by {paired['delta_macro_f1']:+.4f} (paired-bootstrap 95% CI [{ci_low:+.4f}, {ci_high:+.4f}]), with {paired['prediction_flips']} prediction flips, {paired['right_only_correct']} source-only correct and {paired['left_only_correct']} reference-only correct (McNemar p={paired['mcnemar_exact_p']:.4g}). This is a same-cohort paired observation, not a prompt-only causal estimate, because fresh generation can include inference nondeterminism.
"""
    return f"""# Skin_Reaction task-alignment trace audit

## Executive summary

This deterministic audit used all {source['n']} frozen scaffold-valid `full_mechanism` traces from `{prompt_profile}` and made no model calls. The agent made {source['n_errors']} errors ({false_positive} false positives and {false_negative} false negatives).

Only **{headline['final_recoverable']['count']}/{headline['final_recoverable']['total']} ({final_rate})** errors were strictly final-recoverable: Tier 1/2 already contained a confident, transferable signal in the gold direction and no opposite signal. The other **{headline['upstream_not_final_recoverable']['count']}/{headline['upstream_not_final_recoverable']['total']} ({upstream_rate})** lacked that clean upstream basis.

The final answer cited an out-of-scope main evidence type in **{headline['scope_contaminated_errors']['count']}/{headline['scope_contaminated_errors']['total']} ({scope_rate})** errors. Only **{headline['scope_contaminated_and_final_recoverable']}** of those rows also met the strict final-recoverable rule.

## Error decomposition

| Category | Errors | Plain-language meaning |
|---|---:|---|
| Final-recoverable | {counts['final_recoverable']} | Tier 1/2 already cleanly supported gold; final still chose wrong |
| Upstream conflict | {counts['upstream_conflict']} | Tier 1/2 contained confident signals in both directions |
| Upstream wrong direction | {counts['upstream_wrong_direction']} | Tier 1/2 cleanly supported the wrong class |
| Upstream insufficient | {counts['upstream_insufficient']} | Tier 1/2 contained no strict directional signal |

## What this means

The observed bottleneck is mainly before final aggregation. {profile_interpretation} However, a final-only intervention cannot use a correct signal that the upstream branches never produced.

The final-only ICL continuation is intentionally deferred. A separate versioned run is required to evaluate a changed profile.
{paired_text}

## Method and limits

A strict signal was counted only for Tier 1/2 when the group marked it useful, transferability and confidence were high/moderate, and its direction was explicitly positive or negative. This rule is intentionally conservative and label-blind until the audit comparison. It is a descriptive trace decomposition, not proof that the group chemistry was correct and not a causal estimate of any future prompt change.

No chart is included because four exact category counts are clearer than a figure. See `summary.json` for the machine-readable aggregate and `sample_audit.jsonl` for all sample-level classifications.
"""


def _format_rate(row: dict[str, Any]) -> str:
    rate = row.get("rate")
    return "n/a" if rate is None else f"{float(rate):.1%}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", default=str(DEFAULT_SOURCE_BATCH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--expected-rows", type=int, default=245)
    parser.add_argument(
        "--reference-batch",
        default="",
        help="Optional same-cohort batch for paired prediction comparison.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
