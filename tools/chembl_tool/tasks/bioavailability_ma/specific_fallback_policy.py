"""Feature extraction and diagnostics for Bioavailability_Ma fallback states.

The v12 specific pipeline separates deterministic source-level states from
``fallback_uncertain`` cases.  This module keeps the next step deliberately
calibration-friendly: extract fixed evidence features from completed final
artifacts, then evaluate simple candidate fallback policies without changing
the live reasoning pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.task_workflows.reasoning_batch import compute_metrics, prediction_to_label
from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch_specific import CONFIG


RULE_NAMES = [
    "current_prediction",
    "force_state_or_original_prediction",
    "force_state_or_high",
    "force_state_or_low",
    "force_state_or_soft_net_visible",
    "force_state_or_conservative_high_fallback",
    "force_state_or_source_balanced",
    "force_state_or_high_unless_low_anchor",
    "force_state_or_mechanism_low_blockers",
    "force_state_or_starling_soft_high_anchor",
    "force_state_or_no_evidence_high_fallback",
    "force_state_or_valid_selected_high_rescues",
]

PRIOR_TSV_FIELDS = [
    "query_index",
    "run_id",
    "smiles",
    "label",
    "prediction",
    "pred_label",
    "correct",
    "status",
    "final_status",
    "decision_state",
    "forced_class",
    "fallback_candidate_class",
    "recommended_class",
    "recommendation_strength",
    "source_level_consensus",
    "n_eligible_clean_direct_vote_sources",
    "eligible_high_sources",
    "eligible_low_sources",
    "eligible_straddling_sources",
    "n_clean_but_not_direct_vote_sources",
    "soft_high_sources",
    "soft_low_sources",
    "soft_straddling_sources",
    "net_soft_context_direction",
    "n_review_required_context_only_sources",
    "review_high_sources",
    "review_low_sources",
    "review_straddling_sources",
    "starling_source_count",
    "starling_high_sources",
    "starling_low_sources",
    "starling_straddling_sources",
    "transfer_clean_parent_or_close_analog_sources",
    "transfer_same_active_moiety_salt_or_freebase_sources",
    "transfer_active_metabolite_or_prodrug_sources",
    "transfer_special_formulation_or_route_sources",
    "transfer_relative_or_proxy_sources",
    "transfer_non_swallowed_route_sources",
    "transfer_total_radioactivity_sources",
    "transfer_anchor_candidate_sources",
    "transfer_anchor_high_sources",
    "transfer_anchor_low_sources",
    "transfer_anchor_straddling_sources",
    "alert_permanent_quaternary_charge",
    "alert_beta_lactam_anionic",
    "alert_dihydropyridine_diester",
    "alert_flat_high_logp_low_tpsa_hbd0",
    "alert_ester_prodrug_or_active_moiety",
    "alert_high_ionization_low_permeability",
    "oral_bioavailability_prior",
    "absorption_prior",
    "solubility_or_dissolution_prior",
    "metabolism_or_clearance_prior",
    "strong_limiting_factor_count",
    "moderate_analog_risk_factor_count",
    "risk_for_lower_f_factor_count",
    "supports_higher_f_factor_count",
    "class_constraint_count",
    "audit_warning_count",
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = _load_or_extract_rows(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(out_dir / "features.jsonl", rows)
    _write_tsv(out_dir / "features.tsv", rows)
    rule_rows, rule_metrics = evaluate_candidate_rules(rows)
    _write_jsonl(out_dir / "rule_predictions.jsonl", rule_rows)
    summary = summarize_feature_rows(rows)
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "rule_metrics.json", rule_metrics)
    selected_payload = None
    if args.calibration_features_jsonl:
        calibration_rows = _read_jsonl(Path(args.calibration_features_jsonl))
        selected_payload = select_candidate_rule(
            calibration_rows,
            selection_metric=args.selection_metric,
            min_negative_recall=args.min_negative_recall,
        )
        selected_rule = selected_payload["selected_rule"]
        selected_rows, selected_metrics = evaluate_single_rule(rows, selected_rule)
        selected_payload["evaluation_metrics"] = selected_metrics
        _write_json(out_dir / "selected_rule.json", selected_payload)
        _write_jsonl(out_dir / "selected_rule_predictions.jsonl", selected_rows)
        _write_json(out_dir / "selected_rule_metrics.json", selected_metrics)

    print(
        json.dumps(
            {
                "n_features": len(rows),
                "summary": summary,
                "rule_metrics": {
                    name: {
                        "accuracy": metrics["metrics"]["accuracy"],
                        "macro_f1": metrics["metrics"]["macro_f1"],
                        "confusion_matrix": metrics["metrics"]["confusion_matrix"],
                        "n_changed_vs_current": metrics["n_changed_vs_current"],
                    }
                    for name, metrics in rule_metrics.items()
                },
                "selected_rule": selected_payload,
                "out_dir": str(out_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", help="Batch predictions JSONL with final artifact paths.")
    parser.add_argument("--features-jsonl", help="Already extracted feature rows JSONL.")
    parser.add_argument("--out-dir", required=True, help="Directory for extracted features and rule diagnostics.")
    parser.add_argument(
        "--calibration-features-jsonl",
        help="Feature rows JSONL from a calibration split used to select one candidate rule.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["macro_f1", "accuracy", "negative_class_f1", "negative_class_recall", "positive_class_f1"],
        default="macro_f1",
        help="Metric used to select a candidate rule from calibration features.",
    )
    parser.add_argument(
        "--min-negative-recall",
        type=float,
        default=0.0,
        help="Optional calibration guardrail for low-class recall before selecting by metric.",
    )
    parser.add_argument(
        "--final-output-field",
        default="final_reasoning_output",
        help="Prediction-row field containing the final_reasoning_output.json path.",
    )
    parser.add_argument(
        "--skip-missing-final",
        action="store_true",
        help="Skip rows with missing final artifacts instead of raising an error.",
    )
    return parser.parse_args(argv)


def _load_or_extract_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.features_jsonl and args.predictions_jsonl:
        raise SystemExit("Use only one of --features-jsonl or --predictions-jsonl.")
    if args.features_jsonl:
        return _read_jsonl(Path(args.features_jsonl))
    if not args.predictions_jsonl:
        raise SystemExit("One of --features-jsonl or --predictions-jsonl is required.")
    return extract_feature_rows_from_predictions(
        Path(args.predictions_jsonl),
        final_output_field=args.final_output_field,
        skip_missing_final=args.skip_missing_final,
    )


def extract_feature_rows_from_predictions(
    predictions_jsonl: Path,
    *,
    final_output_field: str = "final_reasoning_output",
    skip_missing_final: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction_row in _read_jsonl(predictions_jsonl):
        final_path = Path(str(prediction_row.get(final_output_field) or ""))
        if not final_path.exists():
            if skip_missing_final:
                continue
            raise FileNotFoundError(f"Missing final output for query_index={prediction_row.get('query_index')}: {final_path}")
        final_output = _read_json(final_path)
        feature_row = extract_feature_row(prediction_row, final_output)
        feature_row["final_reasoning_output"] = str(final_path)
        rows.append(feature_row)
    return rows


def extract_feature_row(prediction_row: dict[str, Any], final_output: dict[str, Any]) -> dict[str, Any]:
    compiled = final_output.get("compiled_evidence_summary") or {}
    gate = compiled.get("evidence_gate_summary") or {}
    policy = gate.get("deterministic_decision_policy") or {}
    source_summary = compiled.get("source_level_direct_f_summary") or gate.get("source_level_direct_f_summary") or {}
    source_transfer_summary = compiled.get("source_transfer_summary") or {}
    guard = gate.get("factor_product_guard") or {}
    soft_context = policy.get("soft_direct_f_context") or gate.get("soft_direct_f_context") or {}
    single_prior = gate.get("single_molecule_prior") or guard.get("single_molecule_prior") or {}
    factor_signal_counts = gate.get("factor_signal_counts") or _factor_signal_counts_from_guard(guard)
    factor_strength = guard.get("factor_limiting_strength") or {}
    context_flags = compiled.get("context_flags") or gate.get("context_flags_present") or {}

    eligible_counts = source_summary.get("eligible_clean_source_direction_counts") or {}
    clean_context_counts = source_summary.get("clean_context_source_direction_counts") or {}
    review_counts = source_summary.get("review_context_source_direction_counts") or {}
    soft_counts = soft_context.get("soft_clean_source_direction_counts") or clean_context_counts
    source_rows = list(source_summary.get("source_summaries") or [])
    starling_counts = _direction_counts_for_sources(
        source_rows,
        lambda item: any("starling" in str(source).lower() for source in item.get("evidence_sources") or [])
        or any(str(identifier).startswith("STARLING_") for identifier in item.get("source_molecule_ids") or []),
    )
    chembl_counts = _direction_counts_for_sources(
        source_rows,
        lambda item: any(str(identifier).startswith("CHEMBL") for identifier in item.get("source_molecule_ids") or []),
    )

    class_constraints = guard.get("class_constraints") or policy.get("diagnostic_blocked_classes") or {}
    audit_warnings = guard.get("audit_warnings") or {}
    prior = {
        "oral_bioavailability_prior": single_prior.get("oral_bioavailability_prior"),
        "absorption_prior": single_prior.get("absorption_prior"),
        "solubility_or_dissolution_prior": single_prior.get("solubility_or_dissolution_prior"),
        "metabolism_or_clearance_prior": single_prior.get("metabolism_or_clearance_prior"),
        "single_molecule_prior_confidence": single_prior.get("confidence"),
    }

    feature_row = {
        "query_index": prediction_row.get("query_index"),
        "run_id": prediction_row.get("run_id"),
        "run_dir": prediction_row.get("run_dir"),
        "smiles": prediction_row.get("smiles"),
        "label": prediction_row.get("label"),
        "prediction": prediction_row.get(CONFIG.prediction_field),
        "pred_label": prediction_row.get("pred_label"),
        "confidence": prediction_row.get("confidence"),
        "correct": prediction_row.get("correct"),
        "status": prediction_row.get("status"),
        "final_status": prediction_row.get("final_status") or final_output.get("status"),
        "decision_state": policy.get("decision_state"),
        "forced_class": policy.get("forced_class"),
        "fallback_candidate_class": policy.get("fallback_candidate_class"),
        "recommended_class": policy.get("recommended_class"),
        "recommendation_strength": policy.get("recommendation_strength"),
        "policy_reason_codes": list(policy.get("reason_codes") or []),
        "policy_uncertainty_flags": list(policy.get("uncertainty_flags") or []),
        "source_level_consensus": source_summary.get("source_level_consensus")
        or policy.get("source_level_consensus"),
        "n_source_molecules_with_direct_f_values": int(source_summary.get("n_source_molecules_with_direct_f_values") or 0),
        "n_eligible_clean_direct_vote_sources": int(source_summary.get("n_eligible_clean_direct_vote_sources") or 0),
        "eligible_high_sources": _count_direction(eligible_counts, "supports_high_bioavailability"),
        "eligible_low_sources": _count_direction(eligible_counts, "argues_against_high_bioavailability"),
        "eligible_straddling_sources": _count_direction(eligible_counts, "mixed_threshold_straddling"),
        "n_clean_but_not_direct_vote_sources": int(source_summary.get("n_clean_but_not_direct_vote_sources") or 0),
        "soft_high_sources": _count_direction(soft_counts, "supports_high_bioavailability"),
        "soft_low_sources": _count_direction(soft_counts, "argues_against_high_bioavailability"),
        "soft_straddling_sources": _count_direction(soft_counts, "mixed_threshold_straddling"),
        "n_soft_clean_direct_f_sources": int(soft_context.get("n_soft_clean_direct_f_sources") or 0),
        "net_soft_context_direction": soft_context.get("net_soft_context_direction"),
        "soft_context_min_value_percent": soft_context.get("min_value_percent"),
        "soft_context_max_value_percent": soft_context.get("max_value_percent"),
        "n_review_required_context_only_sources": int(source_summary.get("n_review_required_context_only_sources") or 0),
        "review_high_sources": _count_direction(review_counts, "supports_high_bioavailability"),
        "review_low_sources": _count_direction(review_counts, "argues_against_high_bioavailability"),
        "review_straddling_sources": _count_direction(review_counts, "mixed_threshold_straddling"),
        "n_threshold_straddling_sources": int(source_summary.get("n_threshold_straddling_sources") or 0),
        "starling_source_count": starling_counts["n_sources"],
        "starling_high_sources": starling_counts["supports_high_bioavailability"],
        "starling_low_sources": starling_counts["argues_against_high_bioavailability"],
        "starling_straddling_sources": starling_counts["mixed_threshold_straddling"],
        "chembl_source_count": chembl_counts["n_sources"],
        "chembl_high_sources": chembl_counts["supports_high_bioavailability"],
        "chembl_low_sources": chembl_counts["argues_against_high_bioavailability"],
        "chembl_straddling_sources": chembl_counts["mixed_threshold_straddling"],
        "class_constraint_count": len(class_constraints),
        "class_constraint_keys": sorted(class_constraints),
        "audit_warning_count": len(audit_warnings),
        "audit_warning_keys": sorted(audit_warnings),
        "context_flag_count": len(_context_flag_names(context_flags)),
        "context_flag_names": _context_flag_names(context_flags),
        "has_prodrug_or_active_metabolite_flag": "prodrug_or_active_metabolite" in _context_flag_names(context_flags),
        "has_threshold_sensitive_flag": "threshold_sensitive_15_25_percent" in _context_flag_names(context_flags),
        "has_species_translation_flag": "species_translation" in _context_flag_names(context_flags),
    }
    feature_row.update(_source_transfer_features(source_transfer_summary, source_rows))
    feature_row.update(_single_molecule_alert_features(prediction_row))
    feature_row.update(prior)
    feature_row.update(_factor_features(factor_signal_counts, factor_strength))
    feature_row["strong_absorption_prior"] = (
        feature_row.get("absorption_prior") == "unfavorable"
        and feature_row.get("solubility_or_dissolution_prior") == "unfavorable"
    )
    feature_row["low_property_prior"] = _has_low_property_prior(feature_row)
    feature_row["is_fallback_uncertain"] = feature_row.get("decision_state") == "fallback_uncertain"
    return feature_row


def evaluate_candidate_rules(feature_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rule_prediction_rows: list[dict[str, Any]] = []
    metrics_by_rule: dict[str, Any] = {}

    for rule_name in RULE_NAMES:
        metric_rows, metrics = evaluate_single_rule(feature_rows, rule_name)
        rule_prediction_rows.extend(metric_rows)
        metrics_by_rule[rule_name] = metrics
    return rule_prediction_rows, metrics_by_rule


def evaluate_single_rule(feature_rows: list[dict[str, Any]], rule_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    transitions = Counter()
    changed_wrong_to_correct: list[Any] = []
    changed_correct_to_wrong: list[Any] = []
    for row in feature_rows:
        prediction = candidate_rule_prediction(row, rule_name)
        pred_label = prediction_to_label(CONFIG, prediction)
        label = row.get("label")
        correct = bool(pred_label == label) if pred_label is not None and label in (0, 1) else False
        current_prediction = str(row.get("prediction") or "missing").strip().lower()
        current_correct = bool(row.get("correct"))
        if prediction != current_prediction:
            transitions[f"{current_prediction}->{prediction}"] += 1
            if correct and not current_correct:
                changed_wrong_to_correct.append(row.get("query_index"))
            if current_correct and not correct:
                changed_correct_to_wrong.append(row.get("query_index"))
        metric_rows.append(
            {
                "query_index": row.get("query_index"),
                "rule_name": rule_name,
                "label": label,
                CONFIG.prediction_field: prediction,
                "pred_label": pred_label,
                "correct": correct,
                "status": row.get("status", "ok"),
                "decision_state": row.get("decision_state"),
                "current_prediction": current_prediction,
                "current_correct": current_correct,
            }
        )
    metrics = compute_metrics(CONFIG, metric_rows)
    return metric_rows, {
        "metrics": metrics,
        "n_changed_vs_current": sum(transitions.values()),
        "transitions_vs_current": dict(sorted(transitions.items())),
        "changed_wrong_to_correct_indices": changed_wrong_to_correct,
        "changed_correct_to_wrong_indices": changed_correct_to_wrong,
    }


def select_candidate_rule(
    calibration_rows: list[dict[str, Any]],
    *,
    selection_metric: str = "macro_f1",
    min_negative_recall: float = 0.0,
) -> dict[str, Any]:
    _, metrics_by_rule = evaluate_candidate_rules(calibration_rows)
    candidates = []
    for rule_name, payload in metrics_by_rule.items():
        metrics = payload["metrics"]
        negative_recall = float((metrics.get("per_class") or {}).get("0", {}).get("recall") or 0.0)
        if negative_recall < min_negative_recall:
            continue
        candidates.append(
            (
                _metric_value(metrics, selection_metric),
                negative_recall,
                float((metrics.get("per_class") or {}).get("0", {}).get("f1") or 0.0),
                -int(payload.get("n_changed_vs_current") or 0),
                rule_name,
                payload,
            )
        )
    if not candidates:
        raise ValueError(
            f"No candidate rule satisfied min_negative_recall={min_negative_recall}. "
            "Lower the guardrail or inspect calibration rule_metrics."
        )
    candidates.sort(reverse=True)
    _, _, _, _, selected_rule, selected_payload = candidates[0]
    return {
        "selection_metric": selection_metric,
        "min_negative_recall": min_negative_recall,
        "selected_rule": selected_rule,
        "selected_calibration_metrics": selected_payload,
        "all_calibration_rule_metrics": metrics_by_rule,
    }


def candidate_rule_prediction(row: dict[str, Any], rule_name: str) -> str:
    current_prediction = str(row.get("prediction") or "missing").strip().lower()
    decision_state = str(row.get("decision_state") or "").strip().lower()
    forced_class = str(row.get("forced_class") or "").strip().lower()
    if rule_name == "current_prediction":
        return current_prediction
    if decision_state in {"force_high", "force_low"} and forced_class in {"high", "low"}:
        return forced_class
    if rule_name == "force_state_or_original_prediction":
        return current_prediction
    if rule_name == "force_state_or_high":
        return "high"
    if rule_name == "force_state_or_low":
        return "low"
    if rule_name == "force_state_or_soft_net_visible":
        return _soft_net_visible_prediction(row, current_prediction)
    if rule_name == "force_state_or_conservative_high_fallback":
        return _conservative_high_fallback_prediction(row)
    if rule_name == "force_state_or_source_balanced":
        return _source_balanced_prediction(row, current_prediction)
    if rule_name == "force_state_or_high_unless_low_anchor":
        return _high_unless_low_anchor_prediction(row)
    if rule_name == "force_state_or_mechanism_low_blockers":
        return _mechanism_low_blocker_prediction(row, current_prediction)
    if rule_name == "force_state_or_starling_soft_high_anchor":
        return _starling_soft_high_anchor_prediction(row, current_prediction)
    if rule_name == "force_state_or_no_evidence_high_fallback":
        return _no_evidence_high_fallback_prediction(row, current_prediction)
    if rule_name == "force_state_or_valid_selected_high_rescues":
        prediction = _starling_soft_high_anchor_prediction(row, current_prediction)
        if prediction != current_prediction:
            return prediction
        return _no_evidence_high_fallback_prediction(row, current_prediction)
    raise ValueError(f"Unknown rule_name: {rule_name}")


def summarize_feature_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fallback_rows = [row for row in rows if row.get("decision_state") == "fallback_uncertain"]
    return {
        "n_rows": len(rows),
        "decision_state_counts": dict(sorted(Counter(row.get("decision_state") or "missing" for row in rows).items())),
        "fallback_uncertain_count": len(fallback_rows),
        "fallback_prediction_counts": dict(
            sorted(Counter(row.get("prediction") or "missing" for row in fallback_rows).items())
        ),
        "fallback_label_counts": dict(sorted(Counter(str(row.get("label")) for row in fallback_rows).items())),
        "fallback_correct_count": sum(1 for row in fallback_rows if row.get("correct")),
        "fallback_wrong_count": sum(1 for row in fallback_rows if not row.get("correct")),
        "net_soft_context_counts": dict(
            sorted(Counter(row.get("net_soft_context_direction") or "missing" for row in rows).items())
        ),
        "source_level_consensus_counts": dict(
            sorted(Counter(row.get("source_level_consensus") or "missing" for row in rows).items())
        ),
    }


def _soft_net_visible_prediction(row: dict[str, Any], current_prediction: str) -> str:
    net_soft = str(row.get("net_soft_context_direction") or "")
    strong_limiting = int(row.get("strong_limiting_factor_count") or 0) > 0
    strong_absorption = bool(row.get("strong_absorption_prior"))
    low_property_with_factor_risk = bool(row.get("low_property_prior")) and int(row.get("risk_for_lower_f_factor_count") or 0) > 0
    if net_soft == "soft_context_leans_high" and not strong_limiting and not strong_absorption:
        return "high"
    if net_soft == "soft_context_leans_low" and (
        strong_limiting or strong_absorption or low_property_with_factor_risk
    ):
        return "low"
    return current_prediction if current_prediction in {"high", "low"} else "high"


def _conservative_high_fallback_prediction(row: dict[str, Any]) -> str:
    if int(row.get("strong_limiting_factor_count") or 0) > 0:
        return "low"
    if bool(row.get("strong_absorption_prior")):
        return "low"
    if (
        row.get("oral_bioavailability_prior") == "low"
        and row.get("absorption_prior") == "unfavorable"
        and int(row.get("supports_higher_f_factor_count") or 0) == 0
    ):
        return "low"
    return "high"


def _source_balanced_prediction(row: dict[str, Any], current_prediction: str) -> str:
    soft_high = int(row.get("soft_high_sources") or 0)
    soft_low_or_straddling = int(row.get("soft_low_sources") or 0) + int(row.get("soft_straddling_sources") or 0)
    starling_high = int(row.get("starling_high_sources") or 0)
    starling_low_or_straddling = int(row.get("starling_low_sources") or 0) + int(row.get("starling_straddling_sources") or 0)
    review_low_or_straddling = int(row.get("review_low_sources") or 0) + int(row.get("review_straddling_sources") or 0)
    limiting = int(row.get("strong_limiting_factor_count") or 0) > 0 or bool(row.get("strong_absorption_prior"))
    low_risk = bool(row.get("low_property_prior")) or int(row.get("risk_for_lower_f_factor_count") or 0) > 0
    support_count = int(row.get("supports_higher_f_factor_count") or 0)
    risk_count = int(row.get("risk_for_lower_f_factor_count") or 0)

    if soft_high >= 3 and soft_high >= 2 * max(1, soft_low_or_straddling) and not limiting:
        return "high"
    if starling_high >= 2 and starling_high > starling_low_or_straddling and not limiting:
        return "high"
    if int(row.get("soft_low_sources") or 0) >= 2 and int(row.get("soft_low_sources") or 0) > soft_high and (limiting or low_risk):
        return "low"
    if starling_low_or_straddling >= 2 and starling_low_or_straddling > starling_high and (limiting or low_risk):
        return "low"
    if review_low_or_straddling >= 3 and starling_high == 0 and low_risk and risk_count >= support_count:
        return "low"
    return current_prediction if current_prediction in {"high", "low"} else "high"


def _high_unless_low_anchor_prediction(row: dict[str, Any]) -> str:
    low_anchor_count = (
        int(row.get("soft_low_sources") or 0)
        + int(row.get("soft_straddling_sources") or 0)
        + int(row.get("review_low_sources") or 0)
        + int(row.get("review_straddling_sources") or 0)
        + int(row.get("starling_low_sources") or 0)
        + int(row.get("starling_straddling_sources") or 0)
    )
    high_anchor_count = (
        int(row.get("soft_high_sources") or 0)
        + int(row.get("review_high_sources") or 0)
        + int(row.get("starling_high_sources") or 0)
    )
    limiting = int(row.get("strong_limiting_factor_count") or 0) > 0 or bool(row.get("strong_absorption_prior"))
    risk_count = int(row.get("risk_for_lower_f_factor_count") or 0)
    support_count = int(row.get("supports_higher_f_factor_count") or 0)
    if limiting:
        return "low"
    if low_anchor_count > high_anchor_count and bool(row.get("low_property_prior")) and risk_count >= support_count:
        return "low"
    return "high"


def _mechanism_low_blocker_prediction(row: dict[str, Any], current_prediction: str) -> str:
    """Offline candidate: narrow high->low blockers for mechanism-mismatched soft transfer.

    This is diagnostics-only until validated on calibration data. It deliberately
    targets false-high mechanisms surfaced by failure audits without changing
    fallback lows.
    """
    if current_prediction != "high":
        return current_prediction if current_prediction in {"high", "low"} else "high"
    if row.get("decision_state") != "fallback_uncertain":
        return current_prediction
    if row.get("source_level_consensus") != "no_eligible_clean_source_level_direct_f_vote":
        return current_prediction

    low_property = bool(row.get("low_property_prior"))
    oral_low = row.get("oral_bioavailability_prior") == "low"
    soft_direction = str(row.get("net_soft_context_direction") or "")
    chembl_low = int(row.get("chembl_low_sources") or 0)
    starling_low_or_straddling = int(row.get("starling_low_sources") or 0) + int(
        row.get("starling_straddling_sources") or 0
    )
    active_scope = int(row.get("transfer_active_metabolite_or_prodrug_sources") or 0)
    support_count = int(row.get("supports_higher_f_factor_count") or 0)

    if bool(row.get("alert_permanent_quaternary_charge")) and low_property and int(row.get("starling_low_sources") or 0) >= 1:
        return "low"
    if bool(row.get("alert_beta_lactam_anionic")) and low_property and chembl_low:
        return "low"
    if bool(row.get("alert_dihydropyridine_diester")) and oral_low and chembl_low >= 2:
        return "low"
    if (
        bool(row.get("alert_ester_prodrug_or_active_moiety"))
        and active_scope >= 2
        and low_property
        and starling_low_or_straddling
    ):
        return "low"
    if (
        bool(row.get("alert_flat_high_logp_low_tpsa_hbd0"))
        and support_count >= 1
        and starling_low_or_straddling
        and soft_direction == "soft_context_mixed_or_weak"
    ):
        return "low"
    if (
        bool(row.get("alert_high_ionization_low_permeability"))
        and low_property
        and soft_direction == "soft_context_threshold_sensitive_mixed"
        and starling_low_or_straddling
    ):
        return "low"
    return current_prediction


def _starling_soft_high_anchor_prediction(row: dict[str, Any], current_prediction: str) -> str:
    """Offline candidate: restore Starling-backed soft high direct-F anchors.

    This diagnostics/calibration rule handles cases where the live final LLM
    predicted low despite multiple de-correlated above-threshold direct-F source
    contexts. It is intentionally conservative: it requires no eligible clean
    direct vote, at least two Starling-backed high soft sources, limited
    low/straddling counter-context, and only applies to current low predictions.
    """
    if current_prediction != "low":
        return current_prediction if current_prediction in {"high", "low"} else "high"
    if row.get("decision_state") != "fallback_uncertain":
        return current_prediction
    if row.get("source_level_consensus") != "no_eligible_clean_source_level_direct_f_vote":
        return current_prediction
    soft_high = int(row.get("soft_high_sources") or 0)
    soft_low = int(row.get("soft_low_sources") or 0)
    soft_straddling = int(row.get("soft_straddling_sources") or 0)
    starling_high = int(row.get("starling_high_sources") or 0)
    if soft_high >= 2 and soft_low <= 1 and soft_straddling <= 2 and starling_high >= 2:
        return "high"
    return current_prediction


def _no_evidence_high_fallback_prediction(row: dict[str, Any], current_prediction: str) -> str:
    """Offline candidate: do not let a bare single-molecule prior force low.

    This applies only when the completed pipeline predicted low despite having
    no direct-F source context, no soft direct-F context, no risk factor, and no
    structural mechanism alert. It is a low-confidence forced binary fallback,
    not positive evidence of high bioavailability.
    """
    if current_prediction != "low":
        return current_prediction if current_prediction in {"high", "low"} else "high"
    if row.get("decision_state") != "fallback_uncertain":
        return current_prediction
    if row.get("source_level_consensus") != "no_eligible_clean_source_level_direct_f_vote":
        return current_prediction
    if int(row.get("n_source_molecules_with_direct_f_values") or 0):
        return current_prediction
    if int(row.get("n_clean_but_not_direct_vote_sources") or 0):
        return current_prediction
    if int(row.get("n_review_required_context_only_sources") or 0):
        return current_prediction
    if any(
        int(row.get(key) or 0)
        for key in (
            "soft_high_sources",
            "soft_low_sources",
            "soft_straddling_sources",
            "starling_source_count",
            "chembl_source_count",
            "transfer_anchor_candidate_sources",
        )
    ):
        return current_prediction
    if any(
        bool(row.get(key))
        for key in (
            "alert_permanent_quaternary_charge",
            "alert_beta_lactam_anionic",
            "alert_dihydropyridine_diester",
            "alert_ester_prodrug_or_active_moiety",
            "alert_high_ionization_low_permeability",
        )
    ):
        return current_prediction
    if any(
        int(row.get(key) or 0)
        for key in (
            "risk_for_lower_f_factor_count",
            "strong_limiting_factor_count",
            "moderate_analog_risk_factor_count",
        )
    ):
        return current_prediction
    return "high"


def _metric_value(metrics: dict[str, Any], selection_metric: str) -> float:
    if selection_metric in {"macro_f1", "accuracy", "positive_class_f1"}:
        key = "positive_class_f1" if selection_metric == "positive_class_f1" else selection_metric
        return float(metrics.get(key) or 0.0)
    if selection_metric == "negative_class_f1":
        return float((metrics.get("per_class") or {}).get("0", {}).get("f1") or 0.0)
    if selection_metric == "negative_class_recall":
        return float((metrics.get("per_class") or {}).get("0", {}).get("recall") or 0.0)
    raise ValueError(f"Unknown selection metric: {selection_metric}")


def _factor_features(
    factor_signal_counts: dict[str, Any],
    factor_strength: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    total_signals = Counter()
    total_strength = Counter()
    for factor in ("Fa", "Fg", "Fh"):
        signal_counts = factor_signal_counts.get(factor) or {}
        if isinstance(signal_counts, list):
            signal_counts = Counter(signal_counts)
        strengths = factor_strength.get(factor) or []
        strength_counts = Counter(str(item) for item in strengths)
        result[f"{factor}_n_cards"] = sum(int(value or 0) for value in signal_counts.values())
        result[f"{factor}_supports_higher_f_count"] = int(signal_counts.get("supports_higher_F") or 0)
        result[f"{factor}_risk_for_lower_f_count"] = int(signal_counts.get("risk_for_lower_F") or 0)
        result[f"{factor}_mixed_or_context_count"] = int(signal_counts.get("mixed_or_context") or 0)
        result[f"{factor}_strong_limiting_count"] = int(strength_counts.get("strong_limiting") or 0)
        result[f"{factor}_moderate_analog_risk_count"] = int(strength_counts.get("moderate_analog_risk") or 0)
        result[f"{factor}_weak_context_count"] = int(strength_counts.get("weak_context") or 0)
        total_signals.update({str(key): int(value or 0) for key, value in signal_counts.items()})
        total_strength.update({str(key): int(value or 0) for key, value in strength_counts.items()})
    result["supports_higher_f_factor_count"] = int(total_signals.get("supports_higher_F") or 0)
    result["risk_for_lower_f_factor_count"] = int(total_signals.get("risk_for_lower_F") or 0)
    result["mixed_or_context_factor_count"] = int(total_signals.get("mixed_or_context") or 0)
    result["strong_limiting_factor_count"] = int(total_strength.get("strong_limiting") or 0)
    result["moderate_analog_risk_factor_count"] = int(total_strength.get("moderate_analog_risk") or 0)
    result["weak_context_factor_count"] = int(total_strength.get("weak_context") or 0)
    return result


def _factor_signal_counts_from_guard(guard: dict[str, Any]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for factor, signals in (guard.get("factor_signals") or {}).items():
        result[str(factor)] = dict(Counter(str(signal) for signal in signals or []))
    return result


def _source_transfer_features(
    source_transfer_summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Flatten source-transfer scope classes for fallback diagnostics.

    Older completed batches predate the top-level source_transfer_summary, so
    this also reconstructs coarse classes from source_summaries when needed.
    """
    scope_counts = Counter()
    scope_direction_counts: dict[str, Counter[str]] = {}
    anchor_sources = list(source_transfer_summary.get("anchor_candidate_sources") or [])
    if source_transfer_summary.get("scope_class_counts"):
        for scope_class, count in (source_transfer_summary.get("scope_class_counts") or {}).items():
            scope_counts[str(scope_class)] = int(count or 0)
        for scope_class, counts in (source_transfer_summary.get("scope_direction_counts") or {}).items():
            scope_direction_counts[str(scope_class)] = Counter(
                {str(key): int(value or 0) for key, value in (counts or {}).items()}
            )
    else:
        for source in source_rows:
            classes = _source_transfer_scope_classes_from_source(source)
            for scope_class in classes:
                scope_counts[scope_class] += 1
                scope_direction_counts.setdefault(scope_class, Counter())[str(source.get("source_direction") or "missing")] += 1
            if (
                source.get("label_vote_bucket") == "eligible_clean_direct_vote"
                or "same_active_moiety_salt_or_freebase_context" in set(classes)
            ):
                anchor_sources.append(source)

    anchor_counts = Counter(str(source.get("source_direction") or "missing") for source in anchor_sources)
    return {
        "transfer_clean_parent_or_close_analog_sources": int(scope_counts.get("clean_parent_or_close_analog_context") or 0),
        "transfer_same_active_moiety_salt_or_freebase_sources": int(
            scope_counts.get("same_active_moiety_salt_or_freebase_context") or 0
        ),
        "transfer_active_metabolite_or_prodrug_sources": int(scope_counts.get("active_metabolite_or_prodrug_scope") or 0),
        "transfer_special_formulation_or_route_sources": int(scope_counts.get("special_formulation_or_route_context") or 0),
        "transfer_relative_or_proxy_sources": int(scope_counts.get("relative_or_proxy_context") or 0),
        "transfer_non_swallowed_route_sources": int(scope_counts.get("non_swallowed_route_context") or 0),
        "transfer_total_radioactivity_sources": int(scope_counts.get("total_radioactivity_scope") or 0),
        "transfer_anchor_candidate_sources": len(anchor_sources),
        "transfer_anchor_high_sources": int(anchor_counts.get("supports_high_bioavailability") or 0),
        "transfer_anchor_low_sources": int(anchor_counts.get("argues_against_high_bioavailability") or 0),
        "transfer_anchor_straddling_sources": int(anchor_counts.get("mixed_threshold_straddling") or 0),
        "transfer_scope_class_counts": dict(sorted(scope_counts.items())),
        "transfer_scope_direction_counts": {
            scope_class: dict(sorted(counts.items()))
            for scope_class, counts in sorted(scope_direction_counts.items())
        },
    }


def _source_transfer_scope_classes_from_source(source: dict[str, Any]) -> list[str]:
    explicit = [str(item) for item in source.get("source_transfer_scope_classes") or [] if item]
    if explicit:
        return explicit
    reasons = {str(reason) for reason in source.get("parent_scope_reasons") or []}
    classes: list[str] = []
    if "active_metabolite_prodrug_or_total_radioactivity_scope" in reasons:
        classes.append("active_metabolite_or_prodrug_scope")
    if "total_radioactivity_scope" in reasons:
        classes.append("total_radioactivity_scope")
    if "non_swallowed_or_special_route_scope" in reasons:
        classes.append("non_swallowed_route_context")
    if "formulation_or_salt_specific_scope" in reasons:
        classes.append("same_active_moiety_salt_or_freebase_context")
    if "relative_or_proxy_bioavailability_scope" in reasons:
        classes.append("relative_or_proxy_context")
    if source.get("label_vote_bucket") == "review_required_context_only" and not classes:
        classes.append("special_formulation_or_route_context")
    return classes or ["clean_parent_or_close_analog_context"]


def _single_molecule_alert_features(prediction_row: dict[str, Any]) -> dict[str, Any]:
    content = _single_molecule_content(prediction_row)
    text_parts: list[str] = [
        str(content.get("reasoning_summary") or ""),
    ]
    for key in ("property_drivers", "caveats"):
        values = content.get(key)
        if isinstance(values, list):
            text_parts.extend(str(value or "") for value in values)
    text = " ".join(text_parts).lower()
    return {
        "alert_permanent_quaternary_charge": _has_any_text(
            text,
            ("quaternary ammonium", "permanent positive charge", "permanently charged", "permanent cation"),
        ),
        "alert_beta_lactam_anionic": _has_any_text(text, ("β-lactam", "beta-lactam"))
        and _has_any_text(text, ("anion", "anionic", "dianion", "carboxylic acid", "free carboxylic acid")),
        "alert_dihydropyridine_diester": _has_any_text(text, ("1,4-dihydropyridine", "dihydropyridine"))
        and _has_any_text(text, ("two carboxylic ester", "two ester", "diester")),
        "alert_flat_high_logp_low_tpsa_hbd0": _has_any_text(text, ("low tpsa", "tpsa 34", "tpsa (34"))
        and _has_any_text(text, ("hbd 0", "zero hbd"))
        and _has_any_text(text, ("logp/logd", "logd 3", "logp 3", "high logd", "high logp"))
        and _has_any_text(text, ("aromatic rings", "multiple aromatic", "flat", "rigid")),
        "alert_ester_prodrug_or_active_moiety": _has_any_text(text, ("prodrug", "active metabolite", "active moiety"))
        or (
            _has_any_text(text, ("carboxylic ester", "hydroly", "ester"))
            and _has_any_text(text, ("active", "activation", "metabolite"))
        ),
        "alert_high_ionization_low_permeability": _has_any_text(
            text,
            (
                "neutral fraction 0",
                "fully ionized",
                "dianion",
                "net anionic",
                "permanently charged",
                "permanent positive charge",
            ),
        )
        and _has_any_text(text, ("poor passive", "low permeability", "very poor passive", "permeability restriction")),
    }


def _single_molecule_content(prediction_row: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(prediction_row.get("run_dir") or ""))
    if not run_dir:
        return {}
    path = run_dir / "single_molecule_reasoning_output.json"
    if not path.exists():
        return {}
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    content = ((payload.get("llm") or {}).get("content") or {})
    return content if isinstance(content, dict) else {}


def _has_any_text(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _direction_counts_for_sources(source_rows: list[dict[str, Any]], predicate: Any) -> dict[str, int]:
    selected = [row for row in source_rows if predicate(row)]
    counts = Counter(str(row.get("source_direction") or "missing") for row in selected)
    return {
        "n_sources": len(selected),
        "supports_high_bioavailability": int(counts.get("supports_high_bioavailability") or 0),
        "argues_against_high_bioavailability": int(counts.get("argues_against_high_bioavailability") or 0),
        "mixed_threshold_straddling": int(counts.get("mixed_threshold_straddling") or 0),
    }


def _count_direction(counts: dict[str, Any], direction: str) -> int:
    return int((counts or {}).get(direction) or 0)


def _context_flag_names(context_flags: Any) -> list[str]:
    if isinstance(context_flags, dict):
        return sorted(str(key) for key in context_flags)
    if isinstance(context_flags, list):
        return sorted(str(item) for item in context_flags)
    return []


def _has_low_property_prior(row: dict[str, Any]) -> bool:
    return row.get("oral_bioavailability_prior") == "low" and (
        row.get("absorption_prior") == "unfavorable"
        or row.get("solubility_or_dissolution_prior") == "unfavorable"
        or row.get("metabolism_or_clearance_prior") == "unfavorable"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = _tsv_fields(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _tsv_value(row.get(field)) for field in fields})


def _tsv_fields(rows: list[dict[str, Any]]) -> list[str]:
    keys = set()
    for row in rows:
        keys.update(row)
    prioritized = [field for field in PRIOR_TSV_FIELDS if field in keys]
    return prioritized + sorted(keys.difference(prioritized))


def _tsv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
