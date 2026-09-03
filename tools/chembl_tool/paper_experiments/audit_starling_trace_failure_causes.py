"""Diagnose Starling agent failures from frozen full-mechanism traces.

The audit is deterministic and makes no model calls.  It aligns benchmark label
provenance, direct-outcome retrieval, group interpretation, the single-molecule
prior, and the final prediction at sample level.  The direct-outcome vote is an
audit proxy over the visible neighbor evidence, not a replacement label policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

from tools.chembl_tool.common.json_utils import (
    atomic_output_path,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.paper_experiments.paired_binary_predictions import (
    paired_binary_summary,
)


DEFAULT_RUN_ROOT = Path(
    "outputs/paper/"
    "molecular_evidence_agent_starling_scaffold_record_supported_v2_valid_gpt_oss_120b/"
    "runs_identity_blind_parent_disjoint"
)
DEFAULT_DATA_ROOT = Path("data/gold_labels/legacy/processed_starling_record_supported_v2")
DEFAULT_OUTPUT_DIR = Path(
    "outputs/paper/"
    "starling_trace_failure_cause_audit_record_supported_v2_valid_gpt_oss_120b"
)
STRICT_LEVELS = frozenset({"high", "moderate"})
@dataclass(frozen=True)
class TaskSpec:
    dataset_task: str
    run_task: str
    condition: str
    direct_condition: str
    direct_group_id: str
    useful_fields: tuple[str, ...]
    single_prior_fields: tuple[str, ...]
    direction_fields: tuple[str, ...]
    positive_directions: frozenset[str]
    negative_directions: frozenset[str]
    allowed_directions: frozenset[str]
    neighbor_signal: Callable[[dict[str, Any]], int | None]


def _bbb_neighbor_signal(row: dict[str, Any]) -> int | None:
    direction = row.get("evidence_direction")
    if direction == "supports_bbb_crossing":
        return 1
    if direction == "argues_against_bbb_crossing":
        return 0
    return None


def _bioavailability_neighbor_signal(row: dict[str, Any]) -> int | None:
    try:
        value = float(row.get("source_value_median"))
    except (TypeError, ValueError):
        return None
    return int(value >= 20.0)


def _skin_neighbor_signal(row: dict[str, Any]) -> int | None:
    counts = row.get("source_endpoint_counts") or {}
    positive = int(counts.get("positive") or 0)
    negative = int(counts.get("negative") or 0)
    if positive == negative:
        return None
    return int(positive > negative)


TASK_SPECS = (
    TaskSpec(
        dataset_task="BBB_Martins",
        run_task="bbb_martins",
        condition="starling_full_mechanism",
        direct_condition="starling_direct",
        direct_group_id="Mechanism.tier_1",
        useful_fields=("useful_for_bbb_reasoning",),
        single_prior_fields=("passive_bbb_plausibility",),
        direction_fields=("evidence_direction",),
        positive_directions=frozenset({"supports_bbb_crossing", "influx_support"}),
        negative_directions=frozenset({"argues_against_bbb_crossing", "efflux_risk"}),
        allowed_directions=frozenset(
            {
                "supports_bbb_crossing",
                "argues_against_bbb_crossing",
                "efflux_risk",
                "influx_support",
                "neutral_or_unclear",
            }
        ),
        neighbor_signal=_bbb_neighbor_signal,
    ),
    TaskSpec(
        dataset_task="Bioavailability_Ma",
        run_task="bioavailability_ma",
        condition="starling_full_mechanism",
        direct_condition="starling_direct_full",
        direct_group_id="Observed.direct_oral_bioavailability",
        useful_fields=("useful_for_bioavailability_reasoning",),
        single_prior_fields=("oral_bioavailability_prior",),
        direction_fields=("evidence_direction",),
        positive_directions=frozenset(
            {
                "supports_high_bioavailability",
                "absorption_support",
                "permeability_support",
                "solubility_support",
                "metabolic_stability_support",
            }
        ),
        negative_directions=frozenset(
            {
                "argues_against_high_bioavailability",
                "solubility_risk",
                "first_pass_or_clearance_risk",
                "transporter_efflux_risk",
            }
        ),
        allowed_directions=frozenset(
            {
                "supports_high_bioavailability",
                "argues_against_high_bioavailability",
                "absorption_support",
                "permeability_support",
                "solubility_support",
                "solubility_risk",
                "metabolic_stability_support",
                "first_pass_or_clearance_risk",
                "transporter_efflux_risk",
                "neutral_or_unclear",
            }
        ),
        neighbor_signal=_bioavailability_neighbor_signal,
    ),
    TaskSpec(
        dataset_task="Skin_Reaction",
        run_task="skin_reaction",
        condition="starling_full_mechanism",
        direct_condition="starling_direct",
        direct_group_id="Mechanism.tier_1",
        useful_fields=(
            "useful_for_skin_reaction_reasoning",
            "useful_for_skin_sensitization_reasoning",
        ),
        single_prior_fields=("skin_reaction_prior", "skin_sensitization_prior"),
        direction_fields=("evidence_direction", "sensitization_evidence_direction"),
        positive_directions=frozenset(
            {
                "supports_skin_reaction_risk",
                "sensitization_risk",
                "supports_sensitizer",
            }
        ),
        negative_directions=frozenset(
            {"argues_against_skin_reaction_risk", "argues_against_sensitizer"}
        ),
        allowed_directions=frozenset(
            {
                "supports_skin_reaction_risk",
                "sensitization_risk",
                "argues_against_skin_reaction_risk",
                "neutral_or_unclear",
                "context_dependent",
                "skin_exposure_support",
                "local_skin_damage_risk",
                "irritation_or_corrosion_risk",
                "reduced_skin_exposure",
                "phototoxicity_risk",
                "supports_sensitizer",
                "argues_against_sensitizer",
                "context_only",
            }
        ),
        neighbor_signal=_skin_neighbor_signal,
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--tasks",
        nargs="*",
        choices=tuple(spec.run_task for spec in TASK_SPECS),
        default=[],
        help="Optional task subset; default audits all three tasks.",
    )
    args = parser.parse_args(argv)

    summaries: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    selected = set(args.tasks)
    for spec in TASK_SPECS:
        if selected and spec.run_task not in selected:
            continue
        summary, rows = audit_task(Path(args.run_root), Path(args.data_root), spec)
        summaries.append(summary)
        sample_rows.extend(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = {
        "schema_version": "starling_trace_failure_cause_audit.v3",
        "analysis_type": "deterministic_existing_trace_audit_no_model_calls",
        "task_summaries": summaries,
        "limitations": [
            "The direct-outcome vote is a descriptive proxy derived from visible neighbor evidence.",
            "A proxy vote agreeing with gold does not prove chemical transferability to the query.",
            "The audit attributes the observed decision path; it does not estimate a future prompt's metric.",
        ],
    }
    write_json_atomic(output_dir / "summary.json", combined)
    write_jsonl_atomic(output_dir / "sample_audit.jsonl", sample_rows)
    with atomic_output_path(output_dir / "report.md") as temporary:
        temporary.write_text(render_report(combined), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": combined}, indent=2))
    return 0


def audit_task(
    run_root: Path,
    data_root: Path,
    spec: TaskSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    batch = run_root / spec.run_task / f"{spec.run_task}__{spec.condition}"
    predictions = _read_jsonl(batch / "predictions.jsonl")
    labels = _read_jsonl(
        data_root / spec.dataset_task / "scaffold" / "valid_molecule_labels.jsonl"
    )
    aligned = _align_predictions_with_labels(predictions, labels, spec.dataset_task)
    rows = [
        _audit_sample(spec, prediction, label_row)
        for prediction, label_row in aligned
    ]
    errors = [row for row in rows if not row["correct"]]
    correct = [row for row in rows if row["correct"]]
    pairings = _paired_condition_summaries(run_root, spec)

    flat_mechanism_pair = "starling_full_flat_vs_starling_full_mechanism"
    if flat_mechanism_pair not in pairings:
        raise ValueError(
            f"Missing required paired conditions for {spec.dataset_task}: "
            f"{flat_mechanism_pair}"
        )
    summary = {
        "task": spec.dataset_task,
        "n": len(rows),
        "n_errors": len(errors),
        "confusion_matrix": _confusion_matrix(rows),
        "gold_provenance": {
            "unanimous": _group_error_rate(rows, lambda row: row["agreement_fraction"] == 1.0),
            "nonunanimous": _group_error_rate(rows, lambda row: row["agreement_fraction"] < 1.0),
            "singleton": _group_error_rate(rows, lambda row: row["source_record_count"] == 1),
            "multi_record": _group_error_rate(rows, lambda row: row["source_record_count"] > 1),
        },
        "direct_retrieval": {
            "card_consistency": _direct_card_consistency(predictions, spec),
            "similarity_by_outcome": {
                "correct": _similarity_summary(correct),
                "error": _similarity_summary(errors),
            },
            "error_rate_by_max_similarity": {
                "absent": _group_error_rate(rows, lambda row: row["direct_max_similarity"] is None),
                "below_0.4": _group_error_rate(
                    rows,
                    lambda row: row["direct_max_similarity"] is not None
                    and row["direct_max_similarity"] < 0.4,
                ),
                "0.4_to_0.6": _group_error_rate(
                    rows,
                    lambda row: row["direct_max_similarity"] is not None
                    and 0.4 <= row["direct_max_similarity"] < 0.6,
                ),
                "at_least_0.6": _group_error_rate(
                    rows,
                    lambda row: row["direct_max_similarity"] is not None
                    and row["direct_max_similarity"] >= 0.6,
                ),
            },
            "visible_proxy_state": _counts(rows, "direct_proxy_state"),
            "visible_proxy_state_in_errors": _counts(errors, "direct_proxy_state"),
            "proxy_vs_direct_group_when_proxy_known": _counts(
                [row for row in rows if row["direct_proxy_label"] is not None],
                "direct_group_vs_proxy",
            ),
        },
        "exclusive_error_path": _counts(errors, "exclusive_error_path"),
        "single_prior_in_errors": _counts(errors, "single_prior"),
        "invalid_group_direction": {
            "n_outputs": sum(len(row["invalid_group_directions"]) for row in rows),
            "n_error_samples": sum(bool(row["invalid_group_directions"]) for row in errors),
            "values": dict(
                Counter(
                    direction
                    for row in rows
                    for direction in row["invalid_group_directions"]
                )
            ),
        },
        "structured_output_validation": _structured_validation_summary(rows),
        "paired_conditions": pairings,
        "full_flat_vs_full_mechanism": pairings[flat_mechanism_pair],
    }
    return summary, rows


def _align_predictions_with_labels(
    predictions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    task_name: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Align by frozen query index and verify identity and label provenance."""
    if len(predictions) != len(labels):
        raise ValueError(f"Prediction/label count mismatch for {task_name}")
    by_index: dict[int, dict[str, Any]] = {}
    for prediction in predictions:
        query_index = int(prediction["query_index"])
        if query_index in by_index:
            raise ValueError(f"Duplicate prediction query_index {query_index} for {task_name}")
        by_index[query_index] = prediction
    if set(by_index) != set(range(len(labels))):
        raise ValueError(f"Prediction query_index coverage mismatch for {task_name}")

    aligned: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for query_index, label_row in enumerate(labels):
        prediction = by_index[query_index]
        if str(prediction.get("smiles") or "") != str(label_row.get("drug") or ""):
            raise ValueError(
                f"Prediction/label molecule mismatch for {task_name} index {query_index}"
            )
        if int(prediction["label"]) != int(label_row["Y"]):
            raise ValueError(
                f"Prediction/label gold mismatch for {task_name} index {query_index}"
            )
        aligned.append((prediction, label_row))
    return aligned


def _direct_card_consistency(
    predictions: list[dict[str, Any]],
    spec: TaskSpec,
) -> dict[str, Any]:
    cards: dict[tuple[str, str], dict[str, Any]] = {}
    for prediction in predictions:
        retrieval = json.loads(
            (Path(prediction["run_dir"]) / "retrieval.json").read_text(encoding="utf-8")
        )
        direct = next(
            group
            for group in retrieval["groups"]
            if group["group_id"] == spec.direct_group_id
        )
        for neighbor in direct.get("neighbors") or []:
            for evidence in neighbor.get("evidence_rows") or []:
                key = (
                    str(evidence.get("molecule_chembl_id") or ""),
                    str(evidence.get("assay_chembl_id") or ""),
                )
                cards[key] = evidence

    if spec.dataset_task == "BBB_Martins":
        supporting = [
            row
            for row in cards.values()
            if row.get("evidence_direction") == "supports_bbb_crossing"
        ]
        contradictory = [
            row
            for row in supporting
            if set(row.get("source_transport_labels") or [])
            & {"efflux_substrate", "efflux_limited"}
            or set(row.get("source_permeability_labels") or [])
            & {"impermeable", "poor_penetration", "low_permeability"}
        ]
        return {
            "definition": "support-tagged cards also containing efflux or negative permeability records",
            "n_unique_cards": len(cards),
            "n_eligible_cards": len(supporting),
            "n_mixed_cards": len(contradictory),
        }
    if spec.dataset_task == "Bioavailability_Ma":
        numeric = [
            row
            for row in cards.values()
            if row.get("source_value_min") not in (None, "")
            and row.get("source_value_max") not in (None, "")
        ]
        crossing = [
            row
            for row in numeric
            if float(row["source_value_min"]) < 20 <= float(row["source_value_max"])
        ]
        return {
            "definition": "aggregated numeric cards whose observed range crosses the 20 percent label threshold",
            "n_unique_cards": len(cards),
            "n_eligible_cards": len(numeric),
            "n_mixed_cards": len(crossing),
        }
    mixed = [
        row
        for row in cards.values()
        if int((row.get("source_endpoint_counts") or {}).get("positive") or 0) > 0
        and int((row.get("source_endpoint_counts") or {}).get("negative") or 0) > 0
    ]
    return {
        "definition": "aggregated direct cards containing both positive and negative records",
        "n_unique_cards": len(cards),
        "n_eligible_cards": len(cards),
        "n_mixed_cards": len(mixed),
    }


def _audit_sample(
    spec: TaskSpec,
    prediction: dict[str, Any],
    label_row: dict[str, Any],
) -> dict[str, Any]:
    run_dir = Path(prediction["run_dir"])
    retrieval = json.loads((run_dir / "retrieval.json").read_text(encoding="utf-8"))
    group_outputs = _read_jsonl(run_dir / "group_reasoning_outputs.jsonl")
    group_map = {row["group_id"]: row for row in group_outputs}
    single_output = json.loads(
        (run_dir / "single_molecule_reasoning_output.json").read_text(encoding="utf-8")
    )
    single = ((single_output.get("llm") or {}).get("content") or {})
    final = _llm_content(run_dir / "final_reasoning_output.json")
    direct_retrieval = next(
        group for group in retrieval["groups"] if group["group_id"] == spec.direct_group_id
    )
    direct_output = group_map.get(spec.direct_group_id) or {}
    direct_content = ((direct_output.get("llm") or {}).get("content") or {})

    neighbor_labels = [
        _majority_vote(
            [
                spec.neighbor_signal(evidence)
                for evidence in neighbor.get("evidence_rows") or []
            ]
        )
        for neighbor in direct_retrieval.get("neighbors") or []
    ]
    direct_proxy_label = _majority_vote(neighbor_labels)
    gold = int(prediction["label"])
    direct_proxy_state = (
        "inconclusive"
        if direct_proxy_label is None
        else "gold_aligned"
        if direct_proxy_label == gold
        else "opposite_gold"
    )
    direct_group_signals = _strict_signals(spec, direct_content)
    all_signals: set[int] = set()
    for output in group_outputs:
        all_signals.update(
            _strict_signals(spec, ((output.get("llm") or {}).get("content") or {}))
        )

    if direct_proxy_label is None:
        error_path = "direct_retrieval_inconclusive"
    elif direct_proxy_label != gold:
        error_path = "direct_retrieval_wrong_direction"
    elif gold not in direct_group_signals:
        error_path = "direct_group_lost_gold_signal"
    elif 1 - gold in all_signals:
        error_path = "mechanism_conflict_overrule"
    else:
        error_path = "final_aggregation_error"

    similarities = [
        float(neighbor["similarity"])
        for neighbor in direct_retrieval.get("neighbors") or []
    ]
    invalid = [
        str(_first_value(content, spec.direction_fields) or "")
        for output in group_outputs
        for content in [((output.get("llm") or {}).get("content") or {})]
        if str(_first_value(content, spec.direction_fields) or "")
        not in spec.allowed_directions
    ]
    direct_direction_value = _first_value(direct_content, spec.direction_fields)
    direct_direction = _direction_labels(spec, direct_direction_value)
    if direct_proxy_label is None:
        group_vs_proxy = "proxy_inconclusive"
    elif direct_proxy_label in direct_direction:
        group_vs_proxy = "agree"
    elif not direct_direction:
        group_vs_proxy = "neutral_or_noncanonical"
    else:
        group_vs_proxy = "flip"

    return {
        "task": spec.dataset_task,
        "query_index": int(prediction["query_index"]),
        "label": gold,
        "pred_label": int(prediction["pred_label"]),
        "correct": bool(prediction["correct"]),
        "source_record_count": int(label_row["source_record_count"]),
        "agreement_fraction": float(label_row["agreement_fraction"]),
        "label_decision": label_row["label_decision"],
        "direct_neighbor_count": len(similarities),
        "direct_max_similarity": max(similarities) if similarities else None,
        "direct_mean_similarity": mean(similarities) if similarities else None,
        "direct_neighbor_proxy_labels": neighbor_labels,
        "direct_proxy_label": direct_proxy_label,
        "direct_proxy_state": direct_proxy_state,
        "direct_group_direction": direct_direction_value,
        "direct_group_transferability": direct_content.get("transferability"),
        "direct_group_confidence": direct_content.get("confidence"),
        "direct_group_strict_signals": sorted(direct_group_signals),
        "direct_group_vs_proxy": group_vs_proxy,
        "all_group_strict_signals": sorted(all_signals),
        "single_prior": _first_value(single, spec.single_prior_fields),
        "final_confidence": final.get("confidence"),
        "invalid_group_directions": invalid,
        "single_validation": _validation_receipt(single_output),
        "group_validations": [_validation_receipt(output) for output in group_outputs],
        "exclusive_error_path": "correct" if prediction["correct"] else error_path,
        "run_dir": str(run_dir),
    }


def _strict_signals(spec: TaskSpec, content: dict[str, Any]) -> set[int]:
    if _first_value(content, spec.useful_fields) is not True:
        return set()
    if content.get("transferability") not in STRICT_LEVELS:
        return set()
    if content.get("confidence") not in STRICT_LEVELS:
        return set()
    return _direction_labels(spec, _first_value(content, spec.direction_fields))


def _first_value(content: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field in fields:
        if field in content:
            return content[field]
    return None


def _direction_labels(spec: TaskSpec, direction: Any) -> set[int]:
    labels: set[int] = set()
    for part in str(direction or "").split("|"):
        normalized = part.strip()
        for typo in ("arguments_against_", "argsues_against_", "args_against_"):
            normalized = normalized.replace(typo, "argues_against_")
        if normalized in spec.positive_directions:
            labels.add(1)
        elif normalized in spec.negative_directions:
            labels.add(0)
    return labels


def _majority_vote(values: list[int | None]) -> int | None:
    counts = Counter(value for value in values if value is not None)
    if not counts or counts[0] == counts[1]:
        return None
    return int(counts.most_common(1)[0][0])


def _paired_condition_summaries(
    run_root: Path,
    spec: TaskSpec,
) -> dict[str, dict[str, Any]]:
    condition_pairs = (
        ("none", spec.direct_condition),
        (spec.direct_condition, "starling_full_flat"),
        ("starling_full_flat", "starling_full_mechanism"),
        ("none", "starling_full_mechanism"),
    )
    conditions: dict[str, dict[int, dict[str, Any]]] = {}
    for condition in {condition for pair in condition_pairs for condition in pair}:
        path = (
            run_root
            / spec.run_task
            / f"{spec.run_task}__{condition}"
            / "predictions.jsonl"
        )
        if path.exists():
            conditions[condition] = _prediction_map(path)
    return {
        f"{left_name}_vs_{right_name}": _paired_prediction_summary(
            conditions[left_name], conditions[right_name]
        )
        for left_name, right_name in condition_pairs
        if left_name in conditions and right_name in conditions
    }


def _prediction_map(path: Path) -> dict[int, dict[str, Any]]:
    rows = _read_jsonl(path)
    mapped = {int(row["query_index"]): row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError(f"Duplicate query indices in {path}")
    return mapped


def _paired_prediction_summary(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
    *,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    if left.keys() != right.keys():
        raise ValueError("Paired condition query-index sets do not match")
    indices = sorted(left)
    labels = [int(left[index]["label"]) for index in indices]
    if labels != [int(right[index]["label"]) for index in indices]:
        raise ValueError("Paired condition gold labels do not match")
    left_predictions = [int(left[index]["pred_label"]) for index in indices]
    right_predictions = [int(right[index]["pred_label"]) for index in indices]
    return paired_binary_summary(
        labels,
        left_predictions,
        right_predictions,
        bootstrap_replicates=bootstrap_replicates,
        seed=20260809,
    )


def _similarity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["direct_max_similarity"])
        for row in rows
        if row["direct_max_similarity"] is not None
    ]
    return {
        "n_with_direct_neighbor": len(values),
        "n_without_direct_neighbor": len(rows) - len(values),
        "mean": round(mean(values), 6) if values else None,
        "median": round(median(values), 6) if values else None,
    }


def _group_error_rate(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    selected = [row for row in rows if predicate(row)]
    errors = sum(not row["correct"] for row in selected)
    return {
        "n": len(selected),
        "errors": errors,
        "error_rate": round(errors / len(selected), 6) if selected else None,
    }


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field)) for row in rows).items()))


def _validation_receipt(output: dict[str, Any]) -> dict[str, Any]:
    validation = ((output.get("llm") or {}).get("structured_output_validation") or {})
    return {
        "attempt_count": int(validation.get("attempt_count") or 0),
        "retried": bool(validation.get("retried")),
        "valid": validation.get("valid"),
    }


def _structured_validation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    single = [row["single_validation"] for row in rows]
    groups = [receipt for row in rows for receipt in row["group_validations"]]

    def summarize(receipts: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n_outputs": len(receipts),
            "n_retried": sum(receipt["retried"] for receipt in receipts),
            "max_attempt_count": max(
                (int(receipt["attempt_count"]) for receipt in receipts),
                default=0,
            ),
            "n_invalid": sum(receipt["valid"] is False for receipt in receipts),
        }

    return {"single": summarize(single), "group": summarize(groups)}


def _confusion_matrix(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "tn": sum(row["label"] == 0 and row["pred_label"] == 0 for row in rows),
        "fp": sum(row["label"] == 0 and row["pred_label"] == 1 for row in rows),
        "fn": sum(row["label"] == 1 and row["pred_label"] == 0 for row in rows),
        "tp": sum(row["label"] == 1 and row["pred_label"] == 1 for row in rows),
    }


def render_report(combined: dict[str, Any]) -> str:
    lines = [
        "# Starling full-mechanism trace failure-cause audit",
        "",
        "This deterministic audit aligns frozen scaffold-valid traces with gold provenance. "
        "It makes no model calls. The exclusive path is a diagnostic decomposition, not a "
        "claim that a neighbor vote is chemically causal.",
        "",
        "## Exclusive error path",
        "",
        "| Task | retrieval wrong | retrieval inconclusive | group lost gold proxy | mechanism conflict | final aggregation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in combined["task_summaries"]:
        counts = task["exclusive_error_path"]
        lines.append(
            f"| {task['task']} | {counts.get('direct_retrieval_wrong_direction', 0)} | "
            f"{counts.get('direct_retrieval_inconclusive', 0)} | "
            f"{counts.get('direct_group_lost_gold_signal', 0)} | "
            f"{counts.get('mechanism_conflict_overrule', 0)} | "
            f"{counts.get('final_aggregation_error', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Paired condition changes",
            "",
            "| Task | Pair | Delta macro-F1 (95% CI) | Flips | left-only / right-only correct | McNemar p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for task in combined["task_summaries"]:
        for pair, row in task["paired_conditions"].items():
            ci_low, ci_high = row["delta_macro_f1_bootstrap_95ci"]
            lines.append(
                f"| {task['task']} | {pair} | {row['delta_macro_f1']:+.4f} "
                f"[{ci_low:+.4f},{ci_high:+.4f}] | {row['prediction_flips']} | "
                f"{row['left_only_correct']} / {row['right_only_correct']} | "
                f"{row['mcnemar_exact_p']:.4g} |"
            )
    lines.extend(
        [
            "",
            "## Per-task checks",
            "",
        ]
    )
    for task in combined["task_summaries"]:
        similarity = task["direct_retrieval"]["similarity_by_outcome"]
        pairing = task["full_flat_vs_full_mechanism"]
        invalid = task["invalid_group_direction"]
        validation = task["structured_output_validation"]
        lines.extend(
            [
                f"### {task['task']}",
                "",
                f"- Errors: {task['n_errors']}/{task['n']}; confusion matrix: {task['confusion_matrix']}.",
                f"- Direct max-similarity median: correct={similarity['correct']['median']}, "
                f"error={similarity['error']['median']}.",
                f"- Full-flat to full-mechanism: {pairing['prediction_flips']} flips, "
                f"{pairing['right_only_correct']} gains, "
                f"{pairing['left_only_correct']} harms.",
                f"- Noncanonical group directions: {invalid['n_outputs']} outputs across "
                f"{invalid['n_error_samples']} error samples.",
                f"- Structured-output retries: single={validation['single']['n_retried']}, "
                f"group={validation['group']['n_retried']}; invalid completed outputs: "
                f"single={validation['single']['n_invalid']}, "
                f"group={validation['group']['n_invalid']}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation limits",
            "",
            *[f"- {item}" for item in combined["limitations"]],
            "",
            "See `sample_audit.jsonl` for every sample-level path and source run directory.",
            "",
        ]
    )
    return "\n".join(lines)


def _llm_content(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ((payload.get("llm") or {}).get("content") or {})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
