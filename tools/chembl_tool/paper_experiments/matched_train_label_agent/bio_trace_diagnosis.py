"""Pure trace extraction and summaries for the Bio unanimous-positive audit."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

from .contract import DEFAULT_OUTPUT_ROOT, TASK_SPECS, agent_batch


SCHEMA_VERSION = "bio_unanimous_positive_trace_diagnosis.v1"
DEFAULT_GOLD_AUDIT = Path(
    "data/gold_labels/legacy/processed_starling_record_supported_v2/Bioavailability_Ma/scaffold/"
    "valid_molecule_labels.jsonl"
)


def outcome_cohort(gold: int, agent_prediction: int) -> str:
    """Name the outcome inside the all-Y=1-neighbor stratum."""
    return {
        (1, 0): "harm_flip_to_low",
        (0, 0): "rescue_flip_to_low",
        (1, 1): "correct_keep_high",
        (0, 1): "wrong_keep_high",
    }[(gold, agent_prediction)]


def similarity_bin(top1_similarity: float) -> str:
    if top1_similarity < 0.3:
        return "<0.30"
    if top1_similarity < 0.4:
        return "0.30-0.39"
    return ">=0.40"


def build_case_rows(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    gold_audit_path: str | Path = DEFAULT_GOLD_AUDIT,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    root = Path(output_root)
    spec = TASK_SPECS["bioavailability_ma"]
    batch = agent_batch(root, spec)
    paired = _index(_read_jsonl(root / "analysis" / spec.task / "paired_predictions.jsonl"))
    predictions = _index(_read_jsonl(batch / "predictions.jsonl"))
    if spec.full_pool_direct_batch is None:
        raise ValueError("Bioavailability full-pool direct batch is not configured")
    full_pool = _index(_read_jsonl(spec.full_pool_direct_batch / "predictions.jsonl"))
    gold_rows = _read_jsonl(Path(gold_audit_path))
    gold_by_smiles = {str(row["drug"]): row for row in gold_rows}
    if len(gold_by_smiles) != len(gold_rows):
        raise ValueError("Duplicate molecule in Bioavailability valid gold audit")
    if set(paired) != set(predictions) or set(paired) != set(full_pool):
        raise ValueError("Prediction coverage mismatch")

    rows = [
        _case_row(
            query_index,
            paired[query_index],
            predictions[query_index],
            full_pool[query_index],
            gold_by_smiles,
            batch,
        )
        for query_index in sorted(paired)
    ]
    return rows, {
        "paired_predictions": str(root / "analysis" / spec.task / "paired_predictions.jsonl"),
        "matched_agent_batch": str(batch),
        "full_pool_agent_batch": str(spec.full_pool_direct_batch),
        "gold_audit": str(gold_audit_path),
    }


def summarize_case_rows(
    all_rows: list[dict[str, Any]], source_paths: dict[str, str]
) -> dict[str, Any]:
    cohort_rows = [row for row in all_rows if row["all_three_neighbors_positive"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "Bioavailability_Ma",
        "evaluation_subset": "scaffold-valid",
        "scope": "three Morgan train neighbors all have Y=1",
        "target_contract": "Y=1 iff experimental oral bioavailability F >= 20%",
        "n_valid": len(all_rows),
        "prediction_distribution": _prediction_distribution(all_rows),
        "all_positive_neighbor_stratum": _cohort_summary(cohort_rows),
        "similarity_diagnostic": _similarity_diagnostic(cohort_rows),
        "single_prior_calibration_all_valid": _single_prior_calibration(all_rows),
        "trace_states_by_outcome": _trace_states_by_outcome(cohort_rows),
        "state_combinations": _state_combinations(cohort_rows),
        "harm_gold_reliability": _harm_gold_reliability(cohort_rows),
        "final_reason_keyword_prevalence_in_harms": _keyword_prevalence(cohort_rows),
        "representative_cases": _representative_cases(cohort_rows),
        "diagnosis": _diagnosis(),
        "source_paths": source_paths,
    }


def _case_row(
    query_index: int,
    pair: dict[str, Any],
    prediction: dict[str, Any],
    full_pool: dict[str, Any],
    gold_by_smiles: dict[str, dict[str, Any]],
    batch: Path,
) -> dict[str, Any]:
    gold = gold_by_smiles.get(str(prediction["smiles"]))
    if gold is None or int(pair["Y"]) != int(gold["Y"]):
        raise ValueError(f"Missing or mismatched gold audit for query {query_index}")
    run_dir = batch / "runs" / str(prediction["run_id"])
    single = _llm_content(_read_json(run_dir / "single_molecule_reasoning_output.json"))
    group_rows = _read_jsonl(run_dir / "group_reasoning_outputs.jsonl")
    if len(group_rows) != 1:
        raise ValueError(f"Expected one group output for query {query_index}")
    group = _llm_content(group_rows[0])
    final = _llm_content(_read_json(run_dir / "final_reasoning_output.json"))
    neighbors = list(pair["neighbors"])
    labels = [int(row["Y"]) for row in neighbors]
    top1 = max(float(row["similarity"]) for row in neighbors)
    all_positive = labels == [1, 1, 1]
    return {
        "query_index": query_index,
        "gold": int(pair["Y"]),
        "knn_prediction": int(pair["knn_prediction"]),
        "agent_prediction": int(pair["agent_prediction"]),
        "full_pool_prediction": int(full_pool["pred_label"]),
        "neighbor_labels": labels,
        "neighbor_similarities": [float(item["similarity"]) for item in neighbors],
        "top1_similarity": top1,
        "similarity_bin": similarity_bin(top1),
        "all_three_neighbors_positive": all_positive,
        "cohort": (
            outcome_cohort(int(pair["Y"]), int(pair["agent_prediction"]))
            if all_positive
            else "outside_all_positive_neighbor_stratum"
        ),
        "single_prior": single.get("oral_bioavailability_prior"),
        "absorption_prior": single.get("absorption_prior"),
        "single_confidence": single.get("confidence"),
        "group_direction": group.get("evidence_direction"),
        "group_transferability": group.get("transferability"),
        "group_confidence": group.get("confidence"),
        "neighbor_transferability": [
            item.get("transferability") for item in group.get("key_evidence") or []
        ],
        "final_confidence": final.get("confidence"),
        "single_summary": single.get("reasoning_summary"),
        "group_summary": group.get("reasoning_summary"),
        "final_summary": final.get("final_summary"),
        "gold_agreement": float(gold["agreement_fraction"]),
        "gold_label_decision": gold["label_decision"],
        "gold_record_count": int(gold["source_record_count"]),
        "gold_raw_value_examples": list(gold.get("raw_value_examples") or []),
        "gold_context_examples": list(gold.get("context_examples") or []),
    }


def _prediction_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "gold": _binary_counts(row["gold"] for row in rows),
        "knn": _binary_counts(row["knn_prediction"] for row in rows),
        "matched_agent": _binary_counts(row["agent_prediction"] for row in rows),
        "full_pool_agent": _binary_counts(row["full_pool_prediction"] for row in rows),
    }


def _cohort_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    low_rows = [row for row in rows if row["agent_prediction"] == 0]
    return {
        "n": len(rows),
        "gold": _binary_counts(row["gold"] for row in rows),
        "agent": _binary_counts(row["agent_prediction"] for row in rows),
        "outcomes": dict(sorted(Counter(row["cohort"] for row in rows).items())),
        "agent_low_precision_for_gold_low": _ratio(
            sum(row["gold"] == 0 for row in low_rows), len(low_rows)
        ),
    }


def _similarity_diagnostic(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for name in ("<0.30", "0.30-0.39", ">=0.40"):
        selected = [row for row in rows if row["similarity_bin"] == name]
        predicted_low = [row for row in selected if row["agent_prediction"] == 0]
        result.append({
            "similarity_bin": name,
            "n": len(selected),
            "gold_positive": sum(row["gold"] == 1 for row in selected),
            "gold_positive_rate": _ratio(sum(row["gold"] == 1 for row in selected), len(selected)),
            "agent_predicted_low": len(predicted_low),
            "agent_predicted_low_rate": _ratio(len(predicted_low), len(selected)),
            "agent_low_precision_for_gold_low": _ratio(
                sum(row["gold"] == 0 for row in predicted_low), len(predicted_low)
            ),
        })
    return result


def _single_prior_calibration(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for prior in ("low", "mixed_or_unclear", "high"):
        selected = [row for row in rows if row["single_prior"] == prior]
        result.append({
            "single_prior": prior,
            "n": len(selected),
            "gold_positive": sum(row["gold"] == 1 for row in selected),
            "gold_positive_rate": _ratio(sum(row["gold"] == 1 for row in selected), len(selected)),
            "final_high": sum(row["agent_prediction"] == 1 for row in selected),
            "final_high_rate": _ratio(
                sum(row["agent_prediction"] == 1 for row in selected), len(selected)
            ),
        })
    return result


def _trace_states_by_outcome(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for cohort in (
        "harm_flip_to_low", "rescue_flip_to_low", "correct_keep_high", "wrong_keep_high"
    ):
        selected = [row for row in rows if row["cohort"] == cohort]
        result[cohort] = {
            "n": len(selected),
            "single_prior": _counts(row["single_prior"] for row in selected),
            "absorption_prior": _counts(row["absorption_prior"] for row in selected),
            "single_confidence": _counts(row["single_confidence"] for row in selected),
            "group_direction": _counts(row["group_direction"] for row in selected),
            "group_transferability": _counts(row["group_transferability"] for row in selected),
            "group_confidence": _counts(row["group_confidence"] for row in selected),
            "neighbor_transferability": _counts(
                value for row in selected for value in row["neighbor_transferability"]
            ),
            "final_confidence": _counts(row["final_confidence"] for row in selected),
            "top1_similarity": _numeric_summary(row["top1_similarity"] for row in selected),
            "full_pool_correct": sum(
                row["full_pool_prediction"] == row["gold"] for row in selected
            ),
        }
    return result


def _state_combinations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = Counter(
        (row["single_prior"], row["group_direction"], row["group_transferability"], row["agent_prediction"])
        for row in rows
    )
    return [
        {"single_prior": key[0], "group_direction": key[1],
         "group_transferability": key[2], "agent_prediction": key[3], "n": count}
        for key, count in sorted(keys.items(), key=lambda item: tuple(map(str, item[0])))
    ]


def _harm_gold_reliability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    harms = [row for row in rows if row["cohort"] == "harm_flip_to_low"]
    counts = [row["gold_record_count"] for row in harms]
    return {
        "n": len(harms),
        "unanimous_gold": sum(row["gold_agreement"] == 1.0 for row in harms),
        "accepted_majority_gold": sum(row["gold_agreement"] < 1.0 for row in harms),
        "median_source_record_count": statistics.median(counts) if counts else None,
        "min_source_record_count": min(counts) if counts else None,
        "max_source_record_count": max(counts) if counts else None,
    }


def _keyword_prevalence(rows: list[dict[str, Any]]) -> dict[str, int]:
    harms = [str(row["final_summary"] or "").lower() for row in rows if row["cohort"] == "harm_flip_to_low"]
    terms = {
        "molecular_weight": ("molecular weight", " mw", "size"),
        "lipophilicity": ("logp", "lipophil"),
        "polarity_or_permeability": ("polar", "permeab"),
        "ionization": ("ioniz", "neutral fraction", "charge"),
        "solubility": ("solub", "dissolution"),
        "metabolism_or_clearance": ("metabol", "clearance", "first-pass", "first pass"),
        "efflux_or_transporter": ("efflux", "transporter", "p-gp"),
        "flexibility": ("rotatable", "flexib"),
        "qed_or_lipinski": ("qed", "lipinski", "drug-like"),
    }
    return {name: sum(any(term in text for term in aliases) for text in harms)
            for name, aliases in terms.items()}


def _representative_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_index = {row["query_index"]: row for row in rows}
    fields = (
        "query_index", "cohort", "gold", "top1_similarity", "single_prior",
        "group_direction", "group_transferability", "gold_record_count",
        "gold_raw_value_examples", "final_summary",
    )
    return [{field: by_index[index][field] for field in fields}
            for index in (12, 14, 18, 42, 3, 2) if index in by_index]


def _diagnosis() -> list[dict[str, Any]]:
    return [
        {"rank": 1, "cause": "threshold calibration and default-low mismatch", "status": "supported",
         "evidence": "The benchmark asks only whether F reaches 20%, but mixed/uncertain property assessments are frequently resolved as low."},
        {"rank": 2, "cause": "transferability gate suppresses label direction", "status": "supported",
         "evidence": "Low similarity makes positive labels neutral/unclear even though gold positive prevalence remains stable across similarity bins."},
        {"rank": 3, "cause": "final synthesis treats insufficient positive evidence as negative evidence", "status": "supported",
         "evidence": "Mixed single prior plus neutral low-transfer group usually becomes a low final."},
        {"rank": 4, "cause": "binary label cards omit quantitative F and study context", "status": "contributing",
         "evidence": "The group traces explicitly cite missing quantitative/context information when discounting supervised analog outcomes."},
        {"rank": 5, "cause": "gold ambiguity or formatting failure", "status": "not_primary",
         "evidence": "Most harmful flips have unanimous multi-record gold and all audited calls are valid."},
    ]


def _llm_content(payload: dict[str, Any]) -> dict[str, Any]:
    content = payload.get("llm", {}).get("content")
    if not isinstance(content, dict):
        raise ValueError("Missing structured LLM content")
    return content


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _index(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {int(row["query_index"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate query_index")
    return result


def _counts(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _binary_counts(values: Iterable[int]) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {"0": counts[0], "1": counts[1]}


def _numeric_summary(values: Iterable[float]) -> dict[str, float | None]:
    items = list(values)
    return ({"min": None, "median": None, "max": None} if not items else
            {"min": min(items), "median": statistics.median(items), "max": max(items)})


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
