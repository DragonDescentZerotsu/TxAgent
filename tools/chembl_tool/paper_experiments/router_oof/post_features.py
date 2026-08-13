"""Materialize output-aware evidence and trace features without new LLM calls."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic

from .contract import ROUTER_FEATURE_STEM, TaskSpec, task_root
from .features import EVIDENCE_FEATURE_COLUMNS, KNN_FEATURE_COLUMNS, QUERY_FEATURE_COLUMNS
from .io import read_jsonl as _read_jsonl
from .post_contract import (
    DECISION_EVIDENCE_COLUMNS,
    OUTPUT_COLUMNS,
    POST_FEATURE_SCHEMA_VERSION,
    POST_FEATURE_STEM,
    TRACE_COLUMNS,
)


POST_FEATURE_COLUMNS = (
    OUTPUT_COLUMNS
    + QUERY_FEATURE_COLUMNS
    + KNN_FEATURE_COLUMNS
    + EVIDENCE_FEATURE_COLUMNS
    + DECISION_EVIDENCE_COLUMNS
    + TRACE_COLUMNS
)
POST_FEATURE_PROFILES = {
    "output_query_knn": OUTPUT_COLUMNS + QUERY_FEATURE_COLUMNS + KNN_FEATURE_COLUMNS,
    "output_query_knn_evidence": (
        OUTPUT_COLUMNS
        + QUERY_FEATURE_COLUMNS
        + KNN_FEATURE_COLUMNS
        + EVIDENCE_FEATURE_COLUMNS
        + DECISION_EVIDENCE_COLUMNS
    ),
    "output_query_knn_evidence_trace": POST_FEATURE_COLUMNS,
}


POSITIVE_DIRECTIONS = {
    "bbb_martins": {"supports_bbb_crossing", "influx_support"},
    "bioavailability_ma": {
        "supports_high_bioavailability",
        "absorption_support",
        "permeability_support",
        "solubility_support",
        "metabolic_stability_support",
    },
    "skin_reaction": {
        "supports_skin_reaction_risk",
        "sensitization_risk",
        "irritation_or_corrosion_risk",
        "phototoxicity_risk",
        "local_skin_damage_risk",
        "skin_exposure_support",
    },
}
NEGATIVE_DIRECTIONS = {
    "bbb_martins": {"argues_against_bbb_crossing", "efflux_risk"},
    "bioavailability_ma": {
        "argues_against_high_bioavailability",
        "solubility_risk",
        "first_pass_or_clearance_risk",
        "transporter_efflux_risk",
    },
    "skin_reaction": {"argues_against_skin_reaction_risk", "reduced_skin_exposure"},
}


def build_post_selector_features(
    spec: TaskSpec,
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    """Enrich existing OOF v2 rows with output/evidence/trace features."""
    root = task_root(output_root, spec.task)
    rows = _read_jsonl(root / f"{ROUTER_FEATURE_STEM}.jsonl")
    prediction_cache: dict[Path, dict[int, Mapping[str, Any]]] = {}
    enriched = [_enrich_row(spec, row, prediction_cache) for row in rows]
    enriched.sort(key=lambda row: int(row["train_index"]))
    path = root / f"{POST_FEATURE_STEM}.jsonl"
    write_jsonl_atomic(path, enriched)
    summary = _summary(spec.task, enriched, path)
    write_json_atomic(root / f"{POST_FEATURE_STEM}_summary.json", summary)
    return summary


def build_post_selector_valid_features(
    spec: TaskSpec,
    *,
    output_root: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enrich already contract-checked v2 valid features for v3 evaluation."""
    root = task_root(output_root, spec.task)
    rows = _read_jsonl(root / "router_v2" / "valid" / "features.jsonl")
    prediction_cache: dict[Path, dict[int, Mapping[str, Any]]] = {}
    enriched = [_enrich_row(spec, row, prediction_cache) for row in rows]
    path = (
        Path(output_dir) / "features.jsonl"
        if output_dir is not None
        else root / "post_selector_v3" / "valid" / "features.jsonl"
    )
    write_jsonl_atomic(path, enriched)
    return enriched, _summary(spec.task, enriched, path)


def _enrich_row(
    spec: TaskSpec,
    row: Mapping[str, Any],
    prediction_cache: dict[Path, dict[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    provenance = dict(row.get("provenance") or {})
    retrieval_path = Path(str(provenance["retrieval"]))
    run_dir = retrieval_path.parent
    agent_prediction_path = Path(str(provenance["agent_prediction"]))
    query_key = "local_query_index" if "local_query_index" in row else "query_index"
    query_index = int(row[query_key])
    agent_row = _prediction_by_index(agent_prediction_path, query_index, prediction_cache)
    agent_prediction = int(row["agent_prediction"])
    knn_prediction = int(row["knn_prediction"])
    group_contents = _group_contents(run_dir / "group_reasoning_outputs.jsonl")
    single_content = _llm_content(run_dir / "single_molecule_reasoning_output.json")
    final_payload = _read_json(run_dir / "final_reasoning_output.json")
    final_content = _payload_content(final_payload)
    features = {
        **{key: float(value) for key, value in dict(row["features"]).items()},
        "direction_knn0_agent1": float(knn_prediction == 0 and agent_prediction == 1),
        "direction_knn1_agent0": float(knn_prediction == 1 and agent_prediction == 0),
        **_decision_evidence_features(spec.task, group_contents, agent_prediction),
        **_trace_features(
            spec.task,
            agent_prediction=agent_prediction,
            agent_row=agent_row,
            group_contents=group_contents,
            single_content=single_content,
            final_payload=final_payload,
            final_content=final_content,
        ),
    }
    missing = sorted(set(POST_FEATURE_COLUMNS) - set(features))
    if missing:
        raise ValueError(f"Missing post-selector features for {spec.task}: {missing}")
    return {
        **dict(row),
        "schema_version": POST_FEATURE_SCHEMA_VERSION,
        "features": {key: float(features[key]) for key in POST_FEATURE_COLUMNS},
        "provenance": {
            **provenance,
            "single_trace": str(run_dir / "single_molecule_reasoning_output.json"),
            "group_trace": str(run_dir / "group_reasoning_outputs.jsonl"),
            "final_trace": str(run_dir / "final_reasoning_output.json"),
        },
    }


def _decision_evidence_features(
    task: str,
    groups: list[Mapping[str, Any]],
    agent_prediction: int,
) -> dict[str, float]:
    positive = negative = neutral = 0.0
    useful = high_transferability = high_confidence = 0
    directions: list[int] = []
    confidence_values: list[float] = []
    transfer_values: list[float] = []
    for content in groups:
        is_useful = bool(next((value for key, value in content.items() if key.startswith("useful_for_")), False))
        useful += int(is_useful)
        confidence = _level(str(content.get("confidence") or ""))
        transferability = _level(str(content.get("transferability") or ""))
        confidence_values.append(confidence)
        transfer_values.append(transferability)
        high_confidence += int(str(content.get("confidence") or "").lower() == "high")
        high_transferability += int(str(content.get("transferability") or "").lower() == "high")
        weight = confidence * transferability * float(is_useful)
        direction = str(content.get("evidence_direction") or "neutral_or_unclear")
        if direction in POSITIVE_DIRECTIONS[task]:
            positive += weight
            directions.append(1)
        elif direction in NEGATIVE_DIRECTIONS[task]:
            negative += weight
            directions.append(-1)
        else:
            neutral += weight
            directions.append(0)
    count = len(groups)
    total = positive + negative + neutral
    support_agent = positive if agent_prediction == 1 else negative
    support_knn = negative if agent_prediction == 1 else positive
    return {
        "decision_group_count": float(count),
        "decision_useful_group_fraction": useful / count if count else 0.0,
        "decision_positive_weight": positive,
        "decision_negative_weight": negative,
        "decision_neutral_weight": neutral,
        "decision_signed_margin": (positive - negative) / total if total else 0.0,
        "decision_conflict_fraction": float(1 in directions and -1 in directions),
        "decision_high_transferability_fraction": high_transferability / count if count else 0.0,
        "decision_high_confidence_fraction": high_confidence / count if count else 0.0,
        "decision_supports_agent_weight": support_agent,
        "decision_supports_knn_weight": support_knn,
        "decision_agent_support_margin": (support_agent - support_knn) / total if total else 0.0,
    }


def _trace_features(
    task: str,
    *,
    agent_prediction: int,
    agent_row: Mapping[str, Any],
    group_contents: list[Mapping[str, Any]],
    single_content: Mapping[str, Any],
    final_payload: Mapping[str, Any],
    final_content: Mapping[str, Any],
) -> dict[str, float]:
    final_confidence = str(agent_row.get("confidence") or final_content.get("confidence") or "").lower()
    single_confidence = str(single_content.get("confidence") or "").lower()
    group_confidences = [_level(str(content.get("confidence") or "")) for content in group_contents]
    group_transfer = [_level(str(content.get("transferability") or "")) for content in group_contents]
    group_directions = [_direction_sign(task, str(content.get("evidence_direction") or "")) for content in group_contents]
    single_direction = _single_direction(task, single_content)
    final_raw = str(((final_payload.get("llm") or {}).get("reasoning_content") or ""))
    final_summary = str(agent_row.get("final_summary") or final_content.get("final_summary") or "")
    group_agent = sum(direction == agent_prediction for direction in group_directions if direction in (0, 1))
    group_knn = sum(direction == 1 - agent_prediction for direction in group_directions if direction in (0, 1))
    group_neutral = sum(direction is None for direction in group_directions)
    group_count = len(group_directions)
    return {
        "trace_final_confidence_high": float(final_confidence == "high"),
        "trace_final_confidence_moderate": float(final_confidence == "moderate"),
        "trace_final_confidence_low": float(final_confidence == "low"),
        "trace_single_confidence_high": float(single_confidence == "high"),
        "trace_single_confidence_moderate": float(single_confidence == "moderate"),
        "trace_single_confidence_low": float(single_confidence == "low"),
        "trace_group_confidence_mean": sum(group_confidences) / len(group_confidences) if group_confidences else 0.0,
        "trace_group_transferability_mean": sum(group_transfer) / len(group_transfer) if group_transfer else 0.0,
        "trace_single_supports_agent": float(single_direction == agent_prediction),
        "trace_single_supports_knn": float(single_direction == 1 - agent_prediction),
        "trace_single_neutral": float(single_direction is None),
        "trace_single_final_agreement": float(single_direction == agent_prediction),
        "trace_group_supports_agent_fraction": group_agent / group_count if group_count else 0.0,
        "trace_group_supports_knn_fraction": group_knn / group_count if group_count else 0.0,
        "trace_group_neutral_fraction": group_neutral / group_count if group_count else 0.0,
        "trace_group_final_agreement": group_agent / (group_agent + group_knn) if group_agent + group_knn else 0.0,
        "trace_final_conflicting_evidence_count_log1p": math.log1p(_list_length(final_content.get("conflicting_evidence"))),
        "trace_final_evidence_gap_count_log1p": math.log1p(_list_length(final_content.get("evidence_gaps"))),
        "trace_final_caveat_count_log1p": math.log1p(_list_length(final_content.get("caveats"))),
        "trace_final_reasoning_chars_log1p": math.log1p(len(final_raw)),
        "trace_final_summary_chars_log1p": math.log1p(len(final_summary)),
    }


def _single_direction(task: str, content: Mapping[str, Any]) -> int | None:
    if task == "bbb_martins":
        value = str(content.get("passive_bbb_plausibility") or "").lower()
        return 1 if value == "high" else 0 if value == "low" else None
    if task == "bioavailability_ma":
        value = str(content.get("oral_bioavailability_prior") or "").lower()
        return 1 if value in {"high", "favorable"} else 0 if value in {"low", "unfavorable"} else None
    value = str(content.get("skin_reaction_prior") or "").lower()
    if value in {"high", "risk", "concerning", "favorable"}:
        return 1
    if value in {"low", "no_risk", "not_concerning", "unfavorable"}:
        return 0
    return None


def _direction_sign(task: str, value: str) -> int | None:
    if value in POSITIVE_DIRECTIONS[task]:
        return 1
    if value in NEGATIVE_DIRECTIONS[task]:
        return 0
    return None


def _group_contents(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    output = []
    for row in _read_jsonl(path):
        if str(row.get("status") or "") != "ok":
            continue
        content = ((row.get("llm") or {}).get("content") or {})
        if isinstance(content, Mapping):
            output.append(content)
    return output


def _llm_content(path: Path) -> Mapping[str, Any]:
    return _payload_content(_read_json(path))


def _payload_content(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    content = ((payload.get("llm") or {}).get("content") or {})
    return content if isinstance(content, Mapping) else {}


def _prediction_by_index(
    path: Path,
    index: int,
    cache: dict[Path, dict[int, Mapping[str, Any]]],
) -> Mapping[str, Any]:
    if path not in cache:
        cache[path] = {
            int(row.get("query_index", row.get("index"))): row
            for row in _read_jsonl(path)
            if row.get("query_index", row.get("index")) is not None
        }
    try:
        return cache[path][index]
    except KeyError as error:
        raise ValueError(f"No agent prediction index {index} in {path}") from error


def _level(value: str) -> float:
    return {"high": 1.0, "moderate": 2.0 / 3.0, "low": 1.0 / 3.0}.get(value.lower(), 0.0)


def _list_length(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _summary(task: str, rows: list[Mapping[str, Any]], path: Path) -> dict[str, Any]:
    counts = {name: sum(row["paired_outcome"] == name for row in rows) for name in ("both_correct", "knn_only_correct", "agent_only_correct", "both_wrong")}
    return {
        "schema_version": POST_FEATURE_SCHEMA_VERSION,
        "task": task,
        "n_rows": len(rows),
        "n_disagreements": counts["agent_only_correct"] + counts["knn_only_correct"],
        "paired_outcome_counts": counts,
        "feature_columns": list(POST_FEATURE_COLUMNS),
        "feature_profiles": {key: list(value) for key, value in POST_FEATURE_PROFILES.items()},
        "path": str(path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
