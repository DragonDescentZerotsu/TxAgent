"""Reusable contract and outcome diagnostics for the matched experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from tools.chembl_tool.common.identity_blind import find_identity_blind_leaks


def binary_diagnostics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    pairs = list(zip(labels, predictions, strict=True))
    tn = sum(label == 0 and prediction == 0 for label, prediction in pairs)
    fp = sum(label == 0 and prediction == 1 for label, prediction in pairs)
    fn = sum(label == 1 and prediction == 0 for label, prediction in pairs)
    tp = sum(label == 1 and prediction == 1 for label, prediction in pairs)
    return {
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "recall_y0": tn / (tn + fp) if tn + fp else 0.0,
        "recall_y1": tp / (tp + fn) if tp + fn else 0.0,
        "predicted_y0": tn + fn,
        "predicted_y1": fp + tp,
    }


def stratum_diagnostics(
    labels: list[int],
    knn_predictions: list[int],
    agent_predictions: list[int],
    vote_patterns: list[str],
    pattern: str,
) -> dict[str, Any]:
    rows = [
        (label, knn, agent)
        for label, knn, agent, observed_pattern in zip(
            labels,
            knn_predictions,
            agent_predictions,
            vote_patterns,
            strict=True,
        )
        if observed_pattern == pattern
    ]
    return {
        "n": len(rows),
        "prediction_flips": sum(knn != agent for _, knn, agent in rows),
        "knn_y0_to_agent_y1": sum(knn == 0 and agent == 1 for _, knn, agent in rows),
        "knn_y1_to_agent_y0": sum(knn == 1 and agent == 0 for _, knn, agent in rows),
        "agent_only_correct": sum(agent == label and knn != label for label, knn, agent in rows),
        "knn_only_correct": sum(knn == label and agent != label for label, knn, agent in rows),
    }


def retrieval_signature(retrieval: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            group.get("group_id"),
            neighbor.get("rank"),
            neighbor.get("molecule_chembl_id"),
            neighbor.get("canonical_smiles"),
            neighbor.get("similarity"),
            tuple(
                (row.get("train_index"), row.get("train_label"))
                for row in neighbor.get("evidence_rows") or []
            ),
        )
        for group in retrieval.get("groups") or []
        for neighbor in group.get("neighbors") or []
    ]


def evidence_sources(retrieval: dict[str, Any]) -> list[str]:
    return [
        str(((row.get("minimal_evidence") or {}).get("source") or {}).get("name") or "")
        for group in retrieval.get("groups") or []
        for neighbor in group.get("neighbors") or []
        for row in neighbor.get("evidence_rows") or []
    ]


def run_has_prompt_identity_leak(
    batch: Path,
    prediction: dict[str, Any],
    retrieval: dict[str, Any],
    *,
    read_jsonl: Callable[[Path], list[dict[str, Any]]],
) -> bool:
    run = batch / "runs" / str(prediction["run_id"])
    outputs = [
        json.loads((run / name).read_text(encoding="utf-8"))
        for name in ("single_molecule_reasoning_output.json", "final_reasoning_output.json")
    ]
    outputs.extend(read_jsonl(run / "group_reasoning_outputs_raw.jsonl"))
    for output in outputs:
        messages = [
            message
            for message in (output.get("llm") or {}).get("messages") or []
            if message.get("role") != "assistant"
        ]
        if messages and any(find_identity_blind_leaks(retrieval, messages).values()):
            return True
    return False
