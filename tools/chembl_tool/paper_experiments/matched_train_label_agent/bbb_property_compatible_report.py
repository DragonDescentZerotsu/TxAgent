"""Paired result and trace-control summary for the frozen E16 BBB candidate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic
from tools.chembl_tool.paper_experiments.paired_binary_predictions import (
    paired_binary_summary,
)

from .audit import binary_diagnostics, retrieval_signature, run_has_prompt_identity_leak
from .bbb_property_compatibility import budget_key
from .bbb_property_compatible_contract import (
    AVAILABILITY_ROOT,
    BASELINE_V3_RUN_ROOT,
    DEFAULT_OUTPUT_ROOT,
    SCHEMA_VERSION,
    SIMILARITY_BUDGET,
    SPEC,
)
from .contract import agent_batch, replay_batch


def summarize(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    candidate_batch = agent_batch(output_root, SPEC)
    baseline_batch = (
        BASELINE_V3_RUN_ROOT
        / "bbb_martins"
        / "bbb_martins__matched_train_label_direct"
    )
    candidate_rows = _indexed(_read_jsonl(candidate_batch / "predictions.jsonl"))
    baseline_rows = _indexed(_read_jsonl(baseline_batch / "predictions.jsonl"))
    availability = _indexed(_read_jsonl(AVAILABILITY_ROOT / "query_audit.jsonl"))
    if set(candidate_rows) != set(baseline_rows) or set(candidate_rows) != set(availability):
        raise ValueError("Candidate, baseline, and availability coverage differ")

    labels, baseline_vote, baseline_agent = [], [], []
    candidate_vote, candidate_agent, paired_rows = [], [], []
    group_transferability: dict[str, list[str]] = {"baseline": [], "candidate": []}
    retrieval_mismatches = identity_leaks = 0
    key = budget_key(SIMILARITY_BUDGET)
    for index in sorted(candidate_rows):
        candidate, baseline, audit = candidate_rows[index], baseline_rows[index], availability[index]
        if candidate.get("status") != "ok" or candidate.get("final_status") != "ok":
            raise ValueError(f"Candidate failure at query {index}")
        replay = _load_retrieval(output_root, index, replay=True)
        actual = _load_retrieval(output_root, index, replay=False)
        baseline_retrieval = _load_batch_retrieval(
            baseline_batch, "bbb_martins__matched_train_label_direct", index
        )
        retrieval_mismatches += int(retrieval_signature(replay) != retrieval_signature(actual))
        selected_indices, selected_labels = _neighbor_indices_and_labels(
            replay["groups"][0]["neighbors"]
        )
        baseline_indices, baseline_labels = _neighbor_indices_and_labels(
            baseline_retrieval["groups"][0]["neighbors"]
        )
        if selected_indices != audit["budgets"][key]["train_indices"]:
            retrieval_mismatches += 1
        identity_leaks += int(
            run_has_prompt_identity_leak(
                candidate_batch, candidate, actual, read_jsonl=_read_jsonl
            )
        )
        label = int(candidate["label"])
        original_vote, vote = int(sum(baseline_labels) >= 2), int(sum(selected_labels) >= 2)
        baseline_prediction = int(baseline["pred_label"])
        candidate_prediction = int(candidate["pred_label"])
        labels.append(label)
        baseline_vote.append(original_vote)
        baseline_agent.append(baseline_prediction)
        candidate_vote.append(vote)
        candidate_agent.append(candidate_prediction)
        overlap = len(set(baseline_indices) & set(selected_indices))
        group_transferability["baseline"].append(
            _group_content(baseline_batch, baseline)["transferability"]
        )
        group_transferability["candidate"].append(
            _group_content(candidate_batch, candidate)["transferability"]
        )
        paired_rows.append(
            {
                "query_index": index,
                "Y": label,
                "baseline_morgan_vote": original_vote,
                "baseline_v3_agent": baseline_prediction,
                "property_compatible_vote": vote,
                "property_compatible_v3_agent": candidate_prediction,
                "baseline_train_indices": baseline_indices,
                "selected_train_indices": selected_indices,
                "neighbor_overlap": overlap,
            }
        )
    if retrieval_mismatches or identity_leaks:
        raise ValueError(
            f"Audit failed: retrieval_mismatches={retrieval_mismatches}, identity_leaks={identity_leaks}"
        )

    vote_effect = paired_binary_summary(labels, baseline_vote, candidate_vote)
    primary = paired_binary_summary(labels, baseline_agent, candidate_agent)
    vote_to_agent = paired_binary_summary(labels, candidate_vote, candidate_agent)
    result = {
        "schema_version": SCHEMA_VERSION,
        "n": len(labels),
        "retrieval_audit": {
            "mismatches": 0,
            "identity_leaks": 0,
            "selection_uses_labels": False,
            "similarity_cost_budget": SIMILARITY_BUDGET,
        },
        "baseline_v3_agent": {**_metrics(labels, baseline_agent), "batch": str(baseline_batch)},
        "baseline_morgan_vote": _metrics(labels, baseline_vote),
        "property_compatible_vote": _metrics(labels, candidate_vote),
        "property_compatible_v3_agent": {
            **_metrics(labels, candidate_agent),
            "batch": str(candidate_batch),
        },
        "property_compatible_vote_minus_baseline_morgan_vote": vote_effect,
        "candidate_minus_baseline_v3_agent": primary,
        "candidate_agent_minus_compatible_vote": vote_to_agent,
        "selector_response_audit": _selector_response_audit(
            paired_rows, group_transferability
        ),
        "promotion_gate": {
            "macro_f1_improved": primary["delta_macro_f1"] > 0,
            "paired_ci_lower_above_zero": primary["delta_macro_f1_bootstrap_95ci"][0] > 0,
            "accuracy_not_lower": primary["right_accuracy"] >= primary["left_accuracy"],
        },
    }
    result["promotion_gate"]["passed"] = all(result["promotion_gate"].values())
    analysis = output_root / "analysis"
    write_json_atomic(analysis / "summary.json", result)
    write_jsonl_atomic(analysis / "paired_predictions.jsonl", paired_rows)
    (analysis / "report.md").write_text(_report(result), encoding="utf-8")
    return result


def _metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    paired = paired_binary_summary(labels, predictions, predictions)
    return {
        "accuracy": paired["left_accuracy"],
        "macro_f1": paired["left_macro_f1"],
        **binary_diagnostics(labels, predictions),
    }


def _load_retrieval(output_root: Path, index: int, *, replay: bool) -> dict[str, Any]:
    batch = replay_batch(output_root, SPEC) if replay else agent_batch(output_root, SPEC)
    return _load_batch_retrieval(batch, SPEC.condition, index)


def _load_batch_retrieval(batch: Path, condition: str, index: int) -> dict[str, Any]:
    path = batch / "runs" / f"{condition}_idx{index:05d}" / "retrieval.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _neighbor_indices_and_labels(
    neighbors: list[dict[str, Any]],
) -> tuple[list[int], list[int]]:
    rows = [row for neighbor in neighbors for row in neighbor["evidence_rows"]]
    return (
        [int(row["train_index"]) for row in rows],
        [int(row["train_label"]) for row in rows],
    )


def _group_content(batch: Path, prediction: dict[str, Any]) -> dict[str, Any]:
    path = batch / "runs" / prediction["run_id"] / "group_reasoning_outputs.jsonl"
    return _read_jsonl(path)[0]["llm"]["content"]


def _selector_response_audit(
    rows: list[dict[str, Any]], transferability: dict[str, list[str]]
) -> dict[str, Any]:
    def cohort(predicate: Any) -> dict[str, int]:
        selected = [row for row in rows if predicate(row)]
        flips = [
            row
            for row in selected
            if row["baseline_v3_agent"] != row["property_compatible_v3_agent"]
        ]
        return {
            "n": len(selected),
            "agent_prediction_flips": len(flips),
            "agent_rescues": sum(
                row["property_compatible_v3_agent"] == row["Y"] for row in flips
            ),
            "agent_harms": sum(row["baseline_v3_agent"] == row["Y"] for row in flips),
        }

    return {
        "neighbor_overlap_distribution": {
            str(overlap): sum(row["neighbor_overlap"] == overlap for row in rows)
            for overlap in range(4)
        },
        "same_neighbor_set": cohort(lambda row: row["neighbor_overlap"] == 3),
        "same_ordered_neighbors": cohort(
            lambda row: row["baseline_train_indices"] == row["selected_train_indices"]
        ),
        "reordered_same_neighbor_set": cohort(
            lambda row: row["neighbor_overlap"] == 3
            and row["baseline_train_indices"] != row["selected_train_indices"]
        ),
        "changed_neighbor_set": cohort(lambda row: row["neighbor_overlap"] < 3),
        "same_vote": cohort(
            lambda row: row["baseline_morgan_vote"] == row["property_compatible_vote"]
        ),
        "changed_vote": cohort(
            lambda row: row["baseline_morgan_vote"] != row["property_compatible_vote"]
        ),
        "group_transferability": {
            name: {
                state: values.count(state)
                for state in ("high", "moderate", "low", "not_applicable")
            }
            for name, values in transferability.items()
        },
        "group_transferability_changed": sum(
            left != right
            for left, right in zip(
                transferability["baseline"], transferability["candidate"], strict=True
            )
        ),
    }


def _indexed(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["query_index"]): row for row in rows}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _report(result: dict[str, Any]) -> str:
    baseline, candidate = result["baseline_v3_agent"], result["property_compatible_v3_agent"]
    delta = result["candidate_minus_baseline_v3_agent"]
    low, high = delta["delta_macro_f1_bootstrap_95ci"]
    audit = result["selector_response_audit"]
    return (
        "# BBB property-compatible matched-v3 valid candidate\n\n"
        f"- baseline v3 agent: acc {baseline['accuracy']:.4f}, macro-F1 {baseline['macro_f1']:.4f}\n"
        f"- candidate v3 agent: acc {candidate['accuracy']:.4f}, macro-F1 {candidate['macro_f1']:.4f}\n"
        f"- macro-F1 delta: {delta['delta_macro_f1']:+.4f} [{low:+.4f},{high:+.4f}]\n"
        f"- changed neighbor sets: {audit['changed_neighbor_set']['n']}/{result['n']}\n"
        f"- changed-vote agent response: {audit['changed_vote']['agent_prediction_flips']}/"
        f"{audit['changed_vote']['n']} flips, {audit['changed_vote']['agent_rescues']} rescues / "
        f"{audit['changed_vote']['agent_harms']} harms\n"
        f"- identical ordered-input rerun control: {audit['same_ordered_neighbors']['agent_prediction_flips']}/"
        f"{audit['same_ordered_neighbors']['n']} flips, {audit['same_ordered_neighbors']['agent_rescues']} rescues / "
        f"{audit['same_ordered_neighbors']['agent_harms']} harms\n"
        f"- promotion gate: {'PASS' if result['promotion_gate']['passed'] else 'FAIL'}\n"
    )
