"""Deterministic stage-by-stage trace diagnosis for the frozen E16 BBB run."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic
from tools.chembl_tool.paper_experiments.paired_binary_predictions import (
    paired_binary_summary,
)

from .bbb_property_compatibility import budget_key
from .bbb_property_compatible_contract import (
    AVAILABILITY_ROOT,
    BASELINE_V3_RUN_ROOT,
    DEFAULT_OUTPUT_ROOT,
    SIMILARITY_BUDGET,
    SPEC,
)
from .contract import agent_batch


BASELINE_CONDITION = "bbb_martins__matched_train_label_direct"
GROUP_CORE_FIELDS = (
    "useful_for_bbb_reasoning",
    "transferability",
    "evidence_direction",
    "confidence",
    "observed_outcome_direction",
)


def diagnose(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    candidate_batch = agent_batch(output_root, SPEC)
    baseline_batch = BASELINE_V3_RUN_ROOT / "bbb_martins" / BASELINE_CONDITION
    paired = _read_jsonl(output_root / "analysis" / "paired_predictions.jsonl")
    availability = {
        int(row["query_index"]): row
        for row in _read_jsonl(AVAILABILITY_ROOT / "query_audit.jsonl")
    }
    rows = [
        _paired_trace_row(
            row,
            baseline_batch=baseline_batch,
            candidate_batch=candidate_batch,
            availability=availability[int(row["query_index"])],
        )
        for row in paired
    ]
    if len(rows) != 366:
        raise ValueError(f"Expected 366 paired BBB traces, found {len(rows)}")

    exact = _where(rows, lambda row: row["input_change"] == "exact_order")
    if any(not row["group_input_same"] for row in exact):
        raise ValueError("Exact ordered retrievals must have identical group requests")
    reordered = _where(rows, lambda row: row["input_change"] == "reordered_same_set")
    membership = _where(rows, lambda row: row["input_change"] == "membership_changed")
    vote_rescues = _where(
        rows, lambda row: not row["baseline_vote_correct"] and row["candidate_vote_correct"]
    )
    vote_harms = _where(
        rows, lambda row: row["baseline_vote_correct"] and not row["candidate_vote_correct"]
    )
    positive_overrides = _where(
        rows,
        lambda row: row["candidate_evidence_direction"] == "supports_bbb_crossing"
        and row["candidate_prediction"] == 0,
    )
    labels = [row["Y"] for row in rows]
    predictions = [row["candidate_prediction"] for row in rows]
    positive_override_indices = {row["query_index"] for row in positive_overrides}
    positive_protected = [
        1
        if row["query_index"] in positive_override_indices
        else row["candidate_prediction"]
        for row in rows
    ]

    summary = {
        "schema_version": "bbb_property_compatible_trace_diagnosis.v1",
        "n": len(rows),
        "input_control": {
            "exact_order_and_group_input": {
                **_comparison_stats(exact),
                "group_core_changed": sum(not row["group_core_same"] for row in exact),
                "final_state_changed": sum(not row["final_state_same"] for row in exact),
                "group_response_text_exact": sum(
                    row["group_response_same"] for row in exact
                ),
            },
            "reordered_same_membership": _comparison_stats(reordered),
            "changed_membership": _comparison_stats(membership),
            "all_group_core_same": _comparison_stats(
                _where(rows, lambda row: row["group_core_same"])
            ),
        },
        "selector_signal": {
            "vote_changed": _comparison_stats(
                _where(rows, lambda row: row["baseline_vote"] != row["candidate_vote"])
            ),
            "retrieval_vote_rescues": _comparison_stats(vote_rescues),
            "retrieval_vote_harms": _comparison_stats(vote_harms),
            "property_distance_better": _comparison_stats(
                _where(rows, lambda row: row["property_distance_gain"] > 0)
            ),
            "property_distance_worse": _comparison_stats(
                _where(rows, lambda row: row["property_distance_gain"] < 0)
            ),
            "ionization_match_gain": _comparison_stats(
                _where(rows, lambda row: row["same_ionization_delta"] > 0)
            ),
        },
        "group_stage": {
            "baseline_transferability": _counts(rows, "baseline_transferability"),
            "candidate_transferability": _counts(rows, "candidate_transferability"),
            "core_changed": sum(not row["group_core_same"] for row in rows),
            "property_better_transferability_response": _directional_change(
                _where(rows, lambda row: row["property_distance_gain"] > 0),
                "baseline_transferability",
                "candidate_transferability",
            ),
        },
        "final_stage": {
            "candidate_evidence_direction": _label_state_stats(
                rows, "candidate_evidence_direction"
            ),
            "candidate_negative_basis": _label_state_stats(
                rows, "candidate_negative_basis"
            ),
            "unanimous_positive": _label_stats(
                _where(rows, lambda row: row["candidate_label_pattern"] == "111")
            ),
            "unanimous_positive_overridden_to_fail": _label_stats(
                _where(
                    rows,
                    lambda row: row["candidate_label_pattern"] == "111"
                    and row["candidate_prediction"] == 0,
                )
            ),
            "positive_direction_overridden_to_fail": _label_stats(positive_overrides),
            "positive_protection_counterfactual_not_for_promotion": paired_binary_summary(
                labels, predictions, positive_protected
            ),
        },
        "diagnosis": [
            "The selector changes only 36 majority votes; the agent captures three net correct predictions from those changes.",
            "The label-blind selector score and original Morgan rank are absent from the group prompt, so the group must reconstruct compatibility from long raw property comparisons.",
            "Identical ordered group inputs still produce unstable group core fields and downstream final states.",
            "BBB v3 still permits intrinsic passive barriers to override unanimous positive meaningful-CNS analog outcomes, creating a concentrated false-negative cohort.",
        ],
    }
    analysis = output_root / "analysis" / "trace_diagnosis"
    write_json_atomic(analysis / "summary.json", summary)
    write_jsonl_atomic(analysis / "paired_trace_states.jsonl", rows)
    (analysis / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _paired_trace_row(
    paired: dict[str, Any],
    *,
    baseline_batch: Path,
    candidate_batch: Path,
    availability: dict[str, Any],
) -> dict[str, Any]:
    index = int(paired["query_index"])
    baseline_run = baseline_batch / "runs" / f"{BASELINE_CONDITION}_idx{index:05d}"
    candidate_run = candidate_batch / "runs" / f"{SPEC.condition}_idx{index:05d}"
    baseline_group = _group_content(baseline_run)
    candidate_group = _group_content(candidate_run)
    baseline_final = _final_content(baseline_run)
    candidate_final = _final_content(candidate_run)
    baseline_indices, baseline_labels = _neighbors(baseline_run)
    candidate_indices, candidate_labels = _neighbors(candidate_run)
    baseline_vote, candidate_vote = int(sum(baseline_labels) >= 2), int(
        sum(candidate_labels) >= 2
    )
    budget = availability["budgets"][budget_key(SIMILARITY_BUDGET)]
    current = availability["current"]
    baseline_prediction = int(paired["baseline_v3_agent"])
    candidate_prediction = int(paired["property_compatible_v3_agent"])
    label = int(paired["Y"])
    return {
        "query_index": index,
        "Y": label,
        "baseline_prediction": baseline_prediction,
        "candidate_prediction": candidate_prediction,
        "baseline_correct": baseline_prediction == label,
        "candidate_correct": candidate_prediction == label,
        "baseline_vote": baseline_vote,
        "candidate_vote": candidate_vote,
        "baseline_vote_correct": baseline_vote == label,
        "candidate_vote_correct": candidate_vote == label,
        "baseline_label_pattern": "".join(map(str, baseline_labels)),
        "candidate_label_pattern": "".join(map(str, candidate_labels)),
        "baseline_train_indices": baseline_indices,
        "candidate_train_indices": candidate_indices,
        "input_change": _input_change(baseline_indices, candidate_indices),
        "group_input_same": _request_hash(baseline_run, "Direct.bbb")
        == _request_hash(candidate_run, "Direct.bbb"),
        "group_response_same": _response_text(baseline_run, "Direct.bbb")
        == _response_text(candidate_run, "Direct.bbb"),
        "group_core_same": _core(baseline_group) == _core(candidate_group),
        "baseline_transferability": baseline_group["transferability"],
        "candidate_transferability": candidate_group["transferability"],
        "baseline_evidence_direction": baseline_group["evidence_direction"],
        "candidate_evidence_direction": candidate_group["evidence_direction"],
        "baseline_final_state": baseline_final["integrated_outcome_state"],
        "candidate_final_state": candidate_final["integrated_outcome_state"],
        "final_state_same": baseline_final["integrated_outcome_state"]
        == candidate_final["integrated_outcome_state"],
        "candidate_negative_basis": candidate_final["negative_evidence_basis"],
        "property_distance_gain": current["mean_property_distance"]
        - budget["mean_property_distance"],
        "same_ionization_delta": budget["same_ionization_count"]
        - current["same_ionization_count"],
        "similarity_cost": current["mean_similarity"] - budget["mean_similarity"],
    }


def _comparison_stats(rows: list[dict[str, Any]]) -> dict[str, int]:
    flips = [row for row in rows if row["baseline_prediction"] != row["candidate_prediction"]]
    return {
        "n": len(rows),
        "prediction_flips": len(flips),
        "rescues": sum(row["candidate_correct"] and not row["baseline_correct"] for row in flips),
        "harms": sum(row["baseline_correct"] and not row["candidate_correct"] for row in flips),
        "baseline_correct": sum(row["baseline_correct"] for row in rows),
        "candidate_correct": sum(row["candidate_correct"] for row in rows),
    }


def _label_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "gold_y0": sum(row["Y"] == 0 for row in rows),
        "gold_y1": sum(row["Y"] == 1 for row in rows),
        "predicted_y0": sum(row["candidate_prediction"] == 0 for row in rows),
        "predicted_y1": sum(row["candidate_prediction"] == 1 for row in rows),
        "correct": sum(row["candidate_correct"] for row in rows),
    }


def _label_state_stats(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return {
        state: _label_stats(_where(rows, lambda row, value=state: row[field] == value))
        for state in sorted({str(row[field]) for row in rows})
    }


def _directional_change(
    rows: list[dict[str, Any]], before: str, after: str
) -> dict[str, int]:
    rank = {"not_applicable": -1, "low": 0, "moderate": 1, "high": 2}
    return {
        "n": len(rows),
        "improved": sum(rank[row[after]] > rank[row[before]] for row in rows),
        "unchanged": sum(rank[row[after]] == rank[row[before]] for row in rows),
        "worsened": sum(rank[row[after]] < rank[row[before]] for row in rows),
    }


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def _where(
    rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]
) -> list[dict[str, Any]]:
    return [row for row in rows if predicate(row)]


def _input_change(baseline: list[int], candidate: list[int]) -> str:
    if baseline == candidate:
        return "exact_order"
    if set(baseline) == set(candidate):
        return "reordered_same_set"
    return "membership_changed"


def _neighbors(run: Path) -> tuple[list[int], list[int]]:
    retrieval = json.loads((run / "retrieval.json").read_text(encoding="utf-8"))
    rows = [
        evidence
        for neighbor in retrieval["groups"][0]["neighbors"]
        for evidence in neighbor["evidence_rows"]
    ]
    return (
        [int(row["train_index"]) for row in rows],
        [int(row["train_label"]) for row in rows],
    )


def _group_content(run: Path) -> dict[str, Any]:
    return _read_jsonl(run / "group_reasoning_outputs.jsonl")[0]["llm"]["content"]


def _final_content(run: Path) -> dict[str, Any]:
    return json.loads((run / "final_reasoning_output.json").read_text(encoding="utf-8"))[
        "llm"
    ]["content"]


def _trace(run: Path, task: str) -> dict[str, Any]:
    return next(row for row in _read_jsonl(run / "trace_messages.jsonl") if row["task"] == task)


def _request_hash(run: Path, task: str) -> str:
    payload = json.dumps(
        _trace(run, task)["messages"][:2],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _response_text(run: Path, task: str) -> str:
    return str(_trace(run, task)["response_text"])


def _core(group: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(group[field] for field in GROUP_CORE_FIELDS)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _report(summary: dict[str, Any]) -> str:
    control = summary["input_control"]["exact_order_and_group_input"]
    positive = summary["final_stage"]["unanimous_positive_overridden_to_fail"]
    counterfactual = summary["final_stage"][
        "positive_protection_counterfactual_not_for_promotion"
    ]
    return (
        "# BBB E16 paired trace diagnosis\n\n"
        f"- exact ordered group inputs: {control['n']}; group core changed "
        f"{control['group_core_changed']}, final state changed {control['final_state_changed']}, "
        f"prediction flips {control['prediction_flips']}\n"
        f"- unanimous-positive outcomes overridden to fail: {positive['n']}; "
        f"gold Y=1/Y=0 = {positive['gold_y1']}/{positive['gold_y0']}\n"
        f"- diagnostic positive-protection macro-F1 delta: "
        f"{counterfactual['delta_macro_f1']:+.4f} "
        f"[{counterfactual['delta_macro_f1_bootstrap_95ci'][0]:+.4f},"
        f"{counterfactual['delta_macro_f1_bootstrap_95ci'][1]:+.4f}]\n"
        "- this counterfactual is diagnostic only and is not promotion or test evidence\n"
    )


def main() -> int:
    print(json.dumps(diagnose(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
