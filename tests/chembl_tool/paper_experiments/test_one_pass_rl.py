import hashlib
import importlib
import json
from pathlib import Path

from tools.chembl_tool.common.json_utils import canonical_json_bytes
from tools.chembl_tool.paper_experiments.rl_lora.audit_one_pass_data import audit
from tools.chembl_tool.paper_experiments.rl_lora.materialize_split_one_pass import (
    TASK_SPECS,
    _apply_visibility_contract,
)
from tools.chembl_tool.paper_experiments.rl_lora.one_pass import (
    build_one_pass_row_from_trace,
)
from tools.chembl_tool.paper_experiments.rl_lora.one_pass_contract import (
    CONTRACT_VERSION,
    DEPLOYMENT_VISIBLE_PREFETCHED,
    VISIBLE_PREFETCHED_CONTRACT_VERSION,
    contract_version_for_visibility,
    is_one_pass_contract,
)
from tools.chembl_tool.paper_experiments.rl_lora.one_pass_reward import (
    score_one_pass_response,
)
from tools.chembl_tool.paper_experiments.rl_lora.select_one_pass_checkpoint import (
    select,
)


def _trace_row(task, index, label, payload, assistant="{}"):
    return {
        "task": task,
        "index": index,
        "label": label,
        "status": "ok",
        "messages": [
            {"role": "system", "content": "stage"},
            {"role": "user", "content": json.dumps(payload)},
            {"role": "assistant", "content": assistant},
        ],
    }


def test_fuses_trace_without_assistant_or_gold(tmp_path: Path):
    query = {"molecule_id": "query", "identity_hidden": True}
    single = {
        "query": query,
        "instructions": ["single"],
        "required_json_schema": {"reasoning_summary": "string"},
    }
    group = {
        "query": query,
        "group": {"group_id": "Flat.all_evidence"},
        "neighbors": [
            {
                "molecule_chembl_id": "neighbor_1_1",
                "canonical_smiles": "[hidden]",
            }
        ],
        "instructions": ["analog"],
        "required_json_schema": {"transferability": "high | low"},
    }
    final = {
        "query": query,
        "retrieval_coverage": {"n_neighbors_total": 1},
        "instructions": ["final"],
        "required_json_schema": {"bbb_prediction": "pass | fail"},
    }
    path = tmp_path / "trace_messages.jsonl"
    rows = [
        _trace_row("single_molecule", 7, 1, single, assistant="SECRET_SINGLE"),
        _trace_row("Flat.all_evidence", 7, 1, group, assistant="SECRET_GROUP"),
        _trace_row("final_summary", 7, 1, final, assistant="SECRET_FINAL"),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = build_one_pass_row_from_trace(path, task="bbb_martins", subset="valid")
    visible = json.dumps(result["messages"])
    assert result["contract_version"] == CONTRACT_VERSION
    assert result["gold_label"] == 1
    assert "SECRET" not in visible
    assert '"gold_label"' not in visible
    assert "single_prediction" in visible
    assert "analog_prediction" in visible
    assert result["source_neighbor_ids"] == ["neighbor_1_1"]

    incomplete = score_one_pass_response(
        json.dumps(
            {
                "single_prediction": "pass",
                "single_analysis": {},
                "analog_prediction": "pass",
                "analog_analysis": {},
                "bbb_prediction": "pass",
            }
        ),
        result,
    )
    assert incomplete.parsed_json
    assert not incomplete.full_schema


def test_reward_has_strict_final_dominance_and_conflict_resolution_tie():
    metadata = {
        "gold_label": 1,
        "prediction_field": "bbb_prediction",
        "single_prediction_field": "single_prediction",
        "analog_prediction_field": "analog_prediction",
        "negative_value": "fail",
        "positive_value": "pass",
        "required_fields": [
            "single_prediction",
            "analog_prediction",
            "bbb_prediction",
        ],
    }

    def reward(single, analog, final):
        return score_one_pass_response(
            json.dumps(
                {
                    "single_prediction": single,
                    "analog_prediction": analog,
                    "bbb_prediction": final,
                }
            ),
            metadata,
        ).reward

    correct_both = reward("pass", "pass", "pass")
    correct_one = reward("pass", "fail", "pass")
    correct_none = reward("fail", "fail", "pass")
    wrong_final_best_branches = reward("pass", "pass", "fail")
    assert correct_both == correct_one
    assert correct_one > correct_none > wrong_final_best_branches


def test_reward_final_correctness_dominates_every_branch_combination():
    metadata = {
        "prediction_field": "bioavailability_prediction",
        "single_prediction_field": "single_prediction",
        "analog_prediction_field": "analog_prediction",
        "negative_value": "low",
        "positive_value": "high",
        "required_fields": [
            "single_prediction",
            "analog_prediction",
            "bioavailability_prediction",
        ],
    }
    branch_values = ["low", "high", "invalid", None]
    for gold, correct_final, wrong_final in ((0, "low", "high"), (1, "high", "low")):
        metadata["gold_label"] = gold
        correct_rewards = []
        wrong_rewards = []
        for single in branch_values:
            for analog in branch_values:
                base = {
                    "single_prediction": single,
                    "analog_prediction": analog,
                }
                correct_rewards.append(
                    score_one_pass_response(
                        json.dumps(
                            {**base, "bioavailability_prediction": correct_final}
                        ),
                        metadata,
                    ).reward
                )
                wrong_rewards.append(
                    score_one_pass_response(
                        json.dumps({**base, "bioavailability_prediction": wrong_final}),
                        metadata,
                    ).reward
                )
        assert min(correct_rewards) > max(wrong_rewards)


def test_conflict_resolution_receives_only_the_frozen_conflict_bonus():
    metadata = {
        "gold_label": 0,
        "prediction_field": "skin_reaction_prediction",
        "single_prediction_field": "single_prediction",
        "analog_prediction_field": "analog_prediction",
        "negative_value": "no_risk",
        "positive_value": "risk",
        "required_fields": [
            "single_prediction",
            "analog_prediction",
            "skin_reaction_prediction",
        ],
    }
    result = score_one_pass_response(
        json.dumps(
            {
                "single_prediction": "risk",
                "analog_prediction": "no_risk",
                "skin_reaction_prediction": "no_risk",
            }
        ),
        metadata,
    )
    assert result.branches_disagree
    assert result.conflict_resolved_correctly
    assert result.reward == 1.35


def test_checkpoint_selection_uses_f1_then_accuracy_then_earlier_step(tmp_path: Path):
    common = {"n": 10, "n_failed": 0, "metrics": "m", "metrics_sha256": "x"}
    candidates = [
        {**common, "step": 10, "checkpoint": "early", "macro_f1": 0.7, "accuracy": 0.8},
        {
            **common,
            "step": 20,
            "checkpoint": "lower_f1",
            "macro_f1": 0.69,
            "accuracy": 0.9,
        },
        {**common, "step": 30, "checkpoint": "later", "macro_f1": 0.7, "accuracy": 0.8},
    ]
    result = select(candidates, tmp_path / "selection.json")
    assert result["selected"]["checkpoint"] == "early"
    assert result["test_read_or_used"] is False


def test_visible_prefetched_contract_is_independent_and_auditable(tmp_path: Path):
    row = {
        "messages": [
            {"role": "system", "content": "anonymous system"},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "contract_version": CONTRACT_VERSION,
                        "query": {},
                        "neighbors": [],
                        "output_instructions": [
                            "Do not identify or name the anonymous query or neighbors."
                        ],
                        "required_json_schema": {
                            "single_prediction": "pass | fail",
                            "single_analysis": {},
                            "analog_prediction": "pass | fail",
                            "analog_analysis": {},
                            "bbb_prediction": "pass | fail",
                        },
                    }
                ),
            },
        ]
    }
    _apply_visibility_contract(row, DEPLOYMENT_VISIBLE_PREFETCHED)
    payload = json.loads(row["messages"][1]["content"])
    payload["query"] = {
        "molecule_id": "query",
        "input_smiles": "CCO",
        "canonical_smiles": "CCO",
        "prefetched_molecule_properties": {"status": "ok", "content": "properties"},
    }
    row["messages"][1]["content"] = json.dumps(payload)
    row.update(
        {
            "gold_label": 1,
            "prediction_field": "bbb_prediction",
            "single_prediction_field": "single_prediction",
            "analog_prediction_field": "analog_prediction",
            "negative_value": "fail",
            "positive_value": "pass",
            "required_fields": list(payload["required_json_schema"]),
            "source_task": "bbb_martins",
            "source_subset": "valid",
            "source_index": 0,
            "source_neighbor_ids": [],
            "gold_location": "environment_private_metadata_only",
            "retrieval_identity_policy": "parent_disjoint",
            "tool_calls_available_to_llm": False,
            "prefetched_tool_count": 1,
            "successful_prefetched_tool_count": 1,
            "failed_prefetched_tool_count": 0,
        }
    )
    row["prompt_sha256"] = hashlib.sha256(
        canonical_json_bytes(row["messages"])
    ).hexdigest()
    data = tmp_path / "visible.jsonl"
    source = tmp_path / "source.jsonl"
    data.write_text(json.dumps(row) + "\n")
    source.write_text(json.dumps({"drug": "CCO", "Y": 1}) + "\n")
    result = audit(
        data,
        tmp_path / "visible.audit.json",
        expected_rows=1,
        source_data=source,
        expected_visibility_mode=DEPLOYMENT_VISIBLE_PREFETCHED,
    )
    assert result["status"] == "pass"
    assert row["contract_version"] == VISIBLE_PREFETCHED_CONTRACT_VERSION
    assert (
        contract_version_for_visibility(DEPLOYMENT_VISIBLE_PREFETCHED)
        == row["contract_version"]
    )
    assert is_one_pass_contract(row["contract_version"])
    assert "anonymous query" not in json.dumps(row["messages"])


def test_direct_materializer_uses_public_task_prompt_adapters():
    for spec in TASK_SPECS.values():
        pipeline = importlib.import_module(spec["pipeline"])
        assert callable(pipeline.build_group_prompt_payload)
        assert pipeline._group_prompt_payload is pipeline.build_group_prompt_payload
