import json
from pathlib import Path

from tools.chembl_tool.paper_experiments.rl_lora.experiment_contract import (
    ONE_PASS_REWARD_VERSION,
    ONE_PASS_REWARD_WEIGHTS,
    ONE_PASS_TRAINING_RECIPE,
)
from tools.chembl_tool.paper_experiments.rl_lora.training_contract import (
    reward_metadata_from_row,
    score_training_response,
    training_result_logs,
    training_result_metrics,
)
from tools.chembl_tool.paper_experiments.rl_lora.one_pass_runtime import (
    inference_contract_sha256,
    load_fresh_one_pass_audit,
    one_pass_validation_errors,
)
from tools.chembl_tool.common.json_utils import sha256_file
from tools.chembl_tool.paper_experiments.rl_lora.validate_backend_contract import (
    validate_nemo_config,
)


ROOT = Path(__file__).resolve().parents[3]
FORMAL_NEMO_CONFIG = (
    ROOT
    / "tools/chembl_tool/paper_experiments/rl_lora/configs"
    / "grpo_gpt_oss_20b_one_pass_bio.yaml"
)


def _row() -> dict:
    schema = {
        "single_prediction": "high | low",
        "single_analysis": {"summary": "string"},
        "analog_prediction": "high | low",
        "analog_analysis": {"summary": "string"},
        "bioavailability_prediction": "high | low",
    }
    return {
        "contract_version": "one_pass_full_flat_visible_prefetched.v1",
        "messages": [
            {"role": "system", "content": "classify"},
            {
                "role": "user",
                "content": json.dumps({"required_json_schema": schema}),
            },
        ],
        "gold_label": 1,
        "prediction_field": "bioavailability_prediction",
        "single_prediction_field": "single_prediction",
        "analog_prediction_field": "analog_prediction",
        "negative_value": "low",
        "positive_value": "high",
        "required_fields": list(schema),
        "source_task": "bioavailability_ma",
        "source_fold": 0,
        "source_index": 7,
        "backend_only_field": "must not reach reward",
    }


def test_tinker_and_nemo_share_one_private_reward_surface():
    row = _row()
    metadata = reward_metadata_from_row(row)
    assert "backend_only_field" not in metadata
    assert metadata == reward_metadata_from_row(metadata)

    response = json.dumps(
        {
            "single_prediction": "high",
            "single_analysis": {"summary": "single"},
            "analog_prediction": "low",
            "analog_analysis": {"summary": "analog"},
            "bioavailability_prediction": "high",
        }
    )
    from_tinker_row = score_training_response(response, row)
    from_nemo_metadata = score_training_response(response, metadata)
    assert from_tinker_row == from_nemo_metadata
    assert from_tinker_row.reward == 1.35
    assert training_result_metrics(from_tinker_row)["final_correct"] == 1.0
    assert training_result_logs(from_tinker_row, row)["final_prediction"] == 1


def test_shared_recipe_and_reward_values_are_explicit():
    assert ONE_PASS_TRAINING_RECIPE.reward_version == ONE_PASS_REWARD_VERSION
    assert ONE_PASS_TRAINING_RECIPE.rollout_batch_size == 32
    assert ONE_PASS_TRAINING_RECIPE.max_completion_tokens == 6144
    assert ONE_PASS_REWARD_WEIGHTS.final_correct == 1.0
    assert ONE_PASS_REWARD_WEIGHTS.final_wrong == -1.0
    assert ONE_PASS_REWARD_WEIGHTS.resolved_conflict == 0.15


def test_formal_nemo_config_matches_shared_recipe():
    result = validate_nemo_config(FORMAL_NEMO_CONFIG)
    assert result["status"] == "pass"
    assert result["deviations"] == {}
    assert result["backend_invariant_failures"] == {}


def test_nemo_config_drift_is_reported(tmp_path: Path):
    altered = tmp_path / "drift.yaml"
    altered.write_text(
        "\n".join(
            (
                f"defaults: {FORMAL_NEMO_CONFIG}",
                "policy:",
                "  megatron_cfg:",
                "    optimizer:",
                "      lr: 2.0e-4",
                "",
            )
        ),
        encoding="utf-8",
    )
    result = validate_nemo_config(altered)
    assert result["status"] == "fail"
    assert result["deviations"]["learning_rate"] == {
        "expected": 1.0e-4,
        "observed": 2.0e-4,
    }


def test_one_pass_nested_schema_validation_is_shared():
    row = _row()
    incomplete = {
        "single_prediction": "high",
        "single_analysis": {},
        "analog_prediction": "high",
        "analog_analysis": {"summary": "analog"},
        "bioavailability_prediction": "high",
    }
    assert one_pass_validation_errors(incomplete, row) == [
        "single_analysis_missing_fields:summary"
    ]
    incomplete["single_analysis"] = {"summary": "single"}
    assert one_pass_validation_errors(incomplete, row) == []


def test_inference_contract_hash_binds_decode_settings():
    base = {"model": "gpt-oss", "max_tokens": 6144, "temperature": 0.0}
    assert inference_contract_sha256(base) == inference_contract_sha256(dict(base))
    assert inference_contract_sha256(base) != inference_contract_sha256(
        {**base, "max_tokens": 20480}
    )


def test_training_can_require_v2_data_audit(tmp_path: Path):
    data = tmp_path / "bio.jsonl"
    data.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    audit_path = data.with_suffix(".audit.json")
    common = {
        "status": "pass",
        "errors": [],
        "data_sha256": sha256_file(data),
        "n_rows": 1,
        "gold_visible_in_prompt": False,
        "source_assistant_outputs_in_prompt": False,
        "live_tool_calls_available_to_llm": False,
    }
    audit_path.write_text(
        json.dumps({**common, "audit_contract": "one_pass_data_audit.v1"}),
        encoding="utf-8",
    )
    assert load_fresh_one_pass_audit(data)["audit_contract"].endswith("v1")
    try:
        load_fresh_one_pass_audit(
            data,
            minimum_contract="one_pass_data_audit.v2",
        )
    except ValueError as exc:
        assert "older than required" in str(exc)
    else:
        raise AssertionError("training accepted a v1-only data audit")

    audit_path.write_text(
        json.dumps({**common, "audit_contract": "one_pass_data_audit.v2"}),
        encoding="utf-8",
    )
    assert (
        load_fresh_one_pass_audit(
            data,
            minimum_contract="one_pass_data_audit.v2",
        )["audit_contract"]
        == "one_pass_data_audit.v2"
    )
