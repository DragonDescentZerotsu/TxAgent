import json

import pytest

from tools.chembl_tool.common.evidence_contract import ASSAY_RAW_CARD_PROMPT_PROFILE
from tools.chembl_tool.common.reasoning_calls import (
    RAW_ASSAY_MAX_PAYLOAD_TOKENS,
    bound_group_prompt_payload,
    load_frozen_single_analysis,
)


def test_load_frozen_single_analysis(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / "single_molecule_reasoning_output.json"
    path.write_text(
        json.dumps({"status": "ok", "llm": {"content": {"confidence": "low"}}})
    )

    output = load_frozen_single_analysis(str(run_dir))

    assert output["status"] == "ok"
    assert output["reused_from"] == str(path)


def test_load_frozen_single_analysis_rejects_failed_branch(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "single_molecule_reasoning_output.json").write_text(
        json.dumps({"status": "error"})
    )

    with pytest.raises(ValueError):
        load_frozen_single_analysis(str(run_dir))


def test_group_prompt_bound_leaves_normal_payload_unchanged():
    payload = {"neighbors": [{"evidence_rows": [{"row": 1}]}]}

    assert bound_group_prompt_payload(payload) is payload
    assert "prompt_transport" not in payload


def test_group_prompt_bound_evenly_samples_oversize_neighbor_with_audit_metadata():
    rows = [{"row": index, "text": "x" * 10_000} for index in range(300)]
    payload = {"neighbors": [{"evidence_rows": rows}]}

    result = bound_group_prompt_payload(payload)

    neighbor = result["neighbors"][0]
    assert 0 < len(neighbor["evidence_rows"]) < 100
    assert neighbor["evidence_rows"][0]["row"] == 0
    assert neighbor["evidence_rows"][-1]["row"] == 299
    assert result["prompt_transport"]["oversize_guard_applied"] is True
    assert result["prompt_transport"]["sampling"] == "deterministic_even_spacing"
    assert result["prompt_transport"]["original_evidence_rows"] == 300
    assert result["prompt_transport"]["evidence_rows_in_prompt"] == len(
        neighbor["evidence_rows"]
    )
    assert len(json.dumps(result, ensure_ascii=False).encode("utf-8")) <= 400_000


def test_group_prompt_bound_guards_payload_below_previous_750kb_gate():
    rows = [{"row": index, "text": "x" * 1_000} for index in range(450)]
    payload = {"neighbors": [{"evidence_rows": rows}]}

    result = bound_group_prompt_payload(payload)

    assert len(result["neighbors"][0]["evidence_rows"]) < len(rows)
    assert result["prompt_transport"]["original_bytes"] > 400_000
    assert result["prompt_transport"]["max_bytes"] == 400_000
    assert len(json.dumps(result, ensure_ascii=False).encode("utf-8")) <= 400_000


def test_group_prompt_bound_can_reduce_neighbors_after_one_row_each():
    payload = {
        "neighbors": [
            {"id": index, "evidence_rows": [{"text": "x" * 10_000}]}
            for index in range(100)
        ]
    }

    result = bound_group_prompt_payload(payload)

    assert len(result["neighbors"]) < 100
    assert result["neighbors"][0]["id"] == 0
    assert result["neighbors"][-1]["id"] == 99
    assert result["prompt_transport"]["original_neighbors"] == 100
    assert len(json.dumps(result, ensure_ascii=False).encode("utf-8")) <= 400_000


def test_raw_assay_profile_does_not_use_legacy_400kb_guard():
    payload = {"neighbors": [{"evidence_rows": [{"text": "x" * 500_000}]}]}

    result = bound_group_prompt_payload(
        payload,
        evidence_prompt_profile=ASSAY_RAW_CARD_PROMPT_PROFILE,
    )

    assert result is payload
    assert "prompt_transport" not in result


def test_raw_assay_profile_uses_token_aware_emergency_guard():
    payload = {
        "neighbors": [
            {
                "id": 1,
                "evidence_rows": [
                    {"row": index, "text": "x" * 10_000} for index in range(100)
                ],
            }
        ]
    }

    def token_counter(value):
        return (
            10_000
            * sum(
                len(neighbor.get("evidence_rows") or [])
                for neighbor in value.get("neighbors") or []
            )
            + 100
        )

    result = bound_group_prompt_payload(
        payload,
        evidence_prompt_profile=ASSAY_RAW_CARD_PROMPT_PROFILE,
        token_counter=token_counter,
    )

    transport = result["prompt_transport"]
    assert transport["limit_type"] == "deepseek_payload_tokens"
    assert transport["original_payload_tokens"] == 1_000_100
    assert transport["max_payload_tokens"] == RAW_ASSAY_MAX_PAYLOAD_TOKENS
    assert token_counter(result) <= RAW_ASSAY_MAX_PAYLOAD_TOKENS
