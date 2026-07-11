import json

import pytest

from tools.chembl_tool.common.reasoning_calls import bound_group_prompt_payload, load_frozen_single_analysis


def test_load_frozen_single_analysis(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / "single_molecule_reasoning_output.json"
    path.write_text(json.dumps({"status": "ok", "llm": {"content": {"confidence": "low"}}}))

    output = load_frozen_single_analysis(str(run_dir))

    assert output["status"] == "ok"
    assert output["reused_from"] == str(path)


def test_load_frozen_single_analysis_rejects_failed_branch(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "single_molecule_reasoning_output.json").write_text(json.dumps({"status": "error"}))

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
    assert len(neighbor["evidence_rows"]) == 100
    assert neighbor["evidence_rows"][0]["row"] == 0
    assert neighbor["evidence_rows"][-1]["row"] == 299
    assert neighbor["evidence_rows_total"] == 300
    assert neighbor["evidence_rows_in_prompt"] == 100
    assert neighbor["evidence_rows_truncated"] is True
    assert result["prompt_transport"]["oversize_guard_applied"] is True
    assert result["prompt_transport"]["sampling"] == "deterministic_even_spacing"
