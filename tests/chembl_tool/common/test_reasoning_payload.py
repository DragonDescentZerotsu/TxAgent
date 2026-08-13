from __future__ import annotations

import json
import os

import pytest

from tools.chembl_tool.common.reasoning_payload import (
    clean_exact_match,
    clean_shared_assay_context,
    llm_query_payload,
    load_env_file,
    read_jsonl_record,
    write_trace_jsonl,
)


def test_llm_query_payload_preserves_visible_and_identity_blind_contracts():
    properties = {"output": {"text": "MW 42"}}
    assert llm_query_payload(
        {
            "input_smiles": "CC",
            "canonical_smiles": "CC",
            "prefetched_molecule_properties": properties,
            "internal_id": "hidden",
        }
    ) == {
        "input_smiles": "CC",
        "canonical_smiles": "CC",
        "tools_prefetched": True,
        "prefetched_molecule_properties": properties,
    }
    assert llm_query_payload(
        {
            "identity_hidden": True,
            "input_smiles": "CC",
            "prefetched_molecule_properties": properties,
        }
    ) == {
        "molecule_id": "query",
        "identity_hidden": True,
        "prefetched_molecule_properties": properties,
    }


def test_cleaners_keep_only_frozen_prompt_fields():
    exact = clean_exact_match(
        {"molecule_chembl_id": "CHEMBL1", "alogp": 2.5, "secret": 1}
    )
    assert exact["molecule_chembl_id"] == "CHEMBL1"
    assert exact["alogp"] == 2.5
    assert exact["hba"] == ""
    assert "secret" not in exact

    cleaned = clean_shared_assay_context(
        {
            "same_endpoint_activity": [
                {
                    "assay_chembl_id": "CHEMBL_A",
                    "standard_type_match": True,
                    "query_activity": {
                        "standard_type": "IC50",
                        "standard_value": 7,
                        "secret": 1,
                    },
                    "neighbor_activity": {},
                    "secret": 1,
                }
            ],
            "n_same_endpoint_activity": 1,
        }
    )
    card = cleaned["same_endpoint_activity"][0]
    assert card["query_activity"]["standard_value"] == 7
    assert card["neighbor_activity"]["standard_type"] == ""
    assert "secret" not in card
    assert "secret" not in card["query_activity"]
    assert cleaned["n_same_assay_different_endpoint_activity"] == 0


def test_jsonl_and_env_helpers_preserve_historical_parsing(tmp_path, monkeypatch):
    records = tmp_path / "records.jsonl"
    records.write_text('{"row": 0}\n{"row": 1}\n', encoding="utf-8")
    assert read_jsonl_record(records, 1) == {"row": 1}
    with pytest.raises(SystemExit, match="No record at index 2"):
        read_jsonl_record(records, 2)

    env_file = tmp_path / "run.env"
    env_file.write_text(
        "# comment\n TEST_REASONING_PAYLOAD = 'kept' \ninvalid\n", encoding="utf-8"
    )
    monkeypatch.delenv("TEST_REASONING_PAYLOAD", raising=False)
    load_env_file(env_file)
    assert os.environ["TEST_REASONING_PAYLOAD"] == "kept"


def test_write_trace_jsonl_preserves_stage_order_and_prediction_field(tmp_path):
    output = tmp_path / "trace.jsonl"
    base = {"status": "ok", "llm": {"content": {"chosen_label": "yes"}}}
    write_trace_jsonl(
        output,
        prediction_field="chosen_label",
        query_record={"Y": 1},
        query_index=3,
        smiles="[identity_blind]",
        single_output=base,
        group_outputs=[{"group_id": "direct", **base}],
        final_output=base,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["task"] for row in rows] == [
        "single_molecule",
        "direct",
        "final_summary",
    ]
    assert all(row["prediction"] == "yes" for row in rows)
    assert all(row["molecule_key"] == "index:3" for row in rows)
    assert rows[0]["smiles"] == "[identity_blind]"
