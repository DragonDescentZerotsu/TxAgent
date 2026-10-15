from __future__ import annotations

import json

from predict.harnesses.branches.prompt import attach_external_condition, clean_exact_match, clean_shared_assay_context, llm_evidence_query_payload, llm_query_payload
from predict.utils.query import external_condition_sentence
from predict.harnesses.branches.artifacts import write_trace_jsonl


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


def test_external_condition_is_natural_language_and_evidence_only():
    record = {"condition_group": "disease=bacterial_meningitis"}
    sentence = external_condition_sentence(record)
    assert sentence == (
        "This prediction concerns the query molecule under the disease condition "
        "bacterial meningitis."
    )
    retrieval = {"query": {"identity_hidden": True}}
    attach_external_condition(retrieval, record)
    query = retrieval["query"]
    assert "external_condition" not in llm_query_payload(query)
    assert llm_evidence_query_payload(query)["external_condition"] == sentence


def test_null_or_missing_condition_is_omitted():
    for record in ({}, {"condition_group": "no_reported_external_condition"}):
        retrieval = {"query": {"input_smiles": "CC", "canonical_smiles": "CC"}}
        attach_external_condition(retrieval, record)
        assert "external_condition" not in retrieval["query"]
        assert "external_condition" not in llm_evidence_query_payload(retrieval["query"])


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
