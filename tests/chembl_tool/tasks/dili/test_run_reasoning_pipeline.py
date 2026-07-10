from tools.chembl_tool.tasks.dili.run_reasoning_pipeline import (
    _clean_evidence_row,
    _final_instructions,
    _group_prompt_payload,
    _needs_json_retry,
    _parse_json_content,
)


def test_clean_evidence_row_excludes_internal_derived_fields():
    row = {
        "assay_chembl_id": "CHEMBL1",
        "assay_tier": "Tier 2",
        "standard_type": "ALT",
        "standard_value": 10,
        "assay_description": "Rat liver ALT",
        "evidence_direction": "in_vivo_liver_injury_signal",
        "evidence_strength": "strong",
        "endpoint_group_reason": "derived",
    }

    cleaned = _clean_evidence_row(row)

    assert cleaned["source"]["record_id"] == "CHEMBL1"
    assert cleaned["group"]["tier"] == "Tier 2"
    assert cleaned["endpoint"]["name"] == "ALT"
    assert "evidence_direction" not in cleaned
    assert "evidence_strength" not in cleaned
    assert "endpoint_group_reason" not in cleaned


def test_group_prompt_uses_dili_schema_fields():
    payload = _group_prompt_payload(
        {"input_smiles": "CCO", "canonical_smiles": "CCO"},
        {
            "group_id": "Tier 5.reactive_metabolite_or_covalent_binding",
            "tier": "Tier 5",
            "endpoint_group": "reactive_metabolite_or_covalent_binding",
            "neighbors": [
                {
                    "rank": 1,
                    "molecule_chembl_id": "CHEMBL1",
                    "canonical_smiles": "CCO",
                    "similarity": 0.7,
                    "similarity_bucket": "moderate_analog",
                    "evidence_rows": [{"assay_chembl_id": "CHEMBL1"}],
                }
            ],
        },
    )

    key_schema = payload["required_json_schema"]["key_evidence"][0]
    assert payload["task"] == "Group-level DILI analog transferability analysis."
    assert "effect_on_dili_reasoning" in key_schema
    assert "useful_for_dili_reasoning" in payload["required_json_schema"]


def test_final_instructions_use_dili_prediction_contract():
    text = "\n".join(_final_instructions())

    assert "dili_prediction='dili_risk'" in text
    assert "dili_prediction='no_dili_risk'" in text
    assert "DILI is a human clinical liver-injury phenotype" in text


def test_empty_or_unparsed_json_response_triggers_retry():
    assert _needs_json_retry(_parse_json_content("   "), "   ")
    assert _needs_json_retry(_parse_json_content("not json"), "not json")
    assert not _needs_json_retry(_parse_json_content('{"ok": true}'), '{"ok": true}')
