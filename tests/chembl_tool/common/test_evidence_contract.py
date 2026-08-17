from tools.chembl_tool.common.evidence_contract import (
    CONTRACT_VERSION,
    ASSAY_COMPACT_PROMPT_PROFILE,
    assay_evidence_for_llm,
    attach_minimal_evidence,
    evidence_for_group_llm,
    evidence_for_llm,
    numeric_only_evidence_row,
    validate_minimal_evidence,
)


def test_dense_assay_view_keeps_assay_measurement_and_bounds_repeated_text():
    row = {
        "molecule_chembl_id": "M1",
        "canonical_smiles": "CCO",
        "assay_chembl_id": "ASSAY1",
        "group_id": "Assay.ASSAY1",
        "standard_type": "oral bioavailability",
        "standard_value": "42",
        "standard_units": "%",
        "assay_description": "x" * 1000,
        "evidence_source": "Starling",
        "confidence_score": 0.9,
        "source_record_count": 4,
        "evidence_scope": {
            "assay_context": ["in vivo oral study"],
            "species_context": ["human"],
        },
        "relevance_score": 99,
        "relevance_rank": 1,
    }

    compact = assay_evidence_for_llm(row)

    assert compact["contract_version"] == CONTRACT_VERSION
    assert compact["assay_id"] == "ASSAY1"
    assert compact["endpoint"]["measurement"] == {"value": "42", "unit": "%"}
    assert compact["annotations"]["scope"]["assay_context"] == ["in vivo oral study"]
    assert len(compact["text"]["evidence_excerpt"]) == 160
    assert compact["provenance"]["source_record_count"] == 4
    assert "molecule" not in compact
    assert "relevance" not in str(compact)


def test_group_prompt_profile_isolated_from_standard_group_evidence():
    row = {
        "molecule_chembl_id": "M1",
        "canonical_smiles": "CCO",
        "assay_chembl_id": "A1",
        "standard_type": "endpoint",
        "standard_value": "1",
        "assay_description": "evidence",
        "evidence_source": "Starling",
    }
    standard = evidence_for_group_llm(row, {"group_id": "Mechanism.tier_1"})
    compact = evidence_for_group_llm(
        row,
        {
            "group_id": "Flat.assay_ranked_evidence",
            "evidence_prompt_profile": ASSAY_COMPACT_PROMPT_PROFILE,
        },
    )

    assert standard == evidence_for_llm(row)
    assert standard["molecule"]["id"] == "M1"
    assert "molecule" not in compact


def test_legacy_assay_replay_without_profile_keeps_compact_view():
    compact = evidence_for_group_llm(
        {"assay_chembl_id": "A1", "standard_type": "endpoint"},
        {"group_id": "Flat.assay_ranked_evidence"},
    )
    assert compact["assay_id"] == "A1"


def test_legacy_row_maps_to_minimal_contract_without_internal_direction_fields():
    row = {
        "molecule_chembl_id": "CHEMBL1",
        "canonical_smiles": "CCO",
        "assay_chembl_id": "CHEMBL_A1",
        "assay_tier": "Tier 1",
        "endpoint_group": "direct_outcome",
        "group_id": "Tier 1.direct_outcome",
        "standard_type": "fraction",
        "standard_relation": "=",
        "standard_value": 0.8,
        "standard_units": "ratio",
        "assay_description": "Measured in vivo.",
        "organism": "Homo sapiens",
        "evidence_source": "ChEMBL",
        "evidence_direction": "internal_positive_rule",
        "evidence_strength": "internal_strong_rule",
        "endpoint_group_reason": "internal mapping reason",
        "source_record_examples": [{"support_text": "reported", "pmid": "123"}],
    }

    record = evidence_for_llm(row)

    assert record["contract_version"] == CONTRACT_VERSION
    assert record["source"]["name"] == "ChEMBL"
    assert record["molecule"]["id"] == "CHEMBL1"
    assert record["group"]["id"] == "Tier 1.direct_outcome"
    assert record["endpoint"]["measurement"] == {"relation": "=", "value": 0.8, "unit": "ratio"}
    assert record["annotations"]["transferability"] == "not_assessed"
    assert record["examples"] == [{"support_text": "reported"}]
    assert "evidence_direction" not in str(record)
    assert "endpoint_group_reason" not in str(record)
    assert validate_minimal_evidence(record) == []


def test_attach_preserves_source_row_and_adds_contract():
    row = {
        "molecule_chembl_id": "SRC1",
        "canonical_smiles": "CCN",
        "group_id": "Mechanism.transport",
        "standard_type": "efflux ratio",
        "assay_description": "Bidirectional transport assay.",
        "evidence_source": "example-source",
        "evidence_role": "mechanistic_factor",
        "evidence_scope": {"species": "human"},
    }

    attached = attach_minimal_evidence(row)

    assert attached is row
    assert attached["standard_type"] == "efflux ratio"
    assert attached["minimal_evidence"]["annotations"]["evidence_role"] == "mechanistic_factor"
    assert attached["minimal_evidence"]["annotations"]["scope"] == {"species": "human"}


def test_validation_reports_only_structural_contract_errors():
    assert validate_minimal_evidence({}) == [
        "invalid_contract_version",
        "missing_source_name",
        "missing_molecule_identity",
        "missing_group_id",
        "missing_endpoint_and_evidence_text",
    ]


def test_numeric_only_evidence_removes_text_and_qualitative_only_rows():
    row = {
        "molecule_chembl_id": "M1",
        "canonical_smiles": "CCO",
        "group_id": "Observed.direct",
        "standard_type": "bioavailability",
        "standard_value": 42.0,
        "standard_units": "%",
        "assay_description": "Human oral study with formulation details",
        "source_support_texts": ["Forty-two percent after oral dosing"],
        "source_record_examples": [
            {
                "oral_bioavailability_value_percent": 42.0,
                "species": "human",
                "support_text": "Forty-two percent after oral dosing",
            }
        ],
    }

    numeric = numeric_only_evidence_row(row)

    assert numeric is not None
    assert numeric["minimal_evidence"]["text"] == {"evidence": "", "context": ""}
    assert numeric["minimal_evidence"]["examples"] == [{"oral_bioavailability_value_percent": 42.0}]
    assert numeric_only_evidence_row({**row, "standard_value": "qualitative"}) is None
