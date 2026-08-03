import json

import pytest

from tools.chembl_tool.common.evidence_contract import evidence_for_llm
from tools.chembl_tool.common.starling.normalization.organization import _llm_example
from tools.chembl_tool.tasks.bioavailability_ma.group_prompt_render import (
    _assay_transfer_evidence_record,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_source_column_contracts import (
    SOURCE_COLUMN_CONTRACT_VERSION,
    SOURCE_COLUMNS,
    llm_source_projection,
    normalized_column_contract,
    source_column_contract_manifest,
)


def _fg_record():
    source = {
        "pmid": "12345678",
        "extraction_id": "fg-1",
        "global_identifier": "compound-x",
        "confidence": 0.9,
        "paragraph_idx": 4,
        "support_text": "P_app was 2.5 ×10^-6 cm/s.",
        "molecule_name": "Drug X",
        "gut_wall_process": "permeability",
        "transporter_or_enzyme": None,
        "substrate_status": None,
        "assay_system": "Caco-2",
        "intestinal_site": None,
        "measured_value": "P_app = 2.5 ×10^-6 cm/s",
        "qualifying_conditions": None,
        "extra_details": None,
        "smiles": None,
    }
    return {
        "source_id": "fg",
        "source_payload_json": json.dumps(source),
        "canonical_endpoint": "bidirectional_permeability",
        "finite_scalar_value": 2.5e-6,
    }


def test_projection_contains_every_declared_source_field_and_no_canonical_fields():
    projection = llm_source_projection(_fg_record())

    assert projection["contract_version"] == SOURCE_COLUMN_CONTRACT_VERSION
    assert tuple(projection["source_fields"]) == SOURCE_COLUMNS["fg"]
    assert projection["source_fields"]["pmid"] == "12345678"
    assert projection["source_fields"]["measured_value"] == "P_app = 2.5 ×10^-6 cm/s"
    assert projection["source_fields"]["intestinal_site"] is None
    assert "canonical_endpoint" not in projection["source_fields"]
    assert "finite_scalar_value" not in projection["source_fields"]


def test_manifest_classifies_every_artifact_column_for_every_source():
    columns = [
        "source_record_id", "endpoint_name", "measurement_text", "canonical_endpoint",
        "finite_scalar_value", "unit_notation_factor", "source_payload_json",
        "global_context", "global_species_context",
        "canonical_unit_rule_id", "canonical_unit_conversion_factor",
    ]
    manifest = source_column_contract_manifest(columns)

    for source_id in SOURCE_COLUMNS:
        classified = manifest["sources"][source_id]["normalized_artifact_columns"]
        assert set(classified) == set(columns)
        assert all(set(item) == {"source_or_simply_cleaned"} for item in classified.values())
    assert normalized_column_contract("fg", columns)["measurement_text"] is True
    assert normalized_column_contract("fg", columns)["canonical_endpoint"] is False
    assert normalized_column_contract("fg", columns)["global_context"] is False
    assert normalized_column_contract("fh", columns)["global_species_context"] is False
    assert normalized_column_contract("fa", columns)["canonical_unit_rule_id"] is False
    assert normalized_column_contract("direct_hf", columns)["endpoint_name"] is False


def test_minimal_evidence_preserves_explicit_source_provenance_only_inside_contract():
    projection = llm_source_projection(_fg_record())
    example = {
        "source_contract": {key: value for key, value in projection.items() if key != "source_fields"},
        "source_fields": projection["source_fields"],
        "pmid": "legacy-private-field",
    }
    visible = evidence_for_llm(
        {
            "evidence_source": "Starling normalized oral bioavailability",
            "canonical_smiles": "CCO",
            "group_id": "Fg.test",
            "source_record_examples": [example],
        }
    )["examples"][0]

    assert visible["source_fields"]["pmid"] == "12345678"
    assert "pmid" not in visible


def test_minimal_evidence_rejects_source_fields_outside_the_boolean_allowlist():
    with pytest.raises(ValueError, match="exactly match"):
        evidence_for_llm(
            {
                "evidence_source": "Starling normalized oral bioavailability",
                "canonical_smiles": "CCO",
                "group_id": "Fg.test",
                "source_record_examples": [
                    {
                        "source_contract": {
                            "contract_version": SOURCE_COLUMN_CONTRACT_VERSION,
                            "source_id": "fg",
                            "source_or_simply_cleaned": {"measured_value": True},
                        },
                        "source_fields": {
                            "measured_value": "Fg = 0.5",
                            "canonical_measurement": "0.5",
                        },
                    }
                ],
            }
        )


def test_organized_example_uses_persisted_source_projection_without_alias_fallback():
    projection = llm_source_projection(_fg_record())
    record = {
        "llm_source_contract_json": json.dumps(
            {key: value for key, value in projection.items() if key != "source_fields"}
        ),
        "llm_source_fields_json": json.dumps(projection["source_fields"]),
        "endpoint_name": "canonical-looking alias",
    }

    assert _llm_example(record) == {
        "source_contract": {key: value for key, value in projection.items() if key != "source_fields"},
        "source_fields": projection["source_fields"],
    }


def test_assay_transfer_display_uses_source_projection_and_fails_closed_without_it():
    projection = llm_source_projection(_fg_record())
    selected = {
        "canonical_endpoint_key": "q3.canonical",
        "value": 2.5e-6,
        "unit_basis": "cm/s",
        "source_contract": {key: value for key, value in projection.items() if key != "source_fields"},
        "source_fields": projection["source_fields"],
    }
    rendered = _assay_transfer_evidence_record(
        selected,
        "Starling normalized oral bioavailability",
        {"group_id": "Fg.test", "tier": "Fg", "endpoint_group": "test"},
    )

    text = "\n".join(value for _, value in rendered)
    assert "P_app = 2.5 ×10^-6 cm/s" in text
    assert "q3.canonical" not in text
    with pytest.raises(ValueError, match="canonical display fallback is forbidden"):
        _assay_transfer_evidence_record(
            {"canonical_endpoint_key": "q3.canonical", "value": 2.5e-6},
            "Starling normalized oral bioavailability",
            {"group_id": "Fg.test", "tier": "Fg", "endpoint_group": "test"},
        )
