from __future__ import annotations

import json

import pytest

from tools.chembl_tool.common.identity_blind import expose_neighbor_smiles_only
from tools.chembl_tool.common.task_workflows.analogous_flat_prompt import (
    PROMPT_IDENTITY_VIEW,
    branch_validation,
    build_final_messages,
    build_group_messages,
    final_output_schema,
    prompt_provenance,
)


def _group() -> dict:
    return {
        "group_id": "Flat.all_evidence",
        "tier": "Flat",
        "endpoint_group": "all_evidence",
        "identity_blind": True,
        "neighbors": [
            {
                "rank": 1,
                "molecule_chembl_id": "neighbor_1_1",
                "canonical_smiles": "CCOO",
                "transfer_selection_score": 0.913,
                "transfer_selected_records": [
                    {
                        "record_rank": 1,
                        "transfer_selection_score": 0.913,
                        "transfer_winning_record": {
                            "source_fields": {
                                "pmid": "123",
                                "paragraph_idx": 4,
                                "global_identifier": "named compound",
                                "extraction_id": "row-1",
                                "smiles": "CCOO",
                                "confidence": "high",
                                "endpoint_name": "oral bioavailability",
                                "measurement_text": "35",
                                "unit_text": "%",
                                "species_or_population": "rat",
                                "support_text": "Absolute oral F was 35%.",
                            }
                        },
                    },
                    {
                        "record_rank": 2,
                        "transfer_selection_score": 0.812,
                        "transfer_winning_record": {
                            "source_fields": {
                                "assay_system": "in vivo",
                                "reported_value": "31%",
                            }
                        },
                    },
                ],
                "evidence_rows": [],
            }
        ],
    }


@pytest.mark.parametrize(
    ("task_id", "prediction_field"),
    [
        ("bbb_martins", "bbb_prediction"),
        ("bioavailability_ma", "bioavailability_prediction"),
        ("skin_reaction", "skin_reaction_prediction"),
    ],
)
def test_prompt_is_minimal_smiles_first_and_task_specific(task_id, prediction_field):
    messages = build_group_messages(_group(), task_id=task_id)
    rendered = messages[1]["content"]

    assert "Molecule 1: CCOO" in rendered
    assert "Assay-transfer likelihood: 0.91" in rendered
    assert "Assay-transfer likelihood: 0.81" in rendered
    assert "Endpoint: oral bioavailability" in rendered
    assert "Evidence text: Absolute oral F was 35%." in rendered
    assert "PMID" not in rendered
    assert "paragraph" not in rendered.lower()
    assert "global_identifier" not in rendered
    assert "named compound" not in rendered
    assert "row-1" not in rendered
    assert "molecule_chembl_id" not in rendered
    assert "tool_summary" not in rendered
    assert "mmp_structure_compare" not in rendered
    assert "properties_compare" not in rendered
    assert prediction_field in json.dumps(final_output_schema(task_id))


def test_morgan_prompt_shows_similarity_and_minimal_record_evidence():
    group = _group()
    neighbor = group["neighbors"][0]
    neighbor.pop("transfer_selection_score")
    neighbor.pop("transfer_selected_records")
    neighbor["similarity"] = 0.734
    neighbor["evidence_rows"] = [
        {
            "minimal_evidence": {
                "contract_version": "minimal_evidence.v1",
                "source": {"name": "Starling", "record_id": "private-row"},
                "molecule": {
                    "id": "CHEMBL1",
                    "canonical_smiles": "CCOO",
                    "names": ["private name"],
                },
                "group": {
                    "id": "Observed.direct_oral_bioavailability",
                    "tier": "Observed",
                    "endpoint_group": "direct_oral_bioavailability",
                },
                "endpoint": {
                    "name": "oral bioavailability",
                    "measurement": {"relation": "=", "value": 35, "unit": "%"},
                },
                "text": {"evidence": "Absolute oral F was 35%.", "context": "rat"},
                "annotations": {
                    "evidence_role": "direct_outcome",
                    "scope": {},
                    "transferability": "not_assessed",
                    "uncertainty": [],
                },
                "quality": {"confidence": 0.9},
                "provenance": {"assay_id": "private-assay"},
                "examples": [],
            }
        }
    ]

    rendered = build_group_messages(group, task_id="bioavailability_ma")[1]["content"]

    assert "Molecule 1: CCOO" in rendered
    assert "Morgan Tanimoto similarity: 0.73" in rendered
    assert "Assay-transfer likelihood" not in rendered
    assert "Endpoint: oral bioavailability" in rendered
    assert "Measurement: 35" in rendered
    assert "Evidence text: Absolute oral F was 35%." in rendered
    assert "private-row" not in rendered
    assert "private name" not in rendered
    assert "private-assay" not in rendered


def test_branch_schema_uses_only_molecule_references():
    validation = branch_validation()
    assert "key_evidence" in validation["required_fields"]
    assert "molecule_chembl_id" in validation["forbidden_field_names"]


def test_final_prompt_uses_only_flat_analysis():
    messages = build_final_messages(
        [
            {
                "group_id": "Flat.all_evidence",
                "status": "ok",
                "llm": {
                    "content": {
                        "evidence_direction": "supports_positive",
                        "confidence": "moderate",
                    },
                    "structured_output_validation": {"valid": True},
                },
            }
        ],
        task_id="bioavailability_ma",
    )
    rendered = messages[1]["content"]
    assert "bioavailability_prediction='high'" in rendered
    assert "supports_positive" in rendered
    assert "single_molecule" not in rendered


def test_neighbor_smiles_view_preserves_blind_query_and_redacted_evidence():
    source = {
        "query": {"canonical_smiles": "QUERY"},
        "groups": [
            {
                "group_id": "Flat.all_evidence",
                "neighbors": [
                    {
                        "canonical_smiles": "CCOO",
                        "evidence_rows": [{"molecule_chembl_id": "CHEMBL1"}],
                    }
                ],
            }
        ],
    }
    blind = {
        "query": {"molecule_id": "query", "identity_hidden": True},
        "groups": [
            {
                "group_id": "Flat.all_evidence",
                "neighbors": [
                    {
                        "molecule_chembl_id": "neighbor_1_1",
                        "canonical_smiles": "[hidden]",
                        "evidence_rows": [{"minimal_evidence": {"molecule": {"id": "neighbor_1_1"}}}],
                    }
                ],
            }
        ],
    }
    visible = expose_neighbor_smiles_only(blind, source)
    assert visible["query"] == {"molecule_id": "query", "identity_hidden": True}
    assert visible["groups"][0]["neighbors"][0]["canonical_smiles"] == "CCOO"
    assert "CHEMBL1" not in json.dumps(visible)
    assert visible["experiment"]["prompt_identity_view"] == PROMPT_IDENTITY_VIEW


def test_prompt_provenance_is_versioned_and_task_bound():
    provenance = prompt_provenance("skin_reaction")
    assert provenance["prompt_version"] == "analogous_flat_v1"
    assert provenance["identity_view"] == PROMPT_IDENTITY_VIEW
    assert provenance["task_id"] == "skin_reaction"
    assert len(provenance["contract_sha256"]) == 64
