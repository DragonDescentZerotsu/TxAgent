import json

from tools.chembl_tool.common.experiment_retrieval import (
    EvidenceGroupSpec,
    SourceExperimentConfig,
)
from tools.chembl_tool.common.task_workflows.evidence_library import (
    build_neighbor_index,
)
from tools.chembl_tool.paper_experiments.rl_lora.mixed_retrieval import (
    MIXED_IDENTITY_POLICY_VERSION,
    retrieve_full_flat_mixed_identity,
)
from tools.chembl_tool.paper_experiments.rl_lora.reward import score_response


METADATA = {
    "gold_label": 1,
    "prediction_field": "bioavailability_prediction",
    "negative_value": "low",
    "positive_value": "high",
    "required_fields": ["bioavailability_prediction", "confidence", "final_summary"],
}


def test_correct_full_schema_receives_both_bonuses():
    result = score_response(
        json.dumps(
            {
                "bioavailability_prediction": "high",
                "confidence": "moderate",
                "final_summary": "supported",
            }
        ),
        METADATA,
    )
    assert result.correct
    assert result.full_schema
    assert result.reward == 1.1


def test_wrong_but_valid_json_remains_negative():
    result = score_response(
        json.dumps(
            {
                "bioavailability_prediction": "low",
                "confidence": "moderate",
                "final_summary": "unsupported",
            }
        ),
        METADATA,
    )
    assert not result.correct
    assert result.reward == -0.9


def test_invalid_response_is_incorrect_without_bonuses():
    result = score_response("not json", METADATA)
    assert result.predicted_label is None
    assert not result.parsed_json
    assert result.reward == -1.0


def test_fenced_json_is_parsed_but_partial_schema_gets_one_bonus():
    result = score_response(
        'answer follows\n```json\n{"bioavailability_prediction":"high"}\n```',
        METADATA,
    )
    assert result.correct
    assert result.parsed_json
    assert not result.full_schema
    assert result.reward == 1.05


def _evidence_row(molecule_id, smiles, group_id, value):
    tier, endpoint = group_id.split(".", 1)
    return {
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": smiles,
        "group_id": group_id,
        "assay_tier": tier,
        "endpoint_group": endpoint,
        "standard_type": endpoint,
        "standard_value": value,
        "standard_units": "%",
    }


def test_mixed_full_flat_filters_same_scaffold_only_from_direct_family():
    rows = [
        _evidence_row("direct_same", "CCc1ccccc1", "Tier 1.direct", 1),
        _evidence_row("direct_other", "c1ccncc1", "Tier 1.direct", 2),
        _evidence_row("mechanism_same", "CCCc1ccccc1", "Tier 2.factor", 3),
    ]
    index = build_neighbor_index(rows, index_version="test.mixed.v1")
    config = SourceExperimentConfig(
        source_name="test",
        direct_groups=(
            EvidenceGroupSpec(
                "Direct.outcome",
                "Direct",
                "outcome",
                source_groups=("Tier 1.direct",),
            ),
        ),
        mechanism_groups=(
            EvidenceGroupSpec(
                "Mechanism.direct",
                "Tier 1",
                "direct",
                source_groups=("Tier 1.direct",),
            ),
            EvidenceGroupSpec(
                "Mechanism.factor",
                "Tier 2",
                "factor",
                source_groups=("Tier 2.factor",),
            ),
        ),
    )

    result = retrieve_full_flat_mixed_identity(
        "Cc1ccccc1",
        index,
        config=config,
        top_k_per_group=2,
        min_similarity=0.0,
    )

    neighbors = {
        row["molecule_chembl_id"]: row for row in result["groups"][0]["neighbors"]
    }
    assert "direct_same" not in neighbors
    assert neighbors["direct_other"]["source_group_ids"] == ["Tier 1.direct"]
    assert neighbors["mechanism_same"]["source_group_ids"] == ["Tier 2.factor"]
    family_policy = result["experiment"]["family_identity_policy"]
    assert family_policy["version"] == MIXED_IDENTITY_POLICY_VERSION
    assert family_policy["group_policies"] == {
        "Mechanism.direct": "scaffold_disjoint",
        "Mechanism.factor": "parent_disjoint",
    }
