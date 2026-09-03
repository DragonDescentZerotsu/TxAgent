import json

from tools.chembl_tool.tasks.clintox.experiment_config import STARLING
from tools.chembl_tool.tasks.clintox.run_reasoning_pipeline import (
    _group_prompt_payload,
    _group_provenance_validation_errors,
)
from tools.chembl_tool.tasks.clintox.starling_retrieval import (
    CLINICAL_CONTEXT_GROUP,
    DIRECT_GROUP,
    is_clinical_context_record,
    load_historical_mechanism_catalog,
    trial_failure_gate_reason,
)


def _row(outcome: str, **overrides):
    row = {
        "SMILES": "CCO",
        "pmid": "123",
        "support_text": "Source sentence.",
        "toxicity_outcome": outcome,
        "needs_more_context": False,
    }
    row.update(overrides)
    return row


def test_direct_gate_requires_literal_toxicity_caused_trial_or_development_failure():
    assert trial_failure_gate_reason(
        _row("The clinical trial was terminated due to severe hepatotoxicity.")
    ) == "accept_literal_toxicity_trial_failure"
    assert trial_failure_gate_reason(
        _row("Further development was halted owing to safety concerns.")
    ) == "accept_literal_toxicity_trial_failure"
    assert trial_failure_gate_reason(
        _row("Two clinical trials were terminated due to liver toxicity.")
    ) == "accept_literal_toxicity_trial_failure"
    assert trial_failure_gate_reason(
        _row(
            "Severe side effects led to abandonment of further drug development."
        )
    ) == "accept_literal_toxicity_trial_failure"


def test_direct_gate_rejects_patient_level_efficacy_and_negated_events():
    assert trial_failure_gate_reason(
        _row("Ten patients discontinued treatment due to adverse events.")
    ) == "reject_patient_or_treatment_discontinuation"
    assert trial_failure_gate_reason(
        _row("The trial failed to demonstrate efficacy despite observed toxicity.")
    ) == "reject_efficacy_failure_not_toxicity_failure"
    assert trial_failure_gate_reason(
        _row("The study was not terminated due to toxicity.")
    ) == "reject_negated_failure_event"
    assert trial_failure_gate_reason(
        _row("Study discontinuation due to toxicity occurred in 12% of patients.")
    ) == "reject_no_literal_toxicity_trial_failure"
    assert trial_failure_gate_reason(
        _row(
            "The phase III trial was discontinued; the reason for discontinuation "
            "was not attributed to toxicity."
        )
    ) == "reject_failure_explicitly_not_toxicity_caused"
    assert trial_failure_gate_reason(
        _row("The trial terminated before MTD due to drug supply rather than toxicity.")
    ) == "reject_failure_explicitly_not_toxicity_caused"
    assert trial_failure_gate_reason(
        _row("The open-label safety study terminated treatment owing to adverse events.")
    ) == "reject_patient_or_treatment_discontinuation"
    assert trial_failure_gate_reason(
        _row("The trial was terminated due to toxicity.", needs_more_context=True)
    ) == "reject_needs_more_context"


def test_complete_non_direct_human_rows_remain_context_only():
    row = _row("Dose reductions were required after dose-limiting toxicity.")
    assert is_clinical_context_record(row)
    assert not is_clinical_context_record(
        _row("The trial was stopped because of dose-limiting toxicity.")
    )


def test_historical_human_organ_rows_are_demoted_to_clinical_context(tmp_path):
    source = tmp_path / "evidence.jsonl"
    source.write_text(
        json.dumps(
            {
                "group_id": "Direct.human_organ_toxicity",
                "assay_tier": "Direct",
                "endpoint_group": "human_organ_toxicity",
                "evidence_role": "direct_outcome",
                "uncertainty": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    row = load_historical_mechanism_catalog(source)[0]

    assert row["group_id"] == CLINICAL_CONTEXT_GROUP
    assert row["assay_tier"] == "Mechanism"
    assert row["evidence_role"] == "context_modifier"
    assert "not_strict_clinical_trial_failure_evidence" in row["uncertainty"]


def test_starling_views_preserve_direct_as_its_own_full_reasoning_branch():
    assert tuple(group.group_id for group in STARLING.direct_groups) == (DIRECT_GROUP,)
    assert len(STARLING.mechanism_groups) == 8
    assert STARLING.mechanism_groups[0].group_id == DIRECT_GROUP
    clinical = STARLING.mechanism_groups[1]
    assert clinical.group_id == "Clinical.clinical_human_safety"
    assert clinical.source_groups == (CLINICAL_CONTEXT_GROUP,)


def test_non_direct_group_can_predict_risk_but_cannot_claim_direct_outcome():
    payload = _group_prompt_payload(
        {"input_smiles": "CCO", "canonical_smiles": "CCO"},
        {
            "group_id": "Mechanism.organ_specific_toxicity",
            "tier": "Mechanism",
            "endpoint_group": "organ_specific_toxicity",
            "neighbors": [],
        },
    )
    directions = payload["required_json_schema"]["evidence_direction"]
    assert "supports_toxicity_trial_failure" not in directions
    assert "supports_higher_trial_failure_risk" in directions
    assert "direct_evidence_status" in payload["required_json_schema"]
    assert any("must never be described as an observed" in item for item in payload["instructions"])


def test_direct_group_uses_same_predictive_schema_and_separate_provenance_status():
    payload = _group_prompt_payload(
        {"input_smiles": "CCO", "canonical_smiles": "CCO"},
        {
            "group_id": DIRECT_GROUP,
            "tier": "Direct",
            "endpoint_group": "clinical_trial_failure",
            "neighbors": [],
        },
    )
    directions = payload["required_json_schema"]["evidence_direction"]
    assert "supports_higher_trial_failure_risk" in directions
    assert "supports_toxicity_trial_failure" not in directions
    assert "direct_evidence_status" in payload["required_json_schema"]


def test_flat_group_schema_does_not_become_direct_capable_when_a_direct_row_is_added():
    group = {
        "group_id": "Flat.all_evidence",
        "tier": "Flat",
        "endpoint_group": "all_evidence",
        "neighbors": [
            {
                "rank": 1,
                "molecule_chembl_id": "analog",
                "canonical_smiles": "CCN",
                "similarity": 0.8,
                "similarity_bucket": "close_analog",
                "evidence_rows": [
                    {
                        "group_id": "Mechanism.organ_specific_toxicity",
                        "minimal_evidence": {
                            "group": {"id": "Mechanism.organ_specific_toxicity"}
                        },
                    }
                ],
            }
        ],
    }
    payload = _group_prompt_payload(
        {"input_smiles": "CCO", "canonical_smiles": "CCO"}, group
    )
    assert "supports_toxicity_trial_failure" not in payload["required_json_schema"][
        "evidence_direction"
    ]
    directions_without_direct = payload["required_json_schema"]["evidence_direction"]

    group["neighbors"][0]["evidence_rows"].append(
        {
            "group_id": DIRECT_GROUP,
            "minimal_evidence": {"group": {"id": DIRECT_GROUP}},
        }
    )
    payload = _group_prompt_payload(
        {"input_smiles": "CCO", "canonical_smiles": "CCO"}, group
    )
    assert payload["required_json_schema"]["evidence_direction"] == directions_without_direct
    assert "supports_toxicity_trial_failure" not in payload["required_json_schema"][
        "evidence_direction"
    ]


def test_group_provenance_validator_does_not_constrain_predictive_direction():
    group = {
        "group_id": "Mechanism.organ_specific_toxicity",
        "neighbors": [
            {
                "evidence_rows": [
                    {
                        "minimal_evidence": {
                            "group": {"id": "Mechanism.organ_specific_toxicity"}
                        }
                    }
                ]
            }
        ],
    }
    assert _group_provenance_validation_errors(
        {
            "direct_evidence_status": "no_direct_rows",
            "evidence_direction": "supports_higher_trial_failure_risk",
        },
        group,
    ) == []
    assert _group_provenance_validation_errors(
        {
            "direct_evidence_status": "direct_rows_present_transferable",
            "evidence_direction": "supports_higher_trial_failure_risk",
        },
        group,
    )
