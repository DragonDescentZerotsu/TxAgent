from tools.chembl_tool.tasks.bioavailability_ma.reviewed_context_conditioned_benchmark import (
    propose_source_row,
    relative_effect_exclusion_reason,
)


def _row(value: str, condition: str = "fasted") -> dict:
    return {
        "canonical_claim_id": "claim-1",
        "smiles": "CCO",
        "bioavailability_report_type": "absolute",
        "oral_bioavailability_value": value,
        "species_or_population": "healthy human volunteers",
        "qualifying_conditions": condition,
        "support_text": "Absolute oral bioavailability was measured.",
    }


def test_absolute_human_f_can_enter_review_queue() -> None:
    audit, candidate = propose_source_row(0, _row("35%"))
    assert audit["queue_status"] == "queued"
    assert candidate is not None
    assert candidate["Y"] == 1
    assert candidate["bioavailability_report_type"] == "absolute"
    assert candidate["proposed_condition_group"] == "prandial_state=fasted"


def test_relative_value_is_discarded_before_review() -> None:
    for value in (
        "64% increase",
        "reduced by 22%",
        "decreased from 100 to 50%",
        "30 to 50% less than that of the conventional formulation",
    ):
        audit, candidate = propose_source_row(0, _row(value))
        assert candidate is None
        assert audit["queue_status"] == "not_queued"
        assert audit["proposal_reason"] == "relative_not_absolute_bioavailability"


def test_effect_statement_is_not_a_condition_arm() -> None:
    audit, candidate = propose_source_row(
        0, _row("35%", condition="improved with food intake")
    )
    assert candidate is None
    assert audit["proposal_reason"] == "comparison_or_effect_statement_not_condition_arm"


def test_prodrug_identity_is_discarded_before_review() -> None:
    audit, candidate = propose_source_row(
        0, _row("60%", condition="administered as lenampicillin prodrug")
    )
    assert candidate is None
    assert audit["proposal_reason"] == (
        "prodrug_or_active_moiety_identity_not_an_external_condition"
    )


def test_relative_effect_with_absolute_arm_value_is_hard_discarded() -> None:
    row = _row("7.0%")
    row["support_text"] = (
        "Rifampin decreased oral bioavailability from 14.4% to 7.0%."
    )
    audit, candidate = propose_source_row(0, row)
    assert candidate is None
    assert audit["proposal_reason"] == "relative_effect_record_not_absolute_condition_claim"


def test_iv_reference_for_absolute_f_is_not_relative_effect() -> None:
    row = _row("38.1%")
    row["support_text"] = (
        "Absolute oral bioavailability was 38.1% compared with an intravenous reference."
    )
    row["comparator"] = "intravenous reference dose"
    assert relative_effect_exclusion_reason(row) is None
    audit, candidate = propose_source_row(0, row)
    assert audit["queue_status"] == "queued"
    assert candidate is not None


def test_non_iv_comparison_is_hard_discarded() -> None:
    row = _row("57%", condition="cirrhosis")
    row["support_text"] = "Oral bioavailability was 57% compared with 37% in normals."
    audit, candidate = propose_source_row(0, row)
    assert candidate is None
    assert audit["proposal_reason"] == "non_iv_comparison_record_not_absolute_condition_claim"


def test_prodrug_or_active_metabolite_endpoint_is_hard_discarded() -> None:
    for support in (
        "Absolute bioavailability of GS4071 followed an oral dose of oseltamivir.",
        "The analyte followed oral administration of a pro-drug.",
        "Oral bioavailability was reported for the active metabolite.",
    ):
        row = _row("80%")
        row["support_text"] = support
        audit, candidate = propose_source_row(0, row)
        assert candidate is None
        assert audit["proposal_reason"] == "prodrug_or_indirect_analyte_not_direct_oral_f"


def test_simulated_f_is_hard_discarded() -> None:
    row = _row("70%")
    row["support_text"] = "The simulated absolute bioavailability was 70%."
    audit, candidate = propose_source_row(0, row)
    assert candidate is None
    assert audit["proposal_reason"] == "predicted_or_simulated_f_not_direct_measurement"


def test_non_iv_relative_reference_is_hard_discarded() -> None:
    row = _row("100%")
    row["support_text"] = "F was normalized relative to the highest oral dose."
    audit, candidate = propose_source_row(0, row)
    assert candidate is None
    assert audit["proposal_reason"] == "non_iv_relative_reference_not_absolute_condition_claim"
