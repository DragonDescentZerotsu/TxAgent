from data.processing.evidence_library.versions.v7.tasks.bbb_martins.experimental_metric_direction_review import (
    adjudicate_missing_direction,
    review_missing_direction,
)


def test_recovers_explicit_qualitative_measurement_values() -> None:
    assert review_missing_direction({"quant_value": "not detected"}).label == 0
    assert review_missing_direction({"quant_value": "very high"}).label == 1


def test_recovers_endpoint_specific_result_language() -> None:
    negative = review_missing_direction(
        {"support_text": "Brain exposure was very low after intravenous dosing."}
    )
    positive = review_missing_direction(
        {"support_text": "The compound readily crossed the BBB after oral dosing."}
    )
    assert negative.label == 0
    assert negative.rule_id == "explicit_negative_outcome"
    assert positive.label == 1
    assert positive.rule_id == "explicit_positive_modified_entry"


def test_does_not_label_detectability_or_uninterpreted_numbers() -> None:
    assert review_missing_direction(
        {"support_text": "The compound was detected in brain after dosing."}
    ).label is None
    assert review_missing_direction(
        {"quant_metric": "brain concentration", "quant_value": "51 ng/g"}
    ).label is None


def test_conflicting_source_language_remains_unresolved() -> None:
    review = review_missing_direction(
        {
            "support_text": (
                "Brain exposure was high at 1 hour but brain penetration was low "
                "at 24 hours."
            )
        }
    )
    assert review.label is None
    assert review.rule_id == "qualified_or_conflicting_direction"


def test_comparison_and_decimal_do_not_reverse_direction() -> None:
    comparison = review_missing_direction(
        {
            "support_text": (
                "CNS penetration was low compared to agents with good BBB penetration."
            )
        }
    )
    decimal = review_missing_direction(
        {
            "support_text": (
                "P values were too low between pH 6.5 and 7 to freely cross the BBB."
            )
        }
    )
    assert comparison.label is None
    assert decimal.label is None


def test_frozen_source_index_adjudication_gates_proposed_labels() -> None:
    approved = adjudicate_missing_direction(
        {"quant_value": "very low"}, source_index=5820
    )
    excluded = adjudicate_missing_direction(
        {"quant_value": "very low"}, source_index=15600
    )
    unreviewed = adjudicate_missing_direction(
        {"quant_value": "very low"}, source_index=999999
    )
    assert approved.label == 0
    assert approved.rule_id.startswith("manual_review_approved:")
    assert excluded.label is None
    assert excluded.rule_id.startswith("manual_review_exclusion:")
    assert unreviewed.label is None
    assert unreviewed.rule_id == "unreviewed_proposed_direction"
