from tools.chembl_tool.tasks.bioavailability_ma.condition_ontology import (
    NO_REPORTED_CONDITION,
    classify_external_condition,
)


def test_fasting_synonyms_share_one_group() -> None:
    values = [
        "fasted",
        "fasting",
        "fasted state",
        "fasting state",
        "fasting conditions",
        "overnight fast",
        "on an empty stomach",
    ]
    assert {
        classify_external_condition(value).signature for value in values
    } == {"prandial_state=fasted"}


def test_specific_diseases_do_not_share_broad_disease_group() -> None:
    assert classify_external_condition("liver cirrhosis").signature == "disease=cirrhosis"
    assert (
        classify_external_condition("cystic fibrosis").signature
        == "disease=cystic_fibrosis"
    )
    assert (
        classify_external_condition("chronic renal failure").signature
        == "disease=renal_impairment"
    )


def test_composite_condition_is_not_downgraded_to_fasted() -> None:
    classified = classify_external_condition(
        "fasting conditions, concomitantly with omeprazole"
    )
    assert classified.scope == "external"
    assert classified.signature == (
        "co_treatment=omeprazole+prandial_state=fasted"
    )


def test_mechanisms_do_not_become_condition_groups() -> None:
    for value in ("extensive first-pass metabolism", "poor solubility"):
        classified = classify_external_condition(value)
        assert classified.scope == "mechanism_only"
        assert classified.signature is None


def test_comparison_statement_is_rejected() -> None:
    for value in (
        "not affected by food",
        "improved with food intake",
        "increases when taken with food",
        "reduced if taken on an empty stomach",
        "decreased absorption when not administered with food",
    ):
        classified = classify_external_condition(value)
        assert classified.scope == "excluded"
        assert classified.signature is None


def test_negated_steady_state_is_not_a_steady_state_arm() -> None:
    classified = classify_external_condition("not truly at steady state")
    assert classified.scope == "excluded"


def test_pretransplant_is_not_collapsed_into_transplant_recipient() -> None:
    classified = classify_external_condition("before renal transplantation")
    assert classified.signature == "transplant_status=pre_kidney_transplant"


def test_nonpregnant_does_not_also_emit_pregnancy() -> None:
    classified = classify_external_condition("non-pregnant")
    assert classified.signature == "physiologic_state=nonpregnant"


def test_additional_demographic_context_remains_composite() -> None:
    classified = classify_external_condition("young, white women")
    assert classified.signature == "age_group=young_adult+race=white+sex=female"


def test_null_has_explicit_no_reported_group() -> None:
    classified = classify_external_condition(None)
    assert classified.scope == "none_reported"
    assert classified.signature == NO_REPORTED_CONDITION
