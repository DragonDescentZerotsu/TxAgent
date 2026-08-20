from tools.chembl_tool.tasks.bioavailability_ma.analyze_qualifying_conditions import (
    classify_condition,
    normalize_condition,
)


def test_normalize_condition_collapses_only_cosmetic_variation():
    assert normalize_condition("  Fasted   State. ") == "fasted state"


def test_condition_taxonomy_keeps_multiple_relevant_modifiers():
    primary, groups = classify_condition(
        "Ketoconazole coadministration after a high-fat meal in poor metabolizers"
    )

    assert primary == "co_treatment_or_ddi"
    assert groups == [
        "co_treatment_or_ddi",
        "food_or_prandial_state",
        "genotype_or_metabolizer_phenotype",
    ]


def test_condition_taxonomy_separates_mechanism_from_experimental_context():
    assert classify_condition("extensive hepatic first-pass metabolism")[0] == (
        "first_pass_metabolism_or_clearance"
    )
    assert classify_condition("poor aqueous solubility")[0] == (
        "absorption_solubility_or_permeability"
    )
