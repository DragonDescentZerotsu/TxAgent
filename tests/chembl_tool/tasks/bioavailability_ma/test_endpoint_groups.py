from tools.chembl_tool.tasks.bioavailability_ma.endpoint_groups import assign_endpoint_group


def test_tier1_bioavailability_maps_to_direct_absolute():
    assignment = assign_endpoint_group({"assay_tier": "Tier 1", "standard_type": "Bioavailability"})

    assert assignment.group_id == "Tier 1.direct_absolute_bioavailability"
    assert assignment.evidence_strength == "strong"


def test_tier1_direct_f_uses_20_percent_cutoff_for_direction():
    high = assign_endpoint_group(
        {
            "assay_tier": "Tier 1",
            "standard_type": "F",
            "standard_relation": "=",
            "standard_value": "20",
            "assay_description": "Oral bioavailability in rat",
        }
    )
    low = assign_endpoint_group(
        {
            "assay_tier": "Tier 1",
            "standard_type": "F",
            "standard_relation": "=",
            "standard_value": "19.9",
            "assay_description": "Oral bioavailability in rat",
        }
    )

    assert high.evidence_direction == "supports_high_bioavailability"
    assert low.evidence_direction == "argues_against_high_bioavailability"


def test_tier2_oral_auc_maps_to_oral_exposure():
    assignment = assign_endpoint_group(
        {
            "assay_tier": "Tier 2",
            "standard_type": "AUC",
            "assay_description": "Plasma exposure after oral dose",
        }
    )

    assert assignment.group_id == "Tier 2.oral_auc_exposure"


def test_tier3_caco2_papp_maps_to_cell_permeability():
    assignment = assign_endpoint_group(
        {
            "assay_tier": "Tier 3",
            "standard_type": "Papp",
            "assay_description": "Caco-2 apparent permeability apical to basolateral",
        }
    )

    assert assignment.group_id == "Tier 3.cell_permeability_papp"
    assert assignment.evidence_direction == "permeability_support"


def test_tier4_solubility_maps_to_solubility():
    assignment = assign_endpoint_group({"assay_tier": "Tier 4", "standard_type": "Solubility"})

    assert assignment.group_id == "Tier 4.solubility"


def test_tier5_clint_maps_to_clearance():
    assignment = assign_endpoint_group({"assay_tier": "Tier 5", "standard_type": "CLint"})

    assert assignment.group_id == "Tier 5.intrinsic_or_hepatic_clearance"


def test_tier6_food_effect_maps_to_food_effect():
    assignment = assign_endpoint_group(
        {
            "assay_tier": "Tier 6",
            "standard_type": "AUC ratio",
            "assay_description": "Food effect fed/fasted relative bioavailability study",
        }
    )

    assert assignment.group_id == "Tier 6.food_effect_or_fed_fasted"
    assert assignment.evidence_strength == "weak"
