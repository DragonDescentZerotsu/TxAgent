from tools.chembl_tool.tasks.dili.endpoint_groups import assign_endpoint_group


def test_tier1_hys_law_maps_to_human_liver_lab_signal():
    assignment = assign_endpoint_group(
        {
            "assay_tier": "Tier 1",
            "standard_type": "ALT bilirubin Hy's law",
            "assay_description": "Human clinical DILI laboratory signal",
        }
    )

    assert assignment.group_id == "Tier 1.human_liver_laboratory_signal"
    assert assignment.evidence_direction == "clinical_dili_signal"
    assert assignment.evidence_strength == "strong"


def test_tier2_histopathology_maps_to_in_vivo_liver_histopathology():
    assignment = assign_endpoint_group(
        {
            "assay_tier": "Tier 2",
            "standard_type": "Histopathology",
            "assay_description": "Rat repeated dose liver necrosis",
        }
    )

    assert assignment.group_id == "Tier 2.in_vivo_liver_histopathology"
    assert assignment.evidence_direction == "in_vivo_liver_injury_signal"
    assert assignment.evidence_strength == "strong"


def test_tier2_liver_weight_only_is_moderate_not_strong():
    assignment = assign_endpoint_group(
        {
            "assay_tier": "Tier 2",
            "standard_type": "Liver weight",
            "assay_description": "Effect of compound on relative liver weight in normal chow-fed male rats",
        }
    )

    assert assignment.group_id == "Tier 2.in_vivo_liver_histopathology"
    assert assignment.evidence_direction == "in_vivo_liver_injury_signal"
    assert assignment.evidence_strength == "moderate"


def test_tier3_bsep_gene_maps_to_bsep_efflux():
    assignment = assign_endpoint_group(
        {
            "assay_tier": "Tier 3",
            "standard_type": "IC50",
            "target_genes": ["ABCB11"],
            "assay_description": "BSEP taurocholate efflux inhibition",
        }
    )

    assert assignment.group_id == "Tier 3.bsep_or_bile_acid_efflux"
    assert assignment.evidence_direction == "cholestasis_or_bile_acid_transport_risk"


def test_tier4_ocr_maps_to_mitochondrial_function():
    assignment = assign_endpoint_group(
        {
            "assay_tier": "Tier 4",
            "standard_type": "OCR",
            "assay_description": "Oxygen consumption rate in hepatocytes",
        }
    )

    assert assignment.group_id == "Tier 4.mitochondrial_function_or_respiration"
    assert assignment.evidence_direction == "mitochondrial_or_organelle_stress_risk"


def test_tier4_gsh_content_is_oxidative_stress_but_weak_unclear():
    assignment = assign_endpoint_group(
        {
            "assay_tier": "Tier 4",
            "standard_type": "GSH",
            "assay_description": "In vitro increased GSH content of rat hepatocytes when incubated with compound",
        }
    )

    assert assignment.group_id == "Tier 4.energy_failure_and_oxidative_stress"
    assert assignment.evidence_direction == "neutral_or_unclear"
    assert assignment.evidence_strength == "weak"


def test_tier4_gsh_depletion_remains_organelle_stress_risk():
    assignment = assign_endpoint_group(
        {
            "assay_tier": "Tier 4",
            "standard_type": "GSH depletion",
            "assay_description": "Glutathione depletion and oxidative stress in hepatocytes",
        }
    )

    assert assignment.group_id == "Tier 4.energy_failure_and_oxidative_stress"
    assert assignment.evidence_direction == "mitochondrial_or_organelle_stress_risk"
    assert assignment.evidence_strength == "moderate"


def test_tier5_gsh_adduct_maps_to_reactive_metabolite():
    assignment = assign_endpoint_group(
        {
            "assay_tier": "Tier 5",
            "standard_type": "GSH adduct",
            "assay_description": "Reactive metabolite trapping in liver microsomes",
        }
    )

    assert assignment.group_id == "Tier 5.reactive_metabolite_or_covalent_binding"
    assert assignment.evidence_direction == "reactive_metabolite_or_bioactivation_risk"


def test_tier6_heparg_viability_maps_to_hepatic_cell_injury():
    assignment = assign_endpoint_group(
        {
            "assay_tier": "Tier 6",
            "standard_type": "Cell viability",
            "assay_description": "HepaRG hepatocyte cytotoxicity",
        }
    )

    assert assignment.group_id == "Tier 6.hepatocyte_or_hepatic_cell_injury"
    assert assignment.evidence_direction == "hepatic_cell_injury_risk"


def test_negative_activity_comment_reverses_specific_group_only():
    assignment = assign_endpoint_group(
        {
            "assay_tier": "Tier 3",
            "standard_type": "IC50",
            "target_genes": ["ABCB11"],
            "activity_comment": "inactive",
        }
    )

    assert assignment.group_id == "Tier 3.bsep_or_bile_acid_efflux"
    assert assignment.evidence_direction == "argues_against_dili_risk"
    assert assignment.evidence_strength == "weak"


def test_asbt_ileal_context_stays_context_dependent():
    assignment = assign_endpoint_group(
        {
            "assay_tier": "Tier 3",
            "standard_type": "IC50",
            "target_genes": ["SLC10A2"],
            "assay_description": "ASBT taurocholate uptake in hamster ileal ring",
        }
    )

    assert assignment.group_id == "Tier 3.context_dependent"
    assert assignment.evidence_direction == "context_dependent"
