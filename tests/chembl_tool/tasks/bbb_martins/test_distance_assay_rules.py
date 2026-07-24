from tools.chembl_tool.tasks.bbb_martins.distance_assay_rules import (
    assign_distance_endpoint_group,
    classify_base_measured_states,
    classify_distance_assay,
)


def base_row(**overrides):
    row = {
        "description": "",
        "assay_cell_type": "",
        "assay_tissue": "",
        "assay_organism": "",
        "target_chembl_id": "",
        "target_pref_name": "",
        "target_genes": [],
        "target_synonyms": [],
        "standard_types": ["Activity"],
        "confidence_score": 9,
        "relationship_type": "D",
        "n_unique_molecules": 3,
    }
    row.update(overrides)
    return row


def test_teer_in_bbb_model_is_base_state_overlap():
    decision = classify_distance_assay(
        base_row(description="Increase in transendothelial electrical resistance in hCMEC/D3 cells")
    )

    assert decision.status == "base_state_overlap"
    assert decision.family_id == "tight_junction_integrity"
    assert decision.distance_level == "B"
    assert decision.scope_match == "matched"


def test_teer_in_mdck_is_transportable_base_state_overlap():
    decision = classify_distance_assay(
        base_row(description="Decrease in TEER in a polarized MDCK-II barrier model", confidence_score=0)
    )

    assert decision.status == "base_state_overlap"
    assert decision.scope_match == "transportable"


def test_explicitly_insoluble_teer_assay_is_excluded():
    decision = classify_distance_assay(
        base_row(description="Decrease in TEER in Caco-2 cells; insoluble")
    )

    assert decision.status == "quality_fail"
    assert decision.quality_status == "explicit_assay_quality_failure"


def test_unrelated_tissue_junction_readout_is_out_of_scope():
    decision = classify_distance_assay(
        base_row(description="Increase in occludin expression in mouse colon")
    )

    assert decision.status == "out_of_scope"
    assert decision.family_id == "tight_junction_integrity"


def test_teer_abbreviation_does_not_match_healthy_volunteer():
    decision = classify_distance_assay(
        base_row(description="Cmax in healthy volunteers after a single oral dose")
    )

    assert decision.status == "unmatched"


def test_functional_pxr_agonism_is_base_state_overlap():
    decision = classify_distance_assay(
        base_row(
            target_chembl_id="CHEMBL3401",
            target_pref_name="Nuclear receptor subfamily 1 group I member 2",
            description="Agonist activity at human PXR measured by luciferase reporter assay",
            standard_types=["EC50"],
        )
    )

    assert decision.status == "base_state_overlap"
    assert decision.family_id == "pxr_car_activation"
    assert decision.distance_level == "B"


def test_pxr_binding_only_is_excluded():
    decision = classify_distance_assay(
        base_row(
            target_chembl_id="CHEMBL3401",
            description="Binding affinity towards human PXR by TR-FRET",
            standard_types=["AC50"],
        )
    )

    assert decision.status == "quality_fail"
    assert decision.quality_status == "not_functional_activation"


def test_direct_mmp9_inhibition_is_h1():
    decision = classify_distance_assay(
        base_row(
            target_chembl_id="CHEMBL321",
            target_pref_name="Matrix metalloproteinase-9",
            description="Inhibition of human recombinant MMP9 enzyme activity",
            standard_types=["IC50"],
        )
    )

    assert decision.status == "include"
    assert decision.family_id == "mmp9_activity"
    assert decision.distance_level == "H1"
    assert decision.tree_node_id == "Distance.h1.passive_permeability"
    assert decision.source_group_id == "Distance H1.passive_permeability"


def test_mmp9_binding_only_is_excluded():
    decision = classify_distance_assay(
        base_row(
            target_chembl_id="CHEMBL321",
            description="Binding affinity towards matrix metalloproteinase-9",
            standard_types=["Ki"],
        )
    )

    assert decision.status == "quality_fail"
    assert decision.quality_status == "not_functional_mmp9_activity"


def test_mmp9_dissociation_constant_is_binding_only():
    decision = classify_distance_assay(
        base_row(
            target_chembl_id="CHEMBL321",
            description="Inhibition of MMP9 assessed as dissociation constant",
            standard_types=["IC50"],
        )
    )

    assert decision.status == "quality_fail"
    assert decision.quality_status == "not_functional_mmp9_activity"


def test_mmp9_target_with_only_downstream_cell_invasion_readout_is_excluded():
    decision = classify_distance_assay(
        base_row(
            target_chembl_id="CHEMBL321",
            description="Inhibition of proMMP9 activation assessed as decrease in cell invasion",
            standard_types=["Inhibition"],
        )
    )

    assert decision.status == "quality_fail"
    assert decision.quality_status == "indirect_downstream_readout"


def test_direct_mmp3_inhibition_is_h1_because_it_has_a_junction_shortcut():
    decision = classify_distance_assay(
        base_row(
            target_chembl_id="CHEMBL283",
            target_pref_name="Stromelysin-1",
            description="Inhibition of human recombinant MMP-3 enzyme activity",
            standard_types=["IC50"],
        )
    )

    assert decision.status == "include"
    assert decision.family_id == "mmp3_activity"
    assert decision.distance_level == "H1"
    assert decision.tree_node_id == "Distance.h1.passive_permeability"
    assert decision.source_group_id == "Distance H1.passive_permeability"


def test_mmp3_binding_only_is_excluded():
    decision = classify_distance_assay(
        base_row(
            target_chembl_id="CHEMBL283",
            description="Binding affinity towards human MMP-3",
            standard_types=["Ki"],
        )
    )

    assert decision.status == "quality_fail"
    assert decision.quality_status == "not_functional_mmp3_activity"


def test_nrf2_are_reporter_is_efflux_h1():
    decision = classify_distance_assay(
        base_row(
            target_chembl_id="CHEMBL1075094",
            description="Activation of Nrf2 by ARE-driven luciferase reporter gene assay",
            standard_types=["EC50"],
        )
    )

    assert decision.status == "include"
    assert decision.family_id == "nrf2_activation"
    assert decision.tree_node_id == "Distance.h1.efflux_transport"
    assert decision.distance_level == "H1"


def test_nrf2_downstream_nqo1_enzyme_marker_without_reporter_is_excluded():
    decision = classify_distance_assay(
        base_row(
            target_chembl_id="CHEMBL1075094",
            description="Activation of Nrf2 assessed as increase in NQO1 enzymatic activity by MTT assay",
            standard_types=["EC50"],
        )
    )

    assert decision.status == "quality_fail"
    assert decision.quality_status == "not_functional_nrf2_state"


def test_nrf2_nqo1_are_luciferase_reporter_remains_h1():
    decision = classify_distance_assay(
        base_row(
            target_chembl_id="CHEMBL1075094",
            description="Activation of Nrf2 in NQO1 ARE-Luc cells by luciferase reporter gene assay",
            standard_types=["EC50"],
        )
    )

    assert decision.status == "include"
    assert decision.family_id == "nrf2_activation"


def test_cellular_nrf2_translocation_on_keap1_nrf2_target_is_h1_not_h2():
    decision = classify_distance_assay(
        base_row(
            target_chembl_id="CHEMBL3038498",
            description=(
                "Activation of NRF2 in human U2OS cells co-expressing Keap1 assessed as induction "
                "of NRF2 translocation to nucleus"
            ),
            standard_types=["EC50"],
            confidence_score=5,
        )
    )

    assert decision.status == "include"
    assert decision.family_id == "nrf2_activation"
    assert decision.distance_level == "H1"


def test_biochemical_keap1_nrf2_ppi_inhibition_is_efflux_h2():
    decision = classify_distance_assay(
        base_row(
            target_chembl_id="CHEMBL3038498",
            description=(
                "Inhibition of Keap1-Nrf2 protein-protein interaction by fluorescence polarization assay"
            ),
            standard_types=["IC50"],
            confidence_score=5,
        )
    )

    assert decision.status == "include"
    assert decision.family_id == "keap1_nrf2_interaction"
    assert decision.tree_node_id == "Distance.h2.efflux_transport"
    assert decision.distance_level == "H2"


def test_keap1_thermal_shift_binding_is_not_h2():
    decision = classify_distance_assay(
        base_row(
            target_chembl_id="CHEMBL3038498",
            description="ThermoFluor assay measuring KEAP1 Kelch protein thermal stability",
            standard_types=["Kd"],
            confidence_score=4,
        )
    )

    assert decision.status == "quality_fail"
    assert decision.quality_status == "not_functional_keap1_nrf2_ppi"


def test_hif1_hre_reporter_is_influx_h1():
    decision = classify_distance_assay(
        base_row(
            target_chembl_id="CHEMBL4261",
            description="Activation of HIF1alpha by HRE-luciferase reporter gene assay",
            standard_types=["EC50"],
        )
    )

    assert decision.status == "include"
    assert decision.family_id == "hif1_activation"
    assert decision.tree_node_id == "Distance.h1.influx_transport"
    assert decision.distance_level == "H1"


def test_hif1_downstream_vegf_only_readout_is_excluded():
    decision = classify_distance_assay(
        base_row(
            target_chembl_id="CHEMBL4261",
            description="Inhibition of HIF1-mediated VEGF production by ELISA",
            standard_types=["Inhibition"],
        )
    )

    assert decision.status == "quality_fail"
    assert decision.quality_status == "not_functional_hif1_state"


def test_direct_phd2_hydroxylase_inhibition_is_influx_h2():
    decision = classify_distance_assay(
        base_row(
            target_chembl_id="CHEMBL5697",
            description="Inhibition of PHD2 enzyme using HIF-1alpha peptide as substrate",
            standard_types=["IC50"],
        )
    )

    assert decision.status == "include"
    assert decision.family_id == "phd2_activity"
    assert decision.tree_node_id == "Distance.h2.influx_transport"
    assert decision.distance_level == "H2"


def test_phd2_substrate_binding_displacement_is_not_hydroxylase_activity():
    decision = classify_distance_assay(
        base_row(
            target_chembl_id="CHEMBL5697",
            description="Inhibition of FITC-HIF1alpha binding to PHD2 by fluorescence polarization assay",
            standard_types=["IC50"],
        )
    )

    assert decision.status == "quality_fail"
    assert decision.quality_status == "not_functional_phd2_activity"


def test_phd2_target_with_hif1_hre_cellular_readout_is_h1():
    decision = classify_distance_assay(
        base_row(
            target_chembl_id="CHEMBL5697",
            description="PHD2 inhibition assessed as HIF1 activation by HRE-luciferase reporter assay",
            standard_types=["EC50"],
        )
    )

    assert decision.status == "include"
    assert decision.family_id == "hif1_activation"
    assert decision.distance_level == "H1"


def test_base_census_records_fine_grained_states_already_in_dc():
    assert classify_base_measured_states(
        base_row(
            tier="Tier 3",
            target_chembl_id="CHEMBL3401",
            description="Activation of PXR assessed as induction of P-gp protein expression",
        )
    ) == (
        "efflux_transport",
        "efflux_transporter_abundance",
        "pxr_car_activation",
    )


def test_frozen_distance_manifest_fields_become_source_group_assignment():
    assignment = assign_distance_endpoint_group(
        {
            "assay_chembl_id": "CHEMBL_TEST",
            "distance_level": "H2",
            "distance_family_id": "phd2_activity",
            "source_group_id": "Distance H2.influx_transport",
            "effect_direction": "inhibits PHD2",
            "mapping_reason": "frozen mapping",
        }
    )

    assert assignment.tier == "Distance H2"
    assert assignment.group_id == "Distance H2.influx_transport"
    assert assignment.endpoint_group == "phd2_activity"
