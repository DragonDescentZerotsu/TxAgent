from tools.chembl_tool.tasks.bbb_martins.endpoint_groups import assign_endpoint_group


def test_tier1_kpuu_maps_to_unbound_brain():
    assignment = assign_endpoint_group({"assay_tier": "Tier 1", "standard_type": "K(p,uu,brain)"})

    assert assignment.group_id == "Tier 1.direct_unbound_brain"
    assert assignment.evidence_direction == "supports_bbb_crossing"
    assert assignment.evidence_strength == "strong"


def test_tier2_papp_maps_to_passive_papp():
    assignment = assign_endpoint_group({"assay_tier": "Tier 2", "standard_type": "Papp"})

    assert assignment.group_id == "Tier 2.passive_papp"
    assert assignment.evidence_direction == "permeability_support"


def test_tier3_bidirectional_papp_maps_to_functional_efflux():
    assignment = assign_endpoint_group(
        {
            "assay_tier": "Tier 3",
            "standard_type": "Papp",
            "assay_description": "Bidirectional Papp efflux ratio in MDCK-MDR1 cells",
        }
    )

    assert assignment.group_id == "Tier 3.efflux_functional_ratio_or_bidirectional"
    assert assignment.evidence_direction == "efflux_risk"


def test_tier3_ic50_maps_to_weak_inhibition_binding():
    assignment = assign_endpoint_group(
        {
            "assay_tier": "Tier 3",
            "standard_type": "IC50",
            "target_genes": "ABCB1",
        }
    )

    assert assignment.group_id == "Tier 3.efflux_inhibition_or_binding"
    assert assignment.evidence_direction == "context_dependent"
    assert assignment.evidence_strength == "weak"


def test_tier4_uptake_maps_to_functional_influx():
    assignment = assign_endpoint_group(
        {
            "assay_tier": "Tier 4",
            "standard_type": "Drug uptake",
            "target_genes": "SLC7A5",
        }
    )

    assert assignment.group_id == "Tier 4.influx_functional_uptake_or_transport"
    assert assignment.evidence_direction == "influx_support"
