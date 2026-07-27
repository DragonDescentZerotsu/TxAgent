from tools.chembl_tool.tasks.bbb_martins.audit_role_gated_overlap import (
    classify_role_evidence,
)


def _row(*, group_id: str, standard_type: str, description: str, genes: str = "") -> dict[str, str]:
    return {
        "group_id": group_id,
        "standard_type": standard_type,
        "assay_description": description,
        "target_genes": genes,
        "standard_relation": "=",
        "standard_value": "3.0",
    }


def test_efflux_ratio_for_activity_row_molecule_passes_role_gate():
    decision = classify_role_evidence(
        _row(
            group_id="Tier 3.efflux_functional_ratio_or_bidirectional",
            standard_type="Ratio",
            description=(
                "Ratio of permeability from basolateral to apical over apical to basolateral in human "
                "L-MDR1 cells"
            ),
        )
    )
    assert decision.role == "efflux_substrate:ABCB1"


def test_probe_accumulation_inhibitor_does_not_prove_efflux_substrate_role():
    decision = classify_role_evidence(
        _row(
            group_id="Tier 3.efflux_transport_or_accumulation",
            standard_type="Activity",
            description="Inhibition of human BCRP assessed as mitoxantrone accumulation",
            genes="ABCG2",
        )
    )
    assert decision.role is None


def test_atpase_or_probe_group_is_not_accepted_as_direct_transport():
    decision = classify_role_evidence(
        _row(
            group_id="Tier 3.efflux_atpase_or_probe",
            standard_type="Ratio_ATPase activity",
            description="Stimulation of P-glycoprotein ATPase activity",
            genes="ABCB1",
        )
    )
    assert decision.role is None


def test_low_efflux_ratio_does_not_prove_substrate_role():
    row = _row(
        group_id="Tier 3.efflux_functional_ratio_or_bidirectional",
        standard_type="Ratio",
        description="Efflux ratio of permeability in human MDR1 cells",
    )
    row["standard_value"] = "1.2"
    decision = classify_role_evidence(row)
    assert decision.role is None


def test_direct_glut1_substrate_uptake_passes_role_gate():
    decision = classify_role_evidence(
        _row(
            group_id="Tier 4.influx_functional_uptake_or_transport",
            standard_type="Drug uptake",
            description="Substrate activity at GLUT1 assessed as cellular uptake",
            genes="SLC2A1",
        )
    )
    assert decision.role == "influx_substrate:SLC2A1"


def test_inhibition_of_glucose_uptake_does_not_prove_glut1_substrate_role():
    decision = classify_role_evidence(
        _row(
            group_id="Tier 4.influx_functional_uptake_or_transport",
            standard_type="Activity",
            description="Inhibition of [3H]glucose uptake at mammalian GLUT1",
            genes="SLC2A1",
        )
    )
    assert decision.role is None


def test_not_active_glut1_substrate_assay_does_not_pass_role_gate():
    row = _row(
        group_id="Tier 4.influx_functional_uptake_or_transport",
        standard_type="Activity",
        description="Substrate activity at GLUT1 assessed as cellular uptake",
        genes="SLC2A1",
    )
    row["standard_value"] = ""
    row["activity_comment"] = "Not Active"
    decision = classify_role_evidence(row)
    assert decision.role is None
