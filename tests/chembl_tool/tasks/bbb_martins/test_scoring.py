from tools.chembl_tool.tasks.bbb_martins.scoring import scored_row


def base_row(**overrides):
    row = {
        "description": "",
        "assay_cell_type": "",
        "assay_tissue": "",
        "assay_organism": "",
        "assay_test_type": "",
        "assay_category": "",
        "target_pref_name": "",
        "target_genes": [],
        "target_synonyms": [],
        "component_descriptions": [],
        "standard_types": [],
        "confidence_score": 0,
        "relationship_type": "U",
        "n_unique_molecules": 1,
    }
    row.update(overrides)
    return row


def test_pgp_spellings_match_efflux():
    for spelling in ["P-gp", "P glycoprotein", "P-glycoprotein"]:
        scored = scored_row(base_row(description=f"{spelling} substrate transport assay"), min_score=40)
        assert scored["keep_for_bbb_reasoning"]
        assert scored["tier"] == "Tier 3"


def test_caco2_spellings_match_passive_permeability():
    for spelling in ["Caco-2", "Caco2"]:
        scored = scored_row(base_row(description=f"Apparent permeability Papp in human {spelling} cells"), min_score=40)
        assert scored["keep_for_bbb_reasoning"]
        assert scored["tier"] == "Tier 2"


def test_blood_brain_barrier_spellings_match_direct():
    for spelling in ["blood-brain barrier", "blood brain barrier"]:
        scored = scored_row(base_row(description=f"Ability to penetrate the {spelling}"), min_score=40)
        assert scored["keep_for_bbb_reasoning"]
        assert scored["tier"] == "Tier 1"


def test_kpuu_brain_endpoint_matches_direct():
    for endpoint in ["K(p,uu,brain)", "Kp,uu brain", "Kpuu brain"]:
        scored = scored_row(base_row(standard_types=[endpoint]), min_score=40)
        assert scored["keep_for_bbb_reasoning"]
        assert scored["tier"] == "Tier 1"


def test_csf_to_unbound_plasma_matches_direct():
    scored = scored_row(
        base_row(description="Ratio of drug level in cerebrospinal fluid to unbound plasma concentration"),
        min_score=40,
    )
    assert scored["keep_for_bbb_reasoning"]
    assert scored["tier"] == "Tier 1"


def test_mdck_mdr1_efflux_ratio_prefers_efflux():
    scored = scored_row(
        base_row(
            description="Efflux ratio in MDCK-MDR1 cells by bidirectional Papp assay",
            standard_types=["efflux ratio"],
        ),
        min_score=40,
    )
    assert scored["keep_for_bbb_reasoning"]
    assert scored["tier"] == "Tier 3"


def test_dopamine_receptor_binding_is_not_kept():
    scored = scored_row(base_row(description="Dopamine receptor binding IC50"), min_score=40)
    assert not scored["keep_for_bbb_reasoning"]


def test_abcb1_target_substrate_is_kept():
    scored = scored_row(
        base_row(
            description="Substrate transport assay",
            standard_types=["substrate"],
            target_pref_name="ATP-dependent translocase ABCB1",
            target_genes=["ABCB1"],
            confidence_score=9,
            relationship_type="D",
        ),
        min_score=40,
    )
    assert scored["keep_for_bbb_reasoning"]
    assert scored["tier"] == "Tier 3"


def test_generic_permeability_is_not_kept():
    scored = scored_row(base_row(description="Vascular permeability for anti-inflammatory activity"), min_score=40)
    assert not scored["keep_for_bbb_reasoning"]


def test_papp_endpoint_without_model_context_is_not_kept():
    scored = scored_row(
        base_row(
            description="Inhibition of human factor 9a",
            target_pref_name="Coagulation factor IX",
            standard_types=["papp"],
        ),
        min_score=40,
    )
    assert not scored["keep_for_bbb_reasoning"]


def test_bbb_papp_proxy_is_tier2_not_direct():
    scored = scored_row(
        base_row(
            description="Apparent permeability across blood-brain barrier by PAMPA",
            standard_types=["papp"],
        ),
        min_score=40,
    )
    assert scored["keep_for_bbb_reasoning"]
    assert scored["tier"] == "Tier 2"


def test_generic_drug_uptake_is_not_kept():
    scored = scored_row(base_row(description="Drug uptake in tumor cells"), min_score=40)
    assert not scored["keep_for_bbb_reasoning"]


def test_glucose_uptake_requires_glut1_context():
    generic = scored_row(base_row(description="Glucose uptake stimulation"), min_score=40)
    assert not generic["keep_for_bbb_reasoning"]

    glut1 = scored_row(
        base_row(
            description="Glucose uptake transport assay",
            target_pref_name="Solute carrier family 2 facilitated glucose transporter member 1",
            target_genes=["SLC2A1"],
        ),
        min_score=40,
    )
    assert glut1["keep_for_bbb_reasoning"]
    assert glut1["tier"] == "Tier 4"


def test_non_transporter_atpase_substrate_is_not_efflux():
    scored = scored_row(
        base_row(
            description="Inhibition of proteasome using ATP as substrate",
            target_pref_name="26S proteasome",
            target_synonyms=["26S proteasome AAA-ATPase subunit RPT1"],
            standard_types=["IC50"],
        ),
        min_score=40,
    )
    assert not scored["keep_for_bbb_reasoning"]


def test_brain_plasma_membrane_binding_false_positive_is_excluded():
    scored = scored_row(
        base_row(description="Binding to receptor in mouse brain plasma membrane"),
        min_score=40,
    )
    assert not scored["keep_for_bbb_reasoning"]
    assert scored["reason"].startswith("排除 brain plasma membrane")


def test_real_brain_to_plasma_evidence_is_kept():
    scored = scored_row(
        base_row(description="Ratio of drug level in brain to plasma"),
        min_score=40,
    )
    assert scored["keep_for_bbb_reasoning"]
    assert scored["tier"] == "Tier 1"


def test_nonfunctional_influx_gene_expression_assay_is_excluded():
    scored = scored_row(
        base_row(description="Fold change in Slc2a1 RNA stability measured by ChIP-seq"),
        min_score=40,
    )
    assert not scored["keep_for_bbb_reasoning"]
    assert "非功能性 influx" in scored["reason"]


def test_nonfunctional_glut1_level_western_blot_is_excluded_even_with_target_context():
    scored = scored_row(
        base_row(
            description="Effect on total GLUT1 level in rat L6 cells by Western blotting analysis",
            target_pref_name="L6",
            target_synonyms=["Solute carrier family 2 facilitated glucose transporter member 1"],
        ),
        min_score=40,
    )
    assert not scored["keep_for_bbb_reasoning"]
    assert "非功能性 influx" in scored["reason"]


def test_functional_glut1_uptake_assay_is_kept():
    scored = scored_row(
        base_row(
            description="Inhibition of GLUT1 assessed as reduction in glucose uptake",
            target_genes=["SLC2A1"],
        ),
        min_score=40,
    )
    assert scored["keep_for_bbb_reasoning"]
    assert scored["tier"] == "Tier 4"


def test_nontransporter_resistant_cell_line_noise_is_excluded():
    scored = scored_row(
        base_row(
            description="ABCB1-substrate-selected resistant cell line with doxorubicin treatment",
            target_pref_name="Mitogen-activated protein kinase kinase kinase 5",
            target_genes=["MAP3K5"],
            standard_types=["IC50"],
        ),
        min_score=40,
    )
    assert not scored["keep_for_bbb_reasoning"]
    assert "resistant-cell-line" in scored["reason"]


def test_nontransporter_abcg2_selected_cell_line_noise_is_excluded():
    scored = scored_row(
        base_row(
            description="ABCG2-substrate-selected resistant cell line with mitoxantrone treatment",
            target_pref_name="Mitogen-activated protein kinase kinase kinase 5",
            target_genes=["MAP3K5"],
            standard_types=["IC50"],
        ),
        min_score=40,
    )
    assert not scored["keep_for_bbb_reasoning"]


def test_functional_rhodamine_efflux_assay_without_target_gene_is_kept():
    scored = scored_row(
        base_row(
            description="Inhibition of P-gp-mediated rhodamine 123 efflux in doxorubicin-resistant cells",
            target_pref_name="Unchecked",
        ),
        min_score=40,
    )
    assert scored["keep_for_bbb_reasoning"]
    assert scored["tier"] == "Tier 3"


def test_true_abcb1_target_assay_is_kept():
    scored = scored_row(
        base_row(
            description="ABCB1-mediated transport in MDCK cells",
            target_pref_name="ATP-dependent translocase ABCB1",
            target_genes=["ABCB1"],
        ),
        min_score=40,
    )
    assert scored["keep_for_bbb_reasoning"]
    assert scored["tier"] == "Tier 3"
