from tools.chembl_tool.tasks.dili.scoring import scored_row


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


def test_human_dili_anchor_is_tier1():
    scored = scored_row(
        base_row(description="Human clinical drug-induced liver injury case with hepatotoxicity and jaundice"),
        min_score=40,
    )

    assert scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "Tier 1"


def test_hys_law_laboratory_signal_is_tier1():
    scored = scored_row(
        base_row(
            description="Clinical trial liver safety analysis in patients",
            standard_types=["Hy's law ALT bilirubin"],
        ),
        min_score=40,
    )

    assert scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "Tier 1"


def test_in_vivo_liver_histopathology_is_tier2():
    scored = scored_row(
        base_row(description="Rat repeated dose in vivo liver histopathology with hepatocellular necrosis"),
        min_score=40,
    )

    assert scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "Tier 2"


def test_bsep_transport_assay_is_tier3():
    scored = scored_row(
        base_row(
            description="Functional bile acid efflux inhibition assay",
            target_pref_name="Bile salt export pump",
            target_genes=["ABCB11"],
            standard_types=["IC50"],
            confidence_score=9,
            relationship_type="D",
        ),
        min_score=40,
    )

    assert scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "Tier 3"
    assert "ABCB11" in scored["matched_targets"]


def test_hepatic_mitochondrial_stress_is_tier4():
    scored = scored_row(
        base_row(
            description="Mitochondrial membrane potential loss and ATP depletion in primary human hepatocytes",
            standard_types=["mitochondrial membrane potential"],
        ),
        min_score=40,
    )

    assert scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "Tier 4"


def test_reactive_metabolite_bioactivation_is_tier5():
    scored = scored_row(
        base_row(description="GSH adduct formation by reactive metabolite in human liver microsomes"),
        min_score=40,
    )

    assert scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "Tier 5"


def test_heparg_cell_injury_is_tier6():
    scored = scored_row(
        base_row(description="HepaRG hepatocyte viability cytotoxicity assay with LDH release"),
        min_score=40,
    )

    assert scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "Tier 6"


def test_generic_cyp_inhibition_without_bioactivation_is_excluded():
    scored = scored_row(
        base_row(
            description="CYP3A4 inhibition biochemical assay",
            target_genes=["CYP3A4"],
            standard_types=["IC50"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "excluded"
    assert "CYP inhibition" in scored["reason"]


def test_liver_cancer_growth_inhibition_is_excluded():
    scored = scored_row(
        base_row(description="Growth inhibition in hepatocellular carcinoma HepG2 liver cancer cells"),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "excluded"
    assert "liver cancer" in scored["reason"]


def test_hepatoprotective_apap_necrosis_assay_is_not_dili_evidence():
    scored = scored_row(
        base_row(
            description="Tested for Protection from Acetaminophen-Induced Liver Necrosis in Mice",
            assay_organism="Mus musculus",
            assay_tissue="Liver",
            standard_types=["No. of animals with liver necrosis"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "excluded"
    assert "hepatoprotective" in scored["reason"]


def test_hypolipidemic_liver_weight_efficacy_assay_is_not_dili_evidence():
    scored = scored_row(
        base_row(
            description="Hypolipidemic effects in rat fed a high cholesterol diet by oral administration; increase in liver weight",
            assay_organism="Rattus norvegicus",
            standard_types=["liver weight"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "excluded"
    assert "hypolipidemic" in scored["reason"]


def test_protective_liver_disease_model_is_not_dili_evidence():
    scored = scored_row(
        base_row(
            description="Protection against CCl4-induced liver fibrosis in male mouse assessed as decrease in hepatic hydroxyproline",
            assay_organism="Mus musculus",
            standard_types=["activity"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "excluded"


def test_cytoprotection_against_mitochondrial_dysfunction_is_not_dili_evidence():
    scored = scored_row(
        base_row(
            description="Cytoprotection against rotenone-induced mitochondrial dysfunction in human HepG2 cells assessed as ATP rescue",
            assay_cell_type="HepG2",
            standard_types=["activity"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "excluded"


def test_rotenone_atp_rescue_is_not_mitochondrial_dili_risk():
    scored = scored_row(
        base_row(
            description="Suppression of rotenone-induced ATP depletion in human HepG2 cells assessed as increase in rescue of ATP level",
            assay_cell_type="HepG2",
            standard_types=["activity"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "excluded"


def test_h2o2_antioxidant_protection_is_not_dili_risk():
    scored = scored_row(
        base_row(
            description="Inhibition of H2O2-induced ROS accumulation in human HepG2 cells assessed as ROS generation",
            assay_cell_type="HepG2",
            standard_types=["ROS generation"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "excluded"


def test_liver_protection_index_steatosis_model_is_not_dili_risk():
    scored = scored_row(
        base_row(
            description="Inhibition of liver injury in 40% FBS-induced steatosis human HepG2 cells assessed as increase in liver protection index",
            assay_cell_type="HepG2",
            standard_types=["GSH level"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "excluded"


def test_hfd_antidiabetic_liver_weight_model_is_not_dili_evidence():
    scored = scored_row(
        base_row(
            description="Antidiabetic activity in mouse high fat dietary model assessed as reduction in liver weight",
            assay_organism="Mus musculus",
            standard_types=["activity"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "excluded"


def test_hfd_ccl4_anti_hepatic_steatosis_model_is_not_dili_evidence():
    scored = scored_row(
        base_row(
            description="Anti-hepatic steatosis activity in HFD/CCl4-induced non-alcoholic hepatic steatosis mouse model assessed as reduction in liver fibrosis",
            assay_organism="Mus musculus",
            standard_types=["activity"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "excluded"


def test_hfd_fed_liver_weight_model_is_not_dili_evidence():
    scored = scored_row(
        base_row(
            description="Effect on liver weight in HFD-fed mouse at 5 g/kg of diet after 28 days",
            assay_organism="Mus musculus",
            standard_types=["liver weight"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "excluded"


def test_pcsk9_hepg2_reporter_is_not_hepatic_cell_injury():
    scored = scored_row(
        base_row(
            description="Inhibition of PCSK9 transcriptional activity in human HepG2 cells by luciferase reporter assay",
            target_pref_name="Proprotein convertase subtilisin/kexin type 9",
            standard_types=["inhibition"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "excluded"


def test_abhd10_pubchem_target_assay_is_not_bioactivation():
    scored = scored_row(
        base_row(
            description="PubChem BioAssay to identify inhibitors of PME-1 by Activity-Based Protein Profiling",
            target_pref_name="Palmitoyl-protein thioesterase ABHD10, mitochondrial",
            standard_types=["IC50"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "excluded"


def test_kidney_microsome_acyl_glucuronide_is_not_hepatic_bioactivation():
    scored = scored_row(
        base_row(
            description="Drug metabolism in human kidney microsomes assessed as UGT2B7-mediated acyl glucuronide formation",
            target_pref_name="UDP-glucuronosyltransferase 2B7",
            standard_types=["Km"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "excluded"


def test_ldha_lactate_production_is_not_ldh_release_cell_injury():
    scored = scored_row(
        base_row(
            description="Inhibition of LDHA in mouse primary hepatocyte assessed as inhibition of lactate production",
            target_pref_name="L-lactate dehydrogenase A chain",
            standard_types=["IC50"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "excluded"


def test_tumor_necrosis_without_liver_context_is_not_tier2_dili():
    scored = scored_row(
        base_row(
            description="Percent necrosis with EMT-6 Tumor in BALB/c mice after photodynamic therapy",
            assay_organism="Mus musculus",
            standard_types=["necrosis response"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]


def test_herg_is_not_dili_evidence():
    scored = scored_row(
        base_row(description="hERG potassium channel inhibition cardiac safety assay", standard_types=["IC50"]),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]


def test_nonhepatic_generic_cytotoxicity_is_not_kept():
    scored = scored_row(
        base_row(description="Generic cytotoxicity GI50 in A549 cancer cells", standard_types=["GI50"]),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]


def test_jaundiced_animal_clearance_model_is_not_tier1_dili():
    scored = scored_row(
        base_row(
            description="Compound was tested for % clearance through the faeces in jaundice Wistar rats",
            assay_tissue="Feces",
            assay_organism="Rattus norvegicus",
            standard_types=["feces"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]


def test_bone_marrow_alp_is_not_liver_clinical_chemistry():
    scored = scored_row(
        base_row(
            description="Effect on alkaline phosphatase activity in culture of rat bone marrow stromal cells",
            assay_tissue="Bone element",
            target_pref_name="Alkaline phosphatase, tissue-nonspecific isozyme",
            target_genes=["ALPL"],
            target_synonyms=["Alkaline phosphatase liver/bone/kidney isozyme"],
            standard_types=["ALP activity"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]


def test_asbt_ileal_bile_acid_uptake_is_not_dili_transporter_evidence():
    scored = scored_row(
        base_row(
            description="In vitro inhibition of apical sodium-dependent bile acid transporter by taurocholate uptake in hamster ileal ring",
            target_pref_name="Ileal sodium/bile acid cotransporter SLC10A2",
            target_genes=["SLC10A2"],
            standard_types=["IC50"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]


def test_asbt_target_context_is_excluded_even_when_description_says_bile_acid_transporter():
    scored = scored_row(
        base_row(
            description="In vitro inhibition of taurocholate uptake in cells transfected with human Bile acid transporter",
            target_pref_name="Ileal sodium/bile acid cotransporter SLC10A2",
            target_genes=["SLC10A2"],
            standard_types=["IC50"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]


def test_akr1c_bile_acid_adjacent_enzyme_is_not_cholestasis_evidence():
    scored = scored_row(
        base_row(
            description="Inhibition of human recombinant 20-alpha HSD",
            target_pref_name="Aldo-keto reductase family 1 member C1",
            standard_types=["IC50"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "excluded"


def test_beta_glucosidase_bile_acid_enzyme_is_not_transporter_evidence():
    scored = scored_row(
        base_row(
            description="Inhibition of bile acid beta-glucosidase activity",
            target_pref_name="Beta-glucosidase",
            standard_types=["IC50"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "excluded"


def test_ileal_brush_border_taurocholate_transport_is_not_dili_evidence():
    scored = scored_row(
        base_row(
            description="Inhibition of Na-dependent taurocholate uptake into ileal brush border membrane vesicles",
            assay_organism="Oryctolagus cuniculus",
            standard_types=["IC50"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]


def test_pgp_abcb1_is_not_cholestatic_dili_transporter_evidence():
    scored = scored_row(
        base_row(
            description="Inhibitory activity against P-glycoprotein in mammary carcinoma cells",
            target_pref_name="ATP-dependent translocase ABCB1",
            target_genes=["ABCB1"],
            standard_types=["IC50"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]


def test_receptor_covalent_binding_is_not_reactive_metabolite_evidence():
    scored = scored_row(
        base_row(
            description="Irreversible covalent binding to estrogen receptor alpha",
            target_pref_name="Estrogen receptor",
            standard_types=["Covalent binding"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]


def test_metabolic_covalent_binding_remains_tier5():
    scored = scored_row(
        base_row(
            description="In vitro metabolism of radiolabeled compound in rat liver microsomal protein with NADPH regenerating system",
            standard_types=["Covalent binding"],
        ),
        min_score=40,
    )

    assert scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "Tier 5"


def test_hepg2_2215_antiviral_cytotoxicity_control_is_not_dili_evidence():
    scored = scored_row(
        base_row(
            description="Cytotoxicity in 2.2.15 cells",
            assay_cell_type="HepG2 2.2.15",
            standard_types=["IC50"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]


def test_generic_hepg2_mtt_cytotoxicity_is_not_dili_evidence():
    scored = scored_row(
        base_row(
            description="Cytotoxicity against human HepG2 cells after 3 days by MTT assay",
            target_pref_name="HepG2",
            standard_types=["IC50"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]


def test_hepg2_c3a_admet_toxicity_remains_tier6():
    scored = scored_row(
        base_row(
            description="Toxicity in human HepG2/C3A cells assessed as cell viability after 10 days",
            target_pref_name="ADMET",
            standard_types=["Cell viability"],
        ),
        min_score=40,
    )

    assert scored["keep_for_dili_reasoning"]
    assert scored["tier"] == "Tier 6"


def test_primary_hepatocyte_dna_single_strand_break_genotox_is_not_dili_evidence():
    scored = scored_row(
        base_row(
            description="Ex vivo analysis of DNA single-strand breaks in primary rat hepatocyte",
            standard_types=["viability"],
        ),
        min_score=40,
    )

    assert not scored["keep_for_dili_reasoning"]
