from tools.chembl_tool.tasks.bioavailability_ma.scoring import scored_row


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


def test_absolute_oral_bioavailability_is_tier1():
    scored = scored_row(
        base_row(
            description="Absolute oral bioavailability after oral and intravenous administration in rat",
            standard_types=["Bioavailability"],
        ),
        min_score=40,
    )
    assert scored["keep_for_bioavailability_reasoning"]
    assert scored["tier"] == "Tier 1"


def test_oral_auc_is_tier2_when_route_is_oral():
    scored = scored_row(
        base_row(description="Plasma exposure after oral dose", standard_types=["AUC"]),
        min_score=40,
    )
    assert scored["keep_for_bioavailability_reasoning"]
    assert scored["tier"] == "Tier 2"


def test_caco2_papp_is_tier3():
    scored = scored_row(
        base_row(description="Caco-2 apparent permeability assay apical to basolateral", standard_types=["Papp"]),
        min_score=40,
    )
    assert scored["keep_for_bioavailability_reasoning"]
    assert scored["tier"] == "Tier 3"


def test_solubility_is_tier4():
    scored = scored_row(base_row(description="Aqueous solubility at pH 6.8", standard_types=["Solubility"]), min_score=40)
    assert scored["keep_for_bioavailability_reasoning"]
    assert scored["tier"] == "Tier 4"


def test_microsomal_stability_is_tier5():
    scored = scored_row(
        base_row(description="Human liver microsome metabolic stability by substrate depletion", standard_types=["T1/2"]),
        min_score=40,
    )
    assert scored["keep_for_bioavailability_reasoning"]
    assert scored["tier"] == "Tier 5"


def test_food_effect_is_tier6():
    scored = scored_row(
        base_row(description="Food effect on relative bioavailability in fed and fasted subjects", standard_types=["AUC ratio"]),
        min_score=40,
    )
    assert scored["keep_for_bioavailability_reasoning"]
    assert scored["tier"] == "Tier 6"


def test_cyp_inhibition_is_not_metabolic_stability():
    scored = scored_row(
        base_row(
            description="Inhibition of CYP3A4 enzyme",
            standard_types=["IC50"],
            target_genes=["CYP3A4"],
            confidence_score=9,
            n_unique_molecules=30,
        ),
        min_score=40,
    )
    assert not scored["keep_for_bioavailability_reasoning"]


def test_subrenal_capsule_xenograft_is_not_formulation_capsule():
    scored = scored_row(
        base_row(
            description="In vivo antitumor activity against subrenal capsule mammary carcinoma xenograft",
            assay_tissue="Capsule",
            standard_types=["Antitumor activity"],
        ),
        min_score=40,
    )
    assert not scored["keep_for_bioavailability_reasoning"]


def test_ppb_is_not_kept_by_default():
    scored = scored_row(base_row(description="Human plasma protein binding", standard_types=["Fu plasma"]), min_score=40)
    assert not scored["keep_for_bioavailability_reasoning"]
