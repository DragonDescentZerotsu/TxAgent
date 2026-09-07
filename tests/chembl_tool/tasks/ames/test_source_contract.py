import pytest

from tools.chembl_tool.tasks.ames import source_contract as contract
from tools.chembl_tool.tasks.ames.source_contract import (
    DIRECT,
    gold_candidate as classify,
    strain_panel,
)


def primary(**changes):
    return {
        "SMILES": "c1ccccc1",
        "support_text": "The compound was negative in the Ames assay.",
        "chemical_entity_type": "defined_chemical",
        "assay_family": "bacterial_reverse_mutation",
        "experimental_context": "in_vitro",
        "evidence_basis": "current_study_experiment",
        "needs_more_context": False,
        "qualifying_conditions": None,
        "test_system": "Salmonella typhimurium TA98 and TA100",
        "metabolic_activation": "absent",
        "mutagenicity_result": "negative",
        **changes,
    }


def test_panel_and_activation_are_kept_as_one_record_condition():
    result = classify("ames_base", primary(metabolic_activation="both"))
    assert result.group == DIRECT
    assert result.label == 0
    assert result.condition_atoms == (
        "metabolic_activation=both_reported",
        "strain_panel=TA100,TA98",
    )
    assert strain_panel("TA98 and TA100") == ("TA100", "TA98")
    assert strain_panel("TA97a, WP2 uvrA") == ("TA97A", "WP2UVRA")


@pytest.mark.parametrize(
    "changes",
    [
        {"mutagenicity_result": "equivocal"},
        {"mutagenicity_result": "weak_positive"},
        {"evidence_basis": "positive_control_or_reference_use"},
        {"evidence_basis": "review_or_regulatory_synthesis"},
        {"qualifying_conditions": "Activation with avocado homogenates"},
        {"needs_more_context": True},
        {"metabolic_activation": "not_reported"},
        {"test_system": "Salmonella typhimurium"},
        {"test_system": "TA98NR and TA100"},
        {"test_system": "TA98 and YG1024"},
    ],
)
def test_unresolved_records_cannot_vote(changes):
    result = classify("ames_base", primary(**changes))
    assert result.label is None
    assert result.group != DIRECT


def test_positive_words_in_support_cannot_override_source_negative():
    result = classify(
        "ames_base",
        primary(
            support_text="Compound A was positive; the test compound B was negative."
        ),
    )
    assert result.label == 0


def test_predictions_and_host_mediated_results_cannot_vote():
    result = classify(
        "ames_base", primary(support_text="An in silico prediction was negative.")
    )
    assert not result.group and result.label is None
    result = classify("ames_base", primary(experimental_context="host_mediated"))
    assert not result.group


@pytest.mark.parametrize(
    "panel",
    [
        "TA98, YG1024",
        "TA100NR",
        "TA98 + unknown strain",
        "TA98, 100, 1535",
        "TA98 and mammalian V79 cells",
        "TA98 (nitroreductase deficient)",
        "TA98 positive, TA100 negative",
        "five Salmonella strains (including TA98)",
        "WP2 uvrA (pKM101)",
        "WP2 uvrA/pKM101",
        "WP2UVRAPKM101",
        "TA98 pKM101",
    ],
)
def test_panel_parser_refuses_partial_panels_and_unresolved_genotypes(panel):
    assert strain_panel(panel) == ()
    assert classify("ames_base", primary(test_system=panel)).label is None


@pytest.mark.parametrize(
    "panel,expected",
    [
        ("Salmonella typhimurium TA98, TA100 and TA1535", ("TA100", "TA1535", "TA98")),
        ("S. typhimurium TA-97a; E. coli WP2 uvrA", ("TA97A", "WP2UVRA")),
        ("Salmonella enterica serovar Typhimurium TA98", ("TA98",)),
    ],
)
def test_complete_panels_preserve_conjunctions_and_species(panel, expected):
    assert strain_panel(panel) == expected


@pytest.mark.parametrize(
    "changes,reason",
    [
        (
            {"experimental_context": "host_mediated"},
            "outside_in_vitro_bacterial_target",
        ),
        (
            {"chemical_entity_type": "unresolved_compound_code"},
            "unresolved_chemical_entity",
        ),
        (
            {"extra_details": "UV irradiation"},
            "special_context_requires_semantic_review",
        ),
        (
            {"extra_details": "S9 from mice fed high-fat diets"},
            "activation_or_experiment_arms_require_review",
        ),
    ],
)
def test_prediction_cannot_rescue_out_of_scope_records(changes, reason):
    result = classify(
        "ames_base",
        primary(support_text="A QSAR predicted a positive outcome.", **changes),
    )
    assert result.group == "" and result.reason == reason


@pytest.mark.parametrize(
    "changes",
    [
        {"test_system": "Salmonella (strains not specified)"},
        {"metabolic_activation": "not_reported"},
    ],
)
def test_mixed_predictions_cannot_escape_through_near_fallback(changes):
    decision = classify(
        "ames_base",
        primary(
            support_text="The QSAR prediction was compared with measurements.",
            **changes,
        ),
    )
    assert decision.group == ""
    assert decision.reason == "prediction_or_mixed_experiment_requires_review"


@pytest.mark.parametrize(
    "support",
    [
        "A positive Ames result (ref 25).",
        "A negative result (reference [25]).",
        "Previous studies showed a negative result.",
        "Earlier reports found a positive result.",
        "A cited experiment was positive.",
        "An earlier result [25] was negative.",
        "No increases were observed (RIFM, 2000a,b).",
        "Buschini et al. found a positive outcome.",
        "Smith et al found a negative outcome.",
    ],
)
def test_current_study_enum_cannot_override_citation_provenance(support):
    decision = classify("ames_base", primary(support_text=support))
    assert decision.group == ""
    assert decision.reason == "cited_or_previous_experiment_requires_provenance_review"


def test_reported_alone_does_not_mean_a_previous_study():
    assert (
        classify(
            "ames_base",
            primary(support_text="The authors reported a negative Ames result."),
        ).group
        == DIRECT
    )


@pytest.mark.parametrize(
    "details,reason",
    [
        (
            "Negative liquid pre-incubation, weak-positive plate-incorporation.",
            "assay_method_comparison_requires_review",
        ),
        (
            "Plate incorporation and preincubation gave concordant results.",
            "assay_method_comparison_requires_review",
        ),
        (
            "The probable mutagen in wood emissions was the named compound.",
            "inferred_subject_or_outcome_requires_review",
        ),
        (
            "The compound could explain the mixture response.",
            "inferred_subject_or_outcome_requires_review",
        ),
        (
            "The sample was tested by vapor exposure.",
            "special_exposure_requires_condition_review",
        ),
        (
            "The sample was tested by vapour exposure.",
            "special_exposure_requires_condition_review",
        ),
        (
            "Exposure in a sealed-desiccator for seven hours.",
            "special_exposure_requires_condition_review",
        ),
        ("Exposure in sealed chambers.", "special_exposure_requires_condition_review"),
        (
            "Testing under aerobic and anaerobic conditions.",
            "special_exposure_requires_condition_review",
        ),
        (
            "An unqualified TA1535-only response.",
            "activation_or_experiment_arms_require_review",
        ),
        (
            "An unqualified TA1535‑only response.",
            "activation_or_experiment_arms_require_review",
        ),
    ],
)
def test_primary_review_covers_protocol_subject_and_exposure(details, reason):
    decision = classify("ames_base", primary(extra_details=details))
    assert decision.group == "" and decision.label is None
    assert decision.reason == reason


@pytest.mark.parametrize(
    "name", ["(-)-3-carene", "(−)-3-carene", "(+)-3-carene", "(1R,2S)-compound"]
)
@pytest.mark.parametrize("field", ["molecule_name", "extra_details"])
def test_named_stereochemistry_without_representation_requires_review(name, field):
    decision = classify("ames_base", primary(**{field: name}))
    assert (
        decision.group == ""
        and decision.reason == "named_stereochemistry_missing_from_structure"
    )


@pytest.mark.parametrize("method", ["plate incorporation", "preincubation"])
def test_one_conventional_method_is_not_a_method_comparison(method):
    assert (
        classify(
            "ames_base", primary(extra_details=f"The {method} method was used.")
        ).group
        == DIRECT
    )


def test_panel_any_positive_subset_remains_one_source_summary():
    row = primary(
        test_system="Salmonella typhimurium TA98, TA100 and TA1535",
        metabolic_activation="both",
        mutagenicity_result="positive",
        support_text="The panel was tested with and without S9. TA1535 showed a positive "
        "response; TA98 and TA100 did not respond. The overall panel result was positive.",
    )
    decision = classify("ames_base", row)
    assert decision.group == DIRECT and decision.label == 1
    assert decision.condition_atoms == (
        "metabolic_activation=both_reported",
        "strain_panel=TA100,TA1535,TA98",
    )
    # A favorable strain mentioned in text must never override the source enum.
    row["mutagenicity_result"] = "equivocal"
    assert classify("ames_base", row).label is None


def test_actual_base_38510_cited_comparator_potency_is_not_a_primary_vote():
    # raw_v1 ordinal 38510, PMID 6751587, ext_50, paragraph 50.
    row = primary(
        pmid="6751587",
        extraction_id="ext_50",
        paragraph_idx=50,
        molecule_name="1,3,6-trinitropyrene",
        test_system="Salmonella typhimurium TA98",
        metabolic_activation="present",
        mutagenicity_result="positive",
        SMILES="O=[N+]([O-])c1ccc2ccc3c([N+](=O)[O-])cc([N+](=O)[O-])c4ccc1c2c34",
        support_text="The mutagenic activity of 1,3,6-trinitropyrene on Salmonella typhimurium TA98 "
        "was similar to that of 2-amino-3,4-dimethylimidazo[4,5-f]quinoline, the most mutagenic "
        "substance known on TA98 (specific activity 1.41 x 10^5 revertants/nmol) when tested "
        "by the preincubation procedure in the presence of S9 mix (ref 25).",
    )
    assert (
        classify("ames_base", row).reason
        == "cited_or_previous_experiment_requires_provenance_review"
    )


# Actual raw_v1 records, commit 03e4c7c; ordinal retained for source lookup.
FROZEN_SOURCE_EXAMPLES = [
    (
        "ames_base",
        505,
        {
            "paragraph_idx": 0,
            "support_text": "Nitrilotriacetic acid (NTA) was assayed in the Ames test using TA1537, TA98, "
            "TA100, and TA102 tester strains, both with and without metabolic activation by "
            "aroclor 1254-induced liver or kidney rat S9-mix. Whatever the S9 origin, no "
            "genotoxicity was detected in the Ames test. This negative bacterial "
            "reverse-mutation finding is reported alongside positive in-scope genotoxicity "
            "results (increased micronucleus formation without S9, positive mouse lymphoma "
            "tk+/− gene mutation assay, and positive chromosomal aberrations test on human "
            "lymphocytes) in the same paper.",
            "molecule_name": "Nitrilotriacetic acid",
            "chemical_entity_type": "defined_chemical",
            "mutagenicity_result": "negative",
            "evidence_basis": "current_study_experiment",
            "assay_family": "bacterial_reverse_mutation",
            "experimental_context": "in_vitro",
            "test_system": "Salmonella Typhimurium TA1537, TA98, TA100, TA102",
            "metabolic_activation": "both",
            "qualifying_conditions": None,
            "extra_details": "Ames test with and without aroclor 1254-induced liver or kidney rat S9-mix "
            "(kidney S9 tested with or without arachidonic acid instead of NADP); no "
            "genotoxicity or apoptosis detected regardless of S9 origin. NTA CAS 139-13-9. "
            "The negative Ames result is reported in a paper that also establishes NTA "
            "mutagenicity via micronuclei, tk+/− gene mutation, and chromosome aberrations "
            "endpoints.",
            "confidence": 0.9,
            "needs_more_context": False,
            "pmid": "18449932",
            "extraction_id": "ext_1",
            "SMILES": "O=C(O)CN(CC(=O)O)CC(=O)O",
        },
        "activation_or_experiment_arms_require_review",
    ),
    (
        "ames_base",
        819,
        {
            "paragraph_idx": 51,
            "support_text": "In the in vitro study, hepatic S9 fractions from female BALB/c mice fed low- "
            "or high-fat diets were tested with the food mutagens MeIQ, Trp-P-2 and "
            "aflatoxin B1 using the Salmonella mutagenicity assay (liquid preincubation, 30 "
            "min) with S. typhimurium TA98 as indicator strain. Fractions from mice fed a "
            "high-fat diet exhibited a greater ability to activate MeIQ and Trp-P-2 (olive "
            "oil inducing the greater number of revertants), while dietary fat had no "
            "effect on the ability to activate aflatoxin B1.",
            "molecule_name": "MeIQ",
            "chemical_entity_type": "defined_chemical",
            "mutagenicity_result": "positive",
            "evidence_basis": "current_study_experiment",
            "assay_family": "bacterial_reverse_mutation",
            "experimental_context": "in_vitro",
            "test_system": "Salmonella typhimurium TA98",
            "metabolic_activation": "present",
            "qualifying_conditions": None,
            "extra_details": "Food mutagen (2-amino-3-methylimidazo[4,5-f]quinoline) requiring cytochrome "
            "P450 MFO activation; liquid preincubation for 30 min; hepatic S9 from female "
            "BALB/c mice on purified low-fat (1% safflower oil) or high-fat (additional "
            "25% beef dripping or 25% olive oil) diets, 0-4.0 mg S9 protein/ml; high-fat "
            "S9, especially olive-oil, gave greater revertant numbers.",
            "confidence": 0.8,
            "needs_more_context": False,
            "pmid": "2184313",
            "extraction_id": "ext_7",
            "SMILES": "Cn1c(N)nc2c3cccnc3ccc21",
        },
        "special_context_requires_semantic_review",
    ),
    (
        "ames_v2",
        187,
        {
            "paragraph_idx": 29,
            "support_text": "To identify potential ROS involved in Cu(II)-catalyzed 4E₂ oxidative DNA "
            "damage, ROS modifiers sodium azide (singlet oxygen and hydroxyl radical "
            "scavenger), tiron (superoxide scavenger), catalase (decomposes hydrogen "
            "peroxide), and bathocuproine (a Cu(I)-specific chelator) were added to the "
            "standard reaction mixture. Although no qualitative differences were observed "
            "in the DNA oxidation pattern, significant quantitative differences in 8-oxo-dG "
            "and unidentified oxidative DNA adduct levels occurred, and all four ROS "
            "modifiers significantly inhibited (36–96%) oxidative DNA damage induced by "
            "Cu(II) catalysis of 4E₂.",
            "molecule_role": "test_or_damage_agent",
            "assay_family": "32p_postlabelling",
            "assay_version": "32P-postlabelling of unidentified oxidative DNA adducts and 8-oxo-dG with "
            "ROS-modifier intervention",
            "endpoint_class": "dna_adduct",
            "endpoint_subtype": "oxidative DNA adducts (unidentified oxidation adducts and 8-oxo-dG)",
            "biological_system": "cell-free DNA (isolated DNA)",
            "dose_or_concentration": "100",
            "dose_unit": "µM",
            "metabolic_activation_presence": "absent",
            "result_status": "damage_or_response_decreased",
            "response_value": "36–96",
            "response_unit": "% inhibition of oxidative DNA damage",
            "extra_details": "Challenge/cotreatment ROS modifiers and targets: bathocuproine "
            "(Cu(I)-specific chelator, ~100 µM-range per prior studies), catalase "
            "(hydrogen peroxide), tiron (superoxide), sodium azide (singlet oxygen and "
            "hydroxyl radical). Incubation 1 h at 37 °C, pH 7.4. All four modifiers "
            "significantly inhibited oxidative DNA damage (*P ≤ 0.05 vs vehicle). No "
            "qualitative change in oxidation adduct pattern.",
            "confidence": 0.8,
            "needs_more_context": False,
            "pmid": "22126130",
            "extraction_id": "ext_22",
            "SMILES": "C[C@]12CC[C@@H]3c4ccc(O)c(O)c4CC[C@H]3[C@@H]1CC[C@@H]2O",
        },
        "damage_effect_requires_relational_or_outcome_review",
    ),
    (
        "ames_v3",
        130,
        {
            "paragraph_idx": 36,
            "support_text": "APAP was shown to cause DNA strand breaks; addition of the antioxidants "
            "N-acetylcysteine, the polyphenol silibin and α-tocopherol protected primary "
            "rat hepatocytes from APAP-induced DNA strand breaks, but not APAP-induced "
            "toxicity. The role of oxidant stress in these DNA strand breaks was also "
            "demonstrated by an increase in malondialdehyde production at the doses of "
            "acetaminophen which caused DNA damage.",
            "molecule_role": "parent_test_agent",
            "mechanism_category": "oxidative_dna_damage",
            "assay_family": "mechanistically_informative_genotoxicity",
            "assay_method_and_endpoint": "APAP-induced DNA strand breaks in primary rat hepatocytes, tested "
            "in the presence of the antioxidants N-acetylcysteine, silibin and "
            "α-tocopherol to resolve an oxidative mechanism; lipid-oxidation "
            "marker malondialdehyde production at DNA-damaging APAP doses",
            "biological_system": "Primary rat hepatocytes (in vitro)",
            "metabolic_activation_system": "intact_hepatocytes",
            "exposure_and_mechanistic_conditions": "APAP at doses causing DNA damage; antioxidant "
            "cotreatment with N-acetylcysteine, polyphenol silibin, "
            "and α-tocopherol. Exact concentrations and durations "
            "not reported.",
            "result_direction": "decreased",
            "quantitative_readout_value": None,
            "quantitative_readout_unit": None,
            "cytotoxicity_status": "present",
            "extra_details": "Secondary-citation evidence (reference [86]). The antioxidants protected "
            "against APAP-induced DNA strand breaks but not against APAP-induced toxicity; "
            "oxidant stress involvement additionally supported by increased "
            "malondialdehyde (lipid peroxidation) production at DNA-damaging APAP doses.",
            "confidence": 0.72,
            "needs_more_context": False,
            "pmid": "15853763",
            "extraction_id": "ext_15",
            "SMILES": "CC(=O)Nc1ccc(O)cc1",
        },
        "mechanism_effect_requires_relational_or_outcome_review",
    ),
]


@pytest.mark.parametrize("source,ordinal,row,reason", FROZEN_SOURCE_EXAMPLES)
def test_actual_records_with_missing_conditions_or_misattributed_effects(
    source, ordinal, row, reason
):
    decision = contract.classify(source, row)
    assert decision.label is None, (source, ordinal)
    assert decision.group in {contract.NEAR, contract.DAMAGE, contract.MECHANISM}, (
        source,
        ordinal,
    )


@pytest.mark.parametrize("source", ["ames_base", "ames_v1", "ames_v2", "ames_v3"])
@pytest.mark.parametrize("smiles", ["CC(=O)SC[C@H](C)C(=O)O", "NCc1ccccc1O"])
def test_known_identity_mismatch_is_quarantined_across_sources_and_aliases(
    source, smiles
):
    row = primary(
        pmid="24275315", SMILES=smiles, molecule_name="another extraction alias"
    )
    decision = classify(source, row)
    assert decision.group == "" and decision.reason == "known_source_identity_mismatch"
    row["pmid"] = "another_study"
    assert classify(source, row).reason != "known_source_identity_mismatch"


def test_quarantine_does_not_claim_to_repair_identity():
    row = primary(
        pmid="24275315", SMILES="Cc1ccc(N)cc1N", molecule_name="2,4-toluenediamine"
    )
    assert classify("ames_base", row).reason != "known_source_identity_mismatch"


@pytest.mark.parametrize(
    "name",
    [
        "24",
        "2b",
        "2B",
        " 24 ",
        "compound 24",
        "Compound 2b",
        "compound no. 2b",
        "compound 2b hydrochloride",
    ],
)
def test_defined_chemical_enum_does_not_resolve_study_local_codes(name):
    row = primary(molecule_name=name)
    decision = classify("ames_base", row)
    assert decision.group == "" and decision.reason == "unresolved_local_compound_code"
    row["support_text"] = "The QSAR prediction was positive."
    assert classify("ames_base", row).reason == "unresolved_local_compound_code"


@pytest.mark.parametrize(
    "name", ["DDT", "MMS", "B12", "2-nitroethanol", "1,3,6-trinitropyrene"]
)
def test_global_acronyms_and_numeric_chemical_locants_are_not_local_codes(name):
    assert classify("ames_base", primary(molecule_name=name)).group == DIRECT


# Unmodified raw snapshots from the independent 2026-09-05 48-candidate audit.
INDEPENDENT_AUDIT_EXAMPLES = [
    (
        "ames_base:110687",
        {
            "paragraph_idx": 32,
            "support_text": "In the liquid preincubation assay, benzene dihydrodiol was mutagenic in TA100 "
            "and TM677 with, but not without, PMS (Table I).",
            "molecule_name": "Benzene dihydrodiol",
            "chemical_entity_type": "defined_chemical",
            "mutagenicity_result": "negative",
            "evidence_basis": "current_study_experiment",
            "assay_family": "bacterial_reverse_mutation",
            "experimental_context": "in_vitro",
            "test_system": "Salmonella Typhimurium TA100",
            "metabolic_activation": "absent",
            "qualifying_conditions": None,
            "extra_details": "Liquid preincubation without PMS: fold increases did not reach the criterion "
            "of exceeding twofold the spontaneous level (≤1 at 0.75–3.0 mM). Note this "
            "differs from the weak positive result in the plate incorporation test without "
            "PMS.",
            "confidence": 0.85,
            "needs_more_context": False,
            "pmid": "8908188",
            "extraction_id": "ext_23",
            "SMILES": "OC1C=CC=CC1O",
        },
        "assay_method_comparison_requires_review",
    ),
    (
        "ames_base:247541",
        {
            "paragraph_idx": 91,
            "support_text": "The mutagenic potential of 4-carvomenthenol was tested in a GLP bacterial "
            "reverse mutation assay in accordance with OECD TG 471 using S. typhimurium "
            "strains TA1535, TA97a, TA98, TA100 and TA102 treated with 4-carvomenthenol in "
            "DMSO, with and without metabolic activation (S9), at concentrations of "
            "0.016–1.6 mg/plate for TA97a, 0.016–5 mg/plate for TA98 and TA102, and "
            "0.016–1.6 mg/plate (without S9) and 0.016–5 mg/plate (with S9) for TA100 and "
            "TA1535. No increases in the mean number of revertant colonies were observed at "
            "any tested dose in the presence or absence of S9, and under the conditions of "
            "the study the test material was found to have no mutagenic effects.",
            "molecule_name": "4-Carvomenthenol",
            "chemical_entity_type": "defined_chemical",
            "mutagenicity_result": "negative",
            "evidence_basis": "current_study_experiment",
            "assay_family": "bacterial_reverse_mutation",
            "experimental_context": "in_vitro",
            "test_system": "Salmonella Typhimurium TA1535, TA97a, TA98, TA100, TA102",
            "metabolic_activation": "both",
            "qualifying_conditions": None,
            "extra_details": "GLP bacterial reverse mutation assay per OECD TG 471; test material dissolved "
            "in DMSO; concentrations 0.016–1.6 mg/plate (TA97a), 0.016–5 mg/plate (TA98, "
            "TA102), and for TA100/TA1535 0.016–1.6 mg/plate without S9 and 0.016–5 "
            "mg/plate with S9; no increases in mean revertant colonies at any dose with or "
            "without S9 (RIFM, 2000a,b).",
            "confidence": 0.95,
            "needs_more_context": False,
            "pmid": "28743524",
            "extraction_id": "ext_1",
            "SMILES": "CC1=CCC(O)(C(C)C)CC1",
        },
        "cited_or_previous_experiment_requires_provenance_review",
    ),
    (
        "ames_base:158909",
        {
            "paragraph_idx": 46,
            "support_text": "Mutagenicity of 3-carene in S. typhimurium TA98, TA100 and TA102 without "
            "metabolic activation was measured by the conventional plate incorporation "
            "(indirect) Ames test (means and standard deviations shown). In the conclusions "
            "the authors state that one of the mutagens released into the air by drying "
            "spruce and birch chips at 170°C was probably 3-carene, although 3-carene could "
            "explain only a small fraction of the total mutagenic responses.",
            "molecule_name": "3-carene",
            "chemical_entity_type": "defined_chemical",
            "mutagenicity_result": "positive",
            "evidence_basis": "current_study_experiment",
            "assay_family": "bacterial_reverse_mutation",
            "experimental_context": "in_vitro",
            "test_system": "Salmonella Typhimurium TA98, TA100, TA102",
            "metabolic_activation": "absent",
            "qualifying_conditions": None,
            "extra_details": "Tested without metabolic activation by the conventional plate incorporation "
            "Ames test; means and standard deviations presented. Conclusions indicate "
            "3-carene was probably one of the mutagens in drying fumes from spruce and "
            "birch chips at 170°C, but explains only a small fraction of the total "
            "mutagenic responses, i.e., it was not the sole mutagen. Paper code/label "
            "'3-carene' corresponds to (-)-3-carene. The active compounds were in the gas "
            "phase and gave a slight positive TA102 response for the drying fumes from "
            "birch chips.",
            "confidence": 0.8,
            "needs_more_context": False,
            "pmid": "2202897",
            "extraction_id": "ext_9",
            "SMILES": "CC1=CCC2C(C1)C2(C)C",
        },
        "inferred_subject_or_outcome_requires_review",
    ),
    (
        "ames_base:9584",
        {
            "paragraph_idx": 150,
            "support_text": "The mutagenic activity of Benznidazole 2 was studied in a plate incorporation "
            "assay using Salmonella typhimurium TA100 under aerobic or anaerobic conditions "
            "with/without addition of liver extracts. Benznidazole 2 showed significant "
            "mutagenic activity in the TA100 strain under both aerobic and anaerobic "
            "conditions; addition of liver enzyme did not alter the effects.",
            "molecule_name": "Benznidazole",
            "chemical_entity_type": "defined_chemical",
            "mutagenicity_result": "positive",
            "evidence_basis": "current_study_experiment",
            "assay_family": "bacterial_reverse_mutation",
            "experimental_context": "in_vitro",
            "test_system": "Salmonella typhimurium TA100",
            "metabolic_activation": "both",
            "qualifying_conditions": None,
            "extra_details": "Plate incorporation Ames assay; significant mutagenic activity under both "
            "aerobic and anaerobic conditions; added rat liver enzyme (S9/liver extract) "
            "did not alter the effect. Buschini et al. additionally found Benznidazole "
            "more active for base-pair substitution than frameshift induction in "
            "Salmonella strains TA100, TA100NR, TA98 and TA98NR.",
            "confidence": 0.9,
            "needs_more_context": False,
            "pmid": "35631389",
            "extraction_id": "ext_10",
            "SMILES": "O=C(Cn1ccnc1[N+](=O)[O-])NCc1ccccc1",
        },
        "cited_or_previous_experiment_requires_provenance_review",
    ),
    (
        "ames_base:122412",
        {
            "paragraph_idx": 54,
            "support_text": "The potential of 6-hydroxyindole to induce gene mutation was studied in "
            "Salmonella typhimurium strains TA 98, TA 100, TA 1535, TA 1537, and TA 1538 "
            "and in Escherichia coli strain WP2 uvrA using the reverse mutation assay, with "
            "and without S9 metabolic activation at concentrations of 8, 40, 200, 1000, or "
            "5000 µg/plate. A dose-dependent increase in revertant colony numbers was "
            "observed with and without S9 in TA 1535, and it was concluded that "
            "6-hydroxyindole was mutagenic with and without metabolic activation in S "
            "typhimurium strain TA 1535.",
            "molecule_name": "6-hydroxyindole",
            "chemical_entity_type": "defined_chemical",
            "mutagenicity_result": "positive",
            "evidence_basis": "current_study_experiment",
            "assay_family": "bacterial_reverse_mutation",
            "experimental_context": "in_vitro",
            "test_system": "Salmonella typhimurium strains TA98, TA100, TA1535, TA1537, TA1538; Escherichia "
            "coli WP2 uvrA (Ames/reverse mutation)",
            "metabolic_activation": "both",
            "qualifying_conditions": None,
            "extra_details": "Concentrations 8, 40, 200, 1000, or 5000 µg/plate; dose-dependent increase in "
            "revertants observed with and without S9 in strain TA1535 only; other tested "
            "strains did not show this response.",
            "confidence": 0.95,
            "needs_more_context": False,
            "pmid": "25297906",
            "extraction_id": "ext_1",
            "SMILES": "Oc1ccc2cc[nH]c2c1",
        },
        "explicit_primary_bacterial_outcome",
    ),
    (
        "ames_base:312551",
        {
            "paragraph_idx": 9,
            "support_text": "Of the volatile chemicals, 3-vinyl-7-oxabicyclo[4.1.0]heptane and "
            "3-epoxyethyl-7-oxabicyclo[4.1.0]heptane were mutagenic in all strains. "
            "3-Vinyl-7-oxabicyclo[4.1.0]heptane was the most potent mutagen and did not "
            "require S9 mix for activity.",
            "molecule_name": "3-Vinyl-7-oxabicyclo[4.1.0]heptane",
            "chemical_entity_type": "defined_chemical",
            "mutagenicity_result": "positive",
            "evidence_basis": "current_study_experiment",
            "assay_family": "bacterial_reverse_mutation",
            "experimental_context": "in_vitro",
            "test_system": "Salmonella typhimurium TA1535, TA98 and TA100",
            "metabolic_activation": "both",
            "qualifying_conditions": None,
            "extra_details": "Volatile compound tested by exposure in sealed desiccators for 7 h, then "
            "plates incubated 37 C for 40-50 h. Most potent mutagen of those tested; did "
            "not require S9 for activity. Without S9, TA1535 rose from 45 to 387 (1.0 ml), "
            "TA100 from 134 to 1100 (1.0 ml); TA98 essentially unaffected without S9 but "
            "rose with S9 (from 43 to 95). With S9, TA1535 to 434, TA100 to 1340.",
            "confidence": 0.95,
            "needs_more_context": False,
            "pmid": "7001215",
            "extraction_id": "ext_8",
            "SMILES": "C=CC1CCC2OC2C1",
        },
        "special_exposure_requires_condition_review",
    ),
]


@pytest.mark.parametrize("record_id,row,reason", INDEPENDENT_AUDIT_EXAMPLES)
def test_independent_audit_dispositions_preserve_panel_any_positive(
    record_id, row, reason
):
    decision = classify("ames_base", row)
    if record_id == "ames_base:122412":
        # The TA1535 response supports one positive for the tested six-strain
        # panel; it does not grant positive labels to the five other strains.
        assert decision.group == DIRECT and decision.label == 1
        assert decision.condition_atoms == (
            "metabolic_activation=both_reported",
            "strain_panel=TA100,TA1535,TA1537,TA1538,TA98,WP2UVRA",
        )
    else:
        assert decision.group == "" and decision.label is None, record_id
    assert decision.reason == reason, record_id
