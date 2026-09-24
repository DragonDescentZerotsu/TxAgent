"""Portable DILI v5 contract tests; no TxAgent runtime imports."""

import pytest

from dili_score import score_dili_record


@pytest.mark.parametrize("record", [None, [], "text", 1])
def test_invalid_record_has_a_clear_error(record):
    with pytest.raises(TypeError, match="record must be a mapping"):
        score_dili_record(record)


@pytest.mark.parametrize(
    "record", [{"input": None}, {"input": {}}, {"input": {"source_fields": []}}]
)
def test_invalid_prepared_record_has_a_clear_error(record):
    with pytest.raises(TypeError, match="input.source_fields must be a mapping"):
        score_dili_record(record)


def test_invalid_condition_has_a_clear_error():
    with pytest.raises(TypeError, match="condition must be a string"):
        score_dili_record({}, condition=None)


def test_empty_record_is_json_serializable_and_has_no_information_credit():
    import json

    q = score_dili_record({})
    assert q["score"] == 0
    assert q["result_content_status"] == "missing"
    assert json.loads(json.dumps(q)) == q


def dili_row(**fields):
    return dict(
        canonical_endpoint_name="ROS production",
        biological_system="human hepatocytes",
        canonical_measurement_text="decreased",
        support_text="ROS decreased.",
        **fields,
    )


def test_dili_simple_steatohepatitis_is_explicit_liver_endpoint():
    row = dict(
        clinical_phenotype="steatosis_or_steatohepatitis",
        causal_status="established_or_definite",
        support_text="Clinical phenotype reported.",
    )
    assert score_dili_record(row)["components"]["relevance"] == 1
    # Generic lipid accumulation alone does not establish a liver endpoint.
    assert (
        score_dili_record(
            {
                **row,
                "clinical_phenotype": "steatosis",
                "biological_system": "skeletal muscle",
            }
        )["components"]["relevance"]
        == 0.5
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "FXR protein abundance",
        "HMGB1 translocation",
        "intracellular acidosis",
        "proliferation_or_regenerative_reserve",
        "nucleophile_trapping",
        "nucleophilic trapping",
        "Formation of emodin-cysteine adducts",
        "HMGB-1 release",
        "CYP1A2 time-dependent inhibition",
        "P450 inactivation",
        "cellular_stress_homeostasis",
        "xenobiotic_metabolic_or_transport_function",
    ],
)
def test_dili_simple_readout_repairs_require_hepatic_context(endpoint):
    row = {**dili_row(), "canonical_endpoint_name": endpoint}
    assert score_dili_record(row)["components"]["relevance"] == 1
    assert (
        score_dili_record({**row, "biological_system": "HeLa cells"})["components"][
            "relevance"
        ]
        == 0.5
    )


def test_dili_simple_prose_only_result_is_explicitly_a_structured_field_gap():
    row = {
        **dili_row(),
        "canonical_measurement_text": None,
        "support_text": "Emodin induced a dose-dependent loss of MMP in HepaRG cells.",
    }
    q = score_dili_record(row)
    assert q["score"] == 0.75
    assert q["completeness_basis"] == "structured_fields_and_support_presence"
    assert q["completeness_evidence"]["result"] == []
    assert q["completeness_evidence"]["support"] == ["support_text"]
    assert q["completeness_evidence"]["assay_or_system"] == ["biological_system"]
    # Background hepatic mentions cannot supply the trapping assay's system.
    row.update(
        canonical_endpoint_name="nucleophile trapping",
        biological_system="buffer",
        canonical_measurement_text="adduct detected",
    )
    assert score_dili_record(row)["components"]["relevance"] == 0.5


@pytest.mark.parametrize(
    "system",
    [
        "non-hepatic cells",
        "without hepatocytes",
        "no liver model",
        "HeLa cells, not hepatocytes",
        "not a liver model",
    ],
)
def test_dili_simple_negated_tissue_is_not_hepatic_context(system):
    q = score_dili_record({**dili_row(), "biological_system": system})
    assert q["components"]["relevance"] == 0.5


@pytest.mark.parametrize(
    "system",
    [
        "In vitro metabolic system (liver microsomal/vitro system implied, source not stated in window)",
        "presumed liver microsomes",
        "hepatic origin inferred",
    ],
)
def test_dili_simple_inferred_system_does_not_establish_hepatic_scope(system):
    row = {
        **dili_row(),
        "canonical_endpoint_name": "nucleophile trapping",
        "biological_system": system,
    }
    assert score_dili_record(row)["components"]["relevance"] == 0.5
    assert (
        score_dili_record({**row, "biological_model": "HepG2 cells"})["components"][
            "relevance"
        ]
        == 1
    )


@pytest.mark.parametrize(
    "system",
    [
        "isolated rat livers",
        "HepG2 cells",
        "Hep3B cells",
        "LO2 cells",
        "NAFLD-to-NASH model",
    ],
)
def test_dili_simple_system_aliases_preserve_real_hepatic_context(system):
    assert (
        score_dili_record({**dili_row(), "biological_system": system})["components"][
            "relevance"
        ]
        == 1
    )


@pytest.mark.parametrize(
    "endpoint,system",
    [
        ("air-labile reactive metabolite trapping", "buffer system"),
        ("mitochondrial labile iron and ROS", "cell type not specified"),
        ("cholesterol degradation", "cell-free LDL"),
        ("non-hepatic cell death", "HeLa cells"),
    ],
)
def test_dili_simple_substrings_do_not_invent_liver_or_bile_scope(endpoint, system):
    q = score_dili_record(
        {**dili_row(), "canonical_endpoint_name": endpoint, "biological_system": system}
    )
    assert q["components"]["relevance"] == 0.5


def test_dili_simple_support_does_not_rescue_offtask_or_unrecognized_readout():
    for endpoint, relevance in [("renal toxicity", 0), ("unrecognized readout", 0.5)]:
        q = score_dili_record(
            {
                **dili_row(),
                "canonical_endpoint_name": endpoint,
                "support_text": "Background: FXR and liver injury were studied elsewhere.",
            }
        )
        assert q["components"]["relevance"] == relevance


def test_dili_simple_clinical_severity_resolves_generic_phenotype():
    row = {
        **dili_row(),
        "canonical_endpoint_name": "clinical_phenotype=multiple_phenotypes",
        "clinical_phenotype": "multiple_phenotypes",
        "maximum_reported_severity": "acute_liver_failure",
    }
    assert score_dili_record(row)["components"]["relevance"] == 1
    assert (
        score_dili_record({**row, "maximum_reported_severity": None})["components"][
            "relevance"
        ]
        == 0.5
    )


def test_dili_simple_placeholders_and_metadata_are_not_information():
    q = score_dili_record(
        dict(
            canonical_endpoint_name="missing_endpoint",
            canonical_measurement_text="result_value=not_reported",
            support_text="unknown",
            canonical_species_context="species=null",
            canonical_assay_context="human_evidence_basis=review | exposure_context=therapeutic_use",
        )
    )
    assert q["score"] == 0
    assert not any(q["completeness_checks"].values())
    assert score_dili_record({**dili_row(), "canonical_measurement_text": 0})[
        "completeness_checks"
    ]["result"]


@pytest.mark.parametrize(
    "metadata",
    [
        "quantitative_measure_type=percent_of_control",
        "statistic_type=mean | result_unit=ug/mL",
    ],
)
def test_dili_simple_result_metadata_alone_does_not_count(metadata):
    row = {**dili_row(), "canonical_measurement_text": metadata}
    assert score_dili_record(row)["score"] == 0.75
    assert (
        score_dili_record(
            {**row, "canonical_measurement_text": metadata + " | quantitative_value=0"}
        )["score"]
        == 1
    )
    assert (
        score_dili_record(
            {
                **row,
                "canonical_measurement_text": metadata
                + " | effect_direction=no_change",
            }
        )["score"]
        == 1
    )


@pytest.mark.parametrize(
    "description",
    [
        "Apoptotic rate quantified by flow cytometry (numerical apoptosis values reported in results section not present in this window)",
        "A520 monitored in CaCl2-challenged rat myocardial mitochondria; effect of emodin (10 µM) on swelling/permeability transition depicted graphically in Fig. 7 (numerical value not stated in text)",
        "ROS measured by flow cytometry at 10 uM for 24 h",
        "See Fig. 7 for results",
        "Flow cytometry",
        "Viability assessed to determine whether treatment increased cell death",
        "Apoptosis measured after increased dose",
        "ROS production",
        "Effect shown in figure",
    ],
)
def test_dili_simple_method_or_figure_only_result_is_incomplete(description):
    for measurement in (description, "result_value=" + description):
        q = score_dili_record({**dili_row(), "canonical_measurement_text": measurement})
        assert q["score"] == 0.75
        assert not q["completeness_checks"]["result"]
        assert q["excluded_result_descriptions"]
        assert "result_description_without_reported_outcome" in q["reasons"]


@pytest.mark.parametrize(
    "result",
    [
        "ROS decreased, measured by flow cytometry",
        "No significant change detected",
        "No measurable increase in ROS",
        "Apoptosis was not detected",
        "change in mitochondrial membrane potential detected by JC-1 (direction not stated in passage)",
        "IC50 = 10 uM, measured in cells",
        "40% viability measured by flow cytometry",
        "Apoptosis increased (numerical values not reported)",
        "significant group differences observed (P<0.01), direction shown in figure",
        "GSH conjugate detected by mass spectrometry",
        "ADP/O ratios measured with succinate were equivalent to control values",
        "very weak inhibitor for palmitic acid activation (IC50 not determined)",
        "No covalent modification of COX enzymes detected",
        "not cytotoxic (see Fig. 2)",
        "no significant changes detected within 3 h",
        "P450 isoforms showed differential catalytic capability, quantified by conjugate formation",
        "down-regulation of ATP synthase restrained mitochondrial ATP production (Figure 3C)",
        "Recombinant P450 enzymes catalyzed formation of metabolites (capability assessed; no numeric values given in figure legend)",
        "decrease of respiration and increase of proton leakage (Figure 3C)",
        "0",
        "negative",
        "no_change",
    ],
)
def test_dili_simple_reported_results_survive_description_screen(result):
    q = score_dili_record(
        {**dili_row(), "canonical_measurement_text": "result_value=" + result}
    )
    assert q["score"] == 1
    assert q["completeness_checks"]["result"]


def test_dili_simple_independent_result_rescues_method_description():
    q = score_dili_record(
        {
            **dili_row(),
            "canonical_measurement_text": "result_value=Measured by flow cytometry | effect_direction=no_change",
        }
    )
    assert q["score"] == 1
    assert q["excluded_result_descriptions"]
    assert "result_description_without_reported_outcome" not in q["reasons"]


@pytest.mark.parametrize(
    "result",
    [
        "CPT1 activity was lower in burned rats than controls and partially reversed by carnitine",
        "EC50-MT glucose >20 uM, galactose >20 uM (not determinable)",
        "statistically significant toxicity in both LDH release and viable cell protein assays",
        "cytotoxicity observed in the Glu/Gal assay",
        "altered cell morphology",
        "basal OCR lower than all other measured groups",
        "altered vs FFA",
        "furosemide depleted hepatocellular sulfhydryls",
        "no measurable covalent binding",
        "HNE-SG identified/quantified up to 45 min",
        "IC50 expediently set as 4× the tested concentration",
        "determined concentration 75.5 umol/L, 2.4 ug S/mL",
    ],
)
def test_dili_v5_outcomes_are_stable_under_citation_and_method_suffixes(result):
    for suffix in [
        "",
        " (Fig. 3A)",
        " (qualitative, Fig. 4)",
        " (figure quantitative data)",
        "; measured by flow cytometry",
    ]:
        q = score_dili_record(
            {
                **dili_row(),
                "canonical_measurement_text": "result_value=" + result + suffix,
            }
        )
        assert q["score"] == 1
        assert q["result_content_status"] == "reported"


@pytest.mark.parametrize(
    "description",
    [
        "Percent cell viability versus control reported; IC50 graphically determined (numeric value not stated in the provided window)",
        "Bile samples were collected and prepared for GSH adduct analysis by UPLC-HRMS; adduct-identification outcome is reported in the results section, not in this methods window",
        "Complex I and III activity measured in liver following danthron treatment (n=5)",
        "EC50 generated from concentration-response curve (specific value shown in Fig. 2)",
    ],
)
def test_dili_v5_preparation_and_result_pointers_do_not_supply_an_outcome(description):
    q = score_dili_record(
        {**dili_row(), "canonical_measurement_text": "reported_result=" + description}
    )
    assert q["score"] == 0.75
    assert q["result_content_status"] == "description_only"


def test_dili_v5_unfamiliar_prose_is_explicitly_unresolved_and_citation_stable():
    for suffix in ["", " (Fig. 3A)"]:
        q = score_dili_record(
            {
                **dili_row(),
                "canonical_measurement_text": "unclassified response pattern" + suffix,
            }
        )
        assert q["score"] == 1  # Presence credit, not a claim of semantic verification.
        assert q["result_content_status"] == "unresolved"
        assert q["unresolved_result_descriptions"]


@pytest.mark.parametrize(
    "result",
    [
        "depolarization of mitochondrial membrane potential (qualitative, no numerical value reported)",
        "weak inhibition of beta-oxidation (no IC50 reported)",
        "substantially inactive (no quantitative value reported)",
        "GSSG generation reported (oxidative stress endpoint)",
        "disruptive effect on respiratory complexes (proposed from measured ROS/ATP endpoints)",
    ],
)
def test_dili_v5_unknown_qualitative_prose_keeps_presence_credit(result):
    q = score_dili_record(
        {
            **dili_row(),
            "reported_result": result,
            "canonical_measurement_text": "reported_result=" + result,
        }
    )
    assert q["score"] == 1
    assert q["result_content_status"] == "unresolved"


@pytest.mark.parametrize(
    "result",
    [
        "ATP content measured at 3, 7 and 14 days; specific findings not stated",
        "cell viability reported in Fig. 2A; direction not stated in text",
        "outcome for treated groups not stated (figure legend only; significance symbols *,**; #,##; $ noted)",
        "No quantitative result is provided within this window (result expected in the Results section)",
    ],
)
def test_dili_v5_method_fragments_do_not_become_outcomes_on_recovery(result):
    row = {**dili_row(), "canonical_measurement_text": "result_value=" + result}
    for record in (
        row,
        {"input": {"source_fields": dict(row), "support_text": row["support_text"]}},
    ):
        q = score_dili_record(record)
        assert q["score"] == 0.75
        assert q["result_content_status"] == "description_only"


@pytest.mark.parametrize(
    "result",
    [
        "DCVG present at 5 nmol in 7 ml bile collected over 9 h",
        "WP1130 caused collapse of membrane potential (qualitative), assessed by flow cytometry",
        "Cell viability measured; P<0.05 vs control",
        "NA (low effect on NO production); no half-maximal inhibitory concentration determined",
        "dose-dependent toxicity; TC50 calculated",
        "OCR and ECAR measured; treated parasites fell into the ETC-inhibitor pattern",
        "mitochondrial respiration significantly different from control; numeric value given only in figure",
        "2,4-diene-VPA calculated 10-fold more potent than VPA",
    ],
)
def test_dili_v5_independent_outcome_survives_method_clause(result):
    q = score_dili_record(
        {**dili_row(), "canonical_measurement_text": "result_value=" + result}
    )
    assert q["score"] == 1
    assert q["result_content_status"] == "reported"


def test_dili_v5_semicolon_separates_typed_fields_but_not_prose():
    row = {
        **dili_row(),
        "canonical_measurement_text": "quantitative_measure_type=IC50; result_value=5; result_unit=uM",
    }
    assert score_dili_record(row)["score"] == 1
    row["canonical_measurement_text"] = (
        "result_value=measured by flow cytometry; effect_direction=decrease"
    )
    assert score_dili_record(row)["score"] == 1


def test_dili_v5_hepatic_identity_qualification_and_transport_coverage():
    q = score_dili_record(
        {
            **dili_row(),
            "biological_system": "intact cells (hepatocyte identity and species not confirmed in the provided window)",
        }
    )
    assert q["components"]["relevance"] == 0.5
    row = {
        **dili_row(),
        "canonical_endpoint_name": "assay_category=transporter_substrate_or_kinetics | endpoint_metric=Km of initial sinusoidal uptake velocity",
    }
    assert score_dili_record(row)["components"]["relevance"] == 1
    assert (
        score_dili_record(
            {**row, "biological_system": "recombinant transporter vesicles"}
        )["components"]["relevance"]
        == 0.5
    )


def test_dili_v5_liver_disease_background_does_not_identify_measured_tissue():
    row = {
        **dili_row(),
        "canonical_endpoint_name": "Aortic CYP2J4 gene expression",
        "biological_system": "cirrhotic rats; aorta tissue; liver-stress context",
    }
    assert score_dili_record(row)["components"]["relevance"] == 0.5
    row["biological_system"] += "; paired liver tissue measurement"
    assert score_dili_record(row)["components"]["relevance"] == 1


@pytest.mark.parametrize(
    "qualifier",
    [
        "hepatic metabolic competence not stated",
        "hepatic competence is not reported",
        "hepatic metabolic competence unknown",
    ],
)
def test_dili_simple_missing_competence_is_not_tissue_identity(qualifier):
    row = {**dili_row(), "biological_system": "RAW 264.7 macrophages; " + qualifier}
    assert score_dili_record(row)["components"]["relevance"] == 0.5
    assert (
        score_dili_record({**row, "biological_model": "primary liver cells"})[
            "components"
        ]["relevance"]
        == 1
    )


def test_dili_simple_explicit_cell_identity_disclaimer_limits_relevance():
    row = {
        **dili_row(),
        "biological_system": "HL-7702, misidentified HeLa derivative, not an authenticated normal human hepatocyte model",
    }
    assert score_dili_record(row)["components"]["relevance"] == 0.5
    assert "source_disclaims_hepatic_cell_identity" in score_dili_record(row)["reasons"]
    assert (
        score_dili_record(
            {**dili_row(), "biological_system": "isolated soybean cotyledons"}
        )["components"]["relevance"]
        == 0.25
    )


def test_dili_simple_canonical_wrappers_preserve_native_information():
    row = dict(
        canonical_endpoint_name="clinical_phenotype=multiple_phenotypes",
        canonical_measurement_text="maximum_reported_severity=acute_liver_failure",
        canonical_assay_context="biological_system=human patients",
        support_text="Source report.",
    )
    native = {
        **row,
        "clinical_phenotype": "multiple_phenotypes",
        "maximum_reported_severity": "acute_liver_failure",
        "biological_system": "human patients",
    }
    assert score_dili_record(row)["score"] == score_dili_record(native)["score"] == 1
    assert (
        score_dili_record(
            {**native, "reviewed_record_json": {"maximum_reported_severity": None}}
        )["components"]["relevance"]
        == 0.5
    )
    row = dict(
        canonical_endpoint_name="assay_category=mitochondrial_respiration | assay_detail=oxygen consumption",
        canonical_measurement_text="reported_result=no_change",
        support_text="Source observation.",
    )
    assert (
        score_dili_record(row)["score"]
        == score_dili_record({**row, "assay_detail": "oxygen consumption"})["score"]
    )


def test_dili_simple_equals_in_prose_is_not_a_field_wrapper():
    row = {
        **dili_row(),
        "biological_system": None,
        "canonical_assay_context": "biological_system_context=RNA from livers of controls (n=3) and treated mice",
    }
    assert score_dili_record(row)["components"]["relevance"] == 1


def test_dili_simple_review_flags_roles_labels_and_sign_do_not_change_score():
    row = dili_row()
    original = score_dili_record(row)
    for extra in (
        {"needs_more_context": True},
        {"chemical_entity_type": "defined_chemical"},
        {"Y": 1, "level": 7, "family_id": "x"},
        {"canonical_measurement_text": "increased"},
    ):
        assert score_dili_record({**row, **extra})["score"] == original["score"]
    assert score_dili_record({**row, "support_text": None})["score"] < original["score"]


@pytest.mark.parametrize(
    "fields,condition",
    [
        ({"age": 71}, "age_group=older_than_65"),
        ({"qualifying_conditions": "71-year-old woman"}, "age_group=older_than_65"),
        ({"population_context": "hepatitis_b"}, "population_context=hepatitis_b"),
        ({"regimen": "single_dose_or_one_day"}, "regimen=single_dose_or_one_day"),
        (
            {"qualifying_conditions": "age=71 | population_context=hepatitis_b"},
            "age_group=older_than_65+population_context=hepatitis_b",
        ),
    ],
)
def test_dili_simple_preserves_typed_conditions(fields, condition):
    row = {**dili_row(), **fields}
    q = score_dili_record(row, condition=condition)
    assert q["condition"]["status"] == "matched"
    assert q["score"] == score_dili_record(row)["score"]


def test_dili_simple_condition_unknown_mismatch_and_reviewed_null():
    import json

    row = dili_row(age=71)
    assert (
        score_dili_record({**row, "age": 31}, condition="age_group=older_than_65")[
            "condition"
        ]["status"]
        == "mismatched"
    )
    corrected = {**row, "reviewed_record_json": json.dumps({"age": None})}
    assert (
        score_dili_record(corrected, condition="age_group=older_than_65")["condition"][
            "status"
        ]
        == "unknown"
    )
    canonical_correction = {
        **row,
        "reviewed_record_json": json.dumps({"canonical_endpoint_name": None}),
    }
    assert not score_dili_record(canonical_correction)["completeness_checks"][
        "endpoint"
    ]
    disease = {**row, "population_context": "hepatitis_c"}
    assert (
        score_dili_record(disease, condition="population_context=hepatitis_b")[
            "condition"
        ]["status"]
        == "unknown"
    )
    assert (
        score_dili_record(
            {**row, "age": None, "dose": "71 mg/kg"},
            condition="age_group=older_than_65",
        )["condition"]["status"]
        == "unknown"
    )
    for condition in (
        "age_group=",
        "not-a-condition",
        "Y=1",
        "age_group=pediatric+age_group=older_than_65",
    ):
        with pytest.raises(ValueError):
            score_dili_record(row, condition=condition)


def test_dili_simple_rat_age_is_not_a_pediatric_patient_condition():
    q = score_dili_record(
        {**dili_row(), "age": 2, "biological_system": "rat hepatocytes"},
        condition="age_group=pediatric",
    )
    assert q["condition"]["status"] == "unknown"
