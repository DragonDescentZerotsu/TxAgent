import copy
import json

import pytest

from optimization.record_scoring import VERSIONS, score_record


@pytest.mark.parametrize("task,endpoint,context,tier", [
    ("bbb_martins", "BBB permeability outcome", "rat in vivo", 1),
    ("bbb_martins", "logBB", "human", 1),
    ("bbb_martins", "no BBB penetration", "rat", 1),
    ("bbb_martins", "Kp_uu", "rat", 1),
    ("bbb_martins", "brain_to_plasma_ratio", "rat", 1),
    ("bbb_martins", "blood-to-brain transport", "in situ brain perfusion", 1),
    ("bbb_martins", "BBB permeability", "PAMPA-BBB in vitro", .75),
    ("bbb_martins", "Papp", "hCMEC/D3 monolayer", .75),
    ("bbb_martins", "efflux_ratio", "brain microvascular endothelial cells", .75),
    ("bbb_martins", "brain concentration", "rat", .75),
    ("bbb_martins", "CSF_to_plasma_ratio", "rat", .75),
    ("bbb_martins", "efflux inhibition outcome", "recombinant ABCB1", .5),
    ("bbb_martins", "Papp", "MDCK", .5),
    ("bbb_martins", "Papp", "Caco-2 monolayer", .25),
    ("bbb_martins", "brain concentration", "intracerebral injection", .25),
    ("bbb_martins", "logP", "rat", .25),
    ("bbb_martins", "cell viability", "brain endothelial cells", 0),
    ("bbb_martins", "Ames assay", "Salmonella", 0),
    ("ames", "bacterial_reverse_mutation", "Salmonella TA98 + S9", 1),
    ("ames", "no bacterial reverse mutation", "Salmonella TA98", 1),
    ("ames", "bacterial mutagenicity", "bacterial system unspecified", .5),
    ("ames", "Ames test", "Salmonella TA100 - S9", 1),
    ("ames", "OECD TG 471", "E. coli WP2 uvrA", 1),
    ("ames", "revertant colonies", "Salmonella TA1535", 1),
    ("ames", "mammalian_cell_gene_mutation", "CHO", .75),
    ("ames", "hprt_hgprt_xprt_forward_mutation", "V79", .75),
    ("ames", "tk_mouse_lymphoma", "L5178Y", .75),
    ("ames", "sequencing_based_induced_mutation", "rat", .75),
    ("ames", "micronucleus", "human lymphocytes", .5),
    ("ames", "comet_assay", "rat hepatocytes", .5),
    ("ames", "targeted_dna_adduct_measurement", "human", .5),
    ("ames", "bacterial_ddr_reporter", "Salmonella", .5),
    ("ames", "unspecified_mutagenicity", "rat", .5),
    ("ames", "ROS_or_redox_assay", "human", .25),
    ("ames", "metabolite_formation_or_reaction_phenotyping", "rat", .25),
    ("ames", "antioxidant_or_thiol_assay", "rat", .25),
    ("ames", "BBB penetration", "mouse", 0),
    ("carcinogens", "carcinogenicity", "animal bioassay", 1),
    ("carcinogens", "cancer incidence", "cohort", 1),
    ("carcinogens", "no increase in cancer incidence", "cohort", 1),
    ("carcinogens", "lung carcinoma", "animal carcinogenicity bioassay", 1),
    ("carcinogens", "tumor multiplicity", "mouse", 1),
    ("carcinogens", "bacterial_reverse_mutation", "Salmonella", .75),
    ("carcinogens", "micronucleus", "human lymphocytes", .75),
    ("carcinogens", "anchorage_independent_growth", "soft agar", .75),
    ("carcinogens", "morphological_cell_transformation", "BALB 3T3", .75),
    ("carcinogens", "xenograft tumor formation", "transformed cells", .75),
    ("carcinogens", "DNA_methylation", "human", .5),
    ("carcinogens", "proliferation or DNA synthesis", "human", .5),
    ("carcinogens", "DNA methylation in leukocytes", "human cohort", .5),
    ("carcinogens", "DNA adducts in cancer patients", "human cohort", .75),
    ("carcinogens", "oncogene expression", "human cohort", .5),
    ("carcinogens", "receptor_mediated_growth", "human", .5),
    ("carcinogens", "injury_driven_regeneration", "rat", .5),
    ("carcinogens", "lung cancer", "background patient population", .5),
    ("carcinogens", "lipid_peroxidation", "rat", .25),
    ("carcinogens", "cell viability or survival", "human", .25),
    ("carcinogens", "tumor regression", "human", 0),
    ("carcinogens", "xenograft tumor growth", "human", 0),
    ("carcinogens", "BBB permeability", "rat", 0),
])
def test_task_relevance(task, endpoint, context, tier):
    row = dict(endpoint=endpoint, biological_system=context, reported_result=0, support_text="Source passage")
    result = score_record(row, task=task)
    assert result["components"] == {"relevance": tier, "completeness": 1}
    assert result["score"] == tier
    assert result["version"] == VERSIONS[task]


@pytest.mark.parametrize("task,endpoint", [("bbb_martins", "BBB penetration"),
                                          ("ames", "Ames"), ("carcinogens", "carcinogenicity")])
def test_shared_input_completeness_and_polarity(task, endpoint):
    row = dict(endpoint=endpoint, reported_result="positive", biological_system="rat", support_text="Source")
    original = copy.deepcopy(row)
    assert score_record(row, task=task)["score"] == 1
    row["reported_result"] = "negative"
    assert score_record(row, task=task)["score"] == 1
    row["reported_result"] = "see Figure 2"
    result = score_record(row, task=task)
    assert result["score"] == .75
    assert result["result_content_status"] == "description_only"
    prepared = {"input": {"source_fields": original, "support_text": "Source"}}
    assert score_record(prepared, task=task)["score"] == 1
    assert score_record({"raw_record_json": json.dumps(original)}, task=task)["score"] == 1
    corrected = dict(original, canonical_endpoint_concept=endpoint,
                     reviewed_record_json={"endpoint": None, "reported_result": None})
    result = score_record(corrected, task=task)
    assert result["components"] == {"relevance": .5, "completeness": .5}
    assert score_record({"endpoint": endpoint, "support_text": "positive in rats"}, task=task)["score"] == .5
    assert score_record({"source_id": endpoint, "molecule_name": endpoint, "support_text": endpoint}, task=task)["components"]["relevance"] == .5
    assert original == dict(endpoint=endpoint, reported_result="positive", biological_system="rat", support_text="Source")


@pytest.mark.parametrize("task,field,value,context", [
    ("bbb_martins", "bbb_permeability_label", "impermeable", "rat"),
    ("ames", "mutagenicity_result", "negative", "Ames assay in TA98"),
    ("carcinogens", "carcinogenicity_conclusion", "negative", "animal bioassay"),
    ("carcinogens", "classification_label", "Group 3", "authoritative classification"),
])
def test_named_task_conclusions(task, field, value, context):
    result = score_record({field: value, "biological_system": context}, task=task)
    assert result["components"]["relevance"] == 1
    assert result["result_content_status"] == "reported"


@pytest.mark.parametrize("task,endpoint", [("bbb_martins", "BBB penetration"),
                                          ("ames", "Ames"), ("carcinogens", "carcinogenicity")])
def test_ambiguity_and_predictions(task, endpoint):
    result = score_record({"endpoint": "predicted " + endpoint}, task=task)
    assert result["components"]["relevance"] == .5
    assert "predicted_evidence" in result["ambiguity_flags"]
    assert score_record({"endpoint": endpoint + " (implied)"}, task=task)["components"]["relevance"] == .5
    assert score_record({"endpoint": endpoint + "; antibacterial activity"}, task=task)["components"]["relevance"] == .5
    assert score_record({"endpoint": endpoint, "endpoint_detail": "antibacterial activity"}, task=task)["components"]["relevance"] == 1


def test_uncertain_system_and_treatment_context():
    assert score_record({"endpoint": "Papp", "biological_system": "BBB origin not established"}, task="bbb_martins")["components"]["relevance"] == .5
    result = score_record({"endpoint": "tumor incidence", "study_context": "therapeutic xenograft study"}, task="carcinogens")
    assert result["components"]["relevance"] == .5
    assert "conflicting_task_context" in result["ambiguity_flags"]
    assert not score_record({"endpoint": "Ames", "study_context": "dose 10 mg/kg"}, task="ames")["completeness_checks"]["assay_or_system"]
    assert score_record({"canonical_endpoint_concept": "bbb_permeability_outcome", "bbb_permeability_label": "predicted permeable"}, task="bbb_martins")["components"]["relevance"] == .5


@pytest.mark.parametrize("task,fields,tier", [
    ("bbb_martins", {"quant_metric": "logBB", "quant_value": 0}, 1),
    ("bbb_martins", {"metric_name": "Pe", "metric_value": 0}, .75),
    ("bbb_martins", {"quantitative_metric": "efflux ratio", "quantitative_value": 0}, .75),
    ("bbb_martins", {"transport_endpoint": "blood-to-brain transport", "reported_result": 0}, 1),
    ("ames", {"assay_family": "bacterial_reverse_mutation", "mutagenicity_result": "negative"}, 1),
    ("ames", {"assay_family": "hprt_forward_mutation", "response_value": 0}, .75),
    ("ames", {"assay_family": "comet", "response_value": 0}, .5),
    ("ames", {"assay_family": "ros", "quantitative_readout_value": 0}, .25),
    ("carcinogens", {"assay_family": "micronucleus", "result_value": 0}, .75),
    ("carcinogens", {"endpoint_category": "glutathione_depletion", "reported_result": 0}, .25),
    ("carcinogens", {"assay_domain": "dna_methylation", "endpoint_result": 0}, .5),
    ("carcinogens", {"phenotype_domain": "receptor_mediated_growth", "endpoint_result": 0}, .5),
])
def test_native_source_aliases(task, fields, tier):
    system = "PAMPA-BBB" if task == "bbb_martins" and tier == .75 else "rat in vivo"
    row = {**fields, "biological_system": system, "support_text": "Source"}
    before = copy.deepcopy(row)
    assert score_record(row, task=task)["components"] == {"relevance": tier, "completeness": 1}
    assert row == before


@pytest.mark.parametrize("alias,canonical", [("oxid_family", "ros_or_redox_assay"),
                                           ("reactive_metaphorite_trapping", "reactive_metabolite_trapping")])
def test_unknown_alias_does_not_promote_a_recognized_weak_proxy(alias, canonical):
    result = score_record({"endpoint_name": alias, "canonical_endpoint_name": canonical}, task="ames")
    assert result["components"]["relevance"] == .25
    assert "unrecognized_endpoint" in result["ambiguity_flags"]


def test_unknown_endpoint_is_not_vetoed_by_an_unrelated_endpoint():
    result = score_record({"endpoint_name": "BBB permeability", "endpoint_detail": "unfamiliar endpoint"}, task="ames")
    assert result["components"]["relevance"] == .5


@pytest.mark.parametrize("task,endpoint,context,tier,reason", [
    ("bbb_martins", "IC50", "brain endothelial cells", .5, "potency_target_not_established"),
    ("bbb_martins", "IC50", "ABCB1 in brain endothelial cells", .75, "identified_barrier_transport_or_permeability"),
    ("bbb_martins", "IC50 cytotoxicity", "brain endothelial cells", 0, "explicit_offtask_potency"),
    ("bbb_martins", "IC50 receptor binding", "brain endothelial cells", 0, "explicit_offtask_potency"),
    ("bbb_martins", "inhibition potency", "Caco-2 accumulation assay", .25, "potency_in_distant_system"),
    ("bbb_martins", "Papp", "not a BBB model", .5, "transport_context_incomplete"),
    ("bbb_martins", "Papp", "presumed BBB model", .5, "transport_context_incomplete"),
    ("bbb_martins", "Papp", "not a BBB model; Caco-2 monolayer", .25, "transport_in_distant_system"),
    ("bbb_martins", "CSF concentration", "rat", .75, "csf_exposure_proxy"),
    ("ames", "anti-mutagenicity in Ames test", "TA98", .25, "protective_modulation_not_mutagenicity"),
    ("ames", "mutation suppression", "Salmonella", .25, "protective_modulation_not_mutagenicity"),
    ("ames", "reverse mutation", "Saccharomyces yeast", .75, "nonbacterial_reverse_mutation_proxy"),
    ("ames", "bacterial reverse mutation", "Salmonella with mammalian liver S9", 1, "direct_bacterial_reverse_mutation"),
    ("ames", "bacterial reverse mutation", "non-mammalian in vitro assay", 1, "direct_bacterial_reverse_mutation"),
    ("ames", "reversion", "V79 DNA repair reporter assay", .5, "damage_or_unspecified_mutagenicity_proxy"),
    ("ames", "revertant colonies", "E. coli", 1, "direct_bacterial_reverse_mutation"),
    ("carcinogens", "lung carcinoma", "human cohort", .5, "tumor_endpoint_context_incomplete"),
    ("carcinogens", "lung carcinoma", "case-control study", .5, "tumor_endpoint_context_incomplete"),
    ("carcinogens", "lung carcinoma", "animal carcinogenicity bioassay", 1, "direct_carcinogenicity_evidence"),
    ("carcinogens", "lung carcinoma", "human epidemiology", 1, "direct_carcinogenicity_evidence"),
    ("carcinogens", "cancer incidence", "human cohort", 1, "direct_carcinogenicity_evidence"),
    ("carcinogens", "anti-genotoxicity", "human cells", .25, "protective_modulation_not_carcinogenicity"),
    ("carcinogens", "cell transformation", "BALB 3T3", .75, "cell_transformation_proxy"),
    ("carcinogens", "DNA adducts", "human", .75, "genotoxicity_proxy"),
])
def test_specific_assay_scope_and_rule_reasons(task, endpoint, context, tier, reason):
    result = score_record(dict(endpoint=endpoint, biological_system=context), task=task)
    assert result["components"]["relevance"] == tier
    assert reason in result["reasons"]


@pytest.mark.parametrize("task,endpoint", [("bbb_martins", "BBB penetration"),
                                          ("ames", "Ames test"), ("carcinogens", "carcinogenicity")])
@pytest.mark.parametrize("qualifier", ["predicted ", "not measured "])
def test_explicit_qualifier_is_not_erased_by_an_endpoint_alias(task, endpoint, qualifier):
    result = score_record(dict(endpoint_name=qualifier + endpoint,
                               canonical_endpoint_concept=endpoint), task=task)
    assert result["components"]["relevance"] == .5
    assert any("predicted" in reason or "not_established" in reason for reason in result["reasons"])


@pytest.mark.parametrize("context", ["experimentally measured, not predicted", "without computational predictions",
                                    "measured rather than predicted"])
def test_negated_prediction_is_not_a_prediction(context):
    assert score_record(dict(endpoint="BBB penetration", evidence_basis=context),
                        task="bbb_martins")["components"]["relevance"] == 1


@pytest.mark.parametrize("model", ["predictive value", "BOILED-Egg predictive model", "BOILED-Egg", "SwissADME",
                                  "QikProp", "ADMETlab", "pkCSM"])
def test_named_prediction_models_cannot_become_direct_evidence(model):
    result = score_record(dict(bbb_permeability_label="permeable", assay_model=model), task="bbb_martins")
    assert result["components"]["relevance"] == .5
    assert "predicted_evidence" in result["ambiguity_flags"]


def test_predictive_experimental_assay_is_not_a_computational_model():
    assert score_record(dict(bbb_permeability_label="permeable", assay_model="predictive PAMPA assay"),
                        task="bbb_martins")["components"]["relevance"] == .75


def test_structured_transporter_identity_qualifies_potency_without_supplying_anatomy():
    row = dict(endpoint="IC50", canonical_transporter_identifier="Abcb1b - ATP-dependent translocase ABCB1 [mouse]")
    assert score_record(row, task="bbb_martins")["components"]["relevance"] == .5
    row["assay_model"] = "brain endothelial monolayer"
    assert score_record(row, task="bbb_martins")["components"]["relevance"] == .75
    row["assay_model"] = "Caco-2 monolayer"
    assert score_record(row, task="bbb_martins")["components"]["relevance"] == .25


def test_target_description_cannot_supply_an_assay_model_or_anatomical_site():
    row = dict(endpoint="blood-to-brain transport", mediator_name="cell surface transporter")
    assert score_record(row, task="bbb_martins")["components"]["relevance"] == 1
    row = dict(endpoint="IC50", mediator_name="ABCB1 transporter at the BBB in brain endothelial cells")
    assert score_record(row, task="bbb_martins")["components"]["relevance"] == .5


def test_reviewed_null_target_overrides_stale_canonical_target():
    row = dict(endpoint="IC50", assay_model="brain endothelial monolayer",
               canonical_transporter_identifier="ABCB1", reviewed_record_json={"transporter_identifier": None})
    assert score_record(row, task="bbb_martins")["components"]["relevance"] == .5


@pytest.mark.parametrize("record", [None, [], {"input": []}, {"input": {"source_fields": []}},
                                     {"raw_record_json": "[]"}, {"reviewed_record_json": "{"}])
def test_invalid_inputs(record):
    with pytest.raises((TypeError, ValueError)):
        score_record(record, task="ames")
    with pytest.raises(ValueError, match="Unsupported"):
        score_record({}, task="dili")
