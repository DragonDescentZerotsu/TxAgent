"""Task relevance is independent of outcome polarity and information completeness."""

from copy import deepcopy
import json

import pytest

from optimization.bioavailability_score import score_bioavailability_record as score


def record(endpoint="oral bioavailability", system="human", result="42", **extra):
    return dict(endpoint=endpoint, biological_system=system, reported_result=result,
                support_text="The source reports this observation.", **extra)


@pytest.mark.parametrize("endpoint,system,tier", [
    ("oral bioavailability", "human", 1),
    ("relative oral bioavailability", "rat", 1),
    ("absolute bioavailability", "oral and intravenous study", 1),
    ("corrected bioavailability", "oral dosing", 1),
    ("bioavailability", "human", .5),
    ("bioavailability", "subcutaneous administration", 0),
    ("oral/IV AUC comparison", "human", .75),
    ("AUC0_inf", "oral administration", .75),
    ("AUC∞", "p.o.", .75),
    ("Cmax", "gavage in rats", .75),
    ("Tmax", "oral dose", .75),
    ("AUC", "intravenous only", .25),
    ("AUC", "human", .5),
    ("intestinal effective permeability", "rat perfusion", .75),
    ("Papp", "Caco‐2 cell monolayer", .75),
    ("permeability", "GI-PAMPA", .75),
    ("caco2_mdck_pampa_permeability", "BBB-PAMPA", .25),
    ("caco2_mdck_pampa_permeability", "GI-PAMPA", .75),
    ("permeability", "MDCK kidney-derived cells", .5),
    ("permeability", "PAMPA", .5),
    ("P-gp efflux", "intestinal enterocytes", .75),
    ("ABCB1 substrate", "brain endothelial cells", .25),
    ("uptake", "Ussing chamber", .75),
    ("solubility", "FaSSIF", .75),
    ("dissolution", "simulated gastric fluid", .75),
    ("GI stability", "buffer", .75),
    ("dissolution", "water", .5),
    ("solubility", "aqueous buffer", .5),
    ("hepatic extraction ratio", "rat", .75),
    ("CLint", "human liver microsomes", .75),
    ("intrinsic_clearance", "HLM", .75),
    ("CYP3A4 metabolism", "intestinal microsomes", .75),
    ("UGT glucuronidation", "human hepatocytes", .75),
    ("FMO metabolism", "human liver", .75),
    ("S9 stability", "liver S9", .75),
    ("CYP3A4 metabolism", "recombinant CYP3A4", .5),
    ("metabolic half life", "presumed liver microsomes", .5),
    ("metabolic half life", "not a liver model", .5),
    ("metabolic half life", "skeletal muscle", .5),
    ("renal clearance", "kidney", .25),
    ("logD", "pH 7.4", .25),
    ("plasma protein binding", "human plasma", .25),
    ("pKa", "buffer", .25),
    ("nephrotoxicity", "kidney", 0),
    ("cell viability", "human liver microsomes", 0),
    ("brain penetration", "mouse", 0),
    ("blood-brain barrier permeability", "brain", 0),
    ("skin permeation", "skin", 0),
    ("ocular delivery", "eye", 0),
    ("receptor binding", "intestinal cells", 0),
    ("hepatic clearance and cytotoxicity", "liver", .5),
    ("oral bioavailability and nephrotoxicity", "human", .5),
    ("predicted oral bioavailability", "human", .5),
    ("oral bioavailability not measured", "oral study", .5),
    ("unfamiliar readout", "human", .5),
])
def test_relevance_tiers_and_context_rules(endpoint, system, tier):
    result = score(record(endpoint, system))
    assert result["components"]["relevance"] == tier
    assert 0 <= result["score"] <= 1
    assert result["matched_rules"]


@pytest.mark.parametrize("value", (0, "0%", "high", "low", "not a substrate", "no change"))
def test_outcome_polarity_does_not_change_relevance(value):
    result = score(record(result=value))
    assert result["components"] == dict(relevance=1, completeness=1)


def test_completeness_does_not_extract_results_from_support():
    row = record(result=None)
    row["support_text"] = "Oral bioavailability was 42% in humans."
    result = score(row)
    assert result["score"] == .75
    assert result["missing_fields"] == ["result"]
    assert result["result_content_status"] == "missing"


@pytest.mark.parametrize("value", ("See Figure 7", "AUC measured", "no results reported"))
def test_method_and_figure_only_result_is_not_an_outcome(value):
    result = score(record(result=value))
    assert result["result_content_status"] == "description_only"
    assert not result["completeness_checks"]["result"]


def test_unfamiliar_result_retains_presence_credit_and_explicit_uncertainty():
    result = score(record(result="peculiar redistribution of the concentration envelope"))
    assert result["result_content_status"] == "unresolved"
    assert result["completeness_checks"]["result"]


def test_reviewed_nulls_override_stale_aliases():
    row = dict(canonical_endpoint_name="oral bioavailability", canonical_measurement_text="42",
               canonical_species_context="human", support_text="Source passage",
               raw_record_json={"oral_bioavailability_value": "42"},
               reviewed_record_json={"oral_bioavailability_value": None})
    assert not score(row)["completeness_checks"]["result"]
    row["reviewed_record_json"] = dict(endpoint_name=None, oral_bioavailability_value="42")
    result = score(row)
    assert not result["completeness_checks"]["endpoint"]
    assert result["components"]["relevance"] == .5


def test_native_canonical_raw_and_prepared_inputs_agree_and_remain_unchanged():
    native = record()
    canonical = dict(canonical_endpoint_name="endpoint=oral bioavailability",
                     canonical_measurement_text="reported_result=42",
                     canonical_species_context="biological_system=human", support_text=native["support_text"])
    wrapped = dict(raw_record_json=json.dumps(native))
    prepared = dict(input=dict(source_fields=native, support_text=native["support_text"]))
    for row in (native, canonical, wrapped, prepared):
        before = deepcopy(row)
        result = score(row)
        assert result["components"] == dict(relevance=1, completeness=1)
        assert json.loads(json.dumps(result)) == result
        assert row == before


def test_source_name_and_support_cannot_supply_relevance():
    result = score(dict(source_id="hf_bioavailability", endpoint="skin permeation",
                        reported_result="23", biological_system="skin",
                        support_text="Discussed oral bioavailability and intestinal absorption."))
    assert result["components"]["relevance"] == 0


def test_bibliographic_or_exposure_wrapper_is_not_an_assay():
    result = score(dict(endpoint="oral bioavailability", reported_result="0",
                        canonical_assay_context="pmid=123; oral_dose=10 mg", support_text="p=0.05"))
    assert not result["completeness_checks"]["assay_or_system"]
    assert result["completeness_checks"]["support"]


@pytest.mark.parametrize("row", (None, [], 1, "record", {"input": {}}, {"raw_record_json": []}))
def test_malformed_inputs_fail(row):
    with pytest.raises(TypeError):
        score(row)


def test_invalid_json_fails():
    with pytest.raises(json.JSONDecodeError):
        score({"raw_record_json": "{broken"})


def test_empty_record_has_zero_completeness():
    assert score({})["score"] == 0


def test_all_reviewed_source_endpoint_aliases_have_explicit_rule_families():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / 'data/processing/evidence_library/versions/v10/tasks/bioavailability_ma/data_processing/canonicalization_v7/endpoint_concepts'
    files = list(root.glob('*.json'))
    assert len(files) == 5
    for path in files:
        for mapping in json.loads(path.read_text())["mappings"]:
            endpoint = mapping["endpoint_name"]
            result = score(dict(endpoint_name=endpoint))
            assert any(row["families"] for row in result["matched_rules"]), (path.name, endpoint)


def test_distinct_relevant_readout_is_not_vetoed_by_another_field():
    row = record(endpoint="oral bioavailability", endpoint_category="nephrotoxicity")
    result = score(row)
    assert result["components"]["relevance"] == 1


def test_unestablished_hepatic_identity_does_not_promote_mechanism():
    result = score(record(endpoint="CYP3A4 metabolism", system="hepatic origin not established"))
    assert result["components"]["relevance"] == .5


def test_gene_names_use_boundaries_and_do_not_match_arbitrary_words():
    result = score(record(endpoint="cypress appearance", system="human liver"))
    assert result["reasons"] == ["unrecognized_endpoint"]


def test_generic_extraction_is_not_a_hepatic_disposition_endpoint():
    assert score(record(endpoint="DNA extraction", system="human liver"))["reasons"] == ["unrecognized_endpoint"]


def test_route_only_study_context_does_not_supply_assay_information():
    result = score(dict(endpoint="AUC", reported_result="42", study_context="oral dose 10 mg", support_text="Source"))
    assert not result["completeness_checks"]["assay_or_system"]
    assert result["components"]["relevance"] == .75


@pytest.mark.parametrize('endpoint,system', (
    ('oral clearance', 'human'), ('hepatic uptake', 'human liver'),
    ('substrate depletion', 'human liver microsomes'),
))
def test_identified_disposition_is_a_strong_proxy(endpoint, system):
    assert score(record(endpoint,system))['components']['relevance']==.75


def test_patient_disease_context_is_not_a_tested_organ():
    result = score(dict(endpoint="AUC0_t", study_context="renal-transplant patients, plasma",
                        reported_result="23", support_text="Source"))
    assert result["components"]["relevance"] == .5
    result = score(record(endpoint="fraction_absorbed", system="clinical oral bioavailability",
                          qualifying_conditions="renal transplant patients"))
    assert result["components"]["relevance"] == .75


def test_hepatic_disease_population_does_not_establish_hepatic_assay():
    result = score(dict(endpoint="CYP3A4 metabolism", biological_context="hepatic disease patients",
                        reported_result="23", support_text="Source"))
    assert result["components"]["relevance"] == .5


def test_iv_reference_parameters_are_not_an_iv_only_route():
    result = score(dict(endpoint="bioavailability", study_context="calculated from IV parameters",
                        reported_result="23", support_text="Source"))
    assert result["components"]["relevance"] == .5


def test_coagent_route_is_not_the_tested_molecule_route():
    result = score(record(endpoint="CYP3A4 metabolism", system="in vivo inhibition study",
                          qualifying_conditions="inhibitor administered i.v. prior to test-drug dosing"))
    assert result["components"]["relevance"] == .5


def test_conflicting_explicit_routes_are_flagged():
    result = score(record(system="subcutaneous administration"))
    assert result["components"]["relevance"] == .5
    assert "conflicting_route_context" in result["ambiguity_flags"]
    assert score(record(endpoint="subcutaneous bioavailability"))["components"]["relevance"] == 0


def test_renal_disposition_endpoint_identifies_a_distant_proxy():
    assert score(record(endpoint="renal clearance", system="human"))["components"]["relevance"] == .25


def test_reference_formulation_does_not_hide_explicit_oral_route():
    result = score(dict(endpoint="AUC0_t", reported_result="23",
                        study_context="fasted state, reference formulation, oral tablet",
                        support_text="Source"))
    assert result["components"]["relevance"] == .75


@pytest.mark.parametrize("endpoint,tier,reason", [
    ("UV absorption", 0, "non_adme_measurement"),
    ("absorption spectrum", 0, "non_adme_measurement"),
    ("dissolved oxygen", 0, "non_adme_measurement"),
    ("radiation exposure", 0, "non_adme_measurement"),
    ("no oral bioavailability", 1, "direct_oral_bioavailability"),
    ("no oral bioavailability measurement", .5, "measurement_not_established"),
    ("oral bioavailability not measured", .5, "measurement_not_established"),
    ("intestinal permeability not tested", .5, "measurement_not_established"),
])
def test_specific_endpoint_meaning_and_absent_measurements(endpoint, tier, reason):
    result = score(record(endpoint, "intestinal cells"))
    assert result["components"]["relevance"] == tier
    assert reason in result["reasons"]


@pytest.mark.parametrize("qualifier", ["predicted ", "not measured "])
def test_explicit_qualifier_survives_an_unqualified_alias(qualifier):
    result = score(record(endpoint=qualifier + "oral bioavailability",
                          canonical_endpoint_concept="oral bioavailability"))
    assert result["components"]["relevance"] == .5


@pytest.mark.parametrize("context", ["experimentally measured, not predicted", "no prediction",
                                    "measured rather than predicted"])
def test_negated_prediction_does_not_downgrade_observation(context):
    assert score(record(experimental_conditions=context))["components"]["relevance"] == 1


def test_unknown_alias_does_not_promote_a_known_weak_proxy():
    result = score(record(endpoint="logD", canonical_endpoint_name="unfamiliar alias"))
    assert result["components"]["relevance"] == .25
    assert "unrecognized_endpoint" in result["ambiguity_flags"]
