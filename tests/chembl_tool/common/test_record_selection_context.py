"""Selection isolation and source applicability; no inference or gold labels."""

import pytest

from tools.chembl_tool.common.record_diversity import select_level_records
from tools.chembl_tool.common.record_selection_context import (
    relevant_outcomes, retrieval_ames_rank, selection_context, select_contextual_records,
    score_dili_record,
)


def card(cid, molecule="m", outcome="increase", **extra):
    return dict(id=cid, molecule_id=molecule, features={cid: 1.0},
                outcomes={("oral bioavailability", outcome)}, reported_value="high",
                support_text="Source observation.", **extra)


def select(cards, **kwargs):
    return select_level_records(cards, selection_policy="molecule_local.v1", **kwargs)


def test_other_molecules_and_their_history_cannot_change_selection():
    own = [card("anchor"), card("repeat"), card("opposite", outcome="decrease")]
    alone, _ = select(own, per_molecule=2)
    other = card("other", "n", "decrease")
    other["features"] = {"anchor": 1e6, "opposite": 1e6}
    together, _ = select([other, *own], per_molecule=2,
                         previous_features_by_molecule={"n": [other["features"]]})
    assert [c["id"] for c in alone] == ["anchor", "opposite"]
    assert [c["id"] for c in together if c["molecule_id"] == "m"] == [c["id"] for c in alone]
    with pytest.raises(ValueError, match="keyed"):
        select(own, previous_features=[{"a": 1.0}])


def test_own_history_changes_context_gain_but_other_history_does_not():
    own = [card("anchor"), card("b"), card("c")]
    own[1]["features"] = {"b": 2.0}
    assert [c["id"] for c in select(own, per_molecule=2)[0]] == ["anchor", "b"]
    assert [c["id"] for c in select(own, per_molecule=2,
        previous_features_by_molecule={"m": [{"b": 1000.0}]})[0]] == ["anchor", "c"]


@pytest.mark.parametrize("n", [0, 1, 4, 5])
def test_under_budget_bypasses_features_and_ranking(n):
    cards = [{"id": str(i), "molecule_id": "m"} for i in range(n)]
    selected, audit = select(cards)
    assert selected == cards
    assert not audit["protected_card_ids"]
    assert select(cards, per_molecule=0)[0] == []


def test_duplicate_ids_and_negative_budget_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        select([card("x"), card("x")])
    with pytest.raises(ValueError):
        select([], per_molecule=-1)


def prepared(**fields):
    return {"input": {"source_fields": fields, "support_text": "", "focal_molecule": "A"}}


def dili_row(**fields):
    return dict(canonical_endpoint_name='ROS production', biological_system='human hepatocytes',
                canonical_measurement_text='decreased', support_text='ROS decreased.', **fields)


def test_dili_simple_raw_and_prepared_agree_without_lost_fields_and_do_not_mutate():
    from copy import deepcopy
    from tools.chembl_tool.common.record_diversity import prepare_record
    row = dili_row()
    saved = deepcopy(row)
    source_score = score_dili_record(row)
    prepared_score = score_dili_record(prepare_record(row))
    assert row == saved
    assert {k:v for k,v in source_score.items() if k!='input_kind'} == {
        k:v for k,v in prepared_score.items() if k!='input_kind'}


def test_relevance_filters_observations_not_the_entire_record():
    outcomes = {("oral bioavailability", "increase"), ("parasite load in liver", "decrease"),
                ("parasite load in spleen", "decrease")}
    assert relevant_outcomes("bioavailability_ma", outcomes) == {("oral bioavailability", "increase")}
    assert relevant_outcomes("bioavailability_ma", {("oral bioavailability", "decrease")})
    assert relevant_outcomes("ames", {("unscheduled DNA synthesis", "negative")})


@pytest.mark.parametrize("task,endpoints", [
    ("ames", ["mammalian_cell_gene_mutation", "strand break or alkali labile site",
              "reactive metabolite formation", "aneuploidy or malsegregation", "γH2AX foci"]),
    ("carcinogens", ["hepatoma induction", "chromosome_aberration", "dominant-lethal test",
                     "GSH depletion", "gap junction communication", "H3K9me3 enrichment",
                     "uterine wet weight", "percentage of cells in S phase"]),
    ("bbb_martins", ["P-gp substrate status", "Papp", "LAT1 substrate status",
                     "H+/OC antiporter substrate status", "hippocampal concentration"]),
    ("bioavailability_ma", ["AUC0–inf", "AUCinf", "fraction absorbed", "GI stability",
                           "fg_substrate_outcome:ABCB1", "microsomal stability", "glucuronidation"]),
    ("dili", ["hepatocellular injury", "LDH release", "taurocholate uptake",
              "cellular ATP content", "IL-1β secretion", "intracellular ROS", "GSH/GSSG ratio"]),
    ("skin_reaction", ["positive patch test reaction", "EC3", "cysteine depletion",
                       "protein covalent binding", "IL-1α", "GSH adduct formation"]),
])
def test_actual_source_aliases_cover_both_directions(task, endpoints):
    outcomes = {(e, k) for e in endpoints for k in ("positive", "negative", "increase", "decrease")}
    assert relevant_outcomes(task, outcomes) == outcomes


@pytest.mark.parametrize("task,fields,endpoint", [
    ("bbb_martins", {"canonical_assay_context": "p-gp inhibition"}, "inhibition potency"),
    ("bbb_martins", {"assay_system": "bovine brain endothelial bidirectional permeability"}, "other quantitative endpoint"),
    ("bioavailability_ma", {"canonical_endpoint_name": "efflux_or_secretory_transport"}, "Vmax"),
    ("skin_reaction", {"aop_event": "adverse_outcome_skin_sensitization"}, "erythema"),
    ("skin_reaction", {"assay_type": "non-radioactive LLNA (BrdU)"}, "cell proliferation"),
    ("skin_reaction", {"aop_event": "MIE_protein_binding"}, "adduct formation"),
])
def test_generic_readout_requires_explicit_assay_scope(task, fields, endpoint):
    outcome = {(endpoint, "negative")}
    assert not relevant_outcomes(task, outcome)
    assert relevant_outcomes(task, outcome, fields) == outcome
    # Long passages mentioning a relevant assay cannot lend scope to this readout.
    assert not relevant_outcomes(task, outcome, {"support_text": str(fields), "extra_details": str(fields)})


@pytest.mark.parametrize("task,endpoint", [
    ("bioavailability_ma", "parasite load in liver"),
    ("bioavailability_ma", "parasite load in spleen"),
    ("bbb_martins", "oral bioavailability"),
    ("dili", "renal toxicity"),
    ("dili", "visual disturbances"),
    ("skin_reaction", "skin hyperpigmentation"),
    ("ames", "antimicrobial activity"),
    ("carcinogens", "body weight"),
])
def test_task_scope_does_not_rescue_unrelated_observations(task, endpoint):
    assert not relevant_outcomes(task, {(endpoint, "decrease")}, {
        "canonical_endpoint_name": "oral_bioavailability", "aop_event": "adverse_outcome_skin_sensitization",
        "canonical_assay_context": "P-gp inhibition", "support_text": "DNA damage in liver and brain",
    })


def test_chromosome_positive_survives_competing_negative_records():
    raw = [{"card_id": str(i), "record_id": str(i), "analog_id": "m",
            "endpoint": "chromosome aberration", "reported_value": "negative" if i < 5 else "positive",
            "support_text": "Source observation."} for i in range(6)]
    encoded = {str(i): {**card(str(i)), "outcomes": {("chromosome aberration", "negative" if i < 5 else "positive")}}
               for i in range(6)}
    inputs = {str(i): prepared(canonical_endpoint_name="chromosome_aberration") for i in range(6)}
    selected, audit = select_contextual_records(raw, encoded, inputs, task="carcinogens")
    assert "5" in {c["id"] for c in selected}
    assert audit["molecules"]["m"]["selected_outcomes"] == 2
    assert len(selected) == 5


@pytest.mark.parametrize("panel,expected", [
    ("TA98 (also TA98NR and TA98/1,8-DNP6)", 1),
    ("TA98NR", 2), ("TA98/1,8-DNP6", 2),
    ("TA98 and TA100", 0),
])
def test_retrieval_accepts_explicit_strains_without_assigning_variants(panel, expected):
    f = {"canonical_endpoint_name": "bacterial_reverse_mutation", "metabolic_activation": "absent", "test_system": panel}
    assert retrieval_ames_rank("metabolic_activation=absent+strain_panel=TA98,TA100", f) == expected


def test_chemically_treated_mixture_loses_anchor_priority_to_original_sample():
    native = prepared(canonical_endpoint_name="bacterial_reverse_mutation", metabolic_activation="absent",
                      test_system="TA98 (also TA98NR and TA98/1,8-DNP6)")
    mixture = prepared(canonical_endpoint_name="bacterial_reverse_mutation", metabolic_activation="absent",
                       test_system="TA98 and TA100", qualifying_conditions="Only the reaction mixture with sodium nitrite.")
    condition = "metabolic_activation=absent+strain_panel=TA98,TA100"
    a = selection_context("ames", condition, native, {}, [])
    b = selection_context("ames", condition, mixture, {}, [])
    assert a["condition_rank"] < b["condition_rank"]


def test_carcinogen_species_limitation_is_compatible_not_direct_species_study():
    p = prepared(evidence_population_or_model="male rats", canonical_endpoint_name="kidney tumors")
    p["input"]["support_text"] = "Kidney tumors occurred only in male rats, not in dogs or mice."
    a = selection_context("carcinogens", "species=dog", p, {}, [])
    assert a["condition_rank"][0] == 1
    p["input"]["support_text"] = "Kidney tumors occurred in male rats."
    assert selection_context("carcinogens", "species=dog", p, {}, [])["condition_rank"][0] == 3


def test_carcinogen_native_test_system_supplies_species_without_changing_source():
    p = prepared(test_system="in_vivo; mouse (Mus musculus); bone marrow cells",
                 canonical_endpoint_name="chromosome_aberration")
    assert selection_context("carcinogens", "species=rodent", p, {}, [])["condition_rank"][0] == 0
    assert selection_context("carcinogens", "species=human", p, {}, [])["condition_rank"][0] == 3
    assert "canonical_species_context" not in p["input"]["source_fields"]
    p["input"]["source_fields"]["test_system"] = "in_vitro; human; cultured peripheral blood lymphocytes"
    assert selection_context("carcinogens", "species=rodent", p, {}, [])["condition_rank"][0] == 3


@pytest.mark.parametrize("field", ["biological_test_system", "biological_system", "biological_model"])
def test_experimental_species_overrides_human_dose_or_platform_mentions(field):
    import copy

    p = prepared(**{field: "rat (Rattus norvegicus) sperm DNA, in vivo"},
                 qualifying_conditions="Dose reflects daily use in humans; human assay platform.",
                 canonical_endpoint_name="DNA methylation")
    before = copy.deepcopy(p)
    assert selection_context("carcinogens", "species=human", p, {}, [])["condition_rank"][0] == 3
    assert selection_context("carcinogens", "species=rodent", p, {}, [])["condition_rank"][0] == 0
    assert p == before


def test_incidental_species_without_a_tested_system_stays_unknown():
    p = prepared(qualifying_conditions="Dose reflects daily use in humans.")
    p["input"]["support_text"] = "The dose is not unlike daily use in humans."
    assert selection_context("carcinogens", "species=human", p, {}, [])["condition_rank"][0] == 2


def test_native_human_and_conflicting_systems_are_not_conflated():
    p = prepared(biological_test_system="human cultured endothelial cells",
                 qualifying_conditions="Comparison to rodent exposure levels.")
    assert selection_context("carcinogens", "species=human", p, {}, [])["condition_rank"][0] == 0
    p["input"]["source_fields"]["species"] = "rat"
    assert selection_context("carcinogens", "species=human", p, {}, [])["condition_rank"][0] == 1


def test_fasted_fed_missing_and_absent_query_are_not_conflated():
    def rank(text, condition="prandial_state=fasted"):
        return selection_context("bioavailability_ma", condition, prepared(study_context=text), {}, [])["condition_rank"][0]
    assert rank("fasted") == 0
    assert rank("") == 2
    assert rank("fed") == 3
    assert rank("fasted", "no_reported_external_condition") == 2


def test_native_bbb_and_dili_conditions_are_consumed():
    assert selection_context("bbb_martins", "disease=bacterial_meningitis", prepared(study_context="bacterial meningitis"), {}, [])["condition_rank"][0] <= 1
    assert selection_context("dili", "exposure=overdose_or_supratherapeutic", prepared(exposure_context="overdose_or_supratherapeutic"), {}, [])["condition_rank"][0] == 0


def test_adapter_under_budget_needs_no_context_or_quality_sort():
    cards = [{"card_id": x, "analog_id": "m"} for x in ["b", "a"]]
    selected, audit = select_contextual_records(cards, {x: card(x) for x in ["a", "b"]}, {}, task="ames")
    assert [c["id"] for c in selected] == ["b", "a"]
    assert not audit["record_context"]


def test_contextual_anchor_and_outcomes_preserve_raw_records():
    import copy
    raw = [{"card_id": str(i), "analog_id": "m", "record_id": str(i),
            "reported_value": "9.9" if i == 0 else "low", "reported_unit": "%",
            "support_text": "Source passage."} for i in range(7)]
    before = copy.deepcopy(raw)
    encoded = {c["card_id"]: card(c["card_id"]) for c in raw}
    encoded["1"]["outcomes"] = {("oral bioavailability", "decrease"), ("parasite load", "decrease")}
    inputs = {str(i): prepared(canonical_endpoint_name="oral_bioavailability") for i in range(7)}
    inputs["0"]["input"]["source_fields"]["bioavailability_report_type"] = "relative_comparison"
    selected, audit = select_contextual_records(raw, encoded, inputs, task="bioavailability_ma")
    assert len(selected) == 5
    assert audit["protected_card_ids"] == ["1"]
    assert ("parasite load", "decrease") not in next(c for c in selected if c["id"] == "1")["outcomes"]
    assert raw == before
    assert ("parasite load", "decrease") in encoded["1"]["outcomes"]


def test_measurement_scope_prompt_is_explicitly_opt_in_and_keeps_labels():
    from tools.chembl_tool.tasks.bioavailability_ma.experiment_config import (
        get_progressive_task_contract, get_adapted_prompt_contract,
    )
    original = get_progressive_task_contract()
    adapted = get_adapted_prompt_contract("measurement_scope.v1")
    assert adapted.task_instructions[:-1] == original.task_instructions
    assert "IV-metabolite" in adapted.task_instructions[-1]
    assert adapted.prediction_values == original.prediction_values
    assert get_progressive_task_contract() == original
    with pytest.raises(ValueError):
        get_adapted_prompt_contract("unknown")


def test_measurement_scope_cli_and_hash_are_checked_before_provider_work(monkeypatch):
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_progressive_curve as runner
    monkeypatch.setattr(runner, "_resolve_provider_pool_config", lambda *_: pytest.fail("no provider work"))
    contract = runner._prompt_adaptation("measurement_scope.v1", ["bioavailability_ma"])
    assert len(contract["task_contract_sha256"]["bioavailability_ma"]) == 64
    with pytest.raises(ValueError, match="requires --fresh-query-priors"):
        runner.main(["--tasks", "bioavailability_ma", "--prompt-adaptation", "measurement_scope.v1"])


@pytest.mark.parametrize('population', [
    'human (formaldehyde‐allergic individual)',
    'GMTG-positive hairdressers with ACD',
    'patients allergic to formaldehyde',
])
def test_explicit_prior_allergy_is_qualified_elicitation_scope(population):
    result = selection_context('skin_reaction', '',
        prepared(species_or_population=population), {}, {('patch test', 'positive')})
    assert result['anchor_rank'] == 2
    assert 'elicitation_in_previously_sensitized_subjects' in result['context_audit']['reasons']


def test_new_positive_patch_test_does_not_imply_prior_sensitization():
    result = selection_context('skin_reaction', '',
        prepared(species_or_population='human', assay_context='positive patch test, allergic contact dermatitis'),
        {}, {('patch test', 'positive')})
    assert result['anchor_rank'] == 1


def test_focal_metabolite_identity_is_not_a_metabolite_measurement():
    # This record's own molecule is N-desmethylivabradine, not administered ivabradine.
    p = prepared(canonical_endpoint_name='cyp_metabolism', species='human')
    p['input']['support_text'] = 'The main active metabolite, N-desmethylivabradine, is further metabolised by CYP3A4.'
    result = selection_context('bioavailability_ma', '', p, {}, {('CYP3A4 metabolism', 'unclear')})
    assert result['anchor_rank'] == 1


def test_emulsion_absorption_contrast_retains_formulation_qualification():
    from tools.chembl_tool.common.record_selection_context import record_quality
    p = prepared(canonical_endpoint_name='intestinal_absorption',
                 formulation_or_solid_form='emulsion containing compound 1',
                 assay_system='in vivo oral dosing in beagle dogs')
    p['input']['support_text'] = 'The emulsion improved absorption compared with a methylcellulose suspension.'
    annotation = {'outcomes':[{'endpoint':'intestinal_absorption', 'kind':'increase',
        'focal_scope':'focal', 'evidence_basis':'measured', 'conditions':[],
        'result_scope':'focal_effect', 'limitations':[]}], 'unresolved':[]}
    q = record_quality('bioavailability_ma', '', p, {}, annotation)
    assert q['components']['object'] == .5
    assert 'special_formulation' in q['reasons']
    assert q['components']['evidence_strength'] == 1
    # An emulsion discussed only as background does not qualify the tested agent.
    p['input']['source_fields'].pop('formulation_or_solid_form')
    assert record_quality('bioavailability_ma', '', p, {}, annotation)['components']['object'] == 1


def test_bbb_delivery_system_encoded_as_assay_model_is_qualified():
    p = prepared(canonical_endpoint_name='bbb_permeability_outcome',
                 assay_model='nanocapsules', qualifying_conditions='intravenous injection')
    result = selection_context('bbb_martins', '', p, {}, {('BBB permeability', 'positive')})
    assert result['anchor_rank'] == 2
    assert 'special_formulation' in result['context_audit']['reasons']
    p['input']['source_fields']['assay_model'] = 'in vivo brain distribution'
    p['input']['support_text'] = 'Unlike nanoparticles, the unformulated molecule was injected.'
    assert selection_context('bbb_martins', '', p, {}, [])['anchor_rank'] == 1


@pytest.mark.parametrize('task', ['ames', 'carcinogens'])
@pytest.mark.parametrize('endpoint', ['N-hydroxylation', 'CYP2E1 expression'])
def test_metabolic_readout_requires_explicit_reactive_metabolite_scope(task, endpoint):
    outcome = {(endpoint, 'decrease')}
    assert not relevant_outcomes(task, outcome, {'canonical_assay_context':'general liver metabolism'})
    assert relevant_outcomes(task, outcome, {'canonical_assay_context':'bioactivating enzyme for reactive metabolite formation'}) == outcome
