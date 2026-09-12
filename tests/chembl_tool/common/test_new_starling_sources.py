"""Regression checks for lossless ingestion and direct-label boundaries."""
import json
import pytest

import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.common.starling import build_source_records as records
from tools.chembl_tool.tasks.dili.starling_gold import decide as dili_decide
from tools.chembl_tool.tasks.carcinogens.starling_gold import decide as carcinogen_decide


@pytest.mark.parametrize('support,extra', [
    ('Significant hepatotoxicity has been reported with tacrine.', 'Dosing starts at 10 mg four times daily for 4 weeks.'),
    ('In the summary of a study, 18 lupus patients received para-aminobenzoic acid and one died of toxic hepatitis.', ''),
    ('Four patients treated with azathioprine (refs 144, 153, 156) subsequently developed ANLL.', ''),
    ('In a study by Romanenko, melatonin was administered to mice for 5 months and leukemia developed.', ''),
    ('Allen, Shumaker and co-workers injected monocrotaline into rats at 5 mg/kg for 12 months; 31 tumors developed.', ''),
    ('Hydroxyurea showed no carcinogenic effects in large case series.', 'Cited references 46-49.'),
])
def test_unresolved_secondary_or_dose_only_claim_cannot_make_new_pmid_vote(support, extra):
    from tools.chembl_tool.common.starling.source_gold_review import study_identity
    assert study_identity({'pmid': '111', 'support_text': support, 'extra_details': extra})[0] is None


def test_route_is_not_author_and_institutional_bioassay_is_not_reporting_pmid():
    from tools.chembl_tool.common.starling.source_gold_review import study_identity
    row = {'pmid': '2525415', 'support_text': '64 patients received FUDR, 20 in a pilot study by hepatic artery infusion; chemical hepatitis occurred.'}
    assert study_identity(row)[0] == 'pmid:2525415'
    for pmid in ['111', '222']:
        assert study_identity({'pmid': pmid, 'support_text': 'No neoplasms in rats in a two-year study (NTP, 1992).'})[0] == 'cited:ntp:1992'


@pytest.mark.parametrize('outcome', ['leukemia', 'leukaemia', 'lymphoma', 'adenomas', 'sarcomas'])
def test_cancer_subtype_outcomes_do_not_enter_query_conditions(outcome):
    from tools.chembl_tool.common.starling.source_gold_review import outcome_in_condition, answer_bearing_atoms
    text = 'Treatment increased ' + outcome + ' incidence.'
    assert outcome_in_condition({'qualifying_conditions': text}, 'carcinogens')
    assert answer_bearing_atoms(['reported_condition=' + text], 'carcinogens')
    assert not answer_bearing_atoms(['population_context=patients_with_preexisting_lymphoma', 'dose=10_mg'], 'carcinogens')


def test_observed_negative_cohort_is_distinct_from_no_published_reports():
    row = {'agent_type': 'small_molecule_or_chemical', 'causal_status': 'not_supported',
           'human_evidence_basis': 'observational_epidemiology', 'exposure_context': 'therapeutic_use',
           'support_text': 'In this retrospective study of ketamine treatment, there were no reports of hepatotoxicity.'}
    assert dili_decide(row)[0] == 0
    row['support_text'] = 'There are no reports of hepatotoxicity with this agent.'
    assert dili_decide(row)[0] is None


def test_comparator_negation_is_not_absence_and_risk_model_is_not_observation():
    row = {'agent_type': 'small_molecule_or_chemical', 'causal_status': 'not_supported',
           'human_evidence_basis': 'clinical_trial', 'exposure_context': 'therapeutic_use',
           'support_text': 'No association between rifampicin concentrations and hepatotoxicity in 268 patients; concentrations were similar in patients with and without hepatotoxicity.'}
    assert dili_decide(row)[0] is None
    row = {'agent_category': 'individual_substance_or_drug', 'carcinogenicity_conclusion': 'positive',
           'evidence_basis': 'human_epidemiology', 'evidence_scope': 'human',
           'support_text': 'Acute leukemias were considered benzene-related in additive risk models.'}
    assert carcinogen_decide(row)[1] == 'modeled_risk_not_observed_carcinogenicity'


def test_background_citation_does_not_replace_explicit_current_study():
    from tools.chembl_tool.common.starling.source_gold_review import study_identity
    row = {'pmid': '16622453', 'support_text': 'The goal of this study was to assess VOD in 59 patients.',
           'extra_details': 'Dose-related injury had previously been reported (Honjo 1988).'}
    assert study_identity(row)[0] == 'pmid:16622453'
    assert study_identity({'pmid': '123', 'support_text': 'Lung cancer has been found in mice exposed to vanadium pentoxide.'})[0] is None
    assert study_identity({'pmid': '123', 'support_text': 'Negative rat inhalation studies (NTP 1996a,b,c).'})[0] is None


def test_source_record_preserves_zero_false_and_all_context(tmp_path, monkeypatch):
    raw = {"source_row_uid": "sr_permanent", "SMILES": "CCO", "assay_category": "mitochondrial_respiration",
        "reported_result": "protective; no decrease", "test_concentration": 0,
        "needs_more_context": True, "support_text": "Ethanol protected the cells.",
        "confidence": 0.0, "extra_details": "Measured in a co-exposure model.", "species": "rat"}
    path = tmp_path / "raw.parquet"
    pq.write_table(pa.Table.from_pylist([raw]), path)
    monkeypatch.setattr(records, "_identities", dict([records.identity("CCO")]))
    target, count, _ = records._convert(("dili", "dili_v2", path, 41, 0, tmp_path / "out.parquet"))
    row = pq.read_table(target).to_pylist()[0]
    assert count == 1
    assert json.loads(row["raw_record_json"]) == raw
    assert row["source_record_id"] == row["canonical_record_id"] == "sr_permanent"
    assert row["source_row_number"] == 41
    assert row["retrieval_eligible"]
    assert "test_concentration=0" in row["qualifying_conditions"]
    assert "co-exposure" in row["qualifying_conditions"]
    assert row["confidence"] == 0.0


def test_invalid_identity_is_retained_for_audit(tmp_path, monkeypatch):
    raw = {"source_row_uid": "sr_invalid", "SMILES": "not a molecule", "support_text": "Original text."}
    path = tmp_path / "raw.parquet"
    pq.write_table(pa.Table.from_pylist([raw]), path)
    monkeypatch.setattr(records, "_identities", dict([records.identity(raw["SMILES"])]))
    target, n, _ = records._convert(("carcinogens", "carcinogens_v2", path, 0, 0, tmp_path / "out.parquet"))
    row = pq.read_table(target).to_pylist()[0]
    assert n == 1 and not row["retrieval_eligible"]
    assert json.loads(row["raw_record_json"]) == raw


def test_dili_rare_injury_is_not_negative():
    row = {"agent_type": "small_molecule_or_chemical", "causal_status": "not_supported",
        "human_evidence_basis": "clinical_trial", "exposure_context": "therapeutic_use",
        "support_text": "No evidence of hepatotoxicity in this trial, but rare DILI cases exist."}
    assert dili_decide(row)[0] is None


def test_dili_laboratory_signal_and_association_do_not_vote():
    row = {"agent_type": "small_molecule_or_chemical", "causal_status": "established_or_definite",
        "human_evidence_basis": "clinical_trial", "exposure_context": "therapeutic_use",
        "maximum_reported_severity": "liver_test_abnormality_only", "support_text": "Hepatotoxicity was reported."}
    assert dili_decide(row)[0] is None
    row.update(maximum_reported_severity=None, causal_status="association_signal")
    assert dili_decide(row)[0] is None


def test_carcinogen_any_site_positive_but_site_negative_not_generalized():
    row = {"agent_category": "individual_substance_or_drug", "carcinogenicity_conclusion": "positive",
        "carcinogenic_role": None, "evidence_basis": "animal_carcinogenicity_bioassay",
        "evidence_scope": "experimental_animal", "evidence_population_or_model": "rat",
        "cancer_or_tumor": "liver", "support_text": "Compound X induced liver tumors in rats."}
    label, _, atoms = carcinogen_decide(row)
    assert label == 1 and atoms == ["species=rat"]
    row.update(carcinogenicity_conclusion="negative", support_text="Compound X is not hepatocarcinogenic in rats.")
    assert carcinogen_decide(row)[0] is None


def test_mechanistic_and_promoter_results_never_make_direct_gold():
    row = {"agent_category": "individual_substance_or_drug", "carcinogenicity_conclusion": "positive",
        "carcinogenic_role": None, "evidence_basis": "in_vitro_neoplastic_transformation",
        "evidence_scope": "in_vitro", "support_text": "Positive carcinogenic transformation."}
    assert carcinogen_decide(row)[0] is None
    row.update(evidence_basis="animal_carcinogenicity_bioassay", evidence_scope="experimental_animal",
        evidence_population_or_model="mouse", carcinogenic_role="promoter")
    assert carcinogen_decide(row)[0] is None


def test_optional_class_coverage_keeps_both_labels_in_all_splits():
    from tools.chembl_tool.common.starling.build_record_supported_benchmark import allocate_scaffold_groups, resolve_conditioned_eval_size
    from tools.chembl_tool.common.starling.build_conditioned_random_split import allocate_parent_groups
    rows = [{"molecule_identity_key": f"p{i}", "bemis_murcko_scaffold": "big" if i < 9 else "" if i == 10 else f"s{i}",
        "Y": int(i not in {0, 9, 10}), "source_record_count": 2, "condition_group": "human"} for i in range(30)]
    target = resolve_conditioned_eval_size(rows, nominal_target_size=3, required_condition_groups={"human"}, required_labels=(0, 1))
    assert target == 9
    scaffold_assignment, _ = allocate_scaffold_groups(rows, target_size=target, required_condition_groups={"human"}, required_labels=(0, 1), exclude_empty_scaffold_from_heldout=True)
    parent_assignment, _ = allocate_parent_groups(rows, required_labels=(0, 1))
    for split in ("train", "valid", "test"):
        assert {r["Y"] for r in rows if scaffold_assignment[r["bemis_murcko_scaffold"]] == split} == {0, 1}
        assert {r["Y"] for r in rows if parent_assignment[r["molecule_identity_key"]] == split} == {0, 1}


def test_explicit_negative_category_is_eligible_but_fatal_only_is_not():
    row = {"agent_type": "small_molecule_or_chemical", "causal_status": "not_supported",
        "human_evidence_basis": "explicit_negative_evidence", "exposure_context": "therapeutic_use",
        "support_text": "In 120 treated patients, no cases of drug-induced liver injury were observed."}
    assert dili_decide(row)[0] == 0
    row["support_text"] = "In 120 patients, there were no hepatotoxicity-related fatalities; liver injury occurred."
    assert dili_decide(row)[0] is None
    row["support_text"] = "This patient had no evidence of hepatotoxicity."
    row["extra_details"] = "Single patient followed for nephrotoxicity."
    assert dili_decide(row)[0] is None


def test_not_tumorigenic_is_negative_but_risk_model_is_not_an_experiment():
    row = {"agent_category": "individual_substance_or_drug", "carcinogenicity_conclusion": "negative",
        "evidence_basis": "animal_carcinogenicity_bioassay", "evidence_scope": "experimental_animal",
        "evidence_population_or_model": "mouse", "support_text": "The two-year study concluded X was not tumorigenic in mice."}
    assert carcinogen_decide(row)[0] == 0
    row.update(evidence_basis="human_epidemiology", evidence_scope="human",
        evidence_population_or_model="fish consumers", support_text="Calculated target cancer risk showed no carcinogenic risk.")
    assert carcinogen_decide(row)[1] == "modeled_risk_not_observed_carcinogenicity"


def test_original_study_citation_does_not_use_method_or_secondary_pmid():
    from tools.chembl_tool.common.starling.source_gold_review import study_identity
    first = {"pmid": "111", "support_text": "No evidence of carcinogenicity in rats (Til et al., 1972a)."}
    second = {**first, "pmid": "222"}
    assert study_identity(first)[0] == study_identity(second)[0] == "cited:til:1972"
    method = {"pmid": "333", "support_text": "In this study rats received DEN for 16 weeks and developed liver tumors.",
        "extra_details": "Tumors were classified according to Bannasch and Zerban (1990)."}
    assert study_identity(method)[0] == "pmid:333"
    primary = {"pmid": "444", "support_text": "Their patient developed losartan-induced liver injury on rechallenge.",
        "extra_details": "A table summarizes prior literature cases."}
    assert study_identity(primary)[0] == "pmid:444"


def test_alternative_populations_do_not_become_a_conjunction():
    from tools.chembl_tool.common.starling.source_gold_review import source_conditions
    row = {"support_text": "Patients with psoriasis were compared with rheumatoid arthritis patients."}
    atoms = source_conditions(row, "dili", ["population=human"])
    assert not any(a.startswith("population_context=") for a in atoms)
    assert any(a.startswith("unresolved_population_context=") for a in atoms)
    assert not any("support_text" in a or "hepatotoxic" in a for a in atoms)


def test_answer_bearing_conditions_require_review_before_query_use():
    from tools.chembl_tool.common.starling.source_gold_review import outcome_in_condition
    assert outcome_in_condition({"qualifying_conditions": "No clinically significant hepatotoxicity observed"}, "dili")
    assert outcome_in_condition({"qualifying_conditions": "Negative in female rats"}, "carcinogens")
    assert not outcome_in_condition({"qualifying_conditions": "Female rats exposed orally for 24 months"}, "carcinogens")


def test_random_coverage_can_expand_without_losing_conditions_or_parents():
    from tools.chembl_tool.common.starling.build_conditioned_random_split import allocate_parent_groups
    rows = [{"molecule_identity_key": f"p{p}", "condition_group": f"c{c}",
             "Y": c % 2, "source_record_count": 2} for p in range(3) for c in range(4)]
    assignment, audit = allocate_parent_groups(rows, required_labels=(0, 1), minimum_feasible_eval_size=True)
    assert audit["nominal_target_eval_rows"] == 1
    assert audit["target_rows"] == {"train": 4, "valid": 4, "test": 4}
    for split in ("train", "valid", "test"):
        selected = [row for row in rows if assignment[row["molecule_identity_key"]] == split]
        assert {row["condition_group"] for row in selected} == {"c0", "c1", "c2", "c3"}
        assert {row["Y"] for row in selected} == {0, 1}


def test_rescue_transplantation_is_not_baseline_population():
    from tools.chembl_tool.common.starling.source_gold_review import source_conditions
    atoms = source_conditions({"support_text": "Fulminant hepatitis was treated by liver transplantation."}, "dili", [])
    assert "population_context=liver_transplant" not in atoms
    atoms = source_conditions({"support_text": "Liver transplant recipients received tacrolimus."}, "dili", [])
    assert "population_context=liver_transplant" in atoms


def test_bracketed_original_reference_is_not_a_new_study():
    from tools.chembl_tool.common.starling.source_gold_review import study_identity
    row={'pmid':'39998096', 'support_text':'In one bioassay, 36 rats were fed contaminated corn for 2 years; 24 developed liver cancer.',
         'extra_details':'24 of 36 rats developed PLC; result cited as reference [15].'}
    assert study_identity(row)==(None,'secondary_or_pooled_study_origin_requires_review')


def test_reference_zero_dose_is_a_comparator_not_a_citation():
    from tools.chembl_tool.common.starling.source_gold_review import study_identity
    row={'pmid':'23799501', 'support_text':'The study population included 18 patients with cancer; the reference (0 Gy) group was compared with exposed patients.'}
    assert study_identity(row)==('pmid:23799501','')


def test_reference_laboratory_interval_does_not_hide_original_case():
    from tools.chembl_tool.common.starling.source_gold_review import study_identity
    row={'pmid':'11555130', 'support_text':'In case 1 the patient developed hepatitis and recovered after withdrawal.',
         'extra_details':'IgM 6.15 g/L (reference 0.44-3.40 g/L).'}
    assert study_identity(row)==('pmid:11555130','')
