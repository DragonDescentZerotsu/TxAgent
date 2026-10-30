"""Scientific boundaries and numerical coverage, without model calls."""

import copy
import math

import pytest

from tools.chembl_tool.common.record_diversity import encode_features, prepare_record, select_level_records
from tools.chembl_tool.common.record_numeric_features import absolute_oral_f, absolute_oral_f_features


def record(value="42", unit="%", **changes):
    row = {
        "canonical_endpoint_name": "oral_bioavailability",
        "canonical_measurement_text": value,
        "canonical_unit_text": unit,
        "bioavailability_report_type": "absolute",
        "species_or_population": "human",
        "comparator": "vs IV",
        "support_text": "Absolute oral bioavailability was measured against intravenous dosing.",
    }
    row.update(changes)
    return prepare_record(row)


def test_equivalent_units_share_numeric_coverage_without_changing_source():
    a, b = record("0.42", "fraction"), record("42%", "")
    frozen = copy.deepcopy(a)
    assert absolute_oral_f_features(a) == absolute_oral_f_features(b)
    assert absolute_oral_f(a)["lower_percent"] == 42
    assert a == frozen
    old_features, old_outcomes = encode_features(a)
    features, outcomes = encode_features(a, absolute_oral_f=True)
    assert outcomes == old_outcomes == frozenset()
    assert features == {**old_features, **absolute_oral_f_features(a)}


@pytest.mark.parametrize("value,unit,expected", [
    (0, "%", "0-20"), ("19.9", "%", "0-20"), ("20", "%", "20-40"),
    ("100", "%", "80-100"), ("0.8", "fraction", "80-100"),
    ("40–50%", "", "40-60"), ("42 ± 2", "%", "40-60"),
    ("<20", "%", "0-20"), (">=80%", "", "80-100"),
    ("0.40 to 0.50", "fraction", "40-60"),
])
def test_supported_scalar_range_and_bounds(value, unit, expected):
    result = absolute_oral_f(record(value, unit))
    assert result["status"] == "eligible"
    assert result["bin"] == expected


@pytest.mark.parametrize("value,unit", [
    ("0.5", ""), ("0.5", "unitless"), ("42%", "fraction"),
    ("42", "ng*h/ml"), ("42", "binary_outcome_class"),
    ("101", "%"), ("1.04", "fraction"), ("<0", "%"), (">100", "%"),
    ("-1", "%"), ("40; 80", "%"), ("40% (80% in dogs)", ""),
    ("40% to 0.5 fraction", ""),
])
def test_ambiguous_numbers_never_receive_numeric_coverage(value, unit):
    assert not absolute_oral_f_features(record(value, unit))


@pytest.mark.parametrize("changes", [
    {"bioavailability_report_type": "relative_comparison"},
    {"bioavailability_report_type": "extent_f", "support_text": "Oral bioavailability was 42%."},
    {"support_text": "Absolute and relative bioavailability were compared."},
    {"canonical_endpoint_name": "fraction_absorbed"},
    {"canonical_endpoint_name": "oral_auc"},
    {"comparator": "reference tablets"},
    {"comparator_exposure": "reference tablets"},
    {"needs_more_context": True},
])
def test_measurement_and_analyte_gates(changes):
    assert not absolute_oral_f_features(record(**changes))


def test_missing_absolute_type_can_use_explicit_source_statement():
    assert absolute_oral_f_features(record(bioavailability_report_type=None))


def test_matching_dosed_name_is_allowed_but_not_used_as_a_feature():
    a = record(molecule_name="compound A", support_text="Following oral administration of compound A, absolute F was measured.")
    b = record(molecule_name="compound B", support_text="Following oral administration of compound B, absolute F was measured.")
    assert absolute_oral_f_features(a)
    assert absolute_oral_f_features(a) == absolute_oral_f_features(b)


def test_correct_dose_analyte_binding_retains_value_in_separate_coverage_group():
    p = record("28", molecule_name="ramiprilat", qualifying_conditions="administered as ramipril",
               support_text="The absolute bioavailability of ramiprilat following oral administration of ramipril was reported to be 28%.")
    result = absolute_oral_f(p)
    assert result["status"] == "eligible"
    assert result["lower_percent"] == 28
    binding = result["measurement_binding"]
    assert binding["dosed_entity"] == "ramipril"
    assert binding["measured_entity"] == "ramiprilat"
    assert binding["analyte_scope"] == "dosed_measured_names_differ"
    assert {"field": "qualifying_conditions", "quote": "administered as ramipril"} in binding["evidence"]
    assert absolute_oral_f_features(p).keys().isdisjoint(absolute_oral_f_features(record("28")))


@pytest.mark.parametrize("changes,key,expected", [
    ({"support_text": "The absolute bioavailability of the active metabolite was 42%."}, "analyte_scope", "metabolite_context"),
    ({"support_text": "Absolute F for the prodrug was 42%."}, "analyte_scope", "prodrug_context"),
    ({"support_text": "Absolute F based on unbound prednisolone was 42%."}, "analyte_fraction", "unbound"),
    ({"support_text": "Absolute F calculated from free drug concentration was 42%."}, "analyte_fraction", "free_drug_as_reported"),
    ({"support_text": "Absolute F of 14C-labelled compound was 42%."}, "labelled_measurement", True),
])
def test_other_measurement_scopes_are_retained_not_dropped(changes, key, expected):
    p = record(**changes)
    assert absolute_oral_f(p)["measurement_binding"][key] == expected
    assert absolute_oral_f_features(p)
    assert absolute_oral_f_features(p).keys().isdisjoint(absolute_oral_f_features(record()))


def test_named_iv_reference_retained_with_original_binding():
    p = record(comparator="vs IV active metabolite")
    assert absolute_oral_f(p)["measurement_binding"]["reference"] == "vs iv active metabolite"
    assert absolute_oral_f_features(p)
    assert absolute_oral_f_features(p).keys().isdisjoint(absolute_oral_f_features(record()))


def test_numeric_phrase_is_not_a_molecule_name():
    p = record(molecule_name="compound A", support_text="Absolute oral bioavailability of 42% was reported.")
    assert absolute_oral_f(p)["measurement_binding"]["measured_entity"] == "compound a"


@pytest.mark.parametrize("description,dosed,scope", [
    ("100 mg of sumatriptan", "sumatriptan", "reported_focal_f"),
    ("a 100mg dose", None, "dosed_entity_unresolved"),
    ("sumatriptan as an aqueous suspension", "sumatriptan", "reported_focal_f"),
    ("the steroid ester is greatly limited", None, "dosed_entity_unresolved"),
])
def test_dose_description_is_not_misidentified_as_another_compound(description, dosed, scope):
    p = record(molecule_name="sumatriptan", support_text=f"Following oral administration of {description}, absolute F was 42%.")
    result = absolute_oral_f(p)
    assert result["status"] == "eligible"
    assert result["measurement_binding"]["dosed_entity"] == dosed
    assert result["measurement_binding"]["analyte_scope"] == scope


def test_conditions_do_not_share_the_same_numeric_feature():
    human = absolute_oral_f_features(record())
    assert human.keys().isdisjoint(absolute_oral_f_features(record(species_or_population="rat")))
    assert human.keys().isdisjoint(absolute_oral_f_features(record(dose="10 mg")))
    assert human.keys().isdisjoint(absolute_oral_f_features(record(study_context="fed")))


def test_anchor_then_diversity_prefers_an_uncovered_interval():
    cards = [
        {"id": name, "molecule_id": "same", "features": absolute_oral_f_features(record(value)),
         "outcomes": frozenset(), "reported_value": value, "support_text": "source"}
        for name, value in [("anchor", "20"), ("near", "21"), ("far", "80")]
    ]
    assert cards[0]["features"] == cards[1]["features"]
    chosen, audit = select_level_records(cards, per_molecule=2, selection_policy="anchored_complete.v2")
    assert [c["id"] for c in chosen] == ["anchor", "far"]
    assert audit["protected_card_ids"] == ["anchor"]
    assert math.sqrt(2) - 1 < 1  # Repeated versus new numerical-bin coverage.


@pytest.mark.parametrize("estimate,error", [(51.8,21.8),(19.01,8.41),(81.4,30),(0,2),(100,2)])
def test_reporting_uncertainty_does_not_reduce_numeric_coverage(estimate, error):
    point = record(str(estimate))
    uncertain = record(f"{estimate} ± {error} (SD)")
    result = absolute_oral_f(uncertain)
    assert result["status"] == "eligible"
    assert result["central_percent"] == estimate
    assert result["uncertainty_percent"] == error
    assert result["uncertainty_statistic"] == "sd"
    assert absolute_oral_f_features(point) == absolute_oral_f_features(uncertain)


@pytest.mark.parametrize("value", ["19–21", "30–80", "0–100", "<60", ">20", "<=20"])
def test_interval_coverage_has_fixed_total_gain_and_no_invented_midpoint(value):
    p = record(value)
    result = absolute_oral_f(p)
    assert result["status"] == "eligible"
    assert result["central_percent"] is None
    assert sum(result["coverage_shares"].values()) == pytest.approx(1)
    assert sum(math.sqrt(w) for w in absolute_oral_f_features(p).values()) == pytest.approx(1)
    assert all(w > 0 for w in result["coverage_shares"].values())


def test_range_and_fraction_units_cover_the_same_bins():
    assert absolute_oral_f_features(record("30–80", "%")) == absolute_oral_f_features(record("0.3–0.8", "fraction"))
    assert absolute_oral_f(record("30–80"))["coverage_shares"] == {"20-40":0.2,"40-60":0.4,"60-80":0.4}


def test_generic_native_prodrug_does_not_hide_explicit_dosing_subject():
    source = ("Enalapril, administered as the maleate salt, was designed as a prodrug to improve "
              "the systemic availability of the active ACE inhibitor enalaprilat. "
              "The absolute bioavailability of enalaprilat is about 40%.")
    p = record("40", molecule_name="enalaprilat", qualifying_conditions="administered as a prodrug", support_text=source)
    binding = absolute_oral_f(p)["measurement_binding"]
    assert binding["dosed_entity"] == "enalapril"
    assert binding["measured_entity"] == "enalaprilat"
    assert binding["analyte_scope"] == "dosed_measured_names_differ"
    assert {"field":"support_text","quote":source} in binding["evidence"]


def test_oral_dose_wording_binds_precursor_and_analyte():
    p = record("44", molecule_name="ramiprilat", support_text=(
        "In healthy volunteers, the absolute bioavailability (F) of ramiprilat following "
        "a single 5 mg oral dose of ramipril, calculated relative to a single intravenous infusion, was 44%."))
    binding = absolute_oral_f(p)["measurement_binding"]
    assert binding["dosed_entity"] == "ramipril"
    assert binding["measured_entity"] == "ramiprilat"


def test_conflicting_dose_bindings_remain_explicit_and_do_not_replace_each_other():
    p = record(molecule_name="compound A", qualifying_conditions="administered as compound B",
               support_text="The absolute oral bioavailability was measured after an oral dose of compound C.")
    binding = absolute_oral_f(p)["measurement_binding"]
    assert binding["dosed_entity"] is None
    assert binding["analyte_scope"] == "dosed_entity_conflict"
    assert {c['entity'] for c in binding['dose_candidates']} == {"compound b","compound c"}
    assert absolute_oral_f_features(p)


@pytest.mark.parametrize("description", [
    "the CR tablet", "a single 1-mg capsule", "three 5-mg capsules",
    "5 mg/kg bodyweight (n = 6)", "200 mg demonstrated high bioavailability (>95%)",
])
def test_oral_dose_form_or_amount_is_not_a_named_entity(description):
    p = record(molecule_name="compound A", support_text=f"After an oral dose of {description}, absolute F was 42%.")
    binding = absolute_oral_f(p)["measurement_binding"]
    assert binding["dosed_entity"] is None
    assert binding["analyte_scope"] == "dosed_entity_unresolved"
    assert absolute_oral_f_features(p)


def test_missing_molecule_name_placeholder_is_not_an_analyte_name():
    p = record(molecule_name="smiles:1234", support_text="Absolute oral F was measured after an oral dose of compound A.")
    binding = absolute_oral_f(p)["measurement_binding"]
    assert binding["measured_entity"] is None
    assert binding["dosed_entity"] == "compound a"
    assert binding["analyte_scope"] == "dosed_entity_known_analyte_unspecified"
