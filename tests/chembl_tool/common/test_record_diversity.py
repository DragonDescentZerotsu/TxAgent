"""Scientific boundaries and constrained selection, independent of LLM output."""

import json
import pytest
from tools.chembl_tool.common.record_diversity import (
    prepare_record,
    encode_features,
    select_level_records,
    validate_annotation,
    cache_key,
)


@pytest.mark.parametrize("value", ["unspecified (6)", "unknown (26)", "not specified", "not_reported (2)", "—"])
def test_selection_placeholders_do_not_change_frozen_feature_cleaning(value):
    from tools.chembl_tool.common.record_diversity import selection_text, clean
    assert selection_text(value) == ""
    if value == "unspecified (6)":
        assert clean(value) == value  # Existing annotation/feature contracts are frozen.


@pytest.mark.parametrize("value,unit,expected", [
    ("17.1", "%", True), (0, "ng/ml", True), ("0.20 ×10^-4", "cm/s", True),
    ("0.62 ± 0.11", "fraction", True), ("result_value=14 µM", "", True),
    ("1", "binary_outcome_class", False), ("-1", "binary_outcome_class", False),
    ("classification_label=Group 2B", "", False), ("8-oxodG decreased", "", False),
    ("H2O2 increased 8-oxodG", "", False), ("unspecified (6)", "", False),
    ("dose=50 mg/kg", "", False), ("positive", "", False),
])
def test_numeric_priority_distinguishes_measurements_from_codes_and_names(value,unit,expected):
    from tools.chembl_tool.common.record_diversity import has_measured_value
    assert has_measured_value({'reported_value':value,'reported_unit':unit}) is expected


def test_repaired_anchor_order_and_completeness_preserve_actual_readouts():
    from tools.chembl_tool.common.progressive_assay_reasoning import _card_order
    raw=[{'card_id':'placeholder','reported_value':'unspecified (6)'},
         {'card_id':'code','reported_value':'1','reported_unit':'binary_outcome_class'},
         {'card_id':'measurement','reported_value':'17.1','reported_unit':'%'}]
    for c in raw:c['support_text']='Original source.'
    assert sorted(raw,key=lambda c:_card_order(c,measured_values_only=True))[0]['card_id']=='measurement'
    cards=[{**card(c['card_id'],'m','increase' if c['card_id']!='placeholder' else 'decrease'),**c} for c in sorted(raw,key=lambda c:_card_order(c,measured_values_only=True))]
    selected,_=select_level_records(cards,per_molecule=2,selection_policy='anchored_complete.v2')
    assert [c['id'] for c in selected]==['measurement','code']


@pytest.mark.parametrize('activation,panel,expected',[
    ('absent','Salmonella typhimurium TA98',0),
    ('absent','Salmonella typhimurium TA98 and TA100',1),
    ('both','Salmonella typhimurium TA98',1),
    ('present','Salmonella typhimurium TA98',3),
    ('absent','Salmonella typhimurium TA100',3),
    ('unknown','Salmonella typhimurium TA98',2),
    ('absent','unknown',2),
])
def test_native_ames_conditions_rank_without_reading_results(activation,panel,expected):
    from tools.chembl_tool.common.record_diversity import ames_condition_rank
    fields={'canonical_endpoint_name':'bacterial_reverse_mutation','metabolic_activation':activation,'test_system':panel}
    for label in ['positive','negative']:
        fields['mutagenicity_result']=label
        assert ames_condition_rank('metabolic_activation=absent+strain_panel=TA98',fields)==expected
    fields['canonical_endpoint_name']='micronucleus'
    assert ames_condition_rank('metabolic_activation=absent+strain_panel=TA98',fields)==2


def test_condition_matched_negative_survives_other_molecule_negative_coverage():
    cards=[card('wrong_positive','m','positive',{'many':100.0},rank=3),
           card('matched_negative','m','negative',rank=1),
           card('other_negative','n','negative',rank=0)]
    for c in cards:c.update(reported_value='positive',support_text='Source')
    chosen,_=select_level_records(cards,per_molecule=1,selection_policy='anchored_complete.v2')
    assert {c['id'] for c in chosen}=={'matched_negative','other_negative'}


def source(**changes):
    return dict(
        canonical_record_id="r",
        molecule_name="compound A",
        canonical_endpoint_name="cell proliferation",
        effect_direction="increase",
        species="human",
        support_text="Compound A increased cell proliferation.",
        **changes,
    )


def test_native_context_retained_and_admin_labels_excluded():
    row = source(
        oral_dose="60 mg",
        study_context="fasted human subjects",
        metabolic_activation="with S9",
        comparator="vehicle",
        assay_type="cell assay",
        Y=1,
        is_gold_voter=True,
    )
    p = prepare_record(row)
    assert not p["needs_llm"] and p["outcome_kinds"] == ["increase"]
    assert p["groups"]["exposure"]["oral_dose"] == "60 mg"
    assert p["groups"]["exposure"]["metabolic_activation"] == "with S9"
    assert p["groups"]["comparison_role"]["comparator"] == "vehicle"
    f, o = encode_features(p)
    assert ("cell proliferation", "increase") in o
    assert not any("is_gold" in k or "/Y" in k for k in f)
    changed = prepare_record({**row, "Y": 0, "is_gold_voter": False})
    assert encode_features(changed) == (f, o)


def test_reviewed_null_overrides_old_result_and_canonical_support_wins():
    row = {
        "canonical_record_id": "r",
        "canonical_endpoint_name": "growth",
        "raw_record_json": json.dumps(
            {"effect_direction": "increase", "support_text": "old"}
        ),
        "reviewed_record_json": json.dumps(
            {"effect_direction": None, "support_text": "review"}
        ),
        "support_text": "current repaired support",
    }
    p = prepare_record(row)
    assert p["needs_llm"] and not p["outcome_kinds"]
    assert p["input"]["support_text"] == "current repaired support"


def test_focal_identity_uses_reviewed_native_fields():
    row = {
        "canonical_record_id": "r",
        "canonical_smiles": "CCO",
        "raw_record_json": json.dumps({"molecule_name": "old name"}),
        "reviewed_record_json": json.dumps({"molecule_name": "ethanol"}),
    }
    p = prepare_record(row)
    assert p["input"]["focal_molecule"] == "ethanol"
    assert p["input"]["focal_smiles"] == "CCO"


def test_numeric_and_method_only_do_not_acquire_polarity():
    for value in ("37", "", "not increased", "increase and decrease"):
        row = {
            **source(),
            "effect_direction": None,
            "canonical_measurement_text": value,
        }
        p = prepare_record(row)
        assert p["needs_llm"] and not p["outcome_kinds"]
        assert not encode_features(p)[1]


def test_focal_scope_and_quotes_gate_outcome_features():
    p = prepare_record(source())
    a = {
        "outcomes": [
            {
                "endpoint": "cell proliferation",
                "kind": "increase",
                "focal_scope": "unclear",
                "evidence": [
                    {"field": "support_text", "quote": "increased cell proliferation"}
                ],
            }
        ],
        "assay": [],
        "system": [],
        "exposure": [],
        "comparison_role": [],
        "unresolved": ["subject"],
    }
    a = validate_annotation(a, p)
    assert not encode_features(p, a)[1]
    a["outcomes"][0]["evidence"][0]["quote"] = "fabricated quote"
    with pytest.raises(ValueError):
        validate_annotation(a, p)


def test_cache_binds_source_and_model_settings():
    p = prepare_record(source())
    assert cache_key(p, {"model": "a"}) != cache_key(p, {"model": "b"})
    q = prepare_record({**source(), "support_text": "Changed source"})
    assert cache_key(p, {"model": "a"}) != cache_key(q, {"model": "a"})


def test_quote_pointer_normalization_requires_exact_unambiguous_source():
    from tools.chembl_tool.common.record_diversity import BOUND_ANNOTATION_VERSION

    p = prepare_record(source())
    value = {
        "outcomes": [{
            "endpoint": "cell proliferation", "kind": "increase", "focal_scope": "focal",
            "conditions": [], "comparator": [], "statistical_support": "not_reported",
            "attribution_basis": "record_fields",
            "evidence": [{"field": "source_fields.effect_direction", "quote": "increase"}],
        }],
        "assay": [], "system": [], "exposure": [], "comparison_role": [], "unresolved": [],
    }
    result = validate_annotation(value, p, BOUND_ANNOTATION_VERSION)
    assert result["outcomes"][0]["evidence"][0] == {"field": "effect_direction", "quote": "increase"}
    value["outcomes"][0]["evidence"][0]["field"] = "source_fields"
    assert validate_annotation(value, p, BOUND_ANNOTATION_VERSION)["outcomes"][0]["evidence"][0]["field"] == "effect_direction"
    p["input"]["source_fields"]["other_result"] = "increase"
    with pytest.raises(ValueError, match="exact source substring"):
        validate_annotation(value, p, BOUND_ANNOTATION_VERSION)
    value["outcomes"][0]["evidence"][0] = {"field": "source_fields.effect_direction", "quote": "decrease"}
    with pytest.raises(ValueError, match="exact source substring"):
        validate_annotation(value, p, BOUND_ANNOTATION_VERSION)


def card(cid, molecule, outcome, features=None, rank=0):
    return {
        "id": cid,
        "molecule_id": molecule,
        "outcomes": [("growth", outcome)],
        "features": features or {"context": 1.0},
        "condition_rank": rank,
    }


def test_quote_aliases_cannot_bypass_opposite_context_check():
    from tools.chembl_tool.common.record_diversity import BOUND_ANNOTATION_VERSION

    p = prepare_record(source(dose="1 uM"))
    p["input"]["support_text"] = "growth increased; growth decreased"
    outcomes = []
    for kind, field, quote in (
        ("increase", "source_fields.dose", "growth increased"),
        ("decrease", "dose", "growth decreased"),
    ):
        outcomes.append({
            "endpoint": "growth", "kind": kind, "focal_scope": "focal",
            "evidence": [{"field": "support_text", "quote": quote}],
            "conditions": [{"field": field, "quote": "1 uM"}],
            "comparator": [], "statistical_support": "not_reported",
            "attribution_basis": "record_fields",
        })
    annotation = {
        "outcomes": outcomes, "assay": [], "system": [], "exposure": [],
        "comparison_role": [], "unresolved": [],
    }
    with pytest.raises(ValueError, match="Opposite outcomes"):
        validate_annotation(annotation, p, BOUND_ANNOTATION_VERSION)


def test_level_outcome_priority_keeps_opposite_with_per_molecule_caps():
    cards = [
        card("a", "m", "increase"),
        card("b", "m", "increase", {"many_new_contexts": 10.0}),
        card("c", "m", "decrease"),
    ]
    chosen, audit = select_level_records(
        cards, per_molecule=2, previous_features=[{"context": 1.0}]
    )
    assert {c["outcomes"][0][1] for c in chosen} == {"increase", "decrease"}
    assert audit["selected_per_molecule"] == {"m": 2}
    assert audit["selected_outcomes"] == 2


def test_feature_insertion_order_cannot_break_an_equal_gain_tie():
    features = [("large", 1.0)] + [(f"small_{i}", 1e-34) for i in range(32)]
    cards = [
        card("first", "m", "increase", dict(features)),
        card("second", "m", "increase", dict(reversed(features))),
    ]
    chosen, _ = select_level_records(cards, per_molecule=1)
    assert [c["id"] for c in chosen] == ["first"]
    cards[0]["features"], cards[1]["features"] = cards[1]["features"], cards[0]["features"]
    chosen, _ = select_level_records(cards, per_molecule=1)
    assert [c["id"] for c in chosen] == ["first"]


def test_condition_priority_and_molecule_selection_are_hard_constraints():
    cards = [
        card("a", "m", "increase", rank=0),
        card("b", "m", "decrease", rank=1),
        card("c", "n", "decrease", rank=0),
    ]
    chosen, _ = select_level_records(cards, per_molecule=1)
    assert {c["id"] for c in chosen} == {"a", "c"}
    assert select_level_records(cards, per_molecule=0)[0] == []
    with pytest.raises(ValueError):
        select_level_records(cards + cards[:1])


def test_anchored_selection_seeds_all_molecules_before_outcome_coverage():
    cards = [
        card("m_anchor", "m", "increase"),
        card("m_repeat", "m", "increase", {"context": 100.0}),
        card("m_opposite", "m", "decrease"),
        card("n_anchor", "n", "increase"),
    ]
    for c in cards:
        c.update(reported_value="positive", support_text="Reported outcome.")
    chosen, audit = select_level_records(
        cards, per_molecule=2, selection_policy="anchored_complete.v1"
    )
    assert [c["id"] for c in chosen] == ["m_anchor", "n_anchor", "m_opposite"]
    assert audit["protected_card_ids"] == ["m_anchor", "n_anchor"]
    assert audit["selected_per_molecule"] == {"m": 2, "n": 1}


@pytest.mark.parametrize("missing", [None, "", "unknown", "not reported", "N/A", float("nan")])
def test_complete_records_precede_incomplete_novel_outcomes_and_still_fill(missing):
    cards = [
        card("anchor", "m", "increase"),
        card("incomplete", "m", "decrease"),
        card("complete", "m", "increase"),
    ]
    for c in cards:
        c.update(reported_value=0, support_text="Measured zero response.")
    cards[1]["reported_value"] = missing
    chosen, _ = select_level_records(cards, per_molecule=2, selection_policy="anchored_complete.v1")
    assert [c["id"] for c in chosen] == ["anchor", "complete"]
    chosen, _ = select_level_records(cards, per_molecule=5, selection_policy="anchored_complete.v1")
    assert [c["id"] for c in chosen] == ["anchor", "complete", "incomplete"]


def test_anchor_features_reduce_subsequent_gain_and_missing_support_is_incomplete():
    cards = [
        card("anchor", "m", "increase", {"same": 100.0}),
        card("repeat", "m", "increase", {"same": 100.0}),
        card("new", "m", "increase", {"different": 25.0}),
        card("no_support", "m", "decrease", {"different": 1000.0}),
    ]
    for c in cards:
        c.update(reported_value="negative", support_text="No response was observed.")
    cards[-1]["support_text"] = "not_reported"
    chosen, _ = select_level_records(cards, per_molecule=2, selection_policy="anchored_complete.v1")
    assert [c["id"] for c in chosen] == ["anchor", "new"]


def test_anchored_policy_preserves_condition_priority_zero_budget_and_input_contract():
    cards = [card("wrong_condition", "m", "increase", rank=1), card("matched", "m", "decrease")]
    for c in cards:
        c.update(reported_value="positive", support_text="Outcome.")
    cards[1]["reported_value"] = "unknown"
    chosen, _ = select_level_records(cards, per_molecule=1, selection_policy="anchored_complete.v1")
    assert [c["id"] for c in chosen] == ["matched"]
    assert select_level_records(cards, per_molecule=0, selection_policy="anchored_complete.v1")[0] == []
    with pytest.raises(ValueError, match="requires raw"):
        select_level_records([card("raw_missing", "m", "increase")], selection_policy="anchored_complete.v1")


def test_same_outcome_different_conditions_remain_distinguishable():
    a = prepare_record(source(exposure_time="24 h", species_or_population="rat"))
    b = prepare_record(source(exposure_time="72 h", species_or_population="mouse"))
    af, ao = encode_features(a)
    bf, bo = encode_features(b)
    assert ao == bo and af != bf


def test_explicit_result_does_not_hide_opposite_claim_or_intervention():
    row = {
        **source(),
        "support_text": "Compound A increased growth; conversely, another study found inhibition.",
    }
    p = prepare_record(row)
    assert p["needs_llm"] and "contrast_or_multiple_claims_in_text" in p["llm_reasons"]
    p = prepare_record(source(perturbation="transporter inhibition"))
    assert p["needs_llm"] and not encode_features(p)[1]


def test_source_specific_explicit_calls_need_no_semantic_guess():
    row = {
        **source(),
        "effect_direction": None,
        "result_status": "damage_or_response_decreased",
        "molecule_role": "parent_test_agent",
    }
    p = prepare_record(row)
    assert not p["needs_llm"] and p["outcome_kinds"] == ["decrease"]
    row["result_status"] = "quantitative_result_without_directional_call"
    p = prepare_record(row)
    assert not p["needs_llm"] and not encode_features(p)[1]


@pytest.mark.parametrize("filter_field", ["canonical_smiles", "molecule_id"])
def test_scoped_inventory_deduplicates_splits_and_preserves_level_scope(
    tmp_path, monkeypatch, filter_field
):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tools.chembl_tool.paper_experiments import (
        prepare_record_diversity as inventory,
    )

    rows = []
    for i, (smiles, group, eligible) in enumerate(
        [
            ("A", "g1", True),
            ("A", "g2", True),
            ("B", "g1", True),
            ("B", "g2", True),
            ("C", "g1", True),
            ("A", "g1", False),
        ]
    ):
        rows.append(
            {
                **source(),
                "canonical_record_id": str(i),
                "source_id": "native",
                "canonical_smiles": smiles,
                "molecule_id": "source_" + smiles,
                "group_id": group,
                "retrieval_eligible": eligible,
            }
        )
    records = tmp_path / "records.parquet"
    pq.write_table(pa.Table.from_pylist(rows), records)
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "manifest.json").write_text(
        json.dumps(
            {
                "levels": [
                    {"level": 1, "source_groups": ["g1"]},
                    {"level": 2, "source_groups": ["g2"]},
                ]
            }
        )
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "artifact_root": str(tmp_path),
                "tasks": {
                    "fixture": {
                        "overlay_is_canonical": True,
                        "source_records": str(records),
                        "catalog": "catalog",
                        "overlay_records_sha256": "fixture",
                    }
                },
            }
        )
    )
    monkeypatch.setattr(inventory, "REGISTRY", registry)
    scope = tmp_path / "scope.json"
    prefix = "source_" if filter_field == "molecule_id" else ""
    a, b = prefix + "A", prefix + "B"
    scope.write_text(
        json.dumps(
            {
                "fixture": {
                    "valid": {"1": [a]},
                    "test": {"1": [a, b], "2": [a]},
                }
            }
        )
    )
    output = tmp_path / "filtered"
    result = inventory.scan_task(
        ("fixture", str(output), 1, False, str(scope), filter_field)
    )
    report = json.loads((output / "fixture.json").read_text())
    assert result["rows"] == 3
    assert report["subset_counts"]["valid"]["1"]["rows"] == 1
    assert report["subset_counts"]["test"]["1"]["rows"] == 2
    assert report["subset_counts"]["test"]["2"]["rows"] == 1
    assert not report["unmatched_selected_values"]
    all_rows = inventory.scan_task(
        ("fixture", str(tmp_path / "all"), 1, False, None, filter_field)
    )
    assert all_rows["rows"] == 5
    pq.write_table(
        pa.Table.from_pylist([{**r, "retrieval_eligible": False} for r in rows]),
        records,
    )
    empty = inventory.scan_task(
        ("fixture", str(tmp_path / "empty"), 1, False, str(scope), filter_field)
    )
    assert empty["rows"] == empty["llm_candidates"] == 0


def bound_annotation(p, kind="increase", statistics="not_reported"):
    return {
        "outcomes": [
            {
                "endpoint": "cell_proliferation",
                "kind": kind,
                "focal_scope": "focal",
                "attribution_basis": "record_fields",
                "evidence": [{"field": "effect_direction", "quote": "increase"}],
                "conditions": [{"field": "species", "quote": "human"}],
                "comparator": [],
                "statistical_support": statistics,
            }
        ],
        "assay": [],
        "system": [],
        "exposure": [],
        "comparison_role": [],
        "unresolved": [],
    }


def test_wrong_quote_field_reports_literal_locations_without_silently_rebinding():
    from tools.chembl_tool.common.record_diversity import BOUND_ANNOTATION_VERSION
    p = {'input':{'source_fields':{'exposure_schedule':'Drug plus IR compared with IR alone',
                                 'endpoint_and_comparator':'Activity relative to IR alone'},
                  'support_text':'Activity increased after combined treatment.'}}
    a = bound_annotation(p)
    a['outcomes'][0].update(evidence=[{'field':'support_text','quote':p['input']['support_text']}],
                           conditions=[], comparator=[{'field':'support_text','quote':'IR alone'}])
    with pytest.raises(ValueError, match='literal quote occurs in these fields: endpoint_and_comparator, exposure_schedule'):
        validate_annotation(a, p, BOUND_ANNOTATION_VERSION)
    assert a['outcomes'][0]['comparator'][0]['field'] == 'support_text'
    a['outcomes'][0]['comparator'][0]['field'] = 'exposure_schedule'
    assert validate_annotation(a, p, BOUND_ANNOTATION_VERSION) == a


def test_quote_repair_recovers_literal_typography_and_unique_field_without_mutation():
    import copy
    from tools.chembl_tool.common.record_diversity import repair_annotation_quotes, BOUND_ANNOTATION_VERSION
    p = {"input": {"source_fields": {"effect_direction": "increase", "species": "human"},
                   "support_text": "The cells increased 2–3 fold at 10 µM."}}
    a = bound_annotation(p)
    a["outcomes"][0]["evidence"] = [{"field": "extra_details", "quote": "the cells increased 2-3  fold at 10 μM"}]
    frozen = copy.deepcopy(a)
    repaired, edits = repair_annotation_quotes(a, p)
    assert a == frozen
    assert len(edits) == 1
    assert repaired["outcomes"][0]["evidence"] == [{"field": "support_text", "quote": "The cells increased 2–3 fold at 10 µM"}]
    assert validate_annotation(repaired, p, BOUND_ANNOTATION_VERSION)


@pytest.mark.parametrize("quote", ["increased 20 fold", "decreased 2 fold", "did not increase"])
def test_quote_repair_never_changes_numbers_direction_or_negation(quote):
    from tools.chembl_tool.common.record_diversity import repair_annotation_quotes
    p = {"input": {"source_fields": {}, "support_text": "increased 2 fold"}}
    a = {"outcomes": [{"evidence": [{"field": "support_text", "quote": quote}]}]}
    assert repair_annotation_quotes(a, p) == (a, [])


def test_quote_repair_abstains_on_ambiguous_field_and_keeps_semantic_validation():
    from tools.chembl_tool.common.record_diversity import repair_annotation_quotes, BOUND_ANNOTATION_VERSION
    p = {"input": {"source_fields": {"effect_direction": "increase", "species": "human", "a": "same value", "b": "same value"}, "support_text": ""}}
    a = bound_annotation(p, kind="no_change", statistics="not_significant")
    repaired, edits = repair_annotation_quotes(a, p)
    with pytest.raises(ValueError, match="Non-significance"):
        validate_annotation(repaired, p, BOUND_ANNOTATION_VERSION)
    a["outcomes"][0]["evidence"] = [{"field": "missing", "quote": "same value"}]
    assert repair_annotation_quotes(a, p) == (a, [])


def test_catalog_round_trip_preserves_literal_spans_and_rejects_unknown_references():
    from tools.chembl_tool.common.record_diversity import annotation_quote_catalog, resolve_annotation_quotes, BOUND_ANNOTATION_VERSION
    p = {"input": {"source_fields": {"effect_direction": "increase", "species": "human"}, "support_text": "Value 2.5; Day 2 positive but Day 4 negative."}}
    catalog = annotation_quote_catalog(p)
    assert catalog == annotation_quote_catalog(p)
    source = {**p['input']['source_fields'], 'support_text':p['input']['support_text']}
    assert all(span['quote'] in source[span['field']] for span in catalog.values())
    assert any(span['quote']=='Value 2.5' for span in catalog.values())
    a = bound_annotation(p)
    a['outcomes'][0]['evidence'] = [{'field':'effect_direction','quote':'Q0000'}]
    b = resolve_annotation_quotes(a,catalog)
    assert a['outcomes'][0]['evidence'][0]['quote']=='Q0000'
    assert validate_annotation(b,p,BOUND_ANNOTATION_VERSION)
    a['outcomes'][0]['evidence'][0]['quote']='Q99999'
    with pytest.raises(ValueError,match='Unknown source quote'):
        resolve_annotation_quotes(a,catalog)
    a['outcomes'][0]['evidence'] = [{'field':'Q0000','quote':'increase'}]
    assert resolve_annotation_quotes(a,catalog)['outcomes'][0]['evidence'] == [
        {'field':'effect_direction','quote':'increase'}
    ]
    a['outcomes'][0]['evidence'][0]['quote'] = 'decrease'
    assert resolve_annotation_quotes(a,catalog)['outcomes'][0]['evidence'][0]['field'] == 'Q0000'


def test_v2_prompt_and_cache_remain_frozen():
    import hashlib
    from tools.chembl_tool.common.record_diversity import extraction_system_prompt

    assert hashlib.sha256(extraction_system_prompt().encode()).hexdigest() == (
        "e86de5a711547f20cc60bb89219a6321de7adf42f96240f6fba78619a27ea8e6"
    )
    assert cache_key(
        {"input": {"source_fields": {}, "support_text": ""}}, {"model": "fixture"}
    ) == ("e81de538fa50dc3776d257723ee26eabba0bed9cfcbdfcfbe6c938b26b1d1162")


def test_endpoint_spelling_normalization_preserves_distinct_readouts():
    from tools.chembl_tool.common.record_diversity import normalize_endpoint

    assert normalize_endpoint("Bacterial_reverse_mutation") == normalize_endpoint(
        "Ames test"
    )
    assert normalize_endpoint("brain_to_blood_ratio") != normalize_endpoint(
        "brain-to-plasma ratio"
    )
    assert normalize_endpoint("MRP1 substrate status") != normalize_endpoint(
        "BCRP substrate status"
    )


def test_bound_quotes_and_attribution_validated_without_requiring_display_name():
    from tools.chembl_tool.common.record_diversity import BOUND_ANNOTATION_VERSION as v3

    p = prepare_record({**source(), "molecule_name": ""})
    a = bound_annotation(p)
    assert validate_annotation(a, p, v3) == a
    assert cache_key(p, {}, v3) != cache_key(p, {})
    a["outcomes"][0]["comparator"] = [
        {"field": "support_text", "quote": "invented control"}
    ]
    with pytest.raises(ValueError, match="exact source"):
        validate_annotation(a, p, v3)
    a["outcomes"][0]["comparator"] = []
    a["outcomes"][0]["attribution_basis"] = "unclear"
    with pytest.raises(ValueError, match="Unclear attribution"):
        validate_annotation(a, p, v3)


def test_nonsignificant_trend_is_retained_but_not_prioritized_as_confirmed_outcome():
    p = prepare_record(source())
    a = bound_annotation(p, statistics="not_significant")
    features, outcomes = encode_features(p, a)
    assert not outcomes
    assert any(
        "/cell proliferation/increase/statistics:not_significant" in f for f in features
    )
    a["outcomes"][0]["statistical_support"] = "not_reported"
    assert encode_features(p, a)[1] == {("cell proliferation", "increase")}
    a["outcomes"][0]["kind"] = "numeric_only"
    assert not encode_features(p, a)[1]


def test_unresolved_bindings_keep_features_without_hard_outcome_priority():
    p = prepare_record(source())
    a = bound_annotation(p)
    a["unresolved"] = ["The comparator cannot be assigned to this observation."]
    original_features, original_outcomes = encode_features(p, a)
    features, outcomes = encode_features(p, a, require_resolved_outcomes=True)
    assert original_outcomes
    assert features == original_features and not outcomes
    a["unresolved"] = []
    assert encode_features(p, a, require_resolved_outcomes=True)[1] == original_outcomes


def test_outcome_condition_pairs_do_not_collapse_to_record_level_bag():
    from copy import deepcopy

    p = prepare_record(source())
    a = bound_annotation(p)
    a["outcomes"][0]["conditions"] = [
        {"field": "metabolic_activation", "quote": "with S9"}
    ]
    other = deepcopy(a["outcomes"][0])
    other.update(
        kind="decrease",
        conditions=[{"field": "metabolic_activation", "quote": "without S9"}],
    )
    a["outcomes"].append(other)
    b = deepcopy(a)
    b["outcomes"][0]["conditions"], b["outcomes"][1]["conditions"] = (
        b["outcomes"][1]["conditions"],
        b["outcomes"][0]["conditions"],
    )
    af, ao = encode_features(p, a)
    bf, bo = encode_features(p, b)
    assert ao == bo and af != bf


def test_bound_attribution_can_use_a_full_result_quote_from_support_text():
    from tools.chembl_tool.common.record_diversity import BOUND_ANNOTATION_VERSION as v3

    p = prepare_record(source())
    a = bound_annotation(p)
    a["outcomes"][0]["evidence"] = [
        {"field": "support_text", "quote": "Compound A increased cell proliferation."}
    ]
    assert validate_annotation(a, p, v3) == a
    a["outcomes"][0].update(kind="no_change", statistical_support="not_significant")
    with pytest.raises(ValueError, match="Non-significance"):
        validate_annotation(a, p, v3)


def test_opposite_results_require_distinguishable_observation_bindings():
    from copy import deepcopy
    from tools.chembl_tool.common.record_diversity import BOUND_ANNOTATION_VERSION as v3

    p = prepare_record(source(exposure_time="24 h; 72 h"))
    a = bound_annotation(p)
    a["outcomes"].append({**deepcopy(a["outcomes"][0]), "kind": "decrease"})
    with pytest.raises(ValueError, match="identical conditions"):
        validate_annotation(a, p, v3)
    for outcome, time in zip(a["outcomes"], ["24 h", "72 h"]):
        outcome["conditions"].append({"field": "exposure_time", "quote": time})
    assert validate_annotation(a, p, v3) == a


def test_targeted_review_notes_change_only_the_opt_in_annotation_cache():
    from tools.chembl_tool.common.record_diversity import BOUND_ANNOTATION_VERSION as v3

    p = prepare_record(source())
    reviewed = {**p, "review_notes": "Check the explicitly reported comparator."}
    assert cache_key(p, {}) == cache_key(reviewed, {})
    assert cache_key(p, {}, v3) != cache_key(reviewed, {}, v3)


def test_v4_quality_assessments_require_exact_source_quotes_and_keep_v3_separate():
    from copy import deepcopy
    from tools.chembl_tool.common.record_diversity import QUALITY_ANNOTATION_VERSION, BOUND_ANNOTATION_VERSION, extraction_messages, resolve_annotation_quotes
    p=prepare_record({'canonical_record_id':'r','molecule_name':'compound A',
        'canonical_endpoint_name':'mutation','support_text':'Mutation was negative. Solubility limited the assay.'})
    quote={'field':'support_text','quote':'Solubility limited the assay.'}
    a={'outcomes':[{'endpoint':'mutation','kind':'negative','focal_scope':'focal',
        'evidence':[{'field':'support_text','quote':'Mutation was negative.'}],
        'conditions':[],'comparator':[],'statistical_support':'not_reported','attribution_basis':'record_fields',
        'evidence_basis':'measured','basis_evidence':[{'field':'support_text','quote':'Mutation was negative.'}],
        'result_scope':'focal_effect','limitations':[{'kind':'method_limitation','evidence':[quote]}]}],
        'assay':[],'system':[],'exposure':[],'comparison_role':[],'unresolved':[]}
    assert validate_annotation(a,p,QUALITY_ANNOTATION_VERSION)==a
    bad=deepcopy(a);bad['outcomes'][0]['limitations'][0]['evidence'][0]['quote']='fabricated limitation'
    with pytest.raises(ValueError,match='exact source substring'):
        validate_annotation(bad,p,QUALITY_ANNOTATION_VERSION)
    bad=deepcopy(a);bad['outcomes'][0]['evidence_basis']='methods_only'
    with pytest.raises(ValueError,match='Methods-only'):
        validate_annotation(bad,p,QUALITY_ANNOTATION_VERSION)
    ref=deepcopy(a);ref['outcomes'][0]['limitations'][0]['evidence'][0]={'field':'support_text','quote':'Q0000'}
    assert resolve_annotation_quotes(ref,{'Q0000':quote})==a
    assert extraction_messages(p,QUALITY_ANNOTATION_VERSION)!=extraction_messages(p,BOUND_ANNOTATION_VERSION)
    # An observed negative assay can coexist with explicitly limited study
    # evaluation. Only the inadequate classification itself cannot be negative.
    limited=deepcopy(a)
    limited['outcomes'][0]['limitations'][0]['kind']='insufficient_evaluation'
    assert validate_annotation(limited,p,QUALITY_ANNOTATION_VERSION)
    limited['outcomes'][0]['evidence_basis']='classification'
    with pytest.raises(ValueError,match='Insufficient evaluation'):
        validate_annotation(limited,p,QUALITY_ANNOTATION_VERSION)


def test_v4_explicit_unchanged_and_nonsignificance_can_coexist():
    from tools.chembl_tool.common.record_diversity import QUALITY_ANNOTATION_VERSION
    p=prepare_record({'canonical_record_id':'r','canonical_endpoint_name':'enzyme activity',
        'support_text':'Activity remained essentially unchanged; no significant difference.'})
    quote={'field':'support_text','quote':p['input']['support_text']}
    a={'outcomes':[{'endpoint':'enzyme activity','kind':'no_change','focal_scope':'focal',
        'evidence':[quote],'conditions':[],'comparator':[],'statistical_support':'not_significant',
        'attribution_basis':'record_fields','evidence_basis':'measured','basis_evidence':[quote],
        'result_scope':'focal_effect','limitations':[]}],
        'assay':[],'system':[],'exposure':[],'comparison_role':[],'unresolved':[]}
    assert validate_annotation(a,p,QUALITY_ANNOTATION_VERSION)
    a['outcomes'][0]['evidence']=[{'field':'support_text','quote':'no significant difference.'}]
    with pytest.raises(ValueError,match='Non-significance'):
        validate_annotation(a,p,QUALITY_ANNOTATION_VERSION)


def test_assessment_view_and_quote_catalog_exclude_old_calls_without_mutating_source():
    from copy import deepcopy
    from tools.chembl_tool.common.record_diversity import (
        QUALITY_ANNOTATION_VERSION as v4, BOUND_ANNOTATION_VERSION as v3,
        extraction_messages, annotation_quote_catalog,
    )
    p=prepare_record({'canonical_record_id':'r','canonical_endpoint_name':'mutation',
        'mutagenicity_result':'positive','canonical_measurement_text':'positive',
        'support_text':'The reported assay result was negative.'})
    original=deepcopy(p)
    visible=json.loads(extraction_messages(p,v4)[1]['content'])
    assert 'mutagenicity_result' not in visible['source_fields']
    assert visible['support_text']=='The reported assay result was negative.'
    assert all(q['field'] not in {'mutagenicity_result','canonical_measurement_text'}
               for q in annotation_quote_catalog(p,v4).values())
    assert json.loads(extraction_messages(p,v3)[1]['content'])['source_fields']['mutagenicity_result']=='positive'
    assert p==original
    q={'field':'mutagenicity_result','quote':'positive'}
    a={'outcomes':[{'endpoint':'mutation','kind':'positive','focal_scope':'focal','evidence':[q],
        'conditions':[],'comparator':[],'statistical_support':'not_reported','attribution_basis':'record_fields',
        'evidence_basis':'measured','basis_evidence':[q],'result_scope':'focal_effect','limitations':[]}],
        'assay':[],'system':[],'exposure':[],'comparison_role':[],'unresolved':[]}
    with pytest.raises(ValueError,match='exact source substring'):
        validate_annotation(a,p,v4)


@pytest.mark.parametrize('measurement',['0.2','decreased; 0.2','result_value=0.2'])
def test_assessment_view_preserves_literal_numeric_measurements(measurement):
    from tools.chembl_tool.common.record_diversity import assessment_source_input
    value={'source_fields':{'canonical_measurement_text':measurement,'canonical_unit_text':'µg/mL'},
           'support_text':'An observed decrease was reported.'}
    assert assessment_source_input(value)==value


def test_assessment_view_omits_generated_scalars_but_keeps_source_measurement():
    from copy import deepcopy
    from tools.chembl_tool.common.record_diversity import assessment_source_input
    source = {'source_fields': {'finite_scalar_value': '-1.0',
        'embedded_unit': 'binary_outcome_class', 'reported_value': '0.12',
        'canonical_measurement_text': '0.12', 'canonical_unit_text': 'ng/mL'},
        'support_text': 'Concentration measured in plasma was 0.12 ng/mL.'}
    original = deepcopy(source)
    visible = assessment_source_input(source)
    assert 'finite_scalar_value' not in visible['source_fields']
    assert 'embedded_unit' not in visible['source_fields']
    assert visible['source_fields']['reported_value'] == '0.12'
    assert visible['source_fields']['canonical_measurement_text'] == '0.12'
    assert visible['support_text'] == source['support_text']
    assert source == original


@pytest.mark.parametrize('code', ['1', '-1'])
def test_assessment_view_does_not_expose_binary_call_as_numeric_measurement(code):
    from tools.chembl_tool.common.record_diversity import assessment_source_input
    source = {'source_fields': {'canonical_measurement_text': code,
        'canonical_unit_text': 'binary_outcome_class', 'measurement_text': code,
        'finite_scalar_value': code, 'reported_value': '0.3'},
        'support_text': 'Concentration was 0.3 ng/mL.'}
    assert assessment_source_input(source) == {'source_fields': {'reported_value': '0.3'},
                                             'support_text': source['support_text']}
