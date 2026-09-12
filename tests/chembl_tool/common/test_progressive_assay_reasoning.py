import json
import random

import pytest

import tools.chembl_tool.common.openai_reasoning_client as client_module
import tools.chembl_tool.paper_experiments.run_conditioned_assay_progressive_curve as runner
from tools.chembl_tool.common.progressive_assay_reasoning import _select_cards
from tools.chembl_tool.common.openai_reasoning_client import OpenAICompatibleClient
from tools.chembl_tool.common.progressive_assay_reasoning import (
    ProgressiveTaskContract,
    append_evidence,
    build_progressive_messages,
    card_alias_maps,
    card_ids,
    extract_cumulative_evidence,
    progressive_state_errors,
    restore_card_ids,
    select_initial_evidence,
    select_progressive_delta,
    state_from_content,
    shortlist_progressive_retrievals,
)


def test_opt_in_endpoint_diversity_retains_complementary_assays_with_same_budget():
    cards = [
        {"card_id": "a", "endpoint": "assay_family=bacterial_reverse_mutation | endpoint_detail=TA100",
         "reported_value": "negative", "support_text": "Negative Ames test", "_assay_key": "study_a"},
        {"card_id": "b", "endpoint": "assay_family=bacterial_reverse_mutation | endpoint_detail=TA98",
         "reported_value": "negative", "support_text": "Negative Ames test", "_assay_key": "study_b"},
        {"card_id": "c", "endpoint": "assay_family=micronucleus | endpoint_detail=CHO-K1",
         "reported_value": "positive", "support_text": "Positive micronucleus test", "_assay_key": "study_c"},
    ]
    assert [c["card_id"] for c in _select_cards(cards, 2)] == ["a", "b"]
    assert [c["card_id"] for c in _select_cards(cards, 2, prefer_distinct_endpoints=True)] == ["a", "c"]
    swapped = [{**c, "reported_value": "positive" if c["reported_value"] == "negative" else "negative"}
               for c in cards]
    assert [c["card_id"] for c in _select_cards(swapped, 2, prefer_distinct_endpoints=True)] == ["a", "c"]
    # When the pool has only one endpoint, use remaining slots for its other valid studies.
    assert [c["card_id"] for c in _select_cards(cards[:2], 2, prefer_distinct_endpoints=True)] == ["a", "b"]


def test_initial_condition_priority_retains_fallback_and_default_order():
    def analog(name, similarity, cards):
        return {"analog_id": name, "similarity": similarity, "cards": {
            cid: {"card_id": cid, "support_text": "reported outcome", "_assay_key": cid,
                  "qualifying_conditions": group, "reported_value": "negative"}
            for cid, group in cards}}
    cumulative = {a["analog_id"]: a for a in [
        analog("closest", .9, [("a", "other")]),
        analog("generic", .8, [("b", "unspecified")]),
        analog("match_low", .1, [("c", "overdose")]),
        analog("match_high", .3, [("d", "other"), ("e", "overdose"), ("f", "overdose")]),
    ]}
    groups = {"a": ["other"], "b": ["no_reported_external_condition"], "c": ["overdose"],
              "d": ["other"], "e": ["overdose"], "f": ["overdose"]}
    original, _ = select_initial_evidence(cumulative, molecule_limit=4, card_limit=2)
    for condition in ("", "no_reported_external_condition"):
        same, _ = select_initial_evidence(cumulative, molecule_limit=4, card_limit=2,
                                         query_condition=condition, card_condition_groups=groups)
        assert same == original
    selected, audit = select_initial_evidence(cumulative, molecule_limit=4, card_limit=2,
                                              query_condition="overdose", card_condition_groups=groups)
    assert list(selected) == ["match_high", "match_low", "generic", "closest"]
    assert list(selected["match_high"]["cards"]) == ["e", "f"]
    assert selected["closest"]["cards"]["a"]["qualifying_conditions"] == "other"
    assert audit["n_selected_cards"] == 5
    card_only, _ = select_initial_evidence(
        cumulative, molecule_limit=3, card_limit=2, query_condition="overdose",
        card_condition_groups=groups, condition_priority_scope="cards_only")
    # A distant condition match cannot displace the original three nearest molecules.
    assert list(card_only) == ["closest", "generic", "match_high"]
    assert list(card_only["match_high"]["cards"]) == ["e", "f"]
    for condition in ("", "no_reported_external_condition"):
        unchanged, _ = select_initial_evidence(
            cumulative, molecule_limit=4, card_limit=2, query_condition=condition,
            card_condition_groups=groups, condition_priority_scope="cards_only")
        assert unchanged == original
    for a in cumulative.values():
        for c in a["cards"].values(): c["reported_value"] = "positive"
    flipped, _ = select_initial_evidence(cumulative, molecule_limit=4, card_limit=2,
                                         query_condition="overdose", card_condition_groups=groups)
    assert {a: list(v["cards"]) for a, v in selected.items()} == {a: list(v["cards"]) for a, v in flipped.items()}


def test_exact_condition_only_preserves_fallback_and_is_direction_blind():
    cards = {cid: {"card_id": cid, "support_text": "reported outcome", "_assay_key": cid,
                   "reported_value": "negative"} for cid in ("a", "b", "c", "d")}
    pool = {"near": {"analog_id": "near", "similarity": .9, "cards": cards},
            "far": {"analog_id": "far", "similarity": .1, "cards": cards}}
    groups = {"a": ["other"], "b": ["no_reported_external_condition"], "d": ["HIV"]}
    def select(condition, scope="cards_exact_only"):
        return select_initial_evidence(pool, molecule_limit=1, card_limit=2,
            query_condition=condition, card_condition_groups=groups, condition_priority_scope=scope)[0]
    original = select_initial_evidence(pool, molecule_limit=1, card_limit=2)[0]
    for condition in ("", "no_reported_external_condition", "overdose"):
        assert select(condition) == original
    assert list(select("HIV")) == ["near"]
    assert list(select("HIV")["near"]["cards"]) == ["d", "a"]
    assert list(select("HIV", "cards_only")["near"]["cards"]) == ["d", "b"]
    for c in cards.values():
        c["reported_value"] = "positive"
    assert list(select("HIV")["near"]["cards"]) == ["d", "a"]


def test_execution_l1_only_keeps_full_prompt_plan(monkeypatch, tmp_path):
    from types import SimpleNamespace
    full_plan = [{"level": i, "endpoint_group": f"family{i}"} for i in range(1, 8)]
    monkeypatch.setattr(runner, "_levels", lambda task: full_plan)
    monkeypatch.setattr(runner, "_task_contract", lambda task: _contract())
    active, _ = select_initial_evidence(extract_cumulative_evidence(_retrieval([
        _neighbor("CCO", .7, [_row("assay", "direct", 1, "measured support")])
    ])))
    level = tmp_path / "levels/level_1"
    level.mkdir(parents=True)
    (level / "prepared.json").write_text(json.dumps({
        "active_evidence": active, "tool_prefetch_complete": True, "should_call_model": True,
        "query_smiles": "CCN", "condition_sentence": "", "query_prior": {},
    }))
    runner._run_query(SimpleNamespace(prepare_only=True, max_level=1),
                      runner.PreparedQuery("example", 0, tmp_path), None)
    request = json.loads((level / "request.json").read_text())
    assert "family7" in json.dumps(request)
    assert list((tmp_path / "levels").iterdir()) == [level]
    with pytest.raises(ValueError, match="execution maximum"):
        runner._execution_levels("example", 8)


def test_condition_policy_and_execution_scope_are_resume_invariants():
    for field in ("execution_max_level", "initial_condition_priority"):
        with pytest.raises(ValueError, match=field):
            runner._merge_resume_manifest({field: "old"}, {field: "new"})


@pytest.mark.parametrize("independent", [False, True])
def test_evidence_ablation_omits_entire_prior_and_preserves_update_mechanism(independent):
    cumulative = extract_cumulative_evidence(_retrieval([
        _neighbor("CCO", .2, [_row("a", "direct", 1, "observed negative")]),
    ]))
    active, _ = select_initial_evidence(cumulative)
    kwargs = dict(contract=_contract(), levels=[{"level": 1}], current_level=1,
        query_smiles="CCN", condition_sentence="reported population",
        query_prior={"endpoint_prior": "SENTINEL_PRIOR", "reasoning_summary": "SENTINEL_EXPLANATION"},
        query_tool_summary={"tool_name": "molecule_properties", "content": "properties\nexact molecular weight: 45"},
        active=active, prior_state=None, independent=independent)
    original = json.loads(build_progressive_messages(**kwargs)[1]["content"])
    messages = build_progressive_messages(**kwargs, omit_query_prior=True, evidence_grounding=True)
    payload = json.loads(messages[1]["content"])
    assert "query_prior" not in payload and "SENTINEL" not in json.dumps(messages)
    assert payload["query"] == original["query"]
    assert payload["active_evidence"] == original["active_evidence"]
    for field in ("architecture", "flip_rule", "update_rule"):
        assert payload["protocol"].get(field) == original["protocol"].get(field)
    assert "distinct from the query" in payload["protocol"]["identity_rule"]
    assert "same standards" in payload["protocol"]["evidence_balance_rule"]


def test_query_identity_exclusion_changes_only_exclusion_and_blocks_cross_identity_reuse():
    kwargs = dict(contract=_contract(), levels=[{"level": 1}], current_level=1,
                  query_smiles="CCN", condition_sentence="", query_prior={},
                  query_tool_summary={}, active={}, prior_state=None)
    original = build_progressive_messages(**kwargs)
    anchored = build_progressive_messages(**kwargs, excluded_query_name="Propylamine")
    assert original[0] == anchored[0]
    payload = json.loads(anchored[1]["content"])
    assert payload["query"].pop("identity_exclusion") == "The query molecule is not Propylamine."
    assert payload == json.loads(original[1]["content"])
    prepared = {"query_smiles": "CCN", "level_definition": {"level": 1}}
    anchored_prepared = {**prepared, "query_identity_exclusion": {"name": "Propylamine", "canonical_smiles": "CCN"}}
    assert runner._excluded_query_name(prepared) == ""
    assert runner._excluded_query_name(anchored_prepared) == "Propylamine"
    assert runner._prepared_model_input(prepared) != runner._prepared_model_input(anchored_prepared)
    with pytest.raises(ValueError, match="true query names"):
        runner._prepared_model_input({**prepared, "query_identity_anchor": {"name": "Ethylamine", "canonical_smiles": "CCN"}})
    assert "Ethylamine" not in json.dumps(anchored)
    for anchor in ({"name": "Other", "canonical_smiles": "CCC"},
                   {"name": "Ethylamine", "canonical_smiles": "CCN", "gold": 1}):
        with pytest.raises(ValueError, match="identity exclusion"):
            runner._excluded_query_name({**prepared, "query_identity_exclusion": anchor})


def test_carcinogens_grounding_keeps_progressive_and_prior_with_scoped_transfer():
    from tools.chembl_tool.tasks.carcinogens.experiment_config import get_progressive_task_contract
    kwargs = dict(contract=get_progressive_task_contract(), levels=[{"level": 1}], current_level=1,
                  query_smiles="CCN", condition_sentence="rodent", query_prior={"prior": "preserved"},
                  query_tool_summary={}, active={}, prior_state=None)
    original = json.loads(build_progressive_messages(**kwargs)[1]["content"])
    grounded = json.loads(build_progressive_messages(**kwargs, evidence_grounding=True)[1]["content"])
    for key in ("architecture", "flip_rule", "update_rule"):
        assert original["protocol"][key] == grounded["protocol"][key]
    assert grounded["query_prior"] == original["query_prior"]
    assert "mixed passage" in grounded["protocol"]["identity_rule"]
    assert "without new direct tumor data" in grounded["protocol"]["endpoint_decision_rule"]
    assert "no detected protection" in grounded["protocol"]["comparison_rule"]
    assert "comparison_rule" not in original["protocol"]


@pytest.mark.parametrize("model_called", [False, True])
def test_resumed_evidence_ablation_drops_none_state_but_keeps_evidence_state(monkeypatch, tmp_path, model_called):
    from types import SimpleNamespace
    monkeypatch.setattr(runner, "_levels", lambda task: [{"level": 1}, {"level": 2}])
    monkeypatch.setattr(runner, "_task_contract", lambda task: _contract())
    first = tmp_path / "levels/level_1"
    first.mkdir(parents=True)
    (first / "output.json").write_text(json.dumps({
        "status": "ok" if model_called else "reused_none", "model_called": model_called,
        "state": {"level": 1, "decision_summary": "PREVIOUS_STATE", "example_prediction": "positive"},
    }))
    active, _ = select_initial_evidence(extract_cumulative_evidence(_retrieval([
        _neighbor("CCO", .2, [_row("a", "mechanism", 2, "new observation")]),
    ])), level=2)
    second = tmp_path / "levels/level_2"
    second.mkdir(parents=True)
    (second / "prepared.json").write_text(json.dumps({
        "active_evidence": active, "tool_prefetch_complete": True, "should_call_model": True,
        "query_smiles": "CCN", "condition_sentence": "", "query_prior": {"summary": "OMIT_THIS"},
        "reasoning_policy": {"omit_query_prior_with_evidence": True, "evidence_grounding": True},
    }))
    runner._run_query(SimpleNamespace(prepare_only=True), runner.PreparedQuery("example", 0, tmp_path), None)
    request = json.loads((second / "request.json").read_text())
    payload = json.loads(request["messages"][1]["content"])
    assert ("prior_state" in payload) is model_called
    assert ("PREVIOUS_STATE" in json.dumps(request)) is model_called
    assert "OMIT_THIS" not in json.dumps(request)
    assert "flip_rule" in payload["protocol"]


@pytest.mark.parametrize("seed", range(8))
def test_molecule_shortlist_preserves_all_selected_cards_at_every_level(seed):
    rng = random.Random(seed)
    retrievals = {}
    molecules = []
    unlocked = {}
    for i in range(90):
        levels = {l for l in range(1, 8) if rng.random() < .55} or {7}
        rows = [_row(f"assay-{i}-{l}-{j}", f"family-{l}", l, f"support-{i}-{l}-{j}")
                for l in levels for j in range(rng.randint(1, 6))]
        n = _neighbor(f"MOLECULE-{i // 2}", round((90-i//2)/100, 2), rows)
        n["molecule_chembl_id"] = str(i)
        molecules.append(n)
        unlocked[str(i)] = levels
    for level in range(1, 8):
        neighbors = []
        for n in molecules:
            rows = [r for r in n["evidence_rows"] if r["source_record_examples"][0]["evidence_family_level"] <= level]
            if rows:
                neighbors.append({**n, "evidence_rows": rows})
        retrievals[level] = _retrieval(neighbors)
    bounded = shortlist_progressive_retrievals(retrievals, unlocked)
    states = []
    for inputs in (retrievals, bounded):
        previous, active, snapshots = {}, {}, []
        for level, retrieval in inputs.items():
            current = extract_cumulative_evidence(retrieval)
            if level == 1:
                active, _ = select_initial_evidence(current)
            else:
                new, aug, _ = select_progressive_delta(previous, current, active, level=level)
                active = append_evidence(active, new, aug)
            snapshots.append(active)
            previous = current
        states.append(snapshots)
    assert states[0] == states[1]


@pytest.mark.parametrize("field,value", [("min_similarity", 0.0), ("reasoning_policy", {"omit_query_prior_with_evidence": True})])
def test_resume_rejects_changed_evidence_ablation(field, value):
    previous = {"min_similarity": .3}
    with pytest.raises(ValueError, match=field):
        runner._merge_resume_manifest(previous, {**previous, field: value})


def test_fork_preparation_keeps_query_indices_and_shared_inputs(monkeypatch, tmp_path):
    import concurrent.futures
    import multiprocessing
    def prepare(*, query_index, record, output_root):
        path = output_root / str(query_index)
        path.write_text(record["drug"])
        return runner.PreparedQuery("example", query_index, path)
    monkeypatch.setattr(runner, "_prepare_query", prepare)
    monkeypatch.setattr(runner, "_PREPARATION_CONTEXT", {
        "records": [{"drug": "CCO"}, {"drug": "CCN"}], "kwargs": {"output_root": tmp_path},
    })
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("fork")) as pool:
        results = list(pool.map(runner._prepare_indexed_query, [1, 0]))
    assert [r.index for r in results] == [1, 0]
    assert [r.query_dir.read_text() for r in results] == ["CCN", "CCO"]
from tools.chembl_tool.tasks.bbb_martins import experiment_config as bbb_config
from tools.chembl_tool.tasks.bioavailability_ma import experiment_config as bio_config
from tools.chembl_tool.tasks.skin_reaction import experiment_config as skin_config


def _contract() -> ProgressiveTaskContract:
    return ProgressiveTaskContract(
        task="example",
        endpoint_name="example endpoint",
        label_scope="example.v1",
        prediction_field="example_prediction",
        positive_prediction="positive",
        negative_prediction="negative",
        system_role="You are an example reasoning model.",
        task_instructions=("Use endpoint-compatible evidence.",),
    )


def _retrieval(neighbors):
    return {"groups": [{"neighbors": neighbors}]}


def _neighbor(smiles, similarity, rows):
    return {
        "molecule_chembl_id": "INTERNAL_SOURCE_ID",
        "canonical_smiles": smiles,
        "standard_inchi_key": f"KEY-{smiles}",
        "similarity": similarity,
        "similarity_bucket": "weak_analog",
        "molecule_relation": "structural_analog",
        "evidence_rows": rows,
    }


def _row(assay, family, level, support, value="1"):
    return {
        "assay_chembl_id": assay,
        "confidence_score": 0.8,
        "source_record_examples": [
            {
                "endpoint_type": "endpoint",
                "reported_value": value,
                "reported_units": "unit",
                "assay_context": "context",
                "species_context": "human",
                "qualifying_conditions": "fasted",
                "support_text": support,
                "evidence_family": family,
                "evidence_family_level": level,
            }
        ],
    }


def test_single_reuse_is_resolved_by_parent_condition_not_query_index(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [
        {"molecule_identity_key": "PARENT-B", "condition_group": "null"},
        {"molecule_identity_key": "PARENT-A", "condition_group": "disease=x"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "bbb_martins" / "none" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"input_jsonl": str(source), "n_items": len(rows)})
    )
    runner._single_source_index.cache_clear()
    mapping = runner._single_source_index("bbb_martins", str(tmp_path))
    assert mapping[("PARENT-A", "disease=x")] == 1
    assert mapping[("PARENT-B", "null")] == 0


def test_progressive_query_prior_is_identity_checked(tmp_path):
    prepared_path = (
        tmp_path
        / "bbb_martins"
        / "queries"
        / "query_idx00003"
        / "levels"
        / "level_1"
        / "prepared.json"
    )
    prepared_path.parent.mkdir(parents=True)
    prepared_path.write_text(
        json.dumps(
            {
                "molecule_identity_key": "PARENT-A",
                "condition_group": "disease=x",
                "tool_prefetch_complete": True,
                "query_prior": {"prior": "kept"},
                "query_tool_summary": {"tool": "kept"},
                "reused_none_final": {"status": "ok"},
                "reused_single_source_index": 17,
            }
        )
    )
    loaded = runner._load_progressive_query_prior(
        "bbb_martins",
        3,
        {"molecule_identity_key": "PARENT-A", "condition_group": "disease=x"},
        tmp_path,
    )
    assert loaded == (
        {"prior": "kept"},
        {"tool": "kept"},
        {"status": "ok"},
        17,
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        runner._load_progressive_query_prior(
            "bbb_martins",
            3,
            {"molecule_identity_key": "PARENT-B", "condition_group": "disease=x"},
            tmp_path,
        )


def test_deepseek_provider_aliases_share_reuse_identity():
    assert runner._model_identity(
        "deepseek-ai/DeepSeek-V4-Flash-0731"
    ) == runner._model_identity("deepseek/deepseek-v4-flash-0731")
    assert runner._model_identity(
        "deepseek/deepseek-v4-flash"
    ) == runner._model_identity("deepseek/deepseek-v4-flash-0731")
    assert runner._model_identity("deepseek-v4-flash") == runner._model_identity(
        "deepseek/deepseek-v4-flash-0731"
    )
    assert runner._model_identity("another/model") != runner._model_identity(
        "deepseek/deepseek-v4-flash-0731"
    )


def test_identity_policy_is_split_specific():
    assert runner._neighbor_identity_policy("scaffold") == "scaffold_disjoint"
    assert runner._neighbor_identity_policy("random") == "parent_disjoint"
    with pytest.raises(ValueError, match="unsupported split scheme"):
        runner._neighbor_identity_policy("unknown")


def test_progressive_reuse_signature_ignores_nonvisible_audits():
    prepared = {
        field: {"field": field}
        for field in runner._PREPARED_MODEL_INPUT_FIELDS
    }
    prepared["retrieval_audit"] = {"n_candidates": 20}
    prepared["selection_audit"] = {"n_selected": 3}
    changed_audits = {
        **prepared,
        "retrieval_audit": {"n_candidates": 100},
        "selection_audit": {"n_selected": 4},
    }
    assert runner._prepared_model_input(prepared) == runner._prepared_model_input(
        changed_audits
    )
    changed_prompt = {**prepared, "new_card_ids": ["different-card"]}
    assert runner._prepared_model_input(prepared) != runner._prepared_model_input(
        changed_prompt
    )
    catalog_count_change = {
        **prepared,
        "level_definition": {
            "level": 2,
            "endpoint_group": "near_direct",
            "description": "Near-direct evidence.",
            "new_physical_assays": 147,
        },
    }
    other_catalog_count = {
        **catalog_count_change,
        "level_definition": {
            **catalog_count_change["level_definition"],
            "new_physical_assays": 145,
        },
    }
    assert runner._prepared_model_input(
        catalog_count_change
    ) == runner._prepared_model_input(other_catalog_count)
    changed_level_description = {
        **catalog_count_change,
        "level_definition": {
            **catalog_count_change["level_definition"],
            "description": "Different prompt text.",
        },
    }
    assert runner._prepared_model_input(
        catalog_count_change
    ) != runner._prepared_model_input(changed_level_description)


def test_progressive_reuse_rejects_changed_generation_contract(tmp_path):
    shared = {
        field: {"field": field}
        for field in (
            "experiment",
            "split_scheme",
            "visibility_mode",
            "reference_pool",
            "neighbor_identity_policy",
            "min_similarity",
            "candidate_generation",
            "selection",
            "prompt_profile",
            "condition_policy",
            "temperature",
            "thinking",
            "reasoning_effort",
            "tool_prefetch_complete",
        )
    }
    task_inputs = {
        "input_sha256": "input",
        "family_manifest_sha256": "family",
        "single_source_manifest_sha256": "single",
    }
    source = {
        **shared,
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "max_tokens": 20_480,
        "tasks": ["bioavailability_ma"],
        "inputs": {"bioavailability_ma": task_inputs},
    }
    (tmp_path / "experiment_manifest.json").write_text(json.dumps(source))
    current = {
        **source,
        "model": "deepseek/deepseek-v4-flash",
        "max_tokens": 4096,
    }
    with pytest.raises(ValueError, match="different max_tokens"):
        runner._validate_progressive_reuse_source(
            source_root=tmp_path,
            current_manifest=current,
        )


def test_progressive_reuse_allows_changed_family_lineage(tmp_path):
    shared = {
        field: {"field": field}
        for field in (
            "experiment",
            "split_scheme",
            "visibility_mode",
            "reference_pool",
            "neighbor_identity_policy",
            "min_similarity",
            "candidate_generation",
            "selection",
            "prompt_profile",
            "condition_policy",
            "temperature",
            "thinking",
            "reasoning_effort",
            "tool_prefetch_complete",
        )
    }
    source = {
        **shared,
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "max_tokens": 20_480,
        "tasks": ["skin_reaction"],
        "inputs": {
            "skin_reaction": {
                "input_sha256": "same-benchmark",
                "family_manifest_sha256": "old-family",
                "single_source_manifest_sha256": "same-single",
            }
        },
    }
    (tmp_path / "experiment_manifest.json").write_text(json.dumps(source))
    current = {
        **source,
        "inputs": {
            "skin_reaction": {
                **source["inputs"]["skin_reaction"],
                "family_manifest_sha256": "new-family",
            }
        },
    }
    assert (
        runner._validate_progressive_reuse_source(
            source_root=tmp_path,
            current_manifest=current,
        )
        == source
    )


def test_source_repair_reuses_only_unchanged_prefix(tmp_path, monkeypatch):
    source, target = tmp_path / "old", tmp_path / "new"
    monkeypatch.setattr(runner, "_levels", lambda task: [{"level": i} for i in (1, 2, 3)])
    for level in (1, 2, 3):
        relative = f"ames/queries/query_idx00000/levels/level_{level}"
        for root in (source, target):
            path = root / relative
            path.mkdir(parents=True)
            prepared = {"level": level, "new_card_ids": ["old-card"]}
            if root == target and level == 2:
                prepared["new_card_ids"] = ["replacement-card"]
            (path / "prepared.json").write_text(json.dumps(prepared))
        (source / relative / "output.json").write_text(json.dumps({
            "status": "ok", "model_called": True,
            "state": {"ames_prediction": "positive"},
        }))
    receipt = runner._reuse_unchanged_progressive_prefixes(
        prepared_queries=[runner.PreparedQuery("ames", 0, target / "ames/queries/query_idx00000")],
        output_root=target, source_root=source,
    )
    assert receipt["n_reused_levels"] == 1
    assert (target / "ames/queries/query_idx00000/levels/level_1/output.json").exists()
    # Even an identical later prepared row depends on the changed prior state.
    assert not (target / "ames/queries/query_idx00000/levels/level_3/output.json").exists()


def test_matched_source_repair_rejects_ambiguous_suite_reuse():
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as family
    with pytest.raises(SystemExit):
        family.main(["--matched-progressive-root", "frozen", "--replicate-ids", "1", "2",
                     "--level-reuse-source-root", "one-old-run"])


def test_random_split_requires_explicit_output_root():
    with pytest.raises(SystemExit):
        runner.main(["--split-scheme", "random", "--prepare-only"])


def test_test_subset_uses_separate_inputs_and_same_heldout_indices():
    for scheme in ("scaffold", "random"):
        valid = runner._progressive_task_specs(scheme)
        test = runner._progressive_task_specs(scheme, "test")
        for task in runner.TASK_NAMES:
            assert test[task].input_jsonl == runner.split_path(task, "test", scheme)
            assert test[task].input_jsonl != valid[task].input_jsonl
            assert test[task].index == valid[task].index
            assert test[task].family_manifest == valid[task].family_manifest
    with pytest.raises(SystemExit):
        runner.main(["--evaluation-subset", "test", "--prepare-only"])


def test_resume_rejects_changed_evaluation_subset():
    shared = {field: field for field in runner._RESUME_INVARIANT_FIELDS}
    previous = {**shared, "model": runner.MODEL, "evaluation_subset": "valid"}
    current = {**previous, "evaluation_subset": "test"}
    with pytest.raises(ValueError, match="evaluation_subset"):
        runner._merge_resume_manifest(previous, current)


def test_explicit_endpoint_budget_required_for_three_256_tasks(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "run", lambda args: args)
    command = ["--parallelism", "768", "--parallelism-per-task", "256",
               "--evaluation-subset", "test", "--output-root", str(tmp_path)]
    with pytest.raises(SystemExit):
        runner.main(command)
    args = runner.main(command + ["--endpoint-concurrency-budget", "768"])
    assert args.parallelism == 768
    assert args.parallelism_per_task == 256
    assert args.evaluation_subset == "test"


def test_matched_full_flat_requires_explicit_768_budget(monkeypatch, tmp_path):
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as family

    monkeypatch.setattr(family, "_run_matched_curve", lambda args: args)
    command = ["--matched-progressive-root", str(tmp_path / "source"),
               "--output-root", str(tmp_path / "result"),
               "--parallelism", "768", "--parallelism-per-task", "256"]
    with pytest.raises(SystemExit):
        family.main(command)
    args = family.main(command + ["--endpoint-concurrency-budget", "768"])
    assert args.parallelism == 768 and args.parallelism_per_task == 256


@pytest.mark.parametrize("index_offset", [0, 100])
def test_shared_query_pool_enforces_caps_without_blocking_ready_tasks(
    monkeypatch, tmp_path, index_offset
):
    import threading
    from types import SimpleNamespace

    tasks = list(runner.TASK_NAMES)
    args = SimpleNamespace(tasks=tasks, parallelism=len(tasks), parallelism_per_task=1)
    active = dict.fromkeys(tasks, 0)
    peaks = dict.fromkeys(tasks, 0)
    lock = threading.Lock()
    ready = threading.Barrier(len(tasks))
    total_peak = 0

    def fake_run(args, prepared, client):
        nonlocal total_peak
        with lock:
            active[prepared.task] += 1
            peaks[prepared.task] = max(peaks[prepared.task], active[prepared.task])
            total_peak = max(total_peak, sum(active.values()))
        if prepared.index == tasks.index(prepared.task) * index_offset:
            ready.wait(timeout=5)
        with lock:
            active[prepared.task] -= 1
        return {"status": "ok", "task": prepared.task, "index": prepared.index}

    monkeypatch.setattr(runner, "_run_query_safe", fake_run)
    queries = [
        runner.PreparedQuery(task, i + task_index * index_offset, tmp_path)
        for task_index, task in enumerate(tasks) for i in range(5)
    ]
    results = list(runner._query_results(args, queries, None))
    assert len({(row["task"], row["index"]) for row in results}) == 5 * len(tasks)
    assert all(peak == 1 for peak in peaks.values())
    assert total_peak == len(tasks)


@pytest.mark.parametrize("permanent_failure", [False, True])
def test_failed_queries_retry_with_backoff_and_bounded_status(monkeypatch, tmp_path, permanent_failure):
    from types import SimpleNamespace

    args = SimpleNamespace(prepare_only=False, max_stage_requeues=2, retry_delay_s=60)
    queries = [runner.PreparedQuery("example", index, tmp_path / str(index)) for index in (0, 1)]
    seen, waits = [], []

    def results(args, pending, client):
        seen.append([query.index for query in pending])
        for query in pending:
            failed = query.index == 1 and (permanent_failure or len(seen) < 3)
            yield {"task": query.task, "index": query.index, "status": "error" if failed else "ok"}

    def sleep(delay):
        status = json.loads((tmp_path / "execution_status.json").read_text())
        assert status["phase"] == "retry_wait" and status["next_retry_at"]
        waits.append(delay)

    monkeypatch.setattr(runner, "_query_results", results)
    monkeypatch.setattr(runner.time, "sleep", sleep)
    assert runner._run_query_rounds(args, queries, None, tmp_path) == int(permanent_failure)
    assert seen == [[0, 1], [1], [1]]
    assert waits == [60, 120]
    status = json.loads((tmp_path / "execution_status.json").read_text())
    assert status["phase"] == ("needs_attention" if permanent_failure else "complete")
    assert status["n_succeeded_queries"] == 2 - int(permanent_failure)


def test_matched_prepared_copy_is_atomic_and_rejects_altered_resume(tmp_path):
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as family

    source, target = tmp_path / "source.json", tmp_path / "result/prepared.json"
    source.write_text('{"frozen": true}\n')
    digest = family.sha256_file(source)
    family._copy_matched_prepared(source, target, digest)
    family._copy_matched_prepared(source, target, digest)
    assert target.read_bytes() == source.read_bytes()
    target.write_text('{"frozen": false}\n')
    with pytest.raises(ValueError, match="changed matched prepared"):
        family._copy_matched_prepared(source, target, digest)
    assert target.read_text() == '{"frozen": false}\n'
    missing_target = tmp_path / "other/prepared.json"
    source.write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="changed while copying"):
        family._copy_matched_prepared(source, missing_target, digest)
    assert not missing_target.exists()
    assert not list(missing_target.parent.iterdir())


def test_matched_suite_runs_sequential_distinct_roots_and_keeps_failure_status(monkeypatch, tmp_path):
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as family

    source = tmp_path / "source"
    source.mkdir()
    (source / "experiment_manifest.json").write_text("{}")
    seen = []

    def run(args):
        seen.append((args.output_root, args.matched_organizations[0], args.retry_race_width))
        return int(len(seen) == 1)

    monkeypatch.setattr(family, "_run_matched_curve", run)
    code = family.main([
        "--matched-progressive-root", str(source), "--output-root", str(tmp_path / "suite"),
        "--tasks", "bbb_martins", "bioavailability_ma", "skin_reaction",
        "--replicate-ids", "2", "3", "--matched-organizations", "progressive", "full_flat",
        "--retry-race-width", "6", "--parallelism", "768",
        "--parallelism-per-task", "256", "--endpoint-concurrency-budget", "768",
    ])
    assert code == 1
    assert len({root for root, _, _ in seen}) == 4
    assert [organization for _, organization, _ in seen] == ["progressive", "full_flat"] * 2
    assert all(width == 6 for _, _, width in seen)
    status = json.loads((tmp_path / "suite/suite_status.json").read_text())
    assert status["phase"] == "needs_attention"
    assert [job["status"] for job in status["jobs"]] == ["needs_attention", "complete", "complete", "complete"]


def test_refresh_prior_preserves_tools_and_cards_and_rejects_tool_drift():
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as family

    frozen = {"query_tool_summary": {"text": "frozen properties"}, "cards": [{"value": 42}],
              "query_prior": "old", "reused_none_final": "old", "reused_single_source_index": 0}
    prior = ({"reasoning": "new model"}, frozen["query_tool_summary"], {"prediction": "pass"}, 3)
    refreshed = family._replace_matched_prior(frozen, prior)
    assert refreshed["cards"] == frozen["cards"]
    assert refreshed["query_tool_summary"] == frozen["query_tool_summary"]
    assert refreshed["query_prior"] == prior[0] and refreshed["reused_none_final"] == prior[2]
    assert frozen["query_prior"] == "old"
    with pytest.raises(ValueError, match="changed frozen query tools"):
        family._replace_matched_prior(frozen, (prior[0], {"text": "changed"}, prior[2], 3))


def test_matched_suite_concurrent_runs_share_total_budget(monkeypatch, tmp_path):
    import threading
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as family
    source = tmp_path / "source"
    source.mkdir()
    (source / "experiment_manifest.json").write_text("{}")
    barrier = threading.Barrier(2)
    seen = []

    def run(args):
        seen.append((threading.get_ident(), args.parallelism, args.endpoint_concurrency_budget))
        barrier.wait(timeout=5)
        status = json.loads((tmp_path / "suite/suite_status.json").read_text())
        assert len(status["active_roots"]) == 2
        barrier.wait(timeout=5)
        return 0

    monkeypatch.setattr(family, "_run_matched_curve", run)
    assert family.main(["--matched-progressive-root", str(source), "--output-root", str(tmp_path / "suite"),
                        "--replicate-ids", "1", "--matched-organizations", "progressive", "full_flat",
                        "--concurrent-runs", "2", "--parallelism", "2", "--endpoint-concurrency-budget", "4"]) == 0
    assert len({row[0] for row in seen}) == 2
    assert all(row[1:] == (2, 2) for row in seen)
    assert json.loads((tmp_path / "suite/suite_status.json").read_text())["phase"] == "complete"


def test_matched_suite_rejects_concurrent_budget_overflow(tmp_path):
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as family
    with pytest.raises(SystemExit):
        family.main(["--matched-progressive-root", str(tmp_path / "source"), "--replicate-ids", "1",
                     "--concurrent-runs", "2", "--parallelism", "256", "--endpoint-concurrency-budget", "256"])


def test_native_matched_prior_refresh_copies_only_tools_and_checks_resume(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as family
    source_root, old, output = tmp_path / "source", tmp_path / "old", tmp_path / "new/single_cache"
    source_root.mkdir()
    original_path = old / "ames/none/manifest.json"
    original_path.parent.mkdir(parents=True)
    original_path.write_text(json.dumps(dict(schema_version="conditioned_query_priors.v1", input_sha256="input", indices=[0], n_items=1)))
    source = dict(single_reuse_root=str(old), tasks=["ames"], inputs={"ames": {
        "single_source_manifest_sha256": family.sha256_file(original_path), "input_sha256": "input"}},
        evaluation_indices_by_task={"ames": [0]})
    (source_root / "experiment_manifest.json").write_text(json.dumps(source))
    old_run = runner._source_run_dir("ames", 0, old)
    old_run.mkdir(parents=True)
    for name in ["retrieval.json", "single_molecule_reasoning_output.json", "final_reasoning_output.json"]:
        (old_run / name).write_text(json.dumps({"frozen": name}))
    monkeypatch.setattr(family, "_validate_matched_source", lambda *a, **kw: None)
    monkeypatch.setattr(family, "_validate_matched_task", lambda *a: "spec")
    monkeypatch.setattr(runner, "_resolve_provider_pool_config", lambda a: SimpleNamespace(providers=[SimpleNamespace(max_inflight=2)]))
    calls = []

    def refresh(args, specs, provider):
        calls.append(args.model)
        assert args.retry_race_width == 6 and args.transport_max_retries == 0
        assert specs == {"ames": "spec"} and args.output_root == str(output.parent)
        copied = runner._source_run_dir("ames", 0, output)
        assert (copied / "retrieval.json").read_bytes() == (old_run / "retrieval.json").read_bytes()
        assert not (copied / "single_molecule_reasoning_output.json").exists()
        assert not (copied / "final_reasoning_output.json").exists()
        return 0

    monkeypatch.setattr(runner, "_prepare_fresh_query_priors", refresh)
    args = SimpleNamespace(tasks=["ames"], model="pro", endpoint_concurrency_budget=2, retry_race_width=6)
    assert family._refresh_matched_priors(args, source_root, output) == 0
    assert family._refresh_matched_priors(args, source_root, output) == 0
    assert calls == ["pro"]
    (runner._source_run_dir("ames", 0, output) / "retrieval.json").write_text("{}")
    with pytest.raises(ValueError, match="changed refreshed prior"):
        family._refresh_matched_priors(args, source_root, output)


def test_prior_callback_uses_shared_pool_and_failure_receipts(tmp_path):
    from types import SimpleNamespace

    args = SimpleNamespace(tasks=["example"], parallelism=2, parallelism_per_task=1, prepare_only=False)
    queries = [runner.PreparedQuery("example", i, tmp_path / str(i)) for i in range(2)]
    def callback(args, query, client):
        if query.index:
            raise TimeoutError("prior timeout")
        return {"task": query.task, "index": query.index, "status": "ok"}
    rows = list(runner._query_results(args, queries, None, run_query=callback))
    assert [row["status"] for row in rows] == ["ok", "error"]
    assert (tmp_path / "1/run_error.json").exists()


def test_failed_prior_refresh_stops_suite_before_levels(monkeypatch, tmp_path):
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as family
    source = tmp_path / "source"
    source.mkdir()
    (source / "experiment_manifest.json").write_text("{}")
    def fail(*args):
        raise ValueError("stale inputs")
    monkeypatch.setattr(family, "_refresh_matched_priors", fail)
    monkeypatch.setattr(family, "_run_matched_curve", lambda args: pytest.fail("must not run levels"))
    assert family.main(["--matched-progressive-root", str(source), "--output-root", str(tmp_path / "suite"),
                        "--replicate-ids", "1", "--refresh-query-priors"]) == 1
    status = json.loads((tmp_path / "suite/suite_status.json").read_text())
    assert status["phase"] == "needs_attention" and status["error"] == "stale inputs"


@pytest.mark.parametrize("relative", ["", "nested"])
def test_matched_suite_rejects_source_overlap_before_writing(tmp_path, relative):
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as family
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "experiment_manifest.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="outside its source"):
        family.main(["--matched-progressive-root", str(source), "--output-root", str(source / relative),
                     "--replicate-ids", "1", "--refresh-query-priors"])
    assert list(source.iterdir()) == [manifest]


def test_prior_refresh_rejects_incompatible_source_before_paid_calls(tmp_path):
    from types import SimpleNamespace
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as family
    source = tmp_path / "source"
    source.mkdir()
    (source / "experiment_manifest.json").write_text(json.dumps({"experiment": "wrong-protocol"}))
    with pytest.raises(ValueError, match="source experiment"):
        family._refresh_matched_priors(SimpleNamespace(max_tokens=20480), source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_transport_failure_history_survives_successful_resume(monkeypatch, tmp_path):
    from types import SimpleNamespace

    query = runner.PreparedQuery("example", 1, tmp_path)
    args = SimpleNamespace(prepare_only=False)
    def fail(*args):
        raise TimeoutError("provider timed out")
    monkeypatch.setattr(runner, "_run_query", fail)
    assert runner._run_query_safe(args, query, None)["status"] == "error"
    monkeypatch.setattr(runner, "_run_query", lambda *args: {"status": "ok"})
    assert runner._run_query_safe(args, query, None)["status"] == "ok"
    archived = list((tmp_path / "failed_attempts").glob("*.json"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text())["error_type"] == "TimeoutError"
    assert json.loads((tmp_path / "run_error.json").read_text())["status"] == "resolved"


def test_resume_allows_equivalent_model_provider_fallback():
    shared = {field: field for field in runner._RESUME_INVARIANT_FIELDS}
    previous = {
        **shared,
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "base_url": "http://127.0.0.1:50001/v1",
        "transport_max_retries": 0,
        "started_at": "initial",
    }
    current = {
        **shared,
        "model": "deepseek/deepseek-v4-flash-0731",
        "base_url": "https://openrouter.ai/api/v1",
        "transport_max_retries": 2,
        "started_at": "resume",
    }
    merged = runner._merge_resume_manifest(previous, current)
    assert merged["model"] == previous["model"]
    assert merged["base_url"] == previous["base_url"]
    assert merged["started_at"] == "initial"
    assert len(merged["execution_providers"]) == 2
    assert merged["execution_providers"][1]["base_url"] == current["base_url"]


def test_resume_rejects_different_model_identity():
    shared = {field: field for field in runner._RESUME_INVARIANT_FIELDS}
    previous = {**shared, "model": "model/a"}
    current = {**shared, "model": "model/b"}
    with pytest.raises(ValueError, match="model identity"):
        runner._merge_resume_manifest(previous, current)


def test_prediction_summary_supports_matched_none_reuse(tmp_path):
    predictions = [
        {"label": 0, "pred_label": 0, "correct": True, "model_called": False},
        {"label": 1, "pred_label": 1, "correct": True, "model_called": False},
    ]
    runner._write_prediction_summary(
        task="bbb_martins",
        predictions=predictions,
        summary_dir=tmp_path,
        metric_fields={"level": 0, "family": "none"},
    )
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["n_total"] == 2
    assert metrics["n_failed_runs"] == 0
    assert metrics["macro_f1"] == 1.0
    assert metrics["n_model_called"] == 0


def test_extract_collapses_exact_cards_but_preserves_late_family_cards():
    neighbor = _neighbor(
        "CCO",
        0.7,
        [
            _row("hidden-assay-1", "direct", 1, "complete direct support"),
            _row("hidden-assay-2", "direct", 1, "complete direct support"),
            _row("hidden-assay-1", "mechanism", 2, "complete mechanism support"),
        ],
    )
    analogs = extract_cumulative_evidence(_retrieval([neighbor]))
    analog = next(iter(analogs.values()))
    assert len(analog["cards"]) == 2
    assert {card["evidence_family"] for card in analog["cards"].values()} == {
        "direct",
        "mechanism",
    }
    assert {card["support_text"] for card in analog["cards"].values()} == {
        "complete direct support",
        "complete mechanism support",
    }


def test_later_level_has_independent_new_and_augmentation_quotas():
    previous = extract_cumulative_evidence(
        _retrieval([_neighbor("ACTIVE", 0.6, [_row("a1", "direct", 1, "direct")])])
    )
    active, _ = select_initial_evidence(previous, molecule_limit=1, card_limit=4)
    current_neighbors = [
        _neighbor(
            "ACTIVE",
            0.6,
            [
                _row("a1", "direct", 1, "direct"),
                _row("a2", "mechanism", 2, "augmentation one"),
                _row("a3", "mechanism", 2, "augmentation two"),
                _row("a4", "mechanism", 2, "augmentation three"),
            ],
        )
    ]
    current_neighbors.extend(
        _neighbor(f"NEW-{index}", 0.55 - index / 100, [_row(f"n{index}", "mechanism", 2, f"new {index}")])
        for index in range(7)
    )
    current = extract_cumulative_evidence(_retrieval(current_neighbors))
    new, augmentations, audit = select_progressive_delta(
        previous,
        current,
        active,
        level=2,
    )
    assert len(new) == 3
    assert len(augmentations) == 1
    assert len(next(iter(augmentations.values()))["cards"]) == 2
    assert audit["slot_borrowing"] is False
    updated = append_evidence(active, new, augmentations)
    assert len(updated) == len(active) + 3
    assert card_ids(active) < card_ids(updated)


def test_doubled_card_budget_selects_eight_initial_and_four_delta_cards():
    previous = extract_cumulative_evidence(
        _retrieval(
            [
                _neighbor(
                    "ACTIVE",
                    0.6,
                    [
                        _row(f"direct-{index}", "direct", 1, f"direct {index}")
                        for index in range(10)
                    ],
                )
            ]
        )
    )
    active, initial_audit = select_initial_evidence(
        previous,
        molecule_limit=1,
        card_limit=8,
    )
    assert len(next(iter(active.values()))["cards"]) == 8
    assert initial_audit["n_selected_cards"] == 8

    current = extract_cumulative_evidence(
        _retrieval(
            [
                _neighbor(
                    "ACTIVE",
                    0.6,
                    [
                        *[
                            _row(f"direct-{index}", "direct", 1, f"direct {index}")
                            for index in range(10)
                        ],
                        *[
                            _row(f"later-{index}", "mechanism", 2, f"later {index}")
                            for index in range(6)
                        ],
                    ],
                )
            ]
        )
    )
    _, augmentations, delta_audit = select_progressive_delta(
        previous,
        current,
        active,
        level=2,
        card_limit=4,
    )
    assert len(next(iter(augmentations.values()))["cards"]) == 4
    assert delta_audit["n_selected_cards"] == 4


def test_prompt_is_visible_append_only_and_hides_internal_source_ids():
    cumulative = extract_cumulative_evidence(
        _retrieval([_neighbor("CCO", 0.7, [_row("secret-assay", "direct", 1, "full support text")])])
    )
    active, _ = select_initial_evidence(cumulative, molecule_limit=1, card_limit=4)
    analog = next(iter(active.values()))
    analog["query_analog_tool_summaries"] = [
        {"tool_name": "properties_compare", "status": "ok", "content": "one tool summary"}
    ]
    messages = build_progressive_messages(
        contract=_contract(),
        levels=[{"level": 1, "endpoint_group": "direct", "description": "direct outcome"}],
        current_level=1,
        query_smiles="QUERY",
        condition_sentence="This prediction concerns the query molecule under a fasted state.",
        query_prior={"reasoning_summary": "prior"},
        query_tool_summary={"tool_name": "molecule_properties", "status": "ok", "content": "query properties"},
        active=active,
        prior_state=None,
    )
    prompt = json.loads(messages[1]["content"])
    serialized = messages[1]["content"]
    assert prompt["query"]["canonical_smiles"] == "QUERY"
    assert "external_condition" in prompt["query"]
    assert prompt["active_evidence"][0]["canonical_smiles"] == "CCO"
    assert prompt["active_evidence"][0]["evidence_cards"][0]["support_text"] == "full support text"
    assert prompt["active_evidence"][0]["evidence_cards"][0]["new_this_level"] is True
    assert "not_used_card_ids" not in prompt["required_json_schema"]
    assert serialized.count("one tool summary") == 1
    assert "secret-assay" not in serialized
    assert "INTERNAL_SOURCE_ID" not in serialized
    assert "Use general medicinal-chemistry knowledge" in messages[0]["content"]
    assert "Ground every compound-specific empirical claim" in messages[0]["content"]
    assert "identity_and_selection" not in prompt["protocol"]


def test_progressive_preparation_does_not_freeze_false_none_carry_forward(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(runner, "_levels", lambda task: [{"level": 1}])
    monkeypatch.setattr(runner, "_task_contract", lambda task: _contract())
    level = tmp_path / "levels/level_1"
    level.mkdir(parents=True)
    (level / "prepared.json").write_text(json.dumps({"tool_prefetch_complete": True,
                                                    "should_call_model": False}))
    runner._run_query(SimpleNamespace(prepare_only=True), runner.PreparedQuery("example", 0, tmp_path), None)
    assert not (level / "output.json").exists()


def test_independent_levels_ignore_completed_prior_and_reassess_unchanged_cards(monkeypatch, tmp_path):
    from types import SimpleNamespace

    cumulative = extract_cumulative_evidence(
        _retrieval([_neighbor("CCO", 0.7, [_row("assay", "direct", 1, "measured support")])])
    )
    active, _ = select_initial_evidence(cumulative, molecule_limit=1, card_limit=4)
    levels = [{"level": 1, "endpoint_group": "direct"}, {"level": 2, "endpoint_group": "mechanism"}]
    monkeypatch.setattr(runner, "_levels", lambda task: levels)
    monkeypatch.setattr(runner, "_task_contract", lambda task: _contract())
    first = tmp_path / "levels/level_1"
    first.mkdir(parents=True)
    (first / "output.json").write_text(json.dumps({
        "status": "ok", "state": {"decision_summary": "MUST_NOT_LEAK"}, "model_called": True,
    }))
    second = tmp_path / "levels/level_2"
    second.mkdir(parents=True)
    prepared = {"active_evidence": active, "tool_prefetch_complete": True,
                "should_call_model": False, "query_smiles": "CCN", "condition_sentence": "",
                "query_prior": {"reasoning_summary": "property prior"}, "query_tool_summary": {}}
    (second / "prepared.json").write_text(json.dumps(prepared))
    args = SimpleNamespace(independent_levels=True, prepare_only=True)
    runner._run_query(args, runner.PreparedQuery("example", 0, tmp_path), None)
    request = json.loads((second / "request.json").read_text())
    payload = json.loads(request["messages"][1]["content"])
    assert "MUST_NOT_LEAK" not in json.dumps(request)
    assert "prior_state" not in payload
    assert "new_card_ids" not in payload["level_context"]
    assert not {"flip_rule", "update_rule"} & payload["protocol"].keys()
    assert payload["query_prior"] == prepared["query_prior"]
    cards = [c for a in payload["active_evidence"] for c in a["evidence_cards"]]
    assert len(cards) == 1 and cards[0]["support_text"] == "measured support"
    assert not {"new_this_level", "first_seen_level", "prior_use"} & cards[0].keys()
    assert not (second / "output.json").exists()
    with pytest.raises(ValueError, match="cannot consume prior state"):
        build_progressive_messages(contract=_contract(), levels=levels, current_level=2,
                                   query_smiles="CCN", condition_sentence="", query_prior={},
                                   query_tool_summary={}, active=active, prior_state={"x": 1}, independent=True)
    args.prepare_only = False
    (second / "output.json").write_text(json.dumps({"status": "error", "llm": {"content": "failed trace"}}))
    client = SimpleNamespace(chat_json=lambda messages: {"content": {
        "example_prediction": "positive", "confidence": "low", "revision_action": "initial",
        "supportive_card_ids": ["C01"], "contradictory_card_ids": [],
        "prediction_basis_card_ids": ["C01"], "claims": [], "evidence_gaps": [],
        "decision_summary": "Independent decision using the available card.",
        "new_evidence_assessment": [{"card_ids": ["C01"], "applicability": "low",
                                     "direction": "supportive", "decision_effect": "no_change"}],
    }})
    result = runner._run_query(args, runner.PreparedQuery("example", 0, tmp_path), client)
    assert result["status"] == "ok"
    output = json.loads((second / "output.json").read_text())
    assert output["llm"]["structured_output_validation"]["valid"] is True
    assert output["state"]["revision_action"] == "initial"
    archived = list((second / "failed_attempts").glob("*.json"))
    assert len(archived) == 1 and json.loads(archived[0].read_text())["llm"]["content"] == "failed trace"
    client.chat_json = lambda messages: pytest.fail("successful levels must not call the model again")
    assert runner._run_query(args, runner.PreparedQuery("example", 0, tmp_path), client)["status"] == "ok"


def test_prompt_uses_stable_short_aliases_and_compact_prior_state():
    previous = extract_cumulative_evidence(
        _retrieval(
            [
                _neighbor(
                    "CCO",
                    0.7,
                    [
                        _row("direct-1", "direct", 1, "direct support one"),
                        _row("direct-2", "direct", 1, "direct support two"),
                    ],
                )
            ]
        )
    )
    active, _ = select_initial_evidence(previous, molecule_limit=1, card_limit=4)
    initial_id_to_alias, _ = card_alias_maps(active)
    old_ids = sorted(card_ids(active))
    current = extract_cumulative_evidence(
        _retrieval(
            [
                _neighbor(
                    "CCO",
                    0.7,
                    [
                        _row("direct-1", "direct", 1, "direct support one"),
                        _row("direct-2", "direct", 1, "direct support two"),
                        _row("mechanism-1", "mechanism", 2, "new mechanism support"),
                    ],
                )
            ]
        )
    )
    new, augmentations, _ = select_progressive_delta(
        previous,
        current,
        active,
        level=2,
    )
    active = append_evidence(active, new, augmentations)
    id_to_alias, alias_to_id = card_alias_maps(active)
    assert {card_id: id_to_alias[card_id] for card_id in old_ids} == initial_id_to_alias

    used_id, unused_id = old_ids
    prior_state = {
        "level": 1,
        "example_prediction": "positive",
        "confidence": "low",
        "revision_action": "initial",
        "supportive_card_ids": [used_id],
        "contradictory_card_ids": [],
        "prediction_basis_card_ids": [used_id],
        "not_used_card_ids": [unused_id],
        "claims": [{"claim": "old claim", "card_ids": [used_id]}],
        "new_evidence_assessment": [],
        "evidence_gaps": [],
        "decision_summary": "old decision",
    }
    messages = build_progressive_messages(
        contract=_contract(),
        levels=[
            {"level": 1, "endpoint_group": "direct", "description": "direct"},
            {"level": 2, "endpoint_group": "mechanism", "description": "mechanism"},
        ],
        current_level=2,
        query_smiles="QUERY",
        condition_sentence="",
        query_prior={"reasoning_summary": "prior"},
        query_tool_summary={},
        active=active,
        prior_state=prior_state,
    )
    prompt = json.loads(messages[1]["content"])
    serialized = messages[1]["content"]
    cards = [
        card
        for analog in prompt["active_evidence"]
        for card in analog["evidence_cards"]
    ]
    cards_by_alias = {card["card_id"]: card for card in cards}

    assert "all_visible_card_ids" not in prompt["level_context"]
    assert set(prompt["level_context"]["new_card_ids"]) == {
        id_to_alias[next(iter(card_ids(active) - set(old_ids)))]
    }
    assert all(card["card_id"].startswith("C") for card in cards)
    assert not any(card_id in serialized for card_id in card_ids(active))
    assert cards_by_alias[id_to_alias[used_id]]["prior_use"] == [
        "supportive",
        "prediction_basis",
    ]
    assert "prior_use" not in cards_by_alias[id_to_alias[unused_id]]
    for field in (
        "supportive_card_ids",
        "contradictory_card_ids",
        "prediction_basis_card_ids",
        "not_used_card_ids",
    ):
        assert field not in prompt["prior_state"]
    assert prompt["prior_state"]["claims"][0]["card_ids"] == [id_to_alias[used_id]]

    restored = restore_card_ids(
        {
            "supportive_card_ids": [id_to_alias[used_id]],
            "claims": [{"claim": "mapped", "card_ids": [id_to_alias[used_id]]}],
        },
        alias_to_card_id=alias_to_id,
    )
    assert restored["supportive_card_ids"] == [used_id]
    assert restored["claims"][0]["card_ids"] == [used_id]


def test_state_validator_requires_sparse_valid_citations_and_new_card_for_flip():
    contract = _contract()
    prior = {"example_prediction": "negative"}
    valid = {
        "example_prediction": "positive",
        "confidence": "moderate",
        "revision_action": "flip",
        "supportive_card_ids": ["new"],
        "contradictory_card_ids": ["old"],
        "prediction_basis_card_ids": ["new"],
        "claims": [{"claim": "new evidence supports", "card_ids": ["new"]}],
        "new_evidence_assessment": [
            {
                "family": "mechanism",
                "applicability": "high",
                "direction": "supportive",
                "decision_effect": "changed",
                "card_ids": ["new"],
            }
        ],
        "evidence_gaps": [],
        "decision_summary": "changed",
    }
    assert not progressive_state_errors(
        valid,
        contract=contract,
        visible_card_ids={"old", "new"},
        new_card_ids={"new"},
        prior_state=prior,
    )
    invalid = {**valid, "prediction_basis_card_ids": ["old"]}
    assert "a flip must cite at least one newly added card in prediction basis" in progressive_state_errors(
        invalid,
        contract=contract,
        visible_card_ids={"old", "new"},
        new_card_ids={"new"},
        prior_state=prior,
    )
    state = state_from_content(
        valid,
        contract=contract,
        level=2,
        visible_card_ids={"old", "new", "unused"},
    )
    assert state["not_used_card_ids"] == ["unused"]


def test_visible_task_contracts_do_not_retain_identity_blind_wording():
    for contract in (
        bbb_config.get_progressive_task_contract(),
        bio_config.get_progressive_task_contract(),
        skin_config.get_progressive_task_contract(),
    ):
        instructions = " ".join(contract.task_instructions).lower()
        assert "anonymous query" not in instructions
        assert "ignore that recognition" not in instructions
        assert "external measurements" not in instructions


def test_progressive_prompts_use_positive_grounding_without_identity_priming():
    forbidden = (
        "external facts",
        "remembered facts",
        "remembered external",
        "recognized compound",
        "do not invent external",
        "ignore that recognition",
    )
    for contract in (
        bbb_config.get_progressive_task_contract(),
        bio_config.get_progressive_task_contract(),
        skin_config.get_progressive_task_contract(),
    ):
        messages = build_progressive_messages(
            contract=contract,
            levels=[{"level": 1, "endpoint_group": "direct", "description": "direct outcome"}],
            current_level=1,
            query_smiles="QUERY",
            condition_sentence=None,
            query_prior={"reasoning_summary": "prior"},
            query_tool_summary=None,
            active={},
            prior_state=None,
        )
        serialized = " ".join(message["content"] for message in messages).lower()
        assert "ground every compound-specific empirical claim" in serialized
        assert (
            "do not identify the query by name even if its structure is recognizable"
            in serialized
        )
        for phrase in forbidden:
            assert phrase not in serialized


def test_progressive_client_can_disable_duplicate_transport_retries(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(client_module, "OpenAI", FakeOpenAI)
    OpenAICompatibleClient(
        api_key="key",
        base_url="http://127.0.0.1:50001/v1",
        model="model",
        timeout_s=900,
        max_tokens=100,
        temperature=0.0,
        tool_service_url="http://127.0.0.1:8765",
        enable_group_tools=False,
        max_tool_rounds=0,
        reasoning_effort="",
        enable_thinking=False,
        transport_max_retries=0,
    )
    assert captured["max_retries"] == 0


def test_analog_tool_prefetch_retries_then_preserves_unavailable_receipt(monkeypatch):
    class FakeToolClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def invoke_many(self, calls):
            return [
                {
                    "tool_name": tool_name,
                    "status": "error" if tool_name == "mmp_structure_compare" else "ok",
                    "content": "initial result",
                    "errors": [{"message": "molecule-specific failure"}],
                }
                for tool_name, _ in calls
            ]

        def invoke(self, tool_name, _arguments):
            return {
                "tool_name": tool_name,
                "status": "error",
                "content": "raw internal failure",
                "errors": [{"message": "molecule-specific failure"}],
            }

    monkeypatch.setattr(runner, "ToolServiceClient", FakeToolClient)
    summaries, failures = runner._prefetch_analog_tools(
        query_smiles="CCN",
        analogs={
            "analog_one": {
                "analog_id": "analog_one",
                "canonical_smiles": "CCO",
            }
        },
        tool_service_url="http://127.0.0.1:8765",
        timeout_s=10,
    )

    assert len(failures) == 1
    assert failures[0]["analog_id"] == "analog_one"
    assert failures[0]["errors"][0]["message"] == "molecule-specific failure"
    prompt_result = summaries["analog_one"][0]
    assert prompt_result["status"] == "error"
    assert "Comparison unavailable" in prompt_result["content"]
    assert "molecule-specific failure" not in prompt_result["content"]


def test_progressive_runner_isolates_purity_indices_from_historical_top20_runner():
    assert not hasattr(runner, "TOP20_TASKS")
    for task in ("bbb_martins", "bioavailability_ma", "skin_reaction"):
        spec = runner.PROGRESSIVE_TASKS[task]
        version = {
            "bbb_martins": "source_purity_v6",
            "bioavailability_ma": "legacy_record_supported_v2_vote_pure_v1",
            "skin_reaction": "source_purity_v5",
        }[task]
        assert version in str(spec.index)
        assert version in str(spec.family_manifest)
