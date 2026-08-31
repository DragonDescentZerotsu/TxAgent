import json

import pytest

import tools.chembl_tool.common.openai_reasoning_client as client_module
import tools.chembl_tool.paper_experiments.run_conditioned_assay_progressive_curve as runner
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
)
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


def test_random_split_requires_explicit_output_root():
    with pytest.raises(SystemExit):
        runner.main(["--split-scheme", "random", "--prepare-only"])


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
            "bbb_martins": "source_purity_v5",
            "bioavailability_ma": "legacy_record_supported_v2_vote_pure_v1",
            "skin_reaction": "source_purity_v1",
        }[task]
        assert version in str(spec.index)
        assert version in str(spec.family_manifest)
