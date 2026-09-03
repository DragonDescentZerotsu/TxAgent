import json

import pytest

import predict.llm_engine.client as client_module
import predict.harnesses.progressive.runner as runner
import predict.harnesses.progressive.state as progressive_state
from predict.llm_engine.client import OpenAICompatibleClient
from predict.harnesses.progressive.state import (
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


def test_single_reuse_accepts_current_and_archived_run_directory_names(tmp_path):
    batch = tmp_path / "bbb_martins" / "bbb_martins__none"
    batch.mkdir(parents=True)
    current = batch / "runs" / "bbb_martins__none_idx00001"
    current.mkdir(parents=True)
    assert runner._source_run_dir("bbb_martins", 1, tmp_path) == current

    archived = batch / "runs" / "none_idx00002"
    archived.mkdir()
    assert runner._source_run_dir("bbb_martins", 2, tmp_path) == archived


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
    assert list(prompt) == [
        "protocol",
        "task_definition",
        "level_context",
        "query",
        "query_prior",
        "active_evidence",
        "required_json_schema",
    ]
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


def test_molecule_card_contract_controls_prompt_json(monkeypatch):
    contract = {
        "molecule": {
            "fields": [
                {"name": "analog_id", "source": "analog_id", "required": True},
                {"name": "canonical_smiles", "source": "canonical_smiles", "required": True},
                {"name": "score", "source": "similarity"},
                {"name": "first_seen_level", "source": "first_seen_level", "required": True},
            ],
            "tool_summaries_field": "tools",
            "evidence_cards_field": "records",
        },
        "evidence_card": {
            "fields": [
                {"name": "card_id", "source": "card_id", "required": True},
                {"name": "measurement", "source": "reported_value"},
                {"name": "first_seen_level", "source": "first_seen_level", "required": True},
                {"name": "new_this_level", "source": "new_this_level", "required": True},
            ]
        },
    }
    monkeypatch.setattr(progressive_state, "molecule_card_contract", lambda: contract)
    cumulative = extract_cumulative_evidence(
        _retrieval([_neighbor("CCO", 0.7, [_row("assay", "direct", 1, "support", value="42")])])
    )
    active, _ = select_initial_evidence(cumulative, molecule_limit=1, card_limit=1)
    rendered = progressive_state.render_active_evidence(
        active,
        current_level=1,
        prior_state=None,
        card_id_to_alias=card_alias_maps(active)[0],
    )
    assert rendered == [
        {
            "analog_id": next(iter(active)),
            "canonical_smiles": "CCO",
            "score": 0.7,
            "first_seen_level": 1,
            "records": [
                {
                    "card_id": "C01",
                    "measurement": "42",
                    "first_seen_level": 1,
                    "new_this_level": True,
                }
            ],
        }
    ]


def test_gold_l1_rank_controls_selection_and_standalone_prompt_omits_prior():
    cumulative = extract_cumulative_evidence(
        _retrieval(
            [
                _neighbor("HIGH_SIMILARITY", 0.9, [_row("a", "direct", 1, "high")]),
                _neighbor("V9_FIRST", 0.4, [_row("b", "direct", 1, "first")]),
            ]
        )
    )
    by_smiles = {row["canonical_smiles"]: row for row in cumulative.values()}
    by_smiles["HIGH_SIMILARITY"]["_selection_rank"] = 1
    by_smiles["V9_FIRST"]["_selection_rank"] = 0
    card = next(iter(by_smiles["V9_FIRST"]["cards"].values()))
    card["_selection_rank"] = 0
    card["transfer_likelihood"] = 0.87

    active, _ = select_initial_evidence(cumulative, molecule_limit=1, card_limit=4)
    selected = next(iter(active.values()))
    assert selected["canonical_smiles"] == "V9_FIRST"
    assert next(iter(selected["cards"].values()))["transfer_likelihood"] == 0.87

    messages = build_progressive_messages(
        contract=_contract(),
        levels=[{"level": 1, "endpoint_group": "direct", "description": "direct"}],
        current_level=1,
        query_smiles="QUERY",
        condition_sentence="",
        query_prior=None,
        query_tool_summary=None,
        active=active,
        prior_state=None,
    )
    prompt = json.loads(messages[1]["content"])
    assert "query_prior" not in prompt
    assert prompt["active_evidence"][0]["evidence_cards"][0]["transfer_likelihood"] == 0.87
    assert "neither the query label probability" in prompt["protocol"]["transfer_likelihood_rule"]


def test_gold_l1_appends_residual_before_mechanism_cards(monkeypatch, tmp_path):
    normalized_level_2 = {
        "status": "ok",
        **_retrieval([
            _neighbor(
                "LATER",
                0.7,
                [_row("residual", "direct_residual", 2, "residual")],
            )
        ]),
    }
    normalized_level_3 = {
        "status": "ok",
        **_retrieval([
            _neighbor(
                "LATER",
                0.7,
                [
                    _row("residual", "direct_residual", 2, "residual"),
                    _row("passive", "passive_permeability", 3, "permeability"),
                ],
            )
        ]),
    }
    monkeypatch.setattr(
        runner,
        "retrieve_family_molecule_prefixes",
        lambda *args, **kwargs: {
            1: {"status": "ok", **_retrieval([])},
            2: normalized_level_2,
            3: normalized_level_3,
        },
    )
    gold = {
        "gold": {
            "analog_id": "gold",
            "canonical_smiles": "GOLD",
            "similarity": 0.8,
            "_selection_rank": 0,
            "cards": {
                "gold_card": {
                    "card_id": "gold_card",
                    "evidence_family": "direct_brain_exposure",
                    "reported_value": "positive",
                    "_selection_rank": 0,
                }
            },
        }
    }
    prepared = runner._prepare_query(
        task="bbb_martins",
        query_index=0,
        record={"drug": "QUERY", "benchmark_row_id": "Q"},
        index={},
        levels=[
            {"level": 1, "endpoint_group": "direct_brain_exposure"},
            {"level": 2, "endpoint_group": "direct_residual"},
            {"level": 3, "endpoint_group": "passive_permeability"},
        ],
        gold_l1_candidates=gold,
        output_root=tmp_path,
        single_root=tmp_path,
        query_prior_mode="none",
        l1_source="gold_train",
        l1_ranking="morgan",
        tool_service_url="",
        timeout_s=1,
        prefetch_tools=False,
    )
    level_dir = prepared.query_dir / "levels"
    level_1 = json.loads((level_dir / "level_1" / "prepared.json").read_text())
    level_2 = json.loads((level_dir / "level_2" / "prepared.json").read_text())
    level_3 = json.loads((level_dir / "level_3" / "prepared.json").read_text())
    level_1_ids = set(level_1["new_card_ids"])
    level_2_cards = {
        card_id: card
        for analog in level_2["active_evidence"].values()
        for card_id, card in analog["cards"].items()
    }
    assert level_1_ids <= set(level_2_cards)
    assert {
        level_2_cards[card_id]["evidence_family"]
        for card_id in level_2["new_card_ids"]
    } == {"direct_residual"}
    level_3_cards = {
        card_id: card
        for analog in level_3["active_evidence"].values()
        for card_id, card in analog["cards"].items()
    }
    assert {
        level_3_cards[card_id]["evidence_family"]
        for card_id in level_3["new_card_ids"]
    } == {"passive_permeability"}


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
