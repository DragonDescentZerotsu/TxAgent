from __future__ import annotations

import json
from types import SimpleNamespace

from predict.harnesses.progressive.context_records import profile
from predict.harnesses.progressive.runner import _indirect_record_limits
from predict.harnesses.progressive.state import ProgressiveTaskContract, card_ids
from predict.retrieval.assay_reranking.progressive_levels import (
    LUNA_RELEVANCE_CACHE_PROFILE,
)
from predict.retrieval.policies import seeded_rank_tie_key


def _contexts() -> list[dict]:
    return [
        {
            "context_card_id": "context_a",
            "canonical_smiles": "CCO",
            "condition_group": "no_reported_external_condition",
            "transfer_likelihood": 0.91,
            "available_l1": 2,
            "available_l2": 1,
            "_selection_rank": 0,
            "_gold_record_id": "gold_a",
            "l1_cards": [
                {
                    "card_id": "record_1",
                    "first_seen_level": 1,
                    "level": "L1",
                    "endpoint": "outcome",
                    "result": "positive",
                    "support_text": "Observed outcome.",
                    "source__endpoint_name": "raw outcome",
                    "source__measurement_text": "observed positive",
                    "source__unit_text": "source unit",
                    "source__assay_model": "in vivo exposure study",
                    "source__extra_details": "matched systemic reference",
                    "source__support_text": "Original source sentence.",
                    "canonical_endpoint_name": "canonical outcome",
                    "canonical_measurement_text": "canonical positive",
                    "canonical_unit_text": "canonical unit",
                    "canonical_assay_context": "canonical in vivo study",
                    "canonical_species_context": "rat",
                    "_canonical_record_id": "row_1",
                },
                {
                    "card_id": "record_2",
                    "first_seen_level": 1,
                    "level": "L1",
                    "endpoint": "outcome",
                    "result": "positive",
                    "_canonical_record_id": "row_2",
                },
            ],
            "l2_cards": [
                {
                    "card_id": "record_3",
                    "first_seen_level": 2,
                    "level": "L2",
                    "endpoint": "supporting measurement",
                    "result": "not explicitly reported",
                    "_canonical_record_id": "row_3",
                }
            ],
        }
    ]


def test_luna_bbb_uses_safe_default_final_record_limit():
    args = SimpleNamespace(
        assay_transfer_cache_profile=LUNA_RELEVANCE_CACHE_PROFILE,
        v21_selection_mode="record_only",
        indirect_record_limit_per_level=50,
        indirect_final_level_record_limit=0,
        record_limit_per_context_level=10,
        l2_record_limit_per_context=10,
    )
    assert _indirect_record_limits(args, "bbb_martins") == {
        "L3": 50,
        "L4": 50,
        "L5": 25,
    }
    args.indirect_final_level_record_limit = 17
    assert _indirect_record_limits(args, "bbb_martins")["L5"] == 17


def _morgan_contexts() -> list[dict]:
    contexts = _contexts()
    contexts[0]["morgan_similarity"] = 0.73
    del contexts[0]["transfer_likelihood"]
    return contexts


def _contract(task: str = "example") -> ProgressiveTaskContract:
    return ProgressiveTaskContract(
        task=task,
        endpoint_name="example endpoint",
        label_scope="example scope",
        prediction_field="prediction",
        positive_prediction="positive",
        negative_prediction="negative",
        system_role="Reason about the example endpoint.",
        task_instructions=("Use the supplied records.",),
    )


def _indirect_records(
    levels=("L3", "L4", "L5", "L6"),
    *,
    record_count: int = 50,
) -> dict[str, dict]:
    return {
        level: {
            "available_record_count": 75,
            "records": [
                {
                    "record_id": f"{level}_row_{index:02d}",
                    "reference_molecule_id": f"parent_{index:02d}",
                    "transfer_likelihood": 0.999 - index / 100,
                    "payload": {
                        "record_id": f"{level}_row_{index:02d}",
                        "progressive_level": level,
                        "canonical_smiles": "CCO",
                        "source_id": "source",
                        "measurement_kind": "numeric",
                        "source_fields": {
                            "measurement_text": "1.234",
                            "numeric_metadata": 1.234,
                            "missing": None,
                        },
                    },
                }
                for index in range(record_count)
            ],
        }
        for level in levels
    }


def test_context_record_snapshots_are_append_only() -> None:
    snapshots = profile.snapshots(_contexts())

    assert card_ids(snapshots[1]) == {"record_1", "record_2"}
    assert card_ids(snapshots[2]) == {"record_1", "record_2", "record_3"}
    assert snapshots[1]["context_a"]["available_record_counts"] == {"L1": 2}
    assert snapshots[2]["context_a"]["available_record_counts"] == {
        "L1": 2,
        "L2": 1,
    }


def test_bioavailability_l3_through_l6_append_one_top50_record_bundle_each() -> None:
    snapshots = profile.snapshots(
        _contexts(),
        task="bioavailability_ma",
        indirect_records=_indirect_records(),
    )

    assert list(snapshots) == [1, 2, 3, 4, 5, 6]
    assert card_ids(snapshots[3]) - card_ids(snapshots[2]) == {
        *snapshots[3]["bioavailability_ma_l3_record_bundle"]["cards"]
    }
    assert len(card_ids(snapshots[3]) - card_ids(snapshots[2])) == 50
    assert len(card_ids(snapshots[4]) - card_ids(snapshots[3])) == 50
    assert len(card_ids(snapshots[5]) - card_ids(snapshots[4])) == 50
    assert len(card_ids(snapshots[6]) - card_ids(snapshots[5])) == 50
    assert profile.indirect_level_names("bioavailability_ma") == (
        "L3",
        "L4",
        "L5",
        "L6",
    )

    aliases, _ = profile.card_alias_maps(snapshots[3])
    rendered = profile.render_active_evidence(
        snapshots[3], card_id_to_alias=aliases, prompt_version="v3", compact=True
    )
    bundle = rendered[-1]
    assert bundle["level"] == "L3"
    assert bundle["selected_record_count"] == 50
    assert bundle["record_columns"] == [
        "card_id",
        "reference_molecule_id",
        "transfer_likelihood",
        "source_schema_id",
        "measurement_kind",
        "source_values",
    ]
    assert len(bundle["record_rows"]) == 50
    assert bundle["reference_molecules"] == [{"molecule_id": "M01", "smiles": "CCO"}]
    assert bundle["source_schemas"] == [
        {
            "schema_id": "S01",
            "source_id": "source",
            "source_columns": ["measurement_text", "numeric_metadata"],
        }
    ]
    assert bundle["record_rows"][0][2] == 1.0
    assert bundle["record_rows"][0][-1] == ["1.234", 1.23]


def test_bbb_v21_raw_record_bundles_are_append_only_from_l1() -> None:
    level_names = ("L1", "L2", "L3", "L4", "L5")
    snapshots = profile.snapshots(
        [],
        task="bbb_martins",
        indirect_records=_indirect_records(level_names, record_count=2),
        indirect_record_limit=2,
        record_levels=level_names,
        transfer_model="V21",
    )

    assert list(snapshots) == [1, 2, 3, 4, 5]
    assert [len(card_ids(snapshots[level])) for level in range(1, 6)] == [
        2,
        4,
        6,
        8,
        10,
    ]
    assert snapshots[1]["bbb_martins_l1_record_bundle"]["selection"].endswith(
        "frozen V21 record-transfer likelihood"
    )
    messages = profile.build_messages(
        contract=_contract("bbb_martins"),
        current_level=1,
        query_smiles="CCN",
        condition_sentence="",
        query_prior={"prediction": "positive"},
        query_tool_summary=None,
        active=snapshots[1],
        prior_state=None,
        prompt_version="v4",
        indirect_record_limit=2,
        include_indirect=True,
    )
    payload = json.loads(messages[1]["content"])
    assert payload["active_evidence"][0]["level"] == "L1"
    architecture = payload["protocol"]["architecture"]
    assert "independently ranked Stage 3 record bundle" in architecture
    assert "exact training parent-condition contexts" not in architecture


def test_bbb_v21_raw_record_bundles_accept_per_level_caps() -> None:
    limits = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}
    indirect = _indirect_records(tuple(limits), record_count=5)
    for level, limit in limits.items():
        indirect[level]["records"] = indirect[level]["records"][:limit]
    snapshots = profile.snapshots(
        [],
        task="bbb_martins",
        indirect_records=indirect,
        indirect_record_limit=limits,
        record_levels=tuple(limits),
        transfer_model="V21",
    )

    assert [
        len(card_ids(snapshots[level]))
        - len(card_ids(snapshots[level - 1]))
        if level > 1
        else len(card_ids(snapshots[level]))
        for level in range(1, 6)
    ] == list(limits.values())
    messages = profile.build_messages(
        contract=_contract("bbb_martins"),
        current_level=5,
        query_smiles="CCN",
        condition_sentence="",
        query_prior={"prediction": "positive"},
        query_tool_summary=None,
        active=snapshots[5],
        prior_state=None,
        prompt_version="v4",
        indirect_record_limit=limits,
        include_indirect=True,
    )
    sampling = json.loads(messages[1]["content"])["protocol"]["record_sampling"]
    assert "L1=1, L2=2, L3=3, L4=4, L5=5" in sampling


def test_bbb_v21_molecule_cards_keep_ranked_records_append_only() -> None:
    def ranked_record(record_id: str, level: str, score: float) -> dict:
        return {
            "record_id": record_id,
            "transfer_likelihood": score,
            "payload": {
                "record_id": record_id,
                "progressive_level": level,
                "canonical_record_id": record_id,
                "canonical_endpoint_name": "brain_penetration",
                "canonical_measurement_text": "measured",
                "canonical_smiles": "CCO",
                "source_id": "direct_bbb",
                "support_text": "experimental record",
            },
        }

    contexts = profile.v21_molecule_contexts(
        "query_1",
        [
            {
                "reference_molecule_id": "M1",
                "canonical_smiles": "CCO",
                "transfer_likelihood": 0.9,
                "selection_rank": 0,
                "available_l1": 3,
                "available_l2": 2,
                "l1_records": [ranked_record("r1", "L1", 0.9)],
                "l2_records": [ranked_record("r2", "L2", 0.8)],
            }
        ],
    )
    snapshots = profile.snapshots(
        contexts,
        task="bbb_martins",
        indirect_records=_indirect_records(("L3", "L4", "L5"), record_count=1),
        indirect_record_limit=1,
        record_levels=("L3", "L4", "L5"),
        transfer_model="V21",
    )

    assert len(card_ids(snapshots[1])) == 1
    assert len(card_ids(snapshots[2]) - card_ids(snapshots[1])) == 1
    messages = profile.build_messages(
        contract=_contract("bbb_martins"),
        current_level=3,
        query_smiles="CCN",
        condition_sentence="",
        query_prior={"prediction": "positive"},
        query_tool_summary=None,
        active=snapshots[3],
        prior_state=None,
        prompt_version="v4",
        record_limit=4,
        l2_record_limit=2,
        indirect_record_limit=1,
        include_indirect=True,
    )
    payload = json.loads(messages[1]["content"])
    assert "V21-ranked scaffold-disjoint reference molecules" in payload["protocol"]["architecture"]
    assert payload["active_evidence"][0]["record_rows"][0][2] == "brain_penetration"
    assert "_v21_transfer_likelihood" not in json.dumps(payload)


def test_bioavailability_supports_a_smaller_final_level_record_cap() -> None:
    indirect = _indirect_records(record_count=2)
    indirect["L6"]["records"] = indirect["L6"]["records"][:1]
    limits = {"L3": 2, "L4": 2, "L5": 2, "L6": 1}
    snapshots = profile.snapshots(
        _contexts(),
        task="bioavailability_ma",
        indirect_records=indirect,
        indirect_record_limit=limits,
    )

    assert len(card_ids(snapshots[3]) - card_ids(snapshots[2])) == 2
    assert len(card_ids(snapshots[4]) - card_ids(snapshots[3])) == 2
    assert len(card_ids(snapshots[5]) - card_ids(snapshots[4])) == 2
    assert len(card_ids(snapshots[6]) - card_ids(snapshots[5])) == 1
    messages = profile.build_messages(
        contract=_contract("bioavailability_ma"),
        current_level=6,
        query_smiles="CCN",
        condition_sentence="",
        query_prior={"prediction": "positive"},
        query_tool_summary=None,
        active=snapshots[6],
        prior_state=None,
        prompt_version="v3",
        record_limit=10,
        l2_record_limit=10,
        indirect_record_limit=limits,
        include_indirect=True,
    )
    sampling = json.loads(messages[1]["content"])["protocol"]["record_sampling"]
    assert "L3=2, L4=2, L5=2, L6=1" in sampling


def test_skin_l3_l4_append_without_l5() -> None:
    snapshots = profile.snapshots(
        _contexts(),
        task="skin_reaction",
        indirect_records=_indirect_records(("L3", "L4")),
    )

    assert list(snapshots) == [1, 2, 3, 4]
    assert len(card_ids(snapshots[3]) - card_ids(snapshots[2])) == 50
    assert len(card_ids(snapshots[4]) - card_ids(snapshots[3])) == 50
    assert profile.indirect_level_names("skin_reaction") == ("L3", "L4")

    messages = profile.build_messages(
        contract=_contract("skin_reaction"),
        current_level=4,
        query_smiles="CCN",
        condition_sentence="",
        query_prior={"prediction": "positive"},
        query_tool_summary=None,
        active=snapshots[4],
        prior_state=None,
        prompt_version="v6",
        include_indirect=True,
    )
    assert "L3/L4" in messages[1]["content"]
    assert "L5" not in messages[1]["content"]


def test_asymmetric_record_caps_are_rendered_and_applied() -> None:
    snapshots = profile.snapshots(
        _contexts(),
        task="bioavailability_ma",
        indirect_records=_indirect_records(record_count=12),
        indirect_record_limit=12,
    )

    assert len(card_ids(snapshots[3]) - card_ids(snapshots[2])) == 12
    messages = profile.build_messages(
        contract=_contract("bioavailability_ma"),
        current_level=3,
        query_smiles="CCN",
        condition_sentence="",
        query_prior={"prediction": "positive"},
        query_tool_summary=None,
        active=snapshots[3],
        prior_state=None,
        prompt_version="v3",
        record_limit=4,
        l2_record_limit=2,
        indirect_record_limit=12,
        include_indirect=True,
    )
    sampling = json.loads(messages[1]["content"])["protocol"]["record_sampling"]
    assert "at most 4 raw records at L1" in sampling
    assert "at most 2 new raw records at L2" in sampling
    assert "at most 12 ranked records" in sampling


def test_task_best_prompts_include_skin_v6() -> None:
    assert profile.resolve_prompt_version("bbb_martins", "task_best") == "v4"
    assert profile.resolve_prompt_version("bioavailability_ma", "task_best") == "v3"
    assert profile.resolve_prompt_version("skin_reaction", "task_best") == "v6"
    assert profile.resolve_prompt_version(
        "bbb_martins", "task_best", ranking="morgan"
    ) == "morgan_v4"
    assert profile.resolve_prompt_version(
        "bioavailability_ma", "task_best", ranking="morgan"
    ) == "morgan_v3"
    assert profile.resolve_prompt_version(
        "skin_reaction", "task_best", ranking="morgan"
    ) == "morgan_v6"


def test_seeded_rank_tie_key_is_reproducible() -> None:
    first = seeded_rank_tie_key(0, "task", "query", "molecule", "record")
    assert first == seeded_rank_tie_key(0, "task", "query", "molecule", "record")
    assert first != seeded_rank_tie_key(1, "task", "query", "molecule", "record")


def test_morgan_prompt_exposes_similarity_and_hides_transfer_scores() -> None:
    indirect = _indirect_records(("L3", "L4", "L5"), record_count=12)
    for level in indirect.values():
        for row in level["records"]:
            row["morgan_similarity"] = row.pop("transfer_likelihood")
    active = profile.snapshots(
        _morgan_contexts(),
        task="bbb_martins",
        indirect_records=indirect,
        indirect_record_limit=12,
        prompt_version="morgan_v4",
    )[3]
    messages = profile.build_messages(
        contract=_contract("bbb_martins"),
        current_level=3,
        query_smiles="CCN",
        condition_sentence="",
        query_prior={"prediction": "positive"},
        query_tool_summary=None,
        active=active,
        prior_state=None,
        prompt_version="morgan_v4",
        record_limit=4,
        l2_record_limit=2,
        indirect_record_limit=12,
        include_indirect=True,
    )

    payload = json.loads(messages[1]["content"])
    serialized = json.dumps(messages)
    assert payload["active_evidence"][0]["morgan_similarity"] == 0.73
    assert payload["active_evidence"][-1]["record_columns"][2] == "morgan_similarity"
    assert "morgan_similarity" in serialized
    assert "transfer_likelihood" not in serialized


def test_enabling_l3_l5_does_not_change_l1_prompt() -> None:
    common = {
        "contract": _contract("bioavailability_ma"),
        "current_level": 1,
        "query_smiles": "CCN",
        "condition_sentence": "",
        "query_prior": {"prediction": "positive"},
        "query_tool_summary": None,
        "active": profile.snapshots(_contexts())[1],
        "prior_state": None,
        "prompt_version": "v3",
    }
    assert profile.build_messages(**common) == profile.build_messages(
        **common, include_indirect=True
    )


def test_record_caps_are_deterministic_and_nested() -> None:
    rows = [{"canonical_record_id": f"row_{index:02d}"} for index in range(25)]

    cap_10 = profile._stable_sample(
        rows,
        task="bbb_martins",
        context_id="context_a",
        level=1,
        record_limit=10,
    )
    cap_20 = profile._stable_sample(
        rows,
        task="bbb_martins",
        context_id="context_a",
        level=1,
        record_limit=20,
    )

    assert len(cap_10) == 10
    assert len(cap_20) == 20
    assert cap_10 == cap_20[:10]


def test_context_record_prompt_obeys_yaml_surface() -> None:
    active = profile.snapshots(_contexts())[2]
    messages = profile.build_messages(
        contract=_contract(),
        current_level=2,
        query_smiles="CCN",
        condition_sentence="",
        query_prior={"prediction": "positive"},
        query_tool_summary={"tool_name": "molecule_properties", "content": "properties"},
        active=active,
        prior_state={
            "prediction": "positive",
            "confidence": "moderate",
            "revision_action": "initial",
            "supportive_card_ids": ["record_1"],
            "contradictory_card_ids": [],
            "prediction_basis_card_ids": ["record_1"],
            "not_used_card_ids": ["record_2"],
            "claims": [],
            "new_evidence_assessment": [],
            "evidence_gaps": [],
            "decision_summary": "initial",
        },
    )

    payload = json.loads(messages[1]["content"])
    context = payload["active_evidence"][0]
    assert list(context) == [
        "context_card_id",
        "canonical_smiles",
        "condition_group",
        "transfer_likelihood",
        "available_record_counts",
        "record_cards",
    ]
    assert [card["card_id"] for card in context["record_cards"]] == [
        "C01",
        "C02",
        "C03",
    ]
    serialized = json.dumps(messages)
    for hidden_field in (
        "retrieval_source_id",
        "direct_vote",
        "direct_residual",
        "morgan_tanimoto_similarity",
        "similarity_bucket",
        "distant_analog",
        "very_distant_analog",
        "molecule_relation",
        "query_analog_tool_summaries",
    ):
        assert hidden_field not in serialized


def test_v2_renders_richer_cards_and_task_specific_system_messages() -> None:
    active = profile.snapshots(_contexts())[1]
    system_messages = set()
    for task in ("bbb_martins", "bioavailability_ma", "skin_reaction"):
        messages = profile.build_messages(
            contract=_contract(task),
            current_level=1,
            query_smiles="CCN",
            condition_sentence="",
            query_prior={"prediction": "positive"},
            query_tool_summary=None,
            active=active,
            prior_state=None,
            prompt_version="v2",
        )
        system_messages.add(messages[0]["content"])
        record = json.loads(messages[1]["content"])["active_evidence"][0][
            "record_cards"
        ][0]
        assert {
            "endpoint_name",
            "measurement_text",
            "unit_text",
            "assay_model",
            "extra_details",
            "support_text",
        } <= set(record)
        assert record["endpoint_name"] == "raw outcome"
        assert record["measurement_text"] == "observed positive"
        assert record["support_text"] == "Original source sentence."
        assert not {"endpoint", "result", "unit", "numeric_value"} & set(record)

    assert len(system_messages) == 3


def test_v3_reuses_v2_cards_without_primary_transfer_weighting() -> None:
    active = profile.snapshots(_contexts())[1]
    common = {
        "contract": _contract("bbb_martins"),
        "current_level": 1,
        "query_smiles": "CCN",
        "condition_sentence": "",
        "query_prior": {"prediction": "positive"},
        "query_tool_summary": None,
        "active": active,
        "prior_state": None,
        "record_limit": 20,
    }

    v2 = profile.build_messages(**common, prompt_version="v2")
    v3 = profile.build_messages(**common, prompt_version="v3")

    assert json.loads(v2[1]["content"])["active_evidence"] == json.loads(
        v3[1]["content"]
    )["active_evidence"]
    v2_system = " ".join(v2[0]["content"].split())
    v3_system = " ".join(v3[0]["content"].split())
    assert "primary weight" in v2_system
    assert "one relevance signal" in v3_system
    assert "primary weight" not in v3_system
    assert "at most 20 raw records" in v3[1]["content"]


def test_v4_renders_canonical_values_without_source_fallback() -> None:
    messages = profile.build_messages(
        contract=_contract("bbb_martins"),
        current_level=1,
        query_smiles="CCN",
        condition_sentence="",
        query_prior={"prediction": "positive"},
        query_tool_summary=None,
        active=profile.snapshots(_contexts())[1],
        prior_state=None,
        prompt_version="v4",
    )

    records = json.loads(messages[1]["content"])["active_evidence"][0]["record_cards"]
    assert records[0] == {
        "card_id": "C01",
        "level": "L1",
        "endpoint_name": "canonical outcome",
        "measurement_text": "canonical positive",
        "unit_text": "canonical unit",
        "assay_context": "canonical in vivo study",
        "species_context": "rat",
        "support_text": "Original source sentence.",
    }
    assert records[1] == {"card_id": "C02", "level": "L1"}


def test_v4_ablation_changes_one_card_surface_at_the_final_step() -> None:
    common = {
        "contract": _contract("skin_reaction"),
        "current_level": 1,
        "query_smiles": "CCN",
        "condition_sentence": "",
        "query_prior": {"prediction": "positive"},
        "query_tool_summary": None,
        "active": profile.snapshots(_contexts())[1],
        "prior_state": None,
    }

    evidence = {
        version: json.loads(profile.build_messages(**common, prompt_version=version)[1]["content"])[
            "active_evidence"
        ]
        for version in ("v1", "v4.1", "v4.2", "v4", "v4.3")
    }
    assert evidence["v1"] == evidence["v4.1"] == evidence["v4.2"]
    assert evidence["v4"] == evidence["v4.3"]
    assert evidence["v4.2"] != evidence["v4.3"]


def test_v5_v6_isolate_compact_and_canonical_card_surfaces() -> None:
    common = {
        "contract": _contract("bioavailability_ma"),
        "current_level": 1,
        "query_smiles": "CCN",
        "condition_sentence": "",
        "query_prior": {"prediction": "positive"},
        "query_tool_summary": None,
        "active": profile.snapshots(_contexts())[1],
        "prior_state": None,
    }

    v5 = profile.build_messages(**common, prompt_version="v5")
    v6 = profile.build_messages(**common, prompt_version="v6")
    assert v5[0] == v6[0]
    assert json.loads(v5[1]["content"])["active_evidence"] == json.loads(
        profile.build_messages(**common, prompt_version="v1")[1]["content"]
    )["active_evidence"]
    assert json.loads(v6[1]["content"])["active_evidence"] == json.loads(
        profile.build_messages(**common, prompt_version="v4")[1]["content"]
    )["active_evidence"]


def test_v7_combines_v1_prompt_with_source_native_cards() -> None:
    common = {
        "contract": _contract("bioavailability_ma"),
        "current_level": 1,
        "query_smiles": "CCN",
        "condition_sentence": "",
        "query_prior": {"prediction": "positive"},
        "query_tool_summary": None,
        "active": profile.snapshots(_contexts())[1],
        "prior_state": None,
    }

    v1 = profile.build_messages(**common, prompt_version="v1")
    v2 = profile.build_messages(**common, prompt_version="v2")
    v7 = profile.build_messages(**common, prompt_version="v7")
    assert v7[0] == v1[0]
    assert json.loads(v7[1]["content"])["active_evidence"] == json.loads(
        v2[1]["content"]
    )["active_evidence"]


def test_v8_and_v81_share_the_hybrid_card_surface() -> None:
    common = {
        "contract": _contract("bioavailability_ma"),
        "current_level": 1,
        "query_smiles": "CCN",
        "condition_sentence": "",
        "query_prior": {"prediction": "positive"},
        "query_tool_summary": None,
        "active": profile.snapshots(_contexts())[1],
        "prior_state": None,
    }

    v8 = profile.build_messages(**common, prompt_version="v8")
    v81 = profile.build_messages(**common, prompt_version="v8.1")
    assert json.loads(v8[1]["content"])["active_evidence"] == json.loads(
        v81[1]["content"]
    )["active_evidence"]
    assert v8[0] != v81[0]


def test_v82_keeps_v81_prompt_and_restores_source_native_cards() -> None:
    common = {
        "contract": _contract("bioavailability_ma"),
        "current_level": 1,
        "query_smiles": "CCN",
        "condition_sentence": "",
        "query_prior": {"prediction": "positive"},
        "query_tool_summary": None,
        "active": profile.snapshots(_contexts())[1],
        "prior_state": None,
    }

    v2 = profile.build_messages(**common, prompt_version="v2")
    v81 = profile.build_messages(**common, prompt_version="v8.1")
    v82 = profile.build_messages(**common, prompt_version="v8.2")
    assert v82[0] == v81[0]
    assert json.loads(v82[1]["content"])["active_evidence"] == json.loads(
        v2[1]["content"]
    )["active_evidence"]
