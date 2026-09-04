"""Pinned model and prompt contracts for the supported assay rerankers."""

import json
from pathlib import Path
import sqlite3

import pytest

from predict.retrieval.assay_reranking.runtime import CACHE_ROOT, model_profile
from predict.retrieval.assay_reranking.progressive_levels import (
    ProgressiveV21PromptRenderer,
    ProgressiveV191PromptRenderer,
    _shard_rows,
    _select_luna_relevance_records,
)
from predict.retrieval.policies import decide_candidate, normalize_molecule_identity
from predict.baselines.assay_transfer_knn import select_neighbors
from predict.retrieval.assay_reranking.v19_1 import (
    default_cache_paths as v19_cache_paths,
    verify_vendored_assets as verify_v19_assets,
)
from predict.retrieval.assay_reranking.v9 import (
    V9PromptRenderer,
    default_cache_paths as v9_cache_paths,
    ranking_cache_dir,
    verify_vendored_assets as verify_v9_assets,
)


DIRECT_MODELS = {
    "bbb_martins": (
        "jiosephlee/assay-transfer-tool-soft-v9.0.2-bbb-martins-vote-mean",
        "06b9900222e887597ca06f0015a09fa87b8eb509",
    ),
    "bioavailability_ma": (
        "jiosephlee/assay-transfer-tool-soft-v9-bioavailability-ma-mixed-continuous",
        "6f3aefabc9a07b357066aaf7ca0f69ab63785240",
    ),
    "skin_reaction": (
        "jiosephlee/assay-transfer-tool-soft-v9.0.2-skin-reaction-mixed-continuous",
        "e4e894af28151760d2041275c5ecf136971c3925",
    ),
}


@pytest.mark.parametrize("task_id", sorted(DIRECT_MODELS))
def test_direct_role_is_exactly_v9(task_id):
    profile = model_profile(task_id, "direct")
    assert (profile["model"], profile["revision"]) == DIRECT_MODELS[task_id]
    assert profile["prompt_profile"] == "v9"


@pytest.mark.parametrize(
    ("task_id", "model", "revision"),
    (
        (
            "bbb_martins",
            "jiosephlee/intern-s1-mini-assay-transfer-v19-1-bbb-martins-numeric-best",
            "b93ebfb909de5689fe3b50978d65172a6096b974",
        ),
        (
            "bioavailability_ma",
            "jiosephlee/intern-s1-mini-assay-transfer-v19-1-bioavailability-ma-numeric-best",
            "612cd794583e2129a664defaf9229b26d94b9a69",
        ),
        (
            "skin_reaction",
            "jiosephlee/intern-s1-mini-assay-transfer-v19-1-skin-reaction-numeric-best",
            "f29d100ca2112490d22913736d4efa5bc5308cb6",
        ),
    ),
)
def test_indirect_model_is_pinned(task_id, model, revision):
    profile = model_profile(task_id, "indirect")
    assert (profile["model"], profile["revision"]) == (model, revision)


def test_bbb_all_record_model_is_pinned():
    profile = model_profile("bbb_martins", "all_records")
    assert profile == {
        "model": "jiosephlee/intern-s1-mini-assay-transfer-v21-bbb-martins-mixed-best",
        "revision": "2521166a95cd36bc78fb9b90669cc097ead758ca",
        "prompt_profile": "v21_bbb",
    }


def test_score_shards_batch_prompts_by_length():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE prompt_tasks(prompt_key INTEGER, cache_key TEXT, prompt TEXT)"
    )
    connection.executemany(
        "INSERT INTO prompt_tasks VALUES (?, ?, ?)",
        (
            (1, "long", "xxxx"),
            (2, "other-shard", "x"),
            (3, "short", "xx"),
            (5, "mid", "xxx"),
        ),
    )

    rows = _shard_rows(connection, shard_index=0, num_shards=2)

    assert [cache_key for _, cache_key, _ in rows] == ["short", "mid", "long"]


def test_vendored_prompt_assets_match_their_manifests():
    assert set(verify_v9_assets()) == {"prompt.jinja", "prompt_config.json"}
    assert set(verify_v19_assets()) == {"prompt.jinja", "prompt_projection.json"}


def test_default_caches_live_under_retrieval():
    for paths in (v9_cache_paths("bbb_martins"), v19_cache_paths("bbb_martins")):
        assert Path(paths["cache"]).is_relative_to(CACHE_ROOT)
    assert "v9_direct_gold_morgan200" in str(
        ranking_cache_dir("bbb_martins", pool_size=200)
    )


def test_v9_prompt_contains_gold_value_and_hides_query_value():
    prompt = V9PromptRenderer("bbb_martins").render(
        {"smiles": "CCO", "value": 0.75, "condition_atoms": []},
        {"smiles": "CCN", "condition_atoms": []},
    )
    assert "75.0%" in prompt
    assert prompt.count("75.0%") == 1
    assert "<SMILES>CCO</SMILES>" in prompt
    assert "<SMILES>CCN</SMILES>" in prompt


def test_transfer_knn_restricts_morgan_pool_before_reranking():
    rows = [
        {
            "retrieval_parent_rank": parent_rank,
            "retrieval_parent_context_index": 0,
            "retrieval_record_id": str(parent_rank),
            "model_score": score,
        }
        for parent_rank, score in ((0, 0.1), (1, 0.2), (2, 9.0))
    ]
    selected = select_neighbors(rows, morgan_width=2, k=1)
    assert selected[0]["retrieval_parent_rank"] == 1


def test_scaffold_disjoint_does_not_equate_empty_acyclic_scaffolds():
    query = normalize_molecule_identity("CCO")
    candidate = {"canonical_smiles": "CCN"}
    assert not decide_candidate(query, candidate, "scaffold_disjoint").excluded


def test_progressive_v19_renderer_supports_direct_bbb_records():
    prompt = ProgressiveV191PromptRenderer().render(
        {
            "task_id": "bbb_martins",
            "source_id": "direct_bbb",
            "canonical_smiles": "CCO",
            "endpoint_name": "brain-to-plasma ratio",
            "unit_text": "ratio",
            "measurement_text": "0.5",
            "assay_model": "rat",
            "species": "rat",
        },
        "CCN",
    )
    assert "Known reported measurement: 0.5" in prompt
    assert prompt.count("0.5") == 1
    assert prompt.count("brain-to-plasma ratio") == 2


def test_progressive_v19_renderer_supports_direct_bioavailability_records():
    prompt = ProgressiveV191PromptRenderer("bioavailability_ma").render(
        {
            "task_id": "bioavailability_ma",
            "source_id": "hf_bioavailability",
            "canonical_smiles": "CCO",
            "endpoint_name": "oral bioavailability",
            "measurement_text": "42%",
            "dose": "10 mg/kg oral",
            "species_or_population": "rat",
        },
        "CCN",
    )
    assert "Known reported measurement: 42%" in prompt
    assert prompt.count("42%") == 1
    assert prompt.count("10 mg/kg oral") == 2


def test_progressive_v19_renderer_supports_direct_skin_records():
    prompt = ProgressiveV191PromptRenderer("skin_reaction").render(
        {
            "task_id": "skin_reaction",
            "source_id": "direct_skin_reaction",
            "canonical_smiles": "CCO",
            "endpoint_name": "sensitization",
            "measurement_text": "positive",
            "assay_or_test": "DPRA",
            "outcome_label": "positive",
        },
        "CCN",
    )
    assert prompt.count("Known reported measurement: positive") == 1
    assert prompt.count("Assay or test: DPRA") == 2


def test_progressive_v21_renderer_copies_context_without_query_results():
    prompt = ProgressiveV21PromptRenderer().render(
        {
            "task_id": "bbb_martins",
            "source_id": "direct_bbb",
            "source_smiles": "CCO",
            "endpoint_name": "BBB outcome",
            "measurement_text": "positive",
            "assay_model": "in vivo",
            "species": "rat",
            "support_text": "reported source result",
        },
        "CCN",
    )

    known, query = prompt.split("Experiment B (query measurement hidden)")
    assert "<SMILES>CCO</SMILES>" in known
    assert "<SMILES>CCN</SMILES>" in query
    assert "BBB outcome" in known and "BBB outcome" in query
    assert "in vivo" in known and "in vivo" in query
    assert "positive" in known and "positive" not in query
    assert "reported source result" in known and "reported source result" not in query


def test_luna_relevance_filter_keeps_top_quarter_buckets_and_all_their_records():
    stage3 = []
    mapping = []
    ranking_rows = []
    for index in range(8):
        bucket = json.dumps(
            {"source_id": "efflux_transport", "endpoint": f"endpoint-{index}"}
        )
        ranking_rows.append(
            {
                "source_id": "efflux_transport",
                "relevance_bucket": bucket,
                "bradley_terry_score": float(8 - index),
            }
        )
        for replicate in range(2 if index == 0 else 1):
            record_id = f"record-{index}-{replicate}"
            stage3.append(
                {
                    "canonical_record_id": record_id,
                    "source_id": "efflux_transport",
                    "assay_transfer_eligible": True,
                    "endpoint": f"endpoint-{index}",
                }
            )
            mapping.append(
                {
                    "canonical_record_id": record_id,
                    "progressive_level": "L4",
                    "score_cache_candidate_eligible": True,
                }
            )
    selected, audit = _select_luna_relevance_records(
        stage3,
        mapping,
        ranking_rows,
        {
            "status": "complete",
            "relevance_bucket_columns": {"efflux_transport": ["endpoint"]},
        },
        ("L4",),
    )
    assert selected == {"record-0-0", "record-0-1", "record-1-0"}
    assert audit["stage3_ranking_relevance_identity_sets_equal"] is True
    assert audit["pools"]["L4:efflux_transport"] == {
        "input_relevance_buckets": 8,
        "selected_relevance_buckets": 2,
        "input_records": 9,
        "selected_records": 3,
    }
