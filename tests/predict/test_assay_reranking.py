"""Pinned model and prompt contracts for the supported assay rerankers."""

import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from predict.retrieval.assay_reranking.runtime import (
    ACTIVE_CACHE_ROOT,
    ARCHIVE_CACHE_ROOT,
    DATA_ACTIVE_CACHE_ROOT,
    cache_profile_root,
    model_profile,
)
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
    _current_candidates,
    _resolve_gold_input,
    V9PromptRenderer,
    default_cache_paths as v9_cache_paths,
    ranking_cache_dir,
    reference_provenance,
    validate_direct_release,
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

V10_3_DIRECT_MODELS = {
    "bbb_martins": (
        "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-3-bbb-martins-best",
        "7678f7a1c43932f7612cfd49a8f4872d6e2f4cab",
        120,
    ),
    "bioavailability_ma": (
        "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-3-bioavailability-ma-best",
        "e8837889f707d97f8aeb7e54e74fcf2c7c2968de",
        140,
    ),
}

V10_4_DIRECT_MODELS = {
    "bbb_martins": (
        "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-4-bbb-martins-best",
        "4c064f571a53234c5eb1c9d4c0e068bdd10860c1",
        70,
    ),
    "bioavailability_ma": (
        "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-4-bioavailability-ma-best",
        "e3a5b31ce8c47b4aa950c974cf9c3eb80c67fcae",
        200,
    ),
}

V10_3_0_2_ORAL_MODEL = (
    "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-3-0-2-bioavailability-ma-BEST",
    "29cc74f02df160b1f153edb700da66d1c30ef4ed",
    160,
)


@pytest.mark.parametrize("task_id", sorted(DIRECT_MODELS))
def test_direct_role_is_exactly_v9(task_id):
    profile = model_profile(task_id, "direct")
    assert (profile["model"], profile["revision"]) == DIRECT_MODELS[task_id]
    assert profile["prompt_profile"] == "v9"


@pytest.mark.parametrize("task_id", sorted(V10_3_DIRECT_MODELS))
def test_v10_3_direct_role_and_official_context_provenance(task_id):
    profile = model_profile(task_id, "direct_v10_3")
    assert (profile["model"], profile["revision"], profile["checkpoint_step"]) == (
        V10_3_DIRECT_MODELS[task_id]
    )
    assert profile["prompt_profile"] == "v10_3"
    provenance = reference_provenance(task_id, "v10_3")
    assert provenance["candidate_source"] == "official_conditioned_gold_train_contexts"
    assert provenance["upstream_prompt_sha256"] == verify_v9_assets()["prompt.jinja"]


def test_v10_3_best_uses_new_oral_model_and_released_display_contract():
    profile = model_profile("bioavailability_ma", "direct_v10_3_0_2")
    assert (profile["model"], profile["revision"], profile["checkpoint_step"]) == (
        V10_3_0_2_ORAL_MODEL
    )
    provenance = reference_provenance("bioavailability_ma", "v10_3_best")
    assert provenance["prompt_contract"] == {
        "direct": "frozen_v10.3",
        "indirect": "canonical_first_atomic_pair.v10.3",
    }
    assert provenance["measurement_display"] == {
        "policy": "canonical_first",
        "fallback": "atomic_measurement_unit_pair",
    }


def test_published_direct_gold_release_is_complete_and_valid():
    receipt = validate_direct_release()
    assert receipt["status"] == "pass"
    assert receipt["entries"] == 6
    assert receipt["files"] == 12
    assert receipt["bytes"] == 12_495_428


def test_cache_validation_rebinds_missing_historical_gold_path(
    tmp_path, monkeypatch,
):
    canonical = Path("data/gold.jsonl")
    source = tmp_path / canonical
    source.parent.mkdir(parents=True)
    source.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "predict.retrieval.assay_reranking.v9.REPO_ROOT", tmp_path
    )

    assert _resolve_gold_input("/missing/old/gold.jsonl", canonical, digest) == source
    with pytest.raises(ValueError, match="Canonical Gold input"):
        _resolve_gold_input("/missing/old/gold.jsonl", canonical, "0" * 64)


@pytest.mark.parametrize("task_id", sorted(V10_4_DIRECT_MODELS))
def test_v10_4_direct_role_and_v2_provenance(task_id):
    profile = model_profile(task_id, "direct_v10_4")
    assert (profile["model"], profile["revision"], profile["checkpoint_step"]) == (
        V10_4_DIRECT_MODELS[task_id]
    )
    assert profile["prompt_profile"] == "v10_4"
    provenance = reference_provenance(task_id, "v10_4")
    assert provenance["gold_release"] == "v2"
    assert provenance["prompt_contract"] == "v2_gold_parent_smiles"


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


def test_cache_profiles_resolve_to_explicit_active_or_archive_roots():
    for paths in (v9_cache_paths("bbb_martins"), v19_cache_paths("bbb_martins")):
        assert Path(paths["cache"]).is_relative_to(ARCHIVE_CACHE_ROOT)
    assert ranking_cache_dir("bbb_martins", pool_size=200).is_relative_to(
        ARCHIVE_CACHE_ROOT
    )
    assert ranking_cache_dir(
        "bbb_martins", lineage="v10_3"
    ).is_relative_to(ACTIVE_CACHE_ROOT)
    assert ranking_cache_dir(
        "bbb_martins", lineage="v10_4"
    ).is_relative_to(ACTIVE_CACHE_ROOT)
    assert ranking_cache_dir(
        "bioavailability_ma", lineage="v10_3_best"
    ).is_relative_to(ACTIVE_CACHE_ROOT)
    for profile in (
        "cache_matched_retrieval_v3",
        "v24_1_bbb_uid_levels_morgan75",
        "v25_oral_uid_levels_morgan75",
        "v25_oral_uid_levels_morgan75_l2",
        "v25_oral_uid_levels_morgan75_l3",
    ):
        assert cache_profile_root(profile).parent == ACTIVE_CACHE_ROOT
    assert cache_profile_root("ranked_level_retrieval_v3").parent == DATA_ACTIVE_CACHE_ROOT
    assert ranking_cache_dir("skin_reaction").is_relative_to(ACTIVE_CACHE_ROOT)
    assert ranking_cache_dir(
        "skin_reaction", lineage="v9_scaffold"
    ).is_relative_to(ACTIVE_CACHE_ROOT)


def test_v10_4_candidates_use_stable_parent_order_and_scaffold_exclusion(
    tmp_path, monkeypatch,
):
    rows = [
        {
            "benchmark_row_id": record_id, "molecule_identity_key": parent,
            "molecule_identity": {"parent_smiles": smiles}, "drug": smiles,
            "bemis_murcko_scaffold": scaffold, "label_counts": {"1": 1},
            "source_record_count": 1, "Y": 1,
        }
        for record_id, parent, smiles, scaffold in (
            ("z", "parent-z", "CCN", "shared"),
            ("b", "parent-b", "CCC", "other"),
            ("a", "parent-a", "CCO", "acyclic-a"),
        )
    ]
    query = {
        "benchmark_row_id": "q", "molecule_identity_key": "query",
        "molecule_identity": {"parent_smiles": "CCCl"}, "drug": "CCCl",
        "bemis_murcko_scaffold": "shared", "label_counts": {"0": 1},
        "source_record_count": 1, "Y": 0,
    }
    train_path, query_path = tmp_path / "train.jsonl", tmp_path / "query.jsonl"
    train_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    query_path.write_text(json.dumps(query) + "\n")
    monkeypatch.setattr(
        "predict.retrieval.assay_reranking.v9.reference_provenance", lambda *_: {}
    )

    selected, inputs = _current_candidates(
        "bbb_martins", 2, train_path=train_path, query_path=query_path, lineage="v10_4"
    )

    assert {row["retrieval_molecule_identity_key"] for row in selected} == {
        "parent-a", "parent-b",
    }
    assert inputs["gold_release"] == "v2"
    assert inputs["neighbor_identity_policy"] == "scaffold_disjoint"


def test_v10_3_best_candidates_are_scaffold_disjoint(tmp_path):
    rows = [
        {
            "benchmark_row_id": record_id, "molecule_identity_key": parent,
            "drug": smiles, "bemis_murcko_scaffold": scaffold,
            "label_counts": {"1": 1}, "source_record_count": 1, "Y": 1,
        }
        for record_id, parent, smiles, scaffold in (
            ("same", "parent-same", "CCN", "shared"),
            ("keep", "parent-keep", "CCC", "other"),
        )
    ]
    query = {
        "benchmark_row_id": "q", "molecule_identity_key": "query", "drug": "CCCl",
        "bemis_murcko_scaffold": "shared", "label_counts": {"0": 1},
        "source_record_count": 1, "Y": 0,
    }
    train_path, query_path = tmp_path / "train.jsonl", tmp_path / "query.jsonl"
    train_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    query_path.write_text(json.dumps(query) + "\n")

    selected, inputs = _current_candidates(
        "bbb_martins", 1, train_path=train_path, query_path=query_path,
        lineage="v10_3_best",
    )

    assert [row["retrieval_record_id"] for row in selected] == ["keep"]
    assert inputs["gold_release"] == "v1"
    assert inputs["neighbor_identity_policy"] == "scaffold_disjoint"


def test_v9_skin_successor_candidates_are_scaffold_disjoint(tmp_path):
    rows = [
        {
            "benchmark_row_id": record_id, "molecule_identity_key": parent,
            "drug": smiles, "bemis_murcko_scaffold": scaffold,
            "label_counts": {"1": 1}, "source_record_count": 1, "Y": 1,
        }
        for record_id, parent, smiles, scaffold in (
            ("same", "parent-same", "CCN", "shared"),
            ("keep", "parent-keep", "CCC", "other"),
        )
    ]
    query = {
        "benchmark_row_id": "q", "molecule_identity_key": "query", "drug": "CCCl",
        "bemis_murcko_scaffold": "shared", "label_counts": {"0": 1},
        "source_record_count": 1, "Y": 0,
    }
    train_path, query_path = tmp_path / "train.jsonl", tmp_path / "query.jsonl"
    train_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    query_path.write_text(json.dumps(query) + "\n")

    selected, inputs = _current_candidates(
        "skin_reaction", 1, train_path=train_path, query_path=query_path,
        lineage="v9_scaffold",
    )

    assert [row["retrieval_record_id"] for row in selected] == ["keep"]
    assert inputs["neighbor_identity_policy"] == "scaffold_disjoint"


def test_v10_3_best_parent_candidates_allow_same_scaffold(tmp_path):
    rows = [
        {
            "benchmark_row_id": "same-scaffold", "molecule_identity_key": "other-parent",
            "drug": "CCC", "bemis_murcko_scaffold": "shared",
            "label_counts": {"1": 1}, "source_record_count": 1, "Y": 1,
        },
    ]
    query = {
        "benchmark_row_id": "q", "molecule_identity_key": "query-parent", "drug": "CCCl",
        "bemis_murcko_scaffold": "shared", "label_counts": {"0": 1},
        "source_record_count": 1, "Y": 0,
    }
    train_path, query_path = tmp_path / "train.jsonl", tmp_path / "query.jsonl"
    train_path.write_text(json.dumps(rows[0]) + "\n")
    query_path.write_text(json.dumps(query) + "\n")

    selected, inputs = _current_candidates(
        "bbb_martins", 1, train_path=train_path, query_path=query_path,
        lineage="v10_3_best_parent",
    )

    assert [row["retrieval_record_id"] for row in selected] == ["same-scaffold"]
    assert inputs["neighbor_identity_policy"] == "parent_disjoint"


def test_v9_prompt_contains_gold_value_and_hides_query_value():
    prompt = V9PromptRenderer("bbb_martins").render(
        {"smiles": "CCO", "value": 0.75, "condition_atoms": []},
        {"smiles": "CCN", "condition_atoms": []},
    )
    assert "75.0%" in prompt
    assert prompt.count("75.0%") == 1
    assert "<SMILES>CCO</SMILES>" in prompt
    assert "<SMILES>CCN</SMILES>" in prompt


def test_v9_skin_prompt_matches_archived_cache_hash():
    prompt = V9PromptRenderer("skin_reaction").render(
        {
            "smiles": "Nc1ccc(N=Nc2ccc([N+](=O)[O-])cc2)cc1",
            "value": 79 / 82,
            "condition_atoms": [],
        },
        {
            "smiles": "Cc1ncc([N+](=O)[O-])n1CC(O)CCl",
            "condition_atoms": [],
        },
    )
    assert hashlib.sha256(prompt.encode()).hexdigest() == (
        "983b65caf27f9af3576a2592f496a0b309ada93e38a9c6d7eb90b3bebf6c0d61"
    )


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
