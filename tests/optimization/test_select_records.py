import math
import csv
import json

import pytest

from optimization import gold_joint
from optimization.gold_joint import (
    DIRECT_SCHEMA,
    INDIRECT_SCHEMA,
    DirectProfile,
    IndirectProfile,
    _log_ceiling,
    _load_skin_semantics,
    direct_profiles,
    normalized_log_diversity,
    normalized_sqrt_diversity,
    sqrt_indirect_profiles,
    select_direct_contexts,
    select_joint_indirect,
    compose_mixed,
    validate_selection_manifest,
)
from predict.utils.json import sha256_file
from optimization.select_records import (
    PROFILES,
    _default_release_index,
    _feature_coverage,
    _fingerprint,
    _greedy_coverage_ceiling,
    select_records,
)


def test_default_release_index_uses_ranked_uid_v4() -> None:
    assert _default_release_index("bbb_martins").as_posix() == (
        "data/caches/assay_reranking/active/ranked_level_retrieval_v4/"
        "bbb_martins/RELEASE_INDEX.json"
    )


def _candidates() -> list[dict]:
    return [
        {
            "item_id": "u1", "parent_id": "a", "parent_smiles": "CCO",
            "morgan_similarity": 0.90, "assay_transfer_score": 0.10,
            "semantic_bucket_id": "s1", "semantic_weight": 0.10,
        },
        {
            "item_id": "u2", "parent_id": "a", "parent_smiles": "CCO",
            "morgan_similarity": 0.90, "assay_transfer_score": 0.10,
            "semantic_bucket_id": "s1", "semantic_weight": 0.10,
        },
        {
            "item_id": "u3", "parent_id": "b", "parent_smiles": "c1ccccc1",
            "morgan_similarity": 0.80, "assay_transfer_score": 0.95,
            "semantic_bucket_id": "s2", "semantic_weight": 0.95,
        },
        {
            "item_id": "u4", "parent_id": "c", "parent_smiles": "CCN(C)C",
            "morgan_similarity": 0.70, "assay_transfer_score": 0.80,
            "semantic_bucket_id": "s3", "semantic_weight": 0.80,
        },
    ]


def test_zero_lambdas_are_exact_morgan_top_k() -> None:
    selected, summary = select_records(_candidates(), k=2)

    assert [row["item_id"] for row in selected] == ["u1", "u2"]
    assert summary["morgan"] == pytest.approx(0.9)
    assert summary["objective_score"] == pytest.approx(0.9)


def test_assay_transfer_changes_selection() -> None:
    selected, summary = select_records(_candidates(), k=2, assay_lambda=1)

    assert {row["item_id"] for row in selected} == {"u3", "u4"}
    assert summary["assay"] == pytest.approx(0.875)


def test_gated_assay_requires_both_structure_and_transfer() -> None:
    candidates = _candidates()
    candidates[0]["assay_transfer_score"] = 0.4
    candidates[1]["assay_transfer_score"] = 0.4
    candidates[2]["morgan_similarity"] = 0.2
    candidates[2]["assay_transfer_score"] = 1.0
    candidates[3]["assay_transfer_score"] = 0.1

    additive, _ = select_records(candidates, k=1, assay_lambda=2)
    gated, summary = select_records(candidates, k=1, gated_assay_lambda=2)

    assert additive[0]["item_id"] == "u3"
    assert gated[0]["item_id"] == "u1"
    assert summary["assay_gated"] == pytest.approx(0.36)


def test_normalized_gated_relevance_uses_alpha_and_unit_scale() -> None:
    candidates = _candidates()
    selected, summary = select_records(
        candidates, k=1, gated_assay_lambda=1,
        normalized_gated_objective=True,
    )

    assert selected[0]["item_id"] == "u3"
    assert summary["normalized_relevance"] == pytest.approx((0.8 + 0.76) / 2)
    assert summary["objective_score"] == pytest.approx(summary["normalized_relevance"])


def test_normalized_gated_relevance_falls_back_to_morgan_without_assay() -> None:
    candidates = _candidates()
    for row in candidates:
        row["assay_transfer_score"] = None

    selected, summary = select_records(
        candidates, k=2, gated_assay_lambda=1.25,
        normalized_gated_objective=True,
    )

    assert [row["item_id"] for row in selected] == ["u1", "u2"]
    assert summary["normalized_relevance"] == pytest.approx(summary["morgan"])
    assert summary["gated_assay_lambda_effective"] == 0


def test_feature_coverage_diversifies_without_full_similarity_matrix() -> None:
    baseline, baseline_summary = select_records(_candidates(), k=2)
    selected, summary = select_records(_candidates(), k=2, molecular_lambda=1)

    assert [row["parent_id"] for row in baseline] == ["a", "a"]
    assert len({row["parent_id"] for row in selected}) == 2
    assert summary["molecular_coverage"] > baseline_summary["molecular_coverage"]
    assert summary["molecular_diversity"] > baseline_summary["molecular_diversity"]


def test_feature_coverage_has_diminishing_returns() -> None:
    universe = {1, 2, 3, 4}
    item = {2, 3}
    small = _feature_coverage(({1}, item), universe) - _feature_coverage(({1},), universe)
    large = _feature_coverage(({1}, {2}), universe)
    large = _feature_coverage(({1}, {2}, item), universe) - large

    assert small >= large


def test_greedy_coverage_ceiling_rescales_the_attainable_k_coverage() -> None:
    bits = [frozenset({1, 2}), frozenset({2, 3}), frozenset({4})]

    assert _greedy_coverage_ceiling(bits, frozenset({1, 2, 3, 4}), 2) == 0.75
    _, summary = select_records(
        _candidates(), k=2, molecular_lambda=1,
        normalized_gated_objective=True,
    )
    assert 0 < summary["molecular_coverage_ceiling"] <= 1
    assert 0 <= summary["molecular_coverage_normalized"] <= 1


def test_semantic_relevance_and_diversity_have_separate_effects() -> None:
    relevant, relevance_summary = select_records(
        _candidates(), k=2, semantic_relevance_lambda=1
    )
    diverse, diversity_summary = select_records(
        _candidates(), k=2, semantic_diversity_lambda=1
    )

    assert {row["item_id"] for row in relevant} == {"u3", "u4"}
    assert relevance_summary["semantic_relevance"] == pytest.approx(0.875)
    assert len({row["semantic_bucket_id"] for row in diverse}) == 2
    assert diversity_summary["semantic_coverage"] == pytest.approx(1.0)
    assert diversity_summary["semantic_diversity"] == pytest.approx(1.0)


def test_unavailable_assay_term_is_omitted_for_that_level() -> None:
    candidates = _candidates()
    for row in candidates:
        row["assay_transfer_score"] = None

    selected, summary = select_records(
        candidates, k=2, assay_lambda=1, gated_assay_lambda=1
    )

    assert [row["item_id"] for row in selected] == ["u1", "u2"]
    assert summary["assay_available"] is False
    assert summary["assay_lambda_effective"] == 0
    assert summary["gated_assay_lambda_effective"] == 0


def test_components_are_normalized_for_interpretable_lambdas() -> None:
    _, summary = select_records(
        _candidates(), k=3, assay_lambda=1, molecular_lambda=1,
        semantic_relevance_lambda=1, semantic_diversity_lambda=1,
    )

    for field in (
        "morgan", "assay", "assay_gated", "molecular_coverage", "molecular_diversity",
        "semantic_relevance", "semantic_coverage", "semantic_diversity",
    ):
        assert 0 <= summary[field] <= 1


@pytest.mark.parametrize(
    "kwargs",
    ({"k": 0}, {"k": 5}, {"k": 1, "molecular_lambda": -0.1}),
)
def test_invalid_constraints_fail(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        select_records(_candidates(), **kwargs)


def test_partial_assay_scores_and_inconsistent_parent_similarity_fail() -> None:
    partial = _candidates()
    partial[0]["assay_transfer_score"] = None
    with pytest.raises(ValueError, match="complete or absent"):
        select_records(partial, k=2)

    inconsistent = _candidates()
    inconsistent[1]["morgan_similarity"] = 0.89
    with pytest.raises(ValueError, match="varies within parent"):
        select_records(inconsistent, k=2)


def test_profile_grid_contains_baseline_single_terms_and_balanced_values() -> None:
    assert len(PROFILES) == 16
    assert PROFILES["baseline"].lambdas() == {
        "assay": 0.0, "assay_gated": 0.0, "molecular_coverage": 0.0,
        "semantic_relevance": 0.0, "semantic_coverage": 0.0,
    }
    assert PROFILES["balanced_10"].gated_assay_lambda == 0
    assert set(PROFILES["balanced_10"].lambdas().values()) == {0.0, 1.0}


def test_log_diversity_is_capacity_normalized_and_has_diminishing_returns() -> None:
    capacities = {"a": 3, "b": 1}

    expected = math.log(3) + math.log(2)
    assert _log_ceiling(capacities, 3) == pytest.approx(expected)
    assert normalized_log_diversity(("a", "a", "b"), capacities, 3) == 1
    assert 0 <= normalized_log_diversity(("a",), capacities, 3) <= 1
    first = math.log(2)
    second = math.log(3) - math.log(2)
    assert first > second > 0


def test_sqrt_grid_and_capacity_normalization() -> None:
    profiles = sqrt_indirect_profiles()
    assert len(profiles) == len({profile.name for profile in profiles}) == 48
    assert {profile.gated_assay for profile in profiles} == {0.75, 1.0}
    assert {profile.semantic_relevance for profile in profiles} == {0.1, 0.25, 0.5}
    assert {profile.molecular for profile in profiles} == {0.1, 0.25}
    assert {profile.semantic_diversity for profile in profiles} == {0.1, 0.25}
    assert {profile.level_diversity for profile in profiles} == {0.1, 0.25}
    capacities = {"a": 3, "b": 1}
    assert normalized_sqrt_diversity(("a", "a", "b"), capacities, 3) == pytest.approx(1)
    assert normalized_sqrt_diversity(("a",), capacities, 3) < 1
    assert 1 > math.sqrt(2) - 1 > math.sqrt(3) - math.sqrt(2) > 0


def test_sqrt_indirect_keeps_linear_morgan_coverage() -> None:
    candidates = [
        {"item_id": uid, "parent_smiles": smiles, "level": level,
         "morgan_similarity": similarity, "assay_transfer_score": None,
         "semantic_bucket_id": bucket, "semantic_weight": None}
        for uid, smiles, level, similarity, bucket in (
            ("u1", "CCO", "L2", 0.9, "a"),
            ("u2", "CCN", "L2", 0.8, "a"),
            ("u3", "c1ccccc1", "L3", 0.7, "b"),
        )
    ]
    selected, summary = select_joint_indirect(
        candidates, profile=IndirectProfile("test", 1, 0.25, 0.5, 0.25, 0.25, ()),
        budget=2, diversity_function="sqrt",
    )
    assert len({row["item_id"] for row in selected}) == 2
    assert summary["semantic_relevance_lambda_effective"] == 0
    assert all(0 <= summary[key] <= 1 for key in (
        "molecular_coverage_normalized", "semantic_diversity", "level_diversity",
    ))
    universe = set().union(*(
        set(_fingerprint(row["parent_smiles"]).GetOnBits()) for row in candidates
    ))
    covered = set().union(*(
        set(_fingerprint(row["parent_smiles"]).GetOnBits()) for row in selected
    ))
    assert summary["molecular_coverage_normalized"] == pytest.approx(len(covered) / len(universe))


def test_skin_v6_weights_preserve_unweighted_candidates() -> None:
    rows, audit = _load_skin_semantics("v10_main_universe_v6")
    assert audit["release"] == "v10_main_universe_v6"
    assert len(rows) == audit["weighted_records"] + audit["unweighted_records"]
    assert audit["weighted_records"] > 0 and audit["unweighted_records"] > 0
    assert any(row["semantic_weight"] is None for row in rows.values())
    assert any(row["semantic_weight"] is not None for row in rows.values())

    candidates = [
        {"item_id": uid, "parent_smiles": smiles, "level": "L2",
         "morgan_similarity": 0.5, "assay_transfer_score": None,
         "semantic_bucket_id": bucket, "semantic_weight": weight}
        for uid, smiles, bucket, weight in (
            ("a", "CCO", "x", 0.8), ("b", "CCN", "y", None),
        )
    ]
    _, summary = select_joint_indirect(
        candidates, profile=IndirectProfile("test", 1, 0.1, 0.5, 0.1, 0.1, ()),
        budget=2, diversity_function="sqrt",
    )
    assert not summary["semantic_relevance_available"]
    assert summary["semantic_relevance_lambda_effective"] == 0


def test_direct_selector_uses_cards_bit_coverage_and_log_labels() -> None:
    candidates = [
        {
            "item_id": "c1", "parent_id": "p1", "parent_smiles": "CCO",
            "morgan_similarity": 0.9, "assay_transfer_score": 0.9, "gold_label": 0,
        },
        {
            "item_id": "c2", "parent_id": "p1", "parent_smiles": "CCO",
            "morgan_similarity": 0.89, "assay_transfer_score": 0.9, "gold_label": 0,
        },
        {
            "item_id": "c3", "parent_id": "p2", "parent_smiles": "c1ccccc1",
            "morgan_similarity": 0.7, "assay_transfer_score": 0.8, "gold_label": 1,
        },
    ]
    selected, summary = select_direct_contexts(
        candidates,
        profile=DirectProfile("test", gated_assay=1, molecular=1, label=1),
        budget=2,
    )

    assert {row["item_id"] for row in selected} == {"c1", "c3"}
    assert summary["label_0"] == summary["label_1"] == 1
    assert 0 <= summary["molecular_coverage_normalized"] <= 1
    assert summary["label_diversity"] == 1


def test_joint_indirect_selector_has_no_level_quota_and_normalized_log_terms() -> None:
    candidates = [
        {
            "item_id": "u1", "parent_id": "p1", "parent_smiles": "CCO", "level": "L2",
            "morgan_similarity": 0.9, "assay_transfer_score": 0.8,
            "semantic_bucket_id": "s1", "semantic_weight": 0.5,
        },
        {
            "item_id": "u2", "parent_id": "p2", "parent_smiles": "CCN", "level": "L2",
            "morgan_similarity": 0.8, "assay_transfer_score": None,
            "semantic_bucket_id": "s1", "semantic_weight": 0.4,
        },
        {
            "item_id": "u3", "parent_id": "p3", "parent_smiles": "c1ccccc1", "level": "L3",
            "morgan_similarity": 0.3, "assay_transfer_score": None,
            "semantic_bucket_id": "s2", "semantic_weight": 0.3,
        },
    ]
    profile = IndirectProfile("test", 1, 0, 0, 0, 0, ())
    selected, summary = select_joint_indirect(candidates, profile=profile, budget=2)

    assert [row["item_id"] for row in selected] == ["u1", "u2"]
    assert summary["level_counts"] == {"L2": 2}
    assert 0 <= summary["semantic_diversity"] <= 1
    assert 0 <= summary["level_diversity"] <= 1


def test_direct_grid_adds_only_the_twelve_low_assay_followups() -> None:
    profiles = direct_profiles()
    expected = {
        (ga, mc, label)
        for ga in (0.75, 1.0, 1.25)
        for mc in (0.0, 0.1, 0.25)
        for label in (0.0, 0.1, 0.25)
    } | {
        (ga, mc, label)
        for ga in (0.25, 0.5)
        for mc in (0.0, 0.1)
        for label in (0.1, 0.25, 0.35)
    }

    assert len(profiles) == len(set(profiles)) == 39
    assert {(p.gated_assay, p.molecular, p.label) for p in profiles} == expected


def test_direct_grid_accepts_explicit_zero_assay_profile(tmp_path, monkeypatch) -> None:
    query_path = tmp_path / "test.jsonl"
    query_path.write_text('{"benchmark_row_id":"q","drug":"CCO"}\n', encoding="utf-8")
    rows = [{
        "item_id": f"c{i}", "parent_id": f"p{i}", "parent_smiles": "CCO",
        "morgan_similarity": 0.9 - i / 100,
        "assay_transfer_score": 0.5, "gold_label": i % 2,
    } for i in range(10)]
    monkeypatch.setattr(gold_joint, "_direct_queries", lambda *args: ({"q": "CCO"}, query_path))
    monkeypatch.setattr(gold_joint, "_load_ranked_direct", lambda *args: ({"q": rows}, {}))
    monkeypatch.setattr(gold_joint, "cache_profile_root", lambda *args: tmp_path)
    grid = gold_joint.build_direct(
        tmp_path / "selection", ["bbb_martins"], benchmark="gold_v1", subset="test",
        profile_names=["ga000_mc010_label010"],
    )
    document = json.loads(grid.read_text())
    assert document["profile_count"] == 1
    leaf = grid.parent / document["profiles"][0]["task_manifests"]["bbb_martins"]["path"]
    assert validate_selection_manifest(leaf)["objective"]["gated_assay_lambda"] == 0


@pytest.mark.parametrize("budget", (25, 50, 100))
def test_sqrt_indirect_test_selects_only_requested_profile(tmp_path, monkeypatch, budget) -> None:
    query_path = tmp_path / "test.jsonl"
    query_path.write_text('{"benchmark_row_id":"q","drug":"CCO"}\n', encoding="utf-8")
    selected_profile = "ga000_mc010_sr050_sd010_ld025"
    seen = {}

    monkeypatch.setattr(gold_joint, "_direct_queries", lambda *args: ({"q": "CCO"}, query_path))
    def load(task, queries, **kwargs):
        seen.update(kwargs)
        return {"q": []}, {}
    monkeypatch.setattr(gold_joint, "_load_indirect", load)
    monkeypatch.setattr(gold_joint, "_select_indirect_query", lambda item: (
        item[0], [(item[2][0].name, [
            {"selection_rank": i, "level": "L2", "item_id": f"u{i}"}
            for i in range(1, item[4] + 1)
        ], {"level_counts": {"L2": item[4]}})],
    ))

    grid = gold_joint.build_indirect(
        tmp_path / "selection", ["bbb_martins"], benchmark="gold_v1",
        subset="test", grid="sqrt48", profile_names=[selected_profile], budget=budget,
    )

    document = json.loads(grid.read_text())
    assert seen["subset"] == "test"
    assert seen["budget"] == budget
    assert document["profile_count"] == 1
    assert document["subset"] == "test"
    leaf = grid.parent / document["profiles"][0]["task_manifests"]["bbb_martins"]["path"]
    assert validate_selection_manifest(leaf)["budget"] == budget


def test_partial_indirect_cache_uses_only_scored_reviewed_records(tmp_path, monkeypatch) -> None:
    bundle = tmp_path / "cache.yaml"
    bundle.write_text("caches:\n  carcinogens:\n    later: RELEASE_INDEX.json\n")
    (tmp_path / "RELEASE_INDEX.json").write_text(json.dumps({
        "evidence": {"manifest": "evidence_libraries/carcinogens/v10_main_universe_v3/VERSION.json"},
        "score_coverage": "partial_snapshot",
    }))
    rows = [{"item_id": f"u{i}", "assay_transfer_score": 0.5} for i in range(50)]
    rows.append({"item_id": "unscored", "assay_transfer_score": None})
    rows.append({"item_id": "unreviewed", "assay_transfer_score": 0.5})
    ranked = {level: {"q": rows if level == "L2" else []}
              for level in gold_joint.TASKS["carcinogens"]["levels"]}
    monkeypatch.setattr(gold_joint, "load_ranked_universe", lambda *args, **kwargs: (
        ranked, tmp_path / "evidence.json", {},
    ))
    monkeypatch.setattr(gold_joint, "_load_reviewed_semantics", lambda *args: (
        {(f"u{i}", "L2"): {"semantic_bucket_id": "s", "semantic_weight": 0.5}
         for i in range(50)}, {},
    ))

    selected, audit = gold_joint._load_indirect(
        "carcinogens", {"q": "CCO"}, benchmark="gold_v1", subset="test",
        cache_bundle=bundle,
    )
    assert len(selected["q"]) == 50
    assert audit["cache_bundle_sha256"] == sha256_file(bundle)
    assert audit["excluded_unscored_candidates"] == 1
    assert audit["excluded_unreviewed_candidates"] == 1
    rows.pop(0)
    with pytest.raises(ValueError, match="only 49 eligible"):
        gold_joint._load_indirect(
            "carcinogens", {"q": "CCO"}, benchmark="gold_v1", subset="test",
            cache_bundle=bundle,
        )


def _selection_manifest(tmp_path, schema: str) -> object:
    direct = schema == DIRECT_SCHEMA
    records = tmp_path / ("direct.tsv" if direct else "indirect.tsv")
    fields = (
        ("benchmark_row_id", "selection_rank", "context_id", "parent_id", "gold_label")
        if direct else
        ("benchmark_row_id", "selection_rank", "level", "source_row_uid")
    )
    row = (
        {"benchmark_row_id": "q", "selection_rank": 1, "context_id": "c", "parent_id": "p", "gold_label": 1}
        if direct else
        {"benchmark_row_id": "q", "selection_rank": 1, "level": "L2", "source_row_uid": "u"}
    )
    with records.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    manifest = tmp_path / ("direct.json" if direct else "indirect.json")
    manifest.write_text(json.dumps({
        "schema_version": schema, "status": "complete", "task_id": "bbb_martins",
        "subset": "valid_small", "benchmark_row_ids": ["q"], "budget": 1,
        "records": {"path": records.name, "sha256": sha256_file(records), "row_count": 1},
    }), encoding="utf-8")
    return manifest


def test_mixed_manifest_hash_pins_validated_direct_and_indirect(tmp_path) -> None:
    direct = _selection_manifest(tmp_path, DIRECT_SCHEMA)
    indirect = _selection_manifest(tmp_path, INDIRECT_SCHEMA)

    output = compose_mixed(direct, indirect, tmp_path / "mixed.json")
    document = json.loads(output.read_text())

    assert document["direct"]["sha256"] == sha256_file(direct)
    assert document["indirect"]["sha256"] == sha256_file(indirect)
    assert validate_selection_manifest(direct)["schema_version"] == DIRECT_SCHEMA
    assert validate_selection_manifest(output)["schema_version"] == "gold_mixed_selection.v1"


def test_mixed_manifest_rejects_cross_benchmark_panels(tmp_path) -> None:
    direct = _selection_manifest(tmp_path, DIRECT_SCHEMA)
    indirect = _selection_manifest(tmp_path, INDIRECT_SCHEMA)
    for path, benchmark in ((direct, "gold_v1"), (indirect, "tdc_v1")):
        document = json.loads(path.read_text())
        document["benchmark"] = benchmark
        path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="benchmark"):
        compose_mixed(direct, indirect, tmp_path / "mixed.json")
