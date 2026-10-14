from optimization.selection.objectives import _fingerprint
from optimization.selection import policies
import math
import pytest
from optimization.selection.pool import (
    coverage_band as gold_coverage_band,
    load_coverage,
)
from optimization.selection.objectives import (
    DirectProfile,
    IndirectProfile,
    _log_ceiling,
    normalized_log_diversity,
    normalized_sqrt_diversity,
    select_direct_contexts,
    select_joint_indirect,
)
from predict.utils.json import sha256_file, write_json_atomic


def test_available_score_changes_direct_choice_and_missing_score_uses_morgan() -> None:
    rows = [
        {
            "item_id": "a",
            "parent_smiles": "CCO",
            "morgan_similarity": 0.9,
            "assay_transfer_score": 0.1,
            "gold_label": 0,
            "canonical_smiles": "CCO",
        },
        {
            "item_id": "b",
            "parent_smiles": "CCN",
            "morgan_similarity": 0.8,
            "assay_transfer_score": 1.0,
            "gold_label": 1,
            "canonical_smiles": "CCN",
        },
    ]
    profile = DirectProfile("existing", 1.0, 0.0, 0.0)
    assert (
        select_direct_contexts(rows, profile=profile, budget=1, assay_enabled=False)[0][
            0
        ]["item_id"]
        == "a"
    )
    assert (
        select_direct_contexts(rows, profile=profile, budget=1, assay_enabled=True)[0][
            0
        ]["item_id"]
        == "b"
    )
    rows[0]["assay_transfer_score"] = None
    assert (
        select_direct_contexts(rows, profile=profile, budget=1)[0][0]["item_id"] == "a"
    )


def test_gold_whole_query_bands_and_indirect_assay_gate() -> None:
    assert [gold_coverage_band(n, 100) for n in (40, 41, 69, 70)] == [
        "low",
        "middle",
        "middle",
        "high",
    ]
    rows = [dict(row, level="L2") for row in _candidates()]
    profile = IndirectProfile("gate", 1.0, 0.0, 0.0, 0.0, 0.0, ())
    assert (
        select_joint_indirect(rows, profile=profile, budget=1, assay_enabled=False)[0][
            0
        ]["item_id"]
        == "u1"
    )
    assert (
        select_joint_indirect(rows, profile=profile, budget=1, assay_enabled=True)[0][
            0
        ]["item_id"]
        == "u3"
    )


@pytest.mark.parametrize(
    "task,direct_total", [("carcinogens", 100), ("skin_reaction", 101)]
)
def test_gold_coverage_receipt_rejects_changed_cache(
    tmp_path, task, direct_total
) -> None:
    source = tmp_path / "RELEASE_INDEX.json"
    source.write_text("original")
    receipt = tmp_path / "coverage.json"
    write_json_atomic(
        receipt,
        {
            "schema_version": "gold_mixed_score_coverage.v1",
            "status": "complete",
            "task_id": task,
            "inputs": {
                "direct_index": {"path": str(source), "sha256": sha256_file(source)}
            },
            "queries": {
                "query": {
                    "direct_total": direct_total,
                    "direct_scored": direct_total,
                    "indirect_total": 100,
                    "indirect_scored": 20,
                    "candidate_count": direct_total + 100,
                    "scored_candidates": direct_total + 20,
                    "coverage_band": "middle",
                }
            },
        },
    )
    assert (
        load_coverage(receipt, task=task, queries=["query"], direct_index=source)[
            "queries"
        ]["query"]["coverage_band"]
        == "middle"
    )
    source.write_text("changed")
    with pytest.raises(ValueError, match="direct_index changed"):
        load_coverage(receipt, task=task, queries=["query"], direct_index=source)


def _candidates() -> list[dict]:
    return [
        {
            "item_id": "u1",
            "parent_id": "a",
            "parent_smiles": "CCO",
            "morgan_similarity": 0.9,
            "assay_transfer_score": 0.1,
            "semantic_bucket_id": "s1",
            "semantic_weight": 0.1,
            "canonical_smiles": "CCO",
        },
        {
            "item_id": "u2",
            "parent_id": "a",
            "parent_smiles": "CCO",
            "morgan_similarity": 0.9,
            "assay_transfer_score": 0.1,
            "semantic_bucket_id": "s1",
            "semantic_weight": 0.1,
            "canonical_smiles": "CCO",
        },
        {
            "item_id": "u3",
            "parent_id": "b",
            "parent_smiles": "c1ccccc1",
            "morgan_similarity": 0.8,
            "assay_transfer_score": 0.95,
            "semantic_bucket_id": "s2",
            "semantic_weight": 0.95,
            "canonical_smiles": "c1ccccc1",
        },
        {
            "item_id": "u4",
            "parent_id": "c",
            "parent_smiles": "CCN(C)C",
            "morgan_similarity": 0.7,
            "assay_transfer_score": 0.8,
            "semantic_bucket_id": "s3",
            "semantic_weight": 0.8,
            "canonical_smiles": "CCN(C)C",
        },
    ]


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
    profiles = policies.load_grid("sqrt48", kind="indirect")[0]
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


def test_tied_record_grid_and_parent_diversity() -> None:
    profiles = policies.load_grid("tied_record12", kind="indirect")[0]
    assert len(profiles) == len({profile.name for profile in profiles}) == 12
    gold_profiles = policies.load_grid("gold_skin_tied_record12", kind="indirect")[0]
    assert len(gold_profiles) == 12
    assert {profile.gated_assay for profile in gold_profiles} == {0.25, 0.5}
    ames_profiles = policies.load_grid("gold_ames_tied_record12", kind="indirect")[0]
    assert len(ames_profiles) == len({profile.name for profile in ames_profiles}) == 12
    assert {profile.gated_assay for profile in ames_profiles} == {1.0, 1.25}
    assert {profile.semantic_relevance for profile in ames_profiles} == {0.1, 0.25, 0.5}
    assert {profile.record_diversity for profile in ames_profiles} == {0.1, 0.25}
    assert all(
        (
            profile.molecular
            == profile.semantic_diversity
            == profile.level_diversity
            == profile.record_diversity
            for profile in profiles
        )
    )
    candidates = [
        {
            "item_id": uid,
            "parent_id": parent,
            "parent_smiles": smiles,
            "level": "L2",
            "morgan_similarity": similarity,
            "assay_transfer_score": None,
            "semantic_bucket_id": "same",
            "semantic_weight": 0.5,
            "canonical_smiles": smiles,
        }
        for uid, parent, smiles, similarity in (
            ("a", "p1", "CCO", 0.95),
            ("b", "p1", "CCO", 0.9),
            ("c", "p2", "CCN", 0.1),
        )
    ]
    profile = IndirectProfile("parent", 1, 0, 0, 0, 0, (), 2)
    selected, summary = select_joint_indirect(
        candidates, profile=profile, budget=2, diversity_function="sqrt"
    )
    assert {row["parent_id"] for row in selected} == {"p1", "p2"}
    assert summary["record_diversity"] == pytest.approx(1)
    assert summary["selected_parent_count"] == 2
    assert normalized_sqrt_diversity(("p1", "p1"), {"p1": 2, "p2": 1}, 2) < 1


def test_dili_geometric_relevance_and_tied_grid() -> None:
    profiles = policies.load_grid("dili_geomean_tied12", kind="indirect")[0]
    assert len(profiles) == 12
    assert all(
        (
            p.molecular
            == p.semantic_diversity
            == p.level_diversity
            == p.record_diversity
            for p in profiles
        )
    )
    expanded = policies.load_grid("dili_geomean_tied16", kind="indirect")[0]
    assert len(expanded) == 16
    assert {(p.gated_assay, p.semantic_relevance, p.molecular) for p in expanded} == {
        (ga, sr, diversity)
        for ga in (0.5, 0.75)
        for sr in (0.25, 0.35, 0.5, 0.65)
        for diversity in (0.1, 0.25)
    }
    assert all(
        (
            p.dili_geomean
            and p.molecular
            == p.semantic_diversity
            == p.level_diversity
            == p.record_diversity
            for p in expanded
        )
    )
    candidates = [
        {
            "item_id": uid,
            "parent_id": uid,
            "parent_smiles": smiles,
            "level": "L2",
            "morgan_similarity": 0.5,
            "assay_transfer_score": 0.5,
            "semantic_bucket_id": uid,
            "semantic_weight": weight,
            "dili_tool_score": tool,
            "canonical_smiles": smiles,
        }
        for uid, smiles, weight, tool in (
            ("a", "CCO", 1.0, 0.0),
            ("b", "CCN", 0.25, 1.0),
        )
    ]
    profile = IndirectProfile("dili", 1, 0, 1, 0, 0, (), 0, True)
    selected, summary = select_joint_indirect(
        candidates, profile=profile, budget=1, diversity_function="sqrt"
    )
    assert selected[0]["item_id"] == "b"
    assert summary["dili_geometric_relevance"] == pytest.approx(0.5)
    assert summary["dili_tool_score"] == pytest.approx(1)
    with pytest.raises(ValueError, match="every semantic weight"):
        select_joint_indirect(
            [{**candidates[0], "semantic_weight": None}],
            profile=profile,
            budget=1,
            diversity_function="sqrt",
        )


def test_sqrt_indirect_keeps_linear_morgan_coverage() -> None:
    candidates = [
        {
            "item_id": uid,
            "parent_smiles": smiles,
            "level": level,
            "morgan_similarity": similarity,
            "assay_transfer_score": None,
            "semantic_bucket_id": bucket,
            "semantic_weight": None,
            "canonical_smiles": smiles,
        }
        for uid, smiles, level, similarity, bucket in (
            ("u1", "CCO", "L2", 0.9, "a"),
            ("u2", "CCN", "L2", 0.8, "a"),
            ("u3", "c1ccccc1", "L3", 0.7, "b"),
        )
    ]
    selected, summary = select_joint_indirect(
        candidates,
        profile=IndirectProfile("test", 1, 0.25, 0.5, 0.25, 0.25, ()),
        budget=2,
        diversity_function="sqrt",
    )
    assert len({row["item_id"] for row in selected}) == 2
    assert summary["semantic_relevance_lambda_effective"] == 0
    assert all(
        (
            0 <= summary[key] <= 1
            for key in (
                "molecular_coverage_normalized",
                "semantic_diversity",
                "level_diversity",
            )
        )
    )
    universe = set().union(
        *(set(_fingerprint(row["parent_smiles"]).GetOnBits()) for row in candidates)
    )
    covered = set().union(
        *(set(_fingerprint(row["parent_smiles"]).GetOnBits()) for row in selected)
    )
    assert summary["molecular_coverage_normalized"] == pytest.approx(
        len(covered) / len(universe)
    )


def test_direct_selector_uses_cards_bit_coverage_and_log_labels() -> None:
    candidates = [
        {
            "item_id": "c1",
            "parent_id": "p1",
            "parent_smiles": "CCO",
            "morgan_similarity": 0.9,
            "assay_transfer_score": 0.9,
            "gold_label": 0,
            "canonical_smiles": "CCO",
        },
        {
            "item_id": "c2",
            "parent_id": "p1",
            "parent_smiles": "CCO",
            "morgan_similarity": 0.89,
            "assay_transfer_score": 0.9,
            "gold_label": 0,
            "canonical_smiles": "CCO",
        },
        {
            "item_id": "c3",
            "parent_id": "p2",
            "parent_smiles": "c1ccccc1",
            "morgan_similarity": 0.7,
            "assay_transfer_score": 0.8,
            "gold_label": 1,
            "canonical_smiles": "c1ccccc1",
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
            "item_id": "u1",
            "parent_id": "p1",
            "parent_smiles": "CCO",
            "level": "L2",
            "morgan_similarity": 0.9,
            "assay_transfer_score": 0.8,
            "semantic_bucket_id": "s1",
            "semantic_weight": 0.5,
            "canonical_smiles": "CCO",
        },
        {
            "item_id": "u2",
            "parent_id": "p2",
            "parent_smiles": "CCN",
            "level": "L2",
            "morgan_similarity": 0.8,
            "assay_transfer_score": None,
            "semantic_bucket_id": "s1",
            "semantic_weight": 0.4,
            "canonical_smiles": "CCN",
        },
        {
            "item_id": "u3",
            "parent_id": "p3",
            "parent_smiles": "c1ccccc1",
            "level": "L3",
            "morgan_similarity": 0.3,
            "assay_transfer_score": None,
            "semantic_bucket_id": "s2",
            "semantic_weight": 0.3,
            "canonical_smiles": "c1ccccc1",
        },
    ]
    profile = IndirectProfile("test", 1, 0, 0, 0, 0, ())
    selected, summary = select_joint_indirect(candidates, profile=profile, budget=2)
    assert [row["item_id"] for row in selected] == ["u1", "u2"]
    assert summary["level_counts"] == {"L2": 2}
    assert 0 <= summary["semantic_diversity"] <= 1
    assert 0 <= summary["level_diversity"] <= 1


def test_provisional_weight_does_not_claim_a_reviewed_family() -> None:
    row = {
        "item_id": "u",
        "parent_id": "p",
        "parent_smiles": "CCO",
        "level": "L2",
        "morgan_similarity": 0.5,
        "assay_transfer_score": None,
        "semantic_bucket_id": None,
        "semantic_weight": 0.4,
        "semantic_weight_source": "provisional_task_level_mean",
        "canonical_smiles": "CCO",
    }
    profile = IndirectProfile("provisional", 0, 0, 1, 1, 0, ())
    selected, summary = select_joint_indirect([row], profile=profile, budget=1)
    assert selected[0]["semantic_bucket_id"] is None
    assert summary["semantic_relevance_available"] is True
    with pytest.raises(ValueError, match="reviewed semantic bucket"):
        select_joint_indirect(
            [{**row, "semantic_weight_source": None}], profile=profile, budget=1
        )


def test_direct_grid_adds_only_the_twelve_low_assay_followups() -> None:
    profiles = policies.load_grid("direct", kind="direct")[0]
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
