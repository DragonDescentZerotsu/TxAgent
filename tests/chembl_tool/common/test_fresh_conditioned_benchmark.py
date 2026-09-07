"""Synthetic-only checks for fresh staging and condition/scaffold allocation."""

import pytest

from tools.chembl_tool.common.json_utils import read_jsonl
from tools.chembl_tool.common.starling.build_record_supported_benchmark import (
    allocate_scaffold_groups,
    resolve_conditioned_eval_size,
)
from tools.chembl_tool.common.starling.conditioned_benchmark import (
    NO_REPORTED_CONDITION,
)
from tools.chembl_tool.common.starling.fresh_conditioned_benchmark import (
    build_fresh_conditioned_benchmark,
)
from tools.chembl_tool.common.starling.reviewed_conditioned_benchmark import (
    validate_split_integrity,
)


def _vote(parent, group, index=0, label=1, scaffold=None):
    return {
        "source_record_id": f"{parent}:{group}:{index}",
        "drug": parent,
        "molecule_identity_key": parent,
        "scaffold": parent if scaffold is None else scaffold,
        "Y": label,
        "condition_group": group,
        "condition_atoms": [] if group == NO_REPORTED_CONDITION else [group],
        "pmid": str(100000 + index),
        "label_method": "source_binary",
        "reviewer": "task-rule-v1",
        "raw_value": f"raw-{index}",
        "condition_text": f"condition-{index}",
        "extra_provenance": {"index": index, "text": "full source " * 100},
    }


def _cohort():
    return [
        _vote(f"p{i:02d}", group)
        for i in range(30)
        for group in (NO_REPORTED_CONDITION, "with_s9")
    ]


def test_conditioned_and_unconditioned_agreement_boundaries(tmp_path):
    votes = [row for row in _cohort() if row["drug"] not in {"p00", "p01", "p02"}]
    for parent, positive, total in (("p00", 2, 3), ("p01", 3, 5), ("p02", 7, 10)):
        for group in (NO_REPORTED_CONDITION, "with_s9"):
            votes.extend(_vote(parent, group, i, int(i < positive)) for i in range(total))
    summary = build_fresh_conditioned_benchmark(
        task="ames", record_votes=votes, output_root=tmp_path
    )
    accepted = {
        (row["drug"], row["condition_group"]): row
        for row in read_jsonl(tmp_path / "accepted_parent_conditions.jsonl")
    }
    rejected = read_jsonl(tmp_path / "rejected_parent_conditions.jsonl")
    assert {(r["drug"], r["condition_group"]) for r in rejected} == {
        ("p00", NO_REPORTED_CONDITION), ("p01", NO_REPORTED_CONDITION)
    }
    assert all(r["agreement_threshold"] == 0.7 for r in rejected)
    assert all(r["drop_reason"] == "parent_condition_agreement_below_threshold" for r in rejected)
    for parent in ("p00", "p01", "p02"):
        assert accepted[parent, "with_s9"]["agreement_threshold"] == 0.6
    assert accepted["p02", NO_REPORTED_CONDITION]["agreement_threshold"] == 0.7
    assert accepted["p02", NO_REPORTED_CONDITION]["label_decision"] == "accepted_record_majority_70"
    assert summary["label_policy"]["agreement_threshold_by_condition_scope"] == {
        "none_reported": 0.7, "external": 0.6
    }


@pytest.mark.parametrize("mutation", ["threshold", "label", "missing_row"])
def test_independent_ames_vote_validator_rejects_corrupted_benchmark(tmp_path, mutation):
    from tools.chembl_tool.tasks.ames.validate_retrieval import validate_benchmark_votes

    votes = [
        {**r, "bemis_murcko_scaffold": r["scaffold"]} for r in _cohort()
    ]
    build_fresh_conditioned_benchmark(task="ames", record_votes=votes, output_root=tmp_path)
    splits = {
        s: read_jsonl(tmp_path / f"{s}_molecule_condition_labels.jsonl")
        for s in ("train", "valid", "test")
    }
    assert validate_benchmark_votes(splits, votes)["rows"] == 60
    if mutation == "missing_row":
        splits["train"].pop()
    elif mutation == "label":
        splits["train"][0]["Y"] = 0
    else:
        splits["train"][0]["agreement_threshold"] = 0.5
    with pytest.raises(AssertionError):
        validate_benchmark_votes(splits, votes)


def test_empty_scaffold_audit_distinguishes_conditions_from_parents(tmp_path):
    votes = _cohort()
    for row in votes:
        if row["drug"] in {"p00", "p01"}:
            row["scaffold"] = ""
    summary = build_fresh_conditioned_benchmark(
        task="ames", record_votes=votes, output_root=tmp_path
    )
    assert summary["optimization"]["n_empty_scaffold_rows"] == 4
    assert summary["optimization"]["n_empty_scaffold_parents"] == 2


def test_fresh_builder_full_provenance_agreement_coverage_and_stability(tmp_path):
    votes = _cohort()
    # Exactly 70%, >50 ids/PMIDs and >20 raw/condition strings; no truncation.
    votes = [
        row
        for row in votes
        if not (row["drug"] == "p00" and row["condition_group"] == "with_s9")
    ]
    votes += [_vote("p00", "with_s9", i, int(i < 42)) for i in range(60)]
    votes += [_vote("tie", "with_s9", i, i) for i in range(2)]
    votes += [_vote("low", "with_s9", i, int(i < 4)) for i in range(7)]
    # Three parents, but only two nonempty scaffolds: reject the whole group.
    votes += [
        _vote(f"rare{i}", "rare", scaffold="" if i == 2 else f"r{i}") for i in range(3)
    ]
    # Multiple parents on one scaffold must move together.
    for row in votes:
        if row["drug"] == "p01":
            row["scaffold"] = "p00"
    first, second = tmp_path / "first", tmp_path / "second"
    summary = build_fresh_conditioned_benchmark(
        task="ames", record_votes=votes, output_root=first
    )
    build_fresh_conditioned_benchmark(
        task="ames", record_votes=list(reversed(votes)), output_root=second
    )
    assert summary["n_accepted_parent_conditions"] == 60
    assert summary["n_rejected_parent_conditions"] == 5
    assert summary["nominal_target_eval_rows"] == summary["target_eval_rows"] == 6
    assert summary["target_expansion_reason"] is None
    assert summary["group_gate"]["rare"]["n_nonempty_scaffolds"] == 2
    by_split = {
        s: read_jsonl(first / f"{s}_molecule_condition_labels.jsonl")
        for s in ("train", "valid", "test")
    }
    validate_split_integrity(by_split)
    assert [len(by_split[s]) for s in ("train", "valid", "test")] == [48, 6, 6]
    for split, rows in by_split.items():
        assert {row["condition_group"] for row in rows} == {
            NO_REPORTED_CONDITION,
            "with_s9",
        }
        for row in rows:
            if row["condition_group"] == NO_REPORTED_CONDITION:
                assert row["condition_scope"] == "none_reported"
        minimal = read_jsonl(first / f"{split}.jsonl")
        assert "source_votes" not in minimal[0]
    all_rows = [row for rows in by_split.values() for row in rows]
    row = next(
        row
        for row in all_rows
        if row["drug"] == "p00" and row["condition_group"] == "with_s9"
    )
    assert row["agreement_fraction"] == 0.7
    assert row["agreement_threshold"] == 0.6
    assert row["label_decision"] == "accepted_record_majority_60"
    assert row["vote_unit"] == "task_accepted_source_record"
    assert row["reviewers"] == ["task-rule-v1"]
    for key in (
        "source_votes",
        "source_record_ids",
        "source_pmids",
        "raw_value_examples",
        "condition_text_examples",
    ):
        assert len(row[key]) == 60
    assert row["source_votes"] == sorted(
        (
            vote
            for vote in votes
            if vote["drug"] == "p00" and vote["condition_group"] == "with_s9"
        ),
        key=lambda vote: vote["source_record_id"],
    )
    rejected = read_jsonl(first / "rejected_parent_conditions.jsonl")
    assert {row["drop_reason"] for row in rejected} >= {
        "parent_condition_label_tie",
        "parent_condition_agreement_below_threshold",
    }
    assert all(row["source_votes"] for row in rejected)
    assert (
        read_jsonl(first / "heldout_molecule_condition_labels.jsonl")
        == by_split["valid"] + by_split["test"]
    )
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    # IDs describe the parent-condition unit, independent of source order/label.
    changed = [{**vote, "Y": 1 - vote["Y"]} for vote in votes]
    build_fresh_conditioned_benchmark(
        task="ames", record_votes=changed, output_root=second
    )
    assert {row["benchmark_row_id"] for row in all_rows} == {
        row["benchmark_row_id"]
        for row in read_jsonl(second / "accepted_parent_conditions.jsonl")
    }


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate", "Duplicate source_record_id"),
        ("scaffold", "Inconsistent drug/scaffold"),
        ("atoms", "non-canonical condition signature"),
        ("label", "Invalid binary label"),
    ],
)
def test_fresh_builder_rejects_inconsistent_inputs_before_writing(
    tmp_path, mutation, match
):
    votes = _cohort()
    if mutation == "duplicate":
        votes.append(dict(votes[0]))
    elif mutation == "scaffold":
        votes[1]["scaffold"] = "different"
    elif mutation == "atoms":
        votes[3]["condition_atoms"] = ["different"]
    else:
        votes[0]["Y"] = 2
    with pytest.raises(ValueError, match=match):
        build_fresh_conditioned_benchmark(
            task="ames", record_votes=votes, output_root=tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


def test_fresh_builder_infeasible_size_writes_nothing(tmp_path):
    with pytest.raises(ValueError, match="No scaffold group"):
        build_fresh_conditioned_benchmark(
            task="ames",
            record_votes=_cohort(),
            output_root=tmp_path / "out",
            target_size=1,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "group,atoms",
    [
        ("alias", ["a"]),
        ("b+a", ["a", "b"]),
        ("a", ["a", "a"]),
        ("external", []),
        ("a", [""]),
        (" a", [" a"]),
        ("a+b", ["a+b"]),
        ("a", [None]),
        ("a", [{"atom": "a"}]),
        (f"a+{NO_REPORTED_CONDITION}", ["a", NO_REPORTED_CONDITION]),
        (NO_REPORTED_CONDITION, ["a"]),
    ],
)
def test_noncanonical_conditions_fail_before_writing(tmp_path, group, atoms):
    votes = [{**_vote("parent", group), "condition_atoms": atoms}]
    with pytest.raises(ValueError, match="condition signature|Unreported condition"):
        build_fresh_conditioned_benchmark(
            task="ames", record_votes=votes, output_root=tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


def test_canonical_signature_accepts_atom_order_and_preserves_original_votes(tmp_path):
    votes = [
        {**_vote(f"p{i}", "a+b", j), "condition_atoms": atoms}
        for i in range(10)
        for j, atoms in enumerate((["b", "a"], ["a", "b"]))
    ]
    build_fresh_conditioned_benchmark(
        task="ames", record_votes=votes, output_root=tmp_path
    )
    rows = read_jsonl(tmp_path / "accepted_parent_conditions.jsonl")
    assert len(rows) == 10
    for row in rows:
        assert row["condition_atoms"] == ["a", "b"]
        assert row["source_record_count"] == 2
        assert row["source_votes"] == [
            vote for vote in votes if vote["drug"] == row["drug"]
        ]
    assert read_jsonl(tmp_path / "accepted_record_votes.jsonl") == votes


def test_fresh_staging_can_bootstrap_random_without_a_manifest(tmp_path, monkeypatch):
    from tools.chembl_tool.common.starling import (
        build_conditioned_random_split as random_builder,
    )

    build_fresh_conditioned_benchmark(
        task="ames", record_votes=_cohort(), output_root=tmp_path / "scaffold"
    )
    monkeypatch.setattr(
        random_builder, "task_root", lambda task, scheme: tmp_path / scheme
    )
    result = random_builder.build_task("ames")
    assert result["same_molecule_condition_rows_and_labels"]
    assert result["optimizer"]["record_support_objective_enabled"]
    assert not any(result["pairwise_identity_overlap"].values())
    assert not any(result["missing_conditions_by_split"].values())
    assert not (tmp_path / "manifest.json").exists()


def test_fresh_builder_expands_only_automatic_size_without_dropping_conditions(tmp_path):
    # Rare occurs on three 3-row scaffolds. A 2/2 split is possible using common
    # singletons, but covering rare in all splits requires a 3/3 split instead.
    votes = [
        _vote(f"{scaffold}-{i}", "rare" if i == 0 else "common", scaffold=scaffold)
        for scaffold in ("A", "B", "C") for i in range(3)
    ]
    votes += [_vote(f"common{i}", "common") for i in range(10)]
    votes.append(_vote("empty", "common", scaffold=""))
    root = tmp_path / "auto"
    summary = build_fresh_conditioned_benchmark(task="ames", record_votes=votes, output_root=root)
    assert summary["nominal_target_eval_rows"] == 2
    assert summary["target_eval_rows"] == 3
    assert summary["target_size_policy"] == "minimum_feasible_equal_eval_size"
    assert summary["target_expansion_reason"] == (
        "nominal_size_infeasible_with_whole_scaffold_three_way_condition_coverage"
    )
    splits = {s: read_jsonl(root / f"{s}_molecule_condition_labels.jsonl") for s in ("train", "valid", "test")}
    validate_split_integrity(splits)
    assert [len(rows) for rows in splits.values()] == [14, 3, 3]
    assert all({row["condition_group"] for row in rows} == {"rare", "common"} for rows in splits.values())
    assert any(row["drug"] == "empty" for row in splits["train"])
    assert sorted((row["drug"], row["condition_group"], row["Y"]) for rows in splits.values() for row in rows) == sorted(
        (row["drug"], row["condition_group"], row["Y"]) for row in votes
    )
    with pytest.raises(RuntimeError, match="Scaffold allocation failed"):
        build_fresh_conditioned_benchmark(task="ames", record_votes=votes, output_root=tmp_path / "strict", target_size=2)
    assert not (tmp_path / "strict").exists()
    explicit = build_fresh_conditioned_benchmark(task="ames", record_votes=votes, output_root=tmp_path / "explicit", target_size=3)
    assert explicit["target_size_policy"] == "explicit_strict"
    assert explicit["nominal_target_eval_rows"] == explicit["target_eval_rows"] == 3
    assert explicit["target_expansion_reason"] is None


def test_size_resolution_rejects_a_true_three_way_coverage_contradiction():
    # Every triple of four scaffolds must have all three colors. That forces
    # all four scaffolds to have different colors, impossible with three splits.
    rows = [
        _allocation_row(scaffold, omitted)
        for omitted in "ABCD" for scaffold in "ABCD" if scaffold != omitted
    ]
    with pytest.raises(RuntimeError, match="Scaffold allocation failed"):
        resolve_conditioned_eval_size(rows, nominal_target_size=1, required_condition_groups=set("ABCD"))


def _allocation_row(scaffold, condition, records=1):
    return {
        "bemis_murcko_scaffold": scaffold,
        "condition_group": condition,
        "Y": 1,
        "source_record_count": records,
    }


def test_required_conditions_override_quality_and_cover_train():
    rows = [_allocation_row(f"rare{i}", "rare") for i in range(3)]
    rows += [_allocation_row(f"common{i}", "common", records=10) for i in range(9)]
    default, audit = allocate_scaffold_groups(rows, target_size=2)
    explicit, explicit_audit = allocate_scaffold_groups(
        rows, target_size=2, required_condition_groups=None
    )
    assert (default, audit) == (explicit, explicit_audit)
    assert audit["minimum_singletons_in_heldout"] == 0
    assignment, constrained = allocate_scaffold_groups(
        rows, target_size=2, required_condition_groups={"rare", "common"}
    )
    assert constrained["minimum_singletons_in_heldout"] == 2
    for condition in ("rare", "common"):
        assert {
            assignment[row["bemis_murcko_scaffold"]]
            for row in rows
            if row["condition_group"] == condition
        } == {"train", "valid", "test"}


def test_coverage_counts_all_rows_within_existing_scaffold_grains():
    rows = [
        _allocation_row(scaffold, condition)
        for scaffold in ("A", "B", "C")
        for condition in ("x", "y")
    ]
    assignment, _ = allocate_scaffold_groups(
        rows, target_size=2, required_condition_groups={"x", "y"}
    )
    assert set(assignment.values()) == {"train", "valid", "test"}
    # Three rows on only two scaffolds cannot cover three splits.
    rows = [
        _allocation_row("A", "x"),
        _allocation_row("A", "x"),
        _allocation_row("B", "x"),
        _allocation_row("C", "y"),
        _allocation_row("D", "y"),
    ]
    with pytest.raises(RuntimeError, match="Scaffold allocation failed"):
        allocate_scaffold_groups(rows, target_size=2, required_condition_groups={"x"})
