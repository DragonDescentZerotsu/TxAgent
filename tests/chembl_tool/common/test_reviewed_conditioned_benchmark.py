import json
from pathlib import Path

import pytest

from tools.chembl_tool.common.starling.external_condition import (
    AtomRule,
    propose_pattern_condition,
)
from tools.chembl_tool.common.starling.reviewed_conditioned_benchmark import (
    ConditionedBenchmarkConfig,
    aggregate_reviewed_votes,
    build_reviewed_conditioned_benchmark,
    candidate_group_support,
    validate_review_ledger,
)


def _row(record_id: str, label: int, *, status: str = "accepted") -> dict:
    return {
        "source_record_id": record_id,
        "source_payload_sha256": "a" * 64,
        "review_status": status,
        "reviewer": "reviewer",
        "review_reason": "literal_context_verified",
        "drug": "CCO",
        "molecule_identity_key": "PARENT",
        "bemis_murcko_scaffold": "SCAFFOLD",
        "condition_group": "disease=asthma",
        "condition_atoms": ["disease=asthma"],
        "Y": label,
        "label_method": "direct",
    }


def test_pattern_match_only_proposes_review_candidate() -> None:
    proposal = propose_pattern_condition(
        "studied in the fasted state",
        rules=(AtomRule.make("meal_state", "fasted", r"fasted|fasting"),),
    )
    assert proposal.status == "needs_review"
    assert proposal.signature == "meal_state=fasted"


def test_review_ledger_requires_terminal_unique_verdicts() -> None:
    with pytest.raises(ValueError, match="Non-terminal"):
        validate_review_ledger([{**_row("one", 1), "review_status": "pending"}])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_review_ledger([_row("one", 1), _row("one", 1)])


def test_rejected_review_row_never_votes() -> None:
    audits, votes = validate_review_ledger([_row("one", 1), _row("two", 0, status="rejected")])
    assert len(audits) == 2
    assert sum(map(len, votes.values())) == 1


def test_empty_acyclic_scaffold_is_valid() -> None:
    row = {**_row("acyclic", 1), "bemis_murcko_scaffold": ""}
    _, votes = validate_review_ledger([row])
    assert sum(map(len, votes.values())) == 1


def test_accepted_signature_must_exactly_match_atoms() -> None:
    row = {**_row("bad-signature", 1), "condition_group": "disease=other"}
    with pytest.raises(ValueError, match="non-canonical condition signature"):
        validate_review_ledger([row])


def test_external_votes_use_sixty_percent_and_reject_ties() -> None:
    _, votes = validate_review_ledger(
        [_row("one", 1), _row("two", 1), _row("three", 1), _row("four", 0), _row("five", 0)]
    )
    accepted, rejected = aggregate_reviewed_votes(votes, agreement_threshold=0.60)
    assert not rejected
    assert accepted[0]["Y"] == 1
    assert accepted[0]["agreement_fraction"] == 0.60

    _, tied_votes = validate_review_ledger([_row("a", 1), _row("b", 0)])
    accepted, rejected = aggregate_reviewed_votes(tied_votes, agreement_threshold=0.60)
    assert not accepted
    assert rejected[0]["drop_reason"] == "parent_condition_label_tie"


def test_task_semantic_group_exclusion_precedes_split_allocation() -> None:
    rows = [
        {
            **_row(f"row-{index}", 1),
            "molecule_identity_key": f"PARENT-{index}",
            "bemis_murcko_scaffold": f"SCAFFOLD-{index}",
        }
        for index in range(3)
    ]
    candidates, audit = candidate_group_support(
        rows,
        minimum_parents=3,
        minimum_scaffolds=3,
        excluded_groups={"disease=asthma"},
    )
    assert not candidates
    assert audit["disease=asthma"]["preallocation_exclusion_reasons"] == [
        "task_semantic_group_exclusion"
    ]


def test_task_condition_group_allowlist_precedes_split_allocation() -> None:
    rows = [
        {
            **_row(f"row-{index}", 1),
            "molecule_identity_key": f"PARENT-{index}",
            "bemis_murcko_scaffold": f"SCAFFOLD-{index}",
        }
        for index in range(3)
    ]
    candidates, audit = candidate_group_support(
        rows,
        minimum_parents=3,
        minimum_scaffolds=3,
        allowed_groups={"prandial_state=fasted"},
    )
    assert not candidates
    assert audit["disease=asthma"]["preallocation_exclusion_reasons"] == [
        "not_in_task_condition_group_allowlist"
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_builder_preserves_frozen_rows_and_requires_three_way_group_coverage(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    for split, parent, scaffold in (
        ("train", "FROZEN_T", "FROZEN_ST"),
        ("valid", "FROZEN_V", "FROZEN_SV"),
        ("test", "FROZEN_E", "FROZEN_SE"),
    ):
        full = {
            "drug": f"C{parent}",
            "Y": int(split != "train"),
            "molecule_identity_key": parent,
            "bemis_murcko_scaffold": scaffold,
        }
        _write_jsonl(frozen / f"{split}_molecule_labels.jsonl", [full])
        _write_jsonl(frozen / f"{split}.jsonl", [{"drug": full["drug"], "Y": full["Y"]}])

    source = tmp_path / "source.jsonl"
    source.write_text("source\n", encoding="utf-8")
    reviews = []
    for index in range(3):
        reviews.append(
            {
                **_row(f"external-{index}", index % 2),
                "drug": f"N{index}",
                "molecule_identity_key": f"NOVEL_{index}",
                "bemis_murcko_scaffold": f"NOVEL_S{index}",
            }
        )
    output = tmp_path / "output"
    summary = build_reviewed_conditioned_benchmark(
        config=ConditionedBenchmarkConfig(
            task_name="Synthetic",
            lineage="synthetic_context_v1",
            protocol_version="synthetic.v1",
            frozen_root=frozen,
            output_root=output,
            source_artifacts=(source,),
            row_id_prefix="SYN",
            frozen_lineage="frozen.v1",
        ),
        review_rows=reviews,
    )

    assert summary["counts"]["n_frozen_no_reported_rows"] == 3
    assert summary["counts"]["n_added_external_condition_rows"] == 3
    disease = next(
        row for row in summary["group_distribution"] if row["condition_group"] == "disease=asthma"
    )
    assert (disease["n_train"], disease["n_valid"], disease["n_test"]) == (1, 1, 1)
    assert summary["pairwise_identity_overlap"] == {
        "train_valid": 0,
        "train_test": 0,
        "valid_test": 0,
    }
    for split in ("train", "valid", "test"):
        rows = [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
        assert sum(row["condition_group"] == "no_reported_external_condition" for row in rows) == 1
        assert sum(row["condition_group"] == "disease=asthma" for row in rows) == 1


def test_selected_build_publishes_only_allowlisted_audit_rows(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    for split, parent in (("train", "FT"), ("valid", "FV"), ("test", "FE")):
        full = {
            "drug": f"C{parent}",
            "Y": 1,
            "molecule_identity_key": parent,
            "bemis_murcko_scaffold": f"S{parent}",
        }
        _write_jsonl(frozen / f"{split}_molecule_labels.jsonl", [full])
        _write_jsonl(frozen / f"{split}.jsonl", [{"drug": full["drug"], "Y": 1}])

    selected = []
    for index in range(3):
        selected.append(
            {
                **_row(f"selected-{index}", 1),
                "molecule_identity_key": f"P{index}",
                "bemis_murcko_scaffold": f"SP{index}",
            }
        )
    rejected_other = {
        **_row("other-rejected", 0, status="rejected"),
        "condition_group": "disease=other",
        "condition_atoms": ["disease=other"],
        "proposed_condition_group": "disease=other",
    }
    source = tmp_path / "source.jsonl"
    source.write_text("source\n", encoding="utf-8")
    output = tmp_path / "output"
    summary = build_reviewed_conditioned_benchmark(
        config=ConditionedBenchmarkConfig(
            task_name="Synthetic",
            lineage="selected.v1",
            protocol_version="selected.v1",
            frozen_root=frozen,
            output_root=output,
            source_artifacts=(source,),
            row_id_prefix="SEL",
            frozen_lineage="frozen.v1",
            allowed_condition_groups=("disease=asthma",),
            publish_allowed_group_audit_only=True,
        ),
        review_rows=[*selected, rejected_other],
    )

    published = [
        json.loads(line)
        for line in (output / "source_condition_review.jsonl").read_text().splitlines()
    ]
    assert len(published) == 3
    assert {row["condition_group"] for row in published} == {"disease=asthma"}
    assert summary["counts"]["n_full_review_ledger_records"] == 4
    assert summary["counts"]["n_published_group_review_records"] == 3
