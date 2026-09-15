from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.chembl_tool.common.starling.reviewed_conditioned_benchmark import (
    aggregate_reviewed_votes,
)
from tools.chembl_tool.common.source_family_purity import audit_exact_voter_membership
from tools.chembl_tool.paper_experiments.build_conditioned_source_family_purity import (
    PURITY_VERSION,
    PuritySpec,
    build_overlay,
)
from tools.chembl_tool.tasks.bioavailability_ma.source_family_purity import (
    DIRECT_GROUP as BIO_DIRECT_GROUP,
    direct_like_bioavailability_reason,
)
from tools.chembl_tool.tasks.skin_reaction.source_family_purity import (
    AOP_GROUP as SKIN_AOP_GROUP,
    DIRECT_GROUP as SKIN_DIRECT_GROUP,
    EXCLUDED_GROUP as SKIN_EXCLUDED_GROUP,
    NEAR_DIRECT_GROUP as SKIN_NEAR_DIRECT_GROUP,
    load_voter_source_keys,
    source_record_key_from_id,
    upstream_source_key,
    vote_pure_family_move as skin_vote_pure_family_move,
)
from tools.chembl_tool.tasks.skin_reaction.canonical_starling_source import (
    PartitionDecision,
    normalized_direct_label,
)


def test_overlay_changes_only_family_and_appends_audit(tmp_path: Path):
    source = tmp_path / "source.parquet"
    rows = [
        {
            "group_id": "Fg.gut_wall_efflux_intestinal_metabolism",
            "canonical_bioavailability_evidence_scope": "relative_or_conditional",
            "canonical_endpoint_name": "bioavailability",
            "canonical_measurement_text": "F = 44.8%",
            "canonical_unit_text": "%",
            "support_text": "The overall oral bioavailability F was 44.8%.",
            "source_id": "fg",
            "canonical_record_id": "r1",
            "retrieval_eligible": True,
            "parent_smiles": "CCO",
        },
        {
            "group_id": "Fg.gut_wall_efflux_intestinal_metabolism",
            "canonical_bioavailability_evidence_scope": "mechanistic",
            "canonical_endpoint_name": "intestinal uptake",
            "canonical_measurement_text": "increased",
            "canonical_unit_text": "",
            "support_text": "CNT1 increased intestinal uptake.",
            "source_id": "fg",
            "canonical_record_id": "r2",
            "retrieval_eligible": True,
            "parent_smiles": "CCN",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), source)
    spec = PuritySpec(
        "bioavailability_ma",
        source,
        BIO_DIRECT_GROUP,
        direct_like_bioavailability_reason,
    )

    manifest = build_overlay(spec, tmp_path / "out", batch_size=1)
    output = pq.read_table(tmp_path / "out" / "records.parquet").to_pylist()
    audit = pq.read_table(tmp_path / "out" / "moved_records.parquet").to_pylist()

    assert [row["group_id"] for row in output] == [
        BIO_DIRECT_GROUP,
        "Fg.gut_wall_efflux_intestinal_metabolism",
    ]
    assert output[0]["source_family_purity_version"] == PURITY_VERSION
    assert output[1]["source_family_purity_version"] is None
    assert output[0]["support_text"] == rows[0]["support_text"]
    assert output[1]["support_text"] == rows[1]["support_text"]
    assert len(audit) == 1
    assert audit[0]["source_record_id"] == "r1"
    assert manifest["n_moved_rows"] == 1
    assert manifest["gold_labels_modified"] is False


def test_gold_sensitivity_vote_uses_frozen_record_agreement_contract():
    def row(label: int, index: int) -> dict:
        return {
            "drug": "CCO",
            "Y": label,
            "bemis_murcko_scaffold": "",
            "condition_atoms": ["fed"],
            "source_record_id": f"r{index}",
            "label_method": "test",
            "reviewer": "test",
        }

    rows = [row(label, index) for index, label in enumerate([1, 1, 1, 0])]
    accepted, rejected = aggregate_reviewed_votes(
        {("parent", "fed"): rows}, agreement_threshold=0.70
    )
    assert accepted[0]["Y"] == 1
    assert rejected == []
    accepted, rejected = aggregate_reviewed_votes(
        {("parent", "fed"): [*rows, row(0, 4)]}, agreement_threshold=0.70
    )
    assert accepted == []
    assert rejected[0]["drop_reason"] == "parent_condition_agreement_below_threshold"
    assert normalized_direct_label("positive") == "positive"
    assert normalized_direct_label("negative") == "negative"
    assert normalized_direct_label("inconclusive") == "inconclusive"


def test_skin_l1_uses_exact_source_voter_membership():
    voter = {
        "group_id": SKIN_DIRECT_GROUP,
        "source_id": "direct_skin_reaction",
        "source_row_number": 3,
    }
    direct = PartitionDecision("direct", "scoped_direct_outcome")
    partitions = {
        ("direct_skin_reaction", 2): direct,
        ("direct_skin_reaction", 3): direct,
    }
    assert upstream_source_key(voter) == ("direct_skin_reaction", 2)
    assert (
        skin_vote_pure_family_move(
            voter, {("direct_skin_reaction", 2)}, partitions
        ).reason
        == ""
    )

    nonvoter = {**voter, "source_row_number": 4}
    moved = skin_vote_pure_family_move(
        nonvoter, {("direct_skin_reaction", 2)}, partitions
    )
    assert moved.new_group == SKIN_NEAR_DIRECT_GROUP
    assert moved.reason.startswith("canonical_direct_nonvoter_to_l2:")


def test_skin_voter_ledger_requires_a_resolved_parent_identity(tmp_path: Path):
    source = tmp_path / "direct.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "reaction_type": "sensitization",
                    "outcome_label": "positive",
                    "assay_or_test": "LLNA",
                    "support_text": "experimental positive LLNA",
                    "SMILES": "CCO",
                },
                {
                    "reaction_type": "sensitization",
                    "outcome_label": "positive",
                    "assay_or_test": "LLNA",
                    "support_text": "experimental positive LLNA",
                    "SMILES": "not-a-smiles",
                },
            ]
        ),
        source,
    )
    voters = load_voter_source_keys(
        source_path=source,
        condition_review=tmp_path / "absent-review.jsonl",
    )
    assert voters == {("direct_skin_reaction", 0)}


def test_skin_direct_like_aop_nonvoter_goes_to_l2_not_l1():
    row = {
        "group_id": SKIN_AOP_GROUP,
        "source_id": "sensitization_aop",
        "source_row_number": 1,
        "canonical_assay_type": "local lymph node assay (LLNA)",
        "result_label": "positive",
    }
    partitions = {
        ("sensitization_aop", 0): PartitionDecision(
            "direct", "aop_adverse_outcome_moved_to_direct"
        )
    }
    moved = skin_vote_pure_family_move(row, set(), partitions)
    assert moved.new_group == SKIN_NEAR_DIRECT_GROUP
    assert moved.reason.startswith("canonical_direct_nonvoter_to_l2:")


def test_skin_condition_voter_uses_stable_partition_key_and_is_promoted_to_l1():
    key = source_record_key_from_id("sensitization_aop:22321")
    assert key == ("sensitization_aop", 22321)
    row = {
        "group_id": SKIN_AOP_GROUP,
        "source_id": "sensitization_aop",
        "source_row_number": 22322,
    }
    partitions = {
        key: PartitionDecision("direct", "aop_adverse_outcome_moved_to_direct")
    }
    moved = skin_vote_pure_family_move(row, {key}, partitions)
    assert moved.new_group == SKIN_DIRECT_GROUP
    assert "stable_source_key" in moved.reason


def test_skin_prediction_and_defined_approach_route_by_evidence_object():
    cases = (
        (
            "prediction_only",
            {
                "assay_type": "pkCSM in silico prediction",
                "endpoint_or_target": "skin sensitization prediction",
            },
            SKIN_NEAR_DIRECT_GROUP,
        ),
        (
            "prediction_only",
            {
                "assay_type": "OASIS protein binding alert",
                "aop_event": "MIE_protein_binding",
                "endpoint_or_target": "protein reactivity",
            },
            SKIN_AOP_GROUP,
        ),
        (
            "integrated_prediction_or_defined_approach",
            {
                "assay_type": "2 out of 3 defined approach",
                "endpoint_or_target": "skin sensitization hazard classification",
            },
            SKIN_NEAR_DIRECT_GROUP,
        ),
        (
            "integrated_prediction_or_defined_approach",
            {
                "assay_type": "U-SENS",
                "aop_event": "KE3_dendritic_cell_activation",
                "endpoint_or_target": "CD86 expression",
            },
            SKIN_AOP_GROUP,
        ),
        (
            "integrated_or_unresolved_endpoint",
            {
                "assay_type": "Kao integrated testing strategy",
                "endpoint_or_target": "total score",
                "support_text": "The total score classified the chemical as a strong sensitizer.",
            },
            SKIN_NEAR_DIRECT_GROUP,
        ),
        (
            "integrated_or_unresolved_endpoint",
            {
                "assay_type": "MDAM",
                "endpoint_or_target": "stimulation index",
                "support_text": "The modified Draize assay produced a positive response.",
            },
            SKIN_NEAR_DIRECT_GROUP,
        ),
    )
    for index, (reason, fields, expected) in enumerate(cases):
        key = ("sensitization_aop", index)
        row = {
            "group_id": SKIN_AOP_GROUP,
            "source_id": key[0],
            "source_row_number": index + 1,
            **fields,
        }
        moved = skin_vote_pure_family_move(
            row, set(), {key: PartitionDecision("reject", reason)}
        )
        assert moved.new_group == expected


def test_skin_anchored_unresolved_mechanism_enters_l3_but_empty_record_is_excluded():
    co_culture_key = ("sensitization_aop", 0)
    co_culture = {
        "group_id": SKIN_AOP_GROUP,
        "source_id": co_culture_key[0],
        "source_row_number": 1,
        "assay_type": "co-culture (HaCaT + THP-1) cytokine secretion assay",
        "endpoint_or_target": "IL-8 secretion",
        "support_text": "Exposure triggered IL-8 secretion.",
    }
    moved = skin_vote_pure_family_move(
        co_culture,
        set(),
        {
            co_culture_key: PartitionDecision(
                "reject", "integrated_or_unresolved_endpoint"
            )
        },
    )
    assert moved.new_group == SKIN_AOP_GROUP
    assert "anchored_unspecified_mechanism" in moved.reason

    generic_key = ("sensitization_aop", 2)
    generic = {
        "group_id": SKIN_AOP_GROUP,
        "source_id": generic_key[0],
        "source_row_number": 3,
        "assay_type": "real-time RT-PCR",
        "endpoint_or_target": "IL-6",
        "support_text": "Treatment changed IL-6 expression.",
    }
    moved = skin_vote_pure_family_move(
        generic,
        set(),
        {generic_key: PartitionDecision("reject", "integrated_or_unresolved_endpoint")},
    )
    assert moved.new_group == SKIN_AOP_GROUP
    assert "substantive_unspecified_mechanism" in moved.reason

    empty_key = ("sensitization_aop", 1)
    empty = {
        "group_id": SKIN_AOP_GROUP,
        "source_id": empty_key[0],
        "source_row_number": 2,
        "support_text": "The paper does not contain extractable skin sensitization evidence.",
    }
    moved = skin_vote_pure_family_move(
        empty,
        set(),
        {empty_key: PartitionDecision("reject", "integrated_or_unresolved_endpoint")},
    )
    assert moved.new_group == SKIN_EXCLUDED_GROUP

    truly_empty_key = ("sensitization_aop", 3)
    truly_empty = {
        "group_id": SKIN_AOP_GROUP,
        "source_id": truly_empty_key[0],
        "source_row_number": 4,
    }
    moved = skin_vote_pure_family_move(
        truly_empty,
        set(),
        {
            truly_empty_key: PartitionDecision(
                "reject", "integrated_or_unresolved_endpoint"
            )
        },
    )
    assert moved.new_group == SKIN_EXCLUDED_GROUP


def test_skin_photo_and_irritation_records_remain_excluded_from_levels():
    for index, reason in enumerate(
        ("out_of_scope_photo_hazard", "out_of_scope_irritation")
    ):
        key = ("sensitization_aop", index)
        row = {
            "group_id": SKIN_AOP_GROUP,
            "source_id": key[0],
            "source_row_number": index + 1,
        }
        moved = skin_vote_pure_family_move(
            row, set(), {key: PartitionDecision("reject", reason)}
        )
        assert moved.new_group == SKIN_EXCLUDED_GROUP
        assert moved.reason.endswith(reason)


def test_skin_strict_scope_guard_overrides_stale_direct_or_aop_partition():
    cases = (
        {
            "canonical_assay_context": "photo-LLNA",
            "canonical_endpoint_name": "sensitization",
        },
        {
            "canonical_assay_context": "skin irritation test",
            "canonical_endpoint_name": "sensitization",
            "support_text": "Sodium lauryl sulphate produced erythema.",
        },
        {
            "canonical_assay_context": "case report",
            "canonical_endpoint_name": "Stevens-Johnson syndrome",
        },
    )
    for index, fields in enumerate(cases):
        key = ("direct_skin_reaction", index)
        row = {
            "group_id": SKIN_DIRECT_GROUP,
            "source_id": key[0],
            "source_row_number": index + 1,
            **fields,
        }
        moved = skin_vote_pure_family_move(
            row,
            {key},
            {key: PartitionDecision("direct", "stale_upstream_classification")},
        )
        assert moved.new_group == SKIN_EXCLUDED_GROUP
        assert moved.reason.startswith("strict_target_scope_exclusion:")


def test_exact_voter_gate_rejects_nonvoters_and_misrouted_voters():
    rows = [
        {"record_id": "voter", "group_id": "L2", "retrieval_eligible": True},
        {"record_id": "nonvoter", "group_id": "L1", "retrieval_eligible": True},
    ]
    with pytest.raises(RuntimeError, match="exact voter-membership gate failed"):
        audit_exact_voter_membership(
            rows,
            {"voter"},
            direct_group="L1",
            record_id=lambda row: row["record_id"],
        )


def test_skin_review_preserves_source_and_blocks_stale_or_voter_changes():
    from tools.chembl_tool.tasks.skin_reaction.canonical_starling_source import AOP_PARTITION
    from tools.chembl_tool.tasks.skin_reaction.source_family_purity import (
        load_reviewed_source_decisions,
    )

    for key, decision in load_reviewed_source_decisions().items():
        row = {
            **decision["expected"], "source_id": key[0],
            "source_row_number": key[1] + 1, "group_id": SKIN_AOP_GROUP,
        }
        semantic = {key: PartitionDecision(AOP_PARTITION, "review fixture", "")}
        before = dict(row)
        result = skin_vote_pure_family_move(row, set(), semantic)
        assert result.new_group == decision["new_group"]
        assert row == before
        with pytest.raises(ValueError, match="frozen voter"):
            skin_vote_pure_family_move(row, {key}, semantic)
        with pytest.raises(ValueError, match="signature changed"):
            skin_vote_pure_family_move({**row, "support_text": "revised source"}, set(), semantic)
