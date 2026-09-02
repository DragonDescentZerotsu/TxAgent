from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.chembl_tool.common.starling.benchmark_dataset import LabeledSourceRecord
from tools.chembl_tool.common.source_family_purity import audit_exact_voter_membership
from tools.chembl_tool.paper_experiments.audit_source_family_purity_gold_impact import (
    _skin_binary_label,
    _vote,
)
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
    NEAR_DIRECT_GROUP as SKIN_NEAR_DIRECT_GROUP,
    upstream_source_index,
    vote_pure_family_move as skin_vote_pure_family_move,
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
    rows = [
        LabeledSourceRecord("CCO", label, "source")
        for label in [1, 1, 1, 0]
    ]
    assert _vote(rows)["label"] == 1
    assert _vote([*rows, LabeledSourceRecord("CCO", 0, "candidate")])["label"] is None
    assert _skin_binary_label("positive") == 1
    assert _skin_binary_label("negative") == 0
    assert _skin_binary_label("inconclusive") is None


def test_skin_l1_uses_exact_source_voter_membership():
    voter = {
        "group_id": SKIN_DIRECT_GROUP,
        "source_id": "direct_skin_reaction",
        "source_row_number": 3,
    }
    assert upstream_source_index(voter) == 2
    assert skin_vote_pure_family_move(voter, {2}).reason == ""

    nonvoter = {**voter, "source_row_number": 4}
    moved = skin_vote_pure_family_move(nonvoter, {2})
    assert moved.new_group == SKIN_NEAR_DIRECT_GROUP
    assert moved.reason == "nonvoter_removed_from_l1"


def test_skin_direct_like_aop_nonvoter_goes_to_l2_not_l1():
    row = {
        "group_id": SKIN_AOP_GROUP,
        "source_id": "sensitization_aop",
        "source_row_number": 1,
        "canonical_assay_type": "local lymph node assay (LLNA)",
        "result_label": "positive",
    }
    moved = skin_vote_pure_family_move(row, set())
    assert moved.new_group == SKIN_NEAR_DIRECT_GROUP
    assert moved.reason.startswith("direct_like_nonvoter_to_l2:")


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
