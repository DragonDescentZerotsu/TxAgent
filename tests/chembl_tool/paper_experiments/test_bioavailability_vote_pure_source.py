from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.paper_experiments.build_bioavailability_vote_pure_source import (
    _audit_membership,
)
from tools.chembl_tool.tasks.bioavailability_ma.source_family_purity import (
    DIRECT_GROUP,
    NEAR_DIRECT_GROUP,
    vote_pure_family_move,
)


def _record(group: str, record_id: str) -> dict:
    return {
        "group_id": group,
        "source_id": "hf_bioavailability",
        "source_record_id": record_id,
        "source_row_number": 1,
        "extraction_id": "",
    }


def test_vote_pure_move_keeps_only_exact_voters_in_l1():
    voters = {"hf:7"}
    assert vote_pure_family_move(_record(DIRECT_GROUP, "7"), voters).reason == ""
    moved = vote_pure_family_move(_record(DIRECT_GROUP, "8"), voters)
    assert moved.new_group == NEAR_DIRECT_GROUP
    assert moved.reason == "nonvoter_removed_from_l1"


def test_membership_audit_accepts_absent_upstream_voter(tmp_path: Path):
    path = tmp_path / "records.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _record(DIRECT_GROUP, "7"),
                _record(NEAR_DIRECT_GROUP, "8"),
            ]
        ),
        path,
    )
    audit = _audit_membership(path, {"hf:7", "hf:missing"})
    assert audit["l1_exact_vote_membership"] is True
    assert audit["n_l1_records"] == 1
    assert audit["n_l1_nonvoter_records"] == 0
    assert audit["n_voter_upstream_ids_not_present_in_retrieval_source"] == 1
