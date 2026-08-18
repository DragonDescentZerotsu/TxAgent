from __future__ import annotations

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.common.starling.audit_assay_transfer_retrieval_invariance import (
    compare_snapshot,
)
from tools.chembl_tool.common.starling.retrieval_boundary import (
    FROZEN_RETRIEVAL_FIELDS,
    freeze_normalized_retrieval_identity,
    freeze_retrieval_boundary,
)


def test_retrieval_snapshot_ignores_only_artifact_file_identity() -> None:
    baseline = {
        "audit_version": "assay_transfer_retrieval_invariance.v1",
        "task_id": "bbb_martins",
        "record_contract_version": "records.v7",
        "record_count": 2,
        "retrieval_eligible_count": 1,
        "retrieval_boundary_sha256": "same",
        "source_projection_fields": ["measurement_text", "unit_text"],
        "records_file_sha256": "old",
    }
    current = {**baseline, "records_file_sha256": "new"}

    compare_snapshot(baseline, current)


def test_retrieval_snapshot_rejects_membership_change() -> None:
    baseline = {
        "audit_version": "assay_transfer_retrieval_invariance.v1",
        "task_id": "bbb_martins",
        "record_contract_version": "records.v7",
        "record_count": 2,
        "retrieval_eligible_count": 1,
        "retrieval_boundary_sha256": "old",
        "source_projection_fields": ["measurement_text", "unit_text"],
    }

    with pytest.raises(ValueError, match="retrieval_boundary_sha256"):
        compare_snapshot(baseline, {**baseline, "retrieval_boundary_sha256": "new"})


def test_freeze_retrieval_boundary_preserves_only_owned_fields(tmp_path) -> None:
    frozen = {
        "cleaned_record_id": "clean-1",
        **{field: f"old-{field}" for field in FROZEN_RETRIEVAL_FIELDS},
    }
    frozen["duplicate_group_size"] = 2
    frozen["retrieval_eligible"] = True
    path = tmp_path / "records.parquet"
    pq.write_table(pa.Table.from_pylist([frozen]), path)
    current = {
        "cleaned_record_id": "clean-1",
        "canonical_measurement_text": "-6",
        **{field: f"new-{field}" for field in FROZEN_RETRIEVAL_FIELDS},
    }

    result = freeze_retrieval_boundary([current], path)[0]

    assert result["canonical_measurement_text"] == "-6"
    assert result["canonical_smiles"] == "old-canonical_smiles"
    assert result["retrieval_eligible"] is True
    assert result["duplicate_group_size"] == 2

    normalized = freeze_normalized_retrieval_identity([current], path)[0]
    assert normalized["canonical_smiles"] == "old-canonical_smiles"
    assert normalized["retrieval_eligible"] == "new-retrieval_eligible"
