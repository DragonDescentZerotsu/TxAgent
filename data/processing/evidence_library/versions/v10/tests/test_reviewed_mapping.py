from __future__ import annotations

import json

import pytest

from data.processing.evidence_library.shared.v2 import reviewed_mapping
from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256


@pytest.mark.parametrize("mapping_id", ["exact_measurement_units", "auxiliary_context"])
def test_changed_canonical_mapping_requires_review_receipt(
    tmp_path, monkeypatch, mapping_id
) -> None:
    monkeypatch.setattr(reviewed_mapping, "REPO_ROOT", tmp_path)
    entry = {"sha256": "changed"}
    with pytest.raises(ValueError, match="lacks a valid reviewed mapping receipt"):
        reviewed_mapping.validate_mapping_review(
            "bbb_martins", mapping_id, entry
        )

    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "version": "reviewed_canonical_mapping_receipt.v1",
                "task_id": "bbb_martins",
                "mapping_id": mapping_id,
                "publication_status": "reviewed",
                "mapping": {"sha256": "changed"},
                "review_completion": {"reviewer_id": "test"},
                "validations": {"complete": True},
            }
        )
    )
    entry["review_manifest"] = {
        "path": "receipt.json",
        "sha256": file_sha256(receipt),
    }
    reviewed_mapping.validate_mapping_review(
        "bbb_martins", mapping_id, entry
    )
