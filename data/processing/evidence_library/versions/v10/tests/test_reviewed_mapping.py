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


def test_unreviewed_acceptance_is_hash_and_release_scoped(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(reviewed_mapping, "REPO_ROOT", tmp_path)
    receipt = tmp_path / "acceptance.json"
    receipt.write_text(
        json.dumps(
            {
                "version": "accepted_unreviewed_canonical_mapping_exception.v1",
                "task_id": "ames",
                "mapping_id": "auxiliary_context",
                "publication_status": "accepted_without_additional_review",
                "target_release": "v10_main_universe_v2",
                "mapping": {"sha256": "accepted"},
                "validations": {"complete": True},
            }
        )
    )
    entry = {
        "sha256": "accepted",
        "acceptance_exception": {
            "path": "acceptance.json",
            "sha256": file_sha256(receipt),
        },
    }
    registry = {"mappings": {"auxiliary_context": entry}}

    reviewed_mapping.validate_mapping_review("ames", "auxiliary_context", entry)
    reviewed_mapping.validate_mapping_acceptance_scope(
        "ames", registry, "v10_main_universe_v2"
    )
    with pytest.raises(ValueError, match="not valid for"):
        reviewed_mapping.validate_mapping_acceptance_scope(
            "ames", registry, "v10_main_universe_v3"
        )

    entry["sha256"] = "changed"
    with pytest.raises(ValueError, match="incomplete"):
        reviewed_mapping.validate_mapping_review("ames", "auxiliary_context", entry)
