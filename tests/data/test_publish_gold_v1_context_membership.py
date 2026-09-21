from __future__ import annotations

import json

import pytest

from data.processing.gold_labels import publish_gold_v1_context_membership as publisher
from predict.utils.json import sha256_file


def test_reviewed_voter_exclusion_updates_mean_without_changing_label(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned = tmp_path / "pinned.txt"
    pinned.write_text("source authority\n", encoding="utf-8")
    correction = tmp_path / "correction.json"
    correction.write_text(
        json.dumps(
            {
                "version": "test.v1",
                "status": "reviewed",
                "task": "dili",
                "inputs": [{"path": "pinned.txt", "sha256": sha256_file(pinned)}],
                "decisions": [
                    {
                        "action": "exclude",
                        "source_row_uid": "remove",
                        "expected_split": "train",
                        "expected_benchmark_row_id": "card",
                        "expected_molecule_identity_key": "molecule",
                        "expected_condition_group": "condition",
                        "expected_vote_label": 1,
                        "expected_source_payload_sha256": "payload",
                        "expected_gold_label_before": 1,
                        "expected_gold_label_after": 1,
                        "expected_voter_mean_before": 2 / 3,
                        "expected_voter_mean_after": 0.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(publisher.TASKS, "dili", {"corrections": "correction.json"})
    cards = [
        (
            "train",
            {
                "benchmark_row_id": "card",
                "molecule_identity_key": "molecule",
                "condition_group": "condition",
                "Y": 1,
                "source_record_ids": ["keep-positive", "remove", "keep-negative"],
                "source_votes": [
                    {
                        "source_record_id": "keep-positive",
                        "Y": 1,
                        "source_payload_sha256": "a",
                    },
                    {
                        "source_record_id": "remove",
                        "Y": 1,
                        "source_payload_sha256": "payload",
                    },
                    {
                        "source_record_id": "keep-negative",
                        "Y": 0,
                        "source_payload_sha256": "b",
                    },
                ],
            },
        )
    ]
    corrected, provenance = publisher._apply_reviewed_corrections("dili", cards)

    assert corrected[0][1]["source_record_ids"] == ["keep-positive", "keep-negative"]
    assert provenance is not None
    assert provenance["applied"] == [
        {
            "source_row_uid": "remove",
            "benchmark_row_id": "card",
            "old_voter_mean": 2 / 3,
            "new_voter_mean": 0.5,
            "old_vote_count": 3,
            "new_vote_count": 2,
            "gold_label": 1,
        }
    ]
