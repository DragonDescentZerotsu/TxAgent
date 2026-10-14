import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
from optimization.selection import objectives as gold_joint, pool as tdc_v2_mixed
from predict.utils.json import sha256_file


def test_sqrt_gated_assay_changes_weak_analog_ranking() -> None:
    candidates = [
        {
            "item_id": "assay",
            "parent_smiles": "CC",
            "morgan_similarity": 0.1,
            "assay_transfer_score": 0.9,
            "gold_label": 0,
            "canonical_smiles": "CC",
        },
        {
            "item_id": "morgan",
            "parent_smiles": "CCC",
            "morgan_similarity": 0.2,
            "assay_transfer_score": 0.1,
            "gold_label": 0,
            "canonical_smiles": "CCC",
        },
    ]
    profile = gold_joint.DirectProfile("example", 1, 0, 0)
    old, _ = gold_joint.select_direct_contexts(candidates, profile=profile, budget=1)
    new, _ = gold_joint.select_direct_contexts(
        candidates, profile=profile, budget=1, sqrt_gated_assay=True
    )
    assert old[0]["item_id"] == "morgan"
    assert new[0]["item_id"] == "assay"


def test_tdc_v2_requires_selected_ledger_and_projection_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    ledger = tmp_path / "central_ledger.parquet"
    weights = tmp_path / "weights.parquet"
    records = tmp_path / "projection" / "record_relevance_rankings.parquet"
    records.parent.mkdir()
    pd.DataFrame(
        [
            {
                "source_row_uid": "reviewed",
                "level": "L2",
                "assignment_status": "assigned",
                "own_level_semantic_bucket_id": "family",
            },
            {
                "source_row_uid": "pending",
                "level": "L2",
                "assignment_status": "unassigned",
                "own_level_semantic_bucket_id": None,
            },
        ]
    ).to_parquet(ledger, index=False)
    pd.DataFrame(
        [("L2", "family", 0.2), ("L2", "other", 0.8)],
        columns=["level", "semantic_bucket_id", "weight"],
    ).to_parquet(weights, index=False)
    pd.DataFrame(
        [
            ("reviewed", "L2", "family", 0.2, "reviewed_family_mean"),
            ("pending", "L2", None, 0.5, "provisional_task_level_mean"),
        ],
        columns=[
            "source_row_uid",
            "level",
            "semantic_bucket_id",
            "weight",
            "weight_source",
        ],
    ).to_parquet(records, index=False)
    projection = records.parent / "manifest.json"
    projection.write_text(
        json.dumps(
            {
                "schema_version": "semantic_retrieval_mean_weights.v2",
                "status": "complete_reviewed_with_provisional_mean",
                "inputs": {
                    "ledger": {"sha256": sha256_file(ledger)},
                    "weights": {"sha256": sha256_file(weights)},
                },
            }
        )
    )
    pending = tmp_path / "pending_uids.tsv"
    pending.write_text("source_row_uid\npending\n")
    release = tmp_path / "manifest.json"
    paths = {
        "central_ledger": ledger,
        "semantic_bucket_weights": weights,
        "record_relevance_rankings": records,
        "weight_projection_manifest": projection,
        "pending_uid_worklist": pending,
    }
    release.write_text(
        json.dumps(
            {
                "provisional_retrieval_weight_policy": "mean_of_reviewed_semantic_buckets_in_task_level",
                "files": {
                    key: {
                        "path": str(path.relative_to(tmp_path)),
                        "sha256": sha256_file(path),
                    }
                    for key, path in paths.items()
                },
            }
        )
    )
    artifacts = SimpleNamespace(
        root=tmp_path,
        manifest=release,
        central_ledger=ledger,
        semantic_bucket_weights=weights,
        record_relevance_rankings=records,
    )
    monkeypatch.setattr(
        tdc_v2_mixed, "resolve_semantic_bucket_artifacts", lambda *args: artifacts
    )
    monkeypatch.setattr(
        tdc_v2_mixed,
        "load_reviewed_record_weights",
        lambda *args: (
            {("reviewed", "L2"): {}, ("pending", "L2"): {}},
            {
                "release_manifest": str(release),
                "release_manifest_sha256": sha256_file(release),
            },
        ),
    )
    rows, pins = tdc_v2_mixed._semantic_ledger("ames")
    assert len(rows) == 2 and pins[str(ledger)] == sha256_file(ledger)
    extension_rows = tmp_path / "extension_rows.parquet"
    retrieval_uids = tmp_path / "retrieval_uids.parquet"
    pd.DataFrame(
        [("new", "L3", "family", 0.2, "reviewed_family_mean")],
        columns=[
            "source_row_uid",
            "level",
            "semantic_bucket_id",
            "weight",
            "weight_source",
        ],
    ).to_parquet(extension_rows, index=False)
    pd.DataFrame([("new", "L3")], columns=["source_row_uid", "level"]).to_parquet(
        retrieval_uids, index=False
    )
    extension = tmp_path / "extension.json"
    extension.write_text(
        json.dumps(
            {
                "schema_version": "semantic_retrieval_mean_weights.v2",
                "status": "candidate_unreviewed",
                "inputs": {
                    "ledger": {"sha256": sha256_file(ledger)},
                    "weights": {"sha256": sha256_file(weights)},
                    "retrieval_uids": {
                        "path": str(retrieval_uids),
                        "sha256": sha256_file(retrieval_uids),
                    },
                },
                "projection": {
                    "path": extension_rows.name,
                    "sha256": sha256_file(extension_rows),
                    "rows": 1,
                },
            }
        )
    )
    extended, _ = tdc_v2_mixed._semantic_ledger("ames", extension=extension)
    assert extended["new", "L3"]["semantic_weight"] == 0.2
    changed = json.loads(projection.read_text())
    changed["status"] = "candidate_unreviewed"
    projection.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="projection manifest changed"):
        tdc_v2_mixed._semantic_ledger("ames")
    artifacts.central_ledger = None
    with pytest.raises(ValueError, match="requires the selected central ledger"):
        tdc_v2_mixed._semantic_ledger("ames")
