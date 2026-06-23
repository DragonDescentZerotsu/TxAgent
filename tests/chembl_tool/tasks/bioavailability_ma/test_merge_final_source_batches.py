from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.chembl_tool.common.task_workflows.merge_final_source_batches import merge_source_batches


def _write_run(batch: Path, batch_id: str, idx: int, group_id: str) -> None:
    run = batch / "runs" / f"{batch_id}_idx{idx:05d}"
    run.mkdir(parents=True)
    retrieval = {
        "status": "ok",
        "query": {"smiles": "CCO"},
        "groups": [{"group_id": group_id, "neighbors": [{"source_molecule_id": group_id}]}],
        "coverage": {
            "n_groups": 1,
            "n_groups_with_neighbors": 1,
            "n_neighbors_total": 1,
            "min_similarity": 0.3,
            "top_k_per_group": 3,
        },
        "evidence_source": {"type": group_id},
    }
    (run / "retrieval.json").write_text(json.dumps(retrieval), encoding="utf-8")
    (run / "single_molecule_reasoning_output.json").write_text(
        json.dumps({"status": "ok", "batch": batch_id}),
        encoding="utf-8",
    )
    (run / "group_reasoning_outputs.jsonl").write_text(
        json.dumps({"group_id": group_id, "status": "ok"}) + "\n",
        encoding="utf-8",
    )


def test_merge_source_batches_combines_retrieval_and_group_outputs(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    out = tmp_path / "merged"
    _write_run(primary, "primary", 0, "Tier 1.direct_absolute_bioavailability")
    _write_run(extra, "extra", 0, "Starling.direct_oral_bioavailability")

    manifest = merge_source_batches(
        primary_batch=primary,
        extra_batches=[extra],
        out_batch=out,
    )

    assert manifest["n_items"] == 1
    run = out / "runs" / "merged_idx00000"
    retrieval = json.loads((run / "retrieval.json").read_text(encoding="utf-8"))
    assert [group["group_id"] for group in retrieval["groups"]] == [
        "Tier 1.direct_absolute_bioavailability",
        "Starling.direct_oral_bioavailability",
    ]
    assert retrieval["coverage"]["n_groups"] == 2
    assert retrieval["coverage"]["n_neighbors_total"] == 2
    group_lines = (run / "group_reasoning_outputs.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["group_id"] for line in group_lines] == [
        "Tier 1.direct_absolute_bioavailability",
        "Starling.direct_oral_bioavailability",
    ]
    single = json.loads((run / "single_molecule_reasoning_output.json").read_text(encoding="utf-8"))
    assert single["batch"] == "primary"


def test_merge_source_batches_rejects_duplicate_group_ids(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    out = tmp_path / "merged"
    _write_run(primary, "primary", 0, "duplicate.group")
    _write_run(extra, "extra", 0, "duplicate.group")

    with pytest.raises(ValueError, match="Duplicate group_id"):
        merge_source_batches(
            primary_batch=primary,
            extra_batches=[extra],
            out_batch=out,
        )
