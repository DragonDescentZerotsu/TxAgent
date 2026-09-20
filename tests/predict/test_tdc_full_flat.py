from __future__ import annotations

import json
from pathlib import Path

from predict.harnesses.branches import flat, matrix, runtime
from predict.retrieval.assay_reranking import build_tdc_indirect_ranked_retrieval
from predict.retrieval.assay_reranking.ranked_uid_retrieval import _payload
from predict.tasks.prompt_profiles import require_matching_prompt_profiles
from predict.utils.json import sha256_file


def test_tdc_matrix_uses_tdc_split() -> None:
    args = matrix._selection_args(
        "bbb_martins", Path("/tmp/tdc-full-flat"), limit=0,
        context_v5=True, all_levels=True, benchmark="tdc",
    )
    assert args.benchmark == "tdc"
    assert args.input_jsonl == Path(
        "data/gold_labels/TDC/BBB_Martins/v1/scaffold/valid_molecule_condition_labels.jsonl"
    ).resolve()
    assert args.prompt_version == flat.CONTEXT_V5_PROMPT_VERSION


def test_tdc_indirect_wrapper_selects_tdc_queries(monkeypatch) -> None:
    from predict.retrieval.assay_reranking import build_ranked_uid_retrieval as builder

    monkeypatch.setattr(builder, "QUERY_BENCHMARK", "tdc")
    monkeypatch.setattr(builder, "LABEL_RELEASE", {"benchmark": "tdc", "version": "v1"})
    monkeypatch.setattr(builder, "TASK_LEVELS", build_tdc_indirect_ranked_retrieval.LEVELS)
    path, rows, queries = builder._queries("bioavailability_ma", "test")
    assert "/gold_labels/TDC/Bioavailability_Ma/v1/scaffold/" in str(path)
    assert len(rows) == len(queries) == 128
    assert "L1" not in builder.TASK_LEVELS["bioavailability_ma"]


def test_branch_query_prior_overlay_resolves_hash_pinned_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    single = source / "single_molecule_reasoning_output.json"
    single.write_text(json.dumps({"status": "ok"}))
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps({
        "drug": "C", "molecule_identity_key": "K", "condition_group": "none"
    }) + "\n")
    batch = tmp_path / "overlay"
    batch.mkdir()
    (batch / "manifest.json").write_text(json.dumps({
        "schema_version": "branch_query_prior_overlay.v1",
        "input_jsonl": str(input_path), "input_sha256": sha256_file(input_path),
        "n_items": 1,
        "sources": [{
            "target_index": 0, "molecule_identity_key": "K",
            "condition_group": "none", "run_dir": str(source),
            "files_sha256": {single.name: sha256_file(single)},
        }],
    }))
    runtime._query_prior_overlay.cache_clear()
    assert runtime._source_run_dir(batch, 0) == source


def test_tdc_projection_rows_receive_flat_prompt_source_contract() -> None:
    result = _payload(
        {
            "source_row_uid": "train:1",
            "external_record_id": "train:1",
            "payload": json.dumps({
                "canonical_smiles": "CC",
                "label_source": "tdc_v1",
                "progressive_level": "L1",
                "source_fields": {"label": 1, "source_record_id": "train:1"},
            }),
        },
        {
            "parent_smiles": "CC",
            "parent_id": "PARENT",
            "morgan_similarity": 0.9,
            "morgan_rank": 1,
            "assay_rank": 1,
            "assay_transfer_score": 0.8,
        },
        "assay_transfer",
    )
    payload = result["payload"]
    assert payload["family_key"] == "tdc_training_labels"
    assert payload["source_id"] == "tdc_v1"
    assert payload["record_id"] == payload["source_row_uid"] == "train:1"
    assert payload["source_contract"]["source_or_simply_cleaned"] == {
        "label": True,
        "source_record_id": True,
    }


def test_query_prior_overlay_checks_nested_prompt_profiles(tmp_path: Path) -> None:
    run = tmp_path / "source"
    run.mkdir()
    source_manifest = run / "manifest.json"
    source_manifest.write_text(json.dumps({"task_prompt_profile": "profile-v2"}))
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "manifest.json").write_text(json.dumps({
        "schema_version": "branch_query_prior_overlay.v1",
        "task_prompt_profile": "profile-v2",
        "sources": [{
            "run_dir": str(run),
            "task_prompt_profile": "profile-v2",
            "prompt_profile_manifest": str(source_manifest),
            "prompt_profile_manifest_sha256": sha256_file(source_manifest),
        }],
    }))
    require_matching_prompt_profiles(
        target_profile="profile-v2",
        source_dirs=[overlay],
        historical_profile="legacy",
    )
