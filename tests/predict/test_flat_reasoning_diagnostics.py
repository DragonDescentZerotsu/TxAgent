from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from analysis.prediction.flat_reasoning_diagnostics import build
from predict.harnesses.reasoning_references import match_references


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _trace_request() -> dict:
    return {
        "schema_version": "joseph_flat_context_request.v1",
        "reasoning_reference_contract": {
            "schema_version": "progressive_reasoning_references.v1",
            "layout": "flat_level_grouped",
            "required_mentions": False,
            "matching": "case_insensitive_exact_visible_id",
        },
        "reasoning_reference_index": [
            {"unit_kind": "molecule", "visible_id": "Molecule 1", "stable_id": "mol-1"},
            {"unit_kind": "record", "visible_id": "Record 1-1", "stable_id": "rec-1"},
            {"unit_kind": "molecule", "visible_id": "Molecule 2", "stable_id": "mol-2"},
            {"unit_kind": "record", "visible_id": "C01", "stable_id": "rec-2"},
        ],
    }


def _make_matrix(tmp_path: Path) -> tuple[Path, Path, Path]:
    batch = tmp_path / "matrix-batch"
    condition = batch / "profile" / "with_query_prior" / "task-condition"
    selection_path = condition / "cache_matched_retrieval" / "manifest.json"
    selection_manifest = {
        "task": "bbb_martins",
        "evaluation_subset": "test",
        "prompt_version": "full_flat_context_v5_six_tasks_upstream_v2",
        "query_prior": "cached",
        "selection_contract_sha256": "contract-hash",
    }
    _write(selection_path, selection_manifest)
    pinned = tmp_path / "selection.json"
    _write(pinned, {"status": "complete", "schema_version": "direct_context_selection.v2"})
    matrix_path = batch / "matrix.json"
    _write(matrix_path, {
        "status": "incomplete",
        "benchmark": "gold",
        "prompt_version": "full_flat_context_v5_six_tasks_upstream_v2",
        "model": "model/example",
        "reasoning_effort": "high",
        "selections": [{
            "task": "bbb_martins",
            "evaluation_subset": "test",
            "profile": "ga125_mc000_label025",
            "reranking": "assay-transfer-contrastive",
            "record_pool": "all",
            "preselected_contexts": str(pinned),
            "preselected_uids": "",
            "mixed_selection": "",
            "path": str(selection_path),
            "sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
            "selection_contract_sha256": "contract-hash",
        }],
    })
    _write(condition / "manifest.json", {
        "n_items": 2, "indices": [0, 1], "model": "model/example",
    })

    for query_index in (0, 1):
        run_id = f"bbb_martins__profile_idx{query_index:05d}"
        run = condition / "runs" / run_id
        _write(run / "manifest.json", {"run_id": run_id, "query_index": query_index})
        _write(run / "request.json", _trace_request())
        if query_index == 0:
            _write(run / "final_reasoning_output.json", {
                "status": "ok",
                "llm": {
                    "reasoning_content": (
                        "molecule 1 meets Molecule 1; Record 1-1 and c01; Evidence group 9"
                    ),
                    "usage": {"reasoning_tokens": 17},
                    "content": {"claims": [
                        {"evidence_role": "supportive"},
                        {"evidence_role": "contradictory"},
                        {"evidence_role": "contextual"},
                    ]},
                },
            })
    return matrix_path, condition, batch


def test_shared_matcher_handles_repeats_unknown_ids_and_ambiguous_ids():
    index = [
        {"visible_id": "Molecule 1", "stable_id": "mol-a"},
        {"visible_id": "Molecule 1", "stable_id": "mol-b"},
        {"visible_id": "Record 1-1", "stable_id": "record-a"},
    ]
    found, occurrences, unknown, ambiguous, ambiguous_occurrences = match_references(
        "molecule 1, MOLECULE 1, record 1-1, Record 8-2, C01", index
    )
    assert found == {"record-a"}
    assert occurrences == 1
    assert unknown == ["C01", "Record 8-2"]
    assert ambiguous == ["Molecule 1"]
    assert ambiguous_occurrences == 2


def test_flat_trace_diagnostics_measure_usage_and_publish_partial_then_complete(tmp_path):
    matrix_path, _, _ = _make_matrix(tmp_path)
    output_root = tmp_path / "analysis"

    partial = build([matrix_path], output_root, allow_partial=True)
    assert partial == output_root / "partial"
    partial_manifest = json.loads((partial / "diagnostics_manifest.json").read_text())
    assert partial_manifest["status"] == "partial"
    assert partial_manifest["expected_queries"] == 2
    assert partial_manifest["measured_queries"] == 1
    assert partial_manifest["n_diagnostic_gaps"] == 2
    assert any(path.endswith("request.json") for path in partial_manifest["inputs"])
    assert any(path.endswith("final_reasoning_output.json") for path in partial_manifest["inputs"])
    assert "predict/harnesses/reasoning_references.py" in " ".join(partial_manifest["inputs"])
    with (partial / "reasoning_usage.per_query.tsv").open(newline="") as handle:
        measured = next(csv.DictReader(handle, delimiter="\t"))
    assert measured["visible_molecule_count"] == "2"
    assert measured["mentioned_molecule_count"] == "1"
    assert measured["molecule_coverage"] == "0.500000"
    assert measured["visible_record_count"] == "2"
    assert measured["mentioned_record_count"] == "2"
    assert measured["matched_reference_occurrences"] == "4"
    assert json.loads(measured["unknown_reference_ids"]) == ["Evidence group 9"]
    assert measured["reasoning_word_count"] == "12"
    assert measured["reasoning_tokens"] == "17"
    assert measured["claim_count"] == "3"
    assert measured["supportive_claim_count"] == "1"
    assert measured["contradictory_claim_count"] == "1"
    assert measured["other_claim_count"] == "1"
    assert "Molecule 1 meets" not in (partial / "report.md").read_text()
    with (partial / "reasoning_usage.summary.tsv").open(newline="") as handle:
        summary = next(csv.DictReader(handle, delimiter="\t"))
    assert summary["query_count"] == "1"
    assert summary["expected_query_count"] == "2"
    assert build([matrix_path], output_root, allow_partial=True) == partial

    with pytest.raises(ValueError, match="diagnostic gaps"):
        build([matrix_path], output_root)

    matrix = json.loads(matrix_path.read_text())
    matrix["status"] = "complete"
    _write(matrix_path, matrix)
    second = matrix_path.parent / "profile/with_query_prior/task-condition/runs/"
    _write(second / "bbb_martins__profile_idx00001/final_reasoning_output.json", {
        "status": "ok",
        "llm": {
            "reasoning_content": "",
            "usage": {},
            "content": {"claims": []},
        },
    })

    complete = build([matrix_path], output_root)
    assert complete == output_root / "complete"
    complete_manifest = json.loads((complete / "diagnostics_manifest.json").read_text())
    assert complete_manifest["status"] == "complete"
    assert complete_manifest["n_diagnostic_gaps"] == 0
    with (complete / "reasoning_usage.per_query.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 2
    assert rows[1]["reasoning_char_count"] == "0"
    assert rows[1]["matched_reference_occurrences"] == "0"
    assert rows[1]["molecule_coverage"] == "0.000000"
    assert rows[1]["record_coverage"] == "0.000000"
    assert rows[1]["mentioned_molecule_count"] == "0"
    assert rows[1]["mentioned_record_count"] == "0"
    assert rows[1]["claim_count"] == "0"
    with pytest.raises(FileExistsError, match="completed analysis"):
        build([matrix_path], output_root)
