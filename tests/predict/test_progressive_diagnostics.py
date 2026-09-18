from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from predict.harnesses.progressive.diagnostics import build_run_diagnostics
from predict.harnesses.progressive.references import (
    build_reference_index,
    match_references,
    normalize_headings,
    validate_prompt_index,
    validate_unique_visible_ids,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_weighted_reference_layout_is_unique_and_case_insensitive():
    active = {
        "l1": {
            "_selection_rank": 0,
            "group_kind": "l1_context",
            "cards": {"a": {"first_seen_level": 1}},
        },
        "bucket": {
            "_selection_rank": 1,
            "group_kind": "semantic_bucket",
            "cards": {
                "b": {"first_seen_level": 2, "reference_smiles": "CC"},
                "c": {"first_seen_level": 2, "reference_smiles": "CO"},
            },
        },
    }
    aliases = {"a": "C01", "b": "C02", "c": "C03"}
    index = build_reference_index(
        active,
        card_id_to_alias=aliases,
        current_level=2,
        layout="weighted_l2",
    )
    assert [row["visible_id"] for row in index if row["unit_kind"] != "record"] == [
        "Molecule 1", "Evidence group 1", "Molecule 2", "Molecule 3"
    ]
    rendered = normalize_headings(
        "## Molecule 1\nRecord C01\n## Semantic bucket 1\n"
        "### Molecule 1\nRecord C02\n### Molecule 2\nRecord C03"
    )
    validate_prompt_index(rendered, index)
    validate_unique_visible_ids(index)
    found, occurrences, unknown, ambiguous, ambiguous_occurrences = match_references(
        "mOlEcUlE 1 and c02; MOLECULE 1 but evidence GROUP 9", index
    )
    assert {row["visible_id"] for row in index if row["stable_id"] in found} == {
        "Molecule 1", "C02"
    }
    assert occurrences == 3
    assert unknown == ["evidence GROUP 9"]
    assert ambiguous == []
    assert ambiguous_occurrences == 0

    legacy = build_reference_index(
        active,
        card_id_to_alias=aliases,
        current_level=2,
        layout="weighted_l2_legacy",
    )
    found, _, _, ambiguous, _ = match_references("molecule 1", legacy)
    assert found == set()
    assert ambiguous == ["Molecule 1"]
    validate_prompt_index(
        "## Molecule 1\nRecord C01\n## Semantic bucket 1\n"
        "### Molecule 1\nRecord C02\n### Molecule 2\nRecord C03",
        legacy,
    )
    with pytest.raises(ValueError, match="ambiguous reasoning-reference"):
        validate_unique_visible_ids(legacy)


def test_diagnostics_measure_zero_coverage_and_selected_context_labels(tmp_path):
    _write(tmp_path / "experiment_manifest.json", {"evaluation_subset": "valid"})
    stage = tmp_path / "bbb_martins/queries/query_idx00000/levels/level_1"
    _write(stage / "prepared.json", {
        "task": "bbb_martins",
        "level": 1,
        "query_index": 0,
        "benchmark_row_id": "query-0",
        "should_call_model": True,
        "active_evidence": {
            "analog-a": {
                "_selection_rank": 0,
                "first_seen_level": 1,
                "group_kind": "l1_context",
                "_diagnostic_label": 0,
                "_diagnostic_context_id": "context-a",
                "_diagnostic_parent_id": "parent-a",
                "cards": {"record-a": {"first_seen_level": 1}},
            },
            "analog-b": {
                "_selection_rank": 1,
                "first_seen_level": 1,
                "group_kind": "l1_context",
                "_diagnostic_label": 1,
                "_diagnostic_context_id": "context-b",
                "_diagnostic_parent_id": "parent-b",
                "cards": {"record-b": {"first_seen_level": 1}},
            },
        },
    })
    _write(stage / "output.json", {
        "status": "ok",
        "llm": {"reasoning_content": "No visible evidence identifier is named."},
    })

    manifest = build_run_diagnostics(tmp_path)

    assert manifest["status"] == "complete"
    with (tmp_path / "neighborhood_label_mix.per_query.tsv").open(newline="") as handle:
        mix = next(csv.DictReader(handle, delimiter="\t"))
    assert mix["minority_share"] == "0.5"
    with (tmp_path / "reasoning_reference_coverage.per_query.tsv").open(newline="") as handle:
        coverage = next(csv.DictReader(handle, delimiter="\t"))
    assert coverage["molecule_or_group_coverage"] == "0.0"
    assert coverage["unit_coverage"] == "0.0"

    _write(stage / "output.json", {
        "status": "ok",
        "llm": {
            "reasoning_content": "Molecule 1 agrees with Molecule 1 and C01; Molecule 3 is absent."
        },
    })
    build_run_diagnostics(tmp_path)
    with (tmp_path / "reasoning_reference_coverage.per_query.tsv").open(newline="") as handle:
        coverage = next(csv.DictReader(handle, delimiter="\t"))
    assert coverage["referenced_molecule_or_group_count"] == "1"
    assert coverage["referenced_record_count"] == "1"
    assert coverage["referenced_unit_count"] == "2"
    assert coverage["reference_occurrence_count"] == "3"
    assert coverage["unknown_reference_tokens"] == "Molecule 3"

    _write(stage / "output.json", {"status": "ok", "llm": {"reasoning_content": ""}})
    with pytest.raises(ValueError, match="missing private reasoning"):
        build_run_diagnostics(tmp_path)

    manifest = build_run_diagnostics(tmp_path, allow_partial=True)
    assert manifest["status"] == "partial"
    assert manifest["n_diagnostic_gaps"] == 1
    with (tmp_path / "diagnostic_gaps.tsv").open(newline="") as handle:
        gap = next(csv.DictReader(handle, delimiter="\t"))
    assert gap["status"] == "missing_reasoning"
    with (tmp_path / "neighborhood_label_mix.per_query.tsv").open(newline="") as handle:
        assert next(csv.DictReader(handle, delimiter="\t"))["minority_share"] == "0.5"


def test_historical_weighted_duplicate_molecule_labels_are_undefined(tmp_path):
    _write(tmp_path / "experiment_manifest.json", {
        "evaluation_subset": "valid",
        "prompt_profile": "reranked_progressive_l1_context_l2_weighted_v1",
    })
    base = tmp_path / "bbb_martins/queries/query_idx00000/levels"
    l1_active = {
        "l1": {
            "_selection_rank": 0,
            "group_kind": "l1_context",
            "_diagnostic_label": 1,
            "_diagnostic_context_id": "context-a",
            "_diagnostic_parent_id": "parent-a",
            "cards": {"a": {"first_seen_level": 1}},
        },
    }
    _write(base / "level_1/prepared.json", {
        "task": "bbb_martins", "level": 1, "query_index": 0,
        "benchmark_row_id": "query-0", "should_call_model": True,
        "active_evidence": l1_active,
    })
    _write(base / "level_1/output.json", {
        "status": "ok", "llm": {"reasoning_content": "Molecule 1 and C01"},
    })
    l2_active = {
        **l1_active,
        "bucket": {
            "_selection_rank": 1,
            "group_kind": "semantic_bucket",
            "cards": {"b": {"first_seen_level": 2, "reference_smiles": "CC"}},
        },
    }
    _write(base / "level_2/prepared.json", {
        "task": "bbb_martins", "level": 2, "query_index": 0,
        "benchmark_row_id": "query-0", "should_call_model": True,
        "active_evidence": l2_active,
    })
    _write(base / "level_2/request.json", {"messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": (
            "## Molecule 1\nRecord C01\n## Semantic bucket 1\n"
            "### Molecule 1\nRecord C02"
        )},
    ]})
    _write(base / "level_2/output.json", {
        "status": "ok", "llm": {"reasoning_content": "molecule 1 and c02"},
    })

    build_run_diagnostics(tmp_path)
    with (tmp_path / "reasoning_reference_coverage.per_query.tsv").open(
        newline=""
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    coverage = next(row for row in rows if row["level"] == "2")
    assert coverage["molecule_coverage_status"] == "undefined_ambiguous_visible_ids"
    assert coverage["molecule_coverage"] == ""
    assert coverage["unit_coverage"] == ""
    assert coverage["record_coverage"] == "0.5"
    assert coverage["ambiguous_reference_tokens"] == "Molecule 1"
    assert coverage["ambiguous_reference_occurrence_count"] == "1"


def test_indirect_diagnostics_allow_empty_l1_and_report_filter(tmp_path):
    _write(tmp_path / "experiment_manifest.json", {"evaluation_subset": "valid"})
    stage = tmp_path / "bbb_martins/queries/query_idx00000/levels/level_2"
    _write(stage / "prepared.json", {
        "task": "bbb_martins", "level": 2, "query_index": 0,
        "benchmark_row_id": "query-0", "should_call_model": True,
        "active_evidence": {"m1": {"_selection_rank": 0,
            "group_kind": "parent_molecule", "first_seen_level": 2,
            "cards": {"record-a": {"first_seen_level": 2}}}},
    })
    _write(stage / "output.json", {
        "status": "ok", "llm": {"reasoning_content": "Molecule 1 and C01"},
    })
    receipt = {"status": "ok", "version": "query_conditioned_record_filter_v1",
        "task": "bbb_martins", "query_index": 0, "benchmark_row_id": "query-0",
        "level": 2, "candidate_count": 100, "selected_count": 20,
        "selected_parent_count": 4, "selected_semantic_bucket_count": 7,
        "mean_morgan_similarity": 0.71, "repeated_parent_records": 16,
        "overlap_semantic_top25": 9, "overlap_control25": 3, "records": []}
    _write(stage / "filter/selection.json", receipt)
    _write(stage / "filter/request.json", {"messages": []})
    _write(stage / "filter/output.json", {"status": "ok"})

    manifest = build_run_diagnostics(tmp_path)

    assert manifest["n_l1_queries"] == 0
    assert manifest["n_record_filter_queries"] == 1
    assert "N/A (no L1 stages)" in (tmp_path / "report.md").read_text()
    with (tmp_path / "record_filter.summary.tsv").open(newline="") as handle:
        summary = next(csv.DictReader(handle, delimiter="\t"))
    assert summary["mean_overlap_control25"] == "3.0"
