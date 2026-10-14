from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.compile_candidate_resolution import (
    COMPILED_MAPPING_VERSION,
    compile_mapping,
    validate_compiled_mapping,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_candidate_resolution import (
    CANDIDATE_CONTRACT_VERSION,
    candidate_set_sha256,
)


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _candidate(record_id: str, value: str) -> dict[str, object]:
    candidates = [
        {
            "candidate_id": "1",
            "measurement": value,
            "unit": "mutation frequency",
            "rule_id": "named_readout.v1",
            "evidence": [f"support_text: mutation frequency was {value}"],
            "hints": [],
        }
    ]
    encoded = json.dumps(candidates, ensure_ascii=False, sort_keys=True)
    return {
        "cleaned_record_id": record_id,
        "source_row_uid": f"source-{record_id}",
        "source_id": "fixed_mutation",
        "canonical_endpoint_name": "microbial_forward_mutation",
        "measurement_candidates_json": encoded,
        "candidate_set_sha256": candidate_set_sha256(encoded),
        "candidate_contract_version": CANDIDATE_CONTRACT_VERSION,
    }


def _selection(record_id: str, status: str = "ok") -> dict[str, object]:
    reason = "absolute" if status == "ok" else "missing_unit"
    measurements = (
        [{"measurement": "1", "unit": "candidate_id"}] if status == "ok" else []
    )
    raw = {
        "id": "r1",
        "reason": reason,
        "candidate_id": "1" if status == "ok" else None,
        "status": status,
        "measurements": measurements,
    }
    return {
        "cleaned_record_id": record_id,
        "source_row_uid": f"source-{record_id}",
        "source_id": "fixed_mutation",
        "status": status,
        "measurements_json": json.dumps(measurements),
        "quantity_count": len(measurements),
        "assignment_method": "model_single_pass",
        "raw_response_json": json.dumps(raw),
        "rejected_response_json": None,
        "inference_model": "gpt-5.4-mini-2026-03-17",
    }


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    candidates = tmp_path / "candidates.parquet"
    selections = tmp_path / "selections.parquet"
    _write_parquet(candidates, [_candidate("a", "6"), _candidate("b", "7")])
    candidates.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "inventory_version": CANDIDATE_CONTRACT_VERSION,
                "candidate_rows": 2,
                "candidate_sha256": file_sha256(candidates),
            }
        )
    )
    _write_parquet(selections, [_selection("a"), _selection("b", "unsure")])
    selections.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "mapping_rows": 2,
                "mapping_sha256": file_sha256(selections),
            }
        )
    )
    return candidates, selections


def test_compiler_publishes_and_recursively_revalidates_exact_pairs(tmp_path):
    candidates, selections = _artifacts(tmp_path)
    output = tmp_path / "measurement_resolution.parquet"
    manifest = compile_mapping(
        candidates,
        selections,
        output,
        require_complete=True,
        require_merged_selection=False,
    )
    assert manifest["generation_version"] == COMPILED_MAPPING_VERSION
    assert manifest["complete_candidate_inventory"] is True
    rows = {row["cleaned_record_id"]: row for row in pq.read_table(output).to_pylist()}
    assert json.loads(rows["a"]["measurements_json"]) == [
        {"measurement": "6", "unit": "mutation frequency"}
    ]
    assert rows["a"]["assignment_method"] == "model_single_pass"
    assert rows["b"]["status"] == "unsure"
    validate_compiled_mapping(output, require_complete=False)


def test_compiler_fails_before_output_on_unknown_candidate_id(tmp_path):
    candidates, selections = _artifacts(tmp_path)
    rows = pq.read_table(selections).to_pylist()
    raw = json.loads(rows[0]["raw_response_json"])
    raw["candidate_id"] = "9"
    raw["measurements"][0]["measurement"] = "9"
    rows[0]["raw_response_json"] = json.dumps(raw)
    rows[0]["measurements_json"] = json.dumps(raw["measurements"])
    _write_parquet(selections, rows)
    selections.with_suffix(".manifest.json").write_text(
        json.dumps({"mapping_rows": 2, "mapping_sha256": file_sha256(selections)})
    )
    output = tmp_path / "measurement_resolution.parquet"
    with pytest.raises(ValueError, match="selected candidate ID is absent"):
        compile_mapping(
            candidates,
            selections,
            output,
            require_complete=True,
            require_merged_selection=False,
        )
    assert not output.exists()


def test_subset_compilation_is_explicit_and_not_production_valid(tmp_path):
    candidates, selections = _artifacts(tmp_path)
    _write_parquet(selections, [_selection("a")])
    selections.with_suffix(".manifest.json").write_text(
        json.dumps({"mapping_rows": 1, "mapping_sha256": file_sha256(selections)})
    )
    output = tmp_path / "gold_resolution.parquet"
    manifest = compile_mapping(
        candidates,
        selections,
        output,
        require_complete=False,
        require_merged_selection=False,
    )
    assert manifest["complete_candidate_inventory"] is False
    validate_compiled_mapping(output, require_complete=False)
    with pytest.raises(ValueError, match="full inventory"):
        validate_compiled_mapping(output)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("task_id", "other"),
        ("mapping_path", "/tmp/wrong.parquet"),
        ("mapping_rows", 99),
        ("source_counts", {"fixed_mutation": 99}),
        ("validations", {"one_row_per_selection": True}),
        ("complete_candidate_inventory", False),
    ],
)
def test_compiled_mapping_rejects_false_manifest_claims(tmp_path, field, bad_value):
    candidates, selections = _artifacts(tmp_path)
    output = tmp_path / "measurement_resolution.parquet"
    compile_mapping(
        candidates,
        selections,
        output,
        require_complete=True,
        require_merged_selection=False,
    )
    manifest_path = output.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = bad_value
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_compiled_mapping(output, require_complete=False)
