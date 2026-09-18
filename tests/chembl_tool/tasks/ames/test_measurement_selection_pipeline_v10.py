from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.merge_measurement_resolution import (
    _load_partition_rows,
    _validate_partition_input,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.plan_measurement_resolution import (
    OPENAI_MODEL,
    _assignment_rows,
    _partition_input,
    _write_partition_inputs,
    candidate_inventory_reference,
    provider_contracts,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_candidate_resolution import (
    CANDIDATE_CONTRACT_VERSION,
    candidate_set_sha256,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_measurement_selection import (
    MAPPING_VERSION,
    prompt_manifest,
)


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _candidate(record_id: str, source_id: str) -> dict[str, object]:
    pairs = [
        {
            "candidate_id": "1",
            "measurement": "6",
            "unit": "mutation frequency",
            "rule_id": "named_readout.v1",
            "evidence": ["support_text: mutation frequency was 6"],
            "hints": [],
        }
    ]
    encoded = json.dumps(pairs, sort_keys=True)
    return {
        "id": record_id,
        "source_row_uid": f"source-{record_id}",
        "source_id": source_id,
        "canonical_endpoint_name": "microbial_forward_mutation",
        "measurement_candidates_json": encoded,
        "candidate_set_sha256": candidate_set_sha256(encoded),
    }


def _assignments(candidates: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    partitions = ("gpt_key_one", "gpt_key_two", "deepseek")
    return {
        str(row["id"]): {
            "partition_id": partitions[index],
            "planning_request_id": f"request-{index}",
            "planning_p95_actual_tokens": 10,
        }
        for index, row in enumerate(candidates)
    }


def _contracts() -> dict[str, dict[str, object]]:
    return provider_contracts()


def _selection(record_id: str, contract: dict[str, object]) -> dict[str, object]:
    measurements = [{"measurement": "1", "unit": "candidate_id"}]
    raw = {
        "id": "r1",
        "reason": "absolute",
        "candidate_id": "1",
        "status": "ok",
        "measurements": measurements,
    }
    return {
        "cleaned_record_id": record_id,
        "source_row_uid": f"source-{record_id}",
        "source_id": "fixed_mutation",
        "status": "ok",
        "measurements_json": json.dumps(measurements),
        "quantity_count": 1,
        "assignment_method": "model_single_pass",
        "raw_response_json": json.dumps(raw),
        "rejected_response_json": None,
        "inference_source": "delta_inference",
        "inference_model": contract["requested_model"],
        "returned_model": contract["expected_returned_model"],
        "inference_base_url": contract["base_url"],
        "inference_credential_env": contract["credential_env"],
        "requested_provider": contract["requested_provider_tag"],
        "served_provider": contract["expected_served_provider"],
        "api_response_id": "response-1",
    }


def _partition_case(tmp_path: Path) -> tuple[Path, dict, list, Path]:
    candidate = _candidate("a", "fixed_mutation")
    input_path = tmp_path / "inputs/gpt_key_one.parquet"
    mapping_path = tmp_path / "outputs/gpt_key_one/measurement_selection.parquet"
    _write_parquet(input_path, [_partition_input(candidate)])
    contract = _contracts()["gpt_key_one"]
    _write_parquet(mapping_path, [_selection("a", contract)])
    child_manifest = {
        "task_id": "ames",
        "mapping_version": MAPPING_VERSION,
        "mapping_rows": 1,
        "mapping_sha256": file_sha256(mapping_path),
        "cleaned_records_sha256": file_sha256(input_path),
        "model": OPENAI_MODEL,
        "models": [OPENAI_MODEL],
        "api_base_url": contract["base_url"],
        "api_base_urls": [contract["base_url"]],
        "inference_model_counts": {OPENAI_MODEL: 1},
        "credential_counts": {contract["credential_env"]: 1},
        "inference_base_url_counts": {contract["base_url"]: 1},
        "served_provider_counts": {"None": 1},
        "prompt": prompt_manifest(),
        "inference": {"max_completion_tokens": 8192, "reasoning_mode": "low"},
        "endpoint_profile": {"sha256": "profile-hash"},
        "api_usage_by_returned_model": {OPENAI_MODEL: {"responses_without_usage": 0}},
        "base_mapping": None,
    }
    mapping_path.with_suffix(".manifest.json").write_text(json.dumps(child_manifest))
    assignments = _assignment_rows(
        [candidate],
        {
            "a": {
                "partition_id": "gpt_key_one",
                "planning_request_id": "request-1",
                "planning_p95_actual_tokens": 10,
            }
        },
    )
    plan = {
        "partitions": {
            "gpt_key_one": {
                "rows": 1,
                "input_path": str(input_path.relative_to(tmp_path)),
                "input_sha256": file_sha256(input_path),
                "mapping_path": str(mapping_path.relative_to(tmp_path)),
            }
        },
        "provider_contracts": {"gpt_key_one": contract},
        "prompt": prompt_manifest(),
        "endpoint_profile_sha256": "profile-hash",
        "max_completion_tokens": 8192,
    }
    return tmp_path / "plan.json", plan, assignments, mapping_path


def test_plan_artifacts_carry_and_validate_candidate_set_hashes(tmp_path: Path) -> None:
    sources = ("fixed_mutation", "premutagenic_damage", "mutagenicity_mechanism")
    candidates = [
        _candidate(str(index), source) for index, source in enumerate(sources)
    ]
    assignments = _assignments(candidates)
    rows = _assignment_rows(candidates, assignments)
    specs = _write_partition_inputs(tmp_path, candidates, assignments)
    assert all(
        spec["mapping_path"].endswith("measurement_selection.parquet")
        for spec in specs.values()
    )
    assert {row["candidate_set_sha256"] for row in rows} == {
        row["candidate_set_sha256"] for row in candidates
    }
    for partition, spec in specs.items():
        _validate_partition_input(rows, partition, tmp_path / spec["input_path"])


def test_candidate_inventory_manifest_is_pinned(tmp_path: Path) -> None:
    path = tmp_path / "measurement_candidates.parquet"
    _write_parquet(path, [_partition_input(_candidate("a", "fixed_mutation"))])
    manifest_path = path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "inventory_version": CANDIDATE_CONTRACT_VERSION,
                "candidate_rows": 1,
                "candidate_sha256": file_sha256(path),
            }
        )
    )
    reference = candidate_inventory_reference(path, 1)
    assert reference["manifest_sha256"] == file_sha256(manifest_path)
    manifest_path.write_text(
        json.dumps({**json.loads(manifest_path.read_text()), "candidate_rows": 2})
    )
    with pytest.raises(ValueError, match="candidate inventory manifest mismatch"):
        candidate_inventory_reference(path, 1)


def test_partition_merge_enriches_only_valid_raw_candidate_selection(
    tmp_path: Path,
) -> None:
    plan_path, plan, assignments, _ = _partition_case(tmp_path)
    rows, _, _ = _load_partition_rows(plan_path, plan, assignments, "gpt_key_one")
    assert rows[0]["candidate_set_sha256"] == assignments[0]["candidate_set_sha256"]
    assert rows[0]["candidate_contract_version"] == CANDIDATE_CONTRACT_VERSION
    assert rows[0]["canonical_endpoint_name"] == "microbial_forward_mutation"


def test_partition_merge_rejects_unfrozen_candidate_id(tmp_path: Path) -> None:
    plan_path, plan, assignments, mapping_path = _partition_case(tmp_path)
    row = pq.read_table(mapping_path).to_pylist()[0]
    raw = json.loads(row["raw_response_json"])
    raw["candidate_id"] = "2"
    raw["measurements"][0]["measurement"] = "2"
    row["raw_response_json"] = json.dumps(raw)
    row["measurements_json"] = json.dumps(raw["measurements"])
    _write_parquet(mapping_path, [row])
    manifest_path = mapping_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(
        json.dumps({**manifest, "mapping_sha256": file_sha256(mapping_path)})
    )
    with pytest.raises(ValueError, match="selected candidate ID is absent"):
        _load_partition_rows(plan_path, plan, assignments, "gpt_key_one")
