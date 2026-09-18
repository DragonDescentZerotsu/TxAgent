"""Executable contracts for AMES prompting and frozen inference partitions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.build_measurement_resolution_mapping import (
    RequestBatch,
    TaskConfig,
)
from data.processing.evidence_library.versions.v10.tasks.ames import (
    starling_measurement_selection as config,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.build_token_projection import (
    build_projection,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.compile_candidate_resolution import (
    compile_mapping,
    validate_compiled_mapping,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.merge_measurement_resolution import (
    load_validated_plan,
    merge_partitions,
    validate_merged_mapping,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.plan_measurement_resolution import (
    ACTIVE_EXTRACTION_SOURCE_IDS,
    DEEPSEEK_ARCHITECTURE,
    DEEPSEEK_MODEL_TYPE,
    DEFAULT_MAX_COMPLETION_TOKENS,
    ENDPOINT_RECEIPT_VERSION,
    OPENAI_MINIMUM_HEADROOM,
    OPENAI_MODEL,
    OPENAI_PLANNING_TARGET,
    OPENAI_TOKEN_ALLOCATION,
    PARTITION_IDS,
    SELECTION_CONFIG_MODULE,
    TOKEN_PROJECTION_METRIC,
    TOKEN_PROJECTION_QUANTILE_METHOD,
    TOKEN_PROJECTION_VERSION,
    assign_batches,
    candidate_inventory_reference,
    load_token_projection,
    provider_contracts,
    validate_endpoint_receipt,
    write_partition_plan,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_candidate_resolution import (
    CANDIDATE_CONTRACT_VERSION,
    candidate_set_sha256,
)


def _candidate(index: int, source_id: str = "fixed_mutation") -> dict[str, Any]:
    pair = [
        {
            "candidate_id": "1",
            "measurement": str(index + 1),
            "unit": "mutants/cell",
            "rule_id": "source_pair.v1",
            "evidence": [f"support_text: selected response {index + 1} mutants/cell"],
            "hints": [],
        }
    ]
    encoded = json.dumps(pair, ensure_ascii=False, sort_keys=True)
    return {
        "id": f"record-{index}",
        "source_row_uid": f"source-{index}",
        "source_id": source_id,
        "canonical_endpoint_name": "hprt_hgprt_xprt_forward_mutation",
        "endpoint_name": "hprt_hgprt_xprt_forward_mutation",
        "measurement_text": str(index + 1),
        "unit_text": "mutants/cell",
        "support_text": f"The selected response was {index + 1} mutants/cell.",
        "endpoint_class": "mutation_frequency",
        "study_context": "cell assay",
        "biological_test_system": "mammalian cells",
        "result_call": "positive",
        "measurement_candidates_json": encoded,
        "candidate_set_sha256": candidate_set_sha256(encoded),
        "candidate_contract_version": CANDIDATE_CONTRACT_VERSION,
        "candidate_generation_disposition": "eligible_candidates",
        "candidate_count": 1,
    }


def _write_candidate_inventory(path: Path, candidates: list[dict[str, Any]]) -> None:
    rows = [
        {
            "cleaned_record_id": row["id"],
            **{key: value for key, value in row.items() if key != "id"},
        }
        for row in candidates
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    manifest = {
        "inventory_version": CANDIDATE_CONTRACT_VERSION,
        "candidate_rows": len(rows),
        "candidate_sha256": file_sha256(path),
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest) + "\n")


def _batches(candidates: list[dict[str, Any]]) -> list[RequestBatch]:
    return [
        RequestBatch(
            request_id=f"request-{index}",
            source_id=row["source_id"],
            rows=(row,),
            prompt="fixture prompt",
            max_completion_tokens=32,
        )
        for index, row in enumerate(candidates)
    ]


def _contracts() -> dict[str, dict[str, Any]]:
    return provider_contracts()


def _write_projection(tmp_path: Path, records_path: Path, profile_path: Path) -> Path:
    evidence = {}
    for label in ("cache", "mapping", "mapping_manifest", "gold_fixture"):
        evidence_path = tmp_path / f"gold_{label}.bin"
        evidence_path.write_text(f"{label}\n")
        evidence[label] = {
            "path": str(evidence_path),
            "sha256": file_sha256(evidence_path),
        }
    payload = {
        "projection_version": TOKEN_PROJECTION_VERSION,
        "task_id": "ames",
        "requested_model": OPENAI_MODEL,
        "returned_models": [OPENAI_MODEL],
        "batch_size": config.BATCH_SIZE,
        "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
        "cleaned_records_sha256": file_sha256(records_path),
        "endpoint_profile_sha256": file_sha256(profile_path),
        "usage_metric": TOKEN_PROJECTION_METRIC,
        "quantile": 0.95,
        "quantile_method": TOKEN_PROJECTION_QUANTILE_METHOD,
        "complete_gold_row_coverage": True,
        "complete_usage": True,
        "allow_fallbacks": False,
        "prompt": config.prompt_manifest(),
        "candidate_inventory": candidate_inventory_reference(
            records_path, pq.read_metadata(records_path).num_rows
        ),
        "gold_replay_evidence": evidence,
        "sources": {
            source_id: {"gold_request_count": 2, "p95_actual_tokens": 100}
            for source_id in ACTIVE_EXTRACTION_SOURCE_IDS
        },
    }
    path = tmp_path / "token_projection.json"
    path.write_text(json.dumps(payload) + "\n")
    return path


def _endpoint_receipt_payload() -> dict[str, Any]:
    return {
        "receipt_version": ENDPOINT_RECEIPT_VERSION,
        "checked_at_utc": "2026-09-08T00:00:00Z",
        "selected_endpoint": {
            "host": "dgx027",
            "provider": config.DEEPSEEK_PROVIDER,
            "base_url": config.DEEPSEEK_BASE_URL,
            "requested_model": config.DEEPSEEK_MODEL,
            "returned_model": config.DEEPSEEK_MODEL,
            "credential_env": config.DEEPSEEK_CREDENTIAL_ENV,
            "max_concurrency": config.ENDPOINT_CONCURRENCY_BUDGET,
        },
        "live_checks": {
            "generation": {
                "http_status": 200,
                "returned_model": config.DEEPSEEK_MODEL,
                "sha256": hashlib.sha256(b"fixture generation").hexdigest(),
            },
            "model_info": {
                "http_status": 200,
                "model_path": config.DEEPSEEK_MODEL,
                "served_model_name": config.DEEPSEEK_MODEL,
                "model_type": DEEPSEEK_MODEL_TYPE,
                "architectures": [DEEPSEEK_ARCHITECTURE],
                "sha256": hashlib.sha256(b"fixture model info").hexdigest(),
            },
            "models": {
                "http_status": 200,
                "ids": [config.DEEPSEEK_MODEL],
                "sha256": hashlib.sha256(b"fixture models").hexdigest(),
            },
        },
    }


def _frozen_plan(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    candidates = [
        _candidate(index, source_id)
        for index, source_id in enumerate(ACTIVE_EXTRACTION_SOURCE_IDS)
    ]
    batches = _batches(candidates)
    spent = dict.fromkeys(("gpt_key_one", "gpt_key_two"), OPENAI_PLANNING_TARGET - 100)
    projections = dict.fromkeys(ACTIVE_EXTRACTION_SOURCE_IDS, 100)
    assignments, projected = assign_batches(
        candidates,
        batches,
        spent_tokens=spent,
        source_p95_actual_tokens=projections,
    )
    records_path = tmp_path / "measurement_candidates.parquet"
    profile_path = tmp_path / "profile.json"
    _write_candidate_inventory(records_path, candidates)
    profile_path.write_text("{}\n")
    projection_path = _write_projection(tmp_path, records_path, profile_path)
    endpoint_receipt = tmp_path / "deepseek_endpoint_receipt.json"
    endpoint_receipt.write_text(json.dumps(_endpoint_receipt_payload()) + "\n")
    plan_path = write_partition_plan(
        tmp_path / "plan",
        candidates=candidates,
        assignments=assignments,
        projected_tokens=projected,
        spent_tokens=spent,
        contracts=_contracts(),
        records_path=records_path,
        profile_path=profile_path,
        prompt=config.prompt_manifest(),
        token_projection_path=projection_path,
        endpoint_receipt_path=endpoint_receipt,
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    )
    return plan_path, candidates


def _result_row(
    candidate: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    measurements = [{"measurement": "1", "unit": "candidate_id"}]
    raw = {
        "id": "r1",
        "reason": "absolute",
        "candidate_id": "1",
        "status": "ok",
        "measurements": measurements,
    }
    return {
        "cleaned_record_id": candidate["id"],
        "source_row_uid": candidate["source_row_uid"],
        "source_id": candidate["source_id"],
        "status": "ok",
        "measurements_json": json.dumps(measurements),
        "quantity_count": 1,
        "assignment_method": "model_single_pass",
        "raw_response_json": json.dumps(raw),
        "rejected_response_json": None,
        "api_response_id": f"response-{candidate['id']}",
        "inference_source": "delta_inference",
        "inference_model": contract["requested_model"],
        "returned_model": contract["expected_returned_model"],
        "requested_provider": contract["requested_provider_tag"],
        "served_provider": contract["expected_served_provider"],
        "inference_base_url": contract["base_url"],
        "inference_credential_env": contract["credential_env"],
    }


def _write_child(
    plan_path: Path,
    plan: Mapping[str, Any],
    assignments: list[Mapping[str, Any]],
    candidates: list[Mapping[str, Any]],
    partition: str,
    *,
    returned_model: str | None = None,
) -> None:
    spec, contract = (
        plan["partitions"][partition],
        plan["provider_contracts"][partition],
    )
    by_id = {row["id"]: row for row in candidates}
    selected = [row for row in assignments if row["partition_id"] == partition]
    rows = [_result_row(by_id[row["cleaned_record_id"]], contract) for row in selected]
    if returned_model is not None:
        rows[0]["returned_model"] = returned_model
    mapping_path = plan_path.parent / spec["mapping_path"]
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), mapping_path)
    manifest = {
        "task_id": "ames",
        "mapping_version": config.MAPPING_VERSION,
        "mapping_rows": len(rows),
        "mapping_sha256": file_sha256(mapping_path),
        "cleaned_records_sha256": spec["input_sha256"],
        "model": contract["requested_model"],
        "models": [contract["requested_model"]],
        "api_base_url": contract["base_url"],
        "api_base_urls": [contract["base_url"]],
        "inference_model_counts": {contract["requested_model"]: len(rows)},
        "credential_counts": {contract["credential_env"]: len(rows)},
        "inference_base_url_counts": {contract["base_url"]: len(rows)},
        "served_provider_counts": {
            str(contract["expected_served_provider"]): len(rows)
        },
        "prompt": plan["prompt"],
        "inference": {
            "max_completion_tokens": plan["max_completion_tokens"],
            "reasoning_mode": contract["reasoning_effort"],
        },
        "endpoint_profile": {"sha256": plan["endpoint_profile_sha256"]},
        "api_usage_by_returned_model": {
            contract["expected_returned_model"]: {"responses_without_usage": 0}
        },
        "base_mapping": None,
    }
    mapping_path.with_suffix(".manifest.json").write_text(json.dumps(manifest) + "\n")


def _write_children(
    plan_path: Path,
    candidates: list[dict[str, Any]],
    *,
    bad_deepseek_snapshot: bool = False,
) -> None:
    plan, assignments = load_validated_plan(plan_path)
    for partition in PARTITION_IDS:
        returned = (
            "wrong-snapshot"
            if bad_deepseek_snapshot and partition == "deepseek"
            else None
        )
        _write_child(
            plan_path,
            plan,
            assignments,
            candidates,
            partition,
            returned_model=returned,
        )


def _merged_case(tmp_path: Path) -> tuple[Path, Path]:
    plan_path, candidates = _frozen_plan(tmp_path)
    _write_children(plan_path, candidates)
    output = tmp_path / "merged" / "measurement_resolution.parquet"
    merge_partitions(plan_path, output)
    return plan_path, output


def _write_gold_fixture(path: Path, candidates: list[dict[str, Any]]) -> None:
    manifest = {
        "corpus_version": "ames_measurement_resolution_gold.v1",
        "task_id": "ames",
        "expectation_update_policy": "manual_source_review_only",
        "labelled_before_any_model_run": True,
        "cases": len(candidates),
    }
    cases = [
        {
            "audit_case_id": f"ames:{row['id']}",
            "source_row_uid": row["source_row_uid"],
            "source_id": row["source_id"],
            "canonical_endpoint_name": row["canonical_endpoint_name"],
            "input": {
                "source_id": row["source_id"],
                "canonical_endpoint_name": row["canonical_endpoint_name"],
            },
        }
        for row in candidates
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in [manifest, *cases]))


def _write_gold_mapping(
    mapping_path: Path,
    rows: list[dict[str, Any]],
    records_path: Path,
    profile_path: Path,
) -> None:
    pq.write_table(pa.Table.from_pylist(rows), mapping_path)
    manifest = {
        "task_id": "ames",
        "mapping_version": config.MAPPING_VERSION,
        "mapping_sha256": file_sha256(mapping_path),
        "mapping_rows": len(rows),
        "model": OPENAI_MODEL,
        "prompt": config.prompt_manifest(),
        "base_mapping": None,
        "inference": {
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "reasoning_mode": config.REASONING_EFFORT,
        },
        "cleaned_records_path": str(records_path),
        "cleaned_records_sha256": file_sha256(records_path),
        "endpoint_profile": {
            "path": str(profile_path),
            "sha256": file_sha256(profile_path),
        },
        "validations": {
            "one_row_per_candidate": True,
            "unique_cleaned_record_ids": True,
        },
    }
    mapping_path.with_suffix(".manifest.json").write_text(json.dumps(manifest) + "\n")


def _gold_cache_events(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for index, candidate in enumerate(candidates):
        shared = {
            "request_id": f"gold-request-{index}",
            "source_id": candidate["source_id"],
            "row_ids": [candidate["id"]],
        }
        events.extend(
            (
                {
                    "status": "submitted",
                    **shared,
                    "model": OPENAI_MODEL,
                    "base_url": "https://api.openai.com/v1",
                    "credential_env": "OPENAI_API_KEY_ONE",
                },
                {
                    "status": "api_response",
                    "request_id": shared["request_id"],
                    "returned_model": OPENAI_MODEL,
                    "requested_provider": None,
                    "served_provider": None,
                    "api_response_id": f"response-{candidate['id']}",
                    "response": {
                        "id": f"response-{candidate['id']}",
                        "model": OPENAI_MODEL,
                        "usage": {
                            "prompt_tokens": 100 + index,
                            "completion_tokens": 10,
                        },
                    },
                },
                {
                    "status": "terminal",
                    **shared,
                    "response_status": "valid",
                    "usage": {"input_tokens": 100 + index, "output_tokens": 10},
                },
            )
        )
    return events


def _gold_replay_files(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    records_path = tmp_path / "gold_candidates.parquet"
    profile_path = tmp_path / "gold_profile.json"
    profile_path.write_text("{}\n")
    candidates = [
        _candidate(index, source_id)
        for index, source_id in enumerate(ACTIVE_EXTRACTION_SOURCE_IDS)
    ]
    _write_candidate_inventory(records_path, candidates)
    gold_path = tmp_path / "ames_gold.jsonl"
    _write_gold_fixture(gold_path, candidates)
    rows = [
        _result_row(candidate, _contracts()["gpt_key_one"]) for candidate in candidates
    ]
    mapping_path = tmp_path / "gold_mapping.parquet"
    _write_gold_mapping(mapping_path, rows, records_path, profile_path)
    cache_path = tmp_path / "gold_cache.jsonl"
    events = _gold_cache_events(candidates)
    cache_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    return cache_path, mapping_path, records_path, profile_path, gold_path


def test_ames_config_loads_and_renders_all_sources() -> None:
    task = TaskConfig("ames", SELECTION_CONFIG_MODULE)
    manifest = task.prompt_manifest()
    assert set(manifest["rendered_sha256"]) == set(task.SOURCE_IDS)
    assert all(task.render_prompt(source_id).strip() for source_id in task.SOURCE_IDS)
    assert all(
        "extra_details" not in task.prompt_row_fields(source_id)
        for source_id in task.SOURCE_IDS
    )
    assert all(
        task.rules[source_id].require_positive_value is False
        for source_id in task.SOURCE_IDS
    )
    baseline = task.render_prompt(task.SOURCE_IDS[0])
    profiled = task.render_prompt(
        task.SOURCE_IDS[0], endpoint_profiles=("PROFILE-SENTINEL",)
    )
    assert profiled == baseline and "PROFILE-SENTINEL" not in profiled
    assert "measurement_candidates_json" in task.prompt_row_fields(task.SOURCE_IDS[0])


def test_provider_contract_pins_exact_gpt_and_local_deepseek_routes() -> None:
    contracts = _contracts()
    assert contracts["gpt_key_one"]["requested_model"] == OPENAI_MODEL
    assert contracts["gpt_key_one"]["expected_returned_model"] == OPENAI_MODEL
    assert contracts["gpt_key_one"]["credential_env"] == "OPENAI_API_KEY_ONE"
    local = contracts["deepseek"]
    assert local["provider"] == "local"
    assert local["base_url"] == "http://dgx027:50001/v1"
    assert local["requested_model"] == local["expected_returned_model"]
    assert local["requested_provider_tag"] is None
    assert local["expected_served_provider"] is None
    assert local["credential_env"] == ""
    assert local["max_concurrency"] == 512


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("base_url", "http://127.0.0.1:50001/v1"),
        ("provider", "openrouter"),
        ("returned_model", "wrong/model"),
        ("credential_env", "OPEN_ROUTER_KEY"),
        ("max_concurrency", 8),
    ),
)
def test_endpoint_receipt_requires_exact_local_host_and_returned_model(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = _endpoint_receipt_payload()
    payload["selected_endpoint"][field] = value
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match=field):
        validate_endpoint_receipt(path, _contracts()["deepseek"])


def test_selection_config_rejects_nonlocal_deepseek_route() -> None:
    args = SimpleNamespace(
        task="ames",
        config_module=SELECTION_CONFIG_MODULE,
        provider="local",
        base_url=config.DEEPSEEK_BASE_URL,
        model=config.DEEPSEEK_MODEL,
        api_key_env=None,
        workers=config.ENDPOINT_CONCURRENCY_BUDGET,
        max_completion_tokens=config.MAX_COMPLETION_TOKENS,
        provider_only=None,
        no_token_ledger=True,
        two_key_baidu_run=False,
        require_complete=True,
    )
    config.validate_generation_args(args)
    args.provider = "openrouter"
    args.base_url = "https://openrouter.ai/api/v1"
    args.model = "deepseek/deepseek-v4-flash-0731"
    args.api_key_env = "OPEN_ROUTER_KEY"
    with pytest.raises(SystemExit, match="AMES V10 generation contract mismatch"):
        config.validate_generation_args(args)


def test_frozen_plan_is_disjoint_exhaustive_and_budgeted(tmp_path: Path) -> None:
    plan_path, _ = _frozen_plan(tmp_path)
    plan, rows = load_validated_plan(plan_path)
    assert {row["partition_id"] for row in rows} == set(PARTITION_IDS)
    assert (
        len({row["cleaned_record_id"] for row in rows}) == plan["candidate_rows"] == 3
    )
    assert (
        "mutagenicity_outcomes"
        not in plan["token_projection"]["source_p95_actual_tokens"]
    )
    for partition in ("gpt_key_one", "gpt_key_two"):
        spec = plan["partitions"][partition]
        assert spec["projected_total_tokens"] <= OPENAI_PLANNING_TARGET
        assert spec["runtime_ledger_max_tokens"] == OPENAI_TOKEN_ALLOCATION
        assert spec["minimum_unplanned_headroom_tokens"] == OPENAI_MINIMUM_HEADROOM


def test_projection_builder_uses_complete_actual_usage_without_outcomes(
    tmp_path: Path,
) -> None:
    cache_path, mapping_path, records_path, profile_path, gold_path = (
        _gold_replay_files(tmp_path)
    )
    output_path = tmp_path / "projection.json"
    payload = build_projection(cache_path, mapping_path, output_path, gold_path)
    loaded = load_token_projection(
        output_path,
        records_path=records_path,
        profile_path=profile_path,
        prompt=config.prompt_manifest(),
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    )
    assert loaded == payload
    assert set(payload["sources"]) == set(ACTIVE_EXTRACTION_SOURCE_IDS)
    assert "mutagenicity_outcomes" not in payload["sources"]
    assert payload["sources"]["fixed_mutation"]["p95_actual_tokens"] == 110


def test_projection_rejects_gold_identity_drift(tmp_path: Path) -> None:
    cache, mapping, _, _, gold = _gold_replay_files(tmp_path)
    lines = [json.loads(line) for line in gold.read_text().splitlines()]
    lines[1]["source_row_uid"] = "different-source-row"
    gold.write_text("".join(json.dumps(row) + "\n" for row in lines))
    with pytest.raises(ValueError, match="gold candidate identity mismatch"):
        build_projection(cache, mapping, tmp_path / "projection.json", gold)


def test_projection_rejects_missing_actual_usage(tmp_path: Path) -> None:
    cache, mapping, _, _, gold = _gold_replay_files(tmp_path)
    events = [json.loads(line) for line in cache.read_text().splitlines()]
    next(event for event in events if event["status"] == "terminal").pop("usage")
    cache.write_text("".join(json.dumps(event) + "\n" for event in events))
    with pytest.raises(TypeError, match="lacks reported usage"):
        build_projection(cache, mapping, tmp_path / "projection.json", gold)


def test_projection_rejects_wrong_gpt_snapshot(tmp_path: Path) -> None:
    cache, mapping, _, _, gold = _gold_replay_files(tmp_path)
    rows = pq.read_table(mapping).to_pylist()
    rows[0]["returned_model"] = "gpt-5.4-mini"
    pq.write_table(pa.Table.from_pylist(rows), mapping)
    manifest_path = mapping.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["mapping_sha256"] = file_sha256(mapping)
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="gold mapping provenance mismatch"):
        build_projection(cache, mapping, tmp_path / "projection.json", gold)


def test_projection_revalidates_evidence_hashes(tmp_path: Path) -> None:
    cache, mapping, records, profile, gold = _gold_replay_files(tmp_path)
    output = tmp_path / "projection.json"
    build_projection(cache, mapping, output, gold)
    cache.write_text(cache.read_text() + "\n")
    with pytest.raises(ValueError, match="gold replay cache hash mismatch"):
        load_token_projection(
            output,
            records_path=records,
            profile_path=profile,
            prompt=config.prompt_manifest(),
            max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
        )


def test_strict_merge_accepts_only_the_frozen_provenance(tmp_path: Path) -> None:
    plan_path, candidates = _frozen_plan(tmp_path)
    _write_children(plan_path, candidates)
    output = tmp_path / "merged" / "measurement_resolution.parquet"
    manifest = merge_partitions(plan_path, output)
    assert manifest["mapping_rows"] == 3
    assert manifest["served_provider_counts"] == {"direct_openai": 2, "local": 1}
    assert all(manifest["validations"].values())
    validate_merged_mapping(output)


@pytest.mark.parametrize(
    ("field", "tampered"),
    (
        ("mapping_path", "relative/measurement_resolution.parquet"),
        ("mapping_sha256", "0" * 64),
        ("mapping_rows", 0),
        ("model", "gpt-5.4-mini-2026-03-17"),
        ("models", []),
        ("inference_model_counts", {}),
        ("credential_counts", {}),
        ("served_provider_counts", {}),
        ("status_counts", {}),
        ("rejected_rows", 1),
        ("base_mapping", {"path": "invented.parquet"}),
        ("validations", {}),
        ("cleaned_records_path", "invented.parquet"),
        ("cleaned_records_sha256", "0" * 64),
        ("candidate_inventory", {}),
        ("candidate_identity_sha256", "0" * 64),
        ("prompt", {}),
        ("provider_contracts", {}),
        ("partition_plan", {}),
        ("partition_manifests", []),
    ),
)
def test_merged_manifest_claims_are_recomputed(
    tmp_path: Path, field: str, tampered: object
) -> None:
    _, output = _merged_case(tmp_path)
    manifest_path = output.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = tampered
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError):
        validate_merged_mapping(output)


def test_refreshed_compiled_parent_rejects_tampered_merged_claims(
    tmp_path: Path,
) -> None:
    plan_path, merged = _merged_case(tmp_path)
    plan = json.loads(plan_path.read_text())
    compiled = tmp_path / "compiled" / "measurement_resolution.parquet"
    compile_mapping(
        Path(plan["candidate_inventory"]["path"]),
        merged,
        compiled,
        require_complete=True,
        require_merged_selection=False,
    )
    child_manifest_path = merged.with_suffix(".manifest.json")
    child_manifest = json.loads(child_manifest_path.read_text())
    child_manifest["status_counts"] = {"ok": 999}
    child_manifest_path.write_text(json.dumps(child_manifest) + "\n")
    parent_path = compiled.with_suffix(".manifest.json")
    parent = json.loads(parent_path.read_text())
    parent["raw_selection_mapping"]["manifest_sha256"] = file_sha256(
        child_manifest_path
    )
    parent["merged_selection_validated"] = True
    parent_path.write_text(json.dumps(parent) + "\n")
    with pytest.raises(ValueError, match="merged mapping summary mismatch"):
        validate_compiled_mapping(compiled, require_complete=False)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("returned_model", "wrong/model"),
        ("inference_base_url", "https://openrouter.ai/api/v1"),
        ("inference_credential_env", "OPEN_ROUTER_KEY"),
        ("requested_provider", "open-inference/fp8"),
        ("served_provider", "OpenInference"),
    ),
)
def test_strict_merge_rejects_deepseek_route_drift(
    tmp_path: Path, field: str, value: str
) -> None:
    plan_path, candidates = _frozen_plan(tmp_path)
    _write_children(plan_path, candidates)
    plan = json.loads(plan_path.read_text())
    mapping = plan_path.parent / plan["partitions"]["deepseek"]["mapping_path"]
    rows = pq.read_table(mapping).to_pylist()
    rows[0][field] = value
    pq.write_table(pa.Table.from_pylist(rows), mapping)
    manifest_path = mapping.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["mapping_sha256"] = file_sha256(mapping)
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="result provenance mismatch"):
        merge_partitions(plan_path, tmp_path / "merged.parquet")


def test_strict_merge_rejects_child_manifest_route_drift(tmp_path: Path) -> None:
    plan_path, candidates = _frozen_plan(tmp_path)
    _write_children(plan_path, candidates)
    plan = json.loads(plan_path.read_text())
    mapping = plan_path.parent / plan["partitions"]["deepseek"]["mapping_path"]
    manifest_path = mapping.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["api_base_url"] = "http://127.0.0.1:50001/v1"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="api_base_url"):
        merge_partitions(plan_path, tmp_path / "merged.parquet")
