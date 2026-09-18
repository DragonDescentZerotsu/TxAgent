"""Strictly merge the three frozen AMES raw candidate-selection partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.plan_measurement_resolution import (
    DEEPSEEK_BASE_URL,
    GPT_PARTITIONS,
    OPENAI_MINIMUM_HEADROOM,
    OPENAI_PLANNING_TARGET,
    OPENAI_TOKEN_ALLOCATION,
    PARTITION_IDS,
    PLAN_VERSION,
    SELECTION_CONFIG_MODULE,
    _projection_reference,
    candidate_inventory_reference,
    load_token_projection,
    provider_contracts,
    validate_endpoint_receipt,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_candidate_resolution import (
    CANDIDATE_CONTRACT_VERSION,
    candidate_set_sha256,
    compile_selection,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_measurement_selection import (
    DEFAULT_MAPPING_PATH,
    MAPPING_VERSION,
    prompt_manifest,
)

MERGE_VERSION = "ames_candidate_selection_merge.v2"
TERMINAL_STATUSES = frozenset({"ok", "relative", "unsure", "unavailable"})
SUCCESS_METHODS = frozenset({"model_single_pass", "model_structural_retry"})


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _digest_identities(rows: Sequence[Mapping[str, Any]]) -> str:
    identities = sorted(
        (
            row["cleaned_record_id"],
            row["source_row_uid"],
            row["source_id"],
            row["canonical_endpoint_name"],
            row["candidate_set_sha256"],
        )
        for row in rows
    )
    payload = json.dumps(identities, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_contracts(contracts: Mapping[str, Mapping[str, Any]]) -> None:
    if set(contracts) != set(PARTITION_IDS):
        raise ValueError("provider contract partitions differ from the frozen contract")
    expected = provider_contracts()
    if {key: dict(value) for key, value in contracts.items()} != expected:
        raise ValueError("provider contracts do not match the exact frozen backends")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_validated_plan(plan_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a plan and verify every source, prompt, assignment, and input hash."""
    plan = _read_json(plan_path)
    if (
        plan.get("plan_version") != PLAN_VERSION
        or plan.get("task_id") != "ames"
        or plan.get("mapping_version") != MAPPING_VERSION
        or plan.get("artifact_role") != "raw_candidate_selection"
        or plan.get("selection_config_module") != SELECTION_CONFIG_MODULE
    ):
        raise ValueError("inference assignment plan identity mismatch")
    root = plan_path.parent
    records_path = _validate_candidate_reference(plan)
    profile_path = Path(str(plan["endpoint_profile_path"]))
    assignment_path = _resolve(root, str(plan["assignment_path"]))
    endpoint_receipt = plan.get("deepseek_endpoint_receipt") or {}
    endpoint_receipt_path = _resolve(root, str(endpoint_receipt.get("path") or ""))
    checks = (
        (records_path, plan["cleaned_records_sha256"]),
        (profile_path, plan["endpoint_profile_sha256"]),
        (assignment_path, plan["assignment_sha256"]),
        (endpoint_receipt_path, endpoint_receipt.get("sha256")),
    )
    if any(
        not path.is_file() or file_sha256(path) != digest for path, digest in checks
    ):
        raise ValueError("plan input or assignment hash mismatch")
    if plan.get("prompt") != prompt_manifest(
        batch_size=int(plan["prompt"]["batch_size"])
    ):
        raise ValueError("plan prompt contract drift")
    _validate_token_projection(plan, records_path, profile_path)
    _validate_contracts(plan.get("provider_contracts") or {})
    validate_endpoint_receipt(
        endpoint_receipt_path, plan["provider_contracts"]["deepseek"]
    )
    assignments = pq.read_table(assignment_path).to_pylist()
    _validate_assignment_inventory(plan, assignments, root)
    _validate_candidate_rows(plan, assignments, records_path)
    return plan, assignments


def _validate_candidate_reference(plan: Mapping[str, Any]) -> Path:
    reference = plan.get("candidate_inventory") or {}
    path = Path(str(reference.get("path") or ""))
    expected = candidate_inventory_reference(path, int(plan.get("candidate_rows") or 0))
    aliases = {
        "cleaned_records_path": str(path),
        "cleaned_records_sha256": expected["sha256"],
    }
    mismatches = {key for key, value in aliases.items() if plan.get(key) != value}
    if dict(reference) != expected or mismatches:
        raise ValueError("candidate inventory reference mismatch")
    return path


def _validate_candidate_rows(
    plan: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    columns = (
        "cleaned_record_id",
        "source_row_uid",
        "source_id",
        "canonical_endpoint_name",
        "measurement_candidates_json",
        "candidate_set_sha256",
    )
    if not set(columns) <= set(pq.read_schema(path).names):
        raise ValueError("candidate inventory lacks identity or candidate-set fields")
    rows = pq.read_table(path, columns=list(columns)).to_pylist()
    for row in rows:
        expected = str(row.get("candidate_set_sha256") or "")
        if candidate_set_sha256(row.get("measurement_candidates_json")) != expected:
            raise ValueError(
                "candidate inventory contains a candidate-set hash mismatch"
            )
    if (
        len(rows) != len(assignments)
        or _digest_identities(rows) != plan["candidate_identity_sha256"]
    ):
        raise ValueError("candidate inventory and assignments differ")


def _validate_token_projection(
    plan: Mapping[str, Any], records_path: Path, profile_path: Path
) -> None:
    reference = plan.get("token_projection") or {}
    path = Path(str(reference.get("path") or ""))
    if not path.is_file() or file_sha256(path) != reference.get("sha256"):
        raise ValueError("token projection receipt hash mismatch")
    payload = load_token_projection(
        path,
        records_path=records_path,
        profile_path=profile_path,
        prompt=plan["prompt"],
        max_completion_tokens=int(plan["max_completion_tokens"]),
    )
    if reference != _projection_reference(path, payload):
        raise ValueError("token projection summary drift")


def _validate_assignment_inventory(
    plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], root: Path
) -> None:
    ids = [str(row["cleaned_record_id"]) for row in rows]
    if len(ids) != len(set(ids)) or len(ids) != int(plan["candidate_rows"]):
        raise ValueError("assignment ids are duplicated or incomplete")
    if _digest_identities(rows) != plan["candidate_identity_sha256"]:
        raise ValueError("assignment identity digest mismatch")
    counts = Counter(str(row["partition_id"]) for row in rows)
    if set(counts) != set(PARTITION_IDS):
        raise ValueError("assignment does not contain all three partitions")
    for partition in PARTITION_IDS:
        spec = plan["partitions"][partition]
        input_path = _resolve(root, str(spec["input_path"]))
        selected = [row for row in rows if row["partition_id"] == partition]
        if counts[partition] != int(spec["rows"]):
            raise ValueError(f"partition count mismatch: {partition}")
        if _digest_identities(selected) != spec.get("candidate_identity_sha256"):
            raise ValueError(f"partition candidate identity mismatch: {partition}")
        if not input_path.is_file() or file_sha256(input_path) != spec["input_sha256"]:
            raise ValueError(f"partition input hash mismatch: {partition}")
        _validate_partition_input(rows, partition, input_path)
    for partition in GPT_PARTITIONS:
        spec = plan["partitions"][partition]
        expected = {
            "planning_target_tokens": OPENAI_PLANNING_TARGET,
            "runtime_ledger_max_tokens": OPENAI_TOKEN_ALLOCATION,
            "minimum_unplanned_headroom_tokens": OPENAI_MINIMUM_HEADROOM,
        }
        if any(int(spec.get(key) or 0) != value for key, value in expected.items()):
            raise ValueError(f"OpenAI budget contract mismatch: {partition}")
        spent = int(spec["tokens_spent_before_plan"])
        projected = int(spec["projected_new_tokens"])
        if int(spec["projected_total_tokens"]) != spent + projected:
            raise ValueError(f"OpenAI projection arithmetic mismatch: {partition}")
        if spent + projected > OPENAI_PLANNING_TARGET:
            raise ValueError(f"OpenAI planning target exceeded: {partition}")
    _validate_assignment_projections(plan, rows)


def _validate_partition_input(
    assignments: Sequence[Mapping[str, Any]], partition: str, input_path: Path
) -> None:
    expected = {
        str(row["cleaned_record_id"]): (
            row["source_row_uid"],
            row["source_id"],
            row["canonical_endpoint_name"],
            row["candidate_set_sha256"],
        )
        for row in assignments
        if row["partition_id"] == partition
    }
    input_rows = pq.read_table(input_path).to_pylist()
    found = {
        str(row.get("cleaned_record_id") or ""): (
            row.get("source_row_uid"),
            row.get("source_id"),
            row.get("canonical_endpoint_name"),
            row.get("candidate_set_sha256"),
        )
        for row in input_rows
    }
    for row in input_rows:
        expected_hash = str(row.get("candidate_set_sha256") or "")
        if (
            candidate_set_sha256(row.get("measurement_candidates_json"))
            != expected_hash
        ):
            raise ValueError(f"partition candidate-set hash mismatch: {partition}")
    if len(found) != len(input_rows) or found != expected:
        raise ValueError(f"partition input identity mismatch: {partition}")


def _validate_assignment_projections(
    plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    requests: dict[str, tuple[str, int]] = {}
    for row in rows:
        request_id = str(row.get("planning_request_id") or "")
        value = (
            str(row["partition_id"]),
            int(row.get("planning_p95_actual_tokens") or 0),
        )
        if not request_id or value[1] < 1:
            raise ValueError("assignment lacks a positive request projection")
        if request_id in requests and requests[request_id] != value:
            raise ValueError(f"request projection is inconsistent: {request_id}")
        requests[request_id] = value
    totals = Counter()
    for partition, tokens in requests.values():
        if partition in GPT_PARTITIONS:
            totals[partition] += tokens
    for partition in GPT_PARTITIONS:
        if totals[partition] != int(
            plan["partitions"][partition]["projected_new_tokens"]
        ):
            raise ValueError(f"partition projection mismatch: {partition}")


def _mapping_manifest(mapping_path: Path) -> dict[str, Any]:
    manifest_path = mapping_path.with_suffix(".manifest.json")
    if not mapping_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"partition output is incomplete: {mapping_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("mapping_sha256") != file_sha256(mapping_path):
        raise ValueError(f"partition mapping hash mismatch: {mapping_path}")
    return manifest


def _validate_partition_manifest(
    partition: str,
    spec: Mapping[str, Any],
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    expected = {
        "task_id": "ames",
        "mapping_version": MAPPING_VERSION,
        "mapping_rows": int(spec["rows"]),
        "cleaned_records_sha256": spec["input_sha256"],
        "model": contract["requested_model"],
        "models": [contract["requested_model"]],
        "api_base_url": contract["base_url"],
        "api_base_urls": [contract["base_url"]],
        "inference_model_counts": {contract["requested_model"]: int(spec["rows"])},
        "credential_counts": {contract["credential_env"]: int(spec["rows"])},
        "inference_base_url_counts": {contract["base_url"]: int(spec["rows"])},
        "served_provider_counts": {
            str(contract["expected_served_provider"]): int(spec["rows"])
        },
    }
    mismatches = {key for key, value in expected.items() if manifest.get(key) != value}
    if manifest.get("prompt") != plan["prompt"]:
        mismatches.add("prompt")
    inference = manifest.get("inference") or {}
    if int(inference.get("max_completion_tokens") or 0) != int(
        plan["max_completion_tokens"]
    ):
        mismatches.add("max_completion_tokens")
    if inference.get("reasoning_mode") != contract["reasoning_effort"]:
        mismatches.add("reasoning_mode")
    endpoint_profile = manifest.get("endpoint_profile") or {}
    if endpoint_profile.get("sha256") != plan["endpoint_profile_sha256"]:
        mismatches.add("endpoint_profile")
    usage = manifest.get("api_usage_by_returned_model") or {}
    if set(usage) != {contract["expected_returned_model"]}:
        mismatches.add("returned_model_usage")
    elif int(
        usage[contract["expected_returned_model"]].get("responses_without_usage") or 0
    ):
        mismatches.add("incomplete_usage")
    if manifest.get("base_mapping") is not None:
        mismatches.add("base_mapping")
    if mismatches:
        raise ValueError(f"{partition} manifest mismatch: {sorted(mismatches)}")


def _optional_text(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def _validate_measurement_shape(row: Mapping[str, Any]) -> None:
    status = str(row.get("status") or "")
    if status not in TERMINAL_STATUSES:
        raise ValueError(
            f"nonterminal status for {row.get('cleaned_record_id')}: {status!r}"
        )
    try:
        measurements = json.loads(str(row.get("measurements_json") or "[]"))
    except json.JSONDecodeError as error:
        raise ValueError("invalid measurements_json") from error
    expected_count = 1 if status == "ok" else 0
    if (
        len(measurements) != expected_count
        or int(row.get("quantity_count") or 0) != expected_count
    ):
        raise ValueError(f"status/measurement mismatch: {row.get('cleaned_record_id')}")
    if expected_count and (
        not str(measurements[0].get("measurement") or "").strip()
        or not str(measurements[0].get("unit") or "").strip()
    ):
        raise ValueError(f"incomplete resolved pair: {row.get('cleaned_record_id')}")


def _validate_result_row(
    row: Mapping[str, Any], expected: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    record_id = str(row.get("cleaned_record_id") or "")
    exact = {
        "source_row_uid": expected["source_row_uid"],
        "source_id": expected["source_id"],
        "inference_source": "delta_inference",
        "inference_model": contract["requested_model"],
        "returned_model": contract["expected_returned_model"],
        "inference_base_url": contract["base_url"],
        "inference_credential_env": contract["credential_env"],
    }
    mismatches = {key for key, value in exact.items() if row.get(key) != value}
    optional = {
        "requested_provider": contract["requested_provider_tag"],
        "served_provider": contract["expected_served_provider"],
    }
    mismatches.update(
        key for key, value in optional.items() if _optional_text(row.get(key)) != value
    )
    carried = {
        "candidate_set_sha256": expected["candidate_set_sha256"],
        "candidate_contract_version": CANDIDATE_CONTRACT_VERSION,
        "canonical_endpoint_name": expected["canonical_endpoint_name"],
    }
    mismatches.update(
        key
        for key, value in carried.items()
        if row.get(key) is not None and row.get(key) != value
    )
    if record_id != expected["cleaned_record_id"] or mismatches:
        raise ValueError(
            f"result provenance mismatch for {record_id}: {sorted(mismatches)}"
        )
    if not row.get("api_response_id") or not row.get("raw_response_json"):
        raise ValueError(f"result lacks raw API provenance: {record_id}")
    if (
        row.get("rejected_response_json")
        or row.get("assignment_method") not in SUCCESS_METHODS
    ):
        raise ValueError(f"result contains a terminal inference failure: {record_id}")
    _validate_measurement_shape(row)


def _load_partition_rows(
    plan_path: Path,
    plan: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    partition: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    spec = plan["partitions"][partition]
    contract = plan["provider_contracts"][partition]
    mapping_path = _resolve(plan_path.parent, str(spec["mapping_path"]))
    manifest = _mapping_manifest(mapping_path)
    _validate_partition_manifest(partition, spec, contract, manifest, plan)
    expected = {
        str(row["cleaned_record_id"]): row
        for row in assignments
        if row["partition_id"] == partition
    }
    rows = pq.read_table(mapping_path).to_pylist()
    input_path = _resolve(plan_path.parent, str(spec["input_path"]))
    candidate_rows = pq.read_table(input_path).to_pylist()
    candidate_by_id = {str(row["cleaned_record_id"]): row for row in candidate_rows}
    found = [str(row.get("cleaned_record_id") or "") for row in rows]
    if len(found) != len(set(found)) or set(found) != set(expected):
        raise ValueError(f"{partition} result coverage mismatch")
    enriched = []
    for row in rows:
        record_id = str(row["cleaned_record_id"])
        assignment = expected[record_id]
        _validate_result_row(row, assignment, contract)
        compile_selection(
            row,
            {
                **candidate_by_id[record_id],
                "candidate_contract_version": CANDIDATE_CONTRACT_VERSION,
            },
        )
        enriched.append(
            {
                **row,
                "candidate_set_sha256": assignment["candidate_set_sha256"],
                "candidate_contract_version": CANDIDATE_CONTRACT_VERSION,
                "canonical_endpoint_name": assignment["canonical_endpoint_name"],
            }
        )
    return enriched, manifest, mapping_path


def _arrow_table(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    columns = sorted(set().union(*(row.keys() for row in rows)))
    return pa.Table.from_pylist(
        [{key: row.get(key) for key in columns} for row in rows]
    )


def _row_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = sorted(set().union(*(row.keys() for row in rows)))
    normalized = [
        {key: row.get(key) for key in columns}
        for row in sorted(rows, key=lambda item: str(item["cleaned_record_id"]))
    ]
    payload = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _counts(
    rows: Sequence[Mapping[str, Any]], field: str, *, empty: str = ""
) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) or empty) for row in rows).items()))


def _merged_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    models = sorted({str(row.get("inference_model") or "") for row in rows})
    rejected = sum(
        bool(row.get("rejected_response_json"))
        or row.get("assignment_method") not in SUCCESS_METHODS
        for row in rows
    )
    return {
        "model": "mixed",
        "models": models,
        "inference_model_counts": _counts(rows, "inference_model"),
        "credential_counts": _counts(rows, "inference_credential_env"),
        "served_provider_counts": dict(
            sorted(
                Counter(
                    str(
                        row.get("served_provider")
                        or (
                            "local"
                            if row.get("inference_base_url") == DEEPSEEK_BASE_URL
                            else "direct_openai"
                        )
                    )
                    for row in rows
                ).items()
            )
        ),
        "status_counts": _counts(rows, "status"),
        "rejected_rows": rejected,
        "base_mapping": None,
    }


def _rows_match_assignments(
    rows: Sequence[Mapping[str, Any]], assignments: Sequence[Mapping[str, Any]]
) -> bool:
    expected = {str(row["cleaned_record_id"]): row for row in assignments}
    fields = (
        "source_row_uid",
        "source_id",
        "canonical_endpoint_name",
        "candidate_set_sha256",
    )
    return len(expected) == len(assignments) and all(
        str(row.get("cleaned_record_id") or "") in expected
        and all(
            row.get(field) == expected[str(row["cleaned_record_id"])][field]
            for field in fields
        )
        for row in rows
    )


def _providers_match_plan(
    rows: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> bool:
    assignments_by_id = {str(row["cleaned_record_id"]): row for row in assignments}
    for row in rows:
        assignment = assignments_by_id.get(str(row.get("cleaned_record_id") or ""))
        if assignment is None:
            return False
        contract = plan["provider_contracts"][assignment["partition_id"]]
        exact = {
            "inference_model": contract["requested_model"],
            "returned_model": contract["expected_returned_model"],
            "inference_base_url": contract["base_url"],
            "inference_credential_env": contract["credential_env"],
        }
        if any(row.get(key) != value for key, value in exact.items()):
            return False
        if (
            _optional_text(row.get("requested_provider"))
            != contract["requested_provider_tag"]
            or _optional_text(row.get("served_provider"))
            != contract["expected_served_provider"]
        ):
            return False
    return True


def _measurement_shapes_are_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    try:
        for row in rows:
            _validate_measurement_shape(row)
    except (TypeError, ValueError):
        return False
    return True


def _candidate_selections_compile(
    rows: Sequence[Mapping[str, Any]], candidates: Mapping[str, Mapping[str, Any]]
) -> bool:
    try:
        for row in rows:
            candidate = candidates[str(row["cleaned_record_id"])]
            compile_selection(
                row,
                {
                    **candidate,
                    "candidate_contract_version": CANDIDATE_CONTRACT_VERSION,
                },
            )
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _merged_validations(
    rows: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> dict[str, bool]:
    ids = [str(row.get("cleaned_record_id") or "") for row in rows]
    candidate_path = Path(str(plan["candidate_inventory"]["path"]))
    candidate_rows = pq.read_table(candidate_path).to_pylist()
    candidates = {str(row["cleaned_record_id"]): row for row in candidate_rows}
    assignment_ids = {str(row["cleaned_record_id"]) for row in assignments}
    provider_exact = _providers_match_plan(rows, assignments, plan)
    return {
        "one_row_per_candidate": len(rows) == int(plan["candidate_rows"]),
        "unique_cleaned_record_ids": "" not in ids and len(ids) == len(set(ids)),
        "only_ok_carries_measurements": _measurement_shapes_are_valid(rows),
        "partition_assignment_exact": _rows_match_assignments(rows, assignments),
        "provider_and_snapshot_exact": provider_exact,
        "candidate_inventory_exact": set(ids) == set(candidates) == assignment_ids,
        "candidate_set_hashes_carried": all(
            row.get("candidate_set_sha256")
            == candidates.get(str(row.get("cleaned_record_id") or ""), {}).get(
                "candidate_set_sha256"
            )
            for row in rows
        ),
        "candidate_selections_compile": _candidate_selections_compile(rows, candidates),
        "no_fallbacks": provider_exact
        and all(row.get("inference_source") == "delta_inference" for row in rows)
        and all(
            contract.get("allow_fallbacks") is False
            for contract in plan["provider_contracts"].values()
        ),
    }


def _load_all_partitions(
    plan_path: Path,
    plan: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[str, Mapping[str, Any], Path]]]:
    rows: list[dict[str, Any]] = []
    children = []
    for partition in PARTITION_IDS:
        child_rows, manifest, path = _load_partition_rows(
            plan_path, plan, assignments, partition
        )
        rows.extend(child_rows)
        children.append((partition, manifest, path))
    rows.sort(key=lambda row: str(row["cleaned_record_id"]))
    return rows, children


def _child_references(
    children: Sequence[tuple[str, Mapping[str, Any], Path]],
) -> list[dict[str, Any]]:
    return [
        {
            "partition_id": key,
            "path": str(path.with_suffix(".manifest.json").resolve()),
            "sha256": file_sha256(path.with_suffix(".manifest.json")),
            "mapping_sha256": child["mapping_sha256"],
        }
        for key, child, path in children
    ]


def _write_merged(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    manifest_path = path.with_suffix(".manifest.json")
    if path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to replace frozen merged mapping: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(_arrow_table(rows), temporary)
    temporary.replace(path)


def _merged_manifest(
    output_path: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
    children: Sequence[tuple[str, Mapping[str, Any], Path]],
) -> dict[str, Any]:
    return {
        "generation_version": MERGE_VERSION,
        "task_id": "ames",
        "artifact_role": "raw_candidate_selection",
        "mapping_version": MAPPING_VERSION,
        "mapping_path": str(output_path.resolve()),
        "mapping_sha256": file_sha256(output_path),
        "mapping_rows": len(rows),
        "merged_rows_sha256": _row_digest(rows),
        "cleaned_records_path": plan["cleaned_records_path"],
        "cleaned_records_sha256": plan["cleaned_records_sha256"],
        "candidate_inventory": plan["candidate_inventory"],
        "candidate_identity_sha256": plan["candidate_identity_sha256"],
        "prompt": plan["prompt"],
        **_merged_summary(rows),
        "partition_plan": {
            "path": str(plan_path.resolve()),
            "sha256": file_sha256(plan_path),
            "version": PLAN_VERSION,
            "assignment_sha256": plan["assignment_sha256"],
        },
        "provider_contracts": plan["provider_contracts"],
        "partition_manifests": _child_references(children),
        "validations": _merged_validations(rows, assignments, plan),
    }


def merge_partitions(plan_path: Path, output_path: Path) -> dict[str, Any]:
    """Validate each partition, write one mapping, and retain child lineage."""
    plan, assignments = load_validated_plan(plan_path)
    rows, children = _load_all_partitions(plan_path, plan, assignments)
    if len(rows) != int(plan["candidate_rows"]):
        raise ValueError("merged result does not cover the frozen candidate inventory")
    _write_merged(output_path, rows)
    manifest = _merged_manifest(
        output_path, plan_path, plan, rows, assignments, children
    )
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _validate_merged_summary(
    mapping_path: Path,
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> None:
    validations = _merged_validations(rows, assignments, plan)
    if not all(validations.values()):
        raise ValueError("merged mapping validations failed")
    expected = {
        "mapping_path": str(mapping_path.resolve()),
        "mapping_sha256": file_sha256(mapping_path),
        "mapping_rows": len(rows),
        "merged_rows_sha256": _row_digest(rows),
        **_merged_summary(rows),
        "validations": validations,
    }
    mismatches = {key for key, value in expected.items() if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"merged mapping summary mismatch: {sorted(mismatches)}")


def _validate_merged_plan_claims(
    manifest: Mapping[str, Any], plan_path: Path, plan: Mapping[str, Any]
) -> None:
    expected = {
        "cleaned_records_path": plan["cleaned_records_path"],
        "cleaned_records_sha256": plan["cleaned_records_sha256"],
        "candidate_inventory": plan["candidate_inventory"],
        "candidate_identity_sha256": plan["candidate_identity_sha256"],
        "prompt": plan["prompt"],
        "provider_contracts": plan["provider_contracts"],
        "partition_plan": {
            "path": str(plan_path.resolve()),
            "sha256": file_sha256(plan_path),
            "version": PLAN_VERSION,
            "assignment_sha256": plan["assignment_sha256"],
        },
    }
    mismatches = {key for key, value in expected.items() if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"merged mapping plan claims mismatch: {sorted(mismatches)}")


def validate_merged_mapping(mapping_path: Path) -> None:
    """Revalidate the immutable merge and its complete partition lineage."""
    manifest = _mapping_manifest(mapping_path)
    if (
        manifest.get("generation_version") != MERGE_VERSION
        or manifest.get("task_id") != "ames"
        or manifest.get("mapping_version") != MAPPING_VERSION
        or manifest.get("artifact_role") != "raw_candidate_selection"
    ):
        raise ValueError("merged mapping identity mismatch")
    plan_ref = manifest.get("partition_plan") or {}
    plan_path = Path(str(plan_ref.get("path") or ""))
    if (
        not plan_path.is_file()
        or str(plan_ref.get("path") or "") != str(plan_path.resolve())
        or file_sha256(plan_path) != plan_ref.get("sha256")
        or plan_ref.get("version") != PLAN_VERSION
    ):
        raise ValueError("merged mapping partition-plan hash mismatch")
    plan, assignments = load_validated_plan(plan_path)
    rows = pq.read_table(mapping_path).to_pylist()
    _validate_merged_summary(mapping_path, manifest, rows, assignments, plan)
    _validate_merged_plan_claims(manifest, plan_path, plan)
    expected_rows, children = _load_all_partitions(plan_path, plan, assignments)
    expected_digest = _row_digest(expected_rows)
    if (
        _row_digest(rows) != expected_digest
        or manifest.get("merged_rows_sha256") != expected_digest
    ):
        raise ValueError("merged rows differ from frozen partition outputs")
    if manifest.get("partition_manifests") != _child_references(children):
        raise ValueError("merged partition lineage mismatch")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_MAPPING_PATH)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = merge_partitions(args.plan, args.output)
    print(json.dumps(manifest["status_counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
