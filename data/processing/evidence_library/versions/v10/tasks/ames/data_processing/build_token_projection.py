"""Derive frozen Ames p95 request-token projections from a completed GPT replay."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.plan_measurement_resolution import (
    ACTIVE_EXTRACTION_SOURCE_IDS,
    DEFAULT_MAX_COMPLETION_TOKENS,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    TOKEN_PROJECTION_METRIC,
    TOKEN_PROJECTION_QUANTILE_METHOD,
    TOKEN_PROJECTION_VERSION,
    candidate_inventory_reference,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_candidate_resolution import (
    CANDIDATE_CONTRACT_VERSION,
    compile_selection,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_measurement_selection import (
    MAPPING_VERSION,
    REASONING_EFFORT,
    prompt_manifest,
)

GPT_CREDENTIALS = frozenset({"OPENAI_API_KEY_ONE", "OPENAI_API_KEY_TWO"})
DEFAULT_GOLD = Path(
    "tests/chembl_tool/common/measurement_resolution_quality/gold/ames.v10.jsonl"
)


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _read_events(path: Path) -> list[dict[str, Any]]:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not events or any(not isinstance(event, dict) for event in events):
        raise ValueError("gold replay cache is empty or malformed")
    return events


def _unique_requests(
    events: Sequence[Mapping[str, Any]], status: str
) -> dict[str, Mapping[str, Any]]:
    selected: dict[str, Mapping[str, Any]] = {}
    for event in events:
        if event.get("status") != status:
            continue
        request_id = str(event.get("request_id") or "")
        if not request_id or request_id in selected:
            raise ValueError(f"duplicate or empty {status} request id")
        selected[request_id] = event
    return selected


def _mapping_contract(
    mapping_path: Path, gold_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = mapping_path.with_suffix(".manifest.json")
    manifest = _read_object(manifest_path)
    rows = pq.read_table(mapping_path).to_pylist()
    expected = {
        "task_id": "ames",
        "mapping_version": MAPPING_VERSION,
        "mapping_sha256": file_sha256(mapping_path),
        "mapping_rows": len(rows),
        "model": OPENAI_MODEL,
        "prompt": prompt_manifest(),
        "base_mapping": None,
    }
    mismatches = {key for key, value in expected.items() if manifest.get(key) != value}
    inference = manifest.get("inference") or {}
    if (
        int(inference.get("max_completion_tokens") or 0)
        != DEFAULT_MAX_COMPLETION_TOKENS
    ):
        mismatches.add("max_completion_tokens")
    if inference.get("reasoning_mode") != REASONING_EFFORT:
        mismatches.add("reasoning_mode")
    validations = manifest.get("validations") or {}
    if not validations.get("one_row_per_candidate") or not validations.get(
        "unique_cleaned_record_ids"
    ):
        mismatches.add("candidate_coverage")
    if mismatches:
        raise ValueError(f"gold mapping manifest mismatch: {sorted(mismatches)}")
    records_path, _ = _validate_source_files(manifest)
    candidate_manifest = _read_object(records_path.with_suffix(".manifest.json"))
    candidate_reference = candidate_inventory_reference(
        records_path, int(candidate_manifest.get("candidate_rows") or 0)
    )
    _validate_gold_selections(rows, records_path, gold_path)
    return manifest, rows, candidate_reference


def _validate_source_files(manifest: Mapping[str, Any]) -> tuple[Path, Path]:
    records_path = Path(str(manifest.get("cleaned_records_path") or ""))
    endpoint_profile = manifest.get("endpoint_profile") or {}
    profile_path = Path(str(endpoint_profile.get("path") or ""))
    checks = (
        (records_path, manifest.get("cleaned_records_sha256")),
        (profile_path, endpoint_profile.get("sha256")),
    )
    if any(
        not path.is_file() or file_sha256(path) != digest for path, digest in checks
    ):
        raise ValueError("gold mapping source or endpoint-profile hash mismatch")
    return records_path, profile_path


def _gold_inventory(path: Path) -> dict[str, dict[str, str]]:
    lines = [json.loads(line) for line in path.read_text().splitlines() if line]
    manifest, cases = lines[0], lines[1:]
    expected_manifest = {
        "task_id": "ames",
        "corpus_version": "ames_measurement_resolution_gold.v1",
        "expectation_update_policy": "manual_source_review_only",
        "labelled_before_any_model_run": True,
        "cases": len(cases),
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise ValueError("AMES token projection gold contract mismatch")
    inventory: dict[str, dict[str, str]] = {}
    for case in cases:
        prefix, separator, record_id = str(case.get("audit_case_id") or "").partition(
            ":"
        )
        identity = {
            "source_row_uid": str(case.get("source_row_uid") or ""),
            "source_id": str(case.get("source_id") or ""),
            "canonical_endpoint_name": str(case.get("canonical_endpoint_name") or ""),
        }
        source_input = case.get("input") or {}
        if (
            prefix != "ames"
            or not separator
            or not record_id
            or record_id in inventory
            or any(not value for value in identity.values())
            or identity["source_id"] not in ACTIVE_EXTRACTION_SOURCE_IDS
            or source_input.get("source_id") != identity["source_id"]
            or source_input.get("canonical_endpoint_name")
            != identity["canonical_endpoint_name"]
        ):
            raise ValueError("AMES token projection gold identity mismatch")
        inventory[record_id] = identity
    return inventory


def _candidate_subset(path: Path, ids: set[str]) -> dict[str, dict[str, Any]]:
    columns = [
        "cleaned_record_id",
        "source_row_uid",
        "source_id",
        "canonical_endpoint_name",
        "candidate_contract_version",
        "measurement_candidates_json",
        "candidate_set_sha256",
    ]
    rows = pq.read_table(
        path, columns=columns, filters=[("cleaned_record_id", "in", sorted(ids))]
    ).to_pylist()
    indexed = {str(row["cleaned_record_id"]): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != ids:
        raise ValueError("gold rows differ from the candidate inventory")
    if any(
        row.get("candidate_contract_version") != CANDIDATE_CONTRACT_VERSION
        for row in rows
    ):
        raise ValueError("gold candidate contract version mismatch")
    return indexed


def _validate_gold_selections(
    rows: Sequence[Mapping[str, Any]], records_path: Path, gold_path: Path
) -> None:
    gold = _gold_inventory(gold_path)
    ids = set(gold)
    selected_ids = {str(row.get("cleaned_record_id") or "") for row in rows}
    if len(selected_ids) != len(rows) or selected_ids != ids:
        raise ValueError("mapping does not cover the exact AMES gold inventory")
    candidates = _candidate_subset(records_path, ids)
    for row in rows:
        record_id = str(row["cleaned_record_id"])
        candidate = candidates[record_id]
        if any(candidate.get(key) != value for key, value in gold[record_id].items()):
            raise ValueError(f"gold candidate identity mismatch: {record_id}")
        exact = {
            "source_row_uid": candidate["source_row_uid"],
            "source_id": candidate["source_id"],
        }
        if any(row.get(key) != value for key, value in exact.items()):
            raise ValueError(f"gold selection identity mismatch: {record_id}")
        compile_selection(row, candidate)


def _row_inventory(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, str, str]]:
    inventory: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        record_id = str(row.get("cleaned_record_id") or "")
        source_id = str(row.get("source_id") or "")
        if not record_id or record_id in inventory:
            raise ValueError("gold mapping has an empty or duplicate row id")
        if source_id not in ACTIVE_EXTRACTION_SOURCE_IDS:
            raise ValueError(f"gold mapping has a non-extraction source: {source_id}")
        exact = {
            "inference_source": "delta_inference",
            "inference_model": OPENAI_MODEL,
            "returned_model": OPENAI_MODEL,
            "inference_base_url": OPENAI_BASE_URL,
            "requested_provider": None,
            "served_provider": None,
            "rejected_response_json": None,
        }
        if any(row.get(key) != value for key, value in exact.items()):
            raise ValueError(f"gold mapping provenance mismatch: {record_id}")
        if row.get("inference_credential_env") not in GPT_CREDENTIALS:
            raise ValueError(f"gold mapping credential mismatch: {record_id}")
        if not row.get("api_response_id") or row.get("assignment_method") not in {
            "model_single_pass",
            "model_structural_retry",
        }:
            raise ValueError(f"gold mapping inference evidence mismatch: {record_id}")
        inventory[record_id] = (
            source_id,
            str(row["inference_credential_env"]),
            str(row["api_response_id"]),
        )
    return inventory


def _validate_submitted(
    event: Mapping[str, Any],
) -> tuple[str, tuple[str, ...], str]:
    source_id = str(event.get("source_id") or "")
    row_ids = tuple(str(value) for value in event.get("row_ids") or ())
    credential = str(event.get("credential_env") or "")
    expected = {
        "model": OPENAI_MODEL,
        "base_url": OPENAI_BASE_URL,
    }
    if any(event.get(key) != value for key, value in expected.items()):
        raise ValueError("gold replay request did not use the exact GPT snapshot")
    if credential not in GPT_CREDENTIALS:
        raise ValueError("gold replay request used an unexpected credential")
    if (
        source_id not in ACTIVE_EXTRACTION_SOURCE_IDS
        or not row_ids
        or any(not record_id for record_id in row_ids)
        or len(row_ids) != len(set(row_ids))
    ):
        raise ValueError("gold replay request lacks an active source or rows")
    return source_id, row_ids, credential


def _actual_tokens(event: Mapping[str, Any]) -> int:
    usage = event.get("usage")
    if not isinstance(usage, Mapping):
        raise TypeError("gold replay request lacks reported usage")
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    if input_tokens < 1 or output_tokens < 0:
        raise ValueError("gold replay request has invalid actual usage")
    return input_tokens + output_tokens


def _api_response_tokens(event: Mapping[str, Any], response_id: str) -> int:
    response = event.get("response")
    if (
        not isinstance(response, Mapping)
        or response.get("id") != response_id
        or response.get("model") != OPENAI_MODEL
    ):
        raise ValueError("gold replay API response body provenance mismatch")
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise TypeError("gold replay API response lacks reported usage")
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(
        usage.get("completion_tokens") or usage.get("output_tokens") or 0
    )
    if input_tokens < 1 or output_tokens < 0:
        raise ValueError("gold replay API response has invalid actual usage")
    return input_tokens + output_tokens


def _validate_api_events(
    events: Sequence[Mapping[str, Any]], request_ids: set[str]
) -> tuple[dict[str, set[str]], dict[str, int]]:
    responses: dict[str, set[str]] = defaultdict(set)
    usage: dict[str, int] = defaultdict(int)
    response_ids = set()
    for event in events:
        if event.get("status") != "api_response":
            continue
        request_id = str(event.get("request_id") or "")
        response_id = str(event.get("api_response_id") or "")
        if (
            request_id not in request_ids
            or event.get("returned_model") != OPENAI_MODEL
            or event.get("requested_provider") is not None
            or event.get("served_provider") is not None
            or not response_id
            or response_id in response_ids
        ):
            raise ValueError("gold replay API provenance mismatch")
        responses[request_id].add(response_id)
        usage[request_id] += _api_response_tokens(event, response_id)
        response_ids.add(response_id)
    if set(responses) != request_ids:
        raise ValueError("gold replay lacks API provenance for a completed request")
    return dict(responses), dict(usage)


def _usage_by_source(
    rows: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]
) -> dict[str, list[int]]:
    inventory = _row_inventory(rows)
    submitted = _unique_requests(events, "submitted")
    terminal = _unique_requests(events, "terminal")
    if set(submitted) != set(terminal):
        raise ValueError("gold replay submitted and terminal requests differ")
    api_responses, api_usage = _validate_api_events(events, set(submitted))
    seen_rows: set[str] = set()
    usage: dict[str, list[int]] = defaultdict(list)
    for request_id, submit in submitted.items():
        source_id, row_ids, credential = _validate_submitted(submit)
        finished = terminal[request_id]
        if str(finished.get("source_id") or "") != source_id:
            raise ValueError("gold replay request source changed before completion")
        if tuple(str(value) for value in finished.get("row_ids") or ()) != row_ids:
            raise ValueError(
                "gold replay request row coverage changed before completion"
            )
        if finished.get("response_status") not in {
            "valid",
            "valid_after_structural_retry",
        }:
            raise ValueError("gold replay request did not finish with a valid response")
        if seen_rows.intersection(row_ids):
            raise ValueError("gold replay submitted a row more than once")
        if any(
            inventory.get(record_id, (None, None, None))[:2] != (source_id, credential)
            for record_id in row_ids
        ):
            raise ValueError("gold replay request differs from the mapping inventory")
        if any(
            inventory[record_id][2] not in api_responses[request_id]
            for record_id in row_ids
        ):
            raise ValueError("gold replay mapping response is absent from API evidence")
        actual_tokens = _actual_tokens(finished)
        if actual_tokens != api_usage[request_id]:
            raise ValueError("gold replay terminal usage differs from API evidence")
        seen_rows.update(row_ids)
        usage[source_id].append(actual_tokens)
    if seen_rows != set(inventory):
        raise ValueError("gold replay does not cover every gold mapping row")
    return dict(usage)


def _nearest_rank_p95(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("cannot project a source without gold requests")
    ordered = sorted(int(value) for value in values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _evidence_reference(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def build_projection(
    cache_path: Path,
    mapping_path: Path,
    output_path: Path,
    gold_path: Path = DEFAULT_GOLD,
) -> dict[str, Any]:
    """Validate one complete gold replay and write its source-specific p95 receipt."""
    if output_path.exists():
        raise FileExistsError(f"refusing to replace frozen projection: {output_path}")
    manifest, rows, candidate_reference = _mapping_contract(mapping_path, gold_path)
    usage = _usage_by_source(rows, _read_events(cache_path))
    if set(usage) != set(ACTIVE_EXTRACTION_SOURCE_IDS):
        raise ValueError("gold replay must include every active extraction source")
    profile = manifest["endpoint_profile"]
    payload = {
        "projection_version": TOKEN_PROJECTION_VERSION,
        "task_id": "ames",
        "requested_model": OPENAI_MODEL,
        "returned_models": [OPENAI_MODEL],
        "batch_size": int(manifest["prompt"]["batch_size"]),
        "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
        "cleaned_records_sha256": manifest["cleaned_records_sha256"],
        "endpoint_profile_sha256": profile["sha256"],
        "prompt": manifest["prompt"],
        "usage_metric": TOKEN_PROJECTION_METRIC,
        "quantile": 0.95,
        "quantile_method": TOKEN_PROJECTION_QUANTILE_METHOD,
        "complete_gold_row_coverage": True,
        "complete_usage": True,
        "allow_fallbacks": False,
        "candidate_inventory": candidate_reference,
        "gold_replay_evidence": {
            "cache": _evidence_reference(cache_path),
            "mapping": _evidence_reference(mapping_path),
            "mapping_manifest": _evidence_reference(
                mapping_path.with_suffix(".manifest.json")
            ),
            "gold_fixture": _evidence_reference(gold_path),
        },
        "sources": {
            source_id: {
                "gold_request_count": len(usage[source_id]),
                "p95_actual_tokens": _nearest_rank_p95(usage[source_id]),
            }
            for source_id in ACTIVE_EXTRACTION_SOURCE_IDS
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_projection(args.cache, args.mapping, args.output, args.gold)
    print(json.dumps(payload["sources"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
