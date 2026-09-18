"""Review every frozen AMES endpoint-unit pair through one pinned API route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.build_exact_unit_review import (
    MAX_PACKET_ITEMS,
    load_review_packet_manifest,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.plan_measurement_resolution import (
    provider_contracts,
    validate_endpoint_receipt,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_measurement_selection import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CREDENTIAL_ENV,
    DEEPSEEK_MODEL,
    DEEPSEEK_PROVIDER,
    ENDPOINT_CONCURRENCY_BUDGET,
)
from data.processing.llm_api import openai_compatible_client

MODEL = DEEPSEEK_MODEL
PROVIDER = DEEPSEEK_PROVIDER
BASE_URL = DEEPSEEK_BASE_URL
CREDENTIAL_ENV = DEEPSEEK_CREDENTIAL_ENV
MAX_CONCURRENCY = ENDPOINT_CONCURRENCY_BUDGET
CACHE_VERSION = "ames_exact_unit_review_cache.v3"
RUN_RECEIPT_VERSION = "ames_exact_unit_review_run.v2"
DEFAULT_MAX_TOKENS = 32_768
_CACHE_WRITE_LOCK = threading.Lock()

SYSTEM_PROMPT = """You are conservatively reviewing exact unit strings for AMES evidence.
Return one decision for every supplied endpoint-unit item and no other text.

Use action "map" with mapping_kind "identity" when input_unit is a real,
source-supported unit spelling. Preserve that spelling exactly as canonical_unit.
Use mapping_kind "spelling_alias" only when the supplied evidence explicitly
establishes that input_unit and one unit in allowed_alias_targets_by_endpoint are
spelling variants of the same unit. Include concise equivalence_evidence. A
semantic conversion, scale change, inferred denominator, or inferred power is
never a spelling alias.
Use action "exclude" only when input_unit is not a measurement unit, such as a
condition, identifier, missing-value marker, or prose fragment. The parser output
is advisory evidence and cannot justify an automatic mapping.

Never infer or convert scale. Do not emit scale or domain. For each decision emit:
- common fields: canonical_endpoint, input_unit, action, review_basis
- identity map: add mapping_kind="identity" and canonical_unit=input_unit
- spelling alias: add mapping_kind="spelling_alias", canonical_unit, and
  equivalence_evidence
- exclusion: add no other fields
Return exactly one JSON object shaped as {"decisions": [...]}.
"""


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_text(value).encode("utf-8")).hexdigest()


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("canonical_endpoint") or ""), str(row.get("input_unit") or "")


def _wire_item(row: Mapping[str, Any]) -> dict[str, Any]:
    endpoint, unit = _key(row)
    return {
        "canonical_endpoint": endpoint,
        "input_unit": unit,
        "origins": row.get("origins"),
        "row_count": row.get("rows"),
        "representative_contexts": row.get("representative_contexts"),
        "parser_output": row.get("parser_output"),
    }


def _alias_catalogs(
    batches: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, list[str]]:
    targets: dict[str, set[str]] = {}
    for _, items in batches:
        for row in items:
            endpoint, unit = _key(row)
            targets.setdefault(endpoint, set()).add(unit)
    return {endpoint: sorted(units) for endpoint, units in sorted(targets.items())}


def load_review_batches(
    manifest_path: Path,
) -> tuple[
    dict[str, Any], list[tuple[str, list[dict[str, Any]]]], list[tuple[str, str]]
]:
    """Load the verified packet partition and reject duplicate review keys."""
    manifest, inventory = load_review_packet_manifest(manifest_path)
    rows = inventory.get("observed_pairs")
    specs = manifest.get("packets")
    if not isinstance(rows, list) or not rows or not isinstance(specs, list):
        raise ValueError("exact-unit review manifest has no packet inventory")
    if len(specs) != manifest.get("packet_count"):
        raise ValueError("exact-unit review packet count mismatch")
    keys = [_key(row) for row in rows]
    if any(not all(key) for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("exact-unit review inventory has duplicate or empty keys")
    batches, offset = [], 0
    for spec in specs:
        count = int(spec.get("item_count") or 0)
        selected = rows[offset : offset + count]
        if not 1 <= count <= MAX_PACKET_ITEMS or len(selected) != count:
            raise ValueError("invalid exact-unit review batch size")
        packet_id = str(spec.get("packet_id") or "")
        if not packet_id or any(packet_id == prior[0] for prior in batches):
            raise ValueError("duplicate or empty exact-unit review packet id")
        batches.append((packet_id, [_wire_item(row) for row in selected]))
        offset += count
    if offset != len(rows):
        raise ValueError("exact-unit review packets do not cover the inventory")
    return manifest, batches, keys


def _request_payload(
    packet_id: str,
    items: Sequence[Mapping[str, Any]],
    alias_targets: Mapping[str, list[str]],
    manifest_sha256: str,
    max_tokens: int,
) -> dict[str, Any]:
    if max_tokens < 1:
        raise ValueError("exact-unit max tokens must be positive")
    endpoints = sorted({_key(row)[0] for row in items})
    user = {
        "packet_id": packet_id,
        "packet_manifest_sha256": manifest_sha256,
        "item_count": len(items),
        "items": list(items),
        "allowed_alias_targets_by_endpoint": {
            endpoint: alias_targets[endpoint] for endpoint in endpoints
        },
    }
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _json_text(user)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }


def _review_plan(
    batches: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    manifest_sha256: str,
    max_tokens: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    alias_targets = _alias_catalogs(batches)
    requests = {
        packet_id: _request_payload(
            packet_id, items, alias_targets, manifest_sha256, max_tokens
        )
        for packet_id, items in batches
    }
    planned = {packet: _json_sha256(request) for packet, request in requests.items()}
    return requests, planned


def _response_payload(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        payload = dict(response)
    else:
        try:
            payload = response.model_dump(mode="json")
        except TypeError:
            payload = response.model_dump()
    if not isinstance(payload, dict):
        raise TypeError("OpenAI-compatible response is not an object")
    for field in ("model", "provider"):
        value = getattr(response, field, None)
        if value and not payload.get(field):
            payload[field] = value
    return json.loads(_json_text(payload))


def _response_metadata(payload: Mapping[str, Any]) -> tuple[str, str]:
    extra = payload.get("model_extra")
    extra = extra if isinstance(extra, Mapping) else {}
    returned = str(payload.get("model") or "")
    provider = str(payload.get("provider") or extra.get("provider") or "")
    return returned, provider


def _validate_request_route(request: Mapping[str, Any]) -> None:
    if request.get("model") != MODEL or "extra_body" in request:
        raise ValueError("exact-unit request route drift")


def _validate_response_route(response: Mapping[str, Any]) -> None:
    returned, served = _response_metadata(response)
    if returned != MODEL or served:
        raise ValueError(
            f"exact-unit response route mismatch: model={returned!r}, provider={served!r}"
        )


def _validate_route(request: Mapping[str, Any], response: Mapping[str, Any]) -> None:
    _validate_request_route(request)
    _validate_response_route(response)


def _validate_client_route(client: Any) -> None:
    base_url = str(getattr(client, "base_url", "")).rstrip("/")
    if base_url and base_url != BASE_URL:
        raise ValueError(f"exact-unit client base URL mismatch: {base_url!r}")


def _cache_event(
    packet_id: str,
    manifest_sha256: str,
    credential_env: str,
    request: dict[str, Any],
    response: dict[str, Any],
    attempt_status: str,
    rejection_reason: str | None,
) -> dict[str, Any]:
    returned, provider = _response_metadata(response)
    return {
        "cache_version": CACHE_VERSION,
        "packet_id": packet_id,
        "packet_manifest_sha256": manifest_sha256,
        "credential_env": credential_env,
        "request_sha256": _json_sha256(request),
        "response_sha256": _json_sha256(response),
        "returned_model": returned,
        "served_provider": provider,
        "attempt_status": attempt_status,
        "rejection_reason": rejection_reason,
        "request": request,
        "response": response,
    }


def _validate_attempt_event(event: Any) -> tuple[str, str, str]:
    required = {
        "cache_version",
        "packet_id",
        "packet_manifest_sha256",
        "credential_env",
        "request_sha256",
        "response_sha256",
        "returned_model",
        "served_provider",
        "attempt_status",
        "rejection_reason",
        "request",
        "response",
    }
    if not isinstance(event, dict) or set(event) != required:
        raise ValueError("invalid exact-unit cache event shape")
    request, response = event["request"], event["response"]
    if not isinstance(request, dict) or not isinstance(response, dict):
        raise TypeError("invalid exact-unit cached request or response")
    metadata = ("packet_id", "packet_manifest_sha256")
    if any(not isinstance(event[name], str) or not event[name] for name in metadata):
        raise ValueError("invalid exact-unit cache provenance")
    if event["credential_env"] != CREDENTIAL_ENV:
        raise ValueError("exact-unit cache credential provenance mismatch")
    if event["cache_version"] != CACHE_VERSION:
        raise ValueError("unsupported exact-unit cache version")
    if event["request_sha256"] != _json_sha256(request):
        raise ValueError("exact-unit cached request hash mismatch")
    if event["response_sha256"] != _json_sha256(response):
        raise ValueError("exact-unit cached response hash mismatch")
    returned, provider = _response_metadata(response)
    if (event["returned_model"], event["served_provider"]) != (returned, provider):
        raise ValueError("exact-unit cached response metadata mismatch")
    status, reason = event["attempt_status"], event["rejection_reason"]
    if status == "accepted" and reason is None:
        _validate_route(request, response)
    elif status == "rejected" and isinstance(reason, str) and reason.strip():
        _validate_request_route(request)
    else:
        raise ValueError("invalid exact-unit cache attempt status")
    return str(event["packet_id"]), str(event["request_sha256"]), status


def _invalidation_event(event: Mapping[str, Any], error: Exception) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "packet_id": event["packet_id"],
        "packet_manifest_sha256": event["packet_manifest_sha256"],
        "request_sha256": event["request_sha256"],
        "response_sha256": event["response_sha256"],
        "attempt_status": "invalidated",
        "rejection_reason": f"{type(error).__name__}: {error}",
    }


def _validate_invalidation_event(event: Any) -> tuple[str, str, str]:
    required = {
        "cache_version",
        "packet_id",
        "packet_manifest_sha256",
        "request_sha256",
        "response_sha256",
        "attempt_status",
        "rejection_reason",
    }
    if not isinstance(event, dict) or set(event) != required:
        raise ValueError("invalid exact-unit cache invalidation shape")
    text_fields = required - {"cache_version", "attempt_status"}
    if any(
        not isinstance(event[name], str) or not event[name].strip()
        for name in text_fields
    ):
        raise ValueError("invalid exact-unit cache invalidation provenance")
    if (
        event["cache_version"] != CACHE_VERSION
        or event["attempt_status"] != "invalidated"
    ):
        raise ValueError("invalid exact-unit cache invalidation status")
    return str(event["packet_id"]), str(event["request_sha256"]), "invalidated"


def _validate_cache_event(event: Any) -> tuple[str, str, str]:
    if isinstance(event, dict) and event.get("attempt_status") == "invalidated":
        return _validate_invalidation_event(event)
    return _validate_attempt_event(event)


def _load_cache(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    by_hash: dict[str, dict[str, Any]] = {}
    by_packet: dict[str, str] = {}
    if not path.exists():
        return by_hash, by_packet
    if not path.is_file():
        raise ValueError(f"exact-unit cache is not a file: {path}")
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            raise ValueError(f"blank exact-unit cache event at line {line_number}")
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid exact-unit cache JSON at line {line_number}"
            ) from error
        packet_id, request_hash, status = _validate_cache_event(event)
        prior_hash = by_packet.setdefault(packet_id, request_hash)
        if prior_hash != request_hash:
            raise ValueError("exact-unit cache packet request drift")
        if status == "invalidated":
            accepted = by_hash.get(request_hash)
            matches = accepted is not None and all(
                accepted[field] == event[field]
                for field in ("packet_id", "packet_manifest_sha256", "response_sha256")
            )
            if not matches:
                raise ValueError(
                    "exact-unit cache invalidation has no accepted attempt"
                )
            del by_hash[request_hash]
        elif status == "accepted":
            if request_hash in by_hash:
                raise ValueError("duplicate accepted exact-unit cache request")
            by_hash[request_hash] = event
    return by_hash, by_packet


def _append_cache(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_WRITE_LOCK:
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(_json_text(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _required_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"exact-unit decision has invalid {field}")
    return value.strip()


def _validated_decision(row: Any) -> dict[str, Any]:
    common = {"canonical_endpoint", "input_unit", "action", "review_basis"}
    if not isinstance(row, dict):
        raise TypeError("exact-unit decision is not an object")
    endpoint = _required_text(row, "canonical_endpoint")
    unit = _required_text(row, "input_unit")
    action = _required_text(row, "action")
    basis = _required_text(row, "review_basis")
    if endpoint != row["canonical_endpoint"] or unit != row["input_unit"]:
        raise ValueError("exact-unit decision key contains surrounding whitespace")
    if action == "exclude":
        if set(row) != common:
            raise ValueError("excluded exact-unit decision has mapping fields")
        return {**row, "review_basis": basis}
    if action != "map":
        raise ValueError(f"invalid exact-unit action: {action!r}")
    kind = _required_text(row, "mapping_kind")
    canonical = _required_text(row, "canonical_unit")
    fields = common | {"mapping_kind", "canonical_unit"}
    if kind == "identity":
        if set(row) != fields or canonical != unit:
            raise ValueError("invalid exact-unit identity mapping")
    elif kind == "spelling_alias":
        fields.add("equivalence_evidence")
        evidence = _required_text(row, "equivalence_evidence")
        if set(row) != fields or canonical == unit:
            raise ValueError("invalid exact-unit spelling alias")
        row = {**row, "equivalence_evidence": evidence}
    else:
        raise ValueError(f"invalid exact-unit mapping kind: {kind!r}")
    return {**row, "review_basis": basis, "canonical_unit": canonical}


def _response_content(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("exact-unit response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("exact-unit response has no JSON content")
    return content


def _batch_decisions(
    response: Mapping[str, Any], items: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    try:
        payload = json.loads(_response_content(response))
    except json.JSONDecodeError as error:
        raise ValueError("exact-unit response content is not JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"decisions"}:
        raise ValueError("exact-unit response must contain only decisions")
    rows = payload["decisions"]
    if not isinstance(rows, list):
        raise TypeError("exact-unit decisions must be a list")
    parsed = [_validated_decision(row) for row in rows]
    keys = [_key(row) for row in parsed]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate exact-unit response decision")
    expected = [_key(row) for row in items]
    if set(keys) != set(expected) or len(keys) != len(expected):
        raise ValueError("exact-unit response decision coverage mismatch")
    by_key = dict(zip(keys, parsed))
    return [by_key[key] for key in expected]


def _record_attempt(
    client: Any,
    cache_path: Path,
    packet_id: str,
    manifest_sha256: str,
    credential_env: str,
    request: dict[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    response = _response_payload(client.chat.completions.create(**request))
    try:
        _validate_route(request, response)
        decisions = _batch_decisions(response, items)
    except Exception as error:
        event = _cache_event(
            packet_id,
            manifest_sha256,
            credential_env,
            request,
            response,
            "rejected",
            f"{type(error).__name__}: {error}",
        )
        _validate_cache_event(event)
        _append_cache(cache_path, event)
        raise
    event = _cache_event(
        packet_id,
        manifest_sha256,
        credential_env,
        request,
        response,
        "accepted",
        None,
    )
    _validate_cache_event(event)
    _append_cache(cache_path, event)
    return event, decisions


def _validate_complete(
    decisions: Sequence[Mapping[str, Any]], expected: Sequence[tuple[str, str]]
) -> None:
    keys = [_key(row) for row in decisions]
    if keys != list(expected) or len(set(keys)) != len(keys):
        raise ValueError("complete exact-unit decision order or coverage mismatch")
    by_key = dict(zip(keys, decisions))
    for row in decisions:
        if row["action"] != "map" or row["mapping_kind"] != "spelling_alias":
            continue
        target = (row["canonical_endpoint"], row["canonical_unit"])
        identity = by_key.get(target)
        if identity is None or identity.get("mapping_kind") != "identity":
            raise ValueError(f"spelling alias lacks observed identity target: {target}")


def _invalidate_alias_attempts(
    cache_path: Path,
    results: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
    error: Exception,
) -> None:
    selected = [
        result
        for result in results
        if any(row.get("mapping_kind") == "spelling_alias" for row in result[1])
    ]
    for event, _ in selected or results:
        marker = _invalidation_event(event, error)
        _validate_invalidation_event(marker)
        _append_cache(cache_path, marker)


def _validate_complete_or_invalidate(
    cache_path: Path,
    decisions: Sequence[Mapping[str, Any]],
    expected: Sequence[tuple[str, str]],
    results: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
) -> None:
    try:
        _validate_complete(decisions, expected)
    except Exception as error:
        _invalidate_alias_attempts(cache_path, results, error)
        raise


def _review_row(decision: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "canonical_endpoint": decision["canonical_endpoint"],
        "input_unit": decision["input_unit"],
        "action": decision["action"],
        "domain": "any",
        "review_basis": decision["review_basis"],
        "reviewer": MODEL,
        "review_round": 1,
    }
    if decision["action"] == "map":
        row.update(canonical_unit=decision["canonical_unit"], scale="1")
        if decision["mapping_kind"] == "spelling_alias":
            row["review_basis"] += (
                "; explicit spelling equivalence: " + decision["equivalence_evidence"]
            )
    return row


def review_receipt_path(review_path: Path) -> Path:
    return review_path.with_suffix(".manifest.json")


def _review_text(decisions: Sequence[Mapping[str, Any]]) -> str:
    return "".join(_json_text(_review_row(row)) + "\n" for row in decisions)


def _reference(path: Path, sha256: str | None = None) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256 or file_sha256(path)}


def _receipt_packets(
    results: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
) -> list[dict[str, str]]:
    return [
        {
            "packet_id": str(event["packet_id"]),
            "credential_env": str(event["credential_env"]),
            "request_sha256": str(event["request_sha256"]),
            "response_sha256": str(event["response_sha256"]),
        }
        for event, _ in results
    ]


def _run_receipt(
    manifest_path: Path,
    endpoint_receipt_path: Path,
    cache_path: Path,
    output_path: Path,
    review_sha256: str,
    max_tokens: int,
    results: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    item_count = sum(len(decisions) for _, decisions in results)
    return {
        "version": RUN_RECEIPT_VERSION,
        "task": "ames",
        "route": {
            "base_url": BASE_URL,
            "provider": PROVIDER,
            "model": MODEL,
            "returned_model": MODEL,
            "credential_env": CREDENTIAL_ENV,
            "max_concurrency": MAX_CONCURRENCY,
        },
        "max_tokens": max_tokens,
        "packet_manifest": _reference(manifest_path),
        "endpoint_receipt": _reference(endpoint_receipt_path),
        "cache": _reference(cache_path),
        "review_decisions": _reference(output_path, review_sha256),
        "packets": _receipt_packets(results),
        "counts": {"packets": len(results), "items": item_count},
        "validations": {
            "cache_bound_reviews": True,
            "complete_packet_coverage": True,
            "exact_local_route": True,
        },
    }


def _publish_review_pair(
    review_temporary: Path,
    review_path: Path,
    receipt_temporary: Path,
    receipt_path: Path,
) -> None:
    installed: list[Path] = []
    try:
        os.replace(review_temporary, review_path)
        installed.append(review_path)
        os.replace(receipt_temporary, receipt_path)
        installed.append(receipt_path)
        validate_review_run_receipt(review_path, receipt_path=receipt_path)
    except BaseException:
        for path in reversed(installed):
            path.unlink(missing_ok=True)
        raise


def _write_review_pair(
    manifest_path: Path,
    endpoint_receipt_path: Path,
    cache_path: Path,
    review_path: Path,
    max_tokens: int,
    results: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
) -> None:
    receipt_path = review_receipt_path(review_path)
    temporary = review_path.with_suffix(f"{review_path.suffix}.tmp")
    receipt_temporary = receipt_path.with_suffix(f"{receipt_path.suffix}.tmp")
    targets = (review_path, receipt_path, temporary, receipt_temporary)
    if any(path.exists() for path in targets):
        raise FileExistsError("exact-unit review output or receipt already exists")
    review_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_text(
            _review_text([row for _, rows in results for row in rows]),
            encoding="utf-8",
        )
        receipt = _run_receipt(
            manifest_path,
            endpoint_receipt_path,
            cache_path,
            review_path,
            file_sha256(temporary),
            max_tokens,
            results,
        )
        receipt_temporary.write_text(_json_text(receipt) + "\n", encoding="utf-8")
        _publish_review_pair(temporary, review_path, receipt_temporary, receipt_path)
    finally:
        temporary.unlink(missing_ok=True)
        receipt_temporary.unlink(missing_ok=True)


def _validate_cache_plan(
    cached_packets: Mapping[str, str], planned: Mapping[str, str]
) -> None:
    extra = sorted(set(cached_packets) - set(planned))
    drift = sorted(
        packet
        for packet in set(cached_packets) & set(planned)
        if cached_packets[packet] != planned[packet]
    )
    if extra or drift:
        raise ValueError(
            f"exact-unit cache does not match plan: extra={extra}, drift={drift}"
        )


def _receipt_reference(receipt: Mapping[str, Any], name: str) -> Path:
    reference = receipt.get(name)
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"invalid exact-unit run receipt {name} reference")
    path = Path(str(reference["path"]))
    if not path.is_absolute() or file_sha256(path) != reference["sha256"]:
        raise ValueError(f"exact-unit run receipt {name} hash mismatch")
    return path.resolve()


def _receipt_results(
    manifest_path: Path, cache_path: Path, max_tokens: int
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], list[dict[str, Any]]]]]:
    _, batches, expected = load_review_batches(manifest_path)
    manifest_hash = file_sha256(manifest_path)
    _, planned = _review_plan(batches, manifest_hash, max_tokens)
    cached, cached_packets = _load_cache(cache_path)
    _validate_cache_plan(cached_packets, planned)
    decisions, results = [], []
    for packet_id, items in batches:
        event = cached.get(planned[packet_id])
        if event is None or event["packet_id"] != packet_id:
            raise ValueError(f"no active accepted cache event for {packet_id}")
        if event["packet_manifest_sha256"] != manifest_hash:
            raise ValueError("exact-unit cached manifest hash mismatch")
        batch_decisions = _batch_decisions(event["response"], items)
        decisions.extend(batch_decisions)
        results.append((event, batch_decisions))
    _validate_complete(decisions, expected)
    return decisions, results


def _validate_receipt_header(receipt: Mapping[str, Any]) -> int:
    expected_route = {
        "base_url": BASE_URL,
        "provider": PROVIDER,
        "model": MODEL,
        "returned_model": MODEL,
        "credential_env": CREDENTIAL_ENV,
        "max_concurrency": MAX_CONCURRENCY,
    }
    if receipt.get("version") != RUN_RECEIPT_VERSION or receipt.get("task") != "ames":
        raise ValueError("unsupported AMES exact-unit run receipt")
    if receipt.get("route") != expected_route:
        raise ValueError("exact-unit run receipt route mismatch")
    max_tokens = receipt.get("max_tokens")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 1
    ):
        raise ValueError("invalid exact-unit run receipt max_tokens")
    return max_tokens


def validate_review_run_receipt(
    review_path: Path,
    *,
    receipt_path: Path | None = None,
    packet_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Prove that flattened reviews derive from accepted exact-route cache events."""
    review_path = review_path.resolve()
    receipt_path = (receipt_path or review_receipt_path(review_path)).resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise TypeError("exact-unit run receipt is not an object")
    max_tokens = _validate_receipt_header(receipt)
    manifest_path = _receipt_reference(receipt, "packet_manifest")
    endpoint_receipt_path = _receipt_reference(receipt, "endpoint_receipt")
    validate_endpoint_receipt(endpoint_receipt_path, provider_contracts()["deepseek"])
    cache_path = _receipt_reference(receipt, "cache")
    recorded_review = _receipt_reference(receipt, "review_decisions")
    if recorded_review != review_path:
        raise ValueError("exact-unit run receipt review path mismatch")
    if packet_manifest_path and manifest_path != packet_manifest_path.resolve():
        raise ValueError("exact-unit run receipt packet manifest path mismatch")
    decisions, results = _receipt_results(manifest_path, cache_path, max_tokens)
    if review_path.read_text(encoding="utf-8") != _review_text(decisions):
        raise ValueError("exact-unit reviews differ from accepted cached responses")
    expected = _run_receipt(
        manifest_path,
        endpoint_receipt_path,
        cache_path,
        review_path,
        file_sha256(review_path),
        max_tokens,
        results,
    )
    if receipt != expected:
        raise ValueError("exact-unit run receipt claims mismatch")
    return receipt


def _collect_review_results(
    batches: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    requests: Mapping[str, dict[str, Any]],
    planned: Mapping[str, str],
    cached: Mapping[str, dict[str, Any]],
    cache_path: Path,
    manifest_hash: str,
    client: Any | None,
) -> tuple[list[dict[str, Any]], list[Any], int]:
    selected_credential = CREDENTIAL_ENV
    missing = [batch for batch in batches if planned[batch[0]] not in cached]
    if missing and client is None:
        client, selected_credential = openai_compatible_client(
            base_url=BASE_URL,
            provider=PROVIDER,
            env_file=None,
            max_connections=MAX_CONCURRENCY,
        )
        client = client.with_options(max_retries=0)
        _validate_client_route(client)
    fresh = {}
    with ThreadPoolExecutor(
        max_workers=min(MAX_CONCURRENCY, len(missing)) or 1
    ) as pool:
        futures = {
            packet_id: pool.submit(
                _record_attempt,
                client,
                cache_path,
                packet_id,
                manifest_hash,
                selected_credential,
                requests[packet_id],
                items,
            )
            for packet_id, items in missing
        }
        for packet_id, _ in missing:
            fresh[packet_id] = futures[packet_id].result()
    decisions, results = [], []
    for packet_id, items in batches:
        result = fresh.get(packet_id)
        event = result[0] if result else cached[planned[packet_id]]
        if event["packet_manifest_sha256"] != manifest_hash:
            raise ValueError("exact-unit cached manifest hash mismatch")
        batch_decisions = (
            result[1] if result else _batch_decisions(event["response"], items)
        )
        decisions.extend(batch_decisions)
        results.append((event, batch_decisions))
    return decisions, results, len(missing)


def run_review(
    manifest_path: Path,
    cache_path: Path,
    output_path: Path,
    *,
    endpoint_receipt_path: Path,
    client: Any | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Run or replay all frozen packets, then atomically write complete JSONL."""
    if output_path.exists() or review_receipt_path(output_path).exists():
        raise FileExistsError("refusing to replace exact-unit reviews or receipt")
    validate_endpoint_receipt(endpoint_receipt_path, provider_contracts()["deepseek"])
    if client is not None:
        _validate_client_route(client)
    _, batches, expected = load_review_batches(manifest_path)
    manifest_hash = file_sha256(manifest_path)
    requests, planned = _review_plan(batches, manifest_hash, max_tokens)
    cached, cached_packets = _load_cache(cache_path)
    _validate_cache_plan(cached_packets, planned)
    decisions, results, api_requests = _collect_review_results(
        batches,
        requests,
        planned,
        cached,
        cache_path,
        manifest_hash,
        client,
    )
    _validate_complete_or_invalidate(cache_path, decisions, expected, results)
    _write_review_pair(
        manifest_path,
        endpoint_receipt_path,
        cache_path,
        output_path,
        max_tokens,
        results,
    )
    counts = Counter(
        "exclude" if row["action"] == "exclude" else row["mapping_kind"]
        for row in decisions
    )
    return {
        "items": len(decisions),
        "batches": len(batches),
        "api_requests": api_requests,
        "cache_hits": len(batches) - api_requests,
        "decision_counts": dict(sorted(counts.items())),
        "reviews_sha256": file_sha256(output_path),
        "receipt_sha256": file_sha256(review_receipt_path(output_path)),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--endpoint-receipt", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_review(
        args.packet_manifest,
        args.cache,
        args.output,
        endpoint_receipt_path=args.endpoint_receipt,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASE_URL",
    "CACHE_VERSION",
    "CREDENTIAL_ENV",
    "MAX_CONCURRENCY",
    "MODEL",
    "PROVIDER",
    "RUN_RECEIPT_VERSION",
    "load_review_batches",
    "review_receipt_path",
    "run_review",
    "validate_review_run_receipt",
]
