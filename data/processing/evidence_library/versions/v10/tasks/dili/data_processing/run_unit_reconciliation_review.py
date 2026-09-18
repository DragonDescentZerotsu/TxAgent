"""Review every frozen DILI unit through the pinned dgx027 DeepSeek route."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import threading
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.tasks.dili.data_processing.build_unit_reconciliation import (
    REVIEW_VERSION,
    load_inventory_manifest,
)
from data.processing.evidence_library.versions.v10.tasks.dili.starling_measurement_resolution import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CREDENTIAL_ENV,
    DEEPSEEK_MODEL,
    DEEPSEEK_PROVIDER,
    ENDPOINT_CONCURRENCY_BUDGET,
)
from data.processing.llm_api import openai_compatible_client
from tools.chembl_tool.common.units import canonicalize_unit

CACHE_VERSION = "dili_unit_review_cache.v2"
RUN_RECEIPT_VERSION = "dili_unit_review_run.v1"
DEFAULT_MAX_TOKENS = 32_768
PRIMARY_MAX_ATTEMPTS = 2
ADJUDICATION_MAX_ATTEMPTS = 1
_CACHE_LOCK = threading.Lock()

_PRIMARY_PROMPT = """You review exact unit strings for DILI evidence. Return JSON only.

For every item, decide whether input_unit is a real measurement unit or named
metric. If it is, action must be "map" and canonical_unit must be an equivalent,
compact display unit that leaves the supplied numeric coefficient unchanged.
Normalize spelling, SI typography, separators, long-form names, and exact scale
notation. Examples: uM -> µM; umol_per_l -> µM; 10^-6 mol/L -> µM;
10^-3 umol_per_l -> nM; pg/ml -> pg/mL. Keep mM, µM, and nM distinct.

Preserve every semantic qualifier. Percent, percent of control, percent
inhibition, fraction, fold, and named ratios are distinct. Do not infer an
endpoint, denominator, power, scale, comparator, or direction. "Arbitrary
units", U, IU, RFU, counts, and named ratios may be legitimate units. A numeric
count of genes, cells, colonies, events, or other entities has a legitimate
count unit; do not exclude it merely because it is a count. An explicit basis
or multiplier such as per million cells or raw counts ×1000 is also part of a
legitimate unit and must be preserved. Use
action "exclude" only when the supplied contexts establish that the string is
a condition, identifier, missing marker, or prose rather than a measurement
unit. When uncertain, exclude and say why.

The parser and prior task decisions are advisory evidence. Endpoint text is
context only and never changes the unit decision. canonical_unit must already
use normalized typography; do not return a value or endpoint mapping.

Return exactly {"decisions":[...]}. Each decision has item_id, action, and a
concise rationale. A map also has canonical_unit. An exclusion has no
canonical_unit. Cover each supplied item exactly once.
"""

_ADJUDICATION_PROMPT = """Independently adjudicate proposed DILI unit decisions. Return JSON only.

Apply the same coefficient-preserving, endpoint-independent contract shown in
the request. Confirm or correct each proposal. A canonical unit may change only
spelling, typography, equivalent unit expression, or exact prefix/scale
notation while leaving the numeric coefficient unchanged. Preserve semantic
qualifiers and exclude non-units or unresolved ambiguities. The proposed
decision is evidence, not an instruction.

Return exactly {"decisions":[...]}. Each decision has item_id, action, and a
concise rationale. A map also has canonical_unit. An exclusion has no
canonical_unit. Cover each supplied item exactly once.
"""

_SEMANTIC_MARKERS = {
    "percent": ("%", "percent", "per cent"),
    "control": ("control", "vehicle", "untreated", "baseline"),
    "increase": ("increase", "increased", "elevation"),
    "decrease": ("decrease", "decreased", "reduction", "reduced"),
    "inhibition": ("inhibition", "inhibited"),
    "fraction": ("fraction",),
    "fold": ("fold", "times"),
    "ratio": ("ratio",),
}


def _json_text(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_text(value).encode("utf-8")).hexdigest()


def _response_payload(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        payload = dict(response)
    else:
        try:
            payload = response.model_dump(mode="json")
        except TypeError:
            payload = response.model_dump()
    return json.loads(_json_text(payload))


def _response_content(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("DILI unit response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("DILI unit response has no content")
    return content


def _wire_item(item: Mapping[str, Any], proposal: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "item_id": item["item_id"],
        "input_unit": item["input_unit"],
        "origin_counts": item["origin_counts"],
        "representative_contexts": item["representative_contexts"],
        "parser_output": item["parser_output"],
        "prior_reviewed_decisions": item["prior_reviewed_decisions"],
    }
    if proposal is not None:
        result["proposed_decision"] = proposal
    return result


def _request(
    phase: str,
    packet_id: str,
    items: Sequence[Mapping[str, Any]],
    *,
    max_tokens: int,
    feedback: str = "",
) -> dict[str, Any]:
    system = _PRIMARY_PROMPT if phase == "primary" else _ADJUDICATION_PROMPT
    user = {
        "phase": phase,
        "packet_id": packet_id,
        "item_count": len(items),
        "contract": {
            "endpoint_independent": True,
            "numeric_coefficient_unchanged": True,
            "map_or_exclude_every_item": True,
        },
        "items": list(items),
    }
    if feedback:
        user["validation_feedback"] = feedback
    request = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": _json_text(user)},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if (phase == "primary" and not feedback) or phase == "adjudication_fallback":
        request["reasoning_effort"] = "none"
    return request


def _markers(value: str) -> set[str]:
    lowered = value.casefold().replace("_", " ")
    return {
        name
        for name, words in _SEMANTIC_MARKERS.items()
        if any(word in lowered for word in words)
    }


def _review_parse(value: str) -> Any:
    expanded = value.replace("_per_", "/").replace("_of_", " of ").replace("_", " ")
    return canonicalize_unit(expanded, task="dili")


def _validate_equivalent(source: str, target: str) -> None:
    missing_markers = _markers(source) - _markers(target)
    if missing_markers:
        raise ValueError(
            f"canonical unit drops semantic markers {sorted(missing_markers)}: "
            f"{source!r} -> {target!r}"
        )
    source_unit, target_unit = _review_parse(source), _review_parse(target)
    if not source_unit.unknown_tokens and not target_unit.unknown_tokens:
        same_kind = (
            source_unit.dimension == target_unit.dimension
            and source_unit.transform == target_unit.transform
        )
        same_scale = math.isclose(
            source_unit.scale, target_unit.scale, rel_tol=1e-12, abs_tol=0.0
        )
        if not same_kind or not same_scale:
            raise ValueError(
                f"canonical unit changes parsed coefficient or meaning: "
                f"{source!r} -> {target!r}"
            )


def _parse_decisions(
    response: Mapping[str, Any], items: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    try:
        payload = json.loads(_response_content(response))
    except json.JSONDecodeError as error:
        raise ValueError("DILI unit response content is not JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"decisions"}:
        raise ValueError("DILI unit response must contain only decisions")
    rows = payload["decisions"]
    if not isinstance(rows, list):
        raise TypeError("DILI unit decisions must be a list")
    expected = {str(item["item_id"]): str(item["input_unit"]) for item in items}
    parsed: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("DILI unit decision is not an object")
        item_id = str(row.get("item_id") or "")
        action = str(row.get("action") or "")
        rationale = str(row.get("rationale") or "").strip()
        canonical = str(row.get("canonical_unit") or "").strip()
        allowed = {"item_id", "action", "rationale", "canonical_unit"}
        if item_id not in expected or item_id in parsed or set(row) - allowed:
            raise ValueError(f"invalid or duplicate DILI unit decision: {item_id!r}")
        if action not in {"map", "exclude"} or not rationale:
            raise ValueError(f"invalid DILI unit action or rationale: {item_id}")
        if action == "map":
            if not canonical:
                raise ValueError(f"mapped DILI unit has no target: {item_id}")
            guard_adjustment = ""
            if canonical != expected[item_id]:
                try:
                    _validate_equivalent(expected[item_id], canonical)
                except ValueError as error:
                    rejected = canonical
                    canonical = expected[item_id]
                    guard_adjustment = "unsafe_alias_to_identity"
                    rationale += (
                        f" Proposed alias {rejected!r} was rejected by the "
                        f"deterministic guard ({error}); preserved source identity."
                    )
        elif canonical:
            raise ValueError(f"excluded DILI unit has a target: {item_id}")
        else:
            guard_adjustment = ""
        parsed[item_id] = {
            "item_id": item_id,
            "input_unit": expected[item_id],
            "action": action,
            "canonical_unit": canonical,
            "rationale": rationale,
            "guard_adjustment": guard_adjustment,
        }
    if set(parsed) != set(expected):
        missing = sorted(set(expected) - set(parsed))
        raise ValueError(f"DILI unit response coverage mismatch: {missing[:5]}")
    return [parsed[str(item["item_id"])] for item in items]


def _cache_event(
    *,
    phase: str,
    packet_id: str,
    manifest_sha256: str,
    request: Mapping[str, Any],
    response: Mapping[str, Any] | None,
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "phase": phase,
        "packet_id": packet_id,
        "inventory_manifest_sha256": manifest_sha256,
        "request_sha256": _json_sha256(request),
        "response_sha256": _json_sha256(response) if response is not None else "",
        "status": status,
        "reason": reason,
        "request": request,
        "response": response,
    }


def _append_cache(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_LOCK:
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(_json_text(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _load_cache(
    path: Path, manifest_sha256: str
) -> tuple[dict[tuple[str, str], dict[str, Any]], set[tuple[str, str]]]:
    accepted: dict[tuple[str, str], dict[str, Any]] = {}
    rejected: set[tuple[str, str]] = set()
    if not path.exists():
        return accepted, rejected
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"blank DILI unit cache line: {line_number}")
        event = json.loads(line)
        if event.get("cache_version") != CACHE_VERSION:
            raise ValueError("unsupported DILI unit cache event")
        if event.get("inventory_manifest_sha256") != manifest_sha256:
            raise ValueError("DILI unit cache input drift")
        if event.get("request_sha256") != _json_sha256(event.get("request")):
            raise ValueError("DILI unit cache request hash mismatch")
        if event.get("response") is not None and event.get("response_sha256") != _json_sha256(event["response"]):
            raise ValueError("DILI unit cache response hash mismatch")
        key = (str(event.get("phase")), str(event.get("packet_id")))
        if event.get("status") == "accepted":
            if key in accepted:
                raise ValueError(f"duplicate accepted DILI unit packet: {key}")
            accepted[key] = event
        elif event.get("status") == "rejected":
            rejected.add(key)
    return accepted, rejected


def _call_packet(
    client: Any,
    cache_path: Path,
    manifest_sha256: str,
    phase: str,
    packet_id: str,
    items: Sequence[Mapping[str, Any]],
    max_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    feedback = ""
    max_attempts = (
        PRIMARY_MAX_ATTEMPTS
        if phase in {"primary", "adjudication_fallback"}
        else ADJUDICATION_MAX_ATTEMPTS
    )
    for attempt in range(1, max_attempts + 1):
        request = _request(
            phase, packet_id, items, max_tokens=max_tokens, feedback=feedback
        )
        response: dict[str, Any] | None = None
        try:
            response = _response_payload(client.chat.completions.create(**request))
            if response.get("model") != DEEPSEEK_MODEL:
                raise ValueError(f"returned model mismatch: {response.get('model')!r}")
            decisions = _parse_decisions(response, items)
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            event = _cache_event(
                phase=phase,
                packet_id=packet_id,
                manifest_sha256=manifest_sha256,
                request=request,
                response=response,
                status="rejected",
                reason=reason,
            )
            _append_cache(cache_path, event)
            feedback = reason[:2_000]
            if attempt == max_attempts:
                raise
            continue
        event = _cache_event(
            phase=phase,
            packet_id=packet_id,
            manifest_sha256=manifest_sha256,
            request=request,
            response=response,
            status="accepted",
            reason="",
        )
        _append_cache(cache_path, event)
        return event, decisions
    raise AssertionError("unreachable")


def _endpoint_receipt() -> dict[str, Any]:
    root = DEEPSEEK_BASE_URL.removesuffix("/v1")
    with httpx.Client(timeout=60) as client:
        health = client.get(root + "/health")
        models = client.get(DEEPSEEK_BASE_URL + "/models")
    health.raise_for_status()
    models.raise_for_status()
    payload = models.json()
    model_ids = sorted(
        str(row.get("id"))
        for row in payload.get("data", [])
        if isinstance(row, Mapping) and row.get("id")
    )
    if DEEPSEEK_MODEL not in model_ids:
        raise ValueError(f"dgx027 does not serve {DEEPSEEK_MODEL!r}: {model_ids}")
    return {
        "health_url": root + "/health",
        "health_status": health.status_code,
        "models_url": DEEPSEEK_BASE_URL + "/models",
        "models_status": models.status_code,
        "served_models": model_ids,
    }


def _packets(items: Sequence[Mapping[str, Any]], size: int, prefix: str) -> list[tuple[str, list[Mapping[str, Any]]]]:
    return [
        (f"{prefix}-{offset // size:05d}", list(items[offset : offset + size]))
        for offset in range(0, len(items), size)
    ]


def _cached_decisions(
    event: Mapping[str, Any], items: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    return _parse_decisions(event["response"], items)


def _resolve_packet(
    *,
    phase: str,
    packet_id: str,
    items: Sequence[Mapping[str, Any]],
    client: Any,
    cache_path: Path,
    manifest_sha256: str,
    cached: Mapping[tuple[str, str], Mapping[str, Any]],
    rejected: AbstractSet[tuple[str, str]],
    max_tokens: int,
) -> list[dict[str, str]]:
    event = cached.get((phase, packet_id))
    if event is not None:
        return _cached_decisions(event, items)
    midpoint = len(items) // 2
    fallback_phase = "adjudication_fallback"
    if phase == "adjudication" and not midpoint:
        fallback_event = cached.get((fallback_phase, packet_id))
        if fallback_event is not None:
            return _cached_decisions(fallback_event, items)
        if (phase, packet_id) in rejected:
            _, decisions = _call_packet(
                client,
                cache_path,
                manifest_sha256,
                fallback_phase,
                packet_id,
                items,
                max_tokens,
            )
            return decisions
    child_ids = (packet_id + ".a", packet_id + ".b")
    if midpoint and (
        (phase, packet_id) in rejected
        or any((phase, child_id) in cached for child_id in child_ids)
    ):
        return [
            row
            for child_id, child_items in zip(
                child_ids, (items[:midpoint], items[midpoint:]), strict=True
            )
            for row in _resolve_packet(
                phase=phase,
                packet_id=child_id,
                items=child_items,
                client=client,
                cache_path=cache_path,
                manifest_sha256=manifest_sha256,
                cached=cached,
                rejected=rejected,
                max_tokens=max_tokens,
            )
        ]
    try:
        _, decisions = _call_packet(
            client,
            cache_path,
            manifest_sha256,
            phase,
            packet_id,
            items,
            max_tokens,
        )
        return decisions
    except Exception:
        if not midpoint:
            if phase == "adjudication":
                _, decisions = _call_packet(
                    client,
                    cache_path,
                    manifest_sha256,
                    fallback_phase,
                    packet_id,
                    items,
                    max_tokens,
                )
                return decisions
            raise
        return [
            row
            for child_id, child_items in zip(
                child_ids, (items[:midpoint], items[midpoint:]), strict=True
            )
            for row in _resolve_packet(
                phase=phase,
                packet_id=child_id,
                items=child_items,
                client=client,
                cache_path=cache_path,
                manifest_sha256=manifest_sha256,
                cached=cached,
                rejected=rejected,
                max_tokens=max_tokens,
            )
        ]


def _run_packets(
    packets: Sequence[tuple[str, list[Mapping[str, Any]]]],
    *,
    phase: str,
    client: Any,
    cache_path: Path,
    manifest_sha256: str,
    cached: Mapping[tuple[str, str], Mapping[str, Any]],
    rejected: AbstractSet[tuple[str, str]],
    max_tokens: int,
) -> tuple[list[dict[str, str]], int]:
    results: dict[str, list[dict[str, str]]] = {}
    missing = []
    for packet_id, items in packets:
        event = cached.get((phase, packet_id))
        if event is None:
            missing.append((packet_id, items))
        else:
            results[packet_id] = _cached_decisions(event, items)
    with ThreadPoolExecutor(
        max_workers=min(ENDPOINT_CONCURRENCY_BUDGET, len(missing)) or 1
    ) as pool:
        futures = {
            pool.submit(
                _resolve_packet,
                phase=phase,
                packet_id=packet_id,
                items=items,
                client=client,
                cache_path=cache_path,
                manifest_sha256=manifest_sha256,
                cached=cached,
                rejected=rejected,
                max_tokens=max_tokens,
            ): packet_id
            for packet_id, items in missing
        }
        for future in as_completed(futures):
            packet_id = futures[future]
            results[packet_id] = future.result()
    return (
        [row for packet_id, _ in packets for row in results[packet_id]],
        len(missing),
    )


def _write_outputs(
    output_path: Path,
    decisions: Sequence[Mapping[str, str]],
    receipt: Mapping[str, Any],
) -> None:
    receipt_path = output_path.with_suffix(".manifest.json")
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to replace DILI unit review outputs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    receipt_temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    try:
        temporary.write_text(
            "".join(_json_text(row) + "\n" for row in decisions), encoding="utf-8"
        )
        finalized = dict(receipt)
        finalized["review_sha256"] = file_sha256(temporary)
        receipt_temporary.write_text(_json_text(finalized) + "\n", encoding="utf-8")
        os.replace(temporary, output_path)
        os.replace(receipt_temporary, receipt_path)
    finally:
        temporary.unlink(missing_ok=True)
        receipt_temporary.unlink(missing_ok=True)


def run_review(
    manifest_path: Path,
    cache_path: Path,
    output_path: Path,
    *,
    client: Any | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    if max_tokens < 1:
        raise ValueError("DILI unit max tokens must be positive")
    manifest, inventory = load_inventory_manifest(manifest_path)
    packet_size = int(manifest["packet_size"])
    manifest_sha256 = file_sha256(manifest_path)
    endpoint = _endpoint_receipt() if client is None else {"injected_client": True}
    if client is None:
        client, credential = openai_compatible_client(
            base_url=DEEPSEEK_BASE_URL,
            provider=DEEPSEEK_PROVIDER,
            credential_env=DEEPSEEK_CREDENTIAL_ENV or None,
            max_connections=ENDPOINT_CONCURRENCY_BUDGET,
            timeout_s=1_800,
        )
    else:
        credential = ""
    cached, rejected = _load_cache(cache_path, manifest_sha256)
    primary_packets = _packets(
        [_wire_item(item) for item in inventory["items"]], packet_size, "primary"
    )
    primary, primary_requests = _run_packets(
        primary_packets,
        phase="primary",
        client=client,
        cache_path=cache_path,
        manifest_sha256=manifest_sha256,
        cached=cached,
        rejected=rejected,
        max_tokens=max_tokens,
    )
    by_item = {item["item_id"]: item for item in inventory["items"]}
    adjudication_items = [
        _wire_item(by_item[row["item_id"]], row)
        for row in primary
        if row["action"] == "exclude" or row["canonical_unit"] != row["input_unit"]
    ]
    cached, rejected = _load_cache(cache_path, manifest_sha256)
    adjudication_packets = _packets(
        adjudication_items, min(packet_size, 16), "adjudication"
    )
    adjudicated, adjudication_requests = _run_packets(
        adjudication_packets,
        phase="adjudication",
        client=client,
        cache_path=cache_path,
        manifest_sha256=manifest_sha256,
        cached=cached,
        rejected=rejected,
        max_tokens=max_tokens,
    )
    replacements = {row["item_id"]: row for row in adjudicated}
    final = []
    for row in primary:
        selected = dict(replacements.get(row["item_id"], row))
        selected["review_phase"] = (
            "primary_and_adjudication" if row["item_id"] in replacements else "primary"
        )
        final.append(selected)
    counts = Counter(
        "exclude"
        if row["action"] == "exclude"
        else "alias"
        if row["canonical_unit"] != row["input_unit"]
        else "identity"
        for row in final
    )
    primary_by_item = {row["item_id"]: row for row in primary}
    receipt = {
        "version": RUN_RECEIPT_VERSION,
        "review_version": REVIEW_VERSION,
        "inventory_manifest_path": str(manifest_path.resolve()),
        "inventory_manifest_sha256": manifest_sha256,
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": file_sha256(cache_path),
        "route": {
            "base_url": DEEPSEEK_BASE_URL,
            "provider": DEEPSEEK_PROVIDER,
            "model": DEEPSEEK_MODEL,
            "credential_env": credential,
            "temperature": 0,
            "reasoning": {
                "primary": "none_then_provider_default_on_validation_retry",
                "adjudication": "provider_default",
                "adjudication_singleton_fallback": "none_after_reasoning_failure",
            },
            "max_concurrency": ENDPOINT_CONCURRENCY_BUDGET,
            "max_tokens": max_tokens,
            "max_attempts_before_packet_split": {
                "primary": PRIMARY_MAX_ATTEMPTS,
                "adjudication": ADJUDICATION_MAX_ATTEMPTS,
                "adjudication_singleton_fallback": PRIMARY_MAX_ATTEMPTS,
            },
            "endpoint_check": endpoint,
        },
        "counts": {
            "items": len(final),
            "primary_packets": len(primary_packets),
            "adjudication_items": len(adjudication_items),
            "adjudication_packets": len(adjudication_packets),
            "primary_uncached_packets": primary_requests,
            "adjudication_uncached_packets": adjudication_requests,
            "decisions": dict(sorted(counts.items())),
            "guard_adjusted_aliases": sum(
                row.get("guard_adjustment") == "unsafe_alias_to_identity"
                for row in final
            ),
            "adjudication_changed_decisions": sum(
                row["review_phase"] == "primary_and_adjudication"
                and (row["action"], row["canonical_unit"])
                != (
                    primary_by_item[row["item_id"]]["action"],
                    primary_by_item[row["item_id"]]["canonical_unit"],
                )
                for row in final
            ),
        },
    }
    _write_outputs(output_path, final, receipt)
    return receipt["counts"]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_review(
        args.inventory_manifest,
        args.cache,
        args.output,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CACHE_VERSION", "RUN_RECEIPT_VERSION", "run_review"]
