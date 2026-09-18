"""Run the two-pass, endpoint-independent Skin V10 unit review."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import threading
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.shared.v2.normalization.measurement_resolution import (
    load_exact_unit_mapping,
)
from data.processing.evidence_library.versions.v10.tasks.skin_reaction.data_processing.build_embedding_bucket_mapping import (
    _clusters_from_embeddings,
    _embed_value_groups,
)
from data.processing.evidence_library.versions.v10.tasks.skin_reaction.data_processing.build_unit_reconciliation import (
    _required_units,
)
from data.processing.llm_api import openai_compatible_client
from tools.chembl_tool.common.units import canonicalize_unit

TASK = "skin_reaction"
INVENTORY_VERSION = "skin_reaction_unit_inventory.v2"
REVIEW_VERSION = "skin_reaction_two_pass_unit_review.v1"
MAPPING_VERSION = "starling_exact_measurement_units.v4"
CACHE_VERSION = "skin_reaction_two_pass_unit_review_cache.v1"
RECEIPT_VERSION = "skin_reaction_two_pass_unit_review_run.v1"

REPO_ROOT = Path(__file__).resolve().parents[8]
DEFAULT_CLEANED_RECORDS = (
    REPO_ROOT
    / "data/evidence_libraries/skin_reaction/v10_main_universe_v2/01_cleaned/records.parquet"
)
DEFAULT_RESOLUTION = (
    REPO_ROOT
    / "data/caches/evidence_library/skin_reaction/v10_main_universe_v2/measurement_resolution/measurement_resolution.parquet"
)
DEFAULT_PRIOR_MAPPING = (
    REPO_ROOT
    / "data/caches/evidence_library/skin_reaction/v10_main_universe_v2/unit_reconciliation/exact_measurement_unit_map.json"
)
BASE_URL = "http://dgx020:50002/v1"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
FALLBACK_BASE_URL = "https://api.openai.com/v1"
FALLBACK_MODEL = "gpt-5.4-mini"
FALLBACK_CREDENTIAL_ENV = "OPENAI_API_KEY_ONE"
PACKET_SIZE = 50
MAX_CONCURRENCY = 32
MAX_TOKENS = 524_288
FALLBACK_MAX_TOKENS = 128_000
REQUEST_TIMEOUT_S = 28_800
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLUSTER_RANDOM_SEED = 20260801
_CACHE_LOCK = threading.Lock()

_CLEANED_COLUMNS = (
    "cleaned_record_id",
    "source_row_uid",
    "source_id",
    "canonical_endpoint_name",
    "endpoint_name",
    "measurement_text",
    "unit_text",
    "support_text",
    "measurement_resolution_route",
    "measurement_resolution_exact_measurement",
    "measurement_resolution_exact_unit",
    "measurement_resolution_exact_unit_is_canonical",
)
_RESOLUTION_COLUMNS = (
    "cleaned_record_id",
    "source_row_uid",
    "source_id",
    "status",
    "measurements_json",
)
_ORDINAL = re.compile(r"\b(?:index|score|grade|rating|scale)\b", re.IGNORECASE)
_DENOMINATORS = (
    re.compile(r"\bout of\s+(\d+(?:\.0+)?)\b", re.IGNORECASE),
    re.compile(r"/(\d+)\s*$", re.IGNORECASE),
    re.compile(
        r"\bper\s+(\d+)\s+(?:mice|rats|animals|subjects|patients|people|volunteers|guinea\s+pigs)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\btotal\s+N\s*=\s*(\d+)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:in|of)\s+(\d+)\s+(?:mice|rats|animals|subjects|patients|people|volunteers|guinea\s+pigs)\b",
        re.IGNORECASE,
    ),
)
_SEMANTIC_MARKERS = {
    "positive": ("positive", "positivity", "responders", "sensitized"),
    "negative": ("negative",),
    "reaction": ("reaction", "reacting"),
    "questionable": ("questionable",),
    "control": ("control", "vehicle", "baseline"),
    "inhibition": ("inhibition", "inhibited"),
    "fold": ("fold", "times"),
    "fraction": ("fraction",),
    "percent": ("percent", "%"),
    "ordinal": ("index", "score", "grade", "rating", "scale"),
}

_PRIMARY_PROMPT = """You reconcile exact Skin Reaction measurement-unit strings. Return JSON only.

Review all supplied items together. For every input_unit, return action "map" or
"exclude". A mapped canonical_unit may normalize spelling, Unicode typography,
separators, long-form names, or an algebraically equivalent expression, but it
must preserve the supplied numeric coefficient. For example, nmol/L may become
nM and ng/cm2 may become ng/cm². Never merge nM with mM, or ng/cm² with µg/cm².

Preserve semantic qualifiers: percent, percent of control, fold, positive,
negative, reaction, questionable, and named ratios are distinct. A genuine
count "out of N" maps to a qualified fraction and scale 1/N. Use positive
fraction, negative fraction, reaction fraction, or questionable fraction only
when the input says so; otherwise use neutral fraction. Ordinal scores are not
fractions. All other valid units use scale "1". Exclude only prose, conditions,
identifiers, or missing markers that are not measurement units or named metrics.

Return exactly {"decisions":[...]}. Each decision has item_id, action, scale,
and rationale. A map also has canonical_unit. An exclusion has canonical_unit
"" and scale "". Cover every supplied item exactly once.
"""

_GLOBAL_PROMPT = """Globally reconcile proposed Skin Reaction canonical units. Return JSON only.

Each item is a canonical proposal produced by a prior complete review, with its
source aliases and prior scales. Unify spelling and algebraically equivalent
same-coefficient forms that were separated across earlier batches. Preserve
physical scale and every semantic qualifier. Never merge nM with mM, positive
fraction with negative/reaction/questionable/neutral fraction, percent with
fraction, or a named ratio with a different ratio. An exclusion may be corrected
to a real unit when the supplied aliases establish that clearly.

Return exactly {"decisions":[...]}. Each decision has item_id, action, and a
concise rationale. A map also has canonical_unit; an exclusion has no
canonical_unit. Cover every supplied item exactly once.
"""


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _json_text(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    ) + ("\n" if pretty else "")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_text(value).encode("utf-8")).hexdigest()


def _item_id(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _resolution_rows(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for batch in pq.ParquetFile(path).iter_batches(columns=_RESOLUTION_COLUMNS):
        for row in batch.to_pylist():
            record_id = _text(row["cleaned_record_id"])
            if not record_id or record_id in rows:
                raise ValueError(f"duplicate or empty resolution ID: {record_id!r}")
            rows[record_id] = {name: _text(row[name]) for name in _RESOLUTION_COLUMNS}
    return rows


def _prior_decisions(path: Path) -> dict[str, list[dict[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output: dict[str, set[tuple[str, str, str, str]]] = defaultdict(set)
    for entry in payload.get("entries", []):
        output[_text(entry.get("input_unit"))].add(
            (
                _text(entry.get("action")),
                _text(entry.get("canonical_unit")),
                _text(entry.get("scale")),
                _text(entry.get("domain")) or "any",
            )
        )
    return {
        unit: [
            {"action": action, "canonical_unit": canonical, "scale": scale, "domain": domain}
            for action, canonical, scale, domain in sorted(decisions)
        ]
        for unit, decisions in output.items()
    }


def _parser_evidence(unit: str) -> dict[str, Any]:
    parsed = canonicalize_unit(unit, task=TASK)
    return {
        "canonical": parsed.canonical,
        "scale": parsed.scale,
        "dimension": [list(value) for value in parsed.dimension],
        "unknown_tokens": list(parsed.unknown_tokens),
        "transform": parsed.transform,
    }


def _representative(row: Mapping[str, Any], origin: str, unit: str) -> dict[str, str]:
    return {
        "origin": origin,
        "source_id": _text(row.get("source_id")),
        "source_row_uid": _text(row.get("source_row_uid")),
        "endpoint": _text(row.get("canonical_endpoint_name")),
        "source_measurement": _text(row.get("measurement_text"))[:240],
        "unit": unit,
        "support_excerpt": _text(row.get("support_text"))[:320],
    }


def build_inventory(
    cleaned_path: Path = DEFAULT_CLEANED_RECORDS,
    resolution_path: Path = DEFAULT_RESOLUTION,
    prior_mapping_path: Path = DEFAULT_PRIOR_MAPPING,
) -> dict[str, Any]:
    """Build the active source + exact-route + successful-LLM unit universe."""
    required, counts = _required_units(
        cleaned_path, resolution_path, allow_out_of_scope_resolution=True
    )
    active_units = {unit for _, unit in required}
    resolution = _resolution_rows(resolution_path)
    prior = _prior_decisions(prior_mapping_path)
    groups: dict[str, dict[str, Any]] = {
        unit: {
            "origin_counts": Counter(),
            "endpoint_counts": Counter(),
            "representatives": {},
        }
        for unit in active_units
    }
    seen_resolution: set[str] = set()
    for batch in pq.ParquetFile(cleaned_path).iter_batches(columns=_CLEANED_COLUMNS):
        for row in batch.to_pylist():
            record_id = _text(row["cleaned_record_id"])
            endpoint = _text(row["canonical_endpoint_name"])
            observations: list[tuple[str, str]] = []
            source_unit = _text(row["unit_text"])
            if source_unit:
                observations.append(("source", source_unit))
            if (
                _text(row["measurement_resolution_route"]) == "accept"
                and not row["measurement_resolution_exact_unit_is_canonical"]
            ):
                observations.append(
                    ("source_exact", _text(row["measurement_resolution_exact_unit"]) or source_unit)
                )
            resolved = resolution.get(record_id)
            if resolved is not None:
                seen_resolution.add(record_id)
                if _text(row["measurement_resolution_route"]) == "extract" and resolved["status"] == "ok":
                    measurements = json.loads(resolved["measurements_json"])
                    if len(measurements) != 1:
                        raise ValueError(f"resolved row is not scalar: {record_id}")
                    observations.append(("llm", _text(measurements[0].get("unit"))))
            for origin, unit in observations:
                if unit not in groups:
                    raise ValueError(f"inventory observation escaped required universe: {unit!r}")
                group = groups[unit]
                group["origin_counts"][origin] += 1
                group["endpoint_counts"][endpoint] += 1
                selector = (_text(row["source_id"]), _text(row["source_row_uid"]), record_id)
                current = group["representatives"].get(origin)
                if current is None or selector < current[0]:
                    group["representatives"][origin] = (
                        selector,
                        _representative(row, origin, unit),
                    )
    items = []
    for unit in sorted(groups, key=lambda value: (value.casefold(), value)):
        group = groups[unit]
        items.append(
            {
                "item_id": _item_id("u_", unit),
                "input_unit": unit,
                "origin_counts": dict(sorted(group["origin_counts"].items())),
                "endpoint_counts": dict(group["endpoint_counts"].most_common(12)),
                "representative_contexts": [
                    value[1] for _, value in sorted(group["representatives"].items())
                ],
                "parser_output": _parser_evidence(unit),
                "prior_reviewed_decisions": prior.get(unit, []),
            }
        )
    if len(items) != counts["unique_units"]:
        raise ValueError("Skin active-unit inventory count mismatch")
    return {
        "version": INVENTORY_VERSION,
        "task": TASK,
        "scope": "active_v2_source_exact_and_successful_llm_units",
        "inputs": {
            "cleaned_records": {"path": str(cleaned_path.resolve()), "sha256": file_sha256(cleaned_path)},
            "measurement_resolution": {"path": str(resolution_path.resolve()), "sha256": file_sha256(resolution_path)},
            "prior_mapping": {"path": str(prior_mapping_path.resolve()), "sha256": file_sha256(prior_mapping_path)},
        },
        "counts": counts,
        "items": items,
    }


def write_inventory(output_dir: Path) -> dict[str, Any]:
    inventory_path = output_dir / "inventory.json"
    manifest_path = output_dir / "inventory.manifest.json"
    if inventory_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to replace inventory in {output_dir}")
    inventory = build_inventory()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(_json_text(inventory, pretty=True), encoding="utf-8")
    item_count = len(inventory["items"])
    manifest = {
        "version": "skin_reaction_unit_review_packets.v2",
        "task": TASK,
        "inventory": {"path": str(inventory_path.resolve()), "sha256": file_sha256(inventory_path)},
        "inputs": inventory["inputs"],
        "item_count": item_count,
        "packet_size": PACKET_SIZE,
        "packet_count": math.ceil(item_count / PACKET_SIZE),
    }
    manifest_path.write_text(_json_text(manifest, pretty=True), encoding="utf-8")
    return manifest


def load_inventory(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != "skin_reaction_unit_review_packets.v2":
        raise ValueError("unsupported Skin unit inventory manifest")
    reference = manifest["inventory"]
    inventory_path = Path(reference["path"])
    if file_sha256(inventory_path) != reference["sha256"]:
        raise ValueError("Skin unit inventory hash mismatch")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("version") != INVENTORY_VERSION or len(inventory.get("items", [])) != manifest["item_count"]:
        raise ValueError("unsupported or incomplete Skin unit inventory")
    for reference in inventory["inputs"].values():
        if file_sha256(Path(reference["path"])) != reference["sha256"]:
            raise ValueError(f"Skin unit input hash mismatch: {reference['path']}")
    return manifest, inventory


def _markers(value: str) -> set[str]:
    lowered = value.casefold().replace("_", " ")
    return {
        marker
        for marker, words in _SEMANTIC_MARKERS.items()
        if any(word in lowered for word in words)
    }


def _ratio_qualifier(value: str) -> str:
    markers = _markers(value)
    for marker in ("questionable", "negative", "positive", "reaction"):
        if marker in markers:
            return marker
    return "neutral"


def _ratio_denominator(value: str) -> int | None:
    if _ORDINAL.search(value):
        return None
    for pattern in _DENOMINATORS:
        match = pattern.search(value)
        if match:
            denominator = Decimal(match.group(1))
            if denominator == denominator.to_integral_value() and denominator > 0:
                return int(denominator)
    return None


def _validate_alias(source: str, target: str) -> None:
    denominator = _ratio_denominator(source)
    if denominator is not None:
        expected = _ratio_qualifier(source) + " fraction"
        if target.casefold() != expected:
            raise ValueError(f"explicit denominator must map to {expected!r}")
        return
    missing = _markers(source) - _markers(target)
    if missing:
        raise ValueError(f"canonical unit drops semantic markers {sorted(missing)}")
    source_parsed = canonicalize_unit(source, task=TASK)
    target_parsed = canonicalize_unit(target, task=TASK)
    if not source_parsed.unknown_tokens and not target_parsed.unknown_tokens:
        same_kind = (
            source_parsed.dimension == target_parsed.dimension
            and source_parsed.transform == target_parsed.transform
        )
        same_scale = math.isclose(source_parsed.scale, target_parsed.scale, rel_tol=1e-12)
        if not same_kind or not same_scale:
            raise ValueError(f"canonical unit changes coefficient or meaning: {source!r} -> {target!r}")


def _validate_scale(source: str, scale_text: str) -> str:
    try:
        numerator, separator, denominator_text = scale_text.partition("/")
        scale = (
            Decimal(numerator) / Decimal(denominator_text)
            if separator
            else Decimal(scale_text)
        )
    except (InvalidOperation, ZeroDivisionError) as error:
        raise ValueError(f"invalid unit scale: {scale_text!r}") from error
    if not scale.is_finite() or scale <= 0:
        raise ValueError(f"invalid unit scale: {scale_text!r}")
    denominator = _ratio_denominator(source)
    expected = Decimal(1) / Decimal(denominator) if denominator else Decimal(1)
    return str(expected)


def _response_payload(response: Any) -> dict[str, Any]:
    payload = response if isinstance(response, Mapping) else response.model_dump(mode="json")
    return json.loads(_json_text(payload))


def _response_content(response: Mapping[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Skin unit response has no message content") from error
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Skin unit response content is empty")
    return content


def _parse_decisions(
    response: Mapping[str, Any], items: Sequence[Mapping[str, Any]], phase: str
) -> list[dict[str, str]]:
    payload = json.loads(_response_content(response))
    if not isinstance(payload, dict) or set(payload) != {"decisions"} or not isinstance(payload["decisions"], list):
        raise ValueError("Skin unit response must contain only a decisions list")
    source_field = "input_unit" if phase == "first" else "proposed_unit"
    expected = {str(item["item_id"]): str(item[source_field]) for item in items}
    output: dict[str, dict[str, str]] = {}
    for row in payload["decisions"]:
        if not isinstance(row, dict):
            raise TypeError("Skin unit decision is not an object")
        item_id = _text(row.get("item_id"))
        action = _text(row.get("action"))
        canonical = _text(row.get("canonical_unit"))
        rationale = _text(row.get("rationale"))
        scale = _text(row.get("scale")) if phase == "first" else ""
        allowed = {"item_id", "action", "canonical_unit", "rationale"} | ({"scale"} if phase == "first" else set())
        if item_id not in expected or item_id in output or set(row) - allowed:
            raise ValueError(f"invalid or duplicate Skin unit decision: {item_id!r}")
        if action not in {"map", "exclude"} or not rationale:
            raise ValueError(f"invalid Skin unit action: {item_id}")
        if action == "map":
            if not canonical:
                raise ValueError(f"mapped Skin unit has no target: {item_id}")
            guard_adjustment = ""
            try:
                _validate_alias(expected[item_id], canonical)
            except ValueError as error:
                rejected = canonical
                if _ratio_denominator(expected[item_id]) is not None:
                    canonical = _ratio_qualifier(expected[item_id]) + " fraction"
                    guard_adjustment = "ratio_qualifier_corrected"
                else:
                    canonical = expected[item_id]
                    guard_adjustment = "unsafe_alias_to_identity"
                rationale += (
                    f" Proposed alias {rejected!r} was rejected by the "
                    f"deterministic guard ({error}); applied {canonical!r}."
                )
            if phase == "first":
                proposed_scale = scale
                scale = _validate_scale(expected[item_id], scale)
                if proposed_scale != scale:
                    guard_adjustment = "+".join(
                        value
                        for value in (guard_adjustment, "scale_normalized")
                        if value
                    )
                    rationale += (
                        f" Deterministic scale contract normalized "
                        f"{proposed_scale!r} to {scale!r}."
                    )
        elif canonical or scale:
            raise ValueError(f"excluded Skin unit has a target or scale: {item_id}")
        else:
            guard_adjustment = ""
        output[item_id] = {
            "item_id": item_id,
            "action": action,
            "canonical_unit": canonical,
            "scale": scale,
            "rationale": rationale,
            "guard_adjustment": guard_adjustment,
        }
    if set(output) != set(expected):
        raise ValueError(f"Skin unit response coverage mismatch: {sorted(set(expected) - set(output))[:5]}")
    return [output[str(item["item_id"])] for item in items]


def _request(
    phase: str,
    packet_id: str,
    items: Sequence[Mapping[str, Any]],
    feedback: str = "",
    *,
    model: str = MODEL,
) -> dict[str, Any]:
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": _PRIMARY_PROMPT if phase == "first" else _GLOBAL_PROMPT},
            {
                "role": "user",
                "content": _json_text(
                    {
                        "phase": phase,
                        "packet_id": packet_id,
                        "item_count": len(items),
                        "validation_feedback": feedback,
                        "items": list(items),
                    }
                ),
            },
        ],
        "reasoning_effort": "high",
        "response_format": {"type": "json_object"},
    }
    if model == FALLBACK_MODEL:
        request["max_completion_tokens"] = FALLBACK_MAX_TOKENS
    else:
        request.update(temperature=0, max_tokens=MAX_TOKENS)
    return request


def _load_cache(path: Path, manifest_sha256: str) -> dict[tuple[str, str], dict[str, Any]]:
    accepted: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return accepted
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        event = json.loads(line)
        if not line or event.get("cache_version") != CACHE_VERSION:
            raise ValueError(f"invalid Skin unit cache line: {line_number}")
        if event.get("inventory_manifest_sha256") != manifest_sha256:
            raise ValueError("Skin unit cache input drift")
        if event.get("request_sha256") != _json_sha256(event.get("request")):
            raise ValueError("Skin unit cache request hash mismatch")
        if event.get("response") is not None and event.get("response_sha256") != _json_sha256(event["response"]):
            raise ValueError("Skin unit cache response hash mismatch")
        if event.get("status") == "accepted":
            key = (_text(event.get("phase")), _text(event.get("packet_id")))
            if key in accepted:
                raise ValueError(f"duplicate accepted Skin unit packet: {key}")
            accepted[key] = event
    return accepted


def _accepted_response_provenance(
    path: Path, manifest_sha256: str
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str, int, bool]] = Counter()
    for event in _load_cache(path, manifest_sha256).values():
        request = event["request"]
        ceiling = request.get("max_tokens", request.get("max_completion_tokens", 0))
        counts[
            (
                str(event["phase"]),
                str(event["route"]),
                str(request["model"]),
                int(ceiling),
                bool(event.get("reuse")),
            )
        ] += 1
    return [
        {
            "phase": phase,
            "route": route,
            "model": model,
            "max_output_tokens": ceiling,
            "reused": reused,
            "accepted_packets": count,
        }
        for (phase, route, model, ceiling, reused), count in sorted(counts.items())
    ]


def _append_cache(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_LOCK:
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(_json_text(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _call_packet(
    client: Any,
    fallback_client: Any,
    cache_path: Path,
    manifest_sha256: str,
    phase: str,
    packet_id: str,
    items: Sequence[Mapping[str, Any]],
    cached: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, str]]:
    cached_event = cached.get((phase, packet_id))
    if cached_event is not None:
        return _parse_decisions(cached_event["response"], items, phase)
    feedback = ""
    for attempt in range(1, 4):
        use_fallback = attempt == 3
        requested_model = FALLBACK_MODEL if use_fallback else MODEL
        request = _request(
            phase, packet_id, items, feedback, model=requested_model
        )
        response = None
        status = "rejected"
        reason = ""
        try:
            selected_client = fallback_client if use_fallback else client
            response = _response_payload(
                selected_client.chat.completions.create(**request)
            )
            returned_model = _text(response.get("model"))
            if not (
                returned_model == requested_model
                or (
                    use_fallback
                    and returned_model.startswith(FALLBACK_MODEL + "-")
                )
            ):
                raise ValueError(f"returned model mismatch: {response.get('model')!r}")
            decisions = _parse_decisions(response, items, phase)
            status = "accepted"
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            feedback = reason[:2_000]
        event = {
            "cache_version": CACHE_VERSION,
            "inventory_manifest_sha256": manifest_sha256,
            "phase": phase,
            "packet_id": packet_id,
            "attempt": attempt,
            "route": "openai_fallback" if use_fallback else "dgx020_local",
            "status": status,
            "reason": reason,
            "request": request,
            "request_sha256": _json_sha256(request),
            "response": response,
            "response_sha256": _json_sha256(response) if response is not None else "",
        }
        _append_cache(cache_path, event)
        if status == "accepted":
            return decisions
    raise RuntimeError(f"Skin unit packet failed after three attempts: {phase}/{packet_id}: {feedback}")


def _packets(items: Sequence[Mapping[str, Any]], prefix: str) -> list[tuple[str, list[Mapping[str, Any]]]]:
    return [
        (f"{prefix}-{offset // PACKET_SIZE:05d}", list(items[offset : offset + PACKET_SIZE]))
        for offset in range(0, len(items), PACKET_SIZE)
    ]


def _run_packets(
    packets: Sequence[tuple[str, list[Mapping[str, Any]]]],
    *,
    phase: str,
    client: Any,
    fallback_client: Any,
    cache_path: Path,
    manifest_sha256: str,
    cached: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, str]]:
    results: dict[str, list[dict[str, str]]] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENCY, len(packets)) or 1) as pool:
        futures = {
            pool.submit(
                _call_packet,
                client,
                fallback_client,
                cache_path,
                manifest_sha256,
                phase,
                packet_id,
                items,
                cached,
            ): packet_id
            for packet_id, items in packets
        }
        for future in as_completed(futures):
            packet_id = futures[future]
            results[packet_id] = future.result()
            print(f"[{phase}] completed={len(results)}/{len(packets)} packet={packet_id}", flush=True)
    return [row for packet_id, _ in packets for row in results[packet_id]]


def _first_wire(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "item_id": item["item_id"],
        "input_unit": item["input_unit"],
        "origin_counts": item["origin_counts"],
        "endpoint_counts": item["endpoint_counts"],
        "representative_contexts": item["representative_contexts"],
        "parser_output": item["parser_output"],
        "prior_reviewed_decisions": item["prior_reviewed_decisions"],
    }


def _global_items(
    first: Sequence[Mapping[str, str]], inventory_items: Sequence[Mapping[str, Any]], work_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    input_by_id = {str(item["item_id"]): item for item in inventory_items}
    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in first:
        value = row["canonical_unit"] if row["action"] == "map" else str(input_by_id[row["item_id"]]["input_unit"])
        grouped[(row["action"], value)].append(row)
    values = sorted({value for _, value in grouped}, key=lambda value: (value.casefold(), value))
    embeddings = _embed_value_groups(
        {"skin_unit_proposals": values}, model_name=EMBEDDING_MODEL, batch_size=256, device="cpu"
    )["skin_unit_proposals"]
    clusters = _clusters_from_embeddings(
        values,
        embeddings,
        clustering="lloyd",
        target_size=PACKET_SIZE,
        max_labels_per_call=PACKET_SIZE,
    )
    cluster_by_value = {value: cluster.cluster_id for cluster in clusters for value in cluster.values}
    if set(cluster_by_value) != set(values):
        raise ValueError("global clustering lost proposed canonical units")
    cluster_payload = {
        "version": "skin_reaction_unit_global_clusters.v1",
        "method": {
            "embedding_model": EMBEDDING_MODEL,
            "algorithm": "lloyd",
            "random_seed": CLUSTER_RANDOM_SEED,
            "target_size": PACKET_SIZE,
        },
        "clusters": [{"cluster_id": cluster.cluster_id, "values": list(cluster.values)} for cluster in clusters],
    }
    cluster_path = work_dir / "global_clusters.json"
    cluster_path.write_text(_json_text(cluster_payload, pretty=True), encoding="utf-8")
    output = []
    first_to_global: dict[str, str] = {}
    for action, value in sorted(grouped, key=lambda pair: (cluster_by_value[pair[1]], pair[1].casefold(), pair)):
        rows = grouped[(action, value)]
        item_id = _item_id("p_", action + "\0" + value)
        aliases = sorted(
            {str(input_by_id[row["item_id"]]["input_unit"]) for row in rows},
            key=lambda text: (text.casefold(), text),
        )
        for row in rows:
            first_to_global[row["item_id"]] = item_id
        output.append(
            {
                "item_id": item_id,
                "proposed_unit": value,
                "first_action": action,
                "raw_alias_count": len(aliases),
                "source_aliases": aliases[:30],
                "first_scales": sorted({row["scale"] for row in rows if row["scale"]}),
                "cluster_id": cluster_by_value[value],
            }
        )
    output.sort(key=lambda row: (row["cluster_id"], row["proposed_unit"].casefold(), row["item_id"]))
    return output, first_to_global


def _global_packets(items: Sequence[Mapping[str, Any]]) -> list[tuple[str, list[Mapping[str, Any]]]]:
    by_cluster: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        by_cluster[str(item["cluster_id"])].append(item)
    packets = []
    for index, cluster_id in enumerate(sorted(by_cluster)):
        cluster_items = by_cluster[cluster_id]
        if len(cluster_items) > PACKET_SIZE:
            raise ValueError(f"oversized global cluster: {cluster_id}")
        packets.append((f"global-{index:05d}-{cluster_id}", cluster_items))
    return packets


def _endpoint_receipt() -> dict[str, Any]:
    root = BASE_URL.removesuffix("/v1")
    with httpx.Client(timeout=30) as client:
        health = client.get(root + "/health")
        models = client.get(BASE_URL + "/models")
        loads = client.get(BASE_URL + "/loads")
    health.raise_for_status()
    models.raise_for_status()
    loads.raise_for_status()
    model_ids = sorted(str(row["id"]) for row in models.json().get("data", []) if row.get("id"))
    if model_ids != [MODEL]:
        raise ValueError(f"unexpected dgx020:50002 models: {model_ids}")
    load_payload = loads.json()
    return {
        "health_status": health.status_code,
        "models": model_ids,
        "loads_status": loads.status_code,
        "running": sum(int(row.get("num_running_reqs", 0)) for row in load_payload.get("loads", [])),
        "waiting": sum(int(row.get("num_waiting_reqs", 0)) for row in load_payload.get("loads", [])),
    }


def run_review(manifest_path: Path, cache_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists() or output_path.with_suffix(".manifest.json").exists():
        raise FileExistsError("refusing to replace Skin two-pass review output")
    manifest, inventory = load_inventory(manifest_path)
    if manifest["packet_size"] != PACKET_SIZE:
        raise ValueError("Skin first-pass packet size must be exactly 50")
    endpoint = _endpoint_receipt()
    client, credential = openai_compatible_client(
        base_url=BASE_URL,
        provider="local",
        max_connections=MAX_CONCURRENCY,
        timeout_s=REQUEST_TIMEOUT_S,
        max_retries=0,
    )
    fallback_client, fallback_credential = openai_compatible_client(
        base_url=FALLBACK_BASE_URL,
        provider="openai",
        credential_env=FALLBACK_CREDENTIAL_ENV,
        max_connections=MAX_CONCURRENCY,
        timeout_s=REQUEST_TIMEOUT_S,
        max_retries=0,
    )
    manifest_sha256 = file_sha256(manifest_path)
    cached = _load_cache(cache_path, manifest_sha256)
    first_packets = _packets([_first_wire(item) for item in inventory["items"]], "first")
    first = _run_packets(
        first_packets,
        phase="first",
        client=client,
        fallback_client=fallback_client,
        cache_path=cache_path,
        manifest_sha256=manifest_sha256,
        cached=cached,
    )
    global_items, first_to_global = _global_items(first, inventory["items"], output_path.parent)
    cached = _load_cache(cache_path, manifest_sha256)
    global_packets = _global_packets(global_items)
    global_rows = _run_packets(
        global_packets,
        phase="global",
        client=client,
        fallback_client=fallback_client,
        cache_path=cache_path,
        manifest_sha256=manifest_sha256,
        cached=cached,
    )
    global_by_id = {row["item_id"]: row for row in global_rows}
    input_by_id = {item["item_id"]: item for item in inventory["items"]}
    final = []
    for first_row in first:
        global_row = global_by_id[first_to_global[first_row["item_id"]]]
        source = input_by_id[first_row["item_id"]]["input_unit"]
        row = {
            "item_id": first_row["item_id"],
            "input_unit": source,
            "action": global_row["action"],
            "canonical_unit": global_row["canonical_unit"],
            "scale": (
                first_row["scale"]
                if global_row["action"] == "map" and first_row["action"] == "map"
                else str(Decimal(1) / Decimal(_ratio_denominator(source)))
                if global_row["action"] == "map" and _ratio_denominator(source)
                else "1"
                if global_row["action"] == "map"
                else ""
            ),
            "first_pass": first_row,
            "global_pass": global_row,
        }
        if row["action"] == "map":
            _validate_alias(source, row["canonical_unit"])
            row["scale"] = _validate_scale(source, row["scale"])
        final.append(row)
    if len(final) != manifest["item_count"]:
        raise ValueError("final Skin unit review coverage mismatch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text("".join(_json_text(row) + "\n" for row in final), encoding="utf-8")
    receipt = {
        "version": RECEIPT_VERSION,
        "review_version": REVIEW_VERSION,
        "inventory_manifest": {"path": str(manifest_path.resolve()), "sha256": manifest_sha256},
        "cache": {"path": str(cache_path.resolve()), "sha256": file_sha256(cache_path)},
        "review_sha256": file_sha256(temporary),
        "route": {
            "base_url": BASE_URL,
            "provider": "local",
            "model": MODEL,
            "credential_env": credential,
            "reasoning_effort": "high",
            "max_tokens": MAX_TOKENS,
            "timeout_s": REQUEST_TIMEOUT_S,
            "max_concurrency": MAX_CONCURRENCY,
            "endpoint_preflight": endpoint,
            "retry_policy": {
                "attempt_1": {"base_url": BASE_URL, "model": MODEL},
                "retry_1": {"base_url": BASE_URL, "model": MODEL},
                "retry_2": {
                    "base_url": FALLBACK_BASE_URL,
                    "model": FALLBACK_MODEL,
                    "credential_env": fallback_credential,
                },
            },
            "accepted_response_provenance": _accepted_response_provenance(
                cache_path, manifest_sha256
            ),
        },
        "counts": {
            "raw_units": len(final),
            "first_packets": len(first_packets),
            "first_full_packets": sum(len(items) == PACKET_SIZE for _, items in first_packets),
            "first_tail_items": len(first_packets[-1][1]),
            "global_proposals": len(global_items),
            "global_packets": len(global_packets),
            "mapped": sum(row["action"] == "map" for row in final),
            "excluded": sum(row["action"] == "exclude" for row in final),
        },
    }
    receipt_path = output_path.with_suffix(".manifest.json")
    receipt_temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    receipt_temporary.write_text(_json_text(receipt, pretty=True), encoding="utf-8")
    os.replace(temporary, output_path)
    os.replace(receipt_temporary, receipt_path)
    return receipt["counts"]


def build_mapping(manifest_path: Path, review_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _, inventory = load_inventory(manifest_path)
    receipt_path = review_path.with_suffix(".manifest.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("version") != RECEIPT_VERSION or receipt.get("review_sha256") != file_sha256(review_path):
        raise ValueError("Skin two-pass review receipt mismatch")
    rows = [json.loads(line) for line in review_path.read_text(encoding="utf-8").splitlines() if line]
    expected = {item["item_id"]: item for item in inventory["items"]}
    if len(rows) != len(expected) or {row.get("item_id") for row in rows} != set(expected):
        raise ValueError("Skin two-pass review is incomplete")
    entries = []
    for row in rows:
        item = expected[row["item_id"]]
        if row["input_unit"] != item["input_unit"]:
            raise ValueError(f"Skin unit review identity mismatch: {row['item_id']}")
        entry = {
            "task": TASK,
            "canonical_endpoints": ["*"],
            "input_unit": row["input_unit"],
            "action": row["action"],
            "domain": "nonnegative" if row.get("canonical_unit", "").endswith("fraction") else "any",
            "review_basis": "deepseek_two_pass_global_reconciliation",
            "review_item_id": row["item_id"],
        }
        if row["action"] == "map":
            entry.update(canonical_unit=row["canonical_unit"], scale=row["scale"])
        entries.append(entry)
    entries.sort(key=lambda row: (row["input_unit"].casefold(), row["input_unit"]))
    contract = {
        "version": REVIEW_VERSION,
        "scope": inventory["scope"],
        "inputs": inventory["inputs"],
        "inventory_manifest": {"path": str(manifest_path.resolve()), "sha256": file_sha256(manifest_path)},
        "review": {
            "path": str(review_path.resolve()),
            "sha256": file_sha256(review_path),
            "receipt": {"path": str(receipt_path.resolve()), "sha256": file_sha256(receipt_path)},
            "route": receipt["route"],
        },
        "counts": {**inventory["counts"], **receipt["counts"]},
        "validations": {
            "complete_active_unit_coverage": True,
            "wildcard_endpoint_only": True,
            "two_complete_review_passes": True,
            "non_ratio_scale_is_one": True,
            "frozen_input_hashes_verified": True,
        },
    }
    payload = {"version": MAPPING_VERSION, "entries": entries, "skin_v10_unit_review": contract}
    report = {
        "version": "skin_reaction_unit_reconciliation_manifest.v4",
        "task": TASK,
        "mapping_version": MAPPING_VERSION,
        "counts": contract["counts"],
        "inputs": contract["inputs"],
        "review": contract["review"],
        "validations": contract["validations"],
    }
    return payload, report


def write_mapping(manifest_path: Path, review_path: Path, output_path: Path) -> dict[str, Any]:
    report_path = output_path.with_suffix(".manifest.json")
    if output_path.exists() or report_path.exists():
        raise FileExistsError(f"refusing to replace Skin unit map: {output_path}")
    payload, report = build_mapping(manifest_path, review_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    report_temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(_json_text(payload, pretty=True), encoding="utf-8")
    mapping = load_exact_unit_mapping(temporary)
    if len(mapping) != len(payload["entries"]):
        raise ValueError("Skin v4 wildcard map expansion changed entry count")
    report["mapping"] = {"path": str(output_path.resolve()), "sha256": file_sha256(temporary)}
    report_temporary.write_text(_json_text(report, pretty=True), encoding="utf-8")
    os.replace(temporary, output_path)
    os.replace(report_temporary, report_path)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, required=True)
    review = commands.add_parser("review")
    review.add_argument("--inventory-manifest", type=Path, required=True)
    review.add_argument("--cache", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    consolidate = commands.add_parser("consolidate")
    consolidate.add_argument("--inventory-manifest", type=Path, required=True)
    consolidate.add_argument("--reviews", type=Path, required=True)
    consolidate.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = write_inventory(args.output_dir)
        output = {key: result[key] for key in ("item_count", "packet_count", "packet_size")}
    elif args.command == "review":
        output = run_review(args.inventory_manifest, args.cache, args.output)
    else:
        output = write_mapping(args.inventory_manifest, args.reviews, args.output)["counts"]
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASE_URL",
    "MODEL",
    "PACKET_SIZE",
    "build_inventory",
    "build_mapping",
    "run_review",
]
