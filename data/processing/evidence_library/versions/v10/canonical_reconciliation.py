"""Canonical reconciliation for V10 units and auxiliary values.

Units use one context-grouped LLM pass followed by a guarded agent review.
Auxiliary categorical values continue to use the shared embedding-clustered
implementation. Every unit survives: uncertainty and request failure are identity
mappings, and every published value scale remains exactly one.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import itertools
import json
import math
import os
import re
import shutil
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.clustered_auxiliary_mapping import (
    AuxiliaryExtractionSpec,
    build_clustered_auxiliary_mapping,
    reconciliation_plan,
)
from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from data.processing.evidence_library.shared.v2.normalization.measurement_resolution import (
    EXACT_UNIT_MAPPING_VERSION,
    load_exact_unit_mapping,
)
from data.processing.llm_api import (
    DEFAULT_ENV_FILE,
    async_openai_compatible_client,
    openai_compatible_client,
    resolve_api_key,
)
from data.processing.paths import evidence_library_root
from tools.chembl_tool.common.units import canonicalize_unit

PROTOCOL_VERSION = "starling_canonical_reconciliation_single_pass.v1"
REVIEW_BASIS = "single_pass_llm_plus_agent_review.v1"
LEGACY_PASS_A_VERSIONS = frozenset(
    {
        "starling_unit_reconciliation_two_pass.v1",
        "starling_unit_reconciliation_two_pass.v2",
    }
)
GROUP_SIZE = 50
LOCAL_ENDPOINTS = (
    "http://dgx005:50001/v1",
    "http://dgx017:50001/v1",
    "http://dgx020:50002/v1",
)
LOCAL_ENDPOINT_CONCURRENCY = 512
LOCAL_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-5.4-mini"
OPENAI_CREDENTIAL = "OPENAI_API_KEY_TWO"
OPENAI_FALLBACK_CREDENTIAL = "OPENAI_API_KEY_ONE"
DEFAULT_WORKERS = 512
OPENAI_MAX_TOKENS = 128_000
OPENAI_FALLBACK_MAX_TOKENS = 128_000
OPENAI_CONCURRENCY = 256
LOCAL_MAX_TOKENS = 128_000

V3_PROTOCOL_VERSION = "starling_canonical_reconciliation_endpoint_kmeans.v3_5"
V3_REVIEW_BASIS = "resolved_endpoint_tfidf_kmeans_two_pass.v6"
V3_ENDPOINTS = (
    "http://dgx005:50001/v1",
    "http://dgx017:50001/v1",
    "http://dgx020:50002/v1",
)
V3_ENDPOINT_CONCURRENCY = 512
V3_RANDOM_SEED = 20260801
V3_MAX_ATTEMPTS = 3
V3_QUARANTINE_FAILURES = 3
V3_TFIDF_CONFIG = {
    "analyzer": "char_wb",
    "ngram_range": [2, 5],
    "sublinear_tf": True,
    "norm": "l2",
    "dtype": "float32",
    "min_df": 1,
}

V10_ROOT = Path(__file__).resolve().parent
TASKS = {
    "ames": {
        "resolution": evidence_library_root("ames", "v10")
        / "measurement_resolution_v7/measurement_resolution.parquet",
    },
    "dili": {
        "resolution": evidence_library_root("dili", "v10")
        / "measurement_resolution_v5/measurement_resolution.parquet",
    },
    "carcinogens": {
        "resolution": evidence_library_root("carcinogens", "v10")
        / "measurement_resolution_v4/measurement_resolution.parquet",
    },
}
for _task, _config in TASKS.items():
    _config["cleaned"] = evidence_library_root(_task, "v10") / "01_cleaned/records.parquet"
    _config["output"] = V10_ROOT / (
        f"tasks/{_task}/data_processing/canonicalization_v10/"
        "unit_reconciliation_v2/mapping.json"
    )
    _config["v3_output"] = V10_ROOT / (
        f"tasks/{_task}/data_processing/canonicalization_v10/"
        "unit_reconciliation_v3/mapping.json"
    )
TASKS["ames"]["v3_output"] = V10_ROOT / (
    "tasks/ames/data_processing/canonicalization_v10/"
    "unit_reconciliation_v4/mapping.json"
)

_QUALIFIER_PATTERNS = {
    "percent": re.compile(r"%|\bpercent(?:age)?\b", re.I),
    "control": re.compile(r"\bcontrol\b|\bvehicle\b", re.I),
    "inhibition": re.compile(r"\binhib(?:ition|itory|ited)?\b", re.I),
    "fold": re.compile(r"\bfold\b|\btimes?\b", re.I),
    "ratio": re.compile(r"\bratio\b|\bfraction\b", re.I),
    "count": re.compile(r"\bcount\b|\bnumber\b|\bcells?\b", re.I),
    "per": re.compile(r"/|\bper\b|\^?\s*-\s*1\b", re.I),
}

_BASIS_WORDS = frozenset(
    {
        "activity",
        "binding",
        "count",
        "depletion",
        "frequency",
        "incidence",
        "inhibition",
        "mortality",
        "ratio",
        "residues",
        "score",
        "survival",
        "viability",
        "yield",
    }
)

_PROTECTED_PERCENT_BASIS = re.compile(
    r"\b(?:baseline|control|initial|reference|relative|total|treated|untreated|vehicle|"
    r"population|sample|cells?|animals?|subjects?|patients?|colon(?:y|ies)|metaphases?|nuclei)\b",
    re.I,
)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _wildcard_base_units(entries: Sequence[Mapping[str, Any]], task: str) -> set[str]:
    """Return globally covered units while preserving endpoint-specific overlays."""
    if not entries or any(entry.get("task") != task for entry in entries):
        raise ValueError("incremental base unit mapping has invalid coverage")
    units: list[str] = []
    for entry in entries:
        unit = _text(entry.get("input_unit"))
        endpoints = entry.get("canonical_endpoints")
        if not unit or not isinstance(endpoints, list) or not endpoints:
            raise ValueError("incremental base unit mapping has invalid coverage")
        if endpoints == ["*"]:
            units.append(unit)
    if len(units) != len(set(units)):
        raise ValueError("incremental base unit mapping has duplicate wildcard coverage")
    return set(units)


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _stable_id(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _atomic_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(_json_bytes(value, pretty=pretty))
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(_json_bytes(row))
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def partition_rows(rows: Sequence[Any], size: int = GROUP_SIZE) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("partition size must be positive")
    return [list(rows[start : start + size]) for start in range(0, len(rows), size)]


def context_graph_partition_rows(
    rows: Sequence[Mapping[str, Any]], size: int = GROUP_SIZE
) -> list[list[Mapping[str, Any]]]:
    """Pack related units through their rarest declared context bucket.

    The unit/context relation is a bipartite graph. Assigning each unit to its
    rarest neighboring context avoids giant generic buckets while keeping the
    batching deterministic and linear in the number of observed contexts.
    """
    if size <= 0:
        raise ValueError("partition size must be positive")
    populations = Counter(
        token for row in rows for token in set(row.get("context_tokens") or ())
    )
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        tokens = sorted(set(row.get("context_tokens") or ()))
        anchor = _text(row.get("context_anchor"))
        if not anchor:
            anchor = (
                min(tokens, key=lambda token: (populations[token], token))
                if tokens
                else ""
            )
        buckets[anchor].append(row)
    ordered_buckets = sorted(
        buckets.items(), key=lambda item: (item[0] == "", item[0])
    )
    output: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for _, bucket in ordered_buckets:
        bucket = sorted(bucket, key=lambda row: (_literal_key(str(row["unit"])), row["id"]))
        for chunk in partition_rows(bucket, size):
            if current and len(current) + len(chunk) > size:
                output.append(current)
                current = []
            if len(chunk) == size:
                if current:
                    output.append(current)
                    current = []
                output.append(chunk)
            else:
                current.extend(chunk)
    if current:
        output.append(current)
    if [row["id"] for group in output for row in group] != [
        row["id"]
        for _, bucket in ordered_buckets
        for row in sorted(
            bucket, key=lambda value: (_literal_key(str(value["unit"])), value["id"])
        )
    ]:
        raise AssertionError("context batching changed unit coverage")
    return output


def _task_context_fields(task: str) -> Mapping[str, Sequence[str]]:
    module = importlib.import_module(
        f"data.processing.evidence_library.versions.v10.tasks.{task}."
        "starling_measurement_resolution"
    )
    fields = getattr(module, "UNIT_RECONCILIATION_CONTEXT_FIELDS", None)
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError(f"{task} does not declare unit reconciliation context fields")
    return fields


def _context_tokens(
    row: Mapping[str, Any], fields_by_source: Mapping[str, Sequence[str]]
) -> tuple[str, ...]:
    source = _text(row.get("source_id"))
    if source not in fields_by_source:
        raise ValueError(f"unit row has undeclared source context: {source!r}")
    values = []
    endpoint = _text(row.get("canonical_endpoint_name"))
    if endpoint:
        values.append(f"{source}:canonical_endpoint_name={endpoint}")
    for field in fields_by_source[source]:
        value = _text(row.get(field))
        if value:
            values.append(f"{source}:{field}={value}")
    return tuple(sorted(set(values)))


def _qualifiers(value: str) -> frozenset[str]:
    return frozenset(
        name for name, pattern in _QUALIFIER_PATTERNS.items() if pattern.search(value)
    )


def _notation_matches(left: Any, right: Any) -> bool:
    if left.notation_status != right.notation_status:
        return False
    if left.notation_factor is None or right.notation_factor is None:
        return left.notation_factor is right.notation_factor
    return math.isclose(left.notation_factor, right.notation_factor, rel_tol=1e-12)


def _literal_key(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("μ", "µ")
    return re.sub(r"\s+", " ", value).strip()


def safe_canonical_unit(source: str, proposed: str, task: str) -> tuple[str, str]:
    """Accept only provably scale- and qualifier-preserving aliases."""
    source, proposed = _text(source), _text(proposed)
    if not proposed or proposed == source:
        return source, "identity"
    if not _qualifiers(source).issubset(_qualifiers(proposed)):
        return source, "identity_qualifier_guard"
    left = canonicalize_unit(source.replace("_", " "), task=task)
    right = canonicalize_unit(proposed.replace("_", " "), task=task)
    if left.unknown_tokens or right.unknown_tokens:
        if _literal_key(source) == _literal_key(proposed):
            return proposed, "accepted_literal_alias"
        return source, "identity_unknown_unit_guard"
    if (
        left.dimension != right.dimension
        or left.transform != right.transform
        or not math.isclose(left.scale, right.scale, rel_tol=1e-12)
        or not _notation_matches(left, right)
    ):
        return source, "identity_scale_guard"
    return proposed, "accepted_equivalent_alias"


def _measurements(value: Any) -> list[dict[str, Any]]:
    parsed = json.loads(_text(value) or "[]")
    if not isinstance(parsed, list) or any(not isinstance(row, dict) for row in parsed):
        raise ValueError("measurements_json must be a list of objects")
    return parsed


def prepare(
    task: str,
    run_root: Path,
    *,
    cleaned_path: Path | None = None,
    resolution_path: Path | None = None,
    base_mapping: Path | None = None,
) -> dict[str, Any]:
    config = TASKS[task]
    cleaned = cleaned_path or Path(config["cleaned"])
    resolution = resolution_path or Path(config["resolution"])
    for path in (cleaned, resolution):
        if not path.is_file():
            raise FileNotFoundError(path)
    fields_by_source = _task_context_fields(task)
    counts: dict[str, Counter[str]] = {}
    contexts: dict[str, set[str]] = defaultdict(set)
    context_by_record: dict[str, tuple[str, ...]] = {}

    def add(unit: Any, origin: str, tokens: Sequence[str]) -> None:
        value = _text(unit)
        if value:
            counts.setdefault(value, Counter())[origin] += 1
            contexts[value].update(tokens)

    context_columns = sorted(
        {field for fields in fields_by_source.values() for field in fields}
    )
    columns = (
        "cleaned_record_id",
        "source_id",
        "canonical_endpoint_name",
        "unit_text",
        "measurement_resolution_exact_unit",
        *context_columns,
    )
    for batch in pq.ParquetFile(cleaned).iter_batches(columns=list(columns)):
        for row in batch.to_pylist():
            record_id = _text(row["cleaned_record_id"])
            if not record_id or record_id in context_by_record:
                raise ValueError(f"duplicate or empty cleaned record ID: {record_id!r}")
            tokens = _context_tokens(row, fields_by_source)
            context_by_record[record_id] = tokens
            add(row["unit_text"], "source", tokens)
            add(row["measurement_resolution_exact_unit"], "exact", tokens)
    for batch in pq.ParquetFile(resolution).iter_batches(
        columns=["cleaned_record_id", "status", "measurements_json"]
    ):
        for row in batch.to_pylist():
            if row["status"] != "ok":
                continue
            record_id = _text(row["cleaned_record_id"])
            if record_id not in context_by_record:
                raise ValueError(f"measurement resolution is absent from cleaned data: {record_id}")
            values = _measurements(row["measurements_json"])
            if len(values) != 1:
                raise ValueError("ok resolution row must have exactly one measurement")
            add(values[0].get("unit"), "llm", context_by_record[record_id])
    context_populations = Counter(
        token for unit_contexts in contexts.values() for token in unit_contexts
    )
    rows = [
        {
            "id": _stable_id("u_", unit),
            "unit": unit,
            "counts": dict(sorted(origins.items())),
            "total_count": sum(origins.values()),
            "context_anchor": (
                min(
                    contexts[unit],
                    key=lambda token: (context_populations[token], token),
                )
                if contexts[unit]
                else ""
            ),
            "context_token_count": len(contexts[unit]),
        }
        for unit, origins in sorted(
            counts.items(),
            key=lambda pair: (
                unicodedata.normalize("NFKC", pair[0]).casefold(),
                pair[0],
            ),
        )
    ]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("unit ID collision")
    base_spec = None
    if base_mapping is not None:
        load_exact_unit_mapping(base_mapping)
        base_payload = json.loads(base_mapping.read_text(encoding="utf-8"))
        base_entries = base_payload.get("entries") or []
        base_units = _wildcard_base_units(base_entries, task)
        full_unit_count = len(rows)
        rows = [row for row in rows if row["unit"] not in base_units]
        base_spec = {
            "path": str(base_mapping.resolve()),
            "sha256": file_sha256(base_mapping),
            "entry_count": len(base_entries),
            "wildcard_unit_count": len(base_units),
            "full_unit_count": full_unit_count,
        }
    task_root = run_root / task
    universe = task_root / "universe.jsonl"
    manifest_path = task_root / "universe.manifest.json"
    if universe.exists() or manifest_path.exists():
        raise FileExistsError(f"prepared artifact already exists: {task_root}")
    _write_jsonl(universe, rows)
    manifest = {
        "version": PROTOCOL_VERSION,
        "task": task,
        "group_size": GROUP_SIZE,
        "unit_count": len(rows),
        "base_mapping": base_spec,
        "partition_count": len(context_graph_partition_rows(rows)),
        "batching_strategy": "declared_context_graph_rarest_neighbor.v1",
        "context_fields_by_source": {
            source: list(fields) for source, fields in sorted(fields_by_source.items())
        },
        "cleaned_records": {"path": str(cleaned.resolve()), "sha256": file_sha256(cleaned)},
        "measurement_resolution": {
            "path": str(resolution.resolve()),
            "sha256": file_sha256(resolution),
        },
        "universe_sha256": file_sha256(universe),
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _parse_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("response must be a JSON object")
    return parsed


def _validate_response(
    text: str, rows: Sequence[Mapping[str, Any]], task: str
) -> list[dict[str, Any]]:
    parsed = _parse_object(text)
    mappings = parsed.get("mappings")
    expected = {str(row["id"]): str(row["unit"]) for row in rows}
    if not isinstance(mappings, list) or len(mappings) != len(expected):
        raise ValueError(f"expected {len(expected)} mappings")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for mapping in mappings:
        if not isinstance(mapping, dict):
            raise ValueError("each mapping must be an object")
        identifier = _text(mapping.get("id"))
        if identifier not in expected or identifier in seen:
            raise ValueError(f"unexpected or duplicate mapping ID: {identifier!r}")
        proposed = _text(mapping.get("canonical_unit"))
        canonical, decision = safe_canonical_unit(expected[identifier], proposed, task)
        output.append({"id": identifier, "canonical_unit": canonical, "guard_decision": decision})
        seen.add(identifier)
    if seen != set(expected):
        raise ValueError("response did not cover every input ID")
    return output


def _prompt(task: str, rows: Sequence[Mapping[str, Any]], feedback: str = "") -> str:
    payload = [
        {
            **{"id": row["id"], "unit": row["unit"]},
            **(
                {"member_count": row["member_count"]}
                if "member_count" in row
                else {}
            ),
        }
        for row in rows
    ]
    suffix = f"\nThe previous response failed validation: {feedback}" if feedback else ""
    return f"""Reconcile this partition of measurement units for the {task} evidence library.
Return one JSON object with exactly this schema:
{{"mappings":[{{"id":"input id","canonical_unit":"nonempty unit"}}]}}

Map all {len(rows)} IDs exactly once; do not omit, exclude, or add IDs. Every mapping must
have a nonempty unit. Cluster only true spelling/typography aliases. A unit is mandatory.
Preserve physical scale exactly: nM, µM, mM, and 10^-6 mol/L are distinct and must never
be collapsed into each other. Preserve percent, control, inhibition, ratio, fold, count,
and denominator qualifiers. When uncertain or unsafe, return the original unit unchanged.
Do not return prose or markdown.{suffix}

INPUTS={json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}"""


@dataclass(frozen=True)
class Attempt:
    provider: str
    model: str
    max_tokens: int


ATTEMPTS = (
    Attempt("local", LOCAL_MODEL, LOCAL_MAX_TOKENS),
    Attempt("local", LOCAL_MODEL, LOCAL_MAX_TOKENS),
    Attempt("local", LOCAL_MODEL, LOCAL_MAX_TOKENS),
)


def _identity(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "canonical_unit": row["unit"],
            "guard_decision": "identity_after_retry_exhaustion",
        }
        for row in rows
    ]


async def resolve_partition(
    task: str,
    rows: Sequence[Mapping[str, Any]],
    request: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Resolve one partition; ``request`` is injected to make retry behavior testable."""
    attempts: list[dict[str, Any]] = []
    feedback = ""
    for index, spec in enumerate(ATTEMPTS, start=1):
        try:
            result = await request(spec, _prompt(task, rows, feedback))
            text, usage = result if isinstance(result, tuple) else (result, None)
            mappings = _validate_response(text, rows, task)
            attempt = {"attempt": index, **spec.__dict__, "status": "ok"}
            if usage:
                attempt["usage"] = usage
            attempts.append(attempt)
            return mappings, attempts, "resolved"
        except Exception as error:  # each failure is durable provenance for the next retry
            feedback = f"{type(error).__name__}: {error}"[:1000]
            attempts.append(
                {"attempt": index, **spec.__dict__, "status": "failed", "error": feedback}
            )
            if spec.provider == "openai" and (
                "RateLimitError" in feedback or "Error code: 429" in feedback
            ) and index < len(ATTEMPTS) and ATTEMPTS[index].provider == "openai":
                jitter = int(hashlib.sha256(rows[0]["id"].encode()).hexdigest()[:2], 16)
                await asyncio.sleep(15 + jitter % 31)
    return _identity(rows), attempts, "identity_after_retry_exhaustion"


def terminal_events(
    path: Path,
    *,
    task: str,
    pass_name: str,
    input_sha256: str,
    partitions: Sequence[Sequence[Mapping[str, Any]]],
    resolved_only: bool = False,
) -> dict[int, dict[str, Any]]:
    """Load terminal events only when they exactly match the frozen partition."""
    if not path.exists():
        return {}
    events: dict[int, dict[str, Any]] = {}
    for event in _read_jsonl(path):
        if (
            event.get("task") != task
            or event.get("pass") != pass_name
            or event.get("input_sha256") != input_sha256
        ):
            raise ValueError(f"cache provenance mismatch: {path}")
        partition = int(event["partition"])
        if partition < 0 or partition >= len(partitions):
            raise ValueError(f"cache partition is out of range: {partition}")
        expected_ids = [str(row["id"]) for row in partitions[partition]]
        if event.get("input_ids") != expected_ids:
            raise ValueError(f"cache partition membership mismatch: {partition}")
        status = event.get("status")
        if status not in {"resolved", "identity_after_retry_exhaustion"}:
            raise ValueError(f"cache event is not terminal: {partition}")
        mappings = event.get("mappings")
        mapping_ids = (
            [str(row.get("id")) for row in mappings]
            if isinstance(mappings, list)
            and all(isinstance(row, dict) for row in mappings)
            else []
        )
        if len(mapping_ids) != len(expected_ids) or set(mapping_ids) != set(expected_ids):
            raise ValueError(f"cache mapping coverage mismatch: {partition}")
        if any(not _text(row.get("canonical_unit")) for row in mappings):
            raise ValueError(f"cache mapping has an empty unit: {partition}")
        if partition in events:
            raise ValueError(f"duplicate terminal partition: {partition}")
        if not resolved_only or status == "resolved":
            events[partition] = event
    return events


async def run_pass(
    task: str,
    run_root: Path,
    workers: int,
    seed_terminal_cache: Path | None = None,
) -> dict[str, Any]:
    if workers != len(LOCAL_ENDPOINTS) * LOCAL_ENDPOINT_CONCURRENCY:
        raise ValueError(
            "workers must equal the configured aggregate local capacity "
            f"({len(LOCAL_ENDPOINTS) * LOCAL_ENDPOINT_CONCURRENCY})"
        )
    task_root = run_root / task
    universe_manifest = json.loads((task_root / "universe.manifest.json").read_text())
    input_rows = _read_jsonl(task_root / "universe.jsonl")
    input_sha = universe_manifest["universe_sha256"]
    partitions = context_graph_partition_rows(input_rows)
    pass_name = "single"
    pass_root = task_root / "llm_pass"
    cache_path = pass_root / "terminal.jsonl"
    pass_root.mkdir(parents=True, exist_ok=True)
    completed = terminal_events(
        cache_path,
        task=task,
        pass_name=pass_name,
        input_sha256=input_sha,
        partitions=partitions,
    )
    seed_receipt = None
    if seed_terminal_cache is not None:
        seed_path = seed_terminal_cache.resolve()
        seed_sha256 = file_sha256(seed_path)
        inherited = terminal_events(
            seed_path,
            task=task,
            pass_name=pass_name,
            input_sha256=input_sha,
            partitions=partitions,
            resolved_only=True,
        )
        missing = sorted(set(inherited) - set(completed))
        if missing:
            with cache_path.open("ab") as handle:
                for partition in missing:
                    event = {
                        **inherited[partition],
                        "version": PROTOCOL_VERSION,
                        "inherited_from": {
                            "path": str(seed_path),
                            "sha256": seed_sha256,
                        },
                    }
                    handle.write(_json_bytes(event))
                    completed[partition] = event
                handle.flush()
                os.fsync(handle.fileno())
        inherited_present = sum(
            event.get("inherited_from", {}).get("sha256") == seed_sha256
            for event in completed.values()
        )
        seed_receipt = {
            "path": str(seed_path),
            "sha256": seed_sha256,
            "resolved_partitions_available": len(inherited),
            "resolved_partitions_inherited": inherited_present,
            "resolved_partitions_added_this_resume": len(missing),
        }
    local_clients = [
        async_openai_compatible_client(
            base_url=base_url,
            provider="local",
            max_connections=LOCAL_ENDPOINT_CONCURRENCY,
            timeout_s=3600,
            max_retries=0,
        )[0]
        for base_url in LOCAL_ENDPOINTS
    ]
    uses_openai = any(attempt.provider == "openai" for attempt in ATTEMPTS)
    openai_client = None
    credential = ""
    if uses_openai:
        openai_client, credential = async_openai_compatible_client(
            base_url=OPENAI_BASE_URL,
            provider="openai",
            env_file=DEFAULT_ENV_FILE,
            credential_env=OPENAI_CREDENTIAL,
            max_connections=OPENAI_CONCURRENCY,
            timeout_s=3600,
            max_retries=0,
        )
    semaphore = asyncio.Semaphore(workers)
    openai_semaphore = asyncio.Semaphore(OPENAI_CONCURRENCY)
    endpoint_semaphores = [
        asyncio.Semaphore(LOCAL_ENDPOINT_CONCURRENCY) for _ in LOCAL_ENDPOINTS
    ]
    endpoint_cycle = itertools.cycle(range(len(LOCAL_ENDPOINTS)))
    write_lock = asyncio.Lock()
    openai_enabled = True
    attempt_counts: Counter[str] = Counter()
    usage_totals: Counter[str] = Counter()
    for event in completed.values():
        for attempt in event.get("attempts", []):
            attempt_counts[f"{attempt.get('provider')}:{attempt.get('status')}"] += 1
            for name, value in (attempt.get("usage") or {}).items():
                if isinstance(value, int):
                    usage_totals[f"{attempt.get('provider')}:{name}"] += value

    async def request(spec: Attempt, prompt: str) -> tuple[str, dict[str, Any]]:
        nonlocal openai_enabled
        kwargs: dict[str, Any] = {
            "model": spec.model,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": "high",
            "response_format": {"type": "json_object"},
        }
        if spec.provider == "local":
            kwargs.update(max_tokens=spec.max_tokens, temperature=0)
            endpoint_index = next(endpoint_cycle)
            async with endpoint_semaphores[endpoint_index]:
                response = await local_clients[endpoint_index].chat.completions.create(
                    **kwargs
                )
        else:
            if openai_client is None:
                raise RuntimeError("OpenAI is absent from this run's retry sequence")
            if not openai_enabled:
                raise RuntimeError("OpenAI circuit is open after insufficient_quota")
            kwargs["max_completion_tokens"] = spec.max_tokens
            try:
                async with openai_semaphore:
                    response = await openai_client.chat.completions.create(**kwargs)
            except Exception as error:
                if "insufficient_quota" in str(error):
                    openai_enabled = False
                raise
        usage = response.usage.model_dump(exclude_none=True) if response.usage else {}
        return _text(response.choices[0].message.content), usage

    async def one(index: int, rows: list[dict[str, Any]]) -> None:
        async with semaphore:
            mappings, attempts, status = await resolve_partition(task, rows, request)
        event = {
            "version": PROTOCOL_VERSION,
            "task": task,
            "pass": pass_name,
            "input_sha256": input_sha,
            "partition": index,
            "input_ids": [row["id"] for row in rows],
            "status": status,
            "attempts": attempts,
            "mappings": mappings,
        }
        async with write_lock:
            with cache_path.open("ab") as handle:
                handle.write(_json_bytes(event))
                handle.flush()
                os.fsync(handle.fileno())
            completed[index] = event
            for attempt in attempts:
                attempt_counts[f"{attempt['provider']}:{attempt['status']}"] += 1
                for name, value in (attempt.get("usage") or {}).items():
                    if isinstance(value, int):
                        usage_totals[f"{attempt['provider']}:{name}"] += value
            _atomic_json(
                pass_root / "status.json",
                {
                    "task": task,
                    "pass": pass_name,
                    "finished": len(completed),
                    "total": len(partitions),
                    "identity_after_retry_exhaustion": sum(
                        event["status"] == "identity_after_retry_exhaustion"
                        for event in completed.values()
                    ),
                    "attempt_counts": dict(sorted(attempt_counts.items())),
                    "usage": dict(sorted(usage_totals.items())),
                    "openai_circuit_open": not openai_enabled,
                },
            )

    await asyncio.gather(
        *(one(index, rows) for index, rows in enumerate(partitions) if index not in completed)
    )
    await asyncio.gather(*(client.close() for client in local_clients))
    if openai_client is not None:
        await openai_client.close()
    if set(completed) != set(range(len(partitions))):
        raise ValueError("pass did not reach terminal status for every partition")
    by_id = {
        mapping["id"]: {**mapping, "partition": index, "terminal_status": event["status"]}
        for index, event in completed.items()
        for mapping in event["mappings"]
    }
    expected_ids = [row["id"] for row in input_rows]
    if set(by_id) != set(expected_ids):
        raise ValueError("terminal mappings do not exactly cover pass inputs")
    output_rows = [by_id[identifier] for identifier in expected_ids]
    mapping_path = pass_root / "mapping.jsonl"
    _write_jsonl(mapping_path, output_rows)
    manifest = {
        "version": PROTOCOL_VERSION,
        "task": task,
        "pass": pass_name,
        "group_size": GROUP_SIZE,
        "batching_strategy": universe_manifest.get("batching_strategy"),
        "input_sha256": input_sha,
        "input_count": len(input_rows),
        "partition_count": len(partitions),
        "mapping_sha256": file_sha256(mapping_path),
        "mapping_count": len(output_rows),
        "retry_sequence": [spec.__dict__ for spec in ATTEMPTS],
        "reasoning_effort": "high",
        "credential_env": credential,
        "openai_max_inflight": OPENAI_CONCURRENCY if uses_openai else 0,
        "local_endpoints": [
            {"base_url": base_url, "max_inflight": LOCAL_ENDPOINT_CONCURRENCY}
            for base_url in LOCAL_ENDPOINTS
        ],
        "identity_after_retry_exhaustion_partitions": sum(
            event["status"] == "identity_after_retry_exhaustion" for event in completed.values()
        ),
        "attempt_counts": dict(sorted(attempt_counts.items())),
        "usage": dict(sorted(usage_totals.items())),
        "seed_terminal_cache": seed_receipt,
    }
    _atomic_json(pass_root / "manifest.json", manifest)
    return manifest


def _single_pass_paths(task_root: Path) -> tuple[Path, Path]:
    current = task_root / "llm_pass"
    if (current / "manifest.json").is_file():
        return current / "mapping.jsonl", current / "manifest.json"
    legacy = task_root / "pass_a"
    if (legacy / "manifest.json").is_file():
        return legacy / "mapping.jsonl", legacy / "manifest.json"
    raise FileNotFoundError(f"no completed single pass under {task_root}")


def _single_pass_entries(task: str, run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_root = run_root / task
    universe_path = task_root / "universe.jsonl"
    universe_manifest_path = task_root / "universe.manifest.json"
    universe = _read_jsonl(universe_path)
    universe_manifest = json.loads(universe_manifest_path.read_text())
    mapping_path, pass_manifest_path = _single_pass_paths(task_root)
    pass_rows = _read_jsonl(mapping_path)
    pass_manifest = json.loads(pass_manifest_path.read_text())
    if universe_manifest.get("task") != task or pass_manifest.get("task") != task:
        raise ValueError("single-pass task provenance mismatch")
    allowed_versions = {PROTOCOL_VERSION, *LEGACY_PASS_A_VERSIONS}
    if universe_manifest.get("version") not in allowed_versions:
        raise ValueError("unsupported unit-universe protocol")
    if pass_manifest.get("version") not in allowed_versions:
        raise ValueError("unsupported single-pass protocol")
    if file_sha256(universe_path) != universe_manifest.get("universe_sha256"):
        raise ValueError("unit-universe hash mismatch")
    if file_sha256(mapping_path) != pass_manifest.get("mapping_sha256"):
        raise ValueError("single-pass mapping hash mismatch")
    if len(universe) != len(pass_rows) or len(pass_rows) != pass_manifest.get("mapping_count"):
        raise ValueError("single pass does not cover the unit universe")
    by_id = {str(row.get("id")): row for row in pass_rows}
    if len(by_id) != len(pass_rows) or set(by_id) != {str(row["id"]) for row in universe}:
        raise ValueError("single-pass mapping IDs differ from the unit universe")
    entries = []
    for row in universe:
        canonical = _text(by_id[str(row["id"])].get("canonical_unit"))
        if not canonical:
            raise ValueError(f"single-pass unit is empty: {row['id']}")
        entries.append(
            {
                "task": task,
                "canonical_endpoints": ["*"],
                "input_unit": row["unit"],
                "action": "map",
                "canonical_unit": canonical,
                "scale": "1",
                "domain": "any",
                "review_basis": REVIEW_BASIS,
            }
        )
    base_spec = universe_manifest.get("base_mapping")
    if base_spec:
        base_path = Path(_text(base_spec.get("path")))
        if not base_path.is_file() or file_sha256(base_path) != base_spec.get("sha256"):
            raise ValueError("incremental base unit mapping hash mismatch")
        load_exact_unit_mapping(base_path)
        base_entries = json.loads(base_path.read_text(encoding="utf-8")).get(
            "entries"
        ) or []
        if len(base_entries) != base_spec.get("entry_count"):
            raise ValueError("incremental base unit mapping row count changed")
        new_units = {entry["input_unit"] for entry in entries}
        if _wildcard_base_units(base_entries, task) & new_units:
            raise ValueError("incremental unit mapping overlaps its reviewed base")
        entries = [*base_entries, *entries]
    lineage = {
        "universe": {"path": str(universe_path.resolve()), "sha256": file_sha256(universe_path)},
        "universe_manifest": {
            "path": str(universe_manifest_path.resolve()),
            "sha256": file_sha256(universe_manifest_path),
        },
        "single_pass_mapping": {
            "path": str(mapping_path.resolve()),
            "sha256": file_sha256(mapping_path),
        },
        "single_pass_manifest": {
            "path": str(pass_manifest_path.resolve()),
            "sha256": file_sha256(pass_manifest_path),
        },
        "legacy_lexical_batching": pass_manifest.get("pass") == "a",
        "base_mapping": base_spec,
    }
    return entries, lineage


def _spelling_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().replace("μ", "µ")
    replacements = {
        "micromolar": "um",
        "nanomolar": "nm",
        "millimolar": "mm",
        "microgram": "ug",
        "nanogram": "ng",
        "milligram": "mg",
        "millilitre": "ml",
        "milliliter": "ml",
        "litre": "l",
        "liter": "l",
        "percent": "%",
        " per ": "/",
        "µ": "u",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"[^a-z0-9%]+", "", text)


def _reviewed_spelling_alias_safe(source: str, target: str) -> bool:
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", source)
    if numbers != re.findall(r"[-+]?\d+(?:\.\d+)?", target):
        return False
    if not _qualifiers(source).issubset(_qualifiers(target)):
        return False
    return bool(_spelling_key(source)) and _spelling_key(source) == _spelling_key(target)


def _load_review_decisions(
    path: Path | None, task: str, canonical_units: set[str]
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    if path is None:
        return {}, []
    rows = _read_jsonl(path)
    decisions: dict[str, str] = {}
    for row in rows:
        source = _text(row.get("from_canonical_unit"))
        target = _text(row.get("canonical_unit"))
        kind = _text(row.get("decision_type"))
        rationale = _text(row.get("rationale"))
        if source not in canonical_units or target not in canonical_units:
            raise ValueError("agent review target is outside the task canonical vocabulary")
        if not source or not target or source == target or source in decisions or not rationale:
            raise ValueError(f"invalid or duplicate agent review decision: {source!r}")
        guarded, guard = safe_canonical_unit(source, target, task)
        accepted = guarded == target and guard.startswith("accepted_")
        if kind == "reviewed_spelling_alias":
            accepted = accepted or _reviewed_spelling_alias_safe(source, target)
        elif kind != "equivalent_alias":
            raise ValueError(f"unsupported agent review decision type: {kind!r}")
        if not accepted:
            raise ValueError(f"unsafe agent review alias: {source!r} -> {target!r}")
        decisions[source] = target
    for source in decisions:
        seen = {source}
        target = decisions[source]
        while target in decisions:
            if target in seen:
                raise ValueError(f"agent review alias cycle: {sorted(seen)}")
            seen.add(target)
            target = decisions[target]
    return decisions, rows


def _review_sort_key(unit: str, task: str) -> tuple[Any, ...]:
    parsed = canonicalize_unit(unit, task=task)
    return (
        tuple(parsed.dimension),
        parsed.transform or "",
        parsed.scale,
        parsed.notation_status,
        parsed.notation_factor or 0,
        tuple(sorted(_qualifiers(unit))),
        _spelling_key(unit),
        unit.casefold(),
        unit,
    )


def prepare_review(
    task: str,
    mapping: Path,
    output_dir: Path,
    packet_size: int = 200,
    base_mapping: Path | None = None,
) -> dict[str, Any]:
    if not 1 <= packet_size <= 500:
        raise ValueError("review packet size must be in [1, 500]")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    payload = json.loads(mapping.read_text())
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("unit mapping has no entries")
    base_units: set[str] = set()
    if base_mapping is not None:
        load_exact_unit_mapping(base_mapping)
        base_payload = json.loads(base_mapping.read_text(encoding="utf-8"))
        base_units = _wildcard_base_units(base_payload.get("entries") or [], task)
    review_entries = [
        entry for entry in entries if _text(entry.get("input_unit")) not in base_units
    ]
    members: dict[str, list[str]] = defaultdict(list)
    for entry in review_entries:
        members[_text(entry.get("canonical_unit"))].append(_text(entry.get("input_unit")))
    units = sorted(members, key=lambda unit: _review_sort_key(unit, task))
    packets = []
    for index, chunk in enumerate(partition_rows(units, packet_size), start=1):
        packets.append(
            {
                "packet_id": f"{task}_unit_review_{index:04d}",
                "items": [
                    {
                        "canonical_unit": unit,
                        "member_count": len(members[unit]),
                        "input_examples": sorted(members[unit])[:5],
                    }
                    for unit in chunk
                ],
            }
        )
    output_dir.mkdir(parents=True)
    packet_path = output_dir / "packets.jsonl"
    _write_jsonl(packet_path, packets)
    manifest = {
        "version": "canonical_unit_agent_review_packets.v1",
        "task": task,
        "mapping": {"path": str(mapping.resolve()), "sha256": file_sha256(mapping)},
        "entry_count": len(entries),
        "review_entry_count": len(review_entries),
        "base_mapping": (
            {"path": str(base_mapping.resolve()), "sha256": file_sha256(base_mapping)}
            if base_mapping is not None
            else None
        ),
        "canonical_unit_count": len(units),
        "packet_count": len(packets),
        "packet_size": packet_size,
        "packets": {"path": str(packet_path.resolve()), "sha256": file_sha256(packet_path)},
        "coverage": (
            "every new pre-review canonical unit exactly once"
            if base_mapping is not None
            else "every pre-review canonical unit exactly once"
        ),
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def _validate_review_completion(
    task: str,
    path: Path,
    decisions_path: Path,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    completion = json.loads(path.read_text())
    if (
        completion.get("version") != "canonical_unit_agent_review_completion.v1"
        or completion.get("task") != task
        or not _text(completion.get("reviewer_id"))
    ):
        raise ValueError("invalid agent review completion receipt")
    if completion.get("decisions_sha256") != file_sha256(decisions_path):
        raise ValueError("agent review decision hash mismatch")
    reference = completion.get("packet_manifest") or {}
    manifest_path = Path(_text(reference.get("path")))
    if not manifest_path.is_file() or file_sha256(manifest_path) != reference.get("sha256"):
        raise ValueError("agent review packet manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    mapping_reference = manifest.get("mapping") or {}
    mapping_path = Path(_text(mapping_reference.get("path")))
    if not mapping_path.is_file() or file_sha256(mapping_path) != mapping_reference.get("sha256"):
        raise ValueError("agent review base mapping hash mismatch")
    reviewed_payload = json.loads(mapping_path.read_text())
    if reviewed_payload.get("entries") != list(entries):
        raise ValueError("agent review covered a different base mapping")
    packet_reference = manifest.get("packets") or {}
    packet_path = Path(_text(packet_reference.get("path")))
    if not packet_path.is_file() or file_sha256(packet_path) != packet_reference.get("sha256"):
        raise ValueError("agent review packet hash mismatch")
    packet_ids = [row["packet_id"] for row in _read_jsonl(packet_path)]
    if completion.get("reviewed_packet_ids") != packet_ids:
        raise ValueError("agent review did not cover every packet in order")
    return {
        **completion,
        "completion_sha256": file_sha256(path),
        "packet_manifest_sha256": file_sha256(manifest_path),
        "packets_sha256": file_sha256(packet_path),
    }


def _consolidate(
    task: str,
    run_root: Path,
    output: Path,
    decisions: Path | None,
    review_completion: Path | None,
    *,
    draft: bool,
) -> dict[str, Any]:
    if (decisions is None) != (review_completion is None):
        raise ValueError("agent decisions and review completion must be supplied together")
    if draft != (review_completion is None):
        raise ValueError("draft consolidation must be unreviewed; publication must be reviewed")
    entries, lineage = _single_pass_entries(task, run_root)
    base_entries = [dict(entry) for entry in entries]
    canonical_units = {entry["canonical_unit"] for entry in entries}
    aliases, decision_rows = _load_review_decisions(decisions, task, canonical_units)
    review = (
        _validate_review_completion(task, review_completion, decisions, base_entries)
        if review_completion is not None
        else None
    )
    immutable_base_count = int((lineage.get("base_mapping") or {}).get("entry_count") or 0)
    for index, entry in enumerate(entries):
        if index < immutable_base_count:
            continue
        source = entry["canonical_unit"]
        seen = {source}
        while source in aliases:
            source = aliases[source]
            if source in seen:
                raise ValueError("agent review produced a cycle")
            seen.add(source)
        entry["canonical_unit"] = source
        if len(seen) > 1:
            entry["review_basis"] = f"{REVIEW_BASIS}; guarded_agent_review.v1"
    destination = output
    bundle = destination.parent
    if bundle.exists():
        raise FileExistsError(f"refusing to replace unit reconciliation bundle: {bundle}")
    staging = bundle.with_name(bundle.name + f".tmp.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    staged_mapping = staging / destination.name
    decisions_path = staging / "agent_review.decisions.jsonl"
    manifest_path = staging / "manifest.json"
    report_path = staging / "report.md"
    payload = {
        "version": EXACT_UNIT_MAPPING_VERSION,
        "task": task,
        "protocol_version": PROTOCOL_VERSION,
        "entries": entries,
    }
    try:
        _atomic_json(staged_mapping, payload, pretty=False)
        if decisions is None:
            _write_jsonl(decisions_path, ())
        else:
            shutil.copyfile(decisions, decisions_path)
        loaded = load_exact_unit_mapping(staged_mapping)
        if len(loaded) != len(entries):
            raise ValueError("published exact unit mapping failed coverage validation")
        manifest = {
            "version": PROTOCOL_VERSION,
            "task": task,
            "publication_status": "draft" if draft else "reviewed",
            "mapping": {
                "path": str(destination.resolve()),
                "sha256": file_sha256(staged_mapping),
            },
            "agent_review_decisions": {
                "path": str((bundle / decisions_path.name).resolve()),
                "sha256": file_sha256(decisions_path),
                "accepted_aliases": len(decision_rows),
            },
            "review_completion": (
                {
                    "sha256": review["completion_sha256"],
                    "reviewer_id": review["reviewer_id"],
                    "reviewed_packets": len(review["reviewed_packet_ids"]),
                    "packet_manifest_sha256": review["packet_manifest_sha256"],
                    "packets_sha256": review["packets_sha256"],
                }
                if review_completion is not None
                else None
            ),
            "entry_count": len(entries),
            "canonical_unit_count": len(
                {entry["canonical_unit"] for entry in entries}
            ),
            "all_actions": ["map"],
            "all_scales": ["1"],
            "lineage": lineage,
        }
        _atomic_json(manifest_path, manifest)
        report = (
            f"# {task} V10 unit reconciliation\n\n"
            f"{'Drafted' if draft else 'Published'} {len(entries):,} lossless unit mappings to "
            f"{manifest['canonical_unit_count']:,} canonical labels. "
            f"The guarded agent review accepted {len(decision_rows):,} additional aliases.\n\n"
            "Every rule maps with scale `1`; uncertain units remain unchanged.\n"
        )
        report_path.write_text(report, encoding="utf-8")
        os.replace(staging, bundle)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def consolidate(
    task: str,
    run_root: Path,
    output: Path | None = None,
    decisions: Path | None = None,
    review_completion: Path | None = None,
) -> dict[str, Any]:
    if decisions is None or review_completion is None:
        raise ValueError("publication requires agent decisions and review completion")
    return _consolidate(
        task,
        run_root,
        output or Path(TASKS[task]["output"]),
        decisions,
        review_completion,
        draft=False,
    )


def consolidate_draft(task: str, run_root: Path, output: Path) -> dict[str, Any]:
    if output.resolve() == Path(TASKS[task]["output"]).resolve():
        raise ValueError("draft output must not target the canonical publication path")
    return _consolidate(task, run_root, output, None, None, draft=True)


def validate(
    task: str,
    run_root: Path,
    mapping: Path | None = None,
    *,
    allow_draft: bool = False,
) -> dict[str, Any]:
    path = mapping or Path(TASKS[task]["output"])
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    status = manifest.get("publication_status") or (
        "reviewed" if manifest.get("review_completion") else "draft"
    )
    if status != "reviewed" and not allow_draft:
        raise ValueError("unit reconciliation has not completed agent review")
    if manifest.get("mapping", {}).get("sha256") != file_sha256(path):
        raise ValueError("unit reconciliation manifest has a stale mapping hash")
    if status == "reviewed":
        decisions = path.parent / "agent_review.decisions.jsonl"
        if (
            not manifest.get("review_completion")
            or manifest.get("agent_review_decisions", {}).get("sha256")
            != file_sha256(decisions)
        ):
            raise ValueError("unit reconciliation lacks valid agent review provenance")
    universe = _read_jsonl(run_root / task / "universe.jsonl")
    universe_manifest = json.loads(
        (run_root / task / "universe.manifest.json").read_text(encoding="utf-8")
    )
    payload = json.loads(path.read_text())
    entries = payload.get("entries", [])
    expected_units = []
    base_entries: list[dict[str, Any]] = []
    base_spec = universe_manifest.get("base_mapping")
    if base_spec:
        base_path = Path(_text(base_spec.get("path")))
        if not base_path.is_file() or file_sha256(base_path) != base_spec.get("sha256"):
            raise ValueError("incremental base unit mapping hash mismatch")
        base_payload = json.loads(base_path.read_text(encoding="utf-8"))
        base_entries = base_payload.get("entries", [])
        expected_units.extend(
            entry.get("input_unit") for entry in base_entries
        )
    expected_units.extend(row["unit"] for row in universe)
    if [entry.get("input_unit") for entry in entries] != expected_units:
        raise ValueError("mapping order/coverage differs from the frozen universe")
    if entries[: len(base_entries)] != base_entries:
        raise ValueError("incremental successor changed its reviewed base mapping")
    if any(
        entry.get("task") != task
        or entry.get("canonical_endpoints") != ["*"]
        or entry.get("action") != "map"
        or entry.get("scale") != "1"
        or not _text(entry.get("canonical_unit"))
        for entry in entries[len(base_entries) :]
    ):
        raise ValueError("mapping contains a task, coverage, scale, or unit violation")
    load_exact_unit_mapping(path)
    return {
        "task": task,
        "valid": True,
        "publication_status": status,
        "entry_count": len(entries),
        "sha256": file_sha256(path),
    }


def auxiliary(config_path: Path, *, plan_only: bool) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    if config.get("reasoning_effort") != "high":
        raise ValueError("auxiliary reconciliation must use reasoning_effort=high")
    specs = [
        AuxiliaryExtractionSpec(
            source_id=row["source_id"],
            input_path=Path(row["input_path"]),
            input_column=row.get("input_column"),
            input_columns=tuple(row.get("input_columns", ())),
            input_source_id=row.get("input_source_id"),
            output_field=row["output_field"],
            prompt=row["prompt"],
            null_sentinel=row.get("null_sentinel"),
        )
        for row in config["specs"]
    ]
    if plan_only:
        return reconciliation_plan(specs, cluster_target_size=int(config.get("cluster_target_size", 100)))
    provider = config["provider"]
    credential = config.get("credential_env")
    client, _ = openai_compatible_client(
        base_url=config["base_url"],
        provider=provider,
        env_file=DEFAULT_ENV_FILE,
        credential_env=credential,
        max_connections=int(config.get("workers", 16)),
        max_retries=0,
    )
    return build_clustered_auxiliary_mapping(
        specs=specs,
        output_path=config["output_path"],
        mapping_version=config["mapping_version"],
        prompt_version=config["prompt_version"],
        api_key_loader=None,
        client=client,
        model=config["model"],
        reasoning_effort="high",
        base_url=config.get("base_url"),
        max_tokens=int(config["max_tokens"]) if config.get("max_tokens") else None,
        cluster_target_size=int(config.get("cluster_target_size", 100)),
        workers=int(config.get("workers", 16)),
        max_retries=int(config.get("max_retries", 5)),
    )


def _v3_text_key(value: str) -> str:
    """Normalize spelling for TF-IDF without changing the published label."""
    text = unicodedata.normalize("NFKC", value).casefold().replace("μ", "µ")
    text = text.translate(str.maketrans({"−": "-", "–": "-", "—": "-"}))
    replacements = (
        (r"\bmicromoles?\b", "µmol"),
        (r"\bnanomoles?\b", "nmol"),
        (r"\bmillimoles?\b", "mmol"),
        (r"\bmicromolar\b", "µm"),
        (r"\bnanomolar\b", "nm"),
        (r"\bmillimolar\b", "mm"),
        (r"\bseconds?\b|\bsecs?\b", "s"),
        (r"\bhours?\b|\bhrs?\b", "h"),
        (r"\bminutes?\b|\bmins?\b", "min"),
        (r"\blitres?\b|\bliters?\b", "l"),
        (r"\bpercent(?:age)?\b", "%"),
        (r"\s+per\s+", "/"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def _v3_identifier(prefix: str, task: str, endpoint: str, unit: str) -> str:
    return _stable_id(prefix, f"{task}\0{endpoint}\0{unit}")


def _v3_eligible_rows(task: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return only units that can be consumed by resolved Stage-1 records."""
    config = TASKS[task]
    cleaned = Path(config["cleaned"])
    resolution = Path(config["resolution"])
    endpoints: dict[str, str] = {}
    counts: dict[tuple[str, str], Counter[str]] = {}

    def add(endpoint: str, unit: Any, origin: str) -> None:
        value = _text(unit)
        if not endpoint or not value:
            return
        counts.setdefault((endpoint, value), Counter())[origin] += 1

    columns = [
        "cleaned_record_id",
        "canonical_endpoint_name",
        "measurement_resolution_route",
        "measurement_resolution_exact_unit",
    ]
    for batch in pq.ParquetFile(cleaned).iter_batches(columns=columns):
        for row in batch.to_pylist():
            record_id = _text(row["cleaned_record_id"])
            endpoint = _text(row["canonical_endpoint_name"])
            if not record_id or record_id in endpoints or not endpoint:
                raise ValueError(f"invalid cleaned endpoint identity: {record_id!r}")
            endpoints[record_id] = endpoint
            if row["measurement_resolution_route"] == "accept":
                add(endpoint, row["measurement_resolution_exact_unit"], "source_exact")

    status_counts: Counter[str] = Counter()
    for batch in pq.ParquetFile(resolution).iter_batches(
        columns=["cleaned_record_id", "status", "measurements_json"]
    ):
        for row in batch.to_pylist():
            status = _text(row["status"])
            status_counts[status] += 1
            if status != "ok":
                continue
            record_id = _text(row["cleaned_record_id"])
            if record_id not in endpoints:
                raise ValueError(f"resolved record is absent from cleaned data: {record_id}")
            measurements = _measurements(row["measurements_json"])
            if len(measurements) != 1:
                raise ValueError("ok resolution row must contain exactly one measurement")
            add(endpoints[record_id], measurements[0].get("unit"), "llm_ok")

    rows = [
        {
            "id": _v3_identifier("v3a_", task, endpoint, unit),
            "task": task,
            "endpoint": endpoint,
            "unit": unit,
            "counts": dict(sorted(origins.items())),
            "member_count": sum(origins.values()),
        }
        for (endpoint, unit), origins in sorted(
            counts.items(), key=lambda item: (item[0][0], _v3_text_key(item[0][1]), item[0][1])
        )
    ]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{task} V3 unit ID collision")
    return rows, {
        "task": task,
        "eligible_endpoint_unit_pairs": len(rows),
        "canonical_endpoint_count": len({row["endpoint"] for row in rows}),
        "resolution_status_counts": dict(sorted(status_counts.items())),
        "cleaned_records": {"path": str(cleaned.resolve()), "sha256": file_sha256(cleaned)},
        "measurement_resolution": {
            "path": str(resolution.resolve()),
            "sha256": file_sha256(resolution),
        },
    }


def _v3_kmeans_groups(matrix: Any, indices: Any, *, seed: int) -> list[Any]:
    """Recursively create real K-means groups with a hard request-size bound."""
    import numpy as np
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    if len(indices) <= GROUP_SIZE:
        return [indices]
    cluster_count = math.ceil(len(indices) / GROUP_SIZE)
    with threadpool_limits(limits=16):
        labels = KMeans(
            n_clusters=cluster_count,
            random_state=seed,
            n_init=10,
            algorithm="lloyd",
        ).fit_predict(matrix[indices])
    groups = [indices[np.flatnonzero(labels == label)] for label in range(cluster_count)]
    groups = [group for group in groups if len(group)]
    if len(groups) <= 1:
        return [indices[start : start + GROUP_SIZE] for start in range(0, len(indices), GROUP_SIZE)]
    output = []
    for offset, group in enumerate(groups):
        output.extend(_v3_kmeans_groups(matrix, group, seed=seed + offset + 1))
    return output


def _v3_clusters(
    rows: Sequence[Mapping[str, Any]], pass_name: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    clusters: list[dict[str, Any]] = []
    task_manifests: dict[str, Any] = {}
    for task in sorted({str(row["task"]) for row in rows}):
        task_rows = [dict(row) for row in rows if row["task"] == task]
        units = sorted({str(row["unit"]) for row in task_rows}, key=lambda value: (_v3_text_key(value), value))
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
            min_df=1,
            lowercase=False,
        )
        matrix = vectorizer.fit_transform([_v3_text_key(unit) for unit in units])
        unit_index = {unit: index for index, unit in enumerate(units)}
        endpoint_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in task_rows:
            endpoint_rows[str(row["endpoint"])].append(row)
        task_cluster_count = 0
        for endpoint, members in sorted(endpoint_rows.items()):
            members.sort(key=lambda row: (_v3_text_key(str(row["unit"])), str(row["unit"]), str(row["id"])))
            indices = np.asarray([unit_index[str(row["unit"])] for row in members], dtype=np.int64)
            seed = V3_RANDOM_SEED + int(hashlib.sha256(endpoint.encode()).hexdigest()[:6], 16)
            for group in _v3_kmeans_groups(matrix, indices, seed=seed):
                selected_units = {units[int(index)] for index in group}
                selected = [row for row in members if str(row["unit"]) in selected_units]
                selected.sort(key=lambda row: (_v3_text_key(str(row["unit"])), str(row["unit"]), str(row["id"])))
                if not selected or len(selected) > GROUP_SIZE:
                    raise AssertionError("V3 K-means produced an invalid request cluster")
                digest = _sha(
                    {
                        "pass": pass_name,
                        "task": task,
                        "endpoint": endpoint,
                        "input_ids": [row["id"] for row in selected],
                    }
                )[:20]
                clusters.append(
                    {
                        "cluster_id": f"{pass_name}_{task}_{digest}",
                        "pass": pass_name,
                        "task": task,
                        "endpoint": endpoint,
                        "rows": selected,
                    }
                )
                task_cluster_count += 1
        vocabulary = sorted((token, int(index)) for token, index in vectorizer.vocabulary_.items())
        task_manifests[task] = {
            "distinct_labels": len(units),
            "endpoint_unit_pairs": len(task_rows),
            "endpoint_count": len(endpoint_rows),
            "cluster_count": task_cluster_count,
            "tfidf_vocabulary_size": len(vocabulary),
            "tfidf_vocabulary_sha256": _sha(vocabulary),
        }
    clusters.sort(key=lambda row: row["cluster_id"])
    covered = [str(item["id"]) for cluster in clusters for item in cluster["rows"]]
    expected = [str(row["id"]) for row in rows]
    if sorted(covered) != sorted(expected) or len(covered) != len(set(covered)):
        raise AssertionError("V3 K-means changed endpoint-unit coverage")
    return clusters, {
        "version": V3_PROTOCOL_VERSION,
        "pass": pass_name,
        "group_size": GROUP_SIZE,
        "random_seed": V3_RANDOM_SEED,
        "kmeans": {"n_init": 10, "algorithm": "lloyd", "recursive_max_size": GROUP_SIZE},
        "tfidf": V3_TFIDF_CONFIG,
        "cluster_count": len(clusters),
        "input_count": len(rows),
        "partial_cluster_count": sum(len(cluster["rows"]) < GROUP_SIZE for cluster in clusters),
        "tasks": task_manifests,
    }


def _v3_write_clusters(root: Path, rows: Sequence[Mapping[str, Any]], pass_name: str) -> dict[str, Any]:
    clusters, manifest = _v3_clusters(rows, pass_name)
    pass_root = root / f"pass_{pass_name}"
    if pass_root.exists():
        raise FileExistsError(pass_root)
    pass_root.mkdir(parents=True)
    cluster_path = pass_root / "clusters.jsonl"
    _write_jsonl(cluster_path, clusters)
    manifest["clusters"] = {"path": str(cluster_path.resolve()), "sha256": file_sha256(cluster_path)}
    _atomic_json(pass_root / "clusters.manifest.json", manifest)
    return manifest


def prepare_v3(run_root: Path) -> dict[str, Any]:
    root = run_root / "v3"
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    all_rows: list[dict[str, Any]] = []
    tasks: dict[str, Any] = {}
    try:
        for task in sorted(TASKS):
            rows, manifest = _v3_eligible_rows(task)
            task_root = root / "tasks" / task
            universe = task_root / "universe.jsonl"
            _write_jsonl(universe, rows)
            manifest.update(
                version=V3_PROTOCOL_VERSION,
                universe={"path": str(universe.resolve()), "sha256": file_sha256(universe)},
            )
            _atomic_json(task_root / "universe.manifest.json", manifest)
            tasks[task] = manifest
            all_rows.extend(rows)
        pass_a = _v3_write_clusters(root, all_rows, "a")
        manifest = {
            "version": V3_PROTOCOL_VERSION,
            "eligibility": "source_accept_or_measurement_resolution_ok",
            "excluded_resolution_statuses": ["relative", "unavailable", "unsure"],
            "tasks": tasks,
            "pass_a": pass_a,
        }
        _atomic_json(root / "manifest.json", manifest)
        return manifest
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def prepare_v3_pass_b(run_root: Path) -> dict[str, Any]:
    root = run_root / "v3"
    pass_a_mapping = root / "pass_a/mapping.jsonl"
    pass_a_manifest = root / "pass_a/manifest.json"
    if not pass_a_mapping.is_file() or not pass_a_manifest.is_file():
        raise FileNotFoundError("V3 Pass A must finish before Pass B preparation")
    manifest_a = json.loads(pass_a_manifest.read_text())
    if file_sha256(pass_a_mapping) != manifest_a.get("mapping_sha256"):
        raise ValueError("V3 Pass-A mapping hash mismatch")
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in _read_jsonl(pass_a_mapping):
        counts[(str(row["task"]), str(row["endpoint"]), str(row["canonical_unit"]))] += int(
            row.get("member_count") or 1
        )
    rows = [
        {
            "id": _v3_identifier("v3b_", task, endpoint, unit),
            "task": task,
            "endpoint": endpoint,
            "unit": unit,
            "member_count": count,
        }
        for (task, endpoint, unit), count in sorted(
            counts.items(), key=lambda item: (item[0][0], item[0][1], _v3_text_key(item[0][2]), item[0][2])
        )
    ]
    result = _v3_write_clusters(root, rows, "b")
    input_path = root / "pass_b/input.jsonl"
    _write_jsonl(input_path, rows)
    result["input"] = {"path": str(input_path.resolve()), "sha256": file_sha256(input_path)}
    result["pass_a_mapping_sha256"] = file_sha256(pass_a_mapping)
    _atomic_json(root / "pass_b/input.manifest.json", result)
    return result


def _v3_words(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z]+|%|µ[a-z]+", _v3_text_key(value)))


def _v3_scale_numbers(value: str) -> tuple[str, ...]:
    """Extract scale-bearing numbers, not formula subscripts or inverse-unit syntax."""
    text = _v3_text_key(value)
    scientific = tuple(
        re.sub(r"\s+", "", match)
        for match in re.findall(r"\b10\s*\^\s*[-+]?\d+", text)
    )
    text = re.sub(r"\b10\s*\^\s*[-+]?\d+", " ", text)
    # `mg-1`, `mg^-1`, and `/mg` are equivalent denominator notations. Their
    # structural -1 is not a magnitude that must appear literally.
    text = re.sub(r"\b[a-zµ]+\s*\^?\s*-\s*1\b", " ", text)
    standalone = tuple(
        re.findall(r"(?<![a-zµ\d])[-+]?\d+(?:\.\d+)?(?![a-z\d])", text)
    )
    return scientific + standalone


def _v3_bases(value: str) -> frozenset[str]:
    return _v3_words(value) & _BASIS_WORDS


def _v3_span_grounded(source: str, span: str) -> bool:
    source_key = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", source)).casefold()
    span_key = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", span)).casefold()
    return bool(span_key) and span_key in source_key


def _v3_guarded_unit(task: str, source: str, span: str, proposed: str) -> tuple[str, str]:
    if not _v3_span_grounded(source, span):
        raise ValueError("unit_span is not a verbatim span of the input unit")
    canonical = _text(proposed)
    if not canonical:
        raise ValueError("canonical_unit is empty")
    if canonical == source:
        return canonical, "identity"
    if canonical == span:
        return canonical, "accepted_grounded_span"
    qualifiers = _qualifiers(span)
    if (
        canonical == "%"
        and "percent" in qualifiers
        and not (qualifiers - {"percent", "inhibition"})
        and not _PROTECTED_PERCENT_BASIS.search(span)
        and not _v3_scale_numbers(span)
    ):
        return canonical, "accepted_percent_semantic_reduction"
    guarded, decision = safe_canonical_unit(span, canonical, task)
    if guarded == canonical and decision.startswith("accepted_"):
        return canonical, decision
    if decision == "identity_scale_guard":
        raise ValueError("canonical unit changed a parsed physical scale or dimension")
    if not _qualifiers(span).issubset(_qualifiers(canonical)):
        raise ValueError("canonical unit dropped a unit qualifier")
    if not _v3_bases(span).issubset(_v3_bases(canonical)):
        raise ValueError("canonical unit dropped an endpoint-defined basis")
    return canonical, "accepted_grounded_reduction"


def _v3_prompt(cluster: Mapping[str, Any], feedback: str = "") -> str:
    rows = [
        {
            "id": row["id"],
            "unit": row["unit"],
            "member_count": int(row.get("member_count") or 1),
        }
        for row in cluster["rows"]
    ]
    suffix = f"\nThe previous response failed validation: {feedback}" if feedback else ""
    return f"""Reconcile measurement-unit labels for the {cluster['task']} endpoint
{cluster['endpoint']}. Return JSON only with exactly this schema:
{{"mappings":[{{"id":"input id","unit_span":"verbatim span from input unit","canonical_unit":"concise unit"}}]}}

Cover every supplied ID exactly once. unit_span must be a nonempty contiguous verbatim
span of that input's unit. canonical_unit must retain the physical unit plus any concise
endpoint-defined basis needed to interpret it, while removing analyte, assay, study,
result, confidence-interval, sample-size, and explanatory prose. A unit is mandatory.

Examples:
- "% overall yield of reactions of chlorambucil" -> unit_span "% overall yield", canonical_unit "%"
- "% inhibition of enzyme activity" -> unit_span "% inhibition", canonical_unit "%"
- "% loss" -> unit_span "% loss", canonical_unit "%"
- "% breaks repaired after 1 h" -> unit_span "% breaks repaired", canonical_unit "%"
- "% GSH depleted" -> unit_span "% GSH depleted", canonical_unit "%"
- "% inhibition relative to vehicle control" -> unit_span "% inhibition relative to vehicle control", canonical_unit "% inhibition vs control"
- "% of total cells" -> unit_span "% of total cells", canonical_unit "% of total cells"
- "10^-7 sec^-1 (deamination rate constant at 95 C)" -> unit_span "10^-7 sec^-1", canonical_unit "10^-7 s^-1"
- "modified residues (10 lysine + 2 histidine)" -> unit_span "modified residues", canonical_unit "modified residues"
- "nmol·min^-1·mg protein^-1" and "nmol/min/mg protein" -> canonical_unit "nmol/min/mg protein"
- "nkat mg-1 protein" -> unit_span "nkat mg-1 protein", canonical_unit "nkat/mg protein" (`mg-1` means per mg)
- "nmol H2O2 consumed/min/mg protein" -> unit_span "nmol H2O2 consumed/min/mg protein", canonical_unit "nmol/min/mg protein" (`H2O2` is the removable analyte, not a scale factor)
- "DNA breaks repaired per minute per cell" -> unit_span "breaks repaired per minute per cell", canonical_unit "breaks/min/cell"
- "ratio of treated signal to control signal" -> unit_span "ratio of treated signal to control signal", canonical_unit "treated/control ratio"
- "variant frequency ratio" -> unit_span "variant frequency ratio", canonical_unit "variant frequency ratio"
- "cells x 10^3/well" -> unit_span "cells x 10^3/well", canonical_unit "10^3 cells/well" (`x` is multiplication here, not fold)
- "nmol/min/mg protein" -> unit_span "nmol/min/mg protein", canonical_unit "nmol min^-1 mg protein^-1" (`/` and inverse exponents both mean per)
- "2-fold increase over control" -> unit_span "2-fold increase over control", canonical_unit "2-fold vs control"

Preserve magnitude, scientific-notation factors, population/sample denominators,
control or reference bases, count, score, ratio, fraction, and other named bases.
Outcome or analyte words attached to a percent value—such as yield, inhibition, loss,
repair, viability, survival, incidence, frequency, or depletion—describe the measurement,
not the unit, and may be removed so the canonical unit is `%`. Never collapse nM, µM,
mM, or differently scaled denominators. Digits inside chemical formulas (for example the
2 characters in H2O2) describe the removable analyte, while a trailing `-1` or `^-1`
on a unit means "per" and may be written with `/`. Do not treat those notational digits
as measurement magnitudes. When uncertain, select the complete unit span and keep it
unchanged. Do not return prose or markdown.{suffix}

Qualifier rule: keep percent outcome words in unit_span when they are the grounded unit
phrase, but canonical_unit may reduce them to `%`. Preserve control, vehicle, baseline,
initial, total, or other reference bases; per or `/` denominators; cells or another
population basis; ratio or fraction; fold; and count semantics. A standalone `x`
adjacent to a numeric factor is multiplication notation, not a fold qualifier. Slash
denominators and inverse exponents such as `/mg`, `mg-1`, and `mg^-1` express the same
per-denominator semantics.

Scale rule: clustering only places similar spellings in one request; it never means the
rows should share one scale. Map every ID independently. SI prefixes are magnitude and
must never be added, removed, or changed. In particular, mmol/L -> mmol/L, µmol/L ->
µmol/L, nmol/L -> nmol/L, and pmol/L -> pmol/L; likewise mM, µM, nM, and pM remain
distinct. Never convert a count or frequency unit into a concentration unit.

INPUTS={json.dumps(rows, ensure_ascii=False, separators=(',', ':'))}"""


def _v3_validate_response(
    text: str, cluster: Mapping[str, Any]
) -> list[dict[str, Any]]:
    parsed = _parse_object(text)
    mappings = parsed.get("mappings")
    expected = {str(row["id"]): str(row["unit"]) for row in cluster["rows"]}
    if not isinstance(mappings, list) or len(mappings) != len(expected):
        raise ValueError(f"expected {len(expected)} mappings")
    output = []
    seen: set[str] = set()
    for mapping in mappings:
        if not isinstance(mapping, dict):
            raise ValueError("each mapping must be an object")
        identifier = _text(mapping.get("id"))
        if identifier not in expected or identifier in seen:
            raise ValueError(f"unexpected or duplicate mapping ID: {identifier!r}")
        span = _text(mapping.get("unit_span"))
        canonical, decision = _v3_guarded_unit(
            str(cluster["task"]), expected[identifier], span, _text(mapping.get("canonical_unit"))
        )
        output.append(
            {
                "id": identifier,
                "unit_span": span,
                "canonical_unit": canonical,
                "guard_decision": decision,
            }
        )
        seen.add(identifier)
    if seen != set(expected):
        raise ValueError("response did not cover every input ID")
    return output


def _v3_http_get(url: str) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(request, timeout=8) as response:
        return int(response.status), response.read().decode("utf-8")


@dataclass
class _V3Endpoint:
    base_url: str
    client: Any
    semaphore: asyncio.Semaphore
    failures: int = 0
    active: bool = True


class _V3RequestError(RuntimeError):
    def __init__(self, endpoint: str, error: Exception):
        super().__init__(f"{endpoint}: {type(error).__name__}: {error}")
        self.endpoint = endpoint


async def _v3_preflight(endpoints: Sequence[str]) -> tuple[list[_V3Endpoint], dict[str, Any]]:
    async def one(base_url: str) -> tuple[_V3Endpoint | None, dict[str, Any]]:
        started = datetime.now(timezone.utc).isoformat()
        root = base_url.removesuffix("/v1")
        client = None
        try:
            health_status, _ = await asyncio.to_thread(_v3_http_get, root + "/health")
            model_status, model_text = await asyncio.to_thread(_v3_http_get, base_url + "/models")
            load_status, load_text = await asyncio.to_thread(_v3_http_get, base_url + "/loads")
            models = json.loads(model_text)
            model_ids = [str(row.get("id")) for row in models.get("data", [])]
            loads = json.loads(load_text)
            if (health_status, model_status, load_status) != (200, 200, 200):
                raise RuntimeError("health, models, or loads returned a non-200 status")
            if LOCAL_MODEL not in model_ids:
                raise RuntimeError(f"expected model is absent: {model_ids}")
            client, _ = async_openai_compatible_client(
                base_url=base_url,
                provider="local",
                max_connections=V3_ENDPOINT_CONCURRENCY,
                timeout_s=3600,
                max_retries=0,
            )
            response = await client.chat.completions.create(
                model=LOCAL_MODEL,
                messages=[{"role": "user", "content": 'Return JSON only: {"ready":true}'}],
                reasoning_effort="high",
                response_format={"type": "json_object"},
                max_tokens=128,
                temperature=0,
            )
            ready = _parse_object(_text(response.choices[0].message.content))
            if ready.get("ready") is not True:
                raise RuntimeError("generation preflight returned unexpected JSON")
            receipt = {
                "base_url": base_url,
                "status": "selected",
                "checked_at": started,
                "model": LOCAL_MODEL,
                "model_ids": model_ids,
                "dp_rank_count": len(loads.get("loads") or []),
                "max_inflight": V3_ENDPOINT_CONCURRENCY,
            }
            return _V3Endpoint(base_url, client, asyncio.Semaphore(V3_ENDPOINT_CONCURRENCY)), receipt
        except Exception as error:
            if client is not None:
                await client.close()
            return None, {
                "base_url": base_url,
                "status": "excluded",
                "checked_at": started,
                "error": f"{type(error).__name__}: {error}"[:2000],
            }

    results = await asyncio.gather(*(one(endpoint) for endpoint in endpoints))
    selected = [runtime for runtime, _ in results if runtime is not None]
    rows = [receipt for _, receipt in results]
    if not selected:
        raise RuntimeError("no V3 local endpoint passed preflight")
    receipt = {
        "version": "canonical_reconciliation_provider_preflight.v1",
        "requested_endpoints": list(endpoints),
        "selected_endpoints": [row for row in rows if row["status"] == "selected"],
        "excluded_endpoints": [row for row in rows if row["status"] == "excluded"],
        "per_endpoint_max_inflight": V3_ENDPOINT_CONCURRENCY,
        "aggregate_max_inflight": len(selected) * V3_ENDPOINT_CONCURRENCY,
        "model": LOCAL_MODEL,
        "reasoning_effort": "high",
        "max_tokens": LOCAL_MAX_TOKENS,
    }
    return selected, receipt


class _V3Pool:
    def __init__(self, endpoints: Sequence[_V3Endpoint]):
        self.endpoints = list(endpoints)
        self.lock = asyncio.Lock()
        self.next_index = 0
        self.events: list[dict[str, Any]] = []

    async def _choose(self, avoided: set[str]) -> _V3Endpoint:
        async with self.lock:
            candidates = [endpoint for endpoint in self.endpoints if endpoint.active and endpoint.base_url not in avoided]
            if not candidates:
                candidates = [endpoint for endpoint in self.endpoints if endpoint.active]
            if not candidates:
                raise RuntimeError("every selected endpoint has been quarantined")
            endpoint = candidates[self.next_index % len(candidates)]
            self.next_index += 1
            return endpoint

    async def complete(self, prompt: str, avoided: set[str]) -> tuple[str, dict[str, Any], str]:
        while True:
            endpoint = await self._choose(avoided)
            await endpoint.semaphore.acquire()
            if endpoint.active:
                break
            endpoint.semaphore.release()
            avoided.add(endpoint.base_url)
        try:
            try:
                response = await endpoint.client.chat.completions.create(
                    model=LOCAL_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    reasoning_effort="high",
                    response_format={"type": "json_object"},
                    max_tokens=LOCAL_MAX_TOKENS,
                    temperature=0,
                )
            except Exception as error:
                async with self.lock:
                    is_read_timeout = type(error).__name__ == "APITimeoutError"
                    endpoint.failures = 0 if is_read_timeout else endpoint.failures + 1
                    if endpoint.failures >= V3_QUARANTINE_FAILURES and endpoint.active:
                        endpoint.active = False
                        self.events.append(
                            {
                                "event": "endpoint_quarantined",
                                "base_url": endpoint.base_url,
                                "failures": endpoint.failures,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                raise _V3RequestError(endpoint.base_url, error) from error
            async with self.lock:
                endpoint.failures = 0
            usage = response.usage.model_dump(exclude_none=True) if response.usage else {}
            return _text(response.choices[0].message.content), usage, endpoint.base_url
        finally:
            endpoint.semaphore.release()

    async def close(self) -> None:
        await asyncio.gather(*(endpoint.client.close() for endpoint in self.endpoints))


def _v3_terminal_events(path: Path, clusters: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    expected = {
        str(cluster["cluster_id"]): [str(row["id"]) for row in cluster["rows"]]
        for cluster in clusters
    }
    events: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return events
    for event in _read_jsonl(path):
        cluster_id = _text(event.get("cluster_id"))
        if cluster_id not in expected or event.get("input_ids") != expected[cluster_id]:
            raise ValueError(f"V3 terminal cache cluster drift: {cluster_id!r}")
        if cluster_id in events:
            raise ValueError(f"duplicate V3 terminal cluster: {cluster_id}")
        events[cluster_id] = event
    return events


def _v3_seed_resolved(
    terminal_path: Path,
    seed_path: Path,
    clusters: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    receipt_path = terminal_path.with_name("seed_resolved.receipt.json")
    seed_sha256 = file_sha256(seed_path)
    if terminal_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("seed_terminal_sha256") != seed_sha256:
            raise ValueError("V3 recovery seed changed after execution began")
        return receipt
    seed_events = _v3_terminal_events(seed_path, clusters)
    inherited = []
    for cluster in clusters:
        event = seed_events.get(str(cluster["cluster_id"]))
        if event is None or event.get("status") != "resolved":
            continue
        inherited.append({**event, "inherited_from_terminal_sha256": seed_sha256})
    _write_jsonl(terminal_path, inherited)
    receipt = {
        "version": "canonical_reconciliation_v3_resolved_seed.v1",
        "seed_terminal": {"path": str(seed_path.resolve()), "sha256": seed_sha256},
        "seed_terminal_sha256": seed_sha256,
        "seed_event_count": len(seed_events),
        "inherited_resolved_count": len(inherited),
        "requeued_count": len(clusters) - len(inherited),
    }
    _atomic_json(receipt_path, receipt)
    return receipt


async def run_v3_pass(
    run_root: Path,
    pass_name: str,
    seed_resolved_terminal: Path | None = None,
) -> dict[str, Any]:
    if pass_name not in {"a", "b"}:
        raise ValueError("V3 pass must be 'a' or 'b'")
    root = run_root / "v3"
    pass_root = root / f"pass_{pass_name}"
    cluster_path = pass_root / "clusters.jsonl"
    cluster_manifest_path = pass_root / "clusters.manifest.json"
    clusters = _read_jsonl(cluster_path)
    cluster_manifest = json.loads(cluster_manifest_path.read_text())
    if file_sha256(cluster_path) != cluster_manifest.get("clusters", {}).get("sha256"):
        raise ValueError("V3 cluster hash mismatch")
    terminal_path = pass_root / "terminal.jsonl"
    seed_receipt = None
    if seed_resolved_terminal is not None:
        seed_receipt = _v3_seed_resolved(terminal_path, seed_resolved_terminal, clusters)
    completed = _v3_terminal_events(terminal_path, clusters)
    missing = [cluster for cluster in clusters if cluster["cluster_id"] not in completed]
    runtimes: list[_V3Endpoint] = []
    pool = None
    receipt = {
        "requested_endpoints": list(V3_ENDPOINTS),
        "selected_endpoints": [],
        "excluded_endpoints": [],
        "aggregate_max_inflight": 0,
    }
    if missing:
        runtimes, receipt = await _v3_preflight(V3_ENDPOINTS)
        _atomic_json(pass_root / "provider_receipt.json", receipt)
        pool = _V3Pool(runtimes)
    write_lock = asyncio.Lock()
    aggregate = max(1, int(receipt["aggregate_max_inflight"]))
    semaphore = asyncio.Semaphore(aggregate)

    async def one(cluster: dict[str, Any]) -> None:
        feedback = ""
        attempts = []
        mappings = None
        used: set[str] = set()
        async with semaphore:
            for attempt in range(1, V3_MAX_ATTEMPTS + 1):
                endpoint = ""
                try:
                    assert pool is not None
                    text, usage, endpoint = await pool.complete(_v3_prompt(cluster, feedback), used)
                    used.add(endpoint)
                    mappings = _v3_validate_response(text, cluster)
                    attempts.append(
                        {"attempt": attempt, "status": "ok", "base_url": endpoint, "usage": usage}
                    )
                    break
                except Exception as error:
                    if isinstance(error, _V3RequestError):
                        endpoint = error.endpoint
                        used.add(endpoint)
                    feedback = f"{type(error).__name__}: {error}"[:1000]
                    attempts.append(
                        {"attempt": attempt, "status": "failed", "base_url": endpoint, "error": feedback}
                    )
        status = "resolved"
        if mappings is None:
            status = "identity_after_retry_exhaustion"
            mappings = [
                {
                    "id": row["id"],
                    "unit_span": row["unit"],
                    "canonical_unit": row["unit"],
                    "guard_decision": "identity_after_retry_exhaustion",
                }
                for row in cluster["rows"]
            ]
        event = {
            "version": V3_PROTOCOL_VERSION,
            "pass": pass_name,
            "cluster_id": cluster["cluster_id"],
            "task": cluster["task"],
            "endpoint": cluster["endpoint"],
            "input_ids": [row["id"] for row in cluster["rows"]],
            "status": status,
            "attempts": attempts,
            "mappings": mappings,
        }
        async with write_lock:
            with terminal_path.open("ab") as handle:
                handle.write(_json_bytes(event))
                handle.flush()
                os.fsync(handle.fileno())
            completed[cluster["cluster_id"]] = event
            if len(completed) % 10 == 0 or len(completed) == len(clusters):
                _atomic_json(
                    pass_root / "status.json",
                    {
                        "pass": pass_name,
                        "finished": len(completed),
                        "total": len(clusters),
                        "identity_after_retry_exhaustion": sum(
                            event["status"] == "identity_after_retry_exhaustion"
                            for event in completed.values()
                        ),
                        "selected_endpoint_count": len(runtimes),
                        "aggregate_max_inflight": receipt["aggregate_max_inflight"],
                        "pool_events": list(pool.events) if pool is not None else [],
                    },
                )

    try:
        await asyncio.gather(*(one(cluster) for cluster in missing))
    finally:
        if pool is not None:
            await pool.close()
    if len(completed) != len(clusters):
        raise ValueError("V3 pass did not reach terminal status for every cluster")
    cluster_by_id = {str(cluster["cluster_id"]): cluster for cluster in clusters}
    mapping_rows = []
    for cluster in clusters:
        event = completed[str(cluster["cluster_id"])]
        by_id = {str(row["id"]): row for row in event["mappings"]}
        for source in cluster["rows"]:
            result = by_id[str(source["id"])]
            mapping_rows.append(
                {
                    **result,
                    "task": cluster["task"],
                    "endpoint": cluster["endpoint"],
                    "input_unit": source["unit"],
                    "member_count": int(source.get("member_count") or 1),
                    "cluster_id": cluster["cluster_id"],
                    "terminal_status": event["status"],
                }
            )
    del cluster_by_id
    mapping_path = pass_root / "mapping.jsonl"
    _write_jsonl(mapping_path, mapping_rows)
    manifest = {
        "version": V3_PROTOCOL_VERSION,
        "pass": pass_name,
        "cluster_manifest_sha256": file_sha256(cluster_manifest_path),
        "cluster_count": len(clusters),
        "mapping_count": len(mapping_rows),
        "mapping_sha256": file_sha256(mapping_path),
        "reasoning_effort": "high",
        "max_tokens": LOCAL_MAX_TOKENS,
        "max_attempts": V3_MAX_ATTEMPTS,
        "resolved_seed": seed_receipt,
        "provider_receipt": receipt,
        "pool_events": list(pool.events) if pool is not None else [],
        "identity_after_retry_exhaustion_clusters": sum(
            event["status"] == "identity_after_retry_exhaustion" for event in completed.values()
        ),
    }
    _atomic_json(pass_root / "manifest.json", manifest)
    return manifest


def snapshot_v3_identities(
    run_root: Path, pass_name: str, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    pass_root = run_root / "v3" / f"pass_{pass_name}"
    cluster_path = pass_root / "clusters.jsonl"
    terminal_path = pass_root / "terminal.jsonl"
    clusters = _read_jsonl(cluster_path)
    cluster_by_id = {str(row["cluster_id"]): row for row in clusters}
    raw = terminal_path.read_bytes()
    prefix_bytes = raw.rfind(b"\n") + 1
    prefix = raw[:prefix_bytes]
    events = [json.loads(line) for line in prefix.splitlines() if line.strip()]
    selected_ids = [
        str(event["cluster_id"])
        for event in events
        if event.get("status") == "identity_after_retry_exhaustion"
    ]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("duplicate terminal identity cluster in V3 snapshot")
    selected = [cluster_by_id[cluster_id] for cluster_id in selected_ids]
    output_dir.mkdir(parents=True)
    selected_path = output_dir / "clusters.jsonl"
    _write_jsonl(selected_path, selected)
    manifest = {
        "version": "canonical_reconciliation_v3_identity_snapshot.v1",
        "protocol_version": V3_PROTOCOL_VERSION,
        "pass": pass_name,
        "identity_cluster_count": len(selected),
        "cluster_source": {"path": str(cluster_path.resolve()), "sha256": file_sha256(cluster_path)},
        "terminal_prefix": {
            "path": str(terminal_path.resolve()),
            "prefix_bytes": prefix_bytes,
            "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
            "event_count": len(events),
        },
        "clusters": {"path": str(selected_path.resolve()), "sha256": file_sha256(selected_path)},
    }
    _atomic_json(output_dir / "snapshot.manifest.json", manifest)
    return manifest


def snapshot_v3_identity_artifact(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    cluster_path = source_dir / "clusters.jsonl"
    terminal_path = source_dir / "terminal.jsonl"
    clusters = _read_jsonl(cluster_path)
    cluster_by_id = {str(row["cluster_id"]): row for row in clusters}
    raw = terminal_path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("closed fallback terminal lacks a complete final line")
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    selected_ids = [
        str(event["cluster_id"])
        for event in events
        if event.get("status") == "identity_after_retry_exhaustion"
    ]
    if len(events) != len(clusters) or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("fallback artifact is incomplete or has duplicate identities")
    selected = [cluster_by_id[cluster_id] for cluster_id in selected_ids]
    output_dir.mkdir(parents=True)
    selected_path = output_dir / "clusters.jsonl"
    _write_jsonl(selected_path, selected)
    manifest = {
        "version": "canonical_reconciliation_v3_identity_snapshot.v1",
        "protocol_version": V3_PROTOCOL_VERSION,
        "pass": events[0]["pass"] if events else "a",
        "identity_cluster_count": len(selected),
        "cluster_source": {"path": str(cluster_path.resolve()), "sha256": file_sha256(cluster_path)},
        "terminal_source": {"path": str(terminal_path.resolve()), "sha256": file_sha256(terminal_path)},
        "clusters": {"path": str(selected_path.resolve()), "sha256": file_sha256(selected_path)},
    }
    _atomic_json(output_dir / "snapshot.manifest.json", manifest)
    return manifest


async def run_v3_openai_fallback(snapshot_dir: Path, workers: int) -> dict[str, Any]:
    if workers < 1 or workers > OPENAI_CONCURRENCY:
        raise ValueError(f"OpenAI workers must be between 1 and {OPENAI_CONCURRENCY}")
    manifest_path = snapshot_dir / "snapshot.manifest.json"
    snapshot = json.loads(manifest_path.read_text())
    cluster_path = snapshot_dir / "clusters.jsonl"
    if file_sha256(cluster_path) != snapshot.get("clusters", {}).get("sha256"):
        raise ValueError("OpenAI fallback identity snapshot hash mismatch")
    clusters = _read_jsonl(cluster_path)
    terminal_path = snapshot_dir / "terminal.jsonl"
    completed = _v3_terminal_events(terminal_path, clusters)
    missing = [cluster for cluster in clusters if cluster["cluster_id"] not in completed]
    client, credential = async_openai_compatible_client(
        base_url=OPENAI_BASE_URL,
        provider="openai",
        env_file=DEFAULT_ENV_FILE,
        credential_env=OPENAI_FALLBACK_CREDENTIAL,
        max_connections=workers,
        timeout_s=3600,
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(workers)
    write_lock = asyncio.Lock()

    async def one(cluster: dict[str, Any]) -> None:
        feedback = ""
        attempts = []
        mappings = None
        async with semaphore:
            for attempt in range(1, V3_MAX_ATTEMPTS + 1):
                try:
                    response = await client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[{"role": "user", "content": _v3_prompt(cluster, feedback)}],
                        reasoning_effort="high",
                        response_format={"type": "json_object"},
                        max_completion_tokens=OPENAI_FALLBACK_MAX_TOKENS,
                    )
                    mappings = _v3_validate_response(
                        _text(response.choices[0].message.content), cluster
                    )
                    usage = response.usage.model_dump(exclude_none=True) if response.usage else {}
                    attempts.append({"attempt": attempt, "status": "ok", "usage": usage})
                    break
                except Exception as error:
                    feedback = f"{type(error).__name__}: {error}"[:1000]
                    attempts.append({"attempt": attempt, "status": "failed", "error": feedback})
        status = "resolved"
        if mappings is None:
            status = "identity_after_retry_exhaustion"
            mappings = [
                {
                    "id": row["id"],
                    "unit_span": row["unit"],
                    "canonical_unit": row["unit"],
                    "guard_decision": "identity_after_retry_exhaustion",
                }
                for row in cluster["rows"]
            ]
        event = {
            "version": V3_PROTOCOL_VERSION,
            "pass": snapshot["pass"],
            "cluster_id": cluster["cluster_id"],
            "task": cluster["task"],
            "endpoint": cluster["endpoint"],
            "input_ids": [row["id"] for row in cluster["rows"]],
            "status": status,
            "provider": "openai",
            "model": OPENAI_MODEL,
            "attempts": attempts,
            "mappings": mappings,
        }
        async with write_lock:
            with terminal_path.open("ab") as handle:
                handle.write(_json_bytes(event))
                handle.flush()
                os.fsync(handle.fileno())
            completed[str(cluster["cluster_id"])] = event
            if len(completed) % 10 == 0 or len(completed) == len(clusters):
                _atomic_json(
                    snapshot_dir / "status.json",
                    {
                        "finished": len(completed),
                        "total": len(clusters),
                        "resolved": sum(row["status"] == "resolved" for row in completed.values()),
                        "identity_after_retry_exhaustion": sum(
                            row["status"] == "identity_after_retry_exhaustion"
                            for row in completed.values()
                        ),
                    },
                )

    try:
        await asyncio.gather(*(one(cluster) for cluster in missing))
    finally:
        await client.close()
    result = {
        "version": "canonical_reconciliation_v3_openai_fallback.v1",
        "protocol_version": V3_PROTOCOL_VERSION,
        "snapshot_manifest_sha256": file_sha256(manifest_path),
        "cluster_count": len(clusters),
        "resolved": sum(row["status"] == "resolved" for row in completed.values()),
        "identity_after_retry_exhaustion": sum(
            row["status"] == "identity_after_retry_exhaustion" for row in completed.values()
        ),
        "provider": "openai",
        "credential_env": credential,
        "model": OPENAI_MODEL,
        "reasoning_effort": "high",
        "max_completion_tokens": OPENAI_FALLBACK_MAX_TOKENS,
        "workers": workers,
        "terminal": {"path": str(terminal_path.resolve()), "sha256": file_sha256(terminal_path)},
    }
    _atomic_json(snapshot_dir / "manifest.json", result)
    return result


def _v3_checked_mapping(root: Path, pass_name: str) -> list[dict[str, Any]]:
    pass_root = root / f"pass_{pass_name}"
    mapping = pass_root / "mapping.jsonl"
    manifest_path = pass_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if file_sha256(mapping) != manifest.get("mapping_sha256"):
        raise ValueError(f"V3 Pass-{pass_name.upper()} mapping hash mismatch")
    rows = _read_jsonl(mapping)
    if len(rows) != manifest.get("mapping_count"):
        raise ValueError(f"V3 Pass-{pass_name.upper()} mapping count mismatch")
    return rows


def _v3_composed_rows(run_root: Path, task: str) -> list[dict[str, Any]]:
    root = run_root / "v3"
    universe = _read_jsonl(root / f"tasks/{task}/universe.jsonl")
    pass_a = [row for row in _v3_checked_mapping(root, "a") if row["task"] == task]
    pass_b = [row for row in _v3_checked_mapping(root, "b") if row["task"] == task]
    a_by_id = {str(row["id"]): row for row in pass_a}
    b_by_key = {(str(row["endpoint"]), str(row["input_unit"])): row for row in pass_b}
    if len(a_by_id) != len(pass_a) or len(b_by_key) != len(pass_b):
        raise ValueError("V3 pass contains duplicate mapping keys")
    if set(a_by_id) != {str(row["id"]) for row in universe}:
        raise ValueError("V3 Pass A does not exactly cover the eligible universe")
    output = []
    for row in universe:
        intermediate = str(a_by_id[str(row["id"])]["canonical_unit"])
        key = (str(row["endpoint"]), intermediate)
        if key not in b_by_key:
            raise ValueError(f"V3 Pass B lacks provisional label: {key!r}")
        output.append(
            {
                "task": task,
                "endpoint": row["endpoint"],
                "input_unit": row["unit"],
                "pass_a_unit": intermediate,
                "canonical_unit": b_by_key[key]["canonical_unit"],
            }
        )
    return output


def prepare_v3_conflicts(run_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    task_manifests = {}
    total = 0
    for task in sorted(TASKS):
        rows = _v3_composed_rows(run_root, task)
        families: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
            lambda: {"inputs": set(), "targets": set()}
        )
        for row in rows:
            family = families[(str(row["endpoint"]), _spelling_key(str(row["input_unit"])))]
            family["inputs"].add(str(row["input_unit"]))
            family["targets"].add(str(row["canonical_unit"]))
        conflicts = []
        for (endpoint, key), family in sorted(families.items()):
            if not key or len(family["targets"]) <= 1:
                continue
            conflict_id = _stable_id("conflict_", f"{task}\0{endpoint}\0{key}")
            conflicts.append(
                {
                    "conflict_id": conflict_id,
                    "task": task,
                    "endpoint": endpoint,
                    "lexical_key": key,
                    "input_units": sorted(family["inputs"]),
                    "canonical_units": sorted(family["targets"]),
                }
            )
        task_root = output_dir / task
        packet_path = task_root / "conflicts.jsonl"
        _write_jsonl(packet_path, conflicts)
        manifest = {
            "version": "canonical_reconciliation_v3_conflicts.v1",
            "task": task,
            "conflict_count": len(conflicts),
            "conflicts": {"path": str(packet_path.resolve()), "sha256": file_sha256(packet_path)},
            "coverage": "every endpoint-local normalized input family with multiple targets",
        }
        _atomic_json(task_root / "manifest.json", manifest)
        task_manifests[task] = manifest
        total += len(conflicts)
    result = {
        "version": "canonical_reconciliation_v3_conflicts.v1",
        "conflict_count": total,
        "tasks": task_manifests,
    }
    _atomic_json(output_dir / "manifest.json", result)
    return result


def _v3_review_aliases(task: str, review_dir: Path) -> tuple[dict[tuple[str, str], str], dict[str, Any]]:
    task_root = review_dir / task
    manifest_path = task_root / "manifest.json"
    conflict_path = task_root / "conflicts.jsonl"
    decisions_path = task_root / "decisions.jsonl"
    completion_path = task_root / "completion.json"
    manifest = json.loads(manifest_path.read_text())
    conflicts = _read_jsonl(conflict_path)
    if manifest.get("conflicts", {}).get("sha256") != file_sha256(conflict_path):
        raise ValueError("V3 conflict packet hash mismatch")
    if manifest.get("conflict_count") != len(conflicts):
        raise ValueError("V3 conflict count mismatch")
    if not conflicts:
        return {}, {"conflict_count": 0, "decision_count": 0, "reviewer_id": None}
    decisions = _read_jsonl(decisions_path)
    completion = json.loads(completion_path.read_text())
    if (
        completion.get("version") != "canonical_reconciliation_v3_agent_completion.v1"
        or completion.get("task") != task
        or completion.get("conflicts_sha256") != file_sha256(conflict_path)
        or completion.get("decisions_sha256") != file_sha256(decisions_path)
        or not _text(completion.get("reviewer_id"))
    ):
        raise ValueError("invalid V3 agent conflict completion receipt")
    by_id = {str(row["conflict_id"]): row for row in conflicts}
    if set(by_id) != {str(row.get("conflict_id")) for row in decisions}:
        raise ValueError("V3 agent decisions do not exactly cover conflicts")
    aliases: dict[tuple[str, str], str] = {}
    for decision in decisions:
        conflict = by_id[str(decision["conflict_id"])]
        action = _text(decision.get("action"))
        if action == "keep_separate":
            continue
        if action != "merge" or not _text(decision.get("rationale")):
            raise ValueError("invalid V3 agent conflict decision")
        target = _text(decision.get("canonical_unit"))
        units = set(conflict["canonical_units"])
        if target not in units:
            raise ValueError("V3 conflict target is outside its endpoint-local family")
        for source in sorted(units - {target}):
            guarded, _ = _v3_guarded_unit(task, source, source, target)
            if guarded != target:
                raise ValueError("unsafe V3 agent conflict merge")
            aliases[(str(conflict["endpoint"]), source)] = target
    return aliases, {
        "conflict_count": len(conflicts),
        "decision_count": len(decisions),
        "reviewer_id": completion["reviewer_id"],
        "conflicts_sha256": file_sha256(conflict_path),
        "decisions_sha256": file_sha256(decisions_path),
        "completion_sha256": file_sha256(completion_path),
    }


def consolidate_v3(task: str, run_root: Path, review_dir: Path) -> dict[str, Any]:
    rows = _v3_composed_rows(run_root, task)
    aliases, review = _v3_review_aliases(task, review_dir)
    for row in rows:
        row["canonical_unit"] = aliases.get(
            (str(row["endpoint"]), str(row["canonical_unit"])), row["canonical_unit"]
        )
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        grouped[(str(row["input_unit"]), str(row["canonical_unit"]))].add(str(row["endpoint"]))
    entries = [
        {
            "task": task,
            "canonical_endpoints": sorted(endpoints),
            "input_unit": input_unit,
            "action": "map",
            "canonical_unit": canonical,
            "scale": "1",
            "domain": "any",
            "review_basis": V3_REVIEW_BASIS,
        }
        for (input_unit, canonical), endpoints in sorted(
            grouped.items(), key=lambda item: (_v3_text_key(item[0][0]), item[0][0], item[0][1])
        )
    ]
    destination = Path(TASKS[task]["v3_output"])
    bundle = destination.parent
    if bundle.exists():
        raise FileExistsError(bundle)
    staging = bundle.with_name(bundle.name + f".tmp.{os.getpid()}")
    staging.mkdir(parents=True)
    try:
        mapping = staging / "mapping.json"
        payload = {
            "version": "starling_exact_measurement_units.v3",
            "task": task,
            "protocol_version": V3_PROTOCOL_VERSION,
            "entries": entries,
        }
        _atomic_json(mapping, payload, pretty=False)
        loaded = load_exact_unit_mapping(mapping)
        expected_keys = {(task, str(row["endpoint"]), str(row["input_unit"])) for row in rows}
        if set(loaded) != expected_keys:
            raise ValueError("V3 publication does not exactly cover endpoint-unit keys")
        long_units = sorted({entry["canonical_unit"] for entry in entries if len(entry["canonical_unit"]) > 80})
        manifest = {
            "version": V3_PROTOCOL_VERSION,
            "task": task,
            "mapping": {"path": str(destination.resolve()), "sha256": file_sha256(mapping)},
            "entry_count": len(entries),
            "endpoint_unit_key_count": len(expected_keys),
            "canonical_unit_count": len({entry["canonical_unit"] for entry in entries}),
            "canonical_units_over_80_characters": len(long_units),
            "long_unit_examples": long_units[:20],
            "all_actions": ["map"],
            "all_scales": ["1"],
            "agent_conflict_review": review,
            "lineage": {
                "pass_a_manifest_sha256": file_sha256(run_root / "v3/pass_a/manifest.json"),
                "pass_b_manifest_sha256": file_sha256(run_root / "v3/pass_b/manifest.json"),
                "universe_manifest_sha256": file_sha256(
                    run_root / f"v3/tasks/{task}/universe.manifest.json"
                ),
            },
        }
        _atomic_json(staging / "manifest.json", manifest)
        review_root = review_dir / task
        shutil.copy2(review_root / "conflicts.jsonl", staging / "conflicts.jsonl")
        shutil.copy2(review_root / "manifest.json", staging / "conflicts.manifest.json")
        if review["conflict_count"]:
            shutil.copy2(review_root / "decisions.jsonl", staging / "agent_review.decisions.jsonl")
            shutil.copy2(review_root / "completion.json", staging / "agent_review.completion.json")
        (staging / "report.md").write_text(
            f"# {task} V10 endpoint-aware V3 unit reconciliation\n\n"
            f"Published {len(expected_keys):,} endpoint/unit keys as {len(entries):,} grouped rules "
            f"with {manifest['canonical_unit_count']:,} canonical labels.\n\n"
            f"Canonical labels longer than 80 characters: {len(long_units):,}.\n",
            encoding="utf-8",
        )
        os.replace(staging, bundle)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def validate_v3(task: str, run_root: Path) -> dict[str, Any]:
    path = Path(TASKS[task]["v3_output"])
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("mapping", {}).get("sha256") != file_sha256(path):
        raise ValueError("V3 publication mapping hash mismatch")
    expected = {
        (task, str(row["endpoint"]), str(row["input_unit"]))
        for row in _v3_composed_rows(run_root, task)
    }
    loaded = load_exact_unit_mapping(path)
    if set(loaded) != expected or any(key[1] == "*" for key in loaded):
        raise ValueError("V3 publication endpoint coverage mismatch")
    return {
        "task": task,
        "valid": True,
        "endpoint_unit_key_count": len(expected),
        "entry_count": manifest["entry_count"],
        "canonical_unit_count": manifest["canonical_unit_count"],
        "canonical_units_over_80_characters": manifest["canonical_units_over_80_characters"],
        "sha256": file_sha256(path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "prepare",
        "run",
        "prepare-review",
        "consolidate-draft",
        "consolidate",
        "validate-draft",
        "validate",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--task", choices=sorted(TASKS), required=True)
        if command not in {"prepare-review"}:
            child.add_argument("--run-root", type=Path, required=True)
        if command == "prepare":
            child.add_argument("--cleaned-records", type=Path)
            child.add_argument("--measurement-resolution", type=Path)
            child.add_argument("--base-mapping", type=Path)
        if command == "run":
            child.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
            child.add_argument("--seed-terminal-cache", type=Path)
        if command == "prepare-review":
            child.add_argument("--mapping", type=Path, required=True)
            child.add_argument("--output-dir", type=Path, required=True)
            child.add_argument("--packet-size", type=int, default=200)
            child.add_argument("--base-mapping", type=Path)
        if command == "consolidate-draft":
            child.add_argument("--output", type=Path, required=True)
        if command == "consolidate":
            child.add_argument("--output", type=Path)
            child.add_argument("--decisions", type=Path, required=True)
            child.add_argument("--review-completion", type=Path, required=True)
        if command in {"validate-draft", "validate"}:
            child.add_argument("--mapping", type=Path)
    for command, plan_only in (("auxiliary-plan", True), ("auxiliary-run", False)):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
        child.set_defaults(auxiliary_plan_only=plan_only)
    prepare_v3_parser = subparsers.add_parser("prepare-v3")
    prepare_v3_parser.add_argument("--run-root", type=Path, required=True)
    run_v3_parser = subparsers.add_parser("run-v3")
    run_v3_parser.add_argument("--run-root", type=Path, required=True)
    run_v3_parser.add_argument("--pass", dest="pass_name", choices=("a", "b"), required=True)
    run_v3_parser.add_argument("--seed-resolved-terminal", type=Path)
    snapshot_parser = subparsers.add_parser("snapshot-v3-identities")
    snapshot_parser.add_argument("--run-root", type=Path, required=True)
    snapshot_parser.add_argument("--pass", dest="pass_name", choices=("a", "b"), required=True)
    snapshot_parser.add_argument("--output-dir", type=Path, required=True)
    artifact_snapshot_parser = subparsers.add_parser("snapshot-v3-identity-artifact")
    artifact_snapshot_parser.add_argument("--source-dir", type=Path, required=True)
    artifact_snapshot_parser.add_argument("--output-dir", type=Path, required=True)
    openai_parser = subparsers.add_parser("run-v3-openai-fallback")
    openai_parser.add_argument("--snapshot-dir", type=Path, required=True)
    openai_parser.add_argument("--workers", type=int, default=OPENAI_CONCURRENCY)
    prepare_b_parser = subparsers.add_parser("prepare-v3-pass-b")
    prepare_b_parser.add_argument("--run-root", type=Path, required=True)
    conflicts_parser = subparsers.add_parser("prepare-v3-conflicts")
    conflicts_parser.add_argument("--run-root", type=Path, required=True)
    conflicts_parser.add_argument("--output-dir", type=Path, required=True)
    consolidate_v3_parser = subparsers.add_parser("consolidate-v3")
    consolidate_v3_parser.add_argument("--task", choices=sorted(TASKS), required=True)
    consolidate_v3_parser.add_argument("--run-root", type=Path, required=True)
    consolidate_v3_parser.add_argument("--review-dir", type=Path, required=True)
    validate_v3_parser = subparsers.add_parser("validate-v3")
    validate_v3_parser.add_argument("--task", choices=sorted(TASKS), required=True)
    validate_v3_parser.add_argument("--run-root", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare(
            args.task,
            args.run_root,
            cleaned_path=args.cleaned_records,
            resolution_path=args.measurement_resolution,
            base_mapping=args.base_mapping,
        )
    elif args.command == "run":
        result = asyncio.run(run_pass(args.task, args.run_root, args.workers, args.seed_terminal_cache))
    elif args.command == "prepare-review":
        result = prepare_review(
            args.task,
            args.mapping,
            args.output_dir,
            args.packet_size,
            args.base_mapping,
        )
    elif args.command == "consolidate-draft":
        result = consolidate_draft(args.task, args.run_root, args.output)
    elif args.command == "consolidate":
        result = consolidate(
            args.task, args.run_root, args.output, args.decisions, args.review_completion
        )
    elif args.command == "validate-draft":
        result = validate(args.task, args.run_root, args.mapping, allow_draft=True)
    elif args.command == "validate":
        result = validate(args.task, args.run_root, args.mapping)
    elif args.command == "prepare-v3":
        result = prepare_v3(args.run_root)
    elif args.command == "run-v3":
        result = asyncio.run(
            run_v3_pass(args.run_root, args.pass_name, args.seed_resolved_terminal)
        )
    elif args.command == "snapshot-v3-identities":
        result = snapshot_v3_identities(args.run_root, args.pass_name, args.output_dir)
    elif args.command == "snapshot-v3-identity-artifact":
        result = snapshot_v3_identity_artifact(args.source_dir, args.output_dir)
    elif args.command == "run-v3-openai-fallback":
        result = asyncio.run(run_v3_openai_fallback(args.snapshot_dir, args.workers))
    elif args.command == "prepare-v3-pass-b":
        result = prepare_v3_pass_b(args.run_root)
    elif args.command == "prepare-v3-conflicts":
        result = prepare_v3_conflicts(args.run_root, args.output_dir)
    elif args.command == "consolidate-v3":
        result = consolidate_v3(args.task, args.run_root, args.review_dir)
    elif args.command == "validate-v3":
        result = validate_v3(args.task, args.run_root)
    else:
        result = auxiliary(args.config, plan_only=args.auxiliary_plan_only)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
