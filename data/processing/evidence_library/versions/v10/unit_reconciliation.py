"""Lossless two-pass LLM reconciliation of V10 measurement units.

Each task is prepared independently, but all three use the same frozen protocol:
50 units per request, two shuffled clustering passes, scale-preserving guards, and
identity fallback after two local DeepSeek attempts plus one OpenAI attempt.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from data.processing.evidence_library.shared.v2.normalization.measurement_resolution import (
    EXACT_UNIT_MAPPING_VERSION,
    load_exact_unit_mapping,
)
from data.processing.llm_api import DEFAULT_ENV_FILE, async_openai_compatible_client
from data.processing.paths import evidence_library_root
from tools.chembl_tool.common.units import canonicalize_unit

PROTOCOL_VERSION = "starling_unit_reconciliation_two_pass.v1"
REVIEW_BASIS = "two_pass_llm_unit_reconciliation.v1"
GROUP_SIZE = 50
LOCAL_BASE_URL = "http://dgx005:50002/v1"
LOCAL_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-5.4-mini"
OPENAI_CREDENTIAL = "OPENAI_API_KEY_ONE"
DEFAULT_WORKERS = 512
INITIAL_MAX_TOKENS = 32_768
RETRY_MAX_TOKENS = 65_536

V10_ROOT = Path(__file__).resolve().parent
TASKS = {
    "ames": {
        "resolution": evidence_library_root("ames", "v10")
        / "measurement_resolution_v6/measurement_resolution.parquet",
    },
    "dili": {
        "resolution": evidence_library_root("dili", "v10")
        / "measurement_resolution_v4/measurement_resolution.parquet",
    },
    "carcinogens": {
        "resolution": evidence_library_root("carcinogens", "v10")
        / "measurement_resolution_v4/measurement_resolution.parquet",
    },
}
for _task, _config in TASKS.items():
    _config["cleaned"] = evidence_library_root(_task, "v10") / "01_cleaned/records.parquet"
    _config["output"] = (
        V10_ROOT
        / f"tasks/{_task}/data_processing/canonicalization_v10/{_task}_unit_reconciliation.v2.json"
    )

_QUALIFIER_PATTERNS = {
    "percent": re.compile(r"%|\bpercent(?:age)?\b", re.I),
    "control": re.compile(r"\bcontrol\b|\bvehicle\b", re.I),
    "inhibition": re.compile(r"\binhib(?:ition|itory|ited)?\b", re.I),
    "fold": re.compile(r"\bfold\b|\btimes?\b|\bx\b", re.I),
    "ratio": re.compile(r"\bratio\b|\bfraction\b", re.I),
    "count": re.compile(r"\bcount\b|\bnumber\b|\bcells?\b", re.I),
    "per": re.compile(r"/|\bper\b", re.I),
}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


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


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(_json_bytes(value, pretty=True))
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


def shuffled_rows(rows: Sequence[Any], manifest_sha256: str) -> list[Any]:
    result = list(rows)
    seed = int(hashlib.sha256(manifest_sha256.encode("ascii")).hexdigest()[:16], 16)
    random.Random(seed).shuffle(result)
    return result


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


def prepare(task: str, run_root: Path) -> dict[str, Any]:
    config = TASKS[task]
    cleaned, resolution = Path(config["cleaned"]), Path(config["resolution"])
    for path in (cleaned, resolution):
        if not path.is_file():
            raise FileNotFoundError(path)
    counts: dict[str, Counter[str]] = {}

    def add(unit: Any, origin: str) -> None:
        value = _text(unit)
        if value:
            counts.setdefault(value, Counter())[origin] += 1

    columns = ("unit_text", "measurement_resolution_exact_unit")
    for batch in pq.ParquetFile(cleaned).iter_batches(columns=list(columns)):
        for row in batch.to_pylist():
            add(row["unit_text"], "source")
            add(row["measurement_resolution_exact_unit"], "exact")
    for batch in pq.ParquetFile(resolution).iter_batches(columns=["status", "measurements_json"]):
        for row in batch.to_pylist():
            if row["status"] != "ok":
                continue
            values = _measurements(row["measurements_json"])
            if len(values) != 1:
                raise ValueError("ok resolution row must have exactly one measurement")
            add(values[0].get("unit"), "llm")
    rows = [
        {
            "id": _stable_id("u_", unit),
            "unit": unit,
            "counts": dict(sorted(origins.items())),
            "total_count": sum(origins.values()),
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
        "partition_count": len(partition_rows(rows)),
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
    Attempt("local", LOCAL_MODEL, INITIAL_MAX_TOKENS),
    Attempt("local", LOCAL_MODEL, RETRY_MAX_TOKENS),
    Attempt("openai", OPENAI_MODEL, RETRY_MAX_TOKENS),
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
            text = await request(spec, _prompt(task, rows, feedback))
            mappings = _validate_response(text, rows, task)
            attempts.append({"attempt": index, **spec.__dict__, "status": "ok"})
            return mappings, attempts, "resolved"
        except Exception as error:  # each failure is durable provenance for the next retry
            feedback = f"{type(error).__name__}: {error}"[:1000]
            attempts.append(
                {"attempt": index, **spec.__dict__, "status": "failed", "error": feedback}
            )
    return _identity(rows), attempts, "identity_after_retry_exhaustion"


def _terminal_events(path: Path, pass_name: str, input_sha256: str) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    events: dict[int, dict[str, Any]] = {}
    for event in _read_jsonl(path):
        if event.get("pass") != pass_name or event.get("input_sha256") != input_sha256:
            raise ValueError(f"cache provenance mismatch: {path}")
        partition = int(event["partition"])
        if partition in events:
            raise ValueError(f"duplicate terminal partition: {partition}")
        events[partition] = event
    return events


async def run_pass(task: str, run_root: Path, pass_name: str, workers: int) -> dict[str, Any]:
    task_root = run_root / task
    universe_manifest = json.loads((task_root / "universe.manifest.json").read_text())
    if pass_name == "a":
        input_rows = _read_jsonl(task_root / "universe.jsonl")
        input_sha = universe_manifest["universe_sha256"]
    else:
        pass_a_manifest_path = task_root / "pass_a/manifest.json"
        if not pass_a_manifest_path.is_file():
            raise FileNotFoundError(pass_a_manifest_path)
        pass_a_rows = _read_jsonl(task_root / "pass_a/mapping.jsonl")
        counts = Counter(row["canonical_unit"] for row in pass_a_rows)
        canonical_rows = [
            {"id": _stable_id("c_", unit), "unit": unit, "member_count": count}
            for unit, count in sorted(
                counts.items(), key=lambda item: (item[0].casefold(), item[0])
            )
        ]
        input_rows = shuffled_rows(canonical_rows, file_sha256(pass_a_manifest_path))
        input_path = task_root / "pass_b/input.jsonl"
        if input_path.exists():
            existing = _read_jsonl(input_path)
            if existing != input_rows:
                raise ValueError("Pass-B frozen input differs from regenerated input")
        else:
            _write_jsonl(input_path, input_rows)
        input_sha = file_sha256(input_path)
    partitions = partition_rows(input_rows)
    pass_root = task_root / f"pass_{pass_name}"
    cache_path = pass_root / "terminal.jsonl"
    completed = _terminal_events(cache_path, pass_name, input_sha)
    local_client, _ = async_openai_compatible_client(
        base_url=LOCAL_BASE_URL,
        provider="local",
        max_connections=workers,
        timeout_s=3600,
        max_retries=0,
    )
    openai_client, credential = async_openai_compatible_client(
        base_url=OPENAI_BASE_URL,
        provider="openai",
        env_file=DEFAULT_ENV_FILE,
        credential_env=OPENAI_CREDENTIAL,
        max_connections=min(workers, 512),
        timeout_s=3600,
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(workers)
    write_lock = asyncio.Lock()

    async def request(spec: Attempt, prompt: str) -> str:
        client = local_client if spec.provider == "local" else openai_client
        kwargs: dict[str, Any] = {
            "model": spec.model,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": "high",
            "response_format": {"type": "json_object"},
        }
        if spec.provider == "local":
            kwargs.update(max_tokens=spec.max_tokens, temperature=0)
        else:
            kwargs["max_completion_tokens"] = spec.max_tokens
        response = await client.chat.completions.create(**kwargs)
        return _text(response.choices[0].message.content)

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
                },
            )

    pass_root.mkdir(parents=True, exist_ok=True)
    await asyncio.gather(
        *(one(index, rows) for index, rows in enumerate(partitions) if index not in completed)
    )
    await local_client.close()
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
        "input_sha256": input_sha,
        "input_count": len(input_rows),
        "partition_count": len(partitions),
        "mapping_sha256": file_sha256(mapping_path),
        "mapping_count": len(output_rows),
        "retry_sequence": [spec.__dict__ for spec in ATTEMPTS],
        "reasoning_effort": "high",
        "credential_env": credential,
        "identity_after_retry_exhaustion_partitions": sum(
            event["status"] == "identity_after_retry_exhaustion" for event in completed.values()
        ),
    }
    _atomic_json(pass_root / "manifest.json", manifest)
    return manifest


def consolidate(task: str, run_root: Path, output: Path | None = None) -> dict[str, Any]:
    task_root = run_root / task
    universe = _read_jsonl(task_root / "universe.jsonl")
    pass_a = _read_jsonl(task_root / "pass_a/mapping.jsonl")
    pass_b_input = _read_jsonl(task_root / "pass_b/input.jsonl")
    pass_b = _read_jsonl(task_root / "pass_b/mapping.jsonl")
    if len(universe) != len(pass_a):
        raise ValueError("Pass A does not cover the universe")
    a_by_id = {row["id"]: row["canonical_unit"] for row in pass_a}
    b_input_by_unit = {row["unit"]: row["id"] for row in pass_b_input}
    b_by_id = {row["id"]: row["canonical_unit"] for row in pass_b}
    entries = []
    for row in universe:
        intermediate = a_by_id[row["id"]]
        final = b_by_id[b_input_by_unit[intermediate]]
        entries.append(
            {
                "task": task,
                "canonical_endpoints": ["*"],
                "input_unit": row["unit"],
                "action": "map",
                "canonical_unit": final,
                "scale": "1",
                "domain": "any",
                "review_basis": REVIEW_BASIS,
            }
        )
    destination = output or Path(TASKS[task]["output"])
    if destination.exists():
        raise FileExistsError(destination)
    payload = {
        "version": EXACT_UNIT_MAPPING_VERSION,
        "task": task,
        "protocol_version": PROTOCOL_VERSION,
        "entries": entries,
    }
    _atomic_json(destination, payload)
    loaded = load_exact_unit_mapping(destination)
    if len(loaded) != len(entries) or any(rule["scale"] != "1" for rule in entries):
        destination.unlink(missing_ok=True)
        raise ValueError("published exact unit mapping failed coverage or scale validation")
    manifest = {
        "version": PROTOCOL_VERSION,
        "task": task,
        "mapping": {"path": str(destination.resolve()), "sha256": file_sha256(destination)},
        "entry_count": len(entries),
        "input_count": len(universe),
        "pass_a_manifest_sha256": file_sha256(task_root / "pass_a/manifest.json"),
        "pass_b_manifest_sha256": file_sha256(task_root / "pass_b/manifest.json"),
        "all_actions": ["map"],
        "all_scales": ["1"],
    }
    _atomic_json(destination.with_suffix(".manifest.json"), manifest)
    return manifest


def validate(task: str, run_root: Path, mapping: Path | None = None) -> dict[str, Any]:
    path = mapping or Path(TASKS[task]["output"])
    universe = _read_jsonl(run_root / task / "universe.jsonl")
    payload = json.loads(path.read_text())
    entries = payload.get("entries", [])
    inputs = [row["unit"] for row in universe]
    mapped = [entry.get("input_unit") for entry in entries]
    if mapped != inputs:
        raise ValueError("mapping order/coverage differs from the frozen universe")
    if any(
        entry.get("action") != "map"
        or entry.get("scale") != "1"
        or not _text(entry.get("canonical_unit"))
        for entry in entries
    ):
        raise ValueError("mapping contains exclusion, non-unit scale, or empty unit")
    load_exact_unit_mapping(path)
    return {"task": task, "valid": True, "entry_count": len(entries), "sha256": file_sha256(path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "consolidate", "validate"):
        child = subparsers.add_parser(command)
        child.add_argument("--task", choices=sorted(TASKS), required=True)
        child.add_argument("--run-root", type=Path, required=True)
        if command == "run":
            child.add_argument("--pass-name", choices=("a", "b"), required=True)
            child.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
        if command in {"consolidate", "validate"}:
            child.add_argument("--mapping", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare(args.task, args.run_root)
    elif args.command == "run":
        result = asyncio.run(run_pass(args.task, args.run_root, args.pass_name, args.workers))
    elif args.command == "consolidate":
        result = consolidate(args.task, args.run_root, args.mapping)
    else:
        result = validate(args.task, args.run_root, args.mapping)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
