"""Prepare and run anchored sliding-window semantic-bucket weight assignment."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any

import httpx
from jinja2 import Environment, StrictUndefined
import pandas as pd

from data.processing.llm_api import async_openai_compatible_client
from data.processing import openrouter_provider_pool as provider_pool
from semantic_buckets import artifacts
from semantic_buckets import bbb_semantic_readout_v1 as bbb
from semantic_buckets import bioavailability_semantic_readout_v1 as core
from semantic_buckets import bioavailability_semantic_readout_v2 as oral_v2
from semantic_buckets import speculative_weight_execution as speculative
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "semantic_weight_sliding.v5"
SEED_MODEL = "deepseek/deepseek-v4-pro"
SEED_BASE_URL = "https://openrouter.ai/api/v1"
FOLLOWUP_MODEL = "deepseek/deepseek-v4-flash-0731"
FOLLOWUP_ENDPOINTS = (
    {
        "name": "openrouter_deepseek_v4_flash_0731",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
        "credential_env": "OPEN_ROUTER_KEY",
        "max_inflight": 27,
    },
)
SEED_REQUEST_COUNT = 9
MAX_TOKENS = 20_480
TASK_TARGETS = {
    "bbb_martins": "whether systemic administration produces experimentally meaningful CNS access",
    "bioavailability_ma": "oral bioavailability under the reported condition",
}
TASK_PAIR_COLUMNS = {
    "bbb_martins": bbb.PAIR_COLUMNS,
    "bioavailability_ma": {key: tuple(value) for key, value in core.PAIR_COLUMNS.items()},
}
TASK_PROMPT_COLUMNS = {
    "bbb_martins": {
        source: tuple(column for column in columns if column != "condition_group")
        for source, columns in bbb.PROMPT_DIMENSION_COLUMNS.items()
    },
    "bioavailability_ma": oral_v2.REFINEMENT_COLUMNS,
}
ROOT = Path(__file__).resolve().parent
PROMPT_ROOT = ROOT / "prompts/semantic_weight_sliding_v2"
PROVENANCE_ROOT = ROOT / "provenance/semantic_weight_sliding_v5"
LEGACY_ROOT = ROOT.parent / "data/legacy"
INPUT_BINDINGS_NAME = "execution_input_bindings.json"
INPUT_BINDINGS_VERSION = "semantic_weight_execution_input_bindings.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)


def _input_paths(task: str) -> dict[str, Path]:
    selected = artifacts.resolve_semantic_bucket_artifacts(task, "v10")
    release = json.loads(selected.manifest.read_text(encoding="utf-8"))
    return {
        "release_manifest": selected.manifest,
        "semantic_map": selected.semantic_map,
        "rankings": selected.semantic_bucket_rankings,
        "record_rankings": selected.record_relevance_rankings,
        "pair_bucket_records": Path(release["upstream"]["pair_bucket_records"]),
    }


def _bucket_payloads(
    task: str, bucket_ids: set[str] | None = None,
    paths: Mapping[str, Path] | None = None,
) -> dict[str, dict[str, Any]]:
    paths = dict(paths or _input_paths(task))
    mapping = pd.read_parquet(paths["semantic_map"])
    if bucket_ids is not None:
        mapping = mapping[mapping.semantic_bucket_id.astype(str).isin(bucket_ids)]
    atoms, cards = _active_atom_material(task, paths)
    lookup = core._atom_lookup(atoms)
    payloads = {}
    for bucket, rows in mapping.groupby("semantic_bucket_id", sort=True):
        members = sorted(rows.atom_id.astype(str))
        payloads[str(bucket)] = _bucket_payload(task, str(bucket), members, lookup, cards)
    if bucket_ids is not None and set(payloads) != bucket_ids:
        raise ValueError(f"semantic payload coverage changed for {task}")
    return payloads


def _pair_values(task: str, source: str, pair_key: str) -> dict[str, str]:
    raw = json.loads(pair_key)
    columns = TASK_PAIR_COLUMNS[task][source]
    if not isinstance(raw, list) or raw[:1] != [source] or len(raw) - 1 > len(columns):
        raise ValueError(f"invalid {task}/{source} pair bucket")
    values = {
        column: str(value) for column, value in zip(columns, raw[1:], strict=False)
    }
    values.update({column: "__not_present_in_pair_key__" for column in columns[len(raw) - 1 :]})
    return values


def _active_atom_material(
    task: str, paths: Mapping[str, Path]
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    records = pd.read_parquet(paths["record_rankings"])
    identity = ["level", "source_id", "pair_bucket_key"]
    if records.groupby("atom_id")[identity].nunique().gt(1).any().any():
        raise ValueError(f"active atom identity changed for {task}")
    representatives = records.sort_values("canonical_record_id").drop_duplicates("atom_id")
    counts = records.groupby("atom_id").size().to_dict()
    prompt_columns = TASK_PROMPT_COLUMNS[task]
    record_columns = sorted({
        "source_row_uid",
        *(column for columns in prompt_columns.values() for column in columns),
    })
    source_records = pd.read_parquet(paths["pair_bucket_records"], columns=record_columns)
    source_records = source_records.drop_duplicates("source_row_uid").set_index("source_row_uid")
    atom_rows, cards = [], {}
    for row in representatives.itertuples(index=False):
        values = _pair_values(task, str(row.source_id), str(row.pair_bucket_key))
        source = source_records.loc[row.source_row_uid]
        cards[str(row.atom_id)] = {
            column: source[column]
            for column in prompt_columns[str(row.source_id)]
            if pd.notna(source.get(column)) and str(source[column]).strip()
        }
        atom_rows.append({
            "atom_id": str(row.atom_id), "source_id": str(row.source_id),
            "record_count": int(counts[row.atom_id]),
            "values_json": core._canonical_json(values),
        })
    return pd.DataFrame(atom_rows), cards


def _bucket_payload(
    task: str, bucket: str, atom_ids: list[str], lookup: Mapping[str, Mapping[str, Any]],
    cards: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_source: dict[str, list[str]] = {}
    for atom in atom_ids:
        by_source.setdefault(str(lookup[atom]["source_id"]), []).append(atom)
    components = [
        {"source_id": source, "canonical_dimensions": {
            column: sorted({str(lookup[atom]["values"][column]) for atom in members})
            for column in TASK_PROMPT_COLUMNS[task][source]
        }}
        for source, members in sorted(by_source.items())
    ]
    ordered = sorted(atom_ids, key=lambda atom: (core._stable_id("ranking-sample", bucket, atom), atom))
    selected = _source_balanced_samples(ordered, lookup)
    samples = [{"source_id": str(lookup[atom]["source_id"]), **cards[atom]} for atom in selected]
    return {"identity": {"source_components": components}, "sample_records": samples}


def _source_balanced_samples(
    ordered: Sequence[str], lookup: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    selected, seen_sources = [], set()
    for require_new_source in (True, False):
        for atom in ordered:
            source = str(lookup[atom]["source_id"])
            if atom in selected or (require_new_source and source in seen_sources):
                continue
            selected.append(atom)
            seen_sources.add(source)
            if len(selected) == core.SAMPLE_LIMIT:
                return selected
    return selected


def _schedule_level(task: str, level: str, bucket_ids: list[str]) -> list[dict[str, Any]]:
    if len(bucket_ids) < 12:
        raise ValueError(f"{task}/{level} has fewer than 12 semantic buckets")
    rows = [_schedule_row(task, level, 0, 0, bucket_ids[:12], [])]
    remaining = bucket_ids[12:]
    anchors = bucket_ids[7:12]
    round_index = 1
    while remaining:
        block, remaining = remaining[:21], remaining[21:]
        batches = [block[start : start + 7] for start in range(0, len(block), 7)]
        rows.extend(
            _schedule_row(task, level, round_index, index, batch, anchors)
            for index, batch in enumerate(batches)
        )
        anchors = [bucket for batch in batches for bucket in batch[-2:]]
        round_index += 1
    ranks = {bucket: index for index, bucket in enumerate(bucket_ids, 1)}
    for row in rows:
        candidates = json.loads(row["candidate_bucket_ids_json"])
        row["candidate_level_ranks_json"] = core._canonical_json(
            [ranks[bucket] for bucket in candidates]
        )
    return rows


def _schedule_row(
    task: str, level: str, round_index: int, batch_index: int,
    candidates: Sequence[str], anchors: Sequence[str],
) -> dict[str, Any]:
    batch_id = f"{task}-{level}-r{round_index:04d}-b{batch_index:02d}"
    return {
        "batch_id": batch_id,
        "task": task,
        "level": level,
        "round": round_index,
        "batch": batch_index,
        "candidate_bucket_ids_json": core._canonical_json(list(candidates)),
        "anchor_bucket_ids_json": core._canonical_json(list(anchors)),
    }


def build_schedule() -> pd.DataFrame:
    rows = []
    for task in TASK_TARGETS:
        rankings = pd.read_parquet(_input_paths(task)["rankings"])
        rankings = rankings.sort_values(["level", "level_rank"])
        for level, group in rankings.groupby("level", sort=True):
            ids = group.semantic_bucket_id.astype(str).tolist()
            rows.extend(_schedule_level(task, str(level), ids))
    schedule = pd.DataFrame(rows).sort_values(
        ["round", "task", "level", "batch"]
    ).reset_index(drop=True)
    candidates = [
        bucket
        for value in schedule.candidate_bucket_ids_json
        for bucket in json.loads(value)
    ]
    if len(candidates) != len(set(candidates)):
        raise ValueError("one semantic bucket was scheduled more than once")
    return schedule


def _clean(value: Any) -> str:
    return " ".join(str(value).split())


def _display_values(values: Any) -> str:
    if isinstance(values, Mapping):
        values = values.get("examples", [])
    if not isinstance(values, list):
        values = [values]
    return " | ".join(_clean(value) for value in values)


def _card(label: str, payload: Mapping[str, Any]) -> str:
    lines = [label]
    for component in payload["identity"]["source_components"]:
        lines.append(f"  Source: {_clean(component['source_id'])}")
        lines.append("  Canonical dimensions:")
        for field, values in component["canonical_dimensions"].items():
            lines.append(f"    - {field.replace('_', ' ')}: {_display_values(values)}")
    lines.append("  Representative records:")
    for index, sample in enumerate(payload["sample_records"], 1):
        source = sample.get("source_id")
        suffix = f" ({_clean(source)})" if source else ""
        lines.append(f"    Record {index}{suffix}:")
        for field, value in sample.items():
            if field != "source_id":
                lines.append(f"      - {field.replace('_', ' ')}: {_display_values(value)}")
    return "\n".join(lines)


def _render(
    task: str, level: str, candidate_ids: Sequence[str], anchor_ids: Sequence[str],
    payloads: Mapping[str, Mapping[str, Any]], scores: Mapping[str, Mapping[str, Any]],
) -> str:
    candidate_cards = "\n\n".join(
        _card(f"Candidate {index}", payloads[bucket])
        for index, bucket in enumerate(candidate_ids, 1)
    )
    anchor_cards = "\n\n".join(
        _anchor_card(index, bucket, payloads[bucket], scores[bucket])
        for index, bucket in enumerate(anchor_ids, 1)
    )
    name = "anchored.jinja" if anchor_ids else "initial.jinja"
    template = Environment(undefined=StrictUndefined, autoescape=False).from_string(
        (PROMPT_ROOT / name).read_text(encoding="utf-8")
    )
    return template.render(
        target=TASK_TARGETS[task], level=level, candidate_cards=candidate_cards,
        anchor_cards=anchor_cards,
    ).strip()


def _anchor_card(
    index: int, bucket: str, payload: Mapping[str, Any], score: Mapping[str, Any]
) -> str:
    heading = (
        f"Anchor {index} — locked weight {score['weight']}\n"
        f"  Prior rationale: {_clean(score['rationale'])}"
    )
    body = _card("  Scientific profile", payload).splitlines()[1:]
    return "\n".join([heading, *body])


def _preview_scores(bucket_ids: Sequence[str]) -> dict[str, dict[str, str]]:
    return {
        bucket: {
            "weight": "<weight from the completed preceding call>",
            "rationale": "<rationale from the completed preceding call>",
        }
        for bucket in bucket_ids
    }


def _write_prompt_review(root: Path, schedule: pd.DataFrame) -> dict[str, Any]:
    review = root / "prompt_review"
    review.mkdir(exist_ok=True)
    if any(review.iterdir()):
        raise FileExistsError(f"prompt review directory is not empty: {review}")
    files = []
    for task in TASK_TARGETS:
        task_rows = schedule[schedule.task.eq(task)]
        wanted = {
            bucket
            for value in task_rows[task_rows["round"].le(1)].candidate_bucket_ids_json
            for bucket in json.loads(value)
        }
        payloads = _bucket_payloads(task, wanted)
        for level in sorted(task_rows.level.unique()):
            examples = task_rows[task_rows.level.eq(level)].sort_values(["round", "batch"])
            for row in examples.iloc[:2].itertuples(index=False):
                candidates = json.loads(row.candidate_bucket_ids_json)
                anchors = json.loads(row.anchor_bucket_ids_json)
                prompt = _render(task, level, candidates, anchors, payloads, _preview_scores(anchors))
                path = review / f"{task}_{level}_r{row.round:04d}_b{row.batch:02d}.txt"
                path.write_text(prompt + "\n", encoding="utf-8")
                files.append({"path": path.name, "sha256": _sha256(path)})
    manifest = {
        "version": f"{VERSION}.prompt_review",
        "status": "awaiting_user_prompt_approval",
        "completion_requests_made": 0,
        "files": files,
    }
    write_json_atomic(review / "manifest.json", manifest)
    return manifest


def prepare(run_id: str) -> dict[str, Any]:
    root = _run_root(run_id)
    if (root / "manifest.json").exists():
        raise FileExistsError(root)
    allowed = {"schedule.parquet", "schedule.tsv", "prompt_review"}
    if root.exists() and {path.name for path in root.iterdir()} - allowed:
        raise FileExistsError(f"incomplete run has unexpected files: {root}")
    root.mkdir(parents=True, exist_ok=True)
    schedule = build_schedule()
    _atomic_parquet(schedule, root / "schedule.parquet")
    _atomic_tsv(schedule, root / "schedule.tsv")
    review = _write_prompt_review(root, schedule)
    inputs = {
        f"{task}/{name}": {"path": str(path), "sha256": _sha256(path)}
        for task in TASK_TARGETS
        for name, path in _input_paths(task).items()
    }
    templates = {
        path.name: _sha256(path) for path in sorted(PROMPT_ROOT.glob("*.jinja"))
    }
    manifest = {
        "version": VERSION,
        "status": "awaiting_prompt_approval",
        "run_id": run_id,
        "storage": {"root": str(root), "filesystem": "vast_nfs", "local_staging": False},
        "execution_stages": {
            "seed": {
                "round": 0,
                "model": SEED_MODEL,
                "base_url": SEED_BASE_URL,
                "reasoning_effort": "high",
            },
            "anchored": {
                "rounds": "1+",
                "default_model": FOLLOWUP_MODEL,
                "model_profile": "mixed",
                "models": [provider_pool.DEFAULT_MODEL, provider_pool.MIXED_MODEL],
                "reasoning_effort": "high",
                "endpoints": [dict(endpoint) for endpoint in FOLLOWUP_ENDPOINTS],
                "aggregate_max_inflight": sum(
                    endpoint["max_inflight"] for endpoint in FOLLOWUP_ENDPOINTS
                ),
            },
        },
        "reasoning_effort": "high",
        "max_tokens": MAX_TOKENS,
        "completion_requests_made": 0,
        "bucket_count": _scheduled_bucket_count(schedule),
        "request_count": len(schedule),
        "schedule_sha256": _sha256(root / "schedule.parquet"),
        "prompt_review_manifest_sha256": _sha256(root / "prompt_review/manifest.json"),
        "inputs": inputs,
        "templates": templates,
        "review": review,
    }
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def _run_root(run_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", run_id):
        raise ValueError("run_id must contain only lowercase letters, digits, dot, dash, underscore")
    root = (PROVENANCE_ROOT / run_id).resolve()
    if not root.is_relative_to(PROVENANCE_ROOT.resolve()):
        raise ValueError("run_id escapes the provenance root")
    return root


def _scheduled_bucket_count(schedule: pd.DataFrame) -> int:
    return sum(len(json.loads(value)) for value in schedule.candidate_bucket_ids_json)


def _load_input_bindings(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Path]:
    path = root / INPUT_BINDINGS_NAME
    if not path.exists():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("version") != INPUT_BINDINGS_VERSION:
        raise ValueError("execution input bindings have the wrong version")
    if document.get("run_id") != manifest.get("run_id"):
        raise ValueError("execution input bindings have the wrong run identity")
    bindings = document.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) - set(manifest["inputs"]):
        raise ValueError("execution input bindings name unknown prepared inputs")
    legacy_root = LEGACY_ROOT.resolve()
    resolved = {}
    for name, binding in bindings.items():
        prepared = manifest["inputs"][name]
        if (binding.get("original_path") != prepared["path"] or
                binding.get("expected_sha256") != prepared["sha256"]):
            raise ValueError(f"execution input binding disagrees with {name}")
        replacement = Path(binding["replacement_path"]).resolve()
        if not replacement.is_relative_to(legacy_root):
            raise ValueError(f"execution input binding is not immutable legacy data: {name}")
        if (binding.get("replacement_sha256") != prepared["sha256"] or
                _sha256(replacement) != prepared["sha256"]):
            raise ValueError(f"execution input binding hash mismatch: {name}")
        resolved[name] = replacement
    return resolved


def _resolved_prepared_inputs(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Path]:
    bindings = _load_input_bindings(root, manifest)
    resolved = {}
    for name, item in manifest["inputs"].items():
        original = Path(item["path"])
        if original.exists() and _sha256(original) == item["sha256"]:
            resolved[name] = original
        elif name in bindings:
            resolved[name] = bindings[name]
        else:
            raise ValueError(f"prepared input changed: {item['path']}")
    return resolved


def _verify_prepared(
    root: Path, approved_review_sha256: str
) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION:
        raise ValueError("prepared run has the wrong version")
    review_hash = _sha256(root / "prompt_review/manifest.json")
    if approved_review_sha256 != review_hash:
        raise ValueError("approved prompt-review hash does not match")
    if _sha256(root / "schedule.parquet") != manifest["schedule_sha256"]:
        raise ValueError("prepared schedule changed")
    resolved = _resolved_prepared_inputs(root, manifest)
    for name, digest in manifest["templates"].items():
        if _sha256(PROMPT_ROOT / name) != digest:
            raise ValueError(f"prompt template changed: {name}")
    return manifest, resolved


def _all_bucket_payloads(
    resolved: Mapping[str, Path]
) -> dict[str, dict[str, Any]]:
    payloads = {}
    for task in TASK_TARGETS:
        prefix = f"{task}/"
        paths = {
            name.removeprefix(prefix): path
            for name, path in resolved.items()
            if name.startswith(prefix)
        }
        payloads.update(_bucket_payloads(task, paths=paths))
    return payloads


def _record_input_bindings(manifest: dict[str, Any], root: Path) -> None:
    path = root / INPUT_BINDINGS_NAME
    if path.exists():
        manifest["execution_input_bindings"] = {
            "path": str(path), "sha256": _sha256(path),
        }


def _configure_execution(round_index: int, parallelism: int) -> str:
    if round_index == 0:
        core.MODEL = SEED_MODEL
        core.BASE_URL = SEED_BASE_URL
        core.ENDPOINTS = ({
            "name": "openrouter_deepseek_v4_pro",
            "base_url": SEED_BASE_URL,
            "provider": "openrouter",
            "credential_env": "OPEN_ROUTER_KEY",
            "max_inflight": parallelism,
        },)
        core.WEIGHT_REQUEST_EXTRA_BODY = {
            "thinking": {"type": "enabled"}
        }
        stage = "seed"
    else:
        core.MODEL = FOLLOWUP_MODEL
        core.BASE_URL = FOLLOWUP_ENDPOINTS[0]["base_url"]
        core.ENDPOINTS = FOLLOWUP_ENDPOINTS
        core.WEIGHT_REQUEST_EXTRA_BODY = {
            "thinking": {"type": "enabled"}
        }
        stage = "anchored"
    core.REQUEST_TIMEOUT_S = 300
    core.MAX_ATTEMPTS = 4
    return stage


def _flash_load_receipts() -> list[dict[str, Any]]:
    receipts = []
    for endpoint in FOLLOWUP_ENDPOINTS:
        response = httpx.get(
            endpoint["base_url"].rstrip("/") + "/loads", timeout=10
        )
        if response.status_code == 404:
            metrics_url = endpoint["base_url"].rsplit("/v1", 1)[0] + "/metrics"
            metrics = httpx.get(metrics_url, timeout=10)
            metrics.raise_for_status()
            running = re.findall(r"^vllm:num_requests_running\{[^}]*\}\s+([0-9.eE+-]+)$",
                                 metrics.text, re.MULTILINE)
            waiting = re.findall(r"^vllm:num_requests_waiting\{[^}]*\}\s+([0-9.eE+-]+)$",
                                 metrics.text, re.MULTILINE)
            if not running or len(running) != len(waiting):
                raise ValueError(f"invalid vLLM metrics from {endpoint['name']}")
            loads = [{"num_running_reqs": int(float(value)),
                      "num_waiting_reqs": int(float(queued))}
                     for value, queued in zip(running, waiting, strict=True)]
        else:
            response.raise_for_status()
            loads = response.json().get("loads")
            if not isinstance(loads, list) or not loads:
                raise ValueError(f"invalid load response from {endpoint['name']}")
        receipts.append({
            "name": endpoint["name"],
            "base_url": endpoint["base_url"],
            "data_parallel_ranks": len(loads),
            "running_requests": sum(int(load["num_running_reqs"]) for load in loads),
            "waiting_requests": sum(int(load["num_waiting_reqs"]) for load in loads),
        })
    return receipts


def _completed_scores(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    scores = {}
    aggregates = speculative.aggregate_lookup(connection)
    rows = connection.execute(
        "SELECT request_id,validation_json,response_json,served_model,provider_name,"
        "attempts,input_tokens,output_tokens FROM requests "
        "WHERE kind='weight_assignment' AND status='complete'"
    )
    for row in rows:
        validation = json.loads(row["validation_json"])
        response = json.loads(row["response_json"])
        buckets = dict(zip(validation["candidate_aliases"], validation["candidate_bucket_ids"], strict=True))
        for score in response["scores"]:
            bucket = buckets[score["candidate"]]
            if bucket in scores:
                raise ValueError(f"semantic bucket was completed twice: {bucket}")
            aggregate = aggregates.get(row["request_id"])
            component = (aggregate or {}).get("components", {}).get(score["candidate"])
            scores[bucket] = {
                "weight": float(score["weight"]),
                "rationale": score["rationale"],
                "request_id": row["request_id"],
                "anchor_bucket_ids": validation["anchor_bucket_ids"],
                "served_model": row["served_model"],
                "attempts": row["attempts"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "requested_model": row["served_model"] if aggregate else validation.get("requested_model"),
                "selected_provider": row["provider_name"] or validation.get("selected_provider_route"),
                "provider_pool_snapshot_sha256": validation.get(
                    "provider_pool_snapshot_sha256"
                ),
                "aggregation_method": (aggregate or {}).get(
                    "aggregation_method", "single_response"
                ),
                "replicate_count": (aggregate or {}).get("replicate_count", 1),
                "weight_stddev": component.get("sample_stddev") if component else None,
                "weight_min": component.get("min") if component else float(score["weight"]),
                "weight_max": component.get("max") if component else float(score["weight"]),
                "component_weights_json": core._canonical_json(
                    component["component_weights"] if component else [float(score["weight"])]
                ),
            }
    return scores


def _queue_round(
    connection: sqlite3.Connection, rows: pd.DataFrame,
    payloads: Mapping[str, Mapping[str, Any]], scores: Mapping[str, Mapping[str, Any]],
    provider_snapshot: Mapping[str, Any] | None = None,
    anchored_execution: str = "openrouter",
) -> list[str]:
    request_ids = []
    for row in rows.itertuples(index=True):
        candidates = json.loads(row.candidate_bucket_ids_json)
        anchors = json.loads(row.anchor_bucket_ids_json)
        missing = set(anchors) - scores.keys()
        if missing:
            raise ValueError(f"{row.batch_id} has incomplete anchors: {sorted(missing)}")
        completed = [candidate in scores for candidate in candidates]
        if any(completed):
            if not all(completed):
                raise ValueError(f"{row.batch_id} has partially completed candidates")
            completed_ids = {scores[candidate]["request_id"] for candidate in candidates}
            if len(completed_ids) != 1:
                raise ValueError(f"{row.batch_id} candidates span completed requests")
            request_ids.append(completed_ids.pop())
            continue
        prompt = _render(row.task, row.level, candidates, anchors, payloads, scores)
        request_id = core._request_id("weight_assignment", row.batch_id, prompt)
        if connection.execute(
            "SELECT 1 FROM requests WHERE request_id=?", (request_id,)
        ).fetchone():
            request_ids.append(request_id)
            continue
        aliases = [f"Candidate {index}" for index in range(1, len(candidates) + 1)]
        validation = {"candidate_aliases": aliases, "candidate_bucket_ids": candidates,
                      "anchor_bucket_ids": anchors}
        if row.round > 0:
            if anchored_execution == "local-speculative":
                validation.update(
                    requested_model=speculative.DEFAULT_MODEL,
                    allowed_served_models=[speculative.DEFAULT_MODEL],
                    selected_provider_route="dgx008_speculative_first8",
                )
            else:
                if provider_snapshot is None:
                    raise ValueError("anchored requests require a provider-pool snapshot")
                validation.update(provider_pool.request_assignment(
                    provider_snapshot, int(row.Index), seed_request_count=SEED_REQUEST_COUNT
                ))
        core._queue_request(
            connection, request_id=request_id, kind="weight_assignment", phase=row.batch_id,
            prompt=prompt, reasoning_effort="high", max_tokens=MAX_TOKENS,
            validation=validation, commit=False,
        )
        request_ids.append(request_id)
    connection.commit()
    return request_ids


def _pending_request_count(
    connection: sqlite3.Connection, request_ids: Sequence[str]
) -> int:
    placeholders = ",".join("?" for _ in request_ids)
    return int(connection.execute(
        f"SELECT COUNT(*) FROM requests WHERE request_id IN ({placeholders}) "
        "AND status != 'complete'",
        list(request_ids),
    ).fetchone()[0])


def _score_frame(
    scores: Mapping[str, Mapping[str, Any]], schedule: pd.DataFrame
) -> pd.DataFrame:
    metadata = {}
    for row in schedule.itertuples(index=False):
        candidates = json.loads(row.candidate_bucket_ids_json)
        ranks = json.loads(row.candidate_level_ranks_json)
        for offset, (bucket, level_rank) in enumerate(zip(candidates, ranks, strict=True)):
            metadata[bucket] = {
                "task": row.task, "level": row.level, "round": row.round,
                "batch": row.batch, "position_in_batch": offset + 1,
                "level_rank": level_rank,
            }
    rows = [
        {"semantic_bucket_id": bucket, **metadata[bucket], **score,
         "anchor_bucket_ids_json": core._canonical_json(score["anchor_bucket_ids"])}
        for bucket, score in scores.items()
    ]
    if not rows:
        return pd.DataFrame(columns=[
            "semantic_bucket_id", "task", "level", "level_rank", "weight", "rationale",
        ])
    frame = pd.DataFrame(rows).drop(columns="anchor_bucket_ids", errors="ignore")
    return frame.sort_values(["task", "level", "level_rank"])


def _completion_progress(
    scores: Mapping[str, Mapping[str, Any]], schedule: pd.DataFrame,
) -> tuple[int, dict[str, int]]:
    completed = set(scores)
    progress = {}
    for (task, level), rows in schedule.groupby(["task", "level"], sort=True):
        finished = -1
        for round_index, round_rows in rows.groupby("round", sort=True):
            candidates = [bucket for value in round_rows.candidate_bucket_ids_json
                          for bucket in json.loads(value)]
            if not set(candidates) <= completed:
                break
            finished = int(round_index)
        progress[f"{task}/{level}"] = finished
    global_round = -1
    for round_index, rows in schedule.groupby("round", sort=True):
        candidates = [bucket for value in rows.candidate_bucket_ids_json
                      for bucket in json.loads(value)]
        if not set(candidates) <= completed:
            break
        global_round = int(round_index)
    return global_round, progress


def _checkpoint(
    root: Path, manifest: dict[str, Any], scores: Mapping[str, Mapping[str, Any]],
    schedule: pd.DataFrame,
) -> None:
    frame = _score_frame(scores, schedule)
    _atomic_parquet(frame, root / "weights.partial.parquet")
    _atomic_tsv(frame, root / "weights.partial.tsv")
    global_round, progress = _completion_progress(scores, schedule)
    manifest.update(
        status="running", completed_through_round=global_round,
        completed_through_round_by_task_level=progress,
        completed_bucket_count=len(frame),
        completion_requests_made=len({score["request_id"] for score in scores.values()}),
    )
    write_json_atomic(root / "manifest.json", manifest)


def _checkpoint_completed(
    root: Path, manifest: dict[str, Any], connection: sqlite3.Connection,
    schedule: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    scores = _completed_scores(connection)
    _checkpoint(root, manifest, scores, schedule)
    return scores


def _record_provider_snapshot(
    manifest: dict[str, Any], snapshot: Mapping[str, Any]
) -> None:
    receipts = manifest.setdefault("openrouter_provider_pool_snapshots", [])
    digest = snapshot["snapshot_sha256"]
    if any(receipt["snapshot_sha256"] == digest for receipt in receipts):
        return
    receipts.append({
        "snapshot_sha256": digest, "fetched_at": snapshot["fetched_at"],
        "qualification": snapshot["qualification"], "profile": snapshot["profile"],
        "requests_per_route": provider_pool.REQUESTS_PER_ROUTE,
        "routes": [{key: route[key] for key in (
            "model", "canonical_model", "route_tag", "provider_name",
            "active_output_price", "throughput_p50", "composite_score",
        )} for route in snapshot["routes"]],
    })


def _completion_pool(
    stage: str, pools: dict[str, tuple[Any, list[dict[str, Any]], int]],
    manifest: dict[str, Any], root: Path, parallelism: int,
) -> tuple[Any, list[dict[str, Any]], int]:
    if stage in pools:
        return pools[stage]
    if stage == "anchored" and all(
        endpoint["provider"] == "local" for endpoint in FOLLOWUP_ENDPOINTS
    ):
        loads = _flash_load_receipts()
        preflights = manifest.setdefault("flash_server_load_preflights", [])
        previous = manifest.get("flash_server_load_preflight")
        if previous and not preflights:
            preflights.append(previous)
        preflights.append(loads)
        manifest["flash_server_load_preflight"] = loads
    pools[stage] = core._build_completion_pool(parallelism)
    starts = manifest.setdefault("endpoint_pools_at_start", {})
    if stage in starts:
        manifest.setdefault("endpoint_pool_resumes", []).append({
            "stage": stage, "receipts": pools[stage][1]
        })
    else:
        starts[stage] = pools[stage][1]
    write_json_atomic(root / "manifest.json", manifest)
    return pools[stage]


def _validate_speculative_cutover(
    connection: sqlite3.Connection, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    if manifest.get("status") not in {"paused_at_round_boundary", "incomplete"}:
        raise ValueError("speculative cutover requires a stopped prior controller")
    by_chain: dict[str, dict[int, Counter[str]]] = {}
    rows = connection.execute(
        "SELECT phase,status FROM requests WHERE kind='weight_assignment'"
    )
    for row in rows:
        match = re.search(r"-r(\d{4})-", row["phase"])
        if not match:
            raise ValueError(f"weight request has an invalid phase: {row['phase']}")
        round_index = int(match.group(1))
        chain = row["phase"].split(f"-r{round_index:04d}-", 1)[0]
        by_chain.setdefault(chain, {}).setdefault(round_index, Counter())[row["status"]] += 1
    frontiers = {}
    for chain, by_round in by_chain.items():
        unfinished = [index for index, counts in by_round.items()
                      if set(counts) != {"complete"}]
        if len(unfinished) > 1 or (unfinished and unfinished[0] != max(by_round)):
            raise ValueError(f"{chain} has work beyond an incomplete round")
        frontiers[chain] = {
            "latest_queued_round": max(by_round),
            "incomplete_round": unfinished[0] if unfinished else None,
            "incomplete_status_counts": dict(by_round[unfinished[0]]) if unfinished else {},
        }
    return {
        "completed_through_round": int(manifest.get("completed_through_round", -1)),
        "task_level_frontiers": frontiers,
    }


def _record_speculative_migration(
    manifest: dict[str, Any], benchmark_path: Path, boundary: Mapping[str, Any],
    required_replicas: int, fanout: int | None,
) -> None:
    benchmark, digest = speculative.load_benchmark(benchmark_path)
    receipts = manifest.setdefault("execution_migrations", [])
    method = speculative.aggregation_method(required_replicas)
    selected_fanout = int(fanout or benchmark["selection"]["fanout"])
    if not 1 <= required_replicas <= selected_fanout <= speculative.MAX_FANOUT:
        raise ValueError("speculative migration requires 1 <= replicas <= fanout <= 128")
    if any(row.get("to") == method for row in receipts):
        return
    receipts.append({
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        **boundary,
        "from": receipts[-1]["to"] if receipts else "openrouter_single_response",
        "to": method,
        "endpoint": benchmark["endpoint"],
        "selection": {
            "fanout": selected_fanout,
            "maximum_fanout": selected_fanout if fanout is not None else speculative.MAX_FANOUT,
            "required_replicas": required_replicas,
            "fixed_fanout": fanout is not None,
        },
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": digest,
        "completed_rows_unchanged": True,
    })


def benchmark_speculative(run_id: str, output: Path | None = None) -> dict[str, Any]:
    root = _run_root(run_id)
    path = output or root / "speculative_benchmark_dgx008_first8_v1/manifest.json"
    connection = core._request_database(root / "requests.sqlite3", journal_mode="DELETE")
    try:
        return asyncio.run(speculative.benchmark(connection, path))
    finally:
        connection.close()


def _execute_pending_round(
    connection: sqlite3.Connection, requests: Sequence[str], round_index: int,
    use_openrouter: bool, snapshot: Mapping[str, Any] | None,
    pools: dict[str, tuple[Any, list[dict[str, Any]], int]],
    manifest: dict[str, Any], root: Path, parallelism: int,
    benchmark_path: Path | None, required_replicas: int,
    speculative_fanout: int | None,
) -> None:
    if snapshot is not None:
        _record_provider_snapshot(manifest, snapshot)
    if round_index > 0 and not use_openrouter:
        if benchmark_path is None:
            raise ValueError("speculative execution requires its benchmark")
        receipt = asyncio.run(speculative.execute_pending(
            connection, requests, benchmark_path,
            required=required_replicas, fanout=speculative_fanout,
        ))
        manifest.setdefault("speculative_round_receipts", []).append({
            "round": round_index, **receipt,
        })
        return
    stage = _configure_execution(round_index, parallelism)
    placeholders = ",".join("?" for _ in requests)
    prior_attempts = int(connection.execute(
        f"SELECT COALESCE(MAX(attempts),0) FROM requests "
        f"WHERE request_id IN ({placeholders})", list(requests),
    ).fetchone()[0])
    core.MAX_ATTEMPTS = prior_attempts + 4
    client, _, total_parallelism = _completion_pool(
        stage, pools, manifest, root, parallelism
    )
    core._run_pending(
        connection, requests, parallelism=total_parallelism, client=client,
        request_slots=threading.BoundedSemaphore(total_parallelism),
    )


async def _execute_task_level_chain(
    connection: sqlite3.Connection, rows: pd.DataFrame,
    payloads: Mapping[str, Mapping[str, Any]], schedule: pd.DataFrame,
    manifest: dict[str, Any], root: Path, client: Any,
    endpoint: Mapping[str, Any], benchmark_sha256: str, selection: Mapping[str, Any],
    slots: asyncio.Semaphore, required: int, fanout: int, stop_after_round: int | None,
) -> None:
    task, level = str(rows.iloc[0].task), str(rows.iloc[0].level)
    scores = _completed_scores(connection)
    for round_index, round_rows in rows.groupby("round", sort=True):
        if int(round_index) == 0:
            continue
        candidates = [bucket for value in round_rows.candidate_bucket_ids_json
                      for bucket in json.loads(value)]
        if set(candidates) <= scores.keys():
            continue
        requests = _queue_round(
            connection, round_rows, payloads, scores, None, "local-speculative"
        )
        pending = speculative._pending_rows(connection, requests)
        if not pending:
            continue
        failures = []

        async def execute(row: Mapping[str, Any]) -> dict[str, Any]:
            async with slots:
                return await speculative.execute_request(
                    client, row, model=endpoint["model"], initial_fanout=fanout,
                    hedge_seconds=float(selection["hedge_seconds"]), maximum=fanout,
                    required=required,
                )

        tasks = [asyncio.create_task(execute(row)) for row in pending]
        try:
            for future in asyncio.as_completed(tasks):
                result = await future
                speculative.persist_result(
                    connection, result, base_url=endpoint["base_url"],
                    model=endpoint["model"], benchmark_sha256=benchmark_sha256,
                )
                if result["status"] != "complete":
                    failures.append(result)
        finally:
            for future in tasks:
                if not future.done():
                    future.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if failures:
            raise RuntimeError(f"speculative requests failed for {task}/{level}")
        scores = _checkpoint_completed(root, manifest, connection, schedule)
        manifest.setdefault("speculative_round_receipts", []).append({
            "task": task, "level": level, "round": int(round_index),
            "completed": len(pending), "execution_strategy": "independent_task_level",
        })
        write_json_atomic(root / "manifest.json", manifest)
        if stop_after_round is not None and int(round_index) >= stop_after_round:
            break


async def _execute_task_level_chains(
    connection: sqlite3.Connection, schedule: pd.DataFrame,
    payloads: Mapping[str, Mapping[str, Any]], manifest: dict[str, Any], root: Path,
    client: Any, endpoint: Mapping[str, Any], benchmark_sha256: str,
    selection: Mapping[str, Any], required: int, fanout: int,
    stop_after_round: int | None,
) -> None:
    slots = asyncio.Semaphore(speculative.MAX_ROUND_REQUESTS)
    async with asyncio.TaskGroup() as group:
        for _, rows in schedule.groupby(["task", "level"], sort=True):
            group.create_task(_execute_task_level_chain(
                connection, rows, payloads, schedule, manifest, root, client,
                endpoint, benchmark_sha256, selection, slots, required, fanout,
                stop_after_round,
            ))


async def _execute_independent_task_levels(
    connection: sqlite3.Connection, schedule: pd.DataFrame,
    payloads: Mapping[str, Mapping[str, Any]], manifest: dict[str, Any], root: Path,
    benchmark_path: Path, required: int, requested_fanout: int | None,
    stop_after_round: int | None,
) -> None:
    benchmark, benchmark_sha256 = speculative.load_benchmark(benchmark_path)
    endpoint, selection = benchmark["endpoint"], benchmark["selection"]
    fanout = int(requested_fanout or selection["fanout"])
    if not 1 <= required <= fanout <= speculative.MAX_FANOUT:
        raise ValueError("speculative execution requires 1 <= required <= fanout <= 128")
    speculative.ensure_tables(connection)
    await speculative.wait_for_drain(endpoint["base_url"])
    client, credential = async_openai_compatible_client(
        base_url=endpoint["base_url"], provider="local", env_file=None,
        max_connections=speculative.MAX_ROUND_REQUESTS * fanout,
        timeout_s=300, max_retries=0,
    )
    if credential:
        raise ValueError("local speculative execution unexpectedly selected a credential")
    try:
        models = sorted(model.id for model in (await client.models.list()).data)
        if models != [endpoint["model"]]:
            raise ValueError(f"local endpoint model contract changed: {models}")
        await _execute_task_level_chains(
            connection, schedule, payloads, manifest, root, client, endpoint,
            benchmark_sha256, selection, required, fanout, stop_after_round,
        )
    finally:
        await client.close()
    await speculative.wait_for_drain(endpoint["base_url"])


def _execute_barrier_rounds(
    connection: sqlite3.Connection, schedule: pd.DataFrame,
    payloads: Mapping[str, Mapping[str, Any]], manifest: dict[str, Any], root: Path,
    parallelism: int, anchored_execution: str, benchmark_path: Path | None,
    required_replicas: int, speculative_fanout: int | None,
    stop_after_round: int | None,
) -> bool:
    scores = _completed_scores(connection)
    pools: dict[str, tuple[Any, list[dict[str, Any]], int]] = {}
    round_indices = sorted(schedule["round"].unique())
    if anchored_execution == "local-speculative":
        round_indices = [index for index in round_indices if int(index) == 0]
    for round_index in round_indices:
        rows = schedule[schedule["round"].eq(round_index)]
        use_openrouter = int(round_index) == 0 or anchored_execution == "openrouter"
        snapshot = (provider_pool.load_ranked_pool(True)
                    if int(round_index) > 0 and use_openrouter else None)
        requests = _queue_round(
            connection, rows, payloads, scores, snapshot, anchored_execution
        )
        if _pending_request_count(connection, requests):
            _execute_pending_round(
                connection, requests, int(round_index), use_openrouter, snapshot,
                pools, manifest, root, parallelism, benchmark_path,
                required_replicas, speculative_fanout,
            )
            scores = _checkpoint_completed(root, manifest, connection, schedule)
        if stop_after_round is not None and int(round_index) >= stop_after_round:
            return True
    return False


def run(
    run_id: str, approved_review_sha256: str, parallelism: int = 27, *,
    anchored_execution: str = "openrouter", speculative_benchmark_manifest: Path | None = None,
    stop_after_round: int | None = None, speculative_required_replicas: int = speculative.REQUIRED_REPLICAS,
    speculative_fanout: int | None = None,
) -> dict[str, Any]:
    if parallelism < 1:
        raise ValueError("parallelism must be positive")
    if anchored_execution not in {"openrouter", "local-speculative"}:
        raise ValueError("unknown anchored execution mode")
    root = _run_root(run_id)
    manifest, resolved = _verify_prepared(root, approved_review_sha256)
    schedule = pd.read_parquet(root / "schedule.parquet")
    payloads = _all_bucket_payloads(resolved)
    connection = core._request_database(root / "requests.sqlite3", journal_mode="DELETE")
    boundary = None
    if anchored_execution == "local-speculative":
        if speculative_benchmark_manifest is None:
            raise ValueError("local speculative execution requires a benchmark manifest")
        boundary = _validate_speculative_cutover(connection, manifest)
        _record_speculative_migration(
            manifest, speculative_benchmark_manifest, boundary,
            speculative_required_replicas, speculative_fanout)
    manifest.update(status="running")
    if anchored_execution == "local-speculative":
        manifest.setdefault("task_level_execution", {
            "strategy": "independent_task_level_chains",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "maximum_concurrent_scoring_requests": speculative.MAX_ROUND_REQUESTS,
            "completed_scores_preserved": True,
        })
    _record_input_bindings(manifest, root)
    write_json_atomic(root / "manifest.json", manifest)
    try:
        paused = _execute_barrier_rounds(
            connection, schedule, payloads, manifest, root, parallelism,
            anchored_execution, speculative_benchmark_manifest,
            speculative_required_replicas, speculative_fanout, stop_after_round,
        )
        if anchored_execution == "local-speculative" and not paused:
            asyncio.run(_execute_independent_task_levels(
                connection, schedule, payloads, manifest, root,
                speculative_benchmark_manifest, speculative_required_replicas,
                speculative_fanout, stop_after_round,
            ))
            paused = stop_after_round is not None
    except Exception:
        manifest["status"] = "incomplete"
        write_json_atomic(root / "manifest.json", manifest)
        raise
    finally:
        connection.close()
    manifest.update(status="paused_at_round_boundary" if paused else "complete_unpublished")
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def publish_candidate(run_id: str) -> dict[str, Any]:
    root = _run_root(run_id)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    schedule = pd.read_parquet(root / "schedule.parquet")
    connection = core._request_database(root / "requests.sqlite3", journal_mode="DELETE")
    try:
        frame = _score_frame(_completed_scores(connection), schedule)
    finally:
        connection.close()
    expected = _scheduled_bucket_count(schedule)
    if len(frame) != expected or frame.semantic_bucket_id.duplicated().any():
        raise ValueError(f"candidate is incomplete: {len(frame)}/{expected} buckets")
    _atomic_parquet(frame, root / "weights.parquet")
    _atomic_tsv(frame, root / "weights.tsv")
    summary = frame.groupby(["task", "level"]).weight.agg(["count", "min", "mean", "max"]).reset_index()
    _atomic_tsv(summary, root / "summary.tsv")
    report = _report(summary, expected)
    (root / "report.md").write_text(report, encoding="utf-8")
    manifest.update(
        status="complete_candidate", completed_bucket_count=len(frame),
        active_policy_changed=False,
        artifacts={name: _sha256(root / name) for name in
                   ("weights.parquet", "weights.tsv", "summary.tsv", "report.md")},
    )
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def _report(summary: pd.DataFrame, expected: int) -> str:
    lines = [
        "# Sliding semantic-bucket weight candidate", "",
        f"DeepSeek-v4-pro seeded each task-level and DeepSeek V4 Flash assigned the "
        f"anchored follow-up weights for all {expected:,} scheduled semantic buckets.",
        "This candidate does not modify the active semantic-weight policy.", "",
        "| Task | Level | Buckets | Minimum | Mean | Maximum |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.task} | {row.level} | {row.count} | {row.min:.2f} | "
            f"{row.mean:.4f} | {row.max:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "benchmark-speculative", "run", "publish-candidate"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--approved-review-sha256")
    parser.add_argument("--parallelism", type=int, default=27)
    parser.add_argument(
        "--anchored-execution",
        choices=("openrouter", "local-speculative"),
        default="openrouter",
    )
    parser.add_argument("--speculative-benchmark-manifest", type=Path)
    parser.add_argument("--speculative-required-replicas", type=int, default=8)
    parser.add_argument("--speculative-fanout", type=int)
    parser.add_argument("--stop-after-round", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.run_id)
    elif args.command == "benchmark-speculative":
        result = benchmark_speculative(
            args.run_id, args.speculative_benchmark_manifest
        )
    elif args.command == "run":
        if not args.approved_review_sha256:
            parser.error("run requires --approved-review-sha256")
        result = run(
            args.run_id, args.approved_review_sha256, args.parallelism,
            anchored_execution=args.anchored_execution,
            speculative_benchmark_manifest=args.speculative_benchmark_manifest,
            stop_after_round=args.stop_after_round,
            speculative_required_replicas=args.speculative_required_replicas,
            speculative_fanout=args.speculative_fanout,
        )
    else:
        result = publish_candidate(args.run_id)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
