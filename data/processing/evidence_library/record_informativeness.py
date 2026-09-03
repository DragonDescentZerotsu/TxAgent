"""Score accepted V7 Stage-1 evidence rows with one request per paper molecule."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import heapq
import json
import math
import sqlite3
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from openai import APIStatusError, AsyncOpenAI

from data.processing.llm_api import DEFAULT_ENV_FILE, resolve_api_key
from data.processing.evidence_library.shared.v1.normalization.cleaning import (
    file_sha256,
)
from tools.chembl_tool.common.json_utils import atomic_output_path, write_json_atomic


VERSION = "record_informativeness.v1"
PROMPT_VERSION = "record_informativeness_prompt.v1"
DEFAULT_BASE_URL = "http://epyc-3-6:50000/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEFAULT_CONCURRENCY = 1_024
MAX_CONCURRENCY = 1_024
MAX_COMPLETION_TOKENS = 512 * 1_024
TEMPLATE_PATH = Path(__file__).with_name("prompts") / "record_informativeness_v1.jinja"
GROUP_COLUMNS = ["source_id", "pmid", "canonical_smiles"]
SCORE_COLUMNS = [
    "relevance_score",
    "completeness_score",
    "informativeness_score",
]
CONTEXT_VALUES = {"required", "helpful", "none"}
TASKS = {
    "bbb_martins": {
        "target": "Experimentally meaningful systemic CNS access.",
        "rows": 498_640,
        "groups": 319_358,
        "sha256": "6e3781b264e79207b43c32a0b6a6b501bbde542a8950f6970db67534a7e304fd",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-5.6-luna",
        "reasoning_effort": "low",
        "token_budget": 40_000_000,
    },
    "bioavailability_ma": {
        "target": "Oral bioavailability under the reported condition.",
        "rows": 435_603,
        "groups": 134_402,
        "sha256": "f63561025fe74e0e6eb17fb251b48a02b7186db47a748682f1669625b3598446",
        "base_url": DEFAULT_BASE_URL,
        "model": DEFAULT_MODEL,
        "reasoning_effort": "",
        "token_budget": None,
    },
    "skin_reaction": {
        "target": "Skin sensitization or contact allergy.",
        "rows": 550_192,
        "groups": 152_696,
        "sha256": "7d1f8b3ece3caff027b715ce4047e4abbaa10fa0aaf807d2ba6436bc99cf32ec",
        "base_url": DEFAULT_BASE_URL,
        "model": DEFAULT_MODEL,
        "reasoning_effort": "",
        "token_budget": None,
    },
}
HIDDEN_FIELDS = {
    "canonical_endpoint_name",
    "canonical_smiles",
    "cleaned_record_id",
    "confidence",
    "deduplication_context_id",
    "embedded_unit",
    "extraction_id",
    "global_identifier",
    "measurement_resolution_exact_measurement",
    "measurement_resolution_exact_unit",
    "measurement_resolution_exact_unit_is_canonical",
    "measurement_resolution_route",
    "measurement_resolution_rule_id",
    "measurement_routing_version",
    "needs_more_context",
    "paragraph_idx",
    "pmid",
    "smiles",
    "source_id",
    "source_index",
    "source_name",
    "source_record_id",
    "source_row_number",
    "source_smiles",
    "structure_status",
}
OUTPUT_COLUMNS = [
    "task_id",
    "source_id",
    "pmid",
    "canonical_smiles",
    "cleaned_record_id",
    *SCORE_COLUMNS,
    "context_requirement",
    "request_id",
    "payload_sha256",
    "prompt_version",
    "requested_model",
    "served_model",
]


def _clean(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def visible_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: cleaned
        for field, value in record.items()
        if field not in HIDDEN_FIELDS and (cleaned := _clean(value)) is not None
    }


@lru_cache(maxsize=1)
def _template() -> Any:
    environment = Environment(
        loader=FileSystemLoader(str(TEMPLATE_PATH.parent)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    return environment.get_template(TEMPLATE_PATH.name)


def render_group(group: Mapping[str, Any], target_definition: str) -> str:
    rows = [
        {
            "id": row["id"],
            "payload_json": json.dumps(
                row["payload"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        }
        for row in group["rows"]
    ]
    return _template().render(
        target_definition=target_definition,
        dataset=group["task_id"],
        source_id=group["source_id"],
        pmid=group["pmid"],
        canonical_smiles=group["canonical_smiles"],
        rows=rows,
    )


def _load_frame(path: Path, task_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"accepted V7 Stage-1 records are missing: {path}")
    expected = TASKS[task_id]
    digest = file_sha256(path)
    if digest != expected["sha256"]:
        raise ValueError(
            f"unexpected {task_id} V7 Stage-1 hash: {digest}; "
            f"expected {expected['sha256']}"
        )
    frame = pd.read_parquet(path)
    required = {*GROUP_COLUMNS, "cleaned_record_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Stage-1 input lacks required columns: {missing}")
    if len(frame) != expected["rows"]:
        raise ValueError(
            f"unexpected {task_id} row count: {len(frame)}; expected {expected['rows']}"
        )
    for column in required:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Stage-1 input has missing {column}")
    if frame["cleaned_record_id"].duplicated().any():
        raise ValueError("Stage-1 input has duplicate cleaned_record_id values")
    frame.sort_values([*GROUP_COLUMNS, "cleaned_record_id"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    sizes = frame.groupby(GROUP_COLUMNS, sort=False, observed=True).size()
    if len(sizes) != expected["groups"]:
        raise ValueError(
            f"unexpected {task_id} group count: {len(sizes)}; "
            f"expected {expected['groups']}"
        )
    inventory = {
        "task_id": task_id,
        "input_path": str(path.resolve()),
        "input_sha256": digest,
        "rows": int(len(frame)),
        "groups": int(len(sizes)),
        "papers": int(frame["pmid"].nunique()),
        "group_size": {
            "minimum": int(sizes.min()),
            "median": float(sizes.quantile(0.5)),
            "p90": float(sizes.quantile(0.9)),
            "p95": float(sizes.quantile(0.95)),
            "p99": float(sizes.quantile(0.99)),
            "maximum": int(sizes.max()),
        },
        "rows_by_source": {
            str(key): int(value)
            for key, value in frame["source_id"].value_counts().sort_index().items()
        },
        "groups_by_source": {
            str(key): int(value)
            for key, value in sizes.groupby(level="source_id").size().sort_index().items()
        },
    }
    return frame, inventory


def iter_groups(frame: pd.DataFrame, task_id: str) -> Iterator[dict[str, Any]]:
    prompt_sha256 = file_sha256(TEMPLATE_PATH)
    for (source_id, pmid, canonical_smiles), rows in frame.groupby(
        GROUP_COLUMNS, sort=False, observed=True
    ):
        payload_rows = []
        for local_id, record in enumerate(rows.to_dict(orient="records")):
            payload_rows.append(
                {
                    "id": str(local_id),
                    "cleaned_record_id": str(record["cleaned_record_id"]),
                    "payload": visible_record(record),
                }
            )
        identity = {
            "task_id": task_id,
            "source_id": str(source_id),
            "pmid": str(pmid),
            "canonical_smiles": str(canonical_smiles),
        }
        payload_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "prompt_version": PROMPT_VERSION,
                    "prompt_sha256": prompt_sha256,
                    "target_definition": TASKS[task_id]["target"],
                    "identity": identity,
                    "rows": payload_rows,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        request_id = hashlib.sha256(
            json.dumps(
                {"identity": identity, "payload_sha256": payload_sha256},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        yield {
            **identity,
            "rows": payload_rows,
            "payload_sha256": payload_sha256,
            "request_id": request_id,
        }


def _size_band(size: int) -> str:
    if size == 1:
        return "1"
    if size <= 5:
        return "2-5"
    if size <= 20:
        return "6-20"
    return "21+"


def select_pilot(
    frame: pd.DataFrame, task_id: str, size: int
) -> list[dict[str, Any]]:
    if size <= 0:
        raise ValueError("pilot size must be positive")
    best_by_source: dict[str, tuple[int, tuple[str, str, str]]] = {}
    best_by_band: dict[str, tuple[int, tuple[str, str, str]]] = {}
    ranked: list[tuple[int, tuple[str, str, str]]] = []
    largest: tuple[int, tuple[str, str, str]] | None = None
    sizes = frame.groupby(GROUP_COLUMNS, sort=False, observed=True).size()
    for raw_key, group_size in sizes.items():
        key = tuple(str(value) for value in raw_key)
        rank = int(
            hashlib.sha256(
                json.dumps((task_id, key), separators=(",", ":")).encode()
            ).hexdigest(),
            16,
        )
        source = key[0]
        band = _size_band(int(group_size))
        if source not in best_by_source or rank < best_by_source[source][0]:
            best_by_source[source] = (rank, key)
        if band not in best_by_band or rank < best_by_band[band][0]:
            best_by_band[band] = (rank, key)
        candidate = (int(group_size), key)
        if largest is None or candidate > largest:
            largest = candidate
        ranked_item = (-rank, key)
        if len(ranked) < size:
            heapq.heappush(ranked, ranked_item)
        elif rank < -ranked[0][0]:
            heapq.heapreplace(ranked, ranked_item)
    preferred = [best_by_source[key][1] for key in sorted(best_by_source)]
    preferred.extend(best_by_band[key][1] for key in sorted(best_by_band))
    if largest is not None:
        preferred.append(largest[1])
    preferred.extend(key for _, key in sorted(ranked, key=lambda item: -item[0]))
    selected: list[tuple[str, str, str]] = []
    seen = set()
    for key in preferred:
        if key in seen:
            continue
        seen.add(key)
        selected.append(key)
        if len(selected) >= size:
            break
    row_keys = pd.MultiIndex.from_frame(frame[GROUP_COLUMNS])
    subset = frame[row_keys.isin(selected)]
    return list(iter_groups(subset, task_id))


def response_format(ids: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "record_informativeness",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["items"],
                "properties": {
                    "items": {
                        "type": "array",
                        "minItems": len(ids),
                        "maxItems": len(ids),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "id",
                                *SCORE_COLUMNS,
                                "context_requirement",
                            ],
                            "properties": {
                                "id": {"type": "string", "enum": list(ids)},
                                **{
                                    name: {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    }
                                    for name in SCORE_COLUMNS
                                },
                                "context_requirement": {
                                    "type": "string",
                                    "enum": sorted(CONTEXT_VALUES),
                                },
                            },
                        },
                    }
                },
            },
        },
    }


def validate_response(response: Any, ids: Sequence[str]) -> list[dict[str, Any]]:
    if not isinstance(response, Mapping) or set(response) != {"items"}:
        raise ValueError("response must contain only items")
    items = response["items"]
    if not isinstance(items, list) or len(items) != len(ids):
        raise ValueError("response has the wrong item count")
    expected = set(ids)
    validated = {}
    required = {"id", *SCORE_COLUMNS, "context_requirement"}
    for item in items:
        if not isinstance(item, Mapping) or set(item) != required:
            raise ValueError("response item fields differ from the contract")
        item_id = item["id"]
        if not isinstance(item_id, str) or item_id not in expected or item_id in validated:
            raise ValueError("response contains an invalid or duplicate item ID")
        output = {"id": item_id}
        for name in SCORE_COLUMNS:
            score = item[name]
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0 <= float(score) <= 1
            ):
                raise ValueError(f"{name} must be finite numeric in [0, 1]")
            output[name] = float(score)
        context = item["context_requirement"]
        if context not in CONTEXT_VALUES:
            raise ValueError("invalid context_requirement")
        output["context_requirement"] = context
        validated[item_id] = output
    return [validated[item_id] for item_id in ids]


async def preflight(base_url: str, model: str) -> dict[str, Any]:
    if base_url.startswith("https://openrouter.ai/"):
        return {"provider": "openrouter", "model": model}
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    async with httpx.AsyncClient(timeout=20) as client:
        health = await client.get(f"{root}/health")
        health.raise_for_status()
        models_response = await client.get(f"{root}/v1/models")
        models_response.raise_for_status()
        model_ids = {row["id"] for row in models_response.json().get("data", [])}
        if model_ids != {model}:
            raise RuntimeError(f"served models differ from the pinned model: {model_ids}")
        workers_response = await client.get(f"{root}/workers")
        workers_response.raise_for_status()
        workers = workers_response.json().get("workers", [])
    if not workers or any(
        worker.get("is_healthy") is not True or worker.get("model_id") != model
        for worker in workers
    ):
        raise RuntimeError("router does not report a complete healthy worker pool")
    return {
        "provider": "local_router",
        "health": health.text.strip(),
        "model_ids": sorted(model_ids),
        "worker_count": len(workers),
        "aggregate_load": sum(int(worker.get("load") or 0) for worker in workers),
    }


def _completion_parameters(
    model: str, *, max_tokens: int, reasoning_effort: str
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        (
            "max_completion_tokens"
            if model.rsplit("/", 1)[-1].startswith("gpt-5")
            else "max_tokens"
        ): max_tokens
    }
    if reasoning_effort:
        parameters["reasoning_effort"] = reasoning_effort
    else:
        parameters["temperature"] = 0
    if reasoning_effort:
        parameters["extra_body"] = {"provider": {"require_parameters": True}}
    return parameters


def _usage(completion: Any) -> dict[str, int] | None:
    if completion is None or completion.usage is None:
        return None
    raw = completion.usage.model_dump()
    return {
        "input_tokens": int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0),
        "output_tokens": int(
            raw.get("completion_tokens") or raw.get("output_tokens") or 0
        ),
    }


async def canary(
    client: AsyncOpenAI, model: str, *, reasoning_effort: str
) -> Any:
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Return one item for row 0. Use scores 0.5 and context none. "
                    "Return only the requested JSON."
                ),
            }
        ],
        response_format=response_format(["0"]),
        **_completion_parameters(
            model,
            max_tokens=MAX_COMPLETION_TOKENS,
            reasoning_effort=reasoning_effort,
        ),
    )
    if completion.model != model:
        raise RuntimeError(f"canary was served by {completion.model!r}, expected {model!r}")
    content = completion.choices[0].message.content
    validate_response(json.loads(content or ""), ["0"])
    return completion


async def query_group(
    client: AsyncOpenAI,
    group: Mapping[str, Any],
    *,
    model: str,
    reasoning_effort: str,
    max_attempts: int,
    connection: sqlite3.Connection,
    budget_lock: asyncio.Lock,
    token_budget: int | None,
    budget_exhausted: asyncio.Event,
) -> dict[str, Any]:
    ids = [row["id"] for row in group["rows"]]
    errors = []
    started_at = time.time()
    prompt = render_group(group, TASKS[group["task_id"]]["target"])
    completion_limit = MAX_COMPLETION_TOKENS
    exhausted = False
    for attempt in range(1, max_attempts + 1):
        reservation_id = f"{group['request_id']}:{attempt}"
        if token_budget is not None:
            budget_status = await _reserve_budget(
                connection,
                budget_lock,
                reservation_id=reservation_id,
            )
            if budget_status == "exhausted":
                budget_exhausted.set()
                errors.append("token budget exhausted before submission")
                exhausted = True
            if exhausted:
                break
        completion = None
        release_reason = None
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                response_format=response_format(ids),
                **_completion_parameters(
                    model,
                    max_tokens=completion_limit,
                    reasoning_effort=reasoning_effort,
                ),
            )
            if completion.model != model:
                raise RuntimeError(
                    f"served model {completion.model!r} differs from {model!r}"
                )
            message = completion.choices[0].message
            content = message.content or ""
            scores = validate_response(json.loads(content), ids)
            return {
                **{key: group[key] for key in (*GROUP_COLUMNS, "task_id")},
                "request_id": group["request_id"],
                "payload_sha256": group["payload_sha256"],
                "status": "success",
                "attempts": attempt,
                "served_model": completion.model,
                "response_json": content,
                "reasoning_content": getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning", None),
                "usage_json": json.dumps(
                    completion.usage.model_dump() if completion.usage else {},
                    sort_keys=True,
                ),
                "errors_json": json.dumps(errors, ensure_ascii=False),
                "scores": [
                    {
                        **score,
                        "cleaned_record_id": group["rows"][int(score["id"])][
                            "cleaned_record_id"
                        ],
                    }
                    for score in scores
                ],
                "started_at": started_at,
                "updated_at": time.time(),
            }
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            if isinstance(exc, APIStatusError) and 400 <= exc.status_code < 500:
                release_reason = f"http_{exc.status_code}"
            if attempt < max_attempts:
                await asyncio.sleep(min(30, 2 ** (attempt - 1)))
        finally:
            if token_budget is not None:
                await _complete_budget(
                    connection,
                    budget_lock,
                    reservation_id=reservation_id,
                    usage=_usage(completion),
                    release_reason=release_reason,
                )
    return {
        **{key: group[key] for key in (*GROUP_COLUMNS, "task_id")},
        "request_id": group["request_id"],
        "payload_sha256": group["payload_sha256"],
        "status": "failed",
        "attempts": len([error for error in errors if error.startswith("attempt ")]),
        "served_model": None,
        "response_json": None,
        "reasoning_content": None,
        "usage_json": None,
        "errors_json": json.dumps(errors, ensure_ascii=False),
        "scores": [],
        "started_at": started_at,
        "updated_at": time.time(),
    }


def _connect_database(
    path: Path, *, metadata: Mapping[str, Any], resume: bool
) -> sqlite3.Connection:
    existed = path.is_file() and path.stat().st_size > 0
    if existed and not resume:
        raise FileExistsError(f"request database already exists; pass --resume: {path}")
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS requests (
            request_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            pmid TEXT NOT NULL,
            canonical_smiles TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            requested_model TEXT NOT NULL,
            served_model TEXT,
            response_json TEXT,
            reasoning_content TEXT,
            usage_json TEXT,
            errors_json TEXT NOT NULL,
            started_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scores (
            cleaned_record_id TEXT PRIMARY KEY,
            request_id TEXT NOT NULL,
            local_id TEXT NOT NULL,
            relevance_score REAL NOT NULL,
            completeness_score REAL NOT NULL,
            informativeness_score REAL NOT NULL,
            context_requirement TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS scores_request_id ON scores(request_id);
        CREATE TABLE IF NOT EXISTS budget (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            max_tokens INTEGER NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            conservative_unreported_tokens INTEGER NOT NULL DEFAULT 0,
            requests_completed INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS budget_reservations (
            reservation_id TEXT PRIMARY KEY,
            maximum INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS budget_releases (
            reservation_id TEXT PRIMARY KEY,
            reason TEXT NOT NULL,
            maximum INTEGER NOT NULL,
            updated_at REAL NOT NULL
        );
        """
    )
    prior = {
        key: json.loads(value)
        for key, value in connection.execute("SELECT key, value_json FROM metadata")
    }
    if prior and prior != dict(metadata):
        differing = sorted(
            key
            for key in set(prior) | set(metadata)
            if prior.get(key) != metadata.get(key)
        )
        connection.close()
        raise ValueError(f"request database metadata drift: {differing}")
    if not prior:
        connection.executemany(
            "INSERT INTO metadata(key, value_json) VALUES (?, ?)",
            [(key, json.dumps(value, sort_keys=True)) for key, value in metadata.items()],
        )
        connection.commit()
    return connection


def _initialize_budget(connection: sqlite3.Connection, maximum: int | None) -> None:
    row = connection.execute("SELECT max_tokens FROM budget WHERE id = 1").fetchone()
    if maximum is None:
        if row is not None:
            raise ValueError("an unmetered run cannot resume a metered request database")
        return
    if row is None:
        connection.execute(
            "INSERT INTO budget(id, max_tokens) VALUES (1, ?)", (maximum,)
        )
    elif int(row[0]) != maximum:
        raise ValueError("token budget differs from the existing request database")
    connection.execute("DELETE FROM budget_reservations")
    connection.commit()


async def _reserve_budget(
    connection: sqlite3.Connection,
    lock: asyncio.Lock,
    *,
    reservation_id: str,
) -> str:
    async with lock:
        spent = int(
            connection.execute(
                "SELECT input_tokens + output_tokens FROM budget WHERE id = 1"
            ).fetchone()[0]
        )
        limit = int(
            connection.execute("SELECT max_tokens FROM budget WHERE id = 1").fetchone()[0]
        )
        if spent >= limit:
            return "exhausted"
        connection.execute(
            "INSERT INTO budget_reservations VALUES (?, ?)",
            (reservation_id, 0),
        )
        connection.commit()
        return "reserved"


async def _complete_budget(
    connection: sqlite3.Connection,
    lock: asyncio.Lock,
    *,
    reservation_id: str,
    usage: Mapping[str, int] | None,
    release_reason: str | None = None,
) -> None:
    async with lock:
        row = connection.execute(
            "SELECT maximum FROM budget_reservations WHERE reservation_id = ?",
            (reservation_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"missing token reservation {reservation_id}")
        reservation = int(row[0])
        if release_reason or usage is None:
            connection.execute(
                "INSERT INTO budget_releases VALUES (?, ?, ?, ?)",
                (
                    reservation_id,
                    release_reason or "unreported_usage",
                    reservation,
                    time.time(),
                ),
            )
        else:
            connection.execute(
                "UPDATE budget SET input_tokens = input_tokens + ?, "
                "output_tokens = output_tokens + ?, requests_completed = "
                "requests_completed + 1 WHERE id = 1",
                (usage["input_tokens"], usage["output_tokens"]),
            )
        connection.execute(
            "DELETE FROM budget_reservations WHERE reservation_id = ?",
            (reservation_id,),
        )
        connection.commit()


def _budget_summary(connection: sqlite3.Connection) -> dict[str, int] | None:
    row = connection.execute(
        "SELECT max_tokens, input_tokens, output_tokens, "
        "conservative_unreported_tokens, requests_completed FROM budget WHERE id = 1"
    ).fetchone()
    if row is None:
        return None
    result = {
        "max_tokens": int(row[0]),
        "input_tokens": int(row[1]),
        "output_tokens": int(row[2]),
        "conservative_unreported_tokens": int(row[3]),
        "requests_completed": int(row[4]),
    }
    result["spent_tokens"] = result["input_tokens"] + result["output_tokens"]
    result["released_requests"] = int(
        connection.execute("SELECT COUNT(*) FROM budget_releases").fetchone()[0]
    )
    return result


def _completed_requests(connection: sqlite3.Connection) -> dict[str, tuple[str, int]]:
    return {
        request_id: (payload_sha256, int(score_count))
        for request_id, payload_sha256, score_count in connection.execute(
            "SELECT r.request_id, r.payload_sha256, COUNT(s.cleaned_record_id) "
            "FROM requests r LEFT JOIN scores s USING (request_id) "
            "WHERE r.status = 'success' GROUP BY r.request_id, r.payload_sha256"
        )
    }


async def _write_results(
    queue: asyncio.Queue[Any],
    connection: sqlite3.Connection,
    *,
    requested_model: str,
    progress: dict[str, int],
) -> None:
    last_report = time.monotonic()
    last_count = 0
    pending_commit = 0
    try:
        while True:
            event = await queue.get()
            if event is None:
                queue.task_done()
                break
            connection.execute(
                """
                INSERT OR REPLACE INTO requests VALUES (
                    :request_id, :task_id, :source_id, :pmid, :canonical_smiles,
                    :payload_sha256, :status, :attempts, :requested_model,
                    :served_model, :response_json, :reasoning_content, :usage_json,
                    :errors_json, :started_at, :updated_at
                )
                """,
                {**event, "requested_model": requested_model},
            )
            connection.execute(
                "DELETE FROM scores WHERE request_id = ?", (event["request_id"],)
            )
            if event["status"] == "success":
                connection.executemany(
                    """
                    INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            score["cleaned_record_id"],
                            event["request_id"],
                            score["id"],
                            score["relevance_score"],
                            score["completeness_score"],
                            score["informativeness_score"],
                            score["context_requirement"],
                        )
                        for score in event["scores"]
                    ],
                )
                progress["success"] += 1
            else:
                progress["failed"] += 1
            progress["finished"] += 1
            pending_commit += 1
            now = time.monotonic()
            if pending_commit >= 100 or now - last_report >= 10:
                connection.commit()
                pending_commit = 0
            if now - last_report >= 10:
                rate = (progress["finished"] - last_count) / (now - last_report)
                print(
                    f"finished={progress['finished']:,}/{progress['pending']:,} "
                    f"success={progress['success']:,} failed={progress['failed']:,} "
                    f"rate={rate:.1f} requests/s",
                    flush=True,
                )
                last_report, last_count = now, progress["finished"]
            queue.task_done()
    finally:
        connection.commit()


async def _run_requests(
    groups: Iterator[dict[str, Any]],
    *,
    connection: sqlite3.Connection,
    model: str,
    base_url: str,
    concurrency: int,
    timeout_s: float,
    max_attempts: int,
    total_groups: int,
    reasoning_effort: str,
    api_key: str,
    token_budget: int | None,
) -> tuple[dict[str, int], dict[str, Any], bool]:
    endpoint = await preflight(base_url, model)
    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency
    )
    http_client = httpx.AsyncClient(limits=limits, timeout=timeout_s)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        http_client=http_client,
        max_retries=0,
    )
    completed = _completed_requests(connection)
    progress = {
        "pending": total_groups - len(completed),
        "finished": 0,
        "success": 0,
        "failed": 0,
    }
    result_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=concurrency * 2)
    iterator_lock = asyncio.Lock()
    budget_lock = asyncio.Lock()
    budget_exhausted = asyncio.Event()
    yielded = 0

    async def next_group() -> dict[str, Any] | None:
        nonlocal yielded
        async with iterator_lock:
            if budget_exhausted.is_set():
                return None
            for group in groups:
                cached = completed.get(group["request_id"])
                if cached is not None:
                    if cached != (group["payload_sha256"], len(group["rows"])):
                        raise ValueError(
                            f"cached payload drift for {group['request_id']}"
                        )
                    continue
                yielded += 1
                return group
            return None

    async def worker() -> None:
        while True:
            group = await next_group()
            if group is None:
                return
            event = await query_group(
                client,
                group,
                model=model,
                reasoning_effort=reasoning_effort,
                max_attempts=max_attempts,
                connection=connection,
                budget_lock=budget_lock,
                token_budget=token_budget,
                budget_exhausted=budget_exhausted,
            )
            if event:
                await result_queue.put(event)

    writer = asyncio.create_task(
        _write_results(
            result_queue,
            connection,
            requested_model=model,
            progress=progress,
        )
    )
    workers: list[asyncio.Task[Any]] = []
    try:
        if not completed:
            canary_reservation_id = "__strict_schema_canary__"
            if token_budget is not None:
                canary_budget_status = await _reserve_budget(
                    connection,
                    budget_lock,
                    reservation_id=canary_reservation_id,
                )
                if canary_budget_status != "reserved":
                    raise RuntimeError(
                        "token budget cannot cover the strict-schema canary"
                    )
            completion = None
            try:
                completion = await canary(
                    client, model, reasoning_effort=reasoning_effort
                )
            finally:
                if token_budget is not None:
                    await _complete_budget(
                        connection,
                        budget_lock,
                        reservation_id=canary_reservation_id,
                        usage=_usage(completion),
                    )
        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*workers)
        await result_queue.put(None)
        await result_queue.join()
        await writer
        if not budget_exhausted.is_set() and yielded != progress["pending"]:
            raise ValueError(
                f"pending request count mismatch: yielded={yielded}, "
                f"expected={progress['pending']}"
            )
    finally:
        for task in workers:
            task.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        if not writer.done():
            writer.cancel()
            await asyncio.gather(writer, return_exceptions=True)
        await client.close()
    return progress, endpoint, budget_exhausted.is_set()


def _database_counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        "successful_requests": int(
            connection.execute(
                "SELECT COUNT(*) FROM requests WHERE status = 'success'"
            ).fetchone()[0]
        ),
        "failed_requests": int(
            connection.execute(
                "SELECT COUNT(*) FROM requests WHERE status = 'failed'"
            ).fetchone()[0]
        ),
        "scored_rows": int(connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0]),
    }


def _materialize(
    connection: sqlite3.Connection,
    path: Path,
    *,
    task_id: str,
    requested_model: str,
    expected_rows: int,
) -> str:
    frame = pd.read_sql_query(
        """
        SELECT
            r.task_id, r.source_id, r.pmid, r.canonical_smiles,
            s.cleaned_record_id, s.relevance_score, s.completeness_score,
            s.informativeness_score, s.context_requirement, r.request_id,
            r.payload_sha256, ? AS prompt_version, r.requested_model, r.served_model
        FROM scores s JOIN requests r USING (request_id)
        WHERE r.status = 'success'
        ORDER BY s.cleaned_record_id
        """,
        connection,
        params=(PROMPT_VERSION,),
    )
    if len(frame) != expected_rows or frame["cleaned_record_id"].duplicated().any():
        raise ValueError(
            f"cannot publish {task_id}: scored rows={len(frame)}, expected={expected_rows}"
        )
    if list(frame.columns) != OUTPUT_COLUMNS:
        raise ValueError("output columns differ from the frozen contract")
    if frame["served_model"].ne(requested_model).any() or frame[SCORE_COLUMNS].isna().any().any():
        raise ValueError("output contains a model mismatch or null score")
    with atomic_output_path(path) as temporary:
        frame.to_parquet(temporary, index=False)
    return file_sha256(path)


@contextmanager
def _run_lock(output_dir: Path) -> Iterator[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".run.lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another run holds {output_dir / '.run.lock'}") from exc
        yield


def run_task(args: argparse.Namespace, task_id: str) -> dict[str, Any]:
    task_config = TASKS[task_id]
    base_url = args.base_url or task_config["base_url"]
    model = args.model or task_config["model"]
    reasoning_effort = (
        args.reasoning_effort
        if args.reasoning_effort is not None
        else task_config["reasoning_effort"]
    )
    token_budget = (
        args.token_budget
        if args.token_budget is not None
        else task_config["token_budget"]
    )
    if token_budget is not None and args.pilot_size:
        raise ValueError(
            "metered tasks use the budgeted canary in the full output; "
            "a separate pilot could exceed the global token budget"
        )
    input_path = (
        Path(args.input)
        if args.input
        else Path(f"data/evidence_libraries/{task_id}/v7/01_cleaned/records.parquet")
    )
    frame, inventory = _load_frame(input_path, task_id)
    deferred_groups = pd.DataFrame()
    if args.max_group_rows:
        sizes = frame.groupby(GROUP_COLUMNS, sort=False, observed=True).size()
        deferred_groups = sizes[sizes > args.max_group_rows].rename("rows").reset_index()
        deferred_index = pd.MultiIndex.from_frame(deferred_groups[GROUP_COLUMNS])
        frame = frame[
            ~pd.MultiIndex.from_frame(frame[GROUP_COLUMNS]).isin(deferred_index)
        ].copy()
        inventory["selection"] = {
            "maximum_group_rows": args.max_group_rows,
            "deferred_groups": int(len(deferred_groups)),
            "deferred_rows": int(deferred_groups["rows"].sum()),
            "selected_groups": inventory["groups"] - int(len(deferred_groups)),
            "selected_rows": inventory["rows"] - int(deferred_groups["rows"].sum()),
        }
    if args.inventory_only:
        print(json.dumps(inventory, indent=2, sort_keys=True))
        return inventory
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"data/evidence_libraries/{task_id}/v7/record_informativeness_v1")
    )
    if args.pilot_size:
        output_dir = output_dir / f"pilot_{args.pilot_size}"
        selected = select_pilot(frame, task_id, args.pilot_size)
        groups: Iterator[dict[str, Any]] = iter(selected)
        selected_groups = len(selected)
        selected_rows = sum(len(group["rows"]) for group in selected)
    else:
        groups = iter_groups(frame, task_id)
        selected_groups = inventory.get("selection", {}).get(
            "selected_groups", inventory["groups"]
        )
        selected_rows = inventory.get("selection", {}).get(
            "selected_rows", inventory["rows"]
        )
    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        raise ValueError(f"concurrency must be in [1, {MAX_CONCURRENCY}]")
    metadata = {
        "version": VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": file_sha256(TEMPLATE_PATH),
        "task_id": task_id,
        "input_path": str(input_path.resolve()),
        "input_sha256": inventory["input_sha256"],
        "base_url": base_url.rstrip("/"),
        "requested_model": model,
        "reasoning_effort": reasoning_effort,
        "token_budget": token_budget,
        "completion_policy": {
            "minimum_tokens": MAX_COMPLETION_TOKENS,
            "maximum_tokens": MAX_COMPLETION_TOKENS,
            "per_row_tokens": 0,
        },
        "selected_groups": selected_groups,
        "selected_rows": selected_rows,
    }
    with _run_lock(output_dir):
        if not deferred_groups.empty:
            deferred_path = output_dir / "deferred_large_groups.parquet"
            with atomic_output_path(deferred_path) as temporary:
                deferred_groups.to_parquet(temporary, index=False)
            metadata["deferred_large_groups"] = {
                **inventory["selection"],
                "path": str(deferred_path.resolve()),
                "sha256": file_sha256(deferred_path),
            }
        connection = _connect_database(
            output_dir / "requests.sqlite3", metadata=metadata, resume=args.resume
        )
        _initialize_budget(connection, token_budget)
        if base_url.startswith("https://openrouter.ai/"):
            api_key, credential_env = resolve_api_key(
                "openrouter",
                env_file=Path(args.env_file) if args.env_file else DEFAULT_ENV_FILE,
                credential_env=args.api_key_env,
            )
        else:
            api_key, credential_env = "EMPTY", ""
        manifest = {
            **metadata,
            "status": "running",
            "inventory": inventory,
            "concurrency": args.concurrency,
            "timeout_s": args.timeout_s,
            "max_attempts": args.max_attempts,
            "credential_env": credential_env,
            "started_at": time.time(),
        }
        write_json_atomic(output_dir / "manifest.json", manifest)
        try:
            progress, endpoint, budget_exhausted = asyncio.run(
                _run_requests(
                    groups,
                    connection=connection,
                    model=model,
                    base_url=base_url,
                    concurrency=args.concurrency,
                    timeout_s=args.timeout_s,
                    max_attempts=args.max_attempts,
                    total_groups=selected_groups,
                    reasoning_effort=reasoning_effort,
                    api_key=api_key,
                    token_budget=token_budget,
                )
            )
            integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"SQLite quick_check failed: {integrity}")
            counts = _database_counts(connection)
            complete = (
                counts["successful_requests"] == selected_groups
                and counts["failed_requests"] == 0
                and counts["scored_rows"] == selected_rows
            )
            manifest.update(
                {
                    "status": (
                        "pilot_complete"
                        if complete and args.pilot_size
                        else "complete" if complete else "incomplete"
                    ),
                    "endpoint_preflight": endpoint,
                    "run_progress": progress,
                    "counts": counts,
                    "token_budget": _budget_summary(connection),
                    "sqlite_quick_check": integrity,
                    "completed_at": time.time(),
                }
            )
            if complete:
                parquet_name = (
                    "pilot_record_informativeness.parquet"
                    if args.pilot_size
                    else "record_informativeness.parquet"
                )
                manifest["output_sha256"] = _materialize(
                    connection,
                    output_dir / parquet_name,
                    task_id=task_id,
                    requested_model=model,
                    expected_rows=selected_rows,
                )
            elif budget_exhausted:
                manifest["status"] = "budget_exhausted"
                if counts["scored_rows"]:
                    manifest["partial_output_sha256"] = _materialize(
                        connection,
                        output_dir / "record_informativeness_partial.parquet",
                        task_id=task_id,
                        requested_model=model,
                        expected_rows=counts["scored_rows"],
                    )
            write_json_atomic(output_dir / "manifest.json", manifest)
            if not complete and not budget_exhausted:
                raise RuntimeError(
                    f"{task_id} remains incomplete; resume after inspecting requests.sqlite3"
                )
            return manifest
        except BaseException:
            if manifest.get("status") == "running":
                manifest["status"] = "interrupted"
            manifest["counts"] = _database_counts(connection)
            write_json_atomic(output_dir / "manifest.json", manifest)
            raise
        finally:
            connection.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=[*TASKS, "all"])
    parser.add_argument("--input")
    parser.add_argument("--output-dir")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--pilot-size", type=int, default=0)
    parser.add_argument("--max-group-rows", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--api-key-env")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--timeout-s", type=float, default=7_200)
    parser.add_argument("--max-attempts", type=int, default=4)
    args = parser.parse_args(argv)
    if args.task == "all" and (args.input or args.output_dir):
        parser.error("--input and --output-dir cannot be used with --task all")
    if args.pilot_size < 0 or args.max_attempts < 1:
        parser.error("--pilot-size must be nonnegative and --max-attempts positive")
    if args.max_group_rows is not None and args.max_group_rows < 1:
        parser.error("--max-group-rows must be positive")
    if args.pilot_size and args.max_group_rows:
        parser.error("--max-group-rows applies only to full runs")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    task_ids = TASKS if args.task == "all" else (args.task,)
    for task_id in task_ids:
        run_task(args, task_id)


if __name__ == "__main__":
    main()
