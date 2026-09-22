"""Qualify and rank reusable OpenRouter routes for DeepSeek Flash workloads."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import statistics
import threading
import time
from typing import Any, Mapping, Sequence

import httpx

from data.processing.llm_api import DEFAULT_ENV_FILE, openai_compatible_client, resolve_api_key
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "openrouter_provider_pool.v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
MIXED_MODEL = "deepseek/deepseek-v4.1-flash"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
CACHE_ROOT = Path(__file__).resolve().parents[1] / "caches/openrouter_provider_pool/v1"
PRICE_CAP = 0.66
EXECUTION_PRICE_CAP = 0.659999
TOKEN_MEDIAN_CAP = 16_384
QUALIFICATION_REPETITIONS = 3
REFRESH_TTL = timedelta(hours=1)
TOP_ROUTE_COUNT = 7
REQUESTS_PER_ROUTE = 3
MAX_TOKENS = 20_480
_WRITE_LOCK = threading.Lock()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _models(allow_mixed_flash_models: bool) -> tuple[str, ...]:
    return (DEFAULT_MODEL, MIXED_MODEL) if allow_mixed_flash_models else (DEFAULT_MODEL,)


def _profile_name(allow_mixed_flash_models: bool) -> str:
    return "mixed" if allow_mixed_flash_models else "0731"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _minutes(hhmm: int) -> int:
    hours, minutes = divmod(int(hhmm), 100)
    if not 0 <= hours <= 23 or not 0 <= minutes <= 59:
        raise ValueError(f"invalid UTC HHMM value: {hhmm}")
    return hours * 60 + minutes


def _override_matches(override: Mapping[str, Any], now: datetime) -> bool:
    days = [str(day).casefold() for day in override.get("utc_days", [])]
    if days and now.strftime("%A").casefold() not in days:
        return False
    if "utc_start" not in override and "utc_end" not in override:
        return True
    start = _minutes(int(override.get("utc_start", 0)))
    raw_end = int(override.get("utc_end", 0))
    end = 24 * 60 if raw_end == 0 and start else _minutes(raw_end)
    minute = now.hour * 60 + now.minute
    return start <= minute < end if start <= end else minute >= start or minute < end


def active_token_price(
    pricing: Mapping[str, Any], field: str, now: datetime,
) -> float:
    matches = [item for item in pricing.get("overrides", []) if _override_matches(item, now)]
    if len(matches) > 1:
        raise ValueError("overlapping OpenRouter price overrides")
    selected = matches[0] if matches else pricing
    return float(selected.get(field, pricing[field])) * 1_000_000


def active_completion_price(pricing: Mapping[str, Any], now: datetime) -> float:
    return active_token_price(pricing, "completion", now)


def _get_json(client: httpx.Client, url: str) -> dict[str, Any]:
    response = client.get(url)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"OpenRouter returned a non-object from {url}")
    return payload


def _route_record(
    model: str, canonical_model: str, tag: str, rows: Sequence[Mapping[str, Any]], now: datetime,
) -> dict[str, Any]:
    supported = set(rows[0].get("supported_parameters", []))
    for row in rows[1:]:
        supported &= set(row.get("supported_parameters", []))
    output_prices = [active_completion_price(row["pricing"], now) for row in rows]
    input_prices = [active_token_price(row["pricing"], "prompt", now) for row in rows]
    throughputs = [(row.get("throughput_last_30m") or {}).get("p50") for row in rows]
    measured_throughputs = [float(value) for value in throughputs if value is not None]
    context_lengths = [int(row.get("context_length") or 0) for row in rows]
    completion_limits = [int(row.get("max_completion_tokens") or 0) for row in rows]
    identity = {
        "model": model, "canonical_model": canonical_model, "route_tag": tag,
        "provider_name": str(rows[0]["provider_name"]),
        "quantizations": sorted({str(row.get("quantization")) for row in rows}),
        "supported_parameters": sorted(supported),
    }
    return {
        **identity, "route_key": f"{model}|{tag}",
        "fingerprint_sha256": _sha256_bytes(_canonical_json(identity).encode()),
        "active_input_price": max(input_prices),
        "active_output_price": max(output_prices),
        "throughput_p50": max(measured_throughputs) if measured_throughputs else None,
        "context_length": max(context_lengths),
        "max_completion_tokens": max(completion_limits),
        "max_request_cost_usd": max(
            (
                context_length * input_price
                + completion_limit * output_price
            ) / 1_000_000
            for context_length, completion_limit, input_price, output_price in zip(
                context_lengths, completion_limits, input_prices, output_prices
            )
        ),
        "healthy": all(int(row.get("status", 1)) == 0 for row in rows),
        "supports_response_format": "response_format" in supported,
        "supports_reasoning_effort": "reasoning_effort" in supported,
        "inventory_records": len(rows),
    }


def discover_routes(
    allow_mixed_flash_models: bool, *, now: datetime | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observed = now or _utc_now()
    api_key, _ = resolve_api_key("openrouter", env_file=DEFAULT_ENV_FILE,
                                 credential_env="OPEN_ROUTER_KEY")
    with httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=30) as client:
        models_payload = _get_json(client, f"{OPENROUTER_URL}/models")
        canonical = {row["id"]: row.get("canonical_slug", row["id"])
                     for row in models_payload.get("data", [])}
        raw, routes = {}, []
        for model in _models(allow_mixed_flash_models):
            payload = _get_json(client, f"{OPENROUTER_URL}/models/{model}/endpoints")
            rows = payload.get("data", {}).get("endpoints", [])
            if not rows:
                raise ValueError(f"OpenRouter returned no endpoints for {model}")
            raw[model] = payload
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                grouped.setdefault(str(row.get("tag") or row["provider_name"]), []).append(row)
            routes.extend(_route_record(model, canonical.get(model, model), tag, members, observed)
                          for tag, members in sorted(grouped.items()))
    receipt = {"fetched_at": observed.isoformat(), "models": list(_models(allow_mixed_flash_models)),
               "raw": raw}
    return sorted(routes, key=lambda row: row["route_key"]), receipt


def _eligible(route: Mapping[str, Any]) -> bool:
    return bool(route["healthy"] and route["supports_reasoning_effort"]
                and route["throughput_p50"] is not None
                and float(route["active_output_price"]) < PRICE_CAP)


def _validate_score_response(result: Any, candidate_count: int) -> dict[str, Any]:
    if not isinstance(result, dict) or set(result) != {"scores"}:
        raise ValueError("response must contain only scores")
    scores = result["scores"]
    aliases = [f"Candidate {index}" for index in range(1, candidate_count + 1)]
    if not isinstance(scores, list) or [row.get("candidate") for row in scores] != aliases:
        raise ValueError("response candidates are incomplete or out of order")
    for row in scores:
        if set(row) != {"candidate", "weight", "rationale"}:
            raise ValueError("score has unexpected fields")
        weight = row["weight"]
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not 0 <= weight <= 1:
            raise ValueError("weight is outside [0,1]")
        if not math.isclose(float(weight) * 100, round(float(weight) * 100), abs_tol=1e-9):
            raise ValueError("weight is not in hundredth increments")
        if not isinstance(row["rationale"], str) or not row["rationale"].strip():
            raise ValueError("rationale is empty")
    return result


def _database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=60, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
          route_key TEXT NOT NULL, repetition INTEGER NOT NULL, status TEXT NOT NULL,
          started_at TEXT NOT NULL, finished_at TEXT NOT NULL, elapsed_seconds REAL NOT NULL,
          completion_tokens INTEGER, reasoning_tokens INTEGER, response_json TEXT,
          receipt_json TEXT NOT NULL, error TEXT, PRIMARY KEY(route_key,repetition)
        )
    """)
    connection.commit()
    return connection


def _reasoning_text(message: Any) -> str | None:
    extras = getattr(message, "model_extra", None) or {}
    return (getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
            or extras.get("reasoning_content") or extras.get("reasoning"))


def _call_route(
    client: Any, route: Mapping[str, Any], prompt: str, candidate_count: int,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": route["model"], "messages": [
            {"role": "system", "content": "Return valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "reasoning_effort": "high", "max_tokens": MAX_TOKENS,
        "extra_headers": {"X-OpenRouter-Metadata": "enabled"},
        "extra_body": {"thinking": {"type": "enabled"}, "provider": {
            "order": [route["route_tag"]], "allow_fallbacks": False,
            "require_parameters": True, "max_price": {"completion": EXECUTION_PRICE_CAP},
        }},
    }
    if route["supports_response_format"]:
        request["response_format"] = {"type": "json_object"}
    completion = client.chat.completions.create(**request)
    message = completion.choices[0].message
    parsed = _validate_score_response(json.loads(message.content or ""), candidate_count)
    metadata = (getattr(completion, "openrouter_metadata", None)
                or (getattr(completion, "model_extra", None) or {}).get("openrouter_metadata")
                or {})
    available = metadata.get("endpoints", {}).get("available", [])
    selected = [item for item in available if item.get("selected")]
    if len(selected) != 1 or selected[0].get("provider") != route["provider_name"]:
        raise ValueError("OpenRouter served a different provider")
    if str(completion.model) not in {route["model"], route["canonical_model"]}:
        raise ValueError("OpenRouter served a different model")
    usage = completion.usage.model_dump() if completion.usage is not None else {}
    completion_tokens = usage.get("completion_tokens")
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if completion_tokens is None or reasoning_tokens is None:
        raise ValueError("completion receipt is missing token counts")
    return {"response": parsed, "reasoning_content": _reasoning_text(message),
            "generation_id": str(completion.id), "served_model": str(completion.model),
            "openrouter_metadata": metadata, "usage": usage,
            "completion_tokens": int(completion_tokens),
            "reasoning_tokens": int(reasoning_tokens)}


def _attempt(
    client: Any, route: Mapping[str, Any], repetition: int,
    prompt: str, candidate_count: int,
) -> dict[str, Any]:
    started_at, started = _utc_now(), time.monotonic()
    try:
        receipt = _call_route(client, route, prompt, candidate_count)
        return {"route_key": route["route_key"], "repetition": repetition, "status": "complete",
                "started_at": started_at.isoformat(), "finished_at": _utc_now().isoformat(),
                "elapsed_seconds": time.monotonic() - started,
                "completion_tokens": receipt["completion_tokens"],
                "reasoning_tokens": receipt["reasoning_tokens"],
                "response_json": _canonical_json(receipt["response"]),
                "receipt_json": _canonical_json(receipt), "error": None}
    except Exception as error:
        return {"route_key": route["route_key"], "repetition": repetition, "status": "failed",
                "started_at": started_at.isoformat(), "finished_at": _utc_now().isoformat(),
                "elapsed_seconds": time.monotonic() - started, "completion_tokens": None,
                "reasoning_tokens": None, "response_json": None, "receipt_json": "{}",
                "error": f"{type(error).__name__}: {error}"}


def _store_attempt(connection: sqlite3.Connection, result: Mapping[str, Any]) -> None:
    columns = ("route_key", "repetition", "status", "started_at", "finished_at",
               "elapsed_seconds", "completion_tokens", "reasoning_tokens", "response_json",
               "receipt_json", "error")
    with _WRITE_LOCK:
        connection.execute(
            f"INSERT OR REPLACE INTO attempts ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            [result[column] for column in columns],
        )
        connection.commit()


def _run_route(
    connection: sqlite3.Connection, client: Any, route: Mapping[str, Any],
    prompt: str, candidate_count: int,
) -> None:
    existing = {int(row[0]) for row in connection.execute(
        "SELECT repetition FROM attempts WHERE route_key=?", (route["route_key"],)
    )}
    for repetition in range(1, QUALIFICATION_REPETITIONS + 1):
        if repetition not in existing:
            _store_attempt(connection, _attempt(client, route, repetition, prompt, candidate_count))


def _result_rows(
    connection: sqlite3.Connection, routes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for route in routes:
        attempts = list(connection.execute(
            "SELECT * FROM attempts WHERE route_key=? ORDER BY repetition", (route["route_key"],)
        ))
        complete = [row for row in attempts if row["status"] == "complete"]
        completion = [int(row["completion_tokens"]) for row in complete]
        reasoning = [int(row["reasoning_tokens"]) for row in complete]
        all_valid = len(attempts) == QUALIFICATION_REPETITIONS == len(complete)
        completion_median = statistics.median(completion) if completion else None
        reasoning_median = statistics.median(reasoning) if reasoning else None
        gold = bool(all_valid and completion_median <= TOKEN_MEDIAN_CAP
                    and reasoning_median <= TOKEN_MEDIAN_CAP)
        output.append({**route, "attempt_count": len(attempts), "complete_count": len(complete),
                       "completion_tokens_median": completion_median,
                       "reasoning_tokens_median": reasoning_median, "gold": gold,
                       "errors": " | ".join(row["error"] for row in attempts if row["error"])})
    return output


def _write_tsv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    columns = [key for key in rows[0] if key != "supported_parameters"] if rows else []
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (_canonical_json(value) if isinstance(value, (list, dict)) else value)
                             for key, value in row.items() if key in columns})
    temporary.replace(path)


def _qualification_root(run_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", run_id):
        raise ValueError("qualification run_id contains unsupported characters")
    return CACHE_ROOT / "qualification_runs" / run_id


def qualify(
    run_id: str, prompt_file: Path, *, candidate_count: int = 7,
    allow_mixed_flash_models: bool = False, parallelism: int = 21,
) -> dict[str, Any]:
    root = _qualification_root(run_id)
    root.mkdir(parents=True, exist_ok=True)
    frozen_prompt = root / "prompt.txt"
    if not frozen_prompt.exists():
        shutil.copyfile(prompt_file, frozen_prompt)
    elif _sha256(frozen_prompt) != _sha256(prompt_file):
        raise ValueError("qualification prompt changed on resume")
    routes, inventory = discover_routes(allow_mixed_flash_models)
    candidates = [route for route in routes if _eligible(route)]
    write_json_atomic(root / "inventory.json", inventory)
    manifest = {"version": VERSION, "status": "running", "run_id": run_id,
                "models": list(_models(allow_mixed_flash_models)), "reasoning_effort": "high",
                "prompt_sha256": _sha256(frozen_prompt), "candidate_count": candidate_count,
                "price_cap_per_million_output_tokens": PRICE_CAP,
                "token_median_cap": TOKEN_MEDIAN_CAP, "required_successes": 3,
                "candidate_routes": candidates, "started_at": _utc_now().isoformat()}
    write_json_atomic(root / "manifest.json", manifest)
    connection = _database(root / "requests.sqlite3")
    client, credential = openai_compatible_client(
        base_url=OPENROUTER_URL, provider="openrouter", env_file=DEFAULT_ENV_FILE,
        credential_env="OPEN_ROUTER_KEY", max_connections=parallelism,
        timeout_s=3_600, max_retries=0,
    )
    prompt = frozen_prompt.read_text(encoding="utf-8")
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = [executor.submit(_run_route, connection, client, route,
                                   prompt, candidate_count) for route in candidates]
        for future in as_completed(futures):
            future.result()
    rows = _result_rows(connection, candidates)
    connection.close()
    _write_tsv(rows, root / "results.tsv")
    manifest.update(status="complete", finished_at=_utc_now().isoformat(), credential_env=credential,
                    candidate_route_count=len(candidates), gold_route_count=sum(row["gold"] for row in rows),
                    artifacts={name: _sha256(root / name) for name in
                               ("prompt.txt", "inventory.json", "requests.sqlite3", "results.tsv")})
    write_json_atomic(root / "manifest.json", manifest)
    pointer = {"version": VERSION, "run_id": run_id, "path": str(root),
               "manifest_sha256": _sha256(root / "manifest.json")}
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(CACHE_ROOT / "QUALIFICATION_CURRENT.json", pointer)
    return manifest


def _read_results(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row["gold"] = row["gold"] == "True"
        for key in ("active_output_price", "throughput_p50"):
            row[key] = float(row[key])
    return rows


def _percentiles(values: Sequence[float], *, higher_is_better: bool) -> dict[float, float]:
    ordered = sorted(set(values), reverse=not higher_is_better)
    if len(ordered) == 1:
        return {ordered[0]: 1.0}
    return {value: index / (len(ordered) - 1) for index, value in enumerate(ordered)}


def _rank_routes(routes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not routes:
        return []
    tps = _percentiles([row["throughput_p50"] for row in routes], higher_is_better=True)
    price = _percentiles([row["active_output_price"] for row in routes], higher_is_better=False)
    ranked = []
    for route in routes:
        route = dict(route)
        route["tps_percentile"] = tps[route["throughput_p50"]]
        route["price_percentile"] = price[route["active_output_price"]]
        route["composite_score"] = (2 * route["tps_percentile"] + route["price_percentile"]) / 3
        ranked.append(route)
    ranked.sort(key=lambda row: (-row["composite_score"], -row["throughput_p50"],
                                 row["active_output_price"], row["model"], row["route_tag"]))
    return ranked[:TOP_ROUTE_COUNT]


def _qualification() -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    pointer_path = CACHE_ROOT / "QUALIFICATION_CURRENT.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    root = Path(pointer["path"])
    if _sha256(root / "manifest.json") != pointer["manifest_sha256"]:
        raise ValueError("qualification manifest hash changed")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("qualification is incomplete")
    return pointer, root, _read_results(root / "results.tsv")


def refresh(allow_mixed_flash_models: bool = False) -> dict[str, Any]:
    profile = _profile_name(allow_mixed_flash_models)
    profile_root = CACHE_ROOT / "profiles" / profile
    profile_root.mkdir(parents=True, exist_ok=True)
    lock_path = profile_root / ".refresh.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pointer, qualification_root, qualified = _qualification()
        discovered, inventory = discover_routes(allow_mixed_flash_models)
        gold = {row["route_key"]: row for row in qualified if row["gold"]}
        usable = []
        for route in discovered:
            prior = gold.get(route["route_key"])
            if prior and _eligible(route) and prior["fingerprint_sha256"] == route["fingerprint_sha256"]:
                usable.append(route)
        ranked = _rank_routes(usable)
        if not ranked:
            raise RuntimeError("no currently eligible gold OpenRouter routes")
        current = {"version": VERSION, "profile": profile,
                   "allow_mixed_flash_models": allow_mixed_flash_models,
                   "models": list(_models(allow_mixed_flash_models)),
                   "fetched_at": inventory["fetched_at"], "expires_after_seconds": 3600,
                   "qualification": {**pointer, "results_sha256": _sha256(qualification_root / "results.tsv")},
                   "usable_gold_route_count": len(usable), "selected_route_count": len(ranked),
                   "score": {"tps_weight": 2 / 3, "output_price_weight": 1 / 3,
                             "maximum_routes": TOP_ROUTE_COUNT}, "routes": ranked}
        write_json_atomic(profile_root / "CURRENT.json", current)
        return current


def load_ranked_pool(allow_mixed_flash_models: bool = False) -> dict[str, Any]:
    path = CACHE_ROOT / "profiles" / _profile_name(allow_mixed_flash_models) / "CURRENT.json"
    refresh_needed = True
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(current["fetched_at"])
        refresh_needed = _utc_now() - fetched >= REFRESH_TTL
    if refresh_needed:
        current = refresh(allow_mixed_flash_models)
    current = dict(current)
    current["snapshot_sha256"] = _sha256(path)
    return current


def request_assignment(snapshot: Mapping[str, Any], schedule_index: int,
                       *, seed_request_count: int) -> dict[str, Any]:
    flash_index = schedule_index - seed_request_count
    group = flash_index // REQUESTS_PER_ROUTE
    routes = snapshot.get("routes", [])
    if flash_index < 0 or not routes:
        raise ValueError("provider assignment requires an anchored request and nonempty pool")
    route = routes[group % len(routes)]
    return {
        "requested_model": route["model"],
        "allowed_served_models": [route["model"], route["canonical_model"]],
        "selected_provider_route": route["route_tag"],
        "provider_pool_snapshot_sha256": snapshot["snapshot_sha256"],
        "provider_routing": {
            "order": [route["route_tag"]], "allow_fallbacks": False,
            "require_parameters": True, "max_price": {
                "prompt": route["active_input_price"],
                "completion": route["active_output_price"],
            },
            "omit_response_format": not route["supports_response_format"],
        },
    }


def export_provider_pool(
    output: Path, capacity: int, *, allow_mixed_flash_models: bool = False,
    spend_budget: Mapping[str, Any] | None = None,
    credential_env: str | None = None,
) -> dict[str, Any]:
    """Freeze the ranked qualified routes as one bounded execution pool."""
    if capacity < 1:
        raise ValueError("provider-pool capacity must be positive")
    snapshot = load_ranked_pool(allow_mixed_flash_models)
    routes = snapshot["routes"]
    selected_credential_env = str(
        credential_env
        or (spend_budget or {}).get("credential_env")
        or "OPEN_ROUTER_KEY"
    )
    base, remainder = divmod(capacity, len(routes))
    if base == 0:
        raise ValueError("capacity must cover every selected route")
    providers = []
    for index, route in enumerate(routes):
        route_capacity = base + (index < remainder)
        providers.append({
            "name": (
                "openrouter_"
                f"{route['model'].rsplit('/', 1)[-1].replace('.', '_')}_"
                f"{route['route_tag'].replace('/', '_')}"
            ),
            "base_url": OPENROUTER_URL,
            "model": route["model"],
            "api_key_env": selected_credential_env,
            "max_inflight": route_capacity,
            "initial_latency_s": 120,
            "timeout_s": 3_600,
            "request_extra_body": {
                "thinking": {"type": "enabled"},
                "omit_response_format": not route["supports_response_format"],
                "allowed_served_models": [route["model"], route["canonical_model"]],
                "expected_upstream_provider": route["provider_name"],
                "provider_pool_snapshot_sha256": snapshot["snapshot_sha256"],
                "max_request_cost_usd": route["max_request_cost_usd"],
                "provider": {
                    "order": [route["route_tag"]],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                    "max_price": {
                        "prompt": route["active_input_price"],
                        "completion": route["active_output_price"],
                    },
                },
            },
        })
    payload = {
        "version": "openai_provider_pool.v1",
        "providers": providers,
        "failure_threshold": 3,
        "cooldown_seconds": 60,
        "max_failovers": len(providers) - 1,
        "latency_ewma_alpha": 0.2,
        "openrouter_ranked_profile": {
            "profile": snapshot["profile"],
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "qualification": snapshot["qualification"],
            "score": snapshot["score"],
        },
    }
    if spend_budget is not None:
        payload["spend_budget"] = dict(spend_budget)
    write_json_atomic(output, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    qualify_parser = subparsers.add_parser("qualify")
    qualify_parser.add_argument("--run-id", required=True)
    qualify_parser.add_argument("--prompt-file", type=Path, required=True)
    qualify_parser.add_argument("--candidate-count", type=int, default=7)
    qualify_parser.add_argument("--parallelism", type=int, default=21)
    qualify_parser.add_argument("--allow-mixed-flash-models", action="store_true")
    for command in ("refresh", "status"):
        child = subparsers.add_parser(command)
        child.add_argument("--allow-mixed-flash-models", action="store_true")
    export_parser = subparsers.add_parser("export-provider-pool")
    export_parser.add_argument("--output", required=True, type=Path)
    export_parser.add_argument("--capacity", required=True, type=int)
    export_parser.add_argument("--allow-mixed-flash-models", action="store_true")
    export_parser.add_argument("--spend-budget-config", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "qualify":
        result = qualify(args.run_id, args.prompt_file, candidate_count=args.candidate_count,
                         allow_mixed_flash_models=args.allow_mixed_flash_models,
                         parallelism=args.parallelism)
    elif args.command == "refresh":
        result = refresh(args.allow_mixed_flash_models)
    elif args.command == "export-provider-pool":
        spend_budget = (
            json.loads(args.spend_budget_config.read_text(encoding="utf-8"))
            if args.spend_budget_config else None
        )
        result = export_provider_pool(
            args.output,
            args.capacity,
            allow_mixed_flash_models=args.allow_mixed_flash_models,
            spend_budget=spend_budget,
        )
    else:
        result = load_ranked_pool(args.allow_mixed_flash_models)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
