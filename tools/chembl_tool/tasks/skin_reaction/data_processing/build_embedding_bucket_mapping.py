#!/usr/bin/env python3
"""Build Skin_Reaction auxiliary mappings from embedding-neighbourhood prompts.

Distinct source values are embedded locally, clustered into neighbourhoods of
roughly 100 values, and normalized with one structured-output request per
cluster.  Per-cluster caches and a cumulative token ledger make paid runs
resumable without persisting credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

from tools.chembl_tool.tasks.skin_reaction.data_processing.auxiliary_mapping_helpers.reconciliation import (
    DEFAULT_OUTPUT,
    FORBIDDEN_OUTPUT,
    NULL_LIKE,
    SOURCE_SPECS,
    OutputSpec,
    build_mapping,
    load_reviewed_mapping,
    validate_mapping,
)


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_TOKEN_BUDGET = 500_000
CLUSTER_TARGET_SIZE = 100
CLUSTER_RANDOM_SEED = 20260801
MINIBATCH_SIZE = 4096
MINIBATCH_N_INIT = 3
MINIBATCH_MAX_ITER = 100
PROMPT_VERSION = "starling_skin_embedding_bucket_mapping.v3"
LEDGER_VERSION = "starling_skin_auxiliary_token_ledger.v1"
PROMPT_REGISTRY_PATH = Path(__file__).with_name("auxiliary_value_prompts.json")
BUDGET_EXIT_CODE = 75
USAGE_EXIT_CODE = 76
API_EXIT_CODE = 77


@dataclass(frozen=True)
class ExtractionSpec:
    source_id: str
    input_column: str
    output_name: str
    output_column: str
    prompt: str | None
    label_source: str
    reviewed_mapping_path: Path | None
    null_sentinel: str | None
    bucket_pattern: str | None
    clustering: str
    cluster_target_size: int = CLUSTER_TARGET_SIZE
    max_labels_per_call: int | None = None


@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    values: tuple[str, ...]


class BudgetExhausted(RuntimeError):
    pass


class UsageUnavailable(RuntimeError):
    pass


class NonRetryableAPIError(RuntimeError):
    pass


class ClusterQueryFailure(RuntimeError):
    pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_prompt_registry() -> dict[str, Any]:
    payload = json.loads(PROMPT_REGISTRY_PATH.read_text(encoding="utf-8"))
    if payload.get("registry_version") != "starling_skin_auxiliary_prompts.v3":
        raise ValueError("Skin auxiliary prompt registry version mismatch")
    prompts = payload.get("prompts")
    if not isinstance(prompts, dict):
        raise ValueError("Skin auxiliary prompt registry has no prompts")
    expected = {
        (source_id, output.output_name): output.current_input_column
        for source_id, source in SOURCE_SPECS.items()
        for output in source.outputs
        if output.label_source == "llm"
    }
    actual: dict[tuple[str, str], str] = {}
    for source_id, outputs in prompts.items():
        if not isinstance(outputs, dict):
            raise ValueError(f"prompt source {source_id!r} is not an object")
        for output_name, section in outputs.items():
            if not isinstance(section, dict):
                raise ValueError(f"prompt {source_id}/{output_name} is not an object")
            prompt = section.get("prompt")
            input_column = section.get("input_column")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"prompt {source_id}/{output_name} is empty")
            actual[(source_id, output_name)] = str(input_column or "")
    if actual != expected:
        raise ValueError(
            f"prompt inventory differs from SOURCE_SPECS: expected={expected} actual={actual}"
        )
    return payload


_PROMPT_REGISTRY = _load_prompt_registry()


def _extraction(source_id: str, output: OutputSpec) -> ExtractionSpec:
    prompt = None
    if output.label_source == "llm":
        prompt = _PROMPT_REGISTRY["prompts"][source_id][output.output_name]["prompt"]
    elif output.label_source != "reviewed_table":
        raise ValueError(f"unsupported label source {output.label_source!r}")
    return ExtractionSpec(
        source_id=source_id,
        input_column=output.current_input_column,
        output_name=output.output_name,
        output_column=output.current_output_column,
        prompt=prompt,
        label_source=output.label_source,
        reviewed_mapping_path=output.reviewed_mapping_path,
        null_sentinel=output.null_sentinel,
        bucket_pattern=output.bucket_pattern,
        clustering=output.clustering,
        cluster_target_size=output.cluster_target_size,
        max_labels_per_call=output.max_labels_per_call,
    )


def _source_extractions(source_id: str) -> tuple[ExtractionSpec, ...]:
    return tuple(_extraction(source_id, output) for output in SOURCE_SPECS[source_id].outputs)


def _distinct_values(series: pd.Series) -> list[str]:
    values = {
        str(value).strip()
        for value in series.dropna().tolist()
        if str(value).strip().casefold() not in NULL_LIKE
    }
    return sorted(values, key=lambda value: (value.casefold(), value))


def _mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def _embed_value_groups(
    groups: Mapping[str, list[str]], *, model_name: str, batch_size: int, device: str
) -> dict[str, np.ndarray]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    try:
        output: dict[str, np.ndarray] = {}
        for name, values in groups.items():
            chunks: list[np.ndarray] = []
            started = time.time()
            for start in range(0, len(values), batch_size):
                batch = values[start : start + batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.inference_mode():
                    if device.startswith("cuda"):
                        with torch.autocast("cuda", dtype=torch.float16):
                            hidden = model(**encoded).last_hidden_state
                    else:
                        hidden = model(**encoded).last_hidden_state
                    pooled = _mean_pool(hidden, encoded["attention_mask"])
                    pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                chunks.append(pooled.cpu().numpy())
                done = min(start + batch_size, len(values))
                if done == len(values) or done == batch_size or done % 4096 < batch_size:
                    print(
                        f"[{name}] embedded={done:,}/{len(values):,} "
                        f"elapsed={time.time() - started:.1f}s",
                        flush=True,
                    )
            output[name] = np.concatenate(chunks, axis=0)
        return output
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def _clusters_from_embeddings(
    values: list[str],
    embeddings: np.ndarray,
    *,
    clustering: str,
    target_size: int = CLUSTER_TARGET_SIZE,
    max_labels_per_call: int | None = None,
) -> list[Cluster]:
    if len(values) != len(embeddings):
        raise ValueError("value and embedding counts differ")
    if not values:
        return []
    if target_size < 1:
        raise ValueError("cluster target size must be positive")
    if max_labels_per_call is not None and max_labels_per_call < 1:
        raise ValueError("maximum labels per call must be positive")
    cluster_count = math.ceil(len(values) / target_size)
    if cluster_count == 1:
        labels = np.zeros(len(values), dtype=np.int64)
    elif clustering == "lloyd":
        labels = KMeans(
            n_clusters=cluster_count,
            random_state=CLUSTER_RANDOM_SEED,
            n_init=10,
            algorithm="lloyd",
        ).fit_predict(embeddings)
    elif clustering == "minibatch":
        labels = MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=CLUSTER_RANDOM_SEED,
            n_init=MINIBATCH_N_INIT,
            max_iter=MINIBATCH_MAX_ITER,
            batch_size=MINIBATCH_SIZE,
        ).fit_predict(embeddings)
    else:
        raise ValueError(f"unsupported clustering algorithm {clustering!r}")
    clusters: list[Cluster] = []
    for label in sorted(set(int(item) for item in labels)):
        member_indices = np.flatnonzero(labels == label)
        chunk_size = max_labels_per_call or len(member_indices)
        for start in range(0, len(member_indices), chunk_size):
            members = tuple(
                values[index]
                for index in member_indices[start : start + chunk_size]
            )
            digest = hashlib.sha256(
                json.dumps(members, ensure_ascii=False).encode("utf-8")
            ).hexdigest()[:16]
            clusters.append(Cluster(cluster_id=f"cluster_{digest}", values=members))
    return sorted(clusters, key=lambda cluster: cluster.cluster_id)


def _cluster_item_ids(cluster: Cluster) -> dict[str, str]:
    return {f"v{index:04d}": value for index, value in enumerate(cluster.values)}


def _clean_bucket(
    value: Any, *, null_sentinel: str | None, bucket_pattern: str | None
) -> str | None:
    # JSON mode commonly serializes constrained numeric labels as numbers even
    # when the prompt calls them labels.  Numeric output is only permitted for
    # an extraction with an explicit canonical bucket pattern (currently the
    # direct-reaction 0-4 severity grade); free-form reconciliations remain
    # string-only.
    if (
        bucket_pattern
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
        value = str(value)
    if not isinstance(value, str):
        raise ValueError("bucket is not a string")
    bucket = re.sub(r"\s+", " ", value.strip()).casefold()
    if null_sentinel and bucket == null_sentinel.casefold():
        return None
    if not bucket:
        raise ValueError("bucket is empty")
    if len(bucket) > 120 or "\n" in bucket or "\r" in bucket:
        raise ValueError("bucket is not one concise label")
    if any(token in bucket for token in ("```", "{", "}", "->", "→")):
        raise ValueError("bucket contains non-label syntax")
    if bucket_pattern:
        if re.fullmatch(bucket_pattern, bucket) is None:
            raise ValueError("bucket does not match the required canonical syntax")
    elif FORBIDDEN_OUTPUT.search(bucket):
        raise ValueError("bucket contains an unknown-like value")
    return bucket


def _validate_response(
    content: str | None, *, item_ids: Mapping[str, str], extraction: ExtractionSpec
) -> dict[str, str | None]:
    try:
        payload = json.loads(content or "")
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"mapping"}:
        raise ValueError("response must contain only the mapping object")
    mapping = payload["mapping"]
    if not isinstance(mapping, dict) or set(mapping) != set(item_ids):
        actual = set(mapping) if isinstance(mapping, dict) else set()
        expected = set(item_ids)
        raise ValueError(
            f"mapping ID mismatch: missing={len(expected - actual)} extra={len(actual - expected)}"
        )
    return {
        item_id: _clean_bucket(
            mapping[item_id],
            null_sentinel=extraction.null_sentinel,
            bucket_pattern=extraction.bucket_pattern,
        )
        for item_id in item_ids
    }


def _empty_usage() -> dict[str, int]:
    return {
        "cached_input_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }


def _response_usage(response: Any) -> tuple[dict[str, int], bool]:
    usage = getattr(response, "usage", None)
    if usage is None or getattr(usage, "total_tokens", None) is None:
        return _empty_usage(), False
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return (
        {
            "cached_input_tokens": int(getattr(prompt_details, "cached_tokens", 0) or 0),
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "reasoning_tokens": int(
                getattr(completion_details, "reasoning_tokens", 0) or 0
            ),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        },
        True,
    )


class TokenLedger:
    def __init__(self, path: Path, *, model: str, reasoning_effort: str):
        self.path = path
        self.lock = threading.RLock()
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("ledger_version") != LEDGER_VERSION:
                raise ValueError("token ledger version mismatch")
            if payload.get("prompt_version") != PROMPT_VERSION:
                raise ValueError("token ledger prompt identity mismatch")
            if payload.get("model") != model or payload.get("reasoning_effort") != reasoning_effort:
                raise ValueError("token ledger model/reasoning identity mismatch")
            self.payload = payload
        else:
            self.payload = {
                "ledger_version": LEDGER_VERSION,
                "prompt_version": PROMPT_VERSION,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "lifetime_usage": _empty_usage(),
                "attempts": 0,
                "by_extraction": {},
                "status_counts": {},
            }

    def record(self, extraction_key: str, usage: Mapping[str, int], status: str) -> None:
        with self.lock:
            for key in _empty_usage():
                self.payload["lifetime_usage"][key] = int(
                    self.payload["lifetime_usage"].get(key) or 0
                ) + int(usage.get(key) or 0)
            section = self.payload["by_extraction"].setdefault(
                extraction_key, {"usage": _empty_usage(), "attempts": 0}
            )
            for key in _empty_usage():
                section["usage"][key] = int(section["usage"].get(key) or 0) + int(
                    usage.get(key) or 0
                )
            section["attempts"] = int(section.get("attempts") or 0) + 1
            self.payload["attempts"] = int(self.payload.get("attempts") or 0) + 1
            counts = self.payload["status_counts"]
            counts[status] = int(counts.get(status) or 0) + 1
            _atomic_json(self.path, self.payload)


class TokenBudget:
    def __init__(self, *, limit: int, ledger: TokenLedger):
        self.limit = limit
        self.ledger = ledger
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.accounting_error = threading.Event()
        self.non_retryable_error: str | None = None
        self.run_usage = _empty_usage()

    def before_attempt(self) -> None:
        with self.lock:
            if self.non_retryable_error is not None:
                raise NonRetryableAPIError(self.non_retryable_error)
            if self.accounting_error.is_set():
                raise UsageUnavailable("provider token usage was unavailable")
            if self.stop_event.is_set() or self.run_usage["total_tokens"] >= self.limit:
                self.stop_event.set()
                raise BudgetExhausted("per-invocation token budget reached")

    def record(
        self,
        extraction_key: str,
        usage: Mapping[str, int],
        *,
        status: str,
        usage_present: bool,
    ) -> None:
        with self.lock:
            self.ledger.record(extraction_key, usage, status)
            for key in _empty_usage():
                self.run_usage[key] += int(usage.get(key) or 0)
            if not usage_present:
                self.accounting_error.set()
                self.stop_event.set()
            elif self.run_usage["total_tokens"] >= self.limit:
                self.stop_event.set()

    def stop_for_non_retryable_api_error(self, message: str) -> None:
        with self.lock:
            if self.non_retryable_error is None:
                self.non_retryable_error = message
            self.stop_event.set()

    def summary(self) -> dict[str, Any]:
        return {
            "budget": self.limit,
            "run_usage": dict(self.run_usage),
            "lifetime_usage": dict(self.ledger.payload["lifetime_usage"]),
        }


class ClientHolder:
    def __init__(self, *, api_key_env: str | None):
        self.api_key_env = api_key_env
        self.lock = threading.Lock()
        self.client: Any = None

    def get(self) -> Any:
        with self.lock:
            if self.client is not None:
                return self.client
            if not self.api_key_env:
                raise RuntimeError("--api-key-env is required for paid extraction")
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"environment variable {self.api_key_env} is unset; its value is never read from a file"
                )
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key)
            return self.client


def _cache_identity(
    extraction: ExtractionSpec,
    cluster: Cluster,
    *,
    model: str,
    reasoning_effort: str,
) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "source_id": extraction.source_id,
        "input_column": extraction.input_column,
        "output_name": extraction.output_name,
        "output_column": extraction.output_column,
        "prompt": extraction.prompt,
        "null_sentinel": extraction.null_sentinel,
        "bucket_pattern": extraction.bucket_pattern,
        "cluster_id": cluster.cluster_id,
        "values": cluster.values,
        "model": model,
        "reasoning_effort": reasoning_effort,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return output
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                identity = str(record["identity"])
                output[identity] = record
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid cache line {path}:{line_number}: {exc}") from exc
    return output


def _append_cache(
    path: Path,
    *,
    identity: str,
    cluster: Cluster,
    mapping: Mapping[str, str | None],
    generation: Mapping[str, Any],
    lock: threading.Lock,
) -> None:
    record = json.dumps(
        {
            "generation": generation,
            "identity": identity,
            "mapping": mapping,
            "members": list(cluster.values),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(record + "\n")
            handle.flush()


def _query_cluster(
    client_holder: ClientHolder,
    *,
    extraction: ExtractionSpec,
    cluster: Cluster,
    model: str,
    reasoning_effort: str,
    max_retries: int,
    retry_delay: float,
    budget: TokenBudget,
) -> tuple[dict[str, str | None], dict[str, Any]]:
    item_ids = _cluster_item_ids(cluster)
    attempts: list[dict[str, Any]] = []
    last_error: Exception | None = None
    extraction_key = f"{extraction.source_id}/{extraction.output_name}"
    for attempt in range(1, max_retries + 1):
        budget.before_attempt()
        try:
            prompt = extraction.prompt or ""
            if attempt > 1:
                prompt += (
                    "\n\nThe previous response was invalid. Return every supplied ID exactly once "
                    "inside the single mapping object, with no extra keys. "
                    f"Validator feedback: {last_error}."
                )
            request: dict[str, Any] = {
                "model": model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "response_contract": (
                                    "Return exactly one JSON object containing only the mapping object."
                                ),
                                "items": [
                                    {"id": item_id, "value": value}
                                    for item_id, value in item_ids.items()
                                ]
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            }
            if reasoning_effort:
                request["reasoning_effort"] = reasoning_effort
            response = client_holder.get().chat.completions.create(**request)
            content = response.choices[0].message.content or ""
            usage, usage_present = _response_usage(response)
            try:
                mapping = _validate_response(
                    content, item_ids=item_ids, extraction=extraction
                )
            except Exception as exc:
                last_error = exc
                status = "validation_error"
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": status,
                        "usage": usage,
                        "response_sha256": hashlib.sha256(content.encode()).hexdigest(),
                        "validation_error": f"{type(exc).__name__}: {exc}",
                    }
                )
                budget.record(
                    extraction_key,
                    usage,
                    status=status,
                    usage_present=usage_present,
                )
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
                    continue
                break
            status = "valid"
            attempts.append(
                {
                    "attempt": attempt,
                    "status": status,
                    "usage": usage,
                    "response_sha256": hashlib.sha256(content.encode()).hexdigest(),
                }
            )
            budget.record(
                extraction_key,
                usage,
                status=status,
                usage_present=usage_present,
            )
            generation = {
                "attempt_count": len(attempts),
                "attempts": attempts,
                "usage": {
                    key: sum(int(item["usage"].get(key) or 0) for item in attempts)
                    for key in _empty_usage()
                },
            }
            return mapping, generation
        except (BudgetExhausted, UsageUnavailable, NonRetryableAPIError):
            raise
        except Exception as exc:
            last_error = exc
            status_code = getattr(exc, "status_code", None)
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "api_error",
                    "status_code": status_code,
                    "usage": _empty_usage(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            budget.record(
                extraction_key,
                _empty_usage(),
                status="api_error",
                usage_present=True,
            )
            text = str(exc).casefold()
            non_retryable = status_code in {400, 401, 403, 404} or (
                status_code == 429
                and any(
                    marker in text
                    for marker in (
                        "insufficient_quota",
                        "credit_balance_exhausted",
                        "no credits",
                    )
                )
            )
            if non_retryable:
                message = f"{type(exc).__name__}: {exc}"
                budget.stop_for_non_retryable_api_error(message)
                raise NonRetryableAPIError(message) from exc
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
    raise ClusterQueryFailure(
        f"cluster {cluster.cluster_id} failed after {len(attempts)} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def _map_clusters(
    *,
    client_holder: ClientHolder,
    extraction: ExtractionSpec,
    clusters: list[Cluster],
    model: str,
    reasoning_effort: str,
    workers: int,
    max_retries: int,
    retry_delay: float,
    cache_path: Path,
    budget: TokenBudget,
) -> dict[str, str | None]:
    cached = _load_cache(cache_path)
    cluster_mappings: dict[str, dict[str, str | None]] = {}
    pending: list[tuple[Cluster, str]] = []
    for cluster in clusters:
        identity = _cache_identity(
            extraction, cluster, model=model, reasoning_effort=reasoning_effort
        )
        record = cached.get(identity)
        item_ids = _cluster_item_ids(cluster)
        if record is None:
            pending.append((cluster, identity))
            continue
        if tuple(record.get("members") or ()) != cluster.values:
            raise ValueError(f"cached cluster members differ for {cluster.cluster_id}")
        raw_mapping = record.get("mapping")
        if not isinstance(raw_mapping, dict) or set(raw_mapping) != set(item_ids):
            raise ValueError(f"cached mapping ID mismatch for {cluster.cluster_id}")
        cluster_mappings[cluster.cluster_id] = {
            item_id: (
                None
                if raw_mapping[item_id] is None
                else _clean_bucket(
                    raw_mapping[item_id],
                    null_sentinel=extraction.null_sentinel,
                    bucket_pattern=extraction.bucket_pattern,
                )
            )
            for item_id in item_ids
        }
    print(
        f"[{extraction.source_id}/{extraction.output_name}] clusters={len(clusters):,} "
        f"cached={len(clusters) - len(pending):,} pending={len(pending):,}",
        flush=True,
    )
    failures: list[Exception] = []
    stopped = False
    api_stopped = False
    cache_lock = threading.Lock()
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _query_cluster,
                client_holder,
                extraction=extraction,
                cluster=cluster,
                model=model,
                reasoning_effort=reasoning_effort,
                max_retries=max_retries,
                retry_delay=retry_delay,
                budget=budget,
            ): (cluster, identity)
            for cluster, identity in pending
        }
        for future in as_completed(futures):
            cluster, identity = futures[future]
            try:
                mapping, generation = future.result()
            except BudgetExhausted:
                stopped = True
                continue
            except UsageUnavailable:
                stopped = True
                continue
            except NonRetryableAPIError:
                api_stopped = True
                continue
            except Exception as exc:
                failures.append(exc)
                continue
            cluster_mappings[cluster.cluster_id] = mapping
            _append_cache(
                cache_path,
                identity=identity,
                cluster=cluster,
                mapping=mapping,
                generation=generation,
                lock=cache_lock,
            )
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(pending):
                print(
                    f"[{extraction.source_id}/{extraction.output_name}] "
                    f"completed={completed:,}/{len(pending):,}",
                    flush=True,
                )
    if api_stopped or budget.non_retryable_error is not None:
        raise NonRetryableAPIError(
            budget.non_retryable_error or "provider rejected the request"
        )
    if failures:
        raise RuntimeError(
            f"{len(failures)} cluster(s) failed; first={failures[0]}"
        ) from failures[0]
    if budget.accounting_error.is_set():
        raise UsageUnavailable("provider response omitted total_tokens; cached valid results were retained")
    if stopped or len(cluster_mappings) != len(clusters):
        remaining = len(clusters) - len(cluster_mappings)
        raise BudgetExhausted(
            f"token budget stopped {extraction.source_id}/{extraction.output_name}; "
            f"remaining_clusters={remaining}"
        )
    output: dict[str, str | None] = {}
    for cluster in clusters:
        item_ids = _cluster_item_ids(cluster)
        mapping = cluster_mappings[cluster.cluster_id]
        for item_id, raw_value in item_ids.items():
            output[raw_value] = mapping[item_id]
    return dict(sorted(output.items(), key=lambda item: (item[0].casefold(), item[0])))


def _source_identity(
    source_id: str, *, model: str, reasoning_effort: str, embedding_model: str
) -> str:
    spec = SOURCE_SPECS[source_id]
    extraction_payload = []
    for extraction in _source_extractions(source_id):
        reviewed_hash = (
            _file_sha256(extraction.reviewed_mapping_path)
            if extraction.reviewed_mapping_path is not None
            else None
        )
        extraction_identity = dict(extraction.__dict__)
        if extraction.cluster_target_size == CLUSTER_TARGET_SIZE:
            extraction_identity.pop("cluster_target_size")
        if extraction.max_labels_per_call is None:
            extraction_identity.pop("max_labels_per_call")
        extraction_payload.append(
            {
                **extraction_identity,
                "reviewed_mapping_path": str(extraction.reviewed_mapping_path or ""),
                "reviewed_mapping_sha256": reviewed_hash,
            }
        )
    payload = {
        "prompt_version": PROMPT_VERSION,
        "source_id": source_id,
        "source_sha256": _file_sha256(spec.parquet_path),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "embedding_model": embedding_model,
        "cluster_target_size": CLUSTER_TARGET_SIZE,
        "cluster_seed": CLUSTER_RANDOM_SEED,
        "minibatch": {
            "batch_size": MINIBATCH_SIZE,
            "n_init": MINIBATCH_N_INIT,
            "max_iter": MINIBATCH_MAX_ITER,
        },
        "extractions": extraction_payload,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()


def _snapshot_path(cache_dir: Path, source_id: str) -> Path:
    return cache_dir / f"{source_id}.complete.json"


def _load_snapshot(
    cache_dir: Path,
    source_id: str,
    *,
    model: str,
    reasoning_effort: str,
    embedding_model: str,
) -> dict[str, Any] | None:
    path = _snapshot_path(cache_dir, source_id)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = _source_identity(
        source_id,
        model=model,
        reasoning_effort=reasoning_effort,
        embedding_model=embedding_model,
    )
    if payload.get("identity") != expected:
        return None
    mapping = payload.get("mapping")
    return mapping if isinstance(mapping, dict) else None


def _load_snapshot_from_models(
    cache_dir: Path,
    source_id: str,
    *,
    models: list[str],
    reasoning_effort: str,
    embedding_model: str,
) -> tuple[dict[str, Any], str] | None:
    for model in dict.fromkeys(models):
        mapping = _load_snapshot(
            cache_dir,
            source_id,
            model=model,
            reasoning_effort=reasoning_effort,
            embedding_model=embedding_model,
        )
        if mapping is not None:
            return mapping, model
    return None


def _build_source(
    args: argparse.Namespace,
    *,
    source_id: str,
    cache_dir: Path,
    client_holder: ClientHolder,
    budget: TokenBudget,
) -> dict[str, Any]:
    cached_snapshot = _load_snapshot(
        cache_dir,
        source_id,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        embedding_model=args.embedding_model,
    )
    if cached_snapshot is not None:
        print(f"[{source_id}] complete source snapshot cached", flush=True)
        return cached_snapshot
    spec = SOURCE_SPECS[source_id]
    extractions = _source_extractions(source_id)
    columns = sorted({item.input_column for item in extractions})
    frame = pd.read_parquet(spec.parquet_path, columns=columns)
    values_by_column = {
        column: _distinct_values(frame[column]) for column in columns
    }
    clustered_columns = {
        extraction.input_column: extraction
        for extraction in extractions
        if extraction.label_source == "llm"
    }
    groups = {
        f"{source_id}/{column}": values_by_column[column]
        for column in clustered_columns
    }
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = (
        _embed_value_groups(
            groups,
            model_name=args.embedding_model,
            batch_size=args.embedding_batch_size,
            device=device,
        )
        if groups
        else {}
    )
    clusters_by_config: dict[tuple[str, str, int, int | None], list[Cluster]] = {}
    nested: dict[str, dict[str, dict[str, str | None]]] = {}
    for extraction in extractions:
        values = values_by_column[extraction.input_column]
        if extraction.label_source == "reviewed_table":
            if extraction.reviewed_mapping_path is None:
                raise ValueError(f"{source_id}/{extraction.output_name} has no reviewed table")
            mapping = load_reviewed_mapping(extraction.reviewed_mapping_path)
            uncovered = set(values) - set(mapping)
            if uncovered:
                raise ValueError(
                    f"reviewed table lacks {len(uncovered)} values; first={sorted(uncovered)[0]!r}"
                )
            resolved = {value: mapping[value] for value in values}
        else:
            cluster_config = (
                extraction.input_column,
                extraction.clustering,
                extraction.cluster_target_size,
                extraction.max_labels_per_call,
            )
            clusters = clusters_by_config.get(cluster_config)
            if clusters is None:
                clusters = _clusters_from_embeddings(
                    values,
                    embeddings[f"{source_id}/{extraction.input_column}"],
                    clustering=extraction.clustering,
                    target_size=extraction.cluster_target_size,
                    max_labels_per_call=extraction.max_labels_per_call,
                )
                clusters_by_config[cluster_config] = clusters
            cache_path = cache_dir / f"{source_id}__{extraction.output_name}.jsonl"
            resolved = _map_clusters(
                client_holder=client_holder,
                extraction=extraction,
                clusters=clusters,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                workers=args.workers,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
                cache_path=cache_path,
                budget=budget,
            )
        nested.setdefault(extraction.input_column, {})[
            extraction.output_column
        ] = resolved
        counts = pd.Series(list(resolved.values()), dtype="string").value_counts(
            dropna=False
        )
        print(
            f"[{source_id}/{extraction.output_name}] values={len(resolved):,} "
            f"labels={counts.index.dropna().nunique():,} null={sum(v is None for v in resolved.values()):,}",
            flush=True,
        )
    _atomic_json(
        _snapshot_path(cache_dir, source_id),
        {
            "identity": _source_identity(
                source_id,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                embedding_model=args.embedding_model,
            ),
            "mapping": nested,
        },
    )
    return nested


def dry_run(args: argparse.Namespace) -> None:
    selected = args.sources or list(SOURCE_SPECS)
    total = 0
    for source_id in selected:
        spec = SOURCE_SPECS[source_id]
        columns = sorted({item.current_input_column for item in spec.outputs})
        frame = pd.read_parquet(spec.parquet_path, columns=columns)
        for extraction in _source_extractions(source_id):
            values = _distinct_values(frame[extraction.input_column])
            calls = (
                math.ceil(len(values) / extraction.cluster_target_size)
                if extraction.label_source == "llm"
                else 0
            )
            total += calls
            print(
                f"[{source_id}/{extraction.output_name}] method={extraction.label_source} "
                f"column={extraction.input_column} distinct={len(values):,} "
                f"clustering={extraction.clustering} minimum_calls={calls:,} "
                f"max_labels_per_call={extraction.max_labels_per_call or 'unbounded'}"
            )
    print(f"minimum API calls for selected sources: {total:,}")
    print("dry run complete; no API client or embedding model was loaded")


def run(args: argparse.Namespace) -> int:
    if args.dry_run:
        dry_run(args)
        return 0
    destination = Path(args.output)
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {destination}; pass --overwrite to replace it")
    cache_dir = destination.with_suffix(destination.suffix + ".cache")
    ledger_path = Path(args.token_ledger) if args.token_ledger else destination.with_suffix(
        destination.suffix + ".token_ledger.json"
    )
    ledger = TokenLedger(
        ledger_path, model=args.model, reasoning_effort=args.reasoning_effort
    )
    budget = TokenBudget(limit=args.token_budget, ledger=ledger)
    client_holder = ClientHolder(api_key_env=args.api_key_env)
    selected = args.sources or list(SOURCE_SPECS)
    try:
        for source_id in selected:
            _build_source(
                args,
                source_id=source_id,
                cache_dir=cache_dir,
                client_holder=client_holder,
                budget=budget,
            )
    except BudgetExhausted as exc:
        print(f"budget stop: {exc}", flush=True)
        print(json.dumps(budget.summary(), indent=2, sort_keys=True), flush=True)
        return BUDGET_EXIT_CODE
    except UsageUnavailable as exc:
        print(f"token accounting stop: {exc}", flush=True)
        print(json.dumps(budget.summary(), indent=2, sort_keys=True), flush=True)
        return USAGE_EXIT_CODE
    except NonRetryableAPIError as exc:
        print(f"provider stop: {exc}", flush=True)
        print(json.dumps(budget.summary(), indent=2, sort_keys=True), flush=True)
        return API_EXIT_CODE

    first_stage: dict[str, Any] = {}
    missing: list[str] = []
    accepted_snapshot_models = [args.model, *args.compatible_snapshot_model]
    for source_id in SOURCE_SPECS:
        loaded = _load_snapshot_from_models(
            cache_dir,
            source_id,
            models=accepted_snapshot_models,
            reasoning_effort=args.reasoning_effort,
            embedding_model=args.embedding_model,
        )
        if loaded is None:
            missing.append(source_id)
        else:
            snapshot, snapshot_model = loaded
            first_stage[source_id] = snapshot
            print(
                f"[{source_id}] finalization snapshot model={snapshot_model}",
                flush=True,
            )
    if missing:
        print(
            f"selected phase complete; final mapping not published; missing_sources={missing}",
            flush=True,
        )
        print(json.dumps(budget.summary(), indent=2, sort_keys=True), flush=True)
        return 0

    payload = build_mapping(first_stage)
    audit = validate_mapping(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    print(json.dumps(budget.summary(), indent=2, sort_keys=True), flush=True)
    print(f"wrote {destination}", flush=True)
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--compatible-snapshot-model",
        action="append",
        default=[],
        help=(
            "Additional model identity accepted only when loading already-complete "
            "source snapshots for final publication; may be repeated."
        ),
    )
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--token-ledger")
    parser.add_argument("--api-key-env")
    parser.add_argument(
        "--sources", nargs="+", choices=tuple(SOURCE_SPECS), help="Paid phase sources"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.embedding_batch_size < 1 or args.workers < 1 or args.max_retries < 1:
        parser.error("batch size, workers, and retries must be positive")
    if args.retry_delay < 0:
        parser.error("retry delay cannot be negative")
    if args.token_budget < 1:
        parser.error("--token-budget must be positive")
    if not args.dry_run and not args.api_key_env:
        parser.error("--api-key-env is required for paid extraction or cache finalization")
    return args


def main(argv: list[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
