#!/usr/bin/env python3
"""Build the original Fa/FG/Fh context and species value mappings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer

from tools.chembl_tool.common.llm_client import openai_client
from data.processing.evidence_library.versions.v8.prompts import PROMPT_ROOT


REPO_ROOT = Path(__file__).resolve().parents[8]
DATA_ROOT = REPO_ROOT / "data/raw/starling/bioavailability_ma"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "globally_reconciled_auxiliary_value_mapping.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "medium"
CLUSTER_TARGET_SIZE = 100
CLUSTER_RANDOM_SEED = 20260801
PROMPT_VERSION = "starling_embedding_bucket_mapping.v2"
NULL_LIKE = {
    "",
    "-",
    "n/a",
    "na",
    "nan",
    "none",
    "not specified",
    "not stated",
    "null",
    "unknown",
    "unspecified",
}
FORBIDDEN_OUTPUT = re.compile(
    r"(?<![a-z0-9])(?:unknown|unmapped|null|none|n/?a|not stated|not specified|unspecified)(?![a-z0-9])",
    re.IGNORECASE,
)


PROMPT_REGISTRY_PATH = PROMPT_ROOT / "auxiliary_canonicalization/bioavailability.json"


def _load_prompt_registry() -> dict[str, Any]:
    payload = json.loads(PROMPT_REGISTRY_PATH.read_text(encoding="utf-8"))
    expected = {
        ("fa", "global_context"),
        ("fa", "global_species_context"),
        ("fg", "global_context"),
        ("fg", "global_species_context"),
        ("fh", "global_context"),
        ("fh", "global_species_context"),
    }
    prompts = payload.get("prompts")
    if payload.get("registry_version") != "starling_original_auxiliary_prompts.v1":
        raise ValueError("original auxiliary prompt registry version mismatch")
    if not isinstance(prompts, dict):
        raise ValueError("original auxiliary prompt registry is missing prompts")
    actual = {
        (source_id, output_name)
        for source_id, outputs in prompts.items()
        for output_name in outputs
    }
    if actual != expected:
        raise ValueError(
            f"original auxiliary prompt registry must contain exactly six prompts: {actual}"
        )
    return payload


_PROMPT_REGISTRY = _load_prompt_registry()
FA_CONTEXT_PROMPT = _PROMPT_REGISTRY["prompts"]["fa"]["global_context"]["prompt"]
FA_SPECIES_PROMPT = _PROMPT_REGISTRY["prompts"]["fa"]["global_species_context"]["prompt"]
FG_CONTEXT_PROMPT = _PROMPT_REGISTRY["prompts"]["fg"]["global_context"]["prompt"]
FG_SPECIES_PROMPT = _PROMPT_REGISTRY["prompts"]["fg"]["global_species_context"]["prompt"]
FH_CONTEXT_PROMPT = _PROMPT_REGISTRY["prompts"]["fh"]["global_context"]["prompt"]
FH_SPECIES_PROMPT = _PROMPT_REGISTRY["prompts"]["fh"]["global_species_context"]["prompt"]


@dataclass(frozen=True)
class ExtractionSpec:
    input_column: str
    output_column: str
    prompt: str
    null_sentinel: str | None = None
    bucket_pattern: str | None = None


@dataclass(frozen=True)
class SourceSpec:
    input_path: Path
    extractions: tuple[ExtractionSpec, ...]


SOURCE_SPECS = {
    "fa": SourceSpec(
        input_path=DATA_ROOT / "Fa/extractions.parquet",
        extractions=(
            ExtractionSpec("assay_system", "canonical_context", FA_CONTEXT_PROMPT),
            ExtractionSpec(
                "biological_context",
                "canonical_species",
                FA_SPECIES_PROMPT,
                null_sentinel="no species",
            ),
        ),
    ),
    "fg": SourceSpec(
        input_path=DATA_ROOT / "Fg/extractions.parquet",
        extractions=(
            ExtractionSpec("assay_system", "canonical_context", FG_CONTEXT_PROMPT),
            ExtractionSpec(
                "assay_system",
                "canonical_species",
                FG_SPECIES_PROMPT,
                null_sentinel="no species",
            ),
        ),
    ),
    "fh": SourceSpec(
        input_path=DATA_ROOT / "Fh/extractions.parquet",
        extractions=(
            ExtractionSpec("assay_system", "canonical_context", FH_CONTEXT_PROMPT),
            ExtractionSpec(
                "species",
                "canonical_species",
                FH_SPECIES_PROMPT,
                null_sentinel="no species",
            ),
        ),
    ),
}


@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    values: tuple[str, ...]


class ClusterQueryFailure(RuntimeError):
    """A failed cluster request carrying non-secret attempt and usage audit."""

    def __init__(self, message: str, *, audit: dict[str, Any]):
        super().__init__(message)
        self.audit = audit


def _distinct_values(series: pd.Series) -> list[str]:
    values = {
        str(value).strip()
        for value in series.dropna().tolist()
        if str(value).strip().casefold() not in NULL_LIKE
    }
    return sorted(values, key=lambda value: (value.casefold(), value))


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def _embed_values(
    values: list[str],
    *,
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    try:
        return _embed_values_with_encoder(
            values,
            tokenizer=tokenizer,
            model=model,
            batch_size=batch_size,
            device=device,
        )
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def _embed_value_groups(
    groups: dict[str, list[str]],
    *,
    model_name: str,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Embed multiple named inventories with one resident encoder instance."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    try:
        return {
            name: _embed_values_with_encoder(
                values,
                tokenizer=tokenizer,
                model=model,
                batch_size=batch_size,
                device=device,
                progress_label=name,
            )
            for name, values in groups.items()
        }
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def _embed_values_with_encoder(
    values: list[str],
    *,
    tokenizer: Any,
    model: Any,
    batch_size: int,
    device: str,
    progress_label: str = "embedding",
) -> np.ndarray:
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
                f"[{progress_label}] completed={done:,}/{len(values):,} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    return np.concatenate(chunks, axis=0)


def _clusters_from_embeddings(values: list[str], embeddings: np.ndarray) -> list[Cluster]:
    if len(values) != len(embeddings):
        raise ValueError("value and embedding counts differ")
    if not values:
        return []
    cluster_count = math.ceil(len(values) / CLUSTER_TARGET_SIZE)
    if cluster_count == 1:
        labels = np.zeros(len(values), dtype=np.int64)
    else:
        labels = KMeans(
            n_clusters=cluster_count,
            random_state=CLUSTER_RANDOM_SEED,
            n_init=10,
            algorithm="lloyd",
        ).fit_predict(embeddings)
    clusters: list[Cluster] = []
    for label in range(cluster_count):
        members = tuple(values[index] for index in np.flatnonzero(labels == label))
        digest = hashlib.sha256(
            json.dumps(members, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        clusters.append(Cluster(cluster_id=f"cluster_{digest}", values=members))
    return sorted(clusters, key=lambda cluster: cluster.cluster_id)


def _cluster_item_ids(cluster: Cluster) -> dict[str, str]:
    return {f"v{index:04d}": value for index, value in enumerate(cluster.values)}


def _clean_bucket(
    value: Any,
    *,
    null_sentinel: str | None,
    bucket_pattern: str | None = None,
) -> str | None:
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
    content: str | None,
    *,
    item_ids: dict[str, str],
    null_sentinel: str | None,
    bucket_pattern: str | None = None,
) -> dict[str, str | None]:
    try:
        payload = json.loads(content or "")
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"mapping"}:
        raise ValueError("response must contain only the mapping object")
    mapping = payload["mapping"]
    if not isinstance(mapping, dict):
        raise ValueError("mapping is not an object")
    expected = set(item_ids)
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            f"mapping ID mismatch: missing={len(expected - actual)} extra={len(actual - expected)}"
        )
    return {
        item_id: _clean_bucket(
            mapping[item_id],
            null_sentinel=null_sentinel,
            bucket_pattern=bucket_pattern,
        )
        for item_id in item_ids
    }


def _cache_identity(
    *,
    extraction: ExtractionSpec,
    cluster: Cluster,
    model: str,
    reasoning_effort: str,
) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "prompt": extraction.prompt,
        "input_column": extraction.input_column,
        "output_column": extraction.output_column,
        "cluster_id": cluster.cluster_id,
        "values": cluster.values,
        "model": model,
        "reasoning_effort": reasoning_effort,
    }
    if extraction.bucket_pattern is not None:
        payload["bucket_pattern"] = extraction.bucket_pattern
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _load_cache_records(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                identity = str(record["identity"])
                cache[identity] = {
                    "mapping": dict(record["mapping"]),
                    "generation": dict(record.get("generation") or {}),
                }
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid checkpoint line {line_number}: {exc}") from exc
    return cache


def _load_cache(path: Path) -> dict[str, dict[str, str | None]]:
    return {
        identity: dict(record["mapping"])
        for identity, record in _load_cache_records(path).items()
    }


def _append_cache(
    path: Path,
    *,
    identity: str,
    mapping: dict[str, str | None],
    lock: threading.Lock,
) -> None:
    record = json.dumps(
        {"identity": identity, "mapping": mapping},
        ensure_ascii=False,
        sort_keys=True,
    )
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(record + "\n")
            handle.flush()


def _append_audited_cache(
    path: Path,
    *,
    identity: str,
    mapping: dict[str, str | None],
    generation: dict[str, Any],
    lock: threading.Lock,
) -> None:
    record = json.dumps(
        {
            "generation": generation,
            "identity": identity,
            "mapping": mapping,
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
    client: OpenAI,
    *,
    extraction: ExtractionSpec,
    cluster: Cluster,
    model: str,
    reasoning_effort: str,
    max_retries: int,
    retry_delay: float,
) -> dict[str, str | None]:
    mapping, _ = _query_cluster_with_audit(
        client,
        extraction=extraction,
        cluster=cluster,
        model=model,
        reasoning_effort=reasoning_effort,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    return mapping


def _query_cluster_with_audit(
    client: OpenAI,
    *,
    extraction: ExtractionSpec,
    cluster: Cluster,
    model: str,
    reasoning_effort: str,
    max_retries: int,
    retry_delay: float,
) -> tuple[dict[str, str | None], dict[str, Any]]:
    """Query one cluster while retaining usage from valid and invalid attempts."""
    item_ids = _cluster_item_ids(cluster)
    user_payload = {
        "items": [
            {"id": item_id, "value": value}
            for item_id, value in item_ids.items()
        ]
    }
    last_error: Exception | None = None
    attempts: list[dict[str, Any]] = []
    rate_limit_count = 0
    for attempt in range(1, max_retries + 1):
        try:
            prompt = extraction.prompt
            if attempt > 1:
                prompt += (
                    "\n\nYour previous response was invalid. Return exactly one JSON mapping "
                    "for every supplied ID, with no missing or additional IDs."
                )
            request: dict[str, Any] = {
                "model": model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False),
                    },
                ],
            }
            if reasoning_effort:
                request["reasoning_effort"] = reasoning_effort
            response = client.chat.completions.create(
                **request,
            )
            content = response.choices[0].message.content or ""
            usage = _response_usage(response)
            try:
                mapping = _validate_response(
                    content,
                    item_ids=item_ids,
                    null_sentinel=extraction.null_sentinel,
                    bucket_pattern=extraction.bucket_pattern,
                )
            except Exception as exc:
                last_error = exc
                attempts.append(
                    {
                        "attempt": attempt,
                        "response_sha256": hashlib.sha256(
                            content.encode("utf-8")
                        ).hexdigest(),
                        "status": "validation_error",
                        "usage": usage,
                        "validation_error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
                    continue
                break
            attempts.append(
                {
                    "attempt": attempt,
                    "response_sha256": hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest(),
                    "status": "valid",
                    "usage": usage,
                }
            )
            return mapping, _generation_audit(
                attempts,
                rate_limit_count=rate_limit_count,
            )
        except Exception as exc:
            last_error = exc
            status_code = getattr(exc, "status_code", None)
            if status_code == 429:
                rate_limit_count += 1
            attempts.append(
                {
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "status": "api_error",
                    "status_code": status_code,
                    "usage": _empty_usage(),
                }
            )
            if _is_non_retryable_request_error(exc):
                break
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
    audit = _generation_audit(attempts, rate_limit_count=rate_limit_count)
    raise ClusterQueryFailure(
        f"cluster {cluster.cluster_id} failed after {len(attempts)} attempts: "
        f"{type(last_error).__name__}: {last_error}",
        audit=audit,
    ) from last_error


def _empty_usage() -> dict[str, int]:
    return {
        "cached_input_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }


def _response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return {
        "cached_input_tokens": int(
            getattr(prompt_details, "cached_tokens", 0) or 0
        ),
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "reasoning_tokens": int(
            getattr(completion_details, "reasoning_tokens", 0) or 0
        ),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _generation_audit(
    attempts: list[dict[str, Any]], *, rate_limit_count: int
) -> dict[str, Any]:
    usage = {
        key: sum(int(attempt["usage"].get(key) or 0) for attempt in attempts)
        for key in _empty_usage()
    }
    return {
        "attempt_count": len(attempts),
        "attempts": attempts,
        "rate_limit_count": rate_limit_count,
        "usage": usage,
    }


def _is_non_retryable_request_error(error: Exception) -> bool:
    """Do not multiply permanent model, schema, authentication, or quota errors."""
    status_code = getattr(error, "status_code", None)
    text = str(error).casefold()
    if status_code in {400, 401, 403, 404}:
        return True
    return status_code == 429 and any(
        marker in text
        for marker in ("insufficient_quota", "credit_balance_exhausted", "no credits")
    )


def _map_clusters(
    *,
    client: OpenAI,
    extraction: ExtractionSpec,
    clusters: list[Cluster],
    model: str,
    reasoning_effort: str,
    workers: int,
    max_retries: int,
    retry_delay: float,
    cache_path: Path,
) -> dict[str, str | None]:
    cached = _load_cache(cache_path)
    lock = threading.Lock()
    pending: list[tuple[Cluster, str]] = []
    cluster_mappings: dict[str, dict[str, str | None]] = {}
    for cluster in clusters:
        identity = _cache_identity(
            extraction=extraction,
            cluster=cluster,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        item_ids = _cluster_item_ids(cluster)
        if identity in cached:
            mapping = cached[identity]
            if set(mapping) != set(item_ids):
                raise ValueError(f"cached mapping ID mismatch for {cluster.cluster_id}")
            cluster_mappings[cluster.cluster_id] = {
                item_id: _clean_bucket(
                    mapping[item_id],
                    null_sentinel=extraction.null_sentinel,
                    bucket_pattern=extraction.bucket_pattern,
                )
                if mapping[item_id] is not None
                else None
                for item_id in item_ids
            }
        else:
            pending.append((cluster, identity))
    print(
        f"[{extraction.output_column}] clusters={len(clusters):,} "
        f"cached={len(clusters) - len(pending):,} pending={len(pending):,}",
        flush=True,
    )
    failures: list[tuple[Cluster, Exception]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _query_cluster,
                client,
                extraction=extraction,
                cluster=cluster,
                model=model,
                reasoning_effort=reasoning_effort,
                max_retries=max_retries,
                retry_delay=retry_delay,
            ): (cluster, identity)
            for cluster, identity in pending
        }
        for future in as_completed(futures):
            cluster, identity = futures[future]
            try:
                mapping = future.result()
            except Exception as exc:
                failures.append((cluster, exc))
                print(
                    f"[{extraction.output_column}] failures={len(failures):,}",
                    flush=True,
                )
                continue
            cluster_mappings[cluster.cluster_id] = mapping
            _append_cache(
                cache_path,
                identity=identity,
                mapping=mapping,
                lock=lock,
            )
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(pending):
                print(
                    f"[{extraction.output_column}] completed={completed:,}/{len(pending):,}",
                    flush=True,
                )
    if failures:
        cluster, error = failures[0]
        raise RuntimeError(
            f"{len(failures)} clusters failed; first={cluster.cluster_id}: {error}"
        ) from error

    output: dict[str, str | None] = {}
    for cluster in clusters:
        item_ids = _cluster_item_ids(cluster)
        mapping = cluster_mappings[cluster.cluster_id]
        for item_id, raw_value in item_ids.items():
            output[raw_value] = mapping[item_id]
    expected_values = {value for cluster in clusters for value in cluster.values}
    if set(output) != expected_values:
        raise ValueError("final raw-value coverage mismatch")
    return dict(sorted(output.items(), key=lambda item: (item[0].casefold(), item[0])))


def _print_cluster_distribution(name: str, clusters: list[Cluster]) -> None:
    sizes = np.array([len(cluster.values) for cluster in clusters], dtype=np.int64)
    print(
        json.dumps(
            {
                "extraction": name,
                "clusters": len(clusters),
                "mean_size": float(sizes.mean()),
                "median_size": float(np.median(sizes)),
                "p90_size": float(np.quantile(sizes, 0.90)),
                "p95_size": float(np.quantile(sizes, 0.95)),
                "max_size": int(sizes.max()),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _print_mapping_audit(
    extraction: ExtractionSpec,
    mapping: dict[str, str | None],
) -> None:
    counts = pd.Series(list(mapping.values()), dtype="string").value_counts(dropna=False)
    print(
        f"[{extraction.output_column}] raw_values={len(mapping):,} "
        f"buckets={counts.index.dropna().nunique():,} null={sum(value is None for value in mapping.values()):,}",
        flush=True,
    )
    print(f"[{extraction.output_column}] top_bucket_sizes", flush=True)
    print(counts.head(25).to_string(), flush=True)
    rng = random.Random(CLUSTER_RANDOM_SEED)
    sample = rng.sample(list(mapping), min(50, len(mapping)))
    for raw_value in sample:
        print(
            json.dumps(
                {
                    "extraction": extraction.output_column,
                    "raw_value": raw_value,
                    "bucket": mapping[raw_value],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )


def _build_source_mapping(
    args: argparse.Namespace,
    *,
    source: str,
    destination: Path,
) -> dict[str, Any]:
    source_spec = SOURCE_SPECS[source]
    frame = pd.read_parquet(source_spec.input_path)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    client = openai_client()
    cache_path = destination.with_suffix(
        destination.suffix + f".{source}.partial.jsonl"
    )
    nested_output: dict[str, dict[str, dict[str, dict[str, str | None]]]] = {
        source: {}
    }
    clustered_columns: dict[str, list[Cluster]] = {}
    for extraction in source_spec.extractions:
        values = _distinct_values(frame[extraction.input_column])
        clusters = clustered_columns.get(extraction.input_column)
        if clusters is None:
            print(
                f"[{source}/{extraction.input_column}] embedding "
                f"{len(values):,} distinct values on {device}",
                flush=True,
            )
            embeddings = _embed_values(
                values,
                model_name=args.embedding_model,
                batch_size=args.embedding_batch_size,
                device=device,
            )
            clusters = _clusters_from_embeddings(values, embeddings)
            clustered_columns[extraction.input_column] = clusters
        elif {value for cluster in clusters for value in cluster.values} != set(values):
            raise ValueError(
                f"reused cluster inventory differs for {extraction.input_column}"
            )
        _print_cluster_distribution(extraction.output_column, clusters)
        mapping = _map_clusters(
            client=client,
            extraction=extraction,
            clusters=clusters,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            workers=args.workers,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            cache_path=cache_path,
        )
        nested_output[source].setdefault(extraction.input_column, {})[
            extraction.output_column
        ] = mapping
        _print_mapping_audit(extraction, mapping)
    cache_path.unlink(missing_ok=True)
    return nested_output[source]


def run(args: argparse.Namespace) -> None:
    """Run the six original prompts and publish only the complete global map."""
    from data.processing.evidence_library.versions.v8.tasks.bioavailability_ma.data_processing.auxiliary_mapping_helpers.reconciliation import (
        build_mapping,
        validate_mapping,
    )

    destination = Path(args.output)
    if destination.exists() and not args.overwrite:
        raise FileExistsError(
            f"output exists: {destination}; pass --overwrite to replace it"
        )
    first_stage = {
        source: _build_source_mapping(
            args,
            source=source,
            destination=destination,
        )
        for source in SOURCE_SPECS
    }
    payload = build_mapping(first_stage)
    audit = validate_mapping(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    print(f"wrote {destination}", flush=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.embedding_batch_size < 1 or args.workers < 1 or args.max_retries < 1:
        parser.error("batch size, workers, and retries must be positive")
    if args.retry_delay < 0:
        parser.error("retry delay cannot be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    run(_parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
