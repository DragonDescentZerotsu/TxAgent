"""Deterministic embedding-clustered LLM reconciliation for auxiliary values.

Task modules declare the raw source column, target auxiliary field, and prompt.
This shared implementation embeds each distinct raw value, clusters related
values, asks an OpenAI-compatible model for one complete mapping per cluster,
and publishes the exact tuple-key format consumed by ``AuxiliaryMetadataAttacher``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.processing.evidence_library.shared.v2.normalization.cleaning import clean_scalar


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CLUSTER_TARGET_SIZE = 100
DEFAULT_CLUSTER_RANDOM_SEED = 20260801
DEFAULT_NULL_LIKE = frozenset(
    {
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
)
_FORBIDDEN_OUTPUT = re.compile(
    r"(?<![a-z0-9])(?:unknown|unmapped|null|none|n/?a|not stated|"
    r"not specified|unspecified)(?![a-z0-9])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AuxiliaryExtractionSpec:
    source_id: str
    input_path: Path
    input_column: str
    output_field: str
    prompt: str
    null_sentinel: str | None = None


@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    values: tuple[str, ...]


class ClusterQueryFailure(RuntimeError):
    def __init__(self, message: str, *, audit: dict[str, Any]):
        super().__init__(message)
        self.audit = audit


def distinct_values(
    values: Iterable[Any], *, null_like: Iterable[str] = DEFAULT_NULL_LIKE
) -> list[str]:
    excluded = {str(value).strip().casefold() for value in null_like}
    output = {
        str(value).strip()
        for value in values
        if value is not None
        and not (isinstance(value, float) and math.isnan(value))
        and str(value).strip().casefold() not in excluded
    }
    return sorted(output, key=lambda value: (value.casefold(), value))


def embed_values(
    values: Sequence[str],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 512,
    device: str = "auto",
    progress_label: str = "embedding",
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(resolved_device)
    chunks: list[np.ndarray] = []
    started = time.monotonic()
    try:
        for start in range(0, len(values), batch_size):
            batch = list(values[start : start + batch_size])
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
            with torch.inference_mode():
                hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            chunks.append(pooled.cpu().numpy())
            done = min(start + batch_size, len(values))
            if done == len(values) or done == batch_size or done % 4096 < batch_size:
                print(
                    f"[{progress_label}] completed={done:,}/{len(values):,} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    finally:
        del model
        if resolved_device.startswith("cuda"):
            torch.cuda.empty_cache()
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def cluster_values(
    values: Sequence[str],
    embeddings: np.ndarray,
    *,
    target_size: int = DEFAULT_CLUSTER_TARGET_SIZE,
    max_size: int | None = None,
    random_seed: int = DEFAULT_CLUSTER_RANDOM_SEED,
) -> list[Cluster]:
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    if len(values) != len(embeddings):
        raise ValueError("value and embedding counts differ")
    if target_size < 1 or (max_size is not None and max_size < 1):
        raise ValueError("cluster target and maximum sizes must be positive")
    if not values:
        return []
    cluster_count = math.ceil(len(values) / target_size)
    if cluster_count == 1:
        labels = np.zeros(len(values), dtype=np.int64)
    else:
        # Shared nodes commonly expose more BLAS threads than the linked
        # OpenBLAS build can safely track. Bound only this numerical section;
        # clustering remains deterministic under the frozen random seed.
        with threadpool_limits(limits=16):
            labels = KMeans(
                n_clusters=cluster_count,
                random_state=random_seed,
                n_init=10,
                algorithm="lloyd",
            ).fit_predict(embeddings)
    member_indices: list[np.ndarray] = []
    for label in range(cluster_count):
        indices = np.flatnonzero(labels == label)
        if max_size is None or len(indices) <= max_size:
            member_indices.append(indices)
        else:
            member_indices.extend(
                _bounded_embedding_subclusters(
                    embeddings,
                    indices,
                    max_size=max_size,
                    random_seed=random_seed,
                )
            )
    output: list[Cluster] = []
    for indices in member_indices:
        members = tuple(values[index] for index in indices)
        digest = hashlib.sha256(
            json.dumps(members, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        output.append(Cluster(f"cluster_{digest}", members))
    if max_size is not None and any(len(cluster.values) > max_size for cluster in output):
        raise AssertionError("bounded clustering produced an oversized cluster")
    return sorted(output, key=lambda cluster: cluster.cluster_id)


def _bounded_embedding_subclusters(
    embeddings: np.ndarray,
    indices: np.ndarray,
    *,
    max_size: int,
    random_seed: int,
) -> list[np.ndarray]:
    """Deterministically split one oversized KMeans neighborhood.

    KMeans controls the average cluster size, not its maximum.  API workflows
    need a true request bound, so oversized neighborhoods are recursively
    subclustered in the same embedding space.  A stable index chunk is used
    only when identical embeddings make KMeans unable to divide a group.
    """
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    if len(indices) <= max_size:
        return [indices]
    child_count = math.ceil(len(indices) / max_size)
    with threadpool_limits(limits=16):
        labels = KMeans(
            n_clusters=child_count,
            random_state=random_seed,
            n_init=10,
            algorithm="lloyd",
        ).fit_predict(embeddings[indices])
    groups = [indices[np.flatnonzero(labels == label)] for label in range(child_count)]
    groups = [group for group in groups if len(group)]
    if len(groups) <= 1:
        return [indices[start : start + max_size] for start in range(0, len(indices), max_size)]
    output: list[np.ndarray] = []
    for offset, group in enumerate(groups):
        output.extend(
            _bounded_embedding_subclusters(
                embeddings,
                group,
                max_size=max_size,
                random_seed=random_seed + offset + 1,
            )
        )
    return output


def clean_bucket(value: Any, *, null_sentinel: str | None) -> str | None:
    if not isinstance(value, str):
        raise ValueError("bucket is not a string")
    bucket = re.sub(r"\s+", " ", value.strip()).casefold()
    if null_sentinel and bucket == null_sentinel.casefold():
        return None
    if not bucket or len(bucket) > 120 or "\n" in bucket or "\r" in bucket:
        raise ValueError("bucket is not one concise label")
    if any(token in bucket for token in ("```", "{", "}", "->", "→")):
        raise ValueError("bucket contains non-label syntax")
    if _FORBIDDEN_OUTPUT.search(bucket):
        raise ValueError("bucket contains an unknown-like value")
    return bucket


def validate_response(
    content: str | None,
    *,
    item_ids: Mapping[str, str],
    null_sentinel: str | None,
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
            f"mapping ID mismatch: missing={len(expected - actual)} "
            f"extra={len(actual - expected)}"
        )
    return {
        item_id: clean_bucket(mapping[item_id], null_sentinel=null_sentinel)
        for item_id in item_ids
    }


def build_clustered_auxiliary_mapping(
    *,
    specs: Sequence[AuxiliaryExtractionSpec],
    output_path: str | Path,
    mapping_version: str,
    prompt_version: str,
    api_key_loader: Callable[[], str],
    model: str,
    reasoning_effort: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_batch_size: int = 512,
    device: str = "auto",
    cluster_target_size: int = DEFAULT_CLUSTER_TARGET_SIZE,
    max_cluster_size: int | None = None,
    random_seed: int = DEFAULT_CLUSTER_RANDOM_SEED,
    workers: int = 16,
    max_retries: int = 5,
    retry_delay: float = 2.0,
    overwrite: bool = False,
    base_url: str | None = None,
    reconcile_cleaned_key_conflicts: bool = True,
) -> dict[str, Any]:
    """Build and atomically publish a complete reconciled mapping."""
    from openai import OpenAI

    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output exists: {destination}; pass overwrite=True")
    if not specs:
        raise ValueError("at least one auxiliary extraction spec is required")
    cache_dir = destination.parent / f".{destination.name}.cache"
    frames = {
        path: pd.read_parquet(path)
        for path in sorted({spec.input_path for spec in specs}, key=str)
    }
    inventories: dict[tuple[str, Path, str], list[str]] = {}
    clusters_by_inventory: dict[tuple[str, Path, str], list[Cluster]] = {}
    for spec in specs:
        frame = frames[spec.input_path]
        if spec.input_column not in frame.columns:
            raise KeyError(f"{spec.input_path} lacks {spec.input_column!r}")
        key = (spec.source_id, spec.input_path, spec.input_column)
        if key in inventories:
            continue
        values = distinct_values(frame[spec.input_column].tolist())
        inventories[key] = values
        cluster_cache = cache_dir / (
            f"clusters__{spec.source_id}__{spec.input_column}.json"
        )
        cached_clusters = _load_cluster_cache(
            cluster_cache,
            values=values,
            embedding_model=embedding_model,
            target_size=cluster_target_size,
            max_size=max_cluster_size,
            random_seed=random_seed,
        )
        if cached_clusters is not None:
            clusters_by_inventory[key] = cached_clusters
            print(
                f"[{spec.source_id}/{spec.input_column}] "
                f"loaded_clusters={len(cached_clusters):,}",
                flush=True,
            )
        else:
            embeddings = embed_values(
                values,
                model_name=embedding_model,
                batch_size=embedding_batch_size,
                device=device,
                progress_label=f"{spec.source_id}/{spec.input_column}",
            )
            clusters_by_inventory[key] = cluster_values(
                values,
                embeddings,
                target_size=cluster_target_size,
                max_size=max_cluster_size,
                random_seed=random_seed,
            )
            _write_cluster_cache(
                cluster_cache,
                clusters_by_inventory[key],
                values=values,
                embedding_model=embedding_model,
                target_size=cluster_target_size,
                max_size=max_cluster_size,
                random_seed=random_seed,
            )

    client_kwargs: dict[str, Any] = {"api_key": api_key_loader()}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    output_sources: dict[str, dict[str, Any]] = defaultdict(dict)
    generation_sections: dict[str, Any] = {}
    for spec in specs:
        key = (spec.source_id, spec.input_path, spec.input_column)
        clusters = clusters_by_inventory[key]
        mapping, audit = _map_clusters(
            client=client,
            spec=spec,
            clusters=clusters,
            model=model,
            reasoning_effort=reasoning_effort,
            prompt_version=prompt_version,
            workers=workers,
            max_retries=max_retries,
            retry_delay=retry_delay,
            cache_path=cache_dir / f"{spec.source_id}__{spec.output_field}.jsonl",
        )
        serialized_mapping = {
            json.dumps([raw], ensure_ascii=False, separators=(",", ":")): value
            for raw, value in mapping.items()
        }
        # ``distinct_values`` intentionally excludes source null sentinels.
        # The attacher nevertheless needs one explicit cleaned-null key so
        # None/unknown/not-stated records join deterministically without an
        # invented label.
        serialized_mapping[json.dumps([None], separators=(",", ":"))] = None
        output_sources[spec.source_id][spec.output_field] = {
            "source_columns": [spec.input_column],
            "mapping": serialized_mapping,
        }
        generation_sections[f"{spec.source_id}/{spec.output_field}"] = audit

    conflict_audit = (
        _reconcile_cleaned_key_conflicts(
            client=client,
            specs=specs,
            output_sources=output_sources,
            cache_dir=cache_dir,
            model=model,
            reasoning_effort=reasoning_effort,
            prompt_version=prompt_version,
            workers=workers,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        if reconcile_cleaned_key_conflicts
        else {
            "contract_version": "no_model_global_reconciliation.v1",
            "conflicting_cleaned_keys": None,
            "sections": {},
            "validations": {"global_model_pass_disabled": True},
        }
    )

    expected_sources = {spec.source_id for spec in specs}
    if set(output_sources) != expected_sources:
        raise AssertionError("published source inventory differs from extraction specs")
    payload = {
        "mapping_version": mapping_version,
        "sources": {
            source: dict(sorted(outputs.items()))
            for source, outputs in sorted(output_sources.items())
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    audit_path = destination.with_suffix(destination.suffix + ".generation.json")
    audit_payload = {
        "mapping_version": mapping_version,
        "prompt_version": prompt_version,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "embedding_model": embedding_model,
        "cluster_target_size": cluster_target_size,
        "max_cluster_size": max_cluster_size,
        "cluster_random_seed": random_seed,
        "base_url_configured": bool(base_url),
        "global_model_reconciliation_enabled": reconcile_cleaned_key_conflicts,
        "mapping_sha256": _sha256(destination),
        "sections": generation_sections,
        "cleaned_key_conflict_reconciliation": conflict_audit,
    }
    audit_path.write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit_payload


def _reconcile_cleaned_key_conflicts(
    *,
    client: Any,
    specs: Sequence[AuxiliaryExtractionSpec],
    output_sources: Mapping[str, dict[str, Any]],
    cache_dir: Path,
    model: str,
    reasoning_effort: str,
    prompt_version: str,
    workers: int,
    max_retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    """Resolve only labels that disagree after source-key cleaning.

    Distinct Unicode/raw strings can collapse to one authoritative attachment
    key. Embedding clusters are independent, so a final global pass asks the
    same model for one label per conflicting cleaned key and applies it to all
    raw aliases. No majority vote or lexical tie-break is used.
    """
    spec_by_section = {(spec.source_id, spec.output_field): spec for spec in specs}
    section_audits: dict[str, Any] = {}
    total_conflicts = 0
    for source_id, outputs in sorted(output_sources.items()):
        for output_field, section in sorted(outputs.items()):
            spec = spec_by_section[(source_id, output_field)]
            raw_mapping = section["mapping"]
            grouped: dict[Any, list[tuple[str, str, str | None]]] = defaultdict(list)
            for serialized, label in raw_mapping.items():
                decoded = json.loads(serialized)
                raw_value = str(decoded[0])
                cleaned = clean_scalar(raw_value)
                if (
                    isinstance(cleaned, str)
                    and cleaned.casefold() in DEFAULT_NULL_LIKE
                ):
                    cleaned = None
                grouped[cleaned].append((serialized, raw_value, label))
            conflicts = [
                (cleaned, rows)
                for cleaned, rows in sorted(grouped.items(), key=lambda item: str(item[0]))
                if len({row[2] for row in rows}) > 1
            ]
            total_conflicts += len(conflicts)
            name = f"{source_id}/{output_field}"
            if not conflicts:
                section_audits[name] = {
                    "conflicting_cleaned_keys": 0,
                    "raw_aliases_reconciled": 0,
                    "usage": _empty_usage(),
                }
                continue
            descriptions = tuple(
                json.dumps(
                    {
                        "cleaned_source_key": cleaned,
                        "raw_aliases": [row[1] for row in rows],
                        "provisional_labels": sorted(
                            {row[2] for row in rows}, key=lambda value: str(value)
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                for cleaned, rows in conflicts
            )
            digest = hashlib.sha256(
                json.dumps(descriptions, ensure_ascii=False).encode("utf-8")
            ).hexdigest()[:16]
            conflict_spec = AuxiliaryExtractionSpec(
                source_id=source_id,
                input_path=spec.input_path,
                input_column=spec.input_column,
                output_field=output_field,
                null_sentinel=spec.null_sentinel,
                prompt=(
                    spec.prompt
                    + "\n\nGlobal cleaned-key conflict reconciliation: each input value is "
                    "a JSON object describing raw aliases that collapse to exactly one "
                    "source lookup key and their provisional labels. Return exactly one "
                    "canonical label for each input ID. All aliases in that object will "
                    "receive that one label; do not return an explanation."
                ),
            )
            resolved, audit = _map_clusters(
                client=client,
                spec=conflict_spec,
                clusters=(Cluster(f"conflict_{digest}", descriptions),),
                model=model,
                reasoning_effort=reasoning_effort,
                prompt_version=f"{prompt_version}.cleaned_key_conflicts.v1",
                workers=min(workers, 1),
                max_retries=max_retries,
                retry_delay=retry_delay,
                cache_path=cache_dir / f"conflicts__{source_id}__{output_field}.jsonl",
            )
            for description, (_, rows) in zip(descriptions, conflicts, strict=True):
                label = resolved[description]
                for serialized, _, _ in rows:
                    raw_mapping[serialized] = label
            section_audits[name] = {
                "conflicting_cleaned_keys": len(conflicts),
                "raw_aliases_reconciled": sum(len(rows) for _, rows in conflicts),
                "usage": audit["usage"],
                "cluster_generation": audit["cluster_generation"],
            }
    return {
        "contract_version": "globally_reconciled_cleaned_keys.v1",
        "conflicting_cleaned_keys": total_conflicts,
        "sections": section_audits,
        "validations": {"all_conflicts_resolved_by_model": True},
    }


def reconciliation_plan(
    specs: Sequence[AuxiliaryExtractionSpec],
    *,
    cluster_target_size: int = DEFAULT_CLUSTER_TARGET_SIZE,
) -> dict[str, Any]:
    sections: dict[str, Any] = {}
    total = 0
    frames: dict[Path, pd.DataFrame] = {}
    for spec in specs:
        frame = frames.setdefault(spec.input_path, pd.read_parquet(spec.input_path))
        values = distinct_values(frame[spec.input_column].tolist())
        calls = math.ceil(len(values) / cluster_target_size)
        total += calls
        sections[f"{spec.source_id}/{spec.output_field}"] = {
            "input_column": spec.input_column,
            "distinct_values": len(values),
            "planned_clusters": calls,
        }
    return {"sections": sections, "planned_api_calls": total}


def _map_clusters(
    *,
    client: Any,
    spec: AuxiliaryExtractionSpec,
    clusters: Sequence[Cluster],
    model: str,
    reasoning_effort: str,
    prompt_version: str,
    workers: int,
    max_retries: int,
    retry_delay: float,
    cache_path: Path,
) -> tuple[dict[str, str | None], dict[str, Any]]:
    cache = _load_cache(cache_path)
    lock = threading.Lock()
    results: dict[str, dict[str, str | None]] = {}
    generation: dict[str, Any] = {}
    pending: list[tuple[Cluster, str]] = []
    for cluster in clusters:
        identity = _cache_identity(
            spec=spec,
            cluster=cluster,
            model=model,
            reasoning_effort=reasoning_effort,
            prompt_version=prompt_version,
        )
        cached = cache.get(identity)
        if cached is None:
            pending.append((cluster, identity))
        else:
            results[cluster.cluster_id] = dict(cached["mapping"])
            generation[cluster.cluster_id] = dict(cached.get("generation") or {})
    print(
        f"[{spec.source_id}/{spec.output_field}] clusters={len(clusters):,} "
        f"cached={len(clusters) - len(pending):,} pending={len(pending):,}",
        flush=True,
    )
    failures: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _query_cluster,
                client,
                spec=spec,
                cluster=cluster,
                model=model,
                reasoning_effort=reasoning_effort,
                max_retries=max_retries,
                retry_delay=retry_delay,
            ): (cluster, identity)
            for cluster, identity in pending
        }
        completed = 0
        for future in as_completed(futures):
            cluster, identity = futures[future]
            try:
                mapping, audit = future.result()
            except Exception as exc:
                failures.append((cluster.cluster_id, exc))
                continue
            results[cluster.cluster_id] = mapping
            generation[cluster.cluster_id] = audit
            _append_cache(
                cache_path,
                identity=identity,
                mapping=mapping,
                generation=audit,
                lock=lock,
            )
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(pending):
                print(
                    f"[{spec.source_id}/{spec.output_field}] "
                    f"completed={completed:,}/{len(pending):,}",
                    flush=True,
                )
    if failures:
        cluster_id, error = failures[0]
        raise RuntimeError(
            f"{len(failures)} cluster requests failed; first={cluster_id}: {error}"
        ) from error
    output: dict[str, str | None] = {}
    for cluster in clusters:
        item_ids = _item_ids(cluster)
        cluster_mapping = results[cluster.cluster_id]
        if set(cluster_mapping) != set(item_ids):
            raise ValueError(f"cached mapping ID mismatch for {cluster.cluster_id}")
        for item_id, raw in item_ids.items():
            output[raw] = clean_bucket(
                cluster_mapping[item_id], null_sentinel=spec.null_sentinel
            ) if cluster_mapping[item_id] is not None else None
    expected = {value for cluster in clusters for value in cluster.values}
    if set(output) != expected:
        raise ValueError("final raw-value coverage mismatch")
    usage = {
        key: sum(
            int((entry.get("usage") or {}).get(key) or 0)
            for entry in generation.values()
        )
        for key in _empty_usage()
    }
    return dict(sorted(output.items(), key=lambda item: (item[0].casefold(), item[0]))), {
        "input_column": spec.input_column,
        "raw_values": len(output),
        "clusters": len(clusters),
        "null_mappings": sum(value is None for value in output.values()),
        "usage": usage,
        "cluster_generation": generation,
    }


def _query_cluster(
    client: Any,
    *,
    spec: AuxiliaryExtractionSpec,
    cluster: Cluster,
    model: str,
    reasoning_effort: str,
    max_retries: int,
    retry_delay: float,
) -> tuple[dict[str, str | None], dict[str, Any]]:
    item_ids = _item_ids(cluster)
    user = json.dumps(
        {"items": [{"id": key, "value": value} for key, value in item_ids.items()]},
        ensure_ascii=False,
    )
    attempts: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            prompt = spec.prompt
            prompt += (
                "\n\nReturn valid JSON only, using exactly the requested mapping object."
            )
            if attempt > 1:
                prompt += (
                    "\n\nThe previous response was invalid. Return exactly one JSON "
                    "mapping for every supplied ID, with no extra IDs or text."
                )
            request: dict[str, Any] = {
                "model": model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user},
                ],
            }
            if reasoning_effort:
                request["reasoning_effort"] = reasoning_effort
            response = client.chat.completions.create(**request)
            content = response.choices[0].message.content or ""
            usage = _response_usage(response)
            try:
                mapping = validate_response(
                    content,
                    item_ids=item_ids,
                    null_sentinel=spec.null_sentinel,
                )
            except Exception as exc:
                last_error = exc
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "validation_error",
                        "response_sha256": hashlib.sha256(content.encode()).hexdigest(),
                        "error": f"{type(exc).__name__}: {exc}",
                        "usage": usage,
                    }
                )
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
                    continue
                break
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "valid",
                    "response_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "served_model": str(getattr(response, "model", "") or ""),
                    "usage": usage,
                }
            )
            return mapping, _generation_audit(attempts)
        except Exception as exc:
            last_error = exc
            status_code = getattr(exc, "status_code", None)
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "api_error",
                    "status_code": status_code,
                    "error": f"{type(exc).__name__}: {exc}",
                    "usage": _empty_usage(),
                }
            )
            if _terminal_api_error(exc):
                break
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
    audit = _generation_audit(attempts)
    raise ClusterQueryFailure(
        f"cluster {cluster.cluster_id} failed after {len(attempts)} attempts: "
        f"{type(last_error).__name__}: {last_error}",
        audit=audit,
    ) from last_error


def _terminal_api_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code in {400, 401, 403, 404}:
        return True
    body = getattr(error, "body", None)
    code = ""
    if isinstance(body, Mapping):
        code = str(body.get("code") or "")
    code = code or str(getattr(error, "code", "") or "")
    return status_code == 429 and code in {
        "billing_hard_limit_reached",
        "credit_balance_exhausted",
        "insufficient_quota",
    }


def _item_ids(cluster: Cluster) -> dict[str, str]:
    return {f"v{index:04d}": value for index, value in enumerate(cluster.values)}


def _cache_identity(
    *,
    spec: AuxiliaryExtractionSpec,
    cluster: Cluster,
    model: str,
    reasoning_effort: str,
    prompt_version: str,
) -> str:
    payload = {
        "prompt_version": prompt_version,
        "prompt": spec.prompt,
        "source_id": spec.source_id,
        "input_column": spec.input_column,
        "output_field": spec.output_field,
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
                output[str(record["identity"])] = {
                    "mapping": dict(record["mapping"]),
                    "generation": dict(record.get("generation") or {}),
                }
            except Exception as exc:
                raise ValueError(f"invalid cache line {path}:{line_number}: {exc}") from exc
    return output


def _load_cluster_cache(
    path: Path,
    *,
    values: Sequence[str],
    embedding_model: str,
    target_size: int,
    max_size: int | None = None,
    random_seed: int,
) -> list[Cluster] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = _cluster_cache_identity(
        values,
        embedding_model=embedding_model,
        target_size=target_size,
        max_size=max_size,
        random_seed=random_seed,
    )
    if payload.get("identity") != expected:
        return None
    clusters = [
        Cluster(str(row["cluster_id"]), tuple(str(value) for value in row["values"]))
        for row in payload.get("clusters") or []
    ]
    if {value for cluster in clusters for value in cluster.values} != set(values):
        raise ValueError(f"cluster cache value coverage mismatch: {path}")
    return clusters


def _write_cluster_cache(
    path: Path,
    clusters: Sequence[Cluster],
    *,
    values: Sequence[str],
    embedding_model: str,
    target_size: int,
    max_size: int | None = None,
    random_seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "identity": _cluster_cache_identity(
            values,
            embedding_model=embedding_model,
            target_size=target_size,
            max_size=max_size,
            random_seed=random_seed,
        ),
        "clusters": [
            {"cluster_id": cluster.cluster_id, "values": list(cluster.values)}
            for cluster in clusters
        ],
        "max_cluster_size": max_size,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _cluster_cache_identity(
    values: Sequence[str],
    *,
    embedding_model: str,
    target_size: int,
    max_size: int | None = None,
    random_seed: int,
) -> str:
    payload = {
        "values": list(values),
        "embedding_model": embedding_model,
        "target_size": target_size,
        "random_seed": random_seed,
        "algorithm": "sklearn.KMeans(n_init=10,algorithm=lloyd)",
    }
    # Preserve historical cache identities when no hard request bound was
    # requested.  The new field participates only in bounded workflows.
    if max_size is not None:
        payload["max_size"] = max_size
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _append_cache(
    path: Path,
    *,
    identity: str,
    mapping: Mapping[str, str | None],
    generation: Mapping[str, Any],
    lock: threading.Lock,
) -> None:
    record = json.dumps(
        {"generation": generation, "identity": identity, "mapping": mapping},
        ensure_ascii=False,
        sort_keys=True,
    )
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(record + "\n")
            handle.flush()


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
    prompt = getattr(usage, "prompt_tokens_details", None)
    completion = getattr(usage, "completion_tokens_details", None)
    return {
        "cached_input_tokens": int(getattr(prompt, "cached_tokens", 0) or 0),
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(completion, "reasoning_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _generation_audit(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "attempt_count": len(attempts),
        "attempts": list(attempts),
        "usage": {
            key: sum(int((attempt.get("usage") or {}).get(key) or 0) for attempt in attempts)
            for key in _empty_usage()
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "AuxiliaryExtractionSpec",
    "Cluster",
    "ClusterQueryFailure",
    "DEFAULT_CLUSTER_RANDOM_SEED",
    "DEFAULT_CLUSTER_TARGET_SIZE",
    "DEFAULT_EMBEDDING_MODEL",
    "build_clustered_auxiliary_mapping",
    "clean_bucket",
    "cluster_values",
    "distinct_values",
    "embed_values",
    "reconciliation_plan",
    "validate_response",
]
