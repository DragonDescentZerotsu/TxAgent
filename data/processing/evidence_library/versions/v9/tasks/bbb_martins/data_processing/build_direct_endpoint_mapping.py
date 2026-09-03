"""Build the human-gated Direct BBB endpoint-normalization proposal.

Only ``direct_bbb.quant_metric`` is sent to GPT.  The other BBB endpoint and
context vocabularies are normalized by reviewed deterministic rules in
``starling_endpoint_normalization``.  This command writes a local proposal,
durable cluster provenance, and review packets; it never writes the approved
runtime mapping used by normalized-v6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from data.processing.evidence_library.shared.v2.clustered_auxiliary_mapping import (
    AuxiliaryExtractionSpec,
    Cluster,
    DEFAULT_CLUSTER_RANDOM_SEED,
    DEFAULT_EMBEDDING_MODEL,
    _cache_identity,
    build_clustered_auxiliary_mapping,
    distinct_values,
    reconciliation_plan,
)
from data.processing.evidence_library.versions.v9.prompts import PROMPT_ROOT
from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    endpoint_inventory_hash,
    file_sha256,
    normalize_endpoint_name,
)


REPO_ROOT = Path(__file__).resolve().parents[8]
SOURCE_PATH = REPO_ROOT / "data/raw/starling/bbb_martins/Direct_BBB/records.parquet"
DEFAULT_ROOT = Path(__file__).with_name("direct_endpoint_normalization_v1")
DEFAULT_LOCAL_MAPPING = DEFAULT_ROOT / "provisional/local_endpoint_mapping.json"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "medium"
CLUSTER_TARGET_SIZE = 500
MAX_VALUES_PER_LOCAL_REQUEST = 500
CLUSTER_RANDOM_SEED = DEFAULT_CLUSTER_RANDOM_SEED
PROMPT_VERSION = "bbb_martins_direct_endpoint_local.v1"
MAPPING_VERSION = "bbb_martins_direct_endpoint.local_proposal.v1"
PROVENANCE_VERSION = "bbb_martins_direct_endpoint_local_provenance.v1"
API_KEY_ENV = "OPENAI_API_KEY"
BASE_URL_ENV = "OPENAI_BASE_URL"

DIRECT_ENDPOINT_PROMPT_PATH = PROMPT_ROOT / "endpoint_canonicalization/bbb_direct_v1.txt"
DIRECT_ENDPOINT_PROMPT = DIRECT_ENDPOINT_PROMPT_PATH.read_text(encoding="utf-8").strip()


def extraction_spec() -> AuxiliaryExtractionSpec:
    return AuxiliaryExtractionSpec(
        source_id="direct_bbb",
        input_path=SOURCE_PATH,
        input_column="quant_metric",
        output_field="canonical_endpoint",
        prompt=DIRECT_ENDPOINT_PROMPT,
    )


def _api_key() -> str:
    value = os.environ.get(API_KEY_ENV)
    if not value:
        raise RuntimeError(
            f"{API_KEY_ENV} is unset; inject the private key into the child environment"
        )
    return value


def _base_url() -> str | None:
    return os.environ.get(BASE_URL_ENV) or None


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _accepted_attempt(generation: dict[str, Any]) -> dict[str, Any]:
    attempts = generation.get("attempts") or []
    accepted = [row for row in attempts if row.get("status") == "valid"]
    if len(accepted) != 1:
        raise ValueError("each completed cluster must have exactly one valid response")
    return accepted[0]


def materialize_provenance(*, mapping_path: str | Path) -> dict[str, Any]:
    """Make ignored response caches reviewable without persisting response text."""
    mapping_path = Path(mapping_path)
    artifact_root = mapping_path.parent.parent
    audit_path = mapping_path.with_suffix(mapping_path.suffix + ".generation.json")
    cache_dir = mapping_path.parent / f".{mapping_path.name}.cache"
    cluster_path = cache_dir / "clusters__direct_bbb__quant_metric.json"
    response_path = cache_dir / "direct_bbb__canonical_endpoint.jsonl"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    cluster_payload = json.loads(cluster_path.read_text(encoding="utf-8"))
    response_records = {
        row["identity"]: row
        for row in (
            json.loads(line)
            for line in response_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    section = audit["sections"]["direct_bbb/canonical_endpoint"]
    rows: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []
    raw_seen: set[str] = set()
    for cluster_order, cluster in enumerate(cluster_payload["clusters"]):
        values = tuple(cluster["values"])
        if len(values) > MAX_VALUES_PER_LOCAL_REQUEST:
            raise ValueError(f"oversized Direct endpoint cluster: {cluster['cluster_id']}")
        identity = _cache_identity(
            spec=extraction_spec(),
            cluster=Cluster(cluster["cluster_id"], values),
            model=DEFAULT_MODEL,
            reasoning_effort=DEFAULT_REASONING_EFFORT,
            prompt_version=PROMPT_VERSION,
        )
        cached = response_records.get(identity)
        if cached is None:
            raise ValueError(f"missing response cache for {cluster['cluster_id']}")
        mapping = cached["mapping"]
        item_ids = [f"v{index:04d}" for index in range(len(values))]
        if set(mapping) != set(item_ids):
            raise ValueError(f"item coverage mismatch for {cluster['cluster_id']}")
        generation = cached["generation"]
        if section["cluster_generation"].get(cluster["cluster_id"]) != generation:
            raise ValueError(f"generation audit mismatch for {cluster['cluster_id']}")
        accepted = _accepted_attempt(generation)
        packet_assignments: list[dict[str, Any]] = []
        for item_order, (item_id, raw_value) in enumerate(zip(item_ids, values, strict=True)):
            if raw_value in raw_seen:
                raise ValueError(f"Direct endpoint appears in multiple clusters: {raw_value!r}")
            raw_seen.add(raw_value)
            row = {
                "source_id": "direct_bbb",
                "input_column": "quant_metric",
                "output_field": "canonical_endpoint",
                "cluster_order": cluster_order,
                "cluster_id": cluster["cluster_id"],
                "item_order": item_order,
                "item_id": item_id,
                "raw_value": raw_value,
                "provisional_endpoint": mapping[item_id],
                "cache_identity": identity,
                "served_model": accepted.get("served_model"),
                "response_sha256": accepted.get("response_sha256"),
                "generation_attempt_count": generation.get("attempt_count"),
            }
            rows.append(row)
            packet_assignments.append(
                {
                    "item_id": item_id,
                    "raw_value": raw_value,
                    "provisional_endpoint": mapping[item_id],
                }
            )
        packet_path = artifact_root / "review_packets" / f"{cluster['cluster_id']}.json"
        _write_json(
            packet_path,
            {
                "packet_version": "bbb_direct_endpoint_local_cluster_review.v1",
                "cluster_id": cluster["cluster_id"],
                "assignment_count": len(packet_assignments),
                "assignments": packet_assignments,
            },
        )
        packet_rows.append(
            {
                "cluster_id": cluster["cluster_id"],
                "path": str(packet_path.relative_to(artifact_root)),
                "assignments": len(packet_assignments),
                "sha256": _sha256(packet_path),
            }
        )
    frame = pd.DataFrame(rows)
    provenance_path = artifact_root / "provenance/cluster_assignments.parquet"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(provenance_path, index=False)
    labels = (
        frame.groupby("provisional_endpoint", sort=True, dropna=False)
        .agg(
            assignment_count=("raw_value", "size"),
            cluster_count=("cluster_id", "nunique"),
            raw_examples=("raw_value", lambda values: json.dumps(list(values)[:20], ensure_ascii=False)),
        )
        .reset_index()
    )
    label_catalog_path = artifact_root / "provenance/provisional_label_catalog.parquet"
    labels.to_parquet(label_catalog_path, index=False)
    source_frame = pd.read_parquet(SOURCE_PATH, columns=["quant_metric"])
    normalized_inventory = sorted(
        set(normalize_endpoint_name(value) or "" for value in source_frame["quant_metric"])
    )
    non_null_inventory = distinct_values(source_frame["quant_metric"].tolist())
    validations = {
        "all_non_null_inventory_values_assigned_once": set(non_null_inventory) == raw_seen,
        "all_clusters_at_most_500_values": all(row["assignments"] <= 500 for row in packet_rows),
        "every_cluster_has_one_review_packet": len(packet_rows) == len(cluster_payload["clusters"]),
        "requested_model_is_gpt_5_4_mini": audit.get("model") == DEFAULT_MODEL,
        "global_gpt_reconciliation_disabled": not audit.get("global_model_reconciliation_enabled"),
        "publication_is_human_blocked": True,
    }
    if not all(validations.values()):
        raise ValueError(f"Direct endpoint provenance validation failed: {validations}")
    manifest = {
        "artifact_version": PROVENANCE_VERSION,
        "publication_status": "awaiting_human_approval",
        "publication_blockers": ["global_subagent_reconciliation", "human_approval_not_recorded"],
        "human_approved": False,
        "source": {
            "path": str(SOURCE_PATH),
            "sha256": _sha256(SOURCE_PATH),
            "normalized_inventory_count_including_missing": len(normalized_inventory),
            "normalized_inventory_sha256": endpoint_inventory_hash(normalized_inventory),
            "non_null_raw_values": len(non_null_inventory),
        },
        "generation": {
            "mapping_path": str(mapping_path),
            "mapping_sha256": _sha256(mapping_path),
            "generation_audit_path": str(audit_path),
            "generation_audit_sha256": _sha256(audit_path),
            "model": audit.get("model"),
            "reasoning_effort": audit.get("reasoning_effort"),
            "embedding_model": audit.get("embedding_model"),
            "cluster_target_size": audit.get("cluster_target_size"),
            "max_values_per_local_request": audit.get("max_cluster_size"),
            "cluster_random_seed": audit.get("cluster_random_seed"),
        },
        "counts": {
            "local_clusters": len(packet_rows),
            "assignments": len(frame),
            "provisional_labels": len(labels),
            "max_cluster_size": max((row["assignments"] for row in packet_rows), default=0),
        },
        "artifacts": {
            "cluster_assignments": {"path": str(provenance_path), "sha256": _sha256(provenance_path)},
            "provisional_label_catalog": {"path": str(label_catalog_path), "sha256": _sha256(label_catalog_path)},
            "review_packets": packet_rows,
        },
        "validations": validations,
    }
    _write_json(artifact_root / "provenance/provisional_manifest.json", manifest)
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_LOCAL_MAPPING))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--materialize-provenance-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.embedding_batch_size, args.workers, args.max_retries) < 1:
        parser.error("batch size, workers, and retries must be positive")
    if args.retry_delay < 0:
        parser.error("retry delay cannot be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output = Path(args.output)
    spec = extraction_spec()
    if args.dry_run:
        plan = reconciliation_plan((spec,), cluster_target_size=CLUSTER_TARGET_SIZE)
        plan.update(
            {
                "model": DEFAULT_MODEL,
                "reasoning_effort": DEFAULT_REASONING_EFFORT,
                "max_values_per_local_request": MAX_VALUES_PER_LOCAL_REQUEST,
                "publication_status": "proposal_only",
            }
        )
        print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
        return 0
    if not args.materialize_provenance_only:
        build_clustered_auxiliary_mapping(
            specs=(spec,),
            output_path=output,
            mapping_version=MAPPING_VERSION,
            prompt_version=PROMPT_VERSION,
            api_key_loader=_api_key,
            base_url=_base_url(),
            model=DEFAULT_MODEL,
            reasoning_effort=DEFAULT_REASONING_EFFORT,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            device=args.device,
            cluster_target_size=CLUSTER_TARGET_SIZE,
            max_cluster_size=MAX_VALUES_PER_LOCAL_REQUEST,
            random_seed=CLUSTER_RANDOM_SEED,
            workers=args.workers,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            overwrite=args.overwrite,
            reconcile_cleaned_key_conflicts=False,
        )
    manifest = materialize_provenance(mapping_path=output)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
