"""Freeze disjoint AMES candidate-selection inputs without making API calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.build_measurement_resolution_mapping import (
    RequestBatch,
    TaskConfig,
    candidate_rows,
    load_endpoint_profile,
    plan_batches,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_candidate_resolution import (
    CANDIDATE_CONTRACT_VERSION,
    candidate_set_sha256,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_measurement_selection import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CREDENTIAL_ENV,
    DEEPSEEK_MODEL,
    DEEPSEEK_PROVIDER,
    DEFAULT_CLEANED_RECORDS,
    DEFAULT_PROFILE_PATH,
    ENDPOINT_CONCURRENCY_BUDGET,
    MAPPING_VERSION,
    MAX_COMPLETION_TOKENS,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)

PLAN_VERSION = "ames_candidate_selection_partitions.v2"
SELECTION_CONFIG_MODULE = (
    "data.processing.evidence_library.versions.v10.tasks.ames."
    "starling_measurement_selection"
)
TOKEN_PROJECTION_VERSION = "ames_candidate_selection_gpt_gold_token_projection.v1"
TOKEN_PROJECTION_METRIC = "actual_total_tokens_per_request"
TOKEN_PROJECTION_QUANTILE_METHOD = "nearest_rank"
ENDPOINT_RECEIPT_VERSION = "ames_deepseek_host_endpoint_receipt.v1"
DEEPSEEK_ARCHITECTURE = "DeepseekV4ForCausalLM"
DEEPSEEK_MODEL_TYPE = "deepseek_v4"
PARTITION_IDS = ("gpt_key_one", "gpt_key_two", "deepseek")
GPT_PARTITIONS = PARTITION_IDS[:2]
ACTIVE_EXTRACTION_SOURCE_IDS = (
    "fixed_mutation",
    "premutagenic_damage",
    "mutagenicity_mechanism",
)
OPENAI_TOKEN_ALLOCATION = 10_000_000
OPENAI_PLANNING_TARGET = 9_000_000
OPENAI_MINIMUM_HEADROOM = 1_000_000
DEFAULT_MAX_COMPLETION_TOKENS = MAX_COMPLETION_TOKENS


def _digest_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def candidate_inventory_reference(path: Path, row_count: int) -> dict[str, Any]:
    """Validate and pin the immutable candidate inventory and its manifest."""
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError(f"candidate inventory is incomplete: {path}")
    manifest = _read_object(manifest_path)
    expected = {
        "inventory_version": CANDIDATE_CONTRACT_VERSION,
        "candidate_rows": row_count,
        "candidate_sha256": file_sha256(path),
    }
    mismatches = {key for key, value in expected.items() if manifest.get(key) != value}
    if pq.read_metadata(path).num_rows != row_count:
        mismatches.add("parquet_rows")
    if mismatches:
        raise ValueError(f"candidate inventory manifest mismatch: {sorted(mismatches)}")
    return {
        "path": str(path.resolve()),
        "sha256": expected["candidate_sha256"],
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "version": CANDIDATE_CONTRACT_VERSION,
        "rows": row_count,
    }


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def provider_contracts() -> dict[str, dict[str, Any]]:
    """Declare the exact backend identity expected from each frozen partition."""
    common = {"allow_fallbacks": False, "reasoning_effort": "low"}
    direct = {
        **common,
        "provider": "openai",
        "requested_provider_tag": None,
        "expected_served_provider": None,
        "requested_model": OPENAI_MODEL,
        "expected_returned_model": OPENAI_MODEL,
        "base_url": OPENAI_BASE_URL,
        "token_allocation": OPENAI_TOKEN_ALLOCATION,
    }
    return {
        "gpt_key_one": {**direct, "credential_env": "OPENAI_API_KEY_ONE"},
        "gpt_key_two": {**direct, "credential_env": "OPENAI_API_KEY_TWO"},
        "deepseek": {
            **common,
            "provider": DEEPSEEK_PROVIDER,
            "requested_provider_tag": None,
            "expected_served_provider": None,
            "requested_model": DEEPSEEK_MODEL,
            "expected_returned_model": DEEPSEEK_MODEL,
            "base_url": DEEPSEEK_BASE_URL,
            "credential_env": DEEPSEEK_CREDENTIAL_ENV,
            "max_concurrency": ENDPOINT_CONCURRENCY_BUDGET,
            "token_allocation": None,
        },
    }


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def validate_endpoint_receipt(
    path: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the live local host selection recorded before DeepSeek execution."""
    payload = _read_object(path)
    selected = payload.get("selected_endpoint") or {}
    expected = {
        "host": "dgx027",
        "provider": contract["provider"],
        "base_url": contract["base_url"],
        "requested_model": contract["requested_model"],
        "returned_model": contract["expected_returned_model"],
        "credential_env": contract["credential_env"],
        "max_concurrency": contract["max_concurrency"],
    }
    mismatches = {key for key, value in expected.items() if selected.get(key) != value}
    if payload.get("receipt_version") != ENDPOINT_RECEIPT_VERSION:
        mismatches.add("receipt_version")
    if not str(payload.get("checked_at_utc") or "").strip().endswith("Z"):
        mismatches.add("checked_at_utc")
    live = payload.get("live_checks") or {}
    generation = live.get("generation") or {}
    model_info = live.get("model_info") or {}
    models = live.get("models") or {}
    model = contract["requested_model"]
    live_expected = {
        "generation.http_status": (generation.get("http_status"), 200),
        "generation.returned_model": (
            generation.get("returned_model"),
            contract["expected_returned_model"],
        ),
        "model_info.http_status": (model_info.get("http_status"), 200),
        "model_info.model_path": (model_info.get("model_path"), model),
        "model_info.served_model_name": (
            model_info.get("served_model_name"),
            model,
        ),
        "model_info.model_type": (model_info.get("model_type"), DEEPSEEK_MODEL_TYPE),
        "model_info.architectures": (
            model_info.get("architectures"),
            [DEEPSEEK_ARCHITECTURE],
        ),
        "models.http_status": (models.get("http_status"), 200),
        "models.ids": (models.get("ids"), [model]),
    }
    mismatches.update(
        name for name, (found, required) in live_expected.items() if found != required
    )
    for name, check in (
        ("generation", generation),
        ("model_info", model_info),
        ("models", models),
    ):
        if not _is_lower_sha256(check.get("sha256")):
            mismatches.add(f"{name}.sha256")
    if mismatches:
        raise ValueError(f"DeepSeek endpoint receipt mismatch: {sorted(mismatches)}")
    return payload


def _validate_projection_sources(payload: Mapping[str, Any]) -> dict[str, int]:
    sources = payload.get("sources") or {}
    if set(sources) != set(ACTIVE_EXTRACTION_SOURCE_IDS):
        raise ValueError(
            "token projection must cover active extraction sources exactly"
        )
    projected: dict[str, int] = {}
    for source_id in ACTIVE_EXTRACTION_SOURCE_IDS:
        spec = sources[source_id]
        requests = int(spec.get("gold_request_count") or 0)
        tokens = int(spec.get("p95_actual_tokens") or 0)
        if requests < 1 or tokens < 1:
            raise ValueError(f"invalid gold token projection for {source_id}")
        projected[source_id] = tokens
    return projected


def _validate_projection_evidence(path: Path, payload: Mapping[str, Any]) -> None:
    references = payload.get("gold_replay_evidence") or {}
    if set(references) != {"cache", "mapping", "mapping_manifest", "gold_fixture"}:
        raise ValueError("gold replay evidence inventory mismatch")
    for label, reference in references.items():
        evidence_path = _resolve(path.parent, reference.get("path") or "")
        if not evidence_path.is_file() or file_sha256(evidence_path) != reference.get(
            "sha256"
        ):
            raise ValueError(f"gold replay {label} hash mismatch")


def load_token_projection(
    path: Path,
    *,
    records_path: Path,
    profile_path: Path,
    prompt: Mapping[str, Any],
    max_completion_tokens: int,
) -> dict[str, Any]:
    """Validate the frozen, source-specific p95 estimates from a complete replay."""
    payload = _read_object(path)
    expected = {
        "projection_version": TOKEN_PROJECTION_VERSION,
        "task_id": "ames",
        "requested_model": OPENAI_MODEL,
        "returned_models": [OPENAI_MODEL],
        "batch_size": int(prompt["batch_size"]),
        "max_completion_tokens": int(max_completion_tokens),
        "cleaned_records_sha256": file_sha256(records_path),
        "endpoint_profile_sha256": file_sha256(profile_path),
        "usage_metric": TOKEN_PROJECTION_METRIC,
        "quantile": 0.95,
        "quantile_method": TOKEN_PROJECTION_QUANTILE_METHOD,
        "complete_gold_row_coverage": True,
        "complete_usage": True,
        "allow_fallbacks": False,
    }
    mismatches = {key for key, value in expected.items() if payload.get(key) != value}
    if payload.get("prompt") != dict(prompt):
        mismatches.add("prompt")
    manifest_path = records_path.with_suffix(".manifest.json")
    candidate_manifest = _read_object(manifest_path)
    candidate_reference = candidate_inventory_reference(
        records_path, int(candidate_manifest.get("candidate_rows") or 0)
    )
    if payload.get("candidate_inventory") != candidate_reference:
        mismatches.add("candidate_inventory")
    if mismatches:
        raise ValueError(
            f"gold token projection is not comparable: {sorted(mismatches)}"
        )
    _validate_projection_sources(payload)
    _validate_projection_evidence(path, payload)
    return payload


def _projection_reference(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "version": TOKEN_PROJECTION_VERSION,
        "usage_metric": TOKEN_PROJECTION_METRIC,
        "quantile": 0.95,
        "quantile_method": TOKEN_PROJECTION_QUANTILE_METHOD,
        "source_p95_actual_tokens": _validate_projection_sources(payload),
        "gold_replay_evidence": dict(payload["gold_replay_evidence"]),
    }


def _candidate_index(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in candidates:
        record_id = str(row.get("id") or "")
        if not record_id or record_id in indexed:
            raise ValueError(f"candidate id is missing or duplicated: {record_id!r}")
        if not row.get("source_row_uid") or not row.get("canonical_endpoint_name"):
            raise ValueError(f"candidate identity is incomplete: {record_id}")
        expected_hash = str(row.get("candidate_set_sha256") or "")
        if (
            candidate_set_sha256(row.get("measurement_candidates_json"))
            != expected_hash
        ):
            raise ValueError(f"candidate-set hash mismatch: {record_id}")
        indexed[record_id] = row
    found_sources = {str(row["source_id"]) for row in indexed.values()}
    if found_sources != set(ACTIVE_EXTRACTION_SOURCE_IDS):
        raise ValueError(
            "candidate inventory must contain every active extraction source"
        )
    return indexed


def _validate_spent_tokens(spent_tokens: Mapping[str, int]) -> dict[str, int]:
    if set(spent_tokens) != set(GPT_PARTITIONS):
        raise ValueError("spent-token ledger must cover both GPT keys exactly")
    spent = {key: int(spent_tokens[key]) for key in GPT_PARTITIONS}
    if any(value < 0 or value > OPENAI_PLANNING_TARGET for value in spent.values()):
        raise ValueError("spent tokens must be within each 9M planning target")
    return spent


def assign_batches(
    candidates: Sequence[Mapping[str, Any]],
    batches: Sequence[RequestBatch],
    *,
    spent_tokens: Mapping[str, int],
    source_p95_actual_tokens: Mapping[str, int],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Balance whole batches using frozen p95 usage and leave 1M per GPT key."""
    indexed = _candidate_index(candidates)
    spent = _validate_spent_tokens(spent_tokens)
    projected = dict.fromkeys(GPT_PARTITIONS, 0)
    assignments: dict[str, dict[str, Any]] = {}
    for batch in batches:
        cost = int(source_p95_actual_tokens.get(batch.source_id) or 0)
        if cost < 1:
            raise ValueError(f"missing p95 token projection for {batch.source_id}")
        eligible = [
            key
            for key in GPT_PARTITIONS
            if spent[key] + projected[key] + cost <= OPENAI_PLANNING_TARGET
        ]
        partition = (
            min(eligible, key=lambda key: (spent[key] + projected[key], key))
            if eligible
            else "deepseek"
        )
        if partition in projected:
            projected[partition] += cost
        for record_id in batch.row_ids:
            if record_id not in indexed or record_id in assignments:
                raise ValueError(f"planned batch coverage mismatch: {record_id}")
            assignments[record_id] = {
                "partition_id": partition,
                "planning_request_id": batch.request_id,
                "planning_p95_actual_tokens": cost,
            }
    _validate_assignment_coverage(assignments, indexed)
    return assignments, projected


def _validate_assignment_coverage(
    assignments: Mapping[str, Mapping[str, Any]],
    indexed: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(assignments) != set(indexed):
        raise ValueError("planned batches do not cover the candidate inventory")
    counts = Counter(row["partition_id"] for row in assignments.values())
    if any(not counts[partition] for partition in PARTITION_IDS):
        raise ValueError(f"all three partitions must be nonempty: {dict(counts)}")


def _arrow_table(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    columns = sorted(set().union(*(row.keys() for row in rows)))
    return pa.Table.from_pylist(
        [{key: row.get(key) for key in columns} for row in rows]
    )


def _write_new_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_arrow_table(rows), path)


def _partition_input(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cleaned_record_id": row["id"],
        **{key: value for key, value in row.items() if key != "id"},
    }


def _assignment_rows(
    candidates: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        record_id = str(candidate["id"])
        rows.append(
            {
                "cleaned_record_id": record_id,
                "source_row_uid": candidate["source_row_uid"],
                "source_id": candidate["source_id"],
                "canonical_endpoint_name": candidate["canonical_endpoint_name"],
                "candidate_set_sha256": candidate["candidate_set_sha256"],
                **assignments[record_id],
            }
        )
    return sorted(rows, key=lambda row: row["cleaned_record_id"])


def _identity_digest(candidates: Sequence[Mapping[str, Any]]) -> str:
    identities = sorted(
        (
            row["id"] if "id" in row else row["cleaned_record_id"],
            row["source_row_uid"],
            row["source_id"],
            row["canonical_endpoint_name"],
            row["candidate_set_sha256"],
        )
        for row in candidates
    )
    return _digest_json(identities)


def _write_partition_inputs(
    output_dir: Path,
    candidates: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    partitions: dict[str, dict[str, Any]] = {}
    for partition in PARTITION_IDS:
        selected = [
            _partition_input(row)
            for row in candidates
            if assignments[str(row["id"])]["partition_id"] == partition
        ]
        path = output_dir / "inputs" / f"{partition}.parquet"
        _write_new_parquet(path, selected)
        partitions[partition] = {
            "rows": len(selected),
            "input_path": str(path.relative_to(output_dir)),
            "input_sha256": file_sha256(path),
            "candidate_identity_sha256": _identity_digest(selected),
            "mapping_path": f"outputs/{partition}/measurement_selection.parquet",
        }
    return partitions


def _budgeted_partitions(
    partitions: Mapping[str, Mapping[str, Any]],
    projected_tokens: Mapping[str, int],
    spent_tokens: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    enriched = {key: dict(value) for key, value in partitions.items()}
    for key in GPT_PARTITIONS:
        spent, projected = int(spent_tokens[key]), int(projected_tokens[key])
        enriched[key].update(
            {
                "tokens_spent_before_plan": spent,
                "projected_new_tokens": projected,
                "projected_total_tokens": spent + projected,
                "planning_target_tokens": OPENAI_PLANNING_TARGET,
                "runtime_ledger_max_tokens": OPENAI_TOKEN_ALLOCATION,
                "minimum_unplanned_headroom_tokens": OPENAI_MINIMUM_HEADROOM,
            }
        )
    return enriched


def _plan_payload(
    candidates: Sequence[Mapping[str, Any]],
    partitions: Mapping[str, Mapping[str, Any]],
    contracts: Mapping[str, Mapping[str, Any]],
    records_path: Path,
    profile_path: Path,
    prompt: Mapping[str, Any],
    assignment_path: Path,
    projection_reference: Mapping[str, Any],
    endpoint_receipt_path: Path,
    max_completion_tokens: int,
    candidate_reference: Mapping[str, Any],
) -> dict[str, Any]:
    source_counts = Counter(str(row["source_id"]) for row in candidates)
    return {
        "plan_version": PLAN_VERSION,
        "task_id": "ames",
        "artifact_role": "raw_candidate_selection",
        "selection_config_module": SELECTION_CONFIG_MODULE,
        "mapping_version": MAPPING_VERSION,
        "candidate_rows": len(candidates),
        "candidate_identity_sha256": _identity_digest(candidates),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "candidate_inventory": dict(candidate_reference),
        "cleaned_records_path": str(records_path.resolve()),
        "cleaned_records_sha256": file_sha256(records_path),
        "endpoint_profile_path": str(profile_path.resolve()),
        "endpoint_profile_sha256": file_sha256(profile_path),
        "prompt": dict(prompt),
        "max_completion_tokens": max_completion_tokens,
        "token_projection": dict(projection_reference),
        "deepseek_endpoint_receipt": {
            "path": str(endpoint_receipt_path.resolve()),
            "sha256": file_sha256(endpoint_receipt_path),
        },
        "assignment_path": str(assignment_path.name),
        "assignment_sha256": file_sha256(assignment_path),
        "provider_contracts": {key: dict(contracts[key]) for key in PARTITION_IDS},
        "partitions": {key: dict(partitions[key]) for key in PARTITION_IDS},
        "validations": {
            "disjoint": True,
            "exhaustive": True,
            "all_nonempty": True,
            "candidate_set_hashes_verified": True,
        },
    }


def write_partition_plan(
    output_dir: Path,
    *,
    candidates: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
    projected_tokens: Mapping[str, int],
    spent_tokens: Mapping[str, int],
    contracts: Mapping[str, Mapping[str, Any]],
    records_path: Path,
    profile_path: Path,
    prompt: Mapping[str, Any],
    token_projection_path: Path,
    endpoint_receipt_path: Path,
    max_completion_tokens: int,
) -> Path:
    """Write immutable assignment and per-provider candidate artifacts."""
    manifest_path = output_dir / "candidate_selection_assignment.manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to replace frozen plan: {manifest_path}")
    projection = load_token_projection(
        token_projection_path,
        records_path=records_path,
        profile_path=profile_path,
        prompt=prompt,
        max_completion_tokens=max_completion_tokens,
    )
    validate_endpoint_receipt(endpoint_receipt_path, contracts["deepseek"])
    _candidate_index(candidates)
    candidate_reference = candidate_inventory_reference(records_path, len(candidates))
    rows = _assignment_rows(candidates, assignments)
    assignment_path = output_dir / "candidate_selection_assignments.parquet"
    _write_new_parquet(assignment_path, rows)
    partitions = _write_partition_inputs(output_dir, candidates, assignments)
    budgeted = _budgeted_partitions(partitions, projected_tokens, spent_tokens)
    payload = _plan_payload(
        candidates,
        budgeted,
        contracts,
        records_path,
        profile_path,
        prompt,
        assignment_path,
        _projection_reference(token_projection_path, projection),
        endpoint_receipt_path,
        max_completion_tokens,
        candidate_reference,
    )
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest_path


def create_plan(args: argparse.Namespace) -> Path:
    """Build the deterministic request inventory and freeze its three partitions."""
    config = TaskConfig("ames", SELECTION_CONFIG_MODULE)
    profile = load_endpoint_profile(args.endpoint_profile, config)
    candidates = candidate_rows(args.cleaned_records, config)
    prompt = config.prompt_manifest(batch_size=config.BATCH_SIZE)
    projection = load_token_projection(
        args.token_projection,
        records_path=args.cleaned_records,
        profile_path=args.endpoint_profile,
        prompt=prompt,
        max_completion_tokens=args.max_completion_tokens,
    )
    batches = plan_batches(
        candidates,
        config,
        attempted=set(),
        endpoint_profile=profile,
        profile_digest=file_sha256(args.endpoint_profile),
        model=OPENAI_MODEL,
        max_completion_tokens=args.max_completion_tokens,
    )
    spent = {
        "gpt_key_one": args.gpt_key_one_spent_tokens,
        "gpt_key_two": args.gpt_key_two_spent_tokens,
    }
    contracts = provider_contracts()
    assignments, projected = assign_batches(
        candidates,
        batches,
        spent_tokens=spent,
        source_p95_actual_tokens=_validate_projection_sources(projection),
    )
    return write_partition_plan(
        args.output_dir,
        candidates=candidates,
        assignments=assignments,
        projected_tokens=projected,
        spent_tokens=spent,
        contracts=contracts,
        records_path=args.cleaned_records,
        profile_path=args.endpoint_profile,
        prompt=prompt,
        token_projection_path=args.token_projection,
        endpoint_receipt_path=args.deepseek_endpoint_receipt,
        max_completion_tokens=args.max_completion_tokens,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-inventory",
        "--cleaned-records",
        dest="cleaned_records",
        type=Path,
        default=DEFAULT_CLEANED_RECORDS,
    )
    parser.add_argument("--endpoint-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-projection", type=Path, required=True)
    parser.add_argument("--deepseek-endpoint-receipt", type=Path, required=True)
    parser.add_argument("--gpt-key-one-spent-tokens", type=int, required=True)
    parser.add_argument("--gpt-key-two-spent-tokens", type=int, required=True)
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_completion_tokens < 1:
        raise ValueError("max completion tokens must be positive")
    print(create_plan(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
