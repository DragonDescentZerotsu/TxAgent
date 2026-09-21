"""Build unpublished pair-dimension mapping candidates from task contracts."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from data.processing.evidence_library.shared.v2.clustered_auxiliary_mapping import (
    AuxiliaryExtractionSpec,
    DEFAULT_NULL_LIKE,
    build_clustered_auxiliary_mapping,
    reconciliation_plan,
)
from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from data.processing.llm_api import DEFAULT_ENV_FILE, openai_compatible_client


TASK_CONTRACTS = {
    "ames": (
        "data.processing.evidence_library.versions.v10.tasks.ames.starling_schema",
        "RECORD_CONTRACT",
    ),
    "dili": (
        "data.processing.evidence_library.versions.v10.tasks.dili.starling_schema",
        "RECORD_CONTRACT",
    ),
    "carcinogens": (
        "data.processing.evidence_library.versions.v10.tasks.carcinogens.starling_policy",
        "CONTRACT",
    ),
}
REUSED_FIELDS = {"ames": frozenset({"canonical_endpoint_name"})}
FINAL_MAPPING_VERSIONS = {
    "ames": "ames_pair_context_mapping.v1",
    "dili": "dili_pair_dimensions.v1",
    "carcinogens": "carcinogens_pair_dimensions.v1",
}
FINAL_OUTPUT_FIELDS = {
    "dili": {"canonical_endpoint_name": "canonical_endpoint_concept"},
    "carcinogens": {"canonical_endpoint_name": "canonical_endpoint_concept"},
}
GUIDANCE = {
    "endpoint": (
        "Normalize synonymous endpoint names conservatively. Preserve distinct tumors, "
        "phenotypes, biomarkers, assay families, and biological processes."
    ),
    "assay_context": (
        "Create a concise assay/readout label. Merge spelling and naming variants but "
        "preserve assay formats or readouts that are not experimentally comparable."
    ),
    "species_context": (
        "Normalize species names to concise common labels while preserving species "
        "identity. Do not infer a species that is not stated."
    ),
}


def _contract(task: str) -> Any:
    module_name, attribute = TASK_CONTRACTS[task]
    return getattr(importlib.import_module(module_name), attribute)


def _prompt(task: str, source_id: str, semantic_dimension: str) -> str:
    guidance = GUIDANCE[semantic_dimension]
    return (
        f"You are reconciling the {semantic_dimension.replace('_', ' ')} dimension "
        f"for {task}/{source_id} assay-transfer pair buckets. {guidance} "
        "Each value is either a source string or an ordered JSON tuple of source "
        "fields. Map every input ID to the canonical label you judge scientifically "
        "appropriate, using a JSON string or null. "
        "Do not encode dose, duration, effect direction, result, evidence quality, "
        "benchmark condition, Gold membership, split, voter, molecule, or parent."
    )


def extraction_specs(task: str, input_path: Path) -> tuple[AuxiliaryExtractionSpec, ...]:
    contract = _contract(task)
    reused = REUSED_FIELDS.get(task, frozenset())
    specs = []
    for source_id, profile in contract.sources.items():
        pair_fields = set(contract.pair_buckets[source_id].canonical_dimensions)
        for dimension in profile.canonical_dimensions:
            if (
                dimension.output_field not in pair_fields
                or dimension.output_field in reused
                or dimension.method != "frozen_mapping"
            ):
                continue
            specs.append(
                AuxiliaryExtractionSpec(
                    source_id=source_id,
                    input_path=input_path,
                    input_column=None,
                    input_columns=tuple(dimension.input_fields),
                    output_field=dimension.output_field,
                    prompt=_prompt(task, source_id, dimension.semantic_dimension),
                    null_sentinel="__null__",
                )
            )
    if not specs:
        raise ValueError(f"{task} has no unreused frozen pair dimensions")
    return tuple(specs)


def _load_candidate(task: str, path: Path) -> dict[str, Any]:
    candidate = json.loads(path.read_text(encoding="utf-8"))
    sources = candidate.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(_contract(task).sources):
        raise ValueError("first-pass candidate source inventory differs from contract")
    expected = {
        (spec.source_id, spec.output_field)
        for spec in extraction_specs(task, path)
    }
    actual = {
        (source, field)
        for source, outputs in sources.items()
        for field in outputs
    }
    if actual != expected:
        raise ValueError("first-pass candidate section inventory differs from contract")
    return candidate


def _global_prompt(task: str, source: str, field: str) -> str:
    return (
        f"You are performing the second global reconciliation pass for {task}/{source} "
        f"field {field}. Each input is a provisional canonical label produced by an "
        "independent local-cluster pass. Map synonymous or scientifically equivalent "
        "labels to one shared final label. Preserve real distinctions in endpoint, "
        "assay/readout, measurement scale, mechanism, biological system, and species. "
        "Map every input ID to exactly one JSON string or null. Do not add dose, "
        "duration, effect direction, result, Gold membership, split, molecule, or parent."
    )


def _global_inputs(
    task: str, candidate: dict[str, Any], root: Path
) -> tuple[AuxiliaryExtractionSpec, ...]:
    specs = []
    for source, outputs in sorted(candidate["sources"].items()):
        for field, section in sorted(outputs.items()):
            labels = sorted(
                {value for value in section["mapping"].values() if value is not None},
                key=lambda value: (str(value).casefold(), str(value)),
            )
            path = root / f"{source}__{field}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"provisional_label": labels}).to_parquet(path, index=False)
            specs.append(
                AuxiliaryExtractionSpec(
                    source_id=source,
                    input_path=path,
                    input_column="provisional_label",
                    output_field=field,
                    prompt=_global_prompt(task, source, field),
                    null_sentinel="__null__",
                )
            )
    return tuple(specs)


def global_reconciliation_plan(
    task: str, candidate_path: Path, *, cluster_size: int
) -> dict[str, Any]:
    if cluster_size < 1:
        raise ValueError("global cluster size must be positive")
    candidate = _load_candidate(task, candidate_path)
    sections = {}
    for source, outputs in sorted(candidate["sources"].items()):
        for field, section in sorted(outputs.items()):
            labels = {value for value in section["mapping"].values() if value is not None}
            sections[f"{source}/{field}"] = {
                "provisional_labels": len(labels),
                "planned_clusters": math.ceil(len(labels) / cluster_size),
            }
    return {
        "task": task,
        "first_pass_candidate_sha256": file_sha256(candidate_path),
        "cluster_size": cluster_size,
        "sections": sections,
        "planned_clusters": sum(row["planned_clusters"] for row in sections.values()),
    }


def _compose_global_mapping(
    task: str,
    candidate: dict[str, Any],
    label_mapping: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "mapping_version": FINAL_MAPPING_VERSIONS[task],
        "sources": {},
    }
    for source, sections in sorted(candidate["sources"].items()):
        output["sources"][source] = {}
        for field, section in sorted(sections.items()):
            final_section = label_mapping["sources"][source][field]
            lookup = final_section["mapping"]
            mapping = {}
            for raw_key, provisional in section["mapping"].items():
                stripped = str(provisional).strip() if provisional is not None else None
                normalized = (
                    None
                    if provisional is None
                    or stripped.casefold() in DEFAULT_NULL_LIKE
                    else stripped
                )
                key = json.dumps([normalized], ensure_ascii=False, separators=(",", ":"))
                if key not in lookup:
                    raise ValueError(f"global label pass lacks {source}/{field}: {key}")
                mapping[raw_key] = lookup[key]
            output_field = FINAL_OUTPUT_FIELDS.get(task, {}).get(field, field)
            output["sources"][source][output_field] = {
                "source_columns": section["source_columns"],
                "mapping": mapping,
            }
    return output


def _validate_global_label_cache(
    label_path: Path,
    specs: tuple[AuxiliaryExtractionSpec, ...],
    *,
    task: str,
    model: str,
    cluster_size: int,
    max_cluster_size: int,
) -> None:
    generation_path = label_path.with_suffix(label_path.suffix + ".generation.json")
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    if (
        generation.get("mapping_sha256") != file_sha256(label_path)
        or generation.get("mapping_version")
        != f"{task}_pair_dimension_global_labels.v1"
        or generation.get("model") != model
        or generation.get("reasoning_effort") != "high"
        or generation.get("cluster_target_size") != cluster_size
        or generation.get("max_cluster_size") != max_cluster_size
    ):
        raise ValueError("global-label cache contract differs from this run")
    inputs = generation.get("input_artifacts") or {}
    expected = {str(spec.input_path) for spec in specs}
    if set(inputs) != expected or any(
        inputs[str(spec.input_path)].get("sha256") != file_sha256(spec.input_path)
        for spec in specs
    ):
        raise ValueError("global-label cache input hashes are stale")


def _existing_global_output(
    output_path: Path, candidate_path: Path, label_path: Path
) -> dict[str, Any] | None:
    if not output_path.exists():
        return None
    generation = output_path.with_suffix(output_path.suffix + ".generation.json")
    if not generation.is_file():
        raise FileExistsError(f"global output lacks its generation manifest: {output_path}")
    result = json.loads(generation.read_text(encoding="utf-8"))
    if (
        result.get("output", {}).get("sha256") != file_sha256(output_path)
        or result.get("first_pass_candidate", {}).get("sha256")
        != file_sha256(candidate_path)
        or result.get("global_label_mapping", {}).get("sha256")
        != file_sha256(label_path)
    ):
        raise FileExistsError(f"global output exists under another contract: {output_path}")
    return result


def _write_global_output(
    task: str,
    candidate_path: Path,
    label_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    candidate = _load_candidate(task, candidate_path)
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    payload = _compose_global_mapping(task, candidate, labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return {
        "publication_status": "unpublished_requires_agent_review",
        "mapping_version": payload["mapping_version"],
        "first_pass_candidate": {
            "path": str(candidate_path),
            "sha256": file_sha256(candidate_path),
        },
        "global_label_mapping": {
            "path": str(label_path),
            "sha256": file_sha256(label_path),
            "generation_sha256": file_sha256(
                label_path.with_suffix(label_path.suffix + ".generation.json")
            ),
        },
        "output": {"path": str(output_path), "sha256": file_sha256(output_path)},
    }


def _expected_review_fields(task: str) -> dict[str, set[str]]:
    expected: dict[str, set[str]] = {}
    for spec in extraction_specs(task, Path("unused.parquet")):
        field = FINAL_OUTPUT_FIELDS.get(task, {}).get(spec.output_field, spec.output_field)
        expected.setdefault(spec.source_id, set()).add(field)
    return expected


def _load_review_candidate(task: str, path: Path) -> dict[str, Any]:
    generation_path = path.with_suffix(path.suffix + ".generation.json")
    if not generation_path.is_file():
        raise ValueError("review candidate lacks its global reconciliation manifest")
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    if (
        generation.get("publication_status") != "unpublished_requires_agent_review"
        or generation.get("output", {}).get("sha256") != file_sha256(path)
    ):
        raise ValueError("review candidate global reconciliation manifest is stale")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mapping_version") != FINAL_MAPPING_VERSIONS[task]:
        raise ValueError("review candidate has an unexpected mapping version")
    expected = _expected_review_fields(task)
    sources = payload.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(expected):
        raise ValueError("review candidate source inventory differs from contract")
    for source, fields in expected.items():
        if set(sources[source]) != fields:
            raise ValueError(f"review candidate fields differ for {source}")
        for section in sources[source].values():
            if not isinstance(section.get("mapping"), dict):
                raise ValueError("review candidate section lacks a mapping")
    return payload


def _review_inventory(payload: dict[str, Any]) -> list[dict[str, Any]]:
    inventory = []
    for source, fields in sorted(payload["sources"].items()):
        for field, section in sorted(fields.items()):
            grouped: dict[str | None, list[str]] = {}
            for raw_key, label in section["mapping"].items():
                grouped.setdefault(label, []).append(raw_key)
            labels = [
                {
                    "canonical_label": label,
                    "raw_key_count": len(raw_keys),
                    "raw_key_examples": sorted(raw_keys)[:5],
                }
                for label, raw_keys in sorted(
                    grouped.items(), key=lambda row: (row[0] is None, str(row[0]).casefold())
                )
            ]
            inventory.append({"source_id": source, "field": field, "labels": labels})
    return inventory


def prepare_agent_review(task: str, candidate_path: Path, output_path: Path) -> dict[str, Any]:
    payload = _load_review_candidate(task, candidate_path)
    packet = {
        "version": "pair_dimension_agent_review_packet.v1",
        "task_id": task,
        "mapping_id": "auxiliary_context",
        "candidate": {"path": str(candidate_path), "sha256": file_sha256(candidate_path)},
        "candidate_generation_sha256": file_sha256(
            candidate_path.with_suffix(candidate_path.suffix + ".generation.json")
        ),
        "sections": _review_inventory(payload),
        "required_review": {
            "version": "pair_dimension_agent_review.v1",
            "decision": "approve",
            "review_completion": ["all_sections_reviewed", "all_labels_reviewed"],
            "override_fields": ["source_id", "field", "from", "to", "rationale"],
        },
    }
    _write_json_exclusive(output_path, packet)
    return packet


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validated_agent_review(
    task: str, candidate_path: Path, review_path: Path
) -> dict[str, Any]:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    completion = review.get("review_completion") or {}
    if (
        review.get("version") != "pair_dimension_agent_review.v1"
        or review.get("task_id") != task
        or review.get("candidate_sha256") != file_sha256(candidate_path)
        or review.get("decision") != "approve"
        or not review.get("reviewer")
        or completion.get("all_sections_reviewed") is not True
        or completion.get("all_labels_reviewed") is not True
        or not isinstance(review.get("overrides"), list)
    ):
        raise ValueError("agent review is incomplete or targets another candidate")
    return review


def _apply_review_overrides(
    payload: dict[str, Any], overrides: list[dict[str, Any]]
) -> dict[str, Any]:
    replacements: dict[tuple[str, str, str | None], str | None] = {}
    for item in overrides:
        if set(item) != {"source_id", "field", "from", "to", "rationale"}:
            raise ValueError("agent review override has an invalid schema")
        source, field, old = item["source_id"], item["field"], item["from"]
        if not str(item["rationale"]).strip():
            raise ValueError("agent review override lacks a rationale")
        key = (source, field, old)
        if key in replacements:
            raise ValueError(f"duplicate agent review override: {key}")
        section = payload.get("sources", {}).get(source, {}).get(field)
        if section is None or old not in set(section["mapping"].values()):
            raise ValueError(f"agent review override targets an unknown label: {key}")
        replacements[key] = item["to"]
    for source, fields in payload["sources"].items():
        for field, section in fields.items():
            section["mapping"] = {
                raw: replacements.get((source, field, label), label)
                for raw, label in section["mapping"].items()
            }
    return payload


def establish_reviewed_mapping(
    task: str,
    candidate_path: Path,
    review_path: Path,
    output_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    payload = _load_review_candidate(task, candidate_path)
    review = _validated_agent_review(task, candidate_path, review_path)
    reviewed = _apply_review_overrides(payload, review["overrides"])
    if output_path.exists():
        if json.loads(output_path.read_text(encoding="utf-8")) != reviewed:
            raise FileExistsError(f"reviewed mapping exists with different bytes: {output_path}")
    else:
        _write_json_exclusive(output_path, reviewed)
    receipt = {
        "version": "reviewed_canonical_mapping_receipt.v1",
        "task_id": task,
        "mapping_id": "auxiliary_context",
        "publication_status": "reviewed",
        "mapping": {"path": str(output_path), "sha256": file_sha256(output_path)},
        "review_completion": {
            "reviewer_id": review["reviewer"],
            "review_sha256": file_sha256(review_path),
            "override_count": len(review["overrides"]),
        },
        "validations": {
            "candidate_hash_match": True,
            "complete_section_inventory": True,
            "complete_raw_key_coverage": True,
            "reviewer_approved": True,
            "override_targets_valid": True,
        },
    }
    if receipt_path.exists():
        if json.loads(receipt_path.read_text(encoding="utf-8")) != receipt:
            raise FileExistsError(f"review receipt exists with different bytes: {receipt_path}")
    else:
        _write_json_exclusive(receipt_path, receipt)
    return receipt


def run_global_reconciliation(
    *,
    task: str,
    candidate_path: Path,
    output_path: Path,
    client: Any,
    model: str,
    base_url: str,
    workers: int,
    cluster_size: int,
    max_cluster_size: int,
    max_tokens: int,
) -> dict[str, Any]:
    candidate = _load_candidate(task, candidate_path)
    root = output_path.parent / f".{output_path.name}.global_inputs"
    label_path = output_path.parent / f".{output_path.name}.global_labels.json"
    specs = _global_inputs(task, candidate, root)
    if label_path.is_file():
        _validate_global_label_cache(
            label_path,
            specs,
            task=task,
            model=model,
            cluster_size=cluster_size,
            max_cluster_size=max_cluster_size,
        )
    else:
        build_clustered_auxiliary_mapping(
            specs=specs,
            output_path=label_path,
            mapping_version=f"{task}_pair_dimension_global_labels.v1",
            prompt_version="minimal_pair_dimensions.global_labels.v1",
            api_key_loader=None,
            client=client,
            model=model,
            reasoning_effort="high",
            base_url=base_url,
            cluster_target_size=cluster_size,
            max_cluster_size=max_cluster_size,
            workers=workers,
            max_tokens=max_tokens,
        )
    if existing := _existing_global_output(output_path, candidate_path, label_path):
        return existing
    result = _write_global_output(task, candidate_path, label_path, output_path)
    generation = output_path.with_suffix(output_path.suffix + ".generation.json")
    generation.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "plan",
            "run",
            "global-plan",
            "globalize",
            "prepare-review",
            "establish",
        ),
    )
    parser.add_argument("--task", required=True, choices=tuple(TASK_CONTRACTS))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--additional-local-base-url", action="append", default=[])
    parser.add_argument("--provider-pool-config", type=Path)
    parser.add_argument("--local-concurrency", type=int, default=0)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--cluster-size", type=int, default=50)
    parser.add_argument("--max-cluster-size", type=int)
    parser.add_argument("--max-tokens", type=int, default=131_072)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    if args.command == "plan":
        specs = extraction_specs(args.task, args.input)
        result = reconciliation_plan(specs, cluster_target_size=args.cluster_size)
    elif args.command == "global-plan":
        result = global_reconciliation_plan(
            args.task, args.input, cluster_size=args.cluster_size
        )
    elif args.command == "prepare-review":
        result = prepare_agent_review(args.task, args.input, args.output)
    elif args.command == "establish":
        if not args.review or not args.receipt:
            parser.error("establish requires --review and --receipt")
        result = establish_reviewed_mapping(
            args.task, args.input, args.review, args.output, args.receipt
        )
    else:
        if not args.base_url and not args.provider_pool_config:
            parser.error(f"{args.command} requires a provider")
        if args.additional_local_base_url and not args.provider_pool_config:
            parser.error("--additional-local-base-url requires --provider-pool-config")
        if args.local_concurrency and not (
            (args.base_url or args.additional_local_base_url)
            and args.provider_pool_config
        ):
            parser.error("--local-concurrency requires both provider sources")
        if args.provider_pool_config:
            from predict.api_client.pool import (
                ProviderPoolConfig,
                ProviderSpec,
                build_provider_pool,
                load_provider_pool_config,
                validate_parallelism,
            )

            payload = json.loads(args.provider_pool_config.read_text(encoding="utf-8"))
            profile = payload.get("openrouter_ranked_profile") or {}
            if profile.get("profile") not in {"0731", "mixed"} or not profile.get(
                "snapshot_sha256"
            ):
                raise ValueError("provider pool lacks a pinned ranked profile")
            configured = load_provider_pool_config(args.provider_pool_config)
            allowed = {
                "deepseek/deepseek-v4-flash-0731",
                "deepseek/deepseek-v4.1-flash",
            }
            if {provider.model for provider in configured.providers} - allowed:
                raise ValueError("provider pool contains an unapproved model")
            providers = list(configured.providers)
            local_base_urls = [
                url
                for url in (args.base_url, *args.additional_local_base_url)
                if url
            ]
            for index, base_url in reversed(list(enumerate(local_base_urls, start=1))):
                providers.insert(
                    0,
                    ProviderSpec(
                        name=f"local_pair_reconciliation_{index}",
                        base_url=base_url,
                        model=args.model,
                        api_key_env="",
                        max_inflight=args.local_concurrency,
                    ),
                )
            configured = ProviderPoolConfig(
                providers=tuple(providers),
                failure_threshold=configured.failure_threshold,
                cooldown_seconds=configured.cooldown_seconds,
                max_failovers=configured.max_failovers,
                latency_ewma_alpha=configured.latency_ewma_alpha,
            )
            validate_parallelism(configured, args.workers)
            client = build_provider_pool(
                configured,
                env_file=DEFAULT_ENV_FILE,
                timeout_s=3_600,
                max_tokens=args.max_tokens,
                temperature=None,
                tool_service_url="",
                enable_group_tools=False,
                max_tool_rounds=0,
                reasoning_effort="high",
                enable_thinking=False,
                transport_max_retries=0,
                response_format={"type": "json_object"},
            )
        else:
            client, _ = openai_compatible_client(
                base_url=args.base_url,
                provider="local",
                env_file=DEFAULT_ENV_FILE,
                max_connections=args.workers,
                max_retries=0,
            )
        if args.command == "run":
            result = build_clustered_auxiliary_mapping(
                specs=extraction_specs(args.task, args.input),
                output_path=args.output,
                mapping_version=f"{args.task}_pair_dimension_candidate.v3",
                prompt_version="minimal_pair_dimensions.v2",
                api_key_loader=None,
                client=client,
                model=args.model,
                reasoning_effort="high",
                base_url=args.base_url,
                cluster_target_size=args.cluster_size,
                max_cluster_size=args.max_cluster_size or args.cluster_size,
                workers=args.workers,
                max_tokens=args.max_tokens,
            )
        else:
            result = run_global_reconciliation(
                task=args.task,
                candidate_path=args.input,
                output_path=args.output,
                client=client,
                model=args.model,
                base_url=args.base_url,
                workers=args.workers,
                cluster_size=args.cluster_size,
                max_cluster_size=args.max_cluster_size or args.cluster_size,
                max_tokens=args.max_tokens,
            )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
