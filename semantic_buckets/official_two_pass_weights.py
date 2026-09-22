"""Assign official two-pass weights to a frozen semantic-bucket retrieval world."""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from urllib.request import urlopen

from jinja2 import Environment, StrictUndefined
import pandas as pd

from data.processing.llm_api import DEFAULT_ENV_FILE, async_openai_compatible_client
from data.processing import openrouter_provider_pool
from predict.api_client.pool import (
    build_provider_pool,
    load_provider_pool_config,
    preflight_provider_models,
    primary_capacity,
)
from semantic_buckets import bioavailability_semantic_readout_v1 as core
from semantic_buckets import sliding_weight_assignment as display
from semantic_buckets import speculative_weight_execution as speculative
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "semantic_weight_official_two_pass.v4"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
TARGET = "skin sensitization or allergic contact dermatitis"
MAX_TOKENS = 20_480
PASS1_FANOUT = 1
PASS1_REQUIRED = 1
PASS2_FANOUT = 6
PASS2_REQUIRED = 3
PASS1_MAX_LOGICAL_INFLIGHT = 32
LOCAL_ENDPOINT_MAX_INFLIGHT = 128
PASS2_PROVIDER_CAPACITY = 14
PASS2_BATCH_SIZE = 12
PASS2_CHAIN_WIDTHS = {"L2": 1, "L3": 1, "L4": 4, "L5": 1}
PASS2_LOCAL_ENDPOINT = "dgx027_50002"
GPT_FLEX_PROVIDER = {
    "base_url": openrouter_provider_pool.OPENROUTER_URL,
    "model": "openai/gpt-6-luna",
    "api_key_env": "OPEN_ROUTER_KEY_TWO",
    "max_inflight": PASS2_PROVIDER_CAPACITY,
    "initial_latency_s": 120,
    "timeout_s": 3_600,
    "request_extra_body": {
        "service_tier": "flex",
        "reasoning_effort_override": "medium",
        "allowed_served_models": ["openai/gpt-6-luna"],
    },
}
DIMENSION_VALUE_LIMIT = 8
TEXT_LIMIT = 240
ENDPOINTS: tuple[dict[str, Any], ...] = ()
SOURCE_LABELS = {
    "direct_skin_reaction": "Direct skin sensitization/contact-allergy evidence",
    "sensitization_aop": "Skin-sensitization AOP and mechanistic evidence",
    "skin_exposure": "Dermal exposure evidence",
}
ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
PROVIDER_INVENTORY = REPOSITORY_ROOT / "predict/api_client/providers/current_endpoints.json"
SEMANTIC_ROOT = (
    ROOT / "provenance/source_local_semantic_v4/"
    "skin_main_universe_v5_semantic_parent_v1_20260919/semantic_run"
)
WORLD_ROOT = ROOT / "provenance/retrieval_worlds/skin_morgan_top100_valid_test_l2_l3_v1"
PROMPT_ROOT = ROOT / "prompts/semantic_weight_sliding_v2"
OUTPUT_ROOT = ROOT / "provenance/semantic_weight_two_pass_v4"
PASS1_SOURCE_ROOT = (
    ROOT / "provenance/semantic_weight_two_pass_v3/"
    "ames_morgan100_official_two_pass_v6_cards_v3_20260922"
)
RUN_ID = "skin_morgan100_official_two_pass_v1_20260919"
TASK = "skin_reaction"
LEVELS = ("L2", "L3")
EXPECTED_BUCKET_COUNT = 1_244
REQUIRED_SEMANTIC_STATUS = "complete_reviewed"
FINAL_STATUS = "complete_unselected"
SEMANTIC_REVIEW_STATUS = "reviewed"

TASK_CONFIGS = {
    "skin_reaction": {
        "target": TARGET,
        "source_labels": SOURCE_LABELS,
        "semantic_root": SEMANTIC_ROOT,
        "world_root": WORLD_ROOT,
        "run_id": RUN_ID,
        "levels": LEVELS,
        "bucket_count": EXPECTED_BUCKET_COUNT,
        "semantic_status": REQUIRED_SEMANTIC_STATUS,
        "final_status": FINAL_STATUS,
        "semantic_review_status": SEMANTIC_REVIEW_STATUS,
        "enriched_records": None,
    },
    "ames": {
        "target": "Ames bacterial mutagenicity under the reported assay conditions",
        "source_labels": {
            "fixed_mutation": "Fixed-mutation evidence",
            "mutagenicity_mechanism": "Mutagenicity mechanism evidence",
            "mutagenicity_outcomes": "Mutagenicity outcome evidence",
            "premutagenic_damage": "Pre-mutagenic damage evidence",
        },
        "semantic_root": (
            ROOT / "provenance/source_local_semantic_v5/"
            "ames_dili_carcinogens_main_universe_v3_frozen_queued_boundary_20260922/"
            "ames/semantic_run"
        ),
        "world_root": (
            ROOT / "provenance/retrieval_worlds/"
            "ames_morgan_top100_valid_test_l2_l5_frozen_candidate_v1_20260922"
        ),
        "run_id": "ames_morgan100_official_two_pass_v6_cards_task_level_wavefront_v1_20260922",
        "levels": ("L2", "L3", "L4", "L5"),
        "bucket_count": 11_391,
        "semantic_status": "frozen_queued_boundary",
        "final_status": "complete_unreviewed_candidate",
        "semantic_review_status": "unreviewed_candidate",
        "enriched_records": (
            REPOSITORY_ROOT / "data/evidence_libraries/ames/v10_main_universe_v3/"
            "02_canonicalized/records.parquet"
        ),
    },
}


def configure_task(task: str) -> None:
    """Select one immutable task input bundle for this process."""
    global TASK, TARGET, SOURCE_LABELS, SEMANTIC_ROOT, WORLD_ROOT, RUN_ID
    global LEVELS, EXPECTED_BUCKET_COUNT, REQUIRED_SEMANTIC_STATUS, FINAL_STATUS
    global SEMANTIC_REVIEW_STATUS
    if task not in TASK_CONFIGS:
        raise ValueError(f"unsupported task: {task}")
    TASK = task
    config = TASK_CONFIGS[task]
    TARGET = config["target"]
    SOURCE_LABELS = config["source_labels"]
    SEMANTIC_ROOT = config["semantic_root"]
    WORLD_ROOT = config["world_root"]
    RUN_ID = config["run_id"]
    LEVELS = config["levels"]
    EXPECTED_BUCKET_COUNT = config["bucket_count"]
    REQUIRED_SEMANTIC_STATUS = config["semantic_status"]
    FINAL_STATUS = config["final_status"]
    SEMANTIC_REVIEW_STATUS = config["semantic_review_status"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provider_candidates() -> tuple[list[dict[str, Any]], str]:
    inventory = json.loads(PROVIDER_INVENTORY.read_text(encoding="utf-8"))
    if inventory.get("version") != "openai_provider_pool.v1":
        raise ValueError("provider inventory version changed")
    candidates = []
    for provider in inventory.get("providers", []):
        effort = provider.get("request_extra_body", {}).get(
            "chat_template_kwargs", {}
        ).get("reasoning_effort")
        if provider.get("model") == MODEL and effort == "high":
            candidates.append(provider)
    if not candidates:
        raise ValueError("provider inventory has no high-reasoning target-model endpoints")
    return candidates, _sha256(PROVIDER_INVENTORY)


def _probe_endpoint(endpoint: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        with urlopen(f"{endpoint['base_url'].rstrip('/')}/models", timeout=5) as response:
            payload = json.load(response)
        models = sorted(str(item["id"]) for item in payload.get("data", []))
    except Exception:
        return None
    if MODEL not in models:
        return None
    return {
        "name": endpoint["name"],
        "base_url": endpoint["base_url"],
        "model": endpoint["model"],
        "inventory_max_inflight": int(endpoint["max_inflight"]),
        "max_inflight": min(LOCAL_ENDPOINT_MAX_INFLIGHT, int(endpoint["max_inflight"])),
        "advertised_models": models,
    }


def _select_endpoints() -> tuple[tuple[dict[str, Any], ...], str]:
    candidates, inventory_hash = _provider_candidates()
    with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
        selected = [item for item in executor.map(_probe_endpoint, candidates) if item]
    if not selected:
        raise ValueError("no configured target-model endpoint passed /models preflight")
    return tuple(selected), inventory_hash


def _verify_selected_endpoints(profile: Mapping[str, Any]) -> None:
    candidates, _ = _provider_candidates()
    current = {item["name"]: item for item in candidates}
    for selected in profile["endpoints"]:
        candidate = current.get(selected["name"])
        if candidate is None or candidate["base_url"] != selected["base_url"]:
            raise ValueError(f"selected endpoint left provider inventory: {selected['name']}")


def _atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)


def _paths() -> dict[str, Path]:
    paths = {
        "semantic_manifest": SEMANTIC_ROOT / "semantic_bucket_map_manifest.json",
        "semantic_map": SEMANTIC_ROOT / "semantic_bucket_map.parquet",
        "input_atoms": SEMANTIC_ROOT / "input_atoms.parquet",
        "sample_cards": SEMANTIC_ROOT / "sample_cards.json.gz",
        "world_manifest": WORLD_ROOT / "manifest.json",
        "world": WORLD_ROOT / "semantic_bucket_world.parquet",
        "initial_prompt": PROMPT_ROOT / "initial.jinja",
        "anchored_prompt": PROMPT_ROOT / "anchored.jinja",
    }
    enriched = TASK_CONFIGS[TASK]["enriched_records"]
    if enriched is not None:
        paths["evidence_manifest"] = enriched.parent / "manifest.json"
        paths["enriched_records"] = enriched
    return paths


def _verify_inputs() -> dict[str, dict[str, str]]:
    paths = _paths()
    semantic = json.loads(paths["semantic_manifest"].read_text(encoding="utf-8"))
    world = json.loads(paths["world_manifest"].read_text(encoding="utf-8"))
    if semantic.get("status") != REQUIRED_SEMANTIC_STATUS or semantic.get("task") != TASK:
        raise ValueError(f"{TASK} semantic generation is not the configured immutable input")
    if (
        world.get("status") != "complete"
        or world.get("counts", {}).get("buckets") != EXPECTED_BUCKET_COUNT
    ):
        raise ValueError(f"{TASK} Morgan-top-100 world is incomplete")
    hashes = {name: {"path": str(path), "sha256": _sha256(path)}
              for name, path in paths.items()}
    if hashes["semantic_map"]["sha256"] != semantic["semantic_bucket_map_sha256"]:
        raise ValueError("semantic map changed")
    if hashes["input_atoms"]["sha256"] != semantic["input_atoms_sha256"]:
        raise ValueError("input atoms changed")
    if hashes["world"]["sha256"] != world["files"]["semantic_bucket_world.parquet"]:
        raise ValueError("retrieval world changed")
    if TASK == "ames":
        evidence = json.loads(paths["evidence_manifest"].read_text(encoding="utf-8"))
        expected = evidence["output"]["sha256"]
        if hashes["enriched_records"]["sha256"] != expected:
            raise ValueError("AMES canonical records changed")
    return hashes


def _sample_atoms(bucket: str, atoms: pd.DataFrame) -> list[str]:
    ordered = sorted(
        atoms.atom_id.astype(str),
        key=lambda atom: (core._stable_id("weight-sample", bucket, atom), atom),
    )
    sources = dict(zip(atoms.atom_id.astype(str), atoms.source_id.astype(str)))
    selected: list[str] = []
    seen: set[str] = set()
    for new_source_only in (True, False):
        for atom in ordered:
            source = sources[atom]
            if atom in selected or (new_source_only and source in seen):
                continue
            selected.append(atom)
            seen.add(source)
            if len(selected) == 5:
                return selected
    return selected


def _component(source: str, rows: pd.DataFrame) -> dict[str, Any]:
    values = [json.loads(value) for value in rows.values_json]
    fields = sorted({field for item in values for field in item})
    dimensions = {}
    for field in fields:
        distinct = {str(item.get(field, "__unknown__")) for item in values}
        displayed = sorted(
            distinct,
            key=lambda value: (core._stable_id("weight-dimension", source, field, value), value),
        )[:DIMENSION_VALUE_LIMIT]
        displayed.sort()
        if len(distinct) > DIMENSION_VALUE_LIMIT:
            displayed.append(f"[{len(distinct) - DIMENSION_VALUE_LIMIT} additional values omitted]")
        dimensions[field] = displayed
    return {"source_id": source, "source_label": SOURCE_LABELS[source],
            "canonical_dimensions": dimensions}


AMES_ENRICHED_COLUMNS = (
    "source_name", "evidence_basis", "experimental_context", "test_system",
    "biological_test_system", "metabolic_activation", "metabolic_activation_system",
    "dose_or_concentration", "endpoint_class", "genetic_locus_or_chromosome_target",
    "assay_method_and_endpoint", "result_call", "result_direction",
    "cytotoxicity_status", "mechanism_category", "qualifying_conditions",
)


def _usable(value: Any) -> bool:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return False
    text = str(value).strip().lower()
    return bool(text) and not text.startswith("__not_present") and text != "not_applicable"


def _enriched_cards(atom_ids: set[str]) -> dict[str, dict[str, Any]]:
    record_map = pd.read_parquet(
        SEMANTIC_ROOT / "record_semantic_bucket_map.parquet",
        columns=["atom_id", "source_id", "canonical_record_id", "source_row_uid"],
    )
    representatives = (
        record_map[record_map.atom_id.astype(str).isin(atom_ids)]
        .sort_values(["atom_id", "canonical_record_id"])
        .drop_duplicates("atom_id")
    )
    records = pd.read_parquet(
        _paths()["enriched_records"],
        columns=["source_row_uid", *AMES_ENRICHED_COLUMNS],
        filters=[("source_row_uid", "in", representatives.source_row_uid.tolist())],
    ).drop_duplicates("source_row_uid").set_index("source_row_uid")
    cards = {}
    for row in representatives.itertuples(index=False):
        record = records.loc[row.source_row_uid]
        cards[str(row.atom_id)] = {
            "source_id": str(row.source_id),
            **{column: record[column] for column in AMES_ENRICHED_COLUMNS
               if _usable(record[column])},
        }
    if set(cards) != atom_ids:
        raise ValueError("enriched representative coverage differs from sampled atoms")
    return cards


def _payloads() -> dict[str, dict[str, Any]]:
    world = pd.read_parquet(_paths()["world"])
    wanted = set(world.semantic_bucket_id.astype(str))
    mapping = pd.read_parquet(_paths()["semantic_map"])
    mapping = mapping[mapping.semantic_bucket_id.astype(str).isin(wanted)]
    atoms = pd.read_parquet(_paths()["input_atoms"])
    atoms = atoms.set_index("atom_id", drop=False)
    selected_by_bucket = {
        str(bucket): _sample_atoms(str(bucket), atoms.loc[members.atom_id.astype(str)])
        for bucket, members in mapping.groupby("semantic_bucket_id", sort=True)
    }
    if TASK == "ames":
        cards = _enriched_cards({atom for values in selected_by_bucket.values() for atom in values})
    else:
        with gzip.open(_paths()["sample_cards"], "rt", encoding="utf-8") as handle:
            cards = json.load(handle)
    payloads: dict[str, dict[str, Any]] = {}
    for bucket, members in mapping.groupby("semantic_bucket_id", sort=True):
        rows = atoms.loc[members.atom_id.astype(str)]
        components = [_component(str(source), group) for source, group in rows.groupby("source_id")]
        selected = selected_by_bucket[str(bucket)]
        samples = [{"source_id": str(atoms.loc[atom, "source_id"]),
                    **{field: value for field, value in cards[atom].items()
                       if field not in {"canonical_record_id", "source_id"}}}
                   for atom in selected]
        payloads[str(bucket)] = {
            "identity": {"source_components": components}, "sample_records": samples,
        }
    if set(payloads) != wanted:
        raise ValueError("semantic prompt payload coverage differs from retrieval world")
    return payloads


def _display_value(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("examples", value)
    if isinstance(value, list):
        text = " | ".join(_display_value(item) for item in value if _usable(item))
    else:
        text = display._clean(value)
    if text.lower() in {"__unknown__", "unknown"}:
        text = "not reported"
    return text if len(text) <= TEXT_LIMIT else text[:TEXT_LIMIT - 1].rstrip() + "…"


def _card(label: str, payload: Mapping[str, Any]) -> str:
    lines = [label]
    for component in payload["identity"]["source_components"]:
        lines.extend([f"  Evidence family: {component['source_label']}",
                      "  Canonical bucket identity:"])
        for field, values in component["canonical_dimensions"].items():
            if shown := _display_value(values):
                lines.append(f"    - {field.replace('_', ' ')}: {shown}")
    lines.append("  Representative experimental records:")
    for index, sample in enumerate(payload["sample_records"], 1):
        lines.append(f"    Record {index} ({SOURCE_LABELS[sample['source_id']]}):")
        for field, value in sample.items():
            if field != "source_id" and (shown := _display_value(value)):
                lines.append(f"      - {field.replace('_', ' ')}: {shown}")
    return "\n".join(lines)


def _anchor_card(index: int, payload: Mapping[str, Any], score: Mapping[str, Any]) -> str:
    heading = (f"Anchor {index} — locked weight {score['weight']}\n"
               f"  Prior rationale: {display._clean(score['rationale'])}")
    return "\n".join([heading, *_card("  Scientific profile", payload).splitlines()[1:]])


def _render(candidates: Sequence[str], anchors: Sequence[str], level: str,
            payloads: Mapping[str, Mapping[str, Any]],
            scores: Mapping[str, Mapping[str, Any]]) -> str:
    candidate_cards = "\n\n".join(
        _card(f"Candidate {index}", payloads[bucket])
        for index, bucket in enumerate(candidates, 1)
    )
    anchor_cards = "\n\n".join(
        _anchor_card(index, payloads[bucket], scores[bucket])
        for index, bucket in enumerate(anchors, 1)
    )
    name = "anchored.jinja" if anchors else "initial.jinja"
    template = Environment(undefined=StrictUndefined, autoescape=False).from_string(
        (PROMPT_ROOT / name).read_text(encoding="utf-8")
    )
    return template.render(
        target=TARGET, level=level, candidate_cards=candidate_cards,
        anchor_cards=anchor_cards,
    ).strip()


def _pass1_schedule() -> pd.DataFrame:
    world = pd.read_parquet(_paths()["world"])
    rows = []
    for level, group in world.groupby("level", sort=True):
        buckets = sorted(group.semantic_bucket_id.astype(str))
        for batch, start in enumerate(range(0, len(buckets), 12)):
            rows.append({"stage": "pass1", "level": str(level), "batch": batch,
                         "candidate_bucket_ids_json": core._canonical_json(buckets[start:start + 12]),
                         "anchor_bucket_ids_json": "[]"})
    return pd.DataFrame(rows)


def _pass2_schedule(pass1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level, group in pass1.groupby("level", sort=True):
        ordered = group.sort_values(["weight", "semantic_bucket_id"], ascending=[False, True])
        buckets = ordered.semantic_bucket_id.astype(str).tolist()
        width = PASS2_CHAIN_WIDTHS.get(str(level), 1)
        batches = [buckets[start:start + PASS2_BATCH_SIZE]
                   for start in range(0, len(buckets), PASS2_BATCH_SIZE)]
        previous: list[list[str]] = []
        for wave, start in enumerate(range(0, len(batches), width)):
            current = batches[start:start + width]
            anchors = [batch[-1] for batch in previous]
            for chain, candidates in enumerate(current):
                rows.append({
                    "stage": "pass2", "level": str(level), "batch": start + chain,
                    "wave": wave, "chain": chain, "execution_backend": "fixed_mixed",
                    "candidate_bucket_ids_json": core._canonical_json(candidates),
                    "anchor_bucket_ids_json": core._canonical_json(anchors),
                })
            previous = current
    return pd.DataFrame(rows).sort_values(
        ["level", "wave", "chain"]
    ).reset_index(drop=True)


def _preview(anchors: Sequence[str]) -> dict[str, dict[str, str]]:
    return {bucket: {"weight": "<locked prior score>",
                     "rationale": "<locked prior rationale>"} for bucket in anchors}


def _write_review(root: Path, schedule: pd.DataFrame,
                  payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    review = root / "prompt_review"
    review.mkdir()
    files = []
    for level, group in schedule.groupby("level", sort=True):
        first = group.iloc[0]
        candidates = json.loads(first.candidate_bucket_ids_json)
        initial = _render(candidates, [], str(level), payloads, {})
        anchors, anchored_candidates = candidates[-5:], candidates[:7]
        anchored = _render(anchored_candidates, anchors, str(level), payloads, _preview(anchors))
        for shape, prompt in (("initial", initial), ("anchored", anchored)):
            path = review / f"{level}_{shape}.txt"
            path.write_text(prompt + "\n", encoding="utf-8")
            files.append({"path": path.name, "sha256": _sha256(path),
                          "characters": len(prompt)})
    manifest = {"version": f"{VERSION}.prompt_review", "status": "ready",
                "completion_requests_made": 0, "files": files}
    write_json_atomic(review / "manifest.json", manifest)
    return manifest


def prepare(run_id: str = RUN_ID) -> dict[str, Any]:
    global ENDPOINTS
    root = OUTPUT_ROOT / run_id
    if root.exists():
        raise FileExistsError(root)
    inputs = _verify_inputs()
    payloads = _payloads()
    schedule = _pass1_schedule()
    ENDPOINTS, inventory_hash = _select_endpoints()
    root.mkdir(parents=True)
    _atomic_frame(schedule, root / "pass1_schedule.parquet")
    _atomic_frame(schedule, root / "pass1_schedule.tsv")
    with gzip.open(root / "bucket_prompt_payloads.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(payloads, handle, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    review = _write_review(root, schedule, payloads)
    local_endpoint = next(
        (row["name"] for row in ENDPOINTS if row["name"] == PASS2_LOCAL_ENDPOINT),
        ENDPOINTS[0]["name"],
    )
    provider_profile = _write_fixed_provider_profile(
        root / "pass2_provider_profile.json", local_endpoint=local_endpoint,
        require_luna_qualification=False,
    )
    profile = {"model": MODEL, "reasoning_effort": "high", "max_tokens": MAX_TOKENS,
               "pass1": {"fanout": PASS1_FANOUT,
                         "required_valid_responses": PASS1_REQUIRED,
                         "aggregation": speculative.aggregation_method(PASS1_REQUIRED)},
               "pass2": {"fanout": PASS2_FANOUT,
                         "required_valid_responses": PASS2_REQUIRED,
                         "aggregation": speculative.aggregation_method(PASS2_REQUIRED)},
               "pass1_max_logical_inflight_per_endpoint": PASS1_MAX_LOGICAL_INFLIGHT,
               "pass1_aggregate_logical_capacity": len(ENDPOINTS) * PASS1_MAX_LOGICAL_INFLIGHT,
               "pass2_max_logical_inflight": sum(PASS2_CHAIN_WIDTHS.values()),
               "pass2_aggregate_replica_capacity": PASS2_PROVIDER_CAPACITY * 3,
               "pass2_local_endpoint": local_endpoint,
               "provider_inventory": str(PROVIDER_INVENTORY),
               "provider_inventory_sha256": inventory_hash,
               "endpoints": ENDPOINTS}
    write_json_atomic(root / "execution_profile.json", profile)
    manifest = {"version": VERSION, "status": "awaiting_prompt_review", "run_id": run_id,
                "created_at": _now(), "task": TASK, "levels": list(LEVELS),
                "bucket_count": len(payloads), "pass1_request_count": len(schedule),
                "semantic_review_status": SEMANTIC_REVIEW_STATUS,
                "activation_allowed": SEMANTIC_REVIEW_STATUS == "reviewed",
                "inputs": inputs, "review": review,
                "files": {name: _sha256(root / name) for name in
                          ("pass1_schedule.parquet", "pass1_schedule.tsv",
                           "bucket_prompt_payloads.json.gz", "execution_profile.json")}}
    manifest["files"]["pass2_provider_profile.json"] = _sha256(
        root / "pass2_provider_profile.json"
    )
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def _write_fixed_provider_profile(
    path: Path, *, local_endpoint: str = PASS2_LOCAL_ENDPOINT,
    require_luna_qualification: bool = True,
) -> dict[str, Any]:
    payload = openrouter_provider_pool.export_provider_pool(
        path, 30, credential_env="OPEN_ROUTER_KEY_TWO"
    )
    together = [
        provider for provider in payload["providers"]
        if provider["request_extra_body"].get("expected_upstream_provider") == "Together"
    ]
    if len(together) != 1:
        raise ValueError("the current qualified pool must contain exactly one Together route")
    together[0]["max_inflight"] = PASS2_PROVIDER_CAPACITY
    payload["providers"] = [together[0], {
        **GPT_FLEX_PROVIDER,
        "name": "openrouter_gpt-6-luna_flex",
        "request_extra_body": {
            **GPT_FLEX_PROVIDER["request_extra_body"],
            "expected_upstream_provider": "OpenAI",
        },
    }]
    payload["max_failovers"] = 0
    payload["fixed_replica_counts"] = {
        "openrouter_gpt-6-luna_flex": 2,
        together[0]["name"]: 2,
        local_endpoint: 2,
    }
    payload["local_endpoint"] = local_endpoint
    canary = (
        ROOT / "provenance/semantic_weight_two_pass_v3/"
        "ames_gpt6_luna_openrouter_medium_flex_one_step_canary_20260922/manifest.json"
    )
    if require_luna_qualification or canary.is_file():
        if not canary.is_file():
            raise ValueError(f"required GPT Flex qualification is absent: {canary}")
        payload["luna_qualification"] = {
            "status": "user_reviewed_single_canary_exception",
            "artifact": str(canary), "sha256": _sha256(canary),
        }
    write_json_atomic(path, payload)
    return payload


def _verify_source_manifest(source_root: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = source_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid frozen Pass-1 source manifest: {manifest_path}") from error
    if (
        not str(manifest.get("version", "")).startswith("semantic_weight_official_two_pass.")
        or manifest.get("task") != "ames"
        or set(manifest.get("levels", ())) != set(LEVELS)
        or int(manifest.get("bucket_count", -1)) != EXPECTED_BUCKET_COUNT
    ):
        raise ValueError("the Pass-1 source manifest does not describe the AMES world")
    if manifest.get("status") not in {
        "awaiting_pass2_prompt_review", "running_pass2", "incomplete_pass2",
        "complete_unreviewed_candidate", "complete_unselected",
    }:
        raise ValueError(f"Pass-1 source manifest is not post-Pass-1: {manifest.get('status')}")
    for name in (
        "pass1_schedule.parquet", "pass1_schedule.tsv",
        "bucket_prompt_payloads.json.gz", "execution_profile.json",
    ):
        expected = (manifest.get("files") or {}).get(name)
        path = source_root / name
        if not expected or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"frozen Pass-1 source file is not hash-pinned: {name}")
    return manifest, manifest_path


def _verify_source_requests(source_root: Path, expected_count: int) -> pd.DataFrame:
    database = source_root / "requests.sqlite3"
    if not database.is_file():
        raise ValueError(f"frozen Pass-1 request database is absent: {database}")
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT COUNT(*), SUM(status='complete'), "
            "SUM(response_json IS NOT NULL) FROM requests WHERE phase LIKE 'pass1/%'"
        ).fetchone()
        aggregates = connection.execute(
            "SELECT COUNT(*) FROM speculative_aggregates "
            "WHERE request_id IN (SELECT request_id FROM requests WHERE phase LIKE 'pass1/%')"
        ).fetchone()[0]
    except (sqlite3.Error, OSError) as error:
        raise ValueError("could not inspect the frozen Pass-1 request database") from error
    if row is None or row[0] != expected_count or row[1] != expected_count:
        connection.close()
        raise ValueError("frozen Pass-1 requests are not all complete")
    if row[2] != expected_count or aggregates != expected_count:
        connection.close()
        raise ValueError("frozen Pass-1 requests lack complete aggregate receipts")
    try:
        scores = _scores(connection, "pass1")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("frozen Pass-1 requests cannot reconstruct their scores") from error
    finally:
        connection.close()
    return scores


def _verify_frozen_pass1_source() -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest, manifest_path = _verify_source_manifest(PASS1_SOURCE_ROOT)
    manifest_hash = _sha256(manifest_path)
    schedule_path = PASS1_SOURCE_ROOT / "pass1_schedule.parquet"
    schedule_hash = _sha256(schedule_path)
    expected_count = int(manifest.get("pass1_request_count", -1))
    schedule = pd.read_parquet(schedule_path)
    if expected_count < 1 or len(schedule) != expected_count:
        raise ValueError("frozen Pass-1 schedule count does not match its manifest")
    expected_scores = _verify_source_requests(PASS1_SOURCE_ROOT, expected_count)
    weights_path = PASS1_SOURCE_ROOT / "pass1_weights.parquet"
    weights_hash = _sha256(weights_path)
    pass1 = pd.read_parquet(weights_path)
    if (
        len(pass1) != EXPECTED_BUCKET_COUNT
        or pass1.semantic_bucket_id.duplicated().any()
        or set(pass1.level.astype(str)) != set(LEVELS)
    ):
        raise ValueError("the source Pass-1 weights are incomplete")
    try:
        pd.testing.assert_frame_equal(
            pass1.sort_values(["level", "semantic_bucket_id"]).reset_index(drop=True),
            expected_scores.sort_values(["level", "semantic_bucket_id"]).reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as error:
        raise ValueError("source Pass-1 weights do not match completed request receipts") from error
    if _sha256(manifest_path) != manifest_hash:
        raise ValueError("frozen Pass-1 source manifest changed during verification")
    if _sha256(weights_path) != weights_hash:
        raise ValueError("frozen Pass-1 weights changed during verification")
    for name, expected in (manifest.get("files") or {}).items():
        path = PASS1_SOURCE_ROOT / name
        if path.is_file() and _sha256(path) != expected:
            raise ValueError(f"frozen Pass-1 source file changed during verification: {name}")
    return pass1, {
        "manifest": {"path": str(manifest_path), "sha256": manifest_hash},
        "pass1_weights": {"path": str(weights_path), "sha256": weights_hash},
        "pass1_schedule": {
            "path": str(schedule_path),
            "sha256": schedule_hash,
        },
        "request_count": expected_count,
    }


def prepare_pass2_successor(run_id: str = RUN_ID) -> dict[str, Any]:
    """Prepare a fresh Pass-2 run from the completed immutable AMES Pass 1."""
    global ENDPOINTS
    if TASK != "ames":
        raise ValueError("the Pass-2 successor is currently defined only for AMES")
    root = OUTPUT_ROOT / run_id
    if root.exists():
        raise FileExistsError(root)
    pass1, source_pass1 = _verify_frozen_pass1_source()
    inputs = _verify_inputs()
    inputs["source_pass1"] = source_pass1
    root.mkdir(parents=True)
    copied = (
        "pass1_schedule.parquet", "pass1_schedule.tsv",
        "pass1_weights.parquet", "pass1_weights.tsv",
        "bucket_prompt_payloads.json.gz",
    )
    for name in copied:
        shutil.copy2(PASS1_SOURCE_ROOT / name, root / name)
        inputs[f"source_pass1/{name}"] = {
            "path": str(PASS1_SOURCE_ROOT / name),
            "sha256": _sha256(PASS1_SOURCE_ROOT / name),
        }
    schedule = _pass2_schedule(pass1)
    _atomic_frame(schedule, root / "pass2_schedule.parquet")
    _atomic_frame(schedule, root / "pass2_schedule.tsv")
    with gzip.open(root / "bucket_prompt_payloads.json.gz", "rt", encoding="utf-8") as handle:
        payloads = json.load(handle)
    review = _write_pass2_review(root, schedule, payloads)
    candidates, inventory_hash = _provider_candidates()
    selected = [row for row in candidates if row["name"] == PASS2_LOCAL_ENDPOINT]
    probed = _probe_endpoint(selected[0]) if len(selected) == 1 else None
    if probed is None:
        raise ValueError(f"{PASS2_LOCAL_ENDPOINT} failed exact-model preflight")
    endpoint = {**probed, "max_inflight": PASS2_PROVIDER_CAPACITY}
    ENDPOINTS = (endpoint,)
    provider_path = root / "pass2_provider_profile.json"
    provider_profile = _write_fixed_provider_profile(provider_path)
    profile = {
        "version": f"{VERSION}.execution_profile",
        "model": MODEL, "reasoning_effort": "high", "max_tokens": MAX_TOKENS,
        "pass1_source": str(PASS1_SOURCE_ROOT),
        "pass2": {
            "batch_size": PASS2_BATCH_SIZE,
            "chain_widths": PASS2_CHAIN_WIDTHS,
            "fanout": PASS2_FANOUT,
            "required_valid_responses": PASS2_REQUIRED,
            "aggregation": speculative.aggregation_method(PASS2_REQUIRED),
            "replica_counts": provider_profile["fixed_replica_counts"],
            "progress_after_quorum": True,
            "store_all_replicas": True,
        },
        "pass2_max_logical_inflight": sum(PASS2_CHAIN_WIDTHS.values()),
        "pass2_max_inflight_per_provider_family": PASS2_PROVIDER_CAPACITY,
        "pass2_aggregate_replica_capacity": PASS2_PROVIDER_CAPACITY * 3,
        "provider_inventory": str(PROVIDER_INVENTORY),
        "provider_inventory_sha256": inventory_hash,
        "endpoints": [endpoint],
    }
    write_json_atomic(root / "execution_profile.json", profile)
    files = (
        *copied, "pass2_schedule.parquet", "pass2_schedule.tsv",
        "pass2_provider_profile.json", "execution_profile.json",
    )
    manifest = {
        "version": VERSION, "status": "awaiting_pass2_prompt_review",
        "run_id": run_id, "created_at": _now(), "task": TASK,
        "levels": list(LEVELS), "bucket_count": len(pass1),
        "pass1_request_count": len(pd.read_parquet(root / "pass1_schedule.parquet")),
        "pass2_request_count": len(schedule),
        "semantic_review_status": SEMANTIC_REVIEW_STATUS,
        "activation_allowed": False,
        "predecessors": [
            str(PASS1_SOURCE_ROOT),
            str(PASS1_SOURCE_ROOT.parent / "ames_morgan100_official_two_pass_v6_cards_v4_medium_luna_20260922"),
        ],
        "inputs": inputs, "pass2_review": review,
        "pass2_schedule_sha256": _sha256(root / "pass2_schedule.parquet"),
        "files": {name: _sha256(root / name) for name in files},
    }
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def _load_run(run_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    global ENDPOINTS
    root = OUTPUT_ROOT / run_id
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION:
        raise ValueError("run version changed")
    if manifest.get("task") != TASK:
        raise ValueError(f"run task differs from configured task: {manifest.get('task')} != {TASK}")
    _verify_inputs()
    for name, digest in manifest["files"].items():
        if _sha256(root / name) != digest:
            raise ValueError(f"prepared file changed: {name}")
    profile = json.loads((root / "execution_profile.json").read_text(encoding="utf-8"))
    if profile.get("reasoning_effort") != "high" or profile.get("model") != MODEL:
        raise ValueError("execution profile model or reasoning effort changed")
    _verify_selected_endpoints(profile)
    ENDPOINTS = tuple(profile["endpoints"])
    with gzip.open(root / "bucket_prompt_payloads.json.gz", "rt", encoding="utf-8") as handle:
        payloads = json.load(handle)
    return root, manifest, payloads


def _require_review(root: Path, directory: str, review_hash: str) -> None:
    if _sha256(root / directory / "manifest.json") != review_hash:
        raise ValueError(f"approved {directory} hash changed")


def _queue(connection: sqlite3.Connection, row: Mapping[str, Any], prompt: str,
           endpoint: Mapping[str, str], *, model: str = MODEL) -> str:
    candidates = json.loads(row["candidate_bucket_ids_json"])
    anchors = json.loads(row["anchor_bucket_ids_json"])
    phase = (
        f"{row['stage']}/{row['level']}/w{int(row['wave']):04d}/c{int(row['chain']):02d}"
        if "wave" in row else
        f"{row['stage']}/{row['level']}/{int(row['batch']):04d}"
    )
    request_id = core._request_id("weight_assignment", phase, prompt)
    validation = {"candidate_aliases": [f"Candidate {i}" for i in range(1, len(candidates) + 1)],
                  "candidate_bucket_ids": candidates, "anchor_bucket_ids": anchors,
                  "requested_model": model, "selected_endpoint": endpoint["name"]}
    if "execution_backend" in row:
        validation["execution_backend"] = row["execution_backend"]
    return core._queue_request(
        connection, request_id=request_id, kind="weight_assignment", phase=phase,
        prompt=prompt, reasoning_effort="high", max_tokens=MAX_TOKENS,
        validation=validation,
    )


async def _clients(max_logical_inflight: int,
                   fanout: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clients, receipts = {}, []
    for endpoint in ENDPOINTS:
        client, credential = async_openai_compatible_client(
            base_url=endpoint["base_url"], provider="local", env_file=None,
            max_connections=max_logical_inflight * fanout,
            timeout_s=3600, max_retries=0,
        )
        if credential:
            raise ValueError("local endpoint unexpectedly selected a credential")
        models = sorted(model.id for model in (await client.models.list()).data)
        if models != [MODEL]:
            await client.close()
            raise ValueError(f"endpoint model changed: {endpoint['name']}={models}")
        clients[endpoint["name"]] = client
        receipts.append({**endpoint, "advertised_models": models,
                         "max_inflight": max_logical_inflight})
    return clients, receipts


async def _execute_one(connection: sqlite3.Connection, request_id: str,
                       endpoint: Mapping[str, str], client: Any,
                       profile_hash: str, semaphore: asyncio.Semaphore,
                       *, fanout: int, required: int, model: str = MODEL) -> None:
    row = speculative._pending_rows(connection, [request_id])
    if not row:
        return
    async with semaphore:
        result = await speculative.execute_request(
            client, row[0], model=model, initial_fanout=fanout,
            hedge_seconds=3600, maximum=fanout, required=required,
        )
    speculative.persist_result(
        connection, result, base_url=endpoint["base_url"], model=model,
        benchmark_sha256=profile_hash,
    )
    if result["status"] != "complete":
        raise RuntimeError(f"request failed: {request_id}: {result['error']}")


def _scores(connection: sqlite3.Connection, stage: str) -> pd.DataFrame:
    aggregates = speculative.aggregate_lookup(connection)
    rows = []
    query = connection.execute(
        "SELECT * FROM requests WHERE status='complete' AND phase LIKE ? ORDER BY phase",
        (f"{stage}/%",),
    )
    for request in query:
        validation, response = json.loads(request["validation_json"]), json.loads(request["response_json"])
        buckets = dict(zip(validation["candidate_aliases"], validation["candidate_bucket_ids"], strict=True))
        aggregate = aggregates[request["request_id"]]
        for score in response["scores"]:
            component = aggregate["components"][score["candidate"]]
            rows.append({"task": TASK, "level": request["phase"].split("/")[1],
                         "semantic_bucket_id": buckets[score["candidate"]],
                         "weight": float(score["weight"]), "rationale": score["rationale"],
                         "request_id": request["request_id"],
                         "anchor_bucket_ids_json": core._canonical_json(validation["anchor_bucket_ids"]),
                         "endpoint": validation["selected_endpoint"],
                         "execution_backend": validation.get("execution_backend", "local_dgx"),
                         "weight_stddev": component["sample_stddev"],
                         "weight_min": component["min"], "weight_max": component["max"],
                         "component_weights_json": core._canonical_json(component["component_weights"])})
    return pd.DataFrame(rows)


async def _run_pass1(connection: sqlite3.Connection, root: Path,
                     payloads: Mapping[str, Any], clients: Mapping[str, Any],
                     profile_hash: str, max_inflight: int) -> pd.DataFrame:
    schedule = pd.read_parquet(root / "pass1_schedule.parquet")
    semaphores = {endpoint["name"]: asyncio.Semaphore(max_inflight)
                  for endpoint in ENDPOINTS}
    completed = {
        row[0] for row in connection.execute(
            "SELECT request_id FROM requests WHERE status='complete' AND phase LIKE 'pass1/%'"
        )
    }
    work = []
    for index, row in enumerate(schedule.to_dict("records")):
        endpoint = ENDPOINTS[index % len(ENDPOINTS)]
        candidates = json.loads(row["candidate_bucket_ids_json"])
        prompt = _render(candidates, [], row["level"], payloads, {})
        phase = f"{row['stage']}/{row['level']}/{int(row['batch']):04d}"
        if core._request_id("weight_assignment", phase, prompt) in completed:
            continue
        request_id = _queue(connection, row, prompt, endpoint)
        work.append(_execute_one(connection, request_id, endpoint, clients[endpoint["name"]],
                                 profile_hash, semaphores[endpoint["name"]],
                                 fanout=PASS1_FANOUT, required=PASS1_REQUIRED))
    await asyncio.gather(*work, return_exceptions=True)
    scores = _scores(connection, "pass1")
    if len(scores) != EXPECTED_BUCKET_COUNT or scores.semantic_bucket_id.duplicated().any():
        raise ValueError(f"pass one incomplete: {len(scores)}/{EXPECTED_BUCKET_COUNT}")
    _atomic_frame(scores, root / "pass1_weights.parquet")
    _atomic_frame(scores, root / "pass1_weights.tsv")
    return scores


async def _execute_fixed_one(
    connection: sqlite3.Connection, request_id: str,
    targets: Sequence[tuple[Any, str]], profile_hash: str,
    drains: list[asyncio.Task[None]],
) -> None:
    row = connection.execute(
        "SELECT * FROM requests WHERE request_id=?", (request_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"queued Pass-2 request disappeared: {request_id}")
    if row["status"] == "complete" and speculative.all_replicas_stored(
        connection, request_id, expected_replicas=PASS2_FANOUT
    ):
        return
    result, drain = await speculative.execute_fixed_request(
        targets, row, required=PASS2_REQUIRED
    )
    speculative.persist_result(
        connection, result, base_url="mixed://openrouter+dgx027_50002",
        model="fixed_mixed", benchmark_sha256=profile_hash,
    )
    if drain is not None:
        async def store_late() -> None:
            speculative.persist_late_receipts(connection, result, await drain)

        drains.append(asyncio.create_task(store_late()))
    if result["status"] != "complete":
        raise RuntimeError(f"request failed: {request_id}: {result['error']}")


async def _run_chain(
    level_rows: pd.DataFrame, connection: sqlite3.Connection,
    payloads: Mapping[str, Any], targets: Sequence[tuple[Any, str]],
    profile_hash: str, drains: list[asyncio.Task[None]],
) -> None:
    for wave, rows in level_rows.groupby("wave", sort=True):
        existing = _scores(connection, "pass2")
        score_map = (existing.set_index("semantic_bucket_id").to_dict("index")
                     if len(existing) else {})
        work = []
        for row in rows.sort_values("chain").to_dict("records"):
            candidates = json.loads(row["candidate_bucket_ids_json"])
            candidate_scores = [score_map.get(bucket) for bucket in candidates]
            if all(candidate_scores) and all(
                speculative.all_replicas_stored(
                    connection, score["request_id"], expected_replicas=PASS2_FANOUT
                )
                for score in candidate_scores
            ):
                continue
            anchors = json.loads(row["anchor_bucket_ids_json"])
            if not set(anchors) <= score_map.keys():
                raise ValueError(f"pass-two anchors are incomplete: {row['level']}/{wave}")
            prompt = _render(candidates, anchors, row["level"], payloads, score_map)
            endpoint = {"name": "fixed_mixed", "base_url": "mixed://openrouter+dgx027_50002"}
            request_id = _queue(connection, row, prompt, endpoint, model="fixed_mixed")
            work.append(_execute_fixed_one(
                connection, request_id, targets, profile_hash, drains
            ))
        await asyncio.gather(*work)


def _pool_adapter(config: Any) -> Any:
    pool = build_provider_pool(
        config, env_file=DEFAULT_ENV_FILE, timeout_s=3_600,
        max_tokens=MAX_TOKENS, temperature=None,
        tool_service_url="http://127.0.0.1:1", enable_group_tools=False,
        max_tool_rounds=0, reasoning_effort="high", enable_thinking=False,
        transport_max_retries=0, response_format={"type": "json_object"},
    )
    adapter = core._ProviderPoolCompletionAdapter(pool)

    async def create(**kwargs: Any) -> Any:
        return await asyncio.to_thread(adapter.create, **kwargs)

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        snapshot=adapter.snapshot,
    )


def _prepare_remote_clients(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = root / "pass2_provider_profile.json"
    config = load_provider_pool_config(config_path)
    checks = preflight_provider_models(config)
    clients = {}
    for provider in config.providers:
        single = replace(config, providers=(provider,), max_failovers=0)
        clients[provider.name] = _pool_adapter(single)
    receipt = {
        "config_path": str(config_path), "config_sha256": _sha256(config_path),
        "requested_models": sorted({row.model for row in config.providers}),
        "reasoning_effort_by_model": {
            openrouter_provider_pool.DEFAULT_MODEL: "high",
            "openai/gpt-6-luna": "medium",
        },
        "max_inflight": primary_capacity(config), "preflight": checks,
        "pool": config.public_dict(),
    }
    return clients, receipt


async def _run_pass1_async(connection: sqlite3.Connection, root: Path,
                           payloads: Mapping[str, Any], manifest: dict[str, Any],
                           max_inflight: int) -> None:
    clients, preflight = await _clients(max_inflight, PASS1_FANOUT)
    profile_hash = _sha256(root / "execution_profile.json")
    manifest["endpoint_preflight"] = preflight
    write_json_atomic(root / "manifest.json", manifest)
    try:
        await _run_pass1(
            connection, root, payloads, clients, profile_hash, max_inflight
        )
    finally:
        await asyncio.gather(*(client.close() for client in clients.values()))


async def _run_pass2_async(connection: sqlite3.Connection, root: Path,
                           payloads: Mapping[str, Any], manifest: dict[str, Any]) -> None:
    local_clients, preflight = await _clients(PASS2_PROVIDER_CAPACITY, 2)
    remote_clients, remote_receipt = await asyncio.to_thread(
        _prepare_remote_clients, root
    )
    luna = remote_clients["openrouter_gpt-6-luna_flex"]
    together_name = next(name for name in remote_clients if "together" in name)
    together = remote_clients[together_name]
    provider_profile = json.loads(
        (root / "pass2_provider_profile.json").read_text(encoding="utf-8")
    )
    local_endpoint = provider_profile.get("local_endpoint", PASS2_LOCAL_ENDPOINT)
    try:
        local = local_clients[local_endpoint]
    except KeyError as error:
        raise ValueError(f"Pass-2 local endpoint was not prepared: {local_endpoint}") from error
    targets = [
        (luna, "openai/gpt-6-luna"), (luna, "openai/gpt-6-luna"),
        (together, openrouter_provider_pool.DEFAULT_MODEL),
        (together, openrouter_provider_pool.DEFAULT_MODEL),
        (local, MODEL), (local, MODEL),
    ]
    schedule = pd.read_parquet(root / "pass2_schedule.parquet")
    pass2_profile = {
        "version": f"{VERSION}.pass2_execution_profile",
        "reasoning_effort": {
            "local_dgx": "high",
            openrouter_provider_pool.DEFAULT_MODEL: "high",
            "openai/gpt-6-luna": "medium",
        },
        "fanout": PASS2_FANOUT,
        "required_valid_responses": PASS2_REQUIRED,
        "aggregation": speculative.aggregation_method(PASS2_REQUIRED),
        "prepared_policy": json.loads(
            (root / "execution_profile.json").read_text(encoding="utf-8")
        )["pass2"],
        "policy_change_reason": "user-approved before any Pass-2 request",
        "schedule_sha256": _sha256(root / "pass2_schedule.parquet"),
        "logical_request_counts": {
            str(level): int(count) for level, count in schedule.groupby("level").size().items()
        },
        "local_execution_profile_sha256": _sha256(root / "execution_profile.json"),
        "remote_provider_pool_sha256": remote_receipt["config_sha256"],
        "remote_models": remote_receipt["requested_models"],
        "remote_capacity": remote_receipt["max_inflight"],
    }
    pass2_profile_path = root / "pass2_execution_profile.json"
    if pass2_profile_path.exists():
        existing = json.loads(pass2_profile_path.read_text(encoding="utf-8"))
        if existing != pass2_profile:
            raise ValueError("frozen Pass-2 execution profile changed")
    else:
        write_json_atomic(pass2_profile_path, pass2_profile)
    profile_hash = _sha256(pass2_profile_path)
    manifest["pass2_endpoint_preflight"] = {
        "local_dgx": preflight,
        "remote_pool": remote_receipt,
    }
    write_json_atomic(root / "manifest.json", manifest)
    drains: list[asyncio.Task[None]] = []
    try:
        outcomes = await asyncio.gather(*(
            _run_chain(rows, connection, payloads, targets, profile_hash, drains)
            for _, rows in schedule.groupby("level", sort=True)
        ), return_exceptions=True)
        if drains:
            await asyncio.gather(*drains)
        failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        if failures:
            raise RuntimeError("one or more task levels stopped") from failures[0]
        write_json_atomic(root / "pass2_provider_final_snapshot.json", {
            name: client.snapshot() for name, client in remote_clients.items()
        })
    finally:
        await asyncio.gather(*(client.close() for client in local_clients.values()))


def _write_pass2_review(root: Path, schedule: pd.DataFrame,
                        payloads: Mapping[str, Any]) -> dict[str, Any]:
    review = root / "pass2_prompt_review"
    review.mkdir()
    files = []
    for level, rows in schedule.groupby("level", sort=True):
        ordered = rows.sort_values(["wave", "chain"])
        examples = [*ordered[ordered.wave.eq(0)].to_dict("records")]
        anchored = ordered[ordered.wave.eq(1)]
        if len(anchored):
            examples.append(anchored.iloc[0].to_dict())
        for row in examples:
            candidates = json.loads(row["candidate_bucket_ids_json"])
            anchors = json.loads(row["anchor_bucket_ids_json"])
            prompt = _render(candidates, anchors, str(level), payloads, _preview(anchors))
            shape = "seed" if not anchors else "anchored"
            path = review / (
                f"{level}_w{int(row['wave']):04d}_c{int(row['chain']):02d}_{shape}.txt"
            )
            path.write_text(prompt + "\n", encoding="utf-8")
            files.append({
                "path": path.name, "sha256": _sha256(path),
                "characters": len(prompt), "batch": int(row["batch"]),
                "wave": int(row["wave"]), "chain": int(row["chain"]),
                "candidate_count": len(candidates), "anchor_count": len(anchors),
            })
    manifest = {"version": f"{VERSION}.pass2_prompt_review", "status": "ready",
                "completion_requests_made": 0, "files": files}
    write_json_atomic(review / "manifest.json", manifest)
    return manifest


def _require_complete_pass2_receipts(connection: sqlite3.Connection) -> None:
    request_ids = [
        row["request_id"] for row in connection.execute(
            "SELECT request_id FROM requests "
            "WHERE phase LIKE 'pass2/%' AND status='complete'"
        )
    ]
    incomplete = [
        request_id for request_id in request_ids
        if not speculative.all_replicas_stored(
            connection, request_id, expected_replicas=PASS2_FANOUT
        )
    ]
    if incomplete:
        raise ValueError(
            "Pass-2 cannot publish before all six replica receipts are stored: "
            f"{len(incomplete)} incomplete requests"
        )


def _publish(root: Path, connection: sqlite3.Connection,
             manifest: dict[str, Any]) -> dict[str, Any]:
    _require_complete_pass2_receipts(connection)
    pass1 = pd.read_parquet(root / "pass1_weights.parquet")
    final = _scores(connection, "pass2")
    if (
        len(pass1) != EXPECTED_BUCKET_COUNT
        or len(final) != EXPECTED_BUCKET_COUNT
        or final.semantic_bucket_id.duplicated().any()
    ):
        raise ValueError(f"two-pass result incomplete: pass1={len(pass1)}, pass2={len(final)}")
    order = pass1.sort_values(["level", "weight", "semantic_bucket_id"], ascending=[True, False, True])
    order["pass1_order_rank"] = order.groupby("level").cumcount() + 1
    final = final.merge(order[["semantic_bucket_id", "pass1_order_rank"]], on="semantic_bucket_id")
    final = final.sort_values(["level", "weight", "semantic_bucket_id"], ascending=[True, False, True])
    final["final_rank"] = final.groupby("level").cumcount() + 1
    _atomic_frame(final, root / "weights.parquet")
    _atomic_frame(final, root / "weights.tsv")
    ranking = final[["task", "level", "semantic_bucket_id", "final_rank", "weight",
                     "rationale", "pass1_order_rank"]]
    _atomic_frame(ranking, root / "semantic_bucket_rankings.parquet")
    _atomic_frame(ranking, root / "semantic_bucket_rankings.tsv")
    summary = final.groupby("level").weight.agg(["count", "min", "mean", "max"]).reset_index()
    _atomic_frame(summary, root / "summary.tsv")
    manifest.update(status=FINAL_STATUS, completed_at=_now(), completed_bucket_count=len(final),
                    active_policy_changed=False,
                    artifacts={name: _sha256(root / name) for name in
                               ("pass1_weights.parquet", "pass1_weights.tsv", "weights.parquet",
                                "weights.tsv", "semantic_bucket_rankings.parquet",
                                "semantic_bucket_rankings.tsv", "summary.tsv")})
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def run_pass1(run_id: str, review_hash: str,
              endpoint_name: str | None = None) -> dict[str, Any]:
    global ENDPOINTS
    if os.environ.get("DEEPSEEK_API_KEY") != "EMPTY":
        raise ValueError("set DEEPSEEK_API_KEY=EMPTY for local DGX execution")
    root, manifest, payloads = _load_run(run_id)
    _require_review(root, "prompt_review", review_hash)
    max_inflight = PASS1_MAX_LOGICAL_INFLIGHT
    if endpoint_name:
        selected = tuple(row for row in ENDPOINTS if row["name"] == endpoint_name)
        if len(selected) != 1:
            raise ValueError(f"unknown prepared endpoint: {endpoint_name}")
        ENDPOINTS = selected
        max_inflight = int(selected[0]["max_inflight"])
        manifest["pass1_runtime_endpoint_override"] = {
            "endpoint": endpoint_name,
            "max_logical_inflight": max_inflight,
        }
    manifest.update(status="running_pass1", started_at=manifest.get("started_at", _now()))
    write_json_atomic(root / "manifest.json", manifest)
    connection = core._request_database(root / "requests.sqlite3", journal_mode="DELETE")
    speculative.ensure_tables(connection)
    if endpoint_name:
        migrated = 0
        for row in connection.execute(
            "SELECT request_id,validation_json FROM requests "
            "WHERE phase LIKE 'pass1/%' AND status!='complete'"
        ).fetchall():
            validation = json.loads(row["validation_json"])
            if validation["selected_endpoint"] != endpoint_name:
                validation["selected_endpoint"] = endpoint_name
                connection.execute(
                    "UPDATE requests SET validation_json=? WHERE request_id=?",
                    (core._canonical_json(validation), row["request_id"]),
                )
                migrated += 1
        connection.commit()
        manifest["pass1_runtime_endpoint_override"]["migrated_incomplete_requests"] = migrated
        write_json_atomic(root / "manifest.json", manifest)
    try:
        asyncio.run(_run_pass1_async(
            connection, root, payloads, manifest, max_inflight
        ))
        pass1 = _scores(connection, "pass1")
        schedule = _pass2_schedule(pass1)
        _atomic_frame(schedule, root / "pass2_schedule.parquet")
        _atomic_frame(schedule, root / "pass2_schedule.tsv")
        review = _write_pass2_review(root, schedule, payloads)
        manifest.update(status="awaiting_pass2_prompt_review",
                        pass2_request_count=len(schedule), pass2_review=review,
                        pass2_schedule_sha256=_sha256(root / "pass2_schedule.parquet"))
        write_json_atomic(root / "manifest.json", manifest)
        return manifest
    except Exception:
        manifest["status"] = "incomplete_pass1"
        write_json_atomic(root / "manifest.json", manifest)
        raise
    finally:
        connection.close()


def run_pass2(run_id: str, review_hash: str) -> dict[str, Any]:
    if os.environ.get("DEEPSEEK_API_KEY") != "EMPTY":
        raise ValueError("set DEEPSEEK_API_KEY=EMPTY for local DGX execution")
    root, manifest, payloads = _load_run(run_id)
    _require_review(root, "pass2_prompt_review", review_hash)
    manifest.update(status="running_pass2", pass2_started_at=_now())
    write_json_atomic(root / "manifest.json", manifest)
    connection = core._request_database(root / "requests.sqlite3", journal_mode="DELETE")
    speculative.ensure_tables(connection)
    try:
        asyncio.run(_run_pass2_async(connection, root, payloads, manifest))
        return _publish(root, connection, manifest)
    except Exception:
        manifest["status"] = "incomplete_pass2"
        write_json_atomic(root / "manifest.json", manifest)
        raise
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "prepare-pass2-successor", "run-pass1", "run-pass2")
    )
    parser.add_argument("--task", choices=tuple(TASK_CONFIGS), default="skin_reaction")
    parser.add_argument("--run-id")
    parser.add_argument("--approved-review-sha256")
    parser.add_argument("--endpoint-name")
    args = parser.parse_args()
    configure_task(args.task)
    run_id = args.run_id or RUN_ID
    if args.command == "prepare":
        result = prepare(run_id)
    elif args.command == "prepare-pass2-successor":
        result = prepare_pass2_successor(run_id)
    elif args.command == "run-pass1":
        if not args.approved_review_sha256:
            parser.error("run-pass1 requires --approved-review-sha256")
        result = run_pass1(run_id, args.approved_review_sha256, args.endpoint_name)
    else:
        if not args.approved_review_sha256:
            parser.error("run-pass2 requires --approved-review-sha256")
        result = run_pass2(run_id, args.approved_review_sha256)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
