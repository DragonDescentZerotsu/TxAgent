"""Assign official two-pass weights to a frozen semantic-bucket retrieval world."""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from urllib.request import urlopen

from jinja2 import Environment, StrictUndefined
import pandas as pd

from data.processing.llm_api import async_openai_compatible_client
from data.processing import openrouter_provider_pool
from semantic_buckets import bioavailability_semantic_readout_v1 as core
from semantic_buckets import sliding_weight_assignment as display
from semantic_buckets import speculative_weight_execution as speculative
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "semantic_weight_official_two_pass.v3"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
TARGET = "skin sensitization or allergic contact dermatitis"
MAX_TOKENS = 20_480
PASS1_FANOUT = 1
PASS1_REQUIRED = 1
PASS2_FANOUT = 4
PASS2_REQUIRED = 2
PASS1_MAX_LOGICAL_INFLIGHT = 32
PASS2_MAX_LOGICAL_INFLIGHT = 128
OPENROUTER_CAPACITY = 35
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
OUTPUT_ROOT = ROOT / "provenance/semantic_weight_two_pass_v3"
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
        "run_id": "ames_morgan100_official_two_pass_v6_cards_v3_20260922",
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
        "max_inflight": min(PASS2_MAX_LOGICAL_INFLIGHT, int(endpoint["max_inflight"])),
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
        rows.append({"stage": "pass2", "level": level, "batch": 0,
                     "candidate_bucket_ids_json": core._canonical_json(buckets[:12]),
                     "anchor_bucket_ids_json": "[]"})
        anchors = buckets[7:12]
        for batch, start in enumerate(range(12, len(buckets), 7), 1):
            candidates = buckets[start:start + 7]
            rows.append({"stage": "pass2", "level": level, "batch": batch,
                         "candidate_bucket_ids_json": core._canonical_json(candidates),
                         "anchor_bucket_ids_json": core._canonical_json(anchors)})
            anchors = candidates[-2:]
    schedule = pd.DataFrame(rows).sort_values(["level", "batch"]).reset_index(drop=True)
    schedule["execution_backend"] = [
        "local_dgx" if index % 2 == 0 else "openrouter"
        for index in range(len(schedule))
    ]
    return schedule


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
    profile = {"model": MODEL, "reasoning_effort": "high", "max_tokens": MAX_TOKENS,
               "pass1": {"fanout": PASS1_FANOUT,
                         "required_valid_responses": PASS1_REQUIRED,
                         "aggregation": speculative.aggregation_method(PASS1_REQUIRED)},
               "pass2": {"fanout": PASS2_FANOUT,
                         "required_valid_responses": PASS2_REQUIRED,
                         "aggregation": speculative.aggregation_method(PASS2_REQUIRED)},
               "pass1_max_logical_inflight_per_endpoint": PASS1_MAX_LOGICAL_INFLIGHT,
               "pass1_aggregate_logical_capacity": len(ENDPOINTS) * PASS1_MAX_LOGICAL_INFLIGHT,
               "pass2_max_logical_inflight_per_endpoint": PASS2_MAX_LOGICAL_INFLIGHT,
               "pass2_aggregate_logical_capacity": len(ENDPOINTS) * PASS2_MAX_LOGICAL_INFLIGHT,
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
    phase = f"{row['stage']}/{row['level']}/{int(row['batch']):04d}"
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


async def _clients() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clients, receipts = {}, []
    for endpoint in ENDPOINTS:
        client, credential = async_openai_compatible_client(
            base_url=endpoint["base_url"], provider="local", env_file=None,
            max_connections=PASS2_MAX_LOGICAL_INFLIGHT * PASS2_FANOUT,
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
                         "max_inflight": PASS2_MAX_LOGICAL_INFLIGHT})
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
                     profile_hash: str) -> pd.DataFrame:
    schedule = pd.read_parquet(root / "pass1_schedule.parquet")
    semaphores = {endpoint["name"]: asyncio.Semaphore(PASS1_MAX_LOGICAL_INFLIGHT)
                  for endpoint in ENDPOINTS}
    work = []
    for index, row in enumerate(schedule.to_dict("records")):
        endpoint = ENDPOINTS[index % len(ENDPOINTS)]
        candidates = json.loads(row["candidate_bucket_ids_json"])
        prompt = _render(candidates, [], row["level"], payloads, {})
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


async def _run_chain(level_rows: pd.DataFrame, connection: sqlite3.Connection,
                     payloads: Mapping[str, Any], clients: Mapping[str, Any],
                     profile_hash: str, level_offset: int) -> None:
    for position, row in enumerate(level_rows.sort_values("batch").to_dict("records")):
        existing = _scores(connection, "pass2")
        score_map = (existing.set_index("semantic_bucket_id").to_dict("index")
                     if len(existing) else {})
        candidates = json.loads(row["candidate_bucket_ids_json"])
        if set(candidates) <= score_map.keys():
            continue
        anchors = json.loads(row["anchor_bucket_ids_json"])
        if not set(anchors) <= score_map.keys():
            raise ValueError(f"pass-two anchors are incomplete: {row['level']}/{row['batch']}")
        backend = row["execution_backend"]
        if backend == "openrouter":
            endpoint = {
                "name": "openrouter_qualified_pool",
                "base_url": openrouter_provider_pool.OPENROUTER_URL,
            }
            model = openrouter_provider_pool.DEFAULT_MODEL
        else:
            endpoint = ENDPOINTS[(position + level_offset) % len(ENDPOINTS)]
            model = MODEL
        prompt = _render(candidates, anchors, row["level"], payloads, score_map)
        request_id = _queue(connection, row, prompt, endpoint, model=model)
        client = clients["openrouter"] if backend == "openrouter" else clients["local_dgx"][endpoint["name"]]
        await _execute_one(connection, request_id, endpoint, client,
                           profile_hash, asyncio.Semaphore(1),
                           fanout=PASS2_FANOUT, required=PASS2_REQUIRED, model=model)


def _prepare_openrouter_pool(root: Path) -> tuple[Any, dict[str, Any]]:
    from data.processing.llm_api import DEFAULT_ENV_FILE
    from predict.api_client.pool import (
        build_provider_pool,
        load_provider_pool_config,
        preflight_provider_models,
        primary_capacity,
    )

    config_path = root / "pass2_openrouter_provider_pool.json"
    if not config_path.exists():
        openrouter_provider_pool.export_provider_pool(
            config_path, OPENROUTER_CAPACITY, credential_env="OPEN_ROUTER_KEY_TWO"
        )
    config = load_provider_pool_config(config_path)
    checks = preflight_provider_models(config)
    pool = build_provider_pool(
        config,
        env_file=DEFAULT_ENV_FILE,
        timeout_s=3_600,
        max_tokens=MAX_TOKENS,
        temperature=None,
        tool_service_url="http://127.0.0.1:1",
        enable_group_tools=False,
        max_tool_rounds=0,
        reasoning_effort="high",
        enable_thinking=True,
        transport_max_retries=0,
        response_format={"type": "json_object"},
    )
    adapter = core._ProviderPoolCompletionAdapter(pool)

    async def create(**kwargs: Any) -> Any:
        return await asyncio.to_thread(adapter.create, **kwargs)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    receipt = {
        "name": "openrouter_qualified_pool",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "requested_model": openrouter_provider_pool.DEFAULT_MODEL,
        "reasoning_effort": "high",
        "max_inflight": primary_capacity(config),
        "preflight": checks,
        "pool": config.public_dict(),
    }
    return client, receipt


async def _run_pass1_async(connection: sqlite3.Connection, root: Path,
                           payloads: Mapping[str, Any], manifest: dict[str, Any]) -> None:
    clients, preflight = await _clients()
    profile_hash = _sha256(root / "execution_profile.json")
    manifest["endpoint_preflight"] = preflight
    write_json_atomic(root / "manifest.json", manifest)
    try:
        await _run_pass1(connection, root, payloads, clients, profile_hash)
    finally:
        await asyncio.gather(*(client.close() for client in clients.values()))


async def _run_pass2_async(connection: sqlite3.Connection, root: Path,
                           payloads: Mapping[str, Any], manifest: dict[str, Any]) -> None:
    local_clients, preflight = await _clients()
    openrouter_client, openrouter_receipt = await asyncio.to_thread(
        _prepare_openrouter_pool, root
    )
    clients = {
        "local_dgx": local_clients,
        "openrouter": openrouter_client,
    }
    schedule = pd.read_parquet(root / "pass2_schedule.parquet")
    backend_counts = {
        str(name): int(count)
        for name, count in schedule.execution_backend.value_counts().items()
    }
    pass2_profile = {
        "version": f"{VERSION}.pass2_execution_profile",
        "reasoning_effort": "high",
        "fanout": PASS2_FANOUT,
        "required_valid_responses": PASS2_REQUIRED,
        "aggregation": speculative.aggregation_method(PASS2_REQUIRED),
        "prepared_policy": json.loads(
            (root / "execution_profile.json").read_text(encoding="utf-8")
        )["pass2"],
        "policy_change_reason": "user-approved before any Pass-2 request",
        "schedule_sha256": _sha256(root / "pass2_schedule.parquet"),
        "logical_request_counts": backend_counts,
        "local_execution_profile_sha256": _sha256(root / "execution_profile.json"),
        "openrouter_provider_pool_sha256": openrouter_receipt["config_sha256"],
        "openrouter_model": openrouter_provider_pool.DEFAULT_MODEL,
        "openrouter_capacity": openrouter_receipt["max_inflight"],
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
        "openrouter": openrouter_receipt,
    }
    write_json_atomic(root / "manifest.json", manifest)
    try:
        await asyncio.gather(*(
            _run_chain(rows, connection, payloads, clients, profile_hash, index)
            for index, (_, rows) in enumerate(schedule.groupby("level", sort=True))
        ))
    finally:
        await asyncio.gather(*(client.close() for client in local_clients.values()))


def _write_pass2_review(root: Path, schedule: pd.DataFrame,
                        payloads: Mapping[str, Any]) -> dict[str, Any]:
    review = root / "pass2_prompt_review"
    review.mkdir()
    files = []
    for level, rows in schedule.groupby("level", sort=True):
        seed = rows.sort_values("batch").iloc[0]
        candidates = json.loads(seed.candidate_bucket_ids_json)
        prompt = _render(candidates, [], str(level), payloads, {})
        path = review / f"{level}_seed.txt"
        path.write_text(prompt + "\n", encoding="utf-8")
        files.append({"path": path.name, "sha256": _sha256(path),
                      "characters": len(prompt), "batch": int(seed.batch)})
    manifest = {"version": f"{VERSION}.pass2_prompt_review", "status": "ready",
                "completion_requests_made": 0, "files": files}
    write_json_atomic(review / "manifest.json", manifest)
    return manifest


def _publish(root: Path, connection: sqlite3.Connection,
             manifest: dict[str, Any]) -> dict[str, Any]:
    pass1, final = _scores(connection, "pass1"), _scores(connection, "pass2")
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


def run_pass1(run_id: str, review_hash: str) -> dict[str, Any]:
    if os.environ.get("DEEPSEEK_API_KEY") != "EMPTY":
        raise ValueError("set DEEPSEEK_API_KEY=EMPTY for local DGX execution")
    root, manifest, payloads = _load_run(run_id)
    _require_review(root, "prompt_review", review_hash)
    manifest.update(status="running_pass1", started_at=manifest.get("started_at", _now()))
    write_json_atomic(root / "manifest.json", manifest)
    connection = core._request_database(root / "requests.sqlite3", journal_mode="DELETE")
    speculative.ensure_tables(connection)
    try:
        asyncio.run(_run_pass1_async(connection, root, payloads, manifest))
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
    parser.add_argument("command", choices=("prepare", "run-pass1", "run-pass2"))
    parser.add_argument("--task", choices=tuple(TASK_CONFIGS), default="skin_reaction")
    parser.add_argument("--run-id")
    parser.add_argument("--approved-review-sha256")
    args = parser.parse_args()
    configure_task(args.task)
    run_id = args.run_id or RUN_ID
    if args.command == "prepare":
        result = prepare(run_id)
    elif args.command == "run-pass1":
        if not args.approved_review_sha256:
            parser.error("run-pass1 requires --approved-review-sha256")
        result = run_pass1(run_id, args.approved_review_sha256)
    else:
        if not args.approved_review_sha256:
            parser.error("run-pass2 requires --approved-review-sha256")
        result = run_pass2(run_id, args.approved_review_sha256)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
