"""Calibrate the upper V5 semantic-weight shelf with enriched bucket cards."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import threading
from typing import Any
from urllib.parse import urlsplit

from jinja2 import Environment, StrictUndefined
import pandas as pd

from data.processing import openrouter_provider_pool as provider_pool
from semantic_buckets import bioavailability_semantic_readout_v1 as core
from semantic_buckets import sliding_weight_assignment as v5
from semantic_buckets import speculative_weight_execution as speculative
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "semantic_weight_calibration.v6"
ROOT = Path(__file__).resolve().parent
PROMPT_ROOT = ROOT / "prompts/semantic_weight_calibration_v6"
PROVENANCE_ROOT = ROOT / "provenance/semantic_weight_calibration_v6"
SEED_MODEL = "deepseek/deepseek-v4-pro"
FOLLOWUP_MODEL = "deepseek/deepseek-v4-flash-0731"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
MAX_TOKENS = 20_480
TEXT_LIMIT = 240
CHAIN_WIDTH = 2
SOURCE_LABELS = {
    "direct_bbb": "Direct blood-brain barrier evidence",
    "efflux_transport": "Efflux transporter evidence",
    "influx_transport": "Influx transporter evidence",
    "passive_permeability": "Passive permeability evidence",
    "hf_bioavailability": "Oral bioavailability measurement evidence",
    "oral_exposure": "Oral systemic exposure evidence",
    "fa": "Fraction absorbed evidence",
    "fg": "Intestinal availability evidence",
    "fh": "Hepatic availability evidence",
}
ENRICHED_COLUMNS = {
    "bbb_martins": (
        "canonical_reference_scope", "canonical_reference_basis", "assay_model", "species",
        "assay_system", "biological_system", "perturbation", "evidence_basis",
        "canonical_transporter_identifier", "mediator_identifier", "mediator_name",
        "qualifying_conditions",
    ),
    "bioavailability_ma": (
        "canonical_reference_scope", "study_context", "oral_dose", "assay_system",
        "species", "species_or_population",
        "canonical_biological_matrix", "formulation_or_solid_form",
        "comparator_exposure", "transporter_or_enzyme", "substrate_status",
        "enzyme_or_pathway", "qualifying_conditions",
    ),
}


def _run_root(run_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", run_id):
        raise ValueError("run_id contains unsupported characters")
    root = (PROVENANCE_ROOT / run_id).resolve()
    if not root.is_relative_to(PROVENANCE_ROOT.resolve()):
        raise ValueError("run_id escapes the provenance root")
    return root


def _v5_root(run_id: str) -> Path:
    return v5.PROVENANCE_ROOT / run_id


def _select_task_levels(
    frame: pd.DataFrame, schedule: pd.DataFrame,
    task_levels: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    available = set(zip(schedule.task.astype(str), schedule.level.astype(str)))
    if missing := set(task_levels) - available:
        raise ValueError(f"unknown V5 task-levels: {sorted(missing)}")
    completed = set(frame.semantic_bucket_id.astype(str))
    for task, level in task_levels:
        rows = schedule[schedule.task.eq(task) & schedule.level.eq(level)]
        expected = {bucket for value in rows.candidate_bucket_ids_json
                    for bucket in json.loads(value)}
        if not expected <= completed:
            raise ValueError(f"V5 task-level is incomplete: {task}/{level}")
    wanted = set(task_levels)
    return frame[frame.apply(lambda row: (str(row.task), str(row.level)) in wanted,
                             axis=1)].copy()


def _load_v5(
    run_id: str, task_levels: Sequence[tuple[str, str]] = (),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = _v5_root(run_id)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not task_levels and manifest.get("status") != "complete_candidate":
        raise ValueError(f"V5 must be complete_candidate, found {manifest.get('status')!r}")
    weights_path = root / ("weights.parquet" if manifest.get("status") == "complete_candidate"
                           else "weights.partial.parquet")
    if not task_levels:
        expected_hash = manifest.get("artifacts", {}).get("weights.parquet")
        if not expected_hash or v5._sha256(weights_path) != expected_hash:
            raise ValueError("V5 weights are missing or do not match their published hash")
    frame = pd.read_parquet(weights_path)
    required = {"semantic_bucket_id", "task", "level", "level_rank", "weight", "rationale"}
    if not required <= set(frame) or frame.semantic_bucket_id.duplicated().any():
        raise ValueError("V5 weights have an invalid schema or duplicate buckets")
    if not task_levels and len(frame) != manifest.get("bucket_count"):
        raise ValueError("V5 weights do not cover the declared bucket universe")
    if task_levels:
        frame = _select_task_levels(
            frame, pd.read_parquet(root / "schedule.parquet"), task_levels
        )
    return frame, manifest


def _selected(group: pd.DataFrame, fraction: float = 0.5) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError("selection fraction must be in (0, 1]")
    count = min(len(group), max(32, math.ceil(len(group) * fraction)))
    return group.sort_values(
        ["weight", "level_rank", "semantic_bucket_id"],
        ascending=[False, True, True], kind="stable",
    ).head(count).reset_index(drop=True)


def _schedule_level(task: str, level: str, bucket_ids: list[str]) -> list[dict[str, Any]]:
    if len(bucket_ids) < 12:
        raise ValueError(f"{task}/{level} has fewer than 12 calibration buckets")
    rows = [v5._schedule_row(task, level, 0, 0, bucket_ids[:12], [])]
    remaining, anchors, round_index = bucket_ids[12:], bucket_ids[7:12], 1
    while remaining:
        block, remaining = remaining[: 7 * CHAIN_WIDTH], remaining[7 * CHAIN_WIDTH :]
        batches = [block[start : start + 7] for start in range(0, len(block), 7)]
        rows.extend(v5._schedule_row(task, level, round_index, index, batch, anchors)
                    for index, batch in enumerate(batches))
        anchors = [bucket for batch in batches for bucket in batch[-2:]]
        round_index += 1
    return rows


def build_schedule(
    v5_weights: pd.DataFrame, selection_fraction: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for (task, level), group in v5_weights.groupby(["task", "level"], sort=True):
        chosen = _selected(group, selection_fraction)
        ids = chosen.semantic_bucket_id.astype(str).tolist()
        original_ranks = dict(zip(chosen.semantic_bucket_id, chosen.level_rank, strict=True))
        for row in _schedule_level(str(task), str(level), ids):
            candidates = json.loads(row["candidate_bucket_ids_json"])
            row["candidate_calibration_ranks_json"] = core._canonical_json(
                [ids.index(bucket) + 1 for bucket in candidates]
            )
            row["candidate_original_level_ranks_json"] = core._canonical_json(
                [int(original_ranks[bucket]) for bucket in candidates]
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["round", "task", "level", "batch"]
    ).reset_index(drop=True)


def _usable(value: Any) -> bool:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return False
    text = str(value).strip().lower()
    return bool(text) and not text.startswith("__not_present") and text != "not_applicable"


def _display(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("examples", value)
    if isinstance(value, list):
        text = " | ".join(_display(item) for item in value if _usable(item))
    else:
        text = v5._clean(value)
    if text.lower() in {"__unknown__", "unknown"}:
        text = "not reported"
    return text if len(text) <= TEXT_LIMIT else text[: TEXT_LIMIT - 1].rstrip() + "…"


def _active_material(
    task: str, bucket_ids: set[str], paths: Mapping[str, Path]
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    mapping = pd.read_parquet(paths["semantic_map"])
    mapping = mapping[mapping.semantic_bucket_id.astype(str).isin(bucket_ids)]
    atom_ids = set(mapping.atom_id.astype(str))
    records = pd.read_parquet(paths["record_rankings"])
    records = records[records.atom_id.astype(str).isin(atom_ids)]
    representatives = records.sort_values("canonical_record_id").drop_duplicates("atom_id")
    identity = ["level", "source_id", "pair_bucket_key"]
    if representatives.groupby("atom_id")[identity].nunique().gt(1).any().any():
        raise ValueError(f"active atom identity changed for {task}")
    atoms = _atom_frame(task, records, representatives)
    cards = _record_cards(task, representatives, paths["pair_bucket_records"])
    return atoms, cards


def _atom_frame(
    task: str, records: pd.DataFrame, representatives: pd.DataFrame
) -> pd.DataFrame:
    counts = records.groupby("atom_id").size().to_dict()
    rows = []
    for row in representatives.itertuples(index=False):
        values = v5._pair_values(task, str(row.source_id), str(row.pair_bucket_key))
        rows.append({
            "atom_id": str(row.atom_id), "source_id": str(row.source_id),
            "record_count": int(counts[row.atom_id]),
            "values_json": core._canonical_json(values),
        })
    return pd.DataFrame(rows)


def _record_cards(
    task: str, representatives: pd.DataFrame, record_path: Path
) -> dict[str, dict[str, Any]]:
    identity_columns = {
        column for columns in v5.TASK_PROMPT_COLUMNS[task].values() for column in columns
    }
    columns = ["source_row_uid", "source_name", *sorted(identity_columns | set(ENRICHED_COLUMNS[task]))]
    source = pd.read_parquet(record_path, columns=columns)
    source = source.drop_duplicates("source_row_uid").set_index("source_row_uid")
    cards = {}
    for row in representatives.itertuples(index=False):
        record = source.loc[row.source_row_uid]
        wanted = ["source_name", *v5.TASK_PROMPT_COLUMNS[task][str(row.source_id)],
                  *ENRICHED_COLUMNS[task]]
        cards[str(row.atom_id)] = {
            column: record[column] for column in dict.fromkeys(wanted)
            if column in record and _usable(record[column])
        }
    return cards


def _bucket_payloads(
    task: str, bucket_ids: set[str], paths: Mapping[str, Path] | None = None,
) -> dict[str, dict[str, Any]]:
    paths = dict(paths or v5._input_paths(task))
    mapping = pd.read_parquet(paths["semantic_map"])
    mapping = mapping[mapping.semantic_bucket_id.astype(str).isin(bucket_ids)]
    atoms, cards = _active_material(task, bucket_ids, paths)
    lookup = core._atom_lookup(atoms)
    payloads = {}
    for bucket, rows in mapping.groupby("semantic_bucket_id", sort=True):
        members = sorted(rows.atom_id.astype(str))
        payloads[str(bucket)] = _bucket_payload(task, str(bucket), members, lookup, cards)
    if set(payloads) != bucket_ids:
        raise ValueError(f"semantic payload coverage changed for {task}")
    return payloads


def _bucket_payload(
    task: str, bucket: str, atoms: list[str], lookup: Mapping[str, Mapping[str, Any]],
    cards: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_source: dict[str, list[str]] = {}
    for atom in atoms:
        by_source.setdefault(str(lookup[atom]["source_id"]), []).append(atom)
    components = []
    for source, members in sorted(by_source.items()):
        dimensions = {
            column: sorted({str(lookup[atom]["values"][column]) for atom in members})
            for column in v5.TASK_PROMPT_COLUMNS[task][source]
        }
        components.append({"source_id": source, "source_label": SOURCE_LABELS[source],
                           "canonical_dimensions": dimensions})
    ordered = sorted(atoms, key=lambda atom: (core._stable_id("ranking-sample", bucket, atom), atom))
    selected = v5._source_balanced_samples(ordered, lookup)
    samples = [{"source_id": str(lookup[atom]["source_id"]), **cards[atom]} for atom in selected]
    return {"identity": {"source_components": components}, "sample_records": samples}


def _card(label: str, payload: Mapping[str, Any]) -> str:
    lines = [label]
    for component in payload["identity"]["source_components"]:
        lines.append(f"  Evidence family: {component['source_label']}")
        lines.append("  Canonical bucket identity:")
        for field, values in component["canonical_dimensions"].items():
            shown = _display(values)
            if shown:
                lines.append(f"    - {field.replace('_', ' ')}: {shown}")
    lines.append("  Representative experimental records:")
    for index, sample in enumerate(payload["sample_records"], 1):
        lines.append(f"    Record {index} ({SOURCE_LABELS[sample['source_id']]}):")
        for field, value in sample.items():
            if field != "source_id" and (shown := _display(value)):
                lines.append(f"      - {field.replace('_', ' ')}: {shown}")
    return "\n".join(lines)


def _candidate_card(
    index: int, bucket: str, payload: Mapping[str, Any], prior: Mapping[str, Any]
) -> str:
    body = _card(f"Candidate {index}", payload).splitlines()
    return "\n".join([body[0], f"  Previous weight: {prior['weight']:.2f}", *body[1:]])


def _anchor_card(
    index: int, bucket: str, payload: Mapping[str, Any], prior: Mapping[str, Any],
    score: Mapping[str, Any],
) -> str:
    body = _card(f"Anchor {index}", payload).splitlines()
    return "\n".join([
        body[0], f"  Previous weight: {prior['weight']:.2f}",
        f"  Locked calibrated weight: {score['weight']}",
        f"  Calibration rationale: {v5._clean(score['rationale'])}", *body[1:],
    ])


def _render(
    task: str, level: str, candidates: Sequence[str], anchors: Sequence[str],
    payloads: Mapping[str, Mapping[str, Any]], priors: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
) -> str:
    candidate_cards = "\n\n".join(
        _candidate_card(index, bucket, payloads[bucket], priors[bucket])
        for index, bucket in enumerate(candidates, 1)
    )
    anchor_cards = "\n\n".join(
        _anchor_card(index, bucket, payloads[bucket], priors[bucket], scores[bucket])
        for index, bucket in enumerate(anchors, 1)
    )
    name = "anchored.jinja" if anchors else "initial.jinja"
    template = Environment(undefined=StrictUndefined, autoescape=False).from_string(
        (PROMPT_ROOT / name).read_text(encoding="utf-8")
    )
    return template.render(target=v5.TASK_TARGETS[task], level=level,
                           candidate_cards=candidate_cards, anchor_cards=anchor_cards).strip()


def _prior_lookup(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return frame.set_index("semantic_bucket_id")[["weight", "rationale"]].to_dict("index")


def _write_review(
    root: Path, schedule: pd.DataFrame, priors: Mapping[str, Mapping[str, Any]],
    task_paths: Mapping[str, Mapping[str, Path]],
) -> dict[str, Any]:
    review = root / "prompt_review"
    review.mkdir(exist_ok=False)
    files = []
    for task in sorted(schedule.task.unique()):
        rows = schedule[schedule.task.eq(task)].sort_values(["level", "round", "batch"])
        examples = rows.groupby("level", sort=True).head(2)
        wanted = {bucket for row in examples.itertuples(index=False)
                  for field in (row.candidate_bucket_ids_json, row.anchor_bucket_ids_json)
                  for bucket in json.loads(field)}
        payloads = _bucket_payloads(task, wanted, task_paths[task])
        for row in examples.itertuples(index=False):
            candidates, anchors = json.loads(row.candidate_bucket_ids_json), json.loads(row.anchor_bucket_ids_json)
            scores = {bucket: {"weight": "<preceding calibrated weight>",
                               "rationale": "<preceding calibration rationale>"}
                      for bucket in anchors}
            prompt = _render(task, row.level, candidates, anchors, payloads, priors, scores)
            path = review / f"{task}_{row.level}_r{row.round:04d}_b{row.batch:02d}.txt"
            path.write_text(prompt + "\n", encoding="utf-8")
            files.append({"path": path.name, "sha256": v5._sha256(path)})
    manifest = {"version": f"{VERSION}.prompt_review", "completion_requests_made": 0,
                "status": "awaiting_user_prompt_approval", "files": files}
    write_json_atomic(review / "manifest.json", manifest)
    return manifest


def prepare(
    run_id: str, v5_run_id: str,
    task_levels: Sequence[tuple[str, str]] = (),
    selection_fraction: float = 0.5,
) -> dict[str, Any]:
    root = _run_root(run_id)
    if root.exists():
        raise FileExistsError(root)
    old, old_manifest = _load_v5(v5_run_id, task_levels)
    old_root = _v5_root(v5_run_id)
    resolved = v5._resolved_prepared_inputs(old_root, old_manifest)
    tasks = sorted(old.task.unique())
    task_paths = {
        task: {key.split("/", 1)[1]: path for key, path in resolved.items()
               if key.startswith(f"{task}/")}
        for task in tasks
    }
    root.mkdir(parents=True)
    if task_levels:
        weights_path = root / "v5_weights.snapshot.parquet"
        manifest_path = root / "v5_manifest.snapshot.json"
        v5._atomic_parquet(old, weights_path)
        shutil.copyfile(old_root / "manifest.json", manifest_path)
    else:
        weights_path = old_root / "weights.parquet"
        manifest_path = old_root / "manifest.json"
    schedule = build_schedule(old, selection_fraction)
    v5._atomic_parquet(schedule, root / "schedule.parquet")
    v5._atomic_tsv(schedule, root / "schedule.tsv")
    review = _write_review(root, schedule, _prior_lookup(old), task_paths)
    inputs = {
        "v5_manifest": {"path": str(manifest_path), "sha256": v5._sha256(manifest_path)},
        "v5_weights": {"path": str(weights_path), "sha256": v5._sha256(weights_path)},
    }
    inputs.update({f"{task}/{name}": {"path": str(path), "sha256": v5._sha256(path)}
                   for task in tasks for name, path in task_paths[task].items()})
    manifest = _prepared_manifest(
        run_id, v5_run_id, schedule, review, inputs, len(old),
        old_manifest["status"], task_levels, selection_fraction,
    )
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def _prepared_manifest(
    run_id: str, v5_run_id: str, schedule: pd.DataFrame,
    review: Mapping[str, Any], inputs: Mapping[str, Any], bucket_count: int,
    v5_status: str, task_levels: Sequence[tuple[str, str]], selection_fraction: float,
) -> dict[str, Any]:
    direct = v5._scheduled_bucket_count(schedule)
    templates = {path.name: v5._sha256(path) for path in sorted(PROMPT_ROOT.glob("*.jinja"))}
    return {
        "version": VERSION, "status": "awaiting_prompt_approval", "run_id": run_id,
        "v5_run_id": v5_run_id, "reasoning_effort": "high", "max_tokens": MAX_TOKENS,
        "storage": {"root": str(_run_root(run_id)), "filesystem": "vast_nfs"},
        "selection": ("per task-level: min(N, max(32, ceil(N*fraction))) "
                      "sorted by V5 weight desc"),
        "selection_fraction": selection_fraction,
        "chain_width": CHAIN_WIDTH, "candidates_per_chain": 7,
        "scope": "task_level_shard" if task_levels else "full",
        "task_levels": [f"{task}/{level}" for task, level in task_levels],
        "source_v5_status": v5_status,
        "bucket_count": bucket_count, "direct_v6_bucket_count": direct,
        "carried_forward_v5_bucket_count": bucket_count - direct,
        "request_count": len(schedule), "completion_requests_made": 0,
        "schedule_sha256": v5._sha256(_run_root(run_id) / "schedule.parquet"),
        "prompt_review_manifest_sha256": v5._sha256(_run_root(run_id) / "prompt_review/manifest.json"),
        "models": {"seed": SEED_MODEL, "anchored_default": FOLLOWUP_MODEL,
                   "anchored_profile": "mixed",
                   "anchored_allowed": [provider_pool.DEFAULT_MODEL, provider_pool.MIXED_MODEL]},
        "request_config": {
            "reasoning_effort": "high", "thinking": "enabled",
            "timeout_seconds": 3_600, "maximum_attempts": 2,
            "openrouter_routing": {"profile": "mixed_gold", "allow_fallbacks": False,
                                   "requests_per_route": provider_pool.REQUESTS_PER_ROUTE},
        },
        "representation": {task: list(columns) for task, columns in ENRICHED_COLUMNS.items()},
        "inputs": dict(inputs), "templates": templates, "review": dict(review),
    }


def _verify(root: Path, approved_hash: str) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION:
        raise ValueError("prepared run has the wrong version")
    if v5._sha256(root / "prompt_review/manifest.json") != approved_hash:
        raise ValueError("approved prompt-review hash does not match")
    if v5._sha256(root / "schedule.parquet") != manifest["schedule_sha256"]:
        raise ValueError("prepared schedule changed")
    for item in manifest["inputs"].values():
        if v5._sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"prepared input changed: {item['path']}")
    for name, digest in manifest["templates"].items():
        if v5._sha256(PROMPT_ROOT / name) != digest:
            raise ValueError(f"prompt template changed: {name}")
    return manifest, pd.read_parquet(manifest["inputs"]["v5_weights"]["path"])


def _configure(round_index: int) -> str:
    model, limit = (SEED_MODEL, 9) if round_index == 0 else (FOLLOWUP_MODEL, 27)
    core.MODEL, core.BASE_URL = model, OPENROUTER_URL
    core.ENDPOINTS = ({"name": f"openrouter_{model.rsplit('/', 1)[-1]}",
                       "base_url": OPENROUTER_URL, "provider": "openrouter",
                       "credential_env": "OPEN_ROUTER_KEY", "max_inflight": limit},)
    core.WEIGHT_REQUEST_EXTRA_BODY = {"thinking": {"type": "enabled"}}
    if round_index == 0:
        core.WEIGHT_REQUEST_EXTRA_BODY["provider"] = {
            "sort": "price", "allow_fallbacks": True, "require_parameters": True,
        }
    core.REQUEST_TIMEOUT_S, core.MAX_ATTEMPTS = 3_600, 2
    return "seed" if round_index == 0 else "anchored"


def _completed(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    scores = {}
    query = ("SELECT request_id,validation_json,response_json,served_model,provider_name,"
             "provider_base_url,attempts,input_tokens,output_tokens FROM requests "
             "WHERE kind='weight_assignment' AND status='complete'")
    for row in connection.execute(query):
        validation, response = json.loads(row["validation_json"]), json.loads(row["response_json"])
        aliases = dict(zip(validation["candidate_aliases"], validation["candidate_bucket_ids"], strict=True))
        for score in response["scores"]:
            bucket = aliases[score["candidate"]]
            if bucket in scores:
                raise ValueError(f"bucket completed twice: {bucket}")
            scores[bucket] = {"weight": float(score["weight"]), "rationale": score["rationale"],
                              "request_id": row["request_id"], "served_model": row["served_model"],
                              "requested_model": validation.get("requested_model"),
                              "selected_provider": validation.get("selected_provider_route"),
                              "provider_pool_snapshot_sha256": validation.get(
                                  "provider_pool_snapshot_sha256"),
                              "provider_name": row["provider_name"], "provider_base_url": row["provider_base_url"],
                              "attempts": row["attempts"], "input_tokens": row["input_tokens"],
                              "output_tokens": row["output_tokens"],
                              "anchor_bucket_ids": validation["anchor_bucket_ids"]}
    return scores


def _queue(
    connection: sqlite3.Connection, rows: pd.DataFrame,
    payloads: Mapping[str, Mapping[str, Any]], priors: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
    provider_snapshot: Mapping[str, Any] | None = None,
    local_speculative: bool = False,
    local_base_url: str = speculative.DEFAULT_BASE_URL,
) -> list[str]:
    request_ids = []
    for row in rows.itertuples(index=True):
        candidates, anchors = json.loads(row.candidate_bucket_ids_json), json.loads(row.anchor_bucket_ids_json)
        if missing := set(anchors) - scores.keys():
            raise ValueError(f"{row.batch_id} has incomplete anchors: {sorted(missing)}")
        prompt = _render(row.task, row.level, candidates, anchors, payloads, priors, scores)
        request_id = core._request_id("weight_assignment", row.batch_id, prompt)
        existing = connection.execute(
            "SELECT status,validation_json FROM requests WHERE request_id=?", (request_id,)
        ).fetchone()
        if existing:
            if local_speculative and existing["status"] != "complete":
                validation = json.loads(existing["validation_json"])
                route = urlsplit(local_base_url).netloc.replace(":", "_")
                validation["selected_provider_route"] = f"{route}_speculative_first4"
                connection.execute(
                    "UPDATE requests SET validation_json=? WHERE request_id=?",
                    (core._canonical_json(validation), request_id),
                )
            request_ids.append(request_id)
            continue
        aliases = [f"Candidate {index}" for index in range(1, len(candidates) + 1)]
        validation = {"candidate_aliases": aliases, "candidate_bucket_ids": candidates,
                      "anchor_bucket_ids": anchors}
        if local_speculative:
            route = urlsplit(local_base_url).netloc.replace(":", "_")
            validation.update(
                requested_model=speculative.DEFAULT_MODEL,
                allowed_served_models=[speculative.DEFAULT_MODEL],
                selected_provider_route=f"{route}_speculative_first4",
            )
        elif row.round > 0:
            if provider_snapshot is None:
                raise ValueError("anchored requests require a provider-pool snapshot")
            validation.update(provider_pool.request_assignment(
                provider_snapshot, int(row.Index), seed_request_count=9
            ))
        core._queue_request(connection, request_id=request_id, kind="weight_assignment",
                            phase=row.batch_id, prompt=prompt, reasoning_effort="high",
                            max_tokens=MAX_TOKENS, validation=validation, commit=False)
        request_ids.append(request_id)
    connection.commit()
    return request_ids


def _has_pending(connection: sqlite3.Connection, request_ids: Sequence[str]) -> bool:
    placeholders = ",".join("?" for _ in request_ids)
    count = connection.execute(
        f"SELECT count(*) FROM requests WHERE request_id IN ({placeholders}) AND status != 'complete'",
        list(request_ids),
    ).fetchone()[0]
    return bool(count)


def _score_frame(
    scores: Mapping[str, Mapping[str, Any]], schedule: pd.DataFrame
) -> pd.DataFrame:
    metadata = {}
    for row in schedule.itertuples(index=False):
        candidates = json.loads(row.candidate_bucket_ids_json)
        calibration = json.loads(row.candidate_calibration_ranks_json)
        original = json.loads(row.candidate_original_level_ranks_json)
        for position, (bucket, cal_rank, old_rank) in enumerate(zip(candidates, calibration, original, strict=True), 1):
            metadata[bucket] = {"task": row.task, "level": row.level, "round": row.round,
                                "batch": row.batch, "position_in_batch": position,
                                "calibration_rank": cal_rank, "original_level_rank": old_rank}
    rows = [{"semantic_bucket_id": bucket, **metadata[bucket], **score,
             "anchor_bucket_ids_json": core._canonical_json(score["anchor_bucket_ids"])}
            for bucket, score in scores.items()]
    return pd.DataFrame(rows).drop(columns="anchor_bucket_ids", errors="ignore")


def _checkpoint(
    root: Path, manifest: dict[str, Any], schedule: pd.DataFrame,
    scores: Mapping[str, Mapping[str, Any]], round_index: int,
) -> None:
    frame = _score_frame(scores, schedule)
    v5._atomic_parquet(frame, root / "weights.partial.parquet")
    v5._atomic_tsv(frame, root / "weights.partial.tsv")
    manifest.update(status="running", completed_through_round=round_index,
                    completed_bucket_count=len(frame),
                    completion_requests_made=len({row["request_id"] for row in scores.values()}))
    write_json_atomic(root / "manifest.json", manifest)


def _execute_openrouter_round(
    connection: sqlite3.Connection, requests: Sequence[str], round_index: int,
    snapshot: Mapping[str, Any] | None,
    pools: dict[str, tuple[Any, list[dict[str, Any]], int]],
    manifest: dict[str, Any], root: Path,
) -> None:
    stage = _configure(round_index)
    if snapshot is not None:
        v5._record_provider_snapshot(manifest, snapshot)
    if stage not in pools:
        pools[stage] = core._build_completion_pool(9 if stage == "seed" else 27)
        manifest.setdefault("endpoint_pools_at_start", {})[stage] = pools[stage][1]
        write_json_atomic(root / "manifest.json", manifest)
    client, _, parallelism = pools[stage]
    core._run_pending(
        connection, requests, parallelism=parallelism, client=client,
        request_slots=threading.BoundedSemaphore(parallelism),
    )


def _all_payloads(
    schedule: pd.DataFrame, manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    scheduled = {task: {bucket for value in schedule[schedule.task.eq(task)].candidate_bucket_ids_json
                        for bucket in json.loads(value)}
                 for task in sorted(schedule.task.unique())}
    task_paths = {
        task: {key.split("/", 1)[1]: Path(item["path"])
               for key, item in manifest["inputs"].items()
               if key.startswith(f"{task}/")}
        for task in scheduled
    }
    return {bucket: payload for task, ids in scheduled.items()
            for bucket, payload in _bucket_payloads(
                task, ids, task_paths[task]
            ).items()}


def run(
    run_id: str, approved_hash: str, *, local_speculative: bool = False,
    benchmark_path: Path | None = None, required_replicas: int = 4, fanout: int = 16,
    speculative_base_url: str = speculative.DEFAULT_BASE_URL,
) -> dict[str, Any]:
    root = _run_root(run_id)
    manifest, old = _verify(root, approved_hash)
    schedule = pd.read_parquet(root / "schedule.parquet")
    payloads = _all_payloads(schedule, manifest)
    priors = _prior_lookup(old)
    connection = core._request_database(root / "requests.sqlite3", journal_mode="DELETE")
    pools = {}
    if local_speculative and benchmark_path is None:
        raise ValueError("local speculative execution requires its benchmark")
    if local_speculative:
        previous_endpoint = manifest.get("execution_override", {}).get("endpoint")
        if previous_endpoint and previous_endpoint != speculative_base_url:
            manifest.setdefault("execution_migrations", []).append({
                "from_endpoint": previous_endpoint, "to_endpoint": speculative_base_url,
                "reason": "resume after endpoint outage",
            })
        manifest["execution_override"] = {
            "endpoint": speculative_base_url, "model": speculative.DEFAULT_MODEL,
            "reasoning_effort": "high", "fanout": fanout, "required_replicas": required_replicas,
            "openrouter_responses_reused": False,
        }
    manifest["status"] = "running"
    write_json_atomic(root / "manifest.json", manifest)
    try:
        for round_index in sorted(schedule["round"].unique()):
            scores = _completed(connection)
            snapshot = (provider_pool.load_ranked_pool(True)
                        if int(round_index) > 0 and not local_speculative else None)
            requests = _queue(connection, schedule[schedule["round"].eq(round_index)],
                              payloads, priors, scores, snapshot, local_speculative,
                              speculative_base_url)
            if _has_pending(connection, requests):
                if local_speculative:
                    receipt = asyncio.run(speculative.execute_pending(
                        connection, requests, benchmark_path, required=required_replicas,
                        fanout=fanout, maximum_requests=12,
                        require_endpoint_drain=False,
                        base_url=speculative_base_url,
                    ))
                    manifest.setdefault("speculative_round_receipts", []).append(
                        {"round": int(round_index), **receipt})
                else:
                    _execute_openrouter_round(
                        connection, requests, int(round_index), snapshot, pools, manifest, root)
            _checkpoint(root, manifest, schedule, _completed(connection), int(round_index))
    except Exception:
        manifest["status"] = "incomplete"
        write_json_atomic(root / "manifest.json", manifest)
        raise
    finally:
        connection.close()
    manifest["status"] = "complete_unpublished"
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def _final_frame(old: pd.DataFrame, direct: pd.DataFrame) -> pd.DataFrame:
    old = old.rename(columns={column: f"v5_{column}" for column in old.columns
                              if column not in {"semantic_bucket_id", "task", "level"}})
    direct = direct.rename(columns={column: f"v6_{column}" for column in direct.columns
                                    if column not in {"semantic_bucket_id", "task", "level"}})
    frame = old.merge(direct, on=["semantic_bucket_id", "task", "level"], how="left", validate="one_to_one")
    frame["calibration_status"] = frame.v6_weight.notna().map({True: "direct_v6", False: "carried_forward_v5"})
    frame["final_weight"] = frame.v6_weight.fillna(frame.v5_weight)
    frame["weight_delta"] = frame.final_weight - frame.v5_weight
    frame["original_level_rank"] = frame.v5_level_rank.astype(int)
    return frame.sort_values(["task", "level", "original_level_rank"])


def _band(weight: float) -> str:
    if weight == 0:
        return "unusable"
    boundaries = ((.2, "negligible"), (.4, "weak"), (.6, "moderate"),
                  (.8, "strong"), (.95, "very_strong"), (1.01, "exceptional"))
    return next(label for upper, label in boundaries if weight < upper)


def _manual_audit(frame: pd.DataFrame) -> pd.DataFrame:
    policy = json.loads((ROOT / "policies/semantic_weighted_top10_v2.json").read_text())
    rows = []
    for task, task_data in policy["tasks"].items():
        for level, data in task_data["levels"].items():
            rows.extend({"task": task, "level": level, "semantic_bucket_id": bucket,
                         "manual_weight": weight, "manual_rationale": rationale}
                        for bucket, weight, rationale in zip(data["bucket_ids"], data["weights"],
                                                             data["rationales"], strict=True))
    manual = pd.DataFrame(rows).merge(frame[["semantic_bucket_id", "v5_weight", "final_weight"]],
                                      on="semantic_bucket_id", how="left", validate="one_to_one")
    manual["v5_absolute_error"] = (manual.v5_weight - manual.manual_weight).abs()
    manual["v6_absolute_error"] = (manual.final_weight - manual.manual_weight).abs()
    return manual


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    return float(left.rank(method="average").corr(right.rank(method="average")))


def _audit_summary(direct: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, level), group in direct.groupby(["task", "level"], sort=True):
        delta = group.weight_delta
        rows.append({
            "task": task, "level": level, "calibrated_buckets": len(group),
            "mean_absolute_change": delta.abs().mean(),
            "median_absolute_change": delta.abs().median(),
            "unchanged_fraction": delta.eq(0).mean(),
            "raised_fraction": delta.gt(0).mean(), "lowered_fraction": delta.lt(0).mean(),
            "v5_v6_spearman": _rank_correlation(group.v5_weight, group.final_weight),
        })
    return pd.DataFrame(rows)


def _manual_summary(manual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, level), group in manual.groupby(["task", "level"], sort=True):
        rows.append({
            "task": task, "level": level, "manual_buckets": len(group),
            "v5_manual_mae": group.v5_absolute_error.mean(),
            "v6_manual_mae": group.v6_absolute_error.mean(),
            "v5_manual_spearman": _rank_correlation(group.v5_weight, group.manual_weight),
            "v6_manual_spearman": _rank_correlation(group.final_weight, group.manual_weight),
        })
    return pd.DataFrame(rows)


def _write_audits(root: Path, frame: pd.DataFrame) -> None:
    changes = frame[["semantic_bucket_id", "task", "level", "original_level_rank",
                     "calibration_status", "v5_weight", "v6_weight", "final_weight", "weight_delta"]].copy()
    changes["v5_band"] = changes.v5_weight.map(_band)
    changes["final_band"] = changes.final_weight.map(_band)
    changes["band_transition"] = changes.v5_band + " -> " + changes.final_band
    v5._atomic_tsv(changes, root / "weight_changes.tsv")
    direct = changes[changes.calibration_status.eq("direct_v6")].copy()
    direct["absolute_change"] = direct.weight_delta.abs()
    v5._atomic_tsv(_audit_summary(direct), root / "audit_summary.tsv")
    largest = direct.sort_values(["task", "level", "absolute_change"],
                                 ascending=[True, True, False]).groupby(["task", "level"]).head(20)
    v5._atomic_tsv(largest, root / "largest_revisions.tsv")
    manual = _manual_audit(frame)
    v5._atomic_tsv(manual, root / "manual_weight_agreement.tsv")
    v5._atomic_tsv(_manual_summary(manual), root / "manual_weight_agreement.summary.tsv")


def publish_candidate(run_id: str) -> dict[str, Any]:
    root = _run_root(run_id)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete_unpublished":
        raise ValueError("V6 inference is not complete_unpublished")
    schedule = pd.read_parquet(root / "schedule.parquet")
    connection = core._request_database(root / "requests.sqlite3", journal_mode="DELETE")
    try:
        direct = _score_frame(_completed(connection), schedule)
    finally:
        connection.close()
    if len(direct) != manifest["direct_v6_bucket_count"] or direct.semantic_bucket_id.duplicated().any():
        raise ValueError("direct V6 calibration coverage is incomplete")
    old = pd.read_parquet(manifest["inputs"]["v5_weights"]["path"])
    frame = _final_frame(old, direct)
    if len(frame) != manifest["bucket_count"] or frame.final_weight.isna().any():
        raise ValueError("final V6 candidate does not cover the V5 universe")
    v5._atomic_parquet(frame, root / "weights.parquet")
    v5._atomic_tsv(frame, root / "weights.tsv")
    _write_audits(root, frame)
    report = (f"# V6 semantic-weight calibration candidate\n\n"
              f"Directly recalibrated {len(direct):,} of {len(frame):,} buckets; "
              f"the remaining {len(frame) - len(direct):,} retain their immutable V5 weights. "
              "This candidate does not modify the active retrieval policy.\n")
    (root / "report.md").write_text(report, encoding="utf-8")
    names = ("weights.parquet", "weights.tsv", "weight_changes.tsv", "audit_summary.tsv",
             "largest_revisions.tsv", "manual_weight_agreement.tsv",
             "manual_weight_agreement.summary.tsv", "report.md")
    manifest.update(status="complete_candidate", completed_bucket_count=len(frame),
                    active_policy_changed=False,
                    artifacts={name: v5._sha256(root / name) for name in names})
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def _parse_task_levels(values: Sequence[str]) -> list[tuple[str, str]]:
    parsed = []
    for value in values:
        task, separator, level = value.partition(":")
        if not separator or task not in v5.TASK_TARGETS or not re.fullmatch(r"L\d+", level):
            raise ValueError(f"invalid task-level {value!r}; expected task:L#")
        parsed.append((task, level))
    if len(parsed) != len(set(parsed)):
        raise ValueError("task-level arguments must be unique")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "publish-candidate"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--v5-run-id")
    parser.add_argument("--approved-review-sha256")
    parser.add_argument("--task-level", action="append", default=[])
    parser.add_argument("--selection-fraction", type=float, default=0.5)
    parser.add_argument("--local-speculative", action="store_true")
    parser.add_argument("--speculative-benchmark-manifest", type=Path)
    parser.add_argument("--speculative-required-replicas", type=int, default=4)
    parser.add_argument("--speculative-fanout", type=int, default=16)
    parser.add_argument("--speculative-base-url", default=speculative.DEFAULT_BASE_URL)
    args = parser.parse_args()
    if args.command == "prepare":
        if not args.v5_run_id:
            parser.error("prepare requires --v5-run-id")
        result = prepare(
            args.run_id, args.v5_run_id, _parse_task_levels(args.task_level),
            args.selection_fraction,
        )
    elif args.command == "run":
        if not args.approved_review_sha256:
            parser.error("run requires --approved-review-sha256")
        result = run(
            args.run_id, args.approved_review_sha256,
            local_speculative=args.local_speculative,
            benchmark_path=args.speculative_benchmark_manifest,
            required_replicas=args.speculative_required_replicas,
            fanout=args.speculative_fanout,
            speculative_base_url=args.speculative_base_url,
        )
    else:
        result = publish_candidate(args.run_id)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
