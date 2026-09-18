"""Freeze V3 source-local semantics, build readouts, and rank semantic buckets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

import httpx
from jinja2 import Environment, StrictUndefined
import pandas as pd

from semantic_buckets import bioavailability_semantic_degree25 as degree25
from semantic_buckets import bioavailability_semantic_readout_v1 as core
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "source_local_semantic_v3.v1"
READOUT_VERSION = "source_local_readout.v3"
ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = Path(__file__).with_name("prompts") / "source_local_readout_v1"
READOUT_MAXIMUM_DEPTH = 2
TASK_ALIASES = {
    "bbb": "bbb_martins",
    "bbb_martins": "bbb_martins",
    "oral": "bioavailability_ma",
    "bioavailability_ma": "bioavailability_ma",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def configure(task: str) -> dict[str, Any]:
    """Configure the existing shared engine for one accepted source-local V3 map."""

    task = TASK_ALIASES.get(task, task)
    if task == "bioavailability_ma":
        from semantic_buckets import bioavailability_semantic_readout_v3 as v3
        from analysis.evidence_library import (
            bioavailability_semantic_readout_v3_l4_refinement as repaired,
        )

        v3.configure_core()
        input_root = v3.DEFAULT_OUTPUT
        artifact_root = repaired.DEFAULT_OUTPUT
        source_manifest = artifact_root / "manifest.json"
        ranking_template = (
            Path(__file__).with_name("prompts")
            / "bioavailability_relevance_bucket_level_v2.jinja"
        )
        expected_source_status = "complete"
        expected_source_version = repaired.VERSION
        ranking_version = "bioavailability_semantic_degree25.v3"
    elif task == "bbb_martins":
        from semantic_buckets import bbb_semantic_readout_v1 as bbb
        from semantic_buckets import bbb_semantic_readout_v3 as v3

        v3.configure_workflow()
        bbb.configure_core()
        input_root = v3.DEFAULT_OUTPUT
        artifact_root = v3.DEFAULT_OUTPUT
        source_manifest = artifact_root / "manifest.json"
        ranking_template = bbb.RANKING_TEMPLATE
        expected_source_status = "awaiting_cross_source_mapping"
        expected_source_version = v3.VERSION
        ranking_version = "bbb_semantic_degree25.v3"
    else:
        raise ValueError(f"unsupported task: {task}")

    return {
        "task": task,
        "input_root": input_root,
        "artifact_root": artifact_root,
        "source_map": artifact_root / "source_semantic_bucket_map.parquet",
        "source_manifest": source_manifest,
        "expected_source_status": expected_source_status,
        "expected_source_version": expected_source_version,
        "ranking_template": ranking_template,
        "ranking_version": ranking_version,
        "readout_root": artifact_root / "readout_buckets_v3",
        "readout_seed_cache": artifact_root / "readout_buckets_v2" / "requests.sqlite3",
        "ranking_root": artifact_root / "degree25_rankings",
    }


def _source_map_hash(manifest: Mapping[str, Any], task: str) -> str:
    if task == "bioavailability_ma":
        return str(manifest["output"]["source_map"]["sha256"])
    return str(manifest["source_semantic_bucket_map"]["sha256"])


def _load_inputs(spec: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = json.loads(Path(spec["source_manifest"]).read_text(encoding="utf-8"))
    if manifest.get("version") != spec["expected_source_version"]:
        raise ValueError("source semantic version changed")
    if manifest.get("status") != spec["expected_source_status"]:
        raise ValueError("source semantic artifact is not complete")
    source_path = Path(spec["source_map"])
    if _source_map_hash(manifest, str(spec["task"])) != _sha256(source_path):
        raise ValueError("source semantic map hash changed")

    atoms = pd.read_parquet(Path(spec["input_root"]) / "input_atoms.parquet")
    source = pd.read_parquet(source_path)
    if source["atom_id"].duplicated().any() or atoms["atom_id"].duplicated().any():
        raise ValueError("semantic inputs repeat atom IDs")
    if set(source["atom_id"]) != set(atoms["atom_id"]):
        raise ValueError("source semantic map does not cover every atom")
    if set(source["level"]) != set(core.LEVELS):
        raise ValueError("source semantic levels changed")
    grouped = source.groupby("source_semantic_bucket_id", sort=False).agg(
        levels=("level", "nunique"), sources=("source_id", "nunique")
    )
    if grouped[["levels", "sources"]].ne(1).any().any():
        raise ValueError("a source-local semantic bucket crosses a level or source")
    return atoms, source


def _record_atoms(records: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [
            core._stable_id("atom", level, source, pair_bucket)
            for level, source, pair_bucket in records[
                ["level", "source_id", "pair_bucket_key"]
            ].itertuples(index=False, name=None)
        ],
        index=records.index,
    )


def freeze_semantics(task: str) -> dict[str, Any]:
    spec = configure(task)
    atoms, source = _load_inputs(spec)
    root = Path(spec["artifact_root"])
    final_path = root / "semantic_bucket_map.parquet"
    manifest_path = root / "semantic_bucket_map_manifest.json"

    final = source.copy()
    final["semantic_bucket_id"] = final["source_semantic_bucket_id"]
    final = final[
        ["level", "source_id", "source_semantic_bucket_id", "semantic_bucket_id", "atom_id"]
    ].sort_values(["level", "source_id", "semantic_bucket_id", "atom_id"])
    if final_path.exists():
        existing = pd.read_parquet(final_path)
        if not existing.equals(final):
            raise ValueError("existing final semantic map differs from the source-local freeze")
    else:
        final.to_parquet(final_path, index=False)

    joined = final.merge(
        atoms[["atom_id", "record_count"]], on="atom_id", how="left", validate="one_to_one"
    )
    audit = (
        joined.groupby(["level", "source_id", "semantic_bucket_id"], sort=True)
        .agg(pair_bucket_count=("atom_id", "size"), record_count=("record_count", "sum"))
        .reset_index()
    )
    audit["level_pair_bucket_fraction"] = audit["pair_bucket_count"] / audit.groupby(
        "level"
    )["pair_bucket_count"].transform("sum")
    audit["level_record_fraction"] = audit["record_count"] / audit.groupby("level")[
        "record_count"
    ].transform("sum")
    audit_path = root / "semantic_bucket_size_audit.parquet"
    audit.to_parquet(audit_path, index=False)

    records = pd.read_parquet(core.RECORD_MAP)
    scoped = records[records["level"].isin(core.LEVELS)].copy()
    scoped["atom_id"] = _record_atoms(scoped)
    if len(scoped) != core.EXPECTED_RECORD_COUNT or set(scoped["atom_id"]) != set(final["atom_id"]):
        raise ValueError("frozen semantic map does not cover the scoped V10 records")

    ignored = sorted(
        set().union(*(set(core.PAIR_COLUMNS[source]) - set(columns) for source, columns in core.REFINEMENT_COLUMNS.items()))
    )
    manifest = {
        "version": f"{VERSION}.semantic_freeze",
        "status": "complete_reviewed",
        "task": spec["task"],
        "release": "v10",
        "levels": list(core.LEVELS),
        "cross_source_merging": False,
        "semantic_bucket_id_policy": "source_semantic_bucket_id",
        "source_manifest": str(spec["source_manifest"]),
        "source_manifest_sha256": _sha256(Path(spec["source_manifest"])),
        "source_semantic_bucket_map_sha256": _sha256(Path(spec["source_map"])),
        "input_atoms": {
            "path": str(Path(spec["input_root"]) / "input_atoms.parquet"),
            "sha256": _sha256(Path(spec["input_root"]) / "input_atoms.parquet"),
            "rows": len(atoms),
        },
        "sample_cards": {
            "path": str(Path(spec["input_root"]) / "sample_cards.json.gz"),
            "sha256": _sha256(Path(spec["input_root"]) / "sample_cards.json.gz"),
        },
        "semantic_bucket_map_sha256": _sha256(final_path),
        "semantic_bucket_count": int(final["semantic_bucket_id"].nunique()),
        "atom_count": len(final),
        "record_count": len(scoped),
        "ignored_pair_bucket_columns": ignored,
        "size_audit": {
            "path": audit_path.name,
            "sha256": _sha256(audit_path),
            "maximum_level_pair_bucket_fraction": float(audit["level_pair_bucket_fraction"].max()),
            "maximum_level_record_fraction": float(audit["level_record_fraction"].max()),
            "large_buckets_are_retained_as_reviewed_relevance_categories": True,
        },
        "ranking_prompt_sha256": _sha256(Path(spec["ranking_template"])),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def _render(name: str, payload: Mapping[str, Any]) -> str:
    template = Environment(undefined=StrictUndefined, autoescape=False).from_string(
        (PROMPT_ROOT / f"{name}.jinja").read_text(encoding="utf-8")
    )
    return template.render(compact_payload_json=_canonical_json(payload)).strip()


def _varying_columns(
    source: str,
    atom_ids: Sequence[str],
    lookup: Mapping[str, Mapping[str, Any]],
    consumed: Iterable[str],
) -> list[str]:
    used = set(consumed)
    return [
        column
        for column in core.REFINEMENT_COLUMNS[source]
        if column not in used
        and len({str(lookup[atom]["values"][column]) for atom in atom_ids}) > 1
    ]


def _coherence_prompt(
    branch: Mapping[str, Any], lookup: Mapping[str, Mapping[str, Any]], cards: Mapping[str, Any]
) -> tuple[str, list[str]]:
    consumed = (
        []
        if int(branch["depth"]) == READOUT_MAXIMUM_DEPTH
        else branch["consumed_columns"]
    )
    candidates = _varying_columns(
        str(branch["source_id"]), branch["atom_ids"], lookup, consumed
    )
    payload = core._bucket_payload(
        str(branch["branch_id"]), branch["atom_ids"], lookup, cards
    )
    at_depth_cap = int(branch["depth"]) == READOUT_MAXIMUM_DEPTH
    payload.update(
        {
            "task": core.TASK_NAME,
            "level": branch["level"],
            "source_id": branch["source_id"],
            "semantic_bucket_id": branch["semantic_bucket_id"],
            "readout_depth": branch["depth"],
            "maximum_readout_depth": READOUT_MAXIMUM_DEPTH,
            "consumed_readout_columns": list(branch["consumed_columns"]),
            "candidate_columns": [
                {
                    "column": column,
                    "description": core.COLUMN_DESCRIPTIONS[column],
                    "distinct_values": len(
                        {
                            str(lookup[atom]["values"][column])
                            for atom in branch["atom_ids"]
                        }
                    ),
                }
                for column in candidates
            ],
        }
    )
    if at_depth_cap:
        payload["at_depth_cap"] = True
    template = "readout_depth_cap_coherence" if at_depth_cap else "readout_coherence"
    return _render(template, payload), candidates


def _merge_prompt(
    branch: Mapping[str, Any],
    column: str,
    lookup: Mapping[str, Mapping[str, Any]],
    cards: Mapping[str, Any],
) -> tuple[str, dict[str, list[str]]]:
    payload, items = core._value_payload(
        str(branch["branch_id"]),
        branch["atom_ids"],
        column,
        lookup,
        context_columns=[*branch["consumed_columns"], column],
    )
    payload.update(
        {
            "task": core.TASK_NAME,
            "level": branch["level"],
            "source_id": branch["source_id"],
            "semantic_bucket_id": branch["semantic_bucket_id"],
            "readout_depth": branch["depth"],
            "sample_records": core._bucket_payload(
                str(branch["branch_id"]), branch["atom_ids"], lookup, cards
            )["sample_records"],
        }
    )
    return _render("readout_merge_values", payload), items


def _profile_prompt(
    branch: Mapping[str, Any],
    primary_column: str,
    lookup: Mapping[str, Mapping[str, Any]],
    cards: Mapping[str, Any],
) -> tuple[str, dict[str, list[str]], list[str]]:
    varying = _varying_columns(
        str(branch["source_id"]), branch["atom_ids"], lookup, []
    )
    columns = [primary_column, *[column for column in varying if column != primary_column]]
    by_profile: dict[str, list[str]] = {}
    for atom in branch["atom_ids"]:
        profile = _canonical_json(
            {column: str(lookup[atom]["values"][column]) for column in columns}
        )
        by_profile.setdefault(profile, []).append(atom)

    items = {}
    profiles = []
    for index, (profile, members) in enumerate(sorted(by_profile.items())):
        profile_id = f"p{index:06d}"
        items[profile_id] = sorted(members)
        profiles.append(
            {
                "profile_id": profile_id,
                "canonical_dimensions": json.loads(profile),
                "pair_bucket_count": len(members),
                "record_count": sum(
                    int(lookup[atom]["record_count"]) for atom in members
                ),
            }
        )
    payload = {
        "task": core.TASK_NAME,
        "level": branch["level"],
        "source_id": branch["source_id"],
        "semantic_bucket_id": branch["semantic_bucket_id"],
        "bucket_id": branch["branch_id"],
        "readout_depth": branch["depth"],
        "primary_selected_column": primary_column,
        "profile_columns": columns,
        "profiles": profiles,
        "sample_records": core._bucket_payload(
            str(branch["branch_id"]), branch["atom_ids"], lookup, cards
        )["sample_records"],
    }
    return _render("readout_merge_profiles", payload), items, columns


def _exact_profile_groups(
    branch: Mapping[str, Any], lookup: Mapping[str, Mapping[str, Any]]
) -> list[list[str]]:
    columns = core.REFINEMENT_COLUMNS[str(branch["source_id"])]
    groups: dict[str, list[str]] = {}
    for atom in branch["atom_ids"]:
        profile = _canonical_json(
            {column: str(lookup[atom]["values"][column]) for column in columns}
        )
        groups.setdefault(profile, []).append(str(atom))
    return [sorted(members) for _profile, members in sorted(groups.items())]


def _root_branches(final: pd.DataFrame) -> list[dict[str, Any]]:
    branches = []
    for semantic_bucket, rows in final.groupby("semantic_bucket_id", sort=True):
        level = str(rows["level"].iloc[0])
        source = str(rows["source_id"].iloc[0])
        members = sorted(rows["atom_id"].astype(str))
        branches.append(
            {
                "semantic_bucket_id": str(semantic_bucket),
                "branch_id": core._stable_id("rb", str(semantic_bucket), *members),
                "level": level,
                "source_id": source,
                "depth": 0,
                "consumed_columns": [],
                "atom_ids": members,
            }
        )
    return branches


def _schedule(
    final: pd.DataFrame,
    atoms: pd.DataFrame,
    cards: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    lookup = core._atom_lookup(atoms)
    rows = []
    prompts = []
    for branch in _root_branches(final):
        prompt, candidates = _coherence_prompt(branch, lookup, cards)
        if not candidates:
            rows.append(
                {
                    "semantic_bucket_id": branch["semantic_bucket_id"],
                    "mechanically_atomic": True,
                    "coherence_prompt_sha256": None,
                    "candidate_columns": [],
                    "merge_prompt_sha256": {},
                }
            )
            continue
        merge_hashes = {}
        prompts.append(("readout_coherence", branch["semantic_bucket_id"], prompt))
        for column in candidates:
            merge_prompt, _items = _merge_prompt(branch, column, lookup, cards)
            merge_hashes[column] = hashlib.sha256(merge_prompt.encode()).hexdigest()
            prompts.append(("readout_merge_values", f"{branch['semantic_bucket_id']}|{column}", merge_prompt))
        profile_prompt, _items, _columns = _profile_prompt(
            branch, candidates[0], lookup, cards
        )
        prompts.append(
            ("readout_merge_profiles", branch["semantic_bucket_id"], profile_prompt)
        )
        rows.append(
            {
                "semantic_bucket_id": branch["semantic_bucket_id"],
                "mechanically_atomic": False,
                "coherence_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "candidate_columns": candidates,
                "merge_prompt_sha256": merge_hashes,
            }
        )
    return rows, prompts


def _endpoint_receipts() -> tuple[int, list[dict[str, Any]]]:
    receipts = []
    maximum_context = None
    for spec in core.ENDPOINTS:
        base_url = str(spec["base_url"])
        health = httpx.get(base_url.rsplit("/v1", 1)[0] + "/health", timeout=10)
        health.raise_for_status()
        response = httpx.get(f"{base_url}/models", timeout=10)
        response.raise_for_status()
        matching = [row for row in response.json().get("data", []) if row.get("id") == core.MODEL]
        if len(matching) != 1:
            raise ValueError(f"endpoint {spec['name']} does not uniquely serve {core.MODEL}")
        context = int(matching[0].get("max_model_len") or 0)
        if context <= 0:
            raise ValueError(f"endpoint {spec['name']} omitted max_model_len")
        maximum_context = context if maximum_context is None else min(maximum_context, context)
        receipts.append(
            {
                "name": spec["name"],
                "base_url": base_url,
                "health_status": health.status_code,
                "model": core.MODEL,
                "max_model_len": context,
            }
        )
    if maximum_context is None:
        raise ValueError("no DeepSeek endpoints are configured")
    return maximum_context, receipts


def prepare_readout_prompts(task: str, output: Path) -> dict[str, Any]:
    spec = configure(task)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"prompt review directory is not empty: {output}")
    freeze_path = Path(spec["artifact_root"]) / "semantic_bucket_map_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "complete_reviewed" or freeze.get("cross_source_merging") is not False:
        raise ValueError("source-local semantic freeze is required")
    atoms, _source = _load_inputs(spec)
    final = pd.read_parquet(Path(spec["artifact_root"]) / "semantic_bucket_map.parquet")
    cards = core._read_gzip_json(Path(spec["input_root"]) / "sample_cards.json.gz")
    schedule, prompts = _schedule(final, atoms, cards)
    maximum_context, endpoints = _endpoint_receipts()

    output.mkdir(parents=True)
    representatives = []
    for kind in (
        "readout_coherence",
        "readout_merge_values",
        "readout_merge_profiles",
    ):
        candidates = sorted(
            (row for row in prompts if row[0] == kind), key=lambda row: len(row[2]), reverse=True
        )[:10]
        if not candidates:
            continue
        measured = [(core._token_count(prompt), identity, prompt) for _, identity, prompt in candidates]
        tokens, identity, prompt = max(measured)
        path = output / f"{kind}.txt"
        path.write_text(prompt + "\n", encoding="utf-8")
        reserve = core.LOW_MAX_TOKENS if kind == "readout_coherence" else core.HIGH_MAX_TOKENS
        representatives.append(
            {
                "kind": kind,
                "identity": identity,
                "path": path.name,
                "sha256": _sha256(path),
                "input_tokens": tokens,
                "completion_token_reserve": reserve,
                "fits_model_context": tokens + reserve <= maximum_context,
            }
        )
    manifest = {
        "version": f"{READOUT_VERSION}.prompt_review",
        "status": "awaiting_user_prompt_approval",
        "completion_requests_made": 0,
        "task": spec["task"],
        "model": core.MODEL,
        "endpoint_receipts": endpoints,
        "max_model_len": maximum_context,
        "per_endpoint_parallelism": 64,
        "semantic_freeze_manifest_sha256": _sha256(freeze_path),
        "templates": {
            path.name: _sha256(path) for path in sorted(PROMPT_ROOT.glob("*.jinja"))
        },
        "settings": {
            "maximum_depth": READOUT_MAXIMUM_DEPTH,
            "sample_limit": core.SAMPLE_LIMIT,
            "samples_from_distinct_pair_buckets": True,
            "coherence_reasoning": "low",
            "value_merge_reasoning": "high",
            "mechanically_atomic_bypasses_model": True,
            "depth_cap_split_vetoes_profile_merge": True,
            "cross_source_merging": False,
        },
        "root_count": len(schedule),
        "mechanically_atomic_root_count": sum(row["mechanically_atomic"] for row in schedule),
        "model_review_root_count": sum(not row["mechanically_atomic"] for row in schedule),
        "root_schedule_sha256": hashlib.sha256(_canonical_json(schedule).encode()).hexdigest(),
        "representative_prompts": representatives,
        "non_sendable_prompts": [row["path"] for row in representatives if not row["fits_model_context"]],
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def _request_summary(connection: Any, request_ids: Iterable[str]) -> dict[str, int]:
    rows = []
    request_ids = sorted(set(request_ids))
    for start in range(0, len(request_ids), 500):
        batch = request_ids[start : start + 500]
        placeholders = ",".join("?" for _ in batch)
        rows.extend(
            connection.execute(
                f"SELECT status,input_tokens,output_tokens FROM requests "
                f"WHERE request_id IN ({placeholders})",
                batch,
            )
        )
    statuses = Counter(str(row["status"]) for row in rows)
    return {
        "requests": len(rows),
        "complete": statuses["complete"],
        "failed": statuses["failed"],
        "pending": statuses["pending"],
        "input_tokens": sum(int(row["input_tokens"] or 0) for row in rows),
        "output_tokens": sum(int(row["output_tokens"] or 0) for row in rows),
    }


def run_readouts(
    task: str,
    *,
    review_manifest_path: Path,
    approved_review_sha256: str,
    parallelism: int = 64,
) -> dict[str, Any]:
    spec = configure(task)
    if _sha256(review_manifest_path) != approved_review_sha256:
        raise ValueError("approved readout prompt review hash mismatch")
    review = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    if review.get("version") != f"{READOUT_VERSION}.prompt_review":
        raise ValueError("unexpected readout prompt review version")
    if review.get("status") != "awaiting_user_prompt_approval" or review.get("completion_requests_made") != 0:
        raise ValueError("readout prompt review is not an approval candidate")
    if review.get("task") != spec["task"]:
        raise ValueError("readout prompt review targets another task")

    root = Path(spec["readout_root"])
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("approved_prompt_review_sha256") != approved_review_sha256:
            raise ValueError("existing readout run used another prompt review")
        if manifest.get("status") in {"awaiting_agentic_review", "complete_reviewed"}:
            return manifest
    else:
        seed_path = Path(spec["readout_seed_cache"])
        seed_receipt = core._seed_request_database(
            root / "requests.sqlite3", seed_path if seed_path.is_file() else None
        )
        manifest = {
            "version": READOUT_VERSION,
            "status": "running",
            "task": spec["task"],
            "approved_prompt_review": str(review_manifest_path),
            "approved_prompt_review_sha256": approved_review_sha256,
            "seed_request_cache": seed_receipt,
        }
        write_json_atomic(manifest_path, manifest)

    freeze_path = Path(spec["artifact_root"]) / "semantic_bucket_map_manifest.json"
    if review["semantic_freeze_manifest_sha256"] != _sha256(freeze_path):
        raise ValueError("approved semantic freeze changed")
    if review["templates"] != {
        path.name: _sha256(path) for path in sorted(PROMPT_ROOT.glob("*.jinja"))
    }:
        raise ValueError("approved readout templates changed")
    atoms, _source = _load_inputs(spec)
    final = pd.read_parquet(Path(spec["artifact_root"]) / "semantic_bucket_map.parquet")
    cards = core._read_gzip_json(Path(spec["input_root"]) / "sample_cards.json.gz")
    schedule, _prompts = _schedule(final, atoms, cards)
    if review["root_schedule_sha256"] != hashlib.sha256(_canonical_json(schedule).encode()).hexdigest():
        raise ValueError("approved readout root schedule changed")

    maximum_context, health_receipts = _endpoint_receipts()
    client, endpoint_receipts, total_parallelism = core._build_completion_pool(parallelism)
    slots = threading.BoundedSemaphore(total_parallelism)
    connection = core._request_database(root / "requests.sqlite3")
    lookup = core._atom_lookup(atoms)
    active = _root_branches(final)
    terminal = []
    blocked = []
    decisions = []
    used_request_ids: set[str] = set()
    try:
        for depth in range(READOUT_MAXIMUM_DEPTH + 1):
            if any(int(branch["depth"]) != depth for branch in active):
                raise ValueError("readout phase barrier received the wrong depth")
            model_branches = []
            request_ids = []
            for branch in active:
                prompt, candidates = _coherence_prompt(branch, lookup, cards)
                if not candidates:
                    terminal.append(
                        {
                            **branch,
                            "readout_depth": max(1, depth),
                            "termination_reason": "mechanically_atomic",
                        }
                    )
                    decisions.append(
                        {
                            "semantic_bucket_id": branch["semantic_bucket_id"],
                            "branch_id": branch["branch_id"],
                            "depth": depth,
                            "kind": "mechanical",
                            "decision": "coherent",
                            "column": None,
                            "request_id": None,
                            "response_json": None,
                        }
                    )
                    continue
                request_id = core._queue_checked_request(
                    connection,
                    kind="readout_coherence",
                    phase=f"readout|depth{depth}|{branch['branch_id']}",
                    prompt=prompt,
                    reasoning_effort="low",
                    max_tokens=core.LOW_MAX_TOKENS,
                    validation={"candidate_columns": candidates},
                    maximum_context=maximum_context,
                    commit=False,
                )
                if request_id is None:
                    raise ValueError(f"readout coherence prompt exceeds context: {branch['branch_id']}")
                model_branches.append((branch, request_id))
                request_ids.append(request_id)
                used_request_ids.add(request_id)
            connection.commit()
            core._run_pending(
                connection,
                request_ids,
                parallelism=total_parallelism,
                client=client,
                request_slots=slots,
            )

            split_branches = []
            for branch, request_id in model_branches:
                response = core._response(connection, request_id)
                decisions.append(
                    {
                        "semantic_bucket_id": branch["semantic_bucket_id"],
                        "branch_id": branch["branch_id"],
                        "depth": depth,
                        "kind": "coherence",
                        "decision": response["decision"],
                        "column": response.get("column"),
                        "request_id": request_id,
                        "response_json": _canonical_json(response),
                    }
                )
                if response["decision"] == "coherent":
                    terminal.append(
                        {
                            **branch,
                            "readout_depth": max(1, depth),
                            "termination_reason": "deepseek_coherent",
                        }
                    )
                elif depth == READOUT_MAXIMUM_DEPTH:
                    profile_groups = _exact_profile_groups(branch, lookup)
                    if len(profile_groups) < 2:
                        blocked.append(
                            {
                                **branch,
                                "blocked_depth": depth,
                                "suggested_column": response["column"],
                                "termination_reason": "incoherent_atomic_profile",
                            }
                        )
                        continue
                    for members in profile_groups:
                        terminal.append(
                            {
                                **branch,
                                "branch_id": core._stable_id(
                                    "rb", branch["semantic_bucket_id"], *members
                                ),
                                "atom_ids": members,
                                "readout_depth": depth,
                                "termination_reason": "depth_cap_veto_unmerged_profiles",
                            }
                        )
                else:
                    split_branches.append((branch, str(response["column"])))

            merge_requests = []
            for branch, column in split_branches:
                if depth == READOUT_MAXIMUM_DEPTH - 1:
                    prompt, items, consumed_after = _profile_prompt(
                        branch, column, lookup, cards
                    )
                    kind = "readout_merge_profiles"
                else:
                    prompt, items = _merge_prompt(branch, column, lookup, cards)
                    consumed_after = [*branch["consumed_columns"], column]
                    kind = "readout_merge_values"
                request_id = core._queue_checked_request(
                    connection,
                    kind=kind,
                    phase=f"readout|depth{depth}|values|{branch['branch_id']}|{column}",
                    prompt=prompt,
                    reasoning_effort="high",
                    max_tokens=core.HIGH_MAX_TOKENS,
                    validation={"valid_ids": sorted(items)},
                    maximum_context=maximum_context,
                    commit=False,
                )
                if request_id is None:
                    raise ValueError(f"readout value prompt exceeds context: {branch['branch_id']} {column}")
                merge_requests.append(
                    (branch, column, items, consumed_after, kind, request_id)
                )
                used_request_ids.add(request_id)
            connection.commit()
            core._run_pending(
                connection,
                [row[5] for row in merge_requests],
                parallelism=total_parallelism,
                client=client,
                request_slots=slots,
            )

            next_active = []
            for branch, column, items, consumed_after, kind, request_id in merge_requests:
                response = core._response(connection, request_id)
                children = core.apply_merge_sets(items, response)
                decisions.append(
                    {
                        "semantic_bucket_id": branch["semantic_bucket_id"],
                        "branch_id": branch["branch_id"],
                        "depth": depth,
                        "kind": kind,
                        "decision": "collapsed" if len(children) == 1 else "split",
                        "column": column,
                        "request_id": request_id,
                        "response_json": _canonical_json(response),
                    }
                )
                if len(children) == 1:
                    terminal.append(
                        {
                            **branch,
                            "readout_depth": max(1, depth),
                            "termination_reason": "high_reasoning_merge_reversed_split",
                        }
                    )
                    continue
                for members in children.values():
                    members = sorted(members)
                    next_active.append(
                        {
                            "semantic_bucket_id": branch["semantic_bucket_id"],
                            "branch_id": core._stable_id(
                                "rb", branch["semantic_bucket_id"], *members
                            ),
                            "level": branch["level"],
                            "source_id": branch["source_id"],
                            "depth": depth + 1,
                            "consumed_columns": consumed_after,
                            "atom_ids": members,
                        }
                    )
            active = next_active

        if active:
            raise AssertionError("readout loop retained branches beyond the depth cap")
        decision_path = root / "decisions.parquet"
        pd.DataFrame(decisions).to_parquet(decision_path, index=False)
        manifest.update(
            {
                "model": core.MODEL,
                "maximum_depth": READOUT_MAXIMUM_DEPTH,
                "per_endpoint_parallelism": parallelism,
                "total_parallelism": total_parallelism,
                "health_receipts": health_receipts,
                "endpoint_pool_at_start": endpoint_receipts,
                "endpoint_pool_final_snapshot": client.snapshot(),
                "request_summary": _request_summary(connection, used_request_ids),
                "decision_ledger": {
                    "path": decision_path.name,
                    "sha256": _sha256(decision_path),
                },
            }
        )
        if blocked:
            blocked_path = root / "blocked_branches.parquet"
            pd.DataFrame(blocked).drop(columns="atom_ids").to_parquet(blocked_path, index=False)
            manifest.update(
                {
                    "status": "blocked_incoherent_at_depth_cap",
                    "blocked_branch_count": len(blocked),
                    "blocked_branches": {"path": blocked_path.name, "sha256": _sha256(blocked_path)},
                }
            )
            return manifest

        atom_to_terminal = {}
        candidate_rows = []
        for branch in terminal:
            for atom in branch["atom_ids"]:
                if atom in atom_to_terminal:
                    raise ValueError("readout terminal buckets overlap")
                atom_to_terminal[atom] = branch["branch_id"]
                candidate_rows.append(
                    {
                        "level": branch["level"],
                        "source_id": branch["source_id"],
                        "semantic_bucket_id": branch["semantic_bucket_id"],
                        "readout_bucket_id": branch["branch_id"],
                        "readout_depth": branch["readout_depth"],
                        "termination_reason": branch["termination_reason"],
                        "atom_id": atom,
                    }
                )
        if set(atom_to_terminal) != set(final["atom_id"]):
            raise ValueError("readout candidate map changed atom coverage")
        candidate = pd.DataFrame(candidate_rows).sort_values(
            ["level", "source_id", "semantic_bucket_id", "readout_bucket_id", "atom_id"]
        )
        candidate_path = root / "candidate_readout_bucket_map.parquet"
        candidate.to_parquet(candidate_path, index=False)
        manifest.update(
            {
                "status": "awaiting_agentic_review",
                "candidate_map": {
                    "path": candidate_path.name,
                    "sha256": _sha256(candidate_path),
                    "rows": len(candidate),
                    "readout_bucket_count": int(candidate["readout_bucket_id"].nunique()),
                },
            }
        )
        return manifest
    except Exception:
        manifest["status"] = "incomplete"
        raise
    finally:
        manifest["updated_at"] = time.time()
        write_json_atomic(manifest_path, manifest)
        connection.close()


def publish_readouts(
    task: str,
    review_path: Path,
    *,
    output: Path | None = None,
    readout_version: str | None = None,
) -> dict[str, Any]:
    spec = configure(task)
    root = output or Path(spec["readout_root"])
    version = readout_version or READOUT_VERSION
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "awaiting_agentic_review":
        raise ValueError("readout run is not awaiting agentic review")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    expected = {"version", "reviewer", "candidate_map_sha256", "decision", "rationale"}
    if set(review) != expected or review.get("version") != f"{version}.review":
        raise ValueError("invalid readout review schema")
    if review.get("decision") != "approve":
        raise ValueError("readout review did not approve publication")
    candidate_path = root / manifest["candidate_map"]["path"]
    if review["candidate_map_sha256"] != _sha256(candidate_path):
        raise ValueError("readout review targets another candidate map")

    final_path = root / "readout_bucket_map.parquet"
    candidate_path.replace(final_path)
    mapping = pd.read_parquet(final_path)
    records = pd.read_parquet(core.RECORD_MAP)
    records = records[records["level"].isin(core.LEVELS)].copy()
    records["atom_id"] = _record_atoms(records)
    columns = ["canonical_record_id", "source_row_uid", "level", "source_id", "atom_id"]
    record_map = records[columns].merge(
        mapping[["atom_id", "semantic_bucket_id", "readout_bucket_id", "readout_depth"]],
        on="atom_id",
        how="left",
        validate="many_to_one",
    )
    if len(record_map) != core.EXPECTED_RECORD_COUNT or record_map["readout_bucket_id"].isna().any():
        raise ValueError("published readout map does not cover every scoped V10 record")
    record_path = root / "record_readout_bucket_map.parquet"
    record_map.to_parquet(record_path, index=False)
    manifest.update(
        {
            "status": "complete_reviewed",
            "review": str(review_path),
            "review_sha256": _sha256(review_path),
            "readout_bucket_map": {
                "path": final_path.name,
                "sha256": _sha256(final_path),
                "rows": len(mapping),
                "readout_bucket_count": int(mapping["readout_bucket_id"].nunique()),
            },
            "record_readout_bucket_map": {
                "path": record_path.name,
                "sha256": _sha256(record_path),
                "rows": len(record_map),
            },
        }
    )
    manifest.pop("candidate_map", None)
    write_json_atomic(manifest_path, manifest)
    return manifest


def configure_rankings(task: str) -> dict[str, Any]:
    spec = configure(task)
    degree25.VERSION = spec["ranking_version"]
    degree25.TASK_ID = spec["task"]
    degree25.RECORD_COUNT = core.EXPECTED_RECORD_COUNT
    degree25.SEMANTIC_ROOT = Path(spec["artifact_root"])
    degree25.INPUT_ROOT = Path(spec["input_root"])
    degree25.FINAL_MAP = Path(spec["artifact_root"]) / "semantic_bucket_map.parquet"
    degree25.FINAL_MAP_MANIFEST = Path(spec["artifact_root"]) / "semantic_bucket_map_manifest.json"
    degree25.DEFAULT_OUTPUT = Path(spec["ranking_root"])
    degree25.TEMPLATE = Path(spec["ranking_template"])
    degree25.RANKING_PROMPT_APPROVAL_MANIFEST = degree25.FINAL_MAP_MANIFEST
    degree25.RANKING_PROMPT_APPROVAL_FIELD = "ranking_prompt_sha256"
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze-semantic", "build-rankings", "run-rankings"):
        item = subparsers.add_parser(command)
        item.add_argument("--task", choices=sorted(TASK_ALIASES), required=True)
        if command == "run-rankings":
            item.add_argument("--parallelism", type=int, default=64)
    prepare = subparsers.add_parser("prepare-readout-prompts")
    prepare.add_argument("--task", choices=sorted(TASK_ALIASES), required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run-readouts")
    run.add_argument("--task", choices=sorted(TASK_ALIASES), required=True)
    run.add_argument("--review-manifest", type=Path, required=True)
    run.add_argument("--approved-review-sha256", required=True)
    run.add_argument("--parallelism", type=int, default=64)
    publish = subparsers.add_parser("publish-readouts")
    publish.add_argument("--task", choices=sorted(TASK_ALIASES), required=True)
    publish.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "freeze-semantic":
        result = freeze_semantics(args.task)
    elif args.command == "prepare-readout-prompts":
        result = prepare_readout_prompts(args.task, args.output)
    elif args.command == "run-readouts":
        result = run_readouts(
            args.task,
            review_manifest_path=args.review_manifest,
            approved_review_sha256=args.approved_review_sha256,
            parallelism=args.parallelism,
        )
    elif args.command == "publish-readouts":
        result = publish_readouts(args.task, args.review)
    elif args.command == "build-rankings":
        spec = configure_rankings(args.task)
        result = degree25.build(Path(spec["ranking_root"]), final_map=degree25.FINAL_MAP)
    else:
        spec = configure_rankings(args.task)
        _maximum_context, health_receipts = _endpoint_receipts()
        ranking_manifest_path = Path(spec["ranking_root"]) / "manifest.json"
        ranking_manifest = json.loads(ranking_manifest_path.read_text(encoding="utf-8"))
        ranking_manifest["health_receipts_at_start"] = health_receipts
        write_json_atomic(ranking_manifest_path, ranking_manifest)
        result = degree25.run(Path(spec["ranking_root"]), parallelism=args.parallelism)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
