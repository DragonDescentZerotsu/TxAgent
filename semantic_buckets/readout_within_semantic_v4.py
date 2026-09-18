"""Risk-repair V3 readouts and rank them only within frozen semantic parents."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping

from jinja2 import Environment, StrictUndefined
import pandas as pd

from semantic_buckets import bioavailability_semantic_degree25 as ranking
from semantic_buckets import bioavailability_semantic_readout_v1 as core
from semantic_buckets import source_local_semantic_v3 as source_local
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "readout_within_semantic.v4"
DEPTH_ONE_VERSION = "readout_within_semantic.v5_depth1"
DEGREE = 15
SAMPLE_LIMIT = 12
RANKING_RECORD_LIMIT = 3
L2_PENALTY = 0.3
ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = Path(__file__).with_name("prompts")
READOUT_PROMPT_ROOT = PROMPT_ROOT / "source_local_readout_v4"
RISK_THRESHOLDS = {
    "bbb_martins": {"children": 11, "atoms": 44, "records": 93},
    "bioavailability_ma": {"children": 7, "atoms": 22, "records": 274},
}
TEMPLATES = {
    "bbb_martins": PROMPT_ROOT / "readout_within_semantic_bbb_v1.jinja",
    "bioavailability_ma": PROMPT_ROOT / "readout_within_semantic_oral_v1.jinja",
}
DEPTH_ONE_TEMPLATES = {
    "bbb_martins": PROMPT_ROOT / "readout_within_semantic_bbb_v3.jinja",
    "bioavailability_ma": PROMPT_ROOT / "readout_within_semantic_oral_v3.jinja",
}
TECHNICAL_CONDITION_FIELDS = {"condition_group", "canonical_direct_condition_group"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths(task: str, *, depth_one: bool = False) -> dict[str, Any]:
    spec = source_local.configure(task)
    root = Path(spec["artifact_root"]) / (
        "readout_buckets_v5_depth1" if depth_one else "readout_buckets_v4"
    )
    return {
        **spec,
        "root": root,
        "readout_version": DEPTH_ONE_VERSION if depth_one else VERSION,
        "depth_one": depth_one,
        "old_root": Path(spec["readout_root"]),
        "semantic_map": Path(spec["artifact_root"]) / "semantic_bucket_map.parquet",
        "ranking_root": root / "degree15_within_semantic_rankings_v2",
        "template": (DEPTH_ONE_TEMPLATES if depth_one else TEMPLATES)[str(spec["task"])],
    }


def _configure(task: str, *, depth_one: bool = False) -> dict[str, Any]:
    spec = _paths(task, depth_one=depth_one)
    core.SAMPLE_LIMIT = SAMPLE_LIMIT
    source_local.PROMPT_ROOT = READOUT_PROMPT_ROOT
    return spec


def _atom_values(atoms: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        str(row.atom_id): json.loads(row.values_json)
        for row in atoms[["atom_id", "values_json"]].itertuples(index=False)
    }


def _sample_atom_ids(
    bucket_id: str,
    atom_ids: list[str],
    lookup: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Greedily cover rare canonical values, with stable-hash tie breaking."""
    source = str(lookup[atom_ids[0]]["source_id"])
    columns = core.PROMPT_DIMENSION_COLUMNS[source]
    features = {
        atom: {(column, str(lookup[atom]["values"][column])) for column in columns}
        for atom in atom_ids
    }
    counts = Counter(feature for atom in atom_ids for feature in features[atom])
    selected: list[str] = []
    covered: set[tuple[str, str]] = set()
    remaining = set(atom_ids)
    while remaining and len(selected) < SAMPLE_LIMIT:
        atom = min(
            remaining,
            key=lambda candidate: (
                -sum(
                    1.0 / counts[feature]
                    for feature in features[candidate] - covered
                ),
                core._stable_id("readout-sample", bucket_id, candidate),
                candidate,
            ),
        )
        selected.append(atom)
        covered.update(features[atom])
        remaining.remove(atom)
    return selected


def _bucket_payload(
    bucket_id: str,
    atom_ids: list[str],
    lookup: Mapping[str, Mapping[str, Any]],
    cards: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source = str(lookup[atom_ids[0]]["source_id"])
    columns = core.PROMPT_DIMENSION_COLUMNS[source]
    return {
        "bucket_id": bucket_id,
        "pair_bucket_count": len(atom_ids),
        "record_count": sum(int(lookup[atom]["record_count"]) for atom in atom_ids),
        "canonical_dimensions": {
            column: sorted({str(lookup[atom]["values"][column]) for atom in atom_ids})
            for column in columns
        },
        "sample_records": [
            {column: cards[atom][column] for column in columns if column in cards[atom]}
            for atom in _sample_atom_ids(bucket_id, atom_ids, lookup)
        ],
    }


def _render_readout(name: str, payload: Mapping[str, Any]) -> str:
    template = Environment(undefined=StrictUndefined, autoescape=False).from_string(
        (READOUT_PROMPT_ROOT / f"{name}.jinja").read_text(encoding="utf-8")
    )
    return template.render(compact_payload_json=core._canonical_json(payload)).strip()


def _varying_columns(
    branch: Mapping[str, Any], lookup: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    source = str(branch["source_id"])
    consumed = set(branch["consumed_columns"])
    return [
        column
        for column in core.REFINEMENT_COLUMNS[source]
        if column not in consumed
        and column not in TECHNICAL_CONDITION_FIELDS
        and len({str(lookup[atom]["values"][column]) for atom in branch["atom_ids"]}) > 1
    ]


def _coherence_prompt(
    branch: Mapping[str, Any],
    lookup: Mapping[str, Mapping[str, Any]],
    cards: Mapping[str, Mapping[str, Any]],
) -> tuple[str, list[str]]:
    candidates = _varying_columns(branch, lookup)
    payload = _bucket_payload(
        str(branch["branch_id"]), list(branch["atom_ids"]), lookup, cards
    )
    payload.update({
        "task": core.TASK_NAME,
        "level": branch["level"],
        "source_id": branch["source_id"],
        "semantic_bucket_id": branch["semantic_bucket_id"],
        "refinement_round": branch["round"],
        "consumed_readout_columns": list(branch["consumed_columns"]),
        "candidate_columns": [
            {
                "column": column,
                "description": core.COLUMN_DESCRIPTIONS[column],
                "distinct_values": len(
                    {str(lookup[atom]["values"][column]) for atom in branch["atom_ids"]}
                ),
            }
            for column in candidates
        ],
    })
    return _render_readout("readout_coherence", payload), candidates


def _merge_prompt(
    branch: Mapping[str, Any],
    prior_decision: Mapping[str, Any],
    lookup: Mapping[str, Mapping[str, Any]],
    cards: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, list[str]]]:
    column = str(prior_decision["column"])
    payload, items = core._value_payload(
        str(branch["branch_id"]),
        branch["atom_ids"],
        column,
        lookup,
        context_columns=[*branch["consumed_columns"], column],
    )
    payload.update({
        "task": core.TASK_NAME,
        "level": branch["level"],
        "source_id": branch["source_id"],
        "semantic_bucket_id": branch["semantic_bucket_id"],
        "refinement_round": branch["round"],
        "prior_split_decision": dict(prior_decision),
        "sample_records": _bucket_payload(
            str(branch["branch_id"]), list(branch["atom_ids"]), lookup, cards
        )["sample_records"],
    })
    return _render_readout("readout_merge_values", payload), items


def _root_branches(mapping: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "semantic_bucket_id": str(parent),
            "branch_id": core._stable_id("rb", str(parent), *sorted(rows["atom_id"])),
            "level": str(rows["level"].iloc[0]),
            "source_id": str(rows["source_id"].iloc[0]),
            "round": 0,
            "consumed_columns": [],
            "atom_ids": sorted(rows["atom_id"].astype(str)),
        }
        for parent, rows in mapping.groupby("semantic_bucket_id", sort=True)
    ]


def audit_risk(task: str) -> dict[str, Any]:
    """Select V3 semantic parents whose readout children warrant re-review."""
    spec = _configure(task)
    root = Path(spec["root"])
    root.mkdir(parents=True, exist_ok=True)
    old_map_path = Path(spec["old_root"]) / "readout_bucket_map.parquet"
    old_manifest_path = Path(spec["old_root"]) / "manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    if old_manifest.get("status") != "complete_reviewed":
        raise ValueError("risk repair requires a complete reviewed V3 readout map")
    if old_manifest["readout_bucket_map"]["sha256"] != _sha256(old_map_path):
        raise ValueError("V3 readout map hash changed")

    atoms, _ = source_local._load_inputs(spec)
    mapping = pd.read_parquet(old_map_path)
    values = _atom_values(atoms)
    record_counts = atoms.set_index("atom_id")["record_count"].astype(int)
    child = (
        mapping.groupby(
            ["level", "source_id", "semantic_bucket_id", "readout_bucket_id"],
            sort=True,
        )
        .agg(
            atom_count=("atom_id", "size"),
            termination_reason=("termination_reason", "first"),
        )
        .reset_index()
    )
    child_records = (
        mapping.assign(record_count=mapping["atom_id"].map(record_counts))
        .groupby("readout_bucket_id", sort=True)["record_count"]
        .sum()
    )
    child["record_count"] = child["readout_bucket_id"].map(child_records).astype(int)
    thresholds = RISK_THRESHOLDS[str(spec["task"])]
    child["risk_depth_or_reversal"] = child["termination_reason"].isin(
        {"depth_cap_veto_unmerged_profiles", "high_reasoning_merge_reversed_split"}
    )
    child["risk_large_child"] = (child["atom_count"] >= thresholds["atoms"]) | (
        child["record_count"] >= thresholds["records"]
    )

    varied_children: set[str] = set()
    coherent = child[child["termination_reason"] == "deepseek_coherent"]
    members = mapping.groupby("readout_bucket_id", sort=False)["atom_id"].agg(list)
    for row in coherent.itertuples(index=False):
        dimensions = [
            column
            for column in core.PROMPT_DIMENSION_COLUMNS[str(row.source_id)]
            if column not in TECHNICAL_CONDITION_FIELDS
        ]
        atom_ids = members[str(row.readout_bucket_id)]
        if any(len({str(values[atom].get(column)) for atom in atom_ids}) > 1 for column in dimensions):
            varied_children.add(str(row.readout_bucket_id))
    child["risk_coherent_but_varied"] = child["readout_bucket_id"].isin(varied_children)

    parent = (
        child.groupby(["level", "source_id", "semantic_bucket_id"], sort=True)
        .agg(
            child_count=("readout_bucket_id", "size"),
            max_child_atoms=("atom_count", "max"),
            max_child_records=("record_count", "max"),
            risk_depth_or_reversal=("risk_depth_or_reversal", "max"),
            risk_large_child=("risk_large_child", "max"),
            risk_coherent_but_varied=("risk_coherent_but_varied", "max"),
        )
        .reset_index()
    )
    parent["risk_many_children"] = parent["child_count"] >= thresholds["children"]
    flags = [
        "risk_depth_or_reversal",
        "risk_many_children",
        "risk_large_child",
        "risk_coherent_but_varied",
    ]
    parent["selected_for_repair"] = parent[flags].any(axis=1)
    selected = set(parent.loc[parent["selected_for_repair"], "semantic_bucket_id"])
    semantic = pd.read_parquet(spec["semantic_map"])
    repair_map = semantic[semantic["semantic_bucket_id"].isin(selected)].copy()
    carry = mapping[~mapping["semantic_bucket_id"].isin(selected)].copy()

    parent_path = root / "risk_parent_audit.parquet"
    child_path = root / "risk_child_audit.parquet"
    repair_path = root / "repair_semantic_map.parquet"
    carry_path = root / "carried_forward_v3_map.parquet"
    parent.to_parquet(parent_path, index=False)
    child.to_parquet(child_path, index=False)
    repair_map.to_parquet(repair_path, index=False)
    carry.to_parquet(carry_path, index=False)
    manifest = {
        "version": VERSION,
        "status": "risk_audited",
        "task": spec["task"],
        "semantic_map_sha256": _sha256(spec["semantic_map"]),
        "v3_readout_map_sha256": _sha256(old_map_path),
        "thresholds": thresholds,
        "risk_rule": "union",
        "technical_condition_fields_are_provenance_only": sorted(TECHNICAL_CONDITION_FIELDS),
        "semantic_parent_count": int(parent.shape[0]),
        "selected_parent_count": len(selected),
        "repair_atom_count": len(repair_map),
        "carried_forward_atom_count": len(carry),
        "artifacts": {
            path.name: {"sha256": _sha256(path), "rows": len(frame)}
            for path, frame in (
                (parent_path, parent),
                (child_path, child),
                (repair_path, repair_map),
                (carry_path, carry),
            )
        },
    }
    write_json_atomic(root / "risk_manifest.json", manifest)
    return manifest


def _repair_inputs(spec: Mapping[str, Any], pilot: bool) -> tuple[Path, Path]:
    root = Path(spec["root"])
    if spec["depth_one"]:
        if pilot:
            raise ValueError("depth-one full rebuild does not use the risk-repair pilot")
        return Path(spec["semantic_map"]), root
    if not pilot:
        return root / "repair_semantic_map.parquet", root
    pilot_root = root / "pilot_unbounded_v3"
    pilot_map = pilot_root / "semantic_map.parquet"
    if not pilot_map.exists():
        parents = pd.read_parquet(root / "risk_parent_audit.parquet")
        selected: list[str] = []
        for flag in (
            "risk_depth_or_reversal",
            "risk_many_children",
            "risk_large_child",
            "risk_coherent_but_varied",
        ):
            candidates = parents[parents[flag] & parents["selected_for_repair"]].sort_values(
                ["max_child_atoms", "max_child_records", "semantic_bucket_id"]
            )
            for parent in candidates["semantic_bucket_id"]:
                if parent not in selected:
                    selected.append(str(parent))
                    break
        semantic = pd.read_parquet(spec["semantic_map"])
        pilot_frame = semantic[semantic["semantic_bucket_id"].isin(selected)].copy()
        pilot_root.mkdir(parents=True, exist_ok=True)
        pilot_frame.to_parquet(pilot_map, index=False)
        write_json_atomic(
            pilot_root / "selection.json",
            {"semantic_bucket_ids": selected, "rows": len(pilot_frame)},
        )
    return pilot_map, pilot_root


def prepare_repair(
    task: str, *, pilot: bool = False, depth_one: bool = False
) -> dict[str, Any]:
    spec = _configure(task, depth_one=depth_one)
    map_path, run_root = _repair_inputs(spec, pilot)
    output = run_root / "prompt_review"
    if output.exists() and any(output.iterdir()):
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("semantic_map_sha256") != _sha256(map_path):
            raise ValueError("existing prompt review targets another semantic map")
        return manifest
    atoms, _ = source_local._load_inputs(spec)
    cards = core._read_gzip_json(Path(spec["input_root"]) / "sample_cards.json.gz")
    lookup = core._atom_lookup(atoms)
    roots = _root_branches(pd.read_parquet(map_path))
    prompts: list[tuple[str, str, str]] = []
    schedule = []
    for branch in roots:
        prompt, candidates = _coherence_prompt(branch, lookup, cards)
        schedule.append({
            "semantic_bucket_id": branch["semantic_bucket_id"],
            "mechanically_atomic": not candidates,
            "coherence_prompt_sha256": (
                hashlib.sha256(prompt.encode()).hexdigest() if candidates else None
            ),
        })
        if candidates:
            prompts.append(("readout_coherence", branch["semantic_bucket_id"], prompt))
            for column in candidates:
                merge_prompt, _ = _merge_prompt(
                    branch,
                    {
                        "decision": "split",
                        "column": column,
                        "rationale": "Representative prior split rationale.",
                    },
                    lookup,
                    cards,
                )
                prompts.append((
                    "readout_merge_values",
                    f"{branch['semantic_bucket_id']}|{column}",
                    merge_prompt,
                ))
    maximum_context, endpoints = source_local._endpoint_receipts()
    output.mkdir(parents=True)
    representatives = []
    for kind in ("readout_coherence", "readout_merge_values"):
        candidates = sorted(
            (row for row in prompts if row[0] == kind),
            key=lambda row: len(row[2]),
            reverse=True,
        )[:10]
        if not candidates:
            continue
        measured = [
            (core._token_count(prompt), identity, prompt)
            for _, identity, prompt in candidates
        ]
        tokens, identity, prompt = max(measured)
        path = output / f"{kind}.txt"
        path.write_text(prompt + "\n", encoding="utf-8")
        reserve = core.LOW_MAX_TOKENS if kind == "readout_coherence" else core.HIGH_MAX_TOKENS
        representatives.append({
            "kind": kind,
            "identity": identity,
            "path": path.name,
            "sha256": _sha256(path),
            "input_tokens": tokens,
            "fits_model_context": tokens + reserve <= maximum_context,
        })
    manifest = {
        "version": spec["readout_version"] + ".prompt_review",
        "status": "awaiting_user_prompt_approval",
        "completion_requests_made": 0,
        "task": spec["task"],
        "pilot": pilot,
        "scope": "all_semantic_parents" if depth_one else "risk_selected_parents",
        "maximum_depth": 1 if depth_one else None,
        "model": core.MODEL,
        "endpoint_receipts": endpoints,
        "max_model_len": maximum_context,
        "semantic_freeze_manifest_sha256": _sha256(
            Path(spec["artifact_root"]) / "semantic_bucket_map_manifest.json"
        ),
        "semantic_map_sha256": _sha256(map_path),
        "templates": {
            path.name: _sha256(path) for path in sorted(READOUT_PROMPT_ROOT.glob("*.jinja"))
        },
        "sample_limit": SAMPLE_LIMIT,
        "root_count": len(roots),
        "mechanically_atomic_root_count": sum(not _varying_columns(row, lookup) for row in roots),
        "root_schedule_sha256": hashlib.sha256(
            core._canonical_json(schedule).encode()
        ).hexdigest(),
        "representative_prompts": representatives,
        "non_sendable_prompts": [
            row["path"] for row in representatives if not row["fits_model_context"]
        ],
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def run_repair(
    task: str,
    review_sha256: str,
    *,
    parallelism: int,
    pilot: bool = False,
    depth_one: bool = False,
) -> dict[str, Any]:
    spec = _configure(task, depth_one=depth_one)
    version = str(spec["readout_version"])
    map_path, run_root = _repair_inputs(spec, pilot)
    review_path = run_root / "prompt_review" / "manifest.json"
    if _sha256(review_path) != review_sha256:
        raise ValueError("approved prompt review hash mismatch")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        review.get("version") != version + ".prompt_review"
        or review.get("status") != "awaiting_user_prompt_approval"
        or review.get("completion_requests_made") != 0
        or review.get("semantic_map_sha256") != _sha256(map_path)
    ):
        raise ValueError("prompt review contract changed")
    if review["templates"] != {
        path.name: _sha256(path) for path in sorted(READOUT_PROMPT_ROOT.glob("*.jinja"))
    }:
        raise ValueError("approved readout prompts changed")

    manifest_path = run_root / "manifest.json"
    state_path = run_root / "readout_state.json"
    mapping = pd.read_parquet(map_path)
    atoms, _ = source_local._load_inputs(spec)
    cards = core._read_gzip_json(Path(spec["input_root"]) / "sample_cards.json.gz")
    lookup = core._atom_lookup(atoms)
    if state_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("approved_prompt_review_sha256") != review_sha256:
            raise ValueError("existing run uses another prompt review")
        if manifest.get("status") in {"awaiting_agentic_review", "complete_reviewed"}:
            return manifest
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "version": version,
            "round": 0,
            "active": _root_branches(mapping),
            "terminal": [],
            "decisions": [],
        }
        write_json_atomic(state_path, state)
        seed = None if depth_one else Path(spec["old_root"]) / "requests.sqlite3"
        seed_receipt = core._seed_request_database(
            run_root / "requests.sqlite3", seed if seed and seed.is_file() else None
        )
        manifest = {
            "version": version,
            "status": "running",
            "task": spec["task"],
            "pilot": pilot,
            "scope": "all_semantic_parents" if depth_one else "risk_selected_parents",
            "approved_prompt_review": str(review_path),
            "approved_prompt_review_sha256": review_sha256,
            "semantic_map_sha256": _sha256(map_path),
            "seed_request_cache": seed_receipt,
        }
        write_json_atomic(manifest_path, manifest)

    maximum_context, health_receipts = source_local._endpoint_receipts()
    client, endpoint_receipts, total_parallelism = core._build_completion_pool(parallelism)
    slots = threading.BoundedSemaphore(total_parallelism)
    connection = core._request_database(run_root / "requests.sqlite3")
    used_request_ids = {
        row["request_id"]
        for row in state["decisions"]
        if row.get("request_id") is not None
    }
    try:
        maximum_rounds = 1 if depth_one else max(map(len, core.REFINEMENT_COLUMNS.values())) + 1
        while state["active"]:
            if int(state["round"]) >= maximum_rounds:
                raise AssertionError("readout refinement did not consume a column per split")
            model_branches = []
            coherence_ids = []
            for branch in state["active"]:
                prompt, candidates = _coherence_prompt(branch, lookup, cards)
                if not candidates:
                    state["terminal"].append({
                        **branch,
                        "termination_reason": "columns_exhausted",
                    })
                    state["decisions"].append({
                        "semantic_bucket_id": branch["semantic_bucket_id"],
                        "branch_id": branch["branch_id"],
                        "round": branch["round"],
                        "kind": "mechanical",
                        "decision": "coherent",
                        "column": None,
                        "request_id": None,
                        "response_json": None,
                    })
                    continue
                request_id = core._queue_checked_request(
                    connection,
                    kind="readout_coherence",
                    phase=f"readout|round{branch['round']}|{branch['branch_id']}",
                    prompt=prompt,
                    reasoning_effort="low",
                    max_tokens=core.LOW_MAX_TOKENS,
                    validation={"candidate_columns": candidates},
                    maximum_context=maximum_context,
                    commit=False,
                )
                if request_id is None:
                    raise ValueError(f"coherence prompt exceeds context: {branch['branch_id']}")
                model_branches.append((branch, request_id))
                coherence_ids.append(request_id)
                used_request_ids.add(request_id)
            connection.commit()
            core._run_pending(
                connection,
                coherence_ids,
                parallelism=total_parallelism,
                client=client,
                request_slots=slots,
            )

            split_branches = []
            for branch, request_id in model_branches:
                response = core._response(connection, request_id)
                state["decisions"].append({
                    "semantic_bucket_id": branch["semantic_bucket_id"],
                    "branch_id": branch["branch_id"],
                    "round": branch["round"],
                    "kind": "coherence",
                    "decision": response["decision"],
                    "column": response.get("column"),
                    "request_id": request_id,
                    "response_json": core._canonical_json(response),
                })
                if response["decision"] == "coherent":
                    state["terminal"].append({
                        **branch,
                        "termination_reason": "deepseek_coherent",
                    })
                else:
                    split_branches.append((branch, response))

            merge_rows = []
            for branch, prior in split_branches:
                prompt, items = _merge_prompt(branch, prior, lookup, cards)
                request_id = core._queue_checked_request(
                    connection,
                    kind="readout_merge_values",
                    phase=(
                        f"readout|round{branch['round']}|values|"
                        f"{branch['branch_id']}|{prior['column']}"
                    ),
                    prompt=prompt,
                    reasoning_effort="high",
                    max_tokens=core.HIGH_MAX_TOKENS,
                    validation={
                        "valid_ids": sorted(items),
                        "require_multiple_groups": True,
                    },
                    maximum_context=maximum_context,
                    commit=False,
                )
                if request_id is None:
                    raise ValueError(f"value prompt exceeds context: {branch['branch_id']}")
                merge_rows.append((branch, prior, items, request_id))
                used_request_ids.add(request_id)
            connection.commit()
            core._run_pending(
                connection,
                [row[3] for row in merge_rows],
                parallelism=total_parallelism,
                client=client,
                request_slots=slots,
            )

            next_active = []
            for branch, prior, items, request_id in merge_rows:
                response = core._response(connection, request_id)
                children = core.apply_merge_sets(items, response)
                if len(children) < 2:
                    raise ValueError("binding readout split collapsed to one child")
                state["decisions"].append({
                    "semantic_bucket_id": branch["semantic_bucket_id"],
                    "branch_id": branch["branch_id"],
                    "round": branch["round"],
                    "kind": "readout_merge_values",
                    "decision": "split",
                    "column": prior["column"],
                    "request_id": request_id,
                    "response_json": core._canonical_json(response),
                })
                for members in children.values():
                    members = sorted(members)
                    child = {
                        "semantic_bucket_id": branch["semantic_bucket_id"],
                        "branch_id": core._stable_id(
                            "rb", branch["semantic_bucket_id"], *members
                        ),
                        "level": branch["level"],
                        "source_id": branch["source_id"],
                        "round": int(branch["round"]) + 1,
                        "consumed_columns": [
                            *branch["consumed_columns"], str(prior["column"])
                        ],
                        "atom_ids": members,
                    }
                    if depth_one:
                        state["terminal"].append({
                            **child,
                            "termination_reason": "depth_one_limit",
                        })
                    else:
                        next_active.append(child)
            state["active"] = next_active
            state["round"] = int(state["round"]) + 1
            write_json_atomic(state_path, state)

        repaired_rows = [
            {
                "level": branch["level"],
                "source_id": branch["source_id"],
                "semantic_bucket_id": branch["semantic_bucket_id"],
                "readout_bucket_id": branch["branch_id"],
                "readout_depth": max(1, int(branch["round"])),
                "termination_reason": branch["termination_reason"],
                "atom_id": atom,
            }
            for branch in state["terminal"]
            for atom in branch["atom_ids"]
        ]
        repaired = pd.DataFrame(repaired_rows)
        expected = mapping.set_index("atom_id")["semantic_bucket_id"].sort_index()
        actual = repaired.set_index("atom_id")["semantic_bucket_id"].sort_index()
        if repaired["atom_id"].duplicated().any() or not actual.equals(expected):
            raise ValueError("repaired readouts changed semantic atom coverage")
        if pilot or depth_one:
            candidate = repaired
        else:
            carry = pd.read_parquet(Path(spec["root"]) / "carried_forward_v3_map.parquet")
            old = pd.read_parquet(Path(spec["old_root"]) / "readout_bucket_map.parquet")
            old_carry = old[~old["semantic_bucket_id"].isin(set(mapping["semantic_bucket_id"]))]
            if not carry.reset_index(drop=True).equals(old_carry.reset_index(drop=True)):
                raise ValueError("non-risk V3 carry-forward map changed")
            candidate = pd.concat([carry, repaired], ignore_index=True)
        candidate = candidate.sort_values(
            ["level", "source_id", "semantic_bucket_id", "readout_bucket_id", "atom_id"]
        )
        candidate_path = run_root / "candidate_readout_bucket_map.parquet"
        candidate.to_parquet(candidate_path, index=False)
        decisions = pd.DataFrame(state["decisions"])
        decision_path = run_root / "decisions.parquet"
        decisions.to_parquet(decision_path, index=False)
        manifest.update({
            "status": "awaiting_agentic_review",
            "model": core.MODEL,
            "algorithm": (
                "parent_local_single_split" if depth_one
                else "parent_local_split_only_columns_exhausted"
            ),
            "cross_bucket_merging": False,
            "maximum_depth": 1 if depth_one else None,
            "per_endpoint_parallelism": parallelism,
            "total_parallelism": total_parallelism,
            "health_receipts": health_receipts,
            "endpoint_pool_at_start": endpoint_receipts,
            "endpoint_pool_final_snapshot": client.snapshot(),
            "request_summary": source_local._request_summary(connection, used_request_ids),
            "candidate_map": {
                "path": candidate_path.name,
                "sha256": _sha256(candidate_path),
                "rows": len(candidate),
                "readout_bucket_count": int(candidate["readout_bucket_id"].nunique()),
            },
            "decision_ledger": {
                "path": decision_path.name,
                "sha256": _sha256(decision_path),
                "rows": len(decisions),
            },
        })
    except Exception:
        manifest["status"] = "incomplete"
        raise
    finally:
        connection.close()
        write_json_atomic(manifest_path, manifest)
    return manifest


def publish_repair(
    task: str, review_path: Path, *, depth_one: bool = False
) -> dict[str, Any]:
    spec = _configure(task, depth_one=depth_one)
    return source_local.publish_readouts(
        task,
        review_path,
        output=Path(spec["root"]),
        readout_version=str(spec["readout_version"]),
    )


def _render(template: Path, payload: Mapping[str, Any]) -> str:
    source = Environment(undefined=StrictUndefined, autoescape=False).from_string(
        template.read_text(encoding="utf-8")
    )
    return source.render(compact_payload_json=core._canonical_json(payload)).strip()


def build_rankings(task: str, *, depth_one: bool = False) -> dict[str, Any]:
    spec = _configure(task, depth_one=depth_one)
    output = Path(spec["ranking_root"])
    final_map = Path(spec["root"]) / "readout_bucket_map.parquet"
    readout_manifest = Path(spec["root"]) / "manifest.json"
    if output.exists() and any(output.iterdir()):
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("readout_map_sha256") != _sha256(final_map):
            raise ValueError("existing ranking build uses a different readout map")
        return manifest
    reviewed = json.loads(readout_manifest.read_text(encoding="utf-8"))
    if reviewed.get("status") != "complete_reviewed":
        raise ValueError("rankings require a reviewed readout map")

    atoms, _ = source_local._load_inputs(spec)
    lookup = core._atom_lookup(atoms)
    mapping = pd.read_parquet(final_map)
    semantic = pd.read_parquet(spec["semantic_map"])
    if mapping["atom_id"].duplicated().any() or set(mapping["atom_id"]) != set(semantic["atom_id"]):
        raise ValueError("readout map does not cover the frozen semantic map")

    scientific_fields = sorted({
        "source_name",
        "endpoint_name",
        "measurement_text",
        "unit_text",
        "finite_scalar_value",
        "canonical_measurement_text",
        "support_text",
        "confidence",
        "qualifying_conditions",
        "study_context",
        *(column for columns in core.SAMPLE_CARD_COLUMNS.values() for column in columns),
    })
    available_fields = set(core.pq.read_schema(core.RECORDS).names)
    scientific_fields = [field for field in scientific_fields if field in available_fields]
    record_map = pd.read_parquet(
        Path(spec["root"]) / "record_readout_bucket_map.parquet",
        columns=[
            "canonical_record_id",
            "source_row_uid",
            "atom_id",
            "readout_bucket_id",
        ],
    )
    records = pd.read_parquet(
        core.RECORDS,
        columns=["source_row_uid", *scientific_fields],
    ).drop_duplicates("source_row_uid")
    record_rows = record_map.merge(
        records, on="source_row_uid", how="left", validate="many_to_one"
    )
    record_examples: dict[str, list[dict[str, Any]]] = {}
    for bucket, rows in record_rows.groupby("readout_bucket_id", sort=False):
        ordered = rows.assign(
            _order=[
                core._stable_id("ranking-record", str(bucket), str(record_id))
                for record_id in rows["canonical_record_id"]
            ]
        ).sort_values(["_order", "canonical_record_id"])
        selected = pd.concat(
            [ordered.drop_duplicates("atom_id"), ordered], ignore_index=True
        ).drop_duplicates("source_row_uid").head(RANKING_RECORD_LIMIT)
        record_examples[str(bucket)] = [
            {
                field: value.item() if hasattr(value, "item") else value
                for field, value in row._asdict().items()
                if field in scientific_fields and pd.notna(value) and str(value).strip()
            }
            for row in selected.itertuples(index=False)
        ]

    output.mkdir(parents=True)
    connection = core._request_database(output / "requests.sqlite3")
    schedule: list[dict[str, Any]] = []
    parent_rows: list[dict[str, Any]] = []
    for parent_key, rows in mapping.groupby(
        ["level", "source_id", "semantic_bucket_id"], sort=True
    ):
        level, source_id, semantic_bucket_id = map(str, parent_key)
        buckets = {
            str(bucket): sorted(group["atom_id"].astype(str))
            for bucket, group in rows.groupby("readout_bucket_id", sort=True)
        }
        parent_rows.append({
            "level": level,
            "source_id": source_id,
            "semantic_bucket_id": semantic_bucket_id,
            "readout_bucket_count": len(buckets),
        })
        if len(buckets) == 1:
            continue
        payloads = {
            bucket: {
                "definition": {
                    column: sorted(
                        {str(lookup[atom]["values"][column]) for atom in members}
                    )
                    for column in core.PROMPT_DIMENSION_COLUMNS[source_id]
                },
                "representative_records": record_examples[bucket],
            }
            for bucket, members in buckets.items()
        }
        parent_atoms = sorted(rows["atom_id"].astype(str))
        parent_profile = {
            column: ranking._summarize(
                {str(lookup[atom]["values"][column]) for atom in parent_atoms}
            )
            for column in core.INITIAL_COLUMNS[source_id]
        }
        edges = core.degree_edges(list(buckets), level=semantic_bucket_id, degree=DEGREE)
        for index, (left, right) in enumerate(edges):
            aliases = {"A": left, "B": right}
            prompt = _render(
                Path(spec["template"]),
                {
                    "semantic_parent": {
                        "evidence_family": source_id,
                        "definition": parent_profile,
                    },
                    "candidates": {alias: payloads[bucket] for alias, bucket in aliases.items()},
                },
            )
            phase = f"within-semantic|{semantic_bucket_id}"
            request_id = core._request_id("ranking", phase, prompt)
            core._queue_request(
                connection,
                request_id=request_id,
                kind="ranking",
                phase=phase,
                prompt=prompt,
                reasoning_effort="low",
                max_tokens=core.LOW_MAX_TOKENS,
                validation={"candidate_bucket_ids": [*aliases, "TIE"]},
                commit=False,
            )
            schedule.append({
                "comparison_id": f"{semantic_bucket_id}-{index:05d}",
                "request_id": request_id,
                "level": level,
                "source_id": source_id,
                "semantic_bucket_id": semantic_bucket_id,
                "bucket_a_id": left,
                "bucket_b_id": right,
            })
        connection.commit()
    connection.close()
    schedule_frame = pd.DataFrame(schedule)
    parent_frame = pd.DataFrame(parent_rows)
    schedule_path = output / "comparison_schedule.parquet"
    parents_path = output / "parent_schedule.parquet"
    schedule_frame.to_parquet(schedule_path, index=False)
    parent_frame.to_parquet(parents_path, index=False)
    manifest = {
        "version": str(spec["readout_version"]) + ".degree15.v2",
        "status": "prepared",
        "task": spec["task"],
        "model": core.MODEL,
        "reasoning_effort": "low",
        "target_degree": DEGREE,
        "complete_graph_maximum_children": DEGREE + 1,
        "fit_scope": "semantic_parent",
        "ranking_target": "readout informativeness about the fixed semantic parent's biological meaning",
        "l2_penalty": L2_PENALTY,
        "representative_record_limit": RANKING_RECORD_LIMIT,
        "readout_map_sha256": _sha256(final_map),
        "semantic_map_sha256": _sha256(spec["semantic_map"]),
        "prompt_sha256": _sha256(spec["template"]),
        "semantic_parent_count": len(parent_frame),
        "singleton_parent_count": int((parent_frame["readout_bucket_count"] == 1).sum()),
        "comparison_count": len(schedule_frame),
        "artifacts": {
            schedule_path.name: _sha256(schedule_path),
            parents_path.name: _sha256(parents_path),
        },
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def run_rankings(
    task: str, *, parallelism: int, depth_one: bool = False
) -> dict[str, Any]:
    spec = _configure(task, depth_one=depth_one)
    output = Path(spec["ranking_root"])
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"prepared", "running", "incomplete"}:
        return manifest
    schedule = pd.read_parquet(output / "comparison_schedule.parquet")
    parents = pd.read_parquet(output / "parent_schedule.parquet")
    connection = core._request_database(output / "requests.sqlite3")
    maximum_context, endpoint_models = core._endpoint_contract()
    client, endpoint_receipts, total_parallelism = core._build_completion_pool(parallelism)
    manifest["status"] = "running"
    write_json_atomic(manifest_path, manifest)
    try:
        request_ids = schedule["request_id"].tolist()
        try:
            core._run_pending(
                connection,
                request_ids,
                parallelism=total_parallelism,
                client=client,
                request_slots=threading.BoundedSemaphore(total_parallelism),
            )
        except RuntimeError:
            ranking._adjudicate_unique_prefix_failures(connection, request_ids)
            if connection.execute(
                "SELECT COUNT(*) FROM requests WHERE status!='complete'"
            ).fetchone()[0]:
                raise
        requests = ranking._completed_request_lookup(connection, request_ids)
        comparisons = []
        for row in schedule.itertuples(index=False):
            request = requests[row.request_id]
            comparisons.append({
                **row._asdict(),
                "winner_bucket_id": {
                    "A": row.bucket_a_id,
                    "B": row.bucket_b_id,
                    "TIE": "TIE",
                }[json.loads(request["response_json"])["winner_bucket_id"]],
                "served_model": request["served_model"],
                "provider_name": request["provider_name"],
                "provider_base_url": request["provider_base_url"],
                "attempts": request["attempts"],
                "input_tokens": request["input_tokens"],
                "output_tokens": request["output_tokens"],
                "adjudication_fallback": request["error"],
            })
        comparison_frame = pd.DataFrame(comparisons)
        comparison_path = output / "comparisons.parquet"
        comparison_frame.to_parquet(comparison_path, index=False)

        mapping = pd.read_parquet(Path(spec["root"]) / "readout_bucket_map.parquet")
        ranking_rows = []
        for parent in parents.itertuples(index=False):
            child_ids = sorted(
                mapping.loc[
                    mapping["semantic_bucket_id"] == parent.semantic_bucket_id,
                    "readout_bucket_id",
                ].unique()
            )
            if len(child_ids) == 1:
                ranking_rows.append({
                    "task_id": spec["task"],
                    "level": parent.level,
                    "source_id": parent.source_id,
                    "semantic_bucket_id": parent.semantic_bucket_id,
                    "readout_bucket_id": child_ids[0],
                    "bradley_terry_score": 0.0,
                    "within_parent_rank": 1,
                    "within_parent_percentile": 100.0,
                    "comparison_count": 0,
                })
                continue
            rows = comparison_frame[
                comparison_frame["semantic_bucket_id"] == parent.semantic_bucket_id
            ]
            outcomes = list(
                rows[["bucket_a_id", "bucket_b_id", "winner_bucket_id"]]
                .itertuples(index=False, name=None)
            )
            scores = ranking._fit_bradley_terry(outcomes, l2_penalty=L2_PENALTY)
            counts = Counter(key for left, right, _ in outcomes for key in (left, right))
            ordered = sorted(scores, key=lambda key: (-scores[key], key))
            for index, bucket in enumerate(ordered):
                ranking_rows.append({
                    "task_id": spec["task"],
                    "level": parent.level,
                    "source_id": parent.source_id,
                    "semantic_bucket_id": parent.semantic_bucket_id,
                    "readout_bucket_id": bucket,
                    "bradley_terry_score": scores[bucket],
                    "within_parent_rank": index + 1,
                    "within_parent_percentile": 100 * (len(ordered) - 1 - index) / max(1, len(ordered) - 1),
                    "comparison_count": counts[bucket],
                })
        rankings = pd.DataFrame(ranking_rows)
        ranking_path = output / "readout_bucket_rankings.parquet"
        rankings.to_parquet(ranking_path, index=False)
        record_map = pd.read_parquet(Path(spec["root"]) / "record_readout_bucket_map.parquet")
        ranked_records = record_map.merge(
            rankings,
            on=["level", "source_id", "semantic_bucket_id", "readout_bucket_id"],
            how="left",
            validate="many_to_one",
        )
        if len(ranked_records) != core.EXPECTED_RECORD_COUNT or ranked_records["within_parent_rank"].isna().any():
            raise ValueError("within-semantic rankings do not cover every scoped record")
        records_path = output / "record_readout_rankings.parquet"
        ranked_records.to_parquet(records_path, index=False)
        manifest.update({
            "status": "complete",
            "completed_at": time.time(),
            "per_endpoint_parallelism": parallelism,
            "total_parallelism": total_parallelism,
            "max_model_len": maximum_context,
            "endpoint_models_at_start": endpoint_models,
            "endpoint_pool_at_start": endpoint_receipts,
            "endpoint_pool_final_snapshot": client.snapshot(),
            "artifacts": {
                **manifest["artifacts"],
                comparison_path.name: {"rows": len(comparison_frame), "sha256": _sha256(comparison_path)},
                ranking_path.name: {"rows": len(rankings), "sha256": _sha256(ranking_path)},
                records_path.name: {"rows": len(ranked_records), "sha256": _sha256(records_path)},
            },
            "adjudication_fallback_count": int(comparison_frame["adjudication_fallback"].notna().sum()),
        })
    except Exception:
        manifest["status"] = "incomplete"
        raise
    finally:
        connection.close()
        write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "audit-risk",
            "prepare-repair",
            "run-repair",
            "publish-repair",
            "build-rankings",
            "run-rankings",
        ),
    )
    parser.add_argument("--task", choices=sorted(source_local.TASK_ALIASES), required=True)
    parser.add_argument("--parallelism", type=int, default=256)
    parser.add_argument("--review-sha256")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--depth-one",
        action="store_true",
        help="rebuild every semantic parent from scratch with at most one split",
    )
    args = parser.parse_args()
    if args.command == "audit-risk":
        result = audit_risk(args.task)
    elif args.command == "prepare-repair":
        result = prepare_repair(
            args.task, pilot=args.pilot, depth_one=args.depth_one
        )
    elif args.command == "run-repair":
        if not args.review_sha256:
            parser.error("run-repair requires --review-sha256")
        result = run_repair(
            args.task,
            args.review_sha256,
            parallelism=args.parallelism,
            pilot=args.pilot,
            depth_one=args.depth_one,
        )
    elif args.command == "publish-repair":
        if args.review is None:
            parser.error("publish-repair requires --review")
        result = publish_repair(args.task, args.review, depth_one=args.depth_one)
    elif args.command == "build-rankings":
        result = build_rankings(args.task, depth_one=args.depth_one)
    else:
        result = run_rankings(
            args.task, parallelism=args.parallelism, depth_one=args.depth_one
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
