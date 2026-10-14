"""Source-local semantic grouping with explicit task policies."""

from __future__ import annotations
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
import gzip
from jinja2 import Environment, StrictUndefined
import pandas as pd
from predict.utils.json import write_json_atomic
from .bucket_cards import (
    _column_selection_payload,
    _merge_bucket_payload,
    _value_payload,
)
from .inputs import _load_atoms
from .execution import _REQUEST_DATABASE_WRITE_LOCK
from .execution import _build_completion_pool
from .storage import _canonical_json
from .execution import _endpoint_contract
from .storage import _file_sha256
from .execution import _queue_checked_request
from .storage import _request_database
from .storage import _response
from .execution import _run_pending
from .storage import _seed_request_database
from .storage import _stable_id
from .execution import _token_count


def _render(
    name: str, payload: Mapping[str, Any], *, purpose: str = "semantic", policy: Mapping
) -> str:
    environment = Environment(undefined=StrictUndefined, autoescape=False)
    template = environment.from_string(
        (policy["PROMPT_ROOT"] / f"{name}.jinja").read_text(encoding="utf-8")
    )
    return template.render(
        purpose=purpose, payload=payload, compact_payload_json=_canonical_json(payload)
    ).strip()


def bucket_id(atom_ids: Iterable[str]) -> str:
    members = sorted(set(atom_ids))
    if not members:
        raise ValueError("a bucket must contain at least one atom")
    return _stable_id("sb", *members)


def deterministic_batches(
    item_ids: Sequence[str], *, seed: str, size: int | None = None, policy: Mapping
) -> list[list[str]]:
    if size is None:
        size = policy["MERGE_BATCH_SIZE"]
    if size < 2:
        raise ValueError("merge batch size must be at least two")
    ordered = sorted(
        set(item_ids), key=lambda item: (_stable_id("order", seed, item), item)
    )
    return [ordered[index : index + size] for index in range(0, len(ordered), size)]


def apply_merge_sets(
    items: Mapping[str, Sequence[str]], response: Mapping[str, Any]
) -> dict[str, list[str]]:
    """Merge disjoint item memberships; omitted item IDs remain singleton groups."""
    groups = validate_merge_sets(response, items)
    consumed = {item for group in groups for item in group}
    groups.extend([[item] for item in sorted(items) if item not in consumed])
    output: dict[str, list[str]] = {}
    for group in groups:
        atoms = sorted({atom for item in group for atom in items[item]})
        if sum((len(items[item]) for item in group)) != len(atoms):
            raise ValueError("merge inputs overlap in atom membership")
        output[bucket_id(atoms)] = atoms
    if Counter((atom for atoms in output.values() for atom in atoms)) != Counter(
        (atom for atoms in items.values() for atom in atoms)
    ):
        raise ValueError("merge changed atom coverage")
    return output


def apply_split_decisions(
    active: Mapping[str, Sequence[str]],
    decisions: Mapping[str, str],
    children: Mapping[str, Mapping[str, Sequence[str]]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return active SPLIT/DEFERRED branches and terminal KEEP branches."""
    if set(decisions) != set(active):
        raise ValueError("every active bucket needs exactly one split decision")
    next_active: dict[str, list[str]] = {}
    terminal: dict[str, list[str]] = {}
    for parent_id, atoms in active.items():
        decision = decisions[parent_id]
        if decision == "keep":
            terminal[parent_id] = list(atoms)
            if parent_id in children:
                raise ValueError("KEEP bucket unexpectedly has children")
            continue
        if decision == "defer":
            next_active[parent_id] = list(atoms)
            if parent_id in children:
                raise ValueError("DEFERRED bucket unexpectedly has children")
            continue
        if decision != "split":
            raise ValueError(f"invalid split decision: {decision}")
        partition = children.get(parent_id)
        if not partition or len(partition) < 2:
            raise ValueError("SPLIT bucket must produce at least two children")
        if Counter((atom for child in partition.values() for atom in child)) != Counter(
            atoms
        ):
            raise ValueError("split children do not partition their parent")
        overlap = sum((len(child) for child in partition.values())) != len(
            {atom for child in partition.values() for atom in child}
        )
        if overlap:
            raise ValueError("split children overlap")
        next_active.update({key: list(value) for key, value in partition.items()})
    return (next_active, terminal)


def reverse_collapsed_splits(
    decisions: dict[str, str],
    children: dict[str, dict[str, list[str]]],
    technical_reasons: dict[str, str],
) -> int:
    """Treat a high-reasoning one-child merge as a final KEEP adjudication."""
    collapsed = [bucket for bucket, groups in children.items() if len(groups) == 1]
    for bucket in collapsed:
        decisions[bucket] = "keep"
        technical_reasons[bucket] = "high_reasoning_merge_reversed_split"
        del children[bucket]
    return len(collapsed)


def initial_buckets(atoms: pd.DataFrame) -> dict[tuple[str, str], dict[str, list[str]]]:
    output: dict[tuple[str, str], dict[str, list[str]]] = {}
    for (level, source, node), rows in atoms.groupby(
        ["level", "source_id", "initial_node_key"], sort=True
    ):
        members = sorted(rows["atom_id"].tolist())
        output.setdefault((level, source), {})[bucket_id(members)] = members
    return output


def _atom_lookup(atoms: pd.DataFrame) -> dict[str, dict[str, Any]]:
    output = {}
    for row in atoms.to_dict("records"):
        row = dict(row)
        row["values"] = json.loads(row.pop("values_json"))
        output[row["atom_id"]] = row
    return output


def _write_gzip_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(
            value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    temporary.replace(path)


def _read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _save_state(path: Path, state: Mapping[str, Any]) -> None:
    write_json_atomic(path, dict(state))


def _initialize_semantic_run(
    output: Path,
    *,
    review_manifest_path: Path,
    approved_review_sha256: str,
    policy: Mapping,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
    actual_review_hash = _file_sha256(review_manifest_path)
    if actual_review_hash != approved_review_sha256:
        raise ValueError(
            f"approved review hash mismatch: {actual_review_hash} != {approved_review_sha256}"
        )
    review = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    if review.get("status") != "awaiting_user_prompt_approval":
        raise ValueError("review manifest is not an approval candidate")
    if review.get("construction_policy_sha256") != policy.get("POLICY_SHA256"):
        raise ValueError("construction policy changed; prepare a fresh run")
    if review.get("completion_requests_made") != 0:
        raise ValueError("review artifact unexpectedly contains completion calls")
    current_templates = {
        path.name: _file_sha256(path)
        for path in sorted(policy["PROMPT_ROOT"].glob("*.jinja"))
    }
    if review.get("prompt_template_hashes") != current_templates:
        raise ValueError("reviewed prompt templates changed")
    for field, path in (
        ("source_record_map", policy["RECORD_MAP"]),
        ("v10_records", policy["RECORDS"]),
    ):
        if review[field]["sha256"] != _file_sha256(path):
            raise ValueError(f"reviewed {field} input changed")
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "semantic_state.json"
    atoms_path = output / "input_atoms.parquet"
    samples_path = output / "sample_cards.json.gz"
    manifest_path = output / "manifest.json"
    if state_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("approved_prompt_review_sha256") != approved_review_sha256:
            raise ValueError("existing run uses a different prompt review")
        return (
            pd.read_parquet(atoms_path),
            _read_gzip_json(samples_path),
            json.loads(state_path.read_text(encoding="utf-8")),
        )
    if any(output.iterdir()):
        raise FileExistsError(f"new semantic output directory is not empty: {output}")
    atoms, sample_cards = _load_atoms(policy=policy)
    atoms.to_parquet(atoms_path, index=False)
    _write_gzip_json(samples_path, sample_cards)
    groups = {}
    for (level, source), buckets in initial_buckets(atoms).items():
        key = f"{level}|{source}"
        groups[key] = {
            "level": level,
            "source_id": source,
            "round": 0,
            "consumed_columns": list(policy["INITIAL_COLUMNS"][source]),
            "active": buckets,
            "terminal": {},
            "termination_reasons": {},
            "history": [],
            "finished": False,
            "final_buckets": None,
        }
    state = {
        "version": policy["VERSION"],
        "status": "running_semantic",
        "groups": groups,
    }
    _save_state(state_path, state)
    manifest = {
        "version": policy["VERSION"],
        "status": "running_semantic",
        "approved_prompt_review": str(review_manifest_path),
        "approved_prompt_review_sha256": approved_review_sha256,
        "model": policy["MODEL"],
        "base_url": policy["BASE_URL"],
        "request_timeout_s": policy["REQUEST_TIMEOUT_S"],
        "input_atoms": {"path": atoms_path.name, "rows": len(atoms)},
        "sample_cards": {"path": samples_path.name, "rows": len(sample_cards)},
        "requests": "requests.sqlite3",
    }
    write_json_atomic(manifest_path, manifest)
    return (atoms, sample_cards, state)


def _merge_bucket_round(
    connection: sqlite3.Connection,
    *,
    level: str,
    source: str,
    buckets: Mapping[str, Sequence[str]],
    lookup: Mapping[str, Mapping[str, Any]],
    sample_cards: Mapping[str, Mapping[str, Any]],
    phase: str,
    maximum_context: int,
    parallelism: int,
    client: Any,
    request_slots: threading.BoundedSemaphore,
    policy: Mapping,
) -> dict[str, list[str]]:
    if len(buckets) < 2:
        return {key: list(value) for key, value in buckets.items()}
    pending_batches = deterministic_batches(
        list(buckets), seed=f"{policy['BATCH_SEED_VERSION']}|{phase}", policy=policy
    )
    sendable: list[tuple[list[str], str]] = []
    while pending_batches:
        batch = pending_batches.pop(0)
        if len(batch) < 2:
            sendable.append((batch, ""))
            continue
        batch_buckets = {key: buckets[key] for key in batch}
        prompt = _render(
            "merge_buckets",
            _merge_bucket_payload(
                level, source, batch_buckets, lookup, sample_cards, policy=policy
            ),
            policy=policy,
        )
        if (
            _token_count(prompt, policy=policy) + policy["HIGH_MAX_TOKENS"]
            <= maximum_context
        ):
            sendable.append((batch, prompt))
            continue
        if len(batch) == 2:
            sendable.extend((([item], "") for item in batch))
            continue
        midpoint = len(batch) // 2
        pending_batches[0:0] = [batch[:midpoint], batch[midpoint:]]
    request_ids = []
    with _REQUEST_DATABASE_WRITE_LOCK:
        try:
            for batch_index, (batch, prompt) in enumerate(sendable):
                if len(batch) < 2:
                    continue
                request_id = _queue_checked_request(
                    connection,
                    kind="merge_buckets",
                    phase=f"{phase}|batch{batch_index:04d}",
                    prompt=prompt,
                    reasoning_effort="high",
                    max_tokens=policy["HIGH_MAX_TOKENS"],
                    validation={"valid_ids": batch},
                    maximum_context=maximum_context,
                    commit=False,
                    policy=policy,
                )
                if request_id is None:
                    raise AssertionError("prechecked merge prompt became oversized")
                request_ids.append(request_id)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    _run_pending(
        connection,
        request_ids,
        parallelism=parallelism,
        client=client,
        request_slots=request_slots,
        policy=policy,
    )
    output: dict[str, list[str]] = {}
    request_index = 0
    for batch, _ in sendable:
        batch_buckets = {key: buckets[key] for key in batch}
        if len(batch) < 2:
            merged = batch_buckets
        else:
            merged = apply_merge_sets(
                batch_buckets, _response(connection, request_ids[request_index])
            )
            request_index += 1
        overlap = set(output) & set(merged)
        if overlap:
            raise ValueError(
                f"merge round produced duplicate bucket IDs: {sorted(overlap)}"
            )
        output.update(merged)
    return output


def _run_source_semantic(
    connection: sqlite3.Connection,
    group: dict[str, Any],
    *,
    lookup: Mapping[str, Mapping[str, Any]],
    sample_cards: Mapping[str, Mapping[str, Any]],
    maximum_context: int,
    parallelism: int,
    client: Any,
    request_slots: threading.BoundedSemaphore,
    checkpoint: Callable[[], None],
    policy: Mapping,
) -> None:
    level = str(group["level"])
    source = str(group["source_id"])
    while not group["finished"] and group["round"] < policy["MAX_SOURCE_ROUNDS"]:
        active = {key: list(value) for key, value in group["active"].items()}
        if not active:
            group["finished"] = True
            break
        remaining = [
            column
            for column in policy["REFINEMENT_COLUMNS"][source]
            if column not in set(group["consumed_columns"])
        ]
        if not remaining:
            for bucket, members in active.items():
                group["terminal"][bucket] = members
                group["termination_reasons"][bucket] = "columns_exhausted"
            group["active"] = {}
            group["finished"] = True
            break
        round_number = int(group["round"]) + 1
        selector_payload = _column_selection_payload(
            level, source, active, lookup, group["consumed_columns"], policy=policy
        )
        selector_prompt = _render("select_column", selector_payload, policy=policy)
        selector_id = _queue_checked_request(
            connection,
            kind="select_column",
            phase=f"semantic|{level}|{source}|round{round_number}|select",
            prompt=selector_prompt,
            reasoning_effort="high",
            max_tokens=policy["LOW_MAX_TOKENS"],
            validation={"candidate_columns": remaining},
            maximum_context=maximum_context,
            policy=policy,
        )
        if selector_id is None:
            for bucket, members in active.items():
                group["terminal"][bucket] = members
                group["termination_reasons"][bucket] = (
                    "source_selector_prompt_exceeds_context"
                )
            group["active"] = {}
            group["finished"] = True
            group["history"].append(
                {
                    "round": round_number,
                    "selector_request_id": None,
                    "decision": "technical_stop",
                    "reason": "source_selector_prompt_exceeds_context",
                }
            )
            checkpoint()
            break
        _run_pending(
            connection,
            [selector_id],
            parallelism=1,
            client=client,
            request_slots=request_slots,
            policy=policy,
        )
        selector = _response(connection, selector_id)
        if selector["decision"] == "stop":
            for bucket, members in active.items():
                group["terminal"][bucket] = members
                group["termination_reasons"][bucket] = "source_selector_stop"
            group["active"] = {}
            group["finished"] = True
            group["history"].append(
                {
                    "round": round_number,
                    "selector_request_id": selector_id,
                    "decision": "stop",
                }
            )
            checkpoint()
            break
        column = str(selector["column"])
        context_columns = [*group["consumed_columns"], column]
        group["consumed_columns"].append(column)
        decisions: dict[str, str] = {}
        value_items: dict[str, dict[str, list[str]]] = {}
        split_request_ids: dict[str, str] = {}
        merge_prompts: dict[str, str] = {}
        technical_reasons: dict[str, str] = {}
        for bucket, members in sorted(active.items()):
            payload, items = _value_payload(
                bucket,
                members,
                column,
                lookup,
                context_columns=context_columns,
                policy=policy,
            )
            payload.update(
                {"task": policy["TASK_NAME"], "level": level, "source_id": source}
            )
            if len(items) < 2:
                decisions[bucket] = (
                    "defer" if policy["DEFER_TECHNICAL_BRANCHES"] else "keep"
                )
                technical_reasons[bucket] = "selected_column_has_one_value"
                continue
            split_prompt = _render("split_bucket", payload, policy=policy)
            merge_prompt = _render("merge_values", payload, policy=policy)
            if (
                _token_count(split_prompt, policy=policy) + policy["LOW_MAX_TOKENS"]
                > maximum_context
                or _token_count(merge_prompt, policy=policy) + policy["HIGH_MAX_TOKENS"]
                > maximum_context
            ):
                decisions[bucket] = (
                    "defer" if policy["DEFER_TECHNICAL_BRANCHES"] else "keep"
                )
                technical_reasons[bucket] = "full_value_prompt_exceeds_context"
                continue
            request_id = _queue_checked_request(
                connection,
                kind="split_bucket",
                phase=f"semantic|{level}|{source}|round{round_number}|split|{bucket}",
                prompt=split_prompt,
                reasoning_effort="high",
                max_tokens=policy["LOW_MAX_TOKENS"],
                validation={},
                maximum_context=maximum_context,
                policy=policy,
            )
            if request_id is None:
                raise AssertionError("prechecked split prompt became oversized")
            split_request_ids[bucket] = request_id
            value_items[bucket] = items
            merge_prompts[bucket] = merge_prompt
        _run_pending(
            connection,
            list(split_request_ids.values()),
            parallelism=parallelism,
            client=client,
            request_slots=request_slots,
            policy=policy,
        )
        merge_request_ids: dict[str, str] = {}
        for bucket, request_id in split_request_ids.items():
            decision = str(_response(connection, request_id)["decision"])
            decisions[bucket] = decision
            if decision != "split":
                continue
            merge_id = _queue_checked_request(
                connection,
                kind="merge_values",
                phase=f"semantic|{level}|{source}|round{round_number}|values|{bucket}",
                prompt=merge_prompts[bucket],
                reasoning_effort="high",
                max_tokens=policy["HIGH_MAX_TOKENS"],
                validation={
                    "valid_ids": sorted(value_items[bucket]),
                    "require_multiple_groups": True,
                },
                maximum_context=maximum_context,
                policy=policy,
            )
            if merge_id is None:
                raise AssertionError("prechecked value-merge prompt became oversized")
            merge_request_ids[bucket] = merge_id
        _run_pending(
            connection,
            list(merge_request_ids.values()),
            parallelism=parallelism,
            client=client,
            request_slots=request_slots,
            policy=policy,
        )
        children = {
            bucket: apply_merge_sets(
                value_items[bucket], _response(connection, request_id)
            )
            for bucket, request_id in merge_request_ids.items()
        }
        reverse_collapsed_splits(decisions, children, technical_reasons)
        next_active, newly_terminal = apply_split_decisions(active, decisions, children)
        deferred = {
            bucket: list(active[bucket])
            for bucket, decision in decisions.items()
            if decision == "defer"
        }
        new_children = {
            bucket: members
            for bucket, members in next_active.items()
            if bucket not in deferred
        }
        group["terminal"].update(newly_terminal)
        for bucket in newly_terminal:
            group["termination_reasons"][bucket] = technical_reasons.get(
                bucket, "deepseek_keep"
            )
        split_count = sum((decision == "split" for decision in decisions.values()))
        if new_children:
            for merge_round in (1, 2):
                new_children = _merge_bucket_round(
                    connection,
                    level=level,
                    source=source,
                    buckets=new_children,
                    lookup=lookup,
                    sample_cards=sample_cards,
                    phase=f"semantic|{level}|{source}|round{round_number}|new_children_merge{merge_round}",
                    maximum_context=maximum_context,
                    parallelism=parallelism,
                    client=client,
                    request_slots=request_slots,
                    policy=policy,
                )
        next_active = {**new_children, **deferred}
        group["round"] = round_number
        group["active"] = next_active
        group["history"].append(
            {
                "round": round_number,
                "column": column,
                "selector_request_id": selector_id,
                "input_bucket_count": len(active),
                "split_count": split_count,
                "keep_count": sum(
                    (decision == "keep" for decision in decisions.values())
                ),
                "deferred_count": len(deferred),
                "output_active_bucket_count": len(next_active),
                "technical_outcome_reasons": dict(Counter(technical_reasons.values())),
            }
        )
        if split_count == 0 and (not deferred):
            group["finished"] = True
        checkpoint()
    if group["active"]:
        reason = (
            "three_source_round_cap"
            if group["round"] >= policy["MAX_SOURCE_ROUNDS"]
            else "no_active_split"
        )
        if reason == "three_source_round_cap" and policy["DEFER_TECHNICAL_BRANCHES"]:
            group["blocked_reason"] = "three_source_round_cap_with_active_branches"
            checkpoint()
            raise RuntimeError(
                f"semantic {level}|{source} reached the three-round cap with {len(group['active'])} active branches"
            )
        for bucket, members in group["active"].items():
            group["terminal"][bucket] = members
            group["termination_reasons"][bucket] = reason
        group["active"] = {}
    leaves = {key: list(value) for key, value in group["terminal"].items()}
    for merge_round in (1, 2):
        leaves = _merge_bucket_round(
            connection,
            level=level,
            source=source,
            buckets=leaves,
            lookup=lookup,
            sample_cards=sample_cards,
            phase=f"semantic|{level}|{source}|final_all_leaves_merge{merge_round}",
            maximum_context=maximum_context,
            parallelism=parallelism,
            client=client,
            request_slots=request_slots,
            policy=policy,
        )
    group["final_buckets"] = leaves
    group["finished"] = True
    checkpoint()


def run_semantic(
    output: Path,
    *,
    review_manifest_path: Path,
    approved_review_sha256: str,
    parallelism: int,
    seed_request_cache: Path | None = None,
    policy: Mapping,
) -> dict[str, Any]:
    if parallelism < 1:
        raise ValueError("parallelism must be positive")
    maximum_context, endpoint_models = _endpoint_contract(policy=policy)
    atoms, sample_cards, state = _initialize_semantic_run(
        output,
        review_manifest_path=review_manifest_path,
        approved_review_sha256=approved_review_sha256,
        policy=policy,
    )
    lookup = _atom_lookup(atoms)
    state_path = output / "semantic_state.json"
    database_path = output / "requests.sqlite3"
    seed_receipt = _seed_request_database(
        database_path, seed_request_cache, policy=policy
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if seed_receipt is not None:
        manifest["seed_request_cache"] = seed_receipt
        write_json_atomic(manifest_path, manifest)
    else:
        seed_receipt = manifest.get("seed_request_cache")
    connection = _request_database(database_path)
    connection.close()
    if manifest.get("status") == policy["SOURCE_LOCAL_FINAL_STATUS"] and all(
        (group["finished"] for group in state["groups"].values())
    ):
        return manifest
    client, endpoint_receipts, total_parallelism = _build_completion_pool(
        parallelism, policy=policy
    )
    request_slots = threading.BoundedSemaphore(total_parallelism)
    state_lock = threading.Lock()

    def run_group(group_key: str) -> None:
        group = deepcopy(state["groups"][group_key])
        connection = _request_database(database_path)

        def checkpoint() -> None:
            with state_lock:
                state["groups"][group_key] = deepcopy(group)
                _save_state(state_path, state)

        try:
            print(
                f"semantic {group_key}: round={group['round']} active={len(group['active'])}",
                flush=True,
            )
            _run_source_semantic(
                connection,
                group,
                lookup=lookup,
                sample_cards=sample_cards,
                maximum_context=maximum_context,
                parallelism=total_parallelism,
                client=client,
                request_slots=request_slots,
                checkpoint=checkpoint,
                policy=policy,
            )
            print(
                f"semantic {group_key}: complete buckets={len(group['final_buckets'])}",
                flush=True,
            )
        finally:
            connection.close()

    pending_groups = [
        key
        for key in sorted(state["groups"])
        if state["groups"][key].get("final_buckets") is None
    ]
    with ThreadPoolExecutor(max_workers=len(pending_groups) or 1) as executor:
        futures = [executor.submit(run_group, key) for key in pending_groups]
        for future in as_completed(futures):
            future.result()
    rows = []
    for group in state["groups"].values():
        for source_bucket, atom_ids in group["final_buckets"].items():
            for atom_id in atom_ids:
                rows.append(
                    {
                        "level": group["level"],
                        "source_id": group["source_id"],
                        "source_semantic_bucket_id": source_bucket,
                        "atom_id": atom_id,
                    }
                )
    mapping = pd.DataFrame(rows).sort_values(
        ["level", "source_id", "source_semantic_bucket_id", "atom_id"]
    )
    if len(mapping) != len(atoms) or mapping["atom_id"].duplicated().any():
        raise ValueError(
            "final source-local mapping does not cover every atom exactly once"
        )
    mapping.to_parquet(output / "source_semantic_bucket_map.parquet", index=False)
    state["status"] = policy["SOURCE_LOCAL_FINAL_STATUS"]
    with state_lock:
        _save_state(state_path, state)
    connection = _request_database(output / "requests.sqlite3")
    request_summary = connection.execute("""
        SELECT count(*) AS requests,
               sum(status='complete') AS complete,
               sum(status='failed') AS failed,
               coalesce(sum(input_tokens),0) AS input_tokens,
               coalesce(sum(output_tokens),0) AS output_tokens
        FROM requests
        """).fetchone()
    connection.close()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": policy["SOURCE_LOCAL_FINAL_STATUS"],
            "endpoint_models_at_start": endpoint_models,
            "endpoint_pool_at_start": endpoint_receipts,
            "endpoint_pool_final_snapshot": client.snapshot(),
            "per_endpoint_parallelism": parallelism,
            "total_parallelism": total_parallelism,
            "max_model_len": maximum_context,
            "request_timeout_s": policy["REQUEST_TIMEOUT_S"],
            "seed_request_cache": seed_receipt,
            "source_semantic_bucket_map": {
                "path": "source_semantic_bucket_map.parquet",
                "rows": len(mapping),
                "source_bucket_count": int(
                    mapping["source_semantic_bucket_id"].nunique()
                ),
                "sha256": _file_sha256(output / "source_semantic_bucket_map.parquet"),
            },
            "request_summary": dict(request_summary),
        }
    )
    write_json_atomic(manifest_path, manifest)
    return manifest


def prepare_prompt_review(output: Path, *, policy: Mapping) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"prompt review directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    atoms, sample_cards = _load_atoms(policy=policy)
    lookup = _atom_lookup(atoms)
    buckets_by_group = initial_buckets(atoms)
    rendered: dict[str, str] = {}
    source_examples: dict[str, tuple[str, dict[str, list[str]]]] = {}
    for source in sorted(policy["PAIR_COLUMNS"]):
        candidates = [
            (level, buckets)
            for (level, candidate_source), buckets in buckets_by_group.items()
            if candidate_source == source
        ]
        level, buckets = max(candidates, key=lambda item: (len(item[1]), item[0]))
        source_examples[source] = (level, buckets)
        consumed = policy["INITIAL_COLUMNS"][source]
        selector = _column_selection_payload(
            level, source, buckets, lookup, consumed, policy=policy
        )
        rendered[f"select_column_{source}"] = _render(
            "select_column", selector, policy=policy
        )
        best: tuple[int, str, str] | None = None
        for bucket, members in buckets.items():
            for column in policy["REFINEMENT_COLUMNS"][source]:
                if column in consumed:
                    continue
                distinct = len({lookup[atom]["values"][column] for atom in members})
                if distinct > 1 and (best is None or distinct > best[0]):
                    best = (distinct, bucket, column)
        if best is None:
            bucket = next(iter(buckets))
            column = next(
                (
                    column
                    for column in policy["REFINEMENT_COLUMNS"][source]
                    if column not in consumed
                )
            )
        else:
            _, bucket, column = best
        values, _ = _value_payload(
            bucket,
            buckets[bucket],
            column,
            lookup,
            context_columns=consumed,
            policy=policy,
        )
        values.update(
            {"task": policy["TASK_NAME"], "level": level, "source_id": source}
        )
        rendered[f"split_bucket_{source}"] = _render(
            "split_bucket", values, policy=policy
        )
        rendered[f"merge_values_{source}"] = _render(
            "merge_values", values, policy=policy
        )
    merge_source = max(
        source_examples, key=lambda source: len(source_examples[source][1])
    )
    merge_level, merge_candidates = source_examples[merge_source]
    selected = dict(sorted(merge_candidates.items())[: policy["MERGE_BATCH_SIZE"]])
    rendered["merge_buckets_40"] = _render(
        "merge_buckets",
        _merge_bucket_payload(
            merge_level, merge_source, selected, lookup, sample_cards, policy=policy
        ),
        policy=policy,
    )
    maximum_tokens, endpoint_models = _endpoint_contract(policy=policy)
    prompt_rows = []
    for name, prompt in sorted(rendered.items()):
        path = output / f"{name}.txt"
        path.write_text(prompt + "\n", encoding="utf-8")
        input_tokens = _token_count(prompt, policy=policy)
        completion_reserve = (
            policy["HIGH_MAX_TOKENS"]
            if name.startswith("merge_")
            else policy["LOW_MAX_TOKENS"]
        )
        prompt_rows.append(
            {
                "name": name,
                "path": path.name,
                "sha256": _file_sha256(path),
                "input_tokens": input_tokens,
                "completion_token_reserve": completion_reserve,
                "fits_model_context": input_tokens + completion_reserve
                <= maximum_tokens,
            }
        )
    review_lines = [
        f"# {policy['PROMPT_REVIEW_TITLE']}",
        "",
        "No completion requests were made. Review and approve the prompt hashes in `manifest.json` before running DeepSeek.",
        "",
        "| Prompt | Input tokens | Reserved completion | Sendable |",
        "|---|---:|---:|:---:|",
    ]
    for row in prompt_rows:
        review_lines.append(
            f"| [{row['name']}]({row['path']}) | {row['input_tokens']:,} | {row['completion_token_reserve']:,} | {('yes' if row['fits_model_context'] else 'no')} |"
        )
    review_lines.extend(
        (
            "",
            "A non-sendable full-value prompt is retained to demonstrate the fail-closed size gate. That bucket/column combination must be treated as ineligible; it must not be truncated or sampled.",
            "",
        )
    )
    (output / "README.md").write_text("\n".join(review_lines), encoding="utf-8")
    atoms.to_parquet(output / "input_atoms.parquet", index=False)
    manifest = {
        "version": f"{policy['VERSION']}.prompt_review.v1",
        "status": "awaiting_user_prompt_approval",
        "completion_requests_made": 0,
        "model": policy["MODEL"],
        "base_url": policy["BASE_URL"],
        "endpoint_models": endpoint_models,
        "max_model_len": maximum_tokens,
        "source_record_map": {
            "path": str(policy["RECORD_MAP"]),
            "sha256": _file_sha256(policy["RECORD_MAP"]),
        },
        "v10_records": {
            "path": str(policy["RECORDS"]),
            "sha256": _file_sha256(policy["RECORDS"]),
        },
        "counts": {
            "records_level_scoped": policy["EXPECTED_RECORD_COUNT"],
            "level_source_pair_bucket_atoms": len(atoms),
            "initial_level_scoped_buckets": sum(
                (len(buckets) for buckets in buckets_by_group.values())
            ),
        },
        "settings": {
            "max_source_rounds": policy["MAX_SOURCE_ROUNDS"],
            "merge_batch_size": policy["MERGE_BATCH_SIZE"],
            "merge_rounds_per_operation": 2,
            "sample_limit": policy["SAMPLE_LIMIT"],
            "column_selection_reasoning": "high",
            "split_reasoning": "high",
            "value_and_bucket_merge_reasoning": "high",
            "ranking_reasoning": "high",
            "keep_is_terminal": True,
            "technical_noop_policy": "defer"
            if policy["DEFER_TECHNICAL_BRANCHES"]
            else "keep",
            "batch_seed_version": policy["BATCH_SEED_VERSION"],
            "refinement_columns": {
                source: list(columns)
                for source, columns in sorted(policy["REFINEMENT_COLUMNS"].items())
            },
            "ignored_pair_bucket_columns": {
                source: sorted(set(policy["PAIR_COLUMNS"][source]) - set(columns))
                for source, columns in sorted(policy["REFINEMENT_COLUMNS"].items())
            },
            "same_parent_recombination_allowed": True,
            "readout_maximum_depth": 2,
            "readout_coherence_is_intrinsic": True,
            "readout_keep_is_terminal": True,
            "readout_cross_bucket_merge": False,
        },
        "prompts": prompt_rows,
        "non_sendable_prompts": [
            row["name"] for row in prompt_rows if not row["fits_model_context"]
        ],
        "oversized_prompt_policy": "mark_bucket_column_ineligible_without_a_completion_request",
        "review_index_sha256": _file_sha256(output / "README.md"),
        "prompt_template_hashes": {
            path.name: _file_sha256(path)
            for path in sorted(policy["PROMPT_ROOT"].glob("*.jinja"))
        },
    }
    manifest["construction_policy_sha256"] = policy["POLICY_SHA256"]
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def validate_merge_sets(
    response: Mapping[str, Any], valid_ids: Iterable[str]
) -> list[list[str]]:
    valid = set(valid_ids)
    merge_sets = response.get("merge_sets")
    if not isinstance(merge_sets, list):
        raise ValueError("merge_sets must be a list")
    seen: set[str] = set()
    output: list[list[str]] = []
    for index, entry in enumerate(merge_sets):
        if not isinstance(entry, Mapping):
            raise ValueError(f"merge_sets[{index}] must be an object")
        members = entry.get("member_ids")
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError(
                f"merge_sets[{index}].member_ids must contain at least two IDs"
            )
        normalized = [str(member) for member in members]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"merge_sets[{index}] repeats an ID")
        unknown = set(normalized) - valid
        if unknown:
            raise ValueError(f"merge_sets[{index}] has unknown IDs: {sorted(unknown)}")
        overlap = set(normalized) & seen
        if overlap:
            raise ValueError(f"IDs occur in multiple merge sets: {sorted(overlap)}")
        seen.update(normalized)
        output.append(sorted(normalized))
    return output


def _validate_decision_response(
    kind: str, result: dict[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    decisions = {
        "select_column": {"stop", "select"},
        "split_bucket": {"keep", "split"},
        "readout_coherence": {"coherent", "split"},
        "retrieval_eligibility": {"include", "exclude"},
    }
    decision = result.get("decision")
    if decision not in decisions[kind]:
        raise ValueError(f"invalid {kind} decision")
    expected = {"decision", "rationale"}
    if kind == "retrieval_eligibility":
        expected.add("reason_code")
        reason = result.get("reason_code")
        if reason not in validation["reason_codes"]:
            raise ValueError("retrieval eligibility has an unknown reason code")
        informative = reason == "potentially_informative"
        if informative != (decision == "include"):
            raise ValueError("retrieval eligibility decision and reason disagree")
    elif decision in {"select", "split"} and kind != "split_bucket":
        if result.get("column") not in validation["candidate_columns"]:
            raise ValueError("selected column was not supplied")
        expected.add("column")
    if set(result) != expected:
        raise ValueError(f"{kind} response has unexpected fields")
    return result


def _validate_merge_response(
    kind: str, result: dict[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    if set(result) != {"merge_sets", "rationale"}:
        raise ValueError("merge response has unexpected fields")
    merge_sets = validate_merge_sets(result, validation["valid_ids"])
    if validation.get("require_multiple_groups"):
        merged = {item for group in merge_sets for item in group}
        if len(merge_sets) + len(set(validation["valid_ids"]) - merged) < 2:
            raise ValueError("binding split must retain at least two groups")
    for index, entry in enumerate(result["merge_sets"]):
        if set(entry) != {"member_ids", "label", "rationale"}:
            raise ValueError(f"merge_sets[{index}] has unexpected fields")
        if not all(
            (
                isinstance(entry[field], str) and entry[field].strip()
                for field in ("label", "rationale")
            )
        ):
            raise ValueError(f"merge_sets[{index}] needs label and rationale")
    if kind == "cross_source_merge" and len(result["merge_sets"]) > 1:
        raise ValueError("cross-source response may contain at most one merge set")
    if kind == "cross_source_merge" and result["merge_sets"]:
        members = result["merge_sets"][0]["member_ids"]
        if validation["anchor_bucket_id"] not in members:
            raise ValueError("cross-source merge must include the anchor bucket")
        if len({validation["source_by_id"][member] for member in members}) < 2:
            raise ValueError("cross-source merge must span at least two sources")
    return result


_REQUEST_DATABASE_WRITE_LOCK = threading.RLock()
