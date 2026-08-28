"""Batched informativeness judgments for collapsed non-voting evidence."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tools.chembl_tool.common.json_utils import parse_json_content
from tools.chembl_tool.common.starling.build_measurement_resolution_mapping import (
    DISTILLATION_CANDIDATES,
)
from tools.chembl_tool.common.starling.build_reference_semantics_mapping import (
    BUDGET_EXHAUSTED_EXIT_CODE,
    TokenLedger,
)
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256


VERSION = "collapsed_informativeness.v2"
CACHE_VERSION = "collapsed_informativeness_requests.v2"
TEMPLATE_PATH = (
    Path(__file__).parent
    / "prompt_templates/collapsed_informativeness_v2.jinja"
)
VALUES = {"informative", "uninformative"}
MAX_COMPLETION_TOKENS = 2_048
SEMANTIC_METHODS = {"semantic_support_passthrough", "semantic_llm"}
TASKS = ("bbb_martins", "bioavailability_ma", "skin_reaction")
DIRECT_ENDPOINT_LABELS = {
    "bbb_martins": "CNS access evidence",
    "bioavailability_ma": "oral bioavailability evidence",
    "skin_reaction": "skin sensitization or contact-allergy evidence",
}
DEFAULT_ROOTS = {
    task: Path(f"outputs/chembl_tool/tasks/{task}/evidence_library/starling_normalized_v7")
    for task in TASKS
}
PROMPT_METADATA_FIELDS = {
    "pmid",
    "doi",
    "extraction_id",
    "global_identifier",
    "paragraph_idx",
    "confidence",
    "smiles",
    "source_smiles",
    "source_id",
    "source_name",
    "source_record_id",
    "source_row_number",
    "source_index",
    "skin_source",
}


@dataclass(frozen=True)
class Batch:
    task_id: str
    items: tuple[dict[str, Any], ...]
    direct_label_definition: str

    @property
    def batch_id(self) -> str:
        payload = {
            "task_id": self.task_id,
            "views": [item["view_id"] for item in self.items],
            "payloads": [item["payload_sha256"] for item in self.items],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()


def is_target_record(record: Mapping[str, Any]) -> bool:
    return (
        str(record.get("retrieval_source_id") or "") != "direct_vote"
        and str(record.get("aggregation_status") or "")
        != "pending_semantic_aggregation"
    )


def scientific_source_columns(spec: Any) -> dict[str, tuple[str, ...]]:
    contract = spec.policy.record_contract
    if contract is None:
        raise ValueError("collapsed informativeness requires a record contract")
    output = {}
    for source_id, columns in spec.compact_profile.source_columns.items():
        source = contract.sources[source_id]
        excluded = PROMPT_METADATA_FIELDS | {
            source.endpoint_field,
            source.measurement_field,
            source.unit_field,
            source.smiles_field,
            source.structure_identity_field,
        }
        output[source_id] = tuple(
            field for field in columns if field and field not in excluded
        )
    return output


def build_views(
    collapsed: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    source_columns: Mapping[str, Sequence[str]],
    *,
    direct_label_definition: str,
    task_id: str | None = None,
) -> list[dict[str, Any]]:
    from tools.chembl_tool.common.starling.record_collapse import _semantic_record

    group_id = str(collapsed["collapse_group_key"])
    views = []
    if str(collapsed.get("aggregation_method")) not in SEMANTIC_METHODS:
        candidates = []
        for record in source_records:
            fields = [
                field
                for field in _semantic_record(
                    record,
                    source_columns.get(str(record.get("source_id") or ""), ()),
                )["fields"]
                if field["label"] != "Molecule"
            ]
            confidence = record.get("confidence")
            score = (
                len(fields),
                float(confidence) if _finite(confidence) else -math.inf,
            )
            candidates.append((score, str(record["canonical_record_id"]), fields))
        if not candidates:
            raise ValueError(f"collapse group {group_id} has no retained source row")
        best_score = max(score for score, _, _ in candidates)
        _, representative_id, representative_fields = min(
            (row for row in candidates if row[0] == best_score), key=lambda row: row[1]
        )
        views.append(
            _view(
                group_id,
                "representative",
                representative_fields,
                direct_label_definition,
                representative_record_id=representative_id,
            )
        )
    views.append(
        _view(
            group_id,
            "collapsed",
            _collapsed_fields(
                collapsed,
                endpoint_override=(
                    DIRECT_ENDPOINT_LABELS.get(str(task_id))
                    if collapsed.get("retrieval_source_id") == "direct_residual"
                    else None
                ),
            ),
            direct_label_definition,
        )
    )
    return views


def render_batch(batch: Batch) -> str:
    items = [
        {
            "id": str(index),
            "fields": item["fields"],
        }
        for index, item in enumerate(batch.items)
    ]
    return _template().render(
        direct_label_definition=batch.direct_label_definition,
        items=items,
    )


def validate_response(response: Any, size: int) -> list[dict[str, str]]:
    if not isinstance(response, Mapping) or set(response) != {"items"}:
        fields = sorted(response) if isinstance(response, Mapping) else type(response).__name__
        raise ValueError(
            "informativeness response must contain only items; "
            f"received {fields}"
        )
    items = response["items"]
    if not isinstance(items, list) or len(items) != size:
        raise ValueError("informativeness response has wrong item count")
    by_id: dict[str, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, Mapping) or set(item) != {"id", "informativeness"}:
            raise ValueError("informativeness item fields differ from contract")
        item_id = str(item["id"])
        if item_id in by_id or item_id not in {str(i) for i in range(size)}:
            raise ValueError("informativeness response has invalid item IDs")
        informativeness = str(item["informativeness"])
        if informativeness not in VALUES:
            raise ValueError("informativeness response has invalid flag")
        by_id[item_id] = {"informativeness": informativeness}
    return [{"id": str(i), **by_id[str(i)]} for i in range(size)]


def load_complete_judgments(
    directory: str | Path, *, task_id: str
) -> dict[str, dict[str, Any]]:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION or manifest.get("task_id") != task_id:
        raise ValueError("collapsed informativeness manifest mismatch")
    if manifest.get("status") != "complete":
        raise ValueError(f"collapsed informativeness is incomplete for {task_id}")
    requests = root / "requests.jsonl"
    if manifest.get("requests_sha256") != file_sha256(requests):
        raise ValueError("collapsed informativeness request hash mismatch")
    payload_path = root / "payloads.jsonl"
    if manifest.get("payloads_sha256") != file_sha256(payload_path):
        raise ValueError("collapsed informativeness payload hash mismatch")
    payloads = {
        row["view_id"]: row["payload_sha256"]
        for row in _read_jsonl(payload_path)
    }
    judgments = {
        view_id: row
        for view_id, row in _load_requests(
            requests, template_sha256=str(manifest["template_sha256"])
        ).items()
        if payloads.get(view_id) == row["payload_sha256"]
    }
    if len(judgments) != int(manifest["target_views"]):
        raise ValueError("collapsed informativeness coverage mismatch")
    return judgments


def build_inventories(*, rebuild: bool = False) -> dict[str, dict[str, Any]]:
    manifests = {}
    for task_id in TASKS:
        manifests[task_id] = _build_inventory(task_id, DEFAULT_ROOTS[task_id], rebuild)
    return manifests


def run_generation(args: argparse.Namespace) -> int:
    build_inventories(rebuild=args.rebuild_inventory)
    if args.prior_requests_root is not None:
        for task in TASKS:
            source = Path(args.prior_requests_root) / task / "informativeness_requests.jsonl"
            if source.is_file():
                seeded = seed_exact_requests(
                    DEFAULT_ROOTS[task] / "collapsed_informativeness_v2",
                    source,
                    model=args.model,
                )
                print(f"{task}: seeded {seeded:,} exact pilot view(s)", flush=True)
    _activate_api_key(args.api_key_env)
    llm = _distillation_llm()
    ledger = TokenLedger(
        Path(args.token_ledger),
        epoch=args.budget_epoch,
        start_new_epoch=args.start_new_budget_epoch,
        max_tokens=args.budget_max_tokens,
    )
    states = {
        task: _open_queue(task, DEFAULT_ROOTS[task] / "collapsed_informativeness_v2")
        for task in TASKS
    }
    for task, (connection, root, definition) in states.items():
        cached = _load_requests(
            root / "requests.jsonl", template_sha256=file_sha256(TEMPLATE_PATH)
        )
        models = {str(row.get("model") or "") for row in cached.values()}
        if models - {args.model}:
            raise ValueError(
                f"{task} informativeness cache model mismatch: {sorted(models)}"
            )
        connection.execute("UPDATE queue SET status=0 WHERE status != 1")
        connection.executemany(
            "UPDATE queue SET status=1 WHERE view_id=? AND payload_sha256=?",
            ((view_id, row["payload_sha256"]) for view_id, row in cached.items()),
        )
        connection.commit()
        print(f"{task}: {len(cached):,} cached", flush=True)

    task_index = 0
    failures = 0
    budget_exhausted = False
    finished = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures: dict[concurrent.futures.Future[dict[str, Any]], Batch] = {}
        while futures or not finished:
            while len(futures) < args.workers and not finished:
                batch = None
                for _ in TASKS:
                    task = TASKS[task_index % len(TASKS)]
                    task_index += 1
                    connection, _, definition = states[task]
                    items = _take_batch(connection, args.batch_size)
                    if items:
                        batch = Batch(task, tuple(items), definition)
                        break
                if batch is None:
                    finished = True
                    break
                future = executor.submit(
                    _query_batch,
                    batch,
                    llm=llm,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    ledger=ledger,
                    epoch=args.budget_epoch,
                    max_attempts=args.max_attempts,
                )
                futures[future] = batch
            if not futures:
                break
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                batch = futures.pop(future)
                connection, root, _ = states[batch.task_id]
                try:
                    event = future.result()
                except BudgetExhausted:
                    connection.executemany(
                        "UPDATE queue SET status=0 WHERE view_id=?",
                        ((item["view_id"],) for item in batch.items),
                    )
                    connection.commit()
                    budget_exhausted = True
                    finished = True
                    continue
                except Exception as exc:
                    connection.executemany(
                        "UPDATE queue SET status=-1 WHERE view_id=?",
                        ((item["view_id"],) for item in batch.items),
                    )
                    connection.commit()
                    failures += 1
                    print(f"{batch.task_id}: batch failed: {exc}", flush=True)
                    continue
                _append_jsonl(root / "requests.jsonl", event)
                connection.executemany(
                    "UPDATE queue SET status=1 WHERE view_id=?",
                    ((item["view_id"],) for item in batch.items),
                )
                connection.commit()
    if budget_exhausted:
        ledger.mark_exhausted()
    for task, (connection, root, _) in states.items():
        _write_progress(task, root, connection)
        connection.close()
    if budget_exhausted:
        return BUDGET_EXHAUSTED_EXIT_CODE
    return 1 if failures else 0


class BudgetExhausted(RuntimeError):
    pass


def _build_inventory(task_id: str, normalized_root: Path, rebuild: bool) -> dict[str, Any]:
    out = normalized_root / "collapsed_informativeness_v2"
    out.mkdir(parents=True, exist_ok=True)
    payloads = out / "payloads.jsonl"
    manifest_path = out / "manifest.json"
    stage05 = normalized_root / "05_deduplicated_records/records.parquet"
    duplicate_audit = normalized_root / "05_deduplicated_records/duplicates.parquet"
    stage06 = normalized_root / "06_collapsed_records/records.parquet"
    hashes = {
        "stage05_records_sha256": file_sha256(stage05),
        "stage05_duplicates_sha256": file_sha256(duplicate_audit),
        "stage06_records_sha256": file_sha256(stage06),
        "template_sha256": file_sha256(TEMPLATE_PATH),
    }
    if not rebuild and manifest_path.is_file() and payloads.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(prior.get(key) == value for key, value in hashes.items()):
            return prior
    spec = importlib.import_module(
        f"tools.chembl_tool.tasks.{task_id}.build_starling_downstream_artifacts"
    ).get_spec()
    source_columns = scientific_source_columns(spec)
    collapsed_columns = [
        "collapse_group_key",
        "retrieval_source_id",
        "aggregation_method",
        "aggregation_status",
        "canonical_endpoint_name",
        "display_measurement_text",
        "display_unit_text",
        "canonical_pair_fields_json",
        "source_record_count",
        "aggregate_min",
        "aggregate_max",
        "aggregate_q1",
        "aggregate_q3",
        "aggregate_counts_json",
        "canonical_source_record_ids_json",
    ]
    collapsed_rows = pq.read_table(stage06, columns=collapsed_columns).to_pylist()
    pending = [
        row
        for row in collapsed_rows
        if str(row.get("retrieval_source_id") or "") != "direct_vote"
        and str(row.get("aggregation_status") or "")
        == "pending_semantic_aggregation"
    ]
    if pending:
        raise ValueError(
            "collapsed informativeness requires completed semantic aggregation; "
            f"{len(pending)} group(s) are pending for {task_id}"
        )
    targets = {
        str(row["collapse_group_key"]): row
        for row in collapsed_rows
        if is_target_record(row)
    }
    discarded_record_ids = set(
        pq.read_table(
            duplicate_audit, columns=["discarded_canonical_record_id"]
        ).column(0).to_pylist()
    )
    record_to_group = {}
    for group_id, row in targets.items():
        if str(row.get("aggregation_method")) in SEMANTIC_METHODS:
            continue
        for record_id in json.loads(row["canonical_source_record_ids_json"]):
            if record_id not in discarded_record_ids:
                record_to_group[str(record_id)] = group_id
    available = set(pq.read_schema(stage05).names)
    record_columns = {
        "canonical_record_id",
        "source_id",
        "molecule_name",
        "canonical_endpoint_name",
        "canonical_measurement_text",
        "canonical_unit_text",
        "finite_scalar_value",
        "measurement_kind",
        "canonical_category_id",
        "canonical_category_rank",
        "confidence",
        *(field for columns in source_columns.values() for field in columns),
    }
    records_by_group: dict[str, list[dict[str, Any]]] = {}
    parquet = pq.ParquetFile(stage05)
    for batch in parquet.iter_batches(
        batch_size=10_000, columns=sorted(record_columns & available)
    ):
        for record in batch.to_pylist():
            group_id = record_to_group.get(str(record["canonical_record_id"]))
            if group_id:
                records_by_group.setdefault(group_id, []).append(record)
    definition = str(spec.direct_label_definition)
    with payloads.open("w", encoding="utf-8") as handle:
        target_views = 0
        for group_id in sorted(targets):
            views = build_views(
                targets[group_id],
                records_by_group.get(group_id, ()),
                source_columns,
                direct_label_definition=definition,
                task_id=task_id,
            )
            for item in views:
                item["prompt_bytes"] = sum(
                    len(str(field["value"]).encode("utf-8"))
                    for field in item["fields"]
                )
                handle.write(
                    json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                )
                target_views += 1
    requests = out / "requests.jsonl"
    if not requests.exists():
        requests.touch()
    manifest = {
        "version": VERSION,
        "task_id": task_id,
        "status": "partial",
        "target_groups": len(targets),
        "target_views": target_views,
        "completed_views": len(
            _load_requests(requests, template_sha256=hashes["template_sha256"])
        ),
        "direct_label_definition": definition,
        "payloads_sha256": file_sha256(payloads),
        **hashes,
        "model_view_counts": {},
    }
    _write_json(manifest_path, manifest)
    queue = out / "queue.sqlite"
    if queue.exists():
        queue.unlink()
    _initialize_queue(out)
    return manifest


def _initialize_queue(root: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(root / "queue.sqlite")
    # Responses and payloads are authoritative; the queue is rebuilt on resume.
    connection.execute("PRAGMA journal_mode=MEMORY")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS queue ("
        "view_id TEXT PRIMARY KEY, payload_sha256 TEXT NOT NULL, "
        "prompt_bytes INTEGER NOT NULL, payload TEXT NOT NULL, status INTEGER NOT NULL)"
    )
    if connection.execute("SELECT COUNT(*) FROM queue").fetchone()[0] == 0:
        rows = []
        with (root / "payloads.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                rows.append(
                    (
                        item["view_id"],
                        item["payload_sha256"],
                        int(item["prompt_bytes"]),
                        line.strip(),
                        0,
                    )
                )
                if len(rows) == 1_000:
                    connection.executemany("INSERT INTO queue VALUES (?, ?, ?, ?, ?)", rows)
                    rows.clear()
        if rows:
            connection.executemany("INSERT INTO queue VALUES (?, ?, ?, ?, ?)", rows)
        connection.commit()
    return connection


def _open_queue(task_id: str, root: Path) -> tuple[sqlite3.Connection, Path, str]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return _initialize_queue(root), root, str(manifest["direct_label_definition"])


def _take_batch(connection: sqlite3.Connection, size: int) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT payload FROM queue WHERE status=0 ORDER BY prompt_bytes, view_id LIMIT ?",
        (size,),
    ).fetchall()
    items = [json.loads(row[0]) for row in rows]
    connection.executemany(
        "UPDATE queue SET status=2 WHERE view_id=?",
        ((item["view_id"],) for item in items),
    )
    connection.commit()
    return items


def _query_batch(
    batch: Batch,
    *,
    llm: Any,
    model: str,
    reasoning_effort: str,
    ledger: TokenLedger,
    epoch: str,
    max_attempts: int,
) -> dict[str, Any]:
    prompt = render_batch(batch)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    usage = Counter()
    errors = []
    result = None
    validated = None
    for attempt in range(1, max_attempts + 1):
        request_id = f"{batch.batch_id}:{model}:{attempt}"
        reservation = len(prompt.encode("utf-8")) + 1_024 + MAX_COMPLETION_TOKENS
        if not ledger.reserve(request_id, reservation):
            raise BudgetExhausted(request_id)
        call_usage = None
        try:
            result = llm(
                prompt,
                model=model,
                max_tokens=MAX_COMPLETION_TOKENS,
                temperature=0.0,
                reasoning_effort=reasoning_effort,
                max_retries=1,
                verbose=False,
            )
            call_usage = _usage(result.get("usage"))
            parsed = parse_json_content(str(result.get("content") or ""))
            validated = validate_response(parsed, len(batch.items))
        except Exception as exc:
            errors.append(str(exc))
        finally:
            ledger.complete(request_id, call_usage)
        if validated is not None:
            if call_usage:
                usage.update(call_usage)
            break
        if call_usage:
            usage.update(call_usage)
    if validated is None or result is None:
        raise RuntimeError("; ".join(errors))
    items = []
    for source, response in zip(batch.items, validated, strict=True):
        items.append(
            {
                "view_id": source["view_id"],
                "group_id": source["group_id"],
                "view": source["view"],
                "payload_sha256": source["payload_sha256"],
                "representative_record_id": source.get("representative_record_id"),
                "informativeness": response["informativeness"],
            }
        )
    return {
        "cache_version": CACHE_VERSION,
        "task_id": batch.task_id,
        "batch_id": batch.batch_id,
        "model": model,
        "served_model": str(result.get("model") or ""),
        "budget_epoch": epoch,
        "prompt_sha256": prompt_hash,
        "template_sha256": file_sha256(TEMPLATE_PATH),
        "usage": dict(usage),
        "attempts": len(errors) + 1,
        "validation_errors": errors,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "items": items,
    }


def _load_requests(
    path: Path, *, template_sha256: str | None = None
) -> dict[str, dict[str, Any]]:
    output = {}
    if not path.is_file():
        return output
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("cache_version") != CACHE_VERSION:
                raise ValueError("collapsed informativeness cache version mismatch")
            if (
                template_sha256 is not None
                and event.get("template_sha256") != template_sha256
            ):
                continue
            for item in event.get("items") or ():
                view_id = str(item["view_id"])
                prior = output.get(view_id)
                row = {**item, "model": event.get("model")}
                if prior is not None and prior != row:
                    raise ValueError(f"conflicting informativeness cache row: {view_id}")
                output[view_id] = row
    return output


def seed_exact_requests(
    root: str | Path, prior_requests: str | Path, *, model: str
) -> int:
    """Append only model- and payload-exact pilot judgments to an inventory."""
    target = Path(root)
    current_payloads = {
        str(row["view_id"]): str(row["payload_sha256"])
        for row in _read_jsonl(target / "payloads.jsonl")
    }
    template_sha256 = file_sha256(TEMPLATE_PATH)
    requests = target / "requests.jsonl"
    existing = _load_requests(requests, template_sha256=template_sha256)
    seeded = 0
    for event in _read_jsonl(Path(prior_requests)):
        if (
            event.get("cache_version") != CACHE_VERSION
            or event.get("model") != model
            or event.get("template_sha256") != template_sha256
        ):
            continue
        items = []
        for item in event.get("items") or ():
            view_id = str(item.get("view_id") or "")
            if current_payloads.get(view_id) != item.get("payload_sha256"):
                continue
            prior = existing.get(view_id)
            row = {**item, "model": model}
            if prior is not None:
                if prior != row:
                    raise ValueError(f"conflicting informativeness cache row: {view_id}")
                continue
            existing[view_id] = row
            items.append(item)
        if not items:
            continue
        seeded_event = {
            **event,
            "batch_id": hashlib.sha256(
                json.dumps(
                    {
                        "source_batch_id": event.get("batch_id"),
                        "view_ids": [item["view_id"] for item in items],
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            "items": items,
            "seeded_from": str(prior_requests),
        }
        _append_jsonl(requests, seeded_event)
        seeded += len(items)
    return seeded


def _read_jsonl(path: Path) -> Any:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_progress(task_id: str, root: Path, connection: sqlite3.Connection) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requests = root / "requests.jsonl"
    current_payloads = dict(
        connection.execute("SELECT view_id, payload_sha256 FROM queue")
    )
    judgments = {
        view_id: row
        for view_id, row in _load_requests(
            requests, template_sha256=file_sha256(TEMPLATE_PATH)
        ).items()
        if current_payloads.get(view_id) == row["payload_sha256"]
    }
    counts = Counter(row["model"] for row in judgments.values())
    request_counts: Counter[str] = Counter()
    token_usage: dict[str, Counter[str]] = {}
    for event in _read_jsonl(requests):
        items = event.get("items") or ()
        if not items or any(
            current_payloads.get(str(item.get("view_id")))
            != item.get("payload_sha256")
            for item in items
        ):
            continue
        model = str(event.get("model") or "unknown")
        request_counts[model] += 1
        token_usage.setdefault(model, Counter()).update(event.get("usage") or {})
    total = connection.execute("SELECT COUNT(*) FROM queue").fetchone()[0]
    manifest.update(
        {
            "status": "complete" if len(judgments) == total else "partial",
            "target_views": total,
            "completed_views": len(judgments),
            "pending_views": total - len(judgments),
            "model_view_counts": dict(sorted(counts.items())),
            "model_request_counts": dict(sorted(request_counts.items())),
            "model_token_usage": {
                model: dict(sorted(usage.items()))
                for model, usage in sorted(token_usage.items())
            },
            "requests_sha256": file_sha256(requests),
        }
    )
    _write_json(manifest_path, manifest)


def _collapsed_fields(
    record: Mapping[str, Any], *, endpoint_override: str | None = None
) -> list[dict[str, str]]:
    fields = []
    for label, value in (
        ("Endpoint", endpoint_override or record.get("canonical_endpoint_name")),
        ("Result", record.get("display_measurement_text")),
        ("Unit", record.get("display_unit_text")),
    ):
        _add_field(fields, label, value)
    try:
        pair = json.loads(str(record.get("canonical_pair_fields_json") or "{}"))
    except json.JSONDecodeError:
        pair = {}
    for key, value in sorted(pair.items()):
        label = str(key).removeprefix("canonical_").replace("_", " ").capitalize()
        _add_field(fields, label, value)
    if (
        str(record.get("aggregation_status") or "") == "conflict"
        and record.get("aggregate_counts_json") not in {None, "", "{}"}
    ):
        counts = json.loads(str(record["aggregate_counts_json"]))
        if str(record.get("aggregation_method")) == "direct_binary_vote":
            counts = {
                "negative" if key == "0" else "positive": value
                for key, value in counts.items()
            }
        _add_field(
            fields,
            "Result distribution",
            "; ".join(f"{key}: {value}" for key, value in sorted(counts.items())),
        )
    _add_field(fields, "Source record count", record.get("source_record_count"))
    return fields


def _add_field(fields: list[dict[str, str]], label: str, value: Any) -> None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return
    text = str(value).strip()
    if text and text.casefold() not in {
        "none",
        "nan",
        "null",
        "unknown",
        "__unknown__",
        "not applicable",
        "not_applicable",
    }:
        fields.append({"label": label, "value": text})


def _view(
    group_id: str,
    view: str,
    fields: list[dict[str, str]],
    definition: str,
    *,
    representative_record_id: str | None = None,
) -> dict[str, Any]:
    item = {
        "view_id": f"{group_id}:{view}",
        "group_id": group_id,
        "view": view,
        "fields": fields,
        "representative_record_id": representative_record_id,
    }
    item["payload_sha256"] = _payload_sha256(item, definition)
    return item


def _payload_sha256(item: Mapping[str, Any], definition: str) -> str:
    payload = {
        "direct_label_definition": definition,
        "view_id": item["view_id"],
        "group_id": item["group_id"],
        "view": item["view"],
        "representative_record_id": item.get("representative_record_id"),
        "fields": item["fields"],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


@lru_cache(maxsize=1)
def _template() -> Any:
    environment = Environment(
        loader=FileSystemLoader(str(TEMPLATE_PATH.parent)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    return environment.get_template(TEMPLATE_PATH.name)


@lru_cache(maxsize=1)
def _distillation_llm() -> Any:
    from tools.chembl_tool.common.starling.build_measurement_resolution_mapping import (
        distillation_client,
    )

    return distillation_client()[0]


def _activate_api_key(name: str) -> None:
    secret = os.environ.get(name)
    if not secret:
        for root in DISTILLATION_CANDIDATES:
            path = root / "keys.py"
            if not path.is_file():
                continue
            spec = importlib.util.spec_from_file_location("_informativeness_keys", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            secret = getattr(module, name, None)
            if secret:
                break
    if not secret:
        raise RuntimeError(f"missing API key {name}")
    os.environ["OPENAI_API_KEY"] = str(secret)


def _usage(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, Mapping):
        return None
    return {
        "input_tokens": int(value.get("prompt_tokens") or value.get("input_tokens") or 0),
        "output_tokens": int(
            value.get("completion_tokens") or value.get("output_tokens") or 0
        ),
    }


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _write_json(path: Path, row: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--rebuild-inventory", action="store_true")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--prior-requests-root", type=Path)
    parser.add_argument("--token-ledger", type=Path)
    parser.add_argument("--budget-epoch", default="")
    parser.add_argument("--budget-max-tokens", type=int)
    parser.add_argument("--start-new-budget-epoch", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size != 20:
        parser.error("collapsed informativeness batch size is frozen at 20")
    if not args.inventory_only and (
        args.token_ledger is None or not args.budget_epoch or not args.budget_max_tokens
    ):
        parser.error("paid generation requires ledger, epoch, and token budget")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.inventory_only:
        manifests = build_inventories(rebuild=args.rebuild_inventory)
        print(
            json.dumps(
                {
                    task: {
                        "target_groups": manifest["target_groups"],
                        "target_views": manifest["target_views"],
                        "completed_views": manifest["completed_views"],
                    }
                    for task, manifest in manifests.items()
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return run_generation(args)


if __name__ == "__main__":
    raise SystemExit(main())
