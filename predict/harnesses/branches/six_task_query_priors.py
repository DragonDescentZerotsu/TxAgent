"""Generate hash-pinned full-split query-prior overlays for six tasks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from typing import Any

from data.processing.gold_labels.conditioned_benchmark import split_path
from predict.api_client.pool import (
    load_provider_pool_config,
    primary_capacity,
    select_healthy_providers,
)
from predict.harnesses.branches.matrix import provider_client, sample_provider_loads
from predict.harnesses.branches.flat import (
    CONTEXT_V5_SIX_TASKS_PROMPT_VERSION,
    prompt_assets,
)
from predict.harnesses.progressive.state import ProgressiveTaskContract
from predict.tools.client import ToolServiceClient
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic
from tools.chembl_tool.common.conditioned_query_prior import build_query_prior_messages


TASKS = (
    "bbb_martins", "bioavailability_ma", "skin_reaction",
    "ames", "dili", "carcinogens",
)
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REQUIRED_CONTENT = {
    "endpoint_prior", "confidence", "reasoning_summary", "property_drivers", "caveats",
}


def _contract(task: str) -> Any:
    row = prompt_assets(CONTEXT_V5_SIX_TASKS_PROMPT_VERSION)["tasks"][task]
    return ProgressiveTaskContract(
        task=task,
        endpoint_name=row["endpoint_name"],
        label_scope=row["label_scope"],
        prediction_field=row["prediction_field"],
        positive_prediction=row["positive_prediction"],
        negative_prediction=row["negative_prediction"],
        system_role=row["system_role"],
        task_instructions=tuple(row["task_instructions"]),
    )


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _inputs(task: str, subset: str) -> tuple[Path, list[tuple[int, dict[str, Any]]]]:
    path = split_path(task, subset).with_name(
        f"{subset}_molecule_condition_labels.jsonl"
    ).resolve()
    rows = read_jsonl(path)
    if len({str(row["benchmark_row_id"]) for row in rows}) != len(rows):
        raise ValueError(f"Conditioned benchmark repeats query IDs: {path}")
    return path, list(enumerate(rows))


def _prior_source(run_dir: Path) -> dict[str, Any] | None:
    retrieval = run_dir / "retrieval.json"
    single = run_dir / "single_molecule_reasoning_output.json"
    if not retrieval.is_file() or not single.is_file():
        return None
    result = json.loads(single.read_text(encoding="utf-8"))
    content = ((result.get("llm") or {}).get("content"))
    if result.get("status") != "ok" or not isinstance(content, dict) or not REQUIRED_CONTENT <= content.keys():
        return None
    manifest = run_dir / "manifest.json"
    if not manifest.is_file() or not json.loads(manifest.read_text()).get("task_prompt_profile"):
        manifest = run_dir.parent.parent / "manifest.json"
    return {
        "run_dir": str(run_dir.resolve()),
        "files_sha256": {
            retrieval.name: sha256_file(retrieval), single.name: sha256_file(single),
        },
        "prompt_profile_manifest": str(manifest.resolve()),
        "prompt_profile_manifest_sha256": sha256_file(manifest),
    }


def _reusable_sources(root: Path | None, task: str) -> dict[str, dict[str, Any]]:
    if root is None:
        return {}
    batch = root.resolve() / task / f"{task}__none"
    output = {}
    for manifest_path in sorted((batch / "runs").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        query_id = str(manifest.get("benchmark_row_id") or "")
        source = _prior_source(manifest_path.parent)
        if not query_id or source is None or query_id in output:
            raise ValueError(f"Invalid reusable query-prior source: {manifest_path}")
        output[query_id] = source
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subset", choices=("valid", "test"), required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--reuse-root", type=Path)
    parser.add_argument(
        "--provider-pool-config", type=Path,
        default=Path("predict/api_client/providers/current_endpoints.json"),
    )
    parser.add_argument("--tool-service-url", default="http://127.0.0.1:8765")
    parser.add_argument("--parallelism", type=int)
    parser.add_argument("--max-tokens", type=int, default=20_480)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    root = args.output_root.resolve()
    if root.exists() and not args.resume:
        raise FileExistsError(f"Refusing to replace query-prior run: {root}")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    os.environ.setdefault("DEEPSEEK_API_KEY", "EMPTY")
    config_path = args.provider_pool_config.resolve()
    candidate = load_provider_pool_config(config_path)
    requested = args.parallelism or primary_capacity(candidate)
    selection = select_healthy_providers(candidate, requested)
    loads = sample_provider_loads(selection.config, samples=1, interval_s=0)
    root.mkdir(parents=True, exist_ok=args.resume)
    write_json_atomic(root / "execution.json", {
        "schema_version": "six_task_query_priors.v3",
        "status": "running",
        "host": socket.gethostname(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tasks": list(args.tasks),
        "subset": args.subset,
        "model": MODEL,
        "reasoning_effort": "high",
        "thinking": True,
        "max_tokens": args.max_tokens,
        "timeout_s": args.timeout_s,
        "provider_pool_config": str(config_path),
        "provider_pool_config_sha256": sha256_file(config_path),
        "provider_selection": selection.public_dict(),
        "observed_loads": loads,
    })
    tool = ToolServiceClient(args.tool_service_url, timeout_s=args.timeout_s)
    work = []
    sources: dict[tuple[str, int], dict[str, Any]] = {}
    task_inputs = {}
    for task in args.tasks:
        path, rows = _inputs(task, args.subset)
        task_inputs[task] = {"path": str(path), "sha256": sha256_file(path)}
        contract = _contract(task)
        reusable = _reusable_sources(args.reuse_root, task)
        for index, row in rows:
            query_id = str(row["benchmark_row_id"])
            if query_id in reusable:
                retrieval = json.loads(
                    (Path(reusable[query_id]["run_dir"]) / "retrieval.json").read_text()
                )
                if str((retrieval.get("query") or {}).get("input_smiles") or "") != str(row["drug"]):
                    raise ValueError(f"Reusable query prior differs for {task}/{query_id}")
                sources[task, index] = {**reusable[query_id], "source": "reused"}
                continue
            prior_path = (
                root / task / f"{task}__none" / "runs"
                / f"{task}__none_idx{index:05d}"
                / "single_molecule_reasoning_output.json"
            )
            if args.resume and prior_path.is_file():
                existing = json.loads(prior_path.read_text(encoding="utf-8"))
                content = ((existing.get("llm") or {}).get("content"))
                if existing.get("status") == "ok" and isinstance(content, dict) and REQUIRED_CONTENT <= content.keys():
                    source = _prior_source(prior_path.parent)
                    if source is not None:
                        sources[task, index] = {**source, "source": "fresh"}
                        continue
            properties = tool.invoke(
                "molecule_properties", {"query_smiles": str(row["drug"]), "logd_ph": 7.4}
            )
            if properties.get("status") != "ok":
                raise RuntimeError(f"molecule_properties failed for {task}/{index}: {properties}")
            query = {
                "input_smiles": str(row["drug"]),
                "canonical_smiles": str(row["drug"]),
                "tools_prefetched": True,
                "prefetched_molecule_properties": properties,
            }
            work.append((task, index, row, properties, build_query_prior_messages(contract, query)))
    client = provider_client(
        selection.effective_parallelism,
        provider_pool_config=config_path,
        config=selection.config,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
    )

    def run(item: tuple[Any, ...]) -> tuple[Any, ...]:
        task, index, row, properties, messages = item
        response = client.chat_json(messages, max_tokens=args.max_tokens)
        content = response.get("content")
        if not isinstance(content, dict) or not REQUIRED_CONTENT <= content.keys():
            raise ValueError(f"Invalid query prior for {task}/{index}")
        return task, index, row, properties, messages, response

    failures: list[str] = []
    pending = work
    for attempt in range(1, args.max_attempts + 1):
        if not pending:
            break
        retry = []
        with ThreadPoolExecutor(max_workers=min(selection.effective_parallelism, len(pending))) as executor:
            futures = {executor.submit(run, item): item for item in pending}
            for future in as_completed(futures):
                try:
                    task, index, row, properties, messages, response = future.result()
                except Exception as exc:
                    retry.append(futures[future])
                    failures.append(f"attempt={attempt}: {type(exc).__name__}: {exc}")
                    continue
                batch = root / task / f"{task}__none"
                run_dir = batch / "runs" / f"{task}__none_idx{index:05d}"
                retrieval = {
                    "status": "ok", "evidence_source": {"type": "none"},
                    "experiment": {"mode": "none", "source": "none", "resolved_group_mapping": {}},
                    "query": {"input_smiles": row["drug"], "canonical_smiles": row["drug"]},
                    "groups": [],
                }
                single = {
                    "analysis_id": "single_molecule", "status": "ok",
                    "llm": {**response, "messages": messages, "tool_results": [properties]},
                }
                write_json_atomic(run_dir / "retrieval.json", retrieval)
                write_json_atomic(run_dir / "single_molecule_reasoning_output.json", single)
                write_json_atomic(run_dir / "manifest.json", {
                    "schema_version": "six_task_query_prior_item.v1", "status": "complete",
                    "task_id": task, "canonical_split_index": index,
                    "benchmark_row_id": row["benchmark_row_id"],
                    "reasoning_effort": "high", "thinking": {"type": "enabled"},
                    "messages_sha256": _canonical_hash(messages),
                    "task_prompt_profile": prompt_assets(
                        CONTEXT_V5_SIX_TASKS_PROMPT_VERSION
                    )["tasks"][task]["prompt_profile"],
                })
                sources[task, index] = {**_prior_source(run_dir), "source": "fresh"}
        pending = retry
    if pending:
        execution = json.loads((root / "execution.json").read_text())
        execution.update(status="failed", completed_requests=len(sources), failures=failures)
        write_json_atomic(root / "execution.json", execution)
        raise RuntimeError(f"Query-prior recovery exhausted for {len(pending)} items")
    for task in args.tasks:
        batch = root / task / f"{task}__none"
        path, rows = _inputs(task, args.subset)
        entries = []
        for index, row in rows:
            source = sources.get((task, index))
            if source is None:
                raise ValueError(f"Missing query prior for {task}/{index}")
            entries.append({
                "target_index": index,
                "benchmark_row_id": row["benchmark_row_id"],
                "molecule_identity_key": str(row.get("molecule_identity_key") or ""),
                "condition_group": str(row.get("condition_group") or ""),
                "task_prompt_profile": prompt_assets(
                    CONTEXT_V5_SIX_TASKS_PROMPT_VERSION
                )["tasks"][task]["prompt_profile"],
                **source,
            })
        write_json_atomic(batch / "manifest.json", {
            "schema_version": "branch_query_prior_overlay.v1", "status": "complete",
            "task_id": task, "subset": args.subset, "n_items": len(entries),
            "input_jsonl": str(path), "input_sha256": sha256_file(path),
            "input": task_inputs[task], "model": MODEL, "reasoning_effort": "high",
            "task_prompt_profile": prompt_assets(
                CONTEXT_V5_SIX_TASKS_PROMPT_VERSION
            )["tasks"][task]["prompt_profile"],
            "reused": sum(row["source"] == "reused" for row in entries),
            "fresh": sum(row["source"] == "fresh" for row in entries),
            "sources": entries,
        })
    execution = json.loads((root / "execution.json").read_text())
    execution.update(
        status="complete", completed_requests=len(sources), fresh_requests=len(work),
        reused_requests=len(sources) - len(work),
        recovery_failures=failures, provider_snapshot=client.snapshot(),
    )
    write_json_atomic(root / "execution.json", execution)
    print(json.dumps(execution, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
