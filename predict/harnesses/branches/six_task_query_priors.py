"""Generate hash-pinned full-split query-prior overlays for six tasks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any

from data.processing.gold_labels.conditioned_benchmark import split_path, tdc_split_path
from data.processing.llm_api import provider_from_base_url
from predict.api_client.pool import (
    load_provider_pool_config,
    primary_capacity,
    select_healthy_providers,
)
from predict.harnesses.branches.matrix import provider_client, sample_provider_loads
from predict.harnesses.branches.flat import (
    CONTEXT_V5_SIX_TASKS_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
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
REQUIRED_CONTENT = {
    "endpoint_prior", "confidence", "reasoning_summary", "property_drivers", "caveats",
}
NATIVE_REQUIRED_CONTENT = {
    "bbb_martins": {"passive_bbb_plausibility", "efflux_or_transporter_prior"},
    "bioavailability_ma": {"oral_bioavailability_prior", "absorption_prior"},
}


def _required_content(task: str, native: bool) -> set[str]:
    if not native:
        return REQUIRED_CONTENT
    if task not in NATIVE_REQUIRED_CONTENT:
        raise ValueError(f"No native query-prior contract for {task}")
    return {"confidence", "reasoning_summary", "property_drivers", "caveats"} | NATIVE_REQUIRED_CONTENT[task]


def _contract(task: str, prompt_version: str) -> Any:
    row = prompt_assets(prompt_version)["tasks"][task]
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


def _inputs(
    task: str, subset: str, benchmark: str = "gold",
) -> tuple[Path, list[tuple[int, dict[str, Any]]]]:
    split = tdc_split_path if benchmark == "tdc" else split_path
    path = split(task, subset).with_name(
        f"{subset}_molecule_condition_labels.jsonl"
    ).resolve()
    rows = read_jsonl(path)
    if len({str(row["benchmark_row_id"]) for row in rows}) != len(rows):
        raise ValueError(f"Conditioned benchmark repeats query IDs: {path}")
    return path, list(enumerate(rows))


def _prior_source(run_dir: Path, required: set[str] = REQUIRED_CONTENT) -> dict[str, Any] | None:
    retrieval = run_dir / "retrieval.json"
    single = run_dir / "single_molecule_reasoning_output.json"
    if not retrieval.is_file() or not single.is_file():
        return None
    result = json.loads(single.read_text(encoding="utf-8"))
    content = ((result.get("llm") or {}).get("content"))
    if result.get("status") != "ok" or not isinstance(content, dict) or not required <= content.keys():
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


def _reusable_sources(
    root: Path | None, task: str, rows: list[tuple[int, dict[str, Any]]],
    required: set[str],
) -> dict[str, dict[str, Any]]:
    if root is None:
        return {}
    batch = root.resolve() / task / f"{task}__none"
    overlay_path = batch / "manifest.json"
    if overlay_path.is_file():
        overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
        if overlay.get("schema_version") == "branch_query_prior_overlay.v1":
            output = {}
            for row in overlay.get("sources") or []:
                index = row.get("target_index")
                target = rows[index][1] if isinstance(index, int) and 0 <= index < len(rows) else None
                if target is not None and (
                    str(row.get("molecule_identity_key") or "") != str(target.get("molecule_identity_key") or "")
                    or str(row.get("condition_group") or "") != str(target.get("condition_group") or "")
                ):
                    raise ValueError(f"Reusable query-prior identity differs at {task}/{index}")
                query_id = str(row.get("benchmark_row_id") or (target or {}).get("benchmark_row_id") or "")
                source = _prior_source(Path(str(row.get("run_dir") or "")), required)
                if not query_id or source is None or query_id in output:
                    raise ValueError(f"Invalid reusable query-prior overlay: {overlay_path}")
                output[query_id] = source
            if len(output) != int(overlay.get("n_items", -1)):
                raise ValueError(f"Incomplete reusable query-prior overlay: {overlay_path}")
            return output
    output = {}
    for manifest_path in sorted((batch / "runs").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        query_id = str(manifest.get("benchmark_row_id") or "")
        source = _prior_source(manifest_path.parent, required)
        if not query_id or source is None or query_id in output:
            raise ValueError(f"Invalid reusable query-prior source: {manifest_path}")
        output[query_id] = source
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subset", choices=("valid", "valid_small", "test"), required=True)
    parser.add_argument("--benchmark", choices=("gold", "tdc"), default="gold")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--reuse-root", type=Path)
    parser.add_argument("--reuse-identity-root", type=Path, action="append", default=[])
    parser.add_argument("--reuse-tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument(
        "--prompt-version",
        choices=(
            CONTEXT_V5_SIX_TASKS_PROMPT_VERSION,
            CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
        ),
        default=CONTEXT_V5_SIX_TASKS_PROMPT_VERSION,
    )
    parser.add_argument(
        "--provider-pool-config", type=Path,
        default=Path("predict/api_client/providers/current_endpoints.json"),
    )
    parser.add_argument("--providers", nargs="+", default=[])
    parser.add_argument("--tool-service-url", default="http://127.0.0.1:8765")
    parser.add_argument("--parallelism", type=int)
    parser.add_argument("--max-tokens", type=int, default=20_480)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--per-task-provider-split", action="store_true")
    parser.add_argument("--native-task-priors", action="store_true",
                        help="Reuse and repair the task-specific BBB/Oral single-molecule priors.")
    args = parser.parse_args(argv)
    if args.native_task_priors and set(args.tasks) - NATIVE_REQUIRED_CONTENT.keys():
        parser.error("--native-task-priors supports only BBB and Oral")
    root = args.output_root.resolve()
    if root.exists() and not args.resume:
        raise FileExistsError(f"Refusing to replace query-prior run: {root}")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    os.environ.setdefault("DEEPSEEK_API_KEY", "EMPTY")
    config_path = args.provider_pool_config.resolve()
    candidate = load_provider_pool_config(config_path)
    if args.providers:
        candidate = replace(
            candidate,
            providers=tuple(
                provider for provider in candidate.providers
                if provider.name in args.providers
            ),
        )
        if {provider.name for provider in candidate.providers} != set(args.providers):
            raise ValueError("--providers contains an unknown provider name")
        candidate.validate()
    requested = args.parallelism or primary_capacity(candidate)
    if args.per_task_provider_split and (
        [spec.name for spec in candidate.providers] != ["together", "cohere"]
        or candidate.providers[0].max_inflight != 3 * candidate.providers[1].max_inflight
        or requested != primary_capacity(candidate)
    ):
        raise ValueError("per-task split requires full 3:1 Together/Cohere capacity")
    selection = select_healthy_providers(candidate, requested)
    if args.per_task_provider_split and len(selection.config.providers) != 2:
        raise ValueError("both requested OpenRouter routes must pass preflight")
    loads = (sample_provider_loads(selection.config, samples=1, interval_s=0)
             if provider_from_base_url(candidate.providers[0].base_url) == "local" else [])
    root.mkdir(parents=True, exist_ok=args.resume)
    write_json_atomic(root / "execution.json", {
        "schema_version": "six_task_query_priors.v3",
        "status": "running",
        "host": socket.gethostname(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tasks": list(args.tasks),
        "prompt_version": args.prompt_version,
        "subset": args.subset,
        "benchmark": args.benchmark,
        "model": candidate.providers[0].model,
        "reasoning_effort": "high",
        "thinking": True,
        "max_tokens": args.max_tokens,
        "timeout_s": args.timeout_s,
        "provider_pool_config": str(config_path),
        "provider_pool_config_sha256": sha256_file(config_path),
        "provider_selection": selection.public_dict(),
        "per_task_provider_split": args.per_task_provider_split,
        "native_task_priors": args.native_task_priors,
        "observed_loads": loads,
    })
    tool = ToolServiceClient(args.tool_service_url, timeout_s=args.timeout_s)
    work = []
    sources: dict[tuple[str, int], dict[str, Any]] = {}
    task_inputs = {}
    for task in args.tasks:
        path, rows = _inputs(task, args.subset, args.benchmark)
        task_inputs[task] = {"path": str(path), "sha256": sha256_file(path)}
        contract = _contract(task, args.prompt_version)
        required = _required_content(task, args.native_task_priors)
        reusable = (
            _reusable_sources(args.reuse_root, task, rows, required)
            if task in args.reuse_tasks
            else {}
        )
        expected_profile = prompt_assets(args.prompt_version)["tasks"][task]["prompt_profile"]
        reusable = {
            query_id: source for query_id, source in reusable.items()
            if json.loads(Path(source["prompt_profile_manifest"]).read_text()).get(
                "task_prompt_profile"
            ) == expected_profile
        }
        reusable_by_identity = {}
        for reuse_root in args.reuse_identity_root:
            overlay_path = reuse_root / task / f"{task}__none" / "manifest.json"
            overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
            if overlay.get("schema_version") != "branch_query_prior_overlay.v1":
                raise ValueError(f"Invalid reusable query-prior overlay: {overlay_path}")
            for entry in overlay.get("sources") or []:
                key = (str(entry["molecule_identity_key"]), str(entry.get("condition_group") or ""))
                source = _prior_source(Path(entry["run_dir"]), required)
                if source is None or key in reusable_by_identity:
                    raise ValueError(f"Invalid reusable query-prior identity: {overlay_path}: {key}")
                source_manifest = json.loads(Path(source["prompt_profile_manifest"]).read_text())
                expected_profile = prompt_assets(args.prompt_version)["tasks"][task]["prompt_profile"]
                if source_manifest.get("task_prompt_profile") != expected_profile:
                    continue
                reusable_by_identity[key] = source
        for index, row in rows:
            query_id = str(row["benchmark_row_id"])
            key = (str(row.get("molecule_identity_key") or ""), str(row.get("condition_group") or ""))
            source = reusable.get(query_id) or reusable_by_identity.get(key)
            if source is not None:
                retrieval = json.loads(
                    (Path(source["run_dir"]) / "retrieval.json").read_text()
                )
                if str((retrieval.get("query") or {}).get("input_smiles") or "") != str(row["drug"]):
                    source = None
                else:
                    sources[task, index] = {**source, "source": "reused"}
                    continue
            prior_path = (
                root / task / f"{task}__none" / "runs"
                / f"{task}__none_idx{index:05d}"
                / "single_molecule_reasoning_output.json"
            )
            if args.resume and prior_path.is_file():
                existing = json.loads(prior_path.read_text(encoding="utf-8"))
                content = ((existing.get("llm") or {}).get("content"))
                if existing.get("status") == "ok" and isinstance(content, dict) and required <= content.keys():
                    source = _prior_source(prior_path.parent, required)
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
            messages = None if args.native_task_priors else build_query_prior_messages(contract, query)
            work.append((task, index, row, properties, messages, query))
    if args.per_task_provider_split:
        clients = {
            spec.name: provider_client(
                spec.max_inflight, provider_pool_config=config_path,
                config=replace(selection.config, providers=(spec,), max_failovers=0,
                               cooldown_seconds=0),
                max_tokens=args.max_tokens, timeout_s=args.timeout_s,
            )
            for spec in selection.config.providers
        }
    else:
        client = provider_client(
            selection.effective_parallelism,
            provider_pool_config=config_path,
            config=selection.config,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
        )

    def run(item: tuple[Any, ...]) -> tuple[Any, ...]:
        task, index, row, properties, messages, query = item
        selected_client = (clients["cohere" if index % 4 == 3 else "together"]
                           if args.per_task_provider_split else client)
        for retry in range(2 if args.per_task_provider_split else 1):
            try:
                if args.native_task_priors:
                    if task == "bbb_martins":
                        from predict.harnesses.branches.tasks.bbb_martins.pipeline import _reason_single_molecule
                    else:
                        from predict.harnesses.branches.tasks.bioavailability_ma.pipeline import _reason_single_molecule
                    single = _reason_single_molecule(
                        selected_client, query,
                        prompt_profile=prompt_assets(args.prompt_version)["tasks"][task]["prompt_profile"],
                    )
                    response = single["llm"]
                    messages = response.get("messages") or []
                else:
                    response = selected_client.chat_json(messages, max_tokens=args.max_tokens)
                content = response.get("content")
                if not isinstance(content, dict) or not _required_content(task, args.native_task_priors) <= content.keys():
                    raise ValueError(f"Invalid query prior for {task}/{index}")
                break
            except Exception:
                if retry:
                    raise
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
                    details = getattr(exc, "attempts", [])
                    errors = [(row.get("error_type"), row.get("error")) for row in details]
                    failures.append(
                        f"attempt={attempt}: {type(exc).__name__}: {exc}; "
                        f"provider_errors={errors}"
                    )
                    if len(failures) <= 5:
                        print(
                            f"query-prior failure attempt={attempt} "
                            f"upstream_types={[row.get('error_type') for row in details]}",
                            file=sys.stderr, flush=True,
                        )
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
                    "task_prompt_profile": prompt_assets(args.prompt_version)["tasks"][task][
                        "prompt_profile"
                    ],
                })
                sources[task, index] = {**_prior_source(run_dir, _required_content(task, args.native_task_priors)), "source": "fresh"}
        pending = retry
    if pending:
        execution = json.loads((root / "execution.json").read_text())
        execution.update(
            status="failed", completed_requests=len(sources), failures=failures,
            provider_snapshot=({name: pool.snapshot() for name, pool in clients.items()}
                               if args.per_task_provider_split else client.snapshot()),
        )
        write_json_atomic(root / "execution.json", execution)
        raise RuntimeError(f"Query-prior recovery exhausted for {len(pending)} items")
    for task in args.tasks:
        batch = root / task / f"{task}__none"
        path, rows = _inputs(task, args.subset, args.benchmark)
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
                "task_prompt_profile": prompt_assets(args.prompt_version)["tasks"][task][
                    "prompt_profile"
                ],
                **source,
            })
        write_json_atomic(batch / "manifest.json", {
            "schema_version": "branch_query_prior_overlay.v1", "status": "complete",
            "task_id": task, "subset": args.subset, "n_items": len(entries),
            "benchmark": args.benchmark,
            "input_jsonl": str(path), "input_sha256": sha256_file(path),
            "input": task_inputs[task], "model": candidate.providers[0].model, "reasoning_effort": "high",
            "prompt_version": args.prompt_version,
            "task_prompt_profile": prompt_assets(args.prompt_version)["tasks"][task][
                "prompt_profile"
            ],
            "reused": sum(row["source"] == "reused" for row in entries),
            "fresh": sum(row["source"] == "fresh" for row in entries),
            "sources": entries,
        })
    execution = json.loads((root / "execution.json").read_text())
    execution.update(
        status="complete", completed_requests=len(sources),
        fresh_requests=sum(row["source"] == "fresh" for row in sources.values()),
        reused_requests=sum(row["source"] == "reused" for row in sources.values()),
        generated_this_attempt=len(work),
        recovery_failures=failures,
        provider_snapshot=({name: pool.snapshot() for name, pool in clients.items()}
                           if args.per_task_provider_split else client.snapshot()),
    )
    write_json_atomic(root / "execution.json", execution)
    print(json.dumps(execution, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
