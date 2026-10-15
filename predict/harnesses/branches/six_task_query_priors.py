"""Generate validated full-split query-prior overlays for six tasks."""

from __future__ import annotations

from predict.prompts import query_prior_messages

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any

from data.processing.gold_labels.conditioned_benchmark import split_path, tdc_split_path
from predict.api_client.transport import provider_from_base_url
from predict.api_client.pool import (
    load_provider_pool_config,
    primary_capacity,
    select_healthy_providers,
)
from predict.harnesses.branches.matrix import provider_client, sample_provider_loads
from predict.harnesses.branches.flat import (
    CONTEXT_V5_SIX_TASKS_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
    FINAL_FULL_FLAT_PROMPT_VERSION,
    prompt_assets,
)
from predict.tools.client import ToolServiceClient, retry_failed_tool_calls
from predict.utils.response import (
    allowed_values_from_required_schema, call_with_json_validation,
    structured_response_is_valid, validated_branch_content,
    validate_query_prior_output,
)
from predict.utils.json import read_jsonl, write_json_atomic
from predict.harnesses.branches.manifests import (
    read_record, read_run, write_batch, write_prior_batch, prior_sources, canonical_location,
    PREDICTION_ROOT, prediction_run_root, prediction_batch_directory, query_prior_batch,
)


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


def _contract(task: str, prompt_version: str) -> dict[str, Any]:
    return prompt_assets(prompt_version)["tasks"][task]


build_query_prior_messages = query_prior_messages


def _inputs(
    task: str, subset: str, benchmark: str = "gold", input_root: Path | None = None,
) -> tuple[Path, list[tuple[int, dict[str, Any]]]]:
    if input_root is None:
        split = tdc_split_path if benchmark == "tdc" else split_path
        path = split(task, subset).with_name(
            f"{subset}_molecule_condition_labels.jsonl"
        ).resolve()
    else:
        path = (input_root / f"{task}.jsonl").resolve()
    rows = read_jsonl(path)
    if len({str(row["benchmark_row_id"]) for row in rows}) != len(rows):
        raise ValueError(f"Conditioned benchmark repeats query IDs: {path}")
    return path, list(enumerate(rows))


def _prior_source(run_dir: Path, required: set[str] = REQUIRED_CONTENT) -> dict[str, Any] | None:
    run_dir = canonical_location(run_dir)
    retrieval = run_dir / "retrieval.json"
    single = run_dir / "single_molecule_reasoning_output.json"
    if not retrieval.is_file() or not single.is_file():
        return None
    try:
        result = validate_query_prior_output(json.loads(single.read_text(encoding="utf-8")))
    except ValueError:
        return None
    content = validated_branch_content(result)
    if result.get("status") != "ok" or not isinstance(content, dict) or not required <= content.keys():
        return None
    manifest = run_dir.parent.parent / "manifest.json"
    return {"run_dir": str(run_dir.resolve()),
            "prompt_profile_manifest": str(manifest),
            "source_run_uid": read_run(run_dir).get("run_uid", "")}



def _reusable_sources(
    root: Path | None, task: str, rows: list[tuple[int, dict[str, Any]]],
    required: set[str],
    *, subset: str = "valid", benchmark: str = "gold",
) -> dict[str, dict[str, Any]]:
    if root is None:
        return {}
    batch = query_prior_batch(root, task, subset, benchmark)
    if not (batch / "manifest.json").is_file():
        return {}
    output = {}
    for entry in prior_sources(batch):
        query_id = str(entry.get("benchmark_row_id") or "")
        source = _prior_source(Path(entry["run_dir"]), required)
        if not query_id or source is None or query_id in output:
            raise ValueError(f"Invalid reusable query prior: {batch}")
        output[query_id] = source
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PREDICTION_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--subset", choices=("valid", "valid_small", "test", "fresh"), required=True)
    parser.add_argument("--input-root", type=Path,
                        help="Explicit per-task JSONL directory for fresh unique queries.")
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
            FINAL_FULL_FLAT_PROMPT_VERSION,
        ),
        default=FINAL_FULL_FLAT_PROMPT_VERSION,
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
    if (args.subset == "fresh") != (args.input_root is not None):
        parser.error("--subset fresh and --input-root must be supplied together")
    if args.native_task_priors and set(args.tasks) - NATIVE_REQUIRED_CONTENT.keys():
        parser.error("--native-task-priors supports only BBB and Oral")
    if not args.run_name and (not args.resume or args.output_root == PREDICTION_ROOT):
        parser.error("New query-prior runs require --run-name")
    root = (prediction_run_root(args.output_root, "query-prior", args.prompt_version, args.run_name)
            if args.run_name else canonical_location(args.output_root))
    batches = {task: (prediction_batch_directory(root, args.benchmark, task, args.subset, "query-prior")
                      if args.run_name else query_prior_batch(root, task, args.subset, args.benchmark))
               for task in args.tasks}
    if root.exists() and not args.resume:
        raise FileExistsError(f"Refusing to replace query-prior run: {root}")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    os.environ.setdefault("DEEPSEEK_API_KEY", "EMPTY")
    config_path = args.provider_pool_config.resolve()
    candidate = load_provider_pool_config(config_path)
    if (root / "execution.json").is_file():
        previous = json.loads((root / "execution.json").read_text())
        expected = {"model": candidate.providers[0].model, "prompt_version": args.prompt_version,
                    "subset": args.subset, "benchmark": args.benchmark, "max_tokens": args.max_tokens,
                    "native_task_priors": args.native_task_priors}
        changed = [key for key, value in expected.items() if previous.get(key) != value]
        if changed:
            raise ValueError(f"Named query-prior run has incompatible settings {changed}; use a new --run-name")
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
        "input_root": str(args.input_root.resolve()) if args.input_root else None,
        "model": candidate.providers[0].model,
        "reasoning_effort": "high",
        "thinking": provider_from_base_url(candidate.providers[0].base_url) == "local",
        "max_tokens": args.max_tokens,
        "timeout_s": args.timeout_s,
        "provider_pool_config": str(config_path),
        "provider_profile": config_path.stem,
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
        path, rows = _inputs(task, args.subset, args.benchmark, args.input_root)
        task_inputs[task] = {"path": str(path)}
        batch = batches[task]
        write_batch(batch / "manifest.json", {
            "task_id": task, "subset": args.subset, "benchmark": args.benchmark,
            "input_jsonl": str(path), "batch_id": f"{task}__none", "experiment_mode": "none",
            "run_name": args.run_name or "", "harness_version": "query-prior",
            "model": candidate.providers[0].model, "reasoning_effort": "high",
            "max_tokens": args.max_tokens, "temperature": 0,
            "thinking": provider_from_base_url(candidate.providers[0].base_url) == "local",
            "provider_profile": config_path.stem, "prompt_version": args.prompt_version,
            "task_prompt_profile": prompt_assets(args.prompt_version)["tasks"][task]["prompt_profile"],
        }, records=rows)
        contract = _contract(task, args.prompt_version)
        required = _required_content(task, args.native_task_priors)
        reusable = (
            _reusable_sources(args.reuse_root, task, rows, required, subset=args.subset, benchmark=args.benchmark)
            if task in args.reuse_tasks
            else {}
        )
        expected_profile = prompt_assets(args.prompt_version)["tasks"][task]["prompt_profile"]
        reusable = {
            query_id: source for query_id, source in reusable.items()
            if read_record(Path(source["prompt_profile_manifest"])).get(
                "task_prompt_profile"
            ) == expected_profile
        }
        reusable_by_identity = {}
        for reuse_root in args.reuse_identity_root:
            overlay_path = query_prior_batch(reuse_root, task, args.subset, args.benchmark) / "manifest.json"
            for entry in prior_sources(overlay_path.parent):
                key = (str(entry["molecule_identity_key"]), str(entry.get("condition_group") or ""))
                source = _prior_source(Path(entry["run_dir"]), required)
                if source is None or key in reusable_by_identity:
                    raise ValueError(f"Invalid reusable query-prior identity: {overlay_path}: {key}")
                source_manifest = read_record(Path(source["prompt_profile_manifest"]))
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
                batches[task] / "runs"
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
            tool_call = (
                "molecule_properties", {"query_smiles": str(row["drug"]), "logd_ph": 7.4}
            )
            properties = tool.invoke(*tool_call)
            properties = retry_failed_tool_calls(tool, [tool_call], [properties])[0]
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
        attempts = 2 if args.per_task_provider_split else 1
        for retry in range(attempts):
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
                    response = call_with_json_validation(
                        selected_client.chat_json, messages,
                        required_fields=REQUIRED_CONTENT,
                        allowed_values=allowed_values_from_required_schema(messages),
                        branch_name="query-prior", max_attempts=1,
                    )
                content = response.get("content")
                if (not structured_response_is_valid(response) or not isinstance(content, dict)
                        or not _required_content(task, args.native_task_priors) <= content.keys()):
                    raise ValueError(f"Invalid query prior for {task}/{index}")
                break
            except Exception:
                if retry == attempts - 1:
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
                batch = batches[task]
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
        batch = batches[task]
        path, rows = _inputs(task, args.subset, args.benchmark, args.input_root)
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
        write_prior_batch(batch, {
            "schema_version": "branch_query_prior_overlay.v1", "status": "complete",
            "task_id": task, "subset": args.subset, "n_items": len(entries),
            "benchmark": args.benchmark,
            "input_jsonl": str(path),
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
