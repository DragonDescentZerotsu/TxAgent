"""Replay manifest-bound evidence requests through the shared family runtime."""

from dataclasses import asdict
import json
from pathlib import Path
import time

from tools.chembl_tool.common.record_budget import content_hash
from tools.chembl_tool.common.reasoning_validation import (
    evidence_validation_kwargs as validation_kwargs,
)
from tools.chembl_tool.common.json_utils import (
    read_jsonl,
    sha256_file,
    write_json_atomic,
)
from tools.chembl_tool.common.reasoning_validation import (
    call_with_json_validation,
    response_validation_errors,
    structured_response_is_valid,
)
from tools.chembl_tool.common.reasoning_race import ParallelRetryClient
from tools.chembl_tool.paper_experiments import (
    run_conditioned_assay_progressive_curve as runtime,
)

VERSION = "direct_indirect_budgeted.v3"


def _read(path):
    return json.loads(Path(path).read_text())


def run(args):
    runtime.load_env_file(Path(args.env_file))
    return _run_prepared(args)


def task_contract(args, task):
    from tools.chembl_tool.common.tdc_benchmark import PROFILE, task_contract as adapt

    return adapt(
        runtime._task_contract(task),
        PROFILE if getattr(args, "benchmark_manifest", "") else "",
    )


def _run_query(args, query, client):
    prepared = _read(query.query_dir / "prepared.json")
    if content_hash(prepared["messages"]) != prepared["request_hash"]:
        raise ValueError("prepared request hash mismatch")
    if prepared["manifest_hash"] != args.manifest_hash:
        raise ValueError("prepared manifest hash mismatch")
    if args.prepare_only:
        return {"task": query.task, "index": query.index, "status": "ok"}
    if not prepared.get("context_ready", prepared.get("tool_prefetch_complete")):
        raise ValueError("inference requires tool prefetch; use a fresh output root")
    contract = task_contract(args, query.task)
    validation = validation_kwargs(contract, prepared["card_alias_map"])
    output = query.query_dir / "output.json"
    if output.exists():
        saved = _read(output)
        if saved.get("request_hash") != prepared["request_hash"]:
            raise ValueError("prediction request hash mismatch")
        if saved["status"] == "ok" and not response_validation_errors(
            saved["llm"], **validation
        ):
            return {"task": query.task, "index": query.index, "status": "ok"}

    def call(messages, retry=False):
        if isinstance(client, ParallelRetryClient):
            return client.chat_validated(
                messages,
                task=query.task,
                width=args.retry_race_width
                if retry or getattr(args, "retry_round", 0)
                else 1,
                validate=lambda response: response_validation_errors(
                    response, **validation
                ),
                receipt_path=query.query_dir / "retry_races" / f"{time.time_ns()}.json",
            )
        return client.chat_json(messages)

    response = call_with_json_validation(
        call,
        prepared["messages"],
        **validation,
        retry_call=lambda messages: call(messages, retry=True),
        branch_name=f"{query.task} budgeted decision",
        max_attempts=4,
    )
    status = "ok" if structured_response_is_valid(response) else "error"
    write_json_atomic(
        output,
        {
            "status": status,
            "model_called": True,
            "request_hash": prepared["request_hash"],
            "llm": response,
            "card_alias_map": prepared["card_alias_map"],
        },
    )
    return {"task": query.task, "index": query.index, "status": status}


def _run_prepared(args):
    """Reuse the ordinary client, query runner and summaries for frozen requests."""
    root = Path(args.output_root)
    manifest = _read(root / "experiment_manifest.json")
    if manifest["version"] != VERSION or manifest.get("feature_inputs_only"):
        raise ValueError("requires a budgeted inference manifest")
    if (
        args.matched_progressive_root
        or getattr(args, "baseline_suite", False)
        or getattr(args, "skip_tool_prefetch", False)
    ):
        raise ValueError("prepared replay cannot rebuild or mix legacy/suite inputs")
    args.tasks = args.tasks or list(manifest["inputs"])
    if set(args.tasks) - set(manifest["inputs"]):
        raise ValueError("prepared replay task is absent from the manifest")
    args.retry_race_width = args.retry_race_width or 1
    if not 1 <= args.parallelism <= args.endpoint_concurrency_budget:
        raise ValueError("invalid prepared replay concurrency")
    if not 1 <= args.retry_race_width <= args.parallelism:
        raise ValueError("invalid prepared replay retry race width")
    provider = runtime._resolve_provider_pool_config(args)
    if (
        sum(p.max_inflight for p in provider.providers)
        > args.endpoint_concurrency_budget
    ):
        raise ValueError("provider capacity exceeds prepared replay concurrency budget")
    if (
        manifest["model"] != args.model
        or any(p.model != args.model for p in provider.providers)
        or manifest["max_tokens"] != args.max_tokens
        or manifest["generation"] != runtime._generation_settings(provider)
        or any(
            p.request_extra_body != manifest["request_extra_body"]
            for p in provider.providers
        )
    ):
        raise ValueError("prepared replay generation contract mismatch")
    if bool(manifest.get("benchmark_profile")) != bool(args.benchmark_manifest):
        raise ValueError("prepared replay benchmark profile mismatch")
    if args.benchmark_manifest:
        benchmark = _read(Path(args.benchmark_manifest))
        if benchmark.get("profile") != manifest["benchmark_profile"]:
            raise ValueError("prepared replay benchmark profile mismatch")
    args.manifest_hash = content_hash(manifest)
    args.evidence_setting = manifest["setting"]
    args.indirect_selector = manifest["selection"]["policy"]
    args.indirect_budget = manifest["selection"]["budget"]
    queries = []
    for task in args.tasks:
        if task in manifest.get("task_contracts", {}) and manifest["task_contracts"][
            task
        ] != content_hash(asdict(task_contract(args, task))):
            raise ValueError("prepared replay task contract changed")
        source = manifest["inputs"][task]["input"]
        if args.benchmark_manifest:
            candidates = benchmark.get("tasks", {}).get(task, {}).values()
            if not any(
                isinstance(item, dict)
                and item.get("sha256") == source["sha256"]
                and Path(item.get("path", "")).resolve()
                == Path(source["path"]).resolve()
                for item in candidates
            ):
                raise ValueError("prepared replay benchmark input binding mismatch")
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError("prepared replay benchmark input changed")
        rows = read_jsonl(Path(source["path"]))
        for index in runtime._selected_indices(args, len(rows)):
            directory = root / "runs" / task / f"query_{index:05d}"
            prepared = _read(directory / "prepared.json")
            query_hash = content_hash(
                {
                    "benchmark_profile": manifest.get("benchmark_profile", ""),
                    "canonical_smiles": rows[index]["drug"],
                    "condition_group": rows[index].get("condition_group", ""),
                }
            )
            if (
                prepared["manifest_hash"] != args.manifest_hash
                or prepared.get("query_hash") != query_hash
                or content_hash(prepared["messages"]) != prepared["request_hash"]
                or _read(directory / "evaluation.json")["label"]
                != int(rows[index]["Y"])
            ):
                raise ValueError(
                    "prepared replay request or evaluation binding mismatch"
                )
            queries.append(runtime.PreparedQuery(task, index, directory))
    write_json_atomic(root / "execution_provider.json", asdict(provider))
    client = None if args.prepare_only else runtime._make_client(args, provider)
    try:
        failed = runtime._run_query_rounds(
            args, queries, client, root, run_query=_run_query
        )
    finally:
        if client is not None:
            client.close()
    if not args.prepare_only:
        summarize(args, queries, root)
    return int(bool(failed))


def summarize(args, prepared, root):
    """Validate saved responses and share metrics between single and suite runs."""
    for task in args.tasks:
        predictions = []
        for q in prepared:
            if q.task != task:
                continue
            output = (
                _read(q.query_dir / "output.json")
                if (q.query_dir / "output.json").exists()
                else {}
            )
            saved_input = _read(q.query_dir / "prepared.json")
            label = _read(q.query_dir / "evaluation.json")["label"]
            contract = task_contract(args, task)
            valid = (
                output.get("status") == "ok"
                and saved_input["manifest_hash"] == args.manifest_hash
                and output.get("request_hash")
                == saved_input["request_hash"]
                == content_hash(saved_input["messages"])
                and structured_response_is_valid(output.get("llm", {}))
                and not response_validation_errors(
                    output.get("llm", {}),
                    **validation_kwargs(contract, saved_input["card_alias_map"]),
                )
            )
            pred = (
                runtime._prediction_to_label(
                    contract, output["llm"]["content"][contract.prediction_field]
                )
                if valid
                else None
            )
            predictions.append(
                {
                    "index": q.index,
                    "label": label,
                    "pred_label": pred,
                    "correct": pred == label,
                    "model_called": output.get("model_called", False),
                }
            )
        runtime._write_prediction_summary(
            task=task,
            predictions=predictions,
            summary_dir=root / "summary" / task,
            metric_fields={
                "setting": args.evidence_setting,
                "selector": args.indirect_selector,
                "indirect_budget": args.indirect_budget,
            },
        )
