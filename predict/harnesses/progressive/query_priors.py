"""Build hash-pinned BBB and Oral test query priors in one shared prompt pool."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import socket
from pathlib import Path
import sys
import time
from typing import Any
from urllib.request import urlopen

from data.processing.gold_labels.conditioned_benchmark import split_path
from predict.api_client.pool import (
    load_provider_pool_config,
    primary_capacity,
    select_healthy_providers,
)
from predict.harnesses.branches.matrix import provider_client, sample_provider_loads
from predict.harnesses.branches.scheduler import (
    BatchCommand,
    prepare_batch_commands,
    run_prepared_prompt_pool,
)
from predict.llm_io.response import structured_response_is_valid
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic


TASKS = ("bbb_martins", "bioavailability_ma")
TASK_PROFILES = {
    "bbb_martins": ("--bbb-prompt-profile", "meaningful_cns_access_v1"),
    "bioavailability_ma": (
        "--bioavailability-prompt-profile",
        "f20_evidence_calibrated_v2",
    ),
}
EXPECTED_TEST_ROWS = {"bbb_martins": 393, "bioavailability_ma": 269}
DEFAULT_PROVIDER_CONFIG = Path(__file__).resolve().parents[2] / (
    "api_client/providers/progressive_test_query_priors_four_endpoint_128_high.json"
)
VALID_PRIOR_ROOT = Path(
    "outputs/paper/legacy/"
    "starling_conditioned_gold_l1_deepseek_v4_flash_nvfp4_query_prior/"
    "runs_deployment_visible_parent_disjoint"
)
DEFAULT_OUTPUT_PARENT = Path(
    "outputs/paper/assay_transfer_harness/joseph/query_priors"
)
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
MAX_TOKENS = 20_480
TIMEOUT_S = 900
TOOL_SERVICE_URL = "http://127.0.0.1:8765"
EXPECTED_PROVIDER_URLS = (
    "http://dgx020:50002/v1",
    "http://dgx005:50001/v1",
    "http://dgx011:50001/v1",
    "http://dgx017:50001/v1",
)


class _CaptureClient:
    def __init__(self, response: dict[str, Any]):
        self.response = response
        self.messages: list[dict[str, Any]] = []

    def chat_json(self, messages: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        self.messages = messages
        return self.response


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_visible_payload_contract() -> dict[str, str]:
    """Require current none-branch messages to match the saved valid contract."""
    hashes: dict[str, str] = {}
    for task in TASKS:
        batch = VALID_PRIOR_ROOT / task / f"{task}__none"
        run = batch / "runs" / f"{task}__none_idx00000"
        batch_manifest = json.loads((batch / "manifest.json").read_text())
        retrieval = json.loads((run / "retrieval.json").read_text())
        single = json.loads((run / "single_molecule_reasoning_output.json").read_text())
        final = json.loads((run / "final_reasoning_output.json").read_text())
        saved_single_messages = (single.get("llm") or {}).get("messages") or []
        saved_final_messages = (final.get("llm") or {}).get("messages") or []
        if len(saved_single_messages) < 2 or len(saved_final_messages) < 2:
            raise ValueError(f"{task} validation prior lacks saved visible messages")

        single_payload = json.loads(saved_single_messages[1]["content"])
        query = single_payload["query"]
        chembl_context = single_payload.get("exact_query_chembl_context")
        module = importlib.import_module(
            f"predict.harnesses.branches.tasks.{task}.pipeline"
        )
        single_client = _CaptureClient(single["llm"])
        module._reason_single_molecule(
            single_client,
            query,
            chembl_context,
            prompt_profile=batch_manifest["task_prompt_profile"],
        )
        current_retrieval = dict(retrieval)
        current_retrieval["query"] = query
        final_client = _CaptureClient(final["llm"])
        module._run_final_reasoning(
            final_client,
            current_retrieval,
            single,
            [],
            prompt_profile=batch_manifest["task_prompt_profile"],
        )
        current = {
            "single": single_client.messages,
            "final": final_client.messages,
        }
        saved = {
            "single": saved_single_messages[:2],
            "final": saved_final_messages[:2],
        }
        if current != saved:
            raise ValueError(f"{task} current visible query-prior payload drifted")
        hashes[task] = _canonical_hash(current)
    return hashes


def build_commands(
    output_root: Path,
    provider_config: Path,
    *,
    tool_service_url: str = TOOL_SERVICE_URL,
    limit: int = 0,
) -> list[BatchCommand]:
    commands = []
    for task in TASKS:
        profile_option, profile = TASK_PROFILES[task]
        command = [
            sys.executable,
            "-m",
            f"predict.harnesses.branches.tasks.{task}.contract",
            "--input-jsonl",
            str(split_path(task, "test").resolve()),
            "--batch-root",
            str((output_root / task).resolve()),
            "--batch-id",
            f"{task}__none",
            "--experiment-mode",
            "none",
            "--neighbor-identity-policy",
            "parent_disjoint",
            "--harness-prefetch-tools",
            "--provider-pool-config",
            str(provider_config.resolve()),
            "--tool-service-url",
            tool_service_url,
            "--model",
            MODEL,
            "--reasoning-effort",
            "high",
            "--enable-thinking",
            "--temperature",
            "0",
            "--max-tokens",
            str(MAX_TOKENS),
            "--timeout-s",
            str(TIMEOUT_S),
            "--max-tool-rounds",
            "3",
            "--execution-mode",
            "throughput",
            "--skip-existing",
            "--no-stream-logs",
            "--no-combine-traces",
            profile_option,
            profile,
        ]
        if limit:
            command.extend(["--limit", str(limit)])
        commands.append(BatchCommand(f"{task}__none", command))
    return commands


def _validate_provider_config(candidate: Any, *, parallelism: int) -> None:
    actual = tuple(
        (provider.base_url, provider.max_inflight, provider.model)
        for provider in candidate.providers
    )
    expected = tuple((url, 128, MODEL) for url in EXPECTED_PROVIDER_URLS)
    if actual != expected or primary_capacity(candidate) != 512 or parallelism != 512:
        raise ValueError(
            "test query priors require the pinned four endpoints at 128 requests each"
        )
    reasoning_efforts = {
        str((provider.request_extra_body or {}).get("chat_template_kwargs", {}).get(
            "reasoning_effort"
        ))
        for provider in candidate.providers
    }
    if reasoning_efforts != {"high"}:
        raise ValueError(
            "every query-prior endpoint must enforce reasoning_effort=high"
        )


def validate_release(output_root: Path, *, limit: int = 0) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for task in TASKS:
        input_path = split_path(task, "test").resolve()
        records = read_jsonl(input_path)
        expected = min(limit, len(records)) if limit else EXPECTED_TEST_ROWS[task]
        keys = [
            (row["molecule_identity_key"], row.get("condition_group", ""))
            for row in records[:expected]
        ]
        if len(records) != EXPECTED_TEST_ROWS[task] or len(keys) != len(set(keys)):
            raise ValueError(f"{task} test input count or stable identities changed")
        batch = output_root / task / f"{task}__none"
        manifest = json.loads((batch / "manifest.json").read_text())
        required_manifest = {
            "input_jsonl_sha256": sha256_file(input_path),
            "n_items": expected,
            "model": MODEL,
            "experiment_mode": "none",
            "visibility_mode": "deployment_visible_prefetched",
            "harness_prefetch_tools": True,
            "task_prompt_profile": TASK_PROFILES[task][1],
        }
        mismatches = {
            key: (value, manifest.get(key))
            for key, value in required_manifest.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"{task} query-prior manifest mismatch: {mismatches}")

        artifact_rows = []
        for index, record in enumerate(records[:expected]):
            run = batch / "runs" / f"{task}__none_idx{index:05d}"
            retrieval = json.loads((run / "retrieval.json").read_text())
            single_path = run / "single_molecule_reasoning_output.json"
            final_path = run / "final_reasoning_output.json"
            single = json.loads(single_path.read_text())
            final = json.loads(final_path.read_text())
            run_manifest = json.loads((run / "manifest.json").read_text())
            if str((retrieval.get("query") or {}).get("input_smiles") or "") != str(
                record["drug"]
            ):
                raise ValueError(f"{task} query {index} SMILES mismatch")
            if (
                single.get("status") != "ok"
                or final.get("status") != "ok"
                or not structured_response_is_valid(single.get("llm") or {})
                or not structured_response_is_valid(final.get("llm") or {})
            ):
                raise ValueError(f"{task} query {index} has incomplete reasoning")
            tool_results = (single.get("llm") or {}).get("tool_results") or []
            if not tool_results or tool_results[0].get("status") != "ok":
                raise ValueError(f"{task} query {index} lacks molecule properties")
            if (
                run_manifest.get("reasoning_effort") != "high"
                or run_manifest.get("thinking") != {"type": "enabled"}
            ):
                raise ValueError(f"{task} query {index} has wrong reasoning settings")
            artifact_rows.extend(
                (str(path.relative_to(output_root)), sha256_file(path))
                for path in (single_path, final_path)
            )
        summaries[task] = {
            "input_jsonl": str(input_path),
            "input_jsonl_sha256": sha256_file(input_path),
            "queries": expected,
            "reasoning_outputs": len(artifact_rows),
            "reasoning_outputs_sha256": _canonical_hash(artifact_rows),
            "batch_manifest": str(batch / "manifest.json"),
            "batch_manifest_sha256": sha256_file(batch / "manifest.json"),
        }
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--provider-pool-config", type=Path, default=DEFAULT_PROVIDER_CONFIG)
    parser.add_argument("--parallelism", type=int, default=512)
    parser.add_argument("--preparation-workers", type=int, default=64)
    parser.add_argument("--tool-service-url", default=TOOL_SERVICE_URL)
    parser.add_argument(
        "--wait-for-drain-seconds",
        type=int,
        default=0,
        help="Wait up to this many seconds for every configured endpoint to drain.",
    )
    parser.add_argument(
        "--allow-active-endpoints",
        action="store_true",
        help="Launch despite recorded running or waiting requests on the endpoints.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.parallelism < 1
        or args.preparation_workers < 1
        or args.limit < 0
        or args.wait_for_drain_seconds < 0
    ):
        parser.error(
            "parallelism and preparation workers must be positive; "
            "limit and drain wait must be non-negative"
        )
    output_root = (args.output_root or DEFAULT_OUTPUT_PARENT / time.strftime(
        "scaffold_test_deepseek_v4_flash_0731_high_%Y%m%d_%H%M%S"
    )).resolve()
    if args.validate_only:
        print(json.dumps(validate_release(output_root, limit=args.limit), indent=2))
        return 0

    visible_payload_hashes = validate_visible_payload_contract()
    provider_path = args.provider_pool_config.resolve()
    drain_deadline = time.monotonic() + args.wait_for_drain_seconds
    while True:
        candidate = load_provider_pool_config(provider_path)
        _validate_provider_config(candidate, parallelism=args.parallelism)
        selection = select_healthy_providers(candidate, args.parallelism)
        if len(selection.config.providers) != 4:
            raise RuntimeError("all four query-prior endpoints must pass model preflight")
        loads = sample_provider_loads(selection.config, samples=6, interval_s=1.0)
        active = [row for row in loads[-4:] if row["running"] or row["waiting"]]
        if not active or args.allow_active_endpoints:
            break
        if time.monotonic() >= drain_deadline:
            raise RuntimeError(f"query-prior endpoints are not drained: {active}")
        print(
            json.dumps({"waiting_for_endpoint_drain": active}, sort_keys=True),
            flush=True,
        )
        time.sleep(min(30.0, max(0.0, drain_deadline - time.monotonic())))
    with urlopen(args.tool_service_url.rstrip("/") + "/health", timeout=10) as response:
        tool_health = json.load(response)

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "launcher.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = {
            "schema_version": "progressive_query_prior_release.v1",
            "status": "running",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": socket.gethostname(),
            "subset": "test",
            "tasks": list(TASKS),
            "model": MODEL,
            "reasoning_effort": "high",
            "thinking": True,
            "max_tokens": MAX_TOKENS,
            "timeout_s": TIMEOUT_S,
            "parallelism": args.parallelism,
            "allow_active_endpoints": args.allow_active_endpoints,
            "provider_pool": {
                "path": str(provider_path),
                "sha256": sha256_file(provider_path),
                "selection": selection.public_dict(),
                "load_samples": loads,
            },
            "tool_service_url": args.tool_service_url,
            "tool_health": tool_health,
            "visible_payload_hashes": visible_payload_hashes,
        }
        write_json_atomic(output_root / "manifest.json", receipt)
        commands = build_commands(
            output_root,
            provider_path,
            tool_service_url=args.tool_service_url,
            limit=args.limit,
        )
        prepared = prepare_batch_commands(
            commands,
            max_workers=args.parallelism,
            max_stage_requeues=0,
        )
        client = provider_client(
            args.parallelism,
            provider_pool_config=provider_path,
            config=selection.config,
            max_tokens=MAX_TOKENS,
            timeout_s=TIMEOUT_S,
        )
        failed = run_prepared_prompt_pool(
            prepared,
            max_workers=args.parallelism,
            max_stage_requeues=0,
            preparation_workers=args.preparation_workers,
            stage_client=client,
        )
        receipt["provider_snapshot"] = client.snapshot()
        receipt["failed_batches"] = failed
        if failed:
            receipt["status"] = "incomplete"
            write_json_atomic(output_root / "manifest.json", receipt)
            return 1
        receipt["tasks_summary"] = validate_release(output_root, limit=args.limit)
        receipt["status"] = "complete"
        receipt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json_atomic(output_root / "manifest.json", receipt)
    print(json.dumps({"query_prior_root": str(output_root), "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
