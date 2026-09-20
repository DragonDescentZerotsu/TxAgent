"""Generate missing TDC query priors and publish stable-identity overlays.

Existing Gold-v1 valid/test priors are reused by molecule-condition identity.
Only missing TDC identities are sent to the configured DGX provider pool. The
published overlay pins every source artifact by hash for positional consumers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from data.processing.gold_labels.conditioned_benchmark import split_path, tdc_split_path
from predict.api_client.pool import (
    load_provider_pool_config, primary_capacity, select_healthy_providers,
)
from predict.harnesses.branches.matrix import provider_client
from predict.harnesses.branches.scheduler import (
    BatchCommand, prepare_batch_commands, run_prepared_prompt_pool,
)
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


TASKS = ("bbb_martins", "bioavailability_ma")
SUBSETS = ("valid", "test")
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
PROFILES = {
    "bbb_martins": ("--bbb-prompt-profile", "meaningful_cns_access_v1"),
    "bioavailability_ma": ("--bioavailability-prompt-profile", "f20_evidence_calibrated_v2"),
}
GOLD_PRIORS = {
    "valid": Path(
        "outputs/paper/legacy/"
        "starling_conditioned_gold_l1_deepseek_v4_flash_nvfp4_query_prior/"
        "runs_deployment_visible_parent_disjoint"
    ),
    "test": Path(
        "outputs/paper/assay_transfer_harness/joseph/query_priors/"
        "scaffold_test_deepseek_v4_flash_0731_high_20260918_1744_r2"
    ),
}


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["molecule_identity_key"]), str(row.get("condition_group") or "")


def _run(batch: Path, index: int) -> Path:
    return batch / "runs" / f"{batch.name}_idx{index:05d}"


def _source_prompt_profile(task: str, run: Path) -> tuple[Path, str]:
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = run.parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = str(manifest.get("task_prompt_profile") or "")
    expected = PROFILES[task][1]
    if profile != expected:
        raise ValueError(
            f"Query-prior prompt profile differs for {run}: {profile!r} != {expected!r}"
        )
    return manifest_path, profile


def _gold_sources(task: str) -> dict[tuple[str, str], Path]:
    sources: dict[tuple[str, str], Path] = {}
    for subset, root in GOLD_PRIORS.items():
        batch = (root / task / f"{task}__none").resolve()
        rows = read_jsonl(split_path(task, subset))
        for index, row in enumerate(rows):
            key = _key(row)
            run = _run(batch, index)
            retrieval = run / "retrieval.json"
            single = run / "single_molecule_reasoning_output.json"
            if not retrieval.is_file() or not single.is_file():
                raise ValueError(f"Gold query prior is incomplete: {run}")
            if key in sources:
                raise ValueError(f"Gold query-prior identity repeats across splits: {key}")
            sources[key] = run
    return sources


def prepare_inputs(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    summary = {}
    for task in TASKS:
        existing = _gold_sources(task)
        missing: dict[tuple[str, str], dict[str, Any]] = {}
        reused = 0
        for subset in SUBSETS:
            for row in read_jsonl(tdc_split_path(task, subset)):
                if _key(row) in existing:
                    reused += 1
                else:
                    missing.setdefault(_key(row), row)
        input_path = output_root / "fresh_inputs" / f"{task}.jsonl"
        write_jsonl_atomic(input_path, list(missing.values()))
        summary[task] = {
            "reused": reused, "fresh": len(missing), "input_jsonl": str(input_path),
            "input_sha256": sha256_file(input_path),
        }
    write_json_atomic(output_root / "prepared.json", {
        "schema_version": "tdc_query_prior_build.v1", "status": "prepared",
        "tasks": summary,
    })
    return summary


def build_commands(output_root: Path, provider_config: Path) -> list[BatchCommand]:
    prepared = json.loads((output_root / "prepared.json").read_text())
    commands = []
    for task in TASKS:
        option, profile = PROFILES[task]
        command = [
            sys.executable, "-m", f"predict.harnesses.branches.tasks.{task}.contract",
            "--input-jsonl", prepared["tasks"][task]["input_jsonl"],
            "--batch-root", str((output_root / "fresh" / task).resolve()),
            "--batch-id", f"{task}__none", "--experiment-mode", "none",
            "--neighbor-identity-policy", "parent_disjoint", "--harness-prefetch-tools",
            "--provider-pool-config", str(provider_config.resolve()),
            "--tool-service-url", "http://127.0.0.1:8765", "--model", MODEL,
            "--reasoning-effort", "high", "--enable-thinking", "--temperature", "0",
            "--max-tokens", "20480", "--timeout-s", "900", "--max-tool-rounds", "3",
            "--execution-mode", "throughput", "--skip-existing", "--no-stream-logs",
            "--no-combine-traces", option, profile,
        ]
        commands.append(BatchCommand(f"{task}__none", command))
    return commands


def publish_overlays(output_root: Path) -> dict[str, Any]:
    prepared = json.loads((output_root / "prepared.json").read_text())
    summary = {}
    for task in TASKS:
        existing = _gold_sources(task)
        fresh_rows = read_jsonl(Path(prepared["tasks"][task]["input_jsonl"]))
        fresh_batch = output_root / "fresh" / task / f"{task}__none"
        fresh = {_key(row): _run(fresh_batch, index) for index, row in enumerate(fresh_rows)}
        for key, run in fresh.items():
            if json.loads((run / "single_molecule_reasoning_output.json").read_text()).get("status") != "ok":
                raise ValueError(f"Fresh query prior is incomplete: {run}")
        for subset in SUBSETS:
            input_path = tdc_split_path(task, subset).with_name(
                f"{subset}_molecule_condition_labels.jsonl"
            ).resolve()
            sources = []
            reused = 0
            for index, row in enumerate(read_jsonl(input_path)):
                key = _key(row)
                run = existing.get(key) or fresh.get(key)
                if run is None:
                    raise ValueError(f"No query prior for {task}/{subset}/{key}")
                reused += int(key in existing)
                files = {
                    name: sha256_file(run / name)
                    for name in ("retrieval.json", "single_molecule_reasoning_output.json")
                }
                profile_manifest, profile = _source_prompt_profile(task, run)
                sources.append({
                    "target_index": index, "molecule_identity_key": key[0],
                    "condition_group": key[1], "run_dir": str(run),
                    "source": "gold_v1" if key in existing else "tdc_fresh",
                    "files_sha256": files,
                    "task_prompt_profile": profile,
                    "prompt_profile_manifest": str(profile_manifest),
                    "prompt_profile_manifest_sha256": sha256_file(profile_manifest),
                })
            batch = output_root / "overlays" / subset / task / f"{task}__none"
            batch.mkdir(parents=True, exist_ok=True)
            write_json_atomic(batch / "manifest.json", {
                "schema_version": "branch_query_prior_overlay.v1", "status": "complete",
                "task_id": task, "subset": subset, "input_jsonl": str(input_path),
                "input_sha256": sha256_file(input_path), "n_items": len(sources),
                "task_prompt_profile": PROFILES[task][1],
                "reused": reused, "fresh": len(sources) - reused, "sources": sources,
            })
            summary[f"{task}/{subset}"] = {"queries": len(sources), "reused": reused, "fresh": len(sources) - reused}
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--provider-pool-config", type=Path,
        default=Path("predict/api_client/providers/current_endpoints.json"),
    )
    parser.add_argument("--parallelism", type=int)
    parser.add_argument("--preparation-workers", type=int, default=64)
    parser.add_argument("--publish-only", action="store_true")
    args = parser.parse_args(argv)
    output_root = (args.output_root or Path(
        "outputs/paper/assay_transfer_harness/joseph/query_priors"
    ) / time.strftime("tdc_v1_deepseek_v4_flash_0731_high_%Y%m%d_%H%M%S")).resolve()
    if args.publish_only:
        print(json.dumps(publish_overlays(output_root), indent=2, sort_keys=True)); return 0
    prepare_inputs(output_root)
    config_path = args.provider_pool_config.resolve()
    candidate = load_provider_pool_config(config_path)
    requested = args.parallelism or primary_capacity(candidate)
    selection = select_healthy_providers(candidate, requested)
    commands = build_commands(output_root, config_path)
    prepared = prepare_batch_commands(commands, max_workers=args.preparation_workers, max_stage_requeues=0)
    receipt = {
        "schema_version": "tdc_query_prior_execution.v1", "status": "running",
        "model": MODEL, "reasoning_effort": "high", "thinking": True,
        "max_tokens": 20480, "timeout_s": 900, "transport_max_retries": 0,
        "requested_parallelism": requested,
        "effective_parallelism": selection.effective_parallelism,
        "provider_pool": selection.public_dict(),
    }
    write_json_atomic(output_root / "execution.json", receipt)
    client = provider_client(
        selection.effective_parallelism, provider_pool_config=config_path,
        config=selection.config, max_tokens=20480, timeout_s=900,
    )
    failed = run_prepared_prompt_pool(
        prepared, max_workers=selection.effective_parallelism,
        max_stage_requeues=0, preparation_workers=args.preparation_workers,
        stage_client=client,
    )
    receipt["status"] = "complete" if not failed else "incomplete"
    receipt["failed_batches"] = failed
    receipt["provider_snapshot"] = client.snapshot()
    if not failed:
        receipt["overlays"] = publish_overlays(output_root)
    write_json_atomic(output_root / "execution.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
