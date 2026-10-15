"""Generate missing TDC query priors and publish stable-identity overlays.

Existing Gold-v1 valid/test priors are reused by molecule-condition identity.
Only TDC queries without an exact-identity, exact-SMILES prior are sent to the
configured DeepSeek provider pool. The
published overlay pins every source artifact by hash for positional consumers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from data.processing.gold_labels.conditioned_benchmark import tdc_split_path
from predict.api_client.transport import provider_from_base_url
from predict.api_client.pool import (
    load_provider_pool_config, primary_capacity, select_healthy_providers,
)
from predict.harnesses.branches.flat import (
    CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
    default_query_prior_root,
)
from predict.harnesses.branches.matrix import provider_client
from predict.harnesses.branches.scheduler import (
    BatchCommand, prepare_batch_commands, run_prepared_prompt_pool,
)
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from predict.utils.response import validate_query_prior_output
from predict.harnesses.branches.manifests import (
    read_record, read_run, run_directory, prior_sources, write_prior_batch,
    PREDICTION_ROOT, prediction_run_root, prediction_batch_directory, canonical_location, query_prior_batch,
)


TASKS = (
    "bbb_martins", "bioavailability_ma", "skin_reaction",
    "ames", "dili", "carcinogens",
)
SAFETY_TASKS = frozenset(("ames", "dili", "carcinogens"))
SUBSETS = ("valid", "test")
PROFILES = {
    "bbb_martins": ("--bbb-prompt-profile", "meaningful_cns_access_v1"),
    "bioavailability_ma": ("--bioavailability-prompt-profile", "f20_evidence_calibrated_v2"),
    "skin_reaction": ("--skin-prompt-profile", "skin_sensitization_contact_allergy.v2"),
    "ames": ("--ames-prompt-profile", "ames_bacterial_reverse_mutation.v1"),
    "dili": ("--dili-prompt-profile", "dili_conditioned_or_source_molecule_outcome.v1"),
    "carcinogens": ("--carcinogens-prompt-profile", "carcinogens_starling_only_five_organism_groups.v1"),
}
EXISTING_TDC_PRIORS = (
    Path(
        "outputs/paper/assay_transfer_harness/joseph/query_priors/"
        "tdc_v1_deepseek_v4_flash_0731_high_20260920_1055/overlays"
    ),
    Path(
        "outputs/paper/assay_transfer_harness/joseph/query_priors/"
        "tdc_v1_valid_skin_upstream_v1_20260922_r3/overlays"
    ),
    Path("outputs/paper/assay_transfer_harness/joseph/query_priors/tdc_safety_valid_upstream_20260923_v1"),
    Path("outputs/paper/assay_transfer_harness/joseph/query_priors/tdc_safety_test_upstream_20260923_v1"),
)


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["molecule_identity_key"]), str(row.get("condition_group") or "")


def _run(batch: Path, index: int) -> Path:
    return run_directory(batch, index)


def _source_prompt_profile(task: str, run: Path) -> tuple[Path, str]:
    run = canonical_location(run)
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = run.parent.parent / "manifest.json"
    manifest = read_record(manifest_path)
    profile = str(manifest.get("task_prompt_profile") or "")
    expected = PROFILES[task][1]
    if profile != expected:
        raise ValueError(
            f"Query-prior prompt profile differs for {run}: {profile!r} != {expected!r}"
        )
    return manifest_path, expected


def _gold_sources(task: str) -> dict[tuple[str, str], Path]:
    sources: dict[tuple[str, str], Path] = {}
    for subset in SUBSETS:
        root = default_query_prior_root(
            CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION, "gold", subset
        )
        batch = query_prior_batch(root, task, subset, "gold")
        manifest = read_record(batch / "manifest.json")
        for row in prior_sources(batch):
            key = (str(row["molecule_identity_key"]), str(row.get("condition_group") or ""))
            run = canonical_location(row["run_dir"])
            retrieval = run / "retrieval.json"
            single = run / "single_molecule_reasoning_output.json"
            if not retrieval.is_file() or not single.is_file():
                raise ValueError(f"Gold query prior is incomplete: {run}")
            if key in sources:
                raise ValueError(f"Gold query-prior identity repeats across splits: {key}")
            sources[key] = run
    return sources


def _tdc_sources(task: str, subsets: tuple[str, ...]) -> dict[tuple[str, str], Path]:
    sources = {}
    for root in EXISTING_TDC_PRIORS:
        for subset in subsets:
            manifest_paths = (
                root / subset / task / f"{task}__none/manifest.json",
                root / task / f"{task}__none/manifest.json",
            )
            for manifest_path in manifest_paths:
                manifest_path = canonical_location(manifest_path)
                if not manifest_path.is_file():
                    continue
                manifest = read_record(manifest_path)
                if manifest.get("subset") not in (None, subset):
                    continue
                for row in prior_sources(manifest_path.parent):
                    key = (str(row["molecule_identity_key"]), str(row.get("condition_group") or ""))
                    run = canonical_location(row["run_dir"])
                    try:
                        _source_prompt_profile(task, run)
                    except ValueError:
                        continue
                    sources[key] = run
    return sources


def _matching_source(
    row: dict[str, Any], gold: dict[tuple[str, str], Path],
    tdc: dict[tuple[str, str], Path],
) -> tuple[str, Path] | None:
    for source, candidates in (("tdc_existing", tdc), ("gold_v1", gold)):
        run = candidates.get(_key(row))
        if run is None:
            continue
        retrieval = json.loads((run / "retrieval.json").read_text(encoding="utf-8"))
        if str((retrieval.get("query") or {}).get("input_smiles") or "") == str(row["drug"]):
            try:
                validate_query_prior_output(json.loads((run / "single_molecule_reasoning_output.json").read_text()))
            except (ValueError, FileNotFoundError):
                continue
            return source, run
    return None


def prepare_inputs(
    output_root: Path, tasks: tuple[str, ...], subsets: tuple[str, ...],
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    summary = {}
    source_inputs = []
    for task in tasks:
        gold, tdc = _gold_sources(task), _tdc_sources(task, subsets)
        missing: dict[tuple[tuple[str, str], str], dict[str, Any]] = {}
        reused = 0
        for subset in subsets:
            split_path = tdc_split_path(task, subset)
            label_path = split_path.with_name(f"{subset}_molecule_condition_labels.jsonl")
            for path in (split_path, label_path):
                source_inputs.append({"task": task, "subset": subset,
                    "path": str(path.resolve()), "sha256": sha256_file(path)})
            for row in read_jsonl(split_path):
                if _matching_source(row, gold, tdc):
                    reused += 1
                else:
                    missing.setdefault((_key(row), str(row["drug"])), row)
        input_path = output_root / "fresh_inputs" / f"{task}.jsonl"
        write_jsonl_atomic(input_path, list(missing.values()))
        summary[task] = {
            "reused": reused, "fresh": len(missing), "input_jsonl": str(input_path),
            "input_sha256": sha256_file(input_path),
        }
    write_json_atomic(output_root / "prepared.json", {
        "schema_version": "tdc_query_prior_build.v1", "status": "prepared",
        "task_ids": list(tasks), "subsets": list(subsets),
        "source_inputs": source_inputs, "tasks": summary,
    })
    return summary


def load_prepared_inputs(
    output_root: Path, tasks: tuple[str, ...], subsets: tuple[str, ...],
) -> dict[str, Any]:
    """Reuse a prepared request set only when its benchmark and request bytes match."""
    prepared = json.loads((output_root / "prepared.json").read_text(encoding="utf-8"))
    if prepared.get("schema_version") != "tdc_query_prior_build.v1":
        raise ValueError(f"unsupported prepared TDC query-prior artifact: {output_root}")
    if prepared.get("task_ids") != list(tasks) or prepared.get("subsets") != list(subsets):
        raise ValueError("prepared TDC query-prior task/split selection differs; use a fresh output root")
    expected_sources = {}
    for task in tasks:
        for subset in subsets:
            split_path = tdc_split_path(task, subset)
            label_path = split_path.with_name(f"{subset}_molecule_condition_labels.jsonl")
            expected_sources[str(split_path.resolve())] = sha256_file(split_path)
            expected_sources[str(label_path.resolve())] = sha256_file(label_path)
    recorded_sources = {
        str(item["path"]): str(item["sha256"])
        for item in prepared.get("source_inputs", [])
    }
    if recorded_sources != expected_sources:
        raise ValueError("prepared TDC query-prior source set or hashes changed; use a fresh output root")
    for task in tasks:
        item = prepared["tasks"][task]
        path = Path(item["input_jsonl"])
        if path.resolve() != (output_root / "fresh_inputs" / f"{task}.jsonl").resolve():
            raise ValueError(f"prepared fresh-input path is outside the run root: {path}")
        if not path.is_file() or sha256_file(path) != item["input_sha256"]:
            raise ValueError(f"prepared fresh requests changed: {path}")
    return prepared["tasks"]


def build_commands(
    output_root: Path, provider_config: Path, tasks: tuple[str, ...],
) -> list[BatchCommand]:
    prepared = json.loads((output_root / "prepared.json").read_text())
    provider = load_provider_pool_config(provider_config).providers[0]
    local = provider_from_base_url(provider.base_url) == "local"
    commands = []
    for task in tasks:
        if task in SAFETY_TASKS or not int(prepared["tasks"][task]["fresh"]):
            continue
        option, profile = PROFILES[task]
        command = [
            sys.executable, "-m", f"predict.harnesses.branches.tasks.{task}.contract",
            "--input-jsonl", prepared["tasks"][task]["input_jsonl"],
            "--batch-root", str((output_root / "fresh" / task).resolve()),
            "--batch-id", f"{task}__none", "--experiment-mode", "none",
            "--neighbor-identity-policy", "parent_disjoint", "--harness-prefetch-tools",
            "--provider-pool-config", str(provider_config.resolve()),
            "--tool-service-url", "http://127.0.0.1:8765", "--model", provider.model,
            "--reasoning-effort", "high",
            "--enable-thinking" if local else "--disable-thinking",
            "--temperature", "none" if provider.model == "openrouter/auto" else "0",
            "--max-tokens", "20480", "--timeout-s", str(provider.timeout_s or 900),
            "--max-tool-rounds", "3",
            "--execution-mode", "throughput", "--skip-existing", "--no-stream-logs",
            "--no-combine-traces", option, profile,
        ]
        commands.append(BatchCommand(f"{task}__none", command))
    return commands


def publish_overlays(
    output_root: Path, tasks: tuple[str, ...], subsets: tuple[str, ...],
) -> dict[str, Any]:
    prepared = json.loads((output_root / "prepared.json").read_text())
    summary = {}
    for task in tasks:
        gold = _gold_sources(task)
        tdc = _tdc_sources(task, subsets)
        fresh_rows = read_jsonl(Path(prepared["tasks"][task]["input_jsonl"]))
        fresh_root = "fresh_direct" if task in SAFETY_TASKS else "fresh"
        fresh_batch = output_root / fresh_root / task / f"{task}__none"
        fresh = {(_key(row), str(row["drug"])): _run(fresh_batch, index)
                 for index, row in enumerate(fresh_rows)}
        for key, run in fresh.items():
            validate_query_prior_output(json.loads((run / "single_molecule_reasoning_output.json").read_text()))
        for subset in subsets:
            input_path = tdc_split_path(task, subset).with_name(
                f"{subset}_molecule_condition_labels.jsonl"
            ).resolve()
            sources = []
            reused = 0
            for index, row in enumerate(read_jsonl(input_path)):
                key = _key(row)
                matching = _matching_source(row, gold, tdc)
                source, run = matching if matching else (
                    "tdc_fresh", fresh.get((key, str(row["drug"])))
                )
                if run is None:
                    raise ValueError(f"No query prior for {task}/{subset}/{key}")
                reused += int(matching is not None)
                profile_manifest, profile = _source_prompt_profile(task, run)
                sources.append({
                    "target_index": index, "molecule_identity_key": key[0],
                    "condition_group": key[1], "run_dir": str(run),
                    "source": source,
                    "benchmark_row_id": row["benchmark_row_id"],
                    "source_run_uid": read_run(run).get("run_uid", ""),
                    "task_prompt_profile": profile,
                    "prompt_profile_manifest": str(profile_manifest),

                })
            batch = prediction_batch_directory(output_root, "tdc", task, subset, "query-prior")
            batch.mkdir(parents=True, exist_ok=True)
            write_prior_batch(batch, {
                "schema_version": "branch_query_prior_overlay.v1", "status": "complete",
                "task_id": task, "subset": subset, "input_jsonl": str(input_path),
                "benchmark": "tdc", "harness_version": "query-prior",
                "prompt_version": CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
                "n_items": len(sources),
                "task_prompt_profile": PROFILES[task][1],
                "reused": reused, "fresh": len(sources) - reused, "sources": sources,
            })
            summary[f"{task}/{subset}"] = {"queries": len(sources), "reused": reused, "fresh": len(sources) - reused}
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PREDICTION_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--provider-pool-config", type=Path,
        default=Path("predict/api_client/providers/current_endpoints.json"),
    )
    parser.add_argument("--parallelism", type=int)
    parser.add_argument("--preparation-workers", type=int, default=64)
    parser.add_argument("--max-stage-requeues", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--publish-only", action="store_true")
    args = parser.parse_args(argv)
    if args.max_stage_requeues < 0:
        parser.error("--max-stage-requeues must be non-negative")
    if not args.run_name and not args.publish_only and not (args.output_root / "prepared.json").exists():
        parser.error("New query-prior runs require --run-name")
    output_root = (prediction_run_root(args.output_root, "query-prior",
                                       CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION, args.run_name)
                   if args.run_name else canonical_location(args.output_root))
    tasks = tuple(dict.fromkeys(args.tasks))
    subsets = tuple(dict.fromkeys(args.subsets))
    if args.publish_only:
        overlays = publish_overlays(output_root, tasks, subsets)
        receipt_path = output_root / "execution.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt.update(status="complete", failed_batches=[], overlays=overlays)
            write_json_atomic(receipt_path, receipt)
        print(json.dumps(overlays, indent=2, sort_keys=True))
        return 0
    if (output_root / "prepared.json").is_file():
        prepared = load_prepared_inputs(output_root, tasks, subsets)
    else:
        prepare_inputs(output_root, tasks, subsets)
        prepared = load_prepared_inputs(output_root, tasks, subsets)
    if args.prepare_only:
        print(json.dumps(prepared, indent=2, sort_keys=True))
        return 0
    config_path = args.provider_pool_config.resolve()
    candidate = load_provider_pool_config(config_path)
    requested = args.parallelism or primary_capacity(candidate)
    selection = select_healthy_providers(candidate, requested)
    commands = build_commands(output_root, config_path, tasks)
    batches = prepare_batch_commands(
        commands, max_workers=args.preparation_workers,
        max_stage_requeues=args.max_stage_requeues,
    )
    receipt = {
        "schema_version": "tdc_query_prior_execution.v1", "status": "running",
        "model": selection.config.providers[0].model, "reasoning_effort": "high",
        "thinking": provider_from_base_url(selection.config.providers[0].base_url) == "local",
        "max_tokens": 20480, "timeout_s": 900, "transport_max_retries": 0,
        "max_stage_requeues": args.max_stage_requeues,
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
        batches, max_workers=selection.effective_parallelism,
        max_stage_requeues=args.max_stage_requeues,
        preparation_workers=args.preparation_workers,
        stage_client=client,
    )
    if not failed:
        safety = [task for task in tasks if task in SAFETY_TASKS and int(prepared[task]["fresh"])]
        if safety:
            from predict.harnesses.branches.six_task_query_priors import main as run_direct_priors
            try:
                run_direct_priors([
                    "--output-root", str(output_root / "fresh_direct"),
                    "--input-root", str(output_root / "fresh_inputs"),
                    "--subset", "fresh", "--benchmark", "tdc",
                    "--tasks", *safety, "--prompt-version",
                    CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
                    "--provider-pool-config", str(config_path),
                    "--parallelism", str(selection.effective_parallelism),
                    "--resume",
                ])
            except Exception as exc:
                receipt.update(status="incomplete", safety_query_prior_error=f"{type(exc).__name__}: {exc}")
                write_json_atomic(output_root / "execution.json", receipt)
                raise
    receipt["status"] = "complete" if not failed else "incomplete"
    receipt["failed_batches"] = failed
    receipt["provider_snapshot"] = client.snapshot()
    if not failed:
        receipt["overlays"] = publish_overlays(output_root, tasks, subsets)
    write_json_atomic(output_root / "execution.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
