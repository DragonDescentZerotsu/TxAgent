"""Run a V11 assay-transfer scaffold-validation experiment.

The launcher covers the current BBB_Martins, Bioavailability_Ma, and
Skin_Reaction scaffold-valid lineages with their paper-local Stage 06-09
artifacts and compact V11 caches.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from tools.chembl_tool.common.assay_reranking.v11 import (
    BACKBONE_DTYPE,
    LOGIT_EXTRACTION_DTYPE,
    PROFILE_NAME,
    SCORING_CONTRACT_VERSION,
    TEMPLATE_PROFILE,
    file_sha256,
    model_profile,
)
from tools.chembl_tool.common.assay_transfer_selection import (
    ASSAY_TRANSFER_DIVERSITY_MODES,
    ASSAY_TRANSFER_DIVERSITY_NONE,
    ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
    ASSAY_TRANSFER_SELECTION_MEAN_SCORE_MOLECULE,
    ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE,
    ASSAY_TRANSFER_SELECTION_UNITS,
    validate_assay_transfer_diversity,
    validate_assay_transfer_records_per_molecule,
)
from tools.chembl_tool.common.json_utils import write_json_atomic
from tools.chembl_tool.common.task_workflows.global_prompt_pool import (
    BatchCommand,
    SCHEDULER_VERSION,
    run_global_prompt_pool,
)
from tools.chembl_tool.paper_experiments.molecular_evidence_agent import (
    GLM_API_KEY_ENV,
    GLM_MODEL,
    GLM_REASONING_EFFORT,
    ensure_endpoint_api_key,
    resolve_endpoint_base_url,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = Path("outputs/paper/v11_assay_transfer_scaffold_valid_launcher")
UNIQUE_MOLECULE_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT.with_name(
    "assay_transfer_v11_k3_unique_molecules_scored_assay_schema"
)
RUNS_DIR = "runs_identity_blind_parent_disjoint"
LAUNCH_CONTRACT_VERSION = "v11_assay_transfer_scaffold_valid.v4"
EXPECTED_TOOL_COUNT = 3
ENDPOINT_CONCURRENCY_BUDGET = 500
RETRIEVAL_CONDITION_MORGAN = "morgan"
RETRIEVAL_CONDITION_ASSAY_TRANSFER_RECORD = "assay_transfer_record"
RETRIEVAL_CONDITION_ASSAY_TRANSFER_MOLECULE = "assay_transfer_molecule"
RETRIEVAL_CONDITIONS = (
    RETRIEVAL_CONDITION_MORGAN,
    RETRIEVAL_CONDITION_ASSAY_TRANSFER_RECORD,
    RETRIEVAL_CONDITION_ASSAY_TRANSFER_MOLECULE,
)
MORGAN_K3_CONTROL_ROOT = Path(
    "outputs/paper/"
    "molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid/"
    "morgan_similarity_k3_no_floor/runs_identity_blind_parent_disjoint"
)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    data_name: str
    batch_module: str
    expected_queries: int
    expected_scores: int
    input_jsonl: str
    index: str
    cache: str
    version: str
    paper_output_root: str
    group_output_schema: str = ""
    group_prompt_version: str = ""
    single_analysis_source_batch: str = ""

    @property
    def batch_id(self) -> str:
        suffix = "_assay_schema" if self.group_output_schema else ""
        return (
            f"{self.task_id}__starling_full_mechanism__"
            f"assay_transfer_v11_scored{suffix}_k3_morgan50"
        )


TASKS = (
    TaskSpec(
        task_id="bbb_martins",
        data_name="BBB_Martins",
        batch_module="tools.chembl_tool.tasks.bbb_martins.run_reasoning_batch",
        expected_queries=366,
        expected_scores=633_694,
        input_jsonl="data/processed_starling_experimental_meaningful_cns_access_v2/BBB_Martins/scaffold/valid.jsonl",
        index="outputs/paper/molecular_evidence_agent_starling_scaffold_experimental_meaningful_cns_access_v2/evidence/bbb_starling_v7/08_neighbor_index",
        cache="outputs/paper/molecular_evidence_agent_starling_scaffold_experimental_meaningful_cns_access_v2/assay_transfer_rerank/bbb_starling_v7/v11_with_categorical/scaffold/valid/scores.sqlite3",
        version="outputs/paper/molecular_evidence_agent_starling_scaffold_experimental_meaningful_cns_access_v2/assay_transfer_rerank/bbb_starling_v7/v11_with_categorical/scaffold/valid/VERSION.json",
        paper_output_root="outputs/paper/molecular_evidence_agent_starling_scaffold_experimental_meaningful_cns_access_v2",
        single_analysis_source_batch=str(
            MORGAN_K3_CONTROL_ROOT / "bbb_martins" / "bbb_martins__none"
        ),
    ),
    TaskSpec(
        task_id="bioavailability_ma",
        data_name="Bioavailability_Ma",
        batch_module="tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch",
        expected_queries=209,
        expected_scores=870_332,
        input_jsonl="data/processed_starling_record_supported_v2/Bioavailability_Ma/scaffold/valid.jsonl",
        index="outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2/evidence/bioavailability_starling_v7/08_neighbor_index",
        cache="outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2/assay_transfer_rerank/bioavailability_starling_v7/v11_with_categorical/scaffold/valid/scores.sqlite3",
        version="outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2/assay_transfer_rerank/bioavailability_starling_v7/v11_with_categorical/scaffold/valid/VERSION.json",
        paper_output_root="outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2",
        group_output_schema="assay-transfer",
        group_prompt_version="bioavailability_text_v1",
    ),
    TaskSpec(
        task_id="skin_reaction",
        data_name="Skin_Reaction",
        batch_module="tools.chembl_tool.tasks.skin_reaction.run_reasoning_batch",
        expected_queries=245,
        expected_scores=254_657,
        input_jsonl="data/processed_starling_record_supported_v2/Skin_Reaction/scaffold/valid.jsonl",
        index="outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2/evidence/skin_reaction_starling_v7/08_neighbor_index",
        cache="outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2/assay_transfer_rerank/skin_reaction_starling_v7/v11_with_categorical/scaffold/valid/scores.sqlite3",
        version="outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2/assay_transfer_rerank/skin_reaction_starling_v7/v11_with_categorical/scaffold/valid/VERSION.json",
        paper_output_root="outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2",
    ),
)
ACTIVE_TASKS = TASKS
DEFAULT_TASK_IDS = ("bbb_martins", "bioavailability_ma", "skin_reaction")


def _selected_tasks(args: argparse.Namespace | None = None) -> tuple[TaskSpec, ...]:
    requested = set(getattr(args, "tasks", DEFAULT_TASK_IDS))
    return tuple(task for task in ACTIVE_TASKS if task.task_id in requested)


def _default_output_root(
    top_k: int,
    selection_unit: str,
    experiment_mode: str = "full_flat",
    diversity_mode: str = ASSAY_TRANSFER_DIVERSITY_NONE,
    records_per_molecule: int = ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
    disable_flat_tools: bool = False,
    analogous_reasoning_only: bool = False,
) -> Path:
    suffix = (
        "unique_molecules_scored_assay_schema"
        if selection_unit == ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE
        else "mean_score_molecules_scored_assay_schema"
        if selection_unit == ASSAY_TRANSFER_SELECTION_MEAN_SCORE_MOLECULE
        else "scored_assay_schema"
    )
    if diversity_mode != ASSAY_TRANSFER_DIVERSITY_NONE:
        suffix = suffix.replace(
            "scored_assay_schema", f"{diversity_mode}_diversity_scored_assay_schema"
        )
    if records_per_molecule > ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT:
        suffix = f"{suffix}_r{records_per_molecule}"
    if disable_flat_tools:
        suffix = f"{suffix}_no_flat_tools"
    if analogous_reasoning_only:
        suffix = f"{suffix}_analogous_flat_v1"
    return DEFAULT_OUTPUT_ROOT / f"{experiment_mode}_k{top_k}_{suffix}"


def _task_batch_id(
    task: TaskSpec,
    selection_unit: str,
    experiment_mode: str = "full_flat",
    top_k: int = 3,
    diversity_mode: str = ASSAY_TRANSFER_DIVERSITY_NONE,
    records_per_molecule: int = ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
) -> str:
    batch_id = task.batch_id.replace("full_mechanism", experiment_mode).replace(
        "_k3_morgan50", f"_k{top_k}_morgan50"
    )
    markers = []
    if selection_unit == ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE:
        markers.append("unique_molecule")
    elif selection_unit == ASSAY_TRANSFER_SELECTION_MEAN_SCORE_MOLECULE:
        markers.append("mean_score_molecule")
    if diversity_mode != ASSAY_TRANSFER_DIVERSITY_NONE:
        markers.append(f"{diversity_mode}_diversity")
    if records_per_molecule > ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT:
        markers.append(f"r{records_per_molecule}")
    if not markers:
        return batch_id
    return batch_id.replace(
        f"_k{top_k}_morgan50", f"_{'_'.join(markers)}_k{top_k}_morgan50"
    )


def _task_diversity_score_slack(
    task: TaskSpec, args: argparse.Namespace
) -> float:
    return {
        "bbb_martins": args.bbb_diversity_score_slack,
        "bioavailability_ma": args.bioavailability_diversity_score_slack,
        "skin_reaction": args.skin_reaction_diversity_score_slack,
    }[task.task_id]


@dataclass
class ToolService:
    process: subprocess.Popen[str] | None = None
    log_handle: Any = None
    reused: bool = False

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if self.process.poll() is None:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=10)
        if self.log_handle is not None:
            self.log_handle.close()


def build_batch_commands(args: argparse.Namespace) -> list[BatchCommand]:
    """Build the three lineage-local batch commands consumed by the global pool."""
    if args.retrieval_conditions:
        return _build_retrieval_condition_commands(args)
    commands: list[BatchCommand] = []
    for task in _selected_tasks(args):
        profile = model_profile(task.task_id)
        condition = Path(args.output_root).name
        batch_root = Path(task.paper_output_root) / "assay_transfer_v11_glm" / condition / RUNS_DIR
        batch_id = _task_batch_id(
            task,
            args.assay_transfer_selection_unit,
            args.experiment_mode,
            args.top_k_per_group,
            args.assay_transfer_diversity_mode,
            args.assay_transfer_records_per_molecule,
        )
        command = [
            sys.executable,
            "-m",
            task.batch_module,
            "--input-jsonl",
            task.input_jsonl,
            "--index",
            task.index,
            "--experiment-mode",
            args.experiment_mode,
            "--retrieval-source",
            "starling",
            "--neighbor-identity-policy",
            "parent_disjoint",
            "--identity-blind",
            "--retrieval-strategy",
            "assay_transfer_tool",
            "--assay-transfer-profile",
            PROFILE_NAME,
            "--assay-transfer-selection-unit",
            args.assay_transfer_selection_unit,
            "--assay-transfer-records-per-molecule",
            str(args.assay_transfer_records_per_molecule),
            "--assay-transfer-diversity-mode",
            args.assay_transfer_diversity_mode,
            "--assay-transfer-diversity-score-slack",
            str(_task_diversity_score_slack(task, args)),
            "--assay-transfer-initial-morgan-filter",
            "50",
            "--rerank-catalog",
            "",
            "--rerank-cache",
            task.cache,
            "--rerank-candidate-manifest",
            "",
            "--rerank-cache-version-manifest",
            task.version,
            "--rerank-expected-score-count",
            str(task.expected_scores),
            "--rerank-cache-mode",
            "read_only",
            "--assay-transfer-model",
            str(profile["model"]),
            "--assay-transfer-model-revision",
            str(profile["revision"]),
            "--assay-transfer-template-profile",
            TEMPLATE_PROFILE,
            "--group-prompt-format",
            "assay_transfer_tool",
            "--enable-assay-transfer-scores",
            "--top-k-per-group",
            str(args.top_k_per_group),
            "--min-similarity",
            "0.0",
            "--neighbor-context-profile",
            "standard",
            "--morgan-neighbor-selector",
            "similarity",
            "--batch-root",
            str(batch_root),
            "--batch-id",
            batch_id,
            "--api-key-env",
            args.api_key_env,
            "--base-url",
            args.base_url,
            "--tool-service-url",
            args.tool_service_url,
            "--model",
            args.model,
            "--disable-thinking",
            "--reasoning-effort",
            args.reasoning_effort,
            "--temperature",
            "0",
            "--max-tokens",
            "20480",
            "--timeout-s",
            "300",
            "--max-tool-rounds",
            "3",
            "--parallelism",
            str(args.parallelism),
            "--max-stage-requeues",
            str(args.max_stage_requeues),
            "--skip-existing",
            "--no-stream-logs",
            "--no-combine-traces",
        ]
        if args.limit:
            command.extend(["--limit", str(args.limit)])
        if task.group_output_schema and not args.analogous_reasoning_only:
            command.extend(["--group-output-schema", task.group_output_schema])
        if task.group_prompt_version and not args.analogous_reasoning_only:
            command.extend(["--group-prompt-version", task.group_prompt_version])
        if args.disable_flat_tools:
            command.append("--disable-flat-tools")
        if args.analogous_reasoning_only:
            command.append("--analogous-reasoning-only")
        commands.append(BatchCommand(batch_id, command))
    return commands


def _build_retrieval_condition_commands(
    args: argparse.Namespace,
) -> list[BatchCommand]:
    commands: list[BatchCommand] = []
    for condition_name in args.retrieval_conditions:
        for task in _selected_tasks(args):
            commands.append(
                _retrieval_condition_command(args, task, condition_name)
            )
    return commands


def _retrieval_condition_command(
    args: argparse.Namespace,
    task: TaskSpec,
    condition_name: str,
) -> BatchCommand:
    condition_root = Path(args.output_root).name
    batch_root = (
        Path(task.paper_output_root)
        / "analogous_flat_v1_retrieval_comparison"
        / condition_root
        / RUNS_DIR
    )
    batch_id = _retrieval_condition_batch_id(
        task, condition_name, args.top_k_per_group
    )
    command = [
        sys.executable,
        "-m",
        task.batch_module,
        "--input-jsonl",
        task.input_jsonl,
        "--index",
        task.index,
        "--experiment-mode",
        "full_flat",
        "--retrieval-source",
        "starling",
        "--neighbor-identity-policy",
        "parent_disjoint",
        "--identity-blind",
        "--top-k-per-group",
        str(args.top_k_per_group),
        "--min-similarity",
        "0.0",
        "--neighbor-context-profile",
        "standard",
        "--morgan-neighbor-selector",
        "similarity",
        "--batch-root",
        str(batch_root),
        "--batch-id",
        batch_id,
        "--api-key-env",
        args.api_key_env,
        "--base-url",
        args.base_url,
        "--tool-service-url",
        args.tool_service_url,
        "--model",
        args.model,
        "--disable-thinking",
        "--reasoning-effort",
        args.reasoning_effort,
        "--temperature",
        "0",
        "--max-tokens",
        "20480",
        "--timeout-s",
        "300",
        "--max-tool-rounds",
        "3",
        "--parallelism",
        str(args.parallelism),
        "--max-stage-requeues",
        str(args.max_stage_requeues),
        "--skip-existing",
        "--no-stream-logs",
        "--no-combine-traces",
        "--analogous-reasoning-only",
    ]
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    if condition_name == RETRIEVAL_CONDITION_MORGAN:
        command.extend(["--retrieval-strategy", "morgan_fingerprint"])
    else:
        profile = model_profile(task.task_id)
        selection_unit = (
            ASSAY_TRANSFER_SELECTION_SCORED_RECORD
            if condition_name == RETRIEVAL_CONDITION_ASSAY_TRANSFER_RECORD
            else ASSAY_TRANSFER_SELECTION_MEAN_SCORE_MOLECULE
        )
        records_per_molecule = (
            ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT
            if selection_unit == ASSAY_TRANSFER_SELECTION_SCORED_RECORD
            else 6
        )
        command.extend(
            [
                "--retrieval-strategy",
                "assay_transfer_tool",
                "--assay-transfer-profile",
                PROFILE_NAME,
                "--assay-transfer-selection-unit",
                selection_unit,
                "--assay-transfer-records-per-molecule",
                str(records_per_molecule),
                "--assay-transfer-diversity-mode",
                ASSAY_TRANSFER_DIVERSITY_NONE,
                "--assay-transfer-diversity-score-slack",
                "0.0",
                "--assay-transfer-initial-morgan-filter",
                "50",
                "--rerank-catalog",
                "",
                "--rerank-cache",
                task.cache,
                "--rerank-candidate-manifest",
                "",
                "--rerank-cache-version-manifest",
                task.version,
                "--rerank-expected-score-count",
                str(task.expected_scores),
                "--rerank-cache-mode",
                "read_only",
                "--assay-transfer-model",
                str(profile["model"]),
                "--assay-transfer-model-revision",
                str(profile["revision"]),
                "--assay-transfer-template-profile",
                TEMPLATE_PROFILE,
                "--group-prompt-format",
                "assay_transfer_tool",
                "--enable-assay-transfer-scores",
            ]
        )
        if (batch_root / batch_id / "manifest.json").exists():
            command.append("--reuse-existing-rerank-preflight")
    return BatchCommand(batch_id, command)


def _retrieval_condition_batch_id(
    task: TaskSpec, condition_name: str, top_k: int
) -> str:
    return (
        f"{task.task_id}__starling_full_flat__{condition_name}__"
        f"k{top_k}__analogous_flat_v1"
    )


def _retrieval_condition_batch_dir(
    args: argparse.Namespace, task: TaskSpec, condition_name: str
) -> Path:
    return (
        Path(task.paper_output_root)
        / "analogous_flat_v1_retrieval_comparison"
        / Path(args.output_root).name
        / RUNS_DIR
        / _retrieval_condition_batch_id(task, condition_name, args.top_k_per_group)
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_inputs(
    tasks: tuple[TaskSpec, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fail before endpoint work if a cache/index no longer matches its build."""
    validated: dict[str, dict[str, Any]] = {}
    for task in tasks or _selected_tasks():
        required = [Path(task.input_jsonl), Path(task.index), Path(task.cache), Path(task.version)]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise SystemExit("Missing V11 run inputs:\n" + "\n".join(missing))
        version = _read_json(Path(task.version))
        profile = model_profile(task.task_id)
        expected = {
            "status": "complete",
            "task_id": task.task_id,
            "benchmark_split": "scaffold",
            "evaluation_subset": "valid",
            "profile": PROFILE_NAME,
            "template_profile": TEMPLATE_PROFILE,
            "model": profile["model"],
            "model_revision": profile["revision"],
            "n_queries": task.expected_queries,
            "n_prompt_scores": task.expected_scores,
            "assay_transfer_initial_morgan_filter": 50,
            "neighbor_identity_policy": "parent_disjoint",
            "min_similarity": 0.0,
            "backbone_dtype": BACKBONE_DTYPE,
            "logit_extraction_dtype": LOGIT_EXTRACTION_DTYPE,
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "stage04_or_stage05_filter_applied": False,
        }
        mismatches = {
            field: {"expected": value, "observed": version.get(field)}
            for field, value in expected.items()
            if version.get(field) != value
        }
        if mismatches:
            raise SystemExit(
                f"V11 VERSION mismatch for {task.task_id}: "
                + json.dumps(mismatches, sort_keys=True)
            )
        input_hash = (version.get("inputs") or {}).get(task.input_jsonl)
        observed_hash = file_sha256(task.input_jsonl)
        if input_hash != observed_hash:
            raise SystemExit(f"Query input hash mismatch for {task.task_id}")
        validated[task.task_id] = {
            "version_path": task.version,
            "version_sha256": file_sha256(task.version),
            "input_jsonl_sha256": observed_hash,
            "cache_quick_check": version.get("cache_quick_check"),
            **expected,
        }
    return validated


def _request_json(url: str, *, api_key: str = "", timeout: float = 10) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _check_endpoint(args: argparse.Namespace) -> dict[str, Any]:
    ensure_endpoint_api_key(args.api_key_env, args.base_url)
    payload = _request_json(
        f"{args.base_url.rstrip('/')}/models",
        api_key=os.environ[args.api_key_env],
        timeout=30,
    )
    model_ids = sorted(str(row.get("id")) for row in payload.get("data", []))
    if args.model not in model_ids:
        raise SystemExit(f"Endpoint does not serve required model: {args.model}")
    return {"status": "ok", "base_url": args.base_url, "served_model": args.model}


def _healthy_tool_service(url: str) -> dict[str, Any] | None:
    try:
        payload = _request_json(f"{url.rstrip('/')}/health", timeout=10)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None
    if (
        payload.get("status") == "ok"
        and int(payload.get("tools_total") or 0) == EXPECTED_TOOL_COUNT
        and int(payload.get("tools_initialized") or 0) == EXPECTED_TOOL_COUNT
        and not payload.get("tools_with_initialization_errors")
    ):
        return payload
    raise SystemExit(f"Tool service is reachable but unhealthy: {payload}")


def _start_or_reuse_tool_service(args: argparse.Namespace) -> ToolService:
    healthy = _healthy_tool_service(args.tool_service_url)
    if healthy is not None:
        print(f"[v11-launcher] reusing healthy tool service at {args.tool_service_url}", flush=True)
        return ToolService(reused=True)
    parsed = urlparse(args.tool_service_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise SystemExit("Automatic tool-service startup requires a loopback HTTP URL")
    port = parsed.port or 80
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.tool_cache_path)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SystemExit(
            f"Cannot create tool-service cache directory {cache_path.parent}: {exc}"
        ) from exc
    if not os.access(cache_path.parent, os.W_OK):
        raise SystemExit(
            f"Tool-service cache directory is not writable: {cache_path.parent}"
        )
    log_handle = (output_root / "tool_service.log").open("a", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "TXAGENT_TOOL_SERVICE_HOST": "127.0.0.1",
            "TXAGENT_TOOL_SERVICE_PORT": str(port),
            "TXAGENT_TOOL_NATIVE_THREADS": "1",
            "TXAGENT_TOOL_BATCH_WORKERS": "8",
            "TXAGENT_TOOL_CACHE_MEMORY_ENTRIES": "5000",
            "TXAGENT_TOOL_CACHE_PATH": args.tool_cache_path,
        }
    )
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "tools.service.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        str(args.tool_workers),
    ]
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    service = ToolService(process=process, log_handle=log_handle)
    deadline = time.monotonic() + args.tool_start_timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            service.close()
            raise SystemExit(
                f"Tool service exited during startup; inspect {output_root / 'tool_service.log'}"
            )
        if _healthy_tool_service(args.tool_service_url) is not None:
            print(
                f"[v11-launcher] started {args.tool_workers}-worker tool service at "
                f"{args.tool_service_url}",
                flush=True,
            )
            return service
        time.sleep(2)
    service.close()
    raise SystemExit(
        f"Timed out starting tool service; inspect {output_root / 'tool_service.log'}"
    )


def _batch_dir(
    output_root: Path,
    task: TaskSpec,
    selection_unit: str,
    experiment_mode: str = "full_flat",
    top_k: int = 3,
    diversity_mode: str = ASSAY_TRANSFER_DIVERSITY_NONE,
    records_per_molecule: int = ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
) -> Path:
    condition = output_root.name
    return Path(task.paper_output_root) / "assay_transfer_v11_glm" / condition / RUNS_DIR / _task_batch_id(
        task,
        selection_unit,
        experiment_mode,
        top_k,
        diversity_mode,
        records_per_molecule,
    )


def _audit_retrieval_condition_artifacts(
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_root = Path(args.output_root)
    errors: list[str] = []
    batches: dict[str, Any] = {}
    for condition_name in args.retrieval_conditions:
        expected_strategy = (
            "morgan_fingerprint"
            if condition_name == RETRIEVAL_CONDITION_MORGAN
            else "assay_transfer_tool"
        )
        expected_selection_unit = {
            RETRIEVAL_CONDITION_ASSAY_TRANSFER_RECORD: (
                ASSAY_TRANSFER_SELECTION_SCORED_RECORD
            ),
            RETRIEVAL_CONDITION_ASSAY_TRANSFER_MOLECULE: (
                ASSAY_TRANSFER_SELECTION_MEAN_SCORE_MOLECULE
            ),
        }.get(condition_name)
        for task in _selected_tasks(args):
            key = f"{task.task_id}/{condition_name}"
            batch_id = _retrieval_condition_batch_id(
                task, condition_name, args.top_k_per_group
            )
            batch_dir = _retrieval_condition_batch_dir(args, task, condition_name)
            expected_queries = min(
                task.expected_queries, args.limit or task.expected_queries
            )
            metrics_path = batch_dir / "metrics.json"
            manifest_path = batch_dir / "manifest.json"
            metrics = _read_json(metrics_path) if metrics_path.exists() else {}
            manifest = _read_json(manifest_path) if manifest_path.exists() else {}
            for field, expected in (
                ("n_total", expected_queries),
                ("n_successful", expected_queries),
                ("n_failed_runs", 0),
            ):
                if metrics.get(field) != expected:
                    errors.append(
                        f"{key}: metrics {field}={metrics.get(field)!r}, "
                        f"expected={expected!r}"
                    )
            manifest_expected = {
                "experiment_mode": "full_flat",
                "retrieval_source": "starling",
                "retrieval_strategy": expected_strategy,
                "neighbor_identity_policy": "parent_disjoint",
                "identity_blind": True,
                "top_k_per_group": args.top_k_per_group,
                "analogous_reasoning_only": True,
                "group_prompt_version": "analogous_flat_v1",
                "group_tools_enabled": False,
            }
            if expected_selection_unit is not None:
                manifest_expected.update(
                    {
                        "enable_assay_transfer_scores": True,
                        "assay_transfer_selection_unit": expected_selection_unit,
                        "assay_transfer_min_score": None,
                    }
                )
            for field, expected in manifest_expected.items():
                if manifest.get(field) != expected:
                    errors.append(
                        f"{key}: manifest {field}={manifest.get(field)!r}, "
                        f"expected={expected!r}"
                    )
            run_dirs = sorted((batch_dir / "runs").glob(f"{batch_id}_idx*"))
            if len(run_dirs) != expected_queries:
                errors.append(
                    f"{key}: run directory count={len(run_dirs)}, "
                    f"expected={expected_queries}"
                )
            observed_neighbors = 0
            for run_dir in run_dirs:
                required = (
                    run_dir / "retrieval.json",
                    run_dir / "group_reasoning_outputs.jsonl",
                    run_dir / "single_molecule_reasoning_output.json",
                    run_dir / "final_reasoning_output.json",
                    run_dir / "trace_messages.jsonl",
                )
                missing = [path.name for path in required if not path.exists()]
                if missing:
                    errors.append(f"{key}/{run_dir.name}: missing {missing}")
                    continue
                retrieval = _read_json(run_dir / "retrieval.json")
                policy = retrieval.get("retrieval_policy") or {}
                coverage = retrieval.get("coverage") or {}
                experiment = retrieval.get("experiment") or {}
                if policy.get("neighbor_identity_policy") != "parent_disjoint":
                    errors.append(f"{key}/{run_dir.name}: retrieval is not parent-disjoint")
                if coverage.get("top_k_per_group") != args.top_k_per_group:
                    errors.append(f"{key}/{run_dir.name}: retrieval top-k mismatch")
                groups = retrieval.get("groups") or []
                if len(groups) != 1 or groups[0].get("group_id") != "Flat.all_evidence":
                    errors.append(f"{key}/{run_dir.name}: expected one flat evidence group")
                    continue
                neighbors = groups[0].get("neighbors") or []
                observed_neighbors += len(neighbors)
                maximum_flat_neighbors = args.top_k_per_group * max(
                    1, len(groups[0].get("source_group_ids") or [])
                )
                if not 1 <= len(neighbors) <= maximum_flat_neighbors:
                    errors.append(
                        f"{key}/{run_dir.name}: flat neighbor count outside "
                        f"per-family k={args.top_k_per_group} bound"
                    )
                for neighbor in neighbors:
                    relation = str(neighbor.get("molecule_relation") or "")
                    if relation in {
                        "exact_record",
                        "same_connectivity_variant",
                        "same_parent",
                    }:
                        errors.append(f"{key}/{run_dir.name}: identity leak {relation}")
                    score_field = (
                        "similarity"
                        if condition_name == RETRIEVAL_CONDITION_MORGAN
                        else "transfer_selection_score"
                    )
                    score = neighbor.get(score_field)
                    if not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
                        errors.append(
                            f"{key}/{run_dir.name}: invalid {score_field}"
                        )
                if expected_selection_unit is not None:
                    selection = experiment.get("assay_transfer_selection_policy") or {}
                    diversity = selection.get("diversity") or {}
                    if diversity.get("selection_unit") != expected_selection_unit:
                        errors.append(f"{key}/{run_dir.name}: selection-unit mismatch")
                    if selection.get("min_score") is not None:
                        errors.append(f"{key}/{run_dir.name}: score floor was applied")
            batches[key] = {
                "condition": condition_name,
                "task_id": task.task_id,
                "batch_id": batch_id,
                "batch_dir": str(batch_dir),
                "expected_queries": expected_queries,
                "observed_run_directories": len(run_dirs),
                "observed_neighbors": observed_neighbors,
                "metrics": metrics,
            }
    audit = {
        "contract_version": LAUNCH_CONTRACT_VERSION,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "batches": batches,
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json_atomic(output_root / "completion_audit.json", audit)
    return audit


def _write_retrieval_condition_summary(
    args: argparse.Namespace,
    failed: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "contract_version": LAUNCH_CONTRACT_VERSION,
        "status": "complete" if not failed else "failed",
        "retrieval_conditions": list(args.retrieval_conditions),
        "total_expected_llm_requests": sum(
            min(task.expected_queries, args.limit or task.expected_queries)
            for task in _selected_tasks(args)
        )
        * len(args.retrieval_conditions)
        * 2,
        "failed": failed,
        "completion_audit": {
            "status": audit["status"],
            "path": str(Path(args.output_root) / "completion_audit.json"),
            "n_errors": len(audit["errors"]),
        },
        "batches": audit["batches"],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json_atomic(Path(args.output_root) / "completion_summary.json", summary)
    return summary


def _audit_completed_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    """Gate scheduler success on the frozen retrieval and output contracts."""
    output_root = Path(args.output_root)
    errors: list[str] = []
    task_audits: dict[str, Any] = {}
    expected_group_counts = {
        "bbb_martins": 4,
        "bioavailability_ma": 5,
        "skin_reaction": 2,
    }
    for task in _selected_tasks(args):
        batch_dir = _batch_dir(
            output_root,
            task,
            args.assay_transfer_selection_unit,
            args.experiment_mode,
            args.top_k_per_group,
            args.assay_transfer_diversity_mode,
            args.assay_transfer_records_per_molecule,
        )
        expected_queries = min(task.expected_queries, args.limit or task.expected_queries)
        metrics_path = batch_dir / "metrics.json"
        manifest_path = batch_dir / "manifest.json"
        metrics = _read_json(metrics_path) if metrics_path.exists() else {}
        manifest = _read_json(manifest_path) if manifest_path.exists() else {}
        for field, expected in (
            ("n_total", expected_queries),
            ("n_successful", expected_queries),
            ("n_failed_runs", 0),
        ):
            if metrics.get(field) != expected:
                errors.append(
                    f"{task.task_id}: metrics {field}={metrics.get(field)!r}, expected={expected!r}"
                )
        manifest_expected = {
            "n_items": expected_queries,
            "experiment_mode": args.experiment_mode,
            "retrieval_source": "starling",
            "retrieval_strategy": "assay_transfer_tool",
            "assay_transfer_profile": PROFILE_NAME,
            "enable_assay_transfer_scores": True,
            "assay_transfer_min_score": None,
            "assay_transfer_diversity_mode": args.assay_transfer_diversity_mode,
            "assay_transfer_diversity_score_slack": _task_diversity_score_slack(
                task, args
            ),
            "assay_transfer_selection_unit": args.assay_transfer_selection_unit,
            "assay_transfer_records_per_molecule": (
                args.assay_transfer_records_per_molecule
            ),
            "assay_transfer_initial_morgan_filter": 50,
            "neighbor_identity_policy": "parent_disjoint",
            "neighbor_context_profile": "standard",
            "identity_blind": True,
            "top_k_per_group": args.top_k_per_group,
            "group_output_schema": (
                "analogous_flat_minimal_v1"
                if args.analogous_reasoning_only
                else task.group_output_schema
            ),
            "group_prompt_version": (
                "analogous_flat_v1"
                if args.analogous_reasoning_only
                else task.group_prompt_version
            ),
            "disable_flat_tools": args.disable_flat_tools,
            "group_tools_enabled": (
                not args.disable_flat_tools and not args.analogous_reasoning_only
            ),
            "analogous_reasoning_only": args.analogous_reasoning_only,
        }
        for field, expected in manifest_expected.items():
            if manifest.get(field) != expected:
                errors.append(
                    f"{task.task_id}: manifest {field}={manifest.get(field)!r}, expected={expected!r}"
                )
        preflight = manifest.get("rerank_cache_preflight") or {}
        provenance = preflight.get("provenance") or {}
        if preflight.get("status") != "complete" or preflight.get("cache_quick_check") != "ok":
            errors.append(f"{task.task_id}: cache preflight is not complete/ok")
        if provenance.get("backbone_dtype") != BACKBONE_DTYPE:
            errors.append(f"{task.task_id}: unexpected cache backbone dtype")
        if provenance.get("logit_extraction_dtype") != LOGIT_EXTRACTION_DTYPE:
            errors.append(f"{task.task_id}: cache logits were not extracted in FP32")

        batch_id = _task_batch_id(
            task,
            args.assay_transfer_selection_unit,
            args.experiment_mode,
            args.top_k_per_group,
            args.assay_transfer_diversity_mode,
            args.assay_transfer_records_per_molecule,
        )
        run_dirs = sorted((batch_dir / "runs").glob(f"{batch_id}_idx*"))
        if len(run_dirs) != expected_queries:
            errors.append(
                f"{task.task_id}: run directory count={len(run_dirs)}, expected={expected_queries}"
            )
        n_groups = 0
        n_neighbors = 0
        n_selected_records = 0
        n_molecules_at_record_limit = 0
        maximum_records_per_molecule = 0
        maximum_structural_rank = 0
        for run_dir in run_dirs:
            required = (
                run_dir / "retrieval.json",
                run_dir / "group_reasoning_outputs.jsonl",
                run_dir / "single_molecule_reasoning_output.json",
                run_dir / "final_reasoning_output.json",
                run_dir / "trace_messages.jsonl",
            )
            missing = [path.name for path in required if not path.exists()]
            if missing:
                errors.append(f"{task.task_id}/{run_dir.name}: missing {missing}")
                continue
            retrieval = _read_json(run_dir / "retrieval.json")
            policy = retrieval.get("retrieval_policy") or {}
            experiment = retrieval.get("experiment") or {}
            coverage = retrieval.get("coverage") or {}
            if policy.get("neighbor_identity_policy") != "parent_disjoint":
                errors.append(f"{task.task_id}/{run_dir.name}: retrieval is not parent-disjoint")
            if coverage.get("top_k_per_group") != args.top_k_per_group:
                errors.append(
                    f"{task.task_id}/{run_dir.name}: coverage top-k is not "
                    f"{args.top_k_per_group}"
                )
            selection = experiment.get("assay_transfer_selection_policy") or {}
            if selection.get("min_score") is not None:
                errors.append(f"{task.task_id}/{run_dir.name}: assay-transfer score floor was applied")
            if (selection.get("diversity") or {}).get("selection_unit") != args.assay_transfer_selection_unit:
                errors.append(f"{task.task_id}/{run_dir.name}: wrong selection unit")
            diversity = selection.get("diversity") or {}
            if diversity.get("mode") != args.assay_transfer_diversity_mode:
                errors.append(f"{task.task_id}/{run_dir.name}: wrong diversity mode")
            if diversity.get("score_slack") != _task_diversity_score_slack(task, args):
                errors.append(f"{task.task_id}/{run_dir.name}: wrong diversity score slack")
            score_policy = experiment.get("llm_neighbor_score_policy") or {}
            if score_policy.get("round_decimals") != 2:
                errors.append(f"{task.task_id}/{run_dir.name}: score visibility is not two decimals")
            reranker = experiment.get("retrieval_reranker") or {}
            if reranker.get("logit_extraction_dtype") != LOGIT_EXTRACTION_DTYPE:
                errors.append(f"{task.task_id}/{run_dir.name}: retrieval provenance is not FP32 logits")
            groups = retrieval.get("groups") or []
            expected_group_count = (
                1 if args.experiment_mode == "full_flat" else expected_group_counts[task.task_id]
            )
            if len(groups) != expected_group_count:
                errors.append(
                    f"{task.task_id}/{run_dir.name}: group count={len(groups)}, "
                    f"expected={expected_group_count}"
                )
            for group in groups:
                neighbors = group.get("neighbors") or []
                n_groups += 1
                n_neighbors += len(neighbors)
                if len(neighbors) != args.top_k_per_group:
                    errors.append(
                        f"{task.task_id}/{run_dir.name}/{group.get('group_id')}: "
                        f"neighbor count={len(neighbors)}, expected={args.top_k_per_group}"
                    )
                if (
                    args.assay_transfer_selection_unit
                    in {
                        ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE,
                        ASSAY_TRANSFER_SELECTION_MEAN_SCORE_MOLECULE,
                    }
                    and len({str(row.get("molecule_chembl_id") or "") for row in neighbors})
                    != len(neighbors)
                ):
                    errors.append(f"{task.task_id}/{run_dir.name}: duplicate selected molecule")
                for neighbor in neighbors:
                    structural_rank = int(neighbor.get("structural_rank") or 0)
                    maximum_structural_rank = max(maximum_structural_rank, structural_rank)
                    score = neighbor.get("transfer_selection_score")
                    if not 1 <= structural_rank <= 50:
                        errors.append(
                            f"{task.task_id}/{run_dir.name}: structural rank outside initial pool: "
                            f"{structural_rank}"
                        )
                    if not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
                        errors.append(f"{task.task_id}/{run_dir.name}: invalid transfer score")
                    selected_records = [
                        selected
                        for family in (
                            neighbor.get("transfer_family_selections") or [neighbor]
                        )
                        for selected in (
                            family.get("transfer_selected_records")
                            or [
                                {
                                    "transfer_selection_score": family.get(
                                        "transfer_selection_score"
                                    ),
                                    "transfer_winning_record": family.get(
                                        "transfer_winning_record"
                                    )
                                    or {},
                                }
                            ]
                        )
                    ]
                    if (
                        args.assay_transfer_records_per_molecule > 1
                        and neighbor.get("transfer_records_per_molecule_limit")
                        != args.assay_transfer_records_per_molecule
                    ):
                        errors.append(
                            f"{task.task_id}/{run_dir.name}: missing or incorrect "
                            "records-per-molecule audit"
                        )
                    if len(selected_records) > args.assay_transfer_records_per_molecule:
                        errors.append(
                            f"{task.task_id}/{run_dir.name}: too many records for one molecule"
                        )
                    n_selected_records += len(selected_records)
                    maximum_records_per_molecule = max(
                        maximum_records_per_molecule, len(selected_records)
                    )
                    if len(selected_records) == args.assay_transfer_records_per_molecule:
                        n_molecules_at_record_limit += 1
                    endpoint_keys = [
                        str(
                            (record.get("transfer_winning_record") or {}).get(
                                "canonical_endpoint_key"
                            )
                            or ""
                        )
                        for record in selected_records
                    ]
                    if any(
                        not isinstance(record.get("transfer_selection_score"), (int, float))
                        or not 0.0
                        <= float(record["transfer_selection_score"])
                        <= 1.0
                        for record in selected_records
                    ):
                        errors.append(
                            f"{task.task_id}/{run_dir.name}: invalid bundled transfer score"
                        )
                    if (
                        args.assay_transfer_selection_unit
                        == ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE
                        and args.assay_transfer_records_per_molecule > 1
                        and (
                        any(not key for key in endpoint_keys)
                        or len(endpoint_keys) != len(set(endpoint_keys))
                        )
                    ):
                        errors.append(
                            f"{task.task_id}/{run_dir.name}: selected records are not "
                            "endpoint-distinct"
                        )
        task_audits[task.task_id] = {
            "expected_queries": expected_queries,
            "observed_run_directories": len(run_dirs),
            "observed_groups": n_groups,
            "observed_neighbors": n_neighbors,
            "observed_selected_records": n_selected_records,
            "mean_records_per_selected_molecule": (
                round(n_selected_records / n_neighbors, 6) if n_neighbors else 0.0
            ),
            "maximum_records_per_molecule": maximum_records_per_molecule,
            "n_molecules_at_record_limit": n_molecules_at_record_limit,
            "maximum_structural_rank": maximum_structural_rank,
            "metrics": metrics,
        }
    audit = {
        "contract_version": LAUNCH_CONTRACT_VERSION,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "tasks": task_audits,
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json_atomic(output_root / "completion_audit.json", audit)
    return audit


def _write_completion_summary(
    args: argparse.Namespace,
    failed: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    output_root = Path(args.output_root)
    batches: dict[str, Any] = {}
    for task in _selected_tasks(args):
        batch_dir = _batch_dir(
            output_root,
            task,
            args.assay_transfer_selection_unit,
            args.experiment_mode,
            args.top_k_per_group,
            args.assay_transfer_diversity_mode,
            args.assay_transfer_records_per_molecule,
        )
        metrics_path = batch_dir / "metrics.json"
        metrics = _read_json(metrics_path) if metrics_path.exists() else {}
        batches[task.task_id] = {
            "batch_id": _task_batch_id(
                task,
                args.assay_transfer_selection_unit,
                args.experiment_mode,
                args.top_k_per_group,
                args.assay_transfer_diversity_mode,
                args.assay_transfer_records_per_molecule,
            ),
            "batch_dir": str(batch_dir),
            "metrics_path": str(metrics_path),
            "n_total": metrics.get("n_total"),
            "n_successful": metrics.get("n_successful"),
            "n_failed_runs": metrics.get("n_failed_runs"),
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
        }
    summary = {
        "contract_version": LAUNCH_CONTRACT_VERSION,
        "status": "complete" if not failed else "failed",
        "failed": failed,
        "completion_audit": {
            "status": audit["status"],
            "path": str(output_root / "completion_audit.json"),
            "n_errors": len(audit["errors"]),
        },
        "batches": batches,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json_atomic(output_root / "completion_summary.json", summary)
    lines = [
        "# V11 assay-transfer scaffold-validation run",
        "",
        (
            f"This run used an initial Morgan pool of 50, selected top-{args.top_k_per_group} "
            f"per mechanism group with {args.assay_transfer_selection_unit}, and displayed "
            f"up to {args.assay_transfer_records_per_molecule} record(s) per molecule. "
            "Transfer logits were extracted in FP32."
        ),
        "",
        "| Task | n | Accuracy | Macro-F1 | Records / molecule | Max records |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in _selected_tasks(args):
        batch = batches[task.task_id]
        task_audit = audit["tasks"].get(task.task_id, {})
        lines.append(
            "| {task} | {n} | {accuracy:.4f} | {macro_f1:.4f} | {mean:.3f} | {maximum} |".format(
                task=task.task_id,
                n=batch.get("n_total") or 0,
                accuracy=float(batch.get("accuracy") or 0.0),
                macro_f1=float(batch.get("macro_f1") or 0.0),
                mean=float(task_audit.get("mean_records_per_selected_molecule") or 0.0),
                maximum=task_audit.get("maximum_records_per_molecule") or 0,
            )
        )
    lines.extend(
        [
            "",
            f"Completion audit: **{audit['status']}** ({len(audit['errors'])} errors).",
            "",
        ]
    )
    (output_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(task.task_id for task in ACTIVE_TASKS),
        default=list(DEFAULT_TASK_IDS),
    )
    parser.add_argument(
        "--experiment-mode",
        choices=("full_flat", "full_mechanism"),
        default="full_flat",
    )
    parser.add_argument(
        "--retrieval-condition",
        dest="retrieval_conditions",
        action="append",
        choices=RETRIEVAL_CONDITIONS,
        default=[],
        help=(
            "Repeat to schedule Morgan, assay-transfer record-level, and/or "
            "assay-transfer molecule-level retrieval in one global prompt pool."
        ),
    )
    parser.add_argument(
        "--assay-transfer-selection-unit",
        choices=ASSAY_TRANSFER_SELECTION_UNITS,
        default=ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    )
    parser.add_argument(
        "--assay-transfer-records-per-molecule",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Displayed records per molecule. Defaults to 6 for mean_score_molecule "
            "and 1 otherwise."
        ),
    )
    parser.add_argument(
        "--assay-transfer-diversity-mode",
        choices=ASSAY_TRANSFER_DIVERSITY_MODES,
        default=ASSAY_TRANSFER_DIVERSITY_NONE,
    )
    parser.add_argument(
        "--bbb-diversity-score-slack", type=float, default=0.0
    )
    parser.add_argument(
        "--bioavailability-diversity-score-slack", type=float, default=0.0
    )
    parser.add_argument(
        "--skin-reaction-diversity-score-slack", type=float, default=0.0
    )
    parser.add_argument("--limit", type=int, default=0, help="0 runs all validation queries.")
    parser.add_argument("--top-k-per-group", type=int, default=3)
    parser.add_argument("--parallelism", type=int, default=128)
    parser.add_argument("--retrieval-preparation-workers", type=int, default=8)
    parser.add_argument("--max-stage-requeues", type=int, default=0)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-env", default=GLM_API_KEY_ENV)
    parser.add_argument("--model", default=GLM_MODEL)
    parser.add_argument("--reasoning-effort", default=GLM_REASONING_EFFORT)
    parser.add_argument("--tool-service-url", default="http://127.0.0.1:8766")
    parser.add_argument("--tool-workers", type=int, default=32)
    parser.add_argument(
        "--tool-cache-path",
        default=f"/local/joseph/txagent/assay-v11-k3-tool-cache-{os.getuid()}.sqlite3",
    )
    parser.add_argument("--tool-start-timeout-s", type=int, default=300)
    parser.add_argument(
        "--disable-flat-tools",
        action="store_true",
        help=(
            "Disable flat group-branch molecule comparison tools while retaining "
            "single-branch molecule_properties"
        ),
    )
    parser.add_argument(
        "--analogous-reasoning-only",
        action="store_true",
        help=(
            "Run the versioned tool-free analogous_flat_v1 prompt: omit the "
            "single stage, show neighbor SMILES and record transfer scores, then "
            "run one flat analysis followed by final synthesis."
        ),
    )
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args(argv)
    args.retrieval_conditions = list(dict.fromkeys(args.retrieval_conditions))
    if args.assay_transfer_records_per_molecule is None:
        args.assay_transfer_records_per_molecule = (
            6
            if args.assay_transfer_selection_unit
            == ASSAY_TRANSFER_SELECTION_MEAN_SCORE_MOLECULE
            else ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT
        )
    if not args.output_root:
        if args.retrieval_conditions:
            condition_slug = "_".join(args.retrieval_conditions)
            args.output_root = str(
                DEFAULT_OUTPUT_ROOT
                / f"full_flat_k{args.top_k_per_group}_{condition_slug}_analogous_flat_v1"
            )
        else:
            args.output_root = str(
                _default_output_root(
                    args.top_k_per_group,
                    args.assay_transfer_selection_unit,
                    experiment_mode=args.experiment_mode,
                    diversity_mode=args.assay_transfer_diversity_mode,
                    records_per_molecule=args.assay_transfer_records_per_molecule,
                    disable_flat_tools=args.disable_flat_tools,
                    analogous_reasoning_only=args.analogous_reasoning_only,
                )
            )
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if not 1 <= args.top_k_per_group <= 50:
        parser.error("--top-k-per-group must be between 1 and the Morgan pool size of 50")
    if not 1 <= args.parallelism <= ENDPOINT_CONCURRENCY_BUDGET:
        parser.error(
            "--parallelism must be between 1 and the global endpoint budget of "
            f"{ENDPOINT_CONCURRENCY_BUDGET}"
        )
    if args.retrieval_preparation_workers < 1:
        parser.error("--retrieval-preparation-workers must be positive")
    if args.max_stage_requeues < 0:
        parser.error("--max-stage-requeues must be non-negative")
    if args.disable_flat_tools and args.experiment_mode != "full_flat":
        parser.error("--disable-flat-tools requires --experiment-mode full_flat")
    if args.analogous_reasoning_only and args.experiment_mode != "full_flat":
        parser.error(
            "--analogous-reasoning-only requires --experiment-mode full_flat"
        )
    if args.retrieval_conditions and not args.analogous_reasoning_only:
        parser.error(
            "--retrieval-condition requires --analogous-reasoning-only"
        )
    if args.tool_workers < 1:
        parser.error("--tool-workers must be positive")
    try:
        validate_assay_transfer_records_per_molecule(
            args.assay_transfer_records_per_molecule,
            selection_unit=args.assay_transfer_selection_unit,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if (
        args.assay_transfer_diversity_mode != ASSAY_TRANSFER_DIVERSITY_NONE
        and args.assay_transfer_selection_unit
        != ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE
    ):
        parser.error(
            "this launcher requires --assay-transfer-selection-unit unique_molecule "
            "when diversity selection is enabled"
        )
    for task, score_slack in (
        ("BBB_Martins", args.bbb_diversity_score_slack),
        ("Bioavailability_Ma", args.bioavailability_diversity_score_slack),
        ("Skin_Reaction", args.skin_reaction_diversity_score_slack),
    ):
        try:
            validate_assay_transfer_diversity(
                mode=args.assay_transfer_diversity_mode,
                score_slack=score_slack,
            )
        except ValueError as exc:
            parser.error(f"{task}: {exc}")
    args.base_url = resolve_endpoint_base_url(args.base_url)
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = Path(args.output_root)
    selected_tasks = _selected_tasks(args)
    validated = _validate_inputs(selected_tasks)
    commands = build_batch_commands(args)
    manifest = {
        "contract_version": LAUNCH_CONTRACT_VERSION,
        "scheduler": SCHEDULER_VERSION,
        "benchmark_split": "scaffold",
        "evaluation_subset": "valid",
        "experiment_mode": args.experiment_mode,
        "visibility_mode": (
            "query_blind_neighbor_smiles"
            if args.analogous_reasoning_only
            else "identity_blind"
        ),
        "neighbor_identity_policy": "parent_disjoint",
        "retrieval_strategy": (
            "comparison_suite"
            if args.retrieval_conditions
            else "assay_transfer_tool"
        ),
        "retrieval_conditions": list(args.retrieval_conditions),
        "assay_transfer_profile": PROFILE_NAME,
        "assay_transfer_initial_morgan_filter": 50,
        "top_k_per_group": args.top_k_per_group,
        "min_similarity": 0.0,
        "assay_transfer_min_score": None,
        "assay_transfer_diversity_mode": args.assay_transfer_diversity_mode,
        "assay_transfer_diversity_score_slack_by_task": {
            task.task_id: _task_diversity_score_slack(task, args) for task in selected_tasks
        },
        "assay_transfer_selection_unit": args.assay_transfer_selection_unit,
        "assay_transfer_records_per_molecule": (
            args.assay_transfer_records_per_molecule
        ),
        "scores_visible": True,
        "analogous_reasoning_only": args.analogous_reasoning_only,
        "effective_prompt_version": (
            "analogous_flat_v1" if args.analogous_reasoning_only else ""
        ),
        "group_prompt_versions": {
            task.task_id: (
                "analogous_flat_v1"
                if args.analogous_reasoning_only
                else task.group_prompt_version
            )
            for task in selected_tasks
        },
        "disable_flat_tools": args.disable_flat_tools or args.analogous_reasoning_only,
        "flat_group_tool_policy": (
            "omitted.v1"
            if args.disable_flat_tools or args.analogous_reasoning_only
            else "standard.v1"
        ),
        "score_decimals": 2,
        "backbone_dtype": BACKBONE_DTYPE,
        "logit_extraction_dtype": LOGIT_EXTRACTION_DTYPE,
        "model": args.model,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "reasoning_effort": args.reasoning_effort,
        "enable_thinking_body": False,
        "parallelism": args.parallelism,
        "endpoint_concurrency_budget": ENDPOINT_CONCURRENCY_BUDGET,
        "retrieval_preparation_workers": args.retrieval_preparation_workers,
        "limit": args.limit,
        "tool_service_url": args.tool_service_url,
        "tool_workers": args.tool_workers,
        "validated_caches": validated,
        "tasks": [asdict(task) for task in selected_tasks],
        "commands": [
            {"experiment_name": command.experiment_name, "command": command.command}
            for command in commands
        ],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_root / "launch_manifest.json", manifest)
    if args.manifest_only:
        print(json.dumps({"launch_manifest": str(output_root / "launch_manifest.json")}, indent=2))
        return 0

    endpoint = _check_endpoint(args)
    manifest["endpoint_health"] = endpoint
    write_json_atomic(output_root / "launch_manifest.json", manifest)
    service = (
        ToolService()
        if args.analogous_reasoning_only
        else _start_or_reuse_tool_service(args)
    )
    manifest["tool_service"] = (
        {"status": "omitted", "reason": "analogous_flat_v1"}
        if args.analogous_reasoning_only
        else {
            "status": "healthy",
            "reused_existing": service.reused,
            "workers": args.tool_workers,
            "native_threads": 1,
            "batch_workers_per_process": 8,
        }
    )
    write_json_atomic(output_root / "launch_manifest.json", manifest)
    try:
        failed = run_global_prompt_pool(
            commands,
            max_workers=args.parallelism,
            max_stage_requeues=args.max_stage_requeues,
            preparation_workers=args.retrieval_preparation_workers,
        )
        audit = (
            _audit_retrieval_condition_artifacts(args)
            if args.retrieval_conditions
            else _audit_completed_artifacts(args)
        )
        if audit["status"] != "pass":
            failed.append(
                {
                    "experiment": "completion_audit",
                    "returncode": 1,
                    "n_errors": len(audit["errors"]),
                }
            )
        summary = (
            _write_retrieval_condition_summary(args, failed, audit)
            if args.retrieval_conditions
            else _write_completion_summary(args, failed, audit)
        )
        print(json.dumps(summary, indent=2), flush=True)
        return 1 if failed else 0
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
