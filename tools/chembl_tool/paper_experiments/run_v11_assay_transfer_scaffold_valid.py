"""Run a V11 assay-transfer scaffold-validation experiment.

The launcher covers BBB_Martins, Bioavailability_Ma, and Skin_Reaction with
held-out scaffold-validation indices and complete task-specific V11 caches.
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
    default_cache_paths,
    file_sha256,
    model_profile,
)
from tools.chembl_tool.common.assay_transfer_selection import (
    ASSAY_TRANSFER_DIVERSITY_MODES,
    ASSAY_TRANSFER_DIVERSITY_NONE,
    ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
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
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/paper/"
    "molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid/"
    "assay_transfer_v11_k3_scored_assay_schema"
)
UNIQUE_MOLECULE_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT.with_name(
    "assay_transfer_v11_k3_unique_molecules_scored_assay_schema"
)
RUNS_DIR = "runs_identity_blind_parent_disjoint"
LAUNCH_CONTRACT_VERSION = "v11_assay_transfer_scaffold_valid.v3"
EXPECTED_TOOL_COUNT = 3
ENDPOINT_CONCURRENCY_BUDGET = 500
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
    group_output_schema: str = ""
    single_analysis_source_batch: str = ""

    @property
    def input_jsonl(self) -> str:
        return f"data/processed_starling/{self.data_name}/scaffold/valid.jsonl"

    @property
    def index(self) -> str:
        return (
            f"outputs/chembl_tool/tasks/{self.task_id}/evidence_library/"
            "starling_normalized_v7/08_neighbor_index/scaffold"
        )

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
        expected_queries=500,
        expected_scores=893_133,
        single_analysis_source_batch=str(
            MORGAN_K3_CONTROL_ROOT / "bbb_martins" / "bbb_martins__none"
        ),
    ),
    TaskSpec(
        task_id="bioavailability_ma",
        data_name="Bioavailability_Ma",
        batch_module="tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch",
        expected_queries=209,
        expected_scores=840_608,
        group_output_schema="assay-transfer",
    ),
    TaskSpec(
        task_id="skin_reaction",
        data_name="Skin_Reaction",
        batch_module="tools.chembl_tool.tasks.skin_reaction.run_reasoning_batch",
        expected_queries=245,
        expected_scores=964_543,
    ),
)


def _default_output_root(
    top_k: int,
    selection_unit: str,
    diversity_mode: str = ASSAY_TRANSFER_DIVERSITY_NONE,
    records_per_molecule: int = ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
) -> Path:
    suffix = (
        "unique_molecules_scored_assay_schema"
        if selection_unit == ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE
        else "scored_assay_schema"
    )
    if diversity_mode != ASSAY_TRANSFER_DIVERSITY_NONE:
        suffix = suffix.replace(
            "scored_assay_schema", f"{diversity_mode}_diversity_scored_assay_schema"
        )
    if records_per_molecule > ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT:
        suffix = f"{suffix}_r{records_per_molecule}"
    return DEFAULT_OUTPUT_ROOT.with_name(f"assay_transfer_v11_k{top_k}_{suffix}")


def _task_batch_id(
    task: TaskSpec,
    selection_unit: str,
    top_k: int = 3,
    diversity_mode: str = ASSAY_TRANSFER_DIVERSITY_NONE,
    records_per_molecule: int = ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
) -> str:
    batch_id = task.batch_id.replace("_k3_morgan50", f"_k{top_k}_morgan50")
    markers = []
    if selection_unit == ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE:
        markers.append("unique_molecule")
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
    """Build the three auditable batch commands consumed by the global pool."""
    commands: list[BatchCommand] = []
    for task in TASKS:
        paths = default_cache_paths(task.task_id)
        profile = model_profile(task.task_id)
        batch_root = Path(args.output_root) / RUNS_DIR / task.task_id
        batch_id = _task_batch_id(
            task,
            args.assay_transfer_selection_unit,
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
            "full_mechanism",
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
            paths["catalog"],
            "--rerank-cache",
            paths["cache"],
            "--rerank-candidate-manifest",
            paths["candidate_manifest"],
            "--rerank-cache-version-manifest",
            paths["version"],
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
        if task.group_output_schema:
            command.extend(["--group-output-schema", task.group_output_schema])
        if (
            args.assay_transfer_selection_unit == ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE
            or args.top_k_per_group != 3
            or args.assay_transfer_records_per_molecule
            != ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT
        ):
            source_batch = (
                Path(task.single_analysis_source_batch)
                if task.single_analysis_source_batch
                else DEFAULT_OUTPUT_ROOT / RUNS_DIR / task.task_id / task.batch_id
            )
            command.extend(["--single-analysis-source-batch", str(source_batch)])
        commands.append(BatchCommand(batch_id, command))
    return commands


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_inputs() -> dict[str, dict[str, Any]]:
    """Fail before endpoint work if a cache/index no longer matches its build."""
    validated: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        paths = default_cache_paths(task.task_id)
        required = [Path(task.input_jsonl), Path(task.index), *(Path(value) for value in paths.values())]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise SystemExit("Missing V11 run inputs:\n" + "\n".join(missing))
        version = _read_json(Path(paths["version"]))
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
            "version_path": paths["version"],
            "version_sha256": file_sha256(paths["version"]),
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
    top_k: int = 3,
    diversity_mode: str = ASSAY_TRANSFER_DIVERSITY_NONE,
    records_per_molecule: int = ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
) -> Path:
    return output_root / RUNS_DIR / task.task_id / _task_batch_id(
        task, selection_unit, top_k, diversity_mode, records_per_molecule
    )


def _audit_completed_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    """Gate scheduler success on the frozen retrieval and output contracts."""
    output_root = Path(args.output_root)
    errors: list[str] = []
    task_audits: dict[str, Any] = {}
    expected_group_counts = {
        "bbb_martins": 4,
        "bioavailability_ma": 5,
        "skin_reaction": 4,
    }
    for task in TASKS:
        batch_dir = _batch_dir(
            output_root,
            task,
            args.assay_transfer_selection_unit,
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
            "experiment_mode": "full_mechanism",
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
            "group_output_schema": task.group_output_schema,
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
            if len(groups) != expected_group_counts[task.task_id]:
                errors.append(
                    f"{task.task_id}/{run_dir.name}: group count={len(groups)}, "
                    f"expected={expected_group_counts[task.task_id]}"
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
                    == ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE
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
                    selected_records = neighbor.get("transfer_selected_records") or [
                        {
                            "transfer_selection_score": score,
                            "transfer_winning_record": neighbor.get(
                                "transfer_winning_record"
                            ) or {},
                        }
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
                    if args.assay_transfer_records_per_molecule > 1 and (
                        any(not key for key in endpoint_keys)
                        or len(endpoint_keys) != len(set(endpoint_keys))
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
    for task in TASKS:
        batch_dir = _batch_dir(
            output_root,
            task,
            args.assay_transfer_selection_unit,
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
            "This run used an initial Morgan pool of 50, selected three unique "
            "molecules per mechanism group, and exposed up to five endpoint-distinct "
            "records per molecule. Transfer logits were extracted in FP32."
        ),
        "",
        "| Task | n | Accuracy | Macro-F1 | Records / molecule | Max records |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
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
        "--assay-transfer-selection-unit",
        choices=ASSAY_TRANSFER_SELECTION_UNITS,
        default=ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    )
    parser.add_argument(
        "--assay-transfer-records-per-molecule",
        type=int,
        default=ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
        metavar="N",
        help="Maximum endpoint-distinct records shown per selected unique molecule.",
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
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.output_root:
        args.output_root = str(
            _default_output_root(
                args.top_k_per_group,
                args.assay_transfer_selection_unit,
                args.assay_transfer_diversity_mode,
                args.assay_transfer_records_per_molecule,
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
    validated = _validate_inputs()
    commands = build_batch_commands(args)
    manifest = {
        "contract_version": LAUNCH_CONTRACT_VERSION,
        "scheduler": SCHEDULER_VERSION,
        "benchmark_split": "scaffold",
        "evaluation_subset": "valid",
        "visibility_mode": "identity_blind",
        "neighbor_identity_policy": "parent_disjoint",
        "retrieval_strategy": "assay_transfer_tool",
        "assay_transfer_profile": PROFILE_NAME,
        "assay_transfer_initial_morgan_filter": 50,
        "top_k_per_group": args.top_k_per_group,
        "min_similarity": 0.0,
        "assay_transfer_min_score": None,
        "assay_transfer_diversity_mode": args.assay_transfer_diversity_mode,
        "assay_transfer_diversity_score_slack_by_task": {
            task.task_id: _task_diversity_score_slack(task, args) for task in TASKS
        },
        "assay_transfer_selection_unit": args.assay_transfer_selection_unit,
        "assay_transfer_records_per_molecule": (
            args.assay_transfer_records_per_molecule
        ),
        "scores_visible": True,
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
        "tasks": [asdict(task) for task in TASKS],
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
    service = _start_or_reuse_tool_service(args)
    manifest["tool_service"] = {
        "status": "healthy",
        "reused_existing": service.reused,
        "workers": args.tool_workers,
        "native_threads": 1,
        "batch_workers_per_process": 8,
    }
    write_json_atomic(output_root / "launch_manifest.json", manifest)
    try:
        failed = run_global_prompt_pool(
            commands,
            max_workers=args.parallelism,
            max_stage_requeues=args.max_stage_requeues,
            preparation_workers=args.retrieval_preparation_workers,
        )
        audit = _audit_completed_artifacts(args)
        if audit["status"] != "pass":
            failed.append(
                {
                    "experiment": "completion_audit",
                    "returncode": 1,
                    "n_errors": len(audit["errors"]),
                }
            )
        summary = _write_completion_summary(args, failed, audit)
        print(json.dumps(summary, indent=2), flush=True)
        return 1 if failed else 0
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
