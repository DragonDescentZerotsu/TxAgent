"""Keep a PARCC GLM SSH tunnel and one resumable Starling matrix alive.

The watchdog deliberately never stores an SSH password.  When PARCC asks for
Duo, it selects the configured Push option and waits for the user to approve it.
Matrix recovery is safe because the paper runner always passes --skip-existing
to task batches.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

from tools.chembl_tool.common.retrieval_policy import NEIGHBOR_IDENTITY_POLICIES
from tools.chembl_tool.common.task_workflows.reasoning_batch import (
    prediction_to_label,
)
from tools.chembl_tool.tasks.bbb_martins.run_reasoning_batch import (
    CONFIG as BBB_BATCH_CONFIG,
)
from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch import (
    CONFIG as BIOAVAILABILITY_BATCH_CONFIG,
)
from tools.chembl_tool.tasks.skin_reaction.run_reasoning_batch import (
    CONFIG as SKIN_BATCH_CONFIG,
)


LOGGER = logging.getLogger("glm_tunnel_watchdog")
MATRIX_MODULE = "tools.chembl_tool.paper_experiments.starling_benchmark_matrix"
DEFAULT_PYTHON = "/data1/tianang/anaconda3/envs/vllm/bin/python"
TASK_BATCH_CONFIGS = {
    "bbb_martins": BBB_BATCH_CONFIG,
    "bioavailability_ma": BIOAVAILABILITY_BATCH_CONFIG,
    "skin_reaction": SKIN_BATCH_CONFIG,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--benchmark-split", choices=("random", "scaffold"), required=True)
    parser.add_argument("--evaluation-subset", choices=("valid", "test"), default="valid")
    parser.add_argument(
        "--visibility-mode",
        choices=("identity_blind", "deployment_visible_prefetched", "deployment_visible"),
        required=True,
    )
    parser.add_argument(
        "--neighbor-identity-policy",
        choices=NEIGHBOR_IDENTITY_POLICIES,
        required=True,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:50000/v1")
    parser.add_argument("--model", default="nvidia/GLM-5.2-NVFP4")
    parser.add_argument("--api-key-env", default="GLM_LOCAL_API_KEY")
    parser.add_argument("--python-executable", default=DEFAULT_PYTHON)
    parser.add_argument("--parallelism", type=int, default=128)
    parser.add_argument("--max-stage-requeues", type=int, default=0)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--expected-results", type=int, required=True)
    parser.add_argument("--ssh-host", default="parcc-glm")
    parser.add_argument(
        "--duo-option",
        default="2",
        help="Duo menu option selected automatically; approval still happens on the phone.",
    )
    parser.add_argument("--check-interval-seconds", type=float, default=30.0)
    parser.add_argument("--count-interval-seconds", type=float, default=300.0)
    parser.add_argument("--duo-retry-seconds", type=float, default=900.0)
    parser.add_argument("--launcher-retry-seconds", type=float, default=60.0)
    parser.add_argument("--health-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--duo-approval-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--watchdog-log", required=True)
    parser.add_argument("--launcher-log", required=True)
    parser.add_argument(
        "--check-once",
        action="store_true",
        help="Report current state without reconnecting or launching anything.",
    )
    return parser.parse_args(argv)


def _configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)


def endpoint_healthy(base_url: str, timeout_s: float) -> bool:
    url = f"{base_url.rstrip('/')}/models"
    try:
        with urlopen(url, timeout=timeout_s) as response:
            return response.status == 200
    except (OSError, URLError, TimeoutError):
        return False


def _run_root_name(visibility_mode: str, neighbor_identity_policy: str) -> str:
    if visibility_mode == "identity_blind":
        name = "runs_identity_blind"
    elif visibility_mode == "deployment_visible_prefetched":
        name = "runs_deployment_visible_prefetched"
    else:
        name = "runs_deployment_visible"
    if neighbor_identity_policy != "operational":
        name += f"_{neighbor_identity_policy}"
    return name


def count_final_results(
    output_root: Path,
    visibility_mode: str,
    neighbor_identity_policy: str,
) -> int:
    """Count runs that satisfy the batch runner's completeness contract."""
    run_root = output_root / _run_root_name(
        visibility_mode,
        neighbor_identity_policy,
    )
    if not run_root.exists():
        return 0
    complete = 0
    for base, _, filenames in os.walk(run_root):
        if "final_reasoning_output.json" not in filenames:
            continue
        run_dir = Path(base)
        try:
            final_output = json.loads(
                (run_dir / "final_reasoning_output.json").read_text(encoding="utf-8")
            )
            single_output = json.loads(
                (run_dir / "single_molecule_reasoning_output.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            group_path = run_dir / "group_reasoning_outputs.jsonl"
            group_outputs = (
                [
                    json.loads(line)
                    for line in group_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if group_path.exists()
                else []
            )
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        if final_output.get("status") != "ok" or single_output.get("status") != "ok":
            continue
        if not _final_prediction_is_valid(run_dir, final_output):
            continue
        expected_groups = manifest.get("n_groups_with_neighbors")
        if expected_groups is not None and len(group_outputs) != int(expected_groups):
            continue
        if any(row.get("status") != "ok" for row in group_outputs):
            continue
        if expected_groups and not _group_ids_match_retrieval(
            run_dir,
            group_outputs,
            int(expected_groups),
        ):
            continue
        complete += 1
    return complete


def _group_ids_match_retrieval(
    run_dir: Path,
    group_outputs: list[dict],
    expected_count: int,
) -> bool:
    """Reject duplicate or substituted groups even when the row count matches."""
    try:
        retrieval = json.loads(
            (run_dir / "retrieval.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    expected_ids = [
        str(group.get("group_id") or "")
        for group in retrieval.get("groups") or []
        if group.get("neighbors") and str(group.get("group_id") or "")
    ][:expected_count]
    actual_ids = [str(row.get("group_id") or "") for row in group_outputs]
    return (
        len(expected_ids) == expected_count
        and len(set(expected_ids)) == expected_count
        and len(set(actual_ids)) == expected_count
        and set(actual_ids) == set(expected_ids)
    )


def _final_prediction_is_valid(run_dir: Path, final_output: dict) -> bool:
    """Apply the same task-specific prediction normalization as the batch gate."""
    task = next((part for part in run_dir.parts if part in TASK_BATCH_CONFIGS), "")
    config = TASK_BATCH_CONFIGS.get(task)
    if config is None:
        return False
    content = (final_output.get("llm") or {}).get("content") or {}
    value = content.get(config.prediction_field)
    return prediction_to_label(config, value) is not None


def _iter_processes() -> Iterable[tuple[int, list[str]]]:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        args = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
        if args:
            yield int(entry.name), args


def find_matrix_pids(output_root: Path) -> list[int]:
    matches: list[int] = []
    for pid, args in _iter_processes():
        command = "\0".join(args)
        if MATRIX_MODULE not in command:
            continue
        try:
            root_index = args.index("--output-root") + 1
            configured_root = Path(args[root_index])
            if not configured_root.is_absolute():
                configured_root = (Path(f"/proc/{pid}/cwd").resolve() / configured_root)
            configured_root = configured_root.resolve()
        except (ValueError, IndexError, FileNotFoundError, PermissionError):
            continue
        if configured_root == output_root.resolve():
            matches.append(pid)
    return sorted(matches)


def find_tunnel_pids(ssh_host: str) -> list[int]:
    matches: list[int] = []
    for pid, args in _iter_processes():
        if Path(args[0]).name != "ssh":
            continue
        if ssh_host in args and ("-N" in args or any("N" in arg for arg in args[1:] if arg.startswith("-"))):
            matches.append(pid)
    return sorted(matches)


def _terminate_process_groups(pids: Iterable[int], *, label: str) -> None:
    groups: set[int] = set()
    own_group = os.getpgrp()
    for pid in pids:
        try:
            group = os.getpgid(pid)
        except ProcessLookupError:
            continue
        if group != own_group:
            groups.add(group)
    for group in groups:
        LOGGER.warning("stopping %s process group pgid=%s", label, group)
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15.0
    while groups and time.monotonic() < deadline:
        groups = {group for group in groups if _process_group_exists(group)}
        if groups:
            time.sleep(0.5)
    for group in groups:
        LOGGER.warning("forcing stale %s process group pgid=%s", label, group)
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _process_group_exists(group: int) -> bool:
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_pids(pids: Iterable[int], *, label: str) -> None:
    stopped: list[int] = []
    for pid in pids:
        LOGGER.warning("stopping stale %s pid=%s", label, pid)
        try:
            os.kill(pid, signal.SIGTERM)
            stopped.append(pid)
        except ProcessLookupError:
            continue
    deadline = time.monotonic() + 5.0
    while stopped and time.monotonic() < deadline:
        stopped = [pid for pid in stopped if Path(f"/proc/{pid}").exists()]
        if stopped:
            time.sleep(0.25)


def reconnect_tunnel(args: argparse.Namespace):
    try:
        import pexpect
    except ImportError as exc:  # pragma: no cover - deployment environment gate
        raise RuntimeError("pexpect is required for Duo-aware SSH recovery") from exc

    _terminate_pids(find_tunnel_pids(args.ssh_host), label="SSH tunnel")
    command = [
        "ssh",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=10",
        "-NT",
        args.ssh_host,
    ]
    child = pexpect.spawn(command[0], command[1:], encoding="utf-8", timeout=10)
    deadline = time.monotonic() + args.duo_approval_timeout_seconds
    duo_sent = False
    patterns = [
        r"Passcode or option \(1-7\):",
        r"Success\. Logging you in",
        r"(?i)password:",
        r"Permission denied",
        pexpect.EOF,
        pexpect.TIMEOUT,
    ]
    while time.monotonic() < deadline:
        index = child.expect(patterns, timeout=min(10.0, max(0.1, deadline - time.monotonic())))
        if index == 0:
            child.sendline(args.duo_option)
            duo_sent = True
            LOGGER.warning("Duo Push requested with option %s; waiting for phone approval", args.duo_option)
        elif index == 1:
            LOGGER.info("Duo approved; SSH authentication succeeded")
        elif index == 2:
            LOGGER.error("SSH requested a password; watchdog refuses to persist or replay passwords")
            child.close(force=True)
            return None
        elif index in (3, 4):
            LOGGER.error("SSH recovery exited before the tunnel became healthy")
            child.close(force=True)
            return None
        if endpoint_healthy(args.base_url, args.health_timeout_seconds):
            LOGGER.info("GLM tunnel healthy at %s", args.base_url)
            return child
        if index == 5 and not child.isalive():
            return None
    LOGGER.error(
        "SSH recovery timed out%s",
        " after requesting Duo approval" if duo_sent else " before Duo prompt",
    )
    child.close(force=True)
    return None


def matrix_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python_executable,
        "-u",
        "-m",
        MATRIX_MODULE,
        "--benchmark-split",
        args.benchmark_split,
        "--evaluation-subset",
        args.evaluation_subset,
        "--visibility-mode",
        args.visibility_mode,
        "--neighbor-identity-policy",
        args.neighbor_identity_policy,
        "--api-key-env",
        args.api_key_env,
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--output-root",
        args.output_root,
        "--reasoning-effort",
        "",
        "--parallelism",
        str(args.parallelism),
        "--max-stage-requeues",
        str(getattr(args, "max_stage_requeues", 0)),
        "--timeout-s",
        str(args.timeout_s),
    ]


def start_matrix(args: argparse.Namespace) -> int:
    log_path = Path(args.launcher_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault(args.api_key_env, "local-vllm-no-auth")
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            matrix_command(args),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
    LOGGER.info("started resumable matrix pid=%s log=%s", process.pid, log_path)
    return process.pid


def run(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    args.output_root = str(output_root)
    completed = count_final_results(
        output_root,
        args.visibility_mode,
        args.neighbor_identity_policy,
    )
    healthy = endpoint_healthy(args.base_url, args.health_timeout_seconds)
    launcher_pids = find_matrix_pids(output_root)
    LOGGER.info(
        "initial state completed=%s/%s endpoint_healthy=%s launcher_pids=%s",
        completed,
        args.expected_results,
        healthy,
        launcher_pids,
    )
    if args.check_once:
        return 0

    next_duo_attempt = 0.0
    next_launcher_attempt = 0.0
    next_count = time.monotonic() + args.count_interval_seconds
    tunnel_child = None
    while True:
        now = time.monotonic()
        if now >= next_count:
            completed = count_final_results(
                output_root,
                args.visibility_mode,
                args.neighbor_identity_policy,
            )
            LOGGER.info("progress completed=%s/%s", completed, args.expected_results)
            next_count = now + args.count_interval_seconds
            if completed >= args.expected_results:
                LOGGER.info("expected result count reached; watchdog exiting successfully")
                return 0

        healthy = endpoint_healthy(args.base_url, args.health_timeout_seconds)
        launcher_pids = find_matrix_pids(output_root)
        if not healthy:
            if launcher_pids:
                _terminate_process_groups(launcher_pids, label="GLM matrix")
            if now >= next_duo_attempt:
                LOGGER.warning("GLM endpoint is unavailable; attempting SSH recovery")
                tunnel_child = reconnect_tunnel(args)
                next_duo_attempt = time.monotonic() + args.duo_retry_seconds
                healthy = endpoint_healthy(args.base_url, args.health_timeout_seconds)
                if healthy:
                    next_launcher_attempt = 0.0
            time.sleep(args.check_interval_seconds)
            continue

        if not launcher_pids and now >= next_launcher_attempt:
            completed = count_final_results(
                output_root,
                args.visibility_mode,
                args.neighbor_identity_policy,
            )
            if completed >= args.expected_results:
                LOGGER.info("expected result count reached; watchdog exiting successfully")
                return 0
            start_matrix(args)
            next_launcher_attempt = time.monotonic() + args.launcher_retry_seconds
        if tunnel_child is not None and not tunnel_child.isalive():
            tunnel_child = None
        time.sleep(args.check_interval_seconds)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(Path(args.watchdog_log))
    try:
        return run(args)
    except KeyboardInterrupt:
        LOGGER.info("watchdog interrupted; leaving current tunnel and matrix untouched")
        return 130
    except Exception:
        LOGGER.exception("watchdog failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
