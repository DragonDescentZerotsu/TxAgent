"""Wait for a detached agent launcher, then run gated router finalization."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from tools.chembl_tool.common.json_utils import write_json_atomic

from .contract import DEFAULT_OUTPUT_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-pid", type=int, required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Finalize only these tasks after the watched agent exits.",
    )
    parser.add_argument(
        "--resume-tasks",
        nargs="+",
        help=(
            "After the priority-task finalization succeeds, resume these tasks "
            "and then finalize the complete matrix."
        ),
    )
    parser.add_argument("--parallelism", type=int, default=128)
    args = parser.parse_args(argv)
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")
    output_root = Path(args.output_root)
    state_path = output_root / "postprocess_watch.json"
    state: dict[str, Any] = {
        "schema_version": "router_oof_postprocess_watch.v1",
        "watcher_pid": os.getpid(),
        "agent_pid": args.agent_pid,
        "output_root": str(output_root),
        "tasks": args.tasks,
        "resume_tasks": args.resume_tasks,
        "status": "waiting_for_agent",
        "started_at": _now(),
    }
    write_json_atomic(state_path, state)
    while _is_router_agent(args.agent_pid):
        time.sleep(args.poll_seconds)

    state.update({"status": "finalizing", "agent_finished_at": _now()})
    write_json_atomic(state_path, state)
    command = [
        sys.executable,
        "-m",
        "tools.chembl_tool.paper_experiments.router_oof.cli",
        "finalize",
        "--output-root",
        str(output_root),
    ]
    if args.tasks:
        command.extend(["--tasks", *args.tasks])
    completed = subprocess.run(command, check=False)
    state["priority_finalize_returncode"] = completed.returncode
    if completed.returncode == 0 and args.resume_tasks:
        state.update({"status": "resuming_agent", "resume_started_at": _now()})
        write_json_atomic(state_path, state)
        resume_command = [
            sys.executable,
            "-m",
            "tools.chembl_tool.paper_experiments.router_oof.cli",
            "run-agent",
            "--output-root",
            str(output_root),
            "--tasks",
            *args.resume_tasks,
            "--parallelism",
            str(args.parallelism),
        ]
        resumed = subprocess.run(resume_command, check=False)
        state["resume_returncode"] = resumed.returncode
        if resumed.returncode == 0:
            state.update({"status": "finalizing_all", "resume_finished_at": _now()})
            write_json_atomic(state_path, state)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.chembl_tool.paper_experiments.router_oof.cli",
                    "finalize",
                    "--output-root",
                    str(output_root),
                ],
                check=False,
            )
        else:
            completed = resumed
    state.update(
        {
            "status": "complete" if completed.returncode == 0 else "blocked",
            "finalize_returncode": completed.returncode,
            "finished_at": _now(),
        }
    )
    write_json_atomic(state_path, state)
    return completed.returncode


def _is_router_agent(pid: int) -> bool:
    path = Path(f"/proc/{pid}/cmdline")
    try:
        parts = path.read_bytes().split(b"\0")
    except OSError:
        return False
    return any(part.endswith(b"router_oof.cli") for part in parts) and b"run-agent" in parts


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
