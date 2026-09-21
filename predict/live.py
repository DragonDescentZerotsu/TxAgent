"""Manage review-first prediction runs and publish viewer-safe trace snapshots.

Canonical harnesses register study runs under
``outputs/paper/live/<study>/<method>/<prior>/<run>/<task>/<condition>/``. Legacy
dataset/method/minute runs remain readable. Model stages update that
directory immediately.  This module owns run status, cancellation, immutable
prompt cloning, and the optional GitHub Pages data-branch publisher; it does
not own retrieval, prompt construction, or inference.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from predict.utils.json import write_json_atomic


DEFAULT_LIVE_ROOT = Path("outputs/paper/live")
DEFAULT_PAGES_REPO = "ssh://git@ssh.github.com:443/jiosephlee/jiosephlee.github.io.git"
PUBLIC_BRANCH = "live-traces"
PILOT_SIZE = 3
RUN_TIMEZONE = ZoneInfo("America/New_York")

_PRIVATE_KEYS = {
    "api_key",
    "api_key_env",
    "base_url",
    "checkpoint_path",
    "execution_provider",
    "execution_provider_attempts",
    "source_trace",
    "tool_service_url",
}
_PRIVATE_URL = re.compile(
    r"https?://(?:127\.0\.0\.1|localhost|10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.|(?:dgx|node|epyc)[^./:]*)[^\s\"']*"
)
_ABS_PATH = re.compile(r"(^|[=\s])/(?:vast|data1|home|tmp)/[^\s\"']+")


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "unnamed"


def minute_id() -> str:
    return datetime.now(RUN_TIMEZONE).strftime("%Y-%m-%d_%H-%M")


def ensure_process_group() -> int:
    """Give a foreground harness its own cancellable process group when safe."""
    try:
        if os.getpgrp() != os.getpid():
            os.setpgid(0, 0)
    except OSError:
        pass
    return os.getpgrp()


def create_run(
    *,
    root: str | Path,
    dataset: str,
    method: str,
    command: list[str],
    metadata: Mapping[str, Any],
    requested_id: str = "",
    execution_mode: str = "live",
    study: str = "",
    run_group: str = "",
    condition: str = "",
    query_prior: str = "",
    refresh_catalog: bool = True,
) -> Path:
    """Create or resume one dataset-method child run."""
    if execution_mode not in {"live", "throughput"}:
        raise ValueError("execution_mode must be 'live' or 'throughput'")
    root = Path(root)
    if study:
        if query_prior not in {"cached", "none"}:
            raise ValueError("organized live runs require query_prior='cached' or 'none'")
        group = safe_name(run_group or requested_id or minute_id())
        prior = "with_query_prior" if query_prior == "cached" else "no_query_prior"
        parent = (
            root / safe_name(study) / safe_name(method) / prior / group
            / safe_name(dataset)
        )
        run_dir = parent / safe_name(condition or "default")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        parent = root / safe_name(dataset) / safe_name(method)
        parent.mkdir(parents=True, exist_ok=True)
        run_dir = parent / safe_name(requested_id) if requested_id else parent / minute_id()
    if not study and requested_id:
        run_dir.mkdir(parents=True, exist_ok=True)
    elif not study:
        base = run_dir.name
        suffix = 1
        while True:
            try:
                run_dir.mkdir()
                break
            except FileExistsError:
                suffix += 1
                run_dir = parent / f"{base}-{suffix:02d}"
    ensure_process_group()
    path = run_dir / "run.json"
    previous = _read_json(path) if path.exists() else {}
    document = {
        **previous,
        **dict(metadata),
        "schema_version": "predict_live_run.v1",
        "dataset": dataset,
        "method": method,
        "run_id": safe_name(run_group) if study and run_group else run_dir.name,
        "study": study or previous.get("study"),
        "condition": condition or previous.get("condition"),
        "query_prior": query_prior or previous.get("query_prior"),
        "live_root_depth": len(run_dir.relative_to(root).parts),
        "status": "running",
        "created_at": previous.get("created_at") or _now(),
        "updated_at": _now(),
        "pid": os.getpid(),
        "process_group": os.getpgrp(),
        "process_started_at": _process_start(os.getpid()),
        "command": command,
        "execution_mode": execution_mode,
        "visibility": "public" if execution_mode == "live" else "private",
    }
    write_json_atomic(path, document)
    _write_public_run(run_dir)
    if refresh_catalog:
        _refresh_catalog(root)
    if document["visibility"] == "public":
        request_publish(root)
    return run_dir


def update_run(
    run_dir: str | Path, *, refresh_catalog: bool = True, **fields: Any,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    lock_path = run_dir / ".run.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = run_dir / "run.json"
        document = _read_json(path)
        document.update(fields, updated_at=_now())
        write_json_atomic(path, document)
    root = _live_root(run_dir, document)
    if refresh_catalog:
        _refresh_catalog(root)
    _write_public_run(run_dir)
    if document.get("visibility", "public") == "public":
        request_publish(root)
    return document


def record_trace(run_dir: str | Path, trace: Mapping[str, Any], path: Path) -> None:
    """Publish stage updates only for the run's three review samples."""
    run_dir = Path(run_dir)
    sample_id = safe_name(str(trace["sample_id"]))
    published_ids = [
        sample.name
        for sample in sorted((run_dir / "samples").iterdir())
        if sample.is_dir()
    ][:PILOT_SIZE]
    if sample_id not in published_ids:
        return
    _write_public_run(run_dir)
    root = _live_root(run_dir)
    _refresh_catalog(root)
    if _read_json(run_dir / "run.json").get("visibility", "public") == "public":
        request_publish(root)


def promote_run(run_dir: str | Path) -> dict[str, Any]:
    """Make a private run visible now and publish future stages as they arrive."""
    run_dir = Path(run_dir)
    return update_run(
        run_dir,
        visibility="public",
        promoted_at=_now(),
    )


def sanitize(value: Any) -> Any:
    """Keep the reasoning chain while removing machine and transport details."""
    if isinstance(value, Mapping):
        return {
            str(key): sanitize(item)
            for key, item in value.items()
            if str(key) not in _PRIVATE_KEYS
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        if value.startswith("/"):
            return "[local path omitted]"
        value = _PRIVATE_URL.sub("[private endpoint omitted]", value)
        return _ABS_PATH.sub(lambda match: match.group(1) + "[local path omitted]", value)
    return value


def request_publish(root: str | Path) -> None:
    if os.environ.get("TXAGENT_LIVE_PUBLISH", "1") == "0":
        return
    root = Path(root).resolve()
    (root / ".publish-request").touch()
    with (root / "publisher.log").open("ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", "predict.live", "--trace-root", str(root), "publish"],
            cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.DEVNULL,
            stderr=log,
            start_new_session=True,
        )


def publish_snapshot(root: str | Path) -> None:
    """Push the public projection to the Pages repository data branch."""
    root = Path(root).resolve()
    remote = os.environ.get("TXAGENT_LIVE_PAGES_REPO", DEFAULT_PAGES_REPO)
    remote_id = hashlib.sha256(remote.encode()).hexdigest()[:12]
    cache = Path(tempfile.gettempdir()) / f"txagent-live-pages-{os.getuid()}-{remote_id}"
    if not (cache / ".git").is_dir():
        subprocess.run(
            ["git", "clone", "--no-checkout", remote, str(cache)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    subprocess.run(["git", "fetch", "origin"], cwd=cache, check=True)
    remote_branch = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/remotes/origin/{PUBLIC_BRANCH}"],
        cwd=cache,
    ).returncode == 0
    if remote_branch:
        subprocess.run(
            ["git", "switch", "-C", PUBLIC_BRANCH, f"origin/{PUBLIC_BRANCH}"],
            cwd=cache,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    else:
        subprocess.run(
            ["git", "switch", "--orphan", PUBLIC_BRANCH],
            cwd=cache,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    target = cache / "traces-data"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copy2(root / "catalog.json", target / "catalog.json")
    for run in _run_documents(root):
        if run.get("visibility", "public") != "public":
            continue
        run_dir = Path(run["_run_dir"])
        _write_public_run(run_dir)
        source = run_dir / "public"
        destination = target / run_dir.relative_to(root)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    subprocess.run(["git", "add", "traces-data"], cwd=cache, check=True)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=cache).returncode
    if changed:
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=TxAgent live publisher",
                "-c",
                "user.email=txagent-live@localhost",
                "commit",
                "-m",
                f"Update live traces {minute_id()}",
            ],
            cwd=cache,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "push", "origin", f"HEAD:{PUBLIC_BRANCH}"],
            cwd=cache,
            check=True,
            stdout=subprocess.DEVNULL,
        )


def _publish_worker(root: Path) -> None:
    lock_path = root / ".publisher.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        marker = root / ".publish-request"
        try:
            while marker.exists():
                time.sleep(2)
                marker.unlink(missing_ok=True)
                publish_snapshot(root)
            (root / "publisher_error.json").unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - durable operator-visible failure
            write_json_atomic(
                root / "publisher_error.json",
                {"status": "publication_failed", "error": str(exc), "failed_at": _now()},
            )
            raise


def find_run(identifier: str, root: str | Path = DEFAULT_LIVE_ROOT) -> Path:
    candidate = Path(identifier)
    if (candidate / "run.json").is_file():
        return candidate.resolve()
    candidate = Path(root) / identifier
    if (candidate / "run.json").is_file():
        return candidate.resolve()
    matches = [
        Path(document["_run_dir"])
        for document in _run_documents(Path(root))
        if Path(document["_run_dir"]).name == identifier
        or document.get("run_id") == identifier
    ]
    if len(matches) != 1:
        raise SystemExit(f"run id must match exactly one run; found {len(matches)}: {identifier}")
    return matches[0].resolve()


def cancel_run(run_dir: Path) -> None:
    document = update_run(run_dir, status="cancelling", cancel_requested_at=_now())
    pid = int(document.get("pid") or 0)
    pgid = int(document.get("process_group") or 0)
    if pid > 1 and _same_process(pid, document.get("process_started_at")):
        target = -pgid if pgid == pid else pid
        try:
            os.kill(target, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and _process_exists(pid):
            time.sleep(0.2)
        if _process_exists(pid):
            try:
                os.kill(target, signal.SIGKILL)
            except ProcessLookupError:
                pass
    update_run(run_dir, status="cancelled", cancelled_at=_now())


def launch_saved(run_dir: Path, *, prompt_version: str = "") -> Path:
    document = _read_json(run_dir / "run.json")
    if prompt_version and document.get("harness") != "progressive":
        raise SystemExit("prompt relaunch currently supports progressive runs")
    command = list(document.get("resume_command") or document.get("command") or [])
    if not command:
        raise SystemExit("run has no saved resume command")
    root = _live_root(run_dir, document)
    if os.environ.get("TXAGENT_LIVE_PUBLISH", "1") != "0":
        try:
            publish_snapshot(root)
            (root / "publisher_error.json").unlink(missing_ok=True)
        except (OSError, subprocess.SubprocessError) as exc:
            update_run(run_dir, status="publication_failed", publication_error=str(exc))
            raise SystemExit("live publication failed; the run was not continued") from exc
    if prompt_version:
        command = _replace_option(command, "--prompt-version", prompt_version)
        command = _remove_option(command, "--live-run-id")
        command = _remove_option(command, "--execution-mode")
        command = [item for item in command if item != "--continue-after-pilot"]
        old_output = str(document.get("output_root") or "")
        if old_output:
            new_output = f"{old_output}_{safe_name(prompt_version)}_{minute_id()}"
            command = _replace_option(command, "--output-root", new_output)
        new_run = ""
    else:
        command = _replace_option(command, "--live-run-id", run_dir.name)
        new_run = run_dir.name
    if new_run:
        command = _add_flag(command, "--continue-after-pilot")
        if "predict.harnesses.progressive" not in command:
            command = _add_flag(command, "--skip-existing")
    log_dir = run_dir if new_run else run_dir.parent
    log_path = log_dir / ("continue.log" if new_run else f"relaunch_{minute_id()}.log")
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    if new_run:
        update_run(
            run_dir,
            status="running",
            pid=process.pid,
            process_group=process.pid,
            process_started_at=_process_start(process.pid),
        )
    return log_path


def clone_prompt(run_dir: Path, version: str) -> Path:
    document = _read_json(run_dir / "run.json")
    source_version = str(document.get("prompt_version") or "")
    if not source_version:
        raise SystemExit("this run does not record a versioned progressive prompt")
    from predict.harnesses.progressive.prompt import (
        PROMPT_DIR,
        behavior_version,
        prompt_directory,
    )

    source = prompt_directory(source_version)
    destination = PROMPT_DIR / safe_name(version)
    if destination.exists():
        raise SystemExit(f"prompt bundle already exists: {destination}")
    shutil.copytree(source, destination)
    provenance_path = destination / "provenance.json"
    provenance = _read_json(provenance_path) if provenance_path.exists() else {}
    provenance.update(
        version=version,
        parent_version=source_version,
        runtime_parent_version=behavior_version(source_version),
        cloned_at=_now(),
        cloned_from_run=str(run_dir.relative_to(_live_root(run_dir, document))),
    )
    write_json_atomic(provenance_path, provenance)
    return destination


def _write_public_run(run_dir: Path) -> None:
    document = _read_json(run_dir / "run.json")
    review_path = run_dir / "review.json"
    review = _read_json(review_path) if review_path.is_file() else None
    if review is not None:
        if review.get("schema_version") != "predict_live_review.v1":
            raise ValueError(f"unsupported live review schema: {review_path}")
        annotations = review.get("samples")
        if not isinstance(annotations, dict) or not annotations:
            raise ValueError(f"live review must contain a non-empty samples map: {review_path}")
        published_ids = sorted(annotations)
        expected = review.get("expected_sample_count")
        if expected is not None and expected != len(published_ids):
            raise ValueError(
                f"live review sample count mismatch: expected {expected}, found {len(published_ids)}"
            )
        for sample_id in published_ids:
            sample_dir = run_dir / "samples" / sample_id
            if safe_name(sample_id) != sample_id or not sample_dir.is_dir():
                raise ValueError(f"live review sample is unavailable: {sample_id}")
            if not any(sample_dir.glob("*.json")):
                raise ValueError(f"live review sample has no trace stages: {sample_id}")
    else:
        published_ids = [
            sample.name
            for sample in sorted((run_dir / "samples").glob("*"))
            if sample.is_dir()
        ][:PILOT_SIZE]

    public = run_dir / "public"
    public.mkdir(parents=True, exist_ok=True)
    public_samples = public / "samples"
    for sample in public_samples.glob("*"):
        if sample.is_dir() and sample.name not in published_ids:
            shutil.rmtree(sample)
    for sample_id in published_ids:
        for trace_path in sorted((run_dir / "samples" / sample_id).glob("*.json")):
            write_json_atomic(
                public_samples / sample_id / trace_path.name,
                sanitize(_read_json(trace_path)),
            )
    samples: dict[str, list[str]] = {}
    for path in sorted(public_samples.glob("*/*.json")):
        samples.setdefault(path.parent.name, []).append(path.stem)
    root = _live_root(run_dir, document)
    document = sanitize(document)
    document["path"] = run_dir.relative_to(root).as_posix()
    document["samples"] = samples
    if review is not None:
        document["review"] = sanitize(review)
    write_json_atomic(public / "run.json", document)


def _refresh_catalog(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".catalog.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        for run in _run_documents(root):
            if run.get("visibility", "public") != "public":
                continue
            rows.append(
                {
                    key: run.get(key)
                    for key in (
                        "study",
                        "dataset",
                        "method",
                        "query_prior",
                        "run_id",
                        "status",
                        "created_at",
                        "updated_at",
                        "pilot_size",
                    )
                }
                | {
                    "path": Path(run["_run_dir"]).relative_to(root).as_posix()
                }
            )
        write_json_atomic(root / "catalog.json", {"schema_version": "predict_live_catalog.v2", "runs": rows})


def _run_documents(root: Path) -> list[dict[str, Any]]:
    rows = []
    paths = set(root.glob("*/*/*/run.json"))
    paths.update(root.glob("*/*/*/*/*/*/run.json"))
    for path in sorted(paths):
        try:
            rows.append({**_read_json(path), "_run_dir": str(path.parent)})
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def _live_root(run_dir: Path, document: Mapping[str, Any] | None = None) -> Path:
    document = document or _read_json(run_dir / "run.json")
    depth = int(document.get("live_root_depth") or 3)
    return run_dir.parents[depth - 1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(RUN_TIMEZONE).isoformat(timespec="seconds")


def _process_exists(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().split()
        if len(stat) > 2 and stat[2] == "Z":
            return False
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError, PermissionError):
        return False


def _process_start(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except (OSError, IndexError):
        return ""


def _same_process(pid: int, expected: Any) -> bool:
    return bool(expected) and _process_start(pid) == str(expected)


def _remove_option(command: list[str], option: str) -> list[str]:
    result = []
    skip = False
    for item in command:
        if skip:
            skip = False
            continue
        if item == option:
            skip = True
            continue
        if item.startswith(option + "="):
            continue
        result.append(item)
    return result


def _replace_option(command: list[str], option: str, value: str) -> list[str]:
    return [*_remove_option(command, option), option, value]


def _add_flag(command: list[str], flag: str) -> list[str]:
    return command if flag in command else [*command, flag]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_LIVE_ROOT)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for name in ("show", "cancel", "continue", "promote"):
        child = subparsers.add_parser(name)
        child.add_argument("run_id")
    clone = subparsers.add_parser("clone-prompt")
    clone.add_argument("run_id")
    clone.add_argument("--to", required=True)
    relaunch = subparsers.add_parser("relaunch")
    relaunch.add_argument("run_id")
    relaunch.add_argument("--prompt-version", required=True)
    subparsers.add_parser("list")
    publish = subparsers.add_parser("publish")
    publish.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "list":
        for row in _run_documents(args.trace_root):
            print(f"{row.get('dataset')}\t{row.get('method')}\t{row.get('run_id')}\t{row.get('status')}")
        return 0
    if args.action == "publish":
        publish_snapshot(args.trace_root) if args.once else _publish_worker(args.trace_root)
        return 0
    run_dir = find_run(args.run_id, args.trace_root)
    if args.action == "show":
        print(json.dumps(_read_json(run_dir / "run.json"), indent=2))
    elif args.action == "cancel":
        cancel_run(run_dir)
        print(run_dir)
    elif args.action == "continue":
        print(launch_saved(run_dir))
    elif args.action == "promote":
        promote_run(run_dir)
        print(run_dir)
    elif args.action == "clone-prompt":
        print(clone_prompt(run_dir, args.to))
    elif args.action == "relaunch":
        print(launch_saved(run_dir, prompt_version=args.prompt_version))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
