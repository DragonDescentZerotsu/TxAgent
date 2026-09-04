"""Small shared runtime primitives for deterministic Starling builds.

This module deliberately contains execution mechanics only: ordered process
parallelism, content-keyed reuse metadata, file-digest memoization, and timing.
Scientific normalization and task policy remain in their existing modules.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import resource
import socket
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BUILD_RUNTIME_VERSION = "starling_build_runtime.v2"
BUILD_LOCK_FILENAME = ".build.lock"
INCOMPLETE_BUILD_FILENAME = ".build-incomplete.json"
_ACTIVE_BUILD_ROOTS: dict[Path, tuple[int, int, bool]] = {}
_NON_SEMANTIC_ARGUMENTS = frozenset(
    {
        "cache_mode",
        "from_stage",
        "legacy_task_local_downstream",
        "out_dir",
        "progress_every",
        "through_stage",
        "validation_level",
        "workers",
    }
)


def assert_unpublished_build_root(normalized_root: str | Path) -> None:
    """Reject completed current releases while allowing an incomplete resume."""
    root = Path(normalized_root).resolve()
    current = root.parent / "CURRENT"
    if (
        current.is_file()
        and current.read_text(encoding="utf-8").strip() == root.name
        and (root / "manifest.json").is_file()
        and not (root / INCOMPLETE_BUILD_FILENAME).is_file()
    ):
        raise RuntimeError(
            f"refusing to rebuild published evidence library {root}; "
            "build into an explicit staging root"
        )


@contextmanager
def starling_build_session(
    normalized_root: str | Path,
    *,
    complete: bool = False,
    mark_incomplete: bool = True,
):
    """Hold one non-blocking writer lock for a normalized task root."""
    root = Path(normalized_root).resolve()
    owner = (os.getpid(), threading.get_ident())
    active = _ACTIVE_BUILD_ROOTS.get(root)
    if active is not None and active[:2] == owner:
        yield
        return

    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / BUILD_LOCK_FILENAME
    marker_path = root / INCOMPLETE_BUILD_FILENAME
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            raise RuntimeError(
                f"Starling build already running for {root}: {owner}"
            ) from error

        receipt = {
            "command": sys.argv[0],
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        handle.seek(0)
        handle.truncate()
        json.dump(receipt, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        if mark_incomplete:
            marker_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        _ACTIVE_BUILD_ROOTS[root] = (*owner, complete)
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            if complete and succeeded:
                marker_path.unlink(missing_ok=True)
            _ACTIVE_BUILD_ROOTS.pop(root, None)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def complete_build_session_active(normalized_root: str | Path) -> bool:
    active = _ACTIVE_BUILD_ROOTS.get(Path(normalized_root).resolve())
    owner = (os.getpid(), threading.get_ident())
    return bool(active and active[:2] == owner and active[2])


@dataclass
class FileDigestCache:
    """Hash each unchanged path at most once during one process run."""

    _values: dict[tuple[str, int, int], str] = field(default_factory=dict)

    def sha256(self, path: str | Path) -> str:
        target = Path(path)
        stat = target.stat()
        key = (str(target.resolve()), stat.st_size, stat.st_mtime_ns)
        cached = self._values.get(key)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        self._values[key] = value
        return value


@dataclass
class BuildTimings:
    """Collect inexpensive wall-clock timings for stage manifests."""

    started: float = field(default_factory=time.monotonic)
    phases: dict[str, float] = field(default_factory=dict)

    def measure(self, name: str) -> "_MeasuredPhase":
        return _MeasuredPhase(self, name)

    def manifest(self) -> dict[str, Any]:
        return {
            "elapsed_s": round(time.monotonic() - self.started, 3),
            "phases_s": {key: round(value, 3) for key, value in self.phases.items()},
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        }


class _MeasuredPhase:
    def __init__(self, timings: BuildTimings, name: str) -> None:
        self.timings = timings
        self.name = name
        self.started = 0.0

    def __enter__(self) -> None:
        self.started = time.monotonic()

    def __exit__(self, *_: object) -> None:
        self.timings.phases[self.name] = self.timings.phases.get(
            self.name, 0.0
        ) + (time.monotonic() - self.started)


def semantic_arguments(args: Any, digests: FileDigestCache) -> dict[str, Any]:
    """Return data-affecting CLI arguments, hashing file-valued arguments."""
    output: dict[str, Any] = {}
    for key, value in sorted(vars(args).items()):
        if key in _NON_SEMANTIC_ARGUMENTS:
            continue
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str):
            path = Path(value)
            if path.is_file():
                output[key] = {
                    "path": value,
                    "sha256": digests.sha256(path),
                }
                continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            output[key] = value
        elif isinstance(value, (list, tuple)):
            output[key] = list(value)
        else:
            output[key] = str(value)
    return output


def implementation_fingerprint(
    task_id: str,
    digests: FileDigestCache,
    *,
    scientific_assets: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Fingerprint shared Starling code and the task's build-policy modules."""
    repo_root = Path(__file__).resolve().parents[5]
    common_root = Path(__file__).resolve().parent
    evidence_root = common_root.parents[1]
    tool_common_root = repo_root / "tools/chembl_tool/common"
    task_root = repo_root / "tools/chembl_tool/tasks" / task_id
    paths = sorted(common_root.rglob("*.py"))
    paths.extend(
        path
        for path in (
            evidence_root / "compact_artifacts.py",
            evidence_root / "evidence_library.py",
            tool_common_root / "evidence_contract.py",
            tool_common_root / "molecule_identity.py",
            tool_common_root / "units.py",
            tool_common_root / "task_workflows/evidence_library.py",
            tool_common_root / "contextual_unit_policy.json",
            tool_common_root / "qualifier_vocabulary_policy.json",
        )
        if path.is_file()
    )
    if task_root.is_dir():
        paths.extend(sorted(task_root.glob("starling*.py")))
        paths.extend(sorted(task_root.glob("build_starling*.py")))
        normalized_wrapper = task_root / "build_normalized_starling_evidence_library.py"
        if normalized_wrapper.is_file() and normalized_wrapper not in paths:
            paths.append(normalized_wrapper)
    declared_assets = [Path(path) for path in scientific_assets]
    missing_assets = [str(path) for path in declared_assets if not path.is_file()]
    if missing_assets:
        raise FileNotFoundError(
            "declared Starling scientific assets are missing: "
            + ", ".join(sorted(missing_assets))
        )
    paths.extend(declared_assets)
    files = {
        _fingerprint_path(path, repo_root): digests.sha256(path)
        for path in sorted(set(paths))
    }
    payload = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "version": BUILD_RUNTIME_VERSION,
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "files": files,
    }


def build_cache_metadata(
    *,
    task_id: str,
    completed_stage: str,
    args: Any,
    input_paths: Iterable[str | Path],
    output_paths: Iterable[str | Path],
    digests: FileDigestCache,
    scientific_assets: Iterable[str | Path] = (),
) -> dict[str, Any]:
    implementation = implementation_fingerprint(
        task_id,
        digests,
        scientific_assets=scientific_assets,
    )
    inputs = _path_digests(input_paths, digests)
    outputs = _path_digests(output_paths, digests)
    key_payload = {
        "version": BUILD_RUNTIME_VERSION,
        "task_id": task_id,
        "completed_stage": completed_stage,
        "implementation_sha256": implementation["sha256"],
        "semantic_arguments": semantic_arguments(args, digests),
        "inputs": inputs,
    }
    encoded = json.dumps(key_payload, sort_keys=True, separators=(",", ":"), default=str)
    return {
        **key_payload,
        "content_key": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "implementation": implementation,
        "outputs": outputs,
    }


def cache_metadata_matches(
    metadata: Mapping[str, Any] | None,
    *,
    task_id: str,
    completed_stage: str,
    args: Any,
    digests: FileDigestCache,
    scientific_assets: Iterable[str | Path] = (),
) -> bool:
    if not metadata or metadata.get("version") != BUILD_RUNTIME_VERSION:
        return False
    if metadata.get("task_id") != task_id or metadata.get("completed_stage") != completed_stage:
        return False
    implementation = implementation_fingerprint(
        task_id,
        digests,
        scientific_assets=scientific_assets,
    )
    if metadata.get("implementation_sha256") != implementation["sha256"]:
        return False
    if metadata.get("semantic_arguments") != semantic_arguments(args, digests):
        return False
    for inventory_name in ("inputs", "outputs"):
        inventory = metadata.get(inventory_name)
        if not isinstance(inventory, Mapping):
            return False
        for raw_path, expected in inventory.items():
            path = Path(str(raw_path))
            if not path.is_file() or digests.sha256(path) != str(expected):
                return False
    key_payload = {
        key: metadata.get(key)
        for key in (
            "version",
            "task_id",
            "completed_stage",
            "implementation_sha256",
            "semantic_arguments",
            "inputs",
        )
    }
    encoded = json.dumps(key_payload, sort_keys=True, separators=(",", ":"), default=str)
    return metadata.get("content_key") == hashlib.sha256(
        encoded.encode("utf-8")
    ).hexdigest()


def _path_digests(
    paths: Iterable[str | Path], digests: FileDigestCache
) -> dict[str, str]:
    output: dict[str, str] = {}
    for raw_path in sorted({str(Path(path)) for path in paths}):
        path = Path(raw_path)
        if path.is_file():
            output[raw_path] = digests.sha256(path)
    return output


def _fingerprint_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


_NORMALIZE_RECORDS: Sequence[Mapping[str, Any]] | None = None
_NORMALIZE_HOOKS: Any = None
_NORMALIZE_TASK: str | None = None
_NORMALIZE_POLICY: Any = None


def _normalize_range(bounds: tuple[int, int]) -> list[dict[str, Any]]:
    from data.processing.evidence_library.shared.v2.normalization.measurements import (
        normalize_cleaned_records,
    )

    if _NORMALIZE_RECORDS is None or _NORMALIZE_HOOKS is None:
        raise RuntimeError("normalization worker was not initialized")
    start, stop = bounds
    hooks = _NORMALIZE_HOOKS
    measurement_policy = None
    if _NORMALIZE_POLICY is not None:
        policy_path = _NORMALIZE_POLICY.assay_transfer_measurement_policy
        if policy_path is not None:
            from data.processing.evidence_library.shared.v2.assay_transfer_measurements import (
                load_measurement_policy,
            )

            measurement_policy = load_measurement_policy(policy_path)
    return normalize_cleaned_records(
        _NORMALIZE_RECORDS[start:stop],
        endpoint_normalizer=hooks.endpoint_normalizer,
        endpoint_standardizer=hooks.endpoint_standardizer,
        source_measurement_resolver=hooks.source_measurement_resolver,
        family_resolver=hooks.family_resolver,
        record_enricher=hooks.record_enricher,
        assay_transfer_measurement_policy=measurement_policy,
        assay_transfer_revalidator=hooks.assay_transfer_revalidator,
        task=_NORMALIZE_TASK,
    )


def _normalize_and_project_range(
    bounds: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from data.processing.evidence_library.compact_artifacts import (
        compact_persisted_records,
    )

    normalized = _normalize_range(bounds)
    from data.processing.evidence_library.shared.v2.normalization.audit import (
        validate_stage_schema,
    )

    schema_errors = validate_stage_schema(normalized, "normalize")
    if schema_errors:
        raise ValueError(schema_errors[0])
    if _NORMALIZE_POLICY is None or _NORMALIZE_POLICY.record_contract is None:
        raise RuntimeError("canonical projection worker has no v7 policy")
    attached = _NORMALIZE_POLICY.attach_source_columns(normalized)
    policy_path = _NORMALIZE_POLICY.assay_transfer_measurement_policy
    if policy_path is None:
        projected = [
            _NORMALIZE_POLICY.record_contract.canonical_projection(record)
            for record in attached
        ]
        return attached, compact_persisted_records(projected)
    from data.processing.evidence_library.shared.v2.assay_transfer_measurements import (
        finalize_assay_transfer_measurement,
        load_measurement_policy,
    )

    measurement_policy = load_measurement_policy(policy_path)
    transformed, projected = [], []
    for record in attached:
        base = _NORMALIZE_POLICY.record_contract.canonical_projection(record)
        working, persisted = finalize_assay_transfer_measurement(
            record,
            base,
            record_contract=_NORMALIZE_POLICY.record_contract,
            policy=measurement_policy,
            prune_unreviewed_record=(
                _NORMALIZE_POLICY.prune_unreviewed_assay_transfer_records
            ),
        )
        transformed.append(working)
        projected.append(persisted)
    return transformed, compact_persisted_records(projected)


def normalize_records_ordered(
    records: Sequence[Mapping[str, Any]],
    *,
    hooks: Any,
    task: str,
    workers: int,
    chunk_rows: int = 10_000,
) -> list[dict[str, Any]]:
    """Normalize contiguous chunks in processes and restore exact input order."""
    from data.processing.evidence_library.shared.v2.normalization.measurements import (
        normalize_cleaned_records,
    )

    workers = max(1, int(workers or 1))
    if workers == 1 or len(records) < max(2_000, chunk_rows):
        return normalize_cleaned_records(
            records,
            endpoint_normalizer=hooks.endpoint_normalizer,
            endpoint_standardizer=hooks.endpoint_standardizer,
            source_measurement_resolver=hooks.source_measurement_resolver,
            family_resolver=hooks.family_resolver,
            record_enricher=hooks.record_enricher,
            task=task,
        )
    if "fork" not in mp.get_all_start_methods():
        return normalize_records_ordered(
            records, hooks=hooks, task=task, workers=1, chunk_rows=chunk_rows
        )
    bounds = [
        (start, min(start + chunk_rows, len(records)))
        for start in range(0, len(records), chunk_rows)
    ]
    global _NORMALIZE_RECORDS, _NORMALIZE_HOOKS, _NORMALIZE_TASK
    _NORMALIZE_RECORDS = records
    _NORMALIZE_HOOKS = hooks
    _NORMALIZE_TASK = task
    try:
        with mp.get_context("fork").Pool(processes=workers) as pool:
            chunks = pool.map(_normalize_range, bounds, chunksize=1)
        return [record for chunk in chunks for record in chunk]
    finally:
        _NORMALIZE_RECORDS = None
        _NORMALIZE_HOOKS = None
        _NORMALIZE_TASK = None


def normalize_and_project_records_ordered(
    records: Sequence[Mapping[str, Any]],
    *,
    hooks: Any,
    policy: Any,
    workers: int,
    chunk_rows: int = 10_000,
    release_input: bool = False,
    retain_working: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize, attach source columns, and project v7 rows in one pass."""
    workers = max(1, int(workers or 1))
    bounds = [
        (start, min(start + chunk_rows, len(records)))
        for start in range(0, len(records), chunk_rows)
    ]
    global _NORMALIZE_RECORDS, _NORMALIZE_HOOKS, _NORMALIZE_TASK, _NORMALIZE_POLICY
    _NORMALIZE_RECORDS = records
    _NORMALIZE_HOOKS = hooks
    _NORMALIZE_TASK = policy.task_id
    _NORMALIZE_POLICY = policy
    try:
        if (
            workers == 1
            or len(records) < max(2_000, chunk_rows)
            or "fork" not in mp.get_all_start_methods()
        ):
            normalized: list[dict[str, Any]] = []
            persisted: list[dict[str, Any]] = []
            for start, stop in bounds:
                normalized_chunk, persisted_chunk = _normalize_and_project_range(
                    (start, stop)
                )
                if retain_working:
                    normalized.extend(normalized_chunk)
                persisted.extend(persisted_chunk)
                if release_input:
                    if not isinstance(records, list):
                        raise TypeError("release_input requires a mutable list")
                    records[start:stop] = [None] * (stop - start)
            return (normalized if retain_working else persisted), persisted
        else:
            with mp.get_context("fork").Pool(processes=workers) as pool:
                chunks = pool.map(_normalize_and_project_range, bounds, chunksize=1)
        normalized = [record for chunk, _ in chunks for record in chunk]
        persisted = [record for _, chunk in chunks for record in chunk]
        return normalized, persisted
    finally:
        _NORMALIZE_RECORDS = None
        _NORMALIZE_HOOKS = None
        _NORMALIZE_TASK = None
        _NORMALIZE_POLICY = None


_CLEAN_SOURCE_BATCHES: Sequence[
    tuple[Any, Any, str, Mapping[str, str] | None]
] | None = None


def _clean_source_range(task: tuple[int, int, int]) -> list[dict[str, Any]]:
    from data.processing.evidence_library.shared.v2.normalization.cleaning import clean_source_rows

    if _CLEAN_SOURCE_BATCHES is None:
        raise RuntimeError("cleaning worker was not initialized")
    source_index, start, stop = task
    profile, rows, source_sha256, smiles_mapping = _CLEAN_SOURCE_BATCHES[source_index]
    sliced = (
        rows.iloc[start:stop].to_dict(orient="records")
        if hasattr(rows, "iloc")
        else rows[start:stop]
    )
    return clean_source_rows(
        sliced,
        profile,
        smiles_mapping=smiles_mapping,
        source_sha256=source_sha256,
        source_row_offset=start,
    )


def clean_sources_ordered(
    batches: Sequence[tuple[Any, Any, str, Mapping[str, str] | None]],
    *,
    workers: int,
    chunk_rows: int = 25_000,
) -> list[dict[str, Any]]:
    """Clean source-contiguous chunks while preserving source and row order."""
    tasks = [
        (source_index, start, min(start + chunk_rows, len(rows)))
        for source_index, (_, rows, _, _) in enumerate(batches)
        for start in range(0, len(rows), chunk_rows)
    ]
    workers = max(1, int(workers or 1))
    global _CLEAN_SOURCE_BATCHES
    _CLEAN_SOURCE_BATCHES = batches
    try:
        if workers == 1 or len(tasks) < 2 or "fork" not in mp.get_all_start_methods():
            chunks = [_clean_source_range(task) for task in tasks]
        else:
            with mp.get_context("fork").Pool(processes=workers) as pool:
                chunks = pool.map(_clean_source_range, tasks, chunksize=1)
        return [record for chunk in chunks for record in chunk]
    finally:
        _CLEAN_SOURCE_BATCHES = None


def _parent_identity_item(smiles: str) -> tuple[str, str]:
    from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity

    identity = normalize_molecule_identity(smiles)
    return smiles, identity.parent_inchi_key or identity.parent_smiles


def parent_identity_map(
    smiles_values: Iterable[str], *, workers: int = 1
) -> dict[str, str]:
    """Normalize unique molecular parents once, using an ordered global pool."""
    unique = sorted({str(value or "") for value in smiles_values if str(value or "")})
    workers = max(1, int(workers or 1))
    if workers == 1 or len(unique) < 256:
        return dict(_parent_identity_item(smiles) for smiles in unique)
    chunksize = max(1, min(256, len(unique) // (workers * 4)))
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        return dict(pool.imap(_parent_identity_item, unique, chunksize=chunksize))


__all__ = [
    "BUILD_RUNTIME_VERSION",
    "BuildTimings",
    "FileDigestCache",
    "build_cache_metadata",
    "cache_metadata_matches",
    "clean_sources_ordered",
    "implementation_fingerprint",
    "normalize_and_project_records_ordered",
    "normalize_records_ordered",
    "parent_identity_map",
    "semantic_arguments",
]
