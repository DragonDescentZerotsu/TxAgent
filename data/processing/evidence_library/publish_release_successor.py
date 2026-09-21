"""Republish a validated release with durable manifest paths and unchanged data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

from data.processing.evidence_library.build_release_level_mapping import input_sha256
from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from data.processing.evidence_library.versions.v10.build_current_release import (
    GOLD_OWNED_LEVEL_TASKS,
    _build_gold_owned_level_projection,
    _validate_published_manifest_paths,
    _validate_release,
)
from data.processing.evidence_library.versions.v10.build_normalized_evidence_library import (
    load_task_policy,
)
from data.processing.evidence_library.versions.v10.task_registry import import_task_module
from data.processing.paths import (
    EVIDENCE_LIBRARIES_ROOT,
    KNOWN_RELEASES,
    REPO_ROOT,
    evidence_library_root,
)
from tools.chembl_tool.common.json_utils import write_json_atomic


STAGE_PARTS = (
    "00_source",
    "01_cleaned",
    "02_canonicalized",
    "03_pair_buckets",
    "level_mapping",
    "reviews",
)


def _registry_paths_by_hash(task: str) -> dict[str, Path]:
    registry = import_task_module(task, "mapping_registry").mapping_registry()
    return {
        str(entry["sha256"]): (REPO_ROOT / str(entry["path"])).resolve()
        for entry in registry["mappings"].values()
        if entry.get("path") and entry.get("sha256")
    }


def _internal_suffix(value: str, source_root: Path) -> Path | None:
    path = Path(value)
    source_forms = (str(source_root), str(source_root.relative_to(REPO_ROOT)))
    for prefix in source_forms:
        if value == prefix or value.startswith(prefix + "/"):
            return Path(value[len(prefix) :].lstrip("/"))
    if value.startswith(("/local/", "/tmp/")):
        for part in STAGE_PARTS:
            if part in path.parts:
                return Path(*path.parts[path.parts.index(part) :])
    return None


def _rewrite_manifests(
    root: Path,
    *,
    source_root: Path,
    published_root: Path,
    durable_by_hash: dict[str, Path],
) -> None:
    hash_aliases: dict[str, str] = {}

    def rewrite(value: Any) -> Any:
        if isinstance(value, dict):
            result = dict(value)
            for key, child in value.items():
                if isinstance(child, str) and "path" in key.lower():
                    suffix = _internal_suffix(child, source_root)
                    if suffix is not None:
                        result[key] = str(published_root / suffix)
                        continue
                    if child.startswith(("/local/", "/tmp/")):
                        digest = str(
                            value.get("sha256")
                            or value.get(key.replace("path", "sha256"))
                            or ""
                        )
                        durable = durable_by_hash.get(digest)
                        if durable is None:
                            raise ValueError(f"no durable hash-matched path for {child}")
                        result[key] = str(durable)
                        continue
                result[key] = rewrite(child)
            return result
        if isinstance(value, list):
            return [rewrite(child) for child in value]
        if isinstance(value, str) and value == source_root.name:
            return published_root.name
        return value

    manifests = sorted(root.rglob("*.json"))
    for path in manifests:
        before = file_sha256(path)
        payload = rewrite(json.loads(path.read_text(encoding="utf-8")))
        write_json_atomic(path, payload)
        after = file_sha256(path)
        if before != after:
            hash_aliases[before] = after

    for _ in range(20):
        changed = False

        def replace_hashes(value: Any) -> Any:
            nonlocal changed
            if isinstance(value, dict):
                return {key: replace_hashes(child) for key, child in value.items()}
            if isinstance(value, list):
                return [replace_hashes(child) for child in value]
            if isinstance(value, str) and value in hash_aliases:
                replacement = hash_aliases[value]
                while replacement in hash_aliases:
                    replacement = hash_aliases[replacement]
                changed |= replacement != value
                return replacement
            return value

        for path in manifests:
            before = file_sha256(path)
            payload = replace_hashes(json.loads(path.read_text(encoding="utf-8")))
            write_json_atomic(path, payload)
            after = file_sha256(path)
            if before != after:
                hash_aliases[before] = after
        if not changed:
            return
    raise RuntimeError("manifest hash references did not converge")


def finalize_successor(
    task: str,
    source_version: str,
    target_version: str,
    root: Path,
) -> dict[str, Any]:
    final_root = EVIDENCE_LIBRARIES_ROOT / task / target_version
    if final_root.exists():
        raise FileExistsError(final_root)
    result = _validate_release(task, root)
    result["normalized_root"] = str(final_root.resolve())
    receipt = {
        "version": "evidence_library_metadata_successor.v1",
        "status": "complete",
        "task": task,
        "source_release": source_version,
        "target_release": target_version,
        "scientific_record_artifacts_recomputed": False,
        "level_projection_regenerated_from_reviewed_map": (
            task in GOLD_OWNED_LEVEL_TASKS
        ),
        "validation": result,
    }
    write_json_atomic(root / "publication_receipt.json", receipt)
    _validate_published_manifest_paths(task, root)
    staging_parent = root.parent
    os.replace(root, final_root)
    staging_parent.rmdir()
    return result


def prepare_successor(
    task: str, source_version: str, target_version: str
) -> dict[str, Any]:
    if target_version not in KNOWN_RELEASES.get(task, ()):
        raise ValueError(f"unsupported successor release: {task}/{target_version}")
    source_root = evidence_library_root(task, source_version)
    final_root = EVIDENCE_LIBRARIES_ROOT / task / target_version
    staging_parent = EVIDENCE_LIBRARIES_ROOT / task / f".staging-{target_version}-{os.getpid()}"
    root = staging_parent / target_version
    if final_root.exists() or staging_parent.exists():
        raise FileExistsError(final_root if final_root.exists() else staging_parent)
    shutil.copytree(source_root, root)
    (root / "publication_receipt.json").unlink(missing_ok=True)

    policy = load_task_policy(task)
    if task in GOLD_OWNED_LEVEL_TASKS:
        stage3_manifest_path = root / "03_pair_buckets/manifest.json"
        stage3_manifest = json.loads(stage3_manifest_path.read_text(encoding="utf-8"))
        stage3_manifest["input_hashes"]["reviewed_level_mapping"] = input_sha256(
            Path(policy.source_universe_mapping)
        )
        write_json_atomic(stage3_manifest_path, stage3_manifest)
        shutil.rmtree(root / "level_mapping")
        _build_gold_owned_level_projection(task, policy, root)

    durable_by_hash = _registry_paths_by_hash(task)
    pruning_manifest = root / "reviews/assay_transfer_record_pruning_v4/manifest.json"
    if pruning_manifest.is_file():
        pruning = json.loads(pruning_manifest.read_text(encoding="utf-8"))
        provider = (pruning.get("review") or {}).get("provider_pool") or {}
        transient = Path(str(provider.get("config_path") or ""))
        if str(transient).startswith(("/local/", "/tmp/")):
            destination = pruning_manifest.parent / "provider_pool.json"
            shutil.copy2(transient, destination)
            if file_sha256(destination) != provider.get("config_sha256"):
                raise ValueError("provider-pool snapshot hash mismatch")
            durable_by_hash[file_sha256(destination)] = (
                final_root / destination.relative_to(root)
            ).resolve()

    _rewrite_manifests(
        root,
        source_root=source_root.resolve(),
        published_root=final_root.resolve(),
        durable_by_hash=durable_by_hash,
    )
    return finalize_successor(task, source_version, target_version, root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--target-version", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prepare_successor(args.task, args.source_version, args.target_version),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
