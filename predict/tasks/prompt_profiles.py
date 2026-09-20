"""Shared prompt-profile provenance and artifact-reuse guards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def prompt_profile_from_manifest(
    manifest: dict,
    *,
    historical_profile: str,
) -> str:
    """Read current profile fields, mapping old manifests to a frozen legacy profile."""
    return str(
        manifest.get("task_prompt_profile")
        or manifest.get("skin_prompt_profile")
        or historical_profile
    )


def require_matching_prompt_profiles(
    *,
    target_profile: str,
    source_dirs: Iterable[str | Path],
    historical_profile: str,
) -> None:
    """Reject branch reuse when source and target prompt contracts differ."""
    for source in source_dirs:
        if not source:
            continue
        source_dir = Path(source)
        manifest_path = _reuse_manifest_path(source_dir)
        if not manifest_path.exists():
            raise ValueError(
                f"Prompt-profile reuse source lacks manifest: {source_dir}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_profile = prompt_profile_from_manifest(
            manifest,
            historical_profile=historical_profile,
        )
        if source_profile != target_profile:
            raise ValueError(
                "Prompt-profile branch reuse is forbidden: "
                f"source={source_profile!r} target={target_profile!r} "
                f"artifact={source_dir}"
            )
        if manifest.get("schema_version") == "branch_query_prior_overlay.v1":
            for entry in manifest.get("sources") or []:
                profile_manifest = Path(str(entry["prompt_profile_manifest"]))
                digest = hashlib.sha256(profile_manifest.read_bytes()).hexdigest()
                if digest != entry.get("prompt_profile_manifest_sha256"):
                    raise ValueError(
                        f"Query-prior source prompt manifest changed: {profile_manifest}"
                    )
                source_manifest = json.loads(
                    profile_manifest.read_text(encoding="utf-8")
                )
                nested_profile = prompt_profile_from_manifest(
                    source_manifest,
                    historical_profile=historical_profile,
                )
                if (
                    nested_profile != target_profile
                    or entry.get("task_prompt_profile") != target_profile
                ):
                    raise ValueError(
                        "Prompt-profile overlay source reuse is forbidden: "
                        f"source={nested_profile!r} target={target_profile!r} "
                        f"artifact={entry.get('run_dir')!r}"
                    )


def _reuse_manifest_path(source_dir: Path) -> Path:
    """Resolve either a completed run manifest or its batch-level contract.

    Global-pool preparation is concurrent, so a source run may not have written
    its own manifest yet. The enclosing batch manifest is created before any
    per-run preparation and carries the same frozen prompt profile.
    """
    run_manifest = source_dir / "manifest.json"
    if run_manifest.exists():
        return run_manifest
    if source_dir.parent.name == "runs":
        batch_manifest = source_dir.parent.parent / "manifest.json"
        if batch_manifest.exists():
            return batch_manifest
    return run_manifest
