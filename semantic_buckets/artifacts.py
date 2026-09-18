"""Resolve and load semantic/readout sidecars from an evidence-library release."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = Path(__file__).resolve().parent / "releases"
TASK_ALIASES = {
    "bbb": "bbb_martins",
    "bbb_martins": "bbb_martins",
    "oral": "bioavailability_ma",
    "bioavailability_ma": "bioavailability_ma",
}


@dataclass(frozen=True)
class SemanticBucketArtifacts:
    """Selected semantic/readout artifacts for one evidence-library release."""

    task: str
    release: str
    root: Path
    manifest: Path
    semantic_map: Path
    semantic_map_manifest: Path
    readout_bucket_map: Path
    record_readout_bucket_map: Path
    semantic_bucket_rankings: Path
    record_relevance_rankings: Path
    retrieval_eligibility: Path | None


def normalize_task(task: str) -> str:
    try:
        return TASK_ALIASES[task]
    except KeyError as exc:
        raise ValueError(f"unsupported semantic-bucket task: {task}") from exc


def resolve_release(task: str, release: str = "v10") -> str:
    task = normalize_task(task)
    if release != "CURRENT":
        return release
    current = REPOSITORY_ROOT / "data" / "evidence_libraries" / task / "CURRENT"
    selected = current.read_text(encoding="utf-8").strip()
    if not selected:
        raise ValueError(f"empty evidence-library CURRENT pointer: {current}")
    return selected


def semantic_bucket_root(task: str, release: str = "v10") -> Path:
    task = normalize_task(task)
    return RELEASE_ROOT / task / resolve_release(task, release)


def generation_root(task: str, generation: str, release: str = "v10") -> Path:
    return semantic_bucket_root(task, release) / "generations" / generation


def eligibility_root(task: str, generation: str, release: str = "v10") -> Path:
    return semantic_bucket_root(task, release) / "eligibility" / generation


def _selected_path(root: Path, selected: dict[str, Any], key: str) -> Path:
    value = selected.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"semantic-bucket manifest has no selected.{key}")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"semantic-bucket manifest path escapes its release: {value}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resolve_semantic_bucket_artifacts(
    task: str, release: str = "v10"
) -> SemanticBucketArtifacts:
    """Resolve the reviewed artifact selection recorded by a release manifest."""

    task = normalize_task(task)
    release = resolve_release(task, release)
    root = semantic_bucket_root(task, release)
    manifest_path = root / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "semantic_buckets.release.v1":
        raise ValueError(f"unsupported semantic-bucket manifest: {manifest_path}")
    if document.get("task") != task or document.get("evidence_library_version") != release:
        raise ValueError(f"semantic-bucket manifest identity mismatch: {manifest_path}")
    selected = document.get("selected")
    if not isinstance(selected, dict):
        raise ValueError(f"semantic-bucket manifest has no selected mapping: {manifest_path}")
    eligibility = selected.get("retrieval_eligibility")
    eligibility_path = (
        _selected_path(root, selected, "retrieval_eligibility") if eligibility else None
    )
    return SemanticBucketArtifacts(
        task=task,
        release=release,
        root=root,
        manifest=manifest_path,
        semantic_map=_selected_path(root, selected, "semantic_map"),
        semantic_map_manifest=_selected_path(root, selected, "semantic_map_manifest"),
        readout_bucket_map=_selected_path(root, selected, "readout_bucket_map"),
        record_readout_bucket_map=_selected_path(root, selected, "record_readout_bucket_map"),
        semantic_bucket_rankings=_selected_path(root, selected, "semantic_bucket_rankings"),
        record_relevance_rankings=_selected_path(root, selected, "record_relevance_rankings"),
        retrieval_eligibility=eligibility_path,
    )


def load_record_bucket_map(
    task: str,
    release: str = "v10",
    *,
    eligible_only: bool = False,
) -> pd.DataFrame:
    """Load record-to-semantic/readout assignments, optionally eligibility-filtered."""

    artifacts = resolve_semantic_bucket_artifacts(task, release)
    mapping = pd.read_parquet(artifacts.record_readout_bucket_map)
    if not eligible_only:
        return mapping
    if artifacts.retrieval_eligibility is None:
        raise ValueError(f"no retrieval eligibility is selected for {artifacts.task}")
    eligibility = pd.read_parquet(
        artifacts.retrieval_eligibility,
        columns=[
            "canonical_record_id",
            "source_row_uid",
            "level",
            "semantic_bucket_id",
            "retrieval_eligible",
        ],
    )
    keys = ["canonical_record_id", "source_row_uid", "level", "semantic_bucket_id"]
    if eligibility.duplicated(keys).any():
        raise ValueError(f"retrieval eligibility repeats record assignments for {artifacts.task}")
    joined = mapping.merge(eligibility, on=keys, how="inner", validate="one_to_one")
    if len(joined) != len(eligibility):
        raise ValueError(f"retrieval eligibility references unmapped records for {artifacts.task}")
    return joined[joined["retrieval_eligible"]].reset_index(drop=True)
