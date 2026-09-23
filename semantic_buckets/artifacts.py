"""Resolve and load semantic/readout sidecars from an evidence-library release."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from predict.utils.json import sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = Path(__file__).resolve().parent / "releases"
TASK_ALIASES = {
    "bbb": "bbb_martins",
    "bbb_martins": "bbb_martins",
    "oral": "bioavailability_ma",
    "bioavailability_ma": "bioavailability_ma",
    "skin": "skin_reaction",
    "skin_reaction": "skin_reaction",
    "ames": "ames",
    "dili": "dili",
    "carcinogens": "carcinogens",
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
    readout_bucket_map: Path | None
    record_readout_bucket_map: Path | None
    record_semantic_bucket_map: Path | None
    semantic_bucket_rankings: Path | None
    record_relevance_rankings: Path | None
    retrieval_eligibility: Path | None
    semantic_bucket_weights: Path | None


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
    if not selected or "/" in selected or "\\" in selected:
        raise ValueError(f"invalid evidence-library CURRENT pointer: {current}")
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
    task: str, release: str = "CURRENT"
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
    weights = selected.get("semantic_bucket_weights")
    weights_path = _selected_path(root, selected, "semantic_bucket_weights") if weights else None
    readout = selected.get("readout_bucket_map")
    record_readout = selected.get("record_readout_bucket_map")
    record_semantic = selected.get("record_semantic_bucket_map")
    if not record_readout and not record_semantic:
        raise ValueError(f"semantic-bucket manifest has no record mapping: {manifest_path}")
    return SemanticBucketArtifacts(
        task=task,
        release=release,
        root=root,
        manifest=manifest_path,
        semantic_map=_selected_path(root, selected, "semantic_map"),
        semantic_map_manifest=_selected_path(root, selected, "semantic_map_manifest"),
        readout_bucket_map=_selected_path(root, selected, "readout_bucket_map") if readout else None,
        record_readout_bucket_map=(
            _selected_path(root, selected, "record_readout_bucket_map")
            if record_readout else None
        ),
        record_semantic_bucket_map=(
            _selected_path(root, selected, "record_semantic_bucket_map")
            if record_semantic else None
        ),
        semantic_bucket_rankings=(
            _selected_path(root, selected, "semantic_bucket_rankings")
            if selected.get("semantic_bucket_rankings") else None
        ),
        record_relevance_rankings=(
            _selected_path(root, selected, "record_relevance_rankings")
            if selected.get("record_relevance_rankings") else None
        ),
        retrieval_eligibility=eligibility_path,
        semantic_bucket_weights=weights_path,
    )


def load_reviewed_record_weights(
    task: str, release: str = "CURRENT",
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Load selected record weights once per release-manifest hash."""
    task = normalize_task(task)
    release = resolve_release(task, release)
    manifest = semantic_bucket_root(task, release) / "manifest.json"
    return _cached_record_weights(task, release, sha256_file(manifest))


@lru_cache(maxsize=1)
def _cached_record_weights(
    task: str, release: str, manifest_sha256: str,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    artifacts = resolve_semantic_bucket_artifacts(task, release)
    document = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    rankings = artifacts.record_relevance_rankings
    if document.get("status") != "complete_reviewed" or rankings is None:
        raise ValueError(f"No reviewed semantic weights selected for {task}/{release}")
    entry = document.get("files", {}).get("record_relevance_rankings", {})
    if (sha256_file(artifacts.manifest) != manifest_sha256
            or entry.get("sha256") != sha256_file(rankings)):
        raise ValueError(f"Selected semantic rankings changed for {task}/{release}")
    rows = {}
    for row in pd.read_parquet(rankings, columns=[
        "source_row_uid", "level", "semantic_bucket_id", "weight",
    ]).itertuples(index=False):
        key = (str(row.source_row_uid), str(row.level))
        weight = float(row.weight)
        if key in rows or not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError(f"Invalid reviewed semantic assignment: {task}/{release}/{key}")
        rows[key] = {"semantic_bucket_id": str(row.semantic_bucket_id),
                     "semantic_weight": weight}
    return rows, {
        "release": release, "release_manifest": str(artifacts.manifest),
        "release_manifest_sha256": manifest_sha256,
        "record_relevance_rankings": str(rankings),
        "record_relevance_rankings_sha256": entry["sha256"],
        "weighted_record_count": len(rows),
    }


def load_record_bucket_map(
    task: str,
    release: str = "CURRENT",
    *,
    eligible_only: bool = False,
) -> pd.DataFrame:
    """Load record-to-semantic/readout assignments, optionally eligibility-filtered."""

    artifacts = resolve_semantic_bucket_artifacts(task, release)
    mapping_path = artifacts.record_semantic_bucket_map or artifacts.record_readout_bucket_map
    if mapping_path is None:
        raise ValueError(f"no record mapping is selected for {artifacts.task}")
    mapping = pd.read_parquet(mapping_path)
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
