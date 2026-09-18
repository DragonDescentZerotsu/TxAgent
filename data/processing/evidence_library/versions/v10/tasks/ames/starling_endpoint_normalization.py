"""Reviewed, fail-closed endpoint normalization for the frozen Ames sources."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    normalize_endpoint_name,
)
from data.processing.evidence_library.shared.v2.normalization.measurements import (
    EndpointOrthography,
)

ENDPOINT_NORMALIZATION_VERSION = "ames_endpoint_normalization.v1"
TASK_ROOT = Path(__file__).resolve().parent
SOURCE_MANIFEST_PATH = TASK_ROOT / "source_manifest.json"
DEFAULT_ENDPOINT_MAPPING = (
    TASK_ROOT / "data_processing/endpoint_normalization_v1/endpoint_mapping.json"
)


def _normalized(value: object) -> str:
    return normalize_endpoint_name(value) or ""


def _validate_source_spec(
    source_id: str, spec: dict[str, Any], source_manifest: dict[str, Any]
) -> None:
    canonical = set(spec.get("canonical_endpoints") or ())
    aliases = dict(spec.get("aliases") or {})
    excluded = dict(spec.get("excluded") or {})
    raw_sets = (canonical, set(aliases), set(excluded))
    if not canonical or any(not isinstance(value, str) for value in canonical):
        raise ValueError(f"invalid canonical Ames endpoints for {source_id!r}")
    if (
        raw_sets[0] & raw_sets[1]
        or raw_sets[0] & raw_sets[2]
        or raw_sets[1] & raw_sets[2]
    ):
        raise ValueError(f"overlapping Ames endpoint decisions for {source_id!r}")
    if any(_normalized(value) != value or not value for value in canonical):
        raise ValueError(f"non-canonical Ames endpoint spelling for {source_id!r}")
    if any(target not in canonical for target in aliases.values()):
        raise ValueError(f"Ames endpoint alias has an invalid target for {source_id!r}")
    if any(not reason for reason in excluded.values()):
        raise ValueError(f"Ames endpoint exclusion lacks a reason for {source_id!r}")
    expected_hash = source_manifest["sources"][source_id]["parquet_sha256"]
    if spec.get("source_parquet_sha256") != expected_hash:
        raise ValueError(f"Ames endpoint mapping source hash drift for {source_id!r}")


@cache
def _mapping() -> dict[str, Any]:
    payload = json.loads(DEFAULT_ENDPOINT_MAPPING.read_text(encoding="utf-8"))
    source_manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("version") != ENDPOINT_NORMALIZATION_VERSION:
        raise ValueError("Ames endpoint mapping version mismatch")
    if payload.get("source_snapshot_version") != source_manifest.get("version"):
        raise ValueError("Ames endpoint mapping source snapshot mismatch")
    sources = payload.get("sources") or {}
    if set(sources) != set(source_manifest["sources"]):
        raise ValueError("Ames endpoint mapping source inventory mismatch")
    for source_id, spec in sources.items():
        _validate_source_spec(source_id, spec, source_manifest)
    return payload


def _source_spec(source_id: str) -> dict[str, Any]:
    try:
        return _mapping()["sources"][source_id]
    except KeyError as error:
        raise ValueError(f"unsupported Ames endpoint source: {source_id!r}") from error


def endpoint_decision(source_id: str, raw_endpoint: object) -> dict[str, str]:
    """Return the reviewed identity, alias, or exclusion for one raw endpoint."""
    raw = _normalized(raw_endpoint)
    spec = _source_spec(source_id)
    if raw in spec["canonical_endpoints"]:
        return {
            "raw_endpoint": raw,
            "canonical_endpoint": raw,
            "status": "identity",
            "reason": "reviewed_identity",
        }
    if raw in spec["aliases"]:
        return {
            "raw_endpoint": raw,
            "canonical_endpoint": spec["aliases"][raw],
            "status": "alias",
            "reason": "reviewed_alias",
        }
    if raw in spec["excluded"]:
        return {
            "raw_endpoint": raw,
            "canonical_endpoint": "",
            "status": "excluded",
            "reason": spec["excluded"][raw],
        }
    raise ValueError(f"unmapped Ames endpoint for {source_id!r}: {raw!r}")


def canonical_endpoint_name(source_id: str, raw_endpoint: object) -> str:
    """Return a canonical endpoint, or empty only for an explicit exclusion."""
    return endpoint_decision(source_id, raw_endpoint)["canonical_endpoint"]


def endpoint_orthography(source_id: str, raw_endpoint: object) -> EndpointOrthography:
    """Preserve each reviewed endpoint decision through Stage 2 normalization."""
    decision = endpoint_decision(source_id, raw_endpoint)
    status = {
        "identity": "unchanged",
        "alias": "reviewed_normalization",
        "excluded": "reviewed_exclusion",
    }[decision["status"]]
    return EndpointOrthography(
        endpoint_name=decision["raw_endpoint"],
        spacing_and_spelling_endpoint=decision["canonical_endpoint"],
        status=status,
        reason=decision["reason"],
        spacing_and_spelling_version=ENDPOINT_NORMALIZATION_VERSION,
    )


def expected_raw_endpoints(source_id: str) -> frozenset[str]:
    spec = _source_spec(source_id)
    return frozenset(
        (*spec["canonical_endpoints"], *spec["aliases"], *spec["excluded"])
    )


def validate_endpoint_inventory(
    source_id: str, raw_endpoints: list[str], *, strict: bool
) -> list[dict[str, str]]:
    """Validate every observed value and optionally require the frozen inventory."""
    observed = {_normalized(value) for value in raw_endpoints}
    decisions = [endpoint_decision(source_id, value) for value in sorted(observed)]
    if strict and observed != expected_raw_endpoints(source_id):
        missing = sorted(expected_raw_endpoints(source_id) - observed)
        extra = sorted(observed - expected_raw_endpoints(source_id))
        raise ValueError(
            f"Ames endpoint inventory drift for {source_id!r}: "
            f"missing={missing!r}, extra={extra!r}"
        )
    return decisions


def canonical_endpoints_by_source() -> dict[str, frozenset[str]]:
    return {
        source_id: frozenset(spec["canonical_endpoints"])
        for source_id, spec in _mapping()["sources"].items()
    }


__all__ = [
    "DEFAULT_ENDPOINT_MAPPING",
    "ENDPOINT_NORMALIZATION_VERSION",
    "canonical_endpoint_name",
    "canonical_endpoints_by_source",
    "endpoint_decision",
    "endpoint_orthography",
    "expected_raw_endpoints",
    "validate_endpoint_inventory",
]
