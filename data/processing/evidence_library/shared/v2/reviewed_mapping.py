"""Fail-closed review provenance for canonical mapping registries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from data.processing.paths import REPO_ROOT


REVIEW_REQUIRED_MAPPINGS = frozenset({"auxiliary_context", "exact_measurement_units"})

# Frozen predecessors remain usable, but any changed bytes require a reviewed
# successor receipt.  This avoids relabeling historical work as newly reviewed.
GRANDFATHERED_MAPPING_HASHES = {
    ("bbb_martins", "auxiliary_context"): "15599b9e4cb42fd350453b38cb702ca678729a5f3d8c0300c7769bf73170e2dd",
    ("bbb_martins", "exact_measurement_units"): "9568db3c7abc10bcf37e04b3025188943ff62a701649809f49ac27c9ce21511c",
    ("bioavailability_ma", "auxiliary_context"): "423cb60a17ac10e152092afdce3d225b2e663835aa7e455acd2768a105d196f5",
    ("bioavailability_ma", "exact_measurement_units"): "27672c54bc2bab0d30b181d562a5aa6eb5bd51809811c9ab53e05077c33031e0",
    ("skin_reaction", "auxiliary_context"): "28ccfb3986e2d32dbd71c7dee04814734b8b40850fd146bcda4e1d56ae97c1ac",
}


def validate_mapping_review(
    task: str, mapping_id: str, entry: Mapping[str, Any]
) -> None:
    """Require an exact historical pin or a hash-bound reviewed receipt."""
    if mapping_id not in REVIEW_REQUIRED_MAPPINGS:
        return
    mapping_hash = str(entry.get("sha256") or "")
    if mapping_hash == GRANDFATHERED_MAPPING_HASHES.get((task, mapping_id)):
        return
    exception = entry.get("acceptance_exception") or {}
    if exception:
        path = REPO_ROOT / str(exception.get("path") or "")
        if not path.is_file() or file_sha256(path) != exception.get("sha256"):
            raise ValueError(f"{mapping_id} lacks a valid acceptance exception")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        validations = receipt.get("validations") or {}
        if (
            receipt.get("version")
            != "accepted_unreviewed_canonical_mapping_exception.v1"
            or receipt.get("task_id") != task
            or receipt.get("mapping_id") != mapping_id
            or receipt.get("publication_status")
            != "accepted_without_additional_review"
            or receipt.get("mapping", {}).get("sha256") != mapping_hash
            or not (receipt.get("target_releases") or receipt.get("target_release"))
            or not validations
            or not all(value is True for value in validations.values())
        ):
            raise ValueError(f"{mapping_id} acceptance exception is incomplete")
        return
    reference = entry.get("review_manifest") or {}
    path = REPO_ROOT / str(reference.get("path") or "")
    if not path.is_file() or file_sha256(path) != reference.get("sha256"):
        raise ValueError(f"{mapping_id} lacks a valid reviewed mapping receipt")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validations = receipt.get("validations") or {}
    if (
        receipt.get("version") != "reviewed_canonical_mapping_receipt.v1"
        or receipt.get("task_id") != task
        or receipt.get("mapping_id") != mapping_id
        or receipt.get("publication_status") != "reviewed"
        or receipt.get("mapping", {}).get("sha256") != mapping_hash
        or not receipt.get("review_completion")
        or not validations
        or not all(value is True for value in validations.values())
    ):
        raise ValueError(f"{mapping_id} reviewed mapping receipt is incomplete")


def validate_mapping_acceptance_scope(
    task: str, registry: Mapping[str, Any], target_release: str
) -> None:
    """Keep explicit unreviewed acceptance limited to its named release."""
    for mapping_id, entry in (registry.get("mappings") or {}).items():
        reference = entry.get("acceptance_exception") or {}
        if not reference:
            continue
        path = REPO_ROOT / str(reference.get("path") or "")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        targets = receipt.get("target_releases") or [receipt.get("target_release")]
        if receipt.get("task_id") != task or target_release not in targets:
            raise ValueError(
                f"{mapping_id} acceptance exception is not valid for {target_release}"
            )


__all__ = ["validate_mapping_acceptance_scope", "validate_mapping_review"]
