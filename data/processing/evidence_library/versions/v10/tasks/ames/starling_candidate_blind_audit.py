"""Validate the frozen blind-v3 candidate audit before AMES Stage 2."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)

TASK_ROOT = Path(__file__).resolve().parent
CANDIDATE_ROOT = TASK_ROOT / "data_processing/measurement_resolution_v2"
CANDIDATE_PATH = CANDIDATE_ROOT / "measurement_candidates.parquet"
BLIND_PASS_RECEIPT_PATH = CANDIDATE_ROOT / "candidate_blind_audit_v3.score_report.json"

AUDIT_VERSION = "ames_candidate_generalization_blind_sample.v3"
SCORE_VERSION = "ames_candidate_generalization_blind_score.v3"
BLIND_SOURCES = frozenset(
    {"fixed_mutation", "mutagenicity_mechanism", "premutagenic_damage"}
)
IMPLEMENTATION_INPUTS = frozenset(
    {"generator", "writer", "compiler_contract", "source_fields"}
)
REPORT_INPUTS = frozenset(
    {
        "protocol",
        "scorer",
        "builder",
        "candidate_artifact",
        "candidate_manifest",
        *IMPLEMENTATION_INPUTS,
        "stage1_records",
        "sample",
        "sample_manifest",
        "reviewer_a",
        "reviewer_b",
        "adjudication",
        "final_labels",
        "label_manifest",
    }
)
REPORT_FIELDS = frozenset(
    {
        "score_version",
        "audit_version",
        "inputs",
        "rows",
        "label_status_counts",
        "label_reason_counts",
        "exact_absolute",
        "exact_absolute_by_source",
        "sample_candidate_evidence_grounding_failures",
        "candidate_inventory_validations",
        "gates",
        "decision",
    }
)
GATES = frozenset(
    {
        "overall_exact_absolute_at_least_95_percent",
        "each_absolute_source_at_least_90_percent",
        "zero_sample_candidate_evidence_grounding_failures",
        "candidate_inventory_recursive_validation_passed",
        "candidate_inventory_all_validations_passed",
        "candidate_inventory_no_truncation",
        "candidate_inventory_caps_passed",
        "candidate_inventory_identity_and_stage1_coverage_passed",
    }
)
CANDIDATE_VALIDATIONS = frozenset(
    {
        "exact_stage1_extract_coverage",
        "unique_cleaned_record_ids",
        "unique_source_row_uids",
        "nonempty_identity_fields",
        "known_sources",
        "complete_canonical_endpoints",
        "candidate_sets_hash_valid",
        "candidate_sets_not_truncated",
        "candidate_cap_respected",
        "candidate_field_byte_bound_respected",
        "candidate_json_byte_bound_respected",
    }
)
FROZEN_AUDIT_HASHES = {
    "protocol": "0b4ea8bac551d4fd8c845e30a3532a111c181083458ed125528593e5a4078e6d",
    "scorer": "3ac3f71d1072dd0a0eee3ca8c3cd2a9fc1df039d30734f191b09048b32ba18bf",
    "builder": "a829118cf7178874e79cc30beb0594a1a24ad71977dab2746b54dc956906266d",
}
FROZEN_CANDIDATE_HASHES = {
    "candidate_sha256": "0647b175944400ac9e0522b1472323e0310e47fb7f8da990c37ebdb6b9a4c404",
    "candidate_manifest_sha256": "ac5cde0a6724eaa536f52afdd9ce09b4626f3a2062e86394e3e1110affa8c0fe",
}
SHA256 = re.compile(r"[0-9a-f]{64}")


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _require_exact_true_map(value: object, fields: frozenset[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"blind audit {label} schema mismatch")
    if any(
        type(value[field]) is not bool or value[field] is not True for field in fields
    ):
        raise ValueError(f"blind audit {label} contains a failed claim")


def _validate_rate(value: object, minimum: float, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"correct", "total", "rate"}:
        raise ValueError(f"blind audit {label} schema mismatch")
    correct, total, rate = value["correct"], value["total"], value["rate"]
    if type(correct) is not int or type(total) is not int or total <= 0:
        raise ValueError(f"blind audit {label} counts are invalid")
    if not 0 <= correct <= total or isinstance(rate, bool):
        raise ValueError(f"blind audit {label} values are invalid")
    if not isinstance(rate, (int, float)) or not math.isfinite(rate):
        raise ValueError(f"blind audit {label} rate is invalid")
    if rate != correct / total or rate < minimum:
        raise ValueError(f"blind audit {label} is below its acceptance threshold")


def _validate_input_reference(reference: object, label: str) -> None:
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"blind audit {label} reference schema mismatch")
    path, digest = reference["path"], reference["sha256"]
    if not isinstance(path, str) or not path or not Path(path).is_absolute():
        raise ValueError(f"blind audit {label} path is not absolute")
    if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
        raise ValueError(f"blind audit {label} hash is malformed")


def _validate_report(report: Mapping[str, Any]) -> None:
    if set(report) != REPORT_FIELDS:
        raise ValueError("blind audit report schema mismatch")
    expected = {"audit_version": AUDIT_VERSION, "score_version": SCORE_VERSION}
    if any(report.get(field) != value for field, value in expected.items()):
        raise ValueError("blind audit version mismatch")
    if report.get("decision") != "PASS" or report.get("rows") != 90:
        raise ValueError("blind audit did not record the frozen PASS decision")
    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != REPORT_INPUTS:
        raise ValueError("blind audit input schema mismatch")
    for name, reference in inputs.items():
        _validate_input_reference(reference, name)
    for name, digest in FROZEN_AUDIT_HASHES.items():
        if inputs[name]["sha256"] != digest:
            raise ValueError(f"blind audit {name} hash mismatch")
    _require_exact_true_map(report.get("gates"), GATES, "gates")
    _require_exact_true_map(
        report.get("candidate_inventory_validations"),
        CANDIDATE_VALIDATIONS,
        "candidate validations",
    )
    grounding = report.get("sample_candidate_evidence_grounding_failures")
    if type(grounding) is not int or grounding != 0:
        raise ValueError("blind audit contains candidate evidence-grounding failures")
    _validate_rate(report.get("exact_absolute"), 0.95, "overall exact absolute")
    by_source = report.get("exact_absolute_by_source")
    if not isinstance(by_source, Mapping) or set(by_source) != BLIND_SOURCES:
        raise ValueError("blind audit source coverage mismatch")
    for source, rate in by_source.items():
        _validate_rate(rate, 0.90, f"{source} exact absolute")


def _validate_reference(reference: object, path: Path, label: str) -> None:
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"blind audit {label} reference schema mismatch")
    if Path(str(reference["path"])).resolve() != path.resolve():
        raise ValueError(f"blind audit {label} path mismatch")
    if reference["sha256"] != file_sha256(path):
        raise ValueError(f"blind audit {label} hash mismatch")


def _validate_candidate_binding(
    report: Mapping[str, Any], candidate_path: Path
) -> Path:
    manifest_path = candidate_path.with_suffix(".manifest.json")
    manifest = _read_object(manifest_path)
    inputs = report["inputs"]
    _validate_reference(inputs["candidate_artifact"], candidate_path, "candidate")
    _validate_reference(
        inputs["candidate_manifest"], manifest_path, "candidate manifest"
    )
    if (
        Path(str(manifest.get("candidate_path") or "")).resolve()
        != candidate_path.resolve()
    ):
        raise ValueError("candidate manifest path mismatch")
    if manifest.get("candidate_sha256") != file_sha256(candidate_path):
        raise ValueError("candidate manifest artifact hash mismatch")
    implementations = manifest.get("implementation")
    if (
        not isinstance(implementations, Mapping)
        or set(implementations) != IMPLEMENTATION_INPUTS
    ):
        raise ValueError("candidate implementation schema mismatch")
    for name in IMPLEMENTATION_INPUTS:
        reference = implementations[name]
        if reference != inputs[name] or not isinstance(reference, Mapping):
            raise ValueError(f"blind audit candidate {name} binding mismatch")
        _validate_reference(reference, Path(str(reference.get("path") or "")), name)
    if manifest.get("validations") != report["candidate_inventory_validations"]:
        raise ValueError("blind audit candidate validation binding mismatch")
    return manifest_path


def validate_blind_pass_receipt(
    receipt_path: Path, candidate_path: Path
) -> dict[str, Any]:
    """Return the reference a compiled mapping can pin after complete validation."""
    report = _read_object(receipt_path)
    _validate_report(report)
    manifest_path = _validate_candidate_binding(report, candidate_path)
    return {
        "path": str(receipt_path.resolve()),
        "sha256": file_sha256(receipt_path),
        "audit_version": AUDIT_VERSION,
        "score_version": SCORE_VERSION,
        "decision": "PASS",
        "candidate_sha256": file_sha256(candidate_path),
        "candidate_manifest_sha256": file_sha256(manifest_path),
    }


def validate_canonical_blind_pass() -> dict[str, Any]:
    """Validate the only durable blind receipt accepted for AMES V10 production."""
    reference = validate_blind_pass_receipt(BLIND_PASS_RECEIPT_PATH, CANDIDATE_PATH)
    if any(
        reference[name] != digest for name, digest in FROZEN_CANDIDATE_HASHES.items()
    ):
        raise ValueError("canonical blind audit candidate snapshot mismatch")
    return reference


__all__ = [
    "BLIND_PASS_RECEIPT_PATH",
    "CANDIDATE_PATH",
    "validate_blind_pass_receipt",
    "validate_canonical_blind_pass",
]
