"""Strict, source-reconstructed ClinTox clinical-trial-failure gold contract.

The positive source is the original AACT-derived list of drugs from trials
that failed for toxicity.  The negative comparator is the original
SWEETLEAD/FDA-approved list.  General adverse events, organ injury, preclinical
toxicity, and mechanistic liabilities never create labels in this contract.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from tools.chembl_tool.common.molecule_identity import (
    IDENTITY_NORMALIZER_VERSION,
    bemis_murcko_scaffold,
    normalize_molecule_identity,
)


CONTRACT_VERSION = "clintox_clinical_trial_failure.v1"
LINEAGE = "clinical_trial_failure_v1"
TASK_NAME = "ClinTox"
SOURCE_ROOT = Path("data/starling_data/clintox/tdc_clintox_v1")
CANONICAL_ROOT = Path(
    "data/starling_data/clintox/canonical_clinical_trial_failure_v1"
)
AACT_SOURCE_PATH = SOURCE_ROOT / "aacttox.csv.gz"
COMPARATOR_SOURCE_PATH = SOURCE_ROOT / "sweetfda_approved_processed.csv.gz"
REFERENCE_SOURCE_PATH = SOURCE_ROOT / "clintox.csv.gz"
STARLING_BASE_PATH = Path(
    "data/starling_data/clintox/clintox_base_v1/extractions.parquet"
)

SOURCE_SHA256 = {
    "aacttox.csv.gz": "c132ee41f78b44f72c243643fc9e6676feaa6ff44e0178daaa0570525ec9489f",
    "sweetfda_approved_processed.csv.gz": "d79a9f6e7911241086ebcff50dcb4c86b547fb8005d1e19ce1f4617ed0cfafcb",
    "clintox.csv.gz": "2ee7050a830fe9ae7a8218e6577fa6f54284947034ff904b6d8204e993a297cc",
}
STARLING_BASE_SHA256 = (
    "472b5239e38c59f91d1a60eece7caf7c37a83f3b00b785f9ead59a460b83c285"
)


def build_canonical_frames(
    aacttox_frame: pd.DataFrame,
    comparator_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Normalize immutable sources and return parent labels plus full audits."""
    _validate_binary_source(aacttox_frame, "CT_TOX", expected=1, name="aacttox")
    _validate_binary_source(
        comparator_frame, "FDA_APPROVED", expected=1, name="sweetfda"
    )
    _validate_reference(reference_frame)

    audit_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identity_cache: dict[str, dict[str, Any]] = {}
    source_specs = (
        (
            "aacttox",
            "clinical_trial_failure_positive",
            1,
            aacttox_frame,
        ),
        (
            "sweetfda",
            "fda_approved_comparator",
            0,
            comparator_frame,
        ),
    )
    for source_name, source_role, declared_y, frame in source_specs:
        for source_index, row in enumerate(frame.to_dict(orient="records")):
            smiles = _text(row.get("smiles"))
            identity = _normalized_identity(smiles, identity_cache)
            eligible = identity["status"] == "ok" and bool(identity["parent_smiles"])
            record = {
                "source_name": source_name,
                "source_role": source_role,
                "source_row_number": source_index,
                "source_record_id": f"{source_name}:{source_index:06d}",
                "source_smiles": smiles,
                "declared_Y": declared_y,
                "structure_status": identity["status"],
                "canonical_smiles": identity["canonical_smiles"],
                "parent_smiles": identity["parent_smiles"],
                "parent_inchi_key": identity["parent_inchi_key"],
                "molecule_identity_key": identity["identity_key"],
                "label_eligible": eligible,
                "rejection_reason": "" if eligible else "invalid_or_unresolved_smiles",
            }
            audit_rows.append(record)
            if eligible:
                grouped[identity["identity_key"]].append(record)

    parent_rows: list[dict[str, Any]] = []
    for identity_key, records in sorted(grouped.items()):
        positives = [row for row in records if int(row["declared_Y"]) == 1]
        comparators = [row for row in records if int(row["declared_Y"]) == 0]
        representative = records[0]
        label = int(bool(positives))
        parent_rows.append(
            {
                "drug": representative["parent_smiles"],
                "Y": label,
                "molecule_identity_key": identity_key,
                "parent_inchi_key": representative["parent_inchi_key"],
                "bemis_murcko_scaffold": bemis_murcko_scaffold(
                    representative["parent_smiles"]
                ),
                "source_record_count": len(records),
                "positive_source_record_count": len(positives),
                "comparator_source_record_count": len(comparators),
                "source_role_overlap": bool(positives and comparators),
                "label_decision": (
                    "positive_aacttox_event_overrides_comparator"
                    if positives and comparators
                    else (
                        "positive_aacttox_event"
                        if positives
                        else "negative_fda_approved_comparator"
                    )
                ),
                "source_record_ids": sorted(
                    str(row["source_record_id"]) for row in records
                ),
                "source_smiles_examples": _unique_limited(
                    (str(row["source_smiles"]) for row in records), 20
                ),
            }
        )

    reference_groups, reference_invalid = _reference_parent_labels(reference_frame)
    source_by_key = {
        str(row["molecule_identity_key"]): row for row in parent_rows
    }
    reconciliation_rows: list[dict[str, Any]] = []
    for identity_key in sorted(set(source_by_key) | set(reference_groups)):
        source_row = source_by_key.get(identity_key)
        reference = reference_groups.get(identity_key)
        source_y = None if source_row is None else int(source_row["Y"])
        reference_y = None if reference is None else int(reference["Y"])
        if source_row is None:
            status = "reference_only"
            drug = reference["drug"]
        elif reference is None:
            status = "source_reconstruction_only"
            drug = source_row["drug"]
        elif source_y == reference_y:
            status = "label_match"
            drug = source_row["drug"]
        else:
            status = "label_mismatch"
            drug = source_row["drug"]
        reconciliation_rows.append(
            {
                "molecule_identity_key": identity_key,
                "drug": drug,
                "source_reconstructed_Y": source_y,
                "reference_parent_Y": reference_y,
                "reference_raw_label_conflict": bool(
                    reference and reference["raw_label_conflict"]
                ),
                "reference_source_record_count": (
                    0 if reference is None else reference["source_record_count"]
                ),
                "reconciliation_status": status,
            }
        )

    source_audit = pd.DataFrame(audit_rows)
    parents = pd.DataFrame(parent_rows)
    reconciliation = pd.DataFrame(reconciliation_rows)
    label_counts = Counter(int(value) for value in parents["Y"])
    source_status_counts = Counter(
        "accepted" if row["label_eligible"] else row["rejection_reason"]
        for row in audit_rows
    )
    reconciliation_counts = Counter(reconciliation["reconciliation_status"])
    common = reconciliation[
        reconciliation["reconciliation_status"].isin(
            ["label_match", "label_mismatch"]
        )
    ]
    stats = {
        "n_aacttox_source_rows": len(aacttox_frame),
        "n_comparator_source_rows": len(comparator_frame),
        "n_source_identity_audit_rows": len(source_audit),
        "source_identity_status_counts": dict(sorted(source_status_counts.items())),
        "n_binary_parent_molecules": len(parents),
        "parent_label_counts": {
            "0": label_counts.get(0, 0),
            "1": label_counts.get(1, 0),
        },
        "n_parent_source_role_overlaps": int(parents["source_role_overlap"].sum()),
        "n_positive_parent_duplicates_collapsed": int(
            parents["positive_source_record_count"].sum()
            - (parents["positive_source_record_count"] > 0).sum()
        ),
        "n_comparator_parent_duplicates_collapsed": int(
            parents["comparator_source_record_count"].sum()
            - (parents["comparator_source_record_count"] > 0).sum()
        ),
        "n_reference_rows": len(reference_frame),
        "n_reference_invalid_structure_rows": reference_invalid,
        "n_reference_parent_molecules": len(reference_groups),
        "n_reference_parent_raw_label_conflicts": sum(
            bool(row["raw_label_conflict"]) for row in reference_groups.values()
        ),
        "reference_reconciliation_counts": dict(
            sorted(reconciliation_counts.items())
        ),
        "n_common_reference_parents": len(common),
        "n_common_reference_label_matches": int(
            (common["reconciliation_status"] == "label_match").sum()
        ),
        "n_common_reference_label_mismatches": int(
            (common["reconciliation_status"] == "label_mismatch").sum()
        ),
        "identity_normalizer_version": IDENTITY_NORMALIZER_VERSION,
    }
    if stats["n_common_reference_label_mismatches"]:
        raise AssertionError("source reconstruction disagrees with the joined reference")
    if len(source_audit) != len(aacttox_frame) + len(comparator_frame):
        raise AssertionError("source identity audit does not reconcile to inputs")
    return {
        "source_identity_audit": source_audit,
        "parent_labels": parents,
        "reference_reconciliation": reconciliation,
        "stats": stats,
    }


def build_starling_coverage(
    parent_labels: pd.DataFrame,
    starling_smiles: Iterable[Any],
) -> dict[str, Any]:
    """Audit evidence coverage without allowing Starling rows to create labels."""
    raw_counts = Counter(_text(value) for value in starling_smiles if _text(value))
    parent_row_counts: Counter[str] = Counter()
    invalid_unique = 0
    for smiles, count in raw_counts.items():
        identity = normalize_molecule_identity(smiles)
        if identity.status != "ok" or not identity.parent_smiles:
            invalid_unique += 1
            continue
        identity_key = identity.parent_inchi_key or identity.parent_smiles
        parent_row_counts[identity_key] += count

    rows = []
    for parent in parent_labels.to_dict(orient="records"):
        identity_key = str(parent["molecule_identity_key"])
        count = int(parent_row_counts.get(identity_key, 0))
        rows.append(
            {
                "molecule_identity_key": identity_key,
                "drug": parent["drug"],
                "Y": int(parent["Y"]),
                "has_clintox_base_evidence": count > 0,
                "clintox_base_source_row_count": count,
            }
        )
    coverage = pd.DataFrame(rows)
    by_label = {}
    for label in (0, 1):
        subset = coverage[coverage["Y"] == label]
        by_label[str(label)] = {
            "n_parents": len(subset),
            "n_with_clintox_base_evidence": int(
                subset["has_clintox_base_evidence"].sum()
            ),
        }
    return {
        "coverage": coverage,
        "stats": {
            "n_unique_starling_source_smiles": len(raw_counts),
            "n_invalid_unique_starling_source_smiles": invalid_unique,
            "n_starling_parent_identities": len(parent_row_counts),
            "n_benchmark_parents_with_clintox_base_evidence": int(
                coverage["has_clintox_base_evidence"].sum()
            ),
            "coverage_by_label": by_label,
            "starling_rows_used_to_create_labels": 0,
        },
    }


def _validate_binary_source(
    frame: pd.DataFrame,
    label_column: str,
    *,
    expected: int,
    name: str,
) -> None:
    missing = sorted({"smiles", label_column} - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    values = {int(value) for value in frame[label_column].dropna()}
    if values != {expected} or frame[label_column].isna().any():
        raise ValueError(
            f"{name}.{label_column} must contain only {expected}; found {sorted(values)}"
        )


def _validate_reference(frame: pd.DataFrame) -> None:
    missing = sorted({"smiles", "FDA_APPROVED", "CT_TOX"} - set(frame.columns))
    if missing:
        raise ValueError(f"joined ClinTox reference is missing columns: {missing}")
    for column in ("FDA_APPROVED", "CT_TOX"):
        values = {int(value) for value in frame[column].dropna()}
        if not values <= {0, 1} or frame[column].isna().any():
            raise ValueError(f"reference {column} is not complete binary data")


def _normalized_identity(
    smiles: str, cache: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if smiles not in cache:
        identity = normalize_molecule_identity(smiles)
        parent_smiles = identity.parent_smiles or ""
        cache[smiles] = {
            "status": identity.status,
            "canonical_smiles": identity.canonical_smiles or "",
            "parent_smiles": parent_smiles,
            "parent_inchi_key": identity.parent_inchi_key or "",
            "identity_key": identity.parent_inchi_key or parent_smiles,
        }
    return cache[smiles]


def _reference_parent_labels(
    frame: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], int]:
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    identity_cache: dict[str, dict[str, Any]] = {}
    invalid = 0
    for row in frame.to_dict(orient="records"):
        identity = _normalized_identity(_text(row.get("smiles")), identity_cache)
        if identity["status"] != "ok" or not identity["identity_key"]:
            invalid += 1
            continue
        grouped[identity["identity_key"]].append(
            (int(row["CT_TOX"]), identity["parent_smiles"])
        )
    result = {}
    for identity_key, values in grouped.items():
        labels = {value[0] for value in values}
        result[identity_key] = {
            "Y": max(labels),
            "drug": values[0][1],
            "raw_label_conflict": len(labels) > 1,
            "source_record_count": len(values),
        }
    return result, invalid


def _unique_limited(values: Iterable[str], limit: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
            if len(output) >= limit:
                break
    return output


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "n/a", "na"} else text


__all__ = [
    "AACT_SOURCE_PATH",
    "CANONICAL_ROOT",
    "COMPARATOR_SOURCE_PATH",
    "CONTRACT_VERSION",
    "LINEAGE",
    "REFERENCE_SOURCE_PATH",
    "SOURCE_ROOT",
    "SOURCE_SHA256",
    "STARLING_BASE_PATH",
    "STARLING_BASE_SHA256",
    "TASK_NAME",
    "build_canonical_frames",
    "build_starling_coverage",
]
