"""Build a source-native flat-assay catalog for ClinTox Starling evidence.

ClinTox does not yet have the shared ``starling_normalized_v7`` table used by
BBB, Bioavailability, and Skin.  This adapter therefore freezes an explicit
assay-field mapping for each raw Starling source and publishes a record-to-assay
membership table.  It does not change the clinical_trial_failure_v1 gold label.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
from typing import Any

import pandas as pd

from tools.chembl_tool.common.json_utils import (
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.common.starling.assay_catalog import assay_id


CATALOG_VERSION = "clintox_source_native_flat_assay.v1"
TASK = "clintox"
TASK_DEFINITION = (
    "Predict the frozen clinical_trial_failure_v1 source class: association "
    "with toxicity-caused clinical trial or development failure versus an "
    "FDA-approved comparator without a frozen AACT positive."
)

SOURCE_SPECS = {
    "cellular_stress": {
        "assay_field": "stress_endpoint",
        "fallback_field": "evidence_basis",
        "endpoint_field": "mechanistic_effect",
        "context_fields": ["biological_model", "target_or_pathway", "evidence_basis"],
    },
    "general_cytotoxicity": {
        "assay_field": "assay_method",
        "fallback_field": "endpoint_type",
        "endpoint_field": "endpoint_type",
        "context_fields": ["cell_model", "exposure_time_h"],
    },
    "genotoxicity_carcinogenicity": {
        "assay_field": "assay_type",
        "fallback_field": "endpoint",
        "endpoint_field": "endpoint",
        "context_fields": ["biological_system", "study_context", "evidence_category"],
    },
    "nonclinical_in_vivo_toxicity": {
        "assay_field": "evidence_type",
        "fallback_field": "observed_effect",
        "endpoint_field": "observed_effect",
        "context_fields": ["animal_context", "exposure_context"],
    },
    "off_target_ddi_exposure": {
        "assay_field": "evidence_type",
        "fallback_field": "target_or_endpoint",
        "endpoint_field": "target_or_endpoint",
        "context_fields": ["assay_context", "target_identifier", "result_metric"],
    },
    "organ_specific_toxicity": {
        "assay_field": "evidence_context",
        "fallback_field": "toxicity_endpoint",
        "endpoint_field": "toxicity_endpoint",
        "context_fields": ["organ_system", "biological_system"],
    },
}


def _clean(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    if not text or text in {"none", "null", "nan", "not stated", "not specified", "unknown"}:
        return None
    return text


def _top_counts(series: pd.Series, *, limit: int = 12) -> list[dict[str, Any]]:
    cleaned = series.map(_clean).dropna()
    counts = cleaned.value_counts()
    return [
        {"value": str(value), "record_count": int(count)}
        for value, count in counts.head(limit).items()
    ]


def _read_source(source_root: Path, source_id: str, fields: list[str]) -> pd.DataFrame:
    path = source_root / source_id / "extractions.parquet"
    frame = pd.read_parquet(path, columns=fields).reset_index(names="source_row_number")
    return frame


def _build_raw_source(
    *,
    source_root: Path,
    source_id: str,
    bridge: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    spec = SOURCE_SPECS[source_id]
    fields = list(
        dict.fromkeys(
            [
                spec["assay_field"],
                spec["fallback_field"],
                spec["endpoint_field"],
                *spec["context_fields"],
            ]
        )
    )
    raw = _read_source(source_root, source_id, fields)
    source_bridge = bridge.loc[bridge["source_id"] == source_id].copy()
    joined = source_bridge.merge(raw, on="source_row_number", how="inner", validate="many_to_one")
    primary = joined[spec["assay_field"]].map(_clean)
    fallback = joined[spec["fallback_field"]].map(_clean)
    joined["_native_assay"] = primary.fillna(fallback).fillna("__missing_assay_definition__")
    joined["assay_context"] = source_id + "::" + joined["_native_assay"]
    joined["assay_id"] = joined["assay_context"].map(lambda value: assay_id(TASK, value))

    catalog: list[dict[str, Any]] = []
    for context, group in joined.groupby("assay_context", sort=True):
        descriptions: list[str] = []
        for field in spec["context_fields"]:
            descriptions.extend(item["value"] for item in _top_counts(group[field], limit=4))
        catalog.append(
            {
                "catalog_version": CATALOG_VERSION,
                "task": TASK,
                "task_definition": TASK_DEFINITION,
                "assay_id": str(group["assay_id"].iloc[0]),
                "assay_context": context,
                "assay_unit_source": f"source_native::{source_id}::{spec['assay_field']}",
                "source_id": source_id,
                "source_assay_field": spec["assay_field"],
                "record_count": int(len(group)),
                "unique_molecule_count": int(group["molecule_id"].nunique()),
                "endpoints": _top_counts(group[spec["endpoint_field"]]),
                "species_contexts": [],
                "measurement_kinds": [],
                "assay_descriptions": list(dict.fromkeys(descriptions))[:12],
            }
        )
    membership = joined[
        ["source_id", "source_row_number", "record_id", "molecule_id", "assay_id"]
    ].copy()
    return catalog, membership


def _direct_molecule_id(smiles: Any) -> str:
    value = "" if smiles is None or pd.isna(smiles) else str(smiles)
    return "STARLING_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16].upper()


def _build_direct_source(path: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    frame = pd.read_parquet(path).copy()
    frame["source_id"] = "clinical_trial_failure"
    frame["source_row_number"] = frame["source_row_number"].astype(int)
    category = frame["toxicity_category"].map(_clean).fillna("unspecified_toxicity")
    frame["assay_context"] = "clinical_trial_failure::" + category
    frame["assay_id"] = frame["assay_context"].map(lambda value: assay_id(TASK, value))
    frame["molecule_id"] = frame["SMILES"].map(_direct_molecule_id)
    frame["record_id"] = "clintox_direct:" + frame["source_row_number"].astype(str)
    catalog = []
    for context, group in frame.groupby("assay_context", sort=True):
        catalog.append(
            {
                "catalog_version": CATALOG_VERSION,
                "task": TASK,
                "task_definition": TASK_DEFINITION,
                "assay_id": str(group["assay_id"].iloc[0]),
                "assay_context": context,
                "assay_unit_source": "strict_direct_claim::toxicity_category",
                "source_id": "clinical_trial_failure",
                "source_assay_field": "toxicity_category",
                "record_count": int(len(group)),
                "unique_molecule_count": int(group["molecule_id"].nunique()),
                "endpoints": _top_counts(group["toxicity_outcome"]),
                "species_contexts": [{"value": "human clinical", "record_count": int(len(group))}],
                "measurement_kinds": [{"value": "qualitative", "record_count": int(len(group))}],
                "assay_descriptions": [
                    item["value"] for item in _top_counts(group["clinical_context"], limit=8)
                ],
                "direct_gate_status": "pending_manual_review",
            }
        )
    membership = frame[
        ["source_id", "source_row_number", "record_id", "molecule_id", "assay_id"]
    ].copy()
    return catalog, membership


def build_catalog(
    *,
    source_root: Path,
    bridge_path: Path,
    direct_path: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    bridge = pd.read_parquet(bridge_path)
    catalog: list[dict[str, Any]] = []
    memberships = []
    for source_id in SOURCE_SPECS:
        source_catalog, source_membership = _build_raw_source(
            source_root=source_root,
            source_id=source_id,
            bridge=bridge,
        )
        catalog.extend(source_catalog)
        memberships.append(source_membership)
    direct_catalog, direct_membership = _build_direct_source(direct_path)
    catalog.extend(direct_catalog)
    memberships.append(direct_membership)
    catalog.sort(key=lambda row: (row["assay_context"], row["assay_id"]))
    membership = pd.concat(memberships, ignore_index=True)
    if membership["record_id"].duplicated().any():
        raise ValueError("record_id is not unique in ClinTox assay membership")
    manifest = {
        "catalog_version": CATALOG_VERSION,
        "task": TASK,
        "task_definition": TASK_DEFINITION,
        "assay_unit_definition": "source-native primary assay field; documented per source",
        "source_specs": SOURCE_SPECS,
        "quality_gate": "none beyond the existing evidence-library record bridge and strict direct gate",
        "min_unique_molecules": 1,
        "n_assay_units": len(catalog),
        "n_membership_records": int(len(membership)),
        "n_unique_molecules": int(membership["molecule_id"].nunique()),
        "source_assay_counts": {
            source_id: sum(row["source_id"] == source_id for row in catalog)
            for source_id in [*SOURCE_SPECS, "clinical_trial_failure"]
        },
        "bridge_path": str(bridge_path.resolve()),
        "bridge_sha256": sha256_file(bridge_path),
        "direct_path": str(direct_path.resolve()),
        "direct_sha256": sha256_file(direct_path),
        "direct_gate_status": "pending_manual_review",
    }
    return catalog, membership, manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="data/starling_data/clintox/raw_v1")
    parser.add_argument(
        "--bridge",
        default=(
            "outputs/chembl_tool/tasks/clintox/evidence_library/"
            "starling_raw_v1/03_evidence_catalog/record_bridge.parquet"
        ),
    )
    parser.add_argument(
        "--direct",
        default="data/starling_data/clintox/canonical_retrieval_v1/strict_direct_claims.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/paper/starling_assay_relevance_all_v1/clintox",
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog, membership, manifest = build_catalog(
        source_root=Path(args.source_root),
        bridge_path=Path(args.bridge),
        direct_path=Path(args.direct),
    )
    catalog_path = output_dir / "assay_catalog.jsonl"
    membership_path = output_dir / "record_assay_membership.parquet"
    write_jsonl_atomic(catalog_path, catalog)
    temporary_membership = membership_path.with_suffix(".parquet.tmp")
    membership.to_parquet(temporary_membership, index=False)
    temporary_membership.replace(membership_path)
    manifest.update(
        {
            "catalog_path": str(catalog_path.resolve()),
            "catalog_sha256": sha256_file(catalog_path),
            "membership_path": str(membership_path.resolve()),
            "membership_sha256": sha256_file(membership_path),
        }
    )
    write_json_atomic(output_dir / "catalog_manifest.json", manifest)
    print(
        f"[clintox] wrote {len(catalog):,} assay units and "
        f"{len(membership):,} record memberships"
    )


if __name__ == "__main__":
    main()
