"""Build a Starling Observed/Fa/Fg/Fh index using shared source profiles."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.export import ensure_dir
from tools.chembl_tool.common.evidence_contract import numeric_only_evidence_row
from tools.chembl_tool.common.starling import (
    StarlingSourceProfile,
    build_starling_parquet_evidence_rows,
    write_jsonl,
)
from tools.chembl_tool.common.task_workflows.evidence_library import build_neighbor_index, fingerprint_metadata
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_evidence_library import (
    DEFAULT_SOURCE_PARQUET as DEFAULT_DIRECT_SOURCE_PARQUET,
    build_starling_evidence_rows as build_direct_f_rows,
)
from tools.chembl_tool.common.starling.oral_bioavailability import (
    ORAL_BIOAVAILABILITY_DATASET,
    ORAL_BIOAVAILABILITY_REVISION,
)


DEFAULT_STARLING_DATA_DIR = "data/starling_data/bioavailability_ma"
DEFAULT_OUT_DIR = "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_factor"
EVIDENCE_FILENAME = "starling_factor_evidence.jsonl"
INDEX_FILENAME = "starling_factor_neighbor_index.pkl"
META_FILENAME = "starling_factor_neighbor_index.meta.json"
INDEX_VERSION = "bioavailability_ma_starling_factor_neighbor_index.v2"
EXPECTED_DIRECT_HF_RAW_ROWS = 163_815
EXPECTED_DIRECT_HF_CLEAN_NUMERIC_ROWS = 80_808


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.monotonic()
    out_dir = ensure_dir(args.out_dir)

    direct_rows: list[dict[str, Any]] = []
    direct_stats: dict[str, Any] = {}
    if args.include_direct_hf:
        direct_rows, direct_stats = build_direct_f_rows(
            Path(args.direct_source_parquet),
            include_qualitative=args.evidence_content == "full",
            min_value_percent=args.min_direct_value_percent, max_value_percent=args.max_direct_value_percent,
            max_record_examples=args.max_record_examples,
            max_source_rows=args.max_direct_rows,
        )
        _validate_prepared_direct_hf_counts(
            direct_stats,
            expected_raw_rows=args.expected_direct_raw_rows,
            expected_clean_numeric_rows=args.expected_direct_clean_numeric_rows,
        )

    profiles = bioavailability_profiles(Path(args.starling_data_dir), max_rows=args.max_rows_per_source)
    if args.scope == "direct":
        profiles = profiles[:1]
    factor_rows, factor_stats = build_starling_parquet_evidence_rows(
        profiles,
        max_record_examples=args.max_record_examples,
        min_confidence=args.min_confidence,
    )
    evidence_rows = [*direct_rows, *factor_rows]
    if args.evidence_content == "numeric_only":
        evidence_rows = [numeric for row in evidence_rows if (numeric := numeric_only_evidence_row(row)) is not None]
    index_version = f"{INDEX_VERSION}.{args.scope}.{args.evidence_content}"
    index = build_neighbor_index(
        evidence_rows,
        index_version=index_version,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    index["source"] = {
        "type": "starling_profile_index",
        "dataset": "starling-labs/Bioavailability_Ma",
        "groups": sorted(index.get("group_to_molecule_indices", {})),
        "exact_query_exclusion": True,
    }

    evidence_path = out_dir / EVIDENCE_FILENAME
    index_path = out_dir / INDEX_FILENAME
    meta_path = out_dir / META_FILENAME
    write_jsonl(evidence_path, evidence_rows)
    with index_path.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "index_version": index_version,
        "starling_data_dir": args.starling_data_dir,
        "include_direct_hf": args.include_direct_hf,
        "direct_hf_provenance": {
            "source_mode": "complete_parquet",
            "dataset": ORAL_BIOAVAILABILITY_DATASET,
            "revision": ORAL_BIOAVAILABILITY_REVISION,
            "records_parquet": args.direct_source_parquet,
        } if args.include_direct_hf else None,
        "prepared_hf_row_counts": {
            "raw_rows": int(direct_stats.get("n_source_rows") or 0),
            "clean_numeric_rows": int(direct_stats.get("n_source_rows_kept") or 0),
            "dropped_rows": int(direct_stats.get("n_dropped_rows_scanned") or 0),
        } if args.include_direct_hf else None,
        "underlying_sources": [
            *([ORAL_BIOAVAILABILITY_DATASET] if args.include_direct_hf else []),
            "starling-labs/bioavailability_ma/Oral_AUC-Cmax-Exposure",
            "starling-labs/bioavailability_ma/Fa",
            "starling-labs/bioavailability_ma/Fg",
            "starling-labs/bioavailability_ma/Fh",
        ],
        "scope": args.scope,
        "evidence_content": args.evidence_content,
        "n_direct_evidence_rows": len(direct_rows),
        "n_factor_evidence_rows": len(factor_rows),
        "n_evidence_rows": len(evidence_rows),
        "n_index_molecules": len(index["molecules"]),
        "groups": sorted(index["group_to_molecule_indices"]),
        "fingerprint": fingerprint_metadata(),
        "direct_source_stats": direct_stats,
        "factor_source_stats": factor_stats,
        "elapsed_s": round(time.monotonic() - started, 3),
        "paths": {"evidence_jsonl": str(evidence_path), "index_pkl": str(index_path)},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


def _validate_prepared_direct_hf_counts(
    stats: dict[str, Any], *, expected_raw_rows: int, expected_clean_numeric_rows: int
) -> None:
    clean_rows = int(stats.get("n_source_rows_kept") or 0)
    raw_rows = int(stats.get("n_source_rows") or 0)
    mismatches = []
    if expected_raw_rows and raw_rows != expected_raw_rows:
        mismatches.append(f"raw rows: expected {expected_raw_rows:,}, found {raw_rows:,}")
    if expected_clean_numeric_rows and clean_rows != expected_clean_numeric_rows:
        mismatches.append(
            f"clean numeric rows: expected {expected_clean_numeric_rows:,}, found {clean_rows:,}"
        )
    if mismatches:
        raise ValueError("Prepared HF artifact preflight failed (" + "; ".join(mismatches) + ")")


def bioavailability_profiles(data_dir: Path, *, max_rows: int = 0) -> list[StarlingSourceProfile]:
    """Task configuration only: parquet columns, evidence groups, and roles."""
    return [
        StarlingSourceProfile(
            source_id="observed_direct_bioavailability",
            path=str(data_dir / "Oral_AUC-Cmax_Exposure" / "extractions.parquet"),
            group_id="Observed.direct_oral_bioavailability",
            assay_tier="Observed",
            endpoint_group="direct_oral_bioavailability",
            evidence_source="starling-labs/bioavailability_ma/Oral_AUC-Cmax-Exposure",
            endpoint_field="exposure_measure",
            value_field="parameter_value",
            unit_field="parameter_units",
            context_fields=("statistic_type", "oral_dose", "study_context", "comparator_exposure"),
            scope_fields=("study_context",),
            target_pref_name="oral bioavailability",
            evidence_role="direct_outcome",
            standard_type_prefix="oral bioavailability",
            include_endpoint_values=("bioavailability",),
            max_rows=max_rows,
        ),
        StarlingSourceProfile(
            source_id="observed_oral_auc_cmax_exposure",
            path=str(data_dir / "Oral_AUC-Cmax_Exposure" / "extractions.parquet"),
            group_id="Observed.oral_auc_cmax_exposure",
            assay_tier="Observed",
            endpoint_group="oral_auc_cmax_exposure",
            evidence_source="starling-labs/bioavailability_ma/Oral_AUC-Cmax-Exposure",
            endpoint_field="exposure_measure",
            value_field="parameter_value",
            unit_field="parameter_units",
            context_fields=("statistic_type", "oral_dose", "study_context", "comparator_exposure"),
            scope_fields=("study_context",),
            target_pref_name="oral systemic exposure",
            evidence_role="surrogate_proxy",
            standard_type_prefix="oral exposure",
            exclude_endpoint_values=("bioavailability",),
            max_rows=max_rows,
        ),
        StarlingSourceProfile(
            source_id="fa_absorption_solubility_permeability",
            path=str(data_dir / "Fa" / "extractions.parquet"),
            group_id="Fa.absorption_solubility_permeability",
            assay_tier="Fa",
            endpoint_group="absorption_solubility_permeability",
            evidence_source="starling-labs/bioavailability_ma/Fa",
            endpoint_field="endpoint_category",
            value_field="reported_value",
            unit_field="reported_units",
            context_fields=("assay_system", "condition_medium", "biological_context", "formulation_or_solid_form"),
            scope_fields=("assay_system", "biological_context", "formulation_or_solid_form"),
            target_pref_name="Fa absorption, solubility, permeability, dissolution, and GI stability",
            evidence_role="mechanistic_factor",
            standard_type_prefix="Fa factor",
            max_rows=max_rows,
        ),
        StarlingSourceProfile(
            source_id="fg_gut_wall_efflux_intestinal_metabolism",
            path=str(data_dir / "Fg" / "extractions.parquet"),
            group_id="Fg.gut_wall_efflux_intestinal_metabolism",
            assay_tier="Fg",
            endpoint_group="gut_wall_efflux_intestinal_metabolism",
            evidence_source="starling-labs/bioavailability_ma/Fg",
            endpoint_field="gut_wall_process",
            value_field="measured_value",
            context_fields=("transporter_or_enzyme", "substrate_status", "assay_system", "intestinal_site"),
            scope_fields=("assay_system", "intestinal_site"),
            target_pref_name="Fg gut-wall escape, efflux, and intestinal metabolism",
            evidence_role="mechanistic_factor",
            standard_type_prefix="Fg factor",
            max_rows=max_rows,
        ),
        StarlingSourceProfile(
            source_id="fh_hepatic_clearance_metabolic_stability",
            path=str(data_dir / "Fh" / "extractions.parquet"),
            group_id="Fh.hepatic_clearance_metabolic_stability",
            assay_tier="Fh",
            endpoint_group="hepatic_clearance_metabolic_stability",
            evidence_source="starling-labs/bioavailability_ma/Fh",
            endpoint_field="metric_type",
            value_field="reported_value",
            unit_field="reported_units",
            context_fields=("assay_system", "species", "molecular_form", "enzyme_or_pathway"),
            scope_fields=("assay_system", "species", "molecular_form"),
            target_pref_name="Fh hepatic clearance and metabolic stability",
            evidence_role="mechanistic_factor",
            standard_type_prefix="Fh factor",
            max_rows=max_rows,
        ),
    ]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starling-data-dir", default=DEFAULT_STARLING_DATA_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--include-direct-hf", dest="include_direct_hf", action="store_true", default=True)
    parser.add_argument("--no-include-direct-hf", dest="include_direct_hf", action="store_false")
    parser.add_argument("--scope", choices=["direct", "full"], default="full")
    parser.add_argument("--evidence-content", choices=["numeric_only", "full"], default="full")
    parser.add_argument("--direct-source-parquet", default=DEFAULT_DIRECT_SOURCE_PARQUET)
    parser.add_argument("--max-direct-rows", type=int, default=0)
    parser.add_argument("--min-direct-value-percent", type=float, default=0.0)
    parser.add_argument("--max-direct-value-percent", type=float, default=100.0)
    parser.add_argument("--expected-direct-raw-rows", type=int, default=EXPECTED_DIRECT_HF_RAW_ROWS)
    parser.add_argument(
        "--expected-direct-clean-numeric-rows",
        type=int,
        default=EXPECTED_DIRECT_HF_CLEAN_NUMERIC_ROWS,
    )
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-record-examples", type=int, default=6)
    parser.add_argument("--max-rows-per-source", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
