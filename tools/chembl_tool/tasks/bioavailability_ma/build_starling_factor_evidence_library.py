"""Build a Starling Observed/Fa/Fg/Fh index using shared source profiles."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem

from tools.chembl_tool.common.export import ensure_dir
from tools.chembl_tool.common.evidence_contract import (
    attach_minimal_evidence,
    numeric_only_evidence_row,
)
from data.processing.evidence_library.evidence_library import (
    StarlingSourceProfile,
    build_starling_parquet_evidence_rows,
    starling_molecule_id,
    write_jsonl,
)
from tools.chembl_tool.common.task_workflows.evidence_library import build_neighbor_index, fingerprint_metadata
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_evidence_library import (
    DEFAULT_SOURCE_PARQUET as DEFAULT_DIRECT_SOURCE_PARQUET,
    build_starling_evidence_rows as build_direct_f_rows,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.canonical_source import (
    DIRECT_REPORT_TYPES,
    nondirect_measurement_fields,
)
from tools.chembl_tool.tasks.bioavailability_ma.experiment_config import (
    NONDIRECT_ORAL_BIOAVAILABILITY_GROUP,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.starling_normalization_sources import (
    load_direct_hf_rows,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.oral_bioavailability import (
    ORAL_BIOAVAILABILITY_DATASET,
    ORAL_BIOAVAILABILITY_REVISION,
)


DEFAULT_STARLING_DATA_DIR = "data/raw/starling/bioavailability_ma"
DEFAULT_OUT_DIR = "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_factor"
EVIDENCE_FILENAME = "starling_factor_evidence.jsonl"
INDEX_FILENAME = "starling_factor_neighbor_index.pkl"
META_FILENAME = "starling_factor_neighbor_index.meta.json"
INDEX_VERSION = "bioavailability_ma_starling_factor_neighbor_index.v5"
EXPECTED_COMPLETE_HF_RAW_ROWS = 163_815
EXPECTED_DIRECT_HF_RAW_ROWS = 112_245
EXPECTED_NONDIRECT_HF_RAW_ROWS = 51_570
EXPECTED_DIRECT_HF_CLEAN_NUMERIC_ROWS = 84_672


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.monotonic()
    out_dir = ensure_dir(args.out_dir)

    direct_rows: list[dict[str, Any]] = []
    nondirect_rows: list[dict[str, Any]] = []
    direct_stats: dict[str, Any] = {}
    nondirect_stats: dict[str, Any] = {}
    hf_partition_stats: dict[str, Any] = {}
    if args.include_direct_hf:
        raw_hf_rows = load_direct_hf_rows(
            Path(args.direct_source_parquet), max_rows=args.max_direct_rows
        )
        direct_source_rows, nondirect_source_rows, hf_partition_stats = partition_hf_rows(
            raw_hf_rows,
            validate_complete=not args.max_direct_rows,
        )
        direct_rows, direct_stats = build_direct_f_rows(
            Path(args.direct_source_parquet),
            source_rows=direct_source_rows,
            allowed_report_types=DIRECT_REPORT_TYPES,
            include_qualitative=args.evidence_content == "full",
            min_value_percent=args.min_direct_value_percent, max_value_percent=args.max_direct_value_percent,
            max_record_examples=args.max_record_examples,
            max_source_rows=args.max_direct_rows,
        )
        _validate_prepared_direct_hf_counts(
            direct_stats,
            expected_raw_rows=(None if args.max_direct_rows else args.expected_direct_raw_rows),
            expected_clean_numeric_rows=(
                None
                if args.max_direct_rows
                else args.expected_direct_clean_numeric_rows
            ),
        )
        nondirect_rows, nondirect_stats = build_nondirect_hf_evidence_rows(
            nondirect_source_rows,
            include_qualitative=args.evidence_content == "full",
            max_record_examples=args.max_record_examples,
        )

    profiles = bioavailability_profiles(Path(args.starling_data_dir), max_rows=args.max_rows_per_source)
    if args.scope == "direct":
        profiles = profiles[:1]
    factor_rows, factor_stats = build_starling_parquet_evidence_rows(
        profiles,
        max_record_examples=args.max_record_examples,
        min_confidence=args.min_confidence,
    )
    evidence_rows = [*direct_rows, *nondirect_rows, *factor_rows]
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
            **hf_partition_stats,
            "direct_clean_numeric_rows": int(direct_stats.get("n_source_rows_kept") or 0),
            "direct_dropped_rows": int(direct_stats.get("n_dropped_rows_scanned") or 0),
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
        "n_nondirect_evidence_rows": len(nondirect_rows),
        "n_factor_evidence_rows": len(factor_rows),
        "n_evidence_rows": len(evidence_rows),
        "n_index_molecules": len(index["molecules"]),
        "groups": sorted(index["group_to_molecule_indices"]),
        "fingerprint": fingerprint_metadata(),
        "direct_source_stats": direct_stats,
        "nondirect_source_stats": nondirect_stats,
        "factor_source_stats": factor_stats,
        "elapsed_s": round(time.monotonic() - started, 3),
        "paths": {"evidence_jsonl": str(evidence_path), "index_pkl": str(index_path)},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


def _validate_prepared_direct_hf_counts(
    stats: dict[str, Any],
    *,
    expected_raw_rows: int | None,
    expected_clean_numeric_rows: int | None,
) -> None:
    clean_rows = int(stats.get("n_source_rows_kept") or 0)
    raw_rows = int(stats.get("n_source_rows") or 0)
    mismatches = []
    if expected_raw_rows is not None and raw_rows != expected_raw_rows:
        mismatches.append(f"raw rows: expected {expected_raw_rows:,}, found {raw_rows:,}")
    if (
        expected_clean_numeric_rows is not None
        and clean_rows != expected_clean_numeric_rows
    ):
        mismatches.append(
            f"clean numeric rows: expected {expected_clean_numeric_rows:,}, found {clean_rows:,}"
        )
    if mismatches:
        raise ValueError("Prepared HF artifact preflight failed (" + "; ".join(mismatches) + ")")


def partition_hf_rows(
    rows: list[dict[str, Any]], *, validate_complete: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Partition one immutable HF snapshot without rewriting source rows."""
    direct: list[dict[str, Any]] = []
    nondirect: list[dict[str, Any]] = []
    for row in rows:
        report_type = str(row.get("bioavailability_report_type") or "").strip().lower()
        (direct if report_type in DIRECT_REPORT_TYPES else nondirect).append(row)
    if validate_complete and (
        len(rows) != EXPECTED_COMPLETE_HF_RAW_ROWS
        or len(direct) != EXPECTED_DIRECT_HF_RAW_ROWS
        or len(nondirect) != EXPECTED_NONDIRECT_HF_RAW_ROWS
    ):
        raise ValueError(
            "HF evidence partition drift: "
            f"raw={len(rows):,}/{EXPECTED_COMPLETE_HF_RAW_ROWS:,}, "
            f"direct={len(direct):,}/{EXPECTED_DIRECT_HF_RAW_ROWS:,}, "
            f"nondirect={len(nondirect):,}/{EXPECTED_NONDIRECT_HF_RAW_ROWS:,}"
        )
    return direct, nondirect, {
        "complete_raw_rows": len(rows),
        "direct_partition_rows": len(direct),
        "nondirect_partition_rows": len(nondirect),
        "partition_reconciles": len(direct) + len(nondirect) == len(rows),
        "direct_report_types": sorted(DIRECT_REPORT_TYPES),
    }


def build_nondirect_hf_evidence_rows(
    source_rows: list[dict[str, Any]],
    *,
    include_qualitative: bool,
    max_record_examples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build molecule evidence without treating unitless relative values as F%."""
    numeric: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    qualitative: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_smiles = 0
    extraction_status: dict[str, int] = defaultdict(int)
    for row in source_rows:
        smiles = _canonical_smiles(row.get("smiles"))
        if not smiles:
            invalid_smiles += 1
            continue
        extracted = nondirect_measurement_fields(row.get("oral_bioavailability_value"))
        status = str(extracted["measurement_unit_extraction_status"])
        extraction_status[status] += 1
        example = _nondirect_example(row, extracted)
        if status == "explicit_atomic_scalar_unit":
            numeric[(smiles, str(extracted["value_units"]))].append(example)
        elif include_qualitative:
            qualitative[smiles].append(example)

    evidence_rows = [
        _summarize_nondirect_numeric(
            smiles,
            unit,
            examples,
            max_record_examples=max_record_examples,
        )
        for (smiles, unit), examples in sorted(numeric.items())
    ]
    if include_qualitative:
        evidence_rows.extend(
            _summarize_nondirect_qualitative(
                smiles,
                examples,
                max_record_examples=max_record_examples,
            )
            for smiles, examples in sorted(qualitative.items())
        )
    return evidence_rows, {
        "n_source_rows": len(source_rows),
        "n_invalid_smiles": invalid_smiles,
        "measurement_unit_extraction_status": dict(sorted(extraction_status.items())),
        "n_numeric_evidence_rows": len(numeric),
        "n_qualitative_evidence_rows": len(qualitative) if include_qualitative else 0,
        "n_evidence_rows": len(evidence_rows),
    }


def _nondirect_example(
    row: dict[str, Any], extracted: dict[str, Any]
) -> dict[str, Any]:
    return {
        "source_index": row.get("source_index", ""),
        "molecule_name": str(row.get("molecule_name") or "").strip(),
        # Preserve the complete claim.  Parsed fields are annotations, not a
        # replacement for comparator/direction wording in the source text.
        "reported_value": str(row.get("oral_bioavailability_value") or "").strip(),
        "parsed_numeric_value": extracted.get("numeric_value"),
        "reported_units": str(extracted.get("value_units") or ""),
        "measurement_unit_extraction_status": str(
            extracted.get("measurement_unit_extraction_status") or ""
        ),
        "bioavailability_report_type": str(
            row.get("bioavailability_report_type") or ""
        ).strip(),
        "species_or_population": str(row.get("species_or_population") or "").strip(),
        "dose": str(row.get("dose") or "").strip(),
        "oral_exposure_mode": str(row.get("oral_exposure_mode") or "").strip(),
        "qualifying_conditions": str(row.get("qualifying_conditions") or "").strip(),
        "comparator": str(row.get("comparator") or "").strip(),
        "extra_details": str(row.get("extra_details") or "").strip(),
        "pmid": str(row.get("pmid") or "").strip(),
        "support_text": str(row.get("support_text") or "").strip(),
    }


def _summarize_nondirect_numeric(
    smiles: str,
    unit: str,
    examples: list[dict[str, Any]],
    *,
    max_record_examples: int,
) -> dict[str, Any]:
    ordered = sorted(examples, key=lambda item: int(item.get("source_index") or 0))
    selected = _evenly_spaced(ordered, max_record_examples)
    return _nondirect_evidence_row(
        smiles,
        standard_value="",
        standard_units="",
        examples=selected,
        source_record_count=len(ordered),
        activity_comment=(
            f"Non-direct oral bioavailability context over {len(ordered)} records "
            f"with parsed {unit} values; individual claims are retained without "
            "a synthetic aggregate measurement."
        ),
        uncertainty=["nondirect_measurements_not_aggregated"],
        numeric_examples=True,
    )


def _summarize_nondirect_qualitative(
    smiles: str,
    examples: list[dict[str, Any]],
    *,
    max_record_examples: int,
) -> dict[str, Any]:
    selected = _evenly_spaced(
        sorted(examples, key=lambda item: int(item.get("source_index") or 0)),
        max_record_examples,
    )
    return _nondirect_evidence_row(
        smiles,
        standard_value="",
        standard_units="",
        examples=selected,
        source_record_count=len(examples),
        activity_comment=(
            f"Non-direct oral bioavailability context over {len(examples)} records; "
            "no explicit atomic scalar/unit pair was extracted."
        ),
        uncertainty=["nondirect_qualitative_or_unit_unresolved"],
        numeric_examples=False,
    )


def _nondirect_evidence_row(
    smiles: str,
    *,
    standard_value: float | str,
    standard_units: str,
    examples: list[dict[str, Any]],
    source_record_count: int,
    activity_comment: str,
    uncertainty: list[str],
    numeric_examples: bool,
) -> dict[str, Any]:
    molecule_id = starling_molecule_id(smiles)
    row = {
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": smiles,
        "assay_chembl_id": "STARLING_NONDIRECT_ORAL_BIOAVAILABILITY",
        "assay_tier": "Observed",
        "endpoint_group": "nondirect_oral_bioavailability",
        "group_id": NONDIRECT_ORAL_BIOAVAILABILITY_GROUP,
        "standard_type": "Relative or apparent oral bioavailability",
        "standard_relation": "",
        "standard_value": standard_value,
        "standard_units": standard_units,
        "pchembl_value": "",
        "activity_comment": activity_comment,
        "data_validity_comment": "",
        "assay_description": json.dumps(examples, ensure_ascii=False, default=str),
        "target_pref_name": "relative or apparent oral bioavailability",
        "target_genes": "",
        "organism": "",
        "confidence_score": "",
        "relationship_type": "",
        "evidence_source": ORAL_BIOAVAILABILITY_DATASET,
        "evidence_role": "surrogate_proxy",
        "evidence_scope": {
            "report_types": sorted(
                {str(item.get("bioavailability_report_type") or "") for item in examples}
            )
        },
        "transferability": "not_assessed",
        "uncertainty": uncertainty,
        "source_molecule_names": sorted(
            {str(item.get("molecule_name") or "") for item in examples if item.get("molecule_name")}
        )[:10],
        "source_record_count": source_record_count,
        "source_numeric_record_count": source_record_count if numeric_examples else 0,
        "source_qualitative_record_count": 0 if numeric_examples else source_record_count,
        "source_record_examples": examples if numeric_examples else [],
        "source_qualitative_examples": [] if numeric_examples else examples,
    }
    return attach_minimal_evidence(row)


def _canonical_smiles(value: Any) -> str:
    text = str(value or "").strip()
    molecule = Chem.MolFromSmiles(text) if text else None
    return (
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        if molecule is not None
        else ""
    )


def _evenly_spaced(values: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(values) <= limit:
        return values if limit != 0 else []
    if limit == 1:
        return [values[len(values) // 2]]
    indices = [round(i * (len(values) - 1) / (limit - 1)) for i in range(limit)]
    return [values[index] for index in dict.fromkeys(indices)]


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
