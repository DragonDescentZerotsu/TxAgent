"""Build the versioned TxAgent assay-transfer record catalog from a frozen HF dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.task_workflows.evidence_library import standardize_smiles
from tools.chembl_tool.tasks.bioavailability_ma.reranking.assay_transfer_rerank import (
    CATALOG_SCHEMA_VERSION,
    TEMPLATE_BY_CONCEPT,
    template_bundle_hash,
)


DEFAULT_DATASET = "jiosephlee/assay-transfer-intern"
DEFAULT_DATASET_REVISION = "51d54cdf09630d3dba3d20284e5c4cfa59d97221"
DEFAULT_SPLITS = ("train", "validation", "test")
DEFAULT_STARLING_DATA_DIR = "data/starling_data/bioavailability_ma"
DEFAULT_SOURCE_CATALOG = (
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
    "assay_transfer_rerank/source_catalog.jsonl"
)
DEFAULT_CANDIDATE_MANIFEST = (
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
    "assay_transfer_rerank/validation_candidates.jsonl"
)


class _CandidateCollector:
    name = "assay_transfer_candidate_collector"

    def __init__(self, query_index: int, output: list[dict[str, Any]]):
        self.query_index = query_index
        self.output = output

    def provenance(self) -> dict[str, Any]:
        return {"name": self.name}

    def rerank(
        self, *, query_smiles: str, group_id: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        self.output.append(
            {
                "query_index": self.query_index,
                "query_smiles": query_smiles,
                "group_id": group_id,
                "candidates": candidates,
            }
        )
        return candidates

    def rerank_records(
        self, *, query_smiles: str, group_id: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Collect the full pre-rerank molecule pool for record-level retrieval."""
        return self.rerank(
            query_smiles=query_smiles,
            group_id=group_id,
            candidates=candidates,
        )


def build_candidate_scoped_catalog(
    *,
    records: list[dict[str, Any]],
    indices: list[int],
    smiles_field: str,
    index: dict[str, Any],
    output: Path,
    manifest_output: Path,
    experiment_mode: str,
    top_k_per_group: int,
    min_similarity: float,
    neighbor_identity_policy: str,
    initial_morgan_filter: int,
    index_path: str = "",
    condition_id: str = "validation__candidate_scoped",
) -> dict[str, Any]:
    """Freeze exactly the retained validation candidates and their attached source records."""
    from tools.chembl_tool.common.experiment_retrieval import retrieve_experiment_view
    from tools.chembl_tool.tasks.bioavailability_ma.experiment_config import STARLING

    selections: list[dict[str, Any]] = []
    for query_index in indices:
        query_smiles = str(records[query_index].get(smiles_field) or "")
        collector = _CandidateCollector(query_index, selections)
        retrieve_experiment_view(
            query_smiles,
            index,
            mode=experiment_mode,
            config=STARLING,
            top_k_per_group=top_k_per_group,
            min_similarity=min_similarity,
            neighbor_identity_policy=neighbor_identity_policy,
            reranker=collector,
            assay_transfer_initial_morgan_filter=initial_morgan_filter,
        )

    by_id: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []
    for selection in selections:
        frozen_candidates = []
        for candidate in selection["candidates"]:
            record_ids = []
            for record in _records_from_index_candidate(selection["group_id"], candidate):
                record_id = str(record["record_id"])
                by_id.setdefault(record_id, record)
                record_ids.append(record_id)
            frozen_candidates.append(
                {
                    "molecule_id": candidate["molecule_chembl_id"],
                    "canonical_smiles": candidate["canonical_smiles"],
                    "similarity": candidate["similarity"],
                    "structural_rank": candidate["structural_rank"],
                    "record_ids": sorted(set(record_ids)),
                }
            )
        manifest_rows.append(
            {
                "record_type": "candidate_group",
                "query_index": selection["query_index"],
                "query_smiles": selection["query_smiles"],
                "group_id": selection["group_id"],
                "assay_transfer_initial_morgan_filter": initial_morgan_filter,
                "candidates": frozen_candidates,
            }
        )
    catalog_records = sorted(by_id.values(), key=lambda row: str(row["record_id"]))
    records_digest = hashlib.sha256(
        "\n".join(_canonical_json(row) for row in catalog_records).encode("utf-8")
    ).hexdigest()
    catalog_version = f"{CATALOG_SCHEMA_VERSION}:{records_digest}"
    metadata = {
        "record_type": "catalog_metadata",
        "schema_version": CATALOG_SCHEMA_VERSION,
        "catalog_version": catalog_version,
        "source_mode": "flat_prepared_hf_validation_candidates",
        "conditions": [condition_id],
        "index_version": str(index.get("version") or ""),
        "template_hash": template_bundle_hash(),
        "n_queries": len(indices),
        "n_candidate_groups": len(manifest_rows),
        "n_records": len(catalog_records),
        "records_sha256": records_digest,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
        for record in catalog_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_metadata = {
        "record_type": "manifest_metadata",
        "schema_version": "assay_transfer_candidate_manifest.flat.v2",
        "condition_id": condition_id,
        "catalog_version": catalog_version,
        "index_path": index_path,
        "index_version": str(index.get("version") or ""),
        "identity_policy": neighbor_identity_policy,
        "n_queries": len(indices),
        "n_groups": len(manifest_rows),
        "n_candidates": sum(len(row["candidates"]) for row in manifest_rows),
        "min_similarity": min_similarity,
        "assay_transfer_initial_morgan_filter": initial_morgan_filter,
    }
    with manifest_output.open("w", encoding="utf-8") as handle:
        for row in [manifest_metadata, *manifest_rows]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        **metadata,
        "manifest_metadata": manifest_metadata,
        "candidate_manifest": str(manifest_output),
        "catalog": str(output),
    }


def build_manifest_for_prebuilt_catalog(
    *,
    records: list[dict[str, Any]],
    indices: list[int],
    smiles_field: str,
    index: dict[str, Any],
    catalog_path: Path,
    manifest_output: Path,
    experiment_mode: str,
    top_k_per_group: int,
    min_similarity: float,
    neighbor_identity_policy: str,
    initial_morgan_filter: int,
    index_path: str = "",
    condition_id: str = "validation__prebuilt_catalog",
    template_profile: str = "v6_5_query_context_copy",
) -> dict[str, Any]:
    """Freeze candidate-to-record joins without rebuilding an immutable catalog."""
    from tools.chembl_tool.common.experiment_retrieval import retrieve_experiment_view
    from tools.chembl_tool.tasks.bioavailability_ma.experiment_config import (
        STARLING_IN_DISTRIBUTION,
    )
    from tools.chembl_tool.tasks.bioavailability_ma.reranking.assay_transfer_rerank import (
        AssayTransferCatalog,
    )

    catalog = AssayTransferCatalog(catalog_path, template_profile=template_profile)
    selections: list[dict[str, Any]] = []
    for query_index in indices:
        query_smiles = str(records[query_index].get(smiles_field) or "")
        collector = _CandidateCollector(query_index, selections)
        retrieve_experiment_view(
            query_smiles,
            index,
            mode=experiment_mode,
            config=STARLING_IN_DISTRIBUTION,
            top_k_per_group=top_k_per_group,
            min_similarity=min_similarity,
            neighbor_identity_policy=neighbor_identity_policy,
            reranker=collector,
            assay_transfer_initial_morgan_filter=initial_morgan_filter,
        )

    manifest_rows = [
        _prebuilt_manifest_row(selection, catalog, initial_morgan_filter)
        for selection in selections
    ]
    metadata = {
        "record_type": "manifest_metadata",
        "schema_version": "assay_transfer_candidate_manifest.flat.v2",
        "condition_id": condition_id,
        "catalog_version": catalog.catalog_version,
        "index_path": index_path,
        "index_version": str(index.get("version") or ""),
        "identity_policy": neighbor_identity_policy,
        "n_queries": len(indices),
        "n_groups": len(manifest_rows),
        "n_candidates": sum(len(row["candidates"]) for row in manifest_rows),
        "n_scoreable_candidates": sum(
            bool(candidate["record_ids"])
            for row in manifest_rows
            for candidate in row["candidates"]
        ),
        "min_similarity": min_similarity,
        "assay_transfer_initial_morgan_filter": initial_morgan_filter,
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with manifest_output.open("w", encoding="utf-8") as handle:
        for row in [metadata, *manifest_rows]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {**metadata, "candidate_manifest": str(manifest_output)}


def _prebuilt_manifest_row(
    selection: dict[str, Any],
    catalog: Any,
    initial_morgan_filter: int,
) -> dict[str, Any]:
    candidates = []
    for candidate in selection["candidates"]:
        compatible = catalog.compatible_records(
            selection["group_id"], str(candidate["canonical_smiles"])
        )
        record_ids = sorted(str(record["record_id"]) for record in compatible)
        candidates.append(
            {
                "molecule_id": candidate["molecule_chembl_id"],
                "canonical_smiles": candidate["canonical_smiles"],
                "similarity": candidate["similarity"],
                "structural_rank": candidate["structural_rank"],
                "record_ids": record_ids,
            }
        )
    return {
        "record_type": "candidate_group",
        "query_index": selection["query_index"],
        "query_smiles": selection["query_smiles"],
        "group_id": selection["group_id"],
        "assay_transfer_initial_morgan_filter": initial_morgan_filter,
        "candidates": candidates,
    }


def _records_from_index_candidate(group_id: str, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    concept = {
        "Observed.direct_oral_bioavailability": "oral_bioavailability",
        "Observed.nondirect_oral_bioavailability": "oral_bioavailability",
        "Observed.oral_auc_cmax_exposure": "oral_exposure",
        "Fa.absorption_solubility_permeability": "Fa",
        "Fg.gut_wall_efflux_intestinal_metabolism": "Fg",
        "Fh.hepatic_clearance_metabolic_stability": "Fh",
    }[group_id]
    output = []
    for evidence_row in candidate.get("evidence_rows") or []:
        for position, example in enumerate(evidence_row.get("source_record_examples") or []):
            record = _index_example_record(concept, candidate, evidence_row, example, position)
            if record is not None:
                output.append(record)
    return output


def _index_example_record(
    concept: str,
    candidate: dict[str, Any],
    evidence_row: dict[str, Any],
    example: dict[str, Any],
    position: int,
) -> dict[str, Any] | None:
    if concept == "oral_bioavailability":
        if example.get("oral_bioavailability_value_percent") not in (None, ""):
            # starling-labs/Oral_Bioavailability schema.
            raw_value = example.get("oral_bioavailability_value_percent")
            endpoint = "oral_bioavailability"
            unit_basis = "percent"
            context = {
                "species_or_population": example.get("species_or_population"),
                "report_or_statistic_type": example.get("bioavailability_report_type"),
                "dose": example.get("dose"),
                "oral_exposure_mode": example.get("oral_exposure_mode"),
                "study_or_assay_system": example.get("oral_exposure_mode"),
                "qualifying_conditions": example.get("qualifying_conditions"),
                "comparator": example.get("comparator"),
                "extra_details": example.get("extra_details"),
            }
        else:
            # Oral_AUC-Cmax_Exposure rows assigned to the direct family retain
            # the generic Starling scalar schema. These were previously omitted.
            raw_value = example.get("reported_value")
            endpoint = str(example.get("endpoint_type") or "bioavailability")
            unit_basis = _unit_basis(str(example.get("reported_units") or ""), endpoint)
            raw_context = dict(example.get("context") or {})
            context = {
                "species_or_population": raw_context.get("species")
                or raw_context.get("species_or_population"),
                "report_or_statistic_type": raw_context.get("statistic_type"),
                "dose": raw_context.get("oral_dose"),
                "study_context": raw_context.get("study_context"),
                "study_or_assay_system": raw_context.get("study_context"),
                "qualifying_conditions": raw_context.get("qualifying_conditions"),
                "comparator": raw_context.get("comparator_exposure"),
                "extra_details": example.get("extra_details"),
            }
    else:
        raw_value = example.get("reported_value")
        endpoint = str(example.get("endpoint_type") or "measurement")
        unit_basis = _unit_basis(str(example.get("reported_units") or ""), endpoint)
        raw_context = dict(example.get("context") or {})
        context = {
            "species_or_population": raw_context.get("species") or raw_context.get("species_or_population"),
            "report_or_statistic_type": raw_context.get("statistic_type"),
            "dose": raw_context.get("oral_dose"),
            "study_context": raw_context.get("study_context"),
            "assay_system": raw_context.get("assay_system"),
            "study_or_assay_system": raw_context.get("study_context") or raw_context.get("assay_system"),
            "measured_process": example.get("endpoint_type"),
            "biological_context": raw_context.get("biological_context"),
            "medium": raw_context.get("condition_medium"),
            "formulation_or_solid_form": raw_context.get("formulation_or_solid_form"),
            "transporter_or_enzyme": raw_context.get("transporter_or_enzyme"),
            "substrate_status": raw_context.get("substrate_status"),
            "intestinal_site": raw_context.get("intestinal_site"),
            "molecular_form": raw_context.get("molecular_form"),
            "enzyme_or_pathway": raw_context.get("enzyme_or_pathway"),
            "qualifying_conditions": raw_context.get("qualifying_conditions"),
            "comparator": raw_context.get("comparator_exposure"),
            "extra_details": example.get("extra_details"),
        }
    scalar = _first_number(raw_value)
    if scalar is None:
        return None
    endpoint_subtype = _slug(endpoint)
    metric_type, threshold_display = _v3_scoring_profile(
        concept=concept, endpoint_subtype=endpoint_subtype, unit_basis=unit_basis
    )
    identity = {
        "group": evidence_row.get("group_id"),
        "molecule": candidate.get("molecule_chembl_id"),
        "source_index": example.get("source_index"),
        "source_record_id": example.get("source_record_id"),
        "position": position,
        "endpoint": endpoint,
        "value": str(raw_value),
    }
    record_id = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    return {
        "record_type": "assay_record",
        "record_id": record_id,
        "source_id": str(example.get("source_id") or evidence_row.get("evidence_source") or ""),
        "original_smiles": str(candidate["canonical_smiles"]),
        "canonical_smiles": str(candidate["canonical_smiles"]),
        "assay_concept": concept,
        "canonical_endpoint_key": f"index.{concept}.{endpoint_subtype}.{unit_basis}",
        "endpoint_family": concept,
        "endpoint_subtype": endpoint_subtype,
        "unit_basis": unit_basis,
        "metric_type": metric_type,
        "threshold_display": threshold_display,
        "value": scalar,
        "value_display": str(raw_value).strip(),
        "measurement_label": str(endpoint).replace("_", " ").lower(),
        "template_id": Path(TEMPLATE_BY_CONCEPT[concept]).stem,
        "template_context": _context(**context),
        "source_provenance": identity,
    }


def _first_number(value: Any) -> float | None:
    text = "" if value is None else str(value)
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else None


def _slug(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_") or "measurement"


def _unit_basis(units: str, endpoint: str) -> str:
    lowered = units.lower().strip()
    endpoint_lower = endpoint.lower()
    if "%" in lowered or "percent" in lowered:
        return "percent"
    if "ratio" in endpoint_lower or lowered in {"ratio", "fold", "dimensionless"}:
        return "dimensionless_ratio"
    if "fraction" in endpoint_lower and not lowered:
        return "fraction"
    return _slug(units) if units.strip() else "not_reported"


@dataclass(frozen=True)
class CleanedCatalogSource:
    concept: str
    repo_id: str
    revision: str
    raw_local_path: str = ""
    raw_repo_id: str = ""
    raw_revision: str = ""


CLEANED_SOURCES = (
    CleanedCatalogSource(
        "oral_bioavailability",
        "jiosephlee/starling_oba_cleaned",
        "db1ed61990242a6e978e04c03e9a7a67dfa85a25",
        raw_repo_id="starling-labs/Oral_Bioavailability",
        raw_revision="01bbe3ee9cdd3dc081c39973529c9da0c814d465",
    ),
    CleanedCatalogSource(
        "oral_exposure",
        "jiosephlee/Oral_bioavailability_cleaned",
        "fef03eca7c94329e9ef549c5904e0d5d0b14c9b7",
        raw_local_path="Oral_AUC-Cmax_Exposure/extractions.parquet",
    ),
    CleanedCatalogSource(
        "Fa",
        "jiosephlee/intestinal_absorption_cleaned",
        "a9c8940282795600b477138f76682930d8331282",
        raw_local_path="Fa/extractions.parquet",
    ),
    CleanedCatalogSource(
        "Fg",
        "jiosephlee/gut_wall_cleaned",
        "0443b21b1e712e0269789e52e9863d4cde0eb4ee",
        raw_local_path="Fg/extractions.parquet",
    ),
    CleanedCatalogSource(
        "Fh",
        "jiosephlee/hepatic_cleaned",
        "eaff3469ac7f60b34e27dba645c08bce6a677497",
        raw_local_path="Fh/extractions.parquet",
    ),
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.source_mode == "cleaned_sources":
        records, source_provenance = build_catalog_records_from_cleaned_sources(
            starling_data_dir=Path(args.starling_data_dir),
            force_download=args.force_download,
            local_files_only=args.local_files_only,
        )
    else:
        records = build_catalog_records(
            dataset=args.dataset,
            revision=args.dataset_revision,
            splits=tuple(args.splits),
            force_download=args.force_download,
            local_files_only=args.local_files_only,
        )
        source_provenance = [{"repo_id": args.dataset, "revision": args.dataset_revision}]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records_digest = hashlib.sha256(
        "\n".join(_canonical_json(row) for row in records).encode("utf-8")
    ).hexdigest()
    catalog_version = f"{CATALOG_SCHEMA_VERSION}:{records_digest}"
    metadata = {
        "record_type": "catalog_metadata",
        "schema_version": CATALOG_SCHEMA_VERSION,
        "catalog_version": catalog_version,
        "source_mode": args.source_mode,
        "source_provenance": source_provenance,
        "template_hash": template_bundle_hash(),
        "n_records": len(records),
        "records_sha256": records_digest,
    }
    with output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({**metadata, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


def build_catalog_records_from_cleaned_sources(
    *,
    starling_data_dir: Path,
    force_download: bool = False,
    local_files_only: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize every accepted scalar v3 record, including records unused in SFT pairs."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for source in CLEANED_SOURCES:
        cleaned_path = hf_hub_download(
            repo_id=source.repo_id,
            repo_type="dataset",
            filename="data/train-00000-of-00001.parquet",
            revision=source.revision,
            force_download=force_download,
            local_files_only=local_files_only,
        )
        cleaned_file = pq.ParquetFile(cleaned_path)
        source_count = 0
        for batch in cleaned_file.iter_batches(batch_size=4096):
            for cleaned in batch.to_pylist():
                # The immutable cleaned row already retains every original field used by
                # the v3 prompt. source_row_number is provenance, not a parquet row offset.
                records.append(_cleaned_catalog_record(source.concept, cleaned, cleaned))
                source_count += 1
        provenance.append(
            {
                "concept": source.concept,
                "repo_id": source.repo_id,
                "revision": source.revision,
                "n_records": source_count,
            }
        )
    records.sort(key=lambda row: str(row["record_id"]))
    if len({str(row["record_id"]) for row in records}) != len(records):
        raise ValueError("Cleaned assay-transfer catalog contains duplicate child record IDs")
    return records, provenance


def _cleaned_catalog_record(
    concept: str,
    cleaned: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    from tools.chembl_tool.tasks.bioavailability_ma.starling_source_column_contracts import (
        llm_source_projection_from_mapping,
    )

    canonical_smiles = str(cleaned["smiles"])
    original_smiles = str(raw.get("smiles") or canonical_smiles)
    scalar_value = float(cleaned["scalar_value"])
    unit_basis = str(cleaned["unit_basis"])
    metric_type, threshold_display = _v3_scoring_profile(
        concept=concept,
        endpoint_subtype=str(cleaned["endpoint_subtype"]),
        unit_basis=unit_basis,
    )
    source_contract_id = {
        "oral_bioavailability": "hf_bioavailability",
        "oral_exposure": "oral_exposure",
        "Fa": "fa",
        "Fg": "fg",
        "Fh": "fh",
    }[concept]
    source_projection = llm_source_projection_from_mapping(source_contract_id, raw)
    return {
        "record_type": "assay_record",
        "record_id": str(cleaned["child_id"]),
        "source_id": str(cleaned.get("source_id") or ""),
        "original_smiles": original_smiles,
        "canonical_smiles": canonical_smiles,
        "assay_concept": concept,
        "canonical_endpoint_key": str(cleaned["canonical_endpoint_key"]),
        "endpoint_family": str(cleaned["endpoint_family"]),
        "endpoint_subtype": str(cleaned["endpoint_subtype"]),
        "unit_basis": unit_basis,
        "metric_type": metric_type,
        "threshold_display": threshold_display,
        "value": scalar_value,
        "value_display": _format_cleaned_value(concept, scalar_value),
        "measurement_label": _measurement_label(concept, cleaned, raw),
        "template_id": Path(TEMPLATE_BY_CONCEPT[concept]).stem,
        "template_context": _template_context(concept, cleaned, raw),
        "source_provenance": {
            "parent_provenance_id": str(cleaned.get("parent_provenance_id") or ""),
            "record_id": str(cleaned.get("record_id") or ""),
            "input_sha256": str(cleaned.get("input_sha256") or ""),
            "child_id": str(cleaned.get("child_id") or ""),
        },
        "source_contract": {
            key: value
            for key, value in source_projection.items()
            if key != "source_fields"
        },
        "source_fields": source_projection["source_fields"],
    }


def _v3_scoring_profile(
    *,
    concept: str,
    endpoint_subtype: str,
    unit_basis: str,
) -> tuple[str, str]:
    """Return the immutable metric/threshold mapping used to train the v3 model."""
    if concept == "oral_exposure":
        metric_type = (
            "dimensionless_ratio" if endpoint_subtype == "oral_iv_ratio" else "positive_scalar"
        )
    elif unit_basis == "dimensionless_ratio":
        metric_type = "dimensionless_ratio"
    elif unit_basis == "fraction":
        metric_type = "bounded_fraction"
    elif unit_basis == "percent":
        metric_type = "bounded_percentage"
    else:
        metric_type = "positive_scalar"

    thresholds = {
        "dimensionless_ratio": "within 1.5-fold / at least 3-fold apart",
        "bounded_fraction": "within 0.10 / at least 0.30 apart",
        "bounded_percentage": (
            "within 10 percentage points / at least 30 percentage points apart"
        ),
        "positive_scalar": "within 2-fold / at least 5-fold apart",
    }
    return metric_type, thresholds[metric_type]


def _template_context(concept: str, cleaned: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    if concept == "oral_bioavailability":
        return _context(
            species_or_population=raw.get("species_or_population"),
            report_or_statistic_type=raw.get("bioavailability_report_type"),
            dose=raw.get("dose"),
            oral_exposure_mode=raw.get("oral_exposure_mode"),
            study_or_assay_system=raw.get("oral_exposure_mode"),
            qualifying_conditions=raw.get("qualifying_conditions"),
            comparator=raw.get("comparator"),
            extra_details=raw.get("extra_details"),
        )
    if concept == "oral_exposure":
        return _context(
            species_or_population=cleaned.get("species_exact"),
            measured_process=raw.get("exposure_measure"),
            report_or_statistic_type=raw.get("statistic_type"),
            dose=raw.get("oral_dose"),
            study_context=raw.get("study_context"),
            study_or_assay_system=raw.get("study_context"),
            comparator=raw.get("comparator_exposure"),
            qualifying_conditions=raw.get("qualifying_conditions"),
        )
    if concept == "Fa":
        return _context(
            assay_system=raw.get("assay_system"),
            study_or_assay_system=raw.get("assay_system"),
            biological_context=raw.get("biological_context"),
            medium=raw.get("condition_medium"),
            formulation_or_solid_form=raw.get("formulation_or_solid_form"),
            qualifying_conditions=raw.get("qualifying_conditions"),
            extra_details=raw.get("extra_details"),
        )
    if concept == "Fg":
        return _context(
            measured_process=raw.get("gut_wall_process"),
            transporter_or_enzyme=raw.get("transporter_or_enzyme"),
            substrate_status=raw.get("substrate_status"),
            assay_system=raw.get("assay_system"),
            study_or_assay_system=raw.get("assay_system"),
            intestinal_site=raw.get("intestinal_site"),
            qualifying_conditions=raw.get("qualifying_conditions"),
        )
    return _context(
        assay_system=raw.get("assay_system"),
        study_or_assay_system=raw.get("assay_system"),
        species_or_population=raw.get("species"),
        molecular_form=raw.get("molecular_form"),
        enzyme_or_pathway=raw.get("enzyme_or_pathway"),
        qualifying_conditions=raw.get("qualifying_conditions"),
        extra_details=raw.get("extra_details"),
    )


def _context(**values: Any) -> dict[str, Any]:
    return {key: _clean_value(value) for key, value in values.items()}


def _clean_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


def _format_cleaned_value(concept: str, value: float) -> str:
    return f"{value:.2f}" if concept == "oral_bioavailability" else f"{value:.12g}"


def _measurement_label(concept: str, cleaned: dict[str, Any], raw: dict[str, Any]) -> str:
    if concept == "oral_bioavailability":
        return "oral bioavailability"
    if concept == "oral_exposure":
        return str(raw.get("exposure_measure") or cleaned["endpoint_subtype"]).lower()
    if concept == "Fa":
        return str(raw.get("endpoint_category") or cleaned["endpoint_subtype"]).replace("_", " ")
    if concept == "Fg":
        return str(cleaned["endpoint_subtype"]).replace("_", " ")
    return str(cleaned["endpoint_subtype"]).replace("_", " ")


def build_catalog_records(
    *,
    dataset: str,
    revision: str,
    splits: tuple[str, ...],
    force_download: bool = False,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    by_id: dict[str, dict[str, Any]] = {}
    canonical_smiles_cache: dict[str, str] = {}
    for split in splits:
        path = hf_hub_download(
            repo_id=dataset,
            repo_type="dataset",
            filename=f"{split}/data.parquet",
            revision=revision,
            force_download=force_download,
            local_files_only=local_files_only,
        )
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["prompt", "metadata"], batch_size=4096):
            for row in batch.to_pylist():
                metadata = dict(row["metadata"] or {})
                concept = str(metadata.get("assay_concept") or "")
                if concept not in TEMPLATE_BY_CONCEPT:
                    continue
                record = _catalog_record(
                    str(row["prompt"]), metadata, canonical_smiles_cache=canonical_smiles_cache
                )
                record_id = str(record["record_id"])
                previous = by_id.get(record_id)
                if previous is not None and previous != record:
                    raise ValueError(f"Conflicting catalog payloads for record_id={record_id}")
                by_id[record_id] = record
    return sorted(by_id.values(), key=lambda row: str(row["record_id"]))


def _catalog_record(
    prompt: str,
    metadata: dict[str, Any],
    *,
    canonical_smiles_cache: dict[str, str] | None = None,
) -> dict[str, Any]:
    original_smiles = str(metadata.get("retrieved_original_smiles") or metadata.get("retrieved_smiles") or "")
    input_smiles = str(metadata.get("retrieved_smiles") or original_smiles)
    cache = canonical_smiles_cache if canonical_smiles_cache is not None else {}
    if input_smiles not in cache:
        cache[input_smiles] = standardize_smiles(input_smiles)[0]
    canonical_smiles = cache[input_smiles]
    if not canonical_smiles:
        raise ValueError(f"Invalid retrieved SMILES for record {metadata.get('retrieval_record_id')}")
    unit_basis = str(metadata.get("unit_basis") or "")
    return {
        "record_type": "assay_record",
        "record_id": str(metadata["retrieval_record_id"]),
        "source_id": str(metadata.get("retrieved_source_id") or ""),
        "original_smiles": original_smiles,
        "canonical_smiles": canonical_smiles,
        "assay_concept": str(metadata["assay_concept"]),
        "canonical_endpoint_key": str(metadata["canonical_endpoint_key"]),
        "endpoint_family": str(metadata["endpoint_family"]),
        "endpoint_subtype": str(metadata["endpoint_subtype"]),
        "unit_basis": unit_basis,
        "metric_type": str(metadata["metric_type"]),
        "threshold_display": str(metadata["threshold_display"]),
        "value": float(metadata["retrieved_value"]),
        "value_display": _extract_value_display(prompt, unit_basis),
        "measurement_label": str(metadata.get("retrieved_measurement_label") or ""),
        "template_id": str(metadata["template_id"]),
        "template_context": _clean_mapping(metadata.get("retrieval_context") or {}),
        "source_provenance": _clean_mapping(metadata.get("provenance") or {}),
    }


def _extract_value_display(prompt: str, unit_basis: str) -> str:
    match = re.search(r"(?m)^- known value: (.+)$", prompt)
    if not match:
        raise ValueError("Assay-transfer prompt has no known-value line")
    display = match.group(1).strip()
    suffix = f" {unit_basis}"
    if unit_basis and display.endswith(suffix):
        display = display[: -len(suffix)]
    return display


def _clean_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items() if item is not None}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mode", choices=["cleaned_sources", "pair_dataset"], default="cleaned_sources")
    parser.add_argument("--starling-data-dir", default=DEFAULT_STARLING_DATA_DIR)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--output", default=DEFAULT_SOURCE_CATALOG)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
