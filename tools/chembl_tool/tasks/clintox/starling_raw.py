"""Raw-first ClinTox Starling ingestion, evidence indexing, and benchmark CLI.

The pipeline performs only source-preserving cleanup and organization.  It
does not canonicalize endpoint values/units, infer scalar measurements, build
assay-transfer buckets, or use the supplied ``global_identifier`` mapping.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import shutil
import tarfile
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.common.evidence_contract import attach_minimal_evidence
from tools.chembl_tool.common.molecule_identity import (
    IDENTITY_NORMALIZER_VERSION,
    normalize_molecule_identity,
)
from tools.chembl_tool.common.starling.benchmark_dataset import (
    build_benchmark_dataset,
    sha256_file,
)
from tools.chembl_tool.common.starling.build_record_supported_benchmark import (
    build_task as build_record_supported_task,
)
from tools.chembl_tool.common.starling.evidence_library import starling_molecule_id
from tools.chembl_tool.common.task_workflows.evidence_library import (
    build_neighbor_index,
    fingerprint_metadata,
)
from tools.chembl_tool.tasks.clintox.starling_human_toxicity_benchmark import (
    DEFAULT_SOURCE_PATH,
    LABELS,
    TASK_NAME,
    is_direct_gold_scope,
    load_label_decisions,
)


ARCHIVE_SHA256 = "5f34b79a9b8740ef2371b2f691644f52245bbc6903963abccc92e6000656170b"
RAW_VERSION = "starling_raw_v1"
LIBRARY_VERSION = "clintox.starling_raw_v1.evidence_library.v1"
DEFAULT_ARCHIVE = Path("/vast/projects/myatskar/lab/shared_docs/clintox_send.tar.gz")
DEFAULT_RAW_ROOT = Path("data/starling_data/clintox/raw_v1")
DEFAULT_LOCAL_ROOT = Path(
    "outputs/chembl_tool/tasks/clintox/evidence_library/starling_raw_v1"
)
DEFAULT_BASE_BENCHMARK_ROOT = Path("data/processed_starling")
DEFAULT_RECORD_SUPPORTED_ROOT = Path("data/processed_starling_record_supported_v2")
SOURCE_BATCH_SIZE = 32_768
NULL_TEXT_VALUES = frozenset({"", "nan", "none", "null", "n/a", "na"})


@dataclass(frozen=True)
class SourceSpec:
    archive_version: str
    source_id: str
    title: str
    default_family: str
    endpoint_field: str
    source_fields: tuple[str, ...]

    @property
    def raw_dir(self) -> Path:
        return DEFAULT_RAW_ROOT / self.source_id


SOURCE_SPECS = (
    SourceSpec(
        "v1", "nonclinical_in_vivo_toxicity", "Nonclinical in vivo toxicity",
        "in_vivo_toxicology", "observed_effect",
        ("evidence_type", "endpoint_value", "endpoint_unit", "administered_dose", "animal_context", "exposure_context", "observed_effect", "qualifying_conditions", "extra_details"),
    ),
    SourceSpec(
        "v2", "organ_specific_toxicity", "Organ-specific toxicity",
        "organ_specific_toxicity", "toxicity_endpoint",
        ("organ_system", "toxicity_endpoint", "effect_status", "evidence_context", "biological_system", "exposure_regimen", "quantitative_result", "qualifying_conditions", "extra_details"),
    ),
    SourceSpec(
        "v3", "genotoxicity_carcinogenicity", "Genotoxicity and carcinogenicity",
        "genotoxicity_carcinogenicity", "endpoint",
        ("evidence_category", "assay_type", "study_context", "endpoint", "result_direction", "biological_system", "exposure_conditions", "qualifying_conditions", "extra_details"),
    ),
    SourceSpec(
        "v4", "cellular_stress", "Cellular stress",
        "cellular_stress_pathways", "stress_endpoint",
        ("stress_endpoint", "effect_direction", "evidence_basis", "mechanistic_effect", "target_or_pathway", "biological_model", "dose_and_duration", "qualifying_conditions", "extra_details"),
    ),
    SourceSpec(
        "v5", "general_cytotoxicity", "General cytotoxicity",
        "general_cytotoxicity", "endpoint_type",
        ("cell_model", "endpoint_type", "result_value", "result_unit", "test_concentration", "exposure_time_h", "assay_method", "qualifying_conditions", "extra_details"),
    ),
    SourceSpec(
        "v6", "off_target_ddi_exposure", "Off-target, DDI, and exposure liability",
        "off_target_ddi_exposure", "target_or_endpoint",
        ("evidence_type", "target_or_endpoint", "target_identifier", "result_metric", "result_value", "result_unit", "assay_context", "qualifying_conditions", "extra_details"),
    ),
)
SPEC_BY_ID = {spec.source_id: spec for spec in SOURCE_SPECS}
UNION_SOURCE_FIELDS = tuple(sorted({field for spec in SOURCE_SPECS for field in spec.source_fields}))


def import_archive(
    archive: str | Path = DEFAULT_ARCHIVE,
    *,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    """Extract the six exact Parquets/guides and write a checksummed manifest."""
    archive_path = Path(archive)
    root = Path(raw_root)
    digest = sha256_file(archive_path)
    if digest != ARCHIVE_SHA256:
        raise ValueError(f"unexpected ClinTox archive SHA-256: {digest}")
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(f"refusing to overwrite non-empty raw source root: {root}")
    if root.exists() and force:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    files: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "r:gz") as archive_handle:
        members = {member.name: member for member in archive_handle.getmembers()}
        for spec in SOURCE_SPECS:
            destination = root / spec.source_id
            destination.mkdir(parents=True, exist_ok=True)
            for filename in ("extractions.parquet", "extraction_guidance.json"):
                member_name = f"clintox_send/{spec.archive_version}/{filename}"
                member = members.get(member_name)
                if member is None or not member.isfile():
                    raise FileNotFoundError(f"archive member is absent: {member_name}")
                source = archive_handle.extractfile(member)
                if source is None:
                    raise FileNotFoundError(f"cannot read archive member: {member_name}")
                target = destination / filename
                with target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                files.append(
                    {
                        "source_id": spec.source_id,
                        "archive_member": member_name,
                        "path": str(target),
                        "size": target.stat().st_size,
                        "sha256": sha256_file(target),
                    }
                )

    source_summaries = []
    for spec in SOURCE_SPECS:
        parquet_path = root / spec.source_id / "extractions.parquet"
        parquet = pq.ParquetFile(parquet_path)
        source_summaries.append(
            {
                "archive_version": spec.archive_version,
                "source_id": spec.source_id,
                "title": spec.title,
                "n_rows": parquet.metadata.num_rows,
                "columns": parquet.schema_arrow.names,
                "default_mechanism_family": spec.default_family,
            }
        )
    manifest = {
        "schema_version": "clintox_raw_source_manifest.v1",
        "raw_version": RAW_VERSION,
        "archive": {
            "provided_path": str(archive_path),
            "size": archive_path.stat().st_size,
            "sha256": digest,
        },
        "known_missing_provenance": [
            "upstream dataset revision",
            "extraction model and software version",
            "entity-mapping method",
            "license",
        ],
        "structure_policy": "Exact raw files retain global_identifier; all derived artifacts ignore it and use only source SMILES.",
        "sources": source_summaries,
        "files": files,
    }
    _write_json(root / "SOURCE_MANIFEST.json", manifest)
    (root / "README.md").write_text(_source_readme(manifest), encoding="utf-8")
    return manifest


def build_evidence_library(
    *,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    workers: int = 1,
) -> dict[str, Any]:
    """Build the six simple-cleaning/organization/index/audit stages."""
    raw = Path(raw_root)
    root = Path(local_root)
    root.mkdir(parents=True, exist_ok=True)
    clean_path = root / "01_cleaned" / "records.parquet"
    organized_path = root / "02_organized" / "records.parquet"
    exclusions_path = root / "02_organized" / "structure_exclusions.parquet"
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    organized_path.parent.mkdir(parents=True, exist_ok=True)
    counts = _write_cleaned(raw, clean_path)
    structure_lookup = _build_structure_lookup(clean_path)
    organization = _write_organized(
        clean_path, organized_path, exclusions_path, structure_lookup
    )
    evidence_rows, catalog_stats = _write_evidence_catalog(
        organized_path, root / "03_evidence_catalog"
    )
    index_meta = _write_index(
        evidence_rows,
        root / "04_neighbor_index",
        index_version=LIBRARY_VERSION,
        workers=workers,
    )
    audit_root = root / "05_audits"
    audit_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "clintox_starling_raw_library.v1",
        "raw_version": RAW_VERSION,
        "library_version": LIBRARY_VERSION,
        "identity_normalizer_version": IDENTITY_NORMALIZER_VERSION,
        "cleaning_policy": {
            "operations": ["trim surrounding whitespace", "collapse internal whitespace", "map shared null sentinels to null"],
            "null_sentinels_case_insensitive": sorted(NULL_TEXT_VALUES),
            "not_null_sentinels": ["-", "unspecified"],
            "semantic_normalization": False,
            "global_identifier_used": False,
        },
        "source_counts": counts,
        "organization": organization,
        "catalog": catalog_stats,
        "index": index_meta,
        "stages": [
            "01_cleaned", "02_organized", "03_evidence_catalog",
            "04_neighbor_index", "05_audits", "06_record_supported_v2_scaffold_view",
        ],
    }
    _write_json(audit_root / "summary.json", summary)
    _write_json(root / "manifest.json", summary)
    return summary


def resume_evidence_library(
    *,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    workers: int = 1,
) -> dict[str, Any]:
    """Resume after the expensive cleaned/organized Parquets already exist."""
    root = Path(local_root)
    clean_path = root / "01_cleaned" / "records.parquet"
    organized_path = root / "02_organized" / "records.parquet"
    if not clean_path.is_file() or not organized_path.is_file():
        raise FileNotFoundError("resume requires completed 01_cleaned and 02_organized Parquets")
    evidence_rows, catalog_stats = _write_evidence_catalog(
        organized_path, root / "03_evidence_catalog"
    )
    index_meta = _write_index(
        evidence_rows,
        root / "04_neighbor_index",
        index_version=LIBRARY_VERSION,
        workers=workers,
    )
    audit = _audit_existing_organized(organized_path)
    summary = {
        "schema_version": "clintox_starling_raw_library.v1",
        "raw_version": RAW_VERSION,
        "library_version": LIBRARY_VERSION,
        "identity_normalizer_version": IDENTITY_NORMALIZER_VERSION,
        "cleaning_policy": {
            "operations": ["trim surrounding whitespace", "collapse internal whitespace", "map shared null sentinels to null"],
            "null_sentinels_case_insensitive": sorted(NULL_TEXT_VALUES),
            "not_null_sentinels": ["-", "unspecified"],
            "semantic_normalization": False,
            "global_identifier_used": False,
        },
        "organization": audit,
        "catalog": catalog_stats,
        "index": index_meta,
        "stages": [
            "01_cleaned", "02_organized", "03_evidence_catalog",
            "04_neighbor_index", "05_audits", "06_record_supported_v2_scaffold_view",
        ],
    }
    _write_json(root / "05_audits" / "summary.json", summary)
    _write_json(root / "manifest.json", summary)
    return summary


def build_candidate_benchmark(
    *,
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    base_root: str | Path = DEFAULT_BASE_BENCHMARK_ROOT,
    record_supported_root: str | Path = DEFAULT_RECORD_SUPPORTED_ROOT,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    max_rows: int = 0,
) -> dict[str, Any]:
    """Build 70%-agreement labels, record-supported v2 scaffold split, and QA sample."""
    decisions, metadata = load_label_decisions(source_path=source_path, max_rows=max_rows)
    base_summary = build_benchmark_dataset(
        task_name=TASK_NAME,
        decisions=decisions,
        source_metadata=metadata,
        output_dir=Path(base_root) / TASK_NAME,
    )
    base_task_root = Path(base_root) / TASK_NAME
    legacy_report = base_task_root / "report_zh.md"
    if legacy_report.exists():
        legacy_report.unlink()
    (base_task_root / "REPORT.md").write_text(
        _benchmark_report(base_summary), encoding="utf-8"
    )
    final_summary = build_record_supported_task(
        TASK_NAME,
        source_root=Path(base_root),
        output_root=Path(record_supported_root),
    )
    task_root = Path(record_supported_root) / TASK_NAME
    candidate = {
        "status": "candidate_pending_qa",
        "promotion_policy": "Do not add to default paper matrices until the deterministic stratified source-record QA sample passes.",
        "task_definition": metadata["task_definition"],
        "base_summary": base_summary,
        "record_supported_v2_summary": final_summary,
    }
    _write_json(task_root / "CANDIDATE_STATUS.json", candidate)
    audit_root = Path(local_root) / "05_audits"
    audit_root.mkdir(parents=True, exist_ok=True)
    sample = _write_gold_qa_sample(Path(source_path), audit_root / "gold_qa_sample.parquet")
    _write_json(
        audit_root / "gold_qa_status.json",
        {"status": "pending_manual_review", **sample},
    )
    return candidate


def build_heldout_filtered_view(
    *,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    record_supported_root: str | Path = DEFAULT_RECORD_SUPPORTED_ROOT,
    workers: int = 1,
) -> dict[str, Any]:
    """Remove every valid/test parent from all evidence families and rebuild the index."""
    root = Path(local_root)
    split_root = Path(record_supported_root) / TASK_NAME / "scaffold"
    heldout_path = split_root / "heldout_molecule_labels.jsonl"
    heldout_keys = {
        json.loads(line)["molecule_identity_key"]
        for line in heldout_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    catalog_path = root / "03_evidence_catalog" / "evidence.jsonl"
    retained: list[dict[str, Any]] = []
    excluded = 0
    for line in catalog_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        identity = normalize_molecule_identity(row["canonical_smiles"])
        key = identity.parent_inchi_key or identity.parent_smiles
        if key in heldout_keys and row.get("group_id") == "Direct.human_organ_toxicity":
            excluded += 1
        else:
            retained.append(row)
    stage = root / "06_record_supported_v2_scaffold_view"
    stage.mkdir(parents=True, exist_ok=True)
    _write_jsonl(stage / "evidence.jsonl", retained)
    index_meta = _write_index(
        retained,
        stage,
        index_version=f"{LIBRARY_VERSION}.record_supported_v2_scaffold_heldout_filtered",
        workers=workers,
    )
    retained_parent_keys = {
        normalize_molecule_identity(row["canonical_smiles"]).parent_inchi_key
        or normalize_molecule_identity(row["canonical_smiles"]).parent_smiles
        for row in retained
    }
    direct_parent_keys = {
        (normalize_molecule_identity(row["canonical_smiles"]).parent_inchi_key
         or normalize_molecule_identity(row["canonical_smiles"]).parent_smiles)
        for row in retained
        if row.get("group_id") == "Direct.human_organ_toxicity"
    }
    direct_overlap = len(direct_parent_keys & heldout_keys)
    all_source_overlap = len(retained_parent_keys & heldout_keys)
    summary = {
        "schema_version": "clintox_record_supported_v2_scaffold_view.v1",
        "benchmark_task": TASK_NAME,
        "heldout_contract": str(heldout_path),
        "n_heldout_parents": len(heldout_keys),
        "n_full_evidence_rows": len(retained) + excluded,
        "n_excluded_evidence_rows": excluded,
        "n_retained_evidence_rows": len(retained),
        "exclusion_scope": "direct gold source only; mechanism sources retained",
        "heldout_direct_source_parent_overlap": direct_overlap,
        "heldout_any_source_parent_overlap": all_source_overlap,
        "query_time_identity_policy_required": "parent_disjoint",
        "index": index_meta,
    }
    if direct_overlap:
        raise RuntimeError(f"held-out direct-source evidence leakage detected: {direct_overlap}")
    _write_json(stage / "summary.json", summary)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["record_supported_v2_scaffold_view"] = summary
    _write_json(manifest_path, manifest)
    return summary


def clean_text(value: Any) -> str | None:
    """Apply the exact basic-null and whitespace policy used by this pipeline."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return None if text.lower() in NULL_TEXT_VALUES else text


def route_record(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return mechanism family, retrieval group, and evidence role."""
    source_id = str(row.get("source_id") or "")
    spec = SPEC_BY_ID[source_id]
    if source_id != "organ_specific_toxicity":
        family = spec.default_family
        return family, f"Mechanism.{family}", "mechanistic_factor"
    if str(row.get("evidence_context") or "") == "human_clinical":
        benchmark_row = dict(row)
        benchmark_row["SMILES"] = row.get("source_smiles")
        if is_direct_gold_scope(benchmark_row):
            return "clinical_human_safety", "Direct.human_organ_toxicity", "direct_outcome"
        return "clinical_human_safety", "Context.human_organ_toxicity", "context_modifier"
    return "organ_specific_toxicity", "Mechanism.organ_specific_toxicity", "mechanistic_factor"


def _write_cleaned(raw_root: Path, destination: Path) -> dict[str, Any]:
    schema = _clean_schema()
    writer = pq.ParquetWriter(destination, schema, compression="zstd")
    counts: dict[str, Any] = {}
    try:
        for spec in SOURCE_SPECS:
            path = raw_root / spec.source_id / "extractions.parquet"
            parquet = pq.ParquetFile(path)
            written = 0
            source_row = 0
            for batch in parquet.iter_batches(batch_size=SOURCE_BATCH_SIZE):
                output = []
                for raw in batch.to_pylist():
                    row: dict[str, Any] = {
                        "record_id": f"{RAW_VERSION}:{spec.source_id}:{source_row:09d}",
                        "source_id": spec.source_id,
                        "source_version": spec.archive_version,
                        "source_row_number": source_row,
                        "paragraph_idx": raw.get("paragraph_idx"),
                        "support_text": clean_text(raw.get("support_text")),
                        "confidence": raw.get("confidence"),
                        "needs_more_context": raw.get("needs_more_context"),
                        "pmid": clean_text(raw.get("pmid")),
                        "extraction_id": clean_text(raw.get("extraction_id")),
                        "source_smiles": clean_text(raw.get("SMILES")),
                    }
                    row.update({field: clean_text(raw.get(field)) for field in UNION_SOURCE_FIELDS})
                    output.append(row)
                    source_row += 1
                writer.write_table(pa.Table.from_pylist(output, schema=schema))
                written += len(output)
            counts[spec.source_id] = {
                "path": str(path),
                "n_rows": written,
                "sha256": sha256_file(path),
            }
    finally:
        writer.close()
    return counts


def _build_structure_lookup(clean_path: Path) -> dict[str, dict[str, Any]]:
    values: set[str] = set()
    parquet = pq.ParquetFile(clean_path)
    for batch in parquet.iter_batches(columns=["source_smiles"], batch_size=131_072):
        values.update(value for value in batch.column(0).to_pylist() if value)
    lookup: dict[str, dict[str, Any]] = {}
    for smiles in sorted(values):
        identity = normalize_molecule_identity(smiles)
        structure_status = (
            identity.status
            if identity.status != "ok"
            else ("ok" if identity.parent_smiles else "unresolved_parent")
        )
        lookup[smiles] = {
            "structure_status": structure_status,
            "canonical_smiles": identity.canonical_smiles or None,
            "parent_smiles": identity.parent_smiles or None,
            "parent_inchi_key": identity.parent_inchi_key or None,
            "molecule_id": starling_molecule_id(identity.canonical_smiles) if identity.canonical_smiles else None,
        }
    return lookup


def _write_organized(
    clean_path: Path,
    destination: Path,
    exclusions_path: Path,
    lookup: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    schema = pa.schema(
        [*list(_clean_schema()),
         pa.field("structure_status", pa.string()),
         pa.field("canonical_smiles", pa.large_string()),
         pa.field("parent_smiles", pa.large_string()),
         pa.field("parent_inchi_key", pa.string()),
         pa.field("molecule_id", pa.string()),
         pa.field("mechanism_family", pa.string()),
         pa.field("retrieval_group", pa.string()),
         pa.field("evidence_role", pa.string()),
         pa.field("retrieval_status", pa.string())]
    )
    exclusion_schema = pa.schema(
        [pa.field("record_id", pa.string()), pa.field("source_id", pa.string()),
         pa.field("source_row_number", pa.int64()), pa.field("source_smiles", pa.large_string()),
         pa.field("reason", pa.string())]
    )
    writer = pq.ParquetWriter(destination, schema, compression="zstd")
    exclusion_writer = pq.ParquetWriter(exclusions_path, exclusion_schema, compression="zstd")
    status_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    try:
        parquet = pq.ParquetFile(clean_path)
        for batch in parquet.iter_batches(batch_size=SOURCE_BATCH_SIZE):
            rows = []
            exclusions = []
            for row in batch.to_pylist():
                smiles = row.get("source_smiles")
                identity = lookup.get(smiles or "", {})
                status = str(identity.get("structure_status") or "missing_smiles")
                retrieval_status = "eligible" if status == "ok" else status
                family, group, role = route_record(row)
                row.update(identity)
                row.update(
                    {
                        "structure_status": status,
                        "mechanism_family": family,
                        "retrieval_group": group,
                        "evidence_role": role,
                        "retrieval_status": retrieval_status,
                    }
                )
                rows.append(row)
                status_counts[retrieval_status] += 1
                family_counts[family] += 1
                if retrieval_status != "eligible":
                    exclusions.append(
                        {key: row.get(key) for key in ("record_id", "source_id", "source_row_number", "source_smiles")} | {"reason": retrieval_status}
                    )
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            if exclusions:
                exclusion_writer.write_table(pa.Table.from_pylist(exclusions, schema=exclusion_schema))
    finally:
        writer.close()
        exclusion_writer.close()
    return {
        "retrieval_status_counts": dict(sorted(status_counts.items())),
        "mechanism_family_record_counts": dict(sorted(family_counts.items())),
        "n_unique_source_smiles": len(lookup),
    }


def _write_evidence_catalog(
    organized_path: Path,
    destination: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    bridge_schema = pa.schema(
        [pa.field("record_id", pa.string()), pa.field("source_id", pa.string()),
         pa.field("source_row_number", pa.int64()), pa.field("molecule_id", pa.string()),
         pa.field("canonical_smiles", pa.large_string()), pa.field("mechanism_family", pa.string()),
         pa.field("retrieval_group", pa.string())]
    )
    bridge_writer = pq.ParquetWriter(destination / "record_bridge.parquet", bridge_schema, compression="zstd")
    endpoint_fields = {spec.source_id: spec.endpoint_field for spec in SOURCE_SPECS}
    try:
        parquet = pq.ParquetFile(organized_path)
        for batch in parquet.iter_batches(batch_size=SOURCE_BATCH_SIZE):
            bridges = []
            for row in batch.to_pylist():
                if row.get("retrieval_status") != "eligible":
                    continue
                bridge = {key: row.get(key) for key in bridge_schema.names}
                bridges.append(bridge)
                key = (str(row["molecule_id"]), str(row["retrieval_group"]))
                agg = aggregates.setdefault(
                    key,
                    {
                        "molecule_id": row["molecule_id"],
                        "canonical_smiles": row["canonical_smiles"],
                        "mechanism_family": row["mechanism_family"],
                        "retrieval_group": row["retrieval_group"],
                        "evidence_role": row["evidence_role"],
                        "source_ids": Counter(),
                        "endpoint_counts": Counter(),
                        "record_count": 0,
                        "confidence_sum": 0.0,
                        "confidence_count": 0,
                        "support_texts": [],
                        "examples": [],
                    },
                )
                agg["source_ids"][row["source_id"]] += 1
                endpoint = row.get(endpoint_fields[row["source_id"]]) or "unspecified"
                agg["endpoint_counts"][endpoint] += 1
                agg["record_count"] += 1
                if row.get("confidence") is not None:
                    agg["confidence_sum"] += float(row["confidence"])
                    agg["confidence_count"] += 1
                support = row.get("support_text")
                if support and support not in agg["support_texts"] and len(agg["support_texts"]) < 3:
                    agg["support_texts"].append(support[:1000])
                if len(agg["examples"]) < 3:
                    agg["examples"].append(
                        {
                            "source_id": row["source_id"],
                            "source_record_id": row.get("extraction_id") or row["record_id"],
                            "endpoint_type": endpoint,
                        }
                    )
            if bridges:
                bridge_writer.write_table(pa.Table.from_pylist(bridges, schema=bridge_schema))
    finally:
        bridge_writer.close()

    evidence_rows = [_catalog_row(value) for _, value in sorted(aggregates.items())]
    _write_jsonl(destination / "evidence.jsonl", evidence_rows)
    stats = {
        "n_evidence_rows": len(evidence_rows),
        "n_index_molecules": len({row["molecule_chembl_id"] for row in evidence_rows}),
        "group_counts": dict(sorted(Counter(row["group_id"] for row in evidence_rows).items())),
    }
    _write_json(destination / "summary.json", stats)
    return evidence_rows, stats


def _catalog_row(agg: Mapping[str, Any]) -> dict[str, Any]:
    endpoint_counts = agg["endpoint_counts"]
    confidence_count = int(agg["confidence_count"])
    row = {
        "molecule_chembl_id": agg["molecule_id"],
        "canonical_smiles": agg["canonical_smiles"],
        "assay_chembl_id": f"STARLING_CLINTOX_{agg['retrieval_group'].upper().replace('.', '_')}",
        "assay_tier": "Direct" if str(agg["retrieval_group"]).startswith("Direct.") else "Mechanism",
        "endpoint_group": agg["mechanism_family"],
        "group_id": agg["retrieval_group"],
        "standard_type": "; ".join(f"{key}={count}" for key, count in endpoint_counts.most_common(8)),
        "standard_relation": "",
        "standard_value": "",
        "standard_units": "",
        "activity_comment": f"Starling raw ClinTox summary over {agg['record_count']} source records",
        "assay_description": "\n\n---\n\n".join(agg["support_texts"])[:3000],
        "target_pref_name": agg["mechanism_family"],
        "confidence_score": round(float(agg["confidence_sum"]) / confidence_count, 4) if confidence_count else "",
        "evidence_source": "Starling ClinTox raw_v1",
        "evidence_role": agg["evidence_role"],
        "evidence_scope": {"source_ids": sorted(agg["source_ids"]), "source_record_counts": dict(agg["source_ids"])},
        "transferability": "not_assessed",
        "uncertainty": ["source_claims_not_semantically_normalized"],
        "source_record_count": agg["record_count"],
        "source_endpoint_counts": dict(endpoint_counts.most_common(20)),
        "source_record_examples": agg["examples"],
    }
    attach_minimal_evidence(row)
    row.pop("source_record_examples", None)
    return row


def _audit_existing_organized(path: Path) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    unique_smiles: set[str] = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=131_072,
        columns=["source_id", "source_smiles", "retrieval_status", "mechanism_family"],
    ):
        for row in batch.to_pylist():
            source_counts[str(row["source_id"])] += 1
            status_counts[str(row["retrieval_status"])] += 1
            family_counts[str(row["mechanism_family"])] += 1
            if row.get("source_smiles"):
                unique_smiles.add(str(row["source_smiles"]))
    return {
        "source_record_counts": dict(sorted(source_counts.items())),
        "retrieval_status_counts": dict(sorted(status_counts.items())),
        "mechanism_family_record_counts": dict(sorted(family_counts.items())),
        "n_unique_source_smiles": len(unique_smiles),
    }


def _write_index(
    evidence_rows: list[dict[str, Any]],
    destination: Path,
    *,
    index_version: str,
    workers: int,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    index = build_neighbor_index(
        evidence_rows,
        index_version=index_version,
        workers=workers,
        progress_every=10_000,
    )
    index["source"] = {
        "name": "starling_raw",
        "raw_version": RAW_VERSION,
        "global_identifier_used": False,
    }
    index_path = destination / "neighbor_index.pkl"
    with index_path.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "index_version": index_version,
        "n_evidence_rows": len(evidence_rows),
        "n_index_molecules": len(index["molecules"]),
        "groups": sorted(index["group_to_molecule_indices"]),
        "fingerprint": fingerprint_metadata(),
        "path": str(index_path),
    }
    _write_json(destination / "index_meta.json", meta)
    return meta


def _write_gold_qa_sample(source_path: Path, destination: Path) -> dict[str, Any]:
    candidates: dict[tuple[str, int], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    parquet = pq.ParquetFile(source_path)
    row_number = 0
    for batch in parquet.iter_batches(batch_size=SOURCE_BATCH_SIZE):
        for row in batch.to_pylist():
            if is_direct_gold_scope(row):
                identity = normalize_molecule_identity(str(row.get("SMILES") or ""))
                if identity.status == "ok" and identity.parent_smiles:
                    organ = str(row["organ_system"])
                    label = LABELS[str(row["effect_status"])]
                    sample_row = {
                        "source_row_number": row_number,
                        "record_id": f"{RAW_VERSION}:organ_specific_toxicity:{row_number:09d}",
                        "extraction_id": row.get("extraction_id"),
                        "pmid": row.get("pmid"),
                        "source_smiles": row.get("SMILES"),
                        "parent_smiles": identity.parent_smiles,
                        "organ_system": organ,
                        "Y": label,
                        "effect_status": row.get("effect_status"),
                        "toxicity_endpoint": row.get("toxicity_endpoint"),
                        "biological_system": row.get("biological_system"),
                        "exposure_regimen": row.get("exposure_regimen"),
                        "quantitative_result": row.get("quantitative_result"),
                        "qualifying_conditions": row.get("qualifying_conditions"),
                        "support_text": row.get("support_text"),
                        "review_status": "pending",
                    }
                    rank = hashlib.sha256(sample_row["record_id"].encode("utf-8")).hexdigest()
                    candidates[(organ, label)].append((rank, sample_row))
            row_number += 1
    selected = [
        row
        for key in sorted(candidates)
        for _, row in sorted(candidates[key], key=lambda item: item[0])[:20]
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(selected), destination, compression="zstd")
    return {
        "sampling_policy": "deterministic SHA-256 rank; up to 20 source records per organ_system x label stratum",
        "n_sample_rows": len(selected),
        "target_if_all_strata_available": 360,
        "stratum_counts": {
            f"{organ}:Y={label}": count
            for (organ, label), count in sorted(Counter((row["organ_system"], row["Y"]) for row in selected).items())
        },
        "path": str(destination),
    }


def _clean_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("record_id", pa.string()),
            pa.field("source_id", pa.string()),
            pa.field("source_version", pa.string()),
            pa.field("source_row_number", pa.int64()),
            pa.field("paragraph_idx", pa.int64()),
            pa.field("support_text", pa.large_string()),
            pa.field("confidence", pa.float64()),
            pa.field("needs_more_context", pa.bool_()),
            pa.field("pmid", pa.string()),
            pa.field("extraction_id", pa.string()),
            pa.field("source_smiles", pa.large_string()),
            *(pa.field(field, pa.large_string()) for field in UNION_SOURCE_FIELDS),
        ]
    )


def _source_readme(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# ClinTox Starling raw_v1",
        "",
        "This directory contains the exact six Parquet sources and extraction guides from the supplied archive.",
        "No upstream dataset revision, extraction-model version, mapping method, or license was supplied; see `SOURCE_MANIFEST.json`.",
        "The raw Parquets retain `global_identifier` for fidelity. Derived code never consults it and uses only the source `SMILES` field.",
        "",
        "| Source | Rows | Purpose |",
        "|---|---:|---|",
    ]
    for source in manifest["sources"]:
        lines.append(f"| `{source['source_id']}` | {source['n_rows']:,} | {source['title']} |")
    lines.extend(
        [
            "",
            "The derived pipeline performs only basic null/whitespace cleaning, source organization, molecule validation, evidence aggregation, and indexing. It does not perform v7 assay-transfer normalization.",
            "",
        ]
    )
    return "\n".join(lines)


def _benchmark_report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# ClinTox Human Toxicity candidate benchmark",
            "",
            "This candidate predicts explicit human clinical organ injury and is not equivalent to TDC/MoleculeNet CT_TOX clinical-trial failure.",
            "",
            f"- Binary molecular parents: {summary['n_binary_molecules']:,}",
            f"- Accepted source rows before structure normalization: {summary['n_source_rows_labeled_before_structure_normalization']:,}",
            f"- Parent label counts: `{json.dumps(summary['all_label_counts'], sort_keys=True)}`",
            "- Vote policy: one accepted source record per vote, 70% parent agreement, exact ties rejected.",
            "- Status: candidate pending deterministic source-record QA.",
            "",
        ]
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    importer = subparsers.add_parser("import-raw")
    importer.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    importer.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    importer.add_argument("--force", action="store_true")
    library = subparsers.add_parser("build-library")
    library.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    library.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    library.add_argument("--workers", type=int, default=1)
    resume = subparsers.add_parser("resume-library")
    resume.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    resume.add_argument("--workers", type=int, default=1)
    benchmark = subparsers.add_parser("build-benchmark")
    benchmark.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    benchmark.add_argument("--base-root", type=Path, default=DEFAULT_BASE_BENCHMARK_ROOT)
    benchmark.add_argument("--record-supported-root", type=Path, default=DEFAULT_RECORD_SUPPORTED_ROOT)
    benchmark.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    benchmark.add_argument("--max-rows", type=int, default=0)
    filtered = subparsers.add_parser("build-heldout-view")
    filtered.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    filtered.add_argument("--record-supported-root", type=Path, default=DEFAULT_RECORD_SUPPORTED_ROOT)
    filtered.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.action == "import-raw":
        result = import_archive(args.archive, raw_root=args.raw_root, force=args.force)
    elif args.action == "build-library":
        result = build_evidence_library(raw_root=args.raw_root, local_root=args.local_root, workers=args.workers)
    elif args.action == "resume-library":
        result = resume_evidence_library(local_root=args.local_root, workers=args.workers)
    elif args.action == "build-benchmark":
        result = build_candidate_benchmark(
            source_path=args.source_path,
            base_root=args.base_root,
            record_supported_root=args.record_supported_root,
            local_root=args.local_root,
            max_rows=args.max_rows,
        )
    else:
        result = build_heldout_filtered_view(
            local_root=args.local_root,
            record_supported_root=args.record_supported_root,
            workers=args.workers,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_LOCAL_ROOT",
    "DEFAULT_RAW_ROOT",
    "SOURCE_SPECS",
    "build_candidate_benchmark",
    "build_evidence_library",
    "build_heldout_filtered_view",
    "clean_text",
    "import_archive",
    "route_record",
    "resume_evidence_library",
]
