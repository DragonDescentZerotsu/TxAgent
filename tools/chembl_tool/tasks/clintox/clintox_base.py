"""Raw-first ClinTox-base evidence and gold-candidate build workflow.

The workflow preserves the tracked source, applies only basic null/whitespace
cleaning, and indexes only rows accepted by the gold adapter. It does not run
assay-transfer normalization or infer labels from prose, FDA status, or values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

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
from tools.chembl_tool.tasks.clintox.clintox_base_benchmark import (
    DEFAULT_SOURCE_PATH,
    SOURCE_ID,
    TASK_NAME,
    label_record,
    load_label_decisions,
)
from tools.chembl_tool.tasks.clintox.starling_source import (
    DIRECT_SOURCE_ID,
    EXPECTED_SOURCE_SHA256,
    SOURCE_RELEASE,
)

RAW_VERSION = SOURCE_RELEASE
LIBRARY_VERSION = "clintox.send_v2.direct_evidence_library.v2"
SOURCE_SHA256 = EXPECTED_SOURCE_SHA256[DIRECT_SOURCE_ID]
DEFAULT_LOCAL_ROOT = Path(
    "outputs/chembl_tool/tasks/clintox/evidence_library/clintox_send_v2_direct"
)
DEFAULT_BASE_BENCHMARK_ROOT = Path("data/processed_starling")
DEFAULT_RECORD_SUPPORTED_ROOT = Path("data/processed_starling_record_supported_v2")
GOLD_QA_SUMMARY = Path(
    "tools/chembl_tool/tasks/clintox/data_processing/gold_qa_v1/summary.json"
)
SOURCE_BATCH_SIZE = 32_768
NULL_TEXT_VALUES = frozenset({"", "nan", "none", "null", "n/a", "na"})
DIRECT_GROUP = "Direct.human_clinical_toxicity"

TEXT_FIELDS = (
    "support_text",
    "molecule_name",
    "toxicity_outcome",
    "toxicity_category",
    "outcome_measure",
    "clinical_context",
    "dose_or_exposure",
    "fda_approval_status",
    "approved_indication",
    "extra_details",
    "pmid",
    "extraction_id",
)


def build_evidence_library(
    *,
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    workers: int = 1,
) -> dict[str, Any]:
    """Build cleaned, organized, direct-catalog, index, and audit stages."""
    source = Path(source_path)
    _verify_source(source)
    root = Path(local_root)
    clean_path = root / "01_cleaned" / "records.parquet"
    organized_path = root / "02_organized" / "records.parquet"
    exclusions_path = root / "02_organized" / "excluded_records.parquet"
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    organized_path.parent.mkdir(parents=True, exist_ok=True)

    cleaning = _write_cleaned(source, clean_path)
    structure_lookup = _build_structure_lookup(clean_path)
    organization = _write_organized(
        clean_path, organized_path, exclusions_path, structure_lookup
    )
    evidence_rows, catalog = _write_evidence_catalog(
        organized_path, root / "03_evidence_catalog"
    )
    index = _write_index(
        evidence_rows,
        root / "04_neighbor_index",
        index_version=LIBRARY_VERSION,
        workers=workers,
    )
    summary = {
        "schema_version": "clintox_direct_evidence_library.v2",
        "raw_version": RAW_VERSION,
        "library_version": LIBRARY_VERSION,
        "source_path": str(source),
        "source_sha256": SOURCE_SHA256,
        "identity_normalizer_version": IDENTITY_NORMALIZER_VERSION,
        "cleaning_policy": {
            "operations": [
                "trim surrounding whitespace",
                "collapse internal whitespace",
                "map shared null sentinels to null",
            ],
            "null_sentinels_case_insensitive": sorted(NULL_TEXT_VALUES),
            "semantic_normalization": False,
            "assay_transfer_normalization": False,
            "raw_source_modified": False,
        },
        "qualifying_conditions_policy": {
            "status": "unavailable_in_source_schema",
            "effect": "no qualifying-condition exclusion can be applied",
        },
        "evidence_scope": "gold-eligible direct records only",
        "cleaning": cleaning,
        "organization": organization,
        "catalog": catalog,
        "index": index,
        "stages": [
            "01_cleaned",
            "02_organized",
            "03_evidence_catalog",
            "04_neighbor_index",
            "05_audits",
            "06_record_supported_v2_scaffold_view",
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
    """Build 70%-agreement labels, record-supported v2 split, and QA sample."""
    source = Path(source_path)
    _verify_source(source)
    decisions, metadata = load_label_decisions(source_path=source, max_rows=max_rows)
    base_summary = build_benchmark_dataset(
        task_name=TASK_NAME,
        decisions=decisions,
        source_metadata=metadata,
        output_dir=Path(base_root) / TASK_NAME,
    )
    final_summary = build_record_supported_task(
        TASK_NAME,
        source_root=Path(base_root),
        output_root=Path(record_supported_root),
    )

    audit_root = Path(local_root) / "05_audits"
    audit_root.mkdir(parents=True, exist_ok=True)
    sample_path = audit_root / "gold_qa_sample.parquet"
    _write_gold_qa_sample(source, sample_path)
    gold_qa = json.loads(GOLD_QA_SUMMARY.read_text(encoding="utf-8"))
    if sha256_file(sample_path) != gold_qa["sample"]["sha256"]:
        raise ValueError("ClinTox gold QA sample does not match the frozen review")
    _write_json(audit_root / "gold_qa_status.json", gold_qa)

    candidate = {
        "status": "candidate_pending_qa",
        "active_source_lineage": RAW_VERSION,
        "replacement_scope": "ClinTox source, gold, evidence, and index data pipeline",
        "prior_experiment_compatibility": (
            "incompatible: results built from earlier ClinTox source hashes are "
            "historical only and must not be merged with this candidate"
        ),
        "promotion_policy": (
            "The frozen manual audit failed. Do not add this candidate to default "
            "paper matrices until a source-wide claim policy is implemented, the "
            "gold is rebuilt, and every row in a new frozen sample passes."
        ),
        "task_definition": metadata["task_definition"],
        "missing_semantic_gate": metadata["qualifying_conditions_policy"],
        "gold_qa_gate": {
            key: gold_qa[key]
            for key in (
                "audit_version",
                "status",
                "promotion_gate_passed",
                "n_reviewed",
                "n_nonpass",
                "nonpass_fraction",
                "status_counts",
                "label_status_counts",
                "failure_reason_counts",
                "required_source_correction",
                "conclusion",
                "failure_action",
            )
        },
        "base_summary": base_summary,
        "record_supported_v2_summary": final_summary,
    }
    task_root = Path(record_supported_root) / TASK_NAME
    _write_json(task_root / "CANDIDATE_STATUS.json", candidate)

    base_task_root = Path(base_root) / TASK_NAME
    legacy_report = base_task_root / "report_zh.md"
    if legacy_report.exists():
        legacy_report.unlink()
    (base_task_root / "REPORT.md").write_text(
        _benchmark_report(base_summary, gold_qa=gold_qa), encoding="utf-8"
    )
    return candidate


def build_heldout_filtered_view(
    *,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    record_supported_root: str | Path = DEFAULT_RECORD_SUPPORTED_ROOT,
    workers: int = 1,
) -> dict[str, Any]:
    """Remove valid/test parents from the direct-only evidence catalog."""
    root = Path(local_root)
    split_root = Path(record_supported_root) / TASK_NAME / "scaffold"
    heldout_path = split_root / "heldout_molecule_labels.jsonl"
    heldout_keys = {
        json.loads(line)["molecule_identity_key"]
        for line in heldout_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    retained: list[dict[str, Any]] = []
    excluded = 0
    catalog_path = root / "03_evidence_catalog" / "evidence.jsonl"
    for line in catalog_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        identity = normalize_molecule_identity(row["canonical_smiles"])
        key = identity.parent_inchi_key or identity.parent_smiles
        if key in heldout_keys:
            excluded += 1
        else:
            retained.append(row)

    stage = root / "06_record_supported_v2_scaffold_view"
    stage.mkdir(parents=True, exist_ok=True)
    _write_jsonl(stage / "evidence.jsonl", retained)
    index = _write_index(
        retained,
        stage,
        index_version=f"{LIBRARY_VERSION}.record_supported_v2_scaffold_heldout_filtered",
        workers=workers,
    )
    retained_keys = {
        normalize_molecule_identity(row["canonical_smiles"]).parent_inchi_key
        or normalize_molecule_identity(row["canonical_smiles"]).parent_smiles
        for row in retained
    }
    overlap = len(retained_keys & heldout_keys)
    summary = {
        "schema_version": "clintox_direct_record_supported_v2_scaffold_view.v2",
        "benchmark_task": TASK_NAME,
        "heldout_contract": str(heldout_path),
        "n_heldout_parents": len(heldout_keys),
        "n_full_evidence_rows": len(retained) + excluded,
        "n_excluded_evidence_rows": excluded,
        "n_retained_evidence_rows": len(retained),
        "exclusion_scope": "the sole direct gold source",
        "heldout_direct_source_parent_overlap": overlap,
        "query_time_identity_policy_required": "parent_disjoint",
        "index": index,
    }
    if overlap:
        raise RuntimeError(f"held-out direct-source evidence leakage detected: {overlap}")
    _write_json(stage / "summary.json", summary)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["record_supported_v2_scaffold_view"] = summary
    _write_json(manifest_path, manifest)
    return summary


def clean_text(value: Any) -> str | None:
    """Apply basic null and whitespace cleaning without semantic rewriting."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return None if text.lower() in NULL_TEXT_VALUES else text


def _verify_source(path: Path) -> None:
    digest = sha256_file(path)
    if digest != SOURCE_SHA256:
        raise ValueError(f"unexpected ClinTox base source SHA-256: {digest}")


def _write_cleaned(source: Path, destination: Path) -> dict[str, Any]:
    writer = pq.ParquetWriter(destination, _clean_schema(), compression="zstd")
    parquet = pq.ParquetFile(source)
    written = 0
    try:
        for batch in parquet.iter_batches(batch_size=SOURCE_BATCH_SIZE):
            output = []
            for raw in batch.to_pylist():
                row_number = written + len(output)
                row = {
                    "record_id": f"{RAW_VERSION}:{row_number:09d}",
                    "source_id": SOURCE_ID,
                    "source_version": RAW_VERSION,
                    "source_row_number": row_number,
                    "paragraph_idx": raw.get("paragraph_idx"),
                    **{field: clean_text(raw.get(field)) for field in TEXT_FIELDS},
                    "confidence": raw.get("confidence"),
                    "needs_more_context": raw.get("needs_more_context"),
                    "source_smiles": clean_text(raw.get("SMILES")),
                }
                output.append(row)
            writer.write_table(pa.Table.from_pylist(output, schema=_clean_schema()))
            written += len(output)
    finally:
        writer.close()
    return {"n_rows": written, "path": str(destination), "source_sha256": SOURCE_SHA256}


def _build_structure_lookup(path: Path) -> dict[str, dict[str, Any]]:
    smiles_values: set[str] = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["source_smiles"], batch_size=131_072):
        smiles_values.update(value for value in batch.column(0).to_pylist() if value)
    lookup: dict[str, dict[str, Any]] = {}
    for smiles in sorted(smiles_values):
        identity = normalize_molecule_identity(smiles)
        status = identity.status if identity.status != "ok" else (
            "ok" if identity.parent_smiles else "unresolved_parent"
        )
        lookup[smiles] = {
            "structure_status": status,
            "canonical_smiles": identity.canonical_smiles or None,
            "parent_smiles": identity.parent_smiles or None,
            "parent_inchi_key": identity.parent_inchi_key or None,
            "molecule_id": (
                starling_molecule_id(identity.canonical_smiles)
                if identity.canonical_smiles
                else None
            ),
        }
    return lookup


def _write_organized(
    clean_path: Path,
    destination: Path,
    exclusions_path: Path,
    lookup: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    schema = pa.schema(
        [
            *list(_clean_schema()),
            pa.field("structure_status", pa.string()),
            pa.field("canonical_smiles", pa.large_string()),
            pa.field("parent_smiles", pa.large_string()),
            pa.field("parent_inchi_key", pa.string()),
            pa.field("molecule_id", pa.string()),
            pa.field("Y", pa.int8()),
            pa.field("gold_status", pa.string()),
            pa.field("gold_rejection_reason", pa.string()),
            pa.field("retrieval_group", pa.string()),
            pa.field("evidence_role", pa.string()),
        ]
    )
    exclusion_schema = pa.schema(
        [
            pa.field("record_id", pa.string()),
            pa.field("source_row_number", pa.int64()),
            pa.field("source_smiles", pa.large_string()),
            pa.field("reason", pa.string()),
        ]
    )
    writer = pq.ParquetWriter(destination, schema, compression="zstd")
    exclusion_writer = pq.ParquetWriter(
        exclusions_path, exclusion_schema, compression="zstd"
    )
    status_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    try:
        parquet = pq.ParquetFile(clean_path)
        for batch in parquet.iter_batches(batch_size=SOURCE_BATCH_SIZE):
            rows = []
            exclusions = []
            for row in batch.to_pylist():
                adapter_row = {**row, "SMILES": row.get("source_smiles")}
                decision = label_record(
                    adapter_row, source_index=int(row["source_row_number"])
                )
                identity = dict(lookup.get(row.get("source_smiles") or "", {}))
                structure_status = str(identity.get("structure_status") or "missing_smiles")
                if decision.record is None:
                    status = "rejected"
                    reason = decision.reason
                    label = None
                elif structure_status != "ok":
                    status = "rejected"
                    reason = "invalid_or_unresolved_smiles"
                    label = decision.record.label
                else:
                    status = "accepted"
                    reason = ""
                    label = decision.record.label
                    category_counts[str(row["toxicity_category"])] += 1
                row.update(identity)
                row.update(
                    {
                        "structure_status": structure_status,
                        "Y": label,
                        "gold_status": status,
                        "gold_rejection_reason": reason,
                        "retrieval_group": DIRECT_GROUP if status == "accepted" else "",
                        "evidence_role": "direct_outcome" if status == "accepted" else "",
                    }
                )
                rows.append(row)
                status_counts[reason or "accepted"] += 1
                if status != "accepted":
                    exclusions.append(
                        {
                            "record_id": row["record_id"],
                            "source_row_number": row["source_row_number"],
                            "source_smiles": row.get("source_smiles"),
                            "reason": reason,
                        }
                    )
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            if exclusions:
                exclusion_writer.write_table(
                    pa.Table.from_pylist(exclusions, schema=exclusion_schema)
                )
    finally:
        writer.close()
        exclusion_writer.close()
    return {
        "n_unique_source_smiles": len(lookup),
        "gold_status_counts": dict(sorted(status_counts.items())),
        "accepted_category_counts": dict(sorted(category_counts.items())),
        "qualifying_conditions_status": "unavailable_in_source_schema",
    }


def _write_evidence_catalog(
    organized_path: Path,
    destination: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    aggregates: dict[str, dict[str, Any]] = {}
    bridge_schema = pa.schema(
        [
            pa.field("record_id", pa.string()),
            pa.field("source_row_number", pa.int64()),
            pa.field("molecule_id", pa.string()),
            pa.field("canonical_smiles", pa.large_string()),
            pa.field("parent_inchi_key", pa.string()),
            pa.field("toxicity_category", pa.string()),
            pa.field("Y", pa.int8()),
            pa.field("retrieval_group", pa.string()),
        ]
    )
    bridge_writer = pq.ParquetWriter(
        destination / "record_bridge.parquet", bridge_schema, compression="zstd"
    )
    try:
        parquet = pq.ParquetFile(organized_path)
        for batch in parquet.iter_batches(batch_size=SOURCE_BATCH_SIZE):
            bridges = []
            for row in batch.to_pylist():
                if row.get("gold_status") != "accepted":
                    continue
                bridges.append({key: row.get(key) for key in bridge_schema.names})
                key = str(row["molecule_id"])
                agg = aggregates.setdefault(
                    key,
                    {
                        "molecule_id": row["molecule_id"],
                        "canonical_smiles": row["canonical_smiles"],
                        "categories": Counter(),
                        "labels": Counter(),
                        "record_count": 0,
                        "confidence_sum": 0.0,
                        "confidence_count": 0,
                        "support_texts": [],
                        "molecule_names": [],
                    },
                )
                agg["categories"][row["toxicity_category"]] += 1
                agg["labels"][str(row["Y"])] += 1
                agg["record_count"] += 1
                if row.get("confidence") is not None:
                    agg["confidence_sum"] += float(row["confidence"])
                    agg["confidence_count"] += 1
                support = row.get("support_text")
                if support and support not in agg["support_texts"] and len(agg["support_texts"]) < 3:
                    agg["support_texts"].append(support[:1000])
                name = row.get("molecule_name")
                if name and name not in agg["molecule_names"] and len(agg["molecule_names"]) < 10:
                    agg["molecule_names"].append(name)
            if bridges:
                bridge_writer.write_table(
                    pa.Table.from_pylist(bridges, schema=bridge_schema)
                )
    finally:
        bridge_writer.close()

    evidence_rows = [_catalog_row(value) for _, value in sorted(aggregates.items())]
    _write_jsonl(destination / "evidence.jsonl", evidence_rows)
    stats = {
        "n_evidence_rows": len(evidence_rows),
        "n_index_molecules": len(evidence_rows),
        "group_counts": {DIRECT_GROUP: len(evidence_rows)},
        "source_record_count": sum(row["source_record_count"] for row in evidence_rows),
    }
    _write_json(destination / "summary.json", stats)
    return evidence_rows, stats


def _catalog_row(agg: Mapping[str, Any]) -> dict[str, Any]:
    categories = agg["categories"]
    confidence_count = int(agg["confidence_count"])
    row = {
        "molecule_chembl_id": agg["molecule_id"],
        "canonical_smiles": agg["canonical_smiles"],
        "source_molecule_names": agg["molecule_names"],
        "assay_chembl_id": "STARLING_CLINTOX_BASE_HUMAN_CLINICAL_TOXICITY",
        "assay_tier": "Direct",
        "endpoint_group": "clinical_human_safety",
        "group_id": DIRECT_GROUP,
        "standard_type": "; ".join(
            f"{key}={count}" for key, count in categories.most_common(12)
        ),
        "standard_relation": "",
        "standard_value": "",
        "standard_units": "",
        "activity_comment": (
            f"ClinTox base summary over {agg['record_count']} accepted source records"
        ),
        "assay_description": "\n\n---\n\n".join(agg["support_texts"])[:3000],
        "target_pref_name": "human clinical toxicity",
        "confidence_score": (
            round(float(agg["confidence_sum"]) / confidence_count, 4)
            if confidence_count
            else ""
        ),
        "evidence_source": "Starling ClinTox base v1",
        "evidence_role": "direct_outcome",
        "evidence_scope": {
            "source_id": SOURCE_ID,
            "toxicity_category_counts": dict(categories),
            "label_counts": dict(agg["labels"]),
        },
        "transferability": "not_assessed",
        "uncertainty": [
            "qualifying_conditions_unavailable_in_source_schema",
            "candidate_pending_source_record_qa",
        ],
        "source_record_count": agg["record_count"],
    }
    attach_minimal_evidence(row)
    return row


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
        "name": "clintox_send_v2_direct",
        "raw_version": RAW_VERSION,
        "evidence_scope": "gold-eligible direct records only",
    }
    index_path = destination / "neighbor_index.pkl"
    with index_path.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "index_version": index_version,
        "n_evidence_rows": len(evidence_rows),
        "n_index_molecules": len(index["molecules"]),
        "groups": sorted(index["group_to_molecule_indices"]),
        "fingerprint": fingerprint_metadata(),
        "path": str(index_path),
    }
    _write_json(destination / "index_meta.json", metadata)
    return metadata


def _write_gold_qa_sample(source: Path, destination: Path) -> dict[str, Any]:
    candidates: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    parquet = pq.ParquetFile(source)
    row_number = 0
    for batch in parquet.iter_batches(batch_size=SOURCE_BATCH_SIZE):
        for row in batch.to_pylist():
            decision = label_record(row, source_index=row_number)
            if decision.record is not None:
                identity = normalize_molecule_identity(decision.record.smiles)
                if identity.status == "ok" and identity.parent_smiles:
                    category = str(row["toxicity_category"])
                    sample_row = {
                        "source_row_number": row_number,
                        "record_id": f"{RAW_VERSION}:{row_number:09d}",
                        "extraction_id": row.get("extraction_id"),
                        "pmid": row.get("pmid"),
                        "source_smiles": row.get("SMILES"),
                        "parent_smiles": identity.parent_smiles,
                        "molecule_name": row.get("molecule_name"),
                        "toxicity_category": category,
                        "Y": decision.record.label,
                        "toxicity_outcome": row.get("toxicity_outcome"),
                        "outcome_measure": row.get("outcome_measure"),
                        "clinical_context": row.get("clinical_context"),
                        "dose_or_exposure": row.get("dose_or_exposure"),
                        "fda_approval_status": row.get("fda_approval_status"),
                        "support_text": row.get("support_text"),
                        "qualifying_conditions_source_status": "column_unavailable",
                        "review_status": "pending",
                    }
                    rank = hashlib.sha256(sample_row["record_id"].encode()).hexdigest()
                    candidates[category].append((rank, sample_row))
            row_number += 1
    selected = [
        row
        for category in sorted(candidates)
        for _, row in sorted(candidates[category], key=lambda item: item[0])[:30]
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(selected), destination, compression="zstd")
    return {
        "sampling_policy": "deterministic SHA-256 rank; up to 30 accepted source records per declared toxicity_category",
        "n_sample_rows": len(selected),
        "target_if_all_categories_available": 360,
        "stratum_counts": dict(sorted(Counter(row["toxicity_category"] for row in selected).items())),
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
            *(pa.field(field, pa.large_string()) for field in TEXT_FIELDS),
            pa.field("confidence", pa.float64()),
            pa.field("needs_more_context", pa.bool_()),
            pa.field("source_smiles", pa.large_string()),
        ]
    )


def _benchmark_report(
    summary: Mapping[str, Any], *, gold_qa: Mapping[str, Any]
) -> str:
    return "\n".join(
        [
            "# ClinTox Human Toxicity candidate benchmark",
            "",
            "This active candidate uses the tracked ClinTox base human clinical source. It is not equivalent to TDC/MoleculeNet CT_TOX clinical-trial failure.",
            "",
            f"- Binary molecular parents: {summary['n_binary_molecules']:,}",
            f"- Accepted source rows before structure normalization: {summary['n_source_rows_labeled_before_structure_normalization']:,}",
            f"- Parent label counts: `{json.dumps(summary['all_label_counts'], sort_keys=True)}`",
            "- Vote policy: one accepted source row per vote, 70% parent agreement, exact ties rejected.",
            "- `qualifying_conditions` is unavailable in the delivered source schema; it was not treated as empty.",
            (
                "- Frozen source QA: "
                f"{gold_qa['status']} "
                f"({gold_qa['n_nonpass']}/{gold_qa['n_reviewed']} non-passing rows)."
            ),
            "- Status: candidate pending source-wide claim-policy revision and a new frozen QA sample.",
            "",
        ]
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    library = subparsers.add_parser("build-library")
    library.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    library.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    library.add_argument("--workers", type=int, default=1)
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
    if args.action == "build-library":
        result = build_evidence_library(
            source_path=args.source_path,
            local_root=args.local_root,
            workers=args.workers,
        )
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
    "DIRECT_GROUP",
    "LIBRARY_VERSION",
    "RAW_VERSION",
    "build_candidate_benchmark",
    "build_evidence_library",
    "build_heldout_filtered_view",
    "clean_text",
]
