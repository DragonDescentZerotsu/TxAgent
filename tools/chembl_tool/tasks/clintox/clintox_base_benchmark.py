"""Gold-candidate adapter for the ClinTox base human clinical source.

This is a source-faithful human clinical toxicity-present versus explicitly
toxicity-absent task. It is not a reconstruction of TDC/MoleculeNet CT_TOX.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from tools.chembl_tool.common.starling.benchmark_dataset import (
    LabelDecision,
    LabeledSourceRecord,
    accepted,
    has_reported_text,
    rejected,
    sha256_file,
)
from tools.chembl_tool.tasks.clintox.starling_source import (
    DEFAULT_DATA_ROOT,
    DIRECT_SOURCE_ID,
    EXPECTED_SOURCE_SHA256,
    SOURCE_RELEASE,
)

TASK_NAME = "ClinTox_Human_Toxicity"
ADAPTER_VERSION = "clintox_send_v2_human_toxicity.v2"
SOURCE_ID = f"{SOURCE_RELEASE}:{DIRECT_SOURCE_ID}"
DEFAULT_SOURCE_PATH = DEFAULT_DATA_ROOT / DIRECT_SOURCE_ID / "extractions.parquet"
ALLOWED_CATEGORIES = frozenset(
    {
        "cardiotoxicity",
        "hepatotoxicity",
        "nephrotoxicity",
        "neurotoxicity",
        "hematologic_toxicity",
        "gastrointestinal_toxicity",
        "dermatologic_toxicity",
        "ocular_toxicity",
        "metabolic_or_electrolyte_toxicity",
        "general_adverse_events",
        "toxicity_absent",
        "other",
    }
)
NEGATIVE_CATEGORY = "toxicity_absent"
SOURCE_COLUMNS = (
    "paragraph_idx",
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
    "confidence",
    "needs_more_context",
    "pmid",
    "extraction_id",
    "SMILES",
)


def label_record(
    row: Mapping[str, Any],
    *,
    source_index: int,
    source_id: str = SOURCE_ID,
) -> LabelDecision:
    """Map one source row to one vote or a precise rejection reason."""
    category = _text(row.get("toxicity_category"))
    if category not in ALLOWED_CATEGORIES:
        return _reject("missing_or_invalid_toxicity_category", row, source_index, source_id)
    if not has_reported_text(row.get("toxicity_outcome")):
        return _reject("missing_toxicity_outcome", row, source_index, source_id)
    if row.get("needs_more_context") is not False:
        return _reject("needs_more_context", row, source_index, source_id)
    for field, reason in (
        ("SMILES", "missing_source_smiles"),
        ("support_text", "missing_support_text"),
        ("pmid", "missing_pmid"),
    ):
        if not has_reported_text(row.get(field)):
            return _reject(reason, row, source_index, source_id)

    extraction_id = _text(row.get("extraction_id")) or "missing"
    context = {
        "toxicity_category": category,
        "toxicity_outcome": _text(row.get("toxicity_outcome")),
        "outcome_measure": _text(row.get("outcome_measure")),
        "clinical_context": _text(row.get("clinical_context")),
        "dose_or_exposure": _text(row.get("dose_or_exposure")),
        "fda_approval_status": _text(row.get("fda_approval_status")),
        "approved_indication": _text(row.get("approved_indication")),
    }
    return accepted(
        LabeledSourceRecord(
            smiles=_text(row.get("SMILES")),
            label=0 if category == NEGATIVE_CATEGORY else 1,
            source_id=source_id,
            source_record_id=f"row:{source_index:09d}:{extraction_id}",
            pmid=_text(row.get("pmid")),
            label_method=f"{ADAPTER_VERSION}:controlled_toxicity_category",
            raw_value=category,
            context=json.dumps(context, ensure_ascii=False, sort_keys=True),
        )
    )


def load_label_decisions(
    *,
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    max_rows: int = 0,
    batch_size: int = 65_536,
) -> tuple[Iterable[LabelDecision], dict[str, Any]]:
    """Stream decisions and return complete source and policy provenance."""
    path = Path(source_path)
    source_digest = sha256_file(path)
    expected_digest = EXPECTED_SOURCE_SHA256[DIRECT_SOURCE_ID]
    if source_digest != expected_digest:
        raise ValueError(
            "ClinTox clinical source digest mismatch: "
            f"expected {expected_digest}, found {source_digest}"
        )
    parquet = pq.ParquetFile(path)
    missing = sorted(set(SOURCE_COLUMNS) - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"ClinTox base source is missing required columns: {missing}")
    limit = min(max_rows, parquet.metadata.num_rows) if max_rows > 0 else parquet.metadata.num_rows

    def decisions() -> Iterable[LabelDecision]:
        source_index = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=list(SOURCE_COLUMNS)):
            for row in batch.to_pylist():
                if source_index >= limit:
                    return
                yield label_record(row, source_index=source_index)
                source_index += 1

    metadata = {
        "source": {
            "path": str(path),
            "sha256": source_digest,
            "source_release": SOURCE_RELEASE,
            "source_shard": "clintox_send_v2 human clinical toxicity and FDA extraction",
            "rows_available": parquet.metadata.num_rows,
            "rows_requested": limit,
        },
        "task_definition": {
            "name": TASK_NAME,
            "positive": "a declared human clinical toxicity category other than toxicity_absent",
            "negative": "the declared toxicity_category is exactly toxicity_absent",
            "categories": sorted(ALLOWED_CATEGORIES),
            "not_equivalent_to": "TDC/MoleculeNet toxicity-related clinical-trial failure",
        },
        "adapter_version": ADAPTER_VERSION,
        "label_policy": {
            "source_field": "toxicity_category",
            "negative_category": NEGATIVE_CATEGORY,
            "positive_categories": sorted(ALLOWED_CATEGORIES - {NEGATIVE_CATEGORY}),
            "off_schema_values": "reject_without_rewrite",
            "fda_status_used_for_label": False,
            "confidence_threshold": None,
        },
        "qualifying_conditions_policy": {
            "status": "unavailable_in_source_schema",
            "effect": "no qualifying-condition exclusion can be applied",
            "mitigation": "candidate_pending_qa with deterministic category-stratified review",
        },
    }
    return decisions(), metadata


def is_direct_gold_scope(row: Mapping[str, Any]) -> bool:
    return label_record(row, source_index=0).record is not None


def _reject(
    reason: str,
    row: Mapping[str, Any],
    source_index: int,
    source_id: str,
) -> LabelDecision:
    return rejected(
        reason,
        source_id=source_id,
        source_index=source_index,
        extraction_id=_text(row.get("extraction_id")),
        pmid=_text(row.get("pmid")),
    )


def _text(value: Any) -> str:
    if not has_reported_text(value):
        return ""
    return str(value).strip()


__all__ = [
    "ADAPTER_VERSION",
    "ALLOWED_CATEGORIES",
    "DEFAULT_SOURCE_PATH",
    "NEGATIVE_CATEGORY",
    "SOURCE_COLUMNS",
    "SOURCE_ID",
    "TASK_NAME",
    "is_direct_gold_scope",
    "label_record",
    "load_label_decisions",
]
