"""Candidate human organ-toxicity gold adapter for the ClinTox Starling source.

This is intentionally not a TDC/MoleculeNet CT_TOX reconstruction.  It maps
explicit human clinical organ-injury observations from source shard v2 to a
new, source-faithful binary task named ``ClinTox_Human_Toxicity``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
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


TASK_NAME = "ClinTox_Human_Toxicity"
ADAPTER_VERSION = "clintox_human_organ_toxicity.v1"
DEFAULT_SOURCE_PATH = Path(
    "data/starling_data/clintox/raw_v1/organ_specific_toxicity/extractions.parquet"
)
ALLOWED_ORGAN_SYSTEMS = frozenset(
    {
        "hepatic",
        "renal",
        "cardiac",
        "neurological",
        "hematological",
        "pulmonary",
        "gastrointestinal",
        "reproductive",
        "developmental",
    }
)
LABELS = {"no_injury_observed": 0, "injury_observed": 1}


def label_record(
    row: Mapping[str, Any],
    *,
    source_index: int,
    source_id: str = "starling_raw_v1:organ_specific_toxicity",
) -> LabelDecision:
    """Map one v2 source row to a vote or one precise rejection reason."""
    if _text(row.get("evidence_context")) != "human_clinical":
        return _reject("not_human_clinical", row, source_index, source_id)
    status = _text(row.get("effect_status"))
    if status not in LABELS:
        return _reject("missing_or_invalid_effect_status", row, source_index, source_id)
    organ = _text(row.get("organ_system"))
    if organ not in ALLOWED_ORGAN_SYSTEMS:
        return _reject("missing_or_invalid_organ_system", row, source_index, source_id)
    if has_reported_text(row.get("qualifying_conditions")):
        return _reject(
            "interpretation_altering_qualifying_conditions", row, source_index, source_id
        )
    if row.get("needs_more_context") is not False:
        return _reject("needs_more_context", row, source_index, source_id)
    for field, reason in (
        ("SMILES", "missing_source_smiles"),
        ("support_text", "missing_support_text"),
        ("pmid", "missing_pmid"),
        ("toxicity_endpoint", "missing_toxicity_endpoint"),
    ):
        if not has_reported_text(row.get(field)):
            return _reject(reason, row, source_index, source_id)

    record_id = _text(row.get("extraction_id")) or f"row:{source_index}"
    context = {
        "organ_system": organ,
        "toxicity_endpoint": _text(row.get("toxicity_endpoint")),
        "biological_system": _text(row.get("biological_system")),
        "exposure_regimen": _text(row.get("exposure_regimen")),
        "quantitative_result": _text(row.get("quantitative_result")),
    }
    return accepted(
        LabeledSourceRecord(
            smiles=_text(row.get("SMILES")),
            label=LABELS[status],
            source_id=source_id,
            source_record_id=record_id,
            pmid=_text(row.get("pmid")),
            label_method=f"{ADAPTER_VERSION}:explicit_effect_status",
            raw_value=status,
            context=json.dumps(context, ensure_ascii=False, sort_keys=True),
        )
    )


def load_label_decisions(
    *,
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    max_rows: int = 0,
    batch_size: int = 65_536,
) -> tuple[Iterable[LabelDecision], dict[str, Any]]:
    """Stream v2 decisions and return complete source/policy provenance."""
    path = Path(source_path)
    parquet = pq.ParquetFile(path)
    limit = max_rows if max_rows > 0 else parquet.metadata.num_rows

    def decisions() -> Iterable[LabelDecision]:
        source_index = 0
        columns = [
            "SMILES",
            "support_text",
            "organ_system",
            "toxicity_endpoint",
            "effect_status",
            "evidence_context",
            "biological_system",
            "exposure_regimen",
            "quantitative_result",
            "qualifying_conditions",
            "needs_more_context",
            "pmid",
            "extraction_id",
        ]
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                if source_index >= limit:
                    return
                yield label_record(row, source_index=source_index)
                source_index += 1

    metadata = {
        "source": {
            "path": str(path),
            "sha256": sha256_file(path),
            "source_shard": "v2 organ-specific toxicity",
            "rows_available": parquet.metadata.num_rows,
            "rows_requested": limit,
        },
        "task_definition": {
            "name": TASK_NAME,
            "positive": "explicit injury_observed in a human clinical context",
            "negative": "explicit no_injury_observed in a human clinical context",
            "organ_systems": sorted(ALLOWED_ORGAN_SYSTEMS),
            "not_equivalent_to": "TDC/MoleculeNet toxicity-related clinical-trial failure",
        },
        "adapter_version": ADAPTER_VERSION,
        "qualifying_conditions_policy": {
            "helper": "benchmark_dataset.has_reported_text",
            "empty_values": ["null", "blank", "nan", "none", "n/a", "na"],
            "explicitly_not_empty": ["-", "unspecified"],
        },
        "confidence_threshold": None,
    }
    return decisions(), metadata


def is_direct_gold_scope(row: Mapping[str, Any]) -> bool:
    """Return whether a row passes all semantic gates before RDKit resolution."""
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
    "ALLOWED_ORGAN_SYSTEMS",
    "DEFAULT_SOURCE_PATH",
    "LABELS",
    "TASK_NAME",
    "is_direct_gold_scope",
    "label_record",
    "load_label_decisions",
]
