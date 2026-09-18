"""Gold-label adapters over the repaired, source-native Stage-1 universe."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds

from data.processing.gold_labels.benchmark_dataset import (
    LabelDecision,
    LabeledSourceRecord,
    accepted,
    has_reported_text,
    rejected,
    sha256_file,
)
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.experimental_meaningful_cns_access_benchmark_v4 import (
    CONTRACT_VERSION as BBB_CONTRACT_VERSION,
    SOURCE_REVISION as BBB_SOURCE_REVISION,
    label_record as label_bbb_record,
    load_label_decisions_from_rows as label_bbb_rows,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.canonical_source import (
    LOCAL_PARTITION_DIRECT,
    classify_local_record,
)
from data.processing.evidence_library.versions.v10.tasks.bioavailability_ma.canonical_source import (
    DIRECT_REPORT_TYPES,
)
from data.processing.evidence_library.versions.v10.tasks.bioavailability_ma.starling_benchmark import (
    is_human_context,
    label_bioavailability_value,
)


ADAPTER_VERSION = "stage1_repaired_gold_sources.v1"
SCIENTIFIC_PAYLOAD_VERSION = "physical_condition_review_payload.v1"
SCIENTIFIC_PAYLOAD_FIELDS = {
    "bbb_martins": (
        "source_id",
        "source_record_id",
        "pmid",
        "bbb_permeability_label",
        "endpoint_name",
        "measurement_text",
        "unit_text",
        "assay_model",
        "species",
        "qualifying_conditions",
        "support_text",
        "extra_details",
    ),
    "bioavailability_ma": (
        "source_id",
        "source_record_id",
        "pmid",
        "endpoint_name",
        "measurement_text",
        "unit_text",
        "study_context",
        "species_or_population",
        "oral_dose",
        "dose",
        "oral_exposure_mode",
        "bioavailability_report_type",
        "comparator_exposure",
        "comparator",
        "qualifying_conditions",
        "support_text",
        "extra_details",
    ),
}


def load_bbb_stage1_decisions(
    stage1_path: str | Path,
    *,
    max_rows: int = 0,
) -> tuple[Iterable[LabelDecision], dict[str, Any]]:
    path = Path(stage1_path)
    table = ds.dataset(path).to_table(
        filter=ds.field("source_id") == "direct_bbb"
    )
    rows = table.slice(0, max_rows).to_pylist() if max_rows else table.to_pylist()
    adapted = [
        {
            **row,
            "smiles": row.get("canonical_smiles"),
            "quant_metric": row.get("endpoint_name"),
            "quant_value": row.get("measurement_text"),
            "quant_units": row.get("unit_text"),
        }
        for row in rows
    ]
    decisions, metadata = label_bbb_rows(
        adapted,
        resolved_revision=BBB_SOURCE_REVISION,
    )
    metadata.update(_stage1_metadata(path, len(rows)))
    metadata["gold_contract"]["version"] = BBB_CONTRACT_VERSION
    return decisions, metadata


def load_oral_stage1_decisions(
    stage1_path: str | Path,
    *,
    max_rows: int = 0,
) -> tuple[Iterable[LabelDecision], dict[str, Any]]:
    path = Path(stage1_path)
    table = ds.dataset(path).to_table(
        filter=ds.field("source_id").isin(["hf_bioavailability", "oral_exposure"])
    )
    rows = table.slice(0, max_rows).to_pylist() if max_rows else table.to_pylist()
    metadata = {
        **_stage1_metadata(path, len(rows)),
        "gold_contract": {
            "version": "bioavailability_stage1_physical_voter.v1",
            "target": "human direct oral bioavailability under the reported condition",
            "threshold": "F >= 20%",
            "vote_unit": "retained_source_row",
            "cross_source_duplicate_policy": "strict reviewed duplicates were deleted in Stage 1",
        },
    }
    return (_label_oral_stage1_row(row) for row in rows), metadata


def label_bbb_stage1_row(
    row: Mapping[str, Any], *, allow_conditioned_context: bool = False
) -> LabelDecision:
    source_index = int(
        row.get("source_index")
        if row.get("source_index") is not None
        else int(row.get("source_row_number") or 1) - 1
    )
    adapted = {
        **dict(row),
        "smiles": row.get("canonical_smiles"),
        "quant_metric": row.get("endpoint_name"),
        "quant_value": row.get("measurement_text"),
        "quant_units": row.get("unit_text"),
    }
    label, method = label_bbb_record(
        adapted,
        source_index=source_index,
        allow_conditioned_context=allow_conditioned_context,
    )
    if label is None:
        return rejected(
            method,
            source_id=row.get("source_id"),
            source_record_id=row.get("source_record_id"),
            source_row_uid=row.get("source_row_uid"),
        )
    smiles = str(row.get("canonical_smiles") or "").strip()
    if not smiles:
        return rejected("missing_smiles", source_row_uid=row.get("source_row_uid"))
    return accepted(
        LabeledSourceRecord(
            smiles=smiles,
            label=label,
            source_id=str(row.get("source_id") or "direct_bbb"),
            source_record_id=str(row.get("source_record_id") or f"row:{source_index}"),
            pmid=str(row.get("pmid") or ""),
            label_method=method,
            raw_value=" | ".join(
                str(value)
                for value in (
                    row.get("bbb_permeability_label"),
                    row.get("endpoint_name"),
                    row.get("measurement_text"),
                    row.get("unit_text"),
                )
                if value not in (None, "")
            ),
            context=" | ".join(
                str(value)
                for value in (row.get("assay_model"), row.get("species"))
                if value not in (None, "")
            ),
            source_row_uid=str(row.get("source_row_uid") or ""),
        )
    )


def _label_oral_stage1_row(
    row: Mapping[str, Any], *, allow_conditioned_context: bool = False
) -> LabelDecision:
    source_id = str(row.get("source_id") or "")
    source_index = int(
        row.get("source_index")
        if row.get("source_index") is not None
        else int(row.get("source_row_number") or 1) - 1
    )
    source_record_id = str(row.get("source_record_id") or source_index)
    if source_id == "hf_bioavailability":
        report_type = str(row.get("bioavailability_report_type") or "").strip().lower()
        population = row.get("species_or_population")
        value = row.get("measurement_text")
    elif source_id == "oral_exposure":
        local_row = {
            **dict(row),
            "exposure_measure": row.get("endpoint_name"),
            "parameter_value": row.get("measurement_text"),
            "parameter_units": row.get("unit_text"),
        }
        partition, reason = classify_local_record(local_row)
        if partition != LOCAL_PARTITION_DIRECT:
            return rejected(
                reason,
                source_id=source_id,
                source_index=source_index,
                source_record_id=source_record_id,
            )
        report_type = "absolute"
        population = row.get("study_context")
        value = row.get("measurement_text")
    else:
        raise ValueError(f"unsupported Oral Stage-1 source: {source_id}")

    if report_type not in DIRECT_REPORT_TYPES:
        return rejected(
            "non_direct_oral_bioavailability_report_type",
            source_id=source_id,
            source_index=source_index,
            report_type=report_type,
        )
    if not is_human_context(population):
        return rejected(
            "nonhuman_or_unresolved_population",
            source_id=source_id,
            source_index=source_index,
            population=population,
        )
    if not allow_conditioned_context and has_reported_text(
        row.get("qualifying_conditions")
    ):
        return rejected(
            "interpretation_altering_qualifying_conditions",
            source_id=source_id,
            source_index=source_index,
            qualifying_conditions=row.get("qualifying_conditions"),
        )
    label, method = label_bioavailability_value(value)
    if label is None:
        return rejected(
            method,
            source_id=source_id,
            source_index=source_index,
            value=value,
        )
    smiles = str(row.get("canonical_smiles") or "").strip()
    if not smiles:
        return rejected(
            "missing_smiles",
            source_id=source_id,
            source_index=source_index,
        )
    return accepted(
        LabeledSourceRecord(
            smiles=smiles,
            label=label,
            source_id=source_id,
            source_record_id=source_record_id,
            pmid=str(row.get("pmid") or ""),
            label_method=f"stage1_physical:{method}",
            raw_value=str(value or ""),
            context=str(population or ""),
            source_row_uid=str(row.get("source_row_uid") or ""),
        )
    )


def scientific_payload_sha256(task: str, row: Mapping[str, Any]) -> str:
    """Hash condition/label semantics while permitting SMILES-only reparenting."""
    try:
        fields = SCIENTIFIC_PAYLOAD_FIELDS[task]
    except KeyError as exc:
        raise ValueError(f"unsupported repaired-gold task: {task}") from exc
    payload = {
        "version": SCIENTIFIC_PAYLOAD_VERSION,
        "task": task,
        "fields": {
            field: _json_value(row.get(field))
            for field in fields
        },
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


def _stage1_metadata(path: Path, selected_rows: int) -> dict[str, Any]:
    return {
        "stage1_source": {
            "adapter_version": ADAPTER_VERSION,
            "path": str(path),
            "sha256": sha256_file(path),
            "selected_rows": selected_rows,
        }
    }


__all__ = [
    "label_bbb_stage1_row",
    "load_bbb_stage1_decisions",
    "load_oral_stage1_decisions",
    "scientific_payload_sha256",
]
