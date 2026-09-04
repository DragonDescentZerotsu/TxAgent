"""TDC-compatible binary-label adapter for Starling direct skin reactions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.benchmark_dataset import (
    LabelDecision,
    LabeledSourceRecord,
    accepted,
    rejected,
    sha256_file,
)
from tools.chembl_tool.tasks.skin_reaction.canonical_starling_source import (
    DIRECT_PARTITION,
    classify_direct_source_record,
)

SOURCE_PATH = Path(
    "data/starling_data/skin_reaction/direct_skin_reaction/extractions.parquet"
)


def load_label_decisions(
    *,
    source_path: str | Path = SOURCE_PATH,
    max_rows: int = 0,
) -> tuple[Iterable[LabelDecision], dict[str, Any]]:
    import pandas as pd

    path = Path(source_path)
    frame = pd.read_parquet(path)
    if max_rows:
        frame = frame.head(max_rows)
    records = frame.to_dict(orient="records")
    metadata = {
        "path": str(path),
        "sha256": sha256_file(path),
        "tdc_compatibility": {
            "endpoint_scope": "skin sensitization/contact allergy, not irritation or generic local skin injury",
            "positive_label": "sensitizer/positive direct skin-sensitization outcome",
            "negative_label": "non-sensitizer/negative direct skin-sensitization outcome",
            "numeric_policy": "outcome_label is authoritative; incidence fields are provenance, not a new threshold",
        },
        "eligibility_contract": {
            "version": "skin_gold_record_eligibility.v3",
            "accepted": "measured final sensitization/contact-allergy outcomes with explicit binary labels",
            "excluded": [
                "prediction-only or in-silico records",
                "phototoxicity/photoallergy/photoirritation",
                "irritation or corrosion without a direct sensitization endpoint",
                "photo/light-dependent sensitization and photochemistry records",
                "non-contact severe cutaneous adverse reactions such as SJS/TEN/DRESS/Sweet syndrome",
                "mechanistic AOP assays without a final sensitization outcome",
                "integrated or defined-approach predictions",
            ],
        },
    }
    return (
        (
            _label_record(index, row, source_path=path)
            for index, row in enumerate(records)
        ),
        metadata,
    )


def label_record(record: Mapping[str, Any]) -> tuple[int | None, str]:
    semantic = classify_direct_source_record(record)
    if semantic.partition != DIRECT_PARTITION:
        return None, semantic.reason
    outcome = str(record.get("outcome_label") or "").strip().lower()
    if outcome == "positive":
        return 1, "explicit_positive_skin_sensitization_outcome"
    if outcome == "negative":
        return 0, "explicit_negative_skin_sensitization_outcome"
    if outcome == "inconclusive":
        return None, "inconclusive_outcome"
    return None, "missing_or_unknown_outcome"


def _label_record(
    index: int,
    row: Mapping[str, Any],
    *,
    source_path: Path = SOURCE_PATH,
) -> LabelDecision:
    source_id = f"local:{source_path}"
    label, method = label_record(row)
    if label is None:
        return rejected(
            method,
            source_id=source_id,
            source_index=index,
            outcome_label=row.get("outcome_label"),
            reaction_type=row.get("reaction_type"),
        )
    smiles = str(row.get("SMILES") or row.get("smiles") or "").strip()
    if not smiles:
        return rejected("missing_smiles", source_id=source_id, source_index=index)
    raw_parts = [str(row.get("outcome_label") or "")]
    if row.get("positive_count") not in (None, ""):
        raw_parts.append(f"positive_count={row.get('positive_count')}")
    if row.get("total_tested") not in (None, ""):
        raw_parts.append(f"total_tested={row.get('total_tested')}")
    return accepted(
        LabeledSourceRecord(
            smiles=smiles,
            label=label,
            source_id=source_id,
            source_record_id=str(row.get("extraction_id") or f"row:{index}"),
            pmid=str(row.get("pmid") or ""),
            label_method=method,
            raw_value=" | ".join(raw_parts),
            context=" | ".join(
                str(value)
                for value in (row.get("reaction_type"), row.get("assay_or_test"))
                if value not in (None, "")
            ),
        )
    )
