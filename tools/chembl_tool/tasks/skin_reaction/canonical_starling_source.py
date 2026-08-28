"""Canonical, mutually exclusive Starling evidence partitions for skin sensitization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Mapping

import pyarrow.parquet as pq

from tools.chembl_tool.tasks.skin_reaction.starling_benchmark import (
    is_tdc_skin_sensitization_scope,
)


CANONICAL_VERSION = "skin_sensitization_direct_aop.v3"
RAW_DIRECT_PATH = Path(
    "data/starling_data/skin_reaction/direct_skin_reaction/extractions.parquet"
)
RAW_AOP_PATH = Path(
    "data/starling_data/skin_reaction/sensitization_aop/extractions.parquet"
)
CANONICAL_SOURCE_DIR = Path(
    "data/starling_data/skin_reaction/canonical_sensitization_v3"
)
DIRECT_RECORDS_PATH = CANONICAL_SOURCE_DIR / "direct_records.parquet"
AOP_RECORDS_PATH = CANONICAL_SOURCE_DIR / "aop_records.parquet"
PARTITION_AUDIT_PATH = CANONICAL_SOURCE_DIR / "partition_audit.parquet"
DEDUP_AUDIT_PATH = CANONICAL_SOURCE_DIR / "dedup_audit.parquet"
MANIFEST_PATH = CANONICAL_SOURCE_DIR / "manifest.json"

DIRECT_PARTITION = "direct"
AOP_PARTITION = "aop"
REJECT_PARTITION = "reject"

AOP_EVENTS = frozenset(
    {
        "MIE_protein_binding",
        "KE2_keratinocyte_activation",
        "KE3_dendritic_cell_activation",
        "KE4_T_cell_activation",
    }
)
CANONICAL_SOURCE_IDS = frozenset({"direct_skin_reaction", "sensitization_aop"})


@dataclass(frozen=True)
class PartitionDecision:
    partition: str
    reason: str
    aop_event: str = ""


def partition_record_id(record: Mapping[str, Any]) -> str:
    source_id = str(record.get("source_id") or "")
    if source_id not in CANONICAL_SOURCE_IDS:
        return ""
    return f"{source_id}:{int(record.get('source_row_number') or 0) - 1}"


def partition_for_record(record: Mapping[str, Any]) -> PartitionDecision | None:
    """Return the frozen canonical partition for either acquisition source."""
    record_id = partition_record_id(record)
    if not record_id:
        return None
    try:
        return load_partition_audit()[record_id]
    except KeyError as exc:
        raise ValueError(f"Skin row lacks canonical partition: {record_id}") from exc


@lru_cache(maxsize=1)
def load_partition_audit() -> dict[str, PartitionDecision]:
    table = pq.read_table(
        PARTITION_AUDIT_PATH,
        columns=[
            "source_record_id",
            "partition",
            "partition_reason",
            "canonical_aop_event",
        ],
    )
    output: dict[str, PartitionDecision] = {}
    for row in table.to_pylist():
        record_id = str(row["source_record_id"])
        partition = str(row["partition"])
        if record_id in output or partition not in {
            DIRECT_PARTITION,
            AOP_PARTITION,
            REJECT_PARTITION,
        }:
            raise ValueError(f"invalid canonical Skin partition row: {record_id}")
        output[record_id] = PartitionDecision(
            partition,
            str(row.get("partition_reason") or ""),
            str(row.get("canonical_aop_event") or ""),
        )
    return output


_PHOTO_RE = re.compile(
    r"photo(?:toxic|irrit|allerg|contact|sensiti|safety)|photopatch|berloque|"
    r"\buva\b|\buvb\b|ultraviolet|light[- ]dependent",
    re.IGNORECASE,
)
_IRRITATION_RE = re.compile(
    r"primary (?:skin |dermal )?irrit|irritant dermatitis|"
    r"(?:skin|dermal) irritation|skin corrosion|corrosive assay",
    re.IGNORECASE,
)
_PREDICTION_RE = re.compile(
    r"\bin[ -]?silico\b|\bqsar\b|read[- ]?across|\btopkat\b|\bderek\b|"
    r"\btimes[- ]?(?:ss|m|p)\b|oecd toolbox|predicted value|"
    r"computational (?:model|predict)|model prediction|machine learning",
    re.IGNORECASE,
)
_IN_SILICO_RE = re.compile(r"\bin[ -]?silico\b", re.IGNORECASE)
_INTEGRATED_APPROACH_RE = re.compile(
    r"\b2\s*out\s*of\s*3\b|\b2o3\b|defined approach|integrated (?:testing|approach)|"
    r"\biata\b|integrated testing strateg",
    re.IGNORECASE,
)
_INTEGRATED_CONTEXT_RE = re.compile(
    r"defined approach|integrated (?:testing|approach|testing strateg)|\biata\b",
    re.IGNORECASE,
)
_DIRECT_ASSAY_RE = re.compile(
    r"\bllna\b|local lymph node|\bgpmt\b|guinea pig maximi[sz]ation|"
    r"\bbuehler\b|\bhript\b|\bript\b|human maximi[sz]ation|"
    r"patch test|epicutaneous test|mouse ear swelling|\bmest\b|"
    r"contact hypersensitivity|skin sensiti[sz]ation test|"
    r"guinea pig sensiti[sz]ation|clinical case|clinical observation",
    re.IGNORECASE,
)
_DIRECT_OUTCOME_RE = re.compile(
    r"skin sensiti[sz]|contact allerg|allergic contact dermatitis|"
    r"contact hypersensitiv|\bllna\b|local lymph node|\bgpmt\b|"
    r"\bbuehler\b|\bhript\b|human maximi[sz]ation",
    re.IGNORECASE,
)

_AOP_ASSAY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "MIE_protein_binding",
        re.compile(
            r"\b[a-z]?dpra\b|direct peptide reactivity|\badra(?:[- ]dm|[- ]fl|[- ]uv)?\b|"
            r"peptide (?:depletion|reactivity|adduct)|"
            r"protein binding|cysteine depletion|lysine depletion|"
            r"glutathione|\bgsh\b|haptenation|covalent adduct",
            re.IGNORECASE,
        ),
    ),
    (
        "KE2_keratinocyte_activation",
        re.compile(
            r"keratino\w*sens|\blusens\b|\bsens[- ]?is\b|\bepisensa\b|\barec32\b|"
            r"are[- ]luciferase|nrf2 activation|keratinocyte.*(?:activation|il-18)|"
            r"nctc2544[- ]il[- ]18|il[- ]18.*(?:epiderm|\brhe\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "KE3_dendritic_cell_activation",
        re.compile(
            r"h[- ]?clat|human cell line activation|\bu[- ]?sens\b|\bmusst\b|\bmmusst\b|"
            r"il[- ]8 luc|dendritic cell activation|thp[- ]?1.*(?:cd54|cd86|activation)|"
            r"u937.*cd86|cd(?:54|86) expression|dendritic cell.*(?:activation|assay|model)|"
            r"gardskin|gard skin|\blcsa\b",
            re.IGNORECASE,
        ),
    ),
    (
        "KE4_T_cell_activation",
        re.compile(
            r"t[- ]?cell (?:activation|proliferation)|lymphocyte transformation|"
            r"\bltt\b|lymphocyte proliferation|t[- ]?cell clone proliferation",
            re.IGNORECASE,
        ),
    ),
)


def classify_direct_source_record(record: Mapping[str, Any]) -> PartitionDecision:
    """Route one raw direct-acquisition row without changing the immutable source."""
    if not is_tdc_skin_sensitization_scope(record.get("reaction_type")):
        return PartitionDecision(REJECT_PARTITION, "outside_sensitization_scope")

    assay = _text(record.get("assay_or_test"))
    context = _join_fields(
        record,
        "support_text",
        "assay_or_test",
        "species_or_population",
        "extra_details",
        "dose_or_concentration",
    )
    if _PHOTO_RE.search(context):
        return PartitionDecision(REJECT_PARTITION, "out_of_scope_photo_hazard")
    if _INTEGRATED_APPROACH_RE.search(assay) or _INTEGRATED_CONTEXT_RE.search(context):
        return PartitionDecision(REJECT_PARTITION, "integrated_prediction_or_defined_approach")

    aop_event = infer_aop_event(assay)
    if aop_event:
        if _prediction_only(record, experimental_assay=True):
            return PartitionDecision(REJECT_PARTITION, "prediction_only")
        return PartitionDecision(AOP_PARTITION, "mechanistic_assay_in_direct_source", aop_event)

    direct_assay = bool(_DIRECT_ASSAY_RE.search(assay))
    if _prediction_only(record, experimental_assay=direct_assay):
        return PartitionDecision(REJECT_PARTITION, "prediction_only")
    if _IRRITATION_RE.search(context) and not (
        direct_assay or _DIRECT_OUTCOME_RE.search(_text(record.get("support_text")))
    ):
        return PartitionDecision(REJECT_PARTITION, "out_of_scope_irritation")
    return PartitionDecision(DIRECT_PARTITION, "scoped_direct_outcome")


def classify_aop_source_record(record: Mapping[str, Any]) -> PartitionDecision:
    """Route one raw AOP-acquisition row into direct, mechanistic AOP, or reject."""
    assay = _text(record.get("assay_type"))
    context = _join_fields(
        record,
        "support_text",
        "assay_type",
        "endpoint_or_target",
        "species_or_population",
        "experimental_conditions",
        "qualifying_conditions",
        "extra_details",
    )
    if _PHOTO_RE.search(context):
        return PartitionDecision(REJECT_PARTITION, "out_of_scope_photo_hazard")
    if _IRRITATION_RE.search(context):
        return PartitionDecision(REJECT_PARTITION, "out_of_scope_irritation")
    if _INTEGRATED_APPROACH_RE.search(assay) or _INTEGRATED_CONTEXT_RE.search(context):
        return PartitionDecision(REJECT_PARTITION, "integrated_prediction_or_defined_approach")

    raw_event = _text(record.get("aop_event"))
    inferred_event = infer_aop_event(assay)
    direct_assay = bool(_DIRECT_ASSAY_RE.search(assay))
    experimental_assay = bool(inferred_event or direct_assay)
    if _prediction_only(record, experimental_assay=experimental_assay):
        return PartitionDecision(REJECT_PARTITION, "prediction_only")

    if raw_event in AOP_EVENTS:
        return PartitionDecision(AOP_PARTITION, "explicit_aop_key_event", raw_event)

    if raw_event == "adverse_outcome_skin_sensitization":
        if not _normalized_direct_label(record.get("result_label")):
            return PartitionDecision(REJECT_PARTITION, "direct_outcome_without_usable_label")
        if inferred_event and not direct_assay:
            return PartitionDecision(
                AOP_PARTITION,
                "adverse_outcome_reclassified_by_mechanistic_assay",
                inferred_event,
            )
        if direct_assay or _DIRECT_OUTCOME_RE.search(_text(record.get("support_text"))):
            return PartitionDecision(DIRECT_PARTITION, "aop_adverse_outcome_moved_to_direct")
        return PartitionDecision(REJECT_PARTITION, "adverse_outcome_without_direct_anchor")

    if inferred_event:
        return PartitionDecision(AOP_PARTITION, "unspecified_record_reclassified_by_assay", inferred_event)
    if direct_assay and _normalized_direct_label(record.get("result_label")):
        return PartitionDecision(DIRECT_PARTITION, "unspecified_record_reclassified_as_direct")
    return PartitionDecision(REJECT_PARTITION, "integrated_or_unresolved_endpoint")


def direct_outcome_reason(record: Mapping[str, Any]) -> str:
    """Identify a measured final sensitization outcome in any source schema.

    This deliberately takes precedence over an AOP-event tag.  LLNA, GPMT,
    Buehler, HRIPT/RIPT, and validated human patch outcomes remain direct even
    when an upstream extraction also labels lymphocyte activation as KE4.
    """

    assay = _join_fields(
        record,
        "canonical_assay_type",
        "canonical_assay_or_test",
        "assay_type",
        "assay_or_test",
        "canonical_assay_context",
    )
    direct_assay = bool(_DIRECT_ASSAY_RE.search(assay))
    # The override is intentionally assay-anchored.  Generic prose such as
    # "skin sensitization" also occurs in h-CLAT/DPRA model summaries and is
    # not sufficient to turn a mechanistic record into a final-outcome row.
    if not direct_assay:
        return ""

    label = " | ".join(
        _text(record.get(field))
        for field in (
            "result_label",
            "outcome_label",
            "canonical_measurement_text",
            "measurement_text",
            "support_text",
        )
    )
    usable_label = bool(
        re.search(
            r"\b(?:positive|negative|weak positive|sensiti[sz]er|non[- ]?sensiti[sz]er|"
            r"no (?:skin )?sensiti[sz]ation|did not (?:induce|show|produce).{0,30}sensiti[sz])\b",
            label,
            re.IGNORECASE,
        )
    )
    quantitative_llna = bool(
        direct_assay
        and re.search(
            r"\b(?:stimulation index|\bsi\b|ec3|lymph node (?:weight|proliferation))\b.{0,80}\d",
            label,
            re.IGNORECASE,
        )
    )
    if not (usable_label or quantitative_llna):
        return ""
    return "validated_direct_assay_outcome_overrides_aop_tag"


def infer_aop_event(assay: Any) -> str:
    text = _normalized_assay_text(assay)
    for event, pattern in _AOP_ASSAY_PATTERNS:
        if pattern.search(text):
            return event
    return ""


def normalized_direct_label(value: Any) -> str:
    return _normalized_direct_label(value)


def _normalized_direct_label(value: Any) -> str:
    label = _text(value).lower().replace("-", "_").replace(" ", "_")
    if label in {"positive", "weak_positive"}:
        return "positive"
    if label == "negative":
        return "negative"
    if label in {"equivocal", "inconclusive"}:
        return "inconclusive"
    return ""


def _prediction_only(record: Mapping[str, Any], *, experimental_assay: bool) -> bool:
    assay_context = _join_fields(
        record,
        "assay_or_test",
        "assay_type",
        "endpoint_or_target",
        "species_or_population",
        "experimental_conditions",
        "qualifying_conditions",
        "extra_details",
    )
    support_text = _text(record.get("support_text"))
    if _IN_SILICO_RE.search(assay_context) or _IN_SILICO_RE.search(support_text):
        return True
    if _PREDICTION_RE.search(assay_context):
        return not experimental_assay
    return bool(
        _PREDICTION_RE.search(support_text)
        and not experimental_assay
    )


def _join_fields(record: Mapping[str, Any], *fields: str) -> str:
    return " | ".join(_text(record.get(field)) for field in fields)


def _normalized_assay_text(value: Any) -> str:
    text = _text(value)
    return re.sub(r"[\u00ad\u2010-\u2015\u2212]", "-", text)


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:  # NaN
            return ""
    except Exception:
        pass
    return str(value).strip()
