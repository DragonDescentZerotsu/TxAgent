"""Exact voter-membership rules for the Skin retrieval source."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.source_family_purity import FamilyMove
from tools.chembl_tool.common.starling.conditioned_benchmark import task_root
from tools.chembl_tool.tasks.skin_reaction.canonical_starling_source import (
    AOP_EVENTS,
    AOP_PARTITION,
    DIRECT_PARTITION,
    PARTITION_AUDIT_PATH,
    REJECT_PARTITION,
    PartitionDecision,
    infer_aop_event,
    strict_scope_exclusion_reason,
)
from tools.chembl_tool.tasks.skin_reaction.starling_benchmark import (
    SOURCE_PATH,
    load_label_decisions,
)


DIRECT_GROUP = "Direct.skin_reaction"
NEAR_DIRECT_GROUP = "Observed.nonvoter_skin_outcome"
AOP_GROUP = "Mechanism.sensitization_aop"
EXCLUDED_GROUP = "Excluded.skin_sensitization_source_purity"
PURITY_VERSION = "skin_source_family_purity.strict_target_aligned.v7"
SOURCE_REVIEW = (
    Path(__file__).resolve().parents[4]
    / "data/starling_data/skin_reaction/trace_review_20260914/decisions_rebound.json"
)
DEFAULT_CONDITION_REVIEW = task_root("skin_reaction") / "source_condition_review.jsonl"
SourceRecordKey = tuple[str, int]

_NO_EVIDENCE_RE = re.compile(
    r"does not (?:contain|address|report)|no extractable|contains no|"
    r"unrelated to skin sensiti[sz]ation|without (?:any )?(?:skin )?sensiti[sz]ation evidence",
    re.IGNORECASE,
)
_DIRECT_ASSAY_RE = re.compile(
    r"\bllna\b|local lymph node|\bgpmt\b|guinea pig maximi[sz]ation|"
    r"\bbuehler\b|\bhript\b|\bript\b|human maximi[sz]ation|"
    r"patch test|epicutaneous test|contact hypersensitivity|"
    r"\bmdam\b|modified draize",
    re.IGNORECASE,
)
_OVERALL_CLASSIFICATION_RE = re.compile(
    r"^(?:clinical )?skin sensiti[sz]ation(?: induction)?$|"
    r"^(?:clinical )?sensiti[sz]ation(?: induction)?$|^nesil$|"
    r"skin sensiti[sz]ation (?:prediction|classification|potential|probability|hazard|potency)|"
    r"sensiti[sz]ation (?:prediction|classification|potential|probability|hazard|potency)|"
    r"overall skin sensiti[sz]ation|integrated (?:skin )?sensiti[sz]ation|"
    r"GHS (?:sub)?categor|sensiti[sz]er/non[- ]?sensiti[sz]er|"
    r"^(?:hazard classification|total score|pti score)$",
    re.IGNORECASE,
)
_SENSITIZATION_ANCHOR_RE = re.compile(
    r"skin sensiti[sz]|contact sensiti[sz]|contact allerg|contact hypersens|"
    r"\bhapten|\ballergen",
    re.IGNORECASE,
)
_BIOLOGICAL_MECHANISM_RE = re.compile(
    r"keratin|hacat|thp[- ]?1|u[- ]?937|dendritic|t[- ]?cell|lymphocyt|"
    r"cytokine|interleukin|\bil[- ]?\d|\bros\b|glutathione|cysteine|lysine|"
    r"protein (?:binding|reactivity)|gene expression|cd54|cd86|nrf2|keap1|"
    r"immune|inflamm|pge2|txb2|pgd2",
    re.IGNORECASE,
)
_STRONG_AOP_CONTEXT_RE = re.compile(
    r"keratin|hacat|thp[- ]?1|u[- ]?937|dendritic|t[- ]?cell|lymphocyt|"
    r"glutathione|cysteine|lysine|protein (?:binding|reactivity)|cd54|cd86|"
    r"nrf2|keap1|\bhapten|\ballergen",
    re.IGNORECASE,
)


def load_canonical_partition_decisions(
    path: Path = PARTITION_AUDIT_PATH,
) -> dict[SourceRecordKey, PartitionDecision]:
    """Load the frozen v4 semantic partition keyed by acquisition row."""

    import pyarrow.parquet as pq

    table = pq.read_table(
        path,
        columns=[
            "source_partition",
            "source_index",
            "partition",
            "partition_reason",
            "canonical_aop_event",
        ],
    )
    output: dict[SourceRecordKey, PartitionDecision] = {}
    for row in table.to_pylist():
        key = (str(row["source_partition"]), int(row["source_index"]))
        if key in output:
            raise ValueError(f"duplicate canonical Skin partition key: {key!r}")
        output[key] = PartitionDecision(
            partition=str(row["partition"]),
            reason=str(row["partition_reason"]),
            aop_event=str(row.get("canonical_aop_event") or ""),
        )
    return output


def load_voter_source_keys(
    *,
    source_path: Path = SOURCE_PATH,
    condition_review: Path = DEFAULT_CONDITION_REVIEW,
) -> set[SourceRecordKey]:
    """Load records that emitted a base vote or passed condition review."""

    decisions, _ = load_label_decisions(source_path=source_path)
    voters: set[SourceRecordKey] = set()
    for index, decision in enumerate(decisions):
        if decision.record is None:
            continue
        identity = normalize_molecule_identity(decision.record.smiles)
        if identity.status == "ok" and identity.parent_smiles:
            voters.add(("direct_skin_reaction", index))
    if condition_review.exists():
        with condition_review.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("review_status") or "").lower() == "accepted":
                    voters.add(source_record_key_from_id(row["source_record_id"]))
    return voters


def source_record_key_from_id(value: Any) -> SourceRecordKey:
    """Parse the stable ``<source_partition>:<zero-based-index>`` identity."""

    source_id, separator, index = str(value or "").rpartition(":")
    if not separator or source_id not in {"direct_skin_reaction", "sensitization_aop"}:
        raise ValueError(f"invalid Skin source record id: {value!r}")
    try:
        parsed = int(index)
    except (TypeError, ValueError):
        raise ValueError(f"invalid Skin source record id: {value!r}") from None
    if parsed < 0:
        raise ValueError(f"invalid Skin source record id: {value!r}")
    return source_id, parsed


def upstream_source_key(record: Mapping[str, Any]) -> SourceRecordKey | None:
    """Map a normalized Stage-03 row to its stable acquisition-row identity."""

    source_id = str(record.get("source_id") or "")
    if source_id not in {"direct_skin_reaction", "sensitization_aop"}:
        return None
    try:
        index = int(record.get("source_row_number")) - 1
    except (TypeError, ValueError):
        return None
    return (source_id, index) if index >= 0 else None


def canonical_partition(
    record: Mapping[str, Any],
    canonical_partitions: Mapping[SourceRecordKey, PartitionDecision],
) -> PartitionDecision | None:
    """Look up the frozen v4 semantic decision for a Stage-03 source row."""

    key = upstream_source_key(record)
    return canonical_partitions.get(key) if key is not None else None


@lru_cache(maxsize=1)
def load_reviewed_source_decisions() -> dict[SourceRecordKey, dict[str, Any]]:
    """Load narrow source repairs; raw acquisition and frozen votes stay intact."""
    review = json.loads(SOURCE_REVIEW.read_text(encoding="utf-8"))
    if review.get("version") != "skin_source_review.v1":
        raise ValueError("unsupported Skin source review")
    decisions = {}
    for row in review["decisions"]:
        key = source_record_key_from_id(row["source_record_key"])
        if key in decisions or row["new_group"] not in {EXCLUDED_GROUP, NEAR_DIRECT_GROUP, AOP_GROUP}:
            raise ValueError(f"invalid Skin source review: {key}")
        document = Path(__file__).resolve().parents[4] / row["source_document"]
        if hashlib.sha256(document.read_bytes()).hexdigest() != row["document_sha256"]:
            raise ValueError(f"Skin source review document changed: {key}")
        decisions[key] = row
    return decisions


def vote_pure_family_move(
    record: Mapping[str, Any],
    voter_source_keys: set[SourceRecordKey] | frozenset[SourceRecordKey],
    canonical_partitions: Mapping[SourceRecordKey, PartitionDecision],
) -> FamilyMove:
    """Route actual voters, nonvoter outcomes, and experimental AOP evidence."""

    original = str(record.get("group_id") or "")
    key = upstream_source_key(record)
    semantic = canonical_partition(record, canonical_partitions)
    if key is None:
        return FamilyMove("", "")
    if semantic is None:
        raise RuntimeError(
            f"Skin source row {key!r} is absent from the canonical partition audit"
        )

    review = load_reviewed_source_decisions().get(key)
    if review is not None:
        if key in voter_source_keys:
            raise ValueError(f"Skin source review cannot edit a frozen voter: {key}")
        if any(record.get(k) != v for k, v in review["expected"].items()):
            raise ValueError(f"Skin source review signature changed: {key}")
        return FamilyMove(review["new_group"], "reviewed_source:" + review["reason"])

    strict_exclusion = strict_scope_exclusion_reason(record)
    if strict_exclusion:
        return FamilyMove(
            EXCLUDED_GROUP,
            f"strict_target_scope_exclusion:{strict_exclusion}",
        )

    if key in voter_source_keys:
        if original == DIRECT_GROUP:
            return FamilyMove("", "")
        return FamilyMove(
            DIRECT_GROUP, "accepted_gold_vote_promoted_to_l1_by_stable_source_key"
        )

    if semantic.partition == DIRECT_PARTITION:
        if original == NEAR_DIRECT_GROUP:
            return FamilyMove("", "")
        return FamilyMove(
            NEAR_DIRECT_GROUP,
            f"canonical_direct_nonvoter_to_l2:{semantic.reason}",
        )
    if semantic.partition == AOP_PARTITION:
        if original == AOP_GROUP:
            return FamilyMove("", "")
        return FamilyMove(
            AOP_GROUP, f"canonical_experimental_aop_to_l3:{semantic.reason}"
        )
    if semantic.partition == REJECT_PARTITION:
        retrieval_target = rejected_record_retrieval_target(record, semantic.reason)
        if retrieval_target:
            return retrieval_target
        return FamilyMove(
            EXCLUDED_GROUP, f"excluded_from_sensitization_levels:{semantic.reason}"
        )
    raise RuntimeError(f"unexpected Skin canonical partition: {semantic}")


def rejected_record_retrieval_target(
    record: Mapping[str, Any], rejection_reason: str
) -> FamilyMove | None:
    """Recover relevant non-gold evidence without weakening the gold boundary.

    Predictions and defined approaches remain ineligible for L1.  Their overall
    sensitization classifications are near-direct L2 evidence, while predicted,
    experimental, or integrated mechanistic readouts belong to L3. A formerly
    unresolved acquisition enters L3 when it contains a substantive assay,
    endpoint, measurement, or support record; empty and explicitly unrelated
    extraction artifacts remain excluded.
    """

    if rejection_reason not in {
        "prediction_only",
        "integrated_prediction_or_defined_approach",
        "integrated_or_unresolved_endpoint",
    }:
        return None

    assay = _joined_text(
        record,
        "canonical_assay_type",
        "canonical_assay_or_test",
        "assay_type",
        "assay_or_test",
        "canonical_assay_context",
    )
    endpoint = _joined_text(
        record,
        "canonical_endpoint_name",
        "endpoint_name",
        "endpoint_or_target",
    )
    support = _joined_text(
        record,
        "canonical_measurement_text",
        "measurement_text",
        "support_text",
        "experimental_conditions",
        "qualifying_conditions",
        "extra_details",
    )
    combined = " | ".join((assay, endpoint, support))
    if not any((assay, endpoint, support)) or _NO_EVIDENCE_RE.search(combined):
        return None
    event = _text(record.get("canonical_aop_event")) or _text(record.get("aop_event"))
    if _DIRECT_ASSAY_RE.search(combined):
        return FamilyMove(
            NEAR_DIRECT_GROUP,
            f"relevant_rejected_record_to_l2:direct_outcome:{rejection_reason}",
        )
    if event in AOP_EVENTS:
        return FamilyMove(
            AOP_GROUP,
            f"relevant_rejected_record_to_l3:explicit_aop_event:{rejection_reason}",
        )
    if _OVERALL_CLASSIFICATION_RE.search(assay) or _OVERALL_CLASSIFICATION_RE.search(
        endpoint
    ):
        return FamilyMove(
            NEAR_DIRECT_GROUP,
            f"relevant_rejected_record_to_l2:derived_overall_classification:{rejection_reason}",
        )

    inferred_event = infer_aop_event(" | ".join((assay, endpoint, support)))
    if inferred_event:
        return FamilyMove(
            AOP_GROUP,
            f"relevant_rejected_record_to_l3:inferred_{inferred_event}:{rejection_reason}",
        )
    if _STRONG_AOP_CONTEXT_RE.search(combined) or (
        _SENSITIZATION_ANCHOR_RE.search(combined)
        and _BIOLOGICAL_MECHANISM_RE.search(combined)
    ):
        return FamilyMove(
            AOP_GROUP,
            f"relevant_rejected_record_to_l3:anchored_unspecified_mechanism:{rejection_reason}",
        )
    if rejection_reason == "integrated_prediction_or_defined_approach":
        return FamilyMove(
            NEAR_DIRECT_GROUP,
            "relevant_rejected_record_to_l2:defined_approach_overall_classification",
        )
    if rejection_reason == "prediction_only":
        return FamilyMove(
            NEAR_DIRECT_GROUP,
            "relevant_rejected_record_to_l2:predicted_overall_sensitization",
        )
    if rejection_reason == "integrated_or_unresolved_endpoint":
        return FamilyMove(
            AOP_GROUP,
            "relevant_rejected_record_to_l3:substantive_unspecified_mechanism",
        )
    return None


def _joined_text(record: Mapping[str, Any], *fields: str) -> str:
    values = (_text(record.get(field)) for field in fields)
    return " | ".join(value for value in values if value)


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        pass
    return " ".join(str(value).split())
