"""Audited BBB source-family reassignment for the progressive assay experiment.

This module assigns each retrieval source record to one biological family.  It
does not mutate a gold artifact in place: the corrected v3 benchmark is rebuilt
and revoted independently before this overlay is published.  L1 follows that
current experimental gold contract; explicit predictions remain retrievable
but are not presented as experimental direct evidence.  Assay identity is
retained only as provenance; it does not force all records from an assay into
one family.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping

from tools.chembl_tool.common.source_family_purity import FamilyMove
from tools.chembl_tool.tasks.bbb_martins.experimental_meaningful_cns_access_benchmark_v3 import (
    label_record,
)
from tools.chembl_tool.tasks.bbb_martins.experimental_meaningful_cns_access_benchmark_v4 import (
    label_record as label_record_v4,
)


DIRECT_GROUP = "Tier 1.starling_direct_bbb_evidence"
NEAR_DIRECT_GROUP = "Proxy.central_functional_access"
PASSIVE_GROUP = "Mechanism.passive_permeability"
EFFLUX_GROUP = "Mechanism.efflux_transport"
INFLUX_GROUP = "Mechanism.influx_transport"
PURITY_VERSION = "bbb_source_family_purity.v4"
REVIEW_VERSION = "bbb_near_direct_record_review.v2"

DEFAULT_RECORDS = Path(
    "/data1/joseph/TxAgent/outputs/chembl_tool/tasks/bbb_martins/"
    "evidence_library/starling_normalized_v7/03_records/records.parquet"
)
DEFAULT_REVIEW_LEDGER = Path(
    "data/starling_data/bbb_martins/source_family_purity_v2/"
    "near_direct_record_review.jsonl"
)
DEFAULT_GOLD_ROOT = Path(
    "data/processed_starling_experimental_meaningful_cns_access_v3/BBB_Martins"
)
DEFAULT_CONDITIONED_ROOT = Path(
    "data/processed_starling_context_conditioned_selected_v2/BBB_Martins/scaffold"
)

CLASSIFIER_COLUMNS = (
    "group_id",
    "source_id",
    "source_index",
    "source_record_id",
    "canonical_record_id",
    "extraction_id",
    "retrieval_eligible",
    "canonical_smiles",
    "smiles",
    "pmid",
    "bbb_permeability_label",
    "species",
    "canonical_assay_context",
    "assay_model",
    "assay_system",
    "biological_system",
    "canonical_assay_type",
    "assay_type",
    "canonical_endpoint_name",
    "endpoint_name",
    "canonical_measurement_text",
    "measurement_text",
    "canonical_unit_text",
    "unit_text",
    "bbb_transport_label",
    "canonical_transport_mechanism",
    "transport_mechanism",
    "transporter_identifier",
    "canonical_transporter_identifier",
    "support_text",
    "extra_details",
)

_PAMPA = re.compile(r"\bpampa(?:[- _]?bbb)?\b", re.IGNORECASE)
_MDCK = re.compile(
    r"\bmdck(?:[- _]?(?:ii|mrd1|mdr1|bcrp))?\b|madin[- ]darby|\bllc[- _]?pk1\b",
    re.IGNORECASE,
)
_EFFLUX_SYSTEM = re.compile(
    r"mdr1|p[- ]?gp|p[- ]?glycoprotein|abcb1|bcrp|abcg2|mrp\d*|abcc\d*|"
    r"efflux|transporter[- ]transfected|overexpress",
    re.IGNORECASE,
)
_EFFLUX_ENDPOINT = re.compile(
    r"efflux|b[- _]?to[- _]?a.{0,20}(?:a[- _]?to[- _]?b|ratio)|"
    r"(?:a[- _]?to[- _]?b|ratio).{0,20}b[- _]?to[- _]?a",
    re.IGNORECASE,
)
_INFLUX_DIRECTION = re.compile(
    r"\binflux\b|carrier[- ]mediated (?:transport|uptake)|"
    r"transporter[- ]mediated uptake|active uptake|luminal[- ]to[- ]abluminal",
    re.IGNORECASE,
)
_INFLUX_SYSTEM = re.compile(
    r"\bslc\w*\b|\boatp\w*\b|\blat1\b|\bglut1\b|\boctn?\d*\b|"
    r"\bmct\d*\b|\bpept\d*\b",
    re.IGNORECASE,
)
_PASSIVE_ENDPOINT = re.compile(
    r"papp|permeab|a[- _]?to[- _]?b|b[- _]?to[- _]?a|"
    r"apical.{0,20}basolateral|basolateral.{0,20}apical|transport rate",
    re.IGNORECASE,
)
_PREDICTION = re.compile(
    r"\bin[ -]?silico\b|comput(?:ational|ed)|predict(?:ed|ion|ive)|"
    r"\bqsar\b|qikprop|swiss\s*adme|admet(?:sar)?|pkcsm|boiled[ -]?egg|"
    r"herbal egg|machine learning|neural network|\bpbpk\b|simulat(?:ed|ion)",
    re.IGNORECASE,
)
_QIKPROP = re.compile(r"qikprop|qplogbb|qppmdck|qplogmdck", re.IGNORECASE)
_QIKPROP_PERMEABILITY = re.compile(
    r"qppmdck|qplogmdck|mdck|papp|apparent permeability|cell permeability",
    re.IGNORECASE,
)
_BOILED_EGG = re.compile(r"boiled[ -]?egg|herbal egg", re.IGNORECASE)
_COMPUTED_BBB_SOURCE = re.compile(
    r"\badme/?t analysis\b|\btcmsp database\b", re.IGNORECASE
)
_PASSIVE_MECHANISM = re.compile(
    r"passive diffusion|passively diffus|transcellular passive|artificial membrane|"
    r"lipophilic permeability",
    re.IGNORECASE,
)
_GENERIC_ENDPOINTS = {"", "missing_endpoint", "bbb_permeability_outcome"}
_EFFLUX_ENDPOINT_NAMES = {
    "efflux_inhibition_outcome",
    "efflux_ratio",
    "efflux_substrate_outcome",
}
_INFLUX_ENDPOINT_NAMES = {
    "brain_endothelial_cell_uptake",
    "blood_to_brain_transport",
    "influx",
    "influx_rate_constant",
    "luminal_to_abluminal_transport",
    "unspecified_bbb_influx",
}
# Manual full-record adjudication: normalized-v7 labels this row as an in-silico
# compartmental model, but its source support reports experimentally established
# P-gp/BCRP cooperation restricting measured brain distribution. Do not demote it
# solely because the normalized assay-context field contains prediction language.
_EXPERIMENTAL_PREDICTION_FALSE_POSITIVES = {112971}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def source_index(record: Mapping[str, Any]) -> int | None:
    value = record.get("source_index")
    if value is not None and _text(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    for field in ("source_record_id", "canonical_record_id", "extraction_id"):
        match = re.search(r"(?:^|:)row:(\d+)$", _text(record.get(field)))
        if match:
            return int(match.group(1))
    return None


def _searchable(record: Mapping[str, Any]) -> str:
    return " | ".join(
        _text(record.get(field))
        for field in (
            "canonical_assay_context",
            "assay_model",
            "assay_system",
            "biological_system",
            "canonical_endpoint_name",
            "endpoint_name",
            "canonical_measurement_text",
            "measurement_text",
            "bbb_transport_label",
            "canonical_transport_mechanism",
            "transport_mechanism",
            "transporter_identifier",
            "canonical_transporter_identifier",
            "support_text",
            "extra_details",
        )
    )


def _assay_identity_text(record: Mapping[str, Any]) -> str:
    """Fields that define the assay system, excluding narrative support text."""

    return " | ".join(
        _text(record.get(field))
        for field in (
            "canonical_assay_context",
            "assay_model",
            "assay_system",
            "biological_system",
            "canonical_assay_type",
            "assay_type",
        )
    )


def support_key(record: Mapping[str, Any]) -> str:
    """Stable key used only to keep reviewed duplicate source rows together."""

    fields = (
        _text(record.get("canonical_smiles") or record.get("smiles")),
        _text(record.get("pmid")),
        " ".join(_text(record.get("support_text")).lower().split()),
    )
    return hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()


def is_pampa_record(record: Mapping[str, Any]) -> bool:
    return bool(_PAMPA.search(_assay_identity_text(record)))


def is_mdck_record(record: Mapping[str, Any]) -> bool:
    return bool(_MDCK.search(_assay_identity_text(record)))


def mdck_target(record: Mapping[str, Any]) -> tuple[str, str]:
    """Classify one full MDCK/LLC-PK1 row without dropping sparse rows."""

    text = _searchable(record)
    if _explicit_efflux(record, text):
        return EFFLUX_GROUP, "mdck_transporter_or_efflux_readout"
    if _explicit_influx(record, text):
        return INFLUX_GROUP, "mdck_explicit_influx_or_uptake_readout"
    if _PASSIVE_ENDPOINT.search(text):
        return PASSIVE_GROUP, "mdck_apparent_or_passive_permeability_readout"
    return PASSIVE_GROUP, "mdck_sparse_endpoint_cell_permeability_context"


def _gold_contract_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Map normalized-v7 fields onto the frozen gold classifier surface."""

    mapped = dict(record)
    mapped["quant_metric"] = _text(
        record.get("canonical_endpoint_name") or record.get("endpoint_name")
    )
    mapped["quant_value"] = _text(
        record.get("canonical_measurement_text") or record.get("measurement_text")
    )
    mapped["quant_units"] = _text(
        record.get("canonical_unit_text") or record.get("unit_text")
    )
    return mapped


def gold_contract_decision(record: Mapping[str, Any]) -> tuple[int | None, str]:
    """Replay the conditioned-compatible experimental BBB gold contract."""

    return label_record(
        _gold_contract_record(record),
        source_index=source_index(record),
        allow_conditioned_context=True,
    )


def gold_contract_decision_v4(
    record: Mapping[str, Any],
) -> tuple[int | None, str]:
    """Replay the reviewed v4 BBB gold contract on normalized retrieval rows."""

    return label_record_v4(
        _gold_contract_record(record),
        source_index=source_index(record),
        allow_conditioned_context=True,
    )


def _explicit_influx(record: Mapping[str, Any], text: str) -> bool:
    transport_label = _text(record.get("bbb_transport_label")).lower()
    mechanism = _text(
        record.get("canonical_transport_mechanism")
        or record.get("transport_mechanism")
    ).lower()
    endpoint = _text(
        record.get("canonical_endpoint_name") or record.get("endpoint_name")
    ).lower()
    if endpoint in _EFFLUX_ENDPOINT_NAMES:
        return False
    return (
        transport_label == "influx_substrate"
        or "influx" in mechanism
        or "uptake" in mechanism
        or (
            transport_label == "transporter_mediated"
            and endpoint in _INFLUX_ENDPOINT_NAMES
        )
        or bool(_INFLUX_DIRECTION.search(text))
        or (
            endpoint in _INFLUX_ENDPOINT_NAMES
            and bool(_INFLUX_SYSTEM.search(text))
        )
    )


def _explicit_efflux(record: Mapping[str, Any], text: str) -> bool:
    transport_label = _text(record.get("bbb_transport_label")).lower()
    endpoint = _text(
        record.get("canonical_endpoint_name") or record.get("endpoint_name")
    )
    return (
        transport_label
        in {"efflux_substrate", "not_subject_to_efflux", "efflux_limited"}
        or endpoint.lower() in _EFFLUX_ENDPOINT_NAMES
        or bool(_EFFLUX_ENDPOINT.search(endpoint))
        or bool(_EFFLUX_SYSTEM.search(text))
    )


def _explicit_passive(record: Mapping[str, Any], text: str) -> bool:
    endpoint = _text(
        record.get("canonical_endpoint_name") or record.get("endpoint_name")
    )
    return (
        is_pampa_record(record)
        or (
            endpoint.lower() not in _GENERIC_ENDPOINTS
            and bool(_PASSIVE_ENDPOINT.search(endpoint))
        )
        or bool(_PASSIVE_MECHANISM.search(text))
    )


def is_explicit_prediction_record(record: Mapping[str, Any]) -> bool:
    """Identify source-native prediction rows without broad model false positives."""

    if source_index(record) in _EXPERIMENTAL_PREDICTION_FALSE_POSITIVES:
        return False
    text = _searchable(record)
    endpoint = _text(
        record.get("canonical_endpoint_name") or record.get("endpoint_name")
    ).lower()
    if _QIKPROP.search(text) or _BOILED_EGG.search(text) or _PREDICTION.search(text):
        return True
    return bool(_COMPUTED_BBB_SOURCE.search(text)) and (
        endpoint in {"logbb", "brain_to_blood_ratio", "bbb_permeability_outcome"}
        or "logbb" in text.lower()
        or "bbb parameter" in text.lower()
    )


def nondirect_target(record: Mapping[str, Any], rejection_reason: str) -> FamilyMove:
    """Assign one non-gold-compatible row using its complete semantic record."""

    text = _searchable(record)
    endpoint = _text(
        record.get("canonical_endpoint_name") or record.get("endpoint_name")
    ).lower()
    prediction = is_explicit_prediction_record(record) or (
        rejection_reason == "computational_or_predicted_result"
    )

    if _BOILED_EGG.search(text):
        if _explicit_efflux(record, text):
            return FamilyMove(EFFLUX_GROUP, "boiled_egg_predicted_efflux_or_pgp")
        return FamilyMove(PASSIVE_GROUP, "boiled_egg_passive_permeability_prediction")
    if _QIKPROP.search(text):
        if _explicit_efflux(record, text):
            return FamilyMove(EFFLUX_GROUP, "qikprop_predicted_efflux_or_pgp")
        if _QIKPROP_PERMEABILITY.search(text):
            return FamilyMove(PASSIVE_GROUP, "qikprop_predicted_mdck_or_permeability")
        return FamilyMove(NEAR_DIRECT_GROUP, "qikprop_predicted_bbb_outcome")
    if is_pampa_record(record):
        if _explicit_efflux(record, text):
            return FamilyMove(EFFLUX_GROUP, "pampa_context_explicit_efflux_readout")
        return FamilyMove(PASSIVE_GROUP, "pampa_membrane_permeability_assay")
    if is_mdck_record(record):
        target, reason = mdck_target(record)
        return FamilyMove(target, reason)
    if prediction:
        return FamilyMove(NEAR_DIRECT_GROUP, "computational_or_predicted_bbb_proxy")
    if _explicit_efflux(record, text):
        return FamilyMove(EFFLUX_GROUP, "explicit_efflux_or_transporter_mechanism")
    if _explicit_influx(record, text):
        return FamilyMove(INFLUX_GROUP, "explicit_influx_or_uptake_mechanism")
    if _explicit_passive(record, text):
        return FamilyMove(PASSIVE_GROUP, "explicit_passive_permeability_mechanism")

    if endpoint in _GENERIC_ENDPOINTS:
        return FamilyMove(NEAR_DIRECT_GROUP, "missing_or_generic_bbb_proxy")
    return FamilyMove(
        NEAR_DIRECT_GROUP,
        f"gold_contract_rejected:{rejection_reason or 'unresolved'}",
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _record_id_indices(record_ids: Iterable[Any]) -> set[int]:
    indices: set[int] = set()
    for record_id in record_ids:
        match = re.search(r"(?:^|:)row:(\d+)$", _text(record_id))
        if match:
            indices.add(int(match.group(1)))
    return indices


def load_gold_vote_source_indices(
    gold_root: Path = DEFAULT_GOLD_ROOT,
    conditioned_root: Path = DEFAULT_CONDITIONED_ROOT,
) -> set[int]:
    """Load every source row that participated in any frozen/current gold vote."""

    indices: set[int] = set()
    for name in (
        "molecule_labels.jsonl",
        "conflicting_molecules.jsonl",
        "rejected_parent_molecules.jsonl",
    ):
        for row in _iter_jsonl(gold_root / name):
            indices.update(_record_id_indices(row.get("source_record_ids") or ()))
    for row in _iter_jsonl(conditioned_root / "source_condition_review.jsonl"):
        if _text(row.get("review_status")).lower() == "accepted":
            index = source_index(row)
            if index is not None:
                indices.add(index)
    return indices


@dataclass(frozen=True)
class NearDirectReview:
    source_index: int | None
    canonical_record_id: str
    decision: str
    reason: str
    support_key: str


def load_near_direct_reviews(
    path: Path = DEFAULT_REVIEW_LEDGER,
) -> dict[int | str, NearDirectReview]:
    reviews: dict[int | str, NearDirectReview] = {}
    for row in _iter_jsonl(path):
        raw_index = row.get("source_index")
        index = int(raw_index) if raw_index not in (None, "") else None
        canonical_record_id = _text(row.get("canonical_record_id"))
        if index is not None:
            review_key: int | str = index
        elif canonical_record_id:
            review_key = f"canonical_record_id:{canonical_record_id}"
        else:
            review_key = f"support_key:{_text(row.get('support_key'))}"
        if review_key in reviews:
            raise ValueError(f"duplicate near-direct review key={review_key}")
        reviews[review_key] = NearDirectReview(
            source_index=index,
            canonical_record_id=canonical_record_id,
            decision=_text(row.get("decision")),
            reason=_text(row.get("decision_reason")),
            support_key=_text(row.get("support_key")),
        )
    return reviews


class BBBSourceFamilyClassifier:
    """Record-level classifier backed by gold replay and explicit reviews."""

    def __init__(
        self,
        reviews: Mapping[int | str, NearDirectReview],
        gold_vote_indices: set[int],
        *,
        gold_decision: Callable[
            [Mapping[str, Any]], tuple[int | None, str]
        ] = gold_contract_decision,
    ) -> None:
        self.reviews = dict(reviews)
        self.gold_vote_indices = set(gold_vote_indices)
        self.gold_decision = gold_decision
        self.reviews_by_canonical_id = {
            row.canonical_record_id: row
            for row in reviews.values()
            if row.canonical_record_id
        }
        self.near_direct_support_keys = {
            row.support_key
            for row in reviews.values()
            if row.decision == "move_to_near_direct" and row.support_key
        }

    def classify_target(self, record: Mapping[str, Any]) -> FamilyMove:
        """Return the audited family and reason for every source record."""

        index = source_index(record)

        # Prediction rows are useful retrieval evidence but never experimental
        # direct evidence, even if an older frozen vote artifact included them.
        if is_explicit_prediction_record(record):
            return nondirect_target(record, "computational_or_predicted_result")

        # Frozen/current accepted-vote provenance is the authoritative proof
        # that the original full source row passed the experimental contract.
        # Normalized-v7 can omit fields that were present during gold building,
        # so replaying only its reduced surface would incorrectly demote those
        # rows. Explicit predictions were already handled above.
        if index in self.gold_vote_indices:
            return FamilyMove(
                DIRECT_GROUP,
                "frozen_or_conditioned_gold_vote_source",
            )

        decision: FamilyMove | None = None
        canonical_record_id = _text(record.get("canonical_record_id"))
        review = self.reviews.get(index) if index is not None else None
        if review is None and canonical_record_id:
            review = self.reviews_by_canonical_id.get(canonical_record_id)
        if review is not None:
            if review.decision == "move_to_near_direct":
                decision = FamilyMove(NEAR_DIRECT_GROUP, review.reason)
        elif support_key(record) in self.near_direct_support_keys:
            decision = FamilyMove(
                NEAR_DIRECT_GROUP,
                "reviewed_near_direct_cross_source_duplicate",
            )
        if decision is None:
            label, reason = self.gold_decision(record)
            if label is not None:
                decision = FamilyMove(
                    DIRECT_GROUP,
                    "gold_contract_eligible_experimental_cns_outcome",
                )
            else:
                decision = nondirect_target(record, reason)
        return decision

    def __call__(self, record: Mapping[str, Any]) -> str | FamilyMove:
        original = _text(record.get("group_id"))
        decision = self.classify_target(record)
        return "" if decision.new_group == original else decision
