"""Source-aware endpoint and low-cardinality context normalization for BBB."""

from __future__ import annotations

import json
import hashlib
import math
import re
import unicodedata
from collections import Counter
from functools import cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from data.processing.evidence_library.shared.v1.normalization.cleaning import clean_text
from data.processing.evidence_library.shared.v1.normalization.measurements import (
    EndpointOrthography,
)


ENDPOINT_NORMALIZATION_VERSION = "bbb_martins_endpoint_normalization.v4"
MISSING_ENDPOINT_MAPPING_VERSION = "bbb_martins_missing_endpoint_recovery.v1"
MISSING_ENDPOINT_PRODUCER_ID = "bbb.reviewed_missing_endpoint.v1"
DEFAULT_MISSING_ENDPOINT_MAPPING = (
    Path(__file__).resolve().parent
    / "data_processing/missing_endpoint_recovery_v1/reviewed_endpoint_mapping.parquet"
)
EXPECTED_MISSING_ENDPOINT_DECISIONS = {
    "proposed": 19_876,
    "unresolved": 493,
}
DIRECT_PERMEABILITY_ENDPOINT_PRODUCER_ID = (
    "bbb.direct.permeability_label_endpoint.v1"
)
PASSIVE_INTERPRETATION_ENDPOINT_PRODUCER_ID = (
    "bbb.passive.interpretation_endpoint.v1"
)
EFFLUX_CONCLUSION_ENDPOINT_PRODUCER_ID = (
    "bbb.efflux.interaction_conclusion_endpoint.v1"
)
DIRECT_ENDPOINT_MAPPING_VERSION = "bbb_martins_direct_endpoint.globally_reconciled.v1"
DIRECT_ENDPOINT_MAPPING_V2_VERSION = "bbb_martins_direct_endpoint.errata.v2"
SUPPORTED_DIRECT_ENDPOINT_MAPPING_VERSIONS = frozenset(
    {DIRECT_ENDPOINT_MAPPING_VERSION, DIRECT_ENDPOINT_MAPPING_V2_VERSION}
)
APPROVED_DIRECT_ENDPOINT_V1_MAPPING = (
    Path(__file__).resolve().parent
    / "data_processing/direct_endpoint_normalization_v1/approved/endpoint_mapping.json"
)
DEFAULT_APPROVED_DIRECT_ENDPOINT_MAPPING = APPROVED_DIRECT_ENDPOINT_V1_MAPPING
DIRECT_SOURCE_PATH = (
    Path(__file__).resolve().parents[7]
    / "data/raw/starling/bbb_martins/Direct_BBB/records.parquet"
)
EXPECTED_DIRECT_SOURCE_SHA256 = "14c01314377b6c31f18e6ec9c1ee72a8605a06800d74385fc8a7273f2334df57"
EXPECTED_DIRECT_INVENTORY_COUNT = 22_744
EXPECTED_DIRECT_INVENTORY_SHA256 = "2441c887b9b0c2031e0de4ab54663ca23760e97cf1f095365fe3872d375ace8e"

ENDPOINT_SOURCE_FIELD = {
    "direct_bbb": "quant_metric",
    "passive_permeability": "metric_name",
    "efflux_transport": "quantitative_metric",
    "influx_transport": "transport_endpoint",
}

_NULL_LIKE = frozenset(
    {"", "...", "{}", "n/a", "na", "none", "null", "unknown", "missing_endpoint"}
)


@cache
def _reviewed_missing_endpoint_mapping() -> dict[str, dict[str, str]]:
    columns = [
        "cleaned_record_id",
        "source_id",
        "proposed_canonical_endpoint",
        "review_status",
        "decision_basis",
        "source_evidence_field",
    ]
    frame = pd.read_parquet(DEFAULT_MISSING_ENDPOINT_MAPPING, columns=columns)
    if frame["cleaned_record_id"].duplicated().any():
        raise ValueError("duplicate reviewed missing-endpoint record ID")
    counts = Counter(str(value) for value in frame["review_status"])
    if dict(counts) != EXPECTED_MISSING_ENDPOINT_DECISIONS:
        raise ValueError(f"reviewed missing-endpoint count drift: {dict(counts)}")

    mapping: dict[str, dict[str, str]] = {}
    for row in frame.to_dict("records"):
        row = {
            key: "" if pd.isna(value) else str(value)
            for key, value in row.items()
        }
        record_id = row["cleaned_record_id"]
        status = row["review_status"]
        endpoint = row["proposed_canonical_endpoint"]
        decision = {
            "source_id": row["source_id"],
            "status": status,
            "endpoint": endpoint,
            "decision_basis": row["decision_basis"],
            "source_field": row["source_evidence_field"],
        }
        if (
            not record_id
            or not decision["source_id"]
            or not decision["decision_basis"]
            or not decision["source_field"]
            or status not in EXPECTED_MISSING_ENDPOINT_DECISIONS
            or (status == "proposed" and _snake(endpoint) != endpoint)
            or (status == "unresolved" and endpoint)
        ):
            raise ValueError(f"invalid reviewed missing-endpoint row: {record_id!r}")
        mapping[record_id] = decision
    return mapping


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_direct_endpoint_keys() -> set[str]:
    frame = pd.read_parquet(DIRECT_SOURCE_PATH, columns=["quant_metric"])
    return {
        clean_text(value) or "missing_endpoint"
        for value in frame["quant_metric"].tolist()
    } - {"missing_endpoint"}


def _key(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    text = text.replace("→", " to ").replace(">", " to ")
    text = text.replace("–", "-").replace("—", "-").replace("‑", "-")
    return re.sub(r"\s+", " ", text)


def _snake(value: Any) -> str:
    text = _key(value)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or "missing_endpoint"


def _has_direction(key: str, start: str, end: str) -> bool:
    separated = re.compile(
        rf"(?:^|[^a-z0-9]){start}\s*(?:-|/|\bto\b|2)\s*{end}(?:$|[^a-z0-9])"
    )
    if separated.search(key):
        return True
    compact = re.sub(r"[^a-z0-9]", "", key)
    # Compact AB/BA notation is accepted only directly beside a known assay
    # metric token; never match the letters inside ordinary words such as
    # apparent, incubation, or permeability.
    return bool(
        re.search(
            rf"(?:papp|meanpapp|mdck|flux){start}(?:to|2)?{end}(?:gf\d+)?$",
            compact,
        )
    )


def _has_papp_compartment_direction(key: str, start: str, end: str) -> bool:
    """Recognize AP/BL shorthand only next to a Papp metric token."""
    return bool(
        re.search(
            rf"(?:^|[^a-z0-9])papp\s*[,(_ ]*{start}\s*(?:-|/|\bto\b|2)\s*{end}(?:$|[^a-z0-9])",
            key,
        )
    )


def canonical_passive_endpoint(value: Any) -> tuple[str, str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "missing_endpoint", "passive_missing_endpoint"
    key = _key(value)
    if key in _NULL_LIKE:
        return "missing_endpoint", "passive_missing_endpoint"
    a_to_b = (
        _has_direction(key, "a", "b")
        or _has_papp_compartment_direction(key, "ap", "bl")
        or any(
        marker in key
        for marker in (
            "apical to basal",
            "apical-to-basal",
            "apical to basolateral",
            "apical-to-basolateral",
        )
        )
    )
    b_to_a = (
        _has_direction(key, "b", "a")
        or _has_papp_compartment_direction(key, "bl", "ap")
        or any(
        marker in key for marker in ("basolateral to apical", "basolateral-to-apical")
        )
    )
    compact = re.sub(r"[^a-z0-9]+", "", key)
    reviewed_exact = {
        "peff": "effective_permeability",
        "p_eff": "effective_permeability",
        "pexact": "pexact",
        "mean pe": "effective_permeability",
        "bbb-pe": "effective_permeability",
        "pe_cell": "pe_cell",
        "p_dif": "p_dif",
        "pap-bl": "papp_a_to_b",
        "pampa bbb pm 7.4": "passive_permeability",
        "pe (a to b)": "effective_permeability_a_to_b",
        "mdck_ab_c": "papp_a_to_b",
        "ppe": "negative_log_effective_permeability",
        "ppe bbb": "negative_log_effective_permeability",
        "pincre(corrected)app": "pincre_corrected_app",
        "pincreapp": "pincre_app",
        "rate (apical to basolateral)": "transport_rate_a_to_b",
        "rate (basolateral to apical)": "transport_rate_b_to_a",
        "kintr": "intrinsic_partition_coefficient",
        "mdck-mdr1 ab": "papp_a_to_b",
        "pcall": "pcall",
        "a-b transport": "transport_rate_a_to_b",
        "apical-basolateral (a-b) [nm/s]": "transport_rate_a_to_b",
        "basolateral-apical (b-a) [nm/s]": "transport_rate_b_to_a",
    }
    if key in reviewed_exact:
        endpoint = reviewed_exact[key]
        reason = (
            "passive_literal_review_required"
            if key in {"pexact", "pe_cell", "p_dif", "pincre(corrected)app", "pincreapp", "pcall"}
            else "passive_reviewed_exact"
        )
        return endpoint, reason
    if "effluxratio" in compact or key in {"ba/ab", "b:a ratio", "a-b/b-a ratio"}:
        return "efflux_ratio", "passive_efflux_ratio"
    if "papp" in compact or "apparent permeability" in key or "paap" in compact:
        if "log" in compact:
            return "log_apparent_permeability", "passive_log_papp"
        if a_to_b:
            return "papp_a_to_b", "passive_papp_a_to_b"
        if b_to_a:
            return "papp_b_to_a", "passive_papp_b_to_a"
        return "apparent_permeability", "passive_papp"
    if "permeability" in key and "log" in key:
        return "log_passive_permeability", "passive_log_permeability"
    if re.search(r"(?:^|[^a-z])(?:-\s*)?log\s*p?e(?:[^a-z]|$)", key) or compact in {
        "logpe", "logpeexp", "logpₑ", "plogpe"
    }:
        if key.lstrip().startswith(("-log", "− log")) or compact.startswith("log") is False:
            return "negative_log_effective_permeability", "passive_negative_log_pe"
        return "log_effective_permeability", "passive_log_pe"
    if re.match(r"^(?:-|−)\s*log\s*p(?:$|[^a-z])", key):
        return "negative_log_partition_coefficient", "passive_negative_log_partition"
    if compact in {"pe", "peexp", "peexact", "peeff", "peffective", "pestar"} or (
        "effective permeability" in key
    ):
        return "effective_permeability", "passive_effective_permeability"
    if any(marker in key for marker in ("percent", "percentage", "%", "dose permeated")):
        if a_to_b or "apical to basal" in key:
            return "percent_transport_a_to_b", "passive_percent_transport_a_to_b"
        if "dose" in key:
            return "percent_dose_permeated", "passive_percent_dose"
        return "percent_transport", "passive_percent_transport"
    if "fold" in key and "concentration" in key:
        return "concentration_fold_change", "passive_concentration_fold_change"
    if "fold" in key and "permeab" in key:
        return "permeability_fold_change", "passive_fold_change"
    if re.search(r"(?<![a-z])ratio(?![a-z])", key) or key in {"aq", "t", "mdck a/b", "mdck b/a"}:
        return "relative_permeability_ratio", "passive_relative_ratio"
    if any(marker in key for marker in ("flux", "transport rate", "permeation rate", "permeability rate")):
        if a_to_b:
            return "transport_rate_a_to_b", "passive_transport_rate_a_to_b"
        if b_to_a:
            return "transport_rate_b_to_a", "passive_transport_rate_b_to_a"
        return "transport_rate", "passive_transport_rate"
    if compact in {
        "logp",
        "logp0",
        "logp0pampabbb",
        "logpopampabbb",
        "logd74",
        "logk",
    } or "logkmemb" in compact or any(
        marker in key for marker in ("log p", "logd", "log po")
    ):
        return "log_partition_coefficient", "passive_log_partition_coefficient"
    if compact == "kp" or "partition coefficient" in key:
        return "partition_coefficient", "passive_partition_coefficient"
    if "kiam" in compact:
        return "membrane_partitioning", "passive_literal_review_required"
    if "clearance" in key or key == "cl":
        return "clearance", "passive_clearance"
    if "concentration" in key:
        return "concentration", "passive_concentration"
    if "class" in key or "score" in key or "classification" in key:
        return "permeability_class", "passive_permeability_class"
    if any(marker in key for marker in ("permeab", "permeation")) or compact in {
        "p", "pap", "pampa", "pc", "pm", "pm74", "ppass", "rrck"
    }:
        return "passive_permeability", "passive_generic_permeability"
    return _snake(value), "passive_literal_fallback"


_EFFLUX_ENDPOINTS = {
    "auc10-30": "brain_exposure_auc",
    "ec50": "ec50",
    "ic50": "ic50",
    "km": "michaelis_menten_km",
    "kp_brain": "kp_brain",
    "ld50": "ld50",
    "brain_exposure_fold_change": "brain_exposure_fold_change",
    "brain_to_blood_ratio": "brain_to_blood_ratio",
    "brain_to_plasma_ratio": "brain_to_plasma_ratio",
    "clearance": "clearance",
    "efflux": "efflux",
    "efflux_rate": "efflux_rate",
    "efflux_ratio": "efflux_ratio",
    "fold change": "fold_change",
    "fold_change": "fold_change",
    "inhibition_potency": "inhibition_potency",
    "other": "other_quantitative_endpoint",
    "percent_inhibition": "percent_inhibition",
    "permeability": "permeability",
    "probability": "probability",
}


def canonical_efflux_endpoint(value: Any) -> tuple[str, str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "missing_endpoint", "efflux_missing_endpoint"
    key = _key(value)
    if key in _NULL_LIKE:
        return "missing_endpoint", "efflux_missing_endpoint"
    endpoint = _EFFLUX_ENDPOINTS.get(key, _snake(value))
    return endpoint, "efflux_reviewed_exact" if key in _EFFLUX_ENDPOINTS else "efflux_literal_fallback"


_INFLUX_ENDPOINTS = {
    "bbb substrate relationship": "bbb_substrate_relationship",
    "bbb transport": "bbb_transport",
    "blood-csf barrier transport": "blood_csf_barrier_transport",
    "blood-to-cns transport": "blood_to_cns_transport",
    "blood-to-csf transport": "blood_to_csf_transport",
    "blood-to-brain transport": "blood_to_brain_transport",
    "blood-to-brain transport (implied)": "blood_to_brain_transport_implied",
    "blood-to-spinal cord transport": "blood_to_spinal_cord_transport",
    "brain delivery": "brain_delivery",
    "brain endothelial cell uptake": "brain_endothelial_cell_uptake",
    "brain endothelial cell uptake and luminal-to-abluminal transport": (
        "brain_endothelial_uptake_and_luminal_to_abluminal_transport"
    ),
    "brain uptake": "brain_uptake",
    "luminal-to-abluminal transport": "luminal_to_abluminal_transport",
    "unclear (increased striatal glucose levels, possibly due to increased bbb transport)": (
        "context_dependent_bbb_transport"
    ),
    "unqualified bbb transport": "unqualified_bbb_transport",
    "unspecified bbb influx": "unspecified_bbb_influx",
}


def canonical_influx_endpoint(value: Any) -> tuple[str, str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "missing_endpoint", "influx_missing_endpoint"
    key = _key(value)
    if key in _NULL_LIKE:
        return "missing_endpoint", "influx_missing_endpoint"
    endpoint = _INFLUX_ENDPOINTS.get(key, _snake(value))
    return endpoint, "influx_reviewed_exact" if key in _INFLUX_ENDPOINTS else "influx_literal_fallback"


def canonical_assay_type(value: Any) -> str | None:
    key = _key(value)
    return {
        "mdck": "mdck",
        "pampa-bbb": "pampa_bbb",
        "pampa_other": "pampa_other",
        "brain_endothelial_cell": "brain_endothelial_cell",
        "experimental_passive_conclusion": "experimental_passive_conclusion",
        "membrane_partitioning": "membrane_partitioning",
        "other_cell_monolayer": "other_cell_monolayer",
    }.get(key)


def canonical_evidence_type(value: Any) -> str | None:
    key = _key(value)
    if key in _NULL_LIKE:
        return None
    if key in {"qualitative transpor_claim", "qualitative_transporter_claim"}:
        return "qualitative_transporter_claim"
    return _snake(value)


def canonical_transport_mechanism(value: Any) -> str | None:
    key = _key(value)
    if key in _NULL_LIKE:
        return None
    key = key.replace("carrier-mediated influx (active transport)", "carrier-mediated influx")
    key = key.replace("transporter-mediated endothelial uptake", "carrier-mediated endothelial uptake")
    return _snake(key)


class EndpointNormalizer:
    """Apply deterministic mappings plus an explicitly approved Direct map."""

    def __init__(self, direct_mapping_path: str | Path | None = None):
        self.path = Path(direct_mapping_path) if direct_mapping_path else None
        self.direct_mapping: dict[str, str] = {}
        self.approval: dict[str, Any] | None = None
        self.mapping_sha256: str | None = None
        self.mapping_version: str | None = None
        if self.path is not None and self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            mapping_version = str(payload.get("mapping_version") or "")
            if mapping_version not in SUPPORTED_DIRECT_ENDPOINT_MAPPING_VERSIONS:
                raise ValueError("Direct endpoint mapping version mismatch")
            approval = payload.get("approval")
            if not isinstance(approval, Mapping) or approval.get("human_approved") is not True:
                raise ValueError("Direct endpoint mapping is not human approved")
            if not str(approval.get("approved_by") or "") or not str(
                approval.get("approved_at") or ""
            ):
                raise ValueError("Direct endpoint approval lacks approver or timestamp")
            source = payload.get("source")
            if not isinstance(source, Mapping):
                raise ValueError("Direct endpoint mapping lacks source provenance")
            actual_source_sha = _file_sha256(DIRECT_SOURCE_PATH)
            expected_source = {
                "sha256": EXPECTED_DIRECT_SOURCE_SHA256,
                "inventory_count_including_missing": EXPECTED_DIRECT_INVENTORY_COUNT,
                "inventory_sha256": EXPECTED_DIRECT_INVENTORY_SHA256,
            }
            if actual_source_sha != EXPECTED_DIRECT_SOURCE_SHA256:
                raise ValueError("pinned Direct source digest drift")
            if any(source.get(key) != value for key, value in expected_source.items()):
                raise ValueError("Direct endpoint mapping source provenance mismatch")
            mapping = payload.get("mapping")
            if not isinstance(mapping, Mapping) or not mapping:
                raise ValueError("Direct endpoint mapping is empty")
            self.direct_mapping = {str(key): str(value) for key, value in mapping.items()}
            expected_keys = _expected_direct_endpoint_keys()
            if set(self.direct_mapping) != expected_keys:
                raise ValueError(
                    "Direct endpoint mapping key coverage mismatch: "
                    f"missing={len(expected_keys - set(self.direct_mapping))} "
                    f"extra={len(set(self.direct_mapping) - expected_keys)}"
                )
            if any(not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", value) for value in self.direct_mapping.values()):
                raise ValueError("Direct endpoint mapping contains a non-canonical label")
            self.approval = dict(approval)
            self.mapping_sha256 = _file_sha256(self.path)
            self.mapping_version = mapping_version
            if mapping_version == DIRECT_ENDPOINT_MAPPING_V2_VERSION:
                provenance = payload.get("provenance")
                if not isinstance(provenance, Mapping):
                    raise ValueError("Direct endpoint v2 lacks base-map provenance")
                expected_base_sha = _file_sha256(APPROVED_DIRECT_ENDPOINT_V1_MAPPING)
                if provenance.get("base_mapping_sha256") != expected_base_sha:
                    raise ValueError("Direct endpoint v2 base-map provenance mismatch")

    @property
    def direct_ready(self) -> bool:
        return bool(self.direct_mapping)

    def decision(self, source_id: str, endpoint_name: str) -> EndpointOrthography:
        cleaned = clean_text(endpoint_name) or "missing_endpoint"
        if source_id == "direct_bbb":
            if cleaned == "missing_endpoint":
                endpoint, rule = cleaned, "direct_missing_endpoint"
            elif self.direct_mapping:
                if cleaned not in self.direct_mapping:
                    raise KeyError(f"approved Direct endpoint mapping lacks {cleaned!r}")
                endpoint, rule = self.direct_mapping[cleaned], "direct_approved_global_mapping"
            else:
                endpoint, rule = cleaned, "direct_mapping_awaiting_human_approval"
        elif source_id == "passive_permeability":
            endpoint, rule = canonical_passive_endpoint(cleaned)
        elif source_id == "efflux_transport":
            endpoint, rule = canonical_efflux_endpoint(cleaned)
        elif source_id == "influx_transport":
            endpoint, rule = canonical_influx_endpoint(cleaned)
        else:
            endpoint, rule = cleaned, "unsupported_source"
        changed = endpoint != cleaned
        return EndpointOrthography(
            endpoint_name=cleaned,
            spacing_and_spelling_endpoint=endpoint,
            status="reviewed_normalization" if changed else "unchanged",
            reason=rule,
            spacing_and_spelling_version=ENDPOINT_NORMALIZATION_VERSION,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "version": ENDPOINT_NORMALIZATION_VERSION,
            "endpoint_source_field": ENDPOINT_SOURCE_FIELD,
            "direct_mapping_path": str(self.path) if self.path else None,
            "direct_mapping_loaded": self.direct_ready,
            "direct_mapping_version": self.mapping_version,
            "direct_mapping_entries": len(self.direct_mapping),
            "direct_mapping_sha256": self.mapping_sha256,
            "direct_source_sha256": EXPECTED_DIRECT_SOURCE_SHA256,
            "direct_inventory_count_including_missing": EXPECTED_DIRECT_INVENTORY_COUNT,
            "direct_inventory_sha256": EXPECTED_DIRECT_INVENTORY_SHA256,
            "direct_human_approval": self.approval,
            "api_llm_sources": ["direct_bbb"],
            "deterministic_sources": ["passive_permeability", "efflux_transport", "influx_transport"],
        }


def context_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    source_id = str(record.get("source_id") or "")
    output = {
        "canonical_endpoint_source_field": ENDPOINT_SOURCE_FIELD.get(source_id),
        "canonical_endpoint_policy_status": str(record.get("spacing_and_spelling_status") or ""),
        "canonical_endpoint_rule_id": str(record.get("spacing_and_spelling_reason") or ""),
        "canonical_endpoint_policy_version": ENDPOINT_NORMALIZATION_VERSION,
        "canonical_assay_type": None,
        "canonical_evidence_type": None,
        "canonical_transport_mechanism": None,
    }
    if source_id == "passive_permeability":
        output["canonical_assay_type"] = canonical_assay_type(record.get("assay_type"))
    elif source_id == "efflux_transport":
        output["canonical_evidence_type"] = canonical_evidence_type(record.get("evidence_type"))
    elif source_id == "influx_transport":
        output["canonical_transport_mechanism"] = canonical_transport_mechanism(
            record.get("transport_mechanism")
        )
    return output


def alternate_endpoint_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve a missing primary endpoint from one approved structured field."""
    current = str(
        record.get("canonical_endpoint")
        or record.get("canonical_endpoint_name")
        or ""
    ).strip().casefold()
    if current not in _NULL_LIKE:
        return {}

    source_id = str(record.get("source_id") or "")
    reviewed = _reviewed_missing_endpoint_mapping().get(
        str(record.get("cleaned_record_id") or "")
    )
    if reviewed is not None:
        if reviewed["source_id"] != source_id:
            raise ValueError("reviewed missing-endpoint source mismatch")
        if reviewed["status"] == "unresolved":
            return {}
        endpoint = reviewed["endpoint"]
        source_field = reviewed["source_field"]
        producer_id = MISSING_ENDPOINT_PRODUCER_ID
        rule_id = f"reviewed_missing_endpoint_{reviewed['decision_basis']}"
    elif (
        source_id == "direct_bbb"
        and record.get("categorical_encoder_id") == "bbb_permeability_binary.v1"
    ):
        endpoint = "bbb_permeability_outcome"
        source_field = "bbb_permeability_label"
        producer_id = DIRECT_PERMEABILITY_ENDPOINT_PRODUCER_ID
        rule_id = "direct_permeability_label_outcome"
    elif source_id == "passive_permeability":
        label = _snake(record.get("passive_bbb_interpretation"))
        if label not in {"permeable_or_high", "impermeable_or_low"}:
            return {}
        endpoint = "passive_bbb_permeability_outcome"
        source_field = "passive_bbb_interpretation"
        producer_id = PASSIVE_INTERPRETATION_ENDPOINT_PRODUCER_ID
        rule_id = "passive_interpretation_outcome"
    elif source_id == "efflux_transport":
        label = _snake(record.get("interaction_conclusion"))
        endpoint_by_label = {
            "substrate": "efflux_substrate_outcome",
            "non_substrate": "efflux_substrate_outcome",
            "inhibitor": "efflux_inhibition_outcome",
            "non_inhibitor": "efflux_inhibition_outcome",
            "transporter_limited_brain_exposure": (
                "transporter_limited_brain_exposure_outcome"
            ),
        }
        endpoint = endpoint_by_label.get(label)
        if endpoint is None:
            return {}
        source_field = "interaction_conclusion"
        producer_id = EFFLUX_CONCLUSION_ENDPOINT_PRODUCER_ID
        rule_id = f"efflux_interaction_conclusion_{label}"
    else:
        return {}

    return {
        "canonical_endpoint": endpoint,
        "spacing_and_spelling_endpoint": endpoint,
        "spacing_and_spelling_status": "reviewed_structured_fallback",
        "spacing_and_spelling_reason": rule_id,
        "spacing_and_spelling_version": ENDPOINT_NORMALIZATION_VERSION,
        "canonical_endpoint_source_field": source_field,
        "canonical_endpoint_policy_status": "reviewed_structured_fallback",
        "canonical_endpoint_rule_id": rule_id,
        "canonical_endpoint_policy_version": ENDPOINT_NORMALIZATION_VERSION,
        "canonical_endpoint_producer_id": producer_id,
    }


def validate_missing_endpoint_mapping(
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require every reviewed decision to be applied or preserved unresolved."""
    mapping = _reviewed_missing_endpoint_mapping()
    indexed = {
        str(record.get("cleaned_record_id") or ""): record
        for record in records
        if str(record.get("cleaned_record_id") or "") in mapping
    }
    if set(indexed) != set(mapping):
        raise ValueError(
            f"reviewed missing-endpoint coverage drift: "
            f"expected={len(mapping):,} found={len(indexed):,}"
        )
    for record_id, decision in mapping.items():
        record = indexed[record_id]
        endpoint = str(
            record.get("canonical_endpoint")
            or record.get("canonical_endpoint_name")
            or ""
        )
        if decision["status"] == "proposed":
            valid = (
                endpoint == decision["endpoint"]
                and record.get("canonical_endpoint_producer_id")
                == MISSING_ENDPOINT_PRODUCER_ID
            )
        else:
            valid = endpoint == "missing_endpoint"
        if not valid:
            raise ValueError(
                f"reviewed missing-endpoint decision was not preserved: {record_id}"
            )
    return {
        "mapping_version": MISSING_ENDPOINT_MAPPING_VERSION,
        "mapping_path": str(DEFAULT_MISSING_ENDPOINT_MAPPING),
        "records": len(mapping),
        "status_counts": dict(EXPECTED_MISSING_ENDPOINT_DECISIONS),
        "validations": {
            "all_decisions_present": True,
            "all_proposals_applied": True,
            "all_unresolved_preserved": True,
        },
    }


__all__ = [
    "DEFAULT_APPROVED_DIRECT_ENDPOINT_MAPPING",
    "APPROVED_DIRECT_ENDPOINT_V1_MAPPING",
    "DIRECT_ENDPOINT_MAPPING_VERSION",
    "DIRECT_ENDPOINT_MAPPING_V2_VERSION",
    "DIRECT_PERMEABILITY_ENDPOINT_PRODUCER_ID",
    "DEFAULT_MISSING_ENDPOINT_MAPPING",
    "EFFLUX_CONCLUSION_ENDPOINT_PRODUCER_ID",
    "ENDPOINT_NORMALIZATION_VERSION",
    "ENDPOINT_SOURCE_FIELD",
    "MISSING_ENDPOINT_MAPPING_VERSION",
    "MISSING_ENDPOINT_PRODUCER_ID",
    "EndpointNormalizer",
    "PASSIVE_INTERPRETATION_ENDPOINT_PRODUCER_ID",
    "alternate_endpoint_fields",
    "canonical_assay_type",
    "canonical_efflux_endpoint",
    "canonical_evidence_type",
    "canonical_influx_endpoint",
    "canonical_passive_endpoint",
    "canonical_transport_mechanism",
    "context_fields",
    "validate_missing_endpoint_mapping",
]
