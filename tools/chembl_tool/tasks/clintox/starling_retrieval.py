"""Build ClinTox literature retrieval for the strict trial-failure benchmark.

The AACT/FDA benchmark remains the only label source.  This module builds an
independent analog-evidence library from ``clintox_base_v1`` and the six
Starling toxicity families.  Only literal trial/development-level stoppage
caused by toxicity is tagged as direct evidence; patient-level treatment
discontinuation and broad toxicity findings remain contextual evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.common.evidence_contract import attach_minimal_evidence
from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic
from tools.chembl_tool.common.starling import (
    StarlingSourceProfile,
    build_and_write_starling_index,
    build_starling_parquet_evidence_rows,
)


SOURCE_VERSION = "clintox_starling_clinical_trial_failure_retrieval.v1"
DIRECT_GATE_VERSION = "literal_toxicity_trial_failure.v4"
DEFAULT_DATA_ROOT = Path("data/starling_data/clintox")
DEFAULT_CANONICAL_ROOT = DEFAULT_DATA_ROOT / "canonical_retrieval_v1"
DEFAULT_MECHANISM_CATALOG = Path(
    "outputs/chembl_tool/tasks/clintox/evidence_library/"
    "starling_raw_v1/03_evidence_catalog/evidence.jsonl"
)
DEFAULT_OUT_DIR = Path(
    "outputs/chembl_tool/tasks/clintox/evidence_library/"
    "starling_clinical_trial_failure_v1"
)
EVIDENCE_FILENAME = "starling_clintox_evidence.jsonl"
INDEX_FILENAME = "starling_clintox_neighbor_index.pkl"
META_FILENAME = "starling_clintox_neighbor_index.meta.json"
INDEX_VERSION = "clintox_starling_clinical_trial_failure_neighbor_index.v1"
EXPECTED_MECHANISM_CATALOG_SHA256 = (
    "061abf4baf11686a8b58c2324f4f54708df8327a791b65dbec29d9e71b62046f"
)

DIRECT_GROUP = "Direct.clinical_trial_failure"
CLINICAL_CONTEXT_GROUP = "Mechanism.clinical_human_safety"
MECHANISM_FAMILIES = (
    "clinical_human_safety",
    "in_vivo_toxicology",
    "organ_specific_toxicity",
    "genotoxicity_carcinogenicity",
    "cellular_stress_pathways",
    "general_cytotoxicity",
    "off_target_ddi_exposure",
)

_DEVELOPMENT_SUBJECT = (
    r"(?:clinical\s+development(?:\s+program)?|further\s+(?:drug\s+)?development|"
    r"drug\s+development|development\s+program|development|drug\s+program)"
)
_TRIAL_SUBJECT = (
    r"(?:(?:phase\s+[ivx0-9]+\s+)?(?:clinical\s+)?trials?|"
    r"(?:phase\s+[ivx0-9]+\s+)?clinical\s+stud(?:y|ies)|stud(?:y|ies)|"
    r"recruitment|enrollment|enrolment|accrual)"
)
_SUBJECT = rf"(?:{_DEVELOPMENT_SUBJECT}|{_TRIAL_SUBJECT})"
_TOXIC_CAUSE = (
    r"(?:severe\s+adverse\s+events?|unexpected\s+adverse\s+events?|"
    r"(?:severe\s+|serious\s+|dose[- ]related\s+)?(?:side|adverse)\s+effects?|"
    r"toxicit(?:y|ies)|toxic\s+deaths?|safety\s+(?:concerns?|issues?|"
    r"problems?|signals?|complications?)|intolerab(?:le|ility)|"
    r"dose[- ]limiting\s+toxicit(?:y|ies)|hepatotoxicit(?:y|ies)|"
    r"cardiotoxicit(?:y|ies)|nephrotoxicit(?:y|ies)|"
    r"neurotoxicit(?:y|ies)|hepatic\s+toxicit(?:y|ies))"
)
_TRIAL_VERB_EVENT = r"(?:terminated|suspended|halted|stopped|closed|abandoned)"
_DEVELOPMENT_VERB_EVENT = rf"(?:{_TRIAL_VERB_EVENT}|discontinued)"
_TRIAL_NOUN_EVENT = r"(?:termination|suspension|halt|stoppage|closure|abandonment)"
_DEVELOPMENT_NOUN_EVENT = rf"(?:{_TRIAL_NOUN_EVENT}|discontinuation)"
_VERB_EVENT = rf"(?:{_DEVELOPMENT_VERB_EVENT})"
_NOUN_EVENT = rf"(?:{_DEVELOPMENT_NOUN_EVENT})"
_CAUSAL_CONNECTOR = r"(?:due\s+to|because\s+of|owing\s+to|on\s+account\s+of|following|after|for)"
_CAUSAL_VERB = r"(?:led\s+to|prompted|caused|resulted\s+in|necessitated|triggered)"

_DIRECT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # Trial/study-level events need a finite stoppage verb.  Bare phrases
        # such as "study discontinuation due to toxicity" commonly describe
        # patient treatment discontinuation and are intentionally excluded.
        rf"\b{_TRIAL_SUBJECT}\b\s+(?:was\s+|were\s+|has\s+been\s+|had\s+been\s+)?"
        rf"(?:prematurely\s+|early\s+)?\b{_TRIAL_VERB_EVENT}\b.{{0,80}}"
        rf"\b{_CAUSAL_CONNECTOR}\b.{{0,100}}\b{_TOXIC_CAUSE}\b",
        # Development/program discontinuation is itself the target event.
        rf"\b{_DEVELOPMENT_SUBJECT}\b\s+(?:was\s+|has\s+been\s+|had\s+been\s+)?"
        rf"(?:prematurely\s+|early\s+)?\b{_DEVELOPMENT_VERB_EVENT}\b.{{0,80}}"
        rf"\b{_CAUSAL_CONNECTOR}\b.{{0,100}}\b{_TOXIC_CAUSE}\b",
        rf"\b{_TRIAL_SUBJECT}\s+{_TRIAL_NOUN_EVENT}\b.{{0,80}}"
        rf"\b{_CAUSAL_CONNECTOR}\b.{{0,100}}\b{_TOXIC_CAUSE}\b",
        rf"\b{_DEVELOPMENT_SUBJECT}\s+{_DEVELOPMENT_NOUN_EVENT}\b.{{0,80}}"
        rf"\b{_CAUSAL_CONNECTOR}\b.{{0,100}}\b{_TOXIC_CAUSE}\b",
        # Reverse causal wording must terminate in one compact, explicit
        # trial/development stoppage construction; do not bridge unrelated
        # clauses containing a generic causal verb.
        rf"\b{_TOXIC_CAUSE}\b.{{0,80}}\b{_CAUSAL_VERB}\b.{{0,20}}(?:"
        rf"(?:the\s+)?{_SUBJECT}\s+(?:to\s+be\s+)?(?:prematurely\s+|early\s+)?{_VERB_EVENT}|"
        rf"(?:an?\s+|the\s+)?(?:early\s+|premature\s+)?{_TRIAL_SUBJECT}\s+{_TRIAL_NOUN_EVENT}|"
        rf"(?:the\s+)?{_DEVELOPMENT_NOUN_EVENT}\s+of\s+(?:its\s+|the\s+)?{_DEVELOPMENT_SUBJECT}"
        rf")\b",
    )
)
_NEGATED_EVENT = re.compile(
    rf"\b(?:no|not|without|did\s+not)\b.{{0,50}}\b(?:{_VERB_EVENT}|{_NOUN_EVENT})\b",
    re.IGNORECASE,
)
_EFFICACY_FAILURE = re.compile(
    r"\b(?:trial|study)\s+failed\s+to\s+(?:show|demonstrate|meet|improve|achieve)\b",
    re.IGNORECASE,
)
_NON_TOXIC_CAUSE = re.compile(
    r"\b(?:reason|discontinuation|termination|closure|stoppage|terminated|stopped|closed)\b.{0,80}"
    r"\b(?:not\s+attributed\s+to|not\s+due\s+to|unrelated\s+to|rather\s+than)\b.{0,30}"
    r"\b(?:toxicit(?:y|ies)|safety)\b",
    re.IGNORECASE,
)
_INDIVIDUAL_DISCONTINUATION = re.compile(
    r"\b(?:patients?|participants?|subjects?|individuals?)\b.{0,100}"
    r"\b(?:withdrew|withdrawn|discontinued|removed|taken\s+off)\b|"
    r"\b(?:study|trial)\s+(?:drug|therapy|treatment|medication)\b.{0,60}"
    r"\b(?:withdrawn|discontinued|stopped)\b|"
    r"\b(?:study|trial)\s+terminated\s+(?:the\s+)?treatment\b|"
    r"\b(?:study|trial)\b.{0,40}\b(?:terminated|stopped|discontinued)\b"
    r".{0,30}\b(?:in|for)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+patients?\b",
    re.IGNORECASE,
)


def trial_failure_gate_reason(row: Mapping[str, Any]) -> str:
    """Return the literal direct-evidence decision for one base-source row."""
    if bool(row.get("needs_more_context")):
        return "reject_needs_more_context"
    required = ("SMILES", "pmid", "support_text", "toxicity_outcome")
    if any(not _text(row.get(field)) for field in required):
        return "reject_missing_required_field"
    outcome = _text(row.get("toxicity_outcome"))
    if _NEGATED_EVENT.search(outcome):
        return "reject_negated_failure_event"
    if _EFFICACY_FAILURE.search(outcome):
        return "reject_efficacy_failure_not_toxicity_failure"
    if _NON_TOXIC_CAUSE.search(outcome):
        return "reject_failure_explicitly_not_toxicity_caused"
    if _INDIVIDUAL_DISCONTINUATION.search(outcome):
        return "reject_patient_or_treatment_discontinuation"
    if any(pattern.search(outcome) for pattern in _DIRECT_PATTERNS):
        return "accept_literal_toxicity_trial_failure"
    return "reject_no_literal_toxicity_trial_failure"


def is_strict_trial_failure_record(row: Mapping[str, Any]) -> bool:
    return trial_failure_gate_reason(row).startswith("accept_")


def is_clinical_context_record(row: Mapping[str, Any]) -> bool:
    """Keep complete human-toxicity claims that are not strict direct events."""
    if bool(row.get("needs_more_context")):
        return False
    if any(not _text(row.get(field)) for field in ("SMILES", "pmid", "support_text", "toxicity_outcome")):
        return False
    return not is_strict_trial_failure_record(row)


def build_retrieval_library(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    mechanism_catalog: Path = DEFAULT_MECHANISM_CATALOG,
    canonical_root: Path = DEFAULT_CANONICAL_ROOT,
    out_dir: Path = DEFAULT_OUT_DIR,
    max_rows_per_source: int = 0,
    max_record_examples: int = 6,
    workers: int = 1,
    progress_every: int = 10_000,
) -> dict[str, Any]:
    source_receipt = verify_source_manifests(data_root)
    direct_audit = audit_direct_gate(
        data_root / "clintox_base_v1" / "extractions.parquet",
        canonical_root=canonical_root,
        max_rows=max_rows_per_source,
    )
    base_rows, base_stats = build_starling_parquet_evidence_rows(
        clinical_profiles(data_root, max_rows=max_rows_per_source),
        max_record_examples=max_record_examples,
    )
    base_rows = [_annotate_base_uncertainty(row) for row in base_rows]

    if mechanism_catalog.is_file() and not max_rows_per_source:
        observed = sha256_file(mechanism_catalog)
        if observed != EXPECTED_MECHANISM_CATALOG_SHA256:
            raise ValueError(
                f"unexpected historical mechanism catalog SHA-256: {observed}"
            )
        mechanism_rows = load_historical_mechanism_catalog(mechanism_catalog)
        mechanism_stats: dict[str, Any] = {
            "mode": "verified_historical_catalog_reuse",
            "path": str(mechanism_catalog),
            "sha256": observed,
            "n_evidence_rows": len(mechanism_rows),
        }
    else:
        mechanism_rows, mechanism_stats = build_starling_parquet_evidence_rows(
            mechanism_profiles(data_root, max_rows=max_rows_per_source),
            max_record_examples=max_record_examples,
        )
        mechanism_stats["mode"] = "rebuilt_from_canonical_raw_parquets"

    evidence_rows = [*base_rows, *mechanism_rows]
    group_counts = Counter(str(row.get("group_id") or "") for row in evidence_rows)
    required_groups = {
        DIRECT_GROUP,
        *(f"Mechanism.{family}" for family in MECHANISM_FAMILIES),
    }
    missing = sorted(required_groups - set(group_counts))
    if missing:
        raise AssertionError(f"ClinTox retrieval is missing required groups: {missing}")

    source = {
        "type": "starling_profile_index",
        "dataset": "ClinTox strict-trial-failure analog evidence",
        "source_version": SOURCE_VERSION,
        "gold_source_used_as_evidence": False,
        "direct_gate_version": DIRECT_GATE_VERSION,
        "families": list(MECHANISM_FAMILIES),
        "exact_query_exclusion": True,
        "required_query_time_policy": "parent_disjoint",
    }
    meta = build_and_write_starling_index(
        evidence_rows,
        out_dir=out_dir,
        index_version=INDEX_VERSION,
        source=source,
        source_stats={
            "source_receipt": source_receipt,
            "direct_gate": direct_audit,
            "clinical_base": base_stats,
            "mechanisms": mechanism_stats,
            "group_counts": dict(sorted(group_counts.items())),
        },
        evidence_filename=EVIDENCE_FILENAME,
        index_filename=INDEX_FILENAME,
        meta_filename=META_FILENAME,
        workers=workers,
        progress_every=progress_every,
    )
    result = {
        **meta,
        "canonical_root": str(canonical_root),
        "direct_gate_status": direct_audit["status"],
        "gold_labels_created_from_starling_rows": 0,
    }
    write_json_atomic(canonical_root / "retrieval_build_receipt.json", result)
    return result


def clinical_profiles(data_root: Path, *, max_rows: int = 0) -> list[StarlingSourceProfile]:
    path = data_root / "clintox_base_v1" / "extractions.parquet"
    common = dict(
        path=str(path),
        smiles_field="SMILES",
        context_fields=(
            "toxicity_category",
            "outcome_measure",
            "clinical_context",
            "dose_or_exposure",
            "fda_approval_status",
            "approved_indication",
            "extra_details",
        ),
        scope_fields=("toxicity_category", "clinical_context", "dose_or_exposure"),
        name_fields=("molecule_name",),
        extra_example_fields=(
            "toxicity_category",
            "outcome_measure",
            "clinical_context",
            "dose_or_exposure",
            "fda_approval_status",
            "needs_more_context",
        ),
        max_rows=max_rows,
    )
    return [
        StarlingSourceProfile(
            source_id="clintox_literal_trial_failure",
            group_id=DIRECT_GROUP,
            assay_tier="Direct",
            endpoint_group="clinical_trial_failure",
            evidence_source="Starling/ClinTox/clintox_base_v1",
            endpoint_field="toxicity_outcome",
            target_pref_name="toxicity-caused clinical trial or development failure",
            evidence_role="direct_outcome",
            standard_type_prefix="literal toxicity-caused trial failure",
            record_filter=is_strict_trial_failure_record,
            record_filter_name=DIRECT_GATE_VERSION,
            **common,
        ),
        StarlingSourceProfile(
            source_id="clintox_clinical_context",
            group_id=CLINICAL_CONTEXT_GROUP,
            assay_tier="Mechanism",
            endpoint_group="clinical_human_safety",
            evidence_source="Starling/ClinTox/clintox_base_v1",
            endpoint_field="toxicity_category",
            target_pref_name="human clinical toxicity and regulatory context",
            evidence_role="context_modifier",
            standard_type_prefix="human clinical safety context",
            record_filter=is_clinical_context_record,
            record_filter_name="complete_non_direct_clinical_context.v1",
            **common,
        ),
    ]


def mechanism_profiles(data_root: Path, *, max_rows: int = 0) -> list[StarlingSourceProfile]:
    raw = data_root / "raw_v1"
    profiles = [
        StarlingSourceProfile(
            source_id="clintox_nonclinical_in_vivo",
            path=str(raw / "nonclinical_in_vivo_toxicity" / "extractions.parquet"),
            group_id="Mechanism.in_vivo_toxicology",
            assay_tier="Mechanism",
            endpoint_group="in_vivo_toxicology",
            evidence_source="Starling/ClinTox/nonclinical_in_vivo_toxicity",
            endpoint_field="observed_effect",
            smiles_field="SMILES",
            value_field="endpoint_value",
            unit_field="endpoint_unit",
            context_fields=("evidence_type", "administered_dose", "animal_context", "exposure_context", "qualifying_conditions", "extra_details"),
            scope_fields=("animal_context", "exposure_context", "qualifying_conditions"),
            name_fields=(),
            target_pref_name="nonclinical in vivo toxicity",
            evidence_role="mechanistic_factor",
            extra_example_fields=("evidence_type", "administered_dose", "needs_more_context"),
            max_rows=max_rows,
        ),
        *_organ_profiles(raw, max_rows=max_rows),
        StarlingSourceProfile(
            source_id="clintox_genotoxicity_carcinogenicity",
            path=str(raw / "genotoxicity_carcinogenicity" / "extractions.parquet"),
            group_id="Mechanism.genotoxicity_carcinogenicity",
            assay_tier="Mechanism",
            endpoint_group="genotoxicity_carcinogenicity",
            evidence_source="Starling/ClinTox/genotoxicity_carcinogenicity",
            endpoint_field="endpoint",
            smiles_field="SMILES",
            context_fields=("evidence_category", "assay_type", "study_context", "result_direction", "biological_system", "exposure_conditions", "qualifying_conditions", "extra_details"),
            scope_fields=("assay_type", "study_context", "biological_system", "qualifying_conditions"),
            name_fields=(),
            target_pref_name="genotoxicity and carcinogenicity",
            evidence_role="mechanistic_factor",
            extra_example_fields=("evidence_category", "assay_type", "result_direction", "needs_more_context"),
            max_rows=max_rows,
        ),
        StarlingSourceProfile(
            source_id="clintox_cellular_stress",
            path=str(raw / "cellular_stress" / "extractions.parquet"),
            group_id="Mechanism.cellular_stress_pathways",
            assay_tier="Mechanism",
            endpoint_group="cellular_stress_pathways",
            evidence_source="Starling/ClinTox/cellular_stress",
            endpoint_field="stress_endpoint",
            smiles_field="SMILES",
            context_fields=("effect_direction", "evidence_basis", "mechanistic_effect", "target_or_pathway", "biological_model", "dose_and_duration", "qualifying_conditions", "extra_details"),
            scope_fields=("evidence_basis", "biological_model", "dose_and_duration", "qualifying_conditions"),
            name_fields=(),
            target_pref_name="cellular stress pathways",
            evidence_role="mechanistic_factor",
            extra_example_fields=("effect_direction", "target_or_pathway", "needs_more_context"),
            max_rows=max_rows,
        ),
        StarlingSourceProfile(
            source_id="clintox_general_cytotoxicity",
            path=str(raw / "general_cytotoxicity" / "extractions.parquet"),
            group_id="Mechanism.general_cytotoxicity",
            assay_tier="Mechanism",
            endpoint_group="general_cytotoxicity",
            evidence_source="Starling/ClinTox/general_cytotoxicity",
            endpoint_field="endpoint_type",
            smiles_field="SMILES",
            value_field="result_value",
            unit_field="result_unit",
            context_fields=("cell_model", "test_concentration", "exposure_time_h", "assay_method", "qualifying_conditions", "extra_details"),
            scope_fields=("cell_model", "exposure_time_h", "assay_method", "qualifying_conditions"),
            name_fields=(),
            target_pref_name="general cytotoxicity and viability",
            evidence_role="mechanistic_factor",
            extra_example_fields=("cell_model", "test_concentration", "assay_method", "needs_more_context"),
            max_rows=max_rows,
        ),
        StarlingSourceProfile(
            source_id="clintox_off_target_ddi_exposure",
            path=str(raw / "off_target_ddi_exposure" / "extractions.parquet"),
            group_id="Mechanism.off_target_ddi_exposure",
            assay_tier="Mechanism",
            endpoint_group="off_target_ddi_exposure",
            evidence_source="Starling/ClinTox/off_target_ddi_exposure",
            endpoint_field="target_or_endpoint",
            smiles_field="SMILES",
            value_field="result_value",
            unit_field="result_unit",
            context_fields=("evidence_type", "target_identifier", "result_metric", "assay_context", "qualifying_conditions", "extra_details"),
            scope_fields=("evidence_type", "assay_context", "qualifying_conditions"),
            name_fields=(),
            target_pref_name="off-target, DDI, and exposure liability",
            evidence_role="mechanistic_factor",
            extra_example_fields=("evidence_type", "target_identifier", "result_metric", "needs_more_context"),
            max_rows=max_rows,
        ),
    ]
    return profiles


def _organ_profiles(raw: Path, *, max_rows: int) -> list[StarlingSourceProfile]:
    common = dict(
        path=str(raw / "organ_specific_toxicity" / "extractions.parquet"),
        endpoint_field="toxicity_endpoint",
        smiles_field="SMILES",
        context_fields=("organ_system", "effect_status", "evidence_context", "biological_system", "exposure_regimen", "quantitative_result", "qualifying_conditions", "extra_details"),
        scope_fields=("organ_system", "evidence_context", "biological_system", "exposure_regimen", "qualifying_conditions"),
        name_fields=(),
        extra_example_fields=("organ_system", "effect_status", "evidence_context", "needs_more_context"),
        max_rows=max_rows,
    )
    return [
        StarlingSourceProfile(
            source_id="clintox_human_organ_context",
            group_id=CLINICAL_CONTEXT_GROUP,
            assay_tier="Mechanism",
            endpoint_group="clinical_human_safety",
            evidence_source="Starling/ClinTox/organ_specific_toxicity",
            target_pref_name="human clinical organ toxicity context",
            evidence_role="context_modifier",
            record_filter=lambda row: _text(row.get("evidence_context")).lower() == "human_clinical",
            record_filter_name="evidence_context_is_human_clinical.v1",
            **common,
        ),
        StarlingSourceProfile(
            source_id="clintox_organ_specific_toxicity",
            group_id="Mechanism.organ_specific_toxicity",
            assay_tier="Mechanism",
            endpoint_group="organ_specific_toxicity",
            evidence_source="Starling/ClinTox/organ_specific_toxicity",
            target_pref_name="organ-specific toxicity",
            evidence_role="mechanistic_factor",
            record_filter=lambda row: _text(row.get("evidence_context")).lower() != "human_clinical",
            record_filter_name="evidence_context_is_not_human_clinical.v1",
            **common,
        ),
    ]


def load_historical_mechanism_catalog(path: Path) -> list[dict[str, Any]]:
    """Reuse the verified six-family catalog without its obsolete direct role."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("group_id") in {
                "Direct.human_organ_toxicity",
                "Context.human_organ_toxicity",
            }:
                row.update(
                    {
                        "group_id": CLINICAL_CONTEXT_GROUP,
                        "assay_tier": "Mechanism",
                        "endpoint_group": "clinical_human_safety",
                        "evidence_role": "context_modifier",
                        "target_pref_name": "human clinical organ toxicity context",
                    }
                )
                uncertainty = list(row.get("uncertainty") or [])
                if "not_strict_clinical_trial_failure_evidence" not in uncertainty:
                    uncertainty.append("not_strict_clinical_trial_failure_evidence")
                row["uncertainty"] = uncertainty
                attach_minimal_evidence(row)
            rows.append(row)
    return rows


def audit_direct_gate(
    source_path: Path,
    *,
    canonical_root: Path,
    max_rows: int = 0,
    qa_rows: int = 120,
) -> dict[str, Any]:
    canonical_root.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    decisions: Counter[str] = Counter()
    source_index = 0
    columns = [
        "support_text", "molecule_name", "toxicity_outcome", "toxicity_category",
        "outcome_measure", "clinical_context", "dose_or_exposure",
        "fda_approval_status", "approved_indication", "extra_details", "confidence",
        "needs_more_context", "pmid", "extraction_id", "SMILES",
    ]
    for batch in pq.ParquetFile(source_path).iter_batches(batch_size=32_768, columns=columns):
        for row in batch.to_pylist():
            if max_rows and source_index >= max_rows:
                break
            reason = trial_failure_gate_reason(row)
            decisions[reason] += 1
            if reason.startswith("accept_"):
                accepted.append(
                    {
                        "source_row_number": source_index,
                        "source_record_id": f"clintox_base_v1:{source_index:09d}",
                        **row,
                        "direct_gate_version": DIRECT_GATE_VERSION,
                        "direct_gate_reason": reason,
                        "qualifying_conditions_source_status": "column_unavailable",
                        "review_status": "pending",
                    }
                )
            source_index += 1
        if max_rows and source_index >= max_rows:
            break

    claims_path = canonical_root / "strict_direct_claims.parquet"
    sample_path = canonical_root / "strict_direct_qa_sample.parquet"
    pq.write_table(pa.Table.from_pylist(accepted), claims_path, compression="zstd")
    ranked = sorted(
        accepted,
        key=lambda row: hashlib.sha256(str(row["source_record_id"]).encode()).hexdigest(),
    )[:qa_rows]
    pq.write_table(pa.Table.from_pylist(ranked), sample_path, compression="zstd")
    result = {
        "status": "candidate_strict_literal_gate_pending_qa",
        "gate_version": DIRECT_GATE_VERSION,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "n_source_rows_scanned": source_index,
        "n_accepted_direct_rows": len(accepted),
        "n_unique_direct_smiles": len({_text(row.get("SMILES")) for row in accepted}),
        "n_unique_direct_pmids": len({_text(row.get("pmid")) for row in accepted}),
        "decision_counts": dict(sorted(decisions.items())),
        "qualifying_conditions_source_status": "column_unavailable",
        "qa_sample_status": "pending_manual_review",
        "n_qa_sample_rows": len(ranked),
        "paths": {
            "accepted_claims": str(claims_path),
            "qa_sample": str(sample_path),
        },
    }
    write_json_atomic(canonical_root / "strict_direct_manifest.json", result)
    return result


def verify_source_manifests(data_root: Path) -> dict[str, Any]:
    manifests = [
        data_root / "clintox_base_v1" / "SOURCE_MANIFEST.json",
        data_root / "raw_v1" / "SOURCE_MANIFEST.json",
    ]
    receipts = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checked = []
        for item in manifest.get("files") or []:
            path = Path(str(item["path"]))
            expected = str(item.get("sha256") or "")
            if not path.is_file():
                raise FileNotFoundError(path)
            observed = sha256_file(path)
            if expected and observed != expected:
                raise ValueError(f"source checksum mismatch for {path}: {observed}")
            checked.append({"path": str(path), "sha256": observed})
        receipts.append(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "n_files_verified": len(checked),
                "files": checked,
            }
        )
    return {"status": "verified", "manifests": receipts}


def _annotate_base_uncertainty(row: dict[str, Any]) -> dict[str, Any]:
    uncertainty = list(row.get("uncertainty") or [])
    for value in (
        "qualifying_conditions_unavailable_in_source_schema",
        "strict_direct_partition_pending_source_record_qa",
    ):
        if value not in uncertainty:
            uncertainty.append(value)
    row["uncertainty"] = uncertainty
    attach_minimal_evidence(row)
    return row


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--mechanism-catalog", type=Path, default=DEFAULT_MECHANISM_CATALOG)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-rows-per-source", type=int, default=0)
    parser.add_argument("--max-record-examples", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = build_retrieval_library(
        data_root=args.data_root,
        mechanism_catalog=args.mechanism_catalog,
        canonical_root=args.canonical_root,
        out_dir=args.out_dir,
        max_rows_per_source=args.max_rows_per_source,
        max_record_examples=args.max_record_examples,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLINICAL_CONTEXT_GROUP",
    "DIRECT_GATE_VERSION",
    "DIRECT_GROUP",
    "MECHANISM_FAMILIES",
    "build_retrieval_library",
    "clinical_profiles",
    "is_clinical_context_record",
    "is_strict_trial_failure_record",
    "load_historical_mechanism_catalog",
    "mechanism_profiles",
    "trial_failure_gate_reason",
]
