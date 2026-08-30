"""Prepare and build the reviewed Bioavailability conditioned benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from tools.chembl_tool.common.starling.condition_review import (
    attach_parent_identity,
    merge_terminal_verdicts,
    write_review_queue,
)
from tools.chembl_tool.common.starling.external_condition import payload_sha256
from tools.chembl_tool.common.starling.reviewed_conditioned_benchmark import (
    ConditionedBenchmarkConfig,
    build_reviewed_conditioned_benchmark,
)
from tools.chembl_tool.common.starling.conditioned_benchmark import BUILD_ROOT, CONTRACT
from tools.chembl_tool.tasks.bioavailability_ma.canonical_source import (
    DIRECT_CLAIMS_PATH,
    DIRECT_REPORT_TYPES,
)
from tools.chembl_tool.tasks.bioavailability_ma.condition_ontology import (
    ONTOLOGY_VERSION,
    classify_external_condition,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_benchmark import (
    is_human_context,
    label_bioavailability_value,
)


PROPOSAL_VERSION = "bioavailability_external_condition_proposal.v6"
LINEAGE = CONTRACT
FROZEN_ROOT = Path(
    "data/processed_starling_record_supported_v2/Bioavailability_Ma/scaffold"
)
REVIEW_ROOT = Path(
    "data/starling_data/bioavailability_ma/context_conditioned_review_v2"
)
OUTPUT_ROOT = BUILD_ROOT / "Bioavailability_Ma/scaffold"

CORE_EXTERNAL_CONDITION_GROUPS = (
    "prandial_state=fasted",
    "prandial_state=fed_unspecified",
    "prandial_state=fed_high_fat",
    "co_treatment=rifampin",
    "disease=cirrhosis",
    "disease=cystic_fibrosis",
    "release_profile=modified_release",
)


_RELATIVE_EFFECT_PATTERN = re.compile(
    r"\b(?:increase(?:d|s|ing)?|decrease(?:d|s|ing)?|reduce(?:d|s|ing)?|"
    r"reduction|enhance(?:d|s|ment)?|improve(?:d|s|ment)?|unchanged|"
    r"fold(?:-|\s)?(?:increase|decrease|change)?|food effect)\b|"
    r"\brelative\s+(?:(?:oral|systemic)\s+)?(?:bio)?availabilit(?:y|ies)\b|"
    r"\b(?:no|not)\s+(?:statistically\s+|significantly\s+)?"
    r"(?:different|changed|affected|effect)\b|"
    r"\b(?:higher|lower)\s+than\b|"
    r"\bfrom\s+\d+(?:\.\d+)?\s*%?\s+to\s+\d+(?:\.\d+)?\s*%?\b",
    flags=re.IGNORECASE,
)
_COMPARISON_PATTERN = re.compile(r"\b(?:compared\s+(?:with|to)|versus|vs\.?)\b", re.IGNORECASE)
_IV_REFERENCE_PATTERN = re.compile(r"\b(?:intravenous(?:ly)?|i\.?v\.?)\b", re.IGNORECASE)
_INDIRECT_ANALYTE_PATTERN = re.compile(
    r"\b(?:pro[-\s]?drug|metabolite|pivoxil|axetil|disoproxil|"
    r"ximelagatran|oseltamivir|val(?:aciclovir|ganciclovir)|desglymidodrine|"
    r"enacarbil|bacampicillin|mycophenolate mofetil|tenofovir df|tdf)\b|"
    r"\boxcarbazepine\b.{0,160}\bmhd\b|"
    r"\bdiacetylmorphine\b.{0,160}\bmorphine\b",
    re.IGNORECASE,
)
_NONEMPIRICAL_PATTERN = re.compile(
    r"\b(?:simulat(?:ed|ion)|predict(?:ed|ion)|pbpk|model(?:s|ed|led|ing)?)\b|"
    r"\b(?:population pharmacokinetic|population pk|poppk)\s+model\b",
    re.IGNORECASE,
)
_AMBIGUOUS_ROUTE_PATTERN = re.compile(
    r"\bby mouth\s+or\s+following\s+intramuscular\b", re.IGNORECASE
)


def relative_effect_exclusion_reason(row: dict[str, Any]) -> str | None:
    """Reject effect/comparison claims even when one arm has an absolute value."""
    narrative = " ".join(
        str(row.get(field) or "") for field in ("support_text", "extra_details")
    )
    if _RELATIVE_EFFECT_PATTERN.search(narrative):
        return "relative_effect_record_not_absolute_condition_claim"
    if re.search(r"\brelative\b", narrative, flags=re.IGNORECASE) and not _IV_REFERENCE_PATTERN.search(
        narrative
    ):
        return "non_iv_relative_reference_not_absolute_condition_claim"
    comparator = str(row.get("comparator") or "").strip()
    if comparator and not _IV_REFERENCE_PATTERN.search(comparator):
        return "non_iv_comparator_record_not_absolute_condition_claim"
    if _COMPARISON_PATTERN.search(narrative) and not _IV_REFERENCE_PATTERN.search(narrative):
        return "non_iv_comparison_record_not_absolute_condition_claim"
    return None


def direct_exposure_exclusion_reason(row: dict[str, Any]) -> str | None:
    """Reject indirect analytes and non-empirical F values before review."""
    narrative = " ".join(
        str(row.get(field) or "") for field in ("support_text", "extra_details")
    )
    if _INDIRECT_ANALYTE_PATTERN.search(narrative):
        return "prodrug_or_indirect_analyte_not_direct_oral_f"
    if _NONEMPIRICAL_PATTERN.search(narrative):
        return "predicted_or_simulated_f_not_direct_measurement"
    if _AMBIGUOUS_ROUTE_PATTERN.search(narrative):
        return "mixed_route_statement_not_direct_oral_f"
    return None


def _proposal_audit_base(index: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_record_id": str(row.get("canonical_claim_id") or f"row:{index}"),
        "source_index": int(index),
        "condition_text": str(row.get("qualifying_conditions") or ""),
        "raw_value": str(row.get("oral_bioavailability_value") or ""),
        "report_type": str(row.get("bioavailability_report_type") or ""),
        "species_or_population": str(row.get("species_or_population") or ""),
    }


def propose_source_row(
    index: int, row: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return an auditable proposal; relative/effect rows fail before review."""
    audit = _proposal_audit_base(index, row)
    condition = classify_external_condition(row.get("qualifying_conditions"))
    audit.update(
        {
            "normalized_condition": condition.normalized_text,
            "condition_scope": condition.scope,
            "proposal_reason": condition.reason,
            "proposed_condition_group": condition.signature or "",
        }
    )
    if condition.scope != "external" or not condition.signature:
        return {**audit, "queue_status": "not_queued"}, None
    if any(
        atom.family == "administered_form" and atom.value == "prodrug_unspecified"
        for atom in condition.atoms
    ):
        return {
            **audit,
            "queue_status": "not_queued",
            "proposal_reason": "prodrug_or_active_moiety_identity_not_an_external_condition",
        }, None

    report_type = audit["report_type"].strip().lower()
    if report_type not in DIRECT_REPORT_TYPES:
        return {
            **audit,
            "queue_status": "not_queued",
            "proposal_reason": "non_direct_oral_bioavailability_report_type",
        }, None
    relative_effect_reason = relative_effect_exclusion_reason(row)
    if relative_effect_reason:
        return {
            **audit,
            "queue_status": "not_queued",
            "proposal_reason": relative_effect_reason,
        }, None
    direct_exposure_reason = direct_exposure_exclusion_reason(row)
    if direct_exposure_reason:
        return {
            **audit,
            "queue_status": "not_queued",
            "proposal_reason": direct_exposure_reason,
        }, None
    if not is_human_context(row.get("species_or_population")):
        return {
            **audit,
            "queue_status": "not_queued",
            "proposal_reason": "nonhuman_or_unresolved_population",
        }, None
    label, label_method = label_bioavailability_value(
        row.get("oral_bioavailability_value")
    )
    if label is None:
        return {
            **audit,
            "queue_status": "not_queued",
            "proposal_reason": label_method,
        }, None

    identity = attach_parent_identity({}, row.get("smiles"))
    if not identity.get("molecule_identity_key"):
        return {
            **audit,
            "queue_status": "not_queued",
            "proposal_reason": "invalid_parent_identity",
        }, None

    atoms = [atom.key for atom in condition.atoms]
    record_id = audit["source_record_id"]
    payload_fields = (
        PROPOSAL_VERSION,
        ONTOLOGY_VERSION,
        condition.signature,
        int(label),
        record_id,
        row.get("smiles"),
        report_type,
        row.get("oral_bioavailability_value"),
        row.get("species_or_population"),
        row.get("qualifying_conditions"),
        row.get("support_text"),
        row.get("comparator"),
        row.get("extra_details"),
    )
    candidate = {
        **identity,
        "source_record_id": record_id,
        "source_payload_sha256": payload_sha256(payload_fields),
        "source_index": int(index),
        "pmid": str(row.get("pmid") or ""),
        "molecule_name": str(row.get("molecule_name") or ""),
        "condition_text": audit["condition_text"],
        "support_text": str(row.get("support_text") or ""),
        "raw_value": audit["raw_value"],
        "bioavailability_report_type": report_type,
        "comparator": str(row.get("comparator") or ""),
        "species": audit["species_or_population"],
        "extra_details": str(row.get("extra_details") or ""),
        "proposed_condition_group": condition.signature,
        "proposed_condition_atoms": atoms,
        "proposal_reason": condition.reason,
        "Y": int(label),
        "label_method": f"canonical:{label_method}",
    }
    return {
        **audit,
        "queue_status": "queued",
        "label": int(label),
        "label_method": label_method,
    }, candidate


def prepare_review_queue() -> dict[str, Any]:
    frame = pd.read_parquet(DIRECT_CLAIMS_PATH)
    candidates: list[dict[str, Any]] = []
    proposal_audit: list[dict[str, Any]] = []
    for index, row in enumerate(frame.to_dict(orient="records")):
        if not str(row.get("qualifying_conditions") or "").strip():
            continue
        audit, candidate = propose_source_row(index, row)
        proposal_audit.append(audit)
        if candidate is not None:
            candidates.append(candidate)
    return write_review_queue(
        rows=candidates,
        queue_path=REVIEW_ROOT / "review_queue.jsonl",
        manifest_path=REVIEW_ROOT / "review_queue_manifest.json",
        task_name="Bioavailability_Ma",
        source_artifacts=(DIRECT_CLAIMS_PATH,),
        proposal_version=PROPOSAL_VERSION,
        proposal_audit_rows=proposal_audit,
    )


def build() -> dict[str, Any]:
    reviewed = merge_terminal_verdicts(
        queue_path=REVIEW_ROOT / "review_queue.jsonl",
        verdict_path=REVIEW_ROOT / "review_verdicts.jsonl",
    )
    return build_reviewed_conditioned_benchmark(
        config=ConditionedBenchmarkConfig(
            task_name="Bioavailability_Ma",
            lineage=LINEAGE,
            protocol_version="bioavailability_context_conditioned_selected.v1",
            frozen_root=FROZEN_ROOT,
            output_root=OUTPUT_ROOT,
            source_artifacts=(
                DIRECT_CLAIMS_PATH,
                REVIEW_ROOT / "review_queue.jsonl",
                REVIEW_ROOT / "review_queue_manifest.json",
                REVIEW_ROOT / "model_prereview.jsonl",
                REVIEW_ROOT / "model_prereview.summary.json",
                REVIEW_ROOT / "model_prereview_round2.jsonl",
                REVIEW_ROOT / "model_prereview_round2.summary.json",
                REVIEW_ROOT / "review_comparison_summary.json",
                REVIEW_ROOT / "review_verdicts.jsonl",
            ),
            row_id_prefix="BIOCTX2",
            frozen_lineage="record_supported_v2",
            excluded_condition_groups=(
                # These signatures collapse biologically distinct contexts and
                # therefore are not transferable exact-condition groups.
                "disease=cancer_unspecified",
                "metabolizer_phenotype=unspecified_extensive",
                "metabolizer_phenotype=unspecified_intermediate",
                "metabolizer_phenotype=unspecified_poor",
                "metabolizer_phenotype=unspecified_ultra_rapid",
            ),
            allowed_condition_groups=CORE_EXTERNAL_CONDITION_GROUPS,
            publish_allowed_group_audit_only=True,
            review_policy=(
                "two independent GPT-OSS-120B semantic prereviews under a hashed v2 "
                "contract; only unanimous acceptance is terminally promoted, followed "
                "by task-semantic group and provenance audit"
            ),
        ),
        review_rows=reviewed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-review", "build-selected"))
    args = parser.parse_args(argv)
    result = prepare_review_queue() if args.action == "prepare-review" else build()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
