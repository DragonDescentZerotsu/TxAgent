"""Create the explicit per-record BBB near-direct review ledger.

Text patterns are used only to find a bounded review queue.  Each candidate is
then adjudicated with the complete BBB source row. Direct CNS measurements stay
in L1; functional proxies and label-proximal predictions retain their evidence
type in near-direct rather than being discarded. The ledger keeps both move and
keep decisions so the boundary is inspectable and repeatable.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

from tools.chembl_tool.common.json_utils import (
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.experimental_meaningful_cns_access_benchmark import (
    _INDIRECT_INFERENCE_PATTERN,
    classify_scope,
)
from tools.chembl_tool.tasks.bbb_martins.source_family_purity import (
    DEFAULT_GOLD_ROOT,
    DEFAULT_RECORDS,
    DEFAULT_REVIEW_LEDGER,
    DIRECT_GROUP,
    REVIEW_VERSION,
    load_gold_vote_source_indices,
    source_index,
    support_key,
)


SUPPLEMENTAL_FUNCTIONAL_PROXY_REVIEWS = {
    224755: "systemic_central_pharmacodynamic_response_proxy",
    224756: "systemic_central_pharmacodynamic_response_proxy",
    272573: "systemic_cns_biomarker_or_function_proxy",
}

# Full-record adjudication of one external influx-acquisition row. The row is
# attached to the 4-amino analog, but its support is a comparative qualitative
# BBB outcome: the query compound is said not to penetrate while the analog is
# said to cross. It contains neither an influx assay nor a measured CNS-exposure
# endpoint, so it belongs in near-direct rather than influx or experimental L1.
SUPPLEMENTAL_EXTERNAL_NEAR_DIRECT_REVIEWS = {
    "11dc172aa9c4dd920fe28ffef2668698a41d3de96f63bdb437a836c6ea43c798": (
        "comparative_qualitative_bbb_outcome_without_experimental_measurement"
    ),
}

# These candidate rows contain an actual CNS concentration, ratio, tracer
# uptake, tissue distribution, or imaging measurement.  They remain in L1 even
# though nearby prose also makes an indirect BBB inference.
DIRECT_MEASUREMENT_KEEP_REVIEWS = {
    71912: "direct_csf_level_and_serum_ratio_measurement",
    86259: "direct_brain_tracer_uptake_measurement",
    99046: "direct_serum_and_csf_measurement",
    119696: "direct_radiolabeled_tissue_imaging",
    119697: "direct_radiolabeled_tissue_imaging",
    119698: "direct_radiolabeled_tissue_imaging",
    119699: "direct_radiolabeled_tissue_imaging",
    119700: "direct_radiolabeled_tissue_imaging",
    119701: "direct_radiolabeled_tissue_imaging",
    121127: "direct_brain_to_blood_measurement",
    127611: "direct_unbound_brain_to_blood_measurement",
    133582: "direct_csf_and_plasma_concentration_measurement",
    140435: "direct_tissue_accumulation_measurement",
    147579: "direct_csf_concentration_measurement",
    150352: "direct_contrast_enhancement_measurement",
    157460: "direct_brain_radioactivity_uptake_measurement",
    167828: "direct_csf_level_and_serum_ratio_measurement",
    178850: "direct_brain_distribution_measurement",
    189025: "direct_brain_biodistribution_measurement",
    257875: "direct_regional_penetration_observation",
    284177: "direct_in_vivo_distribution_measurement",
    290193: "direct_bioanalytical_brain_penetration_measurement",
    291271: "direct_tracer_leakage_measurement",
}

_REVIEW_COLUMNS = (
    "group_id",
    "source_index",
    "source_record_id",
    "canonical_record_id",
    "canonical_smiles",
    "pmid",
    "assay_model",
    "canonical_assay_context",
    "canonical_endpoint_name",
    "canonical_measurement_text",
    "species",
    "qualifying_conditions",
    "support_text",
    "extra_details",
)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _review_row(
    record: Mapping[str, Any], gold_indices: set[int]
) -> dict[str, Any] | None:
    index = source_index(record)
    canonical_record_id = _text(record.get("canonical_record_id"))
    external_reason = SUPPLEMENTAL_EXTERNAL_NEAR_DIRECT_REVIEWS.get(
        canonical_record_id
    )
    if external_reason is None and (
        index is None or _text(record.get("group_id")) != DIRECT_GROUP
    ):
        return None
    support = _text(record.get("support_text"))
    supplemental_reason = SUPPLEMENTAL_FUNCTIONAL_PROXY_REVIEWS.get(index)
    if (
        external_reason is None
        and supplemental_reason is None
        and not _INDIRECT_INFERENCE_PATTERN.search(support)
    ):
        return None

    scope = classify_scope(record, allow_conditioned_context=True)
    explicit_keep_reason = DIRECT_MEASUREMENT_KEEP_REVIEWS.get(index)
    if external_reason:
        decision = "move_to_near_direct"
        decision_reason = external_reason
        adjudication = "explicit_full_record_review"
    elif explicit_keep_reason:
        decision = "keep_current_family"
        decision_reason = explicit_keep_reason
        adjudication = "explicit_full_record_review"
    elif supplemental_reason:
        decision = "move_to_near_direct"
        decision_reason = supplemental_reason
        adjudication = "explicit_full_record_review"
    elif scope.endpoint_family:
        decision = "keep_current_family"
        decision_reason = f"direct_measurement:{scope.endpoint_family}"
        adjudication = "full_record_scope_adjudication"
    elif scope.rejection_reason == "computational_or_predicted_result":
        decision = "move_to_near_direct"
        decision_reason = "computational_label_proximal_bbb_prediction"
        adjudication = "full_record_scope_adjudication"
    else:
        decision = "move_to_near_direct"
        decision_reason = (
            "systemic_functional_outcome_used_as_cns_access_proxy"
            if scope.rejection_reason == "indirect_outcome_inference"
            else "functional_or_target_engagement_cns_access_proxy"
        )
        adjudication = "full_record_scope_adjudication"

    return {
        "review_version": REVIEW_VERSION,
        "source_index": index,
        "source_record_id": _text(record.get("source_record_id")),
        "canonical_record_id": canonical_record_id,
        "decision": decision,
        "decision_reason": decision_reason,
        "adjudication": adjudication,
        "candidate_discovery_only": (
            "explicit_query_trace_review"
            if supplemental_reason
            else "indirect_inference_text_candidate"
        ),
        "gold_vote_source": index in gold_indices,
        "support_key": support_key(record),
        "canonical_smiles": _text(record.get("canonical_smiles")),
        "pmid": _text(record.get("pmid")),
        "assay_model": _text(record.get("assay_model")),
        "canonical_assay_context": _text(record.get("canonical_assay_context")),
        "canonical_endpoint_name": _text(record.get("canonical_endpoint_name")),
        "canonical_measurement_text": _text(record.get("canonical_measurement_text")),
        "species": _text(record.get("species")),
        "qualifying_conditions": _text(record.get("qualifying_conditions")),
        "support_text": support,
        "extra_details": _text(record.get("extra_details")),
        "scope_endpoint_family": scope.endpoint_family or "",
        "scope_basis": scope.basis or "",
        "scope_rejection_reason": scope.rejection_reason or "",
    }


def build_review(records_path: Path, output_path: Path) -> dict[str, Any]:
    gold_indices = load_gold_vote_source_indices()
    rows: list[dict[str, Any]] = []
    parquet = pq.ParquetFile(records_path)
    for batch in parquet.iter_batches(batch_size=5_000, columns=_REVIEW_COLUMNS):
        for record in batch.to_pylist():
            row = _review_row(record, gold_indices)
            if row is not None:
                rows.append(row)
    rows.sort(
        key=lambda row: (
            row.get("source_index") is None,
            int(row["source_index"]) if row.get("source_index") is not None else 0,
            str(row.get("canonical_record_id") or ""),
        )
    )
    by_index = {
        int(row["source_index"]): row
        for row in rows
        if row.get("source_index") is not None
    }
    missed_supplemental = [
        index
        for index in SUPPLEMENTAL_FUNCTIONAL_PROXY_REVIEWS
        if by_index.get(index, {}).get("decision") != "move_to_near_direct"
    ]
    if missed_supplemental:
        raise RuntimeError(
            "explicit near-direct reviews were not applied: "
            + ", ".join(map(str, missed_supplemental))
        )
    by_canonical_record_id = {
        str(row.get("canonical_record_id") or ""): row for row in rows
    }
    missed_external = [
        record_id
        for record_id in SUPPLEMENTAL_EXTERNAL_NEAR_DIRECT_REVIEWS
        if by_canonical_record_id.get(record_id, {}).get("decision")
        != "move_to_near_direct"
    ]
    if missed_external:
        raise RuntimeError(
            "explicit external near-direct reviews were not applied: "
            + ", ".join(missed_external)
        )
    moved_gold = [
        row
        for row in rows
        if row["decision"] == "move_to_near_direct" and row["gold_vote_source"]
    ]
    if moved_gold:
        raise RuntimeError(
            "near-direct review attempted to move gold-vote rows: "
            + ", ".join(str(row.get("source_index")) for row in moved_gold[:10])
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_path, rows)
    decisions = Counter(str(row["decision"]) for row in rows)
    reasons = Counter(str(row["decision_reason"]) for row in rows)
    manifest = {
        "review_version": REVIEW_VERSION,
        "records": str(records_path.resolve()),
        "records_sha256": sha256_file(records_path),
        "review_ledger": str(output_path.resolve()),
        "review_ledger_sha256": sha256_file(output_path),
        "gold_root": str(DEFAULT_GOLD_ROOT.resolve()),
        "n_gold_vote_source_indices": len(gold_indices),
        "n_candidate_rows_reviewed": len(rows),
        "decision_counts": dict(sorted(decisions.items())),
        "decision_reason_counts": dict(sorted(reasons.items())),
        "n_moved_gold_vote_rows": 0,
        "candidate_patterns_are_discovery_only": True,
        "full_source_fields_retained_in_ledger": True,
        "n_external_record_reviews": len(SUPPLEMENTAL_EXTERNAL_NEAR_DIRECT_REVIEWS),
    }
    manifest_path = output_path.with_name("near_direct_record_review_manifest.json")
    write_json_atomic(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=str(DEFAULT_RECORDS))
    parser.add_argument("--output", default=str(DEFAULT_REVIEW_LEDGER))
    args = parser.parse_args(argv)
    print(build_review(Path(args.records), Path(args.output)))


if __name__ == "__main__":
    main()
