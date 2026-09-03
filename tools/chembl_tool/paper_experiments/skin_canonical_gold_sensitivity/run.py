"""Audit Skin gold-label sensitivity to canonical source and study-level voting.

This diagnostic is deliberately valid-only.  It never rewrites the frozen
benchmark and never reads the test split.  The canonical direct partition is
treated as the final-outcome source; assay-family classification is used only
for sensitivity analysis and conflict accounting.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
import json
from pathlib import Path
import re
from typing import Any

from tools.chembl_tool.common.json_utils import (
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


CONTRACT_VERSION = "skin_canonical_gold_sensitivity.valid.v1"
AGREEMENT_THRESHOLD = 0.70
DEFAULT_VALID_LABELS = Path(
    "data/gold_labels/legacy/processed_starling_record_supported_v2/Skin_Reaction/scaffold/"
    "valid_molecule_labels.jsonl"
)
DEFAULT_CANONICAL_DIRECT = Path(
    "data/artifacts/starling/skin_reaction/canonical_sources/canonical_sensitization_v3/direct_records.parquet"
)
DEFAULT_CANONICAL_MANIFEST = Path(
    "data/artifacts/starling/skin_reaction/canonical_sources/canonical_sensitization_v3/manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/paper/skin_canonical_gold_sensitivity_record_supported_v2_valid"
)

FAMILIES = (
    "predictive_animal",
    "predictive_human",
    "diagnostic_human",
    "case_occupational",
    "other_direct_ambiguous",
)
PREDICTIVE_FAMILIES = frozenset({"predictive_animal", "predictive_human"})
HUMAN_EVIDENCE_FAMILIES = frozenset(
    {"predictive_human", "diagnostic_human", "case_occupational"}
)

_ANIMAL_RE = re.compile(
    r"\bllna\b|local lymph node|\bgpmt\b|guinea pig maximi[sz]ation|"
    r"\bbuehler\b|mouse ear swelling|\bmest\b|guinea pig sensiti[sz]ation|"
    r"murine .*sensiti[sz]|contact hypersensitivity.*(?:mouse|mice)|"
    r"(?:mouse|mice|guinea pig).*contact hypersensitivity|ear[ -]?(?:swelling|thickness)|"
    r"\bchs\b|contact hypersensitivity|\blnpa\b|\bfcat\b|magnusson[ -]kligman|"
    r"guinea pig (?:assay|experiment|test|testing|technique)|"
    r"open epicutaneous test|\boet\b|draize (?:test|technique)",
    re.IGNORECASE,
)
_ANIMAL_POPULATION_RE = re.compile(
    r"\bmouse\b|\bmice\b|murine|guinea[ -]?pig|cavia porcellus|\banimal(?:s)?\b",
    re.IGNORECASE,
)
_HUMAN_PREDICTIVE_RE = re.compile(
    r"\bhript\b|\bript\b|repeated? (?:human )?insult patch|"
    r"human repeat(?:ed)? insult|human maximi[sz]ation|"
    r"maximi[sz]ation test.*human|human.*maximi[sz]ation test|"
    r"predictive human skin sensiti[sz]ation|"
    r"experimental sensiti[sz]ation.*human volunteer|"
    r"human volunteer.*experimental sensiti[sz]ation",
    re.IGNORECASE,
)
_CASE_OCCUPATIONAL_RE = re.compile(
    r"case report|case series|clinical report|occupational|worker|hairdresser|"
    r"beautician|nurse|healthcare|machinist|construction|dental technician|"
    r"single patient|single case|\b\d{2}[ -]?year[ -]?old (?:woman|man)|"
    r"contact allergy \(clinical history\)",
    re.IGNORECASE,
)
_DIAGNOSTIC_ASSAY_RE = re.compile(
    r"diagnostic patch|patch[ -]?test|patch testing|epicutaneous|"
    r"repeated open application|\broat\b|standard series|"
    r"intradermal test|clinical observation",
    re.IGNORECASE,
)
_PREDICTION_ONLY_ASSAY_RE = re.compile(
    r"prediction|predicted|computational|\bin[ -]?silico\b|\bqsar\b|read[ -]?across|"
    r"\badmet\b|machine learning|oecd toolbox|\bderek\b|\btopkat\b",
    re.IGNORECASE,
)
_PHOTO_ONLY_ASSAY_RE = re.compile(
    r"photo(?:toxic|irrit|allerg|sensiti)|photopatch|light[ -]?dependent|"
    r"ultraviolet|\buva\b|\buvb\b",
    re.IGNORECASE,
)
_IRRITATION_ONLY_ASSAY_RE = re.compile(
    r"(?:skin|dermal)?[ -]?irritation|corrosion|irritant dermatitis",
    re.IGNORECASE,
)
_SENSITIZATION_ANCHOR_RE = re.compile(
    r"sensiti[sz]|contact allerg|allergic contact|hypersensitiv|"
    r"\bllna\b|\bgpmt\b|\bbuehler\b|\bhript\b",
    re.IGNORECASE,
)
_INTEGRATED_ONLY_ASSAY_RE = re.compile(
    r"defined approach|integrated (?:testing|approach)|\biata\b|\b2o3\b|2 out of 3",
    re.IGNORECASE,
)
_HUMAN_POPULATION_RE = re.compile(
    r"\bhuman\b|patient|subject|volunteer|participant|dermatitis|eczema|child|adult",
    re.IGNORECASE,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run_audit(
        valid_labels_path=args.valid_labels,
        canonical_direct_path=args.canonical_direct,
        canonical_manifest_path=args.canonical_manifest,
        output_root=args.output_root,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def classify_evidence_family(record: Mapping[str, Any]) -> tuple[str, str]:
    """Classify a canonical direct record by experimental/observational design."""
    assay = _text(record.get("assay_or_test"))
    population = _text(record.get("species_or_population"))
    support = _text(record.get("support_text"))
    details = _text(record.get("extra_details"))
    structured = " | ".join((assay, population))
    context = " | ".join((assay, population, support, details))

    if _HUMAN_PREDICTIVE_RE.search(structured):
        return "predictive_human", "explicit_hript_ript_or_human_maximization"
    if re.search(r"volunteer", population, re.I) and re.search(
        r"sensiti[sz](?:ation|ing|er)|contact dermatitis sensiti[sz]ation|"
        r"clinical (?:dermal )?sensiti[sz]ation",
        assay,
        re.I,
    ):
        return "predictive_human", "human_volunteer_sensitization_design"
    if _ANIMAL_RE.search(structured) or (
        _ANIMAL_POPULATION_RE.search(population)
        and re.search(r"sensiti[sz]|hypersensitiv|challenge|maximi[sz]", assay, re.I)
    ):
        return "predictive_animal", "explicit_animal_predictive_assay"
    if _CASE_OCCUPATIONAL_RE.search(context):
        return "case_occupational", "explicit_case_or_occupational_context"
    if _DIAGNOSTIC_ASSAY_RE.search(assay) and _HUMAN_POPULATION_RE.search(context):
        if re.fullmatch(r"(?i)\s*(?:human )?patch test(?:ing)?\s*", assay):
            return "diagnostic_human", "generic_human_patch_test_inferred_diagnostic"
        return "diagnostic_human", "explicit_diagnostic_human_test_context"
    return "other_direct_ambiguous", "insufficient_design_metadata"


def direct_quality_gate(record: Mapping[str, Any]) -> tuple[bool, str]:
    """Detect manifest-violating non-outcome rows still present in canonical direct."""
    assay = _text(record.get("assay_or_test"))
    if _PREDICTION_ONLY_ASSAY_RE.search(assay):
        return False, "reject_prediction_only_assay"
    if _PHOTO_ONLY_ASSAY_RE.search(assay):
        return False, "reject_photo_only_assay"
    if _INTEGRATED_ONLY_ASSAY_RE.search(assay):
        return False, "reject_integrated_only_assay"
    if _IRRITATION_ONLY_ASSAY_RE.search(assay) and not _SENSITIZATION_ANCHOR_RE.search(assay):
        return False, "reject_irritation_only_assay"
    return True, "accepted_canonical_direct_candidate"


def consensus_vote(labels: Iterable[int], *, threshold: float = AGREEMENT_THRESHOLD) -> dict[str, Any]:
    """Return a deterministic record/study consensus without inventing a tie label."""
    values = []
    for value in labels:
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            continue
        if normalized in (0, 1):
            values.append(normalized)
    counts = Counter(values)
    if not values:
        return _empty_vote("no_usable_votes")
    if counts[0] == counts[1]:
        return _vote_result(None, counts, len(values), "exact_tie", threshold)
    label = 1 if counts[1] > counts[0] else 0
    agreement = counts[label] / len(values)
    if agreement < threshold:
        return _vote_result(None, counts, len(values), "below_agreement_threshold", threshold)
    return _vote_result(label, counts, len(values), "accepted", threshold)


def collapse_to_study_votes(records: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Give each PMID one vote; a within-PMID direction conflict abstains."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_study_key(record)].append(record)
    votes: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for study_key, rows in sorted(grouped.items()):
        labels = [_record_label(row) for row in rows]
        usable = [label for label in labels if label is not None]
        unique = sorted(set(usable))
        if len(unique) == 1:
            label = unique[0]
            status = "accepted_unanimous_study"
        elif len(unique) > 1:
            label = None
            status = "abstain_within_study_conflict"
        else:
            label = None
            status = "abstain_no_usable_outcome"
        status_counts[status] += 1
        votes.append(
            {
                "study_key": study_key,
                "pmid": _text(rows[0].get("pmid")),
                "label": label,
                "status": status,
                "n_records": len(rows),
                "record_label_counts": _label_counts(usable),
                "source_partitions": sorted({_text(row.get("source_partition")) for row in rows}),
                "source_record_ids": sorted(
                    _text(row.get("source_record_id")) for row in rows if _text(row.get("source_record_id"))
                ),
            }
        )
    return votes, dict(sorted(status_counts.items()))


def run_audit(
    *,
    valid_labels_path: Path = DEFAULT_VALID_LABELS,
    canonical_direct_path: Path = DEFAULT_CANONICAL_DIRECT,
    canonical_manifest_path: Path = DEFAULT_CANONICAL_MANIFEST,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    import pandas as pd

    valid_rows = read_jsonl(valid_labels_path)
    if len(valid_rows) != 245:
        raise ValueError(f"Expected 245 Skin valid rows, found {len(valid_rows)}")
    valid_keys = {str(row["molecule_identity_key"]) for row in valid_rows}
    if len(valid_keys) != len(valid_rows):
        raise ValueError("Skin valid molecule identities are not unique")

    manifest = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
    expected_sha = str(manifest["paths"]["direct_records_sha256"])
    observed_sha = sha256_file(canonical_direct_path)
    if observed_sha != expected_sha:
        raise ValueError("Canonical direct parquet SHA-256 does not match manifest")

    frame = pd.read_parquet(canonical_direct_path)
    frame = frame[frame["parent_inchi_key"].isin(valid_keys)].copy()
    source_records: list[dict[str, Any]] = []
    classification_counts: Counter[str] = Counter()
    classification_reason_counts: Counter[str] = Counter()
    quality_gate_counts: Counter[str] = Counter()
    for source in frame.to_dict(orient="records"):
        row = {key: _json_safe(value) for key, value in source.items()}
        label = _record_label(row)
        if label is None:
            continue
        family, reason = classify_evidence_family(row)
        quality_accepted, quality_reason = direct_quality_gate(row)
        row["normalized_label"] = label
        row["evidence_family"] = family
        row["family_classification_reason"] = reason
        row["direct_quality_gate_accepted"] = quality_accepted
        row["direct_quality_gate_reason"] = quality_reason
        row["study_key"] = _study_key(row)
        source_records.append(row)
        classification_counts[family] += 1
        classification_reason_counts[reason] += 1
        quality_gate_counts[quality_reason] += 1

    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_records:
        by_parent[str(row["parent_inchi_key"])].append(row)

    molecule_rows: list[dict[str, Any]] = []
    study_vote_rows: list[dict[str, Any]] = []
    for query_index, frozen in enumerate(valid_rows):
        parent_key = str(frozen["molecule_identity_key"])
        records = by_parent.get(parent_key, [])
        record_vote = consensus_vote(row["normalized_label"] for row in records)
        quality_records = [row for row in records if row["direct_quality_gate_accepted"]]
        quality_record_vote = consensus_vote(
            row["normalized_label"] for row in quality_records
        )
        all_studies, all_study_status = collapse_to_study_votes(records)
        study_vote = consensus_vote(
            vote["label"] for vote in all_studies if vote["label"] is not None
        )
        quality_studies, quality_study_status = collapse_to_study_votes(quality_records)
        quality_study_vote = consensus_vote(
            vote["label"] for vote in quality_studies if vote["label"] is not None
        )

        predictive_records = [
            row for row in records if row["evidence_family"] in PREDICTIVE_FAMILIES
        ]
        predictive_studies, predictive_study_status = collapse_to_study_votes(
            predictive_records
        )
        predictive_vote = consensus_vote(
            vote["label"] for vote in predictive_studies if vote["label"] is not None
        )

        human_records = [
            row for row in records if row["evidence_family"] in HUMAN_EVIDENCE_FAMILIES
        ]
        human_studies, human_study_status = collapse_to_study_votes(human_records)
        human_vote = consensus_vote(
            vote["label"] for vote in human_studies if vote["label"] is not None
        )

        family_votes: dict[str, dict[str, Any]] = {}
        family_study_counts: dict[str, int] = {}
        for family in FAMILIES:
            family_records = [row for row in records if row["evidence_family"] == family]
            family_studies, family_status = collapse_to_study_votes(family_records)
            family_votes[family] = {
                **consensus_vote(
                    vote["label"] for vote in family_studies if vote["label"] is not None
                ),
                "n_records": len(family_records),
                "n_studies": len(family_studies),
                "study_status_counts": family_status,
            }
            family_study_counts[family] = len(family_studies)

        observed_family_labels = {
            family: vote["label"]
            for family, vote in family_votes.items()
            if vote["label"] is not None
        }
        cross_family_conflict = len(set(observed_family_labels.values())) > 1
        method_votes = {
            "current_frozen": {"label": int(frozen["Y"]), "status": "anchor"},
            "canonical_record_vote": record_vote,
            "canonical_quality_gated_record_vote": quality_record_vote,
            "canonical_study_vote": study_vote,
            "canonical_quality_gated_study_vote": quality_study_vote,
            "predictive_only_study_vote": predictive_vote,
            "human_evidence_study_vote": human_vote,
        }
        molecule_rows.append(
            {
                "query_index": query_index,
                "drug": frozen["drug"],
                "molecule_identity_key": parent_key,
                "frozen_label": int(frozen["Y"]),
                "source_record_count": len(records),
                "source_study_count": len(all_studies),
                "source_partition_counts": dict(
                    sorted(Counter(row["source_partition"] for row in records).items())
                ),
                "family_record_counts": dict(
                    sorted(Counter(row["evidence_family"] for row in records).items())
                ),
                "family_study_counts": family_study_counts,
                "method_votes": method_votes,
                "family_votes": family_votes,
                "observed_family_labels": observed_family_labels,
                "cross_family_conflict": cross_family_conflict,
                "all_study_status_counts": all_study_status,
                "quality_study_status_counts": quality_study_status,
                "predictive_study_status_counts": predictive_study_status,
                "human_study_status_counts": human_study_status,
            }
        )
        for vote in all_studies:
            study_vote_rows.append(
                {
                    "query_index": query_index,
                    "molecule_identity_key": parent_key,
                    **vote,
                }
            )

    methods = (
        "canonical_record_vote",
        "canonical_quality_gated_record_vote",
        "canonical_study_vote",
        "canonical_quality_gated_study_vote",
        "predictive_only_study_vote",
        "human_evidence_study_vote",
    )
    method_summaries = {
        method: _summarize_method(molecule_rows, method) for method in methods
    }
    cross_family_conflicts = [row for row in molecule_rows if row["cross_family_conflict"]]
    family_summaries = {
        family: _summarize_family(molecule_rows, family) for family in FAMILIES
    }
    conflict_pairs: Counter[str] = Counter()
    for row in cross_family_conflicts:
        resolved = sorted(row["observed_family_labels"].items())
        for index, (left_family, left_label) in enumerate(resolved):
            for right_family, right_label in resolved[index + 1 :]:
                if left_label != right_label:
                    conflict_pairs[f"{left_family}__vs__{right_family}"] += 1
    rejected_quality_rows = [
        row for row in source_records if not row["direct_quality_gate_accepted"]
    ]
    summary = {
        "contract_version": CONTRACT_VERSION,
        "scope": {
            "task": "Skin_Reaction",
            "benchmark_lineage": "record_supported_v2",
            "split": "scaffold",
            "evaluation_subset": "valid",
            "test_read": False,
            "frozen_gold_modified": False,
        },
        "label_contracts": {
            "current_frozen": "raw direct source; scoped outcome_label; record vote >=70%",
            "canonical_record_vote": "canonical v3 direct rows; each deduplicated record is one vote",
            "canonical_quality_gated_record_vote": "canonical record vote after rejecting manifest-violating prediction/photo/irritation/integrated assay rows",
            "canonical_study_vote": "canonical v3 direct rows; each parent+PMID is one unanimous vote; conflicting papers abstain",
            "canonical_quality_gated_study_vote": "canonical PMID vote after the explicit direct-quality gate",
            "predictive_only_study_vote": "same study vote restricted to animal/human induction designs",
            "human_evidence_study_vote": "same study vote restricted to predictive human, diagnostic human, and case/occupational evidence",
        },
        "n_valid_molecules": len(valid_rows),
        "n_canonical_direct_records_for_valid": len(source_records),
        "n_valid_molecules_with_canonical_records": len(by_parent),
        "source_partition_counts": dict(
            sorted(Counter(row["source_partition"] for row in source_records).items())
        ),
        "evidence_family_record_counts": dict(sorted(classification_counts.items())),
        "classification_reason_counts": dict(sorted(classification_reason_counts.items())),
        "direct_quality_gate_counts": dict(sorted(quality_gate_counts.items())),
        "method_summaries": method_summaries,
        "family_summaries": family_summaries,
        "study_vote_summary": {
            "n_parent_studies": len(study_vote_rows),
            "n_multi_record_parent_studies": sum(
                int(row["n_records"]) > 1 for row in study_vote_rows
            ),
            "max_records_per_parent_study": max(
                (int(row["n_records"]) for row in study_vote_rows), default=0
            ),
            "status_counts": dict(
                sorted(Counter(row["status"] for row in study_vote_rows).items())
            ),
        },
        "quality_gate_rejections": {
            "n_records": len(rejected_quality_rows),
            "query_indices": sorted(
                {
                    next(
                        row["query_index"]
                        for row in molecule_rows
                        if row["molecule_identity_key"] == source["parent_inchi_key"]
                    )
                    for source in rejected_quality_rows
                }
            ),
            "source_record_ids": [row["source_record_id"] for row in rejected_quality_rows],
        },
        "cross_family_conflict": {
            "n_molecules": len(cross_family_conflicts),
            "rate": len(cross_family_conflicts) / len(valid_rows),
            "query_indices": [row["query_index"] for row in cross_family_conflicts],
            "pair_counts": dict(sorted(conflict_pairs.items())),
        },
        "input_provenance": {
            "valid_labels_path": str(valid_labels_path),
            "valid_labels_sha256": sha256_file(valid_labels_path),
            "canonical_direct_path": str(canonical_direct_path),
            "canonical_direct_sha256": observed_sha,
            "canonical_manifest_path": str(canonical_manifest_path),
            "canonical_manifest_sha256": sha256_file(canonical_manifest_path),
        },
        "paths": {
            "summary": str(output_root / "summary.json"),
            "molecule_sensitivity": str(output_root / "molecule_sensitivity.jsonl"),
            "study_votes": str(output_root / "study_votes.jsonl"),
            "classified_records": str(output_root / "classified_records.jsonl"),
            "report": str(output_root / "report.md"),
        },
    }
    write_jsonl_atomic(output_root / "molecule_sensitivity.jsonl", molecule_rows)
    write_jsonl_atomic(output_root / "study_votes.jsonl", study_vote_rows)
    write_jsonl_atomic(output_root / "classified_records.jsonl", source_records)
    write_json_atomic(output_root / "summary.json", summary)
    (output_root / "report.md").write_text(_render_report(summary, molecule_rows), encoding="utf-8")
    return summary


def _summarize_method(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    covered = [row for row in rows if row["method_votes"][method]["label"] is not None]
    flips = [
        row
        for row in covered
        if int(row["method_votes"][method]["label"]) != int(row["frozen_label"])
    ]
    confusion = Counter(
        (int(row["frozen_label"]), int(row["method_votes"][method]["label"]))
        for row in covered
    )
    return {
        "n_covered": len(covered),
        "coverage": len(covered) / len(rows),
        "n_unresolved": len(rows) - len(covered),
        "n_agree_with_frozen": len(covered) - len(flips),
        "agreement_with_frozen_among_covered": (
            (len(covered) - len(flips)) / len(covered) if covered else None
        ),
        "n_flips": len(flips),
        "flip_rate_among_covered": len(flips) / len(covered) if covered else None,
        "flip_directions": {
            f"{old}_to_{new}": count
            for (old, new), count in sorted(confusion.items())
            if old != new
        },
        "label_counts": _label_counts(
            int(row["method_votes"][method]["label"]) for row in covered
        ),
        "unresolved_reason_counts": dict(
            sorted(
                Counter(
                    row["method_votes"][method]["status"]
                    for row in rows
                    if row["method_votes"][method]["label"] is None
                ).items()
            )
        ),
        "flip_query_indices": [row["query_index"] for row in flips],
    }


def _summarize_family(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    with_records = [row for row in rows if row["family_votes"][family]["n_records"] > 0]
    resolved = [row for row in with_records if row["family_votes"][family]["label"] is not None]
    flips = [
        row
        for row in resolved
        if int(row["family_votes"][family]["label"]) != int(row["frozen_label"])
    ]
    return {
        "n_molecules_with_records": len(with_records),
        "record_coverage": len(with_records) / len(rows),
        "n_molecules_with_resolved_label": len(resolved),
        "resolved_label_coverage": len(resolved) / len(rows),
        "n_flips_vs_frozen": len(flips),
        "agreement_with_frozen_among_resolved": (
            (len(resolved) - len(flips)) / len(resolved) if resolved else None
        ),
        "resolved_label_counts": _label_counts(
            int(row["family_votes"][family]["label"]) for row in resolved
        ),
        "flip_query_indices": [row["query_index"] for row in flips],
    }


def _render_report(summary: Mapping[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Skin canonical-gold sensitivity audit",
        "",
        "This is a valid-only diagnostic. It did not read test or rewrite frozen labels.",
        "",
        "## Coverage and label changes",
        "",
        "| method | covered | coverage | unresolved | flips vs frozen | agreement among covered |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, values in summary["method_summaries"].items():
        lines.append(
            f"| `{method}` | {values['n_covered']} | {values['coverage']:.1%} | "
            f"{values['n_unresolved']} | {values['n_flips']} | "
            f"{values['agreement_with_frozen_among_covered']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Evidence-family records",
            "",
            "| family | records | parents with records | resolved family labels | flips vs frozen |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for family in FAMILIES:
        family_summary = summary["family_summaries"][family]
        lines.append(
            f"| `{family}` | {summary['evidence_family_record_counts'].get(family, 0)} | "
            f"{family_summary['n_molecules_with_records']}/{summary['n_valid_molecules']} | "
            f"{family_summary['n_molecules_with_resolved_label']} | "
            f"{family_summary['n_flips_vs_frozen']} |"
        )
    conflict = summary["cross_family_conflict"]
    lines.extend(
        [
            "",
            "## Cross-family conflicts",
            "",
            f"- Molecules with at least two resolved family labels that disagree: "
            f"{conflict['n_molecules']}/{summary['n_valid_molecules']} "
            f"({conflict['rate']:.1%}).",
            "- These rows are enumerated in `molecule_sensitivity.jsonl`; no conflict was "
            "silently resolved.",
            "",
            "## Interpretation boundary",
            "",
            "- `canonical_record_vote` isolates the canonical-source contract from the frozen source.",
            "- `canonical_quality_gated_*` additionally removes assay names that violate the "
            "canonical manifest's direct-only claim.",
            "- `canonical_study_vote` removes repeated-extraction weighting within one PMID.",
            "- `predictive_only_study_vote` is closest to the original induction-hazard/LLNA task, "
            "but low coverage is itself a result and must not be filled from diagnostic evidence.",
            "- Family classification is deterministic and inspectable, but generic human `patch test` "
            "rows remain a semantic limitation; they are marked by classification reason.",
            "",
            "## Flip cases",
            "",
        ]
    )
    for method in summary["method_summaries"]:
        flips = [
            row
            for row in rows
            if row["method_votes"][method]["label"] is not None
            and row["method_votes"][method]["label"] != row["frozen_label"]
        ]
        lines.append(f"### {method} ({len(flips)})")
        lines.append("")
        for row in flips:
            labels = ", ".join(
                f"{family}={label}" for family, label in row["observed_family_labels"].items()
            )
            lines.append(
                f"- query {row['query_index']}: {row['frozen_label']} -> "
                f"{row['method_votes'][method]['label']}; {labels or 'no resolved family labels'}"
            )
        if not flips:
            lines.append("- None.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _record_label(record: Mapping[str, Any]) -> int | None:
    label = _text(record.get("outcome_label")).lower().replace("-", "_").replace(" ", "_")
    if label in {"positive", "weak_positive"}:
        return 1
    if label == "negative":
        return 0
    return None


def _study_key(record: Mapping[str, Any]) -> str:
    pmid = _text(record.get("pmid"))
    if pmid:
        # The canonical partition can contain the same publication from both raw
        # acquisitions.  PMID, not acquisition lineage, defines the study vote.
        return f"pmid:{pmid}"
    source = _text(record.get("source_partition")) or "unknown_source"
    record_id = _text(record.get("source_record_id")) or _text(record.get("extraction_id"))
    return f"{source}:record:{record_id}"


def _empty_vote(status: str) -> dict[str, Any]:
    return {
        "label": None,
        "status": status,
        "n_votes": 0,
        "label_counts": {"0": 0, "1": 0},
        "agreement_fraction": None,
        "agreement_threshold": AGREEMENT_THRESHOLD,
    }


def _vote_result(
    label: int | None,
    counts: Counter[int],
    n_votes: int,
    status: str,
    threshold: float,
) -> dict[str, Any]:
    majority = max(counts.values()) if counts else 0
    return {
        "label": label,
        "status": status,
        "n_votes": n_votes,
        "label_counts": {"0": counts[0], "1": counts[1]},
        "agreement_fraction": majority / n_votes if n_votes else None,
        "agreement_threshold": threshold,
    }


def _label_counts(values: Iterable[int]) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {"0": counts[0], "1": counts[1]}


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        pass
    return str(value).strip()


def _json_safe(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-labels", type=Path, default=DEFAULT_VALID_LABELS)
    parser.add_argument("--canonical-direct", type=Path, default=DEFAULT_CANONICAL_DIRECT)
    parser.add_argument("--canonical-manifest", type=Path, default=DEFAULT_CANONICAL_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
