"""Build the BBB retrieval-only source-purity overlay and safety audit."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq

from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic
from tools.chembl_tool.paper_experiments.build_conditioned_source_family_purity import (
    PuritySpec,
    build_overlay,
)
from tools.chembl_tool.tasks.bbb_martins.source_family_purity import (
    BBBSourceFamilyClassifier,
    CLASSIFIER_COLUMNS,
    DEFAULT_CONDITIONED_ROOT,
    DEFAULT_GOLD_ROOT,
    DEFAULT_RECORDS,
    DEFAULT_REVIEW_LEDGER,
    DIRECT_GROUP,
    PURITY_VERSION,
    gold_contract_decision,
    gold_contract_decision_v4,
    is_mdck_record,
    is_explicit_prediction_record,
    is_pampa_record,
    load_gold_vote_source_indices,
    load_near_direct_reviews,
)
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_benchmark import (
    NEGATIVE_LABELS,
    POSITIVE_LABELS,
)


DEFAULT_OUTPUT = Path(
    "outputs/paper/starling_conditioned_assay_family_curve_v1/"
    "source_overlays/bbb_source_family_purity_v5"
)
DEFAULT_GOLD_MIGRATION = (
    DEFAULT_GOLD_ROOT / "migration_from_v3.json"
)


def _gold_contract(name: str):
    if name == "v3":
        return gold_contract_decision, "experimental_meaningful_cns_access_v3"
    if name == "v4":
        return gold_contract_decision_v4, "experimental_meaningful_cns_access_v4"
    raise ValueError(f"unsupported BBB gold contract: {name}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _record_indices(row: dict[str, Any]) -> set[int]:
    indices: set[int] = set()
    for record_id in row.get("source_record_ids") or ():
        match = re.search(r"(?:^|:)row:(\d+)$", str(record_id))
        if match:
            indices.add(int(match.group(1)))
    return indices


def _vote_status(counts: Counter[int], threshold: float) -> tuple[int | None, str]:
    total = sum(counts.values())
    if not total:
        return None, "no_votes"
    if counts[0] == counts[1]:
        return None, "exact_tie"
    label = 1 if counts[1] > counts[0] else 0
    if counts[label] / total < threshold:
        return None, "below_agreement_threshold"
    return label, "accepted"


def _gold_vote_sensitivity(
    output_path: Path,
    moved_gold_indices: set[int],
    *,
    gold_root: Path = DEFAULT_GOLD_ROOT,
) -> dict[str, Any]:
    """Audit frozen labels after excluding prediction rows from retrieval L1."""

    if not moved_gold_indices:
        return {
            "n_removed_historical_prediction_votes": 0,
            "n_affected_parents": 0,
            "n_binary_label_flips": 0,
            "n_parent_acceptance_losses": 0,
            "n_valid_or_test_parents_affected": 0,
            "affected_parents": [],
        }
    columns = ["source_index", "bbb_permeability_label"]
    labels: dict[int, int] = {}
    for row in pq.read_table(output_path, columns=columns).to_pylist():
        index = row.get("source_index")
        if index is None or int(index) not in moved_gold_indices:
            continue
        value = str(row.get("bbb_permeability_label") or "").strip().lower()
        if value in POSITIVE_LABELS:
            labels[int(index)] = 1
        elif value in NEGATIVE_LABELS:
            labels[int(index)] = 0
        else:
            raise RuntimeError(
                f"moved historical vote has no binary label: source_index={index}"
            )

    affected: list[dict[str, Any]] = []
    matched_indices: set[int] = set()
    seen_parents: set[str] = set()
    for artifact_name in (
        "molecule_labels.jsonl",
        "conflicting_molecules.jsonl",
        "rejected_parent_molecules.jsonl",
    ):
        for row in _read_jsonl(gold_root / artifact_name):
            removed = sorted(_record_indices(row) & moved_gold_indices)
            if not removed:
                continue
            matched_indices.update(removed)
            parent_key = str(row.get("molecule_identity_key") or row.get("drug") or "")
            if parent_key in seen_parents:
                continue
            seen_parents.add(parent_key)
            old_counts = Counter(
                {int(label): int(count) for label, count in row["label_counts"].items()}
            )
            new_counts = old_counts.copy()
            for index in removed:
                new_counts[labels[index]] -= 1
            threshold = float(row.get("agreement_threshold", 0.70))
            new_label, new_status = _vote_status(new_counts, threshold)
            old_label = row.get("Y")
            if old_label is None:
                old_label = row.get("majority_label") if "drop_reason" not in row else None
            old_status = "accepted" if row.get("Y") is not None else str(
                row.get("drop_reason") or "not_accepted"
            )
            splits = row.get("split_assignments") or {}
            affected.append(
                {
                    "drug": row.get("drug"),
                    "removed_source_indices": removed,
                    "old_counts": {str(k): int(v) for k, v in sorted(old_counts.items())},
                    "new_counts": {str(k): int(v) for k, v in sorted(new_counts.items())},
                    "old_label": old_label,
                    "new_label": new_label,
                    "old_status": old_status,
                    "new_status": new_status,
                    "split_assignments": splits,
                }
            )
    unmatched = sorted(moved_gold_indices - matched_indices)
    if unmatched:
        raise RuntimeError(
            f"moved gold-vote indices absent from frozen vote artifacts: {unmatched}"
        )
    return {
        "n_removed_historical_prediction_votes": len(moved_gold_indices),
        "n_affected_parents": len(affected),
        "n_binary_label_flips": sum(
            row["old_label"] is not None
            and row["new_label"] is not None
            and row["old_label"] != row["new_label"]
            for row in affected
        ),
        "n_parent_acceptance_losses": sum(
            row["old_label"] is not None and row["new_label"] is None
            for row in affected
        ),
        "n_scaffold_valid_or_test_parents_affected": sum(
            row["split_assignments"].get("scaffold") in {"valid", "test"}
            for row in affected
        ),
        "n_scaffold_valid_or_test_label_changes": sum(
            row["split_assignments"].get("scaffold") in {"valid", "test"}
            and row["old_label"] != row["new_label"]
            for row in affected
        ),
        "affected_parents": affected,
    }


def finalize_manifest(
    records_path: Path,
    review_path: Path,
    output_dir: Path,
    *,
    gold_root: Path = DEFAULT_GOLD_ROOT,
    conditioned_root: Path | None = None,
    gold_migration_path: Path = DEFAULT_GOLD_MIGRATION,
    gold_contract_name: str = "v4",
) -> dict:
    decision_fn, gold_lineage = _gold_contract(gold_contract_name)
    gold_indices = load_gold_vote_source_indices(
        gold_root=gold_root,
        conditioned_root=conditioned_root or Path("/__no_conditioned_votes__"),
    )
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = pq.read_table(
        output_dir / "moved_records.parquet",
        columns=["source_index", "purity_reason"],
    )
    moved_indices = {
        int(index) for index in audit["source_index"].to_pylist() if index is not None
    }
    overlap = sorted(moved_indices & gold_indices)

    output_path = output_dir / "records.parquet"
    overlap_rows: dict[int, dict[str, Any]] = {}
    for batch in pq.ParquetFile(output_path).iter_batches(
        batch_size=10_000,
        columns=list(CLASSIFIER_COLUMNS),
    ):
        for row in batch.to_pylist():
            index = row.get("source_index")
            if index is not None and int(index) in overlap:
                overlap_rows[int(index)] = row
    moved_audit = pq.read_table(
        output_dir / "moved_records.parquet",
        columns=["source_index", "new_group_id"],
    ).to_pylist()
    moved_gold_to_direct = {
        int(row["source_index"])
        for row in moved_audit
        if row.get("source_index") is not None
        and int(row["source_index"]) in overlap
        and row.get("new_group_id") == DIRECT_GROUP
    }
    moved_gold_outside_direct = set(overlap) - moved_gold_to_direct
    nonprediction_overlap = sorted(
        index
        for index in moved_gold_outside_direct
        if not is_explicit_prediction_record(overlap_rows[index])
    )
    if nonprediction_overlap:
        raise RuntimeError(
            "gold voters moved outside L1 are not explicit predictions: "
            f"{nonprediction_overlap[:10]}"
        )
    gold_vote_sensitivity = _gold_vote_sensitivity(
        output_path, moved_gold_outside_direct, gold_root=gold_root
    )
    if not gold_migration_path.exists():
        raise FileNotFoundError(
            "BBB gold migration receipt must exist before publishing the "
            "source-family overlay"
        )
    gold_migration = json.loads(gold_migration_path.read_text(encoding="utf-8"))
    identity_columns = [
        "group_id",
        "canonical_assay_context",
        "assay_model",
        "assay_system",
        "biological_system",
        "canonical_assay_type",
        "assay_type",
    ]
    residual_pampa = residual_mdck = 0
    for batch in pq.ParquetFile(output_path).iter_batches(
        batch_size=5_000,
        columns=identity_columns,
    ):
        for row in batch.to_pylist():
            if row.get("group_id") != DIRECT_GROUP:
                continue
            pampa = is_pampa_record(row)
            residual_pampa += int(pampa)
            residual_mdck += int(not pampa and is_mdck_record(row))
    if residual_pampa or residual_mdck:
        raise RuntimeError(
            f"residual L1 assay rows: pampa={residual_pampa}, mdck={residual_mdck}"
        )
    group_counts = {
        str(row["values"]): int(row["counts"])
        for row in pc.value_counts(
            pq.read_table(output_path, columns=["group_id"])["group_id"]
        ).to_pylist()
    }
    moved_rows = pq.read_table(
        output_path,
        columns=[
            "source_family_purity_version",
            "assay_model",
            "canonical_assay_context",
            "canonical_endpoint_name",
            "support_text",
        ],
    )
    moved_rows = moved_rows.filter(
        pc.is_valid(moved_rows["source_family_purity_version"])
    ).to_pandas()
    searchable = (
        moved_rows[["assay_model", "canonical_assay_context", "support_text"]]
        .fillna("")
        .agg(" | ".join, axis=1)
    )
    endpoint = moved_rows["canonical_endpoint_name"].fillna("").str.lower()
    n_prediction_like = int(
        searchable.str.contains(
            r"in[ -]?silico|predict|computational|calculated|qppmdck|model result",
            case=False,
            regex=True,
        ).sum()
    )
    n_missing_endpoint = int(endpoint.isin(("", "missing_endpoint")).sum())
    n_generic_endpoint = int(endpoint.eq("bbb_permeability_outcome").sum())

    residual_noncontract_l1 = 0
    residual_prediction_l1 = 0
    n_contract_eligible_l1 = 0
    n_frozen_vote_l1 = 0
    for batch in pq.ParquetFile(output_path).iter_batches(
        batch_size=5_000,
        columns=list(CLASSIFIER_COLUMNS),
    ):
        for row in batch.to_pylist():
            if row.get("group_id") != DIRECT_GROUP:
                continue
            if is_explicit_prediction_record(row):
                residual_prediction_l1 += 1
                continue
            index = row.get("source_index")
            try:
                normalized_index = int(index) if index is not None else None
            except (TypeError, ValueError):
                normalized_index = None
            if normalized_index in gold_indices:
                n_frozen_vote_l1 += 1
                continue
            label, _ = decision_fn(row)
            if label is None:
                residual_noncontract_l1 += 1
            else:
                n_contract_eligible_l1 += 1
    if residual_noncontract_l1:
        raise RuntimeError(
            "L1 contains non-gold-compatible non-voter rows: "
            f"{residual_noncontract_l1}"
        )
    if residual_prediction_l1:
        raise RuntimeError(f"L1 contains explicit prediction rows: {residual_prediction_l1}")

    manifest.update(
        {
            "near_direct_review": str(review_path.resolve()),
            "near_direct_review_sha256": sha256_file(review_path),
            "n_gold_vote_source_indices": len(gold_indices),
            "n_moved_gold_vote_rows": len(overlap),
            "moved_gold_vote_source_indices": overlap,
            "n_moved_gold_vote_rows_into_l1": len(moved_gold_to_direct),
            "n_moved_gold_vote_rows_outside_l1": len(moved_gold_outside_direct),
            "gold_vote_sensitivity": gold_vote_sensitivity,
            "gold_lineage_migration": gold_migration,
            "n_residual_l1_pampa_rows": residual_pampa,
            "n_residual_l1_mdck_rows": residual_mdck,
            "output_group_counts": dict(sorted(group_counts.items())),
            "l1_contract": (
                "frozen/current gold-vote source OR conditioned-compatible replay of "
                f"{gold_lineage}"
            ),
            "n_frozen_or_conditioned_vote_rows_in_l1": n_frozen_vote_l1,
            "n_additional_gold_contract_eligible_rows_in_l1": n_contract_eligible_l1,
            "n_residual_non_gold_contract_rows_in_l1": residual_noncontract_l1,
            "n_residual_prediction_rows_in_l1": residual_prediction_l1,
            "prediction_rows_retained_outside_l1": True,
            "missing_or_generic_rows_retained_outside_l1": True,
            "n_moved_prediction_like_rows": n_prediction_like,
            "n_moved_missing_endpoint_rows": n_missing_endpoint,
            "n_moved_generic_bbb_endpoint_rows": n_generic_endpoint,
            "family_assignment_unit": "source_record",
            "assay_identity_role": "card provenance and diversity only",
        }
    )
    write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def build(
    records_path: Path,
    review_path: Path,
    output_dir: Path,
    *,
    gold_root: Path = DEFAULT_GOLD_ROOT,
    conditioned_root: Path | None = None,
    gold_migration_path: Path = DEFAULT_GOLD_MIGRATION,
    gold_contract_name: str = "v4",
    purity_version: str = PURITY_VERSION,
) -> dict:
    decision_fn, _ = _gold_contract(gold_contract_name)
    gold_indices = load_gold_vote_source_indices(
        gold_root=gold_root,
        conditioned_root=conditioned_root or Path("/__no_conditioned_votes__"),
    )
    reviews = load_near_direct_reviews(review_path)
    classifier = BBBSourceFamilyClassifier(
        reviews, gold_indices, gold_decision=decision_fn
    )
    spec = PuritySpec(
        task="bbb_martins",
        input_records=records_path,
        direct_group=DIRECT_GROUP,
        classify=classifier,
        purity_version=purity_version,
        allow_move_from_direct=True,
        classifier_columns=CLASSIFIER_COLUMNS,
    )
    build_overlay(spec, output_dir, batch_size=500)
    return finalize_manifest(
        records_path,
        review_path,
        output_dir,
        gold_root=gold_root,
        conditioned_root=conditioned_root,
        gold_migration_path=gold_migration_path,
        gold_contract_name=gold_contract_name,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=str(DEFAULT_RECORDS))
    parser.add_argument("--review-ledger", default=str(DEFAULT_REVIEW_LEDGER))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--gold-root", default=str(DEFAULT_GOLD_ROOT))
    parser.add_argument("--conditioned-root", default=str(DEFAULT_CONDITIONED_ROOT))
    parser.add_argument("--gold-migration", default=str(DEFAULT_GOLD_MIGRATION))
    parser.add_argument("--gold-contract", choices=("v3", "v4"), default="v4")
    parser.add_argument("--purity-version", default=PURITY_VERSION)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args(argv)
    build_fn = finalize_manifest if args.finalize_only else build
    print(
        json.dumps(
            build_fn(
                Path(args.records),
                Path(args.review_ledger),
                Path(args.output_dir),
                gold_root=Path(args.gold_root),
                conditioned_root=Path(args.conditioned_root),
                gold_migration_path=Path(args.gold_migration),
                gold_contract_name=args.gold_contract,
                **({"purity_version": args.purity_version} if not args.finalize_only else {}),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
