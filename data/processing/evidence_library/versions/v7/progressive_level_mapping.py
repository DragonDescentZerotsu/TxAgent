"""Assign current progressive levels to every normalized-v7 Stage 3 record."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v1.normalization.audit import write_parquet
from data.processing.evidence_library.shared.v1.normalization.cleaning import file_sha256


MAPPING_VERSION = "normalized_v7_stage3_progressive_levels.v1"
TASKS = ("bbb_martins", "bioavailability_ma", "skin_reaction")
STAGE3_PATHS = {
    task: Path(f"data/evidence_libraries/{task}/v7/03_pair_buckets/records.parquet")
    for task in TASKS
}
OUTPUT_DIRS = {
    task: Path(
        f"data/artifacts/evidence_library_assets/{task}/construction_assets/"
        "progressive_level_mapping_v1"
    )
    for task in TASKS
}
LEVEL_BY_GROUP = {
    "bbb_martins": {
        "Tier 1.starling_direct_bbb_evidence": "L1",
        "Proxy.central_functional_access": "L2",
        "Mechanism.passive_permeability": "L3",
        "Mechanism.efflux_transport": "L4",
        "Mechanism.influx_transport": "L5",
    },
    "bioavailability_ma": {
        "Observed.direct_oral_bioavailability": "L1",
        "Observed.nondirect_oral_bioavailability": "L2",
        "Observed.oral_auc_cmax_exposure": "L3",
        "Fa.absorption_solubility_permeability": "L4",
        "Fg.gut_wall_efflux_intestinal_metabolism": "L5",
        "Fh.hepatic_clearance_metabolic_stability": "L6",
    },
    "skin_reaction": {
        "Direct.skin_reaction": "L1",
        "Observed.nonvoter_skin_outcome": "L2",
        "Mechanism.sensitization_aop": "L3",
        "Mechanism.phototoxicity_irritation_local_damage": "L4",
    },
}
CACHE_LEVELS = {
    "bbb_martins": {"L2", "L3", "L4", "L5"},
    "bioavailability_ma": {"L2", "L3", "L4", "L5", "L6"},
    "skin_reaction": {"L3", "L4"},
}
DIRECT_EXTENSION_SOURCE = {
    "bbb_martins": "direct_bbb",
    "bioavailability_ma": "hf_bioavailability",
    "skin_reaction": "direct_skin_reaction",
}
EXPECTED_CACHE_EXCLUSIONS = {
    "bbb_martins": {"unresolved_endpoint": 260, "unresolved_parent_identity": 5},
    "bioavailability_ma": {"unresolved_endpoint": 0, "unresolved_parent_identity": 1},
    "skin_reaction": {"unresolved_endpoint": 314, "unresolved_parent_identity": 16},
}


def build_mapping(
    task_id: str,
    *,
    records_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write one current-level sidecar row per Stage 3 canonical record."""
    if task_id not in TASKS:
        raise ValueError(f"unsupported task: {task_id}")
    source = Path(records_path or STAGE3_PATHS[task_id])
    target = Path(output_dir or OUTPUT_DIRS[task_id])
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"mapping output is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)

    records = pq.read_table(source).to_pylist()
    print(f"{task_id}: loaded {len(records)} Stage 3 records", flush=True)
    record_ids = [str(row.get("canonical_record_id") or "") for row in records]
    if not all(record_ids) or len(record_ids) != len(set(record_ids)):
        raise ValueError("Stage 3 canonical_record_id values must be nonempty and unique")

    direct_inputs: Sequence[Path]
    current_votes: set[str]
    if task_id == "bbb_martins":
        from data.processing.evidence_library.versions.v7.tasks.bbb_martins.direct_record_mapping import (
            DIRECT_MAPPING_INPUTS,
            build_direct_record_mapping,
        )
        from tools.chembl_tool.tasks.bbb_martins.source_family_purity import (
            BBBSourceFamilyClassifier,
            PURITY_VERSION,
            load_near_direct_reviews,
        )
        direct_inputs = DIRECT_MAPPING_INPUTS
        direct_rows = build_direct_record_mapping(records)
        current_votes = {
            str(row["canonical_record_id"])
            for row in direct_rows
            if row["direct_vote_status"] == "counted"
        }
        vote_indices = {
            int(row["source_index"])
            for row in records
            if str(row["canonical_record_id"]) in current_votes
            and row.get("source_index") is not None
        }
        classifier = BBBSourceFamilyClassifier(load_near_direct_reviews(), vote_indices)
        family_version = PURITY_VERSION
    elif task_id == "bioavailability_ma":
        from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.direct_record_mapping import (
            DIRECT_MAPPING_INPUTS,
            build_direct_record_mapping,
        )
        from tools.chembl_tool.tasks.bioavailability_ma.source_family_purity import (
            VOTE_PURITY_VERSION,
            upstream_record_key,
            vote_pure_family_move,
        )

        direct_inputs = DIRECT_MAPPING_INPUTS
        direct_rows = build_direct_record_mapping(records)
        current_votes = {
            str(row["canonical_record_id"])
            for row in direct_rows
            if row["direct_vote_status"] == "counted"
        }
        voter_source_ids = {
            upstream_record_key(row)
            for row in records
            if str(row["canonical_record_id"]) in current_votes
        }
        voter_source_ids.discard("")
        family_version = VOTE_PURITY_VERSION
    else:
        from data.processing.evidence_library.versions.v7.tasks.skin_reaction.canonical_starling_source import (
            direct_outcome_reason,
        )
        from data.processing.evidence_library.versions.v7.tasks.skin_reaction.direct_record_mapping import (
            DIRECT_MAPPING_INPUTS,
            build_direct_record_mapping,
        )

        direct_inputs = DIRECT_MAPPING_INPUTS
        direct_rows = build_direct_record_mapping(records)
        current_votes = {
            str(row["canonical_record_id"])
            for row in direct_rows
            if row["direct_vote_status"] == "counted"
            and row["source_id"] == "direct_skin_reaction"
        }
        family_version = "skin_source_family_purity.vote_pure.v1"

    mapped_rows: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str]] = Counter()
    unresolved_endpoint_rows = 0
    unresolved_parent_rows = 0
    for row_number, row in enumerate(records, start=1):
        record_id = str(row["canonical_record_id"])
        original_group = str(row.get("group_id") or "")
        is_current_vote = record_id in current_votes

        if task_id == "bbb_martins":
            move = classifier.classify_target(row)
            assigned_group = move.new_group
            assignment_reason = move.reason
        elif task_id == "bioavailability_ma":
            move = vote_pure_family_move(row, voter_source_ids)
            assigned_group = move.new_group or original_group
            assignment_reason = move.reason or "current_group_retained"
        else:
            if is_current_vote:
                assigned_group = "Direct.skin_reaction"
                assignment_reason = "accepted_gold_vote_promoted_to_l1"
            elif original_group == "Direct.skin_reaction":
                assigned_group = "Observed.nonvoter_skin_outcome"
                assignment_reason = "nonvoter_removed_from_l1"
            elif original_group == "Mechanism.sensitization_aop" and (
                reason := direct_outcome_reason(row)
            ):
                assigned_group = "Observed.nonvoter_skin_outcome"
                assignment_reason = f"direct_like_nonvoter_to_l2:{reason}"
            else:
                assigned_group = original_group
                assignment_reason = "current_group_retained"

        level = LEVEL_BY_GROUP[task_id].get(assigned_group)
        mapping_status = "assigned" if level else "outside_progressive_scope"
        cache_candidate_eligible = False
        cache_exclusion_reason = "task_checkpoint_not_supplied"
        domain_flags = ["task_checkpoint_not_supplied"]
        if task_id in CACHE_LEVELS:
            from predict.retrieval.policies import normalize_molecule_identity

            parent_identity = normalize_molecule_identity(
                str(row.get("canonical_smiles") or "")
            )
            parent_identity_resolvable = bool(
                parent_identity.status == "ok" and parent_identity.parent_smiles
            )
            unresolved_parent_rows += int(not parent_identity_resolvable)
            domain_flags = []
            unresolved_endpoint = (
                str(row.get("source_id") or "") != DIRECT_EXTENSION_SOURCE[task_id]
                and not bool(row.get("assay_transfer_eligible"))
                and str(row.get("assay_transfer_ineligibility_reason") or "")
                == "unresolved_endpoint"
                and str(row.get("canonical_endpoint_name") or "")
                == "missing_endpoint"
            )
            unresolved_endpoint_rows += int(unresolved_endpoint)
            if str(row.get("source_id") or "") == DIRECT_EXTENSION_SOURCE[task_id]:
                domain_flags.append("ood_direct_source_extension")
            if row.get("finite_scalar_value") is None:
                domain_flags.append("ood_nonnumeric_measurement")
            if not bool(row.get("assay_transfer_eligible")) and not unresolved_endpoint:
                domain_flags.append("ood_assay_transfer_ineligible")
            if unresolved_endpoint:
                domain_flags.append("excluded_unresolved_endpoint")
            if not parent_identity_resolvable:
                domain_flags.append("excluded_unresolved_parent_identity")
            cache_candidate_eligible = bool(
                level in CACHE_LEVELS[task_id]
                and not unresolved_endpoint
                and parent_identity_resolvable
            )
            cache_exclusion_reason = (
                f"retained_for_{task_id}_score_cache"
                if cache_candidate_eligible
                else "global_l1_exclusion"
                if level == "L1"
                else "unresolved_endpoint"
                if unresolved_endpoint
                else "unresolved_parent_identity"
                if not parent_identity_resolvable
                else "outside_progressive_scope"
            )

        counts[(level or "outside", assigned_group or "missing")] += 1
        mapped_rows.append(
            {
                "mapping_version": MAPPING_VERSION,
                "family_policy_version": family_version,
                "task_id": task_id,
                "canonical_record_id": record_id,
                "source_id": str(row.get("source_id") or ""),
                "source_row_number": row.get("source_row_number"),
                "source_record_id": str(row.get("source_record_id") or ""),
                "retrieval_source_id_observed": str(row.get("retrieval_source_id") or ""),
                "current_direct_vote": is_current_vote,
                "original_group_id": original_group,
                "assigned_group_id": assigned_group,
                "progressive_level": level,
                "mapping_status": mapping_status,
                "assignment_reason": assignment_reason,
                "assay_transfer_eligible": bool(row.get("assay_transfer_eligible")),
                "assay_transfer_ineligibility_reason": str(
                    row.get("assay_transfer_ineligibility_reason") or ""
                ),
                "finite_scalar_value_present": row.get("finite_scalar_value") is not None,
                "score_cache_candidate_eligible": cache_candidate_eligible,
                "score_cache_exclusion_reason": cache_exclusion_reason,
                "scoring_domain_flags": domain_flags,
            }
        )
        if row_number % 100_000 == 0:
            print(f"{task_id}: mapped {row_number}/{len(records)} records", flush=True)

    expected_exclusions = EXPECTED_CACHE_EXCLUSIONS.get(task_id)
    if (
        expected_exclusions
        and unresolved_endpoint_rows != expected_exclusions["unresolved_endpoint"]
    ):
        raise ValueError(
            f"current {task_id} Stage 3 unresolved-endpoint count drifted: "
            f"expected={expected_exclusions['unresolved_endpoint']}, "
            f"observed={unresolved_endpoint_rows}"
        )
    if (
        expected_exclusions
        and unresolved_parent_rows
        != expected_exclusions["unresolved_parent_identity"]
    ):
        raise ValueError(
            f"current {task_id} Stage 3 parent-identity count drifted: "
            f"expected={expected_exclusions['unresolved_parent_identity']}, "
            f"observed={unresolved_parent_rows}"
        )

    output_path = target / "record_levels.parquet"
    write_parquet(output_path, mapped_rows)
    input_paths = [source, *direct_inputs]
    if task_id == "bbb_martins":
        from tools.chembl_tool.tasks.bbb_martins.source_family_purity import (
            DEFAULT_REVIEW_LEDGER,
        )

        input_paths.append(DEFAULT_REVIEW_LEDGER)
    manifest = {
        "mapping_version": MAPPING_VERSION,
        "family_policy_version": family_version,
        "task_id": task_id,
        "stage": "normalized_v7_stage3_pair_buckets",
        "rows": len(mapped_rows),
        "current_direct_mapping_votes": len(current_votes),
        "direct_mapping_votes_outside_l1": sum(
            row["current_direct_vote"] and row["progressive_level"] != "L1"
            for row in mapped_rows
        ),
        "levels": {
            level: int(sum(count for (found, _), count in counts.items() if found == level))
            for level in sorted(set(LEVEL_BY_GROUP[task_id].values()))
        },
        "outside_progressive_scope": int(
            sum(count for (level, _), count in counts.items() if level == "outside")
        ),
        "counts_by_level_and_group": {
            f"{level}:{group}": count
            for (level, group), count in sorted(counts.items())
        },
        f"{task_id}_score_cache_contract": {
            "global_exclusion": (
                f"L1 records, {unresolved_endpoint_rows} unresolved-endpoint records, "
                f"and {unresolved_parent_rows} records "
                "without a resolvable parent identity"
            ),
            "per_query_exclusion": "scaffold_disjoint",
            "morgan_pool_per_level": 75,
            "minimum_similarity": 0.0,
            "unresolved_endpoint_rows": unresolved_endpoint_rows,
            "unresolved_parent_identity_rows": unresolved_parent_rows,
            "other_out_of_domain_rows_retained_and_flagged": True,
        } if task_id in CACHE_LEVELS else None,
        "inputs": {
            str(path): file_sha256(path)
            for path in input_paths
            if Path(path).is_file()
        },
        "output": {
            "path": str(output_path),
            "sha256": file_sha256(output_path),
        },
        "validations": {
            "one_sidecar_row_per_stage3_record": len(mapped_rows) == len(records),
            "canonical_record_ids_unique": len(record_ids) == len(set(record_ids)),
            "skin_outside_scope_not_assigned_a_level": task_id != "skin_reaction" or all(
                row["progressive_level"] is None
                for row in mapped_rows
                if row["mapping_status"] == "outside_progressive_scope"
            ),
        },
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="*", choices=TASKS, default=list(TASKS))
    args = parser.parse_args(argv)
    for task_id in args.tasks:
        print(json.dumps(build_mapping(task_id), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
