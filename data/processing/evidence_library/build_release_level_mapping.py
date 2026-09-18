"""Build release-owned complete levels or a Gold-owned mapped subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from tools.chembl_tool.common.json_utils import write_json_atomic


LEVELS = {
    "bbb_martins": {
        "Tier 1.starling_direct_bbb_evidence": ("direct_brain_exposure", 1),
        "Proxy.central_functional_access": ("central_functional_access_proxy", 2),
        "Mechanism.passive_permeability": ("passive_permeability", 3),
        "Mechanism.efflux_transport": ("efflux_transport", 4),
        "Mechanism.influx_transport": ("influx_transport", 5),
    },
    "bioavailability_ma": {
        "Observed.direct_oral_bioavailability": ("direct_oral_bioavailability", 1),
        "Observed.nondirect_oral_bioavailability": ("nondirect_oral_bioavailability", 2),
        "Observed.oral_auc_cmax_exposure": ("oral_auc_cmax_exposure", 3),
        "Fa.absorption_solubility_permeability": ("absorption_solubility_permeability", 4),
        "Fg.gut_wall_efflux_intestinal_metabolism": ("gut_wall_efflux_intestinal_metabolism", 5),
        "Fh.hepatic_clearance_metabolic_stability": ("hepatic_clearance_metabolic_stability", 6),
    },
}


def _write(
    records: pd.DataFrame,
    output_root: Path,
    manifest: dict,
    eligibility: pd.DataFrame | None = None,
    published_output_root: Path | None = None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "records.parquet"
    records.to_parquet(path, index=False)
    manifest["output"] = {
        "path": str((published_output_root or output_root) / path.name),
        "sha256": file_sha256(path),
        "rows": len(records),
        "unique_source_row_uids": int(records.source_row_uid.nunique()),
        "rows_by_level": {
            str(level): int(count)
            for level, count in records.groupby("level").size().items()
        },
    }
    if eligibility is not None:
        eligibility_path = output_root / "assay_transfer_record_eligibility.parquet"
        eligibility.to_parquet(eligibility_path, index=False)
        manifest["assay_transfer_record_eligibility"] = {
            "version": "assay_transfer_record_eligibility.v1",
            "path": str(
                (published_output_root or output_root) / eligibility_path.name
            ),
            "sha256": file_sha256(eligibility_path),
            "rows": len(eligibility),
            "scope": "record-level scientific suitability; no bucket support or variance gates",
            "rows_by_level_and_eligibility": {
                str(level): {
                    str(bool(eligible)).lower(): int(count)
                    for eligible, count in level_rows.groupby(
                        "assay_transfer_eligible"
                    ).size().items()
                }
                for level, level_rows in eligibility.groupby("level")
            },
            "ineligibility_reason_counts": {
                str(reason): int(count)
                for reason, count in eligibility.loc[
                    ~eligibility.assay_transfer_eligible,
                    "assay_transfer_ineligibility_reason",
                ].value_counts().items()
            },
        }
    write_json_atomic(output_root / "manifest.json", manifest)


def build_complete(
    task: str,
    records_path: Path,
    prior_path: Path,
    voter_path: Path,
    output_root: Path,
    release_version: str,
    published_records_path: Path | None,
    published_output_root: Path | None = None,
    published_prior_mapping_path: Path | None = None,
) -> None:
    records = pd.read_parquet(records_path)
    prior = pd.read_parquet(prior_path)
    voters = set(pd.read_parquet(voter_path).source_row_uid.astype(str))
    record_uids = set(records.source_row_uid.astype(str))
    if not voters <= record_uids:
        raise ValueError(f"release lacks {len(voters - record_uids)} protected voters")
    prior = prior.drop(columns="canonical_record_id")
    mapped = records.merge(prior, on="source_row_uid", how="left", validate="one_to_one")
    missing = mapped.level.isna()
    if missing.any():
        routed = _route_unmapped(task, mapped.loc[missing], voters, records)
        for column in ("source_group_id", "family_key", "level"):
            mapped[column] = mapped[column].astype(object)
            mapped.loc[missing, column] = routed[column].tolist()
    mapped.loc[mapped.source_row_uid.astype(str).isin(voters), ["source_group_id", "family_key", "level"]] = [
        next(group for group, (_, level) in LEVELS[task].items() if level == 1),
        next(family for family, level in LEVELS[task].values() if level == 1),
        1,
    ]
    nonvoter_l1 = mapped.loc[mapped.level.eq(1) & ~mapped.source_row_uid.astype(str).isin(voters)]
    if len(nonvoter_l1) or mapped.level.isna().any():
        raise ValueError("release level routing violated complete exact-L1 coverage")
    eligibility_columns = [
        "source_row_uid",
        "canonical_record_id",
        "level",
        "pair_bucket_key",
        "measurement_kind",
        "assay_transfer_eligible",
        "assay_transfer_ineligibility_reason",
    ]
    if missing_columns := set(eligibility_columns) - set(mapped.columns):
        raise ValueError(
            "Stage 3 lacks assay-transfer eligibility fields: "
            f"{sorted(missing_columns)}"
        )
    output_columns = ["source_row_uid", "canonical_record_id", "source_group_id", "family_key", "level"]
    mapped["level"] = mapped.level.astype("int64")
    eligibility = mapped[eligibility_columns].copy()
    eligibility["assay_transfer_eligible"] = eligibility[
        "assay_transfer_eligible"
    ].astype(bool)
    missing_reason = eligibility.assay_transfer_ineligibility_reason.fillna("").eq("")
    if not missing_reason.eq(eligibility.assay_transfer_eligible).all():
        raise ValueError("assay-transfer eligibility and reason fields disagree")
    if eligibility.pair_bucket_key.fillna("").eq("").any():
        raise ValueError("assay-transfer eligibility row lacks pair_bucket_key")
    eligibility = eligibility.sort_values("source_row_uid").reset_index(drop=True)
    mapped = mapped[output_columns].sort_values("source_row_uid").reset_index(drop=True)
    _write(
        mapped,
        output_root,
        {
            "version": "evidence_library_level_mapping.v1",
            "status": "complete",
            "task": task,
            "evidence_library_version": release_version,
            "gold_release": "v1",
            "scope": "Complete retained Stage-3 UID universe with exact Gold-v1 physical voter membership at L1",
            "inputs": {
                "stage3_records": {"path": str(published_records_path or records_path), "sha256": file_sha256(records_path)},
                "prior_mapping": {
                    "path": str(published_prior_mapping_path or prior_path),
                    "sha256": file_sha256(prior_path),
                },
                "voter_membership": {"path": str(voter_path), "sha256": file_sha256(voter_path), "physical_voters": len(voters)},
            },
            "validation": {"stage3_uid_coverage": "exact", "l1_equals_gold_physical_voters": True},
            "newly_routed_rows": int(missing.sum()),
        },
        eligibility,
        published_output_root,
    )


def _route_unmapped(
    task: str, rows: pd.DataFrame, voters: set[str], all_records: pd.DataFrame
) -> pd.DataFrame:
    """Apply the existing reviewed purity classifier only to new physical UIDs."""
    if task == "bbb_martins":
        from tools.chembl_tool.tasks.bbb_martins.source_family_purity import (
            BBBSourceFamilyClassifier, load_near_direct_reviews,
        )
        voter_indices = {
            int(value) for value in all_records.loc[
                all_records.source_row_uid.astype(str).isin(voters), "source_index"
            ].dropna()
        }
        classifier = BBBSourceFamilyClassifier(load_near_direct_reviews(), voter_indices)
        groups = [
            classifier.classify_target(row).new_group or row["group_id"]
            for row in rows.to_dict("records")
        ]
    elif task == "bioavailability_ma":
        from tools.chembl_tool.tasks.bioavailability_ma.source_family_purity import (
            upstream_record_key, vote_pure_family_move,
        )
        voter_rows = all_records.loc[all_records.source_row_uid.astype(str).isin(voters)]
        voter_keys = {upstream_record_key(row) for row in voter_rows.to_dict("records")}
        groups = []
        for row in rows.to_dict("records"):
            move = vote_pure_family_move(row, voter_keys)
            groups.append(move.new_group or row["group_id"])
    else:
        raise ValueError(f"complete routing is unsupported for {task}")
    result = pd.DataFrame({"source_group_id": groups}, index=rows.index)
    result[["family_key", "level"]] = [LEVELS[task][group] for group in groups]
    return result


def build_subset(
    task: str,
    records_path: Path,
    prior_path: Path,
    output_root: Path,
    source_value_cleaning_audit: Path | None = None,
    published_records_path: Path | None = None,
    published_output_root: Path | None = None,
    published_prior_mapping_path: Path | None = None,
    published_cleaning_audit_path: Path | None = None,
) -> None:
    records = pd.read_parquet(records_path)[["source_row_uid", "canonical_record_id"]]
    prior = pd.read_parquet(prior_path)[["source_row_uid", "level"]]
    records["source_row_uid"] = records.source_row_uid.astype(str)
    prior["source_row_uid"] = prior.source_row_uid.astype(str)
    record_uids = set(records.source_row_uid)
    missing = set(prior.source_row_uid) - record_uids
    lineage = None
    if missing:
        if source_value_cleaning_audit is None:
            raise ValueError(f"Stage 3 lacks {len(missing)} Gold-owned mapped UIDs")
        audit = pd.read_parquet(source_value_cleaning_audit)
        audit["source_row_uid"] = audit.source_row_uid.astype(str)
        audit = audit.loc[
            audit.source_row_uid.isin(missing)
            & audit.field.eq("record")
            & audit["after"].eq("dropped")
        ].copy()
        if (
            set(audit.source_row_uid) != missing
            or audit.source_row_uid.duplicated().any()
        ):
            raise ValueError(
                "Stage-1 lineage does not resolve every missing Gold mapping UID exactly once"
            )
        duplicate = audit.cleaning_version.eq(
            "skin_reaction_stage1_exact_deduplication.v1"
        )
        reviewed_drop = audit.rule_id.astype(str).str.startswith("reviewed_drop:")
        if not (duplicate | reviewed_drop).all():
            raise ValueError("missing Gold mapping UID has unsupported Stage-1 lineage")
        remapped = audit.loc[duplicate, ["source_row_uid", "retained_source_row_uid"]]
        remapped["retained_source_row_uid"] = remapped.retained_source_row_uid.astype(str)
        if not set(remapped.retained_source_row_uid) <= record_uids:
            raise ValueError("Stage-1 duplicate survivor is absent from Stage 3")
        remapped = remapped.merge(prior, on="source_row_uid", validate="one_to_one")
        remapped = remapped[["retained_source_row_uid", "level"]].rename(
            columns={"retained_source_row_uid": "source_row_uid"}
        )
        resolved = pd.concat(
            [prior.loc[prior.source_row_uid.isin(record_uids)], remapped],
            ignore_index=True,
        )
        if resolved.groupby("source_row_uid")["level"].nunique().gt(1).any():
            raise ValueError("Stage-1 duplicate lineage joins conflicting Gold levels")
        resolved = resolved.drop_duplicates("source_row_uid", keep="first")
        lineage = {
            "policy": "exact_duplicate_to_retained_uid_or_reviewed_source_drop",
            "missing_gold_mapping_uids": len(missing),
            "exact_duplicate_uids_remapped": int(duplicate.sum()),
            "unique_retained_uid_targets": int(remapped.source_row_uid.nunique()),
            "new_retained_uid_targets": int(
                remapped.loc[
                    ~remapped.source_row_uid.isin(set(prior.source_row_uid)),
                    "source_row_uid",
                ].nunique()
            ),
            "reviewed_source_row_drops": int(reviewed_drop.sum()),
        }
        prior = resolved
    mapped = records.merge(prior, on="source_row_uid", validate="one_to_one")
    mapped["level"] = mapped.level.astype("int64")
    mapped = mapped.sort_values("source_row_uid").reset_index(drop=True)
    inputs = {
        "stage3_records": {
            "path": str(published_records_path or records_path),
            "sha256": file_sha256(records_path),
        },
        "gold_level_mapping": {
            "path": str(published_prior_mapping_path or prior_path),
            "sha256": file_sha256(prior_path),
        },
    }
    if source_value_cleaning_audit is not None:
        inputs["stage1_source_value_cleaning_audit"] = {
            "path": str(published_cleaning_audit_path or source_value_cleaning_audit),
            "sha256": file_sha256(source_value_cleaning_audit),
        }
    _write(
        mapped,
        output_root,
        {
            "version": (
                "gold_owned_level_mapping.main_universe_subset_v2"
                if lineage is not None
                else "gold_owned_level_mapping.main_universe_subset_v1"
            ),
            "task": task,
            "mode": "mapped_subset_only",
            "inputs": inputs,
            "stage1_lineage_reconciliation": lineage,
            "unmapped_stage3_rows": len(records) - len(mapped),
            "validations": {
                "all_gold_mapping_uids_resolved": True,
                "all_output_uids_present_in_stage3": True,
                "duplicate_lineage_preserves_level": True,
            },
        },
        published_output_root=published_output_root,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--stage3-records", type=Path, required=True)
    parser.add_argument("--prior-mapping", type=Path, required=True)
    parser.add_argument("--voter-membership", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--release-version", default="v10_main_universe_v1")
    parser.add_argument("--published-stage3-path", type=Path)
    parser.add_argument("--published-output-root", type=Path)
    parser.add_argument("--published-prior-mapping-path", type=Path)
    parser.add_argument("--source-value-cleaning-audit", type=Path)
    parser.add_argument("--published-cleaning-audit-path", type=Path)
    args = parser.parse_args()
    if args.voter_membership:
        build_complete(
            args.task, args.stage3_records, args.prior_mapping,
            args.voter_membership, args.output_root, args.release_version,
            args.published_stage3_path, args.published_output_root,
            args.published_prior_mapping_path,
        )
    else:
        build_subset(
            args.task,
            args.stage3_records,
            args.prior_mapping,
            args.output_root,
            args.source_value_cleaning_audit,
            args.published_stage3_path,
            args.published_output_root,
            args.published_prior_mapping_path,
            args.published_cleaning_audit_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
