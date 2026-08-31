#!/usr/bin/env python3
"""Prepare and consolidate the independent SMILES conflict review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from rdkit import rdBase

from tools.chembl_tool.common.starling.normalization.cleaning import (
    resolve_structure_value,
)


REVIEW_VERSION = "name_smiles_conflict_review.v2"
EXPECTED_INPUT_SHA256 = (
    "52295101f5dcf1aac876d368e09b96830be945363d54c8f194c156fe3708213c"
)
DEFAULT_INPUT = Path(
    "outputs/chembl_tool/smiles_identity_audit_v2/name_smiles_comparison/v1/"
    "review/v1/reviewed_conflict_candidates.v1.parquet"
)
DEFAULT_OUTPUT = Path(
    "outputs/chembl_tool/smiles_identity_audit_v2/name_smiles_comparison/v1/"
    "review/v2"
)
EXACT_CONTEXT_FIELDS = (
    "task_id",
    "extracted_molecule_name",
    "canonical_smiles",
    "pubchem_smiles",
    "support_text",
)
CARD_FIELDS = (
    "candidate_id",
    "task_id",
    "source_id",
    "source_row_number",
    "source_record_id",
    "endpoint_name",
    "measurement_text",
    "support_text",
    "canonical_smiles",
    "source_smiles",
    "extracted_molecule_name",
    "extraction_evidence_span",
    "pubchem_cid",
    "pubchem_title",
    "pubchem_smiles",
    "stored_parent_inchi_key",
    "pubchem_parent_inchi_key",
)
REVIEW_DECISIONS = {"override", "retain_original", "unresolved"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source(path: Path) -> str:
    digest = file_sha256(path)
    if path.resolve() == DEFAULT_INPUT.resolve() and digest != EXPECTED_INPUT_SHA256:
        raise ValueError(
            f"frozen v1 review SHA-256 drift: expected {EXPECTED_INPUT_SHA256}, "
            f"found {digest}"
        )
    return digest


def _value(value: Any) -> Any:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def _hash(values: Iterable[Any]) -> str:
    payload = [_value(value) for value in values]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def exact_context_id(row: pd.Series | dict[str, Any]) -> str:
    return _hash(row[field] for field in EXACT_CONTEXT_FIELDS)


def _resolved(smiles: Any) -> tuple[str | None, str]:
    return resolve_structure_value(smiles, structure_mode="direct")


def select_review_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Select quarantined medium overrides and deterministic high-risk overrides."""
    work = frame.copy()
    work["exact_context_id"] = [
        exact_context_id(row) for _, row in work.iterrows()
    ]
    conflict_contexts = {
        context_id
        for context_id, group in work.groupby("exact_context_id")
        if {"override", "reject"} <= set(group["decision"])
    }
    reasons: dict[str, list[str]] = {}
    for _, row in work.loc[work["decision"].eq("override")].iterrows():
        selected: list[str] = []
        if row["confidence"] == "medium":
            selected.append("medium_confidence_override")
        _, status = _resolved(row["override_smiles"])
        if status != "resolved":
            selected.append("invalid_override_smiles")
        if (
            row["confidence"] == "high"
            and row["exact_context_id"] in conflict_contexts
        ):
            selected.append("exact_context_decision_conflict")
        if (
            row["confidence"] == "high"
            and str(row.get("extracted_molecule_name") or "").casefold() == "pfos"
        ):
            selected.append("known_pfos_abbreviation_collision")
        if selected:
            reasons[str(row["candidate_id"])] = selected
    targets = work.loc[work["candidate_id"].astype(str).isin(reasons)].copy()
    targets["selection_reasons"] = targets["candidate_id"].astype(str).map(reasons)
    return targets.sort_values(
        ["task_id", "exact_context_id", "candidate_id"]
    ).reset_index(drop=True)


def _card(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    return {field: _value(row.get(field)) for field in CARD_FIELDS}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _packet_groups(targets: pd.DataFrame, batch_size: int) -> list[list[str]]:
    groups = [
        list(group["candidate_id"].astype(str))
        for _, group in targets.groupby("exact_context_id", sort=True)
    ]
    if any(len(group) > batch_size for group in groups):
        raise ValueError("one exact-context group exceeds the review batch size")
    packets: list[list[str]] = []
    current: list[str] = []
    for group in groups:
        if current and len(current) + len(group) > batch_size:
            packets.append(current)
            current = []
        current.extend(group)
    if current:
        packets.append(current)
    return packets


def prepare(input_path: Path, output_dir: Path, batch_size: int) -> dict[str, Any]:
    source_sha256 = _verify_source(input_path)
    frame = pd.read_parquet(input_path)
    if frame["candidate_id"].astype(str).duplicated().any():
        raise ValueError("source review contains duplicate candidate IDs")
    targets = select_review_targets(frame)
    packet_dir = output_dir / "checker_packets"
    packet_paths: list[Path] = []
    for index, identifiers in enumerate(_packet_groups(targets, batch_size)):
        packet_id = f"checker_{index:03d}"
        cards = []
        for candidate_id in identifiers:
            row = targets.loc[targets["candidate_id"].astype(str).eq(candidate_id)].iloc[0]
            peers = frame.loc[
                frame["task_id"].eq(row["task_id"])
                & frame["support_text"].eq(row["support_text"])
                & ~frame["candidate_id"].astype(str).eq(candidate_id)
            ]
            cards.append(
                {
                    "target": _card(row),
                    "selection_reasons": list(row["selection_reasons"]),
                    "exact_context_id": str(row["exact_context_id"]),
                    "comparison_rows": [_card(peer) for _, peer in peers.iterrows()],
                }
            )
        path = packet_dir / f"{packet_id}.json"
        _write_json(
            path,
            {
                "review_version": REVIEW_VERSION,
                "packet_id": packet_id,
                "role": "checker",
                "instructions": {
                    "target_only": True,
                    "allowed_decisions": sorted(REVIEW_DECISIONS),
                    "override_rule": (
                        "Override only when the row unambiguously measures the named "
                        "molecule and the replacement is authoritative and RDKit-valid."
                    ),
                    "uncertainty_rule": (
                        "Use unresolved when row context is insufficient; never invent a "
                        "structure or decide comparison rows."
                    ),
                },
                "items": cards,
            },
        )
        packet_paths.append(path)
    manifest = {
        "review_version": REVIEW_VERSION,
        "mode": "checker_packet_preparation",
        "source": str(input_path),
        "source_sha256": source_sha256,
        "target_rows": len(targets),
        "target_counts": {
            f"{task}/{confidence}": int(count)
            for (task, confidence), count in targets.groupby(
                ["task_id", "confidence"]
            ).size().items()
        },
        "selection_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for reasons in targets["selection_reasons"]
                    for reason in reasons
                ).items()
            )
        ),
        "batch_size": batch_size,
        "packet_count": len(packet_paths),
        "packets": {
            path.name: file_sha256(path) for path in packet_paths
        },
        "writes_stage1": False,
    }
    _write_json(output_dir / "checker_packets_manifest.json", manifest)
    return manifest


def _read_reviews(directory: Path, role: str) -> dict[str, dict[str, Any]]:
    reviews: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                candidate_id = str(row.get("candidate_id") or "")
                if not candidate_id or candidate_id in reviews:
                    raise ValueError(
                        f"{path}:{line_number} has a missing or duplicate candidate ID"
                    )
                if row.get("review_role") != role:
                    raise ValueError(f"{path}:{line_number} has the wrong review role")
                if row.get("decision") not in REVIEW_DECISIONS:
                    raise ValueError(f"{path}:{line_number} has an invalid decision")
                if len(str(row.get("rationale") or "")) < 30:
                    raise ValueError(f"{path}:{line_number} needs a substantive rationale")
                if not str(row.get("reviewer") or ""):
                    raise ValueError(f"{path}:{line_number} has no reviewer")
                override = row.get("override_smiles")
                if row["decision"] == "override":
                    canonical, status = _resolved(override)
                    if status != "resolved":
                        raise ValueError(
                            f"{path}:{line_number} has invalid override SMILES"
                        )
                    row["override_smiles"] = canonical
                elif override is not None:
                    raise ValueError(
                        f"{path}:{line_number} retains a molecule but supplies an override"
                    )
                reviews[candidate_id] = row
    return reviews


def _load_context(input_path: Path, output_dir: Path):
    _verify_source(input_path)
    frame = pd.read_parquet(input_path)
    targets = select_review_targets(frame)
    checker = _read_reviews(output_dir / "checker_reviews", "checker")
    expected = set(targets["candidate_id"].astype(str))
    if set(checker) != expected:
        missing = sorted(expected - set(checker))
        extra = sorted(set(checker) - expected)
        raise ValueError(
            f"checker coverage mismatch: missing={missing[:3]} extra={extra[:3]}"
        )
    for _, row in targets.iterrows():
        review = checker[str(row["candidate_id"])]
        if str(review["reviewer"]) == str(row.get("reviewer") or ""):
            raise ValueError(f"checker is not independent for {row['candidate_id']}")
        if review["decision"] == "override":
            candidate, _ = _resolved(row["pubchem_smiles"])
            if review["override_smiles"] != candidate and not (
                review.get("reference_url") and review.get("reference_id")
            ):
                raise ValueError(
                    f"alternate override lacks an authoritative reference for "
                    f"{row['candidate_id']}"
                )
    return frame, targets, checker


def _adjudication_ids(
    targets: pd.DataFrame, checker: dict[str, dict[str, Any]]
) -> set[str]:
    identifiers: set[str] = set()
    for _, row in targets.iterrows():
        candidate_id = str(row["candidate_id"])
        review = checker[candidate_id]
        prior, status = _resolved(row["override_smiles"])
        checked = review.get("override_smiles")
        if (
            status != "resolved"
            or review["decision"] != "override"
            or checked != prior
            or "exact_context_decision_conflict" in row["selection_reasons"]
        ):
            identifiers.add(candidate_id)
    return identifiers


def prepare_adjudication(
    input_path: Path, output_dir: Path, batch_size: int
) -> dict[str, Any]:
    frame, targets, checker = _load_context(input_path, output_dir)
    identifiers = _adjudication_ids(targets, checker)
    selected = targets.loc[targets["candidate_id"].astype(str).isin(identifiers)]
    packet_paths: list[Path] = []
    for index, packet_ids in enumerate(_packet_groups(selected, batch_size)):
        packet_id = f"adjudicator_{index:03d}"
        items = []
        for candidate_id in packet_ids:
            row = selected.loc[
                selected["candidate_id"].astype(str).eq(candidate_id)
            ].iloc[0]
            items.append(
                {
                    "target": _card(row),
                    "selection_reasons": list(row["selection_reasons"]),
                    "primary_review": {
                        "decision": _value(row["decision"]),
                        "override_smiles": _value(row["override_smiles"]),
                        "confidence": _value(row["confidence"]),
                        "rationale": _value(row["rationale"]),
                        "reviewer": _value(row["reviewer"]),
                    },
                    "checker_review": checker[candidate_id],
                }
            )
        path = output_dir / "adjudication_packets" / f"{packet_id}.json"
        _write_json(
            path,
            {
                "review_version": REVIEW_VERSION,
                "packet_id": packet_id,
                "role": "adjudicator",
                "instructions": {
                    "allowed_decisions": sorted(REVIEW_DECISIONS),
                    "conservative_default": "unresolved",
                    "exact_context_rule": (
                        "Do not approve an override that would leave identical-context "
                        "rows with conflicting final decisions."
                    ),
                },
                "items": items,
            },
        )
        packet_paths.append(path)
    manifest = {
        "review_version": REVIEW_VERSION,
        "mode": "adjudication_packet_preparation",
        "source": str(input_path),
        "source_sha256": file_sha256(input_path),
        "checker_rows": len(checker),
        "adjudication_rows": len(identifiers),
        "batch_size": batch_size,
        "packet_count": len(packet_paths),
        "packets": {path.name: file_sha256(path) for path in packet_paths},
        "writes_stage1": False,
    }
    _write_json(output_dir / "adjudication_packets_manifest.json", manifest)
    return manifest


def _review_file_hashes(directory: Path) -> dict[str, str]:
    return {path.name: file_sha256(path) for path in sorted(directory.glob("*.jsonl"))}


def consolidate(input_path: Path, output_dir: Path) -> dict[str, Any]:
    frame, targets, checker = _load_context(input_path, output_dir)
    adjudication_ids = _adjudication_ids(targets, checker)
    adjudicator = _read_reviews(output_dir / "adjudicator_reviews", "adjudicator")
    if set(adjudicator) != adjudication_ids:
        missing = sorted(adjudication_ids - set(adjudicator))
        extra = sorted(set(adjudicator) - adjudication_ids)
        raise ValueError(
            f"adjudicator coverage mismatch: missing={missing[:3]} extra={extra[:3]}"
        )

    output = frame.copy()
    output["propagated_from_candidate_id"] = None
    for _, target in targets.iterrows():
        candidate_id = str(target["candidate_id"])
        checked = checker[candidate_id]
        final = adjudicator.get(candidate_id, checked)
        if candidate_id in adjudicator:
            if str(final["reviewer"]) in {
                str(target.get("reviewer") or ""),
                str(checked["reviewer"]),
            }:
                raise ValueError(f"adjudicator is not independent for {candidate_id}")
        if final["decision"] == "override":
            candidate, _ = _resolved(target["pubchem_smiles"])
            if final["override_smiles"] != candidate and not (
                final.get("reference_url") and final.get("reference_id")
            ):
                raise ValueError(
                    f"alternate final override lacks an authoritative reference for "
                    f"{candidate_id}"
                )
        mask = output["candidate_id"].astype(str).eq(candidate_id)
        output.loc[mask, "prior_decision"] = _value(target["decision"])
        output.loc[mask, "prior_confidence"] = _value(target["confidence"])
        output.loc[mask, "checker_decision"] = checked["decision"]
        output.loc[mask, "checker_reviewer"] = checked["reviewer"]
        output.loc[mask, "adjudicator_decision"] = (
            final["decision"] if candidate_id in adjudicator else None
        )
        output.loc[mask, "adjudicator_reviewer"] = (
            final["reviewer"] if candidate_id in adjudicator else None
        )
        output.loc[mask, "review_resolution"] = (
            "adjudicated" if candidate_id in adjudicator else "primary_checker_agree"
        )
        output.loc[mask, "reviewer"] = final["reviewer"]
        output.loc[mask, "rationale"] = final["rationale"]
        if final["decision"] == "override":
            canonical, status = _resolved(final["override_smiles"])
            before, before_status = _resolved(target["canonical_smiles"])
            if status != "resolved" or (before_status == "resolved" and canonical == before):
                raise ValueError(f"invalid or no-op final override for {candidate_id}")
            output.loc[mask, "decision"] = "override"
            output.loc[mask, "override_smiles"] = canonical
            output.loc[mask, "reject_reason"] = None
            output.loc[mask, "confidence"] = "high"
        else:
            output.loc[mask, "decision"] = "reject"
            output.loc[mask, "override_smiles"] = None
            output.loc[mask, "reject_reason"] = (
                "ambiguous_context"
                if final["decision"] == "unresolved"
                else "stored_smiles_consistent"
            )
            output.loc[mask, "confidence"] = (
                "low" if final["decision"] == "unresolved" else "high"
            )

    output["exact_context_id"] = [exact_context_id(row) for _, row in output.iterrows()]
    target_ids = set(targets["candidate_id"].astype(str))
    propagated_ids: set[str] = set()
    for _, group in output.groupby("exact_context_id"):
        if not ({"override", "reject"} <= set(group["decision"])):
            continue
        reviewed_rejects = group.loc[
            group["candidate_id"].astype(str).isin(target_ids)
            & group["decision"].eq("reject")
        ].sort_values("candidate_id")
        reviewed_overrides = group.loc[
            group["candidate_id"].astype(str).isin(target_ids)
            & group["decision"].eq("override")
        ]
        if reviewed_rejects.empty or not reviewed_overrides.empty:
            continue
        anchor = reviewed_rejects.iloc[0]
        for index, row in group.loc[group["decision"].eq("override")].iterrows():
            candidate_id = str(row["candidate_id"])
            output.at[index, "prior_decision"] = _value(row["decision"])
            output.at[index, "prior_confidence"] = _value(row["confidence"])
            output.at[index, "review_resolution"] = "exact_context_propagated"
            output.at[index, "propagated_from_candidate_id"] = str(
                anchor["candidate_id"]
            )
            output.at[index, "reviewer"] = _value(anchor["reviewer"])
            output.at[index, "rationale"] = (
                "Propagated the independently adjudicated non-override decision "
                f"from exact-context row {anchor['candidate_id']}."
            )
            output.at[index, "decision"] = "reject"
            output.at[index, "override_smiles"] = None
            output.at[index, "reject_reason"] = "exact_context_review"
            output.at[index, "confidence"] = _value(anchor["confidence"])
            propagated_ids.add(candidate_id)

    mixed = [
        context_id
        for context_id, group in output.groupby("exact_context_id")
        if {"override", "reject"} <= set(group["decision"])
    ]
    if mixed:
        raise ValueError(
            f"final review retains {len(mixed)} exact-context decision conflicts"
        )
    invalid: list[str] = []
    for _, row in output.loc[output["decision"].eq("override")].iterrows():
        _, status = _resolved(row["override_smiles"])
        if status != "resolved":
            invalid.append(str(row["candidate_id"]))
    if invalid:
        raise ValueError(f"final review has invalid override SMILES: {invalid[:3]}")

    proposal_dir = output_dir / "proposal"
    parquet_path = proposal_dir / "reviewed_conflict_candidates.v2.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = parquet_path.with_suffix(".parquet.tmp")
    output.drop(columns=["exact_context_id"]).to_parquet(temporary, index=False)
    os.replace(temporary, parquet_path)
    before = frame.set_index(frame["candidate_id"].astype(str))
    after = output.set_index(output["candidate_id"].astype(str))
    changed_ids = {
        candidate_id
        for candidate_id in frame["candidate_id"].astype(str)
        if (
            after.at[candidate_id, "decision"] != before.at[candidate_id, "decision"]
            or after.at[candidate_id, "confidence"]
            != before.at[candidate_id, "confidence"]
            or str(after.at[candidate_id, "override_smiles"] or "")
            != str(before.at[candidate_id, "override_smiles"] or "")
        )
    }
    selected_output = output.loc[
        output["candidate_id"].astype(str).isin(target_ids)
    ]
    affected_output = output.loc[output["review_resolution"].notna()]
    impact = {
        "review_version": REVIEW_VERSION,
        "source_rows": len(frame),
        "review_target_rows": len(targets),
        "checker_rows": len(checker),
        "adjudicated_rows": len(adjudicator),
        "changed_rows": len(changed_ids),
        "changed_target_rows": len(changed_ids & target_ids),
        "exact_context_propagated_rows": len(propagated_ids),
        "checker_decision_counts": dict(
            sorted(Counter(row["decision"] for row in checker.values()).items())
        ),
        "adjudicator_decision_counts": dict(
            sorted(Counter(row["decision"] for row in adjudicator.values()).items())
        ),
        "selected_final_counts": {
            f"{task}/{confidence}/{decision}": int(count)
            for (task, confidence, decision), count in selected_output.groupby(
                ["task_id", "prior_confidence", "decision"]
            ).size().items()
        },
        "affected_source_outcomes": {
            f"{task}/{source}/{decision}": int(count)
            for (task, source, decision), count in affected_output.groupby(
                ["task_id", "source_id", "decision"]
            ).size().items()
        },
        "net_override_delta": int(
            output["decision"].eq("override").sum()
            - frame["decision"].eq("override").sum()
        ),
        "final_decision_counts": {
            str(key): int(value)
            for key, value in output["decision"].value_counts().sort_index().items()
        },
        "final_override_counts": {
            f"{task}/{source}": int(count)
            for (task, source), count in output.loc[
                output["decision"].eq("override")
            ].groupby(["task_id", "source_id"]).size().items()
        },
        "invalid_applied_overrides": 0,
        "exact_context_conflicts": 0,
        "publishes_task_policies": False,
        "rebuilds_stage1": False,
    }
    _write_json(proposal_dir / "impact_report.json", impact)
    manifest = {
        **impact,
        "source": str(input_path),
        "source_sha256": file_sha256(input_path),
        "proposal": str(parquet_path),
        "proposal_sha256": file_sha256(parquet_path),
        "checker_review_sha256": _review_file_hashes(
            output_dir / "checker_reviews"
        ),
        "adjudicator_review_sha256": _review_file_hashes(
            output_dir / "adjudicator_reviews"
        ),
        "runtime": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "pandas": pd.__version__,
        },
        "publication_status": "proposal_only",
    }
    _write_json(proposal_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "prepare-adjudication", "consolidate")
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.input, args.output_dir, args.batch_size)
    elif args.command == "prepare-adjudication":
        result = prepare_adjudication(args.input, args.output_dir, args.batch_size)
    else:
        result = consolidate(args.input, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
