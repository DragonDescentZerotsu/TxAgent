"""Apply explicit, payload-pinned source reviews before study-vote collapse.

Reviews are authored evidence decisions, not model predictions or regex grants.
The builder still enforces chemical identity, deduplication and actual-voter
membership. A reviewed direct claim can originate in any of the four runs.
"""

import hashlib
import json
from pathlib import Path

from tools.chembl_tool.tasks.ames.source_contract import (
    Decision,
    DIRECT,
    NEAR,
    MUTATION,
    DAMAGE,
    MECHANISM,
)

REVIEW_ROOT = Path("data/starling_data/ames/source_review_v4")
DECISIONS = REVIEW_ROOT / "applied_decisions.jsonl"
REVIEW_METHOD = "deterministic_source_policy_with_payload_pinned_agent_semantic_reviews"
PLACEMENTS = Path("data/starling_data/ames/source_review_v6/placement_decisions.jsonl")


def payload_hash(raw: dict) -> str:
    return hashlib.sha256(
        json.dumps(raw, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def load_placements(path: Path = PLACEMENTS) -> dict[str, dict]:
    """Only individually read, immutable records can bypass a broad scope hit."""
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        rid = row["source_record_id"]
        if (
            rid in result
            or row["group_id"] not in {NEAR, MUTATION, DAMAGE, MECHANISM}
            or not row["rationale"]
            or len(row["raw_sha256"]) != 64
            or len(row["canonical_surface_sha256"]) != 64
        ):
            raise ValueError(f"Invalid placement review: {rid}")
        result[rid] = row
    return result


def apply_placement(review: dict, raw: dict, decision: Decision) -> Decision:
    if payload_hash(raw) != review["raw_sha256"]:
        raise ValueError(f"Placement payload changed: {review['source_record_id']}")
    if decision.label is not None or not decision.group:
        raise ValueError(
            f"Placement cannot grant eligibility or change gold: {review['source_record_id']}"
        )
    return Decision(review["group_id"], "payload_pinned_record_placement_review")


def load_reviews(path: Path = DECISIONS) -> dict[str, dict]:
    reviews = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        rid = row["source_record_id"]
        if rid in reviews or row["action"] not in {"accept", "withhold", "exclude"}:
            raise ValueError(f"Invalid or duplicate source review: {rid}")
        if not row["rationale"] or not row["raw_sha256"] or not row["pmid"]:
            raise ValueError(f"Incomplete source review: {rid}")
        if row["action"] == "accept":
            atoms = row["condition_atoms"]
            if (
                row["Y"] not in (0, 1)
                or not row["molecule_name"]
                or not row["parent_inchi_key"]
                or len(atoms) != len(set(atoms))
                or not any(a.startswith("strain_panel=") for a in atoms)
                or not any(a.startswith("metabolic_activation=") for a in atoms)
            ):
                raise ValueError(f"Invalid reviewed outcome: {rid}")
        reviews[rid] = row
    return reviews


def apply_review(review: dict, raw: dict, parent: str) -> Decision:
    rid = review["source_record_id"]
    if payload_hash(raw) != review["raw_sha256"] or str(raw["pmid"]) != review["pmid"]:
        raise ValueError(f"Reviewed source payload changed: {rid}")
    if parent != review["parent_inchi_key"]:
        raise ValueError(f"Reviewed source parent changed: {rid}")
    action = review["action"]
    if action == "exclude":
        return Decision("", "reviewed_source_identity_mismatch")
    if action == "withhold":
        return Decision(NEAR, "reviewed_direct_claim_pending_resolution")
    return Decision(
        DIRECT,
        "reviewed_primary_bacterial_outcome",
        review["Y"],
        tuple(sorted(review["condition_atoms"])),
    )
