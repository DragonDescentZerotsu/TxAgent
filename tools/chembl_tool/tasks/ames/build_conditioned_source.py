"""Restore and canonicalize frozen Ames sources with a complete record ledger."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, rdBase

from tools.chembl_tool.common.json_utils import (
    atomic_output_path,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.common.build_runtime import (
    worker_pool,
    local_input,
    local_workdir,
    publish_file,
    sha256_file,
)
from tools.chembl_tool.common.molecule_identity import (
    bemis_murcko_scaffold,
    normalize_molecule_identity,
)
from tools.chembl_tool.tasks.ames.source_contract import (
    BACTERIAL_CONTEXT,  # noqa: F401 - retained public diagnostic regex
    DIRECT,
    NEAR,
    RETRIEVAL_VERSION,
    VERSION,
    classify,
    has_direct_context,
    text,
)
from tools.chembl_tool.tasks.ames.identity_review import (
    RESPONSES,
    identity_status,
    load_name_resolutions,
)
from tools.chembl_tool.tasks.ames.reviewed_source import (
    DECISIONS,
    REVIEW_METHOD,
    apply_review,
    load_reviews,
    PLACEMENTS,
    load_placements,
    apply_placement,
)

ROOT = Path("data/starling_data/ames")
RAW_ROOT = ROOT / "raw_v1"
CANONICAL_ROOT = ROOT / "canonical_v1"
COMMIT = "03e4c7c694b45bcfdf7776ac1045bc3e69936f0a"
SOURCES = ("ames_base", "ames_v1", "ames_v2", "ames_v3")


def restore() -> None:
    manifest = json.loads((RAW_ROOT / "manifest.json").read_text())
    if manifest["source_commit"] != COMMIT:
        raise ValueError("Ames acquisition commit changed")
    for name, metadata in manifest["files"].items():
        target = RAW_ROOT / name
        if not target.exists() or sha256_file(target) != metadata["sha256"]:
            with atomic_output_path(target) as temporary:
                with temporary.open("wb") as handle:
                    subprocess.run(
                        ["git", "show", f"{COMMIT}:{metadata['original_path']}"],
                        stdout=handle,
                        check=True,
                    )
                if sha256_file(temporary) != metadata["sha256"]:
                    raise ValueError(f"Frozen source hash mismatch: {target}")
        if sha256_file(target) != metadata["sha256"]:
            raise ValueError(f"Frozen source hash mismatch: {target}")


def _identity(smiles: str) -> tuple[str, dict]:
    identity = normalize_molecule_identity(smiles)
    data = identity.to_dict()
    data["bemis_murcko_scaffold"] = (
        bemis_murcko_scaffold(identity.parent_smiles) if identity.parent_smiles else ""
    )
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
        data["multiple_organic_components"] = bool(
            mol
            and sum(
                any(atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms())
                for fragment in Chem.GetMolFrags(mol, asMols=True)
            )
            > 1
        )
        data["unsupported_material_identity"] = bool(
            mol
            and (
                not any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
                or any(
                    atom.GetAtomicNum()
                    in set(range(21, 31)) | set(range(39, 49)) | set(range(57, 81))
                    for atom in mol.GetAtoms()
                )
            )
        )
        # Non-carbon does not mean an unusable molecular structure: e.g. H2O2
        # is a defined DNA-damage reagent. Keep simple molecular inorganics for
        # retrieval, while preserving the original strict gold exclusion.
        data["simple_inorganic_molecule"] = bool(
            mol
            and len(Chem.GetMolFrags(mol)) == 1
            and mol.GetNumHeavyAtoms() >= 2
            and all(
                atom.GetAtomicNum() in {1, 7, 8, 9, 15, 16, 17, 35, 53}
                for atom in mol.GetAtoms()
            )
        )
    return smiles, data


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _first(row: dict, *fields: str) -> str:
    return next((text(row.get(field)) for field in fields if text(row.get(field))), "")


def _canonical(source: str, ordinal: int, raw: dict, identity: dict, decision) -> dict:
    endpoint = _first(raw, "mechanism_category", "endpoint_class", "assay_family")
    result = _first(
        raw, "mutagenicity_result", "result_call", "result_status", "result_direction"
    )
    system = _first(raw, "test_system", "biological_test_system", "biological_system")
    activation = _first(
        raw,
        "metabolic_activation",
        "metabolic_activation_status",
        "metabolic_activation_presence",
        "metabolic_activation_system",
    )
    conditions = {
        key: text(raw.get(key))
        for key in (
            "qualifying_conditions",
            "extra_details",
            "dose_or_concentration",
            "dose_unit",
            "exposure_and_mechanistic_conditions",
            "molecule_role",
            "cytotoxicity_status",
            "evidence_basis",
            "experimental_context",
            "study_context",
            "needs_more_context",
        )
        if text(raw.get(key))
    }
    # Extra endpoint/target fields are source facts. Keep the direct-vote card
    # surface unchanged so a retrieval revision cannot alter vote fingerprints.
    if source != "ames_base":
        for key in ("endpoint_subtype", "genetic_locus_or_chromosome_target"):
            if text(raw.get(key)):
                conditions[key] = text(raw[key])
    value = _first(raw, "response_value", "quantitative_readout_value")
    units = _first(raw, "response_unit", "quantitative_readout_unit")
    assay_context = " | ".join(
        filter(
            None,
            (
                text(raw.get("assay_family")),
                system,
                f"metabolic_activation={activation}" if activation else "",
                _first(raw, "assay_version", "assay_method_and_endpoint"),
            ),
        )
    )
    rid = f"{source}:{ordinal}"
    return {
        "source_id": source,
        "source_name": f"starling/ames/{source}",
        "source_row_number": ordinal,
        "source_record_id": rid,
        "canonical_record_id": _hash([COMMIT, source, ordinal]),
        "source_smiles": text(raw.get("SMILES")),
        "canonical_smiles": identity["canonical_smiles"],
        "molecule_identity_key": identity["parent_inchi_key"],
        "molecule_id": "STARLING_" + _hash(identity["canonical_smiles"])[:16].upper(),
        "molecule_name": text(raw.get("molecule_name")),
        "canonical_endpoint_name": endpoint,
        "canonical_measurement_text": "; ".join(filter(None, (result, value))),
        "canonical_unit_text": units,
        "canonical_assay_context": assay_context,
        "canonical_species_context": system,
        "qualifying_conditions": " | ".join(f"{k}={v}" for k, v in conditions.items()),
        "support_text": text(raw.get("support_text")),
        "confidence": float(raw.get("confidence") or 0),
        "pmid": text(raw.get("pmid")),
        "extraction_id": text(raw.get("extraction_id")),
        "paragraph_idx": str(raw.get("paragraph_idx")),
        "group_id": decision.group,
        "retrieval_eligible": True,
        # Retained indirect cards may quote a bacterial outcome in their full
        # source passage. Conservatively filter those heldout records too.
        "heldout_filter_scope": (
            "bacterial_outcome"
            if decision.group in {DIRECT, NEAR}
            or (
                decision.reason != "payload_pinned_record_placement_review"
                and has_direct_context(
                    assay_context
                    + " | "
                    + text(raw.get("support_text"))
                    + " | "
                    + " ".join(conditions.values())
                )
            )
            else "mechanism"
        ),
        "source_family_purity_reason": decision.reason,
        "record_contract_version": VERSION,
        "retrieval_contract_version": RETRIEVAL_VERSION,
    }


def _prepare_record(source, ordinal, raw, identities, resolutions, reviews, placements):
    pending_identity = []
    decision = classify(source, raw)
    identity = identities[text(raw.get("SMILES"))]
    rid = f"{source}:{ordinal}"
    review = reviews.get(rid)
    if review:
        decision = apply_review(review, raw, identity["parent_inchi_key"], original_decision=decision)
    structure_corrected = bool(review and review["action"] == "correct_structure")
    if structure_corrected:
        identity = _identity(review["corrected_smiles"])[1]
        if identity["status"] != "ok" or identity["parent_inchi_key"] != review["corrected_parent_inchi_key"]:
            raise ValueError(f"Invalid reviewed structure correction: {rid}")
    if rid in placements:
        if review:
            raise ValueError(f"Gold and placement reviews overlap: {rid}")
        decision = apply_placement(placements[rid], raw, decision)
    name = (
        review["molecule_name"]
        if review and review["action"] == "accept"
        else text(raw.get("molecule_name"))
    )
    reason, group = decision.reason, decision.group
    status = (
        identity_status(
            name,
            identity["parent_inchi_key"],
            resolutions,
        )
        if source == "ames_base" or (review and review["action"] == "accept")
        else "source_structure_without_name_field"
    )
    if structure_corrected:
        status = "payload_pinned_structure_correction"
    if decision.label is not None and status == "identity_request_unresolved":
        attempts = resolutions[name]["n_request_attempts"]
        if attempts < 4:
            raise ValueError(f"Incomplete identity acquisition for candidate: {name}")
        pending_identity.append(
            {
                "source_record_id": f"{source}:{ordinal}",
                "pmid": text(raw.get("pmid")),
                "molecule_name": name,
                "source_smiles": text(raw.get("SMILES")),
                "parent_inchi_key": identity["parent_inchi_key"],
                "n_request_attempts": attempts,
                "status": "identity_request_unresolved",
                "action": "excluded_from_votes_and_retrieval_pending_identity_verification",
            }
        )
    if source == "ames_base" and status == "name_structure_mismatch":
        reason, group = "pubchem_name_structure_mismatch", ""
    elif source == "ames_base" and status not in {
        "not_queried",
        "verified_parent_match",
        "payload_pinned_structure_correction",
    }:
        reason, group = "identity_" + status, ""
    elif decision.label is not None and status != "verified_parent_match":
        if status == "not_queried":
            raise ValueError(
                f"Candidate name has no frozen PubChem response: {raw.get('molecule_name')}"
            )
        reason, group = "identity_" + status, ""
    if identity["status"] != "ok" or not identity["parent_inchi_key"]:
        reason, group = "invalid_structure_or_parent", ""
    elif identity["multiple_organic_components"]:
        reason, group = (
            "multiple_organic_components_require_identity_review",
            "",
        )
    elif identity["unsupported_material_identity"]:
        if not identity["simple_inorganic_molecule"]:
            reason, group = (
                "inorganic_or_metal_material_requires_identity_review",
                "",
            )
        elif group == DIRECT:
            reason, group = "simple_inorganic_molecule_is_retrieval_only", NEAR
    duplicate_of = ""
    record = None
    signature = ""
    if group:
        record = _canonical(source, ordinal, raw, identity, decision)
        if review and review["action"] == "accept":
            # Correct only explicitly reviewed factual card fields; the
            # original extraction remains immutable in raw_v1 + audit.
            correction = review.get("evidence_correction", {})
            allowed = {
                "canonical_endpoint_name",
                "canonical_measurement_text",
                "canonical_assay_context",
                "canonical_species_context",
                "support_text",
                "qualifying_conditions",
            }
            if correction.keys() - allowed:
                raise ValueError(f"Unsupported reviewed evidence fields: {rid}")
            record.update(correction)
        if rid in placements:
            surface = [
                text(record[k])
                for k in (
                    "canonical_endpoint_name",
                    "canonical_measurement_text",
                    "canonical_unit_text",
                    "canonical_assay_context",
                    "canonical_species_context",
                    "qualifying_conditions",
                    "support_text",
                )
            ]
            if _hash(surface) != placements[rid]["canonical_surface_sha256"]:
                raise ValueError(f"Reviewed card surface changed: {rid}")
        # Enforce containment on the assembled card as well as the raw
        # record; field ordering must never expose a direct passage late.
        if (
            group not in {DIRECT, NEAR}
            and record["heldout_filter_scope"] == "bacterial_outcome"
        ):
            group, reason = NEAR, "direct_context_in_assembled_card"
        record.update(
            group_id=group,
            source_family_purity_reason=reason,
            identity_verification=status,
        )
        signature = _hash(
            [
                record[k]
                for k in (
                    "molecule_identity_key",
                    "pmid",
                    "canonical_endpoint_name",
                    "canonical_measurement_text",
                    "canonical_unit_text",
                    "canonical_assay_context",
                    "qualifying_conditions",
                    "support_text",
                )
            ]
        )
    candidate = None
    if record is not None:
        record["record_fingerprint_sha256"] = signature
        record["canonical_claim_id"] = ""
        if group == DIRECT:
            candidate = {
                "source_record_id": rid,
                "source_record_fingerprint_sha256": signature,
                "source_payload_sha256": _hash(raw),
                "drug": identity["parent_smiles"],
                "molecule_identity_key": identity["parent_inchi_key"],
                "molecule_identity": {
                    k: v
                    for k, v in identity.items()
                    if k
                    not in {
                        "bemis_murcko_scaffold",
                        "multiple_organic_components",
                        "unsupported_material_identity",
                        "simple_inorganic_molecule",
                    }
                },
                "bemis_murcko_scaffold": identity["bemis_murcko_scaffold"],
                "Y": decision.label,
                "condition_group": "+".join(decision.condition_atoms),
                "condition_atoms": list(decision.condition_atoms),
                "condition_scope": "external",
                "pmid": text(raw.get("pmid")),
                "molecule_name": name,
                "source_smiles": text(raw.get("SMILES")),
                "label_method": decision.reason,
                "reviewer": (
                    "Codex agent:source_review_v4"
                    if review
                    else "deterministic_source_policy:" + VERSION
                ),
                "review_method": (
                    "individual_source_semantic_review"
                    if review
                    else "structured_source_gate_not_manual_review"
                ),
                "raw_value": text(raw.get("mutagenicity_result")),
                "condition_text": record["canonical_assay_context"],
            }
    audit_row = {
        "source_id": source,
        "source_row_number": ordinal,
        "source_record_id": rid,
        "canonical_record_id": _hash([COMMIT, source, ordinal]),
        "pmid": text(raw.get("pmid")),
        "extraction_id": text(raw.get("extraction_id")),
        "parent_inchi_key": identity["parent_inchi_key"],
        "group_id": group,
        "reason": reason,
        "identity_verification": status,
        "is_voter": bool(group == DIRECT),
        "duplicate_of": duplicate_of,
    }
    return (
        record,
        candidate,
        audit_row,
        pending_identity,
        bool(review),
        rid in placements,
    )


_source_batch_context = None


def _source_batch(bounds):
    source, table, identities, resolutions, reviews, placements = _source_batch_context
    start, stop = bounds
    return [
        _prepare_record(
            source, start + offset, raw, identities, resolutions, reviews, placements
        )
        for offset, raw in enumerate(table.slice(start, stop - start).to_pylist())
    ]


def build(*, workers: int = 8) -> dict:
    restore()
    policy_hash = sha256_file(Path(__file__).with_name("source_contract.py"))
    resolutions = load_name_resolutions()
    reviews = load_reviews()
    placements = load_placements()
    seen_placements = set()
    seen_reviews = set()
    CANONICAL_ROOT.mkdir(parents=True, exist_ok=True)
    smiles = set()
    for source in SOURCES:
        smiles.update(
            pq.read_table(
                local_input(RAW_ROOT / f"{source}.parquet"), columns=["SMILES"]
            )["SMILES"].to_pylist()
        )
    smiles = sorted(text(value) for value in smiles)
    with worker_pool(workers) as pool:
        identities = dict(pool.map(_identity, smiles, chunksize=128))
    print(f"Normalized {len(identities)} distinct source structures", flush=True)
    candidate_votes, audit, records, pending_identity = [], [], [], []
    seen = {}
    for source in SOURCES:
        frame = pq.read_table(local_input(RAW_ROOT / f"{source}.parquet"))
        global _source_batch_context
        _source_batch_context = (
            source,
            frame,
            identities,
            resolutions,
            reviews,
            placements,
        )
        bounds = [
            (start, min(start + 2048, len(frame)))
            for start in range(0, len(frame), 2048)
        ]
        try:
            with worker_pool(workers) as pool:
                for batch in pool.map(_source_batch, bounds):
                    for (
                        record,
                        candidate,
                        audit_row,
                        pending,
                        reviewed,
                        placed,
                    ) in batch:
                        rid = audit_row["source_record_id"]
                        if reviewed:
                            seen_reviews.add(rid)
                        if placed:
                            seen_placements.add(rid)
                        pending_identity.extend(pending)
                        if record is not None:
                            signature = record["record_fingerprint_sha256"]
                            if signature in seen:
                                audit_row.update(
                                    group_id="",
                                    reason="exact_claim_duplicate",
                                    is_voter=False,
                                    duplicate_of=seen[signature],
                                )
                            else:
                                seen[signature] = rid
                                records.append(record)
                                if candidate is not None:
                                    candidate_votes.append(candidate)
                        audit.append(audit_row)
        finally:
            _source_batch_context = None
        print(
            f"{source}: {len(frame)} rows; cumulative {len(records)} retrievable, {len(candidate_votes)} candidate voters",
            flush=True,
        )
    if seen_placements != placements.keys():
        raise ValueError(
            f"Missing placement records: {placements.keys() - seen_placements}"
        )
    if seen_reviews != reviews.keys():
        raise ValueError(
            f"Reviews reference missing source rows: {reviews.keys() - seen_reviews}"
        )
    votes, study_audit = collapse_study_votes(candidate_votes)
    accepted_ids = {row["source_record_id"] for row in votes}
    study_reasons = {row["source_record_id"]: row["reason"] for row in study_audit}
    study_claims = {
        row["source_record_id"]: row["study_claim_id"] for row in study_audit
    }
    for row in records:
        row["canonical_claim_id"] = study_claims.get(row["source_record_id"], "")
        if row["group_id"] == DIRECT and row["source_record_id"] not in accepted_ids:
            row["group_id"] = NEAR
            row["source_family_purity_reason"] = study_reasons[row["source_record_id"]]
    for row in audit:
        row["canonical_claim_id"] = study_claims.get(row["source_record_id"], "")
        if row["group_id"] == DIRECT and row["source_record_id"] not in accepted_ids:
            row.update(
                group_id=NEAR,
                reason=study_reasons[row["source_record_id"]],
                is_voter=False,
            )
    if sha256_file(Path(__file__).with_name("source_contract.py")) != policy_hash:
        raise RuntimeError(
            "Source policy changed during build; rerun before publishing"
        )
    assert all(
        r["group_id"] in {DIRECT, NEAR} or r["heldout_filter_scope"] == "mechanism"
        for r in records
    ), "Direct-related content must be assigned to L1/L2, not only heldout-filtered"

    def write_parquet(item):
        name, rows = item
        with local_workdir() as staging:
            target = staging / name
            pd.DataFrame(rows).to_parquet(target, index=False, compression="zstd")
            publish_file(target, CANONICAL_ROOT / name)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(
            pool.map(
                write_parquet,
                (("records.parquet", records), ("record_audit.parquet", audit)),
            )
        )
    write_jsonl_atomic(CANONICAL_ROOT / "source_votes.jsonl", votes)
    write_jsonl_atomic(CANONICAL_ROOT / "study_vote_audit.jsonl", study_audit)
    write_jsonl_atomic(CANONICAL_ROOT / "pending_identity.jsonl", pending_identity)
    assert len(audit) == sum(
        pq.ParquetFile(RAW_ROOT / f"{s}.parquet").metadata.num_rows for s in SOURCES
    )
    assert {v["source_record_id"] for v in votes} == {
        r["source_record_id"] for r in records if r["group_id"] == DIRECT
    }
    manifest = {
        "contract": VERSION,
        "retrieval_contract": RETRIEVAL_VERSION,
        "source_commit": COMMIT,
        "source_manifest_sha256": sha256_file(RAW_ROOT / "manifest.json"),
        "pubchem_name_responses_sha256": sha256_file(RESPONSES),
        "policy_source_sha256": policy_hash,
        "n_source_rows": len(audit),
        "n_retrieval_records": len(records),
        "n_voters": len(votes),
        "n_pending_identity_candidate_rows": len(pending_identity),
        "n_pending_identity_candidate_names": len(
            {r["molecule_name"] for r in pending_identity}
        ),
        "benchmark_scope": "identity_verified_source_subset_with_explicit_pending_identity_exclusions",
        "family_counts": dict(Counter(r["group_id"] for r in records)),
        "decision_counts": dict(
            sorted(
                Counter(
                    f"{r['source_id']}:{r['group_id'] or 'excluded'}:{r['reason']}"
                    for r in audit
                ).items()
            )
        ),
        "builder_sha256": sha256_file(Path(__file__)),
        "identity_policy_sha256": sha256_file(
            Path(__file__).with_name("identity_review.py")
        ),
        "vote_unit": "one_unambiguous_pmid_parent_exact_condition",
        "review_method": REVIEW_METHOD,
        "semantic_reviews_sha256": sha256_file(DECISIONS),
        "semantic_review_policy_sha256": sha256_file(
            Path(__file__).with_name("reviewed_source.py")
        ),
        "n_applied_semantic_reviews": len(reviews),
        "placement_reviews_sha256": sha256_file(PLACEMENTS),
        "n_applied_placement_reviews": len(placements),
        "condition_contract": "exact_reported_standard_strain_panel_and_metabolic_activation",
        "identity_policy": "gold requires one exact normalized PubChem name-parent match; no automatic structure repair",
        "unresolved_condition_policy": "nonvoter; never silently mapped to no_reported_external_condition",
        "files": {
            name: {
                "path": str(CANONICAL_ROOT / name),
                "sha256": sha256_file(CANONICAL_ROOT / name),
            }
            for name in (
                "records.parquet",
                "record_audit.parquet",
                "source_votes.jsonl",
                "study_vote_audit.jsonl",
                "pending_identity.jsonl",
            )
        },
    }
    write_json_atomic(CANONICAL_ROOT / "manifest.json", manifest)
    return manifest


def collapse_study_votes(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """Do not let paraphrases or opposite extractions multiply study support."""
    grouped = defaultdict(list)
    for row in candidates:
        grouped[
            row["pmid"], row["molecule_identity_key"], row["condition_group"]
        ].append(row)
    votes, audit = [], []
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["source_record_id"])
        conflict = len({row["Y"] for row in rows}) != 1
        missing_pmid = not str(key[0]).isdigit()
        retained = "" if conflict or missing_pmid else rows[0]["source_record_id"]
        study_claim_id = _hash(key)
        if retained:
            votes.append(
                {
                    **rows[0],
                    "canonical_claim_id": study_claim_id,
                    "supporting_source_record_ids": [
                        r["source_record_id"] for r in rows
                    ],
                    "vote_unit": "one_unambiguous_pmid_parent_exact_condition",
                }
            )
        for row in rows:
            reason = (
                "missing_valid_pmid"
                if missing_pmid
                else (
                    "conflicting_study_parent_condition"
                    if conflict
                    else (
                        "primary_study_condition_vote"
                        if row["source_record_id"] == retained
                        else "same_study_parent_condition_repeated_description"
                    )
                )
            )
            audit.append(
                {
                    "source_record_id": row["source_record_id"],
                    "study_claim_id": study_claim_id,
                    "retained_source_record_id": retained,
                    "reason": reason,
                }
            )
    return votes, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(build(workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
